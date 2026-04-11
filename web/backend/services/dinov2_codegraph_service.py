"""DINOv2 + CoDeGraph3D anomaly detection service.

Training-free anomaly detection for 3D CT volumes using:
  1. DINOv2 ViT-L/14 (HuggingFace) multi-layer feature extraction (axial only)
  2. Multi-window RGB encoding: Brain [0,80], Subdural [-20,180], Blood [20,60] HU
  3. Per-slice L2 normalization + random projection (1024 → 64)
  4. Temporal-exclusion K-NN scoring (K=1): for each token on slice i,
     K=1 nearest neighbor from slices ≥14 away — no depth pooling
  5. sqrt(FAISS L2²) correction for proper L2 distances (matching torch.cdist)
  6. Multi-layer score averaging → μ+2σ threshold → per-slice ROI extraction

Adapted from CoDeGraph3D (arxiv 2602.15315) for single-volume use.
2D per-slice approach preserves full axial resolution — no 14:1 depth pooling
that dilutes focal lesion signals (e.g., hemorrhage spanning 5-10 slices).
"""

import io
import logging
from typing import Any

import numpy as np
import PIL.Image

logger = logging.getLogger(__name__)

# ── DINOv2 ViT-L/14 parameters ───────────────────────────────────────────
PATCH_SIZE = 14
EMBED_DIM = 1024                      # DINOv2-L hidden_size
PROJ_DIM = 64                         # random projection target dim (match original)
LAYER_INDICES = [6, 12, 18, 24]       # 4 layers for ViT-L (24 total)
TARGET_SIZE = 224                      # resample volume to 224^3
GRID_DIM = TARGET_SIZE // PATCH_SIZE  # = 16 tokens per spatial axis

# ── Scoring parameters ───────────────────────────────────────────────────
TEMPORAL_EXCLUDE = 14                 # min slice distance for temporal exclusion
N_PATCHES = GRID_DIM * GRID_DIM      # 256 tokens per slice (16×16)
ANOMALY_THRESHOLD_SIGMA = 2.0        # μ+Nσ threshold for normalization
MIN_COMPONENT_AREA = 50
TOP_K_SLICES = 5

# ── CT multi-window RGB encoding ─────────────────────────────────────────
# Three clinically distinct windows → R, G, B channels for DINOv2.
# Gives per-channel information instead of grayscale repeated 3x.
WIN_BRAIN = (0, 80)       # R: brain parenchyma (L=40, W=80)
WIN_SUBDURAL = (-20, 180) # G: subdural/wide (L=80, W=200) — captures extra-axial
WIN_BLOOD = (20, 60)      # B: narrow blood (L=40, W=40) — maximizes blood vs brain
BRAIN_HU_LOW = 0          # brain parenchyma mask lower bound
BRAIN_HU_HIGH = 100       # mask upper bound (excludes skull/bone >100 HU)
MASK_COVERAGE_THRESHOLD = 0.5  # avg_pool threshold — require >50% tissue per token


class DINOv2CoDeGraphService:
    """Lazy-loaded DINOv2 + self-referencing anomaly detector for CT volumes."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.encoder_blocks = None
        self._device = None
        self._proj_matrices: dict[int, np.ndarray] = {}  # layer_idx → (1024, 64)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> None:
        """Load DINOv2 ViT-L/14 via HuggingFace transformers. Called lazily."""
        if self.model is not None:
            return

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            logger.error("PyTorch or transformers not available — DINOv2 disabled")
            return

        if not torch.cuda.is_available():
            logger.warning("DINOv2 requires CUDA — anomaly detection disabled")
            return

        try:
            logger.info("Loading DINOv2 ViT-L/14 from HuggingFace...")
            self.processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-large", use_fast=True
            )
            model = AutoModel.from_pretrained("facebook/dinov2-large")
            model = model.half().eval().cuda()
            self.model = model
            self.encoder_blocks = model.encoder.layer
            self._device = "cuda"
            logger.info("DINOv2-L loaded on CUDA (fp16, %d layers, %d-dim)",
                        len(self.encoder_blocks), EMBED_DIM)
        except Exception as exc:
            logger.error("Failed to load DINOv2: %s", exc)
            self.model = None
            return

        # Initialize random projection matrices — one per layer
        for layer_idx in LAYER_INDICES:
            rng = np.random.RandomState(42 + layer_idx)
            mat = rng.randn(EMBED_DIM, PROJ_DIM).astype(np.float32)
            # Column-normalize then JL scaling
            mat /= np.linalg.norm(mat, axis=0, keepdims=True)
            mat *= 1.0 / np.sqrt(PROJ_DIM)
            self._proj_matrices[layer_idx] = mat

    # ── Main pipeline ─────────────────────────────────────────────────────

    def detect_anomaly(self, session_data: Any) -> dict:
        """Run full anomaly detection pipeline on a session's CT volume.

        2D per-slice approach: axial-only encoding, no depth pooling,
        temporal-exclusion K-NN scoring with sqrt(L2²) correction.

        Args:
            session_data: SessionData with .hu_arrays, .pixel_spacings, .metadata

        Returns:
            dict with anomaly_volume, slice_scores, auto_rois, top_slices
        """
        import torch
        import torch.nn.functional as F

        self.load_model()
        if not self.is_loaded:
            raise RuntimeError("DINOv2 model not available")

        # Step 1: Prepare volume — HU windowing, tissue mask, resample to 224^3
        volume, tissue_mask, orig_shape, zoom_factors = self._prepare_volume(session_data)
        logger.info("Volume prepared: %s → (224,224,224), tissue coverage: %.1f%%",
                     orig_shape, tissue_mask.mean() * 100)

        # Step 2: 2D tissue mask — avg_pool2d per slice instead of avg_pool3d
        mask_slices = torch.from_numpy(tissue_mask.astype(np.float32))  # (224, 224, 224)
        mask_slices = mask_slices.unsqueeze(1)  # (224, 1, 224, 224) — N,C,H,W
        pooled_mask = F.avg_pool2d(mask_slices, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        valid_mask_2d = (pooled_mask.squeeze(1) > MASK_COVERAGE_THRESHOLD).numpy()  # (224, 16, 16)

        n_valid = valid_mask_2d.sum()
        total_tokens = TARGET_SIZE * N_PATCHES
        logger.info("2D token mask: %d/%d valid tokens (%.1f%%)",
                     n_valid, total_tokens, n_valid / total_tokens * 100)

        if n_valid < 20:
            raise ValueError(f"Too few valid tissue tokens ({n_valid}). Volume may be empty or improperly windowed.")

        # Step 3: Encode ALL 224 axial slices (NO depth pooling, axial only)
        logger.info("Encoding %d axial slices (all layers)...", TARGET_SIZE)
        axial_slices = [volume[i, :, :, :] for i in range(TARGET_SIZE)]
        all_layer_tokens = self._encode_slices_all_layers(axial_slices)

        # Step 4: Per-layer: L2 norm → project → temporal-exclusion K-NN score
        layer_scores = []
        for layer_idx in LAYER_INDICES:
            logger.info("Scoring layer %d...", layer_idx)
            tokens = all_layer_tokens[layer_idx]  # (224, 256, 1024)
            # L2 normalize each token
            tokens = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-6)
            tokens_np = tokens.cpu().float().numpy()
            # Random projection: (224, 256, 1024) @ (1024, 64) → (224, 256, 64)
            flat = tokens_np.reshape(-1, EMBED_DIM)
            proj = (flat @ self._proj_matrices[layer_idx]).reshape(TARGET_SIZE, N_PATCHES, PROJ_DIM)
            scores = self._knn_scoring_2d(proj, valid_mask_2d)
            layer_scores.append(scores)

        del all_layer_tokens
        torch.cuda.empty_cache()

        # Step 5: Average across layers + μ+2σ threshold
        final_scores = np.mean(layer_scores, axis=0)  # (224, 16, 16)
        valid_mask_flat = valid_mask_2d.reshape(-1)
        scores_flat = final_scores.reshape(-1)
        scores_flat[~valid_mask_flat] = 0.0

        valid_vals = scores_flat[valid_mask_flat]
        mu, sigma = valid_vals.mean(), valid_vals.std()
        threshold = mu + ANOMALY_THRESHOLD_SIGMA * sigma
        score_max = valid_vals.max()

        logger.info("Score stats: μ=%.4f, σ=%.4f, threshold=%.4f, max=%.4f",
                     mu, sigma, threshold, score_max)

        if score_max > threshold:
            scores_flat[valid_mask_flat] = np.clip(
                (valid_vals - threshold) / (score_max - threshold), 0.0, 1.0
            )
        else:
            scores_flat[:] = 0.0
        scores_flat[~valid_mask_flat] = 0.0

        # Step 6: Upsample (224, 16, 16) → original resolution
        score_maps = scores_flat.reshape(TARGET_SIZE, GRID_DIM, GRID_DIM)
        anomaly_volume = self._upsample_to_original(score_maps, orig_shape, zoom_factors)

        # Step 7: Per-slice ROI extraction
        slice_scores, auto_rois, top_slices = self._extract_rois(anomaly_volume)

        logger.info("Anomaly detection complete: %d auto-ROIs, top slices: %s",
                     len(auto_rois), top_slices)

        return {
            "anomaly_volume": anomaly_volume,
            "slice_scores": slice_scores,
            "auto_rois": auto_rois,
            "top_slices": top_slices,
        }

    # ── Volume preparation ────────────────────────────────────────────────

    def _prepare_volume(self, session_data: Any):
        """Convert HU arrays to multi-window RGB volume + tissue mask, resampled to 224^3.

        Three CT windows → R, G, B channels:
          R: Brain [0, 80] HU — standard parenchyma contrast
          G: Subdural [-20, 180] HU — wide range for extra-axial collections
          B: Blood [20, 60] HU — narrow, maximizes blood vs brain contrast

        Returns:
            (volume_rgb_224, tissue_mask_224, original_shape, zoom_factors)
            where volume_rgb_224 is (224, 224, 224, 3) float32 in [0, 1]
        """
        from scipy.ndimage import binary_opening, zoom

        hu_arrays = session_data.hu_arrays
        if not hu_arrays:
            raise ValueError("No HU arrays in session")

        # Stack to 3D volume (raw HU)
        hu_vol = np.stack(hu_arrays).astype(np.float32)  # (N, H, W)
        orig_shape = hu_vol.shape

        # Brain parenchyma mask (excludes air, fat, skull/bone)
        tissue_mask = (hu_vol > BRAIN_HU_LOW) & (hu_vol < BRAIN_HU_HIGH)
        tissue_mask = binary_opening(tissue_mask, iterations=1)

        # Resample RAW HU to 224^3 (before windowing, preserves HU values)
        zoom_factors = tuple(TARGET_SIZE / s for s in orig_shape)
        hu_224 = zoom(hu_vol, zoom_factors, order=1).astype(np.float32)
        mask_224 = zoom(tissue_mask.astype(np.float32), zoom_factors, order=0) > 0.5

        # Multi-window RGB encoding on resampled volume
        def window_norm(vol, lo, hi):
            return (np.clip(vol, lo, hi) - lo) / (hi - lo)

        r = window_norm(hu_224, *WIN_BRAIN)     # brain [0, 80]
        g = window_norm(hu_224, *WIN_SUBDURAL)   # subdural [-20, 180]
        b = window_norm(hu_224, *WIN_BLOOD)      # blood [20, 60]
        volume_rgb = np.stack([r, g, b], axis=-1)  # (224, 224, 224, 3)

        logger.info("Multi-window RGB: brain[%s], subdural[%s], blood[%s]",
                     WIN_BRAIN, WIN_SUBDURAL, WIN_BLOOD)

        return volume_rgb, mask_224, orig_shape, zoom_factors

    # ── Feature extraction ────────────────────────────────────────────────

    def _encode_slices_all_layers(self, slice_list: list) -> dict:
        """Extract DINOv2 tokens at ALL layers for a list of 2D RGB slices.

        Registers hooks on all layers in LAYER_INDICES simultaneously so only
        one forward pass per batch is needed.

        Args:
            slice_list: list of (H, W, 3) float [0, 1] RGB arrays (224x224x3)

        Returns:
            dict[layer_idx → torch.Tensor of shape (N_slices, n_patches, EMBED_DIM)]
        """
        import torch

        all_tokens: dict[int, list] = {li: [] for li in LAYER_INDICES}
        batch_size = 16

        for i in range(0, len(slice_list), batch_size):
            batch_slices = slice_list[i:i + batch_size]

            # Convert [0,1] float RGB → uint8 RGB PIL images
            # Each slice is (H, W, 3) with distinct windows per channel
            pil_imgs = [
                PIL.Image.fromarray((s * 255).astype(np.uint8), mode="RGB")
                for s in batch_slices
            ]

            # AutoImageProcessor handles: grayscale→RGB, rescale, ImageNet normalize
            inputs = self.processor(
                images=pil_imgs,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
                do_pad=False,
                do_rescale=self.processor.do_rescale,
                do_normalize=self.processor.do_normalize,
            )
            inputs = {k: v.cuda().half() for k, v in inputs.items()}

            # Register hooks on ALL target layers simultaneously
            captured: dict[int, Any] = {}
            handles = []

            def make_hook(li: int):
                def hook_fn(module, inp, output):
                    captured[li] = output
                return hook_fn

            for layer_idx in LAYER_INDICES:
                h = self.encoder_blocks[layer_idx - 1].register_forward_hook(
                    make_hook(layer_idx)
                )
                handles.append(h)

            with torch.inference_mode(), torch.amp.autocast("cuda"):
                self.model(**inputs)

            for h in handles:
                h.remove()

            # Extract patch tokens (skip CLS) for each layer
            for layer_idx in LAYER_INDICES:
                hidden = captured[layer_idx]
                if isinstance(hidden, (tuple, list)):
                    hidden = hidden[0]
                tokens = hidden[:, 1:, :]  # (B, n_patches, EMBED_DIM)
                all_tokens[layer_idx].append(tokens)

        return {li: torch.cat(all_tokens[li], dim=0) for li in LAYER_INDICES}

    # ── K-NN scoring (2D temporal exclusion) ──────────────────────────────

    def _knn_scoring_2d(self, tokens: np.ndarray, valid_mask_2d: np.ndarray) -> np.ndarray:
        """Per-slice anomaly scoring with temporal exclusion.

        For each token on slice i, finds K=1 nearest neighbor from tokens
        on slices at least TEMPORAL_EXCLUDE slices away. Uses sqrt on FAISS
        squared distances for proper L2 distances (matching original CoDeGraph3D
        which uses torch.cdist).

        Chunked processing: divides slices into chunks of TEMPORAL_EXCLUDE,
        builds one FAISS index per chunk from distant reference slices.

        Args:
            tokens: (224, 256, 64) float32 — projected token features
            valid_mask_2d: (224, 16, 16) bool — tissue mask per token

        Returns:
            (224, 16, 16) float32 anomaly scores (raw L2 distances)
        """
        n_slices, n_patches, feat_dim = tokens.shape
        scores = np.zeros((n_slices, n_patches), dtype=np.float32)
        flat_mask = valid_mask_2d.reshape(n_slices, -1)  # (224, 256)

        for chunk_start in range(0, n_slices, TEMPORAL_EXCLUDE):
            chunk_end = min(chunk_start + TEMPORAL_EXCLUDE, n_slices)

            # Reference: slices at least TEMPORAL_EXCLUDE away from ANY slice in chunk
            excl_lo = max(0, chunk_start - TEMPORAL_EXCLUDE + 1)
            excl_hi = min(n_slices, chunk_end + TEMPORAL_EXCLUDE - 1)
            ref_slices = [s for s in range(n_slices) if s < excl_lo or s >= excl_hi]

            if len(ref_slices) < 10:
                logger.debug("Chunk [%d:%d] — too few reference slices (%d), skipping",
                             chunk_start, chunk_end, len(ref_slices))
                continue

            # Collect valid reference tokens from distant slices
            ref_token_list = []
            for s in ref_slices:
                mask_s = flat_mask[s]
                if mask_s.sum() > 0:
                    ref_token_list.append(tokens[s][mask_s])

            if len(ref_token_list) == 0:
                continue

            ref_tokens = np.concatenate(ref_token_list, axis=0).astype(np.float32)
            if len(ref_tokens) < 2:
                continue

            # Build FAISS index for this chunk's reference set
            import faiss
            try:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatL2(res, feat_dim)
            except (AttributeError, RuntimeError):
                index = faiss.IndexFlatL2(feat_dim)
            index.add(np.ascontiguousarray(ref_tokens))

            logger.debug("Chunk [%d:%d] — %d ref tokens from %d slices",
                         chunk_start, chunk_end, len(ref_tokens), len(ref_slices))

            # Score each query slice in this chunk
            for s in range(chunk_start, chunk_end):
                mask_s = flat_mask[s]
                if mask_s.sum() == 0:
                    continue
                query = np.ascontiguousarray(tokens[s][mask_s], dtype=np.float32)
                dists_sq, _ = index.search(query, 1)  # (N_q, 1) — L2 SQUARED
                # CRITICAL: sqrt to get actual L2 distance (matching torch.cdist)
                scores[s, mask_s] = np.sqrt(np.maximum(dists_sq[:, 0], 0.0))

        return scores.reshape(n_slices, GRID_DIM, GRID_DIM)

    # ── Upsampling and ROI extraction ─────────────────────────────────────

    def _upsample_to_original(
        self, score_grid: np.ndarray, orig_shape: tuple, zoom_factors: tuple
    ) -> np.ndarray:
        """Upsample (224, 16, 16) score grid back to original volume resolution.

        Two-step: first spatial dims to (224, 224, 224), then to original shape.
        """
        from scipy.ndimage import zoom

        # Step 1: (224, 16, 16) → (224, 224, 224): spatial dims only
        factors_to_224 = (1.0, TARGET_SIZE / GRID_DIM, TARGET_SIZE / GRID_DIM)
        vol_224 = zoom(score_grid, factors_to_224, order=1).astype(np.float32)

        # Step 2: (224, 224, 224) → original shape
        inv_factors = tuple(1.0 / z for z in zoom_factors)
        return zoom(vol_224, inv_factors, order=1).astype(np.float32)

    def _extract_rois(self, anomaly_volume: np.ndarray) -> tuple:
        """Extract per-slice ROIs from the 3D anomaly volume.

        Returns:
            (slice_scores, auto_rois, top_slices)
        """
        from scipy.ndimage import label

        n_slices, h, w = anomaly_volume.shape

        # Per-slice max scores
        slice_scores = [float(anomaly_volume[i].max()) for i in range(n_slices)]

        # Threshold on tissue-only scores (non-zero)
        tissue_scores = anomaly_volume[anomaly_volume > 0]
        if len(tissue_scores) == 0:
            return slice_scores, {}, list(range(min(TOP_K_SLICES, n_slices)))

        # Scores are already μ+2σ normalized — use fixed threshold for ROI
        threshold = 0.3

        auto_rois: dict[int, dict] = {}

        for i in range(n_slices):
            slice_map = anomaly_volume[i]
            binary = slice_map > threshold

            labeled, n_components = label(binary)
            if n_components == 0:
                continue

            best_area = 0
            best_bbox = None
            for comp_id in range(1, n_components + 1):
                comp_mask = labeled == comp_id
                area = comp_mask.sum()
                if area < MIN_COMPONENT_AREA:
                    continue
                if area > best_area:
                    best_area = area
                    ys, xs = np.where(comp_mask)
                    pad_y, pad_x = int(h * 0.03), int(w * 0.03)
                    y_min = max(0, ys.min() - pad_y)
                    y_max = min(h, ys.max() + pad_y)
                    x_min = max(0, xs.min() - pad_x)
                    x_max = min(w, xs.max() + pad_x)
                    best_bbox = (x_min, y_min, x_max, y_max)
                    best_area = area

            if best_bbox:
                x_min, y_min, x_max, y_max = best_bbox
                auto_rois[i] = {
                    "x": x_min / w,
                    "y": y_min / h,
                    "width": (x_max - x_min) / w,
                    "height": (y_max - y_min) / h,
                }

        sorted_indices = sorted(range(n_slices), key=lambda i: slice_scores[i], reverse=True)
        top_slices = sorted_indices[:TOP_K_SLICES]

        return slice_scores, auto_rois, top_slices

    # ── Heatmap rendering ─────────────────────────────────────────────────

    @staticmethod
    def render_heatmap_png(slice_anomaly: np.ndarray) -> bytes:
        """Render a 2D anomaly map as a heatmap PNG — only anomalous pixels shown.

        Scores are already thresholded: 0 = normal (transparent), >0 = anomalous.
        """
        import matplotlib.cm as cm

        h, w = slice_anomaly.shape
        colored = cm.hot(slice_anomaly)  # (H, W, 4) float64
        rgba = (colored * 255).astype(np.uint8)

        # Hard cutoff: only pixels with score > 0 are visible
        # Alpha 100-230 for anomalous regions (high visibility)
        visible = slice_anomaly > 0
        alpha = np.zeros((h, w), dtype=np.uint8)
        alpha[visible] = (slice_anomaly[visible] * 128 + 100).clip(100, 230).astype(np.uint8)
        rgba[:, :, 3] = alpha

        img = PIL.Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
