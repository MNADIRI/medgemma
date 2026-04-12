"""DINOv2 + CoDeGraph3D anomaly detection service.

Based on CoDeGraph3D/MuSc3D (arxiv 2602.15315), adapted for single-volume
brain CT anomaly detection:
  1. DINOv2 ViT-L/14 multi-layer feature extraction (3 axes)
  2. Multi-window RGB encoding: Brain [0,80], Subdural [-20,180], Blood [20,60] HU
  3. Depth pooling (224→16) + L2 normalization + axis permutation → 16³ grid
  4. Random projection (1024→256) per axis per layer, concatenated → 768-dim
  5. Self-referencing MSM scoring: torch.cdist (actual L2) + Chebyshev spatial
     exclusion (R=5) + top-5% mean aggregation
  6. Multi-layer averaging → μ+1.5σ threshold → per-slice ROI extraction

Adapted for single-volume CT: larger exclusion radius (R=5) compensates for
self-referencing (vs cross-volume in original), higher projection dim (256 vs
64) preserves subtle CT feature differences, lower threshold for sensitivity.
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
PROJ_DIM = 256                        # random projection target dim per axis
LAYER_INDICES = [6, 12, 18, 24]       # 4 layers for ViT-L (24 total)
TARGET_SIZE = 224                      # resample volume to 224^3
GRID_DIM = TARGET_SIZE // PATCH_SIZE  # = 16 tokens per axis

# ── Scoring parameters (tuned for single-volume self-referencing) ────────
EXCLUDE_RADIUS = 5                    # Chebyshev spatial exclusion (larger for self-ref)
TOPK_RATIO = 0.05                     # top-5% aggregation (more sensitive than original 10%)
ANOMALY_THRESHOLD_SIGMA = 1.5         # μ+Nσ threshold (lower for CT sensitivity)
MIN_COMPONENT_AREA = 30
TOP_K_SLICES = 5

# ── CT multi-window RGB encoding ─────────────────────────────────────────
WIN_BRAIN = (0, 80)       # R: brain parenchyma (L=40, W=80)
WIN_SUBDURAL = (-20, 180) # G: subdural/wide (L=80, W=200)
WIN_BLOOD = (20, 60)      # B: narrow blood (L=40, W=40)
BRAIN_HU_LOW = 0
BRAIN_HU_HIGH = 100
MASK_COVERAGE_THRESHOLD = 0.5


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
        # Matches original: column-normalized Gaussian × 1/√proj_dim
        for layer_idx in LAYER_INDICES:
            rng = np.random.RandomState(42 + layer_idx)
            mat = rng.randn(EMBED_DIM, PROJ_DIM).astype(np.float32)
            mat /= np.linalg.norm(mat, axis=0, keepdims=True)
            mat *= 1.0 / np.sqrt(PROJ_DIM)
            self._proj_matrices[layer_idx] = mat

    # ── Main pipeline ─────────────────────────────────────────────────────

    def detect_anomaly(self, session_data: Any) -> dict:
        """Run CoDeGraph3D-based anomaly detection on a session's CT volume.

        3 axes, depth pooling, 16³ grid, 768-dim features (3×256),
        torch.cdist distances, top-5% mean scoring, 4-layer averaging.

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

        # Step 1: Prepare volume — multi-window RGB + tissue mask → 224³
        volume, tissue_mask, orig_shape, zoom_factors = self._prepare_volume(session_data)
        logger.info("Volume prepared: %s → (224,224,224), tissue coverage: %.1f%%",
                     orig_shape, tissue_mask.mean() * 100)

        # Step 2: 3D tissue mask — F.max_pool3d > 0 (matching original)
        mask_tensor = torch.from_numpy(tissue_mask.astype(np.float32))
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        pooled_mask = F.max_pool3d(mask_tensor, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        valid_mask = (pooled_mask.squeeze() > 0).view(-1).numpy()  # (4096,)
        n_valid = valid_mask.sum()
        logger.info("3D token mask: %d/%d valid voxels (%.1f%%)",
                     n_valid, GRID_DIM ** 3, n_valid / GRID_DIM ** 3 * 100)

        if n_valid < 20:
            raise ValueError(f"Too few valid tissue voxels ({n_valid}).")

        # Step 3: Multi-axis feature extraction (3 axes × all layers)
        layer_axis_projs: dict[int, list] = {li: [] for li in LAYER_INDICES}

        for axis_name in ("axial", "coronal", "sagittal"):
            logger.info("Encoding %s slices (all layers)...", axis_name)
            slices = self._collect_axis_slices(volume, axis_name)
            all_layer_tokens = self._encode_slices_all_layers(slices)

            for layer_idx in LAYER_INDICES:
                tokens = all_layer_tokens[layer_idx]
                grid_tokens = self._tokens_to_voxel_grid(axis_name, tokens)
                flat = grid_tokens.reshape(-1, EMBED_DIM)
                if isinstance(flat, torch.Tensor):
                    flat = flat.cpu().float().numpy()
                proj = flat @ self._proj_matrices[layer_idx]  # (4096, 64)
                layer_axis_projs[layer_idx].append(proj)

            del all_layer_tokens
            torch.cuda.empty_cache()

        # Step 4: Concatenate 3 axes → 768-dim, score per layer
        layer_scores = []
        for layer_idx in LAYER_INDICES:
            logger.info("Scoring layer %d...", layer_idx)
            fused = np.concatenate(layer_axis_projs[layer_idx], axis=1).astype(np.float32)
            # fused: (4096, 768) — 3 axes × 256-dim
            scores = self._msm_scoring(fused, valid_mask)
            layer_scores.append(scores)

            # Per-layer diagnostics
            lv = scores[valid_mask]
            if lv.max() > 0:
                pcts = np.percentile(lv[lv > 0], [50, 90, 95, 99])
                logger.info("  Layer %d scores: min=%.4f, p50=%.4f, p90=%.4f, "
                            "p95=%.4f, p99=%.4f, max=%.4f",
                            layer_idx, lv[lv > 0].min(), *pcts, lv.max())

        # Step 5: Average across layers + μ+Nσ threshold normalization
        final_scores = np.mean(layer_scores, axis=0)  # (4096,)
        final_scores[~valid_mask] = 0.0

        valid_vals = final_scores[valid_mask]
        mu = np.mean(valid_vals)
        sigma = np.std(valid_vals)
        threshold = mu + ANOMALY_THRESHOLD_SIGMA * sigma
        score_max = valid_vals.max()

        # Detailed score distribution for diagnostics
        pctiles = np.percentile(valid_vals, [25, 50, 75, 90, 95, 99])
        logger.info("Score distribution (valid voxels, n=%d):", n_valid)
        logger.info("  p25=%.4f  p50=%.4f  p75=%.4f  p90=%.4f  p95=%.4f  p99=%.4f",
                     *pctiles)
        logger.info("  μ=%.4f, σ=%.4f, threshold(μ+%.1fσ)=%.4f, max=%.4f",
                     mu, sigma, ANOMALY_THRESHOLD_SIGMA, threshold, score_max)

        # Locate the max-score voxel in grid coordinates
        max_flat_idx = np.argmax(final_scores)
        max_z, max_y, max_x = np.unravel_index(max_flat_idx, (GRID_DIM, GRID_DIM, GRID_DIM))
        logger.info("  Max score voxel at grid (%d, %d, %d), score=%.4f",
                     max_z, max_y, max_x, score_max)

        n_above = (valid_vals > threshold).sum()
        logger.info("  %d/%d voxels above threshold (%.1f%%)",
                     n_above, n_valid, n_above / n_valid * 100)

        if score_max > threshold:
            thresholded = np.clip(
                (valid_vals - threshold) / (score_max - threshold), 0.0, 1.0
            )
        else:
            logger.warning("No voxels above threshold — all scores zeroed!")
            thresholded = np.zeros_like(valid_vals)
        final_scores[valid_mask] = thresholded
        final_scores[~valid_mask] = 0.0

        # Step 6: Reshape to 16³ grid and upsample to original resolution
        score_grid = final_scores.reshape(GRID_DIM, GRID_DIM, GRID_DIM)
        anomaly_volume = self._upsample_to_original(score_grid, orig_shape, zoom_factors)

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

        hu_vol = np.stack(hu_arrays).astype(np.float32)  # (N, H, W)
        orig_shape = hu_vol.shape

        # Brain parenchyma mask (CT-specific, excludes air/fat/skull)
        tissue_mask = (hu_vol > BRAIN_HU_LOW) & (hu_vol < BRAIN_HU_HIGH)
        tissue_mask = binary_opening(tissue_mask, iterations=1)

        # Resample RAW HU to 224^3 (before windowing)
        zoom_factors = tuple(TARGET_SIZE / s for s in orig_shape)
        hu_224 = zoom(hu_vol, zoom_factors, order=1).astype(np.float32)
        mask_224 = zoom(tissue_mask.astype(np.float32), zoom_factors, order=0) > 0.5

        # Multi-window RGB encoding
        def window_norm(vol, lo, hi):
            return (np.clip(vol, lo, hi) - lo) / (hi - lo)

        r = window_norm(hu_224, *WIN_BRAIN)
        g = window_norm(hu_224, *WIN_SUBDURAL)
        b = window_norm(hu_224, *WIN_BLOOD)
        volume_rgb = np.stack([r, g, b], axis=-1)  # (224, 224, 224, 3)

        logger.info("Multi-window RGB: brain[%s], subdural[%s], blood[%s]",
                     WIN_BRAIN, WIN_SUBDURAL, WIN_BLOOD)

        return volume_rgb, mask_224, orig_shape, zoom_factors

    # ── Feature extraction ────────────────────────────────────────────────

    def _collect_axis_slices(self, volume: np.ndarray, axis: str) -> list:
        """Collect ALL 2D slices along an axis from (Z, Y, X, 3) RGB volume."""
        if axis == "axial":
            return [volume[i, :, :, :] for i in range(volume.shape[0])]
        elif axis == "coronal":
            return [volume[:, i, :, :] for i in range(volume.shape[1])]
        elif axis == "sagittal":
            return [volume[:, :, i, :] for i in range(volume.shape[2])]
        else:
            raise ValueError(f"Unknown axis: {axis}")

    def _encode_slices_all_layers(self, slice_list: list) -> dict:
        """Extract DINOv2 tokens at ALL layers for a list of 2D RGB slices.

        Registers hooks on all layers in LAYER_INDICES simultaneously so only
        one forward pass per batch is needed (3 passes total for 3 axes).

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

            pil_imgs = [
                PIL.Image.fromarray((s * 255).astype(np.uint8), mode="RGB")
                for s in batch_slices
            ]

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

            for layer_idx in LAYER_INDICES:
                hidden = captured[layer_idx]
                if isinstance(hidden, (tuple, list)):
                    hidden = hidden[0]
                tokens = hidden[:, 1:, :]  # (B, n_patches, EMBED_DIM)
                all_tokens[layer_idx].append(tokens)

        return {li: torch.cat(all_tokens[li], dim=0) for li in LAYER_INDICES}

    def _tokens_to_voxel_grid(self, axis_name: str, tokens):
        """Pool along depth, L2-normalize, and permute to common (Z, Y, X) frame.

        Matches original CoDeGraph3D _tokens_to_voxel_grid exactly, adapted
        for our volume convention (Z, Y, X) instead of original (X, Y, Z).

        Args:
            axis_name: "axial", "coronal", or "sagittal"
            tokens: (N_slices, n_patches, embed_dim) tensor

        Returns:
            (GRID_DIM, GRID_DIM, GRID_DIM, embed_dim) tensor
        """
        import torch

        pooled = self._pool_along_depth(tokens)  # (16, 256, 1024)

        # L2 normalize — matching original: pooled / (norm + eps) with eps=1e-6
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-6)

        # Reshape to 3D grid
        d = pooled.shape[0]
        side = int(np.sqrt(pooled.shape[1]))  # 16
        grid = pooled.view(d, side, side, -1)  # (16, 16, 16, 1024)

        # Permute to common (Z, Y, X) frame.
        # Our volume = (Z, Y, X). Slices extracted along:
        #   axial→Z:    grid=(d_Z, h_Y, w_X) → identity
        #   coronal→Y:  grid=(d_Y, h_Z, w_X) → permute(1, 0, 2, 3) → (Z, Y, X)
        #   sagittal→X: grid=(d_X, h_Z, w_Y) → permute(1, 2, 0, 3) → (Z, Y, X)
        if axis_name == "axial":
            return grid
        elif axis_name == "coronal":
            return grid.permute(1, 0, 2, 3)
        elif axis_name == "sagittal":
            return grid.permute(1, 2, 0, 3)
        else:
            raise ValueError(f"Unknown axis: {axis_name}")

    @staticmethod
    def _pool_along_depth(tokens):
        """Average-pool tokens along the depth (slice) dimension.

        Matches original: tokens.view(d//k, k, P, D).mean(dim=1)

        Args:
            tokens: (N_slices, n_patches, embed_dim) tensor

        Returns:
            (N_slices // PATCH_SIZE, n_patches, embed_dim) tensor
        """
        d, npatches, dtoken = tokens.shape
        k = PATCH_SIZE
        if d % k != 0:
            tokens = tokens[:d - (d % k)]
            d = tokens.shape[0]
        return tokens.view(d // k, k, npatches, dtoken).mean(dim=1)

    # ── MSM scoring (faithful to original CoDeGraph3D) ────────────────────

    def _msm_scoring(self, tokens: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Self-referencing MSM scoring adapted from CoDeGraph3D/MuSc3D.

        Based on the original:
        - torch.cdist for actual L2 distances (not FAISS L2²)
        - top-k% mean aggregation

        Adapted for single-volume CT:
        - Chebyshev spatial exclusion (radius=5) to prevent self-matching
        - Per-voxel adaptive k based on each voxel's available references
        - top-5% for sharper anomaly discrimination

        Args:
            tokens: (4096, 768) float32 — fused 3-axis projected features
            valid_mask: (4096,) bool — True for brain tissue voxels

        Returns:
            (4096,) anomaly scores (0 for background voxels)
        """
        import torch

        n_tokens = tokens.shape[0]
        scores = np.zeros(n_tokens, dtype=np.float32)

        valid_idx = np.where(valid_mask)[0]
        n_valid = len(valid_idx)
        if n_valid < 20:
            return scores

        valid_tokens = torch.from_numpy(tokens[valid_idx]).float().cuda()

        # Pairwise L2 distances — torch.cdist (actual L2, not squared)
        dist_matrix = torch.cdist(valid_tokens, valid_tokens)  # (n_valid, n_valid)

        # Spatial exclusion: Chebyshev distance > EXCLUDE_RADIUS
        coords = np.array(
            np.unravel_index(valid_idx, (GRID_DIM, GRID_DIM, GRID_DIM))
        ).T  # (n_valid, 3)
        coords_t = torch.from_numpy(coords).float().cuda()
        diff = coords_t.unsqueeze(0) - coords_t.unsqueeze(1)  # (n, n, 3)
        chebyshev = diff.abs().max(dim=-1).values  # (n, n)

        # Exclude self + spatially close voxels (Chebyshev ≤ R)
        exclude = chebyshev <= EXCLUDE_RADIUS
        dist_matrix[exclude] = float('inf')

        # Per-voxel available references
        n_refs = (~exclude).sum(dim=1)  # (n_valid,)
        min_refs = int(n_refs.min().item())
        median_refs = int(n_refs.float().median().item())
        logger.info("MSM scoring: %d valid, R=%d exclusion, refs min=%d median=%d",
                     n_valid, EXCLUDE_RADIUS, min_refs, median_refs)

        # Sort distances per voxel (inf pushed to end)
        sorted_dists, _ = dist_matrix.sort(dim=1)

        # Per-voxel adaptive k: each voxel uses top-TOPK_RATIO of its own refs
        # This prevents edge voxels (fewer refs) from getting noisy scores
        per_voxel_k = (n_refs.float() * TOPK_RATIO).clamp(min=1).long()  # (n_valid,)
        k_max = int(per_voxel_k.max().item())
        logger.info("  top-k: ratio=%.2f, k range=[%d, %d]",
                     TOPK_RATIO, int(per_voxel_k.min().item()), k_max)

        # Gather top-k per voxel with masking for variable k
        top_k_all = sorted_dists[:, :k_max]  # (n_valid, k_max)
        # Create mask: position j is valid for voxel i if j < per_voxel_k[i]
        col_idx = torch.arange(k_max, device=top_k_all.device).unsqueeze(0)
        k_mask = col_idx < per_voxel_k.unsqueeze(1)  # (n_valid, k_max)
        # Also mask inf values (voxels with very few refs)
        k_mask = k_mask & (top_k_all != float('inf'))

        # Masked mean: sum valid distances / count valid
        top_k_all[~k_mask] = 0.0
        sum_dists = top_k_all.sum(dim=1)
        count_valid = k_mask.sum(dim=1).float().clamp(min=1)
        valid_scores = sum_dists / count_valid

        scores[valid_idx] = valid_scores.cpu().numpy()
        return scores

    # ── Upsampling and ROI extraction ─────────────────────────────────────

    def _upsample_to_original(
        self, score_grid: np.ndarray, orig_shape: tuple, zoom_factors: tuple
    ) -> np.ndarray:
        """Upsample (16, 16, 16) score grid back to original volume resolution.

        Two-step: first to (224, 224, 224), then to original shape.
        """
        from scipy.ndimage import zoom

        # Step 1: 16³ → 224³
        factor_to_224 = TARGET_SIZE / GRID_DIM  # = 14
        vol_224 = zoom(score_grid, factor_to_224, order=1).astype(np.float32)

        # Step 2: 224³ → original shape
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

        tissue_scores = anomaly_volume[anomaly_volume > 0]
        if len(tissue_scores) == 0:
            return slice_scores, {}, list(range(min(TOP_K_SLICES, n_slices)))

        # Scores are already μ+Nσ normalized — use low threshold to catch weak anomalies
        threshold = 0.15

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
        """Render a 2D anomaly map as a heatmap PNG — only anomalous pixels shown."""
        import matplotlib.cm as cm

        h, w = slice_anomaly.shape
        colored = cm.hot(slice_anomaly)  # (H, W, 4) float64
        rgba = (colored * 255).astype(np.uint8)

        visible = slice_anomaly > 0
        alpha = np.zeros((h, w), dtype=np.uint8)
        alpha[visible] = (slice_anomaly[visible] * 128 + 100).clip(100, 230).astype(np.uint8)
        rgba[:, :, 3] = alpha

        img = PIL.Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
