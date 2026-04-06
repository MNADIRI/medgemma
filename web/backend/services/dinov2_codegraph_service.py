"""DINOv2 + CoDeGraph3D anomaly detection service.

Training-free anomaly detection for 3D CT volumes using:
  1. DINOv2 ViT-B/14 (HuggingFace) multi-layer feature extraction (3 axes)
  2. Patch-aligned depth pooling + L2 normalization + axis permutation
  3. Random projection (768 → 128) per axis per layer, fused to 384-dim
  4. Self-referencing K-NN scoring with tissue masking and spatial exclusion
  5. Multi-layer score averaging → 3D anomaly map → per-slice ROI extraction

Adapted from CoDeGraph3D (arxiv 2602.15315) for single-volume use.
The cross-volume MSM graph is not applicable to single-volume; instead we use
self-referencing K-NN exploiting brain CT bilateral symmetry.
"""

import io
import logging
from typing import Any

import numpy as np
import PIL.Image

logger = logging.getLogger(__name__)

# ── DINOv2 ViT-B/14 parameters ───────────────────────────────────────────
PATCH_SIZE = 14
EMBED_DIM = 768                       # DINOv2-B hidden_size
PROJ_DIM = 128                        # random projection target dim
LAYER_INDICES = [3, 6, 9, 12]        # 4 layers for ViT-B (12 total)
TARGET_SIZE = 224                      # resample volume to 224^3
GRID_DIM = TARGET_SIZE // PATCH_SIZE  # = 16 tokens per axis

# ── Scoring parameters ───────────────────────────────────────────────────
K_NEIGHBORS = 50
EXCLUDE_RADIUS = 2
MIN_COMPONENT_AREA = 50
ANOMALY_THRESHOLD_SIGMA = 2.0
TOP_K_SLICES = 5

# ── CT windowing ─────────────────────────────────────────────────────────
HU_MIN = -135   # soft-tissue window low
HU_MAX = 215    # soft-tissue window high
TISSUE_HU_THRESHOLD = -200  # air/background threshold


class DINOv2CoDeGraphService:
    """Lazy-loaded DINOv2 + self-referencing anomaly detector for CT volumes."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.encoder_blocks = None
        self._device = None
        self._proj_matrices: dict[int, np.ndarray] = {}  # layer_idx → (768, 128)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> None:
        """Load DINOv2 ViT-B/14 via HuggingFace transformers. Called lazily."""
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
            logger.info("Loading DINOv2 ViT-B/14 from HuggingFace...")
            self.processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-base", use_fast=True
            )
            model = AutoModel.from_pretrained("facebook/dinov2-base")
            model = model.half().eval().cuda()
            self.model = model
            self.encoder_blocks = model.encoder.layer
            self._device = "cuda"
            logger.info("DINOv2-B loaded on CUDA (fp16, %d layers, %d-dim)",
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

        # Step 2: Build token-level tissue mask
        mask_tensor = torch.from_numpy(tissue_mask.astype(np.float32))
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        pooled_mask = F.max_pool3d(mask_tensor, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        valid_mask = (pooled_mask.squeeze() > 0).view(-1).numpy()  # (GRID_DIM^3,)
        n_valid = valid_mask.sum()
        logger.info("Token mask: %d/%d valid tokens (%.1f%%)",
                     n_valid, valid_mask.size, n_valid / valid_mask.size * 100)

        if n_valid < K_NEIGHBORS + 10:
            raise ValueError(f"Too few valid tissue tokens ({n_valid}). Volume may be empty or improperly windowed.")

        # Step 3: Multi-layer feature extraction + scoring
        layer_scores = []
        for layer_idx in LAYER_INDICES:
            logger.info("Processing layer %d/12...", layer_idx)
            fused = self._extract_and_fuse_layer(volume, layer_idx)  # (GRID_DIM^3, 384)
            scores = self._knn_scoring(fused, valid_mask)             # (GRID_DIM^3,)
            layer_scores.append(scores)
            torch.cuda.empty_cache()

        # Step 4: Average across layers
        final_scores = np.mean(layer_scores, axis=0)  # (GRID_DIM^3,)
        final_scores[~valid_mask] = 0.0

        # Normalize valid scores to [0, 1]
        valid_vals = final_scores[valid_mask]
        if valid_vals.max() > valid_vals.min():
            normalized = (valid_vals - valid_vals.min()) / (valid_vals.max() - valid_vals.min())
            final_scores[valid_mask] = normalized

        # Step 5: Reshape to 3D grid and upsample to original resolution
        score_grid = final_scores.reshape(GRID_DIM, GRID_DIM, GRID_DIM)
        anomaly_volume = self._upsample_to_original(score_grid, orig_shape, zoom_factors)

        # Step 6: Per-slice ROI extraction
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
        """Convert HU arrays to windowed [0,1] volume + tissue mask, resampled to 224^3.

        Returns:
            (volume_224, tissue_mask_224, original_shape, zoom_factors)
        """
        from scipy.ndimage import zoom

        hu_arrays = session_data.hu_arrays
        if not hu_arrays:
            raise ValueError("No HU arrays in session")

        # Stack to 3D volume
        volume = np.stack(hu_arrays).astype(np.float32)  # (N, H, W)
        orig_shape = volume.shape

        # Tissue mask (before windowing)
        tissue_mask = volume > TISSUE_HU_THRESHOLD  # bool (N, H, W)

        # Soft-tissue window → [0, 1]
        volume = np.clip(volume, HU_MIN, HU_MAX)
        volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)  # [0, 1]

        # Resample to 224^3
        zoom_factors = tuple(TARGET_SIZE / s for s in orig_shape)
        volume_224 = zoom(volume, zoom_factors, order=1).astype(np.float32)
        mask_224 = zoom(tissue_mask.astype(np.float32), zoom_factors, order=0) > 0.5

        return volume_224, mask_224, orig_shape, zoom_factors

    # ── Feature extraction ────────────────────────────────────────────────

    def _extract_and_fuse_layer(self, volume: np.ndarray, layer_idx: int) -> np.ndarray:
        """Extract features for one DINOv2 layer from all 3 axes and fuse.

        Args:
            volume: (224, 224, 224) float [0, 1]
            layer_idx: which encoder layer to extract from

        Returns:
            (GRID_DIM^3, 3*PROJ_DIM) fused projected tokens
        """
        import torch

        axes = ("axial", "coronal", "sagittal")
        proj_matrix = self._proj_matrices[layer_idx]
        voxel_components = []

        for axis_name in axes:
            # Collect all slices along this axis
            slices = self._collect_axis_slices(volume, axis_name)

            # Extract tokens from DINOv2 at this layer
            tokens = self._encode_slices(slices, layer_idx)  # (N_slices, 16*16, 768) tensor

            # Pool along depth, L2-normalize, permute to common grid
            grid_tokens = self._tokens_to_voxel_grid(axis_name, tokens)  # (16, 16, 16, 768)

            # Random projection
            flat = grid_tokens.reshape(-1, EMBED_DIM)  # (4096, 768)
            if isinstance(flat, torch.Tensor):
                flat = flat.cpu().float().numpy()
            proj = flat @ proj_matrix  # (4096, 128)
            voxel_components.append(proj)

        # Concatenate all axes → 384-dim
        fused = np.concatenate(voxel_components, axis=1)  # (4096, 384)
        return fused.astype(np.float32)

    def _collect_axis_slices(self, volume: np.ndarray, axis: str) -> list:
        """Collect ALL 2D slices along an axis from (D, H, W) volume."""
        if axis == "axial":
            return [volume[i, :, :] for i in range(volume.shape[0])]
        elif axis == "coronal":
            return [volume[:, i, :] for i in range(volume.shape[1])]
        elif axis == "sagittal":
            return [volume[:, :, i] for i in range(volume.shape[2])]
        else:
            raise ValueError(f"Unknown axis: {axis}")

    def _encode_slices(self, slice_list: list, layer_idx: int):
        """Extract DINOv2 tokens at a specific layer for a list of 2D grayscale slices.

        Args:
            slice_list: list of (H, W) float [0, 1] arrays (224x224)
            layer_idx: encoder layer index (1-based)

        Returns:
            torch.Tensor of shape (N_slices, n_patches, embed_dim)
        """
        import torch

        all_tokens = []
        batch_size = 16  # 224x224 slices are small, can batch more

        for i in range(0, len(slice_list), batch_size):
            batch_slices = slice_list[i:i + batch_size]

            # Convert [0,1] float → uint8 PIL for AutoImageProcessor
            pil_imgs = [
                PIL.Image.fromarray((s * 255).astype(np.uint8))
                for s in batch_slices
            ]

            # AutoImageProcessor handles: grayscale→RGB, rescale, ImageNet normalize
            inputs = self.processor(
                images=pil_imgs,
                return_tensors="pt",
                do_resize=False,       # already 224x224
                do_center_crop=False,
                do_pad=False,
                do_rescale=self.processor.do_rescale,
                do_normalize=self.processor.do_normalize,
            )
            inputs = {k: v.cuda().half() for k, v in inputs.items()}

            # Register hook on target layer
            captured = {}

            def hook_fn(module, input, output):
                captured["state"] = output

            handle = self.encoder_blocks[layer_idx - 1].register_forward_hook(hook_fn)

            with torch.inference_mode(), torch.amp.autocast("cuda"):
                self.model(**inputs)

            handle.remove()

            # Extract patch tokens (skip CLS)
            hidden = captured["state"]
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
            tokens = hidden[:, 1:, :]  # (B, n_patches, 768)
            all_tokens.append(tokens)

        return torch.cat(all_tokens, dim=0)  # (N_slices, n_patches, 768)

    def _tokens_to_voxel_grid(self, axis_name: str, tokens):
        """Pool along depth, L2-normalize, and permute to common (x,y,z) frame.

        Args:
            axis_name: "axial", "coronal", or "sagittal"
            tokens: (N_slices, n_patches, embed_dim) tensor

        Returns:
            (GRID_DIM, GRID_DIM, GRID_DIM, embed_dim) tensor
        """
        import torch

        # Pool along depth (groups of PATCH_SIZE=14 slices)
        pooled = self._pool_along_depth(tokens)  # (d, n_patches, embed_dim)

        # L2 normalize each token
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-6)

        # Reshape to 3D grid: (d_depth, d_h, d_w, embed_dim)
        d = pooled.shape[0]
        n_patches_per_slice = pooled.shape[1]
        side = int(np.sqrt(n_patches_per_slice))  # should be 16 for 224/14
        grid = pooled.view(d, side, side, -1)

        # Permute to common coordinate system
        if axis_name == "axial":
            return grid.permute(1, 2, 0, 3)   # (h, w, z, C)
        elif axis_name == "coronal":
            return grid.permute(1, 0, 2, 3)   # (h, z, w, C)
        elif axis_name == "sagittal":
            return grid                         # already (z, h, w, C) — identity
        else:
            raise ValueError(f"Unknown axis: {axis_name}")

    @staticmethod
    def _pool_along_depth(tokens):
        """Average-pool tokens along the depth (slice) dimension.

        Args:
            tokens: (N_slices, n_patches, embed_dim) tensor

        Returns:
            (N_slices // PATCH_SIZE, n_patches, embed_dim) tensor
        """
        d, npatches, dtoken = tokens.shape
        k = PATCH_SIZE
        # Trim to evenly divisible
        if d % k != 0:
            tokens = tokens[:d - (d % k)]
            d = tokens.shape[0]
        return tokens.view(d // k, k, npatches, dtoken).mean(dim=1)

    # ── K-NN scoring ──────────────────────────────────────────────────────

    def _knn_scoring(self, tokens: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Self-referencing K-NN anomaly scoring with tissue masking.

        Args:
            tokens: (N_tokens, feat_dim) float32
            valid_mask: (N_tokens,) bool — True for tissue tokens

        Returns:
            (N_tokens,) anomaly scores (0 for background tokens)
        """
        n_tokens = tokens.shape[0]
        scores = np.zeros(n_tokens, dtype=np.float32)

        # Only process valid (tissue) tokens
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < K_NEIGHBORS + 10:
            return scores

        valid_tokens = np.ascontiguousarray(tokens[valid_indices], dtype=np.float32)
        n_valid = len(valid_indices)

        # Build 3D coordinates for spatial exclusion (in token grid space)
        all_coords = np.array(
            np.unravel_index(np.arange(n_tokens), (GRID_DIM, GRID_DIM, GRID_DIM))
        ).T  # (N_tokens, 3)
        valid_coords = all_coords[valid_indices]  # (n_valid, 3)

        # K-NN search — try FAISS, fallback to scipy
        k_buffer = self._count_spatial_neighbors(EXCLUDE_RADIUS)
        k_search = min(K_NEIGHBORS + k_buffer, n_valid - 1)

        try:
            import faiss
            try:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatL2(res, valid_tokens.shape[1])
            except (AttributeError, RuntimeError):
                index = faiss.IndexFlatL2(valid_tokens.shape[1])
            index.add(valid_tokens)
            distances, nn_indices = index.search(valid_tokens, k_search + 1)
        except ImportError:
            from scipy.spatial import cKDTree
            tree = cKDTree(valid_tokens)
            distances, nn_indices = tree.query(valid_tokens, k=k_search + 1)

        # Score each valid token
        valid_scores = np.zeros(n_valid, dtype=np.float32)
        for i in range(n_valid):
            valid_dists = []
            for j_idx in range(1, k_search + 1):  # skip self at 0
                j = nn_indices[i, j_idx]
                if j < 0 or j >= n_valid:
                    continue
                # Chebyshev spatial distance exclusion
                spatial_dist = np.max(np.abs(valid_coords[i] - valid_coords[j]))
                if spatial_dist <= EXCLUDE_RADIUS:
                    continue
                valid_dists.append(distances[i, j_idx])
                if len(valid_dists) >= K_NEIGHBORS:
                    break
            if valid_dists:
                valid_scores[i] = np.mean(valid_dists)

        # Write back to full score array
        scores[valid_indices] = valid_scores
        return scores

    @staticmethod
    def _count_spatial_neighbors(radius: int) -> int:
        """Count voxels within Chebyshev radius (for K-NN search buffer)."""
        side = 2 * radius + 1
        return side ** 3 - 1

    # ── Upsampling and ROI extraction ─────────────────────────────────────

    def _upsample_to_original(
        self, score_grid: np.ndarray, orig_shape: tuple, zoom_factors: tuple
    ) -> np.ndarray:
        """Upsample (16,16,16) score grid back to original volume resolution.

        Two-step: first to (224,224,224), then to original shape.
        """
        from scipy.ndimage import zoom

        # Step 1: 16^3 → 224^3
        factor_to_224 = TARGET_SIZE / GRID_DIM  # = 14
        vol_224 = zoom(score_grid, factor_to_224, order=1).astype(np.float32)

        # Step 2: 224^3 → original shape
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

        mean_score = float(np.mean(tissue_scores))
        std_score = float(np.std(tissue_scores))
        threshold = mean_score + ANOMALY_THRESHOLD_SIGMA * std_score

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
        """Render a 2D anomaly map as a semi-transparent heatmap PNG."""
        import matplotlib.cm as cm

        h, w = slice_anomaly.shape
        colored = cm.hot(slice_anomaly)  # (H, W, 4) float64
        rgba = (colored * 255).astype(np.uint8)
        # Alpha proportional to score (max 0.6 opacity)
        rgba[:, :, 3] = (slice_anomaly * 153).clip(0, 153).astype(np.uint8)

        img = PIL.Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
