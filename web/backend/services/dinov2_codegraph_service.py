"""DINOv2 + CoDeGraph3D anomaly detection service.

Training-free, zero-shot anomaly detection for 3D CT volumes using:
  1. DINOv2 ViT-B/14 frozen feature extraction (multi-axis)
  2. Patch-aligned cubic pooling → 3D voxel tokens
  3. Gaussian random projection (768 → 128) per axis, fused to 384-dim
  4. Self-referencing K-NN mutual scoring via FAISS
  5. 3D anomaly map → per-slice ROI extraction

Works without any training data — compares tokens within the same volume,
exploiting the symmetry and repetitiveness of normal tissue.
"""

import io
import logging
from typing import Any

import numpy as np
import PIL.Image

logger = logging.getLogger(__name__)

# DINOv2 ViT-B/14 parameters
PATCH_SIZE = 14
INPUT_SIZE = 518  # 37 * 14 = 518 → 37x37 patch grid
GRID_SIZE = 37    # INPUT_SIZE // PATCH_SIZE
EMBED_DIM = 768
PROJ_DIM = 128
K_NEIGHBORS = 50
EXCLUDE_RADIUS = 2
MIN_COMPONENT_AREA = 50   # minimum voxels for a valid anomaly region
ANOMALY_THRESHOLD_SIGMA = 2.0  # mean + N*std threshold
TOP_K_SLICES = 5


class DINOv2CoDeGraphService:
    """Lazy-loaded DINOv2 + CoDeGraph3D anomaly detector for CT volumes."""

    def __init__(self) -> None:
        self.model = None
        self._device = None
        self._proj_matrices: dict[str, np.ndarray] = {}  # axis → (768, 128)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> None:
        """Load DINOv2 ViT-B/14 via torch.hub. Called lazily on first use."""
        if self.model is not None:
            return

        try:
            import torch
        except ImportError:
            logger.error("PyTorch not available — DINOv2 disabled")
            return

        if not torch.cuda.is_available():
            logger.warning("DINOv2 requires CUDA — anomaly detection disabled")
            return

        try:
            logger.info("Loading DINOv2 ViT-B/14 via torch.hub...")
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            model = model.half().eval().cuda()
            self.model = model
            self._device = "cuda"
            logger.info("DINOv2 loaded on CUDA (fp16, ~340MB VRAM)")
        except Exception as exc:
            logger.error("Failed to load DINOv2: %s", exc)
            self.model = None
            return

        # Initialize fixed random projection matrices (one per axis)
        rng = np.random.RandomState(42)
        for axis in ("axial", "coronal", "sagittal"):
            # Gaussian random projection (Johnson-Lindenstrauss)
            mat = rng.randn(EMBED_DIM, PROJ_DIM).astype(np.float32)
            mat /= np.linalg.norm(mat, axis=0, keepdims=True)
            self._proj_matrices[axis] = mat

    def detect_anomaly(self, session_data: Any) -> dict:
        """Run full anomaly detection pipeline on a session's CT volume.

        Args:
            session_data: SessionData with .images (RGB PIL), .pixel_spacings,
                         .slice_metadata

        Returns:
            dict with keys: anomaly_volume, slice_scores, auto_rois, top_slices
        """
        self.load_model()
        if not self.is_loaded:
            raise RuntimeError("DINOv2 model not available")

        images = session_data.images  # list of PIL RGB images
        if not images:
            raise ValueError("No images in session")

        n_slices = len(images)
        h_orig, w_orig = images[0].size[1], images[0].size[0]  # PIL is (W, H)

        # Get voxel spacing for block depth computation
        slice_thickness = self._get_slice_thickness(session_data)
        pixel_spacing_xy = self._get_pixel_spacing_xy(session_data)

        # Step 1: Multi-axis feature extraction
        logger.info("Extracting DINOv2 features for %d slices (3 axes)...", n_slices)
        axial_tokens = self._extract_axial_features(images)  # (N, 37, 37, 768)

        # Build 3D volume array for coronal/sagittal extraction
        volume_rgb = self._images_to_volume(images)  # (N, H, W, 3)

        coronal_tokens = self._extract_reformat_features(volume_rgb, axis="coronal")
        sagittal_tokens = self._extract_reformat_features(volume_rgb, axis="sagittal")

        # Step 2: Cubic pooling — pool along the depth axis to match spatial resolution
        block_depth = max(1, round(PATCH_SIZE * pixel_spacing_xy / max(slice_thickness, 0.1)))
        logger.info("Cubic pooling: block_depth=%d (spacing_xy=%.2f, thickness=%.2f)",
                     block_depth, pixel_spacing_xy, slice_thickness)

        axial_pooled = self._pool_along_depth(axial_tokens, block_depth)  # (D, 37, 37, 768)
        D_ax = axial_pooled.shape[0]

        # For coronal/sagittal, pool spatially to align to same grid
        coronal_pooled = self._align_to_grid(coronal_tokens, target_shape=(D_ax, GRID_SIZE, GRID_SIZE))
        sagittal_pooled = self._align_to_grid(sagittal_tokens, target_shape=(D_ax, GRID_SIZE, GRID_SIZE))

        # Step 3: Random projection per axis (768 → 128)
        ax_proj = axial_pooled.reshape(-1, EMBED_DIM) @ self._proj_matrices["axial"]
        co_proj = coronal_pooled.reshape(-1, EMBED_DIM) @ self._proj_matrices["coronal"]
        sa_proj = sagittal_pooled.reshape(-1, EMBED_DIM) @ self._proj_matrices["sagittal"]

        # Step 4: Multi-view fusion → 384-dim
        n_tokens = ax_proj.shape[0]
        fused = np.concatenate([ax_proj, co_proj, sa_proj], axis=1)  # (N_tokens, 384)
        logger.info("Fused tokens: %d tokens x %d dim", fused.shape[0], fused.shape[1])

        # Step 5: FAISS K-NN mutual scoring
        scores = self._knn_mutual_scoring(fused, D_ax, GRID_SIZE, GRID_SIZE)

        # Step 6: Reshape to 3D grid and upsample
        score_grid = scores.reshape(D_ax, GRID_SIZE, GRID_SIZE)
        anomaly_volume = self._upsample_to_volume(score_grid, n_slices, h_orig, w_orig)

        # Step 7: Per-slice ROI extraction
        slice_scores, auto_rois, top_slices = self._extract_rois(anomaly_volume)

        logger.info("Anomaly detection complete: %d auto-ROIs, top slices: %s",
                     len(auto_rois), top_slices)

        return {
            "anomaly_volume": anomaly_volume,  # (N, H, W) float32
            "slice_scores": slice_scores,       # list[float]
            "auto_rois": auto_rois,             # dict[int, {x, y, width, height}]
            "top_slices": top_slices,            # list[int]
        }

    # ── Feature extraction ────────────────────────────────────────────────

    def _extract_axial_features(self, images: list[PIL.Image.Image]) -> np.ndarray:
        """Extract DINOv2 patch tokens for all axial slices.

        Returns: (N_slices, GRID_SIZE, GRID_SIZE, EMBED_DIM)
        """
        import torch

        all_tokens = []
        batch_size = 8  # Process in batches to limit VRAM

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            tensors = []
            for img in batch_imgs:
                t = self._preprocess_image(img)
                tensors.append(t)

            batch = torch.stack(tensors).cuda()  # (B, 3, 518, 518) fp16

            with torch.inference_mode():
                out = self.model.forward_features(batch)
                patch_tokens = out["x_norm_patchtokens"]  # (B, 37*37, 768)

            tokens_np = patch_tokens.cpu().float().numpy()  # (B, 1369, 768)
            tokens_np = tokens_np.reshape(-1, GRID_SIZE, GRID_SIZE, EMBED_DIM)
            all_tokens.append(tokens_np)

        return np.concatenate(all_tokens, axis=0)  # (N, 37, 37, 768)

    def _extract_reformat_features(self, volume: np.ndarray, axis: str) -> np.ndarray:
        """Extract DINOv2 features from coronal or sagittal reformats.

        Args:
            volume: (N, H, W, 3) uint8 RGB volume
            axis: "coronal" or "sagittal"

        Returns: (N_slices_along_axis, GRID_SIZE, GRID_SIZE, EMBED_DIM)
        """
        import torch

        # Reformat volume along the requested axis
        if axis == "coronal":
            # Coronal: iterate over rows (Y axis)
            n_reformats = volume.shape[1]
            get_slice = lambda idx: volume[:, idx, :, :]  # (N_z, W, 3)
        elif axis == "sagittal":
            # Sagittal: iterate over columns (X axis)
            n_reformats = volume.shape[2]
            get_slice = lambda idx: volume[:, :, idx, :]  # (N_z, H, 3)
        else:
            raise ValueError(f"Unknown axis: {axis}")

        # Subsample reformats to keep computation manageable
        # Target ~GRID_SIZE reformats to match spatial resolution
        step = max(1, n_reformats // GRID_SIZE)
        indices = list(range(0, n_reformats, step))[:GRID_SIZE]

        all_tokens = []
        batch_size = 8

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            tensors = []

            for idx in batch_indices:
                reformat = get_slice(idx)  # (depth, width_or_height, 3)
                img = PIL.Image.fromarray(reformat.astype(np.uint8), "RGB")
                t = self._preprocess_image(img)
                tensors.append(t)

            batch = torch.stack(tensors).cuda()

            with torch.inference_mode():
                out = self.model.forward_features(batch)
                patch_tokens = out["x_norm_patchtokens"]

            tokens_np = patch_tokens.cpu().float().numpy()
            tokens_np = tokens_np.reshape(-1, GRID_SIZE, GRID_SIZE, EMBED_DIM)
            all_tokens.append(tokens_np)

        return np.concatenate(all_tokens, axis=0)

    def _preprocess_image(self, img: PIL.Image.Image):
        """Resize and normalize a PIL image for DINOv2 (returns fp16 tensor)."""
        import torch
        import torchvision.transforms.functional as TF

        img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), PIL.Image.BILINEAR)
        if img_resized.mode != "RGB":
            img_resized = img_resized.convert("RGB")

        t = TF.to_tensor(img_resized)  # (3, 518, 518) float32 [0,1]
        t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return t.half()

    def _images_to_volume(self, images: list[PIL.Image.Image]) -> np.ndarray:
        """Convert list of PIL RGB images to (N, H, W, 3) uint8 array."""
        arrays = []
        # Use a common size (the first image's size)
        target_size = images[0].size  # (W, H)
        for img in images:
            if img.size != target_size:
                img = img.resize(target_size, PIL.Image.BILINEAR)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arrays.append(np.array(img))
        return np.stack(arrays)  # (N, H, W, 3)

    # ── Pooling and alignment ─────────────────────────────────────────────

    def _pool_along_depth(self, tokens: np.ndarray, block_depth: int) -> np.ndarray:
        """Pool axial tokens along the slice (depth) axis.

        Args:
            tokens: (N_slices, Gh, Gw, D_embed)
            block_depth: number of slices to pool together

        Returns: (D_pooled, Gh, Gw, D_embed)
        """
        n, gh, gw, d = tokens.shape
        if block_depth <= 1:
            return tokens

        n_blocks = max(1, n // block_depth)
        # Trim to evenly divisible
        trimmed = tokens[:n_blocks * block_depth]
        pooled = trimmed.reshape(n_blocks, block_depth, gh, gw, d).mean(axis=1)
        return pooled

    def _align_to_grid(self, tokens: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize token grid to match target 3D shape via interpolation.

        Args:
            tokens: (N, Gh, Gw, D_embed)
            target_shape: (D_target, Gh_target, Gw_target)

        Returns: (D_target, Gh_target, Gw_target, D_embed)
        """
        from scipy.ndimage import zoom

        d_in, gh_in, gw_in, embed = tokens.shape
        d_t, gh_t, gw_t = target_shape

        if (d_in, gh_in, gw_in) == (d_t, gh_t, gw_t):
            return tokens

        # Zoom spatial dims, keep embed dim unchanged
        factors = (d_t / d_in, gh_t / gh_in, gw_t / gw_in, 1.0)
        return zoom(tokens, factors, order=1).astype(np.float32)

    # ── K-NN mutual scoring ───────────────────────────────────────────────

    def _knn_mutual_scoring(self, tokens: np.ndarray, D: int, H: int, W: int) -> np.ndarray:
        """Self-referencing K-NN anomaly scoring with spatial neighbor exclusion.

        Args:
            tokens: (N_tokens, feat_dim) float32
            D, H, W: spatial dimensions for neighbor exclusion

        Returns: (N_tokens,) anomaly scores normalized to [0, 1]
        """
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not available, falling back to scipy KNN")
            return self._knn_scoring_scipy(tokens, D, H, W)

        n_tokens, feat_dim = tokens.shape
        tokens_c = np.ascontiguousarray(tokens, dtype=np.float32)

        # Build FAISS index
        k_search = min(K_NEIGHBORS + self._count_spatial_neighbors(EXCLUDE_RADIUS), n_tokens - 1)

        try:
            # Try GPU FAISS first
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, feat_dim)
        except (AttributeError, RuntimeError):
            # Fallback to CPU
            index = faiss.IndexFlatL2(feat_dim)

        index.add(tokens_c)
        distances, indices = index.search(tokens_c, k_search + 1)  # +1 for self

        # Build 3D coordinates for spatial exclusion
        coords = np.array(np.unravel_index(np.arange(n_tokens), (D, H, W))).T  # (N, 3)

        scores = np.zeros(n_tokens, dtype=np.float32)
        for i in range(n_tokens):
            # Filter out self and spatial neighbors
            valid_dists = []
            for j_idx in range(1, k_search + 1):  # skip self at position 0
                j = indices[i, j_idx]
                if j < 0 or j >= n_tokens:
                    continue
                # Check spatial distance
                spatial_dist = np.max(np.abs(coords[i] - coords[j]))
                if spatial_dist <= EXCLUDE_RADIUS:
                    continue
                valid_dists.append(distances[i, j_idx])
                if len(valid_dists) >= K_NEIGHBORS:
                    break

            if valid_dists:
                scores[i] = np.mean(valid_dists)

        # Normalize to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores[:] = 0.0

        return scores

    def _knn_scoring_scipy(self, tokens: np.ndarray, D: int, H: int, W: int) -> np.ndarray:
        """Fallback K-NN scoring using scipy when FAISS is not available."""
        from scipy.spatial import cKDTree

        n_tokens = tokens.shape[0]
        tree = cKDTree(tokens)

        k_search = min(K_NEIGHBORS + self._count_spatial_neighbors(EXCLUDE_RADIUS), n_tokens - 1)
        distances, indices = tree.query(tokens, k=k_search + 1)

        coords = np.array(np.unravel_index(np.arange(n_tokens), (D, H, W))).T

        scores = np.zeros(n_tokens, dtype=np.float32)
        for i in range(n_tokens):
            valid_dists = []
            for j_idx in range(1, k_search + 1):
                j = indices[i, j_idx]
                spatial_dist = np.max(np.abs(coords[i] - coords[j]))
                if spatial_dist <= EXCLUDE_RADIUS:
                    continue
                valid_dists.append(distances[i, j_idx])
                if len(valid_dists) >= K_NEIGHBORS:
                    break
            if valid_dists:
                scores[i] = np.mean(valid_dists)

        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        return scores

    @staticmethod
    def _count_spatial_neighbors(radius: int) -> int:
        """Count voxels within Chebyshev radius (for K-NN search buffer)."""
        side = 2 * radius + 1
        return side ** 3 - 1  # exclude center

    # ── Upsampling and ROI extraction ─────────────────────────────────────

    def _upsample_to_volume(
        self, score_grid: np.ndarray, n_slices: int, h: int, w: int
    ) -> np.ndarray:
        """Upsample 3D anomaly score grid to original volume resolution.

        Args:
            score_grid: (D, Gh, Gw) float32 anomaly scores
            n_slices, h, w: target volume dimensions

        Returns: (n_slices, h, w) float32
        """
        from scipy.ndimage import zoom

        d, gh, gw = score_grid.shape
        factors = (n_slices / d, h / gh, w / gw)
        return zoom(score_grid, factors, order=1).astype(np.float32)

    def _extract_rois(self, anomaly_volume: np.ndarray) -> tuple:
        """Extract per-slice ROIs from the 3D anomaly volume.

        Returns:
            (slice_scores, auto_rois, top_slices)
        """
        from scipy.ndimage import label

        n_slices, h, w = anomaly_volume.shape

        # Per-slice max scores
        slice_scores = [float(anomaly_volume[i].max()) for i in range(n_slices)]

        # Global threshold
        mean_score = float(np.mean(anomaly_volume))
        std_score = float(np.std(anomaly_volume))
        threshold = mean_score + ANOMALY_THRESHOLD_SIGMA * std_score

        auto_rois: dict[int, dict] = {}

        for i in range(n_slices):
            slice_map = anomaly_volume[i]
            binary = slice_map > threshold

            # Connected components
            labeled, n_components = label(binary)
            if n_components == 0:
                continue

            # Find largest component
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
                    # Add some padding (5% of image dims)
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

        # Top-K slices by anomaly score
        sorted_indices = sorted(range(n_slices), key=lambda i: slice_scores[i], reverse=True)
        top_slices = sorted_indices[:TOP_K_SLICES]

        return slice_scores, auto_rois, top_slices

    # ── Heatmap rendering ─────────────────────────────────────────────────

    @staticmethod
    def render_heatmap_png(slice_anomaly: np.ndarray) -> bytes:
        """Render a 2D anomaly map as a semi-transparent heatmap PNG.

        Args:
            slice_anomaly: (H, W) float32 in [0, 1]

        Returns: PNG bytes (RGBA, alpha proportional to anomaly score)
        """
        import matplotlib.cm as cm

        h, w = slice_anomaly.shape
        # Apply 'hot' colormap
        colored = cm.hot(slice_anomaly)  # (H, W, 4) float64 in [0, 1]
        rgba = (colored * 255).astype(np.uint8)

        # Set alpha proportional to anomaly score (max 0.6 opacity)
        rgba[:, :, 3] = (slice_anomaly * 153).clip(0, 153).astype(np.uint8)  # 153/255 ≈ 0.6

        img = PIL.Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_slice_thickness(session_data: Any) -> float:
        """Extract slice thickness from session metadata."""
        # Try from metadata
        meta = session_data.metadata
        if meta and "slice_thickness" in meta:
            try:
                return float(meta["slice_thickness"])
            except (ValueError, TypeError):
                pass

        # Infer from slice positions if available
        if len(session_data.slice_metadata) >= 2:
            positions = []
            for sm in session_data.slice_metadata:
                if isinstance(sm, dict) and "ImagePositionPatient" in sm:
                    pos = sm["ImagePositionPatient"]
                    if pos and len(pos) >= 3:
                        positions.append(float(pos[2]))
            if len(positions) >= 2:
                positions.sort()
                diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
                return abs(float(np.median(diffs)))

        return 1.0  # fallback

    @staticmethod
    def _get_pixel_spacing_xy(session_data: Any) -> float:
        """Extract in-plane pixel spacing (average of row/col) from session."""
        if session_data.pixel_spacings:
            row, col = session_data.pixel_spacings[0]
            return float((row + col) / 2)
        return 1.0  # fallback
