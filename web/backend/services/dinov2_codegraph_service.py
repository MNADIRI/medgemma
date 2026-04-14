"""DINOv2 + SCGAD anomaly detection service.

Spatially-Coherent Graph Anomaly Detection (SCGAD) for single-volume
brain CT, replacing the MSM self-referencing approach:
  1. DINOv2 ViT-L/14 multi-layer feature extraction (3 axes)
  2. Multi-window RGB encoding: Brain [0,80], Subdural [-20,180], Blood [20,60] HU
  3. Depth pooling (224->16) + L2 normalization + axis permutation -> 16^3 grid
  4. Random projection (1024->256) per axis per layer, concatenated -> 768-dim
  5. Multi-layer feature AVERAGING -> L2 normalization -> single 768-dim token set
  6. SCGAD scoring:
     a. Build spatial coherence graph (26-connectivity + cosine similarity)
     b. Adaptive threshold (percentile of neighbor similarities)
     c. Union-Find connected component decomposition
     d. Component scoring: background identification + cosine contrast + size weighting
     e. Multi-scale consensus (3 thresholds) -> mean score
  7. MAD-based adaptive threshold -> per-slice ROI extraction

Key advantages over MSM:
  - O(n) scoring vs O(n^2) pairwise distances
  - No GPU needed for scoring stage (CPU numpy only)
  - Exploits spatial structure of lesions (contiguity + internal homogeneity)
  - Robust to self-referencing artifacts (hemorrhage no longer matches distant tissue)
"""

import io
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import PIL.Image

logger = logging.getLogger(__name__)

# -- DINOv2 ViT-L/14 parameters (unchanged) --------------------------------
PATCH_SIZE = 14
EMBED_DIM = 1024                      # DINOv2-L hidden_size
PROJ_DIM = 256                        # random projection target dim per axis
LAYER_INDICES = [6, 12, 18, 24]       # 4 layers for ViT-L (24 total)
TARGET_SIZE = 224                      # resample volume to 224^3
GRID_DIM = TARGET_SIZE // PATCH_SIZE  # = 16 tokens per axis

# -- CT multi-window RGB encoding (unchanged) -------------------------------
WIN_BRAIN = (0, 80)       # R: brain parenchyma (L=40, W=80)
WIN_SUBDURAL = (-20, 180) # G: subdural/wide (L=80, W=200)
WIN_BLOOD = (20, 60)      # B: narrow blood (L=40, W=40)

# -- Tissue mask - wider range than previous [0, 100] ----------------------
TISSUE_HU_LOW = -20       # includes CSF (0-15 HU)
TISSUE_HU_HIGH = 200      # includes calcifications, excludes compact bone

# -- SCGAD scoring parameters ----------------------------------------------
THRESHOLD_QUANTILES = (15, 25, 35)    # multi-scale consensus (3 passes)
BG_COVERAGE = 0.5                     # background = largest components >= 50%
SIZE_SIGMOID_ALPHA = 0.5              # size weighting steepness
SIZE_SIGMOID_GAMMA = 5                # size weighting inflection (voxels)
MIN_ROI_GRID_VOXELS = 2              # min component size to consider

# -- ROI extraction ---------------------------------------------------------
TOP_K_SLICES = 5
MIN_ROI_AREA_2D = 20                 # min 2D ROI area in pixels
ROI_2D_THRESHOLD = 0.10              # low threshold since scores are MAD-normalized

# -- 26-connectivity neighbor offsets (precomputed, 13 unique directions) ---
_NEIGHBOR_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) > (0, 0, 0)
]


class DINOv2CoDeGraphService:
    """Lazy-loaded DINOv2 + SCGAD anomaly detector for CT volumes."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.encoder_blocks = None
        self._device = None
        self._proj_matrices: dict[int, np.ndarray] = {}

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
            logger.error("PyTorch or transformers not available -- DINOv2 disabled")
            return

        if not torch.cuda.is_available():
            logger.warning("DINOv2 requires CUDA -- anomaly detection disabled")
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

        # Initialize random projection matrices -- one per layer
        for layer_idx in LAYER_INDICES:
            rng = np.random.RandomState(42 + layer_idx)
            mat = rng.randn(EMBED_DIM, PROJ_DIM).astype(np.float32)
            mat /= np.linalg.norm(mat, axis=0, keepdims=True)
            mat *= 1.0 / np.sqrt(PROJ_DIM)
            self._proj_matrices[layer_idx] = mat

    # -- Main pipeline ------------------------------------------------------

    def detect_anomaly(self, session_data: Any) -> dict:
        """Run SCGAD anomaly detection on a session's CT volume.

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

        # Stage A: Prepare volume
        volume, tissue_mask, orig_shape, zoom_factors = self._prepare_volume(session_data)
        logger.info("Volume prepared: %s -> (224,224,224), tissue coverage: %.1f%%",
                     orig_shape, tissue_mask.mean() * 100)

        # Grid mask via max pooling
        mask_tensor = torch.from_numpy(tissue_mask.astype(np.float32))
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        pooled_mask = F.max_pool3d(mask_tensor, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        valid_mask = (pooled_mask.squeeze() > 0).view(-1).numpy()
        n_valid = valid_mask.sum()
        logger.info("3D token mask: %d/%d valid voxels (%.1f%%)",
                     n_valid, GRID_DIM ** 3, n_valid / GRID_DIM ** 3 * 100)

        if n_valid < 20:
            raise ValueError(f"Too few valid tissue voxels ({n_valid}).")

        # Stage B: Multi-axis DINOv2 feature extraction
        layer_fused: dict[int, list] = {li: [] for li in LAYER_INDICES}

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
                proj = flat @ self._proj_matrices[layer_idx]
                layer_fused[layer_idx].append(proj)

            del all_layer_tokens
            torch.cuda.empty_cache()

        # Fuse axes per layer, then average across layers
        layer_tokens_768 = []
        for layer_idx in LAYER_INDICES:
            fused = np.concatenate(layer_fused[layer_idx], axis=1).astype(np.float32)
            layer_tokens_768.append(fused)

        avg_tokens = np.mean(layer_tokens_768, axis=0)  # (4096, 768)
        del layer_fused, layer_tokens_768

        # L2 normalize -- critical for cosine similarity in graph
        norms = np.linalg.norm(avg_tokens, axis=1, keepdims=True)
        avg_tokens = avg_tokens / (norms + 1e-6)

        logger.info("Feature extraction complete: tokens %s, norm range [%.3f, %.3f]",
                     avg_tokens.shape, norms[valid_mask].min(), norms[valid_mask].max())

        # Stages C-E: SCGAD scoring
        raw_scores = self._scgad_scoring(avg_tokens, valid_mask)

        # Stage F: Reshape, upsample, threshold, extract ROIs
        score_grid = raw_scores.reshape(GRID_DIM, GRID_DIM, GRID_DIM)
        anomaly_volume = self._upsample_to_original(score_grid, orig_shape, zoom_factors)
        anomaly_volume = self._threshold_and_normalize(anomaly_volume)

        slice_scores, auto_rois, top_slices = self._extract_rois(anomaly_volume)

        logger.info("SCGAD complete: %d auto-ROIs, top slices: %s",
                     len(auto_rois), top_slices)

        return {
            "anomaly_volume": anomaly_volume,
            "slice_scores": slice_scores,
            "auto_rois": auto_rois,
            "top_slices": top_slices,
        }

    # -- Volume preparation -------------------------------------------------

    def _prepare_volume(self, session_data: Any):
        """Convert HU arrays to multi-window RGB volume + tissue mask at 224^3.

        Tissue mask: [-20, 200] HU (wider than original [0, 100]).
        """
        from scipy.ndimage import binary_opening, zoom

        hu_arrays = session_data.hu_arrays
        if not hu_arrays:
            raise ValueError("No HU arrays in session")

        hu_vol = np.stack(hu_arrays).astype(np.float32)
        orig_shape = hu_vol.shape

        tissue_mask = (hu_vol > TISSUE_HU_LOW) & (hu_vol < TISSUE_HU_HIGH)
        tissue_mask = binary_opening(tissue_mask, iterations=1)

        zoom_factors = tuple(TARGET_SIZE / s for s in orig_shape)
        hu_224 = zoom(hu_vol, zoom_factors, order=1).astype(np.float32)
        mask_224 = zoom(tissue_mask.astype(np.float32), zoom_factors, order=0) > 0.5

        def window_norm(vol, lo, hi):
            return (np.clip(vol, lo, hi) - lo) / (hi - lo)

        r = window_norm(hu_224, *WIN_BRAIN)
        g = window_norm(hu_224, *WIN_SUBDURAL)
        b = window_norm(hu_224, *WIN_BLOOD)
        volume_rgb = np.stack([r, g, b], axis=-1)

        logger.info("Multi-window RGB: brain%s, subdural%s, blood%s, mask HU[%d,%d]",
                     WIN_BRAIN, WIN_SUBDURAL, WIN_BLOOD, TISSUE_HU_LOW, TISSUE_HU_HIGH)

        return volume_rgb, mask_224, orig_shape, zoom_factors

    # -- Feature extraction (unchanged) -------------------------------------

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
        """Extract DINOv2 tokens at ALL layers for a list of 2D RGB slices."""
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
                tokens = hidden[:, 1:, :]
                all_tokens[layer_idx].append(tokens)

        return {li: torch.cat(all_tokens[li], dim=0) for li in LAYER_INDICES}

    def _tokens_to_voxel_grid(self, axis_name: str, tokens):
        """Pool along depth, L2-normalize, permute to common (Z, Y, X) frame."""
        import torch

        pooled = self._pool_along_depth(tokens)
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-6)

        d = pooled.shape[0]
        side = int(np.sqrt(pooled.shape[1]))
        grid = pooled.view(d, side, side, -1)

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
        """Average-pool tokens along the depth (slice) dimension."""
        d, npatches, dtoken = tokens.shape
        k = PATCH_SIZE
        if d % k != 0:
            tokens = tokens[:d - (d % k)]
            d = tokens.shape[0]
        return tokens.view(d // k, k, npatches, dtoken).mean(dim=1)

    # -- SCGAD scoring (replaces _msm_scoring) ------------------------------

    def _scgad_scoring(self, tokens: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Spatially-Coherent Graph Anomaly Detection scoring.

        Stages C-E of the SCGAD pipeline:
        1. Build 26-connectivity graph with cosine similarity edges
        2. Multi-scale: for each threshold quantile, decompose + score
        3. Average scores across scales for consensus

        Args:
            tokens: (4096, 768) float32 L2-normalized
            valid_mask: (4096,) bool

        Returns:
            (4096,) float32 anomaly scores (0 for invalid voxels)
        """
        n_total = tokens.shape[0]
        scores = np.zeros(n_total, dtype=np.float32)

        valid_idx = np.where(valid_mask)[0]
        n_valid = len(valid_idx)
        if n_valid < 20:
            return scores

        valid_tokens = tokens[valid_idx].astype(np.float32)
        coords = np.array(
            np.unravel_index(valid_idx, (GRID_DIM, GRID_DIM, GRID_DIM))
        ).T

        # Stage C: Build spatial adjacency + compute similarities
        edges, sims = self._compute_neighbor_similarities(valid_tokens, coords)
        n_edges = len(sims)

        if n_edges == 0:
            logger.warning("No valid neighbor pairs -- skipping SCGAD")
            return scores

        sims_array = np.array(sims, dtype=np.float32)

        logger.info("SCGAD graph: %d nodes, %d edges, sim [%.3f, %.3f], median=%.3f",
                     n_valid, n_edges,
                     sims_array.min(), sims_array.max(), np.median(sims_array))

        # Stages D-E: Multi-scale consensus
        scale_scores = []
        for q in THRESHOLD_QUANTILES:
            tau = float(np.percentile(sims_array, q))
            components = self._union_find_components(n_valid, edges, sims_array, tau)

            comp_scores = self._score_components(valid_tokens, components)
            scale_scores.append(comp_scores)

            # Diagnostics
            sizes = sorted([len(c) for c in components.values()], reverse=True)
            n_scored = int((comp_scores > 0).sum())
            max_s = float(comp_scores.max())
            logger.info("  P%d: tau=%.4f, %d comps (top sizes: %s), %d scored, max=%.4f",
                         q, tau, len(components), sizes[:5], n_scored, max_s)

        avg_scores = np.mean(scale_scores, axis=0)
        scores[valid_idx] = avg_scores
        return scores

    @staticmethod
    def _compute_neighbor_similarities(
        tokens: np.ndarray,
        coords: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[float]]:
        """Build 26-connectivity adjacency and compute cosine similarities.

        Uses spatial hash for O(1) neighbor lookup. Cosine sim = dot product
        since tokens are L2-normalized.
        """
        n = len(tokens)

        spatial_hash: dict[tuple[int, int, int], int] = {}
        for i in range(n):
            spatial_hash[(int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]))] = i

        edges: list[tuple[int, int]] = []
        similarities: list[float] = []

        for i in range(n):
            z, y, x = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
            for dz, dy, dx in _NEIGHBOR_OFFSETS:
                j = spatial_hash.get((z + dz, y + dy, x + dx))
                if j is not None:
                    sim = float(tokens[i] @ tokens[j])
                    edges.append((i, j))
                    similarities.append(sim)

        return edges, similarities

    @staticmethod
    def _union_find_components(
        n: int,
        edges: list[tuple[int, int]],
        sims: np.ndarray,
        tau: float,
    ) -> dict[int, list[int]]:
        """Union-Find with path halving. Connect edges where sim >= tau."""
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        for idx in range(len(edges)):
            if sims[idx] >= tau:
                union(edges[idx][0], edges[idx][1])

        components: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            components[find(i)].append(i)

        return dict(components)

    @staticmethod
    def _score_components(
        tokens: np.ndarray,
        components: dict[int, list[int]],
    ) -> np.ndarray:
        """Score voxels by their component's contrast to the background.

        1. Background = largest components covering >= BG_COVERAGE
        2. Background centroid + MAD-based dispersion
        3. Non-background components: cosine distance z-score * size sigmoid
        4. Intra-component refinement: core voxels > edge voxels
        """
        n = tokens.shape[0]
        scores = np.zeros(n, dtype=np.float32)

        if not components:
            return scores

        # Step 1: Background identification
        comp_list = sorted(components.values(), key=len, reverse=True)
        total_voxels = sum(len(c) for c in comp_list)

        bg_indices: list[int] = []
        bg_comp_count = 0
        for comp in comp_list:
            bg_indices.extend(comp)
            bg_comp_count += 1
            if len(bg_indices) >= total_voxels * BG_COVERAGE:
                break

        # Step 2: Background statistics
        bg_tokens = tokens[bg_indices]
        bg_centroid = bg_tokens.mean(axis=0)
        bg_norm = np.linalg.norm(bg_centroid)
        if bg_norm < 1e-8:
            return scores
        bg_centroid_n = bg_centroid / bg_norm

        bg_cos_dists = 1.0 - bg_tokens @ bg_centroid_n
        bg_median = float(np.median(bg_cos_dists))
        bg_mad = float(np.median(np.abs(bg_cos_dists - bg_median)))
        bg_sigma = bg_mad * 1.4826

        if bg_sigma < 1e-8:
            bg_sigma = 1e-4

        logger.info("  BG: %d voxels (%d comps, %.0f%%), median_dist=%.4f, sigma=%.4f",
                     len(bg_indices), bg_comp_count,
                     len(bg_indices) / total_voxels * 100, bg_median, bg_sigma)

        # Step 3: Score non-background components
        n_scored = 0
        for comp in comp_list[bg_comp_count:]:
            comp_size = len(comp)
            if comp_size < MIN_ROI_GRID_VOXELS:
                continue

            comp_tokens = tokens[comp]
            comp_centroid = comp_tokens.mean(axis=0)
            comp_norm = np.linalg.norm(comp_centroid)
            if comp_norm < 1e-8:
                continue
            comp_centroid_n = comp_centroid / comp_norm

            # Cosine distance to background
            d_k = 1.0 - float(comp_centroid_n @ bg_centroid_n)

            # Z-score
            z_k = max(0.0, (d_k - bg_median) / bg_sigma)

            # Size sigmoid
            w_k = 1.0 / (1.0 + np.exp(-SIZE_SIGMOID_ALPHA * (comp_size - SIZE_SIGMOID_GAMMA)))

            s_k = z_k * w_k
            if s_k <= 0:
                continue

            n_scored += 1

            # Step 4: Intra-component refinement
            intra_dists = 1.0 - comp_tokens @ comp_centroid_n
            max_intra = float(intra_dists.max()) + 1e-8
            refinement = 1.0 - (intra_dists / max_intra)

            for local_idx, global_idx in enumerate(comp):
                scores[global_idx] = s_k * float(refinement[local_idx])

        return scores

    # -- Thresholding -------------------------------------------------------

    @staticmethod
    def _threshold_and_normalize(anomaly_volume: np.ndarray) -> np.ndarray:
        """MAD-based adaptive thresholding + normalization to [0, 1]."""
        nonzero = anomaly_volume[anomaly_volume > 0]
        if len(nonzero) == 0:
            return anomaly_volume

        median_val = float(np.median(nonzero))
        mad = float(np.median(np.abs(nonzero - median_val)))
        sigma_mad = mad * 1.4826

        if sigma_mad < 1e-8:
            sigma_mad = 1e-4

        theta = median_val + 1.5 * sigma_mad
        score_max = float(anomaly_volume.max())

        logger.info("MAD threshold: median=%.4f, sigma=%.4f, theta=%.4f, max=%.4f",
                     median_val, sigma_mad, theta, score_max)

        if score_max <= theta:
            logger.warning("No voxels above MAD threshold -- all zeroed")
            return np.zeros_like(anomaly_volume)

        normalized = np.clip(
            (anomaly_volume - theta) / (score_max - theta), 0.0, 1.0
        ).astype(np.float32)

        n_above = int((anomaly_volume > theta).sum())
        logger.info("  %d voxels above threshold (%.2f%% of nonzero)",
                     n_above, n_above / max(len(nonzero), 1) * 100)

        return normalized

    # -- Upsampling (unchanged) ---------------------------------------------

    def _upsample_to_original(
        self, score_grid: np.ndarray, orig_shape: tuple, zoom_factors: tuple
    ) -> np.ndarray:
        """Upsample (16, 16, 16) score grid back to original volume resolution."""
        from scipy.ndimage import zoom

        factor_to_224 = TARGET_SIZE / GRID_DIM
        vol_224 = zoom(score_grid, factor_to_224, order=1).astype(np.float32)

        inv_factors = tuple(1.0 / z for z in zoom_factors)
        return zoom(vol_224, inv_factors, order=1).astype(np.float32)

    # -- ROI extraction (updated threshold) ---------------------------------

    def _extract_rois(self, anomaly_volume: np.ndarray) -> tuple:
        """Extract per-slice ROIs from the 3D anomaly volume."""
        from scipy.ndimage import label

        n_slices, h, w = anomaly_volume.shape

        slice_scores = [float(anomaly_volume[i].max()) for i in range(n_slices)]

        tissue_scores = anomaly_volume[anomaly_volume > 0]
        if len(tissue_scores) == 0:
            return slice_scores, {}, list(range(min(TOP_K_SLICES, n_slices)))

        auto_rois: dict[int, dict] = {}

        for i in range(n_slices):
            slice_map = anomaly_volume[i]
            binary = slice_map > ROI_2D_THRESHOLD

            labeled, n_components = label(binary)
            if n_components == 0:
                continue

            best_area = 0
            best_bbox = None
            for comp_id in range(1, n_components + 1):
                comp_mask = labeled == comp_id
                area = comp_mask.sum()
                if area < MIN_ROI_AREA_2D:
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

    # -- Heatmap rendering (unchanged) --------------------------------------

    @staticmethod
    def render_heatmap_png(slice_anomaly: np.ndarray) -> bytes:
        """Render a 2D anomaly map as a heatmap PNG."""
        import matplotlib.cm as cm

        h, w = slice_anomaly.shape
        colored = cm.hot(slice_anomaly)
        rgba = (colored * 255).astype(np.uint8)

        visible = slice_anomaly > 0
        alpha = np.zeros((h, w), dtype=np.uint8)
        alpha[visible] = (slice_anomaly[visible] * 128 + 100).clip(100, 230).astype(np.uint8)
        rgba[:, :, 3] = alpha

        img = PIL.Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
