"""DINOv2 + SCGAD anomaly detection service.

Spatially-Coherent Graph Anomaly Detection (SCGAD) for single-volume
brain CT:
  1. DINOv2 ViT-L/14 multi-layer feature extraction (3 axes)
  2. Multi-window RGB encoding: Brain [0,80], Subdural [-20,180], Blood [20,80] HU
  3. Top-3 depth pooling (224->16) + L2 norm + axis permutation -> 16^3 grid
  4. Random projection (1024->256) per axis per layer -> 768-dim
  5. Multi-layer feature averaging -> L2 norm -> single token set
  6. SCGAD scoring with HU-modulated adaptive threshold
  7. MAD-based threshold -> ROI extraction
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

# -- CT multi-window RGB -------------------------------------------------------
WIN_BRAIN = (0, 80)       # R: brain parenchyma (L=40, W=80)
WIN_SUBDURAL = (-20, 180) # G: subdural/wide (L=80, W=200)
WIN_BLOOD = (20, 80)      # B: blood window (70 HU hemorrhage at 0.83, not saturated)

# -- Tissue mask ----------------------------------------------------------------
# Includes CSF (5-15), WM (25-35), GM (35-45), acute blood (50-90 HU).
# Skull-stripping removes extracranial tissue; upper bound at 100 captures
# hemorrhage without reaching cortical bone (>200 HU).
TISSUE_HU_LOW = -10
TISSUE_HU_HIGH = 100

# -- Skull-stripping parameters -------------------------------------------------
BONE_HU_THRESHOLD = 150       # dense cortical bone (skull)
BRAIN_MASK_CLOSING_MM = 10    # closing radius to seal skull gaps (foramen, sutures)
BRAIN_MASK_EROSION_MM = 3     # erosion to pull away from inner skull table

# -- Grid mask strictness -------------------------------------------------------
# Minimum fraction of tissue voxels within a 14^3 patch for the grid voxel
# to be considered valid. Replaces max_pool3d (any single tissue voxel = valid).
GRID_MASK_TISSUE_FRAC = 0.3

# -- SCGAD scoring parameters ----------------------------------------------
THRESHOLD_QUANTILES = (15, 25, 35)    # multi-scale consensus (3 passes)
BG_COVERAGE = 0.5                     # background = largest components >= 50%
SIZE_SIGMOID_ALPHA = 0.5              # size weighting steepness
SIZE_SIGMOID_GAMMA = 5                # size weighting inflection (voxels)
MIN_ROI_GRID_VOXELS = 2              # min component size to consider
LAMBDA_HU = 0.2                      # HU-modulation strength for adaptive tau
HU_DISCORDANCE_EPS = 1.0             # noise floor (1 HU ≈ CT reconstruction noise)
HU_DISCORDANCE_MAX = 3.0             # clamp to prevent extreme threshold values

# -- CSF exemption parameters -----------------------------------------------
CSF_HU_LOW = -5
CSF_HU_HIGH = 20
CSF_MIN_VOXELS = 8           # min grid voxels for a CSF region (ventricles are large)

# -- Asymmetry parameters ---------------------------------------------------
ASYM_WEIGHT = 0.5            # weight of asymmetry modulation on scores

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
        volume, tissue_mask, orig_shape, zoom_factors, hu_224 = self._prepare_volume(session_data)
        logger.info("Volume prepared: %s -> (224,224,224), tissue coverage: %.1f%%",
                     orig_shape, tissue_mask.mean() * 100)

        # Grid mask via avg pooling with tissue fraction threshold
        mask_tensor = torch.from_numpy(tissue_mask.astype(np.float32))
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        pooled_mask = F.avg_pool3d(mask_tensor, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        valid_mask = (pooled_mask.squeeze() > GRID_MASK_TISSUE_FRAC).view(-1).numpy()
        n_valid = valid_mask.sum()
        logger.info("3D token mask: %d/%d valid voxels (%.1f%%, threshold=%.0f%%)",
                     n_valid, GRID_DIM ** 3, n_valid / GRID_DIM ** 3 * 100,
                     GRID_MASK_TISSUE_FRAC * 100)

        if n_valid < 20:
            raise ValueError(f"Too few valid tissue voxels ({n_valid}).")

        # Compute HU grid at 16^3 via masked mean (tissue voxels only)
        k = PATCH_SIZE
        mask_224_f = tissue_mask.astype(np.float32)
        hu_masked = hu_224 * mask_224_f

        hu_sum = hu_masked.reshape(
            GRID_DIM, k, GRID_DIM, k, GRID_DIM, k
        ).sum(axis=(1, 3, 5))
        mask_count = mask_224_f.reshape(
            GRID_DIM, k, GRID_DIM, k, GRID_DIM, k
        ).sum(axis=(1, 3, 5))
        hu_grid = np.where(mask_count > 0, hu_sum / mask_count, 0.0).astype(np.float32)
        hu_grid_flat = hu_grid.reshape(-1)

        valid_hu_stats = hu_grid_flat[valid_mask]
        if len(valid_hu_stats) > 0:
            logger.info("HU grid (valid): [%.1f, %.1f], median=%.1f, mean=%.1f",
                         valid_hu_stats.min(), valid_hu_stats.max(),
                         np.median(valid_hu_stats), valid_hu_stats.mean())

        # Stage B: Multi-axis DINOv2 feature extraction
        layer_fused: dict[int, list] = {li: [] for li in LAYER_INDICES}

        # DIAGNOSTIC: save one raw pre-projection token set (last layer, axial)
        _diag_raw_1024 = None

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

                # DIAGNOSTIC: capture raw 1024-dim tokens (layer 24, axial only)
                if axis_name == "axial" and layer_idx == LAYER_INDICES[-1]:
                    _diag_raw_1024 = flat.copy()

                proj = flat @ self._proj_matrices[layer_idx]
                layer_fused[layer_idx].append(proj)

            del all_layer_tokens
            torch.cuda.empty_cache()

        # Fuse axes per layer, then average across layers
        layer_tokens_768 = []
        for layer_idx in LAYER_INDICES:
            fused = np.concatenate(layer_fused[layer_idx], axis=1).astype(np.float32)
            layer_tokens_768.append(fused)

        # DIAGNOSTIC: check per-layer discriminability before averaging
        self._diagnostic_per_layer(layer_tokens_768, valid_mask, hu_grid_flat)

        avg_tokens = np.mean(layer_tokens_768, axis=0)

        # DIAGNOSTIC: check averaged (pre-L2-norm) discriminability
        self._diagnostic_stage(avg_tokens, valid_mask, hu_grid_flat,
                               "Averaged 768-dim (pre-L2-norm)")

        del layer_fused, layer_tokens_768

        # L2 normalize -- critical for cosine similarity in graph
        norms = np.linalg.norm(avg_tokens, axis=1, keepdims=True)
        avg_tokens = avg_tokens / (norms + 1e-6)

        logger.info("Feature extraction complete: tokens %s, norm range [%.4f, %.4f]",
                     avg_tokens.shape, norms[valid_mask].min(), norms[valid_mask].max())

        # DIAGNOSTIC: check final (post-L2-norm) discriminability
        self._diagnostic_stage(avg_tokens, valid_mask, hu_grid_flat,
                               "Final 768-dim (post-L2-norm)")

        # DIAGNOSTIC: check raw 1024-dim tokens (pre-projection)
        if _diag_raw_1024 is not None:
            raw_normed = _diag_raw_1024.copy()
            raw_norms = np.linalg.norm(raw_normed, axis=1, keepdims=True)
            raw_normed = raw_normed / (raw_norms + 1e-6)
            self._diagnostic_stage(raw_normed, valid_mask, hu_grid_flat,
                                   "Raw 1024-dim L24-axial (pre-projection)")
            del _diag_raw_1024

        # Hemispheric asymmetry: boost unilateral, suppress bilateral
        asym_scores = self._compute_asymmetry(hu_grid_flat, valid_mask)

        # Stages C-E: SCGAD scoring
        raw_scores = self._scgad_scoring(avg_tokens, valid_mask, hu_grid_flat)

        # Modulate by asymmetry: score * (1 + ASYM_WEIGHT * asymmetry)
        # Unilateral findings (asym~1) get boosted by up to 50%
        # Bilateral findings (asym~0, e.g. ventricles) stay unchanged
        raw_scores = raw_scores * (1.0 + ASYM_WEIGHT * asym_scores)

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

    @staticmethod
    def _get_volume_spacings(session_data: Any) -> tuple[float, float, float]:
        """Extract (sz, sy, sx) spacings in mm from session data."""
        # In-plane spacing
        if session_data.pixel_spacings:
            row_sp, col_sp = session_data.pixel_spacings[0]
        else:
            row_sp, col_sp = 1.0, 1.0

        # Slice spacing from consecutive ImagePositionPatient
        sz = 1.0
        if len(session_data.slice_metadata) >= 2:
            ipp0 = session_data.slice_metadata[0].get("image_position_patient")
            ipp1 = session_data.slice_metadata[1].get("image_position_patient")
            if ipp0 is not None and ipp1 is not None:
                dz = abs(ipp1[2] - ipp0[2])
                if dz > 0.01:
                    sz = dz
                else:
                    sz = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(ipp1, ipp0))))
        if sz < 0.01:
            sz = float((session_data.metadata or {}).get("slice_thickness", 1.0))
        if sz < 0.01:
            sz = 1.0

        return (sz, float(row_sp), float(col_sp))

    @staticmethod
    def _extract_brain_mask(hu_vol: np.ndarray, voxel_size_mm: float = 1.0) -> np.ndarray:
        """Morphological skull-stripping to isolate intracranial contents.

        1. Find skull boundary via bone HU threshold (>150)
        2. Close gaps in skull (foramen magnum, sutures) with large structuring element
        3. Fill interior per axial slice to get intracranial cavity
        4. Erode to pull away from inner skull table
        5. Intersect with tissue HU mask
        """
        from scipy.ndimage import (
            binary_closing, binary_dilation, binary_erosion,
            binary_fill_holes, generate_binary_structure, label,
        )

        bone_mask = hu_vol > BONE_HU_THRESHOLD

        # Close gaps in the skull shell so fill_holes works
        close_r = max(1, int(round(BRAIN_MASK_CLOSING_MM / voxel_size_mm)))
        struct_3d = generate_binary_structure(3, 2)
        skull_closed = binary_closing(bone_mask, structure=struct_3d, iterations=close_r)

        # Fill interior per axial slice (more robust than 3D fill for partial volumes)
        intracranial = np.zeros_like(skull_closed)
        for z in range(skull_closed.shape[0]):
            slice_2d = skull_closed[z]
            if slice_2d.any():
                filled = binary_fill_holes(slice_2d)
                intracranial[z] = filled & ~bone_mask[z]
            else:
                intracranial[z] = False

        # Keep only the largest connected component (the brain cavity)
        labeled, n_comps = label(intracranial)
        if n_comps > 1:
            comp_sizes = np.bincount(labeled.ravel())
            comp_sizes[0] = 0
            largest = comp_sizes.argmax()
            intracranial = labeled == largest

        # Dilate slightly to recapture tissue at the brain surface
        intracranial = binary_dilation(intracranial, structure=struct_3d, iterations=1)

        # Erode to pull away from inner skull table
        erode_r = max(1, int(round(BRAIN_MASK_EROSION_MM / voxel_size_mm)))
        brain_mask = binary_erosion(intracranial, structure=struct_3d, iterations=erode_r)

        # Fallback: if erosion removed everything, use uneroded version
        if not brain_mask.any():
            brain_mask = intracranial

        return brain_mask.astype(bool)

    def _prepare_volume(self, session_data: Any):
        """Convert HU arrays to multi-window RGB volume + tissue mask at 224^3."""
        from scipy.ndimage import binary_opening, zoom

        hu_arrays = session_data.hu_arrays
        if not hu_arrays:
            raise ValueError("No HU arrays in session")

        hu_vol = np.stack(hu_arrays).astype(np.float32)
        orig_shape = hu_vol.shape

        tissue_mask = (hu_vol > TISSUE_HU_LOW) & (hu_vol < TISSUE_HU_HIGH)
        tissue_mask = binary_opening(tissue_mask, iterations=2)

        # Skull-stripping: restrict to intracranial contents only
        spacings = self._get_volume_spacings(session_data)
        voxel_mm = float(np.mean(spacings))
        brain_mask = self._extract_brain_mask(hu_vol, voxel_size_mm=voxel_mm)
        n_before = tissue_mask.sum()
        tissue_mask = tissue_mask & brain_mask
        n_after = tissue_mask.sum()
        logger.info("Skull-stripping: %d -> %d tissue voxels (removed %d extracranial, %.1f%%)",
                     n_before, n_after, n_before - n_after,
                     (n_before - n_after) / max(n_before, 1) * 100)

        # Isotropic resampling before going to 224³
        s_iso = min(spacings)
        max_phys = max(d * s for d, s in zip(orig_shape, spacings))
        s_iso = max(s_iso, max_phys / (TARGET_SIZE * 2))

        iso_factors = tuple(sp / s_iso for sp in spacings)
        hu_iso = zoom(hu_vol, iso_factors, order=1).astype(np.float32)
        mask_iso = zoom(tissue_mask.astype(np.float32), iso_factors, order=0) > 0.5
        iso_shape = hu_iso.shape

        logger.info("Isotropic resample: %s (%.2f/%.2f/%.2f mm) -> %s (%.2f mm)",
                     orig_shape, *spacings, iso_shape, s_iso)

        # Isotropic -> 224³
        zoom_factors = tuple(TARGET_SIZE / s for s in orig_shape)
        vol_224_factors = tuple(TARGET_SIZE / s for s in iso_shape)
        hu_224 = zoom(hu_iso, vol_224_factors, order=1).astype(np.float32)
        mask_224 = zoom(mask_iso.astype(np.float32), vol_224_factors, order=0) > 0.5

        def window_norm(vol, lo, hi):
            return (np.clip(vol, lo, hi) - lo) / (hi - lo)

        r = window_norm(hu_224, *WIN_BRAIN)
        g = window_norm(hu_224, *WIN_SUBDURAL)
        b = window_norm(hu_224, *WIN_BLOOD)
        volume_rgb = np.stack([r, g, b], axis=-1)

        logger.info("Multi-window RGB: brain%s, subdural%s, blood%s, mask HU[%d,%d]",
                     WIN_BRAIN, WIN_SUBDURAL, WIN_BLOOD, TISSUE_HU_LOW, TISSUE_HU_HIGH)

        return volume_rgb, mask_224, orig_shape, zoom_factors, hu_224

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
        """Top-3 depth pooling: keep the 3 most salient slices out of 14.

        Replaces mean pooling which diluted focal lesion features.
        A hemorrhage on 3-5 slices out of 14 was diluted to ~25-35% signal.
        Top-3 by L2 norm preserves the most distinctive slice features.
        """
        import torch

        d, npatches, dtoken = tokens.shape
        k = PATCH_SIZE
        if d % k != 0:
            tokens = tokens[:d - (d % k)]
            d = tokens.shape[0]

        grouped = tokens.view(d // k, k, npatches, dtoken)  # (16, 14, 256, 1024)
        n_groups, depth, n_p, dim = grouped.shape

        # Select top-3 slices per spatial position by token norm
        token_norms = grouped.norm(dim=-1)  # (16, 14, 256)
        top_k = min(3, depth)
        _, top_indices = token_norms.topk(top_k, dim=1)  # (16, 3, 256)

        # Gather and average the top-3 tokens
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
        selected = torch.gather(grouped, 1, top_indices_expanded)  # (16, 3, 256, 1024)
        return selected.mean(dim=1)  # (16, 256, 1024)

    # -- Pipeline diagnostics --------------------------------------------------

    @staticmethod
    def _diagnostic_stage(
        tokens: np.ndarray,
        valid_mask: np.ndarray,
        hu_flat: np.ndarray,
        stage_name: str,
    ) -> None:
        """Check feature discriminability at one pipeline stage.

        Uses HU values to identify potentially abnormal voxels (>2 MAD from
        median), then compares cosine similarities between normal-normal and
        suspicious-normal neighbor pairs. A gap > 0.05 indicates the features
        carry discriminative signal at this stage.
        """
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < 30:
            return

        valid_tokens = tokens[valid_idx]
        valid_hu = hu_flat[valid_idx]

        # Identify suspicious voxels by HU
        hu_med = float(np.median(valid_hu))
        hu_mad = float(np.median(np.abs(valid_hu - hu_med)))
        hu_sig = hu_mad * 1.4826
        if hu_sig < 1e-4:
            hu_sig = 1e-4

        susp_local = set(i for i in range(len(valid_idx))
                         if abs(valid_hu[i] - hu_med) > 2.0 * hu_sig)
        norm_local = set(range(len(valid_idx))) - susp_local

        if len(susp_local) < 2 or len(norm_local) < 10:
            logger.info("DIAG [%s]: <2 suspicious voxels (HU threshold=%.1f±%.1f) — skip",
                         stage_name, hu_med, 2 * hu_sig)
            return

        # Sample cosine similarities: normal-normal vs suspicious-normal
        rng = np.random.RandomState(0)
        norm_list = list(norm_local)
        susp_list = list(susp_local)

        # Normal-Normal: 200 random pairs
        nn_sims = []
        for _ in range(min(200, len(norm_list) * (len(norm_list) - 1) // 2)):
            a, b = rng.choice(norm_list, 2, replace=False)
            nn_sims.append(float(valid_tokens[a] @ valid_tokens[b]))

        # Suspicious-Normal: all suspicious × sample of normals
        sn_sims = []
        for s in susp_list:
            sample_n = rng.choice(norm_list, min(20, len(norm_list)), replace=False)
            for n in sample_n:
                sn_sims.append(float(valid_tokens[s] @ valid_tokens[n]))

        # Suspicious-Suspicious
        ss_sims = []
        if len(susp_list) >= 2:
            for i in range(len(susp_list)):
                for j in range(i + 1, len(susp_list)):
                    ss_sims.append(float(valid_tokens[susp_list[i]] @ valid_tokens[susp_list[j]]))

        nn_med = float(np.median(nn_sims)) if nn_sims else 0.0
        sn_med = float(np.median(sn_sims)) if sn_sims else 0.0
        ss_med = float(np.median(ss_sims)) if ss_sims else 0.0
        gap = nn_med - sn_med

        verdict = "GOOD" if gap > 0.05 else "WEAK" if gap > 0.02 else "NONE"

        logger.info("DIAG [%s]: %d suspicious (HU>%.0f), %d normal",
                     stage_name, len(susp_local), hu_med + 2 * hu_sig, len(norm_local))
        logger.info("  Cosine sim — NN: %.3f, SN: %.3f, SS: %.3f | gap=%.4f → %s",
                     nn_med, sn_med, ss_med, gap, verdict)

    @staticmethod
    def _diagnostic_per_layer(
        layer_tokens: list[np.ndarray],
        valid_mask: np.ndarray,
        hu_flat: np.ndarray,
    ) -> None:
        """Check discriminability for each DINOv2 layer independently."""
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < 30:
            return

        valid_hu = hu_flat[valid_idx]
        hu_med = float(np.median(valid_hu))
        hu_mad = float(np.median(np.abs(valid_hu - hu_med)))
        hu_sig = hu_mad * 1.4826
        if hu_sig < 1e-4:
            hu_sig = 1e-4

        susp_local = set(i for i in range(len(valid_idx))
                         if abs(valid_hu[i] - hu_med) > 2.0 * hu_sig)
        norm_local = set(range(len(valid_idx))) - susp_local

        if len(susp_local) < 2 or len(norm_local) < 10:
            return

        rng = np.random.RandomState(0)
        norm_list = list(norm_local)
        susp_list = list(susp_local)

        for li_idx, (layer_idx, toks) in enumerate(zip(LAYER_INDICES, layer_tokens)):
            vt = toks[valid_idx]
            # L2 normalize for cosine
            norms = np.linalg.norm(vt, axis=1, keepdims=True)
            vt = vt / (norms + 1e-6)

            # Sample NN and SN
            nn = [float(vt[a] @ vt[b])
                  for a, b in [rng.choice(norm_list, 2, replace=False) for _ in range(100)]]
            sn = []
            for s in susp_list[:10]:
                for n in rng.choice(norm_list, min(10, len(norm_list)), replace=False):
                    sn.append(float(vt[s] @ vt[n]))

            nn_med = float(np.median(nn))
            sn_med = float(np.median(sn)) if sn else nn_med
            gap = nn_med - sn_med
            verdict = "GOOD" if gap > 0.05 else "WEAK" if gap > 0.02 else "NONE"

            logger.info("DIAG [Layer %d, 768-dim]: NN=%.3f, SN=%.3f, gap=%.4f → %s",
                         layer_idx, nn_med, sn_med, gap, verdict)

    # -- Hemispheric asymmetry --------------------------------------------------

    @staticmethod
    def _compute_asymmetry(hu_grid_flat: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Compute left-right HU asymmetry on the 16^3 grid.

        For each voxel at (z, y, x), compare its HU to the mirror voxel at
        (z, y, GRID_DIM-1-x). Unilateral lesions (hemorrhage) produce high
        asymmetry; bilateral structures (ventricles) produce low asymmetry.

        Returns per-voxel asymmetry scores normalized to [0, 1].
        """
        hu_grid = hu_grid_flat.reshape(GRID_DIM, GRID_DIM, GRID_DIM)
        valid_3d = valid_mask.reshape(GRID_DIM, GRID_DIM, GRID_DIM)

        # Mirror along X axis (left-right)
        hu_mirror = hu_grid[:, :, ::-1].copy()
        valid_mirror = valid_3d[:, :, ::-1].copy()

        # Both voxel and its mirror must be valid
        both_valid = valid_3d & valid_mirror

        # Raw asymmetry = absolute HU difference with contralateral side
        raw_asym = np.abs(hu_grid - hu_mirror)
        raw_asym[~both_valid] = 0.0

        # Normalize by MAD of the valid asymmetries
        valid_vals = raw_asym[both_valid]
        if len(valid_vals) < 10:
            return np.zeros_like(hu_grid_flat)

        med = float(np.median(valid_vals))
        mad = float(np.median(np.abs(valid_vals - med)))
        sigma = mad * 1.4826
        if sigma < 1e-4:
            sigma = 1e-4

        # Z-score of asymmetry, clipped to [0, 1]
        asym_z = np.clip((raw_asym - med) / sigma, 0.0, 5.0) / 5.0
        asym_z[~both_valid] = 0.0

        n_asym = int((asym_z > 0.2).sum())
        logger.info("Asymmetry: median=%.1f HU, σ=%.1f, %d voxels with asym>0.2",
                     med, sigma, n_asym)

        return asym_z.reshape(-1).astype(np.float32)

    # -- SCGAD scoring ----------------------------------------------------------

    def _scgad_scoring(
        self,
        tokens: np.ndarray,
        valid_mask: np.ndarray,
        hu_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Spatially-Coherent Graph Anomaly Detection scoring.

        Stages C-E of the SCGAD pipeline:
        1. Build 26-connectivity graph with cosine similarity edges
        2. HU-modulated per-edge adaptive threshold
        3. Multi-scale: for each threshold quantile, decompose + score
        4. Average scores across scales for consensus

        Args:
            tokens: (4096, 768) float32 L2-normalized
            valid_mask: (4096,) bool
            hu_values: (4096,) float32 mean HU per grid voxel (optional)

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

        # Extract valid HU values if provided
        valid_hu = hu_values[valid_idx] if hu_values is not None else None

        # Stage C: Build spatial adjacency + compute similarities
        edges, sims, delta_hus = self._compute_neighbor_similarities(
            valid_tokens, coords, valid_hu
        )
        n_edges = len(sims)

        if n_edges == 0:
            logger.warning("No valid neighbor pairs -- skipping SCGAD")
            return scores

        sims_array = np.array(sims, dtype=np.float32)
        dhu_array = np.array(delta_hus, dtype=np.float32) if delta_hus is not None else None

        logger.info("SCGAD graph: %d nodes, %d edges, sim [%.3f, %.3f], median=%.3f",
                     n_valid, n_edges,
                     sims_array.min(), sims_array.max(), np.median(sims_array))
        if dhu_array is not None:
            logger.info("  HU deltas: [%.1f, %.1f], median=%.1f",
                         dhu_array.min(), dhu_array.max(), np.median(dhu_array))

        # Locally-calibrated HU discordance
        if dhu_array is not None:
            # Per-voxel local variability = median |delta_hu| with its neighbors
            voxel_deltas: dict[int, list[float]] = defaultdict(list)
            for (i, j), dhu in zip(edges, delta_hus):
                voxel_deltas[i].append(dhu)
                voxel_deltas[j].append(dhu)

            local_var = np.zeros(n_valid, dtype=np.float32)
            for i in range(n_valid):
                if voxel_deltas[i]:
                    local_var[i] = float(np.median(voxel_deltas[i]))

            # Per-edge discordance normalized by local context
            expected = np.array(
                [(local_var[i] + local_var[j]) / 2.0 for i, j in edges],
                dtype=np.float32,
            )
            hu_discordance = np.minimum(
                dhu_array / (expected + HU_DISCORDANCE_EPS), HU_DISCORDANCE_MAX
            )

            logger.info("  HU discordance (local): [%.2f, %.2f], median=%.2f",
                         hu_discordance.min(), hu_discordance.max(),
                         np.median(hu_discordance))
        else:
            hu_discordance = None

        # DIAGNOSTIC: Edge-level analysis — categorize by HU group
        if valid_hu is not None:
            hu_med = float(np.median(valid_hu))
            hu_mad_val = float(np.median(np.abs(valid_hu - hu_med)))
            hu_sig_val = hu_mad_val * 1.4826
            if hu_sig_val < 1e-4:
                hu_sig_val = 1e-4
            susp_set = set(i for i in range(n_valid) if abs(valid_hu[i] - hu_med) > 2.0 * hu_sig_val)

            if susp_set:
                nn_e, sn_e, ss_e = [], [], []
                for idx, (i, j) in enumerate(edges):
                    si, sj = i in susp_set, j in susp_set
                    if si and sj:
                        ss_e.append(sims_array[idx])
                    elif si or sj:
                        sn_e.append(sims_array[idx])
                    else:
                        nn_e.append(sims_array[idx])

                nn_m = float(np.median(nn_e)) if nn_e else 0.0
                sn_m = float(np.median(sn_e)) if sn_e else 0.0
                ss_m = float(np.median(ss_e)) if ss_e else 0.0
                gap = nn_m - sn_m

                logger.info("DIAG [Graph edges]: %d suspicious voxels, "
                             "NN(%d)=%.3f, SN(%d)=%.3f, SS(%d)=%.3f, gap=%.4f → %s",
                             len(susp_set),
                             len(nn_e), nn_m, len(sn_e), sn_m, len(ss_e), ss_m,
                             gap,
                             "GOOD" if gap > 0.05 else "WEAK" if gap > 0.02 else "NONE")

                # Log suspicious voxel HU values
                susp_hus = [valid_hu[i] for i in susp_set]
                logger.info("DIAG [Suspicious HU]: values=%s",
                             [f"{h:.1f}" for h in sorted(susp_hus, reverse=True)[:20]])

        # Stages D-E: Multi-scale consensus
        scale_scores = []
        for q in THRESHOLD_QUANTILES:
            tau_base = float(np.percentile(sims_array, q))

            # HU-modulated per-edge threshold
            if hu_discordance is not None:
                tau_per_edge = tau_base * (1.0 + LAMBDA_HU * hu_discordance)
            else:
                tau_per_edge = tau_base

            components = self._union_find_components(n_valid, edges, sims_array, tau_per_edge)

            comp_scores = self._score_components(valid_tokens, components, edges, valid_hu)
            scale_scores.append(comp_scores)

            sizes = sorted([len(c) for c in components.values()], reverse=True)
            n_scored = int((comp_scores > 0).sum())
            max_s = float(comp_scores.max())
            logger.info("  P%d: tau_base=%.4f, %d comps (top sizes: %s), %d scored, max=%.4f",
                         q, tau_base, len(components), sizes[:5], n_scored, max_s)

        avg_scores = np.mean(scale_scores, axis=0)
        scores[valid_idx] = avg_scores
        return scores

    @staticmethod
    def _compute_neighbor_similarities(
        tokens: np.ndarray,
        coords: np.ndarray,
        hu_values: np.ndarray | None = None,
    ) -> tuple[list[tuple[int, int]], list[float], list[float] | None]:
        """Build 26-connectivity adjacency and compute cosine similarities.

        Uses spatial hash for O(1) neighbor lookup. Cosine sim = dot product
        since tokens are L2-normalized.

        Returns:
            (edges, similarities, delta_hus) — delta_hus is None if hu_values is None
        """
        n = len(tokens)

        spatial_hash: dict[tuple[int, int, int], int] = {}
        for i in range(n):
            spatial_hash[(int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]))] = i

        edges: list[tuple[int, int]] = []
        similarities: list[float] = []
        delta_hus: list[float] | None = [] if hu_values is not None else None

        for i in range(n):
            z, y, x = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
            for dz, dy, dx in _NEIGHBOR_OFFSETS:
                j = spatial_hash.get((z + dz, y + dy, x + dx))
                if j is not None:
                    sim = float(tokens[i] @ tokens[j])
                    edges.append((i, j))
                    similarities.append(sim)
                    if delta_hus is not None:
                        delta_hus.append(abs(float(hu_values[i]) - float(hu_values[j])))

        return edges, similarities, delta_hus

    @staticmethod
    def _union_find_components(
        n: int,
        edges: list[tuple[int, int]],
        sims: np.ndarray,
        tau: np.ndarray | float,
    ) -> dict[int, list[int]]:
        """Union-Find with path halving. Connect edges where sim >= tau.

        tau can be a scalar float (uniform threshold) or a per-edge ndarray
        (HU-modulated adaptive threshold).
        """
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

        per_edge = isinstance(tau, np.ndarray)
        for idx in range(len(edges)):
            threshold = tau[idx] if per_edge else tau
            if sims[idx] >= threshold:
                union(edges[idx][0], edges[idx][1])

        components: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            components[find(i)].append(i)

        return dict(components)

    @staticmethod
    def _score_components(
        tokens: np.ndarray,
        components: dict[int, list[int]],
        edges: list[tuple[int, int]] | None = None,
        hu_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Score voxels by component contrast to background.

        Uses DIRECTIONAL dual z-scoring:
        - Feature z-score: cosine distance to background centroid
        - HU z-score: only HYPER-dense deviations (hemorrhage direction)
        - CSF (<20 HU): features only — hypo-dense is normal anatomy
        - Hemorrhage (>45 HU): max(feat_z, hu_z) — either signal suffices
        - HU fallback: only flags hyper-dense outliers in background
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

        # Step 2: Background feature statistics
        bg_tokens = tokens[bg_indices]
        bg_centroid = bg_tokens.mean(axis=0)
        bg_norm = np.linalg.norm(bg_centroid)
        if bg_norm < 1e-8:
            return scores
        bg_centroid_n = bg_centroid / bg_norm

        bg_cos_dists = 1.0 - bg_tokens @ bg_centroid_n
        bg_feat_median = float(np.median(bg_cos_dists))
        bg_feat_mad = float(np.median(np.abs(bg_cos_dists - bg_feat_median)))
        bg_feat_sigma = bg_feat_mad * 1.4826
        if bg_feat_sigma < 1e-8:
            bg_feat_sigma = 1e-4

        # Step 2b: Background HU statistics (if available)
        bg_hu_median = 0.0
        bg_hu_sigma = 1e-4
        has_hu = hu_values is not None
        if has_hu:
            bg_hu_vals = hu_values[bg_indices]
            bg_hu_median = float(np.median(bg_hu_vals))
            bg_hu_mad = float(np.median(np.abs(bg_hu_vals - bg_hu_median)))
            bg_hu_sigma = bg_hu_mad * 1.4826
            if bg_hu_sigma < 1e-8:
                bg_hu_sigma = 1e-4

        logger.info("  BG: %d voxels (%d comps, %.0f%%), feat(med=%.4f, σ=%.4f), "
                     "HU(med=%.1f, σ=%.1f)",
                     len(bg_indices), bg_comp_count,
                     len(bg_indices) / total_voxels * 100,
                     bg_feat_median, bg_feat_sigma,
                     bg_hu_median, bg_hu_sigma)

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

            # Feature z-score: cosine distance to background centroid
            d_k = 1.0 - float(comp_centroid_n @ bg_centroid_n)
            feat_z = max(0.0, (d_k - bg_feat_median) / bg_feat_sigma)

            # HU z-score: directional — only HYPER-dense deviations count
            # CSF (5-15 HU) is hypo-dense but NORMAL anatomy → ignore
            # Hemorrhage (50-90 HU) is hyper-dense and pathological → catch
            hu_z = 0.0
            comp_hu_mean = 0.0
            if has_hu:
                comp_hu_mean = float(np.mean(hu_values[comp]))
                hu_delta = comp_hu_mean - bg_hu_median
                if hu_delta > 0:
                    hu_z = hu_delta / bg_hu_sigma
                # hypo-dense: hu_z stays 0 (CSF, edema = normal anatomy)

            # Combined z-score: conditional on HU range
            # Hemorrhage range (>45 HU): HU signal alone is sufficient
            # CSF range (<20 HU): features only — HU deviation is expected
            # Middle range: features + mild HU boost
            if has_hu and comp_hu_mean > 45:
                z_combined = max(feat_z, hu_z)
            elif has_hu and comp_hu_mean < CSF_HU_HIGH:
                z_combined = feat_z
            else:
                z_combined = feat_z + 0.3 * hu_z

            # Size sigmoid
            w_k = 1.0 / (1.0 + np.exp(-SIZE_SIGMOID_ALPHA * (comp_size - SIZE_SIGMOID_GAMMA)))

            s_k = z_combined * w_k

            # Log each non-bg component for diagnostics
            logger.info("    Comp size=%d: feat_d=%.4f feat_z=%.2f, "
                         "HU=%.1f hu_z=%.2f, combined_z=%.2f, w=%.2f, score=%.4f",
                         comp_size, d_k, feat_z,
                         comp_hu_mean, hu_z, z_combined, w_k, s_k)

            if s_k <= 0:
                continue

            n_scored += 1

            # Intra-component refinement
            intra_dists = 1.0 - comp_tokens @ comp_centroid_n
            max_intra = float(intra_dists.max()) + 1e-8
            refinement = 1.0 - (intra_dists / max_intra)

            for local_idx, global_idx in enumerate(comp):
                scores[global_idx] = s_k * float(refinement[local_idx])

        # Step 5: Lesion cluster agglomeration
        if edges is not None:
            scored_set = set(i for i in range(n) if scores[i] > 0)
            if len(scored_set) >= 2:
                scored_edges = [
                    (i, j) for i, j in edges
                    if i in scored_set and j in scored_set
                ]
                if scored_edges:
                    mini_parent = {v: v for v in scored_set}
                    mini_rank = {v: 0 for v in scored_set}

                    def mini_find(x: int) -> int:
                        while mini_parent[x] != x:
                            mini_parent[x] = mini_parent[mini_parent[x]]
                            x = mini_parent[x]
                        return x

                    def mini_union(a: int, b: int) -> None:
                        ra, rb = mini_find(a), mini_find(b)
                        if ra == rb:
                            return
                        if mini_rank[ra] < mini_rank[rb]:
                            ra, rb = rb, ra
                        mini_parent[rb] = ra
                        if mini_rank[ra] == mini_rank[rb]:
                            mini_rank[ra] += 1

                    for i, j in scored_edges:
                        mini_union(i, j)

                    lesion_clusters: dict[int, list[int]] = defaultdict(list)
                    for v in scored_set:
                        lesion_clusters[mini_find(v)].append(v)

                    for cluster in lesion_clusters.values():
                        max_score = max(scores[v] for v in cluster)
                        for v in cluster:
                            scores[v] = max_score

        # Step 6: HU outlier fallback — catches HYPER-DENSE lesions trapped in background
        # Only flags voxels ABOVE background median (hemorrhage direction).
        # CSF/ventricles are hypo-dense but normal → excluded.
        if has_hu and edges is not None:
            bg_hu_vals = hu_values[bg_indices]
            bg_hu_med = float(np.median(bg_hu_vals))
            bg_hu_mad = float(np.median(np.abs(bg_hu_vals - bg_hu_med)))
            bg_hu_sig = bg_hu_mad * 1.4826
            if bg_hu_sig < 1e-4:
                bg_hu_sig = 1e-4

            # Find HYPER-dense HU outliers only (above background median)
            bg_set = set(bg_indices)
            outlier_voxels = set()
            for v in bg_indices:
                if hu_values[v] > bg_hu_med:
                    hz = (hu_values[v] - bg_hu_med) / bg_hu_sig
                    if hz > 2.0:
                        outlier_voxels.add(v)

            if len(outlier_voxels) >= MIN_ROI_GRID_VOXELS:
                # Cluster outliers by spatial adjacency (reuse edges)
                outlier_edges = [
                    (i, j) for i, j in edges
                    if i in outlier_voxels and j in outlier_voxels
                ]
                # Mini union-find on outlier voxels
                ol_parent = {v: v for v in outlier_voxels}
                ol_rank = {v: 0 for v in outlier_voxels}

                def ol_find(x):
                    while ol_parent[x] != x:
                        ol_parent[x] = ol_parent[ol_parent[x]]
                        x = ol_parent[x]
                    return x

                def ol_union(a, b):
                    ra, rb = ol_find(a), ol_find(b)
                    if ra == rb:
                        return
                    if ol_rank[ra] < ol_rank[rb]:
                        ra, rb = rb, ra
                    ol_parent[rb] = ra
                    if ol_rank[ra] == ol_rank[rb]:
                        ol_rank[ra] += 1

                for i, j in outlier_edges:
                    ol_union(i, j)

                ol_clusters: dict[int, list[int]] = defaultdict(list)
                for v in outlier_voxels:
                    ol_clusters[ol_find(v)].append(v)

                n_hu_scored = 0
                for cluster in ol_clusters.values():
                    csize = len(cluster)
                    if csize < MIN_ROI_GRID_VOXELS:
                        continue
                    cluster_hu = float(np.mean(hu_values[cluster]))
                    hz = abs(cluster_hu - bg_hu_med) / bg_hu_sig
                    w = 1.0 / (1.0 + np.exp(-SIZE_SIGMOID_ALPHA * (csize - SIZE_SIGMOID_GAMMA)))
                    s = hz * w
                    if s <= 0:
                        continue
                    n_hu_scored += 1
                    logger.info("    HU-outlier cluster: size=%d, HU=%.1f, z=%.2f, "
                                 "w=%.2f, score=%.4f",
                                 csize, cluster_hu, hz, w, s)
                    for v in cluster:
                        scores[v] = max(scores[v], s)

                if n_hu_scored > 0:
                    logger.info("  HU fallback: %d outlier voxels, %d clusters scored",
                                 len(outlier_voxels), n_hu_scored)

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
