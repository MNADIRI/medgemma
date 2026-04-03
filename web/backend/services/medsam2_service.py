"""MedSAM2 segmentation service — single-slice lesion segmentation with bbox prompt.

Uses the MedSAM2 video predictor in single-frame mode: the radiologist's
bounding box is the prompt, and we get a binary mask of the lesion.

The segmented lesion is then composited as an isolated image (lesion pixels
on black background) for MedGemma to analyze alongside the full slice.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image

logger = logging.getLogger(__name__)

MEDSAM2_CHECKPOINT = os.environ.get("MEDSAM2_CHECKPOINT", "MedSAM2_CTLesion.pt")
MEDSAM2_CONFIG = os.environ.get("MEDSAM2_CONFIG", "configs/sam2.1_hiera_t512.yaml")
MEDSAM2_IMAGE_SIZE = 512

# ImageNet normalization constants used by SAM2
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MedSAM2Service:
    """Loads MedSAM2 and runs single-slice segmentation with bounding box prompts."""

    def __init__(self) -> None:
        self.predictor = None
        self._device = None

    @property
    def is_loaded(self) -> bool:
        return self.predictor is not None

    def load_model(self) -> None:
        """Load MedSAM2 model. Called once at startup."""
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor_npz
        except ImportError:
            logger.warning(
                "MedSAM2 not installed — segmentation disabled. "
                "Install with: pip install -e '.[dev]' from the MedSAM2 repo"
            )
            return

        if not torch.cuda.is_available():
            logger.warning("MedSAM2 requires CUDA — segmentation disabled on CPU")
            return

        # Resolve checkpoint path
        checkpoint = self._resolve_checkpoint()
        if checkpoint is None:
            logger.warning("MedSAM2 checkpoint not found — segmentation disabled")
            return

        # Resolve config path
        config = self._resolve_config()
        if config is None:
            logger.warning("MedSAM2 config not found — segmentation disabled")
            return

        try:
            logger.info("Loading MedSAM2 from %s …", checkpoint)
            self.predictor = build_sam2_video_predictor_npz(config, checkpoint)
            self._device = "cuda"
            logger.info("MedSAM2 loaded on CUDA")
        except Exception as exc:
            logger.error("Failed to load MedSAM2: %s", exc)
            self.predictor = None

    def _resolve_checkpoint(self) -> str | None:
        """Find the MedSAM2 checkpoint file."""
        candidates = [
            MEDSAM2_CHECKPOINT,
            f"checkpoints/{MEDSAM2_CHECKPOINT}",
            f"/content/MedSAM2/checkpoints/{MEDSAM2_CHECKPOINT}",
            os.path.expanduser(f"~/.cache/medsam2/{MEDSAM2_CHECKPOINT}"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path

        # Try downloading from HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            logger.info("Downloading MedSAM2 checkpoint from HuggingFace…")
            return hf_hub_download(
                repo_id="wanglab/MedSAM2",
                filename=MEDSAM2_CHECKPOINT,
            )
        except Exception as exc:
            logger.warning("Could not download MedSAM2 checkpoint: %s", exc)
            return None

    def _resolve_config(self) -> str | None:
        """Find the MedSAM2 config YAML file."""
        candidates = [
            MEDSAM2_CONFIG,
            f"/content/MedSAM2/{MEDSAM2_CONFIG}",
        ]
        # Check relative to the sam2 package location (editable install)
        try:
            import sam2
            pkg_dir = Path(sam2.__file__).parent
            # sam2/__init__.py → sam2/ → parent has configs/
            candidates.append(str(pkg_dir.parent / MEDSAM2_CONFIG))
            # Also check inside the sam2 package directory itself
            candidates.append(str(pkg_dir / MEDSAM2_CONFIG))
            # The config name without the configs/ prefix, in case it's flattened
            config_basename = Path(MEDSAM2_CONFIG).name
            candidates.append(str(pkg_dir / "configs" / config_basename))
            candidates.append(str(pkg_dir.parent / "configs" / config_basename))
        except (ImportError, AttributeError):
            pass

        for path in candidates:
            logger.debug("Checking config path: %s (exists=%s)", path, os.path.isfile(path))
            if os.path.isfile(path):
                return path

        # Log all tried paths for debugging
        logger.warning("MedSAM2 config not found in any of: %s", candidates)
        return None

    def segment_slice(
        self,
        image: PIL.Image.Image,
        roi: dict[str, float],
    ) -> np.ndarray:
        """Segment a lesion on a single CT slice using a bounding box prompt.

        Args:
            image: The display image (grayscale PIL, mode "L") or model image (RGB).
            roi: Normalized ROI {x, y, width, height} in [0, 1].

        Returns:
            Binary mask as numpy bool array of shape (H_orig, W_orig).
        """
        import torch

        if not self.is_loaded:
            raise RuntimeError("MedSAM2 not loaded")

        # Get original dimensions
        w_orig, h_orig = image.size

        # Convert to RGB if grayscale
        if image.mode != "RGB":
            image_rgb = image.convert("RGB")
        else:
            image_rgb = image

        # Resize to 512x512
        img_resized = image_rgb.resize(
            (MEDSAM2_IMAGE_SIZE, MEDSAM2_IMAGE_SIZE), PIL.Image.BILINEAR
        )

        # To numpy (H, W, 3) → (3, H, W), normalize
        img_array = np.array(img_resized, dtype=np.float32)  # (512, 512, 3)
        img_array /= 255.0
        img_array = (img_array - _IMAGENET_MEAN) / _IMAGENET_STD
        img_array = img_array.transpose(2, 0, 1)  # (3, 512, 512)

        # To tensor: (1, 3, 512, 512)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).cuda()

        # Bounding box in original image coordinates
        bbox = np.array([
            roi["x"] * w_orig,
            roi["y"] * h_orig,
            (roi["x"] + roi["width"]) * w_orig,
            (roi["y"] + roi["height"]) * h_orig,
        ])

        # Run inference
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(
                img_tensor,
                video_height=h_orig,
                video_width=w_orig,
            )

            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=bbox,
            )

            # Binary mask at original resolution
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]  # (H_orig, W_orig) bool

            self.predictor.reset_state(inference_state)

        return mask

    def segment_and_isolate(
        self,
        model_image: PIL.Image.Image,
        display_image: PIL.Image.Image,
        roi: dict[str, float],
    ) -> tuple[np.ndarray, PIL.Image.Image]:
        """Segment and produce an isolated lesion image for MedGemma.

        Args:
            model_image: RGB model image (MedGemma windowed) for compositing.
            display_image: Grayscale display image for MedSAM2 input.
            roi: Normalized ROI coordinates.

        Returns:
            (mask, isolated_image):
              - mask: bool array (H, W) of the segmentation
              - isolated_image: RGB PIL image of lesion pixels on black,
                cropped to the ROI bounding box
        """
        # Segment using the display image (single-channel, cleaner for SAM)
        mask = self.segment_slice(display_image, roi)

        # Apply mask to the model image (RGB windowed) to get isolated lesion
        model_array = np.array(model_image)  # (H, W, 3)
        isolated = model_array.copy()
        isolated[~mask] = 0  # black background outside lesion

        # Crop to ROI bounding box to focus on the lesion area
        w, h = model_image.size
        left = max(0, int(roi["x"] * w))
        top = max(0, int(roi["y"] * h))
        right = min(w, int((roi["x"] + roi["width"]) * w))
        bottom = min(h, int((roi["y"] + roi["height"]) * h))

        isolated_crop = isolated[top:bottom, left:right]
        isolated_pil = PIL.Image.fromarray(isolated_crop, mode="RGB")

        return mask, isolated_pil

    def mask_to_png_bytes(self, mask: np.ndarray) -> bytes:
        """Encode a binary mask as a PNG image (white=lesion, transparent=background)."""
        import io

        h, w = mask.shape
        # RGBA: white lesion with alpha, transparent background
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask, 0] = 255   # R
        rgba[mask, 1] = 100   # G (orange-ish tint)
        rgba[mask, 2] = 0     # B
        rgba[mask, 3] = 128   # semi-transparent

        img = PIL.Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
