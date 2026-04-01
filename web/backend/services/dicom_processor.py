"""DICOM CT multi-slice processor.

Reads DICOM files (including from ZIP archives), sorts by slice position,
applies MedGemma CT RGB windowing, and produces PIL Images ready for the model.
"""

import io
import logging
import zipfile
from typing import Any

import numpy as np
import PIL.Image
import pydicom

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CT Windowing classes (adapted from the repo's generic_dicom_handler.py)
# ---------------------------------------------------------------------------

class TraditionalWindow:
    """Single-channel CT window: clips HU values and maps to 0-255."""

    def __init__(self, center: int, width: int):
        self.center = center
        self.width = width

    def apply(self, image: np.ndarray) -> np.ndarray:
        half = self.width // 2
        lo = self.center - half
        hi = self.center + half
        return np.round(
            np.interp(image.clip(lo, hi), (lo, hi), (0, 255)), 0
        ).astype(np.uint8)


class RGBWindow:
    """3-channel CT window matching MedGemma 1.5 training.

    Red:   Wide window         (-1024 to 1024 HU)
    Green: Soft tissue window  (135 to 215 HU)
    Blue:  Brain window        (0 to 80 HU)
    """

    def __init__(
        self,
        red: TraditionalWindow,
        green: TraditionalWindow,
        blue: TraditionalWindow,
    ):
        self._r = red
        self._g = green
        self._b = blue

    def apply(self, image: np.ndarray) -> np.ndarray:
        r = self._r.apply(image)[..., np.newaxis]
        g = self._g.apply(image)[..., np.newaxis]
        b = self._b.apply(image)[..., np.newaxis]
        return np.concatenate([r, g, b], axis=-1)


MEDGEMMA_CT_WINDOW = RGBWindow(
    TraditionalWindow(0, 2048),
    TraditionalWindow(175, 80),
    TraditionalWindow(40, 80),
)


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------

# Files to skip inside ZIPs / uploads
_SKIP_NAMES = {"dicomdir", ".ds_store", "thumbs.db", "desktop.ini"}
_SKIP_EXTENSIONS = {
    ".txt", ".json", ".xml", ".html", ".css", ".js", ".md",
    ".csv", ".log", ".py", ".sh", ".bat", ".exe", ".dll",
}


def _should_skip(filename: str) -> bool:
    name = filename.rsplit("/", 1)[-1].lower()
    if name in _SKIP_NAMES or name.startswith("._"):
        return True
    ext = name[name.rfind("."):] if "." in name else ""
    return ext in _SKIP_EXTENSIONS


def _extract_files_from_uploads(
    file_contents: list[tuple[str, bytes]],
) -> list[tuple[str, bytes]]:
    """Flatten uploads: extract ZIPs, skip junk files."""
    result: list[tuple[str, bytes]] = []
    for filename, raw in file_contents:
        if _should_skip(filename):
            continue
        lower = filename.lower()
        if lower.endswith(".zip"):
            # Extract all files from ZIP
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        inner_name = info.filename
                        if _should_skip(inner_name):
                            continue
                        result.append((inner_name, zf.read(info)))
            except zipfile.BadZipFile:
                logger.warning("Skipping invalid ZIP: %s", filename)
        else:
            result.append((filename, raw))
    return result


# ---------------------------------------------------------------------------
# DICOM processing
# ---------------------------------------------------------------------------

def _slice_sort_key(dcm: pydicom.FileDataset) -> float:
    """Return a numeric key for sorting slices by spatial position."""
    try:
        return float(dcm.ImagePositionPatient[2])
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        return float(dcm.InstanceNumber)
    except (AttributeError, ValueError):
        pass
    try:
        return float(dcm.SliceLocation)
    except (AttributeError, ValueError):
        return 0.0


def _rescale_to_hu(pixel_array: np.ndarray, dcm: pydicom.FileDataset) -> np.ndarray:
    """Apply RescaleSlope / RescaleIntercept to convert to Hounsfield Units."""
    has_slope = "RescaleSlope" in dcm
    has_intercept = "RescaleIntercept" in dcm
    if has_slope and has_intercept:
        arr = pixel_array.astype(np.float64) * float(dcm.RescaleSlope)
        arr += float(dcm.RescaleIntercept)
        return arr
    return pixel_array.astype(np.float64)


def _try_read_dicom(filename: str, raw: bytes) -> pydicom.FileDataset | None:
    """Try to parse bytes as DICOM. Returns None on failure (silently skips)."""
    try:
        dcm = pydicom.dcmread(io.BytesIO(raw))
        # Must have pixel data to be useful
        _ = dcm.pixel_array
        return dcm
    except Exception:
        logger.debug("Skipping non-DICOM file: %s", filename)
        return None


class ProcessedSlice:
    """A single processed CT slice."""

    def __init__(self, index: int, position: float, image: PIL.Image.Image, metadata: dict[str, Any]):
        self.index = index
        self.position = position
        self.image = image
        self.metadata = metadata

    def to_jpeg_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


class CTDicomProcessor:
    """Load a set of DICOM CT files, sort, window, and produce PIL images."""

    def __init__(self):
        self.window = MEDGEMMA_CT_WINDOW

    def process_files(self, file_contents: list[tuple[str, bytes]]) -> tuple[list[ProcessedSlice], dict[str, Any]]:
        """Process uploaded file contents.

        Handles: raw DICOM files (any extension), ZIP archives containing DICOMs,
        folders with mixed content. Non-DICOM files are silently skipped.

        Args:
            file_contents: List of (filename, raw_bytes) tuples.

        Returns:
            (slices, series_metadata) – sorted processed slices and series-level metadata.

        Raises:
            ValueError: If no valid DICOM files are found.
        """
        # Flatten ZIPs and filter junk
        all_files = _extract_files_from_uploads(file_contents)

        # Try to parse each file as DICOM (skip failures silently)
        datasets: list[tuple[str, pydicom.FileDataset]] = []
        for filename, raw in all_files:
            dcm = _try_read_dicom(filename, raw)
            if dcm is not None:
                datasets.append((filename, dcm))

        if not datasets:
            raise ValueError(
                "No valid DICOM files found. "
                "Please upload .dcm files, a folder of DICOMs, or a ZIP archive."
            )

        logger.info("Found %d valid DICOM slices out of %d files", len(datasets), len(all_files))

        # Sort by slice position
        datasets.sort(key=lambda t: _slice_sort_key(t[1]))

        # Extract series-level metadata from first dataset
        _, first_dcm = datasets[0]
        series_meta = self._extract_series_metadata(first_dcm)

        # Process each slice
        slices: list[ProcessedSlice] = []
        for idx, (filename, dcm) in enumerate(datasets):
            pixel_array = dcm.pixel_array
            # Handle multi-frame: take each frame as separate slice
            if pixel_array.ndim == 3 and getattr(dcm, "SamplesPerPixel", 1) == 1:
                for frame_idx in range(pixel_array.shape[0]):
                    frame = pixel_array[frame_idx]
                    pslice = self._process_single_slice(frame, dcm, len(slices))
                    slices.append(pslice)
            else:
                if pixel_array.ndim == 3 and pixel_array.shape[-1] == 1:
                    pixel_array = pixel_array.squeeze(-1)
                pslice = self._process_single_slice(pixel_array, dcm, idx)
                slices.append(pslice)

        return slices, series_meta

    def _process_single_slice(
        self, pixel_array: np.ndarray, dcm: pydicom.FileDataset, index: int
    ) -> ProcessedSlice:
        hu = _rescale_to_hu(pixel_array, dcm)
        windowed = self.window.apply(hu)  # (H, W, 3) uint8
        img = PIL.Image.fromarray(windowed, mode="RGB")
        position = _slice_sort_key(dcm)
        return ProcessedSlice(
            index=index,
            position=position,
            image=img,
            metadata={
                "instance_number": getattr(dcm, "InstanceNumber", None),
                "slice_location": getattr(dcm, "SliceLocation", None),
            },
        )

    def _extract_series_metadata(self, dcm: pydicom.FileDataset) -> dict[str, Any]:
        def _get(attr: str) -> Any:
            val = getattr(dcm, attr, None)
            if val is not None:
                return str(val)
            return None

        return {
            "patient_id": _get("PatientID"),
            "patient_name": _get("PatientName"),
            "study_description": _get("StudyDescription"),
            "series_description": _get("SeriesDescription"),
            "modality": _get("Modality"),
            "slice_thickness": _get("SliceThickness"),
            "study_date": _get("StudyDate"),
            "rows": getattr(dcm, "Rows", None),
            "columns": getattr(dcm, "Columns", None),
        }
