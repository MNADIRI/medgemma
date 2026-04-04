"""Quantitative ROI analysis from MedSAM2 segmentation mask + DICOM HU data.

Extracts structured radiological measurements for chain-of-thought prompting:
  - Bloc 1: Lesion HU densitometry (eroded mask — mean, median, SD, min, max, 9 deciles, asymmetry)
  - Bloc 2: Peri-lesional ring densitometry + delta HU (median-based)
  - Bloc 3: 2D morphometry in mm (original mask — axes, area, perimeter, compactness, solidity)
  - Bloc 4: Spatial localization from DICOM patient coordinates (laterality, antero-posterior)
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Erosion ────────────────────────────────────────────────────────────────

def _erode_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Erode mask with a disk structuring element of given pixel radius.

    Falls back to the original mask if the eroded mask has < 20 pixels.
    """
    from scipy.ndimage import binary_erosion

    # Build disk structuring element
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    struct = (x ** 2 + y ** 2) <= radius ** 2

    eroded = binary_erosion(mask, structure=struct)
    if np.sum(eroded) < 20:
        logger.info("Eroded mask too small (%d px), keeping original mask", np.sum(eroded))
        return mask.copy()
    return eroded


# ── Main extraction ────────────────────────────────────────────────────────

def extract_roi_data(
    hu_array: np.ndarray,
    mask: np.ndarray,
    pixel_spacing: tuple[float, float],
    ring_width_mm: float = 10.0,
    image_position_patient: tuple[float, float, float] | None = None,
    image_orientation_patient: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Extract quantitative data from a segmented ROI.

    Args:
        hu_array: Raw Hounsfield Unit values, shape (H, W), float64.
        mask: Binary segmentation mask, shape (H, W), bool.
        pixel_spacing: (row_spacing_mm, col_spacing_mm) from DICOM PixelSpacing.
        ring_width_mm: Width of the peri-lesional annular ring in mm.
        image_position_patient: DICOM ImagePositionPatient (x, y, z) in mm. None if absent.
        image_orientation_patient: DICOM ImageOrientationPatient (6 direction cosines). None if absent.

    Returns:
        Dictionary with blocks: 'density', 'peri_lesional', 'morphometry', 'spatial'.
    """
    row_sp, col_sp = pixel_spacing

    original_mask = mask.astype(bool)
    original_area_px = int(np.sum(original_mask))

    if original_area_px == 0:
        logger.warning("Empty mask — no lesion pixels")
        return _empty_result()

    # ── Erode mask (2px disk) for density stats ────────────────────────
    eroded_mask = _erode_mask(original_mask, radius=2)
    eroded_area_px = int(np.sum(eroded_mask))
    eroded_area_fraction = eroded_area_px / original_area_px if original_area_px > 0 else 0.0

    # ── Bloc 1: Lesion HU densitometry (eroded mask) ──────────────────
    lesion_hu = hu_array[eroded_mask]
    if lesion_hu.size == 0:
        logger.warning("Eroded mask empty — no lesion pixels for density")
        return _empty_result()

    mean_hu = float(np.mean(lesion_hu))
    median_hu = float(np.median(lesion_hu))

    density = {
        "mean": mean_hu,
        "median": median_hu,
        "sd": float(np.std(lesion_hu)),
        "min": float(np.min(lesion_hu)),
        "max": float(np.max(lesion_hu)),
        "deciles": [float(np.percentile(lesion_hu, p)) for p in range(10, 100, 10)],
        "density_asymmetry": mean_hu - median_hu,
    }

    # ── Bloc 2: Peri-lesional ring (dilate original mask XOR original) ─
    peri_lesional = _compute_peri_lesional(
        hu_array, original_mask, row_sp, col_sp, ring_width_mm, median_hu
    )

    # ── Bloc 3: 2D morphometry (original mask) ────────────────────────
    morphometry = _compute_morphometry(original_mask, row_sp, col_sp)
    morphometry["eroded_area_fraction"] = float(eroded_area_fraction)

    # ── Bloc 4: Spatial localization ──────────────────────────────────
    spatial = _compute_spatial_localization(
        original_mask, pixel_spacing,
        image_position_patient, image_orientation_patient,
    )

    return {
        "density": density,
        "peri_lesional": peri_lesional,
        "morphometry": morphometry,
        "spatial": spatial,
    }


def _compute_peri_lesional(
    hu_array: np.ndarray,
    original_mask: np.ndarray,
    row_sp: float,
    col_sp: float,
    ring_width_mm: float,
    lesion_median_hu: float,
) -> dict[str, Any]:
    """Compute peri-lesional ring HU stats and delta (median-based)."""
    from scipy.ndimage import binary_dilation

    # Dilation radius in pixels (anisotropic spacing)
    radius_row = max(1, int(round(ring_width_mm / row_sp)))
    radius_col = max(1, int(round(ring_width_mm / col_sp)))

    # Create elliptical structuring element
    y, x = np.ogrid[-radius_row:radius_row + 1, -radius_col:radius_col + 1]
    struct = ((y / radius_row) ** 2 + (x / radius_col) ** 2) <= 1.0

    dilated = binary_dilation(original_mask, structure=struct)
    ring = dilated & ~original_mask  # XOR: dilated minus original

    ring_hu = hu_array[ring]
    if ring_hu.size == 0:
        return {"mean": None, "sd": None, "delta_hu_median_parenchyma": None}

    ring_mean = float(np.mean(ring_hu))
    ring_sd = float(np.std(ring_hu))
    delta_hu = lesion_median_hu - ring_mean  # median-based delta

    return {
        "mean": ring_mean,
        "sd": ring_sd,
        "delta_hu_median_parenchyma": delta_hu,
    }


def _compute_morphometry(
    mask: np.ndarray,
    row_sp: float,
    col_sp: float,
) -> dict[str, Any]:
    """Compute 2D morphometric descriptors in mm (on original mask)."""
    # Area in mm²
    pixel_area_mm2 = row_sp * col_sp
    area_px = int(np.sum(mask))
    area_mm2 = area_px * pixel_area_mm2

    if area_px < 3:
        return _empty_morphometry()

    # Find contour points for perimeter
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    boundary_coords = np.argwhere(boundary)  # (N, 2) in (row, col)

    if boundary_coords.shape[0] < 3:
        return _empty_morphometry()

    # Perimeter: boundary pixel count × average spacing
    perimeter_mm = float(boundary_coords.shape[0]) * ((row_sp + col_sp) / 2.0)

    # Major/minor axis via PCA on all lesion pixel coordinates
    lesion_coords = np.argwhere(mask).astype(np.float64)  # (N, 2)
    lesion_coords[:, 0] *= row_sp
    lesion_coords[:, 1] *= col_sp

    centroid = lesion_coords.mean(axis=0)
    centered = lesion_coords - centroid
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending

    # Axis lengths: 4 * sqrt(eigenvalue) gives approximate extent
    major_axis = 4.0 * np.sqrt(max(eigenvalues[0], 0))
    minor_axis = 4.0 * np.sqrt(max(eigenvalues[1], 0)) if len(eigenvalues) > 1 else major_axis

    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0

    # Compactness (isoperimetric quotient): 4π × area / perimeter²
    compactness = (4.0 * np.pi * area_mm2) / (perimeter_mm ** 2) if perimeter_mm > 0 else 0.0
    compactness = min(compactness, 1.0)

    # Solidity: area / convex hull area
    solidity = _compute_solidity(mask, pixel_area_mm2, area_mm2)

    return {
        "major_axis_mm": float(major_axis),
        "minor_axis_mm": float(minor_axis),
        "aspect_ratio": float(aspect_ratio),
        "area_mm2": float(area_mm2),
        "perimeter_mm": float(perimeter_mm),
        "compactness": float(compactness),
        "solidity": float(solidity),
    }


# ── Spatial localization ──────────────────────────────────────────────────

def _compute_spatial_localization(
    mask: np.ndarray,
    pixel_spacing: tuple[float, float],
    image_position_patient: tuple[float, float, float] | None,
    image_orientation_patient: tuple[float, ...] | None,
) -> dict[str, Any]:
    """Compute patient-space localization from DICOM spatial metadata.

    Converts mask centroid (row, col) to patient coordinates (x, y, z) using:
      patient_xyz = ImagePositionPatient
                    + col * PixelSpacing[1] * row_direction_cosines
                    + row * PixelSpacing[0] * col_direction_cosines

    DICOM LPS convention:
      X+ = left, X- = right
      Y+ = posterior, Y- = anterior

    Laterality (on X): > +5mm = "left", < -5mm = "right", else "midline"
    Depth (on Y): > +5mm = "posterior", < -5mm = "anterior", else "central"
    """
    if image_position_patient is None or image_orientation_patient is None:
        return {"laterality": "unknown", "antero_posterior": "unknown", "patient_xyz_mm": None}

    if len(image_orientation_patient) < 6:
        return {"laterality": "unknown", "antero_posterior": "unknown", "patient_xyz_mm": None}

    row_sp, col_sp = pixel_spacing

    # Mask centroid in pixel coordinates
    coords = np.argwhere(mask)  # (N, 2) → (row, col)
    centroid_row = float(np.mean(coords[:, 0]))
    centroid_col = float(np.mean(coords[:, 1]))

    # Direction cosines
    row_cosines = np.array(image_orientation_patient[:3], dtype=np.float64)
    col_cosines = np.array(image_orientation_patient[3:6], dtype=np.float64)

    # Image position patient (origin)
    origin = np.array(image_position_patient, dtype=np.float64)

    # DICOM formula: patient = origin + col * col_spacing * row_cosines + row * row_spacing * col_cosines
    patient_xyz = (
        origin
        + centroid_col * col_sp * row_cosines
        + centroid_row * row_sp * col_cosines
    )

    x, y, _z = patient_xyz

    # Classify laterality (LPS: X+ = left)
    if x > 5.0:
        laterality = "left"
    elif x < -5.0:
        laterality = "right"
    else:
        laterality = "midline"

    # Classify antero-posterior (LPS: Y+ = posterior)
    if y > 5.0:
        antero_posterior = "posterior"
    elif y < -5.0:
        antero_posterior = "anterior"
    else:
        antero_posterior = "central"

    return {
        "laterality": laterality,
        "antero_posterior": antero_posterior,
        "patient_xyz_mm": [float(patient_xyz[0]), float(patient_xyz[1]), float(patient_xyz[2])],
    }


# ── Solidity ──────────────────────────────────────────────────────────────

def _compute_solidity(
    mask: np.ndarray,
    pixel_area_mm2: float,
    area_mm2: float,
) -> float:
    """Compute solidity = area / convex_hull_area."""
    try:
        from scipy.spatial import ConvexHull
        coords = np.argwhere(mask).astype(np.float64)
        if coords.shape[0] < 3:
            return 1.0
        hull = ConvexHull(coords)
        hull_area_mm2 = hull.volume * pixel_area_mm2  # In 2D, hull.volume = area
        return area_mm2 / hull_area_mm2 if hull_area_mm2 > 0 else 1.0
    except Exception:
        return 1.0


def _empty_morphometry() -> dict[str, Any]:
    return {
        "major_axis_mm": 0.0,
        "minor_axis_mm": 0.0,
        "aspect_ratio": 1.0,
        "area_mm2": 0.0,
        "perimeter_mm": 0.0,
        "compactness": 0.0,
        "solidity": 0.0,
        "eroded_area_fraction": 0.0,
    }


def _empty_result() -> dict[str, Any]:
    return {
        "density": {
            "mean": 0, "median": 0, "sd": 0, "min": 0, "max": 0,
            "deciles": [0.0] * 9,
            "density_asymmetry": 0.0,
        },
        "peri_lesional": {"mean": None, "sd": None, "delta_hu_median_parenchyma": None},
        "morphometry": _empty_morphometry(),
        "spatial": {"laterality": "unknown", "antero_posterior": "unknown", "patient_xyz_mm": None},
    }


# ── Formatting ──────────────────────────────────────────────────────────


def format_roi_data(roi_data: dict[str, Any]) -> str:
    """Format ROI data as a structured <ROI_DATA> text block.

    Floats are rounded to 1 decimal place. One line per logical group.
    """
    d = roi_data["density"]
    p = roi_data["peri_lesional"]
    m = roi_data["morphometry"]
    s = roi_data["spatial"]

    def r(v, decimals=1):
        """Round a value, handling None."""
        if v is None:
            return "N/A"
        return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)

    lines = ["<ROI_DATA>"]

    # Density block (eroded mask)
    lines.append(
        f"density_hu: mean={r(d['mean'])} | median={r(d['median'])} "
        f"| sd={r(d['sd'])} | min={r(d['min'], 0)} | max={r(d['max'], 0)}"
    )

    # 9 deciles on one line
    deciles = d.get("deciles", [])
    decile_strs = [f"P{p}={r(v)}" for p, v in zip(range(10, 100, 10), deciles)]
    lines.append(f"density_deciles: {' | '.join(decile_strs)}")

    # Density asymmetry
    lines.append(f"density_asymmetry: {r(d.get('density_asymmetry', 0.0))}")

    # Peri-lesional block
    lines.append(
        f"parenchyma_adjacent_hu: mean={r(p['mean'])} | sd={r(p['sd'])}"
    )
    delta = p.get("delta_hu_median_parenchyma")
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        lines.append(f"delta_hu_median_parenchyma: {sign}{r(delta)}")
    else:
        lines.append("delta_hu_median_parenchyma: N/A")

    # Morphometry block
    lines.append("")
    lines.append(
        f"long_axis_mm: {r(m['major_axis_mm'])} | short_axis_mm: {r(m['minor_axis_mm'])} "
        f"| aspect_ratio: {r(m['aspect_ratio'], 2)}"
    )
    lines.append(
        f"area_mm2: {r(m['area_mm2'])} | perimeter_mm: {r(m['perimeter_mm'])}"
    )
    lines.append(
        f"compactness: {r(m['compactness'], 2)} | solidity: {r(m['solidity'], 2)}"
    )
    lines.append(f"eroded_area_fraction: {r(m.get('eroded_area_fraction', 0.0), 2)}")

    # Spatial localization
    lines.append("")
    lines.append(f"laterality: {s.get('laterality', 'unknown')}")
    lines.append(f"antero_posterior: {s.get('antero_posterior', 'unknown')}")

    lines.append("</ROI_DATA>")
    return "\n".join(lines)
