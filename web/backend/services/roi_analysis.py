"""Quantitative ROI analysis from MedSAM2 segmentation mask + DICOM HU data.

Extracts structured radiological measurements for chain-of-thought prompting:
  - Bloc 1: Lesion HU densitometry (mean, median, SD, min, max, P10, P90)
  - Bloc 2: Peri-lesional ring densitometry + delta HU
  - Bloc 3: 2D morphometry in mm (axes, area, perimeter, compactness, solidity)
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def extract_roi_data(
    hu_array: np.ndarray,
    mask: np.ndarray,
    pixel_spacing: tuple[float, float],
    ring_width_mm: float = 10.0,
) -> dict[str, Any]:
    """Extract quantitative data from a segmented ROI.

    Args:
        hu_array: Raw Hounsfield Unit values, shape (H, W), float64.
        mask: Binary segmentation mask, shape (H, W), bool.
        pixel_spacing: (row_spacing_mm, col_spacing_mm) from DICOM PixelSpacing.
        ring_width_mm: Width of the peri-lesional annular ring in mm.

    Returns:
        Dictionary with three blocks: 'density', 'peri_lesional', 'morphometry'.
    """
    row_sp, col_sp = pixel_spacing

    # ── Bloc 1: Lesion HU densitometry ──────────────────────────────────
    lesion_hu = hu_array[mask]
    if lesion_hu.size == 0:
        logger.warning("Empty mask — no lesion pixels")
        return _empty_result()

    density = {
        "mean": float(np.mean(lesion_hu)),
        "median": float(np.median(lesion_hu)),
        "sd": float(np.std(lesion_hu)),
        "min": float(np.min(lesion_hu)),
        "max": float(np.max(lesion_hu)),
        "p10": float(np.percentile(lesion_hu, 10)),
        "p90": float(np.percentile(lesion_hu, 90)),
    }

    # ── Bloc 2: Peri-lesional ring ──────────────────────────────────────
    peri_lesional = _compute_peri_lesional(
        hu_array, mask, row_sp, col_sp, ring_width_mm, density["mean"]
    )

    # ── Bloc 3: 2D morphometry ──────────────────────────────────────────
    morphometry = _compute_morphometry(mask, row_sp, col_sp)

    return {
        "density": density,
        "peri_lesional": peri_lesional,
        "morphometry": morphometry,
    }


def _compute_peri_lesional(
    hu_array: np.ndarray,
    mask: np.ndarray,
    row_sp: float,
    col_sp: float,
    ring_width_mm: float,
    lesion_mean_hu: float,
) -> dict[str, Any]:
    """Compute peri-lesional ring HU stats and delta."""
    from scipy.ndimage import binary_dilation

    # Dilation radius in pixels (anisotropic spacing)
    radius_row = max(1, int(round(ring_width_mm / row_sp)))
    radius_col = max(1, int(round(ring_width_mm / col_sp)))

    # Create elliptical structuring element
    y, x = np.ogrid[-radius_row:radius_row + 1, -radius_col:radius_col + 1]
    struct = ((y / radius_row) ** 2 + (x / radius_col) ** 2) <= 1.0

    dilated = binary_dilation(mask, structure=struct)
    ring = dilated & ~mask

    ring_hu = hu_array[ring]
    if ring_hu.size == 0:
        return {"mean": None, "sd": None, "delta_hu": None}

    ring_mean = float(np.mean(ring_hu))
    ring_sd = float(np.std(ring_hu))
    delta_hu = lesion_mean_hu - ring_mean

    return {
        "mean": ring_mean,
        "sd": ring_sd,
        "delta_hu": delta_hu,
    }


def _compute_morphometry(
    mask: np.ndarray,
    row_sp: float,
    col_sp: float,
) -> dict[str, Any]:
    """Compute 2D morphometric descriptors in mm."""
    # Area in mm²
    pixel_area_mm2 = row_sp * col_sp
    area_px = int(np.sum(mask))
    area_mm2 = area_px * pixel_area_mm2

    if area_px < 3:
        return _empty_morphometry()

    # Find contour points for perimeter and axis calculations
    # Use the mask boundary pixels
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    boundary_coords = np.argwhere(boundary)  # (N, 2) in (row, col)

    if boundary_coords.shape[0] < 3:
        return _empty_morphometry()

    # Perimeter: sum of distances between consecutive boundary pixels (in mm)
    # Use the actual boundary length via pixel count × spacing
    perimeter_mm = float(boundary_coords.shape[0]) * ((row_sp + col_sp) / 2.0)

    # Major/minor axis via PCA on all lesion pixel coordinates
    lesion_coords = np.argwhere(mask).astype(np.float64)  # (N, 2)
    # Convert to mm
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
    # Perfect circle = 1.0
    compactness = (4.0 * np.pi * area_mm2) / (perimeter_mm ** 2) if perimeter_mm > 0 else 0.0
    compactness = min(compactness, 1.0)  # clamp numerical artifacts

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
    }


def _empty_result() -> dict[str, Any]:
    return {
        "density": {
            "mean": 0, "median": 0, "sd": 0, "min": 0, "max": 0, "p10": 0, "p90": 0,
        },
        "peri_lesional": {"mean": None, "sd": None, "delta_hu": None},
        "morphometry": _empty_morphometry(),
    }


# ── Formatting ──────────────────────────────────────────────────────────


def format_roi_data(roi_data: dict[str, Any]) -> str:
    """Format ROI data as a structured <ROI_DATA> text block.

    Floats are rounded to 1 decimal place. One line per logical group.
    """
    d = roi_data["density"]
    p = roi_data["peri_lesional"]
    m = roi_data["morphometry"]

    def r(v, decimals=1):
        """Round a value, handling None."""
        if v is None:
            return "N/A"
        return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)

    lines = ["<ROI_DATA>"]

    # Density block
    lines.append(
        f"density_hu: mean={r(d['mean'])} | median={r(d['median'])} "
        f"| sd={r(d['sd'])} | min={r(d['min'], 0)} | max={r(d['max'], 0)}"
    )
    lines.append(f"density_p10_p90: [{r(d['p10'])}, {r(d['p90'])}]")

    # Peri-lesional block
    lines.append(
        f"parenchyma_adjacent_hu: mean={r(p['mean'])} | sd={r(p['sd'])}"
    )
    delta = p.get("delta_hu")
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        lines.append(f"delta_hu_lesion_parenchyma: {sign}{r(delta)}")
    else:
        lines.append("delta_hu_lesion_parenchyma: N/A")

    # Morphometry block (blank line separator)
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

    lines.append("</ROI_DATA>")
    return "\n".join(lines)
