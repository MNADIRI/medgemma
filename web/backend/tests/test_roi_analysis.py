"""Tests for roi_analysis module — spatial localization and core extraction."""

import numpy as np
import pytest

from services.roi_analysis import (
    _compute_spatial_localization,
    _erode_mask,
    extract_roi_data,
    format_roi_data,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_circular_mask(shape=(100, 100), center=(50, 50), radius=20):
    """Create a circular binary mask."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = ((y - center[0]) ** 2 + (x - center[1]) ** 2) <= radius ** 2
    return mask


def _make_hu_array(shape=(100, 100), value=40.0):
    """Create a uniform HU array."""
    return np.full(shape, value, dtype=np.float64)


# Standard axial orientation: row cosines = [1,0,0], col cosines = [0,1,0]
AXIAL_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


# ── Spatial localization tests ─────────────────────────────────────────────

class TestSpatialLocalization:
    """Test DICOM patient-coordinate spatial localization."""

    def test_left_laterality(self):
        """Centroid at col that maps to X > +5mm should be 'left'."""
        # Origin at (-50, -50, 0), pixel_spacing (1, 1), axial orientation
        # Centroid at col=60 → X = -50 + 60*1*1 = +10 → left
        mask = _make_circular_mask(center=(50, 60), radius=5)
        result = _compute_spatial_localization(
            mask,
            pixel_spacing=(1.0, 1.0),
            image_position_patient=(-50.0, -50.0, 0.0),
            image_orientation_patient=AXIAL_ORIENTATION,
        )
        assert result["laterality"] == "left"
        assert result["patient_xyz_mm"] is not None

    def test_right_laterality(self):
        """Centroid at col that maps to X < -5mm should be 'right'."""
        # Origin at (-50, -50, 0), centroid at col=40 → X = -50 + 40 = -10 → right
        mask = _make_circular_mask(center=(50, 40), radius=5)
        result = _compute_spatial_localization(
            mask,
            pixel_spacing=(1.0, 1.0),
            image_position_patient=(-50.0, -50.0, 0.0),
            image_orientation_patient=AXIAL_ORIENTATION,
        )
        assert result["laterality"] == "right"

    def test_midline(self):
        """Centroid at col that maps to |X| <= 5mm should be 'midline'."""
        # Origin at (-50, 0, 0), centroid at col=50 → X = -50 + 50 = 0 → midline
        mask = _make_circular_mask(center=(50, 50), radius=5)
        result = _compute_spatial_localization(
            mask,
            pixel_spacing=(1.0, 1.0),
            image_position_patient=(-50.0, 0.0, 0.0),
            image_orientation_patient=AXIAL_ORIENTATION,
        )
        assert result["laterality"] == "midline"

    def test_posterior_position(self):
        """Centroid at row that maps to Y > +5mm should be 'posterior'."""
        # Origin at (0, -50, 0), centroid at row=60 → Y = -50 + 60*1*1 = +10 → posterior
        mask = _make_circular_mask(center=(60, 50), radius=5)
        result = _compute_spatial_localization(
            mask,
            pixel_spacing=(1.0, 1.0),
            image_position_patient=(0.0, -50.0, 0.0),
            image_orientation_patient=AXIAL_ORIENTATION,
        )
        assert result["antero_posterior"] == "posterior"

    def test_anterior_position(self):
        """Centroid at row that maps to Y < -5mm should be 'anterior'."""
        # Origin at (0, -50, 0), centroid at row=40 → Y = -50 + 40 = -10 → anterior
        mask = _make_circular_mask(center=(40, 50), radius=5)
        result = _compute_spatial_localization(
            mask,
            pixel_spacing=(1.0, 1.0),
            image_position_patient=(0.0, -50.0, 0.0),
            image_orientation_patient=AXIAL_ORIENTATION,
        )
        assert result["antero_posterior"] == "anterior"

    def test_central_position(self):
        """Centroid at row that maps to |Y| <= 5mm should be 'central'."""
        mask = _make_circular_mask(center=(50, 50), radius=5)
        result = _compute_spatial_localization(
            mask,
            pixel_spacing=(1.0, 1.0),
            image_position_patient=(0.0, -50.0, 0.0),
            image_orientation_patient=AXIAL_ORIENTATION,
        )
        assert result["antero_posterior"] == "central"

    def test_unknown_when_orientation_absent(self):
        """Both fields should be 'unknown' when ImageOrientationPatient is None."""
        mask = _make_circular_mask(center=(50, 50), radius=5)
        result = _compute_spatial_localization(
            mask,
            pixel_spacing=(1.0, 1.0),
            image_position_patient=(-50.0, -50.0, 0.0),
            image_orientation_patient=None,
        )
        assert result["laterality"] == "unknown"
        assert result["antero_posterior"] == "unknown"
        assert result["patient_xyz_mm"] is None

    def test_unknown_when_position_absent(self):
        """Both fields should be 'unknown' when ImagePositionPatient is None."""
        mask = _make_circular_mask(center=(50, 50), radius=5)
        result = _compute_spatial_localization(
            mask,
            pixel_spacing=(1.0, 1.0),
            image_position_patient=None,
            image_orientation_patient=AXIAL_ORIENTATION,
        )
        assert result["laterality"] == "unknown"
        assert result["antero_posterior"] == "unknown"

    def test_anisotropic_pixel_spacing(self):
        """Pixel spacing should scale the coordinate transform correctly."""
        # Origin (0, 0, 0), spacing (0.5, 2.0), col=10 → X = 0 + 10*2.0*1 = 20 → left
        mask = _make_circular_mask(center=(50, 10), radius=5)
        result = _compute_spatial_localization(
            mask,
            pixel_spacing=(0.5, 2.0),
            image_position_patient=(0.0, 0.0, 0.0),
            image_orientation_patient=AXIAL_ORIENTATION,
        )
        assert result["laterality"] == "left"
        assert result["patient_xyz_mm"][0] == pytest.approx(20.0, abs=1.0)


# ── Erosion tests ──────────────────────────────────────────────────────────

class TestErosion:
    def test_erosion_shrinks_mask(self):
        """Eroded mask should have fewer pixels than original."""
        mask = _make_circular_mask(radius=20)
        eroded = _erode_mask(mask, radius=2)
        assert np.sum(eroded) < np.sum(mask)
        assert np.sum(eroded) >= 20  # still large enough

    def test_erosion_fallback_small_mask(self):
        """Small mask should not be eroded (fallback to original)."""
        mask = _make_circular_mask(radius=3)  # ~28 pixels, erosion would leave < 20
        original_count = np.sum(mask)
        eroded = _erode_mask(mask, radius=2)
        # Should fall back — either same or original count
        assert np.sum(eroded) >= 20 or np.sum(eroded) == original_count


# ── Full extraction tests ──────────────────────────────────────────────────

class TestExtractRoiData:
    def test_basic_extraction(self):
        """Full pipeline should return all expected keys."""
        hu = _make_hu_array(value=50.0)
        mask = _make_circular_mask(radius=20)
        result = extract_roi_data(
            hu, mask, pixel_spacing=(1.0, 1.0),
            image_position_patient=(-50.0, -50.0, 0.0),
            image_orientation_patient=AXIAL_ORIENTATION,
        )

        assert "density" in result
        assert "peri_lesional" in result
        assert "morphometry" in result
        assert "spatial" in result

        # Density should have 9 deciles
        assert len(result["density"]["deciles"]) == 9
        assert "density_asymmetry" in result["density"]

        # Peri-lesional should use median-based delta
        assert "delta_hu_median_parenchyma" in result["peri_lesional"]

        # Morphometry should have eroded_area_fraction
        assert "eroded_area_fraction" in result["morphometry"]
        assert 0.0 < result["morphometry"]["eroded_area_fraction"] <= 1.0

        # Spatial should have laterality
        assert result["spatial"]["laterality"] in ("left", "right", "midline")
        assert result["spatial"]["antero_posterior"] in ("anterior", "posterior", "central")

    def test_empty_mask(self):
        """Empty mask should return empty result without errors."""
        hu = _make_hu_array()
        mask = np.zeros((100, 100), dtype=bool)
        result = extract_roi_data(hu, mask, pixel_spacing=(1.0, 1.0))
        assert result["density"]["mean"] == 0
        assert result["spatial"]["laterality"] == "unknown"

    def test_uniform_density_has_zero_asymmetry(self):
        """Uniform HU → density_asymmetry should be ~0."""
        hu = _make_hu_array(value=40.0)
        mask = _make_circular_mask(radius=20)
        result = extract_roi_data(hu, mask, pixel_spacing=(1.0, 1.0))
        assert abs(result["density"]["density_asymmetry"]) < 0.01


# ── Format tests ───────────────────────────────────────────────────────────

class TestFormatRoiData:
    def test_format_contains_expected_fields(self):
        """Formatted output should contain all new fields."""
        hu = _make_hu_array(value=50.0)
        mask = _make_circular_mask(radius=20)
        result = extract_roi_data(
            hu, mask, pixel_spacing=(1.0, 1.0),
            image_position_patient=(-50.0, -50.0, 0.0),
            image_orientation_patient=AXIAL_ORIENTATION,
        )
        formatted = format_roi_data(result)

        assert "<ROI_DATA>" in formatted
        assert "</ROI_DATA>" in formatted
        assert "density_deciles:" in formatted
        assert "density_asymmetry:" in formatted
        assert "delta_hu_median_parenchyma:" in formatted
        assert "eroded_area_fraction:" in formatted
        assert "laterality:" in formatted
        assert "antero_posterior:" in formatted
