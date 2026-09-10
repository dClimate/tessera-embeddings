"""Unit tests for ingest/roi_processing.py — ROI mask application and coverage filtering."""

from unittest.mock import patch

import dask.array as da
import numpy as np
import xarray as xr

from tessera_embeddings.ingest.roi_processing import apply_roi_mask, filter_low_coverage_dates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(data_vars: dict[str, np.ndarray], times: list[str] | None = None) -> xr.Dataset:
    """Create a lazy xarray Dataset with (time, northing, easting) dimensions.

    Args:
        data_vars: Mapping of variable name to 3D numpy array (time, northing, easting).
        times: ISO date strings for the time coordinate.
    """
    first = next(iter(data_vars.values()))
    nt, ny, nx = first.shape
    if times is None:
        times = [f"2024-06-{i + 1:02d}" for i in range(nt)]
    coords = {
        "time": [np.datetime64(t) for t in times],
        "northing": np.arange(ny),
        "easting": np.arange(nx),
    }
    ds_vars = {}
    for name, arr in data_vars.items():
        ds_vars[name] = (["time", "northing", "easting"], da.from_array(arr, chunks=(1, ny, nx)))
    return xr.Dataset(ds_vars, coords=coords)


# ---------------------------------------------------------------------------
# apply_roi_mask
# ---------------------------------------------------------------------------


class TestApplyROIMask:
    """Tests for apply_roi_mask."""

    def test_applies_mask_to_all_variables(self):
        """Pixels outside ROI should be zeroed for every variable."""
        # 2x2 spatial grid, mask is True in top-left and bottom-right
        band = np.array([[[10, 20], [30, 40]]], dtype=np.int16)
        ds = _make_dataset({"red": band.copy(), "green": band.copy()})

        mask_2d = da.from_array(np.array([[True, False], [False, True]]), chunks=(2, 2))

        with patch("tessera_embeddings.ingest.roi_processing.read_roi_mask", return_value=mask_2d):
            result = apply_roi_mask(ds, "fake.zarr", {"northing": 2, "easting": 2})

        red = result["red"].compute().values[0]
        np.testing.assert_array_equal(red, [[10, 0], [0, 40]])
        green = result["green"].compute().values[0]
        np.testing.assert_array_equal(green, [[10, 0], [0, 40]])

    def test_masking_is_lazy(self):
        """Nothing may compute here: both sensors call this once per date.

        Counts computes rather than inspecting the return value, so eagerness fails
        wherever it is reintroduced.
        """
        band = np.ones((1, 4, 4), dtype=np.int16)
        ds = _make_dataset({"band": band})
        mask_2d = da.from_array(np.ones((4, 4), dtype=bool), chunks=(4, 4))

        computes = []
        with (
            patch("tessera_embeddings.ingest.roi_processing.read_roi_mask", return_value=mask_2d),
            patch.object(da.Array, "compute", lambda self, **kw: computes.append(1)),
        ):
            out = apply_roi_mask(ds, "fake.zarr", {"northing": 4, "easting": 4})

        assert computes == [], f"apply_roi_mask computed eagerly {len(computes)} time(s)"
        assert isinstance(out, xr.Dataset)

    def test_custom_fill_value(self):
        """fill_value should replace pixels outside the ROI mask."""
        band = np.array([[[10, 20], [30, 40]]], dtype=np.int16)
        ds = _make_dataset({"red": band})

        mask_2d = da.from_array(np.array([[True, False], [False, True]]), chunks=(2, 2))

        with patch("tessera_embeddings.ingest.roi_processing.read_roi_mask", return_value=mask_2d):
            result = apply_roi_mask(ds, "fake.zarr", {"northing": 2, "easting": 2}, fill_value=-9999)

        red = result["red"].compute().values[0]
        np.testing.assert_array_equal(red, [[10, -9999], [-9999, 40]])


# ---------------------------------------------------------------------------
# filter_low_coverage_dates
# ---------------------------------------------------------------------------


class TestFilterLowCoverageDates:
    """Tests for filter_low_coverage_dates."""

    def test_drops_dates_below_threshold(self):
        """Dates with valid coverage below threshold should be dropped."""
        # 4x4 grid, roi_pixel_count=16
        # Date 1: all valid (SCL=4 everywhere) -> 100% -> keep
        # Date 2: all invalid (SCL=0 everywhere) -> 0% -> drop
        scl_d1 = np.full((1, 4, 4), 4, dtype=np.uint8)
        scl_d2 = np.full((1, 4, 4), 0, dtype=np.uint8)
        scl = np.concatenate([scl_d1, scl_d2], axis=0)

        band = np.ones_like(scl, dtype=np.int16)
        ds = _make_dataset({"scl": scl, "red": band})

        result = filter_low_coverage_dates(ds, roi_pixel_count=16, quality_band="scl", invalid_values=frozenset({0}))

        assert result.sizes["time"] == 1
        assert str(result.time.values[0])[:10] == "2024-06-01"

    def test_keeps_dates_above_threshold(self):
        """Dates above the threshold should be retained."""
        # Both dates have 100% valid pixels
        scl = np.full((2, 4, 4), 4, dtype=np.uint8)
        band = np.ones_like(scl, dtype=np.int16)
        ds = _make_dataset({"scl": scl, "red": band})

        result = filter_low_coverage_dates(ds, roi_pixel_count=16, quality_band="scl", invalid_values=frozenset({0}))

        assert result.sizes["time"] == 2

    def test_returns_empty_when_all_below(self):
        """All dates below threshold should return an empty time dimension."""
        scl = np.full((3, 4, 4), 0, dtype=np.uint8)
        band = np.ones_like(scl, dtype=np.int16)
        ds = _make_dataset(
            {"scl": scl, "red": band},
            times=["2024-06-01", "2024-06-02", "2024-06-03"],
        )

        result = filter_low_coverage_dates(ds, roi_pixel_count=16, quality_band="scl", invalid_values=frozenset({0}))

        assert result.sizes["time"] == 0

    def test_missing_quality_band_returns_unchanged(self):
        """If the quality band is missing, return the dataset unchanged with a warning."""
        band = np.ones((2, 4, 4), dtype=np.int16)
        ds = _make_dataset({"red": band})

        result = filter_low_coverage_dates(ds, roi_pixel_count=16, quality_band="scl", invalid_values=frozenset({0}))

        assert result.sizes["time"] == 2
        xr.testing.assert_identical(result, ds)

    def test_works_with_s1_like_invalid_values(self):
        """Should work with S1-like parameters (different band name and invalid set)."""
        # S1: quality band is "0_VV", invalid is {0} (nodata)
        # Date 1: 12/16 pixels valid (75%) -> keep at 5% threshold
        # Date 2: 0/16 pixels valid -> drop
        vv_d1 = np.ones((1, 4, 4), dtype=np.float32)
        vv_d1[0, 0, :] = 0  # 4 invalid pixels
        vv_d2 = np.zeros((1, 4, 4), dtype=np.float32)  # all invalid
        vv = np.concatenate([vv_d1, vv_d2], axis=0)

        ds = _make_dataset({"0_VV": vv})

        result = filter_low_coverage_dates(ds, roi_pixel_count=16, quality_band="0_VV", invalid_values=frozenset({0}))

        assert result.sizes["time"] == 1
        assert str(result.time.values[0])[:10] == "2024-06-01"

    def test_multiple_invalid_values(self):
        """Should handle multiple invalid values (like S2 SCL classes)."""
        # 4 pixels, roi_pixel_count=4
        # Date 1: SCL values [4, 8, 9, 5] -> 2 valid (classes 4,5), 2 invalid (8,9) -> 50% -> keep
        # Date 2: SCL values [0, 1, 2, 3] -> 0 valid -> 0% -> drop
        scl_d1 = np.array([[[4, 8], [9, 5]]], dtype=np.uint8)
        scl_d2 = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
        scl = np.concatenate([scl_d1, scl_d2], axis=0)

        ds = _make_dataset({"scl": scl})

        result = filter_low_coverage_dates(
            ds,
            roi_pixel_count=4,
            quality_band="scl",
            invalid_values=frozenset({0, 1, 2, 3, 8, 9}),
        )

        assert result.sizes["time"] == 1
        assert str(result.time.values[0])[:10] == "2024-06-01"

    def test_custom_min_valid_coverage(self):
        """Custom min_valid_coverage threshold should be respected."""
        # 4x4 grid. Date 1: 1 valid pixel out of 16 = 6.25%
        scl = np.full((1, 4, 4), 0, dtype=np.uint8)
        scl[0, 0, 0] = 4  # one valid pixel

        ds = _make_dataset({"scl": scl})

        # 6.25% >= 5% -> keep
        result_low = filter_low_coverage_dates(
            ds, roi_pixel_count=16, quality_band="scl", invalid_values=frozenset({0}), min_valid_coverage=5.0
        )
        assert result_low.sizes["time"] == 1

        # 6.25% < 10% -> drop
        result_high = filter_low_coverage_dates(
            ds, roi_pixel_count=16, quality_band="scl", invalid_values=frozenset({0}), min_valid_coverage=10.0
        )
        assert result_high.sizes["time"] == 0
