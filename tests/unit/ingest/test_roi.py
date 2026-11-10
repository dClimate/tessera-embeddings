"""Unit tests for ingest/roi.py — ROI Zarr metadata extraction."""

import numpy as np
import pytest
import zarr
from affine import Affine

from tessera_embeddings.ingest.roi import ROIMetadata, read_roi_metadata


def _write_test_zarr(
    path,
    crs: str,
    bounds: tuple,
    bbox_wgs84: tuple,
    width: int = 100,
    height: int = 100,
) -> None:
    """Write a minimal Zarr ROI store with the given CRS and bounds.

    Args:
        path: Output directory path.
        crs: CRS string (e.g. "EPSG:32615").
        bounds: (minx, miny, maxx, maxy) in the given CRS.
        bbox_wgs84: WGS84 bounding box (minx, miny, maxx, maxy).
        width: Raster width in pixels.
        height: Raster height in pixels.
    """
    minx, miny, maxx, maxy = bounds
    res_x = (maxx - minx) / width
    res_y = (maxy - miny) / height
    transform = Affine(res_x, 0, minx, 0, -res_y, maxy)

    data = np.ones((height, width), dtype=bool)
    z = zarr.open_array(str(path), mode="w", shape=data.shape, dtype=data.dtype, chunks=(height, width))
    z[:] = data
    z.attrs["crs"] = crs
    z.attrs["transform"] = list(transform)[:6]
    z.attrs["bbox_wgs84"] = list(bbox_wgs84)


class TestReadROIMetadata:
    """Tests for read_roi_metadata."""

    def test_utm_zarr_returns_stored_wgs84_bbox(self, tmp_path):
        """read_roi_metadata should return the exact bbox_wgs84 stored in the Zarr attrs."""
        zarr_path = tmp_path / "utm.zarr"
        utm_bounds = (400000.0, 5000000.0, 410000.0, 5010000.0)
        expected_bbox = (-93.5, 45.1, -93.3, 45.2)
        _write_test_zarr(zarr_path, "EPSG:32615", utm_bounds, bbox_wgs84=expected_bbox)

        result = read_roi_metadata(str(zarr_path))

        assert isinstance(result, ROIMetadata)
        assert result.native_crs == "EPSG:32615"
        assert result.width == 100
        assert result.height == 100
        assert result.bbox_wgs84 == expected_bbox

    def test_missing_crs_raises(self, tmp_path):
        """A Zarr store without CRS attrs should raise KeyError."""
        zarr_path = tmp_path / "no_crs.zarr"
        data = np.ones((10, 10), dtype=bool)
        z = zarr.open_array(str(zarr_path), mode="w", shape=data.shape, dtype=data.dtype)
        z[:] = data

        with pytest.raises(KeyError):
            read_roi_metadata(str(zarr_path))

    def test_missing_store_raises(self):
        """A nonexistent path should raise an error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            read_roi_metadata("/nonexistent/path/to/roi.zarr")
