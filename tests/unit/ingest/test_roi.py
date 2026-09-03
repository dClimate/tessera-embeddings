"""Unit tests for ingest/roi.py — ROI Zarr metadata extraction."""

import numpy as np
import pytest
import zarr
from affine import Affine

from tessera_embeddings.ingest.roi import ROIMetadata, read_roi_mask, read_roi_metadata


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

    def test_wgs84_zarr_returns_stored_bbox(self, tmp_path):
        """read_roi_metadata should return the stored bbox even for a WGS84 CRS."""
        zarr_path = tmp_path / "wgs84.zarr"
        wgs84_bounds = (-95.0, 45.0, -94.0, 46.0)
        expected_bbox = (-95.0, 45.0, -94.0, 46.0)
        _write_test_zarr(zarr_path, "EPSG:4326", wgs84_bounds, bbox_wgs84=expected_bbox, width=50, height=50)

        result = read_roi_metadata(str(zarr_path))

        assert result.native_crs == "EPSG:4326"
        assert result.width == 50
        assert result.height == 50
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


class TestReadROIMask:
    """Tests for read_roi_mask."""

    def test_returns_dask_array(self, tmp_path):
        """read_roi_mask should return a chunked dask array."""
        import dask.array as da

        zarr_path = tmp_path / "mask.zarr"
        data = np.ones((20, 20), dtype=bool)
        data[5:10, 5:10] = False
        z = zarr.open_array(str(zarr_path), mode="w", shape=data.shape, dtype=data.dtype, chunks=(10, 10))
        z[:] = data
        z.attrs["crs"] = "EPSG:32615"
        z.attrs["transform"] = [10, 0, 400000, 0, -10, 5000200]

        result = read_roi_mask(str(zarr_path), {"northing": 10, "easting": 10})

        assert isinstance(result, da.Array)
        computed = result.compute()
        np.testing.assert_array_equal(computed, data)

    def test_chunk_sizes_respected(self, tmp_path):
        """Returned dask array should use the requested chunk sizes."""
        zarr_path = tmp_path / "mask.zarr"
        data = np.ones((100, 100), dtype=bool)
        z = zarr.open_array(str(zarr_path), mode="w", shape=data.shape, dtype=data.dtype, chunks=(50, 50))
        z[:] = data

        result = read_roi_mask(str(zarr_path), {"northing": 25, "easting": 25})

        assert result.chunksize == (25, 25)
