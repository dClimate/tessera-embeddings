"""Unit tests for the empty-store seeder (storage/empty_store.py).

The load-bearing properties:

* grid identity — an empty store is the byte-for-byte grid a real ``write_dataset``
  produces (same vars, dtypes, coords, chunking, CRS), so a later ingest / merge
  aligns to it exactly;
* it computes no pixels (all-fill, no chunk objects) and its cost is independent
  of spatial extent;
* the per-dtype fill (0 int / NaN float) matches what the merge treats as fill;
* the empty store is a valid merge target end-to-end.

The mosaic-specific contract (``MOSAIC_VAR_DTYPES``, per-kind wrappers,
``classify_mosaic``) lives downstream in yield-embeddings and is tested there;
here the var/dtype schema is caller-supplied.
"""

from __future__ import annotations

import time
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from tessera_embeddings.config.ingest import INGEST_CHUNKS
from tessera_embeddings.ingest.roi import ROIMetadata
from tessera_embeddings.storage.empty_store import VarSpec, create_empty_store, create_empty_store_from_coords
from tessera_embeddings.storage.time_axis import daily_times
from tessera_embeddings.storage.zarr_store import open_store, open_store_as_zarr_group, write_dataset

_CRS = "EPSG:32615"
_VARS: dict[str, np.dtype] = {"0_VV": np.dtype("uint16"), "0_VH": np.dtype("uint16")}


def _roi(height: int = 20, width: int = 16) -> ROIMetadata:
    """Build an ROIMetadata from a synthetic geobox (10 m, descending northing)."""
    geobox = GeoBox(
        shape=(height, width),
        affine=Affine(10, 0, 500000, 0, -10, 4000000),
        crs=_CRS,
    )
    return ROIMetadata(
        bbox_wgs84=(0.0, 0.0, 1.0, 1.0),
        native_crs=_CRS,
        geobox=geobox,
        width=width,
        height=height,
    )


def _times(n: int = 2) -> np.ndarray:
    return np.array([f"{2023 + i}-12-31" for i in range(n)], dtype="datetime64[ns]")


def test_grid_identity_matches_real_write(tmp_path) -> None:
    """An empty store matches a real ``write_dataset`` output over the same grid:
    same vars, dtypes, coords, chunking, CRS. Both go through the same OSS create
    path, so this pins the seeder to the ingest grid for a caller-supplied schema.
    """
    roi = _roi()
    times = _times(1)

    coords = roi.geobox.coordinates
    northing, easting = coords["y"].values, coords["x"].values
    shape = (len(times), roi.height, roi.width)
    cs = (INGEST_CHUNKS["time"], INGEST_CHUNKS["northing"], INGEST_CHUNKS["easting"])
    real_ds = xr.Dataset(
        {
            name: (("time", "northing", "easting"), da.zeros(shape, dtype=dtype, chunks=cs))
            for name, dtype in _VARS.items()
        },
        coords={"time": times, "northing": northing, "easting": easting},
    )
    real_path = str(tmp_path / "real.zarr")
    write_dataset(real_path, real_ds, tile_id="roi", baselines={}, chunks=INGEST_CHUNKS, crs=_CRS)

    empty_path = str(tmp_path / "empty.zarr")
    create_empty_store(empty_path, roi=roi, times=times, var_dtypes=_VARS, tile_id="roi", crs=_CRS)

    real, empty = open_store(real_path), open_store(empty_path)
    assert set(empty.data_vars) == set(real.data_vars)
    assert {k: str(v.dtype) for k, v in empty.data_vars.items()} == {k: str(v.dtype) for k, v in real.data_vars.items()}
    assert dict(empty.sizes) == dict(real.sizes)
    np.testing.assert_array_equal(empty.northing.values, real.northing.values)
    np.testing.assert_array_equal(empty.easting.values, real.easting.values)
    np.testing.assert_array_equal(empty.time.values, real.time.values)
    assert empty.attrs.get("crs") == real.attrs.get("crs") == _CRS
    for name in empty.data_vars:
        assert empty[name].chunksizes == real[name].chunksizes


def test_coords_match_geobox(tmp_path) -> None:
    """Empty store coords are exactly the geobox pixel centers (the grid authority)."""
    roi = _roi()
    path = str(tmp_path / "s.zarr")
    create_empty_store(path, roi=roi, times=_times(1), var_dtypes=_VARS, tile_id="roi", crs=_CRS)
    ds = open_store(path)
    coords = roi.geobox.coordinates
    np.testing.assert_array_equal(ds.northing.values, coords["y"].values)
    np.testing.assert_array_equal(ds.easting.values, coords["x"].values)


def test_reads_back_all_fill(tmp_path) -> None:
    """Every pixel reads back as the integer fill (0) — no data computed."""
    roi = _roi()
    path = str(tmp_path / "s.zarr")
    create_empty_store(path, roi=roi, times=_times(2), var_dtypes=_VARS, tile_id="roi", crs=_CRS)
    ds = open_store(path)
    for name in ds.data_vars:
        vals = ds[name].values
        assert vals.dtype == _VARS[name]
        assert np.all(vals == 0)


def test_footprint_is_metadata_scale(tmp_path) -> None:
    """All-fill chunks are not written as objects — on-disk size is metadata-scale,
    far below the nominal extent (independent of extent).
    """
    roi = _roi(height=800, width=800)
    path = str(tmp_path / "s.zarr")
    create_empty_store(path, roi=roi, times=_times(1), var_dtypes=_VARS, tile_id="roi", crs=_CRS)
    nominal = 1 * 800 * 800 * 2 * 2  # 2 vars, uint16
    on_disk = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    # Metadata-scale: orders of magnitude below nominal. Loose bound, not brittle.
    assert on_disk < nominal // 10


def test_float_dtype_fills_nan(tmp_path) -> None:
    """A float var fills with NaN (not 0) — the per-dtype fill resolution."""
    roi = _roi()
    path = str(tmp_path / "s.zarr")
    create_empty_store(path, roi=roi, times=_times(1), var_dtypes={"x": np.dtype("float32")}, tile_id="roi", crs=_CRS)
    ds = open_store(path)
    assert np.all(np.isnan(ds["x"].values))


def test_geobox_coord_mismatch_raises() -> None:
    """A geobox whose coords disagree with (height, width) is rejected before write."""
    roi = _roi(height=20, width=16)
    bad = ROIMetadata(
        bbox_wgs84=roi.bbox_wgs84,
        native_crs=roi.native_crs,
        geobox=roi.geobox,  # 20x16
        width=99,  # lie
        height=99,
    )
    with pytest.raises(ValueError, match="disagree"):
        create_empty_store("x.zarr", roi=bad, times=_times(1), var_dtypes=_VARS, tile_id="roi", crs=_CRS)


def test_creation_is_extent_independent(tmp_path) -> None:
    """Creation time must not scale with spatial extent (the direct-zarr path avoids
    materialising dask chunks). A 40 000 x 40 000 store must complete well under 5s.
    """
    roi = _roi(height=40_000, width=40_000)
    path = str(tmp_path / "large.zarr")
    t0 = time.monotonic()
    create_empty_store(path, roi=roi, times=_times(4), var_dtypes=_VARS, tile_id="roi", crs=_CRS)
    elapsed = time.monotonic() - t0
    assert elapsed < 5, f"create_empty_store took {elapsed:.1f}s on a large store (expected < 5s)"


def test_no_data_chunk_objects(tmp_path) -> None:
    """Data-var arrays exist in metadata but write no chunk objects (all-fill)."""
    roi = _roi()
    path = str(tmp_path / "s.zarr")
    create_empty_store(path, roi=roi, times=_times(1), var_dtypes=_VARS, tile_id="roi", crs=_CRS)
    group = open_store_as_zarr_group(path)
    vv = group["0_VV"]
    assert vv.shape == (1, roi.height, roi.width)
    assert vv.dtype == np.uint16
    assert vv.nchunks_initialized == 0


def test_from_coords_rejects_dim_without_coord(tmp_path) -> None:
    """A ``VarSpec`` naming a dim absent from ``coords`` is a construction error."""
    with pytest.raises(ValueError, match="absent from coords"):
        create_empty_store_from_coords(
            str(tmp_path / "s.zarr"),
            coords={"northing": np.arange(4.0)},
            var_specs={"v": VarSpec(dims=("northing", "easting"), dtype=np.dtype("uint16"), chunks=(4, 4))},
            commit_msg="x",
        )


@pytest.mark.parametrize(
    ("start", "end", "size", "first", "last"),
    [
        ("2025-01-01", "2025-12-31", 365, "2025-01-01", "2025-12-31"),  # full year, inclusive
        ("2025-06-16", "2025-06-16", 1, "2025-06-16", "2025-06-16"),  # single day
    ],
)
def test_daily_times(start, end, size, first, last) -> None:
    """Date-range modality: one timestamp per day, both bounds inclusive, contiguous."""
    t = daily_times(start, end)
    assert t.dtype == np.dtype("datetime64[ns]")
    assert t.size == size
    assert str(t[0])[:10] == first
    assert str(t[-1])[:10] == last
    assert np.all(np.diff(t) == np.timedelta64(1, "D"))  # vacuously true for size 1


def test_daily_times_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="precedes start"):
        daily_times("2025-12-31", "2025-01-01")
