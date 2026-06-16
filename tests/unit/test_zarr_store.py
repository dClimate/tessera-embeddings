"""Unit tests for tessera_embeddings/storage/zarr_store.py.

Tests cover both local and S3-backed stores, verifying:
- Store creation and metadata
- Custom and default-style chunking
- DOY storage and append/merge
- Append operations extending the time dimension
- Transactional isolation (uncommitted writes not visible)
- Parallel access patterns
- Cleanup on write failure

Adapted from the reference ``test_zarr_store.py``:
- The reference's ``write_reflectance`` was removed in the refactor; the
  generic ``write_dataset`` is used instead. It requires explicit ``chunks``
  and a ``crs`` keyword argument.
- ``_compute_doy`` was renamed to the public ``compute_doy``.
- The reference's ``TESSERA_CHUNKS`` / ``DEFAULT_CHUNKS`` constants do not
  exist in the target; chunk dicts are defined locally.
- Cloudmask-store tests are omitted — the target has no cloudmask module.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from icechunk.xarray import to_icechunk

from tessera_embeddings.config import S2_L2A_BANDS
from tessera_embeddings.storage.zarr_store import (
    _CHUNK_CACHE_BYTES,
    S3Config,
    _default_repo_config,
    _open_writable_session,
    _pad_region_to_chunks,
    _write_new,
    compute_doy,
    get_existing_dates,
    open_store,
    resolve_region,
    set_s3_config,
    write_dataset,
    write_region,
)

# Chunk dicts replacing the reference's TESSERA_CHUNKS / DEFAULT_CHUNKS.
# write_dataset clamps spatial chunks to the data extent, so a large value
# yields a single spatial chunk covering the full tile (the default-style
# behavior the reference exercised with DEFAULT_CHUNKS=10980).
DEFAULT_CHUNKS = {"time": 1, "northing": 10980, "easting": 10980}
# A chunk size smaller than the test data extent, to exercise the
# multi-chunk (splitting) path rather than the clamp-to-extent path.
SPLIT_CHUNKS = {"time": 1, "northing": 500, "easting": 500}


def _write_reflectance(store_path, data, tile_id, baselines, chunks=DEFAULT_CHUNKS, crs="EPSG:32633"):
    """Local helper mirroring the reference's removed ``write_reflectance``.

    Delegates to ``write_dataset`` with explicit chunks + crs so the ported
    tests read like the originals.
    """
    write_dataset(store_path, data, tile_id=tile_id, baselines=baselines, chunks=chunks, crs=crs)


class TestReflectanceStore:
    """Tests for reflectance Zarr store operations (via write_dataset)."""

    def test_time_encoding_uses_standard_epoch(self, local_zarr_path, sample_reflectance_data):
        """Time coordinate should use nanoseconds since 1970 for compatibility."""
        dates = ["2024-06-15", "2024-06-20"]
        data = sample_reflectance_data(dates, height=64, width=64)
        store_path = str(local_zarr_path / "test_tile" / "reflectance.zarr")

        _write_reflectance(store_path, data, tile_id="33UUP", baselines=dict.fromkeys(dates, 400))

        ds = open_store(store_path)
        assert ds.sizes["time"] == 2
        assert str(ds.time.values[0])[:10] == "2024-06-15"
        assert str(ds.time.values[1])[:10] == "2024-06-20"

    def test_create_store_initializes_with_correct_metadata(self, local_zarr_path, sample_reflectance_data):
        """Verify store created with correct attributes."""
        dates = ["2024-01-01", "2024-01-06"]
        data = sample_reflectance_data(dates, height=256, width=256)
        store_path = str(local_zarr_path / "test_tile" / "reflectance.zarr")

        _write_reflectance(store_path, data, tile_id="33UUP", baselines={"2024-01-01": 400, "2024-01-06": 400})

        ds = open_store(store_path)
        assert ds.attrs["tile_id"] == "33UUP"
        assert set(ds.attrs["baselines_applied"].keys()) == {"2024-01-01", "2024-01-06"}
        for band in S2_L2A_BANDS:
            assert band in ds.data_vars
        assert ds.sizes == {"time": 2, "northing": 256, "easting": 256}

    def test_create_store_stores_crs(self, local_zarr_path, sample_reflectance_data):
        """write_dataset stores the CRS in root attrs."""
        dates = ["2024-01-01"]
        data = sample_reflectance_data(dates, height=64, width=64)
        store_path = str(local_zarr_path / "crs_tile" / "reflectance.zarr")

        _write_reflectance(store_path, data, tile_id="33UUP", baselines=dict.fromkeys(dates, 400), crs="EPSG:32615")

        ds = open_store(store_path)
        assert ds.attrs["crs"] == "EPSG:32615"

    def test_create_store_uses_expected_chunk_sizes(self, local_zarr_path, sample_reflectance_data):
        """Verify store is created with correct chunking for efficient access."""
        dates = ["2024-01-01", "2024-01-06"]
        data = sample_reflectance_data(dates, height=256, width=256)
        store_path = str(local_zarr_path / "test_tile" / "reflectance.zarr")

        _write_reflectance(store_path, data, tile_id="33UUP", baselines=dict.fromkeys(dates, 400))

        ds = open_store(store_path)
        assert ds["blue"].chunks[0] == (1, 1)  # time chunks
        assert ds["blue"].chunks[1][0] == 256  # y chunk size (clamped to extent)
        assert ds["blue"].chunks[2][0] == 256  # x chunk size (clamped to extent)

    def test_append_extends_time_dimension(self, local_zarr_path, sample_reflectance_data):
        """Verify append extends time dim without corrupting existing data."""
        store_path = str(local_zarr_path / "test_tile" / "reflectance.zarr")

        # Create initial store
        initial_dates = ["2024-01-01", "2024-01-06"]
        initial_data = sample_reflectance_data(initial_dates, height=256, width=256, seed=42)
        _write_reflectance(store_path, initial_data, tile_id="33UUP", baselines=dict.fromkeys(initial_dates, 400))

        original_blue = open_store(store_path)["blue"].values.copy()

        # Append new date
        new_data = sample_reflectance_data(["2024-01-11"], height=256, width=256, seed=99)
        _write_reflectance(store_path, new_data, tile_id="33UUP", baselines={"2024-01-11": 400})

        ds = open_store(store_path)
        assert ds.sizes["time"] == 3
        np.testing.assert_array_equal(ds["blue"].isel(time=slice(0, 2)).values, original_blue)

    def test_append_preserves_crs_attr(self, local_zarr_path, sample_reflectance_data):
        """Appending must not clobber the root crs attr written at creation."""
        store_path = str(local_zarr_path / "crs_append" / "reflectance.zarr")

        initial = sample_reflectance_data(["2024-01-01"], height=64, width=64)
        _write_reflectance(store_path, initial, tile_id="33UUP", baselines={"2024-01-01": 400}, crs="EPSG:32615")

        new_data = sample_reflectance_data(["2024-01-06"], height=64, width=64, seed=99)
        _write_reflectance(store_path, new_data, tile_id="33UUP", baselines={"2024-01-06": 400}, crs="EPSG:32615")

        ds = open_store(store_path)
        assert ds.sizes["time"] == 2
        assert ds.attrs["crs"] == "EPSG:32615"

    def test_get_existing_dates_returns_empty_for_missing_store(self, local_zarr_path):
        """Query non-existent store path returns empty set."""
        assert get_existing_dates(str(local_zarr_path / "nonexistent.zarr")) == set()

    def test_get_existing_dates_returns_correct_dates(self, local_zarr_path, sample_reflectance_data):
        """Verify returned set matches stored dates exactly."""
        dates = ["2024-01-01", "2024-01-06", "2024-01-11"]
        data = sample_reflectance_data(dates, height=64, width=64)
        store_path = str(local_zarr_path / "test_tile" / "reflectance.zarr")

        _write_reflectance(store_path, data, tile_id="33UUP", baselines=dict.fromkeys(dates, 400))

        assert get_existing_dates(store_path) == set(dates)


class TestIcechunkTransactions:
    """Tests for Icechunk transactional behavior."""

    def test_uncommitted_write_not_visible(self, local_zarr_path, sample_reflectance_data):
        """Uncommitted writes should not be visible to new read sessions."""
        data = sample_reflectance_data(["2024-01-01"], height=64, width=64)
        store_path = str(local_zarr_path / "test_tile" / "reflectance.zarr")

        _write_reflectance(store_path, data, tile_id="33UUP", baselines={"2024-01-01": 400})

        # Start uncommitted write
        session, _ = _open_writable_session(store_path)
        new_data = sample_reflectance_data(["2024-01-06"], height=64, width=64, seed=99)
        to_icechunk(
            new_data.chunk({"time": 1, "northing": 64, "easting": 64}),
            session,
            mode="a",
            append_dim="time",
        )
        # DO NOT commit

        # New reader sees only committed data
        assert open_store(store_path).sizes["time"] == 1

    def test_committed_write_visible(self, local_zarr_path, sample_reflectance_data):
        """Committed writes should be visible to new read sessions."""
        data = sample_reflectance_data(["2024-01-01"], height=64, width=64)
        store_path = str(local_zarr_path / "test_tile" / "reflectance.zarr")

        _write_reflectance(store_path, data, tile_id="33UUP", baselines={"2024-01-01": 400})

        session, _ = _open_writable_session(store_path)
        new_data = sample_reflectance_data(["2024-01-06"], height=64, width=64, seed=99)
        to_icechunk(
            new_data.chunk({"time": 1, "northing": 64, "easting": 64}),
            session,
            mode="a",
            append_dim="time",
        )
        session.commit("Add new date")

        assert open_store(store_path).sizes["time"] == 2


class TestParallelAccess:
    """Tests for parallel read/write operations."""

    @pytest.fixture
    def moto_s3_config(self, moto_server, test_bucket):
        """Configure zarr_store to use moto S3."""
        config = S3Config(
            bucket=test_bucket,
            endpoint_url=moto_server,
            allow_http=True,
            access_key_id="testing",
            secret_access_key="testing",
            region="us-east-1",
        )
        set_s3_config(config)
        yield test_bucket
        set_s3_config(None)

    def test_parallel_reads(self, local_zarr_path, sample_reflectance_data):
        """Multiple concurrent readers can access the same store."""
        dates = ["2024-01-01", "2024-01-06", "2024-01-11"]
        data = sample_reflectance_data(dates, height=64, width=64)
        store_path = str(local_zarr_path / "test_tile" / "reflectance.zarr")

        _write_reflectance(store_path, data, tile_id="33UUP", baselines=dict.fromkeys(dates, 400))

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(lambda d: open_store(store_path).sel(time=d)["blue"].values.shape[0], d) for d in dates
            ]
            results = [f.result() for f in as_completed(futures)]

        assert all(y_size == 64 for y_size in results)

    def test_parallel_writes_to_different_stores(self, local_zarr_path, sample_reflectance_data):
        """Concurrent writes to different stores don't conflict."""
        tiles = ["33UUP", "33UVP", "33UWP"]

        def create_store(tile_id, seed):
            data = sample_reflectance_data(["2024-01-01"], height=64, width=64, seed=seed)
            path = str(local_zarr_path / tile_id / "reflectance.zarr")
            _write_reflectance(path, data, tile_id=tile_id, baselines={"2024-01-01": 400})
            return tile_id

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_store, t, i * 100) for i, t in enumerate(tiles)]
            results = {f.result() for f in as_completed(futures)}

        assert results == set(tiles)
        for tile_id in tiles:
            ds = open_store(str(local_zarr_path / tile_id / "reflectance.zarr"))
            assert ds.attrs["tile_id"] == tile_id

    def test_parallel_writes_to_s3(self, moto_s3_config, sample_reflectance_data):
        """Concurrent writes to different S3 stores don't conflict."""
        bucket = moto_s3_config
        tiles = ["33UUP", "33UVP", "33UWP"]

        def create_store(tile_id, seed):
            data = sample_reflectance_data(["2024-01-01"], height=64, width=64, seed=seed)
            _write_reflectance(
                f"s3://{bucket}/{tile_id}/reflectance.zarr",
                data,
                tile_id=tile_id,
                baselines={"2024-01-01": 400},
            )
            return tile_id

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_store, t, i * 100) for i, t in enumerate(tiles)]
            results = {f.result() for f in as_completed(futures)}

        assert results == set(tiles)

    def test_uncommitted_writes_isolated_from_concurrent_reads(self, moto_s3_config, sample_reflectance_data):
        """Uncommitted writes aren't visible to concurrent readers."""
        bucket = moto_s3_config
        store_path = f"s3://{bucket}/test_tile/reflectance.zarr"

        _write_reflectance(
            store_path,
            sample_reflectance_data(["2024-01-01"], height=64, width=64),
            tile_id="33UUP",
            baselines={"2024-01-01": 400},
        )

        # Start uncommitted write
        session, _ = _open_writable_session(store_path)
        new_data = sample_reflectance_data(["2024-01-06"], height=64, width=64, seed=99)
        to_icechunk(
            new_data.chunk({"time": 1, "northing": 64, "easting": 64}),
            session,
            mode="a",
            append_dim="time",
        )

        # Concurrent reads see only committed data
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda _: open_store(store_path).sizes["time"], range(3)))
        assert all(r == 1 for r in results)

        # After commit, reads see new data
        session.commit("Add second date")
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda _: open_store(store_path).sizes["time"], range(3)))
        assert all(r == 2 for r in results)


class TestCleanupOnFailure:
    """Tests for the cleanup_on_failure decorator."""

    def test_failed_write_cleans_up_partial_store(self, local_zarr_path, sample_reflectance_data, monkeypatch):
        """Store path should be deleted if write fails after repo creation."""
        store_path = str(local_zarr_path / "fail_test" / "reflectance.zarr")
        data = sample_reflectance_data(["2024-01-01"], height=64, width=64)
        data.attrs["tile_id"] = "33UUP"

        # Force to_icechunk to fail after repo is created
        def failing_to_icechunk(*args, **kwargs):
            raise RuntimeError("Simulated write failure")

        monkeypatch.setattr("tessera_embeddings.storage.zarr_store.to_icechunk", failing_to_icechunk)

        with pytest.raises(RuntimeError, match="Simulated write failure"):
            _write_new(store_path, data, None, "test")

        # Store directory should not exist after cleanup
        assert not Path(store_path).exists()

    def test_successful_write_preserves_store(self, local_zarr_path, sample_reflectance_data):
        """Successful writes should leave the store intact."""
        store_path = str(local_zarr_path / "success_test" / "reflectance.zarr")
        data = sample_reflectance_data(["2024-01-01"], height=64, width=64)
        _write_reflectance(store_path, data, tile_id="33UUP", baselines={"2024-01-01": 400})

        assert Path(store_path).exists()
        assert open_store(store_path).attrs["tile_id"] == "33UUP"


class TestComputeDoy:
    """Tests for DOY (day-of-year) computation."""

    def test_computes_correct_doy_for_known_dates(self):
        """Known dates should produce correct DOY values."""
        timestamps = np.array(
            [
                "2024-01-01",
                "2024-06-15",
                "2024-12-31",
            ],
            dtype="datetime64[ns]",
        )

        doy = compute_doy(timestamps)

        assert doy[0] == 1  # Jan 1
        assert doy[1] == 167  # June 15 (2024 is leap year)
        assert doy[2] == 366  # Dec 31 (leap year)

    def test_returns_int32_array(self):
        """DOY array should be int32."""
        timestamps = np.array(["2024-03-01"], dtype="datetime64[ns]")
        doy = compute_doy(timestamps)
        assert doy.dtype == np.int32

    def test_non_leap_year(self):
        """Dec 31 in non-leap year should be 365."""
        timestamps = np.array(["2023-12-31"], dtype="datetime64[ns]")
        doy = compute_doy(timestamps)
        assert doy[0] == 365


class TestWriteDataset:
    """Tests for the generalized write_dataset function."""

    def test_write_dataset_splits_chunks_smaller_than_extent(self, local_zarr_path, sample_reflectance_data):
        """A chunk size smaller than the data extent produces multiple spatial chunks."""
        dates = ["2024-01-01"]
        # 1000x1000 data with 500x500 chunks → 2 chunks per spatial dim.
        data = sample_reflectance_data(dates, height=1000, width=1000)
        store_path = str(local_zarr_path / "split_chunks" / "reflectance.zarr")

        write_dataset(
            store_path,
            data,
            tile_id="33UUP",
            baselines=dict.fromkeys(dates, 400),
            chunks=SPLIT_CHUNKS,
            crs="EPSG:32633",
        )

        ds = open_store(store_path)

        assert ds["blue"].shape == (1, 1000, 1000)
        # Not clamped to extent: the spatial dims are tiled into 500-px chunks.
        assert ds["blue"].chunks[1] == (500, 500)
        assert ds["blue"].chunks[2] == (500, 500)
        # Data still round-trips intact across the chunk boundaries.
        np.testing.assert_array_equal(ds["blue"].values, data["blue"].values)

    def test_write_dataset_clamps_chunks_to_extent(self, local_zarr_path, sample_reflectance_data):
        """write_dataset clamps spatial chunks to data extent when smaller than chunk size."""
        dates = ["2024-01-01"]
        data = sample_reflectance_data(dates, height=256, width=256)
        store_path = str(local_zarr_path / "default_chunks" / "reflectance.zarr")

        write_dataset(
            store_path,
            data,
            tile_id="33UUP",
            baselines=dict.fromkeys(dates, 400),
            chunks=DEFAULT_CHUNKS,
            crs="EPSG:32633",
        )

        ds = open_store(store_path)
        # Full spatial extent since 256 < chunk size (10980)
        assert ds["blue"].chunks[1][0] == 256
        assert ds["blue"].chunks[2][0] == 256

    def test_write_dataset_stores_doy(self, local_zarr_path, sample_reflectance_data):
        """write_dataset should store DOY as attribute."""
        dates = ["2024-01-01", "2024-06-15"]
        data = sample_reflectance_data(dates, height=64, width=64)
        store_path = str(local_zarr_path / "doy_test" / "reflectance.zarr")

        write_dataset(
            store_path,
            data,
            tile_id="33UUP",
            baselines=dict.fromkeys(dates, 400),
            chunks=DEFAULT_CHUNKS,
            crs="EPSG:32633",
        )

        ds = open_store(store_path)
        doy = ds.attrs["doy"]
        assert doy[0] == 1  # Jan 1
        assert doy[1] == 167  # June 15 (2024 is leap year)

    def test_write_dataset_append_merges_doy(self, local_zarr_path, sample_reflectance_data):
        """Appending data should merge DOY values."""
        store_path = str(local_zarr_path / "doy_append" / "reflectance.zarr")

        # Create initial store
        data1 = sample_reflectance_data(["2024-01-01"], height=64, width=64)
        write_dataset(
            store_path, data1, tile_id="33UUP", baselines={"2024-01-01": 400}, chunks=DEFAULT_CHUNKS, crs="EPSG:32633"
        )

        # Append
        data2 = sample_reflectance_data(["2024-06-15"], height=64, width=64, seed=99)
        write_dataset(
            store_path, data2, tile_id="33UUP", baselines={"2024-06-15": 400}, chunks=DEFAULT_CHUNKS, crs="EPSG:32633"
        )

        ds = open_store(store_path)
        doy = ds.attrs["doy"]
        assert len(doy) == 2
        assert doy[0] == 1  # Jan 1
        assert doy[1] == 167  # June 15


# Single-block spatial chunking: the whole sub-box is one dask chunk, so the
# RMW pad / region write decides the on-disk layout, not the input chunking.
ONE_BLOCK = {"time": 1, "northing": -1, "easting": -1}


def _single_date_block(date, value, *, height, width, n0=0, e0=0, chunks=None):
    """Build a one-timestep ``blue`` dataset filled with ``value``.

    ``n0``/``e0`` offset the spatial coords so a sub-box carries the right
    coordinate labels (write_region ignores them, but they keep the dataset
    well-formed and document intent). ``chunks`` optionally rechunks.
    """
    ds = xr.Dataset(
        {"blue": (["time", "northing", "easting"], np.full((1, height, width), value, dtype=np.uint16))},
        coords={
            "time": [np.datetime64(date, "ns")],
            "northing": np.arange(n0, n0 + height),
            "easting": np.arange(e0, e0 + width),
        },
    )
    return ds.chunk(chunks) if chunks else ds


def _make_region_store(local_zarr_path, sample_reflectance_data, dates, *, chunks=SPLIT_CHUNKS, name="rw"):
    """Write a 1000x1000 reflectance store under ``name`` and return its path."""
    data = sample_reflectance_data(dates, height=1000, width=1000)
    store_path = str(local_zarr_path / name / "reflectance.zarr")
    write_dataset(
        store_path, data, tile_id="33UUP", baselines=dict.fromkeys(dates, 400), chunks=chunks, crs="EPSG:32615"
    )
    return store_path


class TestResolveRegion:
    """Tests for coordinate-range -> integer-slice resolution."""

    def _store(self, local_zarr_path, sample_reflectance_data, dates):
        return _make_region_store(local_zarr_path, sample_reflectance_data, dates, name="resolve")

    def test_single_date_resolves_to_one_index(self, local_zarr_path, sample_reflectance_data):
        dates = ["2024-01-01", "2024-01-06", "2024-01-11"]
        store_path = self._store(local_zarr_path, sample_reflectance_data, dates)
        assert resolve_region(store_path, time=("2024-01-06", "2024-01-06")) == {"time": slice(1, 2)}

    def test_date_range_resolves_to_half_open_slice(self, local_zarr_path, sample_reflectance_data):
        dates = ["2024-01-01", "2024-01-06", "2024-01-11"]
        store_path = self._store(local_zarr_path, sample_reflectance_data, dates)
        assert resolve_region(store_path, time=("2024-01-01", "2024-01-06")) == {"time": slice(0, 2)}

    def test_spatial_range_resolves_against_coords(self, local_zarr_path, sample_reflectance_data):
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"])
        region = resolve_region(store_path, northing=(100, 299), easting=(250, 609))
        assert region == {"northing": slice(100, 300), "easting": slice(250, 610)}

    def test_none_axis_is_omitted(self, local_zarr_path, sample_reflectance_data):
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01", "2024-01-06"])
        region = resolve_region(store_path, time=("2024-01-01", "2024-01-01"))
        assert set(region) == {"time"}

    def test_nonexistent_date_raises(self, local_zarr_path, sample_reflectance_data):
        """The overwrite-in-place contract: a date not in the store is rejected."""
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"])
        with pytest.raises(ValueError, match="selects no existing coordinate"):
            resolve_region(store_path, time=("2025-01-01", "2025-01-01"))

    def test_non_contiguous_range_raises(self, local_zarr_path, sample_reflectance_data):
        """An out-of-order time axis whose range straddles a gap is rejected."""

        # write_dataset appends without sorting, so append out of order to build
        # the axis [2024-01-06, 2024-01-01, 2024-01-20].
        def _append(date):
            ds = sample_reflectance_data([date], height=1000, width=1000)
            write_dataset(store_path, ds, tile_id="33UUP", baselines={date: 400}, chunks=SPLIT_CHUNKS, crs="EPSG:32615")

        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-06"])
        _append("2024-01-01")
        _append("2024-01-20")

        # Range 01-06..01-20 matches indices 0 and 2 — non-contiguous (gap at 1).
        with pytest.raises(ValueError, match="non-contiguous"):
            resolve_region(store_path, time=("2024-01-06", "2024-01-20"))


class TestPadRegionToChunks:
    """Unit tests for the read-modify-write chunk padding (no store I/O)."""

    def _existing(self, *, height=1000, width=1000, chunk=500):
        ds = xr.Dataset(
            {"blue": (["time", "northing", "easting"], np.zeros((1, height, width), dtype=np.uint16))},
            coords={
                "time": [np.datetime64("2024-01-01", "ns")],
                "northing": np.arange(height),
                "easting": np.arange(width),
            },
        )
        return ds.chunk({"time": 1, "northing": chunk, "easting": chunk})

    def test_aligned_region_needs_no_store_read(self):
        """A chunk-aligned region returns the incoming values unchanged (no store read).

        The data is shape-validated and matched to the store's dim order, but the
        values pass straight through — no shell slab is read or overlaid.
        """
        existing = self._existing()
        data = _single_date_block("2024-01-01", 7, height=500, width=500, chunks=ONE_BLOCK)
        region = {"northing": slice(0, 500), "easting": slice(500, 1000)}
        padded, widened = _pad_region_to_chunks(existing, data, region)
        assert widened == region
        assert padded["blue"].dims == ("time", "northing", "easting")
        assert (padded["blue"].values == 7).all()

    def test_aligned_region_rejects_mismatched_shape(self):
        """The aligned fast path still validates shape (no silent broadcast)."""
        existing = self._existing()
        # region spans 500 northing rows but data supplies 1 -> would broadcast.
        data = _single_date_block("2024-01-01", 7, height=1, width=500, e0=500, chunks=ONE_BLOCK)
        region = {"northing": slice(0, 500), "easting": slice(500, 1000)}
        with pytest.raises(ValueError, match="expected"):
            _pad_region_to_chunks(existing, data, region)

    def test_unaligned_region_widens_to_chunk_bounds(self):
        existing = self._existing()
        data = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"northing": slice(100, 300), "easting": slice(250, 610)}
        _padded, widened = _pad_region_to_chunks(existing, data, region)
        # northing 100:300 -> 0:500 ; easting 250:610 -> 0:1000 (both edges snap out)
        assert widened == {"northing": slice(0, 500), "easting": slice(0, 1000)}

    def test_padded_overlay_preserves_shell_and_writes_data(self):
        """Widened frame holds incoming values inside the region, zeros (store) outside."""
        existing = self._existing()
        data = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"northing": slice(100, 300), "easting": slice(250, 610)}
        padded, widened = _pad_region_to_chunks(existing, data, region)
        arr = padded["blue"].values
        # region cells -> 9 (positions relative to the widened frame, which starts at 0,0)
        assert (arr[0, 100:300, 250:610] == 9).all()
        # everything else in the widened frame -> shell value (0)
        mask = np.ones(arr.shape[1:], dtype=bool)
        mask[100:300, 250:610] = False
        assert (arr[0][mask] == 0).all()

    def test_open_bound_slice_is_normalized(self):
        """slice(None) on a padded dim resolves to the full axis, no None arithmetic."""
        existing = self._existing()
        # northing full axis (slice(None)); easting unaligned so padding kicks in.
        data = _single_date_block("2024-01-01", 4, height=1000, width=360, e0=250, chunks=ONE_BLOCK)
        region = {"northing": slice(None), "easting": slice(250, 610)}
        padded, widened = _pad_region_to_chunks(existing, data, region)
        assert widened == {"northing": slice(0, 1000), "easting": slice(0, 1000)}
        assert (padded["blue"].values[0, :, 250:610] == 4).all()

    def test_mismatched_shape_is_rejected(self):
        """A region-covering promise is enforced: a too-small input can't broadcast."""
        existing = self._existing()
        # region spans 200 northing rows but data only supplies 1 -> would broadcast.
        data = _single_date_block("2024-01-01", 9, height=1, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"northing": slice(100, 300), "easting": slice(250, 610)}
        with pytest.raises(ValueError, match="expected"):
            _pad_region_to_chunks(existing, data, region)

    def test_non_unit_step_is_rejected(self):
        existing = self._existing()
        data = _single_date_block("2024-01-01", 9, height=250, width=500, chunks=ONE_BLOCK)
        with pytest.raises(ValueError, match="step 1"):
            _pad_region_to_chunks(existing, data, {"northing": slice(0, 500, 2)})

    def test_transposed_input_is_realigned(self):
        """Input with swapped northing/easting is transposed to store order, not corrupted."""
        existing = self._existing()
        # Build data with easting before northing, unaligned so padding runs.
        arr = np.full((1, 360, 200), 6, dtype=np.uint16)  # (time, easting, northing)
        data = xr.Dataset(
            {"blue": (["time", "easting", "northing"], arr)},
            coords={"time": [np.datetime64("2024-01-01", "ns")]},
        ).chunk(ONE_BLOCK)
        region = {"northing": slice(100, 300), "easting": slice(250, 610)}
        padded, _ = _pad_region_to_chunks(existing, data, region)
        assert padded["blue"].dims == ("time", "northing", "easting")
        assert (padded["blue"].values[0, 100:300, 250:610] == 6).all()


class TestWriteRegion:
    """End-to-end region overwrite tests."""

    def _store(self, local_zarr_path, sample_reflectance_data, dates, *, name="rw"):
        return _make_region_store(local_zarr_path, sample_reflectance_data, dates, name=name)

    def test_overwrite_full_timestep(self, local_zarr_path, sample_reflectance_data):
        """Overwriting one date leaves the others byte-for-byte intact."""
        dates = ["2024-01-01", "2024-01-06", "2024-01-11"]
        store_path = self._store(local_zarr_path, sample_reflectance_data, dates)
        original = open_store(store_path)["blue"].values.copy()

        new = _single_date_block("2024-01-06", 7, height=1000, width=1000, chunks=SPLIT_CHUNKS)
        region = resolve_region(store_path, time=("2024-01-06", "2024-01-06"))
        write_region(store_path, new, region=region)

        back = open_store(store_path)["blue"].values
        assert (back[1] == 7).all()
        np.testing.assert_array_equal(back[0], original[0])
        np.testing.assert_array_equal(back[2], original[2])

    def test_overwrite_aligned_spatial_box(self, local_zarr_path, sample_reflectance_data):
        """A chunk-aligned spatial sub-box overwrites only its cells."""
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"], name="rw_aligned")
        original = open_store(store_path)["blue"].values.copy()

        sub = _single_date_block("2024-01-01", 5, height=500, width=500, n0=500, e0=0, chunks=ONE_BLOCK)
        region = {"time": slice(0, 1), "northing": slice(500, 1000), "easting": slice(0, 500)}
        write_region(store_path, sub, region=region)

        back = open_store(store_path)["blue"].values
        assert (back[0, 500:1000, 0:500] == 5).all()
        mask = np.ones(back.shape[1:], dtype=bool)
        mask[500:1000, 0:500] = False
        np.testing.assert_array_equal(back[0][mask], original[0][mask])

    def test_overwrite_unaligned_spatial_box_backfills_neighbors(self, local_zarr_path, sample_reflectance_data):
        """An unaligned box is padded; cells sharing its boundary chunks survive."""
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"], name="rw_unaligned")
        original = open_store(store_path)["blue"].values.copy()

        sub = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"time": slice(0, 1), "northing": slice(100, 300), "easting": slice(250, 610)}
        write_region(store_path, sub, region=region)

        back = open_store(store_path)["blue"].values
        assert (back[0, 100:300, 250:610] == 9).all()
        mask = np.ones(back.shape[1:], dtype=bool)
        mask[100:300, 250:610] = False
        np.testing.assert_array_equal(back[0][mask], original[0][mask])

    def test_region_write_preserves_root_attrs(self, local_zarr_path, sample_reflectance_data):
        """CRS / tile_id and other root attrs survive a region write."""
        dates = ["2024-01-01", "2024-01-06"]
        store_path = self._store(local_zarr_path, sample_reflectance_data, dates, name="rw_attrs")

        new = _single_date_block("2024-01-06", 3, height=1000, width=1000, chunks=SPLIT_CHUNKS)
        write_region(store_path, new, region=resolve_region(store_path, time=("2024-01-06", "2024-01-06")))

        ds = open_store(store_path)
        assert ds.attrs["crs"] == "EPSG:32615"
        assert ds.attrs["tile_id"] == "33UUP"

    def test_update_attrs_merged(self, local_zarr_path, sample_reflectance_data):
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"], name="rw_update_attrs")
        new = _single_date_block("2024-01-01", 1, height=1000, width=1000, chunks=SPLIT_CHUNKS)
        write_region(
            store_path,
            new,
            region={"time": slice(0, 1)},
            update_attrs={"reprocessed": "2026-06-15"},
        )
        assert open_store(store_path).attrs["reprocessed"] == "2026-06-15"

    def test_empty_region_rejected(self, local_zarr_path, sample_reflectance_data):
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"], name="rw_empty")
        new = _single_date_block("2024-01-01", 1, height=1000, width=1000, chunks=SPLIT_CHUNKS)
        with pytest.raises(ValueError, match="non-empty region"):
            write_region(store_path, new, region={})

    def test_failed_region_write_leaves_prior_commit_intact(
        self, local_zarr_path, sample_reflectance_data, monkeypatch
    ):
        """If to_icechunk raises mid-write, the last committed state is unchanged."""
        dates = ["2024-01-01", "2024-01-06"]
        store_path = self._store(local_zarr_path, sample_reflectance_data, dates, name="rw_fail")
        original = open_store(store_path)["blue"].values.copy()

        def failing_to_icechunk(*args, **kwargs):
            raise RuntimeError("Simulated region write failure")

        monkeypatch.setattr("tessera_embeddings.storage.zarr_store.to_icechunk", failing_to_icechunk)

        new = _single_date_block("2024-01-06", 7, height=1000, width=1000, chunks=SPLIT_CHUNKS)
        with pytest.raises(RuntimeError, match="Simulated region write failure"):
            write_region(store_path, new, region=resolve_region(store_path, time=("2024-01-06", "2024-01-06")))

        np.testing.assert_array_equal(open_store(store_path)["blue"].values, original)


class TestDefaultRepoConfig:
    """The chunk cache is always applied; max_concurrent_requests stays optional.

    Striped inference re-reads the same (time=1, 4000, 4000) store chunks once
    per northing strip, so the cache must be set on every repo open — including
    the common path where no concurrency cap is passed (the old early-return
    bug dropped the cache entirely in that case).
    """

    def test_cache_applied_without_concurrency_cap(self):
        config = _default_repo_config()
        assert config is not None
        assert config.caching is not None
        assert config.caching.num_bytes_chunks == _CHUNK_CACHE_BYTES

    def test_cache_applied_with_concurrency_cap(self):
        config = _default_repo_config(max_concurrent_requests=64)
        assert config.caching is not None
        assert config.caching.num_bytes_chunks == _CHUNK_CACHE_BYTES
        assert config.max_concurrent_requests == 64

    def test_returns_config_not_none_by_default(self):
        # Previously returned None when no cap was passed; must now always build one.
        assert _default_repo_config() is not None
