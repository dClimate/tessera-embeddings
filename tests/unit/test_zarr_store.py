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

import icechunk
import numpy as np
import pytest
import xarray as xr
from icechunk.xarray import to_icechunk

from tessera_embeddings.config import S2_L2A_BANDS
from tessera_embeddings.storage.region_writes import _aligned_region_sources, _pad_region_to_chunks
from tessera_embeddings.storage.zarr_store import (
    S3Config,
    _default_repo_config,
    _open_repo,
    _open_writable_session,
    _write_new,
    compute_doy,
    get_existing_dates,
    open_store,
    open_store_as_zarr_group,
    resolve_region,
    rollback_commits,
    set_s3_config,
    write_dataset,
    write_region,
    write_regions,
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


class TestRollbackCommits:
    """Tests for rolling a branch's HEAD back by n commits."""

    def _build_history(self, store_path, sample_reflectance_data, dates):
        """Write one create + (len(dates)-1) appends, one commit per date."""
        first, *rest = dates
        data = sample_reflectance_data([first], height=64, width=64, seed=0)
        _write_reflectance(store_path, data, tile_id="33UUP", baselines={first: 400})
        for i, date in enumerate(rest, start=1):
            data = sample_reflectance_data([date], height=64, width=64, seed=i)
            _write_reflectance(store_path, data, tile_id="33UUP", baselines={date: 400})

    def test_rollback_drops_last_commits(self, local_zarr_path, sample_reflectance_data):
        """Rolling back n appends restores the time dim to its earlier length."""
        store_path = str(local_zarr_path / "rb" / "reflectance.zarr")
        dates = ["2024-01-01", "2024-01-06", "2024-01-11", "2024-01-16"]
        self._build_history(store_path, sample_reflectance_data, dates)
        assert open_store(store_path).sizes["time"] == 4

        rollback_commits(store_path, 2)

        ds = open_store(store_path)
        assert ds.sizes["time"] == 2
        assert get_existing_dates(store_path) == set(dates[:2])

    def test_rollback_returns_target_snapshot_id(self, local_zarr_path, sample_reflectance_data):
        """The returned id matches the snapshot n commits back from HEAD."""
        store_path = str(local_zarr_path / "rb_id" / "reflectance.zarr")
        self._build_history(store_path, sample_reflectance_data, ["2024-01-01", "2024-01-06", "2024-01-11"])

        repo = _open_repo(store_path)
        history = list(repo.ancestry(branch="main"))
        expected_target = history[1].id  # one commit back

        new_head = rollback_commits(store_path, 1)

        assert new_head == expected_target
        assert _open_repo(store_path).lookup_branch("main") == expected_target

    def test_dry_run_does_not_move_branch(self, local_zarr_path, sample_reflectance_data):
        """dry_run resolves the target id but leaves HEAD untouched."""
        store_path = str(local_zarr_path / "rb_dry" / "reflectance.zarr")
        self._build_history(store_path, sample_reflectance_data, ["2024-01-01", "2024-01-06", "2024-01-11"])

        head_before = _open_repo(store_path).lookup_branch("main")

        target = rollback_commits(store_path, 1, dry_run=True)

        assert _open_repo(store_path).lookup_branch("main") == head_before
        assert target != head_before
        assert open_store(store_path).sizes["time"] == 3

    def test_rollback_is_reversible(self, local_zarr_path, sample_reflectance_data):
        """The dropped snapshots survive, so resetting back restores the data."""
        store_path = str(local_zarr_path / "rb_rev" / "reflectance.zarr")
        self._build_history(store_path, sample_reflectance_data, ["2024-01-01", "2024-01-06", "2024-01-11"])

        original_head = _open_repo(store_path).lookup_branch("main")
        rollback_commits(store_path, 2)
        assert open_store(store_path).sizes["time"] == 1

        _open_repo(store_path).reset_branch("main", original_head)
        assert open_store(store_path).sizes["time"] == 3

    def test_n_below_one_raises(self, local_zarr_path, sample_reflectance_data):
        store_path = str(local_zarr_path / "rb_zero" / "reflectance.zarr")
        self._build_history(store_path, sample_reflectance_data, ["2024-01-01", "2024-01-06"])
        with pytest.raises(ValueError, match="n must be >= 1"):
            rollback_commits(store_path, 0)

    def test_rollback_past_root_raises(self, local_zarr_path, sample_reflectance_data):
        """Cannot drop the entire history, including the root commit."""
        store_path = str(local_zarr_path / "rb_root" / "reflectance.zarr")
        self._build_history(store_path, sample_reflectance_data, ["2024-01-01", "2024-01-06"])
        # 2 commits (create + 1 append) + the icechunk root => 3 snapshots.
        with pytest.raises(ValueError, match="only"):
            rollback_commits(store_path, 3)


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

    def _group(self, existing):
        """A raw zarr group mirroring ``existing`` (shape/chunks/dtype/values).

        The padding shell now reads straight from the raw zarr group rather than
        ``existing.isel(...)``, so these unit tests need an on-disk-style group
        backing the in-memory ``existing``. Values match (all zeros), so the
        shell-preservation assertions are unchanged.
        """
        import zarr

        store = zarr.storage.MemoryStore()
        g = zarr.open_group(store, mode="w")
        for name in existing.data_vars:
            var = existing[str(name)]
            chunks = tuple(c[0] for c in var.chunks)  # nominal chunk size per dim
            arr = g.create_array(str(name), shape=var.shape, chunks=chunks, dtype=var.dtype)
            arr[:] = var.values
        return zarr.open_group(store, mode="r")

    def test_aligned_region_needs_no_store_read(self):
        """A chunk-aligned region returns the incoming values unchanged (no store read).

        The data is shape-validated and matched to the store's dim order, but the
        values pass straight through — no shell slab is read or overlaid.
        """
        existing = self._existing()
        data = _single_date_block("2024-01-01", 7, height=500, width=500, chunks=ONE_BLOCK)
        region = {"northing": slice(0, 500), "easting": slice(500, 1000)}
        padded, widened = _pad_region_to_chunks(existing, data, region, self._group(existing))
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
            _pad_region_to_chunks(existing, data, region, self._group(existing))

    def test_unaligned_region_widens_to_chunk_bounds(self):
        existing = self._existing()
        data = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"northing": slice(100, 300), "easting": slice(250, 610)}
        _padded, widened = _pad_region_to_chunks(existing, data, region, self._group(existing))
        # northing 100:300 -> 0:500 ; easting 250:610 -> 0:1000 (both edges snap out)
        assert widened == {"northing": slice(0, 500), "easting": slice(0, 1000)}

    def test_padded_overlay_preserves_shell_and_writes_data(self):
        """Widened frame holds incoming values inside the region, zeros (store) outside."""
        existing = self._existing()
        data = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"northing": slice(100, 300), "easting": slice(250, 610)}
        padded, widened = _pad_region_to_chunks(existing, data, region, self._group(existing))
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
        padded, widened = _pad_region_to_chunks(existing, data, region, self._group(existing))
        assert widened == {"northing": slice(0, 1000), "easting": slice(0, 1000)}
        assert (padded["blue"].values[0, :, 250:610] == 4).all()

    def test_mismatched_shape_is_rejected(self):
        """A region-covering promise is enforced: a too-small input can't broadcast."""
        existing = self._existing()
        # region spans 200 northing rows but data only supplies 1 -> would broadcast.
        data = _single_date_block("2024-01-01", 9, height=1, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"northing": slice(100, 300), "easting": slice(250, 610)}
        with pytest.raises(ValueError, match="expected"):
            _pad_region_to_chunks(existing, data, region, self._group(existing))

    def test_non_unit_step_is_rejected(self):
        existing = self._existing()
        data = _single_date_block("2024-01-01", 9, height=250, width=500, chunks=ONE_BLOCK)
        with pytest.raises(ValueError, match="step 1"):
            _pad_region_to_chunks(existing, data, {"northing": slice(0, 500, 2)}, self._group(existing))

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
        padded, _ = _pad_region_to_chunks(existing, data, region, self._group(existing))
        assert padded["blue"].dims == ("time", "northing", "easting")
        assert (padded["blue"].values[0, 100:300, 250:610] == 6).all()

    def _existing_nt(self, *, nt, height=100, width=100, chunk=50):
        """An ``existing`` whose time axis is ``nt`` long (time chunk 1).

        Kept small spatially: the graph-size property is independent of spatial
        extent, and a long time axis must not balloon memory just to assert it.
        """
        ds = xr.Dataset(
            {"blue": (["time", "northing", "easting"], np.zeros((nt, height, width), dtype=np.uint16))},
            coords={
                "time": np.array([np.datetime64("2024-01-01", "ns") + np.timedelta64(i, "D") for i in range(nt)]),
                "northing": np.arange(height),
                "easting": np.arange(width),
            },
        )
        return ds.chunk({"time": 1, "northing": chunk, "easting": chunk})

    @staticmethod
    def _task_groups(graph):
        """Count dask graph keys by task-group prefix (the bit before the token)."""
        from collections import Counter

        groups: Counter[str] = Counter()
        for key in graph:
            name = key[0] if isinstance(key, tuple) else str(key)
            groups[name.split("-")[0] if isinstance(name, str) else name] += 1
        return groups

    def test_padded_shell_graph_is_slab_local_and_flat_in_time(self):
        """The grid-tiled source's dask graph must scale with the region's grid chunks,
        not the store's time length — the property that keeps the source flat in master
        time and kills the scheduler OOM. Reading the backfill through the lazy
        ``existing`` view would carry the whole-store time-chunk layer into it, so the
        graph would grow with the master's total dates; a value-parity test can't catch
        that, only a graph-size check independent of ``nt`` can. The same check also
        guards that the grid-tiled construction emits NO all-to-all ``rechunk`` /
        ``concatenate`` groups (the scheduler-bound ops the off-grid intermediate +
        downstream rechunk used to dominate) and stays ``O(grid chunks)``.
        """
        # 100x100 store at 50-px chunks; region 10:30 x 10:70 widens to a 1x1x2 grid slab.
        region = {"time": slice(0, 1), "northing": slice(10, 30), "easting": slice(10, 70)}
        data = _single_date_block("2024-01-01", 9, height=20, width=60, n0=10, e0=10, chunks=ONE_BLOCK)

        sizes = {}
        groups_by_nt = {}
        for nt in (5, 500):
            existing = self._existing_nt(nt=nt)
            padded, widened = _pad_region_to_chunks(existing, data, region, self._group(existing))
            assert widened["northing"] == slice(0, 50) and widened["easting"] == slice(0, 100)
            graph = padded["blue"].data.__dask_graph__()
            sizes[nt] = len(graph)
            groups_by_nt[nt] = self._task_groups(graph)

        # The source is one task per grid chunk over the widened bounds (here 1x1x2),
        # each a store read, an incoming block, or a boundary blend — a small constant
        # set by the grid, not the slab area or master time. It must NOT grow from nt=5
        # to nt=500; a lazy-view backfill read would scale with nt.
        assert sizes[5] == sizes[500], f"source graph scales with store time length: {sizes}"
        assert sizes[5] < 100, f"source graph has {sizes[5]} keys — not grid-local"

        # The off-grid concat + grid rechunk are gone: no scheduler-bound all-to-all
        # groups remain in the source graph (the slicing of the incoming insert may
        # leave plain ``getitem`` splits, which are embarrassingly parallel, not
        # all-to-all).
        for nt, groups in groups_by_nt.items():
            assert "concatenate" not in groups, f"nt={nt}: source graph still has concatenate: {dict(groups)}"
            assert "rechunk" not in groups, f"nt={nt}: source graph still has rechunk: {dict(groups)}"

    def test_zarr_direct_shell_matches_isel(self):
        """The zarr-direct shell is numerically identical to the old
        ``existing.isel(widened)`` slab — same backfilled store values outside the
        region, same overlaid incoming inside — so output bytes are unchanged.
        """
        rng = np.random.default_rng(0)
        # Non-zero store values so an isel/zarr-direct divergence would actually show.
        store_vals = rng.integers(0, 9000, size=(1, 1000, 1000), dtype=np.uint16)
        existing = xr.Dataset(
            {"blue": (["time", "northing", "easting"], store_vals)},
            coords={
                "time": [np.datetime64("2024-01-01", "ns")],
                "northing": np.arange(1000),
                "easting": np.arange(1000),
            },
        ).chunk({"time": 1, "northing": 500, "easting": 500})
        group = self._group(existing)

        data = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"northing": slice(100, 300), "easting": slice(250, 610)}
        padded, widened = _pad_region_to_chunks(existing, data, region, group)

        # Oracle: the pre-zarr-direct construction (isel shell + positional overlay).
        ref = existing.isel(widened)["blue"].data.copy()
        n0 = region["northing"].start - widened["northing"].start
        e0 = region["easting"].start - widened["easting"].start
        idx = (
            slice(None),
            slice(n0, n0 + (region["northing"].stop - region["northing"].start)),
            slice(e0, e0 + (region["easting"].stop - region["easting"].start)),
        )
        ref[idx] = data["blue"].transpose("time", "northing", "easting").data
        np.testing.assert_array_equal(padded["blue"].values, ref.compute())

    def test_grid_tiled_boundary_blend_multi_time(self):
        """Multi-time unaligned region: every cell matches a per-pixel oracle.

        The store is filled with spatially- AND temporally-varying values, so a
        boundary chunk's blend that mis-placed the incoming overlay (wrong offset,
        wrong axis, or wrong time index) would leave the wrong store value showing
        and fail the assert. A run spanning several time indices (each its own time
        chunk) exercises the time axis as an outer block dim.
        """
        rng = np.random.default_rng(7)
        nt = 5
        store_vals = rng.integers(0, 9000, size=(nt, 60, 60), dtype=np.uint16)
        existing = xr.Dataset(
            {"blue": (["time", "northing", "easting"], store_vals)},
            coords={
                "time": [np.datetime64("2024-01-01", "ns") + np.timedelta64(i, "D") for i in range(nt)],
                "northing": np.arange(60),
                "easting": np.arange(60),
            },
        ).chunk({"time": 1, "northing": 20, "easting": 20})
        group = self._group(existing)

        # 3-date run on a box unaligned on all four spatial faces (13:47 x 7:51).
        t0, run = 1, 3
        inc = rng.integers(0, 9000, size=(run, 34, 44), dtype=np.uint16)
        data = xr.Dataset(
            {"blue": (["time", "northing", "easting"], inc)},
            coords={"time": [existing.time.values[t0 + i] for i in range(run)]},
        ).chunk(ONE_BLOCK)
        region = {"time": slice(t0, t0 + run), "northing": slice(13, 47), "easting": slice(7, 51)}

        padded, widened = _pad_region_to_chunks(existing, data, region, group)
        assert widened == {
            "time": slice(t0, t0 + run),
            "northing": slice(0, 60),
            "easting": slice(0, 60),
        }

        # Per-pixel oracle: store values over the widened box, with incoming overlaid
        # on exactly the region cells (relative to the widened origin).
        ref = store_vals[t0 : t0 + run, 0:60, 0:60].copy()
        ref[:, 13:47, 7:51] = inc
        np.testing.assert_array_equal(padded["blue"].values, ref)

    def test_guard_rejects_transposed_zarr_axis_order(self):
        """The shell indexes the raw zarr array positionally; if its physical axis
        order disagrees with the store view's named dims the guard must fire, since
        the slices would otherwise land on the wrong axes and silently corrupt the
        backfill.
        """
        import zarr

        existing = self._existing(height=1000, width=1000, chunk=500)
        # A zarr array whose physical axes are (time, EASTING, NORTHING) — swapped
        # relative to the view's (time, northing, easting) — but tagged with the
        # view's dim names, so positional indexing would mismatch.
        store = zarr.storage.MemoryStore()
        g = zarr.open_group(store, mode="w")
        g.create_array(
            "blue",
            shape=(1, 1000, 1000),
            chunks=(1, 500, 500),
            dtype="uint16",
            dimension_names=("time", "easting", "northing"),
        )
        group = zarr.open_group(store, mode="r")

        data = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"northing": slice(100, 300), "easting": slice(250, 610)}
        with pytest.raises(ValueError, match="axis order"):
            _pad_region_to_chunks(existing, data, region, group)

    def test_guard_rejects_chunk_grid_mismatch(self):
        """If the raw zarr array's chunk grid disagrees with the store view's, the
        widened region (computed from the view) would not align to the shell blocks;
        the guard must reject it rather than emit a mis-aligned write.
        """
        import zarr

        existing = self._existing(height=1000, width=1000, chunk=500)
        # Same shape/dims as the view, but a different chunk grid (250 vs 500).
        store = zarr.storage.MemoryStore()
        g = zarr.open_group(store, mode="w")
        g.create_array(
            "blue",
            shape=(1, 1000, 1000),
            chunks=(1, 250, 250),
            dtype="uint16",
            dimension_names=("time", "northing", "easting"),
        )
        group = zarr.open_group(store, mode="r")

        data = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"northing": slice(100, 300), "easting": slice(250, 610)}
        with pytest.raises(ValueError, match="Chunk grid"):
            _pad_region_to_chunks(existing, data, region, group)


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

    def test_unaligned_region_write_under_distributed_client(self, local_zarr_path, sample_reflectance_data):
        """An unaligned region write succeeds with a distributed client active.

        Padding builds a lazy dask shell read from the store; ``to_icechunk``
        pickles that source graph to the workers. Reading the shell from the
        *writable* session would raise "must opt-in to pickle writable
        sessions"; reading it from a readonly session pickles cleanly. This
        guards that the read step uses a readonly session.
        """
        distributed = pytest.importorskip("distributed")
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"], name="rw_distributed")
        original = open_store(store_path)["blue"].values.copy()

        # Deliberately unaligned (mid-chunk) box so the padding/read path runs.
        sub = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        region = {"time": slice(0, 1), "northing": slice(100, 300), "easting": slice(250, 610)}

        with distributed.Client(processes=True, n_workers=2, threads_per_worker=1, dashboard_address=":0") as client:
            write_region(store_path, sub, region=region)

        back = open_store(store_path)["blue"].values
        assert (back[0, 100:300, 250:610] == 9).all()
        mask = np.ones(back.shape[1:], dtype=bool)
        mask[100:300, 250:610] = False
        np.testing.assert_array_equal(back[0][mask], original[0][mask])

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


class TestWriteRegions:
    """Batch region overwrite: many regions, one distributed compute, one commit."""

    def _store(self, local_zarr_path, sample_reflectance_data, dates, *, name="rws"):
        return _make_region_store(local_zarr_path, sample_reflectance_data, dates, name=name)

    def test_rejects_empty_items(self, local_zarr_path, sample_reflectance_data):
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"], name="rws_empty_items")
        with pytest.raises(ValueError, match="at least one"):
            write_regions(store_path, [])

    def test_rejects_empty_region_item(self, local_zarr_path, sample_reflectance_data):
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"], name="rws_empty_region")
        new = _single_date_block("2024-01-01", 1, height=1000, width=1000, chunks=SPLIT_CHUNKS)
        with pytest.raises(ValueError, match="non-empty region"):
            write_regions(store_path, [(new, {})])

    def test_multiple_time_disjoint_runs_one_commit(self, local_zarr_path, sample_reflectance_data):
        """Two time-disjoint runs written together land correctly; others untouched.

        Mirrors the merge workload: same spatial box on disjoint time indices,
        which (time chunk == 1) are guaranteed chunk-disjoint.
        """
        dates = ["2024-01-01", "2024-01-06", "2024-01-11", "2024-01-16"]
        store_path = self._store(local_zarr_path, sample_reflectance_data, dates, name="rws_runs")
        original = open_store(store_path)["blue"].values.copy()

        # Aligned spatial box, two runs: a 2-date run {0,1} and a 1-date run {3}.
        runA = xr.Dataset(
            {"blue": (["time", "northing", "easting"], np.full((2, 500, 500), 7, np.uint16))},
            coords={
                "time": [np.datetime64(d, "ns") for d in dates[0:2]],
                "northing": np.arange(500),
                "easting": np.arange(500),
            },
        ).chunk(ONE_BLOCK)
        runB = _single_date_block("2024-01-16", 8, height=500, width=500, n0=0, e0=0, chunks=ONE_BLOCK)

        spatial = {"northing": slice(0, 500), "easting": slice(0, 500)}
        widened = write_regions(
            store_path,
            [(runA, {"time": slice(0, 2), **spatial}), (runB, {"time": slice(3, 4), **spatial})],
        )
        assert len(widened) == 2

        back = open_store(store_path)["blue"].values
        assert (back[0, 0:500, 0:500] == 7).all()
        assert (back[1, 0:500, 0:500] == 7).all()
        assert (back[3, 0:500, 0:500] == 8).all()
        # Untouched date intact.
        np.testing.assert_array_equal(back[2], original[2])
        # Untouched spatial remainder of written dates intact.
        mask = np.ones(back.shape[1:], dtype=bool)
        mask[0:500, 0:500] = False
        for t in (0, 1, 3):
            np.testing.assert_array_equal(back[t][mask], original[t][mask])

    def test_unaligned_runs_backfill_and_preserve_attrs(self, local_zarr_path, sample_reflectance_data):
        """Unaligned boxes are padded; root attrs survive; update_attrs merged."""
        dates = ["2024-01-01", "2024-01-06", "2024-01-11"]
        store_path = self._store(local_zarr_path, sample_reflectance_data, dates, name="rws_unaligned")
        original = open_store(store_path)["blue"].values.copy()

        sub0 = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        sub2 = _single_date_block("2024-01-11", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        spatial = {"northing": slice(100, 300), "easting": slice(250, 610)}
        write_regions(
            store_path,
            [(sub0, {"time": slice(0, 1), **spatial}), (sub2, {"time": slice(2, 3), **spatial})],
            update_attrs={"reprocessed": "2026-06-18"},
        )

        ds = open_store(store_path)
        back = ds["blue"].values
        for t in (0, 2):
            assert (back[t, 100:300, 250:610] == 9).all()
            mask = np.ones(back.shape[1:], dtype=bool)
            mask[100:300, 250:610] = False
            np.testing.assert_array_equal(back[t][mask], original[t][mask])
        np.testing.assert_array_equal(back[1], original[1])
        assert ds.attrs["crs"] == "EPSG:32615"
        assert ds.attrs["tile_id"] == "33UUP"
        assert ds.attrs["reprocessed"] == "2026-06-18"

    def test_batch_under_distributed_client(self, local_zarr_path, sample_reflectance_data):
        """Unaligned batch write succeeds with a real distributed client active.

        Guards that the padding shell (read from a readonly session) and the
        forked write graph both pickle to workers — the multi-region analogue of
        the single-region distributed guard.
        """
        distributed = pytest.importorskip("distributed")
        dates = ["2024-01-01", "2024-01-06"]
        store_path = self._store(local_zarr_path, sample_reflectance_data, dates, name="rws_distributed")
        original = open_store(store_path)["blue"].values.copy()

        sub0 = _single_date_block("2024-01-01", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        sub1 = _single_date_block("2024-01-06", 9, height=200, width=360, n0=100, e0=250, chunks=ONE_BLOCK)
        spatial = {"northing": slice(100, 300), "easting": slice(250, 610)}
        with distributed.Client(processes=True, n_workers=2, threads_per_worker=1, dashboard_address=":0"):
            write_regions(
                store_path,
                [(sub0, {"time": slice(0, 1), **spatial}), (sub1, {"time": slice(1, 2), **spatial})],
            )

        back = open_store(store_path)["blue"].values
        for t in (0, 1):
            assert (back[t, 100:300, 250:610] == 9).all()
            mask = np.ones(back.shape[1:], dtype=bool)
            mask[100:300, 250:610] = False
            np.testing.assert_array_equal(back[t][mask], original[t][mask])

    def test_rejects_chunk_overlapping_items(self, local_zarr_path, sample_reflectance_data):
        """Two items whose widened regions share a Zarr chunk are rejected (no silent loss).

        Same time index, two unaligned spatial boxes that both fall inside the
        500x500 corner chunk — so each widens to 0:500 x 0:500 and they collide
        in chunk space. Nothing must be committed.
        """
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"], name="rws_overlap")
        original = open_store(store_path)["blue"].values.copy()

        a = _single_date_block("2024-01-01", 7, height=200, width=100, n0=0, e0=0, chunks=ONE_BLOCK)
        b = _single_date_block("2024-01-01", 8, height=200, width=100, n0=0, e0=200, chunks=ONE_BLOCK)
        with pytest.raises(ValueError, match="not chunk-disjoint"):
            write_regions(
                store_path,
                [
                    (a, {"time": slice(0, 1), "northing": slice(0, 200), "easting": slice(0, 100)}),
                    (b, {"time": slice(0, 1), "northing": slice(0, 200), "easting": slice(200, 300)}),
                ],
            )
        np.testing.assert_array_equal(open_store(store_path)["blue"].values, original)

    def test_partial_region_writes_omitted_dim_in_full(self, local_zarr_path, sample_reflectance_data):
        """A region omitting a dim writes that dim in full (the slice(None) path).

        ``easting`` is absent from both regions, so each item covers the store's
        full easting span; disjointness comes from the distinct time indices.
        """
        dates = ["2024-01-01", "2024-01-06"]
        store_path = self._store(local_zarr_path, sample_reflectance_data, dates, name="rws_partial")
        original = open_store(store_path)["blue"].values.copy()

        full0 = _single_date_block("2024-01-01", 5, height=500, width=1000, n0=0, e0=0, chunks=ONE_BLOCK)
        full1 = _single_date_block("2024-01-06", 6, height=500, width=1000, n0=0, e0=0, chunks=ONE_BLOCK)
        write_regions(
            store_path,
            [
                (full0, {"time": slice(0, 1), "northing": slice(0, 500)}),
                (full1, {"time": slice(1, 2), "northing": slice(0, 500)}),
            ],
        )

        back = open_store(store_path)["blue"].values
        assert (back[0, 0:500, :] == 5).all()
        assert (back[1, 0:500, :] == 6).all()
        # Northing remainder of each written date untouched.
        np.testing.assert_array_equal(back[0, 500:, :], original[0, 500:, :])
        np.testing.assert_array_equal(back[1, 500:, :], original[1, 500:, :])

    def test_omitted_dim_source_stays_on_store_chunk_grid(self, local_zarr_path, sample_reflectance_data):
        """A dim omitted from the region is rechunked to the store grid, not one block.

        ``easting`` is absent, so the item covers the full 1000-wide axis. The
        source dask array must split that axis on the store's 500-chunk grid
        (two blocks) rather than coalescing it into one full-axis block — a
        full-axis block would destroy store-grid parallelism and can OOM a
        worker on large stores.
        """
        store_path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"], name="rws_grid")
        full = _single_date_block("2024-01-01", 5, height=500, width=1000, n0=0, e0=0, chunks=ONE_BLOCK)

        repo = _open_repo(store_path)
        session = repo.writable_session("main")
        existing = open_store(store_path)
        group = open_store_as_zarr_group(store_path)
        try:
            fork = session.fork()
            sources, _targets, _regions, _widened = _aligned_region_sources(
                existing, [(full, {"time": slice(0, 1), "northing": slice(0, 500)})], fork, group
            )
        finally:
            existing.close()

        # SPLIT_CHUNKS easting == 500 over a 1000-wide axis -> two blocks of 500.
        easting_axis = 2  # (time, northing, easting)
        assert sources[0].chunks[easting_axis] == (500, 500)


class TestDefaultRepoConfig:
    """No chunk-cache override is applied; max_concurrent_requests stays optional.

    A real-store A/B showed a chunk cache does not help striped inference (it
    caches compressed bytes — a hit saves the S3 GET but not the decompression —
    and a dense strip's band-major working set dwarfs any affordable cache, so
    cross-strip reuse thrashes to ~0% hits). The config now leaves icechunk's
    default cache in place rather than pinning a large one that only burned RAM.
    """

    def test_no_cache_override_without_concurrency_cap(self):
        config = _default_repo_config()
        assert config is not None
        # We no longer override caching: it stays at the icechunk default,
        # which is None on a freshly-defaulted config (icechunk then applies
        # its own internal cache sizing).
        assert config.caching == icechunk.RepositoryConfig.default().caching

    def test_concurrency_cap_applied(self):
        config = _default_repo_config(max_concurrent_requests=64)
        assert config.max_concurrent_requests == 64

    def test_returns_config_not_none_by_default(self):
        # Must always build a config even when no cap is passed.
        assert _default_repo_config() is not None
