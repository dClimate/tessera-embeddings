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
from tessera_embeddings.storage.region_writes import _pad_region_to_chunks
from tessera_embeddings.storage.zarr_store import (
    _DEFAULT_CONNECT_TIMEOUT_MS,
    _DEFAULT_OPERATION_ATTEMPT_TIMEOUT_MS,
    _DEFAULT_READ_TIMEOUT_MS,
    _DEFAULT_STORAGE_INITIAL_BACKOFF_MS,
    _DEFAULT_STORAGE_MAX_BACKOFF_MS,
    _DEFAULT_STORAGE_MAX_TRIES,
    S3Config,
    _default_repo_config,
    _loss_date,
    _open_writable_session,
    _write_new,
    compute_doy,
    get_existing_dates,
    open_repo,
    open_store,
    open_store_as_zarr_group,
    project_assessment,
    read_assessment_log,
    record_assessed_window,
    resolve_region,
    rollback_commits,
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

        repo = open_repo(store_path)
        history = list(repo.ancestry(branch="main"))
        expected_target = history[1].id  # one commit back

        new_head = rollback_commits(store_path, 1)

        assert new_head == expected_target
        assert open_repo(store_path).lookup_branch("main") == expected_target

    def test_dry_run_does_not_move_branch(self, local_zarr_path, sample_reflectance_data):
        """dry_run resolves the target id but leaves HEAD untouched."""
        store_path = str(local_zarr_path / "rb_dry" / "reflectance.zarr")
        self._build_history(store_path, sample_reflectance_data, ["2024-01-01", "2024-01-06", "2024-01-11"])

        head_before = open_repo(store_path).lookup_branch("main")

        target = rollback_commits(store_path, 1, dry_run=True)

        assert open_repo(store_path).lookup_branch("main") == head_before
        assert target != head_before
        assert open_store(store_path).sizes["time"] == 3

    def test_rollback_is_reversible(self, local_zarr_path, sample_reflectance_data):
        """The dropped snapshots survive, so resetting back restores the data."""
        store_path = str(local_zarr_path / "rb_rev" / "reflectance.zarr")
        self._build_history(store_path, sample_reflectance_data, ["2024-01-01", "2024-01-06", "2024-01-11"])

        original_head = open_repo(store_path).lookup_branch("main")
        rollback_commits(store_path, 2)
        assert open_store(store_path).sizes["time"] == 1

        open_repo(store_path).reset_branch("main", original_head)
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

    def test_padded_shell_graph_is_slab_local_and_flat_in_time(self):
        """The shell's dask graph must scale with the WIDENED slab's chunks, not the
        store's time length — the property that makes the zarr-direct shell flat in
        master time and kills the scheduler OOM. ``existing.isel(widened)`` carried
        the whole-store time-chunk layer into every shell, so its graph grew with the
        master's total dates; value-parity tests can't catch that regression, only a
        graph-size check independent of ``nt`` can.
        """
        # 100x100 store at 50-px chunks; region 10:30 x 10:70 widens to a 1x1x2 block slab.
        region = {"time": slice(0, 1), "northing": slice(10, 30), "easting": slice(10, 70)}
        data = _single_date_block("2024-01-01", 9, height=20, width=60, n0=10, e0=10, chunks=ONE_BLOCK)

        sizes = {}
        for nt in (5, 500):
            existing = self._existing_nt(nt=nt)
            padded, widened = _pad_region_to_chunks(existing, data, region, self._group(existing))
            assert widened["northing"] == slice(0, 50) and widened["easting"] == slice(0, 100)
            sizes[nt] = len(padded["blue"].data.__dask_graph__())

        # Widened slab is 1 time x 1 northing-block x 2 easting-blocks = 2 read tasks,
        # plus the setitem overlay fan-out — a small constant. The whole point: it must
        # NOT grow from nt=5 to nt=500. An un-fixed isel shell would scale with nt.
        assert sizes[5] == sizes[500], f"shell graph scales with store time length: {sizes}"
        assert sizes[5] < 40, f"shell graph has {sizes[5]} keys — not slab-local"

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

    def test_storage_timeouts_and_retries_applied(self):
        # Every repo open inherits finite per-attempt timeouts and a backed-off
        # retry budget so a wedged socket fails the attempt instead of hanging
        # forever (see context_docs/design/region-merge.md → "Hang protection").
        storage = _default_repo_config().storage
        assert storage is not None
        assert storage.timeouts.connect_timeout_ms == _DEFAULT_CONNECT_TIMEOUT_MS
        assert storage.timeouts.read_timeout_ms == _DEFAULT_READ_TIMEOUT_MS
        assert storage.timeouts.operation_attempt_timeout_ms == _DEFAULT_OPERATION_ATTEMPT_TIMEOUT_MS
        assert storage.retries.max_tries == _DEFAULT_STORAGE_MAX_TRIES
        assert storage.retries.initial_backoff_ms == _DEFAULT_STORAGE_INITIAL_BACKOFF_MS
        assert storage.retries.max_backoff_ms == _DEFAULT_STORAGE_MAX_BACKOFF_MS

    def test_timeouts_compose_with_concurrency_cap(self):
        # The cap and the storage timeouts must coexist in one config — setting the
        # concurrency cap must not drop the hang-protection settings.
        config = _default_repo_config(max_concurrent_requests=64)
        assert config.max_concurrent_requests == 64
        assert config.storage is not None
        assert config.storage.timeouts.read_timeout_ms == _DEFAULT_READ_TIMEOUT_MS
        assert config.storage.retries.max_tries == _DEFAULT_STORAGE_MAX_TRIES


class TestEmptyChunkElision:
    """The zone mosaics stay cheap only if all-fill (ocean/nodata) chunks are
    not materialized as objects. This pins that write_dataset's icechunk write
    elides them (ADR-011's cost assumption), rather than taking it on faith.
    """

    def test_all_fill_chunks_are_not_materialized(self, local_zarr_path):
        store_path = str(local_zarr_path / "elision" / "reflectance.zarr")
        data = xr.Dataset(
            {"red": (("time", "northing", "easting"), np.zeros((1, 8, 8), dtype="uint16"))},
            coords={
                "time": np.array(["2025-06-01"], dtype="datetime64[ns]"),
                "northing": np.arange(8, dtype="float64"),
                "easting": np.arange(8, dtype="float64"),
            },
        )
        data["red"].values[0, 0:4, 0:4] = 1  # only the top-left 4x4 chunk carries data

        write_dataset(
            store_path,
            data,
            tile_id="t",
            baselines={},
            chunks={"time": 1, "northing": 4, "easting": 4},
            crs="EPSG:32601",
        )

        red = open_store_as_zarr_group(store_path)["red"]
        assert red.shape == (1, 8, 8)
        # 2x2 spatial chunks; the three all-zero (fill) chunks are elided.
        assert red.nchunks_initialized == 1


class TestAssessedWindowRecord:
    """The store's account of what was examined, and of the holes inside it.

    Both records are UNIONS with what the store already holds, because a resumed leg re-derives
    only the part of the window it walked. Writing its own answer alone would erase the rest —
    and the assessed window is what makes an absent month a finding rather than a gap, so an
    erased or over-wide one turns a real gap into a clean bill of health.
    """

    def _store(self, local_zarr_path, sample_reflectance_data, dates):
        path = str(local_zarr_path / "assessed" / "reflectance.zarr")
        for i, date in enumerate(dates):
            data = sample_reflectance_data([date], height=64, width=64, seed=i)
            _write_reflectance(path, data, tile_id="33UUP", baselines={date: 400})
        return path

    def _attrs(self, path):
        return dict(open_store_as_zarr_group(path).attrs)

    def test_the_stored_start_is_kept_and_only_the_end_extends(self, local_zarr_path, sample_reflectance_data):
        """A resume begins at its frontier's month, so its own start is LATER than what was
        assessed before. Taking it would retract months an earlier leg did examine; taking the
        earliest start either leg mentions would claim months this one skipped. Keeping the
        stored start does neither.
        """
        path = self._store(local_zarr_path, sample_reflectance_data, ["2018-01-01"])

        record_assessed_window(path, "2018-01-01", "2018-06-30")
        record_assessed_window(path, "2018-05-01", "2018-12-31")

        assert self._attrs(path)["assessed_window"] == ["2018-01-01", "2018-12-31"]

    def test_a_first_record_starts_where_the_run_started(self, local_zarr_path, sample_reflectance_data):
        """The interrupted-leg case. A leg killed before it recorded anything leaves dates on the
        axis and no window at all, so the resume's own start is all there is — and the months
        below it stay OUTSIDE the window, which reads as not assessed rather than as assessed and
        clean. That is the truthful answer: nothing on this store says they were examined.
        """
        path = self._store(local_zarr_path, sample_reflectance_data, ["2018-01-01"])

        record_assessed_window(path, "2018-05-01", "2018-12-31")

        assert self._attrs(path)["assessed_window"] == ["2018-05-01", "2018-12-31"]

    def test_a_gap_between_the_ranges_refuses_the_join(self, local_zarr_path, sample_reflectance_data):
        """The failure the union itself can cause, from the other side.

        January-March assessed, then June-December assessed, and a single range spanning both
        would certify April and May — which neither run queried. The coverage gate reads that as
        permission to excuse two absent months and publish an incomplete mosaic, which is the
        one thing this record must never say.
        """
        path = self._store(local_zarr_path, sample_reflectance_data, ["2018-01-01"])

        record_assessed_window(path, "2018-01-01", "2018-03-31")
        record_assessed_window(path, "2018-06-01", "2018-12-31")

        assert self._attrs(path)["assessed_window"] == ["2018-06-01", "2018-12-31"]

    def test_ranges_that_meet_end_to_end_still_join(self, local_zarr_path, sample_reflectance_data):
        """No day lies between them, so the two describe one examined stretch.

        Refusing this would discard half of any window split across two legs, which is the
        ordinary shape of a resume.
        """
        path = self._store(local_zarr_path, sample_reflectance_data, ["2018-01-01"])

        record_assessed_window(path, "2018-01-01", "2018-06-30")
        record_assessed_window(path, "2018-07-01", "2018-12-31")

        assert self._attrs(path)["assessed_window"] == ["2018-01-01", "2018-12-31"]

    def test_losses_union_across_runs_and_drop_once_the_date_arrives(self, local_zarr_path, sample_reflectance_data):
        """Two runs, two holes, one list — and an entry disappears when the date is filled.

        The union is what keeps a resume from erasing the earlier months' losses it never
        re-derives. Dropping a date that is now on the axis is what the unconditional overwrite
        this replaced was protecting: an audit must not be told pixels are missing that are
        present.
        """
        path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"])

        record_assessed_window(path, "2024-01-01", "2024-01-31", unreadable=[{"date": "2024-01-06"}])
        record_assessed_window(path, "2024-02-01", "2024-02-28", unreadable=[{"date": "2024-02-14"}])
        assert [e["date"] for e in self._attrs(path)["assessed_unreadable_dates"]] == ["2024-01-06", "2024-02-14"]

        data = sample_reflectance_data(["2024-01-06"], height=64, width=64, seed=9)
        _write_reflectance(path, data, tile_id="33UUP", baselines={"2024-01-06": 400})

        # A run over BOTH months reporting no losses says it looked and found none, so the
        # February entry goes too. That is the point: a date once recorded unreadable and later
        # re-read but legitimately filtered — too cloudy, no live window — would otherwise keep a
        # stale entry forever, and if it were that month's only acquisition the month could never
        # pass the coverage gate.
        record_assessed_window(path, "2024-01-01", "2024-02-28")
        assert self._attrs(path)["assessed_unreadable_dates"] == []

    def test_a_loss_outside_the_range_a_run_examined_survives_it(self, local_zarr_path, sample_reflectance_data):
        """Reconciliation reaches only as far as the run looked.

        A resumed run starts at its frontier's month and never re-derives the earlier months'
        losses, so those entries are not stale — they are simply unexamined, and this run has no
        opinion on them. This is also what protects a record placed by hand during a repair: those
        describe months below a resume floor, which is exactly the ground a run skips.
        """
        path = self._store(local_zarr_path, sample_reflectance_data, ["2024-01-01"])
        record_assessed_window(path, "2024-01-01", "2024-01-31", unreadable=[{"date": "2024-01-06"}])

        # A later run examines FEBRUARY only, and reports nothing lost there.
        record_assessed_window(path, "2024-02-01", "2024-02-28")

        assert [e["date"] for e in self._attrs(path)["assessed_unreadable_dates"]] == ["2024-01-06"]


def test_malformed_recorded_evidence_is_refused_rather_than_dropped():
    """Evidence in a shape nobody wrote is not evidence of nothing.

    Silently reading it as empty lets the next run replace whatever it recorded, and since a
    resumed run no longer re-derives the earlier months' losses, that erasure is unrecoverable.
    Refusing keeps the store's own account intact for someone to look at.
    """
    with pytest.raises(TypeError, match="not a sequence"):
        read_assessment_log({"assessment_log": {"oops": True}})

    with pytest.raises(TypeError, match="not a sequence"):
        read_assessment_log(
            {"assessed_window": ["2018-01-01", "2018-01-31"], "assessed_unreadable_dates": "2018-01-06"}
        )


class TestTheAssessmentLog:
    """The record is a history of what each run saw, folded on demand into what readers consume.

    Every case here was previously a RULE — join-or-refuse, prior-wins, reconcile-within-range,
    upgrade-the-verdict — and each rule was a reviewer's finding before it was a rule. They are
    properties of an append-only history rather than policies applied to a mutable summary, which
    is the point: there is no stored answer left to go stale.
    """

    def test_a_store_written_before_the_log_reads_as_one_entry(self):
        """No store has to be rewritten. The old pair says exactly what one entry says."""
        attrs = {
            "assessed_window": ["2018-01-01", "2018-06-30"],
            "assessed_unreadable_dates": [{"date": "2018-03-04", "scope": "unreadable"}],
        }
        assert read_assessment_log(attrs) == [
            {
                "examined": ["2018-01-01", "2018-06-30"],
                "losses": [{"date": "2018-03-04", "scope": "unreadable"}],
                "source": "pre-log",
            }
        ]
        window, losses = project_assessment(read_assessment_log(attrs))
        assert window == ["2018-01-01", "2018-06-30"], "and folds back to what it claimed"
        assert [e["date"] for e in losses] == ["2018-03-04"]

    def test_a_store_with_nothing_recorded_has_an_empty_history(self):
        assert read_assessment_log({}) == []
        assert project_assessment([]) == (None, [])

    def test_disjoint_ranges_under_claim_rather_than_span_the_gap(self):
        """One range cannot describe two, and spanning would certify days nobody examined.

        Under-claiming only makes the coverage gate stricter; over-claiming lets it excuse a month
        that is missing because nobody looked, which is the one failure this record must never
        produce.
        """
        log = [
            {"examined": ["2018-01-01", "2018-03-31"], "losses": []},
            {"examined": ["2018-06-01", "2018-12-31"], "losses": []},
        ]
        window, _ = project_assessment(log)
        assert window == ["2018-06-01", "2018-12-31"], "the block reaching furthest forward"

        # Adjacent ranges are one stretch and DO combine.
        joined, _ = project_assessment(
            [
                {"examined": ["2018-01-01", "2018-03-31"], "losses": []},
                {"examined": ["2018-04-01", "2018-12-31"], "losses": []},
            ]
        )
        assert joined == ["2018-01-01", "2018-12-31"]

    def test_the_run_that_looked_most_recently_decides(self):
        """A later run covering a date supersedes an earlier verdict, including by omission."""
        log = [
            {"examined": ["2018-01-01", "2018-01-31"], "losses": [{"date": "2018-01-06", "scope": "unreadable"}]},
            # Re-examined the same month and found nothing lost — the date now reads, or is
            # legitimately filtered. Either way it is no longer a hole.
            {"examined": ["2018-01-01", "2018-01-31"], "losses": []},
        ]
        _, losses = project_assessment(log)
        assert losses == [], "the stale record is retired by the later look"

    def test_a_date_no_run_examined_keeps_its_record(self):
        """What protects a repair written by hand: it describes ground runs skip."""
        log = [
            {"examined": None, "losses": [{"date": "2018-01-06", "scope": "unfillable"}], "source": "repair"},
            {"examined": ["2018-05-01", "2018-12-31"], "losses": []},
        ]
        _, losses = project_assessment(log)
        assert [e["date"] for e in losses] == ["2018-01-06"], "January was never examined"

    def test_a_date_back_on_the_axis_is_not_a_loss(self):
        log = [{"examined": ["2018-01-01", "2018-01-31"], "losses": [{"date": "2018-01-06"}]}]
        _, losses = project_assessment(log, present={"2018-01-06"})
        assert losses == []

    def test_losses_without_a_window_survive_the_upgrade(self):
        """The state a run leaves when it stops after sealing a loss and before its final record.

        A loss is committed by the write that seals its date; the range is recorded at the end.
        Between those two a store legitimately holds losses and no window — and that is the state
        most in need of preserving, because the record is the hole's only durable account. Reading
        it as an empty history would have let the next append overwrite it.
        """
        attrs = {"assessed_unreadable_dates": [{"date": "2018-03-04", "scope": "unreadable"}]}
        log = read_assessment_log(attrs)
        assert log == [
            {"examined": None, "losses": [{"date": "2018-03-04", "scope": "unreadable"}], "source": "pre-log"}
        ]
        _, losses = project_assessment(log)
        assert [e["date"] for e in losses] == ["2018-03-04"]

    def test_an_entry_this_cannot_read_is_refused_not_filtered(self):
        """An attribute is written whole, so anything the read drops, the rewrite deletes.

        Filtering an unrecognised entry would destroy the evidence the append-only promise exists
        to keep, and would do it silently — the opposite of the fail-closed behaviour the reader
        gives a malformed log as a whole.
        """
        with pytest.raises(TypeError, match="not records"):
            read_assessment_log({"assessment_log": [{"examined": None, "losses": []}, "a bare string"]})

    def test_a_date_lost_cleared_and_lost_again_is_a_new_observation(self):
        """Deduplication keys on the CURRENT verdict, never on every mention ever made.

        Matching against history would suppress the second loss, so the commit sealing it would
        carry no record of it, and an interruption before the final record would leave the hole
        unnamed — the exact durability guarantee this record exists for.
        """
        log = [
            {"examined": ["2018-01-01", "2018-01-31"], "losses": [{"date": "2018-01-06"}]},
            {"examined": ["2018-01-01", "2018-01-31"], "losses": []},  # re-examined, cleared
        ]
        _, current = project_assessment(log)
        assert current == [], "cleared, so nothing is currently advertised"
        assert _loss_date({"date": "2018-01-06"}) not in {_loss_date(x) for x in current}, (
            "so losing it again is a fresh observation and must be recorded"
        )

    def test_a_recovered_date_stays_recovered_across_unrelated_writes(self):
        """The fold replays all history, so it must be filtered against ALL stored dates.

        Filtered against one batch only, a recovered date disappears while that batch folds and is
        republished by the next unrelated write, which sees a different batch and the same history.
        """
        log = [{"examined": ["2018-01-01", "2018-01-31"], "losses": [{"date": "2018-01-06"}]}]

        # the recovery batch
        _, during = project_assessment(log, present={"2018-01-06"})
        assert during == []
        # a later, unrelated write — the axis still holds the recovered date
        _, after = project_assessment(log, present={"2018-01-06", "2018-02-11"})
        assert after == [], "and it must not come back"

    def test_a_date_no_run_considered_keeps_its_record(self):
        """Examining a range is not judging every day in it.

        A run sees only the days the catalogue returned. A day omitted from one response — a
        transient omission is enough — was never judged, so clearing its record on the strength of
        the range would delete real evidence; and if it were its month's only acquisition, the
        month would then read as legitimately empty and an incomplete mosaic could publish.
        """
        log = [
            {"examined": ["2018-01-01", "2018-01-31"], "losses": [{"date": "2018-01-06"}]},
            # A later run over the same month whose catalogue response omitted the 6th.
            {
                "examined": ["2018-01-01", "2018-01-31"],
                "considered": ["2018-01-05", "2018-01-07"],
                "losses": [],
            },
        ]
        _, losses = project_assessment(log)
        assert [e["date"] for e in losses] == ["2018-01-06"], "never judged, so never cleared"

        # And a run that DID consider it, and did not call it lost, does clear it.
        log[1]["considered"] = ["2018-01-05", "2018-01-06", "2018-01-07"]
        _, cleared = project_assessment(log)
        assert cleared == []
