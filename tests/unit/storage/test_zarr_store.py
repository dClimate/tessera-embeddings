"""Unit tests for tessera_embeddings/storage/zarr_store.py.

Tests cover both local and S3-backed stores, verifying:
- Store creation and metadata
- Custom and default-style chunking
- DOY storage and append/merge
- Append operations extending the time dimension
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

from pathlib import Path
from types import SimpleNamespace

import icechunk
import numpy as np
import pytest
import xarray as xr

from tessera_embeddings.config import S2_L2A_BANDS
from tessera_embeddings.storage.time_axis import compute_doy
from tessera_embeddings.storage.zarr_store import (
    _DEFAULT_CONNECT_TIMEOUT_MS,
    _DEFAULT_OPERATION_ATTEMPT_TIMEOUT_MS,
    _DEFAULT_READ_TIMEOUT_MS,
    _DEFAULT_STORAGE_INITIAL_BACKOFF_MS,
    _DEFAULT_STORAGE_MAX_BACKOFF_MS,
    _DEFAULT_STORAGE_MAX_TRIES,
    _default_repo_config,
    _write_new,
    get_existing_dates,
    open_store,
    open_store_as_zarr_group,
    plain_zarr_storage_options,
    resolve_region,
    write_dataset,
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


class TestComputeDoy:
    """Tests for DOY (day-of-year) computation."""

    def test_computes_correct_doy_for_known_dates(self):
        """Known dates should produce correct DOY values."""
        timestamps = np.array(
            [
                "2024-01-01",
                "2024-06-15",
                "2024-12-31",
                "2023-12-31",
            ],
            dtype="datetime64[ns]",
        )

        doy = compute_doy(timestamps)

        assert doy[0] == 1  # Jan 1
        assert doy[1] == 167  # June 15 (2024 is leap year)
        assert doy[2] == 366  # Dec 31 (leap year)
        assert doy[3] == 365  # Dec 31 (non-leap year)


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


_RESOLVE_DATES = ["2024-01-01", "2024-01-06", "2024-01-11"]


def _write_blue(store_path, dates, size=100):
    """Write (or append) a ``size``-square, one-band store over ``dates``; coords are 0..size-1."""
    ds = xr.Dataset(
        {"blue": (["time", "northing", "easting"], np.ones((len(dates), size, size), dtype=np.uint16))},
        coords={
            "time": np.array(dates, dtype="datetime64[ns]"),
            "northing": np.arange(size),
            "easting": np.arange(size),
        },
    )
    write_dataset(
        store_path, ds, tile_id="33UUP", baselines=dict.fromkeys(dates, 400), chunks=SPLIT_CHUNKS, crs="EPSG:32615"
    )


class TestResolveRegion:
    """Tests for coordinate-range -> integer-slice resolution."""

    @pytest.fixture(scope="class")
    def store_path(self, tmp_path_factory):
        """Read-only: one store shared by the class."""
        path = str(tmp_path_factory.mktemp("resolve") / "reflectance.zarr")
        _write_blue(path, _RESOLVE_DATES)
        return path

    def test_single_date_resolves_to_one_index(self, store_path):
        assert resolve_region(store_path, time=("2024-01-06", "2024-01-06")) == {"time": slice(1, 2)}

    def test_date_range_resolves_to_half_open_slice(self, store_path):
        assert resolve_region(store_path, time=("2024-01-01", "2024-01-06")) == {"time": slice(0, 2)}

    def test_spatial_range_resolves_against_coords(self, store_path):
        region = resolve_region(store_path, northing=(10, 29), easting=(25, 60))
        assert region == {"northing": slice(10, 30), "easting": slice(25, 61)}

    def test_none_axis_is_omitted(self, store_path):
        region = resolve_region(store_path, time=("2024-01-01", "2024-01-01"))
        assert set(region) == {"time"}

    def test_nonexistent_date_raises(self, store_path):
        """The overwrite-in-place contract: a date not in the store is rejected."""
        with pytest.raises(ValueError, match="selects no existing coordinate"):
            resolve_region(store_path, time=("2025-01-01", "2025-01-01"))

    def test_non_contiguous_range_raises(self, local_zarr_path):
        """An out-of-order time axis whose range straddles a gap is rejected."""
        # write_dataset appends without sorting, so append out of order to build
        # the axis [2024-01-06, 2024-01-01, 2024-01-20].
        store_path = str(local_zarr_path / "reflectance.zarr")
        for date in ["2024-01-06", "2024-01-01", "2024-01-20"]:
            _write_blue(store_path, [date])

        # Range 01-06..01-20 matches indices 0 and 2 — non-contiguous (gap at 1).
        with pytest.raises(ValueError, match="non-contiguous"):
            resolve_region(store_path, time=("2024-01-06", "2024-01-20"))


class TestDefaultRepoConfig:
    """No chunk-cache override is applied, and no request-concurrency cap.

    A real-store A/B showed a chunk cache does not help striped inference (it
    caches compressed bytes — a hit saves the S3 GET but not the decompression —
    and a dense strip's band-major working set dwarfs any affordable cache, so
    cross-strip reuse thrashes to ~0% hits). The config now leaves icechunk's
    default cache in place rather than pinning a large one that only burned RAM.
    """

    def test_the_request_concurrency_cap_is_icechunks_default(self):
        # NEVER capped by us. A cap of 1 — what the old per-fork budget split produced on every
        # campaign fill — deadlocks commit and rebase/diff inside icechunk's tokio runtime under
        # concurrent writers (context_docs/assembly/icechunk-max-concurrent-requests-1-deadlock.md).
        assert (
            _default_repo_config().max_concurrent_requests
            == icechunk.RepositoryConfig.default().max_concurrent_requests
        )

    def test_storage_timeouts_and_retries_applied(self):
        # Every repo open inherits finite per-attempt timeouts and a backed-off
        # retry budget so a wedged socket fails the attempt instead of hanging
        # forever (see context_docs/ingest/campaign-ingest-measurements.md §14.4, gotcha 12).
        storage = _default_repo_config().storage
        assert storage is not None
        assert storage.timeouts.connect_timeout_ms == _DEFAULT_CONNECT_TIMEOUT_MS
        assert storage.timeouts.read_timeout_ms == _DEFAULT_READ_TIMEOUT_MS
        assert storage.timeouts.operation_attempt_timeout_ms == _DEFAULT_OPERATION_ATTEMPT_TIMEOUT_MS
        assert storage.retries.max_tries == _DEFAULT_STORAGE_MAX_TRIES
        assert storage.retries.initial_backoff_ms == _DEFAULT_STORAGE_INITIAL_BACKOFF_MS
        assert storage.retries.max_backoff_ms == _DEFAULT_STORAGE_MAX_BACKOFF_MS


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


class TestPlainZarrIsReadWithTheRunsCredentials:
    """The ROI mask is a PLAIN zarr, so it is read through fsspec and does not travel on
    the Icechunk callback threaded everywhere else — leaving the one read that decides
    which chunks exist opening on whatever ambient credentials the process had.
    """

    def _creds(self):
        return SimpleNamespace(access_key_id="AK", secret_access_key="SK", session_token="TK")

    def test_a_local_roi_needs_no_options(self, tmp_path):
        assert plain_zarr_storage_options(str(tmp_path / "roi.zarr"), lambda: self._creds(), "us-east-2") is None

    def test_the_callbacks_credentials_and_the_region_both_reach_the_open(self):
        options = plain_zarr_storage_options("s3://in/rois/roi.zarr", lambda: self._creds(), "us-east-2")
        assert options == {
            "client_kwargs": {"region_name": "us-east-2"},
            "key": "AK",
            "secret": "SK",
            "token": "TK",
        }

    def test_no_callback_and_no_region_leaves_the_default_chain_alone(self):
        assert plain_zarr_storage_options("s3://in/rois/roi.zarr", None, None) is None

    def test_the_credential_is_resolved_per_call(self):
        """An IAM credential expires in hours; a value captured once outlives its own TTL."""
        calls = []

        def get_credentials():
            calls.append(1)
            return self._creds()

        plain_zarr_storage_options("s3://in/rois/roi.zarr", get_credentials, None)
        plain_zarr_storage_options("s3://in/rois/roi.zarr", get_credentials, None)
        assert len(calls) == 2
