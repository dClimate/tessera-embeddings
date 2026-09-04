"""Tests for ZarrWriter: staging round-trips, raw-zarr assembly, cleanup dispatch.

Assembly tests exercise the fork/merge engine (``assemble``/``assemble_global``)
directly — no Dask. cleanup_staging coverage diverges from the reference repo:
this repo's cleanup_staging prefers s5cmd for S3 with an fsspec fallback,
so TestCleanupStagingDispatch exercises the routing decision instead of error propagation.
"""

from __future__ import annotations

import datetime
import itertools
import json
import logging
from importlib.metadata import version as _dist_version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr

import tessera_embeddings.inference.assembly as _assembly_mod
from tessera_embeddings.config.inference import EMBEDDING_DIM, INFERENCE_CHUNK_SIZE
from tessera_embeddings.config.store_layout import (
    CARRIED_VARS,
    GLOBAL,
    INNER_PX,
    MONTH_COVERED_VARS,
    MONTHS_IN_YEAR,
    SHARD_PX,
    SINGLE,
)
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.inference.assembly import (
    OBS_COUNT_VARS,
    STAGED_READ_CONFIG_KWARGS,
    TARGET_AGGREGATE_S3_CONCURRENCY,
    AllChunksSkippedError,
    IncompleteStageError,
    SpatialCoords,
    ZarrWriter,
    _layout_matching_store,
    _partition_bands,
    _s3_budget_split,
    _staged_storage_options,
    _write_granularity,
)
from tessera_embeddings.inference.chunk_spec import ChunkSpec, enumerate_chunks, filter_chunks_by_roi_mask
from tessera_embeddings.inference.quantization import quantize_embeddings
from tessera_embeddings.storage.conventions import ENCODER_VERSION
from tessera_embeddings.storage.global_store import create_global_repo, create_layout_arrays, open_global_repo
from tessera_embeddings.storage.zarr_store import (
    TIME_ENCODING,
    open_or_create_repo,
    open_store,
    plain_zarr_storage_options,
)


@pytest.mark.parametrize(
    ("s3_concurrency", "n_workers"),
    [(100, 8), (5, 8), (3, 8), (1, 8), (None, 8), (50, 4), (100, 200), (5, 16)],
)
def test_s3_budget_split_keeps_every_requested_worker(s3_concurrency, n_workers):
    """REPLACES ``test_s3_budget_split_never_exceeds_target``, which asserted
    ``workers * cap <= budget`` and so pinned the clamp that cost the campaign
    two thirds of its assembly width.

    The per-fork cap floors at 1, so that product CANNOT be held below the budget
    once the requested worker count exceeds it — the only way to satisfy it is to
    drop forks, which is what used to happen on every assembly. The invariant is
    now the weaker, honest one: the fleet ceiling holds where it can, and where it
    cannot, the overshoot is exactly the requested worker count.
    """
    budget = s3_concurrency if s3_concurrency is not None else TARGET_AGGREGATE_S3_CONCURRENCY
    workers, cap = _s3_budget_split(s3_concurrency, n_workers)
    assert workers >= 1 and cap >= 1
    assert workers == n_workers, "the S3 budget must never shrink the fork pool"
    assert workers * cap <= max(budget, n_workers)


def test_the_campaigns_own_parameters_no_longer_shrink_the_pool():
    """The exact arithmetic that bound in production: TARGET_AGGREGATE_S3_CONCURRENCY
    (100) // (2 * n_clusters) at ten clusters is 5, against AssemblyConfig's 16 workers.

    Pinned with the campaign's real numbers rather than a synthetic pair, because
    the clamp was invisible precisely where it bound — the requested count stayed
    16 everywhere it was configured, and only the emitted `workers_used` disagreed.
    The measured cost is in `context_docs/storage/writing-to-the-global-store.md`.
    """
    assert _s3_budget_split(5, 16) == (16, 1)


def test_a_budget_above_the_worker_count_still_divides():
    """The half that was never broken: when the budget genuinely exceeds the pool,
    each fork gets its share and the fleet ceiling holds exactly.
    """
    workers, cap = _s3_budget_split(100, 8)
    assert (workers, cap) == (8, 12)
    assert workers * cap <= 100


def _dummy_scales(h: int, w: int) -> np.ndarray:
    """Return a dummy float32 scales array for test write_chunk calls."""
    return np.ones((h, w), dtype=np.float32)


def _make_full_roi_mask(tmp_path, total_y: int, total_x: int) -> str:
    """Write an all-True ROI mask so ``filter_chunks_by_roi_mask`` keeps every chunk.

    Tests that predate the ROI pre-filter exercise the 100%-coverage case.
    """
    path = str(tmp_path / "roi_mask.zarr")
    arr = zarr.open(path, mode="w", shape=(total_y, total_x), chunks=(total_y, total_x), dtype="bool")
    arr[:] = True
    return path


def _make_partial_roi_mask(tmp_path, total_y: int, total_x: int, x_true_stop: int) -> str:
    """An ROI mask True only over columns ``[0, x_true_stop)`` — a shrunken ROI."""
    path = str(tmp_path / "roi_partial.zarr")
    arr = zarr.open(path, mode="w", shape=(total_y, total_x), chunks=(total_y, total_x), dtype="bool")
    arr[:] = False
    arr[:, :x_true_stop] = True
    return path


def _quantized_embeddings(
    rng: np.random.Generator, h: int, w: int, dim: int = EMBEDDING_DIM
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random float32 embeddings and return (int8_quantized, scales)."""
    raw = rng.standard_normal((h, w, dim)).astype(np.float32)
    return quantize_embeddings(raw)


#: The 2x2 tile grid most assembly tests use: 5-px rows except the short bottom
#: band, so partial edge chunks are always in play rather than an even split.
_GRID_2X2 = [
    ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5),
    ChunkSpec(row=0, col=1, y_start=0, y_stop=5, x_start=5, x_stop=10),
    ChunkSpec(row=1, col=0, y_start=5, y_stop=8, x_start=0, x_stop=5),
    ChunkSpec(row=1, col=1, y_start=5, y_stop=8, x_start=5, x_stop=10),
]
#: Its top row alone, for tests that need two chunks rather than four.
_GRID_1X2 = _GRID_2X2[:2]


def _stage_chunks(writer, chunks, run_id, rng, shape) -> np.ndarray:
    """Stage every chunk and return the mosaic they should assemble into.

    Building the expectation from the SAME arrays that were staged is the point:
    a test that recomputed them from the rng would pass even if `write_chunk`
    silently dropped or reordered a chunk.
    """
    expected = np.zeros((*shape, EMBEDDING_DIM), dtype=np.int8)
    for chunk in chunks:
        emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
        writer.write_chunk(chunk, emb, run_id, scales=scales)
        expected[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop, :] = emb
    return expected


def _assembled(output: str) -> np.ndarray:
    """The single-timestep embeddings array from an assembled output store."""
    return open_store(output)["embeddings"].values[0, ...]


def _assembly_summary_records(caplog) -> list[dict]:
    """Every ASSEMBLY_SUMMARY record captured by caplog, parsed from its JSON payload."""
    prefix = "ASSEMBLY_SUMMARY: "
    return [
        json.loads(record.getMessage().removeprefix(prefix))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


class TestStagedStorageOptions:
    """Tests for _staged_storage_options: attach botocore retries on S3, nowhere else.

    Assembly must stay backend-agnostic — the adaptive-retry config is a botocore
    client setting and is meaningless for local/other backends, so it is gated on
    the ``s3`` protocol.
    """

    def test_s3_path_gets_adaptive_retry_config(self):
        opts = _staged_storage_options("s3://bucket/staging/run/chunk_0_0.zarr")
        assert opts == {"config_kwargs": STAGED_READ_CONFIG_KWARGS}
        # adaptive mode is the load-bearing part: it recognizes SlowDown and
        # self-throttles, unlike botocore's default legacy mode.
        assert opts["config_kwargs"]["retries"]["mode"] == "adaptive"

    @pytest.mark.parametrize(
        "path",
        [
            "/tmp/staging/run/chunk_0_0.zarr",
            "file:///tmp/staging/run/chunk_0_0.zarr",
            "relative/staging/chunk_0_0.zarr",
        ],
    )
    def test_non_s3_path_gets_no_options(self, path):
        assert _staged_storage_options(path) is None


class TestWriteChunk:
    """Tests for ZarrWriter.write_chunk (no Dask needed)."""

    def test_write_chunk_roundtrip(self, tmp_path):
        """Write a chunk and read it back — values should match."""
        staging = str(tmp_path / "staging")
        writer = ZarrWriter(staging)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=10, x_start=0, x_stop=10)

        rng = np.random.default_rng(42)
        embeddings = rng.random((10, 10, EMBEDDING_DIM)).astype(np.float32)

        path = writer.write_chunk(chunk, embeddings, run_id="test123", scales=_dummy_scales(10, 10))

        # Read back (staging uses plain Zarr, not Icechunk)
        ds = xr.open_zarr(path)
        result = ds["embeddings"].values
        np.testing.assert_array_almost_equal(result, embeddings, decimal=5)
        assert result.shape == (10, 10, EMBEDDING_DIM)

    def test_write_chunk_records_completion_both_ways(self, tmp_path):
        """A finished write is recorded in-store AND in the staging listing.

        The two signals serve different readers — the attribute is free for anyone
        holding the group open and cannot be bypassed, the sibling marker is visible
        to a prefix LIST — so both must land, and the listing must key on the marker.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=10, x_start=0, x_stop=10)
        embeddings = np.random.default_rng(0).random((10, 10, EMBEDDING_DIM)).astype(np.float32)

        path = writer.write_chunk(chunk, embeddings, run_id="run1", scales=_dummy_scales(10, 10))

        assert zarr.open_group(path, mode="r").attrs["staged_complete"] is True
        assert Path(writer._done_marker_path("run1", chunk)).exists()
        assert writer._list_staged_labels("run1") == [chunk.label]

    @pytest.mark.parametrize("rewrite", [False, True])
    def test_the_completion_protocol_runs_in_order(self, tmp_path, rewrite):
        """The write order every completeness check rests on, pinned end to end.

        Two properties, and reordering any step breaks one of them:

        * ``.done`` LAST — so its presence implies the attribute is set, which implies
          ``to_zarr`` returned. Were it first, a crash in between would leave a tile the
          listing calls complete but every reader rejects: unresumable without manual
          deletion, because re-inference would never be triggered for it.
        * both markers retracted FIRST — the previous ``.done``, so a tile being
          rewritten never has one vouching for it (retracting after the rewrite, or not
          at all, leaves a window in which a listing calls a half-replaced tile
          complete), and any ``.skipped`` from an attempt where this chunk had no valid
          pixels, which would otherwise sit beside the new tile and read as an
          inconsistent artifact — and the COVERAGE-ONLY tile of such an attempt, which
          carries real counts and so would have assembly publish a footprint this write
          has just replaced.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=4, x_start=0, x_stop=4)
        emb, scales = _quantized_embeddings(np.random.default_rng(0), 4, 4)
        if rewrite:
            # The case the retraction exists for: a marker is already there.
            writer.write_chunk(chunk, emb, run_id="run1", scales=scales)
            assert Path(writer._done_marker_path("run1", chunk)).exists()

        seen: list[str] = []
        real_open_group, real_to_zarr, real_fs_for = zarr.open_group, xr.Dataset.to_zarr, _assembly_mod._fs_for

        def spy_open_group(*args, **kwargs):
            if kwargs.get("mode") == "a" or (len(args) > 1 and args[1] == "a"):
                seen.append("attribute")
            return real_open_group(*args, **kwargs)

        def spy_to_zarr(self, *args, **kwargs):
            seen.append("tile")
            return real_to_zarr(self, *args, **kwargs)

        def spy_fs_for(uri, *args, **kwargs):
            fs = real_fs_for(uri, *args, **kwargs)
            return _RecordingMarkerFS(fs, seen) if uri.endswith(".done") else fs

        with (
            patch.object(_assembly_mod.zarr, "open_group", spy_open_group),
            patch.object(xr.Dataset, "to_zarr", spy_to_zarr),
            patch.object(_assembly_mod, "_fs_for", spy_fs_for),
        ):
            writer.write_chunk(chunk, emb, run_id="run1", scales=scales)

        assert seen == ["retract:done", "retract:skipped", "retract:zarr", "tile", "attribute", "done"]

    def test_write_chunk_wrong_shape_raises(self, tmp_path):
        staging = str(tmp_path / "staging")
        writer = ZarrWriter(staging)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=10, x_start=0, x_stop=10)
        bad_embeddings = np.zeros((5, 5, EMBEDDING_DIM), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected shape"):
            writer.write_chunk(chunk, bad_embeddings, run_id="test", scales=_dummy_scales(5, 5))

    def test_coordinates_in_output(self, tmp_path):
        """Written Zarr has correct northing, easting, band coordinates."""
        staging = str(tmp_path / "staging")
        writer = ZarrWriter(staging)
        chunk = ChunkSpec(row=1, col=2, y_start=1500, y_stop=3000, x_start=3000, x_stop=4500)
        embeddings = np.zeros((1500, 1500, EMBEDDING_DIM), dtype=np.float32)

        path = writer.write_chunk(chunk, embeddings, run_id="coord_test", scales=_dummy_scales(1500, 1500))

        ds = xr.open_zarr(path)
        np.testing.assert_array_equal(ds.coords["northing"].values, np.arange(1500, 3000))
        np.testing.assert_array_equal(ds.coords["easting"].values, np.arange(3000, 4500))
        np.testing.assert_array_equal(ds.coords["band"].values, np.arange(EMBEDDING_DIM))

    def test_write_chunk_with_obs_counts(self, tmp_path):
        """Write chunk with obs counts dict, read back, verify shape/dtype/values."""
        staging = str(tmp_path / "staging")
        writer = ZarrWriter(staging)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=10, x_start=0, x_stop=10)

        rng = np.random.default_rng(42)
        embeddings = rng.random((10, 10, EMBEDDING_DIM)).astype(np.float32)
        s2_obs = rng.integers(0, 200, size=(10, 10)).astype(np.uint16)
        s1_asc_obs = rng.integers(0, 50, size=(10, 10)).astype(np.uint16)
        s1_desc_obs = rng.integers(0, 50, size=(10, 10)).astype(np.uint16)

        path = writer.write_chunk(
            chunk,
            embeddings,
            run_id="obs_test",
            scales=_dummy_scales(10, 10),
            obs_counts={
                "s2_obs_count": s2_obs,
                "s1_asc_obs_count": s1_asc_obs,
                "s1_desc_obs_count": s1_desc_obs,
            },
        )

        ds = xr.open_zarr(path)
        for var_name, expected in [
            ("s2_obs_count", s2_obs),
            ("s1_asc_obs_count", s1_asc_obs),
            ("s1_desc_obs_count", s1_desc_obs),
        ]:
            assert var_name in ds
            result = ds[var_name].values
            assert result.shape == (10, 10)
            assert result.dtype == np.uint16
            np.testing.assert_array_equal(result, expected)

    def test_write_chunk_without_obs_counts_backward_compat(self, tmp_path):
        """Write without obs counts, verify no obs count vars in zarr."""
        staging = str(tmp_path / "staging")
        writer = ZarrWriter(staging)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=10, x_start=0, x_stop=10)

        embeddings = np.zeros((10, 10, EMBEDDING_DIM), dtype=np.float32)
        path = writer.write_chunk(chunk, embeddings, run_id="no_obs", scales=_dummy_scales(10, 10))

        ds = xr.open_zarr(path)
        for var_name in OBS_COUNT_VARS:
            assert var_name not in ds


class TestAssembly:
    """Tests for ZarrWriter.assemble (raw-zarr fork/merge engine)."""

    def test_assemble_multiple_chunks(self, tmp_path):
        """Assemble 2x2 chunk grid into single output store."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "assemble_test"

        rng = np.random.default_rng(42)
        chunks = _GRID_2X2

        expected = _stage_chunks(writer, chunks, run_id, rng, (8, 10))

        writer.assemble(
            chunks,
            total_y=8,
            total_x=10,
            run_id=run_id,
            output_path=output,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 8, 10),
            n_workers=1,
        )

        # Read back assembled output (Icechunk store)
        ds = open_store(output)
        result = ds["embeddings"].values[0, ...]
        np.testing.assert_array_equal(result, expected)
        assert ds["embeddings"].shape == (1, 8, 10, EMBEDDING_DIM)
        assert "time" in ds.coords

    def test_assemble_xarray_readable(self, tmp_path):
        """Assembled output must be readable with correct dims and coords."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "xr_test"

        rng = np.random.default_rng(99)
        chunks = _GRID_1X2

        expected = _stage_chunks(writer, chunks, run_id, rng, (5, 10))

        writer.assemble(
            chunks,
            total_y=5,
            total_x=10,
            run_id=run_id,
            output_path=output,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 5, 10),
            n_workers=1,
        )

        ds = open_store(output)
        assert set(ds.data_vars) >= {"embeddings", "scales"}
        assert ds["embeddings"].dims == ("time", "northing", "easting", "band")
        assert ds["embeddings"].shape == (1, 5, 10, EMBEDDING_DIM)
        np.testing.assert_array_equal(ds["embeddings"].values[0, ...], expected)
        assert np.issubdtype(ds.coords["time"].values.dtype, np.datetime64)
        np.testing.assert_array_equal(ds.coords["northing"].values, np.arange(5))
        np.testing.assert_array_equal(ds.coords["easting"].values, np.arange(10))
        np.testing.assert_array_equal(ds.coords["band"].values, np.arange(EMBEDDING_DIM))

    def test_assemble_with_std(self, tmp_path):
        """Assembly includes embedding_std when compute_std=True."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)

        rng = np.random.default_rng(42)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=4, x_start=0, x_stop=4)

        emb, scales = _quantized_embeddings(rng, 4, 4)
        std = rng.random((4, 4, EMBEDDING_DIM)).astype(np.float32)
        writer.write_chunk(chunk, emb, run_id="run1", scales=scales, embeddings_std=std)
        writer.assemble(
            [chunk],
            total_y=4,
            total_x=4,
            run_id="run1",
            output_path=output,
            compute_std=True,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 4, 4),
            n_workers=1,
        )

        ds = open_store(output)
        assert ds["embeddings"].shape == (1, 4, 4, EMBEDDING_DIM)
        assert ds["embedding_std"].shape == (1, 4, 4, EMBEDDING_DIM)
        np.testing.assert_array_equal(ds["embeddings"].values[0, ...], emb)
        np.testing.assert_array_almost_equal(ds["embedding_std"].values[0, ...], std, decimal=5)

    def test_assemble_appends_to_existing_store(self, tmp_path):
        """A second assemble at a NEW time value extends the time axis."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)

        rng = np.random.default_rng(42)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5)

        roi = _make_full_roi_mask(tmp_path, 5, 5)
        day1 = datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC)
        day2 = datetime.datetime(2025, 6, 1, tzinfo=datetime.UTC)

        # Run 1
        emb1, scales1 = _quantized_embeddings(rng, 5, 5)
        writer.write_chunk(chunk, emb1, run_id="run1", scales=scales1)
        writer.assemble(
            [chunk],
            total_y=5,
            total_x=5,
            run_id="run1",
            output_path=output,
            roi_zarr_path=roi,
            run_started_at=day1,
            n_workers=1,
        )

        ds = open_store(output)
        assert ds["embeddings"].shape == (1, 5, 5, EMBEDDING_DIM)
        ds.close()

        # Run 2 at a later date — should append
        emb2, scales2 = _quantized_embeddings(rng, 5, 5)
        writer.write_chunk(chunk, emb2, run_id="run2", scales=scales2)
        writer.assemble(
            [chunk],
            total_y=5,
            total_x=5,
            run_id="run2",
            output_path=output,
            roi_zarr_path=roi,
            run_started_at=day2,
            n_workers=1,
        )

        ds = open_store(output)
        assert ds["embeddings"].shape == (2, 5, 5, EMBEDDING_DIM)
        assert ds.sizes["time"] == 2
        np.testing.assert_array_equal(ds["embeddings"].values[0, ...], emb1)
        np.testing.assert_array_equal(ds["embeddings"].values[1, ...], emb2)
        np.testing.assert_array_equal(
            ds.coords["time"].values, np.array(["2024-06-01", "2025-06-01"], dtype="datetime64[ns]")
        )

    def test_assemble_same_date_overwrites_in_place(self, tmp_path):
        """Re-assembling an existing time value overwrites that index (idempotent resume)."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)

        rng = np.random.default_rng(42)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5)
        roi = _make_full_roi_mask(tmp_path, 5, 5)
        day = datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC)

        emb1, scales1 = _quantized_embeddings(rng, 5, 5)
        writer.write_chunk(chunk, emb1, run_id="run1", scales=scales1)
        writer.assemble(
            [chunk],
            total_y=5,
            total_x=5,
            run_id="run1",
            output_path=output,
            roi_zarr_path=roi,
            run_started_at=day,
            n_workers=1,
        )

        emb2, scales2 = _quantized_embeddings(rng, 5, 5)
        writer.write_chunk(chunk, emb2, run_id="run2", scales=scales2)
        writer.assemble(
            [chunk],
            total_y=5,
            total_x=5,
            run_id="run2",
            output_path=output,
            roi_zarr_path=roi,
            run_started_at=day,
            n_workers=1,
        )

        ds = open_store(output)
        assert ds.sizes["time"] == 1, "same time value must overwrite, not append a duplicate timestep"
        np.testing.assert_array_equal(ds["embeddings"].values[0, ...], emb2)
        assert ds.attrs["run_id"] == "run2"

    def test_same_date_overwrite_clears_skip_marked_footprints(self, tmp_path):
        """A rerun's skip marker must reset the chunk to fill, not expose the prior run's data."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)

        rng = np.random.default_rng(42)
        chunks = _GRID_1X2
        roi = _make_full_roi_mask(tmp_path, 5, 10)
        day = datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC)

        # Run 1: both chunks carry real data.
        for chunk in chunks:
            emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
            writer.write_chunk(chunk, emb, "run1", scales=scales)
        writer.assemble(
            chunks,
            total_y=5,
            total_x=10,
            run_id="run1",
            output_path=output,
            roi_zarr_path=roi,
            run_started_at=day,
            n_workers=1,
        )

        # Run 2, same date: chunk_0_1 now has zero valid pixels (skip marker).
        emb2, scales2 = _quantized_embeddings(rng, 5, 5)
        writer.write_chunk(chunks[0], emb2, "run2", scales=scales2)
        writer.write_skip_marker(chunks[1], "run2")
        writer.assemble(
            chunks,
            total_y=5,
            total_x=10,
            run_id="run2",
            output_path=output,
            roi_zarr_path=roi,
            run_started_at=day,
            n_workers=1,
        )

        ds = open_store(output)
        assert ds.sizes["time"] == 1
        result = ds["embeddings"].values[0, ...]
        np.testing.assert_array_equal(result[:, :5, :], emb2)
        assert np.all(result[:, 5:, :] == 0), "run1's data must not survive under run2's skip marker"
        assert np.all(np.isnan(ds["scales"].values[0, :, 5:])), "scales must reset to NaN fill too"

    def test_same_date_overwrite_with_shrunken_roi_clears_outside(self, tmp_path):
        """A same-date overwrite under a SMALLER ROI clears chunks that were live in
        the prior (larger) ROI but fall outside the new one — otherwise run 1's
        embeddings there stay published (stale) under run 2's narrower footprint.
        """
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        rng = np.random.default_rng(42)
        chunks = _GRID_1X2
        day = datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC)

        # Run 1: FULL ROI — both chunks carry real data.
        for chunk in chunks:
            emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
            writer.write_chunk(chunk, emb, "run1", scales=scales)
        writer.assemble(
            chunks,
            total_y=5,
            total_x=10,
            run_id="run1",
            output_path=output,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 5, 10),
            run_started_at=day,
            n_workers=1,
        )

        # Run 2, same date: SHRUNKEN ROI (only cols 0-5). Only chunk_0_0 is staged;
        # chunk_0_1 is now OUTSIDE the ROI (not merely skip-marked).
        emb2, scales2 = _quantized_embeddings(rng, 5, 5)
        writer.write_chunk(chunks[0], emb2, "run2", scales=scales2)
        writer.assemble(
            chunks,
            total_y=5,
            total_x=10,
            run_id="run2",
            output_path=output,
            roi_zarr_path=_make_partial_roi_mask(tmp_path, 5, 10, x_true_stop=5),
            run_started_at=day,
            n_workers=1,
        )

        ds = open_store(output)
        assert ds.sizes["time"] == 1
        result = ds["embeddings"].values[0, ...]
        np.testing.assert_array_equal(result[:, :5, :], emb2)
        assert np.all(result[:, 5:, :] == 0), "run1's data outside the shrunken ROI must be cleared"

    def test_assemble_multiband_parallel_matches_serial(self, tmp_path):
        """n_workers>1 partitions into y-bands across processes; result matches serial.

        The grid is taller than one output chunk (SINGLE 500-px northing chunks)
        with tile boundaries that do NOT fall on chunk boundaries, so the run
        exercises band-aligned splitting of staged tiles and x/y partial-chunk
        read-modify-writes inside a fork.
        """
        rng = np.random.default_rng(7)
        total_y, total_x = 1100, 12
        chunks = [
            ChunkSpec(row=0, col=0, y_start=0, y_stop=600, x_start=0, x_stop=12),
            ChunkSpec(row=1, col=0, y_start=600, y_stop=1100, x_start=0, x_stop=12),
        ]
        roi = _make_full_roi_mask(tmp_path, total_y, total_x)

        expected = np.zeros((total_y, total_x, EMBEDDING_DIM), dtype=np.int8)
        staged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for chunk in chunks:
            emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
            staged[chunk.label] = (emb, scales)
            expected[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop, :] = emb

        outputs = {}
        for name, n_workers in (("serial", 1), ("parallel", 2)):
            writer = ZarrWriter(str(tmp_path / f"staging_{name}"))
            for chunk in chunks:
                emb, scales = staged[chunk.label]
                writer.write_chunk(chunk, emb, run_id="run1", scales=scales)
            output = str(tmp_path / f"output_{name}.zarr")
            writer.assemble(
                chunks,
                total_y=total_y,
                total_x=total_x,
                run_id="run1",
                output_path=output,
                roi_zarr_path=roi,
                n_workers=n_workers,
            )
            outputs[name] = output

        ds_serial = open_store(outputs["serial"])
        ds_parallel = open_store(outputs["parallel"])
        np.testing.assert_array_equal(ds_parallel["embeddings"].values[0, ...], expected)
        np.testing.assert_array_equal(ds_parallel["embeddings"].values, ds_serial["embeddings"].values)
        np.testing.assert_array_equal(ds_parallel["scales"].values, ds_serial["scales"].values)

    def test_emits_one_assembly_summary_record(self, tmp_path, caplog):
        """One ASSEMBLY_SUMMARY per assemble — never per tile — whose counts match the staging."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        rng = np.random.default_rng(23)
        _stage_chunks(writer, _GRID_1X2, "run1", rng, (5, 10))
        roi = _make_full_roi_mask(tmp_path, 5, 10)
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.inference.assembly"):
            writer.assemble(
                _GRID_1X2,
                total_y=5,
                total_x=10,
                run_id="run1",
                output_path=str(tmp_path / "out.zarr"),
                roi_zarr_path=roi,
                n_workers=1,
            )
        (rec,) = _assembly_summary_records(caplog)
        assert rec["run"] == "run1"
        assert rec["tiles_staged"] == 2 and rec["tiles_cleared"] == 0
        assert rec["workers_requested"] == 1 and rec["workers_used"] == 1
        assert rec["per_worker_s3_cap"] == TARGET_AGGREGATE_S3_CONCURRENCY
        assert rec["tiles"] == 2
        assert rec["writes"] == 4  # 2 tiles x (embeddings + scales)
        # Uncompressed bytes handed to zarr: 5x5 px per tile, int8 emb + f32 scales.
        assert rec["bytes"] == 2 * (5 * 5 * EMBEDDING_DIM + 5 * 5 * 4)
        assert rec["fused_compress_put"] is True
        (worker,) = rec["workers"]
        assert worker["read_s"] >= 0 and worker["write_s"] >= 0
        assert rec["commit_s"] >= 0 and rec["total_s"] > 0

    def test_assemble_append_with_std(self, tmp_path):
        """Append preserves both embedding and embedding_std across runs."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)

        rng = np.random.default_rng(42)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=4, x_start=0, x_stop=4)

        roi = _make_full_roi_mask(tmp_path, 4, 4)

        # Run 1 with std
        emb1, scales1 = _quantized_embeddings(rng, 4, 4)
        std1 = rng.random((4, 4, EMBEDDING_DIM)).astype(np.float32)
        writer.write_chunk(chunk, emb1, run_id="run1", scales=scales1, embeddings_std=std1)
        writer.assemble(
            [chunk],
            total_y=4,
            total_x=4,
            run_id="run1",
            output_path=output,
            compute_std=True,
            roi_zarr_path=roi,
            run_started_at=datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC),
            n_workers=1,
        )

        # Run 2 with std, at a later date
        emb2, scales2 = _quantized_embeddings(rng, 4, 4)
        std2 = rng.random((4, 4, EMBEDDING_DIM)).astype(np.float32)
        writer.write_chunk(chunk, emb2, run_id="run2", scales=scales2, embeddings_std=std2)
        writer.assemble(
            [chunk],
            total_y=4,
            total_x=4,
            run_id="run2",
            output_path=output,
            compute_std=True,
            roi_zarr_path=roi,
            run_started_at=datetime.datetime(2025, 6, 1, tzinfo=datetime.UTC),
            n_workers=1,
        )

        ds = open_store(output)
        assert ds["embeddings"].shape == (2, 4, 4, EMBEDDING_DIM)
        assert ds["embedding_std"].shape == (2, 4, 4, EMBEDDING_DIM)
        np.testing.assert_array_almost_equal(ds["embedding_std"].values[0, ...], std1, decimal=5)
        np.testing.assert_array_almost_equal(ds["embedding_std"].values[1, ...], std2, decimal=5)

    def test_same_date_overwrite_resets_untouched_vars_to_fill(self, tmp_path):
        """A same-date overwrite that drops a variable resets that variable's
        slice to fill, so no stale metadata describes the overwritten data.
        """
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        rng = np.random.default_rng(7)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=4, x_start=0, x_stop=4)
        roi = _make_full_roi_mask(tmp_path, 4, 4)
        started = datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC)

        # Run 1: with std.
        emb1, scales1 = _quantized_embeddings(rng, 4, 4)
        std1 = rng.random((4, 4, EMBEDDING_DIM)).astype(np.float32)
        writer.write_chunk(chunk, emb1, run_id="run1", scales=scales1, embeddings_std=std1)
        writer.assemble(
            [chunk],
            total_y=4,
            total_x=4,
            run_id="run1",
            output_path=output,
            compute_std=True,
            roi_zarr_path=roi,
            run_started_at=started,
            n_workers=1,
        )

        # Run 2: SAME date, WITHOUT std — embedding_std is now untouched and stale.
        emb2, scales2 = _quantized_embeddings(rng, 4, 4)
        writer.write_chunk(chunk, emb2, run_id="run2", scales=scales2)
        writer.assemble(
            [chunk],
            total_y=4,
            total_x=4,
            run_id="run2",
            output_path=output,
            compute_std=False,
            roi_zarr_path=roi,
            run_started_at=started,
            n_workers=1,
        )

        ds = open_store(output)
        assert ds["embeddings"].shape == (1, 4, 4, EMBEDDING_DIM)  # overwrite, not append
        # embedding_std at the overwritten timestep is reset to fill (NaN), not stale std1.
        assert np.isnan(ds["embedding_std"].values[0]).all()
        # The atomic-publish work branch is cleaned up after a successful assemble.
        assert "_assemble-wip" not in open_or_create_repo(output)[0].list_branches()

    def test_append_preserves_time_windows_across_runs(self, tmp_path):
        """Regression: appending a second time window must merge into existing time_windows, not overwrite."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)

        rng = np.random.default_rng(42)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5)

        # Window 1: Jan-Dec 2024
        tw1 = TimeWindow(
            window_start=(2024, 1),
            window_end=(2024, 12),
            months=tuple((2024, m) for m in range(1, 13)),
            window_end_label="2024-12-01",
        )
        roi = _make_full_roi_mask(tmp_path, 5, 5)

        emb1, scales1 = _quantized_embeddings(rng, 5, 5)
        writer.write_chunk(chunk, emb1, run_id="run1", scales=scales1)
        writer.assemble(
            [chunk],
            total_y=5,
            total_x=5,
            run_id="run1",
            output_path=output,
            roi_zarr_path=roi,
            time_window=tw1,
            n_workers=1,
        )

        ds1 = open_store(output)
        tw_dict = ds1.attrs["time_windows"]
        assert "2024-12-01" in tw_dict
        assert tw_dict["2024-12-01"]["range"] == ["2024-01", "2024-12"]
        ds1.close()

        # Window 2: Feb 2024 - Jan 2025
        tw2 = TimeWindow(
            window_start=(2024, 2),
            window_end=(2025, 1),
            months=tuple([(2024, m) for m in range(2, 13)] + [(2025, 1)]),
            window_end_label="2025-01-01",
        )
        emb2, scales2 = _quantized_embeddings(rng, 5, 5)
        writer.write_chunk(chunk, emb2, run_id="run2", scales=scales2)
        writer.assemble(
            [chunk],
            total_y=5,
            total_x=5,
            run_id="run2",
            output_path=output,
            roi_zarr_path=roi,
            time_window=tw2,
            n_workers=1,
        )

        ds2 = open_store(output)
        assert ds2.sizes["time"] == 2
        tw_dict2 = ds2.attrs["time_windows"]
        # Both entries must be present
        assert "2024-12-01" in tw_dict2, "First window entry was overwritten on append"
        assert "2025-01-01" in tw_dict2, "Second window entry missing"
        assert tw_dict2["2024-12-01"]["range"] == ["2024-01", "2024-12"]
        assert tw_dict2["2025-01-01"]["range"] == ["2024-02", "2025-01"]

    def test_assemble_with_obs_counts(self, tmp_path):
        """Stage 2x2 grid with obs counts, assemble, verify final zarr has all three vars."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "obs_assemble"

        rng = np.random.default_rng(42)
        chunks = _GRID_2X2

        expected_obs = {var: np.zeros((8, 10), dtype=np.uint16) for var in OBS_COUNT_VARS}
        for chunk in chunks:
            emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
            obs = {
                var: rng.integers(0, 200, size=(chunk.height, chunk.width)).astype(np.uint16) for var in OBS_COUNT_VARS
            }
            writer.write_chunk(chunk, emb, run_id, scales=scales, obs_counts=obs)
            for var in OBS_COUNT_VARS:
                expected_obs[var][chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop] = obs[var]

        writer.assemble(
            chunks,
            total_y=8,
            total_x=10,
            run_id=run_id,
            output_path=output,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 8, 10),
            n_workers=1,
        )

        ds = open_store(output)
        for var in OBS_COUNT_VARS:
            assert var in ds
            assert ds[var].dims == ("time", "northing", "easting")
            assert ds[var].shape == (1, 8, 10)
            assert ds[var].dtype == np.uint16
            np.testing.assert_array_equal(ds[var].values[0], expected_obs[var])

    def test_assemble_treats_skip_marker_chunks_as_fill(self, tmp_path):
        """A live chunk with a skip marker (no staged zarr) falls through to zero-fill.

        Regression (originally against the Dask engine): skip-marked chunks must
        never be treated as staged — a worker opening the non-existent zarr
        would raise GroupNotFoundError. Their footprint stays at the fill value.
        """
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "skip_marker_test"

        rng = np.random.default_rng(7)
        chunks = _GRID_2X2

        # Three chunks have real data; chunk_1_0 gets a skip marker only.
        skipped_chunk = chunks[2]
        expected = np.zeros((8, 10, EMBEDDING_DIM), dtype=np.int8)
        for chunk in chunks:
            if chunk is skipped_chunk:
                writer.write_skip_marker(chunk, run_id)
                continue
            emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
            writer.write_chunk(chunk, emb, run_id, scales=scales)
            expected[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop, :] = emb

        writer.assemble(
            chunks,
            total_y=8,
            total_x=10,
            run_id=run_id,
            output_path=output,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 8, 10),
            n_workers=1,
        )

        result = _assembled(output)
        # Skipped chunk's footprint is zeros (int8 fill); other chunks match what we staged.
        np.testing.assert_array_equal(result, expected)
        skipped_region = result[
            skipped_chunk.y_start : skipped_chunk.y_stop, skipped_chunk.x_start : skipped_chunk.x_stop
        ]
        assert skipped_region.shape == (skipped_chunk.height, skipped_chunk.width, EMBEDDING_DIM)
        assert np.all(skipped_region == 0)

    def test_assemble_all_skipped_publishes_all_fill(self, tmp_path):
        """Every ROI chunk skip-marked (no valid pixels) → publish an all-fill
        timestep, not abort (the old engine could publish an all-fill output).
        """
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "all_skipped"
        chunks = _GRID_1X2
        for chunk in chunks:
            writer.write_skip_marker(chunk, run_id)

        writer.assemble(
            chunks,
            total_y=5,
            total_x=10,
            run_id=run_id,
            output_path=output,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 5, 10),
            n_workers=1,
        )

        ds = open_store(output)
        assert ds["embeddings"].shape == (1, 5, 10, EMBEDDING_DIM)
        assert bool(np.all(ds["embeddings"].values[0] == 0))  # all-fill timestep published

    def test_assemble_sparse_roi_non_live_chunks_are_zero_filled(self, tmp_path):
        """Chunks outside the ROI mask are zero-filled without requiring staged data."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "sparse_roi_test"

        # 3x3 chunk grid; mask excludes the interior chunk (row=1, col=1)
        chunks = [
            ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5),
            ChunkSpec(row=0, col=1, y_start=0, y_stop=5, x_start=5, x_stop=10),
            ChunkSpec(row=0, col=2, y_start=0, y_stop=5, x_start=10, x_stop=15),
            ChunkSpec(row=1, col=0, y_start=5, y_stop=10, x_start=0, x_stop=5),
            ChunkSpec(row=1, col=1, y_start=5, y_stop=10, x_start=5, x_stop=10),  # non-live
            ChunkSpec(row=1, col=2, y_start=5, y_stop=10, x_start=10, x_stop=15),
            ChunkSpec(row=2, col=0, y_start=10, y_stop=13, x_start=0, x_stop=5),
            ChunkSpec(row=2, col=1, y_start=10, y_stop=13, x_start=5, x_stop=10),
            ChunkSpec(row=2, col=2, y_start=10, y_stop=13, x_start=10, x_stop=15),
        ]
        non_live = chunks[4]  # row=1, col=1

        # ROI mask: all True except the non-live chunk's footprint
        roi_path = str(tmp_path / "roi_mask.zarr")
        roi_arr = zarr.open(roi_path, mode="w", shape=(13, 15), chunks=(13, 15), dtype="bool")
        roi_arr[:] = True
        roi_arr[non_live.y_start : non_live.y_stop, non_live.x_start : non_live.x_stop] = False

        rng = np.random.default_rng(13)
        expected = np.zeros((13, 15, EMBEDDING_DIM), dtype=np.int8)
        for chunk in chunks:
            if chunk is non_live:
                continue  # no staged data for this chunk
            emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
            writer.write_chunk(chunk, emb, run_id, scales=scales)
            expected[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop, :] = emb

        writer.assemble(
            chunks,
            total_y=13,
            total_x=15,
            run_id=run_id,
            output_path=output,
            roi_zarr_path=roi_path,
            n_workers=1,
        )

        result = _assembled(output)
        np.testing.assert_array_equal(result, expected)
        non_live_region = result[non_live.y_start : non_live.y_stop, non_live.x_start : non_live.x_stop]
        assert np.all(non_live_region == 0), "Non-live chunk region must be zero-filled"

    def test_assemble_sparse_roi_scales_nan_filled_for_non_live(self, tmp_path):
        """`scales` variable is NaN-filled for chunks outside the ROI mask."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "sparse_scales_test"

        chunks = [
            ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5),
            ChunkSpec(row=0, col=1, y_start=0, y_stop=5, x_start=5, x_stop=10),
            ChunkSpec(row=1, col=0, y_start=5, y_stop=9, x_start=0, x_stop=5),
            ChunkSpec(row=1, col=1, y_start=5, y_stop=9, x_start=5, x_stop=10),  # non-live
        ]
        non_live = chunks[3]

        roi_path = str(tmp_path / "roi_mask.zarr")
        roi_arr = zarr.open(roi_path, mode="w", shape=(9, 10), chunks=(9, 10), dtype="bool")
        roi_arr[:] = True
        roi_arr[non_live.y_start : non_live.y_stop, non_live.x_start : non_live.x_stop] = False

        rng = np.random.default_rng(21)
        for chunk in chunks:
            if chunk is non_live:
                continue
            emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
            writer.write_chunk(chunk, emb, run_id, scales=scales)

        writer.assemble(
            chunks,
            total_y=9,
            total_x=10,
            run_id=run_id,
            output_path=output,
            roi_zarr_path=roi_path,
            n_workers=1,
        )

        ds = open_store(output)
        scales_out = ds["scales"].values[0, ...]  # (H, W)
        non_live_scales = scales_out[non_live.y_start : non_live.y_stop, non_live.x_start : non_live.x_stop]
        assert np.all(np.isnan(non_live_scales)), "Non-live chunk scales must be NaN-filled"
        live_scales = scales_out[: non_live.y_start, :]
        assert not np.any(np.isnan(live_scales)), "Live chunk scales must not be NaN"


class TestCleanupStaging:
    """cleanup_staging delegates to the shared prefix-delete helper.

    The delete internals (s5cmd --all-versions, fsspec fallback) are
    object_store.delete_prefix's concern (see test_object_store); here we only
    pin that the staging dir for the given run is what gets removed.
    """

    def test_delegates_to_delete_prefix_with_run_target(self):
        writer = ZarrWriter("s3://bucket/staging")
        with patch.object(_assembly_mod, "delete_prefix") as delete_prefix:
            writer.cleanup_staging("run123")
        delete_prefix.assert_called_once()
        assert delete_prefix.call_args.args[0] == "s3://bucket/staging/run123"

    def test_only_the_target_run_is_deleted(self):
        writer = ZarrWriter("s3://bucket/staging")
        with patch.object(_assembly_mod, "delete_prefix") as delete_prefix:
            writer.cleanup_staging("delete_me")
        target = delete_prefix.call_args.args[0]
        assert "delete_me" in target and "keep_me" not in target


class TestDetectStagedChunkSize:
    """Tests for the public detect_staged_chunk_size method."""

    def test_raises_when_no_staged_chunks(self, tmp_path):
        """FileNotFoundError when the staging directory is empty."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        with pytest.raises(FileNotFoundError, match="No staged chunks found"):
            writer.detect_staged_chunk_size("run1")

    # The extents below are deliberately SMALL. What is under test is that the method reads a
    # size off whichever chunk it finds, not that it can handle a production-sized one — and
    # the size is read from the array's shape, so it is scale-free. A real 2000-px chunk is
    # 2000 x 2000 x 128 int8, half a gigabyte to generate, quantise and write, which cost 10 s
    # of suite wall time to assert an integer.

    def test_uses_first_available_chunk(self, tmp_path):
        """Returns chunk_size from first staged label — not chunk_0_0 specifically."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        # Stage only a non-origin chunk (simulates sparse ROI where chunk_0_0 is empty)
        chunk = ChunkSpec(row=3, col=5, y_start=600, y_stop=800, x_start=1000, x_stop=1200)
        rng = np.random.default_rng(0)
        emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
        writer.write_chunk(chunk, emb, "run1", scales=scales)

        result = writer.detect_staged_chunk_size("run1")
        assert result == 200

    def test_returns_max_of_height_width(self, tmp_path):
        """Edge chunks with non-square extents return max(h, w)."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        # Non-square edge chunk: height=50, width=200 — the wider side must win
        chunk = ChunkSpec(row=0, col=1, y_start=0, y_stop=50, x_start=200, x_stop=400)
        rng = np.random.default_rng(1)
        emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
        writer.write_chunk(chunk, emb, "run1", scales=scales)

        result = writer.detect_staged_chunk_size("run1")
        assert result == 200


class _RecordingMarkerFS:
    """Filesystem proxy that records the completion-marker operations, in order."""

    def __init__(self, inner, seen: list[str]) -> None:
        self._inner, self._seen = inner, seen

    def rm(self, *args, **kwargs):
        # Which marker, not just that one was retracted: write_chunk clears BOTH the
        # completion marker and any stale skip marker, and the order of the pair
        # relative to the tile write is the invariant under test.
        self._seen.append(f"retract:{str(args[0]).rsplit('.', 1)[-1]}")
        return self._inner.rm(*args, **kwargs)

    def open(self, *args, **kwargs):
        self._seen.append("done")
        return self._inner.open(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _mark_staged_complete(path: str) -> None:
    """Record a hand-built staged tile as a COMPLETED write, both ways.

    ``write_chunk`` records completion twice — the in-store ``staged_complete``
    attribute (checked by readers that already have the group open) and the
    sibling ``<label>.done`` object (visible in the staging listing). A test that
    hand-rolls a tile and sets only one of them is not simulating a completed
    write; it is simulating a crash between the two.
    """
    zarr.open_group(path, mode="a").attrs["staged_complete"] = True
    Path(path.removesuffix(".zarr") + ".done").write_bytes(b"")


def _write_monolithic_staged_zarr(path: str, embeddings: np.ndarray, y_start: int, x_start: int) -> None:
    """Write a staged Zarr with monolithic chunks (no sub-chunking).

    Mimics production data where chunks span the full spatial extent of the
    staged file (e.g. 2500x2500x128 as a single on-disk chunk).
    """
    h, w, band = embeddings.shape
    ds = xr.Dataset(
        {
            "embeddings": (["northing", "easting", "band"], embeddings),
            "scales": (["northing", "easting"], np.ones((h, w), dtype=np.float32)),
        },
        coords={
            "northing": np.arange(y_start, y_start + h),
            "easting": np.arange(x_start, x_start + w),
            "band": np.arange(band),
        },
    )
    encoding = {
        "embeddings": {
            "chunks": (h, w, band),  # monolithic — single on-disk chunk
            "compressors": None,
        },
        "scales": {
            "chunks": (h, w),
            "compressors": None,
        },
    }
    ds.to_zarr(path, mode="w", encoding=encoding)
    _mark_staged_complete(path)


class TestStagedLayout:
    """The staged-file on-disk layout: inner-chunk-sized pieces, full band, raw.

    A staged tile must be exactly the inner-chunk grid of the output region it
    becomes (ADR-008 D2: the band axis is never split), and the engine must
    also read older/foreign staged layouts (e.g. monolithic chunks) correctly —
    workers slice whatever chunk grid the staged file has.
    """

    def test_write_chunk_stages_inner_chunks_full_band(self, tmp_path):
        """write_chunk sub-chunks at INNER_PX spatially and never splits the band."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=600, x_start=0, x_stop=600)
        rng = np.random.default_rng(42)
        emb, scales = _quantized_embeddings(rng, 600, 600)
        path = writer.write_chunk(chunk, emb, "run1", scales=scales)

        g = zarr.open_group(path, mode="r")
        assert g["embeddings"].metadata.chunk_grid.chunk_shape == (INNER_PX, INNER_PX, EMBEDDING_DIM)
        assert g["scales"].metadata.chunk_grid.chunk_shape == (INNER_PX, INNER_PX)
        # Raw staging: no compressors, zero codec CPU on the GPU actors.
        assert g["embeddings"].compressors == ()

    def test_assemble_monolithic_roundtrip(self, tmp_path):
        """Assembly of monolithic-chunked staged files produces correct output."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "mono_test"

        rng = np.random.default_rng(42)
        chunks = [
            ChunkSpec(row=0, col=0, y_start=0, y_stop=8, x_start=0, x_stop=8),
            ChunkSpec(row=0, col=1, y_start=0, y_stop=8, x_start=8, x_stop=16),
            ChunkSpec(row=1, col=0, y_start=8, y_stop=14, x_start=0, x_stop=8),
            ChunkSpec(row=1, col=1, y_start=8, y_stop=14, x_start=8, x_stop=16),
        ]

        expected = np.zeros((14, 16, EMBEDDING_DIM), dtype=np.int8)
        for chunk in chunks:
            emb, _ = _quantized_embeddings(rng, chunk.height, chunk.width)
            path = writer._staging_path(run_id, chunk)
            _write_monolithic_staged_zarr(path, emb, chunk.y_start, chunk.x_start)
            expected[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop, :] = emb

        writer.assemble(
            chunks,
            total_y=14,
            total_x=16,
            run_id=run_id,
            output_path=output,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 14, 16),
            n_workers=1,
        )

        ds = open_store(output)
        result = ds["embeddings"].values[0, ...]
        np.testing.assert_array_equal(result, expected)
        assert ds["embeddings"].shape == (1, 14, 16, EMBEDDING_DIM)

    def _assemble_one_chunk(self, writer, chunk, tmp_path):
        writer.assemble(
            [chunk],
            total_y=8,
            total_x=8,
            run_id="run1",
            output_path=str(tmp_path / "output.zarr"),
            roi_zarr_path=_make_full_roi_mask(tmp_path, 8, 8),
            n_workers=1,
        )

    def test_assemble_verifies_staged_completeness_itself(self, tmp_path):
        """`assemble` gates on the staging listing without being asked to.

        It derives its own live set from the ROI mask, and it is called directly — an
        assembly-only re-run, a test — not only through the flow that used to verify
        first. So the gate has to live here, or a half-written tile reaches the workers
        and the run fails after the schema commit instead of before any work starts.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=8, x_start=0, x_stop=8)
        emb, _ = _quantized_embeddings(np.random.default_rng(1), 8, 8)
        _write_monolithic_staged_zarr(writer._staging_path("run1", chunk), emb, 0, 0)
        Path(writer._done_marker_path("run1", chunk)).unlink()  # crash before the marker

        with pytest.raises(IncompleteStageError, match="1 interrupted"):
            self._assemble_one_chunk(writer, chunk, tmp_path)

    def test_assemble_rejects_a_tile_the_listing_calls_complete(self, tmp_path):
        """The in-store check, on the one state the listing gate cannot see.

        The tile keeps its ``.done`` marker, so verification passes it — exactly the
        position a reader is in when a tile is rewritten after the listing was taken.
        Only the attribute, absent until a write finishes, reports it as partial.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=8, x_start=0, x_stop=8)
        emb, _ = _quantized_embeddings(np.random.default_rng(1), 8, 8)
        path = writer._staging_path("run1", chunk)
        _write_monolithic_staged_zarr(path, emb, 0, 0)
        del zarr.open_group(path, mode="a").attrs["staged_complete"]
        assert writer._list_staged_labels("run1") == [chunk.label]

        with pytest.raises(IncompleteStageError, match="staged_complete"):
            self._assemble_one_chunk(writer, chunk, tmp_path)


class TestPartitionBands:
    """Band partitioning: granularity-aligned, disjoint, covering, load-balanced."""

    @pytest.mark.parametrize(
        ("total_y", "granularity", "n_workers", "expected"),
        [
            (100, 500, 4, [(0, 100)]),  # smaller than one unit -> one band
            (4500, 500, 3, [(0, 1500), (1500, 3000), (3000, 4500)]),  # even split
            (1100, 500, 2, [(0, 1000), (1000, 1100)]),  # ragged tail on the last band
            (8, 3, 2, [(0, 6), (6, 8)]),  # tiny units still align interior boundaries
        ],
    )
    def test_partition_shapes(self, total_y, granularity, n_workers, expected):
        assert _partition_bands(total_y, granularity, n_workers) == expected

    def test_weighted_partition_balances_clustered_work(self):
        """With live tiles clustered at the bottom, boundaries follow the work, not the height."""
        # 10 units of 500 px; all 20 tiles live in the last two units.
        weights = [0, 0, 0, 0, 0, 0, 0, 0, 10, 10]
        bands = _partition_bands(5000, 500, 2, weights=weights)
        assert bands == [(0, 4500), (4500, 5000)], "each band should carry ~half the live tiles"
        # Uniform weights reproduce the unweighted split.
        assert _partition_bands(5000, 500, 2, weights=[1] * 10) == _partition_bands(5000, 500, 2)

    @pytest.mark.parametrize("weights", [None, [3, 0, 0, 1, 0, 7, 0, 0, 2, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4]])
    @pytest.mark.parametrize(("total_y", "granularity", "n_workers"), [(10_000, 500, 7), (2048, 2048, 8), (1, 500, 3)])
    def test_partition_invariants(self, total_y, granularity, n_workers, weights):
        n_units = -(-total_y // granularity)
        w = weights[:n_units] if weights else None
        bands = _partition_bands(total_y, granularity, n_workers, weights=w)
        assert bands[0][0] == 0
        assert bands[-1][1] == total_y
        assert len(bands) <= n_workers
        for (_, stop), (start, _) in itertools.pairwise(bands):
            assert stop == start, "bands must tile [0, total_y) with no gaps or overlap"
            assert stop % granularity == 0, "interior boundaries must land on the write granularity"


class TestSingleRoiChainIsAligned:
    """The single-ROI chain end to end, at the real tile size, on the real geometry.

    Everything smaller than this passes whatever geometry it is handed: the assembly
    unit tests build their own tiny grids, and nothing else drives enumerate → stage →
    assemble together. So this is the test that would notice the chain coming
    unaligned — a retuned tile size, a layout edit, a band split creeping back.
    """

    #: 1.5 tiles across: a whole 2048 shard AND a ragged edge in one run.
    TOTAL = 3000

    #: Band width for the staged arrays. The geometry under test is SPATIAL, and the
    #: band axis costs 16x here — a full-width 2048x2048x128 float32 tile is ~2 GiB
    #: before quantization, which is not a thing to allocate in a unit suite. The
    #: property that matters (the band axis is never split) is preserved: with a
    #: narrower array the layout's 128-wide chunk clamps to the full width, so
    #: `chunks[3] == array width` still asserts it. `test_store_layout` pins the
    #: nominal 128 against a full-size shape, where it costs nothing.
    BAND = 8

    def _stage_and_assemble(self, tmp_path, roi_mask: str):
        chunks = enumerate_chunks(self.TOTAL, self.TOTAL, chunk_size=INFERENCE_CHUNK_SIZE)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=self.BAND)
        rng = np.random.default_rng(0)
        staged = {}
        for chunk in filter_chunks_by_roi_mask(chunks, roi_mask):
            emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width, dim=self.BAND)
            writer.write_chunk(chunk, emb, "run1", scales=scales)
            staged[chunk.label] = emb
        output = str(tmp_path / "out.zarr")
        writer.assemble(
            chunks,
            total_y=self.TOTAL,
            total_x=self.TOTAL,
            run_id="run1",
            output_path=output,
            roi_zarr_path=roi_mask,
            n_workers=1,
        )
        return staged, open_store(output)

    @pytest.fixture(scope="class")
    def full_roi_run(self, tmp_path_factory):
        """One assemble over a full-coverage ROI, shared by the tests that read it.

        Class-scoped because assembling a 3000-px mosaic is the expensive part and
        three assertions do not each need their own copy of it.
        """
        tmp_path = tmp_path_factory.mktemp("aligned_chain")
        return self._stage_and_assemble(tmp_path, _make_full_roi_mask(tmp_path, self.TOTAL, self.TOTAL))

    def test_output_geometry_matches_the_campaign(self, full_roi_run):
        """Chunks and shards are the campaign's, so nothing rechunks."""
        _, ds = full_roi_run

        emb = ds["embeddings"]
        assert emb.encoding["chunks"][:3] == (1, INNER_PX, INNER_PX)
        assert emb.encoding["shards"][:3] == (1, SHARD_PX, SHARD_PX)
        # The band axis is one piece (D2): a pixel's whole embedding is one object.
        assert emb.encoding["chunks"][3] == emb.shape[3]
        assert ds["scales"].encoding["shards"] == (1, SHARD_PX, SHARD_PX)

    def test_one_inference_tile_is_one_output_shard(self):
        """The identity the alignment exists for, asserted on the enumerated grid."""
        chunks = enumerate_chunks(self.TOTAL, self.TOTAL, chunk_size=INFERENCE_CHUNK_SIZE)
        whole = [c for c in chunks if c.height == INFERENCE_CHUNK_SIZE and c.width == INFERENCE_CHUNK_SIZE]
        assert whole, "expected at least one full-size tile"
        for chunk in whole:
            assert chunk.y_start % SHARD_PX == 0 and chunk.x_start % SHARD_PX == 0
            assert chunk.height == SHARD_PX and chunk.width == SHARD_PX

    def test_values_land_where_they_were_staged(self, full_roi_run):
        """Whole tiles and the ragged edge both round-trip to their own footprints."""
        staged, ds = full_roi_run

        result = ds["embeddings"].values[0]
        for chunk in enumerate_chunks(self.TOTAL, self.TOTAL, chunk_size=INFERENCE_CHUNK_SIZE):
            got = result[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop, :]
            np.testing.assert_array_equal(got, staged[chunk.label], err_msg=chunk.label)

    def test_outside_the_roi_stays_at_fill(self, tmp_path):
        """A chunk the ROI excludes is never staged and never written."""
        mask = _make_partial_roi_mask(tmp_path, self.TOTAL, self.TOTAL, x_true_stop=SHARD_PX)
        staged, ds = self._stage_and_assemble(tmp_path, mask)

        assert set(staged) == {"chunk_0_0", "chunk_1_0"}, "only the left column intersects"
        result = ds["embeddings"].values[0]
        assert (result[:, SHARD_PX:, :] == 0).all(), "excluded footprint must read back as fill"


class TestVariablesAddedToAnExistingStore:
    """A variable joining an existing store adopts THAT store's geometry.

    Every data variable must agree on a write granularity — `_write_granularity`
    raises otherwise, since two disagreeing arrays would let separate forks share an
    output object. So creating one array at the current preset's pitch inside a store
    tiled differently does not make a merely mixed store, it makes an unassemblable
    one. A store written before a preset changed still takes appends, and gaining a
    variable must not be what breaks it.
    """

    LEGACY_CHUNK = 500
    SHAPE = (1, 3000, 3000)

    def _legacy_store(self, tmp_path) -> str:
        """A store on a pitch the current preset does not use, with no obs counts."""
        path = str(tmp_path / "legacy.zarr")
        repo, _ = open_or_create_repo(path)
        session = repo.writable_session("main")
        root = zarr.open_group(session.store, mode="a")
        root.create_array(
            "embeddings",
            shape=(*self.SHAPE, EMBEDDING_DIM),
            chunks=(1, self.LEGACY_CHUNK, self.LEGACY_CHUNK, EMBEDDING_DIM // 32),
            dtype="int8",
            fill_value=0,
            dimension_names=("time", "northing", "easting", "band"),
        )
        root.create_array(
            "scales",
            shape=self.SHAPE,
            chunks=(1, self.LEGACY_CHUNK, self.LEGACY_CHUNK),
            dtype="float32",
            fill_value=float("nan"),
            dimension_names=("time", "northing", "easting"),
        )
        session.commit("a store on the old pitch")
        return path

    def _add(self, path, variables):
        repo, _ = open_or_create_repo(path)
        session = repo.writable_session("main")
        root = zarr.open_group(session.store, mode="a")
        create_layout_arrays(
            root,
            _layout_matching_store(root, SINGLE, variables),
            variables,
            {"time": 1, "northing": 3000, "easting": 3000, "band": EMBEDDING_DIM},
        )
        return root

    def test_added_variables_keep_the_store_assemblable(self, tmp_path):
        """The regression: obs counts at the preset's pitch made assembly raise."""
        root = self._add(self._legacy_store(tmp_path), list(OBS_COUNT_VARS))

        assert _write_granularity(root, ["embeddings", "scales", *OBS_COUNT_VARS]) == self.LEGACY_CHUNK

    def test_geometry_is_copied_per_rank(self, tmp_path):
        """3-D vars follow `scales`, 4-D vars follow `embeddings` — including its band split."""
        root = self._add(self._legacy_store(tmp_path), ["s2_obs_count", "embedding_std"])

        assert root["s2_obs_count"].chunks == root["scales"].chunks
        assert root["s2_obs_count"].shards is None
        assert root["embedding_std"].chunks == root["embeddings"].chunks

    def test_only_geometry_is_borrowed_not_dtype_or_fill(self, tmp_path):
        """The donor supplies tiling; the variable keeps its own type and fill."""
        root = self._add(self._legacy_store(tmp_path), ["s2_obs_count"])

        assert root["s2_obs_count"].dtype == np.uint16  # not scales' float32
        assert root["s2_obs_count"].fill_value == 0  # not scales' NaN

    def test_a_fresh_store_still_gets_the_current_preset(self, tmp_path):
        """With nothing to match, the layout is used unchanged."""
        path = str(tmp_path / "fresh.zarr")
        repo, _ = open_or_create_repo(path)
        session = repo.writable_session("main")
        root = zarr.open_group(session.store, mode="a")
        create_layout_arrays(
            root,
            _layout_matching_store(root, SINGLE, ["embeddings"]),
            ["embeddings"],
            {"time": 1, "northing": 3000, "easting": 3000, "band": EMBEDDING_DIM},
        )

        assert root["embeddings"].chunks == (1, INNER_PX, INNER_PX, EMBEDDING_DIM)
        assert root["embeddings"].shards == (1, SHARD_PX, SHARD_PX, EMBEDDING_DIM)


class TestAssemblyValidation:
    """Tests for assembly-time validation and metadata correctness."""

    def test_assemble_validates_grid_extent(self, tmp_path):
        """Mismatched total_y/total_x raises ValueError."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "extent_test"

        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5)
        emb, scales = _quantized_embeddings(np.random.default_rng(42), 5, 5)
        writer.write_chunk(chunk, emb, run_id, scales=scales)

        with pytest.raises(ValueError, match="doesn't match chunk grid extent"):
            writer.assemble(
                [chunk],
                total_y=10,
                total_x=10,
                run_id=run_id,
                output_path=output,
                roi_zarr_path=_make_full_roi_mask(tmp_path, 10, 10),
                n_workers=1,
            )

    def test_assemble_root_attrs(self, tmp_path):
        """Assembled store has correct root attributes."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "attrs_test"

        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=4, x_start=0, x_stop=4)
        emb, scales = _quantized_embeddings(np.random.default_rng(42), 4, 4)
        writer.write_chunk(chunk, emb, run_id, scales=scales)
        writer.assemble(
            [chunk],
            total_y=4,
            total_x=4,
            run_id=run_id,
            output_path=output,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 4, 4),
            n_workers=1,
        )

        repo, _ = open_or_create_repo(output)
        session = repo.writable_session("main")
        root = zarr.open_group(session.store, mode="r")
        attrs = dict(root.attrs)

        assert attrs["run_id"] == run_id
        assert attrs["total_y"] == 4
        assert attrs["total_x"] == 4
        assert attrs["embedding_dim"] == EMBEDDING_DIM
        assert "run_started_at" in attrs
        assert "run_completed_at" in attrs

    def test_assemble_sets_geozarr_convention_attrs(self, tmp_path):
        """Convention attrs (proj:, spatial:, geoemb:) are set on the root group."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "conv_test"

        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=4, x_start=0, x_stop=4)
        emb = np.random.default_rng(42).standard_normal((4, 4, EMBEDDING_DIM)).astype(np.float32)
        quantized, scales = quantize_embeddings(emb)
        writer.write_chunk(chunk, quantized, run_id, scales=scales)

        writer.assemble(
            [chunk],
            total_y=4,
            total_x=4,
            run_id=run_id,
            output_path=output,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 4, 4),
            tile_id="33UWP",
            model_version="test_model_v1",
            n_workers=1,
        )

        repo, _ = open_or_create_repo(output)
        session = repo.writable_session("main")
        root = zarr.open_group(session.store, mode="r")
        attrs = dict(root.attrs)

        # Convention registration
        assert "zarr_conventions" in attrs
        names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj:" in names
        assert "geoemb:" in names

        # proj:
        assert attrs["proj:code"] == "EPSG:32633"

        # geoemb:
        assert attrs["geoemb:type"] == "pixel"
        assert attrs["geoemb:dimensions"] == EMBEDDING_DIM
        # model is the PUBLIC encoder reference (not the checkpoint id); the supplied
        # model_version ("test_model_v1") is recorded as checkpoint_id provenance.
        assert attrs["geoemb:model"] == f"https://geotessera.org/model/{ENCODER_VERSION}"
        assert attrs["checkpoint_id"] == "test_model_v1"
        # build_version is the software/package version, NOT the encoder/model version.
        assert attrs["geoemb:build_version"] == _dist_version("tessera_embeddings")
        assert attrs["geoemb:data_type"] == "int8"
        assert attrs["geoemb:quantization"]["method"] == "per_pixel_scale"
        assert attrs["geoemb:quantization"]["scale"]["array_name"] == "scales"


class TestScanExistingStagedChunks:
    """Tests for ZarrWriter.scan_existing_staged_chunks (resume from partial runs)."""

    CHUNKS = (
        ChunkSpec(row=0, col=0, y_start=0, y_stop=10, x_start=0, x_stop=10),
        ChunkSpec(row=0, col=1, y_start=0, y_stop=10, x_start=10, x_stop=20),
        ChunkSpec(row=1, col=0, y_start=10, y_stop=20, x_start=0, x_stop=10),
    )

    @staticmethod
    def _stage_chunk(writer: ZarrWriter, chunk: ChunkSpec, run_id: str) -> None:
        emb, scales = _quantized_embeddings(np.random.default_rng(42), chunk.height, chunk.width)
        writer.write_chunk(chunk, emb, run_id, scales=scales)

    def test_no_staging_dir(self, tmp_path):
        """Returns empty set when staging directory doesn't exist."""
        writer = ZarrWriter(str(tmp_path / "nonexistent"))
        result = writer.scan_existing_staged_chunks("run1", self.CHUNKS)
        assert result == set()

    def test_empty_staging_dir(self, tmp_path):
        """Returns empty set when staging directory exists but is empty."""
        staging = tmp_path / "staging" / "run1"
        staging.mkdir(parents=True)
        writer = ZarrWriter(str(tmp_path / "staging"))
        result = writer.scan_existing_staged_chunks("run1", self.CHUNKS)
        assert result == set()

    def test_all_valid(self, tmp_path):
        """All staged chunks valid — returns their labels."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        for chunk in self.CHUNKS:
            self._stage_chunk(writer, chunk, "run1")

        result = writer.scan_existing_staged_chunks("run1", self.CHUNKS)
        assert result == {c.label for c in self.CHUNKS}

    def test_partial_staged(self, tmp_path):
        """Only first chunk staged — returns only that label, not all."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        self._stage_chunk(writer, self.CHUNKS[0], "run1")

        result = writer.scan_existing_staged_chunks("run1", self.CHUNKS)
        assert result == {self.CHUNKS[0].label}

    def test_missing_scales_rejected(self, tmp_path):
        """A crash mid-write_chunk can leave an embeddings-only tile (staged Zarr
        writes are not atomic across arrays). It must be rejected, not counted valid
        and skipped — otherwise run_inference skips it and the fill fails
        permanently at assembly (missing scales) on every retry.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = self.CHUNKS[0]
        self._stage_chunk(writer, chunk, "run1")
        # Simulate the partial write: keep embeddings, drop scales.
        group = zarr.open_group(writer._staging_path("run1", chunk), mode="a")
        del group["scales"]
        with pytest.raises(RuntimeError, match="missing variable 'scales'"):
            writer.scan_existing_staged_chunks("run1", self.CHUNKS)

    def test_missing_done_marker_triggers_reinference(self, tmp_path):
        """A tile whose ``.done`` never landed is not resumed — no tile is opened.

        This is the cheap half of the mechanism: the crash is recognised from the
        staging listing alone, so a run with thousands of tiles does not pay a
        metadata read per tile to find the interrupted ones.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = self.CHUNKS[0]
        self._stage_chunk(writer, chunk, "run1")
        Path(writer._done_marker_path("run1", chunk)).unlink()  # crash before the marker

        # The .zarr is still there and still looks complete inside — but the write
        # never finished, so resume must re-run it rather than trust it.
        assert writer.scan_existing_staged_chunks("run1", self.CHUNKS) == set()

    def test_orphan_done_marker_triggers_reinference(self, tmp_path):
        """A ``.done`` whose tile is gone is a half-landed pair too, with the same fix.

        Reached by an interrupted cleanup or manual surgery. Trusting the marker
        would skip the chunk and assemble nothing for its footprint.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = self.CHUNKS[0]
        self._stage_chunk(writer, chunk, "run1")
        _assembly_mod._fs_for(writer._staging_path("run1", chunk)).rm(
            writer._staging_path("run1", chunk), recursive=True
        )

        assert writer.scan_existing_staged_chunks("run1", self.CHUNKS) == set()

    def test_missing_completion_attribute_triggers_reinference(self, tmp_path):
        """The in-store half of the check, exercised on its own.

        The tile keeps its ``.done`` marker, so the listing hands it over as
        complete and only the attribute can reject it — which is the state a
        reader reaching a tile without consulting the listing would be in. It is
        EXCLUDED from the valid set, so run_inference regenerates it (write_chunk's
        ``mode="w"`` overwrites the partial), rather than raising: with the stable
        input-fingerprinted run_id a raise would re-fire on the same artifact every
        retry and wedge the cell until manual deletion.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = self.CHUNKS[0]
        self._stage_chunk(writer, chunk, "run1")
        # Strip only the in-store attribute — the arrays and the .done marker
        # still look complete from outside.
        group = zarr.open_group(writer._staging_path("run1", chunk), mode="a")
        del group.attrs["staged_complete"]
        assert writer._list_staged_labels("run1") == [chunk.label]
        # Does NOT raise; the tile is simply not counted valid.
        result = writer.scan_existing_staged_chunks("run1", self.CHUNKS)
        assert chunk.label not in result

    def test_invalid_shape_raises(self, tmp_path):
        """A COMPLETE staged chunk (marker present) with wrong shape raises."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = self.CHUNKS[0]
        # Write valid chunk first
        self._stage_chunk(writer, chunk, "run1")
        # Overwrite with wrong shape, then re-stamp the completion marker so it is a
        # COMPLETE-but-structurally-wrong tile (a markerless one would just re-infer).
        bad = np.zeros((5, 5, EMBEDDING_DIM), dtype=np.float32)
        path = writer._staging_path("run1", chunk)
        ds = xr.Dataset(
            {"embeddings": (["northing", "easting", "band"], bad)},
            coords={"northing": np.arange(5), "easting": np.arange(5), "band": np.arange(EMBEDDING_DIM)},
        )
        ds.to_zarr(path, mode="w")
        _mark_staged_complete(path)

        with pytest.raises(RuntimeError, match="invalid staged chunk"):
            writer.scan_existing_staged_chunks("run1", self.CHUNKS)

    def test_missing_variable_raises(self, tmp_path):
        """Staged chunk missing 'embedding' variable raises RuntimeError."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = self.CHUNKS[0]
        path = writer._staging_path("run1", chunk)
        # Write a Zarr with wrong variable name
        ds = xr.Dataset(
            {"wrong_name": (["northing", "easting", "band"], np.zeros((10, 10, EMBEDDING_DIM), dtype=np.float32))},
            coords={"northing": np.arange(10), "easting": np.arange(10), "band": np.arange(EMBEDDING_DIM)},
        )
        ds.to_zarr(path, mode="w")
        _mark_staged_complete(path)  # a COMPLETE-but-invalid tile

        with pytest.raises(RuntimeError, match="missing variable 'embeddings'"):
            writer.scan_existing_staged_chunks("run1", self.CHUNKS)

    def test_stale_chunk_label_raises(self, tmp_path):
        """Staged chunk whose label doesn't match any ChunkSpec raises RuntimeError."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        # Write a chunk with a label not in our chunks list
        stale = ChunkSpec(row=9, col=9, y_start=0, y_stop=10, x_start=0, x_stop=10)
        self._stage_chunk(writer, stale, "run1")

        with pytest.raises(RuntimeError, match="no matching ChunkSpec"):
            writer.scan_existing_staged_chunks("run1", self.CHUNKS)

    def test_compute_std_validation(self, tmp_path):
        """With compute_std=True, missing embedding_std raises RuntimeError."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        # Write chunk WITHOUT std
        self._stage_chunk(writer, self.CHUNKS[0], "run1")

        with pytest.raises(RuntimeError, match="missing variable 'embedding_std'"):
            writer.scan_existing_staged_chunks("run1", self.CHUNKS, compute_std=True)

    def test_reports_all_invalid(self, tmp_path):
        """Error message includes ALL invalid chunks, not just the first."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        # Write two chunks with wrong shapes
        for chunk in self.CHUNKS[:2]:
            bad = np.zeros((5, 5, EMBEDDING_DIM), dtype=np.float32)
            path = writer._staging_path("run1", chunk)
            ds = xr.Dataset(
                {"embeddings": (["northing", "easting", "band"], bad)},
                coords={"northing": np.arange(5), "easting": np.arange(5), "band": np.arange(EMBEDDING_DIM)},
            )
            ds.to_zarr(path, mode="w")
            _mark_staged_complete(path)  # COMPLETE but wrong shape

        with pytest.raises(RuntimeError, match="2 invalid staged chunk") as exc_info:
            writer.scan_existing_staged_chunks("run1", self.CHUNKS)
        # Both chunk labels mentioned in the error
        assert self.CHUNKS[0].label in str(exc_info.value)
        assert self.CHUNKS[1].label in str(exc_info.value)


class TestVerifyStagedCompleteness:
    """Tests for ZarrWriter.verify_staged_completeness (assembly-only guard)."""

    CHUNKS = (
        ChunkSpec(row=0, col=0, y_start=0, y_stop=10, x_start=0, x_stop=10),
        ChunkSpec(row=0, col=1, y_start=0, y_stop=10, x_start=10, x_stop=20),
        ChunkSpec(row=1, col=0, y_start=10, y_stop=20, x_start=0, x_stop=10),
    )

    @staticmethod
    def _stage_chunk(writer: ZarrWriter, chunk: ChunkSpec, run_id: str) -> None:
        emb, scales = _quantized_embeddings(np.random.default_rng(42), chunk.height, chunk.width)
        writer.write_chunk(chunk, emb, run_id, scales=scales)

    def test_all_present_passes(self, tmp_path):
        """No error when all expected chunks are staged."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        for chunk in self.CHUNKS:
            self._stage_chunk(writer, chunk, "run1")

        writer.verify_staged_completeness("run1", list(self.CHUNKS))

    def test_missing_chunks_raises(self, tmp_path):
        """Raises IncompleteStageError listing missing chunks."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        # Only stage the first chunk
        self._stage_chunk(writer, self.CHUNKS[0], "run1")

        with pytest.raises(IncompleteStageError, match="2 missing") as exc_info:
            writer.verify_staged_completeness("run1", list(self.CHUNKS))
        assert "Expected 3 chunks, found 1 staged + 0 skipped" in str(exc_info.value)

    def test_interrupted_chunk_is_reported_as_its_own_category(self, tmp_path):
        """A half-written tile fails verification, named for what it is.

        Reaching assembly is what makes it anomalous — the resume scan re-infers
        interrupted tiles — so it means either inference never ran (an assembly-only
        run) or it crashed the same way twice. Calling it "missing" would send the
        operator looking for a chunk that was never attempted; the remedy differs.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        for chunk in self.CHUNKS:
            self._stage_chunk(writer, chunk, "run1")
        Path(writer._done_marker_path("run1", self.CHUNKS[1])).unlink()

        with pytest.raises(IncompleteStageError, match="1 interrupted") as exc_info:
            writer.verify_staged_completeness("run1", list(self.CHUNKS))
        assert "0 missing" not in str(exc_info.value)
        assert self.CHUNKS[1].label in str(exc_info.value)

    def test_skip_marker_counts_as_resolved(self, tmp_path):
        """A chunk with a skip marker instead of a staged zarr passes verification."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        self._stage_chunk(writer, self.CHUNKS[0], "run1")
        self._stage_chunk(writer, self.CHUNKS[1], "run1")
        writer.write_skip_marker(self.CHUNKS[2], "run1")

        writer.verify_staged_completeness("run1", list(self.CHUNKS))

    def test_a_chunk_that_stops_skipping_clears_its_stale_skip_marker(self, tmp_path):
        """The mirror of the case below: whichever outcome a chunk reaches, no trace
        of the other survives.

        A chunk with no valid pixels on one attempt can produce some on the next, after
        a re-ingested mosaic. Left alone, its old skip marker sits beside the new tile
        and verification reads the pair as inconsistent — and under the stable,
        input-fingerprinted run_id that refusal repeats on every retry, so the cell is
        wedged until someone deletes the marker by hand.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = self.CHUNKS[0]
        writer.write_skip_marker(chunk, "run1")
        assert Path(writer._skip_marker_path("run1", chunk)).exists()

        self._stage_chunk(writer, chunk, "run1")  # this attempt has pixels

        assert not Path(writer._skip_marker_path("run1", chunk)).exists()
        for other in self.CHUNKS[1:]:
            self._stage_chunk(writer, other, "run1")
        writer.verify_staged_completeness("run1", list(self.CHUNKS))

    def test_skip_marker_clears_a_stale_done_marker(self, tmp_path):
        """A chunk that staged, then skipped on a retry, must not look like both.

        ``write_skip_marker`` drops the sibling zarr; it has to drop the completion
        marker too. Leaving it behind would make the chunk both complete and skipped
        on the next verification, which raises — and under the stable run_id it
        raises on every retry, wedging the cell until someone deletes it by hand.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        self._stage_chunk(writer, self.CHUNKS[0], "run1")
        self._stage_chunk(writer, self.CHUNKS[1], "run1")
        self._stage_chunk(writer, self.CHUNKS[2], "run1")

        writer.write_skip_marker(self.CHUNKS[2], "run1")

        assert not Path(writer._done_marker_path("run1", self.CHUNKS[2])).exists()
        writer.verify_staged_completeness("run1", list(self.CHUNKS))

    def test_both_staged_and_skipped_raises(self, tmp_path):
        """A chunk with BOTH a staged zarr and a skip marker is inconsistent.

        The marker is created directly, not via ``write_skip_marker`` — that clears
        the sibling zarr, so our own writer can no longer produce this state. The
        guard remains for prefixes we did not author: a half-cleaned directory, two
        processes on one run_id, manual surgery.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        for chunk in self.CHUNKS:
            self._stage_chunk(writer, chunk, "run1")
        (tmp_path / "staging" / "run1" / f"{self.CHUNKS[0].label}.skipped").write_bytes(b"")

        with pytest.raises(IncompleteStageError, match="BOTH a staged zarr and a skip marker"):
            writer.verify_staged_completeness("run1", list(self.CHUNKS))

    def test_detect_chunk_size_separates_all_skipped_from_no_such_run(self, tmp_path):
        """An all-skipped run must stay resumable; a bogus run_id must not.

        assemble() publishes an all-fill timestep when every chunk skipped, but the
        resume path sizes the chunk grid first — so that lookup has to distinguish
        "real run, nothing to measure" from "no such run", or either the branch is
        unreachable or a mistyped run_id silently becomes a full re-run.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        for chunk in self.CHUNKS:
            writer.write_skip_marker(chunk, "all-skipped")

        with pytest.raises(AllChunksSkippedError, match="all-skipped"):
            writer.detect_staged_chunk_size("all-skipped")
        with pytest.raises(FileNotFoundError):
            writer.detect_staged_chunk_size("typo-run")

    def test_skip_marker_clears_a_stale_staged_zarr(self, tmp_path):
        """A skipping chunk must not leave a staged zarr behind.

        The resume scan excludes an incomplete staged zarr rather than raising (a
        raise would re-fire every retry under the stable run_id). If the chunk then
        skips, the leftover would trip the BOTH check here instead — wedging the
        cell in the very way that exclusion exists to avoid.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        for chunk in self.CHUNKS:
            self._stage_chunk(writer, chunk, "run1")
        assert (tmp_path / "staging" / "run1" / f"{self.CHUNKS[0].label}.zarr").exists()

        writer.write_skip_marker(self.CHUNKS[0], "run1")

        assert not (tmp_path / "staging" / "run1" / f"{self.CHUNKS[0].label}.zarr").exists()
        writer.verify_staged_completeness("run1", list(self.CHUNKS))

    def test_no_staged_chunks_raises(self, tmp_path):
        """Raises IncompleteStageError when staging dir is empty."""
        writer = ZarrWriter(str(tmp_path / "staging"))

        with pytest.raises(IncompleteStageError, match="3 missing"):
            writer.verify_staged_completeness("run1", list(self.CHUNKS))

    def test_extra_chunks_raises(self, tmp_path):
        """Raises IncompleteStageError when staging has chunks not in the grid."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        for chunk in self.CHUNKS:
            self._stage_chunk(writer, chunk, "run1")
        # Stage an extra chunk not in the expected list
        extra = ChunkSpec(row=9, col=9, y_start=0, y_stop=10, x_start=0, x_stop=10)
        self._stage_chunk(writer, extra, "run1")

        with pytest.raises(IncompleteStageError, match="1 unexpected"):
            writer.verify_staged_completeness("run1", list(self.CHUNKS))


class TestAssembleGlobal:
    """assemble_global: staged tiles -> whole shards of a pre-seeded zone group."""

    TILE = 64  # miniature shard pitch so a unit test never touches 2048² arrays
    INNER = 32
    ZONE = "01N"
    YEARS = (2024, 2025)

    def _seed_zone_repo(self, tmp_path, ny: int, nx: int, dim: int, *, carried: tuple[str, ...] = ()):
        """A miniature GLOBAL-shaped zone group: sharded arrays + calendar-year time axis.

        ``carried`` seeds the named :data:`CARRIED_VARS` arrays so tests can exercise the
        optional-variable dtype / heterogeneity guards and the staged-to-destination copy.
        Their geometry and dtype are read off the GLOBAL layout rather than written out here,
        so a variable added to the schema is seeded correctly by this helper without an edit
        — the omission that let ``s2_month_covered`` publish as all-fill was invisible partly
        because every test destination was hand-rolled and so never held the new array.
        """
        store_path = str(tmp_path / "global.icechunk")
        repo = create_global_repo(store_path)
        session = repo.writable_session("main")
        root = zarr.open_group(session.store, mode="a")
        node = root.require_group(self.ZONE)
        times = np.array([f"{y}-01-01" for y in self.YEARS], dtype="datetime64[ns]")
        node.create_array(
            "embeddings",
            shape=(len(times), ny, nx, dim),
            chunks=(1, self.INNER, self.INNER, dim),
            shards=(1, self.TILE, self.TILE, dim),
            dtype="int8",
            fill_value=0,
            dimension_names=("time", "northing", "easting", "band"),
        )
        node.create_array(
            "scales",
            shape=(len(times), ny, nx),
            chunks=(1, self.INNER, self.INNER),
            shards=(1, self.TILE, self.TILE),
            dtype="float32",
            fill_value=float("nan"),
            dimension_names=("time", "northing", "easting"),
        )
        sizes = {"time": len(times), "northing": ny, "easting": nx, "band": dim, "month": MONTHS_IN_YEAR}
        for var in carried:
            layout = GLOBAL.for_var(var)
            # Test-sized chunk/shard pitch, but the layout's own dims, dtype and fill —
            # so the seeded array is exactly the shape the write path will meet.
            trailing = tuple(sizes[d] for d in layout.dims[3:])
            array = node.create_array(
                var,
                shape=tuple(sizes[d] for d in layout.dims),
                chunks=(1, self.INNER, self.INNER, *trailing),
                shards=(1, self.TILE, self.TILE, *trailing),
                dtype=layout.dtype,
                fill_value=layout.fill_value,
                dimension_names=layout.dims,
            )
            # Including the layout's attrs, as the real seeder does — `dtype="bool"` on an int8
            # array is part of its type, not decoration, and a fixture that drops it is a fixture
            # that disagrees with production about what was seeded.
            if layout.attrs:
                array.attrs.update(dict(layout.attrs))
        time_int = times.astype("int64")
        time_arr = node.create_array("time", data=time_int, chunks=(len(time_int),), dimension_names=("time",))
        time_arr.attrs.update(TIME_ENCODING)
        node.attrs["years_complete"] = []
        session.commit("seed test zone")
        return store_path

    def _stage_raw_tile(self, tmp_path, run_id: str, label: str, dim: int, *, extra: dict | None = None):
        """Hand-roll a staged tile (embeddings int8 + scales float32, plus *extra*)."""
        path = str(tmp_path / "staging" / run_id / f"{label}.zarr")
        g = zarr.open_group(path, mode="w")
        g.create_array("embeddings", data=np.ones((self.TILE, self.TILE, dim), dtype="int8"))
        g.create_array("scales", data=np.ones((self.TILE, self.TILE), dtype="float32"))
        for name, arr in (extra or {}).items():
            g.create_array(name, data=arr)
        _mark_staged_complete(path)  # assembly requires both completion signals

    @pytest.mark.parametrize("credentialled", [True, False])
    def test_the_forking_write_asks_icechunk_to_cache_the_credential(self, tmp_path, monkeypatch, credentialled):
        """`write_year_shards` PICKLES this session to spawned children, and an icechunk
        credential fetcher's `initial` cache can never be refilled (icechunk#2077), so a child
        that deserialises without one calls back on every S3 request for the life of the fork.
        Asserted at the call site because that is where the decision lives — the flag puts a
        live secret in the pickle, so it is right for a forking writer and pointless elsewhere.
        True regardless of whether a callback is passed (hence the parameterisation):
        `_create_storage` substitutes a default provider when it is None, so a
        `get_credentials is not None` guard would disable the scatter on exactly that path.
        """
        dim = 8
        store_path = self._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        writer.write_chunk(
            ChunkSpec(row=0, col=0, y_start=0, y_stop=self.TILE, x_start=0, x_stop=self.TILE),
            np.ones((self.TILE, self.TILE, dim), dtype=np.int8),
            "runS",
            scales=np.ones((self.TILE, self.TILE), dtype=np.float32),
        )

        requested: list[bool] = []
        real_open = _assembly_mod.open_global_repo

        def spy(path, **kwargs):
            requested.append(kwargs["scatter_initial_credentials"])
            return real_open(path, **kwargs)

        monkeypatch.setattr(_assembly_mod, "open_global_repo", spy)

        def credentials() -> icechunk.S3StaticCredentials:
            return icechunk.S3StaticCredentials(access_key_id="k", secret_access_key="s")

        writer.assemble_global(
            store_path,
            self.ZONE,
            year=2025,
            run_id="runS",
            n_workers=1,
            get_credentials=credentials if credentialled else None,
        )
        assert requested == [True]

    def test_fills_year_from_staged_tiles(self, tmp_path):
        """Staged tiles land as whole shards at the right year index; provenance attrs update."""
        dim = 8
        ny = nx = 2 * self.TILE  # 2x2 tile/shard grid
        store_path = self._seed_zone_repo(tmp_path, ny, nx, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)

        rng = np.random.default_rng(3)
        expected = np.zeros((ny, nx, dim), dtype=np.int8)
        expected_scales = np.full((ny, nx), np.nan, dtype=np.float32)
        # Stage 3 of 4 tiles — (1, 0) stays ocean/fill.
        for row, col in ((0, 0), (0, 1), (1, 1)):
            chunk = ChunkSpec(
                row=row,
                col=col,
                y_start=row * self.TILE,
                y_stop=(row + 1) * self.TILE,
                x_start=col * self.TILE,
                x_stop=(col + 1) * self.TILE,
            )
            emb = rng.integers(-100, 100, size=(self.TILE, self.TILE, dim)).astype(np.int8)
            scales = rng.random((self.TILE, self.TILE)).astype(np.float32)
            writer.write_chunk(chunk, emb, "runG", scales=scales)
            expected[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop] = emb
            expected_scales[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop] = scales

        snapshot = writer.assemble_global(store_path, self.ZONE, year=2025, run_id="runG", n_workers=1)
        assert snapshot

        repo = open_global_repo(store_path)
        node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[self.ZONE]
        # 2025 is index 1; 2024 (index 0) must remain untouched fill.
        np.testing.assert_array_equal(np.asarray(node["embeddings"][1]), expected)
        np.testing.assert_array_equal(np.asarray(node["embeddings"][0]), np.zeros_like(expected))
        np.testing.assert_array_equal(np.asarray(node["scales"][1]), expected_scales)
        assert node.attrs["years_complete"] == [2025]
        assert node.attrs["runs"]["2025"]["run_id"] == "runG"

    def _stage_tile(self, writer, row, col, dim, run_id, rng):
        chunk = ChunkSpec(
            row=row,
            col=col,
            y_start=row * self.TILE,
            y_stop=(row + 1) * self.TILE,
            x_start=col * self.TILE,
            x_stop=(col + 1) * self.TILE,
        )
        emb = rng.integers(-100, 100, size=(self.TILE, self.TILE, dim)).astype(np.int8)
        writer.write_chunk(chunk, emb, run_id, scales=rng.random((self.TILE, self.TILE)).astype(np.float32))
        return emb

    def test_a_retry_that_skips_a_tile_clears_the_previous_attempt(self, tmp_path):
        """The mixed-year hazard: a year is written in TWO commits, and a crash between
        them leaves shards on a year the campaign then re-dispatches. If the retry's
        mosaic makes a tile skip where the first attempt produced pixels, leaving that
        tile alone publishes the OLD attempt's data under the new attempt's completion
        mark — two inputs in one write-once year, with nothing to show for it.
        """
        dim = 8
        ny = nx = 2 * self.TILE
        store_path = self._seed_zone_repo(tmp_path, ny, nx, dim)
        rng = np.random.default_rng(11)

        # Attempt 1 stages both tiles of the top row and lands its shards.
        first = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        for col in (0, 1):
            self._stage_tile(first, 0, col, dim, "runA", rng)
        first.assemble_global(store_path, self.ZONE, year=2025, run_id="runA", n_workers=1)

        # Attempt 2: (0, 1) now has no valid pixels, so it stages nothing and is
        # reported as skipped — exactly what zone_fill passes for a live tile that
        # resolved to a skip marker.
        second = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        kept = self._stage_tile(second, 0, 0, dim, "runB", rng)
        second.assemble_global(
            store_path,
            self.ZONE,
            year=2025,
            run_id="runB",
            n_workers=1,
            staged_labels=["chunk_0_0"],
            skipped_labels=["chunk_0_1"],
        )

        node = zarr.open_group(open_global_repo(store_path).readonly_session(branch="main").store, mode="r")[self.ZONE]
        result = np.asarray(node["embeddings"][1])
        np.testing.assert_array_equal(result[: self.TILE, : self.TILE], kept)
        assert (result[: self.TILE, self.TILE :] == 0).all(), "attempt 1's tile survived attempt 2"
        # And the float companion is cleared to ITS fill, not to zero.
        assert np.isnan(np.asarray(node["scales"][1])[: self.TILE, self.TILE :]).all()

    def test_a_skipped_tile_is_indistinguishable_from_one_never_written(self, tmp_path):
        """Clearing must reproduce fill exactly, or a consumer can tell them apart."""
        dim = 8
        ny = nx = 2 * self.TILE
        store_path = self._seed_zone_repo(tmp_path, ny, nx, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        self._stage_tile(writer, 0, 0, dim, "runC", np.random.default_rng(12))

        writer.assemble_global(
            store_path,
            self.ZONE,
            year=2025,
            run_id="runC",
            n_workers=1,
            staged_labels=["chunk_0_0"],
            skipped_labels=["chunk_0_1"],  # cleared
        )

        node = zarr.open_group(open_global_repo(store_path).readonly_session(branch="main").store, mode="r")[self.ZONE]
        emb, scales = np.asarray(node["embeddings"][1]), np.asarray(node["scales"][1])
        cleared = (slice(0, self.TILE), slice(self.TILE, None))
        never_written = (slice(self.TILE, None), slice(0, self.TILE))  # (1, 0): outside both sets
        np.testing.assert_array_equal(emb[cleared], emb[never_written])
        assert np.isnan(scales[cleared]).all() and np.isnan(scales[never_written]).all()

    def test_emits_one_assembly_summary_record(self, tmp_path, caplog):
        """The record states what actually ran, not what was asked for.

        Requested workers, effective forks, and the per-fork S3 cap can all
        differ (the budget split and the work partition each cap the count), and
        the totals are sums over workers — the figures that decide whether a
        slow assembly is blocked on staged reads, on compression, or on the
        object store.
        """
        dim = 8
        ny = nx = 2 * self.TILE
        store_path = self._seed_zone_repo(tmp_path, ny, nx, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        rng = np.random.default_rng(21)
        for col in (0, 1):
            self._stage_tile(writer, 0, col, dim, "runT", rng)
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.inference.assembly"):
            writer.assemble_global(
                store_path,
                self.ZONE,
                year=2025,
                run_id="runT",
                n_workers=4,
                s3_concurrency=12,
                staged_labels=["chunk_0_0", "chunk_0_1"],
                skipped_labels=["chunk_1_0"],
            )
        (rec,) = _assembly_summary_records(caplog)
        assert (rec["zone"], rec["year"], rec["run"]) == (self.ZONE, 2025, "runT")
        assert rec["tiles_staged"] == 2 and rec["tiles_cleared"] == 1
        # 4 workers requested, but only 3 live shards exist (2 staged + 1
        # cleared), so only 3 forks actually ran.
        assert rec["workers_requested"] == 4
        assert rec["workers_used"] == 3 == len(rec["workers"])
        assert rec["per_worker_s3_cap"] == 3  # budget 12 split across the 4 requested workers
        # Totals are SUMS across workers: 3 tile loads (cleared included — its
        # fill block is written like any other), 2 vars each.
        assert rec["tiles"] == 3
        assert rec["writes"] == 6
        tile_bytes = self.TILE * self.TILE * dim + self.TILE * self.TILE * 4
        assert rec["bytes"] == 3 * tile_bytes
        assert rec["fused_compress_put"] is True
        for key in ("read_s", "write_s", "fill_wall_s", "merge_s", "commit_s", "attrs_commit_s", "total_s"):
            assert rec[key] >= 0

    def test_a_budget_below_the_worker_count_no_longer_drops_forks(self, tmp_path, caplog):
        """The campaign's real shape, through the real call path: a per-fill S3 budget
        well BELOW the requested worker count.

        This is the end-to-end companion to the ``_s3_budget_split`` unit tests. The
        clamp used to run ``min(n_workers, budget)`` forks — 2 of 4 here, and 5 of 16
        in production — and the only place it showed was this record's ``workers_used``.
        The budget now sets the per-fork request cap alone.
        """
        dim = 8
        ny = nx = 2 * self.TILE
        store_path = self._seed_zone_repo(tmp_path, ny, nx, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        rng = np.random.default_rng(22)
        labels = []
        for row in (0, 1):
            for col in (0, 1):
                self._stage_tile(writer, row, col, dim, "runB", rng)
                labels.append(f"chunk_{row}_{col}")
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.inference.assembly"):
            writer.assemble_global(
                store_path,
                self.ZONE,
                year=2025,
                run_id="runB",
                n_workers=4,
                s3_concurrency=2,
                staged_labels=labels,
            )
        (rec,) = _assembly_summary_records(caplog)
        assert rec["workers_requested"] == 4
        assert rec["workers_used"] == 4 == len(rec["workers"]), "the S3 budget must not drop forks"
        # Floors at 1, so this fill's aggregate concurrency is its worker count. That
        # is the ceiling this change gives up, deliberately — see _s3_budget_split.
        assert rec["per_worker_s3_cap"] == 1

    def _year_record(self, store_path):
        """The landed year's provenance entry (2025 is index 1 on the seeded axis)."""
        node = zarr.open_group(open_global_repo(store_path).readonly_session(branch="main").store, mode="r")[self.ZONE]
        return dict(node.attrs["runs"])["2025"]

    def test_the_skipped_tiles_are_recorded_on_the_year_s_provenance(self, tmp_path):
        """Skipped tiles publish as fill, and ocean is fill too, so a consumer of the
        completed year cannot tell "no valid optical data" from "not land" unless the
        year says which tiles they were. The labels are what make the area maskable;
        the live total is what makes the count interpretable without the land mask.
        """
        dim = 8
        ny = nx = 2 * self.TILE
        store_path = self._seed_zone_repo(tmp_path, ny, nx, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        self._stage_tile(writer, 0, 0, dim, "runOS", np.random.default_rng(31))

        writer.assemble_global(
            store_path,
            self.ZONE,
            year=2025,
            run_id="runOS",
            n_workers=1,
            staged_labels=["chunk_0_0"],
            skipped_labels=["chunk_1_0", "chunk_0_1"],
        )

        skips = self._year_record(store_path)["optical_skips"]
        assert skips["tiles_skipped"] == 2
        assert skips["labels"] == ["chunk_0_1", "chunk_1_0"]
        assert skips["tiles_live"] == 3

    def test_a_resolved_live_set_with_no_skips_records_a_zero(self, tmp_path):
        """An empty skipped set is a resolved fact — every live tile staged data — while
        a caller that resolved no live set at all (``None``) records nothing. Keeping the
        two distinct is what stops an unresolved fill from reading as a measured zero.
        """
        dim = 8
        store_path = self._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        self._stage_tile(writer, 0, 0, dim, "runZ", np.random.default_rng(32))

        writer.assemble_global(
            store_path, self.ZONE, year=2025, run_id="runZ", n_workers=1, staged_labels=["chunk_0_0"], skipped_labels=[]
        )
        skips = self._year_record(store_path)["optical_skips"]
        # The registry's fields come along, empty: no shard was refused, so there is no reason to
        # record. Asserted whole rather than by key, because a MISSING by_reason and an all-zero one
        # say different things — the first is a record that cannot answer, the second is an answer.
        assert skips == {
            "tiles_skipped": 0,
            "tiles_live": 1,
            "labels": [],
            "refused_px_by_reason": {"no_optical": 0, "thin": 0, "no_radar": 0},
            "shards_by_reason": {},
            "shards_mixed": 0,
            "s2_obs_at_refused": {"max": 0, "px_with_any": 0},
            "unrecorded": [],
        }

    def test_an_unresolved_live_set_records_no_summary(self, tmp_path):
        dim = 8
        store_path = self._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        self._stage_tile(writer, 0, 0, dim, "runN", np.random.default_rng(33))

        writer.assemble_global(store_path, self.ZONE, year=2025, run_id="runN", n_workers=1)
        assert "optical_skips" not in self._year_record(store_path)

    def test_re_assembling_the_same_year_leaves_one_record(self, tmp_path):
        """The campaign re-dispatches cells, so a year is assembled more than once. The
        record must be replaced, not appended to or accumulated into.
        """
        dim = 8
        ny = nx = 2 * self.TILE
        store_path = self._seed_zone_repo(tmp_path, ny, nx, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        self._stage_tile(writer, 0, 0, dim, "runI", np.random.default_rng(34))

        # `skipped_labels` names a shard the caller declares skipped, but this test writes no marker
        # for it — so the registry has no record to read and says so under `unrecorded` rather than
        # reporting an all-zero refusal. That distinction is the point: "no reason recorded" is not
        # "nothing was refused", and folding the two would be the error the registry exists to avoid.
        expected = {
            "tiles_skipped": 1,
            "tiles_live": 2,
            "labels": ["chunk_0_1"],
            "refused_px_by_reason": {"no_optical": 0, "thin": 0, "no_radar": 0},
            "shards_by_reason": {},
            "shards_mixed": 0,
            "s2_obs_at_refused": {"max": 0, "px_with_any": 0},
            "unrecorded": ["chunk_0_1"],
        }
        for _ in range(2):
            writer.assemble_global(
                store_path,
                self.ZONE,
                year=2025,
                run_id="runI",
                n_workers=1,
                staged_labels=["chunk_0_0"],
                skipped_labels=["chunk_0_1"],
            )
            assert self._year_record(store_path)["optical_skips"] == expected

        node = zarr.open_group(open_global_repo(store_path).readonly_session(branch="main").store, mode="r")[self.ZONE]
        assert list(dict(node.attrs["runs"])) == ["2025"]
        assert node.attrs["years_complete"] == [2025]

    def test_an_all_skipped_year_records_the_empty_flag_instead(self, tmp_path):
        """The wholly-empty case is already stated by ``empty``, which is also why the
        label list needs no cap — the one situation where it could span a whole zone is
        the situation the flag covers.
        """
        dim = 8
        ny = nx = 2 * self.TILE
        store_path = self._seed_zone_repo(tmp_path, ny, nx, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)

        writer.assemble_global(
            store_path,
            self.ZONE,
            year=2025,
            run_id="runAS",
            n_workers=1,
            staged_labels=(),
            skipped_labels=["chunk_0_0", "chunk_0_1"],
            empty=True,
        )
        record = self._year_record(store_path)
        assert record["empty"] is True
        assert "optical_skips" not in record

    def test_year_off_axis_raises(self, tmp_path):
        """A year outside the pre-allocated axis is a loud error (D1: never resize)."""
        dim = 8
        store_path = self._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=self.TILE, x_start=0, x_stop=self.TILE)
        emb = np.ones((self.TILE, self.TILE, dim), dtype=np.int8)
        writer.write_chunk(chunk, emb, "runG", scales=np.ones((self.TILE, self.TILE), dtype=np.float32))

        with pytest.raises(ValueError, match="not on 01N's pre-allocated time axis"):
            writer.assemble_global(store_path, self.ZONE, year=1999, run_id="runG", n_workers=1)

    def test_tile_shard_mismatch_raises(self, tmp_path):
        """Staged tiles that aren't one-shard-sized violate the 1:1 contract (D3)."""
        dim = 8
        store_path = self._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        half = self.TILE // 2
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=half, x_start=0, x_stop=half)
        emb = np.ones((half, half, dim), dtype=np.int8)
        writer.write_chunk(chunk, emb, "runG", scales=np.ones((half, half), dtype=np.float32))

        with pytest.raises(ValueError, match="1 inference tile == 1 shard"):
            writer.assemble_global(store_path, self.ZONE, year=2025, run_id="runG", n_workers=1)

    def test_no_staged_tiles_raises(self, tmp_path):
        dim = 8
        store_path = self._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        with pytest.raises(IncompleteStageError, match="no staged chunks"):
            writer.assemble_global(store_path, self.ZONE, year=2025, run_id="empty_run", n_workers=1)

    def test_staged_tile_missing_required_var_raises(self, tmp_path):
        """A staged tile without embeddings/scales must abort, not silently drop the variable."""
        dim = 8
        store_path = self._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim)
        staging = tmp_path / "staging"
        # Hand-roll a corrupt staged tile: embeddings only, no scales.
        path = str(staging / "runC" / "chunk_0_0.zarr")
        g = zarr.open_group(path, mode="w")
        g.create_array("embeddings", data=np.ones((self.TILE, self.TILE, dim), dtype="int8"))
        _mark_staged_complete(path)  # a COMPLETED write whose CONTENT is wrong
        writer = ZarrWriter(str(staging), embedding_dim=dim)

        with pytest.raises(IncompleteStageError, match="missing required variable"):
            writer.assemble_global(store_path, self.ZONE, year=2025, run_id="runC", n_workers=1)

    @pytest.mark.parametrize("var", MONTH_COVERED_VARS)
    def test_month_coverage_survives_the_real_staging_path(self, tmp_path, var):
        """A BOOL buffer handed to write_chunk must reach an int8 destination with its values.

        Staged through the production path — ``write_chunk``, which writes an xarray Dataset — rather
        than a hand-rolled zarr tile, because that is the difference the defect lived in.
        ``Dataset.to_zarr`` stores a bool array as **int8** with attrs ``dtype="bool"`` (xarray's own
        boolean representation) and ignores an encoding dtype asking for bool; assembly reads staged
        tiles with RAW zarr, so it sees the int8. A bool-seeded destination therefore refused every
        month tile on the dtype guard — the guard was right, the destination's dtype was wrong.

        The round-trip test beside this one could not catch it: it stages with raw zarr, which keeps
        bool. A fixture that writes by a different route than production is not testing the route.
        """
        dim = 8
        store_path = self._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim, carried=(var,))
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        rng = np.random.default_rng(19)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=self.TILE, x_start=0, x_stop=self.TILE)
        # (month, y, x) as the actor builds it, and a mixed pattern so all-True and all-False both fail.
        covered = rng.random((MONTHS_IN_YEAR, self.TILE, self.TILE)) > 0.35
        writer.write_chunk(
            chunk,
            rng.integers(-100, 100, size=(self.TILE, self.TILE, dim)).astype(np.int8),
            "runM",
            scales=rng.random((self.TILE, self.TILE)).astype(np.float32),
            month_covered={var: covered},
        )
        assert writer.assemble_global(store_path, self.ZONE, year=2025, run_id="runM", n_workers=1)

        repo = open_global_repo(store_path)
        node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[self.ZONE]
        got = np.asarray(node[var][self.YEARS.index(2025)])
        # Destination axis order is (northing, easting, month); the actor's buffer is (month, y, x).
        np.testing.assert_array_equal(got.astype(bool), covered.transpose(1, 2, 0))
        # And the attribute that makes an xarray reader see booleans rather than 0/1 integers.
        assert dict(node[var].attrs).get("dtype") == "bool"

    def _carried_pattern(self, var: str, dim: int, rng) -> np.ndarray:
        """A distinctive per-tile array for *var*, shaped and typed from the GLOBAL layout.

        Deliberately never uniform and never the layout's fill: an assertion against a
        constant array passes both when the value is copied and when the destination is left
        as fill, which is exactly how an omitted variable hides.
        """
        layout = GLOBAL.for_var(var)
        sizes = {"northing": self.TILE, "easting": self.TILE, "band": dim, "month": MONTHS_IN_YEAR}
        shape = tuple(sizes[d] for d in layout.dims[1:])
        if layout.dtype == "bool":
            return rng.random(shape) > 0.4  # a mixed pattern, so all-True fails too
        if np.issubdtype(np.dtype(layout.dtype), np.integer):
            return rng.integers(1, 500, size=shape).astype(layout.dtype)
        return rng.random(shape).astype(layout.dtype)

    def test_every_carried_var_reaches_the_destination(self, tmp_path):
        """Each variable the schema carries must arrive in the destination with its values.

        Looped over :data:`CARRIED_VARS` rather than a written-out list, so an array added to
        the store layout is covered by this test without anyone remembering to extend it.
        That is the defect this test exists for: ``s2_month_covered`` was added to the schema,
        seeded in every zone, and staged with real values by the actors, but assembly's copy
        list was a hand-written tuple that did not name it — so a whole zone-year published
        twelve all-``False`` planes, with the obs counts beside them correct and no error
        anywhere. Nothing was wrong with the array; it was simply never copied.
        """
        dim = 8
        carried = tuple(v for v in CARRIED_VARS if v in GLOBAL.arrays)
        # Every sensor's mask, not just the optical one whose omission first caused this.
        assert all(v in carried for v in MONTH_COVERED_VARS)
        store_path = self._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim, carried=carried)
        rng = np.random.default_rng(11)
        expected = {var: self._carried_pattern(var, dim, rng) for var in carried}
        self._stage_raw_tile(tmp_path, "runV", "chunk_0_0", dim, extra=expected)

        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        assert writer.assemble_global(store_path, self.ZONE, year=2025, run_id="runV", n_workers=1)

        repo = open_global_repo(store_path)
        node = zarr.open_group(repo.readonly_session("main").store, mode="r")[self.ZONE]
        year_index = self.YEARS.index(2025)
        for var, want in expected.items():
            got = np.asarray(node[var][year_index])
            np.testing.assert_array_equal(got, want, err_msg=f"{var} did not survive the staged-to-store copy")

    def test_staged_obs_dtype_mismatch_raises(self, tmp_path):
        """An obs count staged at the wrong dtype must abort, not silently narrow.

        A raw-zarr write C-casts, so an int64 obs count would wrap into a seeded
        uint16 without a loud check. StagedShardSource only cross-checks tiles vs
        the probe (which agree here), so the guard is destination-dtype-based.
        """
        dim = 8
        store_path = self._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim, carried=("s2_obs_count",))
        self._stage_raw_tile(
            tmp_path,
            "runD",
            "chunk_0_0",
            dim,
            extra={"s2_obs_count": np.ones((self.TILE, self.TILE), dtype="int64")},
        )
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        with pytest.raises(IncompleteStageError, match="s2_obs_count dtype int64 but 01N/s2_obs_count is seeded"):
            writer.assemble_global(store_path, self.ZONE, year=2025, run_id="runD", n_workers=1)

    def test_heterogeneous_staged_optional_vars_raises(self, tmp_path):
        """Tiles disagreeing on optional vars abort rather than silently dropping.

        StagedShardSource.load checks EVERY tile: whichever tile is the probe, the
        other one trips either the missing-variable check (probe has the obs count)
        or the extras check (probe lacks it) — both name s2_obs_count and abort.
        """
        dim = 8
        ny, nx = 2 * self.TILE, self.TILE  # 2x1 tile grid
        store_path = self._seed_zone_repo(tmp_path, ny, nx, dim, carried=("s2_obs_count",))
        # Tile (0,0) carries the obs count; tile (1,0) does not.
        self._stage_raw_tile(
            tmp_path,
            "runH",
            "chunk_0_0",
            dim,
            extra={"s2_obs_count": np.ones((self.TILE, self.TILE), dtype="uint16")},
        )
        self._stage_raw_tile(tmp_path, "runH", "chunk_1_0", dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        with pytest.raises(ValueError, match="s2_obs_count"):
            writer.assemble_global(store_path, self.ZONE, year=2025, run_id="runH", n_workers=1)


class TestAssembleGuards:
    """Regression guards from the branch code review: extent, missing vars, tile validation."""

    def _stage_one(self, writer, chunk, run_id, with_obs=False):
        rng = np.random.default_rng(3)
        emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
        obs = (
            {var: rng.integers(0, 9, size=(chunk.height, chunk.width)).astype(np.uint16) for var in OBS_COUNT_VARS}
            if with_obs
            else None
        )
        writer.write_chunk(chunk, emb, run_id, scales=scales, obs_counts=obs)
        return emb

    def test_append_extent_mismatch_raises(self, tmp_path):
        """A mosaic grid that doesn't match the existing store is a loud error, not a corner write."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        output = str(tmp_path / "out.zarr")
        day = datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC)
        big = ChunkSpec(row=0, col=0, y_start=0, y_stop=8, x_start=0, x_stop=8)
        self._stage_one(writer, big, "run1")
        writer.assemble(
            [big],
            total_y=8,
            total_x=8,
            run_id="run1",
            output_path=output,
            roi_zarr_path=_make_full_roi_mask(tmp_path, 8, 8),
            run_started_at=day,
            n_workers=1,
        )

        small = ChunkSpec(row=0, col=0, y_start=0, y_stop=4, x_start=0, x_stop=4)
        self._stage_one(writer, small, "run2")
        with pytest.raises(ValueError, match="does not match existing store"):
            writer.assemble(
                [small],
                total_y=4,
                total_x=4,
                run_id="run2",
                output_path=output,
                roi_zarr_path=_make_full_roi_mask(tmp_path / "roi2", 4, 4),
                run_started_at=datetime.datetime(2025, 6, 1, tzinfo=datetime.UTC),
                n_workers=1,
            )

    def test_append_refuses_a_reordered_interior_on_matching_endpoints(self, tmp_path, monkeypatch):
        """Extent, CRS and endpoints do not pin an axis, and this phase writes POSITIONALLY.

        A mosaic whose interior is reordered — or non-affine — while its first and last
        coordinates match passes every check that existed here, and its pixels then land at
        the wrong coordinates under the store's own unchanged axes. Nothing signals it: the
        arrays are valid, the shapes agree, and the georeferencing looks intact.

        The zone-fill runner already compares complete vectors for exactly this reason. This
        is the single-ROI path, which is the one that accepts a hand-provided mosaic.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        output = str(tmp_path / "out.zarr")
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=8, x_start=0, x_stop=8)
        north = np.arange(8, dtype="float64") * 10.0
        east = np.arange(8, dtype="float64") * 10.0

        coords = {"value": SpatialCoords(northing=north, easting=east)}
        monkeypatch.setattr(_assembly_mod, "read_spatial_coords", lambda *a, **k: coords["value"])

        def _run(run_id: str, when: datetime.datetime, roi: str):
            self._stage_one(writer, chunk, run_id)
            writer.assemble(
                [chunk],
                total_y=8,
                total_x=8,
                run_id=run_id,
                output_path=output,
                roi_zarr_path=roi,
                run_started_at=when,
                n_workers=1,
                mosaic_base=str(tmp_path / "mosaic"),
            )

        _run("run1", datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC), _make_full_roi_mask(tmp_path, 8, 8))

        # Same endpoints, same length, same CRS — two interior values transposed.
        scrambled = north.copy()
        scrambled[3], scrambled[4] = scrambled[4], scrambled[3]
        assert scrambled[0] == north[0] and scrambled[-1] == north[-1]
        coords["value"] = SpatialCoords(northing=scrambled, easting=east)

        with pytest.raises(ValueError, match="reordered or non-affine interior"):
            _run(
                "run2",
                datetime.datetime(2025, 6, 1, tzinfo=datetime.UTC),
                _make_full_roi_mask(tmp_path / "roi2", 8, 8),
            )

    def test_append_creates_missing_variables_from_layout(self, tmp_path):
        """A run staging vars the store lacks creates them (full time extent) instead of KeyError."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        output = str(tmp_path / "out.zarr")
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=6, x_start=0, x_stop=6)
        roi = _make_full_roi_mask(tmp_path, 6, 6)

        # Run 1: no obs counts -> store created without obs arrays.
        self._stage_one(writer, chunk, "run1", with_obs=False)
        writer.assemble(
            [chunk],
            total_y=6,
            total_x=6,
            run_id="run1",
            output_path=output,
            roi_zarr_path=roi,
            run_started_at=datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC),
            n_workers=1,
        )
        ds = open_store(output)
        assert "s2_obs_count" not in ds
        ds.close()

        # Run 2 at a later date stages obs counts -> arrays created, prior step reads fill.
        self._stage_one(writer, chunk, "run2", with_obs=True)
        writer.assemble(
            [chunk],
            total_y=6,
            total_x=6,
            run_id="run2",
            output_path=output,
            roi_zarr_path=roi,
            run_started_at=datetime.datetime(2025, 6, 1, tzinfo=datetime.UTC),
            n_workers=1,
        )
        ds = open_store(output)
        assert ds["s2_obs_count"].shape == (2, 6, 6)
        assert np.all(ds["s2_obs_count"].values[0] == 0), "pre-existing timestep must read as fill"
        assert ds["s2_obs_count"].values[1].max() >= 0
        ds.close()


class TestAssembleGlobalGuards:
    """Per-tile and probe guards on the global shard path."""

    TILE = TestAssembleGlobal.TILE
    ZONE = TestAssembleGlobal.ZONE

    def test_truncated_non_probe_tile_raises(self, tmp_path):
        """A short tile that is NOT the probe still fails loudly, naming the tile."""
        dim = 8
        seeder = TestAssembleGlobal()
        store_path = seeder._seed_zone_repo(tmp_path, 2 * self.TILE, self.TILE, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        full = ChunkSpec(row=0, col=0, y_start=0, y_stop=self.TILE, x_start=0, x_stop=self.TILE)
        short = ChunkSpec(row=1, col=0, y_start=self.TILE, y_stop=self.TILE + 8, x_start=0, x_stop=self.TILE)
        for chunk in (full, short):
            emb = np.ones((chunk.height, chunk.width, dim), dtype=np.int8)
            writer.write_chunk(chunk, emb, "runT", scales=np.ones((chunk.height, chunk.width), dtype=np.float32))

        with pytest.raises(ValueError, match=r"chunk_1_0\.zarr has embeddings extent 8 x 64 px"):
            writer.assemble_global(store_path, self.ZONE, year=2025, run_id="runT", n_workers=1)

    def test_wrong_dtype_probe_raises(self, tmp_path):
        """Float-staged embeddings must not silently C-cast into the int8 store."""
        dim = 8
        seeder = TestAssembleGlobal()
        store_path = seeder._seed_zone_repo(tmp_path, self.TILE, self.TILE, dim)
        writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=dim)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=self.TILE, x_start=0, x_stop=self.TILE)
        emb = np.ones((self.TILE, self.TILE, dim), dtype=np.float32)  # unquantized!
        writer.write_chunk(chunk, emb, "runF", scales=np.ones((self.TILE, self.TILE), dtype=np.float32))

        with pytest.raises(IncompleteStageError, match="dtype float32, expected int8"):
            writer.assemble_global(store_path, self.ZONE, year=2025, run_id="runF", n_workers=1)


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


def test_radar_coverage_counts_fully_free_tiles_from_a_generator():
    """`summarise_radar_coverage` takes an Iterable, so it may traverse it exactly once.

    `tiles_fully_s1_free` was computed by a second pass over `results`. With a list that
    works; with the generator the signature invites, the aggregation loop has already
    exhausted the iterable, so the second pass sees nothing and the year records zero
    fully-radar-free tiles however many there were. Silent, and wrong in the direction that
    hides a coverage gap.

    Two calls with identical content, one materialised and one lazy, must agree.
    """
    tiles = [
        {"status": "success", "valid_pixels": 100, "s1_free_pixels": 100, "s1_thin_pixels": 0, "s2_thin_pixels": 0},
        {"status": "success", "valid_pixels": 100, "s1_free_pixels": 100, "s1_thin_pixels": 0, "s2_thin_pixels": 0},
        {"status": "success", "valid_pixels": 100, "s1_free_pixels": 10, "s1_thin_pixels": 0, "s2_thin_pixels": 0},
    ]

    from_list = _assembly_mod.summarise_radar_coverage(tiles)
    from_generator = _assembly_mod.summarise_radar_coverage(t for t in tiles)

    assert from_list is not None and from_generator is not None
    assert from_list["tiles_fully_s1_free"] == 2, "two of the three tiles are entirely radar-free"
    assert from_generator == from_list, "a one-shot iterable must give the same answer as a list"
