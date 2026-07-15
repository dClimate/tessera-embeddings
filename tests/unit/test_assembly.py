"""Tests for ZarrWriter: round-trip write/read, assembly, and cleanup dispatch.

Ported from the yield-modeling reference repo, adapted to tessera-embeddings'
import paths. cleanup_staging coverage diverges from the reference: this repo's
cleanup_staging prefers s5cmd for S3 with an fsspec fallback (rather than
propagating s5cmd errors), so TestCleanupStagingDispatch exercises the routing
decision instead of error propagation.
"""

from __future__ import annotations

import pickle
from importlib.metadata import version as _dist_version
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
import zarr
from dask.distributed import Client

import tessera_embeddings.inference.assembly as _assembly_mod
from tessera_embeddings.config.inference import EMBEDDING_DIM
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.inference.assembly import (
    BAND_CHUNK_DIVISOR,
    OBS_COUNT_VARS,
    STAGED_READ_CONFIG_KWARGS,
    IncompleteStageError,
    ZarrWriter,
    _staged_storage_options,
)
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.conventions import ENCODER_VERSION
from tessera_embeddings.inference.quantization import quantize_embeddings
from tessera_embeddings.storage.zarr_store import open_or_create_repo, open_store


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


def _quantized_embeddings(
    rng: np.random.Generator, h: int, w: int, dim: int = EMBEDDING_DIM
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random float32 embeddings and return (int8_quantized, scales)."""
    raw = rng.standard_normal((h, w, dim)).astype(np.float32)
    return quantize_embeddings(raw)


@pytest.fixture(scope="class")
def dask_client():
    """Local Dask client for assembly tests (exercises Dask code path in to_icechunk).

    Class-scoped to avoid creating/destroying a Dask cluster for every test method.
    """
    client = Client(n_workers=2, threads_per_worker=1, memory_limit="2GB")
    yield client
    client.close()


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
    """Tests for ZarrWriter.assemble (Dask-based parallel assembly via to_icechunk)."""

    def test_assemble_multiple_chunks(self, tmp_path, dask_client):
        """Assemble 2x2 chunk grid into single output store."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "assemble_test"

        rng = np.random.default_rng(42)
        chunks = [
            ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5),
            ChunkSpec(row=0, col=1, y_start=0, y_stop=5, x_start=5, x_stop=10),
            ChunkSpec(row=1, col=0, y_start=5, y_stop=8, x_start=0, x_stop=5),
            ChunkSpec(row=1, col=1, y_start=5, y_stop=8, x_start=5, x_stop=10),
        ]

        expected = np.zeros((8, 10, EMBEDDING_DIM), dtype=np.int8)
        for chunk in chunks:
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

        # Read back assembled output (Icechunk store)
        ds = open_store(output)
        result = ds["embeddings"].values[0, ...]
        np.testing.assert_array_equal(result, expected)
        assert ds["embeddings"].shape == (1, 8, 10, EMBEDDING_DIM)
        assert "time" in ds.coords

    def test_assemble_xarray_readable(self, tmp_path, dask_client):
        """Assembled output must be readable with correct dims and coords."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "xr_test"

        rng = np.random.default_rng(99)
        chunks = [
            ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5),
            ChunkSpec(row=0, col=1, y_start=0, y_stop=5, x_start=5, x_stop=10),
        ]

        expected = np.zeros((5, 10, EMBEDDING_DIM), dtype=np.int8)
        for chunk in chunks:
            emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
            writer.write_chunk(chunk, emb, run_id, scales=scales)
            expected[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop, :] = emb

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

    def test_assemble_with_std(self, tmp_path, dask_client):
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

    def test_assemble_appends_to_existing_store(self, tmp_path, dask_client):
        """Second assemble call appends along time dimension instead of overwriting."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)

        rng = np.random.default_rng(42)
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5)

        roi = _make_full_roi_mask(tmp_path, 5, 5)

        # Run 1
        emb1, scales1 = _quantized_embeddings(rng, 5, 5)
        writer.write_chunk(chunk, emb1, run_id="run1", scales=scales1)
        writer.assemble(
            [chunk], total_y=5, total_x=5, run_id="run1", output_path=output, roi_zarr_path=roi, n_workers=1
        )

        ds = open_store(output)
        assert ds["embeddings"].shape == (1, 5, 5, EMBEDDING_DIM)
        ds.close()

        # Run 2 — should append
        emb2, scales2 = _quantized_embeddings(rng, 5, 5)
        writer.write_chunk(chunk, emb2, run_id="run2", scales=scales2)
        writer.assemble(
            [chunk], total_y=5, total_x=5, run_id="run2", output_path=output, roi_zarr_path=roi, n_workers=1
        )

        ds = open_store(output)
        assert ds["embeddings"].shape == (2, 5, 5, EMBEDDING_DIM)
        assert ds.sizes["time"] == 2
        np.testing.assert_array_equal(ds["embeddings"].values[0, ...], emb1)
        np.testing.assert_array_equal(ds["embeddings"].values[1, ...], emb2)

    def test_assemble_append_with_std(self, tmp_path, dask_client):
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
            n_workers=1,
        )

        # Run 2 with std
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
            n_workers=1,
        )

        ds = open_store(output)
        assert ds["embeddings"].shape == (2, 4, 4, EMBEDDING_DIM)
        assert ds["embedding_std"].shape == (2, 4, 4, EMBEDDING_DIM)
        np.testing.assert_array_almost_equal(ds["embedding_std"].values[0, ...], std1, decimal=5)
        np.testing.assert_array_almost_equal(ds["embedding_std"].values[1, ...], std2, decimal=5)

    def test_append_preserves_time_windows_across_runs(self, tmp_path, dask_client):
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

    def test_assemble_with_obs_counts(self, tmp_path, dask_client):
        """Stage 2x2 grid with obs counts, assemble, verify final zarr has all three vars."""
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "obs_assemble"

        rng = np.random.default_rng(42)
        chunks = [
            ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5),
            ChunkSpec(row=0, col=1, y_start=0, y_stop=5, x_start=5, x_stop=10),
            ChunkSpec(row=1, col=0, y_start=5, y_stop=8, x_start=0, x_stop=5),
            ChunkSpec(row=1, col=1, y_start=5, y_stop=8, x_start=5, x_stop=10),
        ]

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

    def test_assemble_treats_skip_marker_chunks_as_fill(self, tmp_path, dask_client):
        """A live chunk with a skip marker (no staged zarr) falls through to zero-fill.

        Regression for a crash where assemble() passed skip-marker chunks into
        _build_mosaic's live_labels, and _assemble_var_block then attempted to
        open the non-existent zarr group and raised GroupNotFoundError.
        """
        staging = str(tmp_path / "staging")
        output = str(tmp_path / "output.zarr")
        writer = ZarrWriter(staging)
        run_id = "skip_marker_test"

        rng = np.random.default_rng(7)
        chunks = [
            ChunkSpec(row=0, col=0, y_start=0, y_stop=5, x_start=0, x_stop=5),
            ChunkSpec(row=0, col=1, y_start=0, y_stop=5, x_start=5, x_stop=10),
            ChunkSpec(row=1, col=0, y_start=5, y_stop=8, x_start=0, x_stop=5),
            ChunkSpec(row=1, col=1, y_start=5, y_stop=8, x_start=5, x_stop=10),
        ]

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

        ds = open_store(output)
        result = ds["embeddings"].values[0, ...]
        # Skipped chunk's footprint is zeros (int8 fill); other chunks match what we staged.
        np.testing.assert_array_equal(result, expected)
        skipped_region = result[
            skipped_chunk.y_start : skipped_chunk.y_stop, skipped_chunk.x_start : skipped_chunk.x_stop
        ]
        assert skipped_region.shape == (skipped_chunk.height, skipped_chunk.width, EMBEDDING_DIM)
        assert np.all(skipped_region == 0)

    def test_assemble_sparse_roi_non_live_chunks_are_zero_filled(self, tmp_path, dask_client):
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

        ds = open_store(output)
        result = ds["embeddings"].values[0, ...]
        np.testing.assert_array_equal(result, expected)
        non_live_region = result[non_live.y_start : non_live.y_stop, non_live.x_start : non_live.x_stop]
        assert np.all(non_live_region == 0), "Non-live chunk region must be zero-filled"

    def test_assemble_sparse_roi_scales_nan_filled_for_non_live(self, tmp_path, dask_client):
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


class TestCleanupStagingDispatch:
    """cleanup_staging routes S3 to s5cmd with an fsspec fallback.

    Diverges from the reference repo's TestCleanupStaging, which asserted
    s5cmd errors propagate. This repo's cleanup_staging falls back to fsspec
    instead, so the routing decision is what we pin here. Both _s5cmd_rm and
    the fsspec filesystem are mocked — the routing decision is the unit under
    test, not subprocess argv or S3 DELETE semantics.
    """

    def test_s3_uses_s5cmd_and_skips_fsspec(self):
        """S3 target delegates to _s5cmd_rm with the right URL; fsspec untouched."""
        writer = ZarrWriter("s3://bucket/staging")
        with (
            patch.object(_assembly_mod, "_s5cmd_rm") as s5cmd,
            patch.object(_assembly_mod, "_fs_for") as fs_for,
        ):
            writer.cleanup_staging("run123")

        s5cmd.assert_called_once()
        assert s5cmd.call_args.args[0] == "s3://bucket/staging/run123"
        # s5cmd succeeded — fsspec must not be touched (no double-delete).
        fs_for.assert_not_called()

    def test_cleanup_preserves_other_runs(self):
        """Only the target run_id appears in the S3 URL passed to s5cmd."""
        writer = ZarrWriter("s3://bucket/staging")
        with patch.object(_assembly_mod, "_s5cmd_rm") as s5cmd:
            writer.cleanup_staging("delete_me")

        assert s5cmd.call_count == 1
        assert "delete_me" in s5cmd.call_args.args[0]
        assert "keep_me" not in s5cmd.call_args.args[0]

    @pytest.mark.parametrize(
        "exc",
        [
            FileNotFoundError("s5cmd binary not found"),
            RuntimeError("s5cmd failed (rc=1): boom"),
        ],
    )
    def test_s3_falls_back_to_fsspec_when_s5cmd_fails(self, exc):
        """Both error modes _s5cmd_rm declares fall through to fsspec rm."""
        writer = ZarrWriter("s3://bucket/staging")
        fs = MagicMock()
        fs.exists.return_value = True
        with (
            patch.object(_assembly_mod, "_s5cmd_rm", side_effect=exc) as s5cmd,
            patch.object(_assembly_mod, "_fs_for", return_value=fs) as fs_for,
        ):
            writer.cleanup_staging("run123")

        s5cmd.assert_called_once()
        fs_for.assert_called_once_with("s3://bucket/staging/run123")
        fs.rm.assert_called_once_with("s3://bucket/staging/run123", recursive=True)

    def test_local_path_skips_s5cmd(self):
        """Non-S3 target goes straight to fsspec without invoking s5cmd."""
        writer = ZarrWriter("/tmp/staging")
        fs = MagicMock()
        fs.exists.return_value = True
        with (
            patch.object(_assembly_mod, "_s5cmd_rm") as s5cmd,
            patch.object(_assembly_mod, "_fs_for", return_value=fs) as fs_for,
        ):
            writer.cleanup_staging("run123")

        s5cmd.assert_not_called()
        fs_for.assert_called_once_with("/tmp/staging/run123")
        fs.rm.assert_called_once_with("/tmp/staging/run123", recursive=True)

    def test_local_path_missing_dir_is_noop(self):
        """The fsspec rm is skipped when the staging dir doesn't exist."""
        writer = ZarrWriter("/tmp/staging")
        fs = MagicMock()
        fs.exists.return_value = False
        with patch.object(_assembly_mod, "_fs_for", return_value=fs):
            writer.cleanup_staging("run123")

        fs.rm.assert_not_called()


class TestDetectStagedChunkSize:
    """Tests for the public detect_staged_chunk_size method."""

    def test_raises_when_no_staged_chunks(self, tmp_path):
        """FileNotFoundError when the staging directory is empty."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        with pytest.raises(FileNotFoundError, match="No staged chunks found"):
            writer.detect_staged_chunk_size("run1")

    def test_uses_first_available_chunk(self, tmp_path):
        """Returns chunk_size from first staged label — not chunk_0_0 specifically."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        # Stage only a non-origin chunk (simulates sparse ROI where chunk_0_0 is empty)
        chunk = ChunkSpec(row=3, col=5, y_start=6000, y_stop=8000, x_start=10000, x_stop=12000)
        rng = np.random.default_rng(0)
        emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
        writer.write_chunk(chunk, emb, "run1", scales=scales)

        result = writer.detect_staged_chunk_size("run1")
        assert result == 2000

    def test_returns_max_of_height_width(self, tmp_path):
        """Edge chunks with non-square extents return max(h, w)."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        # Non-square edge chunk: height=500, width=2000
        chunk = ChunkSpec(row=0, col=1, y_start=0, y_stop=500, x_start=2000, x_stop=4000)
        rng = np.random.default_rng(1)
        emb, scales = _quantized_embeddings(rng, chunk.height, chunk.width)
        writer.write_chunk(chunk, emb, "run1", scales=scales)

        result = writer.detect_staged_chunk_size("run1")
        assert result == 2000


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


class TestMonolithicChunks:
    """Tests for assembly of staged files with monolithic (non-sub-chunked) chunks.

    Production data from earlier runs may have chunks=(2500, 2500, 128), meaning each
    staged file is a single on-disk chunk. _build_var_grid must detect this and create
    one task per staged file (not 25 sub-tasks that each read the full chunk).
    """

    def test_detect_monolithic_chunk_size(self, tmp_path):
        """_detect_staged_chunk_size reads actual on-disk chunks, not assumed 500x500."""
        staging = str(tmp_path / "staging")
        writer = ZarrWriter(staging)
        run_id = "mono_detect"

        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=20, x_start=0, x_stop=20)
        emb = np.random.default_rng(42).random((20, 20, EMBEDDING_DIM)).astype(np.float32)
        path = writer._staging_path(run_id, chunk)
        _write_monolithic_staged_zarr(path, emb, 0, 0)

        # Verify on-disk chunks are monolithic
        g = zarr.open_group(path, mode="r")
        assert g["embeddings"].metadata.chunk_grid.chunk_shape == (20, 20, EMBEDDING_DIM)

        # _detect_staged_chunk_size should return the monolithic shape
        h, w, band = writer._detect_staged_chunk_size(run_id, [chunk], "embeddings")
        assert h == 20
        assert w == 20
        assert band == EMBEDDING_DIM

    def test_detect_subchunked_size(self, tmp_path):
        """_detect_staged_chunk_size returns sub-chunked sizes from write_chunk."""
        staging = str(tmp_path / "staging")
        writer = ZarrWriter(staging)
        run_id = "sub_detect"

        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=10, x_start=0, x_stop=10)
        emb, scales = _quantized_embeddings(np.random.default_rng(42), 10, 10)
        writer.write_chunk(chunk, emb, run_id, scales=scales)

        # write_chunk uses min(500, height) for spatial, EMBEDDING_DIM // BAND_CHUNK_DIVISOR for band
        h, w, band = writer._detect_staged_chunk_size(run_id, [chunk], "embeddings")
        assert h == 10
        assert w == 10
        assert band == EMBEDDING_DIM // BAND_CHUNK_DIVISOR

    def test_assemble_monolithic_roundtrip(self, tmp_path, dask_client):
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

    def test_monolithic_one_task_per_chunk(self, tmp_path):
        """Monolithic staged files produce one dask task per ChunkSpec, not 25."""
        staging = str(tmp_path / "staging")
        writer = ZarrWriter(staging)
        run_id = "task_count"

        rng = np.random.default_rng(42)
        chunks = [
            ChunkSpec(row=0, col=0, y_start=0, y_stop=20, x_start=0, x_stop=20),
            ChunkSpec(row=0, col=1, y_start=0, y_stop=20, x_start=20, x_stop=40),
        ]
        for chunk in chunks:
            emb, _ = _quantized_embeddings(rng, chunk.height, chunk.width)
            path = writer._staging_path(run_id, chunk)
            _write_monolithic_staged_zarr(path, emb, chunk.y_start, chunk.x_start)

        live_labels = {c.label for c in chunks}
        arr = writer._build_var_grid(chunks, live_labels, run_id, "embeddings")

        # Monolithic 20x20 chunks → each file is one task → 2 total tasks
        assert arr.npartitions == 2


class TestGraphSize:
    """Regression tests for the assembly graph size.

    The old implementation emitted one materialized graph entry per sub-chunk,
    causing the Dask scheduler to OOM on large ROIs (~3M tasks for a 40x40
    ChunkSpec grid at sub_band=4). The map_blocks design produces two
    unmaterialized Blockwise layers whose size is independent of block count.
    """

    def _chunk_grid(self, n_rows: int, n_cols: int, chunk_size: int = 20) -> list[ChunkSpec]:
        return [
            ChunkSpec(
                row=r,
                col=c,
                y_start=r * chunk_size,
                y_stop=(r + 1) * chunk_size,
                x_start=c * chunk_size,
                x_stop=(c + 1) * chunk_size,
            )
            for r in range(n_rows)
            for c in range(n_cols)
        ]

    def _stage_one_chunk(self, writer: ZarrWriter, chunk: ChunkSpec, run_id: str) -> None:
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((chunk.height, chunk.width, EMBEDDING_DIM)).astype(np.float32)
        _write_monolithic_staged_zarr(writer._staging_path(run_id, chunk), emb, chunk.y_start, chunk.x_start)

    def test_graph_layers_stay_unmaterialized(self, tmp_path):
        """_build_var_grid returns an array whose graph layers are all Blockwise and unmaterialized."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        run_id = "graph_layers"
        chunks = self._chunk_grid(4, 4)
        # Stage one live chunk (the detector probes staged files, so at least one is required).
        self._stage_one_chunk(writer, chunks[0], run_id)
        live_labels = {chunks[0].label}

        arr = writer._build_var_grid(chunks, live_labels, run_id, "embeddings")
        graph = arr.__dask_graph__()
        for layer in graph.layers.values():
            assert not layer.is_materialized(), f"layer {type(layer).__name__} is materialized"

    def test_graph_size_independent_of_grid_size(self, tmp_path):
        """Serialized graph size is nearly constant across grid sizes (1x1, 10x10, 50x50).

        The map_blocks design produces two unmaterialized Blockwise layers whose
        pickled size grows only with hash/chunks-spec metadata, not with n_blocks.
        A 2500x increase in blocks must produce less than a 5x increase in size —
        this would catch a regression to the old per-block materialized graph.
        """
        sizes: dict[int, int] = {}
        for n_side in (1, 10, 50):
            run_id = f"grid_{n_side}"
            writer = ZarrWriter(str(tmp_path / f"staging_{n_side}"))
            chunks = self._chunk_grid(n_side, n_side)
            self._stage_one_chunk(writer, chunks[0], run_id)
            live_labels = {chunks[0].label}

            arr = writer._build_var_grid(chunks, live_labels, run_id, "embeddings")
            graph = arr.__dask_graph__()
            sizes[n_side] = sum(len(pickle.dumps(layer)) for layer in graph.layers.values())

        assert sizes[50] < sizes[1] * 5, f"graph size grew non-trivially with grid size: {sizes}"

    def test_dense_sparse_graph_size_equal(self, tmp_path):
        """Fully dense and fully sparse grids produce comparably small graphs.

        Old implementation: sparse grids caused the explosion (per-fill-chunk tasks).
        New implementation: density affects only the live_lookup dict, not graph size.
        """
        run_id = "density"
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunks = self._chunk_grid(6, 6)
        self._stage_one_chunk(writer, chunks[0], run_id)

        # Sparse: only one live chunk
        arr_sparse = writer._build_var_grid(chunks, {chunks[0].label}, run_id, "embeddings")
        # Dense: pretend all are live (staged file detection only probes one)
        arr_dense = writer._build_var_grid(chunks, {c.label for c in chunks}, run_id, "embeddings")

        size_sparse = sum(len(pickle.dumps(layer)) for layer in arr_sparse.__dask_graph__().layers.values())
        size_dense = sum(len(pickle.dumps(layer)) for layer in arr_dense.__dask_graph__().layers.values())
        # Dense graph can be larger only by the lookup dict; still bounded and small.
        assert size_dense < 500_000, f"dense graph unexpectedly large: {size_dense} bytes"
        assert size_sparse < 500_000, f"sparse graph unexpectedly large: {size_sparse} bytes"


class TestAssemblyValidation:
    """Tests for assembly-time validation and metadata correctness."""

    def test_assemble_validates_grid_extent(self, tmp_path, dask_client):
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

    def test_assemble_root_attrs(self, tmp_path, dask_client):
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

    def test_assemble_sets_geozarr_convention_attrs(self, tmp_path, dask_client):
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

    def test_invalid_shape_raises(self, tmp_path):
        """Staged chunk with wrong shape raises RuntimeError listing the bad path."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = self.CHUNKS[0]
        # Write valid chunk first
        self._stage_chunk(writer, chunk, "run1")
        # Overwrite with wrong shape
        bad = np.zeros((5, 5, EMBEDDING_DIM), dtype=np.float32)
        path = writer._staging_path("run1", chunk)
        ds = xr.Dataset(
            {"embeddings": (["northing", "easting", "band"], bad)},
            coords={"northing": np.arange(5), "easting": np.arange(5), "band": np.arange(EMBEDDING_DIM)},
        )
        ds.to_zarr(path, mode="w")

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

    def test_skip_marker_counts_as_resolved(self, tmp_path):
        """A chunk with a skip marker instead of a staged zarr passes verification."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        self._stage_chunk(writer, self.CHUNKS[0], "run1")
        self._stage_chunk(writer, self.CHUNKS[1], "run1")
        writer.write_skip_marker(self.CHUNKS[2], "run1")

        writer.verify_staged_completeness("run1", list(self.CHUNKS))

    def test_both_staged_and_skipped_raises(self, tmp_path):
        """A chunk with BOTH a staged zarr and a skip marker is inconsistent."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        for chunk in self.CHUNKS:
            self._stage_chunk(writer, chunk, "run1")
        writer.write_skip_marker(self.CHUNKS[0], "run1")

        with pytest.raises(IncompleteStageError, match="BOTH a staged zarr and a skip marker"):
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
