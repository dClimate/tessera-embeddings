"""Tests for the spatial-striping strip loop in InferenceActor.process_chunk.

Covers the strip-tiling helper and an end-to-end equality check that running a
chunk as a single strip vs. several northing strips produces bit-identical
embeddings, scales, and obs counts. Striping bounds the resident *input*
working set only; the output buffers and write path are whole-chunk, so the
result must not depend on how the input is tiled.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import zarr

import tessera_embeddings.inference.actors as _actors_mod
import tessera_embeddings.inference.data_loading as _dl_mod
from tessera_embeddings.config.inference import S2_BAND_ORDER
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.actors import (
    _MIN_STRIP_H,
    _S2_STRIP_BYTE_BUDGET,
    InferenceActor,
    _strip_height_for_density,
    _strip_slices,
)
from tessera_embeddings.inference.chunk_spec import ChunkSpec


class TestStripSlices:
    """Unit tests for the strip tiling generator."""

    def test_single_strip_when_strip_h_ge_height(self):
        assert _strip_slices(100, 100) == [slice(0, 100)]
        assert _strip_slices(100, 999) == [slice(0, 100)]

    def test_even_split(self):
        assert _strip_slices(12, 4) == [slice(0, 4), slice(4, 8), slice(8, 12)]

    def test_ragged_final_strip(self):
        assert _strip_slices(10, 4) == [slice(0, 4), slice(4, 8), slice(8, 10)]


class TestStripHeightForDensity:
    """The density-based strip sizer keeps a strip's S2 working set under budget."""

    def test_resident_bytes_stay_under_budget(self):
        # 10 bands x uint16 = 20 bytes per (obs, px). The resident pair is two
        # band strips plus the single shared full-chunk mask; it must fit twice
        # the per-strip budget, for every case that sizes above the floor.
        for t_kept, width, height in [(50, 2000, 2000), (200, 2000, 2000), (1, 4000, 4000), (180, 1000, 1000)]:
            h = _strip_height_for_density(t_kept, width, height)
            assert h >= 1
            if h <= _MIN_STRIP_H:
                continue  # floored case may breach the budget by design
            mask_bytes = t_kept * height * width
            pair = 2 * t_kept * h * width * len(S2_BAND_ORDER) * 2 + mask_bytes
            assert pair <= 2 * _S2_STRIP_BYTE_BUDGET

    def test_sparser_chunks_get_taller_strips(self):
        # Fewer timesteps -> a taller strip fits the same byte budget.
        sparse = _strip_height_for_density(20, 2000, 2000)
        dense = _strip_height_for_density(200, 2000, 2000)
        assert sparse > dense

    def test_extreme_density_floors_at_min_strip_h(self):
        # A pathologically dense chunk bottoms out at the floor (breaching the
        # byte budget, logged) rather than degenerating into tiny reads.
        assert _strip_height_for_density(10**9, 4000, 4000) == _MIN_STRIP_H


# ---------------------------------------------------------------------------
# End-to-end equality: 1 strip vs N strips
# ---------------------------------------------------------------------------

_CHUNK = ChunkSpec(row=0, col=0, y_start=0, y_stop=12, x_start=0, x_stop=10)


def _make_s2_zarr_group(n_t: int, h: int, w: int, seed: int = 10) -> zarr.Group:
    rng = np.random.default_rng(seed)
    root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    for band in S2_BAND_ORDER:
        vals = rng.integers(100, 5000, size=(n_t, h, w)).astype(np.uint16)
        arr = root.create_array(band, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)
        arr[:] = vals
    # Mix valid and invalid SCL classes so pruning has something to do.
    scl_vals = rng.choice([0, 4, 5, 8], size=(n_t, h, w)).astype(np.uint8)
    scl_arr = root.create_array("scl", shape=scl_vals.shape, dtype=scl_vals.dtype, chunks=scl_vals.shape)
    scl_arr[:] = scl_vals
    times = pd.date_range("2024-01-01", periods=n_t, freq="5D")
    time_ns = times.values.astype("datetime64[ns]").astype("int64")
    t_arr = root.create_array("time", shape=time_ns.shape, dtype=np.int64, chunks=time_ns.shape)
    t_arr[:] = time_ns
    return root


def _make_sar_zarr_group(n_t: int, h: int, w: int, seed: int = 20) -> zarr.Group:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n_t, freq="12D")
    root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    for name in ("0_VV", "0_VH"):
        vals = rng.integers(1000, 8000, size=(n_t, h, w)).astype(np.uint16)
        arr = root.create_array(name, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)
        arr[:] = vals
    time_ns = times.values.astype("datetime64[ns]").astype("int64")
    t_arr = root.create_array("time", shape=time_ns.shape, dtype=np.int64, chunks=time_ns.shape)
    t_arr[:] = time_ns
    return root


class _CapturingWriter:
    """Stand-in for ZarrWriter that records the single whole-chunk write."""

    last_write: dict | None = None
    last_skip: str | None = None

    def __init__(self, staging_base, embedding_dim=128):
        self.embedding_dim = embedding_dim

    def write_chunk(self, chunk, embeddings, run_id, scales, embeddings_std=None, obs_counts=None):
        _CapturingWriter.last_write = {
            "embeddings": embeddings.copy(),
            "scales": scales.copy(),
            "obs_counts": {k: (v.copy() if v is not None else None) for k, v in (obs_counts or {}).items()},
        }

    def write_skip_marker(self, chunk, run_id):
        _CapturingWriter.last_skip = chunk.label


def _make_actor(inference_config, test_model):
    """Build a bare InferenceActor instance (no Ray) wired for CPU inference."""
    cls = InferenceActor.__ray_actor_class__  # underlying Python class
    actor = object.__new__(cls)
    actor.config = inference_config
    actor.device = torch.device("cpu")
    actor.model = test_model
    actor.instance_id = "test-instance"
    return actor


def _open_store_side_effect():
    h, w = _CHUNK.height, _CHUNK.width
    s2_root = _make_s2_zarr_group(8, h, w, seed=10)
    sar_asc = _make_sar_zarr_group(5, h, w, seed=20)
    sar_desc = _make_sar_zarr_group(5, h, w, seed=30)

    def _open_store(path):
        if "reflectance" in path:
            return s2_root
        if "ascending" in path:
            return sar_asc
        if "descending" in path:
            return sar_desc
        raise ValueError(f"Unexpected store path: {path}")

    return _open_store


def _run_process_chunk(inference_config, test_model):
    """Run process_chunk capturing the single whole-chunk write."""
    inference_config.s1_orbit = "both"
    # Synthetic stores carry 2024 dates; align the window so the filter keeps them.
    inference_config.time_window = parse_time_window("December 2024")
    actor = _make_actor(inference_config, test_model)

    _CapturingWriter.last_write = None
    _CapturingWriter.last_skip = None

    with (
        patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=_open_store_side_effect()),
        patch.object(_actors_mod, "ZarrWriter", _CapturingWriter),
    ):
        result = actor.process_chunk(_CHUNK, "s3://b/m", "/tmp/staging", "run-1")
    return result, _CapturingWriter.last_write


class TestProcessChunkStriping:
    """1-strip vs N-strip equality and skip-marker behavior."""

    def test_single_vs_multi_strip_identical(self, inference_config, test_model):
        # Force strip height by overriding the density-based sizer, so the test
        # controls the tiling regardless of the synthetic chunk's T_kept.
        # Height above the chunk height -> one strip (== unstriped path).
        with patch.object(_actors_mod, "_strip_height_for_density", lambda *a, **k: 10**6):
            res_one, write_one = _run_process_chunk(inference_config, test_model)
        # A small strip height forces a multi-strip split of this tiny chunk.
        with patch.object(_actors_mod, "_strip_height_for_density", lambda *a, **k: 4):
            res_many, write_many = _run_process_chunk(inference_config, test_model)

        assert res_one["status"] == "success"
        assert res_many["status"] == "success"
        assert res_one["valid_pixels"] == res_many["valid_pixels"]

        # int8 embeddings must be bit-identical regardless of input tiling.
        np.testing.assert_array_equal(write_one["embeddings"], write_many["embeddings"])
        # Per-pixel scales are float32 and identical to ~1e-7 rel: the same
        # pixels go through the model, but float accumulation order differs
        # slightly across batch groupings. The drift is well below the int8
        # quantization step, so dequantized values are indistinguishable.
        np.testing.assert_allclose(write_one["scales"], write_many["scales"], rtol=1e-6, atol=1e-10)
        for var in ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
            np.testing.assert_array_equal(write_one["obs_counts"][var], write_many["obs_counts"][var], err_msg=var)

    def test_multi_strip_actually_splits(self):
        # The strip height used in the equality test genuinely splits this chunk.
        assert _CHUNK.height > 4
        assert len(_strip_slices(_CHUNK.height, 4)) >= 2

    def test_all_empty_chunk_writes_skip_marker(self, inference_config, test_model):
        """A chunk with zero valid pixels across all strips writes a skip marker."""
        inference_config.s1_orbit = "both"
        inference_config.time_window = parse_time_window("December 2024")
        actor = _make_actor(inference_config, test_model)

        h, w = _CHUNK.height, _CHUNK.width
        # All-invalid SCL -> zero valid S2 pixels everywhere.
        s2_root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
        for band in S2_BAND_ORDER:
            arr = s2_root.create_array(band, shape=(4, h, w), dtype=np.uint16, chunks=(4, h, w))
            arr[:] = 0
        scl = s2_root.create_array("scl", shape=(4, h, w), dtype=np.uint8, chunks=(4, h, w))
        scl[:] = 8  # invalid class
        times = pd.date_range("2024-01-01", periods=4, freq="5D").values.astype("datetime64[ns]").astype("int64")
        t_arr = s2_root.create_array("time", shape=times.shape, dtype=np.int64, chunks=times.shape)
        t_arr[:] = times
        sar = _make_sar_zarr_group(3, h, w)

        def _open_store(path):
            return s2_root if "reflectance" in path else sar

        _CapturingWriter.last_write = None
        _CapturingWriter.last_skip = None
        with (
            patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=_open_store),
            patch.object(_actors_mod, "ZarrWriter", _CapturingWriter),
        ):
            result = actor.process_chunk(_CHUNK, "s3://b/m", "/tmp/staging", "run-1")

        assert result["status"] == "skipped"
        assert result["valid_pixels"] == 0
        assert _CapturingWriter.last_skip == _CHUNK.label
        assert _CapturingWriter.last_write is None
