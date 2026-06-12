"""Tests for data_loading: band stacking, SCL masking, DOY, and load_chunk orchestration."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import zarr

import tessera_embeddings.inference.data_loading as _dl_mod
from tessera_embeddings.config.inference import S2_BAND_ORDER, SCL_VALID_CLASSES
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.data_loading import (
    _load_s2,
    _load_s2_bands,
    _load_sar_orbit,
    _load_scl_mask,
    load_chunk,
    resolve_s1_orbit,
)
from tessera_embeddings.storage.zarr_store import compute_doy


class TestLoadS2Bands:
    """Tests for S2 band stacking."""

    def test_band_order_and_shape(self):
        n_times, h, w = 5, 10, 10
        rng = np.random.default_rng(42)

        root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
        expected = {}
        for i, band in enumerate(S2_BAND_ORDER):
            vals = rng.integers(100 * (i + 1), 100 * (i + 2), size=(n_times, h, w)).astype(np.uint16)
            arr = root.create_array(band, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)
            arr[:] = vals
            expected[band] = vals

        result = _load_s2_bands(root, time_indices=np.arange(n_times), y_slice=slice(0, h), x_slice=slice(0, w))

        assert result.shape == (n_times, h, w, 10)
        assert result.dtype == np.uint16
        for i, band in enumerate(S2_BAND_ORDER):
            np.testing.assert_array_equal(result[:, :, :, i], expected[band])


class TestLoadSclMask:
    """Tests for SCL-based validity masking."""

    @pytest.mark.parametrize(
        "scl_values,expected",
        [
            (list(SCL_VALID_CLASSES), [1] * len(SCL_VALID_CLASSES)),
            ([0, 1, 2, 3, 8, 9], [0] * 6),
            ([0, 4, 8, 5, 3, 11], [0, 1, 0, 1, 0, 1]),
        ],
        ids=["all_valid", "all_invalid", "mixed"],
    )
    def test_scl_masking(self, scl_values, expected):
        scl = np.array(scl_values, dtype=np.uint8).reshape(1, -1, 1)
        root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
        arr = root.create_array("scl", shape=scl.shape, dtype=scl.dtype, chunks=scl.shape)
        arr[:] = scl
        mask = _load_scl_mask(root, np.array([0]), slice(0, scl.shape[1]), slice(0, 1))
        np.testing.assert_array_equal(mask, np.array(expected, dtype=np.bool_).reshape(1, -1, 1))


class TestComputeDoy:
    """Tests for day-of-year computation."""

    def test_doy_values(self):
        times = np.array(["2024-01-01", "2024-06-15", "2024-12-31"], dtype="datetime64[ns]")
        doys = compute_doy(times)
        assert doys.dtype == np.int32
        assert doys[0] == 1
        assert doys[1] == 167
        assert doys[2] == 366

    def test_non_leap_year(self):
        times = np.array(["2023-12-31"], dtype="datetime64[ns]")
        doys = compute_doy(times)
        assert doys[0] == 365


# ---------------------------------------------------------------------------
# Helpers for building synthetic zarr stores (used by load_chunk tests)
# ---------------------------------------------------------------------------

_CHUNK = ChunkSpec(row=0, col=0, y_start=0, y_stop=8, x_start=0, x_stop=10)
_TIME_WINDOW = parse_time_window("December 2024")


def _make_s2_zarr_group(n_t: int, h: int, w: int, seed: int = 42) -> zarr.Group:
    rng = np.random.default_rng(seed)
    root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    for band in S2_BAND_ORDER:
        vals = rng.integers(100, 5000, size=(n_t, h, w)).astype(np.uint16)
        arr = root.create_array(band, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)
        arr[:] = vals
    scl_vals = rng.choice([0, 4, 5, 8], size=(n_t, h, w)).astype(np.uint8)
    scl_arr = root.create_array("scl", shape=scl_vals.shape, dtype=scl_vals.dtype, chunks=scl_vals.shape)
    scl_arr[:] = scl_vals
    times = pd.date_range("2024-01-01", periods=n_t, freq="5D")
    time_ns = times.values.astype("datetime64[ns]").astype("int64")
    t_arr = root.create_array("time", shape=time_ns.shape, dtype=np.int64, chunks=time_ns.shape)
    t_arr[:] = time_ns
    return root


def _make_sar_zarr_group(n_t: int, h: int, w: int, seed: int = 99) -> zarr.Group:
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


class TestLoadSarOrbit:
    """Tests for the _load_sar_orbit helper."""

    def test_shapes_and_dtypes(self):
        n_t, h, w = 6, _CHUNK.height, _CHUNK.width
        root = _make_sar_zarr_group(n_t, h, w)
        with patch.object(_dl_mod, "open_store_as_zarr_group", return_value=root):
            bands, doys = _load_sar_orbit("s3://bucket/mosaic", _CHUNK, "ascending", _TIME_WINDOW)

        assert bands.shape == (n_t, h, w, 2)
        assert bands.dtype == np.uint16
        assert doys.shape == (n_t,)
        assert doys.dtype == np.int32

    def test_store_path_uses_orbit_name(self):
        root = _make_sar_zarr_group(3, _CHUNK.height, _CHUNK.width)
        with patch.object(_dl_mod, "open_store_as_zarr_group", return_value=root) as mock_open:
            _load_sar_orbit("s3://bucket/mosaic", _CHUNK, "descending", _TIME_WINDOW)
        mock_open.assert_called_once_with("s3://bucket/mosaic/sar_descending.zarr")

    def test_band_values_match_source(self):
        n_t, h, w = 4, _CHUNK.height, _CHUNK.width
        root = _make_sar_zarr_group(n_t, h, w)
        with patch.object(_dl_mod, "open_store_as_zarr_group", return_value=root):
            bands, _ = _load_sar_orbit("s3://bucket/mosaic", _CHUNK, "ascending", _TIME_WINDOW)

        np.testing.assert_array_equal(bands[:, :, :, 0], root["0_VV"][:])
        np.testing.assert_array_equal(bands[:, :, :, 1], root["0_VH"][:])


class TestLoadS2:
    """Tests for the simplified _load_s2 helper (load all non-empty S2 timesteps)."""

    def test_shapes_and_dtypes(self):
        n_t, h, w = 10, _CHUNK.height, _CHUNK.width
        root = _make_s2_zarr_group(n_t, h, w)
        with patch.object(_dl_mod, "open_store_as_zarr_group", return_value=root):
            bands, masks, doys, obs_count = _load_s2("s3://b/m", _CHUNK, _TIME_WINDOW)

        t_kept = bands.shape[0]
        assert 0 < t_kept <= n_t
        assert bands.shape == (t_kept, h, w, 10)
        assert bands.dtype == np.uint16
        assert masks.shape == (t_kept, h, w)
        # Target keeps the SCL mask as bool (1 byte/elem) rather than widening to int32.
        assert masks.dtype == np.bool_
        assert doys.shape == (t_kept,)
        assert obs_count.shape == (h, w)
        assert obs_count.dtype == np.uint16

    def test_obs_count_from_full_mask(self):
        n_t, h, w = 15, _CHUNK.height, _CHUNK.width
        root = _make_s2_zarr_group(n_t, h, w)
        with patch.object(_dl_mod, "open_store_as_zarr_group", return_value=root):
            _, _, _, obs_count = _load_s2("s3://b/m", _CHUNK, _TIME_WINDOW)
        # Observation counts can go up to the full timestep count.
        assert obs_count.max() <= n_t


class TestLoadChunkOrchestration:
    """Tests for load_chunk delegating to helpers and respecting s1_orbit."""

    def _make_open_store_side_effect(self, n_t_s2=10, n_t_sar=5):
        h, w = _CHUNK.height, _CHUNK.width
        s2_root = _make_s2_zarr_group(n_t_s2, h, w, seed=10)
        sar_asc = _make_sar_zarr_group(n_t_sar, h, w, seed=20)
        sar_desc = _make_sar_zarr_group(n_t_sar, h, w, seed=30)

        def _open_store(path):
            if "reflectance" in path:
                return s2_root
            if "ascending" in path:
                return sar_asc
            if "descending" in path:
                return sar_desc
            raise ValueError(f"Unexpected store path: {path}")

        return _open_store

    def _run(self, s1_orbit="ascending", n_t_s2=10, n_t_sar=5):
        se = self._make_open_store_side_effect(n_t_s2, n_t_sar)
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=se):
            return load_chunk(_CHUNK, "s3://b/m", time_window=_TIME_WINDOW, s1_orbit=s1_orbit)

    def test_ascending_only_skips_descending(self):
        se = self._make_open_store_side_effect()
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=se) as mock_raw:
            result = load_chunk(_CHUNK, "s3://b/m", time_window=_TIME_WINDOW, s1_orbit="ascending")

        paths = [c.args[0] for c in mock_raw.call_args_list]
        assert any("sar_ascending" in p for p in paths)
        assert not any("sar_descending" in p for p in paths)
        assert result.s1_desc_bands.shape[0] == 0
        assert result.s1_desc_doys.shape[0] == 0

    def test_descending_only_skips_ascending(self):
        se = self._make_open_store_side_effect()
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=se) as mock_raw:
            result = load_chunk(_CHUNK, "s3://b/m", time_window=_TIME_WINDOW, s1_orbit="descending")

        paths = [c.args[0] for c in mock_raw.call_args_list]
        assert any("sar_descending" in p for p in paths)
        assert not any("sar_ascending" in p for p in paths)
        assert result.s1_asc_bands.shape[0] == 0
        assert result.s1_asc_doys.shape[0] == 0

    def test_obs_counts_populated(self):
        result = self._run(s1_orbit="ascending")
        h, w = _CHUNK.height, _CHUNK.width
        assert result.s2_obs_count is not None
        assert result.s2_obs_count.shape == (h, w)
        assert result.s1_asc_obs_count is not None
        assert result.s1_asc_obs_count.shape == (h, w)
        assert result.s1_desc_obs_count is not None
        assert result.s1_desc_obs_count.shape == (h, w)

    def test_empty_orbit_obs_count_is_zero(self):
        result = self._run(s1_orbit="ascending")
        np.testing.assert_array_equal(
            result.s1_desc_obs_count, np.zeros((_CHUNK.height, _CHUNK.width), dtype=np.uint16)
        )

    def test_both_orbits_loads_asc_and_desc(self):
        se = self._make_open_store_side_effect()
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=se) as mock_raw:
            result = load_chunk(_CHUNK, "s3://b/m", time_window=_TIME_WINDOW, s1_orbit="both")

        paths = [c.args[0] for c in mock_raw.call_args_list]
        assert any("sar_ascending" in p for p in paths)
        assert any("sar_descending" in p for p in paths)
        assert result.s1_asc_bands.shape[0] > 0
        assert result.s1_desc_bands.shape[0] > 0

    def test_invalid_orbit_raises(self):
        with pytest.raises(ValueError, match="Invalid s1_orbit"):
            load_chunk(_CHUNK, "s3://b/m", time_window=_TIME_WINDOW, s1_orbit="sideways")


class TestResolveS1Orbit:
    """Tests for resolve_s1_orbit: downgrade 'both' when only one store is present."""

    @staticmethod
    def _probe(present_orbits):
        def _open_store(path):
            for orbit in ("ascending", "descending"):
                if f"sar_{orbit}" in path:
                    if orbit in present_orbits:
                        return object()
                    raise FileNotFoundError(path)
            raise ValueError(f"Unexpected path: {path}")

        return _open_store

    def test_single_orbit_passthrough(self):
        assert resolve_s1_orbit("s3://b/m", "ascending") == "ascending"
        assert resolve_s1_orbit("s3://b/m", "descending") == "descending"

    def test_invalid_rejected(self):
        with pytest.raises(ValueError, match="Invalid s1_orbit"):
            resolve_s1_orbit("s3://b/m", "sideways")

    def test_both_with_both_present_stays_both(self):
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=self._probe({"ascending", "descending"})):
            assert resolve_s1_orbit("s3://b/m", "both") == "both"

    def test_both_with_only_ascending_downgrades(self):
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=self._probe({"ascending"})):
            assert resolve_s1_orbit("s3://b/m", "both") == "ascending"

    def test_both_with_only_descending_downgrades(self):
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=self._probe({"descending"})):
            assert resolve_s1_orbit("s3://b/m", "both") == "descending"

    def test_both_with_neither_present_raises(self):
        with (
            patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=self._probe(set())),
            pytest.raises(InsufficientCoverageError, match="no SAR stores found"),
        ):
            resolve_s1_orbit("s3://b/m", "both")
