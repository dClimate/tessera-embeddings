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
    load_s2_mask_bundle,
    make_store_opener,
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

    def test_concurrent_band_read_matches_serial_on_real_icechunk(self, local_zarr_path, sample_reflectance_data):
        """The parallel band read is correct against a real icechunk store.

        The other tests use an in-memory zarr group; the concurrency concern is
        specific to icechunk's readonly session store, whose reads run through a
        different (async, Rust) backend. Build a real local store shaped like
        reflectance.zarr (time chunked at 1, sub-extent spatial chunks so the
        read spans multiple chunks) and assert the threaded ``_load_s2_bands``
        equals a per-band serial read.
        """
        from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group, write_dataset

        dates = [f"2024-06-{d:02d}" for d in range(1, 9)]  # 8 timesteps
        data = sample_reflectance_data(dates, height=128, width=128)
        store_path = str(local_zarr_path / "conc_tile" / "reflectance.zarr")
        write_dataset(
            store_path,
            data,
            tile_id="33UUP",
            baselines=dict.fromkeys(dates, 400),
            chunks={"time": 1, "northing": 64, "easting": 64},
            crs="EPSG:32615",
        )

        root = open_store_as_zarr_group(store_path)
        ti, ys, xs = np.arange(len(dates)), slice(0, 128), slice(0, 128)

        serial = np.empty((len(dates), 128, 128, len(S2_BAND_ORDER)), dtype=np.uint16)
        for i, band in enumerate(S2_BAND_ORDER):
            serial[:, :, :, i] = root[band].oindex[ti, ys, xs]

        # Run several times so a data race (rather than a deterministic bug)
        # has a chance to surface.
        for _ in range(10):
            result = _load_s2_bands(root, time_indices=ti, y_slice=ys, x_slice=xs)
            np.testing.assert_array_equal(result, serial)


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


class TestStripReassembly:
    """Loading a chunk as N northing strips and concatenating reproduces the whole chunk.

    The storage layout chunks time=1, so narrowing the northing extent never
    changes which on-disk chunks are read; a strip is a self-contained read of
    the same bytes. Per-strip pruning may keep a different T_kept per strip, so
    bands/masks/doys are compared per strip against the matching rows of the
    whole-chunk load.
    """

    # A taller chunk so it actually splits into multiple strips.
    _TALL_CHUNK = ChunkSpec(row=0, col=0, y_start=0, y_stop=12, x_start=0, x_stop=10)

    def _side_effect(self, n_t_s2=10, n_t_sar=5):
        h, w = self._TALL_CHUNK.height, self._TALL_CHUNK.width
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

    def _load(self, y_sub=None, s1_orbit="both"):
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=self._side_effect()):
            return load_chunk(self._TALL_CHUNK, "s3://b/m", time_window=_TIME_WINDOW, s1_orbit=s1_orbit, y_sub=y_sub)

    def test_y_sub_none_matches_full_chunk_dims(self):
        whole = self._load(y_sub=None)
        assert whole.height == self._TALL_CHUNK.height
        assert whole.width == self._TALL_CHUNK.width

    def test_strip_dims_reflect_strip(self):
        strip = self._load(y_sub=slice(4, 8))
        assert strip.height == 4
        assert strip.width == self._TALL_CHUNK.width
        assert strip.s2_bands.shape[1] == 4
        assert strip.s2_bands.shape[2] == self._TALL_CHUNK.width

    def test_strips_reassemble_to_whole_chunk(self):
        whole = self._load(y_sub=None)
        # Three strips of height 4 covering the 12-row chunk.
        slices = [slice(0, 4), slice(4, 8), slice(8, 12)]
        strips = [self._load(y_sub=s) for s in slices]

        # obs_count is per-pixel and row-independent: concatenating strips along
        # northing must equal the whole-chunk obs_count exactly.
        for var in ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
            reassembled = np.concatenate([getattr(st, var) for st in strips], axis=0)
            np.testing.assert_array_equal(reassembled, getattr(whole, var), err_msg=var)

        # Bands/masks: each strip's kept timesteps are a self-contained slice.
        # Compare each strip's rows against the whole-chunk array's matching rows,
        # restricted to the strip's kept doys (a strip may prune differently).
        for s, st in zip(slices, strips, strict=True):
            # S2: align on doys, then compare the strip's rows of those timesteps.
            for d_idx, doy in enumerate(st.s2_doys):
                whole_t = np.where(whole.s2_doys == doy)[0]
                assert whole_t.size == 1, f"doy {doy} not uniquely in whole chunk"
                np.testing.assert_array_equal(
                    st.s2_bands[d_idx], whole.s2_bands[whole_t[0], s, :, :], err_msg=f"s2_bands doy={doy}"
                )
                np.testing.assert_array_equal(
                    st.s2_masks[d_idx], whole.s2_masks[whole_t[0], s, :], err_msg=f"s2_masks doy={doy}"
                )

    def test_strip_reads_identical_pixels(self):
        """A single strip read returns the same pixels as the matching rows of a whole read."""
        whole = self._load(y_sub=None)
        strip = self._load(y_sub=slice(8, 12))
        # SAR has no per-pixel pruning here (random nonzero), so timesteps align 1:1.
        np.testing.assert_array_equal(strip.s1_asc_doys, whole.s1_asc_doys)
        np.testing.assert_array_equal(strip.s1_asc_bands, whole.s1_asc_bands[:, 8:12, :, :])


class TestSharedMaskBundle:
    """A precomputed full-chunk SCL bundle, sliced per strip, matches inline loads.

    The strip loop loads SCL once for the whole chunk and slices it per strip
    instead of re-reading. The bundle path must produce the same bands, masks,
    doys, and obs counts as the inline (mask_bundle=None) path it replaces.
    """

    _CHUNK = ChunkSpec(row=0, col=0, y_start=0, y_stop=12, x_start=0, x_stop=10)

    def _side_effect(self, n_t_s2=10, n_t_sar=5):
        h, w = self._CHUNK.height, self._CHUNK.width
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

    def test_bundle_tkept_matches_inline_prune(self):
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=self._side_effect()):
            bundle = load_s2_mask_bundle("s3://b/m", self._CHUNK, _TIME_WINDOW)
            _, inline_masks, inline_doys, inline_obs = _load_s2("s3://b/m", self._CHUNK, _TIME_WINDOW)
        # Same kept timesteps and same per-pixel obs count.
        np.testing.assert_array_equal(bundle.doys, inline_doys)
        np.testing.assert_array_equal(bundle.obs_count, inline_obs)
        np.testing.assert_array_equal(bundle.mask, inline_masks)

    def test_strip_via_bundle_matches_inline_strip(self):
        strip = slice(4, 8)
        opener = self._side_effect()
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=opener):
            bundle = load_s2_mask_bundle("s3://b/m", self._CHUNK, _TIME_WINDOW)
            bundled = load_chunk(
                self._CHUNK, "s3://b/m", time_window=_TIME_WINDOW, s1_orbit="both", y_sub=strip, mask_bundle=bundle
            )
            inline = load_chunk(self._CHUNK, "s3://b/m", time_window=_TIME_WINDOW, s1_orbit="both", y_sub=strip)
        # The bundle prunes at the chunk level while the inline strip prunes on
        # its own rows, so the bundled strip may carry extra (strip-empty)
        # timesteps. Align on doys and compare the shared timesteps' bands/masks.
        for d_idx, doy in enumerate(inline.s2_doys):
            b_idx = np.where(bundled.s2_doys == doy)[0]
            assert b_idx.size == 1, f"doy {doy} missing from bundled strip"
            np.testing.assert_array_equal(inline.s2_bands[d_idx], bundled.s2_bands[b_idx[0]], err_msg=f"doy={doy}")
            np.testing.assert_array_equal(inline.s2_masks[d_idx], bundled.s2_masks[b_idx[0]], err_msg=f"doy={doy}")
        np.testing.assert_array_equal(inline.s2_obs_count, bundled.s2_obs_count)


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


class TestSharedStoreOpener:
    """A shared opener reuses one repo handle per store path across strip loads.

    Reopening per strip would re-pay the icechunk repo open + manifest load
    every strip; the strip loop builds one ``make_store_opener`` per chunk so
    each strip reuses the same store handle instead.
    """

    _CHUNK = ChunkSpec(row=0, col=0, y_start=0, y_stop=12, x_start=0, x_stop=10)

    def _side_effect(self):
        h, w = self._CHUNK.height, self._CHUNK.width
        s2_root = _make_s2_zarr_group(10, h, w, seed=10)
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

    def test_opener_caches_each_path(self):
        """make_store_opener opens each distinct path once and returns the same group."""
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=self._side_effect()) as mock_open:
            opener = make_store_opener()
            first = opener("s3://b/m/reflectance.zarr")
            second = opener("s3://b/m/reflectance.zarr")
            other = opener("s3://b/m/sar_ascending.zarr")

        assert first is second  # same handle reused
        assert other is not first
        # reflectance opened once despite two requests; sar once.
        assert mock_open.call_count == 2

    def test_strips_share_one_open_per_store(self):
        """Loading three strips through a shared opener opens each store once."""
        with patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=self._side_effect()) as mock_open:
            opener = make_store_opener()
            for y_sub in (slice(0, 4), slice(4, 8), slice(8, 12)):
                load_chunk(
                    self._CHUNK,
                    "s3://b/m",
                    time_window=_TIME_WINDOW,
                    s1_orbit="both",
                    y_sub=y_sub,
                    store_opener=opener,
                )

        opened_paths = [call.args[0] for call in mock_open.call_args_list]
        # reflectance + sar_ascending + sar_descending, each opened once total
        # across all three strips.
        assert sorted(opened_paths) == [
            "s3://b/m/reflectance.zarr",
            "s3://b/m/sar_ascending.zarr",
            "s3://b/m/sar_descending.zarr",
        ]
