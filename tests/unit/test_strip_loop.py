"""Tests for the spatial-striping strip loop in InferenceActor.process_chunk.

Covers the strip-tiling helper and an end-to-end equality check that running a
chunk as a single strip vs. several northing strips produces bit-identical
embeddings, scales, and obs counts. Striping bounds the resident *input*
working set only; the output buffers and write path are whole-chunk, so the
result must not depend on how the input is tiled.
"""

from __future__ import annotations

import contextlib
import threading
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

import tessera_embeddings.inference.actors as _actors_mod
import tessera_embeddings.inference.data_loading as _dl_mod
from tessera_embeddings.config.inference import S2_BAND_ORDER
from tessera_embeddings.config.store_layout import MONTHS_IN_YEAR
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.actors import (
    _MIN_STRIP_H,
    _S2_FOREGROUND_DECODE_READERS,
    _S2_STRIP_BYTE_BUDGET,
    _XCHUNK_DISABLE_ENV,
    InferenceActor,
    _strip_height_for_density,
    _strip_plan,
    _strip_slices,
    _StripPlan,
    _xchunk_rung,
)
from tessera_embeddings.inference.assembly import summarise_optical_skips
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.resource_monitor import ResourceMonitor
from tests.unit.mosaic_stores import S2_SEED, make_s2_group, make_sar_group, store_opener


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
        # 10 bands x uint16 = 20 bytes per (obs, px). Every resident band set —
        # bands(strip_h) + a full mask charge — must fit ONE budget, so the
        # intra-chunk strip-prefetch pair fits 2 x budget with margin (the mask
        # is actually shared; charging it per set is deliberate conservatism).
        # Checked for every case that sizes above the floor.
        for t_kept, width, height in [(50, 2000, 2000), (200, 2000, 2000), (1, 4000, 4000), (180, 1000, 1000)]:
            h = _strip_height_for_density(t_kept, width, height)
            assert h >= 1
            if h <= _MIN_STRIP_H:
                continue  # floored case may breach the budget by design
            mask_bytes = t_kept * height * width
            resident_set = t_kept * h * width * len(S2_BAND_ORDER) * 2 + mask_bytes
            assert resident_set <= _S2_STRIP_BYTE_BUDGET

    def test_sparser_chunks_get_taller_strips(self):
        # Fewer timesteps -> a taller strip fits the same byte budget.
        sparse = _strip_height_for_density(20, 2000, 2000)
        dense = _strip_height_for_density(200, 2000, 2000)
        assert sparse > dense

    def test_single_strip_when_full_height_fits_one_budget(self):
        # A chunk whose full-height bands + mask fit ONE budget runs as a single
        # strip; anything larger splits so the intra-chunk strip-prefetch pair
        # stays bounded by 2 x budget.
        h = _strip_height_for_density(60, 2000, 2000)
        assert h == 2000
        full = 60 * 2000 * 2000 * len(S2_BAND_ORDER) * 2 + 60 * 2000 * 2000
        assert full <= _S2_STRIP_BYTE_BUDGET

    def test_dense_single_chunk_splits_to_respect_budget(self):
        # Regression rooted in the 2026-07-17 OOM (chunk_5_9, T_kept=122): a
        # pair-budget fast path once let a T=122 chunk run as ONE ~10 GB strip
        # and co-residency killed the node at 95% RAM. Every resident band set
        # must individually fit one budget.
        for t_kept in (90, 122, 160, 250):
            h = _strip_height_for_density(t_kept, 2000, 2000)
            mask_bytes = t_kept * 2000 * 2000
            per_row = t_kept * 2000 * len(S2_BAND_ORDER) * 2
            # Whether single or split, each resident band set + full mask charge
            # fits one budget (so any resident pair fits two).
            if h > _MIN_STRIP_H:
                assert per_row * h + mask_bytes <= _S2_STRIP_BYTE_BUDGET
        # The OOM case specifically must split.
        assert _strip_height_for_density(122, 2000, 2000) < 2000

    def test_extreme_density_floors_at_min_strip_h(self):
        # A pathologically dense chunk bottoms out at the floor (breaching the
        # byte budget, logged) rather than degenerating into tiny reads.
        assert _strip_height_for_density(10**9, 4000, 4000) == _MIN_STRIP_H

    def test_pair_budget_strip_charges_decode_transient(self):
        # A pair-budget strip is read in the FOREGROUND, so its momentary peak is
        # the resident stacked result PLUS the concurrent per-band decode
        # temporaries. Sized with decode_readers, that momentary peak — not just
        # the resident set — must fit the pair budget (else a high-T foreground
        # read overshoots the RAM ceiling mid-decode).
        pair = 2 * _S2_STRIP_BYTE_BUDGET
        readers = _S2_FOREGROUND_DECODE_READERS
        for t_kept, width, height in [(120, 2000, 2000), (200, 2000, 2000), (90, 4000, 4000)]:
            h = _strip_height_for_density(t_kept, width, height, pair, decode_readers=readers)
            if h <= _MIN_STRIP_H:
                continue  # floored case may breach the budget by design
            mask_bytes = t_kept * height * width
            resident = t_kept * h * width * len(S2_BAND_ORDER) * 2
            transient = readers * t_kept * h * width * 2
            assert resident + transient + mask_bytes <= pair
        # Sanity: charging the transient yields a strictly shorter strip than
        # sizing the resident set alone (the pre-fix behaviour).
        naive = _strip_height_for_density(200, 2000, 2000, pair)
        charged = _strip_height_for_density(200, 2000, 2000, pair, decode_readers=readers)
        assert charged < naive


class TestStripPlan:
    """The strip-plan chooses a tiling + prefetch mode; RAM stays bounded."""

    def _peak_resident_bytes(self, plan: _StripPlan, t_kept: int, width: int, height: int) -> int:
        # Largest single resident S2 set (a strip's bands + the full mask). With
        # prefetch two sets co-reside; without, only one does. We check the
        # per-set bound and let the caller compare against 1x or 2x budget.
        mask_bytes = t_kept * height * width
        tallest = max(s.stop - s.start for s in plan.strips)
        return t_kept * tallest * width * len(S2_BAND_ORDER) * 2 + mask_bytes

    def test_fits_one_budget_is_single_strip_no_prefetch(self):
        # T=60 @ 2000^2: bands+mask fit one budget -> one strip, nothing to prefetch.
        plan = _strip_plan(t_kept=60, height=2000, width=2000, valid_px=3_800_000)
        assert plan.strips == [slice(0, 2000)]
        assert plan.prefetch is False
        assert plan.strategy == "single"

    def test_dense_hideable_prefetches_and_stays_under_one_budget(self):
        # T=120 @ 2000^2, nearly full -> must split; long inference hides the
        # loads -> prefetch on, each resident set within ONE budget (pair <= 2x).
        plan = _strip_plan(t_kept=120, height=2000, width=2000, valid_px=3_800_000)
        assert plan.prefetch is True
        assert len(plan.strips) >= 2
        assert "dense/prefetch" in plan.strategy
        assert self._peak_resident_bytes(plan, 120, 2000, 2000) <= _S2_STRIP_BYTE_BUDGET

    def test_dense_hideable_applies_starter_strip(self):
        # A tall-body dense chunk gets a small starter first strip so the GPU
        # starts early; the body hides behind it.
        plan = _strip_plan(t_kept=120, height=2000, width=2000, valid_px=5_000_000)
        assert plan.strips[0] == slice(0, 256)
        assert plan.prefetch is True
        assert "starter" in plan.strategy

    def test_wide_but_few_valid_px_disables_prefetch(self):
        # T=200 @ 2000^2 but almost no valid pixels: inference too short to hide
        # loads -> prefetch OFF, strips sized to the pair budget so only ONE set
        # is resident (peak still bounded by the dense pair = 2x budget).
        plan = _strip_plan(t_kept=200, height=2000, width=2000, valid_px=50_000)
        assert plan.prefetch is False
        assert plan.strategy in ("no-prefetch", "single/wide-budget")
        assert self._peak_resident_bytes(plan, 200, 2000, 2000) <= 2 * _S2_STRIP_BYTE_BUDGET

    def test_wide_few_valid_px_fitting_pair_is_single_strip(self):
        # T=100 needs >1 budget but <= the pair EVEN once its foreground decode
        # temporaries are counted; non-hideable -> single strip at the wider
        # budget, prefetch off.
        plan = _strip_plan(t_kept=100, height=2000, width=2000, valid_px=30_000)
        assert plan.strips == [slice(0, 2000)]
        assert plan.prefetch is False
        assert plan.strategy == "single/wide-budget"

    def test_pair_budget_plan_fits_foreground_decode_transient(self):
        # A wider/higher-T non-hideable chunk that would fit the pair budget as
        # one RESIDENT set but NOT once its foreground per-band decode temporaries
        # are counted must SPLIT — so no strip's momentary read peak overshoots
        # the pair ceiling (pre-fix this single-stripped and could OOM mid-read).
        plan = _strip_plan(t_kept=120, height=2000, width=2000, valid_px=1000)
        assert plan.pair_budget is True
        assert plan.prefetch is False
        assert len(plan.strips) >= 2  # would have been a single strip pre-fix
        pair = 2 * _S2_STRIP_BYTE_BUDGET
        readers = _S2_FOREGROUND_DECODE_READERS
        mask_bytes = 120 * 2000 * 2000
        for s in plan.strips:
            sh = s.stop - s.start
            resident = 120 * sh * 2000 * len(S2_BAND_ORDER) * 2
            transient = readers * 120 * sh * 2000 * 2
            assert resident + transient + mask_bytes <= pair


# ---------------------------------------------------------------------------
# End-to-end equality: 1 strip vs N strips
# ---------------------------------------------------------------------------

_CHUNK = ChunkSpec(row=0, col=0, y_start=0, y_stop=12, x_start=0, x_stop=10)


class _CapturingWriter:
    """Stand-in for ZarrWriter that records the single whole-chunk write."""

    last_write: dict | None = None
    last_skip: str | None = None
    last_skip_record: dict | None = None
    discarded: str | None = None

    def __init__(self, staging_base, embedding_dim=128):
        self.embedding_dim = embedding_dim

    def write_chunk(self, chunk, embeddings, run_id, scales, embeddings_std=None, obs_counts=None, month_covered=None):
        _CapturingWriter.last_write = {
            "embeddings": embeddings.copy(),
            "scales": scales.copy(),
            "obs_counts": {k: (v.copy() if v is not None else None) for k, v in (obs_counts or {}).items()},
            # Recorded, not discarded: a fake that accepts an argument and drops it leaves an
            # actor that never populates the buffer indistinguishable from one that does.
            "month_covered": month_covered.copy() if month_covered is not None else None,
        }

    def discard_coverage(self, chunk, run_id):
        """The real writer removes a partial coverage tile here; the fake records the intent."""
        _CapturingWriter.discarded = chunk.label

    def write_skip_marker(self, chunk, run_id, record=None):
        _CapturingWriter.last_skip = chunk.label
        # Recorded, not discarded — a fake that drops the record leaves an actor which never builds
        # one indistinguishable from one that does, and the record is the whole registry.
        _CapturingWriter.last_skip_record = record


def _make_actor(inference_config, test_model):
    """Build a bare InferenceActor instance (no Ray) wired for CPU inference."""
    cls = InferenceActor.__ray_actor_class__  # underlying Python class
    actor = object.__new__(cls)
    actor.config = inference_config
    actor.device = torch.device("cpu")
    actor.model = test_model
    actor.instance_id = "test-instance"
    actor._get_credentials = None  # no scoped provider; opens use the default chain
    actor._s3_region = None  # default region
    # Unstarted monitor: process_chunk tags phase context on it (no thread runs).
    actor._resource_monitor = ResourceMonitor()
    return actor


def _open_store_side_effect():
    return store_opener(_CHUNK, n_t_s2=8, n_t_sar=5)


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
        # The staging write is deferred to the actor's writer thread; drain it
        # so last_write reflects this chunk before we read it.
        flushed = actor.flush_writes()
        assert flushed is None or flushed["ok"], f"deferred write failed: {flushed}"
    return result, _CapturingWriter.last_write


def _force_strip_plan(strip_h: int, prefetch: bool, pair_budget: bool = False):
    """Patch target for ``_strip_plan`` that forces a fixed tiling + prefetch mode.

    Lets a test control the strip count, the prefetch-on/off branch, and the
    ``pair_budget`` flag (which gates the cross-chunk prefetch) regardless of
    the synthetic chunk's density estimates.
    """

    def _plan(_t_kept, height, _width, _valid_px, mask_width=None):
        return _StripPlan(
            strips=_strip_slices(height, strip_h),
            prefetch=prefetch,
            strategy="test",
            strip_h=strip_h,
            pair_budget=pair_budget,
        )

    return _plan


class TestProcessChunkStriping:
    """1-strip vs N-strip equality and skip-marker behavior."""

    def test_single_vs_multi_strip_identical(self, inference_config, test_model):
        # Force the tiling via _strip_plan so the test controls both the strip
        # count and the prefetch branch, regardless of the synthetic chunk's
        # density. strip_h > height -> one strip (== unstriped path).
        with patch.object(_actors_mod, "_strip_plan", _force_strip_plan(10**6, prefetch=False)):
            res_one, write_one = _run_process_chunk(inference_config, test_model)
        # A small strip height forces a multi-strip split of this tiny chunk,
        # exercised on BOTH the prefetch-on (dense) and prefetch-off (sparse)
        # paths — all three tilings must yield bit-identical output.
        with patch.object(_actors_mod, "_strip_plan", _force_strip_plan(4, prefetch=True)):
            res_pf, write_pf = _run_process_chunk(inference_config, test_model)
        with patch.object(_actors_mod, "_strip_plan", _force_strip_plan(4, prefetch=False)):
            res_nopf, write_nopf = _run_process_chunk(inference_config, test_model)

        assert res_one["status"] == res_pf["status"] == res_nopf["status"] == "success"
        assert res_one["valid_pixels"] == res_pf["valid_pixels"] == res_nopf["valid_pixels"]

        for other in (write_pf, write_nopf):
            # int8 embeddings must be bit-identical regardless of input tiling
            # or prefetch mode.
            np.testing.assert_array_equal(write_one["embeddings"], other["embeddings"])
            # Per-pixel scales are float32 and identical to ~1e-7 rel: the same
            # pixels go through the model, but float accumulation order differs
            # slightly across batch groupings. The drift is well below the int8
            # quantization step, so dequantized values are indistinguishable.
            # equal_nan: ungenerated pixels carry a NaN scale in every tiling.
            np.testing.assert_allclose(write_one["scales"], other["scales"], rtol=1e-6, atol=1e-10, equal_nan=True)
            for var in ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
                np.testing.assert_array_equal(write_one["obs_counts"][var], other["obs_counts"][var], err_msg=var)
            # Month coverage is assembled strip by strip like the obs counts, so a
            # tiling-dependent bug would show as a mismatch on the strip boundaries.
            np.testing.assert_array_equal(
                write_one["month_covered"], other["month_covered"], err_msg="s2_month_covered"
            )

    def test_month_coverage_is_populated_and_agrees_with_the_obs_count(self, inference_config, test_model):
        """The actor must pass month flags that describe its own observations.

        A LIVENESS test, deliberately: the buffer is allocated zeroed and threaded through
        several optional arguments, so an actor that never fills it produces all-``False`` —
        which is also what a pixel with no observations correctly produces, and what the store
        held for a whole published zone-year while every other array beside it was right.
        Nothing downstream can tell those apart, so the check has to be that the flags are
        both PRESENT and consistent with the count they partition.
        """
        with patch.object(_actors_mod, "_strip_plan", _force_strip_plan(4, prefetch=False)):
            result, write = _run_process_chunk(inference_config, test_model)

        assert result["status"] == "success"
        covered = write["month_covered"]
        obs = write["obs_counts"]["s2_obs_count"]
        assert covered is not None, "the actor passed no month coverage at all"
        assert covered.shape == (MONTHS_IN_YEAR, _CHUNK.height, _CHUNK.width)
        assert covered.dtype == np.bool_
        assert covered.any(), "month coverage is entirely False despite a successful chunk"
        # Every flagged month needs an observation behind it, and a pixel cannot cover more
        # months than it has observations. Both directions, since either alone admits a bug:
        # an all-True buffer satisfies "some month is set" but not this.
        months_per_px = covered.sum(axis=0)
        assert np.all(months_per_px <= obs), "a pixel covers more months than it has observations"
        assert np.all((months_per_px > 0) == (obs > 0)), "a month flag and the obs count disagree on emptiness"
        # Exactly the months the fixture's own S2 time axis falls in — every pixel of this
        # synthetic chunk is valid at every timestep, so the flags are the axis's months and
        # nothing else. Read off the fixture rather than written out, so it stays true if the
        # fixture's date spacing changes.
        s2_times = np.asarray(make_s2_group(8, _CHUNK.height, _CHUNK.width, seed=S2_SEED)["time"][:])
        want_months = sorted(
            {int(m) for m in s2_times.astype("datetime64[ns]").astype("datetime64[M]").astype(int) % 12 + 1}
        )
        assert [m + 1 for m in range(MONTHS_IN_YEAR) if covered[m].any()] == want_months

    def test_multi_strip_actually_splits(self):
        # The strip height used in the equality test genuinely splits this chunk.
        assert _CHUNK.height > 4
        assert len(_strip_slices(_CHUNK.height, 4)) >= 2

    def test_hung_background_strip_fails_chunk_instead_of_wedging(self, inference_config, test_model, monkeypatch):
        # A background strip load that hangs must make process_chunk FAIL FAST
        # (returning status="failed" so the scheduler replaces the actor), NOT
        # block forever. The strip pool is managed explicitly precisely so the
        # timeout's raise escapes to the failure handler instead of being
        # swallowed by ThreadPoolExecutor.__exit__'s wait=True shutdown
        # re-joining the same wedged worker. If this regressed the test would
        # HANG rather than fail — the watchdog around the suite bounds that.
        inference_config.s1_orbit = "both"
        inference_config.time_window = parse_time_window("December 2024")
        actor = _make_actor(inference_config, test_model)
        real_load = _actors_mod.load_chunk
        release = threading.Event()
        calls = {"n": 0}

        def _blocking_load(*a, **k):
            calls["n"] += 1
            if calls["n"] >= 2:  # strip 0 (prologue) loads; strip 1 (background) hangs
                release.wait()
            return real_load(*a, **k)

        monkeypatch.setattr(_actors_mod, "_BACKGROUND_IO_TIMEOUT_S", 0.3)
        try:
            with (
                patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=_open_store_side_effect()),
                patch.object(_actors_mod, "ZarrWriter", _CapturingWriter),
                patch.object(_actors_mod, "_strip_plan", _force_strip_plan(4, prefetch=True)),
                patch.object(_actors_mod, "load_chunk", side_effect=_blocking_load),
            ):
                result = actor.process_chunk(_CHUNK, "s3://b/m", "/tmp/staging", "run-1")
        finally:
            release.set()  # unblock the leaked worker so the pool can drain
        assert result["status"] == "failed"
        assert "Background strip" in result["error"]

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
        sar = make_sar_group(3, h, w)

        def _open_store(path, region=None):  # region tolerated (make_store_opener threads it)
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
        # THE REGISTRY'S PRODUCER SIDE. A fully refused chunk used to write a zero-byte marker, so a
        # thin-depth refusal could not be told from land that was never imaged. The reason has to be
        # written HERE: the cell is write-once and its mosaic is deleted when it lands.
        record = _CapturingWriter.last_skip_record
        assert record is not None, "a refused chunk must record WHY it was refused"
        assert record["label"] == _CHUNK.label
        assert set(record["refused"]) == {"no_optical", "thin", "no_radar"}
        assert sum(record["refused"].values()) > 0, "something refused every pixel; say what"
        # The observation summary rides along, because the counts alone cannot say HOW thin.
        # MEDIAN as well as mean, and both over pixels that saw ANYTHING: a mean is pulled by a
        # bright patch of deep pixels beside a dark majority, and including never-imaged pixels
        # drags either statistic to zero so it describes neither population.
        assert "s2_obs" in record
        assert set(record["s2_obs"]) == {"px_with_any", "max", "mean_where_any", "median_where_any"}
        # Radar presence, because a tile that is thin AND radar-free is a different cleanup
        # candidate from one that is merely thin — more optical will not fix the first.
        assert "px_with_any_radar" in record


# ---------------------------------------------------------------------------
# Deferred staging writes (actor side)
# ---------------------------------------------------------------------------

_CHUNK_B2 = ChunkSpec(row=1, col=0, y_start=0, y_stop=12, x_start=0, x_stop=10)


class TestDeferredStagingWrites:
    """process_chunk defers the write; outcomes ride the next result / flush."""

    def _run_two_chunks(self, inference_config, test_model, writer_cls):
        inference_config.s1_orbit = "both"
        inference_config.time_window = parse_time_window("December 2024")
        actor = _make_actor(inference_config, test_model)
        with (
            patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=_open_store_side_effect()),
            patch.object(_actors_mod, "ZarrWriter", writer_cls),
        ):
            r1 = actor.process_chunk(_CHUNK, "s3://b/m", "/tmp/staging", "run-1")
            r2 = actor.process_chunk(_CHUNK_B2, "s3://b/m", "/tmp/staging", "run-1")
            flushed = actor.flush_writes()
        return r1, r2, flushed

    def test_writes_defer_and_confirm_in_chain(self, inference_config, test_model):
        r1, r2, flushed = self._run_two_chunks(inference_config, test_model, _CapturingWriter)

        assert r1["status"] == "success" and r1["write_deferred"] is True
        assert r1["prior_write"] is None  # nothing pending on a fresh actor
        # Chunk 2's result carries chunk 1's (successful) write outcome.
        assert r2["prior_write"] == {"label": _CHUNK.label, "ok": True, "error": None}
        # Chunk 2's own write drains via flush.
        assert flushed is not None and flushed["label"] == _CHUNK_B2.label and flushed["ok"] is True
        # And the write actually happened (the capturing writer recorded it).
        assert _CapturingWriter.last_write is not None

    def test_failed_write_surfaces_on_next_call(self, inference_config, test_model):
        class _FailingWriter(_CapturingWriter):
            def write_chunk(
                self, chunk, embeddings, run_id, scales, embeddings_std=None, obs_counts=None, month_covered=None
            ):
                raise OSError("S3 500")

        r1, r2, flushed = self._run_two_chunks(inference_config, test_model, _FailingWriter)

        assert r1["write_deferred"] is True and r1["prior_write"] is None
        prior = r2["prior_write"]
        assert prior["label"] == _CHUNK.label and prior["ok"] is False
        assert "S3 500" in prior["error"]
        assert flushed["label"] == _CHUNK_B2.label and flushed["ok"] is False

    def test_flush_with_nothing_pending_returns_none(self, inference_config, test_model):
        actor = _make_actor(inference_config, test_model)
        assert actor.flush_writes() is None

    def test_hung_write_times_out_and_surfaces_failure(self, inference_config, test_model, monkeypatch):
        # A wedged staging upload must not block the actor forever: the bounded
        # wait returns ok=False so the scheduler can requeue the chunk on a
        # healthy actor (rather than the actor hanging inside process_chunk,
        # where the tail flush_writes() timeout can never reach it).
        actor = _make_actor(inference_config, test_model)
        pool = actor._writer_pool_handle()
        release = threading.Event()
        actor._pending_write = ("chunk_hung", pool.submit(release.wait))
        monkeypatch.setattr(_actors_mod, "_BACKGROUND_IO_TIMEOUT_S", 0.2)
        try:
            outcome = actor.flush_writes()
        finally:
            release.set()  # let the leaked writer task finish so the pool closes
        assert outcome is not None
        assert outcome["label"] == "chunk_hung"
        assert outcome["ok"] is False
        assert outcome["timed_out"] is True  # flags the wedged-writer case to the scheduler
        assert "did not complete" in outcome["error"]

    def test_checked_collect_raises_on_wedged_writer(self, inference_config, test_model, monkeypatch):
        # On the hot path a wedged writer must fail the whole chunk (raise) so
        # the scheduler kills+replaces the actor — rather than deferring another
        # write behind the stuck one. flush_writes (idle path) stays non-raising.
        actor = _make_actor(inference_config, test_model)
        pool = actor._writer_pool_handle()
        release = threading.Event()
        actor._pending_write = ("chunk_hung", pool.submit(release.wait))
        monkeypatch.setattr(_actors_mod, "_BACKGROUND_IO_TIMEOUT_S", 0.2)
        try:
            with pytest.raises(RuntimeError, match="writer pool wedged"):
                actor._collect_prior_write_checked()
        finally:
            release.set()


# ---------------------------------------------------------------------------
# Empty-strip S2 band-read skip
# ---------------------------------------------------------------------------


class TestEmptyStripBandReadSkip:
    """Strips with zero valid pixels must not pay the S2 band read."""

    def _make_half_empty_stores(self):
        """Synthetic stores where rows 0-5 are all-invalid SCL, rows 6-11 valid-ish."""
        h, w = _CHUNK.height, _CHUNK.width
        rng = np.random.default_rng(99)
        s2_root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
        for band in S2_BAND_ORDER:
            vals = rng.integers(100, 5000, size=(6, h, w)).astype(np.uint16)
            arr = s2_root.create_array(band, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)
            arr[:] = vals
        scl_vals = np.full((6, h, w), 8, dtype=np.uint8)  # invalid everywhere...
        scl_vals[:, 6:, :] = rng.choice([4, 5, 8], size=(6, h - 6, w)).astype(np.uint8)  # ...except lower rows
        scl = s2_root.create_array("scl", shape=scl_vals.shape, dtype=scl_vals.dtype, chunks=scl_vals.shape)
        scl[:] = scl_vals
        times = pd.date_range("2024-12-01", periods=6, freq="3D").values.astype("datetime64[ns]").astype("int64")
        t_arr = s2_root.create_array("time", shape=times.shape, dtype=np.int64, chunks=times.shape)
        t_arr[:] = times
        sar_asc = make_sar_group(4, h, w, seed=201)
        sar_desc = make_sar_group(4, h, w, seed=202)

        def _open_store(path, region=None):  # region tolerated (make_store_opener threads it)
            if "reflectance" in path:
                return s2_root
            if "ascending" in path:
                return sar_asc
            return sar_desc

        return _open_store

    def _run(self, inference_config, test_model, strip_h):
        inference_config.s1_orbit = "both"
        inference_config.time_window = parse_time_window("December 2024")
        actor = _make_actor(inference_config, test_model)
        _CapturingWriter.last_write = None
        with (
            patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=self._make_half_empty_stores()),
            patch.object(_actors_mod, "ZarrWriter", _CapturingWriter),
            patch.object(_actors_mod, "_strip_plan", _force_strip_plan(strip_h, prefetch=True)),
            patch.object(_dl_mod, "_load_s2_bands", wraps=_dl_mod._load_s2_bands) as band_spy,
        ):
            result = actor.process_chunk(_CHUNK, "s3://b/m", "/tmp/staging", "run-1")
            flushed = actor.flush_writes()
            assert flushed is None or flushed["ok"]
        return result, _CapturingWriter.last_write, band_spy

    def test_empty_strip_skips_band_read(self, inference_config, test_model):
        # strip_h=6 → strip 0 (rows 0-5) is all-invalid, strip 1 (rows 6-11) valid.
        result, write, band_spy = self._run(inference_config, test_model, strip_h=6)
        assert result["status"] == "success"
        # Only the valid strip paid a band read.
        assert band_spy.call_count == 1
        (_, kwargs) = band_spy.call_args
        assert kwargs["y_slice"] == slice(6, 12)

    def test_outputs_identical_with_and_without_skip(self, inference_config, test_model):
        # Single full-height strip (no skip possible) vs split (top strip skipped):
        # outputs must be bit-identical.
        res_one, write_one, _ = self._run(inference_config, test_model, strip_h=10**6)
        res_two, write_two, _ = self._run(inference_config, test_model, strip_h=6)
        assert res_one["valid_pixels"] == res_two["valid_pixels"] > 0
        np.testing.assert_array_equal(write_one["embeddings"], write_two["embeddings"])
        np.testing.assert_allclose(write_one["scales"], write_two["scales"], rtol=1e-6, atol=1e-10, equal_nan=True)
        for var in ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
            np.testing.assert_array_equal(write_one["obs_counts"][var], write_two["obs_counts"][var], err_msg=var)


# ---------------------------------------------------------------------------
# S2 easting-bbox crop for sparse chunks
# ---------------------------------------------------------------------------


class TestEastingBboxCrop:
    """Sliver chunks read only the valid-column window; outputs are unchanged."""

    def _make_sliver_stores(self):
        """Valid pixels confined to columns 2-4; SCL invalid everywhere else."""
        h, w = _CHUNK.height, _CHUNK.width
        rng = np.random.default_rng(7)
        s2_root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
        for band in S2_BAND_ORDER:
            vals = rng.integers(100, 5000, size=(6, h, w)).astype(np.uint16)
            arr = s2_root.create_array(band, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)
            arr[:] = vals
        scl_vals = np.full((6, h, w), 8, dtype=np.uint8)
        scl_vals[:, :, 2:5] = rng.choice([4, 5, 8], size=(6, h, 3)).astype(np.uint8)
        scl = s2_root.create_array("scl", shape=scl_vals.shape, dtype=scl_vals.dtype, chunks=scl_vals.shape)
        scl[:] = scl_vals
        times = pd.date_range("2024-12-01", periods=6, freq="3D").values.astype("datetime64[ns]").astype("int64")
        t_arr = s2_root.create_array("time", shape=times.shape, dtype=np.int64, chunks=times.shape)
        t_arr[:] = times
        sar_asc = make_sar_group(4, h, w, seed=301)
        sar_desc = make_sar_group(4, h, w, seed=302)

        def _open_store(path, region=None):  # region tolerated (make_store_opener threads it)
            if "reflectance" in path:
                return s2_root
            if "ascending" in path:
                return sar_asc
            return sar_desc

        return _open_store

    def _run(self, inference_config, test_model, crop_threshold, s1_orbit="both"):
        inference_config.s1_orbit = s1_orbit
        inference_config.time_window = parse_time_window("December 2024")
        actor = _make_actor(inference_config, test_model)
        _CapturingWriter.last_write = None
        with (
            patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=self._make_sliver_stores()),
            patch.object(_actors_mod, "ZarrWriter", _CapturingWriter),
            patch.object(_actors_mod, "_X_CROP_MIN_SAVING", crop_threshold),
            patch.object(_dl_mod, "_load_s2_bands", wraps=_dl_mod._load_s2_bands) as band_spy,
        ):
            result = actor.process_chunk(_CHUNK, "s3://b/m", "/tmp/staging", "run-1")
            flushed = actor.flush_writes()
            assert flushed is None or flushed["ok"]
        return result, _CapturingWriter.last_write, band_spy

    def test_crop_reads_only_valid_columns(self, inference_config, test_model):
        result, _, band_spy = self._run(inference_config, test_model, crop_threshold=0.10)
        assert result["status"] == "success"
        # The band read was cropped to columns 2..5 (absolute: x_start + [2, 5)).
        (_, kwargs) = band_spy.call_args
        assert kwargs["x_slice"] == slice(_CHUNK.x_start + 2, _CHUNK.x_start + 5)

    def test_cropped_outputs_bit_identical_to_uncropped(self, inference_config, test_model):
        # crop_threshold > 1 makes the box never "save enough" → crop disabled.
        res_off, write_off, spy_off = self._run(inference_config, test_model, crop_threshold=1.1)
        res_on, write_on, spy_on = self._run(inference_config, test_model, crop_threshold=0.10)

        # The uncropped run read the full width; the cropped run did not.
        assert spy_off.call_args.kwargs["x_slice"] == slice(_CHUNK.x_start, _CHUNK.x_stop)
        assert spy_on.call_args.kwargs["x_slice"] != slice(_CHUNK.x_start, _CHUNK.x_stop)

        assert res_off["valid_pixels"] == res_on["valid_pixels"] > 0
        np.testing.assert_array_equal(write_off["embeddings"], write_on["embeddings"])
        np.testing.assert_allclose(write_off["scales"], write_on["scales"], rtol=1e-6, atol=1e-10, equal_nan=True)
        # Obs layers keep full-extent fidelity, including SAR outside the box.
        for var in ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
            np.testing.assert_array_equal(write_off["obs_counts"][var], write_on["obs_counts"][var], err_msg=var)

    def test_single_orbit_crop(self, inference_config, test_model):
        # Regression (2026-07-17 run, chunk_0_0): with one orbit active, the
        # skipped orbit's empty placeholder must be FULL width going into the
        # x_sub block — a cropped-width placeholder got double-cropped, breaking
        # the dataset build and corrupting s1_*_obs_count_full.
        res_off, write_off, _ = self._run(inference_config, test_model, crop_threshold=1.1, s1_orbit="ascending")
        res_on, write_on, _ = self._run(inference_config, test_model, crop_threshold=0.10, s1_orbit="ascending")

        assert res_off["status"] == res_on["status"] == "success"
        assert write_on["obs_counts"]["s1_desc_obs_count"].shape == (_CHUNK.height, _CHUNK.width)
        assert not write_on["obs_counts"]["s1_desc_obs_count"].any()
        np.testing.assert_array_equal(write_off["embeddings"], write_on["embeddings"])
        np.testing.assert_allclose(write_off["scales"], write_on["scales"], rtol=1e-6, atol=1e-10, equal_nan=True)
        for var in ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
            np.testing.assert_array_equal(write_off["obs_counts"][var], write_on["obs_counts"][var], err_msg=var)


# ---------------------------------------------------------------------------
# Bounded cross-chunk starter prefetch
# ---------------------------------------------------------------------------

_CHUNK_B = ChunkSpec(row=1, col=0, y_start=0, y_stop=12, x_start=0, x_stop=10)
_CHUNK_C = ChunkSpec(row=2, col=0, y_start=0, y_stop=12, x_start=0, x_stop=10)


def _run_chunk_chain(inference_config, test_model, chain, extra_patches=()):
    """Process (chunk, prefetch_hint) pairs in order on ONE actor.

    Returns (writes, actor): each chunk's captured whole-chunk write, in
    order, plus the actor for stash inspection. Writes are flushed between
    chunks so ``last_write`` is unambiguous.
    """
    inference_config.s1_orbit = "both"
    inference_config.time_window = parse_time_window("December 2024")
    actor = _make_actor(inference_config, test_model)
    writes = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=_open_store_side_effect()))
        stack.enter_context(patch.object(_actors_mod, "ZarrWriter", _CapturingWriter))
        for extra in extra_patches:
            stack.enter_context(extra)
        for chunk, hint in chain:
            _CapturingWriter.last_write = None
            result = actor.process_chunk(chunk, "s3://b/m", "/tmp/staging", "run-1", prefetch_hint=hint)
            assert result["status"] == "success"
            flushed = actor.flush_writes()
            assert flushed is None or flushed["ok"], f"deferred write failed: {flushed}"
            writes.append(_CapturingWriter.last_write)
    return writes, actor


def _assert_writes_identical(a, b):
    np.testing.assert_array_equal(a["embeddings"], b["embeddings"])
    np.testing.assert_allclose(a["scales"], b["scales"], rtol=1e-6, atol=1e-10, equal_nan=True)
    for var in ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
        np.testing.assert_array_equal(a["obs_counts"][var], b["obs_counts"][var], err_msg=var)


class TestXChunkPrefetch:
    """Rung rules + end-to-end bit-identity for the cross-chunk prefetch."""

    def _plan(self, prefetch=True, pair_budget=False, starter_first=False):
        return _StripPlan(
            strips=[slice(0, 256), slice(256, 2000)],
            prefetch=prefetch,
            strategy="test",
            strip_h=1744,
            pair_budget=pair_budget,
            starter_first=starter_first,
        )

    def test_rung_rules(self):
        big = ChunkSpec(row=0, col=0, y_start=0, y_stop=2000, x_start=0, x_stop=2000)
        dense_starter = self._plan(starter_first=True)
        # Starter-first plans prefetch their (already small) starter.
        assert _xchunk_rung(big, 60, None, 4_000_000, dense_starter) == "starter"
        # Plain dense split: strips[0] is a budget-sized set — never prefetched.
        assert _xchunk_rung(big, 60, None, 4_000_000, self._plan()) == "mask-only"
        # Pair-budget (non-hideable) plans get the mask.
        assert _xchunk_rung(big, 60, None, 50_000, self._plan(prefetch=False, pair_budget=True)) == "mask-only"
        single = _StripPlan([slice(0, 2000)], prefetch=False, strategy="single", strip_h=2000)
        # Dense single-strip: converting to starter+body nets positive.
        assert _xchunk_rung(big, 60, None, 4_000_000, single) == "starter"
        # Sparse single-strip: the extra fixed read could never hide.
        assert _xchunk_rung(big, 60, None, 100_000, single) == "mask-only"
        # Byte cap: very dense chunks fall back to the mask rung.
        assert _xchunk_rung(big, 400, None, 4_000_000, dense_starter) == "mask-only"

    def test_chain_bit_identical_mask_only_rung(self, inference_config, test_model):
        # The tiny test chunk plans single-strip below the starter height, so
        # the natural rung is mask-only: the stash supplies mask + plan and the
        # first strip loads inline. Output must match a serial run exactly.
        chained, actor = _run_chunk_chain(inference_config, test_model, [(_CHUNK, _CHUNK_B), (_CHUNK_B, None)])
        serial, _ = _run_chunk_chain(inference_config, test_model, [(_CHUNK_B, None)])
        _assert_writes_identical(chained[1], serial[0])
        assert actor._xchunk_prefetched == {}  # stash consumed

    def test_chain_bit_identical_starter_rung(self, inference_config, test_model):
        # Force a multi-strip plan and the starter rung so the stash carries a
        # loaded first strip; the consuming prologue must produce identical
        # output to a serial run under the same tiling.
        patches = (
            patch.object(_actors_mod, "_strip_plan", _force_strip_plan(4, prefetch=True)),
            patch.object(_actors_mod, "_xchunk_rung", lambda *a, **k: "starter"),
        )
        chained, actor = _run_chunk_chain(inference_config, test_model, [(_CHUNK, _CHUNK_B), (_CHUNK_B, None)], patches)
        serial, _ = _run_chunk_chain(inference_config, test_model, [(_CHUNK_B, None)], patches)
        _assert_writes_identical(chained[1], serial[0])
        assert actor._xchunk_prefetched == {}

    def test_stale_stash_evicted_on_reassignment(self, inference_config, test_model):
        # Prefetched B but got C (steal/requeue): C runs serially and the
        # stale B stash is discarded — never consumed for the wrong chunk.
        chained, actor = _run_chunk_chain(inference_config, test_model, [(_CHUNK, _CHUNK_B), (_CHUNK_C, None)])
        serial, _ = _run_chunk_chain(inference_config, test_model, [(_CHUNK_C, None)])
        _assert_writes_identical(chained[1], serial[0])
        assert actor._xchunk_prefetched == {}

    def test_env_hatch_disables_prefetch(self, inference_config, test_model, monkeypatch):
        monkeypatch.setenv(_XCHUNK_DISABLE_ENV, "1")
        _, actor = _run_chunk_chain(inference_config, test_model, [(_CHUNK, _CHUNK_B)])
        # Never started: the lazily-created stash is empty (or never created).
        assert getattr(actor, "_xchunk_prefetched", {}) == {}

    def test_pair_budget_plan_skips_prefetch(self, inference_config, test_model):
        # A pair-budget plan holds a near-2x-budget set on its last strip —
        # NOT a RAM trough — so the cross-chunk prefetch must be skipped or it
        # could breach the ceiling.
        patches = (
            patch.object(_actors_mod, "_strip_plan", _force_strip_plan(10**6, prefetch=False, pair_budget=True)),
        )
        _, actor = _run_chunk_chain(inference_config, test_model, [(_CHUNK, _CHUNK_B)], patches)
        assert getattr(actor, "_xchunk_prefetched", {}) == {}

    def test_wedged_prefetch_pool_raises(self, inference_config, test_model, monkeypatch):
        # The prefetch pool is a single PERSISTENT worker: a hung task never
        # frees and would make every later prefetch time out (600s each). A
        # prefetch timeout must therefore fail the chunk (raise) so the actor +
        # its pool are replaced — not fall back inline like a mere load error.
        actor = _make_actor(inference_config, test_model)
        pool, stash = actor._prefetch_state()
        release = threading.Event()
        stash["chunk_x"] = pool.submit(release.wait)
        monkeypatch.setattr(_actors_mod, "_BACKGROUND_IO_TIMEOUT_S", 0.2)
        try:
            with pytest.raises(RuntimeError, match="prefetch pool wedged"):
                actor._take_prefetched("chunk_x")
        finally:
            release.set()


class TestAllowS2Only:
    """End-to-end through the real model: the optional per-pixel S1 requirement."""

    def _stores_with_sar_gap(self, h: int, w: int, half: int):
        """S2 valid EVERYWHERE (scl=4); SAR zeroed over columns >= half (a coverage gap)."""
        rng = np.random.default_rng(11)
        s2_root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
        for band in S2_BAND_ORDER:
            vals = rng.integers(100, 5000, size=(6, h, w)).astype(np.uint16)
            arr = s2_root.create_array(band, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)
            arr[:] = vals
        scl = s2_root.create_array("scl", shape=(6, h, w), dtype=np.uint8, chunks=(6, h, w))
        scl[:] = 4  # every pixel S2-valid at every timestep
        times = pd.date_range("2024-01-01", periods=6, freq="5D").values.astype("datetime64[ns]").astype("int64")
        t_arr = s2_root.create_array("time", shape=times.shape, dtype=np.int64, chunks=times.shape)
        t_arr[:] = times

        sar_asc = make_sar_group(5, h, w, seed=20)
        sar_desc = make_sar_group(5, h, w, seed=30)
        for grp in (sar_asc, sar_desc):
            for name in ("0_VV", "0_VH"):
                data = grp[name][:]
                data[:, :, half:] = 0  # no SAR observations right of the gap edge
                grp[name][:] = data

        def _open_store(path, region=None):
            if "reflectance" in path:
                return s2_root
            if "ascending" in path:
                return sar_asc
            if "descending" in path:
                return sar_desc
            raise ValueError(f"Unexpected store path: {path}")

        return _open_store

    def _run(self, inference_config, test_model, opener, allow_s2_only: bool):
        inference_config.s1_orbit = "both"
        inference_config.time_window = parse_time_window("December 2024")
        inference_config.allow_s2_only = allow_s2_only
        actor = _make_actor(inference_config, test_model)
        _CapturingWriter.last_write = None
        _CapturingWriter.last_skip = None
        with (
            patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=opener),
            patch.object(_actors_mod, "ZarrWriter", _CapturingWriter),
        ):
            result = actor.process_chunk(_CHUNK, "s3://b/m", "/tmp/staging", "run-1")
            # The staging write is deferred to the actor's writer thread; drain it
            # so last_write reflects this chunk before we read it (mirrors
            # _run_process_chunk — without this the read races the write thread).
            flushed = actor.flush_writes()
            assert flushed is None or flushed["ok"], f"deferred write failed: {flushed}"
        return result, _CapturingWriter.last_write

    def test_sar_gap_pixels_embed_only_with_flag(self, inference_config, test_model):
        h, w = _CHUNK.height, _CHUNK.width
        half = w // 2
        opener = self._stores_with_sar_gap(h, w, half)

        res_off, write_off = self._run(inference_config, test_model, opener, allow_s2_only=False)
        res_on, write_on = self._run(inference_config, test_model, opener, allow_s2_only=True)
        assert res_off["status"] == "success" and res_on["status"] == "success"

        # Provenance is identical either way: zero S1 observations in the gap.
        for write in (write_off, write_on):
            assert np.all(write["obs_counts"]["s1_asc_obs_count"][:, half:] == 0)
            assert np.all(write["obs_counts"]["s1_desc_obs_count"][:, half:] == 0)
            assert np.all(write["obs_counts"]["s2_obs_count"] > 0)

        # Flag OFF (historical gate): gap pixels are never embedded — NaN scale.
        assert np.all(np.isnan(write_off["scales"][:, half:]))
        assert np.all(np.isfinite(write_off["scales"][:, :half]))

        # Flag ON: every S2-valid pixel embeds — finite scales and embeddings
        # everywhere, including the SAR gap (upstream v1.1 zero-input convention).
        assert np.all(np.isfinite(write_on["scales"]))
        assert np.all(np.isfinite(write_on["embeddings"]))
        assert res_on["valid_pixels"] == h * w
        assert res_off["valid_pixels"] == h * half

        # S1-informed pixels are unaffected by the flag: identical int8 embeddings.
        np.testing.assert_array_equal(write_on["embeddings"][:, :half], write_off["embeddings"][:, :half])


class TestPerCellTimeWindow:
    """The window is a PER-CELL value carried on the work item, not an actor constant.

    A chained session's cells may span campaign years, and an actor is built once with one
    config. Reading the window from that config would make every cell of a different year
    read the wrong months, and the session's only mismatch check is on `s1_orbit`, so
    nothing would catch it. These tests pin both halves: threading it changed nothing when
    the value matches, and it genuinely takes effect when it does not.
    """

    @staticmethod
    def _run(inference_config, test_model, *, time_window=None):
        inference_config.s1_orbit = "both"
        inference_config.time_window = parse_time_window("December 2024")
        actor = _make_actor(inference_config, test_model)
        _CapturingWriter.last_write = None
        _CapturingWriter.last_skip = None
        with (
            patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=_open_store_side_effect()),
            patch.object(_actors_mod, "ZarrWriter", _CapturingWriter),
        ):
            result = actor.process_chunk(_CHUNK, "s3://b/m", "/tmp/staging", "run-1", time_window=time_window)
            flushed = actor.flush_writes()
            assert flushed is None or flushed["ok"], f"deferred write failed: {flushed}"
        return result, _CapturingWriter.last_write

    def test_passing_the_config_window_explicitly_is_bit_identical(self, inference_config, test_model):
        """The safety property: threading the window must not have changed the numbers.

        The window reaches three loader call sites — the serial prologue, the cross-chunk
        prefetch starter, and the strip loop — which are documented as needing identical
        kwargs for bit-identity. `_process_chunk` resolves the fallback ONCE into a local and
        passes that value to all three, so they cannot drift; this asserts the outcome.
        """
        res_default, write_default = self._run(inference_config, test_model)
        res_explicit, write_explicit = self._run(
            inference_config, test_model, time_window=parse_time_window("December 2024")
        )
        assert res_default["status"] == res_explicit["status"] == "success"
        assert res_default["valid_pixels"] == res_explicit["valid_pixels"] > 0
        np.testing.assert_array_equal(write_default["embeddings"], write_explicit["embeddings"])
        np.testing.assert_allclose(write_default["scales"], write_explicit["scales"], rtol=0, atol=0, equal_nan=True)
        for var in ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
            np.testing.assert_array_equal(
                write_default["obs_counts"][var], write_explicit["obs_counts"][var], err_msg=var
            )

    def test_a_different_window_actually_changes_what_is_read(self, inference_config, test_model):
        """The liveness property, and the one that would catch the override being ignored.

        A silently-dropped `time_window` argument would leave every test above passing while
        the multi-year campaign read the wrong months. The synthetic stores carry 2024 dates,
        so a 2022 window must filter them all out — a visibly different result rather than
        merely a different number.
        """
        res_2024, write_2024 = self._run(inference_config, test_model, time_window=parse_time_window("December 2024"))
        assert res_2024["valid_pixels"] > 0 and write_2024 is not None

        # The 2022 window filters every date out, and the loader raises rather than
        # quietly embedding nothing — which is a stronger liveness signal than a zero
        # count, because it can only come from the passed window reaching the loader.
        res_2022, _ = self._run(inference_config, test_model, time_window=parse_time_window("December 2022"))
        assert res_2022["status"] == "failed", res_2022
        assert "time window (2022" in res_2022.get("error", ""), res_2022

    def test_the_zone_context_carries_the_window_to_the_dispatch(self):
        """The wiring, unit-tested without a fleet: ZoneContext holds it and defaults to None.

        `None` is what the single-ROI path passes, which is what keeps that path unchanged.
        """
        from tessera_embeddings.inference.scheduling import ZoneContext

        assert ZoneContext("m", "s", "r").time_window is None
        window = parse_time_window("December 2024")
        assert ZoneContext("m", "s", "r", window).time_window is window
        # Frozen + value-based equality still works, and two years of one zone differ —
        # which is what stops a cross-cell prefetch hint reading the wrong window.
        assert ZoneContext("m", "s", "r", window) == ZoneContext("m", "s", "r", window)
        assert ZoneContext("m", "s", "r", window) != ZoneContext("m", "s", "r", parse_time_window("December 2023"))


class TestTheRecordTheConsumerWillRead:
    """The producer's record, fed to the consumer that grades it.

    These two were tested to different contracts and never against each other: the producer test
    asserted the record's SHAPE, the summary's tests asserted its ARITHMETIC over hand-built records,
    and nothing checked that what the producer emits satisfies what the consumer enforces. That gap
    is exactly where the cropping defect lived — `eligible_px` declared the whole tile while the
    reasons counted only the loaded columns, so every cropped shard was graded inconsistent by an
    invariant no test ever fed a real record to.
    """

    @staticmethod
    def _refused_run(inference_config, test_model, monkeypatch, *, crop: slice | None):
        """Drive a fully refused chunk through the real producer; return its record."""
        inference_config.s1_orbit = "both"
        inference_config.time_window = parse_time_window("December 2024")
        actor = _make_actor(inference_config, test_model)
        h, w = _CHUNK.height, _CHUNK.width
        s2_root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
        for band in S2_BAND_ORDER:
            arr = s2_root.create_array(band, shape=(4, h, w), dtype=np.uint16, chunks=(4, h, w))
            arr[:] = 0
        scl = s2_root.create_array("scl", shape=(4, h, w), dtype=np.uint8, chunks=(4, h, w))
        scl[:] = 8  # every class invalid -> every pixel refused
        times = pd.date_range("2024-01-01", periods=4, freq="5D").values.astype("datetime64[ns]").astype("int64")
        t_arr = s2_root.create_array("time", shape=times.shape, dtype=np.int64, chunks=times.shape)
        t_arr[:] = times
        sar = make_sar_group(3, h, w)

        def _open_store(path, region=None):
            return s2_root if "reflectance" in path else sar

        if crop is not None:
            # Force the read plan to narrow the chunk, which no all-invalid fixture does on its own:
            # the crop is derived from the columns holding valid pixels, and here there are none.
            real_plan = _actors_mod._chunk_read_plan

            def _cropped(chunk, mask_bundle):
                _x_sub, valid_px, plan = real_plan(chunk, mask_bundle)
                return crop, valid_px, plan

            monkeypatch.setattr(_actors_mod, "_chunk_read_plan", _cropped)

        _CapturingWriter.last_write = None
        _CapturingWriter.last_skip = None
        _CapturingWriter.last_skip_record = None
        with (
            patch.object(_dl_mod, "open_store_as_zarr_group", side_effect=_open_store),
            patch.object(_actors_mod, "ZarrWriter", _CapturingWriter),
        ):
            result = actor.process_chunk(_CHUNK, "s3://b/m", "/tmp/staging", "run-1")
        assert result["status"] == "skipped"
        record = _CapturingWriter.last_skip_record
        assert record is not None
        return record

    def test_the_consumer_grades_the_producers_record_as_consistent(self, inference_config, test_model, monkeypatch):
        """The round trip, uncropped: the reasons account for every pixel the dataset evaluated."""
        record = self._refused_run(inference_config, test_model, monkeypatch, crop=None)
        label = record["label"]
        summary = summarise_optical_skips(staged=[], skipped=[label], records={label: record})

        assert summary.get("inconsistent") is None, summary
        assert sum(summary["refused_px_by_reason"].values()) == _CHUNK.height * _CHUNK.width
        # Nothing was cropped, so nothing is unaccounted for.
        assert "not_evaluated_px" not in summary

    def test_a_cropped_chunk_is_consistent_and_its_remainder_is_named(self, inference_config, test_model, monkeypatch):
        """The defect two reviewers found, from both ends at once.

        With the read plan narrowed to four of ten columns, the reasons cover 4/10 of the tile. The
        record must say so — and the consumer must grade that as CONSISTENT while still accounting
        for the columns nobody looked at, because they are published as fill just like the refused
        ones and a summary that ignores them looks short of the footprint it explains.
        """
        crop = slice(3, 7)
        record = self._refused_run(inference_config, test_model, monkeypatch, crop=crop)
        label = record["label"]

        assert record["eligible_px"] == _CHUNK.height * 4, "the footprint the reasons actually cover"
        assert record["chunk_px"] == _CHUNK.height * _CHUNK.width, "the footprint that gets published"

        summary = summarise_optical_skips(staged=[], skipped=[label], records={label: record})
        assert summary.get("inconsistent") is None, "a legitimate crop is not a defect"
        assert summary["not_evaluated_px"] == _CHUNK.height * (_CHUNK.width - 4)
        assert (
            sum(summary["refused_px_by_reason"].values()) + summary["not_evaluated_px"] == _CHUNK.height * _CHUNK.width
        ), "refused plus never-evaluated must account for the whole published tile"

    def test_the_record_carries_the_grid_it_was_produced_on(self, inference_config, test_model, monkeypatch):
        """An assembly-only resume of an all-skipped run has no staged tile to read a chunk size
        off, and guessing the current default re-enumerates every label.
        """
        record = self._refused_run(inference_config, test_model, monkeypatch, crop=None)
        assert record["chunk_side_px"] == _CHUNK.width
