"""Live-window derivation from an ROI mask (pure, offline).

Pins the properties the write path builds on: windows are chunk-aligned and
mutually chunk-disjoint, clamped to the extent, and together cover every live
pixel of the mask — because a window the derivation misses is land that silently
never gets ingested.
"""

from __future__ import annotations

import itertools
import random

import fsspec
import numpy as np
import pytest
import zarr

from tessera_embeddings.ingest.live_windows import (
    WINDOW_COST_IN_CHUNKS,
    LiveWindow,
    _chunk_area,
    _union,
    live_chunk_grid,
    live_chunk_grid_from_keys,
    live_windows_for_mask,
    merge_bands,
    row_band_windows,
)

CHUNK = 4  # small stand-in for INGEST_CHUNK_SIZE; the code takes it as a parameter


def _area(windows) -> int:
    """Total chunk area of some windows, the unit the merge objective counts in."""
    return sum(_chunk_area(w, CHUNK) for w in windows)


def _brute_force_best(bands, cap, window_cost, chunk_px) -> int:
    """Cheapest grouping of consecutive bands, by enumerating ALL of them.

    Independent of the implementation under test — that is the point. Exponential,
    so only for the handful-of-bands cases the optimality test uses.
    """
    n = len(bands)
    best = None
    for cuts in itertools.product([0, 1], repeat=max(0, n - 1)):
        groups, start = [], 0
        for i, cut in enumerate(cuts):
            if cut:
                groups.append(bands[start : i + 1])
                start = i + 1
        groups.append(bands[start:])
        windows = [_union(g) for g in groups]
        if any(_chunk_area(w, chunk_px) > cap and len(g) > 1 for w, g in zip(windows, groups, strict=True)):
            continue
        cost = len(windows) * window_cost + sum(_chunk_area(w, chunk_px) for w in windows)
        if best is None or cost < best:
            best = cost
    return best


def _mask_store(tmp_path, mask: np.ndarray) -> str:
    path = str(tmp_path / "roi.zarr")
    z = zarr.open(path, mode="w", shape=mask.shape, chunks=(CHUNK, CHUNK), dtype="bool")
    z[:] = mask
    return path


def test_scattered_islands_one_window_per_live_row(tmp_path):
    # 03S in miniature: a couple of live pixels in an ocean of False.
    mask = np.zeros((12, 16), dtype=bool)
    mask[1, 2] = True  # chunk row 0, col 0
    mask[9, 13] = True  # chunk row 2, col 3
    path = _mask_store(tmp_path, mask)

    live = live_chunk_grid(path, chunk_px=CHUNK)
    assert live.shape == (3, 4)
    assert live.sum() == 2

    windows = live_windows_for_mask(path, chunk_px=CHUNK, merge=False)
    assert windows == [LiveWindow(y0=0, y1=4, x0=0, x1=4), LiveWindow(y0=8, y1=12, x0=12, x1=16)]


def test_row_span_covers_first_to_last_live_column(tmp_path):
    # Live chunks at columns 0 and 3 of one row: the row-band spans the gap —
    # that is the measured trade (within ~1% of exact, campaign-wide).
    mask = np.zeros((4, 16), dtype=bool)
    mask[0, 0] = True
    mask[0, 15] = True
    windows = live_windows_for_mask(_mask_store(tmp_path, mask), chunk_px=CHUNK)
    assert windows == [LiveWindow(y0=0, y1=4, x0=0, x1=16)]


def test_partial_edge_chunks_clamp_to_extent(tmp_path):
    # 10x9 extent with CHUNK=4: last chunk is partial on both axes. A live pixel
    # in the far corner must yield a window clamped to the extent, not the grid.
    mask = np.zeros((10, 9), dtype=bool)
    mask[9, 8] = True
    windows = live_windows_for_mask(_mask_store(tmp_path, mask), chunk_px=CHUNK)
    assert windows == [LiveWindow(y0=8, y1=10, x0=8, x1=9)]


def test_all_ocean_yields_no_windows(tmp_path):
    windows = live_windows_for_mask(_mask_store(tmp_path, np.zeros((8, 8), dtype=bool)), chunk_px=CHUNK)
    assert windows == []


def test_windows_cover_every_live_pixel_and_are_disjoint(tmp_path):
    # Property check on an irregular mask: coverage is total, windows are
    # chunk-aligned, and no two windows overlap.
    rng = np.random.default_rng(7)
    mask = rng.random((20, 24)) > 0.9
    path = _mask_store(tmp_path, mask)
    windows = live_windows_for_mask(path, chunk_px=CHUNK)

    covered = np.zeros_like(mask)
    for w in windows:
        assert w.y0 % CHUNK == 0 and w.x0 % CHUNK == 0
        assert not covered[w.y0 : w.y1, w.x0 : w.x1].any()  # disjoint
        covered[w.y0 : w.y1, w.x0 : w.x1] = True
    assert (mask & ~covered).sum() == 0  # nothing live left uncovered


def test_wrong_shaped_live_grid_is_rejected():
    with pytest.raises(ValueError, match="does not match extent"):
        row_band_windows(np.zeros((2, 2), dtype=bool), height=100, width=100, chunk_px=CHUNK)


def test_non_boolean_mask_is_rejected(tmp_path):
    path = str(tmp_path / "bad.zarr")
    zarr.open(path, mode="w", shape=(8, 8), chunks=(4, 4), dtype="uint8")
    with pytest.raises(ValueError, match="2-D boolean"):
        live_chunk_grid(path, chunk_px=CHUNK)


class TestGridFromChunkKeys:
    """Deriving the live grid from stored chunk keys instead of reading pixels.

    Sound only because an all-ocean chunk is never written, so the two routes must
    agree — which is what most of these assert, using the pixel scan as the oracle
    since it is the behaviour being preserved.
    """

    def test_agrees_with_the_pixel_scan(self, tmp_path):
        rng = np.random.default_rng(11)
        mask = rng.random((20, 24)) > 0.85
        path = _mask_store(tmp_path, mask)
        z = zarr.open(path, mode="r")

        from_keys = live_chunk_grid_from_keys(path, z, chunk_px=CHUNK)
        from_pixels = live_chunk_grid(path, chunk_px=CHUNK)
        assert from_keys is not None
        np.testing.assert_array_equal(from_keys, from_pixels)

    def test_windows_are_identical_either_route(self, tmp_path):
        """The public entry point must not change its answer with prefer_keys."""
        mask = np.zeros((12, 16), dtype=bool)
        mask[1, 2] = True
        mask[9, 13] = True
        path = _mask_store(tmp_path, mask)
        assert live_windows_for_mask(path, chunk_px=CHUNK, prefer_keys=True) == live_windows_for_mask(
            path, chunk_px=CHUNK, prefer_keys=False
        )

    def test_all_ocean_is_an_answer_not_a_failure(self, tmp_path):
        """A store with no chunk objects yields an empty grid, NOT None.

        Returning None here would silently fall back to a full pixel scan of a mask
        that has nothing in it.
        """
        path = _mask_store(tmp_path, np.zeros((8, 8), dtype=bool))
        z = zarr.open(path, mode="r")
        live = live_chunk_grid_from_keys(path, z, chunk_px=CHUNK)
        assert live is not None
        assert live.shape == (2, 2)
        assert not live.any()

    def test_partial_edge_chunks_are_placed_correctly(self, tmp_path):
        """A live pixel in a partial trailing chunk maps to that chunk's index."""
        mask = np.zeros((10, 9), dtype=bool)
        mask[9, 8] = True  # chunk row 2, col 2 of a 3x3 grid over 4px chunks
        path = _mask_store(tmp_path, mask)
        z = zarr.open(path, mode="r")
        live = live_chunk_grid_from_keys(path, z, chunk_px=CHUNK)
        assert live is not None
        assert live.shape == (3, 3)
        assert live[2, 2] and live.sum() == 1

    def test_multi_digit_chunk_indices_parse(self, tmp_path):
        """Indices past single digits must not be truncated or mismatched."""
        mask = np.zeros((60, 60), dtype=bool)
        mask[47, 51] = True  # chunk (11, 12) at 4px chunks
        path = _mask_store(tmp_path, mask)
        z = zarr.open(path, mode="r")
        live = live_chunk_grid_from_keys(path, z, chunk_px=CHUNK)
        assert live is not None
        assert live[11, 12] and live.sum() == 1

    def test_mismatched_chunking_falls_back(self, tmp_path):
        """Chunks that are not chunk_px cannot map 1:1, so refuse rather than guess."""
        path = str(tmp_path / "other.zarr")
        z = zarr.open(path, mode="w", shape=(16, 16), chunks=(8, 8), dtype="bool")
        z[0, 0] = True
        assert live_chunk_grid_from_keys(path, zarr.open(path, mode="r"), chunk_px=CHUNK) is None

    def test_sharded_store_falls_back(self, tmp_path):
        """A shard key is not a chunk index, so the layout is not recognised."""
        path = str(tmp_path / "sharded.zarr")
        z = zarr.create_array(store=path, shape=(16, 16), chunks=(4, 4), shards=(8, 8), dtype="bool")
        z[0, 0] = True
        assert live_chunk_grid_from_keys(path, zarr.open(path, mode="r"), chunk_px=CHUNK) is None

    def test_a_chunk_index_outside_the_grid_falls_back(self, tmp_path, monkeypatch):
        """A key that cannot belong to this grid means the layout was misread.

        Cropping on a misread layout would drop land, so this must refuse rather
        than place what it can.
        """
        mask = np.zeros((8, 8), dtype=bool)
        mask[0, 0] = True
        path = _mask_store(tmp_path, mask)
        z = zarr.open(path, mode="r")

        real = fsspec.core.url_to_fs

        def fake(p, **kw):
            fs, root = real(p, **kw)
            monkeypatch.setattr(fs, "find", lambda _r: [f"{root}/c/0/0", f"{root}/c/99/0"], raising=False)
            return fs, root

        monkeypatch.setattr(fsspec.core, "url_to_fs", fake)
        assert live_chunk_grid_from_keys(path, z, chunk_px=CHUNK) is None

    def test_a_failed_listing_falls_back(self, tmp_path, monkeypatch):
        """Listing errors degrade to the slow-but-correct scan, never to a crash."""
        mask = np.zeros((8, 8), dtype=bool)
        mask[0, 0] = True
        path = _mask_store(tmp_path, mask)
        z = zarr.open(path, mode="r")

        monkeypatch.setattr(fsspec.core, "url_to_fs", lambda *a, **k: (_ for _ in ()).throw(OSError("denied")))
        assert live_chunk_grid_from_keys(path, z, chunk_px=CHUNK) is None
        # and the public entry point still produces the right windows
        monkeypatch.undo()
        assert live_windows_for_mask(path, chunk_px=CHUNK) == [LiveWindow(y0=0, y1=4, x0=0, x1=4)]


class TestMergeBands:
    """Row bands grouped to minimise measured cost, not to minimise area.

    The objective is ``n_windows * WINDOW_COST_IN_CHUNKS + total_area`` and the
    implementation is a DP over consecutive groupings, so these pin optimality and
    the invariants a wrong grouping would break — never a particular shape.
    """

    def _cost(self, ws, window_cost=WINDOW_COST_IN_CHUNKS):
        return len(ws) * window_cost + _area(ws)

    def test_rows_sharing_a_span_merge_for_free(self):
        """Consecutive rows with the same live columns cost nothing to merge."""
        rows = [LiveWindow(y0=r * CHUNK, y1=(r + 1) * CHUNK, x0=0, x1=3 * CHUNK) for r in range(6)]
        merged = merge_bands(rows, chunk_px=CHUNK)
        assert merged == [LiveWindow(y0=0, y1=6 * CHUNK, x0=0, x1=3 * CHUNK)]
        assert _area(merged) == _area(rows)

    def test_extra_area_below_the_window_price_is_taken(self):
        """A merge that adds less area than one window costs is a win, and the
        default price (~200 chunks) makes almost every adjacent merge worth it.
        """
        rows = [
            LiveWindow(y0=0, y1=CHUNK, x0=0, x1=CHUNK),
            LiveWindow(y0=CHUNK, y1=2 * CHUNK, x0=9 * CHUNK, x1=10 * CHUNK),
        ]  # union covers 2x10 = 20 chunks against 2 -> +18, far under the price
        assert merge_bands(rows, chunk_px=CHUNK) == [LiveWindow(y0=0, y1=2 * CHUNK, x0=0, x1=10 * CHUNK)]

    def test_a_free_window_never_merges(self):
        """With a window priced at zero, minimum area wins and nothing merges —
        the degenerate case that proves the objective is really being optimised.
        """
        rows = [
            LiveWindow(y0=0, y1=CHUNK, x0=0, x1=CHUNK),
            LiveWindow(y0=CHUNK, y1=2 * CHUNK, x0=9 * CHUNK, x1=10 * CHUNK),
        ]
        assert merge_bands(rows, chunk_px=CHUNK, window_cost_in_chunks=0) == rows

    def test_expensive_area_splits_what_a_cheap_window_would_join(self):
        rows = [
            LiveWindow(y0=0, y1=CHUNK, x0=0, x1=CHUNK),
            LiveWindow(y0=CHUNK, y1=2 * CHUNK, x0=9 * CHUNK, x1=10 * CHUNK),
        ]
        assert merge_bands(rows, chunk_px=CHUNK, window_cost_in_chunks=5) == rows
        assert len(merge_bands(rows, chunk_px=CHUNK, window_cost_in_chunks=50)) == 1

    def test_chunk_cap_splits_a_tall_band(self):
        """Even a free merge stops at the area cap, so one graph stays bounded."""
        rows = [LiveWindow(y0=r * CHUNK, y1=(r + 1) * CHUNK, x0=0, x1=2 * CHUNK) for r in range(10)]
        merged = merge_bands(rows, chunk_px=CHUNK, max_tasks_per_window=4, tasks_per_chunk=1)
        assert len(merged) == 5  # 4-chunk cap = 2 rows of 2 chunks per band
        assert all(_area([w]) <= 4 for w in merged)

    def test_a_lone_band_over_the_cap_is_still_emitted(self):
        """A single chunk-row cannot be split, so the cap must not drop it —
        losing it would silently skip land.
        """
        rows = [LiveWindow(y0=0, y1=CHUNK, x0=0, x1=9 * CHUNK)]  # 9 chunks
        assert merge_bands(rows, chunk_px=CHUNK, max_tasks_per_window=2, tasks_per_chunk=1) == rows

    def test_matches_brute_force_optimum(self):
        """Optimality, checked exhaustively on small inputs rather than asserted."""
        rng = random.Random(20260725)
        for _ in range(120):
            n = rng.randint(1, 7)
            bands, y = [], 0
            for _ in range(n):
                y += rng.randint(1, 3) * CHUNK  # sometimes leaves dead rows between
                x0 = rng.randint(0, 6) * CHUNK
                bands.append(LiveWindow(y0=y, y1=y + CHUNK, x0=x0, x1=x0 + rng.randint(1, 4) * CHUNK))
            cap = rng.choice([4, 8, 64, 10**6])
            price = rng.choice([0, 1, 5, 200])
            got = merge_bands(
                bands, chunk_px=CHUNK, max_tasks_per_window=cap, tasks_per_chunk=1, window_cost_in_chunks=price
            )
            assert self._cost(got, price) == _brute_force_best(bands, cap, price, CHUNK)

    def test_merged_bands_stay_chunk_aligned_and_disjoint(self):
        """The write path commits one date across all windows in a single session,
        which requires the windows never share a chunk.
        """
        rows = [
            LiveWindow(y0=0, y1=CHUNK, x0=0, x1=2 * CHUNK),
            LiveWindow(y0=CHUNK, y1=2 * CHUNK, x0=CHUNK, x1=3 * CHUNK),
            LiveWindow(y0=5 * CHUNK, y1=6 * CHUNK, x0=8 * CHUNK, x1=9 * CHUNK),
        ]
        merged = merge_bands(rows, chunk_px=CHUNK, max_tasks_per_window=6, tasks_per_chunk=1)
        seen: set[tuple[int, int]] = set()
        for w in merged:
            assert w.y0 % CHUNK == 0 and w.x0 % CHUNK == 0
            cells = {(r, c) for r in range(w.y0 // CHUNK, w.y1 // CHUNK) for c in range(w.x0 // CHUNK, w.x1 // CHUNK)}
            assert not (cells & seen)
            seen |= cells

    def test_empty_input(self):
        assert merge_bands([], chunk_px=CHUNK) == []

    def test_merged_windows_still_cover_every_live_pixel(self, tmp_path):
        """The invariant that matters most: grouping must never drop land."""
        mask = np.zeros((6 * CHUNK, 6 * CHUNK), dtype=bool)
        mask[1, 1] = mask[CHUNK + 2, 3 * CHUNK] = mask[5 * CHUNK, 5 * CHUNK] = True
        path = _mask_store(tmp_path, mask)
        merged = live_windows_for_mask(path, chunk_px=CHUNK, merge=True)
        covered = np.zeros_like(mask)
        for w in merged:
            covered[w.y0 : w.y1, w.x0 : w.x1] = True
        assert bool((mask & ~covered).sum() == 0)

    def test_merge_off_returns_the_row_bands(self, tmp_path):
        mask = np.zeros((4 * CHUNK, 4 * CHUNK), dtype=bool)
        mask[0, 0] = mask[CHUNK, 0] = True
        path = _mask_store(tmp_path, mask)
        assert len(live_windows_for_mask(path, chunk_px=CHUNK, merge=False)) == 2
        assert len(live_windows_for_mask(path, chunk_px=CHUNK, merge=True)) == 1
