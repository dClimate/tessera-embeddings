"""The cropped coverage denominator (pure, offline).

The S2 coverage check crops BOTH sides of its
ratio: the SCL validity reduce (numerator) and the ROI pixel total
(denominator) each run over the live windows instead of the full zone grid.
That is only sound because of one property — the ROI mask is False everywhere
outside the derived windows — so these tests pin that property directly,
against windows produced by the real derivation rather than hand-written ones.

Why it earns its own test rather than trusting the arithmetic: the two sides are
computed by different code on different arrays, and if only one were cropped the
result would still be a plausible percentage. Every date's keep/drop decision
rides on it, and the ingest marker makes a wrongly-filtered year permanent.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
import zarr

from tessera_embeddings.ingest.live_windows import live_windows_for_mask, merge_bands, row_band_windows
from tessera_embeddings.ingest.s2_roi import _sum_over_windows

CHUNK = 4  # small stand-in for INGEST_CHUNK_SIZE; both APIs take it as a parameter


def _mask_store(tmp_path, mask: np.ndarray) -> str:
    path = str(tmp_path / "roi.zarr")
    z = zarr.open(path, mode="w", shape=mask.shape, chunks=(CHUNK, CHUNK), dtype="bool")
    z[:] = mask
    return path


def _windows_for(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Derive windows from an in-memory mask, as the ingest does from the store."""
    h, w = mask.shape
    rows, cols = -(-h // CHUNK), -(-w // CHUNK)
    live = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            live[r, c] = mask[r * CHUNK : (r + 1) * CHUNK, c * CHUNK : (c + 1) * CHUNK].any()
    # Row bands THEN the cost-model merge — the same two stages
    # live_windows_for_mask runs, so this helper keeps mirroring the real
    # derivation rather than one half of it.
    wins = merge_bands(row_band_windows(live, height=h, width=w, chunk_px=CHUNK), chunk_px=CHUNK)
    return [(x.y0, x.y1, x.x0, x.x1) for x in wins]


def _sum(mask: np.ndarray, windows) -> int:
    return int(_sum_over_windows(da.from_array(mask, chunks=(CHUNK, CHUNK)), windows).compute())


def test_window_total_equals_full_extent_total():
    """The property the cropped denominator rests on, on a 03S-like mask."""
    mask = np.zeros((12, 16), dtype=bool)
    mask[1, 2] = True
    mask[9, 13] = True
    mask[9, 14] = True
    windows = _windows_for(mask)
    # The window COUNT is a property of the grouping strategy and deliberately not
    # asserted — what the denominator rests on is that the windows between them
    # cover every live pixel exactly once, whatever shape they take.
    assert _sum(mask, windows) == int(mask.sum()) == 3


def test_row_band_spanning_a_gap_does_not_double_count():
    """A band spanning several chunks is summed once, not once per live chunk."""
    mask = np.zeros((4, 16), dtype=bool)
    mask[0, 0] = True  # chunk col 0
    mask[2, 13] = True  # chunk col 3 — the band spans cols 0..3
    windows = _windows_for(mask)
    assert len(windows) == 1
    assert _sum(mask, windows) == int(mask.sum()) == 2


def test_multiple_bands_are_disjoint():
    """Several windows must not overlap, or the cropped total double-counts.

    Priced so the merge does NOT collapse the rows into one window — with the
    production price it would, and then this test would silently stop testing
    anything about multiple bands.
    """
    mask = np.zeros((16, 8), dtype=bool)
    for row in range(4):  # one live pixel in every chunk-row
        mask[row * CHUNK + 1, 3] = True
    h, w = mask.shape
    live = np.zeros((-(-h // CHUNK), -(-w // CHUNK)), dtype=bool)
    for r in range(live.shape[0]):
        for c in range(live.shape[1]):
            live[r, c] = mask[r * CHUNK : (r + 1) * CHUNK, c * CHUNK : (c + 1) * CHUNK].any()
    bands = merge_bands(
        row_band_windows(live, height=h, width=w, chunk_px=CHUNK),
        chunk_px=CHUNK,
        window_cost_in_chunks=0,  # a free window -> minimum area -> no merging
    )
    windows = [(x.y0, x.y1, x.x0, x.x1) for x in bands]
    assert len(windows) == 4
    assert _sum(mask, windows) == int(mask.sum()) == 4


def test_all_ocean_mask_totals_zero():
    """No windows must yield 0, not a reduce over an empty list."""
    mask = np.zeros((8, 8), dtype=bool)
    assert _windows_for(mask) == []
    assert _sum(mask, []) == 0


def test_the_equivalence_is_not_vacuous():
    """A True pixel outside the windows WOULD be missed — so the test has teeth.

    Guards against the equivalence above passing for the wrong reason (e.g. a
    helper that quietly summed the whole array). Windows are derived from one
    mask and applied to another that has extra land outside them; the totals
    must then differ, which is precisely why the derivation must cover every
    live chunk.
    """
    derived_from = np.zeros((12, 16), dtype=bool)
    derived_from[1, 2] = True
    windows = _windows_for(derived_from)

    with_extra_land = derived_from.copy()
    with_extra_land[9, 13] = True  # a chunk-row no window covers
    assert _sum(with_extra_land, windows) == 1
    assert int(with_extra_land.sum()) == 2


def test_windows_from_the_store_match_the_in_memory_derivation(tmp_path):
    """The real entry point agrees with the helper these tests use."""
    mask = np.zeros((12, 16), dtype=bool)
    mask[1, 2] = True
    mask[9, 13] = True
    path = _mask_store(tmp_path, mask)
    from_store = [(w.y0, w.y1, w.x0, w.x1) for w in live_windows_for_mask(path, chunk_px=CHUNK)]
    assert from_store == _windows_for(mask)
    assert _sum(mask, from_store) == int(mask.sum())
