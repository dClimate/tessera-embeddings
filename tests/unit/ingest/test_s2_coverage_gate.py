"""The S2 per-date coverage gate (pure, offline).

The gate decides which solar days are ingested at all, and the ingest marker makes
that decision permanent for a (zone, year) — a wrongly-dropped date is not
retried and a wrongly-kept one pollutes the mosaic. It also has to keep BOTH sides
of its ratio cropped to the same windows: cropping only the numerator still yields
a plausible-looking percentage, which is the failure these tests exist to catch.

The numerator is the SCL validity reduce in ``_coverage_from_scl``; the denominator is the ROI
pixel total from ``_sum_over_windows``. Cropping both is sound only because the ROI mask is
False everywhere outside the derived windows, so the denominator tests pin that property
against windows produced by the real derivation.
"""

from __future__ import annotations

from types import SimpleNamespace

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from tessera_embeddings.config.satellites import S2_SCL_INVALID_CLASSES
from tessera_embeddings.ingest.live_windows import live_windows_for_mask, merge_bands, row_band_windows
from tessera_embeddings.ingest.s2_roi import _coverage_from_scl, _sum_over_windows

CHUNK = 4
VALID_CLASS = next(c for c in range(12) if c not in S2_SCL_INVALID_CLASSES)
INVALID_CLASS = sorted(S2_SCL_INVALID_CLASSES)[0]


class _SyncClient:
    """Stands in for the distributed client: computes on the calling thread.

    The gate submits its windowed reduce through ``client.compute`` rather than a
    bare ``.compute()``, because it may run off the driver thread and must reach the
    caller's scheduler rather than whichever one dask resolves by default. These
    tests are about the gate's arithmetic, so the stub computes locally and hands
    back the future's ``.result()`` interface.
    """

    @staticmethod
    def compute(collection):
        return SimpleNamespace(result=collection.compute)


SYNC_CLIENT = _SyncClient()


def _scl(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        da.from_array(values.astype("uint8"), chunks=(CHUNK, CHUNK)),
        dims=("northing", "easting"),
    )


def _mask(values: np.ndarray) -> da.Array:
    return da.from_array(values.astype(bool), chunks=(CHUNK, CHUNK))


def test_passes_when_validity_clears_the_threshold():
    scl = _scl(np.full((CHUNK, CHUNK), VALID_CLASS))
    roi = _mask(np.ones((CHUNK, CHUNK)))
    passes, any_valid = _coverage_from_scl(scl, roi, CHUNK * CHUNK, 50.0, SYNC_CLIENT, windows=[(0, CHUNK, 0, CHUNK)])
    assert passes
    assert any_valid is not None


def test_fails_when_validity_is_below_the_threshold():
    scl = _scl(np.full((CHUNK, CHUNK), INVALID_CLASS))
    roi = _mask(np.ones((CHUNK, CHUNK)))
    passes, any_valid = _coverage_from_scl(scl, roi, CHUNK * CHUNK, 50.0, SYNC_CLIENT, windows=[(0, CHUNK, 0, CHUNK)])
    assert not passes
    assert any_valid is None, "a failed date must not hand back a validity mask the caller could use"


def test_only_the_declared_invalid_classes_are_excluded():
    """The gate's meaning rides entirely on this set — an off-by-one silently
    reclassifies every pixel in the campaign.
    """
    values = np.full((CHUNK, CHUNK), VALID_CLASS)
    values[0, :] = INVALID_CLASS
    scl = _scl(values)
    roi = _mask(np.ones((CHUNK, CHUNK)))
    _, any_valid = _coverage_from_scl(scl, roi, CHUNK * CHUNK, 0.0, SYNC_CLIENT, windows=[(0, CHUNK, 0, CHUNK)])
    got = np.asarray(any_valid.compute())
    assert not got[0, :].any()
    assert got[1:, :].all()


def test_cropped_count_equals_the_full_extent_count():
    """Both sides of the ratio must be cropped together.

    The ROI mask is False outside every window, so summing over the windows must
    give the same numerator as summing the whole grid. If only one side were
    cropped the percentage would still look reasonable — hence asserting the
    equality rather than the percentage.
    """
    rng = np.random.default_rng(20260725)
    values = np.where(rng.random((4 * CHUNK, 4 * CHUNK)) < 0.5, VALID_CLASS, INVALID_CLASS)
    roi_arr = np.zeros((4 * CHUNK, 4 * CHUNK), dtype=bool)
    roi_arr[0:CHUNK, 0:CHUNK] = True  # live only in one chunk...
    roi_arr[2 * CHUNK : 3 * CHUNK, CHUNK : 2 * CHUNK] = True  # ...and one other
    windows = [(0, CHUNK, 0, CHUNK), (2 * CHUNK, 3 * CHUNK, CHUNK, 2 * CHUNK)]
    total = int(roi_arr.sum())

    valid_full = int((np.isin(values, list(S2_SCL_INVALID_CLASSES), invert=True) & roi_arr).sum())
    # Threshold set just at the true percentage, so a miscounted numerator flips the verdict.
    threshold = 100.0 * valid_full / total
    passes, _ = _coverage_from_scl(_scl(values), _mask(roi_arr), total, threshold, SYNC_CLIENT, windows=windows)
    assert passes, "the windowed numerator under-counted relative to the full extent"
    passes_above, _ = _coverage_from_scl(
        _scl(values), _mask(roi_arr), total, threshold + 0.01, SYNC_CLIENT, windows=windows
    )
    assert not passes_above, "the windowed numerator over-counted relative to the full extent"


def test_any_valid_stays_lazy_under_windows():
    """Materialising the validity mask would undo the cropping it exists to enable —
    the whole grid is orders of magnitude larger than the windows.
    """
    scl = _scl(np.full((4 * CHUNK, 4 * CHUNK), VALID_CLASS))
    roi = _mask(np.ones((4 * CHUNK, 4 * CHUNK)))
    _, any_valid = _coverage_from_scl(scl, roi, 16 * CHUNK * CHUNK, 0.0, SYNC_CLIENT, windows=[(0, CHUNK, 0, CHUNK)])
    assert hasattr(any_valid.data, "dask"), "any_valid was computed rather than left lazy"


def test_all_valid_passes_at_a_100_percent_threshold():
    """The top boundary is inclusive; the 0% boundary is covered above."""
    threshold = 100.0
    scl = _scl(np.full((CHUNK, CHUNK), VALID_CLASS))
    roi = _mask(np.ones((CHUNK, CHUNK)))
    passes, _ = _coverage_from_scl(scl, roi, CHUNK * CHUNK, threshold, SYNC_CLIENT, windows=[(0, CHUNK, 0, CHUNK)])
    assert passes


def test_an_roi_with_no_live_pixel_fails_the_date_instead_of_dividing_by_zero():
    """An all-ocean ROI yields no live window, so the coverage denominator is zero.

    ``_sum_over_windows`` returns 0 for an empty window list — documented behaviour, not
    an accident — and the ratio below then raised ``ZeroDivisionError`` from inside the
    per-date gate. The campaign never sees it (``zone_has_live_tiles`` screens all-ocean
    zones out before ingest), but the public ROI ingest has no such preflight, so an ROI
    over open water crashed rather than reporting that it had nothing to ingest.
    """
    scl = _scl(np.full((CHUNK, CHUNK), VALID_CLASS))
    roi = _mask(np.zeros((CHUNK, CHUNK)))

    passes, any_valid = _coverage_from_scl(scl, roi, 0, 0.0, SYNC_CLIENT, windows=[])

    assert not passes
    assert any_valid is None


# --- the denominator: window totals equal full-extent totals ---------------------------


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


def test_all_ocean_mask_totals_zero():
    """No windows must yield 0, not a reduce over an empty list."""
    mask = np.zeros((8, 8), dtype=bool)
    assert _windows_for(mask) == []
    assert _sum(mask, []) == 0


def test_windows_from_the_store_match_the_in_memory_derivation(tmp_path):
    """The real entry point agrees with the helper these tests use."""
    mask = np.zeros((12, 16), dtype=bool)
    mask[1, 2] = True
    mask[9, 13] = True
    path = _mask_store(tmp_path, mask)
    from_store = [(w.y0, w.y1, w.x0, w.x1) for w in live_windows_for_mask(path, chunk_px=CHUNK)]
    assert from_store == _windows_for(mask)
    assert _sum(mask, from_store) == int(mask.sum())
