"""Live-window derivation from an ROI mask (pure, offline).

Pins the properties the write path builds on: windows are chunk-aligned and
mutually chunk-disjoint, clamped to the extent, and together cover every live
pixel of the mask — because a window the derivation misses is land that silently
never gets ingested.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from tessera_embeddings.ingest.live_windows import (
    LiveWindow,
    live_chunk_grid,
    live_windows_for_mask,
    row_band_windows,
)

CHUNK = 4  # small stand-in for INGEST_CHUNK_SIZE; the code takes it as a parameter


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

    windows = live_windows_for_mask(path, chunk_px=CHUNK)
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
