"""Per-date footprint narrowing: it must never lose imagery.

The run's live windows cover the ROI's land on every date; a single date's imagery
reaches only part of that. Narrowing to the part it reaches removes tasks whose
results are discarded — but only if the footprint is CONSERVATIVE. Too large merely
computes area that would have been thrown away; too small silently drops imagery from
a mosaic and nothing downstream notices.

So these tests are weighted almost entirely towards over-coverage: every pixel a
bounding box claims must end up inside the returned windows, and every path that
cannot be sure must fall back to the unnarrowed behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.geom import box

from tessera_embeddings.ingest.live_windows import (
    FOOTPRINT_PAD_CELLS,
    LiveWindow,
    footprint_grid,
    grid_from_windows,
    row_band_windows,
    windows_for_date,
)

CELL = 1024  # stand-in for the ingest's window granularity, small enough to reason about
ROWS, COLS = 6, 4
HEIGHT, WIDTH = ROWS * CELL, COLS * CELL

#: UTM 35N, 10 m pixels, north-up — the shape a campaign zone store has.
GEOBOX = GeoBox((HEIGHT, WIDTH), Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0), "EPSG:32635")


def _lonlat_bbox_for_pixels(y0: int, y1: int, x0: int, x1: int) -> tuple[float, float, float, float]:
    """The lon/lat bbox of a pixel rectangle — how a STAC item would describe it."""
    left, top = GEOBOX.transform * (x0, y0)
    right, bottom = GEOBOX.transform * (x1, y1)
    bb = box(min(left, right), min(top, bottom), max(left, right), max(top, bottom), GEOBOX.crs)
    out = bb.to_crs("EPSG:4326").boundingbox
    return (out.left, out.bottom, out.right, out.top)


def _cells(grid: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(grid), strict=True)}


# --- the load-bearing property: never under-cover ---------------------------------


@pytest.mark.parametrize(
    ("y0", "y1", "x0", "x1"),
    [
        (0, CELL, 0, CELL),  # a single cell at the origin
        (CELL, 2 * CELL, CELL, 2 * CELL),  # one interior cell
        (CELL // 3, CELL * 2 + 17, 0, CELL + 5),  # deliberately cell-unaligned
        (0, HEIGHT, 0, WIDTH),  # the whole grid
        (HEIGHT - 3, HEIGHT, WIDTH - 3, WIDTH),  # the far corner
    ],
)
def test_footprint_covers_every_pixel_it_claims(y0: int, y1: int, x0: int, x1: int) -> None:
    """Every cell overlapping the source rectangle must be marked."""
    bbox = _lonlat_bbox_for_pixels(y0, y1, x0, x1)
    grid = footprint_grid([bbox], GEOBOX, chunk_px=CELL)
    assert grid is not None
    for r in range(y0 // CELL, math.ceil(y1 / CELL)):
        for c in range(x0 // CELL, math.ceil(x1 / CELL)):
            assert grid[r, c], f"cell ({r},{c}) overlaps the footprint but was not marked"


def test_footprint_is_padded_beyond_the_strict_bounds() -> None:
    """The pad is what makes reprojection curvature harmless — assert it is applied."""
    bbox = _lonlat_bbox_for_pixels(2 * CELL, 3 * CELL, 2 * CELL, 3 * CELL)
    grid = footprint_grid([bbox], GEOBOX, chunk_px=CELL)
    assert grid is not None
    assert grid[2 - FOOTPRINT_PAD_CELLS, 2], "expected padding above the strict bounds"
    assert grid[2, 2 - FOOTPRINT_PAD_CELLS], "expected padding left of the strict bounds"


def test_windows_for_date_never_drops_a_covered_pixel() -> None:
    """The end-to-end invariant: narrowing keeps every cell the imagery reaches."""
    run = row_band_windows(np.ones((ROWS, COLS), dtype=bool), height=HEIGHT, width=WIDTH, chunk_px=CELL)
    bbox = _lonlat_bbox_for_pixels(CELL, 3 * CELL, 0, 2 * CELL)
    narrowed = windows_for_date(run, [bbox], GEOBOX, chunk_px=CELL)

    kept = grid_from_windows(narrowed, height=HEIGHT, width=WIDTH, chunk_px=CELL)
    for r in range(1, 3):
        for c in range(0, 2):
            assert kept[r, c], f"cell ({r},{c}) is inside the imagery but was narrowed away"


# --- it must actually narrow, or the change is pointless --------------------------


def test_narrowing_removes_windows_the_date_cannot_reach() -> None:
    run = row_band_windows(np.ones((ROWS, COLS), dtype=bool), height=HEIGHT, width=WIDTH, chunk_px=CELL)
    assert len(run) == ROWS
    # Imagery over the top two rows only.
    bbox = _lonlat_bbox_for_pixels(0, CELL, 0, WIDTH)
    narrowed = windows_for_date(run, [bbox], GEOBOX, chunk_px=CELL, merge=False)
    assert 0 < len(narrowed) < len(run)
    kept = grid_from_windows(narrowed, height=HEIGHT, width=WIDTH, chunk_px=CELL)
    assert not kept[ROWS - 1].any(), "the bottom row is unreachable today and should be gone"


def test_narrowing_is_bounded_by_the_run_windows() -> None:
    """A date cannot introduce area the run's own windows exclude (i.e. ocean)."""
    live = np.zeros((ROWS, COLS), dtype=bool)
    live[2, 1] = True  # one live cell in the whole ROI
    run = row_band_windows(live, height=HEIGHT, width=WIDTH, chunk_px=CELL)
    narrowed = windows_for_date(run, [_lonlat_bbox_for_pixels(0, HEIGHT, 0, WIDTH)], GEOBOX, chunk_px=CELL)
    kept = _cells(grid_from_windows(narrowed, height=HEIGHT, width=WIDTH, chunk_px=CELL))
    assert kept == {(2, 1)}


def test_date_reaching_no_live_cell_returns_empty() -> None:
    """Distinct from the None fallback: this date genuinely has nothing to write."""
    live = np.zeros((ROWS, COLS), dtype=bool)
    live[0, 0] = True
    run = row_band_windows(live, height=HEIGHT, width=WIDTH, chunk_px=CELL)
    far = _lonlat_bbox_for_pixels(HEIGHT - CELL, HEIGHT, WIDTH - CELL, WIDTH)
    assert windows_for_date(run, [far], GEOBOX, chunk_px=CELL) == []


# --- every uncertain path must fall back to "assume everything" -------------------


@pytest.mark.parametrize("bad", [None, (), (1.0, 2.0), ("a", "b", "c", "d"), (None, 1.0, 2.0, 3.0)])
def test_unusable_bbox_assumes_everything(bad: object) -> None:
    assert footprint_grid([bad], GEOBOX, chunk_px=CELL) is None  # type: ignore[list-item]


def test_unusable_geobox_assumes_everything() -> None:
    assert footprint_grid([_lonlat_bbox_for_pixels(0, CELL, 0, CELL)], object(), chunk_px=CELL) is None


def test_windows_unchanged_when_footprint_is_unknown() -> None:
    """The fallback must be the caller's PREVIOUS behaviour, not an empty write."""
    run = row_band_windows(np.ones((ROWS, COLS), dtype=bool), height=HEIGHT, width=WIDTH, chunk_px=CELL)
    assert windows_for_date(run, [None], GEOBOX, chunk_px=CELL) == run  # type: ignore[list-item]


def test_no_run_windows_stays_empty() -> None:
    assert windows_for_date([], [_lonlat_bbox_for_pixels(0, CELL, 0, CELL)], GEOBOX, chunk_px=CELL) == []


# --- antimeridian ------------------------------------------------------------------


def test_antimeridian_bbox_is_split_not_rejected() -> None:
    """A west > east bbox is the crossing convention; it must still yield a footprint.

    Reprojecting it unsplit would sweep the whole globe and mark everything, which is
    safe but pointless. The zones that snap past the antimeridian are exactly the ones
    where narrowing still has to work.
    """
    zone60 = GeoBox((HEIGHT, WIDTH), Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0), "EPSG:32660")
    grid = footprint_grid([(179.0, 35.0, -179.0, 36.0)], zone60, chunk_px=CELL)
    assert grid is not None  # a crossing bbox must not force the fallback


# --- grid_from_windows is the inverse of row_band_windows -------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_grid_round_trips_through_windows(seed: int) -> None:
    rng = np.random.default_rng(seed)
    live = rng.random((ROWS, COLS)) > 0.5
    windows = row_band_windows(live, height=HEIGHT, width=WIDTH, chunk_px=CELL)
    back = grid_from_windows(windows, height=HEIGHT, width=WIDTH, chunk_px=CELL)
    # Row bands span first..last live column, so the round trip can only ADD cells.
    assert (back | live == back).all(), "the round trip must not lose a live cell"


def test_grid_from_windows_marks_partial_trailing_cells() -> None:
    """A window clamped to a partial last chunk still owns that cell."""
    w = LiveWindow(y0=0, y1=HEIGHT - 1, x0=0, x1=WIDTH - 1)
    grid = grid_from_windows([w], height=HEIGHT, width=WIDTH, chunk_px=CELL)
    assert grid.all()
