"""ADR 043's precision gate: the cell arithmetic and the rule.

The rule's full behaviour was checked once against the record: re-applied to the recorded per-variant
predictions it reproduces every verdict, z_fam, gated cell and tripwire on all six tiles. These tests pin
the pieces a refactor could break without a GPU run to notice.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tessera_embeddings.profiling.inference import precision_gate as pg


def _grid_tile(side: int = 480) -> pg.Tile:
    """A side x side lattice with four classes in vertical stripes, every pixel labelled."""
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    raw = np.array([10, 30, 40, 50])[(xx * 4) // side]
    return pg.make_tile(raw.ravel(), "worldcover", yy.ravel() * 10.0, xx.ravel() * 10.0)


def test_make_tile_blocks_the_lattice_in_40_pixel_squares() -> None:
    tile = _grid_tile()
    assert tile.n_blocks == 144
    assert tile.names[:4] == ["trees", "grass", "cropland", "built"]
    assert tile.reported == [0, 1, 2, 3]


def test_jackknife_of_identical_blocks_has_no_spread() -> None:
    ratio, se = pg.jackknife_ratio(np.full(10, 2.0), np.full(10, 4.0))
    assert ratio == pytest.approx(0.5)
    assert se == pytest.approx(0.0)


def test_recall_and_tile_loss_from_flipped_predictions() -> None:
    tile = _grid_tile()
    a = tile.lab.copy()
    b = a.copy()
    trees = np.flatnonzero(tile.lab == 0)
    b[trees[: trees.size // 10]] = 1  # 10% of trees called grass
    cells = pg.class_cells(tile, a, b)
    assert cells[0][0] == pytest.approx(10.0)  # trees recall loses 10 points
    assert cells[1][0] == pytest.approx(0.0)
    assert cells[1][2] > 0  # grass precision loses
    loss, se = pg.tile_loss(tile, a, b)
    assert loss == pytest.approx(2.5)
    assert se > 0


def test_refit_spread_is_the_sd_of_each_redraw_s_move_from_the_base_fit() -> None:
    tile = _grid_tile()
    base = {"p_rr": tile.lab.copy(), "k_rr": tile.lab.copy()}
    trees = np.flatnonzero(tile.lab == 0)
    refits = []
    for frac in (0.0, 0.1):  # trees recall moves 0 then 10 points
        r = tile.lab.copy()
        r[trees[: int(trees.size * frac)]] = 1
        refits.append({"p_rr": r, "k_rr": tile.lab.copy()})
    spread = pg.refit_spread(tile, base, refits)
    assert spread["trees|probe|recall"] == pytest.approx(np.std([0.0, 10.0], ddof=1))
    assert spread["trees|knn|recall"] == 0.0


def test_gap_closure_at_delta_interpolates_and_extrapolates_in_proportion() -> None:
    assert pg._gap_closure_at_delta([(0.2, 1.0), (1.0, 3.0)]) == pytest.approx(1.75)
    assert pg._gap_closure_at_delta([(1.0, 2.0), (2.0, 4.0)]) == pytest.approx(1.0)  # δ below every dose
    assert pg._gap_closure_at_delta([(0.1, 1.0), (0.25, 2.0)]) == pytest.approx(4.0)  # δ above every dose


def _score(cells: dict[pg.CellKey, tuple[float, float]], gc: float = 0.0) -> pg.Score:
    return pg.Score(cells=cells, tile={("probe", "ref->x"): (0.0, 0.05)}, gap_closure={0: gc, 1: gc})


GATED: pg.CellKey = (0, "probe", "ref->x", "recall")
UNGATED: pg.CellKey = (1, "probe", "ref->x", "recall")


def _noise() -> dict[str, pg.Score]:
    rng = np.random.default_rng(0)
    return {f"noise_{d}": _score({GATED: (float(rng.normal(0, 0.05)), 0.05), UNGATED: (0.0, 0.3)}) for d in range(20)}


def test_calibration_gates_on_noise_se_and_sorts_candidates() -> None:
    noise = _noise()
    cal = pg.calibrate(noise)
    assert cal.gated == frozenset({GATED})
    assert 0 < cal.z_fam < 3

    fails = _score({GATED: (1.0, 0.1), UNGATED: (0.0, 0.3)})
    own_se = _score({GATED: (0.6, 0.6), UNGATED: (0.0, 0.3)})  # past δ, but its own SE keeps z under z_fam
    ungated = _score({GATED: (0.0, 0.05), UNGATED: (0.6, 0.1)})  # past δ and z_fam, in a cell too noisy to gate
    harmless = _score({GATED: (0.4, 0.05), UNGATED: (0.0, 0.3)})
    assert pg.verdict(fails, cal) == "fail"
    assert pg.verdict(own_se, cal) == "unresolved"
    assert pg.verdict(ungated, cal) == "unresolved"
    assert pg.verdict(harmless, cal) == "pass"
    assert pg.verdict(harmless, cal, envelope_passed=False) == "fail"
    tripping = _score({GATED: (0.4, 0.05), UNGATED: (0.0, 0.3)}, gc=0.02)
    assert pg.verdict(tripping, dataclasses.replace(cal, tripwire={0: 0.01})) == "unresolved"  # a trip needs a decision


def test_a_tile_loss_past_the_bound_fails() -> None:
    cal = pg.calibrate(_noise())
    s = pg.Score(cells={GATED: (0.0, 0.05)}, tile={("probe", "x->ref"): (0.21, 0.05)}, gap_closure={0: 0.0})
    assert pg.verdict(s, cal) == "fail"
