"""ADR 043's precision gate: the cell arithmetic, the rule, and the FP8 exception check.

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


def _variant(fail=(), unresolved=(), fail_tile=(), trip=None, cells=None, tile=None, gc=None, env=True) -> dict:
    return {
        "verdict": "pass" if not (fail or unresolved or fail_tile or trip) and env else "fail",
        "envelope": {"passed": env},
        "fail_cells": list(fail),
        "unresolved": list(unresolved),
        "fail_tile": list(fail_tile),
        "trip": trip or {},
        "cells": cells or {},
        "tile": tile or {},
        "gc": gc or {},
    }


WW = "winter wheat|probe|ref->x|recall"
BOUNDS_KANSAS = {"cells": {WW: {"fp8": 0.82, "mixed": 0.85}}, "trip": {"pasture": {"fp8": 4.8e-3, "mixed": 4.8e-3}}}


def test_exception_clean_inside_and_outside() -> None:
    clean = _variant(cells={WW: [0.3, 0.1]}, gc={"pasture": 1e-3})
    assert pg.check_exception(clean, BOUNDS_KANSAS, "fp8").status == "clean"

    inside = _variant(fail=[WW], cells={WW: [0.8, 0.1]}, trip={"pasture": 4e-3}, gc={"pasture": 4e-3})
    assert pg.check_exception(inside, BOUNDS_KANSAS, "fp8").status == "inside"

    past = _variant(fail=[WW], cells={WW: [0.83, 0.1]}, gc={"pasture": 1e-3})
    assert pg.check_exception(past, BOUNDS_KANSAS, "fp8").status == "outside"
    assert pg.check_exception(past, BOUNDS_KANSAS, "mixed").status == "inside"  # each comparison its own bound

    new_cell = "corn|probe|x->ref|recall"
    unlisted = _variant(unresolved=[new_cell], cells={WW: [0.1, 0.1], new_cell: [0.6, 0.3]}, gc={"pasture": 1e-3})
    res = pg.check_exception(unlisted, BOUNDS_KANSAS, "fp8")
    assert res.status == "outside"
    assert "not in the exception" in res.reasons[0]

    new_trip = _variant(cells={WW: [0.1, 0.1]}, trip={"sorghum": 3e-3}, gc={"pasture": 1e-3, "sorghum": 3e-3})
    assert pg.check_exception(new_trip, BOUNDS_KANSAS, "fp8").status == "outside"


def test_a_listed_cell_that_worsens_past_its_bound_is_outside_even_unflagged() -> None:
    # Past its bound but not flagged, e.g. because a recalibration left it ungated under z_fam.
    v = _variant(cells={WW: [0.9, 0.5]}, gc={"pasture": 1e-3})
    assert pg.check_exception(v, BOUNDS_KANSAS, "fp8").status == "outside"


def test_a_comparison_with_no_record_uses_the_largest_recorded_bound() -> None:
    cell = "sorghum|probe|ref->x|recall"
    bounds = {"cells": {cell: {"mixed": 0.81}}}
    v = _variant(unresolved=[cell], cells={cell: [0.7, 0.3]})
    assert pg.check_exception(v, bounds, "fp8").status == "inside"


def test_tile_with_nothing_listed_holds_every_flag_outside() -> None:
    v = _variant(fail_tile=["probe x->ref"], tile={"probe x->ref": [0.21, 0.05]})
    assert pg.check_exception(v, None, "fp8").status == "outside"


def test_a_failed_envelope_is_outside_any_exception_and_does_not_pass() -> None:
    v = _variant(cells={WW: [0.3, 0.1]}, gc={"pasture": 1e-3}, env=False)
    v["exception"] = {"status": pg.check_exception(v, BOUNDS_KANSAS, "fp8").status}
    assert v["exception"]["status"] == "outside"
    assert not pg.passes(v)


def test_passes_outright_or_inside_an_exception() -> None:
    assert pg.passes(_variant())
    held = _variant(fail=[WW], cells={WW: [0.8, 0.1]}, gc={"pasture": 1e-3})
    assert not pg.passes(held)  # not held to anything
    held["exception"] = {"status": pg.check_exception(held, BOUNDS_KANSAS, "fp8").status}
    assert pg.passes(held)
