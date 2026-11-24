"""ADR 043's gate for a precision change to the forward pass, applied to saved per-pixel predictions.

The gate asks whether a classifier fitted on one side of a change and applied to the other loses
accuracy, per tile and per class, beyond what matched noise does. The predictions come from
``scripts/precision_gate/calibrate.py``, one ``.npz`` per variant. This module turns them into
cells, calibrates the thresholds from the known-good and known-bad variants, and gives each
variant a verdict. It is pure numpy so that the rule can be tested without a GPU or scikit-learn.

Terms (``context_docs/decisions/043-...md``, decision 1):

- A *cell* is one class x consumer (probe, kNN-20) x direction x measure (recall, precision).
  ``ref->x`` is fitted on the reference and applied to the changed side; ``x->ref`` the reverse.
- Loss is in points, positive = worse. SE is a delete-one-block jackknife over 40 x 40 px blocks.
- A cell is *gated* when its median SE over the matched-noise draws is at most δ/4.
- ``z_fam`` is the 95th percentile over the noise draws of the largest loss / SE among gated cells.
- A gated cell past δ with loss / SE above ``z_fam`` fails; past δ with loss / SE at or under it,
  or ungated with loss / SE above it, it is unresolved. A tile loss past 0.2 fails.
- The tripwire: a class's gap closure past half its own control's gap closure at the dose
  costing δ. A trip needs a decision, so it leaves the variant unresolved.
- A comparison whose ADR 012 envelope fails, fails.
- A named exception (decision 2) accepts a failing change's recorded losses: :func:`exception_bounds` sets
  its bounds, :func:`outside_exception` holds a later run to them.

Variant names follow ``calibrate.py``: ``noise_*`` are the matched-noise draws, ``ctrl_<i>_<j>_<cos>``
the class-directed controls, ``ctrl2_*`` the large-to-small control, ``refit_*`` training-sample
redraws.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from tessera_embeddings.profiling.inference.compare_outputs import (
    XCFG_MAX_SCALE_REL_DRIFT,
    XCFG_MIN_COSINE,
    XCFG_MIN_WITHIN_ONE_FRAC,
)

DELTA = 0.5
"""Points a cell may lose before it can fail or be unresolved (decision 1.3)."""
TILE_BOUND = 0.2
"""Points a whole tile may lose, per consumer and direction (decision 1.2)."""
TRIP_ALLOWANCE = 1.1
"""A named exception's bound on a tripped class, as a multiple of its recorded gap closure (decision 2)."""
CONTROL_COSINE = 0.9999
"""The control dose each gated class's control should be rejected at (decision 1.4)."""
MIN_REPORTED_PX = 200
"""Classes with fewer labelled pixels than this are not scored at all."""
BLOCK_PX = 40
SE_FLOOR = 1e-3
"""Smallest SE a loss / SE ratio divides by, so a cell with no spread does not give infinity."""

LABEL_GROUPS: dict[str, dict[str, tuple[int, ...]]] = {
    "cdl": {
        "corn": (1,),
        "soybeans": (5,),
        "developed": (121, 122, 123, 124),
        "forest": (141,),
        "pasture": (176,),
        "winter wheat": (24,),
        "sorghum": (4,),
    },
    "worldcover": {
        "trees": (10,),
        "grass": (30,),
        "cropland": (40,),
        "built": (50,),
        "water": (80,),
        "wetland": (90,),
    },
}
"""Gate classes per label source: USDA CDL 2024 codes, ESA WorldCover 2021 codes."""

CONSUMER_PAIRS: dict[tuple[str, str], tuple[str, str]] = {
    ("probe", "ref->x"): ("p_rr", "p_rx"),
    ("probe", "x->ref"): ("p_xx", "p_xr"),
    ("knn", "ref->x"): ("k_rr", "k_rx"),
    ("knn", "x->ref"): ("k_xx", "k_xr"),
}
"""(consumer, direction) -> (prediction before the change, prediction after it)."""

CellKey = tuple[int, str, str, str]
"""(class index, consumer, direction, measure)."""


def class_labels(raw: np.ndarray, label_key: str) -> tuple[np.ndarray, list[str]]:
    """Map raw label codes to gate class indices (-1 = not a gate class), and the class names."""
    groups = LABEL_GROUPS[label_key]
    lab = np.full(raw.shape, -1, dtype=np.int64)
    for i, codes in enumerate(groups.values()):
        lab[np.isin(raw, codes)] = i
    return lab, list(groups)


@dataclass(frozen=True)
class Tile:
    """A tile's labels and jackknife blocks."""

    names: list[str]
    lab: np.ndarray
    blk: np.ndarray
    n_blocks: int

    @property
    def ok(self) -> np.ndarray:
        """Labelled pixels."""
        return self.lab >= 0

    @property
    def counts(self) -> np.ndarray:
        """Labelled pixels per class."""
        return np.bincount(self.lab[self.ok], minlength=len(self.names))

    @property
    def reported(self) -> list[int]:
        """Classes large enough to score."""
        return [i for i, c in enumerate(self.counts) if c >= MIN_REPORTED_PX]

    def key_name(self, key: CellKey) -> str:
        """``"class|consumer|direction|measure"``, the form reports use."""
        return "|".join([self.names[key[0]], *key[1:]])


def make_tile(raw: np.ndarray, label_key: str, y: np.ndarray, x: np.ndarray) -> Tile:
    """Build a tile from its labels and pixel coordinates; blocks are 40 x 40 lattice pixels."""
    lab, names = class_labels(raw, label_key)
    yi = np.searchsorted(np.unique(y), y)
    xi = np.searchsorted(np.unique(x), x)
    cols = int(xi.max()) // BLOCK_PX + 1
    blk = (yi // BLOCK_PX) * cols + xi // BLOCK_PX
    return Tile(names=names, lab=lab, blk=blk, n_blocks=int(blk.max()) + 1)


def _block_sums(tile: Tile, mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    w = None if weights is None else weights[mask]
    return np.bincount(tile.blk[mask], w, tile.n_blocks).astype(float)


def _jackknife_se(loo: np.ndarray) -> float:
    n = loo.size
    return float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum()))


def jackknife_ratio(num_b: np.ndarray, den_b: np.ndarray) -> tuple[float, float]:
    """``sum(num) / sum(den)`` and its delete-one-block jackknife SE, from per-block sums."""
    n, d = num_b.sum(), den_b.sum()
    loo = (n - num_b) / np.maximum(d - den_b, 1e-12)
    return float(n / max(d, 1e-12)), _jackknife_se(loo)


def _precision_loss(tile: Tile, i: int, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Precision of class ``i`` under ``a`` minus under ``b``, jackknifed as one difference."""
    m = tile.lab == i
    ta, pa = _block_sums(tile, tile.ok & (a == i) & m), _block_sums(tile, tile.ok & (a == i))
    tb, pb = _block_sums(tile, tile.ok & (b == i) & m), _block_sums(tile, tile.ok & (b == i))
    loo = (ta.sum() - ta) / np.maximum(pa.sum() - pa, 1) - (tb.sum() - tb) / np.maximum(pb.sum() - pb, 1)
    loss = ta.sum() / max(pa.sum(), 1) - tb.sum() / max(pb.sum(), 1)
    return float(loss), _jackknife_se(loo)


def class_cells(tile: Tile, a: np.ndarray, b: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
    """Per reported class: (recall loss, SE, precision loss, SE) in points, from predictions ``a`` then ``b``."""
    out = {}
    for i in tile.reported:
        m = tile.lab == i
        recall_drop = (a == i).astype(float) - (b == i)
        rl, rse = jackknife_ratio(_block_sums(tile, m, recall_drop), _block_sums(tile, m))
        pl, pse = _precision_loss(tile, i, a, b)
        out[i] = (100 * rl, 100 * rse, 100 * pl, 100 * pse)
    return out


def tile_loss(tile: Tile, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Whole-tile accuracy loss in points from ``a`` to ``b``, with its jackknife SE."""
    ok = tile.ok
    drop = (a == tile.lab).astype(float) - (b == tile.lab)
    loss, se = jackknife_ratio(_block_sums(tile, ok, drop), _block_sums(tile, ok))
    return 100 * loss, 100 * se


@dataclass(frozen=True)
class Score:
    """One variant's losses: per cell and per tile as (loss, SE), and gap closure per class."""

    cells: dict[CellKey, tuple[float, float]]
    tile: dict[tuple[str, str], tuple[float, float]]
    gap_closure: dict[int, float]


def score_variant(tile: Tile, preds: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray]) -> Score:
    """Score one variant's predictions; ``p_rr`` / ``k_rr`` come from ``base`` unless the variant carries its own."""
    cells: dict[CellKey, tuple[float, float]] = {}
    tiles: dict[tuple[str, str], tuple[float, float]] = {}
    for (consumer, direction), (ka, kb) in CONSUMER_PAIRS.items():
        a = preds[ka] if ka in preds else base[ka]
        b = preds[kb]
        tiles[(consumer, direction)] = tile_loss(tile, a, b)
        for i, (rl, rse, pl, pse) in class_cells(tile, a, b).items():
            cells[(i, consumer, direction, "recall")] = (rl, rse)
            cells[(i, consumer, direction, "precision")] = (pl, pse)
    gc = preds["gc_cls"]
    rep = tile.reported
    closure = {i: float(max(gc[i, j] for j in rep if j != i)) for i in rep}
    return Score(cells=cells, tile=tiles, gap_closure=closure)


def _z(loss: float, se: float) -> float:
    return loss / max(se, SE_FLOOR)


@dataclass(frozen=True)
class Calibration:
    """The thresholds a tile's controls set: gated cells, ``z_fam`` and the tripwire per class."""

    cell_se: dict[CellKey, float]
    gated: frozenset[CellKey]
    z_fam: float
    z_draws: list[float]
    tripwire: dict[int, float] = field(default_factory=dict)

    @property
    def gated_classes(self) -> list[int]:
        """Classes with at least one gated cell."""
        return sorted({k[0] for k in self.gated})


def _worst_gated_z(score: Score, gated: frozenset[CellKey]) -> float:
    return max((_z(max(loss, 0.0), se) for k, (loss, se) in score.cells.items() if k in gated), default=0.0)


def _worst_gated_loss(score: Score, gated: frozenset[CellKey]) -> float:
    return max((loss for k, (loss, _) in score.cells.items() if k in gated), default=0.0)


def controls(scores: Mapping[str, Score]) -> dict[int, dict[float, str]]:
    """Class-directed controls by class and dose: ``ctrl_<i>_<j>_<cos>`` -> {i: {cos: name}}."""
    out: dict[int, dict[float, str]] = {}
    for name in scores:
        if name.startswith("ctrl_"):
            _, i, _, dose = name.split("_")
            out.setdefault(int(i), {})[float(dose)] = name
    return out


def _gap_closure_at_delta(points: Sequence[tuple[float, float]]) -> float | None:
    """Gap closure at the dose that costs δ, from (worst gated loss, gap closure) per dose, sorted by loss.

    Linear between the doses; scaled in proportion from the nearest dose where δ lies outside them. None when
    no dose costs anything, since no dose then establishes what costs δ.
    """
    losses = np.array([p[0] for p in points])
    gaps = np.array([p[1] for p in points])
    if losses[-1] <= 0:
        return None
    if losses[0] <= DELTA <= losses[-1]:
        return float(np.interp(DELTA, losses, gaps))
    nearest = -1 if losses[-1] < DELTA else 0
    return float(gaps[nearest] * DELTA / losses[nearest])


def calibrate(scores: Mapping[str, Score]) -> Calibration:
    """Set the tile's thresholds from its matched-noise draws and controls, never from a candidate."""
    noise = [s for name, s in scores.items() if name.startswith("noise")]
    if not noise:
        raise ValueError("no noise_* variants: the gate cannot be calibrated")
    keys = list(noise[0].cells)
    cell_se = {k: float(np.median([s.cells[k][1] for s in noise])) for k in keys}
    gated = frozenset(k for k in keys if cell_se[k] <= DELTA / 4)
    z_draws = [_worst_gated_z(s, gated) for s in noise]
    trip = {}
    for i, doses in controls(scores).items():
        points = sorted((_worst_gated_loss(scores[v], gated), scores[v].gap_closure[i]) for v in doses.values())
        if (at_delta := _gap_closure_at_delta(points)) is not None:
            trip[i] = at_delta / 2
    return Calibration(
        cell_se=cell_se, gated=gated, z_fam=float(np.percentile(z_draws, 95)), z_draws=z_draws, tripwire=trip
    )


def failed_cells(score: Score, cal: Calibration) -> list[CellKey]:
    """Gated cells past δ with loss / SE above ``z_fam``."""
    return [k for k, (loss, se) in score.cells.items() if k in cal.gated and loss > DELTA and _z(loss, se) > cal.z_fam]


def failed_tile(score: Score) -> list[tuple[str, str]]:
    """(consumer, direction) pairs whose tile loss exceeds the tile bound."""
    return [k for k, (loss, _) in score.tile.items() if loss > TILE_BOUND]


def unresolved_cells(score: Score, cal: Calibration) -> list[CellKey]:
    """Cells past δ the gate can neither fail nor call noise, largest loss first."""
    out = [
        (loss, k)
        for k, (loss, se) in score.cells.items()
        if loss > DELTA and (k in cal.gated) != (_z(loss, se) > cal.z_fam)
    ]
    return [k for _, k in sorted(out, reverse=True)]


def tripped(score: Score, cal: Calibration) -> list[int]:
    """Classes whose gap closure exceeds their tripwire."""
    return [i for i, t in cal.tripwire.items() if score.gap_closure[i] > t]


@dataclass(frozen=True)
class Validity:
    """Decision 1.4's lists: what the gate can and cannot see on this tile."""

    flag_only: list[int]
    uncovered: list[int]
    control_passes: list[int]
    unvalidated: list[int]


def validity(scores: Mapping[str, Score], cal: Calibration) -> Validity:
    """Classify each gated class by how the gate treats its own control at :data:`CONTROL_COSINE`."""
    ctrl = controls(scores)
    flag_only, uncovered, passes, unvalidated = [], [], [], []
    for i in cal.gated_classes:
        name = ctrl.get(i, {}).get(CONTROL_COSINE)
        if name is None:
            unvalidated.append(i)
            continue
        s = scores[name]
        outcome = verdict(s, cal)  # the lists restate the gate's own verdict on the control
        if outcome == "fail":
            continue
        if outcome == "unresolved":
            flag_only.append(i)
        elif max(loss for loss, _ in s.cells.values()) > DELTA:
            uncovered.append(i)
        else:
            passes.append(i)
    return Validity(flag_only=flag_only, uncovered=uncovered, control_passes=passes, unvalidated=unvalidated)


def verdict(score: Score, cal: Calibration, envelope_passed: bool = True) -> str:
    """``fail``, ``unresolved`` or ``pass``: decision 1's envelope, tile and class losses, and tripwire together.

    A trip asks for a decision, as an unresolved cell does, so it leaves the variant unresolved.
    """
    if not envelope_passed or failed_cells(score, cal) or failed_tile(score):
        return "fail"
    return "unresolved" if unresolved_cells(score, cal) or tripped(score, cal) else "pass"


def is_comparison(name: str) -> bool:
    """An embedded comparison (two cards, the candidate, a mixed store), as opposed to a noise draw or control."""
    return not name.startswith(("noise", "ctrl", "refit"))


def _variant_report(tile: Tile, score: Score, cal: Calibration, env: dict | None) -> dict:
    return {
        "verdict": verdict(score, cal, env is None or env["passed"]),
        "envelope": env,
        "fail_cells": [tile.key_name(k) for k in failed_cells(score, cal)],
        "fail_tile": [f"{c} {d}" for c, d in failed_tile(score)],
        "unresolved": [tile.key_name(k) for k in unresolved_cells(score, cal)],
        "trip": {tile.names[i]: score.gap_closure[i] for i in tripped(score, cal)},
        "tile": {f"{c} {d}": list(v) for (c, d), v in score.tile.items()},
        "cells": {tile.key_name(k): list(v) for k, v in score.cells.items()},
        "gc": {tile.names[i]: g for i, g in score.gap_closure.items()},
    }


def refit_spread(tile: Tile, base: Mapping[str, np.ndarray], refits: Sequence[Mapping[str, np.ndarray]]) -> dict:
    """Per class, consumer and measure: the SD in points, over training-sample redraws, of the move from the base fit.

    Reported beside the verdicts, not gated: it is how far a loss can move with the training sample alone.
    """
    moves: dict[str, list[float]] = {}
    for consumer, key in (("probe", "p_rr"), ("knn", "k_rr")):
        for r in refits:
            for i, (rl, _, pl, _) in class_cells(tile, base[key], r[key]).items():
                moves.setdefault(f"{tile.names[i]}|{consumer}|recall", []).append(rl)
                moves.setdefault(f"{tile.names[i]}|{consumer}|precision", []).append(pl)
    return {k: float(np.std(v, ddof=1)) for k, v in moves.items() if len(v) > 1}


def evaluate(tile: Tile, scores: Mapping[str, Score], envelopes: Mapping[str, dict]) -> dict:
    """The tile's calibration, validity lists and every variant's verdict, as a JSON-ready dict.

    ``envelopes`` must hold :func:`envelope` for every embedded comparison; noise draws and controls have none.
    """
    cal = calibrate(scores)
    val = validity(scores, cal)

    def names(idx: list[int]) -> list[str]:
        return [tile.names[i] for i in idx]

    return {
        "classes": {tile.names[i]: int(tile.counts[i]) for i in tile.reported},
        "z_fam": cal.z_fam,
        "z_draws": cal.z_draws,
        "gated_cells": sorted(tile.key_name(k) for k in cal.gated),
        "cell_se": {tile.key_name(k): v for k, v in cal.cell_se.items()},
        "tripwire": {tile.names[i]: t for i, t in cal.tripwire.items()},
        "flag_only": names(val.flag_only),
        "uncovered": names(val.uncovered),
        "control_passes": names(val.control_passes),
        "unvalidated": names(val.unvalidated),
        "variants": {
            v: _variant_report(tile, s, cal, envelopes[v] if is_comparison(v) else None) for v, s in scores.items()
        },
    }


def envelope(ref_q: np.ndarray, ref_s: np.ndarray, test_q: np.ndarray, test_s: np.ndarray) -> dict:
    """ADR 012's cross-config envelope between two int8 embeddings with per-pixel scales."""
    dq = np.abs(ref_q.astype(np.int16) - test_q.astype(np.int16))
    r, t = ref_q * ref_s[:, None].astype(np.float32), test_q * test_s[:, None].astype(np.float32)
    cos = (r * t).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    out = {
        "within_one": float((dq <= 1).mean()),
        "max_delta": int(dq.max()),
        "scale_drift": float(np.abs(test_s / ref_s - 1).max()),
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
    }
    out["passed"] = (
        out["within_one"] >= XCFG_MIN_WITHIN_ONE_FRAC
        and out["scale_drift"] <= XCFG_MAX_SCALE_REL_DRIFT
        and out["cos_min"] >= XCFG_MIN_COSINE
    )
    return out


def coherence(ref: np.ndarray, test: np.ndarray) -> float:
    """Length of the mean unit-vector error over the mean error length: ~0.02 for noise, 1 for a shared shift."""
    e = test / np.linalg.norm(test, axis=1, keepdims=True) - ref / np.linalg.norm(ref, axis=1, keepdims=True)
    return float(np.linalg.norm(e.mean(0)) / (np.linalg.norm(e, axis=1).mean() + 1e-12))


def exception_bounds(variant: Mapping) -> dict[str, float]:
    """The bounds a named exception sets from one variant's report.

    Each failed or unresolved cell and failed tile pair at its loss plus one SE; each tripped class
    (``trip:<class>``) at ``TRIP_ALLOWANCE`` x its gap closure.
    """
    out = {k: sum(variant["cells"][k]) for k in [*variant["fail_cells"], *variant["unresolved"]]}
    out |= {k: sum(variant["tile"][k]) for k in variant["fail_tile"]}
    return out | {f"trip:{k}": TRIP_ALLOWANCE * g for k, g in variant["trip"].items()}


def outside_exception(variant: Mapping, bounds: Mapping[str, float]) -> list[str]:
    """Why a variant is outside an exception's bounds for its tile; empty means inside.

    A failed envelope, a flagged item the exception does not list, or a listed item past its bound.
    """
    values = {k: v[0] for k, v in variant["cells"].items()} | {k: v[0] for k, v in variant["tile"].items()}
    values |= {f"trip:{k}": g for k, g in variant["gc"].items()}
    envelope_fails = variant["envelope"] is not None and not variant["envelope"]["passed"]
    reasons = ["ADR 012's envelope fails, which no exception covers"] if envelope_fails else []
    reasons += [f"{k} {values[k]:.3f} not in the exception" for k in exception_bounds(variant) if k not in bounds]
    return reasons + [
        f"{k} {values[k]:.3f} past its bound {b:.3f}" for k, b in bounds.items() if values.get(k, 0.0) > b
    ]
