"""Apply ADR 043's precision gate to one tile's per-variant predictions, and optionally hold variants to an exception.

Reads the ``.npz`` files ``calibrate.py`` wrote for one tile (``base.npz`` plus one per variant), applies
the gate (``tessera_embeddings.profiling.inference.precision_gate``), prints one line per variant and
writes the full report as JSON.

With ``--bounds``, each variant named in ``--hold`` is also checked against the exception's bounds for
this tile: ``--hold fp8_stage1=stage1:fp8`` holds variant ``fp8_stage1`` to the Stage 1 record's
``fp8`` comparison. The exit code is 3 if any held variant is outside the exception, so a caller can
tell that result from a crash.

usage: apply_gate.py <labels.npz> <label_key> <variant_dir> <tile> <report.json>
                     [--bounds fp8_exception_bounds.json --hold VARIANT=SIZE:COMPARISON ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tessera_embeddings.profiling.inference import precision_gate as pg

OUTSIDE_EXIT = 3


def load_scores(tile: pg.Tile, vdir: str) -> dict[str, pg.Score]:
    """Score every variant file in ``vdir`` against ``base.npz``; refit redraws carry no change to score."""
    base = dict(np.load(Path(vdir) / "base.npz"))
    scores = {}
    for path in sorted(Path(vdir).glob("*.npz")):
        name = path.stem
        if name == "base" or name.startswith("refit"):
            continue
        scores[name] = pg.score_variant(tile, dict(np.load(path)), base)
    return scores


def print_report(tile_name: str, report: dict) -> None:
    """One line per variant: verdict, tile losses, failures, unresolved cells and trips."""
    print(
        f"=== {tile_name}: z_fam {report['z_fam']:.2f} over {len(report['gated_cells'])} gated cells; classes "
        + ", ".join(f"{k} {v}" for k, v in report["classes"].items())
    )
    print(
        f"flag-only {report['flag_only']}; uncovered {report['uncovered']}; "
        f"control passes {report['control_passes']}; unvalidated {report['unvalidated']}"
    )
    print("tripwire x1e3: " + ", ".join(f"{k} {1e3 * v:.2f}" for k, v in report["tripwire"].items()))
    for name, v in report["variants"].items():
        t = v["tile"]
        line = (
            f"{name:28s} {v['verdict']:10s} probe {t['probe ref->x'][0]:+.3f}/{t['probe x->ref'][0]:+.3f} "
            f"knn {t['knn ref->x'][0]:+.3f}/{t['knn x->ref'][0]:+.3f}"
        )
        for label, items in (("FAILS", v["fail_tile"] + v["fail_cells"]), ("UNRESOLVED", v["unresolved"])):
            if items:
                line += f"; {label}: " + " | ".join(f"{k} {_fmt(v, k)}" for k in items)
        if v["trip"]:
            line += "; TRIP: " + ", ".join(f"{k} {1e3 * g:.2f}" for k, g in v["trip"].items())
        print(line)


def _fmt(variant: dict, key: str) -> str:
    loss, se = variant["cells"][key] if key in variant["cells"] else variant["tile"][key]
    return f"{loss:.2f}({se:.2f})"


def hold(report: dict, tile_name: str, bounds_path: str, holds: list[str]) -> bool:
    """Check each held variant against the exception; True if all are clean or inside."""
    bounds = json.loads(Path(bounds_path).read_text())
    ok = True
    for spec in holds:
        variant, size_cmp = spec.split("=")
        size, comparison = size_cmp.split(":")
        res = pg.check_exception(report["variants"][variant], bounds["sizes"][size].get(tile_name), comparison)
        report["variants"][variant]["exception"] = {
            "size": size,
            "comparison": comparison,
            "status": res.status,
            "reasons": res.reasons,
        }
        print(
            f"exception {variant} ({size}, {comparison}): {res.status.upper()}"
            + ("".join(f"\n    {r}" for r in res.reasons))
        )
        ok &= res.status != "outside"
    return ok


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels")
    p.add_argument("label_key", choices=sorted(pg.LABEL_GROUPS))
    p.add_argument("variant_dir")
    p.add_argument("tile")
    p.add_argument("report")
    p.add_argument("--bounds")
    p.add_argument("--hold", action="append", default=[])
    a = p.parse_args(argv)
    labels = np.load(a.labels)
    tile = pg.make_tile(labels[a.label_key], a.label_key, labels["Y"], labels["X"])
    report = pg.evaluate(tile, load_scores(tile, a.variant_dir))
    print_report(a.tile, report)
    ok = hold(report, a.tile, a.bounds, a.hold) if a.bounds else True
    Path(a.report).write_text(json.dumps(report, indent=1))
    return 0 if ok else OUTSIDE_EXIT


if __name__ == "__main__":
    sys.exit(main())
