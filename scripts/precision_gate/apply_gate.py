"""Apply ADR 043's precision gate to one tile's per-variant predictions.

Reads what ``calibrate.py`` wrote for one tile: ``base.npz``, whose ``variants`` list is the manifest (a
missing file is an error, a stray one is ignored), one ``.npz`` per variant and ``envelope.json``. Applies
the gate (``tessera_embeddings.profiling.inference.precision_gate``), prints one line per variant and the
refit spread, and writes the full report as JSON.

Every embedded comparison (each key of ``envelope.json``, known-good ones included) must pass, or, with
``--exception``, stay inside that named exception's bounds for this tile. The exit code is 3 if any does
not, so a caller can tell that result from a crash. Each comparison that does not pass carries the
``exception_bounds`` an exception for it would set; an exception file is
``{"decision": <ADR>, "tiles": {<tile>: {<item>: <bound>}}}``, taking the larger of ``candidate``'s and
``mixed``'s bound per item.

usage: apply_gate.py <labels.npz> <label_key> <variant_dir> <tile> <report.json> [--exception <file.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tessera_embeddings.profiling.inference import precision_gate as pg

NOT_PASSED_EXIT = 3


def load_report(tile: pg.Tile, vdir: str) -> dict:
    """Score every variant ``base.npz`` lists, and evaluate the tile with its envelopes and refit spread."""
    base = dict(np.load(Path(vdir) / "base.npz"))
    scores, refits = {}, []
    for name in base["variants"].tolist():
        preds = dict(np.load(Path(vdir) / f"{name}.npz"))
        if name.startswith("refit"):
            refits.append(preds)  # a redraw of the training sample: no change to score
        else:
            scores[name] = pg.score_variant(tile, preds, base)
    report = pg.evaluate(tile, scores, json.loads((Path(vdir) / "envelope.json").read_text()))
    report["refit_spread"] = pg.refit_spread(tile, base, refits)
    return report


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
    worst = sorted(report["refit_spread"].items(), key=lambda kv: -kv[1])[:5]
    print("refit spread SD (points), largest: " + ", ".join(f"{k} {v:.2f}" for k, v in worst))
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
        if v["envelope"] and not v["envelope"]["passed"]:
            line += "; ENVELOPE FAILS"
        print(line)


def _fmt(variant: dict, key: str) -> str:
    loss, se = variant["cells"][key] if key in variant["cells"] else variant["tile"][key]
    return f"{loss:.2f}({se:.2f})"


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels")
    p.add_argument("label_key", choices=sorted(pg.LABEL_GROUPS))
    p.add_argument("variant_dir")
    p.add_argument("tile")
    p.add_argument("report")
    p.add_argument("--exception")
    a = p.parse_args(argv)
    labels = np.load(a.labels)
    tile = pg.make_tile(labels[a.label_key], a.label_key, labels["Y"], labels["X"])
    report = load_report(tile, a.variant_dir)
    print_report(a.tile, report)
    exception = json.loads(Path(a.exception).read_text())["tiles"].get(a.tile, {}) if a.exception else None
    report["not_passed"] = []
    for name, v in report["variants"].items():
        if not v["envelope"] or v["verdict"] == "pass":
            continue
        v["exception_bounds"] = pg.exception_bounds(v)
        if exception is not None:
            v["outside_exception"] = pg.outside_exception(v, exception)
            print(f"exception {name}: " + ("; ".join(v["outside_exception"]) or "inside"))
            if not v["outside_exception"]:
                continue
        report["not_passed"].append(name)
    print(f"GATE {a.tile}: " + (f"NOT PASSED: {', '.join(report['not_passed'])}" if report["not_passed"] else "PASS"))
    Path(a.report).write_text(json.dumps(report, indent=1))
    return NOT_PASSED_EXIT if report["not_passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
