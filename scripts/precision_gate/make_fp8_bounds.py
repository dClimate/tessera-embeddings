"""Write ADR 043's FP8 exception as bounds a later run is held to (``fp8_exception_bounds.json``).

Decision 2: FP8 may fail or leave unresolved only the cells it did in the record, plus Ecuador's
tile bound, each by no more than its recorded loss plus one jackknife SE, and may trip the tripwire
only in the classes it tripped, by no more than 1.1 times the recorded gap closure. Each size
(``stage1``, ``all4``) is held to its own record, in each comparison: ``fp8`` (L40S FP8 against L40S
bf16) and ``mixed`` (A10G bf16 against L40S FP8).

Input: the record's reports in the form ``apply_gate.py`` writes, one ``<tile>.json`` per tile,
regenerated from the record's predictions (``s3://arbol-tessera-inputs-dev/perf/fp8-mixed/perclass/``).

usage: make_fp8_bounds.py <report_dir> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TRIP_ALLOWANCE = 1.1
SIZES = ("stage1", "all4")
COMPARISONS = ("fp8", "mixed")


def tile_bounds(report: dict, size: str) -> dict:
    """One tile's listed cells, tile bounds and tripwire classes for one size, by comparison."""
    out: dict[str, dict[str, dict[str, float]]] = {"cells": {}, "tile": {}, "trip": {}}
    for cmp in COMPARISONS:
        v = report["variants"][f"{cmp}_{size}"]
        for key in [*v["fail_cells"], *v["unresolved"]]:
            loss, se = v["cells"][key]
            out["cells"].setdefault(key, {})[cmp] = loss + se
        for key in v["fail_tile"]:
            loss, se = v["tile"][key]
            out["tile"].setdefault(key, {})[cmp] = loss + se
        for cls, gc in v["trip"].items():
            out["trip"].setdefault(cls, {})[cmp] = TRIP_ALLOWANCE * gc
    return {k: v for k, v in out.items() if v}


def main(argv: list[str]) -> int:
    """Entry point."""
    report_dir, out_path = argv
    reports = {p.stem: json.loads(p.read_text()) for p in sorted(Path(report_dir).glob("*.json"))}
    sizes = {s: {t: b for t, r in reports.items() if (b := tile_bounds(r, s))} for s in SIZES}
    doc = {
        "decision": "context_docs/decisions/043-fp8-linears-ship-on-cross-change-accuracy-and-the-a10g-stays-bf16.md",
        "rule": "a cell or tile bound: recorded loss + its jackknife SE, in points; a tripwire class: "
        f"{TRIP_ALLOWANCE} x recorded gap closure. A comparison with no record of its own uses the largest recorded.",
        "tiles": sorted(reports),
        "sizes": sizes,
    }
    Path(out_path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
