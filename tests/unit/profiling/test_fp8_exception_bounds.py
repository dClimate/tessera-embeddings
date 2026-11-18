"""``scripts/precision_gate/fp8_exception_bounds.json`` carries ADR 043's FP8 exception as the ADR states it.

The file is generated once from the record's reports by ``make_fp8_bounds.py``; this pins it to the figures
the decision quotes, so a regeneration from different reports cannot silently change the exception.
"""

from __future__ import annotations

import json

import pytest

from tests._paths import REPO_ROOT

BOUNDS = REPO_ROOT / "scripts" / "precision_gate" / "fp8_exception_bounds.json"
WW = "winter wheat|probe|ref->x|recall"


def test_fp8_bounds_match_adr_043() -> None:
    """The bounds file carries the figures ADR 043 quotes: recorded loss + SE, and 1.1 x gap closure."""
    sizes = json.loads(BOUNDS.read_text())["sizes"]
    s1, a4 = sizes["stage1"], sizes["all4"]
    assert s1["fp8-kansas"]["cells"][WW]["fp8"] == pytest.approx(0.66 + 0.16, abs=0.01)
    assert s1["fp8-kansas"]["cells"][WW]["mixed"] == pytest.approx(0.70 + 0.15, abs=0.01)
    assert s1["fp8-ecuador"]["tile"]["probe x->ref"]["mixed"] == pytest.approx(0.2005 + 0.047, abs=0.001)
    assert s1["fp8-ecuador"]["cells"]["wetland|probe|x->ref|recall"]["fp8"] == pytest.approx(0.79 + 0.17, abs=0.01)
    assert a4["cpu-ab-iowa"]["cells"]["forest|probe|x->ref|precision"]["mixed"] == pytest.approx(0.65 + 0.15, abs=0.01)
    assert a4["fp8-ecuador"]["cells"]["trees|probe|x->ref|recall"]["fp8"] == pytest.approx(0.55 + 0.09, abs=0.01)
    assert a4["fp8-kansas"]["cells"][WW]["mixed"] == pytest.approx(1.00 + 0.21, abs=0.01)
    assert a4["fp8-kansas"]["cells"]["sorghum|probe|ref->x|recall"] == pytest.approx({"mixed": 0.51 + 0.30}, abs=0.01)
    assert a4["fp8-punjab"]["cells"]["trees|probe|ref->x|precision"]["fp8"] == pytest.approx(0.53 + 0.10, abs=0.01)
    # Tripwire classes per size, as ADR 043's gap-closure table: none on UK or Finland at Stage 1.
    trips = {size: {(t, c) for t, b in sizes[size].items() for c in b.get("trip", {})} for size in sizes}
    assert trips["stage1"] == {
        ("cpu-ab-iowa", "pasture"),
        ("fp8-ecuador", "trees"),
        ("fp8-ecuador", "grass"),
        ("fp8-punjab", "trees"),
        ("fp8-kansas", "pasture"),
        ("fp8-kansas", "winter wheat"),
        ("fp8-kansas", "sorghum"),
    }
    assert trips["all4"] == trips["stage1"] | {("cpu-ab-uk", "trees"), ("fp8-finland", "grass")}
    assert a4["fp8-kansas"]["trip"]["sorghum"]["fp8"] == pytest.approx(1.1 * 7.03e-3, abs=1e-5)
