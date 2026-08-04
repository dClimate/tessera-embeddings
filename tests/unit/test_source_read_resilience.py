"""The per-date source read: retried, and named when it fails.

Both halves exist because of the same asymmetry. Writes had always retried while reads had
not, so one transient failure reading one granule propagated out of the per-date loop and
failed a whole zone-year — and the message it failed with named neither the zone nor the
date, because it is raised on a Dask worker that knows only the task it was handed.

What these tests pin is therefore behavioural, not structural: that a transient read
survives, that a permanent one still surfaces, and that either way the log line carries
enough to attribute the failure to a cell and a date.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tessera_embeddings.ingest.roi_processing import (
    SOURCE_READ_ATTEMPTS,
    read_failure_context,
    source_read_retrying,
)

#: Read as text rather than imported: these two tests assert on the SHAPE of log calls, which
#: is not observable from the module object once the interpreter has compiled it.
_SRC = Path(__file__).resolve().parents[2] / "src" / "tessera_embeddings"


class _Item:
    """Stands in for a STAC item, which the context reads an ``id`` off."""

    def __init__(self, ident: str) -> None:
        self.id = ident


def _read(log: logging.Logger, fail_times: int) -> int:
    """Call through the retry, failing the first ``fail_times`` attempts."""
    state = {"calls": 0}
    for attempt in source_read_retrying(log):
        with attempt:
            state["calls"] += 1
            if state["calls"] <= fail_times:
                raise OSError("Read failed. See previous exception for details.")
    return state["calls"]


def test_a_transient_read_survives(caplog) -> None:
    log = logging.getLogger("t.transient")
    with caplog.at_level(logging.WARNING):
        calls = _read(log, fail_times=1)
    assert calls == 2, "the retry should have re-read the date once"


def test_a_permanent_read_still_surfaces(caplog) -> None:
    """Retrying must not convert a real failure into silence."""
    log = logging.getLogger("t.permanent")
    with caplog.at_level(logging.WARNING), pytest.raises(OSError, match="Read failed"):
        _read(log, fail_times=SOURCE_READ_ATTEMPTS)


def test_each_retry_is_logged_so_transient_and_permanent_are_distinguishable(caplog) -> None:
    """Without a line per attempt, a retried read and a first-try read look identical."""
    log = logging.getLogger("t.logged")
    with caplog.at_level(logging.WARNING):
        _read(log, fail_times=2)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_failure_names_the_roi_and_date(caplog) -> None:
    """The two fields that make a fleet-wide error attributable to one cell."""
    log = logging.getLogger("t.context")
    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(OSError, match="Read failed"),
        read_failure_context(log, roi="zone_55N", date="2021-03-14", items=[_Item("S2_abc")]),
    ):
        raise OSError("Read failed. See previous exception for details.")
    text = caplog.text
    assert "roi=zone_55N" in text
    assert "date=2021-03-14" in text
    assert "S2_abc" in text


def test_failure_preserves_the_underlying_cause(caplog) -> None:
    """Rasterio says "see previous exception"; that previous exception is the whole point."""
    log = logging.getLogger("t.chain")
    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(OSError, match="Read failed"),
        read_failure_context(log, roi="zone_55N", date="2021-03-14"),
    ):
        try:
            raise ValueError("CPLE_HttpResponse: 503 from the source bucket")
        except ValueError as cause:
            raise OSError("Read failed. See previous exception for details.") from cause
    record = next(r for r in caplog.records if r.levelno == logging.ERROR)
    assert record.exc_info is not None, "the traceback must be logged, not just the message"
    assert "503 from the source bucket" in caplog.text


def test_context_is_transparent_when_nothing_fails(caplog) -> None:
    """It must add no lines on the success path — it wraps every date of every cell."""
    log = logging.getLogger("t.quiet")
    with caplog.at_level(logging.DEBUG), read_failure_context(log, roi="zone_02N", date="2021-01-01"):
        pass
    assert caplog.records == []


class TestZeroDateOutcomeIsAttributable:
    """A radar leg that writes nothing must say which zone, and whether the source was empty.

    This is the shape that made five zones undiagnosable. ``status="skipped"`` reads as
    success to the parent flow, so a cell finished green with an orbit missing from the
    store — and every informational line in the leg except the per-date timing carried the
    ORBIT but not the zone, while the per-date line only fires once a date is written. A leg
    that wrote nothing therefore produced no line attributable to a cell at all.
    """

    def test_every_informational_line_carries_the_roi(self) -> None:
        """A line without ``roi=`` cannot be tied to a cell: the log stream is a task id."""
        import re

        src = (_SRC / "ingest" / "s1_roi.py").read_text()
        # Format strings passed to log.info / log.warning, excluding debug-level detail.
        calls = re.findall(r"log\.(?:info|warning)\(\s*\n?\s*((?:\"[^\"]*\"\s*\n?\s*)+)", src)
        missing = [c.strip() for c in calls if "roi=" not in c and "ROI %s" not in c]
        assert not missing, f"informational line(s) with no ROI: {missing}"

    def test_placeholders_match_arguments(self) -> None:
        """A miscounted %-placeholder loses the line on a worker rather than raising here."""
        import ast

        tree = ast.parse((_SRC / "ingest" / "s1_roi.py").read_text())
        bad = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in {"info", "warning", "error", "debug", "exception"}:
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            text = node.args[0].value
            if not isinstance(text, str):
                continue
            placeholders = text.count("%") - 2 * text.count("%%")
            if placeholders != len(node.args) - 1:
                bad.append((node.lineno, text[:50]))
        assert not bad, f"placeholder/argument mismatch: {bad}"
