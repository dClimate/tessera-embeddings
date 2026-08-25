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

import ast
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from tessera_embeddings.ingest.duplicates import is_provider_refusal, is_unreadable_source
from tessera_embeddings.ingest.loader_failures import clear_local_refusals, install_capture
from tessera_embeddings.ingest.roi_processing import (
    SOURCE_READ_ATTEMPTS,
    read_failure_context,
    source_read_retrying,
)

#: The wording GDAL really used, copied from CloudWatch during the 2026-08-24 outage. Composed
#: text is what let this whole area pass while recording nothing in production.
REAL_LOGGED_REFUSAL = (
    "CPLE_AppDefined in HTTP response code on "
    "https://asf-cumulus-prod-opera-products.s3.us-west-2.amazonaws.com/OPERA_L2_RTC-S1/"
    "OPERA_L2_RTC-S1_T072-152803-IW2_20211108T150433Z_S1B_30_v1.0_VV.tif: 403"
)


def _gdal_logs(message: str) -> None:
    """Emit ``message`` the way rasterio's CPL error handler emits GDAL's warnings."""
    logging.getLogger("rasterio._env").warning(message)


#: Read as text rather than imported: these two tests assert on the SHAPE of log calls, which
#: is not observable from the module object once the interpreter has compiled it.
_SRC = Path(__file__).resolve().parents[2] / "src" / "tessera_embeddings"


class _Item:
    """Stands in for a STAC item, which the context reads an ``id`` off."""

    def __init__(self, ident: str) -> None:
        self.id = ident


def _read(log: logging.Logger, fail_times: int) -> int:
    """Call through the retry, failing the first ``fail_times`` attempts.

    The real ladder spends ~61 s asleep (see :func:`source_read_retrying`), and a permanent
    failure pays all of it. That is a production decision, but it is not something a unit test
    should sit through — it made this file alone 67 s of the suite's wall time.

    Only ``sleep`` is replaced. The stop condition and the wait policy are left exactly as
    production builds them, so the attempt COUNT and the ordering of the log lines are still
    the real ones; what is skipped is the waiting between attempts, which no test here asserts
    on. The ladder's own shape is pinned directly by
    :func:`test_the_retry_ladder_is_the_one_production_pays_for` instead, which checks more
    than experiencing it ever did.
    """
    state = {"calls": 0}
    retrying = source_read_retrying(log)
    retrying.sleep = lambda _seconds: None
    for attempt in retrying:
        with attempt:
            state["calls"] += 1
            if state["calls"] <= fail_times:
                raise OSError("Read failed. See previous exception for details.")
    return state["calls"]


def test_the_retry_ladder_is_the_one_production_pays_for() -> None:
    """The budget a permanently-failing date costs, asserted rather than waited out.

    ``_read`` skips the sleeps so the suite stays fast, which would otherwise leave the ladder
    itself unchecked — a change to the multiplier or the cap would go unnoticed. This pins the
    three parameters that decide the cost, and the total they add up to.

    61 s per permanently-failing date is deliberate: long enough to outlast a provider having
    a bad minute, and paid only on the failing path. ``reraise`` is what stops the last
    attempt's failure being replaced by tenacity's own ``RetryError``, which is how a real
    read failure would otherwise lose its cause.
    """
    retrying = source_read_retrying(logging.getLogger("t.ladder"))

    assert retrying.stop.max_attempt_number == SOURCE_READ_ATTEMPTS
    assert (retrying.wait.multiplier, retrying.wait.min, retrying.wait.max) == (1, 2.0, 15.0)
    assert retrying.reraise is True

    class _State:
        """The one field :class:`wait_exponential` reads off a retry state."""

        def __init__(self, attempt: int) -> None:
            self.attempt_number = attempt

    sleeps = [retrying.wait(_State(n)) for n in range(1, SOURCE_READ_ATTEMPTS)]
    assert sleeps == [2.0, 2.0, 4, 8, 15.0, 15.0, 15.0]
    assert sum(sleeps) == 61.0, "the per-date backoff budget named in source_read_retrying"


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


class TestTheContextCompletesTheReason:
    """The one place both sensors' reads pass through, and so the one place their evidence meets.

    A read failure's reason is not always in the exception. A refused object comes back as an
    error document, GDAL states the refusal in its own log, and the codec raises the decode
    failure it fails on — so the chain says the bytes are bad while the words saying the service
    refused sit one log line away. The two verdicts are opposites: bad bytes means give the date
    up, refused means wait and give up nothing. Collecting that evidence here is what makes one
    classifier decide both paths from all of what is known.
    """

    @pytest.fixture(autouse=True)
    def _capture(self):
        """The capture running and its buffer empty either side of the test."""
        install_capture()
        gdal = logging.getLogger("rasterio._env")
        previous = gdal.level
        gdal.setLevel(logging.WARNING)
        clear_local_refusals()
        yield
        gdal.setLevel(previous)
        clear_local_refusals()

    def _raise_through_the_context(
        self, log: logging.Logger, while_reading: Callable[[], None] | None = None
    ) -> BaseException:
        """One date's read failing the way a refused object fails: a codec, and nothing else.

        Chained with ``raise ... from``, which is the only thing that puts a cause on an
        exception, because the cause is what every verdict here is read from.

        ``while_reading`` runs INSIDE the context, which is where GDAL writes its log line — the
        read is under way by then. Logging it before the context is entered describes a different
        situation entirely: a line left over from an earlier read, which the context deliberately
        will not attach.
        """
        with pytest.raises(OSError) as caught, read_failure_context(log, roi="zone_55N", date="2021-03-14"):
            if while_reading is not None:
                while_reading()
            try:
                raise ValueError("CPLE_AppDefinedError: ZIPDecode:Decoding error at scanline 0")
            except ValueError as cause:
                raise OSError("RasterioIOError: Read failed. See previous exception for details.") from cause
        return caught.value

    def test_a_refusal_only_gdal_logged_reaches_the_verdict(self, caplog) -> None:
        """The failure that cost a hundred and fifty-eight dates, decided the other way."""
        with caplog.at_level(logging.ERROR):
            arrived = self._raise_through_the_context(
                logging.getLogger("t.refused"), while_reading=lambda: _gdal_logs(REAL_LOGGED_REFUSAL)
            )

        assert is_provider_refusal(arrived) is True
        assert is_unreadable_source(arrived) is False, "a refusal must never give a date up"

    def test_a_line_left_over_from_an_earlier_read_is_not_attached(self, caplog) -> None:
        """The bound on the evidence, and the reason it is needed.

        A read that logs a refusal and then SUCCEEDS on a later attempt is what the read retry
        exists to allow, and it drains nothing — the buffer is emptied only when something fails.
        Left unbounded, that line is attached to whatever fails next, and on the optical path a
        genuinely corrupt object then reads as a refusal and the copy ladder never steps down.

        Only the evidence produced DURING the read being judged is attached.
        """
        _gdal_logs(REAL_LOGGED_REFUSAL)
        time.sleep(0.05)
        with caplog.at_level(logging.ERROR):
            arrived = self._raise_through_the_context(logging.getLogger("t.stale"))

        assert is_provider_refusal(arrived) is False, "the stale line was inherited"
        assert is_unreadable_source(arrived) is True, "so the failure is still read as bad bytes"

    def test_the_same_failure_with_nothing_logged_still_reads_as_bad_bytes(self, caplog) -> None:
        """The control: the context must complete a reason, never invent one. A codec failure
        with no refusal behind it is still permanently bad imagery, and still costs its date.
        """
        with caplog.at_level(logging.ERROR):
            arrived = self._raise_through_the_context(logging.getLogger("t.corrupt"))

        assert is_provider_refusal(arrived) is False
        assert is_unreadable_source(arrived) is True

    def test_both_sensors_hand_the_context_their_cluster(self) -> None:
        """The evidence is on the READING workers, so a context with no client decides from the
        chain alone — silently, and only on the path that dropped it. One sensor keeping the
        argument and the other losing it is exactly the split this change exists to close.
        """
        for module in ("s1_roi.py", "s2_roi.py"):
            src = (_SRC / "ingest" / module).read_text()
            contexts = re.findall(r"read_failure_context\(([^)]*)\)", src)
            assert contexts, f"{module}: no read_failure_context call found — has it been renamed?"
            for call in contexts:
                assert "client=client" in call, (
                    f"{module}: `read_failure_context({call})` cannot collect what GDAL logged "
                    "without the cluster, so a refusal there reads as unreadable data"
                )


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
        src = (_SRC / "ingest" / "s1_roi.py").read_text()
        # Format strings passed to log.info / log.warning, excluding debug-level detail.
        calls = re.findall(r"log\.(?:info|warning)\(\s*\n?\s*((?:\"[^\"]*\"\s*\n?\s*)+)", src)
        missing = [c.strip() for c in calls if "roi=" not in c and "ROI %s" not in c]
        assert not missing, f"informational line(s) with no ROI: {missing}"

    # test_placeholders_match_arguments lived here. It walked the AST of ingest/s1_roi.py
    # counting %-placeholders against arguments. Ruff's PLE1205/PLE1206 make the same check
    # over EVERY file rather than this one module, so the rule is enabled in ruff.toml and
    # the test is gone. Removed 2026-08-25 after confirming both a too-many and a too-few
    # mutant in s1_roi.py fail the lint exactly as they used to fail the test.


class TestEveryComputeThatReadsSourceIsAttributed:
    """A read failure must name its granules wherever the read happens, not only at the gate.

    The optical coverage gate computes only SCL, so a reflectance band that cannot be read
    survives the gate and fails in the WRITE's compute instead. That path had no attribution
    context, so the failure named no granule and identifying the object meant correlating
    GDAL's own stderr by timestamp across the whole fleet.
    """

    def test_both_sensor_writes_are_wrapped(self) -> None:
        """No write that computes source pixels may be reached outside ``read_failure_context``.

        Checked by CONTAINMENT, not by textual order. The optical gate's own context appears
        earlier in the file, so "a context exists above the write" is satisfied by a write
        that is not wrapped at all.

        The writers are matched by SHAPE rather than named one by one, because naming them was
        the defect. This check said ``write_day_windows`` and did not say ``write_days_windows``
        — one letter apart, and the batched writer is the DEFAULT path for a compact ROI. It went
        unwrapped, so a batch failure was classified from the codec exception alone: the refusal
        was invisible until the batch had burned its retry ladder and every date in it had been
        recomputed singly.
        """
        for module in ("s1_roi.py", "s2_roi.py"):
            tree = ast.parse((_SRC / "ingest" / module).read_text())
            writes = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and _WRITES_SOURCE_PIXELS.match(node.func.id)
            ]
            assert writes, f"{module}: no source-pixel write found — have they been renamed?"
            for write in writes:
                assert _enclosing_contexts(tree, write) & {"read_failure_context"}, (
                    f"{module}: `{write.func.id}` at line {write.lineno} computes source pixels, "  # type: ignore[attr-defined]
                    "so it must be lexically inside `with read_failure_context(...)` — otherwise "
                    "a failed read names no granule and a logged refusal reaches no verdict"
                )

    def test_the_optical_write_passes_its_items(self) -> None:
        """A context with no items names a count of zero, which is worse than no line.

        The radar path has no per-date item list to pass; the optical path does, and it is
        the only thing that identifies WHICH source object failed.
        """
        src = (_SRC / "ingest" / "s2_roi.py").read_text()
        assert "items=prepared.items" in src, "the optical write must pass the day's items"


#: Every function that computes source pixels on the cluster, matched on shape. A per-date writer
#: and a batched one differ by a single letter, and an enumeration that missed the second is what
#: left the default optical path unwrapped.
_WRITES_SOURCE_PIXELS = re.compile(r"^write_day")


def _enclosing_contexts(tree: object, target: object) -> set[str]:
    """Names of the context managers whose ``with`` blocks lexically contain ``target``."""
    found: set[str] = set()

    def walk(node: object, active: frozenset[str]) -> None:
        if node is target:
            found.update(active)
            return
        if isinstance(node, ast.With | ast.AsyncWith):
            names = {
                item.context_expr.func.id
                for item in node.items
                if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
            }
            for child in node.body:
                walk(child, active | names)
            for item in node.items:
                walk(item.context_expr, active)
            return
        for child in ast.iter_child_nodes(node):  # type: ignore[arg-type]
            walk(child, active)

    walk(tree, frozenset())
    return found
