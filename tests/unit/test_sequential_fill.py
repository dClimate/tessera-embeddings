"""fill_zones_sequential: the chained stream — feeder, tallies, trailing assembly.

The runner is pure orchestration over injected callables (Prefect/Ray-free by
contract). These tests drive it with a synchronous fake ``session`` that
drains ``more_work`` and fires ``on_item_done`` per item — no cluster, no
store, no monkeypatching. The staged-resume scan is the one internal
dependency, stubbed at the module namespace.
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
import threading
import time

import pytest

import tessera_embeddings.orchestration.runners.sequential_fill as mod
from tessera_embeddings.config.fault_injection import (
    FAULT_LOG_PREFIX,
    WITHHOLD_WORK,
    ArmedFault,
    FaultInjection,
)
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.assembly import StagedResume
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.orchestration.runners.sequential_fill import (
    PreparedCell,
    SequentialCell,
    fill_zones_sequential,
)
from tessera_embeddings.orchestration.runners.zone_fill import ZoneFillHandoff, ZonePlan

LOG = logging.getLogger("test-sequential-fill")


def _cells(n: int) -> list[SequentialCell]:
    return [SequentialCell(zone=f"{i + 1:02d}N", year=2025, num_actors=5) for i in range(n)]


class _Config:
    """Minimal config stand-in (orbit, the scan's compute_std probe, the cell's window)."""

    def __init__(self, s1_orbit: str = "both", time_window: object | None = None) -> None:
        self.s1_orbit = s1_orbit
        self.compute_std = False
        # Carried onto every work item as ZoneContext.time_window, so a chained session's
        # cells can span campaign years without being inferred over the session's months.
        self.time_window = time_window if time_window is not None else parse_time_window("December 2025")


def _prepare_for(orbits: dict[str, str] | None = None):
    def _prepare(cell: SequentialCell) -> PreparedCell:
        return PreparedCell(
            mosaic_base=f"m/{cell.zone}",
            staging_base=f"s/{cell.zone}",
            run_id=f"r-{cell.zone}",
            config=_Config((orbits or {}).get(cell.zone, "both")),
        )

    return _prepare


def _tiles(zone: str, n: int) -> list[ChunkSpec]:
    return [ChunkSpec(row=0, col=i, y_start=0, y_stop=64, x_start=i * 64, x_stop=(i + 1) * 64) for i in range(n)]


def _plan_for(tiles: int = 2, done_zones: set[str] | None = None):
    def _plan(cell: SequentialCell, prep: PreparedCell) -> ZonePlan:
        if done_zones and cell.zone in done_zones:
            return ZonePlan(
                cell.zone, cell.year, prep.run_id, 0.0, {}, [], done={"zone": cell.zone, "already_complete": True}
            )
        return ZonePlan(
            cell.zone,
            cell.year,
            prep.run_id,
            0.0,
            {"zone": cell.zone, "live_tiles": tiles},
            _tiles(cell.zone, tiles),
        )

    return _plan


def _sync_session(fail: set[str] | None = None, events: list[str] | None = None):
    """A session that drains more_work synchronously, completing every item."""

    def session(more_work, on_item_done):
        results = []
        while True:
            batch = more_work()
            if batch is None:
                return results
            for item in batch:
                if events is not None:
                    events.append(f"infer:{item.uid}")
                status = "failed" if fail and item.ctx.run_id.removeprefix("r-") in fail else "success"
                result = {"chunk": item.chunk.label, "status": status}
                results.append(result)
                on_item_done(item, result)

    return session


def _assemble(events: list[str] | None = None):
    def assemble(handoff, prep):
        if events is not None:
            events.append(f"assemble:{handoff.zone}")
        return {"zone": handoff.zone, "empty": False, "succeeded": len(handoff.results)}

    return assemble


def _no_fallback(cell, prep, final):
    raise AssertionError("infer_single must not run when no cell defers")


class RecordingInputs:
    """CellInputs fake appending every call to a shared event list (thread-safe)."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._lock = threading.Lock()

    def start(self, zone: str, year: int) -> None:
        with self._lock:
            if f"start:{zone}" not in self.events:
                self.events.append(f"start:{zone}")

    def wait(self, zone: str, year: int, stop: threading.Event | None = None) -> None:
        with self._lock:
            self.events.append(f"wait:{zone}")

    def cleanup(self, zone: str, year: int) -> None:
        with self._lock:
            self.events.append(f"cleanup:{zone}")

    def discard(self, zone: str, year: int) -> None:
        """Forget the cell's attempt so a later `start` counts as a new one.

        Recorded rather than un-recording the `start`: several tests count `start:`
        events as "mosaics this run produced", and a discard does not un-produce one.
        """
        with self._lock:
            self.events.append(f"discard:{zone}")

    def cancel_unstarted(self) -> int:
        """Recorded, not simulated: the fake has no queue, and what the runner's
        contract requires is that it asks BEFORE retrying.
        """
        with self._lock:
            self.events.append("cancel_unstarted")
        return 0


class MosaicCountingInputs(RecordingInputs):
    """RecordingInputs that also tracks the started-not-cleaned high-water mark —
    how many cells hold a mosaic at once.
    """

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.alive = 0
        self.high_water = 0

    def start(self, zone: str, year: int) -> None:
        with self._lock:
            if f"start:{zone}" not in self.events:
                self.events.append(f"start:{zone}")
                self.alive += 1
                self.high_water = max(self.high_water, self.alive)

    def cleanup(self, zone: str, year: int) -> None:
        with self._lock:
            self.events.append(f"cleanup:{zone}")
            self.alive -= 1


class FailingWaitInputs(RecordingInputs):
    """Every ingest FAILS, and fails where the feeder itself sees it — which is what
    makes the retained-failure cap deterministic (see the two cap tests).
    """

    def wait(self, zone: str, year: int, stop: threading.Event | None = None) -> None:
        with self._lock:
            self.events.append(f"wait:{zone}")
        raise RuntimeError(f"ingest failed for {zone}-{year}")


def _resume(done=frozenset(), skipped=frozenset()):
    """A StagedResume stub for the resume scan."""
    return StagedResume(done=set(done), skipped=set(skipped))


@pytest.fixture(autouse=True)
def _no_scan(monkeypatch):
    """Staged-resume scan: nothing staged unless a test overrides."""
    monkeypatch.setattr(
        mod.ZarrWriter,
        "scan_existing_staged_artifacts",
        lambda self, run_id, chunks, **kw: _resume(),
        raising=True,
    )


def _run(cells, *, orbits=None, plan=None, session=None, assemble=None, infer_single=_no_fallback, log=LOG, **kw):
    return fill_zones_sequential(
        cells=cells,
        prepare=_prepare_for(orbits),
        plan=plan or _plan_for(),
        session=session or _sync_session(),
        assemble=assemble or _assemble(),
        infer_single=infer_single,
        session_s1_orbit="both",
        log=log,
        **kw,
    )


def test_all_zones_stream_in_order_and_assemble(monkeypatch):
    events: list[str] = []
    summary = _run(_cells(3), session=_sync_session(events=events), assemble=_assemble(events))
    assert summary["succeeded"] == 3 and summary["failed"] == 0
    infers = [e for e in events if e.startswith("infer:")]
    # Zone order preserved: all of 01N's tiles before 02N's, etc.
    assert infers == sorted(infers)
    assert [e for e in events if e.startswith("assemble:")] == ["assemble:01N", "assemble:02N", "assemble:03N"]


def test_zone_uids_are_run_scoped(monkeypatch):
    """Identical tile labels across zones stay distinct in the stream."""
    events: list[str] = []
    _run(_cells(2), session=_sync_session(events=events))
    uids = {e.removeprefix("infer:") for e in events if e.startswith("infer:")}
    assert uids == {"r-01N:chunk_0_0", "r-01N:chunk_0_1", "r-02N:chunk_0_0", "r-02N:chunk_0_1"}


def test_ingest_lifecycle_and_cleanup_after_assembly():
    events: list[str] = []
    inputs = RecordingInputs(events)
    _run(_cells(3), inputs=inputs, look_ahead=1)
    # Every cell was started/waited, and cleaned only after landing.
    for z in ("01N", "02N", "03N"):
        assert f"start:{z}" in events and f"wait:{z}" in events and f"cleanup:{z}" in events


def test_failed_zone_keeps_mosaic_and_others_land():
    events: list[str] = []
    inputs = RecordingInputs(events)
    with pytest.raises(RuntimeError, match="1/3 cell"):
        _run(_cells(3), session=_sync_session(fail={"02N"}, events=events), inputs=inputs)
    assert "cleanup:02N" not in events
    assert "cleanup:01N" in events and "cleanup:03N" in events


def test_terminal_plan_recorded_without_streaming():
    events: list[str] = []
    inputs = RecordingInputs(events)
    summary = _run(_cells(2), plan=_plan_for(done_zones={"01N"}), session=_sync_session(events=events), inputs=inputs)
    assert summary["succeeded"] == 2
    assert not [e for e in events if e.startswith("infer:r-01N")]  # 01N never streamed
    assert "cleanup:01N" in events  # its mosaic is still released


def test_all_resumed_zone_goes_straight_to_assembly(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        mod.ZarrWriter,
        "scan_existing_staged_artifacts",
        lambda self, run_id, chunks, **kw: _resume({c.label for c in chunks}) if run_id == "r-01N" else _resume(),
    )
    summary = _run(_cells(2), session=_sync_session(events=events), assemble=_assemble(events))
    assert summary["succeeded"] == 2
    assert not [e for e in events if e.startswith("infer:r-01N")]  # nothing left to infer
    assert "assemble:01N" in events  # but it still assembles (verify + tag)


def test_resumed_skip_markers_are_counted_as_skips_not_successes(monkeypatch):
    """A restored artifact must report the outcome it actually recorded.

    Skip markers and staged zarrs both mean "do not re-infer", but only one of them
    produced pixels. Counting a restored skip as a success makes a resumed zone's
    tally disagree with the same zone's tally on a fresh run: an all-skipped resume
    publishes empty while reporting zero skips.
    """
    seen: dict[str, list[dict]] = {}

    def assemble(handoff, prep):
        seen[handoff.zone] = list(handoff.results)
        return {"zone": handoff.zone, "empty": False}

    # 01N resumes entirely from skip markers; 02N resumes entirely from staged zarrs.
    monkeypatch.setattr(
        mod.ZarrWriter,
        "scan_existing_staged_artifacts",
        lambda self, run_id, chunks, **kw: _resume(
            {c.label for c in chunks}, {c.label for c in chunks} if run_id == "r-01N" else set()
        ),
    )
    _run(_cells(2), session=_sync_session(), assemble=assemble)

    assert seen["01N"] and all(r["status"] == "skipped" for r in seen["01N"])
    assert seen["02N"] and all(r["status"] == "success" for r in seen["02N"])


def test_a_cell_with_a_different_orbit_streams_under_its_own_orbit():
    """It used to be deferred out of the stream; it no longer needs to be.

    The orbit rides on each cell's ``ZoneContext``, exactly as its inference window already
    does, so one actor serves cells of differing orbits. That has to work: parts of the globe
    are radar-free in principle, so a cell resolving something other than the session's
    request is a permanent population, and deferring it against a bounded budget meant that
    population could never complete.
    """
    events: list[str] = []
    calls: list[tuple[str, bool]] = []

    def infer_single(cell, prep, final):
        calls.append((cell.zone, final))

        return ZoneFillHandoff(
            zone=cell.zone, year=cell.year, run_id=prep.run_id, t0=0.0, summary={}, live=[], results=[]
        )

    summary = _run(
        _cells(3),
        orbits={"02N": "ascending"},
        session=_sync_session(events=events),
        assemble=_assemble(events),
        infer_single=infer_single,
    )
    assert "deferred_orbit_mismatch" not in summary, "the deferral key should be gone, not merely zero"
    assert calls == [], "the per-cell fallback must not be reached by an orbit mismatch"
    assert [e for e in events if e.startswith("infer:r-02N")], "the mismatched cell must STREAM"
    assert summary["succeeded"] == 3


def test_the_streamed_work_carries_each_cell_s_own_orbit():
    """Streaming it is only correct if the actor is TOLD the cell's orbit.

    Without this the mismatched cell would stream and be read under the session's orbit —
    silently the wrong data, which is worse than the deferral it replaces.
    """
    seen: dict[str, str | None] = {}

    def session(more_work, on_item_done):
        """Drain the stream, recording the orbit each work item carried."""
        results = []
        while True:
            batch = more_work()
            if batch is None:
                return results
            for item in batch:
                seen[item.ctx.run_id.removeprefix("r-")] = item.ctx.s1_orbit
                result = {"chunk": item.chunk.label, "status": "success"}
                results.append(result)
                on_item_done(item, result)

    _run(_cells(3), orbits={"02N": "ascending"}, session=session, assemble=_assemble([]))
    assert seen.get("02N") == "ascending", seen
    assert {v for k, v in seen.items() if k != "02N"} <= {"both"}, seen


def test_assembly_failure_recorded_run_continues():
    def assemble(handoff, prep):
        if handoff.zone == "01N":
            raise RuntimeError("commit refused")
        return {"zone": handoff.zone}

    with pytest.raises(RuntimeError, match="1/2 cell"):
        _run(_cells(2), assemble=assemble)


def test_session_crash_unwinds_feeder_without_deadlock():
    """A crashing session must not leave the feeder blocked on zone slots."""

    def crashing_session(more_work, on_item_done):
        raise RuntimeError("cluster fell over")

    # Enough cells that the feeder WILL block on the slot semaphore
    # (look_ahead+2 slots, none ever released because nothing completes).
    with pytest.raises(RuntimeError, match="cluster fell over"):
        _run(_cells(8), session=crashing_session, look_ahead=1)


def test_prepare_failure_is_cell_scoped():
    def prepare(cell):
        if cell.zone == "01N":
            raise ValueError("mosaic missing")
        return _prepare_for()(cell)

    with pytest.raises(RuntimeError) as exc_info:
        fill_zones_sequential(
            cells=_cells(2),
            prepare=prepare,
            plan=_plan_for(),
            session=_sync_session(),
            assemble=_assemble(),
            infer_single=_no_fallback,
            session_s1_orbit="both",
            log=LOG,
        )
    assert "'phase': 'inputs/prepare'" in str(exc_info.value)
    assert "02N" not in str(exc_info.value)  # the other cell landed


# ---------------------------------------------------------------------------
# Ingest is paced by its own driver, never by the fill's progress
# ---------------------------------------------------------------------------


def test_ingest_starts_for_every_cell_and_is_not_paced_by_the_fill():
    """Every pending cell's ingest starts, however far behind finalization runs.

    This REPLACES a test that asserted the opposite — that started-not-cleaned mosaics
    never exceed ``look_ahead + 2``. That bound admitted ingest *starts*, and because a
    slot was released only when a cell's assembly landed, a new ingest waited on an
    assembly several cells back. Ingest throughput was therefore set by assembly
    throughput: measured live, a fleet configured for 60 concurrent ingests ran 7.

    The bound is gone and this pins its absence. Concurrency is the ingest driver's
    ``max_parallel``, which is not this runner's to enforce — so what is asserted here is
    that the runner does not add a second, tighter gate of its own.
    """
    events: list[str] = []
    inputs = MosaicCountingInputs(events)
    _run(_cells(8), inputs=inputs, look_ahead=1)

    # ORDERING is the isolating property, not the count. Counting starts passes either way:
    # `_start_ingests` runs every feed step and the window slides as cells are taken, so all
    # eight eventually start even when gated. What only holds when ingest is UNGATED is that
    # every start precedes the first finalization — gated, starts interleave with cleanups
    # because each new one waits for a slot that a cleanup frees.
    first_cleanup = next(i for i, e in enumerate(events) if e.startswith("cleanup:"))
    last_start = max(i for i, e in enumerate(events) if e.startswith("start:"))
    assert last_start < first_cleanup, (
        "an ingest started only after a cell was finalized, so ingest is still paced by the "
        f"fill: {events[: first_cleanup + 3]}"
    )
    assert len([e for e in events if e.startswith("start:")]) == 8
    # The change is to WHEN a slot frees, not to whether mosaics are deleted.
    assert inputs.alive == 0
    assert len([e for e in events if e.startswith("cleanup:")]) == 8


def test_failed_cell_releases_budget_slot_and_keeps_mosaic():
    """A failed zone retains its mosaic for staged resume but must give back
    its budget slot — a slot held forever would starve the feeder.
    """
    events: list[str] = []
    inputs = MosaicCountingInputs(events)
    with pytest.raises(RuntimeError, match="1/8 cell"):
        _run(_cells(8), session=_sync_session(fail={"01N"}), inputs=inputs, look_ahead=0)
    assert "cleanup:01N" not in events  # mosaic retained for the retry...
    # ...but every OTHER cell was still admitted and landed (no starvation).
    assert len([e for e in events if e.startswith("cleanup:")]) == 7


def test_every_cell_mismatching_the_session_orbit_still_completes():
    """Replaces a test of the deferral cap, which no longer exists.

    When mismatched cells were deferred they retained their mosaics against a bounded budget,
    and past that bound they FAILED with their mosaics deleted. A whole-run mismatch was
    therefore unfillable — which is exactly what a campaign meets when its zone list holds
    radar-free zones. Now every cell streams under its own orbit, so the run completes and the
    storage bound still holds.
    """
    events: list[str] = []
    inputs = MosaicCountingInputs(events)

    summary = _run(
        _cells(4),
        orbits=dict.fromkeys(("01N", "02N", "03N", "04N"), "ascending"),
        inputs=inputs,
        look_ahead=0,
    )
    assert summary["succeeded"] == 4, "a wholly-mismatched run must complete, not fail"
    assert summary["failed"] == 0
    # No storage-bound assertion here any more: that bound was deleted with the mosaic
    # budget. What this test is about — a wholly-mismatched run completing — is below.
    for z in ("01N", "02N", "03N", "04N"):
        assert f"cleanup:{z}" in events, f"{z}'s mosaic was never released"


def test_stop_unblocks_a_running_ingest_wait():
    """A crashed session must unwind the feeder even while it is blocked inside
    inputs.wait — the stop event is passed through and honored.
    """

    class BlockingInputs(RecordingInputs):
        def wait(self, zone: str, year: int, stop: threading.Event | None = None) -> None:
            assert stop is not None  # the runner must pass its unwind event
            while not stop.is_set():
                time.sleep(0.02)
            raise RuntimeError("aborted: runner stopping")

    def crashing_session(more_work, on_item_done):
        time.sleep(0.1)  # let the feeder reach the blocking wait first
        raise RuntimeError("cluster fell over")

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="cluster fell over"):
        _run(_cells(2), session=crashing_session, inputs=BlockingInputs([]), look_ahead=0)
    assert time.monotonic() - t0 < 30  # no 600s join-timeout hang


def test_a_mismatched_run_does_not_deadlock_the_feeder():
    """The deadlock this replaces came from deferrals sitting on budget slots.

    Streaming them removes the mechanism, but the feeder's slot discipline is what made the
    old bug possible, so the shape is still worth a test: a run where EVERY cell mismatches
    must drain rather than wedge, under a look-ahead that leaves the budget tight.
    """
    events: list[str] = []
    inputs = MosaicCountingInputs(events)
    result: dict = {}

    def target():
        result["summary"] = _run(
            _cells(4),
            orbits=dict.fromkeys(("01N", "02N", "03N", "04N"), "ascending"),
            inputs=inputs,
            look_ahead=1,
        )

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        pytest.fail("feeder deadlocked on a wholly orbit-mismatched run")
    assert result["summary"]["succeeded"] == 4


def test_systematic_failure_stops_feeder_at_retained_cap():
    """A run whose every INGEST fails must stop admitting at the cap.

    The deterministic half: an ingest failure raises inside ``inputs.wait`` on the feeder's
    own thread, so it is counted before the next cap check no matter how fast the fake runs.

    The cap is passed EXPLICITLY. It used to be ``look_ahead + 2``, and this test set
    ``look_ahead=1`` to get a cap of 3 — which is precisely the coupling removed on
    2026-08-27, when a failure budget of 7 derived from an ingest width took nine of ten
    clusters down in one provider outage.
    """
    events: list[str] = []
    inputs = FailingWaitInputs(events)
    with pytest.raises(RuntimeError, match="cell"):
        # attempts=1 disables the in-child retry pass, which calls `inputs.wait` too —
        # counting its attempts as admissions reported 6 for a feeder that stopped at 3.
        _run(_cells(12), inputs=inputs, look_ahead=1, max_retained_failures=3, attempts_per_cell_in_cluster=1)
    waited = len([e for e in events if e.startswith("wait:")])
    started = len([e for e in events if e.startswith("start:")])
    assert waited == 3, f"the feeder must stop admitting at the cap of 3, admitted {waited}"
    assert started == 12, "ingest is no longer gated by admission, so every cell starts"
    assert "cleanup:" not in "".join(events)


def test_failed_cells_are_counted_while_an_assembly_is_stuck():
    """A failed tally needs bookkeeping, not assembly, so it must not queue behind one.

    01N succeeds and its assembly is held by THIS thread for the whole run, so a failure
    counted meanwhile cannot have gone through the finalizer. Asserting WHICH THREAD counts,
    not how many cells were admitted — with instant fakes that count ranged 4 to 12.
    ``look_ahead=6`` keeps the admission bound and failure cap from stopping the feeder first.
    """
    hold = threading.Event()  # released by the test, never by the assembly itself
    reached_cap = threading.Event()
    counted: list[int] = [0]

    class _WatchCap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "retained for resume" in record.getMessage():
                counted[0] += 1
                if counted[0] >= 3:
                    reached_cap.set()

    log = logging.getLogger("test-stuck-assembly")
    handler = _WatchCap()
    log.addHandler(handler)

    def blocking_assemble(handoff, prep):
        hold.wait(timeout=30.0)
        return {"zone": handoff.zone, "empty": False, "succeeded": len(handoff.results)}

    def run() -> None:
        with contextlib.suppress(BaseException):  # every cell but 01N is meant to fail
            _run(
                _cells(8),
                session=_sync_session(fail={f"{i + 2:02d}N" for i in range(7)}),
                assemble=blocking_assemble,
                inputs=RecordingInputs([]),
                look_ahead=6,
                attempts_per_cell_in_cluster=1,
                log=log,
            )

    worker = threading.Thread(target=run, name="stuck-assembly-run", daemon=True)
    worker.start()
    try:
        counted_while_stuck = reached_cap.wait(timeout=10.0)
        # Snapshot BEFORE releasing: once the thread is free the queued tallies are counted
        # after all, and reading it later reports a healthy total for a blind run.
        n_while_stuck = counted[0]
    finally:
        hold.set()
        worker.join(timeout=30.0)
        log.removeHandler(handler)

    assert counted_while_stuck, f"only {n_while_stuck} cell(s) reached the cap while the assembly thread was held"


def test_inference_does_not_wait_for_assembly():
    """THE point of removing the admission bound: a stuck assembly must not stall inference.

    The assembly thread is held until every cell has inferred, which can only happen if
    admission is unbounded. Pre-change (``look_ahead=0`` → two slots released after assembly)
    the feeder stalls at cell 3 of 8 and every streamed cell fails in assembly.
    """
    n = 8
    all_streamed = threading.Event()
    assembled: list[str] = []

    def blocking_assemble(handoff, prep):
        if not all_streamed.wait(timeout=5.0):
            raise AssertionError("inference stalled behind assembly — admission is still bounded")
        assembled.append(handoff.zone)
        return {"zone": handoff.zone, "empty": False, "succeeded": len(handoff.results)}

    def session(more_work, on_item_done):
        results: list[dict] = []
        streamed: set[str] = set()
        while True:
            batch = more_work()
            if batch is None:
                return results
            for item in batch:
                result = {"chunk": item.chunk.label, "status": "success"}
                results.append(result)
                on_item_done(item, result)
                streamed.add(item.ctx.run_id.removeprefix("r-"))
            if len(streamed) == n:
                all_streamed.set()

    _run(_cells(n), session=session, assemble=blocking_assemble, inputs=RecordingInputs([]), look_ahead=0)

    assert len(assembled) == n, f"every cell must assemble once the hold lifts, got {assembled}"


def test_sporadic_failures_below_cap_do_not_stop_the_run():
    """A couple of failures under the cap must NOT halt the feeder — every cell
    is still attempted (only a systematic failure trips the early stop).
    """
    events: list[str] = []
    inputs = MosaicCountingInputs(events)
    with pytest.raises(RuntimeError, match="2/8 cell"):
        _run(_cells(8), session=_sync_session(fail={"03N", "06N"}), inputs=inputs, look_ahead=2)
    # All 8 admitted (2 failed, 6 cleaned) — the run wasn't cut short.
    assert len([e for e in events if e.startswith("start:")]) == 8
    assert len([e for e in events if e.startswith("cleanup:")]) == 6


def test_negative_look_ahead_rejected():
    """look_ahead < 0 would size the ingest driver's pool at zero width, so no cell
    would ever ingest — the runner rejects it up front.
    """
    with pytest.raises(ValueError, match="look_ahead must be >= 0"):
        _run(_cells(1), look_ahead=-1)


def test_feeder_crash_is_surfaced_not_silent_success():
    """An exception OUTSIDE the per-cell guards (here inputs.start, called by
    _start_lookahead) must fail the run — not silently kill the daemon feeder
    and let the drained partial queue look complete.
    """

    class ExplodingInputs(RecordingInputs):
        def start(self, zone: str, year: int) -> None:
            raise RuntimeError("feeder boom")

    with pytest.raises(RuntimeError, match="zone feeder crashed"):
        _run(_cells(3), inputs=ExplodingInputs([]), look_ahead=1)


class _StaggeredInputs(RecordingInputs):
    """Mosaics that land out of order, as ingest really finishes them.

    ``landed`` is what exists now. ``ready`` reports it truthfully. ``wait`` on a
    zone that has NOT landed records the block and then lets it through — which is
    what blocking really does — so a feeder that ignores readiness still completes
    but leaves a visible trail of stalls.
    """

    def __init__(self, events: list[str], landed: set[str]) -> None:
        super().__init__(events)
        self.landed = set(landed)
        self.blocked_on: list[str] = []

    def ready(self, zone: str, year: int) -> bool:
        return zone in self.landed

    def wait(self, zone: str, year: int, stop: threading.Event | None = None) -> None:
        super().wait(zone, year, stop)
        if zone not in self.landed:
            self.blocked_on.append(zone)
            self.landed.add(zone)  # the ingest we waited for finishes


def test_the_feeder_takes_a_landed_cell_over_its_head_cell():
    """The densest zone is ordered first AND ingests slowest, so insisting on it
    idles the fleet while smaller mosaics sit finished.

    Two conditions, not one: a cell must be in the look-ahead window to be
    CONSIDERED (only those have an ingest running), and must have COMPLETED to be
    taken. Inference never sees a partial mosaic.
    """
    events: list[str] = []
    inputs = _StaggeredInputs(events, landed={"02N", "03N", "04N"})  # everything but the head
    _run(_cells(4), inputs=inputs, look_ahead=2)
    order = [e.split(":")[1] for e in events if e.startswith("wait:")]
    assert order[0] == "02N", f"took its head instead of a landed cell: {order}"
    assert inputs.blocked_on == ["01N"], f"stalled on more than the one unlanded zone: {inputs.blocked_on}"


def test_every_cell_is_filled_exactly_once_when_taken_out_of_order():
    """Reordering must not drop, duplicate, or strand work."""
    events: list[str] = []
    inputs = _StaggeredInputs(events, landed={f"{i:02d}N" for i in range(1, 7)})
    result = _run(
        _cells(6), inputs=inputs, look_ahead=3, session=_sync_session(events=events), assemble=_assemble(events)
    )
    assembled = [e.split(":")[1] for e in events if e.startswith("assemble:")]
    assert sorted(assembled) == [f"{i:02d}N" for i in range(1, 7)]
    assert result["succeeded"] == 6 and result["failed"] == 0
    assert inputs.blocked_on == [], "nothing should have stalled: every mosaic had landed"


def test_nothing_landed_falls_back_to_strict_head_order():
    """The previous behaviour, preserved. A genuinely ingest-starved cluster
    blocks on its head rather than spinning over an all-unready window, so this
    change cannot make a starved run worse.
    """
    events: list[str] = []
    inputs = _StaggeredInputs(events, landed=set())
    _run(_cells(3), inputs=inputs, look_ahead=2)
    order = [e.split(":")[1] for e in events if e.startswith("wait:")]
    assert order == ["01N", "02N", "03N"], f"expected strict head order, got {order}"


def test_the_readiest_cell_is_taken_wherever_it_sits():
    """Whichever cell has LANDED is taken first, even the last in density order.

    The inverse of what this asserted before. A cell outside the look-ahead window used to
    be ineligible because its ingest had not been started, so the feeder had to walk the
    order and block on the head. Every pending cell's ingest now starts up front, so the
    only question left is which has landed — and waiting on a dense head while a sparse
    tail sits finished is exactly the idling this avoids.

    Density order still decides what is ASKED for first and still sizes the session; it is
    no longer a barrier to what can be taken.
    """
    events: list[str] = []
    # The LAST cell in density order is the only one that has landed.
    inputs = _StaggeredInputs(events, landed={"05N"})
    _run(_cells(5), inputs=inputs, look_ahead=1)
    order = [e.split(":")[1] for e in events if e.startswith("wait:")]
    assert order[0] == "05N", f"the landed cell must be taken first, order was {order}"


# --- in-child retry (attempts_per_cell_in_cluster) -----------------------------------------------


def _recovering_single(attempts: list[str], fail_zones: set[str] | None = None):
    """A per-cell path that records each attempt and succeeds unless told otherwise."""

    def infer_single(cell, prep, final):
        attempts.append(cell.zone)
        if fail_zones and cell.zone in fail_zones:
            raise RuntimeError(f"{cell.zone} is deterministically broken")
        return ZoneFillHandoff(
            zone=cell.zone, year=cell.year, run_id=prep.run_id, t0=0.0, summary={}, live=[], results=[]
        )

    return infer_single


def test_a_failed_cell_is_retried_in_child_and_recovers():
    """The whole point: recovery is LOCAL, without the driver dispatching anything.

    Without this the driver's retry unit is a whole dispatch, so a cell failing early
    waits for every cluster to finish its entire list before being re-attempted.
    """
    events: list[str] = []
    attempts: list[str] = []
    inputs = RecordingInputs(events)
    summary = _run(
        _cells(3),
        session=_sync_session(fail={"02N"}),
        inputs=inputs,
        infer_single=_recovering_single(attempts),
    )
    assert attempts == ["02N"], attempts
    assert summary["failed"] == 0 and summary["succeeded"] == 3
    # The recovered cell's mosaic is cleaned, like any cell that landed.
    assert "cleanup:02N" in events


class _FlakyIngestInputs(RecordingInputs):
    """Inputs whose FIRST production attempt per cell fails, like a transient ingest.

    Models the real adapter's memoisation: `start` is idempotent per (zone, year), so
    without a discard the failed attempt is what every later `wait` sees.
    """

    def __init__(self, events: list[str], fail_first: set[str]) -> None:
        super().__init__(events)
        self._fail_first = set(fail_first)
        self.attempts: dict[str, int] = {}
        self._live: set[str] = set()

    def start(self, zone: str, year: int) -> None:
        """Idempotent per cell, as the real adapter is — it memoises the future."""
        with self._lock:
            if zone in self._live:
                return
            self._live.add(zone)
            self.attempts[zone] = self.attempts.get(zone, 0) + 1
            self.events.append(f"start:{zone}")

    def ready(self, zone: str, year: int) -> bool:
        return True

    def wait(self, zone: str, year: int, stop=None) -> None:
        with self._lock:
            self.events.append(f"wait:{zone}")
            failing = zone in self._fail_first and self.attempts.get(zone, 0) <= 1
        if failing:
            raise RuntimeError(f"ingest deployment failed for {zone}")

    def discard(self, zone: str, year: int) -> None:
        with self._lock:
            self._live.discard(zone)
            self.events.append(f"discard:{zone}")


def test_an_ingest_failure_is_re_ingested_on_retry():
    """The attempt budget has to buy a NEW ingest, not a re-read of the dead one.

    `start` is idempotent, so a cell that failed while producing its inputs keeps that
    failure cached — and a retry that only re-plans would probe the same missing mosaic
    every time, making `attempts_per_cell_in_cluster` worthless for exactly the transient ingest
    failures it exists to absorb.
    """
    events: list[str] = []
    attempts: list[str] = []
    inputs = _FlakyIngestInputs(events, fail_first={"02N"})

    summary = _run(_cells(3), session=_sync_session(), inputs=inputs, infer_single=_recovering_single(attempts))

    assert inputs.attempts["02N"] == 2, "the retry must submit a second ingest"
    assert "discard:02N" in events, "the dead attempt must be dropped before restarting"
    assert summary["failed"] == 0 and summary["succeeded"] == 3


def test_a_retry_after_inference_does_not_re_ingest():
    """A cell that failed AFTER its inputs landed keeps the mosaic it retained.

    Retention is the whole reason those cells stay eligible, so re-running their ingest
    would redo work already on disk and re-admit budget the retention keeps out.
    """
    events: list[str] = []
    attempts: list[str] = []
    inputs = RecordingInputs(events)

    _run(_cells(3), session=_sync_session(fail={"02N"}), inputs=inputs, infer_single=_recovering_single(attempts))

    assert attempts == ["02N"], "the inference retry ran"
    assert "discard:02N" not in events, "an inference failure must not discard a good mosaic"


def test_a_deterministic_failure_survives_its_retries_and_still_raises():
    """A retry that cannot help must not swallow the failure."""
    attempts: list[str] = []
    with pytest.raises(RuntimeError, match="1/3 cell"):
        _run(
            _cells(3),
            session=_sync_session(fail={"02N"}),
            inputs=RecordingInputs([]),
            infer_single=_recovering_single(attempts, fail_zones={"02N"}),
            attempts_per_cell_in_cluster=3,
        )
    # Attempted once per retry round (2 of 3 attempts are retries), not once forever.
    assert attempts == ["02N", "02N"], attempts


def test_attempts_per_cell_in_cluster_of_one_disables_the_retry():
    """The bound is real, and 1 means the previous behaviour exactly."""
    attempts: list[str] = []
    with pytest.raises(RuntimeError, match="1/2 cell"):
        _run(
            _cells(2),
            session=_sync_session(fail={"01N"}),
            inputs=RecordingInputs([]),
            infer_single=_recovering_single(attempts),
            attempts_per_cell_in_cluster=1,
        )
    assert attempts == [], "no retry may run at attempts_per_cell_in_cluster=1"


def test_the_queued_ingests_are_cancelled_before_the_retry_pass():
    """The retry is only prompt if the ingest queue is cleared FIRST.

    The runner cannot see the pool, so the contract is ordering: ask before the retry
    re-starts anything, or the retry's own `start` lands at the back of the queue it cleared.
    """
    events: list[str] = []
    attempts: list[str] = []

    class _ReingestFailing(RecordingInputs):
        """01N's ingest fails, so the retry takes the re-ingest path."""

        def wait(self, zone: str, year: int, stop: threading.Event | None = None) -> None:
            with self._lock:
                self.events.append(f"wait:{zone}")
            if zone == "01N" and "discard:01N" not in self.events:
                raise RuntimeError("ingest failed for 01N")

    with contextlib.suppress(RuntimeError):
        _run(
            _cells(2),
            inputs=_ReingestFailing(events),
            infer_single=_recovering_single(attempts),
            attempts_per_cell_in_cluster=2,
        )

    assert "cancel_unstarted" in events, "the runner must clear the ingest queue before retrying"
    assert events.index("cancel_unstarted") < events.index("discard:01N"), (
        f"the queue must be cleared BEFORE the retry re-starts an ingest: {events}"
    )


def test_the_in_child_retry_only_considers_cells_that_kept_a_mosaic():
    """The bug this caught in review, pinned so it cannot come back.

    Retry eligibility is "kept its mosaic", NOT "failed": a first version keyed off the
    failure list and silently "recovered" cells whose input had been deleted, running them
    against nothing.

    The scenario used to be built with the orbit-mismatch deferral cap, which deleted the
    mosaics of cells past the cap. That path is gone — cells now stream under their own orbit
    — and every remaining failure path RETAINS its mosaic, so the two sets currently coincide.
    What is pinned here is therefore the filter itself: the retry pass must consult the
    retained set rather than the failure list, so that a future delete-on-failure path cannot
    reintroduce the bug by simply existing.
    """
    src = pathlib.Path(mod.__file__).read_text() if hasattr(mod, "__file__") and mod.__file__ else ""
    assert "pending = failed_keys if inputs is None else [k for k in failed_keys if k in eligible]" in src, (
        "the retry pass must filter the failure list by the retained-mosaic set"
    )
    assert "eligible = set(retained_failed)" in src


# ---------------------------------------------------------------------------
# Withholding supply from a fleet that is already up (the starvation drill)
# ---------------------------------------------------------------------------


def _armed_withhold(hold_minutes: float):
    return FaultInjection(fault=WITHHOLD_WORK, hold_minutes=hold_minutes).arm(
        ssm_prefix="/global-tessera-dev/ray/", supports=(WITHHOLD_WORK,), log=LOG
    )


def _polling_session(polls: list, poll_interval_s: float = 0.02):
    """A session that records every poll and keeps polling through empty ones.

    The real scheduler behaves this way while its source is unexhausted: an empty
    poll is "nothing ready YET", so it waits and asks again rather than finishing.
    That loop is what a withheld supply exploits, and what this fake has to
    reproduce for the test to mean anything.
    """

    def session(more_work, on_item_done):
        results = []
        while True:
            batch = more_work()
            polls.append(batch)
            if batch is None:
                return results
            if not batch:
                time.sleep(poll_interval_s)
                continue
            for item in batch:
                result = {"chunk": item.chunk.label, "status": "success"}
                results.append(result)
                on_item_done(item, result)

    return session


def test_withholding_starves_the_stream_and_then_lets_it_finish(caplog):
    # The drill's whole claim: the fleet is left holding nothing, ALIVE, and the run
    # still completes. A fault that failed the run would exercise the failure path
    # instead of the starvation the tripwire exists to see.
    polls: list = []
    with caplog.at_level(logging.ERROR):
        summary = _run(_cells(2), session=_polling_session(polls), fault=_armed_withhold(0.004))

    assert summary["succeeded"] == 2 and summary["failed"] == 0, "withholding must not fail a cell"
    handed_over = [batch for batch in polls if batch]
    assert len(handed_over) == 2, "both zones must reach the stream once the hold ends"
    said = [r.getMessage() for r in caplog.records if FAULT_LOG_PREFIX in r.getMessage()]
    assert any("FIRING withhold_work" in line for line in said), "the hold must announce itself"
    assert any("RELEASED" in line for line in said), "and must end by itself"


def test_the_first_zone_is_handed_over_before_anything_is_withheld(caplog):
    # A fleet that never received work has not starved; it has never started, which is
    # the shape every detector exempts. So the hold must begin AFTER a hand-over.
    polls: list = []
    with caplog.at_level(logging.ERROR):
        _run(_cells(2), session=_polling_session(polls), fault=_armed_withhold(0.004))
    first_non_empty = next(i for i, batch in enumerate(polls) if batch)
    said = [r.getMessage() for r in caplog.records if "FIRING withhold_work" in r.getMessage()]
    assert said, "the hold must have fired at all"
    assert first_non_empty < len(polls) - 1, "a zone must be streamed before the hold starts"


def test_an_unarmed_run_never_consults_a_fault(monkeypatch):
    # The property that matters more than the fault working: with no request passed,
    # the withholding code is unreachable. Booby-trap it and run an ordinary fill.
    def _must_not_run(self, supply, *, log):
        raise AssertionError("an unarmed run consulted the fault injection")

    monkeypatch.setattr(ArmedFault, "withhold", _must_not_run, raising=True)
    summary = _run(_cells(2))
    assert summary["succeeded"] == 2


# --- the operator pause ---------------------------------------------------------------
#
# Pausing inference holds at the work source, which is the same site the starvation drill
# withholds from. Two properties decide whether it is a pause or an outage, and both are
# about what the source returns: `[]` keeps the session alive with its actors, while `None`
# means exhausted and would retire the fleet and finalize the run. And a paused poll must
# not CONSULT the source, because a hand-over removes the prepared zone from the queue.


def _pausing_session(pause_for_polls: int, seen: list[str]):
    """A session that polls the source, recording what it got, pausing for the first N polls."""

    def session(more_work, on_item_done):
        results: list[dict] = []
        polls = 0
        while True:
            batch = more_work()
            polls += 1
            seen.append("none" if batch is None else ("empty" if not batch else f"work:{len(batch)}"))
            if batch is None:
                return results
            for item in batch:
                result = {"chunk": item.chunk.label, "status": "success"}
                results.append(result)
                on_item_done(item, result)
            if polls > 200:  # pragma: no cover - a stuck pause would hang the suite
                raise AssertionError("source never exhausted")

    return session


def test_a_pause_holds_the_stream_without_ending_it(monkeypatch):
    """The whole mechanism: while paused the source yields nothing, and the moment the pause
    lifts the waiting cell streams. Nothing fails and no cell is skipped.
    """
    seen: list[str] = []
    polls = {"n": 0}

    def paused() -> bool:
        polls["n"] += 1
        return polls["n"] <= 3  # paused for the first three polls, then released

    out = _run(_cells(2), session=_pausing_session(0, seen), paused=paused)
    assert seen[:3] == ["empty", "empty", "empty"], seen
    # Nothing was lost to the pause: both cells still streamed and assembled.
    assert out["succeeded"] == 2 and out["failed"] == 0, out
    assert "none" in seen  # and the source still reached exhaustion afterwards


def test_a_paused_poll_never_reports_exhaustion(monkeypatch):
    """`None` is a teardown, not a pause: it retires the actors and finalizes the run, so a
    cell held behind a pause would be finalized as complete having inferred nothing. The
    pause must answer "nothing ready YET" no matter what the source would have said.
    """
    seen: list[str] = []
    calls = {"n": 0}

    def paused() -> bool:
        calls["n"] += 1
        return calls["n"] <= 5

    _run(_cells(1), session=_pausing_session(0, seen), paused=paused)
    # The first five answers are the pause, and not one of them is `none`.
    assert seen[:5] == ["empty"] * 5, seen


def test_a_paused_poll_does_not_consume_the_prepared_cell():
    """A hand-over REMOVES the zone from the ready queue, so a pause that asked the source
    and discarded the answer would delete prepared work rather than delay it. The cell must
    still be there when the pause lifts — which is what makes a pause free.
    """
    seen: list[str] = []
    state = {"paused": True}

    def paused() -> bool:
        return state["paused"]

    def session(more_work, on_item_done):
        results: list[dict] = []
        # Poll a few times while paused; every answer must be empty and nothing consumed.
        for _ in range(4):
            assert more_work() == [], "a paused poll handed work over"
        state["paused"] = False
        while True:
            batch = more_work()
            if batch is None:
                return results
            for item in batch:
                result = {"chunk": item.chunk.label, "status": "success"}
                results.append(result)
                on_item_done(item, result)

    out = _run(_cells(1), session=session, paused=paused)
    assert out["succeeded"] == 1 and out["failed"] == 0, out


def test_no_pause_callable_means_no_check_at_all():
    """Every path with no gate configured — a direct single-cluster run, a test — must not
    pay a check or change behaviour.
    """
    seen: list[str] = []
    out = _run(_cells(1), session=_pausing_session(0, seen))
    assert out["succeeded"] == 1
    assert "empty" not in seen[:1]


def test_the_cap_is_not_derived_from_look_ahead():
    """The decoupling itself. `look_ahead` sizes INGEST WIDTH; the cap is a FAILURE BUDGET.

    Tying them meant a fleet configured for 3-wide ingest silently accepted a failure budget
    of 5, and one exogenous provider outage on 2026-08-27 then converted into a fleet-wide
    teardown. A wide ingest and a patient failure budget are different decisions.
    """
    import inspect

    params = inspect.signature(mod.fill_zones_sequential).parameters
    assert params["max_retained_failures"].default == 30, (
        "the default is a deliberate ceiling against a SYSTEMATIC fault, not a bad hour"
    )
    events: list[str] = []
    inputs = FailingWaitInputs(events)
    # look_ahead=1 would have forced a cap of 3 under the old derivation; it must not now.
    with pytest.raises(RuntimeError, match="cell"):
        _run(_cells(8), inputs=inputs, look_ahead=1, max_retained_failures=6, attempts_per_cell_in_cluster=1)
    waited = len([e for e in events if e.startswith("wait:")])
    assert waited == 6, f"the cap must be the one passed, not look_ahead + 2; admitted {waited}"


def test_hitting_the_cap_ends_the_run_without_draining_the_assembly_queue():
    """A cap trip ENDS the run instead of draining it — asserted by the CLOCK.

    The point of stopping is to get the unattempted cells back to the driver for
    re-dispatch, and draining first delays exactly that: measured 2026-08-27, cells whose
    optical leg had died stayed open for HOURS behind work that could not change the
    outcome. Mosaics and staged tiles are retained, so a re-dispatch resumes rather than
    rebuilds — which is what makes abandoning the queue cheap enough to be right.

    Asserted by finishing while an assembly is STILL BLOCKED: if the run drained, it could
    not have returned. A timing assertion is used because "did not wait" has no other
    observable — the executor's shutdown arguments are not visible from here.
    """
    release = threading.Event()  # never set until the run has already returned
    entered = threading.Event()

    def blocking_assemble(handoff, *a, **k):
        entered.set()
        release.wait(timeout=30)
        return _assemble()(handoff, *a, **k)

    class _FailAfter(RecordingInputs):
        """Succeeds for the first two cells so assemblies actually QUEUE, then fails —
        which is what lets a cap trip and a non-empty assembly queue coexist.
        """

        def wait(self, zone: str, year: int, stop: threading.Event | None = None) -> None:
            with self._lock:
                self.events.append(f"wait:{zone}")
                n = len([e for e in self.events if e.startswith("wait:")])
            if n > 2:
                raise RuntimeError(f"ingest failed for {zone}-{year}")

    events: list[str] = []
    inputs = _FailAfter(events)
    done = threading.Event()
    box: list[BaseException] = []

    def go():
        try:
            _run(
                _cells(8),
                inputs=inputs,
                look_ahead=1,
                max_retained_failures=2,
                attempts_per_cell_in_cluster=1,
                assemble=blocking_assemble,
            )
        except BaseException as exc:
            box.append(exc)
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    finished = done.wait(timeout=15)
    release.set()
    assert finished, "the run drained the assembly queue instead of ending at the cap"
    assert box and isinstance(box[0], RuntimeError)
