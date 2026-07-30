"""fill_zones_sequential: the chained stream — feeder, tallies, trailing assembly.

The runner is pure orchestration over injected callables (Prefect/Ray-free by
contract). These tests drive it with a synchronous fake ``session`` that
drains ``more_work`` and fires ``on_item_done`` per item — no cluster, no
store, no monkeypatching. The staged-resume scan is the one internal
dependency, stubbed at the module namespace.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

import tessera_embeddings.orchestration.runners.sequential_fill as mod
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
    """Minimal config stand-in (orbit + the scan's compute_std probe)."""

    def __init__(self, s1_orbit: str = "both") -> None:
        self.s1_orbit = s1_orbit
        self.compute_std = False


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


class BudgetProbeInputs(RecordingInputs):
    """RecordingInputs that also tracks the started-not-cleaned high-water mark —
    the mosaic count the budget is supposed to bound.
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


def _run(cells, *, orbits=None, plan=None, session=None, assemble=None, infer_single=_no_fallback, **kw):
    return fill_zones_sequential(
        cells=cells,
        prepare=_prepare_for(orbits),
        plan=plan or _plan_for(),
        session=session or _sync_session(),
        assemble=assemble or _assemble(),
        infer_single=infer_single,
        session_s1_orbit="both",
        log=LOG,
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


def test_orbit_mismatch_defers_to_fallback():
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
    assert summary["deferred_orbit_mismatch"] == 1
    assert calls == [("02N", True)]  # ran after the stream, flagged final
    assert not [e for e in events if e.startswith("infer:r-02N")]  # never streamed
    assert summary["succeeded"] == 3


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
# Mosaic budget: ingest starts admitted through the storage bound
# ---------------------------------------------------------------------------


def test_mosaic_budget_bounds_started_not_cleaned_mosaics():
    """Look-ahead ingest starts go through the mosaic budget, so mosaics alive
    on storage (started, not yet cleaned) never exceed look_ahead + 2 — they
    used to escape the bound and peak at ~2*look_ahead + 2.
    """
    events: list[str] = []
    inputs = BudgetProbeInputs(events)
    _run(_cells(8), inputs=inputs, look_ahead=1)
    assert inputs.high_water <= 3  # look_ahead + 2
    # Everything still landed and was cleaned.
    assert inputs.alive == 0
    assert len([e for e in events if e.startswith("cleanup:")]) == 8


def test_failed_cell_releases_budget_slot_and_keeps_mosaic():
    """A failed zone retains its mosaic for staged resume but must give back
    its budget slot — a slot held forever would starve the feeder.
    """
    events: list[str] = []
    inputs = BudgetProbeInputs(events)
    with pytest.raises(RuntimeError, match="1/8 cell"):
        _run(_cells(8), session=_sync_session(fail={"01N"}), inputs=inputs, look_ahead=0)
    assert "cleanup:01N" not in events  # mosaic retained for the retry...
    # ...but every OTHER cell was still admitted and landed (no starvation).
    assert len([e for e in events if e.startswith("cleanup:")]) == 7


def test_deferred_mosaics_capped_with_loud_failure():
    """Orbit-mismatch deferrals retain mosaics until the post-session fallback,
    so they are capped at look_ahead + 1 budget slots; mismatch cells beyond
    the cap fail loudly (and their mosaics are cleaned) instead of silently
    stacking multi-TB mosaics for the whole run.
    """
    events: list[str] = []
    inputs = BudgetProbeInputs(events)
    calls: list[str] = []

    def infer_single(cell, prep, final):
        calls.append(cell.zone)

        return ZoneFillHandoff(
            zone=cell.zone, year=cell.year, run_id=prep.run_id, t0=0.0, summary={}, live=[], results=[]
        )

    # ALL cells mismatch the session orbit; look_ahead=0 → cap = 1 retained.
    with pytest.raises(RuntimeError) as exc_info:
        _run(
            _cells(4),
            orbits=dict.fromkeys(("01N", "02N", "03N", "04N"), "ascending"),
            inputs=inputs,
            infer_single=infer_single,
            look_ahead=0,
        )
    assert "deferred-mosaic-budget" in str(exc_info.value)
    assert "'02N'" in str(exc_info.value) and "'01N'" not in str(exc_info.value)
    assert calls == ["01N"]  # the retained deferral still ran via fallback
    assert inputs.high_water <= 2  # look_ahead + 2
    # Capped cells' mosaics cleaned immediately; the deferred one after fallback.
    for z in ("01N", "02N", "03N", "04N"):
        assert f"cleanup:{z}" in events


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


def test_lookahead_does_not_deadlock_on_early_deferrals():
    """THE deadlock regression (PR #90 review, HIGH): two early orbit-mismatch
    deferrals retain budget slots and a started look-ahead cell holds the third;
    the feeder must then block only on the CURRENT cell's slot (whose processing
    frees slots), never on a future look-ahead slot — blocking there wedged the
    feeder forever before the fix.
    """
    events: list[str] = []
    inputs = BudgetProbeInputs(events)
    calls: list[str] = []

    def infer_single(cell, prep, final):
        calls.append(cell.zone)

        return ZoneFillHandoff(
            zone=cell.zone, year=cell.year, run_id=prep.run_id, t0=0.0, summary={}, live=[], results=[]
        )

    result: dict = {}

    def target():
        result["summary"] = _run(
            _cells(4),
            orbits={"01N": "ascending", "02N": "ascending"},  # cap (look_ahead+1=2) NOT exceeded
            inputs=inputs,
            infer_single=infer_single,
            look_ahead=1,
        )

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        pytest.fail("feeder deadlocked: blocked on a future cell's budget slot with deferrals holding the rest")
    assert result["summary"]["succeeded"] == 4  # 2 streamed + 2 fallback
    assert result["summary"]["deferred_orbit_mismatch"] == 2
    assert calls == ["01N", "02N"]
    assert inputs.high_water <= 3  # the bound still held throughout


def test_systematic_failure_stops_feeder_at_retained_cap():
    """A run that fails EVERY cell must not free-then-retain every shard's mosaic:
    once retained failures reach look_ahead + 2, the feeder stops admitting, so
    peak retained mosaics stays bounded rather than growing to the whole shard.
    """
    events: list[str] = []
    inputs = BudgetProbeInputs(events)
    with pytest.raises(RuntimeError, match="cell"):
        # 12 cells, all fail inference; look_ahead=1 → retained-failure cap = 3.
        _run(_cells(12), session=_sync_session(fail={f"{i + 1:02d}N" for i in range(12)}), inputs=inputs, look_ahead=1)
    # The feeder stopped admitting once the cap was reached, so it never started
    # every shard's ingest. Retained-but-uncleaned mosaics DO exceed the active
    # budget by design (kept for resume) but stay bounded ~ cap + in-flight, NOT
    # the whole 12-cell shard.
    started = len([e for e in events if e.startswith("start:")])
    assert started < 12  # stopped early — the essential guarantee
    assert started <= 2 * (1 + 2) + 2  # bounded ~ cap + in-flight, not shard size
    assert inputs.high_water == started  # nothing cleaned — every started mosaic retained
    assert "cleanup:" not in "".join(events)


def test_sporadic_failures_below_cap_do_not_stop_the_run():
    """A couple of failures under the cap must NOT halt the feeder — every cell
    is still attempted (only a systematic failure trips the early stop).
    """
    events: list[str] = []
    inputs = BudgetProbeInputs(events)
    with pytest.raises(RuntimeError, match="2/8 cell"):
        _run(_cells(8), session=_sync_session(fail={"03N", "06N"}), inputs=inputs, look_ahead=2)
    # All 8 admitted (2 failed, 6 cleaned) — the run wasn't cut short.
    assert len([e for e in events if e.startswith("start:")]) == 8
    assert len([e for e in events if e.startswith("cleanup:")]) == 6


def test_negative_look_ahead_rejected():
    """look_ahead < 0 would size the budget / zone_slots at <= 0 capacity and
    deadlock the feeder — the runner rejects it up front.
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


def test_only_the_started_window_is_eligible():
    """A cell beyond the look-ahead has no ingest running, so it cannot be picked
    however 'ready' it might look — jumping to it would start a mosaic outside the
    budget that admits them.
    """
    events: list[str] = []
    # The LAST cell has landed; the window (look_ahead=1 -> 2 cells) has not.
    inputs = _StaggeredInputs(events, landed={"05N"})
    _run(_cells(5), inputs=inputs, look_ahead=1)
    order = [e.split(":")[1] for e in events if e.startswith("wait:")]
    assert order[0] == "01N", f"reached outside its window: {order}"


# --- in-child retry (max_cell_attempts) -----------------------------------------------


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


def test_a_deterministic_failure_survives_its_retries_and_still_raises():
    """A retry that cannot help must not swallow the failure."""
    attempts: list[str] = []
    with pytest.raises(RuntimeError, match="1/3 cell"):
        _run(
            _cells(3),
            session=_sync_session(fail={"02N"}),
            inputs=RecordingInputs([]),
            infer_single=_recovering_single(attempts, fail_zones={"02N"}),
            max_cell_attempts=3,
        )
    # Attempted once per retry round (2 of 3 attempts are retries), not once forever.
    assert attempts == ["02N", "02N"], attempts


def test_max_cell_attempts_one_disables_the_retry():
    """The bound is real, and 1 means the previous behaviour exactly."""
    attempts: list[str] = []
    with pytest.raises(RuntimeError, match="1/2 cell"):
        _run(
            _cells(2),
            session=_sync_session(fail={"01N"}),
            inputs=RecordingInputs([]),
            infer_single=_recovering_single(attempts),
            max_cell_attempts=1,
        )
    assert attempts == [], "no retry may run at max_cell_attempts=1"


def test_a_cell_whose_mosaic_was_deleted_is_not_retried():
    """The bug this caught in review, pinned so it cannot come back.

    Two failure paths delete the mosaic on purpose — the orbit-mismatch deferral cap and
    a terminal plan — so that a systematic failure cannot stack multi-terabyte inputs.
    Retrying one of those runs against nothing. Eligibility is "kept its mosaic", which is
    NOT the same as "failed": a first version of the retry pass used the failure list and
    silently "recovered" cells whose input had been deleted.
    """
    events: list[str] = []
    attempts: list[str] = []
    inputs = BudgetProbeInputs(events)
    # Every cell mismatches the session orbit and look_ahead=0 caps deferrals at 1, so the
    # cells past the cap fail with their mosaics CLEANED.
    with pytest.raises(RuntimeError, match="deferral cap reached"):
        _run(
            _cells(4),
            orbits=dict.fromkeys(("01N", "02N", "03N", "04N"), "ascending"),
            inputs=inputs,
            look_ahead=0,
            infer_single=_recovering_single(attempts),
        )
    # The capped cells were cleaned, so they must NOT appear as retry attempts. Only the
    # one legitimately-deferred cell may reach infer_single, via the fallback pass.
    assert len(attempts) <= 1, attempts


def test_retry_runs_before_the_fallback_so_actors_are_still_alive():
    """Ordering is load-bearing: the fallback marks its last cell `final`, which retires
    idle actors. A retry scheduled after that would need actors the fallback released.
    """
    order: list[str] = []

    def infer_single(cell, prep, final):
        order.append(f"{cell.zone}:final={final}")
        return ZoneFillHandoff(
            zone=cell.zone, year=cell.year, run_id=prep.run_id, t0=0.0, summary={}, live=[], results=[]
        )

    _run(
        _cells(2),
        orbits={"02N": "ascending"},  # 02N defers to the fallback
        session=_sync_session(fail={"01N"}),  # 01N fails in the stream → retried
        inputs=RecordingInputs([]),
        infer_single=infer_single,
    )
    assert order == ["01N:final=False", "02N:final=True"], order
