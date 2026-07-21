"""fill_zones_sequential: ordering, look-ahead, trailing assembly, breaker.

The runner is pure sequencing over injected callables (Prefect/Ray-free by
contract), so these tests drive the loop with recording fakes passed straight
through the ``prepare``/``infer``/``assemble`` parameters — no cluster, no
store, no monkeypatching.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

from tessera_embeddings.orchestration.runners.sequential_fill import (
    PreparedCell,
    SequentialCell,
    fill_zones_sequential,
)
from tessera_embeddings.orchestration.runners.zone_fill import ZoneFillHandoff

LOG = logging.getLogger("test-sequential-fill")


def _cells(n: int) -> list[SequentialCell]:
    return [SequentialCell(zone=f"{i + 1:02d}N", year=2025, num_actors=5) for i in range(n)]


def _prepare(cell: SequentialCell) -> PreparedCell:
    return PreparedCell(
        mosaic_base=f"m/{cell.zone}",
        staging_base=f"s/{cell.zone}",
        run_id=f"r-{cell.zone}",
        config=object(),  # opaque to the runner; only threaded through
    )


def _handoff(zone: str, done: dict[str, Any] | None = None) -> ZoneFillHandoff:
    return ZoneFillHandoff(
        zone=zone, year=2025, run_id=f"r-{zone}", t0=0.0, summary={"zone": zone}, live=[], results=[], done=done
    )


class RecordingInputs:
    """CellInputs fake appending every call to a shared event list."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def start(self, zone: str, year: int) -> None:
        # Idempotence is the adapter's contract; record only true starts so
        # the look-ahead assertions see the runner's intent, not its retries.
        if f"start:{zone}" not in self.events:
            self.events.append(f"start:{zone}")

    def wait(self, zone: str, year: int) -> None:
        self.events.append(f"wait:{zone}")

    def cleanup(self, zone: str, year: int) -> None:
        self.events.append(f"cleanup:{zone}")


def _run(cells, events: list[str] | None = None, *, infer=None, assemble=None, **kw) -> dict[str, Any]:
    """Drive the runner with recording default phases (overridable per test)."""
    events = events if events is not None else []

    def default_infer(cell, prep, final):
        events.append(f"infer:{cell.zone}")
        return _handoff(cell.zone)

    def default_assemble(handoff, prep):
        events.append(f"assemble:{handoff.zone}")
        return {"zone": handoff.zone, "empty": False}

    return fill_zones_sequential(
        cells=cells,
        prepare=_prepare,
        infer=infer or default_infer,
        assemble=assemble or default_assemble,
        log=LOG,
        **kw,
    )


def test_cells_processed_in_order_and_all_succeed():
    events: list[str] = []
    summary = _run(_cells(3), events)
    assert [e for e in events if e.startswith("infer:")] == ["infer:01N", "infer:02N", "infer:03N"]
    assert summary["succeeded"] == 3 and summary["failed"] == 0
    # Every cell was assembled (none were terminal handoffs).
    assert sorted(e for e in events if e.startswith("assemble:")) == ["assemble:01N", "assemble:02N", "assemble:03N"]


def test_lookahead_ingests_run_ahead_of_the_inference_head():
    events: list[str] = []
    _run(_cells(4), events, inputs=RecordingInputs(events), look_ahead=2)
    # Cells 1-3 (current + 2 look-ahead) are started before the first wait...
    first_wait = events.index("wait:01N")
    assert set(events[:first_wait]) >= {"start:01N", "start:02N", "start:03N"}
    # ...and cell 4's ingest starts before cell 2's inference finishes (i.e.
    # once the head advances past cell 1) — never all up front.
    assert "start:04N" not in events[:first_wait]
    assert events.index("start:04N") < events.index("infer:02N")
    # Each landed cell's mosaic is cleaned up.
    cleaned = {e for e in events if e.startswith("cleanup:")}
    assert cleaned == {"cleanup:01N", "cleanup:02N", "cleanup:03N", "cleanup:04N"}


def test_final_cell_flag_set_only_on_the_last_cell():
    flags: list[bool] = []

    def infer(cell, prep, final):
        flags.append(final)
        return _handoff(cell.zone)

    _run(_cells(3), infer=infer)
    assert flags == [False, False, True]


def test_trailing_assembly_overlaps_next_inference():
    """Cell 2's inference must start while cell 1's assembly is still running —
    the whole point of the trailing thread. The blocked assembly is released
    only by cell 2's inference starting, so a serialized implementation would
    deadlock (caught by the join timeout) rather than pass by accident.
    """
    next_inference_started = threading.Event()
    events: list[str] = []

    def infer(cell, prep, final):
        events.append(f"infer:{cell.zone}")
        if cell.zone == "02N":
            next_inference_started.set()
        return _handoff(cell.zone)

    def assemble(handoff, prep):
        if handoff.zone == "01N":
            assert next_inference_started.wait(timeout=10), "assembly 1 finished before inference 2 began"
        events.append(f"assemble:{handoff.zone}")
        return {"zone": handoff.zone}

    summary = _run(_cells(2), events, infer=infer, assemble=assemble)
    assert summary["succeeded"] == 2
    assert events.index("infer:02N") < events.index("assemble:01N")


def test_terminal_handoff_skips_assembly():
    events: list[str] = []

    def infer(cell, prep, final):
        events.append(f"infer:{cell.zone}")
        return _handoff(cell.zone, done={"zone": cell.zone, "already_complete": True})

    summary = _run(_cells(1), events, infer=infer, inputs=RecordingInputs(events))
    assert summary["succeeded"] == 1
    assert not [e for e in events if e.startswith("assemble:")]
    assert "cleanup:01N" in events  # terminal cells still release their mosaic


def test_single_cell_failure_recorded_and_run_continues():
    events: list[str] = []

    def infer(cell, prep, final):
        events.append(f"infer:{cell.zone}")
        if cell.zone == "02N":
            raise RuntimeError("boom")
        return _handoff(cell.zone)

    with pytest.raises(RuntimeError, match="1/3 cell"):
        _run(_cells(3), events, infer=infer, inputs=RecordingInputs(events))
    # The failure did not stop the later cell, and the failed cell's mosaic
    # was kept (its retry needs it for the fingerprinted run_id + resume).
    assert "infer:03N" in events
    assert "cleanup:02N" not in events
    assert {"cleanup:01N", "cleanup:03N"} <= set(events)


def test_consecutive_failures_trip_the_breaker():
    calls: list[str] = []

    def infer(cell, prep, final):
        calls.append(cell.zone)
        raise RuntimeError("systemic")

    with pytest.raises(RuntimeError, match="consecutive"):
        _run(_cells(5), infer=infer, max_consecutive_failures=2)
    assert calls == ["01N", "02N"]  # the 3rd cell is never attempted


def test_assembly_failure_keeps_mosaic_and_fails_run():
    events: list[str] = []

    def assemble(handoff, prep):
        raise RuntimeError("commit refused")

    with pytest.raises(RuntimeError, match="assembly"):
        _run(_cells(1), events, assemble=assemble, inputs=RecordingInputs(events))
    assert "cleanup:01N" not in events


def test_prepare_failure_is_a_prepare_phase_failure():
    def prepare(cell):
        raise ValueError("coverage gate")

    def unreachable(*a, **k):
        raise AssertionError("phases must not run when prepare fails")

    with pytest.raises(RuntimeError) as exc_info:
        fill_zones_sequential(cells=_cells(1), prepare=prepare, infer=unreachable, assemble=unreachable, log=LOG)
    assert "'phase': 'prepare'" in str(exc_info.value)


def test_no_inputs_adapter_means_no_lifecycle_calls():
    events: list[str] = []
    summary = _run(_cells(2), events, inputs=None)
    assert summary["succeeded"] == 2
    assert not [e for e in events if e.startswith(("start:", "wait:", "cleanup:"))]
