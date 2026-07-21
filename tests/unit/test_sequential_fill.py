"""fill_zones_sequential: ordering, look-ahead, trailing assembly, breaker.

The runner is Prefect/Ray-free by contract, so these tests patch its two
zone-fill phase imports (``infer_zone_year`` / ``assemble_zone_year``) at the
runner's own namespace and drive the loop with recording fakes — no cluster,
no store.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

import tessera_embeddings.orchestration.runners.sequential_fill as mod
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


def _wire(monkeypatch, events: list[str], *, infer=None, assemble=None):
    """Patch the two phases with recorders (overridable per test)."""

    def default_infer(**kw):
        events.append(f"infer:{kw['zone']}")
        return _handoff(kw["zone"])

    def default_assemble(handoff, **kw):
        events.append(f"assemble:{handoff.zone}")
        return {"zone": handoff.zone, "empty": False}

    monkeypatch.setattr(mod, "infer_zone_year", infer or default_infer)
    monkeypatch.setattr(mod, "assemble_zone_year", assemble or default_assemble)


def _run(cells, *, inputs=None, look_ahead=2, **kw) -> dict[str, Any]:
    return fill_zones_sequential(
        cells=cells,
        store_path="s3://store",
        land_mask_path="s3://mask",
        prepare=_prepare,
        log=LOG,
        inputs=inputs,
        look_ahead=look_ahead,
        **kw,
    )


def test_cells_processed_in_order_and_all_succeed(monkeypatch):
    events: list[str] = []
    _wire(monkeypatch, events)
    summary = _run(_cells(3))
    assert [e for e in events if e.startswith("infer:")] == ["infer:01N", "infer:02N", "infer:03N"]
    assert summary["succeeded"] == 3 and summary["failed"] == 0
    # Every cell was assembled (none were terminal handoffs).
    assert sorted(e for e in events if e.startswith("assemble:")) == ["assemble:01N", "assemble:02N", "assemble:03N"]


def test_lookahead_ingests_run_ahead_of_the_inference_head(monkeypatch):
    events: list[str] = []
    _wire(monkeypatch, events)
    _run(_cells(4), inputs=RecordingInputs(events), look_ahead=2)
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


def test_retirement_enabled_only_on_the_final_cell(monkeypatch):
    flags: list[bool] = []

    def infer(**kw):
        flags.append(kw["retire_idle_actors"])
        return _handoff(kw["zone"])

    _wire(monkeypatch, [], infer=infer)
    _run(_cells(3))
    assert flags == [False, False, True]


def test_trailing_assembly_overlaps_next_inference(monkeypatch):
    """Cell 2's inference must start while cell 1's assembly is still running —
    the whole point of the trailing thread. The blocked assembly is released
    only by cell 2's inference starting, so a serialized implementation would
    deadlock (caught by the join timeout) rather than pass by accident.
    """
    next_inference_started = threading.Event()
    events: list[str] = []

    def infer(**kw):
        events.append(f"infer:{kw['zone']}")
        if kw["zone"] == "02N":
            next_inference_started.set()
        return _handoff(kw["zone"])

    def assemble(handoff, **kw):
        if handoff.zone == "01N":
            assert next_inference_started.wait(timeout=10), "assembly 1 finished before inference 2 began"
        events.append(f"assemble:{handoff.zone}")
        return {"zone": handoff.zone}

    _wire(monkeypatch, events, infer=infer, assemble=assemble)
    summary = _run(_cells(2))
    assert summary["succeeded"] == 2
    assert events.index("infer:02N") < events.index("assemble:01N")


def test_terminal_handoff_skips_assembly(monkeypatch):
    events: list[str] = []

    def infer(**kw):
        events.append(f"infer:{kw['zone']}")
        return _handoff(kw["zone"], done={"zone": kw["zone"], "already_complete": True})

    _wire(monkeypatch, events, infer=infer)
    summary = _run(_cells(1), inputs=RecordingInputs(events))
    assert summary["succeeded"] == 1
    assert not [e for e in events if e.startswith("assemble:")]
    assert "cleanup:01N" in events  # terminal cells still release their mosaic


def test_single_cell_failure_recorded_and_run_continues(monkeypatch):
    events: list[str] = []

    def infer(**kw):
        events.append(f"infer:{kw['zone']}")
        if kw["zone"] == "02N":
            raise RuntimeError("boom")
        return _handoff(kw["zone"])

    _wire(monkeypatch, events, infer=infer)
    inputs = RecordingInputs(events)
    with pytest.raises(RuntimeError, match="1/3 cell"):
        _run(_cells(3), inputs=inputs)
    # The failure did not stop the later cell, and the failed cell's mosaic
    # was kept (its retry needs it for the fingerprinted run_id + resume).
    assert "infer:03N" in events
    assert "cleanup:02N" not in events
    assert {"cleanup:01N", "cleanup:03N"} <= set(events)


def test_consecutive_failures_trip_the_breaker(monkeypatch):
    calls: list[str] = []

    def infer(**kw):
        calls.append(kw["zone"])
        raise RuntimeError("systemic")

    _wire(monkeypatch, [], infer=infer)
    with pytest.raises(RuntimeError, match="consecutive"):
        _run(_cells(5), max_consecutive_failures=2)
    assert calls == ["01N", "02N"]  # the 3rd cell is never attempted


def test_assembly_failure_keeps_mosaic_and_fails_run(monkeypatch):
    events: list[str] = []

    def assemble(handoff, **kw):
        raise RuntimeError("commit refused")

    _wire(monkeypatch, events, assemble=assemble)
    inputs = RecordingInputs(events)
    with pytest.raises(RuntimeError, match="assembly"):
        _run(_cells(1), inputs=inputs)
    assert "cleanup:01N" not in events


def test_prepare_failure_is_a_prepare_phase_failure(monkeypatch):
    _wire(monkeypatch, [])

    def prepare(cell):
        raise ValueError("coverage gate")

    with pytest.raises(RuntimeError) as exc_info:
        fill_zones_sequential(
            cells=_cells(1),
            store_path="s3://store",
            land_mask_path="s3://mask",
            prepare=prepare,
            log=LOG,
            max_consecutive_failures=5,
        )
    assert "'phase': 'prepare'" in str(exc_info.value)


def test_no_inputs_adapter_means_no_lifecycle_calls(monkeypatch):
    events: list[str] = []
    _wire(monkeypatch, events)
    summary = _run(_cells(2), inputs=None)
    assert summary["succeeded"] == 2
    assert not [e for e in events if e.startswith(("start:", "wait:", "cleanup:"))]
