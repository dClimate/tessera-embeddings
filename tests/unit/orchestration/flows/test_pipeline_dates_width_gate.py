"""The width bound on date pipelining (``MAX_PIPELINE_DATES_WORKERS``).

Overlapping a date's preparation with the previous date's write only pays while the write
leaves the fleet room to absorb the coverage gate, which is worker-side work. Above a
measured width the overlap costs more than it saves, so the flow declines to pipeline rather
than obeying a flag that would silently make the run slower — a slower run looks like a
slower run, so the failure would never surface on its own.

These tests pin the DECISION, not the number: every case derives its widths from the constant
so re-calibrating the bound cannot leave a test asserting a stale width.
"""

from __future__ import annotations

from tessera_embeddings.orchestration.prefect.flows.ingest_s2_roi_reflectance import (
    MAX_PIPELINE_DATES_WORKERS,
    _gated_pipeline_dates,
)

BOUND = MAX_PIPELINE_DATES_WORKERS


class _Log:
    """Minimal logger: the flow's real one needs a Prefect run context we do not want here."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args: object) -> None:
        self.warnings.append(msg % args if args else msg)


def _decide(**kwargs: object) -> tuple[bool, list[str]]:
    log = _Log()
    got = _gated_pipeline_dates(log=log, **kwargs)  # type: ignore[arg-type]
    return got, log.warnings


def test_declines_above_the_bound() -> None:
    got, warnings = _decide(pipeline_dates=True, use_local=False, max_workers=BOUND + 1)
    assert got is False
    assert len(warnings) == 1
    assert "WITHOUT pipelining" in warnings[0]
    assert str(BOUND + 1) in warnings[0], "the warning must name the width that triggered it"


def test_allows_at_the_bound() -> None:
    """The bound is inclusive — exactly at it, pipelining still runs, silently."""
    got, warnings = _decide(pipeline_dates=True, use_local=False, max_workers=BOUND)
    assert got is True
    assert warnings == []


def test_allows_well_below_the_bound() -> None:
    got, warnings = _decide(pipeline_dates=True, use_local=False, max_workers=60)
    assert got is True
    assert warnings == []


def test_silent_when_pipelining_was_not_requested() -> None:
    """A wide fleet that never asked for pipelining must not be warned at."""
    got, warnings = _decide(pipeline_dates=False, use_local=False, max_workers=BOUND + 100)
    assert got is False
    assert warnings == []


def test_local_path_is_not_gated() -> None:
    """On use_local, max_workers provisions no workers, so it must not decide anything."""
    got, warnings = _decide(pipeline_dates=True, use_local=True, max_workers=BOUND + 100)
    assert got is True
    assert warnings == []
