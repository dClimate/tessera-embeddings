"""Post-mortem diagnostics for failed Ray inference workers.

Parses ResourceMonitor log lines into structured snapshots and formats a
memory-ramp table. Log retrieval is pluggable: pass a ``fetch_events``
callable to ``build_worker_failure_diagnostic``; the AWS provider supplies
a CloudWatch-backed implementation, and the default is a no-op that returns
an empty list (stdlib logging already captured the live output).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """A single resource monitor reading parsed from a log line."""

    timestamp: str  # "HH:MM:SS" (UTC)
    ram_used_gb: float
    ram_total_gb: float
    ram_pct: int
    gpu_pct: str
    vram: str
    phase: str

    @property
    def ram_summary(self) -> str:
        """Format RAM usage as 'used/total GB (pct%)'."""
        return f"{self.ram_used_gb:.1f}/{self.ram_total_gb:.1f} GB ({self.ram_pct}%)"


# Regex for the RESOURCES log line emitted by ResourceMonitor
_RESOURCE_RE = re.compile(
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r".*RESOURCES:.*"
    r"RAM=(?P<used>[\d.]+)/(?P<total>[\d.]+)\s*GB\s*\((?P<pct>\d+)%\)"
    r".*GPU=(?P<gpu>\d+%)"
    r".*VRAM=(?P<vram>[^\|]+)"
)

# Regex for phase-change log lines
_PHASE_RE = re.compile(
    r"(?P<time>\d{2}:\d{2}:\d{2}).*?"
    r"(?:"
    r"(?P<loading>Loading chunk)|"
    r"(?P<loaded>Loaded chunk)|"
    r"(?P<inference>Starting inference)|"
    r"(?P<batch>Batch \d+/\d+)"
    r")"
)


def _parse_resource_snapshots(events: list[dict]) -> list[ResourceSnapshot]:
    """Parse RESOURCES lines from log events into structured snapshots."""
    phase_transitions: list[tuple[str, str]] = []
    for ev in events:
        msg = ev.get("message", "")
        m = _PHASE_RE.search(msg)
        if m:
            t = m.group("time")
            if m.group("loading"):
                phase_transitions.append((t, "loading"))
            elif m.group("loaded"):
                phase_transitions.append((t, "loaded"))
            elif m.group("inference"):
                phase_transitions.append((t, "inference"))

    snapshots: list[ResourceSnapshot] = []
    for ev in events:
        msg = ev.get("message", "")
        m = _RESOURCE_RE.search(msg)
        if not m:
            continue

        t = m.group("time")
        phase = "idle"
        for pt_time, pt_phase in phase_transitions:
            if pt_time <= t:
                phase = pt_phase

        snapshots.append(
            ResourceSnapshot(
                timestamp=t,
                ram_used_gb=float(m.group("used")),
                ram_total_gb=float(m.group("total")),
                ram_pct=int(m.group("pct")),
                gpu_pct=m.group("gpu"),
                vram=m.group("vram").strip(),
                phase=phase,
            )
        )

    return snapshots


def _select_key_snapshots(snapshots: list[ResourceSnapshot]) -> list[ResourceSnapshot]:
    """Select snapshots that show meaningful RAM changes.

    Keeps first, last, phase transitions, and any snapshot where RAM%
    changed by >= 3 points from the previous kept snapshot.
    """
    if len(snapshots) <= 6:
        return snapshots

    selected: list[ResourceSnapshot] = [snapshots[0]]
    last_pct = snapshots[0].ram_pct
    last_phase = snapshots[0].phase

    for snap in snapshots[1:-1]:
        pct_jump = abs(snap.ram_pct - last_pct) >= 3
        phase_change = snap.phase != last_phase
        if pct_jump or phase_change:
            selected.append(snap)
            last_pct = snap.ram_pct
            last_phase = snap.phase

    selected.append(snapshots[-1])
    return selected


def _format_memory_ramp_table(snapshots: list[ResourceSnapshot]) -> str:
    """Format key snapshots into a compact ASCII table."""
    lines = [
        "Time(UTC) | RAM              | GPU  | VRAM           | Phase",
        "--------- | ---------------- | ---- | -------------- | --------",
    ]
    for s in snapshots:
        lines.append(f"{s.timestamp} | {s.ram_summary:<16s} | {s.gpu_pct:<4s} | {s.vram:<14s} | {s.phase}")
    return "\n".join(lines)


def build_worker_failure_diagnostic(
    instance_id: str,
    chunk_label: str,
    error_msg: str,
    fetch_events: Callable[[str], list[dict]] | None = None,
) -> str | None:
    """Build a diagnostic summary for a failed worker.

    Args:
        instance_id: Identifier for the failed worker (e.g. EC2 instance ID).
        chunk_label: Label of the chunk that failed.
        error_msg: Error message from the failure.
        fetch_events: Optional callable ``(instance_id) -> list[dict]`` where
            each dict has at least a ``"message"`` key. The AWS provider passes
            a CloudWatch-backed implementation; ``None`` means no remote log
            fetch (live stdout logs are the fallback).

    Returns:
        Formatted diagnostic string, or ``None`` if no log data is available.
    """
    if not instance_id or instance_id.startswith("unknown") or instance_id.startswith("pending"):
        return None

    if fetch_events is None:
        return None

    events = fetch_events(instance_id)
    if not events:
        return None

    snapshots = _parse_resource_snapshots(events)
    if not snapshots:
        return None

    key_snapshots = _select_key_snapshots(snapshots)
    table = _format_memory_ramp_table(key_snapshots)

    return (
        f"\n--- Worker failure diagnostic: {chunk_label} on {instance_id} ---\n"
        f"Error: {error_msg}\n"
        f"\nMemory ramp (key moments from {len(snapshots)} readings):\n"
        f"{table}\n"
        f"--- end diagnostic ---"
    )


def log_worker_failure_diagnostic(
    instance_id: str,
    chunk_label: str,
    error_msg: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    fetch_events: Callable[[str], list[dict]] | None = None,
) -> None:
    """Log a diagnostic summary for a failed worker.

    Args:
        instance_id: Identifier for the failed worker.
        chunk_label: Label of the chunk that failed.
        error_msg: Error message from the failure.
        log: Logger to write the diagnostic to.
        fetch_events: Optional event-fetch callable (see
            :func:`build_worker_failure_diagnostic`). When ``None``, logs a
            brief "no diagnostics available" message instead.
    """
    try:
        diagnostic = build_worker_failure_diagnostic(
            instance_id, chunk_label, error_msg, fetch_events=fetch_events
        )
        if diagnostic:
            log.warning(diagnostic)
        else:
            log.info(
                "No remote diagnostics available for %s on %s",
                chunk_label,
                instance_id,
            )
    except Exception:
        log.debug("Failed to build worker failure diagnostic", exc_info=True)
