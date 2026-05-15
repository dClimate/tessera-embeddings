"""Post-mortem diagnostics for failed Ray inference workers.

Provides helpers that query CloudWatch Logs for a failed instance's
telemetry and produce concise summaries so operators can triage failures
without manually searching through hundreds of log lines.

Current diagnostics:
    - **Memory ramp**: parses ResourceMonitor RESOURCES lines into a
      compact table showing RAM/GPU/VRAM evolution leading up to a failure
      (e.g. OOM kills, memory-related actor deaths).

Future diagnostics can follow the same pattern: query the relevant
CloudWatch stream, parse domain-specific log lines, and surface a
summary in the flow runner logs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import boto3

logger = logging.getLogger(__name__)

# CloudWatch log group used by the Ray cluster CloudWatch agent
RAY_LOG_GROUP = "/ec2/yield/ray"


@dataclass
class ResourceSnapshot:
    """A single resource monitor reading parsed from a CloudWatch log line."""

    timestamp: str  # "HH:MM:SS" (UTC)
    ram_used_gb: float
    ram_total_gb: float
    ram_pct: int
    gpu_pct: str
    vram: str
    phase: str  # inferred from surrounding log context

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


def _query_cloudwatch_events(
    instance_id: str,
    stream_suffix: str = "actors",
    filter_pattern: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Query CloudWatch Logs for events from a specific instance stream.

    Args:
        instance_id: EC2 instance ID (e.g., "i-086dcd20878e7d0ae").
        stream_suffix: Log stream suffix ("actors", "raylet", "workers", etc.).
        filter_pattern: Optional CloudWatch filter pattern.
        limit: Maximum number of events to return.

    Returns:
        List of event dicts with 'message' and 'timestamp' keys.
    """
    client = boto3.client("logs", region_name="us-west-2")
    stream_name = f"{instance_id}/{stream_suffix}"

    kwargs: dict = {
        "logGroupName": RAY_LOG_GROUP,
        "logStreamNames": [stream_name],
        "limit": limit,
        "interleaved": False,
    }
    if filter_pattern:
        kwargs["filterPattern"] = filter_pattern

    events: list[dict] = []
    try:
        paginator = client.get_paginator("filter_log_events")
        for page in paginator.paginate(**kwargs):
            events.extend(page.get("events", []))
    except client.exceptions.ResourceNotFoundException:
        logger.debug("CloudWatch stream %s not found in %s", stream_name, RAY_LOG_GROUP)
    except Exception:
        logger.debug("CloudWatch query failed for %s", stream_name, exc_info=True)

    return events


def _parse_resource_snapshots(events: list[dict]) -> list[ResourceSnapshot]:
    """Parse RESOURCES lines from CloudWatch events into structured snapshots."""
    # First pass: build a timeline of phases from ALL events
    phase_transitions: list[tuple[str, str]] = []  # (time, phase)
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

    # Second pass: parse RESOURCES lines and assign phases
    snapshots: list[ResourceSnapshot] = []
    for ev in events:
        msg = ev.get("message", "")
        m = _RESOURCE_RE.search(msg)
        if not m:
            continue

        t = m.group("time")

        # Determine phase: find the latest phase transition at or before this timestamp
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
    """Select key snapshots that show meaningful RAM changes.

    Keeps: first, last, phase transitions, and any snapshot where RAM%
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
) -> str | None:
    """Build a diagnostic summary for a failed worker.

    Queries CloudWatch for the instance's resource monitor history and
    produces a memory ramp table plus the CLI command for full logs.

    Args:
        instance_id: EC2 instance ID of the failed worker.
        chunk_label: Label of the chunk that failed.
        error_msg: Error message from the failure.

    Returns:
        Formatted diagnostic string, or None if CloudWatch data unavailable.
    """
    if not instance_id or instance_id.startswith("unknown") or instance_id.startswith("pending"):
        return None

    events = _query_cloudwatch_events(instance_id, stream_suffix="actors")
    if not events:
        return None

    snapshots = _parse_resource_snapshots(events)
    if not snapshots:
        return None

    key_snapshots = _select_key_snapshots(snapshots)
    table = _format_memory_ramp_table(key_snapshots)

    diagnostic = (
        f"\n--- Worker failure diagnostic: {chunk_label} on {instance_id} ---\n"
        f"Error: {error_msg}\n"
        f"\nMemory ramp (key moments from {len(snapshots)} readings):\n"
        f"{table}\n"
        f"\nFor full resource logs:\n"
        f"  aws --profile cyclops-west --region us-west-2 logs filter-log-events \\\n"
        f'    --log-group-name "{RAY_LOG_GROUP}" \\\n'
        f'    --log-stream-names "{instance_id}/actors" \\\n'
        f'    --filter-pattern "RESOURCES" \\\n'
        f"    --output json | jq -r '.events[].message'\n"
        f"\nFor all actor logs (data loading, inference, errors):\n"
        f"  aws --profile cyclops-west --region us-west-2 logs filter-log-events \\\n"
        f'    --log-group-name "{RAY_LOG_GROUP}" \\\n'
        f'    --log-stream-names "{instance_id}/actors" \\\n'
        f"    --output json | jq -r '.events[].message'\n"
        f"--- end diagnostic ---"
    )
    return diagnostic


def log_worker_failure_diagnostic(
    instance_id: str,
    chunk_label: str,
    error_msg: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> None:
    """Query CloudWatch and log a diagnostic summary for a failed worker.

    This is the main entry point called from the scheduling loop. Catches
    all exceptions internally so it never disrupts the inference pipeline.

    Args:
        instance_id: EC2 instance ID of the failed worker.
        chunk_label: Label of the chunk that failed.
        error_msg: Error message from the failure.
        log: Logger to write the diagnostic to.
    """
    try:
        diagnostic = build_worker_failure_diagnostic(instance_id, chunk_label, error_msg)
        if diagnostic:
            log.warning(diagnostic)
        else:
            log.info(
                "No CloudWatch diagnostics available for %s on %s (stream may not exist or instance ID unresolved)",
                chunk_label,
                instance_id,
            )
    except Exception:
        log.debug("Failed to build worker failure diagnostic", exc_info=True)
