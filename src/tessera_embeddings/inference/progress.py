"""Lightweight Ray actor for tracking batch-level progress of GPU inference actors.

Runs on the head node (num_cpus=0) and receives fire-and-forget updates from GPU actors.
The flow runner polls it periodically to log progress and detect stalls.

Staleness is computed inside the actor (not the flow runner) because time.monotonic()
is per-machine — the Fargate flow runner and Ray head node have unrelated clocks.

Usage:
    tracker = ProgressTracker.remote()
    # GPU actor (fire-and-forget):
    tracker.report.remote(chunk_label, batch_idx, total_batches, "inference")
    # Flow runner (blocking poll):
    progress = ray.get(tracker.get_all.remote())
"""

from __future__ import annotations

import time

import ray


def chunk_uid(run_id: str, chunk_label: str) -> str:
    """The run-qualified progress key: ``"{run_id}:{chunk_label}"``.

    Chunk labels are grid-relative positions (e.g. ``"5_5"``) that REPEAT across zones, so a
    chained multi-zone session can hold two zones' identically-labelled chunks at once during
    tail/head overlap; unqualified, one zone's completion evicts the other's entry and hangs a
    live GPU task. :attr:`WorkItem.uid` builds its retry key here too, so the actor's ``report``
    and the scheduler's ``remove`` cannot drift.
    """
    return f"{run_id}:{chunk_label}"


@ray.remote(num_cpus=0, memory=10 * 1024 * 1024)  # 10 MB
class ProgressTracker:
    """Holds per-chunk progress state, updated by GPU actors.

    Each entry is a dict keyed by the run-qualified chunk uid (:func:`chunk_uid`)
    with values ``(batch_idx, total_batches, staleness_sec, phase)``.
    """

    def __init__(self) -> None:
        self._progress: dict[str, tuple[int, int, float, str]] = {}

    def report(self, uid: str, batch_idx: int, total_batches: int, phase: str = "inference") -> None:
        """Update progress for a chunk (keyed by :func:`chunk_uid`). Fire-and-forget from GPU actors."""
        self._progress[uid] = (batch_idx, total_batches, time.monotonic(), phase)

    def get_all(self) -> dict[str, tuple[int, int, float, str]]:
        """Return progress with staleness computed on the tracker's own clock.

        Returns dict of {uid: (batch_idx, total_batches, staleness_sec, phase)}.
        """
        now = time.monotonic()
        return {uid: (batch, total, now - ts, phase) for uid, (batch, total, ts, phase) in self._progress.items()}

    def remove(self, uid: str) -> None:
        """Remove a completed chunk from tracking (keyed by :func:`chunk_uid`)."""
        self._progress.pop(uid, None)
