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


@ray.remote(num_cpus=0, memory=10 * 1024 * 1024)  # 10 MB
class ProgressTracker:
    """Holds per-chunk progress state, updated by GPU actors.

    Each entry is a dict keyed by chunk_label with values:
    ``(batch_idx, total_batches, staleness_sec, phase)``.
    """

    def __init__(self) -> None:
        self._progress: dict[str, tuple[int, int, float, str]] = {}

    def report(self, chunk_label: str, batch_idx: int, total_batches: int, phase: str = "inference") -> None:
        """Update progress for a chunk. Called fire-and-forget from GPU actors."""
        self._progress[chunk_label] = (batch_idx, total_batches, time.monotonic(), phase)

    def get_all(self) -> dict[str, tuple[int, int, float, str]]:
        """Return progress with staleness computed on the tracker's own clock.

        Returns dict of {chunk_label: (batch_idx, total_batches, staleness_sec, phase)}.
        """
        now = time.monotonic()
        return {label: (batch, total, now - ts, phase) for label, (batch, total, ts, phase) in self._progress.items()}

    def remove(self, chunk_label: str) -> None:
        """Remove a completed chunk from tracking."""
        self._progress.pop(chunk_label, None)
