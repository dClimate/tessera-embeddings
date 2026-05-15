"""Work-stealing chunk scheduler for distributed Ray inference.

Manages actor lifecycle (replacement on death, idle retirement) and
dispatches chunks across a pool of InferenceActor handles.

NOTE this code is highly adapted to a chunk-based workflow so I decided
to keep it within src/inference. However it could reasonably live within
infra/ray if we decide to adopt Ray for other distributed GPU tasks
that work at the level of chunks. This
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import cast

import ray

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.inference.actors import InferenceActor
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.diagnostics import log_worker_failure_diagnostic


class ActorPool:
    """Encapsulates mutable actor state and lifecycle operations.

    Replaces the two closures (_resolve_instance_id, _submit_to_actor) and the
    inline actor-replacement / idle-retirement blocks in the original main loop.
    """

    def __init__(
        self,
        actors: list[ray.actor.ActorHandle],
        actor_instance_ids: list[str],
        config: InferenceConfig,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger],
        *,
        idle_grace_sec: int = 120,
        on_retire: Callable[[str], None] | None = None,
    ) -> None:
        """Initialise the pool from already-ready actor handles.

        Args:
            actors: Ready Ray actor handles (mutated in-place on replacement).
            actor_instance_ids: EC2 instance ID per actor slot (parallel list).
            config: Inference config used when spawning replacement actors.
            log: Logger.
            idle_grace_sec: Seconds an actor may be idle before it is killed.
            on_retire: Optional callback invoked with the EC2 instance ID after
                an idle actor is killed. Used to terminate the underlying EC2
                instance immediately rather than waiting for the autoscaler.
        """
        self.actors = actors
        self.actor_instance_ids = actor_instance_ids
        self.config = config
        self.log = log
        self.idle_grace_sec = idle_grace_sec
        self._on_retire = on_retire

        self.actor_deaths: int = 0
        self.max_actor_deaths: int = len(actors)

        # ref → (chunk_label, actor_idx)
        self.pending: dict[ray.ObjectRef, tuple[str, int]] = {}
        self.chunk_attempts: dict[str, int] = {}

        self._pending_iid_refs: dict[int, ray.ObjectRef] = {}
        self._initializing: set[int] = set()
        self._retired: set[int] = set()
        self._idle_since: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def busy_actors(self) -> set[int]:
        """Set of actor indices that currently have in-flight work."""
        return {aidx for _, aidx in self.pending.values()}

    @property
    def live_count(self) -> int:
        """Number of non-retired actor slots."""
        return len(self.actors) - len(self._retired)

    # ------------------------------------------------------------------
    # Instance-ID resolution
    # ------------------------------------------------------------------

    def resolve_iid(self, actor_idx: int, timeout: float = 0) -> None:
        """Try to resolve a pending instance-ID fetch for a replacement actor.

        Non-blocking by default (timeout=0): returns immediately if not ready.
        Called before error reporting or dispatch so the instance ID is
        available for logs without stalling the main loop.

        Args:
            actor_idx: Index into self.actors.
            timeout: ray.wait timeout in seconds (0 = non-blocking).
        """
        if actor_idx not in self._pending_iid_refs:
            return
        ready, _ = ray.wait([self._pending_iid_refs[actor_idx]], timeout=timeout)
        if ready:
            try:
                self.actor_instance_ids[actor_idx] = ray.get(ready[0])
                self.log.info(
                    "Replacement actor %d resolved to instance %s",
                    actor_idx,
                    self.actor_instance_ids[actor_idx],
                )
            except Exception as exc:
                self.log.debug("Instance ID fetch failed for actor %d: %s", actor_idx, exc)
                self.actor_instance_ids[actor_idx] = "unknown (instance ID fetch failed)"
            del self._pending_iid_refs[actor_idx]
            self._initializing.discard(actor_idx)

    def resolve_initializing(self) -> int:
        """Try to resolve instance IDs for all initializing actors.

        Actors whose ``get_instance_id`` call has completed are moved out of
        ``_initializing``, making them eligible for ``dispatch_idle``.  Called
        each main-loop iteration so that late-joining actors start receiving
        work as soon as they're confirmed alive.

        Returns:
            Number of actors that transitioned from initializing to ready.
        """
        resolved = 0
        for actor_idx in list(self._initializing):
            self.resolve_iid(actor_idx)
            if actor_idx not in self._initializing:
                resolved += 1
        return resolved

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def submit(
        self,
        actor_idx: int,
        chunk: ChunkSpec,
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None,
    ) -> None:
        """Submit a single chunk to an actor and record the pending future.

        Args:
            actor_idx: Index into self.actors.
            chunk: Chunk to process.
            mosaic_base: Base path for mosaic stores.
            staging_base: Base path for staged output.
            run_id: Run identifier.
            tracker: Optional ProgressTracker actor handle.
        """
        self.resolve_iid(actor_idx)  # best-effort resolve before dispatch
        ref: ray.ObjectRef = self.actors[actor_idx].process_chunk.remote(  # type: ignore[union-attr]
            chunk, mosaic_base, staging_base, run_id, tracker=tracker
        )
        self.pending[ref] = (chunk.label, actor_idx)
        self.chunk_attempts[chunk.label] = self.chunk_attempts.get(chunk.label, 0) + 1

    def seed(
        self,
        chunk_queue: deque[ChunkSpec],
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None,
    ) -> None:
        """Submit one chunk to each actor to prime the work-stealing loop.

        Args:
            chunk_queue: Queue of remaining chunks (popleft in-place).
            mosaic_base: Base path for mosaic stores.
            staging_base: Base path for staged output.
            run_id: Run identifier.
            tracker: Optional ProgressTracker actor handle.
        """
        for actor_idx in range(len(self.actors)):
            if not chunk_queue:
                break
            if actor_idx in self._initializing:
                continue
            self.submit(actor_idx, chunk_queue.popleft(), mosaic_base, staging_base, run_id, tracker)
        self.log.info("Seeded %d actors with initial chunks (%d queued)", len(self.pending), len(chunk_queue))

    def dispatch_idle(
        self,
        chunk_queue: deque[ChunkSpec],
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None,
    ) -> None:
        """Dispatch queued chunks to any idle, non-initializing actors.

        Covers cases where a replacement actor was skipped in the main loop —
        other idle actors can pick up the work immediately.

        Args:
            chunk_queue: Queue of remaining chunks (popleft in-place).
            mosaic_base: Base path for mosaic stores.
            staging_base: Base path for staged output.
            run_id: Run identifier.
            tracker: Optional ProgressTracker actor handle.
        """
        busy = self.busy_actors
        while chunk_queue:
            idle_ready = [
                i
                for i in range(len(self.actors))
                if i not in busy and i not in self._initializing and i not in self._retired
            ]
            if not idle_ready:
                break
            idx = idle_ready[0]
            self.submit(idx, chunk_queue.popleft(), mosaic_base, staging_base, run_id, tracker)
            busy.add(idx)

    # ------------------------------------------------------------------
    # Actor lifecycle
    # ------------------------------------------------------------------

    def mark_initializing(self, actor_idx: int, placeholder_iid: str = "pending-init") -> None:
        """Mark an actor slot as still initializing.

        Fires a non-blocking ``get_instance_id.remote()`` so the real EC2
        instance ID is resolved lazily via ``resolve_iid()`` once the actor
        is live. Used both for late-joining actors (from ``_wait_for_actors``)
        and for replacement actors spawned after a death.

        Args:
            actor_idx: Slot index.
            placeholder_iid: Placeholder string for the instance-ID list
                until the real ID is resolved.
        """
        self._pending_iid_refs[actor_idx] = self.actors[actor_idx].get_instance_id.remote()  # type: ignore[union-attr]
        self.actor_instance_ids[actor_idx] = placeholder_iid
        self._initializing.add(actor_idx)

    def replace(self, actor_idx: int, instance_id: str) -> None:
        """Spawn a replacement actor for a dead slot and log the death count.

        Fires a non-blocking get_instance_id.remote() so the new EC2 instance
        ID is resolved lazily before the next dispatch or error report.

        Args:
            actor_idx: Slot index of the dead actor.
            instance_id: Instance ID of the actor that died (for log context).
        """
        self.actor_deaths += 1
        if self.actor_deaths >= self.max_actor_deaths:
            self.log.critical(
                "!!! %d actor deaths — every actor slot has died at least once. "
                "This is systemic (bad checkpoint? memory leak? instance type too small?). "
                "KILL THIS FLOW and investigate before burning more GPU spend.",
                self.actor_deaths,
            )
        elif self.actor_deaths >= self.max_actor_deaths // 2:
            self.log.error(
                "!!! %d / %d actor slots have died — possible systemic issue. "
                "Monitor closely or consider killing this flow.",
                self.actor_deaths,
                len(self.actors),
            )
        else:
            self.log.warning(
                "Replacing dead actor %d (was on %s) [death #%d]",
                actor_idx,
                instance_id,
                self.actor_deaths,
            )

        new_actor = InferenceActor.remote(self.config, self.config.checkpoint_path)  # type: ignore[attr-defined]
        self.actors[actor_idx] = new_actor
        self.mark_initializing(actor_idx, placeholder_iid=f"pending-replacement-of-{instance_id}")

    def retire_idle(self, outstanding_work: int) -> None:
        """Kill actors that have been idle past the grace period.

        Never kills an actor if it would leave fewer live actors than there
        is outstanding work (floor check).

        Args:
            outstanding_work: len(pending) + len(chunk_queue) at call time.
        """
        busy = self.busy_actors
        now = time.monotonic()
        live = self.live_count

        # Grace period gives actors a window to pick up new work before
        # being retired — avoids churn from momentary idleness between
        # chunks, and keeps spare capacity available when in-flight chunks
        # fail and get re-queued.
        grace = self.idle_grace_sec

        for actor_idx in range(len(self.actors)):
            # Skip actors that are busy, still spinning up, or already retired
            if actor_idx in busy or actor_idx in self._initializing or actor_idx in self._retired:
                self._idle_since.pop(actor_idx, None)
                continue

            # First time seeing this actor idle — start the clock.
            if actor_idx not in self._idle_since:
                self._idle_since[actor_idx] = now
                continue

            # Still within the grace period — let it live
            if now - self._idle_since.get(actor_idx, now) < grace:
                continue

            # Don't kill if it would leave fewer live actors than remaining work
            if live - 1 < outstanding_work:
                continue

            self.resolve_iid(actor_idx)  # pick up lazily-resolved EC2 instance ID
            instance_id = self.actor_instance_ids[actor_idx] if actor_idx < len(self.actor_instance_ids) else "unknown"
            with contextlib.suppress(Exception):
                ray.kill(self.actors[actor_idx])
            self._retired.add(actor_idx)
            self._idle_since.pop(actor_idx, None)
            live -= 1
            self.log.info("Killed idle actor %d (instance %s) — releasing GPU node", actor_idx, instance_id)
            if self._on_retire is not None and instance_id.startswith("i-"):
                with contextlib.suppress(Exception):
                    self._on_retire(instance_id)


# ---------------------------------------------------------------------------
# Tracker polling
# ---------------------------------------------------------------------------


def _poll_tracker(
    tracker: ray.actor.ActorHandle,
    n_done: int,
    n_total: int,
    stall_threshold_sec: float,
    max_simultaneous_stalls: int,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> None:
    """Poll ProgressTracker; log stalls; raise RuntimeError on systemic stall.

    This function has no access to pool or loop state — all context is passed
    explicitly so it can be tested in isolation.

    Args:
        tracker: ProgressTracker Ray actor handle.
        n_done: Number of chunks already completed (for the progress log line).
        n_total: Total chunks in the run.
        stall_threshold_sec: Seconds without a batch update before a chunk is
            considered stalled.
        max_simultaneous_stalls: Number of simultaneous stalls that triggers a
            systemic abort (RuntimeError).
        log: Logger.

    Raises:
        RuntimeError: When ``>= max_simultaneous_stalls`` chunks are stalled.
    """
    try:
        progress = cast(dict, ray.get(tracker.get_all.remote(), timeout=5))  # type: ignore[union-attr]
        stalled_chunks: list[str] = []
        for label, (batch, total, staleness, phase) in progress.items():
            if staleness > stall_threshold_sec:
                stalled_chunks.append(label)
                if phase == "inference":
                    log.error(
                        "STALL: chunk %s stuck at batch %d/%d (inference) — no update for %.0fs",
                        label,
                        batch,
                        total,
                        staleness,
                    )
                else:
                    log.warning(
                        "STALL: chunk %s stuck in phase '%s' — no update for %.0fs",
                        label,
                        phase,
                        staleness,
                    )

        if len(stalled_chunks) >= max_simultaneous_stalls:
            msg = (
                f"{len(stalled_chunks)} chunks stalled simultaneously "
                f"({stalled_chunks[:5]}) — aborting (systemic issue)"
            )
            raise RuntimeError(msg)

        n_active = len(progress)
        if n_active > 0:
            phases: dict[str, int] = {}
            for _, (_, _, _, phase) in progress.items():
                phases[phase] = phases.get(phase, 0) + 1
            phase_summary = ", ".join(f"{v} {k}" for k, v in sorted(phases.items()))
            log.info(
                "Progress: %d/%d done, %d active (%s), %d stalled",
                n_done,
                n_total,
                n_active,
                phase_summary,
                len(stalled_chunks),
            )
    except Exception as exc:
        # Tracker is a monitoring aid — never a single point of failure.
        # Swallow all Ray errors (actor death, timeout, serialization)
        # so inference continues without progress visibility.
        if isinstance(exc, RuntimeError):
            raise  # re-raise our own stall-abort RuntimeError
        log.debug("Tracker poll failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------


def _process_chunks_work_stealing(
    actors: list[ray.actor.ActorHandle],
    actor_instance_ids: list[str],
    chunks: list[ChunkSpec],
    mosaic_base: str,
    staging_base: str,
    run_id: str,
    config: InferenceConfig,
    t0: float,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    tracker: ray.actor.ActorHandle | None = None,
    *,
    max_chunk_retries: int = 2,
    still_initializing: set[int] | None = None,
    on_actor_retire: Callable[[str], None] | None = None,
) -> list[dict]:
    """Process chunks with dynamic work-stealing across actors.

    Seeds each actor with one chunk, then dispatches the next queued chunk
    to whichever actor finishes first. This naturally load-balances: fast
    actors cycle through more chunks, slow actors hold fewer.

    If an actor dies (OOM), a replacement is created and the failed chunk is
    re-queued up to ``max_chunk_retries`` times. This prevents a single transient
    failure from killing a multi-hour GPU run.

    Args:
        actors: List of ready Ray actor handles.
        actor_instance_ids: Instance ID per actor (parallel list).
        chunks: All spatial chunks to process.
        mosaic_base: Base path for the mosaic stores.
        staging_base: Base path for staged output.
        run_id: Run identifier.
        config: Inference config (needed to create replacement actors).
        t0: Flow start time for progress logging.
        log: Logger.
        tracker: Optional ProgressTracker actor handle for batch-level progress polling.
        max_chunk_retries: Max times a failed chunk is re-queued before permanent failure.
        still_initializing: Actor indices that haven't finished init yet. These
            actors are included in the pool but skipped during seeding; they
            will pick up work via ``dispatch_idle`` once they come online.
        on_actor_retire: Callback to be triggered when the actor is removed from the pool.
            Used to consistently and swiftly terminate EC2 instances and save compute costs
    Returns:
        List of result dicts (status, chunk label, timing, etc.).
    """
    chunk_queue: deque[ChunkSpec] = deque(chunks)
    n_total = len(chunk_queue)
    chunk_by_label = {c.label: c for c in chunks}

    pool = ActorPool(actors, actor_instance_ids, config, log, on_retire=on_actor_retire)

    # Mark actors that haven't finished __init__ yet. seed() skips them —
    # they'll receive work via dispatch_idle once resolve_initializing
    # confirms they're alive. This avoids sending chunks to actors whose
    # EC2 instances may never provision (AWS capacity limits).
    if still_initializing:
        for idx in still_initializing:
            pool.mark_initializing(idx)

    pool.seed(chunk_queue, mosaic_base, staging_base, run_id, tracker)

    results: list[dict] = []
    last_log_count = 0
    stall_threshold_sec = 300.0
    max_simultaneous_stalls = max(3, len(actors) // 10)

    # --- Main work-stealing loop ---
    # Runs while there is in-flight work OR queued chunks waiting for
    # initializing actors to come online.
    while pool.pending or (chunk_queue and pool._initializing):
        if pool.pending:
            # Block up to 60s for any one chunk to finish.
            ready_refs, _ = ray.wait(list(pool.pending.keys()), num_returns=1, timeout=60)
        else:
            # Nothing in-flight but chunks are queued for initializing actors.
            # Sleep briefly then check if any actors are ready.
            time.sleep(5)
            ready_refs = []

        # Poll tracker on every iteration (including timeouts with no completions)
        # so stall detection stays responsive.
        if tracker:
            _poll_tracker(tracker, len(results), n_total, stall_threshold_sec, max_simultaneous_stalls, log)

        # --- Handle completed (or failed) chunks ---
        for ref in ready_refs:
            chunk_label, actor_idx = pool.pending.pop(ref)

            try:
                # Success path: collect result, clear tracker entry
                result = ray.get(ref)
                results.append(result)
                if tracker:
                    tracker.remove.remote(chunk_label)  # type: ignore[union-attr]
                pool._initializing.discard(actor_idx)
            except Exception as e:
                # Failure path: resolve instance ID for error context, then
                # either re-queue the chunk or mark it permanently failed.
                if tracker:
                    tracker.remove.remote(chunk_label)  # type: ignore[union-attr]
                pool.resolve_iid(actor_idx)
                instance_id = pool.actor_instance_ids[actor_idx]
                attempts = pool.chunk_attempts.get(chunk_label, 1)
                # attempts starts at 1 on first submission, so max_chunk_retries=2
                # means first try + 2 re-queues = 3 total attempts.
                if attempts <= max_chunk_retries:
                    log.warning(
                        "Chunk %s FAILED on instance %s (actor %d, attempt %d/%d): %s — re-queuing",
                        chunk_label,
                        instance_id,
                        actor_idx,
                        attempts,
                        max_chunk_retries,
                        e,
                    )
                    chunk_queue.append(chunk_by_label[chunk_label])
                else:
                    log.error(
                        "Chunk %s PERMANENTLY FAILED on instance %s (actor %d, attempt %d/%d): %s",
                        chunk_label,
                        instance_id,
                        actor_idx,
                        attempts,
                        max_chunk_retries,
                        e,
                    )
                    results.append(
                        {
                            "chunk": chunk_label,
                            "status": "failed",
                            "error": str(e),
                            "instance_id": instance_id,
                            "attempts": attempts,
                        }
                    )

                # Query CloudWatch for resource telemetry leading up to the failure
                log_worker_failure_diagnostic(instance_id, chunk_label, str(e), log)

                # Dead actor → spawn replacement (queues Ray init in background)
                pool.replace(actor_idx, instance_id)

            # Immediately re-feed the actor that just freed up, unless it's a
            # freshly-spawned replacement still initializing (30-120s).
            if chunk_queue and actor_idx not in pool._initializing:
                pool.submit(actor_idx, chunk_queue.popleft(), mosaic_base, staging_base, run_id, tracker)

        # Check if any initializing actors have finished __init__ and can
        # start receiving work, then dispatch queued chunks to all idle
        # actors, then retire actors idle past the grace period.
        pool.resolve_initializing()
        pool.dispatch_idle(chunk_queue, mosaic_base, staging_base, run_id, tracker)
        pool.retire_idle(len(pool.pending) + len(chunk_queue))

        if len(results) > last_log_count:
            elapsed_min = (time.monotonic() - t0) / 60
            log.info(
                "Progress: %d / %d chunks complete (%.1f min elapsed)",
                len(results),
                n_total,
                elapsed_min,
            )
            last_log_count = len(results)

    return results
