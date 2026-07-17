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
from typing import Any, cast

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
        get_credentials: Callable[[], Any] | None = None,
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
            get_credentials: Optional icechunk S3 credential provider injected
                into every actor (including replacements) so store opens refresh
                credentials. See :class:`InferenceActor`.
        """
        self.actors = actors
        self.actor_instance_ids = actor_instance_ids
        self.config = config
        self.log = log
        self.idle_grace_sec = idle_grace_sec
        self._on_retire = on_retire
        self._get_credentials = get_credentials

        self.actor_deaths: int = 0

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

    @property
    def max_actor_deaths(self) -> int:
        """Death count that signals a systemic failure.

        Tracks the current pool size so the threshold scales as later batches
        of actors are added — a fixed batch-1 count would trip the alarm far
        too early on a large fleet built up incrementally.
        """
        return len(self.actors)

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

    def add_actors(self, new_actors: list[ray.actor.ActorHandle]) -> None:
        """Append a freshly-requested batch of actors to the pool.

        Each new slot is marked initializing (with a non-blocking
        ``get_instance_id`` fetch) so it flows into ``dispatch_idle`` once its
        ``__init__`` completes — exactly like the late-joining actors from the
        first batch. Called only from the single-threaded work-stealing loop,
        so no locking is required.

        Args:
            new_actors: Newly created actor handles to add to the pool.
        """
        for actor in new_actors:
            idx = len(self.actors)
            self.actors.append(actor)
            self.actor_instance_ids.append("pending-init")
            self.mark_initializing(idx)

    def replace(self, actor_idx: int, instance_id: str) -> None:
        """Kill the actor in a slot and spawn a replacement; log the death count.

        Fires a non-blocking get_instance_id.remote() so the new EC2 instance
        ID is resolved lazily before the next dispatch or error report.

        The outgoing actor is killed before being replaced. On a process death
        (OOM, segfault) that ``ray.kill`` is a no-op — the actor is already gone.
        When the actor instead caught its error internally and is still alive
        (e.g. a sporadic CUDA error reported as ``status="failed"``), the kill is
        what actually removes the wedged worker so it can't pick up more chunks.

        Args:
            actor_idx: Slot index of the actor to replace.
            instance_id: Instance ID of the actor being replaced (for log context).
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

        # Kill the outgoing actor before replacing the slot. A no-op if it
        # already died; the actual removal when it caught its error and is still
        # live (so a chunk-failing actor never gets handed another chunk).
        with contextlib.suppress(Exception):
            ray.kill(self.actors[actor_idx])

        new_actor = InferenceActor.options(num_gpus=self.config.num_gpus).remote(  # type: ignore[attr-defined]
            self.config, self.config.checkpoint_path, self._get_credentials
        )
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
    elapsed_min: float | None = None,
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
        elapsed_min: Minutes since run start, folded into the single progress
            line (this is the ONLY progress log line — keep it that way).

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
            elapsed = f" — {elapsed_min:.1f} min elapsed" if elapsed_min is not None else ""
            log.info(
                "Progress: %d/%d done, %d active (%s), %d stalled%s",
                n_done,
                n_total,
                n_active,
                phase_summary,
                len(stalled_chunks),
                elapsed,
            )
    except Exception as exc:
        # Tracker is a monitoring aid — never a single point of failure.
        # Swallow all Ray errors (actor death, timeout, serialization)
        # so inference continues without progress visibility.
        if isinstance(exc, RuntimeError):
            raise  # re-raise our own stall-abort RuntimeError
        log.debug("Tracker poll failed (non-fatal): %s", exc)


def _batch_actors_to_request(
    *,
    requested: int,
    target: int,
    outstanding: int,
    alive_gpu_nodes: int,
    nodes_at_last_batch: int,
    last_batch_size: int,
    secs_since_last_batch: float,
    placement_timeout_sec: float,
    batch_size: int,
    placement_tolerance: int = 2,
) -> tuple[int, bool]:
    """Decide how many actors to request for the next batch.

    Pure decision function (no Ray calls) so the gate can be unit-tested in
    isolation. The next batch is requested once the prior batch's instances
    have joined the cluster (placement), or once placement has timed out so a
    capacity shortfall can't gate every remaining batch forever.

    Placement is measured *incrementally* — as nodes joined since the last
    batch was requested, relative to that batch's size — rather than against
    the cumulative requested count. This keeps a single timed-out partial batch
    from permanently gating every later batch on the timeout path: once a
    subsequent batch places, its increment satisfies the check even though the
    earlier shortfall is never made up.

    Args:
        requested: Actors requested so far (``len(pool.actors)``).
        target: Total actors the run should eventually reach.
        outstanding: In-flight + queued chunks. Caps requests so we don't
            provision more actors than there is work left.
        alive_gpu_nodes: GPU nodes currently joined to the cluster.
        nodes_at_last_batch: GPU nodes that were joined when the last batch was
            requested. The baseline for the incremental placement check.
        last_batch_size: Number of actors in the last batch requested. The
            increment expected to join before the prior batch counts as placed.
        secs_since_last_batch: Seconds since the last batch was requested.
        placement_timeout_sec: Placement-wait escape hatch.
        batch_size: Actors per batch.
        placement_tolerance: Stragglers tolerated before the prior batch counts
            as placed, so one slow instance doesn't gate the next request.

    Returns:
        ``(n, timed_out)`` — number of actors to request (0 = none) and whether
        the placement timeout (rather than placement itself) allowed it. The
        flag is only meaningful when ``n > 0``; the caller uses it for logging.
    """
    if requested >= target:
        return 0, False
    # Don't over-provision: once the pool already holds at least as many actors
    # as there is work left, no further batches are needed.
    if requested >= outstanding:
        return 0, False
    placed = alive_gpu_nodes - nodes_at_last_batch >= last_batch_size - placement_tolerance
    timed_out = secs_since_last_batch > placement_timeout_sec
    if not (placed or timed_out):
        return 0, False
    # Clamp to both the remaining target and the remaining work, so a partial
    # final batch never over-provisions actors past what's left to do.
    return min(batch_size, target - requested, outstanding - requested), (timed_out and not placed)


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
    get_credentials: Callable[[], Any] | None = None,
    actor_factory: Callable[[int], list[ray.actor.ActorHandle]] | None = None,
    total_actors_target: int | None = None,
    placement_timeout_sec: float = 300.0,
) -> list[dict]:
    """Process chunks with dynamic work-stealing across actors.

    Seeds each actor with one chunk, then dispatches the next queued chunk
    to whichever actor finishes first. This naturally load-balances: fast
    actors cycle through more chunks, slow actors hold fewer.

    When a chunk fails — whether the actor process died (OOM, segfault) or the
    actor caught its own error and returned ``status="failed"`` (e.g. a sporadic
    CUDA error) — the failing actor is killed and replaced and the chunk is
    re-queued up to ``max_chunk_retries`` times so the retry lands on a healthy
    worker. Killing on every failure (not just death) keeps an actor that has
    started failing, which tends to keep failing, from being handed more chunks.
    This prevents a single transient failure from killing a multi-hour GPU run.

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
        get_credentials: Optional icechunk S3 credential provider injected into
            every actor (seeded and replacement) so store opens refresh creds.
        actor_factory: Optional callable ``(n) -> [handles]`` that requests ``n``
            new actors. When provided (with ``total_actors_target``), the loop
            requests actors in batches: the caller supplies the first batch, and
            subsequent batches are requested here once the prior batch's
            instances have joined the cluster. ``None`` disables batching — the
            caller's ``actors`` list is the whole fleet.
        total_actors_target: Total actors the run should eventually reach. The
            loop stops requesting once ``len(pool.actors)`` reaches this. Ignored
            when ``actor_factory`` is None.
        placement_timeout_sec: Max seconds to wait for a batch's instances to be
            placed before requesting the next batch anyway (capacity-shortfall
            escape hatch).

    Returns:
        List of result dicts (status, chunk label, timing, etc.).
    """
    chunk_queue: deque[ChunkSpec] = deque(chunks)
    n_total = len(chunk_queue)
    chunk_by_label = {c.label: c for c in chunks}

    pool = ActorPool(
        actors, actor_instance_ids, config, log, on_retire=on_actor_retire, get_credentials=get_credentials
    )

    # Mark actors that haven't finished __init__ yet. seed() skips them —
    # they'll receive work via dispatch_idle once resolve_initializing
    # confirms they're alive. This avoids sending chunks to actors whose
    # EC2 instances may never provision (AWS capacity limits).
    if still_initializing:
        for idx in still_initializing:
            pool.mark_initializing(idx)

    pool.seed(chunk_queue, mosaic_base, staging_base, run_id, tracker)

    results: list[dict] = []
    stall_threshold_sec = 300.0

    # Stall threshold scales with the eventual fleet size, not just the first
    # batch. With batching, ``actors`` holds only the initial subset, so a
    # threshold frozen at batch-1/10 would abort a large run after a handful of
    # stalls even once later batches join. Recomputed each iteration (see loop)
    # against the larger of the current pool and the target so it tracks the
    # pool as it grows.
    def _stall_threshold() -> int:
        fleet = max(len(pool.actors), total_actors_target or 0)
        return max(3, fleet // 10)

    # --- Actor-batch requesting ---
    # When batching is enabled the caller supplies only the first batch of
    # actors; the loop requests the rest here, pacing the demand the autoscaler
    # forwards to AWS. The gate is placement (instances joined as GPU nodes),
    # not readiness, so a slow checkpoint load never stalls the next AWS ask.
    batching_enabled = (
        actor_factory is not None and total_actors_target is not None and config.actor_request_batch_size > 0
    )
    last_batch_at = time.monotonic()
    # Placement baseline for the incremental check in _batch_actors_to_request:
    # the GPU-node count and size of the most recently requested batch. The
    # first batch was already requested by the caller, so seed last_batch_size
    # from the pool we were handed; nodes_at_last_batch starts at 0 because no
    # GPU nodes are guaranteed present before the first batch is placed.
    nodes_at_last_batch = 0
    last_batch_size = len(pool.actors)

    def _maybe_request_next_batch() -> None:
        nonlocal last_batch_at, nodes_at_last_batch, last_batch_size
        if not batching_enabled:
            return
        assert actor_factory is not None and total_actors_target is not None  # narrowed by batching_enabled
        alive_gpu_nodes = sum(1 for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0)
        n, timed_out = _batch_actors_to_request(
            requested=len(pool.actors),
            target=total_actors_target,
            outstanding=len(pool.pending) + len(chunk_queue),
            alive_gpu_nodes=alive_gpu_nodes,
            nodes_at_last_batch=nodes_at_last_batch,
            last_batch_size=last_batch_size,
            secs_since_last_batch=time.monotonic() - last_batch_at,
            placement_timeout_sec=placement_timeout_sec,
            batch_size=config.actor_request_batch_size,
        )
        if n == 0:
            return
        pool.add_actors(actor_factory(n))
        last_batch_at = time.monotonic()
        nodes_at_last_batch = alive_gpu_nodes
        last_batch_size = n
        log.info(
            "Requested actor batch: +%d (%d/%d total, %d GPU nodes placed)%s",
            n,
            len(pool.actors),
            total_actors_target,
            alive_gpu_nodes,
            " — placement timed out, requesting anyway" if timed_out else "",
        )

    def _handle_failure(chunk_label: str, actor_idx: int, error: str) -> None:
        """Retry a failed chunk on a different worker and kill the failing actor.

        Shared by both failure modes: an actor whose process died (ray.get
        raised) and an actor that caught its own error and returned
        status="failed" while still alive. In both cases the chunk is re-queued
        up to ``max_chunk_retries`` times and the actor is killed + replaced, so
        the retry lands on a healthy worker and the failing one — which tends to
        keep failing on subsequent chunks — is taken out of rotation.

        Args:
            chunk_label: Label of the chunk that failed.
            actor_idx: Slot of the actor that failed it.
            error: Error string (from the exception or the failed result dict).
        """
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
                error,
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
                error,
            )
            results.append(
                {
                    "chunk": chunk_label,
                    "status": "failed",
                    "error": error,
                    "instance_id": instance_id,
                    "attempts": attempts,
                }
            )

        # Query CloudWatch for resource telemetry leading up to the failure
        log_worker_failure_diagnostic(instance_id, chunk_label, error, log)

        # Kill the failing actor and spawn a replacement (queues Ray init in
        # the background). The kill is a no-op if the actor process already died.
        pool.replace(actor_idx, instance_id)

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
            _poll_tracker(
                tracker,
                len(results),
                n_total,
                stall_threshold_sec,
                _stall_threshold(),
                log,
                elapsed_min=(time.monotonic() - t0) / 60,
            )

        # --- Handle completed (or failed) chunks ---
        for ref in ready_refs:
            chunk_label, actor_idx = pool.pending.pop(ref)

            try:
                result = ray.get(ref)
            except Exception as e:
                # The actor process died (OOM, segfault, CUDA process abort) so
                # ray.get raised. Stringify and route through the same handling
                # as an actor that caught its own error and returned "failed".
                _handle_failure(chunk_label, actor_idx, str(e))
            else:
                # ray.get succeeded, but the actor catches its own exceptions and
                # returns status="failed" rather than raising (see actors.py). A
                # sporadic CUDA error arrives this way with the actor still alive,
                # so failed results take the same kill-and-retry path as a death —
                # otherwise the chunk would never retry and the wedged actor, which
                # tends to keep failing, would be handed the next chunk below.
                if result.get("status") == "failed":
                    _handle_failure(chunk_label, actor_idx, str(result.get("error", "unknown")))
                else:
                    results.append(result)
                    if tracker:
                        tracker.remove.remote(chunk_label)  # type: ignore[union-attr]
                    pool._initializing.discard(actor_idx)

            # Immediately re-feed the actor that just freed up, unless it's a
            # freshly-spawned replacement still initializing (30-120s). A failed
            # actor was killed and replaced by _handle_failure, so it's now
            # initializing and skipped here — its retried chunk goes to a
            # different, healthy actor via dispatch_idle below.
            if chunk_queue and actor_idx not in pool._initializing:
                pool.submit(actor_idx, chunk_queue.popleft(), mosaic_base, staging_base, run_id, tracker)

        # Request the next actor batch if the prior batch has been placed (or
        # placement timed out), then check if any initializing actors have
        # finished __init__ and can start receiving work, then dispatch queued
        # chunks to all idle actors, then retire actors idle past the grace
        # period.
        _maybe_request_next_batch()
        pool.resolve_initializing()
        pool.dispatch_idle(chunk_queue, mosaic_base, staging_base, run_id, tracker)
        pool.retire_idle(len(pool.pending) + len(chunk_queue))


    return results
