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
from dataclasses import dataclass
from typing import Any, cast

import ray

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.inference.actors import InferenceActor
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.diagnostics import log_worker_failure_diagnostic
from tessera_embeddings.inference.progress import chunk_uid


@dataclass(frozen=True)
class ZoneContext:
    """Where a chunk's inputs live and where its staging goes.

    One per (zone, year) cell. A single-zone run has exactly one (built from
    ``_process_chunks_work_stealing``'s scalar args); a chained multi-zone
    session has one per zone, carried on every :class:`WorkItem` so chunks
    from consecutive zones can coexist in one scheduler queue. Frozen so
    equality is value-based — the reservation path uses ``ctx == ctx`` to
    restrict prefetch hints to same-zone successors (a cross-zone hint would
    make the actor prefetch from the wrong mosaic).

    ``time_window`` is the cell's OWN inference window, and it lives here rather
    than on the actor's config because a chained session's cells may span campaign
    YEARS. An actor is built once with one config; reading the window from that
    config would make every cell of a different year read the wrong months, and the
    session's only mismatch check is on ``s1_orbit``, so nothing would catch it.
    ``None`` means "use the actor's own config" — what the single-ROI path passes,
    and what keeps that path byte-for-byte unchanged.

    ``s1_orbit`` is here for the same reason and closes the gap named above. A chained
    session's cells may resolve DIFFERENT orbits: parts of the globe are radar-free in
    principle, so a cell can resolve ``"none"`` while the session was asked for ``"both"``.
    Reading the orbit from the actor's config made that a mismatch the session could only
    handle by deferring the cell and holding its mosaic against a bounded budget — and past
    that bound the cell failed and its mosaic was deleted, so a deterministic, predictable
    population of zones could never complete. Carried per cell, a radar-free cell is ordinary
    work for any actor.
    """

    mosaic_base: str
    staging_base: str
    run_id: str
    time_window: TimeWindow | None = None
    s1_orbit: str | None = None


@dataclass(frozen=True)
class WorkItem:
    """One chunk plus the zone context it must be processed under.

    ``uid`` (run_id-qualified label) keys all scheduler bookkeeping: chunk
    labels repeat across zones (every zone has a ``chunk_0_0``), so bare
    labels would alias retry counts and requeues between a finishing zone's
    tail and the next zone's head. Run ids are campaign-unique per cell
    (input-fingerprinted), so ``run_id:label`` is collision-free.
    """

    chunk: ChunkSpec
    ctx: ZoneContext

    @property
    def uid(self) -> str:
        """Scheduler-unique key for this item (labels alone collide across zones)."""
        return chunk_uid(self.ctx.run_id, self.chunk.label)


def _as_item(work: WorkItem | ChunkSpec, ctx: ZoneContext) -> WorkItem:
    """Normalize a bare ChunkSpec (legacy callers/tests) into a WorkItem."""
    return work if isinstance(work, WorkItem) else WorkItem(chunk=work, ctx=ctx)


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
        s3_region: str | None = None,
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
            s3_region: Optional S3 region injected into every actor (including
                replacements) so a non-default-region fill's reads — and the
                retries after a chunk failure/OOM — open the mosaic in the right
                region. See :class:`InferenceActor`.
        """
        self.actors = actors
        self.actor_instance_ids = actor_instance_ids
        self.config = config
        self.log = log
        self.idle_grace_sec = idle_grace_sec
        self._on_retire = on_retire
        self._get_credentials = get_credentials
        self._s3_region = s3_region

        self.actor_deaths: int = 0

        # ref → (WorkItem, actor_idx). (Some pool unit tests insert bare
        # labels in the WorkItem slot; the pool itself only reads the actor
        # index from entries it did not create.)
        self.pending: dict[ray.ObjectRef, tuple[Any, int]] = {}
        # Retry counts keyed by WorkItem.uid (run_id-qualified — bare labels
        # collide across a chained session's zones).
        self.chunk_attempts: dict[str, int] = {}

        # actor_idx → the item reserved as that actor's next assignment. A
        # reservation is created at submit time and passed to the actor as
        # ``prefetch_hint`` so it can prefetch a BOUNDED starter payload (mask
        # + 256-row starter strip, hard-capped ~2 GiB — see actors.py
        # ``_XCHUNK_PREFETCH_CAP_BYTES``, NOT the full-prologue payload the
        # removed Phase-1 interleaving co-resided) during the current chunk's
        # tail inference. Reservations stop when the queue is shallower than
        # the live pool; a failed actor's reservation is requeued to the front.
        # Only SAME-ZONE successors are reserved — the actor prefetches the
        # hint from the CURRENT call's mosaic, so a cross-zone hint would read
        # the wrong store (see submit()).
        self.reserved: dict[int, WorkItem] = {}

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
        """Number of non-retired actor slots — the pool's CONTROL quantity.

        Slots the pool intends to dispatch to, placed on a node or not. It is
        not billed capacity; for that see :func:`_billed_gpu_count`.
        """
        return len(self.actors) - len(self._retired)

    def outstanding_work(self, queued: int) -> int:
        """In-flight + queued + reserved chunks — the true remaining-work count.

        Reserved next-chunks live in neither ``pending`` nor the queue while a
        busy actor holds them; excluding them would make remaining work look
        smaller, under-provisioning actor batches and over-retiring idle actors
        mid-run. The tail over-provision this can cause is bounded (the
        ``requested >= outstanding`` and ``target - requested`` caps) and
        self-corrects via idle retirement.
        """
        return len(self.pending) + queued + len(self.reserved)

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
        work: WorkItem | ChunkSpec,
        mosaic_base: str | None = None,
        staging_base: str | None = None,
        run_id: str | None = None,
        tracker: ray.actor.ActorHandle | None = None,
        chunk_queue: deque[WorkItem | ChunkSpec] | None = None,
    ) -> None:
        """Submit a single work item to an actor and record the pending future.

        ``work`` is a :class:`WorkItem`, or (legacy callers/tests) a bare
        ``ChunkSpec`` wrapped with the scalar path args into a single-zone
        item. When ``chunk_queue`` is supplied and still deep, the queue head
        is also popped and RESERVED as this actor's next assignment, riding
        along as ``prefetch_hint`` so the actor can prefetch that chunk's
        capped starter payload during this chunk's tail inference (see
        actors.py). Reservations stop when the queue is shallower than the
        live pool (``len(queue) <= live_count``): at the tail of a run a
        reserved chunk would pin work to a busy actor while other actors sit
        idle, which costs more than the prologue overlap saves. A queue head
        from a DIFFERENT zone is never reserved: the actor prefetches the hint
        from the current call's mosaic, so a cross-zone hint would read the
        wrong store — the next zone's first chunks load serially instead.

        Args:
            actor_idx: Index into self.actors.
            work: Work item (or bare chunk, wrapped with the path args below).
            mosaic_base: Base path for mosaic stores (bare-chunk callers only).
            staging_base: Base path for staged output (bare-chunk callers only).
            run_id: Run identifier (bare-chunk callers only).
            tracker: Optional ProgressTracker actor handle.
            chunk_queue: Remaining-work queue; the reservation source. ``None``
                disables reservation (tests and direct callers).
        """
        if isinstance(work, WorkItem):
            item = work
        else:
            if mosaic_base is None or staging_base is None or run_id is None:
                raise TypeError("bare-ChunkSpec submit requires mosaic_base, staging_base and run_id")
            item = WorkItem(chunk=work, ctx=ZoneContext(mosaic_base, staging_base, run_id))
        self.resolve_iid(actor_idx)  # best-effort resolve before dispatch
        if actor_idx in self.reserved:
            # Defensive: a reservation should have been consumed or returned
            # before this actor is re-dispatched; don't strand the chunk.
            stranded = self.reserved.pop(actor_idx)
            self.log.warning(
                "Actor %d re-dispatched holding reservation %s — requeuing it", actor_idx, stranded.chunk.label
            )
            if chunk_queue is not None:
                chunk_queue.appendleft(stranded)

        hint: WorkItem | None = None
        if (
            chunk_queue is not None
            and len(chunk_queue) > self.live_count
            and _as_item(chunk_queue[0], item.ctx).ctx == item.ctx
        ):
            hint = _as_item(chunk_queue.popleft(), item.ctx)
            self.reserved[actor_idx] = hint

        ref: ray.ObjectRef = self.actors[actor_idx].process_chunk.remote(  # type: ignore[union-attr]
            item.chunk,
            item.ctx.mosaic_base,
            item.ctx.staging_base,
            item.ctx.run_id,
            tracker=tracker,
            prefetch_hint=hint.chunk if hint is not None else None,
            time_window=item.ctx.time_window,
            s1_orbit=item.ctx.s1_orbit,
        )
        self.pending[ref] = (item, actor_idx)
        self.chunk_attempts[item.uid] = self.chunk_attempts.get(item.uid, 0) + 1

    def take_reserved(self, actor_idx: int) -> WorkItem | None:
        """Pop and return the item reserved for this actor, if any."""
        return self.reserved.pop(actor_idx, None)

    def seed(
        self,
        chunk_queue: deque[WorkItem | ChunkSpec],
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None,
    ) -> None:
        """Submit one chunk to each actor to prime the work-stealing loop.

        The queue rides along to ``submit()`` so each seeded actor also gets a
        next-chunk reservation (prefetch hint) while the queue is deep — its
        FIRST chunk can already overlap the second chunk's prologue.

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
            self.submit(actor_idx, chunk_queue.popleft(), mosaic_base, staging_base, run_id, tracker, chunk_queue)
        self.log.info("Seeded %d actors with initial chunks (%d queued)", len(self.pending), len(chunk_queue))

    def dispatch_idle(
        self,
        chunk_queue: deque[WorkItem | ChunkSpec],
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None,
    ) -> None:
        """Dispatch queued chunks to any idle, non-initializing actors.

        Covers cases where a replacement actor was skipped in the main loop —
        other idle actors can pick up the work immediately. The queue rides
        along to ``submit()`` so late-joining actors get the same next-chunk
        reservation (prefetch hint) as main-loop dispatches while the queue
        is deep.

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
            self.submit(idx, chunk_queue.popleft(), mosaic_base, staging_base, run_id, tracker, chunk_queue)
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
            self.config, self.config.checkpoint_path, self._get_credentials, self._s3_region
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


def _billed_gpu_count(last_known: float) -> float:
    """GPUs joined to the cluster — the capacity actually being billed.

    Every GPU worker declares one GPU and the head declares none, so the cluster
    TOTAL is the billed worker count. Total, not available: a GPU an actor is
    working on is still billed, so ``available_resources()`` — the idle
    remainder — is the wrong call. A slot Ray has accepted but not yet placed
    contributes nothing, which is the point.

    The guard is load-bearing, not defensive. Its caller sits in the dispatch
    loop OUTSIDE the try/except that wraps the tracker poll, so an exception
    from this lookup would abort a fill over a log number; a failure carries the
    previous count forward instead.

    Args:
        last_known: Count to fall back to when the lookup fails.
    """
    try:
        return float(ray.cluster_resources().get("GPU", 0))
    except Exception:
        return last_known


def _poll_tracker(
    tracker: ray.actor.ActorHandle,
    n_done: int,
    n_total: int,
    stall_threshold_sec: float,
    max_simultaneous_stalls: int,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    elapsed_min: float | None = None,
    gpu_hours: float | None = None,
    recovery_threshold_sec: float | None = None,
) -> list[str]:
    """Poll ProgressTracker; log stalls; return the uids that need recovery.

    This function has no access to pool or loop state — all context is passed
    explicitly so it can be tested in isolation. It therefore cannot kill an
    actor itself: it NAMES the chunks whose actors the caller should replace,
    and the caller (which holds the pool) acts.

    Args:
        tracker: ProgressTracker Ray actor handle.
        n_done: Number of chunks already completed (for the progress log line).
        n_total: Total chunks in the run.
        stall_threshold_sec: Seconds without a batch update before a chunk is
            considered stalled — the WARNING threshold.
        max_simultaneous_stalls: Number of simultaneous stalls that triggers a
            systemic abort (RuntimeError).
        log: Logger.
        elapsed_min: Minutes since INFERENCE began — the dispatch loop's own start, not the run's.
            Folded into the single progress line (this is the ONLY progress log line — keep it that
            way). Measured from the run's start it read as though inference had been going for the
            ingest, cluster-bringup and model-load time too, and disagreed with ``gpu_hours`` on the
            same line.
        gpu_hours: Fleet GPU-hours consumed so far — the cluster's BILLED GPU count integrated
            over wall time (see :func:`_billed_gpu_count`) — folded into the same line.
        recovery_threshold_sec: Seconds without an update before a chunk is
            declared unrecoverable in place and returned for kill-and-requeue.
            Deliberately well above ``stall_threshold_sec`` so a warning always
            fires long before anything is killed, and a merely-slow chunk is
            never destroyed. ``None`` disables recovery (log-only).

    Returns:
        The uids stalled past ``recovery_threshold_sec``, for the caller to
        recover. Empty in the normal case.

    Raises:
        RuntimeError: When ``>= max_simultaneous_stalls`` chunks are stalled.
    """
    try:
        progress = cast(dict, ray.get(tracker.get_all.remote(), timeout=5))  # type: ignore[union-attr]
        stalled_chunks: list[str] = []
        needs_recovery: list[str] = []
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
                if recovery_threshold_sec is not None and staleness > recovery_threshold_sec:
                    needs_recovery.append(label)

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
            # "inferring", not "elapsed": the number now counts from the dispatch loop's start, and
            # a bare "elapsed" beside a chunk counter is what invited reading it as run wall-clock.
            elapsed = f" — {elapsed_min:.1f} min inferring" if elapsed_min is not None else ""
            gpu = f", {gpu_hours:.1f} GPU-hrs" if gpu_hours is not None else ""
            # "chunks done" labels the whole first clause: what follows it counts chunks in
            # flight, not actor slots and not GPUs. GPU-hrs carries its own unit.
            log.info(
                "Progress: %d/%d chunks done, %d active (%s), %d stalled%s%s",
                n_done,
                n_total,
                n_active,
                phase_summary,
                len(stalled_chunks),
                elapsed,
                gpu,
            )
        return needs_recovery
    except Exception as exc:
        # Tracker is a monitoring aid — never a single point of failure.
        # Swallow all Ray errors (actor death, timeout, serialization)
        # so inference continues without progress visibility.
        if isinstance(exc, RuntimeError):
            raise  # re-raise our own stall-abort RuntimeError
        log.debug("Tracker poll failed (non-fatal): %s", exc)
        # A failed poll says nothing about the chunks, so recover nothing. Losing
        # visibility must not be read as "everything is wedged".
        return []


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
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    tracker: ray.actor.ActorHandle | None = None,
    *,
    max_chunk_retries: int = 2,
    still_initializing: set[int] | None = None,
    on_actor_retire: Callable[[str], None] | None = None,
    get_credentials: Callable[[], Any] | None = None,
    s3_region: str | None = None,
    actor_factory: Callable[[int], list[ray.actor.ActorHandle]] | None = None,
    total_actors_target: int | None = None,
    placement_timeout_sec: float = 300.0,
    retire_idle_actors: bool = True,
    more_work: Callable[[], list[WorkItem] | None] | None = None,
    on_item_done: Callable[[WorkItem, dict], None] | None = None,
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
        s3_region: Optional S3 region injected into every actor (seeded and
            replacement) so reads open the mosaic in the caller's region.
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
        retire_idle_actors: Kill actors idle past the grace period as the run's
            tail drains (the default). A chained multi-zone fill leaves this
            True and relies on the ``more_work`` gate below; a caller managing
            zones as separate calls passes False for every zone but its last —
            the "surplus" workers are the NEXT zone's fleet, and retiring them
            would idle-drain the shared cluster's instances at every zone tail
            (see ``orchestration.runners.sequential_fill``).
        more_work: Optional pull source for a chained multi-zone session.
            Polled (non-blocking) whenever the queue is at or below the live
            actor count: a list extends the queue (its items carry their own
            :class:`ZoneContext`), ``[]`` means nothing ready YET (the loop
            stays alive and keeps polling), ``None`` means exhausted forever.
            While the source is unexhausted, idle retirement is suppressed —
            apparently-idle actors are the next zone's fleet — and the loop
            does not exit on an empty queue.
        on_item_done: Optional callback fired exactly once per work item at
            its FINAL outcome — success (after any deferred write confirms) or
            permanent failure — with the item and its result dict. A chained
            session uses it for per-zone completion accounting; it runs on the
            scheduler thread, so it must not block.

    Returns:
        List of result dicts (status, chunk label, timing, etc.).
    """
    default_ctx = ZoneContext(mosaic_base, staging_base, run_id)
    chunk_queue: deque[WorkItem | ChunkSpec] = deque(_as_item(c, default_ctx) for c in chunks)
    n_total = len(chunk_queue)

    pool = ActorPool(
        actors,
        actor_instance_ids,
        config,
        log,
        on_retire=on_actor_retire,
        get_credentials=get_credentials,
        s3_region=s3_region,
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

    #: Seconds of no batch progress before a chunk's actor is killed and the chunk
    #: re-queued. FOUR TIMES the warning threshold, so an operator always sees
    #: several STALL lines before anything is destroyed, and a merely-slow chunk is
    #: never killed.
    #:
    #: The margin is enormous relative to legitimate work: this measures the gap
    #: BETWEEN BATCH UPDATES, which is normally well under a second, while the
    #: slowest whole chunk yet observed takes ~480 s in total. The wedge this exists
    #: for sat 14,396 s. So the choice is not between 20 and 30 minutes — it is
    #: between minutes and forever, and anywhere in that range works.
    stall_recovery_sec = 4 * stall_threshold_sec

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
            outstanding=pool.outstanding_work(len(chunk_queue)),
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

    def _handle_failure(item: WorkItem, actor_idx: int, error: str) -> None:
        """Retry a failed chunk on a different worker and kill the failing actor.

        Shared by both failure modes: an actor whose process died (ray.get
        raised) and an actor that caught its own error and returned
        status="failed" while still alive. In both cases the chunk is re-queued
        up to ``max_chunk_retries`` times and the actor is killed + replaced, so
        the retry lands on a healthy worker and the failing one — which tends to
        keep failing on subsequent chunks — is taken out of rotation.

        Args:
            item: The work item that failed.
            actor_idx: Slot of the actor that failed it.
            error: Error string (from the exception or the failed result dict).
        """
        chunk_label = item.chunk.label
        if tracker:
            tracker.remove.remote(item.uid)  # type: ignore[union-attr]  # run-qualified: labels alias across zones
        # The actor is killed below, taking its writer thread with it — any
        # deferred write it still held is of unknown state, so requeue that
        # chunk too. Safe: staged writes are idempotent (run-scoped keys,
        # mode="w"; resume-scan tolerates rewrites).
        orphaned = pending_write.pop(actor_idx, None)
        if orphaned is not None:
            _requeue_unconfirmed(orphaned, f"actor {actor_idx} failed with the write in flight")

        # Its reserved next chunk (whose starter payload only this actor may
        # have prefetched) goes back to the queue FRONT for a healthy actor.
        reserved = pool.take_reserved(actor_idx)
        if reserved is not None:
            chunk_queue.appendleft(reserved)
            log.info(
                "Returned reserved chunk %s from failed actor %d to the queue front", reserved.chunk.label, actor_idx
            )
        pool.resolve_iid(actor_idx)
        instance_id = pool.actor_instance_ids[actor_idx]
        attempts = pool.chunk_attempts.get(item.uid, 1)
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
            chunk_queue.append(item)
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
            failed = {
                "chunk": chunk_label,
                "status": "failed",
                "error": error,
                "instance_id": instance_id,
                "attempts": attempts,
            }
            results.append(failed)
            if on_item_done is not None:
                on_item_done(item, failed)

        # Query CloudWatch for resource telemetry leading up to the failure
        log_worker_failure_diagnostic(instance_id, chunk_label, error, log)

        # Kill the failing actor and spawn a replacement (queues Ray init in
        # the background). The kill is a no-op if the actor process already died.
        pool.replace(actor_idx, instance_id)

    # --- Deferred staging writes ---
    # Actors return with write_deferred=True while their staging upload runs
    # on a background thread (overlapping the next chunk's serial prologue).
    # Such results are held here — NOT counted complete — until the write's
    # outcome arrives: piggybacked as prior_write on the actor's next result,
    # or pulled via flush_writes() once the actor idles. On failure or actor
    # death the chunk requeues (staged writes are idempotent).
    pending_write: dict[int, tuple[WorkItem, dict]] = {}  # actor_idx -> (item, deferred result)

    def _requeue_unconfirmed(entry: tuple[WorkItem, dict], reason: str) -> None:
        """Requeue a deferred chunk whose write failed or is of unknown state."""
        item, deferred = entry
        label = str(deferred["chunk"])
        attempts = pool.chunk_attempts.get(item.uid, 1)
        if attempts <= max_chunk_retries:
            log.warning(
                "Chunk %s staging write unconfirmed (%s, attempt %d/%d) — re-queuing",
                label,
                reason,
                attempts,
                max_chunk_retries,
            )
            chunk_queue.append(item)
        else:
            log.error("Chunk %s PERMANENTLY FAILED: staging write unconfirmed (%s)", label, reason)
            failed = {"chunk": label, "status": "failed", "error": f"staging write: {reason}", "attempts": attempts}
            results.append(failed)
            if on_item_done is not None:
                on_item_done(item, failed)

    def _finalize_prior_write(actor_idx: int, prior: dict) -> None:
        """Resolve a deferred chunk using the write outcome its actor reported."""
        entry = pending_write.pop(actor_idx, None)
        if entry is None:
            log.warning("Actor %d reported a write outcome for %s but none was pending", actor_idx, prior.get("label"))
            return
        item, deferred = entry
        if prior.get("ok"):
            deferred["write_confirmed"] = True
            results.append(deferred)
            if on_item_done is not None:
                on_item_done(item, deferred)
        else:
            # A plain write error leaves the actor healthy (it just inferred a
            # whole chunk) — requeue without the kill-and-replace used for
            # inference failures.
            _requeue_unconfirmed(entry, str(prior.get("error", "unknown write error")))
            if prior.get("timed_out"):
                # ...but a TIMEOUT means the upload is still wedged in the
                # actor's single-slot writer pool; keep dispatching to it and
                # later writes queue behind the stuck task and time out too.
                # Replace the slot (reaping the writer). Only reached via the
                # idle-flush path — the hot path raises in process_chunk, so
                # this actor is idle and has no chunk mid-assignment.
                log.warning("Actor %d writer pool wedged (write timeout); replacing", actor_idx)
                pool.resolve_iid(actor_idx)
                pool.replace(actor_idx, pool.actor_instance_ids[actor_idx])

    def _flush_idle_writes() -> None:
        """Drain deferred writes on actors with no in-flight call to carry them."""
        busy = pool.busy_actors
        for actor_idx in [a for a in pending_write if a not in busy]:
            try:
                prior = cast(
                    "dict | None",
                    ray.get(pool.actors[actor_idx].flush_writes.remote(), timeout=600),  # type: ignore[union-attr]
                )
            except Exception as exc:
                _requeue_unconfirmed(pending_write.pop(actor_idx), f"flush failed: {exc}")
                # A failed or timed-out flush RPC may leave the (serial) actor
                # still wedged inside flush_writes(); a retry dispatched to it
                # would sit behind that stuck call forever — and invisibly, as
                # a chunk that never starts has no tracker entry for stall
                # detection. Kill + replace the slot so the requeued chunk
                # lands on a healthy actor. Safe: staged writes are idempotent.
                pool.resolve_iid(actor_idx)
                pool.replace(actor_idx, pool.actor_instance_ids[actor_idx])
                continue
            if prior is None:
                _requeue_unconfirmed(pending_write.pop(actor_idx), "actor had no pending write to flush")
            else:
                _finalize_prior_write(actor_idx, prior)

    # --- Main work-stealing loop ---
    # gpu_seconds integrates BILLED GPUs over wall time so the progress line can
    # report fleet GPU-hours consumed so far. Billed, not requested: a slot the
    # pool has asked for but Ray has not placed costs nothing until its instance
    # exists. Logging-only — nothing scales, retires or branches on it.
    gpu_seconds = 0.0
    billed_gpus = 0.0
    last_tick = time.monotonic()
    # The progress line's elapsed clock, and it starts HERE rather than at ``t0``.
    #
    # ``t0`` is the whole run's start: for a chained session that is the top of the stream, before
    # the ingest look-ahead, before ``ray up``, before per-worker EC2 bringup and before the model
    # loads. Printed beside a chunk counter it read as though inference had been running that long —
    # a cell logged "0/10 done, 5 active ... 46.3 min elapsed" seconds after its actors came ready.
    #
    # It was also inconsistent with the other figure on its own line: ``gpu_seconds`` starts at zero
    # right here, so GPU-hours already measured actor time while elapsed measured stream wall-clock.
    # Two different clocks in one sentence is what made the line unreadable; now both start together.
    #
    # ``t0`` keeps its other uses (run summaries, provenance); only this line changes.
    inference_t0 = last_tick
    # A chained session's work source: while unexhausted, the loop must stay
    # alive through empty-queue gaps (the next zone may still be ingesting)
    # and must not retire "idle" actors (they are the next zone's fleet).
    source_active = more_work is not None
    # Stay alive while any work remains: in-flight chunks, deferred writes
    # awaiting confirmation, queued chunks with a live actor to run them, OR
    # an unexhausted work source that may still supply more zones.
    # The queue clause must NOT be gated on _initializing alone — a failed tail
    # flush (_flush_idle_writes, which runs after dispatch) requeues its chunk
    # when no actor is initializing, and gating on _initializing would drop that
    # retry. `live_count > 0` prevents a busy-spin when every actor has died.
    while pool.pending or pending_write or ((chunk_queue or source_active) and pool.live_count > 0):
        # Top up from the work source BEFORE waiting so freshly-ready zones
        # dispatch this iteration. Polled only when the queue is at or below
        # the live actor count — the exhaustion trigger. For zones at least as
        # large as the fleet this keeps them near-sequential (one zone's tail
        # overlaps the next zone's head); a zone smaller than the fleet can't
        # fill it alone, so successive small zones are pulled to pack it — the
        # source hands over one zone per poll and the caller bounds how many
        # coexist.
        if source_active and len(chunk_queue) <= pool.live_count:
            fetched = more_work()  # type: ignore[misc]  # source_active implies more_work is not None
            if fetched is None:
                source_active = False
                log.info(
                    "Work source exhausted — %d queued + %d in-flight chunk(s) remain",
                    len(chunk_queue),
                    len(pool.pending),
                )
            elif fetched:
                chunk_queue.extend(fetched)
                n_total += len(fetched)
                log.info(
                    "Work source added %d chunk(s) (queue now %d, total %d)", len(fetched), len(chunk_queue), n_total
                )
        if pool.pending:
            # Block for any one chunk to finish. At a zone boundary — source still
            # active and the queue drained to the poll trigger — the next zone may
            # become ready momentarily, so cap the wait short: blocking the full
            # 60s on a single tail task would idle the rest of the fleet up to a
            # GPU-minute at every boundary. Otherwise (source exhausted, or a deep
            # queue keeping actors busy) the long wait is fine.
            at_boundary = source_active and len(chunk_queue) <= pool.live_count
            wait_timeout = 5 if at_boundary else 60
            ready_refs, _ = ray.wait(list(pool.pending.keys()), num_returns=1, timeout=wait_timeout)
        else:
            # Nothing in-flight but chunks are queued for initializing actors.
            # Sleep briefly then check if any actors are ready.
            time.sleep(5)
            ready_refs = []
        now = time.monotonic()
        billed_gpus = _billed_gpu_count(billed_gpus)
        gpu_seconds += billed_gpus * (now - last_tick)
        last_tick = now

        # Poll tracker on every iteration (including timeouts with no completions)
        # so stall detection stays responsive.
        if tracker:
            wedged = _poll_tracker(
                tracker,
                len(results),
                n_total,
                stall_threshold_sec,
                _stall_threshold(),
                log,
                elapsed_min=(time.monotonic() - inference_t0) / 60,
                gpu_hours=gpu_seconds / 3600,
                recovery_threshold_sec=stall_recovery_sec,
            )
            # A SINGLE wedged chunk used to hold the whole fleet forever: the poll
            # logged it once a minute and only ever acted on the SIMULTANEOUS-stall
            # threshold, which one chunk never reaches. Measured cost of that on
            # 2026-08-05: 20 actors pinned ~7 h after all other work finished, on a
            # cell whose useful work cost a fraction of it.
            #
            # An in-process timeout cannot fix this. The hang is inside a CUDA call
            # that never returns, so no Python-level timeout can interrupt the
            # thread, and forcing past one would leave the GPU context in an
            # unknown state — turning a loud, localised stall into quiet corruption
            # on later chunks. Killing the actor is the only way to reclaim the GPU,
            # and it routes into the SAME path a crashed actor already takes:
            # requeue the chunk (bounded by max_chunk_retries), hand back any
            # deferred write, return the reserved chunk, replace the slot.
            # A chunk can be BOTH ready and past the stall threshold on the same tick:
            # `ray.wait` returned its ref, but the tracker's last progress report is
            # older than the threshold and the result has not been processed yet, so
            # the item is still in `pending`. Recovering it would pop the ref here and
            # the ready loop below would pop it again — a KeyError that aborts the
            # whole fleet exactly when a long-stalled chunk finally succeeds. A ready
            # ref has a result waiting, so the ready path is the right one: killing its
            # actor would throw away work that is already done.
            ready_uids = {it.uid for it, _ in (pool.pending[r] for r in ready_refs if r in pool.pending)}
            for uid in wedged:
                if uid in ready_uids:
                    continue
                # POP the pending ref before handling. The killed actor's object ref
                # will never resolve, and `_handle_failure` is written for the
                # ready-refs path where the ref is already popped — leaving it in
                # `pending` would have `ray.wait` hand back the same item later and
                # process it twice.
                ref = next((r for r, (it, _) in pool.pending.items() if it.uid == uid), None)
                if ref is None:
                    # It finished between the poll and here, or its actor already
                    # died and was handled. Either way there is nothing to kill.
                    continue
                item, actor_idx = pool.pending.pop(ref)
                log.error(
                    "STALL RECOVERY: chunk %s made no progress for >%.0fs — killing actor %d "
                    "and re-queuing. The GPU cannot be reclaimed any other way.",
                    item.chunk.label,
                    stall_recovery_sec,
                    actor_idx,
                )
                _handle_failure(item, actor_idx, f"stalled >{stall_recovery_sec:.0f}s with no batch progress")

        # --- Handle completed (or failed) chunks ---
        for ref in ready_refs:
            item, actor_idx = pool.pending.pop(ref)

            try:
                result = ray.get(ref)
            except Exception as e:
                # The actor process died (OOM, segfault, CUDA process abort) so
                # ray.get raised. Stringify and route through the same handling
                # as an actor that caught its own error and returned "failed".
                _handle_failure(item, actor_idx, str(e))
            else:
                # ray.get succeeded, but the actor catches its own exceptions and
                # returns status="failed" rather than raising (see actors.py). A
                # sporadic CUDA error arrives this way with the actor still alive,
                # so failed results take the same kill-and-retry path as a death —
                # otherwise the chunk would never retry and the wedged actor, which
                # tends to keep failing, would be handed the next chunk below.
                if result.get("status") == "failed":
                    _handle_failure(item, actor_idx, str(result.get("error", "unknown")))
                else:
                    # Resolve the PREVIOUS deferred write this result carries
                    # before recording this chunk's own deferral.
                    prior = result.pop("prior_write", None)
                    if prior is not None:
                        _finalize_prior_write(actor_idx, prior)
                    if result.get("write_deferred"):
                        # Inference is done and the upload is in flight; hold
                        # the result until the write outcome confirms it. The
                        # tracker entry is removed now — the actor has moved on,
                        # so stall detection has nothing left to watch here.
                        pending_write[actor_idx] = (item, result)
                    else:
                        results.append(result)
                        if on_item_done is not None:
                            on_item_done(item, result)
                    if tracker:
                        tracker.remove.remote(item.uid)  # type: ignore[union-attr]  # run-qualified (see chunk_uid)
                    pool._initializing.discard(actor_idx)

            # Immediately re-feed the actor that just freed up, unless it's a
            # freshly-spawned replacement still initializing (30-120s). A failed
            # actor was killed and replaced by _handle_failure, so it's now
            # initializing and skipped here — its retried chunk goes to a
            # different, healthy actor via dispatch_idle below. The actor's
            # reserved chunk (whose starter payload it may have prefetched)
            # takes precedence over the queue so the prefetch is consumed.
            if actor_idx not in pool._initializing:
                next_work = pool.take_reserved(actor_idx)
                if next_work is None and chunk_queue:
                    next_work = _as_item(chunk_queue.popleft(), default_ctx)
                if next_work is not None:
                    pool.submit(actor_idx, next_work, tracker=tracker, chunk_queue=chunk_queue)

        # Request the next actor batch if the prior batch has been placed (or
        # placement timed out), then check if any initializing actors have
        # finished __init__ and can start receiving work, then dispatch queued
        # chunks to all idle actors, then retire actors idle past the grace
        # period.
        _maybe_request_next_batch()
        pool.resolve_initializing()
        pool.dispatch_idle(chunk_queue, mosaic_base, staging_base, run_id, tracker)
        # Actors left idle after dispatch have no next call to carry their
        # deferred-write confirmation — pull it via flush_writes() (tail of
        # run, and always before such an actor could be retired). Runs even
        # when retirement is gated off: a chained multi-zone fill still needs
        # its zone-tail writes confirmed before the zone's assembly can verify
        # staged completeness.
        _flush_idle_writes()
        # Retirement additionally waits for source exhaustion: while more
        # zones may arrive, an idle actor is the next zone's fleet, and
        # retiring it would idle-drain the shared cluster's instances.
        if retire_idle_actors and not source_active:
            pool.retire_idle(pool.outstanding_work(len(chunk_queue)))

    return results
