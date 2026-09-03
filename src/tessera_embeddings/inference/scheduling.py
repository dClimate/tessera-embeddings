"""Work-stealing chunk scheduler for distributed Ray inference.

Manages actor lifecycle (replacement on death, idle retirement) and dispatches chunks across a
pool of InferenceActor handles.

Kept under ``src/inference`` because it is highly adapted to a chunk-based workflow; it could
reasonably move to ``infra/ray`` if Ray is adopted for other chunk-level distributed GPU tasks.
"""

from __future__ import annotations

import contextlib
import logging
import math
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
    ``_process_chunks_work_stealing``'s scalar args); a chained multi-zone session has one per
    zone, carried on every :class:`WorkItem` so chunks from consecutive zones can coexist in one
    scheduler queue. Frozen so equality is value-based — the reservation path uses ``ctx == ctx``
    to restrict prefetch hints to same-zone successors, a cross-zone hint prefetching from the
    wrong mosaic.

    ``time_window`` and ``s1_orbit`` are the CELL's, not the actor's, because an actor is built
    once with one config while a chained session's cells may span campaign YEARS and may resolve
    DIFFERENT orbits (parts of the globe are radar-free in principle, so a cell resolves
    ``"none"`` while the session asked for ``"both"``). Read from the actor's config instead, a
    cell of another year silently reads the wrong months, and an orbit mismatch can only be
    handled by deferring the cell and holding its mosaic against a bounded budget — past which
    the cell fails and its mosaic is deleted, so a full population of zones never completes.
    ``None`` means "use the actor's own config", which is what the single-ROI path passes.
    """

    mosaic_base: str
    staging_base: str
    run_id: str
    time_window: TimeWindow | None = None
    s1_orbit: str | None = None


@dataclass(frozen=True)
class WorkItem:
    """One chunk plus the zone context it must be processed under.

    ``uid`` (run_id-qualified label) keys all scheduler bookkeeping: chunk labels repeat across
    zones (every zone has a ``chunk_0_0``), so bare labels alias retry counts and requeues
    between a finishing zone's tail and the next zone's head. Run ids are campaign-unique per
    cell (input-fingerprinted), so ``run_id:label`` is collision-free.
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


class FleetDemand:
    """The one place that decides WHAT to publish and survives publishing it.

    Two call sites each computing their own answer produced a defect per forgotten term (PR
    #159); there is one answer and it lives here. The bound is the actor target, the work
    outstanding, and what has actually been REQUESTED, whichever is smallest — "requested" being
    the first batch at the cold start and the live pool afterwards. Bounding by the request is
    what keeps the fleet from outrunning the batching policy that exists to pace it.
    """

    ATTEMPTS = 3
    BACKOFF_S = 2.0

    def __init__(
        self,
        publish: Callable[[int], None] | None,
        num_gpus: float,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    ) -> None:
        self._publish = publish
        self._num_gpus = num_gpus
        self._log = log

    @staticmethod
    def machines(actors: int, num_gpus: float) -> int:
        """Single-GPU machines an actor count implies, which is not the same number.

        They coincide only at the production default of one GPU per actor. A CPU-only run
        reserves none and must ask for none. Fractional reservations PACK, and the packing
        decides the count: five actors at 0.4 fit two to a card and need three machines, not the
        two that scaling by the reservation reports.
        """
        if num_gpus <= 0 or num_gpus > 1:
            # Zero: a CPU-only run must not ask for GPU machines at all. Above one: every rung is
            # single-GPU and Ray cannot combine GPUs across nodes, so no machine this request
            # could buy could host the actor.
            return 0
        per_machine = math.floor(1 / num_gpus)
        return math.ceil(actors / per_machine) if per_machine >= 1 else actors

    def send(self, *, target: int, outstanding: int, requested: int, retry: bool = False) -> None:
        """Publish the fleet shape for this moment. Never raises into the caller.

        ``retry`` is for the cold start only: nothing else republishes until the scheduler loop
        starts, and under the drought this exists for, that loop never starts — so one transient
        failure there becomes the whole hours-long actor wait. Every later call is followed by
        another a moment afterwards, which is retry enough.
        """
        if self._publish is None:
            return
        want = self.machines(max(0, min(target, outstanding, requested)), self._num_gpus)
        attempts = self.ATTEMPTS if retry else 1
        failure: Exception | None = None
        for attempt in range(attempts):
            try:
                self._publish(want)
                return
            except Exception as exc:
                failure = exc
                if attempt + 1 < attempts:
                    time.sleep(self.BACKOFF_S)
        # `exc_info=failure`, NOT `exc_info=True`: this runs OUTSIDE the except block, where there
        # is no active exception, so `True` logs no traceback at all — a warning that names the
        # symptom and silently drops its cause.
        self._log.warning("Could not publish GPU fleet demand (want=%d)", want, exc_info=failure)

    def clear(self) -> None:
        """Drop the request. Machines it holds are exempt from idle termination, so a floor left
        standing pins an idle GPU fleet through the assembly that follows — and this is the LAST
        publication, so nothing retries it afterwards.
        """
        self.send(target=0, outstanding=0, requested=0, retry=True)


class ActorPool:
    """Mutable actor state plus the lifecycle operations over it: dispatch, instance-ID
    resolution, replacement on death, and idle retirement.
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

        # ref → (WorkItem, actor_idx). (Some pool unit tests insert bare labels in the WorkItem
        # slot; the pool itself only reads the actor index from entries it did not create.)
        self.pending: dict[ray.ObjectRef, tuple[Any, int]] = {}
        # Retry counts keyed by WorkItem.uid (run_id-qualified — bare labels collide across a
        # chained session's zones).
        self.chunk_attempts: dict[str, int] = {}

        # actor_idx → the item reserved as that actor's next assignment. Created at submit time
        # and passed to the actor as ``prefetch_hint`` so it can prefetch a BOUNDED starter
        # payload (mask + 256-row starter strip, hard-capped ~2 GiB — actors.py
        # ``_XCHUNK_PREFETCH_CAP_BYTES``) during the current chunk's tail inference. Reservations
        # stop when the queue is shallower than the live pool; a failed actor's reservation is
        # requeued to the front. Only SAME-ZONE successors are reserved — the actor prefetches
        # the hint from the CURRENT call's mosaic (see submit()).
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
        not capacity; for that see :func:`_joined_gpu_count`.
        """
        return len(self.actors) - len(self._retired)

    def outstanding_work(self, queued: int) -> int:
        """In-flight + queued + reserved chunks — the true remaining-work count.

        Reserved next-chunks live in neither ``pending`` nor the queue while a busy actor holds
        them; excluding them would understate remaining work, under-provisioning actor batches
        and over-retiring idle actors mid-run. The tail over-provision this can cause is bounded
        (the ``requested >= outstanding`` and ``target - requested`` caps) and self-corrects via
        idle retirement.
        """
        return len(self.pending) + queued + len(self.reserved)

    @property
    def max_actor_deaths(self) -> int:
        """Death count that signals a systemic failure.

        Tracks the CURRENT pool size so the threshold scales as later batches are added; a fixed
        batch-1 count trips the alarm far too early on a fleet built up incrementally.
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

        Actors whose ``get_instance_id`` call has completed leave ``_initializing`` and become
        eligible for ``dispatch_idle``. Called each main-loop iteration so late-joining actors
        start receiving work as soon as they are confirmed alive.

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

        ``work`` is a :class:`WorkItem`, or (legacy callers/tests) a bare ``ChunkSpec`` wrapped
        with the scalar path args into a single-zone item. When ``chunk_queue`` is supplied and
        still deep, the queue head is also popped and RESERVED as this actor's next assignment,
        riding along as ``prefetch_hint`` so the actor can prefetch that chunk's capped starter
        payload during this chunk's tail inference (see actors.py). Reservations stop once
        ``len(queue) <= live_count``: at the tail of a run a reserved chunk pins work to a busy
        actor while others sit idle, costing more than the prologue overlap saves. A queue head
        from a DIFFERENT zone is never reserved — the actor prefetches the hint from the current
        call's mosaic — so the next zone's first chunks load serially instead.

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

        The queue rides along to ``submit()`` so each seeded actor also gets a next-chunk
        reservation while the queue is deep — its FIRST chunk can already overlap the second's
        prologue.

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

        Covers a replacement actor skipped in the main loop — other idle actors pick the work up
        immediately. The queue rides along to ``submit()`` so late-joining actors get the same
        next-chunk reservation as main-loop dispatches while the queue is deep.

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

        Fires a non-blocking ``get_instance_id.remote()`` so the real EC2 instance ID resolves
        lazily via ``resolve_iid()`` once the actor is live. Used for late-joining actors (from
        ``_wait_for_actors``) and for replacements spawned after a death.

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

        Each new slot is marked initializing (with a non-blocking ``get_instance_id`` fetch) so
        it flows into ``dispatch_idle`` once its ``__init__`` completes, exactly like a
        late-joining actor from the first batch. Called only from the single-threaded
        work-stealing loop, so no locking is required.

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

        Fires a non-blocking get_instance_id.remote() so the new EC2 instance ID resolves lazily
        before the next dispatch or error report.

        The outgoing actor is killed first. On a process death (OOM, segfault) that ``ray.kill``
        is a no-op; where the actor caught its error internally and is still alive (a sporadic
        CUDA error reported as ``status="failed"``), the kill is what removes the wedged worker
        so it cannot pick up more chunks.

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

        # A no-op if the actor already died; the actual removal when it caught its error and is
        # still live, so a chunk-failing actor is never handed another chunk.
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

        # The grace period avoids churn from momentary idleness between chunks, and keeps spare
        # capacity available when an in-flight chunk fails and is re-queued.
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


_CAPACITY_SAMPLE_INTERVAL_SEC = 30.0
"""Minimum seconds between cluster-capacity readings in the dispatch loop.

Half the loop's long ``ray.wait`` timeout, so an all-busy fleet still reads capacity once per
wait while a burst of completions — which the loop takes one at a time — collapses to one
reading instead of one per chunk. The figure this feeds is read by a human deciding whether to
cap a fleet, so its resolution hardly matters; its cost does. The reading is a synchronous
cluster-wide query on the path between a finished chunk and the actor's next one, so unthrottled
it buys precision with the idle GPU time it exists to describe.
"""


def _joined_gpu_count(last_known: float, gpus_per_actor: float) -> float:
    """GPUs JOINED to the cluster. A floor on what is billed, not the billed figure.

    Every GPU worker declares one GPU and the head declares none, so the cluster total is the
    joined worker count. TOTAL, not available: a GPU an actor is working on is still billed, so
    ``available_resources()`` — the idle remainder — is the wrong call. A slot Ray has accepted
    but not yet placed contributes nothing, which is the point.

    **Joined is not billed.** A worker starts charging when its instance launches and Ray counts
    it only once bootstrap has joined it, so every worker's boot-and-bootstrap interval is billed
    and uncounted here. This is a deliberate LOWER BOUND — the safe direction for a number read
    to decide whether to cap a fleet — and its caller integrates it at the lower of each
    interval's endpoints, so the floor survives the integration and not merely the reading.

    **Scoped to this cluster, not to this pool.** In the attached-cluster mode a cluster may hold
    GPUs belonging to other work, which would be counted here; the campaign cannot reach that,
    deriving a cluster name per flow run so a fill owns its cluster alone. Zero is returned
    outright when the pool's actors request no GPU.

    The except is load-bearing, not defensive: the caller sits in the dispatch loop OUTSIDE the
    try/except that wraps the tracker poll, so an exception here would abort a fill over a log
    number. A failure carries the previous count forward instead.

    Args:
        last_known: Count to fall back to when the lookup fails.
        gpus_per_actor: GPUs each actor reserves. Zero means this run is not on GPUs at all.
    """
    if not gpus_per_actor:
        return 0.0
    try:
        return float(ray.cluster_resources().get("GPU", 0))
    except Exception:
        return last_known


def _placed_actor_slots(joined_gpus: float, gpus_per_actor: float) -> int:
    """Actor slots the cluster's joined GPUs can hold — the unit an actor request is in.

    A node count is NOT this quantity, and using one stalls the fleet permanently wherever a
    node holds more than one actor: reserve half a GPU each and 25 actors sit on 13 nodes, so a
    rule comparing a request to a node count converges to a fixed point far below the target and
    then asks for nothing, for ever. Slots and nodes coincide only at exactly one actor per node,
    the production shape, which is why the error is invisible there.

    Reads GPUs because a GPU is what an actor reserves; :func:`_joined_gpu_count` is the reading
    and it is a FLOOR, so this under-states the room to grow and errs toward asking for less.

    Args:
        joined_gpus: GPUs joined to the cluster, from :func:`_joined_gpu_count`.
        gpus_per_actor: GPUs each actor reserves. Zero means this run is not on GPUs,
            for which the request bound is refused at config time rather than guessed at.

    Returns:
        Whole slots. Fractional room is rounded down, keeping the floor a floor.
    """
    if not gpus_per_actor:
        return 0
    return int(joined_gpus / gpus_per_actor)


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

    Holds no pool or loop state — all context is passed explicitly so it can be tested in
    isolation — so it cannot kill an actor itself: it NAMES the chunks whose actors the caller
    (which holds the pool) should replace.

    Args:
        tracker: ProgressTracker Ray actor handle.
        n_done: Number of chunks already completed (for the progress log line).
        n_total: Total chunks in the run.
        stall_threshold_sec: Seconds without a batch update before a chunk is
            considered stalled — the WARNING threshold.
        max_simultaneous_stalls: Number of simultaneous stalls that triggers a
            systemic abort (RuntimeError).
        log: Logger.
        elapsed_min: Minutes since INFERENCE began — the dispatch loop's own start, not the run's,
            which would also count ingest, cluster bringup and model load. Folded into the single
            progress line (this is the ONLY progress log line — keep it that way).
        gpu_hours: Fleet GPU-hours consumed so far — the cluster's JOINED GPU count integrated
            over wall time (see :func:`_joined_gpu_count`, a FLOOR on what is billed) — folded
            into the same line.
        recovery_threshold_sec: Seconds without an update before a chunk is declared
            unrecoverable in place and returned for kill-and-requeue. Deliberately well above
            ``stall_threshold_sec`` so a warning always fires long before anything is killed and
            a merely-slow chunk is never destroyed. ``None`` disables recovery (log-only).

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
            # "inferring", not "elapsed": the number counts from the dispatch loop's start, and a
            # bare "elapsed" beside a chunk counter invites reading it as run wall-clock.
            elapsed = f" — {elapsed_min:.1f} min inferring" if elapsed_min is not None else ""
            gpu = f", {gpu_hours:.1f} GPU-hrs" if gpu_hours is not None else ""
            # "chunks done" labels the whole first clause: what follows counts chunks in flight,
            # not actor slots and not GPUs. GPU-hrs carries its own unit.
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
        # The tracker is a monitoring aid, never a single point of failure: swallow Ray errors
        # (actor death, timeout, serialization) so inference continues without visibility.
        if isinstance(exc, RuntimeError):
            raise  # re-raise our own stall-abort RuntimeError
        log.debug("Tracker poll failed (non-fatal): %s", exc)
        # A failed poll says nothing about the chunks, so recover nothing — lost visibility must
        # not read as "everything is wedged".
        return []


ACTOR_REQUEST_HEADROOM = 25
"""How far an actor request may run ahead of the GPU nodes the fleet actually holds.

The recommended value for ``InferenceConfig.actor_request_headroom``, and the number the whole
rule turns on: a run asks for what it has plus this, never for its target.

Larger ramps a fleet to full width in fewer steps when capacity is free, and wastes more
requests on a region that cannot fill them — every request that never places is retried by the
autoscaler against the account's launch quota, so the surplus is paid in throttling the run then
waits out. A constant rather than a function of the target because that quota is a property of
the ACCOUNT: a fleet of 250 and a fleet of 20 draw on the same bucket, and sizing the allowance
to the ask is what let the larger one drown the smaller. Sizing evidence:
``context_docs/design/ec2_launch_throttling_2026_08.md``.
"""


def _batch_actors_to_request(
    *,
    requested: int,
    target: int,
    outstanding: int,
    placed_actor_slots: int,
    nodes_at_last_batch: int,
    last_batch_size: int,
    secs_since_last_batch: float,
    placement_timeout_sec: float,
    batch_size: int,
    placement_tolerance: int = 2,
    headroom: int | None = None,
) -> tuple[int, bool]:
    """Decide how many actors to request for the next batch.

    Pure decision function (no Ray calls) so the gate can be unit-tested in isolation.

    **Without ``headroom`` (the legacy default)** the next batch is requested once the prior
    batch's instances have joined the cluster, or once placement has timed out so a capacity
    shortfall cannot gate every remaining batch forever. Placement is measured *incrementally* —
    slots joined since the last batch was requested, against that batch's size — rather than
    against the cumulative requested count, so one timed-out partial batch does not permanently
    gate every later one: a subsequent batch's own increment satisfies the check even though the
    earlier shortfall is never made up.

    **``headroom`` replaces all of that with one rule**, because the timeout escape hatch is what
    fails under a real shortage: nothing places, every interval expires, and the request climbs
    to the target on no evidence at all — each request that never places then being retried by
    the autoscaler against an account-wide launch quota, so the run manufactures the throttling
    that keeps it from growing. With ``headroom`` set the request may never exceed the GPU nodes
    the fleet actually holds plus that constant: a small fixed distance ahead of reality rather
    than a climb toward the target, needing no gate at all, since each placement earns the right
    to ask for a little more. Neither ``placement_timeout_sec`` nor the placement gate is
    consulted here; they and ``nodes_at_last_batch``, ``last_batch_size``,
    ``secs_since_last_batch`` and ``placement_tolerance`` remain only for the legacy path.

    **Placed SLOTS, not ready actors and not nodes.** ``placed_actor_slots`` counts the actor
    slots the cluster's joined GPUs can hold, whether or not the actors on them have loaded
    their checkpoint: readiness is the wrong measure because the bound is about hardware the run
    already holds and pays for, and a slow model load would read as a capacity shortage. A NODE
    count is worse — equal to slots only at exactly one actor per node, and anywhere else the
    recurrence reaches a fixed point below the target and stops asking permanently (see
    :func:`_placed_actor_slots`). It is also what makes the rule safe: a request is refused only
    while the fleet is already a full headroom ahead of its placed slots, and every slot that
    joins re-opens exactly that much room, so the fleet cannot stop asking. A run reserving no
    GPU cannot express the bound at all, and ``InferenceConfig`` refuses that combination rather
    than letting it stall here. At zero slots the bound still permits ``headroom`` requests,
    which is what lets a run begin, and ``outstanding`` remains a second ceiling — the smaller
    of the two wins, so a short tail stays safe from over-provisioning.

    Full derivation: ``context_docs/design/ec2_launch_throttling_2026_08.md``.

    Args:
        requested: Actors requested so far (``len(pool.actors)``).
        target: Total actors the run should eventually reach.
        outstanding: In-flight + queued chunks. Caps requests so we do not provision more
            actors than there is work left.
        placed_actor_slots: Actor slots the cluster's joined GPUs can currently hold
            (:func:`_placed_actor_slots`).
        nodes_at_last_batch: Slots placed when the last batch was requested — the baseline for
            the incremental placement check.
        last_batch_size: Actors in the last batch requested; the increment expected to join
            before that batch counts as placed.
        secs_since_last_batch: Seconds since the last batch was requested.
        placement_timeout_sec: Placement-wait escape hatch.
        batch_size: Actors per batch.
        placement_tolerance: Stragglers tolerated before the prior batch counts as placed, so
            one slow instance does not gate the next request.
        headroom: How far the request may run ahead of the fleet's placed GPU nodes. ``None``
            selects the legacy path — a placement gate with a timeout escape hatch, and no
            limit on how far the request runs ahead. Pass :data:`ACTOR_REQUEST_HEADROOM`.

    Returns:
        ``(n, timed_out)`` — actors to request (0 = none) and whether the placement timeout,
        rather than placement itself, allowed it. The flag is only meaningful when ``n > 0``
        and the caller uses it for logging; always ``False`` under ``headroom``.
    """
    if requested >= target:
        return 0, False
    # Once the pool holds at least as many actors as there is work left, no further batches.
    if requested >= outstanding:
        return 0, False
    if headroom is not None:
        # Both ceilings, plus the batch step. The placement ceiling makes a drought stop the
        # fleet growing rather than run the request away from it; `outstanding` keeps a short
        # tail from over-provisioning.
        allowed = placed_actor_slots + headroom - requested
        return max(0, min(batch_size, target - requested, outstanding - requested, allowed)), False
    placed = placed_actor_slots - nodes_at_last_batch >= last_batch_size - placement_tolerance
    timed_out = secs_since_last_batch > placement_timeout_sec
    if not (placed or timed_out):
        return 0, False
    # Clamp to remaining target and remaining work, so a partial final batch never
    # over-provisions past what is left to do.
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
    fleet: FleetDemand | None = None,
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

    Seeds each actor with one chunk, then dispatches the next queued chunk to whichever actor
    finishes first, which load-balances naturally.

    When a chunk fails — the actor process died (OOM, segfault), or the actor caught its own
    error and returned ``status="failed"`` (a sporadic CUDA error) — the failing actor is killed
    and replaced and the chunk re-queued up to ``max_chunk_retries`` times, so the retry lands on
    a healthy worker and an actor that has started failing, which tends to keep failing, is not
    handed more chunks. This is what stops one transient failure killing a multi-hour GPU run.

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
        still_initializing: Actor indices that haven't finished init yet. Included in the pool
            but skipped during seeding; they pick up work via ``dispatch_idle`` once online.
        on_actor_retire: Callback fired when an actor is removed from the pool. Used to
            terminate its EC2 instance promptly rather than waiting for the autoscaler.
        fleet: Optional :class:`FleetDemand`, published once per scheduling round so a
            capacity-refused rung cannot starve the fleet. Runs on the scheduler thread.
        get_credentials: Optional icechunk S3 credential provider injected into every actor
            (seeded and replacement) so store opens refresh creds.
        s3_region: Optional S3 region injected into every actor (seeded and replacement) so
            reads open the mosaic in the caller's region.
        actor_factory: Optional callable ``(n) -> [handles]`` requesting ``n`` new actors. With
            ``total_actors_target``, the loop requests actors in batches: the caller supplies
            the first, later ones are requested here. ``None`` disables batching — the caller's
            ``actors`` list is the whole fleet.
        total_actors_target: Total actors the run should eventually reach; requesting stops
            once ``len(pool.actors)`` reaches it. Ignored when ``actor_factory`` is None.
        placement_timeout_sec: Max seconds to wait for a batch's instances to be placed before
            requesting the next anyway (capacity-shortfall escape hatch).
        retire_idle_actors: Kill actors idle past the grace period as the run's tail drains
            (the default). A chained multi-zone fill leaves this True and relies on the
            ``more_work`` gate below; a caller managing zones as separate calls passes False
            for every zone but its last, since the "surplus" workers are the NEXT zone's fleet
            and retiring them idle-drains the shared cluster's instances at every zone tail
            (see ``orchestration.runners.sequential_fill``).
        more_work: Optional pull source for a chained multi-zone session. Polled (non-blocking)
            whenever the queue is at or below the live actor count: a list extends the queue
            (its items carry their own :class:`ZoneContext`), ``[]`` means nothing ready YET
            (the loop stays alive and keeps polling), ``None`` means exhausted forever. While
            unexhausted, idle retirement is suppressed — apparently-idle actors are the next
            zone's fleet — and the loop does not exit on an empty queue.
        on_item_done: Optional callback fired exactly once per work item at its FINAL outcome —
            success (after any deferred write confirms) or permanent failure — with the item
            and its result dict. A chained session uses it for per-zone completion accounting;
            it runs on the scheduler thread, so it must not block.

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

    # Mark actors that haven't finished __init__ yet. seed() skips them; they receive work via
    # dispatch_idle once resolve_initializing confirms they are alive. Avoids sending chunks to
    # actors whose EC2 instances may never provision (AWS capacity limits).
    if still_initializing:
        for idx in still_initializing:
            pool.mark_initializing(idx)

    pool.seed(chunk_queue, mosaic_base, staging_base, run_id, tracker)

    results: list[dict] = []
    stall_threshold_sec = 300.0

    #: Seconds of no batch progress before a chunk's actor is killed and the chunk re-queued.
    #: FOUR TIMES the warning threshold, so several STALL lines always precede a kill and a
    #: merely-slow chunk is never destroyed. The margin is enormous relative to legitimate work:
    #: this measures the gap BETWEEN BATCH UPDATES, normally well under a second, while the
    #: slowest whole chunk yet observed took ~480 s in total and the wedge this exists for sat
    #: 14,396 s. The choice is between minutes and forever, so anywhere in that range works.
    stall_recovery_sec = 4 * stall_threshold_sec

    # Scales with the EVENTUAL fleet size, not just the first batch: with batching, ``actors``
    # holds only the initial subset, so a threshold frozen at batch-1/10 would abort a large run
    # after a handful of stalls once later batches join. Recomputed each iteration against the
    # larger of the current pool and the target.
    def _stall_threshold() -> int:
        fleet = max(len(pool.actors), total_actors_target or 0)
        return max(3, fleet // 10)

    # --- Actor-batch requesting ---
    # The caller supplies only the first batch; the loop requests the rest here, pacing the
    # demand the autoscaler forwards to AWS. The gate is placement (slots the joined GPUs can
    # hold), not readiness, so a slow checkpoint load never stalls the next AWS ask.
    batching_enabled = (
        actor_factory is not None and total_actors_target is not None and config.actor_request_batch_size > 0
    )
    last_batch_at = time.monotonic()
    # Placement baseline for the incremental check in _batch_actors_to_request. The caller
    # already requested the first batch, so seed last_batch_size from the pool we were handed;
    # nodes_at_last_batch starts at 0 because no slot is guaranteed placed before the first is.
    nodes_at_last_batch = 0
    last_batch_size = len(pool.actors)
    # Carried across calls because the reading can fail, and a failed reading must report the
    # previous count: zero would read as a fleet that has placed nothing and refuse to grow.
    last_joined_gpus = 0.0

    def _publish_fleet_demand() -> None:
        """State the fleet's shape for this round. Called AFTER any new batch is added so those
        actors are counted; publishing first leaves a freshly requested batch creating ordinary
        primary-only demand for a whole round.
        """
        if fleet is not None and total_actors_target is not None:
            fleet.send(
                target=total_actors_target,
                outstanding=pool.outstanding_work(len(chunk_queue)),
                requested=len(pool.actors),
            )

    def _maybe_request_next_batch() -> None:
        nonlocal last_batch_at, nodes_at_last_batch, last_batch_size, last_joined_gpus
        if not batching_enabled:
            _publish_fleet_demand()
            return
        assert actor_factory is not None and total_actors_target is not None  # narrowed by batching_enabled
        last_joined_gpus = _joined_gpu_count(last_joined_gpus, config.num_gpus)
        placed_actor_slots = _placed_actor_slots(last_joined_gpus, config.num_gpus)
        n, timed_out = _batch_actors_to_request(
            requested=len(pool.actors),
            target=total_actors_target,
            outstanding=pool.outstanding_work(len(chunk_queue)),
            placed_actor_slots=placed_actor_slots,
            nodes_at_last_batch=nodes_at_last_batch,
            last_batch_size=last_batch_size,
            secs_since_last_batch=time.monotonic() - last_batch_at,
            placement_timeout_sec=placement_timeout_sec,
            batch_size=config.actor_request_batch_size,
            headroom=config.actor_request_headroom,
        )
        if n == 0:
            _publish_fleet_demand()
            return
        pool.add_actors(actor_factory(n))
        last_batch_at = time.monotonic()
        nodes_at_last_batch = placed_actor_slots
        last_batch_size = n
        log.info(
            "Requested actor batch: +%d (%d/%d total, %d actor slots placed)%s",
            n,
            len(pool.actors),
            total_actors_target,
            placed_actor_slots,
            " — placement timed out, requesting anyway" if timed_out else "",
        )
        _publish_fleet_demand()

    def _handle_failure(item: WorkItem, actor_idx: int, error: str) -> None:
        """Retry a failed chunk on a different worker and kill the failing actor.

        Shared by both failure modes: a dead actor process (ray.get raised) and an actor that
        caught its own error and returned status="failed" while still alive. Either way the
        chunk is re-queued up to ``max_chunk_retries`` times and the actor killed and replaced,
        so the retry lands on a healthy worker and the failing one leaves rotation.

        Args:
            item: The work item that failed.
            actor_idx: Slot of the actor that failed it.
            error: Error string (from the exception or the failed result dict).
        """
        chunk_label = item.chunk.label
        if tracker:
            tracker.remove.remote(item.uid)  # type: ignore[union-attr]  # run-qualified: labels alias across zones
        # The actor is killed below, taking its writer thread with it, so any deferred write it
        # held is of unknown state and that chunk requeues too. Safe: staged writes are
        # idempotent (run-scoped keys, mode="w"; the resume scan tolerates rewrites).
        orphaned = pending_write.pop(actor_idx, None)
        if orphaned is not None:
            _requeue_unconfirmed(orphaned, f"actor {actor_idx} failed with the write in flight")

        # Its reserved next chunk (whose starter payload only this actor may have prefetched)
        # goes back to the queue FRONT for a healthy actor.
        reserved = pool.take_reserved(actor_idx)
        if reserved is not None:
            chunk_queue.appendleft(reserved)
            log.info(
                "Returned reserved chunk %s from failed actor %d to the queue front", reserved.chunk.label, actor_idx
            )
        pool.resolve_iid(actor_idx)
        instance_id = pool.actor_instance_ids[actor_idx]
        attempts = pool.chunk_attempts.get(item.uid, 1)
        # attempts starts at 1 on first submission, so max_chunk_retries=2 means first try + 2
        # re-queues = 3 total attempts.
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

        # Kill and replace (queues Ray init in the background); the kill is a no-op if the actor
        # process already died.
        pool.replace(actor_idx, instance_id)

    # --- Deferred staging writes ---
    # Actors return with write_deferred=True while their staging upload runs on a background
    # thread (overlapping the next chunk's serial prologue). Such results are held here — NOT
    # counted complete — until the write's outcome arrives: piggybacked as prior_write on the
    # actor's next result, or pulled via flush_writes() once the actor idles. On failure or actor
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
            # A plain write error leaves the actor healthy (it just inferred a whole chunk), so
            # requeue without the kill-and-replace used for inference failures.
            _requeue_unconfirmed(entry, str(prior.get("error", "unknown write error")))
            if prior.get("timed_out"):
                # ...but a TIMEOUT means the upload is still wedged in the actor's single-slot
                # writer pool, so later writes would queue behind the stuck task and time out
                # too. Replace the slot, reaping the writer. Only reached via the idle-flush path
                # — the hot path raises in process_chunk — so this actor has no chunk in hand.
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
                # A failed or timed-out flush RPC may leave the (serial) actor still wedged
                # inside flush_writes(); a retry dispatched to it would sit behind that stuck
                # call forever, and invisibly — a chunk that never starts has no tracker entry
                # for stall detection. Kill and replace so the requeued chunk lands on a healthy
                # actor. Safe: staged writes are idempotent.
                pool.resolve_iid(actor_idx)
                pool.replace(actor_idx, pool.actor_instance_ids[actor_idx])
                continue
            if prior is None:
                _requeue_unconfirmed(pending_write.pop(actor_idx), "actor had no pending write to flush")
            else:
                _finalize_prior_write(actor_idx, prior)

    # --- Main work-stealing loop ---
    # gpu_seconds integrates JOINED GPUs over wall time so the progress line can report fleet
    # GPU-hours consumed. Joined, not requested: a slot the pool has asked for but Ray has not
    # placed costs nothing until its instance exists. Logging-only — nothing branches on it.
    gpu_seconds = 0.0
    # Seeded with a real reading rather than zero: the loop starts once actors are ready, so
    # capacity is already non-zero, and the first interval is charged at the lower of its
    # endpoints — against a zero seed that is zero, so the fleet's first interval would be free.
    joined_gpus = _joined_gpu_count(0.0, config.num_gpus)
    last_tick = time.monotonic()
    # The progress line's elapsed clock, starting HERE rather than at ``t0``. ``t0`` is the whole
    # run's start — for a chained session the top of the stream, before the ingest look-ahead,
    # ``ray up``, per-worker EC2 bringup and the model load — so beside a chunk counter it read as
    # though inference had been running that long (a cell logged "0/10 done, 5 active ... 46.3 min
    # elapsed" seconds after its actors came ready) and disagreed with ``gpu_seconds``, which
    # starts at zero right here. ``t0`` keeps its other uses (run summaries, provenance).
    inference_t0 = last_tick
    # A chained session's work source: while unexhausted the loop must stay alive through
    # empty-queue gaps (the next zone may still be ingesting) and must not retire "idle" actors
    # (they are the next zone's fleet).
    source_active = more_work is not None
    # Stay alive while any work remains: in-flight chunks, deferred writes awaiting confirmation,
    # queued chunks with a live actor to run them, or an unexhausted work source. The queue clause
    # must NOT be gated on _initializing alone — a failed tail flush (_flush_idle_writes, after
    # dispatch) requeues its chunk when no actor is initializing, and that retry would be dropped.
    # `live_count > 0` prevents a busy-spin when every actor has died.
    while pool.pending or pending_write or ((chunk_queue or source_active) and pool.live_count > 0):
        # Top up from the work source BEFORE waiting so freshly-ready zones dispatch this
        # iteration. Polled only when the queue is at or below the live actor count. For zones at
        # least as large as the fleet this keeps them near-sequential (one zone's tail overlaps
        # the next zone's head); a smaller zone cannot fill the fleet alone, so successive small
        # zones are pulled to pack it — one zone per poll, the caller bounding how many coexist.
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
            # Block for any one chunk to finish. At a zone boundary — source still active and the
            # queue drained to the poll trigger — the next zone may become ready momentarily, so
            # cap the wait short: blocking the full 60 s on a single tail task idles the rest of
            # the fleet up to a GPU-minute at every boundary.
            at_boundary = source_active and len(chunk_queue) <= pool.live_count
            wait_timeout = 5 if at_boundary else 60
            ready_refs, _ = ray.wait(list(pool.pending.keys()), num_returns=1, timeout=wait_timeout)
        else:
            # Nothing in-flight but chunks are queued for initializing actors. Sleep briefly,
            # then check whether any actor is ready.
            time.sleep(5)
            ready_refs = []
        now = time.monotonic()
        # Sampled on a timer, NOT once per completion: `ray.wait(num_returns=1)` hands back one
        # chunk at a time, so a burst of completions would put a synchronous cluster-wide query
        # between every finished chunk and the actor's next one — spending the idle GPU time this
        # figure exists to describe on describing it. Between samples the previous reading is
        # carried forward and the interval left uncharged, the same shape the failed-lookup path
        # has.
        #
        # An interval is charged at the LOWER of its two endpoints, which is what keeps this the
        # lower bound it claims to be: capacity moves in steps inside an interval a blocking
        # `ray.wait` can already stretch to a minute, and since the step's timing is unknown only
        # the smaller endpoint is safe in both directions (an average would assume it fell
        # halfway). Sampling less often only LOOSENS that bound, never breaks it.
        if now - last_tick >= _CAPACITY_SAMPLE_INTERVAL_SEC:
            previous_gpus = joined_gpus
            joined_gpus = _joined_gpu_count(joined_gpus, pool.config.num_gpus)
            gpu_seconds += min(previous_gpus, joined_gpus) * (now - last_tick)
            last_tick = now

        # Polled every iteration, including timeouts with no completions, so stall detection
        # stays responsive.
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
            # Why a single wedged chunk is killed rather than merely logged: the poll only ever
            # acted on the SIMULTANEOUS-stall threshold, which one chunk never reaches, so on
            # 2026-08-05 one wedge pinned 20 actors for ~7 h after all other work had finished.
            #
            # An in-process timeout cannot fix this. The hang is inside a CUDA call that never
            # returns, so no Python-level timeout can interrupt the thread, and forcing past one
            # would leave the GPU context in an unknown state — turning a loud, localised stall
            # into quiet corruption on later chunks. Killing the actor is the only way to reclaim
            # the GPU, and it routes into the SAME path a crashed actor takes: requeue the chunk
            # (bounded by max_chunk_retries), hand back any deferred write, return the reserved
            # chunk, replace the slot.
            #
            # A chunk can be BOTH ready and past the stall threshold on one tick: `ray.wait`
            # returned its ref while the tracker's last progress report is older than the
            # threshold and the result is unprocessed, so the item is still in `pending`.
            # Recovering it would pop the ref here and the ready loop below would pop it again —
            # a KeyError that aborts the whole fleet exactly when a long-stalled chunk finally
            # succeeds. A ready ref has a result waiting, so the ready path wins.
            ready_uids = {it.uid for it, _ in (pool.pending[r] for r in ready_refs if r in pool.pending)}
            for uid in wedged:
                if uid in ready_uids:
                    continue
                # POP the pending ref before handling: the killed actor's object ref will never
                # resolve, and `_handle_failure` expects the ready-refs path where the ref is
                # already popped — left in `pending`, `ray.wait` hands the item back later and it
                # is processed twice.
                ref = next((r for r, (it, _) in pool.pending.items() if it.uid == uid), None)
                if ref is None:
                    # Finished between the poll and here, or its actor already died and was
                    # handled. Either way there is nothing to kill.
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
                # The actor process died (OOM, segfault, CUDA process abort) so ray.get raised.
                # Route through the same handling as a caught-and-returned "failed".
                _handle_failure(item, actor_idx, str(e))
            else:
                # ray.get succeeded, but the actor catches its own exceptions and returns
                # status="failed" rather than raising (see actors.py). A sporadic CUDA error
                # arrives this way with the actor still alive, so failed results take the same
                # kill-and-retry path as a death — otherwise the chunk never retries and the
                # wedged actor is handed the next chunk below.
                if result.get("status") == "failed":
                    _handle_failure(item, actor_idx, str(result.get("error", "unknown")))
                else:
                    # Resolve the PREVIOUS deferred write this result carries before recording
                    # this chunk's own deferral.
                    prior = result.pop("prior_write", None)
                    if prior is not None:
                        _finalize_prior_write(actor_idx, prior)
                    if result.get("write_deferred"):
                        # Inference is done and the upload in flight; hold the result until the
                        # write outcome confirms it. The tracker entry goes now — the actor has
                        # moved on, so stall detection has nothing left to watch here.
                        pending_write[actor_idx] = (item, result)
                    else:
                        results.append(result)
                        if on_item_done is not None:
                            on_item_done(item, result)
                    if tracker:
                        tracker.remove.remote(item.uid)  # type: ignore[union-attr]  # run-qualified (see chunk_uid)
                    pool._initializing.discard(actor_idx)

            # Immediately re-feed the actor that just freed up, unless it is a freshly-spawned
            # replacement still initializing (30-120 s). A failed actor was killed and replaced
            # by _handle_failure, so it is initializing and skipped here — its retried chunk goes
            # to a healthy actor via dispatch_idle below. The actor's reserved chunk takes
            # precedence over the queue so its prefetch is consumed.
            if actor_idx not in pool._initializing:
                next_work = pool.take_reserved(actor_idx)
                if next_work is None and chunk_queue:
                    next_work = _as_item(chunk_queue.popleft(), default_ctx)
                if next_work is not None:
                    pool.submit(actor_idx, next_work, tracker=tracker, chunk_queue=chunk_queue)

        # Request the next actor batch, promote any initializing actor that has finished
        # __init__, dispatch queued chunks to all idle actors, then retire the long-idle.
        _maybe_request_next_batch()
        pool.resolve_initializing()
        pool.dispatch_idle(chunk_queue, mosaic_base, staging_base, run_id, tracker)
        # Actors left idle after dispatch have no next call to carry their deferred-write
        # confirmation — pull it via flush_writes(), always before such an actor could be
        # retired. Runs even when retirement is gated off: a chained multi-zone fill still needs
        # its zone-tail writes confirmed before that zone's assembly can verify staged
        # completeness.
        _flush_idle_writes()
        # Retirement additionally waits for source exhaustion: while more zones may arrive, an
        # idle actor is the next zone's fleet and retiring it idle-drains the shared cluster.
        if retire_idle_actors and not source_active:
            pool.retire_idle(pool.outstanding_work(len(chunk_queue)))

    return results
