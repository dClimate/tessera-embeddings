"""Chained multi-zone fill: many (zone, year) cells through ONE Ray session.

The per-cell fill chain (:mod:`.zone_fill`) provisions nothing itself — the caller owns the
Ray context. This runner exploits that with **one long-lived cluster whose actors are created
once and stream through every zone**, so the per-cluster costs (``ray up``, per-worker EC2
bringup of minutes each, the model-load cold start on every worker) are paid once per CLUSTER
of a campaign year instead of once per zone. Throughout: "cluster" = one Ray cluster and the
UTM zones assigned to it, "zone" = a UTM zone, "shard" = a storage shard.

Keeping the shared fleet busy is the whole game; this docstring is the canonical statement of
that rationale, and the flow and README point here.

- **Cross-zone interleaving at exhaustion**: every zone flows through ONE work-stealing
  session (``run_inference`` with a ``more_work`` source). When the current zone's queue drops
  to the live actor count the next prepared zone tops it up, so a zone's tail no longer idles
  the fleet for ~half a tile-duration per actor, and actors are never re-created between zones
  (no per-zone model reload). At most one zone's tail overlaps the next zone's head.
- **Readiest-first** (``_take_next``): cells are ordered densest-first and the densest zone is
  also the slowest to ingest, so strict order makes the fleet wait on the last mosaic of the
  opening window while smaller ones sit finished. The feeder takes whichever PENDING cell has
  landed, else the head. Scanning the whole pending list rather than a window costs nothing —
  readiness is a ``Future.done()`` check with no I/O. Density order still sizes the session
  from the largest cell and puts the island tail last; it is simply no longer a barrier.
- **Nothing gates the three stages against each other.** GPUs are by far the most expensive
  resource here, so inference waits only on its own input:

  * INGEST runs ``1 + look_ahead`` cells at a time (which is all ``look_ahead`` sizes), with an
    ingest started for every pending cell up front so the next begins the moment one finishes.
  * INFERENCE is admitted without bound, paced only by ``inputs.wait`` on the chosen cell's
    own mosaic.
  * ASSEMBLY serialises on one trailing thread and may lag arbitrarily far behind inference,
    including past the end of it.

  What stays bounded is FAILURE, not throughput: a failed cell keeps its mosaic for staged
  resume and is counted, and the feeder stops admitting once ``max_retained_failures`` are
  outstanding. The price of decoupling is an assembly backlog, which is the cheap direction to
  fail. Measurements — 60 configured ingests running 7, and the ~1,380 tiles/hour that makes an
  assembly-released gate bind on the slowest stage: ``context_docs/campaign/campaign-plan.md`` §1.
- **In-child retry, on the STANDING fleet**: a failed cell goes to the BACK of the feeder's
  work queue and is served by the same session as everything else (``_readmit``, bounded by
  ``attempts_per_cell_in_cluster``). Without an in-child retry at all the driver's retry unit is
  a whole dispatch, so an early failure would wait for every cluster to finish its list; with
  one that ran after the stream, every retried cell built a GPU fleet from nothing and killed it
  again, because a fleet's lifetime is one ``run_inference`` call. An ingest failure gets a new
  ingest dispatched immediately and non-blockingly, then waits its turn behind the roster's
  other ingests while inference and assembly carry on. Only an ASSEMBLY failure is retried after
  the stream, because it is discovered too late to re-admit — and it provisions nothing.
- **Trailing assembly**: a completed zone's shard assembly runs on a background thread while
  later zones' tiles keep the GPUs busy. Assemblies serialise on one thread; a zone's mosaic
  delete is HANDED TO A SECOND POOL once its assembly lands, so the next assembly is not stuck
  behind a multi-terabyte delete. A zone counts as complete only when every tile's result is
  FINAL — the scheduler fires the completion callback after any deferred staging write
  confirms — so assembly never races an in-flight upload. Assembly is measured, and slower
  than it looks (design note above); nothing here depends on the backlog staying short.

Idle-actor retirement needs no per-zone gating here: the scheduler suppresses it while the
work source is unexhausted and resumes it for the true cluster tail (see
``scheduling._process_chunks_work_stealing``).

A zone whose mosaic resolves a DIFFERENT s1 orbit than the shared session's config STILL joins
the stream: the orbit travels on each cell's ``ZoneContext``, so an actor reads every cell
under that cell's orbit — the same mechanism that lets one session span campaign years. It has
to work that way because parts of the globe are radar-free in principle, so a cell resolving
``"none"`` against a ``"both"`` session is a permanent population rather than an anomaly.
There is consequently no post-stream fallback pass, and every prepared zone's tiles are
dispatchable.

Contracts: Prefect-free (the deployment-backed ingest adapter, the input-fingerprinted run_id,
the per-cell config/plan and the session itself all arrive as callables from the flow layer);
the caller is already inside a Ray context; and cells may span campaign YEARS as well as
zones. Two commits for the same zone group are never in flight, and this needs no caller
guarantee: assemblies serialise on the single trailing thread, so even a multi-year list of
one zone commits its years one after another. Each cell carries its OWN inference window on
its work items (``ZoneContext.time_window``) — actors are built once from the session config,
so a cell of another year read through that config would silently be inferred over the wrong
months.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from tessera_embeddings.config.fault_injection import ArmedFault
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.scheduling import WorkItem, ZoneContext
from tessera_embeddings.orchestration.runners.zone_fill import ZoneFillHandoff, ZonePlan, complete_zone_inference

if TYPE_CHECKING:
    from tessera_embeddings.config.inference import InferenceConfig


class CellInputs(Protocol):
    """Lifecycle of a cell's input mosaics, implemented by the flow layer.

    The runner drives *when* (start every pending cell up front, wait before planning, clean
    up after the cell lands); the implementation owns *how* (typically a Prefect ingest
    deployment per cell). Every method is keyed by ``(zone, year)`` and must be idempotent —
    ``start`` on an already-started cell and ``cleanup`` on a never-ingested one are no-ops.
    """

    def start(self, zone: str, year: int) -> None:
        """Begin producing the cell's mosaics without blocking."""
        ...

    def wait(self, zone: str, year: int, stop: threading.Event | None = None) -> None:
        """Block until the cell's mosaics are ready; raise if production failed.

        When ``stop`` is supplied the implementation must return promptly (by raising) once
        it is set — the runner passes its unwind event so a crashed session is never stuck
        behind a running ingest.
        """
        ...

    def cleanup(self, zone: str, year: int) -> None:
        """Delete the cell's mosaics (only if this adapter produced them).

        MUST RAISE if the delete did not happen: the runner treats a clean return as "the
        mosaic is gone" and frees the cell's budget slot on that basis, so an implementation
        that swallows storage or permission failures lets every cluster's multi-terabyte
        mosaic accumulate off-budget while the fill reports success. A raise is handled — the
        cell has already landed, so it is leaked loudly rather than failed.
        """
        ...

    def discard(self, zone: str, year: int) -> None:
        """Forget a cell's production attempt so ``start`` will run a new one.

        ``start`` is idempotent, which is what lets the feeder call it freely — and which
        also means a FAILED attempt is remembered forever, so a retry would re-observe the
        same failure and spend its attempt budget re-reading one dead result. The retry path
        calls this first. A no-op is valid for an adapter that keeps no state, and discarding
        a cell that was never started must also be a no-op.

        MAY RAISE, and the retry path lets it: an adapter whose previous attempt is still
        running has to end it before a replacement starts and cannot always confirm that it
        did. Raising refuses the retry, costing a recoverable cell; the alternative costs a
        mosaic written by two runs at once, which nothing downstream detects and no retry
        repairs.
        """
        ...

    def cancel_unstarted(self) -> int:
        """Cancel input production that has not begun yet; return how many. Never
        touches work already running.
        """
        ...

    def ready(self, zone: str, year: int) -> bool:
        """True if :meth:`wait` would return immediately. Never blocks.

        Lets the feeder take whichever look-ahead cell has landed instead of stalling on the
        one first in density order — the densest zone is also the slowest to ingest, so
        waiting for it idles the fleet while smaller mosaics sit finished on disk. A
        conservative ``False`` is always safe: the feeder then blocks on its head cell.
        """
        ...


@dataclass
class SequentialCell:
    """One (zone, year) work item, with its preflight-derived tile count.

    ``num_actors`` is the cell's CLAMPED actor request, ``min(fleet, n_tiles)``. The flow sizes
    the shared session from the largest cell's (``live[0].num_actors``); nothing else reads it,
    since every cell — first attempt or retry — runs on that one session.

    ``n_tiles`` is the UNCLAMPED live-tile count, and it is what cells are ordered by.
    Ordering on ``num_actors`` looks equivalent and is not: the clamp collapses every zone
    bigger than the fleet to the same value, losing exactly the range of relative density
    the densest-first ordering exists to sort.
    """

    zone: str
    year: int
    num_actors: int
    n_tiles: int = 0


@dataclass
class PreparedCell:
    """Post-ingest per-cell inputs resolved by the flow's ``prepare`` callable.

    All of these depend on the cell's mosaic existing, which for campaign-managed ingestion
    is only true after ``inputs.wait`` returns: the s1 orbit (hence ``config``) is resolved by
    probing the mosaic, and ``run_id`` fingerprints the mosaic's ingest marker so staging
    resume is keyed to these exact inputs.
    """

    mosaic_base: str
    staging_base: str
    run_id: str
    config: InferenceConfig
    #: What the preflight coverage gate SAW in the mosaics — which months and dates were
    #: actually present, not merely that enough of them were. Measured once, here, and
    #: unrecoverable afterwards, since the mosaics are deleted as soon as the cell lands.
    #: Carried so ``assemble_zone_year`` can persist it in the zone-year's provenance, the
    #: only durable record for a cell filled under ``allow_partial_window``.
    input_coverage: dict | None = None


@dataclass
class _ZoneTally:
    """Per-zone completion scoreboard for the streamed session."""

    cell: SequentialCell
    prep: PreparedCell
    plan: ZonePlan
    remaining: int
    #: Which attempt at this cell the streamed tiles belong to. Carried here because a tally is
    #: the only thing the SCHEDULER thread holds when it discovers that a cell's tiles failed,
    #: and deciding whether to re-queue the cell needs its attempt number.
    attempt: int = 1
    results: list[dict[str, Any]] = field(default_factory=list)
    failed: bool = False


#: How long the feeder sleeps when its queue is empty but a taken cell is still undecided.
#:
#: A LIVENESS BELT, not a poll interval: every add to the queue and every settle notifies the
#: condition, so the feeder normally wakes at once and this timeout expires only if a settle
#: never arrives at all. Seconds rather than minutes so such a lapse costs a short delay rather
#: than a stalled cluster, and long enough that a feeder waiting out a cluster's assembly
#: backlog wakes a few times an hour rather than continuously.
_FEEDER_WAIT_S = 5.0


def fill_zones_sequential(
    *,
    cells: list[SequentialCell],
    prepare: Callable[[SequentialCell], PreparedCell],
    plan: Callable[[SequentialCell, PreparedCell], ZonePlan],
    session: Callable[
        [Callable[[], list[WorkItem] | None], Callable[[WorkItem, dict[str, Any]], None]], list[dict[str, Any]]
    ],
    assemble: Callable[[ZoneFillHandoff, PreparedCell], dict[str, Any]],
    # The pool a landed cell's deletes run on. The CALLER may own it because the staging delete
    # is issued from inside `assemble` (see `assemble_zone_year`'s `defer_cleanup`), which this
    # runner cannot reach — but the runner is what drains, so it is what joins the pool. A
    # caller that passes one must not submit to it after this returns. None means the runner
    # makes its own, which covers the mosaic delete alone.
    housekeeping: ThreadPoolExecutor | None = None,
    session_s1_orbit: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    inputs: CellInputs | None = None,
    look_ahead: int = 2,
    max_retained_failures: int = 100,
    attempts_per_cell_in_cluster: int = 2,
    fault: ArmedFault | None = None,
    paused: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Stream ``cells`` through one shared inference session, assembly trailing.

    A feeder thread walks the cells in order — wait for inputs, ``prepare``
    (orbit/config/run_id), ``plan`` (validation + live tiles), per-zone staged-resume scan —
    and enqueues each zone's tiles for the session's ``more_work`` source. The session
    interleaves zones at queue exhaustion; per-item completion callbacks tally each zone, and
    a completed zone's ``assemble`` (plus mosaic cleanup) runs on the trailing thread. A cell
    that fails in any phase is recorded and the stream continues, then gets
    ``attempts_per_cell_in_cluster - 1`` retries on this still-provisioned cluster before the
    run ends; anything still failing stays pending in the campaign ledger for the driver's
    next pass. No consecutive-failure breaker here: zone outcomes interleave, and the
    scheduler already aborts on systemic actor-death storms.

    Args:
        cells: Ordered (zone, year) work items, largest-first. May span years; a zone's years
            must appear in list order, which the single serialized assembly thread then makes
            commit-safe without any caller guarantee.
        prepare: Resolves a cell's :class:`PreparedCell` once its inputs are ready. Raising
            here fails the cell, not the run.
        plan: Resolves the cell's :class:`~.zone_fill.ZonePlan` (validation, coverage mask,
            live tiles). Terminal plans (already complete / all-ocean) are committed+tagged
            inside and recorded directly.
        session: Runs the shared inference stream — a partial application of
            :func:`tessera_embeddings.inference.runner.run_inference` over
            ``(more_work, on_item_done)``. Blocks until every streamed tile is final.
        assemble: The assembly phase, ``(handoff, prepared) → summary``.
        housekeeping: Pool for a landed cell's staging and mosaic deletes. Optional; the
            runner makes its own when omitted. A caller passes one when it issues deletes
            this runner cannot see — the staging delete comes from inside ``assemble`` — and
            must not submit to it after this function returns, which joins it.
        session_s1_orbit: The shared session's actor-config orbit. A cell whose resolved orbit
            differs is logged and streamed anyway — its orbit rides on its ``ZoneContext`` —
            so this is an observability reference, not a routing decision.
        log: Logger.
        inputs: Mosaic lifecycle adapter; ``None`` means the mosaics already exist upstream
            (no starts, no waits, no cleanup).
        paused: Asked before each hand-over whether inference is paused. While it answers true
            no further cell enters the stream: the chunks already queued run to completion and
            land, the actors stay alive holding nothing, and the session does not finish. Two
            properties elsewhere make an indefinite hold safe, and both are load-bearing —
            idle actors are retired only once the source reports EXHAUSTED (so a paused fleet
            keeps the actors a resume needs, and is billed for them), and a finished chunk is
            removed from the progress tracker (so a drained fleet has no entry whose staleness
            could grow into the systemic-stall abort).

            Cheap and fail-open by contract (see ``pause_signal``): a loop that has to ask
            permission to work must never stop working because the asking failed. ``None``
            disables the check, which is what every path with no gate configured gets.
        attempts_per_cell_in_cluster: Attempts at one cell inside THIS run, counting the first
            — 2 (the default) means one retry. **The cheap retry:** the cell goes to the back of
            the feeder's queue and is served by the session that is already running, its mosaic
            was kept and its staged tiles resume, so it costs the tiles it actually lost.

            Nothing about it provisions hardware, and that is deliberate rather than incidental.
            A retry that ran after the stream had to call ``run_inference`` again, and a fleet's
            lifetime is exactly one such call — so each retried cell created a fleet on a
            cluster whose GPU nodes the autoscaler had already reclaimed, ramped it from zero
            against the launch-rate bound, and killed it at the end of the cell.

            It covers "the work failed but the machine is fine" and nothing else; it cannot
            help when this run itself dies (a killed container, a lost Ray head, a cancelled
            run takes this counter with it), which is what the driver's ``max_dispatch_rounds``
            is for. The two are nested, not alternatives: a deterministic failure burns both,
            and what stops that is the driver's no-progress check rather than either count.
        look_ahead: Sizes INGEST width only — the driver runs ``1 + look_ahead`` cells at a
            time. It bounds neither inference nor assembly, and it is deliberately not tied to
            ``max_retained_failures``, since coupling a failure budget to an ingest width
            meant neither could be tuned alone.
        max_retained_failures: How many failed cells may hold mosaics off-budget before the
            feeder stops admitting. **A ceiling against a systematic fault, not a tripwire for
            a bad hour** — set it near the roster size, because a value low enough for a bad
            hour to reach turns an exogenous failure wave into a fleet-wide teardown.

            Reaching it **alerts and continues**: the run logs ``FAILURE CAP EXCEEDED``, stops
            admitting, and finishes its in-flight work normally. It does NOT tear itself down.
            Ending a fill spends its Ray cluster and every actor on it — hours to rebuild, and
            the most expensive thing the campaign owns — so that is a campaign-manager
            decision, not one a child process takes on its own judgement.
        fault: Supervised-drill hook, consulted where prepared work crosses from the feeder to
            the scheduler. Inert unless the run was armed for the supply-withholding fault
            (:mod:`tessera_embeddings.config.fault_injection`). This is the only point at
            which a fleet can be left genuinely idle without breaking anything: the session's
            liveness, its actors and its retirement policy all key on the source still being
            unexhausted, so withholding here starves the fleet while every other mechanism
            behaves exactly as it does when a cell's ingest is simply slow.

    Returns:
        Summary dict: per-cell outcomes, failure records, deferral count, and timing.

    Raises:
        RuntimeError: After all cells have been attempted, when any cell failed (completed
            cells are already committed + tagged and drop out of the next campaign pass).
    """
    if look_ahead < 0:
        # Sizes the ingest driver's `max_parallel` (1 + look_ahead) in the flow, so a negative
        # value asks for a zero-width pool. The flow validates it earlier too.
        raise ValueError(f"look_ahead must be >= 0, got {look_ahead}")
    t0 = time.monotonic()
    lock = threading.Lock()
    outcomes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    tallies: dict[str, _ZoneTally] = {}  # run_id → tally
    ready: deque[list[WorkItem]] = deque()  # zones awaiting injection, in cell order
    #: The feeder's own queue of cells to admit, each with the attempt number it will be taken
    #: at. A QUEUE rather than the list it used to walk, because three threads can add to it —
    #: the feeder (its own ingest and plan guards), the scheduler thread (a failed tally, via
    #: ``_account_failed_inference``) and the trailing assembly thread (a failed assembly). Every
    #: mutation is under ``lock``.
    work_queue: deque[tuple[SequentialCell, int]] = deque((cell, 1) for cell in cells)
    #: Cells TAKEN from ``work_queue`` whose INFERENCE outcome is not yet decided — mid-plan on
    #: the feeder thread, queued for the stream, or streaming.
    #:
    #: INFERENCE, not the whole cell: a cell handed to the assembly queue is settled at that
    #: moment, because what this count gates is how long the work source stays unexhausted, and
    #: an unexhausted source keeps the GPU fleet standing. Counting a cell until its assembly
    #: returned held the fleet through the entire single-threaded assembly backlog — hours of
    #: work that needs no GPU. An assembly that FAILS is consequently retried after the backlog
    #: drains rather than re-admitted to the stream, which costs nothing: every tile of such a
    #: cell is staged, so a per-cell session returns before it creates an actor.
    #:
    #: THE INVARIANT, and the only thing keeping the feeder's termination honest: every take
    #: from ``work_queue`` is matched by exactly one ``_settle``, so at any instant a cell is in
    #: exactly one of — in ``work_queue``, counted here, or decided. Read only together with
    #: ``work_queue``, under ``lock``. Exactly one site increments it (``_take_next``) and
    #: exactly one decrements it (``_settle``); ``_feed``'s ``finally`` abandons the count
    #: outright, which is the one documented exception.
    #:
    #: The two failure modes it stands between are asymmetric, which is why it exists rather
    #: than a simpler "has the feeder finished its list" flag: finishing EARLY silently drops a
    #: cell that was about to be re-admitted, and finishing LATE would hold a whole GPU cluster.
    undecided = 0
    #: Wakes the feeder when work is added or a cell settles, so an empty queue with cells still
    #: in flight is a wait rather than a spin.
    work_available = threading.Condition(lock)
    feeder_done = threading.Event()
    feeder_error: list[BaseException] = []  # an exception outside the per-cell guards
    stop = threading.Event()  # session crashed — unwind the feeder
    #: Set once ``session`` has RETURNED, however it returned. A cell still undecided at that
    #: point will never be settled — nothing is left to stream it — so the feeder must stop
    #: waiting on the count or it parks until its join times out. Distinct from ``stop``, which
    #: means the session RAISED and the run is unwinding.
    session_over = threading.Event()
    # NOTHING bounds admission to the inference stream except this cap — see the module
    # docstring and the `max_retained_failures` argument above.
    if max_retained_failures < 1:
        # `0 >= max_retained_failures` is true before any cell has failed, so a non-positive
        # cap tears the run down after the ingests have been primed and the Ray cluster
        # started — expensive, and silent about why. Refused here rather than at the flow,
        # because the runner is callable directly.
        msg = f"max_retained_failures must be >= 1, got {max_retained_failures}"
        raise ValueError(msg)
    #: One-shot latch so the cap alert is logged once, not once per later failure. NOTHING
    #: else reads it: a cap trip does not signal teardown and does not abandon queued work.
    #: See ``_retain_failed_mosaic`` for why ending a fill is a campaign-manager decision.
    cap_alerted = False
    retained_failed: set[tuple[str, int]] = set()
    #: Cells that LANDED but whose mosaic delete failed. Tracked apart from
    #: ``retained_failed`` because these must not stop the feeder — see ``_leak_mosaic``.
    cleanup_leaked: set[tuple[str, int]] = set()
    #: Cells submitted to the trailing finalizer and not yet assembled. Reported at the
    #: drain, where the backlog can be most of a cluster's cells and take hours.
    assembly_pending = 0
    finalizer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trailing-assembly")
    #: A landed cell's deletes — its STAGING prefix and its MOSAIC prefix — run HERE, not on
    #: the assembly thread. A mosaic is multi-terabyte and its delete takes as long as it
    #: takes; inside `_finalize` it held the single assembly worker, so the next cell's
    #: assembly could not start even with its tiles fully staged. Measured on 2026-08-31:
    #: seven of nine clusters idle 91-117 minutes after publishing, with 22 fully-staged
    #: cells waiting behind their predecessors' deletes.
    #:
    #: FOUR workers, not one and not unbounded. Deletes are I/O-bound and overlap freely, but
    #: they are also the campaign's heaviest S3 traffic, and an unbounded pool would let a
    #: cluster publishing in a burst aim every delete at the same bucket at once. Four is
    #: comfortably more than the ~one publication per three hours a cluster sustains, so the
    #: queue only builds during a drain, which is when it should.
    mosaic_cleaner = housekeeping or ThreadPoolExecutor(max_workers=4, thread_name_prefix="mosaic-cleanup")

    def _leak_mosaic(cell: SequentialCell) -> None:
        """A LANDED cell whose mosaic delete failed: free the slot, do NOT count it.

        The mosaic stays on disk exactly as a retained failure's does, so the storage concern
        is the same — but the cell succeeded, and `retained_failed` is what stops the feeder
        admitting work. Counting a landed cell there let a run with broken delete permissions
        halt the campaign after ``look_ahead + 2`` cells, every one of which had published
        correctly. Loud and counted separately instead, so an operator sees storage growing
        without the fill refusing to continue.
        """
        if inputs is None:
            return
        with lock:
            cleanup_leaked.add((cell.zone, cell.year))
            n = len(cleanup_leaked)
        log.warning(
            "Mosaic for %s-%d could not be deleted after the cell landed (%d leaked so far). "
            "The cell is published and correct; its mosaics need sweeping.",
            cell.zone,
            cell.year,
            n,
        )

    def _retain_failed_mosaic(cell: SequentialCell) -> None:
        """A failed cell frees its budget slot (no deadlock) but keeps its mosaic (staged
        resume) — counted so a systematic failure cannot accumulate every cluster's mosaic
        off-budget (the feeder stops at ``max_retained_failures``).
        """
        if inputs is None:
            return
        nonlocal cap_alerted
        with lock:
            retained_failed.add((cell.zone, cell.year))
            n = len(retained_failed)
            # Alerted from HERE, not from the feeder's admission check: this is the one place
            # `retained_failed` grows, so the only one that sees every path. The cap-th failure
            # can arrive from an inference or assembly callback after the feeder has drained
            # `pending`, or on the last pending cell, and a check that runs only before the
            # NEXT admission never fires for either.
            newly_alerted = n >= max_retained_failures and not cap_alerted
            if newly_alerted:
                cap_alerted = True
        if newly_alerted:
            # A DISTINCTIVE, GREPPABLE PREFIX, because there is no alerting transport in this
            # repo and monitoring matches on the text — the same convention `DATA LOSS` uses.
            # The run alerts and continues; see `max_retained_failures` for why teardown is a
            # campaign-manager decision.
            log.error(
                "FAILURE CAP EXCEEDED failed=%d/%d — this fill has stopped admitting new cells "
                "and will finish its in-flight work. It will NOT restart itself. A hard restart "
                "is a campaign-manager decision: it re-dispatches the remaining roster at the "
                "cost of this cluster's GPU actors, which take hours to re-gather.",
                n,
                max_retained_failures,
            )
        log.warning(
            "Cell %s-%d failed — mosaic retained for resume, off-budget (%d/%d retained-failure cap)",
            cell.zone,
            cell.year,
            n,
            max_retained_failures,
        )

    def _record_failure(cell: SequentialCell, phase: str, exc: BaseException) -> None:
        with lock:
            failures.append({"zone": cell.zone, "year": cell.year, "phase": phase, "error": str(exc)})
        log.error("Cell %s-%d failed during %s: %s", cell.zone, cell.year, phase, exc, exc_info=exc)

    def _record_outcome(result: dict[str, Any]) -> None:
        with lock:
            outcomes.append(result)

    def _clear_failure(cell: SequentialCell) -> None:
        """Drop a cell's failure record — a retry landed, so the run must not raise on it."""
        with lock:
            failures[:] = [f for f in failures if (f["zone"], f["year"]) != (cell.zone, cell.year)]

    def _start_ingests(pending: list[SequentialCell]) -> None:
        """Start an ingest for EVERY pending cell. Idempotent, so it is safe per feed step.

        Ingest concurrency is the driver's own ``max_parallel`` and nothing else — starting
        only a window here is what chained ingest to assembly. Density order is unaffected:
        the driver works its queue in the order given, which is ``pending`` order.
        """
        if inputs is None:
            return
        for cell in pending:
            inputs.start(cell.zone, cell.year)

    def _finalize(tally: _ZoneTally) -> None:
        """Trailing-thread body: assemble a SUCCESSFUL zone, then delete its mosaic.

        Only reached for a cell whose inference succeeded — ``_submit_assembly`` accounts
        failures itself. A zone that fails to ASSEMBLE keeps its mosaic: the retry re-derives
        its fingerprinted run_id from the mosaic's ingest marker and resumes its staged tiles.
        """
        cell, prep = tally.cell, tally.prep
        try:
            handoff = complete_zone_inference(tally.plan, results=tally.results)
            _record_outcome(assemble(handoff, prep))
        except Exception as exc:
            # FAILED assembly → retain the mosaic for resume and COUNT it, so a systematic
            # failure cannot accumulate every cluster's mosaic unbounded. Pure bookkeeping, so
            # it stays on this thread; there is nothing to delete.
            _record_failure(cell, "assembly", exc)
            _retain_failed_mosaic(cell)
        else:
            # HANDED OFF, not run here. By this point the cell is committed, tagged and
            # recorded as a success, so deleting its mosaics is housekeeping on work that has
            # already landed — and housekeeping must not own the one thread the next cell's
            # assembly needs. See `mosaic_cleaner` for what that cost.
            if inputs is None:
                _leak_mosaic(cell)
            else:
                mosaic_cleaner.submit(_delete_mosaic, cell)
        finally:
            with lock:
                nonlocal assembly_pending
                assembly_pending -= 1

    def _delete_mosaic(cell: SequentialCell) -> None:
        """Delete a landed cell's mosaics, on the cleanup pool rather than the assembly thread.

        A transient S3 error or a missing delete permission must not append a second,
        contradictory `assembly` failure for a cell that succeeded — which is what sharing the
        assembly's try block did, and it also retained the mosaic against the failure cap and
        could stall the feeder. A LANDED cell whose delete failed leaks, uncounted: it is
        correct, and must not consume the cap that exists to stop the feeder.
        """
        assert inputs is not None  # only submitted on the `inputs is not None` branch
        try:
            inputs.cleanup(cell.zone, cell.year)
        except Exception:
            log.exception(
                "Mosaic cleanup failed for %s-%s AFTER the cell landed; the cell stands, "
                "its mosaics are retained and will need sweeping.",
                cell.zone,
                cell.year,
            )
            _leak_mosaic(cell)

    def _account_failed_inference(tally: _ZoneTally) -> None:
        """Record and count a cell whose inference failed. Does NO I/O, by design.

        Must not run on the finalizer thread: it would queue behind the assembly backlog, and
        since nothing bounds admission the feeder would admit the rest of the cluster before
        the first failure was counted — leaving the retained-failure cap blind exactly when a
        systematic failure is what it exists to catch.
        """
        bad = [r for r in tally.results if r.get("status") == "failed"]
        _readmit(
            tally.cell,
            tally.attempt,
            "inference",
            RuntimeError(f"{len(bad)}/{len(tally.results)} tiles failed (e.g. {bad[0]})"),
        )

    def _submit_assembly(tally: _ZoneTally) -> None:
        """Route a completed cell. Failures are accounted NOW; only successes are queued.

        A failed tally needs no assembly, only bookkeeping, so the finalizer stays an ASSEMBLY
        queue and nothing else — which keeps the retained-failure cap prompt however deep the
        backlog is.
        """
        if tally.failed:
            _account_failed_inference(tally)
            return
        nonlocal assembly_pending
        # SETTLED HERE, not when the assembly finishes: what ``undecided`` tracks is whether a
        # cell's INFERENCE outcome is still open, and this cell's is now known. Settling at the
        # end of assembly instead held the work source unexhausted for the whole assembly
        # backlog — so the session, its actors and the GPU cluster stayed up through hours of
        # single-threaded assembly that needs no GPU at all. Measured while building this: the
        # pause test's source never reached exhaustion within 200 polls because it was waiting
        # on `assemble`. An assembly that FAILS is therefore not re-admitted to the stream; it
        # needs no fleet (every tile is staged, so a per-cell session returns before creating
        # an actor) and is retried after the backlog drains.
        _settle(tally.cell)
        with lock:
            assembly_pending += 1
        finalizer.submit(_finalize, tally)

    def _settle(cell: SequentialCell, *, requeue: tuple[SequentialCell, int] | None = None) -> None:
        """One taken cell's inference outcome is now decided — landed, failed, or re-queued.

        THE ONLY site that decrements ``undecided``; ``_take_next`` is the only one that
        increments it. Keeping each to a single site is what makes the invariant above
        checkable by eye rather than by tracing every branch of ``_feed``.

        ``requeue`` re-appends the cell in the SAME lock block as the decrement, which is the
        point of putting it here rather than at the caller: the feeder's exit predicate reads
        the queue and the count together, so a re-admission split across two acquisitions would
        leave an instant in which the cell is in neither and the feeder could finish on it.
        """
        nonlocal undecided
        with lock:
            if requeue is not None:
                work_queue.append(requeue)
            undecided -= 1
            now = undecided
            work_available.notify_all()
        if now < 0:  # pragma: no cover - a take/settle imbalance is a bug in this module
            log.error("Cell accounting went negative at %s-%d — the feeder may finish early", cell.zone, cell.year)

    def _readmit(cell: SequentialCell, attempt: int, phase: str, exc: BaseException) -> None:
        """A cell failed. Send it to the BACK of the work queue, or record it as final.

        THE SINGLE DECISION POINT for a recoverable failure, called from the feeder thread (its
        ingest and plan guards) and from the scheduler thread (a tally whose tiles failed). Both
        outcomes settle the cell exactly once, so the count stays balanced whichever way it goes.

        **The back of the queue, not a pass at the end.** A retry used to run after the whole
        stream had finished, through a per-cell path that called ``run_inference`` again — which
        creates a fleet from nothing and kills it at the end of the cell. On a cluster whose GPU
        nodes the autoscaler has already reclaimed that is minutes of instance launch and model
        load per cell, and it happened once per retried cell. Re-queueing instead means the
        standing fleet serves the retry, because the work source is still live.

        **An ingest failure gets a NEW ingest, immediately and without blocking.** ``start`` is
        idempotent, so a failed production attempt is remembered and a retry that only re-planned
        would re-read the same failure and spend the attempt on nothing; ``discard`` drops it
        first. Neither call waits for the mosaic — the adapter's own pool and the fleet-wide
        ingest gate decide when it actually runs, so the retry queues behind the roster's other
        ingests exactly as a fresh cell would, while inference and assembly carry on. The feeder
        is not held either: ``_take_next`` skips a cell whose mosaic has not landed, so this one
        is only picked up when its ingest is done or when nothing else is left.

        ``discard`` MAY RAISE by contract, and refusing the retry is the right answer when it
        does: it means the previous child could not be confirmed stopped, and two runs writing
        one mosaic prefix is a fault nothing downstream detects and no retry repairs.
        """
        _record_failure(cell, phase, exc)
        _retain_failed_mosaic(cell)  # mosaic retained for resume, counted against the cap
        if attempt >= attempts_per_cell_in_cluster or stop.is_set():
            _settle(cell)
            return
        if phase == "inputs/prepare" and inputs is not None:
            try:
                inputs.discard(cell.zone, cell.year)
                inputs.start(cell.zone, cell.year)
            except Exception:
                log.warning(
                    "Cell %s-%d cannot be re-ingested in-child, so it is not retried here and stays "
                    "pending for the next campaign pass",
                    cell.zone,
                    cell.year,
                    exc_info=True,
                )
                _settle(cell)
                return
        # Dropped BEFORE the re-queue: a cell in flight for a retry must carry at most one
        # failure record, or the closing reconciliation would name it twice and the summary
        # would count one cell as two failures.
        _clear_failure(cell)
        log.warning(
            "Cell %s-%d failed during %s — re-queued for attempt %d/%d on the standing fleet",
            cell.zone,
            cell.year,
            phase,
            attempt + 1,
            attempts_per_cell_in_cluster,
        )
        _settle(cell, requeue=(cell, attempt + 1))

    def _take_next() -> tuple[SequentialCell, int] | None:
        """Take the first QUEUED cell whose mosaic has LANDED, else the head. ``None`` if empty.

        Readiness is probed OUTSIDE ``lock``: ``ready`` is a caller-supplied adapter method, and
        holding the runner's lock across foreign code to save a re-acquire is not a trade worth
        making. The index it picks stays valid because only this thread ever REMOVES from
        ``work_queue`` — the other two threads append to the tail.

        The readiest-first rationale is in the module docstring; measured on the real coverage
        counts, a cluster's opening window spans ~4 h to ~10 h of ingest, so strict density
        order idles the GPUs ~6 h at the start of every year.

        Every pending cell is a candidate, because ``_start_ingests`` has STARTED an ingest for
        all of them; to be TAKEN, a cell's ingest must also have COMPLETED (``ready``), which
        is a ``Future.done()`` check and does no I/O.

        A partial mosaic is never handed to inference under any branch: when nothing has
        landed this returns the head and the caller BLOCKS on it, so an ingest-starved cluster
        behaves as it would in strict order. ``ready`` is also true for a cell whose ingest
        FAILED (the future is done either way), deliberately: the caller's ``wait`` re-raises,
        the cell is recorded as failed, and the cluster continues with its others. Blocking
        forever on a mosaic that will never arrive is the worse outcome.
        """
        nonlocal undecided
        with lock:
            snapshot = list(work_queue)
        if not snapshot:
            return None
        picked = 0
        if inputs is not None:
            for idx, (cell, _) in enumerate(snapshot):
                try:
                    if inputs.ready(cell.zone, cell.year):
                        picked = idx
                        break
                except Exception:  # a broken probe must never stall the feeder
                    log.warning("Readiness probe failed for %s-%d", cell.zone, cell.year, exc_info=True)
        with lock:
            del work_queue[picked]
            undecided += 1
            return snapshot[picked]

    def _feed() -> None:
        """Drain the work queue: inputs → prepare → plan → scan → enqueue, readiest first.

        Ends only when the queue is empty AND no taken cell is still undecided — see
        ``undecided`` for why both halves are needed.
        """
        nonlocal undecided
        try:
            while True:
                with work_available:
                    while not work_queue and undecided > 0 and not stop.is_set() and not session_over.is_set():
                        # Nothing to admit right now, but a cell already taken can still come
                        # back: the scheduler thread re-queues a failed tally and the trailing
                        # thread re-queues a failed assembly. Waiting here rather than finishing
                        # is what stops such a cell being dropped. The timeout is a liveness
                        # belt against a settle that never arrives, not a poll — every add and
                        # every settle notifies.
                        work_available.wait(timeout=_FEEDER_WAIT_S)
                    if not work_queue or session_over.is_set():
                        return
                    pending = [cell for cell, _ in work_queue]
                if stop.is_set():
                    return
                # Stop admitting once too many failed cells are retaining mosaics off-budget:
                # a systematic failure would otherwise keep freeing slots and pile up every
                # cluster's multi-TB input.
                #
                # A cell AWAITING ITS RETRY is not counted, and that exclusion is load-bearing
                # now that a retry comes back through this queue. Its mosaic is retained the
                # same way, but as work in progress rather than as a stuck failure: it will
                # either land (and have its mosaic deleted) or be recorded terminally, at which
                # point it counts. Counting it here instead let a cell's own failure close the
                # gate on its own retry — at a low cap, a deadlock in which the one thing that
                # would clear the cap is the one thing the cap forbids.
                with lock:
                    awaiting_retry = {(cell.zone, cell.year) for cell, _ in work_queue}
                    n_failed = len(retained_failed - awaiting_retry)
                if inputs is not None and n_failed >= max_retained_failures:
                    # NOT the `FAILURE CAP EXCEEDED` prefix: `_retain_failed_mosaic` already
                    # emitted that once for this event, and monitoring matches on the text — a
                    # second line with the same prefix would double-count one cap event.
                    log.error(
                        "Feeder stopping at the failure cap with %d failed cell(s) holding mosaics "
                        "off-budget; %d cell(s) left unattempted, which stay pending for the next "
                        "campaign pass. This run finishes its in-flight work normally.",
                        n_failed,
                        len(pending),
                    )
                    # Recorded as failures, not just logged: the cells that triggered the cap
                    # can RECOVER in the in-child retry pass, and if every one does `failures`
                    # empties and this run reports clean while these cells were never started.
                    # The driver re-reads the store either way, so this is not the only
                    # protection, but a child that under-reports its outcome is not worth
                    # shipping.
                    with lock:
                        failures.extend(
                            {
                                "zone": cell.zone,
                                "year": cell.year,
                                "phase": "unattempted",
                                "error": "never admitted: the feeder stopped at the retained-failure cap",
                            }
                            for cell, _ in work_queue
                        )
                        # EMPTIED, not just abandoned: the queue is half of the exhaustion
                        # predicate the session reads, so leaving cells on it would keep the
                        # stream alive forever waiting for work this feeder has just refused.
                        work_queue.clear()
                        work_available.notify_all()
                    return
                # Start ingests for every pending cell before choosing, so the pick can only
                # ever be a cell whose ingest is already under way.
                _start_ingests(pending)
                if stop.is_set():
                    return
                # Admission is UNBOUNDED: the feeder's only pacing is `inputs.wait` below,
                # which blocks on the chosen cell's ingest. Nothing here waits on an assembly,
                # so the GPU fleet never idles behind one.
                taken = _take_next()
                if taken is None:  # another thread emptied the queue between the peek and here
                    continue
                cell, attempt = taken
                try:
                    if inputs is not None:
                        # stop-aware: the adapter must return promptly (raising) once stop is
                        # set, so a crashed session is never stuck behind a running ingest for
                        # its full duration. A cell `ready()` picked returns immediately.
                        inputs.wait(cell.zone, cell.year, stop=stop)
                    prep = prepare(cell)
                except Exception as exc:
                    if stop.is_set():
                        # Unwinding, not a cell failure — don't record it.
                        return
                    _readmit(cell, attempt, "inputs/prepare", exc)
                    continue
                if prep.config.s1_orbit != session_s1_orbit:
                    # NOT a deferral: the orbit travels on the cell's ZoneContext, so an actor
                    # built for the session's orbit reads this cell under ITS orbit — the same
                    # mechanism that lets one session span campaign years. Deferring instead
                    # would be safe only if a whole zone always carried both orbits, and parts
                    # of the globe are radar-free in principle, so that population could never
                    # complete: every pass would re-ingest and re-fail it. Logged because a
                    # cell read under a different orbit than the session was asked for is
                    # worth seeing, not because anything special happens to it.
                    log.info(
                        "Cell %s-%d resolved s1_orbit=%s != session %s — streaming it under its "
                        "own orbit (carried per cell on the work item)",
                        cell.zone,
                        cell.year,
                        prep.config.s1_orbit,
                        session_s1_orbit,
                    )
                try:
                    zplan = plan(cell, prep)
                    terminal_done = zplan.done is not None
                    if terminal_done:
                        # Terminal (already complete / all-ocean) — committed and tagged
                        # inside plan(); nothing streams, so the staged-resume scan below has
                        # nothing to scan for.
                        assert zplan.done is not None  # narrowed by terminal_done
                        _record_outcome(zplan.done)
                        restored, already = None, set[str]()
                    else:
                        # Per-zone staged-resume scan (the single-zone path does this inside
                        # run_inference; the stream pre-filters here).
                        restored = ZarrWriter(prep.staging_base).scan_existing_staged_artifacts(
                            prep.run_id, zplan.live, compute_std=prep.config.compute_std, log=log
                        )
                        already = restored.done
                except Exception as exc:
                    _readmit(cell, attempt, "plan", exc)
                    continue
                if terminal_done:
                    # OUTSIDE the try, for the same reason the trailing assembly's cleanup is:
                    # plan() has already committed, tagged and recorded this cell, so a failed
                    # mosaic delete must not be caught above and recorded as a `plan` failure
                    # for a cell that succeeded.
                    if inputs is not None:
                        try:
                            inputs.cleanup(cell.zone, cell.year)
                        except Exception:
                            log.exception(
                                "Mosaic cleanup failed for %s-%d after a TERMINAL plan; the cell "
                                "stands, its mosaics are retained and will need sweeping.",
                                cell.zone,
                                cell.year,
                            )
                            _leak_mosaic(cell)
                    # Terminal plans are DECIDED — committed, tagged and recorded inside
                    # `plan()`. Nothing streams for them, so no later thread will settle them.
                    _settle(cell)
                    continue
                assert restored is not None  # only None on the terminal path, which continued
                live = [c for c in zplan.live if c.label not in already]
                # Restore each artifact under the outcome it actually recorded. A skip marker
                # means the tile had no pixels to write; calling that a success makes a
                # resumed zone's tally disagree with the same zone's on a fresh run — an
                # all-skipped resume publishes empty while reporting `skipped: 0`, and a mixed
                # retry quietly inflates the success count.
                resumed = [
                    {
                        "chunk": label,
                        "status": "skipped" if label in restored.skipped else "success",
                        "valid_pixels": 0,
                        "elapsed_sec": 0.0,
                        "resumed": True,
                    }
                    for label in already
                ]
                tally = _ZoneTally(
                    cell=cell, prep=prep, plan=zplan, remaining=len(live), attempt=attempt, results=resumed
                )
                # The cell's OWN window travels with its work items. Actors are built once
                # from the session config, so a cell of a different campaign year would
                # otherwise be inferred over the session's months rather than its own —
                # silently, since the session only checks s1_orbit.
                ctx = ZoneContext(
                    prep.mosaic_base,
                    prep.staging_base,
                    prep.run_id,
                    prep.config.time_window,
                    prep.config.s1_orbit,
                )
                with lock:
                    tallies[prep.run_id] = tally
                    if live:
                        ready.append([WorkItem(chunk=c, ctx=ctx) for c in live])
                log.info(
                    "Zone %s-%d queued for the stream: %d live tile(s), %d resumed",
                    cell.zone,
                    cell.year,
                    len(live),
                    len(resumed),
                )
                if not live:
                    # Everything already staged — straight to assembly.
                    _submit_assembly(tally)
        except BaseException as exc:
            # An exception OUTSIDE the per-cell guards (in _start_ingests, the enqueue, or
            # finalizer.submit) would otherwise just kill this daemon thread: feeder_done
            # fires, _more_work returns None, the session drains the partially-fed queue, and
            # the run returns as if complete with cells silently never enqueued. Capture it so
            # the caller can re-raise after the session drains.
            feeder_error.append(exc)
            log.error("Zone feeder crashed — remaining cells were not enqueued: %s", exc, exc_info=exc)
        finally:
            # RELEASE THE STREAM. The session reads exhaustion from the queue and the undecided
            # count (see `_prepared_zone`), not from this thread being alive — deliberately, so
            # that a feeder stuck inside a probe cannot hold a GPU cluster open. The price is
            # that a feeder which DIES owes the stream an answer: without this, cells left on
            # the queue and cells taken but never settled would keep the source unexhausted
            # forever and the session would never return.
            with lock:
                abandoned = [cell for cell, _ in work_queue]
                work_queue.clear()
                # The one site that touches `undecided` outside the take/settle pair, and it
                # ABANDONS the count rather than balancing it: a crashed feeder cannot know
                # which of the cells it had taken will still be settled by another thread.
                undecided = 0
                work_available.notify_all()
            if abandoned:
                # RECORDED, not merely logged: an abandoned cell that appears in neither
                # `outcomes` nor `failures` makes this run report fewer cells than it was
                # given, and the campaign driver reads that report.
                with lock:
                    failures.extend(
                        {
                            "zone": c.zone,
                            "year": c.year,
                            "phase": "unattempted",
                            "error": "never admitted: the feeder stopped before reaching it",
                        }
                        for c in abandoned
                    )
                log.error(
                    "Zone feeder abandoned %d queued cell(s); they stay pending for the next campaign pass: %s",
                    len(abandoned),
                    ", ".join(f"{c.zone}-{c.year}" for c in abandoned),
                )
            feeder_done.set()

    def _prepared_zone() -> list[WorkItem] | None:
        """One prepared zone, ``[]`` if none is ready YET, ``None`` once none can be.

        Exhaustion is read from the SHARED state — an empty work queue with nothing undecided —
        rather than from ``feeder_done``. Keying it on the feeder thread having exited meant a
        feeder wedged inside a caller-supplied probe held the session, its actors and the whole
        GPU cluster open with no bound; the predicate here is false only while real work is
        outstanding, and a feeder that dies releases it explicitly (see ``_feed``'s ``finally``).
        """
        with lock:
            if ready:
                return ready.popleft()
            if not work_queue and undecided == 0:
                return None
        return []  # nothing ready YET (ingest/plan still running) — keep polling

    def _more_work() -> list[WorkItem] | None:
        """Scheduler-thread source: one prepared zone per poll, None = done."""
        # An operator pause is checked BEFORE the source is consulted, and returns the
        # "nothing ready yet" answer rather than the "exhausted" one. Both halves matter: not
        # consulting keeps the prepared zone on the queue (a hand-over REMOVES it, so asking
        # and discarding would delete prepared work), and `[]` rather than `None` keeps the
        # session alive with its actors — `None` would retire the fleet and finalize the run,
        # a teardown rather than a pause. This is the same site and contract the starvation
        # drill withholds from, which makes "actors stay, nothing fails" tested, not hoped for.
        if paused is not None and paused():
            return []
        # The fault takes the source as a CALLABLE, so a withheld poll never asks for a zone:
        # a hand-over REMOVES the zone from `ready`, so consulting and discarding would delete
        # prepared work instead of delaying it.
        return _prepared_zone() if fault is None else fault.withhold(_prepared_zone, log=log)

    def _on_item_done(item: WorkItem, result: dict[str, Any]) -> None:
        """Scheduler-thread callback: tally the zone; finalize off-thread when full."""
        with lock:
            tally = tallies.get(item.ctx.run_id)
            if tally is None:  # defensive: unknown zone — nothing to account
                log.warning("Result for unknown zone run_id=%s (%s)", item.ctx.run_id, result.get("chunk"))
                return
            tally.results.append(result)
            if result.get("status") == "failed":
                tally.failed = True
            tally.remaining -= 1
            complete = tally.remaining <= 0
        if complete:
            _submit_assembly(tally)

    log.info(
        "Chained fill: %d cell(s) through one session (look_ahead=%d, orbit=%s)",
        len(cells),
        look_ahead,
        session_s1_orbit,
    )
    feeder = threading.Thread(target=_feed, name="zone-feeder", daemon=True)
    feeder.start()
    #: Set when `session` raises, because that path re-raises out of the `finally` below and
    #: so skips BOTH later joins — the feeder-error one and the post-retry one. Without it the
    #: runner could return control with its own executor and multi-terabyte deletes still
    #: running, which is exactly the guarantee its docstring makes.
    session_raised = False
    try:
        session(_more_work, _on_item_done)
    except BaseException:
        # Unwind the feeder (it may be blocked on an ingest wait) before propagating — a hung
        # feeder thread would leak.
        stop.set()
        session_raised = True
        raise
    finally:
        # The feeder waits on cells it has already taken, and nothing will settle them now that
        # the session has returned — so tell it before joining, or it parks for the full join
        # timeout on every run that ends with work in flight.
        with work_available:
            session_over.set()
            work_available.notify_all()
        feeder.join(timeout=600)
        if feeder.is_alive():
            log.warning("Zone feeder did not exit within 600s — continuing teardown (daemon thread)")
        # The backlog here can be most of the cluster's cells, and this drain runs INSIDE the
        # caller's Ray context. Cheap anyway: the session has retired its actors, so GPU
        # workers idle down and only the head node remains. Draining outside the Ray context
        # would need pending assemblies handed back to the flow.
        with lock:
            n_pending = assembly_pending
        if n_pending:
            log.info(
                "Inference complete — draining %d trailing assembly/assemblies, one at a time. "
                "Actors are already retired, so GPU workers idle down while this runs.",
                n_pending,
            )
        finalizer.shutdown(wait=True)
        # `mosaic_cleaner` is DELIBERATELY still alive here. Draining it at this point was a
        # real defect: the in-child retry pass below calls `assemble()` again, and this flow's
        # assemble submits staging cleanup into this pool — so the retry committed and tagged
        # its cell and THEN raised `cannot schedule new futures after shutdown`, which the
        # retry loop catches as a failure, reporting a recovered cell as failed and leaking its
        # staging prefix. It is joined after the retry pass instead, and the caller's own
        # `finally` joins it again on any path that never gets there — EXCEPT the one path
        # that reaches neither: a `session` failure re-raises from here, skipping the retry
        # pass and both later joins. There are no retries to keep it open for, so join now.
        if session_raised:
            # Stop the ingest queue BEFORE the join. This join waits out a staging delete —
            # ~2 h measured — and the exception cannot reach the caller's teardown until it
            # returns, so every queued ingest would keep producing mosaics for that whole
            # window. Best-effort and idempotent: the retry pass calls it too.
            if inputs is not None:
                try:
                    inputs.cancel_unstarted()
                except Exception:
                    log.warning("Could not cancel queued ingests before the cleanup join", exc_info=True)
            mosaic_cleaner.shutdown(wait=True)

    # EVERY EXIT FROM HERE ON JOINS THE POOL, in a `finally` rather than at each exit.
    # Covering exits by hand failed twice: the retry loop catches `except Exception`, so a
    # BaseException from `assemble` or `inputs.wait` (cancellation,
    # KeyboardInterrupt) unwound past all of them, with the session having completed normally
    # so `session_raised` was False too — leaving a runner-owned pool running with
    # multi-terabyte deletes outstanding. A `finally` covers every exit by construction,
    # including any added later. It also sits BEFORE the summary is built, deliberately: every
    # `_leak_mosaic` call must have landed before `cleanup_leaked` is counted.
    try:
        # A feeder crash (captured above) means the session drained only a partial queue and
        # would otherwise look complete — surface it. Committed cells stay tagged; the
        # un-enqueued ones stay pending for the next campaign pass.
        if feeder_error:
            raise RuntimeError(
                "zone feeder crashed before enqueuing all cells — run is incomplete "
                "(unattempted cells remain pending for the next campaign pass)"
            ) from feeder_error[0]

        # THE CLOSING RECONCILIATION. Every cell must be reported as either succeeded or
        # failed, which is what the summary below asserts and what the campaign driver reads.
        # A tally is only accounted when it COMPLETES (`_submit_assembly` fires at
        # `remaining <= 0`), so a session that returned with tiles still queued — a fleet that
        # died out from under the stream, a source held past the end — left its cells in
        # neither list and the run silently reported fewer cells than it was given. This
        # replaces the cover the per-cell retry pass used to provide by re-attempting them.
        #
        # It is a REPORTING fix, not a data one: such a cell never reached `_submit_assembly`,
        # so nothing was assembled or published for it, and even a tally that did complete with
        # a tile missing is refused by `verify_staged_completeness` inside the assembly.
        with lock:
            abandoned_tallies = [t for t in tallies.values() if t.remaining > 0]
        for tally in abandoned_tallies:
            _record_failure(
                tally.cell, "inference", RuntimeError(f"{tally.remaining} chunk(s) never ran before the stream ended")
            )
            _retain_failed_mosaic(tally.cell)

        # ---------------------------------------------------------------------
        # In-child retry of the cells whose ASSEMBLY failed.
        #
        # Every other phase is retried ON THE STREAM now (see `_readmit`), which is the whole
        # point of the queue: a retry runs on the fleet that is already standing instead of
        # rebuilding one. Assembly is the exception, and only because of WHEN its failures are
        # discovered — on the trailing thread, potentially long after the last tile inferred and
        # after the feeder has been joined, so there is no live stream left to re-admit them to.
        #
        # This pass CANNOT provision anything, which is the property that matters. It re-runs
        # `assemble` over the tally the cell already has — no plan, no prepare, no inference —
        # so a fleet cannot be built here however this code is edited later. The previous
        # version went through the per-cell `infer_single` path, which called `run_inference`
        # and so created and destroyed a fleet for every retried cell.
        #
        # SAME-ZONE SAFETY, unchanged: the single assembly thread has been joined by now, so a
        # retry of (Z, y) cannot collide with this child's own (Z, y+1); across children the
        # partition is zone-disjoint. The mosaic-cleanup pool is deliberately still alive, since
        # `assemble` hands its staging delete to it.
        if attempts_per_cell_in_cluster >= 2:
            with lock:
                failed_assembly = {(f["zone"], f["year"]) for f in failures if f["phase"] == "assembly"}
                by_cell = {(t.cell.zone, t.cell.year): t for t in tallies.values() if not t.failed}
            for key in sorted(failed_assembly & by_cell.keys()):
                tally = by_cell[key]
                cell = tally.cell
                try:
                    handoff = complete_zone_inference(tally.plan, results=tally.results)
                    _record_outcome(assemble(handoff, tally.prep))
                except Exception as exc:
                    # Leave the original failure record in place and log the retry's own error,
                    # so the summary still names the cell and the driver still sees it pending.
                    log.error("Cell %s-%d assembly retry failed: %s", cell.zone, cell.year, exc, exc_info=exc)
                    continue
                _clear_failure(cell)
                if inputs is not None:
                    try:
                        inputs.cleanup(cell.zone, cell.year)
                    except Exception:
                        # The cell LANDED — committed, tagged, recorded; only its mosaic prefix
                        # is still there. Letting this propagate would leave the whole child
                        # reporting nothing for every other cell it filled, to punish a failed
                        # delete. `sweep_orphan_mosaics` is the designed remedy for a leaked
                        # prefix, and it needs the prefix named.
                        log.error(
                            "Cell %s-%d recovered but its mosaic was not deleted — it stays until an "
                            "orphan sweep reclaims it",
                            cell.zone,
                            cell.year,
                            exc_info=True,
                        )
                with lock:
                    retained_failed.discard(key)
                log.info("Cell %s-%d recovered on an in-child assembly retry", cell.zone, cell.year)
    finally:
        mosaic_cleaner.shutdown(wait=True)

    elapsed = time.monotonic() - t0
    summary: dict[str, Any] = {
        "cells": len(cells),
        "succeeded": len(outcomes),
        "failed": len(failures),
        "failures": failures,
        "outcomes": outcomes,
        "elapsed_sec": elapsed,
    }
    if failures:
        raise RuntimeError(
            f"{len(failures)}/{len(cells)} cell(s) failed in the chained fill "
            f"(completed cells are committed + tagged and will be skipped on the next campaign pass): {failures}"
        )
    log.info("Chained fill complete: %d/%d cells in %.1f min", len(outcomes), len(cells), elapsed / 60)
    return summary
