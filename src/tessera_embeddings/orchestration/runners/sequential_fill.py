"""Chained multi-zone fill: many (zone, year) cells through ONE Ray session.

The per-cell fill chain (:mod:`.zone_fill`) provisions nothing itself — the
caller owns the Ray context. The parallel campaign exploits that by running
K fills at once, each in its own flow run with its own cluster; this runner
exploits it the other way: **one long-lived cluster whose actors are created
once and stream through every zone**, so the per-cluster costs — ``ray up``,
per-worker EC2 bringup (minutes each), the model-load cold start on every
worker — are paid once per CLUSTER of a campaign year instead of once per zone.
("cluster" throughout means one Ray cluster and the UTM zones assigned to it;
"zone" always means a UTM zone, and "shard" always means a storage shard.)

Keeping the shared fleet busy is the whole game, and three mechanisms
cooperate on it (this docstring is the canonical statement of the rationale —
the flow and README point here):

- **Cross-zone interleaving at exhaustion**: all zones flow through ONE
  work-stealing scheduler session (``run_inference`` with a ``more_work``
  source). When the current zone's queue drops to the live actor count, the
  next prepared zone's tiles top it up — so a zone's tail no longer idles the
  fleet for ~half a tile-duration per actor, and actors are never killed or
  re-created between zones (no per-zone model reload either). Zones stay
  near-sequential: at most one zone's tail overlaps the next zone's head.
- **Readiest-first** (``_take_next``): cells are ordered densest-first, and the
  densest zone is also the slowest to ingest — so taking them strictly in order
  makes the fleet wait on the last mosaic of its opening window while smaller ones
  sit finished. The feeder takes whichever PENDING cell has landed, wherever it
  sits, falling back to the head when none has. It scans the whole pending list
  rather than a look-ahead window, which costs nothing: readiness is a
  ``Future.done()`` check, with no I/O. The density order is unchanged and still
  does its two jobs (sizing the session from the largest cell, putting the island
  tail last); it is simply no longer a barrier.
- **Nothing gates the three stages against each other.** GPUs are by far the most
  expensive resource in the stack, so the ordering principle is that inference
  never waits for anything but its own input:

  * INGEST runs at the ingest driver's own ``max_parallel`` (``1 + look_ahead``,
    which is all ``look_ahead`` still sizes). An ingest is started for every
    pending cell up front, so when one finishes the next begins immediately.
  * INFERENCE is admitted without bound. Its only pacing is ``inputs.wait`` on
    the chosen cell's own mosaic.
  * ASSEMBLY serialises on one trailing thread and may lag arbitrarily far
    behind inference, including past the end of it.

  Two gates used to sit here and BOTH idled the fleet. A mosaic budget admitted
  ingest *starts* against ``look_ahead + 2`` capacity released only after an
  assembly landed, so ingest throughput was set by assembly throughput: a fleet
  configured for 60 concurrent ingests ran 7, measured live. Beside it,
  ``zone_slots`` admitted ``look_ahead + 2`` un-finalized cells on the same
  after-assembly release, which stalled inference once that many cells were
  awaiting assembly and separately capped how many zones' tiles were dispatchable
  at once (the "small-zone fleet fill" limitation, now retired).

  Both were a STORAGE bound originally. They are not any more: the flow ingests
  its whole cluster before requesting GPUs, so a fleet is never billed against an
  unfinished ingest, and peak storage is a cluster's mosaics by design (ADR-011).
  What remains bounded is FAILURE, not throughput: a failed cell keeps its mosaic
  for staged resume and is counted, and the feeder stops admitting once
  ``max_retained_failures`` of them are outstanding.
- **In-child retry**: a failed cell is re-attempted on the still-provisioned
  cluster before the run ends (``attempts_per_cell_in_cluster``), reusing the per-cell
  ``infer_single`` path and the mosaic that was retained for exactly this. Without
  it the driver's retry unit is a whole dispatch, so an early failure would wait
  for every cluster to finish its list. See the block above the fallback pass.
- **Trailing assembly**: a completed zone's shard assembly runs on a background
  thread while later zones' tiles keep the GPUs busy. Assemblies serialize on one
  thread; a zone's mosaic is deleted only after its assembly lands. A zone counts
  as complete only when every tile's result is FINAL — the scheduler fires the
  completion callback after any deferred staging write confirms, so assembly never
  races an in-flight upload.

  This was assumed to be ~10-15% of a cell's inference wall time, which would make
  the backlog self-limiting. Do not rely on that: the first three assemblies of the
  2026-08 global campaign ran at ~1,380 tiles/hour each (1,369 / 1,354 / 1,424 —
  within 5%), which on dense cells is a far larger fraction than assumed. The
  backlog is therefore expected to grow through a cluster's run, and the drain at
  the end is expected to be long. Nothing depends on it being short.

Idle-actor retirement needs no per-zone gating here: the scheduler suppresses
it while the work source is unexhausted and resumes it for the true cluster
tail (see ``scheduling._process_chunks_work_stealing``).

A zone whose mosaic resolves a DIFFERENT s1 orbit than the shared session's
config STILL joins the stream: the orbit travels on each cell's ``ZoneContext``,
so an actor reads every cell under that cell's orbit — the same mechanism that
lets one session span campaign years. It has to work that way because parts of
the globe are radar-free in principle, so a cell resolving ``"none"`` against a
``"both"`` session is a permanent population rather than an anomaly.

There is therefore no post-stream fallback pass any more. ``infer_single`` itself is
still live: the in-child retry runs failed cells on it, on the still-provisioned cluster.

The small-zone fleet-fill limitation is RETIRED. While admission was bounded, that
bound doubled as the fleet-fill parallelism — only that many zones' tiles could be
in flight at once, so a cluster of zones each far smaller than the fleet (an
all-island Pacific cluster, say) left actors idle. Unbounded admission means every
prepared zone's tiles are dispatchable, so the stream fills the fleet from as many
zones as it takes.

The cost of unbounded admission is an assembly BACKLOG: inference can finish a
cluster's cells long before assembly drains them. That is deliberate and it is the
cheap direction to fail — an idle assembly thread costs one CPU, an idle GPU fleet
costs hundreds. See ``ASSEMBLY TAIL`` at the drain site for what it means for the
caller's Ray context.

Contracts: Prefect-free (the deployment-backed ingest adapter, the
input-fingerprinted run_id, the per-cell config/plan, and the session itself
all arrive as callables from the flow layer); the caller is already inside a
Ray context; and cells may span campaign YEARS as well as zones. Two commits for
the same zone group are still never in flight, for a reason that no longer depends
on the caller: assemblies serialize on the single trailing thread, so even a
multi-year list of one zone commits its years one after another. Each cell carries
its OWN inference window on its work items (``ZoneContext.time_window``) — actors are
built once from the session config, so a cell of another year read through that
config would silently be inferred over the wrong months.
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

    The runner drives *when* (start ``look_ahead`` cells early, wait before
    planning, clean up after the cell lands); the implementation owns *how*
    (typically a Prefect ingest deployment per cell). All three methods are
    keyed by ``(zone, year)`` and must be idempotent — ``start`` on an
    already-started cell and ``cleanup`` on a never-ingested cell are no-ops.
    """

    def start(self, zone: str, year: int) -> None:
        """Begin producing the cell's mosaics without blocking."""
        ...

    def wait(self, zone: str, year: int, stop: threading.Event | None = None) -> None:
        """Block until the cell's mosaics are ready; raise if production failed.

        When ``stop`` is supplied the implementation must return promptly
        (by raising) once it is set — the runner passes its unwind event so a
        crashed session is never stuck behind a running ingest.
        """
        ...

    def cleanup(self, zone: str, year: int) -> None:
        """Delete the cell's mosaics (only if this adapter produced them).

        MUST RAISE if the delete did not happen. The runner treats a clean return as
        "the mosaic is gone" and frees the cell's budget slot on that basis, so an
        implementation that swallows storage or permission failures lets every cluster's
        multi-terabyte mosaic accumulate off-budget while the fill reports success.
        A raise is handled — the cell has already landed, so it is leaked loudly rather
        than failed — but only if it reaches the runner.
        """
        ...

    def discard(self, zone: str, year: int) -> None:
        """Forget a cell's production attempt so ``start`` will run a new one.

        ``start`` is idempotent, which is what lets the feeder call it freely — and
        which also means a FAILED attempt is remembered forever. A retry would then
        re-observe the same failure without ever producing the mosaics again, so the
        cell's attempt budget is spent re-reading one dead result. The retry path calls
        this first; a no-op is a valid implementation for an adapter that keeps no
        state, and discarding a cell that was never started must also be a no-op.

        MAY RAISE, and the retry path lets it. An adapter whose previous attempt is still
        running somewhere has to end it before a replacement starts, and it cannot always
        confirm that it did. Raising then refuses the retry, which costs a recoverable
        cell; the alternative costs a mosaic written by two runs at once, which nothing
        downstream detects and no retry repairs.
        """
        ...

    def ready(self, zone: str, year: int) -> bool:
        """True if :meth:`wait` would return immediately. Never blocks.

        Lets the feeder take whichever look-ahead cell has landed instead of
        stalling on the one that happens to be first in density order — the
        densest zone is also the slowest to ingest, so waiting for it idles the
        fleet while smaller mosaics sit finished on disk. A conservative ``False``
        is always safe: the feeder falls back to blocking on its head cell, which
        is the ordering it had before.
        """
        ...


@dataclass
class SequentialCell:
    """One (zone, year) work item, with its preflight-derived tile count.

    ``num_actors`` sizes only the per-cell FALLBACK session (orbit-mismatch
    cells filled after the shared stream); the stream itself is sized by the
    flow's fleet parameter.

    ``n_tiles`` is the UNCLAMPED live-tile count, and it is what cells are ordered
    by. Ordering on ``num_actors`` instead looks equivalent and is not: it is
    ``min(num_actors, n_tiles)``, so every zone bigger than the fleet collapses to
    the same value and their relative density is lost — which is exactly the range
    the densest-first ordering exists to sort.
    """

    zone: str
    year: int
    num_actors: int
    n_tiles: int = 0


@dataclass
class PreparedCell:
    """Post-ingest per-cell inputs resolved by the flow's ``prepare`` callable.

    All three depend on the cell's mosaic existing, which for campaign-managed
    ingestion is only true after ``inputs.wait`` returns: the s1 orbit (hence
    ``config``) is resolved by probing the mosaic, and ``run_id`` fingerprints
    the mosaic's ingest marker so staging resume is keyed to these exact
    inputs.
    """

    mosaic_base: str
    staging_base: str
    run_id: str
    config: InferenceConfig
    #: What the preflight coverage gate SAW in the mosaics — which months and dates were
    #: actually present, not merely that enough of them were. It is measured once, here,
    #: and is unrecoverable afterwards: the mosaics are deleted as soon as the cell lands.
    #: Carried so ``assemble_zone_year`` can persist it in the zone-year's provenance,
    #: which is the only durable record for a cell filled under ``allow_partial_window``.
    input_coverage: dict | None = None


@dataclass
class _ZoneTally:
    """Per-zone completion scoreboard for the streamed session."""

    cell: SequentialCell
    prep: PreparedCell
    plan: ZonePlan
    remaining: int
    results: list[dict[str, Any]] = field(default_factory=list)
    failed: bool = False


def fill_zones_sequential(
    *,
    cells: list[SequentialCell],
    prepare: Callable[[SequentialCell], PreparedCell],
    plan: Callable[[SequentialCell, PreparedCell], ZonePlan],
    session: Callable[
        [Callable[[], list[WorkItem] | None], Callable[[WorkItem, dict[str, Any]], None]], list[dict[str, Any]]
    ],
    assemble: Callable[[ZoneFillHandoff, PreparedCell], dict[str, Any]],
    infer_single: Callable[[SequentialCell, PreparedCell, bool], ZoneFillHandoff],
    session_s1_orbit: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    inputs: CellInputs | None = None,
    look_ahead: int = 2,
    attempts_per_cell_in_cluster: int = 2,
    fault: ArmedFault | None = None,
    paused: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Stream ``cells`` through one shared inference session, assembly trailing.

    A feeder thread walks the cells in order — wait for inputs, ``prepare``
    (orbit/config/run_id), ``plan`` (validation + live tiles), per-zone
    staged-resume scan — and enqueues each zone's tiles for the session's
    ``more_work`` source. The session interleaves zones at queue exhaustion;
    per-item completion callbacks tally each zone, and a completed zone's
    ``assemble`` (plus mosaic cleanup) runs on the trailing thread. A cell
    that fails in any phase is recorded and the stream continues, then gets
    ``attempts_per_cell_in_cluster - 1`` retries on this still-provisioned cluster before
    the run ends; anything still failing stays pending in the campaign ledger
    for the driver's next pass. (There is no consecutive-failure breaker here:
    zone outcomes interleave, and the scheduler already aborts on systemic
    actor-death storms.)

    Args:
        cells: Ordered (zone, year) work items, largest-first. May span years; a
            zone's years must appear in list order, which the single serialized
            assembly thread then makes commit-safe without any caller guarantee.
        prepare: Resolves a cell's :class:`PreparedCell` once its inputs are
            ready. Raising here fails the cell, not the run.
        plan: Resolves the cell's :class:`~.zone_fill.ZonePlan` (validation,
            coverage mask, live tiles). Terminal plans (already complete /
            all-ocean) are committed+tagged inside and recorded directly.
        session: Runs the shared inference stream — a partial application of
            :func:`tessera_embeddings.inference.runner.run_inference` over
            ``(more_work, on_item_done)``. Blocks until every streamed tile
            is final.
        assemble: The assembly phase, ``(handoff, prepared) → summary``.
        infer_single: Per-cell fallback ``(cell, prepared, is_final) →``
            handoff for orbit-mismatch cells, run AFTER the session ends
            (their actor config cannot join the shared stream).
        session_s1_orbit: The shared session's actor-config orbit. A cell whose
            resolved orbit differs is logged and streamed anyway — its orbit rides
            on its ``ZoneContext`` — so this is now an observability reference
            rather than a routing decision.
        log: Logger.
        inputs: Mosaic lifecycle adapter; ``None`` means the mosaics already
            exist upstream (no starts, no waits, no cleanup).
        paused: Asked before each hand-over whether inference is paused. While it answers
            true no further cell enters the stream: the chunks already queued run to
            completion and land, the actors stay alive holding nothing, and the session does
            not finish. Two properties elsewhere are what make an indefinite hold safe, and
            both are load-bearing: idle actors are retired only once the source reports
            EXHAUSTED (so a paused fleet keeps the actors a resume needs, and is billed for
            them), and a finished chunk is removed from the progress tracker (so a drained
            fleet has no entry whose staleness could grow into the systemic-stall abort).

            Cheap and fail-open by contract (see ``pause_signal``) — a loop that has to ask
            permission to work must never stop working because the asking failed.
            ``None`` disables the check entirely, which is what every path that has no gate
            configured gets.
        attempts_per_cell_in_cluster: Attempts at one cell inside THIS run, counting the
            first — 2 (the default) means one retry. **The cheap retry:** the cluster is
            still standing, the cell's mosaic was kept and its staged tiles resume, so it
            costs minutes rather than a fresh zone-year.

            It covers "the work failed but the machine is fine", and nothing else. It
            cannot help when this run itself dies — a killed container, a lost Ray head, a
            cancelled run takes this counter with it — which is what the driver's
            ``max_dispatch_rounds`` is for. The two are nested, not alternatives: a
            deterministic failure will burn both, and what actually stops that is the
            driver's no-progress check rather than either count.
        look_ahead: Sizes INGEST width only — the ingest driver runs
            ``1 + look_ahead`` cells at a time, and every pending cell's ingest is
            started up front so the next begins the moment one finishes. It does not
            bound inference (admission is unbounded) and it does not bound assembly.
            It still sets ``max_retained_failures``, which is a failure cap rather
            than a throughput one.
        fault: Supervised-drill hook, consulted where prepared work crosses from the
            feeder to the scheduler. Inert unless the run was armed for the
            supply-withholding fault (:mod:`tessera_embeddings.config.fault_injection`).
            This is the only point at which a fleet can be left genuinely idle without
            breaking anything: the session's liveness, its actors and its retirement
            policy all key on the source still being unexhausted, so withholding here
            starves the fleet while every other mechanism behaves exactly as it does
            when a cell's ingest is simply slow.

    Returns:
        Summary dict: per-cell outcomes, failure records, deferral count, and
        timing.

    Raises:
        RuntimeError: After all cells have been attempted, when any cell
            failed (completed cells are already committed + tagged and drop
            out of the next campaign pass).
    """
    if look_ahead < 0:
        # Sizes the ingest driver's `max_parallel` (1 + look_ahead) in the flow, so a
        # negative value asks for a zero-width pool. The flow validates it earlier too.
        raise ValueError(f"look_ahead must be >= 0, got {look_ahead}")
    t0 = time.monotonic()
    lock = threading.Lock()
    outcomes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    tallies: dict[str, _ZoneTally] = {}  # run_id → tally
    ready: deque[list[WorkItem]] = deque()  # zones awaiting injection, in cell order
    feeder_done = threading.Event()
    feeder_error: list[BaseException] = []  # an exception outside the per-cell guards
    stop = threading.Event()  # session crashed — unwind the feeder
    # NOTHING bounds admission to the inference stream. GPUs are the most expensive
    # resource in the stack, so inference runs as fast as it can and stops only when it
    # runs out of cells; assembly is allowed to lag arbitrarily behind it, including
    # past the end of inference. The semaphore that used to sit here admitted
    # `look_ahead + 2` un-finalized cells and released the slot only after a cell's
    # ASSEMBLY landed, which idled the fleet two ways: inference stalled once that many
    # cells were awaiting assembly, and the same bound capped how many zones' tiles were
    # dispatchable at once (the "small-zone fleet fill" limitation this retires).
    # A FAILED cell keeps its mosaic (for the retry's staged resume). Retained failures
    # are counted, and once they reach this cap the feeder stops admitting new cells (the
    # run then winds down and fails on the recorded failures) — otherwise a systematic
    # failure would retain every cluster's multi-TB mosaic.
    retained_failed: set[tuple[str, int]] = set()
    #: Cells that LANDED but whose mosaic delete failed. Tracked apart from
    #: ``retained_failed`` because these must not stop the feeder — see ``_leak_mosaic``.
    cleanup_leaked: set[tuple[str, int]] = set()
    #: Cells submitted to the trailing finalizer and not yet assembled. Exists only to
    #: report the size of the assembly backlog at the drain, which can now be most of a
    #: cluster's cells and can take hours.
    assembly_pending: set[tuple[str, int]] = set()
    max_retained_failures = look_ahead + 2
    finalizer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trailing-assembly")

    def _leak_mosaic(cell: SequentialCell) -> None:
        """A LANDED cell whose mosaic delete failed: free the slot, do NOT count it.

        The mosaic stays on disk exactly as a retained failure's does, so the storage
        concern is the same — but the cell succeeded, and `retained_failed` is what stops
        the feeder admitting work. Counting a landed cell there let a run with broken
        delete permissions halt the campaign after ``look_ahead + 2`` cells, every one of
        which had published correctly. Loud and counted separately instead, so an operator
        sees storage growing without the fill refusing to continue.
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
        """A failed cell frees its budget slot (no deadlock) but keeps its mosaic
        (staged resume) — counted so a systematic failure can't accumulate every
        cluster's mosaic off-budget (the feeder stops at ``max_retained_failures``).
        """
        if inputs is None:
            return
        with lock:
            retained_failed.add((cell.zone, cell.year))
            n = len(retained_failed)
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

        **Ingest concurrency is the ingest driver's own ``max_parallel``, and nothing else.**
        This used to start only ``1 + look_ahead`` cells, and because ``pending`` shrinks only
        when the feeder ADMITS a cell — which then needed a budget slot released after the
        cell's assembly lands — every new ingest waited on an assembly six cells back. Ingest
        throughput was therefore set by assembly throughput. Measured live: 7 ingests running
        fleet-wide against a configured 60.

        Density order is unaffected: this only decides WHEN an ingest starts, and the driver
        works its queue in the order given, which is ``pending`` order — densest first.
        """
        if inputs is None:
            return
        for cell in pending:
            inputs.start(cell.zone, cell.year)

    def _finalize(tally: _ZoneTally) -> None:
        """Trailing-thread body: assemble a completed zone, release its slot.

        The mosaic delete (potentially multi-TB) also runs here, overlapping
        later zones' inference. A failed zone keeps its mosaic — the retry
        re-derives its fingerprinted run_id from the mosaic's ingest marker
        and resumes its staged tiles.
        """
        cell, prep = tally.cell, tally.prep
        cleaned = False
        failed_assembly = False
        try:
            if tally.failed:
                bad = [r for r in tally.results if r.get("status") == "failed"]
                _record_failure(
                    cell, "inference", RuntimeError(f"{len(bad)}/{len(tally.results)} tiles failed (e.g. {bad[0]})")
                )
                return
            handoff = complete_zone_inference(tally.plan, results=tally.results)
            _record_outcome(assemble(handoff, prep))
        except Exception as exc:
            failed_assembly = True
            _record_failure(cell, "assembly", exc)
        else:
            # OUTSIDE the assembly try, because by here the cell is committed, tagged
            # and already recorded as a success. Deleting its mosaics is housekeeping
            # on work that has landed, so a transient S3 error or a missing delete
            # permission must not append a second, contradictory `assembly` failure for
            # the same cell — which is what sharing the block above did, and it also
            # retained the mosaic against the failure cap and could stall the feeder.
            # The mosaic is still retained on this path; it is simply not a failure.
            if inputs is not None:
                try:
                    inputs.cleanup(cell.zone, cell.year)
                    cleaned = True
                except Exception:
                    log.exception(
                        "Mosaic cleanup failed for %s-%s AFTER the cell landed; the cell stands, "
                        "its mosaics are retained and will need sweeping.",
                        cell.zone,
                        cell.year,
                    )
        finally:
            # FAILED cell → retain the mosaic for resume and COUNT it, so a systematic
            # failure cannot accumulate every cluster's mosaic unbounded. LANDED cell whose
            # delete failed → leak, uncounted: the cell is correct and must not consume the
            # cap that exists to stop the feeder.
            if not cleaned:
                if tally.failed or failed_assembly:
                    _retain_failed_mosaic(cell)
                else:
                    _leak_mosaic(cell)
            with lock:
                assembly_pending.discard((cell.zone, cell.year))

    def _submit_assembly(tally: _ZoneTally) -> None:
        """Queue a completed cell's assembly. Registered BEFORE submitting, so the
        backlog reported at the drain counts cells still sitting in the queue.
        """
        with lock:
            assembly_pending.add((tally.cell.zone, tally.cell.year))
        finalizer.submit(_finalize, tally)

    def _take_next(pending: list[SequentialCell]) -> SequentialCell:
        """Pop the first PENDING cell whose mosaic has LANDED, else the head.

        Cells are ordered densest-first, and the densest zone is also the slowest
        to ingest — so taking them strictly in order makes the fleet wait on the
        very last mosaic of its opening window while smaller ones sit finished.
        Measured on the real coverage counts: a cluster's window spans ~4 h to
        ~10 h of ingest, so the strict order idles the GPUs for ~6 h at the start
        of every year.

        Every pending cell is a candidate, because an ingest has been STARTED for
        every one of them (``_start_ingests``). To be TAKEN, a cell's ingest must
        additionally have COMPLETED (``ready``). Scanning the whole list rather than
        a window is free: ``ready`` is a ``Future.done()`` check and does no I/O.

        A partial mosaic is never handed to inference under any branch: when nothing
        has landed this returns the head and the caller BLOCKS on it, so an
        ingest-starved cluster behaves exactly as it did before.

        ``ready`` is also true for a cell whose ingest FAILED (the future is done
        either way). That is deliberate — the caller's ``wait`` re-raises, the cell
        is recorded as failed, and the cluster continues with its others. Blocking
        forever on a mosaic that will never arrive is the worse outcome.

        The DENSITY ORDER ITSELF IS UNCHANGED: it still sizes the session (the
        fleet is provisioned for the largest cell up front) and still puts the
        island tail last. What changes is that it is no longer a barrier.
        """
        if inputs is not None:
            for idx, cell in enumerate(pending):
                try:
                    if inputs.ready(cell.zone, cell.year):
                        return pending.pop(idx)
                except Exception:  # a broken probe must never stall the feeder
                    log.warning("Readiness probe failed for %s-%d", cell.zone, cell.year, exc_info=True)
        return pending.pop(0)

    def _feed() -> None:
        """Drain cells: inputs → prepare → plan → scan → enqueue, readiest first."""
        try:
            pending = list(cells)
            while pending:
                # Stop admitting once too many failed cells are retaining mosaics
                # off-budget: a systematic failure would otherwise keep freeing
                # slots and pile up every cluster's multi-TB input. The in-flight
                # cells finish and the run fails on the recorded failures; the
                # unattempted cells stay pending for the next campaign pass.
                with lock:
                    n_failed = len(retained_failed)
                if inputs is not None and n_failed >= max_retained_failures:
                    log.error(
                        "Retained-failure cap reached (%d failed cell(s) holding mosaics off-budget) — "
                        "stopping the feeder before admitting more ingests; %d cell(s) left unattempted "
                        "(they stay pending for the next campaign pass). Likely a systematic failure — investigate.",
                        n_failed,
                        len(pending),
                    )
                    # Record them as failures, not just in the log. The cells that
                    # triggered this cap can RECOVER in the in-child retry pass, and if
                    # every one does, `failures` empties and this run reports clean —
                    # while these cells were never started. The driver re-reads the
                    # store either way, so this is not the only protection, but a child
                    # that under-reports its own outcome is worth not shipping.
                    with lock:
                        failures.extend(
                            {
                                "zone": cell.zone,
                                "year": cell.year,
                                "phase": "unattempted",
                                "error": "never admitted: the feeder stopped at the retained-failure cap",
                            }
                            for cell in pending
                        )
                    return
                # Start ingests for every pending cell before choosing, so the pick can
                # only ever be a cell whose ingest is already under way.
                _start_ingests(pending)
                if stop.is_set():
                    return
                # Admission is UNBOUNDED: the feeder's only pacing is `inputs.wait`
                # below, which blocks on the chosen cell's ingest. Nothing here waits on
                # an assembly, so the GPU fleet never idles behind one.
                cell = _take_next(pending)
                try:
                    if inputs is not None:
                        # stop-aware: the adapter must return promptly (raising)
                        # once stop is set, so a crashed session is never stuck
                        # behind a running ingest for its full duration. A cell
                        # `ready()` picked returns from here immediately.
                        inputs.wait(cell.zone, cell.year, stop=stop)
                    prep = prepare(cell)
                except Exception as exc:
                    if stop.is_set():
                        # Unwinding, not a cell failure — don't record it.
                        return
                    _record_failure(cell, "inputs/prepare", exc)
                    _retain_failed_mosaic(cell)  # mosaic (if any) retained for resume, counted
                    continue
                if prep.config.s1_orbit != session_s1_orbit:
                    # NOT a deferral any more. The orbit travels on the cell's ZoneContext, so
                    # an actor built for the session's orbit reads this cell under ITS orbit —
                    # the same mechanism that already lets one session span campaign years.
                    #
                    # It used to be deferred and its mosaic retained against a bounded budget,
                    # and past that bound the cell FAILED and its mosaic was deleted. That was
                    # safe only while a whole zone always carried both orbits. It does not:
                    # parts of the globe are radar-free in principle, so that population could
                    # never complete — every pass re-ingested and re-failed it. Logged because
                    # a cell read under a different orbit than the session was asked for is
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
                        # Terminal (already complete / all-ocean) — committed
                        # and tagged inside plan(); nothing streams, so the
                        # staged-resume scan below has nothing to scan for.
                        assert zplan.done is not None  # narrowed by terminal_done
                        _record_outcome(zplan.done)
                        restored, already = None, set[str]()
                    else:
                        # Per-zone staged-resume scan (the single-zone path does
                        # this inside run_inference; the stream pre-filters here).
                        restored = ZarrWriter(prep.staging_base).scan_existing_staged_artifacts(
                            prep.run_id, zplan.live, compute_std=prep.config.compute_std, log=log
                        )
                        already = restored.done
                except Exception as exc:
                    _record_failure(cell, "plan", exc)
                    _retain_failed_mosaic(cell)  # mosaic retained for resume, counted
                    continue
                if terminal_done:
                    # OUTSIDE the try, for the same reason the trailing assembly's cleanup
                    # is: plan() has already committed, tagged and recorded this cell, so a
                    # failed mosaic delete must not be caught above and recorded as a `plan`
                    # failure for a cell that succeeded. This was the sibling path missed
                    # when _finalize's copy of the bug was fixed.
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
                    continue
                assert restored is not None  # only None on the terminal path, which continued
                live = [c for c in zplan.live if c.label not in already]
                # Restore each artifact under the outcome it actually recorded. A skip
                # marker means the tile had no pixels to write; calling that a success
                # makes a resumed zone's tally disagree with the same zone's tally on a
                # fresh run — an all-skipped resume publishes empty while reporting
                # `skipped: 0`, and a mixed retry quietly inflates the success count.
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
                tally = _ZoneTally(cell=cell, prep=prep, plan=zplan, remaining=len(live), results=resumed)
                # The cell's OWN window travels with its work items. Actors are built
                # once from the session config, so a cell of a different campaign year
                # would otherwise be inferred over the session's months rather than its
                # own — silently, since the session only checks s1_orbit.
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
            # An exception OUTSIDE the per-cell guards (e.g. in _start_ingests,
            # the enqueue, or finalizer.submit) would otherwise just kill this
            # daemon thread: feeder_done fires, _more_work returns None, the
            # session drains the partially-fed queue, and the run returns as if
            # complete with cells silently never enqueued. Capture it so the
            # caller can re-raise after the session drains.
            feeder_error.append(exc)
            log.error("Zone feeder crashed — remaining cells were not enqueued: %s", exc, exc_info=exc)
        finally:
            feeder_done.set()

    def _prepared_zone() -> list[WorkItem] | None:
        """One prepared zone, ``[]`` if none is ready YET, ``None`` once none can be."""
        with lock:
            if ready:
                return ready.popleft()
        if feeder_done.is_set():
            with lock:
                return ready.popleft() if ready else None
        return []  # nothing ready YET (ingest/plan still running) — keep polling

    def _more_work() -> list[WorkItem] | None:
        """Scheduler-thread source: one prepared zone per poll, None = done."""
        # An operator pause is checked BEFORE the source is consulted, and returns the
        # "nothing ready yet" answer rather than the "exhausted" one. Both halves matter:
        # not consulting keeps the prepared zone on the queue (a hand-over REMOVES it, so
        # asking and discarding would delete prepared work), and `[]` rather than `None`
        # keeps the session alive with its actors — `None` would retire the fleet and
        # finalize the run, which is a teardown, not a pause. This is the same site and the
        # same contract the starvation drill withholds from, which is what makes the
        # "actors stay, nothing fails" property tested rather than hoped for.
        if paused is not None and paused():
            return []
        # The fault takes the source as a CALLABLE, so a withheld poll never asks for a
        # zone — a hand-over REMOVES the zone from `ready`, so consulting and discarding
        # would delete prepared work instead of delaying it.
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
    try:
        session(_more_work, _on_item_done)
    except BaseException:
        # Unwind the feeder (it may be blocked on an ingest wait) before
        # propagating — a hung feeder thread would leak.
        stop.set()
        raise
    finally:
        feeder.join(timeout=600)
        if feeder.is_alive():
            log.warning("Zone feeder did not exit within 600s — continuing teardown (daemon thread)")
        # ASSEMBLY TAIL. Inference is admitted without bound, so by here the backlog can
        # be most of the cluster's cells, and they assemble one at a time on this thread.
        # That is the intended trade: assembly is allowed to lag arbitrarily so the GPU
        # fleet never waits on it.
        #
        # The tail is NOT free, because the caller owns the Ray context and this drain runs
        # inside it. What makes it cheap rather than ruinous is that the session has already
        # retired its actors, so the cluster's GPU workers go idle and the autoscaler
        # terminates them on `idle_timeout_minutes`; the head node is what stays up for the
        # remainder. Moving this drain outside the Ray context would remove even that, at
        # the cost of handing pending assemblies back to the flow — a change to this
        # runner's contract, deliberately not made here.
        with lock:
            n_pending = len(assembly_pending)
        if n_pending:
            log.info(
                "Inference complete — draining %d trailing assembly/assemblies, one at a time. "
                "Actors are already retired, so GPU workers idle down while this runs.",
                n_pending,
            )
        finalizer.shutdown(wait=True)

    # A feeder crash (captured above) means the session drained only a partial
    # queue and would otherwise look complete — surface it. Committed cells stay
    # tagged; the un-enqueued ones stay pending for the next campaign pass.
    if feeder_error:
        raise RuntimeError(
            "zone feeder crashed before enqueuing all cells — run is incomplete "
            "(unattempted cells remain pending for the next campaign pass)"
        ) from feeder_error[0]

    # ---------------------------------------------------------------------
    # In-child retry of the stream's failed cells.
    #
    # WHY HERE rather than leaving it to the campaign driver: the driver's retry
    # unit is a whole dispatch, so a cell that fails early would otherwise wait
    # for every cluster to finish its entire list before being re-attempted —
    # hours to days. The cluster is still provisioned right now, the failed
    # cell's mosaic was deliberately RETAINED for exactly this, and its staged
    # tiles resume, so a retry here is usually minutes.
    #
    # HOW: reuse `infer_single` — the same per-cell path the orbit-mismatch
    # fallback uses. It composes plan + run_inference, so it re-validates and
    # returns `done` for a cell that turns out to be complete, and it does its
    # own staged-resume internally. Nothing here re-implements the feeder.
    #
    # BEFORE the fallback pass, deliberately: the fallback marks its last cell
    # `final`, which retires idle actors. Running retries first means they never
    # need actors that a fallback cell has already released. Fallback failures
    # are therefore NOT retried in-child — they are orbit-mismatch cells, rare,
    # and the driver's next round covers them.
    #
    # SAME-ZONE SAFETY: a retry of (Z, y) cannot collide with this child's own
    # (Z, y+1) — assemblies serialise on the single finalizer thread and that
    # thread has been joined by now. Across children it cannot collide either,
    # because the partition is zone-disjoint.
    #
    # EVERY failing path retains its cell's mosaic, so every failed cell is retryable here.
    # `retained_failed` is that set; with no mosaic budget at all the inputs are upstream and
    # permanent, so everything is eligible.
    for attempt in range(2, attempts_per_cell_in_cluster + 1):
        with lock:
            # Cells the feeder never admitted are recorded as failures so the run
            # REPORTS them, but they are not retry candidates: the feeder stopped
            # because too many failures were holding mosaics off-budget, and admitting
            # more work here is the one thing that cap exists to prevent. They stay
            # pending for the driver's next pass, which is what their record says.
            failed_keys = [(f["zone"], f["year"]) for f in failures if f["phase"] != "unattempted"]
            eligible = set(retained_failed)
            # Cells whose INPUTS failed need their production re-run; every other
            # phase failed with a usable mosaic already on disk. See the discard
            # below for why the distinction matters.
            reingest = {(f["zone"], f["year"]) for f in failures if f["phase"] == "inputs/prepare"}
        pending = failed_keys if inputs is None else [k for k in failed_keys if k in eligible]
        if skipped := [k for k in failed_keys if k not in pending]:
            log.warning(
                "Not retrying %d cell(s) in-child — their mosaic slots were released, so a retry would "
                "run against nothing (they stay pending for the driver's next pass): %s",
                len(skipped),
                ", ".join(f"{z}-{y}" for z, y in skipped),
            )
        if not pending:
            break
        by_key = {(c.zone, c.year): c for c in cells}
        log.warning(
            "In-child retry attempt %d/%d over %d failed cell(s): %s",
            attempt,
            attempts_per_cell_in_cluster,
            len(pending),
            ", ".join(f"{z}-{y}" for z, y in pending),
        )
        for zone_name, year in pending:
            cell = by_key.get((zone_name, year))
            if cell is None:  # pragma: no cover - a failure record always names a cell
                continue
            try:
                # ONLY for a cell whose inputs failed: drop the dead production attempt
                # and run a new one. `start` is idempotent, so the failed future stays
                # cached and the retry would otherwise re-read the same failure without
                # ever producing the mosaics — spending the attempt budget on nothing.
                #
                # Not for the other phases. A cell that failed in plan, inference or
                # assembly RETAINED its mosaic precisely so the retry could use it, so
                # re-ingesting would redo work that is already on disk and re-admit
                # budget the retention was designed to keep out.
                if inputs is not None and (zone_name, year) in reingest:
                    inputs.discard(zone_name, year)
                    inputs.start(zone_name, year)
                    inputs.wait(zone_name, year, stop=stop)
                prep = prepare(cell)
                handoff = infer_single(cell, prep, False)
                _record_outcome(assemble(handoff, prep))
            except Exception as exc:
                # Leave the original failure record in place and log the retry's own
                # error, so the summary still names the cell and the driver still
                # sees it as pending.
                log.error("Cell %s-%d retry attempt %d failed: %s", zone_name, year, attempt, exc, exc_info=exc)
                continue
            _clear_failure(cell)
            if inputs is not None:
                try:
                    inputs.cleanup(zone_name, year)
                except Exception:
                    # The cell LANDED — committed, tagged, recorded. Only its mosaic
                    # prefix is still there. Letting this propagate would leave the whole
                    # child reporting nothing for every other cell it filled, to punish a
                    # failed delete; the orphan sweep (`sweep_orphan_mosaics`) is the
                    # designed remedy for a leaked prefix, and it needs the prefix named.
                    log.error(
                        "Cell %s-%d recovered but its mosaic was not deleted — it stays until an "
                        "orphan sweep reclaims it",
                        zone_name,
                        year,
                        exc_info=True,
                    )
            with lock:
                retained_failed.discard((zone_name, year))
            log.info("Cell %s-%d recovered on in-child retry attempt %d", zone_name, year, attempt)

    # No post-stream fallback pass. There was one, for cells whose resolved orbit could not
    # join the shared session; the orbit now travels on each cell's ZoneContext so every cell
    # streams, and the pass had nothing left to receive. ``infer_single`` is still LIVE — the
    # in-child retry above runs on it — so only the pass is gone, not the hook.
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
