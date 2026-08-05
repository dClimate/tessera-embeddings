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
- **Readiest-first within the window** (``_take_next``): cells are ordered
  densest-first, and the densest zone is also the slowest to ingest — so taking
  them strictly in order makes the fleet wait on the last mosaic of its opening
  window while smaller ones sit finished. The feeder takes whichever look-ahead
  cell has LANDED, falling back to its head when none has. The density order is
  unchanged and still does its two jobs (sizing the session from the largest
  cell, putting the island tail last); it is simply no longer a barrier.
- **Ingest look-ahead** (``inputs``): the next cells' mosaics are ingested
  *while earlier cells infer*. Two gates cooperate: ``zone_slots`` admits at
  most ``look_ahead + 2`` *un-finalized* zones (a zone holds its slot from
  prepare until its assembly lands), and a mosaic budget
  (:class:`_MosaicBudget`, same capacity) admits every ingest *start*.
  Together they pace how far the feeder runs ahead of finalization.

  These were a STORAGE bound while the caller primed only a look-ahead window.
  They are not any more: the flow ingests its whole cluster before requesting
  GPUs, so a fleet is never billed against an unfinished ingest, and peak
  storage is a cluster's mosaics by design (ADR-011). The gates still matter for
  pacing and for the fleet-fill limitation noted below. Mosaics RETAINED —
  failed cells (kept for staged resume) and orbit-mismatch deferrals awaiting
  the fallback — are handled explicitly: failures release their budget slot
  with a warning (storage honesty over a feeder deadlock), and deferrals may
  hold at most ``look_ahead + 1`` slots before further mismatch cells fail
  loudly instead of silently stacking multi-TB mosaics.
- **In-child retry**: a failed cell is re-attempted on the still-provisioned
  cluster before the run ends (``max_cell_attempts``), reusing the per-cell
  ``infer_single`` path and the mosaic that was retained for exactly this. Without
  it the driver's retry unit is a whole dispatch, so an early failure would wait
  for every cluster to finish its list. See the block above the fallback pass.
- **Trailing assembly**: a completed zone's shard assembly (~10-15% of its
  inference wall time) runs on a background thread while later zones' tiles
  keep the GPUs busy. Assemblies serialize on one thread; a zone's mosaic is
  deleted only after its assembly lands. A zone counts as complete only when
  every tile's result is FINAL — the scheduler fires the completion callback
  after any deferred staging write confirms, so assembly never races an
  in-flight upload.

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

KNOWN LIMITATION — small-zone fleet fill. The ``look_ahead + 2`` admission
bound doubles as the fleet-fill parallelism: only that many zones' tiles can
be in flight at once. When a zone is at least as large as the fleet this is
irrelevant (one zone fills every actor), but a cluster of zones each far
smaller than the fleet (e.g. an all-island Pacific cluster) can leave actors
idle — at most ``(look_ahead + 2) * tiles_per_zone`` tiles are ever
dispatchable. Largest-first zone ordering pushes these to the low-cost tail,
so the wasted GPU time is bounded; a cluster known to be all-small should be
run with a larger ``look_ahead`` (its mosaics are small, so the storage bound
above is slack). Decoupling admission (a storage-bytes budget) from
fleet-fill parallelism is a possible future refinement — measure first.

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
        """Delete the cell's mosaics (only if this adapter produced them)."""
        ...

    def discard(self, zone: str, year: int) -> None:
        """Forget a cell's production attempt so ``start`` will run a new one.

        ``start`` is idempotent, which is what lets the feeder call it freely — and
        which also means a FAILED attempt is remembered forever. A retry would then
        re-observe the same failure without ever producing the mosaics again, so the
        cell's attempt budget is spent re-reading one dead result. The retry path calls
        this first; a no-op is a valid implementation for an adapter that keeps no
        state, and discarding a cell that was never started must also be a no-op.
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


class _MosaicBudget:
    """Paces mosaic admission: ingest *start* through cleanup.

    ``zone_slots`` alone bounds only *admitted* zones — look-ahead ingest
    starts happen before admission, so without this gate they escape the
    pacing. A slot is acquired (idempotently, keyed by cell) before a
    cell's ingest is started and released at its mosaic cleanup — or, for
    mosaics deliberately retained past the run (failed cells kept for staged
    resume), released explicitly with a warning so a permanently-held slot
    can never starve the feeder into deadlock.

    NOT a storage bound any more. It was one while the caller primed only a
    look-ahead window, but the flow now ingests its ENTIRE cluster before asking
    for GPUs, so every mosaic is already on storage before this gate sees it —
    ``acquire`` finds each ingest already started and returns immediately. What
    it still does is pace how far the feeder runs ahead of finalization, which
    is what keeps `zone_slots` and the trailing assembly honest. Peak storage is
    now the cluster, deliberately: see ADR-011 and
    :mod:`...prefect.flows.fill_zones_sequential`.
    """

    def __init__(self, slots: int) -> None:
        self._sem = threading.Semaphore(slots)
        self._held: set[tuple[str, int]] = set()
        self._lock = threading.Lock()

    def acquire(self, key: tuple[str, int], stop: threading.Event, *, blocking: bool = True) -> bool:
        """Admit ``key``'s mosaic; False = no slot (non-blocking) or ``stop`` set.

        ``blocking=False`` is for LOOK-AHEAD admissions: the feeder must only
        ever block on the CURRENT cell's slot (whose processing is what frees
        slots) — blocking on a future cell's slot while the current one sits
        unprocessed is a deadlock when retained mosaics hold the rest
        of the budget. A denied look-ahead is retried on a later feed step.
        """
        with self._lock:
            if key in self._held:
                return True  # idempotent: look-ahead windows overlap
        if blocking:
            while not self._sem.acquire(timeout=1.0):
                if stop.is_set():
                    return False
        elif not self._sem.acquire(blocking=False):
            return False
        with self._lock:
            self._held.add(key)
        return True

    def release(self, key: tuple[str, int]) -> None:
        """Release ``key``'s slot (idempotent no-op for keys never admitted)."""
        with self._lock:
            if key not in self._held:
                return
            self._held.discard(key)
        self._sem.release()


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
    max_cell_attempts: int = 2,
) -> dict[str, Any]:
    """Stream ``cells`` through one shared inference session, assembly trailing.

    A feeder thread walks the cells in order — wait for inputs, ``prepare``
    (orbit/config/run_id), ``plan`` (validation + live tiles), per-zone
    staged-resume scan — and enqueues each zone's tiles for the session's
    ``more_work`` source. The session interleaves zones at queue exhaustion;
    per-item completion callbacks tally each zone, and a completed zone's
    ``assemble`` (plus mosaic cleanup) runs on the trailing thread. A cell
    that fails in any phase is recorded and the stream continues, then gets
    ``max_cell_attempts - 1`` retries on this still-provisioned cluster before
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
        max_cell_attempts: Attempts per cell WITHIN this child, including the
            first. 2 (the default) means one retry. This is deliberately a
            different bound from the driver's ``max_zone_attempts``, which counts
            whole-dispatch rounds: the two failure modes are different, and one
            knob covering both would silently change what either means. The retry
            reuses ``infer_single`` on the still-provisioned cluster and resumes
            the failed cell's retained mosaic and staged tiles, so it is normally
            minutes rather than a fresh zone-year.
        look_ahead: Cells beyond the feed head to keep in ingest flight; also
            sizes the un-finalized-zone bound AND the mosaic storage budget
            (both ``look_ahead + 2`` — see :class:`_MosaicBudget`).

    Returns:
        Summary dict: per-cell outcomes, failure records, deferral count, and
        timing.

    Raises:
        RuntimeError: After all cells have been attempted, when any cell
            failed (completed cells are already committed + tagged and drop
            out of the next campaign pass).
    """
    if look_ahead < 0:
        # Would size zone_slots/_MosaicBudget at <= 0 capacity → the feeder
        # blocks forever on acquire. The flow validates this earlier too.
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
    # Bounds un-finalized zones (mosaic held from prepare until assembly lands):
    # the current zone + one trailing assembly + the ingest look-ahead.
    zone_slots = threading.Semaphore(look_ahead + 2)
    # Bounds mosaics alive on storage (ingest START through cleanup) at the same
    # capacity — look-ahead starts are admitted through it, so they can no longer
    # escape the storage bound (see _MosaicBudget). No inputs → no mosaics.
    budget = _MosaicBudget(look_ahead + 2) if inputs is not None else None
    # A FAILED cell keeps its mosaic (for the retry's staged resume) but frees
    # its budget slot — otherwise a systematic failure deadlocks the feeder (no
    # cell ever succeeds to free a slot). But freeing every failed slot lets a
    # systematic failure ADMIT and retain every cluster's multi-TB mosaic,
    # bypassing the budget the other way. So retained failures are counted, and
    # once they reach this cap the feeder stops admitting new cells (the run
    # then winds down and fails on the recorded failures). Bounded either way.
    retained_failed: set[tuple[str, int]] = set()
    max_retained_failures = look_ahead + 2
    finalizer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trailing-assembly")

    def _release_mosaic(cell: SequentialCell) -> None:
        if budget is not None:
            budget.release((cell.zone, cell.year))

    def _retain_failed_mosaic(cell: SequentialCell) -> None:
        """A failed cell frees its budget slot (no deadlock) but keeps its mosaic
        (staged resume) — counted so a systematic failure can't accumulate every
        cluster's mosaic off-budget (the feeder stops at ``max_retained_failures``).
        """
        if budget is None:
            return
        with lock:
            retained_failed.add((cell.zone, cell.year))
            n = len(retained_failed)
        budget.release((cell.zone, cell.year))
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

    def _start_lookahead(pending: list[SequentialCell]) -> None:
        """Keep ingests running ahead of the feed head (idempotent starts).

        Every start is admitted through the mosaic budget first, so look-ahead
        can never materialize more mosaics than the bound. Only the CURRENT
        cell's admission may block (its processing is what frees slots —
        blocking on a future cell's slot while the current one sits unprocessed
        deadlocks once retained mosaics hold the rest of the budget);
        look-ahead cells are admitted non-blocking and simply retried on later
        feed steps when the budget is tight. That degradation IS the intended
        backpressure: ingest paced by fill throughput (ADR-011).
        """
        if inputs is None or budget is None:
            return
        for offset, cell in enumerate(pending[: 1 + look_ahead]):
            if not budget.acquire((cell.zone, cell.year), stop, blocking=offset == 0):
                return  # stop set (current) or budget tight (look-ahead) — retry next step
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
        try:
            if tally.failed:
                bad = [r for r in tally.results if r.get("status") == "failed"]
                _record_failure(
                    cell, "inference", RuntimeError(f"{len(bad)}/{len(tally.results)} tiles failed (e.g. {bad[0]})")
                )
                return
            handoff = complete_zone_inference(tally.plan, results=tally.results)
            _record_outcome(assemble(handoff, prep))
            if inputs is not None:
                inputs.cleanup(cell.zone, cell.year)
                cleaned = True
        except Exception as exc:
            _record_failure(cell, "assembly", exc)
        finally:
            # Success → free the slot (clean); failure → retain the mosaic for
            # resume but free + COUNT the slot (bounded by the retained-failure
            # cap the feeder checks). Both free the slot exactly once.
            if cleaned:
                _release_mosaic(cell)
            else:
                _retain_failed_mosaic(cell)
            zone_slots.release()

    def _take_next(pending: list[SequentialCell]) -> SequentialCell:
        """Pop the first look-ahead cell whose mosaic has LANDED, else the head.

        Cells are ordered densest-first, and the densest zone is also the slowest
        to ingest — so taking them strictly in order makes the fleet wait on the
        very last mosaic of its opening window while smaller ones sit finished.
        Measured on the real coverage counts: a cluster's window spans ~4 h to
        ~10 h of ingest, so the strict order idles the GPUs for ~6 h at the start
        of every year.

        TWO separate conditions, easily conflated. To be CONSIDERED, a cell must be
        in the look-ahead window — those are the only ones whose ingest has been
        *started*, and reaching past them would begin a mosaic the budget has not
        admitted. To be TAKEN, its ingest must have *completed* (``ready``). A
        partial mosaic is never handed to inference under any branch: when nothing
        has landed this returns the head and the caller BLOCKS on it, which is
        exactly the previous behaviour, so an ingest-starved cluster is unaffected.

        ``ready`` is also true for a cell whose ingest FAILED (the future is done
        either way). That is deliberate — the caller's ``wait`` re-raises, the cell
        is recorded as failed, and the cluster continues with its others. Blocking
        forever on a mosaic that will never arrive is the worse outcome.

        The DENSITY ORDER ITSELF IS UNCHANGED: it still sizes the session (the
        fleet is provisioned for the largest cell up front) and still puts the
        island tail last. What changes is that it is no longer a barrier.
        """
        if inputs is not None:
            for idx, cell in enumerate(pending[: 1 + look_ahead]):
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
                if budget is not None and n_failed >= max_retained_failures:
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
                # Start ingests for the window BEFORE choosing, so a cell can only
                # be picked once its start has been admitted through the budget.
                _start_lookahead(pending)
                # Bounded admission; poll so a crashed session unwinds us.
                while not zone_slots.acquire(timeout=1.0):
                    if stop.is_set():
                        return
                if stop.is_set():
                    zone_slots.release()
                    return
                # Chosen AFTER the slot is held, so the pick reflects what has
                # landed by the time this cell can actually be worked.
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
                        zone_slots.release()
                        return
                    _record_failure(cell, "inputs/prepare", exc)
                    _retain_failed_mosaic(cell)  # mosaic (if any) retained for resume, counted
                    zone_slots.release()
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
                    if zplan.done is not None:
                        # Terminal (already complete / all-ocean) — committed
                        # and tagged inside plan(); nothing streams.
                        _record_outcome(zplan.done)
                        if inputs is not None:
                            inputs.cleanup(cell.zone, cell.year)
                        _release_mosaic(cell)
                        zone_slots.release()
                        continue
                    # Per-zone staged-resume scan (the single-zone path does
                    # this inside run_inference; the stream pre-filters here).
                    restored = ZarrWriter(prep.staging_base).scan_existing_staged_artifacts(
                        prep.run_id, zplan.live, compute_std=prep.config.compute_std, log=log
                    )
                    already = restored.done
                except Exception as exc:
                    _record_failure(cell, "plan", exc)
                    _retain_failed_mosaic(cell)  # mosaic retained for resume, counted
                    zone_slots.release()
                    continue
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
                    finalizer.submit(_finalize, tally)
        except BaseException as exc:
            # An exception OUTSIDE the per-cell guards (e.g. in _start_lookahead,
            # the enqueue, or finalizer.submit) would otherwise just kill this
            # daemon thread: feeder_done fires, _more_work returns None, the
            # session drains the partially-fed queue, and the run returns as if
            # complete with cells silently never enqueued. Capture it so the
            # caller can re-raise after the session drains.
            feeder_error.append(exc)
            log.error("Zone feeder crashed — remaining cells were not enqueued: %s", exc, exc_info=exc)
        finally:
            feeder_done.set()

    def _more_work() -> list[WorkItem] | None:
        """Scheduler-thread source: one prepared zone per poll, None = done."""
        with lock:
            if ready:
                return ready.popleft()
        if feeder_done.is_set():
            with lock:
                return ready.popleft() if ready else None
        return []  # nothing ready YET (ingest/plan still running) — keep polling

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
            finalizer.submit(_finalize, tally)

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
        # Unwind the feeder (it may be blocked on the slot semaphore or an
        # ingest wait) before propagating — a hung feeder thread would leak.
        stop.set()
        raise
    finally:
        feeder.join(timeout=600)
        if feeder.is_alive():
            log.warning("Zone feeder did not exit within 600s — continuing teardown (daemon thread)")
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
    # ONLY cells whose mosaic still EXISTS are eligible, and that is not the same as
    # "cells that failed": retrying a cell whose input was deleted runs against nothing,
    # which a first version of this pass did by keying off the failure list.
    #
    # As it stands every FAILING path retains its mosaic, so the two sets coincide — the
    # one path that deletes is a terminal plan, and that records a success. The filter is
    # kept because the property it protects is not a coincidence worth relying on: any
    # future delete-on-failure path reintroduces the bug the moment it lands, and this is
    # what makes that land safely. (The orbit-mismatch deferral cap used to be such a
    # path; cells now stream under their own orbit and nothing deletes on that account.)
    #
    # `retained_failed` is the set that kept its mosaic, so it is the eligibility list;
    # when there is no budget at all the mosaics are upstream and permanent, so
    # everything is eligible.
    for attempt in range(2, max_cell_attempts + 1):
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
        pending = failed_keys if budget is None else [k for k in failed_keys if k in eligible]
        if skipped := [k for k in failed_keys if k not in pending]:
            log.warning(
                "Not retrying %d cell(s) in-child — their mosaics were released or deleted, so a retry "
                "would run against nothing (they stay pending for the driver's next pass): %s",
                len(skipped),
                ", ".join(f"{z}-{y}" for z, y in skipped),
            )
        if not pending:
            break
        by_key = {(c.zone, c.year): c for c in cells}
        log.warning(
            "In-child retry attempt %d/%d over %d failed cell(s): %s",
            attempt,
            max_cell_attempts,
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
                inputs.cleanup(zone_name, year)
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
