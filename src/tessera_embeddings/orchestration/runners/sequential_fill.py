"""Chained multi-zone fill: many (zone, year) cells through ONE Ray session.

The per-cell fill chain (:mod:`.zone_fill`) provisions nothing itself — the
caller owns the Ray context. The parallel campaign exploits that by running
K fills at once, each in its own flow run with its own cluster; this runner
exploits it the other way: **one long-lived cluster whose actors are created
once and stream through every zone**, so the per-cluster costs — ``ray up``,
per-worker EC2 bringup (minutes each), the model-load cold start on every
worker — are paid once per shard of a campaign year instead of once per zone.

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
- **Ingest look-ahead** (``inputs``): the next cells' mosaics are ingested
  *while earlier cells infer*, bounded so in-flight mosaics stay within
  ADR-011's "peak input storage bounded by in-flight cells" (a semaphore of
  ``look_ahead + 2`` un-finalized zones, plus the ingest adapter's own
  concurrent-run bound).
- **Trailing assembly**: a completed zone's shard assembly (~10-15% of its
  inference wall time) runs on a background thread while later zones' tiles
  keep the GPUs busy. Assemblies serialize on one thread; a zone's mosaic is
  deleted only after its assembly lands. A zone counts as complete only when
  every tile's result is FINAL — the scheduler fires the completion callback
  after any deferred staging write confirms, so assembly never races an
  in-flight upload.

Idle-actor retirement needs no per-zone gating here: the scheduler suppresses
it while the work source is unexhausted and resumes it for the true shard
tail (see ``scheduling._process_chunks_work_stealing``).

A zone whose mosaic resolves a DIFFERENT s1 orbit than the shared session's
config cannot join the stream (actor configs are fixed at creation); such
cells are deferred and filled per-cell after the session ends, via the
caller-supplied ``infer_single`` fallback.

Contracts: Prefect-free (the deployment-backed ingest adapter, the
input-fingerprinted run_id, the per-cell config/plan, and the session itself
all arrive as callables from the flow layer); the caller is already inside a
Ray context; cells are one year's distinct zones, so no two commits for the
same zone group can ever be in flight (assemblies serialize; zones are
distinct).
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

    def wait(self, zone: str, year: int) -> None:
        """Block until the cell's mosaics are ready; raise if production failed."""
        ...

    def cleanup(self, zone: str, year: int) -> None:
        """Delete the cell's mosaics (only if this adapter produced them)."""
        ...


@dataclass
class SequentialCell:
    """One (zone, year) work item, with its preflight-derived tile count.

    ``num_actors`` sizes only the per-cell FALLBACK session (orbit-mismatch
    cells filled after the shared stream); the stream itself is sized by the
    flow's fleet parameter.
    """

    zone: str
    year: int
    num_actors: int


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
) -> dict[str, Any]:
    """Stream ``cells`` through one shared inference session, assembly trailing.

    A feeder thread walks the cells in order — wait for inputs, ``prepare``
    (orbit/config/run_id), ``plan`` (validation + live tiles), per-zone
    staged-resume scan — and enqueues each zone's tiles for the session's
    ``more_work`` source. The session interleaves zones at queue exhaustion;
    per-item completion callbacks tally each zone, and a completed zone's
    ``assemble`` (plus mosaic cleanup) runs on the trailing thread. A cell
    that fails in any phase is recorded and the stream continues — the cell
    stays pending in the campaign ledger for the next pass. (There is no
    consecutive-failure breaker here: zone outcomes interleave, and the
    scheduler already aborts on systemic actor-death storms.)

    Args:
        cells: Ordered (zone, year) work items — one year's distinct zones,
            largest-first.
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
        session_s1_orbit: The shared session's actor-config orbit; a cell
            whose resolved orbit differs is deferred to ``infer_single``.
        log: Logger.
        inputs: Mosaic lifecycle adapter; ``None`` means the mosaics already
            exist upstream (no starts, no waits, no cleanup).
        look_ahead: Cells beyond the feed head to keep in ingest flight; also
            sizes the un-finalized-zone bound (``look_ahead + 2``).

    Returns:
        Summary dict: per-cell outcomes, failure records, deferral count, and
        timing.

    Raises:
        RuntimeError: After all cells have been attempted, when any cell
            failed (completed cells are already committed + tagged and drop
            out of the next campaign pass).
    """
    t0 = time.monotonic()
    lock = threading.Lock()
    outcomes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    deferred: list[tuple[SequentialCell, PreparedCell]] = []
    tallies: dict[str, _ZoneTally] = {}  # run_id → tally
    ready: deque[list[WorkItem]] = deque()  # zones awaiting injection, in cell order
    feeder_done = threading.Event()
    stop = threading.Event()  # session crashed — unwind the feeder
    # Bounds un-finalized zones (mosaic held from prepare until assembly lands):
    # the current zone + one trailing assembly + the ingest look-ahead.
    zone_slots = threading.Semaphore(look_ahead + 2)
    finalizer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trailing-assembly")

    def _record_failure(cell: SequentialCell, phase: str, exc: BaseException) -> None:
        with lock:
            failures.append({"zone": cell.zone, "year": cell.year, "phase": phase, "error": str(exc)})
        log.error("Cell %s-%d failed during %s: %s", cell.zone, cell.year, phase, exc, exc_info=exc)

    def _record_outcome(result: dict[str, Any]) -> None:
        with lock:
            outcomes.append(result)

    def _start_lookahead(feed_index: int) -> None:
        """Keep ingests running ahead of the feed head (idempotent starts)."""
        if inputs is None:
            return
        for cell in cells[feed_index : feed_index + 1 + look_ahead]:
            inputs.start(cell.zone, cell.year)

    def _finalize(tally: _ZoneTally) -> None:
        """Trailing-thread body: assemble a completed zone, release its slot.

        The mosaic delete (potentially multi-TB) also runs here, overlapping
        later zones' inference. A failed zone keeps its mosaic — the retry
        re-derives its fingerprinted run_id from the mosaic's ingest marker
        and resumes its staged tiles.
        """
        cell, prep = tally.cell, tally.prep
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
        except Exception as exc:
            _record_failure(cell, "assembly", exc)
        finally:
            zone_slots.release()

    def _feed() -> None:
        """Walk cells in order: inputs → prepare → plan → scan → enqueue."""
        try:
            for i, cell in enumerate(cells):
                _start_lookahead(i)
                # Bounded admission; poll so a crashed session unwinds us.
                while not zone_slots.acquire(timeout=1.0):
                    if stop.is_set():
                        return
                if stop.is_set():
                    zone_slots.release()
                    return
                try:
                    if inputs is not None:
                        inputs.wait(cell.zone, cell.year)
                    prep = prepare(cell)
                except Exception as exc:
                    _record_failure(cell, "inputs/prepare", exc)
                    zone_slots.release()
                    continue
                if prep.config.s1_orbit != session_s1_orbit:
                    # Actor configs are fixed at session creation — this cell
                    # runs per-cell after the stream (see infer_single).
                    log.info(
                        "Cell %s-%d resolved s1_orbit=%s != session %s — deferring to the per-cell fallback",
                        cell.zone,
                        cell.year,
                        prep.config.s1_orbit,
                        session_s1_orbit,
                    )
                    with lock:
                        deferred.append((cell, prep))
                    zone_slots.release()
                    continue
                try:
                    zplan = plan(cell, prep)
                    if zplan.done is not None:
                        # Terminal (already complete / all-ocean) — committed
                        # and tagged inside plan(); nothing streams.
                        _record_outcome(zplan.done)
                        if inputs is not None:
                            inputs.cleanup(cell.zone, cell.year)
                        zone_slots.release()
                        continue
                    # Per-zone staged-resume scan (the single-zone path does
                    # this inside run_inference; the stream pre-filters here).
                    already = ZarrWriter(prep.staging_base).scan_existing_staged_chunks(
                        prep.run_id, zplan.live, compute_std=prep.config.compute_std, log=log
                    )
                except Exception as exc:
                    _record_failure(cell, "plan", exc)
                    zone_slots.release()
                    continue
                live = [c for c in zplan.live if c.label not in already]
                resumed = [
                    {"chunk": label, "status": "success", "valid_pixels": 0, "elapsed_sec": 0.0, "resumed": True}
                    for label in already
                ]
                tally = _ZoneTally(cell=cell, prep=prep, plan=zplan, remaining=len(live), results=resumed)
                ctx = ZoneContext(prep.mosaic_base, prep.staging_base, prep.run_id)
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

    # Orbit-mismatch cells: per-cell sessions on the same (still-provisioned)
    # cluster, after the stream so they never contend with it for GPUs.
    with lock:
        fallback_cells = list(deferred)
    for j, (cell, prep) in enumerate(fallback_cells):
        try:
            handoff = infer_single(cell, prep, j == len(fallback_cells) - 1)
            _record_outcome(assemble(handoff, prep))
            if inputs is not None:
                inputs.cleanup(cell.zone, cell.year)
        except Exception as exc:
            _record_failure(cell, "fallback", exc)

    elapsed = time.monotonic() - t0
    summary: dict[str, Any] = {
        "cells": len(cells),
        "succeeded": len(outcomes),
        "failed": len(failures),
        "deferred_orbit_mismatch": len(fallback_cells),
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
