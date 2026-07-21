"""Sequential multi-zone fill: many (zone, year) cells through ONE Ray cluster.

The per-cell fill chain (:mod:`.zone_fill`) provisions nothing itself — the
caller owns the Ray context. The parallel campaign exploits that by running
K fills at once, each in its own flow run with its own cluster; this runner
exploits it the other way: **one long-lived cluster, cells strictly
sequential**, so the per-cluster costs — `ray up`, per-worker EC2 bringup
(minutes each), the model-load cold start on every worker — are paid once per
campaign year instead of once per zone.

Keeping the shared fleet busy between cells is the whole game, and three
mechanisms cooperate on it:

- **Ingest look-ahead** (``inputs``): the next cells' mosaics are ingested
  *while the current cell infers*, bounded by ``look_ahead`` so in-flight
  mosaics stay within ADR-011's "peak input storage bounded by in-flight
  cells" (at most ``1 + look_ahead`` cells hold a live mosaic, plus one
  trailing cell whose mosaic is deleted after its assembly lands).
- **Trailing assembly**: a cell's shard assembly (~10-15% of its inference
  wall time) runs on a background thread while the next cell's inference
  keeps the GPUs busy. Depth 1 — a cell's assembly is joined before the next
  one is submitted. Within one run every cell is a distinct zone group (the
  caller passes one year's zones), so a trailing commit can never hit the
  same-zone attr conflict that forbids concurrent same-zone fills.
- **Fleet retention**: every cell but the last runs inference with
  ``retire_idle_actors=False`` so its tail doesn't idle-kill the actors whose
  instances the next cell needs (an actor-less node hits the autoscaler idle
  timeout and drains); the final cell retires normally so the fleet tapers
  as the run ends. Order ``cells`` largest-first — the fleet then only ever
  shrinks, and small island zones land at the natural taper.

Contracts: Prefect-free (the deployment-backed ingest adapter, the
input-fingerprinted run_id, and the per-cell config all arrive as callables
from the flow layer); the caller is already inside a Ray context; cells are
one year's distinct zones (see the trailing-assembly note above for why).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from tessera_embeddings.orchestration.runners.zone_fill import assemble_zone_year, infer_zone_year

if TYPE_CHECKING:
    import icechunk

    from tessera_embeddings.config.inference import InferenceConfig
    from tessera_embeddings.storage.shard_writer import CommitGate


class CellInputs(Protocol):
    """Lifecycle of a cell's input mosaics, implemented by the flow layer.

    The runner drives *when* (start ``look_ahead`` cells early, wait before
    inferring, clean up after the cell lands); the implementation owns *how*
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
    """One (zone, year) work item, with its preflight-derived actor budget."""

    zone: str
    year: int
    # min(fleet size, live tiles) — a 3-tile island zone must not request a
    # 30-actor fleet whose surplus would idle-drain mid-campaign.
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


def fill_zones_sequential(
    *,
    cells: list[SequentialCell],
    store_path: str,
    land_mask_path: str,
    prepare: Callable[[SequentialCell], PreparedCell],
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    inputs: CellInputs | None = None,
    look_ahead: int = 2,
    gate: CommitGate | None = None,
    n_assembly_workers: int | None = None,
    s3_concurrency: int | None = None,
    cleanup_staging: bool = True,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
    on_actor_retire: Callable[[str], None] | None = None,
    max_consecutive_failures: int = 3,
) -> dict[str, Any]:
    """Fill ``cells`` in order on the caller's Ray cluster, assembly trailing.

    Per cell: wait for its inputs (mosaics), top up the ingest look-ahead,
    ``prepare`` it (orbit/config/run_id/coverage gate — the flow's callable),
    run the inference phase on the shared cluster, then hand the assembly
    phase to the trailing thread and move on. A cell that fails (ingest,
    prepare, inference, or assembly) is recorded and the run continues — the
    cell stays pending in the campaign ledger for the next pass — unless
    ``max_consecutive_failures`` cells fail in a row, which indicates a
    systemic problem (bad checkpoint, broken store) not worth burning the
    fleet on.

    Args:
        cells: Ordered (zone, year) work items — one year's distinct zones,
            largest-first (see the module docstring for both constraints).
        store_path: URI of the global Icechunk repo.
        land_mask_path: Campaign coverage-store URI.
        prepare: Resolves a cell's :class:`PreparedCell` once its inputs are
            ready. Raising here (coverage gate, fingerprint failure) fails the
            cell, not the run.
        log: Logger.
        inputs: Mosaic lifecycle adapter; ``None`` means the mosaics already
            exist upstream (no starts, no waits, no cleanup).
        look_ahead: Cells beyond the current one to keep in ingest flight.
        gate: In-process commit gate shared with any concurrent committers.
        n_assembly_workers: Assembly worker-process count per cell
            (``None`` = sized from each cell's live-tile count).
        s3_concurrency: Each assembly's slice of the fleet S3-PUT budget.
        cleanup_staging: Delete a cell's staged tiles after it lands.
        get_credentials: Icechunk credential callback for actors + stores.
        s3_region: S3 region for the global store + mosaics.
        on_actor_retire: EC2 terminator for retired/replaced actors. Passed to
            every cell; it only fires from idle retirement (gated off until
            the final cell) or actor replacement, so intermediate cells keep
            their fleet while dead instances still get reclaimed promptly.
        max_consecutive_failures: Consecutive-cell-failure circuit breaker.

    Returns:
        Summary dict: per-cell outcomes, failure records, and timing.

    Raises:
        RuntimeError: When the circuit breaker trips, or — after all cells
            have been attempted — when any cell failed (so the enclosing flow
            run fails and the campaign driver surfaces it; completed cells
            are already committed + tagged and drop out of the next work
            list).
    """
    t0 = time.monotonic()
    outcomes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    consecutive = 0
    # (cell, future) of the one in-flight trailing assembly, if any.
    trailing: tuple[SequentialCell, Future[dict[str, Any]]] | None = None

    def _fail(cell: SequentialCell, phase: str, exc: BaseException) -> None:
        nonlocal consecutive
        failures.append({"zone": cell.zone, "year": cell.year, "phase": phase, "error": str(exc)})
        consecutive += 1
        log.error("Cell %s-%d failed during %s: %s", cell.zone, cell.year, phase, exc, exc_info=exc)
        if consecutive >= max_consecutive_failures:
            raise RuntimeError(
                f"{consecutive} consecutive cell failures (latest: {cell.zone}-{cell.year} during {phase}) — "
                "aborting the sequential fill rather than burning the fleet on a systemic problem. "
                f"All failures so far: {failures}"
            ) from exc

    def _succeed(cell: SequentialCell, result: dict[str, Any]) -> None:
        nonlocal consecutive
        consecutive = 0
        outcomes.append(result)
        if inputs is not None:
            inputs.cleanup(cell.zone, cell.year)

    def _join_trailing() -> None:
        """Collect the in-flight assembly's outcome (success resets the breaker)."""
        nonlocal trailing
        if trailing is None:
            return
        cell, future = trailing
        trailing = None
        try:
            result = future.result()
        except Exception as exc:
            # Keep the cell's mosaic: a retry re-derives its fingerprinted
            # run_id from the mosaic's ingest marker, and resumes its staged
            # tiles — deleting the mosaic here would orphan both.
            _fail(cell, "assembly", exc)
        else:
            _succeed(cell, result)

    def _start_lookahead(next_index: int) -> None:
        """Keep the next ``look_ahead`` cells' ingests in flight (idempotent)."""
        if inputs is None:
            return
        for cell in cells[next_index : next_index + look_ahead]:
            inputs.start(cell.zone, cell.year)

    log.info(
        "Sequential fill: %d cell(s) on one cluster, look_ahead=%d, breaker at %d consecutive failures",
        len(cells),
        look_ahead,
        max_consecutive_failures,
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trailing-assembly")
    try:
        if inputs is not None and cells:
            inputs.start(cells[0].zone, cells[0].year)
            _start_lookahead(1)
        for i, cell in enumerate(cells):
            final = i == len(cells) - 1
            try:
                if inputs is not None:
                    inputs.wait(cell.zone, cell.year)
                    _start_lookahead(i + 1)
                prep = prepare(cell)
            except Exception as exc:
                _fail(cell, "prepare", exc)
                continue
            try:
                handoff = infer_zone_year(
                    store_path=store_path,
                    zone=cell.zone,
                    year=cell.year,
                    land_mask_path=land_mask_path,
                    mosaic_base=prep.mosaic_base,
                    staging_base=prep.staging_base,
                    config=prep.config,
                    num_actors=cell.num_actors,
                    log=log,
                    run_id=prep.run_id,
                    gate=gate,
                    get_credentials=get_credentials,
                    s3_region=s3_region,
                    on_actor_retire=on_actor_retire,
                    # Only the final cell's tail may drain the fleet — earlier
                    # tails would idle-kill the very actors (and, via the
                    # autoscaler idle timeout, instances) the next cell needs.
                    retire_idle_actors=final,
                )
            except Exception as exc:
                _fail(cell, "inference", exc)
                continue
            # Depth-1 trailing: collect the previous cell's assembly before
            # submitting this one. By now it has typically long finished
            # (assembly ≈ 10-15% of a cell's inference time), so this join is
            # usually instant; when it isn't, blocking here is the
            # backpressure that keeps assemblies from stacking up.
            _join_trailing()
            if handoff.done is not None:
                # Terminal in the inference phase (already complete / all-ocean
                # empty) — committed and tagged there; nothing to assemble.
                _succeed(cell, handoff.done)
                continue
            trailing = (
                cell,
                executor.submit(
                    assemble_zone_year,
                    handoff,
                    store_path=store_path,
                    staging_base=prep.staging_base,
                    log=log,
                    gate=gate,
                    n_assembly_workers=n_assembly_workers,
                    s3_concurrency=s3_concurrency,
                    cleanup_staging=cleanup_staging,
                    get_credentials=get_credentials,
                    s3_region=s3_region,
                ),
            )
        _join_trailing()
    finally:
        # On the breaker path a submitted assembly may still be committing —
        # wait for it rather than abandoning a commit mid-flight. Its outcome
        # is already accounted for unless we're unwinding on the breaker, in
        # which case the ledger is the authority (a landed+tagged cell drops
        # out of the next work list either way).
        if trailing is not None:
            log.warning("Waiting for the trailing assembly of %s-%d before exiting", trailing[0].zone, trailing[0].year)
        executor.shutdown(wait=True)

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
            f"{len(failures)}/{len(cells)} cell(s) failed in the sequential fill "
            f"(completed cells are committed + tagged and will be skipped on the next campaign pass): {failures}"
        )
    log.info("Sequential fill complete: %d/%d cells in %.1f min", len(outcomes), len(cells), elapsed / 60)
    return summary
