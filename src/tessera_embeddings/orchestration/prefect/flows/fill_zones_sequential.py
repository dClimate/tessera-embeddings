"""Fill many zones of one campaign year through a SINGLE shared Ray cluster.

The parallel campaign path (:mod:`.run_global_campaign` dispatching
:mod:`.fill_zone_year` per cell) provisions a Ray cluster per (zone, year) —
each one re-paying ``ray up`` (~5-10 min wall), per-worker EC2 bringup
(minutes of billed GPU idle each), the per-worker model-load cold start, and
a fresh roll of the EC2 capacity dice. This flow amortizes all of that across
a whole year's zones: ONE cluster, ONE set of actors, zones streamed
**near-sequentially** — strictly ordered, with the next zone's tiles
interleaving only once the current zone's queue is exhausted, so zone tails
never idle the fleet and actors are never re-created between zones. How the
stream works (interleaving, ingest look-ahead, trailing assembly) is the
chained runner's story — see
:mod:`tessera_embeddings.orchestration.runners.sequential_fill`; this flow
supplies the Prefect-facing pieces (deployment-backed ingest, fingerprinted
run_ids, per-cell config, the shared session itself) and the cluster, with
its autoscaler ``idle_timeout_minutes`` raised so ingest-starved gaps in the
stream never shed workers.

Cheap cells never touch the cluster: already-complete cells are retagged and
all-ocean cells are marked complete-empty *before* Ray is provisioned — and,
unlike the per-cell path, an all-ocean cell is never ingested at all. Live
cells are ordered largest-first (clamped ``min(num_actors, live tiles)``
requests) so the autoscaled fleet only ever shrinks across the run.

Triage also preflights each live cell's OPTICAL source
(:func:`tessera_embeddings.ingest.source_coverage.preflight_optical_source`) when this
flow will ingest: a cell whose source catalogue confirmably publishes nothing reaching
its live land in its window is refused here, where refusal costs seconds, instead of
failing its coverage gate after its ingest has run and the shared fleet is up. What the
preflight guarantees: it refuses only on a POSITIVE finding of absence — probes of the
ingest's own catalogue, over boxes jointly covering all live land, with the window
padded per the solar-day convention, all cleanly empty — a condition under which even
the ``allow_partial_window``-relaxed gate must fail. What it does NOT guarantee: that a
passed cell is buildable (a catalogue hit is provisional; the fill's gates remain the
authority), nor that every doomed cell is caught (an inconclusive probe passes the cell
through — losing a buildable cell to a flaky check would cost campaign coverage, which
is worse than the cluster time it saves). Refused cells are reported in the summary
(``no_optical``) and left unfilled; ``ingest=False`` skips the preflight entirely,
because pre-built mosaics, not the catalogue, are then the source of truth.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ThreadPoolExecutor, wait
from functools import cache, partial
from typing import TYPE_CHECKING, Any

from prefect import flow, get_run_logger
from prefect.client.orchestration import get_client
from prefect.deployments import run_deployment
from prefect.runtime import flow_run as flow_run_ctx
from prefect.states import Cancelling

from tessera_embeddings.config.fault_injection import WITHHOLD_WORK, FaultInjection
from tessera_embeddings.config.inference import checkpoint_filename
from tessera_embeddings.config.ingest import IngestSettings
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.config.time_windows import TimeWindow, parse_time_window
from tessera_embeddings.inference.data_loading import check_time_window_coverage, resolve_s1_orbit
from tessera_embeddings.inference.orchestration_helpers import build_inference_config
from tessera_embeddings.inference.runner import run_inference
from tessera_embeddings.inference.scheduling import WorkItem
from tessera_embeddings.ingest.source_coverage import SourceFinding, preflight_optical_source
from tessera_embeddings.orchestration.prefect._fleet_gate import FleetGate, pause_signal
from tessera_embeddings.orchestration.prefect.flows._cell_validation import (
    cell_validation_parameters,
    dispatch_cell_validation,
    validation_run_tag,
)
from tessera_embeddings.orchestration.prefect.flows._child_runs import (
    CANCELLATION_CONFIRM_S,
    child_run_tag,
    make_child_cancel_hook,
)
from tessera_embeddings.orchestration.prefect.flows._overrides import set_overrides
from tessera_embeddings.orchestration.prefect.flows._ray_lifecycle import (
    activate,
    deactivate,
    ray_cleanup_on_cancellation,
)
from tessera_embeddings.orchestration.prefect.flows.fill_zone_year import (
    _assert_seeded_model_matches,
    _optical_min_obs_from_store,
)
from tessera_embeddings.orchestration.prefect.flows.run_global_campaign import (
    _ingest_dispatch_params,
    _resolve_ami_id,
    _resolve_code_identity,
    _staging_run_id,
)
from tessera_embeddings.orchestration.prefect.flows.tessera_full_pipeline import _check_completed
from tessera_embeddings.orchestration.runners.sequential_fill import (
    PreparedCell,
    SequentialCell,
    fill_zones_sequential,
)
from tessera_embeddings.orchestration.runners.zone_fill import (
    ZonePlan,
    assemble_zone_year,
    assert_calendar_year_window,
    fill_zone_year,
    infer_zone_year,
    plan_zone_inference,
    zone_live_tile_count,
    zone_year_complete,
    zone_year_on_axis,
)
from tessera_embeddings.storage.object_store import delete_prefix
from tessera_embeddings.storage.zone_grid import canonicalize_zone

if TYPE_CHECKING:
    from tessera_embeddings.config.inference import InferenceConfig
    from tessera_embeddings.orchestration.runners.zone_fill import ZoneFillHandoff

# How often a cell-ingest worker polls its child flow run for a terminal state.
# Ingests run tens of minutes, so this granularity is noise there; shutdown
# doesn't wait on it (pollers wake immediately via the _stopping event).
_INGEST_POLL_S = 15.0
# Consecutive read_flow_run errors tolerated before a poller gives up. A
# transient Prefect API/network blip must not fail the ingest wait (or, worse,
# deregister a still-running child so shutdown can no longer cancel it); we
# retry across it. A PERSISTENT failure eventually raises — leaving the id
# registered so shutdown's sweep can still reach the live child.
_INGEST_POLL_MAX_ERRORS = 10
# Lease length for a held fleet-wide ingest slot. Renewed in the background for the
# life of the ingest; generous so a slow renewal round-trip never drops the slot.
_INGEST_LEASE_S = 900.0


class _DeploymentCellInputs:
    """Ingest-deployment adapter satisfying the runner's ``CellInputs`` protocol.

    ``start`` submits a worker thread that creates the ingest flow run
    (``run_deployment(timeout=0)``, returning immediately with the run's id)
    and then POLLS it to a terminal state, so a cell's ``wait`` is a future
    join and — crucially — the child run's id is known while it executes:
    ``shutdown`` can request cancellation of every in-flight ingest instead of
    orphaning children that keep writing after the parent has failed (a quick
    retry would then race the orphan on the same mosaic prefix). ``max_parallel``
    sizes the executor, and the flow passes its full cell count: every UTM zone of
    a cluster is submitted at once so they finish together, since the cluster waits
    for all of them before requesting GPUs. What actually limits how many RUN at
    once is ``ingest_limit_name``, a Prefect global concurrency limit shared by
    every cluster — the clusters are separate flow runs, so only a server-side gate
    can bound them together. ``cleanup`` deletes only mosaics THIS adapter produced
    (a cell never started here is someone else's input), matching the parallel
    driver's ``did_ingest and cleanup_mosaics`` rule.

    Threads don't inherit the flow's Prefect context, so ingest runs dispatch
    without a parent-run link — they are still tracked as ordinary runs of the
    ingest deployment.
    """

    def __init__(
        self,
        *,
        deployment: str,
        params_for: Callable[[str, int], dict[str, Any]],
        inputs_bucket: str,
        cleanup_mosaics: bool,
        max_parallel: int,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger],
        child_tag: str | None = None,
        ingest_limit_name: str | None = None,
    ) -> None:
        self._deployment = deployment
        self._params_for = params_for
        self._inputs_bucket = inputs_bucket
        self._cleanup_mosaics = cleanup_mosaics
        self._log = log
        self._ingest_limit_name = ingest_limit_name
        # Deterministic tag stamped on every child run, derived from the PARENT
        # flow-run id — the cancellation/crash hook re-derives it in a fresh
        # process and sweeps live children by tag (the in-process shutdown()
        # never runs when Prefect kills the flow process). Mirrors the Ray
        # terminate-instances-by-tag pattern.
        self._child_tag = child_tag
        # Kept, not just handed to the executor: it is also the failure quorum
        # ``wait_first`` aborts on — see there for why that must not be the whole list.
        self._max_parallel = max(1, max_parallel)
        self._executor = ThreadPoolExecutor(max_workers=self._max_parallel, thread_name_prefix="cell-ingest")
        self._futures: dict[tuple[str, int], Future[None]] = {}
        self._lock = threading.Lock()
        self._inflight: dict[tuple[str, int], Any] = {}  # (zone, year) → flow-run id
        self._stopping = threading.Event()

    def _run(self, zone: str, year: int) -> None:
        """Hold a fleet-wide ingest slot for this zone, then dispatch and poll it.

        The slot is the ONLY thing limiting how many zones ingest at once. Every
        cluster submits all of its zones immediately; the gate decides how many
        actually start, so a cluster spins up only as many as fit under the
        fleet-wide cap and the rest wait here for a slot. Held across the whole
        ingest (dispatch through terminal state), because that is the window in
        which the zone owns a Dask cluster.

        A Prefect GLOBAL concurrency limit rather than a local semaphore: the
        clusters are separate flow runs on separate machines, so nothing
        in-process can see across them.

        **This is also the campaign's pause lever.** Lowering the limit to zero
        makes every stream HOLD here — the cell in flight finishes, no new cell is
        taken up, and nothing fails (:class:`FleetGate`). Raising it resumes. The
        hold is abandoned if the runner is already shutting down, so a pause
        cannot outlive the run it is pausing.
        """
        if self._ingest_limit_name is None:
            self._dispatch_and_wait(zone, year)
            return
        with FleetGate(
            self._ingest_limit_name,
            occupy=1,
            log=self._log,
            # A stream winding down must not park here waiting for a limit that a
            # human may never raise; its cells are the driver's next round's work.
            should_stop=self._stopping.is_set,
            # Generous, because an ingest runs tens of minutes to hours and the
            # lease is renewed in the background throughout.
            lease_duration=_INGEST_LEASE_S,
            # ...but a renewal blip must not kill an ingest that is already hours
            # deep. The cap is a cost control, not a correctness invariant, so a
            # transient overshoot beats discarding the work.
            raise_on_lease_renewal_failure=False,
        ):
            self._dispatch_and_wait(zone, year)

    def _dispatch_and_wait(self, zone: str, year: int) -> None:
        # Never CREATE a child once shutdown began — a run dispatched after the
        # cancellation sweep would be the one orphan the sweep can't see.
        if self._stopping.is_set():
            raise RuntimeError(f"ingest {zone}-{year} not dispatched: adapter shutting down")
        self._log.info("Ingest dispatch: %s-%d via %s", zone, year, self._deployment)
        # timeout=0: return as soon as the run is created so its id is in hand —
        # a blocking run_deployment() only yields the run AFTER it finishes,
        # leaving shutdown nothing to cancel.
        # run_deployment is @sync_compatible (typed as a union with its coroutine
        # form); in these worker threads it always runs synchronously.
        run: Any = run_deployment(
            self._deployment,
            parameters=self._params_for(zone, year),
            timeout=0,
            tags=[self._child_tag] if self._child_tag else None,
        )
        with self._lock:
            self._inflight[(zone, year)] = run.id
        run = self._poll_until_terminal(run.id, f"{zone}-{year}")
        # Deregister ONLY after a CONFIRMED terminal state. A transient poll
        # error or a shutdown-abandon raises above WITHOUT reaching here, so the
        # id stays in _inflight and shutdown's sweep (and the abandon-cancel
        # below) can still reach a child that is still running server-side.
        with self._lock:
            self._inflight.pop((zone, year), None)
        _check_completed(run, f"ingest {zone}-{year}")

    def _poll_until_terminal(self, flow_run_id: Any, label: str) -> Any:  # noqa: ANN401 — Prefect FlowRun
        errors = 0
        with get_client(sync_client=True) as client:
            while not self._stopping.is_set():
                try:
                    run = client.read_flow_run(flow_run_id)
                except Exception:
                    # Transient API/network blip: retry rather than fail the
                    # ingest wait — and never deregister here, so the id stays
                    # visible to shutdown. Give up only after a persistent run.
                    errors += 1
                    self._log.warning(
                        "Error polling ingest %s (%s), attempt %d/%d — retrying",
                        label,
                        flow_run_id,
                        errors,
                        _INGEST_POLL_MAX_ERRORS,
                        exc_info=True,
                    )
                    if errors >= _INGEST_POLL_MAX_ERRORS:
                        raise RuntimeError(
                            f"ingest {label}: gave up after {errors} consecutive poll errors "
                            f"(run {flow_run_id} may still be live — left for the shutdown sweep)"
                        ) from None
                    self._stopping.wait(_INGEST_POLL_S)
                    continue
                errors = 0
                if run.state is not None and run.state.is_final():
                    return run
                self._stopping.wait(_INGEST_POLL_S)
            # Abandoning: this child may have been created after (or registered
            # too late for) shutdown's snapshot sweep — cancel it OURSELVES with
            # the already-open client before raising, so a dispatch racing
            # shutdown can't leave a server-side ingest running.
            try:
                client.set_flow_run_state(flow_run_id, state=Cancelling())
                self._log.warning("Cancelled abandoned ingest %s (%s)", label, flow_run_id)
            except Exception:
                self._log.warning("Could not cancel abandoned ingest %s (%s)", label, flow_run_id, exc_info=True)
        raise RuntimeError(f"ingest {label} abandoned: adapter shutting down")

    def start(self, zone: str, year: int) -> None:
        if (zone, year) not in self._futures:
            self._futures[(zone, year)] = self._executor.submit(self._run, zone, year)

    def wait(self, zone: str, year: int, stop: threading.Event | None = None) -> None:
        self.start(zone, year)
        fut = self._futures[(zone, year)]
        while True:
            try:
                return fut.result(timeout=1.0)
            except TimeoutError:
                # Runner unwind (crashed session): return promptly instead of
                # sitting behind the full remaining ingest duration.
                if stop is not None and stop.is_set():
                    raise RuntimeError(f"wait for ingest {zone}-{year} aborted: runner stopping") from None

    def ready(self, zone: str, year: int) -> bool:
        """True if this cell's ingest has finished (or failed). Never blocks."""
        fut = self._futures.get((zone, year))
        return fut is not None and fut.done()

    def wait_first(self, cells: list[tuple[str, int]], timeout: float | None = None) -> tuple[str, int] | None:
        """Block until ANY of ``cells`` has ingested; return which, or None on timeout.

        The flow waits for a mosaic before requesting GPUs, and which one it waits
        for decides how long a fleet sits idle. Waiting on a NAMED cell means
        waiting on the densest zone in the window — the slowest to ingest — while
        smaller ones land hours earlier. Any landed mosaic is enough to start.

        A FAILED ingest is not a landed mosaic, so it does not end the wait while any
        sibling is still running. The caller's next act is to request a GPU fleet, and a
        fast failure — bad credentials, a bad parameter — would otherwise boot one within
        seconds for a mosaic that does not exist. Keep waiting instead: whichever sibling
        lands first is a real mosaic, and the failure still surfaces when the runner
        reaches that cell.

        Once every cell of the FIRST WAVE — the first ``max_parallel`` of ``cells``, which
        are the ones the pool starts — has failed, this RAISES rather than blocking on the
        rest or nominating one of the failures. Reporting a failed cell as landed would send
        the caller straight into ``ray up`` — five to ten minutes of billed GPU bringup for a
        mosaic that does not exist, torn down again the moment the feeder reaches the same
        failure. Raising costs nothing and loses nothing: the underlying ingest error is
        chained as the cause, and the priming this call belongs to runs inside the flow's
        shutdown guard, so the started children are still cancelled on the way out.

        The quorum is that named wave, NOT the whole list and NOT a running failure count.
        The list would hold a systematic failure through wave after wave before surfacing it.
        A running count is the subtler trap: the executor starts a queued cell as soon as a
        worker frees, so fast failures accumulate across waves while a slow first-wave cell
        is still running and about to succeed — aborting there cancels the very ingest that
        was going to land.

        ``None`` therefore means one thing only: the wait timed out (or no cell of
        ``cells`` was ever started).
        """
        futs = {self._futures[c]: c for c in cells if c in self._futures}
        if not futs:
            return None
        deadline = None if timeout is None else time.monotonic() + timeout
        pending = set(futs)
        # The abort signal is a fully-failed FIRST WAVE — not a fully-failed list (the caller
        # offers every live cell, so that would run a doomed cluster through wave after wave),
        # and not a running failure total. The total is the trap: the pool starts a queued cell
        # as soon as a worker frees, so fast failures accumulate across waves while a slow
        # first-wave cell is still running and about to succeed, and aborting cancels it.
        # These cells are the ones the pool starts first, so "all failed" cannot be true while
        # any might still succeed. Same quorum as when the caller passed a window.
        ordered = [c for c in cells if c in self._futures]
        first_wave = {self._futures[c] for c in ordered[: self._max_parallel]}
        failed_first_wave: set[Future[None]] = set()
        while pending:
            budget = None if deadline is None else max(0.0, deadline - time.monotonic())
            done, pending = wait(pending, timeout=budget, return_when=FIRST_COMPLETED)
            if not done:
                return None
            for fut in done:
                # .exception() cannot block — every future in `done` is settled. A
                # CANCELLED one raises instead of returning, and is not a mosaic either.
                try:
                    if fut.exception() is None:
                        return futs[fut]
                except CancelledError:
                    pass
                if fut in first_wave:
                    failed_first_wave.add(fut)
            if failed_first_wave == first_wave:
                # EVERY first-wave failure, not this iteration's `done`: the loop drains
                # `pending` over several waits, so `done` holds only whatever settled last —
                # and reporting that made a whole pool's failure read as one cell, hiding both
                # the extent and, when they failed for one shared reason, the cause.
                self._raise_window_all_failed(futs, failed_first_wave)
        return None

    @staticmethod
    def _raise_window_all_failed(futs: dict[Future[None], tuple[str, int]], failed: set[Future[None]]) -> None:
        """A pool's worth of ingests failed with none succeeding — abort the priming wait.

        ``failed`` must be every failure seen so far, not one wait's slice of them: the
        message names the cells an operator will go and look at, and a short list points at
        the wrong subsystem when one shared fault took the whole pool down.
        """
        cells = sorted(futs[f] for f in failed)
        cause: BaseException | None = None
        for fut in failed:
            try:
                cause = fut.exception()
            except CancelledError as exc:
                cause = exc
            if cause is not None:
                break
        listed = ", ".join(f"{z}-{y}" for z, y in cells)
        raise RuntimeError(
            f"{len(cells)} ingest(s) failed with none landing ({listed}) — there is no mosaic "
            f"to infer, so no GPU fleet is requested. The first failure is chained below."
        ) from cause

    def cleanup(self, zone: str, year: int) -> None:
        """Delete the cell's mosaics, and RAISE if that did not happen.

        ``strict=True`` because the caller acts on the outcome: a mosaic that silently
        failed to delete is multi-terabyte data left behind while the runner frees the
        cell's budget slot and admits another ingest, so best-effort here lets repeated
        storage or permission failures accumulate every cluster's mosaic off-budget. The
        runner catches this and leaks the cell loudly rather than failing it — the cell has
        already landed — but it can only do that if the failure reaches it.
        """
        if (zone, year) not in self._futures or not self._cleanup_mosaics:
            return
        delete_prefix(f"{self._inputs_bucket.rstrip('/')}/mosaics/{zone}/{year}", log=self._log, strict=True)

    def discard(self, zone: str, year: int) -> None:
        """Forget the cell's ingest future, cancelling any child still registered for it.

        Whatever the attempt committed to the mosaic stays, and the fresh ingest that
        follows resumes it rather than rebuilding.

        Ending the old child is the load-bearing part. A cell reaches here because its
        ingest FAILED, and for one failure mode the child is still alive: when polling
        gives up after a persistent run of API errors it deliberately leaves the run
        registered, because the server-side ingest may well be running fine and only the
        parent's view of it broke. Re-dispatching on top of that gives one mosaic prefix
        two concurrent writers — and the ingest path forbids exactly that, since its
        commits do not rebase and the second writer's ConflictError is not retried.

        **A cancellation is a REQUEST, not a fact**, and this used to treat it as one:
        it asked, dropped the run id, and returned. Between the request and the run
        actually stopping, the child is still writing — and having dropped the id,
        ``shutdown``'s sweep could no longer see it either, so the one thing that would
        have caught it later was given up in the same breath. So the request is now
        followed until the run reports a TERMINAL state, and the id stays registered the
        whole time.

        Raises:
            RuntimeError: if the old child cannot be confirmed finished. The caller must
                not re-dispatch — better to lose a recoverable cell to a failure that
                names itself than to publish a mosaic two runs wrote at once, which
                nothing downstream can detect and no retry repairs.
        """
        self._futures.pop((zone, year), None)
        with self._lock:
            fr_id = self._inflight.get((zone, year))
        if fr_id is None:
            return
        try:
            with get_client(sync_client=True) as client:
                client.set_flow_run_state(fr_id, state=Cancelling())
                self._log.warning(
                    "Cancelling still-registered ingest %s-%d (%s) before re-dispatching it", zone, year, fr_id
                )
                deadline = time.monotonic() + CANCELLATION_CONFIRM_S
                while time.monotonic() < deadline and not self._stopping.is_set():
                    run = client.read_flow_run(fr_id)
                    if run.state is not None and run.state.is_final():
                        with self._lock:
                            self._inflight.pop((zone, year), None)
                        self._log.info("Ingest %s-%d (%s) is terminal; the retry may start", zone, year, fr_id)
                        return
                    # `_stopping` doubles as the sleep, so a shutdown wakes this at once
                    # rather than holding the unwind for the rest of the confirmation
                    # window — and the loop condition then ends it, since a retry during
                    # shutdown is not wanted anyway.
                    self._stopping.wait(_INGEST_POLL_S)
        except Exception as exc:
            raise RuntimeError(
                f"could not confirm the previous ingest of {zone}-{year} ({fr_id}) has stopped: {exc}. "
                f"Refusing to start a second one over the same mosaic prefix."
            ) from exc
        raise RuntimeError(
            f"the previous ingest of {zone}-{year} ({fr_id}) did not reach a terminal state within "
            f"{CANCELLATION_CONFIRM_S:.0f}s of being cancelled. Refusing to start a second one over the same "
            f"mosaic prefix — those commits do not rebase, so the loser's failure is terminal."
        )

    def cancel_unstarted(self) -> int:
        """Cancel queued ingests that have not begun, and forget them. Returns how many.

        ``Future.cancel()`` succeeds only for a task still sitting in the pool's queue, so
        a running or finished ingest is untouched — this drops work, never interrupts it.

        Called before the in-child retry pass: every pending cell's ingest is submitted up
        front, so a retry's fresh ``start`` would otherwise sit behind most of a cluster in
        this FIFO queue and block for hours on a still-billing cluster. Those cells are
        unattempted either way and stay pending for the next campaign pass.
        """
        cancelled = 0
        for key, fut in list(self._futures.items()):
            if fut.cancel():
                self._futures.pop(key, None)
                cancelled += 1
        if cancelled:
            self._log.info("Cancelled %d queued ingest(s) that had not started", cancelled)
        return cancelled

    def shutdown(self) -> None:
        """Stop dispatching, cancel in-flight child ingest runs, and WAIT for them to stop.

        Queued futures are cancelled outright; running ones are unblocked by
        ``_stopping`` at their next poll tick. The child flow runs themselves
        would otherwise keep executing server-side after this parent dies —
        request their cancellation so a prompt retry never races an orphaned
        ingest writing the same mosaic prefix.

        **Waiting is the point, for the same reason ``discard`` waits.** Cancellation is a
        REQUEST: Prefect marks the run Cancelling and a worker acts on it later, so
        returning as soon as the request is sent leaves a live writer behind. ``discard``
        can at least refuse its own retry; teardown cannot, because the retry that races
        this child is a WHOLE NEW PARENT RUN — it derives its own child tag from its own
        flow-run id, so it can neither find this child nor be told about it. This wait is
        the only place the two can be kept apart.

        Best effort at the end of it: a child that will not confirm is logged by name and
        left to the orphan sweep, because a teardown that blocks forever is its own
        outage. But it is waited for FIRST, and the log says which ones were not confirmed
        rather than implying all of them stopped.

        The wait also gives up early on a run of read errors, on the same budget the
        poller uses. An unreachable API is the case where confirmation is impossible
        rather than merely slow, and spending the full window re-asking a server that is
        not answering delays a teardown whose whole job is to stop costing money.
        """
        # Snapshot BEFORE waking the pollers: a woken worker deregisters its
        # run id on the way out, and an id that leaves the map un-swept is an
        # un-cancelled child still running server-side.
        with self._lock:
            inflight = dict(self._inflight)
        self._stopping.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if not inflight:
            return
        try:
            with get_client(sync_client=True) as client:
                unconfirmed = dict(inflight)
                for (zone, year), fr_id in inflight.items():
                    try:
                        client.set_flow_run_state(fr_id, state=Cancelling())
                        self._log.warning("Requested cancellation of in-flight ingest %s-%d (%s)", zone, year, fr_id)
                    except Exception:
                        self._log.warning(
                            "Could not cancel in-flight ingest %s-%d (%s) — it may keep running; check the Prefect UI",
                            zone,
                            year,
                            fr_id,
                            exc_info=True,
                        )
                # Then wait for them, all together — the requests are already in, so this
                # costs one child's teardown, not the sum of them.
                deadline = time.monotonic() + CANCELLATION_CONFIRM_S
                # A run of read errors ends the wait early, on the same budget the poller
                # uses. An API that is not answering is the case where confirmation is
                # IMPOSSIBLE rather than slow, and spending the full window re-asking it
                # delays a teardown whose whole job is to stop the run costing money.
                errors = 0
                while unconfirmed and time.monotonic() < deadline and errors < _INGEST_POLL_MAX_ERRORS:
                    for cell, fr_id in list(unconfirmed.items()):
                        try:
                            run = client.read_flow_run(fr_id)
                        except Exception:
                            errors += 1
                            continue
                        errors = 0
                        if run.state is not None and run.state.is_final():
                            del unconfirmed[cell]
                    if unconfirmed:
                        time.sleep(_INGEST_POLL_S)
                for (zone, year), fr_id in unconfirmed.items():
                    self._log.error(
                        "Ingest %s-%d (%s) did not reach a terminal state within %.0fs of being cancelled. "
                        "It may still be writing its mosaic prefix — a retry of this cell is a NEW parent run "
                        "and cannot see this child, so confirm it stopped before re-running the cell.",
                        zone,
                        year,
                        fr_id,
                        CANCELLATION_CONFIRM_S,
                    )
        except Exception:
            self._log.warning("Ingest cancellation sweep failed — check the Prefect UI for orphans", exc_info=True)


#: Tag prefix for this flow's child ingest deployments.
_INGEST_TAG_PREFIX = "chained-ingest"


def _ingest_child_tag(flow_run_id: object) -> str | None:
    """Deterministic tag for this run's child ingest deployments."""
    return child_run_tag(_INGEST_TAG_PREFIX, flow_run_id)


#: The in-process ``inputs.shutdown()`` covers normal failure paths; a cancelled or
#: crashed flow is killed before its ``finally`` runs, and its child ingests — which
#: are dispatched from threads, so carry no parent-run link — keep writing the mosaic
#: prefix a retry will race.
_cancel_child_ingests_on_cancellation = make_child_cancel_hook(_INGEST_TAG_PREFIX, "child ingest run")


@flow(
    name="fill-zones-sequential",
    # Both lists hold the SAME function: a crashed run leaks exactly like a
    # cancelled one. Keep the hook IDEMPOTENT — cancelling a parent and its child
    # together delivers the transition twice and runs it twice (2026-07-25).
    on_cancellation=[ray_cleanup_on_cancellation, _cancel_child_ingests_on_cancellation],
    on_crashed=[ray_cleanup_on_cancellation, _cancel_child_ingests_on_cancellation],
)
def fill_zones_sequential_flow(
    *,
    cells: list[tuple[str, int]],
    paths: BucketPaths,
    ami_ssm_name: str,
    ami_id: str | None = None,
    time_window_end: str | None = None,
    store_name: str = "tessera",
    mask_name: str = "global",
    ssm_prefix: str = "/tessera/ray/",
    cloudwatch_log_group: str = "/ec2/tessera/ray",
    code_bucket: str | None = None,
    code_suffix: str = "",
    staging_code_identity: str | None = None,
    num_actors: int = 20,
    s1_orbit: str = "both",
    require_s1: bool = False,
    s3_region: str | None = None,
    cleanup_staging: bool = True,
    n_assembly_workers: int | None = None,
    allow_partial_window: bool = False,
    allow_s2_only: bool = False,
    allow_model_mismatch: bool = False,
    allow_ingest_code_mismatch: bool = False,
    s3_concurrency: int | None = None,
    launch_pacing: bool = False,
    actor_request_headroom: int | None = None,
    actor_request_batch_size: int | None = None,
    idle_timeout_minutes: int = 10,
    ingest: bool = True,
    ingest_deployment: str = "ingest-zone-year/ingest-zone-year",
    branch: str | None = None,
    look_ahead: int = 2,
    attempts_per_cell_in_cluster: int = 2,
    inference_pause_gate: str | None = None,
    ingest_limit_name: str | None = None,
    cleanup_mosaics: bool = True,
    ingest_settings: IngestSettings = IngestSettings(),  # noqa: B008
    fault_injection: FaultInjection | None = None,
    validation_deployment: str | None = None,
) -> dict[str, Any]:
    """Fill many (zone, year) cells sequentially on a single shared Ray cluster.

    Args:
        cells: ``(zone, year)`` pairs for this cluster — zone as a UTM common name,
            e.g. ``[("33N", 2025), ("15S", 2025)]``. **May span campaign years**, which
            is what lets the driver drop the year barrier. Two properties make that
            safe, and neither asks anything of the caller: each cell carries its OWN
            inference window to the actors (``ZoneContext.time_window``, so a cell is
            never inferred over another year's months), and assemblies serialize on the
            runner's single trailing thread, so two years of one zone commit one after
            another rather than colliding on the group's attrs.
        paths: Deployment storage contract (global store, land mask, mosaics).
        ami_ssm_name: SSM parameter name for the Ray GPU AMI ID (used only when
            ``ami_id`` is not given).
        ami_id: Pre-resolved AMI ID the campaign pinned. Used for BOTH the shared
            cluster's image AND this flow's own staging fingerprint, so a
            mid-campaign re-bake can't split them. ``None`` (direct calls) resolves
            ``ami_ssm_name`` as before.
        time_window_end: End month of the inference window as ``"Month Year"``;
            defaults to ``"December {year}"`` per cell (the calendar-year window).
            Only valid for a SINGLE-year ``cells`` list — one literal override cannot
            describe several years, so a multi-year list with this set is rejected
            rather than silently applying one year's window to all of them.
        store_name: Global-store repo basename (``paths.global_store``).
        mask_name: Coverage-store repo basename (``paths.land_mask_store``).
        ssm_prefix: SSM prefix for the Ray cluster resource IDs.
        cloudwatch_log_group: CloudWatch log group the Ray workers write to.
        code_bucket: S3 bucket workers pull the source tarball from (``None`` =
            AMI-baked source).
        code_suffix: Source-tarball filename suffix (lets branches coexist).
        staging_code_identity: The campaign's code component for the staging
            fingerprint — the narrowed inference-source hash, with its
            ``force_staging_reuse`` / ``force_staging_restage`` overrides already
            applied. Passed by ``run_global_campaign`` so both fill strategies resume
            and restage identically. ``None`` (a direct call) falls back to the
            AMI-plus-tarball artifact identity.
        num_actors: Fleet-size ceiling; each cell requests
            ``min(num_actors, its live tiles)``.
        s1_orbit: ``"ascending"``, ``"descending"``, or ``"both"`` (resolved
            per cell against its mosaic, post-ingest).
        require_s1: Demand radar rather than request it. ``"both"`` normally resolves to
            ``"none"`` for a cell with no SAR store, because parts of the globe are
            radar-free in principle and a global run cannot refuse them. Set this only
            where every cell in the sweep is known to be imaged.
        s3_region: Optional S3 region for the global store + mosaics.
        cleanup_staging: Delete each cell's staged tiles after it lands.
        n_assembly_workers: Override the assembly process-pool size for every cell in this
            run; ``None`` uses ``AssemblyConfig``'s sizing. Applies per cell, not per run —
            assemblies here run one at a time on the trailing thread, so each gets the whole
            pool rather than a share of it.
        allow_partial_window: Relax each cell's temporal-coverage gate to
            "non-empty".
        allow_s2_only: Embed S2-valid pixels with ZERO S1 observations (sub-zone
            SAR coverage gaps) via the upstream v1.1 missing-S1 convention instead
            of skipping them. PER-PIXEL only — zone-level SAR gates stay strict.
            Folded into each cell's staging run_id so a retry across a flipped flag
            never resumes mixed tiles. Quality is unvalidated — see the optional-S1
            ADR. Default False (historical behaviour).
        allow_model_mismatch: Fill even when the seeded store advertises a
            different encoder/checkpoint than this build (default rejects).
        allow_ingest_code_mismatch: Resume a store built by different ingest code (off by default).
        s3_concurrency: Each trailing assembly's slice of the fleet S3-PUT
            budget. ``None`` = the aggregate target halved, leaving headroom
            for the live cell's concurrent staging writes.
        launch_pacing: Pace this cluster's EC2 launch requests against the
            account's shared RunInstances quota. Same shape as ``s3_concurrency``:
            a budget concurrent clusters share, except this one is a request RATE
            and the enforcement lives in the client rather than in a count we
            divide. Default ``False`` keeps today's launch behaviour; the campaign
            turns it on when it runs more than one cluster.
        actor_request_headroom: Hold the actor request to the actor slots this cluster
            has actually placed plus this many, rather than letting it climb toward
            ``num_actors`` on a region that is not placing them
            (:func:`tessera_embeddings.inference.scheduling._batch_actors_to_request`).
            ``None`` keeps today's behaviour;
            :data:`~tessera_embeddings.inference.scheduling.ACTOR_REQUEST_HEADROOM` is
            the value to pass.
        actor_request_batch_size: Actors per batch, overriding the inference
            default. The quota that batches contend for is a rate over CALLS, so
            this is the cheapest available relief: a smaller batch is fewer
            simultaneous launch requests per cluster. ``None`` keeps the default.
        idle_timeout_minutes: Autoscaler idle-down override for the shared
            cluster (template default 2 min suits per-cell fills; the
            inter-zone seam here needs more slack).
        ingest: Produce each cell's mosaics via the ingest deployment, look-ahead
            pipelined with inference (default). False = mosaics exist upstream.
        ingest_deployment: ``flow-name/deployment-name`` of the ingest deployment
            (already branch-resolved by the campaign driver, if any).
        branch: Route the S1/S2 ingest grandchildren dispatched by each cell's
            ``ingest_zone_year`` to their branch-scoped deployments (see
            :func:`run_global_campaign._dpl`). ``None`` (default) = unsuffixed prod
            refs. The direct ``ingest_deployment`` above is resolved by the caller.
        inference_pause_gate: Prefect global concurrency limit READ as a pause flag for
            inference — zero means paused, any positive value means run. It is read, never
            acquired, so it holds no slot and needs no lease. While it reads zero this
            cluster finishes and lands the chunks already queued, then takes on no further
            cell: the actors stay alive and the run does not end, so raising the limit
            resumes where it stopped. It pauses what enters INFERENCE; ``ingest_limit_name``
            at zero pauses what enters ingest, and the two are independent. Neither stops the
            fleet being billed — only cancelling does that. ``None`` disables the check.
        ingest_limit_name: Prefect global concurrency limit bounding how many UTM
            zones ingest simultaneously across ALL clusters. Every zone of this
            cluster is submitted at once and each holds one slot for the duration
            of its ingest, so a cluster starts only as many as fit under the
            fleet-wide cap and the rest queue. ``None`` disables the gate — every
            submitted zone starts immediately, which suits a direct single-cluster
            run and not a campaign.
        attempts_per_cell_in_cluster: Attempts per cell within THIS cluster, including the
            first; 2 means one retry on the still-provisioned fleet. Distinct from
            the campaign driver's `max_dispatch_rounds`, which counts whole-dispatch
            rounds — see `sequential_fill.fill_zones_sequential`.
        look_ahead: Sizes INGEST width only — the ingest driver runs ``1 + look_ahead``
            cells at a time, which also sets the priming abort quorum and the runner's
            retained-failure cap. It no longer bounds in-flight mosaics: peak storage is a
            cluster's mosaics by design (ADR-011).
        cleanup_mosaics: Delete each campaign-ingested mosaic after its cell
            lands (transient input). Ignored for ``ingest=False`` mosaics.
        ingest_settings: Grouped ingest tuning knobs (worker bounds, S2
            coverage threshold, S1 batch window), forwarded verbatim to each
            cell's ingest — see
            :class:`tessera_embeddings.config.ingest.IngestSettings`.
        fault_injection: A supervised failure drill's request to inject one deliberate
            fault into THIS run. Absent by default, and this flow hosts only the
            withholding of supply from a fleet that is already up — anything else, and
            any deployment outside the drill allowlist, is refused before the flow does
            any work (:mod:`tessera_embeddings.config.fault_injection`). A run that
            carries it announces itself as a drill in its own logs.
        validation_deployment: ``flow-name/deployment-name`` of a deployment that
            validates a landed cell. Dispatched from the TRAILING ASSEMBLY THREAD as each
            cell is tagged and never waited on (:mod:`._cell_validation`), so cell N is
            validated while cell N+1 is still inferring. ``None`` (the default)
            validates nothing: the validator is a consumer's flow, so the library names
            none.

    Returns:
        Summary dict: triage counts (retag / empty / live), the sequential
        runner's outcome summary, and elapsed seconds.
    """
    log = get_run_logger()
    t0_flow = time.monotonic()

    # Validate BEFORE any side effect (triage reads, primed ingests, `ray up`):
    # look_ahead < 0 sizes the ingest driver's thread pool (1 + look_ahead) at zero
    # width, so no cell would ever ingest — and it would only wedge AFTER priming
    # has launched ingests and the GPU cluster is up. Fail fast instead.
    if look_ahead < 0:
        raise ValueError(f"look_ahead must be >= 0, got {look_ahead}")
    if attempts_per_cell_in_cluster < 1:
        raise ValueError(
            f"attempts_per_cell_in_cluster must be >= 1, got {attempts_per_cell_in_cluster} "
            "(no cell would be attempted)"
        )
    # Same rule, same reason: on a direct invocation nothing else rejects this until
    # run_inference validates `session_actors`, by which point triage has run, the
    # look-ahead ingests are away and the shared Ray cluster is up.
    if num_actors < 1:
        raise ValueError(f"num_actors must be >= 1, got {num_actors} (no actor would ever run inference)")

    # BEFORE triage, before any ingest is primed and before `ray up`: a refused fault
    # must cost nothing, and an accepted one must be on the record before the run can
    # produce the failure it was armed for. `ssm_prefix` is the run's injected
    # deployment identity — arm() resolves it, so this flow cannot claim an account of
    # its own choosing.
    fault = (
        fault_injection.arm(ssm_prefix=ssm_prefix, supports=(WITHHOLD_WORK,), log=log)
        if fault_injection is not None
        else None
    )

    # Lazily import the AWS providers so the flow file imports on machines
    # without ray/boto installed (arch tests, local inspection).
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials
    from tessera_embeddings.providers.aws.ray import (
        cluster_name_for_flow_run,
        make_instance_terminator,
        ray_cluster,
    )

    # Canonicalize + dedupe, preserving caller order until the size sort. Dedupe on the
    # PAIR: the same zone may legitimately appear for several years, and only an exact
    # repeat is a caller mistake (it would dispatch one cell twice).
    cells = list(dict.fromkeys((canonicalize_zone(z), int(y)) for z, y in cells))
    cell_years = sorted({y for _, y in cells})
    if time_window_end is not None and len(cell_years) > 1:
        raise ValueError(
            f"time_window_end={time_window_end!r} was given with cells spanning years {cell_years}. "
            "One literal window cannot describe several years; drop the override (each cell then "
            "takes its own calendar year) or dispatch one year per flow run."
        )
    store_path = paths.global_store(store_name)
    land_mask_path = paths.land_mask_store(mask_name)
    checkpoint_path = f"{paths.inputs.rstrip('/')}/models/{checkpoint_filename()}"

    @cache
    def _window_for(cell_year: int) -> TimeWindow:
        """The cell's calendar-year window, one per YEAR rather than one per run.

        Cached because a cluster's cells share few distinct years and each cell's
        `prepare` asks for its own; the assertion runs before any ingest dispatch or
        ray_cluster, so an offset/partial window is rejected while it is still free
        rather than after the look-ahead ingests are away and the fleet is up.
        """
        w = parse_time_window(time_window_end or f"December {cell_year}")
        assert_calendar_year_window(w, cell_year)
        return w

    # Validate EVERY year up front, so a bad `time_window_end` or an off-convention year
    # is rejected before any ingest is dispatched rather than at some later cell's prepare.
    for cell_year in cell_years:
        _window_for(cell_year)

    # The STORE is the authority on the depth rule, exactly as in the per-cell flow: it is
    # part of the root's write-once identity, so a fill that took it from a dispatch
    # parameter could write one zone under 25 and its neighbour under 30 with nothing
    # afterwards able to tell them apart. Read ONCE here rather than per cell — every cell
    # in this run writes the same store, and the root cannot change under it.
    #
    # This is the DEFAULT campaign strategy, and it was reading no rule at all: a store
    # seeded with a minimum published pixels below its own advertised line.
    @cache
    def _store_optical_min_obs() -> int | None:
        """The rule, read once and only if a cell actually reaches inference.

        LAZY for the same reason the per-cell flow resolves it inside its preflight: a run
        whose cells all settle in triage — a retag, an empty mark, an all-ocean zone — must
        not touch the store at all, and reading at flow start made every such path require a
        live repository.
        """
        value = _optical_min_obs_from_store(store_path, get_credentials=iam_icechunk_credentials, s3_region=s3_region)
        log.info(
            "Optical depth rule for this fill, from the store root: %s",
            value if value is not None else "no rule",
        )
        return value

    @cache
    def _config_for(resolved_orbit: str, cell_year: int) -> InferenceConfig:
        return build_inference_config(
            s1_orbit=resolved_orbit,
            time_window=_window_for(cell_year),
            checkpoint_path=checkpoint_path,
            inputs_bucket=paths.inputs,
            output_bucket=paths.outputs,
            chunk_size=SHARD_PX,  # 1 inference tile == 1 shard (D3)
            allow_s2_only=allow_s2_only,
            optical_min_obs=_store_optical_min_obs(),
            actor_request_headroom=actor_request_headroom,
            # Omitted when unset so the inference default stands — and keyed on None, not
            # on truthiness, because 0 is a documented mode here rather than an absent value.
            **set_overrides(actor_request_batch_size=actor_request_batch_size),
        )

    # ------------------------------------------------------------------
    # Triage (cheap metadata reads, all before any cluster exists): cells
    # needing only a retag or an empty mark are settled right here — and an
    # all-ocean cell is never ingested at all, unlike the per-cell path. An
    # off-axis year is a configuration error worth failing the whole run for
    # (every cell of this run shares the year).
    # ------------------------------------------------------------------
    retagged: list[dict[str, Any]] = []
    empty: list[dict[str, Any]] = []
    no_optical: list[dict[str, Any]] = []
    live: list[SequentialCell] = []

    # Resolved ONCE here, in the flow's own thread. The trailing assembly thread carries no
    # Prefect context, so reading the runtime id from inside `_validate` would yield None
    # for every cell this cluster fills — and the validations would be untraceable back to
    # the fill that produced them.
    validation_tag = validation_run_tag(flow_run_ctx.id)

    def _validate(zone: str, year: int, summary: dict[str, Any]) -> dict[str, Any]:
        """Hand one landed cell to its validation deployment, without waiting for it.

        Called for every cell as it finishes, on whichever thread finished it — the
        trailing assembly thread for a filled cell, this flow's own thread for a retag or
        an empty mark. Returns the summary unchanged: a landed, tagged cell must not be
        affected by anything that happens to its validation (see :mod:`._cell_validation`).

        The cell's coordinates come from the CALLER rather than out of ``summary``. Every
        caller has them in hand, and reading them back out of the summary would make the
        dispatch depend on the shape of a dict that several fill paths each build
        separately — a cell whose summary spelled them differently would raise here,
        inside the thread that has just landed it.
        """
        dispatch_cell_validation(
            validation_deployment,
            zone=zone,
            year=year,
            summary=summary,
            parameters=cell_validation_parameters(
                zone=zone,
                year=year,
                paths=paths,
                store_name=store_name,
                mask_name=mask_name,
                s3_region=s3_region,
            ),
            log=log,
            tag=validation_tag,
        )
        return summary

    def _fill_no_cluster(zone: str, year: int, run_id: str) -> dict[str, Any]:
        # fill_zone_year's no-Ray paths: repo metadata + coverage bitmap only.
        # s1_orbit is never resolved against a mosaic on these paths, so the
        # unresolved value is fine.
        #
        # Validated like any other cell (`_validate` returns the summary unchanged). Both
        # callers reach a TAGGED cell: a retag-only recovery, whose data landed under some
        # earlier run that may never have been checked, and an all-ocean mark, which
        # `_validate` declines for having nothing embedded.
        return _validate(
            zone,
            year,
            fill_zone_year(
                store_path=store_path,
                zone=zone,
                year=year,
                land_mask_path=land_mask_path,
                mosaic_base=f"{paths.inputs.rstrip('/')}/mosaics/{zone}/{year}",
                staging_base=f"{paths.outputs.rstrip('/')}/staging/{zone}/{year}",
                config=_config_for(s1_orbit, year),
                num_actors=1,
                log=log,
                run_id=run_id,
                get_credentials=iam_icechunk_credentials,
                s3_region=s3_region,
            ),
        )

    for zone, year in cells:
        if zone_year_complete(store_path, zone, year, get_credentials=iam_icechunk_credentials, s3_region=s3_region):
            retagged.append(_fill_no_cluster(zone, year, f"{zone}-{year}-retag"))
            continue
        if not zone_year_on_axis(store_path, zone, year, get_credentials=iam_icechunk_credentials, s3_region=s3_region):
            raise ValueError(
                f"Year {year} is not on zone {zone}'s pre-allocated axis (fixed at seeding, ADR-008 D1) — "
                "reseed the store or drop the year."
            )
        n_tiles = zone_live_tile_count(
            land_mask_path, zone, get_credentials=iam_icechunk_credentials, s3_region=s3_region
        )
        if n_tiles == 0:
            # No mosaic exists or is needed; run_id is provenance-only here
            # (there is no staging to fingerprint and no mosaic to read).
            empty.append(_fill_no_cluster(zone, year, f"{zone}-{year}-empty"))
            continue
        # Optical preflight, only when THIS flow produces the mosaics: a cell whose
        # source confirmably publishes nothing for its live land would otherwise pay
        # for its ingest and a share of the fleet before failing its coverage gate.
        # Only a CONFIRMED_ABSENT refuses — an inconclusive or provisional answer
        # passes the cell through unchanged (see the module docstring's guarantees).
        # With ingest=False the mosaics already exist upstream and are the source of
        # truth; the catalogue can say nothing about them, so no preflight runs.
        if ingest:
            pre = preflight_optical_source(
                zone,
                _window_for(year),
                land_mask_path=land_mask_path,
                get_credentials=iam_icechunk_credentials,
                s3_region=s3_region,
                log=log,
            )
            if pre.finding is SourceFinding.CONFIRMED_ABSENT:
                log.warning(
                    "Optical preflight refused %s-%d after %d probe(s): %s — the cell is left "
                    "unfilled (no ingest, no fleet share); it will keep appearing in the campaign's "
                    "unfilled report until the source publishes it or it is dropped from the ask.",
                    zone,
                    year,
                    pre.probes,
                    pre.reason,
                )
                no_optical.append({"zone": zone, "year": year, "reason": pre.reason})
                continue
        live.append(SequentialCell(zone=zone, year=year, num_actors=min(num_actors, n_tiles), n_tiles=n_tiles))

    # DENSEST FIRST, on the unclamped tile count. Two things depend on it: the
    # shared fleet only ever shrinks across the run (no mid-campaign worker
    # relaunch, island zones at the natural taper), and — since the cluster starts
    # inferring as soon as its FIRST zone lands — that first zone must be big enough
    # to keep the fleet busy while the rest of the window ingests behind it.
    #
    # NOT `num_actors`, which is `min(num_actors, n_tiles)`: every zone bigger than
    # the fleet clamps to the same value, so sorting on it left the whole dense end
    # of the list in arbitrary order — precisely the part this ordering is for.
    live.sort(key=lambda c: c.n_tiles, reverse=True)
    log.info(
        "Triage for year(s) %s: %d retagged, %d empty, %d live cell(s), %d refused (no optical) — order: %s",
        cell_years,
        len(retagged),
        len(empty),
        len(live),
        len(no_optical),
        [f"{c.zone}-{c.year}" for c in live],
    )

    summary: dict[str, Any] = {
        "years": cell_years,
        "cells": [[c.zone, c.year] for c in live],
        "retagged": len(retagged),
        "empty": len(empty),
        "live": len(live),
        "no_optical": [[c["zone"], c["year"]] for c in no_optical],
    }
    if not live:
        log.info("No live cells — no Ray cluster needed")
        return {**summary, "sequential": None}

    # Fail loudly BEFORE provisioning Ray if the store was seeded for a
    # different encoder/checkpoint than this build embeds with.
    _assert_seeded_model_matches(
        store_path,
        build_checkpoint=checkpoint_filename(),
        allow_model_mismatch=allow_model_mismatch,
        get_credentials=iam_icechunk_credentials,
        s3_region=s3_region,
    )

    _ingest_params = partial(
        _ingest_dispatch_params,
        paths=paths,
        mask_name=mask_name,
        s1_orbit=s1_orbit,
        ingest_settings=ingest_settings,
        allow_partial_window=allow_partial_window,
        allow_ingest_code_mismatch=allow_ingest_code_mismatch,
        s3_region=s3_region,
        branch=branch,
    )

    inputs = (
        _DeploymentCellInputs(
            deployment=ingest_deployment,
            params_for=_ingest_params,
            inputs_bucket=paths.inputs,
            cleanup_mosaics=cleanup_mosaics,
            # This cluster's SHARE of the fleet-wide ingest cap, so the clusters
            # divide it evenly by construction rather than racing for slots on the
            # global gate. The gate (`ingest_limit_name`) is still the hard ceiling —
            # this only decides how many a single cluster ever asks for at once.
            # 1 + look_ahead, matching the cells the flow PRIMES. Sized at look_ahead
            # alone, the opening window's sibling ingests sat queued behind the head
            # cell rather than running — so "start on whichever mosaic lands first"
            # could only ever land the head one, and the fleet waited on it even when
            # it was the slowest zone in the window.
            max_parallel=1 + look_ahead,
            log=log,
            # Parent-derived tag so the cancellation/crash hook can sweep live
            # child ingests from a fresh process (see _ingest_child_tag). None
            # outside a Prefect run (unit tests) — children just go untagged.
            child_tag=_ingest_child_tag(flow_run_ctx.id) if flow_run_ctx.id else None,
            ingest_limit_name=ingest_limit_name,
        )
        if ingest
        else None
    )

    # Resolve the immutable code artifact (AMI ID + optional tarball ETag) ONCE for
    # the whole run — it's a run-wide constant every cell's staging run_id folds in.
    # NOT the mutable `code_suffix` label (a re-baked AMI or overwritten tarball
    # must start fresh staging prefixes), and resolved in the Ray provisioning
    # region (None → us-west-2), not the storage `s3_region`. Placed after the
    # no-live-cells early return so triage-only runs make no AWS call.
    # Pin the AMI component to the same id the cluster boots, so this flow's own
    # staging fingerprint and its provisioned image can't disagree. Called direct
    # (not via run-global-campaign) ami_id is None, and leaving it None would read the
    # SSM pointer twice — here and again in ray_cluster — so a re-bake landing between
    # them boots a different image than the prefix was fingerprinted against.
    ami_id = ami_id or _resolve_ami_id(ami_ssm_name, None)
    # The campaign's staging identity when it supplied one, and only otherwise the
    # AMI-plus-tarball artifact identity. This is the DEFAULT fill strategy, so
    # recomputing the artifact identity here undid the narrowing for the path that
    # actually runs: an AMI re-bake or any tarball change abandoned every staged tile,
    # and `force_staging_reuse` / `force_staging_restage` — parameters the campaign
    # documents — reached the per-zone path only and did nothing here. The fallback is
    # for a direct call, which has no parent to inherit from.
    code_identity = staging_code_identity or _resolve_code_identity(
        ami_ssm_name, code_bucket, code_suffix, None, ami_id
    )

    def _prepare(cell: SequentialCell) -> PreparedCell:
        # Everything here needs the cell's mosaic, so it runs only after the
        # cell's ingest (if any) has landed: orbit resolution probes the
        # mosaic, the coverage gate reads its months, and the run_id
        # fingerprints its ingest marker (identical inputs → resume staging;
        # any change → fresh prefix).
        mosaic_base = f"{paths.inputs.rstrip('/')}/mosaics/{cell.zone}/{cell.year}"
        resolved = resolve_s1_orbit(
            mosaic_base,
            s1_orbit,
            allow_none=not require_s1,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )
        # KEPT, not just gated on. The summary names the months and dates the mosaics
        # actually held; the mosaics are deleted once the cell lands, so if it is dropped
        # here nothing downstream can ever recover it — and a cell filled with
        # ``allow_partial_window`` would publish with no record of what was missing.
        coverage = check_time_window_coverage(
            mosaic_base,
            _window_for(cell.year),
            s1_orbit=resolved,
            skip_coverage_check=allow_partial_window,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )
        return PreparedCell(
            mosaic_base=mosaic_base,
            staging_base=f"{paths.outputs.rstrip('/')}/staging/{cell.zone}/{cell.year}",
            run_id=_staging_run_id(
                cell.zone,
                cell.year,
                inputs_bucket=paths.inputs,
                min_valid_coverage=ingest_settings.min_valid_coverage,
                s1_orbit=s1_orbit,
                allow_partial_window=allow_partial_window,
                allow_s2_only=allow_s2_only,
                optical_min_obs=_store_optical_min_obs(),
                code_identity=code_identity,
                get_credentials=iam_icechunk_credentials,
                s3_region=s3_region,
            ),
            config=_config_for(resolved, cell.year),
            input_coverage=coverage,
        )

    # One assembly at a time trails the live cell's staging writes, so split
    # the fleet PUT budget between them rather than letting the pair burst
    # ~2x the target (the parallel driver divides by max_parallel_zones for
    # the same reason).
    if s3_concurrency is None:
        from tessera_embeddings.inference.assembly import TARGET_AGGREGATE_S3_CONCURRENCY

        s3_concurrency = max(1, TARGET_AGGREGATE_S3_CONCURRENCY // 2)

    # on_actor_retire fires only from idle retirement, which the scheduler
    # suppresses while the zone stream is unexhausted — so this terminator is
    # inert mid-stream and only drains the fleet early during the true cluster
    # tail. A dead actor's abandoned instance is reclaimed by the autoscaler
    # idle timeout instead (idle_timeout_minutes, below).
    terminator = make_instance_terminator(log=log)

    def _plan(cell: SequentialCell, prep: PreparedCell) -> ZonePlan:
        return plan_zone_inference(
            store_path=store_path,
            zone=cell.zone,
            year=cell.year,
            land_mask_path=land_mask_path,
            mosaic_base=prep.mosaic_base,
            config=prep.config,
            log=log,
            run_id=prep.run_id,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )

    # The shared stream session: ONE set of actors for EVERY zone in the cluster,
    # whatever orbit each resolves, fed via the runner's more_work source. The
    # placeholder path args are never used — the initial chunk list is empty and
    # every streamed item carries its own ZoneContext.
    #
    # The session orbit is simply the REQUEST (default "both"), and it does not
    # constrain which cells can join: the resolved orbit travels on each cell's
    # ZoneContext, exactly as its time window does, so one actor reads each cell
    # under that cell's own orbit. A radar-free cell resolving `none` is therefore
    # ordinary work rather than something to route around — which matters, because
    # radar-free zone-years are a predictable population, not an anomaly.
    # The densest live cell's year, NOT a leaked triage loop variable. This config sizes
    # the shared session and builds its actors; its window is only a DEFAULT now, because
    # every work item carries its own cell's window (ZoneContext.time_window) and the
    # actors prefer that. Pinning it to live[0] keeps it deterministic and matches the cell
    # the session is sized for.
    session_config = _config_for(s1_orbit, live[0].year)

    # Size the shared session by the LARGEST cell's clamped request, not the raw
    # fleet ceiling: the session provisions its first actor batch before any
    # streamed work arrives, so a small cluster (e.g. a one-tile zone list) would
    # otherwise request the full default fleet for work that can never use it.
    # live is sorted num_actors-descending and each cell is already clamped to
    # min(num_actors, its live tiles), so live[0] IS the run's true ceiling.
    session_actors = live[0].num_actors

    def _session(
        more_work: Callable[[], list[WorkItem] | None],
        on_item_done: Callable[[WorkItem, dict[str, Any]], None],
    ) -> list[dict[str, Any]]:
        return run_inference(
            session_actors,
            session_config,
            [],
            "chained-session",
            "chained-session",
            "chained-session",
            t0_flow,
            log,
            on_actor_retire=terminator,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
            retire_idle_actors=True,  # the scheduler holds off until the source is exhausted
            more_work=more_work,
            on_item_done=on_item_done,
        )

    def _infer_single(cell: SequentialCell, prep: PreparedCell, final: bool) -> ZoneFillHandoff:
        # A per-cell session with the cell's own config, on the still-provisioned
        # cluster, for the cells the shared stream did not finish — a crashed
        # session's survivors and the retry pass.
        return infer_zone_year(
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
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
            on_actor_retire=terminator,
            retire_idle_actors=final,
        )

    def _assemble(handoff: ZoneFillHandoff, prep: PreparedCell) -> dict[str, Any]:
        # The validation dispatch rides HERE, on the trailing assembly thread, immediately
        # after the cell's tag is created — not at the end of the flow. The runner calls
        # this once per cell while the next cell is already inferring, so a per-cell
        # dispatch inherits that pipelining for free: cell N is being validated while cell
        # N+1 works. Batching them at the end of the cluster's zone list would instead put
        # every cell's verdict hours after the defect it would have caught.
        return _validate(
            handoff.zone,
            handoff.year,
            assemble_zone_year(
                handoff,
                store_path=store_path,
                staging_base=prep.staging_base,
                # The published registry's ROOT, derived from the STORE THIS RUN WRITES. Deriving
                # it from the default repo name instead put this flow's parts beside
                # `tessera.icechunk` while it filled `tessera-radar` (2026-08-19). Bucket
                # conventions belong to BucketPaths (§0.3); the part's name is composed downstream,
                # where the run id is.
                registry_root=paths.optical_registry(store_path),
                # The depth rule every registry row is judged against, from the store root — the
                # same cached read `_config_for` gates inference on, so the row states the line the
                # fill actually applied. This call site is a SECOND entry into
                # `assemble_zone_year` (`fill_zone_year` is the other), and omitting it published
                # rows whose `obs_max` and `median_obs_where_any` had no line to be measured from,
                # with `None` reading as "this fill applied no depth rule".
                optical_min_obs=_store_optical_min_obs(),
                input_coverage=prep.input_coverage,
                log=log,
                s3_concurrency=s3_concurrency,
                cleanup_staging=cleanup_staging,
                n_assembly_workers=n_assembly_workers,
                get_credentials=iam_icechunk_credentials,
                s3_region=s3_region,
            ),
        )

    # START THE INGEST WINDOW, THEN WAIT FOR THE FIRST MOSAIC TO LAND — ANY OF THEM.
    #
    # A GPU fleet is never booted speculatively: `ray up` is not requested until a
    # real mosaic exists. But WHICH mosaic decides how long the fleet waits, and
    # waiting on a named cell means waiting on the DENSEST zone in the window,
    # because that is what `live[0]` is. The densest zone is also the slowest to
    # ingest. On the real coverage counts a cluster's opening window spans about
    # 4 h to 10 h of ingest, so blocking on the head idles the fleet for ~6 h with
    # finished mosaics already on disk — and the wider the fleet, the larger that
    # is as a share of the run.
    #
    # So: start the whole window, then take whichever lands first. The feeder does
    # the same thing thereafter (sequential_fill._take_next), which is what keeps
    # the saving instead of handing it back at the next cell.
    #
    # The densest-first ORDER is unchanged and still doing its two jobs: it sizes
    # the session from the largest cell, and it puts the island tail last. What it
    # is no longer is a barrier.
    # INSIDE the shutdown guard, deliberately. `start` submits every cell in the window
    # at once, so from the first submission onward there are child ingest runs that only
    # `shutdown()` can cancel. Priming above the `try` meant any failure between the first
    # submission and the `with` — a raising `start`, an unexpected error in the wait —
    # returned without cancelling them, leaving children writing to mosaic prefixes
    # server-side that a prompt retry of this flow would then race.
    try:
        if inputs is not None:
            # EVERY live cell, not a look-ahead window. The driver's `max_parallel` is what
            # bounds concurrency; starting only a window meant a new ingest could not begin
            # until the feeder admitted a cell, which waited on an assembly — so a cluster
            # configured for 6 concurrent ingests ran 1.
            ingest_window = live
            for cell in ingest_window:
                inputs.start(cell.zone, cell.year)

            log.info(
                "Ingesting %d cell(s), %d at a time; GPUs are requested as soon as the first "
                "mosaic lands (densest first: %s tiles)",
                len(ingest_window),
                1 + look_ahead,
                ", ".join(f"{c.n_tiles:,}" for c in ingest_window[:6]),
            )
            t0 = time.monotonic()
            first = inputs.wait_first([(c.zone, c.year) for c in ingest_window])
            if first is None:  # nothing was started (no adapter cells) — nothing to wait for
                log.warning("No ingest was started for this cluster; requesting GPUs without waiting")
            else:
                landed = next(c for c in ingest_window if (c.zone, c.year) == first)
                log.info(
                    "Mosaic for %s-%d (%s tiles) ready after %.1fs — requesting GPUs; %d zone(s) still ingesting",
                    landed.zone,
                    landed.year,
                    f"{landed.n_tiles:,}",
                    time.monotonic() - t0,
                    len(ingest_window) - 1,
                )

        # Pin a deterministic cluster name from the flow-run id so the cancellation
        # hook can re-derive it and terminate the fleet by tag even in a fresh module
        # import (globals unset) — a cancel before activate() records the name must
        # not hit the "no cluster name" path and leak the shared GPU fleet.
        with ray_cluster(
            log,
            ami_ssm_name=ami_ssm_name,
            ami_id=ami_id,
            ssm_prefix=ssm_prefix,
            cloudwatch_log_group=cloudwatch_log_group,
            code_bucket=code_bucket,
            code_suffix=code_suffix,
            idle_timeout_minutes=idle_timeout_minutes,
            cluster_name=cluster_name_for_flow_run(flow_run_ctx.id),
            launch_pacing=launch_pacing,
        ) as resolved_yaml:
            activate(resolved_yaml)
            seq = fill_zones_sequential(
                cells=live,
                prepare=_prepare,
                plan=_plan,
                session=_session,
                assemble=_assemble,
                infer_single=_infer_single,
                session_s1_orbit=s1_orbit,
                log=log,
                inputs=inputs,
                look_ahead=look_ahead,
                attempts_per_cell_in_cluster=attempts_per_cell_in_cluster,
                fault=fault,
                # Built here rather than in the runner: the runner is Prefect-free, so the
                # question "is inference paused" reaches it as a plain callable.
                paused=(pause_signal(inference_pause_gate, log=log) if inference_pause_gate else None),
            )
    finally:
        if inputs is not None:
            inputs.shutdown()
        # The context manager has already torn the cluster down (or the hook
        # will, on cancellation) — clear the hook state even when the runner
        # raises (its partial-failure RuntimeError is a NORMAL exit path).
        deactivate()

    log.info(
        "Year %d sequential fill: %d/%d live cells landed (plus %d retagged, %d empty)",
        year,
        seq["succeeded"],
        seq["cells"],
        len(retagged),
        len(empty),
    )
    return {**summary, "sequential": seq}
