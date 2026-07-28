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
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

from prefect import flow, get_run_logger
from prefect.client.orchestration import get_client
from prefect.deployments import run_deployment
from prefect.runtime import flow_run as flow_run_ctx
from prefect.states import Cancelling

from tessera_embeddings.config.inference import checkpoint_filename
from tessera_embeddings.config.ingest import IngestSettings
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.data_loading import check_time_window_coverage, resolve_s1_orbit
from tessera_embeddings.inference.orchestration_helpers import build_inference_config
from tessera_embeddings.inference.runner import run_inference
from tessera_embeddings.inference.scheduling import WorkItem
from tessera_embeddings.orchestration.prefect.flows._child_runs import child_run_tag, make_child_cancel_hook
from tessera_embeddings.orchestration.prefect.flows._ray_lifecycle import (
    activate,
    deactivate,
    ray_cleanup_on_cancellation,
)
from tessera_embeddings.orchestration.prefect.flows.fill_zone_year import (
    _assert_seeded_model_matches,
    _PrefectCommitGate,
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


class _DeploymentCellInputs:
    """Ingest-deployment adapter satisfying the runner's ``CellInputs`` protocol.

    ``start`` submits a worker thread that creates the ingest flow run
    (``run_deployment(timeout=0)``, returning immediately with the run's id)
    and then POLLS it to a terminal state, so a cell's ``wait`` is a future
    join and — crucially — the child run's id is known while it executes:
    ``shutdown`` can request cancellation of every in-flight ingest instead of
    orphaning children that keep writing after the parent has failed (a quick
    retry would then race the orphan on the same mosaic prefix). The executor
    is sized to ``look_ahead`` so concurrent ingest Dask clusters stay bounded
    exactly like the parallel driver's ``max_parallel_ingest``. ``cleanup``
    deletes only mosaics THIS adapter produced (a cell never started here is
    someone else's input), matching the parallel driver's ``did_ingest and
    cleanup_mosaics`` rule.

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
    ) -> None:
        self._deployment = deployment
        self._params_for = params_for
        self._inputs_bucket = inputs_bucket
        self._cleanup_mosaics = cleanup_mosaics
        self._log = log
        # Deterministic tag stamped on every child run, derived from the PARENT
        # flow-run id — the cancellation/crash hook re-derives it in a fresh
        # process and sweeps live children by tag (the in-process shutdown()
        # never runs when Prefect kills the flow process). Mirrors the Ray
        # terminate-instances-by-tag pattern.
        self._child_tag = child_tag
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_parallel), thread_name_prefix="cell-ingest")
        self._futures: dict[tuple[str, int], Future[None]] = {}
        self._lock = threading.Lock()
        self._inflight: dict[tuple[str, int], Any] = {}  # (zone, year) → flow-run id
        self._stopping = threading.Event()

    def _run(self, zone: str, year: int) -> None:
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

    def cleanup(self, zone: str, year: int) -> None:
        if (zone, year) not in self._futures or not self._cleanup_mosaics:
            return
        delete_prefix(f"{self._inputs_bucket.rstrip('/')}/mosaics/{zone}/{year}", log=self._log)

    def shutdown(self) -> None:
        """Stop dispatching AND cancel in-flight child ingest runs (best effort).

        Queued futures are cancelled outright; running ones are unblocked by
        ``_stopping`` at their next poll tick. The child flow runs themselves
        would otherwise keep executing server-side after this parent dies —
        request their cancellation so a prompt retry never races an orphaned
        ingest writing the same mosaic prefix.
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
    zones: list[str],
    year: int,
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
    num_actors: int = 20,
    s1_orbit: str = "both",
    s3_region: str | None = None,
    commit_limit_name: str | None = None,
    cleanup_staging: bool = True,
    allow_partial_window: bool = False,
    allow_s2_only: bool = False,
    allow_model_mismatch: bool = False,
    s3_concurrency: int | None = None,
    idle_timeout_minutes: int = 10,
    ingest: bool = True,
    ingest_deployment: str = "ingest-zone-year/ingest-zone-year",
    branch: str | None = None,
    look_ahead: int = 2,
    cleanup_mosaics: bool = True,
    ingest_settings: IngestSettings = IngestSettings(),  # noqa: B008
) -> dict[str, Any]:
    """Fill one year's zones sequentially on a single shared Ray cluster.

    Args:
        zones: Zone group names for THIS year (UTM common names, e.g.
            ``["33N", "15S"]``). One year per flow run — the driver stays
            year-serial, and single-year cells are all distinct zone groups,
            which is what makes the runner's trailing assembly conflict-free.
        year: Campaign calendar year to fill (must be on the seeded axis).
        paths: Deployment storage contract (global store, land mask, mosaics).
        ami_ssm_name: SSM parameter name for the Ray GPU AMI ID (used only when
            ``ami_id`` is not given).
        ami_id: Pre-resolved AMI ID the campaign pinned. Used for BOTH the shared
            cluster's image AND this flow's own staging fingerprint, so a
            mid-campaign re-bake can't split them. ``None`` (direct calls) resolves
            ``ami_ssm_name`` as before.
        time_window_end: End month of the inference window as ``"Month Year"``;
            defaults to ``"December {year}"`` (the calendar-year window).
        store_name: Global-store repo basename (``paths.global_store``).
        mask_name: Coverage-store repo basename (``paths.land_mask_store``).
        ssm_prefix: SSM prefix for the Ray cluster resource IDs.
        cloudwatch_log_group: CloudWatch log group the Ray workers write to.
        code_bucket: S3 bucket workers pull the source tarball from (``None`` =
            AMI-baked source).
        code_suffix: Source-tarball filename suffix (lets branches coexist).
        num_actors: Fleet-size ceiling; each cell requests
            ``min(num_actors, its live tiles)``.
        s1_orbit: ``"ascending"``, ``"descending"``, or ``"both"`` (resolved
            per cell against its mosaic, post-ingest).
        s3_region: Optional S3 region for the global store + mosaics.
        commit_limit_name: Prefect global concurrency limit bounding fleet-wide
            simultaneous committers (D6). ``None`` = ungated.
        cleanup_staging: Delete each cell's staged tiles after it lands.
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
        s3_concurrency: Each trailing assembly's slice of the fleet S3-PUT
            budget. ``None`` = the aggregate target halved, leaving headroom
            for the live cell's concurrent staging writes.
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
        look_ahead: Cells beyond the current one kept in ingest flight (bounds
            concurrent ingest Dask clusters AND in-flight mosaics, ADR-011).
        cleanup_mosaics: Delete each campaign-ingested mosaic after its cell
            lands (transient input). Ignored for ``ingest=False`` mosaics.
        ingest_settings: Grouped ingest tuning knobs (worker bounds, S2
            coverage threshold, S1 batch window), forwarded verbatim to each
            cell's ingest — see
            :class:`tessera_embeddings.config.ingest.IngestSettings`.

    Returns:
        Summary dict: triage counts (retag / empty / live), the sequential
        runner's outcome summary, and elapsed seconds.
    """
    log = get_run_logger()
    t0_flow = time.monotonic()

    # Validate BEFORE any side effect (triage reads, primed ingests, `ray up`):
    # look_ahead < 0 sizes the mosaic budget / zone_slots semaphore at zero
    # capacity, deadlocking the feeder — and it would only wedge AFTER priming
    # has launched ingests and the GPU cluster is up. Fail fast instead.
    if look_ahead < 0:
        raise ValueError(f"look_ahead must be >= 0, got {look_ahead}")
    # Same rule, same reason: on a direct invocation nothing else rejects this until
    # run_inference validates `session_actors`, by which point triage has run, the
    # look-ahead ingests are away and the shared Ray cluster is up.
    if num_actors < 1:
        raise ValueError(f"num_actors must be >= 1, got {num_actors} (no actor would ever run inference)")

    # Lazily import the AWS providers so the flow file imports on machines
    # without ray/boto installed (arch tests, local inspection).
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials
    from tessera_embeddings.providers.aws.ray import (
        cluster_name_for_flow_run,
        make_instance_terminator,
        ray_cluster,
    )

    # Canonicalize + dedupe, preserving caller order until the size sort.
    zones = list(dict.fromkeys(canonicalize_zone(z) for z in zones))
    store_path = paths.global_store(store_name)
    land_mask_path = paths.land_mask_store(mask_name)
    window = parse_time_window(time_window_end or f"December {year}")
    # Before any ingest dispatch or ray_cluster: planning would reject an
    # offset/partial window, but only after look-ahead ingests have started and
    # the shared fleet is up.
    assert_calendar_year_window(window, year)
    checkpoint_path = f"{paths.inputs.rstrip('/')}/models/{checkpoint_filename()}"
    gate = _PrefectCommitGate(commit_limit_name) if commit_limit_name else None

    def _config_for(resolved_orbit: str) -> InferenceConfig:
        return build_inference_config(
            s1_orbit=resolved_orbit,
            time_window=window,
            checkpoint_path=checkpoint_path,
            inputs_bucket=paths.inputs,
            output_bucket=paths.outputs,
            chunk_size=SHARD_PX,  # 1 inference tile == 1 shard (D3)
            allow_s2_only=allow_s2_only,
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
    live: list[SequentialCell] = []

    def _fill_no_cluster(zone: str, run_id: str) -> dict[str, Any]:
        # fill_zone_year's no-Ray paths: repo metadata + coverage bitmap only.
        # s1_orbit is never resolved against a mosaic on these paths, so the
        # unresolved value is fine.
        return fill_zone_year(
            store_path=store_path,
            zone=zone,
            year=year,
            land_mask_path=land_mask_path,
            mosaic_base=f"{paths.inputs.rstrip('/')}/mosaics/{zone}/{year}",
            staging_base=f"{paths.outputs.rstrip('/')}/staging/{zone}/{year}",
            config=_config_for(s1_orbit),
            num_actors=1,
            log=log,
            run_id=run_id,
            gate=gate,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )

    for zone in zones:
        if zone_year_complete(store_path, zone, year, get_credentials=iam_icechunk_credentials, s3_region=s3_region):
            retagged.append(_fill_no_cluster(zone, f"{zone}-{year}-retag"))
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
            empty.append(_fill_no_cluster(zone, f"{zone}-{year}-empty"))
            continue
        live.append(SequentialCell(zone=zone, year=year, num_actors=min(num_actors, n_tiles)))

    # Largest-first: the shared fleet only ever shrinks across the run, so no
    # mid-campaign worker relaunch; island zones land at the natural taper.
    live.sort(key=lambda c: c.num_actors, reverse=True)
    log.info(
        "Triage for year %d: %d retagged, %d empty, %d live cell(s) — order: %s",
        year,
        len(retagged),
        len(empty),
        len(live),
        [c.zone for c in live],
    )

    summary: dict[str, Any] = {
        "year": year,
        "zones": zones,
        "retagged": len(retagged),
        "empty": len(empty),
        "live": len(live),
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
        s3_region=s3_region,
        branch=branch,
    )

    inputs = (
        _DeploymentCellInputs(
            deployment=ingest_deployment,
            params_for=_ingest_params,
            inputs_bucket=paths.inputs,
            cleanup_mosaics=cleanup_mosaics,
            max_parallel=look_ahead,
            log=log,
            # Parent-derived tag so the cancellation/crash hook can sweep live
            # child ingests from a fresh process (see _ingest_child_tag). None
            # outside a Prefect run (unit tests) — children just go untagged.
            child_tag=_ingest_child_tag(flow_run_ctx.id) if flow_run_ctx.id else None,
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
    code_identity = _resolve_code_identity(ami_ssm_name, code_bucket, code_suffix, None, ami_id)

    def _prepare(cell: SequentialCell) -> PreparedCell:
        # Everything here needs the cell's mosaic, so it runs only after the
        # cell's ingest (if any) has landed: orbit resolution probes the
        # mosaic, the coverage gate reads its months, and the run_id
        # fingerprints its ingest marker (identical inputs → resume staging;
        # any change → fresh prefix).
        mosaic_base = f"{paths.inputs.rstrip('/')}/mosaics/{cell.zone}/{cell.year}"
        resolved = resolve_s1_orbit(
            mosaic_base, s1_orbit, get_credentials=iam_icechunk_credentials, s3_region=s3_region
        )
        check_time_window_coverage(
            mosaic_base,
            window,
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
                code_identity=code_identity,
                get_credentials=iam_icechunk_credentials,
                s3_region=s3_region,
            ),
            config=_config_for(resolved),
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
    # inert mid-stream and only drains the fleet early during the true shard
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
            gate=gate,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )

    # The shared stream session: ONE set of actors for every same-orbit zone,
    # fed via the runner's more_work source. Its config carries the session
    # orbit; zones resolving a different orbit are deferred by the runner to
    # _infer_single below. The placeholder path args are never used — the
    # initial chunk list is empty and every streamed item carries its own
    # ZoneContext.
    #
    # The session orbit is simply the REQUEST (default "both"). A whole UTM zone
    # is anticipated to always carry BOTH S1 orbits — single/no-orbit is a
    # sub-zone and pixel-level reality (handled per-pixel in the mosaic and by
    # `allow_s2_only`), NOT a whole-zone one — so `resolve_s1_orbit` returns
    # "both" for every cell and none ever mismatches a "both" session. Resolving
    # the session orbit from the cells' data would only matter for a
    # single-orbit whole zone, which does not occur; the orbit-mismatch
    # deferral + fallback below remains as a safety net for an explicit
    # single-orbit request (or that non-scenario), bounded by the deferral cap.
    session_config = _config_for(s1_orbit)

    # Size the shared session by the LARGEST cell's clamped request, not the raw
    # fleet ceiling: the session provisions its first actor batch before any
    # streamed work arrives, so a small shard (e.g. a one-tile zone list) would
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
        # Orbit-mismatch fallback: a per-cell session with the cell's own
        # config, after the shared stream ends.
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
            gate=gate,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
            on_actor_retire=terminator,
            retire_idle_actors=final,
        )

    def _assemble(handoff: ZoneFillHandoff, prep: PreparedCell) -> dict[str, Any]:
        return assemble_zone_year(
            handoff,
            store_path=store_path,
            staging_base=prep.staging_base,
            log=log,
            gate=gate,
            s3_concurrency=s3_concurrency,
            cleanup_staging=cleanup_staging,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )

    # Prime the pipeline: kick off the initial ingest window BEFORE `ray up`, so
    # the first cell's mosaic materializes during cluster bring-up instead of a
    # freshly-provisioned GPU fleet idling through a full ingest at the start of
    # every year. Idempotent — the runner's feeder immediately re-issues these
    # same starts (and admits them through its mosaic budget); the window is
    # 1 + look_ahead ≤ the budget's look_ahead + 2, so priming can't overshoot it.
    if inputs is not None:
        for cell in live[: 1 + look_ahead]:
            inputs.start(cell.zone, cell.year)

    try:
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
