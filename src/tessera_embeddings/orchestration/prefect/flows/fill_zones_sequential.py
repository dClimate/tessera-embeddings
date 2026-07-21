"""Fill many zones of one campaign year through a SINGLE shared Ray cluster.

The parallel campaign path (:mod:`.run_global_campaign` dispatching
:mod:`.fill_zone_year` per cell) provisions a Ray cluster per (zone, year) —
each one re-paying ``ray up`` (~5-10 min wall), per-worker EC2 bringup
(minutes of billed GPU idle each), the per-worker model-load cold start, and
a fresh roll of the EC2 capacity dice. This flow amortizes all of that across
a whole year's zones: ONE cluster, zones filled **strictly sequentially**.
How the shared fleet is kept busy at the seams (ingest look-ahead, trailing
assembly, fleet retention) is the sequential runner's story — see
:mod:`tessera_embeddings.orchestration.runners.sequential_fill`; this flow
supplies the Prefect-facing pieces (deployment-backed ingest, fingerprinted
run_ids, per-cell config) and the cluster itself, with its autoscaler
``idle_timeout_minutes`` raised so the short inter-zone seam never sheds
workers.

Cheap cells never touch the cluster: already-complete cells are retagged and
all-ocean cells are marked complete-empty *before* Ray is provisioned — and,
unlike the per-cell path, an all-ocean cell is never ingested at all. Live
cells are ordered largest-first (clamped ``min(num_actors, live tiles)``
requests) so the autoscaled fleet only ever shrinks across the run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from prefect import flow, get_run_logger
from prefect.deployments import run_deployment

from tessera_embeddings.config.inference import checkpoint_filename
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.data_loading import check_time_window_coverage, resolve_s1_orbit
from tessera_embeddings.inference.orchestration_helpers import build_inference_config
from tessera_embeddings.orchestration.prefect.flows._ray_lifecycle import (
    activate,
    deactivate,
    ray_cleanup_on_cancellation,
)
from tessera_embeddings.orchestration.prefect.flows.fill_zone_year import (
    _assert_seeded_model_matches,
    _PrefectCommitGate,
)
from tessera_embeddings.orchestration.prefect.flows.ingest_zone_year import DEFAULT_MIN_VALID_COVERAGE
from tessera_embeddings.orchestration.prefect.flows.run_global_campaign import _staging_run_id
from tessera_embeddings.orchestration.prefect.flows.tessera_full_pipeline import _check_completed
from tessera_embeddings.orchestration.runners.sequential_fill import (
    PreparedCell,
    SequentialCell,
    fill_zones_sequential,
)
from tessera_embeddings.orchestration.runners.zone_fill import (
    assemble_zone_year,
    fill_zone_year,
    infer_zone_year,
    zone_live_tile_count,
    zone_year_complete,
    zone_year_on_axis,
)
from tessera_embeddings.storage.object_store import delete_prefix
from tessera_embeddings.storage.zone_grid import canonicalize_zone

if TYPE_CHECKING:
    from tessera_embeddings.config.inference import InferenceConfig
    from tessera_embeddings.orchestration.runners.zone_fill import ZoneFillHandoff


class _DeploymentCellInputs:
    """Ingest-deployment adapter satisfying the runner's ``CellInputs`` protocol.

    ``start`` submits a worker thread that dispatches the ingest deployment and
    blocks until that flow run finishes, so a cell's ``wait`` is just a future
    join; the executor is sized to ``look_ahead`` so concurrent ingest Dask
    clusters stay bounded exactly like the parallel driver's
    ``max_parallel_ingest``. ``cleanup`` deletes only mosaics THIS adapter
    produced (a cell never started here is someone else's input), matching the
    parallel driver's ``did_ingest and cleanup_mosaics`` rule.

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
    ) -> None:
        self._deployment = deployment
        self._params_for = params_for
        self._inputs_bucket = inputs_bucket
        self._cleanup_mosaics = cleanup_mosaics
        self._log = log
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_parallel), thread_name_prefix="cell-ingest")
        self._futures: dict[tuple[str, int], Future[None]] = {}

    def _run(self, zone: str, year: int) -> None:
        self._log.info("Ingest dispatch: %s-%d via %s", zone, year, self._deployment)
        run = run_deployment(self._deployment, parameters=self._params_for(zone, year))
        _check_completed(run, f"ingest {zone}-{year}")

    def start(self, zone: str, year: int) -> None:
        if (zone, year) not in self._futures:
            self._futures[(zone, year)] = self._executor.submit(self._run, zone, year)

    def wait(self, zone: str, year: int) -> None:
        self.start(zone, year)
        self._futures[(zone, year)].result()

    def cleanup(self, zone: str, year: int) -> None:
        if (zone, year) not in self._futures or not self._cleanup_mosaics:
            return
        delete_prefix(f"{self._inputs_bucket.rstrip('/')}/mosaics/{zone}/{year}", log=self._log)

    def shutdown(self) -> None:
        """Stop dispatching; don't wait on ingests nobody will consume."""
        self._executor.shutdown(wait=False, cancel_futures=True)


@flow(name="fill-zones-sequential", on_cancellation=[ray_cleanup_on_cancellation])
def fill_zones_sequential_flow(
    *,
    zones: list[str],
    year: int,
    paths: BucketPaths,
    ami_ssm_name: str,
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
    allow_model_mismatch: bool = False,
    s3_concurrency: int | None = None,
    idle_timeout_minutes: int = 10,
    max_consecutive_failures: int = 3,
    ingest: bool = True,
    ingest_deployment: str = "ingest-zone-year/ingest-zone-year",
    look_ahead: int = 2,
    cleanup_mosaics: bool = True,
    ingest_min_workers: int = 1,
    ingest_max_workers: int = 50,
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
    batch_days: int = 30,
) -> dict[str, Any]:
    """Fill one year's zones sequentially on a single shared Ray cluster.

    Args:
        zones: Zone group names for THIS year (UTM common names, e.g.
            ``["33N", "15S"]``). One year per flow run — the driver stays
            year-serial, and single-year cells are all distinct zone groups,
            which is what makes the runner's trailing assembly conflict-free.
        year: Campaign calendar year to fill (must be on the seeded axis).
        paths: Deployment storage contract (global store, land mask, mosaics).
        ami_ssm_name: SSM parameter name for the Ray GPU AMI ID.
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
        allow_model_mismatch: Fill even when the seeded store advertises a
            different encoder/checkpoint than this build (default rejects).
        s3_concurrency: Each trailing assembly's slice of the fleet S3-PUT
            budget. ``None`` = the aggregate target halved, leaving headroom
            for the live cell's concurrent staging writes.
        idle_timeout_minutes: Autoscaler idle-down override for the shared
            cluster (template default 2 min suits per-cell fills; the
            inter-zone seam here needs more slack).
        max_consecutive_failures: The runner's circuit breaker.
        ingest: Produce each cell's mosaics via the ingest deployment, look-ahead
            pipelined with inference (default). False = mosaics exist upstream.
        ingest_deployment: ``flow-name/deployment-name`` of the ingest deployment.
        look_ahead: Cells beyond the current one kept in ingest flight (bounds
            concurrent ingest Dask clusters AND in-flight mosaics, ADR-011).
        cleanup_mosaics: Delete each campaign-ingested mosaic after its cell
            lands (transient input). Ignored for ``ingest=False`` mosaics.
        ingest_min_workers: Lower Dask worker bound per ingest.
        ingest_max_workers: Upper Dask worker bound per ingest.
        min_valid_coverage: S2 per-solar-day keep threshold forwarded to ingest.
        batch_days: S1 CMR batch window forwarded to ingest.

    Returns:
        Summary dict: triage counts (retag / empty / live), the sequential
        runner's outcome summary, and elapsed seconds.
    """
    log = get_run_logger()

    # Lazily import the AWS providers so the flow file imports on machines
    # without ray/boto installed (arch tests, local inspection).
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials
    from tessera_embeddings.providers.aws.ray import make_instance_terminator, ray_cluster

    # Canonicalize + dedupe, preserving caller order until the size sort.
    zones = list(dict.fromkeys(canonicalize_zone(z) for z in zones))
    store_path = paths.global_store(store_name)
    land_mask_path = paths.land_mask_store(mask_name)
    window = parse_time_window(time_window_end or f"December {year}")
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

    def _ingest_params(zone: str, cell_year: int) -> dict[str, Any]:
        return {
            "zone": zone,
            "year": cell_year,
            "paths": paths.model_dump(),
            "mask_name": mask_name,
            "s1_orbit": s1_orbit,
            "min_workers": ingest_min_workers,
            "max_workers": ingest_max_workers,
            "min_valid_coverage": min_valid_coverage,
            "batch_days": batch_days,
            "allow_partial_window": allow_partial_window,
        }

    inputs = (
        _DeploymentCellInputs(
            deployment=ingest_deployment,
            params_for=_ingest_params,
            inputs_bucket=paths.inputs,
            cleanup_mosaics=cleanup_mosaics,
            max_parallel=look_ahead,
            log=log,
        )
        if ingest
        else None
    )

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
                min_valid_coverage=min_valid_coverage,
                s1_orbit=s1_orbit,
                allow_partial_window=allow_partial_window,
                code_suffix=code_suffix,
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

    # Fires only on actor replacement (dead instance reclaim) until the final
    # cell, whose tail retirement drains the fleet early.
    terminator = make_instance_terminator(log=log)

    def _infer(cell: SequentialCell, prep: PreparedCell, final: bool) -> ZoneFillHandoff:
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

    try:
        with ray_cluster(
            log,
            ami_ssm_name=ami_ssm_name,
            ssm_prefix=ssm_prefix,
            cloudwatch_log_group=cloudwatch_log_group,
            code_bucket=code_bucket,
            code_suffix=code_suffix,
            idle_timeout_minutes=idle_timeout_minutes,
        ) as resolved_yaml:
            activate(resolved_yaml)
            seq = fill_zones_sequential(
                cells=live,
                prepare=_prepare,
                infer=_infer,
                assemble=_assemble,
                log=log,
                inputs=inputs,
                look_ahead=look_ahead,
                max_consecutive_failures=max_consecutive_failures,
            )
    finally:
        if inputs is not None:
            inputs.shutdown()
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
