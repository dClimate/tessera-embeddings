"""Sentinel-2 L2A reflectance ingestion flow for ROI-based regions.

Outer flow provisions a Dask cluster (AWS Fargate by default; local
LocalCluster when ``use_local=True``); the inner flow is invoked via
``.with_options(task_runner=...)`` so Prefect binds the task runner
late, after the cluster context has produced a scheduler address.

The two-flow pattern is intentional and load-bearing: ``task_runner=``
must be set at flow definition time or via ``.with_options()`` on a
callable, which requires an inner ``@flow``. Do not try to collapse it
into a single ``@flow``.
"""

from __future__ import annotations

import logging
from typing import Any

from prefect import flow, get_run_logger
from prefect.runtime import flow_run as flow_run_ctx

from tessera_embeddings.ingest.roi_processing import DEFAULT_MIN_VALID_COVERAGE
from tessera_embeddings.orchestration.prefect.flows._dask_lifecycle import (
    dask_cleanup_on_cancellation,
    dask_resource_tags,
)
from tessera_embeddings.orchestration.prefect.flows._dask_runner import get_task_runner_for_cluster
from tessera_embeddings.orchestration.prefect.tasks.ingest import process_roi_reflectance

MAX_PIPELINE_DATES_WORKERS = 140
"""Widest fleet on which date pipelining is allowed to run.

Overlapping the next date's preparation with the current date's write only pays while the
write leaves the fleet room to absorb the coverage gate — that gate is worker-side work, so
on a fleet the write already keeps busy it is additive no matter how it is scheduled, and
past some width the overlap costs more than the client-side graph build it saves. This bound
is a measured calibration, not a property of the code; the measurements and the break-even
estimate behind it live in `yield-embeddings/context_docs/measurements/`.

Above it the flow declines to pipeline rather than obeying, because the flag reaches this
code from many callers and the failure is silent — a slower run looks like a slower run.
Declining can only prevent harm: pipelining is off by default, so nothing that did not ask
for it is affected.
"""


def _gated_pipeline_dates(
    *,
    pipeline_dates: bool,
    use_local: bool,
    max_workers: int,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> bool:
    """Return the pipelining flag actually in force, declining above the width bound.

    Scoped to the Fargate path: on ``use_local`` the ``max_workers`` parameter provisions
    nothing, so gating on it would disable pipelining for a cluster whose width it does not
    describe. The bound is inclusive.
    """
    if pipeline_dates and not use_local and max_workers > MAX_PIPELINE_DATES_WORKERS:
        log.warning(
            "pipeline_dates requested with max_workers=%d, above the %d-worker bound where "
            "overlapping preparation still pays; running WITHOUT pipelining. See "
            "MAX_PIPELINE_DATES_WORKERS.",
            max_workers,
            MAX_PIPELINE_DATES_WORKERS,
        )
        return False
    return pipeline_dates


@flow(name="ingest_s2_roi_impl")
def _ingest_s2_roi_impl(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
    provider: str = "earth-search",
    collection: str = "sentinel-2-l2a",
    storage_options: dict | None = None,
    crop_to_live_windows: bool = False,
    stream_stac_monthly: bool = True,
    overlap_window_writes: bool = True,
    pipeline_dates: bool = False,
) -> dict[str, Any]:
    """Inner flow: submits the S2 ingestion task to the configured Dask runner."""
    future = process_roi_reflectance.submit(
        roi_zarr_path=roi_zarr_path,
        start_date=start_date,
        end_date=end_date,
        store_path=store_path,
        min_valid_coverage=min_valid_coverage,
        provider=provider,
        collection=collection,
        storage_options=storage_options,
        crop_to_live_windows=crop_to_live_windows,
        stream_stac_monthly=stream_stac_monthly,
        overlap_window_writes=overlap_window_writes,
        pipeline_dates=pipeline_dates,
    )
    return future.result()


@flow(
    name="ingest_s2_roi_reflectance",
    # Both lists hold the SAME function: a crashed run leaks exactly like a
    # cancelled one. Keep the hook IDEMPOTENT — cancelling a parent and its child
    # together delivers the transition twice and runs it twice (2026-07-25).
    on_cancellation=[dask_cleanup_on_cancellation],
    on_crashed=[dask_cleanup_on_cancellation],
)
def ingest_s2_roi_reflectance(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    min_workers: int = 1,
    max_workers: int = 50,
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
    provider: str = "earth-search",
    collection: str = "sentinel-2-l2a",
    ec2_scheduler: bool = False,
    use_local: bool = False,
    storage_options: dict | None = None,
    perf_report_uri: str | None = None,
    crop_to_live_windows: bool = False,
    stream_stac_monthly: bool = True,
    overlap_window_writes: bool = True,
    pipeline_dates: bool = False,
    worker_env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Ingest S2 L2A reflectance for an ROI using Dask workers.

    Reads the Zarr ROI store to extract a WGS84 bounding box, queries
    STAC for all intersecting tiles, and writes a mosaicked
    ``reflectance.zarr`` store under ``store_path`` with the
    ingestion pipeline's INGEST_CHUNKS (4000x4000).

    Args:
        roi_zarr_path: Any fsspec-compatible URI to the Zarr ROI store.
        start_date: Inclusive start date (``YYYY-MM-DD``).
        end_date: Inclusive end date (``YYYY-MM-DD``).
        store_path: Base path for satellite mosaics; the function
            creates ``reflectance.zarr`` underneath.
        min_workers: Minimum Dask workers for adaptive scaling.
        max_workers: Maximum Dask workers for adaptive scaling.
        min_valid_coverage: Minimum percentage of valid ROI pixels
            (computed from SCL) required to keep a date.
        provider: STAC provider key.
        collection: Collection alias within the provider.
        ec2_scheduler: Run the Dask scheduler on EC2 instead of Fargate
            (better single-thread CPU for large graphs). Ignored when
            ``use_local=True``.
        use_local: Use the local-machine Dask provider instead of AWS.
            For tests and dev iteration on a single machine.
        storage_options: fsspec storage options forwarded to the
            domain function.
        perf_report_uri: Optional fsspec URI; when set, a Dask
            performance-report HTML for this run is captured and
            uploaded there (probe-rung profiling; default off).
            Ignored on the ``use_local`` path, which warns.
        stream_stac_monthly: Query the STAC catalog one calendar month at a time,
            prefetching the next while the current is processed, rather than querying
            the whole window up front. Bounds retained items so a year-long window fits
            the worker; ``False`` is the rollback path only.
        crop_to_live_windows: Restrict mosaic writes (and the S2 coverage
            reduce) to the chunk-aligned windows intersecting the ROI mask —
            one commit per date. Default False = legacy full-extent path.
        overlap_window_writes: Submit a date's windows as ONE dask compute rather
            than one blocking compute per window, so their critical paths overlap
            across the fleet instead of summing. Identical stores either way, and
            it falls back to the sequential write when the overlapped machinery is
            unavailable. Only meaningful with ``crop_to_live_windows``.
        pipeline_dates: Prepare the next date (load graph, coverage gate, footprint
            narrowing, masking) on a background thread while the current date is
            written, so preparation costs wall clock only where the write cannot
            cover it. The write itself stays serial and in date order — one commit
            per date — and the stores are identical either way. Ignored with a
            warning above ``MAX_PIPELINE_DATES_WORKERS``, where the overlap stops
            paying; narrow fleets benefit most.
        worker_env_overrides: Env vars merged into every Dask worker's environment
            for THIS run only, for A/B-ing worker-side tuning (allocator and cache
            behaviour) one arm at a time. Not a configuration channel: anything
            meant to hold for every run belongs in ``FargateConfig``. Ignored on
            the ``use_local`` path, which provisions no Fargate workers.

    Returns:
        ``IngestResult`` serialised as a dict (see
        :class:`tessera_embeddings.ingest.s2_roi.IngestResult`).
    """
    log = get_run_logger()
    log.info("Starting ingest_s2_roi_reflectance for %s", roi_zarr_path)

    pipeline_dates = _gated_pipeline_dates(
        pipeline_dates=pipeline_dates, use_local=use_local, max_workers=max_workers, log=log
    )

    if use_local:
        from tessera_embeddings.providers.local.dask import local_cluster

        if perf_report_uri:
            # Say so rather than no-op: an operator who set this and finds nothing
            # at the URI would otherwise suspect the upload or their credentials.
            log.warning("perf_report_uri is ignored on the local-cluster path (use_local=True)")
        with local_cluster() as cluster:
            log.info("Local Dask cluster ready: scheduler=%s", cluster.scheduler_address)
            task_runner = get_task_runner_for_cluster(cluster.scheduler_address)
            return _ingest_s2_roi_impl.with_options(task_runner=task_runner)(  # type: ignore[arg-type]
                roi_zarr_path=roi_zarr_path,
                start_date=start_date,
                end_date=end_date,
                store_path=store_path,
                min_valid_coverage=min_valid_coverage,
                provider=provider,
                collection=collection,
                storage_options=storage_options,
                crop_to_live_windows=crop_to_live_windows,
                stream_stac_monthly=stream_stac_monthly,
                overlap_window_writes=overlap_window_writes,
                pipeline_dates=pipeline_dates,
            )

    from tessera_embeddings.providers.aws.dask import ecs_cluster, maybe_performance_report

    with ecs_cluster(
        log,
        min_workers=min_workers,
        max_workers=max_workers,
        ec2_scheduler=ec2_scheduler,
        # A capped task stream silently truncates the report to the run's last few
        # dates; raise it only when a report is actually being captured.
        diagnostic_task_stream=bool(perf_report_uri),
        extra_worker_env=worker_env_overrides,
        # Tag every cluster resource with this run's id so the cancellation/crash
        # hook can sweep the tasks from a fresh process (see _dask_lifecycle).
        resource_tags=dask_resource_tags(flow_run_ctx.id),
    ) as cluster:
        task_runner = get_task_runner_for_cluster(cluster.scheduler_address)
        log.info("Task runner connected to scheduler at %s", cluster.scheduler_address)
        with maybe_performance_report(cluster.scheduler_address, perf_report_uri, log):
            return _ingest_s2_roi_impl.with_options(task_runner=task_runner)(  # type: ignore[arg-type]
                roi_zarr_path=roi_zarr_path,
                start_date=start_date,
                end_date=end_date,
                store_path=store_path,
                min_valid_coverage=min_valid_coverage,
                provider=provider,
                collection=collection,
                storage_options=storage_options,
                crop_to_live_windows=crop_to_live_windows,
                stream_stac_monthly=stream_stac_monthly,
                overlap_window_writes=overlap_window_writes,
                pipeline_dates=pipeline_dates,
            )
