"""Sentinel-2 L2A reflectance ingestion flow for ROI-based regions.

The outer flow provisions a Dask cluster (AWS Fargate by default; ``LocalCluster`` when
``use_local=True``) and invokes the inner flow via ``.with_options(task_runner=...)``, so
Prefect binds the runner late, after the cluster context has produced a scheduler address.

The two-flow split is load-bearing: ``task_runner=`` must be set at flow definition time or
via ``.with_options()`` on a callable, which requires an inner ``@flow``. Do not collapse it.
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
    get_task_runner_for_cluster,
)
from tessera_embeddings.orchestration.prefect.tasks.ingest import process_roi_reflectance

MAX_PIPELINE_DATES_WORKERS = 140
"""Widest fleet on which date pipelining is allowed to run.

Overlapping the next date's preparation with the current date's write only pays while the
write leaves the fleet room to absorb the coverage gate. That gate is worker-side work, so on
a fleet the write already keeps busy it is additive however it is scheduled, and past some
width the overlap costs more than the client-side graph build it saves. A measured
calibration, not a property of the code: see `yield-embeddings/context_docs/measurements/`.

Above the bound the flow declines to pipeline rather than obeying, because the flag arrives
from many callers and the failure is silent — a slower run just looks like a slower run.
Declining can only prevent harm; pipelining is off by default.
"""


def _gated_pipeline_dates(
    *,
    pipeline_dates: bool,
    use_local: bool,
    max_workers: int,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> bool:
    """Return the pipelining flag actually in force, declining above the width bound.

    Fargate-only: on ``use_local`` the ``max_workers`` parameter provisions nothing, so
    gating on it would disable pipelining for a cluster whose width it does not describe.
    The bound is inclusive.
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
    stream_stac_monthly: bool = True,
    overlap_window_writes: bool = True,
    pipeline_dates: bool = False,
    batch_dates: int | None = None,
    allow_ingest_code_mismatch: bool = False,
    s3_region: str | None = None,
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
        stream_stac_monthly=stream_stac_monthly,
        overlap_window_writes=overlap_window_writes,
        pipeline_dates=pipeline_dates,
        batch_dates=batch_dates,
        allow_ingest_code_mismatch=allow_ingest_code_mismatch,
        s3_region=s3_region,
    )
    return future.result()


@flow(
    name="ingest_s2_roi_reflectance",
    # Both lists hold the SAME function: a crashed run leaks exactly like a cancelled one.
    # Keep the hook IDEMPOTENT — cancelling a parent and its child together delivers the
    # transition twice and runs it twice (2026-07-25).
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
    stream_stac_monthly: bool = True,
    overlap_window_writes: bool = True,
    pipeline_dates: bool = False,
    batch_dates: int | None = None,
    worker_env_overrides: dict[str, str] | None = None,
    allow_ingest_code_mismatch: bool = False,
    s3_region: str | None = None,
) -> dict[str, Any]:
    """Ingest S2 L2A reflectance for an ROI using Dask workers.

    Reads the Zarr ROI store for a WGS84 bounding box, queries STAC for intersecting tiles,
    and writes a mosaicked ``reflectance.zarr`` under ``store_path`` at the ingestion
    pipeline's INGEST_CHUNKS (4000x4000).

    Args:
        roi_zarr_path: Any fsspec-compatible URI to the Zarr ROI store.
        start_date: Inclusive start date (``YYYY-MM-DD``).
        end_date: Inclusive end date (``YYYY-MM-DD``).
        store_path: Base path for satellite mosaics; ``reflectance.zarr`` is created under it.
        min_workers: Minimum Dask workers for adaptive scaling.
        max_workers: Maximum Dask workers for adaptive scaling.
        min_valid_coverage: Minimum percentage of valid ROI pixels (from SCL) to keep a date.
        provider: STAC provider key.
        collection: Collection alias within the provider.
        ec2_scheduler: Run the Dask scheduler on EC2 instead of Fargate (better
            single-thread CPU for large graphs). Ignored when ``use_local=True``.
        use_local: Use the local-machine Dask provider instead of AWS, for tests and dev.
        storage_options: fsspec storage options forwarded to the domain function.
        perf_report_uri: When set, a Dask performance-report HTML for this run is uploaded
            there (probe-rung profiling; off by default). Ignored, with a warning, on the
            ``use_local`` path.
        stream_stac_monthly: Query STAC one calendar month at a time, prefetching the next
            while the current is processed. Bounds retained items so a year-long window
            fits the worker; ``False`` is the rollback path only.
        overlap_window_writes: Submit a date's windows as ONE dask compute rather than a
            blocking compute per window, so their critical paths overlap across the fleet
            instead of summing. Identical stores either way; falls back to the sequential
            write when the overlapped machinery is unavailable.
        pipeline_dates: Prepare the next date (load graph, coverage gate, footprint
            narrowing, masking) on a background thread while the current date is written,
            so preparation costs wall clock only where the write cannot cover it. The write
            stays serial and in date order — one commit per date — and stores are identical
            either way. Ignored with a warning above ``MAX_PIPELINE_DATES_WORKERS``; narrow
            fleets benefit most.
        batch_dates: Compute up to this many consecutive passing dates as one dask graph and
            land them as ONE commit, so their work packs the fleet together and the per-date
            drain tail and commit gap are paid once per batch. The commit unit becomes the
            batch: a mid-batch failure commits none of its dates, and the retry re-ingests
            exactly those. Identical stores either way. Composes with ``pipeline_dates`` —
            the look-ahead becomes the batch, so a whole batch's preparation hides behind
            the previous batch's write. Leave at ``None`` to size it from the ROI, which is
            what the campaign wants: the benefit is not monotonic in ROI size, so one global
            value regresses part of the range. Pin an int only for one arm of a comparison.
        worker_env_overrides: Env vars merged into every Dask worker's environment for THIS
            run only, to A/B worker-side tuning (allocator, cache behaviour) one arm at a
            time. Not a configuration channel — anything meant to hold for every run belongs
            in ``FargateConfig``. Ignored on the ``use_local`` path.
        allow_ingest_code_mismatch: Resume a store built by different ingest code (off by default).

        s3_region: S3 region for the mosaic Icechunk store. ``None`` uses the storage
            layer's default (us-west-2); set it when the input bucket lives elsewhere, or
            the mosaic writes sign against the wrong region and fail after the preflight
            checks have already passed.

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
            # Say so rather than no-op: an operator finding nothing at the URI would
            # otherwise suspect the upload or their credentials.
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
                stream_stac_monthly=stream_stac_monthly,
                overlap_window_writes=overlap_window_writes,
                pipeline_dates=pipeline_dates,
                batch_dates=batch_dates,
                allow_ingest_code_mismatch=allow_ingest_code_mismatch,
                s3_region=s3_region,
            )

    from tessera_embeddings.providers.aws.dask import ecs_cluster, maybe_performance_report

    with ecs_cluster(
        log,
        min_workers=min_workers,
        max_workers=max_workers,
        ec2_scheduler=ec2_scheduler,
        # A capped task stream silently truncates the report to the run's last few dates;
        # raise it only when a report is actually being captured.
        diagnostic_task_stream=bool(perf_report_uri),
        extra_worker_env=worker_env_overrides,
        # Tag every cluster resource with this run's id so the cancellation/crash hook can
        # sweep the tasks from a fresh process (see _dask_lifecycle).
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
                stream_stac_monthly=stream_stac_monthly,
                overlap_window_writes=overlap_window_writes,
                pipeline_dates=pipeline_dates,
                batch_dates=batch_dates,
                allow_ingest_code_mismatch=allow_ingest_code_mismatch,
                s3_region=s3_region,
            )
