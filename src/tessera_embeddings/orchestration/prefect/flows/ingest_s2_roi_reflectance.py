"""Sentinel-2 L2A reflectance ingestion flow for ROI-based regions.

The flow provisions a Dask cluster (AWS Fargate by default; ``LocalCluster`` when
``use_local=True``) and runs the ``process_roi_reflectance`` task shell on it with
:func:`._dask_lifecycle.run_on_cluster`, as a task run of this flow run, so the ingest's progress
logs land on this run.
"""

from __future__ import annotations

import logging
from typing import Any

from prefect import flow, get_run_logger
from prefect.runtime import flow_run as flow_run_ctx

from tessera_embeddings.ingest.roi_processing import DEFAULT_MIN_VALID_COVERAGE
from tessera_embeddings.ingest.s2_roi import BATCH_CHUNKS_PER_WORKER
from tessera_embeddings.orchestration.prefect.flows._dask_lifecycle import (
    dask_cleanup_on_cancellation,
    dask_resource_tags,
    run_on_cluster,
)
from tessera_embeddings.orchestration.prefect.tasks.ingest import process_roi_reflectance
from tessera_embeddings.providers.local.dask import local_cluster

MAX_PIPELINE_DATES_WORKERS = 140
"""Widest Fargate fleet on which the flow prepares the next S2 date during the current write.

Preparing a date has two parts. Building its read plan runs in the ingest's driver: one thread,
in the task that drives the ingest on one worker. Checking its cloud mask reads the date's
cloud-mask (SCL) data as tasks spread across the fleet, on the same workers that run the write.
The overlap prepares the next date on a background thread of the driver while the fleet writes
the current one. The read plan then costs nothing, because the driver would otherwise wait on
the write. The cloud-mask check costs nothing only where the write leaves workers idle; where the
write keeps every worker busy, the check's tasks take turns with the write's and lengthen it.

Empirically, preparation shrinks far less than the write as the fleet widens, and why is not
settled. The suspicion, not established: the read plan is one thread, so more workers cannot speed
it; and the cloud-mask check is a small graph, so once a fleet has more workers than its tasks,
per-request read latency rather than worker count sets its time.

So the overlap saves at least the read-plan time, plus whatever part of the check fits into idle
worker time. In paired tests it was never slower, and it gained most on narrower fleets. No fleet
wider than this bound has been measured, so the overlap is off above it as a precaution, not
because a loss was seen there. Evidence: ``context_docs/ingest/campaign-ingest-measurements.md``
§3.21.
"""


def _pipelines_dates(
    *,
    use_local: bool,
    max_workers: int,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> bool:
    """Decide whether this run prepares each S2 date while the previous one is written.

    Yes, unless the Fargate fleet is wider than ``MAX_PIPELINE_DATES_WORKERS`` (inclusive
    bound). On ``use_local`` the ``max_workers`` parameter provisions nothing, so it decides
    nothing there and the local cluster always pipelines.
    """
    if use_local or max_workers <= MAX_PIPELINE_DATES_WORKERS:
        return True
    log.info(
        "Preparing S2 dates one at a time: this %d-worker fleet is wider than the %d workers up to "
        "which preparing the next date during the current write is known to pay. See "
        "MAX_PIPELINE_DATES_WORKERS.",
        max_workers,
        MAX_PIPELINE_DATES_WORKERS,
    )
    return False


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
    batch_dates: int | None = None,
    worker_env_overrides: dict[str, str] | None = None,
    allow_ingest_code_mismatch: bool = False,
    s3_region: str | None = None,
) -> dict[str, Any]:
    """Ingest S2 L2A reflectance for an ROI using Dask workers.

    Reads the Zarr ROI store for a WGS84 bounding box, queries STAC for intersecting tiles,
    and writes a mosaicked ``reflectance.zarr`` under ``store_path`` at the ingestion
    pipeline's INGEST_CHUNKS (4000x4000).

    Each date's preparation (load graph, coverage gate, footprint narrowing, masking) runs on a
    background thread while the previous date is written, so it costs wall clock only where the
    write cannot cover it. Automatic, with no setting: every fleet up to
    ``MAX_PIPELINE_DATES_WORKERS`` pipelines, and a wider one prepares dates one at a time.
    Identical stores either way.

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
        batch_dates: The most consecutive passing dates computed as one dask graph and landed
            as ONE commit, so the per-write overhead (graph submission, drain tail, commit) is
            paid once per batch. The commit unit becomes the batch: a mid-batch failure commits
            none of its dates, and the retry re-ingests exactly those. Identical stores either
            way. When dates are pipelined the look-ahead becomes the batch. ``None`` caps it at
            ``AUTO_BATCH_DATES`` (4). How many dates a batch actually holds is set per date by a
            budget of ``BATCH_CHUNKS_PER_WORKER`` storage chunks per worker: batching pays only
            for sparse writes, so sparse dates fill a batch and a date that would push it past
            the budget starts a new one. Pin an int only for one arm of a comparison.
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

    pipeline_dates = _pipelines_dates(use_local=use_local, max_workers=max_workers, log=log)
    # From the fleet's planned width, since a Fargate fleet may still be starting when the ingest
    # counts its workers. A local cluster is up already, so the ingest counts it itself (None).
    batch_chunk_budget = None if use_local else BATCH_CHUNKS_PER_WORKER * max_workers

    task_parameters: dict[str, Any] = {
        "roi_zarr_path": roi_zarr_path,
        "start_date": start_date,
        "end_date": end_date,
        "store_path": store_path,
        "min_valid_coverage": min_valid_coverage,
        "provider": provider,
        "collection": collection,
        "storage_options": storage_options,
        "stream_stac_monthly": stream_stac_monthly,
        "overlap_window_writes": overlap_window_writes,
        "pipeline_dates": pipeline_dates,
        "batch_dates": batch_dates,
        "batch_chunk_budget": batch_chunk_budget,
        "allow_ingest_code_mismatch": allow_ingest_code_mismatch,
        "s3_region": s3_region,
    }

    if use_local:
        if perf_report_uri:
            # Say so rather than no-op: an operator finding nothing at the URI would
            # otherwise suspect the upload or their credentials.
            log.warning("perf_report_uri is ignored on the local-cluster path (use_local=True)")
        with local_cluster() as cluster:
            log.info("Local Dask cluster ready: scheduler=%s", cluster.scheduler_address)
            return run_on_cluster(cluster.scheduler_address, process_roi_reflectance, task_parameters)

    # Deferred: providers.aws.dask needs the `aws` extra, which a use_local run must not.
    from tessera_embeddings.providers.aws.dask import ecs_cluster, maybe_performance_report

    with (
        ecs_cluster(
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
        ) as cluster,
        maybe_performance_report(cluster.scheduler_address, perf_report_uri, log),
    ):
        return run_on_cluster(cluster.scheduler_address, process_roi_reflectance, task_parameters)
