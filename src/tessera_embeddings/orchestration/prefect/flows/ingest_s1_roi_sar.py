"""Sentinel-1 OPERA RTC SAR ingestion flow for ROI-based regions.

The outer flow provisions a Dask cluster (AWS Fargate by default) and threads EDL
credentials into worker env via ``extra_worker_env``; the inner flow runs the per-batch task
on that cluster.

EDL credentials enter at the **flow boundary** (``EARTHDATA_USERNAME`` /
``EARTHDATA_PASSWORD`` env vars, or a Prefect ``Secret`` block) — never inside a domain
function. Hard rule #5.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from prefect import flow, get_run_logger
from prefect.runtime import flow_run as flow_run_ctx

from tessera_embeddings.ingest.auth import get_s3_credentials, set_s3_credentials
from tessera_embeddings.ingest.s1_roi import S1Orbit
from tessera_embeddings.orchestration.prefect.flows._dask_lifecycle import (
    dask_cleanup_on_cancellation,
    dask_resource_tags,
)
from tessera_embeddings.orchestration.prefect.flows._dask_runner import get_task_runner_for_cluster
from tessera_embeddings.orchestration.prefect.tasks.ingest import process_roi_sar


@flow(name="ingest_s1_roi_impl")
def _ingest_s1_roi_impl(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    orbit: S1Orbit,
    batch_days: int,
    edl_credentials_fn: Callable[[], dict[str, str]] | None,
    apply_credentials_fn: Callable[[dict[str, str]], None] | None,
    use_s3_direct: bool,
    storage_options: dict | None,
    overlap_window_writes: bool,
    pipeline_batches: bool,
    narrow_windows_per_date: bool,
    allow_ingest_code_mismatch: bool,
    s3_region: str | None = None,
) -> dict[str, Any]:
    """Inner flow: submits the S1 ingestion task to the configured Dask runner."""
    future = process_roi_sar.submit(
        roi_zarr_path=roi_zarr_path,
        start_date=start_date,
        end_date=end_date,
        store_path=store_path,
        orbit=orbit,
        batch_days=batch_days,
        edl_credentials_fn=edl_credentials_fn,
        apply_credentials_fn=apply_credentials_fn,
        use_s3_direct=use_s3_direct,
        storage_options=storage_options,
        overlap_window_writes=overlap_window_writes,
        pipeline_batches=pipeline_batches,
        narrow_windows_per_date=narrow_windows_per_date,
        allow_ingest_code_mismatch=allow_ingest_code_mismatch,
        s3_region=s3_region,
    )
    return future.result()


def _default_edl_env() -> dict[str, str]:
    """Read EDL credentials from environment variables.

    Production deployments get these from the Prefect work-pool job template, which injects
    ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` on the flow runner. To source them from
    a Prefect ``Secret`` block instead, read the block here in the flow body — hard rule #5:
    secrets enter at the flow boundary, not via injected callables, which cannot cross the
    deployment's JSON parameter boundary anyway.
    """
    return {
        "EARTHDATA_USERNAME": os.environ["EARTHDATA_USERNAME"],
        "EARTHDATA_PASSWORD": os.environ["EARTHDATA_PASSWORD"],
    }


@flow(
    name="ingest_s1_roi_sar",
    # Both lists hold the SAME function: a crashed run leaks exactly like a
    # cancelled one. Keep the hook IDEMPOTENT — cancelling a parent and its child
    # together delivers the transition twice and runs it twice (2026-07-25).
    on_cancellation=[dask_cleanup_on_cancellation],
    on_crashed=[dask_cleanup_on_cancellation],
)
def ingest_s1_roi_sar(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    min_workers: int = 1,
    max_workers: int = 50,
    batch_days: int = 30,
    orbit: S1Orbit,
    use_s3_direct: bool = True,
    use_local: bool = False,
    storage_options: dict | None = None,
    perf_report_uri: str | None = None,
    overlap_window_writes: bool = True,
    pipeline_batches: bool = True,
    narrow_windows_per_date: bool = True,
    allow_ingest_code_mismatch: bool = False,
    s3_region: str | None = None,
) -> dict[str, Any]:
    """Ingest OPERA RTC-S1 SAR for an ROI using Dask workers.

    Args:
        roi_zarr_path: Any fsspec-compatible URI to the Zarr ROI store.
        start_date: Inclusive start date (``YYYY-MM-DD``).
        end_date: Inclusive end date (``YYYY-MM-DD``).
        store_path: Base path for satellite mosaics; creates
            ``sar_<orbit>.zarr`` underneath.
        min_workers: Minimum Dask workers.
        max_workers: Maximum Dask workers.
        batch_days: Days per time batch.
        orbit: Orbit direction. Multi-orbit ingestion is a flow-level concern (call this
            flow twice).
        use_s3_direct: Use ASF in-region S3 endpoints (requires us-west-2 reachability and
            STS creds). When ``True`` the flow wires ``get_s3_credentials`` /
            ``set_s3_credentials`` as the STS refresh and broadcast callbacks: workers get
            EDL env vars via ``extra_worker_env`` but need STS tokens for the OPERA bucket.
        use_local: Use the local Dask provider for testing.
        storage_options: fsspec storage options forwarded to the domain function.
        perf_report_uri: Optional fsspec URI; when set, a Dask performance-report HTML for
            this run is captured and uploaded there (probe-rung profiling; default off).
            Ignored on the ``use_local`` path, which warns.
        overlap_window_writes: Submit a date's windows as ONE dask compute rather than one
            blocking compute per window, so they share the fleet instead of each waiting its
            turn. Identical store either way. **Defaults ON.** Also selects the window merge
            exchange rate, which prices a boundary by how it is written, so the two cannot
            drift apart.
        narrow_windows_per_date: Write only the live windows a date's own imagery reaches,
            as the S2 path does. **Defaults ON**: six times fewer windows per date in both
            zones measured, worth 7-20% of per-date wall clock. Dates reaching NO live window
            are skipped unconditionally, independent of this flag.
        pipeline_batches: Prepare the NEXT batch's catalogue query while the current batch
            writes, so only the first batch pays its query on the critical path. **Defaults
            ON.** Look-ahead is one batch and not configurable: a batch's write is one long
            consume, so depth 1 covers it, and deeper retention once deadlocked the S2
            driver. Set False for a strictly serial query-then-write loop.
        allow_ingest_code_mismatch: Resume a store built by different ingest code (off by
            default).

        s3_region: S3 region for the mosaic Icechunk store. ``None`` uses the storage
            layer's default (us-west-2); set it when the input bucket lives elsewhere, or
            the mosaic writes sign against the wrong region and fail after the preflight
            checks have already passed.

    Returns:
        ``SarIngestResult`` serialised as a dict.
    """
    log = get_run_logger()
    log.info("Starting ingest_s1_roi_sar for %s (orbit=%s)", roi_zarr_path, orbit)

    # STS credential callbacks for the domain function, gated on use_s3_direct. Workers get
    # EARTHDATA_USERNAME/PASSWORD via extra_worker_env but cannot reach
    # s3://asf-cumulus-prod-opera-products with IAM task-role credentials — they need
    # short-lived STS tokens from ASF's cumulus endpoint. The plain runner wires these the
    # same way; without them every batch fails with AccessDenied on the first S3 read.
    edl_credentials_fn = get_s3_credentials if use_s3_direct else None
    apply_credentials_fn = set_s3_credentials if use_s3_direct else None

    # IAM storage_options for the ROI-mask reads are resolved on the worker in
    # process_roi_sar, not here: they carry a live access key / secret / STS token, and a
    # flow parameter is persisted in Prefect's DB and shown in the UI as plaintext.

    if use_local:
        from tessera_embeddings.providers.local.dask import local_cluster

        if perf_report_uri:
            # Say so rather than no-op: an operator who set this and finds nothing at the
            # URI would otherwise suspect the upload or their credentials.
            log.warning("perf_report_uri is ignored on the local-cluster path (use_local=True)")
        with local_cluster() as cluster:
            log.info("Local Dask cluster ready: scheduler=%s", cluster.scheduler_address)
            task_runner = get_task_runner_for_cluster(cluster.scheduler_address)
            return _ingest_s1_roi_impl.with_options(task_runner=task_runner)(  # type: ignore[arg-type]
                roi_zarr_path=roi_zarr_path,
                start_date=start_date,
                end_date=end_date,
                store_path=store_path,
                orbit=orbit,
                batch_days=batch_days,
                edl_credentials_fn=edl_credentials_fn,
                apply_credentials_fn=apply_credentials_fn,
                use_s3_direct=use_s3_direct,
                storage_options=storage_options,
                overlap_window_writes=overlap_window_writes,
                pipeline_batches=pipeline_batches,
                narrow_windows_per_date=narrow_windows_per_date,
                allow_ingest_code_mismatch=allow_ingest_code_mismatch,
                s3_region=s3_region,
            )

    from tessera_embeddings.providers.aws.dask import ecs_cluster, maybe_performance_report

    edl_env = _default_edl_env()

    with ecs_cluster(
        log,
        min_workers=min_workers,
        max_workers=max_workers,
        extra_worker_env=edl_env,
        # A capped task stream silently truncates the report to the run's last few dates;
        # raise it only when a report is actually being captured.
        diagnostic_task_stream=bool(perf_report_uri),
        # Tag every cluster resource with this run's id so the cancellation/crash hook can
        # sweep the tasks from a fresh process (see _dask_lifecycle).
        resource_tags=dask_resource_tags(flow_run_ctx.id),
    ) as cluster:
        task_runner = get_task_runner_for_cluster(cluster.scheduler_address)
        log.info("Task runner connected to scheduler at %s", cluster.scheduler_address)
        with maybe_performance_report(cluster.scheduler_address, perf_report_uri, log):
            return _ingest_s1_roi_impl.with_options(task_runner=task_runner)(  # type: ignore[arg-type]
                roi_zarr_path=roi_zarr_path,
                start_date=start_date,
                end_date=end_date,
                store_path=store_path,
                orbit=orbit,
                batch_days=batch_days,
                edl_credentials_fn=edl_credentials_fn,
                apply_credentials_fn=apply_credentials_fn,
                use_s3_direct=use_s3_direct,
                storage_options=storage_options,
                overlap_window_writes=overlap_window_writes,
                pipeline_batches=pipeline_batches,
                narrow_windows_per_date=narrow_windows_per_date,
                allow_ingest_code_mismatch=allow_ingest_code_mismatch,
                s3_region=s3_region,
            )
