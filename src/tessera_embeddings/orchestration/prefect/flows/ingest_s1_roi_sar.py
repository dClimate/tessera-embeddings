"""Sentinel-1 OPERA RTC SAR ingestion flow for ROI-based regions.

Outer flow provisions a Dask cluster (AWS Fargate by default) and
threads EDL credentials into worker env via ``extra_worker_env``; the
inner flow runs the per-batch task on that cluster.

EDL credentials enter at the **flow boundary** (read from
``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` env vars or a Prefect
``Secret`` block) — never inside a domain function. Hard rule #5.
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
    crop_to_live_windows: bool,
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
        crop_to_live_windows=crop_to_live_windows,
    )
    return future.result()


def _default_edl_env() -> dict[str, str]:
    """Read EDL credentials from environment variables.

    The credential source for production deployments is the Prefect
    work-pool job template, which injects ``EARTHDATA_USERNAME`` /
    ``EARTHDATA_PASSWORD`` as env vars on the flow runner. To source
    these from a Prefect ``Secret`` block instead, read the block here
    in the flow body (Hard rule #5: secrets enter at the flow boundary,
    not via injected callables — which cannot cross the deployment's
    JSON parameter boundary anyway).
    """
    return {
        "EARTHDATA_USERNAME": os.environ["EARTHDATA_USERNAME"],
        "EARTHDATA_PASSWORD": os.environ["EARTHDATA_PASSWORD"],
    }


@flow(
    name="ingest_s1_roi_sar",
    # Both lists hold the SAME function: a crashed run leaks exactly like a
    # cancelled one. Hooks here have been seen to run TWICE for one transition
    # (2026-07-25) — they must stay idempotent; see flows/_hook_invocation.py.
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
    crop_to_live_windows: bool = False,
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
        orbit: Orbit direction. Multi-orbit ingestion is a flow-level
            concern (call this flow twice).
        use_s3_direct: Use ASF in-region S3 endpoints (requires
            us-west-2 reachability and STS creds). When ``True``, the
            flow wires ``get_s3_credentials`` / ``set_s3_credentials`` as
            the STS refresh + broadcast callbacks for the domain function
            (workers receive EDL env vars via ``extra_worker_env`` but
            need STS tokens for the OPERA bucket; ``extra_worker_env``
            alone is not sufficient).
        use_local: Use the local Dask provider for testing.
        storage_options: fsspec storage options forwarded to the
            domain function.
        perf_report_uri: Optional fsspec URI; when set, a Dask
            performance-report HTML for this run is captured and
            uploaded there (probe-rung profiling; default off).
            Ignored on the ``use_local`` path, which warns.
        crop_to_live_windows: Restrict mosaic writes (and the S2 coverage
            reduce) to the chunk-aligned windows intersecting the ROI mask —
            one commit per date. Default False = legacy full-extent path.

    Returns:
        ``SarIngestResult`` serialised as a dict.
    """
    log = get_run_logger()
    log.info("Starting ingest_s1_roi_sar for %s (orbit=%s)", roi_zarr_path, orbit)

    # STS credential callbacks for the domain function, gated on
    # use_s3_direct. Workers receive EARTHDATA_USERNAME/PASSWORD via
    # extra_worker_env but cannot access s3://asf-cumulus-prod-opera-products
    # with IAM task-role credentials — they need short-lived STS tokens from
    # ASF's cumulus endpoint. The plain runner wires these the same way; the
    # Prefect flow must too, or every batch fails with AccessDenied on the
    # first S3 read.
    edl_credentials_fn = get_s3_credentials if use_s3_direct else None
    apply_credentials_fn = set_s3_credentials if use_s3_direct else None

    # IAM storage_options for the ROI-mask reads are resolved on the worker in
    # process_roi_sar, not here: they carry a live access key / secret / STS
    # token, and a flow parameter would be persisted in Prefect's DB and shown
    # in the UI as a plaintext credential.

    if use_local:
        from tessera_embeddings.providers.local.dask import local_cluster

        if perf_report_uri:
            # Say so rather than no-op: an operator who set this and finds nothing
            # at the URI would otherwise suspect the upload or their credentials.
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
                crop_to_live_windows=crop_to_live_windows,
            )

    from tessera_embeddings.providers.aws.dask import ecs_cluster, maybe_performance_report

    edl_env = _default_edl_env()

    with ecs_cluster(
        log,
        min_workers=min_workers,
        max_workers=max_workers,
        extra_worker_env=edl_env,
        # Tag every cluster resource with this run's id so the cancellation/crash
        # hook can sweep the tasks from a fresh process (see _dask_lifecycle).
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
                crop_to_live_windows=crop_to_live_windows,
            )
