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

from tessera_embeddings.ingest.auth import get_s3_credentials, set_s3_credentials
from tessera_embeddings.ingest.s1_roi import S1Orbit
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
    )
    return future.result()


def _default_edl_env() -> dict[str, str]:
    """Read EDL credentials from environment variables.

    The default credential source for production deployments is the
    Prefect work-pool job template, which injects
    ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` as env vars on the
    flow runner. Override at flow-call time by passing an alternative
    ``edl_env_fn`` (e.g. closing over a Prefect ``Secret`` block).
    """
    return {
        "EARTHDATA_USERNAME": os.environ["EARTHDATA_USERNAME"],
        "EARTHDATA_PASSWORD": os.environ["EARTHDATA_PASSWORD"],
    }


@flow(name="ingest_s1_roi_sar")
def ingest_s1_roi_sar(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    min_workers: int = 1,
    max_workers: int = 50,
    batch_days: int = 30,
    orbit: S1Orbit = "ascending",
    use_s3_direct: bool = True,
    use_local: bool = False,
    edl_env_fn: Callable[[], dict[str, str]] = _default_edl_env,
    edl_credentials_fn: Callable[[], dict[str, str]] | None = None,
    apply_credentials_fn: Callable[[dict[str, str]], None] | None = None,
    storage_options: dict | None = None,
) -> dict[str, Any]:
    """Ingest OPERA RTC-S1 SAR for an ROI using Dask workers.

    Args:
        roi_zarr_path: Any fsspec-compatible URI to the Zarr ROI store.
        start_date: Inclusive start date (``YYYY-MM-DD``).
        end_date: Exclusive end date (``YYYY-MM-DD``).
        store_path: Base path for satellite mosaics; creates
            ``sar_<orbit>.zarr`` underneath.
        min_workers: Minimum Dask workers.
        max_workers: Maximum Dask workers.
        batch_days: Days per time batch.
        orbit: Orbit direction. Multi-orbit ingestion is a flow-level
            concern (call this flow twice).
        use_s3_direct: Use ASF in-region S3 endpoints (requires
            us-west-2 reachability and STS creds).
        use_local: Use the local Dask provider for testing.
        edl_env_fn: Callable returning EDL env vars to inject into
            workers via ``extra_worker_env``. Defaults to reading
            ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` from the
            flow-runner environment. Override to close over a Prefect
            ``Secret`` block.
        edl_credentials_fn: STS credential refresh callback forwarded to
            the domain function. Defaults to ``get_s3_credentials`` when
            ``use_s3_direct=True`` — workers receive EDL env vars via
            ``extra_worker_env`` but need STS tokens for the OPERA bucket;
            ``extra_worker_env`` alone is not sufficient. Override to close
            over a Prefect ``Secret`` block or pass ``None`` only when the
            calling environment already injects STS tokens out-of-band.
        apply_credentials_fn: Companion to ``edl_credentials_fn`` that
            applies the returned creds to the orchestrator env and running
            cluster workers. Defaults to ``set_s3_credentials`` when
            ``use_s3_direct=True``.
        storage_options: fsspec storage options forwarded to the
            domain function.

    Returns:
        ``SarIngestResult`` serialised as a dict.
    """
    log = get_run_logger()
    log.info("Starting ingest_s1_roi_sar for %s (orbit=%s)", roi_zarr_path, orbit)

    # Default STS credential callbacks when use_s3_direct=True.
    # Workers receive EARTHDATA_USERNAME/PASSWORD via extra_worker_env but
    # cannot access s3://asf-cumulus-prod-opera-products with IAM task-role
    # credentials — they need short-lived STS tokens from ASF's cumulus
    # endpoint. The plain runner wires these explicitly; the Prefect flow
    # must too, or every batch fails with AccessDenied on the first S3 read.
    if use_s3_direct and edl_credentials_fn is None:
        edl_credentials_fn = get_s3_credentials
        apply_credentials_fn = set_s3_credentials
        log.debug("Defaulting edl_credentials_fn=get_s3_credentials, apply_credentials_fn=set_s3_credentials")

    # When S3 direct access is enabled, set_s3_credentials will overwrite
    # AWS_* env vars with OPERA-scoped STS tokens.  Any da.from_zarr on our
    # own ROI store that runs after the first cred refresh would pick up those
    # tokens and get AccessDenied.  Resolve IAM creds now (before STS injection)
    # and pass them as explicit storage_options so the ROI reads are immune.
    if use_s3_direct and roi_zarr_path.startswith("s3://") and storage_options is None:
        from tessera_embeddings.providers.aws.credentials import iam_s3_storage_options

        storage_options = iam_s3_storage_options()
        log.debug("Resolved IAM storage options for ROI mask reads (bypass OPERA STS env vars)")

    if use_local:
        from tessera_embeddings.providers.local.dask import local_cluster

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
            )

    from tessera_embeddings.providers.aws.dask import ecs_cluster

    edl_env = edl_env_fn()

    with ecs_cluster(
        log,
        min_workers=min_workers,
        max_workers=max_workers,
        extra_worker_env=edl_env,
    ) as cluster:
        task_runner = get_task_runner_for_cluster(cluster.scheduler_address)
        log.info("Task runner connected to scheduler at %s", cluster.scheduler_address)
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
        )
