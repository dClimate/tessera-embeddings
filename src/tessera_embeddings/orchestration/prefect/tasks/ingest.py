"""Prefect task shells for ingest domain functions.

Each shell is ~20 LOC: pull ``client`` and ``log`` from Prefect / Dask
context, delegate to the domain function, convert the dataclass
result to a dict at the boundary.

This file is one of the few places in the package that imports from
:mod:`prefect`. Domain modules under ``ingest/`` never do.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from dask.distributed import get_client
from prefect import get_run_logger, task

from tessera_embeddings.ingest.roi_processing import DEFAULT_MIN_VALID_COVERAGE
from tessera_embeddings.ingest.s1_roi import S1Orbit, ingest_s1_roi_sar
from tessera_embeddings.ingest.s2_roi import ingest_s2_roi_reflectance


@task(name="process-roi-reflectance")
def process_roi_reflectance(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
    provider: str = "earth-search",
    collection: str = "sentinel-2-l2a",
    storage_options: dict | None = None,
) -> dict[str, Any]:
    """Prefect task: ingest S2 reflectance for one ROI.

    Pulls the Dask client + run logger from Prefect / Dask context,
    delegates to :func:`ingest_s2_roi_reflectance`, and returns the
    resulting dataclass as a dict (Prefect's UI displays dicts cleanly).

    Retry policy lives inside the domain function via tenacity (narrow
    scope: just the ``write_dataset`` call). Do **not** add
    ``@task(retries=...)`` here — domain retries already cover the
    transient cases, and outer retries would re-run the whole
    multi-day loop.
    """
    result = ingest_s2_roi_reflectance(
        roi_zarr_path=roi_zarr_path,
        start_date=start_date,
        end_date=end_date,
        store_path=store_path,
        client=get_client(),
        min_valid_coverage=min_valid_coverage,
        provider=provider,
        collection=collection,
        log=get_run_logger(),
        storage_options=storage_options,
    )
    return asdict(result)


@task(name="process-roi-sar")
def process_roi_sar(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    orbit: S1Orbit = "ascending",
    batch_days: int = 30,
    edl_credentials_fn: Callable[[], dict[str, str]] | None = None,
    apply_credentials_fn: Callable[[dict[str, str]], None] | None = None,
    use_s3_direct: bool = True,
    storage_options: dict | None = None,
) -> dict[str, Any]:
    """Prefect task: ingest S1 OPERA SAR for one ROI.

    The credential callbacks (``edl_credentials_fn``,
    ``apply_credentials_fn``) are forwarded as-is. Hard rule #5
    (secrets enter at flow entry only) means the *flow* constructs the
    closures over Prefect Blocks / env vars; this task shell never
    reads credentials directly.
    """
    result = ingest_s1_roi_sar(
        roi_zarr_path=roi_zarr_path,
        start_date=start_date,
        end_date=end_date,
        store_path=store_path,
        client=get_client(),
        orbit=orbit,
        batch_days=batch_days,
        edl_credentials_fn=edl_credentials_fn,
        apply_credentials_fn=apply_credentials_fn,
        use_s3_direct=use_s3_direct,
        log=get_run_logger(),
        storage_options=storage_options,
    )
    return asdict(result)
