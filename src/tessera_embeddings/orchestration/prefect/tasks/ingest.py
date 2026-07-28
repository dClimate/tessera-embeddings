"""Prefect task shells for ingest domain functions.

Each shell is ~20 LOC: pull ``client`` and ``log`` from Prefect / Dask
context, delegate to the domain function, convert the dataclass
result to a dict at the boundary.

This file is one of the few places in the package that imports from
:mod:`prefect`. Domain modules under ``ingest/`` never do.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any

from dask.distributed import get_client
from prefect import get_run_logger, task

from tessera_embeddings.config.ingest import INGEST_MANIFEST_SPLIT
from tessera_embeddings.ingest.roi_processing import DEFAULT_MIN_VALID_COVERAGE
from tessera_embeddings.ingest.s1_roi import S1Orbit, ingest_s1_roi_sar
from tessera_embeddings.ingest.s2_roi import ingest_s2_roi_reflectance
from tessera_embeddings.storage.zarr_store import credentials_provider, manifest_split


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
    crop_to_live_windows: bool = False,
    stream_stac_monthly: bool = True,
    overlap_window_writes: bool = True,
    pipeline_dates: bool = False,
    # ``int | None``, not ``int``: None is the value that means "derive from the ROI's size"
    # (config.ingest.auto_batch_dates), so an ``int``-annotated shell defaulting to 1 cannot
    # express it and pins any caller that omits the flag to never batching.
    batch_dates: int | None = None,
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
    # Shard the mosaic's manifests: the store is created and appended to entirely
    # within this call, so create and every later append see the same config.
    with manifest_split(INGEST_MANIFEST_SPLIT):
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
            crop_to_live_windows=crop_to_live_windows,
            stream_stac_monthly=stream_stac_monthly,
            overlap_window_writes=overlap_window_writes,
            pipeline_dates=pipeline_dates,
            batch_dates=batch_dates,
        )
    return asdict(result)


@task(name="process-roi-sar")
def process_roi_sar(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    orbit: S1Orbit,
    batch_days: int = 30,
    edl_credentials_fn: Callable[[], dict[str, str]] | None = None,
    apply_credentials_fn: Callable[[dict[str, str]], None] | None = None,
    use_s3_direct: bool = True,
    storage_options: dict | None = None,
    crop_to_live_windows: bool = False,
    # Mirror the domain defaults: these shells forward knobs, so a default here that
    # disagrees with s1_roi.ingest_s1_roi_sar silently changes behaviour for any caller
    # that does not pass the flag explicitly.
    overlap_window_writes: bool = True,
    pipeline_batches: bool = True,
    narrow_windows_per_date: bool = False,
) -> dict[str, Any]:
    """Prefect task: ingest S1 OPERA SAR for one ROI.

    The credential callbacks (``edl_credentials_fn``,
    ``apply_credentials_fn``) are forwarded as-is. Hard rule #5
    (secrets enter at flow entry only) means the *flow* constructs the
    closures over Prefect Blocks / env vars; this task shell never
    reads credentials directly.

    When ``use_s3_direct`` is set, this task resolves IAM credentials in the
    Dask **worker** process where the domain function's store writes and ROI
    reads actually run — the worker holds the IAM role, and the credentials
    (live access key / secret / STS token) stay off the orchestration
    boundary rather than crossing it as serialized parameters (Hard rule #5):

    1. Registers an IAM-resolving icechunk credential provider for the
       duration of the ingest call, so store writes keep using IAM-role
       creds after ``set_s3_credentials`` overwrites the ``AWS_*`` env vars
       with OPERA-scoped STS tokens. Scoped to a ``with`` block so a reused
       Dask worker is not left pinned to the IAM provider for later,
       unrelated icechunk opens.
    2. Resolves IAM ``storage_options`` for the ROI-mask reads, so those
       reads survive the same env-var overwrite.

    Confining the botocore-backed helpers to providers/aws/ (imported
    lazily) keeps the storage layer cloud-agnostic per the architecture
    rules.
    """
    cred_provider_cm: Any = nullcontext()
    if use_s3_direct:
        # Lazy import: providers.aws.credentials pulls in botocore, which lives
        # only in the optional `aws` extra. A prefect-only install (e.g. Prefect
        # on GCP) must be able to import this task module without it.
        from tessera_embeddings.providers.aws.credentials import (
            iam_icechunk_credentials,
            iam_s3_storage_options,
        )

        cred_provider_cm = credentials_provider(iam_icechunk_credentials)

        if storage_options is None and roi_zarr_path.startswith("s3://"):
            storage_options = iam_s3_storage_options()

    # manifest_split for the same reason as the S2 task: this mosaic is
    # region-written once per date batch, so an unsharded manifest makes each
    # commit rewrite every ref written so far.
    with cred_provider_cm, manifest_split(INGEST_MANIFEST_SPLIT):
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
            crop_to_live_windows=crop_to_live_windows,
            overlap_window_writes=overlap_window_writes,
            pipeline_batches=pipeline_batches,
            narrow_windows_per_date=narrow_windows_per_date,
        )
    return asdict(result)
