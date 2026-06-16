"""Master orchestration flow: ROI generation → S1/S2 ingestion → Tessera embeddings.

Chains the three pipeline stages and dispatches each to its registered
Prefect deployment via ``arun_deployment``. The two ingestion stages
run concurrently. Cancelling this flow in the Prefect UI propagates
cancellation to all running child deployments.

Differences from the reference repo:

* The coarsening stage is **not included** — the coarsen flow is out
  of OSS scope (plan §17). Downstream consumers can chain a coarsen
  step themselves.
* The deployment names are caller-supplied via :class:`PipelineDeployments`
  rather than module-level constants pinned to a private deployment
  layout.
* The ``BucketPaths`` parameter replaces ``resolve_buckets(dev)``; CRS
  suffix logic is preserved via :func:`tessera_embeddings.orchestration.prefect.flows.generate_roi._crs_suffix`.
"""

from __future__ import annotations

import asyncio

import zarr
from prefect import flow, get_run_logger
from prefect.deployments import arun_deployment
from prefect.states import StateType
from pydantic import BaseModel

from tessera_embeddings.config.dask import compute_pipeline_cluster_sizing
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.data_loading import _active_orbits
from tessera_embeddings.orchestration.prefect.flows.generate_roi import _crs_suffix


class PipelineDeployments(BaseModel):
    """Deployment refs (``flow_name/deployment_name``) for the master pipeline.

    The reference repo hardcodes deployment names as module constants;
    making them caller-configurable lets the same pipeline flow drive
    multiple environments (prod, dev, staging) without code changes.
    """

    generate_roi: str = "generate-roi/generate-roi"
    ingest_s1_roi_sar: str = "ingest_s1_roi_sar/ingest-s1-roi-sar"
    ingest_s2_roi_reflectance: str = "ingest_s2_roi_reflectance/ingest-s2-roi-reflectance"
    tessera_embeddings: str = "tessera_embeddings/tessera-embeddings"


def _count_roi_chunks(roi_zarr_path: str) -> int:
    """Count total spatial chunks in the ROI zarr."""
    import math

    z = zarr.open_array(roi_zarr_path, mode="r")
    height, width = z.shape
    chunk_h, chunk_w = z.chunks
    return math.ceil(height / chunk_h) * math.ceil(width / chunk_w)


def _check_completed(flow_run: object, stage: str) -> None:
    """Raise if a child flow run did not complete successfully."""
    state = getattr(flow_run, "state", None)
    if state is None or state.type != StateType.COMPLETED:
        state_name = state.name if state else "UNKNOWN"
        raise RuntimeError(f"{stage} did not complete successfully (state={state_name})")


@flow(name="tessera-full-pipeline", log_prints=True)
async def tessera_full_pipeline(
    *,
    paths: BucketPaths,
    time_window_end: str,
    deployments: PipelineDeployments = PipelineDeployments(),  # noqa: B008
    # ROI inputs
    roi_name: str | None = None,
    tile_names: str | None = None,
    roi_override_name: str | None = None,
    resolution: float = 10.0,
    force_crs: str | None = None,
    # Cluster sizing (None = auto-size from ROI chunk count)
    ingest_min_workers: int | None = None,
    ingest_max_workers: int | None = None,
    num_actors: int | None = None,
    # Behaviour
    s1_orbit: str = "ascending",
    skip_coverage_check: bool = False,
    ami_ssm_name: str = "/tessera/ray/ami-id",
    ssm_prefix: str = "/tessera/ray/",
    cloudwatch_log_group: str = "/ec2/tessera/ray",
    code_bucket: str | None = None,
    code_suffix: str = "",
    sync_source_path: str | None = None,
) -> dict:
    """End-to-end pipeline: ROI → S1+S2 ingestion → Tessera embeddings.

    Args:
        paths: Deployment-supplied storage URIs (see :class:`BucketPaths`).
        time_window_end: End month of the 12-month window as
            ``"Month Year"`` (e.g. ``"June 2025"``).
        deployments: Deployment refs to dispatch to.
        roi_name: GeoJSON-mode ROI name. Mutually exclusive with
            ``tile_names``.
        tile_names: Comma-separated MGRS tile IDs. Mutually exclusive
            with ``roi_name``.
        roi_override_name: Human-readable nickname in tile mode.
        resolution: ROI rasterisation pixel size (metres).
        force_crs: Optional CRS override for ROI generation.
        ingest_min_workers: Override for ingest Dask min_workers (None
            = auto-size from chunk count).
        ingest_max_workers: Override for ingest Dask max_workers.
        num_actors: Override for Ray GPU actor count.
        s1_orbit: SAR orbit direction — ``"ascending"``, ``"descending"``,
            or ``"both"``. ``"both"`` ingests both orbits concurrently.
        skip_coverage_check: Skip the time-window coverage validation
            on the embeddings stage.
        ami_ssm_name: SSM parameter name for the Ray GPU AMI ID.
        ssm_prefix: SSM Parameter Store prefix under which the Ray
            cluster resource IDs are published by the deployment's infra.
            Forwarded to the embeddings stage; deployments that publish
            under a different prefix must override the OSS ``/tessera/ray/``.
        cloudwatch_log_group: CloudWatch log group the Ray workers write
            agent logs to. Forwarded to the embeddings stage; must match
            the group the deployment's infra creates and grants access to.
        code_bucket: S3 bucket (no ``s3://`` prefix) workers pull the
            source tarball from. Setting it only points workers at an
            existing tarball (expected to be uploaded by CI — the general
            production path); it does not upload one. Forwarded to the
            embeddings stage; leave ``None`` for AMI-baked source.
        code_suffix: Source tarball filename suffix. Forwarded to the
            embeddings stage.
        sync_source_path: Dev-iteration only. Local source dir to tar and
            upload before ``ray up`` (requires ``code_bucket``), so
            workers run your working-tree code without a CI round-trip.
            Forwarded into the embeddings stage's ``dev_params``.

    Returns:
        Dict with the run IDs of every child flow.
    """
    log = get_run_logger()

    _active_orbits(s1_orbit)  # validates

    inputs_bucket = paths.inputs

    time_window = parse_time_window(time_window_end)
    start_date, end_date = time_window.to_date_range()
    log.info("Time window: %s → %s (end month: %s)", start_date, end_date, time_window_end)

    roi_name = roi_name.strip() if roi_name else None
    tile_names = tile_names.strip() if tile_names else None
    if (roi_name and tile_names) or not (roi_name or tile_names):
        raise ValueError("Exactly one of roi_name or tile_names is required")

    if roi_name:
        canonical_name = roi_name
    else:
        parts = [t.strip() for t in tile_names.split(",") if t.strip()]  # type: ignore[union-attr]
        if not parts:
            raise ValueError("tile_names must contain at least one non-empty tile ID")
        canonical_name = roi_override_name or "_".join(parts)

    # Stage 1: Generate ROI
    log.info("Stage 1: Generating ROI mask")
    roi_run = await arun_deployment(
        deployments.generate_roi,
        parameters={
            "roi_bucket": f"{inputs_bucket.rstrip('/')}/rois",
            "roi_name": roi_name,
            "tile_names": tile_names,
            "output_name": roi_override_name,
            "resolution": resolution,
            "force_crs": force_crs,
        },
    )
    _check_completed(roi_run, "generate_roi")

    # The CRS suffix is part of the ROI's canonical identity, not just the
    # ROI-zarr filename: it must thread through the mosaic dir and the
    # downstream embeddings roi_name too, or `store_for(roi_name, "roi")` in
    # the embeddings flow rebuilds an unsuffixed path that doesn't exist (and
    # CRS variants would collide on the mosaic/embeddings paths). generate_roi
    # stays the sole place that *derives* the suffix; everything after it uses
    # this suffixed id uniformly. force_crs=None → roi_id == canonical_name.
    roi_id = f"{canonical_name}{_crs_suffix(force_crs)}"
    roi_zarr_path = f"{inputs_bucket.rstrip('/')}/rois/zarrs/{roi_id}.zarr"
    store_path = f"{inputs_bucket.rstrip('/')}/mosaics/{roi_id}"
    log.info("ROI generated: %s (run_id=%s)", roi_zarr_path, roi_run.id)

    # Auto-size cluster from ROI chunk count
    n_chunks = _count_roi_chunks(roi_zarr_path)
    ingest_min_workers, ingest_max_workers, num_actors = compute_pipeline_cluster_sizing(
        n_chunks,
        ingest_min_workers=ingest_min_workers,
        ingest_max_workers=ingest_max_workers,
        num_actors=num_actors,
    )
    log.info(
        "ROI has %d chunks → ingest workers %d/%d, actors %d",
        n_chunks,
        ingest_min_workers,
        ingest_max_workers,
        num_actors,
    )

    # Stage 2: Concurrent S1 + S2 ingestion
    log.info("Stage 2: Ingesting S1 SAR + S2 reflectance concurrently")
    ingest_params_common = {
        "roi_zarr_path": roi_zarr_path,
        "start_date": start_date,
        "end_date": end_date,
        "store_path": store_path,
        "min_workers": ingest_min_workers,
        "max_workers": ingest_max_workers,
    }
    s1_orbits_to_ingest = _active_orbits(s1_orbit)
    s1_coros = [
        arun_deployment(
            deployments.ingest_s1_roi_sar,
            parameters={**ingest_params_common, "orbit": orbit},
        )
        for orbit in s1_orbits_to_ingest
    ]
    s2_coro = arun_deployment(deployments.ingest_s2_roi_reflectance, parameters=ingest_params_common)
    results = await asyncio.gather(*s1_coros, s2_coro)
    *s1_runs, s2_run = results
    for orbit, s1_run in zip(s1_orbits_to_ingest, s1_runs, strict=True):
        _check_completed(s1_run, f"ingest_s1_roi_sar ({orbit})")
        log.info("S1 %s ingestion complete (run_id=%s)", orbit, s1_run.id)
    _check_completed(s2_run, "ingest_s2_roi_reflectance")
    log.info("S2 ingestion complete (run_id=%s)", s2_run.id)

    # Stage 3: Tessera embeddings
    log.info("Stage 3: Running Tessera embedding inference")
    embeddings_run = await arun_deployment(
        deployments.tessera_embeddings,
        parameters={
            "roi_name": roi_id,
            "time_window_end": time_window_end,
            "paths": paths.model_dump(),
            "ami_ssm_name": ami_ssm_name,
            "ssm_prefix": ssm_prefix,
            "cloudwatch_log_group": cloudwatch_log_group,
            "code_bucket": code_bucket,
            "code_suffix": code_suffix,
            "num_actors": num_actors,
            "s1_orbit": s1_orbit,
            "dev_params": {
                "skip_coverage_check": skip_coverage_check,
                "sync_source_path": sync_source_path,
            },
        },
    )
    _check_completed(embeddings_run, "tessera_embeddings")
    log.info("Embeddings complete (run_id=%s)", embeddings_run.id)

    return {
        "roi_run_id": str(roi_run.id),
        "s1_run_ids": [str(r.id) for r in s1_runs],
        "s2_run_id": str(s2_run.id),
        "embeddings_run_id": str(embeddings_run.id),
    }
