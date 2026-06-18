"""Prefect task shells for inference domain functions.

Two shells:

* :func:`run_inference_task` — wraps
  :func:`tessera_embeddings.inference.runner.run_inference`. Runs on
  the Prefect flow runner (no GPU); the runner connects to the Ray
  cluster the flow has already entered.
* :func:`assemble_embeddings_task` — wraps
  :class:`tessera_embeddings.inference.assembly.ZarrWriter` ``.assemble``.
  Runs on the Prefect flow runner; the assembly Dask cluster is
  managed by the calling flow.

Cluster lifecycle (Ray + Dask) is the *flow's* concern, not the task's.
That keeps each task shell thin enough to be obviously correct.
"""

from __future__ import annotations

import datetime
import logging
import time
from collections.abc import Callable
from typing import Any

from prefect import get_run_logger, task

from tessera_embeddings.config.inference import InferenceConfig, TimeWindow
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.chunk_spec import ChunkSpec, enumerate_chunks
from tessera_embeddings.inference.orchestration_helpers import (
    checkpoint_to_version,
    read_upstream_manifests,
)
from tessera_embeddings.inference.runner import run_inference
from tessera_embeddings.storage.manifest import EmbeddingManifest
from tessera_embeddings.storage.zarr_store import manifest_split


@task(name="run-inference")
def run_inference_task(
    *,
    num_actors: int,
    config: InferenceConfig,
    chunks: list[ChunkSpec],
    mosaic_base: str,
    staging_base: str,
    run_id: str,
    t0: float,
    on_actor_retire: Callable[[str], None] | None = None,
) -> list[dict]:
    """Prefect task: create Ray actors, run work-stealing inference.

    Caller must already be inside a Ray context (the flow's
    ``ray_cluster`` ctx manager). This task shell pulls the Prefect
    logger and delegates to :func:`run_inference`.
    """
    return run_inference(
        num_actors=num_actors,
        config=config,
        chunks=chunks,
        mosaic_base=mosaic_base,
        staging_base=staging_base,
        run_id=run_id,
        t0=t0,
        log=get_run_logger(),
        on_actor_retire=on_actor_retire,
    )


@task(name="assemble-embeddings")
def assemble_embeddings_task(
    *,
    chunk_size: int,
    n_live_chunks: int,
    total_y: int,
    total_x: int,
    run_id: str,
    staging_base: str,
    output_bucket: str,
    roi_name: str,
    roi_zarr_path: str,
    config: InferenceConfig,
    t0: float,
    n_workers: int,
    run_started_at: datetime.datetime | None = None,
    result_stats: dict | None = None,
    mosaic_base: str | None = None,
    time_window: TimeWindow | None = None,
    cleanup_staging: bool = True,
    output_name_suffix: str = "",
    get_credentials: Callable[[], Any] | None = None,
    s3_region: str | None = None,
) -> dict:
    """Prefect task: assemble staged chunks into the final Icechunk store.

    The Dask cluster is managed by the flow that calls this task — the
    task itself does no cluster provisioning. ``n_workers`` is the
    cluster's ``max_workers`` value, used by the assembler to divide
    the fleet-wide S3 concurrency budget across workers.

    The full chunk grid is reconstructed in-task from ``total_y``,
    ``total_x``, and ``chunk_size`` via :func:`enumerate_chunks` rather
    than passed in. The grid is purely a function of those three scalars,
    and shipping the materialized ``list[ChunkSpec]`` across the inner
    flow's parameter boundary blows Prefect's 524,288-byte flow-run
    parameter limit for large ROIs.

    Args:
        chunk_size: Square chunk edge length in pixels; the grid is
            re-enumerated from this plus the mosaic dimensions.
        n_live_chunks: Number of chunks intersecting the ROI mask.
        total_y: Mosaic height in pixels.
        total_x: Mosaic width in pixels.
        run_id: Run identifier.
        staging_base: Base path for staged chunk Zarrs.
        output_bucket: Base S3 path for the output store.
        roi_name: ROI identifier (used in the output filename).
        roi_zarr_path: Path to the ROI boolean zarr.
        config: Inference configuration.
        t0: Flow start time for elapsed logging.
        n_workers: Max Dask worker count for this assembly run.
        run_started_at: Flow trigger time for the time coordinate.
        results: Inference result dicts; ``None`` in assemble-only mode.
        mosaic_base: Base path for input mosaic stores. Used to copy
            projected coordinates and CRS from the reflectance store.
        time_window: 12-month inference window. Falls back to
            ``config.time_window``.
        result_stats: Pre-aggregated inference outcome counts
            (``succeeded``, ``skipped``, ``failed``, ``total_valid_pixels``);
            ``None`` in assemble-only mode. Aggregated in the calling flow
            so the per-chunk result dicts never cross the inner flow's
            parameter boundary (the 524,288-byte limit).
        cleanup_staging: If True, delete staged chunk zarrs after
            successful assembly. Disable for resumable dev runs.
        output_name_suffix: Optional suffix appended to the output
            filename (before ``.zarr``). Use to avoid clobbering when
            iterating on dev runs.
        get_credentials: Optional Icechunk credential callback (see
            :func:`tessera_embeddings.storage.zarr_store._create_storage`).
        s3_region: Optional S3 region override.
    """
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] = get_run_logger()

    chunks = enumerate_chunks(total_y, total_x, chunk_size)

    zarr_name = f"{roi_name}{output_name_suffix}"
    output_path = f"{output_bucket.rstrip('/')}/embeddings/{zarr_name}.zarr"
    log.info("Assembling %d chunks into %s", len(chunks), output_path)
    writer = ZarrWriter(staging_base)

    model_version = checkpoint_to_version(config.checkpoint_path)
    upstream_manifests = read_upstream_manifests(mosaic_base, config.s1_orbit) if mosaic_base else {}
    embedding_manifest = EmbeddingManifest.from_upstream_stores(
        model_checkpoint=model_version,
        num_obs_checkpoints=config.num_obs_checkpoints,
        upstream_manifests=upstream_manifests,
    )

    # Split each spatial axis's manifest into 32-chunk shards. Embeddings are
    # written in 500-px spatial chunks, so a 32-chunk shard is ~16k px/axis —
    # matching DEFAULT_MANIFEST_SPLIT_SIZES' ~16k-px target. No time split:
    # assembly only ever writes a single timestep, so one time shard == the
    # whole array and splitting time would be a no-op.
    with manifest_split({"northing": 32, "easting": 32}):
        writer.assemble(
            chunks,
            total_y,
            total_x,
            run_id,
            output_path,
            roi_zarr_path=roi_zarr_path,
            run_started_at=run_started_at,
            mosaic_base=mosaic_base,
            log=log,
            time_window=time_window or config.time_window,
            tile_id=roi_name,
            model_version=model_version,
            manifest=embedding_manifest,
            n_workers=n_workers,
            get_credentials=get_credentials,
            s3_region=s3_region,
        )

    if cleanup_staging:
        try:
            writer.cleanup_staging(run_id, log)
        except Exception:
            log.warning("Staging cleanup failed for run %s", run_id, exc_info=True)

    elapsed = time.monotonic() - t0
    total_pixels = result_stats["total_valid_pixels"] if result_stats else 0
    succeeded = result_stats["succeeded"] if result_stats else len(chunks)
    skipped = result_stats["skipped"] if result_stats else 0
    failed = result_stats["failed"] if result_stats else 0

    summary = {
        "run_id": run_id,
        "roi_name": roi_name,
        "n_live_chunks": n_live_chunks,
        "total_chunks": len(chunks),
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
        "total_valid_pixels": total_pixels,
        "output_path": output_path,
        "elapsed_sec": elapsed,
    }

    log.info("Embeddings assembly complete: %s", summary)
    return summary
