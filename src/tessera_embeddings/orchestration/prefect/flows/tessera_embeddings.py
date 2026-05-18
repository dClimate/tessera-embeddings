"""Tessera embedding generation flow.

Orchestrates distributed GPU inference via Ray to generate
128-dimensional per-pixel embeddings from mosaicked Icechunk/Zarr
stores (S2 reflectance + S1 SAR).

Architecture::

    Prefect flow runner (no GPU)
      ├── Read Zarr metadata → enumerate spatial chunk grid
      ├── Pre-filter chunks against the ROI mask
      ├── Spin up Ray cluster (head on-demand, GPU workers configurable)
      ├── Submit chunk work to Ray GPU actors via run_inference_task
      ├── Spin up Dask cluster, run assemble_embeddings_task
      └── Tear down both clusters; on-cancellation hook covers partials

The ``BucketPaths`` parameter is the deployment-supplied storage
contract — there is no ``dev: bool`` toggle. Callers (typically a
Prefect deployment with parameters) construct paths once at flow
boundary and the rest of the code is path-agnostic.
"""

from __future__ import annotations

import datetime
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from prefect import flow, get_run_logger
from pydantic import BaseModel

from tessera_embeddings.config.dask import AssemblyConfig
from tessera_embeddings.config.inference import DEFAULT_CHUNK_SIZE, checkpoint_filename
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.chunk_spec import filter_chunks_by_roi_mask
from tessera_embeddings.inference.data_loading import check_time_window_coverage
from tessera_embeddings.inference.orchestration_helpers import (
    build_inference_config,
    compute_assembly_worker_counts,
    enumerate_mosaic_chunks,
)
from tessera_embeddings.orchestration.prefect.flows._dask_runner import get_task_runner_for_cluster
from tessera_embeddings.orchestration.prefect.tasks.inference import (
    assemble_embeddings_task,
    run_inference_task,
)
from tessera_embeddings.providers.aws.ray import (
    cleanup_ray_tempfiles,
    terminate_ray_instances_by_tag,
)

# Module-level state for the cancellation hook. The flow body sets these
# on entry and clears them on normal exit; the hook reads them.
_active_resolved_yaml: str | None = None
_active_cluster_name: str | None = None


class EmbeddingsDevParams(BaseModel):
    """Development-mode toggles for the embeddings flow.

    Grouping them in a Pydantic model makes them legible in the Prefect
    Cloud UI parameter form.
    """

    assembly_only: bool = False
    inference_only: bool = False
    previous_run_id: str | None = None
    skip_coverage_check: bool = False
    cleanup_staging: bool = True
    output_name_suffix: str = ""


def _ray_cleanup_on_cancellation(flow: object, flow_run: object, state: object) -> None:  # noqa: ARG001
    """Emergency teardown when the flow is cancelled via the Prefect UI.

    Reads the resolved YAML path from module state and runs ``ray down``.
    Falls back to terminating instances by EC2 tag when the resolved
    YAML is unavailable (e.g. cancelled before ``ray up`` completed).
    """
    log = logging.getLogger(__name__)
    log.warning("Flow cancelled — tearing down Ray cluster")

    if _active_resolved_yaml and Path(_active_resolved_yaml).exists():
        log.info("Running ray down with %s", _active_resolved_yaml)
        subprocess.run(["ray", "down", _active_resolved_yaml, "-y"], check=False)
        cleanup_ray_tempfiles(_active_resolved_yaml)
    elif _active_cluster_name:
        log.warning("No resolved YAML — terminating instances for cluster '%s'", _active_cluster_name)
        terminate_ray_instances_by_tag(cluster_name=_active_cluster_name, log=log)
    else:
        log.warning(
            "Cancellation fired before cluster was provisioned — "
            "no YAML or cluster name available. Check the AWS console manually."
        )


@flow(name="tessera_embeddings", on_cancellation=[_ray_cleanup_on_cancellation])
def tessera_embeddings(
    *,
    roi_name: str,
    time_window_end: str,
    paths: BucketPaths,
    ami_ssm_name: str,
    num_actors: int = 20,
    s1_orbit: str = "ascending",
    dev_params: EmbeddingsDevParams = EmbeddingsDevParams(),  # noqa: B008
) -> dict[str, Any]:
    """Generate Tessera embeddings for a mosaicked ROI.

    Args:
        roi_name: ROI identifier (used to derive both the ROI Zarr path
            and the output filename).
        time_window_end: End month of the 12-month window as
            ``"Month Year"`` (e.g. ``"June 2025"``).
        paths: Deployment-supplied storage URIs (see
            :class:`BucketPaths`). Replaces the reference repo's
            ``dev: bool`` toggle.
        ami_ssm_name: SSM parameter name for the Ray GPU AMI ID.
        num_actors: Number of GPU actors to create.
        s1_orbit: ``"ascending"``, ``"descending"``, or ``"both"``.
        dev_params: See :class:`EmbeddingsDevParams`.

    Returns:
        Summary dict with run_id, chunk counts, timing, and output path.
    """
    log = get_run_logger()

    inputs_bucket = paths.inputs
    output_bucket = paths.outputs
    roi_zarr_path = paths.store_for(roi_name, "roi")

    if dev_params.assembly_only and dev_params.inference_only:
        raise ValueError("Only one of assembly_only, inference_only can be True")
    if dev_params.assembly_only and not dev_params.previous_run_id:
        raise ValueError("assembly_only=True requires previous_run_id")

    run_id = dev_params.previous_run_id or uuid.uuid4().hex[:12]
    run_started_at = datetime.datetime.now(datetime.UTC)
    t0 = time.monotonic()

    time_window = parse_time_window(time_window_end)
    log.info(
        "Time window: %d-%02d through %d-%02d inclusive. Output time label: %s",
        time_window.months[0][0],
        time_window.months[0][1],
        time_window.months[-1][0],
        time_window.months[-1][1],
        time_window.window_end_label,
    )

    checkpoint_path = f"{inputs_bucket.rstrip('/')}/models/{checkpoint_filename()}"

    config = build_inference_config(
        s1_orbit=s1_orbit,
        time_window=time_window,
        checkpoint_path=checkpoint_path,
        inputs_bucket=inputs_bucket,
        output_bucket=output_bucket,
    )

    mosaic_base = f"{inputs_bucket.rstrip('/')}/mosaics/{roi_name}"
    log.info("Starting tessera_embeddings: roi=%s, mosaic_base=%s, run_id=%s", roi_name, mosaic_base, run_id)

    staging_base = f"{output_bucket.rstrip('/')}/staging"

    # Detect staged chunk size from prior runs — chunk_size may differ
    # between a resumed run and the current config.
    chunk_size = config.chunk_size
    if dev_params.previous_run_id:
        detected = ZarrWriter(staging_base).detect_staged_chunk_size(dev_params.previous_run_id)
        if detected != chunk_size:
            log.warning(
                "Staged chunks use chunk_size=%d (current config: %d) — using staged value", detected, chunk_size
            )
            chunk_size = detected

    chunks, total_y, total_x = enumerate_mosaic_chunks(mosaic_base, chunk_size or DEFAULT_CHUNK_SIZE, log)

    live_chunks = filter_chunks_by_roi_mask(chunks, roi_zarr_path)
    log.info(
        "ROI filter: %d/%d chunks intersect the ROI, sending %d to GPU actors",
        len(live_chunks),
        len(chunks),
        len(live_chunks),
    )

    check_time_window_coverage(
        mosaic_base, time_window, s1_orbit=config.s1_orbit, skip_coverage_check=dev_params.skip_coverage_check
    )

    # Lazily import the AWS Ray provider so the embeddings flow file
    # can be inspected (for arch tests) on machines without ray
    # installed. The provider is only needed when the flow actually
    # runs.
    from tessera_embeddings.providers.aws.ray import ray_cluster

    assemble_kwargs: dict[str, Any] = {
        "n_live_chunks": len(live_chunks),
        "total_y": total_y,
        "total_x": total_x,
        "run_id": run_id,
        "staging_base": staging_base,
        "output_bucket": output_bucket,
        "roi_name": roi_name,
        "roi_zarr_path": roi_zarr_path,
        "config": config,
        "t0": t0,
        "run_started_at": run_started_at,
        "mosaic_base": mosaic_base,
        "time_window": time_window,
        "cleanup_staging": dev_params.cleanup_staging,
        "output_name_suffix": dev_params.output_name_suffix,
    }

    if dev_params.assembly_only:
        log.info("Assembly-only mode: verifying staged chunks from run %s", dev_params.previous_run_id)
        ZarrWriter(staging_base).verify_staged_completeness(run_id, live_chunks, log=log)
        return _run_assembly(log=log, chunks=chunks, results=None, **assemble_kwargs)

    global _active_resolved_yaml, _active_cluster_name
    with ray_cluster(log, ami_ssm_name=ami_ssm_name) as resolved_yaml:
        _active_resolved_yaml = resolved_yaml
        if resolved_yaml and Path(resolved_yaml).exists():
            with Path(resolved_yaml).open() as _f:
                _active_cluster_name = yaml.safe_load(_f).get("cluster_name")

        results = run_inference_task(
            num_actors=num_actors,
            config=config,
            chunks=live_chunks,
            mosaic_base=mosaic_base,
            staging_base=staging_base,
            run_id=run_id,
            t0=t0,
        )
    _active_resolved_yaml = None
    _active_cluster_name = None

    succeeded = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]
    log.info(
        "Chunk results: %d succeeded, %d skipped, %d failed", len(succeeded), len(skipped), len(failed)
    )
    if failed:
        for failure in failed:
            log.error(
                "Failed chunk %s (instance %s): %s",
                failure["chunk"],
                failure.get("instance_id", "unknown"),
                failure.get("error", "unknown"),
            )
        msg = f"{len(failed)} chunks failed"
        raise RuntimeError(msg)

    if dev_params.inference_only:
        elapsed = time.monotonic() - t0
        return {
            "run_id": run_id,
            "roi_name": roi_name,
            "total_chunks": len(chunks),
            "live_chunks": len(live_chunks),
            "succeeded": len(succeeded),
            "skipped": len(skipped),
            "failed": len(failed),
            "staging_base": staging_base,
            "elapsed_sec": elapsed,
            "inference_only": True,
        }

    ZarrWriter(staging_base).verify_staged_completeness(run_id, live_chunks, log=log)
    return _run_assembly(log=log, chunks=chunks, results=results, **assemble_kwargs)


def _run_assembly(
    *,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    chunks: list,
    results: list | None,
    **assemble_kwargs: Any,  # noqa: ANN401 — pass-through to the assembly task
) -> dict[str, Any]:
    """Provision the assembly Dask cluster and submit the assembly task."""
    n_live_chunks = assemble_kwargs["n_live_chunks"]
    min_workers, max_workers = compute_assembly_worker_counts(n_live_chunks, AssemblyConfig())

    from tessera_embeddings.providers.aws.dask import ecs_cluster

    extra_worker_env = {
        "AWS_NO_SIGN_REQUEST": "NO",  # use signed requests for the project's S3 bucket
        "MALLOC_TRIM_THRESHOLD_": "0",  # eagerly return freed memory to the OS
    }
    with ecs_cluster(
        log,
        min_workers=min_workers,
        max_workers=max_workers,
        extra_worker_env=extra_worker_env,
    ) as cluster:
        log.info("Assembly Dask cluster ready: scaling to %d workers", max_workers)
        task_runner = get_task_runner_for_cluster(cluster.scheduler_address)
        return _assemble_inner.with_options(task_runner=task_runner)(  # type: ignore[arg-type]
            chunks=chunks,
            results=results,
            n_workers=max_workers,
            **assemble_kwargs,
        )


@flow(name="tessera_embeddings_assemble_inner")
def _assemble_inner(
    *,
    chunks: list,
    results: list | None,
    n_workers: int,
    **assemble_kwargs: Any,  # noqa: ANN401 — pass-through to the assembly task
) -> dict[str, Any]:
    """Inner flow that submits the assembly task to the configured Dask runner."""
    future = assemble_embeddings_task.submit(
        chunks=chunks,
        results=results,
        n_workers=n_workers,
        **assemble_kwargs,
    )
    return future.result()
