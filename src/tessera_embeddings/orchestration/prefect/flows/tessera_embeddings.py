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
from prefect.runtime import flow_run as flow_run_ctx
from pydantic import BaseModel

from tessera_embeddings.config.dask import AssemblyConfig
from tessera_embeddings.config.inference import (
    DEFAULT_MODEL_VERSION,
    INFERENCE_CHUNK_SIZE,
    ModelVersion,
    checkpoint_filename,
)
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.chunk_spec import filter_chunks_by_roi_mask
from tessera_embeddings.inference.data_loading import check_time_window_coverage, resolve_s1_orbit
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
    DEFAULT_CLUSTER_TEMPLATE,
    cleanup_ray_tempfiles,
    terminate_ray_instances_by_tag,
)

# Module-level state for the cancellation hook. The flow body sets these
# on entry and clears them on normal exit; the hook reads them.
_active_resolved_yaml: str | None = None
_active_cluster_name: str | None = None


#: Staging-run prefix marking chunks produced by a v2 student. v1.1 keeps the
#: historical bare-uuid run_id, so nothing about the existing path changes.
V2_RUN_PREFIX = "v2-"


def _resolve_run_id(previous_run_id: str | None, *, model_version: ModelVersion) -> str:
    """Derive the staging run_id, encoding which model version produced it.

    ``run_inference()`` reuses staged ``.zarr``/``.skipped`` artifacts by run_id
    alone, and a staged chunk is (H, W, 128) int8 whichever student wrote it —
    v1.1 saves the first 128 of its 192-d representation, v2 emits 128 natively.
    So nothing about the artifacts distinguishes them. Since ``model_version``
    defaults to v1.1, a resume that simply omits it would finish a v2 run with
    the v1.1 encoder, publish a mix, and stamp the store ``geoemb:model`` = 1.1.

    Encoding the version in the run_id — which namespaces the staging directory —
    lets a resume detect the flip and refuse. Fresh v1.1 runs keep the historical
    bare-uuid form, so the single-model path is untouched.

    Unlike the S2-only mode this mirrors, **assembly-only resumes are not
    exempt**: assembly is what writes the provenance attrs, so an assembly-only
    pass under the wrong version is precisely the case that mislabels a store.
    """
    if previous_run_id:
        resumed_v2 = previous_run_id.startswith(V2_RUN_PREFIX)
        wants_v2 = model_version != "v1.1"
        if resumed_v2 != wants_v2:
            staged = "v2" if resumed_v2 else "v1.1"
            raise ValueError(
                f"Cannot resume run {previous_run_id!r} (staged by {staged}) with "
                f"model_version={model_version!r}: its staged chunks came from the other "
                f"encoder, and continuing would publish a mix of both and stamp the store "
                f"with one of them. Resume with the {staged} model, or start a fresh run."
            )
        return previous_run_id
    return (V2_RUN_PREFIX if model_version != "v1.1" else "") + uuid.uuid4().hex[:12]


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
    # Dev-iteration escape hatch. When set (requires code_bucket), the provider
    # tars this local dir and uploads it to s3://{code_bucket}/code/... *before*
    # ``ray up``, so workers run your working-tree code without a CI round-trip.
    # None (default) = no upload: workers use AMI-baked source, or a tarball a
    # CI workflow already put in code_bucket. See providers/aws/gotchas.md.
    sync_source_path: str | None = None


def _cluster_name_for_flow_run(flow_run_id: object) -> str:
    """Derive the deterministic Ray cluster name for a flow run.

    The name must be recomputable from nothing but the flow-run id:
    Prefect executes cancellation/crash hooks in a freshly imported copy
    of this module after the flow's child process has been killed, so any
    state the hook needs has to be derivable, not stored. The base name
    comes from the shipped cluster template so the two stay in sync.
    """
    with DEFAULT_CLUSTER_TEMPLATE.open() as f:
        base = yaml.safe_load(f).get("cluster_name", "tessera-inference")
    return f"{base}-{str(flow_run_id).replace('-', '')[:8]}"


def _ray_cleanup_on_cancellation(flow: object, flow_run: object, state: object) -> None:  # noqa: ARG001
    """Emergency teardown when the flow is cancelled or crashes.

    Prefect runs these hooks in a FRESH import of this module after the
    flow's child process has been killed, so the module globals set by
    the flow body are normally ``None`` here and ``ray down`` (which
    needs the dead process's resolved YAML tempfile) is normally
    impossible. The authoritative path is therefore tag-based: re-derive
    the cluster name from the flow-run id and terminate every instance
    carrying its ``ray-cluster-name`` tag. The YAML fast path is kept for
    the rare same-process case; both paths are idempotent.
    """
    log = logging.getLogger(__name__)
    log.warning("Flow cancelled/crashed — tearing down Ray cluster")

    if _active_resolved_yaml and Path(_active_resolved_yaml).exists():
        log.info("Running ray down with %s", _active_resolved_yaml)
        subprocess.run(["ray", "down", _active_resolved_yaml, "-y"], check=False)
        cleanup_ray_tempfiles(_active_resolved_yaml)

    run_id = getattr(flow_run, "id", None)
    cluster_name = _active_cluster_name or (_cluster_name_for_flow_run(run_id) if run_id else None)
    if cluster_name:
        log.warning("Terminating instances for cluster '%s'", cluster_name)
        terminate_ray_instances_by_tag(cluster_name=cluster_name, log=log)
    else:
        log.warning(
            "No flow-run id or cluster name available — cannot derive the cluster tag. Check the AWS console manually."
        )


@flow(
    name="tessera_embeddings",
    on_cancellation=[_ray_cleanup_on_cancellation],
    on_crashed=[_ray_cleanup_on_cancellation],
)
def tessera_embeddings(
    *,
    roi_name: str,
    time_window_end: str,
    paths: BucketPaths,
    ami_ssm_name: str,
    ssm_prefix: str = "/tessera/ray/",
    cloudwatch_log_group: str = "/ec2/tessera/ray",
    code_bucket: str | None = None,
    code_suffix: str = "",
    num_actors: int = 20,
    s1_orbit: str = "both",
    model_version: ModelVersion = DEFAULT_MODEL_VERSION,
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
        ssm_prefix: SSM Parameter Store prefix under which the Ray
            cluster resource IDs (security group, instance profile,
            subnets, key pair) are published by the deployment's infra.
            Defaults to the OSS ``/tessera/ray/``; deployments that
            publish under a different prefix must override this.
        cloudwatch_log_group: CloudWatch log group the Ray workers write
            agent logs to. Must match the group the deployment's infra
            creates and grants the worker role access to.
        code_bucket: S3 bucket (no ``s3://`` prefix) workers pull the
            source tarball from, at
            ``s3://{code_bucket}/code/src{code_suffix}.tar.gz``. Setting
            this only *points* workers at the tarball — it does not
            upload one; an external/CI workflow is expected to have put
            it there (the general production path when source ships as an
            S3 artifact). Leave ``None`` for AMI-baked source. See
            ``dev_params.sync_source_path`` to also upload from the local
            tree. See ``providers/aws/gotchas.md`` for the three modes.
        code_suffix: Filename suffix for the source tarball (e.g.
            ``"-mybranch"``, letting branches coexist in one bucket).
            Empty for production tarballs.
        num_actors: Number of GPU actors to create.
        s1_orbit: ``"ascending"``, ``"descending"``, or ``"both"``.
        model_version: Which Tessera model to run — ``"v1.1"`` (default) or
            ``"v2-large"``. Also selects the checkpoint filename expected under
            ``{inputs}/models/``.
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

    run_id = _resolve_run_id(dev_params.previous_run_id, model_version=model_version)
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

    checkpoint_path = f"{inputs_bucket.rstrip('/')}/models/{checkpoint_filename(model_version=model_version)}"

    mosaic_base = f"{inputs_bucket.rstrip('/')}/mosaics/{roi_name}"
    log.info("Starting tessera_embeddings: roi=%s, mosaic_base=%s, run_id=%s", roi_name, mosaic_base, run_id)

    resolved_s1_orbit = resolve_s1_orbit(mosaic_base, s1_orbit)
    if resolved_s1_orbit != s1_orbit:
        log.info("s1_orbit resolved: %s → %s", s1_orbit, resolved_s1_orbit)

    config = build_inference_config(
        s1_orbit=resolved_s1_orbit,
        time_window=time_window,
        checkpoint_path=checkpoint_path,
        inputs_bucket=inputs_bucket,
        output_bucket=output_bucket,
        model_version=model_version,
    )

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

    chunks, total_y, total_x = enumerate_mosaic_chunks(mosaic_base, chunk_size or INFERENCE_CHUNK_SIZE, log)

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
        "chunk_size": chunk_size or INFERENCE_CHUNK_SIZE,
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
        return _run_assembly(log=log, result_stats=None, **assemble_kwargs)

    global _active_resolved_yaml, _active_cluster_name
    # Deterministic, flow-run-derived cluster name so the cancellation/crash
    # hook can re-derive it in a fresh process (see _cluster_name_for_flow_run).
    # Outside a Prefect run (unit tests) the id is None and ray_cluster falls
    # back to its own random suffix.
    with ray_cluster(
        log,
        ami_ssm_name=ami_ssm_name,
        cluster_name=_cluster_name_for_flow_run(flow_run_ctx.id) if flow_run_ctx.id else None,
        ssm_prefix=ssm_prefix,
        cloudwatch_log_group=cloudwatch_log_group,
        code_bucket=code_bucket,
        code_suffix=code_suffix,
        sync_source_path=Path(dev_params.sync_source_path) if dev_params.sync_source_path else None,
    ) as resolved_yaml:
        _active_resolved_yaml = resolved_yaml
        if resolved_yaml and Path(resolved_yaml).exists():
            with Path(resolved_yaml).open() as _f:
                _active_cluster_name = yaml.safe_load(_f).get("cluster_name")

        from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials

        results = run_inference_task(
            num_actors=num_actors,
            config=config,
            chunks=live_chunks,
            mosaic_base=mosaic_base,
            staging_base=staging_base,
            run_id=run_id,
            t0=t0,
            get_credentials=iam_icechunk_credentials,
        )
    _active_resolved_yaml = None
    _active_cluster_name = None

    succeeded = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]
    log.info("Chunk results: %d succeeded, %d skipped, %d failed", len(succeeded), len(skipped), len(failed))
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
    result_stats = {
        "succeeded": len(succeeded),
        "skipped": len(skipped),
        "failed": len(failed),
        "total_valid_pixels": sum(r.get("valid_pixels", 0) for r in results),
    }
    return _run_assembly(log=log, result_stats=result_stats, **assemble_kwargs)


def _run_assembly(
    *,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    result_stats: dict | None,
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
        # 24 GiB for headroom on the assembly commit, which is memory intensive:
        # merging the write changeset and building the manifest for a full-
        # spatial-extent timestep peaks well above the ingest workers' needs.
        # 24576 MiB is a valid Fargate combo at 4 vCPU.
        worker_mem=24576,
        extra_worker_env=extra_worker_env,
    ) as cluster:
        log.info("Assembly Dask cluster ready: scaling to %d workers", max_workers)
        task_runner = get_task_runner_for_cluster(cluster.scheduler_address)
        return _assemble_inner.with_options(task_runner=task_runner)(  # type: ignore[arg-type]
            result_stats=result_stats,
            n_workers=max_workers,
            **assemble_kwargs,
        )


@flow(name="tessera_embeddings_assemble_inner")
def _assemble_inner(
    *,
    result_stats: dict | None,
    n_workers: int,
    **assemble_kwargs: Any,  # noqa: ANN401 — pass-through to the assembly task
) -> dict[str, Any]:
    """Inner flow that submits the assembly task to the configured Dask runner.

    The full chunk grid is intentionally *not* a parameter here — the
    task re-enumerates it from ``total_y``/``total_x``/``chunk_size`` (in
    ``assemble_kwargs``). Likewise the per-chunk inference results are
    pre-aggregated into ``result_stats`` upstream. Both keep this flow's
    serialized parameters under Prefect's 524,288-byte limit on large ROIs.
    """
    future = assemble_embeddings_task.submit(
        result_stats=result_stats,
        n_workers=n_workers,
        **assemble_kwargs,
    )
    return future.result()
