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
      ├── Run assemble_embeddings_task on the flow runner (worker processes)
      └── Tear down the Ray cluster; on-cancellation hook covers partials

The ``BucketPaths`` parameter is the deployment-supplied storage
contract — there is no ``dev: bool`` toggle. Callers (typically a
Prefect deployment with parameters) construct paths once at flow
boundary and the rest of the code is path-agnostic.
"""

from __future__ import annotations

import datetime
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger
from prefect.runtime import flow_run as flow_run_ctx
from pydantic import BaseModel

from tessera_embeddings.config.assembly import AssemblyConfig
from tessera_embeddings.config.inference import INFERENCE_CHUNK_SIZE, checkpoint_filename
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.assembly import AllChunksSkippedError, ZarrWriter
from tessera_embeddings.inference.chunk_spec import filter_chunks_by_roi_mask
from tessera_embeddings.inference.data_loading import check_time_window_coverage, resolve_s1_orbit
from tessera_embeddings.inference.orchestration_helpers import (
    assert_output_store_accepts,
    build_inference_config,
    enumerate_mosaic_chunks,
)
from tessera_embeddings.orchestration.prefect.flows._ray_lifecycle import (
    activate,
    deactivate,
    ray_cleanup_on_cancellation,
)
from tessera_embeddings.orchestration.prefect.tasks.inference import (
    assemble_embeddings_task,
    run_inference_task,
)
from tessera_embeddings.storage.zarr_store import plain_zarr_storage_options

# Fresh runs with allow_s2_only=True carry this run_id prefix. The run_id
# namespaces the staging directory, so the prefix records the per-pixel S1
# mode that produced a run's staged chunks — letting a resume detect (and
# refuse) a mode flip that would otherwise mix S1-gated and S2-only chunks in
# one assembled output. Default runs keep the historical bare-uuid run_id, so
# the single-ROI path is byte-for-byte unchanged when the flag is off.
S2_ONLY_RUN_PREFIX = "s2only-"


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


def _resolve_run_id(previous_run_id: str | None, *, allow_s2_only: bool, assembly_only: bool) -> str:
    """Derive the staging run_id, encoding the S2-only mode via S2_ONLY_RUN_PREFIX.

    ``run_inference()`` reuses staged ``.zarr``/``.skipped`` artifacts by run_id
    alone. If a resume flipped ``allow_s2_only``, old skip markers (S2-valid /
    zero-S1 pixels the flag now embeds) would be kept for already-staged chunks
    while only the not-yet-staged chunks were recomputed under the new gate, and
    assembly would publish a mix of S1-gated and S2-only tiles. Encoding the mode
    in the run_id — which namespaces the staging directory — lets a resume detect
    and refuse a mode flip. Fresh default-mode runs keep the historical bare-uuid
    run_id, so the single-ROI path is unchanged when the flag is off.

    Assembly-only resumes never run the per-pixel gate, so the requested flag cannot
    change WHICH pixels they publish — but it still reaches the ``EmbeddingManifest``
    they write, so exempting them outright let a staged S2-only run be published as
    flag-off, after which a later flag-off append mixes incompatible slices into one
    store. :func:`staged_s2_only_mode` is what the caller uses instead: the prefix is the
    record of what the staged pixels ARE, so the mode is read off it rather than trusted
    from a parameter nobody had to set.
    """
    if previous_run_id:
        resumed_s2_only = previous_run_id.startswith(S2_ONLY_RUN_PREFIX)
        if not assembly_only and resumed_s2_only != allow_s2_only:
            raise ValueError(
                f"Cannot resume run {previous_run_id!r} (allow_s2_only={resumed_s2_only}) with "
                f"allow_s2_only={allow_s2_only}: its staged chunks were produced under the other "
                f"per-pixel S1 mode, so continuing would publish a mix of S1-gated and S2-only "
                f"tiles. Resume with allow_s2_only={resumed_s2_only}, or start a fresh run."
            )
        return previous_run_id
    return (S2_ONLY_RUN_PREFIX if allow_s2_only else "") + uuid.uuid4().hex[:12]


def _assert_resume_mode_matches(previous_run_id: str | None, *, allow_s2_only: bool, assembly_only: bool) -> None:
    """The mode-mixing refusal alone, callable before any store is opened.

    :func:`_resolve_run_id` has to run LATE, because the flag it must encode is the
    config's effective one and that is not known until the orbit has been resolved
    against the mosaic. But a resume whose staged mode already contradicts the REQUEST
    can be refused straight away, and should be: the whole point of this check is that it
    costs nothing, and a run that will refuse regardless should not open a store first.
    """
    _resolve_run_id(previous_run_id, allow_s2_only=allow_s2_only, assembly_only=assembly_only)


def staged_s2_only_mode(previous_run_id: str | None) -> bool:
    """Whether a staged run's chunks were produced under the S2-only pixel gate.

    The run_id prefix is the durable record of that, and on an ASSEMBLY-ONLY resume it is
    the only one: nothing recomputes the pixels, so the flow parameter says nothing about
    what is in the staging directory. Publishing on the parameter alone wrote a manifest
    claiming S1-gated pixels over S2-only ones, and a later flag-off append then mixed
    two policies into one store with nothing objecting.
    """
    return bool(previous_run_id and previous_run_id.startswith(S2_ONLY_RUN_PREFIX))


@flow(
    name="tessera_embeddings",
    # Both lists hold the SAME function: a crashed run leaks exactly like a
    # cancelled one. Keep the hook IDEMPOTENT — cancelling a parent and its child
    # together delivers the transition twice and runs it twice (2026-07-25).
    on_cancellation=[ray_cleanup_on_cancellation],
    on_crashed=[ray_cleanup_on_cancellation],
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
    require_s1: bool = True,
    s3_region: str | None = None,
    allow_s2_only: bool = False,
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
        require_s1: Demand radar rather than request it, and **True by default here**
            because this flow fills ONE cell: an operator naming a single zone-year over
            terrain that should be imaged wants to be told when its radar is missing, not
            to receive optical-only embeddings quietly. Set False for a cell that is
            genuinely radar-free — parts of the globe have no dual-pol coverage at all, and
            the ingest's per-orbit item count is what distinguishes that from a lost orbit.
            The global campaign passes False for the same reason.
        s3_region: Optional S3 region for the mosaic/store opens — threaded, like the
            IAM credential callback, through orbit resolution, chunk enumeration, the
            coverage gate, and the assembly task. ``None`` uses the default Icechunk
            region; set it for a store outside the default region.
        allow_s2_only: Embed S2-valid pixels that have ZERO S1 observations
            (sub-zone SAR coverage gaps) via the upstream v1.1 missing-S1
            convention (all-zeros normalized S1 input) instead of skipping
            them. Default False (historical behavior). Affected pixels are
            identifiable afterwards via ``s1_asc_obs_count +
            s1_desc_obs_count == 0``; quality is unvalidated — see the
            optional-S1 ADR.
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
    # Decidable from parameters, so decide it before anything is provisioned:
    # run_inference rejects this too, but only from inside the Ray context, after
    # a GPU cluster has been paid for. Mirrors fill_zone_year.
    if not dev_params.assembly_only and num_actors < 1:
        raise ValueError(f"num_actors must be >= 1, got {num_actors} (no actor would ever run inference)")

    # Refuse a contradicted resume before anything is opened. The run_id itself is minted
    # much later, from the config's EFFECTIVE flag (see below), but this half of the check
    # needs only the request and must not wait behind a store probe.
    _assert_resume_mode_matches(
        dev_params.previous_run_id, allow_s2_only=allow_s2_only, assembly_only=dev_params.assembly_only
    )
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

    mosaic_base = f"{inputs_bucket.rstrip('/')}/mosaics/{roi_name}"

    # Probe the SAR stores with the same credential callback + region the assemble
    # step uses, so orbit resolution doesn't fall back to the default Icechunk chain.
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials

    resolved_s1_orbit = resolve_s1_orbit(
        mosaic_base,
        s1_orbit,
        allow_none=not require_s1,
        get_credentials=iam_icechunk_credentials,
        s3_region=s3_region,
    )
    if resolved_s1_orbit != s1_orbit:
        log.info("s1_orbit resolved: %s → %s", s1_orbit, resolved_s1_orbit)

    config = build_inference_config(
        s1_orbit=resolved_s1_orbit,
        time_window=time_window,
        checkpoint_path=checkpoint_path,
        inputs_bucket=inputs_bucket,
        output_bucket=output_bucket,
        # An assembly-only resume republishes staged pixels without recomputing them, so
        # what they ARE is recorded in the run_id prefix and nowhere in this call's
        # parameters. The prefix is therefore the ONLY source, not one of two: OR-ing the
        # requested flag in let `allow_s2_only=True` over a bare staged run publish an
        # S2-only manifest for S1-gated tiles, which is the same mislabelling in the other
        # direction — and the direction that makes a later honest flag-off append fail.
        allow_s2_only=(staged_s2_only_mode(dev_params.previous_run_id) if dev_params.assembly_only else allow_s2_only),
    )

    # AFTER the config, and from the config's own flag. `InferenceConfig` FORCES
    # allow_s2_only when the orbit resolves to none — every pixel there has zero S1
    # observations, so the default gate would skip all of them — and minting the run_id
    # from the requested flag missed that: S2-only chunks landed under an unprefixed
    # staging prefix, where an explicit S2-only resume is then refused and the same bare
    # id can be reused under the S1-gated mode the prefix exists to separate.
    run_id = _resolve_run_id(
        dev_params.previous_run_id,
        allow_s2_only=config.allow_s2_only,
        assembly_only=dev_params.assembly_only,
    )
    log.info(
        "Starting tessera_embeddings: roi=%s, mosaic_base=%s, run_id=%s, allow_s2_only=%s",
        roi_name,
        mosaic_base,
        run_id,
        config.allow_s2_only,
    )

    staging_base = f"{output_bucket.rstrip('/')}/staging"

    # Detect staged chunk size from prior runs — chunk_size may differ
    # between a resumed run and the current config.
    chunk_size = config.chunk_size
    if dev_params.previous_run_id:
        try:
            detected = ZarrWriter(staging_base).detect_staged_chunk_size(dev_params.previous_run_id)
        except AllChunksSkippedError:
            # Nothing staged to measure, but the run is real and assemble() publishes
            # an all-fill timestep for it — keep the configured size and continue.
            log.info(
                "Run %s staged no tiles (all chunks skipped) — using configured chunk_size", dev_params.previous_run_id
            )
        else:
            if detected != chunk_size:
                log.warning(
                    "Staged chunks use chunk_size=%d (current config: %d) — using staged value", detected, chunk_size
                )
                chunk_size = detected

    chunks, total_y, total_x = enumerate_mosaic_chunks(
        mosaic_base,
        chunk_size or INFERENCE_CHUNK_SIZE,
        log,
        get_credentials=iam_icechunk_credentials,
        s3_region=s3_region,
    )

    live_chunks = filter_chunks_by_roi_mask(
        chunks,
        roi_zarr_path,
        # The SAME options assembly uses. This filter and assembly's must derive the
        # identical live set from the identical mask; opening one on the ambient chain
        # and the other on the run's credentials can fail here before Ray, or — worse —
        # succeed against a different mask than the one the output is assembled from.
        storage_options=plain_zarr_storage_options(roi_zarr_path, iam_icechunk_credentials, s3_region),
    )
    log.info(
        "ROI filter: %d/%d chunks intersect the ROI, sending %d to GPU actors",
        len(live_chunks),
        len(chunks),
        len(live_chunks),
    )

    check_time_window_coverage(
        mosaic_base,
        time_window,
        s1_orbit=config.s1_orbit,
        skip_coverage_check=dev_params.skip_coverage_check,
        get_credentials=iam_icechunk_credentials,
        s3_region=s3_region,
    )

    # Same structural check `assemble` makes before extending an existing store —
    # model, sampler checkpoints, upstream ingest identity — run HERE, on metadata
    # only, so an append that can never be accepted is rejected before a GPU fleet
    # is provisioned rather than after the whole inference is paid for.
    #
    # Only on runs that will actually append. `assembly_only` reaches assemble
    # immediately, which validates for itself; `inference_only` returns after
    # staging and never touches the output store, so gating it would block a
    # legitimate dev run — staging a new checkpoint against an ROI whose published
    # store was written by a different one — over an append it is not making.
    if not (dev_params.assembly_only or dev_params.inference_only):
        assert_output_store_accepts(
            output_bucket=output_bucket,
            roi_name=roi_name,
            output_name_suffix=dev_params.output_name_suffix,
            config=config,
            mosaic_base=mosaic_base,
            log=log,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )

    # Lazily import the AWS Ray provider so the embeddings flow file
    # can be inspected (for arch tests) on machines without ray
    # installed. The provider is only needed when the flow actually
    # runs.
    from tessera_embeddings.providers.aws.ray import (
        cluster_name_for_flow_run,
        make_instance_terminator,
        ray_cluster,
    )

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
        # Same credential callback + region the orbit probe uses — so the assembly
        # task's manifest read + writer.assemble open the stores with them, not the
        # default Icechunk chain (callback-only / non-default-region deployments).
        "get_credentials": iam_icechunk_credentials,
        "s3_region": s3_region,
    }

    if dev_params.assembly_only:
        # Staged completeness is verified by `assemble` itself, against the same live
        # set it derives from the ROI mask — so it holds however assembly is entered.
        log.info("Assembly-only mode: assembling staged chunks from run %s", dev_params.previous_run_id)
        return _run_assembly(log=log, result_stats=None, **assemble_kwargs)

    # Deterministic, flow-run-derived cluster name so the cancellation/crash hook can
    # re-derive it in a fresh process (see _ray_lifecycle). Outside a Prefect run
    # (unit tests) the id is None and ray_cluster falls back to its own random suffix.
    try:
        with ray_cluster(
            log,
            ami_ssm_name=ami_ssm_name,
            cluster_name=cluster_name_for_flow_run(flow_run_ctx.id) if flow_run_ctx.id else None,
            ssm_prefix=ssm_prefix,
            cloudwatch_log_group=cloudwatch_log_group,
            code_bucket=code_bucket,
            code_suffix=code_suffix,
            sync_source_path=Path(dev_params.sync_source_path) if dev_params.sync_source_path else None,
        ) as resolved_yaml:
            activate(resolved_yaml)

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
                s3_region=s3_region,
                # Terminate the EC2 instance behind each retired idle actor at once,
                # rather than holding idle GPU nodes to the end of the run on the Ray
                # autoscaler's idle timeout, which is unreliable after ray.kill()
                # (providers/aws/gotchas.md). Actors go idle at the tail, while the
                # last chunks finish, so this is where a run stops paying for GPUs it
                # is done with. The callback runs driver-side; its boto3 client never
                # ships to a worker. Matches fill_zone_year.
                on_actor_retire=make_instance_terminator(log=log),
            )
    finally:
        # Clear the hook state on the EXCEPTION path too. Cleared only on success, a run
        # whose inference raised inside the Ray context left `_ray_lifecycle`'s
        # process-wide cluster name pointing at it — and Prefect reuses worker processes,
        # so a LATER run's cancellation hook would prefer that stale name over its own
        # flow-run fallback, tear down a cluster already gone, and leak the live fleet it
        # was called to reclaim. Matches fill_zone_year and fill_zones_sequential, which
        # both already do this.
        deactivate()

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
    """Run the assembly task on the flow runner with a local worker-process pool.

    No cluster to provision: the raw-zarr engine forks worker processes on this
    host (see ``inference.assembly``), so the task runs directly under the
    flow's default runner. ``AssemblyConfig`` sizes the pool from the live
    chunk count within its RAM/S3 budget.
    """
    n_workers = AssemblyConfig().compute_n_workers(assemble_kwargs["n_live_chunks"])
    log.info("Assembling on the flow runner with %d worker process(es)", n_workers)
    return assemble_embeddings_task(
        result_stats=result_stats,
        n_workers=n_workers,
        **assemble_kwargs,
    )
