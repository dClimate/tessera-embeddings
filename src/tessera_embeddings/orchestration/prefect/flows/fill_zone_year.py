"""Fill one (zone, year) of the global embeddings store via Ray (ADR-008 W5).

Provisions a Ray GPU cluster, runs the zone-fill runner
(:func:`tessera_embeddings.orchestration.runners.zone_fill.fill_zone_year` —
coverage mask → inference → shard assembly → tag), and tears the cluster down.
Mirrors :mod:`tessera_embeddings`'s single-flow Ray pattern (``ray_cluster``
context manager; the cancellation hook is shared via :mod:`._ray_lifecycle`).

**Concurrency model (ADR-008 D6).** Inference is embarrassingly parallel across
zones — nothing is shared and nothing commits — so many of these flow runs can
do GPU work at once. Only the *commit* contends: the runner gates
``assemble_global``'s commit on ``gate`` and leaves ``run_inference`` ungated.
Pass ``commit_limit_name`` to make ``gate`` a **Prefect global concurrency
limit**, so no more than that limit's slots' worth of fill runs commit
simultaneously across the whole fleet (avoiding S3 rebase/commit storms) while
inference stays unbounded. Same-zone serialization (whose attr commits genuinely
conflict) is the campaign driver's job, not this flow's.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from prefect import flow, get_run_logger
from prefect.concurrency.sync import concurrency
from prefect.runtime import flow_run as flow_run_ctx

from tessera_embeddings.config.inference import checkpoint_filename
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.conventions import expected_model_url
from tessera_embeddings.inference.data_loading import check_time_window_coverage, resolve_s1_orbit
from tessera_embeddings.inference.orchestration_helpers import build_inference_config
from tessera_embeddings.orchestration.prefect.flows._ray_lifecycle import (
    activate,
    deactivate,
    ray_cleanup_on_cancellation,
)
from tessera_embeddings.orchestration.runners.zone_fill import (
    assert_calendar_year_window,
    fill_zone_year,
    zone_has_live_tiles,
    zone_year_complete,
    zone_year_on_axis,
)
from tessera_embeddings.providers.aws.ray import cluster_name_for_flow_run
from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group
from tessera_embeddings.storage.zone_grid import canonicalize_zone

if TYPE_CHECKING:
    import icechunk


class _PrefectCommitGate(AbstractContextManager):
    """A ``CommitGate`` backed by a Prefect global concurrency limit.

    Each ``with gate:`` acquires one slot of the named limit for the duration of
    a commit and releases it after, so the campaign's committer count is bounded
    fleet-wide (across separate flow runs / machines) without limiting inference.
    A fresh :func:`concurrency` context is opened per entry so the gate is
    reusable across the (few) commits a single fill performs.

    ``strict=True``: this gate is LOAD-BEARING for fleet-wide commit contention, so
    an absent or misspelled limit must fail closed. Prefect's ``concurrency``
    defaults ``strict=False``, which would only log a warning and let the commit
    proceed UNGATED — silently reintroducing the rebase/commit storm the gate
    exists to prevent. Strict mode raises instead, so the limit must be
    provisioned explicitly (see the campaign runbook).

    THREAD-SAFE: the active context lives in a per-thread stack, not an instance
    slot — the chained fill shares ONE gate between its feeder thread (terminal
    plans commit inside ``plan``) and its trailing-assembly thread, and an
    instance slot would let a concurrent enter overwrite the other thread's
    context and release the wrong slot on exit.
    """

    def __init__(self, name: str, occupy: int = 1) -> None:
        self._name = name
        self._occupy = occupy
        self._local = threading.local()

    def __enter__(self) -> None:
        cm = concurrency(self._name, occupy=self._occupy, strict=True)
        cm.__enter__()
        stack: list[AbstractContextManager[Any]] = getattr(self._local, "stack", [])
        stack.append(cm)
        self._local.stack = stack

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        cm = self._local.stack.pop()
        cm.__exit__(exc_type, exc, tb)


def _assert_seeded_model_matches(
    store_path: str,
    *,
    build_checkpoint: str,
    allow_model_mismatch: bool,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None,
    s3_region: str | None = None,
) -> None:
    """Refuse to fill a store seeded for a different encoder/checkpoint than this build.

    The seed stamps ``geoemb:model`` (the encoder-version URL) once at the store
    root; the fill re-derives it from the running code. If a model upgrade slipped
    in between seeding and filling, the fill would write embeddings from a NEW
    encoder while the root still advertises the OLD one — mixing encoders under a
    single store and permanently tagging the result. A metadata-only read catches
    it before Ray. ``allow_model_mismatch`` is the deliberate-override escape hatch
    (e.g. a store seeded with a custom ``model_url`` the code can't re-derive).

    ``geoemb:model`` versions only the ENCODER (e.g. "1.1"); it does NOT distinguish
    the concrete checkpoint / norm source (the ``aws`` and ``mpc`` v1.1 checkpoints
    share one URL). So when the store also recorded a ``checkpoint_id`` (the seed's
    ``model_version``), require it to match this build's checkpoint (*build_checkpoint*
    = :func:`checkpoint_filename`) — otherwise two same-URL-but-different checkpoints
    could be mixed. Absent ``checkpoint_id`` (the default), only the encoder URL is
    gated. Seed with ``model_version=checkpoint_filename()`` to enable checkpoint-level
    gating.
    """
    root = open_store_as_zarr_group(store_path, get_credentials=get_credentials, region=s3_region)
    seeded = cast("str | None", root.attrs.get("geoemb:model"))
    expected = expected_model_url()
    if seeded is not None and seeded != expected and not allow_model_mismatch:
        raise ValueError(
            f"Global store {store_path} was seeded for encoder {seeded!r} but this build embeds with "
            f"{expected!r} — filling would mix encoders under one store (its root still advertises "
            f"{seeded!r}). Reseed for the new model, or pass allow_model_mismatch=True to override."
        )
    seeded_ckpt = cast("str | None", root.attrs.get("checkpoint_id"))
    if seeded_ckpt is not None and seeded_ckpt != build_checkpoint and not allow_model_mismatch:
        raise ValueError(
            f"Global store {store_path} was seeded for checkpoint {seeded_ckpt!r} but this build fills with "
            f"{build_checkpoint!r} — the encoder URL matches but the concrete checkpoint (norm source) differs, "
            "so the embeddings would be mixed. Reseed, or pass allow_model_mismatch=True to override."
        )


@flow(
    name="fill-zone-year",
    # Both lists hold the SAME function: a crashed run leaks exactly like a
    # cancelled one. Keep the hook IDEMPOTENT — cancelling a parent and its child
    # together delivers the transition twice and runs it twice (2026-07-25).
    on_cancellation=[ray_cleanup_on_cancellation],
    on_crashed=[ray_cleanup_on_cancellation],
)
def fill_zone_year_flow(
    *,
    zone: str,
    year: int,
    paths: BucketPaths,
    ami_ssm_name: str,
    ami_id: str | None = None,
    time_window_end: str | None = None,
    store_name: str = "tessera",
    mask_name: str = "global",
    ssm_prefix: str = "/tessera/ray/",
    cloudwatch_log_group: str = "/ec2/tessera/ray",
    code_bucket: str | None = None,
    code_suffix: str = "",
    num_actors: int = 20,
    s1_orbit: str = "both",
    s3_region: str | None = None,
    commit_limit_name: str | None = None,
    cleanup_staging: bool = True,
    allow_partial_window: bool = False,
    allow_model_mismatch: bool = False,
    allow_s2_only: bool = False,
    mosaic_base: str | None = None,
    s3_concurrency: int | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Fill one ``(zone, year)`` cell of the global store on a Ray cluster.

    Args:
        zone: Zone group name — UTM common name, e.g. ``"30N"``/``"19S"``.
        year: Campaign calendar year to fill (must be on the seeded axis).
        paths: Deployment storage contract (global store, land mask, mosaics).
        ami_ssm_name: SSM parameter name for the Ray GPU AMI ID (used only when
            ``ami_id`` is not given).
        ami_id: Pre-resolved AMI ID the campaign pinned into this fill's staging
            fingerprint. When set, the cluster boots exactly this image instead of
            re-reading ``ami_ssm_name`` — so a mid-campaign re-bake can't run new code
            under a fingerprint that recorded the old image. ``None`` (direct calls)
            resolves ``ami_ssm_name`` at provisioning as before.
        time_window_end: End month of the inference window as ``"Month Year"``;
            defaults to ``"December {year}"`` (the calendar-year window). The
            runner REQUIRES the exact January-December window for ``year``
            (the store guarantees calendar-year slots), so any other value fails
            loudly at the preflight gate — non-calendar 12-month windows belong
            in the single-ROI ``12mo_window_end`` path, not the global store.
        store_name: Global-store repo basename (``paths.global_store``).
        mask_name: Coverage-store repo basename (``paths.land_mask_store``).
        ssm_prefix: SSM prefix for the Ray cluster resource IDs.
        cloudwatch_log_group: CloudWatch log group the Ray workers write to.
        code_bucket: S3 bucket workers pull the source tarball from (``None`` =
            AMI-baked source). See :mod:`tessera_embeddings`.
        code_suffix: Source-tarball filename suffix (lets branches coexist).
        num_actors: GPU actor count for inference.
        s1_orbit: ``"ascending"``, ``"descending"``, or ``"both"``.
        s3_region: Optional S3 region for the global store + mosaics, threaded through
            the preflight reads and the zone-fill runner (default region if None).
        commit_limit_name: Prefect global concurrency limit that bounds the
            fleet's simultaneous committers (D6). ``None`` = ungated (a single
            isolated run has no commit contention).
        cleanup_staging: Delete staged tiles after a successful fill.
        allow_partial_window: Relax the pre-Ray temporal-coverage gate to
            "non-empty" (default requires the mosaic's months to span the window).
        allow_model_mismatch: Fill even when the seeded store advertises a
            different encoder than this build (default rejects — a mid-campaign
            model upgrade would otherwise mix encoders under one store).
        allow_s2_only: Embed S2-valid pixels that have ZERO S1 observations
            (sub-zone SAR coverage gaps) using the upstream v1.1 missing-S1
            convention (all-zeros normalized S1 input). Default False: such
            pixels are skipped. Only the PER-PIXEL gate is relaxed — the
            zone-level gates (SAR store presence, temporal coverage, grid
            validation) stay strict, so a zone with NO SAR at all still fails
            loudly (that signals an ingest bug, not geography). Identify the
            affected pixels afterwards via ``s1_asc_obs_count +
            s1_desc_obs_count == 0``. Embedding quality for S2-only pixels is
            unvalidated — see the optional-S1 ADR before production use.
        mosaic_base: Override for the input mosaic prefix (default
            ``{inputs}/mosaics/{zone}/{year}`` — the campaign's per-year layout).
        s3_concurrency: This fill's slice of the fleet S3-PUT budget for the shard
            write (``None`` = the full target, for a lone fill). The campaign passes
            ``target // max_parallel_zones`` so K concurrent fills stay near target.
        run_id: Reuse a prior run's id to resume it (staged tiles are skipped).

    Returns:
        The zone-fill summary dict (zone, year, run_id, snapshot_id, tag,
        tile/inference counts, ``empty`` flag, elapsed seconds).
    """
    log = get_run_logger()

    # Checked HERE, not left to run_inference: this deployment is invoked directly as
    # well as by the campaign, and on that path the value is only validated after the
    # flow has cleared preflight and entered `ray_cluster` — so a typo buys a billable
    # GPU cluster before failing on something knowable at call time. The campaign
    # wrapper rejects it too; the child is the authority for a direct invocation.
    if num_actors < 1:
        raise ValueError(f"num_actors must be >= 1, got {num_actors} (no actor would ever run inference)")

    # Lazily import the AWS providers so the flow file imports on machines
    # without ray/boto installed (arch tests, local inspection).
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials
    from tessera_embeddings.providers.aws.ray import make_instance_terminator, ray_cluster

    zone = canonicalize_zone(zone)
    store_path = paths.global_store(store_name)
    land_mask_path = paths.land_mask_store(mask_name)
    # Campaign mosaics live per (zone, year); `mosaic_base` overrides for a
    # hand-provided mosaic (e.g. a shared multi-year store).
    mosaic_base = mosaic_base or f"{paths.inputs.rstrip('/')}/mosaics/{zone}/{year}"

    # Preflight in ascending cost order — cheapest global-store metadata reads
    # first, then the mask, then (only if we will truly infer) the mosaic probe
    # and Ray. GPU work is needed ONLY for a cell that is on-axis, not already
    # complete, AND has live coverage. Each cheaper short-circuit avoids the more
    # expensive reads below it; fill_zone_year re-validates as the authority.
    #  - already-complete cell: the campaign dispatches landed-but-untagged cells
    #    for crash recovery, where fill_zone_year merely re-creates the tag. This
    #    needs ONLY the global store — so it is checked before the land mask,
    #    whose unavailability must never block a retag-only recovery.
    #  - off-axis / unseeded year: fill_zone_year rejects it outright, so resolve
    #    it here (a global-store read) BEFORE standing up a cluster we'd tear
    #    straight back down.
    #  - all-ocean cell (no live tiles): may have no mosaic to probe; fill empty.
    already_complete = zone_year_complete(
        store_path, zone, year, get_credentials=iam_icechunk_credentials, s3_region=s3_region
    )
    on_axis = already_complete or zone_year_on_axis(
        store_path, zone, year, get_credentials=iam_icechunk_credentials, s3_region=s3_region
    )
    # Only touch the mask once completion + axis are cleared: an unavailable mask
    # must not block retag-only recovery, and an off-axis year needs no mask.
    has_live = (
        zone_has_live_tiles(land_mask_path, zone, get_credentials=iam_icechunk_credentials, s3_region=s3_region)
        if (on_axis and not already_complete)
        else False
    )
    needs_cluster = on_axis and not already_complete and has_live

    # resolve_s1_orbit probes the mosaics (with the same credential callback / region
    # the rest of the fill uses), so only do it when we'll actually infer.
    resolved_s1 = (
        resolve_s1_orbit(mosaic_base, s1_orbit, get_credentials=iam_icechunk_credentials, s3_region=s3_region)
        if needs_cluster
        else s1_orbit
    )
    # The strict Jan-Dec calendar-year window for `year`: `December {year}` yields a
    # 12-month window spanning exactly Jan-Dec (the store's guaranteed convention).
    # A `time_window_end` override producing any other window is rejected loudly by
    # the runner's calendar-year gate — the single enforcement chokepoint.
    window = parse_time_window(time_window_end or f"December {year}")
    # Before ray_cluster: the runner's gate would reject an offset/partial window
    # anyway, but only after a billable GPU fleet is up. Decidable from config, so
    # decide it here.
    assert_calendar_year_window(window, year)
    config = build_inference_config(
        s1_orbit=resolved_s1,
        time_window=window,
        checkpoint_path=f"{paths.inputs.rstrip('/')}/models/{checkpoint_filename()}",
        inputs_bucket=paths.inputs,
        output_bucket=paths.outputs,
        # 1 inference tile == 1 output shard == 2x2 tiles per 4096-px ingest chunk
        # (D3), so assembly writes whole, lean shards with no read-modify-write.
        # Explicit, NOT the INFERENCE_CHUNK_SIZE default: that default belongs to
        # the single-ROI output geometry (500-px chunks) and must not be retuned
        # for this path.
        chunk_size=SHARD_PX,
        allow_s2_only=allow_s2_only,
    )

    if needs_cluster:
        # Fail loudly BEFORE provisioning Ray if the store was seeded for a
        # different encoder than this build embeds with — a mid-campaign model
        # upgrade would otherwise mix encoders under one store and tag it
        # permanently. A cheap metadata-only read; the escape hatch is explicit.
        _assert_seeded_model_matches(
            store_path,
            build_checkpoint=checkpoint_filename(),
            allow_model_mismatch=allow_model_mismatch,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )
        # Fail loudly on a partial/absent mosaic BEFORE provisioning Ray: a
        # zone-wide mosaic missing months is an ingest failure, and the write-once
        # zone-year tag would otherwise make partial embeddings permanent (the
        # zone-fill chain has no other temporal-coverage gate). `allow_partial_window`
        # relaxes the month-span check to "non-empty" for a legitimately partial
        # edge zone; an empty store still fails.
        check_time_window_coverage(
            mosaic_base,
            config.time_window,
            s1_orbit=resolved_s1,
            skip_coverage_check=allow_partial_window,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )

    gate = _PrefectCommitGate(commit_limit_name) if commit_limit_name else None

    fill_kwargs: dict[str, Any] = {
        "store_path": store_path,
        "zone": zone,
        "year": year,
        "land_mask_path": land_mask_path,
        "mosaic_base": mosaic_base,
        # Staging is scoped by (zone, year), not just run_id: a reused/retained
        # run_id can't then match another cell's staged labels+shapes and trick
        # resume detection into skipping inference for the wrong year.
        "staging_base": f"{paths.outputs.rstrip('/')}/staging/{zone}/{year}",
        "config": config,
        "num_actors": num_actors,
        "log": log,
        "run_id": run_id,
        "gate": gate,
        "cleanup_staging": cleanup_staging,
        "s3_concurrency": s3_concurrency,
        "get_credentials": iam_icechunk_credentials,
        "s3_region": s3_region,
    }

    if not needs_cluster:
        if already_complete:
            reason = "already complete (retag only)"
        elif not on_axis:
            reason = "year off the pre-allocated axis (runner will reject)"
        else:
            reason = "no live tiles (all-ocean)"
        log.info("Zone %s year %d %s — no Ray cluster", zone, year, reason)
        return fill_zone_year(**fill_kwargs)

    # Ray path only: terminate the EC2 instance behind each retired idle actor
    # immediately (the runner's on_actor_retire hook), instead of holding idle
    # GPU nodes for the rest of a multi-hour fill and relying on the autoscaler's
    # unreliable-after-ray.kill() idle timeout (providers/aws/gotchas.md). The
    # callback runs driver-side, so its boto3 client never ships to a Ray worker.
    # Region matches ray_cluster's default (this flow provisions there).
    fill_kwargs["on_actor_retire"] = make_instance_terminator(log=log)

    try:
        # Pin a deterministic cluster name from the flow-run id so the cancellation
        # hook can re-derive it and terminate the fleet by tag even when it runs in a
        # fresh module import (globals unset) — without this, a cancel before
        # `activate()` records the name would hit the "no cluster name" path and leak
        # the GPU fleet.
        with ray_cluster(
            log,
            ami_ssm_name=ami_ssm_name,
            ami_id=ami_id,
            ssm_prefix=ssm_prefix,
            cloudwatch_log_group=cloudwatch_log_group,
            code_bucket=code_bucket,
            code_suffix=code_suffix,
            cluster_name=cluster_name_for_flow_run(flow_run_ctx.id),
        ) as resolved_yaml:
            activate(resolved_yaml)
            summary = fill_zone_year(**fill_kwargs)
    finally:
        # Clear the hook state only AFTER the context manager's teardown has
        # run (or failed into the hook's remit) — and also on the exception
        # path, which the old success-only clearing missed.
        deactivate()

    log.info("Zone %s year %d filled: %s", zone, year, summary.get("tag"))
    return summary
