"""Drive the global embeddings campaign: fill every pending (zone, year).

Reads the live fill progress from the seeded store
(:func:`~tessera_embeddings.storage.campaign.campaign_status`) and dispatches
fills for every pending cell via ``arun_deployment`` — a ``fill-zone-year`` run
per cell (``fill_strategy="cluster-per-zone"``), or size-balanced
``fill-zones-sequential`` shards on long-lived clusters (``"chained-clusters"``)
— mirroring :mod:`tessera_full_pipeline`'s driver pattern.

**Scheduling (ADR-008 D6 + the runner contract).** Inference is parallel across
zones; only commits contend, and only *same-zone* fills conflict (shared
``years_complete``/``runs`` attrs → ``RebaseFailedError``). So the driver runs
**year by year** (an outer serial loop) and, within a year, dispatches its zones
**concurrently** up to ``max_parallel_zones`` — all distinct zones, so no
same-zone overlap is ever possible. The fleet-wide committer bound is a separate
knob: ``commit_limit_name`` (a Prefect global concurrency limit) is passed to
every fill so commits stay under the storm threshold while inference runs free.
``pending()`` is year-major for exactly this drain pattern.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Collection
from functools import partial
from typing import Any

import icechunk
import zarr
from prefect import flow, get_run_logger
from prefect.deployments import arun_deployment
from prefect.runtime import flow_run as flow_run_ctx

from tessera_embeddings.config.inference import checkpoint_filename
from tessera_embeddings.config.ingest import IngestSettings
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.inference.assembly import TARGET_AGGREGATE_S3_CONCURRENCY
from tessera_embeddings.inference.data_loading import _active_orbits
from tessera_embeddings.orchestration.prefect.flows._child_runs import child_run_tag, make_child_cancel_hook
from tessera_embeddings.orchestration.prefect.flows.fill_zone_year import _assert_seeded_model_matches
from tessera_embeddings.orchestration.prefect.flows.ingest_zone_year import IngestDeployments
from tessera_embeddings.orchestration.prefect.flows.tessera_full_pipeline import _check_completed
from tessera_embeddings.orchestration.runners.zone_fill import (
    zone_has_live_tiles,
    zone_live_tile_count,
    zone_year_on_axis,
)
from tessera_embeddings.storage.campaign import campaign_status, campaign_work_list, tag_year_complete, zone_year_tag
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.object_store import delete_prefix
from tessera_embeddings.storage.zarr_store import is_missing_repo, open_store_group_and_tip
from tessera_embeddings.storage.zone_grid import CAMPAIGN_YEARS, canonicalize_zone

# S1/S2 grandchild ingest refs are derived from this single source (rather than
# duplicated string literals), so a future rename of the ingest deployments stays
# in one place — ingest_zone_year.IngestDeployments.
_INGEST_DEPLOYMENT_DEFAULTS = IngestDeployments()


#: Tag prefix for every deployment this campaign dispatches — ingests, per-cell
#: fills and chained-fill shards alike.
_CHILD_TAG_PREFIX = "campaign"


def _child_tag(flow_run_id: object) -> str | None:
    """Deterministic tag for this campaign's dispatched children."""
    return child_run_tag(_CHILD_TAG_PREFIX, flow_run_id)


#: Cancelling the campaign must stop what it started. Each `arun_deployment` creates
#: an INDEPENDENT run: killing this flow leaves its ingests and fills running, still
#: writing mosaics and store attrs a retry will race, with their Dask and Ray fleets
#: still billing — and a campaign has many in flight at once. Registered as both
#: terminal hooks because a crashed parent orphans children exactly like a cancelled
#: one. The child flows keep their OWN teardown hooks; this one stops the runs those
#: hooks then clean up after.
_cancel_children_on_cancellation = make_child_cancel_hook(_CHILD_TAG_PREFIX, "campaign child run")


def _dpl(prod_ref: str, branch: str | None) -> str:
    """Route a child deployment ref to its branch-scoped variant.

    A blank ``branch`` (``None``, empty, or whitespace) returns the ref
    unchanged — production behaviour. A real branch slug is appended to the
    *deployment name*: ``"flow/name"`` -> ``"flow/name-<branch>"`` (dev branches
    register every flow under the suffixed name with their own image + task def).
    """
    slug = (branch or "").strip()
    return f"{prod_ref}-{slug}" if slug else prod_ref


def _mosaic_identity(
    zone: str,
    year: int,
    *,
    inputs_bucket: str,
    s1_orbit: str,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> str:
    """Identity string of the mosaic a fill will read, per ACTIVE child store.

    The AUTHORITATIVE term is each store's branch-tip SNAPSHOT ID: it is
    Icechunk's own content identity, so it moves on any commit that can change
    the pixels a fill reads — including ones no bookkeeping attribute records
    (a re-run of a single date's region write, a hand-repaired store, a
    prebuilt mosaic maintained outside this pipeline). The attrs below are a
    proxy that is only as good as the writer's discipline; keying the fingerprint
    on them alone means a mosaic can change underneath a staging prefix, so a
    retry resumes tiles built from the OLD pixels and publishes a silent mixture
    of two inputs. The snapshot ID closes that.

    The attrs ride along for legibility, not for safety: ``ingest_marker``
    (window + coverage-delivery sha + min_valid_coverage + orbit +
    allow_partial_window — the exact per-(zone,year) fingerprint that produced
    the store) plus the ``ingest_completed_at`` that separates two builds sharing
    one policy, or ``last_appended``/``created_at`` for a prebuilt
    (``ingest=False``) mosaic. They make a changed run_id diagnosable from the
    string itself; a store carrying none of them is no longer fatal, because the
    snapshot ID alone is a complete identity.

    This is deliberately per-(zone,year) and post-ingest, NOT one global
    coverage sha read once up front: a partial ``build_all(zones=...)`` can leave
    zones on different coverage revisions, and coverage can change after an early
    read — either would let stale staged tiles resume against a rebuilt mosaic.
    Genuinely-absent stores (missing repo / not found) are skipped.

    SCOPE: this pins the inputs at RESUME time, not at read time. The tip is
    sampled once, here, while the fill's actors later open each mosaic from the
    moving ``main``. So it answers "have the inputs changed since the last
    attempt?" — the retry question the staging prefix exists for — and NOT "did
    the inputs hold still while this attempt ran?". Nothing in the pipeline
    writes a mosaic during its own fill (ingest for a cell completes before that
    cell dispatches, mosaics are per-(zone,year), and cleanup runs after), so the
    remaining window needs an out-of-band writer committing mid-fill. Closing it
    would mean threading these snapshot IDs through the fill parameters into every
    actor store open so reads are pinned too; until then, do not read this
    function as a guarantee that a fill saw one consistent mosaic revision.

    Only the ACTIVE orbit set is fingerprinted (reflectance +
    ``_active_orbits(s1_orbit)``): the fill reads only those, so an inactive
    opposite-orbit store that happens to be present (a stale/markerless
    leftover, or a prebuilt ``ingest=False`` mosaic that shipped both orbits)
    must NOT be opened here — it can't affect the embeddings, yet
    fingerprinting it could raise (no identity) or perturb the run_id and
    needlessly restart a valid single-orbit fill.

    Module-level (not a flow closure) because both fill strategies need it:
    the cluster-per-zone dispatch here and the chained-clusters fill's
    ``prepare`` callable (:mod:`.fill_zones_sequential`).
    """
    base = f"{inputs_bucket.rstrip('/')}/mosaics/{zone}/{year}"
    ids: list[str] = []
    for store in ["reflectance", *(f"sar_{orbit}" for orbit in _active_orbits(s1_orbit))]:
        path = f"{base}/{store}.zarr"
        try:
            grp, tip = open_store_group_and_tip(path, get_credentials=get_credentials, region=s3_region)
        except zarr.errors.GroupNotFoundError:
            # PRESENT but rootless — see the identical guard in data_loading's orbit
            # probe. Skipping it here would fingerprint (and later fill) a zone as
            # single-orbit because the other orbit's repo was damaged, not absent.
            raise
        except FileNotFoundError:
            continue  # absent store (single-orbit mosaic / unproduced orbit) — not active
        except icechunk.IcechunkError as exc:
            if is_missing_repo(exc):
                continue
            raise  # transient/auth: fail closed rather than fingerprint a partial view
        marker = grp.attrs.get("ingest_marker")
        provenance = (
            # marker + completed_at, not the marker alone: the marker is policy, so a
            # rebuild under identical settings reproduces it byte for byte. completed_at
            # is stamped in the same commit and only moves when ingest actually re-ran.
            f"{marker!r}@{grp.attrs.get('ingest_completed_at')}"
            if isinstance(marker, dict)
            else (grp.attrs.get("last_appended") or grp.attrs.get("created_at"))
        )
        # snapshot FIRST: it is the term that makes this safe. The attrs follow so an
        # operator diffing two run_ids can see WHY the mosaic changed, not just that it
        # did; a store with no provenance attrs is still fully identified by its tip.
        ids.append(f"{store}=snapshot:{tip}" + (f"+{provenance}" if provenance is not None else ""))
    if not ids:
        raise ValueError(f"No mosaic stores found under {base} — nothing to fingerprint or fill.")
    return "|".join(ids)


def _staging_run_id(
    zone: str,
    year: int,
    *,
    inputs_bucket: str,
    min_valid_coverage: float,
    s1_orbit: str,
    allow_partial_window: bool,
    allow_s2_only: bool,
    code_identity: str,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> str:
    """Deterministic staging run_id fingerprinting everything that determines the embeddings.

    Covers the config (threshold/orbit/window/checkpoint/S2-only), the CODE the
    fill runs (``code_identity`` — the resolved AMI ID + tarball ETag from
    :func:`_resolve_code_identity`, NOT the mutable ``code_suffix`` label, so a
    re-baked AMI or overwritten tarball starts a fresh prefix), and the
    per-(zone,year) mosaic identity (:func:`_mosaic_identity`). A retry with
    identical inputs resumes the same prefix (findable for cleanup); ANY change
    starts a fresh prefix, so old tiles are never resumed under new inputs.
    Call AFTER ingest so ingest=True reads the freshly-written marker.
    """
    key = (
        year,
        min_valid_coverage,
        s1_orbit,
        allow_partial_window,
        # allow_s2_only changes WHICH pixels get embeddings, so a retry across a
        # flipped flag must start a fresh staging prefix — resuming would mix
        # S1-gated and S2-only tiles under one run.
        allow_s2_only,
        # Checkpoint identity is the FILENAME, not the weight bytes — deliberately.
        # A new model ships under a new filename (norm_source → a distinct name),
        # which both flips this fingerprint AND is rejected by the seeded model gate
        # (checkpoint_id / geoemb:model). The checkpoint is fixed for a campaign and
        # never overwritten mid-run, so a same-filename byte-swap is a non-scenario;
        # content-hashing it here would buy nothing. (Do not "fix" to an ETag.)
        checkpoint_filename(),
        code_identity,
        _mosaic_identity(
            zone,
            year,
            inputs_bucket=inputs_bucket,
            s1_orbit=s1_orbit,
            get_credentials=get_credentials,
            s3_region=s3_region,
        ),
    )
    return f"{zone}-{year}-{hashlib.sha256(repr(key).encode()).hexdigest()[:8]}"


def _ingest_dispatch_params(
    zone: str,
    year: int,
    *,
    paths: BucketPaths,
    mask_name: str,
    s1_orbit: str,
    ingest_settings: IngestSettings,
    allow_partial_window: bool,
    s3_region: str | None = None,
    branch: str | None = None,
) -> dict[str, Any]:
    """Parameter dict for one ``ingest-zone-year`` deployment run.

    Shared by both fill strategies' dispatch sites — the cluster-per-zone
    driver's per-cell chain and the chained fill's look-ahead adapter
    (:mod:`.fill_zones_sequential`) — so the two cannot drift from the
    deployment's signature independently. ``s3_region`` is the same region the
    fill uses, so ingest's metadata opens (mask liveness, coverage sha, marker,
    coverage gate) hit the same stores rather than defaulting to us-west-2 on a
    non-default-region deployment. ``branch`` routes the S1/S2 grandchildren
    ``ingest_zone_year`` dispatches to their branch-scoped deployments (see
    :func:`_dpl`) — without it, a branch campaign 404s one level down on the
    unsuffixed prod refs.
    """
    return {
        "zone": zone,
        "year": year,
        "paths": paths.model_dump(),
        "mask_name": mask_name,
        "s1_orbit": s1_orbit,
        "ingest_settings": ingest_settings.model_dump(),
        "allow_partial_window": allow_partial_window,
        "s3_region": s3_region,
        # Grandchild routing: the S1/S2 ingest deployments dispatched BY
        # ingest_zone_year. Without this key its IngestDeployments() defaults
        # (unsuffixed prod refs) always win. Built from the defaults model so a
        # rename stays single-sourced.
        "deployments": {
            "ingest_s1_roi_sar": _dpl(_INGEST_DEPLOYMENT_DEFAULTS.ingest_s1_roi_sar, branch),
            "ingest_s2_roi_reflectance": _dpl(_INGEST_DEPLOYMENT_DEFAULTS.ingest_s2_roi_reflectance, branch),
        },
    }


def _partition_by_live_tiles(
    zones: list[str],
    n_shards: int,
    *,
    land_mask_path: str,
    known_complete: Collection[str] = (),
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> list[list[str]]:
    """Split ``zones`` into ``n_shards`` lists of ~equal total live-tile count.

    Longest-processing-time greedy: zones descending by coverage-bitmap tile
    count, each assigned to the currently-lightest shard — so the sequential
    fill's shared clusters finish their shards at roughly the same time.
    ``known_complete`` zones (retag-only crash-recovery cells) weigh zero — the
    child settles them without GPU work, so their tile counts would only skew
    the balance (and their mask reads are skipped). Tile counts are one ~1 KB
    GET per zone; ``n_shards == 1`` skips the reads entirely. Empty shards
    (more shards than zones) are dropped.
    """
    if n_shards <= 1:
        return [zones]
    weight = {
        z: 0
        if z in known_complete
        else zone_live_tile_count(land_mask_path, z, get_credentials=get_credentials, s3_region=s3_region)
        for z in zones
    }
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    totals = [0] * n_shards
    for z in sorted(zones, key=lambda z: weight[z], reverse=True):
        i = totals.index(min(totals))
        shards[i].append(z)
        totals[i] += weight[z]
    return [sh for sh in shards if sh]


def _resolve_code_identity(
    ami_ssm_name: str, code_bucket: str | None, code_suffix: str, region: str | None, ami_id: str | None = None
) -> str:
    """Lazy-import wrapper over the AWS provider's code-artifact resolver.

    Kept module-level (and thin) so this flow file still imports on non-AWS machines
    (arch tests) — boto3 lives only under ``providers/aws`` — and tests can stub it.
    Region ``None`` → us-west-2, the region the fills' ``ray_cluster`` PROVISIONS
    from: the AMI SSM parameter and source tarball live there, NOT in the storage
    ``s3_region`` (which may differ for a non-default-region store). ``ami_id`` pins
    the AMI component to a pre-resolved id so the fingerprint matches the image the
    fill is pinned to boot.
    See :func:`tessera_embeddings.providers.aws.ray.resolve_code_artifact_identity`.
    """
    from tessera_embeddings.providers.aws.ray import resolve_code_artifact_identity

    return resolve_code_artifact_identity(ami_ssm_name, code_bucket, code_suffix, region or "us-west-2", ami_id=ami_id)


def _resolve_ami_id(ami_ssm_name: str, region: str | None) -> str:
    """Lazy-import wrapper over the AWS provider's AMI-ID resolver (see
    :func:`tessera_embeddings.providers.aws.ray.resolve_ami_id`). Region ``None`` →
    us-west-2, the Ray provisioning region (where the AMI SSM parameter lives), NOT
    the storage ``s3_region``. Kept thin so the flow imports on non-AWS machines.
    """
    from tessera_embeddings.providers.aws.ray import resolve_ami_id

    return resolve_ami_id(ami_ssm_name, region or "us-west-2")


@flow(
    name="run-global-campaign",
    on_cancellation=[_cancel_children_on_cancellation],
    on_crashed=[_cancel_children_on_cancellation],
)
async def run_global_campaign(
    *,
    paths: BucketPaths,
    ami_ssm_name: str,
    branch: str | None = None,
    fill_deployment: str | None = None,
    store_name: str = "tessera",
    years: tuple[int, ...] | None = None,
    zones: list[str] | None = None,
    max_parallel_zones: int = 8,
    fill_strategy: str = "cluster-per-zone",
    chained_fill_deployment: str | None = None,
    commit_limit_name: str = "tessera-global-commits",
    num_actors: int = 20,
    s1_orbit: str = "both",
    s3_region: str | None = None,
    ssm_prefix: str = "/tessera/ray/",
    cloudwatch_log_group: str = "/ec2/tessera/ray",
    code_bucket: str | None = None,
    code_suffix: str = "",
    # Campaign-triggered per-zone ingestion
    ingest: bool = True,
    ingest_deployment: str | None = None,
    mask_name: str = "global",
    max_parallel_ingest: int = 2,
    cleanup_mosaics: bool = True,
    ingest_settings: IngestSettings = IngestSettings(),  # noqa: B008
    allow_partial_window: bool = False,
    allow_s2_only: bool = False,
    allow_model_mismatch: bool = False,
    sweep_orphan_mosaics: bool = False,
) -> dict[str, Any]:
    """Fill every pending (zone, year), year-serial with bounded zone parallelism.

    Args:
        paths: Deployment storage contract.
        ami_ssm_name: SSM parameter name for the Ray GPU AMI (forwarded to fills).
        branch: Route child deployments to their branch-scoped variants. ``None``
            (production) leaves every ref at its unsuffixed default — behaviour is
            byte-for-byte unchanged. A slug (e.g. ``"global-tessera"``) suffixes
            every *derived* child deployment name with ``-<slug>``: the fill, the
            ingest, and — crucially — the S1/S2 grandchildren that
            ``ingest_zone_year`` dispatches (unreachable otherwise, so a branch
            campaign with ``ingest=True`` used to 404 one level down). An
            explicitly-passed ``fill_deployment``/``ingest_deployment`` is used
            verbatim and never re-suffixed; only the defaults are derived. This is
            orthogonal to ``code_suffix`` (which scopes the source tarball, not
            deployment names).
        fill_deployment: ``flow-name/deployment-name`` of the fill deployment.
            ``None`` (default) derives ``fill-zone-year/fill-zone-year`` routed by
            ``branch``; an explicit value is used verbatim.
        store_name: Global-store repo basename.
        years: Campaign years to drive (default: all campaign years).
        zones: Restrict the fill chain (inference + assembly) to these UTM zones in
            the ergonomic ``"<1-60><N|S>"`` form, e.g. ``["33N", "15S"]``; ``None``
            (default) drives all 120. Either way only cells still needing work are
            dispatched, so a default re-run of a partially-complete year skips the
            finished zones and fills only the unfinished ones (see
            :func:`campaign_work_list`).
        max_parallel_zones: Bounds simultaneous Ray clusters within a year (a
            cost knob, distinct from the commit gate): the concurrent per-cell
            fill runs under ``"cluster-per-zone"``, or the shard/cluster count
            under ``"chained-clusters"``.
        fill_strategy: Named for the CLUSTER LIFECYCLE — both strategies run
            up to ``max_parallel_zones`` zones at once. ``"cluster-per-zone"``
            (default) dispatches one ``fill-zone-year`` run per cell, each
            provisioning its own short-lived Ray cluster.
            ``"chained-clusters"`` dispatches up to ``max_parallel_zones``
            ``fill-zones-sequential`` runs per year, each owning ONE
            long-lived Ray cluster whose actors stream through a
            size-balanced shard of the year's zones — strictly ordered, the
            next zone's tiles interleaving only at queue exhaustion, so zone
            tails never idle the fleet and there is no per-zone teardown,
            actor churn, or model reload. Amortizes ``ray up`` + per-worker
            bringup + model-load cold starts across the whole shard; ingest
            look-ahead and trailing assembly cover the remaining seams (see
            the runner's module docstring). ``max_parallel_zones=1``
            degenerates to a single cluster for the whole year.
        chained_fill_deployment: ``flow-name/deployment-name`` of the chained
            fill deployment (``fill_strategy="chained-clusters"`` only). ``None``
            (default) derives ``fill-zones-sequential/fill-zones-sequential``
            routed by ``branch``; an explicit value is used verbatim.
        commit_limit_name: Prefect global concurrency limit bounding fleet-wide
            simultaneous committers (D6); forwarded to every fill.
        num_actors: GPU actor count, forwarded to each fill.
        s1_orbit: S1 orbit selection, forwarded to both ingest and fill.
        s3_region: Optional S3 region for the global store, forwarded to the driver's
            own reads and to each fill so a non-default-region deployment works.
        ssm_prefix: SSM prefix for Ray resources, forwarded to each fill.
        cloudwatch_log_group: CloudWatch log group, forwarded to each fill.
        code_bucket: Source-tarball bucket, forwarded to each fill.
        code_suffix: Source-tarball suffix, forwarded to each fill.
        ingest: Trigger per-zone ingestion (``ingest_zone_year``) before each
            fill (default). Set False when the mosaics already exist upstream.
        ingest_deployment: ``flow-name/deployment-name`` of the ingest deployment.
            ``None`` (default) derives ``ingest-zone-year/ingest-zone-year`` routed
            by ``branch``; an explicit value is used verbatim. The S1/S2 refs it
            dispatches are always branch-derived (see ``branch``).
        mask_name: Coverage-store basename, forwarded to ingest.
        max_parallel_ingest: Max concurrent ingests (each provisions its own Dask
            cluster) — a separate, smaller knob than the fill cap. Under
            ``"chained-clusters"`` it is divided across the shards as their
            ingest look-aheads with a floor of one per shard, so with more
            shards than this cap the effective ceiling is the shard count
            (warned at dispatch); a hard fleet-wide cap needs a concurrency
            limit on the ingest deployment itself.
        cleanup_mosaics: Delete ``mosaics/{zone}/{year}`` (all versions) after the
            fill lands (default; the mosaic is a transient input). Keep for dev.
        ingest_settings: Grouped ingest tuning knobs (worker bounds, S2
            coverage threshold, S1 batch window), forwarded verbatim to every
            ingest — see :class:`tessera_embeddings.config.ingest.IngestSettings`.
        allow_partial_window: Relax the coverage gate (ingest + fill) to
            "non-empty" for legitimately partial edge zones.
        allow_s2_only: Forwarded to every fill: embed S2-valid pixels with ZERO
            S1 observations (sub-zone SAR coverage gaps) via the upstream v1.1
            missing-S1 convention instead of skipping them. PER-PIXEL only —
            zone-level SAR gates stay strict (a zone with no SAR at all still
            fails loudly; that signals an ingest bug). Folded into the staging
            run_id so a retry across a flipped flag never resumes mixed tiles.
            S2-only pixel quality is unvalidated (see the optional-S1 ADR);
            affected pixels are identifiable via s1_*_obs_count == 0.
        allow_model_mismatch: Proceed even though the store was seeded for a
            different encoder/checkpoint than this build embeds with. The gate runs
            ONCE up front, before any ingest is dispatched, so a mismatch costs a
            metadata read rather than a mosaic per in-flight zone. Deliberate
            override only — mixing encoders under one store is permanent.
        sweep_orphan_mosaics: Before the run, delete mosaics for cells that are
            already complete+tagged in scope — recovering orphans left by a
            per-cell cleanup that failed after tagging (that cell is no longer in
            `work`, so it is never retried otherwise). Off by default; the zero-cost
            backstop for transient mosaics is an S3 lifecycle rule on the mosaics
            prefix, which this complements for immediate reclamation.

    Returns:
        Summary: pending count at start, dispatched run ids per year, totals.
    """
    log = get_run_logger()
    # Stamped on every deployment this campaign dispatches so the terminal hooks can
    # find them from the flow-run id alone (see _child_tag). None outside a Prefect
    # run — a direct .fn() call in tests dispatches nothing worth sweeping.
    _tags = [t] if (t := _child_tag(flow_run_ctx.id)) else None
    # Resolve branch-scoped deployment names once. A default (None) child ref is
    # derived from `branch`; an explicit ref passes through verbatim. The S1/S2
    # grandchildren are derived from the same `branch` in `_ingest_params`.
    fill_deployment = fill_deployment or _dpl("fill-zone-year/fill-zone-year", branch)
    ingest_deployment = ingest_deployment or _dpl("ingest-zone-year/ingest-zone-year", branch)
    chained_fill_deployment = chained_fill_deployment or _dpl("fill-zones-sequential/fill-zones-sequential", branch)
    if max_parallel_zones < 1:
        raise ValueError(f"max_parallel_zones must be >= 1, got {max_parallel_zones} (Semaphore(0) blocks forever)")
    if max_parallel_ingest < 1:
        raise ValueError(f"max_parallel_ingest must be >= 1, got {max_parallel_ingest} (Semaphore(0) blocks forever)")
    # Checked HERE, not left to run_inference: the child validates only after both fill
    # strategies have entered ray_cluster, and the chained strategy has primed its
    # look-ahead ingests first. A typo would otherwise buy a Ray head and a round of
    # multi-hour ingests before failing deterministically on a value known up front.
    if num_actors < 1:
        raise ValueError(f"num_actors must be >= 1, got {num_actors} (no actor would ever run inference)")
    if fill_strategy not in ("cluster-per-zone", "chained-clusters"):
        raise ValueError(f"fill_strategy must be 'cluster-per-zone' or 'chained-clusters', got {fill_strategy!r}")
    campaign_years = tuple(years) if years is not None else CAMPAIGN_YEARS

    # Lazy AWS import so the flow file imports on non-AWS machines (arch tests).
    # The driver reads the global store directly (status, tags, on-axis probe)
    # BEFORE any child flow is dispatched, so a deployment whose store authenticates
    # only through the callback needs it here too — not just in the children.
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials

    store_path = paths.global_store(store_name)
    land_mask_path = paths.land_mask_store(mask_name)
    repo = open_global_repo(store_path, get_credentials=iam_icechunk_credentials, region=s3_region)
    # ONCE, before any dispatch. Each fill re-checks this, but by then the campaign has
    # already paid for that cell's ingest — and it dispatches cells concurrently, so a
    # store seeded for a different encoder buys a multi-terabyte mosaic per in-flight
    # zone before the first fill fails. Those mosaics are retained on failure (that is
    # what makes a resume cheap), so the disk stays occupied too. A metadata-only read.
    _assert_seeded_model_matches(
        store_path,
        build_checkpoint=checkpoint_filename(),
        allow_model_mismatch=allow_model_mismatch,
        get_credentials=iam_icechunk_credentials,
        s3_region=s3_region,
    )
    status = campaign_status(repo, years=campaign_years)
    existing_tags = set(repo.list_tags())

    # Reject off-axis years up front, before dispatching any ingest/fill: the time
    # axis is fixed at seeding (ADR-008 D1), so e.g. `years=(2026,)` against a
    # 2017-2025 store would otherwise ingest + provision Ray for every zone only
    # for the fill to reject each one. Validate against a seeded zone's axis.
    seeded_zones = sorted(status.zones)
    if seeded_zones:
        off_axis = [
            y
            for y in campaign_years
            if not zone_year_on_axis(
                store_path, seeded_zones[0], y, get_credentials=iam_icechunk_credentials, s3_region=s3_region
            )
        ]
        if off_axis:
            raise ValueError(
                f"Year(s) {off_axis} are not on the store's pre-allocated axis "
                f"(fixed at seeding, ADR-008 D1) — reseed the store or drop them from `years`."
            )

    # Normalize the requested zones to canonical common names ("33N", "07S") so a
    # bad id fails loudly here and matches the store's group names exactly.
    expected_zones = [canonicalize_zone(z) for z in zones] if zones is not None else None
    # An explicit zones subset must be seeded: a typo that canonicalizes to a valid
    # but unseeded zone would otherwise be treated as pending and ingest+fill it
    # only for the fill to reject an unseeded group. (The default None spans all 120;
    # a not-fully-seeded store surfaces at the fill, as before.)
    if expected_zones is not None:
        unseeded = [z for z in expected_zones if z not in status.zones]
        if unseeded:
            raise ValueError(f"Requested zone(s) {unseeded} are not seeded in {store_path} — run the seed flow first.")

    # Work list = cells still needing a fill, restricted to `zones` (default: all
    # 120). campaign_work_list is tag-aware: it skips landed-and-tagged cells (so a
    # default re-run of a partially-complete year only touches the unfinished zones)
    # and INCLUDES complete-but-untagged cells (crash between fill and tag → the
    # runner's idempotent retag path runs). See its docstring for the two use cases.
    work = campaign_work_list(status, existing_tags, expected_zones=expected_zones, years=campaign_years)

    # Every work cell must be seeded before we dispatch. The explicit-zones guard
    # above covers a requested subset, but the default `zones=None` expands to all
    # 120 in campaign_work_list — so a partially-seeded store would otherwise ingest
    # each unseeded cell (expensive) only for the fill to reject the missing group.
    # Validate the actual work zones here to catch that before any ingestion.
    unseeded_work = sorted({z for z, _ in work if z not in status.zones})
    if unseeded_work:
        raise ValueError(
            f"Zone(s) {unseeded_work} in the work list are not seeded in {store_path} — run the seed flow "
            "first (a partially-seeded store would ingest each cell only for the fill to reject it)."
        )

    # Orphan-mosaic recovery: a per-cell cleanup that failed after tagging leaves
    # the mosaic behind, and that cell is no longer in `work`, so it is never
    # retried. Sweep complete-and-tagged cells in scope (best-effort) before the
    # run. Opt-in — the recommended zero-cost backstop is an S3 lifecycle rule.
    if sweep_orphan_mosaics and ingest and cleanup_mosaics:
        work_set = set(work)
        sweep_zones = expected_zones if expected_zones is not None else sorted(status.zones)
        swept = 0
        for y in set(campaign_years):
            for z in sweep_zones:
                if (z, y) not in work_set and status.has(z, y) and zone_year_tag(z, y) in existing_tags:
                    delete_prefix(f"{paths.inputs.rstrip('/')}/mosaics/{z}/{y}", log=log)
                    swept += 1
        log.info("Orphan-mosaic sweep: reclaimed %d completed-cell mosaic prefix(es)", swept)

    log.info(
        "Campaign: %d (zone, year) cell(s) need work across %d year(s); <=%d concurrent fills/year, commit limit %r",
        len(work),
        len({y for _, y in work}),
        max_parallel_zones,
        commit_limit_name,
    )

    code_identity_cache: list[str] = []

    def _code_identity() -> str:
        # Resolve the immutable code artifact (AMI ID + optional tarball ETag) once and
        # reuse it for every cell — the AMI/tarball are campaign-wide constants, so a
        # single SSM/S3 lookup suffices. Lazy so a retag-only or all-ocean campaign (no
        # real fingerprint computed) makes no AWS call.
        #
        # region=None → us-west-2, the region the fill's ray_cluster PROVISIONS from
        # (its default; the campaign does not thread a Ray region through). The AMI SSM
        # parameter and the source tarball live in that region — NOT the storage
        # `s3_region` (which may differ for a non-default-region store). Resolving here
        # with s3_region would look up the AMI in the wrong SSM namespace and either
        # fail or fingerprint an artifact the workers never boot.
        if not code_identity_cache:
            # Pin the AMI component to the SAME id every fill provisions (below), so
            # the fingerprint and the booted image can't disagree via two separate
            # SSM reads at different instants.
            code_identity_cache.append(_resolve_code_identity(ami_ssm_name, code_bucket, code_suffix, None, _ami_id()))
        return code_identity_cache[0]

    ami_id_cache: list[str] = []

    def _ami_id() -> str:
        # Resolve the worker AMI ID ONCE per campaign and PIN it into every fill's
        # provisioning (below), so a mid-campaign re-bake that repoints ami_ssm_name
        # can't make a fill boot a different image than its staging fingerprint (which
        # keys on this same resolved id via _code_identity) recorded. Same region
        # rationale as _code_identity (None → us-west-2, the Ray provisioning region).
        if not ami_id_cache:
            ami_id_cache.append(_resolve_ami_id(ami_ssm_name, None))
        return ami_id_cache[0]

    def _fill_params(zone: str, year: int, run_id: str, *, needs_cluster: bool) -> dict[str, Any]:
        return {
            "zone": zone,
            "year": year,
            "paths": paths.model_dump(),
            # run_id is computed by the caller (post-ingest): an input-fingerprinted id
            # for a real fill (so a retry resumes staging only on identical inputs), or
            # a stable "-retag" id for an already-complete cell (which does no
            # inference/staging and must not inspect a possibly-cleaned-up mosaic).
            "run_id": run_id,
            "ami_ssm_name": ami_ssm_name,
            # Pin the exact image the staging fingerprint recorded (see _ami_id), so
            # the fill can't re-resolve ami_ssm_name to a re-baked AMI mid-campaign.
            #
            # ONLY for a cell that will actually start Ray. Retag-only and all-ocean
            # cells never provision, and _code_identity is deliberately lazy for the
            # same reason: a pure tag-repair or all-ocean campaign should need no SSM
            # read at all, and resolving here would make one — turning a missing
            # parameter or an SSM-less role into a failure on precisely the cheap
            # recovery path that has no use for the answer. None leaves the fill's own
            # fallback in place, which those cells never reach either.
            "ami_id": _ami_id() if needs_cluster else None,
            "store_name": store_name,
            "mask_name": mask_name,
            "num_actors": num_actors,
            "s1_orbit": s1_orbit,
            "s3_region": s3_region,
            "commit_limit_name": commit_limit_name,
            "allow_partial_window": allow_partial_window,
            "allow_s2_only": allow_s2_only,
            # Divide the fleet S3-PUT budget across concurrent fills so K shard-write
            # phases don't burst K times the target PUTs (the ~800-req SlowDown). D6
            # gates committers; this bounds the ungated upload phase.
            "s3_concurrency": max(1, TARGET_AGGREGATE_S3_CONCURRENCY // max_parallel_zones),
            "ssm_prefix": ssm_prefix,
            "cloudwatch_log_group": cloudwatch_log_group,
            "code_bucket": code_bucket,
            "code_suffix": code_suffix,
        }

    _ingest_params = partial(
        _ingest_dispatch_params,
        paths=paths,
        mask_name=mask_name,
        s1_orbit=s1_orbit,
        ingest_settings=ingest_settings,
        allow_partial_window=allow_partial_window,
        s3_region=s3_region,
        branch=branch,
    )

    fill_sem = asyncio.Semaphore(max_parallel_zones)
    ingest_sem = asyncio.Semaphore(max_parallel_ingest)
    # Bound the cells holding a live mosaic (ingested but not yet cleaned) so
    # ingestion cannot run ahead of fills and pile up dozens of multi-TB mosaics —
    # ADR-011's "peak input storage bounded by in-flight cells". Acquired BEFORE
    # ingest and held through fill + cleanup; sized to fills-at-capacity plus an
    # ingest look-ahead, so retained mosaics peak at ~this sum, not the whole year.
    inflight_sem = asyncio.Semaphore(max_parallel_zones + max_parallel_ingest)

    async def _process(zone: str, year: int) -> str:
        # Ingest (if enabled) → fill → drop the transient mosaic. The outer
        # inflight_sem (held across ALL three) backpressures ingestion by fill
        # throughput so mosaics don't accumulate; ingest_sem/fill_sem cap the
        # expensive Dask/Ray clusters within.
        #
        # A complete-but-untagged cell (in years_complete, missing its tag) is in
        # the work list only so the fill re-creates the tag — no inference, no
        # mosaic. Skip ingest and cleanup for it entirely.
        async with inflight_sem:
            retag_only = status.has(zone, year)
            did_ingest = ingest and not retag_only
            if did_ingest:
                async with ingest_sem:  # each ingest provisions its own Dask cluster
                    irun = await arun_deployment(ingest_deployment, parameters=_ingest_params(zone, year), tags=_tags)
                    _check_completed(irun, f"ingest {zone}-{year}")
            # An all-ocean cell (no live tiles) produces no mosaic: ingest returns
            # skipped_ocean, and the fill marks it empty + tags with no staging/inference
            # (the same signal the fill preflights on). There is nothing to fingerprint —
            # _staging_run_id would raise "No mosaic stores found" and strand the whole
            # year — and nothing to clean up. Give it a stable, mosaic-free id. Checked
            # for every non-retag cell (one cheap bitmap GET); a live cell falls through
            # to the input-fingerprinted id. Covers ingest=False empty cells too.
            empty_cell = not retag_only and not zone_has_live_tiles(
                land_mask_path, zone, get_credentials=iam_icechunk_credentials, s3_region=s3_region
            )
            # Compute the staging run_id AFTER ingest so ingest=True fingerprints the
            # freshly-written mosaic marker. A retag-only cell does no inference/staging
            # and its mosaic may already be cleaned up — give it a stable id that does
            # NOT inspect the mosaic (fingerprinting would raise before the tag repair).
            if retag_only:
                run_id = f"{zone}-{year}-retag"
            elif empty_cell:
                run_id = f"{zone}-{year}-empty"
            else:
                run_id = _staging_run_id(
                    zone,
                    year,
                    inputs_bucket=paths.inputs,
                    min_valid_coverage=ingest_settings.min_valid_coverage,
                    s1_orbit=s1_orbit,
                    allow_partial_window=allow_partial_window,
                    allow_s2_only=allow_s2_only,
                    code_identity=_code_identity(),
                    get_credentials=iam_icechunk_credentials,
                    s3_region=s3_region,
                )
            async with fill_sem:  # bound concurrent fills (hence Ray clusters) within a year
                frun = await arun_deployment(
                    fill_deployment,
                    parameters=_fill_params(zone, year, run_id, needs_cluster=not (retag_only or empty_cell)),
                    tags=_tags,
                )
                _check_completed(frun, f"fill {zone}-{year}")
            # Only delete a mosaic THIS campaign produced: with ingest=False the mosaic
            # is an upstream input we must not remove; an empty cell produced none.
            # Offload the blocking s5cmd/fsspec delete to a thread so a multi-TB mosaic
            # teardown doesn't stall the event loop (freezing every other zone's
            # deployment waits) while it runs.
            if did_ingest and not empty_cell and cleanup_mosaics:
                await asyncio.to_thread(delete_prefix, f"{paths.inputs.rstrip('/')}/mosaics/{zone}/{year}", log=log)
            return str(frun.id)

    # Descending: the campaign fills the current year first, then backwards
    # (ADR-008) — deliver the most-recent year soonest. Override via `years`.
    # Dedupe (campaign_work_list already dedupes years, so a duplicate here would
    # re-dispatch the same static work entries against a non-refreshed status).
    runs_by_year: dict[int, list[str]] = {}
    for year in sorted(set(campaign_years), reverse=True):
        year_zones = [z for z, y in work if y == year]
        if year_zones and fill_strategy == "chained-clusters":
            # Up to max_parallel_zones child runs fill the year, each owning
            # ONE shared Ray cluster and draining its own zone shard strictly
            # sequentially — clusters take up their next zone without actor
            # churn instead of being torn down per cell. Shards are
            # size-balanced (LPT over live-tile counts) so they finish
            # together. Ingest, per-cell run_ids, retag/empty triage, and
            # mosaic cleanup all live inside the children (their ingest
            # look-ahead pipelines mosaics against each cluster), so the
            # per-cell _process chain is bypassed; the global ingest bound is
            # preserved by dividing max_parallel_ingest across the shards.
            shards = _partition_by_live_tiles(
                year_zones,
                min(max_parallel_zones, len(year_zones)),
                land_mask_path=paths.land_mask_store(mask_name),
                known_complete={z for z in year_zones if status.has(z, year)},
                get_credentials=iam_icechunk_credentials,
                s3_region=s3_region,
            )
            log.info(
                "Year %d: dispatching %d chained fill(s) for %d zone(s): %s",
                year,
                len(shards),
                len(year_zones),
                [len(sh) for sh in shards],
            )
            # No silent caps: each shard needs a look-ahead of >= 1 to make
            # progress, so with more shards than max_parallel_ingest the true
            # concurrent-ingest ceiling is len(shards), not the requested cap —
            # only a cross-run gate (a concurrency limit on the ingest
            # deployment) can enforce it below one-per-shard.
            if ingest and len(shards) > max_parallel_ingest:
                log.warning(
                    "chained-clusters: %d shards each hold an ingest look-ahead of >= 1, so up to %d "
                    "ingest Dask clusters may run concurrently — exceeding max_parallel_ingest=%d. "
                    "Reduce max_parallel_zones, or set a concurrency limit on the ingest deployment "
                    "for a hard fleet-wide cap.",
                    len(shards),
                    len(shards),
                    max_parallel_ingest,
                )

            def _shard_needs_cluster(shard: list[str], chained_year: int) -> bool:
                """True if any cell in the shard will actually provision Ray."""
                return any(
                    not status.has(z, chained_year)
                    and zone_has_live_tiles(
                        land_mask_path, z, get_credentials=iam_icechunk_credentials, s3_region=s3_region
                    )
                    for z in shard
                )

            def _chained_params(shard: list[str], n_shards: int, chained_year: int) -> dict[str, Any]:
                return {
                    "zones": shard,
                    "year": chained_year,
                    "paths": paths.model_dump(),
                    "ami_ssm_name": ami_ssm_name,
                    # Pin the fingerprinted image (see _ami_id / _fill_params) — but
                    # ONLY for a shard that will actually start Ray. A shard whose
                    # cells are all already-complete (retag) or all-ocean never
                    # provisions, and forcing an SSM read there breaks the cheap
                    # recovery paths when the parameter is missing or the role lacks
                    # access. The ocean probe is one small bitmap GET per zone — the
                    # same price the per-cell path already pays — and it runs only
                    # for cells that are not already complete.
                    "ami_id": _ami_id() if _shard_needs_cluster(shard, chained_year) else None,
                    "store_name": store_name,
                    "mask_name": mask_name,
                    "num_actors": num_actors,
                    "s1_orbit": s1_orbit,
                    "s3_region": s3_region,
                    "commit_limit_name": commit_limit_name,
                    "allow_partial_window": allow_partial_window,
                    "allow_s2_only": allow_s2_only,
                    "ssm_prefix": ssm_prefix,
                    "cloudwatch_log_group": cloudwatch_log_group,
                    "code_bucket": code_bucket,
                    "code_suffix": code_suffix,
                    "ingest": ingest,
                    "ingest_deployment": ingest_deployment,
                    # Route this shard's S1/S2 ingest grandchildren (see `branch`);
                    # the direct ingest/fill refs above are already branch-resolved.
                    "branch": branch,
                    # Split the strategy-level bounds across the shards: the
                    # look-aheads together match max_parallel_ingest (the same
                    # global ingest bound the parallel strategy enforces), and
                    # each child's staging+assembly slice keeps the fleet-wide
                    # S3-PUT rate at the aggregate target.
                    "look_ahead": max(1, max_parallel_ingest // n_shards),
                    "s3_concurrency": max(1, TARGET_AGGREGATE_S3_CONCURRENCY // (2 * n_shards)),
                    "cleanup_mosaics": cleanup_mosaics,
                    "ingest_settings": ingest_settings.model_dump(),
                }

            chained_results = await asyncio.gather(
                *(
                    arun_deployment(
                        chained_fill_deployment, parameters=_chained_params(sh, len(shards), year), tags=_tags
                    )
                    for sh in shards
                ),
                return_exceptions=True,
            )
            chained_failures = [r for r in chained_results if isinstance(r, BaseException)]
            if chained_failures:
                raise RuntimeError(
                    f"year {year}: {len(chained_failures)}/{len(shards)} chained fill(s) failed "
                    f"(e.g. {chained_failures[0]})"
                )
            chained_runs = [r for r in chained_results if not isinstance(r, BaseException)]
            for frun in chained_runs:
                _check_completed(frun, f"chained fill year {year}")
            runs_by_year[year] = [str(r.id) for r in chained_runs]
            log.info("Year %d: %d chained fill(s) landed", year, len(shards))
        elif year_zones:
            log.info("Year %d: dispatching %d zone fill(s)", year, len(year_zones))
            # All zones in a year are distinct groups → safe to fill concurrently.
            # The outer loop is serial, so the SAME zone never fills two years at once.
            # return_exceptions=True so a single zone's failure doesn't abandon its
            # siblings mid-flight (default gather() would leave them running orphaned);
            # we let the whole year settle, then fail loudly with every failure.
            results = await asyncio.gather(*(_process(z, year) for z in year_zones), return_exceptions=True)
            failures = [(z, r) for z, r in zip(year_zones, results, strict=True) if isinstance(r, BaseException)]
            if failures:
                raise RuntimeError(
                    f"year {year}: {len(failures)}/{len(year_zones)} zone fill(s) failed "
                    f"(e.g. {failures[0][0]}: {failures[0][1]})"
                )
            runs_by_year[year] = [str(r) for r in results]
            log.info("Year %d: %d fill(s) landed", year, len(runs_by_year[year]))

        # Backfill the year-complete milestone/retention tag even when NO fills
        # ran this pass — a prior run may have completed every zone for the year
        # but crashed before tagging. The milestone is defined over ALL 120 zones
        # (expected_zones defaults to the full set, NOT this run's `zones` subset,
        # so a subset/repair run never stamps the global tag prematurely — and
        # tags are write-once). ValueError = not yet all-120 complete → defer.
        try:
            year_tag = tag_year_complete(repo, year)
            log.info("Year %d tagged complete (all 120 zones): %s", year, year_tag)
        except ValueError as exc:
            log.debug("Year %d not yet complete across all 120 zones (%s) — milestone tag deferred", year, exc)

    dispatched = sum(len(v) for v in runs_by_year.values())
    log.info("Campaign dispatch complete: %d fill run(s) across %d year(s)", dispatched, len(runs_by_year))
    return {"work_at_start": len(work), "dispatched": dispatched, "runs_by_year": runs_by_year}
