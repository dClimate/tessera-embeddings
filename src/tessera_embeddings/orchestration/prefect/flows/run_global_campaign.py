"""Drive the global embeddings campaign: fill every pending (zone, year).

Reads live fill progress from the seeded store
(:func:`~tessera_embeddings.storage.campaign.campaign_status`) and dispatches every pending
cell via ``arun_deployment`` — size-balanced ``fill-zones-sequential`` runs, each owning one
long-lived Ray cluster (``fill_strategy="chained-clusters"``, the default), or one
``fill-zone-year`` run per cell (``"cluster-per-zone"``) — mirroring
:mod:`tessera_full_pipeline`'s driver pattern.

**Ingest runs WIDE and ahead of inference, under ONE fleet-wide cap.** Ingest is the cheap
half and scales better across many narrow fleets than a few wide ones, so the only limit on
it is ``max_parallel_ingest``: the UTM zones that may ingest simultaneously across the whole
campaign, however many clusters are running. Backpressure against fill throughput is
deliberately absent — a year's mosaics, hundreds of terabytes and transient, may accumulate
ahead of the fills that consume them — but cleanup is not: ``cleanup_mosaics`` deletes each
one as its fill lands and ``sweep_orphan_mosaics`` collects what a crash left. The cap lives
on the Prefect server as a global concurrency limit (``ingest_limit_name``) because the
clusters are separate flow runs; it is upserted from ``max_parallel_ingest`` at start so the
two cannot drift, and each cluster takes an even share of it as its own window rather than
racing for slots.

GPUs are the resource that must not wait, and DENSITY ORDERING keeps them fed. Zones are
dealt to clusters densest-first, so each opens on one of the year's biggest zones and tapers
towards sparse ones: a cluster waits for that opening zone alone before requesting a fleet,
never booting GPUs speculatively or paying for a whole cluster's ingest up front, and the
opening zone is dense enough that the rest of the window lands behind it. Inference is slower
than ingest in almost every case, so the stream does not run dry.

**Scheduling (ADR-008 D6 + the runner contract).** Inference is parallel across zones; only
commits contend, and only *same-zone* fills conflict (shared ``years_complete``/``runs``
attrs → ``RebaseFailedError``). By default (``overlap_years``) every requested year is ONE
batch partitioned by ZONE, so a zone's years all land in one cluster and serialize on its
single trailing assembly thread — which keeps same-zone fills apart without a year barrier.
``overlap_years=False`` restores the year-by-year outer loop, dispatching each year's zones
concurrently, where distinct zones make same-zone overlap impossible by construction. Either
way ``max_parallel_clusters`` bounds concurrency. Commits are UNGATED: at 10 clusters the
committer ceiling is ~2N=20 on a single branch tip, measured at ~2.2 s commits
(``context_docs/storage/writing-to-the-global-store.md``). ``pending()`` is year-major, the
drain pattern the barrier path relies on.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable, Mapping
from functools import partial
from logging import LoggerAdapter
from typing import Any

import icechunk
import zarr
from prefect import flow, get_run_logger
from prefect.client.orchestration import get_client
from prefect.deployments import arun_deployment
from prefect.runtime import flow_run as flow_run_ctx
from prefect.states import StateType

from tessera_embeddings.config.inference import checkpoint_filename, inference_code_identity
from tessera_embeddings.config.ingest import IngestSettings
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.inference.assembly import TARGET_AGGREGATE_S3_CONCURRENCY
from tessera_embeddings.inference.data_loading import _active_orbits
from tessera_embeddings.orchestration.prefect.flows._child_runs import (
    CANCELLATION_CONFIRM_S,
    _check_completed,
    child_run_tag,
    make_child_cancel_hook,
)
from tessera_embeddings.orchestration.prefect.flows._overrides import set_overrides
from tessera_embeddings.orchestration.prefect.flows.fill_zone_year import (
    _assert_seeded_model_matches,
    _optical_min_obs_from_store,
)
from tessera_embeddings.orchestration.prefect.flows.ingest_zone_year import IngestDeployments
from tessera_embeddings.orchestration.runners.zone_fill import (
    zone_has_live_tiles,
    zone_work_weight,
    zone_year_on_axis,
)
from tessera_embeddings.storage.campaign import (
    CampaignStatus,
    campaign_status,
    campaign_work_list,
    tag_year_complete,
    zone_year_tag,
)
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.object_store import delete_prefix
from tessera_embeddings.storage.zarr_store import is_missing_repo, open_store_group_and_tip
from tessera_embeddings.storage.zone_grid import CAMPAIGN_YEARS, canonicalize_zone

# S1/S2 grandchild ingest refs derive from this single source rather than duplicated string
# literals, so a rename of the ingest deployments stays in ingest_zone_year.IngestDeployments.
_INGEST_DEPLOYMENT_DEFAULTS = IngestDeployments()


#: Tag prefix for every deployment this campaign dispatches — ingests, per-cell
#: fills and chained-fill clusters alike.
_CHILD_TAG_PREFIX = "campaign"


def _child_tag(flow_run_id: object) -> str | None:
    """Deterministic tag for this campaign's dispatched children."""
    return child_run_tag(_CHILD_TAG_PREFIX, flow_run_id)


#: Cancelling the campaign must stop what it started. Each `arun_deployment` creates an
#: INDEPENDENT run, so killing this flow otherwise leaves its many in-flight ingests and
#: fills running — writing mosaics and store attrs a retry will race, with their Dask and Ray
#: fleets still billing. Registered as both terminal hooks because a crashed parent orphans
#: children exactly like a cancelled one. The child flows keep their OWN teardown hooks; this
#: one stops the runs those hooks then clean up after.
_cancel_children_on_cancellation = make_child_cancel_hook(_CHILD_TAG_PREFIX, "campaign child run")


#: Terminal states after which a chained fill has PROVABLY stopped writing — the only ones an
#: immediate replacement may follow.
#:
#: The proof is a chain of ``finally`` blocks, not an assumption about the state name: either
#: state means the fill's function returned or raised, so the runner joined its trailing
#: assembly thread (the only thread that commits to the embeddings store), then the flow
#: cancelled its child ingests and WAITED for them to confirm terminal before tearing its
#: fleet down — all before the state is set.
#:
#: ``CRASHED`` and ``CANCELLED`` carry no such proof: the process that would run the
#: ``finally`` is the one that died, and a crash verdict can come from missed heartbeats
#: while the run is still writing. Those cells wait for the dispatch round to close.
#: Re-dispatching them promptly needs fencing at the WRITE — a lease the committer checks —
#: rather than an inference about who has stopped.
_QUIESCENT_TERMINAL_STATES = frozenset({StateType.FAILED, StateType.COMPLETED})

#: How long a settled fill's cells are left alone before a replacement may be dispatched.
#:
#: DERIVED, not chosen. Cancellation is asynchronous and the run tree is three deep: the
#: fill, its zone ingests, their S1/S2 grandchildren. The fill's own teardown already spends
#: :data:`CANCELLATION_CONFIRM_S` on level two before the state this driver reads is set —
#: but that wait GIVES UP when its budget runs out, and level three is only ever asked, by a
#: hook that does not block. This is that same budget once more, for what those two left
#: unfinished. Counted from the cancellation request the two come to twice this, which is
#: the interval the crash-recovery record recommends between a run's death and re-dispatching
#: its cells; deriving it from the mechanism means it follows if the confirmation budget
#: changes.
#:
#: Time rather than inspection, because a census can only report what was true a moment ago
#: and cannot make a lingering child stop. This delay is what lets the two mechanisms that do
#: act — the teardown that waits for children, and the orphan sweep — finish.
_SETTLE_DELAY_S = CANCELLATION_CONFIRM_S


def _still_running(live: dict[asyncio.Task[Any], int | None]) -> bool:
    """Is any dispatch in ``live`` genuinely unfinished?

    Not ``bool(live)``: a completed task stays in the map until the drain loop's next pass,
    so a truthiness test would claim "a sibling is running" about work that has stopped.
    Only ``done()`` is the authority, and the immediate-refill decision rests on it — a
    replacement is worth an extra fleet only while something else holds the round open.
    """
    return any(not task.done() for task in live)


def _upsert_limit(name: str, limit: int, *, what: str, log: logging.Logger | LoggerAdapter) -> None:
    """Set a Prefect global concurrency limit to ``limit``.

    Both fleet-wide caps this campaign relies on — simultaneous ingests and simultaneous
    committers — are enforced by gates in CHILD flow runs on other machines, which nothing
    in-process can see across, so the number lives on the Prefect server. Writing it from
    the flow parameter makes that parameter the single source of truth, instead of a
    server-side number an operator keeps in step by hand.

    Raises on failure, deliberately. In preflight, before a cluster is dispatched, failing
    costs nothing; later it costs the choice between proceeding ungated — blowing through
    the cap this call exists to set — and stopping running clusters one by one at their
    first acquire. A limit that cannot be written is a broken precondition, not a degraded
    mode.
    """
    with get_client(sync_client=True) as client:
        client.upsert_global_concurrency_limit_by_name(name, limit)
    log.info("Fleet-wide %s limit %r set to %d", what, name, limit)


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

    The AUTHORITATIVE term is each store's branch-tip SNAPSHOT ID — Icechunk's own content
    identity, so it moves on any commit that can change the pixels a fill reads, including
    ones no bookkeeping attribute records (a re-run of one date's region write, a
    hand-repaired store, a prebuilt mosaic maintained outside this pipeline). Keyed on the
    attrs alone, a mosaic could change underneath a staging prefix and a retry would resume
    tiles built from the OLD pixels, publishing a silent mixture of two inputs.

    The attrs ride along for legibility, not safety: ``ingest_marker`` (window +
    coverage-delivery sha + min_valid_coverage + orbit + allow_partial_window — the exact
    per-(zone,year) fingerprint that produced the store) plus the ``ingest_completed_at``
    that separates two builds sharing one policy, or ``last_appended``/``created_at`` for a
    prebuilt (``ingest=False``) mosaic. They make a changed run_id diagnosable from the
    string; a store carrying none of them is still fully identified by its tip.

    Per-(zone,year) and post-ingest, NOT one global coverage sha read up front: a partial
    ``build_all(zones=...)`` can leave zones on different coverage revisions, and coverage
    can change after an early read — either would let stale staged tiles resume against a
    rebuilt mosaic. Genuinely-absent stores (missing repo / not found) are skipped.

    SCOPE: this pins the inputs at RESUME time, not at read time. The tip is sampled once
    here while the fill's actors later open each mosaic from the moving ``main``, so it
    answers "have the inputs changed since the last attempt?" and NOT "did they hold still
    while this attempt ran?". Nothing in the pipeline writes a mosaic during its own fill (a
    cell's ingest completes before it dispatches, mosaics are per-(zone,year), cleanup runs
    after), so the remaining window needs an out-of-band writer committing mid-fill. Closing
    it would mean threading these snapshot IDs into every actor store open.

    Only the ACTIVE orbit set is fingerprinted (reflectance + ``_active_orbits(s1_orbit)``),
    because the fill reads only those. An inactive opposite-orbit store that happens to be
    present (a stale markerless leftover, or a prebuilt ``ingest=False`` mosaic shipping both
    orbits) cannot affect the embeddings, yet fingerprinting it could raise (no identity) or
    perturb the run_id and needlessly restart a valid single-orbit fill.

    Module-level rather than a flow closure because both fill strategies need it: the
    cluster-per-zone dispatch here and the chained fill's ``prepare`` callable
    (:mod:`.fill_zones_sequential`).
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
            # rebuild under identical settings reproduces it byte for byte. completed_at is
            # stamped in the same commit and only moves when ingest actually re-ran.
            f"{marker!r}@{grp.attrs.get('ingest_completed_at')}"
            if isinstance(marker, dict)
            else (grp.attrs.get("last_appended") or grp.attrs.get("created_at"))
        )
        # Snapshot FIRST — it is the term that makes this safe. The attrs follow so an
        # operator diffing two run_ids can see WHY the mosaic changed, not just that it did.
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
    optical_min_obs: int | None,
    code_identity: str,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> str:
    """Deterministic staging run_id fingerprinting everything that determines the embeddings.

    Covers the config (threshold/orbit/window/checkpoint/S2-only), the INFERENCE CODE the
    fill runs (``code_identity``), and the per-(zone,year) mosaic identity
    (:func:`_mosaic_identity`). A retry with identical inputs resumes the same prefix
    (findable for cleanup); a change to any of them starts a fresh prefix, so old tiles are
    never resumed under new inputs. Call AFTER ingest so ingest=True reads the freshly-written
    marker.

    ``code_identity`` is the inference SOURCE hash only, not the whole build — the AMI is
    still resolved and pinned into every fill's provisioning, it just no longer decides
    staging reuse. Mechanism and failure modes:
    ``context_docs/storage/staging-identity-and-resume.md``.
    """
    key = (
        year,
        min_valid_coverage,
        s1_orbit,
        allow_partial_window,
        # allow_s2_only changes WHICH pixels get embeddings, so a retry across a flipped flag
        # must start a fresh prefix — resuming would mix S1-gated and S2-only tiles.
        allow_s2_only,
        # The store's minimum-depth rule, for the same reason: it decides WHICH pixels get
        # embeddings. Staging lives under `outputs/staging/{zone}/{year}`, which is not
        # namespaced by store, so re-running the same mosaics into a store with a different
        # rule would resume tiles staged under the old one and publish a mix of two depth
        # policies under one fingerprint.
        optical_min_obs,
        # Checkpoint identity is the FILENAME, not the weight bytes, deliberately. A new
        # model ships under a new filename (norm_source → a distinct name), which both flips
        # this fingerprint and is rejected by the seeded model gate (checkpoint_id /
        # geoemb:model). The checkpoint is fixed for a campaign and never overwritten
        # mid-run, so a same-filename byte-swap is a non-scenario. (Do not "fix" to an ETag.)
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
    allow_ingest_code_mismatch: bool = False,
    s3_region: str | None = None,
    branch: str | None = None,
) -> dict[str, Any]:
    """Parameter dict for one ``ingest-zone-year`` deployment run.

    Shared by both fill strategies' dispatch sites — the cluster-per-zone driver's per-cell
    chain and the chained fill's look-ahead adapter (:mod:`.fill_zones_sequential`) — so the
    two cannot drift from the deployment's signature independently. ``s3_region`` is the
    fill's region, so ingest's metadata opens (mask liveness, coverage sha, marker, coverage
    gate) hit the same stores rather than defaulting to us-west-2 on a non-default-region
    deployment. ``branch`` routes the S1/S2 grandchildren to their branch-scoped deployments
    (see :func:`_dpl`); without it a branch campaign 404s one level down on the prod refs.
    """
    return {
        "zone": zone,
        "year": year,
        "paths": paths.model_dump(),
        "mask_name": mask_name,
        "s1_orbit": s1_orbit,
        "ingest_settings": ingest_settings.model_dump(),
        "allow_partial_window": allow_partial_window,
        "allow_ingest_code_mismatch": allow_ingest_code_mismatch,
        "s3_region": s3_region,
        # Grandchild routing: the S1/S2 ingest deployments dispatched BY ingest_zone_year.
        # Without this key its IngestDeployments() defaults (unsuffixed prod refs) always
        # win. Built from the defaults model so a rename stays single-sourced.
        "deployments": {
            "ingest_s1_roi_sar": _dpl(_INGEST_DEPLOYMENT_DEFAULTS.ingest_s1_roi_sar, branch),
            "ingest_s2_roi_reflectance": _dpl(_INGEST_DEPLOYMENT_DEFAULTS.ingest_s2_roi_reflectance, branch),
        },
    }


def _partition_by_live_tiles(
    zones: list[str],
    n_clusters: int,
    *,
    land_mask_path: str,
    pending_years: Mapping[str, int] | None = None,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> list[list[str]]:
    """Split ``zones`` into ``n_clusters`` lists of ~equal total inference WORK.

    Longest-processing-time greedy: zones descending by work, each assigned to the
    currently-lightest cluster, so the sequential fill's clusters finish their zone lists at
    roughly the same time.

    **Work, not area.** Inference consumes one sequence per pixel, so cost scales with
    `pixels x observations` and observation count varies about twofold with latitude — two
    clusters with identical tile counts can carry materially different work.
    :func:`zone_work_weight` weights each live tile row by its latitude band's observation
    count, at the same one-GET cost. It matters more than it looks: clusters are long-lived
    and their finish times set the campaign's, so a heavy cluster is never averaged away, and
    the imbalance is systematic rather than random — a cluster drawing high-latitude zones is
    heavy in every year.

    **Weighted by the YEARS assigned with the zone.** A cluster receives every pending
    ``(zone, year)`` of the zones it owns, so a zone's cost to it is per-year work times the
    years it carries; weighed once, a zone missing five years would look like one missing a
    single year and the heavy cluster would drain its extra years while the rest sit idle.
    ``pending_years`` is that count, and a zone with none weighs zero — which covers the
    retag-only crash-recovery cells the child settles without GPU work, whose tile counts
    would otherwise skew the balance, and skips their mask reads. Omit it and every zone
    counts as one year. Tile counts are one ~1 KB GET per zone; ``n_clusters == 1`` skips the
    reads entirely, and empty clusters (more clusters than zones) are dropped.

    The within-cluster ORDER is deliberately untouched: the chained fill re-sorts its own
    zones by tile count (``fill_zones_sequential``), because ordering and actor clamping are
    properties of AREA — a zone needs one actor per tile however many observations each pixel
    has, and the descending sort is what lets an autoscaled fleet only ever shrink. This
    function decides *which* cluster gets a zone; the child decides *when*.

    A second property falls out of the same ordering and is load-bearing: the first
    ``n_clusters`` zones are the densest in the year and every cluster gets exactly one,
    because all totals start at zero. Each cluster therefore OPENS on a dense zone and tapers
    towards sparse ones, which is what lets its fleet start as soon as that one zone has
    ingested — the opening zone takes long enough to infer that the rest of the window lands
    behind it. Change the assignment order and that guarantee goes with it.
    """
    if n_clusters <= 1:
        return [zones]
    years = pending_years or {}
    n_years = {z: years.get(z, 1) if pending_years is not None else 1 for z in zones}
    weight = {
        z: 0.0
        if n_years[z] <= 0
        else n_years[z] * zone_work_weight(land_mask_path, z, get_credentials=get_credentials, s3_region=s3_region)
        for z in zones
    }
    clusters: list[list[str]] = [[] for _ in range(n_clusters)]
    totals = [0.0] * n_clusters
    for z in sorted(zones, key=lambda z: weight[z], reverse=True):
        i = totals.index(min(totals))
        clusters[i].append(z)
        totals[i] += weight[z]
    return [cl for cl in clusters if cl]


def _resolve_code_identity(
    ami_ssm_name: str, code_bucket: str | None, code_suffix: str, region: str | None, ami_id: str | None = None
) -> str:
    """Lazy-import wrapper over the AWS provider's code-artifact resolver.

    Module-level and thin so this flow still imports on non-AWS machines (arch tests) — boto3
    lives only under ``providers/aws`` — and tests can stub it. Region ``None`` → us-west-2,
    the region the fills' ``ray_cluster`` PROVISIONS from, where the AMI SSM parameter and
    source tarball live; NOT the storage ``s3_region``, which may differ. ``ami_id`` pins the
    AMI component to a pre-resolved id so the fingerprint matches the image the fill will
    boot. See :func:`tessera_embeddings.providers.aws.ray.resolve_code_artifact_identity`.
    """
    from tessera_embeddings.providers.aws.ray import resolve_code_artifact_identity

    return resolve_code_artifact_identity(ami_ssm_name, code_bucket, code_suffix, region or "us-west-2", ami_id=ami_id)


def _resolve_tarball_identity(code_bucket: str | None, code_suffix: str, region: str | None) -> str:
    """Lazy-import wrapper over the AWS provider's tarball-ETag resolver (see
    :func:`tessera_embeddings.providers.aws.ray.source_tarball_identity`). Empty string when
    no tarball overlays the AMI, i.e. production. Kept thin so this flow imports on non-AWS
    machines, where boto3 is absent.
    """
    if not code_bucket:
        return ""
    from tessera_embeddings.providers.aws.ray import source_tarball_identity

    return source_tarball_identity(code_bucket, code_suffix, region or "us-west-2")


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
    max_parallel_clusters: int = 10,
    launch_pacing: bool = False,
    gpu_fallback_instance_types: list[str] | None = None,
    gpu_fallback_vcpu_budget: int | None = None,
    actor_request_headroom: int | None = None,
    actor_request_batch_size: int | None = None,
    fill_strategy: str = "chained-clusters",
    chained_fill_deployment: str | None = None,
    num_actors: int = 250,
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
    max_parallel_ingest: int = 60,
    max_dispatch_rounds: int = 2,
    immediate_refill: bool = False,
    # Staging-reuse escape hatches. All THREE default off; see `_staging_code_identity`.
    force_staging_reuse: bool = False,
    force_staging_restage: str = "",
    staging_code_identity: str = "",
    overlap_years: bool = True,
    ingest_limit_name: str = "tessera-global-ingests",
    inference_pause_gate: str = "tessera-global-inference",
    cleanup_mosaics: bool = True,
    ingest_settings: IngestSettings = IngestSettings(),  # noqa: B008
    allow_partial_window: bool = False,
    allow_s2_only: bool = False,
    allow_model_mismatch: bool = False,
    allow_ingest_code_mismatch: bool = False,
    sweep_orphan_mosaics: bool = False,
    validation_deployment: str | None = None,
) -> dict[str, Any]:
    """Fill every pending (zone, year) — all years in one batch, bounded zone parallelism.

    Args:
        paths: Deployment storage contract.
        ami_ssm_name: SSM parameter name for the Ray GPU AMI (forwarded to fills).
        branch: Route child deployments to their branch-scoped variants. ``None``
            (production) leaves every ref at its unsuffixed default. A slug (e.g.
            ``"global-tessera"``) suffixes every *derived* child deployment name with
            ``-<slug>``: the fill, the ingest, and — crucially — the S1/S2 grandchildren
            ``ingest_zone_year`` dispatches, which are otherwise unreachable and 404 one
            level down on a branch campaign with ``ingest=True``. An explicitly-passed
            ``fill_deployment``/``ingest_deployment`` is used verbatim and never
            re-suffixed. Orthogonal to ``code_suffix``, which scopes the source tarball
            rather than deployment names.
        fill_deployment: ``flow-name/deployment-name`` of the fill deployment. ``None``
            (default) derives ``fill-zone-year/fill-zone-year`` routed by ``branch``; an
            explicit value is used verbatim.
        store_name: Global-store repo basename.
        years: Campaign years to drive (default: all campaign years).
        zones: Restrict the fill chain (inference + assembly) to these UTM zones in the
            ergonomic ``"<1-60><N|S>"`` form, e.g. ``["33N", "15S"]``; ``None`` (default)
            drives all 120. Either way only cells still needing work are dispatched, so a
            re-run of a partially-complete year fills only the unfinished zones (see
            :func:`campaign_work_list`).
        max_parallel_clusters: Bounds simultaneous Ray clusters within a year (a cost knob,
            distinct from the commit gate): the concurrent per-cell fill runs under
            ``"cluster-per-zone"``, or the number of Ray clusters under
            ``"chained-clusters"``. **Defaults to 10**, the campaign's planned width: 10 x
            250 actors reaches the full 2,500-actor quota while keeping each cluster's
            assembly thread under its ~275-actor ceiling, and balance holds to ~16. Not to
            be confused with ``max_parallel_ingest`` (60), which caps simultaneous
            zone-INGESTS — and 60 divides evenly by 10, which matters: the per-cluster share
            is the cap over the cluster count ROUNDED UP, so an uneven split aims the fleet
            above its own gate and logs a width it never runs at. Measurement favours many
            narrow fleets over few wide ones, so raising this means sizing ``num_actors`` and
            ``IngestSettings.max_workers`` down to match, or the aggregate fleet exceeds the
            account's EC2 quota. This, ``max_parallel_ingest``, ``num_actors`` and
            ``overlap_years`` are ONE decision and move together
            (``context_docs/campaign/campaign-plan.md`` §3). A wrong width does not fail: the
            run completes and publishes real data at a fraction of the intended rate, with
            no symptom but a wall clock nobody has a baseline for.
        launch_pacing: Pace every fill's EC2 launch requests against the account's shared
            RunInstances quota, a small burst capacity refilled at a fixed, unadjustable
            rate. It belongs at campaign level because contention is a property of the
            CAMPAIGN, not of a fill: one cluster growing alone contends with nothing, while
            ``n`` autoscalers requesting at the same moment drain the bucket and are refused.
            Forwarded to whichever fill strategy runs; ``False`` (default) keeps today's
            launch behaviour, so enabling it is a deliberate act on a running campaign.
        gpu_fallback_instance_types: EC2 instance types this campaign may fall back to when
            the production rung has no capacity, e.g. ``["g5.2xlarge"]``. Named as instance
            types rather than cards so this and the ``gpu-worker-ladder`` key speak one
            vocabulary and the size -- a host-RAM safety decision -- is explicit. ``None``
            (default) keeps the fleet on ``g6e.xlarge`` alone.

            Naming a card does two things per fill: it opens that card's rung, and it
            installs an autoscaler scorer that demotes a rung AWS has just refused for want
            of capacity. Ray's stock scorer cannot see capacity at all, so without the scorer
            an open rung is never reached; without the rung the scorer has nothing to
            promote. The fleet does not get bigger -- ``num_actors`` still bounds it -- only
            differently made. The price: fallback sizes are 8 vCPU per GPU against
            ``g6e.xlarge``'s 4, so each fallback GPU spends twice the G-and-VT quota, and
            they run at 0.46 (A10G) or 0.32 (L4) of an L40S.
        gpu_fallback_vcpu_budget: Optional per-cluster vCPU budget for the GPU fleet. The
            G-and-VT quota is counted in vCPU and the fallback cards spend twice as much of
            it per GPU, so a ceiling in CARDS means a different quota bill depending on which
            card the fleet lands on. Given a budget, each rung is ceilinged at what the
            budget affords it -- 250 cards at 4 vCPU/GPU, 125 at 8. It bounds each PURE fleet
            exactly; a mixture can still exceed it, because Ray's ceilings count nodes and
            carry no weight.
        actor_request_headroom: Hold every fill's actor request to the slots it has actually
            placed plus this many, rather than letting it climb toward ``num_actors``. The
            unbounded case is what feeds the launch quota: a fill asking for its whole target
            while a handful of instances have placed leaves the rest to be retried by its
            autoscaler for as long as the fill lives. ``launch_pacing`` makes each remaining
            request cheaper; this makes there be fewer of them, and it is the larger half.
            ``None`` keeps today's behaviour;
            :data:`~tessera_embeddings.inference.scheduling.ACTOR_REQUEST_HEADROOM` is the
            value to pass.
        actor_request_batch_size: Actors per batch for every fill, overriding the inference
            default. Config rather than code, and therefore the fastest relief available: the
            quota is a rate over CALLS, and a batch of B actors is ceil(B /
            instances-per-call) calls per cluster per placement round. ``None`` keeps the
            default.
        fill_strategy: Named for the CLUSTER LIFECYCLE — both strategies run up to
            ``max_parallel_clusters`` zones at once. ``"chained-clusters"`` (default)
            dispatches up to ``max_parallel_clusters`` ``fill-zones-sequential`` runs per
            year, each owning ONE long-lived Ray cluster whose actors stream through a
            size-balanced share of the year's zones — strictly ordered, the next zone's tiles
            interleaving only at queue exhaustion, so zone tails never idle the fleet and
            there is no per-zone teardown, actor churn or model reload. It amortizes ``ray
            up`` + per-worker bringup + model-load cold starts across the whole cluster;
            ingest look-ahead and trailing assembly cover the remaining seams (see the
            runner's module docstring). Each cluster ingests EVERY one of its cells before
            requesting GPUs, so a fleet is never billed against an unfinished ingest — which
            makes the up-front ingest scale with cluster size, and
            ``max_parallel_clusters=1`` degenerate to a whole year of ingest before any
            inference. ``"cluster-per-zone"`` dispatches one ``fill-zone-year`` run per cell
            instead, each provisioning its own short-lived Ray cluster — simpler, and it pays
            a full cluster bringup and model load per zone.
        chained_fill_deployment: ``flow-name/deployment-name`` of the chained fill deployment
            (``fill_strategy="chained-clusters"`` only). ``None`` (default) derives
            ``fill-zones-sequential/fill-zones-sequential`` routed by ``branch``; an explicit
            value is used verbatim.
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
        max_parallel_ingest: How many UTM zones may ingest simultaneously across the WHOLE
            campaign, each provisioning its own Dask cluster. The only limit on ingestion:
            clusters submit every zone they own at once and each holds one slot for the
            duration of its ingest, so a cluster starts as many as fit under this cap and
            queues the rest. Under ``"chained-clusters"`` it is enforced by
            ``ingest_limit_name``, because the clusters are separate flow runs and the cap
            has to live server-side; under ``"cluster-per-zone"`` an in-process semaphore is
            equivalent, that strategy having one driver.
        force_staging_reuse: Reuse staged tiles across an inference code change. All three
            staging levers move only the STAGING fingerprint — which S3 prefix intermediate
            tiles are written to and read back from. None of them touches the published store
            or relaxes a gate on it: a cell's completion mark, its write-once tag and its
            manifest checks are unaffected. Full mechanism:
            ``context_docs/storage/staging-identity-and-resume.md``.

            **This flag cannot reach staging created without it.** It substitutes a constant
            into the run-id hash, so it yields a *different* prefix from the one an unflagged
            run staged under, and preserves reuse only between runs that both set it. Use
            ``staging_code_identity`` to resume a prefix an earlier campaign staged.

            Set it when a change to the inference source provably cannot alter staged output
            (a log line, a comment, a type annotation) and the staging hours are worth having.
            Unsafe if that judgement is wrong: it mixes two code versions into one write-once
            zone-year, and ``assemble_global`` probes the variable set from a single tile on
            the assumption that a staging prefix is homogeneous.
        staging_code_identity: The staging fingerprint's code component, stated OUTRIGHT
            instead of derived — the only lever that can reach a prefix an earlier campaign
            staged under, precisely because it does not derive the value it is matching.
            Empty (default) derives it.

            Its case is a campaign RESTART across a code change that provably cannot alter
            staged tile CONTENT: an orchestration or scheduling fix, a timeout, a preflight
            gate. Derivation cannot see that distinction — the fingerprint covers the whole
            inference closure, so a change to how many actors are requested abandons every
            staged tile exactly as a change to the model would, and on a deployment shipping a
            source tarball re-uploading it moves the fingerprint whatever the change was.

            **Copy the value from the driver's log**, which announces it once per campaign: a
            chained fill records it as a parameter too, but a ``cluster-per-zone`` campaign
            hashes it irreversibly into its children's run ids. A value nothing was staged
            under simply starts a fresh prefix, costing re-inference and nothing else.

            It rests on the operator's judgement, the same one ``force_staging_reuse`` asks
            for. Wrong, and two code versions land in one write-once zone-year:
            ``StagedShardSource``'s per-tile validation catches the coarse form (missing
            variable, wrong shard extent, dtype mismatch, failing by name) but not
            same-shape-different-numbers, so a change to the model, the checkpoint or the
            pixel maths is never a candidate.
        force_staging_restage: An arbitrary token mixed into the staging fingerprint, so a
            change the source hash CANNOT see starts a fresh prefix — a deliberate dependency
            upgrade mid-campaign, where a new torch changes the numbers without changing our
            source. Any new value starts fresh; the same value resumes what it staged. Unlike
            ``force_staging_reuse`` this is a production tool, because abandoning stale staged
            work is always safe where reusing it across a real change is not.
        overlap_years: Drop the YEAR BARRIER — dispatch every requested year as one batch
            instead of one batch per year, so a cluster works a multi-year list and year
            N+1's ingest overlaps year N's inference. **Default ON**: it is the campaign's
            planned shape and what makes ``max_parallel_ingest`` reachable, since with the
            barrier in place the ceiling is 45 whatever the cap says. Certified on six cells
            each carrying both radar orbits, including a same-zone year rollover inside one
            cluster. What makes it safe is not the flag but two things underneath it: each
            cell carries its own inference window to the actors, and the zone-group attribute
            commit is separate and retried, so two years of one zone no longer collide. The
            partition is over ZONES, so a zone's every year lands in one cluster and its
            assemblies serialize on that cluster's single trailing thread. ``False`` restores
            the barrier, which a repair pass over one year still wants.
        max_dispatch_rounds: How many ROUNDS of re-dispatch the campaign runs — not a
            per-zone budget. Each round dispatches everything still missing, whatever zones
            that is, and re-reads the STORE to decide, so it never re-does landed work and an
            interrupted mosaic is resumed rather than rebuilt.

            **This is the outer recovery, and the only one that survives a child run dying.**
            A killed container or a cancelled run takes its in-cluster attempt counter with
            it, so nothing inside the run can retry; interruptions are EXPECTED here, because
            the orphan sweeper cancels child runs by design. The child's own
            ``attempts_per_cell_in_cluster`` covers the cheaper case where the run survives
            and one cell failed.

            Rounds stop early when one makes no progress at all — what a DETERMINISTIC
            failure looks like from out here (a coverage gate, a fingerprint mismatch, an
            unseeded group), which wants a human rather than another cluster, and which
            bounds the two budgets compounding on a cell that will never succeed. Set to 1 to
            disable retries.
        immediate_refill: Replace a settled chained fill's still-missing cells at once,
            instead of waiting for the whole round to close. **Off by default, and off is
            byte-for-byte today's behaviour** — nothing is ever added to the set of live
            dispatches, so a round waits for every cluster and reports outcomes in the
            clusters' original order.

            The round is a BARRIER: a cluster owns a roster of zones, so when one dies early
            its whole roster waits on the slowest sibling before the store is re-read. On,
            the settled cluster's cells are re-dispatched into the slot it just vacated,
            inheriting its ingest share, its committer share and its place under
            ``max_parallel_clusters`` — the fleet's width and cost do not change.

            Two conditions must BOTH hold, and either failing returns the cells to the round,
            which is never worse than the default: the predecessor reached a state that
            PROVES it stopped writing (``_QUIESCENT_TERMINAL_STATES``, so a crash, a
            cancellation or a dispatch that raised all decline), and ``_SETTLE_DELAY_S`` has
            passed since it settled, which is what the level below its own children needs.

            **Bounded per CELL for the life of the campaign, not per round.** A cell that has
            had its replacement is never eligible for another, a replacement is not itself
            eligible, and none is issued unless a sibling is still running — so a cell gains
            at most ONE attempt beyond ``max_dispatch_rounds`` and a persistent capacity
            shortage cannot become a dispatch loop. The two-writer invariant is structural: a
            replacement covers only zones its predecessor owned and the round's clusters
            partition zones disjointly, so it cannot reach a zone a live sibling holds.
            Nothing locks a cell against a dispatcher outside this campaign, which the store
            makes loud rather than silent — mosaic commits do not rebase.

            Not part of the staging fingerprint: it changes WHEN work is dispatched, not what
            is computed. Design record:
            ``context_docs/campaign/campaign-plan.md`` (§3, ``immediate_refill``).
        ingest_limit_name: Prefect global concurrency limit backing ``max_parallel_ingest``
            under ``"chained-clusters"``. The campaign upserts it to that value at start, so
            the parameter is the single place the number is written and cannot drift from
            the server's.
        inference_pause_gate: Name of the global concurrency limit that PAUSES inference.
            Upserted to 1 at start (any positive value means run) and read — never acquired —
            by every cluster's dispatch loop. **``"chained-clusters"`` only**, the campaign's
            strategy: a ``cluster-per-zone`` per-cell run has no stream to hold, so the gate
            is neither created nor forwarded there and cancelling is the only lever. Set it
            to **0** and each cluster lands the chunks it has queued and then takes on no
            further cell, keeping its actors and its run alive; set it back to 1 and they
            carry on where they stopped. Nothing fails either way and no cell is
            half-written, because a cell enters inference whole.

            **One of the two pause levers, covering different halves.** Zeroing
            ``ingest_limit_name`` stops mosaics being built; zeroing this stops embeddings
            being computed from the mosaics that exist. Zero both to wind the campaign down
            to the cells already in flight. **Neither stops the bill** — a cluster holds its
            GPU fleet for its whole multi-cell walk, so a paused fleet idles at full width.
            Only cancelling ends the spend.
        cleanup_mosaics: Delete ``mosaics/{zone}/{year}`` (all versions) after the fill lands
            (default; the mosaic is a transient input). Keep for dev. Nothing throttles
            ingest against fill throughput, so at these defaults a year's mosaics can
            accumulate faster than the fills drain them — this keeps that from being
            permanent.
        ingest_settings: Grouped ingest tuning knobs (worker bounds, S2 coverage threshold,
            S1 batch window), forwarded verbatim to every ingest — see
            :class:`tessera_embeddings.config.ingest.IngestSettings`.
        allow_partial_window: Relax the coverage gate (ingest + fill) to "non-empty" for
            legitimately partial edge zones.
        allow_s2_only: Forwarded to every fill: embed S2-valid pixels with ZERO S1
            observations (sub-zone SAR coverage gaps) via the upstream v1.1 missing-S1
            convention instead of skipping them. PER-PIXEL only — zone-level SAR gates stay
            strict, and a zone with no SAR at all still fails loudly because that signals an
            ingest bug. Folded into the staging run_id so a retry across a flipped flag never
            resumes mixed tiles. S2-only pixel quality is unvalidated (see the optional-S1
            ADR); affected pixels are identifiable via s1_*_obs_count == 0.
        allow_model_mismatch: Proceed even though the store was seeded for a different
            encoder/checkpoint than this build embeds with. The gate runs ONCE up front,
            before any ingest is dispatched, so a mismatch costs a metadata read rather than
            a mosaic per in-flight zone. Deliberate override only — mixing encoders under one
            store is permanent.
        allow_ingest_code_mismatch: Forwarded to every ingest: resume interrupted mosaics
            built by different ingest code, instead of demanding they be deleted.
        sweep_orphan_mosaics: Before the run, delete mosaics for cells already
            complete-and-tagged in scope — orphans left by a per-cell cleanup that failed
            after tagging, which are never retried because the cell has left `work`. Off by
            default; the zero-cost backstop is an S3 lifecycle rule on the mosaics prefix,
            which this complements for immediate reclamation.
        validation_deployment: ``flow-name/deployment-name`` of a deployment that checks a
            landed cell. Each fill dispatches it per cell as that cell is tagged and never
            waits on it, so a cell is validated while the next is filled — see
            :mod:`tessera_embeddings.orchestration.prefect.flows._cell_validation`.

            NOT derived from ``branch``, unlike the refs above: the validator is a consumer's
            flow, so there is no base name here to suffix. And unlike every other setting,
            ``None`` (the default) is not forwarded — the parameter is omitted from the
            dispatch entirely, so each fill deployment's own registered value stands, which
            is where a consumer names its validator and where a branch-scoped registration
            already carries the suffix. Forwarding None would override that with nothing and
            switch validation off for every cell of the run.

    Returns:
        Summary: pending count at start, dispatched run ids per year, totals.
    """
    log = get_run_logger()
    # Stamped on every deployment this campaign dispatches so the terminal hooks can find
    # them from the flow-run id alone (see _child_tag). None outside a Prefect run — a direct
    # .fn() call in tests dispatches nothing worth sweeping.
    _tags = [t] if (t := _child_tag(flow_run_ctx.id)) else None
    # Branch-scoped deployment names, resolved once. A default (None) child ref derives from
    # `branch`; an explicit ref passes through verbatim. The S1/S2 grandchildren derive from
    # the same `branch` in `_ingest_params`.
    fill_deployment = fill_deployment or _dpl("fill-zone-year/fill-zone-year", branch)
    ingest_deployment = ingest_deployment or _dpl("ingest-zone-year/ingest-zone-year", branch)
    chained_fill_deployment = chained_fill_deployment or _dpl("fill-zones-sequential/fill-zones-sequential", branch)
    if max_parallel_clusters < 1:
        msg = f"max_parallel_clusters must be >= 1, got {max_parallel_clusters} (Semaphore(0) blocks forever)"
        raise ValueError(msg)
    if max_parallel_ingest < 1:
        raise ValueError(f"max_parallel_ingest must be >= 1, got {max_parallel_ingest} (Semaphore(0) blocks forever)")
    if max_dispatch_rounds < 1:
        raise ValueError(
            f"max_dispatch_rounds must be >= 1, got {max_dispatch_rounds} (no cell would ever be dispatched)"
        )
    # EVERY value this can reject on its own, BEFORE any shared control is touched.
    #
    # Checked here rather than left to run_inference: the child validates only after both fill
    # strategies have entered ray_cluster, and the chained strategy has primed its look-ahead
    # ingests first. A typo would otherwise buy a Ray head and a round of multi-hour ingests
    # before failing deterministically on a value known up front. The limit upserts sit after
    # this whole read-only preflight for the same reason — see there.
    if num_actors < 1:
        raise ValueError(f"num_actors must be >= 1, got {num_actors} (no actor would ever run inference)")
    # HERE, before any dispatch. `_apply_gpu_fallback` refuses the same values, but it runs
    # inside the child, AFTER that child has upserted shared limits and primed its look-ahead
    # mosaics — so a deterministic configuration error would spend real ingest work per
    # cluster before anything said so, on a campaign that cannot succeed as configured.
    if gpu_fallback_instance_types:
        # Imported HERE, not at module scope: `providers.aws.ray` pulls boto3 and ray, from
        # the `aws` and `inference` extras, so a module-level import would stop a
        # `tessera_embeddings[prefect]` install from importing or registering this flow and
        # defeat the deferred AWS imports that keep this module dependency-neutral.
        from tessera_embeddings.providers.aws.ray import GPU_FALLBACK_INSTANCE_TYPES

        unknown = sorted(set(gpu_fallback_instance_types) - GPU_FALLBACK_INSTANCE_TYPES)
        if unknown:
            raise ValueError(
                f"gpu_fallback_instance_types names unsupported type(s) {unknown}. Supported: "
                f"{sorted(GPU_FALLBACK_INSTANCE_TYPES)}. Sizes are restricted on HOST RAM - the "
                "vCPU-matched `xlarge` of each family would OOM the loader before inference."
            )
        if len(set(gpu_fallback_instance_types)) > 1:
            # Refused HERE as well as in the provider, and the duplication is the point: the
            # provider's guard fires inside `ray_cluster`, by which time a chained fill has
            # primed its look-ahead ingests. A configuration that cannot work should cost
            # nothing to discover.
            raise ValueError(
                f"Only one GPU fallback instance type may be opened at a time, got "
                f"{sorted(set(gpu_fallback_instance_types))} - the fleet mix is a floor, and "
                "ordinary actor demand above it is still scored by Ray, which breaks the tie "
                "between identically-shaped fallbacks on node-type name rather than throughput."
            )
        if gpu_fallback_vcpu_budget is not None:
            if gpu_fallback_vcpu_budget <= 0:
                raise ValueError(f"gpu_fallback_vcpu_budget must be > 0, got {gpu_fallback_vcpu_budget}")
            # 8 vCPU is the cheapest supported fallback instance, so a budget below it cannot
            # seat one node of the rung it is about to open.
            if gpu_fallback_vcpu_budget < 8:
                raise ValueError(
                    f"gpu_fallback_vcpu_budget={gpu_fallback_vcpu_budget} cannot afford one "
                    "fallback instance (8 vCPU). Raise it, or drop the fallback request."
                )
    if fill_strategy not in ("cluster-per-zone", "chained-clusters"):
        raise ValueError(f"fill_strategy must be 'cluster-per-zone' or 'chained-clusters', got {fill_strategy!r}")
    # Each of the three staging levers CLAIMS the identity, so any pair is a contradiction
    # rather than a combination, and resolving one silently lands the run on a prefix nobody
    # chose. Refused rather than ordered by precedence: guessing wrong costs a whole
    # campaign's re-inference in the cheap direction and mixed code versions in the expensive
    # one, neither worth inferring from a parameter the caller did not mean.
    _staging_levers = [
        name
        for name, given in (
            ("staging_code_identity", bool(staging_code_identity)),
            ("force_staging_reuse", force_staging_reuse),
            ("force_staging_restage", bool(force_staging_restage)),
        )
        if given
    ]
    if len(_staging_levers) > 1:
        raise ValueError(
            f"{' and '.join(_staging_levers)} each decide the staging identity, and they disagree. "
            f"Pass exactly one (staging_code_identity={staging_code_identity!r}, "
            f"force_staging_reuse={force_staging_reuse!r}, force_staging_restage={force_staging_restage!r})."
        )

    campaign_years = tuple(years) if years is not None else CAMPAIGN_YEARS

    # Lazy AWS import so the flow file imports on non-AWS machines (arch tests). The driver
    # reads the global store directly (status, tags, on-axis probe) BEFORE any child flow is
    # dispatched, so a deployment whose store authenticates only through the callback needs
    # it here too, not just in the children.
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials, icechunk_credentials_for

    store_path = paths.global_store(store_name)
    land_mask_path = paths.land_mask_store(mask_name)
    # THE STORE may be the partner-owned published bucket, which the task role cannot open.
    # Only the four store reads take this; the land-mask reads, the mosaic fingerprint and the
    # live-tile partition stay on the task role, the only credential that can read OUR buckets.
    store_credentials = icechunk_credentials_for(store_path, iam_icechunk_credentials)
    repo = open_global_repo(store_path, get_credentials=store_credentials, region=s3_region)
    # ONCE, before any dispatch. Each fill re-checks this, but by then the campaign has paid
    # for that cell's ingest — and cells dispatch concurrently, so a store seeded for a
    # different encoder buys a multi-terabyte mosaic per in-flight zone before the first fill
    # fails, and those mosaics are retained on failure (which is what makes a resume cheap).
    # A metadata-only read.
    _assert_seeded_model_matches(
        store_path,
        build_checkpoint=checkpoint_filename(),
        allow_model_mismatch=allow_model_mismatch,
        get_credentials=store_credentials,
        s3_region=s3_region,
    )
    status = campaign_status(repo, years=campaign_years)
    existing_tags = set(repo.list_tags())

    # Reject off-axis years up front: the time axis is fixed at seeding (ADR-008 D1), so
    # e.g. `years=(2026,)` against a 2017-2025 store would otherwise ingest and provision Ray
    # for every zone only for the fill to reject each one. Validated against a seeded zone.
    seeded_zones = sorted(status.zones)
    if seeded_zones:
        off_axis = [
            y
            for y in campaign_years
            if not zone_year_on_axis(
                store_path, seeded_zones[0], y, get_credentials=store_credentials, s3_region=s3_region
            )
        ]
        if off_axis:
            raise ValueError(
                f"Year(s) {off_axis} are not on the store's pre-allocated axis "
                f"(fixed at seeding, ADR-008 D1) — reseed the store or drop them from `years`."
            )

    # Canonical common names ("33N", "07S"), so a bad id fails loudly here and the rest match
    # the store's group names exactly.
    expected_zones = [canonicalize_zone(z) for z in zones] if zones is not None else None
    # An explicit zones subset must be seeded: a typo that canonicalizes to a valid but
    # unseeded zone would otherwise count as pending and be ingested + filled only for the
    # fill to reject an unseeded group. (The default None spans all 120; a not-fully-seeded
    # store surfaces at the fill.)
    if expected_zones is not None:
        unseeded = [z for z in expected_zones if z not in status.zones]
        if unseeded:
            raise ValueError(f"Requested zone(s) {unseeded} are not seeded in {store_path} — run the seed flow first.")

    # Cells still needing a fill, restricted to `zones` (default: all 120).
    # campaign_work_list is tag-aware: it skips landed-and-tagged cells and INCLUDES
    # complete-but-untagged ones (crash between fill and tag → the runner's idempotent retag
    # path runs). See its docstring for the two use cases.
    work = campaign_work_list(status, existing_tags, expected_zones=expected_zones, years=campaign_years)

    # Every work cell must be seeded before we dispatch. The guard above covers a requested
    # subset, but the default `zones=None` expands to all 120 in campaign_work_list, so a
    # partially-seeded store would ingest each unseeded cell (expensive) only for the fill to
    # reject the missing group.
    unseeded_work = sorted({z for z, _ in work if z not in status.zones})
    if unseeded_work:
        raise ValueError(
            f"Zone(s) {unseeded_work} in the work list are not seeded in {store_path} — run the seed flow "
            "first (a partially-seeded store would ingest each cell only for the fill to reject it)."
        )

    # SHARED CONTROLS ARE WRITTEN LAST, once every read-only check has passed and there is
    # work to do. One of these RESETS the fleet-wide inference pause gate to running, and zero
    # is how an operator pauses a campaign already in flight — so a run that flipped the gate
    # and then rejected would resume somebody else's paused fleet on its way out. Everything
    # that can reject sits above: the parameter checks, the store read, the model gate, the
    # year-axis probe, the seeded-zone guards and the work list. A no-work invocation is the
    # same hazard without an error, so `work` gates this too.
    #
    # Cheap, idempotent and safe to repeat — the ordering is the only thing that matters.
    if work:
        if ingest and fill_strategy == "chained-clusters":
            _upsert_limit(ingest_limit_name, max_parallel_ingest, what="ingest", log=log)
        if inference_pause_gate and fill_strategy == "chained-clusters":
            # Upserted to ONE, and one is not a cap: the gate is read as a flag rather than
            # acquired, so any positive value means "run" and nothing consumes a slot.
            # Created here so pausing is always available — a gate an operator has to create
            # first is not a lever they can reach at 3 a.m. — and reset to running at every
            # start that has work, so a campaign never inherits a pause left behind.
            _upsert_limit(inference_pause_gate, 1, what="inference pause (1 = running)", log=log)

    # Orphan-mosaic recovery: a per-cell cleanup that failed after tagging leaves the mosaic
    # behind, and that cell has left `work`, so it is never retried. Best-effort sweep of
    # complete-and-tagged cells in scope. Opt-in; the zero-cost backstop is an S3 lifecycle
    # rule.
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
        "Campaign: %d (zone, year) cell(s) need work across %d year(s); <=%d concurrent fills/year",
        len(work),
        len({y for _, y in work}),
        max_parallel_clusters,
    )

    optical_rule_cache: list[int | None] = []

    def _campaign_optical_min_obs() -> int | None:
        """The store's minimum-depth rule, for the staging fingerprint.

        Read from the store because the store is the authority (its root attr is part of a
        write-once identity), and cached because every cell writes the same store and the
        root cannot change under it. In the fingerprint for the same reason
        ``allow_s2_only`` is: it decides which pixels get embeddings, and staging is not
        namespaced by store.
        """
        if not optical_rule_cache:
            optical_rule_cache.append(
                _optical_min_obs_from_store(store_path, get_credentials=store_credentials, s3_region=s3_region)
            )
        return optical_rule_cache[0]

    code_identity_cache: list[str] = []
    announced_staging_identities: set[str] = set()

    def _staging_code_identity() -> str:
        """The staging fingerprint's code component — narrowed, with three levers.

        Default: the inference-source hash, so an orchestration or ingest hotfix reuses
        staged tiles instead of abandoning them. ``force_staging_reuse`` collapses it to a
        constant so even an INFERENCE change reuses the prefix — deliberately unsafe, for the
        one case the narrowing cannot judge (a log line, a comment, a type annotation).
        ``force_staging_restage`` mixes in a caller token so a change the hash CANNOT see
        starts a fresh prefix (a mid-campaign torch upgrade changes the numbers without
        changing our source). ``staging_code_identity`` STATES the answer, and is the only
        lever that can reach a prefix an earlier campaign staged under, precisely because it
        does not derive: no computation over CHANGED code reproduces the identity that
        changed code replaced.

        The three are mutually exclusive, refused at preflight rather than ordered here. See
        ``context_docs/storage/staging-identity-and-resume.md``.
        """
        identity = _derive_staging_code_identity()
        # ANNOUNCED, never cached, and the difference is load-bearing. The tarball's ETag is
        # the one term that can move WHILE a campaign runs: replace the object and later
        # dispatches download new code. Freezing the identity at first use would send those
        # dispatches back to the prefix the OLD code staged, mixing two code versions in one
        # write-once zone-year — the failure the fingerprint exists to prevent. Re-deriving
        # means a replaced tarball starts a FRESH prefix, which costs re-inference and can
        # never mix. (A pinned identity is a constant and never consults the tarball, so the
        # hazard belongs to the derived path alone.)
        #
        # SAID BY THE DRIVER whatever the strategy, because a restart needs the earlier
        # campaign's value and the child parameters are not a general source: a
        # `cluster-per-zone` campaign hashes the identity irreversibly into its children's
        # run_ids. Once per DISTINCT value, so a mid-campaign tarball swap announces itself
        # rather than hiding behind the first line.
        if identity not in announced_staging_identities:
            announced_staging_identities.add(identity)
            log.info(
                "Staging code identity for this campaign: %s (%s). Pass this back as "
                "`staging_code_identity` to restart onto the tiles this campaign stages; once the "
                "inference source or the code tarball moves it cannot be recomputed.",
                identity,
                "stated by staging_code_identity" if staging_code_identity else "derived",
            )
        return identity

    def _derive_staging_code_identity() -> str:
        """:func:`_staging_code_identity` without the memo or the announcement."""
        if staging_code_identity:
            return staging_code_identity
        if force_staging_reuse:
            return "infcode-forced-reuse"
        identity = inference_code_identity()
        # The source hash covers what the flow runner can SEE. Workers execute a tarball they
        # download instead, and replacing it changes what runs without changing anything here
        # — so a retry would resume tiles staged by the old code and add new ones from the
        # new. The tarball's ETag closes that, and it is EMPTY only when `code_bucket` is
        # None. NOT a dev-only term: the global campaign runs with a code_bucket, so
        # re-uploading the tarball abandons every staged tile whatever the change was, which
        # is a large part of why `staging_code_identity` exists.
        #
        # Deliberately the tarball term ALONE, not resolve_code_artifact_identity's whole
        # string: re-baking an image does not change what a staged tile contains, so folding
        # the AMI in would abandon every staged tile for nothing.
        tarball = _resolve_tarball_identity(code_bucket, code_suffix, None)
        if tarball:
            identity = f"{identity}|{tarball}"
        return f"{identity}+{force_staging_restage}" if force_staging_restage else identity

    def _code_identity() -> str:
        # Resolve the immutable code artifact (AMI ID + optional tarball ETag) once and reuse
        # it for every cell: both are campaign-wide constants, so one SSM/S3 lookup suffices.
        # Lazy, so a retag-only or all-ocean campaign (no fingerprint computed) makes no AWS
        # call.
        #
        # region=None → us-west-2, the region the fill's ray_cluster PROVISIONS from (the
        # campaign does not thread a Ray region through). The AMI SSM parameter and the
        # source tarball live there, NOT in the storage `s3_region`, which may differ;
        # resolving with s3_region would look up the AMI in the wrong SSM namespace and
        # either fail or fingerprint an artifact the workers never boot.
        if not code_identity_cache:
            # Pin the AMI component to the SAME id every fill provisions (below), so the
            # fingerprint and the booted image cannot disagree via two SSM reads at different
            # instants.
            code_identity_cache.append(_resolve_code_identity(ami_ssm_name, code_bucket, code_suffix, None, _ami_id()))
        return code_identity_cache[0]

    ami_id_cache: list[str] = []

    def _ami_id() -> str:
        # Resolve the worker AMI ID ONCE per campaign and PIN it into every fill's
        # provisioning (below), so a mid-campaign re-bake that repoints ami_ssm_name cannot
        # make a fill boot a different image than its staging fingerprint recorded (which
        # keys on this same resolved id via _code_identity). Same region rationale as
        # _code_identity: None → us-west-2, the Ray provisioning region.
        if not ami_id_cache:
            ami_id_cache.append(_resolve_ami_id(ami_ssm_name, None))
        return ami_id_cache[0]

    def _fill_params(zone: str, year: int, run_id: str, *, needs_cluster: bool) -> dict[str, Any]:
        return {
            "zone": zone,
            "year": year,
            "paths": paths.model_dump(),
            # run_id is computed by the caller (post-ingest): an input-fingerprinted id for a
            # real fill, so a retry resumes staging only on identical inputs, or a stable
            # "-retag" id for an already-complete cell, which does no inference/staging and
            # must not inspect a possibly-cleaned-up mosaic.
            "run_id": run_id,
            "ami_ssm_name": ami_ssm_name,
            # Pin the exact image the staging fingerprint recorded (see _ami_id), so the fill
            # cannot re-resolve ami_ssm_name to a re-baked AMI mid-campaign.
            #
            # ONLY for a cell that will actually start Ray. Retag-only and all-ocean cells
            # never provision, and _code_identity is lazy for the same reason: a pure
            # tag-repair or all-ocean campaign should need no SSM read, and resolving here
            # would turn a missing parameter or an SSM-less role into a failure on precisely
            # the cheap recovery path that has no use for the answer. None leaves the fill's
            # own fallback in place, which those cells never reach either.
            "ami_id": _ami_id() if needs_cluster else None,
            "store_name": store_name,
            "mask_name": mask_name,
            "num_actors": num_actors,
            "s1_orbit": s1_orbit,
            "s3_region": s3_region,
            "allow_partial_window": allow_partial_window,
            "allow_s2_only": allow_s2_only,
            # Explicit, and it must stay explicit: `fill-zone-year` DEMANDS radar by default,
            # because a lone cell dispatched by hand should report missing radar rather than
            # quietly produce optical-only embeddings. A global campaign is the opposite case
            # — parts of the globe have no dual-pol coverage at all, and refusing them fails
            # those cells on every retry forever.
            "require_s1": False,
            # The child re-runs the model gate this flow cleared in preflight, so without
            # forwarding the override it rejects the same store after its ingest has been
            # paid for. The campaign-level flag has to reach the gate that fires.
            "allow_model_mismatch": allow_model_mismatch,
            # Divide the fleet S3-PUT budget across concurrent fills so K shard-write phases
            # don't burst K times the target PUTs (the ~800-req SlowDown). D6 gates
            # committers; this bounds the ungated upload phase.
            "s3_concurrency": max(1, TARGET_AGGREGATE_S3_CONCURRENCY // max_parallel_clusters),
            # The same idea one layer down: a fleet-wide rate the concurrent fills share,
            # here the account's RunInstances quota. Unlike the S3 budget there is no share
            # to divide — each autoscaler runs the limiter client-side — so the campaign
            # passes only whether to run it at all.
            "launch_pacing": launch_pacing,
            "gpu_fallback_instance_types": gpu_fallback_instance_types,
            "gpu_fallback_vcpu_budget": gpu_fallback_vcpu_budget,
            # The other half of that problem: pacing makes a launch request cheaper, this
            # bounds how many a fill makes at all. Both OMITTED when unset so the fill's own
            # default decides — keyed on None rather than truthiness, since 0 is a documented
            # batch mode and not an absence.
            **set_overrides(
                actor_request_headroom=actor_request_headroom,
                actor_request_batch_size=actor_request_batch_size,
            ),
            "ssm_prefix": ssm_prefix,
            "cloudwatch_log_group": cloudwatch_log_group,
            "code_bucket": code_bucket,
            "code_suffix": code_suffix,
            # OMITTED when this campaign names no validator — the difference between "the
            # campaign decides" and "the campaign overrides". An omitted parameter takes the
            # CHILD DEPLOYMENT's own registered value, which is where a consumer names its
            # validator and where a branch-scoped registration already carries the suffix.
            # Passing None would override that with nothing and silently switch validation
            # off for every cell.
            **({"validation_deployment": validation_deployment} if validation_deployment else {}),
        }

    _ingest_params = partial(
        _ingest_dispatch_params,
        paths=paths,
        mask_name=mask_name,
        s1_orbit=s1_orbit,
        ingest_settings=ingest_settings,
        allow_partial_window=allow_partial_window,
        allow_ingest_code_mismatch=allow_ingest_code_mismatch,
        s3_region=s3_region,
        branch=branch,
    )

    fill_sem = asyncio.Semaphore(max_parallel_clusters)
    ingest_sem = asyncio.Semaphore(max_parallel_ingest)
    # NO cap on cells holding a live mosaic: ADR-011's in-flight bound was removed
    # deliberately, because it made ingest wait on fill throughput and so throttled the cheap
    # half of the campaign to protect a storage cost we would rather pay. Mosaics may
    # accumulate up to a whole year's worth (~120 cells) before the fills drain them. What is
    # gone is the BACKPRESSURE, not the cleanup: `cleanup_mosaics` still deletes each one as
    # its fill lands and `sweep_orphan_mosaics` still collects what a crash left behind.
    #
    # `cluster-per-zone` only. Under `chained-clusters` (the default) the per-cell chain below
    # is bypassed and the child bounds its mosaics not at all — peak storage is a cluster's
    # mosaics by design, and the two semaphores that used to bound them throttled ingest and
    # inference behind assembly (`context_docs/campaign/campaign-plan.md` §1).

    async def _process(zone: str, year: int) -> str:
        # Ingest (if enabled) → fill → drop the transient mosaic. ingest_sem/fill_sem cap the
        # expensive Dask/Ray clusters; nothing gates the chain as a whole.
        #
        # A complete-but-untagged cell (in years_complete, missing its tag) is in the work
        # list only so the fill re-creates the tag — no inference, no mosaic. Skip ingest and
        # cleanup for it entirely.
        retag_only = status.has(zone, year)
        did_ingest = ingest and not retag_only
        if did_ingest:
            async with ingest_sem:  # each ingest provisions its own Dask cluster
                irun = await arun_deployment(ingest_deployment, parameters=_ingest_params(zone, year), tags=_tags)
                _check_completed(irun, f"ingest {zone}-{year}")
        # An all-ocean cell (no live tiles) produces no mosaic: ingest returns skipped_ocean
        # and the fill marks it empty + tags with no staging/inference. There is nothing to
        # fingerprint — _staging_run_id would raise "No mosaic stores found" and strand the
        # whole year — and nothing to clean up, so it gets a stable mosaic-free id. Checked
        # for every non-retag cell at one cheap bitmap GET, ingest=False cells included; a
        # live cell falls through to the input-fingerprinted id.
        empty_cell = not retag_only and not zone_has_live_tiles(
            land_mask_path, zone, get_credentials=iam_icechunk_credentials, s3_region=s3_region
        )
        # AFTER ingest, so ingest=True fingerprints the freshly-written mosaic marker. A
        # retag-only cell does no inference/staging and its mosaic may already be cleaned up,
        # so it gets a stable id that does NOT inspect the mosaic — fingerprinting would
        # raise before the tag repair.
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
                optical_min_obs=_campaign_optical_min_obs(),
                code_identity=_staging_code_identity(),
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
        # Only delete a mosaic THIS campaign produced: with ingest=False it is an upstream
        # input we must not remove, and an empty cell produced none. Offloaded to a thread so
        # a multi-TB teardown does not stall the event loop and freeze every other zone's
        # deployment wait.
        if did_ingest and not empty_cell and cleanup_mosaics:
            await asyncio.to_thread(delete_prefix, f"{paths.inputs.rstrip('/')}/mosaics/{zone}/{year}", log=log)
        return str(frun.id)

    runs_by_year: dict[int, list[str]] = {}
    #: Cells still missing after every attempt, per year. Collected rather than
    #: raised on, so one bad zone cannot cost the years that follow it.
    unfilled: dict[int, list[str]] = {}
    #: Cells that have already had their one immediate replacement, for the life of the
    #: CAMPAIGN rather than of a round. The bound `immediate_refill` advertises is per cell —
    #: at most one attempt beyond the round budget — and rounds are re-entered, so a marker
    #: scoped to a round would grant that extra attempt again on every one.
    refilled_cells: set[tuple[str, int]] = set()
    # Descending, so the campaign delivers the most recent year soonest (ADR-008 D1);
    # override via `years`. Deduped because campaign_work_list already dedupes years, so a
    # duplicate here would re-dispatch the same static work entries against a non-refreshed
    # status. `overlap_years` chooses only the BATCHING — one batch per year keeps the
    # year-serial shape, one batch for everything drops the barrier — and everything inside
    # is identical, which is what keeps the two paths from drifting.
    ordered_years = sorted(set(campaign_years), reverse=True)
    batches: list[list[int]] = [ordered_years] if overlap_years else [[y] for y in ordered_years]
    for batch_years in batches:
        # BOUNDED RE-DISPATCH. A failed zone must not cost the campaign the rest of the
        # batch, nor the batches after it — interruptions are EXPECTED here, because the
        # orphan sweeper cancels child runs by design. Every round re-reads the STORE (not
        # our bookkeeping) for what is still missing, so a retry can only target genuinely
        # unfilled cells and an interrupted mosaic is resumed rather than rebuilt.
        remaining = [(z, y) for z, y in work if y in set(batch_years)]
        for attempt in range(1, max_dispatch_rounds + 1):
            if not remaining:
                break
            batch_cells = list(remaining)
            # Distinct zones, order-preserving. A zone with several years in this batch
            # appears ONCE, so the partition assigns all of its years to one cluster —
            # which is what keeps a zone's years off two clusters at once.
            batch_zones = list(dict.fromkeys(z for z, _y in batch_cells))
            round_failures: list[str] = []
            round_runs: list[str] = []
            runs_this_round_by_year: dict[int, list[str]] = {}

            if batch_zones and fill_strategy == "chained-clusters":
                # Up to max_parallel_clusters child runs fill the year, each owning ONE shared
                # Ray cluster and draining its own zone list strictly sequentially — taking
                # up the next zone without actor churn instead of being torn down per cell.
                # Shards are size-balanced (LPT) so they finish together. Ingest, per-cell
                # run_ids, retag/empty triage and mosaic cleanup all live inside the children,
                # whose ingest look-ahead pipelines mosaics against each cluster, so the
                # per-cell _process chain is bypassed; the global ingest bound is preserved by
                # dividing max_parallel_ingest across the clusters.
                clusters = _partition_by_live_tiles(
                    batch_zones,
                    min(max_parallel_clusters, len(batch_zones)),
                    land_mask_path=paths.land_mask_store(mask_name),
                    # A cluster gets every pending (zone, year) of the zones it owns (see
                    # `cells_for` below), so a zone's cost to it is per-year work times the
                    # years it carries. Already-landed years count zero — the child retags
                    # them without GPU work — and a zone with none left is weightless
                    # outright, its mask read skipped.
                    pending_years={
                        z: sum(1 for zz, y in batch_cells if zz == z and not status.has(z, y)) for z in batch_zones
                    },
                    get_credentials=iam_icechunk_credentials,
                    s3_region=s3_region,
                )
                # Each cluster's own cells: every (zone, year) pair whose zone it owns.
                cells_for = {id(cl): [[z, y] for z, y in batch_cells if z in set(cl)] for cl in clusters}
                # The same partition keyed by year, kept rather than re-derived from
                # `cells_for`, which is JSON-shaped for the deployment parameters (lists, not
                # tuples) and has lost the year's type by the time it is read.
                years_for = {id(cl): sorted({y for z, y in batch_cells if z in set(cl)}) for cl in clusters}
                log.info(
                    "Year(s) %s: dispatching %d chained fill(s) for %d zone(s), %d cell(s): %s",
                    batch_years,
                    len(clusters),
                    len(batch_zones),
                    len(batch_cells),
                    [len(cells_for[id(cl)]) for cl in clusters],
                )
                # Each cluster's EVEN SHARE of the fleet-wide ingest cap, so the clusters
                # divide it by construction rather than racing for slots on the global gate,
                # which sets the hard ceiling but not who reaches it first. Rounded up:
                # flooring would silently deliver less ingest width than was asked for.
                #
                # MINUS ONE, because `look_ahead` counts cells BEYOND the current one — the
                # fill runs `1 + look_ahead` ingests per cluster (its ingest driver is sized
                # `max_parallel=1 + look_ahead`). Passing the whole share aimed each cluster
                # at `1 + share`, so the fleet asked for 48 against a cap of 40 at the shipped
                # 8 clusters: the gate held the line, so it oversubscribed rather than
                # breached, but the clusters queued and the log below reported a width the
                # fleet never ran at. Floored at 0 so a one-cell-at-a-time cluster can still
                # start its current cell. Rounding up still overshoots by up to `clusters - 1`
                # on an uneven cap (45 over 8 aims at 48); the gate remains the ceiling, and
                # erring wide is the deliberate side.
                share = -(-max_parallel_ingest // len(clusters)) if clusters else 1
                look_ahead = max(0, share - 1)
                if ingest:
                    log.info(
                        "chained-clusters: %d cluster(s) x %d zone(s) at a time = %d simultaneous ingest(s), "
                        "hard-capped fleet-wide at max_parallel_ingest=%d by %r",
                        len(clusters),
                        1 + look_ahead,
                        (1 + look_ahead) * len(clusters),
                        max_parallel_ingest,
                        ingest_limit_name,
                    )

                # `status` is re-read between retry rounds, so it is bound as a default rather
                # than captured: a closure outliving its round would otherwise read whichever
                # snapshot the loop left behind.
                def _needs_ray(cells: list[list[Any]], snapshot: CampaignStatus = status) -> bool:
                    """True if any UTM zone in this cluster will actually provision Ray."""
                    return any(
                        not snapshot.has(z, y)
                        and zone_has_live_tiles(
                            land_mask_path, z, get_credentials=iam_icechunk_credentials, s3_region=s3_region
                        )
                        for z, y in cells
                    )

                # Every per-year value this closure reads is passed IN rather than captured:
                # it is defined inside the year loop, so a captured name would be whatever the
                # last iteration left behind if a call ever outlived its iteration.
                def _chained_params(
                    cells: list[list[Any]],
                    n_clusters: int,
                    cell_look_ahead: int,
                    snapshot: CampaignStatus = status,
                ) -> dict[str, Any]:
                    return {
                        # (zone, year) PAIRS. With `overlap_years` a cluster's pairs span
                        # years; without it they all share one. Either way a zone's years
                        # are all in ONE cluster, because the partition is over zones.
                        "cells": cells,
                        "paths": paths.model_dump(),
                        "ami_ssm_name": ami_ssm_name,
                        # Pin the fingerprinted image (see _ami_id / _fill_params), but ONLY
                        # for a cluster that will actually start Ray: one whose cells are all
                        # retag or all-ocean never provisions, and forcing an SSM read there
                        # breaks the cheap recovery paths when the parameter is missing or
                        # the role lacks access. The ocean probe is one small bitmap GET per
                        # zone — the price the per-cell path already pays — and runs only for
                        # cells that are not already complete. The snapshot is passed THROUGH
                        # rather than defaulted here, so an immediate refill probes against
                        # the store it just read rather than the one the round opened on.
                        "ami_id": _ami_id() if _needs_ray(cells, snapshot) else None,
                        "store_name": store_name,
                        "mask_name": mask_name,
                        "num_actors": num_actors,
                        "s1_orbit": s1_orbit,
                        "s3_region": s3_region,
                        "allow_partial_window": allow_partial_window,
                        "allow_ingest_code_mismatch": allow_ingest_code_mismatch,
                        "allow_s2_only": allow_s2_only,
                        # As in _fill_params: the chained child must not demand radar either.
                        "require_s1": False,
                        # As in _fill_params: the chained child gates on the seeded model too.
                        "allow_model_mismatch": allow_model_mismatch,
                        "ssm_prefix": ssm_prefix,
                        "cloudwatch_log_group": cloudwatch_log_group,
                        "code_bucket": code_bucket,
                        "code_suffix": code_suffix,
                        # The narrowed staging identity with the force_staging_* overrides
                        # applied — the same value the per-zone path folds into its run_id.
                        # Without it the child recomputes the AMI/tarball artifact identity,
                        # so the default strategy re-infers on every re-bake and ignores both
                        # escape hatches.
                        "staging_code_identity": _staging_code_identity(),
                        "ingest": ingest,
                        "ingest_deployment": ingest_deployment,
                        # Route this cluster's S1/S2 ingest grandchildren (see `branch`);
                        # the direct ingest/fill refs above are already branch-resolved.
                        "branch": branch,
                        # This cluster's OWN zone count, so its admission gates never limit
                        # fleet fill (see the dispatch comment above). The staging+assembly
                        # slice below keeps the fleet-wide S3-PUT rate at the aggregate
                        # target.
                        "look_ahead": cell_look_ahead,
                        "s3_concurrency": max(1, TARGET_AGGREGATE_S3_CONCURRENCY // (2 * n_clusters)),
                        # As in _fill_params: the account's RunInstances quota is the
                        # fleet-wide rate these clusters share, and each autoscaler enforces
                        # its own share client-side rather than being handed a divided count.
                        "launch_pacing": launch_pacing,
                        "gpu_fallback_instance_types": gpu_fallback_instance_types,
                        "gpu_fallback_vcpu_budget": gpu_fallback_vcpu_budget,
                        # And as in _fill_params, the bound on how many launch requests a
                        # cluster makes, beside the pacing that makes each one cheaper.
                        # Omitted when unset so the child's own default stands.
                        **set_overrides(
                            actor_request_headroom=actor_request_headroom,
                            actor_request_batch_size=actor_request_batch_size,
                        ),
                        "cleanup_mosaics": cleanup_mosaics,
                        "ingest_settings": ingest_settings.model_dump(),
                        # The shared cap. Every cluster names the same limit, which is
                        # what makes it fleet-wide rather than per-cluster.
                        "ingest_limit_name": ingest_limit_name,
                        # And the shared pause. Same reason it is one name for every
                        # cluster: an operator pauses the CAMPAIGN, not a cluster.
                        "inference_pause_gate": inference_pause_gate,
                        # As in _fill_params: omitted when unset, so the child's own registered
                        # validator stands. The child dispatches each cell's validation off
                        # its trailing assembly thread as that cell is tagged.
                        **({"validation_deployment": validation_deployment} if validation_deployment else {}),
                    }

                async def _settle(parameters: dict[str, Any]) -> Any:  # noqa: ANN401 — Prefect FlowRun
                    """Dispatch one chained fill and return its outcome, run OR exception.

                    Mirrors ``gather(return_exceptions=True)`` exactly, ``BaseException``
                    included: a dispatch that raises is a RESULT here, because one cluster
                    failing must not abandon the round, and the bookkeeping below tells the
                    two apart by type rather than by how it was delivered.
                    """
                    try:
                        return await arun_deployment(chained_fill_deployment, parameters=parameters, tags=_tags)
                    except BaseException as exc:  # mirrors gather(return_exceptions=True)
                        return exc

                # `batch_years` is passed IN rather than captured, for the reason
                # `_chained_params` gives beside it.
                async def _refill(
                    zones: list[str],
                    outcome: Any,  # noqa: ANN401 — Prefect FlowRun or the exception that replaced it
                    years: tuple[int, ...] = tuple(batch_years),
                    n_clusters: int = len(clusters),
                    cell_look_ahead: int = look_ahead,
                ) -> tuple[dict[str, Any], list[str], list[int]] | None:
                    """A ready-to-dispatch replacement for a settled fill, or ``None`` to decline.

                    Returns the finished PARAMETERS rather than a plan, deliberately:
                    everything that can fail — the store re-read and the land-mask and AMI
                    probes inside ``_chained_params`` — then sits inside this function's
                    guard, and the caller is left with a step that cannot raise. See the
                    dispatch loop for why.

                    DECLINING IS ALWAYS SAFE, and it is the answer to everything this function
                    cannot establish, its own inability to ask included: the cells stay in
                    `remaining` and the round re-dispatches them as it does today. That
                    asymmetry is the shape of the function.

                    Condition one is the predecessor's terminal state, which shows the fill's
                    OWN writers stopped (see `_QUIESCENT_TERMINAL_STATES`). Condition two is
                    `_SETTLE_DELAY_S`, covering what condition one does not: the fill asked
                    its children to stop and waited, but that wait is best effort and gives
                    up, and the grandchildren are only asked by a hook that does not block.
                    Time is what an asynchronous cancellation needs; nothing here can be
                    observed into stopping, and nothing here reserves a cell.
                    """
                    if isinstance(outcome, BaseException):
                        # Its own reason, not folded into the state check below: a dispatch
                        # that RAISED and a fill that ended in an untrustworthy state are
                        # different facts, and a declined refill has to say which.
                        log.info(
                            "Not refilling %d zone(s) immediately: the dispatch itself raised, so it may "
                            "still have dispatched and establishes nothing about what is running. They "
                            "wait for the round.",
                            len(zones),
                        )
                        return None
                    state = getattr(outcome, "state", None)
                    if state is None or state.type not in _QUIESCENT_TERMINAL_STATES:
                        log.info(
                            "Not refilling %d zone(s) immediately: the fill ended %s, which does not "
                            "establish that it stopped writing. They wait for the round.",
                            len(zones),
                            state.name if state is not None else "UNKNOWN",
                        )
                        return None
                    # ONE guard over every read this decision makes. Each reaches a network —
                    # the store's branch tip and tag list, and the land-mask and SSM probes
                    # inside `_chained_params` — and each can fail transiently. Uncertainty is
                    # not permission: a failure anywhere in here means nothing was
                    # established, so it declines. Left unguarded it would escape into the
                    # drain loop and fail the whole campaign while sibling fills were still
                    # running, and an ordinary FAILED state does not fire the
                    # child-cancellation hook.
                    try:

                        def _missing(snapshot: CampaignStatus) -> list[tuple[str, int]]:
                            """This roster's cells the STORE still lacks and that may be refilled.

                            The store decides, exactly as the round does — never the child's
                            return value, which can report success on cells it never
                            attempted. Intersected with the inherited roster as well as scoped
                            by it: the scope is the work list's contract, the intersection is
                            structural, and it is what makes "a replacement can never reach a
                            zone a live sibling holds" a property of this function rather than
                            a delegated one.
                            """
                            return [
                                (z, y)
                                for z, y in campaign_work_list(
                                    snapshot, set(repo.list_tags()), expected_zones=zones, years=years
                                )
                                if z in roster and (z, y) not in refilled_cells
                            ]

                        roster = set(zones)
                        # Read FIRST and cheaply, so the settling wait below is only paid for a
                        # roster that has something to refill. Most clusters settle having
                        # landed everything, and those must not hold the drain loop for a wait
                        # protecting a dispatch that is not going to happen.
                        if not _missing(campaign_status(repo, years=campaign_years)):
                            return None
                        # THE SETTLING WAIT. The fill is terminal, so its own writers are
                        # provably stopped and its direct children were cancelled AND waited
                        # for. What has not been waited for is the level below them, whose
                        # cancellation a non-blocking hook merely requested; this is the time
                        # that request needs to take effect.
                        log.info(
                            "Waiting %.0fs before refilling %d zone(s), so the settled fill's remaining "
                            "descendants have time to stop.",
                            _SETTLE_DELAY_S,
                            len(zones),
                        )
                        await asyncio.sleep(_SETTLE_DELAY_S)
                        # RE-READ after the wait, so the dispatch rests on the settled picture:
                        # a cell that landed meanwhile is no longer missing and costs no fleet.
                        fresh = campaign_status(repo, years=campaign_years)
                        missing = _missing(fresh)
                        if not missing:
                            return None
                        # The cluster count is the ROUND'S: the replacement takes the vacated
                        # slot, so its ingest share and S3 concurrency slice are the ones the
                        # round sized. Inside the guard because its land-mask and SSM probes
                        # must decline a refill, not end a campaign.
                        cells = [[z, y] for z, y in missing]
                        return (
                            _chained_params(cells, n_clusters, cell_look_ahead, fresh),
                            sorted({str(z) for z, _y in missing}),
                            sorted({y for _z, y in missing}),
                        )
                    except Exception:
                        log.warning(
                            "Could not establish what is safe to refill — declining the immediate refill of "
                            "%d zone(s); they wait for the round.",
                            len(zones),
                            exc_info=True,
                        )
                        return None

                # DISPATCH, then drain as each cluster SETTLES rather than all at once. With
                # `immediate_refill` off nothing is ever added to `dispatches`, so this waits for
                # every cluster just as a gather would and the bookkeeping below reads the
                # outcomes in the clusters' ORIGINAL order — dispatch sequence, log lines and
                # returned summary identical, not merely equivalent.
                #
                # THE RULE THIS BLOCK EXISTS TO KEEP: nothing that can await or raise may sit
                # between deciding to dispatch and dispatching. Two consequences a `gather`
                # gave for free and a hand-rolled loop has to earn. Every parameter set is
                # built BEFORE any task exists, because a starred `gather` consumed its whole
                # generator before scheduling anything, so a `_chained_params` that raised (its
                # land-mask probe, its SSM read) dispatched nothing at all; interleaving build
                # and schedule would leave earlier fills running while the campaign fails, and
                # an ordinary FAILED state does not fire the child-cancel hook, so those runs
                # would be orphaned with their fleets billing. And every task is created INSIDE
                # the try, so no task can exist outside the cancellation path.
                params_by_slot = [_chained_params(cells_for[id(cl)], len(clusters), look_ahead) for cl in clusters]
                #: The round's fills, and any replacement dispatched into a vacated slot. This
                #: is what "the round is still open" means and the only thing the liveness
                #: gate consults — a planner that has decided nothing yet is not a reason to
                #: spend another fleet.
                dispatches: dict[asyncio.Task[Any], int | None] = {}
                #: Refill planners in flight. Each waits out the settling delay for ONE
                #: settled slot and then dispatches into `dispatches` or declines. They run
                #: CONCURRENTLY, which is why they are tasks: awaiting a planner inline would
                #: stop the drain loop harvesting anything else, so several clusters failing
                #: together would serialise their waits end to end, and a sibling finishing
                #: mid-wait could not close the round until the wait expired.
                planners: set[asyncio.Task[Any]] = set()
                chained_results: list[Any] = [None] * len(clusters)
                #: Replacements, in dispatch order: the zones and years each covered, and the
                #: task carrying its outcome. Kept apart from `chained_results` so the
                #: originals' bookkeeping is untouched by whether any replacement happened.
                pending_refills: list[tuple[list[str], list[int], asyncio.Task[Any]]] = []

                # `dispatches` and `pending_refills` are passed IN rather than captured: both
                # are created per round, so a captured name would be whatever the last round
                # left behind if a planner ever outlived its round.
                async def _plan_and_dispatch(
                    zones: list[str],
                    outcome: Any,  # noqa: ANN401 — Prefect FlowRun or the exception that replaced it
                    open_dispatches: dict[asyncio.Task[Any], int | None] = dispatches,
                    refill_log: list[tuple[list[str], list[int], asyncio.Task[Any]]] = pending_refills,
                ) -> None:
                    """Wait out the settling delay for one settled slot, then replace it.

                    Runs as its own task so the delay is per slot rather than in series, and
                    so the drain loop keeps harvesting while it waits.

                    Every failure declines. A planner that raised would take the campaign with
                    it while sibling fills were still running, and an ordinary FAILED state does
                    not fire the child-cancellation hook that would sweep them.
                    """
                    try:
                        planned = await _refill(zones, outcome)
                        if planned is None:
                            return
                        replacement, take_zones, take_years = planned
                        # THE COMMIT GATE, and nothing between it and the dispatch may await or
                        # raise. Planning waits out the settling delay and re-reads the store,
                        # and the round can finish across that, so the check made before
                        # planning is stale by now. Re-read here, where the answer cannot go
                        # stale — otherwise the round is already over and the replacement buys
                        # an extra fleet and an extra attempt for no wall clock at all.
                        if not _still_running(open_dispatches):
                            log.info(
                                "Not refilling %d zone(s) immediately: the round finished while the refill "
                                "was being planned, so it is the round's to re-dispatch.",
                                len(take_zones),
                            )
                            return
                        log.warning(
                            "Refilling %d cell(s) across %d zone(s) at once rather than waiting for the round: %s",
                            len(replacement["cells"]),
                            len(take_zones),
                            ", ".join(take_zones),
                        )
                        # The half of the bound that spans ROUNDS. `dispatches` is rebuilt each
                        # round, so a marker kept there would hand a cell fresh eligibility
                        # every round and the documented "at most one extra attempt" would
                        # become one per round. This set outlives the round.
                        refilled_cells.update((z, y) for z, y in replacement["cells"])
                        refill_task = asyncio.ensure_future(_settle(replacement))
                        open_dispatches[refill_task] = None
                        refill_log.append((take_zones, take_years, refill_task))
                    except Exception:
                        log.warning(
                            "Immediate refill of %d zone(s) failed while being planned — they wait for the round.",
                            len(zones),
                            exc_info=True,
                        )

                try:
                    for slot, params in enumerate(params_by_slot):
                        dispatches[asyncio.ensure_future(_settle(params))] = slot
                    while dispatches or planners:
                        done, _ = await asyncio.wait(set(dispatches) | planners, return_when=asyncio.FIRST_COMPLETED)
                        # Deregister everything that settled BEFORE deciding anything, so "a
                        # sibling is still running" is read off work that really is running.
                        planners -= done
                        settled = [(dispatches.pop(task), task) for task in done if task in dispatches]
                        for settled_slot, task in settled:
                            outcome = task.result()
                            if settled_slot is None:
                                # A replacement carries NO slot, and that is half the bound: a
                                # replacement is not itself eligible for one. Its outcome was
                                # recorded at dispatch; there is nothing to decide here.
                                continue
                            chained_results[settled_slot] = outcome
                            # A cheap pre-filter, not the safety gate: it keeps a round that is
                            # already over from paying for a store read and a settling wait for a
                            # dispatch that the commit gate would refuse anyway.
                            if not immediate_refill or not _still_running(dispatches):
                                continue
                            planners.add(
                                asyncio.ensure_future(_plan_and_dispatch(list(clusters[settled_slot]), outcome))
                            )
                        # A planner has nothing left to outrun once no dispatch remains: its
                        # own gate declines, and the round's re-dispatch covers the same cells
                        # on the same terms. Waiting out its settling sleep would hold the
                        # store re-read and the whole next round for a replacement that cannot
                        # happen — the barrier this feature exists to remove, reintroduced at
                        # the end of it.
                        #
                        # Safe to cancel rather than await: a planner that HAS dispatched put
                        # its task into `dispatches`, so an empty `dispatches` proves none of
                        # them did. Awaited after cancelling for the reason the teardown below
                        # gives — a request to cancel is not a cancellation.
                        if not dispatches and planners:
                            # Said out loud, in the same words the planner would have used had
                            # it woken to decide this itself: an operator seeing a settled
                            # roster get no replacement needs to know it was refused for a
                            # reason and not missed.
                            log.info(
                                "Not refilling %d settled roster(s) immediately: the round finished while "
                                "the refill was being planned, so a replacement would buy an extra fleet "
                                "and no wall clock — the round re-dispatches these cells anyway. The "
                                "settling wait is dropped rather than served out.",
                                len(planners),
                            )
                            for planner in planners:
                                planner.cancel()
                            await asyncio.gather(*planners, return_exceptions=True)
                            planners.clear()
                finally:
                    # What `gather` did for its own children when the outer await was
                    # cancelled, both halves of it. Cancelling the WAIT does not cancel the
                    # child run; the flow's terminal hook sweeps those by tag.
                    #
                    # The await is not a formality. `cancel()` only REQUESTS cancellation, so
                    # returning here would let the flow reach its terminal hook while a
                    # dispatch was still unwinding — and a dispatch mid-unwind may still be
                    # registering a child run server-side, which would then appear AFTER the
                    # hook's one-shot sweep and never be collected. `return_exceptions` keeps
                    # a cancelled task's own `CancelledError` from displacing the exception
                    # that brought us here.
                    outstanding = set(dispatches) | planners
                    for task in outstanding:
                        task.cancel()
                    if outstanding:
                        await asyncio.gather(*outstanding, return_exceptions=True)
                # Replacement outcomes are futures until here, because the planners record
                # them at dispatch and cannot know the result yet. Every one is done —
                # `dispatches` emptying is what ended the loop.
                refills = [(zs, ys, task.result()) for zs, ys, task in pending_refills]
                # RECORDED, never raised: one cluster failing must not abandon the year, nor
                # the years after it. What each cluster landed is already committed and tagged
                # per zone-year, so the round's survivors keep their work and the retry below
                # targets only what is genuinely still missing.
                for cl, r in zip(clusters, chained_results, strict=True):
                    if isinstance(r, BaseException):
                        round_failures.append(f"chained fill {cl[0]}..({len(cl)} zones): {r!r}")
                        continue
                    try:
                        _check_completed(r, f"chained fill year(s) {batch_years}")
                    except Exception as exc:  # a returned-but-not-COMPLETED terminal state
                        round_failures.append(str(exc))
                        continue
                    round_runs.append(str(r.id))
                    # Credited to the years this cluster ACTUALLY owns, not to the batch. A
                    # fresh campaign gives every cluster every year, but a RESUMED one
                    # partitions over what is still missing, so a cluster can hold zones each
                    # short a different single year — crediting the batch would claim years
                    # the run never touched and inflate `dispatched`, which sums these lists.
                    for y in years_for[id(cl)]:
                        runs_this_round_by_year.setdefault(y, []).append(str(r.id))
                log.info(
                    "Year(s) %s: %d/%d chained fill(s) landed",
                    batch_years,
                    len(clusters) - len(round_failures),
                    len(clusters),
                )
                # The replacements, after their predecessors and in dispatch order, so the
                # count above keeps meaning the round's own clusters. Same three records as
                # the loop above: a failure is reported, a landed run is credited, and it is
                # credited to the years IT owns.
                refills_landed = 0
                for refill_zones, refill_years, r in refills:
                    if isinstance(r, BaseException):
                        round_failures.append(f"immediate refill {refill_zones[0]}..({len(refill_zones)} zones): {r!r}")
                        continue
                    try:
                        _check_completed(r, f"immediate refill year(s) {batch_years}")
                    except Exception as exc:  # a returned-but-not-COMPLETED terminal state
                        round_failures.append(str(exc))
                        continue
                    refills_landed += 1
                    round_runs.append(str(r.id))
                    for y in refill_years:
                        runs_this_round_by_year.setdefault(y, []).append(str(r.id))
                if refills:
                    log.info(
                        "Year(s) %s: %d/%d immediate refill(s) landed",
                        batch_years,
                        refills_landed,
                        len(refills),
                    )
            elif batch_cells:
                log.info("Year(s) %s: dispatching %d zone fill(s)", batch_years, len(batch_cells))

                # Different zones are different store groups, so they fill concurrently. A
                # zone's own years go one at a time — but NOT because the store requires it:
                # `write_year_shards` (d1a379c) makes every chunk and shard 1 in the time
                # dimension, so a zone's years write strictly disjoint objects and rebase
                # cleanly, and the only collision left is two group attrs, which commit
                # separately and retry with each writer inserting only its own year's key.
                # This serialisation is scheduling, not correctness, and is a candidate for
                # removal; the open question is the attr retry budget
                # (`commit_year_attrs(tries=8)`) once many years of one zone contend.
                # return_exceptions=True so one zone's failure does not abandon its siblings
                # mid-flight (a default gather() would leave them running orphaned): let the
                # whole year settle, then fail loudly with every failure.
                years_of: dict[str, list[int]] = {}
                for z, y in batch_cells:
                    years_of.setdefault(z, []).append(y)

                async def _zone_years(zone: str, years: list[int]) -> list[Any]:
                    """One zone's years, in order, each attempted whatever the last one did.

                    Sequential, so a zone never has two years in flight; one result per year,
                    so the round still accounts for every cell it was given.

                    IT DOES NOT STOP AT THE FIRST FAILURE, and that is the point. Years are
                    ordered oldest-first and every retry rebuilds the list the same way, so a
                    repeating failure that belongs to ONE year — a coverage gate is evaluated
                    per zone-year, and radar coverage genuinely varies year to year — would
                    otherwise keep its zone's other years from ever being attempted on any
                    round, and the no-progress guard below would eventually halt the campaign
                    over a cell that was never going to succeed while the years behind it were
                    fillable.

                    Stopping would buy fail-fast, and that is worth little here: the
                    deterministic zone-wide failures (unseeded group, destination dtype, year
                    off the axis, no live tiles) surface in `fill_zone_year_flow`'s preflight
                    BEFORE a GPU fleet is provisioned, so continuing past one costs a few
                    metadata reads. Same rule the chained path above states for clusters.
                    """
                    out: list[Any] = []
                    for y in years:
                        try:
                            out.append(await _process(zone, y))
                        except Exception as exc:  # mirrors gather(return_exceptions=True)
                            out.append(exc)
                    return out

                per_zone = await asyncio.gather(*(_zone_years(z, ys) for z, ys in years_of.items()))
                by_cell = {
                    (z, y): r
                    for (z, ys), rs in zip(years_of.items(), per_zone, strict=True)
                    for y, r in zip(ys, rs, strict=True)
                }
                cell_results = [by_cell[(z, y)] for z, y in batch_cells]
                for (z, y), cr in zip(batch_cells, cell_results, strict=True):
                    if isinstance(cr, BaseException):
                        round_failures.append(f"{z}-{y}: {cr!r}")
                    else:
                        round_runs.append(str(cr))
                        # Per-cell runs fill ONE year each, so each is attributed to its own.
                        # Copying the whole round into every year would report each year as
                        # holding the others' runs and overcount `dispatched` by the number of
                        # years in the batch — invisible until `overlap_years` puts several
                        # years in one batch.
                        runs_this_round_by_year.setdefault(y, []).append(str(cr))
                log.info("Year(s) %s: %d/%d fill(s) landed", batch_years, len(round_runs), len(batch_cells))

            # Attribution is per strategy — a chained cluster spans the batch's years while a
            # per-cell run fills exactly one — and both populate `runs_this_round_by_year`
            # above, so this only merges it.
            for y, runs in runs_this_round_by_year.items():
                runs_by_year.setdefault(y, []).extend(runs)
            # Re-read the STORE every round, including rounds that reported no failures. What
            # landed is a property of the store, never of a child's return value: a child can
            # report success having silently not attempted cells at all — `sequential_fill`'s
            # feeder stops admitting work when too many failures are retaining mosaics, and if
            # those failures then recover in its retry pass it returns clean with cells still
            # pending. Concluding from an empty failure list marks those as landed and ends
            # the campaign.
            before = set(remaining)
            status = campaign_status(repo, years=campaign_years)
            remaining = list(
                campaign_work_list(
                    status, set(repo.list_tags()), expected_zones=expected_zones, years=tuple(batch_years)
                )
            )
            if not round_failures and remaining:
                log.warning(
                    "Year(s) %s attempt %d/%d: the fill(s) reported no failures, but %d cell(s) are still "
                    "missing from the store — they were never attempted. Re-dispatching: %s",
                    batch_years,
                    attempt,
                    max_dispatch_rounds,
                    len(remaining),
                    ", ".join(f"{z}-{y}" for z, y in remaining[:10]),
                )
            for f in round_failures:
                log.warning("Year(s) %s attempt %d/%d: %s", batch_years, attempt, max_dispatch_rounds, f)
            if not remaining:
                break
            if attempt >= max_dispatch_rounds:
                break
            if set(remaining) == before and attempt > 1:
                # A whole round achieved nothing: a DETERMINISTIC failure as seen from out
                # here — a coverage gate, a fingerprint mismatch, an unseeded group — which
                # wants a human, not another cluster. Stop spending fleets on it.
                log.error(
                    "Year(s) %s: attempt %d made no progress on %d cell(s) — treating as deterministic "
                    "and stopping retries for this batch. Investigate before re-running: %s",
                    batch_years,
                    attempt,
                    len(remaining),
                    ", ".join(f"{z}-{y}" for z, y in sorted(remaining)[:10]),
                )
                break
            log.warning(
                "Year(s) %s: %d cell(s) still unfilled after attempt %d — re-dispatching (%d attempt(s) left)",
                batch_years,
                len(remaining),
                attempt,
                max_dispatch_rounds - attempt,
            )
        for z, y in remaining:
            if z not in unfilled.setdefault(y, []):
                unfilled[y].append(z)

        # Backfill the year-complete milestone/retention tag even when NO fills ran this pass
        # — a prior run may have completed every zone for the year but crashed before
        # tagging. The milestone is defined over ALL 120 zones and `tag_year_complete` takes
        # no scope argument, so this run's `zones` subset cannot narrow it and a repair run
        # cannot stamp the global tag prematurely, which matters because tags are write-once
        # forever. Attempted for EVERY year in the batch: an unfinished year raises
        # ValueError and is free, which is what lets the overlapping path use the same code
        # as the year-serial one instead of detecting "was that the year's last cell".
        for y in batch_years:
            try:
                year_tag = tag_year_complete(repo, y)
                log.info("Year %d tagged complete (all 120 zones): %s", y, year_tag)
            except ValueError as exc:
                log.debug("Year %d not yet complete across all 120 zones (%s) — milestone tag deferred", y, exc)

    dispatched = sum(len(v) for v in runs_by_year.values())
    log.info("Campaign dispatch complete: %d fill run(s) across %d year(s)", dispatched, len(runs_by_year))
    if unfilled:
        # LOUDLY, and last — but the run SUCCEEDS. A handful of cells failing every attempt
        # is an expected outcome at this scale, not a failed campaign: every other cell is
        # committed and tagged, and the product of those failures is a list of zones for a
        # second wave. Raising would mark 1,000 landed cells a failure because 3 did not land,
        # fire the driver-stopped alert (which means "nothing is being filled any more", true
        # here only because the campaign FINISHED), and leave an operator reading a stack
        # trace to find a list.
        #
        # The list is not quiet either: WARNING under a banner in the log, and `unfilled` in
        # the returned summary, so a caller's report carries it as data rather than prose to
        # re-parse. Nothing about a cell on it is ambiguous — it was dispatched, attempted
        # every round, and is missing from the store.
        detail = "; ".join(f"{y}: {', '.join(z)}" for y, z in sorted(unfilled.items(), reverse=True))
        total = sum(len(z) for z in unfilled.values())
        log.warning("=== CAMPAIGN FINISHED WITH %d UNFILLED CELL(S) — SECOND-WAVE LIST FOLLOWS ===", total)
        log.warning(
            "%d cell(s) did not land after up to %d dispatch round(s) each. Every OTHER cell is "
            "committed and tagged, so a re-run resumes from here and re-attempts only these. "
            "Unfilled: %s",
            total,
            max_dispatch_rounds,
            detail,
        )
        log.warning("=== END OF UNFILLED LIST (%d cell(s)) ===", total)
    return {
        "work_at_start": len(work),
        "dispatched": dispatched,
        "runs_by_year": runs_by_year,
        # Always present, empty when everything landed: a caller testing for the KEY rather
        # than for its truthiness must not have to know which shape it gets.
        "unfilled": {year: sorted(zones) for year, zones in sorted(unfilled.items())},
    }
