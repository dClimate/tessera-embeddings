"""Drive the global embeddings campaign: fill every pending (zone, year).

Reads the live fill progress from the seeded store
(:func:`~tessera_embeddings.storage.campaign.campaign_status`) and dispatches
fills for every pending cell via ``arun_deployment`` — size-balanced
``fill-zones-sequential`` runs, each owning one long-lived Ray cluster
(``fill_strategy="chained-clusters"``, the default), or a ``fill-zone-year`` run
per cell (``"cluster-per-zone"``) — mirroring :mod:`tessera_full_pipeline`'s
driver pattern.

**Ingest runs WIDE and ahead of inference, under ONE fleet-wide cap.** Ingest is
the cheap half and scales better across many narrow fleets than a few wide ones,
so the only thing limiting it is ``max_parallel_ingest`` — the number of UTM zones
that may ingest simultaneously across the entire campaign, however many clusters
are running. Nothing throttles it against fill throughput. A year's mosaics can
therefore accumulate ahead of the fills that consume them — hundreds of terabytes,
transient, deleted per cell as each fill lands (``cleanup_mosaics``) and swept
after a crash (``sweep_orphan_mosaics``). What is deliberately absent is
backpressure; what remains is cleanup.

GPUs are the resource that must not wait, and DENSITY ORDERING is what keeps them
fed. Zones are dealt to clusters densest-first, so every cluster opens on one of
the year's biggest zones and tapers towards sparse ones. A cluster starts its
ingest window, waits for that opening zone alone, and only then requests a fleet —
never booting GPUs speculatively, and never paying for a whole cluster's ingest up
front. Because the opening zone is dense it takes long enough to infer that the
rest of the window lands behind it, and inference is slower than ingest in almost
every case, so the stream does not run dry.

The clusters are separate flow runs, so their shared ingest cap lives on the
Prefect server: a global concurrency limit (``ingest_limit_name``), upserted from
``max_parallel_ingest`` at campaign start so the two can never drift. Each cluster
also takes an even share of that cap as its own window, so the clusters divide it
by construction rather than racing for slots.

**Scheduling (ADR-008 D6 + the runner contract).** Inference is parallel across
zones; only commits contend, and only *same-zone* fills conflict (shared
``years_complete``/``runs`` attrs → ``RebaseFailedError``). By default
(``overlap_years``) the driver dispatches **every requested year as one batch** and
partitions by ZONE, so a zone's every year lands in one cluster and its assemblies
serialize on that cluster's single trailing thread — which is what keeps same-zone
fills from colliding without a year barrier. ``overlap_years=False`` restores the
older shape: **year by year** in an outer serial loop, dispatching each year's zones
concurrently, where distinct zones make same-zone overlap impossible by construction.
Either way concurrency is bounded by ``max_parallel_clusters``.

The fleet-wide committer bound is a separate knob: ``commit_limit_name`` (a Prefect
global concurrency limit) is passed to every fill so commits stay under the storm
threshold while inference runs free. ``pending()`` is year-major, which is the drain
pattern the barrier path relies on.
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
    child_run_tag,
    make_child_cancel_hook,
)
from tessera_embeddings.orchestration.prefect.flows.fill_zone_year import (
    _assert_seeded_model_matches,
    _optical_min_obs_from_store,
)
from tessera_embeddings.orchestration.prefect.flows.ingest_zone_year import IngestDeployments
from tessera_embeddings.orchestration.prefect.flows.tessera_full_pipeline import _check_completed
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

# S1/S2 grandchild ingest refs are derived from this single source (rather than
# duplicated string literals), so a future rename of the ingest deployments stays
# in one place — ingest_zone_year.IngestDeployments.
_INGEST_DEPLOYMENT_DEFAULTS = IngestDeployments()


#: Tag prefix for every deployment this campaign dispatches — ingests, per-cell
#: fills and chained-fill clusters alike.
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


#: Ceiling on simultaneous zone-year committers, from ADR-008 D5/D6 run 1. Commit
#: contention is on the branch-tip CAS — every commit re-serialises the repo-global
#: snapshot — so rebase retries scale with the number of racing writers and the
#: aggregate wasted work with its square. Measured: N=2 -> 0.5 retries / 0.5 s,
#: N=8 -> 3.5 / 1.3 s, N=16 -> 7.5 / 2.2 s (which BREACHED the run's own
#: <=2x-serial acceptance criterion), N=120 -> 58 / 15 s. The recorded firm
#: constraint is 4-8; this is its upper end.
MAX_SIMULTANEOUS_COMMITTERS = 8


#: Terminal states after which a chained fill has PROVABLY stopped writing, and is
#: therefore the only ones an immediate replacement may follow.
#:
#: The proof is a chain of ``finally`` blocks, not an assumption about the state name.
#: Reaching either of these means the fill's own function returned or raised, so the
#: runner joined its trailing assembly thread — the only thread that commits to the
#: embeddings store — and the flow then cancelled its child ingests and WAITED for them
#: to confirm terminal before tearing its fleet down. All of that completes before the
#: state is set.
#:
#: ``CRASHED`` and ``CANCELLED`` are excluded because neither carries that proof: the
#: process that would run the ``finally`` is the process that died, and a crash verdict
#: can be reached from missed heartbeats while the run is still writing. Those cells
#: wait for the dispatch round to close, which is what they do today. Re-dispatching them
#: promptly is a separate change and needs fencing at the WRITE — a lease the committer
#: checks — rather than an inference about who has stopped.
_QUIESCENT_TERMINAL_STATES = frozenset({StateType.FAILED, StateType.COMPLETED})

#: How long a settled fill's cells are left alone before a replacement may be dispatched.
#:
#: DERIVED, not chosen. Cancellation is asynchronous, and the run tree is three levels
#: deep: the fill, its zone ingests, and their S1/S2 grandchildren. A fill's own teardown
#: already spends :data:`CANCELLATION_CONFIRM_S` waiting for level two to confirm terminal
#: BEFORE the state this driver reads is set — so reaching a quiescent state already buys
#: one budget. What nothing waited for is level three, whose cancellation is requested by a
#: hook that does not block. This is that same budget again, once, for that level.
#:
#: Counted from the cancellation request rather than from the terminal state, the two
#: together come to twice this — which is the interval the crash-recovery record already
#: recommends between a run's death and re-dispatching its cells. Arriving at the recorded
#: number from the mechanism rather than adopting it is the point: if the confirmation
#: budget changes, this follows.
#:
#: Time, not inspection, because that is what an asynchronous cancellation actually needs.
#: A census of who is writing can only report what was true a moment ago and cannot make a
#: lingering child stop; waiting gives it the thing it needs. Two mechanisms already act
#: rather than observe — the teardown that waits for children, and the orphan sweep that
#: collects whatever the teardown could not confirm — and this delay is what lets them.
_SETTLE_DELAY_S = CANCELLATION_CONFIRM_S


def _still_running(live: dict[asyncio.Task[Any], int | None]) -> bool:
    """Is any dispatch in ``live`` genuinely unfinished?

    Not ``bool(live)``. Tasks are harvested by the drain loop, so one that has completed
    stays in the map until the next pass over it — which makes a truthiness test say "a
    sibling is running" about work that has already stopped. Only ``done()`` is the
    authority, and this is the question the immediate-refill decision rests on: a
    replacement is worth an extra fleet only while something else still holds the round
    open.
    """
    return any(not task.done() for task in live)


def _upsert_limit(name: str, limit: int, *, what: str, log: logging.Logger | LoggerAdapter) -> None:
    """Set a Prefect global concurrency limit to ``limit``.

    Both fleet-wide caps this campaign relies on — simultaneous ingests and
    simultaneous committers — are enforced by gates in CHILD flow runs, on other
    machines. Nothing in-process can see across those, so the number has to live on
    the Prefect server; and a server-side number the operator keeps in step with a
    flow parameter by hand is a silent-drift bug. Writing it from the parameter
    makes the parameter the single source of truth.

    Raises on failure, deliberately. This runs in preflight, before a single cluster
    is dispatched, so failing here costs nothing. Failing later costs a great deal:
    the clusters would already be running, and the choice at that point is between
    proceeding ungated — blowing through the cap this call exists to set — and
    stopping them one by one at their first acquire, after their work has been
    scheduled. A limit that cannot be written is a broken precondition, not a
    degraded mode.
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
    optical_min_obs: int | None,
    code_identity: str,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> str:
    """Deterministic staging run_id fingerprinting everything that determines the embeddings.

    Covers the config (threshold/orbit/window/checkpoint/S2-only), the INFERENCE CODE
    the fill runs (``code_identity``), and the per-(zone,year) mosaic identity
    (:func:`_mosaic_identity`). A retry with identical inputs resumes the same prefix
    (findable for cleanup); a change to any of them starts a fresh prefix, so old tiles
    are never resumed under new inputs. Call AFTER ingest so ingest=True reads the
    freshly-written marker.

    **``code_identity`` was narrowed on 2026-07-30.** It used to be the resolved AMI ID
    plus the source tarball's ETag — the whole build. That was correct but far wider than
    the invariant needs: a re-baked AMI, or a hotfix anywhere in the repo, abandoned every
    staged tile and re-ran inference for no semantic reason. It is now
    :func:`~tessera_embeddings.config.inference.inference_code_identity`, a hash of only
    the source that determines what a staged tile CONTAINS. The AMI is still resolved and
    pinned into every fill's provisioning — it just no longer decides staging reuse.
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
        # The store's minimum-depth rule, for exactly the same reason as allow_s2_only: it
        # decides WHICH pixels get embeddings. Staging lives under `outputs/staging/{zone}/
        # {year}`, which is not namespaced by store, so re-running the same mosaics into a
        # store with a different rule would otherwise resume tiles staged under the old one
        # and publish a mix of two depth policies under one fingerprint.
        optical_min_obs,
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
    allow_ingest_code_mismatch: bool = False,
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
        "allow_ingest_code_mismatch": allow_ingest_code_mismatch,
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
    n_clusters: int,
    *,
    land_mask_path: str,
    pending_years: Mapping[str, int] | None = None,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> list[list[str]]:
    """Split ``zones`` into ``n_clusters`` lists of ~equal total inference WORK.

    Longest-processing-time greedy: zones descending by work, each assigned to the
    currently-lightest cluster — so the sequential fill's clusters finish their zone
    lists at roughly the same time.

    **Work, not area (changed 2026-07-30).** This balanced on raw live-tile counts,
    which balances AREA. Inference consumes one sequence per pixel, so its cost scales
    with `pixels x observations`, and observation count varies about twofold with
    latitude — so two clusters with identical tile counts could carry materially
    different work and finish at different times. :func:`zone_work_weight` weights each
    live tile row by its latitude band's observation count, at the same one-GET cost.

    Two reasons this mattered more than it looks. Clusters are long-lived and their
    finish times set the campaign's, so a systematically heavy cluster is not averaged
    away. And the imbalance is not random: observation count varies with latitude, so a
    cluster that happens to draw high-latitude zones is heavy in every year, not just
    one.

    The within-cluster ORDER is deliberately untouched: the chained fill re-sorts its
    own zones by tile count (``fill_zones_sequential``), because ordering and actor
    clamping are properties of area — a zone needs one actor per tile however many
    observations each pixel has, and the descending sort is what lets an autoscaled
    fleet only ever shrink. This function decides *which* cluster gets a zone; the
    child decides *when*.

    A second property falls out of the same ordering and is load-bearing: the
    first ``n_clusters`` zones are the densest in the year and every cluster gets
    exactly one, because all totals start at zero. So each cluster OPENS on a
    dense zone and tapers towards sparse ones. That is what lets its fleet start
    as soon as that one zone has ingested: a big opening zone takes long enough to
    infer that the rest of the cluster's ingest window lands behind it, and
    inference is slower than ingest in almost every case. Change the assignment
    order and that guarantee goes with it.
    **Weighted by the YEARS assigned with the zone.** A cluster receives every pending
    ``(zone, year)`` of the zones it owns, so a zone's cost to that cluster is its
    per-year work times how many years it carries. Weighing it once made a zone missing
    five years the same as one missing one — irrelevant while every zone owed the same
    single year, and wrong the moment ``overlap_years`` let a batch span several, or a
    repair run left zones with uneven gaps. The heavy cluster then drains its extra years
    while the rest sit idle, and its finish time is the campaign's.

    ``pending_years`` is that count per zone, and a zone with none weighs zero — which
    covers the retag-only crash-recovery cells the child settles without GPU work, whose
    tile counts would otherwise skew the balance, and skips their mask reads. Omit it and
    every zone counts as one year, the single-year behaviour. Tile counts are one ~1 KB
    GET per zone; ``n_clusters == 1`` skips the reads entirely. Empty clusters (more
    clusters than zones) are dropped.
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
    fill_strategy: str = "chained-clusters",
    chained_fill_deployment: str | None = None,
    commit_limit_name: str = "tessera-global-commits",
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
    # Staging-reuse escape hatches. Both default off; see `_staging_code_identity`.
    force_staging_reuse: bool = False,
    force_staging_restage: str = "",
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
        max_parallel_clusters: Bounds simultaneous Ray clusters within a year (a
            cost knob, distinct from the commit gate): the concurrent per-cell
            fill runs under ``"cluster-per-zone"``, or the number of Ray clusters
            under ``"chained-clusters"``. **Defaults to 10**, the campaign's planned
            width: 10 x 250 actors reaches the full 2,500-actor quota while keeping each
            cluster's assembly thread under its ~275-actor ceiling, and balance holds to
            ~16. Do not confuse it with ``max_parallel_ingest`` (60), which caps
            simultaneous zone-INGESTS. 60 divides evenly by 10, which matters: the
            per-cluster share is the cap over the cluster count ROUNDED UP, so an
            uneven split aims the fleet above its own gate and logs a width it never
            runs at. Measurement favours many narrow fleets over few
            wide ones, so raising this means sizing ``num_actors`` and
            ``IngestSettings.max_workers`` down to match, or the aggregate fleet will
            exceed the account's EC2 quota.

            These four — this, ``max_parallel_ingest``, ``num_actors`` and
            ``overlap_years`` — are ONE decision and move together
            (``context_docs/design/campaign-plan.md`` §3). They default to the campaign's
            shape rather than to something conservative because this flow has one caller:
            the global campaign. A width that is wrong does not fail — the run completes and
            publishes real data at a fraction of the intended rate, with no error and no
            symptom but a wall clock nobody has a baseline for.
        fill_strategy: Named for the CLUSTER LIFECYCLE — both strategies run
            up to ``max_parallel_clusters`` zones at once.
            ``"chained-clusters"`` (default) dispatches up to ``max_parallel_clusters``
            ``fill-zones-sequential`` runs per year, each owning ONE
            long-lived Ray cluster whose actors stream through a
            size-balanced share of the year's zones — strictly ordered, the
            next zone's tiles interleaving only at queue exhaustion, so zone
            tails never idle the fleet and there is no per-zone teardown,
            actor churn, or model reload. Amortizes ``ray up`` + per-worker
            bringup + model-load cold starts across the whole cluster; ingest
            look-ahead and trailing assembly cover the remaining seams (see
            the runner's module docstring). Each cluster also ingests EVERY one of
            its cells before requesting GPUs, so a fleet is never billed against
            an unfinished ingest — which makes the up-front ingest scale with
            cluster size. ``max_parallel_clusters=1`` degenerates to a single cluster
            for the whole year, and therefore to a whole year of ingest before
            any inference. ``"cluster-per-zone"`` dispatches one
            ``fill-zone-year`` run per cell instead, each provisioning its own
            short-lived Ray cluster — simpler, and it pays a full cluster
            bringup and model load per zone.
        chained_fill_deployment: ``flow-name/deployment-name`` of the chained
            fill deployment (``fill_strategy="chained-clusters"`` only). ``None``
            (default) derives ``fill-zones-sequential/fill-zones-sequential``
            routed by ``branch``; an explicit value is used verbatim.
        commit_limit_name: Prefect global concurrency limit bounding fleet-wide
            simultaneous committers to the global store (ADR-008 D6); forwarded to
            every fill, which holds one slot for the duration of each zone-year
            commit. Its VALUE is derived, not a parameter: the campaign upserts it
            to ``min(max_parallel_clusters, MAX_SIMULTANEOUS_COMMITTERS)`` at
            preflight — a cluster's trailing assembly is single-threaded, so more
            slots than clusters could never be used, and the run-1 curve caps it at
            8 however many clusters run. Set to ``""`` to disable the gate; the
            fills then commit ungated, which the run-1 storm makes a bad idea.
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
        max_parallel_ingest: How many UTM zones may ingest simultaneously across the
            WHOLE campaign, each provisioning its own Dask cluster. This is the only
            limit on ingestion — clusters submit every zone they own at once, and
            each holds one slot for the duration of its ingest, so a cluster starts
            as many as fit under this cap and queues the rest. Under
            ``"chained-clusters"`` it is enforced by ``ingest_limit_name`` (the
            clusters are separate flow runs, so the cap has to live server-side);
            under ``"cluster-per-zone"`` an in-process semaphore is equivalent,
            because that strategy has one driver.
        force_staging_reuse: Reuse staged tiles across an inference code change. Both
            ``force_staging_*`` knobs move the STAGING fingerprint — which S3 prefix the
            intermediate tiles are written to and read back from — and nothing else. Neither
            one touches the published store or relaxes any gate on it: a cell's completion
            mark, its write-once tag and its manifest checks are unaffected either way.

            **This flag cannot reach staging that was created without it.** It substitutes a
            constant into the run-id hash, so it produces a *different* prefix from the one an
            unflagged run staged under; it preserves reuse only between runs that both set it.
            To resume a specific existing prefix, pass ``fill-zone-year``'s explicit ``run_id``
            — that is the only reliable lever.

            Set it when a change to the inference source provably cannot alter staged output (a
            log line, a comment, a type annotation) and the staging hours are worth having.
            Unsafe if that judgement is wrong: it mixes two code versions into one write-once
            zone-year, and ``assemble_global`` probes the variable set from a single tile on the
            assumption that a staging prefix is homogeneous.
        force_staging_restage: An arbitrary token mixed into the staging fingerprint, so a
            change the source hash CANNOT see starts a fresh prefix. Its case is the opposite
            of ``force_staging_reuse``'s: a deliberate dependency upgrade mid-campaign, where a
            new torch changes the numbers without changing our source. Any new value starts
            fresh; reusing the same value resumes what that value staged.

            Unlike ``force_staging_reuse`` this is a production tool — abandoning stale staged
            work is always safe, where reusing it across a real change is not.
        overlap_years: Drop the YEAR BARRIER — dispatch every requested year as one
            batch instead of one batch per year, so a cluster works a multi-year list and
            year N+1's ingest overlaps year N's inference. **Default ON**: it is the
            campaign's planned shape, and it is what makes ``max_parallel_ingest`` reachable
            — with the barrier in place the ceiling is 45 whatever the cap says. Certified
            on six cells each carrying both radar orbits, including a same-zone year
            rollover inside one cluster. What makes it safe is not this flag but two things
            underneath it: each cell carries its own inference window to the actors, and
            the zone-group attribute commit is separate and retried, so two years of one
            zone no longer collide. The partition is over ZONES, so a zone's every year
            lands in one cluster and its assemblies serialize on that cluster's single
            trailing thread. Pass ``False`` to restore the barrier — a repair pass over one
            year still wants it.
        max_dispatch_rounds: How many ROUNDS of re-dispatch the campaign runs — not a
            per-zone budget. Each round dispatches everything still missing, whatever
            zones that is, and re-reads the STORE to decide, so it never re-does landed
            work and an interrupted mosaic is resumed rather than rebuilt.

            **This is the outer recovery, and the only one that survives a child run
            dying.** A killed container or a cancelled run takes its in-cluster attempt
            counter with it, so nothing inside the run can retry anything; interruptions
            are EXPECTED here, because the orphan sweeper cancels child runs by design.
            The child's own ``attempts_per_cell_in_cluster`` covers the cheaper case where
            the run survives and one cell failed.

            Rounds stop early when one makes no progress at all — that is what a
            DETERMINISTIC failure looks like from here (a coverage gate, a fingerprint
            mismatch, an unseeded group), and it is what bounds the two budgets
            compounding on a cell that will never succeed. Those want a human, not
            another cluster. Set to 1 to disable retries.
        immediate_refill: Replace a settled chained fill's still-missing cells at
            once, instead of waiting for the whole round to close. **Off by default,
            and off is byte-for-byte today's behaviour** — the dispatch path is the
            same either way, and with the flag off nothing is ever added to the set
            of live dispatches, so a round waits for every cluster exactly as it did
            and reports its outcomes in the clusters' original order.

            The problem it addresses: the round above is a BARRIER. A cluster owns a
            roster of zones, so when one dies early its whole roster waits for every
            sibling to finish before the store is re-read — and the sibling that
            decides that wait is the slowest one in the round.

            When on, a settled cluster's cells are re-dispatched into the slot it just
            vacated, so the replacement inherits its ingest share, its committer share
            and its place under ``max_parallel_clusters``: the fleet's width and cost
            do not change. Two conditions must BOTH hold first, and when either fails
            the cells simply wait for the round, which is never worse than the default:

            1. The predecessor reached a state that PROVES it stopped writing
               (``_QUIESCENT_TERMINAL_STATES``). A crash, a cancellation, or a
               dispatch that raised — which may still have dispatched — all decline.
            2. ``_SETTLE_DELAY_S`` has passed since it settled, so the asynchronous
               cancellation of the level BELOW its own children has had time to take
               effect. Condition 1 covers the fill's own writers and, because its teardown
               waits for them, its direct children too; what nothing waited for is their
               grandchildren, and the delay is that same confirmation budget again for
               that level.

            **Why a wait rather than a check of who is writing.** A census of live runs
            was tried and removed. It can only report what was true a moment ago, and it
            cannot make a lingering child stop — the wrong instrument for settling an
            asynchronous cancellation, and one whose paging, state-filter and staleness
            semantics were all load-bearing and none of them reliable. Two mechanisms
            already ACT rather than observe, and the delay is what lets them finish: the
            fill's own teardown, which cancels its children and waits for each to confirm
            terminal; and the orphan sweep, which independently finds and stops whatever
            outlived a teardown, on its own schedule.

            What remains is not a reservation. Nothing here locks a cell, so a dispatcher
            outside this campaign could still start a writer. That is not introduced here
            — the round's own re-dispatch has always worked this way — and closing it means
            fencing at the WRITE, the same prerequisite the crashed and cancelled cases
            above are waiting on. The store makes the residual affordable rather than
            silent: mosaic commits do not rebase, so a second writer fails loudly.

            The two-writer invariant is otherwise unchanged and still structural: a
            replacement covers ONLY zones its predecessor owned, and the round's
            clusters partition zones disjointly, so it cannot reach a zone a live
            sibling holds.

            Bounded per CELL and for the life of the campaign, not per round: a cell that
            has had its replacement is not eligible for another on any later round. A
            replacement is also not itself eligible for one, and none is issued unless a
            sibling is still running — with the round over the loop re-dispatches anyway,
            so a replacement then would be an extra fleet for no wall clock. A cell
            therefore gains at most ONE attempt beyond ``max_dispatch_rounds`` for the
            whole campaign, and a persistent capacity shortage cannot become a dispatch
            loop.

            Every read the decision makes — the store's tip and tags, and the
            replacement's own land-mask and SSM probes — declines on failure rather than
            propagating. Uncertainty is not permission, and an exception escaping mid-round
            would fail the campaign while sibling fills were still running, which an
            ordinary FAILED state does not sweep.

            NOT part of the staging fingerprint, deliberately: this changes WHEN work is
            dispatched, not what is computed, and folding it in would abandon the staged
            tiles a retry exists to resume.
        ingest_limit_name: Prefect global concurrency limit backing
            ``max_parallel_ingest`` under ``"chained-clusters"``. The campaign
            upserts it to that value at start, so the parameter is the single place
            the number is written and cannot drift from the server's.
        inference_pause_gate: Name of the global concurrency limit that PAUSES inference.
            Upserted to 1 at start (any positive value means run) and read — never acquired
            — by every cluster's dispatch loop. **``"chained-clusters"`` only**, which is the
            campaign's strategy: ``cluster-per-zone`` dispatches one ``fill-zone-year`` per
            cell, and a per-cell run has no stream to hold, so the gate is neither created nor
            forwarded there. Cancelling is the only lever on that path. Set it to **0** and each cluster lands the
            chunks it has queued and then takes on no further cell, keeping its actors and
            its run alive; set it back to 1 and they carry on where they stopped. Nothing
            fails either way and no cell is half-written, because a cell enters inference
            whole.

            **This is one of the two pause levers, and they cover different halves.** Zeroing
            ``ingest_limit_name`` stops mosaics being built; zeroing this stops embeddings
            being computed from the mosaics that exist. Zero both to wind the campaign down
            to the cells already in flight. **Neither stops the bill** — a cluster holds its
            GPU fleet for its whole multi-cell walk, so a paused fleet idles at full width.
            Only cancelling ends the spend.
        cleanup_mosaics: Delete ``mosaics/{zone}/{year}`` (all versions) after the
            fill lands (default; the mosaic is a transient input). Keep for dev.
            Nothing throttles ingest against fill throughput, so at these defaults
            a year's mosaics can accumulate faster than the fills drain them —
            this is what keeps that from becoming permanent.
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
        allow_ingest_code_mismatch: Forwarded to every ingest: resume interrupted mosaics
            built by different ingest code, instead of demanding they be deleted.
        sweep_orphan_mosaics: Before the run, delete mosaics for cells that are
            already complete+tagged in scope — recovering orphans left by a
            per-cell cleanup that failed after tagging (that cell is no longer in
            `work`, so it is never retried otherwise). Off by default; the zero-cost
            backstop for transient mosaics is an S3 lifecycle rule on the mosaics
            prefix, which this complements for immediate reclamation.
        validation_deployment: ``flow-name/deployment-name`` of a deployment that checks a
            landed cell. Each fill dispatches it per cell as that cell is tagged and never
            waits on it, so a cell is validated while the next one is being filled — see
            :mod:`tessera_embeddings.orchestration.prefect.flows._cell_validation`.

            Unlike the refs above it is NOT derived from ``branch``: the validator is a
            consumer's flow rather than one of this library's, so there is no base name
            here to suffix. And unlike every other setting here, ``None`` (the default) is
            NOT forwarded — the parameter is omitted from the dispatch entirely, so each
            fill deployment's own registered value stands. That is where a consumer names
            its validator, and a branch-scoped registration already carries the branch
            suffix. Forwarding None would override that with nothing and switch validation
            off for every cell of the run.

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
    # before failing deterministically on a value known up front.
    #
    # And ahead of everything that touches SHARED state. The limit upserts used to sit right
    # below; they now run after the whole read-only preflight, for the reason this paragraph
    # gave and which turned out to cover more cases than it named — see there.
    if num_actors < 1:
        raise ValueError(f"num_actors must be >= 1, got {num_actors} (no actor would ever run inference)")
    if fill_strategy not in ("cluster-per-zone", "chained-clusters"):
        raise ValueError(f"fill_strategy must be 'cluster-per-zone' or 'chained-clusters', got {fill_strategy!r}")

    campaign_years = tuple(years) if years is not None else CAMPAIGN_YEARS

    # Lazy AWS import so the flow file imports on non-AWS machines (arch tests).
    # The driver reads the global store directly (status, tags, on-axis probe)
    # BEFORE any child flow is dispatched, so a deployment whose store authenticates
    # only through the callback needs it here too — not just in the children.
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials, icechunk_credentials_for

    store_path = paths.global_store(store_name)
    land_mask_path = paths.land_mask_store(mask_name)
    # THE STORE may be the partner-owned published bucket, which the task role cannot open. Only
    # the four store reads take this; the land-mask reads, the mosaic fingerprint and the live-tile
    # partition stay on the task role, which is the only credential that can read OUR buckets.
    store_credentials = icechunk_credentials_for(store_path, iam_icechunk_credentials)
    repo = open_global_repo(store_path, get_credentials=store_credentials, region=s3_region)
    # ONCE, before any dispatch. Each fill re-checks this, but by then the campaign has
    # already paid for that cell's ingest — and it dispatches cells concurrently, so a
    # store seeded for a different encoder buys a multi-terabyte mosaic per in-flight
    # zone before the first fill fails. Those mosaics are retained on failure (that is
    # what makes a resume cheap), so the disk stays occupied too. A metadata-only read.
    _assert_seeded_model_matches(
        store_path,
        build_checkpoint=checkpoint_filename(),
        allow_model_mismatch=allow_model_mismatch,
        get_credentials=store_credentials,
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
                store_path, seeded_zones[0], y, get_credentials=store_credentials, s3_region=s3_region
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

    # SHARED CONTROLS ARE WRITTEN LAST, once every read-only check has passed and there is work
    # to do. One of these RESETS the fleet-wide inference pause gate to running, and zero is how
    # an operator pauses a campaign already in flight — so a run that flipped the gate and then
    # rejected would have resumed somebody else's paused fleet on its way out. The parameter
    # checks were moved above the upserts for that reason; the store read, the model gate, the
    # year-axis probe, the seeded-zone guards and the work list all sit between them, and any of
    # those can reject too. A no-work invocation is the same hazard without an error: it would
    # resume the gate and then dispatch nothing, so `work` gates this as well.
    #
    # Cheap, idempotent and safe to repeat — the ordering is the only thing that matters.
    if work:
        if ingest and fill_strategy == "chained-clusters":
            _upsert_limit(ingest_limit_name, max_parallel_ingest, what="ingest", log=log)
        if inference_pause_gate and fill_strategy == "chained-clusters":
            # Upserted to ONE, and one is not a cap: this gate is read as a flag rather than
            # acquired, so any positive value means "run" and nothing consumes a slot. Created
            # here so that pausing is always available — a gate an operator has to create first
            # is not a lever they can reach at 3 a.m. — and reset to running at every start that
            # has work, so a campaign can never inherit a pause somebody left behind.
            _upsert_limit(inference_pause_gate, 1, what="inference pause (1 = running)", log=log)
        if commit_limit_name:
            # DERIVED, not a parameter. A cluster's trailing assembly is a single
            # thread, so N clusters can produce at most N assembly commits at once —
            # a larger limit would be a number that never binds. And the run-1 curve
            # says never exceed MAX_SIMULTANEOUS_COMMITTERS however many clusters run.
            #
            # A cluster's FEEDER can also commit (a terminal plan inside `plan()`),
            # so the fleet's true ceiling is 2N and the gate can briefly queue those.
            # That is the intent: the gate is a bound, not an operating point, and a
            # queued commit costs seconds against zones that run for hours.
            commit_limit = min(max_parallel_clusters, MAX_SIMULTANEOUS_COMMITTERS)
            _upsert_limit(commit_limit_name, commit_limit, what="commit", log=log)

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
        max_parallel_clusters,
        commit_limit_name,
    )

    optical_rule_cache: list[int | None] = []

    def _campaign_optical_min_obs() -> int | None:
        """The store's minimum-depth rule, for the staging fingerprint.

        Read from the store because the store is the authority (its root attr is part of a
        write-once identity), and cached because every cell this campaign dispatches writes
        the same store and the root cannot change under it. In the fingerprint for the same
        reason ``allow_s2_only`` is: it decides which pixels get embeddings, and staging is
        not namespaced by store.
        """
        if not optical_rule_cache:
            optical_rule_cache.append(
                _optical_min_obs_from_store(store_path, get_credentials=store_credentials, s3_region=s3_region)
            )
        return optical_rule_cache[0]

    code_identity_cache: list[str] = []

    def _staging_code_identity() -> str:
        """The staging fingerprint's code component — narrowed, with two escape hatches.

        Default: the inference-source hash, so an orchestration or ingest hotfix reuses
        staged tiles instead of abandoning them.

        ``force_staging_reuse`` collapses this to a constant, so even an INFERENCE change
        reuses the prefix. That is deliberately unsafe and exists for the one case the
        narrowing cannot judge: a change inside the inference package that the author knows
        does not alter staged output (a log line, a comment, a type annotation). It mixes
        code versions in one write-once zone-year if that judgement is wrong.

        ``force_staging_restage`` mixes in a caller-supplied token, so a change the hash
        CANNOT see starts a fresh prefix. The case that needs it is a deliberate library
        upgrade mid-campaign — a new torch changes the numbers without changing our source.
        """
        if force_staging_reuse:
            return "infcode-forced-reuse"
        identity = inference_code_identity()
        # The source hash covers what the flow runner can SEE. On the dev-overlay path the
        # workers execute a tarball they download instead, and replacing that tarball changes
        # what runs without changing anything here — so a retry would resume tiles staged by
        # the old code and add new ones from the new. The tarball's ETag closes that, and it
        # is EMPTY for a baked-AMI deploy (code_bucket=None), which is production: this term
        # costs prod nothing and exists only where the exposure does.
        #
        # Deliberately the tarball term ALONE, not resolve_code_artifact_identity's whole
        # string. Folding the AMI back in is what the 2026-07-30 narrowing removed, and for a
        # good reason — re-baking an image does not change what a staged tile contains, so it
        # abandoned every staged tile for nothing.
        tarball = _resolve_tarball_identity(code_bucket, code_suffix, None)
        if tarball:
            identity = f"{identity}|{tarball}"
        return f"{identity}+{force_staging_restage}" if force_staging_restage else identity

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
            # Explicit, and it must stay explicit: `fill-zone-year` DEMANDS radar by
            # default, because a lone cell dispatched by hand should report missing radar
            # rather than quietly produce optical-only embeddings. A global campaign is the
            # opposite case — parts of the globe have no dual-pol coverage at all, and
            # refusing them fails those cells on every retry forever.
            "require_s1": False,
            # The child re-runs the model gate this flow already cleared in preflight, so
            # without forwarding the override it rejects the same store — after its ingest
            # has been paid for. The campaign-level flag has to reach the gate that fires.
            "allow_model_mismatch": allow_model_mismatch,
            # Divide the fleet S3-PUT budget across concurrent fills so K shard-write
            # phases don't burst K times the target PUTs (the ~800-req SlowDown). D6
            # gates committers; this bounds the ungated upload phase.
            "s3_concurrency": max(1, TARGET_AGGREGATE_S3_CONCURRENCY // max_parallel_clusters),
            "ssm_prefix": ssm_prefix,
            "cloudwatch_log_group": cloudwatch_log_group,
            "code_bucket": code_bucket,
            "code_suffix": code_suffix,
            # OMITTED when this campaign names no validator, and that is the difference
            # between "the campaign decides" and "the campaign overrides". An omitted
            # parameter takes the CHILD DEPLOYMENT's own registered value, which is where a
            # consumer names its validator (and where a branch-scoped registration already
            # carries the branch suffix). Passing None instead would override that with
            # nothing and silently switch validation off for every cell — the failure mode
            # of a default nobody sets.
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
    # NO cap on cells holding a live mosaic. There used to be one (ADR-011's "peak
    # input storage bounded by in-flight cells", sized to fills + an ingest
    # look-ahead), and it was removed deliberately: it made ingest wait on fill
    # throughput, which throttles the cheap half of the campaign to protect a
    # storage cost we would rather pay. Ingest is far more efficient run wide, so
    # mosaics may now accumulate up to a whole year's worth (~120 cells) before the
    # fills drain them. `cleanup_mosaics` still deletes each one as its fill lands,
    # and `sweep_orphan_mosaics` still collects what a crash left behind; what is
    # gone is the BACKPRESSURE, not the cleanup.
    #
    # This applies to `cluster-per-zone` only. In `chained-clusters` — the default —
    # the per-cell chain below is bypassed and each child bounds its own mosaics at
    # `look_ahead + 2` (sequential_fill._MosaicBudget). Across the cluster count those
    # settings imply, that bound already exceeds the zones in a year, so it is not
    # the binding constraint there either.

    async def _process(zone: str, year: int) -> str:
        # Ingest (if enabled) → fill → drop the transient mosaic. ingest_sem/fill_sem
        # cap the expensive Dask/Ray clusters; nothing gates the chain as a whole.
        #
        # A complete-but-untagged cell (in years_complete, missing its tag) is in
        # the work list only so the fill re-creates the tag — no inference, no
        # mosaic. Skip ingest and cleanup for it entirely.
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
    #: Cells still missing after every attempt, per year. Collected rather than
    #: raised on, so one bad zone cannot cost the years that follow it.
    unfilled: dict[int, list[str]] = {}
    #: Cells that have already had their one immediate replacement, for the life of the
    #: CAMPAIGN rather than of a round. The bound `immediate_refill` advertises is per
    #: cell — at most one attempt beyond the round budget — and rounds are re-entered, so
    #: a marker scoped to a round would grant that extra attempt again on every one.
    refilled_cells: set[tuple[str, int]] = set()
    # Descending, so the campaign delivers the most recent year soonest (ADR-008 D1).
    # `overlap_years` chooses only the BATCHING: one batch per year keeps the historical
    # year-serial shape, one batch for everything drops the year barrier. Everything
    # inside is identical, which is the point — the two paths cannot drift.
    ordered_years = sorted(set(campaign_years), reverse=True)
    batches: list[list[int]] = [ordered_years] if overlap_years else [[y] for y in ordered_years]
    for batch_years in batches:
        # BOUNDED RE-DISPATCH. A failed zone must not cost the campaign the rest of
        # the batch, nor the batches after it — interruptions are EXPECTED here, because
        # the orphan sweeper cancels child runs by design. Every round re-reads the
        # STORE (not our bookkeeping) for what is still missing, so a retry can only
        # ever target genuinely unfilled cells and an interrupted mosaic is resumed
        # rather than rebuilt.
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
                # Up to max_parallel_clusters child runs fill the year, each owning
                # ONE shared Ray cluster and draining its own list of zones strictly
                # sequentially — clusters take up their next zone without actor
                # churn instead of being torn down per cell. Shards are
                # size-balanced (LPT over live-tile counts) so they finish
                # together. Ingest, per-cell run_ids, retag/empty triage, and
                # mosaic cleanup all live inside the children (their ingest
                # look-ahead pipelines mosaics against each cluster), so the
                # per-cell _process chain is bypassed; the global ingest bound is
                # preserved by dividing max_parallel_ingest across the clusters.
                clusters = _partition_by_live_tiles(
                    batch_zones,
                    min(max_parallel_clusters, len(batch_zones)),
                    land_mask_path=paths.land_mask_store(mask_name),
                    # A cluster gets every pending (zone, year) of the zones it owns —
                    # see `cells_for` below — so a zone's cost to it is per-year work
                    # times the years it carries. Already-landed years count zero: the
                    # child retags them without GPU work, and a zone with none left is
                    # weightless outright (its mask read is skipped too).
                    pending_years={
                        z: sum(1 for zz, y in batch_cells if zz == z and not status.has(z, y)) for z in batch_zones
                    },
                    get_credentials=iam_icechunk_credentials,
                    s3_region=s3_region,
                )
                # Each cluster's own cells: every (zone, year) pair whose zone it owns.
                cells_for = {id(cl): [[z, y] for z, y in batch_cells if z in set(cl)] for cl in clusters}
                # The same partition, keyed by year, kept here rather than re-derived from
                # `cells_for` because that one is JSON-shaped for the deployment parameters
                # (lists, not tuples) and has lost the year's type by the time it is read.
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
                # divide it by construction rather than racing for slots on the global
                # gate — which sets the hard ceiling, not who reaches it first. Rounded
                # up: flooring would silently deliver less ingest width than was asked for.
                #
                # MINUS ONE, because `look_ahead` counts cells BEYOND the current one:
                # the fill runs `1 + look_ahead` ingests per cluster (its ingest driver is
                # sized `max_parallel=1 + look_ahead`). Passing the whole share therefore
                # aimed each cluster at `1 + share`, so the fleet asked for
                # `(1 + share) * clusters` — 48 against a cap of 40 at the shipped 8
                # clusters. The global gate still held the line, so this oversubscribed
                # rather than breached: clusters queued on the gate, and the log below
                # reported a width the fleet never ran at. Floored at 0 so a
                # one-cell-at-a-time cluster can still start its current cell.
                #
                # Rounding up still overshoots by up to `clusters - 1` when the cap does
                # not divide evenly (45 over 8 aims at 48); the gate remains the ceiling,
                # and widening the share is the deliberate side to err on.
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

                # `status` is re-read between retry rounds, so it is bound as a default
                # rather than captured: a closure that outlived its round would otherwise
                # read whichever snapshot the loop left behind.
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
                # it is defined inside the year loop, so a captured name would be whatever
                # the last iteration left behind if the call ever outlived its iteration.
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
                        # Pin the fingerprinted image (see _ami_id / _fill_params) — but
                        # ONLY for a cluster that will actually start Ray. A cluster whose
                        # cells are all already-complete (retag) or all-ocean never
                        # provisions, and forcing an SSM read there breaks the cheap
                        # recovery paths when the parameter is missing or the role lacks
                        # access. The ocean probe is one small bitmap GET per zone — the
                        # same price the per-cell path already pays — and it runs only
                        # for cells that are not already complete.
                        # The snapshot is passed THROUGH rather than defaulted here, so an
                        # immediate refill probes against the store it just read rather than
                        # against the one the round opened on.
                        "ami_id": _ami_id() if _needs_ray(cells, snapshot) else None,
                        "store_name": store_name,
                        "mask_name": mask_name,
                        "num_actors": num_actors,
                        "s1_orbit": s1_orbit,
                        "s3_region": s3_region,
                        "commit_limit_name": commit_limit_name,
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
                        # The narrowed staging identity, with the force_staging_*
                        # overrides applied — same value the per-zone path folds into
                        # its run_id. Without it the child recomputed the AMI/tarball
                        # artifact identity, so the default strategy re-inferred on
                        # every re-bake and ignored both escape hatches.
                        "staging_code_identity": _staging_code_identity(),
                        "ingest": ingest,
                        "ingest_deployment": ingest_deployment,
                        # Route this cluster's S1/S2 ingest grandchildren (see `branch`);
                        # the direct ingest/fill refs above are already branch-resolved.
                        "branch": branch,
                        # This cluster's OWN zone count, so its admission gates never
                        # limit fleet fill (see the dispatch comment above). The
                        # staging+assembly slice below keeps the fleet-wide S3-PUT rate
                        # at the aggregate target.
                        "look_ahead": cell_look_ahead,
                        "s3_concurrency": max(1, TARGET_AGGREGATE_S3_CONCURRENCY // (2 * n_clusters)),
                        "cleanup_mosaics": cleanup_mosaics,
                        "ingest_settings": ingest_settings.model_dump(),
                        # The shared cap. Every cluster names the same limit, which is
                        # what makes it fleet-wide rather than per-cluster.
                        "ingest_limit_name": ingest_limit_name,
                        # And the shared pause. Same reason it is one name for every
                        # cluster: an operator pauses the CAMPAIGN, not a cluster.
                        "inference_pause_gate": inference_pause_gate,
                        # As in _fill_params: omitted when unset, so the child's own
                        # registered validator stands. Each cell's validation is dispatched
                        # by the child off its trailing assembly thread as the cell is tagged.
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
                # `_chained_params` gives beside it: this is defined inside the batch loop, so
                # a captured name would be whatever the last iteration left behind if the call
                # ever outlived its iteration.
                async def _refill(
                    zones: list[str],
                    outcome: Any,  # noqa: ANN401 — Prefect FlowRun or the exception that replaced it
                    years: tuple[int, ...] = tuple(batch_years),
                    n_clusters: int = len(clusters),
                    cell_look_ahead: int = look_ahead,
                ) -> tuple[dict[str, Any], list[str], list[int]] | None:
                    """A ready-to-dispatch replacement for a settled fill, or ``None`` to decline.

                    Returns the finished PARAMETERS rather than a plan, deliberately: everything
                    that can fail — the store re-read, the claim query, the land-mask and AMI
                    probes inside ``_chained_params`` — then sits inside this function's guard,
                    and the caller is left with a step that cannot raise. See the dispatch loop
                    below for why that matters.

                    DECLINING IS ALWAYS SAFE, and it is the answer to everything this function
                    cannot establish, its own inability to ask included. The cells stay in
                    `remaining`, so the round re-dispatches them exactly as it does today. That
                    asymmetry is the whole shape of the function.

                    Condition one is the predecessor's terminal state, which is what proves the
                    fill's OWN writers stopped (see `_QUIESCENT_TERMINAL_STATES`). Condition two
                    is that nothing live claims the cells, which is what covers everything
                    DOWNSTREAM of it: a fill's grandchild ingests are cancelled one level further
                    out and asynchronously, and the in-child sweep says of itself that it is best
                    effort.

                    Refusal is at ZONE granularity, never per cell. Dropping one claimed year and
                    keeping its zone's others would put that zone's years on two clusters, which
                    is the invariant the whole partition exists to hold.
                    """
                    if isinstance(outcome, BaseException):
                        # Its own reason, not folded into the state check below. A dispatch that
                        # RAISED and a fill that ended in a state we cannot trust are different
                        # facts, and an operator reading a declined refill needs to know which.
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
                    # ONE guard over every read this decision makes. Each of them reaches a
                    # network — the store's branch tip and tag list, the Prefect run set, and the
                    # land-mask and SSM probes inside `_chained_params` — and each can fail
                    # transiently. Uncertainty is not permission: a failure anywhere in here
                    # means nothing was established about who is writing, so it declines. Left
                    # unguarded it would instead escape into the drain loop and fail the whole
                    # campaign while sibling fills were still running, and an ordinary FAILED
                    # state does not fire the child-cancellation hook that would clean them up.
                    # ONE guard over every read this decision makes. Each reaches a network —
                    # the store's branch tip and tag list, and the land-mask and SSM probes
                    # inside `_chained_params` — and each can fail transiently. Uncertainty is
                    # not permission: a failure anywhere in here means nothing was established,
                    # so it declines. Left unguarded it would escape into the drain loop and
                    # fail the whole campaign while sibling fills were still running, and an
                    # ordinary FAILED state does not fire the child-cancellation hook.
                    try:

                        def _missing(snapshot: CampaignStatus) -> list[tuple[str, int]]:
                            """This roster's cells the STORE still lacks and that may be refilled.

                            The store decides, exactly as the round does — never the child's
                            return value, which can report success on cells it never attempted.
                            Intersected with the inherited roster as well as scoped by it: the
                            scope is the work list's contract, the intersection is structural,
                            and it is what makes "a replacement can never reach a zone a live
                            sibling holds" a property of this function rather than one delegated.
                            """
                            return [
                                (z, y)
                                for z, y in campaign_work_list(
                                    snapshot, set(repo.list_tags()), expected_zones=zones, years=years
                                )
                                if z in roster and (z, y) not in refilled_cells
                            ]

                        roster = set(zones)
                        # Read FIRST, and cheaply, so the settling wait below is only ever paid
                        # for a roster that actually has something to refill. Most clusters
                        # settle having landed everything, and those must not hold the drain
                        # loop for a wait whose whole purpose is to protect a dispatch that is
                        # not going to happen.
                        if not _missing(campaign_status(repo, years=campaign_years)):
                            return None
                        # THE SETTLING WAIT. The fill is terminal, so its own writers are
                        # provably stopped and its direct children were cancelled AND waited
                        # for. What has not been waited for is the level below them, whose
                        # cancellation was requested by a hook that does not block. This is the
                        # time that request needs to take effect — and time is the instrument
                        # that suits it, because nothing can be observed into stopping.
                        log.info(
                            "Waiting %.0fs before refilling %d zone(s), so the settled fill's remaining "
                            "descendants have time to stop.",
                            _SETTLE_DELAY_S,
                            len(zones),
                        )
                        await asyncio.sleep(_SETTLE_DELAY_S)
                        # RE-READ after the wait, so the dispatch is based on the settled
                        # picture rather than on what was true before it. A cell that landed
                        # meanwhile is no longer missing, and a fleet is not spent on it.
                        fresh = campaign_status(repo, years=campaign_years)
                        missing = _missing(fresh)
                        if not missing:
                            return None
                        # The cluster count is the ROUND'S: the replacement takes the vacated
                        # slot, so its ingest share and S3 concurrency slice are the ones the
                        # round sized. Inside this guard because its land-mask and SSM probes
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
                # `immediate_refill` off nothing is ever added to `live`, so this waits for
                # every cluster just as a gather would, and the bookkeeping below still reads
                # the outcomes in the clusters' ORIGINAL order — so the dispatch sequence, the
                # log lines and the returned summary are identical, not merely equivalent.
                #
                # THE RULE THIS BLOCK EXISTS TO KEEP: nothing that can await or raise may sit
                # between deciding to dispatch and dispatching. Two consequences, both of which
                # a `gather` used to give for free and a hand-rolled loop has to earn:
                #
                # Every parameter set is built BEFORE any task exists. A starred `gather`
                # consumed its whole generator before scheduling anything, so a `_chained_params`
                # that raised — its land-mask probe, its SSM read — dispatched nothing at all.
                # Interleaving build and schedule would instead leave earlier fills running while
                # the campaign fails, and an ordinary FAILED state does not fire the child-cancel
                # hook, so those runs would be orphaned with their fleets billing.
                #
                # And every task is created INSIDE the try, so no task can exist outside the
                # cancellation path.
                params_by_slot = [_chained_params(cells_for[id(cl)], len(clusters), look_ahead) for cl in clusters]
                live: dict[asyncio.Task[Any], int | None] = {}
                chained_results: list[Any] = [None] * len(clusters)
                #: Replacements, in dispatch order: the zones and years each covered, and the
                #: task carrying its outcome. Kept apart from `chained_results` so the
                #: originals' bookkeeping is untouched by whether any replacement happened.
                pending_refills: list[tuple[list[str], list[int], asyncio.Task[Any]]] = []
                try:
                    for slot, params in enumerate(params_by_slot):
                        live[asyncio.ensure_future(_settle(params))] = slot
                    while live:
                        done, _ = await asyncio.wait(set(live), return_when=asyncio.FIRST_COMPLETED)
                        # Deregister everything that settled BEFORE deciding anything, so
                        # "a sibling is still running" is read off work that really is running.
                        settled = [(live.pop(task), task) for task in done]
                        for settled_slot, task in settled:
                            outcome = task.result()
                            if settled_slot is None:
                                # A replacement carries NO slot, and that is half the bound: a
                                # replacement is not itself eligible for one. Its outcome was
                                # recorded at dispatch; there is nothing to decide here.
                                continue
                            chained_results[settled_slot] = outcome
                            if not immediate_refill or not _still_running(live):
                                continue
                            planned = await _refill(list(clusters[settled_slot]), outcome)
                            if planned is None:
                                continue
                            replacement, take_zones, take_years = planned
                            # THE COMMIT GATE, and nothing between it and the dispatch may await
                            # or raise. `_refill` reads the store and queries Prefect, and a
                            # sibling can finish across those awaits while its task sits in
                            # `live` unharvested until the next `asyncio.wait` — so the check
                            # made before planning is stale by now. Re-read it here, where the
                            # answer cannot go stale, or the round is already over and the
                            # replacement buys an extra fleet and an extra attempt for no wall
                            # clock at all.
                            if not _still_running(live):
                                log.info(
                                    "Not refilling %d zone(s) immediately: the round finished while the "
                                    "refill was being planned, so it is the round's to re-dispatch.",
                                    len(take_zones),
                                )
                                continue
                            log.warning(
                                "Refilling %d cell(s) across %d zone(s) at once rather than waiting for the round: %s",
                                len(replacement["cells"]),
                                len(take_zones),
                                ", ".join(take_zones),
                            )
                            # The other half of the bound, and the half that spans ROUNDS. `live`
                            # is rebuilt every round, so a marker kept there would hand a cell
                            # fresh eligibility on each one and the documented "at most one extra
                            # attempt" would be one per round instead. This set outlives the round.
                            refilled_cells.update((z, y) for z, y in replacement["cells"])
                            refill_task = asyncio.ensure_future(_settle(replacement))
                            live[refill_task] = None
                            pending_refills.append((take_zones, take_years, refill_task))
                finally:
                    # What `gather` did for its own children when the outer await was
                    # cancelled. Cancelling the WAIT does not cancel the child run; the
                    # flow's terminal hook sweeps those by tag, exactly as before.
                    for task in live:
                        task.cancel()
                # Replacement outcomes are futures until here, because the loop records them
                # at dispatch and cannot know their result yet. Every one is done — `live`
                # emptying is what ended the loop.
                refills = [(zs, ys, task.result()) for zs, ys, task in pending_refills]
                # RECORDED, never raised: one cluster failing must not abandon the
                # year, and certainly not the years after it. What each cluster
                # actually landed is already committed and tagged per zone-year, so
                # the round's survivors keep their work and the retry below targets
                # only what is genuinely still missing.
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
                    # Credited to the years this cluster ACTUALLY owns, not to the batch.
                    # A fresh campaign gives every cluster every year, which is what made
                    # `for y in batch_years` look right; a RESUMED one partitions over what
                    # is still missing, so a cluster can hold zones that are each short a
                    # different single year. Crediting the batch then claimed years the run
                    # never touched and inflated `dispatched`, which sums these lists.
                    for y in years_for[id(cl)]:
                        runs_this_round_by_year.setdefault(y, []).append(str(r.id))
                log.info(
                    "Year(s) %s: %d/%d chained fill(s) landed",
                    batch_years,
                    len(clusters) - len(round_failures),
                    len(clusters),
                )
                # The replacements, after their predecessors and in dispatch order, so the
                # count above keeps meaning what it has always meant: the round's own
                # clusters. Same three records as the loop above — a failure is reported, a
                # landed run is credited, and it is credited to the years IT owns.
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

                # Different zones are different store groups, so they fill concurrently.
                # A zone's own years are dispatched one at a time here — but NOT because the
                # store requires it. `write_year_shards` relaxed that on 2026-07-30 (d1a379c):
                # every chunk and shard is 1 in the time dimension, so a zone's years write
                # strictly disjoint objects and always rebased cleanly; the only collision was
                # two group attrs, and those now commit separately and retry, each writer
                # inserting only its own year's key. This serialisation therefore reflects
                # scheduling, not correctness, and it is a candidate for removal — the open
                # question is the attr retry budget (`commit_year_attrs(tries=8)`) once many
                # years of one zone contend. Until that is answered, keep it sequential.
                # return_exceptions=True so a single zone's failure doesn't abandon its
                # siblings mid-flight (default gather() would leave them running orphaned);
                # we let the whole year settle, then fail loudly with every failure.
                years_of: dict[str, list[int]] = {}
                for z, y in batch_cells:
                    years_of.setdefault(z, []).append(y)

                async def _zone_years(zone: str, years: list[int]) -> list[Any]:
                    """One zone's years, in order, each attempted whatever the last one did.

                    Sequential, so a zone never has two years in flight; one result per year,
                    so the round still accounts for every cell it was given.

                    IT DOES NOT STOP AT THE FIRST FAILURE, and that is the point. It used to
                    copy the stopping exception into every later year without dispatching it.
                    Years are ordered oldest-first and every retry rebuilds the list the same
                    way, so a failure that repeats and belongs to ONE year — a coverage gate is
                    evaluated per zone-year, and radar coverage genuinely varies year to year —
                    kept its zone's other years from ever being attempted on any round, then
                    reported them unfilled. The no-progress guard below would eventually halt
                    the campaign over a cell that was never going to succeed while the years
                    behind it were fillable.

                    What that cost bought was fail-fast, and it buys much less than it did: the
                    deterministic zone-wide failures (unseeded group, destination dtype, year
                    off the axis, no live tiles) now surface in `fill_zone_year_flow`'s
                    preflight, BEFORE a GPU fleet is provisioned. Continuing past one costs a
                    few metadata reads. The chained path above already reasons this way
                    ("one cluster failing must not abandon the year, and certainly not the
                    years after it"); this is the same rule on the per-cell path.
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
                        # Copying the whole round into every year reported each year as
                        # holding the others' runs and overcounted `dispatched` by the number
                        # of years in the batch — invisible with one year per round, which is
                        # every batch until `overlap_years` puts several in one.
                        runs_this_round_by_year.setdefault(y, []).append(str(cr))
                log.info("Year(s) %s: %d/%d fill(s) landed", batch_years, len(round_runs), len(batch_cells))

            # Attribution is per strategy, and the two differ: a chained cluster spans the
            # batch's years while a per-cell run fills exactly one. Both populate
            # `runs_this_round_by_year` above, so this only merges it.
            for y, runs in runs_this_round_by_year.items():
                runs_by_year.setdefault(y, []).extend(runs)
            # Re-read the STORE, every round, including the rounds that reported no
            # failures. What landed is a property of the store, never of a child's
            # return value: a child can report success having silently not attempted
            # cells at all — `sequential_fill`'s feeder stops admitting work when too
            # many failures are retaining mosaics, and if those failures then recover
            # in its retry pass it returns clean with cells still pending. Concluding
            # from an empty failure list marked those as landed and ended the campaign.
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
                # A whole round achieved nothing. That is what a DETERMINISTIC
                # failure looks like from out here — a coverage gate, a fingerprint
                # mismatch, an unseeded group — and it wants a human, not another
                # cluster. Stop spending fleets on it.
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

        # Backfill the year-complete milestone/retention tag even when NO fills
        # ran this pass — a prior run may have completed every zone for the year
        # but crashed before tagging. The milestone is defined over ALL 120 zones:
        # `tag_year_complete` takes no scope argument, so this run's `zones` subset
        # cannot narrow it and a subset/repair run cannot stamp the global tag
        # prematurely — which matters because tags are write-once forever.
        # ValueError = not yet all-120 complete → defer.
        # Attempted for EVERY year in the batch. `tag_year_complete` raises ValueError
        # when the year is not yet complete across all 120 zones, so attempting it for an
        # unfinished year is free — which is what lets the overlapping path use the same
        # code as the year-serial one instead of detecting "was that the year's last cell".
        for y in batch_years:
            try:
                year_tag = tag_year_complete(repo, y)
                log.info("Year %d tagged complete (all 120 zones): %s", y, year_tag)
            except ValueError as exc:
                log.debug("Year %d not yet complete across all 120 zones (%s) — milestone tag deferred", y, exc)

    dispatched = sum(len(v) for v in runs_by_year.values())
    log.info("Campaign dispatch complete: %d fill run(s) across %d year(s)", dispatched, len(runs_by_year))
    if unfilled:
        # LOUDLY, and last — but the run SUCCEEDS. A handful of cells failing every attempt is
        # an expected outcome at this scale, not a failed campaign: every other cell is
        # committed and tagged, and the product of those failures is a list of zones for a
        # second wave. Raising here would have said the opposite — it would mark 1,000 landed
        # cells a failure because 3 did not land, fire the driver-stopped alert (which means
        # "nothing is being filled any more", and nothing here is being filled any more
        # because the campaign FINISHED), and leave an operator reading a stack trace to find
        # a list.
        #
        # The list is not quiet, either. It goes in the log at WARNING under a banner, and in
        # the returned summary under `unfilled`, so the report a caller keeps carries it as
        # data rather than as prose to re-parse. Nothing about a cell on this list is
        # ambiguous: it was dispatched, it was attempted every round, and it is missing from
        # the store.
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
