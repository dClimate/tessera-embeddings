"""Ingest one ``(zone, year)`` of the campaign onto the fixed zone grid.

The global campaign reads its per-zone mosaics from ``{inputs}/mosaics/{zone}/{year}``;
this flow produces them by **reusing the ROI ingest engine unchanged**. A
zone-shaped ROI zarr (:func:`tessera_embeddings.ingest.land_mask.export_zone_roi`)
pins the grid to the fixed :class:`~tessera_embeddings.storage.zone_grid.ZoneSpec`
extent — so the resulting mosaic passes the zone-fill's exact-grid validation,
which ``generate_roi``'s bbox-fit grid could not — and the S1/S2 ROI ingest
deployments write onto it. Only orchestration lives here (``arun_deployment``);
each ingest deployment provisions its own Dask cluster.

**Idempotent, and RESUMABLE.** A per-store completion marker (root attr
``ingest_marker``, fingerprinting window + min_valid_coverage + s1_orbit +
allow_partial_window + coverage sha) gates the work: a matching marker on every required
store short-circuits, and the marker is written only after coverage is verified.

Three states, three answers:

* **absent** — ingest it.
* **present and unmarked** — an interrupted attempt at this same work. **RESUMED.** Dates
  already committed are skipped rather than rewritten, because Icechunk commits a date's
  time slot atomically WITH its pixels, so a date present is complete and a date absent
  was never started. Both ingests already skip what they find, so resuming is simply not
  clearing.
* **present and marked DIFFERENTLY** — raises ``ConfigMismatchError``. See the branch
  itself for why neither automatic answer is defensible.

**Campaign assumption this rests on (documented, not enforced here).** A campaign holds its
inputs fixed: one land mask, one date window per year, one coverage threshold, both orbits,
one partial-window policy. So an unmarked store can only be an interruption of the same
work — there is no other configuration for it to be left over from. The one input whose
change would silently corrupt rather than raise is the LAND MASK, and that is enforced
independently: ``IngestManifest.coverage_sha256`` is validated on every write, so a changed
mask fails at the first append.

**Two writers** is the hazard resume makes reachable, and Icechunk guards it: these commits
do not rebase, so a second writer's commit fails with ``ConflictError`` against a moved
branch tip instead of merging silently. **The write retry must exclude that error for the
guard to hold** — retrying re-opens the session from the moved tip and so converts the
refusal into a success; see ``storage.zarr_store.store_write_retrying``. Nothing here
prevents a second run from starting, so the outcome is a wasted cell rather than a corrupt
store: it fails, and a later attempt resumes from what the surviving writer committed. The
dispatch-side guard against launching the duplicate at all lives in
``scripts/run_campaign_cell.py`` in yield-embeddings, deliberately not in this flow — a
refusal here would strand a crashed run that Prefect still reports as ``RUNNING``.

**Coverage gate (ADR-011).** After ingestion, :func:`check_time_window_coverage`
requires the mosaics' months to span the window; ``allow_partial_window`` relaxes
that to "non-empty" for a legitimately partial edge zone. This is the ingest-side
guard; the fill re-checks before provisioning Ray.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import monotonic
from typing import Any, cast

import icechunk
import zarr
from prefect import flow, get_run_logger
from prefect.deployments import arun_deployment
from prefect.runtime import flow_run as flow_run_ctx
from pydantic import BaseModel

from tessera_embeddings.config.ingest import IngestSettings
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.errors import ConfigMismatchError, InsufficientCoverageError
from tessera_embeddings.inference.data_loading import (
    S1_ORBIT_NONE,
    _active_orbits,
    check_time_window_coverage,
    resolve_s1_orbit,
)
from tessera_embeddings.ingest.catalogue_refusal import RefusalKind, refusal_in, repeat_is_deterministic
from tessera_embeddings.ingest.land_mask import export_zone_roi, live_chunk_count
from tessera_embeddings.orchestration.prefect.flows._child_runs import child_run_tag, make_child_cancel_hook
from tessera_embeddings.orchestration.prefect.flows.tessera_full_pipeline import _check_completed
from tessera_embeddings.orchestration.runners.zone_fill import zone_has_live_tiles
from tessera_embeddings.storage.manifest import IngestManifest, extract_manifest
from tessera_embeddings.storage.zarr_store import (
    is_missing_repo,
    open_repo,
    open_store_as_zarr_group,
    plain_zarr_storage_options,
)
from tessera_embeddings.storage.zone_grid import canonicalize_zone
from tessera_embeddings.utils import utcnow_iso


#: Percent-of-ROI valid-pixel threshold for keeping an S2 solar-day. Far below
#: the ROI default (5.0): a single day's swath covers only a sliver of a whole
#: 6° zone, so a high bar would drop nearly every date (see ADR-011).
class IngestDeployments(BaseModel):
    """Deployment refs (``flow_name/deployment_name``) the campaign dispatches to."""

    ingest_s1_roi_sar: str = "ingest_s1_roi_sar/ingest-s1-roi-sar"
    ingest_s2_roi_reflectance: str = "ingest_s2_roi_reflectance/ingest-s2-roi-reflectance"


def _mosaic_stores(mosaic_base: str, s1_orbit: str) -> list[str]:
    """The child stores a fill reads under ``mosaic_base`` for ``s1_orbit``."""
    stores = [f"{mosaic_base}/reflectance.zarr"]
    stores.extend(f"{mosaic_base}/sar_{orbit}.zarr" for orbit in _active_orbits(s1_orbit))
    return stores


_Creds = Callable[[], "icechunk.S3StaticCredentials"] | None


def _coverage_sha(land_mask_path: str, zone: str, *, get_credentials: _Creds, s3_region: str | None) -> str | None:
    """The coverage delivery sha (``registry_sha256``) for ``zone``, or ``None``."""
    cov = open_store_as_zarr_group(land_mask_path, group=zone, get_credentials=get_credentials, region=s3_region)
    return cast("str | None", cov.attrs.get("registry_sha256"))


def _probe_marker(store_path: str, *, get_credentials: _Creds, s3_region: str | None) -> tuple[bool, dict | None]:
    """``(exists, marker)`` for a mosaic child store.

    ``exists`` is whether the repo is physically present; ``marker`` is its
    ``ingest_marker`` fingerprint dict (``None`` when present but unmarked). A
    transient/auth ``IcechunkError`` re-raises rather than reporting "absent":
    conflating it with a missing repo would have the caller ingest over data it cannot
    read. Only a genuinely-missing repo reports ``(False, None)``.

    A ROOTLESS repo — created, then crashed before its schema was committed —
    reports ``(True, None)``: present and unmarked, so the caller RESUMES it (the child's
    own seed probe recreates the missing schema). It must be caught BEFORE
    ``FileNotFoundError``, which ``GroupNotFoundError`` subclasses; reporting it absent
    would make the caller treat the prefix as empty.
    """
    try:
        root = open_store_as_zarr_group(store_path, get_credentials=get_credentials, region=s3_region)
    except zarr.errors.GroupNotFoundError:
        return (True, None)
    except FileNotFoundError:
        return (False, None)
    except icechunk.IcechunkError as exc:
        if is_missing_repo(exc):
            return (False, None)
        raise
    raw = root.attrs.get("ingest_marker")
    return (True, dict(raw) if isinstance(raw, dict) else None)


def _assert_store_manifest_matches(
    store_path: str,
    roi_zarr_path: str,
    *,
    min_valid_coverage: float | None,
    get_credentials: _Creds,
    s3_region: str | None,
) -> None:
    """Refuse to mark a store whose existing manifest disagrees with this run.

    The ingest's own append path validates the manifest on every batch, which is what
    makes RESUMING an interrupted store safe: a changed mask, admission threshold or
    ingest code raises at the first write, before a date commits. A resume that appends
    NOTHING never reaches that check — and one exists, routinely. An attempt that wrote
    every date and crashed before its marker leaves a store the next run adopts as
    unmarked; the child legs then skip every date as already present, write nothing, and
    the marker is stamped over a mosaic built under whatever the previous run believed.
    After that the fingerprint matches and every later run skips the cell entirely.

    So the same check runs HERE too, on the zero-write path the append cannot cover. A
    store with no manifest at all is a legacy store and passes with a warning, exactly as
    it does on the append path — the two must not disagree about what is acceptable.
    """
    expected = IngestManifest.from_roi_store(
        roi_zarr_path,
        min_valid_coverage=min_valid_coverage,
        # The ROI is a plain zarr, so it does not travel on the Icechunk callback. Without
        # this the check itself fails on a callback-only deployment — leaving the stores
        # UNMARKED after a successful ingest, so every retry re-reads a finished mosaic as
        # an interrupted one.
        storage_options=plain_zarr_storage_options(roi_zarr_path, get_credentials, s3_region),
    )
    root = open_store_as_zarr_group(store_path, get_credentials=get_credentials, region=s3_region)
    expected.validate_against(extract_manifest(dict(root.attrs)), store_path)


def _write_ingest_marker(store_path: str, fingerprint: dict, *, get_credentials: _Creds, s3_region: str | None) -> None:
    """Stamp the ``ingest_marker`` fingerprint + ``ingest_completed_at`` on a store's root.

    OPENS, never creates. This runs only for a store that was just written, so creation is
    never the right answer, and the cost of getting that wrong is the whole ingest: a
    throttle or an expired credential mistaken for absence would leave a complete
    multi-terabyte store UNMARKED, and the next run would then re-query its whole window to
    rediscover that every date is already present. Surfacing the real error costs a retry.

    `open_or_create_repo` no longer conflates the two — it creates only on proven absence —
    but this call stays :func:`open_repo` regardless, because "must already exist" is both
    true and a stronger statement here than "create if needed".
    """
    repo = open_repo(store_path, get_credentials=get_credentials, region=s3_region)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    root.attrs["ingest_marker"] = dict(fingerprint)
    root.attrs["ingest_completed_at"] = utcnow_iso()
    session.commit(f"ingest marker: {fingerprint}")


# Chunk-scaled worker sizing for cropped ingests (upstream roi_fanout's pattern:
# workers proportional to the cell's work measure, clamped). One 4096-px ingest
# chunk is the unit of work once writes crop to live windows, so a 4-tile zone no
# longer gets the same fleet as a dense one. The floor keeps a tiny cell from
# starving on one worker; the caller's max_workers stays the hard cap (quota).
_WORKERS_PER_LIVE_CHUNK = 0.5
_WORKERS_FLOOR = 10


def _scaled_max_workers(live_chunks: int, settings: IngestSettings) -> int:
    """Clamp(0.5 x live chunks) into [max(min_workers, floor), max_workers]."""
    # `or 1` keeps the pre-sentinel meaning: min_workers used to default to 1, so this
    # max() was a no-op then and must stay one now. This is a floor on the derived WIDTH,
    # a different question from the fleet's adaptive minimum — see IngestSettings.floor_for.
    floor = max(settings.min_workers or 1, min(_WORKERS_FLOOR, settings.max_workers))
    return max(floor, min(settings.max_workers, round(live_chunks * _WORKERS_PER_LIVE_CHUNK)))


def _s1_max_workers(s2_max_workers: int, settings: IngestSettings) -> int:
    """One S1 orbit's fleet width, as a fraction of the S2 fleet's.

    A cell runs three fleets concurrently and lasts as long as its LONGEST, which is always
    S2 — so S1's width is chosen to finish inside S2's runtime, not to go fast. Giving S1
    the same width as S2 (which one shared ``max_workers`` did) left it idle for most of the
    cell while still holding quota, and quota is what caps how many cells run at once.

    Never wider than S2 and never below ``min_workers``: the fraction is calibrated against
    a 60-worker S2 fleet, and a sparse zone can scale S2 down far enough that the raw
    fraction would round below a single worker.
    """
    # S2 is the CEILING and is applied last. Clamping before the floor let
    # `min_workers` win outright: with min_workers=40 and a 13-worker S2 leg for a sparse
    # zone, this returned 40 — wider than S2, which is the one thing the docstring above
    # promises it never is. The floor still applies wherever S2 leaves room for it.
    return min(
        s2_max_workers,
        max(settings.min_workers or 1, round(s2_max_workers * settings.s1_worker_fraction)),
    )


#: Tag prefix for the S1/S2 ROI ingest runs this flow dispatches.
#: Markers in a failed leg's detail that mean a re-dispatch CANNOT help. Everything else
#: is retried, and that polarity is deliberate: a wasted retry costs one leg's remaining
#: work on a resumable ingest, while a missed retry leaves a mosaic incomplete until a
#: human notices.
#:
#: Keep this list to failures that are deterministic in the INPUT, not merely severe. A
#: crash, a throttle, an expired source credential and a warp error are all transient by
#: nature: the same dispatch can succeed next time because it resumes from the dates
#: already committed.
#:
#: One failure class is deterministic without being nameable here, because whether it is
#: deterministic is not a property of a single message: a source catalogue that refuses
#: the identical request the identical way on a later attempt. That is settled by
#: observing the repeat inside the attempt loop below, not by a marker.
_NON_RETRYABLE_LEG_MARKERS = (
    "InsufficientCoverageError",  # the source has no such data; asking again gets the same answer
    "ObjectNotFound",  # the child deployment is not registered — a registration bug, not a blip
    "ValidationError",  # malformed parameters
    "calendar-year",  # the year-window gate rejected the request itself
    "no live tiles",  # nothing to ingest in this zone
    # The store disagrees with this run about the mask, the admission threshold, or the
    # ingest code that produced its dates. Every retry re-reads the same store and gets
    # the same answer, and the resolution is a human deleting the interrupted store —
    # the same answer a mismatched completion marker gets, for the same reason.
    "ConfigMismatchError",
    # The mosaic prefix holds objects but no Icechunk repository, so `Repository.create`
    # refuses it under the clean-prefix rule. Deterministic in the input in the strictest
    # sense available: the prefix's contents decide it, and no attempt changes them. Added
    # 2026-08-18, when zone 47S spent its whole attempt budget re-reading the same twenty
    # orphaned chunk objects — three dispatches, three identical failures, an hour of a
    # cell's critical path, and nothing learned after the first. `open_or_create_repo`
    # creates ONLY on proven absence, so this error no longer has a transient form: a
    # momentary open failure is re-raised as itself rather than routed into the create leg.
    # `scripts/audit_mosaic_prefixes.py` in the consumer repo finds these before a dispatch.
    "CorruptedStoreError",
)


def _leg_failure_detail(run: object, label: str) -> str | None:
    """A one-line failure detail for a settled leg, or ``None`` if it completed.

    Prefers the child's own state MESSAGE over the state name, because the name
    ("FAILED") cannot be classified and the message carries the exception text that
    :data:`_NON_RETRYABLE_LEG_MARKERS` is matched against.
    """
    if isinstance(run, BaseException):
        return f"{label}: {run!r}"
    try:
        _check_completed(run, label)
    except Exception as exc:  # a returned-but-not-COMPLETED terminal state
        state = getattr(run, "state", None)
        message = getattr(state, "message", None) or ""
        return f"{exc}{f': {message}' if message else ''}"
    return None


def _is_retryable_leg_failure(detail: str) -> bool:
    """Whether re-dispatching this leg could plausibly succeed.

    Default TRUE. The ingest resumes — committed dates are skipped, not rewritten — so a
    retry is cheap and the failure modes actually observed are transient. Only a failure
    that is deterministic in the input is excluded.
    """
    lowered = detail.lower()
    return not any(marker.lower() in lowered for marker in _NON_RETRYABLE_LEG_MARKERS)


_CHILD_TAG_PREFIX = "ingest-zone-year"


#: The campaign cancels the runs it started, which reaches THIS flow — but its own S1/S2
#: children are separate deployment runs that nothing was cancelling. They would keep
#: their Dask fleets billing and keep writing into the mosaic prefix that a retry is
#: about to clear and rebuild, which is the one race the clear-and-rebuild recovery
#: cannot survive. Registered on both terminal hooks: a crashed parent orphans children
#: exactly as a cancelled one does.
_cancel_children_on_cancellation = make_child_cancel_hook(_CHILD_TAG_PREFIX, "child ingest run")


@flow(
    name="ingest-zone-year",
    on_cancellation=[_cancel_children_on_cancellation],
    on_crashed=[_cancel_children_on_cancellation],
)
async def ingest_zone_year(
    *,
    zone: str,
    year: int,
    paths: BucketPaths,
    deployments: IngestDeployments = IngestDeployments(),  # noqa: B008
    mask_name: str = "global",
    s1_orbit: str = "both",
    ingest_settings: IngestSettings = IngestSettings(),  # noqa: B008
    time_window_end: str | None = None,
    allow_partial_window: bool = False,
    s3_region: str | None = None,
    use_local: bool = False,
) -> dict[str, Any]:
    """Ingest the S1/S2 mosaics for one ``(zone, year)`` onto the zone grid.

    Args:
        zone: UTM common name (e.g. ``"33N"``); canonicalized on entry.
        year: Campaign calendar year (the default Dec-Y calendar-year window).
        paths: Deployment storage contract (mosaics, ROI zarrs, land mask).
        deployments: S1/S2 ingest deployment refs.
        mask_name: Coverage-store basename (``paths.land_mask_store``).
        s1_orbit: ``"ascending"``, ``"descending"``, or ``"both"``.
        ingest_settings: Grouped ingest tuning knobs (worker bounds, S2
            coverage threshold, S1 batch window) fanned out to the base
            S1/S2 ingest flows. See
            :class:`tessera_embeddings.config.ingest.IngestSettings`.
        time_window_end: ``"Month Year"`` override; defaults to ``"December {year}"``.
        allow_partial_window: Relax the coverage gate to "non-empty" (escape
            hatch for a legitimately partial edge zone).
        s3_region: Optional S3 region for this flow's Icechunk metadata opens
            (mask liveness, coverage sha, marker probe/write, coverage gate) and
            the zone-ROI synthesis — mirrors the campaign/fill region threading so
            a non-default-region deployment reads the same stores the fill will.
            The child S1/S2 ROI ingest deployments that write the mosaic go through
            the ROI engine's own storage path, which is us-west-2-only today (a
            pre-existing limitation independent of this flow; all campaign data
            lives in us-west-2 by the RTC-archive constraint).
        use_local: Run ingestion on a local Dask cluster (dev).

    Returns:
        ``{zone, year, status, ...}`` where status is ``skipped_ocean`` (no live
        tiles), ``already_ingested`` (marker matched), or ``ingested``.
    """
    log = get_run_logger()

    # Lazy AWS import so the flow file imports on non-AWS machines (arch tests), and
    # skipped entirely under `use_local`: the documented dev path runs against local
    # stores that need no IAM callback, and importing the provider anyway made it
    # require botocore — so an install without the optional `aws` extra could not reach
    # the local branch it was told to use. `None` is what every store open below means
    # by "no callback", so local mode simply passes that.
    iam_icechunk_credentials: _Creds = None
    if not use_local:
        from tessera_embeddings.providers.aws import credentials as _aws_credentials

        iam_icechunk_credentials = _aws_credentials.iam_icechunk_credentials

    # Validate the orbit FIRST. It is part of the ingest fingerprint, so a typo makes
    # every correctly-marked mosaic look stale — and the stale branch below deletes
    # multi-terabyte stores. `_active_orbits` would reject it, but only after that
    # deletion has already run.
    _active_orbits(s1_orbit)
    zone = canonicalize_zone(zone)
    land_mask_path = paths.land_mask_store(mask_name)
    mosaic_base = f"{paths.inputs.rstrip('/')}/mosaics/{zone}/{year}"
    roi_path = paths.zone_roi_store(zone)

    # (1) All-ocean zone: no ROI, no ingest — the fill marks it empty.
    if not zone_has_live_tiles(land_mask_path, zone, get_credentials=iam_icechunk_credentials, s3_region=s3_region):
        log.info("Zone %s has no live tiles (all-ocean) — skipping ingest", zone)
        return {"zone": zone, "year": year, "status": "skipped_ocean"}

    window = parse_time_window(time_window_end or f"December {year}")
    start_date, end_date = window.to_date_range()
    # Fingerprint the ingest INPUTS and the acceptance POLICY, not just the window:
    # rebuilding the coverage (new registry_sha256), changing min_valid_coverage, the
    # window, or the REQUESTED orbit set all change the mosaic that should be produced.
    # Including s1_orbit is what makes an ascending-only run's marker mismatch a later
    # "both" request (so the missing orbit is actually ingested, not skipped).
    # allow_partial_window is in the fingerprint because the marker short-circuits
    # BEFORE coverage validation: a mosaic accepted under the relaxed policy must NOT
    # satisfy a later strict run (its fill would then fail strict preflight forever) —
    # a strict run's differing fingerprint forces a re-ingest that re-runs the strict
    # coverage gate rather than silently reusing a partial mosaic.
    fingerprint = {
        "window": [start_date, end_date],
        "min_valid_coverage": ingest_settings.min_valid_coverage,
        "s1_orbit": s1_orbit,
        "allow_partial_window": allow_partial_window,
        "coverage_sha256": _coverage_sha(
            land_mask_path, zone, get_credentials=iam_icechunk_credentials, s3_region=s3_region
        ),
    }

    def _resolved_stores() -> list[str] | None:
        """Stores for the orbits actually present, or None if no SAR store exists yet.

        `s1_orbit="both"` downgrades to whatever ingested (an orbit with no
        granules for this zone/window writes no store), matching the fill's
        resolve_s1_orbit — so we never require a SAR store that will never exist.
        """
        try:
            effective = resolve_s1_orbit(
                mosaic_base,
                s1_orbit,
                # Only where "both" was a request rather than a demand. An operator who named
                # one orbit and got no store asked for something specific and must be told.
                allow_none=(s1_orbit == "both"),
                get_credentials=iam_icechunk_credentials,
                s3_region=s3_region,
            )
        except InsufficientCoverageError:
            return None
        return _mosaic_stores(mosaic_base, effective)

    # (2) Marker probe over the MAXIMAL candidate set (reflectance + BOTH SAR
    #     orbits), not just the resolved orbit set: a prior attempt that wrote one
    #     child store then crashed before stamping any marker leaves data the
    #     resolved-orbit probe would miss (with no SAR store, `_resolved_stores` is
    #     None), and appending onto it would dedupe against stale dates then stamp
    #     the new fingerprint over mixed inputs. Physical existence is the signal.
    candidates = _mosaic_stores(mosaic_base, "both")
    probed = {s: _probe_marker(s, get_credentials=iam_icechunk_credentials, s3_region=s3_region) for s in candidates}
    # An UNMARKED store is an interrupted attempt at THIS work, and is resumed. A store
    # marked with a DIFFERENT fingerprint is a violated assumption, and raises — see the
    # two branches below for why those answers are opposite.
    unmarked = [s for s, (exists, marker) in probed.items() if exists and marker is None]
    mismatched = {
        s: marker for s, (exists, marker) in probed.items() if exists and marker is not None and marker != fingerprint
    }
    # Resolve orbits ONLY when every present store is complete. `resolve_s1_orbit`
    # re-raises GroupNotFoundError deliberately — at fill time a rootless SAR store must
    # never be read as an absent orbit and quietly halve the radar. But an interrupted
    # store is exactly how one becomes rootless, and resolving first put that raise in
    # front of the resume. Skipping the call costs nothing: an incomplete store cannot
    # match the fingerprint and so could never have short-circuited anyway.
    resolved = None if (unmarked or mismatched) else _resolved_stores()
    if resolved is not None and all(probed[s][1] == fingerprint for s in resolved):
        log.info("Zone %s year %d already ingested for %s — skipping", zone, year, fingerprint["window"])
        return {"zone": zone, "year": year, "status": "already_ingested", "fingerprint": fingerprint}

    # A completed store whose fingerprint DISAGREES means one of the campaign's documented
    # invariants was broken: the window, the coverage gate, the orbit set, the
    # partial-window policy and the land mask are all fixed for a campaign. Neither
    # automatic answer is right. Resuming would append dates admitted under one
    # configuration onto dates admitted under another and then stamp one fingerprint over
    # the mixture; clearing would destroy a complete, correct mosaic because a parameter
    # was mistyped. So this raises and a human decides — the situation is rare by
    # construction and expensive to get wrong either way.
    if mismatched:
        detail = "; ".join(f"{s.rsplit('/', 1)[-1]}: stored {m}" for s, m in mismatched.items())
        raise ConfigMismatchError(
            f"Zone {zone} year {year}: {len(mismatched)} store(s) are COMPLETE under a different "
            f"fingerprint than this run requests. Wanted {fingerprint}. {detail}. A campaign holds "
            f"these inputs fixed, so this is a changed parameter rather than an interrupted run. "
            f"Delete the store(s) deliberately if the new inputs are intended, or correct the "
            f"parameters — this flow will not choose between discarding a finished mosaic and "
            f"mixing two configurations."
        )

    # RESUME, do not rebuild. An unmarked store holds dates this same work already
    # committed, and Icechunk commits a date's time slot atomically WITH its pixels, so a
    # date present is complete and a date absent was never started — there is no partial
    # date to repair. Both ingests already skip dates they find (`get_existing_dates`), so
    # simply not clearing is the whole mechanism.
    #
    # This replaced a clear-and-rebuild. At campaign scale that discarded everything a
    # cell had ingested for any interruption at all — and interruptions are expected, since
    # the orphan sweeper cancels runs by design. A dense zone interrupted near the end lost
    # hours to re-pay deterministically on every retry.
    #
    # TWO WRITERS is the hazard resume makes reachable, and Icechunk is what guards it:
    # these commits do NOT rebase, so a second writer's commit fails with ConflictError
    # against a moved branch tip rather than merging silently. That protection is a
    # property of NOT passing `rebase_with` — see `storage.shard_writer.commit_with_rebase`,
    # which deliberately does the opposite for disjoint groups. The ingest path must never
    # adopt it, and the write RETRY must never retry the resulting error: retrying re-opens
    # the session from the tip the other writer moved, which is `rebase_with` by another
    # route. `storage.zarr_store.store_write_retrying` excludes it by type.
    if unmarked:
        log.info(
            "Zone %s year %d: RESUMING %d interrupted store(s) — dates already committed are "
            "skipped, not rewritten: %s",
            zone,
            year,
            len(unmarked),
            ", ".join(s.rsplit("/", 1)[-1] for s in unmarked),
        )

    # (3) Ensure the zone ROI zarr (idempotent; regenerates if coverage changed).
    export_zone_roi(
        zone,
        land_mask_path=land_mask_path,
        dest_path=roi_path,
        get_credentials=iam_icechunk_credentials,
        s3_region=s3_region,
    )

    # (4) Size the fleet from the cell's LIVE work. Writes are restricted to live
    # windows, so the live chunk count IS the work measure — the 03S incident showed
    # extent-sized fleets are wrong by orders of magnitude.
    n_chunks = live_chunk_count(
        zone, land_mask_path=land_mask_path, get_credentials=iam_icechunk_credentials, s3_region=s3_region
    )
    max_workers = _scaled_max_workers(n_chunks, ingest_settings)
    log.info("Zone %s: %d live chunk(s) -> max_workers=%d", zone, n_chunks, max_workers)

    # Dispatch S1 (per REQUESTED orbit) + S2 ingestion concurrently onto the ROI.
    common: dict[str, Any] = {
        "roi_zarr_path": roi_path,
        "start_date": start_date,
        "end_date": end_date,
        "store_path": mosaic_base,
        "min_workers": ingest_settings.floor_for(max_workers),
        "max_workers": max_workers,
        "use_local": use_local,
        # The children open and CREATE the mosaic repos; without this they sign
        # against the storage default while this flow's own probes use s3_region,
        # so a non-default-region campaign fails inside the costly S1/S2 jobs
        # rather than at preflight.
        "s3_region": s3_region,
    }
    orbits = _active_orbits(s1_orbit)
    # Optional perf-report capture: the setting is a base URI. Scope it by CELL
    # first, then by child. run-global-campaign hands the SAME IngestSettings to
    # every (zone, year), so a base-only path would have concurrent cells racing
    # on one s2.html and later cells overwriting earlier ones; the per-child
    # suffix then separates the S1 orbits from S2 within a cell.
    perf_cell = ingest_settings.perf_report_uri
    perf_base = f"{perf_cell.rstrip('/')}/{zone}-{year}" if perf_cell else None
    # S1 runs NARROWER than S2, overriding the shared width in `common`. Its work is a
    # fixed fraction of S2's, so an equal fleet finishes early and then holds quota it
    # cannot use — and quota is what limits concurrent cells, hence the campaign's schedule.
    s1_workers = _s1_max_workers(max_workers, ingest_settings)
    log.info("Zone %s: S2 max_workers=%d, each S1 orbit max_workers=%d", zone, max_workers, s1_workers)
    # Stamped on every child so the terminal hooks can find them from this flow-run id
    # alone (see _CHILD_TAG_PREFIX). None outside a Prefect run — a direct .fn() call in
    # tests dispatches nothing worth sweeping.
    child_tags = [t] if (t := child_run_tag(_CHILD_TAG_PREFIX, flow_run_ctx.id)) else None
    # One (label, deployment, parameters) per leg, so a failed leg can be re-dispatched
    # verbatim rather than reconstructed.
    legs: list[tuple[str, str, dict]] = [
        (
            f"ingest_s1_roi_sar ({orbit})",
            deployments.ingest_s1_roi_sar,
            {
                **common,
                "orbit": orbit,
                "batch_days": ingest_settings.batch_days,
                # Its OWN floor: `common` resolved against the S2 width, and an S1 orbit is a
                # fraction of it, so inheriting that would request more workers than this leg
                # was sized for and the fleet could never reach its minimum.
                "min_workers": ingest_settings.floor_for(s1_workers),
                "max_workers": s1_workers,
                "perf_report_uri": f"{perf_base}/s1-{orbit}.html" if perf_base else None,
            },
        )
        for orbit in orbits
    ] + [
        (
            "ingest_s2_roi_reflectance",
            deployments.ingest_s2_roi_reflectance,
            {
                **common,
                "min_valid_coverage": ingest_settings.min_valid_coverage,
                "perf_report_uri": f"{perf_base}/s2.html" if perf_base else None,
            },
        )
    ]

    # ONE RETRY LOOP PER LEG, joined only at the end — the legs no longer wait for each other.
    #
    # This was a barrier: every leg was joined, then the failed ones re-dispatched together. Its
    # stated reason was that a re-dispatch could clear the mosaic prefix while a sibling was still
    # writing into it. That reason is gone. The clear-and-rebuild recovery it guarded was replaced
    # by resume (see "RESUME, do not rebuild" above), so a re-dispatch clears nothing, and each leg
    # writes its OWN child store under the base regardless.
    #
    # What the barrier still cost was latency on the cell's critical path, which is the whole
    # point of it: on 2026-08-18 zone 47S's descending leg failed one minute in and was not
    # re-dispatched for an hour, because it waited on an optical leg that had another hour to run.
    # A cell cannot infer until every leg has landed, so that hour was added for nothing.
    #
    # Two things the barrier did without saying so, now explicit rather than accidental: attempts
    # are spaced by `leg_retry_backoff_s` rather than by however long the slowest sibling happened
    # to take, and the wall-clock budget is per leg, anchored at that leg's own first dispatch.
    #
    # THE INVARIANT THAT SURVIVES UNCHANGED, and it is the important one: a leg that can never
    # succeed must fail the cell, and a sibling's later success must not erase it. `s1_orbit="both"`
    # with one SAR deployment permanently broken must not resolve to the single orbit that did
    # ingest, pass the coverage gate on it, and stamp a "both" marker over half the radar — after
    # which every later run reads the marker and skips the cell. The old loop protected that by
    # breaking before `errors` could be rebuilt on the next pass; here each leg owns its own result
    # and nothing can clear another's, so the property is structural rather than a matter of order.
    #
    # A terminally failed leg does NOT cancel its siblings. Their committed dates are useful to the
    # next dispatch either way, and interrupting a running write is how a store becomes an
    # interrupted one — the failure is reported as soon as they settle, and the cell returns to the
    # campaign work list to resume.
    #
    # It DOES stop them spending further attempts, which the barrier used to do by aborting the
    # whole loop. Once any leg has failed in a way no re-dispatch can fix, the cell cannot succeed,
    # so a sibling's remaining retries would buy nothing and each one holds a Dask fleet while it
    # runs. `doomed` gates only the decision to START another attempt — the same thing the
    # wall-clock bound gates, and for the same reason: a leg already running is never interrupted.
    doomed = asyncio.Event()

    def _leg_backoff_s(attempt: int) -> float:
        """Delay before a leg's next attempt: doubling from the base, capped at four times it.

        Deliberately short. The ingest resumes, so a retry costs only the work actually lost, and
        the campaign's throughput is bounded by legs landing — a long backoff idles a GPU fleet
        waiting on a mosaic. What the delay buys is the difference between a momentary source
        refusal and a structural one, and that does not take minutes to establish.
        """
        base = ingest_settings.leg_retry_backoff_s
        return min(base * (2 ** (attempt - 1)), base * 4)

    async def _run_leg(label: str, dep: str, params: dict) -> str | None:
        """Dispatch one leg, retrying on its own schedule; its failure detail, or None if it ran.

        Every classification here is the one the joined loop applied — a deterministic repeat from
        the source catalogue, a marker meaning a re-dispatch cannot help, a load refusal that is
        evidence FOR patience — read per leg instead of per round.
        """
        started = monotonic()
        last_token: str | None = None
        detail: str | None = None
        for attempt in range(1, ingest_settings.max_leg_attempts + 1):
            try:
                run: object = await arun_deployment(dep, parameters=params, tags=child_tags)
            except Exception as exc:
                # `CancelledError` is deliberately NOT caught: cancelling this flow must cancel
                # its legs rather than be recorded as one leg's failure detail.
                run = exc
            detail = _leg_failure_detail(run, label)
            if detail is None:
                return None
            refusal = refusal_in(detail)
            previous = last_token
            if refusal is not None:
                last_token = refusal.token
            if refusal is not None and refusal.token == previous and repeat_is_deterministic(refusal.kind):
                log.error(
                    "Zone %s year %s: %s — the source catalogue refused the SAME request the SAME "
                    "way again (%s). That is deterministic in the request, not congestion: the "
                    "remaining attempts cannot succeed and would only spend this cell's budget "
                    "more slowly. Report the request to the archive's operator — a reliably "
                    "failing query is as actionable for them as a coverage gap. Detail: %s",
                    zone,
                    year,
                    label,
                    refusal.token,
                    detail,
                )
                doomed.set()
                return detail
            if not _is_retryable_leg_failure(detail):
                log.error(
                    "Zone %s year %s: %s failed in a way a re-dispatch cannot fix — not retrying: %s",
                    zone,
                    year,
                    label,
                    detail,
                )
                doomed.set()
                return detail
            if refusal is not None and refusal.kind is RefusalKind.LOAD:
                # Said out loud because it is the branch that deliberately keeps waiting: a source
                # naming itself as the constraint is exactly what the expansive retry exists for,
                # and a repeat of it is evidence FOR patience rather than against it.
                log.warning(
                    "Zone %s year %s: %s — the source catalogue refused under load (%s). "
                    "Retrying: waiting is the remedy for this class, however often it recurs.",
                    zone,
                    year,
                    label,
                    refusal.token,
                )
            if attempt == ingest_settings.max_leg_attempts:
                return detail
            if doomed.is_set():
                # Another leg has failed in a way no re-dispatch can fix, so this cell cannot
                # succeed however well this leg's next attempt goes. Its own failure is still
                # what gets reported for this leg — the cell fails on the union of them.
                log.warning(
                    "Zone %s year %s: %s will not be re-dispatched — another leg of this cell "
                    "failed terminally, so the cell cannot succeed and a further attempt would "
                    "hold a fleet to learn nothing. Detail: %s",
                    zone,
                    year,
                    label,
                    detail,
                )
                return detail
            # The wall-clock bound, checked ONLY here — at the decision to start another attempt.
            # A leg that is RUNNING is never measured against it, so a slow-but-succeeding leg
            # cannot be why this stopped; what the bound refuses is re-dispatching after patience
            # has already had a deadline's worth of wall clock. Per leg now, because the legs no
            # longer share a loop: one leg's slow retries must not spend another's budget.
            elapsed = monotonic() - started
            if elapsed >= ingest_settings.max_leg_wall_clock_s:
                log.error(
                    "Zone %s year %s: %s — %.0f s elapsed since this leg's first attempt, past the "
                    "wall-clock budget for starting another (max_leg_wall_clock_s=%d), so attempt "
                    "%d/%d is refused even though the attempt budget has room. No running leg was "
                    "interrupted. The cell fails back to the campaign work list and a later "
                    "dispatch RESUMES from the dates already committed — this costs latency, not "
                    "work. Detail: %s",
                    zone,
                    year,
                    label,
                    elapsed,
                    ingest_settings.max_leg_wall_clock_s,
                    attempt + 1,
                    ingest_settings.max_leg_attempts,
                    detail,
                )
                return detail
            wait = _leg_backoff_s(attempt)
            # A re-dispatch RESUMES: already-committed dates are skipped, not rewritten, so the
            # retry costs only the work that was actually lost. That idempotency is what makes
            # retrying the default rather than the exception.
            log.warning(
                "Zone %s year %s: %s failed attempt %d/%d — re-dispatching in %.0f s (it resumes "
                "from the dates already committed): %s",
                zone,
                year,
                label,
                attempt,
                ingest_settings.max_leg_attempts,
                wait,
                detail,
            )
            await asyncio.sleep(wait)
            if doomed.is_set():
                # Re-checked after the wait: a sibling can reach its terminal failure while this
                # leg is backing off, and the point of the gate is to not start the attempt.
                log.warning(
                    "Zone %s year %s: %s will not be re-dispatched — another leg failed terminally "
                    "while this one was waiting to retry. Detail: %s",
                    zone,
                    year,
                    label,
                    detail,
                )
                return detail
        return detail

    settled = await asyncio.gather(*(_run_leg(label, dep, params) for label, dep, params in legs))
    errors: list[str] = [detail for detail in settled if detail is not None]

    if errors:
        raise RuntimeError(f"ingest deployment(s) failed for zone {zone} year {year}: " + "; ".join(errors))

    # (5) Resolve the orbit set from what actually ingested, then verify + mark
    #     only those stores. `s1_orbit="both"` with one empty orbit downgrades
    #     here rather than failing the coverage check on a store that never exists.
    stores = _resolved_stores()
    if stores is None:
        msg = f"s1_orbit={s1_orbit!r} but no SAR store was produced for zone {zone} year {year}"
        raise InsufficientCoverageError(msg)
    # Derived from the COUNT of resolved stores, so the radar-free case is not mistaken for a
    # single-orbit one: with reflectance alone, `stores[-1]` is the reflectance store and the
    # old rsplit produced "reflectance" as an orbit name.
    if len(stores) == 3:
        effective_orbit = "both"
    elif len(stores) == 1:
        effective_orbit = S1_ORBIT_NONE
        log.warning(
            "Zone %s year %s has NO radar store — proceeding optical-only. Legitimate where the "
            "ROI has no dual-pol VV+VH coverage (ice is imaged HH/HV in Extra Wide mode); the "
            "S1 legs' per-batch item counts say which case this is.",
            zone,
            year,
        )
    else:
        effective_orbit = stores[-1].rsplit("sar_", 1)[-1].removesuffix(".zarr")
    check_time_window_coverage(
        mosaic_base,
        window,
        s1_orbit=effective_orbit,
        skip_coverage_check=allow_partial_window,
        get_credentials=iam_icechunk_credentials,
        s3_region=s3_region,
    )

    # (6) Marker last: a crash before this point re-runs incrementally.
    #
    # Validated first, and in a separate pass: a store the legs never appended to has had
    # no manifest check at all this run, and marking a mosaic built under different inputs
    # makes every later run skip it. Two passes rather than one so a disagreement on the
    # LAST store cannot leave the earlier ones already marked.
    for store in stores:
        _assert_store_manifest_matches(
            store,
            roi_path,
            # Optical only. The SAR legs apply no admission threshold, so their manifests
            # carry none and expecting one would fail every radar store.
            min_valid_coverage=ingest_settings.min_valid_coverage if store.endswith("reflectance.zarr") else None,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )
    for store in stores:
        _write_ingest_marker(store, fingerprint, get_credentials=iam_icechunk_credentials, s3_region=s3_region)

    log.info("Zone %s year %d ingested (orbit=%s, %s)", zone, year, effective_orbit, fingerprint["window"])
    return {"zone": zone, "year": year, "status": "ingested", "fingerprint": fingerprint, "stores": stores}
