"""Ingest one ``(zone, year)`` of the campaign onto the fixed zone grid.

The global campaign reads its per-zone mosaics from ``{inputs}/mosaics/{zone}/{year}``; this
flow produces them by **reusing the ROI ingest engine unchanged**. A zone-shaped ROI zarr
(:func:`tessera_embeddings.ingest.land_mask.export_zone_roi`) pins the grid to the fixed
:class:`~tessera_embeddings.storage.zone_grid.ZoneSpec` extent — so the mosaic passes the
zone-fill's exact-grid validation, which ``generate_roi``'s bbox-fit grid could not — and
the S1/S2 ROI ingest deployments write onto it. Only orchestration lives here
(``arun_deployment``); each ingest deployment provisions its own Dask cluster.

**Idempotent, and RESUMABLE.** A per-store completion marker (root attr
``ingest_marker``, fingerprinting window + min_valid_coverage + s1_orbit +
allow_partial_window + coverage sha) gates the work: a matching marker on every required
store short-circuits, and the marker is written only after coverage is verified.

Three states, three answers:

* **absent** — ingest it.
* **present and unmarked** — an interrupted attempt at this same work. **RESUMED.** Dates
  already committed are skipped rather than rewritten: Icechunk commits a date's time slot
  atomically WITH its pixels, so a date present is complete and a date absent was never
  started. Both ingests already skip what they find, so resuming is simply not clearing.
* **present and marked DIFFERENTLY** — raises ``ConfigMismatchError``. See the branch itself
  for why neither automatic answer is defensible.

**Campaign assumption this rests on (documented, not enforced here).** A campaign holds its
inputs fixed: one land mask, one date window per year, one coverage threshold, both orbits,
one partial-window policy. So an unmarked store can only be an interruption of the same work.
The one input whose change would silently corrupt rather than raise is the LAND MASK, and
that is enforced independently — ``IngestManifest.coverage_sha256`` is validated on every
write, so a changed mask fails at the first append.

**Two writers** is the hazard resume makes reachable, and Icechunk guards it: these commits
do not rebase, so a second writer's commit fails with ``ConflictError`` against a moved
branch tip instead of merging silently. **The write retry must exclude that error for the
guard to hold** — retrying re-opens the session from the moved tip, converting the refusal
into a success; see ``storage.zarr_store.store_write_retrying``. Nothing here prevents a
second run from starting, so the outcome is a wasted cell rather than a corrupt store: it
fails, and a later attempt resumes from what the surviving writer committed. The
dispatch-side guard against launching the duplicate lives in
``scripts/run_campaign_cell.py`` in yield-embeddings, deliberately not here — a refusal in
this flow would strand a crashed run that Prefect still reports as ``RUNNING``.

**Coverage gate (ADR-011).** After ingestion, :func:`check_time_window_coverage` requires the
mosaics' months to span the window; ``allow_partial_window`` relaxes that to "non-empty" for
a legitimately partial edge zone. This is the ingest-side guard; the fill re-checks before
provisioning Ray.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from hashlib import sha256
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
from tessera_embeddings.errors import ConfigMismatchError, InsufficientCoverageError, ProviderRefusedReadsError
from tessera_embeddings.inference.data_loading import (
    S1_ORBIT_NONE,
    _active_orbits,
    check_time_window_coverage,
    resolve_s1_orbit,
)
from tessera_embeddings.ingest.catalogue_refusal import RefusalKind, refusal_in, repeat_is_deterministic
from tessera_embeddings.ingest.duplicates import unreadable_source_in
from tessera_embeddings.ingest.land_mask import export_zone_roi, live_chunk_count
from tessera_embeddings.orchestration.prefect.flows._child_runs import (
    _check_completed,
    child_run_tag,
    make_child_cancel_hook,
)
from tessera_embeddings.orchestration.runners.zone_fill import zone_has_live_tiles
from tessera_embeddings.storage.manifest import IngestManifest, extract_manifest
from tessera_embeddings.storage.zarr_store import (
    get_existing_dates,
    is_missing_repo,
    open_repo,
    open_store_as_zarr_group,
    plain_zarr_storage_options,
)
from tessera_embeddings.storage.zone_grid import canonicalize_zone
from tessera_embeddings.utils import utcnow_iso


class IngestDeployments(BaseModel):
    """Deployment refs (``flow_name/deployment_name``) the campaign dispatches to."""

    ingest_s1_roi_sar: str = "ingest_s1_roi_sar/ingest-s1-roi-sar"
    ingest_s2_roi_reflectance: str = "ingest_s2_roi_reflectance/ingest-s2-roi-reflectance"


def _leg_store(mosaic_base: str, orbit: str | None) -> str:
    """The child store ONE leg writes: a radar orbit's, or — for ``None`` — the optical one.

    One place names them: the set a fill reads and the store a leg's progress is read from
    must be the same object.
    """
    return f"{mosaic_base}/sar_{orbit}.zarr" if orbit else f"{mosaic_base}/reflectance.zarr"


def _mosaic_stores(mosaic_base: str, s1_orbit: str) -> list[str]:
    """The child stores a fill reads under ``mosaic_base`` for ``s1_orbit``."""
    stores = [_leg_store(mosaic_base, None)]
    stores.extend(_leg_store(mosaic_base, orbit) for orbit in _active_orbits(s1_orbit))
    return stores


_Creds = Callable[[], "icechunk.S3StaticCredentials"] | None


def _coverage_sha(land_mask_path: str, zone: str, *, get_credentials: _Creds, s3_region: str | None) -> str | None:
    """The coverage delivery sha (``registry_sha256``) for ``zone``, or ``None``."""
    cov = open_store_as_zarr_group(land_mask_path, group=zone, get_credentials=get_credentials, region=s3_region)
    return cast("str | None", cov.attrs.get("registry_sha256"))


def _probe_marker(store_path: str, *, get_credentials: _Creds, s3_region: str | None) -> tuple[bool, dict | None]:
    """``(exists, marker)`` for a mosaic child store.

    ``exists`` is whether the repo is physically present; ``marker`` is its ``ingest_marker``
    fingerprint dict (``None`` when present but unmarked). A transient/auth
    ``IcechunkError`` re-raises rather than reporting "absent", which would have the caller
    ingest over data it cannot read. Only a genuinely-missing repo reports ``(False, None)``.

    A ROOTLESS repo — created, then crashed before its schema was committed — reports
    ``(True, None)``: present and unmarked, so the caller RESUMES it and the child's own seed
    probe recreates the schema. Catch it BEFORE ``FileNotFoundError``, which
    ``GroupNotFoundError`` subclasses; reporting it absent would read the prefix as empty.
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


def _committed_date_count(
    store_path: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    get_credentials: _Creds,
    s3_region: str | None,
) -> int | None:
    """How many dates a leg's store holds, or ``None`` when that cannot be established.

    The parent's only view of whether a leg is getting anywhere. Read from the STORE, not
    inferred from the failure detail: the store is ground truth for every failure class,
    including a leg that crashed with no message, and a leg dies naming a date that FAILED,
    which is not a date it committed. Deliberately the ingest's own reader, so a count taken
    here and one taken by the leg cannot disagree.

    ``None`` rather than an exception: this reading is not worth failing a leg over, and it is
    the same answer as no progress, leaving the deadline exactly where it was.
    """
    try:
        return len(get_existing_dates(store_path, get_credentials=get_credentials, s3_region=s3_region))
    # Broad on purpose: every way a store read can fail — auth, throttling, a decode error, a
    # repo mid-creation — means the same thing here, that this leg's progress is unknown.
    except Exception as exc:
        log.warning(
            "Could not read %s to see whether its leg is still committing dates, so no progress "
            "can be credited to it: %r",
            store_path,
            exc,
        )
        return None


def _assert_store_manifest_matches(
    store_path: str,
    roi_zarr_path: str,
    *,
    min_valid_coverage: float | None,
    allow_ingest_code_mismatch: bool,
    get_credentials: _Creds,
    s3_region: str | None,
) -> None:
    """Refuse to mark a store whose existing manifest disagrees with this run.

    The ingest's append path validates the manifest on every batch, which is what makes
    RESUMING safe: a changed mask, admission threshold or ingest code raises at the first
    write. But a resume that appends NOTHING never reaches that check, and one occurs
    routinely — an attempt that wrote every date then crashed before its marker leaves a
    store the next run adopts as unmarked; the legs skip every date as already present, write
    nothing, and the marker is stamped over a mosaic built under whatever the previous run
    believed. After that the fingerprint matches and every later run skips the cell.

    So the same check runs HERE, on the zero-write path. A store with no manifest at all is a
    legacy store and passes with a warning, exactly as on the append path — the two must not
    disagree about what is acceptable. Same for ``allow_ingest_code_mismatch``: an override
    the appends were given but this check was not would refuse to mark the store it wrote.
    """
    expected = IngestManifest.from_roi_store(
        roi_zarr_path,
        min_valid_coverage=min_valid_coverage,
        allow_ingest_code_mismatch=allow_ingest_code_mismatch,
        # The ROI is a plain zarr and does not travel on the Icechunk callback. Without this
        # the check fails on a callback-only deployment, leaving the stores UNMARKED after a
        # successful ingest so every retry re-reads a finished mosaic as an interrupted one.
        storage_options=plain_zarr_storage_options(roi_zarr_path, get_credentials, s3_region),
    )
    root = open_store_as_zarr_group(store_path, get_credentials=get_credentials, region=s3_region)
    expected.validate_against(extract_manifest(dict(root.attrs)), store_path)


def _write_ingest_marker(store_path: str, fingerprint: dict, *, get_credentials: _Creds, s3_region: str | None) -> None:
    """Stamp the ``ingest_marker`` fingerprint + ``ingest_completed_at`` on a store's root.

    OPENS, never creates — :func:`open_repo`, because "must already exist" is true here and a
    stronger statement than "create if needed". This runs only for a store just written, and
    the cost of treating a throttle or an expired credential as absence is the whole ingest:
    a complete multi-terabyte store left UNMARKED, whose next run re-queries the entire window
    to rediscover that every date is present. Surfacing the real error costs a retry.
    """
    repo = open_repo(store_path, get_credentials=get_credentials, region=s3_region)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    root.attrs["ingest_marker"] = dict(fingerprint)
    root.attrs["ingest_completed_at"] = utcnow_iso()
    session.commit(f"ingest marker: {fingerprint}")


# Chunk-scaled worker sizing for cropped ingests: workers proportional to the cell's work
# measure, clamped. One 4096-px ingest chunk is the unit of work once writes crop to live
# windows, so a 4-tile zone does not get a dense zone's fleet. The floor keeps a tiny cell off
# a single worker; the caller's max_workers stays the hard cap (quota).
_WORKERS_PER_LIVE_CHUNK = 0.5
_WORKERS_FLOOR = 10


def _scaled_max_workers(live_chunks: int, settings: IngestSettings) -> int:
    """Clamp(0.5 x live chunks) into [max(min_workers, floor), max_workers]."""
    # `or 1` reads the sentinel as 1, keeping this max() a no-op for an unset min_workers.
    # It floors the derived WIDTH — a different question from the fleet's adaptive minimum;
    # see IngestSettings.floor_for.
    floor = max(settings.min_workers or 1, min(_WORKERS_FLOOR, settings.max_workers))
    return max(floor, min(settings.max_workers, round(live_chunks * _WORKERS_PER_LIVE_CHUNK)))


def _s1_max_workers(s2_max_workers: int, settings: IngestSettings) -> int:
    """One S1 orbit's fleet width, as a fraction of the S2 fleet's.

    A cell runs three fleets concurrently and lasts as long as its LONGEST, which is always
    S2 — so S1's width is chosen to finish inside S2's runtime, not to go fast. An S1 leg as
    wide as S2 sits idle for most of the cell while still holding quota, and quota is what
    caps how many cells run at once.

    Never wider than S2 and never below ``min_workers``: the fraction is calibrated against a
    60-worker S2 fleet, and a sparse zone can scale S2 down far enough that the raw fraction
    would round below a single worker.
    """
    # S2 is the CEILING and is applied LAST. Clamping before the floor lets `min_workers` win
    # outright — min_workers=40 against a 13-worker S2 leg returns 40, wider than S2, the one
    # thing this promises never happens. The floor still applies wherever S2 leaves room.
    return min(
        s2_max_workers,
        max(settings.min_workers or 1, round(s2_max_workers * settings.s1_worker_fraction)),
    )


#: Markers in a failed leg's detail that mean a re-dispatch CANNOT help. Everything else is
#: retried, and that polarity is deliberate: a wasted retry costs one leg's remaining work on
#: a resumable ingest, while a missed retry leaves a mosaic incomplete until a human notices.
#:
#: Keep this list to failures deterministic in the INPUT, not merely severe. A crash, a
#: throttle, an expired source credential and a warp error are all transient: the same
#: dispatch can succeed next time because it resumes from the dates already committed.
#:
#: One deterministic class is not nameable here, because determinism is not a property of a
#: single message: a source catalogue refusing the identical request the identical way on a
#: later attempt. That is settled by observing the repeat inside the attempt loop below.
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
    # refuses it under the clean-prefix rule. Deterministic in the strictest sense: the
    # prefix's contents decide it and no attempt changes them. Zone 47S once spent its whole
    # attempt budget re-reading the same twenty orphaned chunk objects — three dispatches,
    # three identical failures, an hour of a cell's critical path (2026-08-18).
    # `open_or_create_repo` creates ONLY on proven absence, so a momentary open failure is
    # re-raised as itself rather than routed into the create leg, and this error has no
    # transient form. `scripts/audit_mosaic_prefixes.py` in the consumer repo finds these
    # before a dispatch.
    "CorruptedStoreError",
    # A radar leg past `MAX_GIVEN_UP_DATES`. A provider refusal is not a reason to give up a
    # date, so every date counted is one whose bytes will not read: the re-dispatch re-reads
    # the same objects, spends the per-read retry ladder on each, and holds a fleet to reach
    # the identical answer.
    "TooManyGivenUpDatesError",
    # The leg offered the store a date OLDER than its time axis already holds. The axis is
    # append-only, so the identical date is refused identically every attempt and the only
    # remedy is a human deleting the store — a re-dispatch provisions and tears down a whole
    # Dask fleet to reach the same date and die there. Deterministic in the strictest sense:
    # the store's own contents decide it, and no attempt moves them.
    "NonMonotonicDateError",
)


def _leg_failure_detail(run: object, label: str) -> str | None:
    """A one-line failure detail for a settled leg, or ``None`` if it completed.

    Prefers the child's own state MESSAGE over the state name: "FAILED" cannot be classified,
    while the message carries the exception text :data:`_NON_RETRYABLE_LEG_MARKERS` matches.

    **What crosses the deployment boundary is a STRING, and only a string.** A leg is a
    separate deployment run read back over the API, so ``isinstance`` is unavailable at any
    price — Prefect attaches the exception to the state for in-process use only, and what the
    API stores is ``f"{type(exc).__name__}: {exc}"`` behind a fixed prefix. No traceback and
    no ``__cause__`` chain: the OUTERMOST exception is the whole of it.

    So classification matches the exception's class NAME, which survives intact, and never a
    field position or a formatted number, which do not. The coupling is to Prefect's message
    format; the class name sits immediately after its prefix, so a reworded prefix cannot
    move it.
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


def _leg_stagger_s(zone: str, year: int, label: str, window_s: int) -> float:
    """Seconds this leg delays its FIRST dispatch by, to decorrelate the fleet's cold start.

    Derived from the leg's identity rather than drawn at random, so the delay is stable across
    runs and a test can assert it. ``hash`` will not do — it is salted per process, so two
    workers would not agree on the spread. Spread across the whole window rather than by a
    fixed step because a leg cannot know how many others are running; identical identities
    collide, but a collision leaves two legs in phase, not the whole fleet.

    Returns 0 for a window of 0, which is how the stagger is turned off.
    """
    if window_s <= 0:
        return 0.0
    digest = sha256(f"{zone}/{year}/{label}".encode()).digest()
    return window_s * int.from_bytes(digest[:8], "big") / 2**64


def _is_retryable_leg_failure(detail: str) -> bool:
    """Whether re-dispatching this leg could plausibly succeed.

    **Default TRUE, and the polarity is load-bearing.** The ingest resumes — committed dates
    are skipped, not rewritten — so retrying a permanent failure costs one leg's fleet, while
    declining to retry a genuine transient strands the cell until a human notices. Only a
    POSITIVELY IDENTIFIED permanent verdict returns ``False``; anything unrecognised retries.

    Two independent grounds for that verdict, and a failure matching neither is retried:

    * a marker in :data:`_NON_RETRYABLE_LEG_MARKERS`, naming a class whose answer is fixed by
      the input;
    * the read-failure taxonomy saying the DATA is gone — reused rather than restated, so
      "these bytes will not read" means one thing repo-wide. It reports ``True`` for exactly
      ``UNREADABLE`` and ``ABSENT``. A provider refusal, our own expired credential and a
      chain carrying no evidence stay retryable, which is what the ladder and its long
      refusal backoff exist for.
    """
    lowered = detail.lower()
    if any(marker.lower() in lowered for marker in _NON_RETRYABLE_LEG_MARKERS):
        return False
    return not unreadable_source_in(detail)


#: Tag prefix for the S1/S2 ROI ingest runs this flow dispatches.
_CHILD_TAG_PREFIX = "ingest-zone-year"


#: The campaign cancels the runs it started, which reaches THIS flow — but its S1/S2 children
#: are separate deployment runs that nothing else cancels. Left alone they keep their Dask
#: fleets billing and keep writing into a mosaic prefix a retry is about to touch. Registered
#: on both terminal hooks: a crashed parent orphans children exactly as a cancelled one does.
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
    allow_ingest_code_mismatch: bool = False,
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
        ingest_settings: Grouped ingest tuning knobs (worker bounds, S2 coverage threshold,
            S1 batch window) fanned out to the base S1/S2 ingest flows. See
            :class:`tessera_embeddings.config.ingest.IngestSettings`.
        time_window_end: ``"Month Year"`` override; defaults to ``"December {year}"``.
        allow_partial_window: Relax the coverage gate to "non-empty" (escape hatch for a
            legitimately partial edge zone).
        allow_ingest_code_mismatch: Resume interrupted stores built by different ingest code.
            Deliberately NOT in the fingerprint below — a mosaic completed under it must
            still satisfy a later strict run.

            Note what that does NOT mean. The completion marker carries neither the code
            identity nor this flag, so a later strict run matches the same fingerprint and
            returns ``already_ingested`` from the marker fast path without reaching manifest
            validation — accepting the mixed mosaic rather than refusing it. Intended: an
            operator chose the mixture knowingly, and a store that may never be embedded is
            not worth re-ingesting. The store records which identities contributed under
            ``MIXED_CODE_IDENTITIES_ATTR`` in storage/manifest.py.

            A permanent, manually-invoked capability rather than a one-off, so treat it as
            part of the interface: off by default, chosen per run, never reached
            automatically.
        s3_region: Optional S3 region for this flow's Icechunk metadata opens (mask liveness,
            coverage sha, marker probe/write, coverage gate) and the zone-ROI synthesis,
            mirroring the campaign/fill region threading so a non-default-region deployment
            reads the same stores the fill will. The child S1/S2 ROI ingest deployments that
            write the mosaic go through the ROI engine's own storage path, which is
            us-west-2-only today — a pre-existing limitation independent of this flow, and
            all campaign data lives in us-west-2 by the RTC-archive constraint.
        use_local: Run ingestion on a local Dask cluster (dev).

    Returns:
        ``{zone, year, status, ...}`` where status is ``skipped_ocean`` (no live
        tiles), ``already_ingested`` (marker matched), or ``ingested``.
    """
    log = get_run_logger()

    # Lazy AWS import so the flow file imports on non-AWS machines (arch tests), and skipped
    # entirely under `use_local`: the dev path runs against local stores that need no IAM
    # callback, and importing the provider anyway would require botocore, so an install
    # without the optional `aws` extra could not reach the local branch it was told to use.
    # `None` is what every store open below means by "no callback".
    iam_icechunk_credentials: _Creds = None
    if not use_local:
        from tessera_embeddings.providers.aws import credentials as _aws_credentials

        iam_icechunk_credentials = _aws_credentials.iam_icechunk_credentials

    # Validate the orbit FIRST. It is part of the ingest fingerprint, so a typo makes every
    # correctly-marked mosaic look stale — and the stale branch below deletes multi-terabyte
    # stores. `_active_orbits` would reject it, but only after that deletion has run.
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
    # Fingerprint the ingest INPUTS and the acceptance POLICY, not just the window: rebuilding
    # the coverage (new registry_sha256), changing min_valid_coverage, the window, or the
    # REQUESTED orbit set all change the mosaic that should be produced. Including s1_orbit is
    # what makes an ascending-only run's marker mismatch a later "both" request, so the
    # missing orbit is actually ingested. allow_partial_window is in the fingerprint because
    # the marker short-circuits BEFORE coverage validation: a mosaic accepted under the
    # relaxed policy must NOT satisfy a later strict run, whose fill would then fail strict
    # preflight forever — a differing fingerprint forces a re-ingest that re-runs the strict
    # gate rather than silently reusing a partial mosaic.
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

        `s1_orbit="both"` downgrades to whatever ingested (an orbit with no granules for this
        zone/window writes no store), matching the fill's resolve_s1_orbit, so a SAR store
        that will never exist is never required.
        """
        try:
            effective = resolve_s1_orbit(
                mosaic_base,
                s1_orbit,
                # Only where "both" was a request rather than a demand: an operator who named
                # one orbit and got no store asked for something specific and must be told.
                allow_none=(s1_orbit == "both"),
                get_credentials=iam_icechunk_credentials,
                s3_region=s3_region,
            )
        except InsufficientCoverageError:
            return None
        return _mosaic_stores(mosaic_base, effective)

    # (2) Marker probe over the MAXIMAL candidate set (reflectance + BOTH SAR orbits), not the
    #     resolved orbit set: a prior attempt that wrote one child store then crashed before
    #     stamping any marker leaves data the resolved-orbit probe would miss (with no SAR
    #     store `_resolved_stores` is None), and appending onto it would dedupe against stale
    #     dates then stamp the new fingerprint over mixed inputs. Physical existence is the
    #     signal.
    candidates = _mosaic_stores(mosaic_base, "both")
    probed = {s: _probe_marker(s, get_credentials=iam_icechunk_credentials, s3_region=s3_region) for s in candidates}
    # An UNMARKED store is an interrupted attempt at THIS work and is resumed. A store marked
    # with a DIFFERENT fingerprint is a violated assumption and raises — see the two branches
    # below for why those answers are opposite.
    unmarked = [s for s, (exists, marker) in probed.items() if exists and marker is None]
    mismatched = {
        s: marker for s, (exists, marker) in probed.items() if exists and marker is not None and marker != fingerprint
    }
    # Resolve orbits ONLY when every present store is complete. `resolve_s1_orbit` re-raises
    # GroupNotFoundError deliberately — at fill time a rootless SAR store must never read as
    # an absent orbit and quietly halve the radar — but an interrupted store is exactly how
    # one becomes rootless, so resolving first would put that raise in front of the resume.
    # Skipping the call costs nothing: an incomplete store cannot match the fingerprint.
    resolved = None if (unmarked or mismatched) else _resolved_stores()
    if resolved is not None and all(probed[s][1] == fingerprint for s in resolved):
        log.info("Zone %s year %d already ingested for %s — skipping", zone, year, fingerprint["window"])
        return {"zone": zone, "year": year, "status": "already_ingested", "fingerprint": fingerprint}

    # A completed store whose fingerprint DISAGREES means a documented campaign invariant was
    # broken: window, coverage gate, orbit set, partial-window policy and land mask are all
    # fixed for a campaign. Neither automatic answer is right — resuming would append dates
    # admitted under one configuration onto dates admitted under another and stamp one
    # fingerprint over the mixture; clearing would destroy a complete, correct mosaic over a
    # mistyped parameter. So a human decides. Rare by construction, expensive either way.
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

    # RESUME, do not rebuild. An unmarked store holds dates this same work already committed,
    # and Icechunk commits a date's time slot atomically WITH its pixels, so a date present is
    # complete and a date absent was never started — there is no partial date to repair. Both
    # ingests already skip dates they find (`get_existing_dates`), so simply not clearing is
    # the whole mechanism. Interruptions are expected (the orphan sweeper cancels runs by
    # design), and discarding a dense zone's near-complete ingest costs hours on every retry.
    #
    # TWO WRITERS is the hazard resume makes reachable, and Icechunk guards it: these commits
    # do NOT rebase, so a second writer's commit fails with ConflictError against a moved
    # branch tip rather than merging silently. That protection is a property of NOT passing
    # `rebase_with` — see `storage.shard_writer.commit_with_rebase`, which deliberately does
    # the opposite for disjoint groups. The ingest path must never adopt it, and the write
    # RETRY must never retry the resulting error: retrying re-opens the session from the tip
    # the other writer moved, which is `rebase_with` by another route.
    # `storage.zarr_store.store_write_retrying` excludes it by type.
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

    # (4) Size the fleet from the cell's LIVE work. Writes are restricted to live windows, so
    # the live chunk count IS the work measure — the 03S incident showed extent-sized fleets
    # are wrong by orders of magnitude.
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
        "allow_ingest_code_mismatch": allow_ingest_code_mismatch,
        # The children open and CREATE the mosaic repos; without this they sign against the
        # storage default while this flow's probes use s3_region, so a non-default-region
        # campaign fails inside the costly S1/S2 jobs rather than at preflight.
        "s3_region": s3_region,
    }
    orbits = _active_orbits(s1_orbit)
    # Optional perf-report capture: the setting is a base URI, scoped by CELL then by child.
    # run-global-campaign hands the SAME IngestSettings to every (zone, year), so a base-only
    # path would have concurrent cells racing on one s2.html with later cells overwriting
    # earlier ones; the per-child suffix separates the S1 orbits from S2 within a cell.
    perf_cell = ingest_settings.perf_report_uri
    perf_base = f"{perf_cell.rstrip('/')}/{zone}-{year}" if perf_cell else None
    # S1 runs NARROWER than S2, overriding the shared width in `common`. Its work is a fixed
    # fraction of S2's, so an equal fleet finishes early then holds quota it cannot use — and
    # quota is what limits concurrent cells, hence the campaign's schedule.
    s1_workers = _s1_max_workers(max_workers, ingest_settings)
    log.info("Zone %s: S2 max_workers=%d, each S1 orbit max_workers=%d", zone, max_workers, s1_workers)
    # Stamped on every child so the terminal hooks can find them from this flow-run id alone
    # (see _CHILD_TAG_PREFIX). None outside a Prefect run — a direct .fn() call in tests
    # dispatches nothing worth sweeping.
    child_tags = [t] if (t := child_run_tag(_CHILD_TAG_PREFIX, flow_run_ctx.id)) else None
    # One (label, deployment, parameters, store) per leg, so a failed leg is re-dispatched
    # verbatim rather than reconstructed. The store is the leg's own child store, where its
    # progress is legible — see `_committed_date_count`.
    legs: list[tuple[str, str, dict, str]] = [
        (
            f"ingest_s1_roi_sar ({orbit})",
            deployments.ingest_s1_roi_sar,
            {
                **common,
                "orbit": orbit,
                "batch_days": ingest_settings.batch_days,
                # Its OWN floor: `common` resolved against the S2 width, and an S1 orbit is a
                # fraction of that, so inheriting it would request more workers than this leg
                # was sized for and the fleet could never reach its minimum.
                "min_workers": ingest_settings.floor_for(s1_workers),
                "max_workers": s1_workers,
                "perf_report_uri": f"{perf_base}/s1-{orbit}.html" if perf_base else None,
            },
            _leg_store(mosaic_base, orbit),
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
            _leg_store(mosaic_base, None),
        )
    ]

    # ONE RETRY LOOP PER LEG, joined only at the end — the legs do not wait for each other.
    #
    # Joining every leg before re-dispatching the failed ones puts sibling latency on the
    # cell's critical path: on 2026-08-18 zone 47S's descending leg failed one minute in and
    # was not re-dispatched for an hour, waiting on an optical leg with another hour to run,
    # while a cell cannot infer until every leg has landed anyway. Per leg, attempts are
    # spaced by `leg_retry_backoff_s` rather than by whatever the slowest sibling took, and
    # the wall-clock budget is per leg, anchored at that leg's own first dispatch. Nothing is
    # lost by not joining: each leg writes its OWN child store under the base, and a
    # re-dispatch clears nothing (see "RESUME, do not rebuild" above).
    #
    # THE INVARIANT, and it is the important one: a leg that can never succeed must fail the
    # cell, and a sibling's later success must not erase it. `s1_orbit="both"` with one SAR
    # deployment permanently broken must not resolve to the single orbit that did ingest, pass
    # the coverage gate on it, and stamp a "both" marker over half the radar — after which
    # every later run reads the marker and skips the cell. Here each leg owns its own result
    # and nothing can clear another's, so the property is structural rather than ordering.
    #
    # A terminally failed leg does NOT cancel its siblings: their committed dates help the
    # next dispatch either way, and interrupting a running write is how a store becomes an
    # interrupted one. The failure is reported as soon as they settle and the cell returns to
    # the campaign work list.
    #
    # It DOES stop them spending further attempts. Once any leg has failed unfixably the cell
    # cannot succeed, so a sibling's remaining retries buy nothing while each holds a Dask
    # fleet. `doomed` gates only the decision to START another attempt — the same thing the
    # wall-clock bound gates, for the same reason: a leg already running is never interrupted.
    doomed = asyncio.Event()

    def _leg_backoff_base_s(detail: str) -> float:
        """The FIRST rung of a leg's retry ladder, chosen by what the failure is.

        Deliberately short for the ordinary failure. The ingest resumes, so a retry costs only
        the work actually lost, and campaign throughput is bounded by legs landing — a long
        backoff idles a GPU fleet waiting on a mosaic.

        **Long for one class, and the reason is where the fleet is.** A short delay cannot
        distinguish a provider that has stopped serving reads from one that has stopped for
        good, so a short backoff walks re-dispatches straight back into a refusing source and
        spends a cell's whole attempt budget in two minutes. A failed leg has already released
        its Dask fleet, so waiting HERE costs latency and nothing else, where the same
        patience inside the leg holds hundreds of vCPU idle.

        Named separately from the ladder because it is the rung the ladder falls back TO, and
        a base chosen in two places is a base that can be chosen two ways.
        """
        if ProviderRefusedReadsError.__name__ in detail and ingest_settings.leg_refusal_backoff_s > 0:
            return float(ingest_settings.leg_refusal_backoff_s)
        return float(ingest_settings.leg_retry_backoff_s)

    def _leg_backoff_s(attempt: int, detail: str, remaining_s: float) -> float | None:
        """The longest rung that FITS ``remaining_s``, or ``None`` if not even the base rung does.

        The ladder doubles from the base, capped at four times it. When the rung this attempt
        escalated to is longer than the budget has left, the ladder DESCENDS to the rung
        beneath rather than ending the retry: a shorter rung is the same policy one escalation
        earlier, and the base states how long this failure class is worth waiting for — a leg
        with a quarter of an hour left is not helped by hearing that twenty minutes does
        not fit.

        This is NOT capping to the remainder, and that distinction is the whole of it. Waiting
        exactly what is left lands the next dispatch ON the deadline every time, turning a
        race into a guarantee of the thing the budget forbids. A rung is returned only if it
        is strictly shorter than what remains, so the attempt it buys has room to start.
        """
        base = _leg_backoff_base_s(detail)
        wait = min(base * (2 ** (attempt - 1)), base * 4)
        while wait > base and wait >= remaining_s:
            wait /= 2
        return wait if wait < remaining_s else None

    async def _run_leg(label: str, dep: str, params: dict, store: str) -> str | None:
        """Dispatch one leg, retrying on its own schedule; its failure detail, or None if it ran.

        Classifies per leg: a deterministic repeat from the source catalogue, a marker meaning
        a re-dispatch cannot help, or a load refusal that is evidence FOR patience.
        """
        # Before the wall-clock anchor, so a leg's stagger is not charged to the budget that
        # decides whether it may attempt again.
        if (stagger := _leg_stagger_s(zone, year, label, ingest_settings.leg_stagger_window_s)) > 0:
            log.info(
                "Zone %s year %s: %s — waiting %.0f s before its first dispatch so the fleet "
                "does not reach the catalogue in phase (leg_stagger_window_s=%d).",
                zone,
                year,
                label,
                stagger,
                ingest_settings.leg_stagger_window_s,
            )
            # Raced against `doomed`, not slept through. A sibling can fail terminally seconds
            # after this leg starts waiting, and at the default window a leg near the top of
            # the spread would otherwise hold its cell's slot — and the campaign's ingest slot
            # behind it — for most of ten minutes to learn something already decided.
            try:
                await asyncio.wait_for(doomed.wait(), timeout=stagger)
            except TimeoutError:
                pass  # the stagger elapsed with the cell still viable, which is the ordinary path
            if doomed.is_set():
                # Re-checked after the wait, for the reason the retry backoff re-checks after
                # its own: a sibling can reach its terminal failure while this leg waits, and
                # the gate exists to not START work for a cell that cannot succeed.
                #
                # Returns a DETAIL, not None. None is this function's "the leg ran", so a leg
                # that never dispatched would be gathered as a success and the cell finalised
                # on two legs' worth of work. Having no failure of its own, it reports not
                # having started.
                never_started = (
                    "not dispatched: another leg of this cell failed terminally before this "
                    "leg's stagger elapsed, so the cell cannot succeed"
                )
                log.warning(
                    "Zone %s year %s: %s was never dispatched — another leg failed terminally "
                    "while this one was waiting out its stagger. Detail: %s",
                    zone,
                    year,
                    label,
                    never_started,
                )
                return never_started
        # Where this leg's store stood before it ran, so "has it committed anything since" has
        # an answer at the FIRST re-dispatch decision as well as the last. Taken before the
        # wall-clock anchor, for the reason the stagger is: a reading is not patience, and
        # charging it to the budget would let the measurement shorten what it measures.
        # Skipped entirely when the extension is off, so turning it off costs no reads.
        committed = (
            _committed_date_count(store, log, get_credentials=iam_icechunk_credentials, s3_region=s3_region)
            if ingest_settings.leg_progress_extension_s > 0
            else None
        )
        started = monotonic()
        # The deadline this leg is held to: the setting until the leg EARNS more. A local
        # rather than a recomputation, so every gate below reads one number and a grant cannot
        # be silently undone by the next gate re-reading the setting.
        budget_s = float(ingest_settings.max_leg_wall_clock_s)

        def _credit_progress() -> bool:
            """Grow this leg's deadline, if its store has gained dates since the last look.

            The deadline bounds PATIENCE — wall clock a leg spends not getting anywhere. Counted
            from the first dispatch it cannot tell that from wall clock spent WORKING: a leg
            that committed steadily for hours and then hit a transient failure is charged for
            every productive hour and refused the attempt that would have resumed from them. A
            store that has grown is not the cell the deadline was written for; it resumes from
            what it committed, so its next attempt starts further along than its last.

            Bounded by construction, and both bounds are needed: a grant costs a FIXED
            extension, and each must be PAID FOR by dates committed since the previous grant,
            so a store that stops growing stops earning them.

            Payment is what limits the payout rate, which is why this may be asked wherever the
            deadline is about to refuse rather than at one chosen gate. Only a RUNNING leg
            commits and every ask sits after a failed attempt, so the asks within one attempt
            compete for the same growth and at most one is paid. Grants are therefore bounded
            by a leg's re-dispatch decisions, one fewer than its attempts, giving a ceiling of
            ``max_leg_wall_clock_s + (max_leg_attempts - 1) * leg_progress_extension_s``. A leg
            that commits nothing never leaves ``max_leg_wall_clock_s``.

            The extension is FIXED, so the caller re-reads the deadline after a grant rather
            than assuming one sufficed. An attempt that overran by more than a whole extension
            is past what progress buys: sizing the grant to the overrun would make the ceiling
            a function of how long the leg ran, the very thing the deadline bounds.

            A store that cannot be read earns nothing, the same answer as no progress.
            """
            nonlocal budget_s, committed
            if ingest_settings.leg_progress_extension_s <= 0:
                return False
            grown = _committed_date_count(store, log, get_credentials=iam_icechunk_credentials, s3_region=s3_region)
            if committed is None or grown is None or grown <= committed:
                return False
            before, committed = committed, grown
            budget_s += ingest_settings.leg_progress_extension_s
            log.warning(
                "Zone %s year %s: %s reached its wall-clock budget while still committing dates — "
                "its store now holds %d, up from %d — so the budget is extended by %d s to %.0f s "
                "and the attempt it was about to be refused goes ahead. A leg that stops committing "
                "stops earning this.",
                zone,
                year,
                label,
                grown,
                before,
                ingest_settings.leg_progress_extension_s,
                budget_s,
            )
            return True

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
                # TERMINAL FOR THE CELL, so say so. `gather` must fail the cell once this leg
                # gives up, and a sibling backing off right now would otherwise keep launching
                # its remaining attempts — holding a Dask fleet for a cell that cannot succeed
                # and delaying the failure report. Same reasoning as the unfixable-failure
                # branch above; only the route differs.
                doomed.set()
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
            # The wall-clock bound, checked ONLY here — at the decision to START another
            # attempt. A RUNNING leg is never measured against it, so a slow-but-succeeding leg
            # cannot be why this stopped; the bound refuses re-dispatching after patience has
            # already had a deadline's worth of wall clock. Per leg, so one leg's slow retries
            # cannot spend another's budget.
            elapsed = monotonic() - started
            # ONE credit decision per failed attempt, taken before EITHER deadline-based
            # refusal below. Those are the same deadline asked two ways — "no time left" and
            # "no time left to wait first" — so asking at only one would recognise a leg's
            # progress according to which side of the deadline its attempt happened to land
            # on. The deadline is re-read after: a grant is fixed, and not owed to be enough.
            if elapsed >= budget_s or _leg_backoff_base_s(detail) >= budget_s - elapsed:
                _credit_progress()
            if elapsed >= budget_s:
                log.error(
                    "Zone %s year %s: %s — %.0f s elapsed since this leg's first attempt, past its "
                    "%.0f s wall-clock budget for starting another (max_leg_wall_clock_s=%d), so "
                    "attempt %d/%d is refused even though the attempt budget has room. No running "
                    "leg was interrupted. The cell fails back to the campaign work list and a later "
                    "dispatch RESUMES from the dates already committed — this costs latency, not "
                    "work. Detail: %s",
                    zone,
                    year,
                    label,
                    elapsed,
                    budget_s,
                    ingest_settings.max_leg_wall_clock_s,
                    attempt + 1,
                    ingest_settings.max_leg_attempts,
                    detail,
                )
                # Terminal for the cell for the same reason as the attempt-budget branch: this
                # leg will start nothing further, so no sibling should either.
                doomed.set()
                return detail
            # A rung longer than what is left descends the ladder; only a leg with no room for
            # even the base rung is refused. Refusing outright would be a verdict from the
            # ESCALATION rather than the budget — denying a leg with a quarter of an hour left
            # because the next wait is twenty minutes, when the rung beneath it fits easily.
            wait = _leg_backoff_s(attempt, detail, budget_s - elapsed)
            if wait is None:
                log.error(
                    "Zone %s year %s: %s — %.0f s elapsed and not even the shortest backoff for "
                    "this failure (%.0f s) fits its %.0f s wall-clock budget for starting another "
                    "attempt (max_leg_wall_clock_s=%d), so attempt %d/%d is refused rather than "
                    "waited for. No running leg was interrupted. The cell fails back to the "
                    "campaign work list and a later dispatch RESUMES from the dates already "
                    "committed. Detail: %s",
                    zone,
                    year,
                    label,
                    elapsed,
                    _leg_backoff_base_s(detail),
                    budget_s,
                    ingest_settings.max_leg_wall_clock_s,
                    attempt + 1,
                    ingest_settings.max_leg_attempts,
                    detail,
                )
                doomed.set()
                return detail
            # A re-dispatch RESUMES: already-committed dates are skipped, not rewritten, so the
            # retry costs only the work actually lost. That idempotency is what makes retrying
            # the default rather than the exception.
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
            # RACED against `doomed`, not slept through, for the reason the stagger races its
            # own wait: a refusal's backoff is minutes, and a leg that sleeps through a
            # sibling's terminal failure holds its cell's slot — and the campaign's ingest
            # slot behind it — for all of it, to learn something already decided.
            try:
                await asyncio.wait_for(doomed.wait(), timeout=wait)
            except TimeoutError:
                pass  # the backoff elapsed with the cell still viable, which is the ordinary path
            # Re-read the clock rather than trusting the fit guard's arithmetic. That guard is
            # a PREDICTION — elapsed plus the scheduled wait — and `wait_for` returns when the
            # event loop gets to it, so a wait chosen as just inside the budget can land just
            # outside it. Only a reading after the await observes what happened.
            elapsed = monotonic() - started
            # Same rule: a deadline refusal asks first. Nothing ran during the backoff, so this
            # can only be paid for by progress the gates above had no reason to ask about —
            # which is the case this gate exists for, a wait that landed outside the budget
            # because the event loop overran.
            if elapsed >= budget_s:
                _credit_progress()
            if elapsed >= budget_s:
                log.error(
                    "Zone %s year %s: %s — the backoff ended %.0f s after this leg's first "
                    "attempt, at or past its %.0f s wall-clock budget for starting another "
                    "(max_leg_wall_clock_s=%d), so attempt %d/%d is refused. No running leg was "
                    "interrupted and a later dispatch RESUMES from the dates already committed. "
                    "Detail: %s",
                    zone,
                    year,
                    label,
                    elapsed,
                    budget_s,
                    ingest_settings.max_leg_wall_clock_s,
                    attempt + 1,
                    ingest_settings.max_leg_attempts,
                    detail,
                )
                doomed.set()
                return detail
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

    settled = await asyncio.gather(*(_run_leg(label, dep, params, store) for label, dep, params, store in legs))
    errors: list[str] = [detail for detail in settled if detail is not None]

    if errors:
        raise RuntimeError(f"ingest deployment(s) failed for zone {zone} year {year}: " + "; ".join(errors))

    # (5) Resolve the orbit set from what actually ingested, then verify and mark only those
    #     stores. `s1_orbit="both"` with one empty orbit downgrades here rather than failing
    #     the coverage check on a store that never exists.
    stores = _resolved_stores()
    if stores is None:
        msg = f"s1_orbit={s1_orbit!r} but no SAR store was produced for zone {zone} year {year}"
        raise InsufficientCoverageError(msg)
    # Derived from the COUNT of resolved stores, so the radar-free case is not mistaken for a
    # single-orbit one: with reflectance alone `stores[-1]` is the reflectance store, and
    # rsplitting it for an orbit name yields "reflectance".
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
    # Validated first and in a SEPARATE pass: a store the legs never appended to has had no
    # manifest check at all this run, and marking a mosaic built under different inputs makes
    # every later run skip it. Two passes so a disagreement on the LAST store cannot leave the
    # earlier ones already marked.
    for store in stores:
        _assert_store_manifest_matches(
            store,
            roi_path,
            # Optical only: the SAR legs apply no admission threshold, so their manifests
            # carry none and expecting one would fail every radar store.
            min_valid_coverage=ingest_settings.min_valid_coverage if store.endswith("reflectance.zarr") else None,
            allow_ingest_code_mismatch=allow_ingest_code_mismatch,
            get_credentials=iam_icechunk_credentials,
            s3_region=s3_region,
        )
    for store in stores:
        _write_ingest_marker(store, fingerprint, get_credentials=iam_icechunk_credentials, s3_region=s3_region)

    log.info("Zone %s year %d ingested (orbit=%s, %s)", zone, year, effective_orbit, fingerprint["window"])
    return {"zone": zone, "year": year, "status": "ingested", "fingerprint": fingerprint, "stores": stores}
