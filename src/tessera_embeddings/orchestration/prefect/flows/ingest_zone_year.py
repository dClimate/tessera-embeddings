"""Ingest one ``(zone, year)`` of the campaign onto the fixed zone grid.

The global campaign reads its per-zone mosaics from ``{inputs}/mosaics/{zone}/{year}``;
this flow produces them by **reusing the ROI ingest engine unchanged**. A
zone-shaped ROI zarr (:func:`tessera_embeddings.ingest.land_mask.export_zone_roi`)
pins the grid to the fixed :class:`~tessera_embeddings.storage.zone_grid.ZoneSpec`
extent — so the resulting mosaic passes the zone-fill's exact-grid validation,
which ``generate_roi``'s bbox-fit grid could not — and the S1/S2 ROI ingest
deployments write onto it. Only orchestration lives here (``arun_deployment``);
each ingest deployment provisions its own Dask cluster.

**Idempotent + crash-safe.** A per-store completion marker (root attr
``ingest_marker``, fingerprinting window + min_valid_coverage + s1_orbit +
allow_partial_window + coverage sha) gates the work: a matching marker on every
required store short-circuits, and the marker is written only after coverage is
verified. Recovery is by re-run, but NOT an incremental append: the probe keys on
physical existence over the maximal candidate set, so a stale, markerless, or
half-written mosaic is CLEARED (``delete_prefix``, strict) and rebuilt cleanly
rather than appended onto. A changed input (rebuilt coverage, new threshold,
different orbit or window, or a flipped ``allow_partial_window``) changes the
fingerprint and likewise forces a clean rebuild.

**Coverage gate (ADR-011).** After ingestion, :func:`check_time_window_coverage`
requires the mosaics' months to span the window; ``allow_partial_window`` relaxes
that to "non-empty" for a legitimately partial edge zone. This is the ingest-side
guard; the fill re-checks before provisioning Ray.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

import icechunk
import zarr
from prefect import flow, get_run_logger
from prefect.deployments import arun_deployment
from pydantic import BaseModel

from tessera_embeddings.config.ingest import IngestSettings
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.data_loading import (
    _active_orbits,
    _is_missing_repo,
    check_time_window_coverage,
    resolve_s1_orbit,
)
from tessera_embeddings.ingest.land_mask import export_zone_roi, live_chunk_count
from tessera_embeddings.orchestration.prefect.flows.tessera_full_pipeline import _check_completed
from tessera_embeddings.orchestration.runners.zone_fill import zone_has_live_tiles
from tessera_embeddings.storage.object_store import delete_prefix
from tessera_embeddings.storage.zarr_store import open_or_create_repo, open_store_as_zarr_group
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
    conflating it with a missing repo would let one unreadable store trip the
    clear-and-rebuild branch and delete a valid mosaic (or ingest over unknown
    data). Only a genuinely-missing repo reports ``(False, None)``.
    """
    try:
        root = open_store_as_zarr_group(store_path, get_credentials=get_credentials, region=s3_region)
    except FileNotFoundError:
        return (False, None)
    except icechunk.IcechunkError as exc:
        if _is_missing_repo(exc):
            return (False, None)
        raise
    raw = root.attrs.get("ingest_marker")
    return (True, dict(raw) if isinstance(raw, dict) else None)


def _write_ingest_marker(store_path: str, fingerprint: dict, *, get_credentials: _Creds, s3_region: str | None) -> None:
    """Stamp the ``ingest_marker`` fingerprint + ``ingest_completed_at`` on a store's root."""
    repo = open_or_create_repo(store_path, get_credentials=get_credentials, region=s3_region)[0]
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
    floor = max(settings.min_workers, min(_WORKERS_FLOOR, settings.max_workers))
    return max(floor, min(settings.max_workers, round(live_chunks * _WORKERS_PER_LIVE_CHUNK)))


@flow(name="ingest-zone-year")
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

    # Lazy AWS import so the flow file imports on non-AWS machines (arch tests).
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials

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
                mosaic_base, s1_orbit, get_credentials=iam_icechunk_credentials, s3_region=s3_region
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
    resolved = _resolved_stores()
    if resolved is not None and all(probed[s][1] == fingerprint for s in resolved):
        log.info("Zone %s year %d already ingested for %s — skipping", zone, year, fingerprint["window"])
        return {"zone": zone, "year": year, "status": "already_ingested", "fingerprint": fingerprint}
    if any(exists for exists, _ in probed.values()):
        # Something exists under mosaic_base but it is NOT a clean, fully-marked
        # mosaic for the current fingerprint: a stale marker (changed inputs, or an
        # ascending-only prior run now asked for both), a markerless half-write, or a
        # SAR crash before marking. Appending would dedupe against stale dates then
        # stamp the new fingerprint over mixed inputs — so clear the whole prefix for
        # a clean rebuild (strict=True: a FAILED delete aborts rather than ingesting
        # onto stale data and marking the result complete).
        log.info(
            "Zone %s year %d mosaic is stale or partial — clearing %s for a clean rebuild", zone, year, mosaic_base
        )
        delete_prefix(mosaic_base, log=log, strict=True)

    # (3) Ensure the zone ROI zarr (idempotent; regenerates if coverage changed).
    export_zone_roi(
        zone,
        land_mask_path=land_mask_path,
        dest_path=roi_path,
        get_credentials=iam_icechunk_credentials,
        s3_region=s3_region,
    )

    # (4) Size the fleet from the cell's live work when cropping: with writes
    # restricted to live windows the chunk count IS the work measure, and the
    # 03S incident showed extent-sized fleets are wrong by orders of magnitude.
    max_workers = ingest_settings.max_workers
    if ingest_settings.crop_to_live_windows:
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
        "min_workers": ingest_settings.min_workers,
        "max_workers": max_workers,
        "use_local": use_local,
        "crop_to_live_windows": ingest_settings.crop_to_live_windows,
    }
    orbits = _active_orbits(s1_orbit)
    # Optional perf-report capture: the setting is a base URI. Scope it by CELL
    # first, then by child. run-global-campaign hands the SAME IngestSettings to
    # every (zone, year), so a base-only path would have concurrent cells racing
    # on one s2.html and later cells overwriting earlier ones; the per-child
    # suffix then separates the S1 orbits from S2 within a cell.
    perf_cell = ingest_settings.perf_report_uri
    perf_base = f"{perf_cell.rstrip('/')}/{zone}-{year}" if perf_cell else None
    s1_coros = [
        arun_deployment(
            deployments.ingest_s1_roi_sar,
            parameters={
                **common,
                "orbit": orbit,
                "batch_days": ingest_settings.batch_days,
                "perf_report_uri": f"{perf_base}/s1-{orbit}.html" if perf_base else None,
            },
        )
        for orbit in orbits
    ]
    s2_coro = arun_deployment(
        deployments.ingest_s2_roi_reflectance,
        parameters={
            **common,
            "min_valid_coverage": ingest_settings.min_valid_coverage,
            "perf_report_uri": f"{perf_base}/s2.html" if perf_base else None,
        },
    )
    # return_exceptions=True so we WAIT for every deployment to settle before raising:
    # a plain gather() surfaces the first failure while the sibling S1/S2 jobs keep
    # writing to mosaic_base, and a retry could then clear the prefix mid-write. Join
    # them all, then report every failure at once.
    *s1_runs, s2_run = await asyncio.gather(*s1_coros, s2_coro, return_exceptions=True)
    labelled = [
        *((f"ingest_s1_roi_sar ({o})", r) for o, r in zip(orbits, s1_runs, strict=True)),
        ("ingest_s2_roi_reflectance", s2_run),
    ]
    errors: list[str] = []
    for label, run in labelled:
        if isinstance(run, BaseException):
            errors.append(f"{label}: {run!r}")
            continue
        try:
            _check_completed(run, label)
        except Exception as exc:  # a returned-but-not-COMPLETED terminal state
            errors.append(str(exc))
    if errors:
        raise RuntimeError(f"ingest deployment(s) failed for zone {zone} year {year}: " + "; ".join(errors))

    # (5) Resolve the orbit set from what actually ingested, then verify + mark
    #     only those stores. `s1_orbit="both"` with one empty orbit downgrades
    #     here rather than failing the coverage check on a store that never exists.
    stores = _resolved_stores()
    if stores is None:
        msg = f"s1_orbit={s1_orbit!r} but no SAR store was produced for zone {zone} year {year}"
        raise InsufficientCoverageError(msg)
    effective_orbit = "both" if len(stores) == 3 else stores[-1].rsplit("sar_", 1)[-1].removesuffix(".zarr")
    check_time_window_coverage(
        mosaic_base,
        window,
        s1_orbit=effective_orbit,
        skip_coverage_check=allow_partial_window,
        get_credentials=iam_icechunk_credentials,
        s3_region=s3_region,
    )

    # (6) Marker last: a crash before this point re-runs incrementally.
    for store in stores:
        _write_ingest_marker(store, fingerprint, get_credentials=iam_icechunk_credentials, s3_region=s3_region)

    log.info("Zone %s year %d ingested (orbit=%s, %s)", zone, year, effective_orbit, fingerprint["window"])
    return {"zone": zone, "year": year, "status": "ingested", "fingerprint": fingerprint, "stores": stores}
