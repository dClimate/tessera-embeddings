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
``ingest_marker``, fingerprinting window + min_valid_coverage + coverage sha)
gates the work: a matching marker on every required store short-circuits, and a
crash mid-ingest is repaired by a plain re-run — the ROI flows dedupe
already-present dates (incremental append) and the marker is written only after
coverage is verified. A changed input (rebuilt coverage, new threshold) changes
the fingerprint, forcing a re-ingest.

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

from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.data_loading import _active_orbits, check_time_window_coverage, resolve_s1_orbit
from tessera_embeddings.ingest.land_mask import export_zone_roi
from tessera_embeddings.orchestration.prefect.flows.tessera_full_pipeline import _check_completed
from tessera_embeddings.orchestration.runners.zone_fill import zone_has_live_tiles
from tessera_embeddings.storage.object_store import delete_prefix
from tessera_embeddings.storage.zarr_store import open_or_create_repo, open_store_as_zarr_group
from tessera_embeddings.storage.zone_grid import canonicalize_zone
from tessera_embeddings.utils import utcnow_iso

#: Percent-of-ROI valid-pixel threshold for keeping an S2 solar-day. Far below
#: the ROI default (5.0): a single day's swath covers only a sliver of a whole
#: 6° zone, so a high bar would drop nearly every date (see ADR-011).
DEFAULT_MIN_VALID_COVERAGE = 0.1


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


def _read_ingest_marker(store_path: str, *, get_credentials: _Creds, s3_region: str | None) -> dict | None:
    """The store's ``ingest_marker`` fingerprint dict, or ``None`` if absent / unopenable."""
    try:
        root = open_store_as_zarr_group(store_path, get_credentials=get_credentials, region=s3_region)
    except (FileNotFoundError, icechunk.IcechunkError):
        return None
    raw = root.attrs.get("ingest_marker")
    return dict(raw) if isinstance(raw, dict) else None


def _write_ingest_marker(store_path: str, fingerprint: dict, *, get_credentials: _Creds, s3_region: str | None) -> None:
    """Stamp the ``ingest_marker`` fingerprint + ``ingest_completed_at`` on a store's root."""
    repo = open_or_create_repo(store_path, get_credentials=get_credentials, region=s3_region)[0]
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    root.attrs["ingest_marker"] = dict(fingerprint)
    root.attrs["ingest_completed_at"] = utcnow_iso()
    session.commit(f"ingest marker: {fingerprint}")


@flow(name="ingest-zone-year")
async def ingest_zone_year(
    *,
    zone: str,
    year: int,
    paths: BucketPaths,
    deployments: IngestDeployments = IngestDeployments(),  # noqa: B008
    mask_name: str = "global",
    s1_orbit: str = "both",
    min_workers: int = 1,
    max_workers: int = 50,
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
    batch_days: int = 30,
    time_window_end: str | None = None,
    allow_partial_window: bool = False,
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
        min_workers: Lower Dask worker bound forwarded to each ingest.
        max_workers: Upper Dask worker bound forwarded to each ingest.
        min_valid_coverage: S2 per-solar-day keep threshold (percent of the ROI);
            defaults low for zone-scale ROIs.
        batch_days: S1 CMR batch window forwarded to the SAR ingest.
        time_window_end: ``"Month Year"`` override; defaults to ``"December {year}"``.
        allow_partial_window: Relax the coverage gate to "non-empty" (escape
            hatch for a legitimately partial edge zone).
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
    roi_path = f"{paths.inputs.rstrip('/')}/rois/zarrs/zone_{zone}.zarr"

    # (1) All-ocean zone: no ROI, no ingest — the fill marks it empty.
    if not zone_has_live_tiles(land_mask_path, zone, get_credentials=iam_icechunk_credentials):
        log.info("Zone %s has no live tiles (all-ocean) — skipping ingest", zone)
        return {"zone": zone, "year": year, "status": "skipped_ocean"}

    window = parse_time_window(time_window_end or f"December {year}")
    start_date, end_date = window.to_date_range()
    # Fingerprint the ingest INPUTS, not just the window: rebuilding the coverage
    # (new registry_sha256), changing min_valid_coverage, the window, or the
    # REQUESTED orbit set all change the mosaic that should be produced. Including
    # s1_orbit is what makes an ascending-only run's marker mismatch a later "both"
    # request (so the missing orbit is actually ingested, not skipped).
    fingerprint = {
        "window": [start_date, end_date],
        "min_valid_coverage": min_valid_coverage,
        "s1_orbit": s1_orbit,
        "coverage_sha256": _coverage_sha(
            land_mask_path, zone, get_credentials=iam_icechunk_credentials, s3_region=None
        ),
    }

    def _resolved_stores() -> list[str] | None:
        """Stores for the orbits actually present, or None if no SAR store exists yet.

        `s1_orbit="both"` downgrades to whatever ingested (an orbit with no
        granules for this zone/window writes no store), matching the fill's
        resolve_s1_orbit — so we never require a SAR store that will never exist.
        """
        try:
            effective = resolve_s1_orbit(mosaic_base, s1_orbit, get_credentials=iam_icechunk_credentials)
        except InsufficientCoverageError:
            return None
        return _mosaic_stores(mosaic_base, effective)

    # (2) Completion-marker probe: every store present for the resolved orbit set
    #     carries this exact fingerprint? A crash-resumed run then no-ops.
    probe_stores = _resolved_stores()
    existing = (
        [_read_ingest_marker(s, get_credentials=iam_icechunk_credentials, s3_region=None) for s in probe_stores]
        if probe_stores is not None
        else []
    )
    if existing and all(m == fingerprint for m in existing):
        log.info("Zone %s year %d already ingested for %s — skipping", zone, year, fingerprint["window"])
        return {"zone": zone, "year": year, "status": "already_ingested", "fingerprint": fingerprint}
    if any(m is not None for m in existing):
        # A present-but-mismatched marker (changed inputs, or an ascending-only prior
        # run now asked for both): the mosaic was built under different inputs, so a
        # plain re-ingest would append onto stale data. Clear it for a clean rebuild
        # — strict=True so a FAILED delete aborts rather than silently ingesting onto
        # (deduplicated) stale dates and marking the result complete.
        log.info("Zone %s year %d has a stale ingest marker — clearing %s for a clean rebuild", zone, year, mosaic_base)
        delete_prefix(mosaic_base, log=log, strict=True)

    # (3) Ensure the zone ROI zarr (idempotent; regenerates if coverage changed).
    export_zone_roi(zone, land_mask_path=land_mask_path, dest_path=roi_path, get_credentials=iam_icechunk_credentials)

    # (4) Dispatch S1 (per REQUESTED orbit) + S2 ingestion concurrently onto the ROI.
    common: dict[str, Any] = {
        "roi_zarr_path": roi_path,
        "start_date": start_date,
        "end_date": end_date,
        "store_path": mosaic_base,
        "min_workers": min_workers,
        "max_workers": max_workers,
        "use_local": use_local,
    }
    orbits = _active_orbits(s1_orbit)
    s1_coros = [
        arun_deployment(deployments.ingest_s1_roi_sar, parameters={**common, "orbit": orbit, "batch_days": batch_days})
        for orbit in orbits
    ]
    s2_coro = arun_deployment(
        deployments.ingest_s2_roi_reflectance, parameters={**common, "min_valid_coverage": min_valid_coverage}
    )
    *s1_runs, s2_run = await asyncio.gather(*s1_coros, s2_coro)
    for orbit, s1_run in zip(orbits, s1_runs, strict=True):
        _check_completed(s1_run, f"ingest_s1_roi_sar ({orbit})")
    _check_completed(s2_run, "ingest_s2_roi_reflectance")

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
    )

    # (6) Marker last: a crash before this point re-runs incrementally.
    for store in stores:
        _write_ingest_marker(store, fingerprint, get_credentials=iam_icechunk_credentials, s3_region=None)

    log.info("Zone %s year %d ingested (orbit=%s, %s)", zone, year, effective_orbit, fingerprint["window"])
    return {"zone": zone, "year": year, "status": "ingested", "fingerprint": fingerprint, "stores": stores}
