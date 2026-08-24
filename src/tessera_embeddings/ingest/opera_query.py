"""OPERA RTC-S1 query utilities: MGRS bbox, orbit filtering, UTM EPSG.

OPERA RTC-S1 items on CMR-STAC lack MGRS tile IDs, orbit direction,
and projection metadata. This module provides helpers to query by
spatial extent, filter by orbit via the native CMR Granule Search API,
derive the CRS, and prepare items for loading.
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from collections import Counter
from collections.abc import Callable
from typing import Any

import mgrs
import requests
from pystac import Asset, Item
from pystac.utils import str_to_datetime
from requests.adapters import HTTPAdapter

from tessera_embeddings.config import S1_OPERA_BANDS
from tessera_embeddings.ingest._http import make_logging_retry
from tessera_embeddings.ingest.auth import get_edl_session, resolve_item_assets, rewrite_assets_to_s3
from tessera_embeddings.ingest.solar_days import normalize_to_solar_day

logger = logging.getLogger(__name__)

# CMR Granule Search API — queried directly instead of CMR-STAC search.
# CMR-STAC's cursor-based pagination intermittently 500s on CONUS-scale
# queries (nasa/cmr-stac#408) and pages internally at ~100 items regardless
# of the requested limit (#411); the native granule API pages cleanly at
# 2000 against the same host. It also supports ``attribute[]`` filtering,
# which CMR-STAC silently ignores — so orbit direction is filtered
# server-side here rather than by intersecting a separate STAC result.
_CMR_GRANULE_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
_CMR_OPERA_SHORT_NAME = "OPERA_L2_RTC-S1_V1"
_CMR_PROVIDER = "ASF"
_CMR_PAGE_SIZE = 2000

#: Polarisations required of every granule, filtered server-side. BOTH of them.
#:
#: CMR matches a multi-valued attribute if ANY of its values matches, and ANDs repeated
#: ``attribute[]`` entries on the same attribute name. Requiring VV alone was therefore not
#: equivalent to requiring the pair: besides the dual VV+VH granules we want and the cross-pol
#: HH/HV ones we do not, the archive holds genuinely single-polarisation VV-only granules,
#: which include VV and so passed. Ingest cannot use half a pair, so every one of them was
#: paged and then rejected by the client-side check in `_granule_to_item` — which is a filter
#: the server can apply for free. Requiring both is EXACT, not merely tighter: it admits no
#: single-pol granule and drops no dual-pol one.
_REQUIRED_POLARIZATIONS = ("VV", "VH")

# Granule data links carry the per-band COG download URLs. The ``rel`` ends
# in ``/data#`` and the HTTPS href ends in ``_<BAND>.tif``. HH/HV are matched
# for RECOGNITION ONLY (EW-mode polar granules carry them instead of VV/VH) so
# the skip log can say what a rejected granule actually had — the pipeline
# still ingests exclusively dual-pol VV+VH (``S1_OPERA_BANDS``).
_CMR_DATA_REL_SUFFIX = "/data#"
_BAND_HREF_RE = re.compile(r"_(VV|VH|HH|HV|mask)\.tif$")


# CMR intermittently times out or returns 5xx under load. urllib3's Retry
# retries read timeouts (``read`` defaults to ``total``) as well as the
# listed statuses, with exponential backoff honoring Retry-After.
_CMR_RETRY = make_logging_retry(
    "CMR",
    total=6,
    backoff_factor=2,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True,
)


def _cmr_session() -> requests.Session:
    """Build a requests Session with retry/backoff for CMR Granule queries."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_CMR_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def mgrs_tile_to_bbox(tile_id: str) -> tuple[float, float, float, float]:
    """Convert an MGRS tile ID to a WGS84 bounding box.

    Uses 1m precision (10-digit MGRS coordinates) for the SW and NE
    corners of the 100km tile.

    Args:
        tile_id: MGRS tile ID (e.g., "33UUP")

    Returns:
        (west, south, east, north) in WGS84 degrees
    """
    m = mgrs.MGRS()

    # Tile corners near band boundaries produce a harmless latitude warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Warning in.*Convert_MGRS_To_Geodetic")

        # SW corner: easting=00000, northing=00000
        sw_mgrs = f"{tile_id}0000000000"
        sw_lat, sw_lon = m.toLatLon(sw_mgrs)

        # NE corner: easting=99999, northing=99999
        ne_mgrs = f"{tile_id}9999999999"
        ne_lat, ne_lon = m.toLatLon(ne_mgrs)

    return (sw_lon, sw_lat, ne_lon, ne_lat)


def mgrs_tile_to_utm_epsg(tile_id: str) -> str:
    """Derive UTM EPSG code from an MGRS tile ID.

    MGRS tile IDs encode the UTM zone number (first 2 digits) and
    hemisphere (latitude band letter: C-M = south, N-X = north).

    Args:
        tile_id: MGRS tile ID (e.g., "33UUP")

    Returns:
        EPSG code string (e.g., "EPSG:32633" for 33N, "EPSG:32752" for 52S)
    """
    zone_number = int(tile_id[:2])
    lat_band = tile_id[2].upper()

    # Latitude bands C-M are southern hemisphere, N-X are northern
    if lat_band >= "N":
        return f"EPSG:326{zone_number:02d}"
    else:
        return f"EPSG:327{zone_number:02d}"


def normalize_opera_timestamps(items: list[Any], *, mid_longitude: float | None = None) -> list[Any]:
    """Stamp OPERA burst granules with noon UTC of the **solar day** they belong to.

    A thin wrapper over
    :func:`~tessera_embeddings.ingest.solar_days.normalize_to_solar_day`, kept because the
    OPERA reason for normalising is its own: a single bbox query returns ~10 burst granules
    per pass, each with a slightly different sub-second timestamp, and ``odc.stac.load``
    would make every burst its own time step instead of mosaicking them. Sharing one
    timestamp is what merges them.

    **This grouped by UTC DATE until 2026-07-30, and that made the whole solar-day
    apparatus on the S1 path inert.** Everything downstream — ownership, footprint
    derivation, the written label — derived its "solar day" from a timestamp that had
    already been flattened to noon of the UTC date, so it recovered the UTC date and
    nothing else. Radar was therefore labelled in UTC while optical was labelled in solar
    days, and in a high-offset zone the same calendar label in the two stores meant
    different 24-hour windows that inference then paired per pixel.
    """
    return normalize_to_solar_day(items, mid_longitude=mid_longitude)


def _granule_to_item(entry: dict[str, Any], skip_counts: Counter[str] | None = None) -> Item | None:
    """Build a pystac ``Item`` from one CMR granule search entry.

    Maps the granule's data download links onto the ``S1_OPERA_BANDS`` asset
    keys (``0_VV``, ``0_VH``) so the constructed item is shape-compatible with
    the CMR-STAC items the rest of the pipeline expects. odc.stac.load reads
    dtype, nodata, and CRS from the COGs themselves (we pass an explicit
    geobox), so only the band hrefs, id, datetime, geometry, and bbox matter.

    Only dual-pol VV+VH granules are accepted — a partial-pol granule would
    otherwise ingest a fabricated all-nodata band that the encoder reads as a
    confident physical signal (see the optional-S1 ADR). The query already
    requires both polarisations server-side, so this is the safety net rather
    than the filter: it catches a granule whose metadata claims the pair and
    whose data links do not carry it. The skip log names what WAS found
    (``VV-only``, cross-pol ``HH/HV``, …); ``skip_counts`` (optional)
    accumulates the same categories for a per-query summary.

    Args:
        entry: One element of ``feed.entry`` from the CMR granule search JSON.
        skip_counts: Optional counter of skip categories, incremented on skip.

    Returns:
        A pystac ``Item``, or ``None`` if the entry lacks BOTH VV and VH data
        links (single-pol granule, EW-mode HH/HV granule, or a non-RTC product
        that slipped through the query).
    """
    # Map band suffix (VV/VH/HH/HV) -> datapool HTTPS href from the data links.
    band_hrefs: dict[str, str] = {}
    for link in entry.get("links", []):
        if not link.get("rel", "").endswith(_CMR_DATA_REL_SUFFIX):
            continue
        match = _BAND_HREF_RE.search(link.get("href", ""))
        if match:
            band_hrefs[match.group(1)] = link["href"]

    # Asset keys follow the CMR-STAC convention: band index 0, suffix VV/VH.
    assets: dict[str, Asset] = {}
    for band_key in S1_OPERA_BANDS:
        suffix = band_key.split("_", 1)[1]  # "0_VV" -> "VV"
        href = band_hrefs.get(suffix)
        if href is not None:
            assets[band_key] = Asset(href=href, roles=["data"])

    if len(assets) < len(S1_OPERA_BANDS):
        pols = sorted(k for k in band_hrefs if k != "mask")
        if {"HH", "HV"} & set(pols):
            # NOT "EW-mode": this said so speculatively for a while and cost real
            # investigation time. Checked against CMR, the Greenland granules carrying HH/HV
            # report BEAM_MODE=IW, and a BEAM_MODE=EW query over that region returns nothing.
            # Cross-pol is a choice of polarisation, not of swath mode — so name only what
            # was observed.
            category = f"{'/'.join(pols)} (cross-pol, no VV+VH)"
        elif pols:
            category = f"{'/'.join(pols)}-only"
        else:
            category = "no data links"
        if skip_counts is not None:
            skip_counts[category] += 1
        # A SAFETY NET, not the primary filter: the query requires POLARIZATION to name BOTH
        # polarisations, so reaching here means a granule whose metadata claims the pair and
        # whose data links do not carry it. A catalogue inconsistency worth seeing, so WARNING —
        # and now genuinely rare. It was not while the query required VV alone: single-pol
        # VV-only granules passed and arrived here in bulk, which is a fixed property of the
        # early archive rather than the upstream change an earlier comment here claimed.
        #
        # The granule response carries no polarisation field at all, so this names what the
        # QUERY required and must not claim to quote the catalogue. See ADR-009.
        logger.warning(
            "Skipping granule %s: published bands are %s, but it matched a query requiring "
            "POLARIZATION to include %s — its metadata and its data links disagree",
            entry.get("title", "<unknown>"),
            pols or "none",
            "+".join(_REQUIRED_POLARIZATIONS),
        )
        return None

    # OPERA polygons are space-separated "lat lon lat lon ..." rings; pystac /
    # odc only need a valid GeoJSON geometry + bbox for spatial bookkeeping.
    polygons = entry.get("polygons")
    geometry = _polygon_to_geojson(polygons[0][0]) if polygons else None
    bbox = _geometry_bbox(geometry) if geometry else None

    return Item(
        id=entry["title"],
        geometry=geometry,
        bbox=bbox,
        datetime=str_to_datetime(entry["time_start"]),
        properties={},
        assets=assets,
    )


def _polygon_to_geojson(ring: str) -> dict[str, Any]:
    """Convert a CMR ``"lat lon lat lon ..."`` ring string to GeoJSON Polygon.

    CMR encodes polygon rings as space-separated lat/lon pairs; GeoJSON wants
    ``[lon, lat]`` coordinate pairs.
    """
    nums = [float(v) for v in ring.split()]
    coords = [[nums[i + 1], nums[i]] for i in range(0, len(nums), 2)]
    return {"type": "Polygon", "coordinates": [coords]}


def _geometry_bbox(geometry: dict[str, Any]) -> list[float]:
    """Compute a ``[minx, miny, maxx, maxy]`` bbox from a GeoJSON Polygon."""
    coords = geometry["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def _query_cmr_granules(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    orbit_direction: str,
) -> list[Item]:
    """Query the CMR Granule Search API for OPERA items in a bbox/date/orbit.

    CMR's ``bounding_box`` is lower-left/upper-right (west,south,east,north) and
    does NOT read a west>east box as antimeridian-crossing — it would match
    nothing. An ROI whose live-tile envelope crosses ±180° (zones 01*/60*) is
    stored with a GeoJSON-style west>east bbox (see ``land_mask.export_zone_roi``),
    so here we split it at the antimeridian into two CMR-valid boxes and merge
    (dedup by granule id). A normal box queries once.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees; west>east = crosses ±180°.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        orbit_direction: "ascending" or "descending".

    Returns:
        List of pystac ``Item`` objects matching the orbit direction.
    """
    west, south, east, north = bbox
    if west > east:
        left = _query_cmr_granules_one((west, south, 180.0, north), start_date, end_date, orbit_direction)
        right = _query_cmr_granules_one((-180.0, south, east, north), start_date, end_date, orbit_direction)
        merged: list[Item] = []
        seen: set[str] = set()
        for item in (*left, *right):
            if item.id not in seen:
                seen.add(item.id)
                merged.append(item)
        logger.info(
            f"CMR granule query ({orbit_direction}): antimeridian bbox split into 2 → "
            f"{len(merged)} unique items ({len(left)}+{len(right)} pre-dedup)"
        )
        return merged
    return _query_cmr_granules_one(bbox, start_date, end_date, orbit_direction)


def _query_cmr_granules_one(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    orbit_direction: str,
) -> list[Item]:
    """Query CMR for OPERA items in ONE (west<=east) bbox, paging all results.

    Pages cleanly at ``_CMR_PAGE_SIZE`` against ``cmr.earthdata.nasa.gov``,
    using the ``CMR-Search-After`` header for pagination and ``attribute[]``
    for server-side orbit-direction filtering.
    """
    # BOTH filters server-side. The polarisation one is what stops us paying to page granules
    # we will always reject: ingest needs dual-pol VV+VH, so a granule whose POLARIZATION does
    # not name both can never qualify, and asking CMR to exclude them discards nothing
    # reachable. A list value is how requests encodes a repeated query parameter — a dict
    # cannot hold two "attribute[]" keys — and repeating the polarisation name is what makes
    # CMR require both rather than either.
    params: dict[str, str | int | list[str]] = {
        "short_name": _CMR_OPERA_SHORT_NAME,
        "provider": _CMR_PROVIDER,
        "bounding_box": ",".join(str(c) for c in bbox),
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "page_size": _CMR_PAGE_SIZE,
        "attribute[]": [
            f"string,ASCENDING_DESCENDING,{orbit_direction.upper()}",
            *(f"string,POLARIZATION,{pol}" for pol in _REQUIRED_POLARIZATIONS),
        ],
    }

    items: list[Item] = []
    headers: dict[str, str] = {}
    skip_counts: Counter[str] = Counter()

    logger.info(f"CMR granule query ({orbit_direction}) starting for {start_date}..{end_date} bbox={bbox}")
    t0 = time.monotonic()
    session = _cmr_session()
    page = 0
    while True:
        page += 1
        resp = session.get(_CMR_GRANULE_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        body = resp.json()

        entries = body.get("feed", {}).get("entry", [])
        if not entries:
            break

        items.extend(item for item in (_granule_to_item(e, skip_counts) for e in entries) if item is not None)
        logger.info(
            f"CMR granule query ({orbit_direction}): page {page} returned {len(entries)} entries "
            f"({len(items)} items so far, {time.monotonic() - t0:.1f}s)"
        )

        search_after = resp.headers.get("CMR-Search-After")
        if search_after and len(entries) == params["page_size"]:
            headers["CMR-Search-After"] = search_after
        else:
            break

    logger.info(
        f"CMR granule query ({orbit_direction}): {len(items)} items across {page} page(s) "
        f"in {time.monotonic() - t0:.1f}s"
    )
    if skip_counts:
        # These SURVIVED the server-side polarisation filter and were still unusable, so the
        # count is no longer a measure of how cross-pol a region is — it is a measure of
        # catalogue inconsistency. While the filter required VV alone, single-pol VV-only
        # granules landed here too and this line read as normal over polar land; requiring both
        # excludes them at the server, so a non-trivial count here is now a genuine anomaly.
        logger.warning(
            "CMR granule query (%s): %d granule(s) passed the %s filter but published no "
            "VV+VH pair: %s — inconsistent metadata upstream, not a regional coverage fact",
            orbit_direction,
            sum(skip_counts.values()),
            "+".join(_REQUIRED_POLARIZATIONS),
            dict(sorted(skip_counts.items())),
        )
    return items


def make_s1_item_provider(
    orbit_direction: str,
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    use_s3_direct: bool = True,
    *,
    mid_longitude: float | None = None,
) -> Callable[[], list[Item]]:
    """Create an item_provider_fn that builds OPERA items from the native CMR granule API.

    The returned callable replaces CMR-STAC ``client.search()`` for the
    ``cmr-asf`` provider — bypassing CMR-STAC's 500-prone cursor pagination
    (nasa/cmr-stac#408) and its 28x latency penalty (#411). It:

    1. **Queries the native CMR Granule Search API**, which filters orbit
       direction server-side via ``attribute[]`` and pages cleanly at 2000.
       Items are built directly from the granule data links — no STAC search
       and no local granule-ID intersection (both were needed only because
       CMR-STAC could not orbit-filter).
    2. **Rewrites asset URLs** — OPERA RTC-S1 asset URLs point to ASF's
       datapool HTTPS endpoints. For in-region access these are rewritten to
       S3 URIs; for out-of-region access they are resolved through ASF's OAuth
       redirect chain to obtain CloudFront signed URLs.
    3. **Normalizes timestamps** — OPERA bursts on the same date have slightly
       different timestamps. These are normalized to noon UTC so that
       odc.stac.load groups them into a single time slice for mosaicking.

    Args:
        orbit_direction: "ascending" or "descending".
        bbox: (west, south, east, north) in WGS84 degrees.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        use_s3_direct: When True (default), rewrite datapool HTTPS URLs to S3
            URIs for direct in-region bucket access. When False, resolve each
            asset through ASF's OAuth redirect chain to obtain a CloudFront
            signed URL.
        mid_longitude: ROI geobox centroid longitude, used to stamp each granule with
            the SOLAR day it belongs to. Supply it whenever the loader groups by solar
            day — which the campaign always does. Omitting it groups by UTC date.

    Returns:
        Zero-argument callable that returns a list of ready-to-load pystac Items.

    Note:
        The returned callable issues a fresh CMR granule query on **every**
        invocation — it does not cache at construction time. The current S1
        flow builds one provider per batch and invokes it once (a single
        ``query_stac_items`` call), so this is fine. If a future caller feeds
        the same provider to both ``has_new_stac_dates`` and
        ``query_stac_items``, it will run two full CONUS-scale CMR queries per
        batch; build a separate provider per call site instead.
    """
    if orbit_direction not in ("ascending", "descending"):
        raise ValueError(f"orbit_direction must be 'ascending' or 'descending', got '{orbit_direction}'")

    def provide_items() -> list[Item]:
        items = _query_cmr_granules(bbox, start_date, end_date, orbit_direction)

        if use_s3_direct:
            for item in items:
                rewrite_assets_to_s3(item, S1_OPERA_BANDS)
        else:
            session = get_edl_session()
            for item in items:
                resolve_item_assets(session, item, S1_OPERA_BANDS)

        normalize_opera_timestamps(items, mid_longitude=mid_longitude)
        return items

    return provide_items
