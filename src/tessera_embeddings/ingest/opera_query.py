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
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import mgrs
import requests
from pystac import Asset, Item
from pystac.utils import str_to_datetime
from requests.adapters import HTTPAdapter

from tessera_embeddings.config import S1_OPERA_BANDS
from tessera_embeddings.ingest._http import make_logging_retry
from tessera_embeddings.ingest.auth import get_edl_session, resolve_item_assets, rewrite_assets_to_s3

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

# Granule data links carry the per-band COG download URLs. The ``rel`` ends
# in ``/data#`` and the HTTPS href ends in ``_<BAND>.tif`` (BAND ∈ VV/VH/mask).
_CMR_DATA_REL_SUFFIX = "/data#"
_BAND_HREF_RE = re.compile(r"_(VV|VH|mask)\.tif$")


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


def normalize_opera_timestamps(items: list[Any]) -> list[Any]:
    """Normalize OPERA item datetimes so bursts on the same date share a timestamp.

    OPERA RTC-S1 products are individual burst granules. A single MGRS tile
    bbox query returns ~10 bursts per date, each with a slightly different
    sub-second timestamp. When passed to odc.stac.load, each burst becomes
    a separate time step instead of being mosaicked into one.

    This function groups items by date (YYYY-MM-DD) and sets all items in
    each group to noon UTC of that date. odc.stac.load then auto-mosaics
    spatially overlapping items sharing the same timestamp into a single
    time step.

    Args:
        items: List of pystac Items with datetime attributes.

    Returns:
        The same items with datetimes normalized (modified in-place).
    """
    groups: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        date_key = item.datetime.strftime("%Y-%m-%d")
        groups[date_key].append(item)

    for date_str, group in groups.items():
        year, month, day = (int(p) for p in date_str.split("-"))
        canonical = datetime(year, month, day, 12, 0, 0, tzinfo=UTC)
        for item in group:
            item.datetime = canonical

    return items


def _granule_to_item(entry: dict[str, Any]) -> Item | None:
    """Build a pystac ``Item`` from one CMR granule search entry.

    Maps the granule's data download links onto the ``S1_OPERA_BANDS`` asset
    keys (``0_VV``, ``0_VH``) so the constructed item is shape-compatible with
    the CMR-STAC items the rest of the pipeline expects. odc.stac.load reads
    dtype, nodata, and CRS from the COGs themselves (we pass an explicit
    geobox), so only the band hrefs, id, datetime, geometry, and bbox matter.

    Args:
        entry: One element of ``feed.entry`` from the CMR granule search JSON.

    Returns:
        A pystac ``Item``, or ``None`` if the entry lacks the expected VV/VH
        data links (e.g. a non-RTC product slipped through the query).
    """
    # Map band suffix (VV/VH) -> datapool HTTPS href from the data links.
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
        logger.warning("Skipping granule %s: missing VV/VH data links", entry.get("title", "<unknown>"))
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

    Pages cleanly at ``_CMR_PAGE_SIZE`` against ``cmr.earthdata.nasa.gov``,
    using the ``CMR-Search-After`` header for pagination and ``attribute[]``
    for server-side orbit-direction filtering. Returns fully-built pystac
    ``Item`` objects so callers never touch CMR-STAC search.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        orbit_direction: "ascending" or "descending".

    Returns:
        List of pystac ``Item`` objects matching the orbit direction.
    """
    params: dict[str, str | int] = {
        "short_name": _CMR_OPERA_SHORT_NAME,
        "provider": _CMR_PROVIDER,
        "bounding_box": ",".join(str(c) for c in bbox),
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "page_size": _CMR_PAGE_SIZE,
        "attribute[]": f"string,ASCENDING_DESCENDING,{orbit_direction.upper()}",
    }

    items: list[Item] = []
    headers: dict[str, str] = {}

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

        items.extend(item for item in (_granule_to_item(e) for e in entries) if item is not None)
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
    return items


def make_s1_item_provider(
    orbit_direction: str,
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    use_s3_direct: bool = True,
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

        normalize_opera_timestamps(items)
        return items

    return provide_items
