"""OPERA RTC-S1 query utilities: MGRS bbox, orbit filtering, UTM EPSG.

OPERA RTC-S1 items on CMR-STAC lack MGRS tile IDs, orbit direction,
and projection metadata. This module provides helpers to query by
spatial extent, filter by orbit via the native CMR Granule Search API,
derive the CRS, and prepare items for loading.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import mgrs
import requests

from tessera_embeddings.config import S1_OPERA_BANDS
from tessera_embeddings.ingest.auth import get_edl_session, resolve_item_assets, rewrite_assets_to_s3

logger = logging.getLogger(__name__)

# CMR Granule Search API — used for orbit-direction filtering because
# CMR-STAC silently ignores the ``query`` extension for additional attributes.
_CMR_GRANULE_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
_CMR_OPERA_SHORT_NAME = "OPERA_L2_RTC-S1_V1"
_CMR_PROVIDER = "ASF"


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


def _query_cmr_granule_ids(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    orbit_direction: str,
) -> set[str]:
    """Query CMR Granule Search API for OPERA granule IDs matching a bbox, date range, and orbit direction.

    CMR-STAC silently ignores the ``query`` extension for CMR additional
    attributes like ``ASCENDING_DESCENDING``.  The native CMR Granule Search
    API supports ``attribute[]`` filtering, so we use it to obtain the set
    of granule IDs for the desired orbit in a single paginated request.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        orbit_direction: "ascending" or "descending".

    Returns:
        Set of ``producer_granule_id`` strings matching the orbit direction.
    """
    params: dict[str, str | int] = {
        "short_name": _CMR_OPERA_SHORT_NAME,
        "provider": _CMR_PROVIDER,
        "bounding_box": ",".join(str(c) for c in bbox),
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "page_size": 2000,
        "attribute[]": f"string,ASCENDING_DESCENDING,{orbit_direction.upper()}",
    }

    ids: set[str] = set()
    headers: dict[str, str] = {}

    while True:
        resp = requests.get(_CMR_GRANULE_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        body = resp.json()

        entries = body.get("feed", {}).get("entry", [])
        if not entries:
            break

        ids.update(entry["producer_granule_id"] for entry in entries)

        search_after = resp.headers.get("CMR-Search-After")
        if search_after and len(entries) == params["page_size"]:
            headers["CMR-Search-After"] = search_after
        else:
            break

    logger.debug(f"CMR orbit query ({orbit_direction}): {len(ids)} granules for bbox={bbox}")
    return ids


def make_s1_item_rewriter(
    orbit_direction: str,
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    use_s3_direct: bool = True,
) -> Callable[[list[Any]], list[Any]]:
    """Create an item_filter_fn that filters by orbit, rewrites URLs, and normalizes timestamps.

    The returned callable performs three operations that must happen before
    odc.stac.load attempts to read the COGs:

    1. **Orbit filtering** — CMR-STAC silently ignores the ``query`` extension
       for additional attributes like ``ASCENDING_DESCENDING``. We query the
       native CMR Granule Search API (which supports ``attribute[]`` filtering)
       to obtain valid granule IDs, then filter STAC items by ID match.
    2. **URL rewriting** — OPERA RTC-S1 asset URLs on CMR-STAC point to ASF's
       datapool HTTPS endpoints. For in-region access these are rewritten to
       S3 URIs; for out-of-region access they are resolved through ASF's OAuth
       redirect chain to obtain CloudFront signed URLs.
    3. **Timestamp normalization** — OPERA bursts on the same date have slightly
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
        Callable that filters, rewrites, and normalizes a list of pystac Items.
    """
    if orbit_direction not in ("ascending", "descending"):
        raise ValueError(f"orbit_direction must be 'ascending' or 'descending', got '{orbit_direction}'")

    # Query CMR once at construction time — the returned callable may be
    # invoked more than once (e.g. has_new_stac_dates + query_stac_items).
    orbit_ids = _query_cmr_granule_ids(bbox, start_date, end_date, orbit_direction)

    def filter_rewrite_normalize(items: list[Any]) -> list[Any]:
        items = [item for item in items if item.id in orbit_ids]
        logger.info(f"Orbit filter ({orbit_direction}): kept {len(items)} items")

        if use_s3_direct:
            for item in items:
                rewrite_assets_to_s3(item, S1_OPERA_BANDS)
        else:
            session = get_edl_session()
            for item in items:
                resolve_item_assets(session, item, S1_OPERA_BANDS)

        normalize_opera_timestamps(items)
        return items

    return filter_rewrite_normalize
