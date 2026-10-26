"""OPERA RTC-S1 query utilities: MGRS bbox, orbit filtering, UTM EPSG, burst fusion order.

OPERA RTC-S1 items on CMR-STAC carry no MGRS tile ID, orbit direction or projection metadata, so
these helpers query by spatial extent, filter orbit via the native CMR Granule Search API, derive
the CRS, and prepare items for loading. They also rank each burst by how often it was acquired over
the ingest window, which decides which burst supplies a pixel two bursts of one solar day cover
(``context_docs/decisions/033-radar-overlaps-go-to-the-most-acquired-burst.md``).
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any, cast
from xml.etree import ElementTree

import mgrs
import requests
from pystac import Asset, Item
from pystac.utils import str_to_datetime
from requests.adapters import HTTPAdapter

from tessera_embeddings.config import S1_OPERA_BANDS
from tessera_embeddings.ingest._http import (
    NonJsonResponseError,
    json_or_raise,
    make_logging_retry,
    spawn_abandonable,
)
from tessera_embeddings.ingest.auth import get_edl_session, resolve_item_assets, rewrite_assets_to_s3
from tessera_embeddings.ingest.solar_days import normalize_to_solar_day, solar_day_of_instant, whole_window_range

logger = logging.getLogger(__name__)

# CMR Granule Search API, queried directly instead of CMR-STAC search: CMR-STAC's cursor
# pagination intermittently 500s at zone scale (nasa/cmr-stac#408) and pages internally at ~100
# items whatever limit is asked (#411), while the native API pages cleanly at 2000 on the same
# host and honours ``attribute[]`` filtering, which CMR-STAC silently ignores — so orbit direction
# is filtered server-side rather than by intersecting a separate STAC result.
_CMR_GRANULE_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
#: The same search in CMR's reference format: each granule's title and concept id and nothing else,
#: a small fraction of the JSON's bytes. All the burst census needs.
_CMR_REFERENCE_URL = "https://cmr.earthdata.nasa.gov/search/granules.xml"
_CMR_OPERA_SHORT_NAME = "OPERA_L2_RTC-S1_V1"
_CMR_PROVIDER = "ASF"
_CMR_PAGE_SIZE = 2000

#: Polarisations required of every granule, filtered server-side. BOTH of them.
#:
#: CMR matches a multi-valued attribute if ANY of its values matches, and ANDs repeated
#: ``attribute[]`` entries on the same attribute name. Naming VV alone is therefore NOT equivalent
#: to requiring the pair — the archive holds genuinely single-polarisation VV-only granules, which
#: contain VV and so pass, and ingest cannot use half a pair. Naming both is EXACT rather than
#: merely tighter: it admits no single-pol granule and drops no dual-pol one, and it costs nothing
#: to page granules the client would reject anyway.
_REQUIRED_POLARIZATIONS = ("VV", "VH")

# Granule data links carry the per-band COG download URLs: ``rel`` ends in ``/data#`` and the
# HTTPS href in ``_<BAND>.tif``. HH/HV are matched for RECOGNITION ONLY, so the skip log can say
# what a rejected granule had; ingest takes dual-pol VV+VH (``S1_OPERA_BANDS``) exclusively.
_CMR_DATA_REL_SUFFIX = "/data#"
_BAND_HREF_RE = re.compile(r"_(VV|VH|HH|HV|mask)\.tif$")

#: An OPERA RTC-S1 granule title, e.g.
#: ``OPERA_L2_RTC-S1_T115-245714-IW1_20240101T015432Z_20240102T091011Z_S1A_30_v1.0``: the burst id
#: (relative orbit, ESA burst number, sub-swath), the acquisition start, then the production time.
#: The burst id's fields are fixed-width, so its string order is relative orbit, then burst number,
#: then sub-swath.
_TITLE_RE = re.compile(
    r"^OPERA_L2_RTC-S1_(?P<burst>T(?P<track>\d{3})-\d{6}-(?P<subswath>IW[123]))_(?P<acquired>\d{8}T\d{6}Z)_"
)

#: Item properties naming the burst a granule images, parsed from its title. Absent where the title
#: does not parse.
BURST_ID = "opera:burst_id"
RELATIVE_ORBIT = "sat:relative_orbit"
SUBSWATH = "opera:subswath"
#: The burst's acquisitions over the whole ingest window, stamped by the item provider from the
#: :class:`BurstCensus`. What :func:`burst_fusion_sort_key` ranks bursts by.
BURST_ACQUISITIONS = "tessera:burst_acquisitions"
#: Root attribute of a radar mosaic store: the :class:`BurstCensus` counts that ranked every date it
#: holds, written in the first date's commit and reused by every resumed leg.
BURST_CENSUS_ATTR = "burst_census"


# CMR intermittently times out or 5xxes under load. urllib3's Retry covers read timeouts (``read``
# defaults to ``total``) as well as the listed statuses, backing off and honouring Retry-After.
_CMR_RETRY = make_logging_retry(
    "CMR",
    total=6,
    backoff_factor=2,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True,
)


#: Attempts at ONE page whose body did not parse, and the seconds between them (times the
#: attempt number). Scoped to that case alone: it is the single failure ``_CMR_RETRY``
#: cannot see, since its ladder keys on status and this response's status is fine.
_UNPARSEABLE_PAGE_ATTEMPTS = 4
_UNPARSEABLE_PAGE_BACKOFF_S = 5.0


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

    A thin wrapper over :func:`~tessera_embeddings.ingest.solar_days.normalize_to_solar_day`, kept
    because the OPERA reason for normalising is its own: one bbox query returns ~10 burst granules
    per pass with slightly different sub-second timestamps, and ``odc.stac.load`` would make each
    burst its own time step instead of mosaicking them. Sharing one timestamp merges them.

    It must be the SOLAR day, not the UTC date. Everything downstream — ownership, footprint
    derivation, the written label — recovers its day from this timestamp, so flattening to the UTC
    date here labels radar in UTC while optical is labelled in solar days; in a high-offset zone
    the same calendar label then means two different 24-hour windows, which inference pairs per
    pixel. See ADR-014 (``solar-offset-applied-only-in-solar-days``).
    """
    return normalize_to_solar_day(items, mid_longitude=mid_longitude)


def _granule_to_item(entry: dict[str, Any], skip_counts: Counter[str] | None = None) -> Item | None:
    """Build a pystac ``Item`` from one CMR granule search entry.

    Maps the granule's data download links onto the ``S1_OPERA_BANDS`` asset keys (``0_VV``,
    ``0_VH``) so the item is shape-compatible with the CMR-STAC items the rest of the pipeline
    expects. odc.stac.load reads dtype, nodata and CRS from the COGs themselves (we pass an
    explicit geobox), so only band hrefs, id, datetime, geometry and bbox matter to the load.

    Two things are kept for the steps before it. ``properties["datetime"]`` holds the raw
    acquisition time, which ``normalize_to_solar_day`` reads and never overwrites. The burst the
    title names (:data:`BURST_ID`, :data:`RELATIVE_ORBIT`, :data:`SUBSWATH`) is what the fusion
    order ranks; a title that does not parse leaves them absent rather than failing, and the item
    sorts after every named burst.

    Only dual-pol VV+VH granules are accepted: a partial-pol granule would ingest a fabricated
    all-nodata band that the encoder reads as a confident physical signal (see the optional-S1
    ADR). The query already requires both polarisations server-side, so this is the safety net —
    it catches a granule whose metadata claims the pair and whose data links do not carry it. The
    skip log names what WAS found (``VV-only``, cross-pol ``HH/HV``, …); ``skip_counts``
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
            # Name only the polarisations observed, never a swath mode: the Greenland granules
            # carrying HH/HV report BEAM_MODE=IW in CMR, and a BEAM_MODE=EW query over that
            # region returns nothing. Cross-pol is a choice of polarisation, not of swath mode.
            category = f"{'/'.join(pols)} (cross-pol, no VV+VH)"
        elif pols:
            category = f"{'/'.join(pols)}-only"
        else:
            category = "no data links"
        if skip_counts is not None:
            skip_counts[category] += 1
        # A SAFETY NET, not the primary filter: the query requires POLARIZATION to name BOTH
        # polarisations, so reaching here means a granule whose metadata claims the pair and whose
        # data links do not carry it — a rare catalogue inconsistency worth seeing, hence WARNING.
        # The granule response carries no polarisation field at all, so this names what the QUERY
        # required and must not claim to quote the catalogue. See ADR-009.
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

    properties: dict[str, Any] = {"datetime": entry["time_start"]}
    if burst := _TITLE_RE.match(entry["title"]):
        properties |= {BURST_ID: burst["burst"], RELATIVE_ORBIT: int(burst["track"]), SUBSWATH: burst["subswath"]}

    return Item(
        id=entry["title"],
        geometry=geometry,
        bbox=bbox,
        datetime=str_to_datetime(entry["time_start"]),
        properties=properties,
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


def _cmr_boxes(bbox: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    """``bbox`` as boxes CMR can read: itself, or its two halves where it crosses ±180°.

    CMR's ``bounding_box`` is lower-left/upper-right (west,south,east,north) and does NOT read a
    west>east box as antimeridian-crossing — it would match nothing. An ROI whose live-tile
    envelope crosses ±180° (zones 01*/60*) is stored with a GeoJSON-style west>east bbox (see
    ``land_mask.export_zone_roi``), so it is queried as the two halves either side of the line.
    """
    west, south, east, north = bbox
    return [(west, south, 180.0, north), (-180.0, south, east, north)] if west > east else [bbox]


def _query_cmr_granules(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    orbit_direction: str,
) -> list[Item]:
    """Query the CMR Granule Search API for OPERA items in a bbox/date/orbit.

    A box crossing ±180° is queried as two halves (:func:`_cmr_boxes`) and merged, deduped by
    granule id, since a granule straddling the line comes back from both. A normal box queries
    once.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees; west>east = crosses ±180°.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        orbit_direction: "ascending" or "descending".

    Returns:
        List of pystac ``Item`` objects matching the orbit direction.
    """
    boxes = _cmr_boxes(bbox)
    if len(boxes) == 1:
        return _query_cmr_granules_one(bbox, start_date, end_date, orbit_direction)
    left, right = (_query_cmr_granules_one(box, start_date, end_date, orbit_direction) for box in boxes)
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


def _granule_params(
    bbox: tuple[float, float, float, float], start_date: str, end_date: str, orbit_direction: str
) -> dict[str, str | int | list[str]]:
    """The search parameters for OPERA granules in ONE (west<=east) box, window and orbit.

    BOTH filters server-side; the polarisation one stops us paying to page granules ingest can
    never use, and excludes nothing reachable. The list value is how requests encodes a repeated
    query parameter (a dict cannot hold two "attribute[]" keys), and repeating the polarisation
    name is what makes CMR require both rather than either.
    """
    return {
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


def _json_entries(resp: requests.Response) -> list[dict[str, Any]]:
    """The granule entries of one JSON page; raises :class:`NonJsonResponseError` if it is not JSON."""
    body = cast(dict[str, Any], json_or_raise(resp, log_body=True))
    return cast(list[dict[str, Any]], body.get("feed", {}).get("entry", []))


def _reference_titles(resp: requests.Response) -> list[str]:
    """The granule titles of one reference-format page; raises ``ParseError`` for any other body.

    The message never quotes the body, for the reason :func:`json_or_raise` gives.
    """
    root = ElementTree.fromstring(resp.content)
    if root.tag != "results":
        raise ElementTree.ParseError(
            f"{resp.url} answered HTTP {resp.status_code} with a body that is not a CMR result"
        )
    return [name.text or "" for name in root.iterfind("references/reference/name")]


def _fetch_granule_page[T](
    session: requests.Session,
    params: dict[str, str | int | list[str]],
    headers: dict[str, str],
    page: int,
    *,
    url: str,
    parse: Callable[[requests.Response], T],
) -> tuple[T, str | None]:
    """Fetch ONE page of granules from ``url``, re-asking when ``parse`` cannot read the body.

    Returns the parsed page and the cursor to carry into the next page.

    Retried here because the session's ladder keys on status and an unparseable body arrives
    under a success status that ``raise_for_status`` passes. Unretried, one such page ends a leg
    holding a zone-year of already-written batches. Safe to repeat: a GET whose cursor the archive
    itself issued names the identical page.
    """
    for attempt in range(1, _UNPARSEABLE_PAGE_ATTEMPTS + 1):
        resp = session.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        try:
            body = parse(resp)
        except (NonJsonResponseError, ElementTree.ParseError) as exc:
            if attempt == _UNPARSEABLE_PAGE_ATTEMPTS:
                raise
            logger.warning(
                "CMR page %d unparseable (attempt %d/%d: %s), re-asking in %.0fs",
                page,
                attempt,
                _UNPARSEABLE_PAGE_ATTEMPTS,
                exc,
                _UNPARSEABLE_PAGE_BACKOFF_S * attempt,
            )
            time.sleep(_UNPARSEABLE_PAGE_BACKOFF_S * attempt)
            continue
        return body, resp.headers.get("CMR-Search-After")
    raise AssertionError("unreachable: the loop either returns or raises")  # pragma: no cover


def _granule_pages[T](
    params: dict[str, str | int | list[str]], *, url: str, parse: Callable[[requests.Response], list[T]]
) -> Iterator[list[T]]:
    """Every page of one search, in order, following the ``CMR-Search-After`` cursor.

    Ends at the first empty page or the first short one, since CMR sends no cursor past the last.
    """
    session = _cmr_session()
    headers: dict[str, str] = {}
    page = 0
    while True:
        page += 1
        entries, search_after = _fetch_granule_page(session, params, headers, page, url=url, parse=parse)
        if not entries:
            return
        yield entries
        if not (search_after and len(entries) == _CMR_PAGE_SIZE):
            return
        headers["CMR-Search-After"] = search_after


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
    params = _granule_params(bbox, start_date, end_date, orbit_direction)
    items: list[Item] = []
    skip_counts: Counter[str] = Counter()

    logger.info(f"CMR granule query ({orbit_direction}) starting for {start_date}..{end_date} bbox={bbox}")
    t0 = time.monotonic()
    page = 0
    for page, entries in enumerate(_granule_pages(params, url=_CMR_GRANULE_URL, parse=_json_entries), start=1):
        items.extend(item for item in (_granule_to_item(e, skip_counts) for e in entries) if item is not None)
        logger.info(
            f"CMR granule query ({orbit_direction}): page {page} returned {len(entries)} entries "
            f"({len(items)} items so far, {time.monotonic() - t0:.1f}s)"
        )

    logger.info(
        f"CMR granule query ({orbit_direction}): {len(items)} items across {page} page(s) "
        f"in {time.monotonic() - t0:.1f}s"
    )
    if skip_counts:
        # These SURVIVED the server-side polarisation filter and were still unusable, so the
        # count measures catalogue inconsistency, not how cross-pol a region is. Single-pol
        # granules are excluded at the server, so a non-trivial count here is a genuine anomaly.
        logger.warning(
            "CMR granule query (%s): %d granule(s) passed the %s filter but published no "
            "VV+VH pair: %s — inconsistent metadata upstream, not a regional coverage fact",
            orbit_direction,
            sum(skip_counts.values()),
            "+".join(_REQUIRED_POLARIZATIONS),
            dict(sorted(skip_counts.items())),
        )
    if unnamed := sum(BURST_ID not in item.properties for item in items):
        logger.warning(
            "CMR granule query (%s): %d granule title(s) name no burst; they sort after every named "
            "burst of their day, so they only fill gaps",
            orbit_direction,
            unnamed,
        )
    return items


def burst_acquisitions(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    orbit_direction: str,
    *,
    mid_longitude: float | None = None,
) -> dict[str, int]:
    """How many times each burst touching ``bbox`` was acquired in the window, by burst id.

    Read from granule titles alone, through CMR's reference format, so a whole window costs a
    small fraction of what the batch queries page through as JSON. Counts distinct ACQUISITIONS,
    not granules: a reprocessed acquisition is published again under the same burst and acquisition
    time with a new production time, and counting granules would rank a burst by how often it was
    reprocessed. A title that names no burst counts toward nothing. Cost and rationale:
    ``context_docs/decisions/033-radar-overlaps-go-to-the-most-acquired-burst.md``.

    ``start_date..end_date`` are SOLAR days at ``mid_longitude``, as the ingest window is: the
    catalogue is asked for a UTC day either side (:func:`~.solar_days.whole_window_range`) and
    each title's acquisition instant is placed on its solar day before it is counted, so an
    acquisition the batches own is counted and one they do not is not. Counted on UTC dates, an
    edge acquisition would be counted or missed by the hour of the pass, and a count off by one
    flips the ties that are the common case. ``None`` counts on UTC dates, as the loader groups
    without an offset.

    A burst's count does not depend on ``bbox`` beyond whether the burst touches it, so a query
    over a subset of an area ranks the bursts it shares with the whole the same way.
    """
    window = whole_window_range(start_date, end_date)
    acquired: dict[str, set[str]] = {}
    titles = unnamed = 0
    t0 = time.monotonic()
    for box in _cmr_boxes(bbox):
        params = _granule_params(box, window.query_start, window.query_end, orbit_direction)
        for page in _granule_pages(params, url=_CMR_REFERENCE_URL, parse=_reference_titles):
            titles += len(page)
            for title in page:
                if burst := _TITLE_RE.match(title):
                    instant = datetime.strptime(burst["acquired"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
                    if window.owns(solar_day_of_instant(instant, mid_longitude=mid_longitude).isoformat()):
                        acquired.setdefault(burst["burst"], set()).add(burst["acquired"])
                else:
                    unnamed += 1
    logger.info(
        "Burst census (%s) %s..%s: %d granule(s), %d burst(s), %d acquisition(s), %d unnamed, in %.1fs",
        orbit_direction,
        start_date,
        end_date,
        titles,
        len(acquired),
        sum(len(times) for times in acquired.values()),
        unnamed,
        time.monotonic() - t0,
    )
    return {burst: len(times) for burst, times in acquired.items()}


class BurstCensus:
    """One window's :func:`burst_acquisitions`, queried once in the background and shared.

    Constructing one queries nothing. The first :meth:`start` sends the query on a daemon thread,
    so the radar ingest can run it alongside its first batch's granule query, the one query its
    batch look-ahead cannot hide; :meth:`counts` waits for the answer, and every later call
    returns the same one. The batches that share it are prepared one at a time, so it takes no
    lock.

    ``counts`` given up front, as read back from a store's :data:`BURST_CENSUS_ATTR`, are the
    answer and nothing is queried: the catalogue keeps changing after a leg's first date commits,
    a backfilled acquisition can flip a tied ranking, and a resumed leg that re-counted would fuse
    its later days by a different burst than the days already written. The census that ranked a
    store's first date ranks every date after it.
    """

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        orbit_direction: str,
        *,
        mid_longitude: float | None = None,
        counts: dict[str, int] | None = None,
    ) -> None:
        self._args = (bbox, start_date, end_date, orbit_direction)
        self._mid_longitude = mid_longitude
        self._waiter: Callable[..., dict[str, int]] | None = None if counts is None else lambda: counts

    def start(self) -> Callable[..., dict[str, int]]:
        """Send the query if nothing has yet, and return the waiter for its answer."""
        if self._waiter is None:
            self._waiter = spawn_abandonable(
                lambda args: burst_acquisitions(*args, mid_longitude=self._mid_longitude), self._args
            )
        return self._waiter

    def counts(self) -> dict[str, int]:
        """Acquisitions per burst id, waiting for the query (and sending it, if nothing has)."""
        return self.start()()


def burst_fusion_sort_key(item: Any) -> tuple[str, bool, int, str, str, str]:  # noqa: ANN401 — any STAC-like item
    """The order a solar day's radar items must be handed to the loader in.

    The loader keeps the FIRST valid source of a solar day and later sources only fill the gaps it
    left, so this order decides which burst supplies a pixel that two bursts cover. The burst
    acquired most often over the whole window comes first. A burst's footprint is fixed, so among
    the bursts covering a pixel, the most-acquired one is the look geometry most of that pixel's
    time series comes from, and putting it first makes contested days agree with it.

    Ties go to the burst id, whose fixed-width fields order it by relative orbit, burst number and
    sub-swath, then to the raw acquisition time and the item id, so the order is total and a rerun
    builds the same mosaic. A title that names no burst sorts after every named one and can only
    fill gaps. Keyed on the SOLAR day, like
    :func:`~tessera_embeddings.ingest.stac.solar_day_sort_key`, which is the optical counterpart.
    See ``context_docs/decisions/033-radar-overlaps-go-to-the-most-acquired-burst.md``.
    """
    properties = item.properties
    burst = properties.get(BURST_ID)
    return (
        item.datetime.strftime("%Y-%m-%d"),
        burst is None,
        -properties.get(BURST_ACQUISITIONS, 0),
        burst or "",
        properties.get("datetime", ""),
        item.id,
    )


def make_s1_item_provider(
    orbit_direction: str,
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    use_s3_direct: bool = True,
    *,
    burst_census: BurstCensus,
    mid_longitude: float | None = None,
) -> Callable[[], list[Item]]:
    """Create an item_provider_fn that builds OPERA items from the native CMR granule API.

    Replaces CMR-STAC ``client.search()`` for the ``cmr-asf`` provider, bypassing its 500-prone
    cursor pagination (nasa/cmr-stac#408) and 28x latency penalty (#411). It:

    1. **Queries the native CMR Granule Search API**, which filters orbit direction server-side
       via ``attribute[]`` and pages cleanly at 2000. Items are built straight from the granule
       data links — no STAC search, no local granule-ID intersection.
    2. **Rewrites asset URLs** from ASF's datapool HTTPS endpoints: to S3 URIs for in-region
       access, or through ASF's OAuth redirect chain to CloudFront signed URLs otherwise.
    3. **Normalizes timestamps** so same-day bursts share one, and odc.stac.load groups them into
       a single time slice for mosaicking.
    4. **Ranks each item's burst**, stamping :data:`BURST_ACQUISITIONS` from ``burst_census`` for
       the fusion sort. The census is started before this query and waited for after it, so the
       two overlap.

    Args:
        orbit_direction: "ascending" or "descending".
        bbox: (west, south, east, north) in WGS84 degrees.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        use_s3_direct: When True (default), rewrite datapool HTTPS URLs to S3
            URIs for direct in-region bucket access. When False, resolve each
            asset through ASF's OAuth redirect chain to obtain a CloudFront
            signed URL.
        burst_census: The acquisitions per burst over the WHOLE ingest window, shared by every
            provider of one run. A burst missing from it ranks as never acquired.
        mid_longitude: ROI geobox centroid longitude, used to stamp each granule with
            the SOLAR day it belongs to. Supply it whenever the loader groups by solar
            day — which the campaign always does. Omitting it groups by UTC date.

    Returns:
        Zero-argument callable that returns a list of ready-to-load pystac Items.

    Note:
        The returned callable issues a fresh CMR granule query on **every** invocation; it does
        not cache at construction. The S1 flow builds one provider per batch and invokes it once.
        Feeding one provider to both ``has_new_stac_dates`` and ``query_stac_items`` would run two
        full zone-scale CMR queries per batch — build a separate provider per call site.
    """
    if orbit_direction not in ("ascending", "descending"):
        raise ValueError(f"orbit_direction must be 'ascending' or 'descending', got '{orbit_direction}'")

    def provide_items() -> list[Item]:
        burst_census.start()
        items = _query_cmr_granules(bbox, start_date, end_date, orbit_direction)

        if use_s3_direct:
            for item in items:
                rewrite_assets_to_s3(item, S1_OPERA_BANDS)
        else:
            session = get_edl_session()
            for item in items:
                resolve_item_assets(session, item, S1_OPERA_BANDS)

        counts = burst_census.counts()
        for item in items:
            if (burst := item.properties.get(BURST_ID)) is not None:
                item.properties[BURST_ACQUISITIONS] = counts.get(burst, 0)

        normalize_opera_timestamps(items, mid_longitude=mid_longitude)
        return items

    return provide_items
