"""STAC data ingestion utilities with baseline correction.

This module provides functions for loading satellite data via STAC catalogs
using odc.stac.load, applying processing baseline corrections, and filtering
by existing dates.

Supports multiple providers (Earth Search, Planetary Computer) and collections
(Sentinel-2 L2A, Sentinel-1 GRD, Landsat).
"""

import datetime
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import count
from typing import Any, final

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox
from pystac import Item
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3.util.retry import Retry

from tessera_embeddings.config import (
    PROVIDERS,
    CollectionConfig,
    STACProvider,
)
from tessera_embeddings.config.environment import configure_gdal_environment
from tessera_embeddings.config.ingest import INGEST_CHUNKS
from tessera_embeddings.ingest._http import make_logging_retry, spawn_abandonable
from tessera_embeddings.ingest.asset_locations import (
    Harmonisation,
    SettledProducer,
    asset_bucket,
    asset_href,
)
from tessera_embeddings.ingest.boa_offset import OffsetDecision, source_decision
from tessera_embeddings.ingest.catalogue_refusal import (
    OVERSIZED_RESPONSE_STATUSES,
    THROTTLE_STATUSES,
    CatalogueRequest,
    bbox_area_label,
    classify_refusal,
    is_oversized_response,
    raise_catalogue_query_error,
)
from tessera_embeddings.ingest.duplicates import log_duplicate_selection, select_preferred_duplicates
from tessera_embeddings.ingest.item_baselines import processing_baseline as _declared_baseline
from tessera_embeddings.ingest.solar_days import (
    SolarDayRange,
    month_ranges,
    normalize_to_solar_day,
    owned_items,
    resolve_grouping_longitude,
)

# =============================================================================
# GDAL/Rasterio Configuration
# =============================================================================
# These must be set BEFORE importing rasterio/odc.stac to take effect.
# They configure network resilience for HTTP requests to cloud storage.

configure_gdal_environment()

import odc.stac  # noqa: E402
from odc.loader import RioDriver, RioReader  # noqa: E402

# Private, and the only private odc import here. `_resolve_driver` uses a driver's own `md_parser`
# when it supplies one, so stamping the per-source offset at parse time means subclassing odc's
# STAC parser. Paid for by what the parser carries instead of a lookup table — 158 bytes against a
# measured 14.6 MB, see `BoaOffsetParser` — and the reason `odc-stac` is pinned rather than
# floating.
from odc.stac._mdtools import StacMDParser  # noqa: E402

# Dimension name mapping: the project uses northing/easting everywhere, but
# odc.stac.load works with y/x internally. This dict is applied to the chunks
# kwarg before passing it to odc.stac.load.
_DIM_TO_ODC = {"northing": "y", "easting": "x"}


def chunks_to_odc(chunks: dict[str, int]) -> dict[str, int]:
    """Translate chunk dimension names from project convention to odc.stac convention.

    Maps ``northing`` → ``y`` and ``easting`` → ``x``. Passes through keys that
    are already in odc convention (``y``, ``x``, ``time``).
    """
    return {_DIM_TO_ODC.get(k, k): v for k, v in chunks.items()}


logger = logging.getLogger(__name__)


#: Statuses every catalogue's page fetches are retried in place for. Provider flags add to
#: and subtract from this — see :func:`_retry_statuses`.
_STAC_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _retry_statuses(provider: STACProvider) -> frozenset[int]:
    """Which statuses this provider's requests are retried IN PLACE for.

    Both adjustments are per provider rather than applied to the session, because each rests
    on a measurement of one catalogue and a catalogue that does not behave that way wants the
    default. 429 and 503 are in every ladder, since those are the refusals where waiting is
    unambiguously the remedy, and connection and read failures keep the ladder either way
    because they carry no status at all.

    **502 comes OUT for a provider that refuses an oversized response with one.** There the
    remedy is to ask a DIFFERENT request — a shorter date window, or fewer items per page — so
    retrying the same one first buys nothing and costs the ladder's whole backoff before the
    remedy can start.

    **403 goes IN for a provider that throttles with it.** Otherwise it reaches the query
    layer on the first refusal, finds no remedy there — asking for less does not help a
    catalogue that is refusing the rate — and fails the whole leg. A leg is hours of work and
    a fleet, so a momentary refusal costs both; the ladder absorbs it instead.
    """
    statuses = _STAC_RETRY_STATUSES
    if provider.refuses_oversized_pages:
        statuses -= OVERSIZED_RESPONSE_STATUSES
    if provider.throttles_with_forbidden:
        statuses |= THROTTLE_STATUSES
    return statuses


def _throttle_statuses(provider: STACProvider) -> frozenset[int]:
    """The statuses that mean "slow down" from THIS provider, for classifying its refusals.

    Read off the same flag the ladder is built from, so a status the ladder waits out cannot
    be classified as something patience does not fix.
    """
    return THROTTLE_STATUSES if provider.throttles_with_forbidden else frozenset()


def _retry_for(provider: STACProvider) -> Retry:
    """The retry ladder this provider's requests should sit behind."""
    return make_logging_retry(
        "STAC",
        total=8,
        backoff_factor=2,
        status_forcelist=tuple(sorted(_retry_statuses(provider))),
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
    )


# (connect_timeout, read_timeout) in seconds. Without an explicit timeout,
# a stalled TCP connection blocks indefinitely and the retry logic never fires.
_STAC_TIMEOUT = (10, 60)

# Items one paginated search is walked for before its window is asked for in shorter
# pieces.
#
# This bounds COST, not correctness, and the distinction matters because the refusal it
# was written for is not a depth limit. Earth Search refuses certain individual page
# requests with a 502, deterministically, as a function of the whole request — pagination
# cursor and date window together — and not of how deep the walk has got: the same refusal
# was observed at page 289 of one window and page 14 of a shorter one sharing its late
# bound. What clears it is a window with a different END, which re-partitioning produces
# and which `_query_stac_items` retries on; see the refusal handler there. What this
# ceiling buys is that a refusal throws away a bounded walk instead of hundreds of pages.
#
# In ITEMS rather than pages, so it keeps its meaning when a provider's page size changes
# and so a denser year is cut into more pieces rather than walked further.
#
# A smaller page IS a real alternative, and the same refusal `max_page_size` was lowered
# for: Earth Search refuses a request whose response would exceed ~6 MB, so whether a given
# hundred items is served depends on how fat those particular hundred are. Re-asking the
# refused page at a smaller size is the most direct remedy and is not used here only because
# `pystac_client` bakes the limit into a search and offers no way to resume from a cursor at
# a different size. Re-cutting the window reaches the same end through the library's own API:
# it regroups the items so no page is fat enough to be refused.
# Measurements in context_docs/design/ingest_optimization_campaign_2026_07.md.
_MAX_QUERY_ITEMS = 10_000

# Date windows of ONE query walked at the same time.
#
# The worklist's windows are INDEPENDENT searches — each its own paginated walk with its
# own cursor — so the only thing that serialised them was the loop. What they spend their
# time on is the catalogue thinking, not the wire: a 100-item page of `sentinel-2-l2a`
# takes ~1.2 s from the end of the request to the first byte of the response, against a
# 12-36 ms round-trip to the same host and ~0.2 s to stream the gzipped body — and ~0.8 s
# of that 1.2 s is the catalogue's own per-item work, measured as a 7.8 ms/item slope
# against page size. So a walk is idle for almost all of its wall clock, overlapping walks
# is the only lever that moves it, and moving the client into the catalogue's own region is
# NOT one: the round-trip it removes is a few percent of a page.
#
# Bounded, and deliberately low. The campaign runs tens of cells against this one provider
# at once, so this is a MULTIPLIER on the concurrent search streams Element 84 sees, and
# what they answer an overload with — 429, 503 — is a load refusal the whole leg then waits
# out. Measurements, including per-page latency against concurrency, are in
# context_docs/design/ingest_optimization_campaign_2026_07.md.
_QUERY_WINDOW_WORKERS = 6

# Times one query may answer a refusal by replacing a window before it gives up.
#
# Bounds AMPLIFICATION, and the bound is the point. Each lever is individually terminating —
# windows bottom out at a day, pages at `_MIN_PAGE_SIZE` — but a refusal that keeps arriving,
# a real gateway outage rather than the size cap, makes both recurse across the whole window.
# Measured against a stub that refuses every walk past page 1: a one-month query fires 328
# requests before giving up, a one-year query 3,648, and nine years 32,868. That is a lot of
# traffic aimed at a service already answering with 5xx.
#
# Counts refusal responses ONLY, not the proactive cut off `numberMatched`, which is bounded by
# the item count and is the ordinary path. Twenty is roughly six times the most any real query
# has needed — the benchmark month uses three — so this cannot refuse legitimate work while it
# holds a month to about twenty extra searches instead of a hundred and sixty.
_MAX_REFUSAL_RE_PARTITIONS = 20


# Smallest page a refused window is re-asked with before the refusal is reported.
#
# Earth Search refuses a request whose RESPONSE would exceed about 6 MB, so asking for fewer
# items per page is the direct remedy — and the only one for the two cases the window re-cut
# cannot reach: a FIRST page, which a shorter window asks identically, and a single-day
# window, which is the re-cut's floor. Both of those fail a leg outright without this.
#
# Halved each time rather than eased down, because a threshold converges and a curve invites
# tuning: 100 -> 50 -> 25 -> 12 stops here. The floor exists so a window that cannot be
# served at any size is reported rather than retried forever; it is not a size anything is
# expected to need. A page this small is ~0.7 MB against the cap.
_MIN_PAGE_SIZE = 10

# Attempts, and the pause between them, for the catalogue ROOT.
#
# The root is a small static document. It cannot be made smaller, so neither a shorter window nor
# a smaller page is any use to it, and taking 502 out of this provider's ladder therefore left the
# root with no retry at all. Worse than one wasted attempt: the leg-retry layer above declares a
# refusal deterministic once it sees the identical signature twice, so a gateway blip on the root
# could END a leg rather than delay it. This restores exactly what the ladder stopped giving —
# nothing else is retried here, because everything else has a remedy of its own.
_ROOT_OPEN_ATTEMPTS = 3
_ROOT_OPEN_BACKOFF_S = 2.0


def _get_provider_config(provider_name: str) -> STACProvider:
    """Get a provider configuration by name.

    Args:
        provider_name: Provider identifier (e.g., "earth-search")

    Returns:
        STACProvider configuration

    Raises:
        ValueError: If provider not found
    """
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")
    return PROVIDERS[provider_name]


def _get_collection_config(
    provider_name: str,
    collection_alias: str,
) -> CollectionConfig:
    """Get collection configuration for a provider.

    Args:
        provider_name: Provider identifier (e.g., "earth-search")
        collection_alias: Collection alias (e.g., "sentinel-2-l2a")

    Returns:
        CollectionConfig for the specified collection

    Raises:
        ValueError: If provider or collection not found
    """
    provider = _get_provider_config(provider_name)
    if collection_alias not in provider.collections:
        available = ", ".join(provider.collections.keys())
        raise ValueError(
            f"Unknown collection '{collection_alias}' for provider '{provider_name}'. Available: {available}"
        )
    return provider.collections[collection_alias]


def split_antimeridian_bbox(
    bbox: tuple[float, float, float, float] | None,
) -> list[tuple[float, float, float, float] | None]:
    """Split an antimeridian-crossing bbox into catalog-safe halves.

    A ``west > east`` box is the GeoJSON/STAC convention for "crosses +/-180",
    which :func:`~tessera_embeddings.ingest.land_mask._live_tile_bbox_wgs84`
    emits for the UTM zones that snap just past the antimeridian (zones 01 and 60) —
    without it, plain min/max would produce a box spanning nearly the globe.

    Forwarding that tuple to a catalog search is a gamble on the server reading
    it as a crossing rather than as an empty or globe-spanning range, and the
    two failure modes are opposite: no dates at all, or an unbounded item set.
    Splitting at +/-180 into two ordinary west-to-east boxes is correct either
    way, so this does not depend on any server's interpretation. The native
    CMR/S1 path already does this (``opera_query._query_cmr_granules``, where
    CMR is known to reject the crossing form); this is the STAC counterpart, and
    it is deliberately at the shared query chokepoint so the single-ROI and
    campaign paths, streamed and unstreamed, all get it.

    Returns a single-element list for an ordinary (or absent) bbox, so callers
    loop unconditionally.
    """
    if bbox is None or bbox[0] <= bbox[2]:
        return [bbox]
    west, south, east, north = bbox
    logger.info("Antimeridian bbox %s split into (%s..180) and (-180..%s)", bbox, west, east)
    return [(west, south, 180.0, north), (-180.0, south, east, north)]


def _build_stac_query(
    collection_config: CollectionConfig,
    tile_id: str | None,
    start_date: str,
    end_date: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Build STAC query parameters for a tile and date range.

    No cloud cover filtering is applied - the cloud mask model handles
    cloud classification at the pixel level.

    When tile_id is provided and the collection has a tile_id_property,
    queries by that property. When tile_id is None or tile_id_property is
    None (e.g., OPERA RTC-S1 on CMR-STAC), falls back to bbox-based
    spatial query.

    Args:
        collection_config: Collection configuration
        tile_id: Tile identifier (e.g., "33UUP" for S2 MGRS), or None
              for ROI-based bbox queries.
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326.
              Required when tile_id is None or tile_id_property is None.

    Returns:
        Dict of query parameters for pystac_client.Client.search()
    """
    query_params: dict[str, Any] = {
        "collections": [collection_config.collection_id],
        "datetime": f"{start_date}/{end_date}",
    }

    if tile_id is not None and collection_config.tile_id_property is not None:
        # Standard property-based query (S2, Landsat)
        tile_query_value = f"{collection_config.tile_id_prefix}{tile_id}"
        query_params["query"] = {collection_config.tile_id_property: {"eq": tile_query_value}}
    elif bbox is not None:
        # Spatial query fallback (OPERA RTC-S1 on CMR-STAC, or ROI-based)
        query_params["bbox"] = bbox
    else:
        raise ValueError(
            f"Collection '{collection_config.collection_id}' has no tile_id_property "
            f"and no bbox was provided. Supply a bbox for spatial querying."
        )

    return query_params


def _extract_baseline(item: Item) -> int:
    """The baseline an item REPORTS, as an integer. A parser, and nothing more.

    Deliberately free of the harmonisation policy: whether that baseline is owed a correction
    depends on which producer served the pixels, and that belongs in
    :func:`extract_baselines`, which is where the per-date decision is made. Keeping this a
    parser is what lets it be tested as one.

    Args:
        item: pystac Item or similar object with properties dict

    Returns:
        Baseline as integer (e.g., "04.00" -> 400, "05.10" -> 510)
        Returns 0 if baseline property is not found.
    """
    declared = _declared_baseline(item)
    return 0 if declared is None else declared


def solar_day_sort_key(item: Any) -> tuple[str, float, str]:  # noqa: ANN401 — any STAC-like item
    """The order a solar day's items must be handed to the loader in.

    The loader's fuser keeps the FIRST valid source of a group and later sources only fill the
    gaps it left, so this order decides which scene supplies a contested pixel. Clearest first:
    the clearest scene wins the ground it covers, and its holes fall through to the next-clearest
    rather than to nothing.

    ``100`` is a real cloud reading, so a missing one sorts after every measured value instead of
    borrowing it — unknown never displaces measured. ``id`` settles what is left, which would
    otherwise be settled by whatever order the supplier happened to produce, making output pixels
    a function of how the query was partitioned.

    **Both ingest paths sort on this function.** They each built their own key once, and the defect
    that replaced was the two of them disagreeing about the direction.
    """
    cloud = item.properties.get("eo:cloud_cover")
    return (
        item.datetime.strftime("%Y-%m-%d"),
        float("inf") if cloud is None else float(cloud),
        item.id,
    )


def extract_baselines(items: list[Any]) -> dict[str, int]:
    """The processing baseline to apply per date, over exactly ``items``.

    FIRST item of a date wins, which under both ingest paths' clearest-first sort is the scene
    that actually supplied most of the day's pixels — odc's fuser keeps the first valid source and
    later ones only fill its gaps. Keeping the LAST item instead named the CLOUDIEST scene, the one
    contributing least. Where a date's tiles disagree the pick is a property of the caller's sort
    order either way, which is why this is derived from the items being LOADED rather than computed
    once per query: a map built over a wider list records a baseline belonging to an item the loader
    never sees, so the provenance would describe imagery that is not there. It selects nothing —
    the correction is decided per source as each source is read.

    **A date whose tiles declare DIFFERENT baselines is described by one of them.** Such a date
    loads: the correction is decided per source, so a day holding 00.01 and 05.00 imagery needs no
    single answer for its pixels. This map still holds one integer per date, and it is the FIRST
    item's — the clearest tile on both ingest paths, because the query sorts a date's items
    cloud-ascending, and so the scene that supplied most of the day's pixels. The day's other
    vintages are recorded nowhere.

    Documented rather than widened because nothing reads this. It is written to the store's root
    attrs and merged forward on append; no code here consumes a value, and no correction, mask or
    coverage decision derives from one. Naming every baseline a date carries would change the
    attribute's type on newly written dates while existing ones stay integers — a cost paid by
    whoever eventually reads it, for a record nobody reads yet.

    **Reports what each item DECLARES, and nothing about whether a correction is owed.** The
    returned map is provenance: it is carried through ``_PreparedDate`` into the store's
    ``baselines_applied`` attribute, whose contract is the processing baseline of the item
    actually loaded. Encoding a correction decision here — writing 0 for an item that is
    already harmonised — made the store misreport its own vintage, which is worse than the
    error it was avoiding because it cannot be recovered afterwards. The decision lives in
    :func:`~tessera_embeddings.ingest.boa_offset.source_decision`, per asset — so a date can hold
    items handled differently while this map still names one baseline per date, which is exactly
    the separation that keeps provenance honest.

    Args:
        items: STAC items, in the order they will be loaded.

    Returns:
        Dict mapping date strings (YYYY-MM-DD) to the baseline each date's item REPORTS.
    """
    baselines: dict[str, int] = {}
    for item in items:
        # `setdefault`, not assignment: the FIRST item of a date is the one to keep. See above.
        baselines.setdefault(item.datetime.strftime("%Y-%m-%d"), _extract_baseline(item))
    return baselines


def selection_read_keys(config: CollectionConfig, extra_bands: "list[str] | None" = None) -> tuple[str, ...]:
    """The asset set duplicate selection may judge readability and locality over.

    Empty where the collection's configured names are not its asset keys: Planetary Computer serves
    the same bands as ``B02`` and ``SCL`` and relies on the loader's alias table, so looking those
    names up directly reports every copy incomplete and remote — worse than not asking, because an
    actually asset-incomplete copy could then win on the terms that remain. An empty set makes both
    terms tie, so they decide nothing.
    """
    if not config.band_names_are_asset_keys:
        return ()
    return _requested_assets(config, extra_bands)


def collection_harmonisation(config: CollectionConfig) -> SettledProducer | None:
    """The producer state the COLLECTION settles, or ``None`` where it does not settle one.

    ``None`` for two different reasons that need the same handling. A collection whose
    harmonisation varies by item has no collection-wide answer by definition. And a collection with
    no correction threshold is owed no offset at all, so naming a producer for it would be a claim
    about data nobody corrects — Landsat is not "harmonised", it is unrelated.

    Otherwise the configuration already answers it: a correction threshold on a collection whose
    producer cannot vary says every item is unharmonised, which is what the threshold exists to
    correct. Supplying that answer is what lets a provider serving its bands under native asset
    keys (``B02``, ``SCL``) be judged from its items at all, since a per-item read looks assets up
    by the names in ``bands`` and finds nothing there.
    """
    if config.harmonisation_varies_by_item or not config.requires_baseline_correction:
        return None
    return Harmonisation.RAW


class HeterogeneousProducerError(RuntimeError):
    """A source's producer or baseline cannot be determined, so its offset is unknowable.

    Deterministic: re-dispatching cannot change what the catalogue says about an item, so this is
    not something a retry ladder should absorb — and it is not a read failure, so the fallback
    ladder must not be handed a copy that raises it.

    Raised per SOURCE now rather than per solar day. The correction is applied to each image as it
    is read, so a day mixing producers is decided tile by tile and no longer has to be refused as a
    whole; what survives is a copy nobody can classify, which is a property of that copy alone.
    """


def _filter_existing_dates(
    items: list[Any],
    existing_dates: set[str],
) -> list[Any]:
    """Filter STAC items to exclude dates already in the store.

    Items must already be normalised by
    :func:`~tessera_embeddings.ingest.solar_days.normalize_to_solar_day`, which happens at
    the query chokepoint. No offset is applied here, deliberately — that function is the
    single place it is applied, and a second application would silently shift the key.

    Both sides of the comparison are therefore SOLAR days: the store's dates are the days
    it was written under. Keying this on the UTC date instead matched only the items on the
    near side of midnight, so the rest of an already-committed group survived the filter,
    loaded, regrouped onto the same solar day and was written a second time — reachable
    only in the far-eastern and far-western zones, which is exactly where a group splits.

    Args:
        items: Solar-day-normalised pystac Items.
        existing_dates: Set of date strings (YYYY-MM-DD) already processed.

    Returns:
        Filtered list of items not in existing_dates
    """
    filtered = []
    for item in items:
        try:
            date_str = item.datetime.strftime("%Y-%m-%d")
            if date_str not in existing_dates:
                filtered.append(item)
            else:
                logger.debug(f"Skipping {date_str} - already in store")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Could not parse date from item {getattr(item, 'id', 'unknown')}: {e}")
            continue

    skipped = len(items) - len(filtered)
    if skipped > 0:
        logger.info(f"Filtered out {skipped} items already in store")

    return filtered


# =============================================================================
# odc.stac.load Wrapper
# =============================================================================


def normalize_odc_dims(ds: xr.Dataset) -> xr.Dataset:
    """Rename y/x → northing/easting and drop spatial_ref from an odc.stac.load result.

    odc.stac.load always emits y/x dimensions; this normalises them to the
    project-wide northing/easting convention. Also drops the ``spatial_ref``
    variable (data_var or coordinate, depending on odc version) since CRS
    metadata is carried via ``proj:`` convention attributes instead.
    """
    rename_map = {old: new for old, new in {"y": "northing", "x": "easting"}.items() if old in ds.dims}
    if rename_map:
        ds = ds.rename(rename_map)
    ds = ds.drop_vars("spatial_ref", errors="ignore")
    return ds


def _load_from_stac(
    items: list[Any],
    collection_config: CollectionConfig,
    bbox: tuple[float, float, float, float] | None = None,
    chunks: dict[str, int] | None = None,
    extra_bands: list[str] | None = None,
    resampling: str = "bilinear",
    crs: str | None = None,
    groupby: str = "solar_day",
    resolution: int | None = None,
    geobox: GeoBox | None = None,
    driver: Any = None,  # noqa: ANN401 — odc's ReaderDriverSpec
) -> xr.Dataset:
    """Load satellite data from STAC items using odc.stac.load.

    This function wraps odc.stac.load with sensible defaults for the pipeline.
    Resampling to target resolution happens during load (via COG overviews),
    not as a post-processing step.

    Args:
        items: List of pystac Items to load
        collection_config: Collection configuration with bands and resolution
        bbox: Optional bounding box (minx, miny, maxx, maxy) in EPSG:4326
              to spatially subset the load. Ignored when geobox is provided.
        chunks: Optional chunk sizes for dask. Defaults to INGEST_CHUNKS.
              Accepts both project convention (northing/easting) and odc
              convention (y/x) — translated automatically.
        extra_bands: Additional bands to load alongside the collection's primary
              bands (e.g., ["scl"] for S2 Scene Classification Layer). Extra bands
              always use nearest-neighbor resampling regardless of the resampling
              parameter.
        resampling: Resampling method for primary bands. Default "bilinear".
              When extra_bands are provided and resampling is not "bilinear", a
              per-band resampling dict is built automatically.
        crs: Explicit CRS for odc.stac.load (e.g., "EPSG:32633"). Required for
              collections whose STAC items lack the proj extension (e.g., OPERA
              RTC-S1 on CMR-STAC). Default None uses the CRS from STAC items.
              Ignored when geobox is provided.
        groupby: How to group STAC items into time slices. Must be "solar_day",
              which groups by solar day from longitude, merging same-pass tiles
              from adjacent orbits into one mosaic — the convention this package
              uses everywhere, for both sensors. Anything else is rejected; see
              below for why the alternative cannot work here.
        resolution: Override pixel resolution in metres. When None (default),
              uses collection_config.resolution. Ignored when geobox is provided.
        geobox: Optional odc.geo.geobox.GeoBox specifying the exact output
              grid (CRS, transform, shape). When provided, overrides bbox,
              crs, and resolution — the output will match this grid exactly.
        driver: Optional odc reader driver. Used to remove the Sentinel-2 BOA offset from each
              image AS IT IS READ, before resampled sources are fused into one solar-day
              mosaic — see :class:`_BoaCorrectingDriver`. ``None`` leaves odc's default rio
              driver in place, which is every collection owed no offset.

    Returns:
        xarray Dataset with bands as variables and (time, northing, easting) dimensions
    """
    if not items:
        raise ValueError("No items to load")
    # The load is the LAST place that could still honour an exact timestamp, and by then
    # there is none left to honour: `query_stac_items` stamps every item to noon of its
    # solar day, once, as the package's single application of the offset. Grouping those
    # items by "time" therefore does not preserve separate same-day acquisitions — it
    # collapses them, exactly as "solar_day" would, while reporting a different
    # convention. Refuse rather than quietly agree.
    if groupby != "solar_day":
        raise ValueError(
            f"groupby={groupby!r} is not supported: items reaching the loader have already been "
            "normalised to noon of their solar day, so no exact acquisition timestamp survives to "
            "group on. Use groupby='solar_day'."
        )

    load_chunks = chunks_to_odc(chunks if chunks is not None else INGEST_CHUNKS)

    # Merge extra bands with primary bands
    all_bands = list(collection_config.bands)
    if extra_bands:
        all_bands.extend(extra_bands)

    # Build per-band resampling dict if extra bands need different resampling
    if extra_bands and resampling != "nearest":
        resampling_dict: dict[str, str] = dict.fromkeys(collection_config.bands, resampling)
        resampling_dict.update(dict.fromkeys(extra_bands, "nearest"))
        effective_resampling: str | dict[str, str] = resampling_dict
    else:
        effective_resampling = resampling

    load_kwargs: dict[str, Any] = {
        "bands": all_bands,
        "resampling": effective_resampling,
        "chunks": load_chunks,
        "groupby": groupby,
        # True only where THIS package decided the order. `query_stac_items` sorts a collection
        # carrying an SCL band with `solar_day_sort_key`, and the fuser keeps the first valid
        # source, so that order has to survive into the loader. Nothing here sorts any other
        # collection, and for those odc's own default — `(time, id)` within a group — is what makes
        # the fused result a function of the items rather than of the order the provider returned.
        #
        # TODO: Planetary Computer's `sentinel-2-l2a` config does not set `has_scl`, so this reads
        # False there and odc reorders the group by `(time, id)` — and because normalisation gives a
        # group one shared timestamp, `id` rather than cloud cover then decides an overlap, losing
        # `s2_roi`'s clearest-first sort. OUT OF SCOPE: the campaign reads Earth Search only. The
        # fix is to key this on whether the CALLER imposed an order, not on a collection capability.
        "preserve_original_order": collection_config.has_scl,
    }

    if geobox is not None:
        load_kwargs["geobox"] = geobox
    else:
        effective_resolution = resolution if resolution is not None else collection_config.resolution
        load_kwargs["resolution"] = effective_resolution
        if bbox is not None:
            load_kwargs["bbox"] = bbox
        if crs is not None:
            load_kwargs["crs"] = crs

    if driver is not None:
        load_kwargs["driver"] = driver

    res_label = f"geobox {geobox.shape}" if geobox is not None else f"{load_kwargs.get('resolution')}m"
    logger.info(f"Loading {len(items)} items with odc.stac.load (bands={all_bands}, resolution={res_label})")

    ds = odc.stac.load(items, **load_kwargs)
    ds = normalize_odc_dims(ds)

    return ds


#: The Sentinel-2 nodata code. The one DN that carries no BOA offset, at any baseline.
_NODATA = 0

#: The lowest code a VALID corrected pixel may take. Not zero, because zero is nodata: flooring a
#: real dark pixel there makes it indistinguishable from no observation at all, and every
#: downstream mask would drop it. Measured against Element 84's harmonised COGs, which do the
#: same — see ``context_docs/decisions/020-boa-offset-applies-to-every-valid-dn.md``.
_MIN_VALID_REFLECTANCE = 1


def correct_boa_dn(pixels: np.ndarray, offset: int) -> np.ndarray:
    """Remove the BOA offset from one source's pixels, returning the input dtype.

    **The offset applies to every valid DN, not only to bright ones.** ESA's baseline 04.00
    products add it across the whole reflectance range precisely so that negative surface
    reflectance — routine over water and deep shadow — can be encoded in an unsigned type. Leaving
    DN 1-999 alone left those pixels up to 999 too high and made a corrected raw copy disagree with
    a harmonised one across every dark surface.

    **The floor is 1, not 0**, and this is the part no amount of reading settles. Flooring at zero
    is what the widely circulated harmonisation snippets do, and it is wrong here because zero is
    the nodata code: it converts a real dark observation into no observation, and every downstream
    mask then drops it. Element 84 floors at 1. Established by measuring their COGs, not by
    argument — see ``context_docs/decisions/020-boa-offset-applies-to-every-valid-dn.md``.

    Nodata is the caller's business, not this function's: the reader applies the result only where
    the source was valid, so DN 0 never reaches the arithmetic.

    Args:
        pixels: one source's pixels, as read. Any integer dtype.
        offset: the value to add, negative for a correction.

    Returns:
        A new array of the input dtype. Nothing here can exceed the input's range: the offset is
        negative and the floor is positive, so an unsigned input stays representable.
    """
    # Widened for the arithmetic: adding a negative Python int to a uint16 array raises
    # OverflowError under numpy 2.
    shifted = pixels.astype(np.int32) + offset
    return np.clip(shifted, _MIN_VALID_REFLECTANCE, None).astype(pixels.dtype)


def _reflectance_asset_keys(items: list[Any], collection_config: CollectionConfig) -> frozenset[str]:
    """The ASSET KEYS serving this collection's reflectance bands, through odc's alias table.

    Not the configured band names, and the difference decides whether the correction fires at all.
    Earth Search names its assets after the bands (``blue``), so the two coincide; Planetary
    Computer serves ``blue`` as an asset called ``B02`` and relies on odc resolving the ``eo:bands``
    common name. Deciding the offset against configured names would therefore match none of that
    provider's assets and correct nothing, silently — which is why the resolution is odc's own
    rather than a mapping maintained here.

    Resolved from ONE item rather than from all of them. ``extract_collection_metadata`` builds the
    alias table from an item's assets, and the assumption that assets of one name share a structure
    across a collection is odc's own — the same one the load itself relies on. Doing it per item
    would pay a full parse twice over.

    ``scl`` is excluded by construction: only ``collection_config.bands`` is resolved, and the
    scene classification layer is not among them. It is read, and it counts for locality, but it is
    categorical and never corrected.

    Falls back to the configured band names when odc's own resolution yields nothing, which is
    correct wherever ``band_names_are_asset_keys`` is set and is what every other asset lookup on
    this path already assumes. Where that flag is NOT set the fallback is known to be wrong, so the
    caller refuses instead of correcting nothing — see :func:`load_stac_items`.

    Returns:
        The resolved asset keys, or an empty set when nothing could be resolved.
    """
    for item in items:
        try:
            metadata = odc.stac.extract_collection_metadata(item)
        except Exception as exc:
            # Deliberately broad. This is a best-effort read of a third party's metadata over items
            # from a catalogue, and every failure has the same answer: try the next item, then fall
            # back. A narrower list would turn one unforeseen shape into a failed ingest.
            logger.debug("Could not read collection metadata from %s: %s", getattr(item, "id", "?"), exc)
            continue
        resolved: set[str] = set()
        for band in collection_config.bands:
            try:
                asset_name, _ = metadata.band_key(band)
            except ValueError:
                # A band the items do not carry. Not fatal here: `odc.stac.load` resolves the same
                # names and raises its own "No such band/alias" for an asset-incomplete item, which
                # `s2_roi` recognises and recovers from by stepping down to another copy. Refusing
                # here would pre-empt that recovery with an error it does not know.
                logger.debug("Band %r resolves to no asset on %s", band, getattr(item, "id", "?"))
                continue
            resolved.add(asset_name)
        if resolved:
            return frozenset(resolved)
    return frozenset()


#: Key under which the per-source offset rides on ``RasterSource.driver_data``.
#:
#: An explicit sentinel rather than a bare number, and its ABSENCE is meaningful: a reflectance
#: source always carries this key — ``0`` where nothing is owed — while a source that is not
#: reflectance carries no decision at all. That is what lets the reader tell "decided, owes
#: nothing" apart from "never asked", and it is the difference between a correction that is
#: correctly inert and one that silently never fired.
_OFFSET_KEY = "boa_offset"


class BoaOffsetParser(StacMDParser):
    """Stamps every reflectance source with the offset it is owed, at parse time.

    **Why the metadata parser and not a lookup table on the driver.** The driver is embedded in
    every dask task, and ``distributed`` serialises tasks individually rather than sharing
    references between them — so a ``(uri, band) -> offset`` table on the driver is duplicated
    once per task. Measured at ROI scale: a 3,081-entry table cost **14.6 MB** of graph across 64
    chunk tasks, where this parser costs **158 bytes** and does not grow with the item count. The
    decision instead rides on ``RasterSource.driver_data``, which is part of the source object the
    graph was already carrying.

    **It also removes a whole class of bug.** A table has to be keyed on the uri odc will read,
    and every way of getting that key wrong — href resolution, a signed url, a future
    ``patch_url`` — produces a table that matches nothing, corrects nothing, and says nothing. Here
    the decision is computed from the item odc is parsing, so there is no key to get wrong.

    **Refusals happen here, before any graph exists.** ``odc.stac.load`` parses items
    synchronously on the calling thread, so raising from :meth:`driver_data` surfaces as an
    exception out of the load rather than as a task failure hours later.
    """

    def __init__(
        self,
        cfg: Any,  # noqa: ANN401 — odc's ConversionConfig
        reflectance_assets: frozenset[str],
        threshold: int,
        offset: int,
        known_harmonisation: SettledProducer | None,
    ) -> None:
        """Build a parser that decides the offset for ``reflectance_assets``.

        Args:
            cfg: odc's conversion config, passed through to :class:`StacMDParser`.
            reflectance_assets: the ASSET KEYS of the reflectance bands, already resolved through
                odc's alias table. Not the configured band names: Planetary Computer serves
                ``blue`` as an asset called ``B02``, so a set of configured names would match none
                of its assets and every correction would silently go missing.
            threshold: baselines at or above this carry the offset.
            offset: the value to add, negative for a correction.
            known_harmonisation: the producer for every source, where the collection settles it.
        """
        super().__init__(cfg)
        #: The asset keys this parser will decide for. Public because it is the thing that goes
        #: wrong when a correction silently reaches nothing, so it is what a test asserts on.
        self.reflectance_assets = reflectance_assets
        self._threshold = threshold
        self._offset = offset
        self._known = known_harmonisation
        #: How many reflectance sources were stamped, and how many were owed the offset. Read by
        #: the caller AFTER the load, as the guard that this parser was wired to real assets at
        #: all — the failure mode a table would have had is still reachable through an empty or
        #: mistaken ``reflectance_assets``.
        self.stamped = 0
        self.owed = 0

    def driver_data(self, md: Any, band_key: tuple[str, int]) -> Any:  # noqa: ANN401 — pystac Item
        """The offset owed to one source, or the inherited value where none is owed.

        Raises:
            HeterogeneousProducerError: this source's producer or baseline cannot be determined
                and it is at or above the threshold, so whether its pixels carry the offset is
                unknowable. Correcting and exempting are wrong by the same amount in opposite
                directions, and both silent.
        """
        asset_name, _ = band_key
        if asset_name not in self.reflectance_assets:
            # SCL and odc's own auxiliary bands. SCL is categorical and never corrected —
            # subtracting the offset from a class label is meaningless — so it carries no decision
            # rather than a decision of zero.
            return super().driver_data(md, band_key)

        asset = getattr(md, "assets", {}).get(asset_name)
        href = asset_href(asset) if asset is not None else None
        decision = source_decision(
            asset_bucket(href) if href is not None else None,
            _declared_baseline(md),
            self._threshold,
            self._known,
        )
        if decision is OffsetDecision.UNDECIDABLE:
            raise HeterogeneousProducerError(
                f"{getattr(md, 'id', '?')}: asset {asset_name!r} is at or above baseline "
                f"{self._threshold} and its producer cannot be determined, so whether its pixels "
                f"already carry the {abs(self._offset)} offset is unknowable. Either its bucket is "
                f"classified as neither harmonised nor unharmonised — add it to the appropriate set "
                f"in `asset_locations` — or the item declares no readable `s2:processing_baseline`, "
                f"which is a catalogue defect to fix rather than to guess around."
            )
        self.stamped += 1
        if decision is OffsetDecision.OWED:
            self.owed += 1
        return {_OFFSET_KEY: self._offset if decision is OffsetDecision.OWED else 0}


class _BoaCorrectingReader(RioReader):
    """One source's reader, with the BOA offset removed from what it returns.

    Subclasses odc's rio reader and overrides ``read`` alone, so the read itself — the GDAL
    environment, the credential session :mod:`tessera_embeddings.ingest.auth` injects per thread,
    and the log records :mod:`tessera_embeddings.ingest.loader_failures` attributes back to an
    object — is exactly the one odc would have performed.
    """

    def read(
        self,
        cfg: Any,  # noqa: ANN401 — odc's RasterLoadParams
        dst_geobox: Any,  # noqa: ANN401 — odc.geo GeoBox
        *,
        dst: np.ndarray | None = None,
        selection: Any = None,  # noqa: ANN401 — odc's ReaderSubsetSelection
    ) -> tuple[tuple[slice, slice], np.ndarray]:
        """Read this source and correct it, before anything fuses it with another."""
        roi, pixels = super().read(cfg, dst_geobox, dst=dst, selection=selection)
        driver_data = self._src.driver_data
        offset = driver_data.get(_OFFSET_KEY) if isinstance(driver_data, dict) else None
        if offset:
            # In place, which is safe because odc never hands a reader a shared destination: the
            # only caller passes no `dst`, so this buffer belongs to this source alone until the
            # fuser copies out of it. A zero-size array from a tolerated read failure, and a
            # wholly-nodata array from a source that does not overlap, are both no-ops here.
            #
            # `where` is what keeps nodata nodata. DN 0 is the one code carrying no offset, and
            # correcting it would both invent a dark observation and destroy the gap marker.
            np.copyto(pixels, correct_boa_dn(pixels, offset), where=pixels > _NODATA)
        return roi, pixels


class _BoaCorrectingDriver(RioDriver):
    """The rio reader driver, correcting each image before the mosaic rather than after it.

    This is the whole point of the class: ``odc.stac.load`` fuses a solar day's tiles into one
    time slice, so a correction applied to its output is applied to every tile at once, and a day
    whose imagery disagrees then has no correct single answer — which is why such days were
    refused, at a measured cost of 347 days of one region-year. Applied here, each image answers
    for itself and there is no pixel that is both corrected and uncorrected.

    Overrides ``open`` and nothing else. ``restore_env`` and ``capture_env`` are inherited
    deliberately: they are what put the read inside the GDAL environment and the per-thread
    credential session that :mod:`tessera_embeddings.ingest.auth` patches, so overriding either
    would silently take long-lived workers off credential refresh.
    """

    def __init__(self, parser: BoaOffsetParser) -> None:
        """Build a driver whose sources were stamped by ``parser``."""
        super().__init__(md_parser=parser)

    def open(self, src: Any, ctx: Any) -> RioReader:  # noqa: ANN401 — odc's RasterSource/LocalContext
        """Open ``src`` as a rio reader that corrects its own pixels."""
        return _BoaCorrectingReader(src, ctx)


# =============================================================================
# High-Level Ingestion API
# =============================================================================


def _loadable_assets(collection_config: CollectionConfig, extra_bands: "list[str] | None" = None) -> frozenset[str]:
    """Every asset the loader could read for this collection.

    ``extra_bands`` must be included: pruning happens at query time, before
    ``odc.stac.load`` runs, so an asset dropped here is simply gone. Without it a
    caller asking for a QA or visualisation band gets a missing-band error instead
    of the band — and `ingest_tile`/`load_stac_items` both still document that option.
    """
    names = set(collection_config.bands)
    if collection_config.has_scl:
        names.add("scl")
    names.update(extra_bands or ())
    return frozenset(names)


def _requested_assets(collection_config: CollectionConfig, extra_bands: "list[str] | None" = None) -> tuple[str, ...]:
    """The assets a load will actually REQUEST, in a stable order.

    Narrower than :func:`_loadable_assets`, which is a pruning set and deliberately generous — it
    keeps ``scl`` for any collection that has one, whether or not this call asks for it. Ranking a
    duplicate copy on that broader set penalises it for lacking an asset the load never reads.
    """
    return (*collection_config.bands, *(b for b in (extra_bands or ()) if b not in collection_config.bands))


def _prune_item_dict(item: dict[str, Any], keep_assets: frozenset[str]) -> dict[str, Any]:
    """Drop assets the loader never reads, plus links, from a STAC item dict.

    Retained items dominate the ingest driver's memory: streaming holds a month plus the
    prefetched next month, and a catalogue item is mostly assets for bands nobody asked
    for. Sentinel-2 L2A on earth-search carries 35 assets — previews, per-band JP2
    variants, metadata documents — of which the ingest loads 11.

    Deliberately a DENY-list: every remaining key, and every field of every kept asset, is
    preserved verbatim. An allow-list of properties was tried first and silently omitted
    the CRS, because this collection carries it in ``proj:code`` where the list expected
    ``proj:epsg``. Dropping whole unread assets cannot lose metadata the loader needs,
    and it is where nearly all of the saving is anyway.

    Leaves the item untouched unless EVERY requested name is present as an asset key. The
    old test was ``if not kept`` — zero matches — which protected a collection named
    entirely differently but not one named PARTLY differently. An item mixing native keys
    with aliases (asked for ``blue`` and ``scl``, carrying ``B02`` and ``scl``) matched on
    one name, so the prune kept ``scl`` and dropped ``B02`` — and the loader then asked for
    ``blue``, whose only source had just been deleted. That fails the load having passed
    every check before it.

    Nothing here can see the alias table that maps a band name to an asset key, so a name
    that is absent as a key is indistinguishable from one that is served under another key.
    Retaining the whole item is the conservative reading, and the cost is asymmetric: a
    retained item spends memory, a wrongly pruned one loses the band.
    """
    assets = item.get("assets")
    if not assets:
        return item
    kept = {name: a for name, a in assets.items() if name in keep_assets}
    if len(kept) < len(keep_assets):
        return item
    pruned = dict(item)
    pruned["assets"] = kept
    pruned["links"] = []  # still valid for from_dict; the loader never follows them
    return pruned


def _area_label(query_params: dict[str, Any], tile_id: str | None) -> str:
    """The spatial term of a built query, as a stable identity fragment.

    Reads the query that will actually be sent, so the name in a failure log cannot
    disagree with what was asked. A query carrying neither spatial term is named as
    such rather than raising: this feeds a diagnostic, and the collection, window and
    page still identify the request without it.
    """
    if "query" in query_params:
        return f"tile={tile_id}"
    if (built := query_params.get("bbox")) is not None:
        return bbox_area_label(built)
    return "area=unspecified"


#: Instant a window's interior boundary is rendered at. Consecutive windows abut on the
#: SAME instant and the catalogue's range is inclusive at both ends, so their union has no
#: gap and their only overlap is that one instant, which the caller's id dedupe absorbs.
_BOUNDARY_INSTANT = "T00:00:00Z"


def _window_days(start_date: str, end_date: str) -> tuple[datetime.date, int]:
    """A window's first day and its length in days, whichever form its bounds take.

    Bounds arrive either as a bare ``YYYY-MM-DD`` — which the catalogue client expands to
    that whole day — or as one of the boundary instants this function's own output carries.
    An instant end is EXCLUSIVE of its own day for length purposes: a window ending
    ``2019-03-06T00:00:00Z`` covers through the 5th plus a single instant of the 6th.
    """
    first = datetime.date.fromisoformat(start_date[:10])
    last_day = datetime.date.fromisoformat(end_date[:10])
    last_exclusive = last_day if len(end_date) > 10 else last_day + datetime.timedelta(days=1)
    return first, (last_exclusive - first).days


def split_query_window(start_date: str, end_date: str, parts: int) -> list[tuple[str, str]]:
    """``[start_date, end_date]`` as ``parts`` shorter windows whose union is the input.

    A re-partition, never a narrowing, and exact in both directions: the outer bounds are
    the caller's own strings passed straight back, and every interior boundary is one
    instant shared by the window that ends there and the window that starts there. The
    union is therefore the input window with no gap, and the overlap is a single instant per
    seam rather than a whole repeated day.

    That exactness is why the boundary is an instant and not a date. A bare date end is
    expanded by the client to ``T23:59:59Z``, so windows that abut on consecutive DATES
    leave the last second of each seam's earlier day unasked for — a second the unsplit
    window would have covered.

    Returns ``[]`` for fewer than two parts and for a single day — the caller's signal to
    stop rather than re-partition again, and what makes re-partitioning terminate.
    """
    first, days = _window_days(start_date, end_date)
    parts = min(parts, days)
    if parts < 2:
        return []
    edges = [first + datetime.timedelta(days=days * i // parts) for i in range(parts)]
    bounds = [start_date] + [f"{edge}{_BOUNDARY_INSTANT}" for edge in edges[1:]] + [end_date]
    return [(bounds[i], bounds[i + 1]) for i in range(parts)]


def _parts_for_depth(page: dict[str, Any]) -> int:
    """Shorter windows this window needs, from the match count beside its first page.

    1 when it needs none, or when the catalogue reports no count — the walk is then
    bounded only by a refusal, as it was before.
    """
    matched = page.get("numberMatched")
    if matched is None:
        matched = (page.get("context") or {}).get("matched")
    if not isinstance(matched, int) or matched <= _MAX_QUERY_ITEMS:
        return 1
    return -(-matched // _MAX_QUERY_ITEMS)


@final
class _RePartitionBudget:
    """How many more times this query may replace a refused window.

    One per query, shared by every window walked for it, so the bound is on the QUERY rather
    than on any one branch of its tree. Thread-safe because the windows are walked at once.
    """

    def __init__(self, limit: int) -> None:
        self._left = limit
        self._lock = threading.Lock()

    def take(self) -> bool:
        """Claim one replacement, or False when the query has spent them all."""
        with self._lock:
            if self._left <= 0:
                return False
            self._left -= 1
            return True


@final
@dataclass
class _WindowWalk:
    """One date window of the worklist, what it returned, and what replaced it.

    A node of the tree the worklist walk describes. The serial walk WAS a depth-first
    traversal of exactly this tree: it appended a window's items, then descended into the
    shorter windows that window asked for instead of the rest of its own pages. Making the
    tree explicit is what lets the fetching run concurrently while the output stays in the
    order the serial walk produced — and the order is load-bearing, because
    :func:`query_stac_items` sorts the result on keys items can TIE on, Python's sort is
    stable, and the surviving input order therefore decides which scene supplies a pixel:
    odc-loader's fuser fills only where the destination is still nodata, so the FIRST valid
    source of a group wins. The item list is inside the mosaic fingerprint as well.

    Attributes:
        bbox: Spatial term this window is asked with — one antimeridian half, the whole
            box, or None for a property-based query.
        window: ``(start, end)`` as the catalogue is asked for it. Either bound may be a
            bare date or one of :func:`split_query_window`'s boundary instants.
        page_size: Items per page request, or None for the provider's own. Set only when a
            refusal has already been answered by asking for a smaller response, and
            inherited by children so a window never returns to a size that was refused.
        items: What this window's OWN pages returned, in page order, hydrated and pruned
            but not deduplicated — assembly dedupes across the whole tree at once.

            Hydrated HERE rather than at assembly, even though that hydrates duplicates a
            re-cut window's children will return again. Hydration is ~1.1 ms/item of pure
            Python, so deferring it moves ~60 s of a zone-month query out of the walks,
            where it overlaps network waiting, and into the single-threaded assembly, where
            nothing hides it. The cost is holding duplicate items until assembly: about 4%
            of a clean query, and ~30% more peak memory on a refusal-heavy one.
        children: Shorter windows that must be walked after this one, in date order.
            Non-empty exactly when this window's own walk stopped early.
    """

    bbox: tuple[float, float, float, float] | None
    window: tuple[str, str]
    page_size: int | None = None
    items: list[Any] = field(default_factory=list)
    children: list["_WindowWalk"] = field(default_factory=list)

    def preorder(self) -> Iterator["_WindowWalk"]:
        """This node, then each child's subtree in order — the serial walk's visit order."""
        yield self
        for child in self.children:
            yield from child.preorder()


@final
class _PerThreadClient:
    """One :class:`~pystac_client.Client` per thread that asks for one.

    ``StacApiIO`` holds a ``requests.Session``, which is not documented thread-safe, and
    the windows are now walked concurrently. A client per worker thread costs one GET of
    the catalogue root per thread and removes the question rather than reasoning about it;
    it also gives each thread its own HTTP connection, which is what a concurrent walk
    wants anyway.

    One root fetch per worker thread is the cost, and it is the whole cost: the root is a small
    static document behind a CDN, and a thread opens it once for the life of the query. There
    was briefly an extra one — the calling thread opened a client eagerly so an unreachable
    catalogue would be named as the root rather than surfacing from inside a window's walk — but
    that client was never used to search, because the walks all run on pool threads. It was
    removed once it was clear the naming does not depend on where the failure happens: this class
    builds the page-0 `catalogue-root` error itself, and `_fill_window_tree` re-raises it
    unchanged, so a root failure reads identically from a worker.
    """

    def __init__(self, provider: STACProvider, collection_id: str, window: str) -> None:
        self._provider = provider
        self._collection_id = collection_id
        self._window = window
        self._local = threading.local()

    def get(self) -> Client:
        """This thread's client, opening it on first use."""
        client: Client | None = getattr(self._local, "client", None)
        if client is None:
            self._local.client = client = self._open()
        return client

    def _open(self) -> Client:
        """Open the catalogue root, retrying the one status its ladder no longer holds.

        Page 0 for reporting: the root is fetched before any search exists, so a refusal here
        is named separately rather than read as a refusal of the search it precedes.
        """
        stac_io = StacApiIO(max_retries=_retry_for(self._provider), timeout=_STAC_TIMEOUT)
        request = CatalogueRequest(self._collection_id, self._window, "catalogue-root", 0)
        for attempt in range(1, _ROOT_OPEN_ATTEMPTS + 1):
            try:
                return Client.open(self._provider.catalog_url, stac_io=stac_io)
            except Exception as exc:
                # Read off the same set the ladder drops, so the two cannot fall out of step.
                dropped_by_the_ladder = classify_refusal(exc).status in OVERSIZED_RESPONSE_STATUSES
                if attempt < _ROOT_OPEN_ATTEMPTS and dropped_by_the_ladder:
                    logger.warning(
                        "Catalogue root refused (%s), attempt %d of %d — retrying in %.0fs",
                        self._provider.catalog_url,
                        attempt,
                        _ROOT_OPEN_ATTEMPTS,
                        _ROOT_OPEN_BACKOFF_S * attempt,
                    )
                    time.sleep(_ROOT_OPEN_BACKOFF_S * attempt)
                    continue
                raise_catalogue_query_error(
                    request, exc, log=logger, throttle_statuses=_throttle_statuses(self._provider)
                )
        raise AssertionError("unreachable: the loop either returns or raises")  # pragma: no cover


def _walk_query_window(
    client: Client,
    provider: STACProvider,
    collection_config: CollectionConfig,
    tile_id: str | None,
    keep_assets: frozenset[str],
    budget: _RePartitionBudget,
    node: _WindowWalk,
) -> None:
    """Walk one window's pages, filling ``node.items`` and ``node.children`` in place.

    Everything the serial loop did for a SINGLE window, with the worklist decision handed
    back as ``node.children`` instead of taken. Serial within the window — every page's
    cursor comes from the page before it — and self-contained, refusal handling included,
    which is what lets a caller run many of these at once without any task ever waiting on
    another task.

    Raises:
        CatalogueQueryError: The catalogue refused a request that shortening the window
            cannot route around. The item count in that log line is now THIS window's own:
            with windows in flight concurrently there is no whole-query total to report at
            the moment one of them fails.
        ValueError: A returned item carries no ``id``, so it cannot be deduplicated.
    """
    win_start, win_end = node.window
    window = f"{win_start}/{win_end}"
    query_params = _build_stac_query(collection_config, tile_id, win_start, win_end, bbox=node.bbox)
    # Read off the query that was BUILT rather than re-deciding property-versus-bbox
    # here: a second copy of that branch could disagree with the one that ran, and
    # then the log would name a request the catalogue was never asked. Tolerant of a
    # query carrying neither term, because a diagnostic must never be the thing that
    # raises — an unnamed area still leaves the collection, window and page named.
    area = _area_label(query_params, tile_id)
    page_size = node.page_size or provider.max_page_size
    search = client.search(**query_params, limit=page_size, max_items=None)
    # Paged EXPLICITLY (`pages_as_dicts`, which is what `items_as_dicts` iterates
    # internally) so a failure can name the page ordinal it died on. Each page is a
    # separate HTTP request behind its own retry ladder, so "the whole year fails" and
    # "one deep cursor fails" are different defects with different reproductions, and
    # an item count cannot tell them apart.
    #
    # Dicts rather than Items so a full item is never hydrated: pruning first and
    # hydrating second is both smaller to retain and ~3x cheaper to build, because most
    # of an item's construction cost is the assets that get dropped.
    #
    # `iter()` rather than trusting the return: advancing by hand needs an ITERATOR, and
    # this seam is injectable, so a search that returns an iterable which is not one
    # (a list of pages, a stub) would otherwise be a TypeError — or, for a stub that
    # answers every call, an unbounded loop. Asking for an iterator costs nothing on the
    # real generator, which is its own iterator.
    pages = iter(search.pages_as_dicts())
    for page_no in count(1):
        request = CatalogueRequest(collection_config.collection_id, window, area, page_no)
        try:
            page = next(pages)
        except StopIteration:
            return
        except Exception as exc:
            # Scoped to the FETCH alone. Wrapping the body below as well would
            # classify our own validation failure as a catalogue refusal, which the
            # retry policy would then act on.
            #
            # THIS is the fix for the Earth Search 502: a page request the catalogue
            # will not serve is re-asked as shorter windows, which pose it as
            # different requests. Waiting cannot help, so 502 is kept out of the
            # retry ladder above and arrives here on the first refusal. Shortening
            # the window's END is what clears it; shortening only its start does
            # not, so this recurses until a window completes or is down to a
            # single day.
            #
            # Not on a stated overload, which is what the ladder above is for and which
            # slicing would only multiply.
            #
            # Two levers, tried in that order. A shorter WINDOW is preferred because its
            # halves between them walk about as many pages as this window would have, while
            # a smaller PAGE re-walks the whole window at twice the requests. But shortening
            # cannot reach every refusal: a FIRST page is asked identically by a shorter
            # window, and a single day is the re-cut's floor. Those two both failed the leg
            # outright before this, and a smaller page is the only lever left for them.
            shorter = split_query_window(win_start, win_end, 2)
            smaller = page_size // 2
            # Only a size refusal, not every backend failure. 500 and 504 are still
            # force-listed, so one arrives here having already spent the ladder's whole
            # backoff — and re-cutting it would hand that same ladder to each child, turning
            # a persistent outage into minutes of backoff multiplied across a recursion.
            if provider.refuses_oversized_pages and is_oversized_response(classify_refusal(exc)):
                if page_no > 1 and shorter and budget.take():
                    logger.warning(
                        "Catalogue refused %s — retrying it as %d shorter window(s)",
                        request.label,
                        len(shorter),
                    )
                    node.children = [_WindowWalk(node.bbox, w, node.page_size) for w in shorter]
                    return
                if smaller >= _MIN_PAGE_SIZE and budget.take():
                    # Same window, same items, same order — only the response is smaller,
                    # which is what the cap is measured against. The pages already walked
                    # are re-fetched and the assembly step dedupes them away.
                    logger.warning(
                        "Catalogue refused %s — retrying the whole window at %d items per page (was %d)",
                        request.label,
                        smaller,
                        page_size,
                    )
                    node.children = [_WindowWalk(node.bbox, node.window, smaller)]
                    return
            raise_catalogue_query_error(
                request,
                exc,
                log=logger,
                items_so_far=len(node.items),
                throttle_statuses=_throttle_statuses(provider),
            )
        for raw in page.get("features", []):
            # `id` is required by the STAC spec; an item without one cannot be deduped by
            # the assembly step, and defaulting it would collapse EVERY such item into a
            # single entry — so say so rather than silently drop data.
            if raw.get("id") is None:
                raise ValueError(f"STAC item without an 'id' from {provider.catalog_url} — cannot dedupe it")
            node.items.append(Item.from_dict(_prune_item_dict(raw, keep_assets)))
        # Sized off the first page's match count, so a window too deep to walk is
        # re-cut before any deep cursor is requested rather than after one is refused.
        if page_no == 1 and (shorter := split_query_window(win_start, win_end, _parts_for_depth(page))):
            logger.info(
                "Window %s %s matches over %d items — querying it as %d shorter window(s)",
                window,
                area,
                _MAX_QUERY_ITEMS,
                len(shorter),
            )
            node.children = [_WindowWalk(node.bbox, w, node.page_size) for w in shorter]
            return


def _fill_window_tree(
    walk: Callable[[_WindowWalk], None],
    roots: list[_WindowWalk],
    workers: int,
) -> None:
    """Fill every node of every root's tree, walking up to ``workers`` windows at once.

    A worklist driven from THIS thread, never from inside the pool. Tasks only walk: they
    neither submit nor wait, so no arrangement of them can deadlock — which is the failure
    mode of the obvious recursive shape, where a task waits on a child queued behind it in
    a saturated pool. Whenever a window finishes, the shorter windows it asked for instead
    join the worklist and the freed worker takes whatever is next, so the narrow tail of a
    refusal chain — two windows wide at every step — overlaps the wide walks still running
    instead of queueing behind them.

    Scheduling changes WHEN a window is walked, never the order of what it returned: the
    output order is read back off the finished tree, depth-first, by
    :meth:`_WindowWalk.preorder`.

    Every window the tree contains is walked even once one of them has failed, and the
    failure re-raised is the one whose window comes FIRST in that same depth-first order.
    Both halves are deliberate. Stopping early would make WHICH failure is raised depend on
    which task happened to finish first, and the layer that owns the attempt budget decides
    a refusal is deterministic by seeing the identical signature again on a later attempt —
    so that signature has to be a function of the query, not of a race. Continuing cannot
    enlarge the work either: a window that fails contributes no children.
    """
    failures: dict[int, BaseException] = {}

    def sequenced(node: _WindowWalk) -> _WindowWalk:
        """Walk one window, recording a failure instead of raising it."""
        try:
            walk(node)
        except BaseException as exc:  # re-raised below, in a fixed order
            failures[id(node)] = exc
        return node

    pending: deque[_WindowWalk] = deque(roots)
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="stac-window") as pool:
        running: set[Future[_WindowWalk]] = set()
        while pending or running:
            while pending and len(running) < max(1, workers):
                running.add(pool.submit(sequenced, pending.popleft()))
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for finished in done:
                pending.extend(finished.result().children)

    if failures:
        depth_first = {id(node): rank for rank, node in enumerate(n for r in roots for n in r.preorder())}
        raise failures[min(failures, key=lambda node_id: depth_first[node_id])]


def _query_stac_items(
    provider: STACProvider,
    collection_config: CollectionConfig,
    tile_id: str | None,
    start_date: str,
    end_date: str,
    bbox: tuple[float, float, float, float] | None = None,
    item_provider_fn: Callable[[], list[Any]] | None = None,
    extra_bands: list[str] | None = None,
) -> list[Any]:
    """Query STAC catalog for items matching tile and date range.

    Internal function - use ingest_tile() for the full workflow.

    Args:
        provider: STAC provider configuration
        collection_config: Collection configuration
        tile_id: Tile identifier, or None for bbox-based queries
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        bbox: Optional bounding box for spatial query fallback
        item_provider_fn: Optional callable that returns ready-to-load items,
            bypassing CMR-STAC ``client.search()`` entirely. Used by the
            ``cmr-asf`` OPERA path, which builds items from the native CMR
            granule API to avoid CMR-STAC's 500-prone cursor pagination.

        extra_bands: Additional assets the caller will load. Kept in the pruned
            items — pruning runs at query time, so an asset dropped here cannot
            be loaded later.

    Returns:
        List of pystac Items

    Raises:
        CatalogueQueryError: The catalogue refused a request, classified and naming the
            request that was refused. The client library reports transport failures with
            the search discarded, so this is where a refusal becomes identifiable at all —
            see :mod:`~tessera_embeddings.ingest.catalogue_refusal`. An upstream error past
            the first page is NOT reported here while the window can still be shortened:
            it is re-queried as shorter windows (:func:`split_query_window`), which returns
            the same items. One that shortening cannot route around still raises.
        ValueError: A returned item carries no ``id``, so it cannot be deduplicated.
            Deliberately NOT wrapped as a refusal: it is a defect in what the catalogue
            returned rather than a refusal to answer, and the retry policy keyed on
            refusals must not act on it.
    """
    if item_provider_fn is not None:
        return item_provider_fn()

    window = f"{start_date}/{end_date}"
    t0 = time.monotonic()
    logger.info(f"Opening STAC catalog: {provider.catalog_url}")
    clients = _PerThreadClient(provider, collection_config.collection_id, window)

    keep_assets = _loadable_assets(collection_config, extra_bands)
    # A worklist of date windows, not one walk: a window the catalogue will not page to the
    # end of is replaced by shorter ones covering the same days. The worklist is a TREE —
    # see :class:`_WindowWalk` — and its windows are independent searches, so it is filled
    # concurrently and read back depth-first, in date order, which is the order the serial
    # walk produced. The same searches run on every attempt, so the layer above can still
    # decide a refusal is deterministic by seeing the identical signature twice.
    roots = [_WindowWalk(sub_bbox, (start_date, end_date)) for sub_bbox in split_antimeridian_bbox(bbox)]
    budget = _RePartitionBudget(_MAX_REFUSAL_RE_PARTITIONS)
    _fill_window_tree(
        lambda node: _walk_query_window(clients.get(), provider, collection_config, tile_id, keep_assets, budget, node),
        roots,
        _QUERY_WINDOW_WORKERS,
    )

    # Dedupe across every search this query ran, first occurrence winning, in tree order.
    # A granule straddling +/-180 is returned by both antimeridian halves, one acquired on
    # a seam instant by both windows that share it, and every item of a re-cut window's
    # first page by the shorter windows that replaced it. Done HERE rather than as pages
    # arrive because which page arrives first is now a matter of timing, and
    # first-occurrence-wins has to mean first in the WALK, not first off the wire.
    items: list[Any] = []
    seen: set[str] = set()
    for root in roots:
        for node in root.preorder():
            for item in node.items:
                if item.id not in seen:
                    seen.add(item.id)
                    items.append(item)
    logger.info(f"STAC query returned {len(items)} items in {time.monotonic() - t0:.1f}s total")

    return items


def has_new_stac_dates(
    provider: str,
    collection: str,
    tile_id: str | None,
    start_date: str,
    end_date: str,
    existing_dates: set[str],
    bbox: tuple[float, float, float, float] | None = None,
    item_filter_fn: Callable[[list[Any]], list[Any]] | None = None,
    item_provider_fn: Callable[[], list[Any]] | None = None,
    mid_longitude: float | None = None,
) -> bool:
    """Check whether a STAC catalog has new dates not yet in the store.

    Lightweight check that queries STAC and filters without loading raster data.
    Intended as a pre-flight gate so a flow can decide whether to spin up a
    Dask cluster before iterating batches — today nothing does this, so the
    caller is responsible for not passing date ranges that overlap the store.

    .. note::
       **Currently unused.** No flow calls this yet; wiring it into the S1/S2
       ROI flows is tracked in
       https://github.com/dClimate/tessera-embeddings/issues/47.

       When wiring it up, do **not** pass the same ``item_provider_fn`` object
       to both this function and ``query_stac_items`` for one batch: the
       provider built by ``opera_query.make_s1_item_provider`` issues a fresh
       CONUS-scale CMR granule query on every call (it no longer caches at
       construction time), so reusing it across the pre-check and the real
       query would double the query cost. Build a provider per call site, or
       have the gate consume the items the subsequent ingest will load.

    Args:
        provider: Provider name (e.g., "earth-search", "cmr-asf")
        collection: Collection alias (e.g., "sentinel-2-l1c", "opera-rtc-s1")
        tile_id: Tile identifier (e.g., "33UUP"), or None for bbox queries
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        existing_dates: Dates already in store (YYYY-MM-DD strings)
        bbox: Optional bounding box for spatial query fallback
        item_filter_fn: Optional pre-filter (e.g., orbit direction filtering)
        item_provider_fn: Optional callable returning items directly,
            bypassing CMR-STAC search (OPERA native-granule path).
        mid_longitude: ROI geobox centroid longitude. Supply it whenever the store was
            written in solar days — which the campaign always is — so the comparison is
            keyed the same way. Omitting it compares UTC dates.

    Returns:
        True if at least one new date is available
    """
    provider_config = _get_provider_config(provider)
    collection_config = _get_collection_config(provider, collection)

    query_label = f"tile {tile_id}" if tile_id else f"bbox {bbox}"
    logger.debug(f"Querying {provider}/{collection} for {query_label}")
    items = _query_stac_items(
        provider_config,
        collection_config,
        tile_id,
        start_date,
        end_date,
        bbox=bbox,
        item_provider_fn=item_provider_fn,
    )
    if not items:
        logger.debug(f"No items found for {query_label}")
        return False

    if item_filter_fn is not None:
        items = item_filter_fn(items)
        if not items:
            logger.debug(f"No items remaining after filtering for {query_label}")
            return False

    # Solar-day stamp before comparing: the store's dates are solar days, so matching a
    # raw UTC date against them under-reports new data in exactly the high-offset zones
    # where a day straddles midnight. Same reason as the ingest path — see
    # ingest.solar_days — and cheap here because nothing is loaded.
    # A longitude is REQUIRED for solar-day grouping — `resolve_grouping_longitude` refuses
    # rather than falling back to UTC dates, deriving from this call's own `bbox` when the
    # caller gave none. See that function for why the old None-means-UTC default was unsafe.
    items = normalize_to_solar_day(items, mid_longitude=resolve_grouping_longitude(mid_longitude, bbox))
    new_items = _filter_existing_dates(items, existing_dates)
    has_new = len(new_items) > 0
    logger.debug(f"{query_label}: {len(new_items)} new dates (of {len(items)} total)")
    return has_new


def query_stac_items(
    provider: str,
    collection: str,
    tile_id: str | None,
    start_date: str,
    end_date: str,
    existing_dates: set[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    item_filter_fn: Callable[[list[Any]], list[Any]] | None = None,
    item_provider_fn: Callable[[], list[Any]] | None = None,
    extra_bands: list[str] | None = None,
    mid_longitude: float | None = None,
) -> tuple[list[Any], dict[str, int]]:
    """Query STAC catalog, filter items, and extract baselines.

    Combines STAC catalog query, optional item filtering, cloud-cover sorting,
    baseline extraction, and existing-date filtering. This is the query half
    of ``ingest_tile``; pair with ``load_stac_items`` to load data separately.

    Args:
        provider: Provider name (e.g., "earth-search")
        collection: Collection alias (e.g., "sentinel-2-l2a")
        tile_id: Tile identifier (e.g., "33UUP"), or None for bbox queries
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        existing_dates: Dates already in store (to skip). None = return all.
        bbox: Optional bounding box for spatial query fallback
        item_filter_fn: Optional pre-filter (e.g., orbit direction filtering)
        item_provider_fn: Optional callable returning items directly,
            bypassing CMR-STAC search (OPERA native-granule path).

        extra_bands: Additional assets the caller will load. Kept in the pruned
            items — pruning runs at query time, so an asset dropped here cannot
            be loaded later.
        mid_longitude: ROI centroid longitude, whenever the load groups by solar
            day. ``existing_dates`` is then keyed on solar days, and matching an
            item's UTC date against it half-filters a committed group. Omit for
            callers that group by UTC date.

    Returns:
        Tuple of (items, baselines) where:
        - items: Filtered list of pystac Items (excluding existing dates)
        - baselines: Dict mapping date strings to baseline integers (all dates
          that passed item_filter_fn, including existing ones)
    """
    provider_config = _get_provider_config(provider)
    collection_config = _get_collection_config(provider, collection)

    items = _query_stac_items(
        provider_config,
        collection_config,
        tile_id,
        start_date,
        end_date,
        bbox=bbox,
        item_provider_fn=item_provider_fn,
        extra_bands=extra_bands,
    )

    query_label = f"tile {tile_id}" if tile_id else f"bbox {bbox}"
    if not items:
        logger.warning(f"No items found for {query_label} in date range")
        return [], {}

    if item_filter_fn is not None:
        items = item_filter_fn(items)
        if not items:
            logger.info("No items remaining after item_filter_fn")
            return [], {}

    # THE single solar-offset application for this path. Everything below — the cloud sort,
    # the baseline map, the existing-date dedup, and every consumer of the returned items —
    # then reads the solar day straight off `item.datetime` with no offset of its own. Before
    # this, the sort and the baseline map keyed on the UTC date while the loader grouped by
    # solar day, so on a day straddling UTC midnight the group was not sorted as intended and
    # half its baseline entries never matched.
    # Same rule as the date probe above, and it matters more here: this is THE single
    # solar-offset application for the path, so a UTC-day stamp taken by default would be
    # read as the solar day by the sort, the baseline map, the dedup and every consumer.
    items = normalize_to_solar_day(items, mid_longitude=resolve_grouping_longitude(mid_longitude, bbox))

    # `solar_day_sort_key` owns the order and the reasoning; `load_kwargs` below sets
    # `preserve_original_order=True` so it is that order the loader fuses in. Pinned end to end by
    # `tests/unit/test_solar_day_fusion.py`, which loads two real scenes of one solar day through
    # this path and asserts which one survives.
    #
    # Keyed on the SOLAR day, matching `normalize_to_solar_day` just above and the loader's own
    # grouping. Keying on the UTC date instead splits a group the loader treats as one, in the
    # far-eastern and far-western zones where the solar offset crosses midnight.
    # TODO: this sort runs only for a collection WITH an SCL band, which today means
    # `sentinel-2-l2a` alone. Every other collection keeps the assembly order, and that order
    # follows the window tree rather than a single catalogue walk — parent prefix first, then
    # children oldest-first, against the catalogue's newest-first. So for a non-SCL collection
    # the fused result would depend on how the query was partitioned. Not reachable today:
    # the only paged caller is `s2_roi`, which asks for `sentinel-2-l2a`, and the radar path
    # returns its items from a provider function before any window exists. Left alone because
    # imposing an order on Landsat, which currently has none, is a change to that collection's
    # output that nothing here can verify. Fix it alongside the first paged non-SCL caller.
    if collection_config.has_scl:
        items.sort(key=solar_day_sort_key)

    baselines = extract_baselines(items)

    if existing_dates:
        items = _filter_existing_dates(items, existing_dates)

    return items, baselines


def load_stac_items(
    items: list[Any],
    provider: str,
    collection: str,
    baselines: dict[str, int] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    chunks: dict[str, int] | None = None,
    extra_bands: list[str] | None = None,
    resampling: str = "bilinear",
    crs: str | None = None,
    resolution: int | None = None,
    post_load_fn: Callable[[xr.Dataset], xr.Dataset] | None = None,
    groupby: str = "solar_day",
    geobox: GeoBox | None = None,
    selection_label: str | None = None,
) -> xr.Dataset:
    """Load STAC items into an xarray Dataset with corrections applied.

    Wraps ``_load_from_stac`` and adds baseline correction and post-load
    transforms. This is the load half of ``ingest_tile``; pair with
    ``query_stac_items`` for the full workflow.

    Args:
        items: List of pystac Items to load (must be non-empty)
        provider: Provider name (e.g., "earth-search")
        collection: Collection alias (e.g., "sentinel-2-l2a")
        baselines: Dict mapping date strings to baseline integers. PROVENANCE only — the
            correction is derived from ``items``, so this decides no pixel.
        bbox: Optional bounding box for spatial subsetting
        chunks: Optional chunk sizes for odc.stac.load
        extra_bands: Additional bands to load (e.g., ["scl"])
        resampling: Resampling method for primary bands (default "bilinear")
        crs: Explicit CRS override for odc.stac.load
        resolution: Override pixel resolution in metres
        post_load_fn: Optional function applied after loading and correction
        groupby: How to group items into time slices. Must be "solar_day" —
              :func:`_load_from_stac` rejects anything else and says why.
        geobox: Optional output grid specification
        selection_label: What to call this call's area in the duplicate-selection audit log.
              For a caller that queried by TILE: the tile id never reaches this function, and
              without it concurrent same-collection tile ingests log identical lines. Defaults
              to the bbox, which is per region of interest.

    Returns:
        Corrected xarray Dataset

    Expects items that have already passed ``normalize_to_solar_day`` and, for a collection whose
    harmonisation varies by item, duplicate selection — both of which :func:`query_stac_items`
    does. Selection runs again below, idempotently, because the documented ``query_stac_items``
    -> ``load_stac_items`` workflow is the one path that has had neither; it groups by
    ``solar_day_of``, which refuses an un-normalised item rather than deriving a plausible-looking
    wrong day from its UTC stamp.

    **A refusal belongs to the ITEM LIST this is called with, not to a day inside it.**
    ``HeterogeneousProducerError`` is raised while ``odc.stac.load`` parses that list
    synchronously, so it abandons the whole call. Pass ONE SOLAR DAY at a time wherever a day that
    cannot be decided should be skipped alone — which is what ``s2_roi`` does, and why the
    refusals it reports cost one date each. A multi-day list handed straight to this function
    forfeits every day in it, sound or not.

    """
    collection_config = _get_collection_config(provider, collection)

    # Gated on there BEING an offset decision, not on the producer varying between items. Both are
    # exposed to the same fusion: `odc.stac.load` blends every copy of one acquisition into one
    # pixel stack, so redundant copies are read, resampled and painted over one another for a
    # single observation. Measured on the live Planetary Computer catalogue over six tiles: 1,000
    # redundant copies of one observation fused. Selection also routes around the one refusal that
    # remains — a copy whose producer cannot be determined is ranked last and withheld from the
    # fallback ladder — which a date carrying a sound copy beside it should not pay for.
    #
    # NOT ungated further, though the same fusion argument reaches any collection with duplicates.
    # Landsat items key perfectly well here (`grid:code` is `WRS2-190028`), so removing the gate
    # would start reducing them too — a change nothing in this branch has measured. It is owed
    # separately rather than taken as a side effect.
    if collection_config.requires_baseline_correction:
        # Also here, and idempotently. `ingest_tile` prunes before extracting provenance and
        # `s2_roi` prunes before building its fallback ladder, but the documented
        # `query_stac_items` -> `load_stac_items` workflow passes through neither. Selecting again
        # over an already-selected set is a no-op, and this function already requires normalised
        # items for the correction decision, so the precondition is unchanged.
        read_keys = selection_read_keys(collection_config, extra_bands)
        pruned, alternates = select_preferred_duplicates(items, read_keys, collection_harmonisation(collection_config))
        # Gated on whether SELECTION CHANGED THE ITEMS, not on whether any usable fallback
        # survived. `select_preferred_duplicates` drops rejected copies that would refuse their
        # date, so `alternates` can be empty on a tile-date that was still pruned — and gating on
        # it left the caller's map describing a copy that is not being loaded.
        if len(pruned) != len(items):
            # Labelled with the AREA, not just the collection. This log is the only record of
            # which copy supplied a pixel, and a fleet runs sixty cells or tiles of one
            # collection at once — `load sentinel-2-l2a` on every one of them names none of
            # them. `selection_label` is what a caller that queried by TILE passes down, since a
            # tile id never reaches this function otherwise; the bbox covers everyone else.
            where = selection_label or (f"bbox {bbox}" if bbox is not None else f"load {collection}")
            log_duplicate_selection(logger, where, alternates, kept=pruned, read_keys=read_keys, items=items)
            # Realign the caller's provenance with what is about to be loaded. `baselines` becomes
            # the store's `baselines_applied`, and on the split workflow it was built by
            # `query_stac_items` from the UNPRUNED list — so a rejected copy that sorted last could
            # describe pixels the selected copy provided. Updated in place: the caller holds this
            # dict and there is no return value to carry it.
            if baselines is not None:
                baselines.update(extract_baselines(pruned))
        items = pruned

    # Built BEFORE the load, because the correction now happens inside it — per image, as each
    # source is read and before anything fuses it with another. That is what dissolves the
    # date-wide conflict: different tiles occupy different ground, so no pixel is both corrected
    # and uncorrected, and a day mixing producers no longer has to be refused.
    parser: BoaOffsetParser | None = None
    reflectance_assets: frozenset[str] = frozenset()
    if collection_config.requires_baseline_correction:
        reflectance_assets = _reflectance_asset_keys(items, collection_config)
        if not reflectance_assets:
            # Fall back to the configured band names rather than refusing, and the reason is a
            # trap this branch has already been caught by once.
            #
            # If these names really cannot be resolved, `odc.stac.load` fails on the SAME
            # resolution a moment later with "No such band/alias" — a `ValueError` that `s2_roi`
            # recognises as an asset-incomplete item and recovers from by stepping down to another
            # copy. Raising here instead would replace that recoverable error with a refusal, which
            # nothing retries: one unreadable item would abort a leg that used to survive it.
            #
            # Silent inertness is guarded from the other end instead. `driver_data` REFUSES a
            # source it cannot classify rather than exempting it, so a wrong answer is loud; and
            # the count below says how many sources were actually decided.
            reflectance_assets = frozenset(collection_config.bands)
            logger.warning(
                "Could not resolve which assets carry the reflectance bands for %s; falling back to "
                "the configured band names %s. If those are not this collection's asset keys the "
                "load will fail on the same resolution and report which band is missing.",
                collection,
                sorted(reflectance_assets),
            )
        parser = BoaOffsetParser(
            # `{}` and not `None`: odc's own resolver substitutes an empty dict for an absent
            # config, and `MDParseConfig.from_dict` does a containment test on it.
            {},
            reflectance_assets,
            collection_config.baseline_threshold,  # type: ignore[arg-type]
            collection_config.baseline_offset,
            collection_harmonisation(collection_config),
        )

    data = _load_from_stac(
        items,
        collection_config,
        bbox=bbox,
        chunks=chunks,
        extra_bands=extra_bands,
        resampling=resampling,
        crs=crs,
        groupby=groupby,
        resolution=resolution,
        geobox=geobox,
        driver=None if parser is None else _BoaCorrectingDriver(parser),
    )

    if parser is not None and not parser.stamped:
        # The no-silent-no-op check. A correction that reaches no source corrects nothing, produces
        # plausible pixels 1000 too high, and says nothing about it — the defect `main` shipped.
        #
        # A warning rather than an error, because a caller that replaces the loader also reports
        # zero and must not be failed for it. What makes this sufficient is that the failure it
        # describes cannot be silent anywhere else: an unclassifiable source refuses, and an
        # unresolvable band name fails the load.
        logger.warning(
            "BOA offset reached NO source for %s across %d item(s) — nothing was corrected. If any "
            "of these items are ESA originals at baseline >= %s their pixels are still %d too high. "
            "Reflectance assets were resolved as %s.",
            collection,
            len(items),
            collection_config.baseline_threshold,
            abs(collection_config.baseline_offset),
            sorted(reflectance_assets),
        )
    elif parser is not None:
        logger.info(
            "BOA offset decided per source for %s: %d of %d reflectance source(s) owed the %d offset.",
            collection,
            parser.owed,
            parser.stamped,
            collection_config.baseline_offset,
        )

    if post_load_fn is not None:
        data = post_load_fn(data)

    return data


def group_items_by_date(items: list[Any]) -> dict[str, list[Any]]:
    """Group STAC items by day, matching how the loader will group them.

    Groups on ``solar_day_of``, which reads the solar day each item was already stamped with.
    Items must therefore have passed :func:`normalize_to_solar_day` first — it is what applies the
    longitude shift, and grouping by UTC calendar date instead lets this and the loader disagree,
    so a group the caller believes is one day loads as TWO time slices.

    That divergence is not uniform: it appears where the solar offset is large enough to push
    acquisitions across UTC midnight, i.e. the far-eastern and far-western zones, and never in the
    middle longitudes. Downstream code assumes one slice per group (the cloud mask is reduced to a
    single 2-D slice), so the mismatch surfaces as a dimension conflict rather than as anything
    that names the cause.

    Within-group order is the caller's. ``query_stac_items`` sorts a date CLOUD-ASCENDING, so the
    clearest tile comes first — which is what the loader's fuser needs, since it writes only where
    the destination is still empty and so keeps the FIRST valid source of a group.

    Returns:
        Dict mapping ``YYYY-MM-DD`` to lists of items, preserving insertion order and
        within-group order.
    """
    groups: dict[str, list[Any]] = {}
    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        groups.setdefault(date_str, []).append(item)
    return groups


def ingest_tile(
    provider: str,
    collection: str,
    tile_id: str | None,
    start_date: str,
    end_date: str,
    existing_dates: set[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    chunks: dict[str, int] | None = None,
    extra_bands: list[str] | None = None,
    resampling: str = "bilinear",
    crs: str | None = None,
    resolution: int | None = None,
    post_load_fn: Callable[[xr.Dataset], xr.Dataset] | None = None,
    item_filter_fn: Callable[[list[Any]], list[Any]] | None = None,
    item_provider_fn: Callable[[], list[Any]] | None = None,
    groupby: str = "solar_day",
    geobox: GeoBox | None = None,
    mid_longitude: float | None = None,
) -> tuple[xr.Dataset | None, dict[str, int]]:
    """Query STAC, filter existing dates, load data, and apply corrections.

    This is the high-level API for ingesting satellite data. It combines:
    1. STAC catalog query
    2. Optional item filtering (e.g., orbit direction)
    3. Filtering out already-processed dates
    4. Loading data via odc.stac.load
    5. Applying baseline corrections (for Sentinel-2)
    6. Optional post-load transformation (e.g., amplitude to dB)

    Args:
        provider: Provider name (e.g., "earth-search")
        collection: Collection alias (e.g., "sentinel-2-l2a")
        tile_id: Tile identifier (e.g., "33UUP"), or None for ROI-based
              bbox queries where cross-tile mosaicking is handled by odc.stac.
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        existing_dates: Set of dates already in store (to skip). None = process all.
        bbox: Optional bounding box for spatial subsetting (minx, miny, maxx, maxy).
              Also used as STAC query fallback when tile_id is None or
              tile_id_property is None.
        chunks: Optional chunk sizes for odc.stac.load. Defaults to
              INGEST_CHUNKS. Accepts northing/easting or y/x keys —
              translated automatically before passing to odc.stac.load.
        extra_bands: Additional bands to load (e.g., ["scl"]).
        resampling: Resampling method for primary bands (default "bilinear").
        crs: Explicit CRS override for odc.stac.load (e.g., "EPSG:32633").
        resolution: Override pixel resolution in metres. When None (default),
              uses the collection's native resolution (e.g., 10m for S2, 30m
              for S1). Use to upsample S1 to 10m for a common grid with S2.
        post_load_fn: Optional function applied to the dataset after loading
              and baseline correction (e.g., amplitude-to-dB conversion).
        item_filter_fn: Optional function applied to STAC items before date
              filtering (e.g., orbit direction filtering).
        item_provider_fn: Optional callable that returns ready-to-load items,
              bypassing CMR-STAC search entirely. Used by the OPERA RTC-S1
              path to build items from the native CMR granule API.
        groupby: How to group STAC items into time slices. Must be "solar_day",
              which merges same-day tiles into one mosaic; :func:`_load_from_stac`
              rejects anything else and says why.
        geobox: Optional odc.geo.geobox.GeoBox specifying the exact output
              grid. When provided, overrides bbox/crs/resolution for the load
              step (bbox is still used for the STAC query).
        mid_longitude: ROI centroid longitude. Supply it with
              ``groupby="solar_day"``: ``existing_dates`` then holds solar days,
              which an item's UTC date does not match wherever the offset crosses
              midnight, and half a committed group would survive the filter.

    Returns:
        Tuple of (dataset, baselines) where:
        - dataset: Corrected xarray Dataset, or None if all dates already exist
        - baselines: Dict mapping date strings to baseline integers (always returned)
    """
    # Resolved ONCE here, where both a geobox and a bbox are in scope, and handed down as a
    # concrete longitude — the inner entry points can only see a bbox. Refuses if this call
    # carries no geometry at all: with `_load_from_stac` accepting solar_day alone, there is
    # no reading of "no longitude" that is correct, and the old default silently stamped UTC.
    mid_longitude = resolve_grouping_longitude(mid_longitude, bbox, geobox=geobox)
    items, baselines = query_stac_items(
        provider=provider,
        collection=collection,
        tile_id=tile_id,
        start_date=start_date,
        end_date=end_date,
        existing_dates=existing_dates,
        bbox=bbox,
        item_filter_fn=item_filter_fn,
        item_provider_fn=item_provider_fn,
        # Asset pruning happens inside the query, so a band not forwarded here is gone
        # before load_stac_items below asks odc for it — turning a valid extra_bands
        # request into a missing-band failure.
        extra_bands=extra_bands,
        mid_longitude=mid_longitude,
    )

    if not items:
        if baselines:
            logger.info("All dates already exist in store - nothing to load")
        return None, baselines

    # Duplicates are selected by `load_stac_items`, not here. It prunes on the same gate, with the
    # same read keys and the same collection answer, and it realigns `baselines` IN PLACE — the
    # dict this function returns — so the provenance reaches the return value without a second
    # pass. Selecting here as well changed nothing: selection is deterministic, and
    # `log_duplicate_selection` returns early when nothing was pruned.
    data = load_stac_items(
        items,
        provider=provider,
        collection=collection,
        baselines=baselines,
        bbox=bbox,
        chunks=chunks,
        extra_bands=extra_bands,
        resampling=resampling,
        crs=crs,
        resolution=resolution,
        post_load_fn=post_load_fn,
        groupby=groupby,
        geobox=geobox,
        # The tile is this function's alone — `load_stac_items` never sees `tile_id` — and it is
        # what distinguishes concurrent same-collection tile ingests in the selection audit log.
        selection_label=f"tile {tile_id}" if tile_id else None,
    )

    return data, baselines


def _prefetch[A, T](fn: Callable[[A], T], arg: A) -> Callable[..., T]:
    """Start ``fn(arg)`` on a daemon thread; return a callable that waits for it.

    The month-streaming loop always waits for its prefetch, so this passes no timeout — what it needs
    from :func:`spawn_abandonable` is the daemon thread, so an abandoned prefetch cannot hold the
    interpreter (or, on the Prefect path, the ECS task) open for the remaining timeout-and-retry
    budget of an HTTP walk whose result nobody wants.
    """
    return spawn_abandonable(fn, arg)


def stream_stac_months(
    *,
    provider: str,
    collection: str,
    tile_id: str | None,
    start_date: str,
    end_date: str,
    bbox: tuple[float, float, float, float] | None,
    existing_dates_fn: Callable[[], set[str]],
    query_fn: Callable[..., tuple[list[Any], dict[str, int]]] | None = None,
    mid_longitude: float | None = None,
    extra_bands: list[str] | None = None,
    log: logging.Logger | logging.LoggerAdapter | None = None,
) -> Iterator[tuple[SolarDayRange, list[Any], dict[str, int]]]:
    """Yield one calendar month of STAC items at a time, prefetching the next.

    Exists because querying a whole window up front retains every item for the run's
    duration: a zone-year is hundreds of thousands of items, which exhausts the worker
    the ingest runs on long before the first date is written. Streaming bounds retention
    to the month being processed plus the one buffered behind it.

    The next month's query runs on a single background thread while the caller processes
    the current one. Depth-1 is intrinsic — one future in flight — and is sufficient
    rather than arbitrary: a month's query is a small fraction of a month's processing,
    so deeper buffering would cost memory to hide nothing. The query is pure network I/O,
    so a thread overlaps it despite the GIL, and a failed query surfaces when the loop
    advances rather than needing a sentinel protocol.

    Items are filtered to the month's OWNED UTC dates before yielding, so the caller sees
    a clean partition and needs no cross-month deduplication — nothing to lose if the
    worker restarts. Empty months are skipped with a log rather than yielded.

    ``existing_dates_fn`` is called once per month at query-submit time, so a month's
    query already excludes dates committed by earlier months. Staleness is harmless: the
    dedupe is an optimisation, and the write path's duplicate-date guard is the actual
    protection.

    Args:
        provider: Provider name, e.g. ``"earth-search"``.
        collection: Collection alias, e.g. ``"sentinel-2-l2a"``.
        tile_id: Tile identifier, or None for bbox queries.
        start_date: Inclusive window start, ``YYYY-MM-DD``.
        end_date: Inclusive window end, ``YYYY-MM-DD``.
        bbox: Optional WGS84 bbox for the spatial query.
        existing_dates_fn: Returns dates already committed; called per month.
        query_fn: Query implementation, for tests that must not touch the network.
            Defaults to :func:`query_stac_items`.
        mid_longitude: ROI centroid longitude, whenever the load groups by solar day.
            Month ownership then uses the SOLAR date, matching
            :func:`group_items_by_date`; without it a solar day straddling a month
            boundary is split across two slices and written twice. Omit for callers
            that group by UTC date.
        extra_bands: Additional assets the caller will load. Forwarded to the query
            because pruning runs THERE — an asset the caller loads but does not
            name here is dropped before the loader ever sees it.
        log: Optional logger.

    Yields:
        ``(month_range, items, baselines)`` per non-empty month, items sorted as
        :func:`query_stac_items` sorts them.
    """
    log = log or logger
    fetch = query_fn or query_stac_items
    months = month_ranges(start_date, end_date)
    log.info("Streaming the STAC query over %d month(s) of %s..%s", len(months), start_date, end_date)

    def run(mr: SolarDayRange) -> tuple[list[Any], dict[str, int]]:
        # Logged at SUBMIT time, not yield time: whether the prefetch is actually
        # overlapping is otherwise only inferable from the gap between one month's last
        # commit and the next month's first, which is a weak signal in a long run.
        t0 = time.monotonic()
        log.info("Querying %s..%s (prefetch)", mr.query_start, mr.query_end)
        result = fetch(
            provider=provider,
            collection=collection,
            tile_id=tile_id,
            start_date=mr.query_start,
            end_date=mr.query_end,
            existing_dates=existing_dates_fn(),
            bbox=bbox,
            # Key the existing-date filter the same way the store was written. The
            # committed dates are solar days whenever this is set, so filtering on UTC
            # dates would drop only the half of a committed group that falls on the
            # matching side of midnight and rewrite the rest as a duplicate slice.
            mid_longitude=mid_longitude,
            # Asset pruning happens HERE, at query time. A band the caller will load but
            # does not name is gone before the loader runs — see _loadable_assets.
            extra_bands=extra_bands,
        )
        log.info("Queried %s..%s in %.1fs", mr.query_start, mr.query_end, time.monotonic() - t0)
        return result

    # One query in flight, one month in the caller's hands: the depth-1 buffer.
    pending = _prefetch(run, months[0]) if months else None
    for i, mr in enumerate(months):
        items, baselines = pending()  # type: ignore[misc]
        if i + 1 < len(months):
            pending = _prefetch(run, months[i + 1])
        # Normalise HERE as well as in query_stac_items: `query_fn` is injectable, so this
        # is the last point that can guarantee the items are solar-day stamped before
        # ownership reads their dates. normalize_to_solar_day is idempotent, so paying it
        # twice on the default path costs a dict build and nothing else.
        # NOT resolved-or-refused, unlike the two call sites above, and the difference is
        # deliberate. This generator has a documented UTC mode for callers that do not group
        # by solar day (`test_the_same_acquisition_stays_in_january_without_a_longitude` pins
        # it), and it hands items back rather than loading them — so it is not the place that
        # can know whether a solar day was wanted. The caller that DOES load resolves at its
        # own entry point.
        items = normalize_to_solar_day(items, mid_longitude=mid_longitude)
        # Ownership by SOLAR day, applied BEFORE the loader sees an item — see
        # ingest.solar_days for why the queried range and the owned range differ.
        owned = owned_items(items, mr)
        dropped = len(items) - len(owned)
        if not owned:
            log.info("Month %s..%s: no new items", mr.own_start, mr.own_end)
            continue
        log.info(
            "Month %s..%s: %d item(s)%s",
            mr.own_start,
            mr.own_end,
            len(owned),
            f" ({dropped} outside the month)" if dropped else "",
        )
        yield mr, owned, baselines
