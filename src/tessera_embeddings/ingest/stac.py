"""STAC data ingestion utilities with baseline correction.

This module provides functions for loading satellite data via STAC catalogs
using odc.stac.load, applying processing baseline corrections, and filtering
by existing dates.

Supports multiple providers (Earth Search, Planetary Computer) and collections
(Sentinel-2 L2A, Sentinel-1 GRD, Landsat).
"""

import logging
import time
from collections.abc import Callable, Iterator
from itertools import count
from typing import Any

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox
from pystac import Item
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO

from tessera_embeddings.config import (
    PROVIDERS,
    S2_BASELINE_OFFSET,
    S2_BASELINE_THRESHOLD,
    CollectionConfig,
    STACProvider,
)
from tessera_embeddings.config.environment import configure_gdal_environment
from tessera_embeddings.config.ingest import INGEST_CHUNKS
from tessera_embeddings.ingest._http import make_logging_retry, spawn_abandonable
from tessera_embeddings.ingest.catalogue_refusal import (
    CatalogueRequest,
    bbox_area_label,
    raise_catalogue_query_error,
)
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


_STAC_RETRY = make_logging_retry(
    "STAC",
    total=8,
    backoff_factor=2,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]),
    respect_retry_after_header=True,
)
# (connect_timeout, read_timeout) in seconds. Without an explicit timeout,
# a stalled TCP connection blocks indefinitely and the retry logic never fires.
_STAC_TIMEOUT = (10, 60)


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
    """Extract processing baseline from a STAC item as an integer.

    Args:
        item: pystac Item or similar object with properties dict

    Returns:
        Baseline as integer (e.g., "04.00" -> 400, "05.10" -> 510)
        Returns 0 if baseline property is not found.
    """
    try:
        baseline_str = item.properties.get("s2:processing_baseline", "")
        if not baseline_str:
            return 0
        # Convert "04.00" to 400, "05.10" to 510
        # Use round() to avoid float precision issues (5.10 * 100 = 509.999...)
        return round(float(baseline_str) * 100)
    except (AttributeError, ValueError, TypeError):
        return 0


def extract_baselines(items: list[Any]) -> dict[str, int]:
    """The processing baseline to apply per date, over exactly ``items``.

    Last item of a date wins. Where a date's tiles disagree, that makes the pick a
    property of the caller's sort order, which is why this is derived from the items
    being LOADED rather than computed once per query: a map built over a wider list can
    name a baseline belonging to an item the loader never sees, and the reflectance
    offset it selects is applied to the pixels of the items it does.

    Args:
        items: STAC items, in the order they will be loaded.

    Returns:
        Dict mapping date strings (YYYY-MM-DD) to baseline integers.
    """
    baselines = {}
    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        baselines[date_str] = _extract_baseline(item)
    return baselines


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
        "preserve_original_order": True,
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

    res_label = f"geobox {geobox.shape}" if geobox is not None else f"{load_kwargs.get('resolution')}m"
    logger.info(f"Loading {len(items)} items with odc.stac.load (bands={all_bands}, resolution={res_label})")

    ds = odc.stac.load(items, **load_kwargs)
    ds = normalize_odc_dims(ds)

    return ds


def _apply_baseline_corrections_by_date(
    ds: xr.Dataset,
    baselines: dict[str, int],
    baseline_threshold: int = S2_BASELINE_THRESHOLD,
    baseline_offset: int = S2_BASELINE_OFFSET,
    bands: list[str] | None = None,
    preserve_low_values: bool = False,
) -> xr.Dataset:
    """Apply baseline corrections to dataset based on per-date baselines.

    Unlike _apply_baseline_correction which applies a single baseline to all data,
    this function handles datasets with multiple dates where each date may have
    a different baseline.

    Two modes of operation:
    - Default (preserve_low_values=False): Subtracts offset from all pixels,
      allowing values below abs(offset) to go negative. Matches
      generate_cloudmask.py behavior where negative values signal nodata/dark.
    - Tessera mode (preserve_low_values=True): Only subtracts from pixels
      where value >= abs(offset), leaving low values unchanged. Matches
      Tessera's harmonize_arr() behavior.

    Args:
        ds: xarray Dataset with time dimension
        baselines: Dict mapping date strings (YYYY-MM-DD) to baseline integers
        baseline_threshold: Minimum baseline requiring correction (default: S2 threshold)
        baseline_offset: Value to add to pixels (typically negative, default: S2 offset)
        bands: List of band names to correct. If None, corrects all data variables.
        preserve_low_values: When True, only subtract from pixels where
            value >= abs(offset), leaving low/dark pixels unchanged. When False,
            subtract from all pixels (low values go negative).

    Returns:
        Corrected xarray Dataset
    """
    bands_to_correct: list[str] = bands if bands is not None else list(ds.data_vars)  # type: ignore[arg-type]

    # Build a boolean mask for which time slices need correction
    dates = [str(t.values)[:10] for t in ds.time]
    needs_correction = xr.DataArray(
        [baselines.get(d, 0) >= baseline_threshold for d in dates],
        dims=["time"],
        coords={"time": ds.time},
    )

    corrected_count = sum(needs_correction.values)
    if corrected_count > 0:
        logger.info(
            f"Applying baseline correction to {corrected_count}/{len(dates)} dates for bands: {bands_to_correct}"
        )

    abs_offset = abs(baseline_offset)
    clamp_max = 65535 + baseline_offset  # 64535 for S2

    # Apply correction vectorized across time
    result_vars = {}
    for var in ds.data_vars:
        if var in bands_to_correct:
            if preserve_low_values:
                # Tessera mode: only subtract from pixels >= abs(offset),
                # leave low/dark pixels unchanged
                pixel_eligible = ds[var] >= abs_offset
                corrected_values = xr.where(
                    pixel_eligible,
                    ds[var].clip(max=clamp_max) + baseline_offset,
                    ds[var],
                )
                corrected = xr.where(
                    needs_correction,
                    corrected_values,
                    ds[var],
                ).astype(np.int16)
            else:
                # Default: subtract from all pixels, low values go negative
                clamped = ds[var].clip(max=clamp_max).astype(np.int32)
                corrected_values = clamped + baseline_offset
                corrected = xr.where(
                    needs_correction,
                    corrected_values,
                    ds[var].astype(np.int32),
                ).astype(np.int16)
            result_vars[str(var)] = corrected
        else:
            result_vars[str(var)] = ds[str(var)]

    return xr.Dataset(result_vars, coords=ds.coords, attrs=ds.attrs)


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
            see :mod:`~tessera_embeddings.ingest.catalogue_refusal`.
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
    stac_io = StacApiIO(max_retries=_STAC_RETRY, timeout=_STAC_TIMEOUT)
    try:
        client = Client.open(provider.catalog_url, stac_io=stac_io)
    except Exception as exc:
        # Page 0: the catalogue root, fetched before any search exists. Named separately
        # so a refusal here is not read as a refusal of the search it precedes.
        raise_catalogue_query_error(
            CatalogueRequest(collection_config.collection_id, window, "catalogue-root", 0), exc, log=logger
        )
    logger.info(f"STAC catalog opened in {time.monotonic() - t0:.1f}s, executing search")

    items: list[Any] = []
    seen: set[str] = set()
    keep_assets = _loadable_assets(collection_config, extra_bands)
    for sub_bbox in split_antimeridian_bbox(bbox):
        query_params = _build_stac_query(collection_config, tile_id, start_date, end_date, bbox=sub_bbox)
        # Read off the query that was BUILT rather than re-deciding property-versus-bbox
        # here: a second copy of that branch could disagree with the one that ran, and
        # then the log would name a request the catalogue was never asked. Tolerant of a
        # query carrying neither term, because a diagnostic must never be the thing that
        # raises — an unnamed area still leaves the collection, window and page named.
        area = _area_label(query_params, tile_id)
        search = client.search(**query_params, limit=provider.max_page_size, max_items=None)
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
                break
            except Exception as exc:
                # Scoped to the FETCH alone. Wrapping the body below as well would
                # classify our own validation failure as a catalogue refusal, which the
                # retry policy would then act on.
                raise_catalogue_query_error(request, exc, log=logger, items_so_far=len(items))
            for raw in page.get("features", []):
                # Dedupe across the two halves: a granule straddling +/-180 is returned by
                # both searches, and loading it twice would double-count the solar day.
                # `id` is required by the STAC spec; an item without one cannot be deduped,
                # and defaulting it would collapse EVERY such item into a single entry —
                # so say so rather than silently drop data.
                if raw.get("id") is None:
                    raise ValueError(f"STAC item without an 'id' from {provider.catalog_url} — cannot dedupe it")
                item_id = str(raw["id"])
                if item_id not in seen:
                    seen.add(item_id)
                    items.append(Item.from_dict(_prune_item_dict(raw, keep_assets)))
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

    # THE single solar-offset application for this path. Everything below — the
    # painter's-algorithm sort, the baseline map, the existing-date dedup, and every
    # consumer of the returned items — then reads the solar day straight off
    # `item.datetime` with no offset of its own. Before this, the sort and the baseline
    # map keyed on the UTC date while the loader grouped by solar day, so on a day
    # straddling UTC midnight the group was not actually sorted clearest-last and half
    # its baseline entries never matched.
    # Same rule as the date probe above, and it matters more here: this is THE single
    # solar-offset application for the path, so a UTC-day stamp taken by default would be
    # read as the solar day by the sort, the baseline map, the dedup and every consumer.
    items = normalize_to_solar_day(items, mid_longitude=resolve_grouping_longitude(mid_longitude, bbox))

    # Sort by (solar date, cloud_cover DESCENDING) so same-day tiles are adjacent and the
    # CLEAREST tile comes LAST.
    #
    # Last, not first, because `load_kwargs` below sets `preserve_original_order=True` with
    # `groupby="solar_day"`, and odc.stac's painter keeps the last item written for a pixel. Sorting
    # clearest-first therefore let the CLOUDIEST scene win wherever two scenes of one solar day
    # overlap — the opposite of the intent, and silent, because the output has the right shape and
    # the right dates. The campaign's own S2 path (`s2_roi._ingest_s2_dates`) already reverses the
    # order for exactly this reason and says so; the two orderings disagreed, and this generic API
    # was the one that was wrong.
    #
    # Keyed on the SOLAR day, matching `normalize_to_solar_day` just above and the loader's own
    # grouping. Keying on the UTC date instead splits a group the loader treats as one, in the
    # far-eastern and far-western zones where the solar offset crosses midnight.
    if collection_config.has_scl:
        items.sort(
            key=lambda item: (
                item.datetime.strftime("%Y-%m-%d"),
                -float(item.properties.get("eo:cloud_cover", 100)),
            )
        )

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
    preserve_low_values: bool = False,
    groupby: str = "solar_day",
    geobox: GeoBox | None = None,
) -> xr.Dataset:
    """Load STAC items into an xarray Dataset with corrections applied.

    Wraps ``_load_from_stac`` and adds baseline correction and post-load
    transforms. This is the load half of ``ingest_tile``; pair with
    ``query_stac_items`` for the full workflow.

    Args:
        items: List of pystac Items to load (must be non-empty)
        provider: Provider name (e.g., "earth-search")
        collection: Collection alias (e.g., "sentinel-2-l2a")
        baselines: Dict mapping date strings to baseline integers. Required
            for collections that need baseline correction.
        bbox: Optional bounding box for spatial subsetting
        chunks: Optional chunk sizes for odc.stac.load
        extra_bands: Additional bands to load (e.g., ["scl"])
        resampling: Resampling method for primary bands (default "bilinear")
        crs: Explicit CRS override for odc.stac.load
        resolution: Override pixel resolution in metres
        post_load_fn: Optional function applied after loading and correction
        preserve_low_values: Tessera-style baseline correction mode
        groupby: How to group items into time slices. Must be "solar_day" —
              :func:`_load_from_stac` rejects anything else and says why.
        geobox: Optional output grid specification

    Returns:
        Corrected xarray Dataset
    """
    collection_config = _get_collection_config(provider, collection)

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
    )

    if collection_config.requires_baseline_correction and baselines:
        loaded_dates = {str(t.values)[:10] for t in data.time}
        filtered_baselines = {d: b for d, b in baselines.items() if d in loaded_dates}
        data = _apply_baseline_corrections_by_date(
            data,
            filtered_baselines,
            baseline_threshold=collection_config.baseline_threshold,  # type: ignore[arg-type]
            baseline_offset=collection_config.baseline_offset,
            bands=list(collection_config.bands),
            preserve_low_values=preserve_low_values,
        )

    if post_load_fn is not None:
        data = post_load_fn(data)

    return data


def group_items_by_date(items: list[Any]) -> dict[str, list[Any]]:
    """Group STAC items by day, matching how the loader will group them.

    Pass ``mid_longitude`` — the ROI geobox centroid's longitude in WGS84 — whenever the
    load uses ``groupby="solar_day"``, which is every S2 path. The loader groups by LOCAL
    solar day, shifting each timestamp by that longitude; grouping here by UTC calendar
    date instead lets the two disagree, and a group the caller believes is one day then
    loads as TWO time slices.

    That divergence is not hypothetical and not uniform: it appears where the solar offset
    is large enough to push acquisitions across UTC midnight, i.e. the far-eastern and
    far-western zones, and never in the middle longitudes. Downstream code assumes one
    slice per group (the cloud mask is reduced to a single 2-D slice), so the mismatch
    surfaces as a dimension conflict rather than as anything that names the cause.

    Omitting ``mid_longitude`` keeps the old UTC-date behaviour, which is correct only for
    callers that do not group by solar day.

    Items should typically be pre-sorted by (date, cloud_cover) via ``query_stac_items``
    so that within each group, clearer tiles come first.

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
    preserve_low_values: bool = False,
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
        preserve_low_values: When True, baseline correction only subtracts
              from pixels >= abs(offset), matching Tessera's harmonize_arr().
              When False (default), subtracts from all pixels.
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
        preserve_low_values=preserve_low_values,
        groupby=groupby,
        geobox=geobox,
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
