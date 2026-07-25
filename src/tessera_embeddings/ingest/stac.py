"""STAC data ingestion utilities with baseline correction.

This module provides functions for loading satellite data via STAC catalogs
using odc.stac.load, applying processing baseline corrections, and filtering
by existing dates.

Supports multiple providers (Earth Search, Planetary Computer) and collections
(Sentinel-2 L2A, Sentinel-1 GRD, Landsat).
"""

import calendar
import datetime
import logging
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from tessera_embeddings.ingest._http import make_logging_retry

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


def _extract_baselines(items: list[Any]) -> dict[str, int]:
    """Extract processing baselines for multiple STAC items.

    Args:
        items: List of pystac Items

    Returns:
        Dict mapping date strings (YYYY-MM-DD) to baseline integers
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

    Args:
        items: List of pystac Items
        existing_dates: Set of date strings (YYYY-MM-DD) already processed

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
    groupby: str = "time",
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
        groupby: How to group STAC items into time slices. "time" (default)
              uses each item's exact timestamp. "solar_day" groups items by
              solar day based on longitude, merging same-pass tiles from
              adjacent orbits into a single mosaic. Use "solar_day" for
              ROI-based queries that span multiple tiles.
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


def _query_stac_items(
    provider: STACProvider,
    collection_config: CollectionConfig,
    tile_id: str | None,
    start_date: str,
    end_date: str,
    bbox: tuple[float, float, float, float] | None = None,
    item_provider_fn: Callable[[], list[Any]] | None = None,
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

    Returns:
        List of pystac Items
    """
    if item_provider_fn is not None:
        return item_provider_fn()

    t0 = time.monotonic()
    logger.info(f"Opening STAC catalog: {provider.catalog_url}")
    stac_io = StacApiIO(max_retries=_STAC_RETRY, timeout=_STAC_TIMEOUT)
    client = Client.open(provider.catalog_url, stac_io=stac_io)
    logger.info(f"STAC catalog opened in {time.monotonic() - t0:.1f}s, executing search")

    items: list[Any] = []
    seen: set[str] = set()
    for sub_bbox in split_antimeridian_bbox(bbox):
        query_params = _build_stac_query(collection_config, tile_id, start_date, end_date, bbox=sub_bbox)
        search = client.search(**query_params, limit=provider.max_page_size, max_items=None)
        for item in search.items():
            # Dedupe across the two halves: a granule straddling +/-180 is returned by
            # both searches, and loading it twice would double-count the solar day.
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

    # Sort by (date, cloud_cover) so same-day tiles are adjacent and the
    # clearest tile comes first for SCL-based mosaicking.
    if collection_config.has_scl:
        items.sort(
            key=lambda item: (
                str(item.datetime)[:10],
                float(item.properties.get("eo:cloud_cover", 100)),
            )
        )

    baselines = _extract_baselines(items)

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
    groupby: str = "time",
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
        groupby: How to group items into time slices (default "time")
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
    """Group STAC items by calendar date string.

    Items should typically be pre-sorted by (date, cloud_cover) via
    ``query_stac_items`` so that within each group, clearer tiles come first.

    Args:
        items: List of pystac Items

    Returns:
        Dict mapping date strings (YYYY-MM-DD) to lists of items,
        preserving insertion order and within-group order.
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
    groupby: str = "time",
    geobox: GeoBox | None = None,
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
        groupby: How to group STAC items into time slices. "time" (default)
              uses exact timestamps. "solar_day" merges same-day tiles into
              a single mosaic — use for ROI queries spanning multiple tiles.
        geobox: Optional odc.geo.geobox.GeoBox specifying the exact output
              grid. When provided, overrides bbox/crs/resolution for the load
              step (bbox is still used for the STAC query).

    Returns:
        Tuple of (dataset, baselines) where:
        - dataset: Corrected xarray Dataset, or None if all dates already exist
        - baselines: Dict mapping date strings to baseline integers (always returned)
    """
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


@dataclass(frozen=True)
class MonthRange:
    """One calendar month's slice of a query window: what to ASK for vs what to KEEP.

    ``query_*`` is the range handed to the STAC query; ``own_*`` is the range of UTC
    calendar dates this slice is responsible for. They differ by one day of padding
    on the query's end — see :func:`iter_month_ranges`.
    """

    query_start: str
    query_end: str
    own_start: str
    own_end: str


def iter_month_ranges(start_date: str, end_date: str) -> list[MonthRange]:
    """Partition an inclusive ``[start_date, end_date]`` window into calendar months.

    Owned ranges PARTITION the window: every UTC calendar date belongs to exactly one
    month, so a date can never be processed twice or skipped. That is what makes the
    slices independent — no cross-month state is needed to deduplicate, which matters
    because the ingest runs on a worker that may be restarted at any point.

    The query end is padded by one day (clamped to ``end_date``) because a date-only
    interval end is expanded to that day's last second: without the pad, items in the
    final seconds of a month's last day could fall outside every slice's query. The
    padding cannot cause double-processing — ownership is by UTC date, and the padded
    day is owned by the NEXT month.

    Args:
        start_date: Inclusive window start, ``YYYY-MM-DD``.
        end_date: Inclusive window end, ``YYYY-MM-DD``.

    Returns:
        One :class:`MonthRange` per calendar month intersecting the window, in order.

    Raises:
        ValueError: If ``end_date`` precedes ``start_date``.
    """
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date} precedes start_date {start_date}")

    ranges: list[MonthRange] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        last_day = calendar.monthrange(year, month)[1]
        own_start = max(start, datetime.date(year, month, 1))
        own_end = min(end, datetime.date(year, month, last_day))
        query_end = min(end, own_end + datetime.timedelta(days=1))
        ranges.append(
            MonthRange(
                query_start=own_start.isoformat(),
                query_end=query_end.isoformat(),
                own_start=own_start.isoformat(),
                own_end=own_end.isoformat(),
            )
        )
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return ranges


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
    log: logging.Logger | logging.LoggerAdapter | None = None,
) -> Iterator[tuple[MonthRange, list[Any], dict[str, int]]]:
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
        log: Optional logger.

    Yields:
        ``(month_range, items, baselines)`` per non-empty month, items sorted as
        :func:`query_stac_items` sorts them.
    """
    log = log or logger
    fetch = query_fn or query_stac_items
    months = iter_month_ranges(start_date, end_date)
    log.info("Streaming the STAC query over %d month(s) of %s..%s", len(months), start_date, end_date)

    def run(mr: MonthRange) -> tuple[list[Any], dict[str, int]]:
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
        )
        log.info("Queried %s..%s in %.1fs", mr.query_start, mr.query_end, time.monotonic() - t0)
        return result

    # max_workers=1 IS the depth-1 buffer: one query in flight, one month in the
    # caller's hands. cancel_futures on exit so a failure mid-month does not block on
    # the next month's in-flight HTTP walk.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stac-prefetch") as pool:
        try:
            pending = pool.submit(run, months[0]) if months else None
            for i, mr in enumerate(months):
                items, baselines = pending.result()  # type: ignore[union-attr]
                if i + 1 < len(months):
                    pending = pool.submit(run, months[i + 1])
                owned = [it for it in items if mr.own_start <= str(it.datetime)[:10] <= mr.own_end]
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
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
