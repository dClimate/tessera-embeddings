"""Sentinel-2 L2A reflectance ingestion for ROI-based regions.

Pure-domain implementation extracted from the reference repo's
``flows/ingest_s2_roi_reflectance.py::process_roi_reflectance``. No
Prefect imports, no ``get_run_logger``, no ``get_client``: callers
supply a connected :class:`dask.distributed.Client` and a logger.

The algorithm is unchanged from the reference:

1. Read ROI metadata + mask, persist the mask on workers.
2. Query STAC for the full date range; sort cloudiest-first so the
   painter's-algorithm mosaic picks the clearest tile last.
3. Group items by ``solar_day``.
4. For each day:

   * Phase 1 — load only SCL, compute coverage from the same
     ``solar_day`` mosaic; reject days below ``min_valid_coverage``.
   * Phase 2 — load all bands, mask invalid pixels via the Phase 1
     ``any_valid`` mask, apply ROI mask, write via
     :func:`tessera_embeddings.storage.zarr_store.write_dataset` with a
     narrow tenacity retry on transient GDAL errors.

This module imports nothing from Prefect or any cloud provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast, final

import dask.array as da
import dask.distributed
import numpy as np
import odc.stac
import xarray as xr
from odc.geo.geobox import GeoBox
from tenacity import Retrying, before_sleep_log, stop_after_attempt, wait_exponential

from tessera_embeddings.config.ingest import INGEST_CHUNKS
from tessera_embeddings.config.satellites import S2_SCL_INVALID_CLASSES
from tessera_embeddings.ingest.roi import read_roi_mask, read_roi_metadata
from tessera_embeddings.ingest.roi_processing import DEFAULT_MIN_VALID_COVERAGE, apply_roi_mask
from tessera_embeddings.ingest.stac import (
    chunks_to_odc,
    group_items_by_date,
    load_stac_items,
    normalize_odc_dims,
    query_stac_items,
)
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import get_existing_dates, write_dataset


@final
@dataclass(frozen=True)
class IngestResult:
    """Return value from :func:`ingest_s2_roi_reflectance`.

    Replaces the dict that the reference repo's ``@task`` returned.
    Task shells convert back to a dict via ``dataclasses.asdict()`` at
    the Prefect boundary.

    Attributes:
        roi_path: Echo of the input ``roi_zarr_path`` for caller bookkeeping.
        status: ``"success"`` if at least one date was written, otherwise
            ``"skipped"``.
        dates_processed: Number of dates whose reflectance was written.
        dates_filtered_coverage: Number of dates rejected by the SCL-based
            coverage filter.
    """

    roi_path: str
    status: str
    dates_processed: int
    dates_filtered_coverage: int


def _load_scl_only(items: list, geobox: GeoBox, load_chunks: dict[str, int]) -> xr.Dataset:
    """Load only the SCL band via a minimal ``odc.stac.load`` call.

    Bypasses :func:`load_stac_items` intentionally — SCL is categorical
    and needs no baseline correction. ``groupby="solar_day"`` fuses
    overlapping tiles into a single time slice via painter's algorithm
    (last item wins). Items must be pre-sorted cloudiest-first so the
    clearest tile paints last.
    """
    ds = odc.stac.load(
        items,
        bands=["scl"],
        groupby="solar_day",
        preserve_original_order=True,
        geobox=geobox,
        resampling="nearest",
        chunks=cast(dict, chunks_to_odc(load_chunks)),
    )
    return normalize_odc_dims(ds)


def _compute_scl_phase(
    day_items: list,
    geobox: GeoBox,
    load_chunks: dict[str, int],
    roi_mask: da.Array,
    roi_pixel_count: int,
    min_valid_coverage: float,
    client: dask.distributed.Client,
) -> tuple[bool, xr.DataArray | None]:
    """Phase 1: load only SCL, compute coverage and per-pixel validity.

    Uses ``solar_day`` grouping so overlapping tiles are fused into one
    mosaic via painter's algorithm (items must be pre-sorted
    cloudiest-first so the clearest tile paints last). The resulting
    ``any_valid`` mask is coherent with Phase 2, which uses the same
    grouping and sort order.

    Args:
        day_items: STAC items for one calendar day, pre-sorted by
            descending ``eo:cloud_cover`` so the clearest tile wins.
        geobox: Target GeoBox for the ROI.
        load_chunks: Chunk sizes for ``odc.stac.load``.
        roi_mask: Pre-computed boolean ROI mask, persisted on workers.
        roi_pixel_count: Number of ``True`` pixels in ``roi_mask``.
        min_valid_coverage: Minimum valid coverage percentage.
        client: Connected Dask client used to persist intermediates.

    Returns:
        ``(passes_coverage, any_valid)``. ``any_valid`` is persisted on
        workers when ``passes_coverage`` is True; ``None`` otherwise.
    """
    scl_ds = _load_scl_only(day_items, geobox, load_chunks)
    invalid_classes = np.array(sorted(S2_SCL_INVALID_CLASSES), dtype=scl_ds["scl"].dtype)

    # solar_day grouping fuses all same-day tiles into exactly one time slice
    scl_2d = scl_ds["scl"].isel(time=0)
    any_valid = ~scl_2d.isin(invalid_classes)
    valid_count = (any_valid & roi_mask).sum()

    any_valid, valid_count = client.persist([any_valid, valid_count])
    valid_count_val = valid_count.compute().item()

    passes = float(100.0 * valid_count_val / roi_pixel_count) >= min_valid_coverage
    if not passes:
        return False, None
    return True, any_valid


def ingest_s2_roi_reflectance(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    client: dask.distributed.Client,
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
    provider: str = "earth-search",
    collection: str = "sentinel-2-l2a",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    storage_options: dict | None = None,
) -> IngestResult:
    """Ingest S2 L2A reflectance for an ROI defined by a Zarr mask.

    Orchestrator-unaware. Same algorithm as the reference repo's
    ``flows/ingest_s2_roi_reflectance.py::process_roi_reflectance``.

    Args:
        roi_zarr_path: Path to the Zarr ROI store (any fsspec-compatible
            URI).
        start_date: Inclusive start date (``YYYY-MM-DD``).
        end_date: Inclusive end date (``YYYY-MM-DD``).
        store_path: Base path for satellite mosaics; the function
            creates ``reflectance.zarr`` underneath.
        client: Connected :class:`dask.distributed.Client`. The flow
            shell or plain runner provides this. We do NOT call
            :func:`dask.distributed.get_client` inside.
        min_valid_coverage: Minimum percentage of valid ROI pixels
            (computed from SCL) required to keep a date.
        provider: STAC provider key from
            :data:`tessera_embeddings.config.providers.PROVIDERS`.
        collection: Collection alias within the provider.
        log: Optional logger; defaults to ``logging.getLogger(__name__)``.
        storage_options: fsspec storage options for reading the ROI
            mask. ``None`` lets fsspec auto-detect from the URI.

    Returns:
        :class:`IngestResult`. ``status="skipped"`` if zero STAC items
        were returned or zero dates passed the coverage filter.
    """
    log = log or logging.getLogger(__name__)
    reflectance_store = f"{store_path}/reflectance.zarr"
    roi = read_roi_metadata(roi_zarr_path)

    ingest_manifest = IngestManifest.from_roi_store(roi_zarr_path)

    # time=1 matches INGEST_CHUNKS so each date is an independent Dask task:
    # fully parallel across dates with no rechunk at write time.
    spatial_chunks = {"northing": INGEST_CHUNKS["northing"], "easting": INGEST_CHUNKS["easting"]}

    # Persist the ROI mask on workers so per-day graphs reference small
    # future keys instead of re-reading from Zarr each time.
    roi_mask = read_roi_mask(roi_zarr_path, spatial_chunks, storage_options=storage_options)
    roi_mask = client.persist(roi_mask)
    roi_pixel_count = int(roi_mask.sum().compute())

    existing_dates = get_existing_dates(reflectance_store)

    items, baselines = query_stac_items(
        provider=provider,
        collection=collection,
        tile_id=None,
        start_date=start_date,
        end_date=end_date,
        existing_dates=existing_dates,
        bbox=roi.bbox_wgs84,
    )

    if not items:
        log.info("No STAC items found for date range %s..%s", start_date, end_date)
        return IngestResult(roi_path=roi_zarr_path, status="skipped", dates_processed=0, dates_filtered_coverage=0)

    # query_stac_items sorts by (date, cloud_cover ASC). Re-sort cloudiest-first
    # so the clearest tile paints last (wins) in solar_day's painter's algorithm.
    # group_items_by_date preserves this within-group order.
    items.sort(key=lambda it: (str(it.datetime)[:10], -float(it.properties.get("eo:cloud_cover", 100))))
    items_by_date = group_items_by_date(items)
    total_processed = 0
    total_filtered = 0

    for day_items in items_by_date.values():
        # Phase 1: SCL-only coverage check — tiny graph.
        passes, any_valid = _compute_scl_phase(
            day_items, roi.geobox, INGEST_CHUNKS, roi_mask, roi_pixel_count, min_valid_coverage, client
        )
        if not passes:
            total_filtered += 1
            continue

        # Phase 2: load all bands with the same solar_day grouping.
        day_ds = load_stac_items(
            day_items,
            provider=provider,
            collection=collection,
            baselines=baselines,
            bbox=roi.bbox_wgs84,
            chunks=INGEST_CHUNKS,
            extra_bands=["scl"],
            resampling="nearest",
            groupby="solar_day",
            geobox=roi.geobox,
        )
        date = day_ds.time.dt.date.values[0]
        day_ds["time"] = [np.datetime64(date, "ns")]
        if any_valid is not None:
            for var in day_ds.data_vars:
                if str(var) != "scl":
                    day_ds[str(var)] = day_ds[str(var)].where(any_valid, other=0)

        day_ds, _ = apply_roi_mask(day_ds, roi_zarr_path, spatial_chunks, roi_mask=roi_mask)

        # Tenacity retry on intermittent GDAL errors under high parallelism.
        # Icechunk writes are atomic, so retry is safe.
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                write_dataset(
                    reflectance_store,
                    day_ds,
                    tile_id=roi_zarr_path,
                    baselines=baselines,
                    chunks=INGEST_CHUNKS,
                    manifest=ingest_manifest,
                    crs=roi.native_crs,
                )
        total_processed += 1

    log.info("%d/%d dates passed coverage filter", len(items_by_date) - total_filtered, len(items_by_date))

    if total_processed == 0:
        return IngestResult(
            roi_path=roi_zarr_path,
            status="skipped",
            dates_processed=0,
            dates_filtered_coverage=total_filtered,
        )

    return IngestResult(
        roi_path=roi_zarr_path,
        status="success",
        dates_processed=total_processed,
        dates_filtered_coverage=total_filtered,
    )
