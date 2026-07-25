"""Sentinel-2 L2A reflectance ingestion for ROI-based regions.

Pure-domain implementation extracted from the reference repo's
``flows/ingest_s2_roi_reflectance.py::process_roi_reflectance``. No
Prefect imports, no ``get_run_logger``, no ``get_client``: callers
supply a connected :class:`dask.distributed.Client` and a logger.

The algorithm is unchanged from the reference:

1. Read ROI metadata + mask and total its pixels for the coverage denominator.
   Persisted on workers on the full-extent path; lazy and window-totalled under
   ``crop_to_live_windows``.
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
import operator
from dataclasses import dataclass
from functools import reduce
from typing import cast, final

import dask.array as da
import dask.distributed
import numpy as np
import odc.stac
import xarray as xr
from odc.geo.geobox import GeoBox
from tenacity import Retrying, before_sleep_log, stop_after_attempt, wait_exponential

from tessera_embeddings.config.ingest import (
    INGEST_CHUNKS,
    INGEST_LOAD_CHUNK_SIZE,
    INGEST_LOAD_CHUNKS,
)
from tessera_embeddings.config.satellites import S2_SCL_INVALID_CLASSES
from tessera_embeddings.ingest.live_windows import live_windows_for_mask
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
from tessera_embeddings.storage.zarr_store import get_existing_dates, write_dataset, write_day_windows

logger = logging.getLogger(__name__)


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


def _sum_over_windows(mask: da.Array, windows: list[tuple[int, int, int, int]]) -> da.Array:
    """Total a full-extent ROI mask over its live windows only.

    Equal to the full-extent sum because the mask is False outside every window,
    and free of double counting because row-band windows are chunk-disjoint. An
    empty window list yields 0 (an all-ocean ROI has no live pixels).
    """
    parts = [mask[y0:y1, x0:x1].sum() for y0, y1, x0, x1 in windows]
    return reduce(operator.add, parts) if parts else da.zeros((), dtype="int64")


def _compute_scl_phase(
    day_items: list,
    geobox: GeoBox,
    load_chunks: dict[str, int],
    roi_mask: da.Array,
    roi_pixel_count: int,
    min_valid_coverage: float,
    client: dask.distributed.Client,
    windows: list[tuple[int, int, int, int]] | None = None,
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
        roi_mask: Boolean ROI mask on the full zone grid. Persisted on workers on
            the full-extent path; lazy under ``windows`` (dask culls its reads to
            the window slices this function and the write path take).
        roi_pixel_count: Number of ``True`` pixels in ``roi_mask``.
        min_valid_coverage: Minimum valid coverage percentage.
        client: Connected Dask client used to persist intermediates.
        windows: Live windows for the cropped path — the coverage reduce runs
            over these only and ``any_valid`` is returned LAZY (never
            persisted full-extent). ``None`` = legacy full-extent behaviour.

    Returns:
        ``(passes_coverage, any_valid)``. ``any_valid`` is persisted on
        workers when ``passes_coverage`` is True (legacy path); ``None``
        when the date fails coverage.
    """
    try:
        scl_ds = _load_scl_only(day_items, geobox, load_chunks)
    except ValueError as exc:
        # Earth-search occasionally publishes asset-incomplete items (missing
        # SCL and/or reflectance bands). odc.stac.load resolves bands eagerly,
        # so one such item in the day group raises "No such band/alias: scl"
        # before any graph is built. Drop the whole date rather than crash the
        # run; Phase 2 would hit the same wall on the same items.
        if "No such band/alias" not in str(exc):
            raise
        logger.warning("Dropping date: SCL load failed on asset-incomplete STAC item(s): %s", exc)
        return False, None

    invalid_classes = np.array(sorted(S2_SCL_INVALID_CLASSES), dtype=scl_ds["scl"].dtype)

    # solar_day grouping fuses all same-day tiles into exactly one time slice
    scl_2d = scl_ds["scl"].isel(time=0)
    any_valid = ~scl_2d.isin(invalid_classes)

    if windows is not None:
        # Cropped: reduce only the live windows (dask culls the graph to their
        # chunks — the mask is False outside every window, so the total is
        # identical to the full reduce by the windows' coverage-totality property).
        # any_valid stays LAZY: persisting it would materialise the full extent,
        # and Phase 2's zeroing pulls only window slices of it anyway.
        masked = any_valid & roi_mask
        parts = [masked.isel(northing=slice(y0, y1), easting=slice(x0, x1)).sum() for y0, y1, x0, x1 in windows]
        valid_count_val = int(reduce(operator.add, parts).compute()) if parts else 0
    else:
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
    crop_to_live_windows: bool = False,
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
        crop_to_live_windows: Write only the chunk-aligned windows that
            intersect the ROI mask (``ingest.live_windows``) instead of the
            full extent — one commit per date either way. Default False
            preserves the byte-identical legacy path while the cropped one
            is validated. The coverage check is cropped too, on BOTH sides of
            its ratio — the SCL reduce and the ROI pixel total each run over
            the windows only, giving identical numbers because the mask is
            False outside them — and neither the mask nor ``any_valid`` is
            persisted, so no full-extent array is ever materialised.

    Returns:
        :class:`IngestResult`. ``status="skipped"`` if zero STAC items
        were returned or zero dates passed the coverage filter.
    """
    log = log or logging.getLogger(__name__)
    reflectance_store = f"{store_path}/reflectance.zarr"
    roi = read_roi_metadata(roi_zarr_path)

    ingest_manifest = IngestManifest.from_roi_store(roi_zarr_path)

    # Live windows for the cropped write path, derived once per run from the same
    # mask this ingest already reads (plain tuples: storage takes no ingest types).
    live_windows: list[tuple[int, int, int, int]] | None = None
    if crop_to_live_windows:
        wins = live_windows_for_mask(roi_zarr_path, window_px=INGEST_LOAD_CHUNK_SIZE, storage_options=storage_options)
        live_windows = [(w.y0, w.y1, w.x0, w.x1) for w in wins]
        log.info("Cropping writes to %d live window(s)", len(wins))

    # LOAD-side blocks, which are a multiple of the store's chunks (see
    # config.ingest.INGEST_LOAD_CHUNK_SIZE): fewer, larger blocks mean a smaller
    # graph, and the write rechunks them down to store chunks as a pure split.
    spatial_chunks = {"northing": INGEST_LOAD_CHUNKS["northing"], "easting": INGEST_LOAD_CHUNKS["easting"]}

    roi_mask = read_roi_mask(roi_zarr_path, spatial_chunks, storage_options=storage_options)
    if live_windows is not None:
        # Cropped: mask stays LAZY, total comes from the live windows only.
        #
        # No persist — it would materialise the whole zone grid, while every
        # consumer (the SCL reduce below, apply_roi_mask) slices to windows, so
        # dask culls the reads and a per-date re-read beats pinning the grid.
        #
        # The window total equals the full-extent total because the mask is False
        # outside every window, and row bands are chunk-disjoint so nothing is
        # counted twice. _compute_scl_phase relies on that same property for the
        # numerator: both sides of the coverage ratio must stay cropped together,
        # or every percentage is silently skewed.
        roi_pixel_count = int(_sum_over_windows(roi_mask, live_windows).compute())
    else:
        # Persist the ROI mask on workers so per-day graphs reference small
        # future keys instead of re-reading from Zarr each time.
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
            day_items,
            roi.geobox,
            INGEST_LOAD_CHUNKS,
            roi_mask,
            roi_pixel_count,
            min_valid_coverage,
            client,
            windows=live_windows,
        )
        if not passes:
            total_filtered += 1
            continue

        # Phase 2: load all bands with the same solar_day grouping.
        # Reflectance bands resample bilinear; load_stac_items pins the
        # categorical SCL band to nearest so its class codes stay valid.
        day_ds = load_stac_items(
            day_items,
            provider=provider,
            collection=collection,
            baselines=baselines,
            bbox=roi.bbox_wgs84,
            chunks=INGEST_LOAD_CHUNKS,
            extra_bands=["scl"],
            resampling="bilinear",
            groupby="solar_day",
            geobox=roi.geobox,
        )
        date = day_ds.time.dt.date.values[0]
        day_ds["time"] = [np.datetime64(date, "ns")]
        if any_valid is not None:
            for var in day_ds.data_vars:
                if str(var) != "scl":
                    day_ds[str(var)] = day_ds[str(var)].where(any_valid, other=0)

        day_ds = apply_roi_mask(day_ds, roi_zarr_path, spatial_chunks, roi_mask=roi_mask)

        # Tenacity retry on intermittent GDAL errors under high parallelism.
        # Icechunk writes are atomic, so retry is safe — a failed cropped date
        # commits nothing (batched_region_writes) and the retry starts clean.
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                if live_windows is not None:
                    write_day_windows(
                        reflectance_store,
                        day_ds,
                        live_windows,
                        roi=roi,
                        manifest=ingest_manifest,
                        baselines=baselines,
                        tile_id=roi_zarr_path,
                        crs=roi.native_crs,
                        chunks=INGEST_CHUNKS,
                    )
                else:
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
