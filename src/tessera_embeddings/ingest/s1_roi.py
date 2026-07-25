"""Sentinel-1 OPERA RTC SAR ingestion for ROI-based regions.

Pure-domain implementation extracted from the reference repo's
``flows/ingest_s1_roi_sar.py::process_roi_sar``. No Prefect imports,
no ``get_run_logger``, no ``get_client``, no direct env-var reads for
secrets — callers supply a connected :class:`dask.distributed.Client`,
a logger, and (where required) a credential callback.

Algorithm (unchanged from the reference):

1. Read ROI metadata + lazy mask once per batch (so Dask graphs
   reference fresh keys whenever credentials are refreshed).
2. Walk the date range in batches of ``batch_days`` to keep each
   ``compute()`` call's task graph manageable.
3. Each batch:

   * Refresh STS credentials if more than ``cred_refresh_interval``
     seconds have elapsed since the last refresh (default 30 minutes,
     well within the 1-hour STS TTL).
   * Build an OPERA RTC item filter for the orbit / bounding box /
     batch window.
   * Call :func:`ingest_tile` to produce a per-batch ``xarray.Dataset``.
   * Apply the ROI mask, then write via
     :func:`tessera_embeddings.storage.zarr_store.write_dataset` with a
     narrow tenacity retry on transient GDAL errors.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, final

import dask.distributed
from tenacity import Retrying, before_sleep_log, stop_after_attempt, wait_exponential

from tessera_embeddings.config.ingest import INGEST_CHUNKS
from tessera_embeddings.ingest.live_windows import live_windows_for_mask
from tessera_embeddings.ingest.opera_query import make_s1_item_provider
from tessera_embeddings.ingest.roi import read_roi_mask, read_roi_metadata
from tessera_embeddings.ingest.roi_processing import apply_roi_mask
from tessera_embeddings.ingest.stac import ingest_tile
from tessera_embeddings.ingest.transforms import amplitude_to_db
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import get_existing_dates, write_dataset, write_day_windows

DEFAULT_CRED_REFRESH_INTERVAL_SEC = 30 * 60
"""STS credentials are refreshed at most every 30 minutes (1-hour TTL minus headroom)."""

S1Orbit = Literal["ascending", "descending"]


@final
@dataclass(frozen=True)
class SarIngestResult:
    """Return value from :func:`ingest_s1_roi_sar`.

    Replaces the dict that the reference repo's ``@task`` returned.
    Task shells convert back to a dict via ``dataclasses.asdict()`` at
    the Prefect boundary.

    Attributes:
        roi_path: Echo of the input ``roi_zarr_path``.
        status: ``"success"`` if at least one date was written, else
            ``"skipped"``.
        dates_processed: ``{orbit: count}``. The single-orbit shape
            mirrors the reference repo and lets a multi-orbit caller
            merge two results trivially.
    """

    roi_path: str
    status: str
    dates_processed: dict[str, int] = field(default_factory=dict)


def ingest_s1_roi_sar(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    client: dask.distributed.Client,
    orbit: S1Orbit,
    batch_days: int = 30,
    edl_credentials_fn: Callable[[], dict[str, str]] | None = None,
    apply_credentials_fn: Callable[[dict[str, str]], None] | None = None,
    use_s3_direct: bool = True,
    cred_refresh_interval_sec: float = DEFAULT_CRED_REFRESH_INTERVAL_SEC,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    storage_options: dict | None = None,
    crop_to_live_windows: bool = False,
) -> SarIngestResult:
    """Ingest OPERA RTC-S1 SAR for an ROI using batched time windows.

    Orchestrator-unaware. Same algorithm as the reference repo's
    ``flows/ingest_s1_roi_sar.py::process_roi_sar``.

    Args:
        roi_zarr_path: Path to the Zarr ROI store (any fsspec-compatible URI).
        start_date: Inclusive start date (``YYYY-MM-DD``).
        end_date: Inclusive end date (``YYYY-MM-DD``).
        store_path: Base path for satellite mosaics; the function
            creates ``sar_<orbit>.zarr`` underneath.
        client: Connected :class:`dask.distributed.Client`. Callers
            create this; we do not call ``get_client``.
        orbit: ``"ascending"`` or ``"descending"`` — one orbit per call.
            Multi-orbit ingestion is a flow-level concern (call twice).
        batch_days: Days per time batch. Smaller values keep each Dask
            graph small at the cost of more STAC queries.
        edl_credentials_fn: Callable returning STS credentials (e.g. the
            dict from ``get_s3_credentials()``). Called when the cached
            credentials have aged past ``cred_refresh_interval_sec``.
            Required when accessing OPERA's S3 direct endpoints; the
            plain runner passes a closure over env vars, the Prefect
            flow passes a closure over a credentials block. ``None``
            means no per-batch refresh — only safe when credentials are
            already injected by the substrate (e.g. a Dask worker
            plugin set up at cluster start).
        apply_credentials_fn: Callable that takes the dict returned by
            ``edl_credentials_fn`` and applies it (typically by setting
            env vars on the orchestrator and registering a Dask
            ``WorkerPlugin``). When ``None``, the credentials returned
            are still fetched on schedule but not applied — useful for
            tests, otherwise pair with ``edl_credentials_fn``.
        use_s3_direct: When ``True``, use ASF's in-region S3 direct
            endpoints (requires us-west-2 reachability and STS creds).
            When ``False``, fall back to CloudFront-signed HTTPS URLs;
            useful for local development.
        cred_refresh_interval_sec: Refresh interval for the credential
            callback.
        log: Optional logger; defaults to ``logging.getLogger(__name__)``.
        storage_options: fsspec storage options for the ROI mask reads.
        crop_to_live_windows: Write only the chunk-aligned windows that
            intersect the ROI mask (``ingest.live_windows``), one commit per
            date within each batch. Default False = legacy full-extent path.

    Returns:
        :class:`SarIngestResult`. ``status="skipped"`` if zero dates
        were written.
    """
    log = log or logging.getLogger(__name__)
    roi = read_roi_metadata(roi_zarr_path)

    ingest_manifest = IngestManifest.from_roi_store(roi_zarr_path)

    # Load chunks: spatial multiples of INGEST_CHUNKS so rechunk at
    # write time is a pure split with no cross-chunk shuffling.
    spatial_chunks = {"northing": INGEST_CHUNKS["northing"], "easting": INGEST_CHUNKS["easting"]}

    last_cred_refresh: float = float("-inf")

    orbit_store = f"{store_path}/sar_{orbit}.zarr"

    # Cropped write path: windows derived once from the same mask this ingest

    # reads (see ingest.live_windows; identical mechanics to the S2 path).

    live_windows: list[tuple[int, int, int, int]] | None = None

    if crop_to_live_windows:
        live_windows = [(w.y0, w.y1, w.x0, w.x1) for w in live_windows_for_mask(roi_zarr_path)]

        log.info("Cropping writes to %d live window(s)", len(live_windows))
    total_processed = 0
    batch_start = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    while batch_start <= end_dt:
        # Inclusive batch window of up to ``batch_days`` calendar days. CMR/STAC
        # also treat their end date as inclusive, and the loop advances to the
        # day after ``batch_end``, so each day is queried by exactly one batch.
        batch_end = min(batch_start + timedelta(days=batch_days - 1), end_dt)
        batch_start_str = batch_start.strftime("%Y-%m-%d")
        batch_end_str = batch_end.strftime("%Y-%m-%d")

        log.info("[%s] Batch %s..%s: querying catalog", orbit, batch_start_str, batch_end_str)

        # Re-read existing dates each batch so prior-batch writes are skipped.
        existing_dates = get_existing_dates(orbit_store)

        # Rebuild the lazy ROI mask each batch so frozen IAM creds inside
        # any embedded boto chain are fresh. The call is cheap (graph
        # construction only); actual S3 reads happen during compute(),
        # by which point the credential refresh below has applied any new
        # session token.
        roi_mask = read_roi_mask(roi_zarr_path, spatial_chunks, storage_options=storage_options)
        if live_windows is None:
            # Ensure the mask is materialised on workers so per-batch graphs
            # reference small future keys.
            roi_mask = client.persist(roi_mask)
        # Cropped: left LAZY on purpose. persist() materialises every chunk of
        # the full zone grid — for 03S that is 3,706 chunks / ~60 GiB of mostly
        # ocean, pinned for the run — while the only consumer, apply_roi_mask,
        # is written out to the live windows and so touches a handful of them.
        # Dask culls the reads to those chunks, making a per-batch re-read far
        # cheaper than the pin. (S2 does the same, and additionally crops the
        # coverage denominator, which it alone computes.)

        # Refresh STS creds + apply (typically: env vars + Dask plugin)
        # once per ``cred_refresh_interval_sec``.
        now = time.monotonic()
        if edl_credentials_fn is not None and now - last_cred_refresh > cred_refresh_interval_sec:
            creds = edl_credentials_fn()
            if apply_credentials_fn is not None:
                apply_credentials_fn(creds)
            last_cred_refresh = now

        data, baselines = ingest_tile(
            provider="cmr-asf",
            collection="opera-rtc-s1",
            tile_id=None,
            start_date=batch_start_str,
            end_date=batch_end_str,
            existing_dates=existing_dates,
            bbox=roi.bbox_wgs84,
            chunks=INGEST_CHUNKS,
            resampling="bilinear",
            groupby="solar_day",
            item_provider_fn=make_s1_item_provider(
                orbit,
                roi.bbox_wgs84,
                batch_start_str,
                batch_end_str,
                use_s3_direct=use_s3_direct,
            ),
            post_load_fn=amplitude_to_db,
            geobox=roi.geobox,
        )

        if data is not None:
            data, _ = apply_roi_mask(data, roi_zarr_path, spatial_chunks, roi_mask=roi_mask)

            def _retrying() -> Retrying:
                return Retrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=8),
                    before_sleep=before_sleep_log(log, logging.WARNING),
                    reraise=True,
                )

            if live_windows is not None:
                # A batch holds many NON-contiguous dates, each its own atomic
                # commit — so the retry scope is PER DATE. One retry around the
                # whole loop would restart at a date an earlier attempt already
                # committed and trip the duplicate-date guard.
                for i in range(data.sizes["time"]):
                    for attempt in _retrying():
                        with attempt:
                            write_day_windows(
                                orbit_store,
                                data.isel(time=slice(i, i + 1)),
                                live_windows,
                                roi=roi,
                                manifest=ingest_manifest,
                                baselines=baselines,
                                tile_id=roi_zarr_path,
                                crs=roi.native_crs,
                                chunks=INGEST_CHUNKS,
                            )
            else:
                for attempt in _retrying():
                    with attempt:
                        write_dataset(
                            orbit_store,
                            data,
                            tile_id=roi_zarr_path,
                            baselines=baselines,
                            chunks=INGEST_CHUNKS,
                            manifest=ingest_manifest,
                            crs=roi.native_crs,
                        )
            n = data.sizes["time"]
            total_processed += n
            log.info(
                "[%s] Batch %s..%s: wrote %d dates (total: %d)",
                orbit,
                batch_start_str,
                batch_end_str,
                n,
                total_processed,
            )

        batch_start = batch_end + timedelta(days=1)

    if total_processed == 0:
        return SarIngestResult(
            roi_path=roi_zarr_path,
            status="skipped",
            dates_processed={orbit: 0},
        )

    return SarIngestResult(
        roi_path=roi_zarr_path,
        status="success",
        dates_processed={orbit: total_processed},
    )
