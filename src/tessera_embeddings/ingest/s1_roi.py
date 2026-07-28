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

   * Renew the OPERA read credentials whenever they are close to expiring —
     checked at the start of a batch AND before every date's write, because
     the credential's roughly one-hour life is unrelated to batch boundaries:
     a longer batch outlives it and every read afterwards fails.
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
from datetime import UTC, datetime, timedelta
from typing import Literal, final

import dask.distributed
from tenacity import Retrying, before_sleep_log, stop_after_attempt, wait_exponential

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE, INGEST_CHUNKS
from tessera_embeddings.ingest._pipeline import pipelined
from tessera_embeddings.ingest.live_windows import (
    WINDOW_COST_IN_CHUNKS,
    WINDOW_COST_IN_CHUNKS_OVERLAPPED,
    live_windows_for_mask,
)
from tessera_embeddings.ingest.opera_query import make_s1_item_provider
from tessera_embeddings.ingest.roi import read_roi_mask, read_roi_metadata
from tessera_embeddings.ingest.roi_processing import apply_roi_mask
from tessera_embeddings.ingest.stac import ingest_tile
from tessera_embeddings.ingest.transforms import amplitude_to_db
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import get_existing_dates, write_dataset, write_day_windows

logger = logging.getLogger(__name__)

DEFAULT_CRED_REFRESH_INTERVAL_SEC = 30 * 60
"""Fallback refresh cadence, used only when the credential advertises no expiry."""

CRED_EXPIRY_MARGIN_SEC = 15 * 60
"""Renew this far ahead of expiry.

Must exceed the longest single date write, because a credential that is valid when a
write starts must still be valid when it ends — the reads happen throughout. Renewing is
two cheap HTTP calls, so a generous margin costs almost nothing and the failure it
prevents costs the remainder of the run.
"""

S1Orbit = Literal["ascending", "descending"]


def _parse_credential_expiry(creds: dict[str, str]) -> float | None:
    """Epoch seconds at which ``creds`` expires, or ``None`` if not stated or unparseable.

    Returning ``None`` rather than raising is deliberate: an unreadable expiry must
    degrade to the age-based cadence, not sink an ingest that would otherwise run.
    """
    raw = creds.get("expiration")
    if not raw:
        return None
    try:
        # ASF returns ISO-8601; tolerate both the "Z" and "+00:00" spellings, and a
        # space instead of "T".
        text = str(raw).strip().replace("Z", "+00:00").replace(" ", "T", 1)
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except ValueError:
        logger.warning("Could not parse credential expiry %r; falling back to age-based refresh", raw)
        return None


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


@final
@dataclass(frozen=True)
class _PreparedBatch:
    """One batch carried from the catalogue query to the write loop.

    The split is the same one the phase logging measures: everything up to a write-ready
    dataset on one side (catalogue query plus lazy graph build, no cluster reads), the
    per-date writes on the other. That boundary is what makes the query hideable — it
    touches no store and holds no credential, so it can run on a background thread while
    the previous batch writes.

    ``data`` is ``None`` when the batch has no new dates, which is a normal outcome and
    not an error: a sparse region or an already-ingested range yields nothing to write.
    """

    start: str
    end: str
    data: object | None
    baselines: dict[str, int]
    query_s: float


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
    overlap_window_writes: bool = True,
    pipeline_batches: bool = True,
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
        overlap_window_writes: Defaults ON. Submit a date's windows as ONE dask compute
            rather than one blocking compute per window, so their critical paths
            overlap across the fleet instead of summing. Produces an identical store
            either way. Also selects the window merge exchange rate, since that prices
            a window boundary by how it is written — the two must not drift apart.
        crop_to_live_windows: Write only the chunk-aligned windows that
            intersect the ROI mask (``ingest.live_windows``), one commit per
            date within each batch. Default False = legacy full-extent path.
        pipeline_batches: Defaults ON. Prepare the NEXT batch's catalogue query while
            the current batch writes, so only the first batch pays its query on the
            critical path. Shares ``ingest._pipeline.pipelined`` with the S2 date loop.
            Look-ahead is fixed at one batch: a batch's write is one long consume, so
            depth 1 already covers it, and deeper retention is what once deadlocked the
            S2 driver. Set False to restore the strictly serial query-then-write loop.

    Returns:
        :class:`SarIngestResult`. ``status="skipped"`` if zero dates
        were written.
    """
    log = log or logging.getLogger(__name__)
    roi = read_roi_metadata(roi_zarr_path)

    ingest_manifest = IngestManifest.from_roi_store(roi_zarr_path)

    # Load blocks match the store's chunks: one read task per (chunk, band), and the
    # write needs no rechunk. See config.ingest for why a coarser load block was removed.
    spatial_chunks = {"northing": INGEST_CHUNKS["northing"], "easting": INGEST_CHUNKS["easting"]}

    last_cred_refresh: float = float("-inf")
    cred_expires_at: float | None = None

    def refresh_credentials_if_stale() -> None:
        """Re-fetch the OPERA read credentials when they are close to expiring.

        Driven by the credential's OWN expiry rather than a fixed cadence, and called
        before every date's write rather than only between batches. Both changes fix the
        same defect: the credential ASF mints lives about an hour, so any unit of work
        longer than that outlives it, and every subsequent read fails with an expired
        token. Refreshing only between batches meant a single long batch could never
        renew — the credential's clock does not care about batch boundaries.

        Calling it per date bounds staleness to one date's write, which is the smallest
        unit this loop can renew between. The margin must exceed that duration, since a
        credential valid at the start of a write must still be valid at its end.
        """
        nonlocal last_cred_refresh, cred_expires_at
        if edl_credentials_fn is None:
            return
        now_wall, now_mono = time.time(), time.monotonic()
        if cred_expires_at is not None:
            fresh_enough = now_wall < cred_expires_at - CRED_EXPIRY_MARGIN_SEC
        else:
            # No expiry advertised: fall back to the age-based cadence.
            fresh_enough = now_mono - last_cred_refresh <= cred_refresh_interval_sec
        if fresh_enough:
            return
        creds = edl_credentials_fn()
        if apply_credentials_fn is not None:
            apply_credentials_fn(creds)
        last_cred_refresh = now_mono
        cred_expires_at = _parse_credential_expiry(creds)

    orbit_store = f"{store_path}/sar_{orbit}.zarr"

    # Cropped write path: windows derived once from the same mask this ingest
    # reads (see ingest.live_windows; identical mechanics to the S2 path).
    live_windows: list[tuple[int, int, int, int]] | None = None
    if crop_to_live_windows:
        live_windows = [
            (w.y0, w.y1, w.x0, w.x1)
            for w in live_windows_for_mask(
                roi_zarr_path,
                window_px=INGEST_CHUNK_SIZE,
                # The merge exchange rate follows how this run WRITES, exactly as on the
                # S2 path: overlapped windows share one graph, so a boundary is cheap and
                # the DP should stop trading ocean area for fewer windows. A sequential
                # writer still pays the serial cost per boundary and keeps the high rate.
                window_cost_in_chunks=(
                    WINDOW_COST_IN_CHUNKS_OVERLAPPED if overlap_window_writes else WINDOW_COST_IN_CHUNKS
                ),
                storage_options=storage_options,
            )
        ]
        log.info("Cropping writes to %d live window(s)", len(live_windows))

    total_processed = 0
    batch_start = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Inclusive batch windows of up to ``batch_days`` calendar days. CMR/STAC also treat
    # their end date as inclusive, and each range starts the day after the previous one
    # ends, so every day is queried by exactly one batch.
    #
    # Materialised as a list rather than advanced in the loop because the look-ahead has
    # to know what comes next before the current batch is done with.
    batch_ranges: list[tuple[str, str]] = []
    while batch_start <= end_dt:
        batch_end = min(batch_start + timedelta(days=batch_days - 1), end_dt)
        batch_ranges.append((batch_start.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d")))
        batch_start = batch_end + timedelta(days=1)

    # Read ONCE, then maintained in-process as dates are written.
    #
    # The serial loop re-read this from the store every batch, so that a date written by
    # an earlier batch would be skipped by a later one. Under a look-ahead that read is
    # unsafe: the next batch's query is prepared BEFORE the current batch has written, so
    # a store read there would miss dates that are about to exist. Tracking writes
    # in-process is correct regardless of when the query runs, and drops a per-batch S3
    # read as a side effect.
    #
    # Why any of this is needed at all: batches are cut on UTC dates while the loader
    # groups by SOLAR day, so an acquisition late on a batch's last UTC day can belong to
    # the next batch's solar day, and two consecutive batches can then contain the same
    # day. Probing real catalogue responses did not find such a boundary (12 boundaries
    # across a +1 h and a +10 h zone), but the offset is a translation rather than a
    # split, so nothing rules it out — and the cost of being wrong is a duplicate-date
    # commit. The consume side below is therefore the authority on what has been written,
    # and the query filter is only an optimisation.
    written_dates: set[str] = get_existing_dates(orbit_store)
    # Frozen at the start so the background thread reads an object nothing mutates.
    already_present = frozenset(written_dates)

    def _prepare_batch(rng: tuple[str, str]) -> _PreparedBatch:
        """Catalogue query plus lazy graph build for one batch. No cluster reads.

        Runs on the pipeline's background thread when ``pipeline_batches`` is on, so it
        must touch nothing the write loop owns: no credential application, no store
        writes, and only the immutable snapshot of ``written_dates`` taken when the
        pipeline started.
        """
        batch_start_str, batch_end_str = rng
        log.info("[%s] Batch %s..%s: querying catalog", orbit, batch_start_str, batch_end_str)

        # Rebuild the lazy ROI mask per batch so frozen IAM creds inside any embedded
        # boto chain are fresh. Graph construction only; the actual S3 reads happen
        # during the write's compute(), by which point the credential refresh has
        # applied any new session token.
        batch_mask = read_roi_mask(roi_zarr_path, spatial_chunks, storage_options=storage_options)
        if live_windows is None:
            # Materialise on workers so per-batch graphs reference small future keys.
            batch_mask = client.persist(batch_mask)
        # Cropped: left LAZY on purpose. persist() materialises every chunk of the full
        # zone grid — for 03S that is 3,706 chunks / ~60 GiB of mostly ocean, pinned for
        # the run — while the only consumer, apply_roi_mask, is written out to the live
        # windows and so touches a handful of them. Dask culls the reads to those chunks,
        # making a per-batch re-read far cheaper than the pin. (S2 does the same, and
        # additionally crops the coverage denominator, which it alone computes.)

        query_started = time.monotonic()
        data, baselines = ingest_tile(
            provider="cmr-asf",
            collection="opera-rtc-s1",
            tile_id=None,
            start_date=batch_start_str,
            end_date=batch_end_str,
            # The snapshot, not the live set: a background thread must not read a set the
            # write loop is mutating. Anything it misses is caught when consuming.
            existing_dates=already_present,
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
        query_s = time.monotonic() - query_started
        if data is not None:
            data = apply_roi_mask(data, roi_zarr_path, spatial_chunks, roi_mask=batch_mask)
        return _PreparedBatch(batch_start_str, batch_end_str, data, baselines, query_s)

    # depth=1: one batch prepared ahead. A batch's write is one long consume, so a single
    # look-ahead already covers it; more would retain catalogue items to hide nothing.
    # Unpipelined, `pipelined` is bypassed entirely rather than run at depth 0 — the
    # serial path must stay available as a rollback that shares no machinery.
    if pipeline_batches:
        prepared_batches = pipelined(batch_ranges, _prepare_batch, depth=1)
    else:
        prepared_batches = ((_prepare_batch(rng), 0.0) for rng in batch_ranges)

    for prepared, stall_s in prepared_batches:
        batch_start_str, batch_end_str = prepared.start, prepared.end
        data, baselines, query_s = prepared.data, prepared.baselines, prepared.query_s

        refresh_credentials_if_stale()

        if data is not None:
            # The ROI mask was applied in _prepare_batch, on the same side of the phase
            # boundary as the graph build it belongs to.
            write_total_s = 0.0
            written_this_batch = 0

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
                    # datetime64 stringifies as "YYYY-MM-DDThh:mm:ss…"; the date half is
                    # all this needs, and it matches get_existing_dates' own slicing.
                    date_str = str(data["time"].values[i])[:10]
                    # THE authority on what has been written, checked here rather than
                    # relying on the query filter: under a look-ahead the query for this
                    # batch may have been built before an earlier batch committed, so a
                    # solar day shared across a UTC batch boundary could arrive twice.
                    # Writing it twice would trip the duplicate-date guard mid-run.
                    if date_str in written_dates:
                        log.info("[%s] Skipping date %s: already written", orbit, date_str)
                        continue
                    # Inside the per-date loop deliberately: a batch can outlive the
                    # credential, so renewing only at batch boundaries is what failed.
                    refresh_credentials_if_stale()
                    date_started = time.monotonic()
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
                                parallel_windows=overlap_window_writes,
                            )
                    date_s = time.monotonic() - date_started
                    write_total_s += date_s
                    written_dates.add(date_str)
                    written_this_batch += 1
                    # ``mode`` is the load-bearing field: sequential means this date
                    # cost the SUM of its windows' critical paths rather than their
                    # maximum, which is the single largest difference between how S1
                    # and S2 write today.
                    log.info(
                        "[%s] S1 stage timings date=%s: write=%.1fs windows=%d mode=%s",
                        orbit,
                        date_str,
                        date_s,
                        len(live_windows),
                        "parallel" if overlap_window_writes else "sequential",
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
            # Count what was WRITTEN, not what the batch held: under a look-ahead a date
            # can arrive already committed by an earlier batch and be skipped above, and
            # reporting it as processed would overstate a resumed run's progress.
            n = written_this_batch if live_windows is not None else data.sizes["time"]
            total_processed += n
            log.info(
                "[%s] Batch %s..%s: wrote %d dates (total: %d)",
                orbit,
                batch_start_str,
                batch_end_str,
                n,
                total_processed,
            )
            # `stall` is what the look-ahead failed to hide — how long this batch waited
            # for its own query after the previous batch's writes finished. Near zero
            # means the query hid completely; approaching `query` means it hid nothing,
            # which is the expected reading for the FIRST batch since nothing precedes
            # it. `hidden` is therefore the saving, and it is what to watch: if it stays
            # near zero on later batches the look-ahead is not paying.
            log.info(
                "[%s] S1 batch timings %s..%s n=%d: query=%.1fs hidden=%.1fs stall=%.1fs write=%.1fs per_date=%.1fs",
                orbit,
                batch_start_str,
                batch_end_str,
                n,
                query_s,
                max(query_s - stall_s, 0.0),
                stall_s,
                write_total_s,
                (stall_s + write_total_s) / n if n else 0.0,
            )

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
