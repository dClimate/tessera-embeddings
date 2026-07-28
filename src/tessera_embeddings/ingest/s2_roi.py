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

Step 4 is split into a prepare half and a write half so ``pipeline_dates`` can
overlap one date's preparation with the previous date's write, and so
``batch_dates`` can compute several dates' writes as one graph (one commit per
BATCH — see ``storage.zarr_store.write_days_windows`` for why that unit is
forced). Writes stay in date order under every mode.

This module imports nothing from Prefect or any cloud provider.
"""

from __future__ import annotations

import logging
import operator
import time
from dataclasses import dataclass
from datetime import timedelta
from functools import partial, reduce
from typing import final

import dask.array as da
import dask.distributed
import numpy as np
import xarray as xr
from tenacity import Retrying, before_sleep_log, stop_after_attempt, wait_exponential

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE, INGEST_CHUNKS, auto_batch_dates
from tessera_embeddings.config.satellites import S2_SCL_INVALID_CLASSES
from tessera_embeddings.ingest._pipeline import pipelined
from tessera_embeddings.ingest.live_windows import (
    WINDOW_COST_IN_CHUNKS,
    WINDOW_COST_IN_CHUNKS_OVERLAPPED,
    live_windows_for_mask,
    windows_for_date,
)
from tessera_embeddings.ingest.roi import read_roi_mask, read_roi_metadata
from tessera_embeddings.ingest.roi_processing import DEFAULT_MIN_VALID_COVERAGE
from tessera_embeddings.ingest.stac import (
    group_items_by_date,
    load_stac_items,
    query_stac_items,
    solar_day_offset_seconds,
    stream_stac_months,
)
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import (
    get_existing_dates,
    write_dataset,
    write_day_windows,
    write_days_windows,
)

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


def _sum_over_windows(mask: da.Array, windows: list[tuple[int, int, int, int]]) -> da.Array:
    """Total a full-extent ROI mask over its live windows only.

    Equal to the full-extent sum because the mask is False outside every window,
    and free of double counting because row-band windows are chunk-disjoint. An
    empty window list yields 0 (an all-ocean ROI has no live pixels).
    """
    parts = [mask[y0:y1, x0:x1].sum() for y0, y1, x0, x1 in windows]
    return reduce(operator.add, parts) if parts else da.zeros((), dtype="int64")


def _coverage_from_scl(
    scl_2d: xr.DataArray,
    roi_mask: da.Array,
    roi_pixel_count: int,
    min_valid_coverage: float,
    client: dask.distributed.Client,
    windows: list[tuple[int, int, int, int]] | None = None,
) -> tuple[bool, xr.DataArray | None]:
    """Decide whether a date passes coverage, from an ALREADY-LOADED SCL slice.

    Takes the SCL array rather than loading it so the gate and the write share one
    ``odc.stac.load`` graph: SCL is one of the written bands, so loading it twice
    per date cost a second client-side graph build and a second read of the same
    data for no information.

    Both sides of the coverage ratio must be cropped together or every percentage
    is skewed: under ``windows`` the numerator reduces over the windows only, which
    equals the full-extent count because the mask is False outside them and row
    bands are chunk-disjoint. ``any_valid`` stays LAZY there — materialising it
    would defeat the cropping, and the write pulls only window slices of it.

    Returns:
        ``(passes, any_valid)``; ``any_valid`` is ``None`` when the date fails.
    """
    invalid_classes = np.array(sorted(S2_SCL_INVALID_CLASSES), dtype=scl_2d.dtype)
    any_valid = ~scl_2d.isin(invalid_classes)

    if windows is not None:
        masked = any_valid & roi_mask
        parts = [masked.isel(northing=slice(y0, y1), easting=slice(x0, x1)).sum() for y0, y1, x0, x1 in windows]
        # Submitted through the client explicitly rather than by a bare .compute():
        # the gate runs off the driver thread under date pipelining, and the
        # scheduler this reduce goes to must be the caller's, not whichever one
        # dask's default resolution finds from the thread it happens to be on.
        valid_count_val = int(client.compute(reduce(operator.add, parts)).result()) if parts else 0
    else:
        valid_count = (any_valid & roi_mask).sum()
        any_valid, valid_count = client.persist([any_valid, valid_count])
        valid_count_val = valid_count.compute().item()

    passes = float(100.0 * valid_count_val / roi_pixel_count) >= min_valid_coverage
    return (True, any_valid) if passes else (False, None)


@dataclass
class _PreparedDate:
    """One date's write-ready state, or the reason it has none.

    ``day_ds is None`` means the date was skipped before the write (asset-incomplete
    items, coverage-gate failure, or no reachable live window); ``skip_reason`` says
    which, for the consume-side counters. Everything a retry of the write needs is
    here, so preparation never re-runs on write failure.
    """

    date: str
    day_ds: xr.Dataset | None
    windows: list[tuple[int, int, int, int]]
    build_s: float
    gate_s: float
    skip_reason: str | None = None


def _solar_grouping_longitude(roi: object) -> float | None:
    """The longitude to group solar days by, matching the loader's own choice.

    The loader shifts every item by ONE longitude — its geobox extent's centroid in
    WGS84 — so preferring the geobox here makes the two agree by construction rather
    than by approximation. Both must land on the same whole-hour offset, and a bbox
    midpoint can differ from an extent centroid by enough to cross a 15-degree
    boundary and so disagree.

    Falls back to the WGS84 bbox midpoint, then to ``None``, which restores UTC-date
    grouping. Degrading rather than raising is deliberate: a missing geobox is a
    caller that is not loading by solar day, and no longitude is recoverable from
    nothing.
    """
    geobox = getattr(roi, "geobox", None)
    if geobox is not None:
        try:
            ((lon, _),) = geobox.extent.centroid.to_crs("epsg:4326").points
            return float(lon)
        except (AttributeError, TypeError, ValueError):
            pass
    bbox = getattr(roi, "bbox_wgs84", None)
    if bbox is not None and len(bbox) >= 3:
        return (float(bbox[0]) + float(bbox[2])) / 2.0
    return None


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
    stream_stac_monthly: bool = True,
    overlap_window_writes: bool = True,
    pipeline_dates: bool = False,
    batch_dates: int | None = None,
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
        stream_stac_monthly: Query the STAC catalog one calendar month at a
            time, prefetching the next month while the current one is
            processed, instead of querying the whole window up front. Bounds
            retained items to two months: a whole year's items do not fit in
            the worker this runs on. ``False`` restores the single up-front
            query and is the rollback path only — a year-long window cannot
            complete under it.
        overlap_window_writes: Submit every window of a date as one dask
            compute instead of one blocking compute per window, so the
            windows' critical paths overlap across the fleet rather than
            summing. Identical stores either way; falls back to the
            sequential write when the overlapped machinery is unavailable.
        pipeline_dates: Prepare the next date — its load graph, its coverage
            gate, its footprint narrowing and masking — on a background thread
            while the current date is being written, so that preparation costs
            wall clock only when the write cannot cover it. The WRITE stays
            serial and in date order: one commit per date, and the store has
            exactly one writer either way. Identical stores either way, which
            rests on preparation being side-effect-free — it must touch nothing
            but the dataset it returns.
        batch_dates: Write up to this many consecutive PASSING dates as one dask
            compute and one commit, so the dates' graphs pack the fleet together —
            one date's straggling reads backfill with another's work — and the
            per-date drain tail and commit gap are paid once per batch. The commit
            unit becomes the batch: a mid-batch failure commits none of its dates
            and the retry re-ingests exactly those (per-date sessions are
            impossible — each date's append resizes the time axis, so sibling
            sessions conflict on array metadata). Identical stores either way.
            Requires ``crop_to_live_windows``. COMPOSES with ``pipeline_dates``:
            the look-ahead is then sized to the batch, so the next batch's whole
            preparation overlaps this batch's write instead of only one date's.
            ``None`` (the default) derives it from the ROI's covered window area via
            :func:`~tessera_embeddings.config.ingest.auto_batch_dates`, because the
            benefit is not monotonic in ROI size; 1 forces the one-commit-per-date
            path.

    Returns:
        :class:`IngestResult`. ``status="skipped"`` if zero STAC items
        were returned or zero dates passed the coverage filter.
    """
    log = log or logging.getLogger(__name__)
    if batch_dates is not None and batch_dates < 1:
        raise ValueError(f"batch_dates must be >= 1 or None for auto, got {batch_dates}")
    if batch_dates is not None and batch_dates > 1 and not crop_to_live_windows:
        # The batched write is the windowed write lifted across dates; the legacy
        # full-extent path has no windowed form to lift.
        raise ValueError("batch_dates > 1 requires crop_to_live_windows")
    reflectance_store = f"{store_path}/reflectance.zarr"
    roi = read_roi_metadata(roi_zarr_path)

    ingest_manifest = IngestManifest.from_roi_store(roi_zarr_path)

    mid_longitude = _solar_grouping_longitude(roi)

    # Live windows for the cropped write path, derived once per run from the same
    # mask this ingest already reads (plain tuples: storage takes no ingest types).
    #
    # These describe where the ROI has LAND, so they are the same on every date. Each
    # date is then narrowed further to the land its own imagery can reach
    # (``windows_for_date``, applied per date below), because a satellite covers only
    # a fraction of a wide ROI per pass.
    live_windows: list[tuple[int, int, int, int]] | None = None
    run_windows: list = []
    if crop_to_live_windows:
        run_windows = live_windows_for_mask(
            roi_zarr_path,
            window_px=INGEST_CHUNK_SIZE,
            # The merge exchange rate follows how this run WRITES: overlapped windows
            # share one graph, so a boundary is cheap and the DP should stop trading
            # ocean area for fewer windows. Sequential writes still pay the serial cost.
            window_cost_in_chunks=(
                WINDOW_COST_IN_CHUNKS_OVERLAPPED if overlap_window_writes else WINDOW_COST_IN_CHUNKS
            ),
            storage_options=storage_options,
        )
        live_windows = [(w.y0, w.y1, w.x0, w.x1) for w in run_windows]
        log.info("Cropping writes to %d live window(s)", len(run_windows))

    # Resolve `batch_dates=None` (auto) now that the windows are known. Derived from
    # the area those windows COVER, which is what the write graph touches. Falls back
    # to 1 on the uncropped path, where a batched write has no windowed form to lift.
    if batch_dates is None:
        covered_chunks = sum(
            ((w.y1 - w.y0) // INGEST_CHUNK_SIZE) * ((w.x1 - w.x0) // INGEST_CHUNK_SIZE) for w in run_windows
        )
        batch_dates = auto_batch_dates(covered_chunks) if crop_to_live_windows else 1
        log.info(
            "batch_dates=auto -> %d (%d chunk(s) covered by live windows)",
            batch_dates,
            covered_chunks,
        )

    # Load blocks match the store's chunks, so a date's read parallelism is one task
    # per (chunk, band) and the write needs no rechunk at all.
    spatial_chunks = {"northing": INGEST_CHUNKS["northing"], "easting": INGEST_CHUNKS["easting"]}

    roi_mask = read_roi_mask(roi_zarr_path, spatial_chunks, storage_options=storage_options)
    if live_windows is not None:
        # Cropped: mask stays LAZY, total comes from the live windows only.
        #
        # No persist — it would materialise the whole zone grid, while every
        # consumer (the SCL reduce below, the masking pass) slices to windows, so
        # dask culls the reads and a per-date re-read beats pinning the grid.
        #
        # The window total equals the full-extent total because the mask is False
        # outside every window, and row bands are chunk-disjoint so nothing is
        # counted twice. _coverage_from_scl relies on that same property for the
        # numerator: both sides of the coverage ratio must stay cropped together,
        # or every percentage is silently skewed.
        roi_pixel_count = int(_sum_over_windows(roi_mask, live_windows).compute())
    else:
        # Persist the ROI mask on workers so per-day graphs reference small
        # future keys instead of re-reading from Zarr each time.
        roi_mask = client.persist(roi_mask)
        roi_pixel_count = int(roi_mask.sum().compute())

    total_processed = 0
    total_filtered = 0
    total_seen = 0

    def _prepare_date(day_items: list, baselines: dict[str, int]) -> _PreparedDate:
        """Build one solar day's write-ready dataset, or the reason it has none.

        A closure so the streamed and single-query paths run byte-identical work; the
        per-date logic must not fork on how its items were supplied.

        SIDE-EFFECT-FREE by contract: under ``pipeline_dates`` this runs on a
        background thread while the previous date is being written, so anything it
        mutated outside its return value would race the writer. Everything the write
        needs travels back in the :class:`_PreparedDate`.
        """
        # The group's own date, which is all a skipped date needs; refined below to the
        # mosaic's solar day, which is only knowable once the load has resolved it.
        date = day_items[0].datetime.strftime("%Y-%m-%d")

        # ONE load per date, serving both the coverage gate and the write. SCL is
        # among the written bands, so a separate gate-only load re-read it and paid
        # a second client-side graph build. A date that then fails the gate has
        # built a graph it discards — cheap, because construction is the same order
        # either way and nothing was computed.
        stage_started = time.monotonic()
        try:
            day_ds = load_stac_items(
                day_items,
                provider=provider,
                collection=collection,
                baselines=baselines,
                bbox=roi.bbox_wgs84,
                chunks=INGEST_CHUNKS,
                extra_bands=["scl"],
                resampling="bilinear",
                groupby="solar_day",
                geobox=roi.geobox,
            )
        except ValueError as exc:
            # Earth-search occasionally publishes asset-incomplete items (missing
            # SCL and/or reflectance bands). odc.stac.load resolves bands eagerly,
            # so one such item raises before any graph is built. Drop the date.
            if "No such band/alias" not in str(exc):
                raise
            log.warning("Dropping date: load failed on asset-incomplete STAC item(s): %s", exc)
            return _PreparedDate(date, None, [], time.monotonic() - stage_started, 0.0, "asset-incomplete")

        built_at = time.monotonic()
        passes, any_valid = _coverage_from_scl(
            day_ds["scl"].isel(time=0),
            roi_mask,
            roi_pixel_count,
            min_valid_coverage,
            client,
            windows=live_windows,
        )
        build_s, gate_s = built_at - stage_started, time.monotonic() - built_at
        if not passes:
            return _PreparedDate(date, None, [], build_s, gate_s, "coverage")

        date = str(day_ds.time.dt.date.values[0])
        day_ds["time"] = [np.datetime64(date, "ns")]

        # Narrow this date's writes to the land its own imagery reaches. The run's
        # windows cover the whole ROI's land on every date, but one pass images a
        # fraction of a wide ROI, so most of them hold nothing today: those tasks run,
        # find no data, and write nothing, because an all-fill chunk is never stored.
        # Dropping them cannot change the mosaic — only what is computed to produce it.
        #
        # The COVERAGE GATE above deliberately keeps the run's full window set. Its
        # ratio is "how much of the ROI's land did this date see", so its denominator
        # must stay the whole ROI; cropping the gate would rescale every percentage.
        # The numerator is unaffected either way, since there are no valid pixels
        # outside the footprint to count.
        date_windows: list[tuple[int, int, int, int]] = live_windows or []
        if live_windows is not None and run_windows:
            narrowed = windows_for_date(
                run_windows,
                [getattr(item, "bbox", None) for item in day_items],  # type: ignore[misc]
                roi.geobox,
                chunk_px=INGEST_CHUNK_SIZE,
            )
            if not narrowed:
                # No live cell is reachable today. Nothing to write, and writing an
                # empty window set would commit a date holding nothing.
                log.info("Skipping date: its imagery reaches no live window")
                return _PreparedDate(date, None, [], build_s, gate_s, "no-live-window")
            date_windows = [(w.y0, w.y1, w.x0, w.x1) for w in narrowed]
            if len(narrowed) != len(run_windows):
                log.info(
                    "Date footprint: writing %d of %d live window(s)",
                    len(narrowed),
                    len(run_windows),
                )

        # ONE masking pass, not two. Zeroing invalid pixels and zeroing outside the
        # ROI both fill with 0, so `x.where(A, 0).where(B, 0)` is `x.where(A & B, 0)`
        # — and each `where` is a graph task per (chunk, band), which is the budget
        # that limits ingest. SCL keeps only the ROI mask: it is categorical, and
        # zeroing it by its own validity would rewrite the class codes it carries.
        roi_2d = roi_mask if roi_mask is not None else read_roi_mask(roi_zarr_path, spatial_chunks)
        keep = roi_2d if any_valid is None else (any_valid & roi_2d)
        for var in day_ds.data_vars:
            mask_for_var = roi_2d if str(var) == "scl" else keep
            day_ds[str(var)] = day_ds[str(var)].where(mask_for_var, other=0)

        return _PreparedDate(date, day_ds, date_windows, build_s, gate_s)

    def _write_date(prepared: _PreparedDate, stall_s: float, baselines: dict[str, int]) -> None:
        """Write one prepared date's pixels: the store's only writer.

        Runs on the driver thread, one date at a time, under both modes — icechunk
        commits are sequential on one branch and one commit per date is the contract,
        so this half is what stays serial when preparation is overlapped.
        """
        day_ds = prepared.day_ds
        assert day_ds is not None, f"date {prepared.date} was skipped ({prepared.skip_reason}); nothing to write"
        write_started = time.monotonic()

        # Tenacity retry on intermittent GDAL errors under high parallelism.
        # Icechunk writes are atomic, so retry is safe — a failed cropped date
        # commits nothing (batched_region_writes) and the retry starts clean.
        # A retry re-runs only the write, from the graph preparation already built.
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
                        prepared.windows,
                        roi=roi,
                        manifest=ingest_manifest,
                        baselines=baselines,
                        tile_id=roi_zarr_path,
                        crs=roi.native_crs,
                        chunks=INGEST_CHUNKS,
                        parallel_windows=overlap_window_writes,
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
        # One line per kept date, partitioning its wall clock into the client-side
        # graph build, the coverage-gate compute, and the write (windows + commit).
        # Stable format: CloudWatch queries and the pipeline analysis key off it.
        # `total` remains build+gate+write — the date's SERIAL-EQUIVALENT cost, so the
        # figure stays comparable across both modes however much of it was hidden.
        write_s = time.monotonic() - write_started
        prepare_s = prepared.build_s + prepared.gate_s
        log.info(
            "Stage timings date=%s: build=%.1fs gate=%.1fs write=%.1fs total=%.1fs windows=%d mode=%s",
            prepared.date,
            prepared.build_s,
            prepared.gate_s,
            write_s,
            prepare_s + write_s,
            len(prepared.windows) if live_windows is not None else 0,
            "parallel" if overlap_window_writes else "sequential",
        )
        # What the overlap actually bought: `stall` is the preparation the write could
        # not cover, `hidden` the part it did. Emitted in BOTH modes — serially every
        # date stalls for its whole preparation and hides none of it — so the two are
        # comparable from one grep.
        log.info(
            "Pipeline date=%s: prepare=%.1fs hidden=%.1fs stall=%.1fs",
            prepared.date,
            prepare_s,
            max(prepare_s - stall_s, 0.0),
            stall_s,
        )

    def _write_batch(batch: list[_PreparedDate], baselines: dict[str, int], stall_s: float = 0.0) -> None:
        """Write a batch of prepared dates: one dask compute, ONE commit.

        The same retry contract as ``_write_date``, at batch granularity: the
        batched write commits nothing on failure, so a retry re-runs the whole
        batch cleanly from the graphs already prepared.
        """
        write_started = time.monotonic()
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                days: list[tuple[xr.Dataset, list[tuple[int, int, int, int]]]] = []
                for p in batch:
                    assert p.day_ds is not None, f"date {p.date} was skipped ({p.skip_reason}); not batchable"
                    days.append((p.day_ds, p.windows))
                write_days_windows(
                    reflectance_store,
                    days,
                    roi=roi,
                    manifest=ingest_manifest,
                    baselines=baselines,
                    tile_id=roi_zarr_path,
                    crs=roi.native_crs,
                    chunks=INGEST_CHUNKS,
                    parallel_windows=overlap_window_writes,
                )
        # The batch's write is ONE compute, so a per-date write time does not exist
        # as a measurement — this line is the batched counterpart of `Stage timings`
        # and analysis divides by n. build/gate are sums of the real per-date values.
        write_s = time.monotonic() - write_started
        prepare_s = sum(p.build_s + p.gate_s for p in batch)
        log.info(
            "Batch timings dates=%s..%s n=%d: build=%.1fs gate=%.1fs write=%.1fs windows=%d "
            "prepare=%.1fs hidden=%.1fs stall=%.1fs",
            batch[0].date,
            batch[-1].date,
            len(batch),
            sum(p.build_s for p in batch),
            sum(p.gate_s for p in batch),
            write_s,
            sum(len(p.windows) for p in batch),
            prepare_s,
            # What the look-ahead actually bought: `stall` is the preparation the previous
            # batch's write could not cover. Unpipelined every batch stalls for all of its
            # preparation and hides none, so both modes read from one grep — and, as with
            # the per-date line, `hidden` is bounded by the SERIAL prepare cost, never by
            # a pipelined `prepare` figure inflated by contention.
            max(prepare_s - stall_s, 0.0),
            stall_s,
        )

    def _drive(items: list, baselines: dict[str, int]) -> None:
        """Sort one supply of items cloudiest-first, group by date, and ingest each."""
        nonlocal total_seen

        def _consume(prepared: _PreparedDate, stall_s: float) -> None:
            """Count or write one prepared date — the ONE consume path both modes take.

            Serial and pipelined differ only in where ``prepared`` came from, so the
            counters and the write cannot drift between them.
            """
            nonlocal total_processed, total_filtered
            if prepared.day_ds is None:
                total_filtered += 1
                return
            _write_date(prepared, stall_s, baselines)
            total_processed += 1

        # query_stac_items sorts by (date, cloud_cover ASC). Re-sort cloudiest-first so
        # the clearest tile paints last (wins) in solar_day's painter's algorithm.
        # group_items_by_date preserves this within-group order.
        #
        # Both the sort and the grouping key off the SOLAR day, not the UTC date, because
        # that is what the loader groups by. Using UTC here let a group the loader saw as
        # two solar days arrive as two time slices, against a cloud mask reduced to one —
        # a dimension conflict, and one that fires only where the solar offset is large
        # enough to cross UTC midnight (the far-eastern and far-western zones).
        day_offset = timedelta(seconds=solar_day_offset_seconds(mid_longitude) if mid_longitude is not None else 0)
        items.sort(
            key=lambda it: (
                (it.datetime + day_offset).strftime("%Y-%m-%d"),
                -float(it.properties.get("eo:cloud_cover", 100)),
            )
        )
        by_date = group_items_by_date(items, mid_longitude=mid_longitude)
        total_seen += len(by_date)
        prepare = partial(_prepare_date, baselines=baselines)
        if batch_dates > 1:
            # Only PASSING dates occupy batch slots — a skipped date adds no work to
            # the batch's compute, so letting it consume a slot would shrink batches
            # exactly where the gate filters most. The trailing partial batch flushes
            # at the end of each _drive call (one streamed month), like the pipeline's
            # month-boundary drain.
            nonlocal total_processed, total_filtered
            # With pipelining the look-ahead is the BATCH size, not 1: a batch's write
            # is one long consume, so a depth-1 buffer would hide only one date's
            # preparation out of k. Depth k has the next batch ready as this one lands.
            # Preparation stays single-threaded either way (see ingest._pipeline).
            supply = (
                pipelined(by_date.values(), prepare, depth=batch_dates)
                if pipeline_dates
                # Unpipelined: prepare inline and report the whole preparation as stall,
                # so the two modes stay comparable from one log line.
                else ((lambda p: (p, p.build_s + p.gate_s))(prepare(d)) for d in by_date.values())
            )
            batch: list[_PreparedDate] = []
            batch_stall = 0.0
            for prepared, stall_s in supply:
                if prepared.day_ds is None:
                    total_filtered += 1
                    continue
                batch.append(prepared)
                batch_stall += stall_s
                if len(batch) == batch_dates:
                    _write_batch(batch, baselines, batch_stall)
                    total_processed += len(batch)
                    batch, batch_stall = [], 0.0
            if batch:
                _write_batch(batch, baselines, batch_stall)
                total_processed += len(batch)
        elif pipeline_dates:
            # The pipeline drains at each month boundary, since it lives inside one
            # _drive call: one unhidden preparation per month, not worth threading
            # the buffer across the streamed months.
            for prepared, stall_s in pipelined(by_date.values(), prepare):
                _consume(prepared, stall_s)
        else:
            for day_items in by_date.values():
                prepared = prepare(day_items)
                # Serially the driver waited out the whole preparation, so ALL of it is
                # stall and none of it is hidden. Reporting 0.0 here would print the
                # serial baseline as a perfectly-hidden pipeline and make the two modes
                # incomparable, which is the one thing this line exists to do.
                _consume(prepared, prepared.build_s + prepared.gate_s)

    if stream_stac_monthly:
        # Stream month by month: querying the whole window up front retains every item
        # for the run's duration, which a zone-year cannot fit on this worker.
        for _mr, month_items, month_baselines in stream_stac_months(
            provider=provider,
            collection=collection,
            tile_id=None,
            start_date=start_date,
            end_date=end_date,
            bbox=roi.bbox_wgs84,
            existing_dates_fn=lambda: get_existing_dates(reflectance_store),
            log=log,
        ):
            _drive(month_items, month_baselines)
    else:
        items, baselines = query_stac_items(
            provider=provider,
            collection=collection,
            tile_id=None,
            start_date=start_date,
            end_date=end_date,
            existing_dates=get_existing_dates(reflectance_store),
            bbox=roi.bbox_wgs84,
        )
        if items:
            _drive(items, baselines)

    log.info("%d/%d dates passed coverage filter", total_processed, total_seen)

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
