"""Sentinel-2 L2A reflectance ingestion for ROI-based regions.

Pure-domain implementation extracted from the reference repo's
``flows/ingest_s2_roi_reflectance.py::process_roi_reflectance``. No
Prefect imports, no ``get_run_logger``, no ``get_client``: callers
supply a connected :class:`dask.distributed.Client` and a logger.

The algorithm is unchanged from the reference:

1. Read ROI metadata + mask and total its pixels for the coverage denominator.
   Lazy and totalled over the live windows only — the coverage ratio is cropped on
   BOTH sides, giving identical numbers because the mask is False outside them, and
   neither the mask nor ``any_valid`` is persisted, so no full-extent array is ever
   materialised.
2. Query STAC for the full date range; reduce duplicate items to one copy per
   tile-date (newest reprocessing preferred — see ``ingest.duplicates``), then
   sort cloudiest-first so the painter's-algorithm mosaic picks the clearest
   tile last.
3. Group items by ``solar_day``.
4. For each day:

   * Phase 1 — load only SCL, compute coverage from the same
     ``solar_day`` mosaic; reject days below ``min_valid_coverage``.
   * Phase 2 — load all bands, mask invalid pixels via the Phase 1
     ``any_valid`` mask, apply ROI mask, write the date's live windows via
     :func:`tessera_embeddings.storage.zarr_store.write_day_windows` with a
     narrow tenacity retry on transient GDAL errors. Past that retry, a source
     object that will never read — corrupt, or never published — steps DOWN to the
     tile-date's next catalogue copy; when every copy has failed the date is skipped
     and recorded, so the loss is a finding on the store rather than an unexplained
     gap.

Step 4 is split into a prepare half and a write half so ``pipeline_dates`` can
overlap one date's preparation with the previous date's write, and so
``batch_dates`` can compute several dates' writes as one graph (one commit per
BATCH — see ``storage.zarr_store.write_days_windows`` for why that unit is
forced). Writes stay in date order under every mode.

This module imports nothing from Prefect or any cloud provider.
"""

from __future__ import annotations

import logging
import math
import operator
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import reduce
from typing import final

import dask.array as da
import dask.distributed
import numpy as np
import xarray as xr

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE, INGEST_CHUNKS, auto_batch_dates
from tessera_embeddings.config.providers import PROVIDERS
from tessera_embeddings.config.satellites import S2_SCL_INVALID_CLASSES
from tessera_embeddings.ingest._pipeline import pipelined
from tessera_embeddings.ingest.asset_locations import Harmonisation
from tessera_embeddings.ingest.duplicates import (
    copies_label,
    is_unreadable_source,
    item_tile,
    log_duplicate_selection,
    select_preferred_duplicates,
    step_down_copies,
)
from tessera_embeddings.ingest.live_windows import (
    WINDOW_COST_IN_CHUNKS,
    WINDOW_COST_IN_CHUNKS_OVERLAPPED,
    live_windows_for_mask,
    windows_for_date,
)
from tessera_embeddings.ingest.loader_failures import (
    collect_aborted_hrefs,
    implicated_items,
    implicated_tile_dates,
    install_capture_everywhere,
    label_objects,
)
from tessera_embeddings.ingest.roi import (
    StorageOptions,
    read_roi_mask,
    read_roi_metadata,
    resolve_storage_options,
)
from tessera_embeddings.ingest.roi_processing import (
    DEFAULT_MIN_VALID_COVERAGE,
    read_failure_context,
    source_read_retrying,
)
from tessera_embeddings.ingest.solar_days import (
    normalize_to_solar_day,
    owned_items,
    solar_grouping_longitude,
    whole_window_range,
)
from tessera_embeddings.ingest.stac import (
    HeterogeneousProducerError,
    collection_harmonisation,
    extract_baselines,
    group_items_by_date,
    load_stac_items,
    query_stac_items,
    selection_read_keys,
    stream_stac_months,
)
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import (
    get_existing_dates,
    record_assessed_window,
    store_write_retrying,
    write_day_windows,
    write_days_windows,
)

logger = logging.getLogger(__name__)

#: Assets this ingest loads beyond the collection's primary bands.
#:
#: Named in ONE place because it has to reach two very different calls: the loader,
#: which reads SCL to build the cloud mask, and the STAC query, which prunes assets
#: off each item before the loader ever runs (see ``stac._loadable_assets``). The
#: query keeps ``scl`` on its own only for collections configured ``has_scl=True``;
#: a provider without that flag — Planetary Computer's ``sentinel-2-l2a`` — drops
#: the asset at query time and the load then fails on a band that was there in the
#: catalogue. Passing the same tuple to both is what makes the two agree.
_LOADED_EXTRA_BANDS = ["scl"]


def _known_harmonisation(provider: str, collection: str) -> Harmonisation | None:
    """The producer state this driver's collection settles, for duplicate selection to rank on.

    Paired with :func:`_read_asset_keys` and load-bearing where that returns nothing: without it a
    collection whose asset keys cannot be inspected reports every copy's producer as undecidable,
    and a spare that WILL refuse its date is offered to the fallback ladder — which recovers from a
    read failure and not from a refusal, so reaching that rung aborts the ingest.
    """
    return collection_harmonisation(PROVIDERS[provider].collections[collection])


def _read_asset_keys(provider: str, collection: str) -> tuple[str, ...]:
    """The assets this driver's loads request, for duplicate selection to judge readability over.

    This driver's extra bands applied to the shared rule — see
    :func:`~tessera_embeddings.ingest.stac.selection_read_keys` for why the answer can be empty.
    """
    return selection_read_keys(PROVIDERS[provider].collections[collection], _LOADED_EXTRA_BANDS)


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
            coverage filter. ONLY that — a date lost for any other reason is
            counted under its own field, or the two become indistinguishable
            in the one place a caller looks.
        dates_refused_producer_conflict: Number of dates refused because no
            single BOA-offset decision fits the day. Deliberate losses, not
            coverage rejections, and durably recorded on the store as
            ``assessed_unreadable_dates`` with ``scope=producer-conflict``.
    """

    roi_path: str
    status: str
    dates_processed: int
    dates_filtered_coverage: int
    dates_refused_producer_conflict: int = 0


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
    # An ROI with no live pixel at all — an all-ocean mask yields no live window and so a
    # zero denominator (see _sum_over_windows). No date can have coverage of nothing, so
    # fail the date rather than divide: the campaign screens these out with
    # zone_has_live_tiles, but the public ROI path has no such preflight and used to raise
    # ZeroDivisionError from inside the per-date gate.
    if roi_pixel_count == 0:
        return (False, None)

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
    items: Sequence[object] = ()
    """The day's STAC items, carried so the WRITE can name them on failure.

    The write's compute is where the reflectance bands are first read, so a source object
    that cannot be read fails there — after the coverage gate has already passed on SCL.
    Without the items the failure names no granule, and identifying the object then means
    correlating GDAL's own stderr by timestamp across every worker in the fleet.
    """
    baselines: dict[str, int] = field(default_factory=dict)
    """The processing baselines of THESE items — the correction applied, and recorded.

    Derived where the items are known rather than once per query, because the two lists
    differ: the query's map is built before duplicate copies are pruned and before a read
    failure steps down to an older one, so it can name the baseline of a copy the loader
    never opened. Sentinel-2's reflectance offset is keyed on that number and duplicate
    copies straddle the threshold it tests, so the wrong entry silently shifts every pixel
    of the date.
    """
    read_error: BaseException | None = None
    """Set when PREPARATION hit a source that would not read, instead of raising.

    The coverage gate is the first compute of the date, so an unreadable object fails
    there — before the write, which is where the duplicate-copy ladder lives. Raising
    would carry that failure past the ladder and out of the leg, stranding the whole
    zone-year on one bad object; returning it lets the consume side step down exactly as
    it does for a write failure. ``day_ds`` is None alongside it, so a caller that only
    checks for a skip must check this too before counting the date as filtered.
    """


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
    storage_options: StorageOptions = None,
    stream_stac_monthly: bool = True,
    overlap_window_writes: bool = True,
    pipeline_dates: bool = False,
    batch_dates: int | None = None,
    allow_ingest_code_mismatch: bool = False,
    s3_region: str | None = None,
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
            COMPOSES with ``pipeline_dates``:
            the look-ahead is then sized to the batch, so the next batch's whole
            preparation overlaps this batch's write instead of only one date's.
            ``None`` (the default) derives it from the ROI's covered window area via
            :func:`~tessera_embeddings.config.ingest.auto_batch_dates`, because the
            benefit is not monotonic in ROI size; 1 forces the one-commit-per-date
            path.
        allow_ingest_code_mismatch: Off by default. See the field of the same name on
            :class:`~tessera_embeddings.storage.manifest.IngestManifest`.

        s3_region: S3 region for the mosaic Icechunk store. ``None`` uses the
            storage layer's default; set it when the bucket lives elsewhere, or
            every write below signs against the wrong region.

    Returns:
        :class:`IngestResult`. ``status="skipped"`` if zero STAC items
        were returned or zero dates passed the coverage filter.
    """
    log = log or logging.getLogger(__name__)
    #: Short identifier for this ROI, stamped on every progress line so a fleet-wide log
    #: query can attribute a commit to a cell — see the note in ``s1_roi``.
    roi_label = roi_zarr_path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".zarr")
    if batch_dates is not None and batch_dates < 1:
        raise ValueError(f"batch_dates must be >= 1 or None for auto, got {batch_dates}")
    reflectance_store = f"{store_path}/reflectance.zarr"

    # Before the first read, because what it captures is only ever recorded as the read
    # happens: the loader names the object it gave up on in its own log record on whichever
    # worker hit it, and nothing in the exception carries it. Installed on every current and
    # future worker; never fatal, since the recovery it sharpens works without it.
    install_capture_everywhere(client)

    roi = read_roi_metadata(roi_zarr_path, storage_options=storage_options)

    # The coverage threshold goes in the manifest because it decides WHICH dates this
    # store holds, and an interrupted store records it nowhere else — so a resume at a
    # different threshold would skip the dates the old one admitted and append new ones
    # under the new rule. Validated on every write, so that refusal lands before the
    # resumed run commits a date.
    ingest_manifest = IngestManifest.from_roi_store(
        roi_zarr_path,
        min_valid_coverage=min_valid_coverage,
        storage_options=resolve_storage_options(storage_options),
        allow_ingest_code_mismatch=allow_ingest_code_mismatch,
    )

    mid_longitude = solar_grouping_longitude(roi)

    # Live windows for the cropped write path, derived once per run from the same
    # mask this ingest already reads (plain tuples: storage takes no ingest types).
    #
    # These describe where the ROI has LAND, so they are the same on every date. Each
    # date is then narrowed further to the land its own imagery can reach
    # (``windows_for_date``, applied per date below), because a satellite covers only
    # a fraction of a wide ROI per pass.
    # The merge exchange rate follows how this run WRITES: overlapped windows share one
    # graph, so a boundary is cheap and the DP should stop trading ocean area for fewer
    # windows. Sequential writes still pay the serial cost. Bound once because per-date
    # narrowing re-merges on the same terms — a second, differing rate there would undo
    # this for every narrowed date.
    window_cost = WINDOW_COST_IN_CHUNKS_OVERLAPPED if overlap_window_writes else WINDOW_COST_IN_CHUNKS
    run_windows = live_windows_for_mask(
        roi_zarr_path,
        window_px=INGEST_CHUNK_SIZE,
        window_cost_in_chunks=window_cost,
        storage_options=storage_options,
    )
    live_windows: list[tuple[int, int, int, int]] = [(w.y0, w.y1, w.x0, w.x1) for w in run_windows]
    log.info("Writing %d live window(s)", len(run_windows))

    # Resolve `batch_dates=None` (auto) now that the windows are known. Derived from
    # the area those windows COVER, which is what the write graph touches.
    if batch_dates is None:
        # CEIL, not floor. Windows are clamped to the ROI extent, so an edge window can
        # be narrower or shorter than one chunk — floor counts that dimension as zero, and
        # a tall narrow ROI (one partial-width column over many rows) totals zero covered
        # chunks. auto_batch_dates would then read a large graph as empty and enable
        # 4-date batching on it, which is the case the threshold exists to prevent.
        covered_chunks = sum(
            math.ceil((w.y1 - w.y0) / INGEST_CHUNK_SIZE) * math.ceil((w.x1 - w.x0) / INGEST_CHUNK_SIZE)
            for w in run_windows
        )
        batch_dates = auto_batch_dates(covered_chunks)
        log.info(
            "batch_dates=auto -> %d (%d chunk(s) covered by live windows)",
            batch_dates,
            covered_chunks,
        )

    # Load blocks match the store's chunks, so a date's read parallelism is one task
    # per (chunk, band) and the write needs no rechunk at all.
    spatial_chunks = {"northing": INGEST_CHUNKS["northing"], "easting": INGEST_CHUNKS["easting"]}

    # Leg-entry read, for the pixel total ONLY — computed immediately below, so the
    # credential it resolves cannot go stale before use. Every per-date consumer builds its
    # own graph instead (see ``date_mask``); do not reuse this one there.
    roi_mask = read_roi_mask(roi_zarr_path, spatial_chunks, storage_options=storage_options)
    # The mask stays LAZY and the total comes from the live windows only.
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

    if roi_pixel_count == 0:
        # Say so ONCE, here, rather than leaving the reader to infer it from every date
        # failing coverage: an ROI with no live pixel can only ever produce a skip.
        log.warning("ROI has no live pixels — every date will fail the coverage gate")

    total_processed = 0
    total_filtered = 0
    total_refused = 0
    total_seen = 0
    #: Tile-dates whose every copy failed to read. Their pixels are absent from the mosaic,
    #: so this is the ONLY record of where the loss is; it is re-stated at the end of the run.
    unreadable_tile_dates: list[dict[str, str]] = []
    # Days refused because no single offset decision fits them. Kept apart from
    # `unreadable_tile_dates` in ORIGIN and joined to it at the end for STORAGE, because both are
    # deliberate losses inside an assessed window and the store's record of "where the holes are"
    # must name them all. Kept out of the coverage counter, whose contract is the SCL gate: a run
    # that lost most of a year to metadata was reporting it as coverage filtering.
    producer_conflict_dates: list[dict[str, str]] = []

    def _prepare_date(day_items: list) -> _PreparedDate:
        """Build one solar day's write-ready dataset, or the reason it has none.

        A closure so the streamed and single-query paths run byte-identical work; the
        per-date logic must not fork on how its items were supplied.

        The processing baselines are derived HERE, from ``day_items``, and travel back on
        the result. They used to be a whole-query map threaded in from the caller, built
        before duplicate copies were pruned and never rebuilt when a read failure stepped
        down to an older copy — so the offset applied to a date's pixels, and the
        provenance recorded beside them, could belong to a copy the loader never opened.
        Deriving them from the items being loaded makes that disagreement unrepresentable
        rather than merely fixed.

        SIDE-EFFECT-FREE by contract: under ``pipeline_dates`` this runs on a
        background thread while the previous date is being written, so anything it
        mutated outside its return value would race the writer. Everything the write
        needs travels back in the :class:`_PreparedDate` — including a read failure,
        which is RETURNED rather than raised for the same reason (see ``read_error``).
        """
        baselines = extract_baselines(day_items)
        # THE GROUP'S SOLAR DAY, which is what this mosaic slice represents. Every item in
        # the group shares it by construction — it is the grouping key — so any item yields
        # it, and shifting one item is cheaper than threading the key through the pipeline.
        #
        # Deliberately NOT taken from the loaded dataset's own time coordinate. odc stamps
        # each group with `group[0].nominal_datetime`, and `preserve_original_order=True`
        # (needed so the clearest tile paints last) makes group[0] the CLOUDIEST item, whose
        # acquisition time is arbitrary within the day. Where the solar offset is large
        # enough to cross UTC midnight, that timestamp's calendar date can be the day
        # BEFORE the solar day — so two consecutive solar days can normalise onto the same
        # date and collide. Measured on 56N (+10 h): six of twenty-two days landed on the
        # previous date, and which six depended on cloud cover, not geography.
        # Items are solar-day-normalised at the query chokepoint, so every item in the
        # group carries the same canonical timestamp and this is the solar day itself —
        # no offset here, and no dependence on WHICH item the sort left first.
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
                extra_bands=_LOADED_EXTRA_BANDS,
                resampling="bilinear",
                groupby="solar_day",
                geobox=roi.geobox,
            )
        except HeterogeneousProducerError as exc:
            # The day has no correct offset decision, so it is SKIPPED — loudly, and alone.
            #
            # Not carried back as a read failure. The duplicate-copy ladder recovers from an
            # object that will not read, and this is not that. The ladder cannot contain a refusing
            # copy by construction: `_preference_key` ranks one last and `select_preferred_duplicates`
            # withholds it from the alternates, so stepping down would burn every rung and refuse
            # identically.
            #
            # And not raised, which is what it used to do. A refusal is deterministic and belongs
            # to ONE day, so propagating it failed the whole leg — every retry of the leg then
            # reached the same day and died the same way, losing a whole zone-year to one day's
            # metadata. Measured on zone 01N: 347 of 366 days in 2024 refused, so propagating made
            # that cell unfillable rather than merely incomplete.
            #
            # The correction stays refused either way: nothing here corrects a day it cannot
            # decide. This changes only how much is lost when it cannot — one day instead of the
            # year. What remains reachable is narrow: the offset is decided per source, so a day
            # mixing producers no longer refuses. A source whose bucket is classified as neither
            # harmonised nor unharmonised does, and the fix for that is to classify the bucket in
            # `asset_locations` or to correct the catalogue's `s2:processing_baseline` — see
            # `context_docs/decisions/021-correct-the-boa-offset-per-image.md`.
            log.warning(
                "Skipping %s for roi=%s: no single offset decision fits the day. %s",
                date,
                roi_label,
                exc,
            )
            return _PreparedDate(
                date,
                None,
                [],
                time.monotonic() - stage_started,
                0.0,
                "producer-conflict",
                items=day_items,
                baselines=baselines,
            )
        except ValueError as exc:
            # Earth-search occasionally publishes asset-incomplete items (missing SCL
            # and/or reflectance bands). odc.stac.load resolves bands eagerly, so one such
            # item raises before any graph is built.
            #
            # Carried back as a read failure, NOT as a plain skip. A missing asset is a
            # property of the COPY, not of the day: a different reprocessing of the same
            # tile-date routinely publishes the full set, and the ladder that would try it
            # is the same one an unreadable object uses. Returning a bare skip here lost
            # the whole solar day while a usable copy sat one rung down.
            if "No such band/alias" not in str(exc):
                raise
            log.warning("Load failed on asset-incomplete STAC item(s): %s", exc)
            return _PreparedDate(
                date,
                None,
                [],
                time.monotonic() - stage_started,
                0.0,
                "asset-incomplete",
                items=day_items,
                baselines=baselines,
                read_error=exc,
            )

        # The gate is where the graph is first COMPUTED, so it is where a source read
        # actually fails — `load_stac_items` above only builds. Retried per date and named
        # on failure: an unretried read here used to propagate out of the loop and fail the
        # whole zone-year, and its message carried neither the zone nor the date.
        built_at = time.monotonic()
        # The mask graph is rebuilt PER DATE, not reused from leg entry. The reads it
        # performs recur per date either way (nothing is persisted), but a graph built
        # once carries whatever credential was resolved then — and an IAM role credential
        # expires in hours, which a leg outlives. Rebuilding moves the resolution to the
        # date that consumes it; the graph itself is cheap next to the pixels it reads.
        date_mask = read_roi_mask(roi_zarr_path, spatial_chunks, storage_options=storage_options)
        try:
            with read_failure_context(log, roi=roi_label, date=date, items=day_items):
                for attempt in source_read_retrying(log):
                    with attempt:
                        passes, any_valid = _coverage_from_scl(
                            day_ds["scl"].isel(time=0),
                            date_mask,
                            roi_pixel_count,
                            min_valid_coverage,
                            client,
                            windows=live_windows,
                        )
        except Exception as exc:
            # The gate is the FIRST compute of the date, so an SCL object that will never
            # read fails here — one stage before the write, which is where the
            # duplicate-copy ladder lives. Raising sent that failure straight out of the
            # leg, so a single bad preferred copy stranded the whole zone-year and did so
            # identically on every retry. Returned instead, it reaches the same ladder the
            # write's failures do, and an older copy of the same tile-date gets its turn.
            if not is_unreadable_source(exc):
                raise
            build_s, gate_s = built_at - stage_started, time.monotonic() - built_at
            return _PreparedDate(
                date, None, [], build_s, gate_s, "unreadable", items=day_items, baselines=baselines, read_error=exc
            )
        build_s, gate_s = built_at - stage_started, time.monotonic() - built_at
        if not passes:
            return _PreparedDate(date, None, [], build_s, gate_s, "coverage")

        # Stamp the slice with its SOLAR DAY. The axis is day-granular either way — this
        # only decides WHICH day — and taking it from the grouping rather than from odc's
        # label is what makes the value identify the day it describes, unique per slice and
        # monotonic across them.
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
        date_windows: list[tuple[int, int, int, int]] = live_windows
        if run_windows:
            narrowed = windows_for_date(
                run_windows,
                [getattr(item, "bbox", None) for item in day_items],  # type: ignore[misc]
                roi.geobox,
                chunk_px=INGEST_CHUNK_SIZE,
                window_cost_in_chunks=window_cost,
            )
            if not narrowed:
                # No live cell is reachable today. Nothing to write, and writing an
                # empty window set would commit a date holding nothing.
                # DEBUG: per skipped date; counted in the coverage-filter summary.
                log.debug("Skipping date: its imagery reaches no live window")
                return _PreparedDate(date, None, [], build_s, gate_s, "no-live-window")
            date_windows = [(w.y0, w.y1, w.x0, w.x1) for w in narrowed]
            if len(narrowed) != len(run_windows):
                # DEBUG, not INFO: this fires once per date, and Prefect ships every task
                # log line to the orchestrator API from whichever DASK WORKER ran the task
                # (logging.to_api is on by default). Every worker in every fleet is
                # therefore an API client, so a per-date INFO line scales with total
                # worker count rather than with cell count. The per-date TIMING line stays
                # at INFO because it is the progress signal; this one is detail, and its
                # numbers appear there too.
                log.debug(
                    "Date footprint: writing %d of %d live window(s)",
                    len(narrowed),
                    len(run_windows),
                )

        # ONE masking pass, not two. Zeroing invalid pixels and zeroing outside the
        # ROI both fill with 0, so `x.where(A, 0).where(B, 0)` is `x.where(A & B, 0)`
        # — and each `where` is a graph task per (chunk, band), which is the budget
        # that limits ingest. SCL keeps only the ROI mask: it is categorical, and
        # zeroing it by its own validity would rewrite the class codes it carries.
        roi_2d = date_mask
        keep = roi_2d if any_valid is None else (any_valid & roi_2d)
        for var in day_ds.data_vars:
            mask_for_var = roi_2d if str(var) == "scl" else keep
            day_ds[str(var)] = day_ds[str(var)].where(mask_for_var, other=0)

        return _PreparedDate(date, day_ds, date_windows, build_s, gate_s, items=day_items, baselines=baselines)

    def _write_date(prepared: _PreparedDate, stall_s: float) -> None:
        """Write one prepared date's pixels: the store's only writer.

        Runs on the driver thread, one date at a time, under both modes — icechunk
        commits are sequential on one branch and one commit per date is the contract,
        so this half is what stays serial when preparation is overlapped.
        """
        day_ds = prepared.day_ds
        assert day_ds is not None, f"date {prepared.date} was skipped ({prepared.skip_reason}); nothing to write"
        write_started = time.monotonic()

        # Retries intermittent GDAL errors under high parallelism; a failed cropped date
        # commits nothing, so the retry re-runs only the write, from the graph preparation
        # already built. A second writer is excluded from the retry — see
        # store_write_retrying.
        #
        # Named on failure, exactly as the coverage gate and the radar write are. The
        # reflectance bands are first READ here, so a source object that cannot be read
        # fails at this point rather than at the gate, which only reads SCL — and a
        # permanently unreadable object exhausts the retry and kills the leg. Which object
        # it was is the difference between a one-command diagnosis and a fleet-wide log
        # correlation.
        with read_failure_context(log, roi=roi_label, date=prepared.date, items=prepared.items):
            for attempt in store_write_retrying(log):
                with attempt:
                    write_day_windows(
                        reflectance_store,
                        day_ds,
                        prepared.windows,
                        roi=roi,
                        manifest=ingest_manifest,
                        baselines=prepared.baselines,
                        tile_id=roi_zarr_path,
                        crs=roi.native_crs,
                        chunks=INGEST_CHUNKS,
                        parallel_windows=overlap_window_writes,
                        s3_region=s3_region,
                    )
        # One line per kept date, partitioning its wall clock into the client-side
        # graph build, the coverage-gate compute, and the write (windows + commit).
        # Stable format: CloudWatch queries and the pipeline analysis key off it.
        # `total` remains build+gate+write — the date's SERIAL-EQUIVALENT cost, so the
        # figure stays comparable across both modes however much of it was hidden.
        write_s = time.monotonic() - write_started
        prepare_s = prepared.build_s + prepared.gate_s
        log.info(
            "Stage timings roi=%s date=%s: build=%.1fs gate=%.1fs write=%.1fs total=%.1fs windows=%d mode=%s",
            roi_label,
            prepared.date,
            prepared.build_s,
            prepared.gate_s,
            write_s,
            prepare_s + write_s,
            len(prepared.windows),
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

    def _write_batch(batch: list[_PreparedDate], stall_s: float = 0.0) -> None:
        """Write a batch of prepared dates: one dask compute, ONE commit.

        The same retry contract as ``_write_date``, at batch granularity: the
        batched write commits nothing on failure, so a retry re-runs the whole
        batch cleanly from the graphs already prepared.
        """
        write_started = time.monotonic()
        for attempt in store_write_retrying(log):
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
                    baselines={d: b for p in batch for d, b in p.baselines.items()},
                    tile_id=roi_zarr_path,
                    crs=roi.native_crs,
                    chunks=INGEST_CHUNKS,
                    parallel_windows=overlap_window_writes,
                    s3_region=s3_region,
                )
        # The batch's write is ONE compute, so a per-date write time does not exist
        # as a measurement — this line is the batched counterpart of `Stage timings`
        # and analysis divides by n. build/gate are sums of the real per-date values.
        write_s = time.monotonic() - write_started
        prepare_s = sum(p.build_s + p.gate_s for p in batch)
        log.info(
            "Batch timings roi=%s dates=%s..%s n=%d: build=%.1fs gate=%.1fs write=%.1fs windows=%d "
            "prepare=%.1fs hidden=%.1fs stall=%.1fs",
            roi_label,
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

    def _drive(items: list) -> None:
        """Sort one supply of items cloudiest-first, group by date, and ingest each.

        Normalises defensively on the way in. Both suppliers — the streamed months and the
        single whole-window query — already do it, but both are injectable, and this is the
        last point before a date is derived. The operation is idempotent, so the honest
        cost of the guarantee is one dict build per supply.
        """
        nonlocal total_seen
        # SCOPED TO THIS SUPPLY, not to the run. The ladder is only ever consulted while
        # recovering a date from THIS supply — `step_down_copies` looks up the keys of the
        # date it is given — so entries from earlier months are dead weight that nothing can
        # reach. Holding them made the rejected STAC items of a whole year resident and
        # defeated the month-at-a-time memory bound the streaming supplier exists to provide.
        date_alternates: dict[tuple[str, str], list] = {}
        items = normalize_to_solar_day(items, mid_longitude=mid_longitude)
        # Duplicate items for one tile-date must be reduced to ONE copy before the loader
        # sees them, because the loader FUSES a solar-day group: with both copies in it, an
        # unreadable copy fails the date and there is nothing to fall back to. Rejected
        # copies are retained as the fallback ladder the write steps down on a persistent
        # read failure. (The BASELINE a date is corrected by is derived later, from the
        # items a preparation actually loads — pruning here does not by itself make the two
        # agree, and an earlier version of this comment claimed it did.)
        supplied = items
        read_keys = _read_asset_keys(provider, collection)
        known_kind = _known_harmonisation(provider, collection)
        items, alternates = select_preferred_duplicates(items, read_keys, known_kind)
        log_duplicate_selection(log, roi_label, alternates, kept=items, read_keys=read_keys, items=supplied)
        date_alternates.update(alternates)

        def _record_skip(prepared: _PreparedDate) -> None:
            """Account for a date that produced nothing, under the reason it actually had.

            A producer conflict is not a coverage rejection and must not be counted as one: the
            coverage counter's contract is the SCL gate, and a run that lost most of a year to
            catalogue metadata was reporting that as coverage filtering. It is also recorded
            durably, for the same reason an unreadable date is — an assessed window makes absence
            inside it a finding rather than a gap, so a hole nobody recorded is a hole no later run
            revisits, and the warning explaining it does not outlive the log.
            """
            nonlocal total_filtered, total_refused
            if prepared.skip_reason != "producer-conflict":
                total_filtered += 1
                return
            total_refused += 1
            producer_conflict_dates.append(
                {
                    "date": prepared.date,
                    "tiles": ",".join(sorted({t for it in prepared.items if (t := item_tile(it)) is not None})),
                    "tried": copies_label(prepared.items),
                    "objects": "",
                    "scope": "producer-conflict",
                }
            )

        def _record_unreadable(
            prepared: _PreparedDate,
            exc: BaseException,
            tried: list[str],
            blamed: set[tuple[str, str]] | None,
            hrefs: list[str],
        ) -> None:
            """Accept the loss for one date, as loudly as a log can manage.

            Every copy of some object in this date failed to read, so its pixels cannot be
            produced by any retry. The date is skipped rather than the leg failed — losing
            one date beats losing every later date — which makes this line the ONLY place
            the loss is visible, and the reason it names each copy it tried. The same set is
            re-stated at the end of the run and recorded on the store, because a log line
            alone is lost the moment nobody greps for it.

            ``scope`` on the durable record says how precisely the loss is located.
            ``attributed`` means the named objects are the ones the loader actually gave up
            on, so the tiles listed are the tiles that lost pixels. ``whole-date`` means the
            failing object could not be identified, and the tiles listed are every tile in
            the date — of which an unknown few lost pixels. Recording which of the two it is
            matters because the second cannot be acted on the way the first can.
            """
            all_tiles = sorted({t for it in prepared.items if (t := item_tile(it)) is not None})
            attributed = bool(blamed)
            tiles = sorted({tile for tile, _ in blamed}) if blamed else all_tiles
            unreadable_tile_dates.append(
                {
                    "date": prepared.date,
                    "tiles": ",".join(tiles),
                    "tried": ",".join(tried) or copies_label(prepared.items, only=blamed),
                    "objects": label_objects(hrefs),
                    "scope": "attributed" if attributed else "whole-date",
                }
            )
            log.error(
                "DATA LOSS roi=%s date=%s: every catalogue copy failed to read, so this date "
                "is SKIPPED and its pixels are absent from the mosaic. objects=%s scope=%s "
                "tiles=%s copies_tried=%s last_error=%s — the objects are unreadable at the "
                "provider, so no retry of this run recovers it; re-check the catalogue for a "
                "newly reprocessed copy.",
                roi_label,
                prepared.date,
                label_objects(hrefs),
                "attributed" if attributed else "whole-date",
                ",".join(tiles) or "unknown",
                ",".join(tried) or copies_label(prepared.items, only=blamed),
                exc,
                exc_info=True,
            )

        def _consume(prepared: _PreparedDate, stall_s: float) -> None:
            """Count or write one prepared date — the ONE consume path both modes take.

            Serial and pipelined differ only in where ``prepared`` came from, so the
            counters and the write cannot drift between them.

            A persistent source-read failure steps DOWN the duplicate ladder here rather
            than failing the leg: the transient retry inside the write has already been
            exhausted by this point, so what is left is an object that does not read, and a
            duplicated tile-date has another copy to try. Only when every copy has failed is
            the date given up, and that is recorded rather than swallowed.

            Which copies step down is decided by ASKING THE CLUSTER what it aborted on, since
            the exception itself names nothing. That collection is destructive and
            cluster-wide, so a date failing while a concurrent date's read is aborting can
            drain the other's evidence; the other date then attributes nothing and steps its
            whole ladder, which is the behaviour it had before attribution existed. That is
            the right direction for the race to fail — a lost attribution costs precision,
            while a borrowed one would step down a tile that read.
            """
            nonlocal total_processed, total_filtered
            if prepared.day_ds is None and prepared.read_error is None:
                _record_skip(prepared)
                return
            attempt: _PreparedDate = prepared
            tried: list[str] = []
            while True:
                # A date can arrive here ALREADY failed: the coverage gate is the first
                # compute, so an unreadable SCL object fails during preparation, one stage
                # before the write. Preparation returns that failure rather than raising
                # it (see `_PreparedDate.read_error`) precisely so it lands in this ladder
                # instead of leaving the leg — the two failures want the same remedy, and
                # only one of them used to get it.
                exc: BaseException | None = attempt.read_error
                try:
                    if exc is None:
                        _write_date(attempt, stall_s)
                        total_processed += 1
                        return
                except Exception as write_exc:
                    if not is_unreadable_source(write_exc):
                        raise
                    exc = write_exc

                hrefs = collect_aborted_hrefs(client)
                # An href that matches nothing in THIS date is another date's, or a stale
                # line from a failure already handled — the collection is cluster-wide and
                # destructive, so both happen. An empty match is therefore "could not
                # attribute", not "no copies to try": passing the empty set on would step
                # down nothing and record a recoverable date as permanently lost. `None`
                # puts it back on the whole-date ladder, which is what this did before
                # attribution existed.
                blamed = (implicated_tile_dates(attempt.items, hrefs) or None) if hrefs else None
                # The failing ITEMS, not just their tile-dates: a tile-date can hold several
                # acquisitions, and the ladder must step the one that failed rather than the
                # one whose spare happens to rank highest (see step_down_copies).
                bad_items = implicated_items(attempt.items, hrefs) if hrefs else []
                stepped = step_down_copies(date_alternates, attempt.items, only=blamed, implicated=bad_items)
                if stepped is None:
                    _record_unreadable(attempt, exc, tried, blamed, hrefs)
                    total_filtered += 1
                    return
                stepped_items, swapped = stepped
                before = copies_label(attempt.items, only=swapped)
                after = copies_label(stepped_items, only=swapped)
                if not tried:
                    tried.append(before)
                tried.append(after)
                log.warning(
                    "Source read failed roi=%s date=%s on objects=%s — falling back from "
                    "copies=%s to %s (%d tile-date(s) stepped, attribution=%s). The "
                    "preferred copy is the newer reprocessing, so this trades processing "
                    "baseline for a date that reads.",
                    roi_label,
                    attempt.date,
                    label_objects(hrefs),
                    before,
                    after,
                    len(swapped),
                    "objects" if blamed else "whole-date",
                )
                attempt = _prepare_date(stepped_items)
                if attempt.day_ds is None and attempt.read_error is None:
                    # The fallback copy was prepared and skipped on its own merits (it
                    # failed coverage, or reaches no live window). That is a legitimate
                    # skip, not a read failure, and must not be recorded as data loss.
                    # A fallback that ALSO failed to read keeps its place in the ladder.
                    total_filtered += 1
                    return

        def _write_batch_or_isolate(batch: list[_PreparedDate], stall_s: float) -> None:
            """Write a batch, falling back to one date at a time if a source will not read.

            The duplicate ladder and the give-up-and-record path both live in
            :func:`_consume`, which the batched path never reaches — so without this a
            single permanently unreadable object anywhere in a batch fails the whole S2
            leg, and fails it identically on every retry, stranding the zone-year. Compact
            ROIs auto-batch, so that is the DEFAULT path for them rather than an opt-in.

            Isolating is safe because the batched write commits nothing on failure (see
            :func:`_write_batch`): re-running its dates singly starts from the same clean
            state, and each then gets the per-date recovery — alternate copies first, and
            only the date that has run out of copies given up and recorded.

            Only an unreadable SOURCE is isolated. Anything else propagates, because
            re-running k dates one by one to watch each hit the same non-data failure
            costs k times as long to reach the same answer.
            """
            nonlocal total_processed
            try:
                _write_batch(batch, stall_s)
            except Exception as exc:
                if not is_unreadable_source(exc):
                    raise
                log.warning(
                    "Batched write failed roi=%s dates=%s..%s n=%d on an unreadable source (%s) — "
                    "retrying the batch one date at a time so the duplicate-copy fallback applies "
                    "and at most the unreadable date is lost.",
                    roi_label,
                    batch[0].date,
                    batch[-1].date,
                    len(batch),
                    exc,
                )
                # Zero stall: the batch's preparation stall was already spent and cannot be
                # re-attributed per date. The metric compares steady-state modes, and this
                # path is not one.
                for prepared in batch:
                    _consume(prepared, 0.0)
                return
            total_processed += len(batch)

        # query_stac_items sorts by (date, cloud_cover ASC). Re-sort cloudiest-first so
        # the clearest tile paints last (wins) in solar_day's painter's algorithm.
        # group_items_by_date preserves this within-group order.
        #
        # Both the sort and the grouping key off the SOLAR day, not the UTC date, because
        # that is what the loader groups by. Using UTC here let a group the loader saw as
        # two solar days arrive as two time slices, against a cloud mask reduced to one —
        # a dimension conflict, and one that fires only where the solar offset is large
        # enough to cross UTC midnight (the far-eastern and far-western zones).
        items.sort(
            key=lambda it: (
                it.datetime.strftime("%Y-%m-%d"),
                -float(it.properties.get("eo:cloud_cover", 100)),
            )
        )
        by_date = group_items_by_date(items)
        total_seen += len(by_date)
        prepare = _prepare_date
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
                if prepared.read_error is not None:
                    # Preparation could not read a source. It has no dataset to batch, and
                    # the duplicate ladder lives in the per-date consume — so hand it there
                    # rather than counting it as a filtered date, which would record a
                    # recoverable date as gone without ever trying its older copy.
                    #
                    # FLUSH FIRST. Dates arrive in ascending order, so everything already
                    # held is EARLIER than this one, and `_consume` commits immediately on
                    # a successful recovery. Recovering before flushing therefore commits a
                    # later date ahead of the held earlier ones, which the time axis refuses
                    # (`NonMonotonicDateError`) — so the leg fails on a date it had already
                    # read successfully. Ordering is nearly free to preserve here and
                    # impossible to repair afterwards.
                    if batch:
                        _write_batch_or_isolate(batch, batch_stall)
                        batch, batch_stall = [], 0.0
                    _consume(prepared, stall_s)
                    continue
                if prepared.day_ds is None:
                    _record_skip(prepared)
                    continue
                batch.append(prepared)
                batch_stall += stall_s
                if len(batch) == batch_dates:
                    _write_batch_or_isolate(batch, batch_stall)
                    batch, batch_stall = [], 0.0
            if batch:
                _write_batch_or_isolate(batch, batch_stall)
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
        for _mr, month_items, _month_baselines in stream_stac_months(
            provider=provider,
            collection=collection,
            tile_id=None,
            start_date=start_date,
            end_date=end_date,
            bbox=roi.bbox_wgs84,
            existing_dates_fn=lambda: get_existing_dates(reflectance_store, s3_region=s3_region),
            # Same assets _drive will load: pruning happens in the query, so a band
            # named only at load time is already gone by then.
            extra_bands=_LOADED_EXTRA_BANDS,
            # The loader groups by solar day, so the month partition must too — see
            # stream_stac_months. Without it a solar day straddling a month boundary
            # is written twice, once per half.
            mid_longitude=mid_longitude,
            log=log,
        ):
            _drive(month_items)
    else:
        # One query for the whole window, and it needs the SAME own-versus-query
        # separation the streamed path uses. Asking for exactly [start, end] cannot
        # return the imagery that belongs to the first and last solar day but carries an
        # adjacent UTC date, so those two days were quietly written short. The range
        # pads the query and still owns only the window — see ingest.solar_days.
        window = whole_window_range(start_date, end_date)
        items, _baselines = query_stac_items(
            provider=provider,
            collection=collection,
            tile_id=None,
            start_date=window.query_start,
            end_date=window.query_end,
            existing_dates=get_existing_dates(reflectance_store, s3_region=s3_region),
            bbox=roi.bbox_wgs84,
            # As in the streamed branch: the query prunes, so it must be told.
            extra_bands=_LOADED_EXTRA_BANDS,
            # The committed dates are SOLAR days (that is what _drive groups and writes),
            # so the filter has to key on solar days too — see stream_stac_months, which
            # passes the same value for the same reason.
            mid_longitude=mid_longitude,
        )
        owned = owned_items(normalize_to_solar_day(items, mid_longitude=mid_longitude), window)
        if owned:
            _drive(owned)

    log.info("%d/%d dates passed coverage filter", total_processed, total_seen)

    # Record the range examined IN FULL, so a month absent from this store reads as a
    # finding rather than a gap (storage.zarr_store.record_assessed_window). S2 skips a date
    # whose imagery reaches no live window exactly as S1 does, so it needs the same record.
    # Only when a store exists: with nothing written there is nothing to annotate.
    #
    # `reflectance_store`, NOT `store_path`: the attr belongs on the repo the coverage gate
    # opens, and `store_path` is the mosaic parent directory holding all three child repos.
    # record_assessed_window only logs when the open fails, so passing the parent left the
    # attr unwritten and silently turned every legitimately empty month back into a gap.
    #
    # Keyed on the STORE, not on `total_processed`. That counter is what THIS invocation
    # wrote, and the case that needs the attr most writes nothing: a run interrupted after
    # its last date commit but before this line leaves every date present and the attr
    # absent, so the resume dedupes all of them away, takes the zero-write path, and skips
    # the record again — on every retry, forever. The gate then reads a legitimately empty
    # month as an unexplained gap and the zone-year can never complete. The extra probe
    # runs ONLY in the zero-write case, so a normal run pays nothing for it.
    if total_processed or get_existing_dates(reflectance_store, s3_region=s3_region):
        # Both kinds of deliberate loss, in ONE durable list. The attr answers "where are the
        # holes in this assessed window", and a hole is a hole whether the objects would not read
        # or the day had no correct offset decision. `scope` distinguishes them for anyone acting
        # on the record.
        record_assessed_window(
            reflectance_store,
            start_date,
            end_date,
            unreadable=[*unreadable_tile_dates, *producer_conflict_dates],
            s3_region=s3_region,
        )

    if unreadable_tile_dates:
        # Re-stated at the END, because the per-date line is thousands of lines back by now
        # and a reader scanning a finished leg would never reach it. This is the summary that
        # says a green leg is nonetheless missing pixels, and where.
        log.error(
            "DATA LOSS SUMMARY roi=%s: %d date(s) skipped because every catalogue copy was "
            "unreadable — %s. Recorded on the store as assessed_unreadable_dates, so the gap "
            "reads as a finding rather than as an unexamined window.",
            roi_label,
            len(unreadable_tile_dates),
            "; ".join(f"{u['date']} objects={u['objects']} scope={u['scope']}" for u in unreadable_tile_dates),
        )

    if producer_conflict_dates:
        # Stated at the END for the same reason as the unreadable summary, and separately from it
        # because the cause and the remedy differ: this one is not a read failure and no retry or
        # fallback copy addresses it. It is the catalogue offering a day that cannot be corrected
        # as a whole.
        log.error(
            "DATA LOSS SUMMARY roi=%s: %d date(s) refused because no single offset decision fits "
            "the day — %s. Recorded on the store as assessed_unreadable_dates with "
            "scope=producer-conflict. These are NOT coverage rejections and are counted "
            "separately, and no retry or fallback copy addresses one. The remedy is to classify the "
            "source's bucket in `asset_locations` as harmonised or unharmonised, or to fix the "
            "catalogue item's `s2:processing_baseline`; the exception text names which applies.",
            roi_label,
            len(producer_conflict_dates),
            "; ".join(f"{c['date']} tiles={c['tiles']}" for c in producer_conflict_dates[:20]),
        )

    if total_processed == 0:
        return IngestResult(
            roi_path=roi_zarr_path,
            status="skipped",
            dates_processed=0,
            dates_filtered_coverage=total_filtered,
            dates_refused_producer_conflict=total_refused,
        )

    return IngestResult(
        roi_path=roi_zarr_path,
        status="success",
        dates_processed=total_processed,
        dates_filtered_coverage=total_filtered,
        dates_refused_producer_conflict=total_refused,
    )
