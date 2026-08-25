"""Sensor-agnostic ROI processing: mask application and coverage filtering.

These operations work on any xarray/Dask Dataset with spatial dimensions (northing, easting)
and a quality band. They are shared by S2 and S1 ROI ingestion flows.
"""

import logging
from collections.abc import Iterator, Sequence, Set
from contextlib import contextmanager
from typing import cast

import dask.array as da
import dask.distributed
import numpy as np
import xarray as xr
from tenacity import Retrying, before_sleep_log, stop_after_attempt, wait_exponential

from tessera_embeddings.ingest.duplicates import cause_was_flattened
from tessera_embeddings.ingest.loader_failures import carry_logged_refusal
from tessera_embeddings.ingest.roi import read_roi_mask

logger = logging.getLogger(__name__)

# Default minimum percentage of valid ROI pixels required to keep a date.
DEFAULT_MIN_VALID_COVERAGE = 5.0

#: Attempts for ONE date's source read.
#:
#: Sized to OUTLAST a provider having a bad minute, not merely to survive a single dropped
#: packet. Three attempts on the ladder below spend about six seconds of patience, which is
#: nothing against a source that is throttling: the read then fails, and a failure at this
#: layer is what decides whether a date is dropped. So the depth of this number is the
#: difference between waiting out a refusal and recording recoverable imagery as lost.
#:
#: The cost is bounded and falls entirely on the failing path: a transient read succeeds on its
#: second or third attempt and pays only the backoff, while a genuinely dead object pays the
#: full ladder once and then fails the leg — which is the intended outcome, because the leg
#: retries from the dates it already committed.
#:
#: Read alongside the two levers this multiplies with: the alternate-copy step-down per date,
#: and the leg's own attempts. The product is what a struggling provider sees, so raising this
#: is not free and the other two are the reason it is not raised further.
SOURCE_READ_ATTEMPTS = 8


def source_read_retrying(log: logging.Logger | logging.LoggerAdapter[logging.Logger]) -> Retrying:
    """Retry the source read for one date.

    Writes have always retried; reads did not, and that asymmetry cost whole cells. A
    transient failure reading a single granule propagates out of the per-date loop and
    fails the entire zone-year, abandoning every month the run had already committed.

    Scoped to ONE date deliberately. A retry at the task level would re-run the whole
    multi-day loop, which is why the ingest task refuses ``@task(retries=...)``; this sits
    inside the loop instead, so a retry re-reads only the date that failed.

    Not narrowed by exception type. Reads fail through several unrelated surfaces —
    rasterio, GDAL/CPL, botocore, plain socket timeouts — and enumerating them is how a new
    transient class silently becomes fatal.

    A read is idempotent, so retrying one that turns out to be permanent is safe — but it is no
    longer cheap, and the cost belongs here rather than in a fleet-budget guess. The ladder is
    :data:`SOURCE_READ_ATTEMPTS` attempts with seven exponential sleeps: **about 61 seconds of
    backoff per permanently-failing date**, on top of the read time each attempt spends. That is
    the price of outlasting a provider having a bad minute, and it falls entirely on the failing
    path — a transient read succeeds on its second or third attempt and pays only the first
    sleeps.
    """
    return Retrying(
        stop=stop_after_attempt(SOURCE_READ_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        before_sleep=before_sleep_log(cast("logging.Logger", log), logging.WARNING),
        reraise=True,
    )


@contextmanager
def read_failure_context(
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    roi: str,
    date: str,
    items: Sequence[object] = (),
    client: dask.distributed.Client | None = None,
) -> Iterator[None]:
    """Name the ROI, the date and the granules on any failure raised inside, and complete its reason.

    Per-date telemetry is emitted *after* a date commits, so the last date in a log is the
    last date that WORKED — a date that failed leaves no record of having been attempted,
    and the furthest date reached reads as progress rather than as the failure point.

    The ROI label matters as much as the date. The failure is raised on a Dask worker whose
    message carries no zone, so at fleet width an error cannot be attributed to a cell at
    all: the same error text appears for every zone running at once.

    rasterio compounds both. It reports ``Read failed. See previous exception for details.``
    and that previous exception is GDAL's, which is discarded unless the chain is logged —
    ``log.exception`` is what preserves it, and without it the message names a detail the
    reader has no way to reach.

    And a chain that never crossed the worker boundary cannot be logged at all, which is the
    second line here: it fires only for a failure that arrived already flattened, and it is the
    only warning that a verdict about to be taken has no evidence under it. Anything it reports
    is a worker ``ingest/loader_failures.py`` did not reach.

    Emitted only on failure, so it costs nothing on the success path and does not scale
    with worker count the way a per-date INFO line does.

    And the reason itself may be INCOMPLETE, which is what ``client`` is for. GDAL states some
    refusals only in its own log and raises the codec failure that follows from them, so the
    chain can say the bytes are bad while the words saying the service refused sit in a log
    line. Both sensors' reads pass through here, so this is where that evidence is collected
    and attached — one classifier, one set of evidence, on either path. Without a client the
    cluster cannot be asked and the chain alone decides, which is what a serial run gets.

    Args:
        log: Logger the failure is named on.
        roi: ROI label, the field that attributes a fleet-wide error to one cell.
        date: The date being read.
        items: The catalogue items this read opened, named in the log line. Optical passes
            them; radar keeps no per-date item list.
        client: Cluster to collect GDAL's logged refusals from, if there is one.
    """
    try:
        yield
    except Exception as exc:
        # Before anything reads a verdict off this failure, including the line below.
        carry_logged_refusal(exc, client)
        ids = ", ".join(str(getattr(i, "id", "?")) for i in items[:4]) or "none"
        log.exception("READ FAILED roi=%s date=%s items=%d first=%s", roi, date, len(items), ids)
        if cause_was_flattened(exc):
            log.error(
                "READ CAUSE LOST roi=%s date=%s: the failure arrived without its cause, so nothing "
                "can say WHY the read failed and this leg's read verdicts are undecidable. The "
                "reading worker did not have the cause-preserving reducer installed.",
                roi,
                date,
            )
        raise


def apply_roi_mask(
    data: xr.Dataset,
    roi_zarr_path: str,
    spatial_chunks: dict[str, int],
    fill_value: int = 0,
    roi_mask: da.Array | None = None,
) -> xr.Dataset:
    """Apply a binary ROI mask to all variables in a dataset.

    Reads the Zarr ROI store, and sets pixels outside the ROI to
    ``fill_value`` for every data variable.

    Lazy by contract: builds the masking graph and returns. Both sensor paths call
    this once per date over a full zone grid, so nothing here may compute eagerly.
    Callers needing the ROI pixel total compute it themselves, once.

    Args:
        data: Dataset with (time, northing, easting) dimensions and an ``odc.geobox``.
        roi_zarr_path: Path to the Zarr ROI store (local or s3://).
        spatial_chunks: Dict with ``"northing"`` and ``"easting"`` chunk sizes for the
            broadcast dask array (should match the dataset's load chunks).
        fill_value: Value to assign outside the ROI. Default 0.
        roi_mask: Pre-computed boolean dask array (northing, easting), already
            persisted on workers or left lazy. When provided, avoids re-reading
            from the Zarr store.

    Returns:
        The masked dataset.
    """
    mask_2d = roi_mask if roi_mask is not None else read_roi_mask(roi_zarr_path, spatial_chunks)
    mask_da = mask_2d[np.newaxis, :, :]
    for var in data.data_vars:
        data[var] = data[var].where(mask_da, other=fill_value)
    return data


def filter_low_coverage_dates(
    data: xr.Dataset,
    roi_pixel_count: int,
    quality_band: str,
    invalid_values: Set[int],
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
) -> xr.Dataset:
    """Drop time steps where valid coverage within the ROI is too low.

    Expects the ROI mask to have already been applied (out-of-ROI pixels
    set to a value included in ``invalid_values``) so that the quality band
    captures both no-data and outside-ROI.

    Only the per-date valid-pixel fractions (one scalar per time step) are
    computed to decide which dates to keep; all band arrays stay lazy.

    Args:
        data: Dataset with (time, northing, easting) dims and the ROI mask already applied.
        roi_pixel_count: Number of pixels inside the ROI mask, used as the
            denominator for coverage percentage.
        quality_band: Name of the quality/classification variable to check
            (e.g. ``"scl"`` for S2, ``"0_VV"`` for S1).
        invalid_values: Set of values in the quality band considered invalid.
        min_valid_coverage: Minimum percentage of valid pixels to keep a
            date. Dates below this are dropped.

    Returns:
        Dataset with low-coverage dates removed. If all dates pass,
        the original (lazy) dataset is returned unchanged.
    """
    if quality_band not in data.data_vars:
        logger.warning(f"No '{quality_band}' band in dataset — skipping coverage filter")
        return data

    qb = data[quality_band]  # lazy (time, northing, easting)

    # isin reads qb once in a single fused pass. The previous OR-chain over
    # individual equality checks created a fan-out (one qb read per invalid
    # value) that Dask cannot fuse, causing repeated S3 fetches of the SCL band.
    valid_counts = (~qb.isin(list(invalid_values))).sum(dim=("northing", "easting"))  # (time,), lazy

    # Compute only the tiny per-date counts — band arrays stay lazy.
    valid_counts_np = valid_counts.compute()
    pcts = 100.0 * valid_counts_np.values / roi_pixel_count
    dates = [str(t.values)[:10] for t in data.time]
    keep_mask = pcts >= min_valid_coverage

    dropped = [d for d, keep in zip(dates, keep_mask, strict=True) if not keep]
    if dropped:
        logger.info(f"Dropping {len(dropped)} dates with <{min_valid_coverage}% valid coverage: {dropped}")
        for d, pct in zip(dates, pcts, strict=True):
            if pct < min_valid_coverage:
                logger.debug(f"  {d}: {pct:.1f}% valid")

    if not keep_mask.any():
        return data.isel(time=[])  # empty along time

    if keep_mask.all():
        return data  # nothing to drop

    return data.isel(time=keep_mask)


def identify_low_coverage_ds(
    data: xr.Dataset,
    roi_pixel_count: int,
    quality_band: str,
    invalid_values: Set[int],
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
) -> xr.Dataset:
    """Lazily zero out all bands when valid coverage is below the threshold.

    Unlike :func:`filter_low_coverage_dates` (which eagerly computes coverage
    to decide which timesteps to drop), this function keeps the entire Dask
    graph lazy by using ``xr.where`` to mask bands to ``fill_value`` when
    coverage is insufficient.  Best suited for single-timestep datasets where
    there is no need to shrink the time dimension.

    Args:
        data: Dataset with (time, northing, easting) dims and the ROI mask already applied.
        roi_pixel_count: Number of pixels inside the ROI mask, used as the
            denominator for coverage percentage.
        quality_band: Name of the quality/classification variable to check
            (e.g. ``"scl"`` for S2, ``"0_VV"`` for S1).
        invalid_values: Set of values in the quality band considered invalid.
        min_valid_coverage: Minimum percentage of valid pixels to keep a
            date. Dates below this are zeroed out.
        fill_value: Value to assign when coverage is too low. Default 0.

    Returns:
        Dataset with the same shape and a ``valid_coverage`` coordinate on
        the time dimension (lazy bool).  Downstream code can check
        ``ds["valid_coverage"].compute()`` — a single scalar — to decide
        whether to skip the timestep without reading any band data.
        If coverage is below the threshold, all bands are set to
        ``fill_value``; otherwise the data is unchanged.
        The entire operation remains lazy.
    """
    if quality_band not in data.data_vars:
        logger.warning(f"No '{quality_band}' band in dataset — skipping coverage filter")
        return data

    qb = data[quality_band]  # lazy (time, northing, easting)
    valid_count = (~qb.isin(list(invalid_values))).sum(dim=("northing", "easting"))  # lazy
    is_valid = (100.0 * valid_count / roi_pixel_count) >= min_valid_coverage  # lazy bool

    data = data.assign_coords(valid_coverage=("time", is_valid.data))

    return data
