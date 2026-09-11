"""Sensor-agnostic ROI processing: mask application and coverage filtering.

These operations work on any xarray/Dask Dataset with spatial dimensions (northing, easting)
and a quality band. They are shared by S2 and S1 ROI ingestion flows.
"""

import logging
import time
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
#: Sized to outlast a provider having a bad minute, not just a dropped packet: three attempts on
#: the ladder below spend ~6 s, which is nothing against a throttling source, and a failure at
#: this layer is what drops a date — the depth of this number is the difference between waiting
#: out a refusal and recording recoverable imagery as lost. Cost falls entirely on the failing
#: path; a dead object pays the full ~61 s ladder once and then fails the leg, which is intended
#: because the leg resumes from its committed dates. Not raised further because it MULTIPLIES
#: with the per-date alternate-copy step-down and the leg's own attempts, and that product is
#: what a struggling provider sees.
SOURCE_READ_ATTEMPTS = 8


def source_read_retrying(log: logging.Logger | logging.LoggerAdapter[logging.Logger]) -> Retrying:
    """Retry the source read for one date.

    Without this, a transient failure on a single granule propagates out of the per-date loop
    and fails the whole zone-year, abandoning every month already committed.

    Scoped to ONE date deliberately: a task-level retry would re-run the entire multi-day loop,
    which is why the ingest task refuses ``@task(retries=...)``. Not narrowed by exception type
    either — reads fail through rasterio, GDAL/CPL, botocore and plain socket timeouts, and
    enumerating those is how a new transient class silently becomes fatal.

    Reads are idempotent, so retrying a permanent failure is safe but not free: the ladder is
    :data:`SOURCE_READ_ATTEMPTS` attempts over seven exponential sleeps, ~61 s of backoff per
    permanently-failing date on top of each attempt's read time.
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

    Per-date telemetry is emitted *after* a date commits, so the last date in a log is the last
    date that WORKED and a failed date leaves no trace of having been attempted. The ROI label
    matters as much: the failure is raised on a Dask worker whose message carries no zone, so at
    fleet width the same error text appears for every zone at once.

    rasterio compounds both — it reports ``Read failed. See previous exception for details.`` and
    that previous exception is GDAL's, discarded unless the chain is logged, which is what
    ``log.exception`` does here.

    A chain that never crossed the worker boundary cannot be logged at all; the second line fires
    only for a failure that arrived already flattened, and is the only warning that a verdict is
    about to be taken with no evidence under it. Anything it reports is a worker
    ``ingest/loader_failures.py`` did not reach.

    ``client`` completes an INCOMPLETE reason: GDAL states some refusals only in its own log and
    raises the codec failure that follows, so the chain can say the bytes are bad while the words
    saying the service refused sit in a log line. Both sensors read through here, so evidence is
    collected once, on either path. Without a client the chain alone decides, which is what a
    serial run gets. Emitted only on failure, so the success path pays nothing.

    Args:
        log: Logger the failure is named on.
        roi: ROI label, the field that attributes a fleet-wide error to one cell.
        date: The date being read.
        items: The catalogue items this read opened, named in the log line. Optical passes
            them; radar keeps no per-date item list.
        client: Cluster to collect GDAL's logged refusals from, if there is one.
    """
    entered = time.monotonic()
    try:
        yield
    except Exception as exc:
        # Before anything reads a verdict off this failure, including the line below. Bounded by
        # how long THIS read has been running: a line older than that was logged before the read
        # began, and belongs to a read that has since succeeded or been judged already.
        carry_logged_refusal(exc, client, since=entered)
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
    """Apply a binary ROI mask to all variables in a dataset, setting outside pixels to
    ``fill_value``.

    Lazy by contract: builds the masking graph and returns. Both sensor paths call this once per
    date over a full zone grid, so nothing here may compute eagerly — callers needing the ROI
    pixel total compute it themselves, once.

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

    Expects the ROI mask already applied — out-of-ROI pixels set to a value in
    ``invalid_values`` — so the quality band captures both no-data and outside-ROI. Only the
    per-date valid-pixel fractions, one scalar per time step, are computed; bands stay lazy.

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

    # isin reads qb once in a single fused pass. An OR-chain of equality checks instead fans out
    # to one qb read per invalid value, which Dask cannot fuse — repeated S3 fetches of SCL.
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
    """Lazily flag time steps whose valid coverage within the ROI is below the threshold.

    Unlike :func:`filter_low_coverage_dates`, which eagerly computes coverage to decide which
    timesteps to drop, this keeps the whole Dask graph lazy and attaches the verdict as a
    coordinate. Suited to single-timestep datasets, where shrinking the time dimension buys
    nothing.

    Args:
        data: Dataset with (time, northing, easting) dims and the ROI mask already applied.
        roi_pixel_count: Number of pixels inside the ROI mask, used as the
            denominator for coverage percentage.
        quality_band: Name of the quality/classification variable to check
            (e.g. ``"scl"`` for S2, ``"0_VV"`` for S1).
        invalid_values: Set of values in the quality band considered invalid.
        min_valid_coverage: Minimum percentage of valid pixels for a date to be flagged valid.

    Returns:
        The same dataset, bands untouched, with a lazy boolean ``valid_coverage`` coordinate on
        the time dimension. Downstream code computes that one scalar to decide whether to skip
        the timestep, without reading any band data.
    """
    if quality_band not in data.data_vars:
        logger.warning(f"No '{quality_band}' band in dataset — skipping coverage filter")
        return data

    qb = data[quality_band]  # lazy (time, northing, easting)
    valid_count = (~qb.isin(list(invalid_values))).sum(dim=("northing", "easting"))  # lazy
    is_valid = (100.0 * valid_count / roi_pixel_count) >= min_valid_coverage  # lazy bool

    data = data.assign_coords(valid_coverage=("time", is_valid.data))

    return data
