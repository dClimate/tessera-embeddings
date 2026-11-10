"""Sensor-agnostic ROI processing: the per-date source-read retry, failure context and ROI mask."""

import logging
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import cast

import dask.array as da
import dask.distributed
import numpy as np
import xarray as xr
from tenacity import Retrying, before_sleep_log, stop_after_attempt, wait_exponential

from tessera_embeddings.ingest.duplicates import cause_was_flattened
from tessera_embeddings.ingest.loader_failures import carry_logged_refusal

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
) -> Generator[None]:
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


def apply_roi_mask(data: xr.Dataset, roi_mask: da.Array) -> xr.Dataset:
    """Set every pixel outside the ROI to 0, in every variable of ``data``.

    Lazy by contract: builds the masking graph and returns. The radar leg calls this once per
    date over a full zone grid, so nothing here may compute eagerly.

    Args:
        data: Dataset with (time, northing, easting) dimensions.
        roi_mask: Boolean dask array (northing, easting), read once by the caller with its
            credentials and either persisted on workers or left lazy.

    Returns:
        The masked dataset.
    """
    mask_da = roi_mask[np.newaxis, :, :]
    for var in data.data_vars:
        data[var] = data[var].where(mask_da, other=0)
    return data
