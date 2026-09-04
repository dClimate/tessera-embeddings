"""Custom error classes for the Zarr pipeline."""


class CorruptedStoreError(Exception):
    """Raised when a Zarr store path contains corrupted or incomplete data."""


class InsufficientCoverageError(Exception):
    """Raised when a mosaic store lacks data to cover the requested time window."""


class ConfigMismatchError(Exception):
    """Raised when a store's structural parameters don't match the current config."""


class NonMonotonicDateError(ValueError):
    """Raised when a date being appended is OLDER than a date already on the time axis.

    Deliberately not a :data:`CONCURRENT_WRITER_ERRORS` member and not a sibling of
    :class:`~tessera_embeddings.storage.zarr_store.DuplicateDateError`: a duplicate means
    another writer moved the branch, while this means one writer offered its own dates out of
    order — a bug in the caller.

    The time axis is read POSITIONALLY downstream (the deterministic resampler selects
    observations by position, not by timestamp), so an out-of-order axis yields different
    embeddings from a chronologically-ingested store holding the same dates, and nothing
    downstream would notice: the arrays are all valid and all the right shape.
    """


class ProviderRefusedReadsError(RuntimeError):
    """Raised when the source provider refused a read for longer than one write may wait.

    NOTHING is lost or skipped: the store's time axis is unmoved and a re-dispatch resumes from
    the dates already committed. The type makes the verdict visible to the layer that decides
    WHEN to re-dispatch, the only layer that can wait on a timescale longer than a leg.

    Named rather than absorbed because the two waits cost different things: waiting inside a leg
    holds that leg's Dask fleet idle, while failing the leg releases the fleet so the cell's own
    retry can wait far longer for almost nothing. A leg-failure message alone cannot be told
    apart from a crash, hence a TYPE rather than a log line.

    Deliberately absent from the leg classifier's non-retryable set: waiting is the remedy, so a
    re-dispatch is exactly what it is asking for.
    """
