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
    :class:`DuplicateDateError`: a duplicate means another writer moved the branch, while this
    means one writer offered its own dates out of order, and only the second is a bug in the
    caller.

    It matters because the time axis is read POSITIONALLY downstream — the deterministic
    resampler selects observations by position, not by timestamp — so a store whose axis is
    out of order yields different embeddings from a chronologically-ingested store holding the
    same dates. Nothing downstream would notice; the arrays are all valid and all the right
    shape. Refusing the append is what makes the ordering assumption enforceable rather than
    merely documented.
    """


class DuplicateDateError(ValueError):
    """Raised when a date being appended is already on a store's time axis.

    A ``ValueError`` subclass because that is what the guard raised before it had a
    type of its own, and every caller that catches ``ValueError`` must keep working.
    The type exists so a retry can single this failure out: almost every other write
    failure is worth retrying and this one never is, because the ingest paths dedupe
    against the store before writing — so a duplicate reaching the append means
    something outside this process moved the branch. See
    :data:`~tessera_embeddings.storage.zarr_store.CONCURRENT_WRITER_ERRORS`.
    """


class InconclusiveStoreProbeError(Exception):
    """The emptiness probe on the create path could not answer.

    Exempt from ``cleanup_on_failure``'s delete for the same reason
    :class:`StoreHoldsCommittedDataError` is, and by the stronger argument: that one says
    the store is NOT ours, this one says we do not know. The probe reads the network, so a
    transient failure — or a decode error while inspecting a repo another writer is
    creating — is ordinary. Treating "could not tell" as "safe to delete" is what turns a
    blip into the erasure of somebody else's committed store, so deletion happens only on
    POSITIVE evidence that the prefix is ours.
    """


class StoreHoldsCommittedDataError(Exception):
    """Raised when the create path is handed a store that already holds committed data.

    Exempt from ``cleanup_on_failure``'s delete, and that exemption is the whole point
    of having a type here. The decorator removes the half-written store a failed create
    leaves behind, which is right for every other failure on that path. This one means
    the opposite: the store was NOT written by this attempt, it was already there and
    intact, and the create was reached because a date probe misreported it as empty.
    Deleting it would turn a refusal to overwrite data into the deletion of that same
    data — the guard destroying what it exists to protect.
    """


class ProviderRefusedReadsError(RuntimeError):
    """Raised when the source provider refused a read for longer than one write may wait.

    NOTHING is lost and NOTHING is skipped when this is raised — the store's time axis is
    unmoved and a re-dispatch resumes from the dates already committed. It exists to make the
    verdict visible to the layer that decides WHEN to re-dispatch, which is the only layer that
    can wait on a timescale longer than a leg.

    Named rather than absorbed because the two waits cost different things. Waiting inside a leg
    holds that leg's Dask fleet idle; failing the leg releases the fleet, so the cell's own retry
    can wait far longer for almost nothing. A refusal is the one failure class where the cheap
    wait is the right one, and a leg-failure message that does not say so cannot be told apart
    from a crash — which is why this is a TYPE rather than a log line.

    Deliberately absent from the leg classifier's non-retryable set: waiting is the remedy for
    this class, so a re-dispatch is exactly what it is asking for.
    """
