"""Custom error classes for the Zarr pipeline."""


class CorruptedStoreError(Exception):
    """Raised when a Zarr store path contains corrupted or incomplete data."""


class InsufficientCoverageError(Exception):
    """Raised when a mosaic store lacks data to cover the requested time window."""


class ConfigMismatchError(Exception):
    """Raised when a store's structural parameters don't match the current config."""


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
