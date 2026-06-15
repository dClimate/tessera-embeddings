"""Zarr store management utilities for reflectance and cloudmask data.

This module provides functions for creating, reading, and writing Zarr stores
that hold preprocessed Sentinel-2 reflectance data and cloud masks.

Uses Icechunk for transactional writes with atomic commit semantics.

Three write paths, all committing atomically:

- **create** (:func:`write_dataset` on a fresh store) — ``to_icechunk`` mode
  ``"w"`` writes the whole array.
- **append** (:func:`write_dataset` on an existing store) — mode ``"a"`` with
  ``append_dim="time"`` extends the time axis.
- **region overwrite** (:func:`write_region`) — mode ``"r+"`` rewrites an
  existing temporal/spatial slice in place. The region need not be
  chunk-aligned: :func:`_pad_region_to_chunks` widens unaligned edges to chunk
  boundaries and backfills the untouched cells from the store
  (read-modify-write), because ``align_chunks=True`` only remaps the producer's
  dask blocks and ``mode="r+"`` rejects partial-chunk writes. Use
  :func:`resolve_region` to turn coordinate ranges into the integer slices
  ``write_region`` expects.
"""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import fsspec
import icechunk
import numpy as np
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk

from tessera_embeddings.errors import CorruptedStoreError
from tessera_embeddings.storage.manifest import IngestManifest, extract_manifest
from tessera_embeddings.utils import utcnow_iso

logger = logging.getLogger(__name__)

# Standard time encoding for all stores
TIME_ENCODING = {"units": "nanoseconds since 1970-01-01", "calendar": "proleptic_gregorian"}


# =============================================================================
# Store Cleanup Utilities
# =============================================================================


def _delete_store(store_path: str) -> bool:
    """Delete a store at the given path. Returns True if deleted, False if not found."""
    try:
        fs = fsspec.filesystem(fsspec.utils.get_protocol(store_path))
        if fs.exists(store_path):
            fs.rm(store_path, recursive=True)
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to delete store {store_path}: {e}")
        return False


def cleanup_on_failure[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that deletes the store at store_path (first arg) if the function fails.

    Use on store creation functions to prevent leaving corrupted/partial stores.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        store_path: str = args[0] if args else kwargs.get("store_path")  # type: ignore[assignment]
        if not store_path:
            raise ValueError("cleanup_on_failure requires store_path as first argument")
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.warning(f"Store creation failed, cleaning up {store_path}")
            _delete_store(store_path)
            raise

    return wrapper


# =============================================================================
# Core Repository Helpers
# =============================================================================


def _parse_s3_url(url: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    path = url[5:]
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


# NOTE: Hardcoding a default region is normally an anti-pattern in this
# package — paths and credentials are caller-supplied so the code works for
# any deployment. We make a deliberate exception here: the Sentinel-1 OPERA
# RTC archive lives in us-west-2 and authentication to it only succeeds
# from in-region clients (cross-region requests are rejected, not just
# charged for). Any pipeline running this code against OPERA must therefore
# be in us-west-2, so defaulting the region here keeps that path zero-config
# while still letting callers override via the ``region`` argument.
_DEFAULT_S3_REGION = "us-west-2"


# Process-wide fallback credential provider for icechunk S3 storage. When set,
# _create_storage uses it for any S3 open that didn't receive an explicit
# get_credentials. The S1 ingest path registers an IAM-resolving callback here
# (via the AWS provider) so icechunk writes keep using IAM-role creds even
# after set_s3_credentials overwrites the AWS_* env vars with OPERA-scoped STS
# tokens. Stays None in the open-source layer, keeping it cloud-agnostic.
_default_credentials_provider: "Callable[[], icechunk.S3StaticCredentials] | None" = None


@contextmanager
def credentials_provider(
    provider: "Callable[[], icechunk.S3StaticCredentials] | None",
) -> "Iterator[None]":
    """Temporarily register a fallback credential provider for icechunk S3 opens.

    Used by :func:`_create_storage` whenever an S3 open inside the ``with``
    block has no explicit ``get_credentials``. Scoped to the block so a
    reused process (e.g. a Dask worker) is not left pinned to ``provider``
    for later, unrelated icechunk opens. The previous provider is restored
    even if the body raises. See the module-level
    ``_default_credentials_provider`` note for why the S1 ingest path needs
    this.
    """
    global _default_credentials_provider
    previous = _default_credentials_provider
    _default_credentials_provider = provider
    try:
        yield
    finally:
        _default_credentials_provider = previous


def _create_storage(
    store_path: str,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    scatter_initial_credentials: bool = False,
) -> icechunk.Storage:
    """Create Icechunk storage for local or S3 paths.

    Args:
        store_path: Local path or S3 URI.
        get_credentials: Optional credential callback for Icechunk's S3
            client. The callable returns a fresh
            :class:`icechunk.S3StaticCredentials` on each invocation; setting
            ``expires_after`` on the returned object tells Icechunk when to
            call back for refresh. When ``None`` on an S3 path, Icechunk
            falls back to the default AWS credential chain (env, instance
            profile, etc.), which is suitable for local testing with moto.
            The AWS provider in the closed-source repo supplies a
            botocore-backed callback that adapts boto's
            ``RefreshableCredentials`` into ``S3StaticCredentials``.
        region: Optional S3 region. Defaults to ``us-west-2`` because the
            Sentinel-1 OPERA RTC archive only authenticates in-region.
            Override for stores outside us-west-2.
        scatter_initial_credentials: When True, Icechunk eagerly calls
            ``get_credentials`` once and caches the result so that pickled
            copies of the storage (e.g. shipped to Ray actors or Dask workers
            during ``to_icechunk``) don't all stampede the credential
            provider on deserialisation. Set True for distributed assembly.
    """
    if store_path.startswith("s3://"):
        bucket, prefix = _parse_s3_url(store_path)
        if _s3_config_override:
            return _s3_config_override.make_storage(prefix_override=prefix)
        # Fall back to a globally-registered credential provider when the
        # caller didn't pass one explicitly. This is how the S1 ingest path
        # keeps icechunk on IAM-role creds: set_s3_credentials overwrites the
        # AWS_* env vars with OPERA-scoped STS tokens for GDAL reads, and
        # icechunk's default AWS chain would otherwise pick those up and get
        # AccessDenied writing our own store. The S1 ingest task registers an
        # IAM-resolving callback via credentials_provider(). See
        # tessera_embeddings.providers.aws.credentials.
        if get_credentials is None:
            get_credentials = _default_credentials_provider
        s3_kwargs: dict = {
            "bucket": bucket,
            "prefix": prefix,
            "region": region if region is not None else _DEFAULT_S3_REGION,
        }
        if get_credentials is not None:
            s3_kwargs["get_credentials"] = get_credentials
            s3_kwargs["scatter_initial_credentials"] = scatter_initial_credentials
        return icechunk.s3_storage(**s3_kwargs)
    # icechunk.local_filesystem_storage takes a plain path, not a URI.
    # Strip the file:// scheme that fsspec-style configs commonly use.
    local_path = store_path[7:] if store_path.startswith("file://") else store_path
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    return icechunk.local_filesystem_storage(local_path)


# Per-repo chunk cache budget. Inference reads each (time=1, 4000, 4000) store
# chunk once per spatial strip; when a dense chunk is split into northing strips
# the same on-disk chunk is touched by every strip, so an LRU chunk cache turns
# the per-strip re-reads into cache hits. 512 MB holds enough decoded chunks to
# keep the measured strip read penalty at ~1.35x (vs 2.37x uncached), which
# stays under the per-chunk compute envelope and is hidden by the strip prefetch
# pipeline. Applied to every repo open — the cache is bounded and harmless for
# write paths that never re-read.
_CHUNK_CACHE_BYTES = 512 * 1024**2


def _default_repo_config(max_concurrent_requests: int | None = None) -> icechunk.RepositoryConfig:
    """Build the RepositoryConfig overrides applied to every repo open.

    Always sets a bounded chunk cache (see ``_CHUNK_CACHE_BYTES``) so striped
    inference reads hit cache instead of re-fetching shared store chunks.

    When ``max_concurrent_requests`` is provided, also caps per-repo HTTP
    concurrency. Assembly at cornbelt scale fans out thousands of concurrent
    PUTs to one zarr prefix, blowing past S3's ~3.5K/s per-prefix limit and
    triggering 503 SlowDown. Icechunk's default is 256 concurrent HTTP
    requests per repo; callers that fan out across many workers should set
    this lower (e.g. 64) so aggregate request rate stays under S3's ceiling.
    Retry/backoff is left at icechunk's defaults — SlowDown is already
    classified as retriable by the underlying AWS SDK.
    """
    config = icechunk.RepositoryConfig.default()
    config.caching = icechunk.CachingConfig(num_bytes_chunks=_CHUNK_CACHE_BYTES)
    if max_concurrent_requests is not None:
        config.max_concurrent_requests = max_concurrent_requests
    return config


def _open_repo(
    store_path: str,
    max_concurrent_requests: int | None = None,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    scatter_initial_credentials: bool = False,
) -> icechunk.Repository:
    """Open an existing Icechunk repository."""
    return icechunk.Repository.open(
        _create_storage(
            store_path,
            get_credentials=get_credentials,
            region=region,
            scatter_initial_credentials=scatter_initial_credentials,
        ),
        config=_default_repo_config(max_concurrent_requests),
    )


def _create_repo(
    store_path: str,
    max_concurrent_requests: int | None = None,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    scatter_initial_credentials: bool = False,
) -> icechunk.Repository:
    """Create a new Icechunk repository."""
    try:
        return icechunk.Repository.create(
            _create_storage(
                store_path,
                get_credentials=get_credentials,
                region=region,
                scatter_initial_credentials=scatter_initial_credentials,
            ),
            config=_default_repo_config(max_concurrent_requests),
        )
    except icechunk.IcechunkError as e:
        if "repositories can only be created in clean prefixes" in str(e):
            raise CorruptedStoreError(
                f"Store {store_path} appears corrupted. Delete it or use a different path."
            ) from e
        raise


def open_or_create_repo(
    store_path: str,
    max_concurrent_requests: int | None = None,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    scatter_initial_credentials: bool = False,
) -> tuple[icechunk.Repository, bool]:
    """Open existing or create new Icechunk repository.

    Args:
        store_path: Local path or S3 URI.
        max_concurrent_requests: Optional cap on concurrent S3 requests per repo.
        get_credentials: Optional credential callback for Icechunk's S3 client.
            See :func:`_create_storage` for details.
        region: Optional S3 region override.
        scatter_initial_credentials: When True with a distributed writer,
            cache the initial credentials so pickled storage copies don't
            re-invoke the callback on each worker. See :func:`_create_storage`.

    Returns:
        Tuple of ``(repository, is_new)``. ``is_new`` is True if the repo was
        just created.
    """
    try:
        return _open_repo(
            store_path,
            max_concurrent_requests,
            get_credentials=get_credentials,
            region=region,
            scatter_initial_credentials=scatter_initial_credentials,
        ), False
    except (FileNotFoundError, icechunk.IcechunkError):
        return _create_repo(
            store_path,
            max_concurrent_requests,
            get_credentials=get_credentials,
            region=region,
            scatter_initial_credentials=scatter_initial_credentials,
        ), True


# =============================================================================
# Session Helpers (used by all read/write operations)
# =============================================================================


def _open_readonly(store_path: str) -> xr.Dataset:
    """Open store for reading. Returns xarray Dataset."""
    repo = _open_repo(store_path)
    session = repo.readonly_session(branch="main")
    return xr.open_zarr(session.store, consolidated=False)


@cleanup_on_failure
def _write_new(
    store_path: str,
    data: xr.Dataset,
    encoding: dict[str, Any] | None,
    message: str,
) -> None:
    """Create a new store with data. Cleans up on failure."""
    repo = _create_repo(store_path)
    session = repo.writable_session("main")
    to_icechunk(data, session, mode="w", encoding=encoding, align_chunks=True)
    session.commit(message)
    logger.info(f"Created {store_path}")


def _commit_preserving_attrs(
    session: "icechunk.Session",
    write: "Callable[[icechunk.Session], None]",
    message: str,
    update_attrs: dict[str, Any] | None = None,
) -> None:
    """Run a ``to_icechunk`` write, then commit with root attrs preserved.

    ``to_icechunk`` overwrites root attrs with the (typically empty)
    ``data.attrs`` — destroying crs, ``_manifest``, etc. We snapshot them
    before the write and restore (plus any ``update_attrs``) after, then commit.
    Shared by the append and region-write paths.
    """
    root = zarr.open_group(session.store, mode="r")
    preserved_attrs = dict(root.attrs)

    write(session)

    root = zarr.open_group(session.store, mode="r+")
    root.attrs.update(preserved_attrs)
    if update_attrs:
        root.attrs.update(update_attrs)
    session.commit(message)


def _write_append(
    store_path: str,
    data: xr.Dataset,
    message: str,
    update_attrs: dict[str, Any] | None = None,
) -> None:
    """Append data to existing store."""
    repo = _open_repo(store_path)
    session = repo.writable_session("main")

    _commit_preserving_attrs(
        session,
        lambda s: to_icechunk(data, s, mode="a", append_dim="time", align_chunks=True),
        message,
        update_attrs,
    )
    logger.info(f"Appended to {store_path}")


# =============================================================================
# Region Writes (overwrite-in-place of a temporal/spatial slice)
# =============================================================================


def _store_chunk_sizes(existing: xr.Dataset, dims: "tuple[str, ...]") -> dict[str, int]:
    """Read the on-disk chunk size per dimension from the existing store.

    The store is authoritative for chunking (config may have drifted), so we
    read it off the opened arrays rather than INGEST_CHUNKS. All data_vars in
    our stores share identical chunking on the spatial/temporal dims; we assert
    that here so the single widened region we compute is valid for every var.
    """
    sizes: dict[str, int] = {}
    for dim in dims:
        per_var = set()
        for var in existing.data_vars:
            chunks = existing[var].chunks
            if chunks is None:
                continue
            axis = existing[var].dims.index(dim)
            per_var.add(chunks[axis][0])  # first block = nominal chunk size
        if len(per_var) > 1:
            raise ValueError(
                f"Store data_vars disagree on chunk size for {dim!r}: {per_var}. "
                "Region writes require uniform chunking across variables."
            )
        if per_var:
            sizes[dim] = per_var.pop()
    return sizes


def _pad_region_to_chunks(
    existing: xr.Dataset,
    data: xr.Dataset,
    region: dict[str, slice],
) -> "tuple[xr.Dataset, dict[str, slice]]":
    """Widen ``region`` to whole-chunk bounds, backfilling the shell from the store.

    Icechunk's ``mode="r+"`` rejects a region whose edges fall mid-chunk
    (xarray ``allow_partial_chunks=False``), and ``align_chunks=True`` only
    remaps the *producer's* dask blocks — it never reads neighbouring on-disk
    data. So an unaligned region must be padded out to its enclosing chunk
    boundaries, with the newly-included-but-unchanged cells read back from the
    store and the incoming values overlaid on top.

    Returns ``(padded, widened)``. When ``region`` is already chunk-aligned this
    returns ``(data, region)`` unchanged — no store read.

    Only boundary chunks are actually fetched: the shell is read as a lazy
    dask slab on the store's chunk grid, then a positional ``setitem`` overlays
    the incoming data. Interior chunks are wholly overwritten, so dask culls
    their read tasks; only the partially-overwritten edge chunks are read.
    """
    chunk_sizes = _store_chunk_sizes(existing, tuple(region))

    widened: dict[str, slice] = {}
    needs_pad = False
    for dim, sl in region.items():
        cs = chunk_sizes[dim]
        size = existing.sizes[dim]
        start, stop, _ = sl.indices(size)
        new_start = (start // cs) * cs
        new_stop = min(((stop + cs - 1) // cs) * cs, size)
        widened[dim] = slice(new_start, new_stop)
        if new_start != start or new_stop != stop:
            needs_pad = True

    if not needs_pad:
        return data, region

    # Lazy shell from the store on its own chunk grid (committed data).
    slab = existing.isel(widened)

    padded_vars: dict[str, Any] = {}
    for var in data.data_vars:
        name = str(var)
        shell = slab[name].data.copy()  # dask array on store chunk grid
        idx = tuple(
            slice(region[dim].start - widened[dim].start, region[dim].stop - widened[dim].start)
            if dim in region
            else slice(None)
            for dim in data[name].dims
        )
        shell[idx] = data[name].data  # positional overlay; incoming wins
        padded_vars[name] = (data[name].dims, shell)

    # Coords come from the store slab (authoritative); they're dropped before
    # the region write anyway, but carrying them keeps the dataset well-formed.
    padded = xr.Dataset(padded_vars, coords=slab.coords)
    return padded, widened


def _drop_region_coords(data: xr.Dataset, region_dims: "set[str]") -> xr.Dataset:
    """Drop coords xarray won't accept on a region write.

    xarray rejects writing a region-dim's own coordinate (the store's coords are
    authoritative), and a region write requires every written variable to share
    a dim with the region — so coords spanning only non-region dims must go too.
    """
    drop = [name for name in data.coords if name in region_dims or not region_dims.intersection(data[name].dims)]
    return data.drop_vars(drop)


def _write_region(
    store_path: str,
    data: xr.Dataset,
    region: dict[str, slice],
    message: str,
    update_attrs: dict[str, Any] | None = None,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region_name: str | None = None,
) -> None:
    """Overwrite an existing region of a store in a single atomic commit."""
    repo = _open_repo(store_path, get_credentials=get_credentials, region=region_name)
    session = repo.writable_session("main")

    # Committed view of the store, used to pad unaligned regions. Nothing has
    # been written in this session yet, so this reflects committed data.
    existing = xr.open_zarr(session.store, consolidated=False)
    padded, widened = _pad_region_to_chunks(existing, data, region)
    to_write = _drop_region_coords(padded, set(widened))

    _commit_preserving_attrs(
        session,
        lambda s: to_icechunk(to_write, s, mode="r+", region=widened, align_chunks=True, split_every=8),
        message,
        update_attrs,
    )
    logger.info(f"Wrote region {widened} to {store_path}")


# =============================================================================
# Public API
# =============================================================================


def open_store(store_path: str) -> xr.Dataset:
    """Open an Icechunk store for reading as an xarray Dataset."""
    return _open_readonly(store_path)


def open_store_as_zarr_group(store_path: str) -> zarr.Group:
    """Open an Icechunk store for reading as a raw zarr Group.

    Bypasses xarray/dask entirely. Use when you need to read large arrays into
    numpy without the dask task-graph overhead — e.g., loading a chunk of
    reflectance bands for inference, where each ``.values`` call on the xarray
    path would build and execute a fresh dask graph per variable and hold
    scheduler state until the dataset handle dies.
    """
    repo = _open_repo(store_path)
    session = repo.readonly_session(branch="main")
    return zarr.open_group(session.store, mode="r")


def get_existing_dates(store_path: str) -> set[str]:
    """Get dates already present in a store. Returns empty set if store doesn't exist."""
    t0 = time.monotonic()
    logger.debug(f"Opening store: {store_path}")
    try:
        ds = _open_readonly(store_path)
        dates = {str(t.values)[:10] for t in ds.time}
        ds.close()
        logger.debug(f"Store has {len(dates)} existing dates ({time.monotonic() - t0:.1f}s)")
        return dates
    except (FileNotFoundError, icechunk.IcechunkError):
        logger.debug(f"Store not found ({time.monotonic() - t0:.1f}s): {store_path}")
        return set()
    except Exception as e:
        logger.warning(f"Could not read dates from {store_path}: {e}")
        return set()


def resolve_region(
    store_path: str,
    *,
    time: "tuple[Any, Any] | None" = None,
    northing: "tuple[float, float] | None" = None,
    easting: "tuple[float, float] | None" = None,
) -> dict[str, slice]:
    """Map coordinate-value ranges to half-open integer slices for an existing store.

    Each argument is an inclusive ``(low, high)`` coordinate range or ``None``
    (= the full axis, omitted from the result). ``time`` accepts anything
    ``np.datetime64`` understands. Spatial bounds are matched against the
    store's ``northing``/``easting`` coordinate values regardless of axis
    direction (northing typically descends).

    Returns a dict of integer slices suitable for :func:`write_region`. Only
    dims with a non-``None`` range are included; an empty dict means "the whole
    array" (which the caller should treat as a plain overwrite, not a region).

    Enforces the overwrite-in-place contract: the requested range must select at
    least one existing coordinate. A range that matches nothing (e.g. a date not
    in the store) raises ``ValueError`` — appending new coordinates is
    :func:`write_dataset`'s job, not a region write's.
    """
    ds = _open_readonly(store_path)
    try:
        ranges = {"time": time, "northing": northing, "easting": easting}
        region: dict[str, slice] = {}
        for dim, rng in ranges.items():
            if rng is None:
                continue
            coord = ds[dim].values
            raw_low, raw_high = rng
            if dim == "time":
                low: Any = np.datetime64(str(raw_low), "ns")
                high: Any = np.datetime64(str(raw_high), "ns")
            else:
                low, high = raw_low, raw_high
            lo_b, hi_b = (low, high) if low <= high else (high, low)
            mask = (coord >= lo_b) & (coord <= hi_b)
            hits = np.flatnonzero(mask)
            if hits.size == 0:
                raise ValueError(
                    f"Region {dim}={rng!r} selects no existing coordinate in {store_path}. "
                    "Region writes overwrite in place; use write_dataset to add new coordinates."
                )
            region[dim] = slice(int(hits[0]), int(hits[-1]) + 1)
        return region
    finally:
        ds.close()


def write_region(
    store_path: str,
    data: xr.Dataset,
    *,
    region: dict[str, slice],
    update_attrs: dict[str, Any] | None = None,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
) -> None:
    """Overwrite a region of an existing store in place, in one atomic commit.

    ``region`` maps dimension names (``time``/``northing``/``easting``) to
    integer slices — typically from :func:`resolve_region`. Dimensions absent
    from ``region`` are written in full. ``data`` must already cover exactly the
    cells the region selects (same shape along each region dim); its own
    coordinate values are ignored (the store's coords are authoritative) and are
    dropped before writing.

    The region need not be chunk-aligned: unaligned edges are padded out to the
    enclosing chunk boundaries and the untouched cells backfilled from the store
    (read-modify-write), so the caller never has to reason about chunk
    boundaries. ``update_attrs`` is merged into root attrs after the write
    (root attrs are otherwise preserved across the write).

    Contract: every region dim must already exist in the store at the given
    indices. This overwrites committed data — see the region-writes design doc
    for the consistency caveats when a single logical region is split across
    multiple calls.
    """
    if not region:
        raise ValueError("write_region requires a non-empty region; use write_dataset for full writes.")
    _write_region(
        store_path,
        data,
        region,
        message=f"Overwrite region {region}",
        update_attrs=update_attrs,
        get_credentials=get_credentials,
        region_name=s3_region,
    )


def compute_doy(timestamps: np.ndarray) -> np.ndarray:
    """Compute day-of-year from datetime64 timestamps.

    Returns (N,) array of int32 DOY values (1-366).
    """
    years = timestamps.astype("datetime64[Y]")
    return ((timestamps.astype("datetime64[D]") - years).astype(int) + 1).astype(np.int32)


def write_dataset(
    store_path: str,
    data: xr.Dataset,
    tile_id: str,
    baselines: dict[str, int],
    chunks: dict[str, int],
    manifest: IngestManifest | None = None,
    *,
    crs: str,
) -> None:
    """Write dataset to Icechunk Zarr store, creating or appending as needed.

    Args:
        store_path: Path to the Zarr store (local or s3://).
        data: xarray Dataset with (time, northing, easting) dimensions.
        tile_id: Tile identifier for store metadata.
        baselines: Dict mapping date strings to baseline integers.
        chunks: Chunk sizes dict with ``time``, ``northing``, and ``easting`` keys.
        manifest: Typed manifest for append-safety validation.
            Written on create, validated on append.
        crs: CRS authority code (e.g. ``"EPSG:32615"``). Stored in root
            attrs so downstream consumers can determine the projection.
    """
    existing_dates = get_existing_dates(store_path)

    # Normalize time to nanosecond resolution to match TIME_ENCODING.
    # Newer pandas/xarray versions may produce datetime64[us]; coerce
    # so the zarr encoding round-trips correctly.
    if data.time.dtype != np.dtype("datetime64[ns]"):
        data = data.assign_coords(time=data.time.values.astype("datetime64[ns]"))

    doy = compute_doy(data.time.values)

    if existing_dates:
        # Single store open: read manifest + baseline attrs together
        existing_ds = _open_readonly(store_path)
        if manifest:
            manifest.validate_against(extract_manifest(existing_ds.attrs), store_path)
        merged_baselines = dict(existing_ds.attrs.get("baselines_applied", {}))
        merged_baselines.update(baselines)
        existing_doy = list(existing_ds.attrs.get("doy", []))
        existing_ds.close()

        _write_append(
            store_path,
            data,
            message=f"Append {data.sizes['time']} dates",
            update_attrs={
                "baselines_applied": merged_baselines,
                "doy": existing_doy + doy.tolist(),
                "last_appended": utcnow_iso(),
            },
        )
    else:
        chunk_sizes = (
            chunks["time"],
            min(chunks["northing"], data.sizes["northing"]),
            min(chunks["easting"], data.sizes["easting"]),
        )
        encoding: dict[str, Any] = {str(var): {"chunks": chunk_sizes} for var in data.data_vars}
        encoding["time"] = TIME_ENCODING

        store_attrs: dict[str, Any] = {
            "tile_id": tile_id,
            "baselines_applied": baselines,
            "doy": doy.tolist(),
            "created_at": utcnow_iso(),
            "last_appended": utcnow_iso(),
            "crs": crs,
        }
        if manifest:
            store_attrs["_manifest"] = manifest.to_dict()
            logger.info("Writing _manifest to %s", store_path)

        data.attrs.update(store_attrs)
        _write_new(store_path, data, encoding, f"Create with {data.sizes['time']} dates")


# =============================================================================
# Test Helpers (for verifying Icechunk transactional behavior)
# =============================================================================


def _open_writable_session(store_path: str) -> tuple[icechunk.Session, icechunk.IcechunkStore]:
    """Open a writable session. Caller must commit."""
    repo = _open_repo(store_path)
    session = repo.writable_session("main")
    return session, session.store


# =============================================================================
# S3 Configuration (for testing with moto)
# =============================================================================


@dataclass
class S3Config:
    """S3 configuration for testing with moto or custom endpoints."""

    bucket: str
    prefix: str = ""
    endpoint_url: str | None = None
    allow_http: bool = False
    access_key_id: str | None = None
    secret_access_key: str | None = None
    region: str | None = None

    def make_storage(self, prefix_override: str | None = None) -> icechunk.Storage:
        """Create Icechunk Storage from this configuration."""
        kwargs: dict[str, Any] = {
            "bucket": self.bucket,
            "prefix": prefix_override if prefix_override is not None else self.prefix,
        }
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.allow_http:
            kwargs["allow_http"] = True
        if self.access_key_id and self.secret_access_key:
            kwargs["access_key_id"] = self.access_key_id
            kwargs["secret_access_key"] = self.secret_access_key
        if self.region:
            kwargs["region"] = self.region
        return icechunk.s3_storage(**kwargs)


_s3_config_override: S3Config | None = None


def set_s3_config(config: S3Config | None) -> None:
    """Set a global S3 configuration override for testing."""
    global _s3_config_override
    _s3_config_override = config


def get_s3_config() -> S3Config | None:
    """Get the current S3 configuration override."""
    return _s3_config_override
