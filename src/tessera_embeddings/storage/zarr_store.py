"""Zarr store management utilities for reflectance and cloudmask data.

This module provides functions for creating, reading, and writing Zarr stores
that hold preprocessed Sentinel-2 reflectance data and cloud masks.

Uses Icechunk for transactional writes with atomic commit semantics.

Five write paths, all committing atomically:

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
- **windowed per-date batch** (:func:`write_day_windows`, on
  :func:`batched_region_writes`) — the cropped-ingest counterpart of
  ``write_dataset``: seed an all-fill store with an EMPTY time axis once, then
  per date append the time slot atomically WITH that date's chunk-disjoint
  live-window region writes, all under ONE commit. Same bookkeeping contract
  as create/append (attr set, baselines/doy merge, per-write manifest
  validation); write volume scales with live area instead of extent.
- **shard-assemble** (embeddings only; lives in
  :mod:`tessera_embeddings.inference.assembly` +
  :mod:`tessera_embeddings.storage.shard_writer`) — staged inference tiles
  written straight into the output arrays as raw-zarr fork/merge region
  writes: granularity-aligned northing bands for standalone stores, whole
  2048-px shards of a pre-allocated global-store zone group for the campaign
  (one commit per zone-year, ADR-008).

The batch variant of the region-overwrite path (a Dask-graph write of many
regions at once — one graph spanning every region, built on the flow runner)
was removed as unused; the windowed per-date batch is NOT its return, since it
builds no graph at all.
"""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, cast

import fsspec
import icechunk
import numpy as np
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk

from tessera_embeddings.errors import CorruptedStoreError
from tessera_embeddings.storage.manifest import IngestManifest, extract_manifest
from tessera_embeddings.storage.region_writes import (
    _drop_region_coords,
    _pad_region_to_chunks,
)
from tessera_embeddings.utils import utcnow_iso

logger = logging.getLogger(__name__)

# Standard time encoding for all stores
TIME_ENCODING = {"units": "nanoseconds since 1970-01-01", "calendar": "proleptic_gregorian"}


def read_time_values(node: zarr.Group) -> np.ndarray:
    """Decode a group's ``time`` coordinate to ``datetime64[ns]`` values.

    Raw-zarr counterpart to xarray's CF decoding for the one convention every
    engine-written store uses (:data:`TIME_ENCODING`); anything else is a loud
    error rather than a silent misread.
    """
    time_arr = node["time"]
    units = str(time_arr.attrs.get("units", ""))
    if not units.startswith("nanoseconds since 1970-01-01"):
        raise ValueError(
            f"Unsupported time units {units!r} on {node.store!r} — every engine-written "
            "store uses TIME_ENCODING (nanoseconds since 1970-01-01)."
        )
    return np.asarray(time_arr[:]).astype("int64").astype("datetime64[ns]")  # type: ignore[index]


def time_index_of(node: zarr.Group, value: np.datetime64) -> int | None:
    """Index of ``value`` on a group's time axis, or ``None`` if absent."""
    hits = np.flatnonzero(read_time_values(node) == value)
    return int(hits[0]) if hits.size else None


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


# Process-wide opt-in for manifest splitting. None = off (icechunk's default
# single-manifest-per-array layout); a ``{dim_name: shard_size}`` dict = split
# each named dimension's manifest into shards of that many chunks. A store's
# manifest is the index mapping every chunk to its storage location; by default
# it is ONE object per array, so every commit rewrites the whole manifest —
# O(total store size), independent of how few chunks the commit changed. At
# continental scale (hundreds of dates x thousands of spatial chunks x many
# bands) that single manifest is huge and dominates commit time. Splitting
# bounds a commit's rewrite to the shards it touches.
#
# WHICH AXIS TO SPLIT DEPENDS ENTIRELY ON WHAT ONE COMMIT TOUCHES. Two workloads
# in this repo want opposite answers, so read this before choosing sizes.
#
# (a) Region-write MERGE workload — split SPATIALLY (the default below). Its
#     manifest entry count is n_dates x northing_chunks x easting_chunks with tens
#     of chunks per spatial axis against often <256 dates, so it is spatially
#     dominated, and each write_region commits one compact, scattered ~3x3-chunk
#     block. A 2D split localizes the rewrite to the few tiles that block overlaps.
#     A time split is a no-op here at <=256 dates (one shard == the whole array).
#
# (b) Campaign zone INGEST — split by TIME ONLY (config.ingest.INGEST_MANIFEST_SPLIT).
#     One commit is one DATE, and a date writes every live window, i.e. essentially
#     the zone's whole live area. So a spatial split cannot localise anything —
#     every commit touches nearly every spatial shard — and all it adds is object
#     count: measured on a 6-degree zone, a 4x4 split rewrote ~5,097 manifest
#     objects per commit instead of ~14, and those PUT latencies made ingest 30-50%
#     SLOWER than no split at all. A time split is what localises a per-date commit,
#     and at a size well under the date count it is emphatically not a no-op.
#
# The rule behind both: split the axis along which a single commit is NARROW. Check
# that against the caller's write shape rather than inheriting the default.
#
# This is OFF by default and opt-in via ``manifest_split`` because (a) it's only
# a win on large, frequently-region-written stores, and (b) the split config
# must be applied consistently across a store's create and all later opens — a
# process-wide override (like ``credentials_provider``) guarantees that without
# threading a parameter through every public entry point.
_manifest_split_sizes: dict[str, int] | None = None

# Default for the region-write merge workload (case (a) above) — NOT for campaign
# ingest, which wants time-only. A 2D spatial split at 4 chunks per axis.
# With INGEST_CHUNK_SIZE=4096 that's ~16k px/shard — a touch larger than a
# typical ~3x3-chunk region write, so most commits hit only 1-4 tiles, while
# shard objects stay in the low hundreds on a ~50x50-chunk store. Region writes
# are spatially scattered, so a per-write commit rewrites only its tiles rather
# than a full-height stripe (which a 1D split would force).
DEFAULT_MANIFEST_SPLIT_SIZES = {"northing": 4, "easting": 4}

# Per-attempt request timeouts (ms) pushed onto every repo's icechunk storage
# config (see _default_repo_config) and inherited by every session and fork. These
# cap a SINGLE object-store attempt so a hung socket fails the attempt instead of
# blocking forever; the retry settings below then re-issue it with backoff.
# ``read_timeout_ms`` is the one that bites the diagnosed production hang (a worker
# stuck in ``sk_wait_data`` mid-response). The failure mode is request-level, not
# merge-specific — every open (writes, reads, distributed assembly) is exposed — so
# these are package-wide defaults.
_DEFAULT_CONNECT_TIMEOUT_MS = 30_000
_DEFAULT_READ_TIMEOUT_MS = 120_000
_DEFAULT_OPERATION_ATTEMPT_TIMEOUT_MS = 180_000

# Storage retry policy applied alongside the timeouts. Icechunk's default is a
# single try with no backoff, so a timed-out attempt would propagate as an error
# rather than being retried; bump tries + exponential backoff so a transient drop
# is absorbed in-process. SlowDown (503) is already retriable in the AWS SDK; this
# extends that to the timed-out/dropped-connection case.
_DEFAULT_STORAGE_MAX_TRIES = 10
_DEFAULT_STORAGE_INITIAL_BACKOFF_MS = 200
_DEFAULT_STORAGE_MAX_BACKOFF_MS = 30_000


@contextmanager
def manifest_split(split_sizes: "dict[str, int] | None" = DEFAULT_MANIFEST_SPLIT_SIZES) -> "Iterator[None]":
    """Enable manifest splitting for repos opened in this block.

    Every ``_open_repo`` / ``_create_repo`` inside the ``with`` block applies a
    :class:`icechunk.ManifestSplittingConfig` built from ``split_sizes`` — a
    ``{dimension_name: shard_size_in_chunks}`` mapping — so commits rewrite only
    the touched shards rather than the whole array manifest. The default splits
    the two spatial axes (4 chunks each); pass e.g. ``{"northing": 4, "easting":
    4, "time": 256}`` to also shard time, or ``None`` to explicitly disable
    within a block.

    Scoped like :func:`credentials_provider`: the previous setting is restored on
    exit even if the body raises, so a reused process (e.g. a Dask worker) is not
    left globally pinned. The split config must match across a store's create and
    every later open — wrap the whole merge in one block to keep them consistent.

    What splitting buys, pictorially (spatial 2D split shown; ``time@1`` is the
    same idea along the time axis — see :func:`global_store_config`)::

        unsplit: 1 manifest/array          split 4x4: 1 manifest/16-chunk tile
        ┌─────────────────────┐            ┌────┬────┬────┬────┐
        │ every chunk's ref   │            │    │    │    │    │
        │ (all years, all     │            ├────┼────┼─▓▓─┼────┤ a region write
        │  positions)         │            ├────┼────┼────┼────┤ rewrites only
        └─────────▲───────────┘            └────┴────┴────┴────┘ its tile(s) ▓
                  └── ANY commit rewrites it all: O(store)
    """
    global _manifest_split_sizes
    previous = _manifest_split_sizes
    _manifest_split_sizes = split_sizes
    try:
        yield
    finally:
        _manifest_split_sizes = previous


def _manifest_splitting_config(split_sizes: dict[str, int]) -> icechunk.ManifestSplittingConfig:
    """Split every array's manifest along each named dimension into ``shard_size`` chunks.

    Applies to all arrays (``AnyArray``), keyed on dimension *name* so each split
    tracks its axis regardless of position. Multiple entries (e.g. northing +
    easting) tile the manifest into a multi-dimensional grid of shards.
    """
    return icechunk.ManifestSplittingConfig.from_dict(
        {
            icechunk.ManifestSplitCondition.AnyArray(): {
                icechunk.ManifestSplitDimCondition.DimensionName(dim): size for dim, size in split_sizes.items()
            }
        }
    )


def _default_repo_config(max_concurrent_requests: int | None = None) -> icechunk.RepositoryConfig:
    """Build the RepositoryConfig overrides applied to every repo open.

    Chunk cache is left at icechunk's (small) default — see
    context_docs/decisions/007-icechunk-chunk-cache-disabled.md.

    **Manifest splitting** is applied only when opted in via
    :func:`manifest_split` (off by default). When active it bounds a commit's
    manifest rewrite to the shards it touched, the dominant region-write cost on
    large continental stores.

    When ``max_concurrent_requests`` is provided, caps per-repo HTTP
    concurrency. Assembly at cornbelt scale fans out thousands of concurrent
    PUTs to one zarr prefix, blowing past S3's ~3.5K/s per-prefix limit and
    triggering 503 SlowDown. Icechunk's default is 256 concurrent HTTP
    requests per repo; callers that fan out across many workers should set
    this lower (e.g. 64) so aggregate request rate stays under S3's ceiling.

    **Storage timeouts + retries** are always applied. Icechunk defaults to
    unbounded per-attempt timeouts and a single try, so a wedged socket
    (diagnosed in production: a worker stuck in ``sk_wait_data`` mid-response)
    blocks forever. We set finite connect/read/operation-attempt timeouts so a
    single attempt fails instead of hanging, and a backed-off retry budget so a
    transient drop is absorbed in-process. Tradeoff: a *retriable* failure
    (sustained 5xx, repeated timeouts) now takes minutes to surface across the
    ``max_tries`` budget instead of failing fast; non-retriable errors (403 etc.)
    still fail immediately.
    """
    config = icechunk.RepositoryConfig.default()
    if _manifest_split_sizes:
        config.manifest = icechunk.ManifestConfig(splitting=_manifest_splitting_config(_manifest_split_sizes))
    if max_concurrent_requests is not None:
        config.max_concurrent_requests = max_concurrent_requests
    # Mutate the existing StorageSettings in place (RepositoryConfig.default() may
    # leave storage=None) so any fields icechunk seeded survive.
    storage = config.storage or icechunk.StorageSettings()
    storage.timeouts = icechunk.StorageTimeoutSettings(
        connect_timeout_ms=_DEFAULT_CONNECT_TIMEOUT_MS,
        read_timeout_ms=_DEFAULT_READ_TIMEOUT_MS,
        operation_attempt_timeout_ms=_DEFAULT_OPERATION_ATTEMPT_TIMEOUT_MS,
    )
    storage.retries = icechunk.StorageRetriesSettings(
        max_tries=_DEFAULT_STORAGE_MAX_TRIES,
        initial_backoff_ms=_DEFAULT_STORAGE_INITIAL_BACKOFF_MS,
        max_backoff_ms=_DEFAULT_STORAGE_MAX_BACKOFF_MS,
    )
    config.storage = storage
    return config


# Preload scan cap for the global store: 120 groups x 10 nodes each (6 data
# arrays + 4 coords under GLOBAL) = 1200 nodes, so the icechunk default of 50
# (issue #1464) never reaches later groups' coord arrays. 2400 leaves 2x
# headroom so schema growth can't silently push trailing zones' coord manifests
# out of preload; refs cap is generous because coord manifests are tiny.
_GLOBAL_PRELOAD_MAX_ARRAYS = 2400
_GLOBAL_PRELOAD_MAX_REFS = 1_000_000


def global_store_config(max_concurrent_requests: int | None = None) -> icechunk.RepositoryConfig:
    """RepositoryConfig for the 120-group global store (ADR-008 D4/D5).

    Layers on :func:`_default_repo_config` (timeouts + retries): manifest split
    **time@1** (one manifest per year per array, so a year fill rewrites only
    that year's manifests), and preload tuning so coordinate manifests across all
    120 groups are preloaded. Persist with ``repo.save_config()`` on create so
    re-opens (and forked workers) inherit it.
    """
    config = _default_repo_config(max_concurrent_requests)
    config.manifest = icechunk.ManifestConfig(
        splitting=_manifest_splitting_config({"time": 1}),
        preload=icechunk.ManifestPreloadConfig(
            max_arrays_to_scan=_GLOBAL_PRELOAD_MAX_ARRAYS,
            max_total_refs=_GLOBAL_PRELOAD_MAX_REFS,
        ),
    )
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


def rollback_commits(
    store_path: str,
    n: int,
    *,
    branch: str = "main",
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    dry_run: bool = False,
) -> str:
    """Roll a branch's HEAD back by ``n`` commits and return the new HEAD id.

    Icechunk has no destructive "undo": a rollback is a non-fast-forward
    ``reset_branch`` that re-points the branch at an older snapshot. The ``n``
    snapshots being dropped are not deleted — they remain reachable by id until
    expiry/garbage collection, so the rollback is itself reversible (reset back
    to the old HEAD id).

    Args:
        store_path: Local path or S3 URI of the store.
        n: Number of commits to drop from the tip. Must be >= 1 and leave at
            least one snapshot (you cannot roll back past the root/init commit).
        branch: Branch to reset. Defaults to ``"main"``.
        get_credentials: Optional credential callback (see :func:`_create_storage`).
        region: Optional S3 region override.
        dry_run: When True, resolve and return the target snapshot id without
            moving the branch. Useful for eyeballing the target before committing.

    Returns:
        The id of the snapshot the branch now points at (or *would* point at,
        when ``dry_run`` is True).

    Raises:
        ValueError: If ``n < 1`` or ``n`` would drop the entire history.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    repo = _open_repo(store_path, get_credentials=get_credentials, region=region)

    # ancestry is newest-first: history[0] is the current HEAD, history[n] is
    # the snapshot n commits back. The final entry is the repo's root commit.
    history = list(repo.ancestry(branch=branch))
    current = history[0]
    if n >= len(history):
        raise ValueError(
            f"Cannot roll back {n} commits: branch {branch!r} has only "
            f"{len(history)} snapshot(s), including the root commit."
        )

    target = history[n]
    logger.info(
        "Rolling back %s on %r by %d commit(s): %s -> %s%s",
        store_path,
        branch,
        n,
        current.id,
        target.id,
        " (dry run)" if dry_run else "",
    )
    if not dry_run:
        repo.reset_branch(branch, target.id, from_snapshot_id=current.id)
    return target.id


# =============================================================================
# Session Helpers (used by all read/write operations)
# =============================================================================


# Sentinel: caller did not pass ``chunks``, so let ``xr.open_zarr`` apply its own
# default (one dask block per on-disk chunk). Distinct from ``chunks=None``, which
# is a real, meaningful value (open zarr-lazy with no dask graph).
class _ChunksUnset:
    pass


_CHUNKS_UNSET = _ChunksUnset()

# Accepted ``chunks`` values, mirroring ``xr.open_zarr``: a per-dimension mapping,
# a uniform block size, ``"auto"``, or ``None`` (no dask graph).
type ChunksArg = dict | int | str | None | _ChunksUnset


def _open_readonly(
    store_path: str,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    chunks: "ChunksArg" = _CHUNKS_UNSET,
    group: str | None = None,
) -> xr.Dataset:
    """Open store for reading. Returns xarray Dataset.

    ``chunks`` is forwarded to :func:`xarray.open_zarr` only when set; the default
    builds one dask task per on-disk chunk — fine for tile-scale stores, but on a
    CONUS-scale embeddings store the band axis alone is 32 chunks and the graph is
    ``n_time x n_y x n_x x 32`` tasks, large enough that even a lazy ``isel``/``sel``
    OOMs while manipulating the graph. Pass ``chunks=None`` to open zarr-lazy with
    no dask graph: slicing is then pure metadata and chunks are read only when
    ``.values`` is pulled — the right choice for interactive or selective reads of
    large stores.

    ``group`` selects a Zarr group within the store (the global store's per-zone
    layout). ``None`` reads the root. Readers should target a single group; never
    open the whole 120-group repo as a datatree (~200x slower — ADR-008 D5).
    """
    repo = _open_repo(store_path, get_credentials=get_credentials, region=region)
    session = repo.readonly_session(branch="main")
    if isinstance(chunks, _ChunksUnset):
        return xr.open_zarr(session.store, consolidated=False, group=group)
    return xr.open_zarr(session.store, consolidated=False, chunks=chunks, group=group)


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
    data: xr.Dataset,
    write_kwargs: dict[str, Any],
    message: str,
    update_attrs: dict[str, Any] | None = None,
) -> None:
    """Run a ``to_icechunk`` write, then commit with root attrs preserved.

    ``to_icechunk`` overwrites root attrs with the (typically empty)
    ``data.attrs`` — destroying crs, ``_manifest``, etc. We snapshot them
    before the write and restore (plus any ``update_attrs``) after, then commit.
    Shared by the append and region-write paths; ``write_kwargs`` carries the
    mode-specific arguments (``mode``, ``append_dim``/``region``, etc.).
    """
    root = zarr.open_group(session.store, mode="r")
    preserved_attrs = dict(root.attrs)

    to_icechunk(data, session, **write_kwargs)

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
        data,
        {"mode": "a", "append_dim": "time", "align_chunks": True},
        message,
        update_attrs,
    )
    logger.info(f"Appended to {store_path}")


# =============================================================================
# Region Writes (overwrite-in-place of a temporal/spatial slice)
# =============================================================================


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

    # Committed view for padding unaligned regions. Read from a *readonly*
    # session (not ``session.store``): the lazy pad shell's graph is handed to
    # ``to_icechunk``, which under a distributed client pickles it to the
    # workers. A writable session refuses to pickle; a readonly one pickles
    # freely. Pin it to the writable session's base snapshot rather than
    # re-resolving ``branch="main"`` so the shell is read from exactly the
    # snapshot we commit on top of — a concurrent commit between the two opens
    # can't make the shell come from a different snapshot than the write base.
    # The write target still uses the writable session, which ``to_icechunk``
    # forks internally.
    read_session = repo.readonly_session(snapshot_id=session.snapshot_id)
    existing = xr.open_zarr(read_session.store, consolidated=False)
    # Raw zarr group on the SAME readonly session, for the zarr-direct padding
    # shell (one read task per overlapping chunk, no whole-store layer). Must be
    # this session — same pinned snapshot as ``existing`` and pickle-safe to
    # workers — not a fresh branch="main" group.
    group = zarr.open_group(read_session.store, mode="r")
    try:
        padded, widened = _pad_region_to_chunks(existing, data, region, group)
        to_write = _drop_region_coords(padded, set(widened))

        _commit_preserving_attrs(
            session,
            to_write,
            {"mode": "r+", "region": widened, "align_chunks": True, "split_every": 8},
            message,
            update_attrs,
        )
    finally:
        existing.close()
    logger.info(f"Wrote region {widened} to {store_path}")


# =============================================================================
# Public API
# =============================================================================


def open_store(
    store_path: str,
    chunks: "ChunksArg" = _CHUNKS_UNSET,
    group: str | None = None,
    *,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
) -> xr.Dataset:
    """Open an Icechunk store for reading as an xarray Dataset.

    Pass ``chunks=None`` for large (e.g. CONUS-scale) stores to skip the dask
    task graph and slice lazily on metadata alone — see :func:`_open_readonly`.
    ``group`` selects one Zarr group (the global store's per-zone layout).
    ``get_credentials``/``region`` thread a credential callback / region to the
    repo open, for callback-only or non-default-region deployments.
    """
    return _open_readonly(store_path, get_credentials=get_credentials, region=region, chunks=chunks, group=group)


def open_store_as_zarr_group(
    store_path: str,
    max_concurrent_requests: int | None = None,
    group: str | None = None,
    *,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
) -> zarr.Group:
    """Open an Icechunk store for reading as a raw zarr Group.

    Bypasses xarray/dask entirely. Use when you need to read large arrays into
    numpy without the dask task-graph overhead — e.g., loading a chunk of
    reflectance bands for inference, where each ``.values`` call on the xarray
    path would build and execute a fresh dask graph per variable and hold
    scheduler state until the dataset handle dies.

    ``max_concurrent_requests`` caps per-repo HTTP concurrency (icechunk default
    256). Pass it when many processes read one S3 prefix concurrently — e.g. the
    region merge's worker forks all reading one feature store — so the aggregate
    GET rate stays under S3's per-prefix ceiling. ``group`` selects one Zarr
    group (the global store's per-zone layout); ``None`` returns the root group.
    ``get_credentials``/``region`` are forwarded to the opener so a store outside
    the default S3 region — or one reachable only via an explicit credential
    callback — can be read with the same options used for the writer.
    """
    return open_store_group_and_tip(
        store_path,
        max_concurrent_requests=max_concurrent_requests,
        group=group,
        get_credentials=get_credentials,
        region=region,
    )[0]


def open_store_group_and_tip(
    store_path: str,
    max_concurrent_requests: int | None = None,
    group: str | None = None,
    *,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    branch: str = "main",
) -> tuple[zarr.Group, str]:
    """Open a store for reading and also return ``branch``'s tip snapshot ID.

    Same as :func:`open_store_as_zarr_group`, plus the commit the returned group
    is a view of. The snapshot ID is the store's canonical CONTENT identity —
    it moves on every commit and cannot be left stale the way a bookkeeping
    attribute can — so callers deciding "is this the same data I saw last
    time?" should key on it rather than on attrs a writer is trusted to update.
    Both come from ONE repo open, so asking for the tip costs no extra round
    trip over opening the group alone.
    """
    repo = _open_repo(
        store_path, max_concurrent_requests=max_concurrent_requests, get_credentials=get_credentials, region=region
    )
    tip = repo.lookup_branch(branch)
    session = repo.readonly_session(branch=branch)
    root = zarr.open_group(session.store, mode="r")
    if group is None:
        return root, tip
    member = root[group]
    if not isinstance(member, zarr.Group):
        raise ValueError(f"{group!r} is not a group in {store_path}")
    return member, tip


def get_existing_dates(store_path: str, group: str | None = None) -> set[str]:
    """Get dates already present in a store. Returns empty set if store doesn't exist."""
    t0 = time.monotonic()
    logger.debug(f"Opening store: {store_path}")
    try:
        ds = _open_readonly(store_path, group=group)
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
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
    group: str | None = None,
) -> dict[str, slice]:
    """Map coordinate-value ranges to half-open integer slices for an existing store.

    Each argument is a fully-inclusive ``(low, high)`` coordinate range or
    ``None`` (= the full axis, omitted from the result). Both ends are matched
    (``coord >= low & coord <= high``); the returned slices are half-open
    ``[start, stop)`` in the usual Python sense. ``(low, high)`` may be given in
    either order — they're sorted before matching — so a descending axis can be
    addressed low-to-high or high-to-low. ``time`` accepts anything
    ``np.datetime64`` understands. Spatial bounds are matched against the
    store's ``northing``/``easting`` coordinate values regardless of axis
    direction (northing typically descends).

    ``get_credentials``/``s3_region`` are forwarded to the store opener so a
    store outside the default S3 region can be resolved with the same options
    later passed to :func:`write_region`.

    Returns a dict of integer slices suitable for :func:`write_region`. Only
    dims with a non-``None`` range are included; an empty dict means "the whole
    array" (which the caller should treat as a plain overwrite, not a region).

    Enforces the overwrite-in-place contract: the requested range must select at
    least one existing coordinate. A range that matches nothing (e.g. a date not
    in the store) raises ``ValueError`` — appending new coordinates is
    :func:`write_dataset`'s job, not a region write's. The matched coordinates
    must also be contiguous; a range straddling a gap (e.g. an out-of-order time
    axis) raises rather than silently widening the slice to cover the gap.
    """
    # chunks=None: only the 1-D northing/easting/time coords are read here, never
    # pixels, so skip building a per-chunk dask graph — matters when this is called
    # repeatedly (e.g. resolving many features against a continental master).
    ds = _open_readonly(store_path, get_credentials=get_credentials, region=s3_region, chunks=None, group=group)
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
            # Accept the range in either order: a descending axis (northing) is
            # often addressed high-to-low, so sort the bounds before masking.
            lo_b, hi_b = (low, high) if low <= high else (high, low)
            mask = (coord >= lo_b) & (coord <= hi_b)
            hits = np.flatnonzero(mask)
            if hits.size == 0:
                raise ValueError(
                    f"Region {dim}={rng!r} selects no existing coordinate in {store_path}. "
                    "Region writes overwrite in place; use write_dataset to add new coordinates."
                )
            if hits[-1] - hits[0] + 1 != hits.size:
                raise ValueError(
                    f"Region {dim}={rng!r} selects non-contiguous coordinates in {store_path} "
                    "(the axis is unsorted or has gaps). A region must be a single contiguous slice."
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


class RegionWriteBatch:
    """Many positional window writes under ONE icechunk commit.

    Handed out by :func:`batched_region_writes`. Direct zarr assignment on the
    session's store — no Dask, no task graph (the avoid-Dask principle in the
    live-tile-cropping design note): the caller computes each window to numpy and
    this places it. Windows must be mutually chunk-disjoint (the chunk-snapped
    window derivation guarantees it); two writes straddling one chunk in a session
    are unsupported.
    """

    def __init__(self, session: "icechunk.Session") -> None:
        #: The batch's session. Exposed because window PIXEL data at volume should
        #: be written as ``to_icechunk(win, batch.session, mode="r+", region=...)``
        #: — placement stays on the Dask workers that computed the pixels instead
        #: of funnelling the date's live volume through the one worker running the
        #: caller (dense-zone arithmetic in the design note). Same session, same
        #: single commit; ``write_window`` below is for volumes that comfortably
        #: fit on the calling worker.
        self.session = session
        #: The store's root group, writable. Exposed so callers can read/merge root
        #: attrs (baselines, doy, last_appended) — attr edits land in the same commit.
        self.group: zarr.Group = zarr.open_group(session.store, mode="r+")

    def append_time_slot(self, when: np.datetime64) -> int:
        """Grow the time axis by one date and return its index. Metadata only.

        Resizes the time coord and every time-dimensioned array by one slot —
        icechunk arrays are sparse, so the new slot costs no chunks until written.
        The session sees its own uncommitted resize, so window writes into the
        returned index work immediately (verified by experiment; see the design
        note). A date already on the axis raises: the callers dedupe upstream, so
        a duplicate here means a retry bug about to double-stamp one date.
        """
        when_ns = np.asarray(when, dtype="datetime64[ns]")
        existing = read_time_values(self.group)
        if (existing == when_ns).any():
            raise ValueError(f"date {when_ns} is already on the time axis; refusing a duplicate slot")
        t_index = len(existing)
        time_arr = self.group["time"]
        assert isinstance(time_arr, zarr.Array)
        time_arr.resize((t_index + 1,))
        time_arr[t_index] = when_ns.astype("int64")  # int64 ns per TIME_ENCODING
        for name, arr in self.group.arrays():
            # zarr v3 metadata carries dimension_names; every engine-written store
            # is v3, and a v2 array would predate the convention entirely.
            dims = getattr(arr.metadata, "dimension_names", None)
            if name != "time" and dims and dims[0] == "time":
                arr.resize((t_index + 1, *arr.shape[1:]))
        return t_index

    def write_window(self, t_index: int, y: slice, x: slice, values: dict[str, np.ndarray]) -> None:
        """Assign one window's pixels for one date, var by var."""
        for var, data in values.items():
            arr = self.group[var]
            assert isinstance(arr, zarr.Array)
            arr[t_index, y, x] = data


@contextmanager
def batched_region_writes(
    store_path: str,
    *,
    message: str,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
) -> Iterator[RegionWriteBatch]:
    """One writable session, N window writes (+ optional attr edits), one commit.

    The per-date write unit of the live-window ingest path: everything done through
    the yielded :class:`RegionWriteBatch` lands atomically in a single snapshot on
    exit. On an exception nothing commits — the abandoned session is invisible to
    readers, so a retry starts clean from the last committed state.

    Deliberately NOT the removed Dask-graph batch write (`write_regions`): there is
    no graph here at all. For a single arbitrary (unaligned) region overwrite, use
    :func:`write_region`, which pads to chunk boundaries; this path requires
    chunk-disjoint windows and one commit is the point.
    """
    repo = _open_repo(store_path, get_credentials=get_credentials, region=s3_region)
    session = repo.writable_session("main")
    yield RegionWriteBatch(session)
    session.commit(message)


def write_day_windows(
    store_path: str,
    day_ds: xr.Dataset,
    windows: "list[tuple[int, int, int, int]]",
    *,
    # Duck-typed like create_empty_store's roi param: needs .geobox/.height/.width.
    # Typed Any because RoiMetadata lives in ingest/, which imports storage — a
    # concrete annotation here would be a layering cycle.
    roi: Any,  # noqa: ANN401 — see comment
    manifest: IngestManifest | None,
    baselines: dict[str, int],
    tile_id: str,
    crs: str,
    chunks: dict[str, int],
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
) -> None:
    """Write ONE date's live windows into a mosaic store, one commit for the date.

    The cropped counterpart of :func:`write_dataset` (same bookkeeping contract,
    write volume proportional to live area instead of extent). ``day_ds`` is the
    date's full-extent LAZY dataset (``time`` size 1); each ``(y0, y1, x0, x1)``
    window — chunk-disjoint, from ``ingest.live_windows`` — is written as a
    ``to_icechunk`` region on one shared session, so the pixels flow from the Dask
    workers that computed them (never materialised on the caller — see the design
    note's dense-zone arithmetic) and the date lands as ONE snapshot.

    A missing store is seeded all-fill with an EMPTY time axis via
    :func:`~.empty_store.create_empty_store` (schema-only: cost independent of
    extent, same attr set :func:`write_dataset` creates); every date — the first
    included — then appends its time slot atomically WITH its windows and merges
    attrs exactly as the append path does (baselines union / doy concat /
    ``last_appended`` bump). The manifest is validated against the store BEFORE
    anything is written — the per-append structural gate must not be lost to
    batching.
    """
    from tessera_embeddings.storage.empty_store import create_empty_store  # local: storage-internal, avoids cycle

    if day_ds.sizes["time"] != 1:
        raise ValueError(f"write_day_windows writes one date per call; got time size {day_ds.sizes['time']}")

    # The windowed write does NOT realign the producer's blocks to the store's chunks
    # (see the to_icechunk call below), which is only sound while every store chunk a
    # window covers is written WHOLLY by that window. Checked here rather than trusted:
    # a misaligned window would either be rejected by mode="r+" deep in the write or,
    # worse, straddle a chunk with a neighbouring window and make the result depend on
    # write order. Ends are exempt — the array's own last chunk is short.
    _cy, _cx = chunks["northing"], chunks["easting"]
    height, width = roi.height, roi.width
    for y0, y1, x0, x1 in windows:
        if y0 % _cy or x0 % _cx or (y1 % _cy and y1 != height) or (x1 % _cx and x1 != width):
            raise ValueError(
                f"window ({y0}, {y1}, {x0}, {x1}) is not aligned to the store's "
                f"({_cy}, {_cx}) chunks; the windowed write cannot straddle a chunk"
            )
    when = np.asarray(day_ds.time.values, dtype="datetime64[ns]")[0]
    date_str = str(when)[:10]

    # Seed with an EMPTY time axis, so the first date takes the same atomic
    # append+windows commit as every other date. Seeding times=[when] would
    # commit the date BEFORE its pixels: a crash between the two leaves an
    # all-fill timestep that get_existing_dates then reports as ingested — the
    # retry hits the duplicate-date guard and the STAC dedupe filters the date
    # forever. Existence is probed on the repo, not the (possibly empty) axis.
    try:
        # Probe the ROOT GROUP, not just the repo: _create_repo creates the repo
        # before create_empty_store writes and commits the schema, so a crash in
        # between leaves a rootless repo. Probing the repo alone would then find
        # it, skip seeding, and every retry would fail opening the missing group —
        # wedged forever. (Same failure the global-store seeder hit; see its
        # GroupNotFoundError handling.)
        open_store_as_zarr_group(store_path, get_credentials=get_credentials, region=s3_region)
    except (FileNotFoundError, KeyError, zarr.errors.GroupNotFoundError, icechunk.IcechunkError):
        create_empty_store(
            store_path,
            roi=roi,
            times=np.array([], dtype="datetime64[ns]"),
            var_dtypes={str(v): day_ds[v].dtype for v in day_ds.data_vars},
            tile_id=tile_id,
            crs=crs,
            chunks=chunks,
            baselines={},  # merged per date below, exactly like the append path
            manifest=manifest,
            # Seed through a repo opened with THIS call's credentials/region. Letting
            # create_empty_store open its own would use the default storage config,
            # so the probe above and every write below would honour a callback or a
            # non-default region while the one-time seed silently did not — failing
            # the first cropped date of any such deployment.
            repo=_create_repo(store_path, get_credentials=get_credentials, region=s3_region),
        )

    with batched_region_writes(
        store_path,
        message=f"date {date_str}: {len(windows)} live window(s)",
        get_credentials=get_credentials,
        s3_region=s3_region,
    ) as batch:
        if manifest:
            manifest.validate_against(extract_manifest(dict(batch.group.attrs)), store_path)
        t = batch.append_time_slot(when)
        attrs = batch.group.attrs
        merged = dict(cast("dict", attrs.get("baselines_applied", {})))
        merged.update(baselines)
        attrs["baselines_applied"] = merged
        attrs["doy"] = list(cast("list", attrs.get("doy", []))) + compute_doy(np.array([when])).tolist()
        attrs["last_appended"] = utcnow_iso()
        drop = [c for c in ("time", "northing", "easting") if c in day_ds.coords]
        for y0, y1, x0, x1 in windows:
            win = day_ds.isel(northing=slice(y0, y1), easting=slice(x0, x1)).drop_vars(drop)
            to_icechunk(
                win,
                batch.session,
                mode="r+",
                region={"time": slice(t, t + 1), "northing": slice(y0, y1), "easting": slice(x0, x1)},
                # NO align_chunks here, deliberately: it would remap the producer's
                # blocks to the store's chunks, which multiplies the write tasks by
                # the block/chunk ratio AND adds split nodes upstream — measured at
                # 2.37x the graph for identical bytes at the current geometry. Graph
                # task count is what limits this ingest.
                #
                # Safe only because of an invariant the caller guarantees: windows are
                # derived on the LOAD-BLOCK grid (live_windows_for_mask's window_px)
                # and load blocks are a whole multiple of the store's chunks, so every
                # store chunk this region covers is written WHOLLY by exactly one
                # block. Nothing partial, so mode="r+" has no partial-chunk write to
                # reject and no read-modify-write is needed. Break that alignment and
                # the remap becomes necessary again.
                #
                # (An earlier note here recorded dropping align_chunks as ~11% slower.
                # That was measured when blocks and store chunks were the SAME size, so
                # the remap was a no-op and the difference sat inside the ~19% per-date
                # variance we have since quantified.)
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
