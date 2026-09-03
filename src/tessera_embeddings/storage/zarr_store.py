"""Zarr store management utilities for reflectance and cloudmask data.

Creating, reading and writing the Icechunk-backed Zarr stores that hold preprocessed
Sentinel-2 reflectance and cloud masks. Five write paths, all committing atomically:

- **create** (:func:`write_dataset` on a fresh store) — ``to_icechunk`` mode ``"w"``
  writes the whole array.
- **append** (:func:`write_dataset` on an existing store) — mode ``"a"`` with
  ``append_dim="time"`` extends the time axis.
- **region overwrite** (:func:`write_region`) — mode ``"r+"`` rewrites an existing
  temporal/spatial slice in place. The region need not be chunk-aligned:
  :func:`_pad_region_to_chunks` widens unaligned edges to chunk boundaries and backfills
  the untouched cells from the store (read-modify-write), because ``align_chunks=True``
  only remaps the producer's dask blocks and ``mode="r+"`` rejects partial-chunk writes.
  Use :func:`resolve_region` to turn coordinate ranges into the integer slices it wants.
- **windowed per-date batch** (:func:`write_day_windows`, on
  :func:`batched_region_writes`) — the cropped-ingest counterpart of ``write_dataset``:
  seed an all-fill store with an EMPTY time axis once, then per date append the time slot
  atomically WITH that date's chunk-disjoint live-window region writes, all under ONE
  commit. Same bookkeeping contract as create/append (attr set, baselines/doy merge,
  per-write manifest validation); write volume scales with live area instead of extent.
  ``parallel_windows`` submits the date's windows as a SINGLE dask compute rather than one
  blocking compute per window, so their critical paths overlap across the fleet instead of
  summing — the same store either way, and the windows' chunk-disjointness is what makes
  the merged changesets conflict-free.
- **shard-assemble** (embeddings only; in :mod:`tessera_embeddings.inference.assembly` +
  :mod:`tessera_embeddings.storage.shard_writer`) — staged inference tiles written
  straight into the output arrays as raw-zarr fork/merge region writes:
  granularity-aligned northing bands for standalone stores, whole 2048-px shards of a
  pre-allocated global-store zone group for the campaign (one commit per zone-year,
  ADR-008).
"""

import itertools
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
from tenacity import (
    RetryCallState,
    Retrying,
    before_sleep_log,
    retry_if_not_exception_type,
    wait_exponential,
    wait_random,
)

from tessera_embeddings.errors import (
    CorruptedStoreError,
    DuplicateDateError,
    InconclusiveStoreProbeError,
    NonMonotonicDateError,
    StoreHoldsCommittedDataError,
)
from tessera_embeddings.storage.manifest import MIXED_CODE_IDENTITIES_ATTR, IngestManifest, extract_manifest
from tessera_embeddings.storage.region_writes import (
    _drop_region_coords,
    _pad_region_to_chunks,
)
from tessera_embeddings.utils import utcnow_iso

logger = logging.getLogger(__name__)

# Standard time encoding for all stores
TIME_ENCODING = {"units": "nanoseconds since 1970-01-01", "calendar": "proleptic_gregorian"}

#: Write failures that mean **another writer holds this store**, and so must never be
#: retried. Pass to ``tenacity``'s ``retry_if_not_exception_type`` wherever a store write
#: is retried.
#:
#: Retrying either is worse than failing. The ingest path commits with no ``rebase_with``
#: precisely so a second writer's commit fails against a moved branch tip rather than merging
#: silently,
#: and a retry re-opens the session from the NEW tip — turning that refusal back into a
#: success and letting two writers interleave dates on one axis, the exact outcome the
#: no-rebase choice prevents. When the two collide on the SAME date the retry instead fails as
#: a duplicate, so the visible error names a date rather than the collision.
#:
#: Everything else a store write can raise — throttling, expired credentials, a transient GDAL
#: read — is worth retrying, and a failed attempt commits nothing, so the retry starts from
#: clean committed state.
CONCURRENT_WRITER_ERRORS: tuple[type[BaseException], ...] = (icechunk.ConflictError, DuplicateDateError)

#: Failures that must NEVER trigger `cleanup_on_failure`'s delete, because each one means
#: the store holds data somebody else committed. Named rather than unpacked inline in the
#: `except` clause: mypy cannot verify a starred tuple there and rejects it outright.
NEVER_CLEAN_UP: tuple[type[BaseException], ...] = (
    StoreHoldsCommittedDataError,
    InconclusiveStoreProbeError,
    *CONCURRENT_WRITER_ERRORS,
)

#: Attempts per store write. Three is enough for the transient failures this path sees
#: (throttling, a credential rolling over mid-write, a flaky COG read), and NOT enough for a
#: source provider that stops serving reads for minutes at a time — a different kind of
#: failure, not a bigger one, since only waiting resolves it and three exponential backoffs do
#: not wait long enough to matter. A caller that can identify such a failure passes
#: ``wait_out`` to :func:`store_write_retrying` for :data:`WAIT_OUT_BACKOFF_S` of patience on
#: that failure alone.
STORE_WRITE_ATTEMPTS = 3

#: The total backoff ONE store write may spend on a failure its caller has said to wait out —
#: the single lever for that tolerance, with the ladder's cap below turning it into an attempt
#: count. Budgeted as accumulated BACKOFF rather than wall clock so the bound is a property of
#: the policy alone; the work each attempt does is the caller's and unmeasurable from here.
#:
#: **Minutes, not tens of minutes, and the ceiling is a COST not a guess.** A write waits with
#: its leg's whole Dask fleet held idle behind it, so this is the expensive place to be
#: patient. The cheap place is between leg attempts, where the fleet has been released — so
#: this budget covers the ordinary wobble and a source that stays down escalates to the leg.
#: Raising it trades fleet-hours for events the cheap wait already covers.
WAIT_OUT_BACKOFF_S = 300.0

#: Ceiling on ONE sleep in the ladder. Reached only by a write that is waiting something out:
#: :data:`STORE_WRITE_ATTEMPTS` attempts sleep twice and never approach it, so the ordinary
#: policy is unchanged by it.
WAIT_OUT_MAX_SLEEP_S = 60.0


def store_write_retrying(
    log: "logging.Logger | logging.LoggerAdapter[logging.Logger]",
    *,
    wait_out: Callable[[BaseException], bool] | None = None,
) -> Retrying:
    """The retry policy for ONE store write, shared by every ingest write site.

    Exponential backoff with jitter, re-raising the last failure, and **never retrying** a
    :data:`CONCURRENT_WRITER_ERRORS` member. :data:`STORE_WRITE_ATTEMPTS` attempts by default.

    Shared rather than built per site — S1's per-date write, S2's per-date write and S2's
    per-batch write all use it — because the exclusion is the easy part to omit, and one
    policy in one place cannot half-apply. Retrying is safe precisely because a failed write
    commits NOTHING: every write site goes through a single-session commit, so an abandoned
    attempt is invisible to readers and the next starts from the last committed state.

    The jitter is unconditional: at fleet width thousands of writes fail in the same second
    when a shared source falters, and an undithered ladder re-issues them all in the same
    second too.

    Args:
        log: Logger for the before-sleep line.
        wait_out: Predicate over the failure deciding whether this write may keep going past
            the attempt limit, up to :data:`WAIT_OUT_BACKOFF_S` of accumulated backoff. A
            predicate, not a flag, because the classification belongs to the ingest layer
            that owns the source's vocabulary; storage owns only the budget. Default ``None``
            keeps the attempt limit for every failure — what a caller wants when its answer
            to a source refusing reads is to read a DIFFERENT copy of the object, since a
            long wait per copy would multiply with that ladder.

    Returns:
        A ``Retrying`` to iterate around the write.
    """

    def _stop(state: RetryCallState) -> bool:
        exc = state.outcome.exception() if state.outcome is not None else None
        if wait_out is not None and exc is not None and wait_out(exc):
            # Refusing the sleep that would cross the budget, rather than the one after it, so
            # the ceiling is the real bound and not the bound plus one full sleep.
            return state.idle_for + state.upcoming_sleep > WAIT_OUT_BACKOFF_S
        return state.attempt_number >= STORE_WRITE_ATTEMPTS

    return Retrying(
        stop=_stop,
        wait=wait_exponential(multiplier=1, min=2, max=WAIT_OUT_MAX_SLEEP_S) + wait_random(0, 5),
        before_sleep=before_sleep_log(cast("logging.Logger", log), logging.WARNING),
        retry=retry_if_not_exception_type(CONCURRENT_WRITER_ERRORS),
        reraise=True,
    )


def read_time_values(node: zarr.Group) -> np.ndarray:
    """Decode a group's ``time`` coordinate to ``datetime64[ns]`` values.

    Raw-zarr counterpart to xarray's CF decoding for the one convention every engine-written
    store uses (:data:`TIME_ENCODING`); anything else is a loud error, not a silent misread.
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


def _fsspec_credentials(
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
) -> dict[str, Any]:
    """Storage options for fsspec, drawn from the SAME credential source icechunk uses.

    fsspec resolves the ``AWS_*`` environment variables, and on the S1 ingest path those have
    been overwritten with OPERA-scoped STS tokens for downloading NASA granules — so a bare
    ``fsspec.filesystem("s3")`` there authenticates as OPERA against OUR bucket and is
    refused. That is how a store cleanup came to fail with ``Forbidden`` while every icechunk
    write beside it succeeded. Explicit credentials, or the registered provider, are the only
    reliable source in that process. See ``_default_credentials_provider``.
    """
    provider = get_credentials or _default_credentials_provider
    if provider is None:
        return {}
    creds = provider()
    opts: dict[str, Any] = {"key": creds.access_key_id, "secret": creds.secret_access_key}
    if creds.session_token:
        opts["token"] = creds.session_token
    if region:
        opts["client_kwargs"] = {"region_name": region}
    return opts


def _delete_store(
    store_path: str,
    *,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
) -> bool:
    """Delete a store at the given path. Returns True if deleted, False if not found.

    Takes credentials for the reason :func:`_fsspec_credentials` gives: without them this is
    the one S3 operation in the module authenticating from the environment, and on the S1
    path the environment points somewhere else entirely.

    Returns rather than raises — it runs while another exception is propagating and must not
    mask it — but logs a failure at ERROR with its consequence, because a silent failed
    cleanup is what turns an interrupted write into a wedged prefix.
    """
    protocol = fsspec.utils.get_protocol(store_path)
    try:
        options = _fsspec_credentials(get_credentials, s3_region) if protocol == "s3" else {}
        fs = fsspec.filesystem(protocol, **options)
        if fs.exists(store_path):
            fs.rm(store_path, recursive=True)
            return True
        return False
    except Exception as e:
        logger.error(
            "Failed to delete store %s: %s. The prefix may now hold objects with no "
            "readable repository; a retry can adopt it, but inspect it if the ingest "
            "does not recover.",
            store_path,
            e,
        )
        return False


def cleanup_on_failure[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that deletes the store at store_path (first arg) if the function fails.

    Use on store creation functions so a failure leaves no corrupt or partial store.

    Forwards the wrapped call's ``get_credentials`` / ``s3_region`` to the delete, so the
    cleanup authenticates the same way the write it is cleaning up after did — with ambient
    credentials it would authenticate as OPERA on the S1 path (see
    :func:`_fsspec_credentials`).

    Best-effort, and not the only line of defence: an interrupted create leaves a repo the
    next attempt ADOPTS rather than re-creates (:func:`_write_new`), so a failed cleanup
    costs a wasted prefix rather than a wedged one.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        store_path: str = args[0] if args else kwargs.get("store_path")  # type: ignore[assignment]
        if not store_path:
            raise ValueError("cleanup_on_failure requires store_path as first argument")
        try:
            return func(*args, **kwargs)
        except NEVER_CLEAN_UP:
            # NOT ours to delete. Every other failure leaves a half-written store worth
            # removing; these fired because the store holds data somebody else committed.
            #
            # StoreHoldsCommittedDataError is the check-time half — the store already held
            # data when we looked. The concurrent-writer errors are the RACE half: two
            # first-date writers can both pass the empty-store probe, and the loser then
            # fails at ``session.commit()`` with ConflictError, which is not the guard's
            # error, so it would fall through to the delete below and take the WINNER's
            # committed data with it. A conflict is positive evidence that the prefix is not
            # ours, so it is the one class of failure that must never clean up.
            raise
        except Exception:
            logger.warning(f"Store creation failed, cleaning up {store_path}")
            _delete_store(
                store_path,
                get_credentials=kwargs.get("get_credentials"),  # type: ignore[arg-type]
                s3_region=kwargs.get("s3_region"),  # type: ignore[arg-type]
            )
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


# A hardcoded default region is normally an anti-pattern here — paths and credentials are
# caller-supplied so the code works for any deployment — but the Sentinel-1 OPERA RTC archive
# lives in us-west-2 and authenticates only from in-region clients (cross-region requests are
# rejected, not merely charged for). Any pipeline running against OPERA must therefore be in
# us-west-2, so this keeps that path zero-config; callers override via ``region``.
_DEFAULT_S3_REGION = "us-west-2"


# Process-wide fallback credential provider for icechunk S3 storage. When set, _create_storage
# uses it for any S3 open with no explicit get_credentials. The S1 ingest path registers an
# IAM-resolving callback here (via the AWS provider) so icechunk writes keep using IAM-role
# creds even after set_s3_credentials overwrites the AWS_* env vars with OPERA-scoped STS
# tokens. Stays None in the open-source layer, keeping it cloud-agnostic.
_default_credentials_provider: "Callable[[], icechunk.S3StaticCredentials] | None" = None


def plain_zarr_storage_options(
    path: str,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None,
    s3_region: str | None,
) -> dict | None:
    """Fsspec options for a PLAIN zarr at ``path``, from this run's Icechunk credentials.

    An ROI mask is a plain zarr, not an Icechunk store, so it is opened through fsspec and
    does not travel on the credential callback its callers thread everywhere else. Left to
    the ambient chain those opens fail outright in a callback-only deployment — and they are
    the reads that decide which chunks exist and where the grid is, so they happen before
    anything that was wired correctly.

    Lives here rather than beside either caller because BOTH the ingest's ROI export and the
    inference assembly need it, and neither package should import the other for five lines.
    Resolved at CALL time, not stored: an IAM credential expires in hours, and the point of a
    callback is that each consumer asks when it needs one.
    """
    if fsspec.utils.get_protocol(path) != "s3":
        return None
    options: dict[str, Any] = {}
    if s3_region:
        options["client_kwargs"] = {"region_name": s3_region}
    if get_credentials is not None:
        creds = get_credentials()
        options |= {
            "key": creds.access_key_id,
            "secret": creds.secret_access_key,
            "token": creds.session_token,
        }
    return options or None


@contextmanager
def credentials_provider(
    provider: "Callable[[], icechunk.S3StaticCredentials] | None",
) -> "Iterator[None]":
    """Temporarily register a fallback credential provider for icechunk S3 opens.

    Used by :func:`_create_storage` whenever an S3 open inside the ``with`` block has no
    explicit ``get_credentials``. Scoped to the block, and restored even if the body raises,
    so a reused process (e.g. a Dask worker) is not left pinned to ``provider`` for later,
    unrelated icechunk opens. See the module-level ``_default_credentials_provider`` note for
    why the S1 ingest path needs this.
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
        get_credentials: Optional credential callback for Icechunk's S3 client, returning a
            fresh :class:`icechunk.S3StaticCredentials` per invocation; ``expires_after`` on
            the returned object tells Icechunk when to call back for a refresh. ``None`` on
            an S3 path falls back to the default AWS credential chain (env, instance profile,
            etc.), which suits local testing with moto. The AWS provider in the closed-source
            repo supplies a botocore-backed callback adapting boto's
            ``RefreshableCredentials`` into ``S3StaticCredentials``.
        region: Optional S3 region. Defaults to ``us-west-2`` because the Sentinel-1 OPERA
            RTC archive only authenticates in-region. Override for stores outside it.
        scatter_initial_credentials: When True, Icechunk eagerly calls ``get_credentials``
            once and caches the result, so pickled copies of the storage (shipped to Ray
            actors or Dask workers during ``to_icechunk``) do not all stampede the provider
            on deserialisation. True for distributed assembly.
    """
    if store_path.startswith("s3://"):
        bucket, prefix = _parse_s3_url(store_path)
        if _s3_config_override:
            return _s3_config_override.make_storage(prefix_override=prefix)
        # Fall back to the globally-registered provider when the caller passed none. This is
        # how the S1 ingest path keeps icechunk on IAM-role creds: set_s3_credentials
        # overwrites the AWS_* env vars with OPERA-scoped STS tokens for GDAL reads, and
        # icechunk's default AWS chain would otherwise pick those up and get AccessDenied
        # writing our own store. See tessera_embeddings.providers.aws.credentials.
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
# single-manifest-per-array layout); a ``{dim_name: shard_size}`` dict splits each named
# dimension's manifest into shards of that many chunks. A store's manifest is the index
# mapping every chunk to its storage location, and by default it is ONE object per array, so
# every commit rewrites the whole thing — O(total store size), independent of how few chunks
# the commit changed. At continental scale that dominates commit time; splitting bounds a
# commit's rewrite to the shards it touches.
#
# WHICH AXIS TO SPLIT DEPENDS ENTIRELY ON WHAT ONE COMMIT TOUCHES — split the axis along
# which a single commit is NARROW, and check that against the caller's write shape rather
# than inheriting the default. Two workloads here want opposite answers:
#
# (a) Region-write MERGE workload — split SPATIALLY (the default below). Manifest entries are
#     n_dates x northing_chunks x easting_chunks, tens of chunks per spatial axis against
#     often <256 dates, so it is spatially dominated, and each write_region commits one
#     compact scattered ~3x3-chunk block that a 2D split localises. A time split is a no-op
#     here at <=256 dates (one shard == the whole array).
#
# (b) Campaign zone INGEST — split by TIME ONLY (config.ingest.INGEST_MANIFEST_SPLIT). One
#     commit is one DATE and a date writes essentially the zone's whole live area, so a
#     spatial split localises nothing and only adds object count: on a 6-degree zone a 4x4
#     split rewrote ~5,097 manifest objects per commit instead of ~14, and those PUT
#     latencies made ingest 30-50% SLOWER than no split at all.
#
# OFF by default and opt-in via ``manifest_split`` because it only wins on large,
# frequently-region-written stores, and because the split config must be applied consistently
# across a store's create and all later opens — a process-wide override (like
# ``credentials_provider``) guarantees that without threading a parameter through every
# public entry point.
_manifest_split_sizes: dict[str, int] | None = None

# Default for the region-write merge workload, case (a) above — NOT for campaign ingest,
# which wants time-only. With INGEST_CHUNK_SIZE=4096 a 4x4 spatial split is ~16k px/shard, a
# touch larger than a typical ~3x3-chunk region write, so most commits hit only 1-4 tiles
# while shard objects stay in the low hundreds on a ~50x50-chunk store. Region writes are
# spatially scattered, so a per-write commit rewrites its tiles rather than the full-height
# stripe a 1D split would force.
DEFAULT_MANIFEST_SPLIT_SIZES = {"northing": 4, "easting": 4}

# Per-attempt request timeouts (ms) pushed onto every repo's icechunk storage config (see
# _default_repo_config) and inherited by every session and fork, capping a SINGLE object-store
# attempt so a hung socket fails it instead of blocking forever; the retry settings below then
# re-issue with backoff. ``read_timeout_ms`` is the one that bites the diagnosed production
# hang, a worker stuck in ``sk_wait_data`` mid-response. Package-wide defaults because the
# failure is request-level, not merge-specific: every open is exposed.
_DEFAULT_CONNECT_TIMEOUT_MS = 30_000
_DEFAULT_READ_TIMEOUT_MS = 120_000
_DEFAULT_OPERATION_ATTEMPT_TIMEOUT_MS = 180_000

# Storage retry policy applied alongside the timeouts. Icechunk defaults to a single try with
# no backoff, so a timed-out attempt would propagate as an error rather than being retried;
# more tries plus exponential backoff absorbs a transient drop in-process. SlowDown (503) is
# already retriable in the AWS SDK; this extends that to timed-out/dropped connections.
_DEFAULT_STORAGE_MAX_TRIES = 10
_DEFAULT_STORAGE_INITIAL_BACKOFF_MS = 200
_DEFAULT_STORAGE_MAX_BACKOFF_MS = 30_000


@contextmanager
def manifest_split(split_sizes: "dict[str, int] | None" = DEFAULT_MANIFEST_SPLIT_SIZES) -> "Iterator[None]":
    """Enable manifest splitting for repos opened in this block.

    Every ``open_repo`` / ``_create_repo`` inside the ``with`` block applies a
    :class:`icechunk.ManifestSplittingConfig` built from ``split_sizes``, a
    ``{dimension_name: shard_size_in_chunks}`` mapping, so commits rewrite only the touched
    shards rather than the whole array manifest. The default splits the two spatial axes at 4
    chunks each; pass e.g. ``{"northing": 4, "easting": 4, "time": 256}`` to shard time too,
    or ``None`` to disable within a block.

    Scoped like :func:`credentials_provider` — the previous setting is restored on exit even
    if the body raises, so a reused process (e.g. a Dask worker) is not left globally pinned.
    The split config must match across a store's create and every later open, so wrap the
    whole merge in one block.

    What splitting buys, pictorially (spatial 2D split shown; ``time@1`` is the same idea
    along the time axis — see :func:`global_store_config`)::

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

    Applies to all arrays (``AnyArray``), keyed on dimension *name* so a split tracks its
    axis regardless of position. Multiple entries tile the manifest into a grid of shards.
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

    **Manifest splitting** is applied only when opted in via :func:`manifest_split` (off by
    default). When active it bounds a commit's manifest rewrite to the shards it touched, the
    dominant region-write cost on large continental stores.

    ``max_concurrent_requests`` caps per-repo HTTP concurrency (icechunk's default is 256).
    Assembly at cornbelt scale fans out thousands of concurrent PUTs to one zarr prefix,
    blowing past S3's ~3.5K/s per-prefix limit and triggering 503 SlowDown, so callers
    fanning out across many workers should set it lower (e.g. 64).

    **Storage timeouts + retries** are always applied, because icechunk defaults to unbounded
    per-attempt timeouts and a single try, so a wedged socket (diagnosed in production: a
    worker stuck in ``sk_wait_data`` mid-response) blocks forever. Finite
    connect/read/operation-attempt timeouts make a single attempt fail instead of hanging,
    and a backed-off retry budget absorbs a transient drop in-process. Tradeoff: a *retriable*
    failure (sustained 5xx, repeated timeouts) now takes minutes to surface across the
    ``max_tries`` budget instead of failing fast; non-retriable errors (403 etc.) still fail
    immediately.
    """
    config = icechunk.RepositoryConfig.default()
    if _manifest_split_sizes:
        config.manifest = icechunk.ManifestConfig(splitting=_manifest_splitting_config(_manifest_split_sizes))
    if max_concurrent_requests is not None:
        config.max_concurrent_requests = max_concurrent_requests
    # Mutate the existing StorageSettings in place (RepositoryConfig.default() may leave
    # storage=None) so any fields icechunk seeded survive.
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


# Preload scan cap for the global store: 120 groups x 10 nodes each (6 data arrays + 4 coords
# under GLOBAL) = 1200 nodes, so the icechunk default of 50 (issue #1464) never reaches later
# groups' coord arrays. 2400 leaves 2x headroom so schema growth cannot silently push trailing
# zones' coord manifests out of preload; the refs cap is generous because coord manifests are
# tiny.
_GLOBAL_PRELOAD_MAX_ARRAYS = 2400
_GLOBAL_PRELOAD_MAX_REFS = 1_000_000


def global_store_config(max_concurrent_requests: int | None = None) -> icechunk.RepositoryConfig:
    """RepositoryConfig for the 120-group global store (ADR-008 D4/D5).

    Layers on :func:`_default_repo_config` (timeouts + retries): manifest split **time@1**,
    one manifest per year per array so a year fill rewrites only that year's, plus preload
    tuning so coordinate manifests across all 120 groups are preloaded. Persist with
    ``repo.save_config()`` on create so re-opens and forked workers inherit it.
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


def is_missing_repo(exc: icechunk.IcechunkError) -> bool:
    """Whether an ``IcechunkError`` means the repo is genuinely ABSENT.

    Icechunk reports a missing repository as "the repository doesn't exist". Every other
    ``IcechunkError`` — auth, throttling, timeout, real corruption — must be told apart from
    absence, because absence means create and anything else means surface the error.
    Conflating them sends a transient failure down the create path, where it resurfaces as a
    dirty-prefix ``CorruptedStoreError`` naming the wrong problem.

    Lives here rather than in the inference layer that first needed it: it is a fact about
    Icechunk repositories, and the storage layer cannot import upward to reach it.
    """
    msg = str(exc).lower()
    return "doesn't exist" in msg or "does not exist" in msg


def open_repo(
    store_path: str,
    max_concurrent_requests: int | None = None,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    scatter_initial_credentials: bool = False,
) -> icechunk.Repository:
    """Open an EXISTING Icechunk repository; raise if it is not there.

    The counterpart to :func:`open_or_create_repo`, and the right call whenever the store is
    known to exist — writing a marker onto a store just written, say. Saying "this must
    already be there" makes a missing store an error rather than a silent creation, the
    stronger statement wherever it is true.
    """
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
            # Say what the state IS; do not instruct a delete. Reaching here means the prefix
            # holds objects but no readable repository — a writer interrupted between
            # creating the repo and committing its schema — and it may still hold chunks
            # somebody wants, so deleting is the operator's judgement to make.
            raise CorruptedStoreError(
                f"Store {store_path} holds objects but no readable repository — a writer was "
                f"interrupted between creating it and committing its schema. If nothing has "
                f"been successfully ingested here, clear the prefix and re-run; if it may "
                f"hold data, inspect it before deleting anything."
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

    Creates ONLY on proven absence — see :func:`is_missing_repo`. Every other failure of the
    open leg is re-raised unchanged, because routing one into the create leg is actively
    harmful: the create trips Icechunk's clean-prefix rule and surfaces as a
    ``CorruptedStoreError`` naming a perfectly healthy store and advising its deletion, and
    it is DETERMINISTIC from then on. That last part does the damage. Callers wrap these
    writes in a retry-everything loop, so a transient open failure is survivable — but once
    the repo exists every retry's create fails identically and the retry cannot escape, so a
    momentary blip becomes a hard failure reported as corruption with the real error
    destroyed.
    """
    try:
        return open_repo(
            store_path,
            max_concurrent_requests,
            get_credentials=get_credentials,
            region=region,
            scatter_initial_credentials=scatter_initial_credentials,
        ), False
    except FileNotFoundError:
        # Unambiguous absence: a local path that is not there, and zarr's
        # GroupNotFoundError, which subclasses this.
        pass
    except icechunk.IcechunkError as exc:
        if not is_missing_repo(exc):
            raise
    # Outside the `try` deliberately, so a failure from the create leg cannot be caught by
    # this function's own handlers and re-attributed.
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

    Icechunk has no destructive "undo": a rollback is a non-fast-forward ``reset_branch``
    re-pointing the branch at an older snapshot. The ``n`` dropped snapshots are not deleted
    — they stay reachable by id until expiry/garbage collection — so the rollback is itself
    reversible by resetting back to the old HEAD id.

    Args:
        store_path: Local path or S3 URI of the store.
        n: Number of commits to drop from the tip. Must be >= 1 and leave at least one
            snapshot (you cannot roll back past the root/init commit).
        branch: Branch to reset. Defaults to ``"main"``.
        get_credentials: Optional credential callback (see :func:`_create_storage`).
        region: Optional S3 region override.
        dry_run: When True, resolve and return the target snapshot id without moving the
            branch — useful for eyeballing the target first.

    Returns:
        The id of the snapshot the branch now points at, or *would* point at under
        ``dry_run``.

    Raises:
        ValueError: If ``n < 1`` or ``n`` would drop the entire history.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    repo = open_repo(store_path, get_credentials=get_credentials, region=region)

    # ancestry is newest-first: history[0] is the current HEAD, history[n] the snapshot n
    # commits back, and the final entry the repo's root commit.
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


# Sentinel: caller did not pass ``chunks``, so let ``xr.open_zarr`` apply its own default (one
# dask block per on-disk chunk). Distinct from ``chunks=None``, itself a meaningful value
# (open zarr-lazy with no dask graph).
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

    ``chunks`` is forwarded to :func:`xarray.open_zarr` only when set; the default builds one
    dask task per on-disk chunk, fine for tile-scale stores but on a CONUS-scale embeddings
    store the band axis alone is 32 chunks and the graph is ``n_time x n_y x n_x x 32``
    tasks, large enough that even a lazy ``isel``/``sel`` OOMs while manipulating the graph.
    Pass ``chunks=None`` to open zarr-lazy with no dask graph: slicing is then pure metadata
    and chunks are read only when ``.values`` is pulled — right for interactive or selective
    reads of large stores.

    ``group`` selects a Zarr group within the store (the global store's per-zone layout);
    ``None`` reads the root. Readers should target a single group and never open the whole
    120-group repo as a datatree (~200x slower — ADR-008 D5).
    """
    repo = open_repo(store_path, get_credentials=get_credentials, region=region)
    session = repo.readonly_session(branch="main")
    if isinstance(chunks, _ChunksUnset):
        return xr.open_zarr(session.store, consolidated=False, group=group)
    return xr.open_zarr(session.store, consolidated=False, chunks=chunks, group=group)


def _assert_no_committed_dates(repo: "icechunk.Repository", store_path: str) -> None:
    """Refuse to adopt a repo that already holds committed dates.

    The last thing standing between a failed date probe and a destroyed mosaic. The create
    path adopts an existing repo and writes ``mode="w"``, right for the empty shell an
    interrupted first write leaves behind and catastrophic for a store with data in it. Which
    one it is comes from :func:`get_existing_dates`, a network read; this asks the repo
    directly, from the session about to be overwritten.

    Reads the time axis, not the manifest or attrs: dates are what the caller claimed were
    absent, so dates are what gets checked. A repo with no root group at all is the canonical
    thing this path adopts — an interrupted first write commits the repo's initialization
    snapshot and nothing else — so that reads as zero dates rather than an error.

    Raises:
        StoreHoldsCommittedDataError: If the repo holds one or more committed dates. A
            dedicated type because ``cleanup_on_failure`` must NOT delete this store.
    """
    try:
        root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    except FileNotFoundError:  # GroupNotFoundError: initialized, never written
        return
    time_array = root.get("time")
    n_dates = 0 if time_array is None else int(cast("zarr.Array", time_array).shape[0])
    if n_dates:
        raise StoreHoldsCommittedDataError(
            f"Refusing to create over {store_path}: it already holds {n_dates} committed date(s). "
            "The caller reached the create path, so its date probe reported none — which means the "
            "probe failed rather than the store being empty. Retry; this store is intact."
        )


@cleanup_on_failure
def _write_new(
    store_path: str,
    data: xr.Dataset,
    encoding: dict[str, Any] | None,
    message: str,
    *,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
) -> None:
    """Write ``data`` as a store's first commit, ADOPTING a repo an earlier attempt left.

    ``open_or_create_repo``, not ``_create_repo``: an interrupted first write leaves a repo
    holding only its initialization snapshot, and creating unconditionally then trips
    Icechunk's clean-prefix rule and raises ``CorruptedStoreError`` on EVERY retry, making a
    momentary failure permanent — the retry cannot escape an error that is deterministic and
    of its own making. Adopting instead means an interruption costs the work in flight and
    nothing more.

    Reached only when the store holds no dates (see :func:`write_dataset`), so writing
    ``mode="w"`` over whatever the interrupted attempt staged loses no committed data; a
    store WITH dates takes the append path and is never handed here. That premise is
    RE-CHECKED rather than trusted, because it is the difference between adopting an empty
    shell and destroying a finished mosaic, and the caller establishes it with a probe that
    has to read the network to answer.
    """
    repo, is_new = open_or_create_repo(store_path, get_credentials=get_credentials, region=s3_region)
    if not is_new:
        # The probe reads the network, so it can fail without answering. An unanswered probe
        # must not reach `cleanup_on_failure`'s generic handler, which would delete the whole
        # prefix — possibly another writer's committed store — on the strength of not
        # knowing. Deletion requires POSITIVE evidence the prefix is ours.
        try:
            _assert_no_committed_dates(repo, store_path)
        except StoreHoldsCommittedDataError:
            raise
        except Exception as exc:
            raise InconclusiveStoreProbeError(
                f"Could not establish whether {store_path} holds committed data, so it will not be "
                f"created over or cleaned up. Retry once the store is reachable."
            ) from exc
        logger.info("Adopting existing repo at %s — an earlier attempt left it uncommitted", store_path)
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

    ``to_icechunk`` overwrites root attrs with the (typically empty) ``data.attrs``,
    destroying crs, ``_manifest`` and the rest, so they are snapshotted before the write and
    restored (plus any ``update_attrs``) after. Shared by the append and region-write paths;
    ``write_kwargs`` carries the mode-specific arguments.
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
    *,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
) -> None:
    """Append data to existing store."""
    repo = open_repo(store_path, get_credentials=get_credentials, region=s3_region)
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
    repo = open_repo(store_path, get_credentials=get_credentials, region=region_name)
    session = repo.writable_session("main")

    # Committed view for padding unaligned regions, read from a *readonly* session rather
    # than ``session.store``: the lazy pad shell's graph is handed to ``to_icechunk``, which
    # under a distributed client pickles it to the workers, and a writable session refuses to
    # pickle. Pinned to the writable session's base snapshot rather than re-resolving
    # ``branch="main"``, so a concurrent commit between the two opens cannot make the shell
    # come from a different snapshot than the write base. The write target still uses the
    # writable session, which ``to_icechunk`` forks internally.
    read_session = repo.readonly_session(snapshot_id=session.snapshot_id)
    existing = xr.open_zarr(read_session.store, consolidated=False)
    # Raw zarr group on the SAME readonly session, for the zarr-direct padding shell (one
    # read task per overlapping chunk, no whole-store layer). Must be this session — same
    # pinned snapshot as ``existing``, and pickle-safe to workers.
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

    Pass ``chunks=None`` for large (e.g. CONUS-scale) stores to skip the dask task graph and
    slice lazily on metadata alone — see :func:`_open_readonly`. ``group`` selects one Zarr
    group (the global store's per-zone layout). ``get_credentials``/``region`` thread a
    credential callback and region to the repo open, for callback-only or
    non-default-region deployments.
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

    Bypasses xarray/dask entirely — use it to read large arrays into numpy without the dask
    task-graph overhead, e.g. loading a chunk of reflectance bands for inference, where each
    ``.values`` call on the xarray path would build and execute a fresh dask graph per
    variable and hold scheduler state until the dataset handle dies.

    ``max_concurrent_requests`` caps per-repo HTTP concurrency (icechunk default 256); pass
    it when many processes read one S3 prefix at once — the region merge's worker forks all
    reading one feature store — so the aggregate GET rate stays under S3's per-prefix
    ceiling. ``group`` selects one Zarr group (the global store's per-zone layout); ``None``
    returns the root. ``get_credentials``/``region`` are forwarded so a store outside the
    default S3 region, or reachable only via an explicit credential callback, can be read
    with the same options as the writer used.
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

    Same as :func:`open_store_as_zarr_group`, plus the commit the returned group is a view
    of. The snapshot ID is the store's canonical CONTENT identity — it moves on every commit
    and cannot be left stale the way a bookkeeping attribute can — so callers deciding "is
    this the same data I saw last time?" should key on it rather than on attrs a writer is
    trusted to update. Both come from ONE repo open, so the tip costs no extra round trip.

    The session is opened AT ``tip``, not at ``branch``: resolving the tip and then opening
    the branch separately lets a commit land between them, returning content from the NEW
    snapshot labelled with the OLD id — fresh data under a stale identity, precisely the
    confusion the snapshot is here to prevent.
    """
    repo = open_repo(
        store_path, max_concurrent_requests=max_concurrent_requests, get_credentials=get_credentials, region=region
    )
    tip = repo.lookup_branch(branch)
    session = repo.readonly_session(snapshot_id=tip)
    root = zarr.open_group(session.store, mode="r")
    if group is None:
        return root, tip
    member = root[group]
    if not isinstance(member, zarr.Group):
        raise ValueError(f"{group!r} is not a group in {store_path}")
    return member, tip


ASSESSED_WINDOW_ATTR = "assessed_window"

"""Root attribute naming the date range an ingest examined in full.

A month absent from a mosaic's time axis means either "the ingest looked and there was
nothing reachable" or "the ingest never got there", and without this they are
indistinguishable. The coverage gate must fail on the second and not on the first.

Written by the ingest paths themselves rather than by the completion marker, because the
gate runs BEFORE the marker on a first ingest, and again later from the fill.
"""


def record_assessed_window(
    store_path: str,
    start_date: str,
    end_date: str,
    *,
    empty_dates: int = 0,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
) -> None:
    """Record on ``store_path`` that ``start_date..end_date`` was examined in full.

    A month absent INSIDE this range is then a finding — the imagery either did not exist or
    reached no live window — rather than a gap. Absence outside it remains a gap, so widening
    a window later cannot be excused by an older, narrower assessment.

    ``empty_dates`` is observability only, unread by the gate: how many dates were examined
    and skipped as reaching no live window, which distinguishes "sparse region" from
    "something is wrong with the footprints".

    **It does not record WHICH dates were lost, deliberately.** A store's time axis only
    grows, so once a later date commits every day at or below it is closed to that store for
    good; a run never revisits such a day and a record of it unlocks no action. What survives
    a fill is the embeddings store, which already publishes per-pixel observation counts and
    a per-month coverage mask per sensor, and the mosaic this attribute sits on is deleted
    once a cell lands. See ``ingest/README.md``.

    OPENS, never creates: it runs only against a store just written. Failing here must not be
    fatal — the assessment only optimises the gate's judgement, and a store without it falls
    back to the stricter every-month-present rule.
    """
    try:
        repo = open_repo(store_path, get_credentials=get_credentials, region=s3_region)
        session = repo.writable_session("main")
        root = zarr.open_group(session.store, mode="a")
        root.attrs[ASSESSED_WINDOW_ATTR] = [start_date, end_date]
        root.attrs["assessed_empty_dates"] = int(empty_dates)
        # ``allow_empty`` because re-recording the SAME window writes no bytes and icechunk
        # refuses a commit with no changes ("cannot commit, no changes made to the session").
        # That refusal surfaced as a WARNING on healthy stores — every resumed leg that had
        # already recorded its window logged one — and a warning that fires routinely teaches
        # the reader to skip the line, including when it means something. The record is
        # idempotent, so committing it again is harmless.
        session.commit(
            f"assessed window {start_date}..{end_date} ({empty_dates} empty date(s))",
            allow_empty=True,
        )
        logger.info(f"Recorded assessed window {start_date}..{end_date} on {store_path}")
    except Exception as exc:
        logger.warning(f"Could not record assessed window on {store_path}: {exc}")


def get_existing_dates(
    store_path: str,
    group: str | None = None,
    *,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
) -> set[str]:
    """Get dates already present in a store. Returns an empty set only if it has none.

    **Fails closed.** An empty return means "this store holds no dates", and callers act on
    that: :func:`write_dataset` routes it to the CREATE path, which adopts an existing repo
    and writes ``mode="w"``. An empty set produced by a transient read failure is therefore
    not a missed optimisation but a request to overwrite committed data, so only genuine
    absence answers empty; everything else — auth, throttling, timeout, corrupt metadata, a
    decode failure — propagates.

    Absence covers both a missing repo and a repo with no root group (an interrupted first
    write), the state ``_write_new`` exists to adopt.
    """
    t0 = time.monotonic()
    logger.debug(f"Opening store: {store_path}")
    try:
        ds = _open_readonly(store_path, group=group, get_credentials=get_credentials, region=s3_region)
        dates = {str(t.values)[:10] for t in ds.time}
        ds.close()
        logger.debug(f"Store has {len(dates)} existing dates ({time.monotonic() - t0:.1f}s)")
        return dates
    except FileNotFoundError:  # GroupNotFoundError (rootless repo) subclasses this
        logger.debug(f"Store not found ({time.monotonic() - t0:.1f}s): {store_path}")
        return set()
    except icechunk.IcechunkError as exc:
        if is_missing_repo(exc):
            logger.debug(f"Repo absent ({time.monotonic() - t0:.1f}s): {store_path}")
            return set()
        raise


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

    Each argument is a fully-inclusive ``(low, high)`` coordinate range or ``None`` (= the
    full axis, omitted from the result). Both ends are matched (``coord >= low & coord <=
    high``) and the returned slices are half-open ``[start, stop)``. The bounds may be given
    in either order — they are sorted before matching — so a descending axis (northing
    typically) can be addressed either way. ``time`` accepts anything ``np.datetime64``
    understands. ``get_credentials``/``s3_region`` are forwarded to the store opener so a
    store outside the default S3 region resolves with the same options later passed to
    :func:`write_region`.

    Returns a dict of integer slices for :func:`write_region`, containing only dims with a
    non-``None`` range; an empty dict means "the whole array", which the caller should treat
    as a plain overwrite rather than a region.

    Enforces the overwrite-in-place contract. A range matching nothing (a date not in the
    store) raises ``ValueError`` — appending new coordinates is :func:`write_dataset`'s job.
    The matched coordinates must also be contiguous; a range straddling a gap (an
    out-of-order time axis) raises rather than silently widening the slice over it.
    """
    # chunks=None: only the 1-D northing/easting/time coords are read here, never pixels, so
    # skip building a per-chunk dask graph — it matters when this is called repeatedly, e.g.
    # resolving many features against a continental master.
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
            # Either order: a descending axis (northing) is often addressed high-to-low.
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

    ``region`` maps dimension names (``time``/``northing``/``easting``) to integer slices,
    typically from :func:`resolve_region`; dimensions absent from it are written in full.
    ``data`` must cover exactly the cells the region selects (same shape along each region
    dim), and its own coordinate values are dropped before writing since the store's coords
    are authoritative.

    The region need not be chunk-aligned: unaligned edges are padded out to the enclosing
    chunk boundaries and the untouched cells backfilled from the store (read-modify-write),
    so the caller never reasons about chunk boundaries. ``update_attrs`` is merged into root
    attrs after the write, which otherwise preserves them.

    Contract: every region dim must already exist in the store at the given indices. This
    overwrites committed data — see ``context_docs/design/region-writes.md`` for the
    consistency caveats when one logical region is split across multiple calls.
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

    Handed out by :func:`batched_region_writes`. Direct zarr assignment on the session's
    store — no Dask, no task graph (the avoid-Dask principle in
    ``context_docs/design/ingest-live-tile-cropping.md``): the caller computes each window to
    numpy and this places it. Windows must be mutually chunk-disjoint, which the chunk-snapped
    window derivation guarantees; two writes straddling one chunk in a session are
    unsupported.
    """

    def __init__(self, session: "icechunk.Session", store_path: str | None = None) -> None:
        #: Which store this is, for the refusals below. A session does not carry its own path,
        #: and a refusal naming only a date tells an operator nothing actionable: the remedy is
        #: to wipe and re-ingest ONE store, so the message has to say which.
        self.store_path = store_path
        #: The batch's session. Exposed because window PIXEL data at volume should be written
        #: as ``to_icechunk(win, batch.session, mode="r+", region=...)`` — placement then stays
        #: on the Dask workers that computed the pixels instead of funnelling the date's live
        #: volume through the one worker running the caller (dense-zone arithmetic in the
        #: design note). Same session, same single commit; ``write_window`` below is for
        #: volumes that comfortably fit on the calling worker.
        self.session = session
        #: The store's root group, writable. Exposed so callers can read and merge root attrs
        #: (baselines, doy, last_appended) — attr edits land in the same commit.
        self.group: zarr.Group = zarr.open_group(session.store, mode="r+")

    def append_time_slot(self, when: np.datetime64) -> int:
        """Grow the time axis by one date and return its index. Metadata only.

        Resizes the time coord and every time-dimensioned array by one slot — icechunk arrays
        are sparse, so the new slot costs no chunks until written. The session sees its own
        uncommitted resize, so window writes into the returned index work immediately.

        A date already on the axis raises :class:`~tessera_embeddings.errors.DuplicateDateError`.
        **The likeliest cause is a SECOND WRITER on this store**, not a caller bug: the ingest
        paths read the committed dates and skip what they find, so the only way a date they
        decided to write is already present is that another process committed it after that
        read. A retry bug double-stamping one date is the remaining, rarer possibility.
        """
        when_ns = np.asarray(when, dtype="datetime64[ns]")
        existing = read_time_values(self.group)
        if (existing == when_ns).any():
            raise DuplicateDateError(
                f"date {when_ns} is already on the time axis of {self.store_path or '<unknown store>'}; "
                "refusing a duplicate slot. Another writer has almost certainly committed this date "
                "to this store — check for a second run ingesting the same zone/year/orbit before "
                "looking for a date-derivation bug."
            )
        # The axis is read POSITIONALLY downstream (the resampler samples by position, not by
        # timestamp), so appending a date older than one already stored silently changes which
        # observations a repaired store yields against a clean one holding the same dates.
        # Inserting in place would mean moving every array's data, so this refuses instead and
        # the caller decides — loudly, rather than committing a store nothing downstream can
        # tell is wrong.
        if len(existing) and when_ns < existing.max():
            raise NonMonotonicDateError(
                f"date {when_ns} is older than the latest date already on the time axis "
                f"({existing.max()}) of {self.store_path or '<unknown store>'}; refusing to append "
                "it out of order. The axis is sampled positionally, so an out-of-order slot changes "
                "which observations this store yields without changing anything a reader can check, "
                "and it cannot be inserted in place without moving every array's data. "
                "ACTION: this store cannot be completed — delete it and re-ingest its window in "
                "order. Reaching here means an earlier attempt gave up on this date and then "
                "committed a later one, so the date is now unreachable. Every cause a leg may give "
                "up for is meant to be DETERMINISTIC — it recomputes to the same verdict, so a "
                "re-offer skips again harmlessly. This firing means one was not: find which "
                "transient failure is being read as a permanent one."
            )
        t_index = len(existing)
        time_arr = self.group["time"]
        assert isinstance(time_arr, zarr.Array)
        time_arr.resize((t_index + 1,))
        time_arr[t_index] = when_ns.astype("int64")  # int64 ns per TIME_ENCODING
        for name, arr in self.group.arrays():
            # zarr v3 metadata carries dimension_names; every engine-written store is v3, and
            # a v2 array would predate the convention entirely.
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

    The per-date write unit of the live-window ingest path: everything done through the
    yielded :class:`RegionWriteBatch` lands atomically in a single snapshot on exit. On an
    exception nothing commits — the abandoned session is invisible to readers, so a retry
    starts clean from the last committed state.

    There is no Dask graph here at all. For a single arbitrary (unaligned) region overwrite
    use :func:`write_region`, which pads to chunk boundaries; this path requires
    chunk-disjoint windows and one commit is the point.
    """
    repo = open_repo(store_path, get_credentials=get_credentials, region=s3_region)
    session = repo.writable_session("main")
    yield RegionWriteBatch(session, store_path)
    # Timed because the commit (manifest + snapshot writes) is serial per-date work no fleet
    # width can compress, and the pipeline instrumentation needs it separable from the window
    # computes it follows. DEBUG, so one line per DATE on the ingest path: the duration is
    # already inside the caller's per-date/per-batch line, and every ingest task runs on a
    # Dask worker whose Prefect logs are shipped to the orchestrator API.
    commit_started = time.monotonic()
    session.commit(message)
    logger.debug("Committed '%s' in %.1fs", message, time.monotonic() - commit_started)


#: Fan-in of the tree reduction that merges window changesets back into the session. Shared by
#: both write paths so they reduce identically — the overlapped path merges every window's
#: changesets in one reduction, the sequential path one window's at a time, and a difference
#: here would show up as a behaviour difference between them.
_MERGE_SPLIT_EVERY = 8


def _window_slice(day_ds: xr.Dataset, window: "tuple[int, int, int, int]", drop: "list[str]") -> xr.Dataset:
    """The date's dataset restricted to one window, coords dropped for a region write."""
    y0, y1, x0, x1 = window
    return day_ds.isel(northing=slice(y0, y1), easting=slice(x0, x1)).drop_vars(drop)


def _window_region(window: "tuple[int, int, int, int]", time_index: int) -> dict:
    """The zarr region one window occupies at a date's time index."""
    y0, y1, x0, x1 = window
    return {
        "time": slice(time_index, time_index + 1),
        "northing": slice(y0, y1),
        "easting": slice(x0, x1),
    }


def _write_windows_overlapped(
    session: "icechunk.Session",
    writes: "list[tuple[xr.Dataset, list[tuple[int, int, int, int]], int, list[str]]]",
) -> bool:
    """Write every window of one or more dates as ONE dask compute on one forked session.

    The sequential path issues one ``to_icechunk`` per window and runs each graph to
    completion before the next starts, so a date costs the SUM of its windows' critical paths
    while the fleet works on one window at a time. This lifts icechunk's own dask sequence
    (fork → lazy stored arrays → merge reduction) one level so all windows share a single
    compute and overlap across the fleet. The windows' chunk-disjointness, enforced by the
    caller's alignment guard, is what makes the merged changesets conflict-free — the same
    property the sequential path relies on for one-commit-per-date.

    ``writes`` carries ``(day_ds, windows, time_index, drop)`` per date. Several dates in one
    call is the same induction applied once more: distinct time indices make the dates' chunk
    writes mutually disjoint exactly as windows within a date are, so they share the single
    fork, compute and merge. What dates cannot share is a session each — every date's append
    resizes the time axis, so sibling sessions forked from one snapshot conflict on array
    METADATA even though their chunk data never overlaps. A multi-date batch is therefore one
    session and one commit by construction, not by preference.

    Returns ``False`` without computing anything when the icechunk internals this lifts are
    unavailable or have drifted, so the caller falls back to the sequential path. Metadata
    writes performed before such a fallback change no array values (the windows are region
    writes into existing arrays), so the sequential rewrite of the same windows is safe. The
    two paths are byte-identical by test; drift degrades to the shipped behaviour, never to a
    failure.
    """
    try:
        from icechunk.dask import session_merge_reduction
        from icechunk.xarray import _XarrayDatasetWriter
    except ImportError as exc:
        logger.warning("Overlapped window write unavailable (%s); writing sequentially", exc)
        return False

    started = time.monotonic()
    n_windows = sum(len(windows) for _, windows, _, _ in writes)
    try:
        writers: list = []
        for day_ds, windows, time_index, drop in writes:
            for window in windows:
                writer = _XarrayDatasetWriter(
                    _window_slice(day_ds, window, drop),
                    store=session.store,
                    safe_chunks=True,
                    # Same memory rationale as the sequential path below: each write task
                    # carries one store chunk, not one whole load block.
                    align_chunks=True,
                )
                writer._open_group(group=None, mode="r+", append_dim=None, region=_window_region(window, time_index))
                writer.write_metadata(None)
                writer.write_eager()
                writers.append(writer)

        # ONE fork for all windows of all dates: every lazy region write lands its
        # changeset in the same fork, and one merge returns them to the session together.
        fork = session.fork()
        stored: list = []
        for writer in writers:
            if not writer.writer.sources:
                continue  # nothing lazy in this window; write_eager covered it
            writer.writer.targets = [
                zarr.open_array(fork.store, path=target.path, mode="a") for target in writer.writer.targets
            ]
            stored.extend(
                writer.writer.sync(
                    compute=False,
                    chunkmanager_store_kwargs={"load_stored": False, "return_stored": True},
                )
            )
    except (AttributeError, TypeError) as exc:
        logger.warning("Overlapped window write failed to assemble (%s); writing sequentially", exc)
        return False

    if stored:
        # The single compute: every window's loads, masks and chunk writes in one graph,
        # reduced to one mergeable changeset.
        session.merge(session_merge_reduction(stored, split_every=_MERGE_SPLIT_EVERY))
    # DEBUG, not INFO: the chattiest line this path produces, once per write, and Prefect
    # ships task logs to the orchestrator API from whichever Dask worker ran the task, so it
    # scales with total worker count. The timing is already in the caller's per-batch line.
    logger.debug(
        "Parallel window compute: %d window(s) across %d date(s) in one graph: %.1fs",
        n_windows,
        len(writes),
        time.monotonic() - started,
    )
    return True


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
    parallel_windows: bool = False,
) -> None:
    """Write ONE date's live windows into a mosaic store, one commit for the date.

    The single-date form of :func:`write_days_windows` and the everyday entry point: one call
    is one date is one snapshot, the granularity the resume machinery
    (``get_existing_dates`` + gap backfill) and the retry story are built around. This
    delegates verbatim; see :func:`write_days_windows` for everything else.
    """
    # This function's own contract, checked here so the error names the caller's mistake
    # ("per call") rather than the batch form's ("per entry").
    if day_ds.sizes["time"] != 1:
        raise ValueError(f"write_day_windows writes one date per call; got time size {day_ds.sizes['time']}")
    write_days_windows(
        store_path,
        [(day_ds, windows)],
        roi=roi,
        manifest=manifest,
        baselines=baselines,
        tile_id=tile_id,
        crs=crs,
        chunks=chunks,
        get_credentials=get_credentials,
        s3_region=s3_region,
        parallel_windows=parallel_windows,
    )


def write_days_windows(
    store_path: str,
    days: "list[tuple[xr.Dataset, list[tuple[int, int, int, int]]]]",
    *,
    roi: Any,  # noqa: ANN401 — duck-typed; see write_day_windows
    manifest: IngestManifest | None,
    baselines: dict[str, int],
    tile_id: str,
    crs: str,
    chunks: dict[str, int],
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
    parallel_windows: bool = False,
) -> None:
    """Write one or more dates' live windows into a mosaic store, ONE commit for the batch.

    ``parallel_windows`` submits every window of every date as a single dask compute
    (:func:`_write_windows_overlapped`) instead of one blocking compute per window, so the
    critical paths overlap across the fleet rather than summing — and with several dates, one
    date's straggling reads backfill with another date's work. Both paths produce identical
    stores (pinned by test), and when the overlapped machinery is unavailable the sequential
    path runs regardless, so the flag can never fail a write that would otherwise succeed.

    The batch is one commit BY CONSTRUCTION: each date's append resizes the time axis, so
    per-date sessions forked from one snapshot would conflict on array metadata even though
    their chunk data is disjoint. The atomicity unit is therefore the batch — a failure
    commits none of its dates, the retry starts clean from the last committed state, and
    ``get_existing_dates`` sees exactly the committed dates either way. Dates must arrive in
    strictly increasing order, because their append order IS the time axis's order.

    The cropped counterpart of :func:`write_dataset`: same bookkeeping contract, write volume
    proportional to live area instead of extent. Each ``day_ds`` is that date's full-extent
    LAZY dataset (``time`` size 1), and each chunk-disjoint ``(y0, y1, x0, x1)`` window from
    ``ingest.live_windows`` is written as a ``to_icechunk`` region on one shared session, so
    pixels flow from the Dask workers that computed them and are never materialised on the
    caller (dense-zone arithmetic in the design note).

    A missing store is seeded all-fill with an EMPTY time axis via
    :func:`~.empty_store.create_empty_store` (schema-only, so its cost is independent of
    extent, with the same attr set :func:`write_dataset` creates); every date, the first
    included, then appends its time slot atomically WITH its windows and merges attrs exactly
    as the append path does (baselines union, doy concat, ``last_appended`` bump). The
    manifest is validated BEFORE anything is written — the per-append structural gate must
    not be lost to batching.
    """
    from tessera_embeddings.storage.empty_store import create_empty_store  # local: storage-internal, avoids cycle

    if not days:
        return  # nothing to write; a commit recording nothing would be noise

    whens: list[np.datetime64] = []
    _cy, _cx = chunks["northing"], chunks["easting"]
    height, width = roi.height, roi.width
    for day_ds, windows in days:
        if day_ds.sizes["time"] != 1:
            raise ValueError(f"write_days_windows writes one date per entry; got time size {day_ds.sizes['time']}")
        # The windowed write does NOT realign the producer's blocks to the store's chunks
        # (see the to_icechunk call below), which is only sound while every store chunk a
        # window covers is written WHOLLY by that window. Checked rather than trusted: a
        # misaligned window would either be rejected by mode="r+" deep in the write or, worse,
        # straddle a chunk with a neighbour and make the result depend on write order. Ends
        # are exempt — the array's own last chunk is short.
        for y0, y1, x0, x1 in windows:
            if y0 % _cy or x0 % _cx or (y1 % _cy and y1 != height) or (x1 % _cx and x1 != width):
                raise ValueError(
                    f"window ({y0}, {y1}, {x0}, {x1}) is not aligned to the store's "
                    f"({_cy}, {_cx}) chunks; the windowed write cannot straddle a chunk"
                )
        whens.append(np.asarray(day_ds.time.values, dtype="datetime64[ns]")[0])
    if any(b <= a for a, b in itertools.pairwise(whens)):
        raise ValueError(f"batch dates must be strictly increasing; got {[str(w)[:10] for w in whens]}")

    day_ds = days[0][0]  # schema donor for seeding; every date shares the store's schema
    date_str = str(whens[0])[:10]

    # Seed with an EMPTY time axis so the first date takes the same atomic append+windows
    # commit as every other date. Seeding times=[when] would commit the date BEFORE its
    # pixels: a crash between the two leaves an all-fill timestep that get_existing_dates
    # reports as ingested, so the retry hits the duplicate-date guard and the STAC dedupe
    # filters the date forever. Existence is probed on the repo, not the (possibly empty) axis.
    needs_seed = False
    try:
        # Probe the ROOT GROUP, not just the repo: _create_repo creates the repo before
        # create_empty_store writes and commits the schema, so a crash in between leaves a
        # rootless repo. Probing the repo alone would find it, skip seeding, and leave every
        # retry failing to open the missing group — wedged forever. (Same failure the
        # global-store seeder hit; see its GroupNotFoundError handling.)
        open_store_as_zarr_group(store_path, get_credentials=get_credentials, region=s3_region)
    except icechunk.IcechunkError as exc:
        # ONLY genuine absence means "seed me". Treating every IcechunkError as absence sends
        # an auth failure, a throttle, a timeout or real corruption into the create path,
        # where it resurfaces as a dirty-prefix CorruptedStoreError, masking the real cause
        # and defeating the write retry's ability to report it.
        if not is_missing_repo(exc):
            raise
        needs_seed = True
    except (FileNotFoundError, KeyError, zarr.errors.GroupNotFoundError):
        needs_seed = True

    if needs_seed:
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
            # create_empty_store open its own would use the default storage config, so the
            # probe above and every write below would honour a callback or a non-default
            # region while the one-time seed silently did not, failing the first cropped date
            # of any such deployment.
            #
            # open_or_create, NOT create: the probe above fires alike for a MISSING repo and
            # an existing-but-ROOTLESS one (GroupNotFoundError subclasses FileNotFoundError).
            # Creating unconditionally would hit Icechunk's clean-prefix rule on the rootless
            # case and raise CorruptedStoreError on every retry — wedging exactly the crash
            # window this recovery exists for.
            repo=open_or_create_repo(store_path, get_credentials=get_credentials, region=s3_region)[0],
        )

    total_windows = sum(len(w) for _, w in days)
    if len(days) == 1:
        # A stable interface: commit-cadence tooling and log queries parse `date <iso>:` out
        # of snapshot messages.
        message = f"date {date_str}: {total_windows} live window(s)"
    else:
        message = f"dates {date_str}..{str(whens[-1])[:10]} ({len(days)} dates): {total_windows} live window(s)"

    with batched_region_writes(
        store_path,
        message=message,
        get_credentials=get_credentials,
        s3_region=s3_region,
    ) as batch:
        mixed = manifest.validate_against(extract_manifest(dict(batch.group.attrs)), store_path) if manifest else []
        # Slots are appended for every date up front — the session sees its own uncommitted
        # resizes, so each date's windows write into its own index. Appends and windows land
        # in the ONE commit together, preserving the no-date-before-its-pixels invariant at
        # batch granularity.
        writes: list[tuple[xr.Dataset, list[tuple[int, int, int, int]], int, list[str]]] = []
        for (one_ds, one_windows), when in zip(days, whens, strict=True):
            t = batch.append_time_slot(when)
            drop = [c for c in ("time", "northing", "easting") if c in one_ds.coords]
            writes.append((one_ds, one_windows, t, drop))
        attrs = batch.group.attrs
        merged = dict(cast("dict", attrs.get("baselines_applied", {})))
        merged.update(baselines)
        attrs["baselines_applied"] = merged
        attrs["doy"] = list(cast("list", attrs.get("doy", []))) + compute_doy(np.array(whens)).tolist()
        attrs["last_appended"] = utcnow_iso()
        if mixed:
            # Same commit as the dates it describes; a union, since one resume writes many.
            prior = cast("list", attrs.get(MIXED_CODE_IDENTITIES_ATTR, []))
            attrs[MIXED_CODE_IDENTITIES_ATTR] = sorted(set(prior) | set(mixed))
        if parallel_windows and _write_windows_overlapped(batch.session, writes):
            return  # normal context exit: the batched commit below still runs
        for one_ds, one_windows, t, drop in writes:
            for i, window in enumerate(one_windows, 1):
                window_started = time.monotonic()
                to_icechunk(
                    _window_slice(one_ds, window, drop),
                    batch.session,
                    mode="r+",
                    region=_window_region(window, t),
                    # ON for memory, not graph size. Dropping it shrinks the graph a little
                    # and ran ~4% faster, but each write task then carries a whole load block
                    # instead of one store chunk, which pushed workers over their spill
                    # threshold: peak spill 3.19 GiB across ~30% of scheduler samples, against
                    # zero with it on. Spill scales badly here, so the ~4% is not worth it.
                    align_chunks=True,
                    split_every=_MERGE_SPLIT_EVERY,
                )
                # Each window is a blocking compute, so these lines ARE the write pipeline's
                # decomposition: their sum against the date's write phase says whether
                # windows serialise, and their spread says what overlap can buy.
                logger.info(
                    "Window %d/%d rows=%d..%d: %.1fs",
                    i,
                    len(one_windows),
                    window[0],
                    window[1],
                    time.monotonic() - window_started,
                )


def compute_doy(timestamps: np.ndarray) -> np.ndarray:
    """Compute day-of-year from datetime64 timestamps: an (N,) int32 array of 1-366."""
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
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    s3_region: str | None = None,
) -> None:
    """Write dataset to Icechunk Zarr store, creating or appending as needed.

    Args:
        store_path: Path to the Zarr store (local or s3://).
        data: xarray Dataset with (time, northing, easting) dimensions.
        tile_id: Tile identifier for store metadata.
        baselines: Dict mapping date strings to baseline integers.
        chunks: Chunk sizes dict with ``time``, ``northing``, and ``easting`` keys.
        manifest: Typed manifest for append-safety validation, written on create and
            validated on append.
        crs: CRS authority code (e.g. ``"EPSG:32615"``), stored in root attrs so downstream
            consumers can determine the projection.
        get_credentials: Optional credential callback for Icechunk's S3 client.
        s3_region: Optional S3 region override. Threaded through EVERY open below, not just
            the first: a store outside the default region must be read, created and appended
            the same way, or the ingest fails partway.
    """
    existing_dates = get_existing_dates(store_path, get_credentials=get_credentials, s3_region=s3_region)

    # Normalize time to nanoseconds to match TIME_ENCODING: newer pandas/xarray may produce
    # datetime64[us], which would not round-trip through the zarr encoding.
    if data.time.dtype != np.dtype("datetime64[ns]"):
        data = data.assign_coords(time=data.time.values.astype("datetime64[ns]"))

    doy = compute_doy(data.time.values)

    if existing_dates:
        # Single store open: read manifest + baseline attrs together
        existing_ds = _open_readonly(store_path, get_credentials=get_credentials, region=s3_region)
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
            get_credentials=get_credentials,
            s3_region=s3_region,
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
        _write_new(
            store_path,
            data,
            encoding,
            f"Create with {data.sizes['time']} dates",
            get_credentials=get_credentials,
            s3_region=s3_region,
        )


# =============================================================================
# Test Helpers (for verifying Icechunk transactional behavior)
# =============================================================================


def _open_writable_session(store_path: str) -> tuple[icechunk.Session, icechunk.IcechunkStore]:
    """Open a writable session. Caller must commit."""
    repo = open_repo(store_path)
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
