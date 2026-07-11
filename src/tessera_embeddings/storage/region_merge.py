"""Merge per-feature mosaics into a single master mosaic by direct Zarr chunk-writes.

The no-cluster region-merge path. A **feature** store is any store whose grid is
an exact pixel-subset of a shared master grid: same CRS and resolution, with
``northing``/``easting`` coordinate values that form a contiguous subset of the
master's, and dates that all exist in the master's ``time`` axis. Given a master
seeded with the union of the feature dates (see :mod:`.empty_store` and
:func:`gather_time_union`), this module copies each feature's pixels into the
master's positional slice, one commit per feature, then the temp per-feature
stores can be deleted.

The merge is **pure byte movement** — no compute, no Dask cluster. Each feature's
pixels are copied straight from its raw Zarr arrays into the master's positional
slice by a **process-parallel** raw-zarr chunk-write, committed once per feature.
The design (why processes not threads, the four correctness invariants, the
two-layer hang protection) is documented in
``context_docs/design/region-merge.md``; the docstrings below cover only what
each function does. The short version: forking the icechunk session per worker is
the only axis that scales (the store mutex is per session), and copying at chunk
granularity is ``O(chunks)`` versus the retired ``write_regions`` Dask graph that
made continental merges take days.

Domain-layer rules apply: stdlib logging only (no orchestrator imports), storage
via the fsspec-backed helpers (no boto3).
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import fsspec
import icechunk
import numpy as np
import psutil
import zarr

from tessera_embeddings.storage.empty_store import create_empty_store
from tessera_embeddings.storage.zarr_store import (
    DEFAULT_MANIFEST_SPLIT_SIZES,
    _open_repo,
    manifest_split,
    open_store,
    open_store_as_zarr_group,
    resolve_region,
)

if TYPE_CHECKING:
    import xarray as xr

    from tessera_embeddings.ingest.roi import ROIMetadata

logger = logging.getLogger(__name__)

_EMPTY_TIMES = np.array([], dtype="datetime64[ns]")

# Default number of worker PROCESSES for the per-feature copy. Within one process
# every chunk-write serializes on that session's store mutex (icechunk Rust) —
# diagnosed in production: 32 vs 128 threads were identical (~250 MB/s), threads
# parked in futex_wait_queue, NIC at ~10% and CPU bursting only during the
# GIL-released codec windows. The lock is PER SESSION, so the only axis that
# scales is processes: each fork is an independent session with its own lock and
# its own object-store connection pool. One process per core by default.
_DEFAULT_MAX_PROCESSES = os.cpu_count() or 8

# Threads per worker process. A few overlap S3 latency within the process's own
# fork-session lock-release windows; more just re-lengthen that one futex queue
# (the single-process plateau), so this stays small — processes do the scaling.
_DEFAULT_THREADS_PER_PROCESS = 4

# Stall detection (CPU-progress watchdog — full rationale in
# context_docs/design/region-merge.md → "Hang protection"). Kill only after CPU
# has been flat for this grace window: a large-but-progressing copy keeps
# system-wide CPU climbing and is never flagged, while a true wedge (every worker
# parked on a dead socket) trips it. Ten minutes is long enough to ride out one
# worker waiting on a slow-but-alive S3 attempt (Layer-1
# ``zarr_store._DEFAULT_OPERATION_ATTEMPT_TIMEOUT_MS`` is 180s) while the others
# keep CPU climbing.
_DEFAULT_STALL_GRACE_SEC = 600.0

# How often the coordinator samples CPU while waiting. ``concurrent.futures`` returns
# early the instant all shards finish, so this only sets the stall-detection
# resolution, not the happy-path latency.
_DEFAULT_STALL_POLL_SEC = 10.0

# Minimum busy CPU-seconds gained between samples to count as forward progress. A
# healthy pool accrues many CPU-seconds per poll window; a wedged pool ~zero. Small
# and nonzero so scheduler jitter / a stray wakeup on an otherwise-dead pool doesn't
# masquerade as progress.
_STALL_CPU_EPSILON_SEC = 1.0


def _kill_pool_workers(pool: ProcessPoolExecutor, pids: list[int]) -> None:
    """Forcibly abandon a ``ProcessPoolExecutor`` whose worker may be wedged on a dead socket.

    A normal ``shutdown(wait=True)`` / ``__exit__`` JOINS the workers, so it would
    itself hang forever on the very worker we are trying to escape. So we
    ``shutdown(wait=False, cancel_futures=True)`` to drop queued work and detach
    without joining, then ``SIGKILL`` every worker PID (captured at submit time, and
    re-read from the live ``pool._processes`` map in case the pool replaced any) so
    no orphan survives. ``SIGKILL`` (not terminate) because a process blocked in an
    uninterruptible socket syscall may ignore ``SIGTERM`` until the read returns.
    """
    pool.shutdown(wait=False, cancel_futures=True)
    live = set(pids) | set(getattr(pool, "_processes", {}) or {})
    for pid in live:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # already exited
        except OSError as e:
            logger.warning("Could not SIGKILL merge worker pid %s: %s", pid, e)


def _busy_cpu_seconds() -> float:
    """System-wide busy CPU-seconds (``user+system`` across all cores) since boot.

    This is the merge's forward-progress signal. We read it host-wide rather than
    per-worker-PID because the merge owns a dedicated container: during a feature copy
    the coordinator is blocked waiting, so essentially all CPU on the box is the
    workers'. A healthy copy keeps every worker burning CPU in the GIL-released
    encode/store-set windows, so this climbs continuously; workers all wedged in the
    kernel on a dead socket accrue ~zero. System-wide also sidesteps the per-PID
    bookkeeping (child threads, a worker reaped mid-poll) that a sum-over-PIDs would
    need. (Trade-off: on a shared host other tenants' CPU would mask a stall — run
    the merge on a dedicated machine/container, or rely on the optional hard
    ceiling.)
    """
    t = psutil.cpu_times()
    return t.user + t.system


def _wait_with_stall_detection(
    futures: list[concurrent.futures.Future],
    *,
    grace_sec: float,
    poll_sec: float,
    hard_timeout_sec: float | None,
    log: logging.Logger | logging.LoggerAdapter,
    feature_path: str,
) -> tuple[set, set]:
    """Wait for worker futures, killing them only after sustained ZERO-CPU progress.

    Polls every ``poll_sec``. Tracks system-wide busy CPU (:func:`_busy_cpu_seconds`);
    whenever it advances by at least :data:`_STALL_CPU_EPSILON_SEC` the stall clock
    resets. Only when CPU has been flat for ``grace_sec`` continuously does this treat
    the pool as wedged and return with the unfinished futures still ``pending`` (the
    caller kills them). A large-but-healthy copy keeps CPU climbing, so it is never
    flagged no matter how long it runs — which is the whole point over a fixed budget.

    ``hard_timeout_sec`` is an optional absolute ceiling regardless of CPU progress
    (``None`` disables it, leaving stall detection as the only guard); it exists as an
    operator escape hatch and to keep tests bounded. Returns ``(completed, pending)``
    exactly like :func:`concurrent.futures.wait`, so a non-empty ``pending`` means the
    caller should kill and raise.
    """
    t_start = time.monotonic()
    last_cpu = _busy_cpu_seconds()
    t_last_progress = t_start
    while True:
        completed, pending = concurrent.futures.wait(futures, timeout=poll_sec)
        if not pending:
            return completed, pending

        now = time.monotonic()
        if hard_timeout_sec is not None and now - t_start >= hard_timeout_sec:
            log.warning(
                "Merge of %s hit the %.0fs hard ceiling with %d shard(s) unfinished",
                feature_path,
                hard_timeout_sec,
                len(pending),
            )
            return completed, pending

        cpu = _busy_cpu_seconds()
        if cpu - last_cpu >= _STALL_CPU_EPSILON_SEC:
            last_cpu = cpu
            t_last_progress = now
            continue

        stalled_for = now - t_last_progress
        if stalled_for >= grace_sec:
            log.warning(
                "Merge of %s: system CPU flat for %.0fs (>= %.0fs grace) with %d shard(s) "
                "unfinished — workers appear wedged (dead socket), killing",
                feature_path,
                stalled_for,
                grace_sec,
                len(pending),
            )
            return completed, pending


def _times_of(ds: xr.Dataset) -> np.ndarray:
    """The ``time`` coord of an open dataset as ``datetime64[ns]`` (empty if none)."""
    if "time" not in ds.coords and "time" not in ds.dims:
        return _EMPTY_TIMES
    return np.asarray(ds["time"].values, dtype="datetime64[ns]")


def read_store_times(store_path: str) -> np.ndarray:
    """Return the ``time`` coordinate of a store as a ``datetime64[ns]`` array.

    Returns an empty array if the store does not exist or carries no time axis —
    a feature ingest that produced nothing (e.g. every date filtered for low
    coverage, or an orbit with no passes) simply contributes no dates, tolerating
    a missing store the same way :func:`~.zarr_store.get_existing_dates` does.
    """
    try:
        # Metadata-only (chunks=None): only the time coord is read, never pixels.
        ds = open_store(store_path, chunks=None)
    except (FileNotFoundError, icechunk.IcechunkError):
        # Absent store — contributes no dates to the union (matches get_existing_dates).
        return _EMPTY_TIMES
    except Exception as e:
        logger.warning("Could not read times from %s: %s", store_path, e)
        return _EMPTY_TIMES
    try:
        return _times_of(ds)
    finally:
        ds.close()


def gather_time_union(store_paths: list[str]) -> np.ndarray:
    """Sorted, de-duplicated union of the ``time`` values across stores.

    This is the date axis the master store must be seeded with so that every
    feature's dates already exist as coordinates to overwrite. Missing / empty
    feature stores contribute nothing. Returns ``datetime64[ns]``.
    """
    arrays = [t for p in store_paths if (t := read_store_times(p)).size]
    if not arrays:
        return _EMPTY_TIMES
    # Inputs are already datetime64[ns] (read_store_times → _times_of), so
    # concatenate+unique preserves the dtype — no re-cast needed.
    return np.unique(np.concatenate(arrays))


def read_master_axes(master_path: str) -> tuple[np.ndarray, set[str]]:
    """Read a master store's ``time`` axis and data-variable names in one open.

    The merge needs both the master's sorted date axis (to map feature dates to
    positional indices) and its variable set (to drop feature-only auxiliary
    bands). These are constant for a master once seeded, so a caller looping
    over many features should read them once and pass them into
    :func:`merge_feature_into_master` rather than paying this open per feature.

    Opened metadata-only (``chunks=None``) so a continental master's per-chunk dask
    graph is never built — only the ``time`` coord and variable names are read.
    """
    mds = open_store(master_path, chunks=None)
    try:
        return _times_of(mds), {str(v) for v in mds.data_vars}
    finally:
        mds.close()


def _feature_master_indices(
    feature_path: str,
    master_path: str,
    feature_times: np.ndarray,
    master_times: np.ndarray,
) -> np.ndarray:
    """Map a feature's dates (in feature-array order) to their master positional indices.

    Returns an array ``m_idx`` where ``m_idx[f]`` is the master time index of the
    feature's ``f``-th date — pairing feature array position ``f`` directly with
    its master slot, so the copy loop reads ``feature[f]`` and writes ``master[m]``
    with no sorting or run-coalescing.

    The master is seeded from the union that includes this feature, so every
    feature date must be present; a missing one is a seeding bug, surfaced here as
    a ``ValueError`` rather than a silent index mismatch. Every date being present
    means ``searchsorted`` lands on its exact master index.

    ``searchsorted`` assumes ``master_times`` is sorted ascending. ``gather_time_union``
    returns a sorted axis, but the empty-store helpers persist ``times`` as-is and
    ``read_master_axes`` returns store order, so an unsorted master would silently map
    dates to the wrong rows — guarded here rather than trusted.
    """
    if master_times.size > 1 and (master_times[1:] < master_times[:-1]).any():
        raise ValueError(
            f"Master axis at {master_path} is not sorted ascending; date→index mapping would be wrong. "
            "Seed the master with a sorted time axis (gather_time_union returns one)."
        )
    present = np.isin(feature_times, master_times)
    if not present.all():
        missing = [str(d)[:10] for d in feature_times[~present]]
        raise ValueError(
            f"Feature {feature_path} has date(s) absent from the master axis at {master_path}: "
            f"{missing}. The master must be seeded from the feature date union."
        )
    return np.searchsorted(master_times, feature_times)


def _shared_feature_vars(fds: xr.Dataset, feature_path: str, master_path: str, master_vars: set[str]) -> list[str]:
    """Variables present in both the feature store and the master.

    Restricting to the intersection keeps a feature-only band from blowing up a
    write against a master seeded without it; this sheds only genuine schema
    drift. An empty intersection means a wrong store mapping or schema drift,
    raised here rather than silently no-oping with a success report.
    """
    feature_vars = [str(v) for v in fds.data_vars if v in master_vars]
    if not feature_vars:
        raise ValueError(
            f"Feature store {feature_path} shares no variables with master {master_path} "
            f"(feature has {sorted(map(str, fds.data_vars))}, master has {sorted(master_vars)}). "
            "Check the store-to-master mapping."
        )
    return feature_vars


def _chunk_blocks(start: int, stop: int, chunk: int) -> list[slice]:
    """Tile the master-absolute span ``[start, stop)`` on the master chunk grid.

    Each returned slice lies within a single master chunk (the first and last may
    be partial — a feature edge falling mid-chunk), so a write to any one block
    touches exactly one Zarr chunk. Blocks are therefore mutually chunk-disjoint,
    which is what makes the per-feature thread fan-out race-free.
    """
    blocks: list[slice] = []
    pos = start
    while pos < stop:
        chunk_hi = (pos // chunk + 1) * chunk
        hi = min(stop, chunk_hi)
        blocks.append(slice(pos, hi))
        pos = hi
    return blocks


def _fill_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask of the cells that are this array's nodata fill (True = fill).

    The fill is dtype-resolved exactly as :mod:`.empty_store` seeds it: ``NaN``
    for floating-point variables, ``0`` for integer variables (``0`` is the nodata
    sentinel the ingest paths write for absent/masked pixels — see that module's
    "Fill value" note). The merge uses this to copy only a feature's *real* pixels,
    never its out-of-polygon fill (see :func:`_copy_units_in_process`).
    """
    if np.issubdtype(values.dtype, np.floating):
        return np.isnan(values)
    return values == 0


def _copy_units_in_process(
    fork: icechunk.ForkSession,
    feature_path: str,
    units: list[tuple[str, int, int, slice, slice, slice, slice]],
    threads: int,
    max_concurrent_requests: int | None = None,
) -> icechunk.ForkSession:
    """Write one shard of copy units into the master via a forked session, in a worker process.

    Runs in a child process (``ProcessPoolExecutor``). ``fork`` is this process's own
    pickled :class:`icechunk.ForkSession` — an independent session with its own store
    mutex and object-store pool, so it does not contend with the other processes'
    forks. Each unit is ``(var, m_time, f_time, dst_y, dst_x, src_y, src_x)``: it
    reads the feature's sub-window from the feature's raw zarr and writes its real
    (non-fill) pixels into the master's chunk-aligned block as a single chunk write.
    The chunk bytes flush to the object store from THIS process; only the changeset
    reference travels back when the parent merges the returned fork.

    The per-unit write is a **fill-masked overlay**, not a blind copy: ``where(feature
    is fill, master, feature)``, so a feature never clobbers a neighbour's pixels with
    its own out-of-polygon fill (an all-real interior block short-circuits the master
    read; the masked read-modify-write is taken only for partial-edge blocks). Full
    rationale and the order-independence/boundary-chunk argument are in
    ``context_docs/design/region-merge.md`` → "Correctness — four invariants".

    A small in-process thread pool overlaps S3 latency; all of this process's units
    touch distinct master chunks (distinct var/date/block), so the threads never
    collide. The feature group and master arrays are opened HERE (in the child): only
    the fork, paths, and integer/slice unit descriptors pickle across the boundary.

    ``max_concurrent_requests`` caps this worker's feature-read concurrency; with
    every worker reading one feature prefix, the parent passes its cap through so the
    aggregate GET rate (else up to ``n_shards`` x icechunk's default 256) stays under
    S3's per-prefix ceiling — the same throttle the master forks already honor.
    """
    feature_group = open_store_as_zarr_group(feature_path, max_concurrent_requests=max_concurrent_requests)
    master_group = zarr.open_group(fork.store, mode="r+")

    def _do(unit: tuple[str, int, int, slice, slice, slice, slice]) -> None:
        var, m_time, f_time, dst_y, dst_x, src_y, src_x = unit
        # zarr v3 stubs type Group.__getitem__ as Array | Group; these paths are
        # data-var arrays, so narrow for the positional numpy read/write below.
        src_arr = cast("zarr.Array", feature_group[var])
        dst_arr = cast("zarr.Array", master_group[var])
        src = cast("np.ndarray", src_arr[f_time, src_y, src_x])
        is_fill = _fill_mask(src)
        if is_fill.all():
            return  # nothing real to contribute in this block; leave the master as-is
        if not is_fill.any():
            # All-real block (the common case for an interior chunk fully inside the
            # polygon): the overlay degenerates to `where(False, master, src) == src`,
            # so write src directly and skip the master read. This elides ~half the S3
            # GETs on a large feature, where most chunks are interior. The fill-masked
            # read-modify-write below is still needed for partial-edge blocks, where
            # src carries out-of-polygon fill that must not clobber a neighbour.
            dst_arr[m_time, dst_y, dst_x] = src
            return
        dst = dst_arr[m_time, dst_y, dst_x]
        dst_arr[m_time, dst_y, dst_x] = np.where(is_fill, dst, src)

    if threads <= 1:
        for unit in units:
            _do(unit)
    else:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for _ in pool.map(_do, units):
                pass
    return fork


def merge_feature_into_master(
    master_path: str,
    feature_path: str,
    *,
    master_times: np.ndarray,
    master_vars: set[str],
    max_workers: int | None = None,
    threads_per_process: int | None = None,
    max_concurrent_requests: int | None = None,
    stall_grace_sec: float | None = None,
    stall_poll_sec: float | None = None,
    feature_timeout_sec: float | None = None,
    feature_retries: int = 1,
    log: logging.Logger | logging.LoggerAdapter | None = None,
) -> int:
    """Copy a feature store into the master with a process-parallel zarr region-write, one commit.

    No Dask, no cluster: the spatial slice is resolved once (the feature's extent is
    constant across dates), each date maps to one master time index by exact
    ``datetime64``, and the work is tiled to the master chunk grid into ``(var, date,
    chunk-block)`` units, sharded across worker processes, and committed once. The
    design — why processes (per-session store mutex), the disjointness/fill-overlay
    invariants, and the two-layer hang protection — is in
    ``context_docs/design/region-merge.md``; this docstring covers the call contract.

    The feature grid must be an exact pixel-subset of the master grid — "master-
    snapped": same CRS and resolution, coords a contiguous subset (the caller's
    contract, guaranteed by producing features on the master grid). What is validated
    here (raising ``ValueError``): the feature's spatial extent AND coordinates match
    the resolved master slice (to a quarter-pixel — catches a reversed/permuted/
    misaligned grid), the master axis is sorted and every feature date is present and
    distinct, the shared master variables are 3-D with matching dtypes and a single
    uniform chunk grid, and no two merged dates share a master time chunk (seed the
    master with time chunk size 1 — the simple way).

    ``master_times`` / ``master_vars`` are the master's date axis and variable set.
    They are constant for a master, so the caller reads them once via
    :func:`read_master_axes` and passes them in (this never re-reads them per
    feature — that is a continental-scale master open).

    ``max_workers`` is the worker-process count (default
    :data:`_DEFAULT_MAX_PROCESSES`, one per core); ``threads_per_process`` sizes each
    process's thread pool (default :data:`_DEFAULT_THREADS_PER_PROCESS`).
    ``max_concurrent_requests`` caps per-repo HTTP concurrency, inherited by every
    master fork AND applied to each worker's feature-store read; many processes
    hitting ONE S3 prefix can trip 503 SlowDown, so lower it (e.g. 64) to throttle
    aggregate request rate without dropping process count. ``None`` leaves icechunk's
    default (256), so behavior is unchanged unless set.

    Hang protection (see the design doc) is governed by ``stall_grace_sec`` /
    ``stall_poll_sec`` (the CPU-stall watchdog) and ``feature_retries`` (default 1,
    fresh session per retry — the region-write is idempotent). ``feature_timeout_sec``
    is an optional absolute hard ceiling regardless of CPU progress (``None`` →
    disabled, watchdog is the only guard; an operator escape hatch).

    ``log`` is the orchestrator's run logger when called from a flow (so timing
    lines surface in its UI); the module logger is used when omitted.

    Returns the number of dates written (0 if the feature store is missing/empty).
    Raises :class:`TimeoutError` if a worker stall (or the optional
    ``feature_timeout_sec`` ceiling) is hit on every attempt through
    ``feature_retries`` retries.
    """
    _log = log or logger
    n_processes = max_workers or _DEFAULT_MAX_PROCESSES
    n_threads = threads_per_process or _DEFAULT_THREADS_PER_PROCESS
    grace_sec = _DEFAULT_STALL_GRACE_SEC if stall_grace_sec is None else stall_grace_sec
    poll_sec = _DEFAULT_STALL_POLL_SEC if stall_poll_sec is None else stall_poll_sec
    t_start = time.monotonic()
    try:
        # Metadata-only open (chunks=None → no dask graph): this reads the feature's
        # time/coords/var names, never its pixels — the copy reads those via raw zarr
        # in the workers. Skips building a per-chunk dask graph on continental features.
        fds = open_store(feature_path, chunks=None)
    except (FileNotFoundError, icechunk.IcechunkError) as e:
        # ONLY a genuinely-absent store is a no-op. A corrupt, unauthorized, or
        # transiently-failing store must fail loudly: silently returning 0 would let
        # the master commit as "successful" while dropping this feature's data, after
        # which the caller may delete the temp store.
        if isinstance(e, FileNotFoundError) or "doesn't exist" in str(e):
            _log.info("Feature store %s does not exist — nothing to merge", feature_path)
            return 0
        raise
    try:
        # Feature dates in FEATURE-ARRAY order (no sort): m_idx[f] is the master
        # time index for the feature's f-th date, pairing array position to slot.
        ftimes = _times_of(fds)
        if not ftimes.size:
            _log.info("Feature store %s has no dates — nothing to merge", feature_path)
            return 0
        m_idx = _feature_master_indices(feature_path, master_path, ftimes, master_times)

        north, east = fds["northing"].values, fds["easting"].values
        feature_vars = _shared_feature_vars(fds, feature_path, master_path, master_vars)
        # Captured before the feature handle closes; compared against the master arrays
        # after the master open (the schema-drift guard). Stores carry no CF scale/
        # offset, so the xarray-reported dtype is the stored dtype.
        feature_dtypes = {v: fds[v].dtype for v in feature_vars}

        # Resolve the spatial region once (the feature's extent is constant across
        # dates). This is a metadata-only open of the master's 1-D coord arrays.
        t_resolve = time.monotonic()
        spatial = resolve_region(master_path, northing=(north.min(), north.max()), easting=(east.min(), east.max()))
        ys, xs = spatial["northing"], spatial["easting"]
        if (ys.stop - ys.start, xs.stop - xs.start) != (north.size, east.size):
            raise ValueError(
                f"Feature {feature_path} spatial extent {(north.size, east.size)} does not match the "
                f"resolved master slice {(ys.stop - ys.start, xs.stop - xs.start)}; the feature grid is "
                "not an exact pixel-subset of the master grid (it must be master-snapped)."
            )
        _log.info(
            "Merging %s → master: %d date(s), %d var(s), northing=[%.1f, %.1f], easting=[%.1f, %.1f] "
            "(resolved spatial slice in %.2fs)",
            feature_path,
            ftimes.size,
            len(feature_vars),
            north.min(),
            north.max(),
            east.min(),
            east.max(),
            time.monotonic() - t_resolve,
        )
    finally:
        # Close on every path — including the empty-feature early return and any raise
        # — so a run over many (often empty) feature stores can't leak handles.
        fds.close()

    # The master chunk grid is constant, so the units (and their round-robin
    # shards) are built ONCE and reused across any retry; only the
    # session/forks/pool are rebuilt per attempt. We open the master here for the
    # chunk shape — not create.
    chunk_repo = _open_repo(master_path, max_concurrent_requests=max_concurrent_requests)
    master_group = zarr.open_group(chunk_repo.readonly_session("main").store, mode="r")

    # Validate the merge contract against the master's actual arrays. The master_group
    # is already open for the chunk grid, so every check here is metadata-only.
    master_arrays = {var: cast("zarr.Array", master_group[var]) for var in feature_vars}
    for var, m_arr in master_arrays.items():
        # 3-D only: the copy tiles (time, northing, easting). A 4-D embedding store
        # (time, northing, easting, band) is assembled by inference.assembly, not
        # merged here — reject it with a clear message instead of an unpack error.
        if m_arr.ndim != 3:
            raise ValueError(
                f"Master {master_path} variable {var!r} has {m_arr.ndim} dims; region merge supports "
                "3-D (time, northing, easting) stores only."
            )
        # Same-name/different-dtype would let the raw positional assignment silently
        # cast (e.g. uint16 reflectance truncated into a uint8 master). Fail on drift.
        if m_arr.dtype != feature_dtypes[var]:
            raise ValueError(
                f"Variable {var!r} dtype differs between feature ({feature_dtypes[var]}) and master "
                f"({m_arr.dtype}); a raw copy would silently cast. Re-seed the master to match."
            )

    # Every shared variable must share one chunk grid: the copy tiles all variables on
    # feature_vars[0]'s grid, so a variable with a different grid could have a chunk
    # split across units sharded to different forks — a silent same-chunk race.
    grids = {var: master_arrays[var].chunks for var in feature_vars}
    if len(set(grids.values())) > 1:
        raise ValueError(
            f"Master {master_path} shared variables have non-uniform chunk grids {grids}; seed every "
            "merged variable with the same chunking so the process-parallel copy stays chunk-disjoint."
        )
    chunk_t, chunk_y, chunk_x = grids[feature_vars[0]]

    # The positional copy pairs feature index i with master-slice index i, so the
    # feature's spatial coords must match the master's over the resolved slice. The
    # extent guard above only proved equal lengths; a reversed, permuted, or half-
    # pixel-misaligned feature with the same bounds would pass it yet silently write
    # pixels into the wrong cells. Tolerance = a quarter-pixel, so float
    # representation never false-positives while any real misalignment (>= one pixel)
    # is caught. (The lengths are guaranteed equal by the extent guard.)
    for name, f_coord, m_coord in (
        ("northing", north, cast("zarr.Array", master_group["northing"])[ys]),
        ("easting", east, cast("zarr.Array", master_group["easting"])[xs]),
    ):
        pitch = abs(float(f_coord[1] - f_coord[0])) if f_coord.size > 1 else 1.0
        if not np.allclose(f_coord, m_coord, atol=pitch / 4, rtol=0):
            raise ValueError(
                f"Feature {feature_path} {name} coordinates do not match master {master_path} over the "
                "resolved slice (reversed, permuted, or misaligned grid); the positional copy would "
                "misplace pixels. The feature must be master-snapped."
            )

    # Copy units are chunk-disjoint only if no two merged dates share a master chunk —
    # the temporal half of the disjointness invariant, in two parts:
    #  - distinct dates must map to distinct master slots: a feature with duplicate
    #    dates would target one chunk from different forks. This bypasses the time-
    #    chunk guard below when the master is chunked time=1, so it is checked
    #    unconditionally.
    #  - and, when time chunk size > 1, distinct dates must fall in distinct time
    #    chunks (a master chunked time=1 satisfies this by construction).
    # Either violation means two forks writing one chunk with no conflict resolution.
    if np.unique(m_idx).size < m_idx.size:
        raise ValueError(
            f"Feature {feature_path} maps two or more dates to the same master time index (duplicate "
            "dates in the feature store); each merged date must be distinct."
        )
    if chunk_t > 1 and np.unique(m_idx // chunk_t).size < m_idx.size:
        raise ValueError(
            f"Master {master_path} has time chunk size {chunk_t} and feature {feature_path} maps "
            "two or more dates into the same time chunk; the process-parallel copy requires every "
            "merged date in its own time chunk (seed the master with time chunk size 1)."
        )

    y_blocks = _chunk_blocks(ys.start, ys.stop, chunk_y)
    x_blocks = _chunk_blocks(xs.start, xs.stop, chunk_x)

    # One copy unit per (var, date, chunk-block), as plain (str, int, slice) tuples
    # so they pickle to worker processes. Each owns a distinct master chunk
    # (distinct var → distinct array, date → distinct time-chunk, block → distinct
    # spatial chunk), so no two units — across processes or threads — write the same
    # chunk, making the merge of the forks conflict-free.
    units: list[tuple[str, int, int, slice, slice, slice, slice]] = []
    for var in feature_vars:
        for f_time, m_time in enumerate(m_idx):
            for yb in y_blocks:
                for xb in x_blocks:
                    units.append(
                        (
                            var,
                            int(m_time),
                            int(f_time),
                            yb,
                            xb,
                            slice(yb.start - ys.start, yb.stop - ys.start),
                            slice(xb.start - xs.start, xb.stop - xs.start),
                        )
                    )

    # Shard the units round-robin across the worker processes (round-robin spreads
    # each var/date evenly so no process gets a disproportionately heavy shard).
    n_shards = max(1, min(n_processes, len(units)))
    shards: list[list] = [units[i::n_shards] for i in range(n_shards)]

    commit_msg = f"Merge {feature_path} ({ftimes.size} dates, {len(feature_vars)} vars)"

    def _run_attempt() -> tuple[float, float]:
        """One full copy+merge+commit attempt on a fresh session. Returns (t_copy, t_merge).

        Raises ``TimeoutError`` if the workers stall (aggregate CPU flat for the grace
        window, or the optional ``feature_timeout_sec`` ceiling) after forcibly killing
        the wedged pool, or re-raises a worker exception. A fresh ``writable_session`` +
        forks every attempt so a retry never reuses a session whose forks are entangled
        with a dead worker. Layer-1 hang protection (per-attempt S3 timeouts +
        retries) is inherited from ``_open_repo``'s defaults, so each worker's S3
        attempts can't hang.
        """
        repo = _open_repo(master_path, max_concurrent_requests=max_concurrent_requests)
        session = repo.writable_session("main")

        t_copy = time.monotonic()
        if n_shards == 1:
            # One shard → no process pool; fork once and write inline (avoids spawn
            # cost for tiny features). No separate process can wedge here, but the
            # inline S3 calls still go through the timeout-protected repo (Layer 1),
            # so a hung socket fails the attempt rather than blocking forever; the
            # optional hard ceiling is enforced by the deadline check below.
            done = [_copy_units_in_process(session.fork(), feature_path, shards[0], n_threads, max_concurrent_requests)]
            if feature_timeout_sec is not None and time.monotonic() - t_copy > feature_timeout_sec:
                raise TimeoutError(
                    f"Inline merge of {feature_path} exceeded {feature_timeout_sec:.0f}s feature_timeout_sec"
                )
        else:
            # One fork per shard — each an independent session (own lock, own
            # connection pool). Workers write chunk bytes straight to storage; only
            # changeset refs come back. spawn (not fork) so a child never inherits the
            # parent's icechunk runtime/credentials state.
            ctx = mp.get_context("spawn")
            pool = ProcessPoolExecutor(max_workers=n_shards, mp_context=ctx)
            try:
                futures = [
                    pool.submit(
                        _copy_units_in_process, session.fork(), feature_path, shard, n_threads, max_concurrent_requests
                    )
                    for shard in shards
                ]
                worker_pids = list(getattr(pool, "_processes", {}) or {})
                # Wait with the CPU-stall watchdog: a large-but-progressing copy keeps
                # aggregate worker CPU climbing and is never killed; only a sustained
                # flat-CPU stall (or the optional hard ceiling) leaves shards pending.
                completed, pending = _wait_with_stall_detection(
                    futures,
                    grace_sec=grace_sec,
                    poll_sec=poll_sec,
                    hard_timeout_sec=feature_timeout_sec,
                    log=_log,
                    feature_path=feature_path,
                )
                if pending:
                    # Workers wedged (diagnosed: dead socket in ``sk_wait_data``).
                    # Don't join — that would hang on the same worker. Kill and bail.
                    _kill_pool_workers(pool, worker_pids)
                    raise TimeoutError(
                        f"Merge of {feature_path} stalled with {len(pending)} of {n_shards} "
                        "worker shard(s) unfinished; killed workers and aborting attempt"
                    )
                # All finished; ``f.result()`` surfaces any worker exception. No worker
                # is wedged, so this cannot hang.
                done = [f.result() for f in completed]
            finally:
                # On the happy path nothing is wedged so this returns at once; on the
                # stall path the workers were already SIGKILLed, so wait=False keeps
                # teardown from re-blocking on a dead worker.
                pool.shutdown(wait=False, cancel_futures=True)

        t_merge = time.monotonic()
        session.merge(*done)
        session.commit(commit_msg)
        return t_copy, t_merge

    last_exc: TimeoutError | None = None
    for attempt in range(feature_retries + 1):
        try:
            t_copy, t_merge = _run_attempt()
            break
        except TimeoutError as e:
            last_exc = e
            if attempt < feature_retries:
                _log.warning(
                    "%s; retrying feature %s in a fresh session (attempt %d of %d)",
                    e,
                    feature_path,
                    attempt + 2,
                    feature_retries + 1,
                )
                continue
            _log.error(
                "Merge of %s timed out after %d attempt(s); failing the run loudly",
                feature_path,
                feature_retries + 1,
            )
            raise
    else:  # pragma: no cover — loop always breaks or raises
        raise last_exc  # type: ignore[misc]

    _log.info(
        "Merged %d date(s) x %d var(s) from %s into %s: %d chunk-write(s) over %d process(es) in %.1fs "
        "(copy %.1fs, merge+commit %.1fs)",
        ftimes.size,
        len(feature_vars),
        feature_path,
        master_path,
        len(units),
        n_shards,
        time.monotonic() - t_start,
        t_merge - t_copy,
        time.monotonic() - t_merge,
    )
    return int(ftimes.size)


def delete_store(store_path: str) -> bool:
    """Delete a store. Returns True if deleted, False if not found.

    The public counterpart of ``zarr_store._delete_store`` (fsspec recursive rm);
    used to drop the temp per-feature stores after their data is safely in the
    master. Never raises — a failed cleanup is logged, not fatal (the merge
    already succeeded).
    """
    try:
        fs = fsspec.filesystem(fsspec.utils.get_protocol(store_path))
        if fs.exists(store_path):
            fs.rm(store_path, recursive=True)
            logger.info("Deleted store %s", store_path)
            return True
        logger.info("Store already absent: %s", store_path)
        return False
    except Exception as e:
        logger.warning("Failed to delete store %s: %s", store_path, e)
        return False


def _store_exists(store_path: str) -> bool:
    """True if a store already exists at ``store_path`` (fsspec, protocol-aware)."""
    fs = fsspec.filesystem(fsspec.utils.get_protocol(store_path))
    return bool(fs.exists(store_path))


def merge_stores(
    master_path: str,
    feature_paths: list[str],
    *,
    roi: ROIMetadata,
    var_dtypes: dict[str, np.dtype],
    tile_id: str,
    crs: str | None = None,
    chunks: dict[str, int] | None = None,
    manifest_split_sizes: dict[str, int] | None = None,
    delete_temp: bool = False,
    overwrite_master: bool = False,
    resume: bool = False,
    max_workers: int | None = None,
    threads_per_process: int | None = None,
    max_concurrent_requests: int | None = None,
    stall_grace_sec: float | None = None,
    feature_timeout_sec: float | None = None,
    feature_retries: int = 1,
    log: logging.Logger | logging.LoggerAdapter | None = None,
) -> dict:
    """Merge many master-snapped feature stores into one master mosaic — the full driver.

    The one-call sequence for the multiple-regional-inserts use case: seed a master
    over the union of the features' dates, then region-write each feature into its
    positional slice, one commit per feature, and (optionally) drop the temp
    feature stores. Wraps the individual primitives (:func:`gather_time_union`,
    :func:`~.empty_store.create_empty_store`, :func:`read_master_axes`,
    :func:`merge_feature_into_master`, :func:`delete_store`) in the order and under
    the invariants they require, so callers don't have to re-derive the recipe. No
    Dask, no cluster — the merge is process-parallel raw-Zarr chunk movement.

    Every feature grid must be an **exact pixel-subset of the master grid**
    ("master-snapped": same CRS, resolution, and axis order, coords a contiguous
    subset). That is the caller's precondition — produce each feature on a window of
    the master grid (e.g. ``master_geobox.enclosing(geom)``); :func:`merge_feature_into_master`
    validates extent, coordinates, dtype, and chunking and raises if it is not met.

    Steps:

    1. :func:`gather_time_union` over ``feature_paths`` → the master's date axis
       (sorted, de-duped; missing/empty features contribute nothing). If no feature
       has any dates, returns early with ``skipped=True``.
    2. Seed the master with :func:`~.empty_store.create_empty_store` over ``roi`` and
       that union (skipped when ``resume=True``). Metadata-only — no pixels computed.
    3. :func:`read_master_axes` once (the axes are constant per master).
    4. :func:`merge_feature_into_master` for each feature, **sequentially** — which
       is also what makes overlapping boundary chunks correct (successive commits
       reconcile a shared chunk by read-modify-write, so features need not be
       chunk-disjoint).
    5. If ``delete_temp``, :func:`delete_store` each feature path — only after the
       whole merge succeeds.

    The seed and every merge open run inside one :func:`~.zarr_store.manifest_split`
    block so the icechunk manifest split is consistent across create and all opens
    (a mismatched open would rewrite the whole manifest, undoing the win).

    Args:
        master_path: Destination master store URI (created here unless ``resume``).
        feature_paths: The per-feature store URIs to merge, in the order to apply
            them. Each must be master-snapped (see above).
        roi: Master grid authority (from ``read_roi_metadata``), passed to the seed.
        var_dtypes: ``{var_name: dtype}`` the master is seeded with — the schema the
            features must match.
        tile_id: Store-metadata identifier for the seeded master.
        crs: CRS authority code; defaults to ``roi.native_crs``.
        chunks: Seed chunk sizes (``time``/``northing``/``easting``); defaults to
            the ingest chunking. Must keep ``time`` chunk size 1 for the merge's
            temporal disjointness invariant (the default does).
        manifest_split_sizes: icechunk manifest split applied across the seed and
            every merge open. ``None`` (default) → :data:`~.zarr_store.DEFAULT_MANIFEST_SPLIT_SIZES`;
            pass ``{}`` to disable splitting.
        delete_temp: When True, delete each feature store after a successful merge.
            Default False (the caller owns temp lifecycle).
        overwrite_master: When the master already exists and ``resume`` is False:
            False (default) raises; True deletes and re-seeds it.
        resume: Merge into an existing master without re-seeding (pick up a partial
            run). Requires the master to exist; pass only the not-yet-merged
            features (region writes are idempotent, so re-passing an interrupted one
            is safe). ``overwrite_master`` is ignored.
        max_workers: Worker processes per feature copy (forwarded to
            :func:`merge_feature_into_master`; default one per core).
        threads_per_process: Thread-pool width inside each worker (forwarded).
        max_concurrent_requests: Per-repo/per-fork S3 concurrency cap (forwarded).
        stall_grace_sec: CPU-stall watchdog grace window per feature (forwarded).
        feature_timeout_sec: Optional per-feature hard ceiling (forwarded).
        feature_retries: Retries per feature on a stall (forwarded; default 1).
        log: Orchestrator run logger (e.g. Prefect's) for per-feature timing lines;
            the module logger is used when omitted.

    Returns:
        Summary dict: ``master_path``, ``n_dates`` (the master's date count),
        ``merged`` (``{feature_path: dates_written}``), ``deleted`` (temp stores
        removed), ``skipped`` (True iff no feature had any dates), ``elapsed_sec``.

    Raises:
        ValueError: ``feature_paths`` is empty.
        FileExistsError: the master exists and neither ``overwrite_master`` nor
            ``resume`` is set.
        FileNotFoundError: ``resume=True`` but no master exists.
        TimeoutError: propagated from :func:`merge_feature_into_master` if a feature
            stalls through all its retries.
    """
    _log = log or logger
    t0 = time.monotonic()
    if not feature_paths:
        raise ValueError("feature_paths is empty; nothing to merge")
    crs = crs or roi.native_crs
    split_sizes = dict(DEFAULT_MANIFEST_SPLIT_SIZES) if manifest_split_sizes is None else manifest_split_sizes

    master_exists = _store_exists(master_path)
    if resume:
        if not master_exists:
            raise FileNotFoundError(
                f"resume=True but no master exists at {master_path}. "
                "Resume picks up a partially-written master; run without resume to seed one."
            )
        _log.info("resume=True — merging into existing master %s without re-seeding", master_path)
    elif master_exists:
        if not overwrite_master:
            raise FileExistsError(
                f"Master already exists: {master_path}. Pass overwrite_master=True to rebuild it, "
                "resume=True to pick up a partial run, or delete it first."
            )
        _log.warning("overwrite_master=True — deleting existing master %s", master_path)
        delete_store(master_path)

    # 1. Date union across the features (exact datetime64; empty/missing contribute
    #    nothing). Nothing to seed or merge if no feature carries a date.
    times = gather_time_union(feature_paths)
    if not times.size:
        _log.warning("No dates across any feature store — nothing to merge into %s", master_path)
        return {
            "master_path": master_path,
            "n_dates": 0,
            "merged": {},
            "deleted": [],
            "skipped": True,
            "elapsed_sec": time.monotonic() - t0,
        }

    # The seed's create and every merge open must see the SAME manifest split, so
    # both run inside one manifest_split block (a mismatched open would rewrite the
    # whole single manifest, undoing the split's per-commit locality win).
    with manifest_split(split_sizes):
        if resume:
            _log.info("resume=True — reusing existing master %s axis/grid", master_path)
        else:
            _log.info("Seeding master %s with %d date(s) over %d variable(s)", master_path, times.size, len(var_dtypes))
            create_empty_store(
                master_path,
                roi=roi,
                times=times,
                var_dtypes=var_dtypes,
                tile_id=tile_id,
                crs=crs,
                chunks=chunks,
            )

        # The master's date axis and variable set are constant once seeded — read
        # once here rather than re-opening the (continental) master per feature.
        master_times, master_vars = read_master_axes(master_path)
        merged: dict[str, int] = {}
        for i, fpath in enumerate(feature_paths):
            _log.info("Merging feature %d/%d: %s", i + 1, len(feature_paths), fpath)
            merged[fpath] = merge_feature_into_master(
                master_path,
                fpath,
                master_times=master_times,
                master_vars=master_vars,
                max_workers=max_workers,
                threads_per_process=threads_per_process,
                max_concurrent_requests=max_concurrent_requests,
                stall_grace_sec=stall_grace_sec,
                feature_timeout_sec=feature_timeout_sec,
                feature_retries=feature_retries,
                log=log,
            )
            _log.info("Feature %d/%d done: %d date(s) written", i + 1, len(feature_paths), merged[fpath])

    # Cleanup only after the whole merge succeeds — a mid-merge failure leaves the
    # temps for a re-run.
    deleted: list[str] = []
    if delete_temp:
        for fpath in feature_paths:
            if delete_store(fpath):
                deleted.append(fpath)
        _log.info("Deleted %d temp feature store(s)", len(deleted))

    return {
        "master_path": master_path,
        "n_dates": int(master_times.size),
        "merged": merged,
        "deleted": deleted,
        "skipped": False,
        "elapsed_sec": time.monotonic() - t0,
    }


__all__ = [
    "delete_store",
    "gather_time_union",
    "merge_feature_into_master",
    "merge_stores",
    "read_master_axes",
    "read_store_times",
]
