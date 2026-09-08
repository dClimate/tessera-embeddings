"""Shard-aligned, land-masked writer for the global store (ADR-008 D3/D6).

Fills one (zone, year) with whole shards in a single Icechunk commit. The
d3v2-verified write path: each worker reads its assigned shard *sources* and
writes each shard's region in one raw-zarr assignment - real data in land inner
chunks, fill (elided by the sharding codec) elsewhere - so every shard object is
emitted once, no read-modify-write, no dense nodata.

Cooperative fork/merge: the coordinator forks the session, workers write into
their fork, the coordinator merges and makes **one commit per (zone, year)**,
updating ``years_complete`` in the same commit (D1). Commits are UNGATED. They do
contend on the branch-tip CAS -- all 120 zone groups share one repo -- but that
was measured at 2.2 s for 16 simultaneous committers and 15 s for 120, with zero
unresolvable conflicts at every N. See
``context_docs/storage/writing-to-the-global-store.md``.

A :class:`ShardSource` decouples the writer from *where* shard data comes from
(staged inference files in production; synthetic in tests), and must be picklable
so it can be shipped to spawned workers. :func:`run_forked` is the shared
fork → parallel-write → merge scaffolding; the single-ROI assembly engine
(:mod:`tessera_embeddings.inference.assembly`) drives it with a different
worker body.

Progress is reported at two levels, because no single scope can see both. The
coordinator (:func:`_await_forks`) knows the total and states, on a timer, how
many payloads are still outstanding — but a payload only completes near the end
of the write, so it cannot draw a curve alone. Each worker
(:func:`_write_shards_worker`) states its own within-payload progress on the
same timer, and since workers are separate spawned processes with no shared
counter, those lines are what a log reader aggregates into the write-wide curve.
Routing differs too: the coordinator logs through the caller-supplied ``log``
(a Prefect flow passes its run logger down, the only route to the Prefect API —
this module's own logger reaches only the process's log stream), while a worker
can only use the module logger of its own spawned process.
"""

from __future__ import annotations

import ctypes
import faulthandler
import functools
import logging
import multiprocessing
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from typing import Any, Protocol, cast

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.environment import code_identity, configure_logging
from tessera_embeddings.config.fault_injection import ArmedFault
from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage.icechunk_logging import traced_commit
from tessera_embeddings.storage.publication_spacing import publication
from tessera_embeddings.storage.session_catch_up import (
    CATCH_UP_INTERVAL_S,
    CatchUpAbortedTheWaitError,
    CatchUpDidNotStopError,
    catch_up_best_effort,
    fresh_session_checked,
    ticking,
)
from tessera_embeddings.storage.time_axis import read_time_values, year_of

_log = logging.getLogger(__name__)

#: How often :func:`run_forked`'s coordinator states how many payloads are still
#: outstanding, and how often a worker states its own progress within one.
#: A forked write emits nothing between its start and its finish, so without a
#: periodic line a healthy long write and a hung one are indistinguishable in the
#: log — and the operator's only recourse is to guess. Set far enough apart to stay
#: quiet for short writes and close enough to bound how long a stall hides.
PROGRESS_INTERVAL_S = 300.0

#: How long the fork phase may make NO shard progress before the watchdog declares it wedged,
#: dumps every thread's stack, and tears the worker pool down so the fill fails instead of
#: hanging. On 2026-09-04 five assemblies hung in the shard-write tail and never returned — no
#: exception, no CPU — each blocking its cluster's entire single-threaded assembly backlog for
#: days (see ``context_docs/assembly/assembly-wedges-during-fork-phase-2026-09-04.md``).
#:
#: THIRTY MINUTES, and the choice is one-sided. Healthy workers write shards continuously —
#: measured 200-260 shards every five minutes on dense zones, and no healthy run has ever paused
#: shard writing for even one full progress interval. A stall that lasts thirty minutes is ~6x
#: the longest gap a healthy run produces and cannot be reached by normal S3 jitter or a slow
#: band. The failure it guards against sat for days, so erring long costs nothing: the point is
#: to convert an unbounded hang into a bounded, re-dispatchable failure, not to trim minutes.
FORK_STALL_TIMEOUT_S = 1800.0

#: How long each step of the publish child gets — re-home, merge and fill commit; then the
#: completion mark. Every commit ever measured finished in 0.6-5 s and the read-only conflict diff
#: in under half a second; unpickling sixteen forks and importing the child's dependencies add tens
#: of seconds at most. Ten minutes is far past all of that and far short of the hang, which sat for
#: days. When it expires the child is killed and the step is retried ONCE in a fresh child on a
#: fresh session — the finished forks are the parent's, so nothing written is lost.
PUBLISH_STEP_TIMEOUT_S = 600.0


class PhaseTimer:
    """Wall- and CPU-second accumulator for the phases of a fork worker.

    Two clocks per span because they answer different questions: wall time is how
    long a phase held the worker, CPU time how much of that was computation. Their
    difference is time spent *blocked* — on the object store, almost always — which
    a wall clock alone cannot distinguish from work, and exposing that distinction
    is what the caller's summary record is for. CPU time is process-wide
    (:func:`time.process_time`), so work the storage layer or a codec does on its
    own threads is still counted: the CPU figure is an upper bound on the phase's
    in-process compute, never an undercount.

    Spans with the same name accumulate. Spans must not overlap or nest: each
    moment of the worker's life belongs to at most one phase, so the per-phase
    walls plus the un-phased residue sum to the worker's total wall time — the
    invariant that lets a reader attribute every second of a slow worker.

    The cost per span is four clock reads, so per-tile spans in a hot loop are
    safe; nothing here writes a log line.
    """

    class _Span:
        """One timed entry into a phase; ``with timer.phase(name):`` scoped."""

        __slots__ = ("_cpu0", "_name", "_timer", "_wall0")

        def __init__(self, timer: PhaseTimer, name: str) -> None:
            self._timer = timer
            self._name = name

        def __enter__(self) -> PhaseTimer._Span:
            self._wall0 = time.monotonic()
            self._cpu0 = time.process_time()
            return self

        def __exit__(self, *exc: object) -> None:
            self._timer._add(self._name, time.monotonic() - self._wall0, time.process_time() - self._cpu0)

    def __init__(self) -> None:
        self._wall0 = time.monotonic()
        self._cpu0 = time.process_time()
        self._wall: dict[str, float] = {}
        self._cpu: dict[str, float] = {}

    def _add(self, name: str, wall: float, cpu: float) -> None:
        self._wall[name] = self._wall.get(name, 0.0) + wall
        self._cpu[name] = self._cpu.get(name, 0.0) + cpu

    def phase(self, name: str) -> PhaseTimer._Span:
        """A context manager timing one entry into phase ``name``."""
        return PhaseTimer._Span(self, name)

    def stats(self) -> dict[str, float]:
        """Accumulated ``{<phase>_s, <phase>_cpu_s}`` per phase, plus ``wall_s``/``cpu_s`` since construction.

        Values are rounded — these feed a single JSON log record, and
        sub-millisecond precision is noise at the durations that matter here.
        """
        out: dict[str, float] = {}
        for name, wall in self._wall.items():
            out[f"{name}_s"] = round(wall, 3)
            out[f"{name}_cpu_s"] = round(self._cpu[name], 3)
        out["wall_s"] = round(time.monotonic() - self._wall0, 3)
        out["cpu_s"] = round(time.process_time() - self._cpu0, 3)
        return out


class ShardSource(Protocol):
    """Supplies the shard data for one (zone, year) fill.

    Implementations must be picklable (a frozen dataclass) so the writer can ship
    them to spawned workers.
    """

    def live_shards(self) -> Iterable[tuple[int, int]]:
        """Return the ``(sy, sx)`` shard indices that have data to write."""
        ...

    def load(self, shard: tuple[int, int]) -> dict[str, np.ndarray]:
        """Return ``{var_name: block}`` for a shard - each block covers the whole
        (edge-clamped) shard region with ocean inner chunks at the array fill
        value. Return ``{}`` to skip a shard entirely.
        """
        ...


def commit_with_rebase(
    session: icechunk.Session,
    message: str,
    *,
    tries: int = 1000,
) -> str:
    """Commit, auto-rebasing on a moved branch tip; return the snapshot id.

    Uses icechunk's built-in rebase loop with a :class:`ConflictDetector` - enough
    for our write model, where concurrent commits touch disjoint groups/regions
    and always rebase cleanly (run-1 T0/T5: zero unresolvable conflicts). A real
    chunk conflict surfaces as ``RebaseFailedError`` rather than being masked.

    Commits are UNGATED: concurrency here buys a SLOWDOWN, not a failure -- 2.2 s
    commits at 16 concurrent committers, 15 s at 120, zero unresolvable conflicts at
    every N. See ``context_docs/storage/writing-to-the-global-store.md``.

    **Timed here, and here is the only place that sees every commit.** That decision's
    reopen criterion is commit LATENCY, and the obvious detector -- ``commit_s`` in
    ``ASSEMBLY_SUMMARY`` -- cannot see the dominant source of concurrent commits: a
    terminal cell marks itself through ``mark_zone_year_empty`` and returns without ever
    reaching ``assemble_global``, and terminal cells were **72 of the first 78**
    completions. One line per commit, so the volume is one per zone-year.
    """
    started = time.monotonic()
    snapshot = traced_commit(session, message, rebase_with=icechunk.ConflictDetector(), rebase_tries=tries)
    _log.info("COMMIT %.2fs: %s", time.monotonic() - started, message)
    return snapshot


def shard_pitch(arr: zarr.Array) -> int:
    """An array's northing write granularity: shard height if sharded, else chunk height."""
    return (arr.shards or arr.chunks)[1]


#: Live shard counts, ``[done_0, total_0, ...]`` by worker index. Shared memory reaches a spawned
#: child only through the pool INITIALIZER -- it cannot be pickled into a submitted payload.
#: ``None`` off the pool path, which includes the single-payload in-process run.
_PROGRESS_SLOTS: ctypes.Array[ctypes.c_long] | None = None


def _init_fork_worker(slots: ctypes.Array[ctypes.c_long]) -> None:
    """Runs once per spawned child: configure logging (a spawned process inherits none, so the
    root WARNING default would discard its records) and stash the slots it reports counts into.
    """
    global _PROGRESS_SLOTS
    configure_logging()
    _PROGRESS_SLOTS = slots


def report_shard_progress(worker_index: int, done: int, total: int) -> None:
    """Publish a worker's shard counts where the COORDINATOR can read them: a worker's own log
    line never reaches the orchestrator. Unlocked on purpose -- monotone counters where a torn
    read costs one stale line, against a lock taken on every shard. No-op off the pool path.
    """
    if _PROGRESS_SLOTS is None:
        return
    _PROGRESS_SLOTS[2 * worker_index] = done
    _PROGRESS_SLOTS[2 * worker_index + 1] = total


def _await_forks(
    futures: list[Future],
    progress_interval_s: float,
    *,
    unit: str = "partitions",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    slots: ctypes.Array[ctypes.c_long] | None = None,
    abort: threading.Event | None = None,
    stalled: threading.Event | None = None,
) -> list[Any]:
    """Collect ``futures`` in submission order, logging what is still outstanding.

    Returns results positionally rather than by completion order, because a fork's
    position is its band and :meth:`icechunk.Session.merge` is given them as the
    caller's payloads were ordered. Re-raises the first failure in that same order,
    matching what ``Executor.map`` would have done.

    ``abort``, when set, ends the wait early — it answers "is there any point still
    waiting?". The forks' own failures already stop the wait; this covers the case
    where something outside them has made the fill uncommittable, so the writes still
    running are known to be wasted.

    ``unit`` is the caller's name for one payload. What a payload holds is the
    caller's decision (northing bands for one, round-robin tile partitions for
    another), so a fixed noun would misdescribe the work for all callers but one.
    ``log`` is where the lines go: the module logger reaches only the process's own
    log stream, so a caller inside a flow passes its run logger to make the wait
    visible to the orchestrator as well.
    """
    logger = log or _log
    started = time.monotonic()
    pending: set[Future] = set(futures)
    while pending:
        done, pending = wait(pending, timeout=progress_interval_s, return_when=FIRST_COMPLETED)
        if stalled is not None and stalled.is_set():
            # The watchdog tore the pool down: the WRITE stopped, which is a different fact from a
            # failed catch-up and must read as one in the failure.
            raise ForkPhaseStalledError(
                "the fork phase made no shard progress for the stall timeout; the worker pool was "
                "terminated and this cell is retained for resume"
            )
        if abort is not None and abort.is_set():
            # The caller will raise the real cause; this only stops the waiting. Returning
            # instead would look like success, and the fill would merge forks it must not.
            raise CatchUpAbortedTheWaitError("a periodic catch-up failed, so this fill can no longer commit safely")
        # SURFACE A FAILURE AS SOON AS IT LANDS. A fork dies on a deterministic fault — a
        # corrupt staged tile, a dtype the destination cannot hold — and every other fork is
        # going to hit the same wall or write shards that will be discarded anyway. Collecting
        # results only once `pending` empties costs the rest of a multi-hour assembly to learn
        # what the first fork already knew.
        for future in done:
            if future.exception() is not None:
                future.result()  # re-raises, with the worker's traceback attached
        if pending:
            # SHARDS, not payloads: a payload completes only when its worker returns and they all
            # return at the end, so the payload figure reads 0/N for the whole write. `slots` is
            # what the workers have actually written; withheld until ALL have reported a total.
            n = len(futures)
            shards = ""
            if slots is not None:
                totals = [slots[2 * i + 1] for i in range(n)]
                # EVERY worker's total, or none. Workers start staggered, so a denominator summed
                # over only those that have reported is short -- and a short denominator reports
                # near-completion while most of the write is still outstanding.
                if all(t > 0 for t in totals):
                    got = sum(slots[2 * i] for i in range(n))
                    want = sum(totals)
                    shards = f"{got}/{want} shards written ({100.0 * got / want:.0f}%), "
            logger.info(
                "Assembly progress: %s%d/%d %s outstanding after %.0f min",
                shards,
                len(pending),
                n,
                unit,
                (time.monotonic() - started) / 60.0,
            )
    return [future.result() for future in futures]


class ForkPhaseStalledError(RuntimeError):
    """The fork phase made no shard progress for the watchdog's timeout and was torn down.

    Its own type, distinct from :class:`CatchUpAbortedTheWaitError`: both end the wait, but one
    means "a catch-up failed, so this fill cannot commit safely" and the other means "the write
    itself stopped moving" — and an operator reading the failure needs to know which. The cell is
    retained for resume either way.
    """


def _terminate_pool(ex: ProcessPoolExecutor) -> None:
    """Tear a fork pool down without waiting on it, then kill whatever is still alive.

    The one place two callers reach — the ``except BaseException`` unwind and the stall
    watchdog — so the "how" of killing a pool lives once. ``cancel_futures`` reaches only
    queued work, so a multi-hour shard writer already running keeps running (and keeps writing
    fork objects) after the coordinator has moved on unless the live processes are terminated
    too; Python's own executor atexit hook would then block interpreter shutdown on them.

    Killing them is safe because nothing a worker has written is IN the store: workers write
    into a fork, and a fork joins the repository only when the coordinator merges and commits.
    A dropped, unmerged ``ForkSession`` is icechunk's own documented way to orphan chunks, so a
    kill costs unreferenced objects that GC reclaims — never committed data.

    ``_processes`` is private (there is no public terminate API) and read BEFORE ``shutdown``:
    ``shutdown`` sets ``_processes = None`` unconditionally, so reading it afterwards yields
    ``None`` and ``.values()`` raises ``AttributeError``, masking the failure this exists to
    surface. Guarded so a future Python that renames the attribute degrades to a wait-free
    shutdown rather than an ``AttributeError``.
    """
    procs = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        if proc.is_alive():
            proc.terminate()


@contextmanager
def _fork_stall_watchdog(
    slots: ctypes.Array[ctypes.c_long] | None,
    n_workers: int,
    stalled: threading.Event,
    ex: ProcessPoolExecutor | None,
    *,
    timeout_s: float,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> Iterator[None]:
    """Fail a wedged fork phase instead of letting it hang forever.

    The forks write for hours and then, on 2026-09-04, five of them stopped in the write's tail
    and never returned — no exception, no CPU, no shard progress — each freezing the one trailing
    thread its whole cluster assembles on. The commit alarm (:func:`traced_commit`) could not see
    it: the wedge is upstream of the commit. Nothing watched the fork phase, and nothing bounded
    it. This does both.

    On its own daemon thread it watches the shard counters the workers write straight into shared
    memory (``slots``), which advance whether or not the coordinator thread is running. If the
    total does not move for ``timeout_s`` it:

    1. dumps every thread's Python stack with :func:`faulthandler.dump_traceback` — the stack
       that could not be got any other way (Fargate denies ``CAP_SYS_PTRACE``, so ``py-spy`` and
       ``/proc/<pid>/stack`` are refused, and the process runs pre-3.14 so thread names are
       invisible too), so the next wedge names its own stuck call;
    2. sets ``stalled``, so :func:`_await_forks` raises :class:`ForkPhaseStalledError` at its
       next wake rather than waiting out another timeout; and
    3. terminates the worker pool (:func:`_terminate_pool`), which resolves any pending future
       as ``BrokenProcessPool`` and lets a ``wait=True`` shutdown join return — so a coordinator
       parked in ``_await_forks`` or ``ex.shutdown`` is freed and the fill fails cleanly, its
       cell retained for the campaign to re-dispatch.

    **What it does NOT recover, stated plainly.** If the coordinator thread is itself parked
    inside a pyo3 call into icechunk (a catch-up ``rebase`` or the ``merge``), no Python action
    unwinds it and terminating the workers cannot free it. The stack dump still fires, so the
    wedge is diagnosed; preventing that case is the job of publication spacing, not this net.
    See ``context_docs/assembly/assembly-wedges-during-fork-phase-2026-09-04.md``.

    Best-effort throughout: a watchdog that dies on one failed read, or that could itself end a
    healthy write, is worse than none. ``slots`` or ``ex`` being ``None`` (the single-payload
    in-process path builds neither) makes it a no-op — that path writes one shard and cannot
    exhibit the multi-worker stall this guards.
    """
    logger = log or _log
    if slots is None or ex is None:
        yield
        return

    def _written() -> int:
        # Best-effort read of a lock-free shared array; a torn long only misreads the total by
        # one worker's count for one tick, which cannot manufacture a 30-minute stall.
        try:
            return sum(slots[2 * i] for i in range(n_workers))
        except Exception:
            return -1

    stopped = threading.Event()

    def _watch() -> None:
        last_seen = _written()
        last_change = time.monotonic()
        # Poll well inside the timeout so a real stall is caught promptly and a brief pause is
        # not; the timeout, not this period, is what a healthy write is judged against.
        period = min(30.0, timeout_s / 10.0)
        while not stopped.wait(period):
            now = _written()
            if now != last_seen:
                last_seen, last_change = now, time.monotonic()
                continue
            if time.monotonic() - last_change < timeout_s:
                continue
            with suppress(Exception):
                logger.critical(
                    "ASSEMBLY FORK PHASE STALLED: no shard progress for %.0f min (%d shards "
                    "written across %d workers). The fill publishes nothing further; tearing the "
                    "worker pool down so it fails and its cell can be re-dispatched. Thread stacks "
                    "follow.",
                    (time.monotonic() - last_change) / 60.0,
                    max(last_seen, 0),
                    n_workers,
                )
            with suppress(Exception):
                faulthandler.dump_traceback()
            stalled.set()
            with suppress(Exception):
                _terminate_pool(ex)
            return  # one shot: the pool is gone, so there is nothing left to watch

    watcher = threading.Thread(target=_watch, name="fork-stall-watchdog", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stopped.set()


def _run_pool(
    worker_fn: Callable[[dict[str, Any]], Any],
    payloads: list[dict[str, Any]],
    indices: list[int],
    *,
    progress_interval_s: float,
    fork_stall_timeout_s: float,
    unit: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None,
    abort: threading.Event,
) -> tuple[dict[int, Any], list[int]]:
    """Run ``payloads[i]`` for ``i in indices`` on a fresh spawn pool; salvage what finishes.

    Returns ``(finished, stalled)``: results by payload index, and the indices whose worker had
    not returned when the watchdog tore the pool down. A clean run returns every index and an
    empty list. Any failure OTHER than a stall — a worker's deterministic fault, a catch-up that
    failed (``abort``) — propagates after the pool is terminated, as before.

    **Finished workers are terminated, never joined.** Their result — the fork — is already in
    the parent's hands, so a worker process that will not exit (one of the four places the
    2026-09-04 fills could have parked) costs nothing to kill and everything to wait for.
    """
    ctx = multiprocessing.get_context("spawn")
    # `initializer` runs once per spawned child before any payload. A spawned process inherits no
    # logging config, so without it the root WARNING default discards every INFO record a worker
    # produces. Slots are sized for EVERY payload so a re-run's worker indices still address them.
    slots = ctx.Array("l", 2 * len(payloads), lock=False)
    ex = ProcessPoolExecutor(max_workers=len(indices), mp_context=ctx, initializer=_init_fork_worker, initargs=(slots,))
    stalled = threading.Event()
    futures: list[Future] = []
    try:
        with _fork_stall_watchdog(slots, len(payloads), stalled, ex, timeout_s=fork_stall_timeout_s, log=log):
            futures = [ex.submit(worker_fn, payloads[i]) for i in indices]
            try:
                results = _await_forks(
                    futures, progress_interval_s, unit=unit, log=log, slots=slots, abort=abort, stalled=stalled
                )
            except ForkPhaseStalledError:
                finished = {
                    i: f.result()
                    for i, f in zip(indices, futures, strict=True)
                    if f.done() and not f.cancelled() and f.exception() is None
                }
                return finished, [i for i in indices if i not in finished]
        return dict(zip(indices, results, strict=True)), []
    finally:
        # Every exit: the workers' results are in hand or the pool is being abandoned; either
        # way nothing is gained by joining a process and a wedged one would never be joined.
        _terminate_pool(ex)


class PublishWedgedError(RuntimeError):
    """The publish child did not finish a step within its timeout, twice.

    Both children were killed; the forks are still the parent's and nothing was lost — but the
    cell has not landed and is retained for resume. Its own type so the failure reads as what it
    is: the store's commit path hung, rather than any of the things a generic failure could mean.
    """


def _publish_child(
    conn: Any,  # noqa: ANN401 — a multiprocessing Connection
    repo: icechunk.Repository,
    group: str,
    base: str,
    forks: list[Any],
    fill_message: str,
    attrs: dict[str, Any],
    mode: str,
    step_timeout_s: float,
) -> None:
    """The publish, in a process the parent can kill: fresh session, merge, commit, then mark.

    Two phases over ``conn`` so the parent can run the between-commits drill hook and the
    publication spacing around both: ``("fill", snapshot, seconds)`` after the shard commit,
    then wait for the parent's go-ahead, then ``("done", snapshot, seconds)`` after the mark.
    ``mode="mark"`` skips to the mark — used when the fill landed but the mark wedged, so a retry
    does not commit the shards twice. Any exception is sent as ``("error", repr)``.

    ``faulthandler.dump_traceback_later`` is armed a few seconds inside the parent's timeout, so
    a child that is about to be killed for wedging prints every thread's stack first — the
    diagnostic the 2026-09-04 fills could not produce.
    """
    configure_logging()
    faulthandler.dump_traceback_later(max(step_timeout_s - 5.0, 1.0), repeat=False)
    try:
        if mode == "publish":
            session = fresh_session_checked(repo, group, base=base)
            t_merge = time.monotonic()
            session.merge(*forks)
            t0 = time.monotonic()
            snapshot = commit_with_rebase(session, fill_message)
            conn.send(("fill", snapshot, round(time.monotonic() - t0, 3), round(t0 - t_merge, 3)))
            faulthandler.cancel_dump_traceback_later()
            if conn.recv() != "go":
                return
            faulthandler.dump_traceback_later(max(step_timeout_s - 5.0, 1.0), repeat=False)
        t1 = time.monotonic()
        marked = commit_year_attrs(repo, group, **attrs)
        conn.send(("done", marked, round(time.monotonic() - t1, 3)))
    except BaseException as exc:  # the parent must hear about it; a dead child says nothing
        with suppress(Exception):
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        faulthandler.cancel_dump_traceback_later()


def publish_forks_in_child(
    repo: icechunk.Repository,
    group: str,
    forks: list[Any],
    *,
    base: str,
    fill_message: str,
    attrs: dict[str, Any],
    fault: ArmedFault | None = None,
    year_label: int | None = None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    step_timeout_s: float = PUBLISH_STEP_TIMEOUT_S,
    retries: int = 1,
    _child: Callable[..., None] = _publish_child,
) -> tuple[str, dict[str, Any]]:
    """Merge finished forks into a fresh session and publish them, in a child the parent can kill.

    **This is #165's recovery made the normal path and made killable.** The coordinator never
    commits from its own session: the finished forks are pickled to a child that opens a fresh
    session at the tip, checks the skipped range for a same-group commit
    (:func:`fresh_session_checked`), merges, commits the shards, and — after the parent's
    go-ahead — commits the completion mark. Each step runs under ``step_timeout_s``. A child that
    does not answer in time is killed and the step is retried once in a new child on a new
    session. The forks stay in the parent throughout, so a wedge costs one kill and one retry,
    not the write.

    Why a child and not a thread: the 2026-09-04 fills parked their coordinator INSIDE icechunk,
    where no Python mechanism unwinds a thread. A process can be killed. Why always, not only on a
    wedge: a coordinator already wedged cannot start a retry, and pickling the forks after a
    wedge would touch objects the stuck thread may hold.

    The parent runs ``fault.die_between_commits`` between the two phases, where the drill has to
    land, and the caller wraps this whole call in the publication spacing slot.

    Returns:
        ``(mark_snapshot, timings)`` — the snapshot a tag must point at, and ``merge_s``,
        ``commit_s``, ``attrs_commit_s``, ``publish_retries``.

    Raises:
        PublishWedgedError: a step timed out in both attempts.
        RuntimeError: the child reported a failure (message carried verbatim).
    """
    logger = log or _log
    ctx = multiprocessing.get_context("spawn")
    timings: dict[str, Any] = {"publish_retries": 0}
    mode = "publish"
    fill_snapshot: str | None = None
    for attempt in range(1, retries + 2):
        parent, child_end = ctx.Pipe()
        proc = ctx.Process(
            target=_child,
            args=(child_end, repo, group, base, forks, fill_message, attrs, mode, step_timeout_s),
            name=f"publish-{group}",
            daemon=True,
        )
        proc.start()
        child_end.close()
        try:
            if mode == "publish":
                if not parent.poll(step_timeout_s):
                    raise TimeoutError("fill")
                kind, *rest = parent.recv()
                if kind == "error":
                    raise RuntimeError(f"publish child failed: {rest[0]}")
                fill_snapshot, timings["commit_s"], timings["merge_s"] = rest
                if fault is not None and year_label is not None:
                    # The drill's death lands HERE, between the two commits — in the parent, which
                    # is the process the drill means to kill. The child is a daemon and dies with it.
                    fault.die_between_commits(group, year_label, log=logger)
                parent.send("go")
            if not parent.poll(step_timeout_s):
                raise TimeoutError("mark")
            kind, *rest = parent.recv()
            if kind == "error":
                raise RuntimeError(f"publish child failed: {rest[0]}")
            marked, timings["attrs_commit_s"] = rest
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()
            return marked, timings
        except TimeoutError as step:
            proc.kill()
            proc.join(timeout=30)
            timings["publish_retries"] += 1
            with suppress(Exception):
                logger.critical(
                    "PUBLISH WEDGED: the %s step for %s did not return in %.0f s (attempt %d). The child was "
                    "killed; its thread stacks are on its stderr. %s",
                    step,
                    group,
                    step_timeout_s,
                    attempt,
                    "Retrying once on a fresh session." if attempt <= retries else "No retries left.",
                )
            if attempt > retries:
                raise PublishWedgedError(
                    f"the publish {step} step for {group} wedged {attempt} times; the cell is retained for resume"
                ) from None
            # The fill landed and only the mark wedged: retry the mark alone, so the shards are
            # not committed twice.
            if str(step) == "mark" and fill_snapshot is not None:
                mode = "mark"
        finally:
            parent.close()
    raise AssertionError("unreachable")  # pragma: no cover


def run_forks(
    session: icechunk.Session,
    worker_fn: Callable[[dict[str, Any]], Any],
    payloads: list[dict[str, Any]],
    *,
    progress_interval_s: float = PROGRESS_INTERVAL_S,
    fork_stall_timeout_s: float = FORK_STALL_TIMEOUT_S,
    unit: str = "partitions",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    catch_up: Callable[[], str] | None = None,
) -> tuple[dict[str, Any], list[Any], bool]:
    """The fork phase alone: fork, run every payload, hand back the finished forks.

    Returns ``(telemetry, results in payload order, catch_up_wedged)``. ``catch_up_wedged`` is
    True when the periodic catch-up thread was still inside a call after the workers finished
    (2026-08-31's failure): the coordinator's session is then unsafe to touch, which the callers
    handle — :func:`run_forked` re-homes, :func:`write_year_shards` never commits from it anyway.

    **A stalled partition is re-run once, on a fresh pool, keeping every finished fork.** The
    watchdog's stall (:data:`FORK_STALL_TIMEOUT_S`) used to fail the cell and discard hours of
    shard writes; now the finished forks are salvaged, the stuck worker is killed, and only its
    partition is written again (whole-shard overwrites, so a re-run is safe). A second stall on
    the same partition raises :class:`ForkPhaseStalledError` — the fault is then deterministic and
    the cell is retained for resume.
    """
    t0 = time.monotonic()
    fork = session.fork()
    # Copies, not mutation: callers keep their payload dicts fork-free.
    payloads = [
        {**payload, "fork": fork, "worker_index": i, "progress_interval_s": progress_interval_s}
        for i, payload in enumerate(payloads)
    ]
    catch_ups: Counter[str] = Counter()
    tally_lock = threading.Lock()

    def _tick() -> None:
        if catch_up is None:
            return
        outcome = catch_up()
        with tally_lock:  # the timer thread and this one both reach it
            catch_ups[outcome] += 1

    abort = threading.Event()
    # WORKERS FINISHED, not "we reached the exit". `ticking` raises from its `finally`, which
    # runs on the failure path too — and an exception raised there REPLACES the body's own. So a
    # worker that died while the ticker happened to be wedged would arrive at the handler looking
    # exactly like a clean run whose catch-up hung. This flag is the only thing that tells the two
    # apart.
    workers_finished = False
    catch_up_wedged = False
    rerun: list[int] = []
    try:
        with ticking(CATCH_UP_INTERVAL_S, _tick if catch_up is not None else None, abort=abort):
            if len(payloads) == 1:
                results = [worker_fn(payloads[0])]
            else:
                run_pool = functools.partial(
                    _run_pool,
                    worker_fn,
                    payloads,
                    progress_interval_s=progress_interval_s,
                    fork_stall_timeout_s=fork_stall_timeout_s,
                    unit=unit,
                    log=log,
                    abort=abort,
                )
                finished, rerun = run_pool(list(range(len(payloads))))
                if rerun:
                    (log or _log).warning(
                        "Fork phase stalled on %d of %d %s; keeping the %d finished fork(s) and re-running the "
                        "stalled partition(s) %s once on a fresh pool.",
                        len(rerun),
                        len(payloads),
                        unit,
                        len(finished),
                        rerun,
                    )
                    again, still = run_pool(rerun)
                    finished.update(again)
                    if still:
                        raise ForkPhaseStalledError(
                            f"{unit} {still} stalled again on the re-run; the fault is deterministic and this "
                            f"cell is retained for resume"
                        )
                results = [finished[i] for i in range(len(payloads))]
            workers_finished = True
    except CatchUpDidNotStopError:
        if not workers_finished:
            raise
        # THE WORKERS ARE DONE AND THE SESSION IS NOT SAFE. The write survives; only the
        # coordinator's session is lost, and the caller decides what to commit from instead.
        catch_up_wedged = True
    telemetry: dict[str, Any] = {
        "workers": [stats for _, stats in results],
        "wall_s": round(time.monotonic() - t0, 3),
    }
    if rerun:
        telemetry["partitions_rerun"] = rerun
    if catch_up is not None:
        telemetry["catch_ups"] = dict(catch_ups)
    return telemetry, results, catch_up_wedged


def run_forked(
    session: icechunk.Session,
    worker_fn: Callable[[dict[str, Any]], Any],
    payloads: list[dict[str, Any]],
    *,
    progress_interval_s: float = PROGRESS_INTERVAL_S,
    fork_stall_timeout_s: float = FORK_STALL_TIMEOUT_S,
    unit: str = "partitions",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    catch_up: Callable[[], str] | None = None,
    rehome: Callable[[], icechunk.Session] | None = None,
) -> tuple[dict[str, Any], icechunk.Session]:
    """Fork ``session``, run ``worker_fn`` over ``payloads``, merge the forks back.

    The shared coordinator scaffolding for cooperative writes: each payload is
    shipped to ``worker_fn`` with three keys added (a pickled copy per spawned
    worker; caller dicts are not mutated) — ``"fork"``, the forked session to
    write into; ``"worker_index"``, the payload's position (also its stats and
    merge position); and ``"progress_interval_s"``, so a worker body that
    self-reports does so against the same clock as the coordinator. The worker
    writes into its fork and returns ``(fork, stats)`` — the fork for the
    coordinator to merge, and a JSON-serialisable dict of whatever the worker
    measured about its own run (``{}`` when it measured nothing). One payload
    runs in-process; more spawn a process pool (``spawn`` context — workers must
    be module-level functions and payloads picklable).

    Every payload gets its own process, so the pool never queues and the whole
    write is as long as its slowest band. Hence progress on a TIMER rather than
    per completion: waiting on completions alone says nothing until the first band
    lands, which on a dense zone is the bulk of the write. ``progress_interval_s``
    is the reporting period; a single payload runs in-process and the coordinator
    reports nothing, having no concurrency to describe — a worker body's own
    progress lines are then the only signal. ``unit`` and ``log`` are the
    coordinator lines' payload noun and destination (see :func:`_await_forks`).

    Returns the write's telemetry rather than nothing, because this is the only
    scope that sees all three of the fork, the workers, and the merge:

    * ``workers`` — the per-worker stats dicts, in payload order (a stats entry's
      index IS its payload's index, matching how forks are merged).
    * ``wall_s`` — fork creation through merge completion.
    * ``merge_s`` — the merge alone.
    * ``catch_ups`` — tally of :func:`catch_up_to_branch` outcomes, when ``catch_up`` was
      given. Reported because the fix it implements is invisible in a healthy run: a commit
      that does not stall looks the same whether the session was kept current or simply got
      lucky. The tally is the only evidence that it ran.
    * ``rehomed`` — True when a wedged catch-up forced the forks onto a fresh session. Present
      only on that path, so its absence is the normal case rather than a false negative.

    ``rehome`` is the escape hatch for a catch-up that wedges: called only when the workers all
    finished and the timer then refused to stop, it must return a fresh session to merge into.
    Without it that case fails the fill, discarding hours of shard writes. Returning the session
    used is why this returns a PAIR — after a re-home it is no longer the one passed in.

    ``catch_up`` is called on a timer for the whole fork phase and NOT again afterwards. The
    periodic calls are the entire point: they keep each catch-up short, where a single deep one
    would walk the same distance, through the same call, that the commit would have. No final
    synchronous call, deliberately — it would sit outside the timer with nothing bounding it, and
    a hang there is silent where a hung COMMIT at least raises the stall alarm. See
    :func:`~tessera_embeddings.storage.session_catch_up.catch_up_to_branch` for why an assembly
    needs this and when it deliberately refuses.
    """
    telemetry, results, catch_up_wedged = run_forks(
        session,
        worker_fn,
        payloads,
        progress_interval_s=progress_interval_s,
        fork_stall_timeout_s=fork_stall_timeout_s,
        unit=unit,
        log=log,
        catch_up=catch_up,
    )
    rehomed = False
    if catch_up_wedged:
        # Separable, and separating them is the whole point: the write survives, only the session
        # is lost. Re-home the finished forks onto one no other thread holds; without `rehome`
        # this is a failed cell with hours of shard writes discarded.
        if rehome is None:
            raise CatchUpDidNotStopError(
                "a catch-up was still running after the workers finished; the session cannot be merged or "
                "committed while it is in use, and this caller gave no way to re-home"
            )
        session = rehome()
        rehomed = True
    t_merge = time.monotonic()
    session.merge(*(fork_result for fork_result, _ in results))
    telemetry["merge_s"] = round(time.monotonic() - t_merge, 3)
    telemetry["wall_s"] = round(telemetry["wall_s"] + telemetry["merge_s"], 3)
    if rehomed:
        telemetry["rehomed"] = True
    return telemetry, session


def _group_node(store: Any, group: str) -> zarr.Group:  # noqa: ANN401 — icechunk store handle
    """Open a repo's zarr root and return the named group node (typed as Group)."""
    return cast(zarr.Group, zarr.open_group(store, mode="a")[group])


def read_years_complete(node: zarr.Group) -> list[int]:
    """A group's ``years_complete`` attr as a sorted list of ints (the one parser)."""
    raw = node.attrs.get("years_complete", [])
    return sorted(int(y) for y in raw) if isinstance(raw, list) else []


def run_provenance(
    existing: object,
    year: int,
    run_id: str,
    *,
    empty: bool = False,
    radar_coverage: dict | None = None,
    optical_skips: dict | None = None,
    input_coverage: dict | None = None,
    code: dict | None = None,
) -> dict:
    """Merge a per-year run record into a group's ``runs`` attr (the schema's one owner).

    ``input_coverage`` answers the question the calendar-year guarantee below does NOT: not
    what window was *requested*, but **how much of it the input actually held** — per source
    store, the months present of those required and the first/last in-window date, plus
    whether the every-month rule was relaxed
    (:func:`~tessera_embeddings.inference.data_loading.check_time_window_coverage` owns the
    shape). A cell can be filled from a partial input through that relaxation and still be
    marked complete, and the source mosaics are deleted once it lands — so without this field
    "was this year built on a full year of imagery?" is answerable only until cleanup runs.
    Recorded for every fill rather than only relaxed ones, because a field that appears only
    on suspect cells cannot be used to find them.

    ``code`` is which build produced the cell
    (:func:`~tessera_embeddings.config.environment.code_identity`), resolved here when the
    caller does not pass one. **Recorded, never compared** — a mid-campaign change is a normal
    event, and a value that differs between cells is a diagnostic aid rather than a condition.

    Both fill paths use this — the shard write (:func:`write_year_shards`) and
    the no-data marking (``campaign.mark_zone_year_empty``) — so the provenance
    record shape can only change in one place. The record carries no REQUESTED window: the
    store GUARANTEES calendar-year slots (the zone-fill gate rejects any window
    that is not exactly Jan-Dec of the slot's year), and each slot's true interval
    is stated by the seeded ``time_bnds`` CF-bounds variable.

    ``radar_coverage`` records how much of the year's embedded area had no radar, or
    little of it. It belongs PER YEAR rather than per zone because radar coverage is a
    property of what was acquired, not of the terrain: one year of a zone can be
    radar-free where another is not, so a zone-level figure would be wrong for at least
    one of them. Exact per-pixel counts already live in the store's
    ``s1_asc_obs_count``/``s1_desc_obs_count`` arrays; this is the summary that makes the
    question answerable without reading a zone-sized grid.

    ``optical_skips`` records the live tiles the fill resolved to a SKIP — no pixel
    survived the validity filter, so nothing was staged and the tile published as fill.
    Fill is also what ocean reads as, so without this field a consumer of a completed
    year cannot tell "no valid optical data" from "not land". Per year for the same
    reason ``radar_coverage`` is: a skip is a property of what the year's acquisitions
    yielded, not of the terrain
    (:func:`~tessera_embeddings.inference.assembly.summarise_optical_skips` owns the
    dict's shape). A summary of ZERO skips is recorded rather than omitted — it
    affirms that every live tile staged data, a distinct fact from a caller that never
    resolved the live set, which passes ``None`` and is recorded as nothing. A year
    marked ``empty`` carries no ``optical_skips`` (normalised here, the schema owner):
    the flag already states the whole live footprint is fill, and the label list would
    restate the land mask at zone size.
    """
    record: dict = {"run_id": run_id, "assembled_at": datetime.now(UTC).isoformat()}
    if empty:
        record["empty"] = True
    if radar_coverage:
        record["radar_coverage"] = dict(radar_coverage)
    if optical_skips and not empty:
        record["optical_skips"] = dict(optical_skips)
    if input_coverage:
        record["input_coverage"] = dict(input_coverage)
    # Resolved HERE rather than threaded from the flow: it is an ambient fact about the process,
    # not a decision any caller makes, and threading it would put an argument on five signatures
    # that have no opinion about it. Callers may pass one to pin it.
    resolved_code = code if code is not None else code_identity()
    if resolved_code:
        record["code"] = dict(resolved_code)
    return {**(dict(existing) if isinstance(existing, dict) else {}), str(year): record}


def commit_year_attrs(
    repo: icechunk.Repository,
    group: str,
    year_label: int,
    *,
    run_id: str | None = None,
    empty: bool = False,
    radar_coverage: dict | None = None,
    optical_skips: dict | None = None,
    input_coverage: dict | None = None,
    tries: int = 8,
    skip_if_marked: bool = False,
) -> str:
    """Advance one year's ``years_complete``/``runs`` in its own small commit, retrying.

    The single writer of those two attrs, and the reason concurrent fills of the SAME
    zone group are safe.

    **Why a separate commit.** Chunk data for different years of one zone is strictly
    disjoint — every chunk and shard is 1 in the time dimension — so those writes always
    rebase cleanly. The only thing that ever collided was these two attrs, because
    icechunk's :class:`~icechunk.ConflictDetector` treats attributes as an opaque value
    and cannot merge them. Bundling them into the shard commit meant a collision threw
    away the whole assembly; here it throws away a sub-second commit.

    **Why re-reading and retrying is CORRECT rather than hopeful.** Both attrs are keyed
    by year and each writer only ever inserts its OWN key — ``years_complete`` is a set
    union, ``runs`` a per-year dict insert — so there is no semantic conflict to resolve:
    a loser that re-reads the winner's value and re-applies its own key produces exactly
    the state both writers intended, in either order. That makes this a plain
    optimistic-concurrency loop rather than a lossy merge. Each attempt opens a FRESH
    session, so it cannot re-apply onto a stale read.

    **The two callers want different idempotency, and the difference is deliberate.**
    ``skip_if_marked=True`` returns the branch tip untouched when the year is already in
    ``years_complete``, EVEN IF a different ``run_id`` was passed — that is what
    ``mark_zone_year_empty`` needs, because a re-mark of an empty cell must not mint a new
    snapshot: :func:`~tessera_embeddings.storage.campaign.tag_zone_year` refuses to move a
    tag, so a new snapshot would leave the existing zone-year tag pointing at an ancestor
    and the original provenance is the one worth keeping. The default (``False``) records
    the new run, which is what a genuine refill through :func:`write_year_shards` means:
    shards were rewritten, so the provenance should say by which run.
    """
    for attempt in range(1, tries + 1):
        session = repo.writable_session("main")
        node = _group_node(session.store, group)
        done = read_years_complete(node)
        if year_label in done and (run_id is None or skip_if_marked):
            return repo.lookup_branch("main")
        if year_label not in done:
            node.attrs["years_complete"] = sorted([*done, year_label])
        if run_id is not None:
            node.attrs["runs"] = run_provenance(
                node.attrs.get("runs"),
                year_label,
                run_id,
                empty=empty,
                radar_coverage=radar_coverage,
                optical_skips=optical_skips,
                input_coverage=input_coverage,
            )
        try:
            return commit_with_rebase(session, f"mark {group} year {year_label} complete")
        except icechunk.RebaseFailedError:
            if attempt == tries:
                raise
            # Another year of THIS group committed between our read and our commit. Re-read
            # and re-apply; the loop is bounded so a genuine defect still surfaces.
            continue
    raise AssertionError("unreachable")  # pragma: no cover


def _year_label(node: zarr.Group, year_index: int) -> int:
    """Read the calendar year at ``year_index`` from a group's time coordinate.

    Decodes via :func:`read_time_values` so a foreign store with non-TIME_ENCODING
    units errors loudly instead of yielding an epoch-adjacent bogus year.
    """
    return year_of(read_time_values(node)[year_index])


def partition_round_robin(items: list, n: int) -> list[list]:
    """Round-robin partition ``items`` into up to ``n`` non-empty lists."""
    parts = [items[i::n] for i in range(n)]
    return [p for p in parts if p]


def _write_shards_worker(payload: dict[str, Any]) -> Any:  # noqa: ANN401 - returns (ForkSession, stats)
    """Write assigned shards into a forked session; return ``(fork, stats)`` for merge.

    Each shard is timed into two phases. ``read`` is ``source.load`` — in
    production an object-store fetch of a staged tile, though a source may
    instead build its blocks (a cleared position's fill block, a synthetic test
    source), in which case the "read" is that construction. ``write`` is the
    raw-zarr region assignment, inside which the codec pipeline encodes the shard
    AND the store uploads it — fused in one call this worker cannot see into, so
    the phase's CPU/wall split (:class:`PhaseTimer`) is the only honest
    decomposition: CPU seconds bound the encode cost, the remainder is time
    blocked on the store. ``bytes`` is the uncompressed block bytes handed to
    zarr — the logical write volume, not what landed on the wire.

    Progress within the assignment is this worker's own to report: the
    coordinator sees only whole payloads complete, so without these lines a
    partition is silent for its entire life. Reported on a TIMER (the payload's
    ``progress_interval_s``) rather than every N shards, because shard cost
    varies and a count-based cadence would speed up and slow down with the very
    thing an operator is trying to observe. Workers are separate spawned
    processes with no shared counter, so each line carries the worker's own
    index and done/total for a reader to aggregate. These lines reach the
    process's log stream only, never the Prefect API — a spawned worker has no
    run logger to route through; :func:`run_forked` configures logging in each
    child, without which none of them would exist.
    """
    fork = payload["fork"]
    group = payload["group"]
    year = int(payload["year_index"])
    shard_px = int(payload["shard_px"])
    source: ShardSource = payload["source"]
    worker_index = payload.get("worker_index", 0)
    progress_interval_s = float(payload.get("progress_interval_s", PROGRESS_INTERVAL_S))
    total = len(payload["shards"])

    node = _group_node(fork.store, group)
    arrays: dict[str, zarr.Array] = {}
    timer = PhaseTimer()
    tiles = writes = nbytes = 0
    last_report = time.monotonic()
    # Publish the DENOMINATOR before the first shard. The coordinator sums totals across workers,
    # so until every worker has reported once its percentage is measured against a short total —
    # and a worker that finishes inside one reporting interval would never report at all.
    report_shard_progress(worker_index, 0, total)
    for sy, sx in payload["shards"]:
        if time.monotonic() - last_report >= progress_interval_s:
            _log.info("Assembly worker %d progress: %d/%d shards written (%s)", worker_index, tiles, total, group)
            report_shard_progress(worker_index, tiles, total)
            last_report = time.monotonic()
        with timer.phase("read"):
            blocks = source.load((sy, sx))
        tiles += 1
        for var, block in blocks.items():
            arr = arrays.get(var)
            if arr is None:
                arr = arrays[var] = cast(zarr.Array, node[var])
            y0, x0 = sy * shard_px, sx * shard_px
            h, w = block.shape[1], block.shape[2]
            # Trailing dims (band) not indexed are written in full, so one
            # assignment covers both the 3-D and 4-D arrays.
            with timer.phase("write"):
                arr[year : year + 1, y0 : y0 + h, x0 : x0 + w] = block
            writes += 1
            nbytes += block.nbytes
    # And the final count, which no timed checkpoint reaches: the loop exits without one, so a
    # finished worker's slot would sit at its last checkpoint and understate the total for as long
    # as any slower worker kept the coordinator reporting.
    report_shard_progress(worker_index, tiles, total)
    return fork, {"tiles": tiles, "writes": writes, "bytes": nbytes, **timer.stats()}


def write_year_shards(
    repo: icechunk.Repository,
    group: str,
    year_index: int,
    source: ShardSource,
    *,
    n_workers: int = 1,
    shard_px: int = SHARD_PX,
    commit_msg: str | None = None,
    run_id: str | None = None,
    radar_coverage: dict | None = None,
    optical_skips: dict | None = None,
    input_coverage: dict | None = None,
    empty: bool = False,
    telemetry: dict[str, Any] | None = None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    fault: ArmedFault | None = None,
) -> str:
    """Fill one (zone, year) with whole shards from ``source`` in one commit.

    Forks the session, writes the source's live shards across ``n_workers``
    (in-process when 1, else spawned processes), merges, advances
    ``years_complete``, and commits via :func:`commit_with_rebase`.
    When ``run_id`` is given, per-year run provenance (:func:`run_provenance`)
    is merged into the group's ``runs`` attr in the same commit — read and
    written inside THIS writable session, so a commit landing between a
    caller's earlier probe and this write cannot be silently clobbered.

    **Concurrency contract:** concurrent fills of different groups rebase cleanly, and so do
    concurrent fills of the SAME group in different years. Chunk data never collides —
    every chunk and shard is 1 in the time dimension, so different years of one zone write
    strictly disjoint objects — and the two group attrs that DO collide commit separately
    and retry via :func:`commit_year_attrs`, which is correct because each writer only
    inserts its own year's key.

    So this issues TWO commits: the shards, then the year's attrs. A consequence worth
    knowing: if the shard commit lands and the attr commit then exhausts its retries, the
    year holds data but is not marked complete. The work list reads the marks, so that cell
    simply looks pending and a retry re-writes the same shards (a whole-shard overwrite) and
    re-marks.

    ``empty`` records the year as holding no data, for the case where every live tile
    resolved to a skip. Such a year still writes shards — fill over the whole live
    footprint, so a previous attempt's data cannot survive under this run's completion
    mark — so it comes through here rather than through ``mark_zone_year_empty``, which
    writes the attrs alone.

    ``radar_coverage`` and ``optical_skips`` are the year's per-run coverage summaries,
    recorded on the provenance entry (see :func:`run_provenance` for what each means
    and the ``empty``-year normalisation).

    ``telemetry`` is an out-parameter: pass a dict to receive the fill's timing
    facts — the per-worker stats and ``wall_s``/``merge_s`` from
    :func:`run_forked`, its ``catch_ups`` tally, plus ``commit_s`` and
    ``attrs_commit_s`` (each measured around its commit). An out-parameter
    rather than a changed return, so the snapshot id the callers and tags key on
    stays a plain string; the caller that wants a summary record owns emitting it.

    ``log`` is where the coordinator's progress lines go while the fill's forks
    are outstanding — the longest phase of a fill. A caller inside a flow passes
    its run logger so that phase is visible to the orchestrator; ``None`` keeps
    the lines on this module's logger, which reaches only the process's own log
    stream. Workers additionally self-report within their partitions
    (:func:`_write_shards_worker`), to their own process streams.

    ``fault`` is the supervised-drill hook for the gap between the two commits, a no-op
    unless the run was armed for exactly that fault and exactly this cell
    (:mod:`tessera_embeddings.config.fault_injection`). It exists because the gap is the
    one state above that no operator can produce deliberately — bounded by two commits of
    one function, there is nothing outside the process to aim a kill at, and a state
    documented as benign but never observed is an assumption.

    Returns the ATTR commit's snapshot id — a tag must point at a state where the
    year is both written and marked.
    """
    session = repo.writable_session("main")
    shards = list(source.live_shards())
    if not shards:
        raise ValueError(f"source has no live shards for {group} year_index={year_index}")

    payloads: list[dict[str, Any]] = [
        {"group": group, "year_index": year_index, "shards": part, "source": source, "shard_px": shard_px}
        for part in partition_round_robin(shards, max(1, n_workers))
    ]
    # `unit` below says "tile partitions" because that is what these payloads are; "band writes"
    # would describe the OTHER caller of `run_forked`.
    #
    # CAPTURED HERE, before anything forks. If the catch-up wedges, this is the one base that
    # can still be read: the session itself is held by a thread we cannot stop, and asking it
    # for its `snapshot_id` would reach into exactly the object we have given up on. It is also
    # the safe answer — the range base..tip is a superset of what was skipped, so checking it
    # can only refuse more often than strictly necessary, never less.
    base_before_forking = session.snapshot_id
    year_label = _year_label(_group_node(session.store, group), year_index)
    fill, forks, catch_up_wedged = run_forks(
        session,
        _write_shards_worker,
        payloads,
        unit="tile partitions",
        log=log,
        # Keep the session current WHILE the workers write. The commit no longer happens from
        # this session (see below), so the catch-up's remaining value is the depth telemetry it
        # produces — and a wedged one costs a daemon thread, not the write.
        catch_up=lambda: catch_up_best_effort(repo, session, group, log=log),
    )
    if catch_up_wedged:
        (log or _log).warning(
            "The catch-up for %s wedged and cannot be stopped; the coordinator's session is abandoned. "
            "The publish below runs from a fresh session regardless, so the write is unaffected.",
            group,
        )
    # THE PUBLISH RUNS IN A CHILD THE COORDINATOR CAN KILL, from a fresh session at the tip after
    # the conflict check over base..tip (#165's re-home, made the normal path). The 2026-09-04
    # fills parked their coordinator thread inside icechunk with the shard write complete and
    # nothing able to unwind it; a wedged child is killed and the step retried once, and the
    # forks — the hours of shard writes — never leave this process. The between-commits drill
    # hook runs in the parent, between the child's two phases.
    # ONE publication, both commits: the fleet's spacing slot is held from the shard commit
    # through the completion mark — both happen in the child below — and then for a full
    # spacing interval, so the next writer's publication cannot land inside one catch-up
    # interval of this one. See `publication_spacing`. A no-op when no gate is installed.
    with publication(log=log):
        snapshot, timings = publish_forks_in_child(
            repo,
            group,
            [fork_result for fork_result, _ in forks],
            base=base_before_forking,
            fill_message=commit_msg or f"fill {group} year {year_label}",
            attrs={
                "year_label": year_label,
                "run_id": run_id,
                "radar_coverage": radar_coverage,
                "optical_skips": optical_skips,
                "input_coverage": input_coverage,
                "empty": empty,
            },
            fault=fault,
            year_label=year_label,
            log=log,
        )
    if telemetry is not None:
        telemetry.update(
            workers=fill["workers"],
            fill_wall_s=fill["wall_s"],
            merge_s=timings.get("merge_s", 0.0),
            # Carried so ASSEMBLY_SUMMARY records whether the session was kept current, and how
            # often the guard refused. A healthy commit looks identical either way, so without
            # this the fix is unobservable in production.
            catch_ups=fill.get("catch_ups", {}),
            # True when the catch-up ticker wedged and the coordinator's session was abandoned.
            # The publish always runs from a fresh session now, so this records the WEDGE, not a
            # different commit path; kept under its historical name for the readers of the record.
            rehomed=catch_up_wedged,
            # The recovery counters: partitions written twice because their worker stalled, and
            # publish steps retried in a fresh child because the first wedged. Zero on a healthy
            # cell; anything else is a wedge that was survived and should be looked at.
            partitions_rerun=fill.get("partitions_rerun", []),
            publish_retries=timings.get("publish_retries", 0),
            commit_s=timings.get("commit_s"),
            attrs_commit_s=timings.get("attrs_commit_s"),
        )
    return snapshot
