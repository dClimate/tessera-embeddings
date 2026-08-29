"""Shard-aligned, land-masked writer for the global store (ADR-008 D3/D6).

Fills one (zone, year) with whole shards in a single Icechunk commit. The
d3v2-verified write path: each worker reads its assigned shard *sources* and
writes each shard's region in one raw-zarr assignment - real data in land inner
chunks, fill (elided by the sharding codec) elsewhere - so every shard object is
emitted once, no read-modify-write, no dense nodata.

Cooperative fork/merge: the coordinator forks the session, workers write into
their fork, the coordinator merges and makes **one commit per (zone, year)**,
updating ``years_complete`` in the same commit (D1). Commits are UNGATED. They do
contend on the branch-tip CAS -- all 120 zone groups share one repo -- but run 1
measured that as 2.2 s at 16 simultaneous committers and 15 s at 120, with zero
unresolvable conflicts at every N. See
``context_docs/design/commit-gate-removal-2026_08.md``.

A :class:`ShardSource` decouples the writer from *where* shard data comes from
(staged inference files in production; synthetic in tests), and must be picklable
so it can be shipped to spawned workers. :func:`run_forked` is the shared
fork → parallel-write → merge scaffolding; the single-ROI assembly engine
(:mod:`tessera_embeddings.inference.assembly`) drives it with a different
worker body.

Progress is reported at two levels, because no single scope can see both. The
coordinator (:func:`_await_forks`) knows the total and states, on a timer, how
many payloads are still outstanding — but a payload only completes near the end
of the write, so the coordinator alone cannot draw a curve. Each worker
(:func:`_write_shards_worker`) states its own within-payload progress on the
same timer; workers are separate spawned processes with no shared counter, so
the per-worker lines are what a log reader aggregates into the write-wide curve.
Routing differs too: the coordinator logs through the caller-supplied ``log``
(a Prefect flow passes its run logger down, which is the only route to the
Prefect API — this module's own logger reaches only the process's log stream),
while a worker can only ever use the module logger of its own spawned process,
so worker lines appear in the container's log stream alone.
"""

from __future__ import annotations

import ctypes
import logging
import multiprocessing
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from typing import Any, Protocol, cast

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.environment import code_identity, configure_logging
from tessera_embeddings.config.fault_injection import ArmedFault
from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage.zarr_store import read_time_values
from tessera_embeddings.storage.zone_grid import year_of

_log = logging.getLogger(__name__)

#: How often :func:`run_forked`'s coordinator states how many payloads are still
#: outstanding, and how often a worker states its own progress within one.
#: A forked write emits nothing between its start and its finish, so without a
#: periodic line a healthy long write and a hung one are indistinguishable in the
#: log — and the operator's only recourse is to guess. Set far enough apart to stay
#: quiet for short writes and close enough to bound how long a stall hides.
PROGRESS_INTERVAL_S = 300.0

#: Shards per minute the fork-write phase is credited with. Measured consistently, the
#: phase turns out to be a steady rate — the observed cells span barely a quarter — so this
#: is set at HALF the slowest one seen rather than at the slow end of a wide spread. The
#: margin is against variation the measurements could not see (a denser cell, a busier
#: campaign shape) rather than against a measured range. Rates and their derivation live in
#: ``context_docs/design/assembly-deadlines-2026_08.md``.
FORK_PHASE_SHARDS_PER_MINUTE = 20.0

#: Multiplier on the work-derived fork-phase budget. Deliberately large, and for what the
#: measurements CANNOT show rather than what they do: a handful of cells at one campaign
#: shape is far too narrow a base to predict a denser cell or a wider fleet. On top of the
#: halved rate above it puts the budget several times past the slowest fill on record, which
#: costs nothing when the work is healthy and buys the one property that matters — it will
#: not kill a slow assembly. So this is a backstop against a permanently stuck fork phase,
#: NOT a detector; the detector is :data:`POST_FORK_BUDGET_S`.
FORK_PHASE_SAFETY_FACTOR = 3.0

#: Floor under the fork-phase budget, so a sparse zone-year is never held to a budget of
#: minutes just because it had little to write.
FORK_PHASE_FLOOR_S = 3600.0

#: Ceiling on everything AFTER the forks return — pool teardown, merge, the
#: years-complete read, and both commits. Flat rather than work-derived because this
#: phase's work does not scale with the fill: it is one merge and two commits however
#: many shards were written. Two orders of magnitude above anything it has been measured
#: taking, and still an order of magnitude above the worst commit latency measured at
#: full campaign width (``commit-gate-removal-2026_08.md``), so a contention storm cannot
#: reach it. This is the bound that can actually catch a stall.
POST_FORK_BUDGET_S = 600.0


class AssemblyDeadlineError(RuntimeError):
    """A fill phase outlived its budget and was abandoned rather than waited on.

    Not a diagnosis. It says a phase stopped finishing, names which one and what it was
    allowed, and leaves the cause open. Callers treat it as a failed assembly, which
    retains the cell's mosaic so a retry resumes from its staged tiles.
    """


def fork_phase_budget_s(n_shards: int) -> float:
    """How long a fill's fork-write phase may run before it is abandoned.

    Derived from the WORK rather than from a clock: this phase writes ``n_shards``
    shards and its duration tracks that count, so one fixed cutoff would be too tight
    for a dense zone-year and too loose for a sparse one at the same time. Scaled from a
    rate set well under the slowest observed, padded by :data:`FORK_PHASE_SAFETY_FACTOR`,
    and floored so a small fill still gets a generous allowance.
    """
    scaled = 60.0 * n_shards / FORK_PHASE_SHARDS_PER_MINUTE * FORK_PHASE_SAFETY_FACTOR
    return max(FORK_PHASE_FLOOR_S, scaled)


class _Deadline:
    """A budget, and the instant it runs out once :meth:`start` marks the phase begun.

    An object rather than a timeout argument because the post-fork phase spans two
    functions — :func:`run_forked` tears the pool down and merges,
    :func:`write_year_shards` reads the group and makes both commits — and two
    independent ceilings would let a fill sit for twice the budget before anyone gave
    up. Passing the clock itself is what makes them share one. Starting is separate from
    construction for the same reason: the post-fork budget begins at a moment only
    ``run_forked`` can see, with the last fork's result in hand.
    """

    __slots__ = ("_at", "budget_s")

    def __init__(self, budget_s: float) -> None:
        self.budget_s = budget_s
        self._at: float | None = None

    def start(self) -> None:
        """Begin the budget. Called once, when the forks have returned."""
        self._at = time.monotonic() + self.budget_s

    def remaining_s(self) -> float:
        """Seconds left before the phase is abandoned; never negative."""
        if self._at is None:
            raise RuntimeError("post-fork deadline consulted before the forks returned")
        return max(0.0, self._at - time.monotonic())


def _run_before_deadline[T](deadline: _Deadline | None, fn: Callable[[], T], what: str) -> T:
    """Run ``fn``, and stop WAITING for it once ``deadline`` passes.

    ``None`` runs ``fn`` inline and waits forever, which is exactly what every caller
    that asks for no budget keeps.

    Otherwise ``fn`` runs on a single-use daemon thread and this returns control at the
    deadline whether or not it finished, raising :exc:`AssemblyDeadlineError`. The
    thread is not cancelled, and cannot be: these calls descend into icechunk's Rust
    extension, and Python cannot interrupt a thread blocked below the interpreter. So it
    is LEAKED, deliberately, and that is the whole trade:

    * What it buys — the caller's thread comes back. In the campaign runner that thread
      is a single-slot executor shared by every later cell, so one wedged fill otherwise
      stops its whole fill publishing for the rest of the run.
    * What it costs — the leaked thread still holds its icechunk session, its merged
      forks and their memory, and it is still a writer. If it ever unblocks it will
      commit, possibly after a retry of the same cell has already committed. Both write
      the same year's shards from the same staged tiles and ``years_complete`` inserts
      one key idempotently, so the visible cost is a duplicate write and a second
      provenance entry rather than a damaged year — but it is a live writer nobody is
      waiting on, which is why the budgets above are generous rather than tight.
    * Why it is affordable — the runner is 16 vCPU / 64 GiB and sits near a gigabyte
      resident, against tens of cells per fill. The thread is a daemon, so a leaked one
      cannot hold up interpreter exit.
    """
    if deadline is None:
        return fn()
    box: dict[str, Any] = {}
    finished = threading.Event()

    def _body() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:
            box["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=_body, name=f"assembly-{what}", daemon=True).start()
    if not finished.wait(timeout=deadline.remaining_s()):
        raise AssemblyDeadlineError(
            f"{what} had not returned {deadline.budget_s:.0f}s after the forks did, and was abandoned "
            f"(the thread doing it is leaked, not killed)"
        )
    if "error" in box:
        raise box["error"]
    return cast("T", box["value"])


class PhaseTimer:
    """Wall- and CPU-second accumulator for the phases of a fork worker.

    Two clocks per span because they answer different questions: wall time is how
    long a phase held the worker, CPU time is how much of that was computation.
    Their difference is time the phase spent *blocked* — on the object store,
    almost always — which a wall clock alone cannot distinguish from work, and
    that distinction is what the caller's summary record exists to expose. CPU
    time is process-wide (:func:`time.process_time`), so work the storage layer
    or a codec does on its own threads is still counted; the CPU figure is an
    upper bound on the phase's in-process compute, never an undercount.

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

    Commits are UNGATED. A fleet-wide committer limit used to wrap this call; it was
    removed because what it bounded is a SLOWDOWN, not a failure -- run 1 measured 2.2 s
    commits at 16 concurrent committers and 15 s at 120, with zero unresolvable
    conflicts at every N. See ``context_docs/design/commit-gate-removal-2026_08.md``.

    **Timed here, and here is the only place that sees every commit.** The removal's
    reopen criterion is commit LATENCY, and the obvious detector -- ``commit_s`` in
    ``ASSEMBLY_SUMMARY`` -- cannot see the dominant source of concurrent commits: a
    terminal cell marks itself through ``mark_zone_year_empty`` and returns without ever
    reaching ``assemble_global``, and terminal cells were **72 of the first 78**
    completions. One line per commit, so the volume is one per zone-year.
    """
    started = time.monotonic()
    snapshot = session.commit(message, rebase_with=icechunk.ConflictDetector(), rebase_tries=tries)
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
    budget_s: float | None = None,
) -> list[Any]:
    """Collect ``futures`` in submission order, logging what is still outstanding.

    Returns results positionally rather than by completion order, because a fork's
    position is its band and :meth:`icechunk.Session.merge` is given them as the
    caller's payloads were ordered. Re-raises the first failure in that same order,
    matching what ``Executor.map`` would have done.

    ``unit`` is the caller's name for one payload. What a payload holds is the
    caller's decision (northing bands for one, round-robin tile partitions for
    another), so a fixed noun here would misdescribe the work for all callers but
    one. ``log`` is where the lines go: the module logger reaches only the
    process's own log stream, so a caller inside a flow passes its run logger to
    make the wait visible to the orchestrator as well.

    ``budget_s`` bounds the whole wait: past it, whatever is still outstanding is
    abandoned with :exc:`AssemblyDeadlineError`. ``None`` waits forever, which is
    what a caller that names no budget keeps. Giving up HERE rather than around the
    call is what makes the fork phase cheap to abandon — the caller's handler already
    terminates the worker processes, and nothing they wrote is in the store until a
    fork is merged, so the only cost of the kill is unreferenced objects.
    """
    logger = log or _log
    started = time.monotonic()
    pending: set[Future] = set(futures)
    while pending:
        # Each wait is capped by what is LEFT of the budget, so the deadline lands when it
        # falls rather than up to one progress interval later. A phase that FINISHES inside
        # that overshoot is still a success — the budget exists to stop waiting on work that
        # will never finish, not to discard work that finished slightly late.
        timeout = progress_interval_s
        if budget_s is not None:
            timeout = min(timeout, max(0.0, budget_s - (time.monotonic() - started)))
        done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
        # SURFACE A FAILURE AS SOON AS IT LANDS. A fork dies on a deterministic fault — a
        # corrupt staged tile, a dtype the destination cannot hold — and every other fork
        # is going to hit the same wall or write shards that will be discarded anyway.
        # Collecting results only after `pending` emptied meant the coordinator sat through
        # the rest of a multi-hour assembly to learn something the first fork already knew.
        for future in done:
            if future.exception() is not None:
                future.result()  # re-raises, with the worker's traceback attached
        if pending and budget_s is not None and time.monotonic() - started >= budget_s:
            raise AssemblyDeadlineError(
                f"fork-write phase still had {len(pending)}/{len(futures)} {unit} outstanding after "
                f"{(time.monotonic() - started) / 60.0:.0f} min, past its {budget_s / 60.0:.0f} min budget"
            )
        if pending:
            # SHARDS, not payloads: a payload completes only when its worker returns and they all
            # return at the end, so the payload figure read 0/N for the whole write. `slots` is what
            # the workers have actually written; withheld until ALL of them have reported a total.
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


def run_forked(
    session: icechunk.Session,
    worker_fn: Callable[[dict[str, Any]], Any],
    payloads: list[dict[str, Any]],
    *,
    progress_interval_s: float = PROGRESS_INTERVAL_S,
    unit: str = "partitions",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    fork_budget_s: float | None = None,
    post_fork: _Deadline | None = None,
) -> dict[str, Any]:
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
    write is as long as its slowest band. That is why progress is reported on a
    TIMER rather than per completion: waiting on completions alone would say
    nothing until the first band landed, which on a dense zone is the bulk of the
    write. ``progress_interval_s`` is the reporting period; a single payload runs
    in-process and the coordinator reports nothing, having no concurrency to
    describe — a worker body's own progress lines are then the only signal.
    ``unit`` and ``log`` are the coordinator lines' payload noun and destination
    (see :func:`_await_forks`).

    Returns the write's telemetry rather than nothing, because this is the only
    scope that sees all three of the fork, the workers, and the merge:

    * ``workers`` — the per-worker stats dicts, in payload order (a stats entry's
      index IS its payload's index, matching how forks are merged).
    * ``wall_s`` — fork creation through merge completion.
    * ``merge_s`` — the merge alone.

    ``fork_budget_s`` and ``post_fork`` bound the two halves separately, and both
    default to unbounded. They are separate because the halves have nothing in common:
    the fork writes scale with the payloads and take hours, while what follows them is
    one pool teardown and one merge however large the write was. A single cutoff across
    both would have to clear the hours, which is far too loose to notice the seconds
    going wrong. ``post_fork`` is STARTED here, at the one moment that phase begins —
    the last fork's result in hand — and the caller then holds the same object over its
    own commits, so the two functions share one ceiling rather than one each.
    """
    t0 = time.monotonic()
    fork = session.fork()
    # Copies, not mutation: callers keep their payload dicts fork-free.
    payloads = [
        {**payload, "fork": fork, "worker_index": i, "progress_interval_s": progress_interval_s}
        for i, payload in enumerate(payloads)
    ]
    if len(payloads) == 1:
        # One payload runs in-process, so there is no worker process to terminate — the
        # fork budget is enforced the way the post-fork phase is, by abandoning a thread.
        # It has to be enforced SOMEHOW: a sparse cell partitions to a single payload
        # whatever `n_workers` says, and an unbudgeted one would block the caller's
        # finalizer exactly as the stall this exists for does.
        #
        # Cheaper to abandon here than after the forks: the worker writes into a fork that
        # is never merged, so nothing it produced can reach the store. What the process
        # path buys and this cannot is the KILL — an abandoned thread keeps running and
        # keeps writing fork objects. They are unreferenced either way.
        fork_deadline = None
        if fork_budget_s is not None:
            fork_deadline = _Deadline(fork_budget_s)
            fork_deadline.start()
        results = [_run_before_deadline(fork_deadline, lambda: worker_fn(payloads[0]), "fork-write phase")]
        if post_fork is not None:
            post_fork.start()
    else:
        ctx = multiprocessing.get_context("spawn")
        # `initializer` runs once per spawned child before any payload. A spawned process
        # inherits no logging config, so without it the root WARNING default discards every
        # INFO record a worker produces. Set HERE rather than in each worker body so a worker
        # added later cannot omit it. The single-payload path above needs nothing: it runs in
        # the coordinator, which is already configured.
        slots = ctx.Array("l", 2 * len(payloads), lock=False)
        ex = ProcessPoolExecutor(
            max_workers=len(payloads),
            mp_context=ctx,
            initializer=_init_fork_worker,
            initargs=(slots,),
        )
        try:
            futures = [ex.submit(worker_fn, payload) for payload in payloads]
            results = _await_forks(
                futures, progress_interval_s, unit=unit, log=log, slots=slots, budget_s=fork_budget_s
            )
        except BaseException:
            # Cancel what has not started, then TERMINATE what has. `cancel_futures` only
            # reaches queued work, so without the second step a multi-hour shard writer keeps
            # running — and keeps writing fork objects — after the coordinator has already
            # raised, and Python's own executor atexit hook then blocks interpreter shutdown
            # on it. A `with` block would be worse still: it joins every worker here, which is
            # the whole delay `_await_forks` exists to avoid.
            #
            # Killing them is safe because nothing they have written is IN the store. Workers
            # write into a fork; a fork becomes part of the repository only when the
            # coordinator merges it and commits, and neither happens on this path. Icechunk's
            # own guidance says as much — dropping a ForkSession without merging is its
            # documented way to orphan chunks deliberately. So the cost of a kill is
            # unreferenced objects, which garbage collection reclaims, and the cost of NOT
            # killing is CPU, S3 writes, and a retry racing the previous attempt in the same
            # process.
            #
            # `_processes` is private: ProcessPoolExecutor exposes no terminate API. Guarded
            # so a future Python that renames it degrades to the old wait-free shutdown
            # rather than masking the original failure with an AttributeError.
            # SNAPSHOT BEFORE SHUTDOWN. `ProcessPoolExecutor.shutdown` sets `_processes = None`
            # unconditionally — it drops references to objects holding file descriptors — so
            # reading it afterwards yields None and `.values()` raises AttributeError, masking
            # the assembly failure this handler exists to propagate. Worse than the leak it was
            # meant to fix, and a stub executor whose `shutdown` does not null the attribute
            # cannot catch it.
            procs = list((getattr(ex, "_processes", None) or {}).values())
            ex.shutdown(wait=False, cancel_futures=True)
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            raise
        # The post-fork clock starts HERE, with every fork's result in hand and before the
        # teardown, because that is the boundary the observed stall sits behind: the workers
        # had all finished and exited, and nothing after them ever completed.
        if post_fork is not None:
            post_fork.start()
        # Snapshot BEFORE the teardown, for the same reason the handler above does it:
        # `shutdown` sets `_processes = None` unconditionally.
        procs = list((getattr(ex, "_processes", None) or {}).values())
        try:
            _run_before_deadline(post_fork, ex.shutdown, "pool teardown")
        except AssemblyDeadlineError:
            # Kill the workers, but do NOT reuse the handler above: its first act is another
            # `ex.shutdown(...)`, which takes the shutdown lock the abandoned thread is still
            # holding — that would wedge this thread, the one the budget exists to free. The
            # terminate is worth doing on its own. A stuck `shutdown` is stuck in a JOIN, so
            # ending the processes is the thing most likely to release both it and the
            # executor's manager thread, which `concurrent.futures` joins at interpreter exit.
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            raise
    t_merge = time.monotonic()
    _run_before_deadline(post_fork, lambda: session.merge(*(fork_result for fork_result, _ in results)), "merge")
    done = time.monotonic()
    return {
        "workers": [stats for _, stats in results],
        "wall_s": round(done - t0, 3),
        "merge_s": round(done - t_merge, 3),
    }


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
    marked complete, and the source mosaics are deleted once it lands — so without this
    field "was this year built on a full year of imagery?" is answerable only until cleanup
    runs, and after that never again. Recorded for every fill rather than only relaxed ones,
    because a field that appears only on suspect cells cannot be used to find them.

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
    year cannot tell "no valid optical data" from "not land". It belongs per year for
    the same reason ``radar_coverage`` does: a skip is a property of what the year's
    acquisitions yielded, not of the terrain
    (:func:`~tessera_embeddings.inference.assembly.summarise_optical_skips` owns the
    dict's shape). A summary of ZERO skips is recorded rather than omitted, because it
    is an affirmative statement that every live tile staged data — a distinct fact from
    a caller that never resolved the live set, which passes ``None`` and is recorded as
    nothing. A year marked ``empty`` carries no ``optical_skips`` (normalised here, the
    schema owner): the flag already states that the whole live footprint is fill, and
    the label list would restate the land mask at zone size.
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
    # Resolved HERE rather than threaded from the flow, because it is an ambient fact about
    # the process, not a decision any caller makes — threading it would put an argument for it
    # on five signatures that have no opinion about it. Callers may pass one to pin it.
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
    union, ``runs`` a per-year dict insert. So there is no semantic conflict to resolve:
    a loser that re-reads the winner's value and re-applies its own key produces exactly
    the state both writers intended, in either order. That is what makes this a plain
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
    raw-zarr region assignment, inside which the codec pipeline encodes
    (compresses) the shard AND the store uploads it — the two are fused in one
    call this worker cannot see into, so the phase's CPU/wall split
    (:class:`PhaseTimer`) is the only honest decomposition: CPU seconds bound the
    encode cost, and the remainder is time blocked on the store. ``bytes`` is
    the uncompressed block bytes handed to zarr — the logical write volume, not
    what landed on the wire.

    Progress within the assignment is this worker's own to report: the
    coordinator sees only whole payloads complete, so without these lines a
    partition is silent for its entire life. Reported on a TIMER (the payload's
    ``progress_interval_s``) rather than every N shards, because shard cost
    varies — a count-based cadence would speed up and slow down with the very
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

    Concurrency contract (RELAXED 2026-07-30): concurrent fills of different groups
    rebase cleanly, and concurrent fills of the SAME group but different years are
    now safe too. This used to require "one fill per zone at a time", which is what
    made the campaign's years serial. Two changes to that reasoning:

    * Chunk data was never the problem — every chunk and shard is 1 in the time
      dimension, so different years of one zone write strictly disjoint objects.
    * The two group attrs that DID collide now commit separately and retry, via
      :func:`commit_year_attrs`, which is correct because each writer only inserts
      its own year's key.

    So this issues TWO commits: the shards, then the year's attrs. A consequence
    worth knowing: if the shard commit lands and the attr commit then exhausts its
    retries, the year holds data but is not marked complete. The work list reads the
    marks, so that cell simply looks pending and a retry re-writes the same shards
    (a whole-shard overwrite) and re-marks. That is strictly better than the previous
    behaviour, where a collision discarded the shards as well.

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
    :func:`run_forked`, plus ``commit_s`` and ``attrs_commit_s`` (each measured
    around its commit). An out-parameter
    rather than a changed return, so the snapshot id the callers and tags key on
    stays a plain string; the caller that wants a summary record owns emitting it.

    ``log`` is where the coordinator's progress lines go while the fill's forks
    are outstanding — the longest phase of a fill. A caller inside a flow passes
    its run logger so that phase is visible to the orchestrator; ``None`` keeps
    the lines on this module's logger, which reaches only the process's own log
    stream. Workers additionally self-report within their partitions
    (:func:`_write_shards_worker`), to their own process streams.

    ``fault`` is the supervised-drill hook for the gap between the two commits, and
    is a no-op unless the run was armed for exactly that fault and exactly this cell
    (:mod:`tessera_embeddings.config.fault_injection`). It exists because the gap is
    the one state above that no operator can produce deliberately: it is bounded by
    two commits of one function, so there is nothing outside the process to aim a kill
    at, and a state documented as benign but never observed is an assumption.

    BOUNDED IN TWO PARTS. The fork-write phase gets a budget derived from the shard
    count (:func:`fork_phase_budget_s`); everything after the forks return gets the much
    tighter :data:`POST_FORK_BUDGET_S`. Overrunning either raises
    :exc:`AssemblyDeadlineError` instead of waiting indefinitely — containment for a
    stall whose cause is not yet known, not a fix for it. What that costs, and what it
    leaves uncovered, is in ``context_docs/design/assembly-deadlines-2026_08.md``.

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
    # The payloads are round-robin partitions of the source's tiles — name them
    # that way; "band writes" would describe the OTHER caller of run_forked.
    #
    # TWO budgets, not one, because this fill has two phases of completely different
    # shape and only the second is where fills have been seen to stop. The fork writes
    # scale with the shard count and run for hours; everything after them is one merge
    # and two commits, fixed work that has never taken more than a couple of seconds.
    # A single cutoff would have to clear the hours and so could never notice the
    # seconds. The rates, the derivation and what is deliberately left outside both
    # budgets are in ``context_docs/design/assembly-deadlines-2026_08.md``.
    post_fork = _Deadline(POST_FORK_BUDGET_S)
    fill = run_forked(
        session,
        _write_shards_worker,
        payloads,
        unit="tile partitions",
        log=log,
        fork_budget_s=fork_phase_budget_s(len(shards)),
        post_fork=post_fork,
    )

    # Each of these names its own phase, so an abandoned fill says which call stopped
    # returning rather than only that one did — the cause is not yet known, and that
    # name is the first thing an operator has to go on.
    year_label = _run_before_deadline(
        post_fork, lambda: _year_label(_group_node(session.store, group), year_index), "years-complete read"
    )
    t_commit = time.monotonic()
    _run_before_deadline(
        post_fork, lambda: commit_with_rebase(session, commit_msg or f"fill {group} year {year_label}"), "shard commit"
    )
    if fault is not None:
        # The drill's death lands HERE, between the two commits, and nowhere else can
        # produce this state on purpose. Inert for any other fault or any other cell.
        fault.die_between_commits(group, year_label, log=log or _log)
    t_attrs = time.monotonic()
    # The per-year attrs go in their OWN commit (see `commit_year_attrs`), so a same-zone
    # collision costs a sub-second retry instead of this whole assembly. Return that
    # snapshot rather than the shard one: a tag must point at a state where the year is
    # both written AND marked.
    snapshot = _run_before_deadline(
        post_fork,
        lambda: commit_year_attrs(
            repo,
            group,
            year_label,
            run_id=run_id,
            radar_coverage=radar_coverage,
            optical_skips=optical_skips,
            input_coverage=input_coverage,
            empty=empty,
        ),
        "year-attrs commit",
    )
    if telemetry is not None:
        telemetry.update(
            workers=fill["workers"],
            fill_wall_s=fill["wall_s"],
            merge_s=fill["merge_s"],
            commit_s=round(t_attrs - t_commit, 3),
            attrs_commit_s=round(time.monotonic() - t_attrs, 3),
        )
    return snapshot
