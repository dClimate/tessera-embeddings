"""Frisky: an experimental Rust scheduler loaded onto the Dask cluster a provider built.

`Frisky <https://getfrisky.dev/>`_ reimplements Dask's scheduler and workers in Rust, with far
more telemetry. It is pre-1.0 and closed-source, so this module is the only place the pipeline
calls its API: a change in the library lands here. Opt-in per run (``use_frisky`` on the ingest
flows, ``frisky:`` in the plain runner's config); with it off nothing here runs. How to use it,
and its do's and don'ts: ``docs/frisky.md``. Why it is wired this way, and the evidence:
``context_docs/ingest/frisky-experiment.md``.

The providers build the same Dask cluster as always and :func:`hijack` loads Frisky onto it: a
Frisky scheduler inside the Dask scheduler process and a Frisky worker inside every Dask worker
process, late joiners included. Dask keeps everything Frisky does not replace — provisioning and
teardown, the Prefect task runner (the ingest task still runs as a Dask task on one worker), and
the two calls the ingest makes that Frisky's client lacks, ``register_plugin`` and ``run``. Those
still reach Frisky's tasks because they run in the same processes. :func:`connect` hands the
ingest a client that sends compute to Frisky and everything else to Dask.

Four behaviours differ from Dask and are handled here or by the providers:

- **Thread state.** Frisky destroys a thread's Python state after each task, so every
  ``threading.local`` cache in the read stack is rebuilt per task, and pyproj segfaults the worker
  on the first real image read (:func:`_pin_thread_state` says why). Dask restarts the worker, the
  next one dies the same way, and the ingest waits forever. :func:`hijack` pins each thread's
  state for the thread's life, as Python's own threads have it.
- **No adaptive scaling.** Dask sizes an adaptive fleet from its own task load, which a hijacked
  cluster no longer has, so it would shrink to the minimum and retire workers holding Frisky's
  data. ``ecs_cluster(frisky=True)`` fixes the fleet at ``max_workers`` instead.
- **Exception chains.** A task's exception keeps its ``__cause__`` only for classes tblib has
  registered, which happens once, when the worker imports Dask. Dask registers the failing chain
  on every failure and Frisky never does, so rasterio's GDAL errors, imported later, lose the cause
  every read-failure verdict is decided from. :func:`hijack` registers them on every worker after
  importing the ingest; with ``loader_failures.keep_causes_picklable`` (installed by the ingest)
  the whole chain arrives.
- **Graph pickling before the first task.** Dask's scheduler pickles each task as it dispatches
  it, while the fleet works. Frisky's client pickles the whole graph before it submits, while the
  fleet waits, so a task that pickles slowly costs idle fleet time. :func:`connect` makes the
  store write's slowest ones cheap (:func:`_picklable_merge_reduction`), and the S2 store write
  submits each date as a few graphs so the fleet starts while the rest convert
  (``storage.zarr_store.WRITE_SUBMISSION_GROUPS``).
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import gzip
import json
import logging
import pickle
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import frisky
import fsspec
import icechunk.dask
import numpy as np
import tblib.pickling_support
from distributed import Client, WorkerPlugin

#: The client methods Frisky implements and the ingest calls. Every other attribute of the client
#: :func:`connect` yields is Dask's; when Frisky's client grows one, adding it here is the change.
FRISKY_METHODS = frozenset({"compute", "persist", "submit", "map", "gather", "scatter"})

#: Most recent spans :func:`maybe_capture_telemetry` keeps in ``spans.json``, Frisky's own default
#: for its views. The scheduler assembles a query's whole answer in its own memory, so this bounds
#: what one query can add there; a longer run keeps its tail, like Dask's capped task stream.
SPANS_CAPTURE_LIMIT = 200_000

#: The same, with the span drain on. The drain already holds every task span, so the tail only has
#: to keep the bundle readable as it is: one page of Frisky's span API, about 7 s of an Iowa run.
SPANS_CAPTURE_LIMIT_DRAINED = 100_000

#: Seconds between the live snapshots :func:`maybe_capture_telemetry` takes while a run is going.
LIVE_SNAPSHOT_INTERVAL_S = 300.0

#: Spans each Frisky process keeps (``FRISKY_TRACING_CAPACITY``, Frisky's default 1,000,000),
#: set on the scheduler and every worker by ``ecs_cluster(frisky=True)``. At the default the
#: scheduler's buffer grew to 5.7 GiB of its 8 GiB on zone 35N and its process died at the end of
#: the run, and every span query slowed as buffers filled. 200,000 is still several minutes of a
#: worker's spans, ample for a drain each minute.
TRACING_CAPACITY = 200_000

#: Scheduler events per query, the most ``frisky observe events`` returns.
EVENTS_LIMIT = 2_000

#: Span names the drain keeps, as prefixes: task execution (the call, its GIL wait and its
#: deserialisation), data transfer, spill, and the client's own work turning each graph into
#: Frisky's tasks and submitting them. They are a fifth of Frisky's spans and hold every second of
#: task time; the rest is the scheduler's and comms' own bookkeeping.
SPAN_DRAIN_NAMES = ("worker.exec", "worker.transfer", "spill", "client")

#: Seconds between drains. Each process keeps its own span buffer (1,000,000 spans by default,
#: ``FRISKY_TRACING_CAPACITY``), which a worker fills in about an hour at Iowa's rate, so a drain a
#: minute leaves wide slack for one that fails.
SPAN_DRAIN_INTERVAL_S = 60.0

#: How far behind the present a drain stops, so a span still being recorded lands in the next one.
_SPAN_DRAIN_LAG_NS = 10_000_000_000

#: Time slice one span request covers. A whole minute of a busy zone run timed out in the Dask
#: dashboard's proxy (HTTP 504), and each miss made the retry bigger.
_SPAN_DRAIN_SLICE_NS = 15_000_000_000


#: Native ids of the Frisky threads whose Python thread state is pinned (see :func:`_pin_thread_state`).
_PINNED_THREADS: set[int] = set()


def _pin_thread_state() -> None:
    """Keep this thread's Python thread state for the thread's life, as Python's own threads do.

    Frisky enters Python with ``PyGILState_Ensure`` and ``PyGILState_Release`` around each piece of
    work, and the release destroys the thread state, taking every ``threading.local`` with it.
    pyproj keeps the PROJ context it frees there but its pointer in OS-thread storage, so the next
    task on that thread uses freed memory and segfaults the worker. One unmatched ``Ensure`` keeps
    the state alive; on a thread Python started it changes nothing.
    """
    thread = threading.get_native_id()
    if thread not in _PINNED_THREADS:
        ctypes.pythonapi.PyGILState_Ensure()
        _PINNED_THREADS.add(thread)


def _pinning_loads(loads: Any) -> Any:  # noqa: ANN401 — the wrapped pickle.loads
    """``pickle.loads``, pinning the calling thread first: the first Python a Frisky thread runs
    for any task is unpickling it, and a task's arguments can create PROJ contexts as they load.
    """

    @functools.wraps(loads)
    def pinned(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        _pin_thread_state()
        return loads(*args, **kwargs)

    pinned._tessera_pins = True  # type: ignore[attr-defined]
    return pinned


class _MatchDaskWorker(WorkerPlugin):
    """Give a worker's Frisky worker what its Dask worker has: thread state that outlives a task,
    and exception chains that survive pickling.
    """

    name = "tessera-frisky-match-dask-worker"

    def setup(self, worker: object) -> None:  # noqa: ARG002 — plugin interface
        """Runs before the Frisky worker starts; see :func:`hijack`."""
        # Frisky looks pickle.loads up when it first deserialises, after this runs. If an upgrade
        # binds it earlier the pin stops working, which test_frisky.py catches.
        if not getattr(pickle.loads, "_tessera_pins", False):
            pickle.loads = _pinning_loads(pickle.loads)
        # Imported for their side effect: tblib registers only the classes that exist when it
        # runs, and these two seed the ingest's import closure — rasterio, GDAL, odc, icechunk
        # and our own errors. Classes created later (botocore's per-service errors) are not
        # covered and arrive without their cause.
        from tessera_embeddings.ingest import s1_roi, s2_roi  # noqa: F401

        tblib.pickling_support.install()


def hijack(dask_client: Client) -> None:
    """Load Frisky onto the cluster behind ``dask_client`` (see the module docstring).

    :class:`_MatchDaskWorker` is registered first because Dask runs a worker's plugins in that
    order, late joiners included, and the pin must be in place before Frisky's worker
    deserialises its first task. Both plugins outlive ``dask_client``.
    """
    dask_client.register_plugin(_MatchDaskWorker())
    frisky.hijack(dask_client, connect_client=False)


class _HijackedClient:
    """A Dask client whose compute runs on Frisky: :data:`FRISKY_METHODS` go there, the rest to Dask."""

    def __init__(self, dask_client: Client, frisky_client: Any) -> None:  # noqa: ANN401 — frisky is untyped
        self.dask_client = dask_client
        self.frisky_client = frisky_client

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 — whatever the routed client returns
        return getattr(self.frisky_client if name in FRISKY_METHODS else self.dask_client, name)


def _frisky_address(dask_scheduler: Any) -> str:  # noqa: ANN401 — runs on the Dask scheduler
    """Where ``frisky.hijack`` recorded the Frisky scheduler's address."""
    return str(dask_scheduler.frisky_address)


def _meta_or_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """``icechunk.dask.computing_meta``'s wrapper, at module level so it pickles by reference."""
    if kwargs.get("computing_meta", False):
        return np.array([object()], dtype=object)
    return func(*args, **kwargs)


def _computing_meta(func: Callable[..., Any]) -> Callable[..., Any]:
    return functools.partial(_meta_or_call, func)


@contextlib.contextmanager
def _picklable_merge_reduction() -> Iterator[None]:
    """Make the store write's merge tasks pickle by reference while the block runs.

    ``icechunk.dask.session_merge_reduction`` wraps its two functions in a ``functools.wraps``
    closure. Plain pickle refuses it, so every task holding one, about 22,000 of an Iowa batch's
    47,000, is pickled by value with cloudpickle instead. Frisky's client pickles every task
    before it submits any, so that cost holds the whole fleet idle at the start of each write
    batch. The partial computes the same thing.
    """
    original = icechunk.dask.computing_meta
    icechunk.dask.computing_meta = _computing_meta
    try:
        yield
    finally:
        icechunk.dask.computing_meta = original


@contextlib.contextmanager
def connect(dask_client: Client, *, enabled: bool = True) -> Iterator[Client]:
    """Yield the ingest's client: Frisky's compute on a hijacked cluster, else ``dask_client`` itself.

    Creating the Frisky client also points ``dask.compute`` and bare ``.compute()`` at Frisky for
    this process, from any thread; closing it on exit hands them back to Dask. Typed as
    :class:`Client` because the ingest takes one, and :class:`_HijackedClient` provides every
    method it calls.
    """
    if not enabled:
        yield dask_client
        return

    frisky_client = frisky.Client(dask_client.run_on_scheduler(_frisky_address))
    try:
        with _picklable_merge_reduction():
            yield cast(Client, _HijackedClient(dask_client, frisky_client))
    finally:
        frisky_client.close()


def _frisky_cli(*args: str) -> str:
    """Run a ``frisky`` CLI command and return its stdout.

    The CLI rather than the dashboard's REST API, because it is Frisky's documented interface for
    people and agents, and its output is the same thing they will read.
    """
    command = [sys.executable, "-m", "frisky.cli", *args]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    if done.returncode:
        # The CLI says why on stderr, then prints usage hints; CalledProcessError's message drops both.
        lines = done.stderr.strip().splitlines()
        reason = next((line for line in reversed(lines) if "Error" in line), lines[-1] if lines else "no stderr")
        raise RuntimeError(f"frisky {args[0]} {args[1] if len(args) > 1 else ''} exited {done.returncode}: {reason}")
    return done.stdout


def _write(uri: str, text: str) -> None:
    with fsspec.open(uri, "w") as out:
        out.write(text)


def _live_snapshot(
    dashboard_url: str, uri: str, log: logging.Logger | logging.LoggerAdapter[Any], *, spans: bool = True
) -> None:
    """Overwrite ``live/overview.json`` and log the cluster's state as one ``frisky state:`` line.

    Without ``spans``, ``live/cluster.json`` holds the cluster's counts alone. An overview's span
    query has the scheduler assemble every process's buffer before ``--limit`` trims it: on zone
    35N each snapshot stepped the scheduler up by 0.5 to 1.8 GiB. The span drain's queries, filtered
    by name and time, do not.
    """
    started = time.monotonic()
    view = "overview" if spans else "cluster"
    bundle = _frisky_cli("observe", view, dashboard_url, "--json")
    _write(f"{uri}/live/{view}.json", bundle)
    state = json.loads(bundle)["state"] if spans else json.loads(bundle)
    log.info(
        "frisky state: workers=%d idle=%d processing=%d waiting=%d queued=%d memory=%d erred=%d snapshot=%.1fs",
        *(state[k] for k in ("workers_total", "workers_idle", "tasks_processing", "tasks_waiting")),
        *(state[k] for k in ("tasks_queued", "tasks_memory", "tasks_erred")),
        time.monotonic() - started,
    )


class _SpanDrain:
    """Copies the :data:`SPAN_DRAIN_NAMES` spans out of Frisky as the run goes.

    Each drain writes the spans that ended since the previous one to
    ``{uri}/spans/part-NNNNNN.json.gz``, a gzipped JSON list in the format of ``spans.json``, so
    the parts together are the whole run. A span belongs to the drain whose window its end falls
    in, so no span is written twice.
    """

    def __init__(self, dashboard_url: str, uri: str) -> None:
        self.dashboard_url, self.uri = dashboard_url, uri
        self.since_ns: int | None = None  # the first drain takes everything since the hijack
        self.parts = self.spans = 0
        self._lock = threading.Lock()  # the final drain waits for one still in flight

    def drain(self, *, final: bool = False) -> None:
        with self._lock:
            self._drain(final)

    def _drain(self, final: bool) -> None:
        until_ns = time.time_ns() - (0 if final else _SPAN_DRAIN_LAG_NS)
        # One request per name and time slice. The first drain has no start, and covers only the
        # run's first minute.
        starts = [None] if self.since_ns is None else list(range(self.since_ns, until_ns, _SPAN_DRAIN_SLICE_NS))
        spans: list[dict[str, Any]] = []
        reached = self.since_ns
        try:
            for start, end in zip(starts, [*starts[1:], until_ns], strict=True):
                spans += [
                    span
                    for name in SPAN_DRAIN_NAMES
                    for span in frisky.query_spans(
                        name=name, start_ns=start, end_ns=end, limit=sys.maxsize, dashboard_url=self.dashboard_url
                    )
                    if (start is None or span["end_ns"] >= start) and span["end_ns"] < end
                ]
                reached = end
        finally:
            # Keep what was read even when a later slice failed, so a retry starts where this stopped.
            if spans:
                # One-shot dumps at gzip's level 6: json.dump runs the pure-Python encoder, and level
                # 9 compresses 9% smaller in three times the time. Together four times faster.
                with fsspec.open(f"{self.uri}/spans/part-{self.parts:06d}.json.gz", "wb") as out:
                    out.write(gzip.compress(json.dumps(spans).encode(), compresslevel=6))
                self.parts += 1
                self.spans += len(spans)
            self.since_ns = reached


def _repeat(
    stop: threading.Event,
    interval_s: float,
    action: Callable[[], None],
    what: str,
    log: logging.Logger | logging.LoggerAdapter[Any],
) -> None:
    """Run ``action`` every ``interval_s`` until ``stop`` is set, logging rather than raising."""
    while not stop.wait(interval_s):
        try:
            action()
        except Exception as e:
            log.warning("Frisky %s failed: %s", what, e)


def _end_of_run(dashboard_url: str, uri: str, drain: _SpanDrain | None, log: Any) -> list[str]:  # noqa: ANN401
    """Write the end-of-run bundle, one file each, so a failure costs only its own file.

    The scheduler serves ONE span query at a time: the final drain, then ``spans.json``. Both
    overviews are computed from that file in this process rather than asked of the scheduler again,
    since every span query is assembled in the scheduler's memory, and three at once helped kill
    one. The event and log captures are cheap and run alongside. Returns what was written.
    """
    written: list[str] = []

    def attempt(name: str, produce: Callable[[], str]) -> None:
        try:
            _write(f"{uri}/{name}", produce())
            written.append(name)
        except Exception as e:
            log.warning("failed to capture Frisky %s to %s: %s", name, uri, e)

    def events() -> str:
        lifecycle = ("--kind", "worker_added,worker_removed")
        return json.dumps(
            {
                "lifecycle": json.loads(
                    _frisky_cli("observe", "events", dashboard_url, "--json", *lifecycle, "-n", str(EVENTS_LIMIT))
                ),
                "recent": json.loads(
                    _frisky_cli("observe", "events", dashboard_url, "--json", "-n", str(EVENTS_LIMIT))
                ),
            }
        )

    def logs() -> str:
        return _frisky_cli("logs", "--url", dashboard_url, "--json", "--level", "warn", "--limit", "50000")

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="frisky-telemetry") as pool:
        pool.submit(attempt, "events.json", events)
        pool.submit(attempt, "logs.json", logs)
        if drain:
            try:
                drain.drain(final=True)
            except Exception as e:
                log.warning("Frisky span drain failed: %s", e)
            written.append(f"spans/ ({drain.spans} spans in {drain.parts} parts)")
        with tempfile.TemporaryDirectory() as tmp:
            local = f"{tmp}/spans.json"
            limit = SPANS_CAPTURE_LIMIT_DRAINED if drain else SPANS_CAPTURE_LIMIT
            attempt(
                "spans.json",
                lambda: _keep(local, json.dumps(frisky.query_spans(limit=limit, dashboard_url=dashboard_url))),
            )
            if "spans.json" in written:
                attempt("overview.txt", lambda: _frisky_cli("observe", "overview", local))
                attempt("overview.json", lambda: _frisky_cli("observe", "overview", local, "--json"))
    return written


def _keep(path: str, text: str) -> str:
    """Write ``text`` to a local ``path`` too, and pass it on."""
    Path(path).write_text(text)
    return text


@contextlib.contextmanager
def maybe_capture_telemetry(
    dashboard_url: str,
    uri: str | None,
    log: logging.Logger | logging.LoggerAdapter[Any],
    *,
    interval_s: float = LIVE_SNAPSHOT_INTERVAL_S,
    drain_spans: bool = False,
) -> Iterator[None]:
    """Write Frisky's telemetry for this run under the prefix ``uri`` (any fsspec target).

    **While the body runs,** every ``interval_s``: ``live/overview.json`` (``live/cluster.json``,
    the counts alone, with ``drain_spans``) is overwritten and one ``frisky state:`` line is
    logged, so a running, hung or killed run can be read from storage or the run log with no
    port-forward. With ``drain_spans``, every :data:`SPAN_DRAIN_INTERVAL_S` the task, transfer and
    spill spans of the run so far are appended under ``spans/`` as well (:class:`_SpanDrain`), so
    the whole run is kept rather than its tail. **After it:** the bundle
    below, captured all at once before the cluster closes, so any ``worker_removed`` event in it
    is a worker that left mid-run.

    =================  ==========================================================================
    ``spans.json``     the most recent :data:`SPANS_CAPTURE_LIMIT` spans of every kind
                       (:data:`SPANS_CAPTURE_LIMIT_DRAINED` with ``drain_spans``); ``frisky
                       observe overview spans.json`` and the other offline views read it
    ``overview.txt``   ``frisky observe overview`` of ``spans.json``, computed here, not by the scheduler
    ``overview.json``  the same as a structured bundle: perf, costliest spans, outliers
    ``events.json``    ``lifecycle`` (every worker joining and leaving) and ``recent`` (the last
                       :data:`EVENTS_LIMIT` scheduler events)
    ``logs.json``      Frisky's own scheduler and worker warnings and errors. Task logs are not
                       here; they stay in the workers' log streams.
    ``spans/``         with ``drain_spans`` only: the whole run's task, transfer and spill spans,
                       as gzipped parts written while it ran
    =================  ==========================================================================

    Frisky's counterpart to ``providers.aws.dask.maybe_performance_report``, isolated the same way:
    it never raises, so diagnostics cannot fail a run or mask its exception. No-op without a
    ``uri``. ``dashboard_url`` is the Dask dashboard link, which serves Frisky's once hijacked.
    """
    if not uri:
        yield
        return

    stop = threading.Event()
    drain = _SpanDrain(dashboard_url, uri) if drain_spans else None
    # The drain keeps every span, so its live snapshots take none from the scheduler.
    loops = [(interval_s, lambda: _live_snapshot(dashboard_url, uri, log, spans=not drain), "live snapshot")]
    if drain:
        loops.append((SPAN_DRAIN_INTERVAL_S, drain.drain, "span drain"))
    threads = [
        threading.Thread(target=_repeat, args=(stop, *loop, log), name="frisky-telemetry", daemon=True)
        for loop in loops
    ]
    for thread in threads:
        thread.start()
    try:
        yield
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=30)  # a capture in flight must not hold up the end of the run
        written = _end_of_run(dashboard_url, uri, drain, log)
        log.info("wrote Frisky telemetry to %s: %s", uri, ", ".join(written) or "nothing")
