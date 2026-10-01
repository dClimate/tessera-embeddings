"""Frisky: an experimental Rust scheduler loaded onto the Dask cluster a provider built.

`Frisky <https://getfrisky.dev/>`_ reimplements Dask's scheduler and workers in Rust, with far
more telemetry. It is pre-1.0 and closed-source, so this module is the only place the pipeline
calls its API: a change in the library lands here. Opt-in per run (``use_frisky`` on the ingest
flows, ``frisky:`` in the plain runner's config); with it off nothing here runs. Background,
measurements and the do's and don'ts: ``context_docs/ingest/frisky-experiment.md``.

The providers build the same Dask cluster as always and :func:`hijack` loads Frisky onto it: a
Frisky scheduler inside the Dask scheduler process and a Frisky worker inside every Dask worker
process, late joiners included. Dask keeps everything Frisky does not replace — provisioning and
teardown, the Prefect task runner (the ingest task still runs as a Dask task on one worker), and
the two calls the ingest makes that Frisky's client lacks, ``register_plugin`` and ``run``. Those
still reach Frisky's tasks because they run in the same processes. :func:`connect` hands the
ingest a client that sends compute to Frisky and everything else to Dask.

Three behaviours differ from Dask and are handled here or by the providers:

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
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import json
import logging
import pickle
import threading
from collections.abc import Iterator
from typing import Any, cast

import frisky
import fsspec
import tblib.pickling_support
from distributed import Client, WorkerPlugin

#: The client methods Frisky implements and the ingest calls. Every other attribute of the client
#: :func:`connect` yields is Dask's; when Frisky's client grows one, adding it here is the change.
FRISKY_METHODS = frozenset({"compute", "persist", "submit", "map", "gather", "scatter"})

#: Most recent spans :func:`maybe_capture_spans` keeps. A span is a dict of about a kilobyte on the
#: flow runner, so this bounds the capture near 500 MB; a longer run keeps its tail, like Dask's
#: capped task stream.
SPANS_CAPTURE_LIMIT = 500_000


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
        yield cast(Client, _HijackedClient(dask_client, frisky_client))
    finally:
        frisky_client.close()


@contextlib.contextmanager
def maybe_capture_spans(
    dashboard_url: str,
    uri: str | None,
    log: logging.Logger | logging.LoggerAdapter[Any],
) -> Iterator[None]:
    """Upload Frisky's spans to ``uri`` (any fsspec target) after the body, for offline analysis.

    Frisky's counterpart to ``providers.aws.dask.maybe_performance_report``, isolated the same way:
    it never raises, so diagnostics cannot fail a run or mask its exception. No-op without a
    ``uri``. The file is what ``frisky observe spans`` writes, so ``frisky observe overview
    spans.json`` reads it once the cluster is gone. ``dashboard_url`` is the Dask dashboard link,
    which serves Frisky's dashboard and API once hijacked.
    """
    try:
        yield
    finally:
        if uri:
            try:
                spans = frisky.query_spans(limit=SPANS_CAPTURE_LIMIT, dashboard_url=dashboard_url)
                with fsspec.open(uri, "w") as out:
                    json.dump(spans, out)
                capped = " (capped: the run's tail only)" if len(spans) >= SPANS_CAPTURE_LIMIT else ""
                log.info("wrote %d Frisky spans to %s%s", len(spans), uri, capped)
            except Exception as e:
                log.warning("failed to capture Frisky spans to %s: %s", uri, e)
