"""Orchestrator-agnostic concurrency primitives.

Two tools, neither importing Prefect:

* :func:`sliding_window_submit` — bounded fan-out over any Future-like submitter
  (Prefect task, thread pool, plain runner).
* :class:`DispatchThrottle` — bounded-concurrency + rate-paced launcher for a burst
  of async dispatches against a rate-limited backend (e.g. many deployment runs).

The reference repo's flow utilities ship a Prefect-only
``sliding_window_submit`` that calls ``task_fn.submit(**kwargs)``. We
generalise here: callers pass a ``submit_fn`` callable that returns
any Future-like object exposing ``.add_done_callback``. Both
:class:`prefect.futures.PrefectFuture` and
:class:`concurrent.futures.Future` satisfy this Protocol, so the same
function works in Prefect flows, in the plain runner, and in unit
tests that fake out the executor.

Usage in a Prefect flow::

    from tessera_embeddings.orchestration.concurrency import sliding_window_submit

    completed = sliding_window_submit(
        submit_fn=lambda kwargs: my_task.submit(**kwargs),
        jobs=[(roi_name, {"path": "..."}) for roi_name in rois],
        max_concurrent=10,
    )

Usage with a thread pool::

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as pool:
        completed = sliding_window_submit(
            submit_fn=lambda kwargs: pool.submit(do_work, **kwargs),
            jobs=jobs,
            max_concurrent=10,
        )

This module imports nothing from Prefect.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Protocol, TypeVar


class _Future(Protocol):
    """Minimal Future-like interface used by :func:`sliding_window_submit`.

    Both :class:`concurrent.futures.Future` and
    :class:`prefect.futures.PrefectFuture` satisfy this Protocol; we
    only require ``add_done_callback`` for completion notification.
    """

    def add_done_callback(self, fn: Callable[[Any], None], /) -> None:
        """Register a callback invoked when the future completes."""
        ...


def sliding_window_submit[K](
    submit_fn: Callable[[dict[str, Any]], _Future],
    jobs: Iterable[tuple[K, dict[str, Any]]],
    max_concurrent: int = 10,
) -> list[tuple[K, _Future]]:
    """Submit work in a sliding window of ``max_concurrent`` at a time.

    Fills an initial window of ``max_concurrent`` submissions, then
    submits a new job each time an existing one completes. Maintains
    up to ``max_concurrent`` in-flight tasks until ``jobs`` is
    exhausted.

    Args:
        submit_fn: Callable that submits one job. Receives a kwargs
            dict and returns a Future-like object whose
            ``add_done_callback`` will be called when the job
            finishes. The shape is ``submit_fn(kwargs)``, not
            ``submit_fn(*args, **kwargs)`` — pass a closure if your
            real submitter has a different signature.
        jobs: Iterable of ``(key, kwargs)`` pairs. ``key`` is opaque
            and is returned alongside the resulting future for caller
            bookkeeping.
        max_concurrent: Maximum number of in-flight submissions.

    Returns:
        List of ``(key, completed_future)`` pairs in completion order.
        Callers that need exception propagation should call
        ``future.result()`` on each returned future.
    """
    job_iter = iter(jobs)
    active: dict[_Future, K] = {}
    completed: list[tuple[K, _Future]] = []

    finished_futures: list[_Future] = []
    finished_event = threading.Event()
    finished_lock = threading.Lock()

    def _on_done(future: _Future) -> None:
        with finished_lock:
            finished_futures.append(future)
            finished_event.set()

    # Fill initial window
    for key, kwargs in itertools.islice(job_iter, max_concurrent):
        future = submit_fn(kwargs)
        active[future] = key
        future.add_done_callback(_on_done)

    while active:
        finished_event.wait()
        with finished_lock:
            done = finished_futures[:]
            finished_futures.clear()
            finished_event.clear()

        for future in done:
            key = active.pop(future)
            completed.append((key, future))

        # Refill slots as completions free them up
        for key, kwargs in itertools.islice(job_iter, max_concurrent - len(active)):
            future = submit_fn(kwargs)
            active[future] = key
            future.add_done_callback(_on_done)

    return completed


# --------------------------------------------------------------------------- #
# Rate-limited async dispatch throttle
# --------------------------------------------------------------------------- #

# Minimum seconds between two consecutive dispatch launches. Bounds the launch
# rate to ≈ 1 / this — sized comfortably under a backend's sustained API rate so a
# wide fan-out never trips it (e.g. ECS RegisterTaskDefinition throttling when a
# flow fans out many deployment runs at once).
DISPATCH_MIN_INTERVAL_SEC = 1.0

# Upper bound (seconds) on the extra uniform random jitter added to each launch
# interval, so launches are not perfectly periodic (which could resonate with
# another actor dispatching on the same beat). Launches end up spaced
# DISPATCH_MIN_INTERVAL_SEC .. DISPATCH_MIN_INTERVAL_SEC + DISPATCH_JITTER_SEC apart.
DISPATCH_JITTER_SEC = 2.0

_T = TypeVar("_T")


class DispatchThrottle:
    """Bound concurrency and pace the launch rate of many async dispatches.

    For fan-outs that fire a burst of asynchronous calls (e.g. dispatching many
    deployment runs via a Prefect ``arun_deployment``) against a backend with an
    API rate limit. Firing in lockstep trips the limit; this owns the mitigation in
    one place so every fan-out gets it rather than re-implementing per-call jitter.
    It:

    * caps how many dispatches are in flight at once (a semaphore), and
    * **paces the launch moments** — it serialises the instants dispatches are
      released so consecutive launches are at least :data:`DISPATCH_MIN_INTERVAL_SEC`
      apart, plus uniform random jitter up to :data:`DISPATCH_JITTER_SEC`. The
      guaranteed minimum interval bounds the call rate deterministically — unlike a
      bare per-dispatch random sleep, it does not cluster when many semaphore slots
      free simultaneously.

    This is deterministic pacing, **not** retry machinery — no backoff loop, no
    catch of the throttle exception. Pure asyncio (no Prefect, no boto3): callers
    pass a zero-arg coroutine factory that performs the actual dispatch, so the
    dispatch surface stays patchable under test. Construct one per fan-out; it is
    single-event-loop and lives for the duration of one run.
    """

    def __init__(self, max_concurrency: int) -> None:
        self._sem = asyncio.Semaphore(max_concurrency)
        # Serialises the pacing computation+sleep so launches are released one at a
        # time, each at least an interval after the previous.
        self._gate = asyncio.Lock()
        # Monotonic instant the next launch is allowed. 0.0 ⇒ the first launch is
        # immediate (it never waits).
        self._next_at = 0.0

    async def run(self, launch: Callable[[], Awaitable[_T]]) -> _T:
        """Acquire a concurrency slot, pace the launch, then await ``launch()``.

        ``launch`` is a zero-arg callable returning the dispatch awaitable (e.g.
        ``lambda: arun_deployment(ref, parameters=params)``). The dispatches still
        run concurrently up to the semaphore; the throttle only staggers the
        *moments* they start.
        """
        async with self._sem:
            await self._pace()
            return await launch()

    async def _pace(self) -> None:
        """Block until this dispatch's paced launch slot, holding the gate so
        siblings queue behind it and inherit the next slot.
        """
        async with self._gate:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now += wait
            self._next_at = now + DISPATCH_MIN_INTERVAL_SEC + random.uniform(0, DISPATCH_JITTER_SEC)
