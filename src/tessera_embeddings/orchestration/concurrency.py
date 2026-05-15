"""Sliding-window concurrent submission, orchestrator-agnostic.

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

import itertools
import threading
from collections.abc import Callable, Iterable
from typing import Any, Protocol


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
