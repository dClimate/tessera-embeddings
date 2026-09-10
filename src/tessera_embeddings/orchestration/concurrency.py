"""Sliding-window concurrent submission, orchestrator-agnostic.

Callers pass a ``submit_fn`` returning any Future-like object with ``.add_done_callback``,
which both :class:`prefect.futures.PrefectFuture` and :class:`concurrent.futures.Future`
satisfy, so one implementation covers Prefect flows, the plain runner, and unit tests that
fake the executor. This module imports nothing from Prefect
(context_docs/decisions/001-thin-prefect-wrapping.md).

::

    # In a Prefect flow:
    completed = sliding_window_submit(
        submit_fn=lambda kwargs: my_task.submit(**kwargs),
        jobs=[(roi_name, {"path": "..."}) for roi_name in rois],
        max_concurrent=10,
    )

    # With a thread pool:
    with ThreadPoolExecutor(max_workers=10) as pool:
        completed = sliding_window_submit(lambda kwargs: pool.submit(do_work, **kwargs), jobs)
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterable
from typing import Any, Protocol


class _Future(Protocol):
    """Minimal Future-like interface used by :func:`sliding_window_submit`."""

    def add_done_callback(self, fn: Callable[[Any], None], /) -> None:
        """Register a callback invoked when the future completes."""
        ...


def sliding_window_submit[K](
    submit_fn: Callable[[dict[str, Any]], _Future],
    jobs: Iterable[tuple[K, dict[str, Any]]],
    max_concurrent: int = 10,
) -> list[tuple[K, _Future]]:
    """Submit work in a sliding window of ``max_concurrent`` at a time.

    Args:
        submit_fn: Submits one job. The shape is ``submit_fn(kwargs)``, not
            ``submit_fn(*args, **kwargs)`` — pass a closure if your submitter differs.
        jobs: Iterable of ``(key, kwargs)`` pairs; ``key`` is opaque caller bookkeeping.
        max_concurrent: Maximum number of in-flight submissions.

    Returns:
        ``(key, completed_future)`` pairs in completion order. Exceptions are not raised
        here — call ``future.result()`` on each to propagate them.
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
