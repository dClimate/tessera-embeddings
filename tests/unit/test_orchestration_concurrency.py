"""Unit tests for ``orchestration.concurrency.sliding_window_submit``.

These tests use ``concurrent.futures.ThreadPoolExecutor`` as the submit
backend so we exercise the real Future protocol without a Prefect
runtime. We verify three behaviours:

1. The max-concurrent invariant is honoured.
2. Every job appears exactly once in the returned list.
3. Exceptions raised inside jobs surface via ``future.result()``.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from tessera_embeddings.orchestration.concurrency import sliding_window_submit


def test_returns_one_pair_per_job() -> None:
    """Every job key is returned exactly once with a completed future."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [(i, {"x": i}) for i in range(20)]
        completed = sliding_window_submit(
            submit_fn=lambda kw: pool.submit(lambda x: x * 2, **kw),
            jobs=jobs,
            max_concurrent=4,
        )

    keys = [k for k, _ in completed]
    assert sorted(keys) == list(range(20))
    assert all(isinstance(f, Future) for _, f in completed)
    assert all(f.done() for _, f in completed)


def test_respects_max_concurrent_invariant() -> None:
    """The number of in-flight submissions never exceeds ``max_concurrent``."""
    max_concurrent = 3
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def slow_job(_: int) -> int:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return 0

    with ThreadPoolExecutor(max_workers=10) as pool:
        # Pool capacity (10) is intentionally larger than max_concurrent so the
        # only thing limiting concurrency is sliding_window_submit itself.
        jobs = [(i, {"_": i}) for i in range(20)]
        sliding_window_submit(
            submit_fn=lambda kw: pool.submit(slow_job, **kw),
            jobs=jobs,
            max_concurrent=max_concurrent,
        )

    assert peak <= max_concurrent, f"peak in-flight {peak} exceeded cap {max_concurrent}"


def test_propagates_exceptions_via_result() -> None:
    """Failed jobs are returned with a future whose ``.result()`` raises."""
    def boom(should_fail: bool) -> str:
        if should_fail:
            raise RuntimeError("kapow")
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs: list[tuple[int, dict]] = [
            (0, {"should_fail": False}),
            (1, {"should_fail": True}),
            (2, {"should_fail": False}),
        ]
        completed = sliding_window_submit(
            submit_fn=lambda kw: pool.submit(boom, **kw),
            jobs=jobs,
            max_concurrent=2,
        )

    by_key = dict(completed)
    assert by_key[0].result() == "ok"
    assert by_key[2].result() == "ok"
    with pytest.raises(RuntimeError, match="kapow"):
        by_key[1].result()


def test_handles_empty_input() -> None:
    """An empty job iterable returns an empty list without blocking."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = sliding_window_submit(
            submit_fn=lambda kw: pool.submit(lambda: None),
            jobs=[],
            max_concurrent=4,
        )
    assert completed == []
