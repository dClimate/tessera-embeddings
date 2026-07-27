"""The depth-1 prepare/consume pipeline (pure, offline).

The pipeline's whole value is that preparation of the next item genuinely runs
DURING consumption of the current one, so these tests prove overlap from
recorded timelines rather than from a mock agreeing with itself: a stubbed
executor would pass every one of them while the production loop ran serially.

The other three properties pinned here are the ones a caller's correctness
rests on — results arrive in the input's order, a failing preparation surfaces
at the point the serial loop would have raised, and abandoning the generator
leaves no thread behind.
"""

from __future__ import annotations

import threading
import time
from typing import NamedTuple

import pytest

from tessera_embeddings.ingest._pipeline import pipelined

PREPARE_S = 0.05
CONSUME_S = 0.15


class _Span(NamedTuple):
    """When one prepare or one consume ran, on the monotonic clock."""

    start: float
    end: float


def _sleeper(seconds: float, spans: list[_Span]):
    """Return a ``prepare`` that sleeps and records its own span."""

    def prepare(item):
        started = time.monotonic()
        time.sleep(seconds)
        spans.append(_Span(started, time.monotonic()))
        return item

    return prepare


def test_prepare_overlaps_consume():
    """Each preparation must START before the previous consumption ENDS.

    The direct observable of overlap, and the only assertion that distinguishes
    this pipeline from a serial loop: under serial execution every preparation
    begins after the previous consumption has returned.
    """
    prepares: list[_Span] = []
    consumes: list[_Span] = []
    results = []

    for prepared, _stall in pipelined(range(4), _sleeper(PREPARE_S, prepares)):
        started = time.monotonic()
        time.sleep(CONSUME_S)
        consumes.append(_Span(started, time.monotonic()))
        results.append(prepared)

    assert results == [0, 1, 2, 3], "the pipeline must not reorder its items"
    assert len(prepares) == 4
    for i in range(1, 4):
        assert prepares[i].start < consumes[i - 1].end, f"preparation {i} did not overlap consumption {i - 1}"


def test_stall_measures_starvation():
    """``stall`` is the health metric: the preparation cost nothing hid.

    Both directions matter. A stall that stayed near zero however slowly
    preparation ran would report a healthy pipeline that was in fact serial;
    a stall that tracked preparation regardless of consumption would report
    starvation on a pipeline that was working.
    """
    starved = [stall for _p, stall in pipelined(range(3), _sleeper(0.2, []))]
    assert all(stall > 0.15 for stall in starved), f"a slow preparation must show as stall: {starved}"

    hidden = []
    for _prepared, stall in pipelined(range(3), _sleeper(0.0, [])):
        time.sleep(0.2)
        hidden.append(stall)
    assert all(stall < 0.05 for stall in hidden), f"an instant preparation must not read as starvation: {hidden}"


def test_prepare_exception_surfaces_in_order():
    """A failure must reach the consumer where the serial loop would have raised.

    The third item's preparation runs while the SECOND is being consumed, so the
    exception is already waiting when the consumer asks for it — it must not
    arrive early (truncating the second item) nor be swallowed.
    """

    def prepare(item):
        if item == 2:
            raise RuntimeError("prepare exploded")
        return item

    gen = pipelined(range(5), prepare)
    assert [next(gen)[0] for _ in range(2)] == [0, 1]
    with pytest.raises(RuntimeError, match="exploded"):
        next(gen)


def test_empty_iterable_yields_nothing_and_never_prepares():
    calls: list[_Span] = []
    assert list(pipelined([], _sleeper(0.0, calls))) == []
    assert calls == [], "an empty input must not start a preparation (nor an executor)"


def test_single_item_stalls_for_its_whole_preparation():
    """One item has nothing to hide behind, so its stall IS its preparation."""
    (prepared, stall), *rest = list(pipelined(["only"], _sleeper(0.1, [])))
    assert rest == []
    assert prepared == "only"
    assert stall > 0.09


def test_consumer_break_joins_background_thread():
    """Abandoning the generator must not leak the preparation thread.

    A consumer that raises or breaks leaves one preparation in flight; the
    executor's context exit joins it, so the cost of abandonment is bounded by
    a single preparation rather than by a thread that outlives the run.
    """
    baseline = threading.active_count()
    workers: list[threading.Thread] = []

    def prepare(item):
        workers.append(threading.current_thread())
        time.sleep(0.2)
        return item

    gen = pipelined(range(4), prepare)
    for prepared, _stall in gen:
        assert prepared == 0
        break
    gen.close()

    assert workers and not workers[0].is_alive(), "the preparation thread outlived the generator"
    assert threading.active_count() == baseline


# ── depth > 1: the look-ahead a BATCHING consumer needs ──


def test_depth_buffers_that_many_ahead_not_more():
    """With depth=k, exactly k preparations are in flight while the first is consumed.

    The bound matters in both directions: too few and a batching consumer hides only one
    item's preparation per batch; too many and prepared results pile up in memory.
    """
    started = threading.Semaphore(0)
    release = threading.Event()

    def prepare(i: int) -> int:
        started.release()
        release.wait(5)
        return i

    gen = pipelined(range(10), prepare, depth=3)
    # Pull one; its own preparation must complete, and the look-ahead must be primed.
    release.set()
    first, _ = next(gen)
    assert first == 0
    # 3 primed + 1 refilled after the first yield = at most 4 ever submitted so far.
    counted = sum(1 for _ in range(4) if started.acquire(timeout=1))
    assert counted == 4, f"expected 4 submissions with depth=3, saw {counted}"
    assert not started.acquire(timeout=0.2), "depth bound exceeded"
    gen.close()


def test_depth_one_is_unchanged_and_default():
    """depth=1 must stay the historical one-ahead behaviour: first item unhidden."""
    order: list[str] = []

    def prepare(i: int) -> int:
        order.append(f"prep{i}")
        return i

    got = []
    for prepared, _stall in pipelined(range(3), prepare):
        order.append(f"consume{prepared}")
        got.append(prepared)
    assert got == [0, 1, 2]
    # prep0 before consume0, and prep1 issued before consume0 returns.
    assert order[0] == "prep0"
    assert order.index("prep1") < order.index("consume1")


def test_depth_preserves_order_and_single_threading():
    """Deeper buffering must not reorder items nor prepare two at once."""
    concurrent = []
    live = 0
    lock = threading.Lock()

    def prepare(i: int) -> int:
        nonlocal live
        with lock:
            live += 1
            concurrent.append(live)
        time.sleep(0.01)
        with lock:
            live -= 1
        return i

    got = [p for p, _ in pipelined(range(8), prepare, depth=4)]
    assert got == list(range(8)), "depth must not reorder"
    assert max(concurrent) == 1, f"preparation must stay single-threaded, saw {max(concurrent)}"


def test_depth_rejects_zero():
    with pytest.raises(ValueError, match="depth must be >= 1"):
        next(pipelined(range(3), lambda i: i, depth=0))
