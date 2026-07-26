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
