"""Prepare/consume pipeline with a bounded look-ahead.

Overlaps the PREPARATION of later items with the CONSUMPTION of the current one on one
background thread. Exists so a consumer that must run serially (e.g. a writer holding an
exclusive resource) can still hide the preparation cost of what comes next inside its own
runtime.

``depth`` is how many prepared results may be buffered ahead, and it exists because the
consumer's unit is not always one item. Depth 1 is right when the consumer consumes items
one at a time: preparation is a small fraction of consumption, so a deeper buffer would hold
results in memory to hide nothing. When the consumer instead BATCHES k items before acting
(the S2 date loop under ``batch_dates``), depth 1 hides only one item's preparation per
batch — the look-ahead has to reach k for the next batch to be ready when the current one
finishes.

Preparation stays SINGLE-THREADED at any depth: one worker, items prepared in order. Depth
buys buffering, never concurrency, so ``prepare`` still sees the same one-at-a-time
world and the side-effect-free contract below is unchanged.

An exception raised by ``prepare`` surfaces on the consuming thread at the point its result
would have been used, so failure ordering matches the serial loop's — with depth > 1 that
means later items may already have been prepared when an earlier one raises, which is
harmless precisely because preparation has no side effects. ``prepare`` must not mutate
shared state; the consumer owns all side effects.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor


def pipelined[T, P](items: Iterable[T], prepare: Callable[[T], P], *, depth: int = 1) -> Iterator[tuple[P, float]]:
    """Yield ``(prepared, stall_seconds)`` per item, preparing up to ``depth`` ahead.

    ``stall_seconds`` is how long the consumer waited for that item's preparation to
    finish — the pipeline's own health metric: near zero when preparation hides fully
    inside consumption, and approaching the preparation cost when something starves it.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    it = iter(items)
    # A sentinel rather than None: an item may legitimately BE None, and exhausting on
    # it would silently truncate the run.
    done = object()
    # Exited WITHOUT joining on the abnormal path. A `with` block calls
    # `shutdown(wait=True)`, so if the consumer raises or breaks mid-loop, generator
    # finalisation blocks until the in-flight preparation returns — and preparation is a
    # read against the very Dask cluster the failing flow is trying to tear down. A read
    # that has stalled there holds the unwind for as long as it stalls, with the whole
    # billed fleet still up, which is the opposite of what the failure path needs.
    #
    # `cancel_futures=True` drops the ones not yet started; the one already running cannot
    # be interrupted (nothing in Python can interrupt it), so what changes is that cleanup
    # no longer WAITS for it. The thread is still non-daemon and still joined at
    # interpreter exit, so this abandons no work silently — it only stops the teardown
    # queueing behind a read that may never finish.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        # Exactly `depth` in flight, so depth=1 keeps the original one-ahead behaviour:
        # the FIRST item's preparation is never hidden (nothing precedes it to hide
        # behind), and every later one overlaps the consumption of its predecessor.
        pending: deque[Future[P]] = deque()
        for _ in range(depth):
            nxt = next(it, done)
            if nxt is done:
                break
            pending.append(pool.submit(prepare, nxt))  # type: ignore[arg-type]
        while pending:
            waited_from = time.monotonic()
            prepared = pending.popleft().result()
            stall = time.monotonic() - waited_from
            # Refill BEFORE yielding, so the background thread has work for the whole of
            # the consumer's runtime rather than waiting to be asked once it returns.
            nxt = next(it, done)
            if nxt is not done:
                pending.append(pool.submit(prepare, nxt))  # type: ignore[arg-type]
            yield prepared, stall
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
