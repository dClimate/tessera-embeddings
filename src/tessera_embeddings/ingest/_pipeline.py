"""Depth-1 prepare/consume pipeline.

Overlaps the PREPARATION of item N+1 with the CONSUMPTION of item N on one
background thread. Exists so a consumer that must run serially (e.g. a writer
holding an exclusive resource) can still hide the preparation cost of the next
item inside its own runtime.

Depth-1 is intrinsic rather than tuned: preparation is a small fraction of
consumption wherever this is worth using, so a deeper buffer would hold
results in memory to hide nothing. An exception raised by ``prepare``
surfaces on the consuming thread at the point its result would have been
used, so failure ordering matches the serial loop's. ``prepare`` must not
mutate shared state; the consumer owns all side effects.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor


def pipelined[T, P](items: Iterable[T], prepare: Callable[[T], P]) -> Iterator[tuple[P, float]]:
    """Yield ``(prepared, stall_seconds)`` per item, preparing one ahead.

    ``stall_seconds`` is how long the consumer waited for the preparation to
    finish — the pipeline's own health metric: near zero when preparation
    hides fully inside consumption, and approaching the preparation cost when
    something starves it.
    """
    it = iter(items)
    try:
        first = next(it)
    except StopIteration:
        return
    # The `with` block is the leak bound: if the consumer raises or breaks
    # mid-loop, generator finalisation exits the executor, which joins the
    # in-flight preparation — one preparation of waiting, never a stray thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(prepare, first)
        for nxt in it:
            waited_from = time.monotonic()
            prepared = pending.result()
            stall = time.monotonic() - waited_from
            pending = pool.submit(prepare, nxt)  # N+1 prepares while the caller consumes N
            yield prepared, stall
        waited_from = time.monotonic()
        yield pending.result(), time.monotonic() - waited_from
