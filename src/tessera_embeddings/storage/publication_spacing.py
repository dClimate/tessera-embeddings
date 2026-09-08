"""Keep the fleet's publications far enough apart that no catch-up ever has to cross two.

**The invariant this module holds:** two publications never land inside one catch-up interval,
so a coordinator that catches up on every tick is never more than one publication (two
snapshots) behind the branch tip when it rebases.

**Why.** icechunk's ``rebase`` has hung, without exception, every time a session was four or more
snapshots behind the tip — seven of nine commits on 2026-08-29, two of two catch-ups on 08-31,
and five fills in the shard-write tail on 09-04 — and never once at zero or two. A cell publishes
TWO snapshots (the fill and the completion mark), so "four behind" is "two publications landed
since I last caught up", and the periodic catch-up (:mod:`session_catch_up`) can only guarantee
that does not happen if publications are spaced further apart than its interval. Bounding the
hang's cost is the job of the fork-phase watchdog and the drain ceiling; REMOVING its
precondition is the job of this module. ``context_docs/storage/writing-to-the-global-store.md``
§3 and §5 carry the measurements and the history of deferring this.

**How.** Every campaign publication passes through :func:`publication`, a context manager
that takes a fleet-wide slot — a Prefect global concurrency limit of ONE, held through a
:class:`~tessera_embeddings.orchestration.prefect.flows._fleet_gate.FleetGate` — around the
cell's commits, and then keeps holding it for a further :data:`PUBLICATION_SPACING_S` after the
last of them. Consecutive publications are therefore at least that far apart — as the store's own
snapshot timestamps record them — fleet-wide, by construction: no shared clock, no coordination
beyond the one slot.

**This is a SPACING MUTEX, not a committer cap, and the distinction is the whole reason it can
exist.** TE #151 removed a fleet-wide commit gate on cost: what it bounded was a slowdown measured
in seconds, and its ``active_slots`` was twice misread as a progress signal. Neither argument
carries here. The cost of spacing is bounded and small — a publication takes a second or two and
then idles the slot for the rest of the interval, so even ten coordinators finishing at once
delay the last by minutes against assemblies measured in hours — and it buys the removal of a
failure that cost days. And this slot's occupancy is ``0`` or ``1`` and means nothing about
progress; anyone reading it as one has been warned here.

**Prefect-free, deliberately.** This module holds a callable that PRODUCES the gate; the flow that
owns the process installs one (:func:`install_publication_gate`) and the storage layer never
imports Prefect. With nothing installed — a single-ROI fill on its own store, a test, the plain
runner — :func:`publication` is a no-op, which is correct: spacing only matters when several
writers share one branch tip.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from tessera_embeddings.storage.session_catch_up import CATCH_UP_INTERVAL_S

_log = logging.getLogger(__name__)

#: Minimum time between consecutive publications, fleet-wide. DERIVED from the catch-up interval,
#: never written as its own literal, so the two cannot drift apart: the guarantee is that at most
#: one publication lands between two ticks, which needs spacing > interval by enough margin for a
#: tick's own rebase (measured 0.5-2.8 s at the depths it now sees). Three intervals gives a
#: ten-second margin at the current five-second tick; the earlier design note worried about a
#: one-second margin, and this is the answer to it.
PUBLICATION_SPACING_S = 3 * CATCH_UP_INTERVAL_S

GateFactory = Callable[[], AbstractContextManager[Any]]

_installed: GateFactory | None = None
_install_lock = threading.Lock()


def install_publication_gate(factory: GateFactory | None) -> GateFactory | None:
    """Install (or, with ``None``, remove) the process-wide publication gate. Returns the previous one.

    Called by the flow that owns the process, once, before any cell publishes, and reversed in
    its teardown. Process-global because a publication is a process-global fact — the trailing
    assembly thread and the feeder thread (which marks terminal cells) both publish, and neither
    can be handed a gate through the Prefect-free runner without threading a parameter through
    every layer #151 took it out of. One owner, one setter, like the icechunk log filter.
    """
    global _installed
    with _install_lock:
        previous, _installed = _installed, factory
    return previous


def installed_publication_gate() -> GateFactory | None:
    """The factory in force, for the flow's own assertions and for tests."""
    return _installed


@contextmanager
def publication(
    *,
    spacing_s: float | None = None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> Iterator[None]:
    """Hold the fleet's publication slot around a cell's commits, then for the rest of the spacing.

    Wrap a WHOLE publication — both commits of a filled cell, or the one commit of a terminal
    mark — so the two snapshots land together and the next writer's first snapshot lands no
    sooner than ``spacing_s`` after this one's last. On an exception the slot is released at
    once: nothing was published, so there is nothing to space.

    No gate installed means no-op, by design (see the module docstring).
    """
    factory = _installed
    if factory is None:
        yield
        return
    wait_s = PUBLICATION_SPACING_S if spacing_s is None else spacing_s
    logger = log or _log
    with factory():
        yield
        # A FULL spacing after the last commit, not the remainder of one measured from the
        # acquire: measured from the acquire, a slow commit eats into the gap the next writer
        # sees — the 2026-09-08 dev run recorded 12.1 s between publications against a 15 s
        # spacing when one commit took 3 s. Counting from the last snapshot makes the gap the
        # STORE observes at least the spacing, exactly, whatever the commits cost.
        logger.debug("Publication done; holding the fleet's slot %.1fs to space the next one", wait_s)
        time.sleep(wait_s)
