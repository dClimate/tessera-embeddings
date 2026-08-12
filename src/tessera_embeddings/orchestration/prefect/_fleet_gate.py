"""Fleet-wide concurrency gates that HOLD when their limit is lowered below what they need.

The campaign is throttled by two named Prefect global concurrency limits — one bounding how
many zone ingests run at once, one bounding simultaneous commits. Both are acquired through
:class:`FleetGate`.

**Why this exists rather than a bare ``concurrency()`` call.** Prefect's server refuses a
request for more slots than a limit holds with ``422``, which is a client error the acquisition
service does not retry: the acquirer raises immediately. So *lowering a gate to zero* — the
obvious way to hold work back — failed the next thing to reach it instead of making it wait,
and there was no way to stop a campaign taking on new cells short of cancelling runs. This
class turns that state into what an operator means by it:

* **limit at or above what the acquirer needs, slots free** — proceeds, as before.
* **limit satisfiable but currently full** — Prefect answers ``423`` and already waits and
  retries. Untouched; that is ordinary queueing.
* **limit lowered below what the acquirer needs (in practice, zero)** — HOLD. Log it and keep
  asking until the limit rises or the run is cancelled. This is the pause lever.
* **limit does not exist** — still fails immediately, loudly, as it must: an absent gate is a
  misconfiguration and running ungated is the thing the gate exists to prevent.

The four cases are distinguished by what the server said, not by reading the limit ourselves:
a limit read before acquiring is a different fact from the one the acquisition acted on.

**A hold is not free and does not stop a running fleet.** A cluster holds its GPU fleet across
its whole multi-cell walk, so holding the ingest gate stops new cells being taken up while the
cell already in flight finishes — after which the fleet idles at full width and full cost.
Holding is "stop taking on work", not "stop the meter"; only cancelling does that.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any

from prefect.concurrency.sync import concurrency
from prefect.exceptions import PrefectHTTPStatusError

if TYPE_CHECKING:
    import logging

#: How long to wait between attempts while a gate is holding. Tens of seconds, not seconds:
#: the thing being waited for is a human raising a limit, and the cells behind the gate run
#: for hours, so a tighter poll only adds orchestrator load to a paused campaign.
HOLD_POLL_S = 30.0
#: How often a holding gate repeats itself in the log. A hold is indefinite by design, so
#: silence would be indistinguishable from a hung run — and the elapsed time is the number an
#: operator wants when deciding whether the pause was forgotten.
HOLD_LOG_EVERY_S = 300.0

#: The server's message when a request asks for more slots than the limit holds. Matched only
#: as a belt-and-braces companion to the 422 status; the status is the signal.
_TOO_SMALL = "greater than the limit"


def gate_is_holding(exc: BaseException) -> bool:
    """Did this acquisition failure mean "the limit is currently too small", or something else?

    ``422`` is the server's answer to a request for more slots than the limit holds — which,
    with every caller here occupying one slot, means the limit is zero. Anything else (a
    missing limit under ``strict``, a network failure, an unauthorised client) is a real
    failure and must propagate: treating those as a hold would park a run forever on a
    misconfiguration.

    The status has to be dug out of the cause chain because Prefect wraps the HTTP error in
    ``ConcurrencySlotAcquisitionError``, whose own message and type say nothing about which of
    the two happened.
    """
    seen: set[int] = set()
    cause: BaseException | None = exc
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        response = getattr(cause, "response", None)
        status = getattr(response, "status_code", None)
        if status == 422 and (isinstance(cause, PrefectHTTPStatusError) or _TOO_SMALL in str(cause)):
            return True
        cause = cause.__cause__
    return False


class FleetGate(AbstractContextManager):
    """One slot of a named Prefect global concurrency limit, held for the ``with`` body.

    THREAD-SAFE: the active context lives in a per-thread stack rather than an instance slot.
    One gate object is shared between a chained fill's feeder thread (terminal plans commit
    inside ``plan``) and its trailing-assembly thread, and an instance slot would let a
    concurrent enter overwrite the other thread's context and release the wrong slot on exit.

    ``strict=True`` on every acquisition: an absent or misspelled limit must fail closed.
    Prefect defaults to warning and proceeding UNGATED, which would silently reintroduce
    exactly the contention these gates bound.
    """

    def __init__(
        self,
        name: str,
        *,
        log: logging.Logger | logging.LoggerAdapter | None = None,
        occupy: int = 1,
        poll_s: float = HOLD_POLL_S,
        should_stop: Callable[[], bool] | None = None,
        **concurrency_kwargs: float | bool | None,
    ) -> None:
        """Args:
        name: The global concurrency limit to acquire from.
        log: Where a hold announces itself. A hold with no log line is a hung run to
            anyone reading the run's output.
        occupy: Slots per acquisition. One everywhere today, which is what makes "limit
            below what we need" and "limit zero" the same state.
        poll_s: Seconds between attempts while holding.
        should_stop: Consulted while holding, so a runner that is winding down abandons
            the wait instead of parking past its own shutdown. Its exception is the
            original acquisition error, which is the honest one for the caller.
        concurrency_kwargs: Passed through to :func:`concurrency` — lease duration and
            lease-renewal policy differ between the two gates.
        """
        self._name = name
        self._log = log
        self._occupy = occupy
        self._poll_s = poll_s
        self._should_stop = should_stop
        self._kwargs = concurrency_kwargs
        self._local = threading.local()

    def _acquire(self) -> AbstractContextManager[Any]:
        """Enter one ``concurrency`` context, holding while the gate's limit is too small."""
        held_for = 0.0
        announced_at: float | None = None
        while True:
            cm = concurrency(self._name, occupy=self._occupy, strict=True, **self._kwargs)
            try:
                cm.__enter__()
            # Caught broadly and re-raised unless the cause chain says "limit too small":
            # Prefect's own ConcurrencySlotAcquisitionError lives in a private module, and the
            # decision here does not need its type — it needs what the server answered.
            except Exception as exc:
                if not gate_is_holding(exc):
                    raise
                if self._should_stop is not None and self._should_stop():
                    raise
                if announced_at is None or held_for - announced_at >= HOLD_LOG_EVERY_S:
                    if self._log is not None:
                        self._log.warning(
                            "Gate %r is at zero — HELD, not failed: waiting for the limit to be "
                            "raised (%.0f min so far). Nothing is lost while held; raise the limit "
                            "with `prefect global-concurrency-limit update %s --limit N` to resume.",
                            self._name,
                            held_for / 60.0,
                            self._name,
                        )
                    announced_at = held_for
                time.sleep(self._poll_s)
                held_for += self._poll_s
                continue
            if held_for and self._log is not None:
                self._log.info("Gate %r released after a %.0f min hold — proceeding", self._name, held_for / 60.0)
            return cm

    def __enter__(self) -> None:
        cm = self._acquire()
        stack: list[AbstractContextManager[Any]] = getattr(self._local, "stack", [])
        stack.append(cm)
        self._local.stack = stack

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        cm = self._local.stack.pop()
        cm.__exit__(exc_type, exc, tb)
