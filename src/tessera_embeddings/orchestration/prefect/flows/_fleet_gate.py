"""Fleet-wide concurrency gates that HOLD when their limit is lowered below what they need.

The campaign is throttled by two named Prefect global concurrency limits — one bounding how many
zone ingests run at once, one bounding simultaneous commits. Both are acquired through
:class:`FleetGate`.

**Why this exists rather than a bare ``concurrency()`` call.** Prefect's server refuses a request
for more slots than a limit holds with ``422``, a client error the acquisition service does not
retry, so the acquirer raises immediately: *lowering a gate to zero* — the obvious way to hold
work back — fails the next thing to reach it instead of making it wait, leaving no way to stop a
campaign taking on new cells short of cancelling runs. This class turns that state into what an
operator means by it:

* **limit at or above what the acquirer needs, slots free** — proceeds.
* **limit satisfiable but currently full** — Prefect answers ``423`` and already waits and
  retries. Untouched; that is ordinary queueing.
* **limit lowered below what the acquirer needs (in practice, zero)** — HOLD. Log it and keep
  asking until the limit rises or the run is cancelled. This is the pause lever.
* **limit does not exist** — still fails immediately, loudly, as it must: an absent gate is a
  misconfiguration and running ungated is the thing the gate exists to prevent.

The four cases are distinguished by what the server said, not by reading the limit ourselves: a
limit read before acquiring is a different fact from the one the acquisition acted on.

**A hold is not free and does not stop a running fleet.** A cluster holds its GPU fleet across its
whole multi-cell walk, so holding the ingest gate stops new cells being taken up while the cell
already in flight finishes — after which the fleet idles at full width and full cost. Holding is
"stop taking on work", not "stop the meter"; only cancelling does that.
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

#: How long to wait between attempts while a gate is holding. Tens of seconds, not seconds: what
#: is being waited for is a human raising a limit, and the cells behind the gate run for hours, so
#: a tighter poll only adds orchestrator load to a paused campaign.
HOLD_POLL_S = 30.0
#: How often a holding gate repeats itself in the log. A hold is indefinite by design, so silence
#: would be indistinguishable from a hung run — and the elapsed time is the number an operator
#: wants when deciding whether the pause was forgotten.
HOLD_LOG_EVERY_S = 300.0

#: The server's message when a request asks for more slots than the limit holds. Matched only as a
#: belt-and-braces companion to the 422 status; the status is the signal.
_TOO_SMALL = "greater than the limit"


def _is_request_validation_error(response: Any) -> bool:  # noqa: ANN401 — httpx.Response, kept loose for fakes
    """Is this ``422`` FastAPI refusing the REQUEST, rather than the server refusing the ask?

    Both arrive as ``422`` from the same endpoint, and only one of them is a hold.

    * The hold is raised by the concurrency router itself as
      ``HTTPException(422, detail="Slots requested is greater than the limit")``, which FastAPI
      serialises with ``detail`` as a plain **string**.
    * A parameter-validation failure — ``slots`` at or below zero, a ``lease_duration`` outside
      the server's ``ge=60, le=86400`` bounds — never reaches that code at all. FastAPI rejects
      it first, with ``detail`` as a **list** of per-field objects each carrying ``loc``.

    Verified against the installed Prefect (3.7.0) and its FastAPI rather than inferred: the two
    parameter bounds are declared at ``prefect/server/api/concurrency_limits_v2.py`` on the
    ``increment-with-lease`` body, and the list-of-``loc`` shape was reproduced from them.

    The alternative discrimination — requiring the ``"greater than the limit"`` wording — was
    rejected deliberately. The wording is present today, but making the campaign's pause lever
    depend on a server-side message staying phrased the same way trades a latent bug for a
    brittle one: a rephrasing upstream would turn every deliberate pause into a hard failure of
    the next cell to reach the gate. The status stays the signal; this only carves out the one
    shape that provably is not a hold.

    Unreadable bodies answer ``False`` — i.e. still a hold. A 422 whose body cannot be parsed is
    far more likely to be the server's own than FastAPI's, and holding is the recoverable
    direction: an operator sees a gate that will not release and raises the limit, whereas a
    wrongly-propagated failure has already failed a cell by the time anyone looks.
    """
    getter = getattr(response, "json", None)
    if not callable(getter):
        return False
    try:
        body = getter()
    except Exception:
        return False
    if not isinstance(body, dict):
        return False
    detail = body.get("detail")
    if not isinstance(detail, list):
        return False
    return any(isinstance(entry, dict) and "loc" in entry for entry in detail)


def gate_is_holding(exc: BaseException) -> bool:
    """Did this acquisition failure mean "the limit is currently too small", or something else?

    ``422`` is the server's answer to a request for more slots than the limit holds — which, with
    every caller here occupying one slot, means the limit is zero. Anything else (a missing limit
    under ``strict``, a network failure, an unauthorised client) is a real failure and must
    propagate: treating those as a hold would park a run forever on a misconfiguration.

    The status has to be dug out of the cause chain because Prefect wraps the HTTP error in
    ``ConcurrencySlotAcquisitionError``, whose own message and type say nothing about which of the
    two happened.

    One ``422`` is excluded: a FastAPI request-validation failure, which shares the status but is
    a permanent client-side mistake rather than a lowered limit. Treating it as a hold parked the
    gate forever on a malformed call, and the ``isinstance`` arm below matches every Prefect HTTP
    error regardless of wording, so nothing else would have caught it. See
    :func:`_is_request_validation_error` for how the two are told apart and why not by wording.
    """
    seen: set[int] = set()
    cause: BaseException | None = exc
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        response = getattr(cause, "response", None)
        status = getattr(response, "status_code", None)
        if (
            status == 422
            and not _is_request_validation_error(response)
            and (isinstance(cause, PrefectHTTPStatusError) or _TOO_SMALL in str(cause))
        ):
            return True
        cause = cause.__cause__
    return False


class FleetGate(AbstractContextManager):
    """One slot of a named Prefect global concurrency limit, held for the ``with`` body.

    THREAD-SAFE: the active context lives in a per-thread stack rather than an instance slot. One
    gate object is shared between a chained fill's feeder thread (terminal plans commit inside
    ``plan``) and its trailing-assembly thread, and an instance slot would let a concurrent enter
    overwrite the other thread's context and release the wrong slot on exit.

    ``strict=True`` on every acquisition: an absent or misspelled limit must fail closed. Prefect
    defaults to warning and proceeding UNGATED, which would silently reintroduce exactly the
    contention these gates bound.
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
        log: Where a hold announces itself. A hold with no log line is a hung run to anyone
            reading the run's output.
        occupy: Slots per acquisition. One everywhere today, which is what makes "limit below
            what we need" and "limit zero" the same state.
        poll_s: Seconds between attempts while holding.
        should_stop: Consulted while holding, so a runner that is winding down abandons the wait
            instead of parking past its own shutdown. Its exception is the original acquisition
            error, which is the honest one for the caller.
        concurrency_kwargs: Passed through to :func:`concurrency` — lease duration and
            lease-renewal policy differ between the two gates.
        """
        self._name = name
        self._log = log
        self._occupy = occupy
        self._poll_s = poll_s
        self._should_stop = should_stop
        # Typed loosely at the splat, deliberately: the parameter above declares what a CALLER may
        # pass, which is the useful contract, while `concurrency` itself takes heterogeneous
        # parameters (int | None, float, a lease holder, bool | None) that a homogeneous
        # `**dict[str, X]` cannot be checked against — mypy reports one error per parameter
        # whatever X is. A `**kwargs` typing limitation, confined to this one attribute.
        self._kwargs: dict[str, Any] = dict(concurrency_kwargs)
        self._local = threading.local()

    def _acquire(self) -> AbstractContextManager[Any]:
        """Enter one ``concurrency`` context, holding while the gate's limit is too small."""
        held_for = 0.0
        announced_at: float | None = None
        while True:
            cm = concurrency(self._name, occupy=self._occupy, strict=True, **self._kwargs)
            try:
                cm.__enter__()
            # Caught broadly and re-raised unless the cause chain says "limit too small": Prefect's
            # ConcurrencySlotAcquisitionError lives in a private module, and the decision here
            # needs what the server answered, not its type.
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
        """Release the slot; a failure to release never fails the work the slot guarded.

        The work is finished by the time this runs. Raising here recorded six completed cells as
        ingest failures on 2026-09-09 (a Prefect 503 on the decrement) and sent them to the
        fleet-rebuilding retry pass. Swallowing is safe because a slot is leased: renewal stops on
        exit and the server reclaims it one lease later (measured on dev at 1.22 lease periods), so
        the cost is one idle slot for ~15 min on the 900 s ingest lease, not a cumulative leak. A
        clean release is unchanged. The ``pop`` stays outside the guard: an unbalanced stack is our
        bug, not a server condition.
        """
        cm = self._local.stack.pop()
        try:
            cm.__exit__(exc_type, exc, tb)
        except Exception as release_exc:
            if self._log is not None:
                self._log.warning(
                    "Gate %r could not be released (%s: %s); continuing — the slot is leased and "
                    "the server reclaims it, and the guarded work is already done.",
                    self._name,
                    type(release_exc).__name__,
                    release_exc,
                )


#: How long a pause reading is trusted before the server is asked again. What is watched is a human
#: typing one command, so tens of seconds is responsive enough — and the read happens on the
#: inference driver of every cluster, a Prefect API client like any other, so a tight poll would
#: spend orchestrator capacity on asking whether to work.
PAUSE_TTL_S = 30.0


def pause_signal(
    name: str,
    *,
    log: logging.Logger | logging.LoggerAdapter | None = None,
    ttl_s: float = PAUSE_TTL_S,
    read_limit: Callable[[str], tuple[int, bool] | None] | None = None,
) -> Callable[[], bool]:
    """Is work paused? — a cheap callable for a loop that is not allowed to import Prefect.

    A gate used as a PAUSE FLAG rather than a capacity cap: the limit is READ, never acquired, and
    a limit of zero on an active gate means paused. Reading rather than acquiring is what makes
    this usable inside a dispatch loop — no slot to hold, no lease to renew, nothing to release if
    the process dies mid-pause.

    **Fail-open, always.** A read that errors, a gate that does not exist, a server that is unwell
    — all answer "not paused". A campaign that stops working because a monitoring-adjacent read
    failed is a far worse failure than a pause that takes another ``ttl_s`` to take effect. A pause
    is an operator's convenience; running is the job.

    The returned callable is cheap enough for a tight loop (one read per ``ttl_s`` at most) and is
    safe to share across threads: a stale answer by up to ``ttl_s`` is the contract, so a race that
    returns the previous reading is not a defect.

    Args:
        name: The global concurrency limit read as a flag.
        log: Where the transitions announce themselves — entering a pause, and leaving it.
        ttl_s: How long a reading is trusted.
        read_limit: Injection point for tests; returns ``(limit, active)`` or ``None`` when the
            gate is absent. Defaults to a Prefect client read.
    """
    reader = read_limit if read_limit is not None else _read_limit_via_prefect
    state: dict[str, Any] = {"checked_at": None, "paused": False, "since": None, "announced_at": None}
    lock = threading.Lock()

    def paused() -> bool:
        now = time.monotonic()
        with lock:
            fresh = state["checked_at"] is not None and now - state["checked_at"] < ttl_s
            if fresh:
                return bool(state["paused"])
        try:
            reading = reader(name)
        except Exception as exc:
            if log is not None:
                log.warning("Could not read pause gate %r (%s) — treating as NOT paused", name, exc)
            with lock:
                state["checked_at"] = now
                state["paused"] = False
            return False
        is_paused = reading is not None and reading[0] == 0 and reading[1]
        with lock:
            was = bool(state["paused"])
            state["checked_at"] = now
            state["paused"] = is_paused
            if is_paused and not was:
                state["since"] = now
                state["announced_at"] = None
            if is_paused:
                since = state["since"]
                held = now - (since if since is not None else now)
                announce = state["announced_at"] is None or held - state["announced_at"] >= HOLD_LOG_EVERY_S
                if announce:
                    state["announced_at"] = held
            else:
                announce = False
                # `is not None`, not truthiness: a pause that began at monotonic 0.0 is still a
                # pause, and treating it as absent loses the resume line for it.
                resumed_after = now - state["since"] if was and state["since"] is not None else None
                state["since"] = None
        if log is not None:
            if is_paused and announce:
                log.warning(
                    "Inference is PAUSED by gate %r (limit 0) — the fleet holds and takes on no new "
                    "cell (%.0f min so far). In-flight chunks finish and land. Raise the limit with "
                    "`prefect global-concurrency-limit update %s --limit 1` to resume.",
                    name,
                    held / 60.0,
                    name,
                )
            elif not is_paused and resumed_after:
                log.info("Pause gate %r cleared after %.0f min — resuming", name, resumed_after / 60.0)
        return is_paused

    return paused


def _read_limit_via_prefect(name: str) -> tuple[int, bool] | None:
    """``(limit, active)`` for one global concurrency limit, or ``None`` if it does not exist.

    Reads BY NAME rather than listing. The list form takes a `limit` whose default the server caps
    at 200 and truncates silently, so on a workspace holding more limits than that this gate could
    fall off the page and read as absent — and absent is "not paused". An operator lowering the
    gate to zero would then watch clusters keep taking cells, with nothing reporting that the pause
    had not been seen. Asking for the one name removes the boundary rather than moving it.
    """
    from prefect.client.orchestration import get_client
    from prefect.exceptions import ObjectNotFound
    from prefect.utilities.asyncutils import run_coro_as_sync

    async def _read() -> tuple[int, bool] | None:
        async with get_client() as client:
            try:
                gl = await client.read_global_concurrency_limit_by_name(name)
            except ObjectNotFound:
                # A gate that was never created is not a pause — same answer the list form gives
                # when no row matches, and `pause_signal` documents why.
                return None
            return int(gl.limit), bool(gl.active)

    return run_coro_as_sync(_read())
