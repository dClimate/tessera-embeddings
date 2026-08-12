"""The gate that HOLDS on a zeroed limit instead of failing the work behind it.

Lowering a gate to zero is the campaign's pause lever, so the tests here are about one
distinction: "the limit is currently too small" (hold, indefinitely, and say so) versus every
other acquisition failure (fail now, loudly). Getting it backwards either way is expensive —
holding on a missing limit parks a campaign forever on a typo, and failing on a zeroed limit
kills a cell every time an operator tries to pause.

The 422 exception is built from a REAL ``httpx`` response through Prefect's own wrapper, not
from a stub raising a chosen type. A hand-made exception would prove the matcher recognises
hand-made exceptions.
"""

from __future__ import annotations

import logging
import threading

import httpx
import pytest
from prefect.exceptions import PrefectHTTPStatusError

from tessera_embeddings.orchestration.prefect import _fleet_gate as mod
from tessera_embeddings.orchestration.prefect._fleet_gate import FleetGate, gate_is_holding


def _server_said(status: int, detail: str) -> PrefectHTTPStatusError:
    """The exception Prefect raises when the API answers ``status``, built through its own path."""
    request = httpx.Request("POST", "http://prefect/api/v2/concurrency_limits/increment-with-lease")
    response = httpx.Response(status, json={"detail": detail}, request=request)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return PrefectHTTPStatusError.from_httpx_error(exc)
    raise AssertionError(f"{status} did not raise")  # pragma: no cover


def _wrapped(inner: BaseException) -> Exception:
    """The shape Prefect delivers: its own error with the HTTP error as the cause."""
    exc = RuntimeError("Unable to acquire concurrency slots on ['tessera-global-ingests']")
    exc.__cause__ = inner
    return exc


class _Opens:
    """A concurrency context that refuses ``fail_times`` times, then admits."""

    def __init__(self, error: BaseException, fail_times: int) -> None:
        self.error = error
        self.remaining = fail_times
        self.entered = 0
        self.exited = 0

    def __call__(self, name, occupy=1, strict=True, **kw):
        return self

    def __enter__(self):
        if self.remaining > 0:
            self.remaining -= 1
            raise self.error
        self.entered += 1

    def __exit__(self, *exc):
        self.exited += 1
        return False


@pytest.fixture
def no_sleeping(monkeypatch):
    """Hold instantly, and count the waits — a real 30 s poll would make this a slow test."""
    waits: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", waits.append)
    return waits


# --- what counts as a hold -------------------------------------------------------------


def test_a_422_from_the_server_is_a_hold() -> None:
    """The server's answer to "more slots than the limit holds", which with occupy=1 means the
    limit is zero. This is the whole pause mechanism."""
    exc = _wrapped(_server_said(422, "Slots requested is greater than the limit"))
    assert gate_is_holding(exc)


def test_a_missing_limit_is_not_a_hold() -> None:
    """Strict mode's own error carries no HTTP status at all. It must fail: an absent gate is a
    misconfiguration, and holding on it would park the campaign on a typo with no way to tell
    that from a deliberate pause."""
    assert not gate_is_holding(RuntimeError("Concurrency limits ['typo'] must be created before acquiring slots"))


def test_a_full_gate_is_not_a_hold_here() -> None:
    """423 is ordinary queueing and Prefect already waits on it, so it never reaches this code.
    If it ever did, treating it as a hold would double the waiting."""
    assert not gate_is_holding(_wrapped(_server_said(423, "Locked")))


@pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
def test_no_other_status_is_a_hold(status: int) -> None:
    """An unauthorised client and an unwell server are failures, not pauses. Holding on either
    turns a broken campaign into a silently stalled one."""
    assert not gate_is_holding(_wrapped(_server_said(status, "nope")))


def test_a_cause_cycle_does_not_hang_the_matcher() -> None:
    """Exception chains can be cyclic once something sets __cause__ by hand, and this walk runs
    inside the acquisition path of every commit."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert not gate_is_holding(a)


# --- what the gate does about it --------------------------------------------------------


def test_a_zeroed_gate_holds_and_then_proceeds(monkeypatch, no_sleeping) -> None:
    """The pause lever, end to end: three refusals, then the limit is raised and the work runs.
    Nothing fails and nothing is lost."""
    cm = _Opens(_wrapped(_server_said(422, "Slots requested is greater than the limit")), fail_times=3)
    monkeypatch.setattr(mod, "concurrency", cm)
    gate = FleetGate("tessera-global-ingests", log=logging.getLogger("test"), poll_s=30.0)
    with gate:
        pass
    assert cm.entered == 1 and cm.exited == 1
    assert no_sleeping == [30.0, 30.0, 30.0]


def test_a_missing_gate_fails_immediately_without_holding(monkeypatch, no_sleeping) -> None:
    """The regression that matters most in the other direction."""
    cm = _Opens(RuntimeError("Concurrency limits ['typo'] must be created before acquiring slots"), fail_times=99)
    monkeypatch.setattr(mod, "concurrency", cm)
    with pytest.raises(RuntimeError, match="must be created"):
        with FleetGate("typo", log=logging.getLogger("test")):
            pass  # pragma: no cover
    assert no_sleeping == []


def test_a_hold_is_abandoned_when_the_runner_is_shutting_down(monkeypatch, no_sleeping) -> None:
    """A pause must not outlive the run it is pausing. The cells behind the gate are the
    driver's next round's work, so raising here loses nothing — while parking past a shutdown
    would leave a thread waiting on a limit no one is going to raise."""
    stopping = threading.Event()
    stopping.set()
    cm = _Opens(_wrapped(_server_said(422, "Slots requested is greater than the limit")), fail_times=99)
    monkeypatch.setattr(mod, "concurrency", cm)
    with pytest.raises(RuntimeError, match="Unable to acquire"):
        with FleetGate("gate", log=logging.getLogger("test"), should_stop=stopping.is_set):
            pass  # pragma: no cover
    assert no_sleeping == []


def test_a_hold_announces_itself_once_and_then_periodically(monkeypatch, no_sleeping, caplog) -> None:
    """A hold is indefinite, so silence would be indistinguishable from a hung run — but a line
    every poll would bury the log of a paused campaign. First attempt, then every
    HOLD_LOG_EVERY_S of holding."""
    polls = int(mod.HOLD_LOG_EVERY_S / 30.0)
    cm = _Opens(
        _wrapped(_server_said(422, "Slots requested is greater than the limit")),
        fail_times=polls * 2 + 1,
    )
    monkeypatch.setattr(mod, "concurrency", cm)
    with caplog.at_level(logging.WARNING):
        with FleetGate("gate", log=logging.getLogger("hold-test"), poll_s=30.0):
            pass
    held = [r for r in caplog.records if "HELD" in r.getMessage()]
    assert len(held) == 3, [r.getMessage() for r in held]
    # The message has to carry the way out; a paused campaign is read by whoever is on call.
    assert "global-concurrency-limit update" in held[0].getMessage()


def test_the_gate_survives_having_no_logger(monkeypatch, no_sleeping) -> None:
    """The commit gate is constructed before a run logger exists on some paths, and a hold must
    not turn into an AttributeError inside a commit."""
    cm = _Opens(_wrapped(_server_said(422, "Slots requested is greater than the limit")), fail_times=1)
    monkeypatch.setattr(mod, "concurrency", cm)
    with FleetGate("gate"):
        pass
    assert cm.entered == 1


def test_the_lease_kwargs_reach_prefect(monkeypatch, no_sleeping) -> None:
    """The ingest gate's generous lease and its "do not die on a renewal blip" policy are what
    let an hours-long ingest hold a slot. Passing them through the new indirection is not
    optional, and a dropped kwarg is invisible until an ingest dies at five minutes."""
    seen: dict = {}

    class _CM:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    def fake(name, occupy=1, strict=True, **kw):
        seen.update({"name": name, "occupy": occupy, "strict": strict, **kw})
        return _CM()

    monkeypatch.setattr(mod, "concurrency", fake)
    with FleetGate("gate", occupy=2, lease_duration=900.0, raise_on_lease_renewal_failure=False):
        pass
    assert seen == {
        "name": "gate",
        "occupy": 2,
        "strict": True,
        "lease_duration": 900.0,
        "raise_on_lease_renewal_failure": False,
    }
