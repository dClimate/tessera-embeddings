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
from types import SimpleNamespace

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
    limit is zero. This is the whole pause mechanism.
    """
    exc = _wrapped(_server_said(422, "Slots requested is greater than the limit"))
    assert gate_is_holding(exc)


def test_a_missing_limit_is_not_a_hold() -> None:
    """Strict mode's own error carries no HTTP status at all. It must fail: an absent gate is a
    misconfiguration, and holding on it would park the campaign on a typo with no way to tell
    that from a deliberate pause.
    """
    assert not gate_is_holding(RuntimeError("Concurrency limits ['typo'] must be created before acquiring slots"))


def test_a_full_gate_is_not_a_hold_here() -> None:
    """423 is ordinary queueing and Prefect already waits on it, so it never reaches this code.
    If it ever did, treating it as a hold would double the waiting.
    """
    assert not gate_is_holding(_wrapped(_server_said(423, "Locked")))


@pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
def test_no_other_status_is_a_hold(status: int) -> None:
    """An unauthorised client and an unwell server are failures, not pauses. Holding on either
    turns a broken campaign into a silently stalled one.
    """
    assert not gate_is_holding(_wrapped(_server_said(status, "nope")))


def test_a_cause_cycle_does_not_hang_the_matcher() -> None:
    """Exception chains can be cyclic once something sets __cause__ by hand, and this walk runs
    inside the acquisition path of every commit.
    """
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert not gate_is_holding(a)


# --- what the gate does about it --------------------------------------------------------


def test_a_zeroed_gate_holds_and_then_proceeds(monkeypatch, no_sleeping) -> None:
    """The pause lever, end to end: three refusals, then the limit is raised and the work runs.
    Nothing fails and nothing is lost.
    """
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
    with pytest.raises(RuntimeError, match="must be created"), FleetGate("typo", log=logging.getLogger("test")):
        pass  # pragma: no cover
    assert no_sleeping == []


def test_a_hold_is_abandoned_when_the_runner_is_shutting_down(monkeypatch, no_sleeping) -> None:
    """A pause must not outlive the run it is pausing. The cells behind the gate are the
    driver's next round's work, so raising here loses nothing — while parking past a shutdown
    would leave a thread waiting on a limit no one is going to raise.
    """
    stopping = threading.Event()
    stopping.set()
    cm = _Opens(_wrapped(_server_said(422, "Slots requested is greater than the limit")), fail_times=99)
    monkeypatch.setattr(mod, "concurrency", cm)
    with (
        pytest.raises(RuntimeError, match="Unable to acquire"),
        FleetGate("gate", log=logging.getLogger("test"), should_stop=stopping.is_set),
    ):
        pass  # pragma: no cover
    assert no_sleeping == []


def test_a_hold_announces_itself_once_and_then_periodically(monkeypatch, no_sleeping, caplog) -> None:
    """A hold is indefinite, so silence would be indistinguishable from a hung run — but a line
    every poll would bury the log of a paused campaign. First attempt, then every
    HOLD_LOG_EVERY_S of holding.
    """
    polls = int(mod.HOLD_LOG_EVERY_S / 30.0)
    cm = _Opens(
        _wrapped(_server_said(422, "Slots requested is greater than the limit")),
        fail_times=polls * 2 + 1,
    )
    monkeypatch.setattr(mod, "concurrency", cm)
    with caplog.at_level(logging.WARNING), FleetGate("gate", log=logging.getLogger("hold-test"), poll_s=30.0):
        pass
    held = [r for r in caplog.records if "HELD" in r.getMessage()]
    assert len(held) == 3, [r.getMessage() for r in held]
    # The message has to carry the way out; a paused campaign is read by whoever is on call.
    assert "global-concurrency-limit update" in held[0].getMessage()


def test_the_gate_survives_having_no_logger(monkeypatch, no_sleeping) -> None:
    """The commit gate is constructed before a run logger exists on some paths, and a hold must
    not turn into an AttributeError inside a commit.
    """
    cm = _Opens(_wrapped(_server_said(422, "Slots requested is greater than the limit")), fail_times=1)
    monkeypatch.setattr(mod, "concurrency", cm)
    with FleetGate("gate"):
        pass
    assert cm.entered == 1


def test_the_lease_kwargs_reach_prefect(monkeypatch, no_sleeping) -> None:
    """The ingest gate's generous lease and its "do not die on a renewal blip" policy are what
    let an hours-long ingest hold a slot. Passing them through the new indirection is not
    optional, and a dropped kwarg is invisible until an ingest dies at five minutes.
    """
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


# --- the pause signal ------------------------------------------------------------------
#
# A gate read as a FLAG rather than acquired, for a loop that has to ask permission to work.
# The asymmetry is deliberate and is the whole safety argument: a wrong "paused" stops a
# campaign, a wrong "running" costs nothing, so every uncertain answer is "running".


def _reading(limit: int, active: bool = True):
    return lambda name: (limit, active)


def test_zero_on_an_active_gate_is_paused() -> None:
    assert mod.pause_signal("g", read_limit=_reading(0))() is True


@pytest.mark.parametrize("limit", [1, 8, 61])
def test_any_positive_limit_means_run(limit: int) -> None:
    """The gate is a flag, not a cap — nothing acquires a slot, so the magnitude is only
    "not zero". A campaign start writes 1.
    """
    assert mod.pause_signal("g", read_limit=_reading(limit))() is False


def test_an_inactive_gate_is_not_a_pause() -> None:
    """Deactivating a gate makes the server grant slots to everyone, so reading it as a
    pause would give one state two opposite meanings across the two gate mechanisms.
    """
    assert mod.pause_signal("g", read_limit=_reading(0, active=False))() is False


def test_an_absent_gate_is_not_a_pause() -> None:
    """A campaign whose gate was never created must run, not sit still."""
    assert mod.pause_signal("g", read_limit=lambda name: None)() is False


class TestTheGateIsReadByName:
    """The lookup must not be able to miss a gate that exists.

    It used to list limits with `limit=200` and scan for the name. The server caps that
    page and truncates silently, so on a workspace holding more limits than the page the
    gate could fall off it and read as ABSENT — which `pause_signal` treats as "not
    paused". An operator lowering the gate to zero would then watch clusters keep taking
    cells, with nothing reporting that the pause had not been seen.
    """

    @staticmethod
    def _client(monkeypatch, *, by_name, listed=None):
        """Stub the Prefect client, failing loudly if the list form is used at all."""

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def read_global_concurrency_limit_by_name(self, name):
                return by_name(name)

            async def read_global_concurrency_limits(self, limit=10, offset=0):
                raise AssertionError("the gate must be read by name, not by listing")

        monkeypatch.setattr("prefect.client.orchestration.get_client", lambda: _Client())

    def test_a_gate_is_found_regardless_of_how_many_others_exist(self, monkeypatch) -> None:
        self._client(monkeypatch, by_name=lambda n: SimpleNamespace(name=n, limit=0, active=True))
        assert mod._read_limit_via_prefect("campaign-pause") == (0, True)

    def test_an_absent_gate_still_reads_as_absent(self, monkeypatch) -> None:
        """`None`, not an exception — `pause_signal` turns it into "not paused"."""
        from prefect.exceptions import ObjectNotFound

        def _missing(_name):
            # ObjectNotFound requires the HTTP error it wraps — construct it as the client would.
            raise ObjectNotFound(http_exc=Exception("404"))

        self._client(monkeypatch, by_name=_missing)
        assert mod._read_limit_via_prefect("never-created") is None


def test_a_failed_read_is_not_a_pause() -> None:
    """Fail-open, and this is the case that matters: the read happens on every cluster's
    inference driver, so a wobbly API would otherwise stop the entire campaign working.
    """

    def boom(name: str):
        raise RuntimeError("api down")

    assert mod.pause_signal("g", read_limit=boom)() is False


def test_a_reading_is_cached_for_its_ttl(monkeypatch) -> None:
    """One read per TTL at most. The check sits in a dispatch loop that runs per chunk
    hand-over, and 2,500 actors' worth of hand-overs asking the orchestrator whether to work
    would spend real API capacity on the question.
    """
    reads = {"n": 0}

    def counted(name: str):
        reads["n"] += 1
        return (1, True)

    clock = {"t": 1000.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
    paused = mod.pause_signal("g", read_limit=counted, ttl_s=30.0)
    for _ in range(50):
        paused()
    assert reads["n"] == 1
    clock["t"] += 31.0
    paused()
    assert reads["n"] == 2


def test_the_pause_is_noticed_within_a_ttl(monkeypatch) -> None:
    """The other side of caching: a stale answer must expire, or the pause never takes."""
    state = {"limit": 1}
    clock = {"t": 0.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
    paused = mod.pause_signal("g", read_limit=lambda name: (state["limit"], True), ttl_s=30.0)
    assert paused() is False
    state["limit"] = 0
    assert paused() is False, "within the TTL the previous reading stands"
    clock["t"] += 31.0
    assert paused() is True


def test_entering_and_leaving_a_pause_are_both_logged(monkeypatch, caplog) -> None:
    """A paused fleet is an expensive, silent state — the log is the only place an operator
    sees that it is deliberate, and the resume line is what confirms the lever worked.
    """
    state = {"limit": 0}
    clock = {"t": 0.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
    log = logging.getLogger("pause-test")
    paused = mod.pause_signal("g", log=log, read_limit=lambda name: (state["limit"], True), ttl_s=30.0)
    with caplog.at_level(logging.INFO):
        assert paused() is True
        clock["t"] += 31.0
        state["limit"] = 1
        assert paused() is False
    messages = [r.getMessage() for r in caplog.records]
    assert any("PAUSED" in m and "global-concurrency-limit update" in m for m in messages), messages
    assert any("cleared after" in m for m in messages), messages
