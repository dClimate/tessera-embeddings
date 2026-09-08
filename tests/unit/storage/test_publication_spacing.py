"""`publication_spacing`: the fleet's publications land at least one spacing apart, or not at all."""

from __future__ import annotations

import itertools
import threading
import time
from contextlib import contextmanager

import pytest

from tessera_embeddings.storage import publication_spacing as mod
from tessera_embeddings.storage.session_catch_up import CATCH_UP_INTERVAL_S


@pytest.fixture(autouse=True)
def _clean_install():
    previous = mod.install_publication_gate(None)
    yield
    mod.install_publication_gate(previous)


class _Recording:
    def __init__(self) -> None:
        self.events: list[str] = []

    @contextmanager
    def gate(self):
        self.events.append("acquire")
        try:
            yield
        finally:
            self.events.append("release")


def test_the_spacing_is_derived_from_the_catch_up_interval_with_room_for_a_tick():
    """Two publications inside one interval is the hang's precondition, so spacing must exceed
    the interval — by enough for a tick's own rebase (0.5-2.8 s measured) to complete inside it.
    """
    assert mod.PUBLICATION_SPACING_S == 3 * CATCH_UP_INTERVAL_S
    assert mod.PUBLICATION_SPACING_S - CATCH_UP_INTERVAL_S >= 5.0, "less than five seconds of margin for a tick"


def test_nothing_installed_means_no_op_and_no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))
    with mod.publication():
        pass
    assert slept == [], "with no gate there is nothing to space against; sleeping would only slow a lone writer"


def test_the_slot_is_held_through_the_body_and_then_for_the_rest_of_the_spacing(monkeypatch):
    rec = _Recording()
    mod.install_publication_gate(rec.gate)
    clock = {"t": 100.0}
    slept: list[float] = []
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))
    with mod.publication(spacing_s=15.0):
        rec.events.append("commit")
        clock["t"] += 2.0  # the commits took two seconds
    assert rec.events == ["acquire", "commit", "release"], "the slot must be taken before and released after"
    assert slept == [13.0], "the holder sleeps the REMAINDER of the spacing, not the whole of it"


def test_a_body_longer_than_the_spacing_does_not_sleep(monkeypatch):
    rec = _Recording()
    mod.install_publication_gate(rec.gate)
    clock = {"t": 0.0}
    slept: list[float] = []
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))
    with mod.publication(spacing_s=15.0):
        clock["t"] += 40.0  # eight mark-commit retries later
    assert slept == []


def test_an_exception_releases_the_slot_at_once_without_spacing(monkeypatch):
    """Nothing was published, so there is nothing to space — and holding the fleet's slot for
    fifteen seconds on every failed commit would tax everyone else for one writer's error.
    """
    rec = _Recording()
    mod.install_publication_gate(rec.gate)
    slept: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))
    with pytest.raises(RuntimeError, match="rebase failed"), mod.publication(spacing_s=15.0):
        raise RuntimeError("rebase failed")
    assert rec.events == ["acquire", "release"]
    assert slept == []


def test_install_returns_the_previous_gate_and_none_removes_it():
    gate = _Recording().gate  # bound once: every attribute access would mint a new bound method
    assert mod.install_publication_gate(gate) is None
    assert mod.installed_publication_gate() is gate
    assert mod.install_publication_gate(None) is gate
    assert mod.installed_publication_gate() is None


def test_consecutive_publications_across_threads_start_at_least_one_spacing_apart():
    """The property the whole module exists for, with a REAL lock as the gate and a real (short)
    spacing: however many writers race, publication start times are pairwise >= spacing apart.

    A mutex alone would let publications land back to back; the sleep-after is what makes them
    spaced. Remove the sleep and this fails.
    """
    lock = threading.Lock()

    @contextmanager
    def gate():
        with lock:
            yield

    mod.install_publication_gate(gate)
    spacing = 0.15
    starts: list[float] = []
    starts_lock = threading.Lock()

    def publish():
        with mod.publication(spacing_s=spacing), starts_lock:
            starts.append(time.monotonic())

    threads = [threading.Thread(target=publish) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(starts) == 5
    gaps = [b - a for a, b in itertools.pairwise(sorted(starts))]
    assert all(g >= spacing * 0.95 for g in gaps), f"publications landed closer than the spacing: {gaps}"
