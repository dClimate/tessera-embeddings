"""fill_zones_sequential's ``_DeploymentCellInputs``: the threaded adapter that runs each cell's
ingest as a child deployment run.

It dispatches child runs inside the fill's run context, polls them to a terminal state, tags them
so the cancellation hook can sweep them, and on shutdown cancels and waits for every child it
started. ``wait_first`` decides when the GPUs may be requested: a failed ingest is not a landed
mosaic, and the priming abort quorum is the first wave. Driven against a fake Prefect client; no
engine runs.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from prefect.states import StateType

import tessera_embeddings.orchestration.prefect.flows._child_runs as _child_runs
import tessera_embeddings.orchestration.prefect.flows.fill_zones_sequential as mod
from tests.unit.deadline import run_under_deadline

# ---------------------------------------------------------------------------
# _DeploymentCellInputs: poll-based child runs + shutdown cancellation
# ---------------------------------------------------------------------------


def _adapter(**over):
    kwargs = dict(
        deployment="ingest-zone-year/ingest-zone-year",
        params_for=lambda z, y: {"zone": z, "year": y},
        inputs_bucket="s3://in",
        cleanup_mosaics=True,
        max_parallel=2,
        log=logging.getLogger("test-adapter"),
    )
    kwargs.update(over)
    return mod._DeploymentCellInputs(**kwargs)


class _FakeClient:
    """Sync-client stand-in: scripted read_flow_run states + cancel recording."""

    def __init__(self, states):
        self._states = list(states)
        self.cancelled: list = []
        self.reads_after_cancel = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read_flow_run(self, fr_id):
        if self.cancelled:
            self.reads_after_cancel += 1
        state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        return SimpleNamespace(id=fr_id, state=state)

    def set_flow_run_state(self, fr_id, state, **kw):
        self.cancelled.append((fr_id, state.type))


def _state(state_type, *, final):
    return SimpleNamespace(type=state_type, name=state_type.value, is_final=lambda: final)


def _running_child(monkeypatch, run_id: str):
    """An adapter whose dispatched child run never reaches a terminal state.

    The shape every shutdown/abort test needs: a fast poll interval, a
    ``run_deployment`` that reports one child id, and a client that keeps
    answering RUNNING — so the poller stays in its loop and the test can act on a
    genuinely in-flight run rather than a race against a child that finished.
    Returns ``(adapter, client)``; the client records what was cancelled.
    """
    monkeypatch.setattr(mod, "_INGEST_POLL_S", 0.01)
    # This child NEVER reports terminal, and both discard() and shutdown() now wait for
    # confirmation — so the confirmation window has to be short or every test using this
    # helper sits out the production five minutes.
    monkeypatch.setattr(mod, "CANCELLATION_CONFIRM_S", 0.05)
    monkeypatch.setattr(
        mod, "run_deployment", lambda dep, parameters, timeout, tags=None: SimpleNamespace(id=run_id, state=None)
    )
    client = _FakeClient([_state(StateType.RUNNING, final=False)])
    monkeypatch.setattr(mod, "get_client", lambda sync_client=True: client)
    return _adapter(), client


def _await_inflight(adapter, expected: dict) -> None:
    """Block until the adapter's worker thread has registered its in-flight run."""
    for _ in range(200):
        if adapter._inflight:
            break
        time.sleep(0.01)
    assert adapter._inflight == expected


def test_ingest_runs_are_created_inside_the_fills_run_context(monkeypatch):
    """Worker threads inherit none of the flow's context, so without entering a copy of the fill's
    run context around the dispatch, every ingest run is created unattached to the fill.
    """
    entered: list[str] = []

    @contextmanager
    def fill_context():
        entered.append("in")
        yield
        entered.append("out")

    monkeypatch.setattr(mod, "FlowRunContext", SimpleNamespace(get=lambda: SimpleNamespace(model_copy=fill_context)))
    seen = []

    def run_deployment(dep, parameters, timeout, tags=None):
        seen.append((threading.current_thread().name.startswith("cell-ingest"), list(entered)))
        return SimpleNamespace(id="fr-1", state=None)

    monkeypatch.setattr(mod, "run_deployment", run_deployment)
    monkeypatch.setattr(
        mod, "get_client", lambda sync_client=True: _FakeClient([_state(StateType.COMPLETED, final=True)])
    )
    adapter = _adapter()
    try:
        adapter.wait("33N", 2025)
    finally:
        adapter.shutdown()
    assert seen == [(True, ["in"])] and entered == ["in", "out"]


def test_a_dispatch_queued_while_shutdown_begins_creates_no_run(monkeypatch):
    """A worker past the first stop check but still waiting for the lock must create nothing:
    shutdown has already snapshotted the runs it will cancel, so a run made now is never swept.
    """
    created = []
    monkeypatch.setattr(
        mod, "run_deployment", lambda *a, **k: created.append(1) or SimpleNamespace(id="fr-late", state=None)
    )
    adapter = _adapter()
    with adapter._lock:
        adapter.start("33N", 2025)
        time.sleep(0.2)  # the worker is now queued on the lock
        adapter._stopping.set()
    with pytest.raises(RuntimeError, match="not dispatched"):
        adapter._futures[("33N", 2025)].result(timeout=5)
    adapter.shutdown()
    assert created == []


def test_shutdown_raises_the_stop_flag_before_waiting_for_the_lock():
    """Dispatchers queued on the lock must see the flag the moment they get it, so none of them
    creates a run while shutdown is still waiting its turn behind an in-progress creation.
    """
    adapter = _adapter()
    with adapter._lock:  # stands in for a creation in progress
        stopper = threading.Thread(target=adapter.shutdown)
        stopper.start()
        assert adapter._stopping.wait(2), "the flag must rise before shutdown takes the lock"
    stopper.join(5)


def test_adapter_shutdown_cancels_in_flight_child_runs(monkeypatch):
    """The reviewer's orphan scenario: parent dies while a child ingest runs.
    shutdown() must request cancellation of the in-flight child flow run —
    otherwise it keeps writing the mosaic prefix a prompt retry would race.
    """
    adapter, client = _running_child(monkeypatch, "fr-9")
    adapter.start("33N", 2025)
    _await_inflight(adapter, {("33N", 2025): "fr-9"})
    adapter.shutdown()
    # The sweep cancels synchronously; the woken poller may ALSO self-cancel
    # (belt + suspenders for the late-registration window) — assert the sweep's.
    assert client.cancelled[0] == ("fr-9", StateType.CANCELLING)


def test_discard_waits_for_the_cancelled_child_to_actually_stop(monkeypatch):
    """A retry must not put a second ingest onto a mosaic prefix the first still holds.

    Polling gives up after a persistent run of API errors WITHOUT deregistering the
    child, precisely because the server-side ingest may still be healthy. The retry path
    then calls discard() then start().

    Cancellation is a REQUEST: Prefect marks the run Cancelling and a worker acts on it
    later, so asking and returning leaves a live writer behind — and dropping the run id
    in the same breath removed it from shutdown's sweep too, which was the one thing that
    would have caught it. discard therefore FOLLOWS the request to a terminal state.
    """
    monkeypatch.setattr(mod, "_INGEST_POLL_S", 0.01)
    monkeypatch.setattr(
        mod, "run_deployment", lambda dep, parameters, timeout, tags=None: SimpleNamespace(id="fr-x", state=None)
    )
    # RUNNING while the poller holds it, then terminal once cancellation lands.
    client = _FakeClient([_state(StateType.RUNNING, final=False)] * 3 + [_state(StateType.CANCELLED, final=True)])
    monkeypatch.setattr(mod, "get_client", lambda sync_client=True: client)
    adapter = _adapter()
    adapter.start("33N", 2025)
    _await_inflight(adapter, {("33N", 2025): "fr-x"})

    try:
        adapter.discard("33N", 2025)
        assert ("fr-x", StateType.CANCELLING) in client.cancelled
        assert adapter._inflight == {}  # released only once it was CONFIRMED finished
        assert ("33N", 2025) not in adapter._futures  # a fresh start() will re-submit
    finally:
        adapter.shutdown()


def test_discard_refuses_the_retry_when_the_child_will_not_confirm(monkeypatch):
    """Unconfirmed is not the same as stopped.

    Losing a recoverable cell to a failure that names itself is the cheaper mistake: a
    mosaic written by two runs at once is undetectable downstream, and the ingest path's
    commits do not rebase, so the loser's failure is terminal anyway.
    """
    adapter, client = _running_child(monkeypatch, "fr-stuck")  # never reports terminal
    adapter.start("33N", 2025)
    _await_inflight(adapter, {("33N", 2025): "fr-stuck"})

    try:
        with pytest.raises(RuntimeError, match="did not reach a terminal state"):
            adapter.discard("33N", 2025)
        # STILL registered, so shutdown's sweep can reach it.
        assert adapter._inflight == {("33N", 2025): "fr-stuck"}
    finally:
        adapter.shutdown()


def test_discard_refuses_the_retry_when_the_api_cannot_answer(monkeypatch):
    """The Prefect API being unreachable is the likeliest reason this path is reached at
    all, and it is exactly when 'the old run has stopped' cannot be established.
    """
    adapter, client = _running_child(monkeypatch, "fr-unreachable")
    adapter.start("33N", 2025)
    _await_inflight(adapter, {("33N", 2025): "fr-unreachable"})

    def _boom(*_a, **_k):
        raise ConnectionError("prefect api down")

    client.set_flow_run_state = _boom
    try:
        with pytest.raises(RuntimeError, match="could not confirm"):
            adapter.discard("33N", 2025)
    finally:
        adapter._inflight.clear()  # else shutdown re-raises into the same dead client
        adapter.shutdown()


def test_adapter_wait_honors_the_runner_stop_event(monkeypatch):
    """A blocked wait() returns promptly (raising) once the runner's stop event
    is set — the feeder must never sit behind a full ingest during unwind.
    """
    adapter, client = _running_child(monkeypatch, "fr-2")
    stop = threading.Event()
    t0 = time.monotonic()
    threading.Timer(0.05, stop.set).start()
    try:
        with pytest.raises(RuntimeError, match="aborted: runner stopping"):
            adapter.wait("33N", 2025, stop=stop)
        assert time.monotonic() - t0 < 30
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# Shutdown-race hardening + the cancellation-hook child sweep
# ---------------------------------------------------------------------------


def test_abandoned_poller_cancels_its_own_child(monkeypatch):
    """A poller woken by shutdown cancels its own child with its open client
    before raising — closing the late-registration window the snapshot sweep
    can miss.
    """
    adapter, client = _running_child(monkeypatch, "fr-7")
    adapter.start("33N", 2025)
    fut = adapter._futures[("33N", 2025)]

    for _ in range(200):  # let the worker register before shutting down
        if adapter._inflight:
            break
        time.sleep(0.01)
    adapter.shutdown()
    with pytest.raises(RuntimeError, match="abandoned"):
        fut.result(timeout=5)
    assert ("fr-7", StateType.CANCELLING) in client.cancelled  # self-cancel happened


def test_adapter_tags_child_runs_with_parent_derived_tag(monkeypatch):
    monkeypatch.setattr(mod, "_INGEST_POLL_S", 0.01)
    captured: dict = {}

    def fake_run_deployment(dep, parameters, timeout, tags=None):
        captured["tags"] = tags
        return SimpleNamespace(id="fr-3", state=None)

    monkeypatch.setattr(mod, "run_deployment", fake_run_deployment)
    client = _FakeClient([_state(StateType.COMPLETED, final=True)])
    monkeypatch.setattr(mod, "get_client", lambda sync_client=True: client)
    adapter = _adapter(child_tag="chained-ingest:run-xyz")
    try:
        adapter.wait("33N", 2025)
    finally:
        adapter.shutdown()
    assert captured["tags"] == ["chained-ingest:run-xyz"]


def test_cancel_child_ingests_hook_sweeps_live_runs_by_tag(monkeypatch):
    """The hook re-derives the parent tag and cancels only LIVE children — the
    in-process shutdown() never runs when Prefect kills the flow process.

    Patched on `_child_runs`, which is where the hook resolves its client: the
    sweep is shared with run-global-campaign rather than copied per flow.
    """
    live = SimpleNamespace(id="fr-live", state=_state(StateType.RUNNING, final=False))
    done = SimpleNamespace(id="fr-done", state=_state(StateType.COMPLETED, final=True))

    class _SweepClient(_FakeClient):
        def __init__(self):
            super().__init__([_state(StateType.RUNNING, final=False)])
            self.filters: list = []

        def read_flow_runs(self, flow_run_filter=None, limit=None, offset=0):
            self.filters.append(flow_run_filter)
            return [live, done]

    client = _SweepClient()
    monkeypatch.setattr(_child_runs, "get_client", lambda sync_client=True: client)
    mod._cancel_child_ingests_on_cancellation(None, SimpleNamespace(id="run-1"), None)
    assert client.cancelled == [("fr-live", StateType.CANCELLING)]  # final child untouched
    assert client.filters[0].tags.all_ == ["chained-ingest:run-1"]


def test_child_sweep_pages_past_the_server_default(monkeypatch):
    """A live child beyond the first page is still cancelled.

    A campaign tags one child per (zone, year) and they keep the tag after they
    finish, so the tagged set outgrows one page long before the run ends. An
    unpaginated sweep would cancel nothing outside the first page and leave that
    child's GPU fleet billing against the mosaic a retry is about to rewrite.
    """
    page_size = _child_runs._PAGE
    done = [SimpleNamespace(id=f"fr-{i}", state=_state(StateType.COMPLETED, final=True)) for i in range(page_size)]
    live = SimpleNamespace(id="fr-live-late", state=_state(StateType.RUNNING, final=False))

    class _PagedClient(_FakeClient):
        def __init__(self):
            super().__init__([_state(StateType.RUNNING, final=False)])
            self.offsets: list[int] = []

        def read_flow_runs(self, flow_run_filter=None, limit=None, offset=0):
            self.offsets.append(offset)
            return done if offset == 0 else [live]

    client = _PagedClient()
    monkeypatch.setattr(_child_runs, "get_client", lambda sync_client=True: client)
    mod._cancel_child_ingests_on_cancellation(None, SimpleNamespace(id="run-1"), None)
    assert client.offsets == [0, page_size]  # a full page means ask again
    assert client.cancelled == [("fr-live-late", StateType.CANCELLING)]


class _FlakyClient(_FakeClient):
    """read_flow_run raises `errors` times, then returns a terminal run."""

    def __init__(self, errors: int, terminal_state):
        super().__init__([terminal_state])
        self._errors_left = errors
        self.reads = 0

    def read_flow_run(self, fr_id):
        self.reads += 1
        if self._errors_left > 0:
            self._errors_left -= 1
            raise ConnectionError("transient prefect api blip")
        return SimpleNamespace(id=fr_id, state=self._states[0])


def test_poll_retries_transient_errors_then_completes(monkeypatch):
    """A transient read error must NOT fail the ingest wait — the poller retries
    across it and the id stays registered throughout (visible to shutdown).
    """
    monkeypatch.setattr(mod, "_INGEST_POLL_S", 0.001)
    monkeypatch.setattr(
        mod, "run_deployment", lambda dep, parameters, timeout, tags=None: SimpleNamespace(id="fr-4", state=None)
    )
    client = _FlakyClient(errors=3, terminal_state=_state(StateType.COMPLETED, final=True))
    monkeypatch.setattr(mod, "get_client", lambda sync_client=True: client)
    adapter = _adapter()
    try:
        adapter.wait("33N", 2025)  # succeeds despite 3 transient errors
        assert client.reads == 4  # 3 errors + 1 terminal read
        assert adapter._inflight == {}  # deregistered only after the terminal read
    finally:
        adapter.shutdown()


def test_poll_error_keeps_id_registered_for_shutdown(monkeypatch):
    """A persistent poll failure gives up — but must leave the child id in
    _inflight so shutdown can still cancel the live server-side run.
    """
    monkeypatch.setattr(mod, "_INGEST_POLL_S", 0.001)
    monkeypatch.setattr(mod, "_INGEST_POLL_MAX_ERRORS", 3)
    monkeypatch.setattr(mod, "CANCELLATION_CONFIRM_S", 0.05)
    monkeypatch.setattr(
        mod, "run_deployment", lambda dep, parameters, timeout, tags=None: SimpleNamespace(id="fr-5", state=None)
    )

    class _AlwaysErrors(_FakeClient):
        def read_flow_run(self, fr_id):
            raise ConnectionError("prefect api down")

    client = _AlwaysErrors([_state(StateType.RUNNING, final=False)])
    monkeypatch.setattr(mod, "get_client", lambda sync_client=True: client)
    adapter = _adapter()
    adapter.start("33N", 2025)
    with pytest.raises(RuntimeError, match="gave up after"):
        adapter._futures[("33N", 2025)].result(timeout=5)
    assert ("33N", 2025) in adapter._inflight  # NOT deregistered → shutdown can sweep it
    adapter.shutdown()
    assert ("fr-5", StateType.CANCELLING) in client.cancelled


# ---------------------------------------------------------------------------
# wait_first: a failed ingest is not a landed mosaic
# ---------------------------------------------------------------------------


def _settled(exc: BaseException | None):
    """A finished ``Future`` carrying ``exc`` (or a result when None)."""
    fut: Future = Future()
    fut.set_exception(exc) if exc is not None else fut.set_result(None)
    return fut


def test_wait_first_skips_a_failed_cell_while_a_sibling_is_still_ingesting():
    """A fast failure must not be mistaken for a mosaic and boot a GPU fleet.

    The caller's next act after this returns is ``ray up``. Bad credentials or a bad
    parameter fail a child ingest within seconds, so returning it as "landed" spun up the
    paid fleet immediately, for a mosaic that does not exist, only for the feeder to
    surface the same failure and tear it back down. The sibling is what the wait is for.
    """
    adapter = _adapter()
    try:
        pending: Future = Future()
        adapter._futures = {("01N", 2024): _settled(RuntimeError("bad credentials")), ("02N", 2024): pending}
        threading.Timer(0.05, lambda: pending.set_result(None)).start()

        assert adapter.wait_first([("01N", 2024), ("02N", 2024)]) == ("02N", 2024)
    finally:
        adapter.shutdown()


def test_cancel_unstarted_drops_queued_ingests_but_not_running_ones():
    """A retry must not queue behind a cluster's worth of unattempted ingests.

    Future.cancel() succeeds only for a task still queued, so this must drop the queued ones
    and leave the RUNNING one alone — dropping work, never interrupting it.
    """
    adapter = _adapter(max_parallel=1)
    started = threading.Event()
    release = threading.Event()

    def _occupy(zone: str, year: int) -> None:
        started.set()
        release.wait(timeout=10.0)

    adapter._run = _occupy  # the one worker is busy from here on
    try:
        for i in range(1, 6):
            adapter.start(f"{i:02d}N", 2024)
        assert started.wait(timeout=5.0), "the first ingest never started"

        cancelled = adapter.cancel_unstarted()

        assert cancelled == 4, f"the four queued ingests must be cancelled, got {cancelled}"
        assert ("01N", 2024) in adapter._futures, "a RUNNING ingest must not be cancelled"
        assert not adapter._futures[("01N", 2024)].cancelled()
        # Forgotten, so a retry's `start` creates a fresh attempt rather than re-reading a
        # cancelled future.
        assert ("05N", 2024) not in adapter._futures
    finally:
        release.set()
        adapter.shutdown()


def test_cancel_unstarted_can_be_restricted_to_named_cells():
    """The retained-failure cap must be able to cancel ONLY what it refused.

    It refuses the never-attempted cells while the run keeps retrying others, and a retry's
    re-ingest sits in this same queue — so a blanket cancel would strand the retry the standing
    fleet exists to serve. A key with nothing queued (a running or already-finished cell, or one
    this adapter never saw) is a no-op rather than an error.
    """
    adapter = _adapter(max_parallel=1)
    started = threading.Event()
    release = threading.Event()

    def _occupy(zone: str, year: int) -> None:
        started.set()
        release.wait(timeout=10.0)

    adapter._run = _occupy  # the one worker is busy from here on
    try:
        for i in range(1, 6):
            adapter.start(f"{i:02d}N", 2024)
        assert started.wait(timeout=5.0), "the first ingest never started"

        cancelled = adapter.cancel_unstarted([("03N", 2024), ("04N", 2024), ("99N", 2024)])

        assert cancelled == 2, f"only the two named queued cells may be cancelled, got {cancelled}"
        for kept in (("01N", 2024), ("02N", 2024), ("05N", 2024)):
            assert kept in adapter._futures, f"{kept} was not named and must be left queued"
            assert not adapter._futures[kept].cancelled()
        assert ("03N", 2024) not in adapter._futures
        assert ("04N", 2024) not in adapter._futures
    finally:
        release.set()
        adapter.shutdown()


def test_the_priming_abort_quorum_is_the_first_wave():
    """Abort when the opening pool has all failed — not the whole list, not a running count.

    The list would hold a doomed cluster through wave after wave. A running count is the
    subtler trap: the pool starts a queued cell as soon as a worker frees, so fast failures
    accumulate across waves while a slow first-wave cell is still running and about to
    succeed, and aborting there cancels it.
    """
    # (a) first wave all failed, four cells still ingesting -> raise (pre-change: None).
    adapter = _adapter(max_parallel=2)
    try:
        adapter._futures = {
            ("01N", 2024): _settled(RuntimeError("bad credentials")),
            ("02N", 2024): _settled(RuntimeError("bad credentials")),
            **{(f"{i:02d}N", 2024): Future() for i in range(3, 7)},
        }
        with pytest.raises(RuntimeError, match=r"ingest\(s\) failed with none landing") as caught:
            adapter.wait_first([(f"{i:02d}N", 2024) for i in range(1, 7)], timeout=2.0)
        assert "01N-2024" in str(caught.value) and "02N-2024" in str(caught.value)
        assert "03N-2024" not in str(caught.value), "a cell that never settled is not a failure"
        # Raising rather than nominating a failure as "landed" keeps the GPUs down; the ingest
        # error is chained so the cause is not lost.
        assert "bad credentials" in str(caught.value.__cause__), "the ingest error must be chained"
    finally:
        adapter.shutdown()

    # (b) a second-wave failure must NOT complete the quorum while 02N can still land.
    adapter = _adapter(max_parallel=2)
    slow_but_good: Future = Future()
    try:
        adapter._futures = {
            ("01N", 2024): _settled(RuntimeError("fast failure, first wave")),
            ("02N", 2024): slow_but_good,
            ("03N", 2024): _settled(RuntimeError("fast failure, second wave")),
            ("04N", 2024): Future(),
        }
        threading.Timer(0.3, lambda: slow_but_good.set_result(None)).start()
        got = adapter.wait_first([(f"{i:02d}N", 2024) for i in range(1, 5)], timeout=5.0)
        assert got == ("02N", 2024), f"a still-running first-wave cell must be allowed to land, got {got}"
    finally:
        adapter.shutdown()


def test_the_all_failed_report_names_cells_that_failed_in_earlier_waits():
    """A fleet-wide failure must not report as one cell.

    `wait_first` drains its futures over SEVERAL waits — `FIRST_COMPLETED` returns as soon
    as any one settles — so the final wait's `done` holds only whatever finished last. The
    error was built from that slice, so a window where every ingest failed for one shared
    reason (bad credentials, an unreachable API) named the single cell that happened to
    finish last. An operator then goes looking at that zone instead of at the credential.

    The two cells here settle in DIFFERENT waits, which the sibling test above cannot show
    because both of its futures are already settled when the first wait runs.
    """
    adapter = _adapter()
    try:
        late: Future = Future()
        adapter._futures = {
            ("01N", 2024): _settled(RuntimeError("bad credentials")),  # settles in wait #1
            ("02N", 2024): late,  # settles in wait #2
        }
        threading.Timer(0.05, lambda: late.set_exception(RuntimeError("bad credentials"))).start()

        with pytest.raises(RuntimeError, match="ingest\\(s\\) failed with none landing") as caught:
            adapter.wait_first([("01N", 2024), ("02N", 2024)])

        message = str(caught.value)
        assert "02N-2024" in message, "the last cell to settle must be named"
        assert "01N-2024" in message, "the cell that failed in an EARLIER wait must be named too"
    finally:
        adapter.shutdown()


@pytest.mark.parametrize(
    ("settle", "reason"),
    [
        (None, "limit reached"),
        (lambda fut: fut.set_result(None), "every ingest finished"),
        # A failed ingest is finished too: nothing more will arrive from it.
        (lambda fut: fut.set_exception(RuntimeError("bad credentials")), "every ingest finished"),
    ],
)
def test_the_backlog_hold_ends_at_its_limit_or_once_nothing_more_can_land(caplog, settle, reason):
    """The hold waits on the ingests themselves, so it ends the moment the last one settles
    rather than sitting out the rest of a limit that can no longer buy a mosaic.
    """
    adapter = _adapter()
    late: Future = Future()
    adapter._futures = {("01N", 2024): _settled(None), ("02N", 2024): late}
    if settle is not None:
        threading.Timer(0.05, settle, args=(late,)).start()
    limit_s = 0.3 if settle is None else 60.0
    t0 = time.monotonic()
    try:
        with caplog.at_level(logging.INFO, logger="test-adapter"):
            run_under_deadline(10.0, lambda: adapter.hold_for_backlog([("01N", 2024), ("02N", 2024)], limit_s / 60))
    finally:
        adapter.shutdown()
    elapsed = time.monotonic() - t0
    assert "1 of 2 mosaic(s) landed" in caplog.text  # the opening line counts the first landing
    assert f"({reason})" in caplog.text
    assert (limit_s * 0.9 <= elapsed) if settle is None else (elapsed < 5.0)


def test_shutdown_waits_for_the_children_it_cancelled(monkeypatch):
    """Teardown must not return while a child is still writing.

    ``discard`` can refuse its own retry when a child will not confirm. Teardown cannot:
    the run that races this child is a WHOLE NEW PARENT, deriving its own child tag from
    its own flow-run id, so it can neither find this child nor be told about it. This
    wait is the only place the two are kept apart — and those commits do not rebase, so
    a collision is terminal for one of them.
    """
    monkeypatch.setattr(mod, "_INGEST_POLL_S", 0.01)
    monkeypatch.setattr(mod, "CANCELLATION_CONFIRM_S", 5.0)
    monkeypatch.setattr(
        mod, "run_deployment", lambda dep, parameters, timeout, tags=None: SimpleNamespace(id="fr-slow", state=None)
    )
    # RUNNING while the poller holds it, then terminal once cancellation lands.
    client = _FakeClient([_state(StateType.RUNNING, final=False)] * 4 + [_state(StateType.CANCELLED, final=True)])
    monkeypatch.setattr(mod, "get_client", lambda sync_client=True: client)
    adapter = _adapter()
    adapter.start("33N", 2025)
    _await_inflight(adapter, {("33N", 2025): "fr-slow"})

    adapter.shutdown()
    assert ("fr-slow", StateType.CANCELLING) in client.cancelled
    # It read the run back after cancelling — the wait happened, rather than the request
    # being sent and the method returning.
    assert client.reads_after_cancel > 0


def test_shutdown_gives_up_on_an_unreachable_api_rather_than_holding_teardown(monkeypatch):
    """An API that is not answering is the case where confirmation is impossible rather
    than slow, and a teardown whose job is to stop costing money must not sit out the
    whole window re-asking it.
    """
    monkeypatch.setattr(mod, "_INGEST_POLL_S", 0.001)
    monkeypatch.setattr(mod, "_INGEST_POLL_MAX_ERRORS", 3)
    monkeypatch.setattr(mod, "CANCELLATION_CONFIRM_S", 60.0)  # long: the error budget must end it
    monkeypatch.setattr(
        mod, "run_deployment", lambda dep, parameters, timeout, tags=None: SimpleNamespace(id="fr-dead", state=None)
    )

    class _DeadAfterCancel(_FakeClient):
        def read_flow_run(self, fr_id):
            if self.cancelled:
                raise ConnectionError("prefect api down")
            return SimpleNamespace(id=fr_id, state=_state(StateType.RUNNING, final=False))

    client = _DeadAfterCancel([_state(StateType.RUNNING, final=False)])
    monkeypatch.setattr(mod, "get_client", lambda sync_client=True: client)
    adapter = _adapter()
    adapter.start("33N", 2025)
    _await_inflight(adapter, {("33N", 2025): "fr-dead"})

    started = time.monotonic()
    adapter.shutdown()
    assert time.monotonic() - started < 10.0, "the error budget, not the deadline, must end the wait"
