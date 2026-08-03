"""Unit tests for the AWS Dask provider's cluster-start retry policy.

These cover the retry *decision* — which cluster-start failures are transient
and worth retrying vs. which must fail fast — without provisioning real
Fargate. The predicate is the load-bearing piece (a misclassification either
burns retries on a permanent misconfiguration or gives up on a recoverable
ENI/control-plane race), so it is tested directly, and the tenacity policy it
feeds is exercised against a stub callable to confirm the wiring.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import pickle
import threading
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import psutil
import pytest
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_fixed

pytest.importorskip("dask_cloudprovider", reason="dask-cloudprovider not installed (AWS extras)")

from tessera_embeddings.config.ingest import IngestSettings

# The consumers of this provider's log contract, imported here so a change on
# either side fails in one place: the health-line parser (see
# test_emitted_line_parses_with_the_profiling_parser) and the log group the
# profiling CLIs query (see test_profiling_tools_target_the_providers_log_group).
from tessera_embeddings.profiling.ingest import DEFAULT_INGEST_LOG_GROUP
from tessera_embeddings.profiling.ingest.watch_scheduler import parse_health_line
from tessera_embeddings.providers.aws import dask as dask_mod
from tessera_embeddings.providers.aws.dask import (
    _RETRYABLE_CLUSTER_START_ERRORS,
    SchedulerResourceLogger,
    _is_retryable_cluster_start_error,
    maybe_performance_report,
)


class TestRetryablePredicate:
    """``_is_retryable_cluster_start_error`` classification."""

    @pytest.mark.parametrize("needle", _RETRYABLE_CLUSTER_START_ERRORS)
    def test_known_transient_errors_retry(self, needle: str) -> None:
        """Each documented transient substring is matched, wrapped as _start() wraps it."""
        exc = RuntimeError(f"Cluster failed to start: {needle} (xyz)")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_scheduler_failed_to_start_retries(self) -> None:
        """The ENI-exhaustion failure observed in the fan-out → merge handoff."""
        exc = RuntimeError("Cluster failed to start: Scheduler failed to start")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_eni_placement_failure_retries(self) -> None:
        """RunTask placement rejection: dask raises RuntimeError(response) with
        the whole run_task response, whose failure reason is ``RESOURCE:ENI``.
        """
        exc = RuntimeError("{'tasks': [], 'failures': [{'reason': 'RESOURCE:ENI'}]}")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_describe_tasks_race_retries(self) -> None:
        """The original describe_tasks read-after-write race still matches."""
        exc = RuntimeError("Cluster failed to start: not enough values to unpack (expected 1, got 0)")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_unrelated_runtime_error_does_not_retry(self) -> None:
        """A RuntimeError that isn't a known transient must fail fast."""
        exc = RuntimeError("Cluster failed to start: image pull failed (manifest unknown)")
        assert _is_retryable_cluster_start_error(exc) is False

    def test_non_runtime_error_does_not_retry(self) -> None:
        """Even a matching substring on a non-RuntimeError is not retried — the
        isinstance guard keeps the policy scoped to dask-cloudprovider's wrapper.
        """
        exc = ValueError("Scheduler failed to start")
        assert _is_retryable_cluster_start_error(exc) is False


class TestRetryPolicyWiring:
    """The tenacity policy built from the predicate retries/aborts as intended.

    Uses ``wait_fixed(0)`` to keep the test instant — the wait *schedule* is a
    config value asserted by reading it, not something to sit through here.
    """

    def _run(self, fn, *, max_attempts: int = 4):
        retrying = Retrying(
            retry=retry_if_exception(_is_retryable_cluster_start_error),
            stop=stop_after_attempt(max_attempts),
            wait=wait_fixed(0),
            reraise=True,
        )
        return retrying(fn)

    def test_retries_until_success(self) -> None:
        """A transient failure on early attempts is ridden out; the call succeeds."""
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Cluster failed to start: Scheduler failed to start")
            return "cluster"

        assert self._run(flaky) == "cluster"
        assert calls["n"] == 3  # failed twice, succeeded on the third

    def test_gives_up_after_max_attempts_and_reraises_original(self) -> None:
        """A persistently-transient failure exhausts attempts and re-raises the
        ORIGINAL RuntimeError (reraise=True), not a tenacity RetryError.
        """
        calls = {"n": 0}

        def always_transient():
            calls["n"] += 1
            raise RuntimeError("Cluster failed to start: Scheduler failed to start")

        with pytest.raises(RuntimeError, match="Scheduler failed to start"):
            self._run(always_transient, max_attempts=4)
        assert calls["n"] == 4  # 1 initial + 3 retries

    def test_non_transient_fails_immediately(self) -> None:
        """A non-matching error is not retried — it surfaces on the first attempt."""
        calls = {"n": 0}

        def misconfigured():
            calls["n"] += 1
            raise RuntimeError("Cluster failed to start: image pull failed")

        with pytest.raises(RuntimeError, match="image pull failed"):
            self._run(misconfigured)
        assert calls["n"] == 1  # no retries burned on a permanent failure


def test_profiling_tools_target_the_providers_log_group() -> None:
    """The tools must query the group this provider actually ships to.

    They can't import the provider for it — that would drag dask, distributed and
    dask-cloudprovider into a CLI that only needs boto3 — so the constant is
    duplicated by necessity and pinned here instead. Without this, changing the
    provider's group leaves the tools silently querying an empty one, which reads
    identically to "the run produced no logs".
    """
    assert DEFAULT_INGEST_LOG_GROUP == dask_mod.DEFAULT_CLOUDWATCH_LOG_GROUP


class _FakeMemoryState:
    """The ``MemoryState`` fields the fleet-memory aggregation sums, in bytes."""

    def __init__(self, process_gib: float, managed_gib: float, spilled_gib: float) -> None:
        self.process = int(process_gib * 1024**3)
        self.managed = int(managed_gib * 1024**3)
        self.spilled = int(spilled_gib * 1024**3)


class _FakeWorker:
    """Minimal WorkerState stand-in exposing only what the logger reads.

    ``memory`` is omitted entirely when None, so a test can reproduce a
    distributed version whose WorkerState lacks it.
    """

    def __init__(self, processing: int, memory: _FakeMemoryState | None = None) -> None:
        self.processing = list(range(processing))
        if memory is not None:
            self.memory = memory


class _FakeLoop:
    def __init__(self, t: float) -> None:
        self._t = t

    def time(self) -> float:
        return self._t


class _FakeScheduler:
    """Scheduler stand-in with the attributes ``_log_usage`` touches."""

    def __init__(
        self,
        *,
        workers_processing,
        tasks: int,
        unrunnable: int,
        now: float = 1000.0,
        worker_memory: list[_FakeMemoryState] | None = None,
    ) -> None:
        mems = worker_memory if worker_memory is not None else [None] * len(workers_processing)
        self.workers = {
            f"tcp://w{i}": _FakeWorker(p, m) for i, (p, m) in enumerate(zip(workers_processing, mems, strict=True))
        }
        self.tasks = dict.fromkeys(range(tasks))
        self.unrunnable = set(range(unrunnable))
        self.loop = _FakeLoop(now)


class TestSchedulerResourceLogger:
    """The scheduler health heartbeat plugin.

    ``_log_usage`` is exercised directly against a fake scheduler so the line
    contents are asserted deterministically, without standing up a cluster or
    sleeping for a PeriodicCallback tick.
    """

    def _emit(self, plugin: SchedulerResourceLogger, sched: _FakeScheduler, caplog) -> str:
        """Run one probe with the plugin bound to ``sched`` and return the line."""
        plugin._scheduler = sched
        with caplog.at_level(logging.INFO, logger="distributed.scheduler"):
            plugin._log_usage()
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("scheduler health:")]
        assert len(lines) == 1, f"expected one health line, got {lines}"
        return lines[0]

    def _make_started(self) -> SchedulerResourceLogger:
        """A logger past ``start()`` but with its PeriodicCallback torn down so
        no timer fires during the test.
        """
        # Sampling off: these tests assert the health line only, and a triggered
        # sampler would spawn a thread and add nondeterministic WARNING lines.
        plugin = SchedulerResourceLogger(interval_s=30.0, stack_sampling=False)

        plugin._proc = psutil.Process()
        plugin._proc.cpu_percent(None)
        plugin._mem_limit_bytes = 8 * 1024**3
        return plugin

    def test_emitted_line_parses_with_the_profiling_parser(self, caplog) -> None:
        """The heartbeat format and the tool that reads it must not drift apart.

        ``watch_scheduler``'s regex mirrors this plugin's format string by hand.
        On drift the parser matches nothing, ``--report`` finds zero samples, and
        the dossier reports "was the heartbeat plugin attached?" — a wrong
        diagnosis, and one otherwise only discoverable during a live at-scale run.
        So this parses a REAL emitted line rather than a copy of the format.
        """
        plugin = self._make_started()
        sched = _FakeScheduler(workers_processing=[3, 4], tasks=42, unrunnable=7)
        line = self._emit(plugin, sched, caplog)

        sample = parse_health_line(line)
        assert sample is not None, f"watch_scheduler's parser did not match an emitted line: {line!r}"
        # Every field the tool derives its alerts from must survive the round trip.
        assert sample["workers"] == 2
        assert sample["tasks"] == 42
        assert sample["processing"] == 7  # 3 + 4 in flight
        assert sample["no_worker"] == 7
        assert sample["mem"] == pytest.approx(100.0 * plugin._proc.memory_info().rss / (8 * 1024**3), abs=1.0)

    def test_line_reports_all_signals(self, caplog) -> None:
        """One probe emits CPU, memory, lag, fds/threads and the task/worker
        backlog — every field the operator diagnoses from.
        """
        plugin = self._make_started()
        sched = _FakeScheduler(workers_processing=[3, 5], tasks=42, unrunnable=7)
        line = self._emit(plugin, sched, caplog)

        assert "cpu=" in line
        assert "rss=" in line and "GiB" in line
        assert "mem=" in line
        assert "lag=" in line
        assert "fds=" in line
        assert "threads=" in line
        assert "workers=2" in line
        assert "tasks=42" in line
        assert "processing=8" in line  # 3 + 5 summed across workers
        assert "no-worker=7" in line

    def test_fleet_memory_is_summed_and_round_trips_through_the_parser(self, caplog) -> None:
        """Fleet totals are summed across workers; ``wmax`` is the hottest ONE.

        The hottest-worker figure is what distinguishes an evenly loaded fleet
        from one worker holding the graph — invisible in a sum alone.
        """
        plugin = self._make_started()
        sched = _FakeScheduler(
            workers_processing=[1, 1, 1],
            tasks=10,
            unrunnable=0,
            worker_memory=[
                _FakeMemoryState(process_gib=2.0, managed_gib=1.0, spilled_gib=0.0),
                _FakeMemoryState(process_gib=8.0, managed_gib=6.0, spilled_gib=4.0),
                _FakeMemoryState(process_gib=4.0, managed_gib=3.0, spilled_gib=1.5),
            ],
        )
        sample = parse_health_line(self._emit(plugin, sched, caplog))

        assert sample is not None
        assert sample["worker_mem_gib"] == pytest.approx(14.0)
        assert sample["worker_managed_gib"] == pytest.approx(10.0)
        assert sample["worker_spill_gib"] == pytest.approx(5.5)
        assert sample["worker_max_gib"] == pytest.approx(8.0)

    def test_an_idle_fleet_reports_measured_zeros(self, caplog) -> None:
        """No workers yet is 0.0, not unknown — the cluster is simply empty."""
        plugin = self._make_started()
        sched = _FakeScheduler(workers_processing=[], tasks=0, unrunnable=0)
        sample = parse_health_line(self._emit(plugin, sched, caplog))

        assert sample is not None
        assert sample["worker_spill_gib"] == 0.0
        assert sample["worker_mem_gib"] == 0.0

    def test_unreadable_worker_state_costs_only_the_fleet_fields(self, caplog) -> None:
        """A WorkerState without ``memory`` must not take the heartbeat with it.

        A distributed upgrade that moves worker memory should degrade the four
        fleet numbers to unknown, not blank the line the operator diagnoses from.
        """
        plugin = self._make_started()
        sched = _FakeScheduler(workers_processing=[3, 4], tasks=42, unrunnable=7)  # no memory attr
        line = self._emit(plugin, sched, caplog)
        sample = parse_health_line(line)

        assert sample is not None, f"the heartbeat was lost entirely: {line!r}"
        assert sample["workers"] == 2 and sample["tasks"] == 42 and sample["no_worker"] == 7
        assert sample["worker_spill_gib"] is None
        assert sample["worker_mem_gib"] is None

    def test_memory_percent_uses_limit(self, caplog) -> None:
        """mem% is RSS as a fraction of the configured container limit, so a
        tiny limit drives the percentage up for the same process.
        """
        plugin = self._make_started()
        plugin._mem_limit_bytes = 64 * 1024**2  # 64 MiB: any real RSS exceeds this
        sched = _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0)
        line = self._emit(plugin, sched, caplog)

        mem_token = next(t for t in line.split() if t.startswith("mem="))
        pct = float(mem_token.removeprefix("mem=").removesuffix("%"))
        assert pct > 100.0  # RSS far exceeds a 64 MiB ceiling

    def test_lag_is_zero_on_first_probe_then_tracks_overrun(self, caplog) -> None:
        """The first probe has no prior schedule to compare against (lag 0);
        the next probe measures how far past the expected fire time it ran.
        """
        plugin = self._make_started()
        # First probe at t=1000 — primes _expected_next_s = 1000 + 30.
        first = self._emit(plugin, _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0, now=1000.0), caplog)
        assert "lag=0.0s" in first

        caplog.clear()
        # Next probe fires 5s late: clock at 1035 vs expected 1030.
        second = self._emit(plugin, _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0, now=1035.0), caplog)
        assert "lag=5.0s" in second

    def test_probe_before_start_is_a_noop(self, caplog) -> None:
        """Defensive: a probe with no primed process (pre-``start``) logs nothing
        rather than raising.
        """
        plugin = SchedulerResourceLogger()
        with caplog.at_level(logging.INFO, logger="distributed.scheduler"):
            plugin._log_usage()
        assert not [r for r in caplog.records if r.getMessage().startswith("scheduler health:")]

    def test_before_close_stops_callback(self) -> None:
        """``before_close`` must stop the PeriodicCallback so a torn-down
        scheduler isn't left with a live timer.
        """
        plugin = SchedulerResourceLogger()

        class _FakeCallback:
            def __init__(self) -> None:
                self.stopped = False

            def stop(self) -> None:
                self.stopped = True

        cb = _FakeCallback()
        plugin._callback = cb
        asyncio.run(plugin.before_close())

        assert cb.stopped is True
        assert plugin._callback is None

    def test_plugin_is_picklable_for_register_plugin(self) -> None:
        """The plugin MUST survive serialization to the remote scheduler.

        ``Client.register_plugin`` pickles the instance to the scheduler process.
        A ``threading.Event`` built in ``__init__`` owns a ``_thread.lock`` that
        neither pickle nor cloudpickle can serialize, so registration would raise
        — and ``ecs_cluster`` catches registration errors as best-effort, meaning
        the failure is SILENT and the run emits no health lines at all. Hence the
        event is created in ``start()``, and this pins that.
        """
        plugin = SchedulerResourceLogger()
        for dumps in (pickle.dumps, cloudpickle.dumps):
            revived = pickle.loads(dumps(plugin))
            assert revived.name == SchedulerResourceLogger.name
            assert revived._sampler_active is None  # created in start(), not __init__

    def test_start_creates_the_sampler_event(self) -> None:
        """``start()`` runs in the scheduler process, so the Event is built there."""
        plugin = SchedulerResourceLogger(interval_s=30.0)
        sched = _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0)
        asyncio.run(plugin.start(sched))
        try:
            assert plugin._sampler_active is not None
            assert plugin._sampler_active.is_set() is False
        finally:
            asyncio.run(plugin.before_close())  # stop the PeriodicCallback

    def test_start_records_the_loop_thread_not_the_main_thread(self) -> None:
        """The sampler attributes the stall to the thread ``lag`` measures.

        ``start()`` is awaited ON the scheduler's event loop, so it records that
        thread's id. Driven from a NON-main thread on purpose: under pytest the
        loop thread and MainThread are the same, so a same-thread assertion would
        pass even if the id were never recorded at all — which is exactly the bug
        this replaces (inferring the loop thread as MainThread).
        """
        plugin = SchedulerResourceLogger(interval_s=30.0)
        sched = _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0)
        worker_tid: list[int] = []

        def _run_start() -> None:
            worker_tid.append(threading.get_ident())
            asyncio.run(plugin.start(sched))
            asyncio.run(plugin.before_close())

        t = threading.Thread(target=_run_start, name="fake-scheduler-loop")
        t.start()
        t.join(timeout=10)

        assert plugin._loop_tid == worker_tid[0]
        assert plugin._loop_tid != threading.main_thread().ident

    def test_sampler_falls_back_to_main_thread_before_start(self, caplog) -> None:
        """A probe driven without ``start()`` still samples (tests do this)."""
        plugin = SchedulerResourceLogger(stack_samples=1, stack_sample_gap_s=0.0)
        plugin._sampler_active = threading.Event()
        plugin._sampler_active.set()
        assert plugin._loop_tid is None
        with caplog.at_level(logging.WARNING, logger="distributed.scheduler"):
            plugin._sample_stacks_worker(95.0, 4.0)
        assert any(r.getMessage().startswith("scheduler stack sample") for r in caplog.records)

    def test_stack_sampler_logs_and_clears(self, caplog) -> None:
        """The worker logs one collapsed stack tally with the cpu/lag context
        and clears the single-flight flag when done.
        """
        plugin = SchedulerResourceLogger(stack_samples=2, stack_sample_gap_s=0.0)
        # start() normally creates this in the scheduler process; stand it up here.
        plugin._sampler_active = threading.Event()
        plugin._sampler_active.set()
        with caplog.at_level(logging.WARNING, logger="distributed.scheduler"):
            plugin._sample_stacks_worker(95.0, 4.0)
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("scheduler stack sample")]
        assert len(lines) == 1
        assert "cpu=95% lag=4.0s, 2 samples" in lines[0]
        assert plugin._sampler_active.is_set() is False

    def test_stack_sampler_single_flight(self) -> None:
        """A second trigger while a sampler is active is a no-op."""
        plugin = SchedulerResourceLogger()
        started: list = []
        plugin._sample_stacks_worker = lambda *a: started.append(a)  # type: ignore[method-assign]
        plugin._sampler_active = threading.Event()
        plugin._sampler_active.set()  # pretend one is already running
        plugin._maybe_sample_stacks(99.0, 5.0)
        assert started == []

    def test_stack_sampler_triggers_when_idle(self) -> None:
        """When idle, a trigger spawns the worker off the event loop."""
        plugin = SchedulerResourceLogger()
        ran = threading.Event()

        def _fake(cpu: float, lag: float) -> None:
            ran.set()
            plugin._sampler_active.clear()

        plugin._sample_stacks_worker = _fake  # type: ignore[method-assign]
        plugin._maybe_sample_stacks(99.0, 5.0)
        assert ran.wait(timeout=2.0)

    def test_log_usage_calls_sampler_on_high_lag(self, caplog) -> None:
        """The health probe hands a duress reading (lag>=3s) to the sampler."""
        plugin = self._make_started()
        plugin._stack_sampling = True  # helper disables it; re-enable for this test
        calls: list = []
        plugin._maybe_sample_stacks = lambda cpu, lag: calls.append((cpu, lag))  # type: ignore[method-assign]
        self._emit(plugin, _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0, now=1000.0), caplog)
        caplog.clear()  # _emit asserts exactly one health line per call
        self._emit(plugin, _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0, now=1035.0), caplog)
        assert calls, "expected the sampler to be triggered by lag>=3s"
        assert calls[-1][1] == 5.0


@pytest.mark.integration
class TestSchedulerResourceLoggerOnCluster:
    """End-to-end: registering the plugin on a real (local) scheduler makes it
    log on its event loop. Guards the registration/start wiring that the unit
    tests stub out.
    """

    def test_registered_plugin_emits_on_real_scheduler(self, caplog) -> None:
        import time

        from distributed import Client, LocalCluster

        with (
            caplog.at_level(logging.INFO, logger="distributed.scheduler"),
            LocalCluster(n_workers=1, dashboard_address=None, processes=False) as cluster,
            Client(cluster) as client,
        ):
            client.register_plugin(
                SchedulerResourceLogger(interval_s=0.3),
                name=SchedulerResourceLogger.name,
            )
            client.gather(client.map(lambda x: x * x, range(50)))
            time.sleep(1.0)  # allow a couple of PeriodicCallback ticks

        health = [r.getMessage() for r in caplog.records if r.getMessage().startswith("scheduler health:")]
        assert health, "expected at least one health line from the running scheduler"
        assert "workers=1" in health[-1]


def test_ingest_settings_perf_report_uri_defaults_none() -> None:
    """The perf-report knob is off by default, so normal runs never capture one."""
    assert IngestSettings().perf_report_uri is None


def _report_stub(exit_error: Exception | None = None):
    """A performance_report stand-in, optionally failing on exit (the render step).

    It WRITES the file on a successful exit, because the real one does and the
    upload step reads it back. A stub that skipped that would make an upload test
    pass for the wrong reason: the missing-file error and the upload error are
    caught by the same handler, so the assertion would hold even with the upload
    never attempted.
    """

    @contextlib.contextmanager
    def _report(filename=None):
        yield
        if exit_error is not None:
            raise exit_error
        if filename:
            Path(filename).write_bytes(b"<html>report</html>")

    return _report


class _FakeFsspec:
    """Stand-in for the module-level ``fsspec`` in the provider.

    Substituted on the provider module's own attribute rather than on the real
    fsspec: ``dask.py`` binds the module at import time, so patching the library
    in place would leak into every other test in the session (the from-import
    binding trap, documented in the repo's architecture notes).
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.written: dict[str, bytes] = {}

    def open(self, uri, mode="rb"):
        if self.error is not None:
            raise self.error
        written = self.written
        sink = io.BytesIO()

        @contextlib.contextmanager
        def _cm():
            try:
                yield sink
            finally:
                written[uri] = sink.getvalue()

        return _cm()


class TestMaybePerformanceReport:
    """A probe-only diagnostics artifact must never affect the ingest it observes."""

    def test_noop_without_uri(self) -> None:
        """With no URI the helper is a pure pass-through — no client, no Dask calls."""
        ran = []
        with maybe_performance_report("tcp://unused:8786", None, logging.getLogger("t")):
            ran.append(True)
        assert ran == [True]

    def test_setup_failure_still_runs_the_body(self, monkeypatch, caplog) -> None:
        """If the diagnostic client can't connect, the ingest runs anyway.

        Otherwise an unreachable scheduler — or a typo'd probe URI — would stop
        the actual ingest from happening at all.
        """

        def _boom(*a, **k):
            raise OSError("cannot connect to scheduler")

        monkeypatch.setattr(dask_mod, "Client", _boom)

        ran = []
        with (
            caplog.at_level(logging.WARNING),
            maybe_performance_report("tcp://x:8786", "s3://b/p.html", logging.getLogger("t")),
        ):
            ran.append(True)
        assert ran == [True], "body must still execute when diagnostics setup fails"

    def test_render_failure_does_not_mask_the_body_exception(self, monkeypatch) -> None:
        """A report-render failure must not replace the ingest's own exception.

        The render happens in ``performance_report.__exit__``, so unguarded it
        would surface INSTEAD of the real ingest error and send debugging in
        entirely the wrong direction.
        """
        monkeypatch.setattr(dask_mod, "Client", lambda *a, **k: contextlib.nullcontext())
        monkeypatch.setattr(dask_mod, "performance_report", _report_stub(RuntimeError("render blew up")))

        with (
            pytest.raises(ValueError, match="the real ingest failure"),
            maybe_performance_report("tcp://x:8786", "s3://b/p.html", logging.getLogger("t")),
        ):
            raise ValueError("the real ingest failure")

    def test_render_failure_does_not_fail_a_successful_ingest(self, monkeypatch, caplog) -> None:
        """A render failure after a clean ingest is logged, not raised."""
        monkeypatch.setattr(dask_mod, "Client", lambda *a, **k: contextlib.nullcontext())
        monkeypatch.setattr(dask_mod, "performance_report", _report_stub(RuntimeError("render blew up")))

        ran = []
        with (
            caplog.at_level(logging.WARNING),
            maybe_performance_report("tcp://x:8786", "s3://b/p.html", logging.getLogger("t")),
        ):
            ran.append(True)
        assert ran == [True]

    def test_rendered_report_is_uploaded_to_the_uri(self, monkeypatch) -> None:
        """The happy path: the rendered HTML reaches the target URI."""
        fake = _FakeFsspec()
        monkeypatch.setattr(dask_mod, "Client", lambda *a, **k: contextlib.nullcontext())
        monkeypatch.setattr(dask_mod, "performance_report", _report_stub())
        monkeypatch.setattr(dask_mod, "fsspec", fake)

        with maybe_performance_report("tcp://x:8786", "s3://b/p.html", logging.getLogger("t")):
            pass
        assert fake.written == {"s3://b/p.html": b"<html>report</html>"}

    def test_upload_failure_does_not_fail_the_ingest(self, monkeypatch, caplog) -> None:
        """A denied PUT or missing bucket is logged, never raised.

        fsspec backends raise botocore ClientError / credential errors, none of
        which subclass OSError — so this is the third of the three containment
        paths, and the one an operator is most likely to hit first (a typo'd
        bucket, or a role without write access to the profiling prefix).
        """
        fake = _FakeFsspec(error=RuntimeError("AccessDenied on PutObject"))
        monkeypatch.setattr(dask_mod, "Client", lambda *a, **k: contextlib.nullcontext())
        monkeypatch.setattr(dask_mod, "performance_report", _report_stub())
        monkeypatch.setattr(dask_mod, "fsspec", fake)

        ran = []
        with (
            caplog.at_level(logging.WARNING),
            maybe_performance_report("tcp://x:8786", "s3://b/p.html", logging.getLogger("t")),
        ):
            ran.append(True)
        assert ran == [True]
        assert any("failed to upload" in r.getMessage() for r in caplog.records)


class TestStopEcsTasksByTag:
    """The ingest cluster's emergency teardown: sweep this run's tasks by tag.

    A hard-cancelled flow skips ecs_cluster's `finally`, leaving scheduler +
    workers running in ECS (the 03S leak: 23 workers + a scheduler). The flows
    tag every cluster resource with their flow-run id, so the hook finds the
    leak from nothing but the flow_run argument, in a fresh process.
    """

    class _FakeEcs:
        def __init__(self, tagged: dict[str, dict[str, str]]) -> None:
            self.tagged = tagged  # taskArn -> tags
            self.stopped: list[str] = []

        def get_paginator(self, name):
            arns = list(self.tagged)

            class _P:
                def paginate(self, cluster):
                    # two pages, to prove pagination is honoured
                    yield {"taskArns": arns[:1]}
                    yield {"taskArns": arns[1:]}

            return _P()

        def describe_tasks(self, cluster, tasks, include):
            assert include == ["TAGS"]
            return {
                "tasks": [
                    {"taskArn": a, "tags": [{"key": k, "value": v} for k, v in self.tagged[a].items()]} for a in tasks
                ]
            }

        def stop_task(self, cluster, task, reason):
            self.stopped.append(task)

    def _sweep(self, monkeypatch, tagged, *, arn="arn:aws:ecs:eu-west-1:1:cluster/test", **kw):
        fake = self._FakeEcs(tagged)
        monkeypatch.setenv("ECS_CLUSTER_ARN", arn)
        import boto3

        self.client_kwargs: dict = {}
        monkeypatch.setattr(boto3, "client", lambda name, **ck: (self.client_kwargs.update(ck), fake)[1])
        n = dask_mod.stop_ecs_tasks_by_tag("tessera-flow-run-id", "run-A", log=logging.getLogger("t"), **kw)
        return n, fake

    def test_client_is_built_in_the_cluster_arns_region(self, monkeypatch):
        """A cluster ARN carries its region, and a default-region client does not follow
        it — a cross-region sweep would list nothing and leave the fleet billing.
        """
        self._sweep(monkeypatch, {}, arn="arn:aws:ecs:eu-west-1:1:cluster/test")
        assert self.client_kwargs["region_name"] == "eu-west-1"

    def test_unparseable_arn_leaves_region_to_boto(self, monkeypatch):
        """Guessing a region would be worse than boto3's own resolution."""
        self._sweep(monkeypatch, {}, arn="arn:cluster/test")
        assert self.client_kwargs["region_name"] is None

    def test_stops_only_this_runs_tasks(self, monkeypatch):
        tagged = {
            "arn:task/sched-A": {"tessera-flow-run-id": "run-A"},
            "arn:task/worker-A": {"tessera-flow-run-id": "run-A", "createdBy": "dask-cloudprovider"},
            "arn:task/worker-B": {"tessera-flow-run-id": "run-B"},  # another run's cluster: untouched
            "arn:task/untagged": {},
        }
        n, fake = self._sweep(monkeypatch, tagged)
        assert n == 2
        assert sorted(fake.stopped) == ["arn:task/sched-A", "arn:task/worker-A"]

    def test_missing_cluster_arn_warns_and_returns_zero(self, monkeypatch, caplog):
        monkeypatch.delenv("ECS_CLUSTER_ARN", raising=False)
        with caplog.at_level(logging.WARNING):
            n = dask_mod.stop_ecs_tasks_by_tag("k", "v", log=logging.getLogger("t"))
        assert n == 0  # the hook must never mask the flow's own terminal state
        assert any("ECS_CLUSTER_ARN" in r.getMessage() for r in caplog.records)


class TestDashboardSsmCommand:
    """The copy-pasteable port-forward command logged at cluster start."""

    def _cluster(self, task_arn: str):
        scheduler = SimpleNamespace(
            task_arn=task_arn,
            task={
                "containers": [{"name": "sidecar", "runtimeId": "nope"}, {"name": "dask-scheduler", "runtimeId": "rt9"}]
            },
        )
        return SimpleNamespace(scheduler=scheduler)

    def test_region_comes_from_the_task_arn(self, caplog):
        """--region is the CLUSTER's region, not the caller's default.

        SSM resolves the target within one region, so a caller whose default
        differs gets "instance not found" without it. The ARN already carries
        the answer; --profile stays a placeholder because it belongs to the
        caller's credentials, not the cluster.
        """
        cluster = self._cluster("arn:aws:ecs:eu-central-1:123456789012:task/my-cluster/abc123")
        with caplog.at_level(logging.INFO):
            dask_mod.log_dashboard_ssm_command(logging.getLogger("t"), cluster)
        msg = caplog.records[-1].getMessage()
        assert "--region eu-central-1 \\\n  --profile <your-aws-profile>" in msg
        assert "--target ecs:my-cluster_abc123_rt9" in msg  # container looked up by name, not order

    def test_unexpected_task_shape_warns_without_raising(self, caplog):
        with caplog.at_level(logging.WARNING):
            dask_mod.log_dashboard_ssm_command(logging.getLogger("t"), SimpleNamespace(scheduler=None))
        assert "Could not build SSM dashboard command" in caplog.records[-1].getMessage()


# --- stable Dask task definitions ------------------------------------------------------


def _fargate_config(**overrides):
    from tessera_embeddings.providers.aws.dask import FargateConfig

    base = dict(
        cluster_arn="arn:cluster",
        image="img:dev",
        vpc="vpc-1",
        subnets=["subnet-1"],
        security_groups=["sg-1"],
        execution_role_arn="arn:exec",
        task_role_arn="arn:task",
        scheduler_cpu=4096,
        scheduler_mem=8192,
        worker_cpu=4096,
        worker_mem=16384,
        cloudwatch_logs_group="/ecs/tessera/dask",
    )
    return FargateConfig(**{**base, **overrides})


class TestStableDaskTaskDefinitions:
    """Pinned task definitions take RegisterTaskDefinition calls to zero.

    `dask_cloudprovider` registers a scheduler AND a worker definition per cluster
    unless given ARNs for both. At campaign width that burst trips the account-wide
    `RegisterTaskDefinition` rate limit — observed at 37 concurrent cells.
    """

    def test_unset_falls_back_to_dynamic_registration(self):
        """The historical path must survive, so this ships safely before the CDK does."""
        kw = _fargate_config().to_cluster_kwargs()
        assert "scheduler_task_definition_arn" not in kw
        assert "worker_task_definition_arn" not in kw

    def test_both_set_are_passed_through(self):
        kw = _fargate_config(
            scheduler_task_definition_arn="arn:sched", worker_task_definition_arn="arn:work"
        ).to_cluster_kwargs()
        assert kw["scheduler_task_definition_arn"] == "arn:sched"
        assert kw["worker_task_definition_arn"] == "arn:work"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scheduler_task_definition_arn": "arn:sched"},
            {"worker_task_definition_arn": "arn:work"},
        ],
        ids=["scheduler only", "worker only"],
    )
    def test_only_one_set_pins_neither(self, kwargs):
        """Both or neither, deliberately.

        The library takes one ARN per role and registers whichever it was not given, so
        pinning half would still leave one registration per cluster — the exact thing this
        removes. Half-pinning is therefore treated as not pinning, rather than as a
        partial win that quietly keeps the burst.
        """
        kw = _fargate_config(**kwargs).to_cluster_kwargs()
        assert "scheduler_task_definition_arn" not in kw
        assert "worker_task_definition_arn" not in kw

    def test_env_vars_are_read(self, monkeypatch):
        """The CDK injects these onto every runner family; match that contract."""
        from tessera_embeddings.providers.aws import dask as mod

        monkeypatch.setenv("DASK_SCHEDULER_TASK_DEFINITION_ARN", "arn:sched-env")
        monkeypatch.setenv("DASK_WORKER_TASK_DEFINITION_ARN", "arn:work-env")
        cfg = mod.get_fargate_config()
        assert cfg.scheduler_task_definition_arn == "arn:sched-env"
        assert cfg.worker_task_definition_arn == "arn:work-env"
