"""fill_zones_sequential_flow: triage, ordering, and the pre-Ray gates.

Exercises the flow body via ``.fn`` with every external touchpoint mocked
(mirroring test_fill_zone_year_flow), so no Prefect engine, Ray cluster, or
S3 runs. The load-bearing behaviors: cheap cells (retag / all-ocean) are
settled without a cluster, live cells are ordered largest-first with clamped
actor requests, the model guard fires before Ray, and the shared cluster is
provisioned once with the idle-timeout override.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import pytest
from prefect.states import StateType

import tessera_embeddings.orchestration.prefect.flows._child_runs as _child_runs
import tessera_embeddings.orchestration.prefect.flows.fill_zones_sequential as mod
from tessera_embeddings.config.paths import BucketPaths

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


@pytest.fixture()
def wired(monkeypatch):
    """Mock every external touchpoint; return a record of what the flow did."""
    rec: dict = {"no_cluster_fills": [], "ray_kwargs": None, "seq_kwargs": None, "order": []}
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-seq-flow"))
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.credentials.iam_icechunk_credentials", object(), raising=False
    )
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.ray.make_instance_terminator",
        lambda log: lambda iid: None,
        raising=False,
    )

    @contextmanager
    def fake_ray_cluster(log, **kwargs):
        rec["ray_kwargs"] = kwargs
        rec["order"].append("ray_up")
        yield None  # no resolved YAML → the hook state stays unset

    monkeypatch.setattr("tessera_embeddings.providers.aws.ray.ray_cluster", fake_ray_cluster, raising=False)

    # Default triage: nothing complete, everything on-axis, 10 live tiles.
    monkeypatch.setattr(mod, "zone_year_complete", lambda *a, **k: False)
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: True)
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda *a, **k: 10)

    # Default optical preflight: provisional PRESENT for every cell (the real one
    # probes a catalogue over the network). Preflight tests override it; the record
    # lets them assert whether, and for which cells, it ran.
    rec["preflights"] = []

    def fake_preflight(zone, window, **kwargs):
        rec["preflights"].append(zone)
        return SimpleNamespace(finding=mod.SourceFinding.PRESENT, reason="test default", probes=1)

    monkeypatch.setattr(mod, "preflight_optical_source", fake_preflight)

    def fake_no_cluster_fill(**kwargs):
        rec["no_cluster_fills"].append(kwargs)
        return {"zone": kwargs["zone"], "run_id": kwargs["run_id"]}

    monkeypatch.setattr(mod, "fill_zone_year", fake_no_cluster_fill)
    monkeypatch.setattr(mod, "_assert_seeded_model_matches", lambda *a, **k: rec["order"].append("model_guard"))
    # The chained path reads the depth rule off the store root, like the per-cell flow.
    monkeypatch.setattr(mod, "_optical_min_obs_from_store", lambda *a, **k: None)
    monkeypatch.setattr(mod, "build_inference_config", lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(mod, "resolve_s1_orbit", lambda *a, **k: "both")
    monkeypatch.setattr(mod, "check_time_window_coverage", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_staging_run_id", lambda z, y, **k: f"{z}-{y}-fingerprint")
    # Immutable code identity (AMI ID + tarball ETag) — mocked so tests make no
    # SSM/S3 call; resolved once per run and folded into each cell's run_id.
    monkeypatch.setattr(mod, "_resolve_code_identity", lambda *a: "ami=ami-test")
    # Same for the AMI pointer, which an unpinned (direct, non-campaign) run resolves.
    monkeypatch.setattr(mod, "_resolve_ami_id", lambda *a: "ami-resolved")

    def fake_sequential(**kwargs):
        rec["seq_kwargs"] = kwargs
        return {"cells": len(kwargs["cells"]), "succeeded": len(kwargs["cells"]), "failed": 0}

    monkeypatch.setattr(mod, "fill_zones_sequential", fake_sequential)

    # The per-cell validation dispatch. Stubbed at THIS module's name rather than at
    # `run_deployment`: the helper holds its own reference to that, so patching the ingest
    # adapter's copy below would leave a validation dispatch reaching a real Prefect API.
    # Recorded so a test can assert the dispatch is genuinely made per cell.
    rec["validations"] = []

    def fake_validation_dispatch(deployment, *, zone, year, summary, parameters, log, tag=None):
        rec["validations"].append(
            {
                "deployment": deployment,
                "zone": zone,
                "year": year,
                "summary": summary,
                "parameters": parameters,
                "tag": tag,
            }
        )
        return "validation-run"

    monkeypatch.setattr(mod, "dispatch_cell_validation", fake_validation_dispatch)

    # The ingest adapter is the REAL one (several tests assert on it), so its one
    # external call is stubbed instead of the class. Needed because the flow now
    # BLOCKS on the first cell's ingest before `ray up`: previously `start` merely
    # queued a worker whose failure nothing ever joined, so an ingest that could not
    # dispatch was invisible here. Immediately-terminal so priming does not stall.
    monkeypatch.setattr(mod, "_INGEST_POLL_S", 0.01)
    monkeypatch.setattr(
        mod,
        "run_deployment",
        lambda dep, parameters, timeout, tags=None: SimpleNamespace(
            id=f"fr-{parameters['zone']}-{parameters['year']}", state=None
        ),
    )
    monkeypatch.setattr(
        mod,
        "get_client",
        lambda sync_client=True: _FakeClient([_state(StateType.COMPLETED, final=True)]),
    )
    return rec


def _run(**overrides):
    """Invoke the flow, translating the readable `zones=`/`year=` form into `cells`.

    The flow takes `(zone, year)` pairs so a cluster can span campaign years. Most tests
    below are about something else entirely, so they keep saying `zones=[...]` and this
    does the conversion in one place; `cells=` is passed straight through for the tests
    that ARE about multi-year lists.
    """
    kwargs = {"paths": _PATHS, "ami_ssm_name": "ami"}
    if "cells" in overrides:
        kwargs["cells"] = overrides.pop("cells")
    else:
        year = overrides.pop("year", 2025)
        kwargs["cells"] = [(z, year) for z in overrides.pop("zones", ["33N"])]
    kwargs.update(overrides)
    return mod.fill_zones_sequential_flow.fn(**kwargs)


def test_retag_and_ocean_cells_never_touch_the_cluster(wired, monkeypatch):
    monkeypatch.setattr(mod, "zone_year_complete", lambda store, zone, year, **k: zone == "01N")
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda mask, zone, **k: 0)  # the rest: all-ocean
    result = _run(zones=["01N", "02N"])
    # Both cells settled by the no-cluster fill path with the right run_ids...
    run_ids = {f["zone"]: f["run_id"] for f in wired["no_cluster_fills"]}
    assert run_ids == {"01N": "01N-2025-retag", "02N": "02N-2025-empty"}
    # ...and no cluster was ever provisioned.
    assert wired["ray_kwargs"] is None and result["sequential"] is None
    assert result["retagged"] == 1 and result["empty"] == 1 and result["live"] == 0


class TestPerCellValidation:
    """Each cell is handed to the validation deployment as it lands, not at the end.

    Asserted on the ``assemble`` callable the runner is given, because that is where the
    timing lives: the runner calls it once per cell on the trailing thread while the next
    cell infers. A dispatch anywhere else in this flow would still pass a "was it
    dispatched" test while delivering every verdict hours late.
    """

    @staticmethod
    def _landed(monkeypatch, summary):
        monkeypatch.setattr(mod, "assemble_zone_year", lambda handoff, **k: summary)
        return SimpleNamespace(zone="33N", year=2025)

    def test_the_assemble_callable_validates_the_cell_it_just_landed(self, wired, monkeypatch):
        handoff = self._landed(monkeypatch, {"zone": "33N", "year": 2025, "tag": "zone-33N-2025"})
        _run(zones=["33N"], validation_deployment="validate-zone-year/validate-zone-year-x")
        prepared = SimpleNamespace(staging_base="s3://out/staging", input_coverage=None)
        assert wired["seq_kwargs"]["assemble"](handoff, prepared) == {
            "zone": "33N",
            "year": 2025,
            "tag": "zone-33N-2025",
        }, "the assembly summary must pass through untouched"
        (call,) = wired["validations"]
        assert call["deployment"] == "validate-zone-year/validate-zone-year-x"
        assert (call["zone"], call["year"]) == ("33N", 2025)
        assert call["parameters"]["paths"] == _PATHS.model_dump()

    def test_the_cell_is_identified_by_the_handoff_not_the_summary(self, wired, monkeypatch):
        """A summary missing its coordinates must not raise inside the assembly thread."""
        handoff = self._landed(monkeypatch, {"tag": "zone-33N-2025"})
        _run(zones=["33N"], validation_deployment="v/v")
        wired["seq_kwargs"]["assemble"](handoff, SimpleNamespace(staging_base="s3://out/staging", input_coverage=None))
        (call,) = wired["validations"]
        assert (call["zone"], call["year"]) == ("33N", 2025)

    def test_a_retagged_cell_is_validated_too(self, wired, monkeypatch):
        """Its data landed under an earlier run, which may never have been checked."""
        monkeypatch.setattr(mod, "zone_year_complete", lambda store, zone, year, **k: zone == "01N")
        monkeypatch.setattr(mod, "zone_live_tile_count", lambda mask, zone, **k: 0)
        _run(zones=["01N", "02N"], validation_deployment="v/v")
        # Both no-cluster cells reach the dispatch; declining the all-ocean one is the
        # helper's own decision (it is marked empty), which its unit tests cover.
        assert [(c["zone"], c["year"]) for c in wired["validations"]] == [("01N", 2025), ("02N", 2025)]

    def test_nothing_is_dispatched_when_no_validator_is_configured(self, wired, monkeypatch):
        """The default: the library names no consumer's flow, so the ref is None."""
        handoff = self._landed(monkeypatch, {"zone": "33N", "year": 2025, "tag": "t"})
        _run(zones=["33N"])
        wired["seq_kwargs"]["assemble"](handoff, SimpleNamespace(staging_base="s3://out/staging", input_coverage=None))
        assert [c["deployment"] for c in wired["validations"]] == [None]


def test_off_axis_year_fails_the_run(wired, monkeypatch):
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: False)
    with pytest.raises(ValueError, match="pre-allocated axis"):
        _run()


def _finding(finding):
    return SimpleNamespace(finding=finding, reason="scripted", probes=2)


def test_confirmed_absent_optical_refuses_the_cell_before_any_dispatch(wired, monkeypatch):
    """A refused cell is neither ingested nor given a fleet share — and is reported."""
    monkeypatch.setattr(
        mod,
        "preflight_optical_source",
        lambda zone, window, **k: _finding(
            mod.SourceFinding.CONFIRMED_ABSENT if zone == "01N" else mod.SourceFinding.PRESENT
        ),
    )
    result = _run(zones=["01N", "02N"])
    assert result["no_optical"] == [["01N", 2025]]
    assert result["live"] == 1
    assert [c.zone for c in wired["seq_kwargs"]["cells"]] == ["02N"]
    # Not settled through the no-cluster fill either: the cell is left unfilled.
    assert wired["no_cluster_fills"] == []


def test_inconclusive_preflight_passes_the_cell_through(wired, monkeypatch):
    """Refuse only on a POSITIVE finding: 'could not determine' must not lose the cell."""
    monkeypatch.setattr(
        mod, "preflight_optical_source", lambda zone, window, **k: _finding(mod.SourceFinding.INCONCLUSIVE)
    )
    result = _run(zones=["01N"])
    assert result["no_optical"] == []
    assert [c.zone for c in wired["seq_kwargs"]["cells"]] == ["01N"]


def test_every_cell_refused_means_no_cluster(wired, monkeypatch):
    monkeypatch.setattr(
        mod, "preflight_optical_source", lambda zone, window, **k: _finding(mod.SourceFinding.CONFIRMED_ABSENT)
    )
    result = _run(zones=["01N", "02N"])
    assert result["live"] == 0 and result["sequential"] is None
    assert wired["ray_kwargs"] is None


def test_preflight_skipped_for_upstream_mosaics(wired):
    """With ingest=False the pre-built mosaics are the source of truth, not the catalogue."""
    _run(ingest=False)
    assert wired["preflights"] == []


def test_preflight_runs_only_for_live_cells(wired, monkeypatch):
    """Retagged and all-ocean cells are settled before the catalogue is ever asked."""
    monkeypatch.setattr(mod, "zone_year_complete", lambda store, zone, year, **k: zone == "01N")
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda mask, zone, **k: 0 if zone == "02N" else 10)
    _run(zones=["01N", "02N", "03N"])
    assert wired["preflights"] == ["03N"]


def test_live_cells_ordered_by_clamped_fleet_desc(wired, monkeypatch):
    counts = {"01N": 3, "02N": 500, "03N": 12}
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda mask, zone, **k: counts[zone])
    _run(zones=["01N", "02N", "03N"], num_actors=20)
    cells = wired["seq_kwargs"]["cells"]
    assert [(c.zone, c.num_actors) for c in cells] == [("02N", 20), ("03N", 12), ("01N", 3)]


def test_model_guard_runs_before_ray(wired):
    _run()
    assert wired["order"] == ["model_guard", "ray_up"]


def test_model_mismatch_prevents_cluster(wired, monkeypatch):
    def guard(*a, **k):
        raise ValueError("seeded for a different encoder")

    monkeypatch.setattr(mod, "_assert_seeded_model_matches", guard)
    with pytest.raises(ValueError, match="different encoder"):
        _run()
    assert wired["ray_kwargs"] is None


def test_cluster_gets_idle_timeout_and_code_params(wired):
    _run(idle_timeout_minutes=15, code_bucket="cb", code_suffix="-branch")
    assert wired["ray_kwargs"]["idle_timeout_minutes"] == 15
    assert wired["ray_kwargs"]["code_bucket"] == "cb"
    assert wired["ray_kwargs"]["code_suffix"] == "-branch"


def test_pinned_ami_id_reaches_ray_cluster(wired):
    """A campaign-pinned ami_id is threaded into the shared cluster's ray_cluster,
    so the fleet boots the exact image this flow's staging fingerprint recorded.
    """
    _run(ami_id="ami-pinned-01")
    assert wired["ray_kwargs"]["ami_id"] == "ami-pinned-01"


def test_unpinned_ami_is_resolved_once_and_reused(wired, monkeypatch):
    """Called direct (no campaign pin), the SSM pointer is read once.

    Resolving separately for the fingerprint and for ray_cluster would let a re-bake
    between the two boot an image the staging prefix was not fingerprinted against.
    """
    calls: list = []
    monkeypatch.setattr(mod, "_resolve_ami_id", lambda *a: calls.append(a) or "ami-resolved")
    _run(ami_id=None)
    assert len(calls) == 1
    assert wired["ray_kwargs"]["ami_id"] == "ami-resolved"


def test_ingest_false_passes_no_inputs_adapter(wired):
    _run(ingest=False)
    assert wired["seq_kwargs"]["inputs"] is None


def test_ingest_true_wires_the_deployment_adapter(wired):
    _run(ingest=True, look_ahead=3)
    inputs = wired["seq_kwargs"]["inputs"]
    assert isinstance(inputs, mod._DeploymentCellInputs)
    assert wired["seq_kwargs"]["look_ahead"] == 3
    inputs.shutdown()


def test_prepare_fingerprints_after_resolving_orbit(wired):
    _run()
    prepare = wired["seq_kwargs"]["prepare"]
    prep = prepare(SimpleNamespace(zone="33N", year=2025, num_actors=5))
    assert prep.run_id == "33N-2025-fingerprint"
    assert prep.mosaic_base == "s3://in/mosaics/33N/2025"
    assert prep.staging_base == "s3://out/staging/33N/2025"
    # The per-cell config carries the resolved orbit and the shard chunk size.
    assert prep.config.s1_orbit == "both"


def test_duplicate_and_lowercase_zones_are_canonicalized(wired):
    _run(zones=["33n", "33N"])
    assert [c.zone for c in wired["seq_kwargs"]["cells"]] == ["33N"]


def test_stream_contract_wired(wired):
    """The runner receives the stream contract: plan/session/infer_single +
    the session orbit the feeder gates zone deferral on.
    """
    _run(s1_orbit="both")
    kw = wired["seq_kwargs"]
    assert kw["session_s1_orbit"] == "both"
    assert callable(kw["plan"]) and callable(kw["session"]) and callable(kw["infer_single"])
    assert "infer" not in kw and "max_consecutive_failures" not in kw


# ---------------------------------------------------------------------------
# Pipeline priming + session sizing
# ---------------------------------------------------------------------------


class _RecordingInputs:
    """Records the priming sequence: which cells start, which are waited on.

    ``wait_first`` returns the SMALLEST-numbered zone of those offered, standing in
    for "whichever mosaic lands first" — the real adapter resolves that from its
    futures, and in the campaign the smallest zone is the one that finishes first.
    """

    def __init__(self, order: list, **kwargs):
        self.kwargs = kwargs
        self._order = order

    def start(self, zone, year):
        self._order.append(f"start:{zone}")

    def wait(self, zone, year, stop=None):
        self._order.append(f"wait:{zone}")

    def ready(self, zone, year):
        return True

    def wait_first(self, cells, timeout=None):
        self._order.append("wait_first:" + ",".join(z for z, _ in cells))
        return cells[-1] if cells else None

    def shutdown(self):
        self._order.append("inputs_shutdown")


def test_every_cell_ingests_before_ray_up(wired, monkeypatch):
    """EVERY cell is kicked off before the cluster, not a `1 + look_ahead` window.

    The window was the bug: a cell outside it had no ingest running, and could not get one
    until the feeder admitted a cell, which waited on an assembly. `1 + look_ahead` remains
    the ingest driver's CONCURRENCY — how many run at once — but no longer decides how many
    are asked for.
    """
    monkeypatch.setattr(mod, "_DeploymentCellInputs", partial(_RecordingInputs, wired["order"]))
    _run(zones=["01N", "02N", "03N"], ingest=True, look_ahead=1)
    order = wired["order"]
    assert [e for e in order if e.startswith("start:")] == ["start:01N", "start:02N", "start:03N"]
    assert order.index("start:03N") < order.index("ray_up"), "GPUs must not be asked for first"


def test_gpus_wait_for_whichever_mosaic_lands_first(wired, monkeypatch):
    """Not for a NAMED zone — which would mean the densest, the slowest to ingest.

    A cluster's opening window spans roughly 4 h to 10 h of ingest on the real
    coverage counts, so blocking on the head idles the fleet for about six hours
    with finished mosaics already on disk. The whole window is offered and the
    first to land wins.
    """
    monkeypatch.setattr(mod, "_DeploymentCellInputs", partial(_RecordingInputs, wired["order"]))
    _run(zones=["01N", "02N", "03N"], ingest=True, look_ahead=1)
    order = wired["order"]
    # Every started zone is offered — now the whole cell list, not a window — and no single
    # zone is waited on by name.
    assert "wait_first:01N,02N,03N" in order
    assert not [e for e in order if e.startswith("wait:")]
    assert order.index("wait_first:01N,02N,03N") < order.index("ray_up")


def test_cells_are_ordered_by_true_tile_count_not_the_clamped_fleet_request(wired, monkeypatch):
    """Densest first, on the UNCLAMPED count.

    `num_actors` is `min(num_actors, n_tiles)`, so every zone bigger than the fleet
    collapses to the same value — sorting on it leaves the whole dense end of the
    list in arbitrary order, which is exactly the part that decides which zone the
    fleet opens on.
    """
    counts = {"01N": 900, "02N": 100, "03N": 500}
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda mask, zone, **k: counts[zone])
    _run(zones=["01N", "02N", "03N"], num_actors=20)
    cells = wired["seq_kwargs"]["cells"]
    assert [c.zone for c in cells] == ["01N", "03N", "02N"]
    # ...and all three clamp to the same actor request, which is what made the old
    # sort key unable to tell 900 tiles from 500.
    assert [c.num_actors for c in cells] == [20, 20, 20]


def test_no_ingest_means_no_wait_before_the_cluster(wired, monkeypatch):
    """With mosaics supplied upstream there is nothing to wait for, and blocking
    on a cell this run never started would hang forever.
    """
    monkeypatch.setattr(mod, "_DeploymentCellInputs", partial(_RecordingInputs, wired["order"]))
    _run(zones=["01N", "02N"], ingest=False)
    assert not [e for e in wired["order"] if e.startswith("wait:")]
    assert "ray_up" in wired["order"]


def test_ingest_false_skips_priming(wired, monkeypatch):
    monkeypatch.setattr(
        mod, "_DeploymentCellInputs", lambda **k: (_ for _ in ()).throw(AssertionError("no adapter expected"))
    )
    _run(ingest=False)
    assert not [e for e in wired["order"] if e.startswith("start:")]


def test_session_sized_to_largest_cell_not_fleet_ceiling(wired, monkeypatch):
    """The shared session's actor request is the LARGEST cell's clamped request:
    a small shard must not ask AWS for the full default fleet before any
    streamed work arrives.
    """
    counts = {"01N": 3, "02N": 5}
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda mask, zone, **k: counts[zone])
    captured: dict = {}
    monkeypatch.setattr(mod, "run_inference", lambda n, *a, **k: captured.update(n=n) or [])
    _run(zones=["01N", "02N"], num_actors=20)
    wired["seq_kwargs"]["session"](lambda: None, lambda item, result: None)
    assert captured["n"] == 5  # max cell request, not 20


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


def test_adapter_wait_polls_child_run_to_completion(monkeypatch):
    monkeypatch.setattr(mod, "_INGEST_POLL_S", 0.01)
    monkeypatch.setattr(
        mod, "run_deployment", lambda dep, parameters, timeout, tags=None: SimpleNamespace(id="fr-1", state=None)
    )
    client = _FakeClient([_state(StateType.RUNNING, final=False), _state(StateType.COMPLETED, final=True)])
    monkeypatch.setattr(mod, "get_client", lambda sync_client=True: client)
    adapter = _adapter()
    try:
        adapter.wait("33N", 2025)  # returns (no raise) once the child completes
        assert adapter._inflight == {}  # deregistered after the poll
    finally:
        adapter.shutdown()
    assert client.cancelled == []  # nothing in flight at shutdown


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
    threading.Timer(0.3, stop.set).start()
    try:
        with pytest.raises(RuntimeError, match="aborted: runner stopping"):
            adapter.wait("33N", 2025, stop=stop)
        assert time.monotonic() - t0 < 30
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# Shutdown-race hardening + the cancellation-hook child sweep
# ---------------------------------------------------------------------------


def test_adapter_refuses_dispatch_after_shutdown(monkeypatch):
    """A worker that reaches its dispatch AFTER shutdown began must NOT create a
    child run — a run created after the cancellation sweep is the one orphan the
    sweep can't see. Exercised on _run directly: this is the race where the
    worker was submitted before shutdown but starts executing after (a fresh
    start() would already be rejected by the shut-down executor).
    """
    calls: list = []
    monkeypatch.setattr(mod, "run_deployment", lambda *a, **k: calls.append(1))
    adapter = _adapter()
    adapter._stopping.set()  # shutdown has begun; this worker thread is late
    try:
        with pytest.raises(RuntimeError, match="not dispatched"):
            adapter._run("33N", 2025)
        assert calls == []  # no child was ever created
    finally:
        adapter.shutdown()


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


def test_ingest_child_tag_derivation():
    assert mod._ingest_child_tag("abc-123") == "chained-ingest:abc-123"
    assert mod._ingest_child_tag(None) is None


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


def test_flow_registers_child_ingest_sweep_on_cancel_and_crash():
    flow = mod.fill_zones_sequential_flow
    assert mod._cancel_child_ingests_on_cancellation in flow.on_cancellation_hooks
    assert mod._cancel_child_ingests_on_cancellation in flow.on_crashed_hooks


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


def test_negative_look_ahead_rejected_before_any_side_effect(wired):
    """look_ahead < 0 is rejected before triage / priming / `ray up` — a
    deadlock that only manifested after the GPU cluster was provisioned.
    """
    with pytest.raises(ValueError, match="look_ahead must be >= 0"):
        _run(look_ahead=-2)
    assert wired["ray_kwargs"] is None  # no cluster provisioned
    assert wired["seq_kwargs"] is None  # runner never entered


def test_session_orbit_is_the_request(wired):
    """The session orbit is simply the request — a whole UTM zone always carries
    both orbits, so resolving it from the cells' data would only matter for a
    single-orbit whole zone, which does not occur (sub-zone/pixel single-orbit
    is handled per-pixel, not here).
    """
    _run(s1_orbit="both")
    assert wired["seq_kwargs"]["session_s1_orbit"] == "both"


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


def test_wait_first_raises_rather_than_booting_gpus_when_every_cell_failed():
    """No mosaic is coming, so the one thing that must not happen is `ray up`.

    Nominating one of the failures as "landed" sent the caller into a five-to-ten-minute
    billed GPU bringup for a mosaic that does not exist, torn down again as soon as the
    feeder reached the same failure. Raising costs nothing: the ingest error is chained as
    the cause, and priming runs inside the flow's shutdown guard so the children are still
    cancelled. Returning ``None`` would be worse still — it reads as a timeout.
    """
    adapter = _adapter()
    try:
        adapter._futures = {
            ("01N", 2024): _settled(RuntimeError("bad credentials")),
            ("02N", 2024): _settled(RuntimeError("bad credentials")),
        }
        with pytest.raises(RuntimeError, match="every ingest in the opening window failed") as caught:
            adapter.wait_first([("01N", 2024), ("02N", 2024)])
        assert "01N-2024" in str(caught.value) and "02N-2024" in str(caught.value)
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "bad credentials" in str(caught.value.__cause__), "the ingest error must be chained"
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

        with pytest.raises(RuntimeError, match="every ingest in the opening window failed") as caught:
            adapter.wait_first([("01N", 2024), ("02N", 2024)])

        message = str(caught.value)
        assert "02N-2024" in message, "the last cell to settle must be named"
        assert "01N-2024" in message, "the cell that failed in an EARLIER wait must be named too"
    finally:
        adapter.shutdown()


def test_a_failure_while_priming_still_cancels_the_ingests_it_started(wired, monkeypatch):
    """Priming lives inside the shutdown guard, so no child ingest is ever orphaned.

    ``start`` submits the whole window at once, so from the first submission there are
    child runs that only ``shutdown()`` can cancel. With priming above the ``try``, a
    failure between the first submission and the cluster — a raising ``start``, an
    unexpected error in the wait — returned without cancelling them, leaving children
    writing mosaic prefixes server-side that a prompt retry would then race.
    """

    class _FailingInputs(_RecordingInputs):
        def wait_first(self, cells, timeout=None):
            self._order.append("wait_first:raised")
            raise RuntimeError("prefect API unreachable")

    monkeypatch.setattr(mod, "_DeploymentCellInputs", partial(_FailingInputs, wired["order"]))
    with pytest.raises(RuntimeError, match="prefect API unreachable"):
        _run(zones=["01N", "02N", "03N"], ingest=True, look_ahead=1)

    order = wired["order"]
    assert "start:01N" in order, "the window was primed, so its children must be cancelled"
    assert "inputs_shutdown" in order, "shutdown never ran; the started ingests are orphans"
    assert "ray_up" not in order, "no cluster should be requested once the wait failed"


# --- multi-year cells ------------------------------------------------------------------


def test_a_cluster_can_span_years_and_each_cell_keeps_its_own_window(wired):
    """The point of the change: one cluster, cells from different campaign years.

    Each cell must carry ITS year's window to the actors. The actors are built once from
    the session config, so a cell read through that config would be inferred over another
    year's months — silently, since the session only checks `s1_orbit`.
    """
    _run(cells=[("33N", 2025), ("34N", 2024)])
    prepare = wired["seq_kwargs"]["prepare"]
    windows = {}
    for zone, year in (("33N", 2025), ("34N", 2024)):
        prep = prepare(SimpleNamespace(zone=zone, year=year, num_actors=5))
        windows[(zone, year)] = prep.config.time_window
    assert windows[("33N", 2025)] != windows[("34N", 2024)]
    assert windows[("33N", 2025)].months[-1] == (2025, 12)
    assert windows[("34N", 2024)].months[-1] == (2024, 12)


def test_the_same_zone_may_appear_for_several_years(wired):
    """Two years of one zone in one cluster — the case the year barrier forbade.

    Safe without any caller guarantee: assemblies serialize on the runner's single
    trailing thread, so the two years commit one after another rather than colliding on
    the zone group's attrs.
    """
    result = _run(cells=[("33N", 2025), ("33N", 2024)])
    assert result["years"] == [2024, 2025]
    assert result["live"] == 2


def test_an_exact_duplicate_cell_is_deduped_but_a_repeated_zone_is_not(wired):
    """Dedupe on the PAIR. A repeated zone across years is legitimate; an exact repeat
    would dispatch one cell twice.
    """
    result = _run(cells=[("33N", 2025), ("33n", 2025), ("33N", 2024)])
    assert result["live"] == 2, result
    assert result["years"] == [2024, 2025]


def test_a_literal_window_override_is_rejected_for_a_multi_year_list(wired):
    """One literal window cannot describe several years, so it must not be applied to all.

    Silently reusing 2025's window for a 2024 cell is precisely the failure this whole
    change exists to prevent, so the ambiguous request is refused instead.
    """
    with pytest.raises(ValueError, match="cannot describe several years"):
        _run(cells=[("33N", 2025), ("34N", 2024)], time_window_end="December 2025")
    # Still allowed for a single-year list, where it is unambiguous.
    _run(cells=[("33N", 2025)], time_window_end="December 2025")


def test_every_year_is_validated_before_any_ingest_is_dispatched(wired):
    """A bad year must fail while that is still free — not after the look-ahead ingests
    are away and the shared fleet is up.
    """
    with pytest.raises(ValueError):
        _run(cells=[("33N", 2025), ("34N", 2024)], time_window_end="June 2025")
    assert wired["ray_kwargs"] is None
    assert not wired.get("ingest_dispatches"), wired.get("ingest_dispatches")


def test_the_ingest_adapter_can_run_every_cell_the_flow_primes(wired):
    """The opening window is ``1 + look_ahead`` cells and the flow waits for whichever
    mosaic lands FIRST — which only means anything if they are all actually started.

    Sized at ``look_ahead`` alone, the executor had one fewer thread than the flow
    primed, so the siblings sat queued behind the head cell and the shared GPU fleet
    still waited on whichever zone happened to be first in the order.
    """
    _run(zones=["01N", "02N", "03N"], look_ahead=2)
    inputs = wired["seq_kwargs"]["inputs"]
    assert inputs._executor._max_workers == 3


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


# --- the inference pause reaches the runner --------------------------------------------


def test_the_pause_gate_reaches_the_runner_as_a_live_callable(wired, monkeypatch):
    """A parameter that is declared, documented and then not threaded through passes every
    other test in this file: the flow accepts it, the run succeeds, and inference can never be
    paused. So this asserts the runner actually receives a callable AND that the callable
    reads the named gate.
    """
    asked: list[str] = []
    monkeypatch.setattr(
        mod,
        "pause_signal",
        lambda name, **kw: lambda: bool(asked.append(name)),
    )
    _run(zones=["33N"], inference_pause_gate="tessera-global-inference")
    paused = wired["seq_kwargs"]["paused"]
    assert callable(paused), wired["seq_kwargs"].keys()
    paused()
    assert asked == ["tessera-global-inference"]


def test_no_pause_gate_passes_no_callable(wired):
    """Every non-campaign path — a fill dispatched by hand, a dev run — configures no gate,
    and must not acquire a per-poll check it never asked for.
    """
    _run(zones=["33N"])
    assert wired["seq_kwargs"]["paused"] is None


def test_the_chained_fill_applies_the_rule_its_store_advertises(wired, monkeypatch) -> None:
    """The DEFAULT campaign strategy was reading no depth rule at all.

    `fill_zone_year` resolves `optical_min_obs` from the store root, but this path built its
    config without it — so a store seeded with a minimum would have published pixels below
    its own advertised line, through the strategy the campaign actually runs.
    """
    built: list[dict] = []

    def record(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(mod, "build_inference_config", record)
    monkeypatch.setattr(mod, "_optical_min_obs_from_store", lambda *a, **k: 25)
    _run(zones=["33N"])
    assert built, "no InferenceConfig was built"
    assert {c.get("optical_min_obs") for c in built} == {25}, "the store's rule must reach every cell's config"


def test_a_store_declaring_no_rule_leaves_the_config_unconstrained(wired, monkeypatch) -> None:
    """Stores seeded before the rule existed carry no attr, and must keep filling as before."""
    built: list[dict] = []
    monkeypatch.setattr(mod, "build_inference_config", lambda **k: built.append(k) or SimpleNamespace(**k))
    monkeypatch.setattr(mod, "_optical_min_obs_from_store", lambda *a, **k: None)
    _run(zones=["33N"])
    assert built and {c.get("optical_min_obs") for c in built} == {None}
