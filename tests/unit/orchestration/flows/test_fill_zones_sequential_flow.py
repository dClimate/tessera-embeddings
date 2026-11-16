"""fill_zones_sequential_flow: triage, ordering, and the pre-Ray gates.

Exercises the flow body via ``.fn`` with every external touchpoint mocked
(mirroring test_fill_zone_year_flow), so no Prefect engine, Ray cluster, or
S3 runs. The load-bearing behaviors: cheap cells (retag / all-ocean) are
settled without a cluster, live cells are ordered largest-first with clamped
actor requests, the model guard fires before Ray, and the shared cluster is
provisioned once with the idle-timeout override. The ingest adapter the flow drives,
``_DeploymentCellInputs``, is tested in ``test_deployment_cell_inputs.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import pytest
from prefect.exceptions import TerminationSignal
from prefect.states import StateType

import tessera_embeddings.orchestration.prefect.flows.fill_zones_sequential as mod
from tessera_embeddings.config.inference import assert_run_id_matches_model
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.storage.update import UpdatePlan
from tests.unit.deadline import run_under_deadline
from tests.unit.orchestration.flows.test_deployment_cell_inputs import _FakeClient, _state

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")
_AMI = {"ami_ssm_name": "ami", "ami_ssm_name_by_model": {"v1.1": "ami"}}  # v1.1 needs its own image (ADR 025)


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
    monkeypatch.setattr(mod, "store_model_version", lambda *a, **k: rec["order"].append("model_guard") or "v1.1")
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
    monkeypatch.setattr(mod, "_resolve_env_image_uri", lambda *a: "s3://b/ray-env-images/env-resolved.img")

    def fake_sequential(**kwargs):
        rec["seq_kwargs"] = kwargs
        return {"cells": len(kwargs["cells"]), "succeeded": len(kwargs["cells"]), "failed": 0}

    monkeypatch.setattr(mod, "fill_zones_sequential", fake_sequential)

    # The per-cell validation dispatch. Stubbed at THIS module's name rather than at
    # `run_deployment`: the helper holds its own reference to that, so patching the ingest
    # adapter's copy below would leave a validation dispatch reaching a real Prefect API.
    # Recorded so a test can assert the dispatch is genuinely made per cell.
    rec["validations"] = []

    def fake_validation_dispatch(deployment, *, zone, year, summary, target, log, tag=None):
        rec["validations"].append(
            {
                "deployment": deployment,
                "zone": zone,
                "year": year,
                "summary": summary,
                "target": target,
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
    kwargs = {"paths": _PATHS, **_AMI}
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
    # ...with the registry, so planning can catch up the master of a fill that died before merging it.
    assert {f["registry_root"] for f in wired["no_cluster_fills"]} == {_PATHS.optical_registry(_PATHS.global_store())}
    # ...and no cluster was ever provisioned.
    assert wired["ray_kwargs"] is None and result["sequential"] is None
    assert result["retagged"] == 1 and result["empty"] == 1 and result["live"] == 0


def test_a_streamed_cell_is_planned_with_the_registry(wired, monkeypatch):
    """A cell the session finds already landed has its master caught up in planning, which needs the registry."""
    seen: dict = {}
    monkeypatch.setattr(mod, "plan_zone_inference", lambda **k: seen.update(k))
    _run(zones=["33N"])
    wired["seq_kwargs"]["plan"](
        SimpleNamespace(zone="33N", year=2025), SimpleNamespace(mosaic_base="m", config=None, run_id="r")
    )
    assert seen["registry_root"] == _PATHS.optical_registry(_PATHS.global_store())


class TestAnUpdate:
    """A targeted update triages by its own record and shards, and never retags or marks."""

    PLAN = UpdatePlan(
        name="u1",
        mode="shards",
        mask_name="alt",
        cells={"01N/2025": ["chunk_0_0"], "02N/2025": None, "03N/2025": None, "04N/2025": ["chunk_5_5"]},
    )

    @pytest.fixture()
    def update_wired(self, wired, monkeypatch):
        monkeypatch.setattr(mod, "load_update", lambda paths, name, mask_name: self.PLAN)
        monkeypatch.setattr(mod, "open_global_repo", lambda *a, **k: object())
        states = {
            ("01N", 2025): "pending",
            ("02N", 2025): "done",
            ("03N", 2025): "not_complete",
            ("04N", 2025): "pending",
        }
        monkeypatch.setattr(mod, "update_states", lambda repo, name, cells: {c: states[c] for c in cells})
        monkeypatch.setattr(mod, "zone_year_complete", lambda *a, **k: pytest.fail("an update must never retag"))
        wired["tiles"], wired["ingests"], wired["finished"], wired["tightening"] = {}, [], [], []
        monkeypatch.setattr(
            mod,
            "check_no_tightening",
            lambda repo, name, rule, cells: wired["tightening"].append(
                (name, rule, list(cells), len(wired["ingests"]))
            ),
        )
        monkeypatch.setattr(mod, "finish_update", lambda repo, plan: wired["finished"].append(plan.name) or "t")

        def count(mask, zone, **k):
            wired["tiles"][zone] = (mask, k["tiles"])
            return 0 if zone == "04N" else 3

        monkeypatch.setattr(mod, "zone_live_tile_count", count)
        real = mod.run_deployment

        def record(dep, parameters, timeout, tags=None):
            wired["ingests"].append(parameters)
            return real(dep, parameters, timeout, tags)

        monkeypatch.setattr(mod, "run_deployment", record)
        return wired

    def test_cells_come_from_the_plan_and_settle_by_its_record(self, update_wired):
        result = _run(cells=None, update="u1")

        root = _PATHS.update_root("u1")
        assert (result["update"], result["update_done"], result["not_filled"]) == ("u1", 1, [["03N", 2025]])
        assert result["live"] == 1 and result["empty"] == 1
        assert update_wired["tiles"]["01N"] == (_PATHS.land_mask_store("alt"), frozenset({"chunk_0_0"}))
        # The recorded cell goes through the runner too, so planning catches up its master; the
        # dead-target cell records itself there. Both with no cluster, as the update.
        recorded, dead = sorted(update_wired["no_cluster_fills"], key=lambda f: f["zone"])
        assert recorded["zone"] == "02N" and recorded["update"].name == "u1"
        assert recorded["registry_root"] == dead["registry_root"] == _PATHS.optical_registry(_PATHS.global_store())
        assert dead["update"].tiles == frozenset({"chunk_5_5"})
        assert dead["mosaic_base"] == f"{root}/mosaics/04N/2025"
        assert dead["staging_base"] == f"{_PATHS.update_staging('u1', _PATHS.global_store())}/04N/2025"
        # The live cell's ingest runs as the update, and its fill is planned as one.
        assert [p["update"] for p in update_wired["ingests"]] == ["u1"]
        assert update_wired["seq_kwargs"]["cells"][0].zone == "01N"
        # A one-cluster update pins its complete tag too, once every planned cell holds the record.
        assert update_wired["finished"] == ["u1"] and result["update_complete_tag"] == "t"

    def test_staging_is_per_destination_store(self, update_wired):
        # A holes update infers only its own store's holes, so a second store must never resume its tiles.
        for store in ("tessera", "other"):
            _run(cells=[("04N", 2025)], update="u1", store_name=store)
        first, second = (f["staging_base"] for f in update_wired["no_cluster_fills"])
        assert first != second
        assert second == f"{_PATHS.update_staging('u1', _PATHS.global_store('other'))}/04N/2025"

    def test_a_stricter_line_is_checked_before_any_ingest(self, update_wired, monkeypatch):
        """Only the cells still pending, at the run's own rule, and before a single ingest is dispatched."""
        monkeypatch.setattr(mod, "_optical_min_obs_from_store", lambda *a, **k: 15)
        _run(cells=None, update="u1")
        [(name, rule, cells, ingests_before)] = update_wired["tightening"]
        assert (name, rule, cells, ingests_before) == ("u1", 15, [("01N", 2025), ("04N", 2025)], 0)

    def test_a_cell_outside_the_plan_is_refused(self, update_wired):
        with pytest.raises(ValueError, match="not in update 'u1'"):
            _run(cells=[("09N", 2025)], update="u1")

    @pytest.mark.parametrize(("line", "applied"), [(None, 15), (1, 1)])
    def test_the_plans_own_line_reaches_the_config_and_the_fingerprint(self, update_wired, monkeypatch, line, applied):
        """The store root stays at 15: an update planned at 1 fills its cells at 1, and stages them
        apart from a run at 15; a plan without a line applies the store's.
        """
        plan = dataclasses.replace(self.PLAN, optical_min_obs=line)
        monkeypatch.setattr(mod, "load_update", lambda paths, name, mask_name: plan)
        monkeypatch.setattr(mod, "_optical_min_obs_from_store", lambda *a, **k: 15)
        fingerprinted: list[int] = []

        def fingerprint(zone, year, **k):
            fingerprinted.append(k["optical_min_obs"])
            return f"{zone}-{year}-fingerprint"

        monkeypatch.setattr(mod, "_staging_run_id", fingerprint)
        _run(cells=None, update="u1")
        prepare = update_wired["seq_kwargs"]["prepare"]
        assert prepare(SimpleNamespace(zone="01N", year=2025, num_actors=5)).config.optical_min_obs == applied
        assert set(fingerprinted) == {applied}


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
        assert call["target"].paths == _PATHS

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


def test_model_mismatch_prevents_cluster(wired, monkeypatch):
    def guard(*a, **k):
        raise ValueError("seeded for a different encoder")

    monkeypatch.setattr(mod, "store_model_version", guard)
    with pytest.raises(ValueError, match="different encoder"):
        _run()
    assert wired["ray_kwargs"] is None


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("ami_id", "ami-pinned-01"),
        ("env_image_uri", "s3://b/ray-env-images/env-pinned.img"),
        ("idle_timeout_minutes", 15),
        ("code_bucket", "cb"),
        ("code_suffix", "-branch"),
    ],
)
def test_campaign_pins_reach_ray_cluster(wired, kwarg, value):
    """A campaign-pinned AMI or environment image is threaded into the shared cluster's
    ray_cluster, so the fleet boots exactly what this flow's staging fingerprint recorded.
    """
    _run(**{kwarg: value})
    assert wired["ray_kwargs"][kwarg] == value


@pytest.mark.parametrize("model", ["v1.1", "v2-large"])
def test_the_stores_models_image_reaches_ray_cluster(wired, monkeypatch, model):
    """The cells run the model their store holds, so that model's entry is the image (ADR 025)."""
    monkeypatch.setattr(mod, "store_model_version", lambda *a, **k: model)
    _run(ami_ssm_name_by_model={"v1.1": "ami-v11", "v2-large": "ami-v2"})
    assert wired["ray_kwargs"]["ami_ssm_name"] == {"v1.1": "ami-v11", "v2-large": "ami-v2"}[model]


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


def test_prepare_fingerprints_after_resolving_orbit(wired):
    _run()
    prepare = wired["seq_kwargs"]["prepare"]
    prep = prepare(SimpleNamespace(zone="33N", year=2025, num_actors=5))
    assert prep.run_id == "33N-2025-fingerprint"
    assert prep.mosaic_base == "s3://in/mosaics/33N/2025"
    assert prep.staging_base == "s3://out/staging/33N/2025"
    # The per-cell config carries the resolved orbit and the shard chunk size.
    assert prep.config.s1_orbit == "both"


def test_stream_contract_wired(wired):
    """The runner receives the stream contract: plan/session + the session orbit. `infer_single`
    is deliberately absent — a second inference entry point is what rebuilt the fleet per cell.
    """
    _run(s1_orbit="both")
    kw = wired["seq_kwargs"]
    assert kw["session_s1_orbit"] == "both"
    assert callable(kw["plan"]) and callable(kw["session"])
    assert "infer_single" not in kw, "a second inference entry point is what rebuilt the fleet"
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

    def hold_for_backlog(self, cells, minutes):
        self._order.append(f"hold:{minutes}")

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


@pytest.mark.parametrize(
    ("delay", "held", "announced"),
    [(0.0, [], "as soon as the first mosaic lands"), (5.0, ["hold:5.0"], "5 min after the first mosaic lands")],
)
def test_the_gpu_request_is_held_only_when_a_delay_is_asked_for(wired, monkeypatch, caplog, delay, held, announced):
    """The default goes straight from the first landing to `ray up`, exactly as before; a delay
    holds between the two, and nowhere else. The opening log line says which.
    """
    monkeypatch.setattr(mod, "_DeploymentCellInputs", partial(_RecordingInputs, wired["order"]))
    with caplog.at_level(logging.INFO, logger="test-seq-flow"):
        _run(zones=["01N", "02N"], ingest=True, inference_start_delay_minutes=delay)
    order = wired["order"]
    assert order[order.index("wait_first:01N,02N") + 1 : order.index("ray_up")] == held
    assert f"GPUs are requested {announced}" in caplog.text


def test_cells_are_ordered_by_true_tile_count_not_the_clamped_fleet_request(wired, monkeypatch):
    """Densest first, on the UNCLAMPED count.

    `num_actors` is `min(num_actors, n_tiles)`, so every zone bigger than the fleet
    collapses to the same value — sorting on it leaves the whole dense end of the
    list in arbitrary order, which is exactly the part that decides which zone the
    fleet opens on.
    """
    counts = {"01N": 900, "02N": 100, "03N": 500, "04N": 3}
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda mask, zone, **k: counts[zone])
    _run(zones=["01N", "02N", "03N", "04N"], num_actors=20)
    cells = wired["seq_kwargs"]["cells"]
    assert [c.zone for c in cells] == ["01N", "03N", "02N", "04N"]
    # ...and the dense three clamp to the same actor request, which is what made the old
    # sort key unable to tell 900 tiles from 500; a zone smaller than the fleet asks for its tiles.
    assert [c.num_actors for c in cells] == [20, 20, 20, 3]


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


@pytest.mark.parametrize("model", ["v1.1", "v2-large"])
def test_the_shared_session_passes_the_runners_encoder_check(wired, monkeypatch, model):
    """The session's placeholder run id must name the store's encoder, or a v2 fill is refused."""
    monkeypatch.setattr(mod, "store_model_version", lambda *a, **k: model)
    captured: dict = {}
    monkeypatch.setattr(
        mod,
        "run_inference",
        lambda n, config, chunks, mosaics, staging, run_id, *a, **k: (
            captured.update(run_id=run_id, model=config.model_version) or []
        ),
    )
    _run(ami_ssm_name_by_model={"v1.1": "ami-v11", "v2-large": "ami-v2"})
    wired["seq_kwargs"]["session"](lambda: None, lambda item, result: None)
    assert captured["model"] == model
    assert_run_id_matches_model(captured["run_id"], model)


def test_a_nonpositive_failure_cap_is_rejected_before_any_side_effect(wired):
    """Same block, same reason as `look_ahead`: `0 >= cap` is true before any cell has failed, so
    a non-positive value stops the feeder immediately — but the runner that used to be the only
    place checking it is not reached until every live ingest is primed and `ray_cluster` entered.
    The cheap refusal would arrive after the expensive part.
    """
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_retained_failures must be >= 1"):
            _run(max_retained_failures=bad)
        assert wired["ray_kwargs"] is None, "no cluster may be provisioned"
        assert wired["seq_kwargs"] is None, "the runner must never be entered"


def test_the_flow_default_matches_the_runner_default(wired):
    """The flow ALWAYS forwards its own value, so its default is what production runs.

    Left at 30 while the runner said 100, every campaign launch would have used 30 — the raised
    cap would have existed only for direct callers, which is nobody in production. Found in review.
    """
    import inspect

    from tessera_embeddings.orchestration.runners import sequential_fill as runner_mod

    flow = getattr(mod.fill_zones_sequential_flow, "fn", mod.fill_zones_sequential_flow)
    flow_default = inspect.signature(flow).parameters["max_retained_failures"].default
    runner_default = inspect.signature(runner_mod.fill_zones_sequential).parameters["max_retained_failures"].default
    assert flow_default == runner_default == 100, (
        f"flow={flow_default} runner={runner_default} — a divergence here is invisible in "
        "production because the flow always forwards its own value"
    )


@pytest.mark.parametrize("param", ["look_ahead", "inference_start_delay_minutes"])
def test_a_negative_width_or_delay_is_rejected_before_any_side_effect(wired, param):
    """Rejected before triage / priming / `ray up`. A negative look_ahead is a deadlock that
    only manifested after the GPU cluster was provisioned.
    """
    with pytest.raises(ValueError, match=f"{param} must be >= 0"):
        _run(**{param: -2})
    assert wired["preflights"] == []  # no triage
    assert wired["ray_kwargs"] is None  # no cluster provisioned
    assert wired["seq_kwargs"] is None  # runner never entered


@pytest.mark.parametrize(
    ("step", "exc"),
    [
        ("wait_first", RuntimeError("prefect API unreachable")),
        # What Prefect raises in the flow process when the run is cancelled.
        ("hold_for_backlog", TerminationSignal(signal=15)),
    ],
)
def test_a_failure_while_priming_still_cancels_the_ingests_it_started(wired, monkeypatch, step, exc):
    """Priming, the first-landing wait and the backlog hold all live inside the shutdown guard,
    so no child ingest is ever orphaned.

    ``start`` submits the whole window at once, so from the first submission there are
    child runs that only ``shutdown()`` can cancel. With priming above the ``try``, a
    failure between the first submission and the cluster — a raising ``start``, an
    unexpected error in the wait — returned without cancelling them, leaving children
    writing mosaic prefixes server-side that a prompt retry would then race.
    """

    def _raise(self, *a, **k):
        raise exc

    failing = type("_FailingInputs", (_RecordingInputs,), {step: _raise})
    monkeypatch.setattr(mod, "_DeploymentCellInputs", partial(failing, wired["order"]))
    with pytest.raises(type(exc)):
        _run(zones=["01N", "02N", "03N"], ingest=True, look_ahead=1, inference_start_delay_minutes=5.0)

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
    assert wired["seq_kwargs"]["look_ahead"] == 2


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


@pytest.mark.parametrize("rule", [25, None])
def test_the_chained_fill_applies_the_rule_its_store_advertises(wired, monkeypatch, rule) -> None:
    """The DEFAULT campaign strategy was reading no depth rule at all.

    `fill_zone_year` resolves `optical_min_obs` from the store root, but this path built its
    config without it — so a store seeded with a minimum would have published pixels below
    its own advertised line, through the strategy the campaign actually runs. A store seeded
    before the rule existed declares none, and keeps filling unconstrained.
    """
    built: list[dict] = []

    def record(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(mod, "build_inference_config", record)
    monkeypatch.setattr(mod, "_optical_min_obs_from_store", lambda *a, **k: rule)
    _run(zones=["33N"])
    assert built, "no InferenceConfig was built"
    assert {c.get("optical_min_obs") for c in built} == {rule}, "the store's rule must reach every cell's config"


# --- a wedged trailing assembly ends the process rather than reporting FAILED (2026-09-04) ------


class TestWedgedDrainEndsTheProcess:
    """`FAILED` tells the driver a fill stopped writing, which a wedged drain does not establish,
    so the flow ends its process instead — AFTER teardown, and via the one hard-exit site.
    """

    def test_the_helper_exits_through_the_one_sanctioned_site_with_the_temp_fail_status(self, monkeypatch, caplog):
        import logging as _logging

        from tessera_embeddings.orchestration.prefect.flows import fill_zones_sequential as flowmod
        from tessera_embeddings.orchestration.runners.sequential_fill import TrailingAssemblyWedgedError

        calls: list[dict] = []
        monkeypatch.setattr(
            flowmod, "hard_exit_after_flush", lambda status, **kw: calls.append({"status": status, **kw})
        )
        log = _logging.getLogger("test-wedged-exit")
        flowmod._end_process_after_wedged_drain(TrailingAssemblyWedgedError(4), log)
        assert [c["status"] for c in calls] == [flowmod.WEDGED_DRAIN_EXIT_STATUS]
        assert "Ending this process" in calls[0]["message"] and calls[0]["args"][0] == 4, (
            "the exit must say why, and how many cells it leaves"
        )

    def test_a_ray_teardown_that_also_raises_cannot_swallow_the_wedge(self, wired, monkeypatch):
        """THE THIRD PLACE THIS INVARIANT LEAKED. The wedge is raised inside the Ray context, so
        leaving it runs `ray.shutdown()` and `ray down` — and an exception there REPLACES the
        wedge, after which the outer `finally` sees none and Prefect reports FAILED with the
        assembly thread alive. Latching at detection is what closes it.
        """
        from tessera_embeddings.orchestration.runners.sequential_fill import TrailingAssemblyWedgedError

        @contextmanager
        def exploding_teardown(log, **kwargs):
            try:
                yield None
            finally:
                raise RuntimeError("ray down failed")

        monkeypatch.setattr("tessera_embeddings.providers.aws.ray.ray_cluster", exploding_teardown, raising=False)
        monkeypatch.setattr(
            mod, "fill_zones_sequential", lambda **k: (_ for _ in ()).throw(TrailingAssemblyWedgedError(3))
        )
        exited: list[int] = []
        monkeypatch.setattr(mod, "hard_exit_after_flush", lambda status, **k: exited.append(status))

        with pytest.raises(RuntimeError, match="ray down failed"):
            _run(zones=["33N"])
        assert exited == [mod.WEDGED_DRAIN_EXIT_STATUS], "the teardown failure swallowed the wedge"

    def test_a_cleanup_that_never_returns_cannot_hold_up_the_wedged_exit(self, wired, monkeypatch):
        """The housekeeping pool runs prefix deletes with THREE unbounded legs — the `s5cmd`
        subprocess, the `fsspec` recursive `rm` it falls back to, and the read-back that lists
        objects — so joining it on the wedged path is the six-hour drain ceiling guaranteeing
        nothing: the flow parks forever with the abandoned assembly thread alive. Only the
        wedged path skips the join; the healthy one still waits, which its own comment explains.
        """
        from tessera_embeddings.orchestration.runners.sequential_fill import TrailingAssemblyWedgedError

        hold = threading.Event()  # never set while the flow runs: the delete never returns
        started = threading.Event()

        def wedge_behind_a_hung_delete(**kwargs):
            def _delete_that_never_returns() -> None:
                started.set()
                hold.wait(60.0)

            kwargs["housekeeping"].submit(_delete_that_never_returns)
            assert started.wait(5.0), "the delete never started, so the join would not have blocked"
            raise TrailingAssemblyWedgedError(2)

        monkeypatch.setattr(mod, "fill_zones_sequential", wedge_behind_a_hung_delete)
        exited: list[int] = []
        monkeypatch.setattr(mod, "hard_exit_after_flush", lambda status, **k: exited.append(status))

        try:
            with pytest.raises(TrailingAssemblyWedgedError):
                run_under_deadline(
                    20,
                    lambda: _run(zones=["33N"]),
                    what="the flow whose cleanup hung behind a wedge",
                )
        finally:
            hold.set()
        assert exited == [mod.WEDGED_DRAIN_EXIT_STATUS], "the hung delete kept the process alive"
