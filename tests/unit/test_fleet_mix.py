"""The GPU fleet-mix policy, and its contract with the cluster template.

Every number in `TestTheCampaignShape` was derived by hand against the campaign's
real configuration and then checked against a live dev cluster on 2026-08-28. They
are a guard, not an illustration: the policy decides what a production fleet is
made of, and each of its terms lives somewhere a later edit could move silently.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from tessera_embeddings.providers.aws import fleet_mix
from tessera_embeddings.providers.aws import ray as tray

TEMPLATE = pathlib.Path(tray.__file__).parent / "cluster.yaml.template"

L40S, A10G, L4 = "g6e.xlarge", "g5.2xlarge", "g6.2xlarge"
#: The campaign's configuration as of 2026-08-28 — see docs/providers/aws.md.
LADDER, BUDGET = "g6e.xlarge:101", 840


def _campaign_ceilings() -> dict[str, int]:
    config = yaml.safe_load(TEMPLATE.read_text())
    tray._apply_gpu_worker_ladder(config, LADDER)
    tray._apply_gpu_fallback(config, [A10G], BUDGET)
    plan = fleet_mix.plan_from_resolved_config(config, BUDGET)
    assert plan is not None
    return plan.ceilings


def _vcpu(asks: dict[str, int]) -> int:
    return sum(n * fleet_mix.RUNGS_BY_INSTANCE_TYPE[t].vcpu_per_node for t, n in asks.items())


def _config():
    """A minimal InferenceConfig; only the actor-request knobs matter here."""
    from tessera_embeddings.config.inference import InferenceConfig
    from tessera_embeddings.config.time_windows import parse_time_window

    return InferenceConfig(
        time_window=parse_time_window("December 2024"),
        checkpoint_path="s3://ckpt/model.pt",
        actor_request_batch_size=4,
    )


def _log():
    import logging

    return logging.getLogger("test-fleet-mix")


class TestTheCampaignShape:
    """The four states the running campaign actually moves between."""

    def test_the_ceilings_come_out_at_101_and_105(self) -> None:
        assert _campaign_ceilings() == {L40S: 101, A10G: 105}

    def test_under_drought_the_fallback_takes_the_rest_of_the_budget(self) -> None:
        """35 L40S supplied. The primary gets a bounded probe above what it holds;
        everything the budget still affords goes to the fallback.
        """
        asks = fleet_mix.fleet_asks(
            want_gpus=250,
            live_by_instance_type={L40S: 35},
            ceilings=_campaign_ceilings(),
            vcpu_budget=BUDGET,
        )
        assert asks == {L40S: 51, A10G: 79}
        assert _vcpu(asks) <= BUDGET

    def test_when_the_primary_recovers_the_fallback_ask_falls(self) -> None:
        """The budget is reallocated to the better card. Fallback machines above the
        new ask lose their constraint protection and drain on the idle timeout.
        """
        asks = fleet_mix.fleet_asks(
            want_gpus=250,
            live_by_instance_type={L40S: 101},
            ceilings=_campaign_ceilings(),
            vcpu_budget=BUDGET,
        )
        assert asks == {L40S: 101, A10G: 54}
        assert _vcpu(asks) <= BUDGET

    def test_from_cold_the_primary_gets_only_a_probe(self) -> None:
        """Nothing live yet, so the primary is asked for the probe alone and the
        fallback takes the rest — which is what makes both pools fill AT ONCE
        rather than one after the other.
        """
        asks = fleet_mix.fleet_asks(
            want_gpus=250,
            live_by_instance_type={},
            ceilings=_campaign_ceilings(),
            vcpu_budget=BUDGET,
        )
        assert asks == {L40S: fleet_mix.DEFAULT_PROBE, A10G: 97}
        assert asks[A10G] > 0, "the fallback must be asked for before the primary fills"

    def test_a_draining_queue_collapses_the_asks(self) -> None:
        """This is what stops a standing ask from buying machines the actor pool
        then idle-retires, which the ask would immediately re-buy.
        """
        asks = fleet_mix.fleet_asks(
            want_gpus=12,
            live_by_instance_type={L40S: 35},
            ceilings=_campaign_ceilings(),
            vcpu_budget=BUDGET,
        )
        assert asks == {L40S: 12, A10G: 0}


class TestTheCeilingClamp:
    """An ask above a rung's ceiling provisions NOTHING — measured, not assumed.

    `_enforce_resource_constraints` discards a partially infeasible constraint whole
    rather than filling what it can. On a dev cluster on 2026-08-28 an ask of 30
    against a ceiling of 25 produced zero machines in three minutes, where a feasible
    ask of 10 produced six within two seconds. The clamp is load-bearing.
    """

    def test_no_ask_ever_exceeds_its_ceiling(self) -> None:
        asks = fleet_mix.fleet_asks(
            want_gpus=10_000,
            live_by_instance_type={},
            ceilings={L40S: 3, A10G: 5},
        )
        assert asks == {L40S: 3, A10G: 5}

    def test_a_huge_demand_cannot_push_a_rung_past_its_ceiling(self) -> None:
        ceilings = _campaign_ceilings()
        asks = fleet_mix.fleet_asks(
            want_gpus=10_000,
            live_by_instance_type={L40S: 101},
            ceilings=ceilings,
            vcpu_budget=100_000,
        )
        for instance_type, count in asks.items():
            assert count <= ceilings[instance_type]


class TestTheBudget:
    """The vCPU quota is the binding constraint, and it is counted per instance."""

    def test_the_asks_never_exceed_the_budget(self) -> None:
        for live in ({}, {L40S: 20}, {L40S: 101}, {L40S: 50, A10G: 20}):
            asks = fleet_mix.fleet_asks(
                want_gpus=250,
                live_by_instance_type=live,
                ceilings=_campaign_ceilings(),
                vcpu_budget=BUDGET,
            )
            assert _vcpu(asks) <= BUDGET, f"{live} -> {asks}"

    def test_without_a_budget_only_demand_and_ceilings_bind(self) -> None:
        asks = fleet_mix.fleet_asks(
            want_gpus=250,
            live_by_instance_type={},
            ceilings=_campaign_ceilings(),
        )
        assert asks == {L40S: fleet_mix.DEFAULT_PROBE, A10G: 105}

    def test_a_budget_too_small_for_the_fallback_asks_zero_rather_than_overspending(self) -> None:
        asks = fleet_mix.fleet_asks(
            want_gpus=250,
            live_by_instance_type={},
            ceilings={L40S: 101, A10G: 105},
            vcpu_budget=20,
        )
        assert asks == {L40S: 5, A10G: 0}

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_budget_is_refused(self, bad: int) -> None:
        with pytest.raises(ValueError, match="vcpu_budget"):
            fleet_mix.fleet_asks(want_gpus=1, live_by_instance_type={}, ceilings={L40S: 1}, vcpu_budget=bad)


class TestMoreThanOneFallback:
    """The registry is n-ary. Opening two fallbacks used to be refused because Ray
    broke the score tie on node-type name; the mix is stated now, so it is allowed.
    """

    def test_the_better_ratio_takes_the_budget_first(self) -> None:
        asks = fleet_mix.fleet_asks(
            want_gpus=250,
            live_by_instance_type={},
            ceilings={L40S: 101, A10G: 105, L4: 105},
            vcpu_budget=BUDGET,
        )
        assert asks[A10G] == 97
        assert asks[L4] == 0, "the worst ratio on the board gets what the others leave"

    def test_the_worst_rung_is_reached_when_the_better_ones_are_capped(self) -> None:
        asks = fleet_mix.fleet_asks(
            want_gpus=250,
            live_by_instance_type={},
            ceilings={L40S: 2, A10G: 4, L4: 100},
            vcpu_budget=BUDGET,
        )
        assert asks == {L40S: 2, A10G: 4, L4: 100}

    def test_the_registry_is_ordered_by_value_per_vcpu(self) -> None:
        values = [r.value_per_vcpu for r in fleet_mix.GPU_RUNGS]
        assert values == sorted(values, reverse=True)


class TestBundles:
    """What `request_resources` is actually handed."""

    def test_one_bundle_per_machine_naming_only_that_rung(self) -> None:
        bundles = fleet_mix.bundles_for({L40S: 2, A10G: 1})
        assert bundles.count({f"{fleet_mix.MARKER_PREFIX}{L40S}": 1}) == 2
        assert bundles.count({f"{fleet_mix.MARKER_PREFIX}{A10G}": 1}) == 1
        assert len(bundles) == 3

    def test_an_unknown_instance_type_is_refused(self) -> None:
        with pytest.raises(KeyError, match=r"p5\.48xlarge"):
            fleet_mix.bundles_for({"p5.48xlarge": 1})

    def test_zero_asks_produce_no_bundles(self) -> None:
        assert fleet_mix.bundles_for({L40S: 0, A10G: 0}) == []


class TestPlanFromConfig:
    """Reading the open rungs out of the config `ray up` was handed."""

    def test_a_config_with_no_fallback_open_yields_no_plan(self) -> None:
        """Inert by default: with nothing to mix, a constraint would add nothing."""
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, LADDER)
        assert fleet_mix.plan_from_resolved_config(config, BUDGET) is None

    def test_two_node_types_offering_one_instance_type_are_refused(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        types = config["available_node_types"]
        types["gpu-workers-ondemand-duplicate"] = {
            "node_config": {"InstanceType": L40S},
            "resources": {"CPU": 4, "GPU": 1},
            "max_workers": 5,
        }
        types["gpu-workers-ondemand"]["max_workers"] = 5
        with pytest.raises(RuntimeError, match="no single target"):
            fleet_mix.plan_from_resolved_config(config, BUDGET)

    def test_the_ceilings_are_read_from_the_config_ray_was_given(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:7,g5.2xlarge:9")
        plan = fleet_mix.plan_from_resolved_config(config, None)
        assert plan is not None
        assert plan.ceilings == {L40S: 7, A10G: 9}


class TestTheTemplateContract:
    """The marker resources the mechanism addresses rungs by, pinned to the template.

    A bundle can only reach a rung that declares the marker it names, so a template
    that drops one makes the fleet mix silently unable to ask for that card.
    """

    @staticmethod
    def _node_types() -> dict:
        return yaml.safe_load(TEMPLATE.read_text())["available_node_types"]

    def test_every_registry_rung_has_a_template_rung_declaring_its_marker(self) -> None:
        declared = {
            cfg["node_config"]["InstanceType"]: cfg.get("resources", {})
            for name, cfg in self._node_types().items()
            if name.startswith(tray.GPU_WORKER_NODE_TYPE_PREFIX)
        }
        for rung in fleet_mix.GPU_RUNGS:
            assert rung.instance_type in declared, f"{rung.instance_type} has no on-demand rung"
            assert declared[rung.instance_type].get(rung.marker) == 1

    def test_the_spot_rung_carries_no_marker(self) -> None:
        """It shares `g6e.xlarge` with the on-demand rung. If it declared the same
        marker, an L40S ask could be satisfied by INTERRUPTIBLE capacity — which is
        exactly why the marker is ours and named per node type rather than per card.
        """
        spot = self._node_types()["gpu-workers-spot"]["resources"]
        assert not any(k.startswith(fleet_mix.MARKER_PREFIX) for k in spot)

    def test_the_head_carries_no_marker(self) -> None:
        head = self._node_types()["head"]["resources"]
        assert not any(k.startswith(fleet_mix.MARKER_PREFIX) for k in head)

    def test_the_declared_vcpu_matches_the_registry(self) -> None:
        """`fleet_asks` prices the budget from the registry; Ray scales against the
        template. A disagreement would spend a quota nobody budgeted for.
        """
        for name, cfg in self._node_types().items():
            if not name.startswith(tray.GPU_WORKER_NODE_TYPE_PREFIX):
                continue
            rung = fleet_mix.RUNGS_BY_INSTANCE_TYPE[cfg["node_config"]["InstanceType"]]
            assert cfg["resources"]["CPU"] == rung.vcpu_per_node
            assert cfg["resources"]["GPU"] == rung.gpus_per_node


class TestRefusals:
    """Inputs that cannot mean anything are refused rather than coerced."""

    @pytest.mark.parametrize("bad", [-1, -100])
    def test_a_negative_demand_is_refused(self, bad: int) -> None:
        with pytest.raises(ValueError, match="want_gpus"):
            fleet_mix.fleet_asks(want_gpus=bad, live_by_instance_type={}, ceilings={L40S: 1})

    def test_a_negative_probe_is_refused(self) -> None:
        with pytest.raises(ValueError, match="probe"):
            fleet_mix.fleet_asks(want_gpus=1, live_by_instance_type={}, ceilings={L40S: 1}, probe=-1)

    def test_no_open_rung_asks_for_nothing(self) -> None:
        assert fleet_mix.fleet_asks(want_gpus=250, live_by_instance_type={}, ceilings={}) == {}


class TestTheDemandIsPublishedBeforeTheFleetIsWaitedOn:
    """The initial ask must precede `wait_for_actors`, not follow it.

    The scheduler republishes the demand every round, but it does not start until
    `wait_for_actors` returns. Under a TOTAL primary-capacity drought no first-batch
    actor ever initializes, so that wait runs to its hours-long timeout and the run
    dies having never once asked for the fallback — reproducing exactly the starvation
    this machinery exists to prevent. Found in review of PR #159.
    """

    @staticmethod
    def _run(published: list[int], *, wait_raises: bool, chunks: list | None = None, chained: bool = False) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from tessera_embeddings.inference import runner as runner_mod

        def _wait(*_args: object, **_kwargs: object) -> tuple:
            if wait_raises:
                raise RuntimeError("Only 0 / 8 actors initialized within 600s")
            return ([], [], set())

        with (
            patch.object(runner_mod, "InferenceActor") as actor,
            patch.object(runner_mod, "wait_for_actors", side_effect=_wait),
            patch.object(runner_mod, "ZarrWriter") as writer,
        ):
            # Nothing already staged, so the chunk list reaches the ask intact.
            writer.return_value.scan_existing_staged_artifacts.return_value = SimpleNamespace(done=set(), skipped=set())
            actor.options.return_value.remote.return_value = object()
            with pytest.raises(RuntimeError):
                runner_mod.run_inference(
                    8,
                    _config(),
                    chunks or [],
                    "m",
                    "s",
                    "r",
                    0.0,
                    _log(),
                    on_fleet_demand=published.append,
                    more_work=(lambda: None) if chained else None,
                )

    def test_a_drought_that_kills_the_actor_wait_has_still_asked_for_the_fallback(self) -> None:
        published: list[int] = []
        self._run(published, wait_raises=True, chained=True)
        assert published == [8], "a chained session's work arrives later, so ask for the whole target"

    def test_a_static_run_asks_only_for_the_work_it_holds(self) -> None:
        """The scheduler cannot correct an over-ask until the first actor arrives, so a
        20-actor default against one live tile would launch and bill 20 machines for one
        chunk. Bounded here rather than later.
        """
        published: list[int] = []
        self._run(published, wait_raises=True, chunks=[object(), object()])
        assert published == [2]

    def test_every_session_that_gets_a_terminator_also_gets_the_publisher(self) -> None:
        """Structural, because both reviewers missed the same call site independently.

        `fill_zones_sequential` starts inference from two places — the shared stream and
        the per-cell retry. Wiring only the first leaves a retry cold-starting on Ray's
        primary-only greedy choice, which starves exactly as before. The invariant is
        "wherever a fleet is driven, its shape is published", so assert it over the call
        sites rather than over one of them.
        """
        import ast
        import inspect

        from tessera_embeddings.orchestration.prefect.flows import fill_zones_sequential as mod

        tree = ast.parse(inspect.getsource(mod))
        sites = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and any(kw.arg == "on_actor_retire" for kw in node.keywords)
        ]
        assert len(sites) >= 2, "expected the shared session and the per-cell retry"
        for site in sites:
            passed = {kw.arg for kw in site.keywords}
            assert "on_fleet_demand" in passed, f"line {site.lineno} drives a fleet without publishing its shape"
