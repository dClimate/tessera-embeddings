"""The GPU fleet-mix policy, its bounds, and the lifecycle that publishes it.

Every figure in `TestTheCampaignShape` was derived by hand against the campaign's real
configuration and checked against a live dev cluster on 2026-08-28. They are a guard, not
an illustration: this decides what a production fleet is made of.

**Every bound below is load-bearing, not defensive.** Ray discards a partially infeasible
constraint whole rather than filling what it can — measured, an ask of 30 against a
ceiling of 25 produced zero machines in three minutes where a feasible ask of 10 produced
six in two seconds. An unbounded ask does not overshoot; it provisions nothing.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from tessera_embeddings.inference.scheduling import FleetDemand
from tessera_embeddings.providers.aws import fleet_mix
from tessera_embeddings.providers.aws import ray as tray

TEMPLATE = pathlib.Path(tray.__file__).parent / "cluster.yaml.template"
L40S, A10G, L4 = "g6e.xlarge", "g5.2xlarge", "g6.2xlarge"
#: The campaign's configuration as of 2026-08-28 — see docs/providers/aws.md.
LADDER, BUDGET = "g6e.xlarge:101", 840
PROBE = fleet_mix.DEFAULT_PROBE


def _log():
    import logging

    return logging.getLogger("test-fleet-mix")


def _config():
    """A minimal InferenceConfig; only the actor-request knobs matter here."""
    from tessera_embeddings.config.inference import InferenceConfig
    from tessera_embeddings.config.time_windows import parse_time_window

    return InferenceConfig(
        time_window=parse_time_window("December 2024"),
        checkpoint_path="s3://ckpt/model.pt",
        actor_request_batch_size=4,
    )


def _ceilings() -> dict[str, int]:
    config = yaml.safe_load(TEMPLATE.read_text())
    tray._apply_gpu_worker_ladder(config, LADDER)
    tray._apply_gpu_fallback(config, [A10G], BUDGET)
    plan = fleet_mix.plan_from_resolved_config(config, BUDGET)
    assert plan is not None
    return plan.ceilings


def _held_vcpu(asks: dict[str, int], live: dict[str, int]) -> int:
    """What the fleet will HOLD, which is what the budget must bound: an ask cannot remove
    a busy machine, so live counts above their new ask still cost quota.
    """
    return sum(
        max(asks.get(t, 0), live.get(t, 0)) * fleet_mix.RUNGS_BY_INSTANCE_TYPE[t].vcpu_per_node
        for t in set(asks) | set(live)
    )


class TestTheCampaignShape:
    """The states the running campaign actually moves between."""

    def test_the_ceilings_come_out_at_101_and_105(self) -> None:
        assert _ceilings() == {L40S: 101, A10G: 105}

    @pytest.mark.parametrize(
        ("case", "want", "live", "expected"),
        [
            # Cold: the primary gets a probe, the fallback the rest — which is what makes
            # both pools fill AT ONCE rather than one after the other.
            ("cold start", 250, {}, {L40S: PROBE, A10G: 97}),
            # Drought: a standing ask above what is supplied; fallback takes the rest.
            ("drought", 250, {L40S: 35}, {L40S: 51, A10G: 79}),
            # Recovery: budget reallocated to the better card; surplus fallback drains.
            ("recovered", 250, {L40S: 101}, {L40S: 101, A10G: 54}),
            # Tail: the ask falls BELOW the live count so surplus machines idle out,
            # rather than a stale floor holding them through assembly.
            ("draining queue", 12, {L40S: 35}, {L40S: 12, A10G: 0}),
        ],
    )
    def test_the_shape(self, case: str, want: int, live: dict, expected: dict) -> None:
        asks = fleet_mix.fleet_asks(
            want_gpus=want, live_by_instance_type=live, ceilings=_ceilings(), vcpu_budget=BUDGET
        )
        assert asks == expected, case
        assert _held_vcpu(asks, live) <= BUDGET, case

    @pytest.mark.parametrize("live", [{}, {L40S: 20}, {L40S: 51, A10G: 79}, {L40S: 101, A10G: 54}, {A10G: 105}])
    def test_the_live_fleet_stays_inside_the_budget_from_every_state(self, live: dict) -> None:
        """Pricing the ASKS alone let the fleet ratchet past the budget: at 51 L40S and 79
        A10G the next round asked 67 L40S while all 79 stayed up — 900 vCPU against 840,
        repeatable. Live machines are charged first now.
        """
        asks = fleet_mix.fleet_asks(want_gpus=250, live_by_instance_type=live, ceilings=_ceilings(), vcpu_budget=BUDGET)
        assert _held_vcpu(asks, live) <= BUDGET


class TestTheBoundsAreLoadBearing:
    """Each bound exists because without it the request provisions NOTHING."""

    def test_no_ask_exceeds_its_rung_ceiling(self) -> None:
        assert fleet_mix.fleet_asks(want_gpus=10_000, live_by_instance_type={}, ceilings={L40S: 3, A10G: 5}) == {
            L40S: 3,
            A10G: 5,
        }

    def test_the_combined_ask_fits_under_the_cluster_wide_ceiling(self) -> None:
        """The per-rung ceilings do not bound their sum: 16 plus 500 exceeds a 500-worker
        cluster, and that is discarded whole rather than partially filled.
        """
        asks = fleet_mix.fleet_asks(
            want_gpus=600, live_by_instance_type={}, ceilings={L40S: 500, A10G: 500}, max_total=500
        )
        assert sum(asks.values()) <= 500

    def test_a_small_ask_still_reaches_the_fallback(self) -> None:
        """A demand at or below the probe would otherwise go entirely to the rung that is
        refusing, leaving the cold-start request with no fallback at all.
        """
        asks = fleet_mix.fleet_asks(want_gpus=PROBE, live_by_instance_type={}, ceilings=_ceilings(), vcpu_budget=BUDGET)
        assert asks[A10G] >= 1

    def test_a_budget_that_binds_the_probe_still_reaches_the_fallback(self) -> None:
        """A fallback machine costs more quota than a primary one, so reserving a slot in
        the demand is not enough — the quota has to be reserved too.
        """
        asks = fleet_mix.fleet_asks(want_gpus=6, live_by_instance_type={}, ceilings=_ceilings(), vcpu_budget=20)
        assert asks[A10G] == 1

    def test_a_live_primary_does_not_cancel_the_fallback_probe(self) -> None:
        """The floor is keyed on the FALLBACK's live count, not the primary's. An earlier
        version dropped it as soon as the primary had any machine, reasoning that the
        scheduler would correct — but the scheduler recomputes this same function, so it
        never did: eight wanted GPUs with four live L40S asked for zero A10G forever, even
        with the primary refusing everything above four.
        """
        asks = fleet_mix.fleet_asks(
            want_gpus=8, live_by_instance_type={L40S: 4}, ceilings=_ceilings(), vcpu_budget=BUDGET
        )
        assert asks[A10G] >= 1

    def test_a_tight_aggregate_ceiling_still_reserves_for_the_fallback(self) -> None:
        asks = fleet_mix.fleet_asks(
            want_gpus=50, live_by_instance_type={}, ceilings=_ceilings(), vcpu_budget=BUDGET, max_total=3
        )
        assert asks[A10G] >= 1
        assert sum(asks.values()) <= 3

    def test_a_draining_tail_does_not_buy_a_probe_it_cannot_use(self) -> None:
        """The floor only applies while the machines we hold cannot already cover the
        demand — otherwise a tail would buy a machine for work that does not exist.
        """
        asks = fleet_mix.fleet_asks(
            want_gpus=12, live_by_instance_type={L40S: 35}, ceilings=_ceilings(), vcpu_budget=BUDGET
        )
        assert asks == {L40S: 12, A10G: 0}

    def test_without_a_budget_only_demand_and_ceilings_bind(self) -> None:
        asks = fleet_mix.fleet_asks(want_gpus=250, live_by_instance_type={}, ceilings=_ceilings())
        assert asks[L40S] == PROBE
        assert asks[A10G] >= 100

    @pytest.mark.parametrize(("kwargs", "match"), [({"vcpu_budget": 0}, "vcpu_budget"), ({"probe": -1}, "probe")])
    def test_meaningless_inputs_are_refused(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            fleet_mix.fleet_asks(want_gpus=1, live_by_instance_type={}, ceilings={L40S: 1}, **kwargs)

    def test_a_negative_demand_is_refused(self) -> None:
        with pytest.raises(ValueError, match="want_gpus"):
            fleet_mix.fleet_asks(want_gpus=-1, live_by_instance_type={}, ceilings={L40S: 1})

    def test_no_open_rung_asks_for_nothing(self) -> None:
        assert fleet_mix.fleet_asks(want_gpus=250, live_by_instance_type={}, ceilings={}) == {}


class TestTheAllocationIsNAry:
    """`_apply_gpu_fallback` opens one fallback, but the policy is n-ary and pinned for the
    day that lifts — which is why the registry carries a value-per-vCPU order rather than a
    hardcoded pair.
    """

    def test_the_better_ratio_takes_the_budget_first(self) -> None:
        asks = fleet_mix.fleet_asks(
            want_gpus=250,
            live_by_instance_type={},
            ceilings={L40S: 101, A10G: 105, L4: 105},
            vcpu_budget=BUDGET,
        )
        # The best ratio takes the bulk; the worst gets only its reserved machine, which
        # exists so an unproven middle rung cannot starve it entirely.
        assert asks[A10G] == 96
        assert asks[L4] == 1

    def test_the_worst_rung_is_reached_when_the_better_ones_are_capped(self) -> None:
        asks = fleet_mix.fleet_asks(
            want_gpus=250, live_by_instance_type={}, ceilings={L40S: 2, A10G: 4, L4: 100}, vcpu_budget=BUDGET
        )
        assert asks == {L40S: 2, A10G: 4, L4: 100}

    def test_the_registry_is_ordered_by_value_per_vcpu(self) -> None:
        values = [r.value_per_vcpu for r in fleet_mix.GPU_RUNGS]
        assert values == sorted(values, reverse=True)


class TestARungMustBeAddressable:
    """A bundle can only reach a rung declaring the marker it names, so a rung without one
    is worse than no rung: nothing could satisfy the request and the fallback would sit
    idle while the run believed it had asked.
    """

    @staticmethod
    def _node_types() -> dict:
        return yaml.safe_load(TEMPLATE.read_text())["available_node_types"]

    def test_every_gpu_rung_declares_its_marker_and_matches_the_registry(self) -> None:
        for name, cfg in self._node_types().items():
            if not name.startswith(tray.GPU_WORKER_NODE_TYPE_PREFIX):
                continue
            rung = fleet_mix.RUNGS_BY_INSTANCE_TYPE[cfg["node_config"]["InstanceType"]]
            assert cfg["resources"].get(rung.marker) == 1
            # `fleet_asks` prices the budget from the registry; Ray scales against the
            # template. A disagreement spends a quota nobody budgeted for.
            assert cfg["resources"]["CPU"] == rung.vcpu_per_node
            assert cfg["resources"]["GPU"] == rung.gpus_per_node

    @pytest.mark.parametrize("node_type", ["gpu-workers-spot", "head"])
    def test_the_spot_rung_and_head_carry_no_marker(self, node_type: str) -> None:
        """The spot rung shares `g6e.xlarge` with the on-demand one. A shared marker would
        let an L40S ask be satisfied by INTERRUPTIBLE capacity.
        """
        resources = self._node_types()[node_type]["resources"]
        assert not any(k.startswith(fleet_mix.MARKER_PREFIX) for k in resources)

    def test_a_rung_without_its_marker_is_excluded(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:7,g5.2xlarge:9")
        del config["available_node_types"]["gpu-workers-ondemand-a10g-2xl"]["resources"][
            f"{fleet_mix.MARKER_PREFIX}{A10G}"
        ]
        assert fleet_mix.plan_from_resolved_config(config, None) is None

    def test_one_bundle_per_machine_naming_only_that_rung(self) -> None:
        bundles = fleet_mix.bundles_for({L40S: 2, A10G: 1})
        assert bundles.count({f"{fleet_mix.MARKER_PREFIX}{L40S}": 1}) == 2
        assert len(bundles) == 3
        assert fleet_mix.bundles_for({L40S: 0}) == []

    def test_an_unknown_instance_type_is_refused(self) -> None:
        with pytest.raises(KeyError, match=r"p5\.48xlarge"):
            fleet_mix.bundles_for({"p5.48xlarge": 1})


class TestThePlan:
    """Read from the resolved config Ray was handed, so the ceilings enforced and the
    ceilings priced against are one set of numbers.
    """

    def test_no_fallback_open_yields_no_plan(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, LADDER)
        assert fleet_mix.plan_from_resolved_config(config, BUDGET) is None

    def test_it_reads_the_ceilings_and_the_global_cap(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:7,g5.2xlarge:9")
        plan = fleet_mix.plan_from_resolved_config(config, None)
        assert plan is not None
        assert plan.ceilings == {L40S: 7, A10G: 9}
        assert plan.max_total == config["max_workers"]

    def test_two_node_types_offering_one_instance_type_are_refused(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        types = config["available_node_types"]
        types["gpu-workers-ondemand"]["max_workers"] = 5
        types["gpu-workers-ondemand-duplicate"] = {
            "node_config": {"InstanceType": L40S},
            "resources": {"CPU": 4, "GPU": 1, f"{fleet_mix.MARKER_PREFIX}{L40S}": 1},
            "max_workers": 5,
        }
        with pytest.raises(RuntimeError, match="no single target"):
            fleet_mix.plan_from_resolved_config(config, BUDGET)


class TestOneFallbackAtATime:
    """Stating the mix removes Ray's node-name tie for the machines the request covers, but
    the request is a FLOOR — ordinary demand above it is scored by Ray as before, and the
    supported fallbacks are identical to that scorer. Refused in both places, because the
    provider's guard fires only after a chained fill has primed its ingests.
    """

    def test_the_provider_refuses_two(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, LADDER)
        with pytest.raises(RuntimeError, match=r"[Oo]nly one GPU fallback"):
            tray._apply_gpu_fallback(config, [A10G, L4], BUDGET)

    def test_the_campaign_refuses_two_before_any_ingest(self) -> None:
        from tessera_embeddings.orchestration.prefect.flows import run_global_campaign as rgc

        assert "Only one GPU fallback instance type" in pathlib.Path(rgc.__file__).read_text()


class TestFleetDemand:
    """The single place that decides what to publish, and survives publishing it.

    Every review finding on PR #159 that was not about the arithmetic was about this:
    which site publishes, when, bounded by what, converted how, retried or cleared. Two
    sites each computing their own answer produced a defect per forgotten term.
    """

    @pytest.mark.parametrize(
        ("actors", "num_gpus", "machines"),
        [
            (8, 1.0, 8),  # the production default: actors and machines coincide
            (250, 0, 0),  # a CPU-only run must not ask for GPU machines at all
            (250, 0.5, 125),
            (5, 0.4, 3),  # fractional reservations PACK: two to a card, so three machines
            # Every rung is single-GPU and Ray cannot combine GPUs across nodes, so an
            # actor reserving more than one cannot be placed on anything this could buy.
            (10, 2.0, 0),
        ],
    )
    def test_actors_are_converted_to_machines(self, actors: int, num_gpus: float, machines: int) -> None:
        assert FleetDemand.machines(actors, num_gpus) == machines

    def test_it_publishes_the_smallest_of_target_work_and_requested(self) -> None:
        """Bounding by the REQUEST is what keeps the fleet from outrunning the batching
        policy that exists to pace it.
        """
        seen: list[int] = []
        FleetDemand(seen.append, 1.0, _log()).send(target=250, outstanding=99, requested=25)
        assert seen == [25]

    def test_a_failure_never_reaches_the_caller(self) -> None:
        def boom(_: int) -> None:
            raise RuntimeError("GCS unavailable")

        FleetDemand(boom, 1.0, _log()).send(target=1, outstanding=1, requested=1)

    def test_the_cold_start_retries_because_nothing_else_will(self) -> None:
        """The scheduler does not republish until it starts, and under the drought this
        exists for it never starts — one transient failure becomes the whole actor wait.
        """
        calls: list[int] = []

        def flaky(want: int) -> None:
            calls.append(want)
            if len(calls) < 2:
                raise RuntimeError("transient")

        with patch("tessera_embeddings.inference.scheduling.time.sleep"):
            FleetDemand(flaky, 1.0, _log()).send(target=4, outstanding=4, requested=4, retry=True)
        assert len(calls) == 2

    def test_no_publisher_is_a_no_op(self) -> None:
        FleetDemand(None, 1.0, _log()).send(target=9, outstanding=9, requested=9)
        FleetDemand(None, 1.0, _log()).clear()


class TestTheRunnerLifecycle:
    """Publish before the actor wait, and clear on every exit path."""

    @staticmethod
    def _run(published: list[int], *, chunks: list | None = None, chained: bool = False) -> None:
        from tessera_embeddings.inference import runner as runner_mod

        def _wait(*_a: object, **_k: object) -> tuple:
            raise RuntimeError("Only 0 / 8 actors initialized within 600s")

        with (
            patch.object(runner_mod, "InferenceActor") as actor,
            patch.object(runner_mod, "wait_for_actors", side_effect=_wait),
            patch.object(runner_mod, "ZarrWriter") as writer,
            patch("tessera_embeddings.inference.scheduling.time.sleep"),
            # WITHOUT THIS, THIS TEST BOOTS A REAL LOCAL RAY CLUSTER. `wait_for_actors`
            # raises above, but the actor factory has already handed back stand-ins, so
            # `run_inference`'s `finally` reaches `ray.kill(actor)` — and `ray.kill` is
            # wrapped in Ray's auto-init hook, which calls `ray.init()`. That init hashes and
            # uploads the whole working directory; the same hazard once ate ~60 GB of RAM
            # across three concurrent runs, and here it cost ~3 s, making this the
            # second-slowest test in the unit suite. `test_scheduling.py` patches `ray.kill`
            # in an autouse fixture for exactly this reason; this file needs it too.
            patch.object(runner_mod.ray, "kill"),
        ):
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

    def test_a_drought_that_kills_the_actor_wait_has_still_asked(self) -> None:
        """Without this the run reaches its hours-long timeout having never once asked for
        the fallback — precisely the starvation the feature exists to prevent.
        """
        published: list[int] = []
        self._run(published, chained=True)
        # The FIRST BATCH, not the 8-actor target: the batching policy must survive.
        assert published[0] == _config().initial_actor_request(8) == 4

    def test_a_static_run_asks_only_for_the_work_it_holds(self) -> None:
        published: list[int] = []
        self._run(published, chunks=[object(), object()])
        assert published[0] == 2

    def test_the_request_is_cleared_when_the_run_ends(self) -> None:
        """Machines a request holds are exempt from idle termination, so a floor left
        standing pins an idle GPU fleet through the assembly that follows.
        """
        published: list[int] = []
        self._run(published, chunks=[object(), object()])
        assert published[-1] == 0

    def test_every_session_that_gets_a_terminator_also_gets_the_publisher(self) -> None:
        """Structural, because two reviewers missed the same call site independently: the
        sequential flow starts inference from two places and only one was wired.
        """
        import ast
        import inspect

        from tessera_embeddings.orchestration.prefect.flows import fill_zones_sequential as mod

        sites = [
            node
            for node in ast.walk(ast.parse(inspect.getsource(mod)))
            if isinstance(node, ast.Call) and any(kw.arg == "on_actor_retire" for kw in node.keywords)
        ]
        assert len(sites) >= 2, "expected the shared session and the per-cell retry"
        for site in sites:
            assert "on_fleet_demand" in {kw.arg for kw in site.keywords}, f"line {site.lineno}"

    def test_a_ladder_that_opens_a_third_rung_is_trimmed_to_two(self) -> None:
        """The SSM worker ladder can open a third rung without naming it in
        `gpu_fallback_instance_types`, walking past both guards. Trimmed on the resolved
        plan rather than refused — dropping the whole mix would be worse than running on
        the best two.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:20,g5.2xlarge:20,g6.2xlarge:20")
        plan = fleet_mix.plan_from_resolved_config(config, BUDGET)
        assert plan is not None
        assert set(plan.ceilings) == {L40S, A10G}, "the worst ratio is the one dropped"
