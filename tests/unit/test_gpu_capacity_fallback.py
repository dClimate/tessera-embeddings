"""Falling the fleet over to a second GPU card when the first has no capacity.

Two halves, tested together because neither is any use alone: opening a fallback rung
(`ray.py::_apply_gpu_fallback`) and making the autoscaler willing to use it
(`autoscaler_scorer`). The scoring tests run Ray's REAL `get_nodes_for` against the
real cluster template, so what they assert is what the autoscaler would do.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from ray.autoscaler._private.aws import node_provider as np_mod
from ray.autoscaler._private.node_provider_availability_tracker import (
    NodeAvailabilityRecord,
    NodeAvailabilitySummary,
    UnavailableNodeInformation,
)
from ray.autoscaler._private.resource_demand_scheduler import (
    _default_utilization_scorer,
    get_nodes_for,
)

from tessera_embeddings.providers.aws import ray as tray
from tessera_embeddings.providers.aws.autoscaler_scorer import (
    SCORER_PATH,
    capacity_aware_scorer,
)

TEMPLATE = Path(tray.__file__).parent / "cluster.yaml.template"
EC2_TYPES = Path(__file__).parents[1] / "fixtures" / "ec2_describe_instance_types_gpu.json"

#: What one InferenceActor asks Ray for. Every score below is relative to this.
ACTOR_BUNDLE = {"CPU": 1, "GPU": 1}

PRODUCTION = "gpu-workers-ondemand"
A10G = "gpu-workers-ondemand-a10g-2xl"


def _node_types(ladder: str | None = None, fallback: list[str] | None = None) -> dict:
    """The template's node types as RAY sees them: ladder applied, resources filled."""
    config = yaml.safe_load(TEMPLATE.read_text())
    if ladder:
        tray._apply_gpu_worker_ladder(config, ladder)
    if fallback:
        tray._apply_gpu_fallback(config, fallback)
    with patch.object(np_mod, "list_ec2_instances", return_value=json.loads(EC2_TYPES.read_text())):
        config = np_mod.AWSNodeProvider.fillout_available_node_types_resources(config)
    return {n: c for n, c in config["available_node_types"].items() if c["max_workers"] > 0}


def _refused(*node_types: str, category: str = "InsufficientInstanceCapacity") -> NodeAvailabilitySummary:
    """A summary in which each named type was last seen refused for ``category``."""
    return NodeAvailabilitySummary(
        node_availabilities={
            name: NodeAvailabilityRecord(
                node_type=name,
                is_available=False,
                last_checked_timestamp=1.0,
                unavailable_node_information=UnavailableNodeInformation(category=category, description="-"),
            )
            for name in node_types
        }
    )


def _launched(summary: NodeAvailabilitySummary, *, scorer=capacity_aware_scorer, demand: int = 250) -> dict[str, int]:
    """What Ray would launch for ``demand`` actors, given that availability picture."""
    chosen, _ = get_nodes_for(
        _node_types("g6e.xlarge:250", fallback=["A10G"]),
        {},
        "head",
        9999,
        [ACTOR_BUNDLE] * demand,
        partial(scorer, node_availability_summary=summary),
    )
    return dict(chosen)


class TestFailoverOnCapacityRefusal:
    """The behaviour the feature exists for, against Ray's own scheduler."""

    def test_the_default_scorer_never_moves_off_a_refusing_rung(self) -> None:
        """The defect this replaces, asserted so the fix cannot be quietly reverted.

        Ray's scorer takes a `node_availability_summary` and never reads it, so the
        production rung stays top-scored however often it refuses — and an open
        fallback sitting under it is never reached.
        """
        assert _launched(_refused(PRODUCTION), scorer=_default_utilization_scorer) == {PRODUCTION: 250}

    def test_production_is_preferred_while_it_has_capacity(self) -> None:
        """The fallback must never displace the cheaper card just by being open."""
        assert _launched(_refused()) == {PRODUCTION: 250}

    def test_a_capacity_refusal_moves_the_fleet_to_the_fallback(self) -> None:
        assert _launched(_refused(PRODUCTION)) == {A10G: 250}

    @pytest.mark.parametrize("category", ["RequestLimitExceeded", "InstanceLimitExceeded", "Unsupported"])
    def test_only_capacity_refusals_move_it(self, category: str) -> None:
        """A throttle clears in seconds and a quota refusal gets WORSE on the fallback,
        which spends twice the vCPU per GPU. Neither is a reason to move the fleet.
        """
        assert _launched(_refused(PRODUCTION, category=category)) == {PRODUCTION: 250}

    def test_every_rung_refused_still_launches_something(self) -> None:
        """Demotion, not exclusion. With nothing available the ordering is unchanged and
        the autoscaler keeps trying — this can slow a scale-up, never wedge it.
        """
        assert _launched(_refused(PRODUCTION, A10G)) == {PRODUCTION: 250}

    def test_recovery_needs_no_operator_action(self) -> None:
        """Ray clears the record on the next successful launch of that type, so the
        fleet returns to the cheaper card on its own.
        """
        assert _launched(_refused(PRODUCTION)) == {A10G: 250}
        assert _launched(_refused()) == {PRODUCTION: 250}


class TestApplyGpuFallback:
    """Opening the rungs, which is the half the scorer cannot do for itself."""

    def test_absent_by_default_and_changes_nothing(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        before = yaml.safe_dump(config, sort_keys=True)
        assert tray._apply_gpu_fallback(config, []) == []
        assert yaml.safe_dump(config, sort_keys=True) == before

    def test_opens_the_named_card_at_the_production_ceiling(self) -> None:
        """Matched to production so either card can carry the whole fleet. It does not
        make the fleet bigger — the global ceiling and inference demand still bound it.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        assert tray._apply_gpu_fallback(config, ["A10G"]) == [A10G]
        types = config["available_node_types"]
        assert types[A10G]["max_workers"] == types[PRODUCTION]["max_workers"] == 500

    def test_follows_a_ladder_that_narrows_production(self) -> None:
        """Applied after the ladder, so a campaign running 250 per cluster gets a 250
        fallback rather than the template's wider default.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:250")
        tray._apply_gpu_fallback(config, ["A10G"])
        assert config["available_node_types"][A10G]["max_workers"] == 250

    def test_refuses_an_unknown_card(self) -> None:
        """A typo would otherwise leave the fleet quietly unable to fail over, and that
        is only discovered during the capacity event the feature exists for.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        with pytest.raises(RuntimeError, match="Unknown GPU fallback card"):
            tray._apply_gpu_fallback(config, ["H100"])

    def test_refuses_when_no_production_rung_is_open(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        for cfg in config["available_node_types"].values():
            cfg["max_workers"] = 0
        with pytest.raises(RuntimeError, match="nothing to fall back FROM"):
            tray._apply_gpu_fallback(config, ["A10G"])

    def test_every_known_card_has_a_rung_in_the_template(self) -> None:
        """`GPU_FALLBACK_CARDS` and the template must not drift: a card that names an
        instance type nothing ships is a failover that raises when it is needed.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        offered = {c["node_config"]["InstanceType"] for c in config["available_node_types"].values()}
        assert set(tray.GPU_FALLBACK_CARDS.values()) <= offered


class TestVcpuBudget:
    """Ceilings priced in vCPU, because the quota is.

    The G-and-VT quota counts vCPU, not cards, and the rungs are not equally priced --
    the fallback sizes are 8 vCPU per GPU against production's 4. A ceiling expressed in
    NODES therefore means a different quota bill depending on which card wins.
    """

    def test_each_rung_is_ceilinged_at_what_the_budget_affords_it(self) -> None:
        """1,000 vCPU is 250 cards at 4 vCPU/GPU and 125 at 8 -- the honest statement of
        "as many of this card as the quota allows".
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_fallback(config, ["A10G"], 1000)
        types = config["available_node_types"]
        assert types[PRODUCTION]["max_workers"] == 250
        assert types[A10G]["max_workers"] == 125

    def test_the_budget_reprices_production_too(self) -> None:
        """Not only the fallback. A budget that bound one rung and not the other would
        describe a fleet nobody asked for.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_fallback(config, [], 1000)
        assert config["available_node_types"][PRODUCTION]["max_workers"] == 250

    def test_each_pure_fleet_lands_exactly_on_the_budget(self) -> None:
        """What the feature actually guarantees: a fleet entirely on either card spends
        the budget and no more. A MIXTURE can still exceed it -- Ray's ceilings count
        nodes and carry no weight -- and that is stated rather than tested, because it
        is a property of Ray's scheduler, not of this code.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_fallback(config, ["A10G"], 1000)
        types = config["available_node_types"]
        for name in (PRODUCTION, A10G):
            res = types[name]["resources"]
            assert types[name]["max_workers"] * (res["CPU"] // res["GPU"]) == 1000

    def test_absent_budget_keeps_the_production_ceiling(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_fallback(config, ["A10G"])
        types = config["available_node_types"]
        assert types[A10G]["max_workers"] == types[PRODUCTION]["max_workers"] == 500

    def test_the_budget_narrows_a_wider_ladder(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:500")
        tray._apply_gpu_fallback(config, ["A10G"], 1000)
        assert config["available_node_types"][PRODUCTION]["max_workers"] == 250

    def test_the_budget_never_widens_a_deliberate_cap(self) -> None:
        """Capping the production rung at what it can actually BE SUPPLIED is what
        pushes surplus demand onto the fallback -- and it is the only lever that works
        when the production card is trickling rather than refusing outright, because a
        partially filled launch counts as a success and never marks the type
        unavailable. A budget that raised that cap back up would silently undo it.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:50")
        tray._apply_gpu_fallback(config, ["A10G"], 1000)
        types = config["available_node_types"]
        assert types[PRODUCTION]["max_workers"] == 50, "the deliberate cap must stand"
        assert types[A10G]["max_workers"] == 125

    def test_refuses_a_nonpositive_budget(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        with pytest.raises(ValueError, match="must be > 0"):
            tray._apply_gpu_fallback(config, ["A10G"], 0)

    def test_refuses_a_rung_it_cannot_price(self) -> None:
        """A rung with no declared resources cannot be costed against a vCPU budget, and
        guessing would size the fleet against a fiction.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        config["available_node_types"][PRODUCTION].pop("resources")
        with pytest.raises(RuntimeError, match="Cannot price node type"):
            tray._apply_gpu_fallback(config, ["A10G"], 1000)

    def test_a_budget_too_small_to_seat_a_node_is_refused(self) -> None:
        """REPLACES a test that asserted a floor of 1, which was the defect: a budget of
        5 would open one 8-vCPU node and overspend the caller's stated quota by 60%.

        The caller stated a quota, not a preference. Refusing here is cheaper than
        discovering it as an unexplained `InstanceLimitExceeded` during a capacity event.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        with pytest.raises(ValueError, match="cannot afford a single"):
            tray._apply_gpu_fallback(config, ["A10G"], 1)

    def test_a_budget_that_seats_production_but_not_the_fallback_is_refused(self) -> None:
        """4 vCPU seats one production GPU and zero fallback GPUs. Opening the fallback
        anyway would be the same overspend by a narrower margin.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        with pytest.raises(ValueError, match="cannot afford a single"):
            tray._apply_gpu_fallback(config, ["A10G"], 4)


class TestTheCampaignRestartConfiguration:
    """The exact numbers the 2026-08-28 restart dispatches with, pinned.

    Not a unit of behaviour -- a guard. These ceilings are the campaign's contract with
    a 10,000 vCPU account quota, and every term that decides them lives somewhere this
    test can see: the ladder value, the budget, and the two rungs' declared vCPU per GPU.
    A template edit that changed any of them would otherwise re-shape the production
    fleet with nothing failing.
    """

    LADDER = "g6e.xlarge:101"
    VCPU_BUDGET = 840
    CLUSTERS = 10
    ACCOUNT_VCPU_QUOTA = 10_000
    #: What AWS has actually been supplying per cluster while capacity is short. The
    #: ceiling is a reservation limit, not a promise -- see the fleet arithmetic below.
    OBSERVED_L40S_PER_CLUSTER = 39

    def _ceilings(self) -> dict[str, int]:
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, self.LADDER)
        tray._apply_gpu_fallback(config, ["A10G"], self.VCPU_BUDGET)
        return {n: c["max_workers"] for n, c in config["available_node_types"].items()}

    def test_it_yields_101_l40s_and_105_a10g(self) -> None:
        ceilings = self._ceilings()
        assert ceilings[PRODUCTION] == 101
        assert ceilings[A10G] == 105

    def test_the_realistic_fleet_fits_the_account_quota(self) -> None:
        """With the L40S trickling at ~39 per cluster, the A10G ceiling is what decides
        the bill. 39x4 + 105x8 = 996 vCPU per cluster, 9,960 across ten -- inside 10,000
        BY DESIGN rather than by AWS refusing the last few hundred launches.
        """
        ceilings = self._ceilings()
        per_cluster = self.OBSERVED_L40S_PER_CLUSTER * 4 + ceilings[A10G] * 8
        assert per_cluster * self.CLUSTERS <= self.ACCOUNT_VCPU_QUOTA
        assert per_cluster * self.CLUSTERS == 9_960

    def test_the_l40s_ceiling_leaves_room_to_grow(self) -> None:
        """101 is deliberately well above the ~39 being supplied. The ceiling costs
        nothing unclaimed, and an L40S actor is half the quota of an A10G one -- so any
        recovery in supply converts straight into more actors per vCPU.
        """
        assert self._ceilings()[PRODUCTION] > self.OBSERVED_L40S_PER_CLUSTER

    def test_a_fully_supplied_fleet_would_exceed_the_quota(self) -> None:
        """Stated rather than guarded, because it is the accepted limitation: ceilings
        count nodes and cannot be jointly weighted, so both rungs full is 1,244 vCPU per
        cluster. AWS refuses with `InstanceLimitExceeded` at the account quota, and the
        scorer deliberately does NOT treat that as a reason to fall back further.
        """
        ceilings = self._ceilings()
        both_full = ceilings[PRODUCTION] * 4 + ceilings[A10G] * 8
        assert both_full * self.CLUSTERS > self.ACCOUNT_VCPU_QUOTA


class TestCeilingsAreDerivedFromProductionOnly:
    """Fallback rungs share the node-type prefix, which made them look like production."""

    def test_a_ladder_opened_fallback_does_not_supply_the_production_ceiling(self) -> None:
        """`g6e.xlarge:250,g5.2xlarge:50` opens both. The 50 was deliberate, and matching
        the fallback to "the widest open rung" must not read the fallback's own ceiling.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:250,g5.2xlarge:50")
        tray._apply_gpu_fallback(config, ["A10G"])
        assert config["available_node_types"][A10G]["max_workers"] == 50, (
            "an explicit ladder ceiling on the fallback must not be widened"
        )

    def test_an_all_fallback_ladder_has_nothing_to_fall_back_from(self) -> None:
        """The guard read "any open rung with the prefix", which a fallback satisfies --
        so a ladder opening ONLY the fallback passed a check whose whole point is that a
        production rung exists to fail over from.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g5.2xlarge:50")
        with pytest.raises(RuntimeError, match="no PRODUCTION GPU rung is open"):
            tray._apply_gpu_fallback(config, ["A10G"])


class TestOnlyOneCardAtATime:
    """Ray cannot express a preference between two equally-scored node types."""

    def test_two_cards_are_refused(self) -> None:
        """Both fallbacks are 8 vCPU / 1 GPU with identical declared resources, so Ray
        scores them equally and breaks the tie on node-type NAME, descending -- the L4
        would win every request despite being measurably the slower card.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        with pytest.raises(RuntimeError, match="Only one GPU fallback card"):
            tray._apply_gpu_fallback(config, ["A10G", "L4"])

    def test_the_same_card_twice_is_not_two_cards(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        assert tray._apply_gpu_fallback(config, ["A10G", "A10G"]) == [A10G, A10G]


class TestScorerInstallation:
    """The scorer only works if Ray can find it on the head node."""

    def test_ray_resolves_the_dotted_path(self) -> None:
        """Ray loads this with `load_function_or_class` inside the autoscaler's
        constructor. A path that does not resolve raises THERE, and the cluster then
        does not scale at all — so this is the check that the constant is not a guess.
        """
        from ray.autoscaler._private.loader import load_function_or_class

        assert load_function_or_class(SCORER_PATH) is capacity_aware_scorer

    def test_the_env_var_is_the_one_ray_reads(self) -> None:
        from ray.autoscaler._private.constants import AUTOSCALER_UTILIZATION_SCORER_KEY

        assert set(tray.GPU_FALLBACK_SCORER_ENV) == {AUTOSCALER_UTILIZATION_SCORER_KEY}
        assert tray.GPU_FALLBACK_SCORER_ENV[AUTOSCALER_UTILIZATION_SCORER_KEY] == SCORER_PATH

    def test_the_scorer_rides_on_the_heads_ray_start(self) -> None:
        """Head only, and on the `ray start` line itself: the autoscaler is a child of
        that process, and Ray runs each command entry in its own shell, so an `export`
        on its own line would reach nothing.
        """
        commands = yaml.safe_load(TEMPLATE.read_text())["head_start_ray_commands"]
        placed = tray._pace_ray_start(commands, tray.GPU_FALLBACK_SCORER_ENV)
        start = next(c for c in placed if "ray start" in c)
        assert f"RAY_AUTOSCALER_UTILIZATION_SCORER={SCORER_PATH} ray start" in start
        assert "PYTHONPATH=$HOME/tessera-embeddings" in start, "the scorer must be importable"
