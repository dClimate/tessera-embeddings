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
