"""Opening a second GPU rung, and pricing the ceilings the vCPU quota allows.

This covers only the OPENING half. What makes the autoscaler actually use an open
rung is the standing per-rung request in `fleet_mix`, tested in `test_fleet_mix.py`
— opening a rung on its own leaves it idle however long the primary refuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tessera_embeddings.providers.aws import ray as tray

TEMPLATE = Path(tray.__file__).parent / "cluster.yaml.template"

PRODUCTION = "gpu-workers-ondemand"
A10G = "gpu-workers-ondemand-a10g-2xl"


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
        assert tray._apply_gpu_fallback(config, ["g5.2xlarge"]) == [A10G]
        types = config["available_node_types"]
        assert types[A10G]["max_workers"] == types[PRODUCTION]["max_workers"] == 500

    def test_follows_a_ladder_that_narrows_production(self) -> None:
        """Applied after the ladder, so a campaign running 250 per cluster gets a 250
        fallback rather than the template's wider default.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:250")
        tray._apply_gpu_fallback(config, ["g5.2xlarge"])
        assert config["available_node_types"][A10G]["max_workers"] == 250

    def test_refuses_an_unknown_card(self) -> None:
        """A typo would otherwise leave the fleet quietly unable to fail over, and that
        is only discovered during the capacity event the feature exists for.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        with pytest.raises(RuntimeError, match="Unsupported GPU fallback instance type"):
            tray._apply_gpu_fallback(config, ["p5.48xlarge"])

    def test_refuses_when_no_production_rung_is_open(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        for cfg in config["available_node_types"].values():
            cfg["max_workers"] = 0
        with pytest.raises(RuntimeError, match="nothing to fall back FROM"):
            tray._apply_gpu_fallback(config, ["g5.2xlarge"])

    def test_every_known_card_has_a_rung_in_the_template(self) -> None:
        """`GPU_FALLBACK_CARDS` and the template must not drift: a card that names an
        instance type nothing ships is a failover that raises when it is needed.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        offered = {c["node_config"]["InstanceType"] for c in config["available_node_types"].values()}
        assert set(tray.GPU_FALLBACK_INSTANCE_TYPES) <= offered


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
        tray._apply_gpu_fallback(config, ["g5.2xlarge"], 1000)
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

    def test_a_multi_gpu_rung_is_priced_per_instance(self) -> None:
        """`max_workers` counts instances, not cards.

        Pricing per GPU reads a 96-CPU/8-GPU rung as 12 vCPU, so a 960-vCPU budget would
        permit 80 instances -- 7,680 vCPU, eight times the budget. Invisible while every
        shipped rung is single-GPU, which is exactly why it needs a test: the wider rungs
        arrive in the follow-up PR.
        """
        rung = {"resources": {"CPU": 96, "GPU": 8}, "node_config": {"InstanceType": "g6e.48xlarge"}}
        assert tray._vcpu_per_node(rung) == 96
        assert tray._vcpu_budget_ceiling(rung, 960) == 10  # 10 x 96 = 960, exactly

    def test_the_budget_narrows_a_wider_ladder(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:500")
        tray._apply_gpu_fallback(config, ["g5.2xlarge"], 1000)
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
        tray._apply_gpu_fallback(config, ["g5.2xlarge"], 1000)
        types = config["available_node_types"]
        assert types[PRODUCTION]["max_workers"] == 50, "the deliberate cap must stand"
        assert types[A10G]["max_workers"] == 125

    def test_refuses_a_nonpositive_budget(self) -> None:
        config = yaml.safe_load(TEMPLATE.read_text())
        with pytest.raises(ValueError, match="must be > 0"):
            tray._apply_gpu_fallback(config, ["g5.2xlarge"], 0)

    def test_refuses_a_rung_it_cannot_price(self) -> None:
        """A rung with no declared resources cannot be costed against a vCPU budget, and
        guessing would size the fleet against a fiction.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        config["available_node_types"][PRODUCTION].pop("resources")
        with pytest.raises(RuntimeError, match="Cannot price node type"):
            tray._apply_gpu_fallback(config, ["g5.2xlarge"], 1000)

    @pytest.mark.parametrize("budget", [1, 4], ids=["seats-nothing", "seats-production-only"])
    def test_a_budget_too_small_to_seat_a_node_is_refused(self, budget: int) -> None:
        """A budget of 5 would open one 8-vCPU node and overspend the caller's stated quota by
        60%; 4 seats one production GPU and no fallback GPU, the same overspend narrower.

        The caller stated a quota, not a preference. Refusing here is cheaper than
        discovering it as an unexplained `InstanceLimitExceeded` during a capacity event.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        with pytest.raises(ValueError, match="cannot afford a single"):
            tray._apply_gpu_fallback(config, ["g5.2xlarge"], budget)


class TestCeilingsAreDerivedFromProductionOnly:
    """Fallback rungs share the node-type prefix, which made them look like production."""

    def test_a_ladder_opened_fallback_does_not_supply_the_production_ceiling(self) -> None:
        """`g6e.xlarge:250,g5.2xlarge:50` opens both. The 50 was deliberate, and matching
        the fallback to "the widest open rung" must not read the fallback's own ceiling.
        """
        config = yaml.safe_load(TEMPLATE.read_text())
        tray._apply_gpu_worker_ladder(config, "g6e.xlarge:250,g5.2xlarge:50")
        tray._apply_gpu_fallback(config, ["g5.2xlarge"])
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
            tray._apply_gpu_fallback(config, ["g5.2xlarge"])
