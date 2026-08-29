"""State the GPU fleet's SHAPE, so both instance pools are asked for at once.

Ray's autoscaler picks node types with a greedy loop: score every eligible type,
take the single best, add one node of it, repeat. A type only stops winning when
its own ``max_workers`` is exhausted. Our production rung (``g6e.xlarge``, 4 vCPU)
packs a one-CPU one-GPU actor more tightly than any 8 vCPU fallback, so it wins
every unit of demand — and while demand stays under its ceiling, **the fallback is
never asked for at all, however long the primary has been refusing capacity.**

Measured on ``global-tessera-dev`` on 2026-08-28: 18 consecutive launch attempts,
every one for the L40S, every one refused, and **not one attempt at the A10G**
sitting open at a ceiling of 25.

**Neither of Ray's own escapes works.** Autoscaler v2 — which ``ray up`` has
started by default since Ray 2.50, and which every one of our clusters runs —
has no plugin point for a scoring function at all, so the ``RAY_AUTOSCALER_``
``UTILIZATION_SCORER`` hook that v1 honoured is dead code here. And v2's native
capacity signal is unreachable twice over: it is only the tiebreak AFTER the
utilisation score, which our rungs never tie on, and it is only populated on
``ALLOCATION_TIMEOUT`` while our refusals arrive as ``ALLOCATION_FAILED``.

So this module does not try to influence which type Ray prefers. It **states how
many of each we want** through ``request_resources``, which v2 satisfies in the
same scheduling pass as ordinary actor demand — making the preference irrelevant.

Two behaviours of that pass are load-bearing, and both were measured rather than
merely read:

* **An over-ask provisions NOTHING.** A partially infeasible constraint is
  discarded whole rather than filled as far as it goes. Asking for 30 against a
  ceiling of 25 produced zero machines in three minutes, where a feasible ask of
  10 produced six within two seconds. :func:`fleet_asks` clamps to the ceiling for
  that reason, and the clamp is not defensive.
* **Constraint-held machines are exempt from idle termination.** A stale ask
  therefore holds idle hardware indefinitely, which is why the ask is recomputed
  from live counts on every pass and is allowed to fall.

**One fallback at a time, enforced upstream.** Stating the mix removes Ray's node-name
tie for the machines this request covers, but the request is a floor: ordinary actor
demand above it is scored by Ray as before, and the supported fallbacks are identical to
that scorer. `_apply_gpu_fallback` therefore still refuses to open two at once. The
registry and this policy are n-ary and ready for the day that guard lifts; until then a
third rung cannot be starved by a dry second one, because there is no third rung.

**The budget bounds what this REQUESTS, not everything Ray may launch.** Ordinary actor
demand above the requested shape is still scheduled by Ray, so a fleet sized by
``num_actors`` well past what the budget affords can exceed it — the pre-existing
limitation `_apply_gpu_vcpu_budget` already documents, unchanged here. Closing it means
capping actor submissions to the planned capacity, which shrinks the fleet and is a
campaign-level decision rather than a property of this module.

**An ask is a floor, not a delivery.** Ray launches with ``MinCount=1``, so AWS
fills what it can and reports success: one measured call asked for six machines
and got one, with a launch-failure count of zero throughout. Anything watching
this must compare ASKED against LIVE — a failure count cannot see it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_LOG = logging.getLogger(__name__)

MARKER_PREFIX = "tessera_rung:"
"""Prefix of the custom resource that identifies ONE on-demand GPU rung.

Declared by us in ``cluster.yaml.template`` rather than reusing Ray's autofilled
``accelerator_type:<CARD>``, for two reasons. The card is **not unique**: the
on-demand and spot ``g6e.xlarge`` rungs both carry ``accelerator_type:L40S``, so a
bundle naming the card could be satisfied by the spot rung the moment anyone opens
it — quietly moving the fleet onto interruptible capacity. And the autofill depends
on one live ``DescribeInstanceTypes`` call wrapped in a bare ``except Exception``,
so a declared marker also removes a network dependency from the mechanism.

Named by INSTANCE TYPE because that is the vocabulary the ``gpu-worker-ladder`` SSM
key already uses; two vocabularies for one domain is how a template edit and an
operator's value drift apart.
"""

DEFAULT_PROBE = 16
"""How many machines beyond the live count to keep asking the best rung for.

A DEFAULT, not a derivation. Asking the primary for its full ceiling would reserve
budget for machines AWS is refusing — at the campaign's numbers, 404 of 840 vCPU
held for L40S that never arrive, cutting the A10G ask from 87 to 54. Asking only
for what is already up would never capture a recovery in supply. A bounded standing
request does both. Sized below the 25-instance launch batch so one probe cannot
fill a batch on its own.
"""


@dataclass(frozen=True)
class GpuRung:
    """One on-demand GPU rung, priced for the vCPU quota it spends."""

    instance_type: str
    card: str
    vcpu_per_node: int
    gpus_per_node: int
    throughput_index: float
    """Throughput relative to the best rung, measured on one cluster in the same
    minutes on the same work — see ``context_docs/design/gpu-card-choice-2026_08.md``."""

    @property
    def marker(self) -> str:
        """The custom resource a bundle names to demand THIS rung and no other."""
        return f"{MARKER_PREFIX}{self.instance_type}"

    @property
    def value_per_vcpu(self) -> float:
        """Throughput per unit of the quota that actually binds."""
        return self.throughput_index / self.vcpu_per_node


GPU_RUNGS: tuple[GpuRung, ...] = (
    GpuRung("g6e.xlarge", "L40S", 4, 1, 1.00),
    GpuRung("g5.2xlarge", "A10G", 8, 1, 0.46),
    GpuRung("g6.2xlarge", "L4", 8, 1, 0.32),
)
"""Every rung we will run, best value-per-vCPU first.

**The order is a computation, not a preference.** The G-and-VT quota is counted in
vCPU and the cards are not equally priced in it — 0.250, 0.0575 and 0.040 of an
L40S-equivalent per vCPU respectively. Filling the best ratio first until the budget
runs out is the fractional-knapsack greedy, which is optimal for a divisible budget.

Sizes are restricted on HOST RAM, which is why no ``xlarge`` fallback appears: the
vCPU-matched ``g5.xlarge`` and ``g6.xlarge`` carry 16 GiB against a measured ~17.7 GB
per-actor requirement, the exact shape that OOMed the loader on the earlier 16 GB
workers. Adding a card is a row here plus a rung in the template.
"""

_ORDER_ERROR = "GPU_RUNGS must be ordered by value_per_vcpu, best first"
if [r.value_per_vcpu for r in GPU_RUNGS] != sorted(
    (r.value_per_vcpu for r in GPU_RUNGS), reverse=True
):  # pragma: no cover — import-time guard
    raise RuntimeError(_ORDER_ERROR)

RUNGS_BY_INSTANCE_TYPE: dict[str, GpuRung] = {r.instance_type: r for r in GPU_RUNGS}


def fleet_asks(
    *,
    want_gpus: int,
    live_by_instance_type: Mapping[str, int],
    ceilings: Mapping[str, int],
    vcpu_budget: int | None = None,
    max_total: int | None = None,
    probe: int = DEFAULT_PROBE,
) -> dict[str, int]:
    """How many machines of each open rung to hold a standing request for.

    Walks the open rungs best-value-first, giving each what the remaining unmet
    demand and the remaining vCPU budget can pay for, bounded by its ceiling.

    The BEST open rung is additionally bounded to ``live + probe`` rather than to
    its ceiling. That is the whole of "primary first, fallback takes the rest": a
    ceiling-sized ask would reserve budget for machines that are being refused,
    and a live-sized ask would never grow.

    Args:
        want_gpus: GPUs the run actually wants right now — the smaller of the
            actor target and the outstanding work. Bounding by work is what stops
            a standing ask from buying machines the pool then idle-retires, which
            the ask would immediately re-buy.
        live_by_instance_type: Machines currently UP per instance type. Live rather
            than requested, because a refused launch spends no quota; reading it
            this way errs toward asking for more fallback, which is the right
            direction under a drought.
        max_total: The cluster-wide ``max_workers``. Bounds the SUM, which the per-rung
            ceilings do not: individually valid asks can exceed the aggregate, and Ray
            discards an infeasible constraint whole rather than filling part of it — so
            an uncapped sum does not merely overshoot, it provisions nothing at all.
        ceilings: ``max_workers`` per instance type, for the OPEN rungs only. Read
            from the resolved cluster config — an ask above a ceiling provisions
            nothing at all, so this bound is load-bearing.
        vcpu_budget: The cluster's G-and-VT vCPU allowance, or ``None`` for no
            budget term.
        probe: See :data:`DEFAULT_PROBE`.

    Returns:
        Instance type -> machines to request. Rungs absent from ``ceilings`` are
        absent here; a rung the budget cannot pay for maps to ``0``.
    """
    if want_gpus < 0:
        raise ValueError(f"want_gpus must be >= 0, got {want_gpus}")
    if probe < 0:
        raise ValueError(f"probe must be >= 0, got {probe}")
    if vcpu_budget is not None and vcpu_budget <= 0:
        raise ValueError(f"vcpu_budget must be > 0 when given, got {vcpu_budget}")

    open_rungs = [r for r in GPU_RUNGS if ceilings.get(r.instance_type, 0) > 0]
    if not open_rungs:
        return {}

    # An ask is a TOTAL, not an increment: `request_resources` says "the cluster must
    # be able to fit these bundles", and Ray fits them on existing machines before
    # launching any.
    #
    # So the budget is spent by what the fleet will HOLD, which is not the same as what
    # we ask for. Lowering an ask does not remove a busy machine, and ordinary actor
    # demand provisions nodes this request never sees. Pricing the asks alone let the
    # live fleet ratchet past the budget: at 51 L40S and 79 A10G the next round would
    # ask 67 L40S while all 79 A10G stayed up — 900 vCPU against a budget of 840, and
    # repeatable. So charge what is already live FIRST, and allocate only the headroom
    # that leaves.
    committed = sum(live_by_instance_type.get(r.instance_type, 0) * r.vcpu_per_node for r in GPU_RUNGS)
    headroom = None if vcpu_budget is None else max(0, vcpu_budget - committed)
    remaining_want = want_gpus
    remaining_total = max_total

    asks: dict[str, int] = {}
    for index, rung in enumerate(open_rungs):
        live = live_by_instance_type.get(rung.instance_type, 0)
        # How far this rung may GROW, before demand is considered.
        growth_bounds = [ceilings[rung.instance_type] - live]
        if index == 0:
            # The best open rung grows by a bounded probe rather than to its ceiling: a
            # ceiling-sized ask reserves budget for machines AWS is refusing, and an ask
            # of exactly what is live would never grow at all.
            growth_bounds.append(probe)
        if headroom is not None:
            growth_bounds.append(headroom // rung.vcpu_per_node)
        growth = max(0, min(growth_bounds))
        # Capped by the demand still unplaced, which is what lets an ask fall BELOW the
        # live count as a queue drains — surplus machines then lose their constraint
        # protection and idle out, instead of being held forever by a stale floor.
        ask = max(0, min(live + growth, remaining_want // rung.gpus_per_node))
        if remaining_total is not None:
            ask = min(ask, remaining_total)
            remaining_total = max(0, remaining_total - ask)
        asks[rung.instance_type] = ask
        remaining_want = max(0, remaining_want - ask * rung.gpus_per_node)
        if headroom is not None:
            headroom = max(0, headroom - max(0, ask - live) * rung.vcpu_per_node)
    return asks


def bundles_for(asks: Mapping[str, int]) -> list[dict[str, int]]:
    """Turn per-rung asks into the bundle list ``request_resources`` takes.

    One bundle per machine, naming only that rung's marker: each rung declares
    exactly one of its marker, so a bundle can be satisfied by one machine of that
    rung and by nothing else — not by another rung, and not by the head.
    """
    bundles: list[dict[str, int]] = []
    for instance_type, count in sorted(asks.items()):
        rung = RUNGS_BY_INSTANCE_TYPE.get(instance_type)
        if rung is None:
            raise KeyError(f"{instance_type!r} is not a known GPU rung")
        bundles.extend([{rung.marker: 1}] * max(0, count))
    return bundles


@dataclass(frozen=True)
class GpuFleetMixPlan:
    """The open rungs and their ceilings, read from the config Ray was given."""

    ceilings: dict[str, int]
    vcpu_budget: int | None
    max_total: int | None = None
    probe: int = DEFAULT_PROBE

    @property
    def has_fallback(self) -> bool:
        """True when more than one rung is open, so a mix is possible at all."""
        return len(self.ceilings) > 1


def plan_from_resolved_config(config: Mapping[str, Any], vcpu_budget: int | None) -> GpuFleetMixPlan | None:
    """Read the open GPU rungs out of a RESOLVED cluster config.

    The resolved config is the document actually handed to ``ray up``, so its
    ``max_workers`` are the ceilings the autoscaler enforces. Reading them rather
    than recomputing them from the ladder and the budget keeps one number in one
    place — a second computation of the same figure is how the two drift apart.

    Returns:
        ``None`` when fewer than two GPU rungs are open, which makes the whole
        feature inert: with nothing to mix, a constraint would add nothing.
    """
    from tessera_embeddings.providers.aws.ray import GPU_WORKER_NODE_TYPE_PREFIX

    ceilings: dict[str, int] = {}
    for name, cfg in (config.get("available_node_types") or {}).items():
        if not name.startswith(GPU_WORKER_NODE_TYPE_PREFIX):
            continue
        ceiling = int(cfg.get("max_workers", 0) or 0)
        instance_type = (cfg.get("node_config") or {}).get("InstanceType")
        if ceiling <= 0 or instance_type not in RUNGS_BY_INSTANCE_TYPE:
            continue
        if instance_type in ceilings:
            raise RuntimeError(
                f"Two on-demand GPU node types offer {instance_type!r}, so a fleet-mix ask "
                "has no single target. Give them distinct instance types."
            )
        ceilings[instance_type] = ceiling

    global_ceiling = config.get("max_workers")
    plan = GpuFleetMixPlan(
        ceilings=ceilings,
        vcpu_budget=vcpu_budget,
        max_total=int(global_ceiling) if isinstance(global_ceiling, int) and global_ceiling > 0 else None,
    )
    return plan if plan.has_fallback else None


def plan_from_resolved_yaml(path: str | Path, vcpu_budget: int | None) -> GpuFleetMixPlan | None:
    """:func:`plan_from_resolved_config`, reading the resolved YAML off disk."""
    with Path(path).open() as handle:
        return plan_from_resolved_config(yaml.safe_load(handle), vcpu_budget)


def publisher(plan: GpuFleetMixPlan) -> Callable[[int], None]:
    """A callable that publishes the fleet mix for a given wanted-GPU count.

    **This owns the cluster's resource constraint.** Autoscaler v2 permits exactly
    one, and ``request_resources`` replaces it wholesale, so a second caller
    anywhere in this codebase would silently clobber the fleet's shape. Nothing
    else calls it today; keep it that way.

    The returned callable is cheap to call often: it skips the round trip entirely
    when the computed asks have not moved, so a settled fleet writes nothing.
    """
    last: dict[str, int] | None = None

    def publish(want_gpus: int) -> None:
        nonlocal last
        import ray
        from ray.autoscaler.sdk import request_resources

        totals = ray.cluster_resources()
        live = {
            rung.instance_type: int(totals.get(rung.marker, 0))
            for rung in GPU_RUNGS
            if rung.instance_type in plan.ceilings
        }
        asks = fleet_asks(
            want_gpus=want_gpus,
            live_by_instance_type=live,
            ceilings=plan.ceilings,
            vcpu_budget=plan.vcpu_budget,
            max_total=plan.max_total,
            probe=plan.probe,
        )
        if asks == last:
            return
        request_resources(bundles=bundles_for(asks))
        last = asks
        _LOG.info(
            "GPU fleet mix: want=%d live=%s -> asking %s",
            want_gpus,
            dict(sorted(live.items())),
            dict(sorted(asks.items())),
        )

    return publish


def publisher_for_resolved_yaml(
    resolved_yaml: str | None,
    vcpu_budget: int | None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> Callable[[int], None] | None:
    """Build the per-rung fleet request for a cluster, or ``None`` if it has no mix.

    Reads the ceilings out of the RESOLVED cluster config — the document `ray up`
    was actually handed — so the numbers the autoscaler enforces and the numbers we
    price against are the same numbers. Returns ``None`` when fewer than two GPU
    rungs are open, which is the shipped default and makes this inert.

    Never raises into the fill: a fleet that cannot state its shape still runs, it
    just runs the way it does today.
    """
    if not resolved_yaml:
        return None
    try:
        plan = plan_from_resolved_yaml(resolved_yaml, vcpu_budget)
    except Exception:
        log.warning("Could not read the GPU fleet mix from %s; continuing without it", resolved_yaml, exc_info=True)
        return None
    if plan is None:
        return None
    log.info("GPU fleet mix active: ceilings=%s vcpu_budget=%s", plan.ceilings, vcpu_budget)
    return publisher(plan)
