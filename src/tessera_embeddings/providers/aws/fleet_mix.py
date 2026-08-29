"""State the GPU fleet's shape, so every instance pool is asked for at once.

Ray's autoscaler picks node types greedily and stops choosing one only when *that
type's* ``max_workers`` is exhausted. Our production rung packs a one-CPU one-GPU actor
more tightly than any 8 vCPU fallback, so it wins every unit of demand — and while
demand stays under its ceiling an open fallback is never asked for at all, however long
the primary has been refusing capacity.

Neither of Ray's escapes works: autoscaler v2 (the default since Ray 2.50) has no plugin
point for a scoring function, and its own capacity signal is both a tiebreak these rungs
never reach and populated only on a launch *timeout* rather than a refusal.

So this does not argue with the ranking. It states how many machines of each rung the
fleet should hold, through ``request_resources``, which v2 satisfies in the same
scheduling pass as ordinary actor demand.

Three properties of that pass are load-bearing, all measured on a live cluster rather
than read (evidence in ``temp/simultaneous-gpu-fill-plan.md``):

* an ask above a rung's ceiling provisions **nothing** — a partially infeasible
  constraint is discarded whole, so every bound here is required, not defensive;
* machines held by a request are exempt from idle termination, so the ask is recomputed
  from live counts every round and allowed to fall;
* an ask is a **floor, not a delivery** — AWS fills what it can and reports success, so
  monitoring must compare asked against live rather than count failures.

What this does NOT bound is everything Ray may launch: ordinary actor demand above the
requested shape is scheduled normally, so a fleet sized well past what the budget affords
can exceed it. That is the pre-existing limitation ``_apply_gpu_vcpu_budget`` documents.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

_LOG = logging.getLogger(__name__)

MARKER_PREFIX = "tessera_rung:"
"""Prefix of the custom resource identifying ONE on-demand GPU rung.

Ours rather than Ray's autofilled ``accelerator_type:<CARD>``, because the card is not
unique — the spot rung is the same ``g6e.xlarge``, so a card-named bundle could be
satisfied by interruptible capacity. Named by instance type, the vocabulary the
``gpu-worker-ladder`` SSM key already uses.
"""

DEFAULT_PROBE = 16
"""Machines beyond the live count to keep asking the best rung for.

A default, not a derivation. Asking its ceiling would reserve budget for machines AWS is
refusing; asking only what is live would never grow.
"""

_LIVE_READING_TTL_S = 5.0
"""How long a live node-count reading may be reused — counts move on EC2 timescales."""


@dataclass(frozen=True)
class GpuRung:
    """One on-demand GPU rung, priced for the vCPU quota it spends."""

    instance_type: str
    card: str
    vcpu_per_node: int
    gpus_per_node: int
    throughput_index: float

    @property
    def marker(self) -> str:
        """The resource a bundle names to demand THIS rung and no other."""
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

The order is a computation, not a preference: the quota is counted in vCPU and the cards
are not equally priced in it, so filling the best ratio first is the fractional-knapsack
greedy. Sizes are restricted on host RAM — the vCPU-matched ``xlarge`` of each family
carries 16 GiB against a measured ~17.7 GB per actor. Adding a card is a row here plus a
rung in the template. Throughput measured; see ``gpu-card-choice-2026_08.md``.
"""

RUNGS_BY_INSTANCE_TYPE: dict[str, GpuRung] = {r.instance_type: r for r in GPU_RUNGS}

if [r.value_per_vcpu for r in GPU_RUNGS] != sorted(
    (r.value_per_vcpu for r in GPU_RUNGS), reverse=True
):  # pragma: no cover — import-time guard
    raise RuntimeError("GPU_RUNGS must be ordered by value_per_vcpu, best first")


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

    Walks the open rungs best-value-first. Each ask is a TOTAL, not an increment, because
    ``request_resources`` states a shape the cluster must fit and Ray fits it on existing
    machines before launching any.

    Args:
        want_gpus: Single-GPU machines the run wants now. Bounding by real demand is what
            stops a standing ask buying machines the pool then idle-retires, which the ask
            would immediately re-buy.
        live_by_instance_type: Machines currently UP. Live rather than requested, because
            a refused launch spends no quota.
        ceilings: ``max_workers`` per instance type, for the OPEN rungs only.
        vcpu_budget: The cluster's G-and-VT allowance, or ``None`` for no budget term.
        max_total: The cluster-wide ``max_workers``, bounding the SUM, which the per-rung
            ceilings do not.
        probe: See :data:`DEFAULT_PROBE`.

    Returns:
        Instance type -> machines to request, for the open rungs only.
    """
    if want_gpus < 0 or probe < 0:
        raise ValueError(f"want_gpus and probe must be >= 0, got {want_gpus} and {probe}")
    if vcpu_budget is not None and vcpu_budget <= 0:
        raise ValueError(f"vcpu_budget must be > 0 when given, got {vcpu_budget}")

    open_rungs = [r for r in GPU_RUNGS if ceilings.get(r.instance_type, 0) > 0]
    if not open_rungs:
        return {}

    live = {r.instance_type: live_by_instance_type.get(r.instance_type, 0) for r in open_rungs}
    # Charge what is already live FIRST. Lowering an ask cannot remove a busy machine, so
    # pricing the asks alone let the live fleet ratchet past the budget round on round.
    committed = sum(live_by_instance_type.get(r.instance_type, 0) * r.vcpu_per_node for r in GPU_RUNGS)
    headroom = None if vcpu_budget is None else max(0, vcpu_budget - committed)
    want, total_left = want_gpus, max_total
    granted = dict.fromkeys(live, 0)

    def take(rung: GpuRung, up_to: int) -> int:
        """Grant ADDITIONAL machines toward a total of ``up_to``, spending every
        constraint as it goes.

        Additional, not total: the budgets below are what REMAINS after earlier grants, so
        comparing a total against them double-counts what this rung already holds and
        leaves that much of the quota permanently unspent.
        """
        nonlocal want, headroom, total_left
        t = rung.instance_type
        more = min(up_to - granted[t], want // rung.gpus_per_node, ceilings[t] - live[t] - granted[t])
        if headroom is not None:
            more = min(more, headroom // rung.vcpu_per_node)
        if total_left is not None:
            more = min(more, total_left - sum(granted.values()) - sum(live.values()))
        more = max(0, more)
        granted[t] += more
        want -= more * rung.gpus_per_node
        if headroom is not None:
            headroom -= more * rung.vcpu_per_node
        return more

    # PHASE 1 — a floor for every rung we have never obtained, taken before anything is
    # distributed. A rung with nothing running is a rung whose availability is UNKNOWN, and
    # the only way to find out is to keep asking; without this the best rung absorbs the
    # whole demand, quota or aggregate ceiling and the fallback is never requested at all.
    # It is keyed on the rung's OWN live count, not the primary's: an earlier version
    # dropped the reservation once the primary had any machine, on the reasoning that the
    # scheduler would correct it. The scheduler recomputes THIS function, so it never did.
    #
    # Only while the machines we hold cannot already cover the demand: at a draining tail
    # a probe would buy a machine for work that does not exist.
    floors = dict.fromkeys(live, 0)
    if want_gpus > sum(live[r.instance_type] * r.gpus_per_node for r in open_rungs):
        for rung in open_rungs:
            if live[rung.instance_type] == 0:
                floors[rung.instance_type] = take(rung, 1)

    # PHASE 2 — distribute what is left, best value per vCPU first. The primary grows by a
    # bounded probe rather than to its ceiling: a ceiling-sized ask reserves budget for
    # machines AWS is refusing, and an ask of exactly what is live would never grow.
    for index, rung in enumerate(open_rungs):
        take(rung, probe if index == 0 else ceilings[rung.instance_type])

    # An ask is a total. Capped by demand so it can fall BELOW the live count as a queue
    # drains — surplus machines then lose their protection and idle out, rather than being
    # held by a stale floor through the assembly that follows.
    # The floor is EXEMPT from that cap. It is one machine on a pool whose availability is
    # unknown, and capping it away is precisely how the fallback stopped being requested.
    asks: dict[str, int] = {}
    remaining = want_gpus
    for rung in open_rungs:
        t = rung.instance_type
        allowed = max(remaining // rung.gpus_per_node, floors[t])
        ask = max(0, min(live[t] + granted[t], allowed))
        if total_left is not None:
            ask = min(ask, total_left)
            total_left -= ask
        asks[t] = ask
        remaining = max(0, remaining - ask * rung.gpus_per_node)
    return asks


def bundles_for(asks: Mapping[str, int]) -> list[dict[str, int]]:
    """One bundle per machine, naming only that rung's marker.

    Each rung declares exactly one marker, so a bundle can be satisfied by one machine of
    that rung and by nothing else — not another rung, and not the head.
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
    """The open, addressable rungs and their ceilings, from the config Ray was given."""

    ceilings: dict[str, int]
    vcpu_budget: int | None
    max_total: int | None = None
    probe: int = DEFAULT_PROBE


def plan_from_resolved_config(config: Mapping[str, Any], vcpu_budget: int | None) -> GpuFleetMixPlan | None:
    """Read the open GPU rungs out of a RESOLVED cluster config.

    That config is the document handed to ``ray up``, so its ``max_workers`` are the
    ceilings the autoscaler enforces — reading them beats recomputing them from the ladder
    and budget, which is how two numbers drift apart.

    Returns ``None`` when fewer than two rungs are addressable, which makes the whole
    feature inert: with nothing to mix, a request would add nothing.
    """
    from tessera_embeddings.providers.aws.ray import GPU_WORKER_NODE_TYPE_PREFIX

    ceilings: dict[str, int] = {}
    for name, cfg in (config.get("available_node_types") or {}).items():
        instance_type = (cfg.get("node_config") or {}).get("InstanceType")
        if not name.startswith(GPU_WORKER_NODE_TYPE_PREFIX) or not isinstance(instance_type, str):
            continue
        rung = RUNGS_BY_INSTANCE_TYPE.get(instance_type)
        if rung is None:
            continue
        if int(cfg.get("max_workers", 0) or 0) <= 0:
            continue
        if instance_type in ceilings:
            raise RuntimeError(
                f"Two on-demand GPU node types offer {instance_type!r}, so a fleet-mix ask has "
                "no single target. Give them distinct instance types."
            )
        # A rung we cannot address is worse than no rung: the bundles would name a resource
        # nothing declares, so the fallback would sit idle while the run believed it had
        # asked. Custom templates predating the marker get today's behaviour, loudly.
        if (cfg.get("resources") or {}).get(rung.marker) != 1:
            _LOG.warning("GPU rung %s is open but declares no %s — excluded", name, rung.marker)
            continue
        ceilings[instance_type] = int(cfg["max_workers"])

    if len(ceilings) < 2:
        return None
    global_ceiling = config.get("max_workers")
    return GpuFleetMixPlan(
        ceilings=ceilings,
        vcpu_budget=vcpu_budget,
        max_total=global_ceiling if isinstance(global_ceiling, int) and global_ceiling > 0 else None,
    )


def plan_from_resolved_yaml(path: str | Path, vcpu_budget: int | None) -> GpuFleetMixPlan | None:
    """:func:`plan_from_resolved_config`, reading the resolved YAML off disk."""
    with Path(path).open() as handle:
        return plan_from_resolved_config(yaml.safe_load(handle), vcpu_budget)


def _ensure_gcs_client_for_request_resources(ray: ModuleType) -> None:
    """Make ``request_resources`` usable from a driver that connected to a live cluster.

    It reads the GCS address off Ray's *global* internal-KV client, which is a module
    global set during ``worker.connect()`` and cleared by any ``disconnect()``. A driver
    that reconnects with ``ignore_reinit_error=True`` returns early and never re-seeds it,
    so the call fails with ``AssertionError: GCS client is not available`` even though the
    driver is connected and everything else works — which is exactly what happened on the
    first end-to-end run, silently, because the whole fleet mix is best-effort.

    Re-seeding is idempotent and cheap. The address comes from the public runtime context.
    """
    from ray.experimental.internal_kv import _initialize_internal_kv, internal_kv_get_gcs_client

    if internal_kv_get_gcs_client() is not None:
        return
    # `ray._raylet` is a compiled extension, so mypy cannot see `GcsClient` on it.
    from ray._raylet import GcsClient  # type: ignore[attr-defined]

    _initialize_internal_kv(GcsClient(address=ray.get_runtime_context().gcs_address))


def publisher(plan: GpuFleetMixPlan) -> Callable[[int], None]:
    """A callable that publishes the fleet mix for a wanted machine count.

    **This owns the cluster's resource constraint.** v2 permits exactly one and
    ``request_resources`` replaces it wholesale, so a second caller anywhere would
    silently clobber the fleet's shape. Nothing else calls it; keep it that way.

    Cheap to call often: the live reading is cached briefly and an unchanged answer skips
    the round trip, so a settled fleet writes nothing.
    """
    last: dict[str, int] | None = None
    cached: tuple[float, dict[str, int]] | None = None

    def publish(want_gpus: int) -> None:
        nonlocal last, cached
        import ray
        from ray.autoscaler.sdk import request_resources

        _ensure_gcs_client_for_request_resources(ray)

        # Clearing must not depend on reading the cluster. It runs in a teardown path,
        # often against a cluster that is going away, and a failed live query there would
        # leave the previous floor standing over the assembly that follows.
        if want_gpus <= 0:
            if last != {}:
                request_resources(bundles=[])
                last = {}
                _LOG.info("GPU fleet mix cleared")
            return

        now = time.monotonic()
        if cached is None or now - cached[0] >= _LIVE_READING_TTL_S:
            totals = ray.cluster_resources()
            counts = {
                r.instance_type: int(totals.get(r.marker, 0)) for r in GPU_RUNGS if r.instance_type in plan.ceilings
            }
            cached = (now, counts)
        live = cached[1]
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
            "GPU fleet mix: want=%d live=%s -> %s", want_gpus, dict(sorted(live.items())), dict(sorted(asks.items()))
        )

    return publish


def publisher_for_resolved_yaml(
    resolved_yaml: str | None,
    vcpu_budget: int | None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> Callable[[int], None] | None:
    """Build the fleet-mix publisher for a cluster, or ``None`` if it has no mix.

    Never raises into the fill: a fleet that cannot state its shape still runs, it just
    runs the way it does today.
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
