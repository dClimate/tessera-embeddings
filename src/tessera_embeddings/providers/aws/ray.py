"""AWS-backed Ray cluster provisioning.

The :func:`ray_cluster` context manager handles SSM config resolution,
SSH key materialisation, ``ray up`` / ``ray down`` subprocess calls,
CloudWatch agent configuration, and (optionally) code-tarball sync to
S3. No ML logic, no Prefect tasks — orchestration code imports this as a
plain context manager.

Adapted from the reference repo's ``infra/ray/cluster.py``. Notable
differences for the open-source release:

* Account-bound IDs (security groups, AMIs, IAM roles, subnets, key
  pairs, EC2 tags) are read from SSM under a configurable prefix —
  there are no hardcoded ``/yield/ray/`` paths.
* The dev/prod bucket toggle from the reference repo is removed; bucket
  names are caller-supplied.
* Source-code tarball sync is opt-in (``sync_source_path``); the default
  assumes a pre-baked AMI and skips the sync step entirely.
* CloudWatch log-group name is a parameter, not a hardcoded
  ``/ec2/yield/ray``.

# NOTE — cancellation hook dependency:
# :func:`terminate_ray_instances_by_tag` and :func:`cleanup_ray_tempfiles` are
# intended to be wired into the orchestrator's on-cancellation/on-crashed
# hooks. Those hooks are the real last line of defence: the autoscaler idle
# timeout only drains workers above a node type's ``min_workers`` floor and
# NEVER terminates the head node, so an untorn-down cluster runs until
# someone terminates it by tag. See gotchas.md ("Teardown").
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import subprocess
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import boto3
import ray
import yaml

from tessera_embeddings.providers.aws import autoscaler_scorer

DEFAULT_CLUSTER_TEMPLATE = Path(__file__).parent / "cluster.yaml.template"
"""Path to the cluster YAML template shipped with this provider."""

DEFAULT_CLOUDWATCH_TEMPLATE = Path(__file__).parent / "cloudwatch-agent.json.tpl"
"""Path to the CloudWatch agent JSON template shipped with this provider."""

DEFAULT_SSM_PREFIX = "/tessera/ray/"
"""Default SSM Parameter Store prefix for Ray cluster resource IDs.

Production deployments override this — there is no global default that
makes sense across organisations. Configure the prefix to match wherever
you've stored the EC2 resource IDs your Ray nodes need.
"""

DEFAULT_CLOUDWATCH_LOG_GROUP = "/ec2/tessera/ray"
"""Default CloudWatch log group for Ray agent logs."""

RAY_DOWN_TIMEOUT_S = 300
"""Upper bound (seconds) on a cancellation-time ``ray down``.

``ray down`` SSHes into the head to tear the cluster down; if the head is
unreachable or the CLI wedges it can block forever. A cancellation hook that
hangs here never reaches the tag-based EC2 termination fallback, leaving GPU
workers running and billed — so the call is bounded and a timeout falls through
to tag termination. Generous enough for a normal teardown (~1-2 min)."""

PROJECT_TAG_VALUE = "tessera-embeddings"
"""Value of the ``Project`` EC2 tag stamped on every Ray node.

The deployment's runner IAM role conditions ``ec2:TerminateInstances`` on
``aws:ResourceTag/Project`` equal to this value, so that ``ray down`` (which
terminates nodes using the driver's credentials) and the
:func:`terminate_ray_instances_by_tag` fallback are both authorised. Without
this tag every teardown terminate is IAM-denied and the instances leak. Keep
this in lockstep with the deployment's IAM condition (yield CDK:
``ray_inference.py`` ``RayEc2Terminate``)."""

_REQUIRED_SSM_KEYS = frozenset(
    {"security-group-id", "instance-profile-arn", "private-subnet-ids", "key-pair-name", "key-pair-id"}
)

LAUNCH_PACING_CLIENT_ENV = {
    # Client-side rate limiting on whatever EC2 client the process builds. Ray sets only
    # `max_attempts` on its launch client's botocore Config and never `mode`, so the mode
    # is still resolved from the environment — this is the one place we can reach inside
    # Ray's launch path without owning its retry loop. Adaptive mode paces sends through a
    # token bucket that narrows when EC2 answers with a throttle and widens again on
    # success. The INVARIANT that makes it safe to hand a fleet's growth to: botocore
    # floors that bucket's fill rate, so a paced client never stops asking.
    #
    # Safe for ANY launching process because it only spaces attempts out — it never
    # removes one. That is what separates it from the autoscaler settings below.
    "AWS_RETRY_MODE": "adaptive",
}
"""Pacing that is safe wherever instances are launched, including one-shot launches."""

LAUNCH_PACING_AUTOSCALER_ENV = {
    # Instances per RunInstances call. The request quota is a rate over CALLS, not
    # over instances, so asking for more per call buys fleet at a fixed price in
    # quota. MinCount stays 1, so a call the region can only partly fill returns what
    # it can rather than failing — a larger batch does not trade capacity for volume.
    # It also narrows the burst: Ray sizes its launcher-thread pool as
    # ceil(max_concurrent_launches / max_launch_batch), so a bigger batch is fewer
    # threads calling at once.
    "AUTOSCALER_MAX_LAUNCH_BATCH": "25",
    # Ray retries a failed launch in a loop with no delay between attempts, rotating
    # to the next subnet each time, for max(this, subnet count) attempts. A rotation
    # is spent whatever the error was, so a throttled attempt consumes an availability
    # zone that a capacity-short attempt still needs. Setting this below the subnet
    # count leaves exactly one pass over the zones: capacity failover is preserved and
    # the surplus no-delay attempts are not made.
    "BOTO_CREATE_MAX_RETRIES": "1",
    # Seconds between autoscaler passes; Ray's default is 5. This is the only setting that
    # bounds how OFTEN a cluster asks, as opposed to how much it asks for per call — and the
    # request quota is a rate over calls. The two settings above make each call carry more and
    # waste fewer attempts; neither reduces the calls per minute, so at fleet width the
    # autoscalers still overshoot the account's shared bucket together.
    #
    # Why a fixed slowdown rather than backoff: botocore's adaptive mode is congestion control,
    # and it converges only for a client that can see the whole pipe. Every cluster runs its own
    # autoscaler with its own controller against ONE account-wide bucket, so each settles at a
    # rate that is reasonable alone and collectively is not. A longer loop is the one lever that
    # composes, because it divides every participant's rate by the same factor.
    #
    # Sized against the bucket rather than picked: at the default the fleet's aggregate call rate
    # runs a small multiple of what the bucket admits, and this divides it by three. It costs
    # latency in the autoscaler's OTHER duties — noticing a dead node, acting on a resource
    # demand — which is why it is not longer.
    "AUTOSCALER_UPDATE_INTERVAL_S": "15",
}
"""Tuning for the AUTOSCALER's launch loop, and for nothing else.

Separate from :data:`LAUNCH_PACING_CLIENT_ENV` because these two settings change
how many attempts a launch gets, not merely how they are spaced. That is safe for
a worker request, which the autoscaler will make again on its next cycle, and
unsafe for a launch that happens once. Give this to a process that runs the
autoscaler; give the client pacing to anything else that launches.

The account's RunInstances quota is a token bucket — a small burst capacity,
refilled at a fixed rate — and it is not adjustable. Concurrent clusters share
one bucket, so several autoscalers requesting at the same moment drain it and
everything behind them is refused; those refusals are retried and drain it
again. None of that loop is ours: the call is made by Ray's AWS node provider
and the retry around it is Ray's. What we can set is the environment the
autoscaler process runs in, and every name here is one Ray or botocore reads
from the environment.
"""

LAUNCH_PACING_ENV = {**LAUNCH_PACING_CLIENT_ENV, **LAUNCH_PACING_AUTOSCALER_ENV}
"""Everything the head node wants: it both launches and hosts the autoscaler.

Applied only when a caller asks for it (``launch_pacing=True``). Default-off
because it changes how a live fleet grows.
"""

GPU_WORKER_LADDER_SSM_KEY = "gpu-worker-ladder"
"""OPTIONAL SSM key under ``ssm_prefix`` that releases the template's GPU rungs.

Value is a comma-separated list of ``<instance-type>:<max_workers>`` pairs, e.g.::

    g6e.xlarge:100,g6e.2xlarge:150

**Absent means the template stands untouched**, which is today's behaviour
byte-for-byte — that is the whole point of putting the switch here rather than on
the flow. A flow parameter would change the deployment's schema, and a schema
change forces every deployment to be re-registered, which drops hand-set
parameters on a live campaign. An SSM key is read at ``ray up`` time by whatever
code the runner already has, so a rung can be released mid-campaign with a
`put-parameter` and no release, no re-registration and no AMI re-bake.

Why ``max_workers`` and not a smarter switch: node-type choice in Ray 2.55.1
has no feedback from capacity errors, so ``max_workers`` per node type is the
only mechanism that moves the autoscaler. ``0`` makes a rung unreachable.
"""

GPU_WORKER_NODE_TYPE_PREFIX = "gpu-workers-ondemand"
"""Prefix identifying the ON-DEMAND GPU worker node types the ladder governs.

The ladder addresses node types by their EC2 instance type, but an instance type
does not identify a node type on its own: the template ships ``g6e.xlarge`` twice,
once on-demand and once spot. So the ladder's domain is fixed by NAME — every
``available_node_types`` key with this prefix — and spot rungs are outside it and
never touched. Keep this in lockstep with ``cluster.yaml.template``; a template
rename that breaks the prefix makes the ladder silently govern nothing, which is
why :func:`_apply_gpu_worker_ladder` refuses an empty domain rather than
returning quietly.
"""


#: Instance types this campaign will accept as a capacity fallback. Named the way the
#: `gpu-worker-ladder` key names them, because two vocabularies for one domain is how a
#: template edit and an operator's value drift apart.
#:
#: One size per card, chosen on HOST RAM rather than price: the vCPU-matched `g5.xlarge`
#: and `g6.xlarge` carry 16 GiB against a measured ~17.7 GB per-actor requirement, which
#: is the exact shape that OOMed the loader on the earlier 16 GB g5-class workers. The
#: `2xlarge` of each family gives 32 GiB, matching `g6e.xlarge`. Sizes outside this set
#: are refused rather than trusted -- an operator naming `g5.xlarge` under capacity
#: pressure would get a fleet that cannot run the work.
#:
#: The cost of the safe size is quota, and it is not small: the G-and-VT quota is counted
#: in vCPU, and these are 8 vCPU per GPU against `g6e.xlarge`'s 4. A fallback GPU spends
#: TWICE the quota of a production one -- 10,000 vCPU buys 2,500 L40S or 1,250 A10G.
GPU_FALLBACK_INSTANCE_TYPES: frozenset[str] = frozenset({"g5.2xlarge", "g6.2xlarge"})

GPU_FALLBACK_SCORER_ENV = {"RAY_AUTOSCALER_UTILIZATION_SCORER": autoscaler_scorer.SCORER_PATH}


_RAY_START = "ray start"
"""The token in a ``*_start_ray_commands`` entry that :func:`_pace_ray_start` prefixes.

Ray runs each entry as its own shell command over SSH, so an ``export`` on its own
line reaches nothing. Assignments have to ride on the command that starts Ray — and
the autoscaler is a child of that process, which is what carries the pacing to the
code that actually launches nodes.
"""


def _pace_ray_start(commands: list[str], pacing: dict[str, str]) -> list[str]:
    """Return ``commands`` with ``pacing`` assigned on the entry that starts Ray.

    Args:
        commands: A cluster-YAML ``*_start_ray_commands`` list.
        pacing: Environment names and values to assign.

    Returns:
        A new list, with the assignments prefixed to the ``ray start`` invocation.

    Raises:
        ValueError: If no entry starts Ray. A pacing request that lands nowhere is
            worse than one refused: the cluster comes up looking configured and
            launches at the unpaced rate, so this refuses rather than warns.
    """
    assignments = " ".join(f"{name}={value}" for name, value in pacing.items())
    paced = []
    applied = False
    for cmd in commands:
        if not applied and isinstance(cmd, str) and _RAY_START in cmd:
            paced.append(cmd.replace(_RAY_START, f"{assignments} {_RAY_START}", 1))
            applied = True
        else:
            paced.append(cmd)
    if not applied:
        msg = f"launch pacing requested but no start command invokes {_RAY_START!r}: {commands}"
        raise ValueError(msg)
    return paced


def _parse_gpu_worker_ladder(raw: str) -> list[tuple[str, int]]:
    """Parse a ``gpu-worker-ladder`` value into ordered ``(instance_type, max_workers)`` pairs.

    Args:
        raw: The SSM parameter value, e.g. ``"g6e.xlarge:100,g6e.2xlarge:150"``.
            Whitespace around any token is tolerated; empty entries (a trailing
            comma) are dropped.

    Returns:
        Pairs in the order written. Order is preserved because it is what a human
        reads back, NOT because it decides anything — Ray scores node types from
        their resources and ignores declaration order entirely (see
        ``TestRayNodeTypePreference``).

    Raises:
        RuntimeError: On any entry that is not exactly ``name:non-negative-int``,
            or on a repeated instance type. Both REFUSE rather than warn: this
            value sizes a GPU fleet, and the failure mode of a lenient parser is
            a rung silently left at ``0`` (a campaign that never grows) or a
            typo'd count read as capacity. A refusal at ``ray up`` costs one
            corrected parameter; a silent misparse costs a run.
    """
    pairs: list[tuple[str, int]] = []
    seen: set[str] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, count = entry.partition(":")
        name, count = name.strip(), count.strip()
        # Match the LEXICAL form, not a prefix of it: `partition` on a missing
        # separator yields an empty tail, and `int("")` would raise a ValueError
        # that reads as a bug rather than as bad configuration.
        if not sep or not name or not count.isdigit():
            msg = (
                f"Malformed {GPU_WORKER_LADDER_SSM_KEY} entry {entry!r}: expected "
                "'<instance-type>:<max_workers>' with a non-negative integer count"
            )
            raise RuntimeError(msg)
        if name in seen:
            msg = f"Duplicate instance type {name!r} in {GPU_WORKER_LADDER_SSM_KEY}: {raw!r}"
            raise RuntimeError(msg)
        seen.add(name)
        pairs.append((name, int(count)))
    if not pairs:
        msg = f"{GPU_WORKER_LADDER_SSM_KEY} is set but names no rung: {raw!r}"
        raise RuntimeError(msg)
    return pairs


def _apply_gpu_worker_ladder(config: dict[str, Any], raw: str) -> None:
    """Rewrite the on-demand GPU rungs' ``max_workers`` from a ladder value, in place.

    The ladder is AUTHORITATIVE over its domain, not additive to it: every
    on-demand GPU node type (see :data:`GPU_WORKER_NODE_TYPE_PREFIX`) the value
    does not name is set to ``0``. An additive reading would make
    ``g6e.2xlarge:150`` mean "add 150 of these to the 500 g6e.xlarge already
    allowed", so releasing a rung would also raise the fleet ceiling — two
    changes from one edit, and the second one unstated. Naming every rung you
    want makes the whole fleet shape readable from the parameter.

    Args:
        config: A loaded cluster config. Mutated in place.
        raw: The raw SSM value; see :func:`_parse_gpu_worker_ladder`.

    Raises:
        RuntimeError: If the config declares no on-demand GPU node type (a
            template drift that would otherwise make the ladder a no-op), or if
            the ladder names an instance type no such node type offers, or if two
            of them offer the same instance type so the target is ambiguous.
            Refusing beats warning here for the reason a warn-and-continue guard
            always loses: the run proceeds, and what it proceeds with is the
            fleet shape the operator was trying to change.
    """
    node_types = config["available_node_types"]
    domain = {name: cfg for name, cfg in node_types.items() if name.startswith(GPU_WORKER_NODE_TYPE_PREFIX)}
    if not domain:
        msg = (
            f"{GPU_WORKER_LADDER_SSM_KEY} is set but the cluster template declares no node type "
            f"named {GPU_WORKER_NODE_TYPE_PREFIX!r}* — the ladder would govern nothing. "
            f"Node types present: {sorted(node_types)}"
        )
        raise RuntimeError(msg)

    by_instance_type: dict[str, list[str]] = defaultdict(list)
    for name, cfg in domain.items():
        by_instance_type[cfg["node_config"]["InstanceType"]].append(name)

    pairs = _parse_gpu_worker_ladder(raw)
    unknown = [t for t, _ in pairs if t not in by_instance_type]
    if unknown:
        msg = (
            f"{GPU_WORKER_LADDER_SSM_KEY} names instance type(s) {unknown} with no on-demand GPU "
            f"node type in the cluster template. Available: {sorted(by_instance_type)}. Add the "
            "rung to cluster.yaml.template (at max_workers: 0) and ship it before releasing it."
        )
        raise RuntimeError(msg)
    ambiguous = {t: names for t, names in by_instance_type.items() if len(names) > 1}
    if ambiguous:
        msg = (
            f"Cannot apply {GPU_WORKER_LADDER_SSM_KEY}: instance type(s) {ambiguous} are offered by "
            "more than one on-demand GPU node type, so a ladder entry has no single target"
        )
        raise RuntimeError(msg)

    # Close every rung first, then open the named ones. Two passes, so an
    # unnamed rung is closed whatever order the ladder lists things in.
    for cfg in domain.values():
        cfg["max_workers"] = 0
    for instance_type, max_workers in pairs:
        node_types[by_instance_type[instance_type][0]]["max_workers"] = max_workers

    # `max_workers` per node type is a ceiling under the CLUSTER-wide ceiling, not
    # beside it: Ray takes the min. A ladder summing above the global value would
    # be capped there silently, so raise the global to fit the ladder it was given.
    #
    # And size it for EVERY open worker type, not just the ladder's. The cluster
    # ceiling is one budget shared by all of them, so a bundled spot rung — or any
    # custom type a template adds — left open at N consumes N of it and leaves the
    # ladder's own per-rung ceilings unreachable by exactly that much. The head node
    # is not a worker and does not count.
    ladder_total = sum(count for _, count in pairs)
    head_node_type = config.get("head_node_type")
    non_ladder_workers = sum(
        int(cfg.get("max_workers", 0) or 0)
        for name, cfg in node_types.items()
        if name != head_node_type and name not in domain
    )
    needed = ladder_total + non_ladder_workers
    if needed > config.get("max_workers", 0):
        config["max_workers"] = needed
    logging.getLogger(__name__).info(
        "Applied %s: %s (cluster max_workers=%d)",
        GPU_WORKER_LADDER_SSM_KEY,
        ", ".join(f"{t}={n}" for t, n in pairs),
        config["max_workers"],
    )


def _vcpu_per_node(node_type: dict[str, Any]) -> int:
    """What one INSTANCE of this rung costs in vCPU, from its DECLARED resources.

    Per instance, not per GPU, because ``max_workers`` counts instances. The two are the
    same number only while every rung is single-GPU, which is true of everything shipped
    today and is exactly the assumption that makes the difference invisible: a rung
    declaring 96 CPU and 8 GPU priced per-GPU reads as 12, and a 960-vCPU budget would
    then permit 80 instances -- 7,680 vCPU, eight times the budget.

    Read from the config rather than the EC2 catalogue because the declaration is what
    the autoscaler itself scales against, and because it needs no API call at resolve
    time. `test_declared_resources_match_the_ec2_catalogue` is what keeps the two honest.
    """
    resources = node_type.get("resources") or {}
    cpu, gpu = int(resources.get("CPU", 0)), int(resources.get("GPU", 0))
    if cpu <= 0 or gpu <= 0:
        msg = (
            f"Cannot price node type {node_type.get('node_config', {}).get('InstanceType')!r} "
            f"against a vCPU budget: it declares CPU={cpu}, GPU={gpu}. Every GPU rung must "
            "declare both -- see the resources block in cluster.yaml.template."
        )
        raise RuntimeError(msg)
    return cpu


def _vcpu_budget_ceiling(node_type: dict[str, Any], vcpu_budget: int) -> int:
    """How many of this rung the vCPU budget affords.

    Raises rather than flooring at 1. A floor would open a rung the budget cannot pay
    for -- a budget of 5 buying one 8-vCPU node overspends by 60% -- and the caller
    stated a quota, not a preference. A budget too small to seat one node of an open
    rung is a configuration error, and it is cheaper to say so here than to discover it
    as an unexplained `InstanceLimitExceeded` during a capacity event.
    """
    per_node = _vcpu_per_node(node_type)
    affords = vcpu_budget // per_node
    if affords < 1:
        instance = node_type.get("node_config", {}).get("InstanceType")
        msg = (
            f"gpu_fallback_vcpu_budget={vcpu_budget} cannot afford a single {instance!r} "
            f"at {per_node} vCPU per instance. Raise the budget to at least {per_node}, or "
            "close that rung."
        )
        raise ValueError(msg)
    return affords


def _apply_gpu_vcpu_budget(config: dict[str, Any], vcpu_budget: int) -> None:
    """Re-ceiling every OPEN on-demand GPU rung to what ``vcpu_budget`` affords.

    The G-and-VT service quota is counted in vCPU, not in cards, and rungs are not
    equally priced: the fallback sizes are 8 vCPU per GPU against the production rung's
    4. A ceiling expressed in NODES therefore means a different quota bill depending on
    which card wins, and an all-fallback fleet at the production node count quietly
    spends twice the quota.

    So each rung gets the count the budget affords IT -- 250 instances at 4 vCPU each,
    125 at 8. Priced PER INSTANCE, because that is what ``max_workers`` counts.
    That is the honest statement of "as many of this card as the quota allows".

    It NARROWS only, never widens: a ceiling already lower than the budget affords was
    set deliberately and stands.

    **It does not bound a MIXTURE, and cannot.** Ray's ceilings count nodes and carry no
    weight (`resource_demand_scheduler` has no cost concept at all), so a fleet that is
    part production and part fallback can exceed the budget -- bounded by the widest
    per-rung ceiling, not equal to it. Making it exact would need a controller outside
    Ray adjusting demand as the mix shifts. What this does remove is the worst case: a
    fleet that has fallen over entirely can no longer run at the production node count.
    """
    if vcpu_budget <= 0:
        msg = f"gpu_fallback_vcpu_budget must be > 0, got {vcpu_budget}"
        raise ValueError(msg)
    node_types = config["available_node_types"]
    for name, cfg in node_types.items():
        if not name.startswith(GPU_WORKER_NODE_TYPE_PREFIX) or cfg.get("max_workers", 0) <= 0:
            continue
        # NARROWS ONLY. A budget must never RAISE a ceiling someone set deliberately:
        # capping the production rung at what it can actually be supplied is the lever
        # that pushes surplus demand onto the fallback, and a budget that overrode it
        # would quietly undo the operator's decision. The budget is a maximum, not a
        # target.
        cfg["max_workers"] = min(cfg["max_workers"], _vcpu_budget_ceiling(cfg, vcpu_budget))


def _apply_gpu_fallback(
    config: dict[str, Any], instance_types: Sequence[str], vcpu_budget: int | None = None
) -> list[str]:
    """Open a fallback rung per named instance type and make the autoscaler use it.

    Two halves, and NEITHER works alone. Opening a rung gives the autoscaler somewhere
    to go; the scorer is what sends it there, because Ray's default scoring cannot see
    capacity at all (see :mod:`~tessera_embeddings.providers.aws.autoscaler_scorer`).
    Opening rungs without the scorer leaves them idle forever, and the scorer without
    open rungs has nothing to promote.

    Each fallback rung is opened to the SAME ceiling as the production rung, so either
    card can carry the whole fleet. That does not make the fleet bigger: the cluster's
    global ``max_workers`` still caps the sum, and in practice the count is bound by
    inference demand well below it. What this changes is what the fleet is MADE OF.

    Args:
        config: A loaded cluster config. Mutated in place.
        instance_types: EC2 instance types from :data:`GPU_FALLBACK_INSTANCE_TYPES`
            (e.g. ``["g5.2xlarge"]``). Empty opens no rung and installs no scorer.
        vcpu_budget: Optional G-and-VT vCPU budget for this cluster's GPU fleet. When
            given, every open GPU rung -- production included -- is ceilinged at what the
            budget affords IT rather than at a flat node count, which is the only way a
            heterogeneous fleet can be held to a quota counted in vCPU. See
            :func:`_apply_gpu_vcpu_budget` for what it does and does not bound.

    Returns:
        The node type names opened, for logging by the caller.

    Raises:
        RuntimeError: If a card has no known instance type, or that type has no rung in
            the template, or the template declares no production GPU rung to match. Each
            would otherwise leave the fleet quietly unable to fail over -- which is only
            discovered during the capacity event the feature exists for.
    """
    node_types = config["available_node_types"]
    if vcpu_budget is not None:
        _apply_gpu_vcpu_budget(config, vcpu_budget)
    if not instance_types:
        return []

    if len(set(instance_types)) > 1:
        msg = (
            f"Only one GPU fallback instance type may be opened at a time, got "
            f"{sorted(set(instance_types))}. The supported fallbacks are all 8 vCPU / 1 GPU "
            "with identical declared resources, so Ray scores them EQUALLY and breaks the "
            "tie on node-type name, descending -- which would pick a card on alphabetical "
            "order rather than on throughput. Ray's scorer has no priority concept to "
            "express the preference with, so the choice is made here, by naming one type."
        )
        raise RuntimeError(msg)

    unknown = [t for t in instance_types if t not in GPU_FALLBACK_INSTANCE_TYPES]
    if unknown:
        msg = (
            f"Unsupported GPU fallback instance type(s) {unknown}. Supported: "
            f"{sorted(GPU_FALLBACK_INSTANCE_TYPES)}. Sizes are restricted on HOST RAM: the "
            "vCPU-matched `xlarge` of each family carries 16 GiB against a measured ~17.7 GB "
            "per-actor requirement and would OOM the loader before inference."
        )
        raise RuntimeError(msg)

    # ALL names per instance type, not the last one. An instance type does not identify a
    # node type -- a template may offer the same type twice under different launch config,
    # which is why the ladder refuses the same ambiguity rather than picking one.
    by_instance_type: dict[str, list[str]] = defaultdict(list)
    for name, cfg in node_types.items():
        if name.startswith(GPU_WORKER_NODE_TYPE_PREFIX):
            by_instance_type[cfg["node_config"]["InstanceType"]].append(name)
    ambiguous = {t: n for t, n in by_instance_type.items() if t in instance_types and len(n) > 1}
    if ambiguous:
        msg = (
            f"Cannot open a GPU fallback: instance type(s) {ambiguous} are offered by more "
            "than one on-demand GPU node type, so the request has no single target."
        )
        raise RuntimeError(msg)

    # The production ceiling every fallback is matched to, from the widest open PRODUCTION
    # rung. Fallbacks are excluded by instance type, not by ceiling: they share the
    # node-type prefix, so a ladder that had already opened one would otherwise contribute
    # its own ceiling here -- and an all-fallback ladder would satisfy the "something to
    # fall back FROM" guard with nothing to fall back from at all.
    open_ceilings = [
        cfg["max_workers"]
        for name, cfg in node_types.items()
        if name.startswith(GPU_WORKER_NODE_TYPE_PREFIX)
        and cfg["node_config"]["InstanceType"] not in GPU_FALLBACK_INSTANCE_TYPES
        and cfg.get("max_workers", 0) > 0
    ]
    if not open_ceilings:
        msg = (
            "GPU fallback was requested but no PRODUCTION GPU rung is open, so there is no "
            "ceiling to match and nothing to fall back FROM. Open one with the "
            f"{GPU_WORKER_LADDER_SSM_KEY} key, or drop the fallback request."
        )
        raise RuntimeError(msg)
    ceiling = max(open_ceilings)

    # NEVER above the cluster-wide ceiling. A fallback changes what the fleet is made of,
    # not how big it is, so a template that deliberately caps the aggregate at 100 must
    # keep that cap -- raising the global to fit a fallback would grow the fleet by exactly
    # the amount the operator said no to.
    global_ceiling = config.get("max_workers")
    if isinstance(global_ceiling, int) and global_ceiling > 0:
        ceiling = min(ceiling, global_ceiling)

    opened = []
    for instance_type in instance_types:
        name = by_instance_type[instance_type][0]
        want = _vcpu_budget_ceiling(node_types[name], vcpu_budget) if vcpu_budget is not None else ceiling
        if isinstance(global_ceiling, int) and global_ceiling > 0:
            want = min(want, global_ceiling)
        # A ladder that already opened this rung chose its ceiling on purpose; opening it
        # again must not widen it. Same rule as the budget's, for the same reason.
        current = node_types[name].get("max_workers", 0)
        node_types[name]["max_workers"] = min(current, want) if current > 0 else want
        opened.append(name)

    widest = max(node_types[n]["max_workers"] for n in opened)
    logging.getLogger(__name__).info(
        "GPU fallback enabled for %s: opened %s at max_workers=%d, scorer=%s",
        ", ".join(instance_types),
        ", ".join(opened),
        widest,
        autoscaler_scorer.SCORER_PATH,
    )
    return opened


def resolve_ami_id(ami_ssm_name: str, region: str = "us-west-2") -> str:
    """Resolve the worker AMI ID the ``ami_ssm_name`` SSM parameter currently points at.

    The campaign resolves this ONCE up front and pins it into every fill's
    provisioning (``ray_cluster(ami_id=...)``), so a re-bake that repoints the SSM
    parameter mid-campaign can't make a fill boot a different image than the one its
    staging fingerprint recorded — see :func:`resolve_code_artifact_identity`.
    """
    return boto3.client("ssm", region_name=region).get_parameter(Name=ami_ssm_name)["Parameter"]["Value"]


def resolve_code_artifact_identity(
    ami_ssm_name: str,
    code_bucket: str | None = None,
    code_suffix: str = "",
    region: str = "us-west-2",
    ami_id: str | None = None,
) -> str:
    """Immutable identity of the code a Ray fill will run, for the staging fingerprint.

    Returns ``ami=<ami-id>`` and, when a source tarball overlays the AMI, appends
    ``|tarball=<etag>`` — the same two artifacts :func:`provision_ray_cluster` boots
    from (the AMI behind ``ami_ssm_name`` and ``s3://{code_bucket}/code/src{suffix}.tar.gz``).

    The global campaign folds this into each cell's staging ``run_id`` because
    ``code_suffix`` alone is NOT immutable: it is empty for a baked production AMI and
    only a filename/branch stem for a tarball, so re-baking the AMI under the same SSM
    name, or overwriting the tarball, leaves it unchanged. A retry would then resume
    tiles staged by the OLD code while remaining tiles run the NEW code, permanently
    publishing a mixed-version year. Resolving the real AMI ID and tarball ETag makes
    any code change flip the fingerprint, so a fresh staging prefix is used.

    KNOWN RESIDUAL WINDOW (dev-overlay path only). The ETag is read here, once, while
    workers later download the mutable key ``code/src{code_suffix}.tar.gz``. Overwrite
    that object mid-campaign and workers boot code the fingerprint does not describe.
    Re-reading the ETag just before launch would narrow the window, not close it — the
    overwrite can land between that HEAD and the worker's GET — so this is left as a
    constraint rather than a partial mitigation: DO NOT overwrite a tarball a campaign
    is running against. Production is unaffected (a baked AMI passes ``code_bucket=None``
    and has no tarball term at all). To close it properly, upload content-addressed keys
    (``code/src-<sha>.tar.gz``) so the object is immutable by construction, rather than
    threading an S3 versionId through provisioning. Same reasoning as the model
    checkpoint's filename-not-bytes identity in ``_staging_run_id``.

    Args:
        ami_ssm_name: SSM parameter holding the worker AMI ID.
        code_bucket: S3 bucket of the source tarball; ``None`` for a pure-AMI deploy.
        code_suffix: Tarball filename suffix (``code/src{code_suffix}.tar.gz``).
        region: AWS region for the SSM/S3 clients (the store's region; us-west-2 default).
        ami_id: A pre-resolved AMI ID to fingerprint instead of reading ``ami_ssm_name``.
            Pass the SAME id that provisioning is pinned to (``ray_cluster(ami_id=...)``)
            so the fingerprint and the booted image are guaranteed identical — a caller
            that pins provisioning must pin the fingerprint too, or the two could resolve
            the SSM pointer at different instants and disagree.
    """
    parts = [f"ami={ami_id if ami_id is not None else resolve_ami_id(ami_ssm_name, region)}"]
    tarball = source_tarball_identity(code_bucket, code_suffix, region)
    if tarball:
        parts.append(tarball)
    return "|".join(parts)


def source_tarball_identity(code_bucket: str | None, code_suffix: str, region: str) -> str:
    """``tarball=<etag>`` for the source archive workers overlay, or ``""`` when there is none.

    Split out of :func:`resolve_code_artifact_identity` so the staging fingerprint can carry
    the TARBALL term without the AMI one. Those two have opposite properties for staging
    reuse: re-baking an AMI does not change what a staged tile contains, so folding it in
    abandoned every staged tile for nothing — which is why the staging identity was narrowed
    on 2026-07-30 — while replacing the tarball changes exactly what a worker executes.

    **Empty for a baked-AMI deploy** (``code_bucket=None``), which is production. So a
    fingerprint that includes this is unchanged there and gains the term only on the
    dev-overlay path, where it is the whole exposure.

    The residual window in :func:`resolve_code_artifact_identity` applies here unchanged: the
    ETag is read once while workers later GET the mutable key, so do not overwrite a tarball
    a campaign is running against.
    """
    if not code_bucket:
        return ""
    s3 = boto3.client("s3", region_name=region)
    etag = s3.head_object(Bucket=code_bucket, Key=f"code/src{code_suffix}.tar.gz")["ETag"].strip('"')
    return f"tarball={etag}"


def cluster_name_for_flow_run(flow_run_id: object, cluster_yaml: Path = DEFAULT_CLUSTER_TEMPLATE) -> str | None:
    """Deterministic Ray cluster name for a flow run, or ``None`` if no id is known.

    The name must be recomputable from nothing but the flow-run id: Prefect runs
    cancellation/crash hooks in a freshly imported module after the flow's child
    process is killed, so a hook can re-derive the cluster tag (and terminate the
    fleet) even with the flow's module globals unset. Both the ``tessera_embeddings``
    and ``fill-zone-year`` flows pass this as ``ray_cluster(cluster_name=...)`` so the
    provisioned name and the hook's re-derived name always match. The base comes from
    the shipped cluster template so it stays in sync with what ``ray up`` uses.
    """
    if not flow_run_id:
        return None
    with cluster_yaml.open() as f:
        base = yaml.safe_load(f).get("cluster_name", "tessera-inference")
    return f"{base}-{str(flow_run_id).replace('-', '')[:8]}"


def _build_cloudwatch_setup_command(
    cloudwatch_template: Path = DEFAULT_CLOUDWATCH_TEMPLATE,
    log_group: str = DEFAULT_CLOUDWATCH_LOG_GROUP,
) -> str:
    """Build a shell command that configures the CloudWatch agent on a Ray node.

    Reads the human-readable JSON template, compacts it to a single line,
    substitutes the EC2 instance ID at boot, and writes the resulting
    config in place. Heredocs in YAML setup_commands break when Ray
    sends them over SSH (indented terminators are never matched), so we
    inline the whole config as a single shell command.

    Args:
        cloudwatch_template: Path to the JSON template. Defaults to the
            template shipped with this provider.
        log_group: CloudWatch log group name to receive Ray logs.
    """
    with cloudwatch_template.open() as f:
        template = json.load(f)
    compact = json.dumps(template, separators=(",", ":"))
    compact = compact.replace("__LOG_GROUP__", log_group)
    # Escape single quotes so embedding in a single-quoted shell string is safe.
    compact = compact.replace("'", "'\\''")
    return (
        "sudo mkdir -p /opt/aws/amazon-cloudwatch-agent/etc"
        " && INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
        f" && echo '{compact}' | sed \"s/__INSTANCE_ID__/$INSTANCE_ID/g\""
        " | sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json > /dev/null"
        " && sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl"
        " -a fetch-config -m ec2 -s"
        " -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json"
        ' || echo "WARNING: CloudWatch agent not available — logs will not ship to CloudWatch" >&2'
    )


def _resolve_ray_config(
    cluster_yaml: str | Path,
    *,
    region: str = "us-west-2",
    ami_ssm_name: str,
    ami_id: str | None = None,
    ssm_prefix: str = DEFAULT_SSM_PREFIX,
    cluster_name: str | None = None,
    instance_tags: list[dict[str, str]] | None = None,
    code_bucket: str | None = None,
    code_suffix: str = "",
    cloudwatch_log_group: str = DEFAULT_CLOUDWATCH_LOG_GROUP,
    cloudwatch_template: Path = DEFAULT_CLOUDWATCH_TEMPLATE,
    idle_timeout_minutes: int | None = None,
    launch_pacing: bool = False,
    gpu_fallback_instance_types: Sequence[str] = (),
    gpu_fallback_vcpu_budget: int | None = None,
) -> str:
    """Inject AWS resource IDs from SSM into a Ray cluster YAML template.

    Reads SSM parameters under ``ssm_prefix``, looks up the AMI ID from
    ``ami_ssm_name``, lists every private subnet on every node type (multi-AZ
    with launch-time capacity failover — see the subnet block below),
    materialises the SSH key from SSM into a tempfile, injects the
    CloudWatch agent setup command, substitutes ``{CODE_BUCKET}`` and
    ``{CODE_SUFFIX}`` in setup_commands, and writes the resolved config to
    a tempfile.

    Args:
        cluster_yaml: Path to the cluster YAML template. Use
            :data:`DEFAULT_CLUSTER_TEMPLATE` for the bundled template.
        region: AWS region for SSM and EC2 clients.
        ami_ssm_name: SSM parameter name holding the worker AMI ID. Used only
            when ``ami_id`` is not given.
        ami_id: A pre-resolved worker AMI ID that PINS the image, bypassing the
            ``ami_ssm_name`` lookup. The campaign resolves the AMI once and threads
            it here so a mid-campaign re-bake can't repoint the SSM parameter and
            boot a different image than a fill's staging fingerprint recorded.
        ssm_prefix: Prefix under which Ray resource IDs are stored.
            Required keys: ``security-group-id``, ``instance-profile-arn``,
            ``private-subnet-ids``, ``key-pair-name``, ``key-pair-id``.
            Optional key: ``gpu-worker-ladder`` (see
            :data:`GPU_WORKER_LADDER_SSM_KEY`) — releases the template's wider
            GPU rungs without a release or a deployment re-registration. Absent
            leaves the template's node types untouched.
        cluster_name: Override the template's ``cluster_name``. Required
            for running multiple clusters concurrently.
        instance_tags: Extra EC2 tags to apply to every node, on top of
            the always-present ``Project`` tag (see :data:`PROJECT_TAG_VALUE`)
            and Ray's own ``ray-cluster-name`` tag. List of
            ``{"Key": str, "Value": str}`` dicts; a ``Project`` key here
            overrides the default. ``None`` means only the defaults.
        code_bucket: S3 bucket name (without ``s3://``) substituted for
            ``{CODE_BUCKET}`` in setup_commands. ``None`` leaves the
            placeholder; pair with ``sync_source_path=None`` to disable
            tarball sync entirely.
        code_suffix: Substituted for ``{CODE_SUFFIX}``. Empty string for
            production tarballs; ``"-mybranch"`` for dev branches.
        cloudwatch_log_group: CloudWatch log group for Ray agent logs.
        cloudwatch_template: Path to the CloudWatch agent JSON template.
        idle_timeout_minutes: Override the template's autoscaler idle-down
            delay. The template default (2 min) suits single-ROI runs, where
            any idle worker is surplus; a multi-zone sequential fill holds one
            cluster across zones and must survive the inter-zone gap
            (staged-completeness verify + next zone's dispatch), so it passes
            a larger value. ``None`` keeps the template's value.
        launch_pacing: Assign :data:`LAUNCH_PACING_ENV` on the head's ``ray start``
            command, so the autoscaler it spawns paces its EC2 launch requests
        gpu_fallback_instance_types: Card names (see :data:`GPU_FALLBACK_CARDS`) this run may fall
            back to when the production rung has no capacity. Empty -- the default --
            leaves the cluster exactly as it is today. Naming a card opens its rung AND
            installs the capacity-aware autoscaler scorer; see :func:`_apply_gpu_fallback`
            for why both are needed and neither is sufficient.
        gpu_fallback_vcpu_budget: Optional G-and-VT vCPU budget for this cluster's GPU
            fleet. Rungs are not equally priced -- the fallback sizes are 8 vCPU per GPU
            against the production rung's 4 -- so a ceiling in NODES means a different
            quota bill depending on which card wins. Given a budget, each rung is
            ceilinged at what the budget affords IT.
            against the account's shared request quota. ``False`` leaves the
            template's commands untouched; pass ``True`` when several clusters
            will be growing at once. Worker start commands are never touched —
            only the head runs an autoscaler.

    Returns:
        Path to the resolved YAML tempfile.

    Raises:
        RuntimeError: If required SSM parameters are missing or SSH key
            material cannot be retrieved.
    """
    cluster_yaml = Path(cluster_yaml)
    with cluster_yaml.open() as f:
        config = yaml.safe_load(f)

    if cluster_name:
        config["cluster_name"] = cluster_name
    if idle_timeout_minutes is not None:
        config["idle_timeout_minutes"] = idle_timeout_minutes

    ssm = boto3.client("ssm", region_name=region)

    # Fetch all params under the prefix in one call
    params: dict[str, str] = {}
    paginator = ssm.get_paginator("get_parameters_by_path")
    for page in paginator.paginate(Path=ssm_prefix, Recursive=True):
        for p in page["Parameters"]:
            key = p["Name"].rsplit("/", 1)[-1]
            params[key] = p["Value"]

    missing = _REQUIRED_SSM_KEYS - params.keys()
    if missing:
        msg = f"Missing required SSM parameters under {ssm_prefix!r}: {sorted(missing)}"
        raise RuntimeError(msg)

    # OPTIONAL, and its absence is the default: no key means the template's rungs
    # stand exactly as shipped. Applied before anything else touches the node
    # types so a refusal costs no tempfiles.
    if GPU_WORKER_LADDER_SSM_KEY in params:
        _apply_gpu_worker_ladder(config, params[GPU_WORKER_LADDER_SSM_KEY])

    # AFTER the ladder, so a ladder that narrows the production rung narrows the
    # fallbacks with it. Empty is the default and changes nothing.
    fallback_opened = _apply_gpu_fallback(config, gpu_fallback_instance_types, gpu_fallback_vcpu_budget)

    sg_ids = [params["security-group-id"]]
    all_subnet_ids = [s.strip() for s in params["private-subnet-ids"].split(",")]
    iam_profile = {"Arn": params["instance-profile-arn"]}
    key_name = params["key-pair-name"]
    # Always stamp the Project tag so teardown terminates are IAM-authorised
    # (see PROJECT_TAG_VALUE). Caller-supplied tags win on key collision.
    merged_tags = [{"Key": "Project", "Value": PROJECT_TAG_VALUE}]
    caller_keys = {t["Key"] for t in (instance_tags or [])}
    if "Project" in caller_keys:
        merged_tags = []
    merged_tags.extend(instance_tags or [])
    tag_specs: list[dict[str, Any]] = [{"ResourceType": "instance", "Tags": merged_tags}]

    # Every node type gets ALL private subnets, in SSM-param order. Ray's AWS
    # node provider launches in the FIRST listed subnet and rotates to the
    # next on a launch ClientError (e.g. InsufficientInstanceCapacity),
    # trying every subnet before giving up — so the fleet lands mostly in one
    # AZ with automatic capacity spillover to the others (a 2026-07-17 run
    # stalled at 2 workers when its pinned AZ ran out of g6e capacity).
    # Cross-AZ exposure is negligible by construction: inference's bulk data
    # plane is actor↔S3 only (free in-region via the subnets' S3 gateway
    # endpoints — verify routes when subnets change, see gotchas.md) and
    # head↔worker traffic is KB/s control RPCs. INVARIANT that keeps this
    # cheap: Ray actors must never exchange bulk data node-to-node — all bulk
    # I/O goes to S3. (The Dask provider keeps its single-AZ pin; Dask
    # genuinely shuffles between workers.)
    ec2 = boto3.client("ec2", region_name=region)
    subnet_resp = ec2.describe_subnets(SubnetIds=all_subnet_ids)
    if not subnet_resp.get("Subnets"):
        msg = f"No subnets found for IDs: {all_subnet_ids}"
        raise RuntimeError(msg)
    # describe_subnets returns arbitrary order; order the AZ list to match the
    # SSM subnet order (= launch-preference order), deduping shared AZs.
    az_by_subnet = {s["SubnetId"]: s["AvailabilityZone"] for s in subnet_resp["Subnets"]}
    azs = list(dict.fromkeys(az_by_subnet[sid] for sid in all_subnet_ids))
    config["provider"]["availability_zone"] = ",".join(azs)

    # Prefer a caller-PINNED AMI ID (the campaign resolves it once and threads it
    # through every fill) over re-reading the SSM pointer here: re-reading would let
    # a mid-campaign re-bake boot a different image than the fill's staging
    # fingerprint recorded. Fall back to the SSM lookup when unpinned (direct/dev
    # invocations), where the pointer is authoritative.
    resolved_ami_id = ami_id if ami_id is not None else ssm.get_parameter(Name=ami_ssm_name)["Parameter"]["Value"]

    for node_type_cfg in config["available_node_types"].values():
        nc = node_type_cfg["node_config"]
        nc["ImageId"] = resolved_ami_id
        nc["KeyName"] = key_name
        nc["SecurityGroupIds"] = sg_ids
        nc["IamInstanceProfile"] = iam_profile
        nc["SubnetIds"] = list(all_subnet_ids)
        if tag_specs:
            nc["TagSpecifications"] = tag_specs

    # Retrieve SSH private key from SSM (stored by AWS at key-pair creation time)
    key_pair_id = params["key-pair-id"]
    try:
        key_resp = ssm.get_parameter(Name=f"/ec2/keypair/{key_pair_id}", WithDecryption=True)
        ssh_key_material = key_resp["Parameter"]["Value"]
    except ssm.exceptions.ParameterNotFound as exc:
        msg = f"SSH key not found in SSM at /ec2/keypair/{key_pair_id}"
        raise RuntimeError(msg) from exc

    ssh_key_fd, ssh_key_path = tempfile.mkstemp(prefix="ray_ssh_", suffix=".pem")
    with os.fdopen(ssh_key_fd, "w") as kf:
        kf.write(ssh_key_material)
    Path(ssh_key_path).chmod(stat.S_IRUSR)  # 0o400

    config["auth"]["ssh_private_key"] = ssh_key_path

    # Inject the CloudWatch agent setup command (replaces any heredoc-style
    # cloudwatch entry in the template, which would break over SSH).
    #
    # Append it to the *_start_ray_commands, NOT setup_commands: setup_commands
    # run before `ray start`, so the agent would resolve its file paths while
    # /tmp/ray/session_latest is empty or stale, then `ray start` repoints the
    # session_latest symlink out from under it and no logs ship. Starting the
    # agent after `ray start` guarantees the session and its log files already
    # exist when fetch-config discovers them. Strip any pre-existing cloudwatch
    # entries from every command list so we don't double-start.
    cw_cmd = _build_cloudwatch_setup_command(cloudwatch_template, cloudwatch_log_group)
    for key in ("setup_commands", "head_start_ray_commands", "worker_start_ray_commands"):
        config[key] = [cmd for cmd in config.get(key, []) if "cloudwatch" not in str(cmd).lower()]
    config["head_start_ray_commands"].append(cw_cmd)
    config["worker_start_ray_commands"].append(cw_cmd)

    # Pace this cluster's EC2 launch requests. Head only: the autoscaler that issues
    # RunInstances is a child of the head's `ray start`, and a worker has none to
    # inherit the setting. Applied after the CloudWatch append so the entry that
    # starts Ray is the one that gets the assignments.
    if launch_pacing:
        config["head_start_ray_commands"] = _pace_ray_start(config["head_start_ray_commands"], LAUNCH_PACING_ENV)

    # The other half of GPU fallback, and it rides here for the same reason the pacing
    # does: Ray reads this when the autoscaler is constructed, and the autoscaler is a
    # child of the head's `ray start`. Assigned only when a rung was actually opened, so
    # a run without fallback gets Ray's stock scorer and behaves exactly as before.
    if fallback_opened:
        config["head_start_ray_commands"] = _pace_ray_start(config["head_start_ray_commands"], GPU_FALLBACK_SCORER_ENV)

    # Substitute {CODE_BUCKET} and {CODE_SUFFIX} in setup_commands. With no bucket the
    # command is DROPPED, not left alone: an unsubstituted `aws s3 cp
    # s3://{CODE_BUCKET}/...` is a valid command line over a bucket name that cannot
    # exist, so `ray up` runs it on every node and every node fails setup. `None` is the
    # documented default and means the AMI already carries the code, so there is nothing
    # to fetch — dropping the line is what that default has to mean for the cluster to
    # boot at all.
    if code_bucket is not None:
        config["setup_commands"] = [
            cmd.replace("{CODE_BUCKET}", code_bucket).replace("{CODE_SUFFIX}", code_suffix)
            if isinstance(cmd, str)
            else cmd
            for cmd in config["setup_commands"]
        ]
    else:
        config["setup_commands"] = [cmd for cmd in config["setup_commands"] if "{CODE_BUCKET}" not in str(cmd)]

    resolved_fd, resolved_path = tempfile.mkstemp(prefix="ray_cluster_", suffix=".yaml")
    with os.fdopen(resolved_fd, "w") as rf:
        yaml.dump(config, rf, default_flow_style=False)

    return resolved_path


def cleanup_ray_tempfiles(resolved_yaml: str | None) -> None:
    """Best-effort cleanup of resolved YAML and SSH key tempfiles.

    Safe to call from an on-cancellation hook: silently swallows all
    errors so a partially-resolved cluster doesn't block cancellation.
    """
    if not resolved_yaml:
        return
    with contextlib.suppress(Exception):
        with Path(resolved_yaml).open() as f:
            config = yaml.safe_load(f)
        ssh_key_path = config.get("auth", {}).get("ssh_private_key")
        if ssh_key_path:
            Path(ssh_key_path).unlink(missing_ok=True)
    with contextlib.suppress(Exception):
        Path(resolved_yaml).unlink(missing_ok=True)


def _sync_code_to_s3(
    src_dir: Path,
    s3_bucket: str,
    s3_key: str,
) -> None:
    """Tar a source directory and upload it to S3.

    Workers pull this tarball on startup instead of using Ray
    ``file_mounts``, which depends on SSH-based rsync and bottlenecks at
    100-500+ workers. The S3 download is parallel across all workers.

    Args:
        src_dir: Directory to package.
        s3_bucket: Destination S3 bucket name (no ``s3://`` prefix).
        s3_key: S3 key for the tarball (e.g. ``"code/src.tar.gz"``).
    """
    tarball_fd, tarball = tempfile.mkstemp(suffix=".tar.gz", prefix="src_sync_")
    os.close(tarball_fd)
    try:
        subprocess.run(
            ["tar", "-czf", tarball, "-C", str(src_dir.parent), src_dir.name],
            check=True,
        )
        boto3.client("s3").upload_file(tarball, s3_bucket, s3_key)
    finally:
        Path(tarball).unlink(missing_ok=True)


def _start_ray_cluster(
    resolved_yaml: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    launch_pacing: bool = False,
) -> str:
    """Launch a Ray cluster via ``ray up`` on a resolved YAML; return the head IP.

    Config resolution happens in the caller (:func:`ray_cluster`) so the
    resolved path is already bound when a failed launch unwinds to the
    teardown block — ``ray down`` must target the real (uuid-suffixed)
    cluster, not the unresolved template.

    ``launch_pacing`` paces the head node's OWN launch. ``ray up`` runs the node
    provider in this process, not on the head, so the head's RunInstances call
    draws on the same account quota from here — and several clusters starting
    together is exactly when that matters.

    It gets :data:`LAUNCH_PACING_CLIENT_ENV` only, NOT the autoscaler tuning. This
    launch happens once: nothing retries a head that failed to start, and losing it
    fails the whole fill. Spacing its attempts out is a help; taking attempts away
    from it is a fleet-ending risk taken in exchange for nothing, since the batch
    size is meaningless for a single instance. The autoscaler's own settings travel
    separately, in the resolved YAML (see :func:`_resolve_ray_config`).
    """
    log.info("Starting Ray cluster from %s", resolved_yaml)
    ray_up = subprocess.run(
        ["ray", "up", resolved_yaml, "-y", "--no-config-cache"],
        capture_output=True,
        text=True,
        env={**os.environ, **LAUNCH_PACING_CLIENT_ENV} if launch_pacing else None,
    )
    if ray_up.returncode != 0:
        log.error("ray up failed (exit %d)", ray_up.returncode)
        log.error("ray up stdout:\n%s", ray_up.stdout[-5000:] if ray_up.stdout else "(empty)")
        log.error("ray up stderr:\n%s", ray_up.stderr[-5000:] if ray_up.stderr else "(empty)")
        msg = f"ray up failed with exit code {ray_up.returncode}"
        raise RuntimeError(msg)
    log.info("Ray cluster started")

    result = subprocess.run(
        ["ray", "get-head-ip", resolved_yaml],
        capture_output=True,
        text=True,
        check=True,
    )
    # ray get-head-ip emits log lines before the IP; take the last line
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("ray get-head-ip returned no output")
    head_ip = lines[-1].strip()
    log.info("Ray head node IP: %s", head_ip)

    return head_ip


def _log_ray_dashboard_ssm_command(
    cluster_name: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    region: str,
) -> None:
    """Log a copy-pasteable SSM command to port-forward the Ray dashboard.

    Finds the head node by its ``ray-cluster-name`` + ``ray-node-type=head``
    tags and emits an ``aws ssm start-session`` block that forwards the
    dashboard (port 8265) to ``localhost``. The head listens on 8265 with
    ``--dashboard-host=0.0.0.0`` and the EC2 role carries
    ``AmazonSSMManagedInstanceCore``, so the port-forward works against the
    instance ID.

    Uses ``AWS-StartPortForwardingSession``, which targets a port on the
    managed instance itself. The ``...ToRemoteHost`` variant addresses hosts
    reachable *from* the instance, and the SSM agent rejects loopback
    destinations for it.

    ``--region`` is filled in from the region the head node was looked up in,
    so an operator whose default region differs doesn't get a "target not
    connected" from SSM. No ``--profile`` is printed — credentials come from
    the operator's environment (``AWS_PROFILE`` or their default).

    Best-effort: warns and returns on any failure, never raises.

    Args:
        cluster_name: Resolved, unique ``ray-cluster-name`` tag value (with
            the uuid8 suffix) that Ray actually tagged instances with.
        log: Logger.
        region: AWS region.
    """
    try:
        ec2 = boto3.client("ec2", region_name=region)
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "tag:ray-cluster-name", "Values": [cluster_name]},
                {"Name": "tag:ray-node-type", "Values": ["head"]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ],
        )
        reservations = resp.get("Reservations", [])
        instances = reservations[0].get("Instances", []) if reservations else []
        if not instances:
            log.warning("Could not find Ray head node for dashboard command")
            return
        instance_id = instances[0]["InstanceId"]
        log.info(
            "To view the Ray dashboard, run:\n\n"
            "aws ssm start-session \\\n"
            f"  --target {instance_id} \\\n"
            "  --document-name AWS-StartPortForwardingSession \\\n"
            f"  --region {region} \\\n"
            '  --parameters \'{"portNumber":["8265"],"localPortNumber":["8265"]}\'\n\n'
            "Then open http://localhost:8265"
        )
    except Exception:
        log.warning("Could not generate Ray dashboard command", exc_info=True)


def make_instance_terminator(
    region: str = "us-west-2",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> Callable[[str], None]:
    """Create a callback that terminates a single EC2 instance by ID.

    Wired into the inference scheduler's ``on_actor_retire`` hook so that
    GPU nodes are terminated immediately after retiring idle actors,
    rather than waiting for the Ray autoscaler's idle timeout (which is
    unreliable after ``ray.kill()`` because it relies on the node
    self-reporting empty).

    Args:
        region: AWS region.
        log: Logger for termination events.
    """
    _log = log or logging.getLogger(__name__)
    ec2 = boto3.client("ec2", region_name=region)

    def _terminate(instance_id: str) -> None:
        try:
            ec2.terminate_instances(InstanceIds=[instance_id])
            _log.info("Terminated EC2 instance %s", instance_id)
        except Exception:
            _log.warning("Failed to terminate EC2 instance %s", instance_id, exc_info=True)

    return _terminate


def terminate_ray_instances_by_tag(
    cluster_name: str,
    *,
    region: str = "us-west-2",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    prefix_match: bool = False,
) -> None:
    """Terminate all running/pending EC2 instances belonging to a Ray cluster.

    Fallback used when the resolved cluster YAML is unavailable (e.g. the
    flow was cancelled before ``ray up`` wrote it). Finds instances by
    the ``ray-cluster-name`` tag that Ray sets on every node it launches.

    Args:
        cluster_name: Value (or prefix) of the ``ray-cluster-name`` EC2 tag.
        region: AWS region.
        log: Optional logger; silently swallows errors if ``None``.
        prefix_match: If True, match clusters whose ``ray-cluster-name``
            *starts with* ``cluster_name``. Use with caution — this will
            terminate instances from ALL matching clusters.
    """
    _log = log or logging.getLogger(__name__)
    try:
        ec2 = boto3.client("ec2", region_name=region)
        tag_value = f"{cluster_name}*" if prefix_match else cluster_name
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "tag:ray-cluster-name", "Values": [tag_value]},
                {"Name": "instance-state-name", "Values": ["running", "pending"]},
            ],
        )
        instance_ids = [i["InstanceId"] for r in resp.get("Reservations", []) for i in r.get("Instances", [])]
        if instance_ids:
            _log.info("Terminating %d Ray instances: %s", len(instance_ids), instance_ids)
            ec2.terminate_instances(InstanceIds=instance_ids)
        else:
            _log.info("No running Ray instances found for cluster '%s'", cluster_name)
    except Exception:
        _log.exception("Failed to terminate Ray instances by tag")


def _stop_ray_cluster(
    cluster_yaml: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> bool:
    """Tear down the Ray cluster via ``ray down``. Best-effort — does not raise.

    Returns True if ``ray down`` exited 0, False otherwise (caller may then fall
    back to tag-based termination so a head that ``ray down`` couldn't reach
    doesn't leak).

    BOUNDED by ``RAY_DOWN_TIMEOUT_S``, for the reason that constant documents: this
    SSHes into the head, and an unreachable head makes it hang indefinitely. Unbounded,
    the fallback in the sentence above is unreachable exactly when it is needed — the
    call never returns, so nothing terminates the fleet by tag and the GPUs bill on. A
    timeout is therefore reported as a failed ``ray down``, not raised.
    """
    log.info("Tearing down Ray cluster")
    try:
        result = subprocess.run(["ray", "down", cluster_yaml, "-y"], check=False, timeout=RAY_DOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        log.warning(
            "ray down did not finish within %ds — treating as failed so the caller falls "
            "back to terminating instances by ray-cluster-name tag.",
            RAY_DOWN_TIMEOUT_S,
        )
        return False
    if result.returncode == 0:
        log.info("Ray cluster stopped")
        return True
    log.warning(
        "ray down exited with code %d — cluster may still be running; idle workers "
        "self-drain after the idle timeout, but the head node does NOT self-terminate. "
        "Terminate by ray-cluster-name tag if this persists.",
        result.returncode,
    )
    return False


@contextlib.contextmanager
def ray_cluster(
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    ami_ssm_name: str,
    ami_id: str | None = None,
    ray_address: str | None = None,
    cluster_yaml: str | Path | None = None,
    cluster_name: str | None = None,
    region: str = "us-west-2",
    ssm_prefix: str = DEFAULT_SSM_PREFIX,
    instance_tags: list[dict[str, str]] | None = None,
    sync_source_path: Path | None = None,
    code_bucket: str | None = None,
    code_suffix: str = "",
    cloudwatch_log_group: str = DEFAULT_CLOUDWATCH_LOG_GROUP,
    cloudwatch_template: Path = DEFAULT_CLOUDWATCH_TEMPLATE,
    idle_timeout_minutes: int | None = None,
    launch_pacing: bool = False,
    gpu_fallback_instance_types: Sequence[str] = (),
    gpu_fallback_vcpu_budget: int | None = None,
) -> Iterator[str | None]:
    """Provision an AWS-backed Ray cluster; tear it down on exit.

    Two modes:

    * ``ray_address`` set — connect to an existing cluster at that
      address. No ``ray up``/``ray down`` is performed; the caller owns
      the lifecycle.
    * Default — call ``ray up`` against the resolved cluster YAML, then
      ``ray.init`` against the head node, and tear the cluster down on
      exit.

    Args:
        log: Logger.
        ami_ssm_name: SSM parameter name holding the worker AMI ID.
            Required even when ``ray_address`` is given (kept as a
            consistent signature; ignored in that path). Used only when
            ``ami_id`` is not given.
        ami_id: Pre-resolved AMI ID that PINS the worker image (bypasses the
            ``ami_ssm_name`` lookup). The campaign threads the AMI it resolved
            once into every fill so a mid-campaign re-bake can't boot a
            different image than the fill's staging fingerprint recorded.
        ray_address: Connect to an existing cluster instead of launching one.
        cluster_yaml: Path to the cluster YAML template. Defaults to the
            template shipped at :data:`DEFAULT_CLUSTER_TEMPLATE`.
        cluster_name: Override the YAML's ``cluster_name``. When ``None``,
            an auto-generated name is used so concurrent clusters get
            distinct EC2 tags.
        region: AWS region (must match the SSM region).
        ssm_prefix: SSM Parameter Store prefix for Ray resource IDs.
        instance_tags: EC2 tags applied to every node.
        sync_source_path: When provided, tar this directory and upload it
            to ``s3://{code_bucket}/code/src{code_suffix}.tar.gz`` before
            ``ray up``. Use for dev iteration; production deployments
            should bake source into the AMI and leave this ``None``.
        code_bucket: S3 bucket for the source tarball. Required when
            ``sync_source_path`` is set.
        code_suffix: Filename suffix for the tarball.
        cloudwatch_log_group: CloudWatch log group for Ray agent logs.
        cloudwatch_template: CloudWatch agent JSON template path.
        idle_timeout_minutes: Optional override of the template's autoscaler
            idle-down delay; ``None`` keeps the template's value. Rationale at
            :func:`_resolve_ray_config`.
        launch_pacing: Pace this cluster's EC2 launch requests against the
            account's shared RunInstances quota — see :data:`LAUNCH_PACING_ENV`.
        gpu_fallback_instance_types: Card names this cluster may fall back to when the production
            GPU rung has no capacity -- see :data:`GPU_FALLBACK_CARDS`. Empty (the
            default) leaves behaviour exactly as it is today.
        gpu_fallback_vcpu_budget: Optional G-and-VT vCPU budget for this cluster's GPU
            fleet. Rungs are not equally priced -- the fallback sizes are 8 vCPU per GPU
            against the production rung's 4 -- so a ceiling in NODES means a different
            quota bill depending on which card wins. Given a budget, each rung is
            ceilinged at what the budget affords IT.
            Default ``False`` keeps today's launch behaviour. Pass ``True`` when
            several clusters grow concurrently; one cluster alone contends with
            nothing and gains only a smaller call count. Ignored in the
            ``ray_address`` path, which provisions nothing.

    Yields:
        Path to the resolved cluster YAML tempfile when this context
        manages the cluster lifecycle, or ``None`` when ``ray_address``
        was supplied. The yielded path is intended for an
        on-cancellation hook to call :func:`cleanup_ray_tempfiles`.
    """
    resolved_yaml: str | None = None
    manages_cluster = False
    cluster_yaml = Path(cluster_yaml) if cluster_yaml is not None else DEFAULT_CLUSTER_TEMPLATE
    try:
        if ray_address:
            log.info("Connecting to Ray at %s", ray_address)
            ray.init(address=ray_address, ignore_reinit_error=True)
        else:
            manages_cluster = True
            if cluster_name is None:
                with cluster_yaml.open() as f:
                    base_name = yaml.safe_load(f).get("cluster_name", "tessera-inference")
                suffix = uuid.uuid4().hex[:8]
                cluster_name = f"{base_name}-{suffix}"
            log.info("Using cluster name: %s", cluster_name)

            if sync_source_path is not None:
                if not code_bucket:
                    raise ValueError("code_bucket is required when sync_source_path is set")
                s3_key = f"code/src{code_suffix}.tar.gz"
                log.info("Syncing %s → s3://%s/%s", sync_source_path, code_bucket, s3_key)
                _sync_code_to_s3(sync_source_path, code_bucket, s3_key)

            # Resolve BEFORE launching so `resolved_yaml` is bound when a
            # failed `ray up` unwinds to the finally-block: a partial launch
            # can leave a provisioned head behind, and `ray down` against the
            # unresolved template (whose cluster_name lacks the uuid suffix)
            # matches nothing — that exact path leaked a head on 2026-07-16.
            # _resolve_ray_config lists every private subnet on every node
            # type (multi-AZ with launch-time capacity failover).
            log.info("Resolving Ray cluster config from SSM (cluster_name=%s)", cluster_name)
            resolved_yaml = _resolve_ray_config(
                cluster_yaml,
                region=region,
                ami_ssm_name=ami_ssm_name,
                ami_id=ami_id,
                ssm_prefix=ssm_prefix,
                cluster_name=cluster_name,
                instance_tags=instance_tags,
                code_bucket=code_bucket,
                code_suffix=code_suffix,
                cloudwatch_log_group=cloudwatch_log_group,
                cloudwatch_template=cloudwatch_template,
                idle_timeout_minutes=idle_timeout_minutes,
                launch_pacing=launch_pacing,
                gpu_fallback_instance_types=gpu_fallback_instance_types,
                gpu_fallback_vcpu_budget=gpu_fallback_vcpu_budget,
            )
            head_ip = _start_ray_cluster(resolved_yaml, log, launch_pacing=launch_pacing)
            head_address = f"ray://{head_ip}:10001"
            log.info("Connecting to Ray at %s", head_address)
            ray.init(address=head_address, ignore_reinit_error=True)

            _log_ray_dashboard_ssm_command(cluster_name, log, region=region)

        yield resolved_yaml
    finally:
        ray.shutdown()
        if manages_cluster:
            if resolved_yaml is None:
                # Config resolution failed before launch, so no resolved YAML
                # exists. `ray down` on the UNRESOLVED template would target its
                # base cluster_name (no flow-specific uuid suffix) and could tear
                # down an unrelated `tessera-inference` cluster — skip it and
                # terminate anything tagged with our exact cluster_name instead.
                if cluster_name:
                    terminate_ray_instances_by_tag(cluster_name=cluster_name, region=region, log=log)
            elif not _stop_ray_cluster(resolved_yaml, log) and cluster_name:
                # When `ray down` can't tear the cluster down (unreachable head,
                # stale YAML), fall back to exact-tag termination so a normally-
                # completed run can't leave the fleet billing. The
                # cancellation/crash hook only covers cancelled/crashed flows,
                # not this path.
                terminate_ray_instances_by_tag(cluster_name=cluster_name, region=region, log=log)
            cleanup_ray_tempfiles(resolved_yaml)
