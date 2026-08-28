"""Node-type scoring that reacts to capacity refusals.

Ray's autoscaler picks a node type for unfulfilled demand by scoring every type and
taking the best. **Its default scorer cannot see capacity.**
``_resource_based_utilization_scorer`` accepts a ``node_availability_summary`` and
never reads it, so a rung that refuses with ``InsufficientInstanceCapacity`` stays
top-scored and is re-requested indefinitely. A second rung sitting ready underneath
it is never reached, however long the refusal lasts.

Ray populates the summary regardless: ``AWSNodeProvider`` turns a launch
``ClientError`` into a ``NodeLaunchException`` carrying the AWS error code, and
``NodeLauncher`` records the type unavailable — then records it available again on
the next successful launch of that type. The information is there; nothing consumes
it. This module is the consumer, installed through Ray's own extension point
(``RAY_AUTOSCALER_UTILIZATION_SCORER``).

**Demotion, not exclusion.** An unavailable type keeps a real score and only loses
the leading term, so it ranks below every available type that fits and above nothing
else. If EVERY type is unavailable the ordering among them is unchanged and the
fleet still tries to launch — this can slow a scale-up, never wedge it.

**Capacity refusals only.** A throttle (``RequestLimitExceeded``) or a quota refusal
(``InstanceLimitExceeded``) is not a reason to move the fleet onto a different card:
the first clears in seconds, and the second gets worse on a rung that costs more
quota per GPU. Only the two genuine "the region has none" codes demote.

Records expire after ``AUTOSCALER_NODE_AVAILABILITY_MAX_STALENESS_S`` (30 min by
default) and are cleared immediately by a successful launch, so recovery is
automatic and needs no operator action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing only
    from ray.autoscaler._private.node_provider_availability_tracker import (
        NodeAvailabilitySummary,
    )
    from ray.autoscaler._private.util import ResourceDict

#: Dotted path Ray resolves with ``load_function_or_class``. Kept as a constant so
#: the value injected into the head's environment and the function that must answer
#: to it cannot drift apart -- a mismatch would raise inside the autoscaler's
#: constructor and stop the cluster scaling at all.
SCORER_PATH = "tessera_embeddings.providers.aws.autoscaler_scorer.capacity_aware_scorer"

#: The AWS error codes that mean "this region has none of this instance type right
#: now". Deliberately narrow -- see the module docstring on why throttles and quota
#: refusals must NOT move the fleet.
CAPACITY_REFUSAL_CODES = frozenset(
    {
        "InsufficientInstanceCapacity",
        "InsufficientHostCapacity",
    }
)


def _is_capacity_refused(node_type: str, summary: NodeAvailabilitySummary | None) -> bool:
    """True if Ray last saw ``node_type`` refused for want of capacity."""
    if not summary:
        return False
    record = getattr(summary, "node_availabilities", {}).get(node_type)
    if record is None or record.is_available:
        return False
    info = record.unavailable_node_information
    return info is not None and info.category in CAPACITY_REFUSAL_CODES


def capacity_aware_scorer(
    node_resources: ResourceDict,
    resources: list[ResourceDict],
    node_type: str,
    *,
    node_availability_summary: NodeAvailabilitySummary,
) -> tuple[bool, int, float, float] | None:
    """Ray's default score, with capacity-refused node types demoted below the rest.

    Signature is Ray's ``UtilizationScorer`` protocol; it is called once per node type
    per scheduling round with the bundles that still need placing.

    Returns whatever the default scorer returns (``None`` means "nothing fits here",
    which excludes the type), except that a capacity-refused type has its leading
    term cleared so every available type outranks it.
    """
    from ray.autoscaler._private.resource_demand_scheduler import (
        _default_utilization_scorer,
    )

    score = _default_utilization_scorer(
        node_resources,
        resources,
        node_type,
        node_availability_summary=node_availability_summary,
    )
    if score is None or not _is_capacity_refused(node_type, node_availability_summary):
        return score

    # The default score's first element is a bool (`gpu_ok`), and Ray compares these
    # tuples. Clearing it ranks this type below every available type that fits while
    # leaving the remaining terms to order the refused ones among themselves.
    #
    # NOT logged. This runs once per node type per PENDING BUNDLE per scheduling
    # round, so at a 250-actor demand a log line here is thousands of lines a minute
    # on the head. The demotion is visible where it matters -- in which node type the
    # autoscaler launches, and in Ray's own `Failed to launch ... (category)` line
    # that put the record there in the first place.
    return (False, *tuple(score)[1:])
