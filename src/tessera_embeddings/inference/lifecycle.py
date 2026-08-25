"""Ray actor lifecycle helpers.

Pure-domain support code for ``inference.runner.run_inference``. Lives in
the domain layer (no Prefect, no AWS-specific code) so it can be reused by
the plain runner, by tests, and by Prefect tasks alike.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import ray

#: How long to wait for a fleet's FIRST actor before giving the fill up.
#:
#: Long, because of what this bound actually gates. The wait ends the moment ``min_required``
#: actors are ready and lets the rest join while work proceeds, so in a healthy account it is
#: never reached — a fleet that is filling normally leaves after the first actor answers. It binds
#: only when the account cannot supply a SINGLE GPU instance, which is a property of the region's
#: spare capacity rather than of anything the fill did wrong.
#:
#: That makes the two outcomes badly asymmetric. Waiting costs an idle head node and no GPU at
#: all. Giving up fails the fill, and a chained fill owns a whole roster of zones — so it takes
#: the ingest it would have dispatched with it, and the work waits for the driver's next
#: re-dispatch round rather than for capacity. Erring long is much cheaper than erring short.
#:
#: **But not unbounded, and that is the reason there is a number here at all.** A misconfigured
#: launch — a stale image, a revoked permission, an exhausted subnet — presents exactly as a
#: drought does: zero actors arriving. This bound is the only thing that distinguishes the two, so
#: removing it converts a failure that says what went wrong into a run that waits forever looking
#: patient. It would also stop the driver's dispatch round from ever closing, since a round closes
#: only when every fill returns, which is the very recovery a starved fill needs.
#:
#: So: long enough to outlast any real drought, short enough that nothing legitimate reaches it.
#: Placing one instance does not take hours when the capacity exists.
ACTOR_INIT_TIMEOUT_SEC = 21600
"""Maximum wall-clock seconds to wait for actor ``__init__`` to complete.

Sized for the worst case: instance launch + container pull + checkpoint
download + model load to GPU. Tune via ``run_inference``'s parameter rather
than monkey-patching this module-level default.
"""


def wait_for_actors(
    actors: list[ray.actor.ActorHandle],
    num_requested: int,
    min_required: int,
    started_at: float,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    init_timeout_sec: float = ACTOR_INIT_TIMEOUT_SEC,
) -> tuple[list[ray.actor.ActorHandle], list[str], set[int]]:
    """Wait for enough actors to initialize, then return all actors with readiness info.

    Waits up to ``init_timeout_sec`` for at least ``min_required`` of the
    requested actors to be ready. Once that threshold is met, returns
    immediately so work can start. Actors that aren't ready yet are still
    included in the returned lists — the work-stealing scheduler treats
    them as initializing and will dispatch work to them once they come
    online.

    Callers typically pass ``min_required=1`` so the run starts as soon as
    the first actor is live: cloud providers roll out instances with large
    timing variation, and waiting for a fraction of the fleet just stalls
    the run. The 30s poll window below still scoops up any fast-arriving
    actors before returning.

    Args:
        actors: List of actor handles (just created, not yet confirmed ready).
        num_requested: Original number of actors requested.
        min_required: Minimum number of actors that must be ready before work
            starts. Usually 1 so the run starts with the first live actor and
            the rest join via work-stealing.
        started_at: Monotonic timestamp when actor creation started.
        log: Logger.
        init_timeout_sec: Per-call timeout override.

    Returns:
        Tuple of (actors, actor_instance_ids, still_initializing). Instance
        IDs are real for ready actors and ``"pending-init"`` placeholders
        for still-initializing ones. ``still_initializing`` is the set of
        actor indices that haven't responded yet.

    Raises:
        RuntimeError: If fewer than ``min_required`` actors initialized
            within the timeout.
    """
    ping_refs: list[Any] = [a.ping.remote() for a in actors]  # type: ignore[union-attr]
    ref_to_idx: dict[Any, int] = {ref: i for i, ref in enumerate(ping_refs)}

    ready_indices: set[int] = set()
    pending: list[Any] = list(ping_refs)

    # Poll with 30s sub-timeouts so progress logs land while waiting. Once
    # min_required actors have responded we break out and let the
    # scheduler start; the still-pending actors stay in the list with
    # their indices in ``still_initializing``. Ray queues calls to those
    # actors behind their __init__, so once they finish loading the model
    # they begin processing whatever the scheduler has already dispatched.
    deadline = time.monotonic() + init_timeout_sec
    while pending and time.monotonic() < deadline:
        remaining_sec = max(0, deadline - time.monotonic())
        timeout = min(remaining_sec, 30.0)
        done, pending = ray.wait(pending, num_returns=len(pending), timeout=timeout)  # type: ignore[attr-defined]
        for ref in done:
            try:
                ray.get(ref)  # type: ignore[attr-defined]
                ready_indices.add(ref_to_idx[ref])
            except Exception as exc:
                idx = ref_to_idx[ref]
                log.warning("Actor %d failed during init: %s", idx, exc)
        log.info(
            "%d / %d actors ready (%.0fs elapsed)",
            len(ready_indices),
            num_requested,
            time.monotonic() - started_at,
        )
        if len(ready_indices) >= min_required and not pending:
            break
        if len(ready_indices) >= min_required and pending:
            log.info(
                "%d actors still initializing — starting work with %d ready actors; "
                "remaining actors will join as they come online",
                len(pending),
                len(ready_indices),
            )
            break

    if len(ready_indices) < min_required:
        msg = (
            f"Only {len(ready_indices)} / {num_requested} actors initialized "
            f"within {init_timeout_sec:.0f}s (minimum required: {min_required}). Cannot proceed."
        )
        raise RuntimeError(msg)

    # Resolve instance IDs for ready actors now (they're confirmed live, so
    # this is fast). Still-initializing actors get placeholders; their real
    # IDs are resolved lazily by the scheduler once the actor responds.
    actor_instance_ids: list[str] = ["pending-init"] * len(actors)
    if ready_indices:
        sorted_ready = sorted(ready_indices)
        iid_refs = [actors[i].get_instance_id.remote() for i in sorted_ready]  # type: ignore[union-attr]
        iids: list[str] = ray.get(iid_refs, timeout=30)  # type: ignore[attr-defined]
        for i, iid in zip(sorted_ready, iids, strict=True):
            actor_instance_ids[i] = iid

    still_initializing = set(range(len(actors))) - ready_indices
    for i, iid in enumerate(actor_instance_ids):
        status = " (still initializing)" if i in still_initializing else ""
        log.info("Actor %d → %s%s", i, iid, status)

    return actors, actor_instance_ids, still_initializing
