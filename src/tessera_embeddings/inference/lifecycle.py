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

ACTOR_INIT_TIMEOUT_SEC = 600
"""Maximum wall-clock seconds to wait for actor ``__init__`` to complete.

Sized for the worst case: instance launch + container pull + checkpoint
download + model load to GPU. Tune via ``run_inference``'s parameter rather
than monkey-patching this module-level default.
"""

MIN_ACTOR_FRACTION = 0.25
"""Fraction of requested actors that must be ready before work can start.

A quarter of the fleet is enough to make progress while the rest catches
up. The work-stealing scheduler will dispatch chunks to actors that come
online later, so we don't need to wait for everyone.
"""


def wait_for_actors(
    actors: list[ray.actor.ActorHandle],
    num_requested: int,
    started_at: float,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    init_timeout_sec: float = ACTOR_INIT_TIMEOUT_SEC,
    min_actor_fraction: float = MIN_ACTOR_FRACTION,
) -> tuple[list[ray.actor.ActorHandle], list[str], set[int]]:
    """Wait for enough actors to initialize, then return all actors with readiness info.

    Waits up to ``init_timeout_sec`` for at least ``min_actor_fraction`` of
    the requested actors to be ready. Once the minimum threshold is met,
    returns immediately so work can start. Actors that aren't ready yet
    are still included in the returned lists — the work-stealing scheduler
    treats them as initializing and will dispatch work to them once they
    come online.

    Args:
        actors: List of actor handles (just created, not yet confirmed ready).
        num_requested: Original number of actors requested.
        started_at: Monotonic timestamp when actor creation started.
        log: Logger.
        init_timeout_sec: Per-call timeout override.
        min_actor_fraction: Fraction of requested actors that must be ready
            before this function returns. Defaults to
            :data:`MIN_ACTOR_FRACTION`.

    Returns:
        Tuple of (actors, actor_instance_ids, still_initializing). Instance
        IDs are real for ready actors and ``"pending-init"`` placeholders
        for still-initializing ones. ``still_initializing`` is the set of
        actor indices that haven't responded yet.

    Raises:
        RuntimeError: If fewer than ``min_actor_fraction`` of the requested
            actors initialized within the timeout.
    """
    min_required = max(1, int(num_requested * min_actor_fraction))
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
