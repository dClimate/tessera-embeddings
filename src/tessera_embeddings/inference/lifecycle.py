"""Ray actor lifecycle helpers for ``inference.runner.run_inference``.

Pure domain layer — no Prefect, no AWS-specific code — so the plain runner, tests and Prefect
tasks can all reuse it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import ray

#: How long to wait for a fleet's FIRST actor before giving the fill up (6 h).
#:
#: The wait ends as soon as ``min_required`` actors answer, so a healthy fill never reaches it; it
#: binds only when the region cannot supply a single GPU instance. The outcomes are asymmetric —
#: waiting costs an idle head node, while giving up fails a chained fill that owns a whole roster
#: of zones and takes its ingest with it — so err long. Bounded rather than infinite because a
#: misconfigured launch (stale image, revoked permission, exhausted subnet) looks exactly like a
#: capacity drought, and this number is the only thing that turns "waits forever looking patient"
#: back into a failure that says what went wrong. A driver dispatch round closes only when every
#: fill returns, so an unbounded wait would also block the re-dispatch a starved fill needs.
#: Sizing history: context_docs/design/immediate-refill-of-a-settled-fill.md.
ACTOR_INIT_TIMEOUT_SEC = 21600
"""Maximum wall-clock seconds to wait for actor ``__init__``: instance launch, container pull,
checkpoint download and model load to GPU. Override via ``run_inference``, not by patching this."""


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

    Returns as soon as ``min_required`` of the requested actors are ready, or raises at
    ``init_timeout_sec``. Not-yet-ready actors stay in the returned lists; the work-stealing
    scheduler dispatches to them once they come online. Callers typically pass
    ``min_required=1`` — instance launch times vary widely, so waiting for a fraction of the
    fleet just stalls the run, and the 30 s poll window still scoops up fast arrivals.

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

    # 30 s sub-timeouts so progress logs land while waiting. Ray queues calls to a still-pending
    # actor behind its __init__, so it picks up whatever the scheduler already dispatched.
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

    # Ready actors are confirmed live, so resolving their instance IDs is fast; the rest get
    # placeholders and the scheduler resolves them lazily once the actor responds.
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
