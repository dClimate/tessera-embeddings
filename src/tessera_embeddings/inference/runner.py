"""Domain-pure inference runner.

Orchestrates a Ray-based embedding-inference run end-to-end without any Prefect or
cloud-provider coupling: create actors, wait for readiness, dispatch chunks via the
work-stealing scheduler, tear down actors. The caller connects to Ray (``ray.init`` or an
existing cluster) and supplies the ``InferenceConfig``.

Output goes to staging via :class:`ZarrWriter`; final assembly is a separate step
(:meth:`ZarrWriter.assemble`), which keeps GPU actors free of icechunk write contention.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.inference.actors import InferenceActor
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.lifecycle import wait_for_actors
from tessera_embeddings.inference.progress import ProgressTracker
from tessera_embeddings.inference.scheduling import FleetDemand, WorkItem, _process_chunks_work_stealing


def _resumed_result(label: str, *, skipped: bool) -> dict:
    """One restored outcome for a tile a previous attempt already staged.

    ``status`` mirrors what the actor reported at the time — ``skipped`` for a tile it
    found nothing to write in, ``success`` otherwise — so a resume and a fresh run agree
    on what happened. The counters are absent rather than zero: this run did not measure
    them, and a zero would read as a measurement of none.
    """
    return {
        "chunk": label,
        "status": "skipped" if skipped else "success",
        "valid_pixels": 0,
        "elapsed_sec": 0.0,
        "resumed": True,
    }


def run_inference(
    num_actors: int,
    config: InferenceConfig,
    chunks: list[ChunkSpec],
    mosaic_base: str,
    staging_base: str,
    run_id: str,
    t0: float,  # noqa: ARG001 — accepted and ignored; see the docstring and public-api.md
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    on_actor_retire: Callable[[str], None] | None = None,
    on_fleet_demand: Callable[[int], None] | None = None,
    get_credentials: Callable[[], Any] | None = None,
    s3_region: str | None = None,
    retire_idle_actors: bool = True,
    more_work: Callable[[], list[WorkItem] | None] | None = None,
    on_item_done: Callable[[WorkItem, dict], None] | None = None,
) -> list[dict]:
    """Create Ray actors, run work-stealing inference, return per-chunk results.

    Pure-domain function — no Prefect, no cloud SDKs. The caller must already be connected
    to Ray and have a placement group if one is required.

    Args:
        num_actors: Number of GPU/CPU inference actors to create. Must be at least 1.
        config: Inference configuration. ``config.num_gpus`` is the per-actor resource
            reservation passed to ``.options()``.
        chunks: Spatial chunks to process.
        mosaic_base: Base path for input mosaic stores (any fsspec URI).
        staging_base: Base path for staged output stores.
        run_id: Unique run identifier, for resume detection and staging-path namespacing.
        t0: **Accepted and ignored.** For a chained session the run's start is the top of
            the whole stream, so this counted the ingest look-ahead, ``ray up``, EC2 bringup
            and model load, and read as though inference had been running that long; it also
            disagreed with the GPU-hours on its own line, which measures actor time. The
            progress line now starts its clock when the dispatch loop does. Kept in the
            signature only because ``run_inference`` is documented in ``docs/public-api.md``
            — remove it on the next deliberate pass at that API.
        log: Logger.
        on_fleet_demand: Optional callback ``(want_gpus) -> None`` letting the AWS provider
            publish a per-instance-type fleet request each scheduling round; see
            ``providers.aws.fleet_mix``.
        on_actor_retire: Optional callback ``(instance_id) -> None`` invoked when a
            misbehaving actor leaves the pool. The AWS provider injects an EC2-terminator
            here so dead instances stop billing immediately; the local provider passes None.
        get_credentials: Optional icechunk S3 credential provider injected into every actor
            so store opens refresh credentials. The AWS provider passes
            ``iam_icechunk_credentials``; the local provider passes ``None`` (icechunk's
            default chain). See :class:`InferenceActor`.
        s3_region: Optional S3 region for the mosaic repos, injected into every actor so its
            reads open the store in the same region the caller's preflight/assembly opens
            use. ``None`` uses icechunk's default region.
        retire_idle_actors: Kill actors idle past the grace period at the run tail (default);
            see ``scheduling._process_chunks_work_stealing`` for when a caller passes False.
        more_work: Optional chained-session work source (see the scheduler's docstring). With
            a source, ``chunks`` is typically empty and every item carries its own
            :class:`~tessera_embeddings.inference.scheduling.ZoneContext`; the single-zone
            resume scan is skipped, since the source's feeder scans per zone before
            enqueueing.
        on_item_done: Optional per-item final-outcome callback (chained sessions use it for
            per-zone completion accounting). Runs on the scheduler thread — must not block.

    Returns:
        Per-chunk result dicts (status, valid pixel count, timing, etc.), with
        ``"resumed": True`` on entries already staged by a prior run.

    Raises:
        ValueError: If ``num_actors < 1``.
        RuntimeError: If too few actors initialize within the timeout (see
            :func:`tessera_embeddings.inference.lifecycle.wait_for_actors`).
    """
    if num_actors < 1:
        raise ValueError(f"num_actors must be >= 1, got {num_actors}")

    # --- Resume check: skip chunks already staged from a prior run ---
    # Gated on a non-empty chunk list: a chained session starts empty (work
    # arrives via more_work, each zone pre-scanned by its feeder), and its
    # placeholder staging_base must not be listed.
    already_done: set[str] = set()
    resumed_skips: set[str] = set()
    if chunks:
        writer = ZarrWriter(staging_base)
        # The ARTIFACT form, not the label-set form: both mean "do not re-infer", but a
        # staged zarr produced pixels while a skip marker recorded that the tile had none.
        # Restoring both as successes makes a resumed zone's tally disagree with a fresh
        # run's, and the year's radar-coverage provenance derives from those statuses —
        # `summarise_radar_coverage` returns None when tiles report no counters, so
        # miscounted skips can suppress the whole summary.
        staged = writer.scan_existing_staged_artifacts(
            run_id,
            chunks,
            compute_std=config.compute_std,
            log=log,
        )
        already_done, resumed_skips = staged.done, staged.skipped
    if already_done:
        remaining = len(chunks) - len(already_done)
        log.info(
            "Resuming: %d/%d chunks already staged, %d remaining",
            len(already_done),
            len(chunks),
            remaining,
        )
        chunks = [c for c in chunks if c.label not in already_done]
        if not chunks:
            log.info("All chunks already staged — skipping inference entirely")
            return [_resumed_result(label, skipped=label in resumed_skips) for label in already_done]

    # --- Create actors ---
    log.info(
        "Creating %d inference actors (num_gpus=%s, checkpoint: %s)",
        num_actors,
        config.num_gpus,
        config.checkpoint_path,
    )
    t_actors = time.monotonic()
    actor_cls = InferenceActor.options(num_gpus=config.num_gpus)  # type: ignore[attr-defined]

    def actor_factory(n: int) -> list[ray.actor.ActorHandle]:
        """Request ``n`` new inference actors (one .remote() each)."""
        return [actor_cls.remote(config, config.checkpoint_path, get_credentials, s3_region) for _ in range(n)]

    # Request the first batch up front; the work-stealing scheduler requests the rest as
    # slots are placed (scheduling._maybe_request_next_batch). The size is the config's to
    # decide — the batching sentinel and the headroom's cold start live in one method there,
    # so this cannot drift from what the scheduler goes on to do.
    batch_size = config.actor_request_batch_size
    first_batch = config.initial_actor_request(num_actors)
    # BEFORE the first batch, because the scheduler republishes every round but does not
    # start until `wait_for_actors` returns — and under a total primary drought no
    # first-batch actor ever initializes, so that wait runs to its hours-long timeout and
    # the run dies having never once asked for the fallback.
    fleet = FleetDemand(on_fleet_demand, config.num_gpus, log)
    outstanding = num_actors if more_work is not None else len(chunks)
    fleet.send(target=num_actors, outstanding=outstanding, requested=first_batch, retry=True)

    actors: list[ray.actor.ActorHandle] = []
    progress_tracker: ray.actor.ActorHandle | None = None
    # The try opens BEFORE the first batch, because the fleet floor is already published
    # by this point: an actor-creation failure outside this scope would leave that floor
    # standing, holding GPU nodes on a cluster where inference never started.
    try:
        actors = actor_factory(first_batch)
        if batch_size > 0 and first_batch < num_actors:
            log.info(
                "Actor batching enabled: requesting %d at a time (batch 1/%d actors now)",
                batch_size,
                num_actors,
            )
        # Start as soon as a single actor is live: cloud providers roll instances out with
        # huge timing variation, so blocking on a fraction of the fleet just stalls the run.
        # The work-stealing scheduler dispatches to the rest as they come online.
        min_required = 1
        log.info("Waiting for actors to initialize (need at least %d / %d)...", min_required, num_actors)
        actors, actor_instance_ids, still_initializing = wait_for_actors(
            actors, len(actors), min_required, t_actors, log
        )

        # --- Process chunks with work-stealing ---
        if more_work is not None:
            log.info("Processing a chained multi-zone stream across %d actors (work-stealing)", len(actors))
        else:
            log.info("Processing %d chunks across %d actors (work-stealing)", len(chunks), len(actors))

        # Pin the lightweight ProgressTracker to a GPU-less node (typically the head): off
        # the GPU workers it neither competes with actors for memory nor loses progress
        # state when a worker dies.
        head_nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) == 0]  # type: ignore[attr-defined]
        if head_nodes:
            head_strategy = NodeAffinitySchedulingStrategy(node_id=head_nodes[0]["NodeID"], soft=True)
            progress_tracker = ProgressTracker.options(scheduling_strategy=head_strategy).remote()  # type: ignore[attr-defined]
            log.info("ProgressTracker pinned to head node %s", head_nodes[0]["NodeID"][:8])
        else:
            log.warning("Could not identify head node — ProgressTracker will use default scheduling")
            progress_tracker = ProgressTracker.remote()  # type: ignore[attr-defined]

        results = _process_chunks_work_stealing(
            actors=actors,
            actor_instance_ids=actor_instance_ids,
            chunks=chunks,
            mosaic_base=mosaic_base,
            staging_base=staging_base,
            run_id=run_id,
            config=config,
            log=log,
            tracker=progress_tracker,
            still_initializing=still_initializing,
            on_actor_retire=on_actor_retire,
            fleet=fleet,
            get_credentials=get_credentials,
            s3_region=s3_region,
            actor_factory=actor_factory,
            total_actors_target=num_actors,
            placement_timeout_sec=config.actor_batch_placement_timeout_sec,
            retire_idle_actors=retire_idle_actors,
            more_work=more_work,
            on_item_done=on_item_done,
        )
    finally:
        fleet.clear()
        log.info("Killing %d actors to release resource reservations", len(actors))
        for actor in actors:
            with contextlib.suppress(Exception):
                ray.kill(actor)  # type: ignore[attr-defined]
        del actors
        if progress_tracker is not None:
            with contextlib.suppress(Exception):
                ray.kill(progress_tracker)  # type: ignore[attr-defined]

    # Merge resumed chunks (already staged) with newly-processed results
    if already_done:
        resumed = [_resumed_result(label, skipped=label in resumed_skips) for label in already_done]
        results = resumed + results

    return results
