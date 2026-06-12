"""Domain-pure inference runner.

Orchestrates a Ray-based embedding-inference run end-to-end without any
Prefect or cloud-provider coupling: create actors, wait for readiness,
dispatch chunks via the work-stealing scheduler, tear down actors. The
caller is responsible for connecting to Ray (``ray.init`` or attaching to
an existing cluster) and supplying the ``InferenceConfig``.

Output is written to staging via :class:`ZarrWriter`; final assembly is a
separate step (:meth:`ZarrWriter.assemble`). This separation keeps GPU
actors free of icechunk write contention.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.inference.actors import InferenceActor
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.lifecycle import MAX_ACTORS_TO_WAIT_FOR, MIN_ACTOR_FRACTION, wait_for_actors
from tessera_embeddings.inference.progress import ProgressTracker
from tessera_embeddings.inference.scheduling import _process_chunks_work_stealing


def run_inference(
    num_actors: int,
    config: InferenceConfig,
    chunks: list[ChunkSpec],
    mosaic_base: str,
    staging_base: str,
    run_id: str,
    t0: float,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    on_actor_retire: Callable[[str], None] | None = None,
) -> list[dict]:
    """Create Ray actors, run work-stealing inference, return per-chunk results.

    Pure-domain function — no Prefect, no cloud SDKs. Caller must have
    already connected to Ray (e.g. via ``ray.init`` or an attached
    cluster) and ensured a placement group exists if one is required.

    Args:
        num_actors: Number of GPU/CPU inference actors to create. Must be
            at least 1.
        config: Inference configuration. ``config.num_gpus`` controls the
            per-actor resource reservation passed to ``.options()``.
        chunks: Spatial chunks to process.
        mosaic_base: Base path for input mosaic stores (any fsspec URI).
        staging_base: Base path for staged output stores.
        run_id: Unique run identifier; used for resume detection and
            staging-path namespacing.
        t0: Monotonic timestamp from the run's start, used for elapsed
            logging in the scheduler.
        log: Logger.
        on_actor_retire: Optional callback ``(instance_id) -> None`` invoked
            when a misbehaving actor is removed from the pool. The AWS
            provider injects an EC2-terminator here so dead instances stop
            billing immediately; the local provider passes ``None``.

    Returns:
        Per-chunk result dicts (status, valid pixel count, timing, etc.),
        with ``"resumed": True`` on entries that were already staged from
        a prior run.

    Raises:
        ValueError: If ``num_actors < 1``.
        RuntimeError: If too few actors initialize within the timeout (see
            :func:`tessera_embeddings.inference.lifecycle.wait_for_actors`).
    """
    if num_actors < 1:
        raise ValueError(f"num_actors must be >= 1, got {num_actors}")

    # --- Resume check: skip chunks already staged from a prior run ---
    writer = ZarrWriter(staging_base)
    already_done = writer.scan_existing_staged_chunks(
        run_id,
        chunks,
        compute_std=config.compute_std,
        log=log,
    )
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
            return [
                {"chunk": label, "status": "success", "valid_pixels": 0, "elapsed_sec": 0.0, "resumed": True}
                for label in already_done
            ]

    # --- Create actors ---
    log.info(
        "Creating %d inference actors (num_gpus=%s, checkpoint: %s)",
        num_actors,
        config.num_gpus,
        config.checkpoint_path,
    )
    t_actors = time.monotonic()
    actor_cls = InferenceActor.options(num_gpus=config.num_gpus)  # type: ignore[attr-defined]
    actors = [actor_cls.remote(config, config.checkpoint_path) for _ in range(num_actors)]
    progress_tracker: ray.actor.ActorHandle | None = None
    try:
        min_required = min(max(1, int(num_actors * MIN_ACTOR_FRACTION)), MAX_ACTORS_TO_WAIT_FOR)
        log.info("Waiting for actors to initialize (need at least %d / %d)...", min_required, num_actors)
        actors, actor_instance_ids, still_initializing = wait_for_actors(
            actors, num_actors, min_required, t_actors, log
        )

        # --- Process chunks with work-stealing ---
        log.info("Processing %d chunks across %d actors (work-stealing)", len(chunks), len(actors))

        # Pin the ProgressTracker to a node with no GPUs (typically the head
        # node in a Ray cluster). Tracker is lightweight; keeping it off
        # GPU workers prevents it from competing with actors for memory and
        # avoids losing progress state when a worker dies.
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
            t0=t0,
            log=log,
            tracker=progress_tracker,
            still_initializing=still_initializing,
            on_actor_retire=on_actor_retire,
        )
    finally:
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
        resumed = [
            {"chunk": label, "status": "success", "valid_pixels": 0, "elapsed_sec": 0.0, "resumed": True}
            for label in already_done
        ]
        results = resumed + results

    return results
