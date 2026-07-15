"""Zone-fill runner: one (zone, year) of the global store, end to end.

The orchestration-facing composition for the global campaign (ADR-008,
implementation plan W5): enumerate the zone's shard-aligned tile grid against
the **partner-supplied campaign land mask**, run Ray inference over the live
tiles, assemble the staged tiles into whole shards of the pre-seeded zone
group, and tag the landed commit. The Prefect/AWS wiring (work queues, the
fleet-wide commit gate as a Prefect global concurrency limit, credentials)
lives downstream — this module ships the plain callable it wraps.

Contracts (all caller-owned):

- **Ray**: the caller is already inside a Ray context (``ray.init`` or an
  attached cluster), exactly as for
  :func:`tessera_embeddings.inference.runner.run_inference`.
- **Land mask**: ``land_mask_path`` is a boolean zarr on the zone's pixel grid
  (same contract as the single-ROI mask). It is partner-supplied and assumed
  delivered — no fallback mask is built (plan Q8).
- **Zone group**: already seeded
  (:func:`tessera_embeddings.storage.global_store.seed_zone_groups`); the
  group's array shape and shard size are the grid authority. Nothing is
  created or resized here (D1).
- **Ingest mosaics**: cover the zone grid exactly; a mismatch is a loud error.
- **Commit gating**: ``gate`` bounds concurrent commits within this process;
  fleet-wide gating across machines is the orchestrator's job (D6, ≤4-8
  simultaneous committers).
- **One fill per zone at a time**: concurrent fills of *different* zones
  commit to disjoint groups and rebase cleanly, but two concurrent fills of
  the *same* zone (different years) both rewrite that group's
  ``years_complete``/``runs`` attrs and icechunk's ``ConflictDetector`` cannot
  auto-merge attribute conflicts — the loser raises ``RebaseFailedError``.
  Schedule zone-parallel, year-serial.

An all-ocean cell (the mask selects no tiles) — or a cell whose live tiles all
skip (zero valid pixels) — is legitimate: it is marked complete with no data
(:func:`~tessera_embeddings.storage.campaign.mark_zone_year_empty`) and tagged,
so the campaign work list converges. A cell that is already complete *and*
tagged short-circuits without re-running anything (a deliberate refill
requires deleting the ``zone-{zone}-{year}`` tag first — campaign history is
never silently rewritten).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any, cast

import icechunk
import zarr

from tessera_embeddings.config.assembly import AssemblyConfig
from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.chunk_spec import enumerate_chunks, filter_chunks_by_roi_mask
from tessera_embeddings.inference.orchestration_helpers import enumerate_mosaic_chunks
from tessera_embeddings.inference.runner import run_inference
from tessera_embeddings.storage.campaign import mark_zone_year_empty, tag_zone_year, zone_year_tag
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.shard_writer import CommitGate


def fill_zone_year(
    *,
    store_path: str,
    zone: str,
    year: int,
    land_mask_path: str,
    mosaic_base: str,
    staging_base: str,
    config: InferenceConfig,
    num_actors: int,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    run_id: str | None = None,
    gate: CommitGate | None = None,
    n_assembly_workers: int | None = None,
    cleanup_staging: bool = True,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
    on_actor_retire: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fill one (zone, year): mask → inference → shard assembly → tag.

    Args:
        store_path: URI of the global Icechunk repo (``BucketPaths.global_store()``).
        zone: Zone group name (EPSG code string, e.g. ``"32601"``).
        year: Campaign calendar year — must be on the group's pre-allocated axis.
        land_mask_path: Partner-supplied boolean zarr on the zone pixel grid.
        mosaic_base: Base path of the zone's ingest mosaic stores.
        staging_base: Base path for staged inference output.
        config: Inference configuration; ``config.chunk_size`` must equal the
            group's shard size (1 tile == 1 shard, D3).
        num_actors: Ray inference actor count.
        log: Logger (e.g. Prefect's run logger downstream).
        run_id: Run identifier; a fresh one is minted when omitted. Reuse a
            prior run's id to resume it (staged tiles are skipped, the year
            index is overwritten idempotently).
        gate: Optional in-process commit gate shared across concurrent fills.
        n_assembly_workers: Assembly worker-process count; defaults to
            ``AssemblyConfig`` sizing from the live-tile count.
        cleanup_staging: Delete staged tiles after a successful fill.
        get_credentials: Optional icechunk credential callback (actors + store).
        s3_region: Optional S3 region override for the global store.
        on_actor_retire: Optional callback when a misbehaving actor is retired
            (the AWS provider injects an EC2 terminator).

    Returns:
        Summary dict: zone, year, run_id, snapshot_id, tag, tile counts,
        inference outcome counts, ``empty`` flag, and elapsed seconds.
    """
    t0 = time.monotonic()
    run_id = run_id or uuid.uuid4().hex[:12]

    # The seeded zone group is the grid authority.
    repo = open_global_repo(store_path, get_credentials=get_credentials, region=s3_region)
    root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    if zone not in root:
        raise ValueError(f"Zone group {zone!r} is not seeded in {store_path} — run seed_zone_groups first (D1).")
    node = cast(zarr.Group, root[zone])

    # Idempotent retry: a cell that already landed AND was tagged is done —
    # re-running it would produce a new snapshot that tag_zone_year rightly
    # refuses to move the tag to. Deliberate refills must delete the tag first.
    landed_years = node.attrs.get("years_complete", [])
    tag = zone_year_tag(zone, year)
    if isinstance(landed_years, list) and year in landed_years and tag in repo.list_tags():
        snapshot = repo.lookup_tag(tag)
        log.info("Zone %s year %d already complete and tagged (%s) — nothing to do", zone, year, tag)
        return {
            "zone": zone,
            "year": year,
            "run_id": run_id,
            "already_complete": True,
            "snapshot_id": snapshot,
            "tag": tag,
            "elapsed_sec": time.monotonic() - t0,
        }

    emb = cast(zarr.Array, node["embeddings"])
    _, ny, nx, _ = emb.shape
    shard_px = (emb.shards or emb.chunks)[1]
    if config.chunk_size != shard_px:
        raise ValueError(
            f"config.chunk_size={config.chunk_size} but {zone} shards are {shard_px} px — "
            "the global write path requires 1 inference tile == 1 shard (ADR-008 D3)."
        )

    # The ingest mosaics must be on the zone grid exactly (campaign contract).
    _, total_y, total_x = enumerate_mosaic_chunks(mosaic_base, shard_px, log)
    if (total_y, total_x) != (ny, nx):
        raise ValueError(
            f"Mosaic grid ({total_y} x {total_x}) at {mosaic_base} does not match "
            f"zone {zone}'s grid ({ny} x {nx}) — the campaign ingest must cover the zone extent exactly."
        )

    chunks = enumerate_chunks(ny, nx, shard_px)
    live = filter_chunks_by_roi_mask(chunks, land_mask_path)
    log.info(
        "Zone %s year %d: %d/%d tiles intersect the campaign land mask (run %s)",
        zone,
        year,
        len(live),
        len(chunks),
        run_id,
    )

    summary: dict[str, Any] = {
        "zone": zone,
        "year": year,
        "run_id": run_id,
        "total_tiles": len(chunks),
        "live_tiles": len(live),
    }

    if not live:
        # All-ocean cell: nothing to stage or write, but the campaign work
        # list must still see it land.
        snapshot = mark_zone_year_empty(repo, zone, year, run_id=run_id, gate=gate)
        tag = tag_zone_year(repo, zone, year, snapshot_id=snapshot)
        log.info("Zone %s year %d has no land under the mask — marked complete empty (%s)", zone, year, tag)
        return {**summary, "empty": True, "snapshot_id": snapshot, "tag": tag, "elapsed_sec": time.monotonic() - t0}

    results = run_inference(
        num_actors,
        config,
        live,
        mosaic_base,
        staging_base,
        run_id,
        t0,
        log,
        on_actor_retire=on_actor_retire,
        get_credentials=get_credentials,
    )
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        raise RuntimeError(f"{len(failed)} tiles failed during inference for zone {zone} year {year} (run {run_id})")

    writer = ZarrWriter(staging_base)
    writer.verify_staged_completeness(run_id, live, log=log)

    if not writer._list_staged_labels(run_id):
        # Every live tile resolved to a skip marker (zero valid pixels under
        # the validity filters) — a legitimate no-data cell, same as all-ocean.
        snapshot = mark_zone_year_empty(repo, zone, year, run_id=run_id, gate=gate)
        tag = tag_zone_year(repo, zone, year, snapshot_id=snapshot)
        log.info("Zone %s year %d: all %d live tiles skipped — marked complete empty (%s)", zone, year, len(live), tag)
        if cleanup_staging:
            try:
                writer.cleanup_staging(run_id, log)
            except Exception:
                log.warning("Staging cleanup failed for run %s", run_id, exc_info=True)
        return {
            **summary,
            "empty": True,
            "snapshot_id": snapshot,
            "tag": tag,
            "skipped": sum(r["status"] == "skipped" for r in results),
            "elapsed_sec": time.monotonic() - t0,
        }

    n_workers = n_assembly_workers or AssemblyConfig().compute_n_workers(len(live))
    snapshot = writer.assemble_global(
        store_path,
        zone,
        year=year,
        run_id=run_id,
        n_workers=n_workers,
        gate=gate,
        get_credentials=get_credentials,
        s3_region=s3_region,
        log=log,
    )
    tag = tag_zone_year(repo, zone, year, snapshot_id=snapshot)

    if cleanup_staging:
        try:
            writer.cleanup_staging(run_id, log)
        except Exception:
            log.warning("Staging cleanup failed for run %s", run_id, exc_info=True)

    return {
        **summary,
        "empty": False,
        "snapshot_id": snapshot,
        "tag": tag,
        "succeeded": sum(r["status"] == "success" for r in results),
        "skipped": sum(r["status"] == "skipped" for r in results),
        "resumed": sum(bool(r.get("resumed")) for r in results),
        "elapsed_sec": time.monotonic() - t0,
    }
