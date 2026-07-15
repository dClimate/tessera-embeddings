"""Zone-fill runner: one (zone, year) of the global store, end to end.

The orchestration-facing composition for the global campaign (ADR-008,
implementation plan W5): enumerate the zone's shard-aligned tile grid against
the **campaign land-mask coverage bitmap**, run Ray inference over the live
tiles, assemble the staged tiles into whole shards of the pre-seeded zone
group, and tag the landed commit. The Prefect/AWS wiring (work queues, the
fleet-wide commit gate as a Prefect global concurrency limit, credentials)
lives downstream — this module ships the plain callable it wraps.

Contracts (all caller-owned):

- **Ray**: the caller is already inside a Ray context (``ray.init`` or an
  attached cluster), exactly as for
  :func:`tessera_embeddings.inference.runner.run_inference`.
- **Land mask**: ``land_mask_path`` is the coverage Icechunk repo
  (:meth:`BucketPaths.land_mask_store`); this zone's group holds the
  registry-derived ``tile_live_2048`` bitmap built by
  :mod:`tessera_embeddings.ingest.land_mask` (ADR-010). A tile is live iff a
  land cell's footprint intersects it; one ~1 KB GET replaces the per-tile
  windowed reads of the old pixel mask. Within a live tile every
  observation-valid pixel is embedded, and water is SCL-valid — the v1.1 mask
  is all-1s with a ~1-cell sea buffer, so there is no per-pixel land signal to
  apply, and pixel-level masking inside tiles is moot (not merely deferred).
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
so the campaign work list converges. A cell that is already complete
short-circuits without re-running anything (creating the tag if a crash
landed the fill untagged). Zone-year tags are permanent: icechunk forbids
recreating even a deleted tag name, so a deliberate refill keeps the old tag
pointing at the old snapshot and must pin its new snapshot under a fresh,
manually chosen tag name — campaign history is never silently rewritten.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any, cast

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.assembly import AssemblyConfig
from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.inference.assembly import ZarrWriter, read_spatial_coords
from tessera_embeddings.inference.chunk_spec import enumerate_chunks
from tessera_embeddings.inference.runner import run_inference
from tessera_embeddings.storage.campaign import mark_zone_year_empty, tag_zone_year, zone_year_tag
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.shard_writer import CommitGate, read_years_complete, shard_pitch
from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group, time_index_of
from tessera_embeddings.storage.zone_grid import PIXEL_M, year_timestamp


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

    # Validate the year BEFORE any expensive work: assemble_global re-checks,
    # but hitting that check after hours of GPU inference would be brutal.
    if time_index_of(node, year_timestamp(year)) is None:
        raise ValueError(
            f"Year {year} is not on {zone}'s pre-allocated time axis — the axis is fixed at seeding (ADR-008 D1)."
        )
    # The slot being filled and the window inference computes over must agree:
    # a window that never touches `year` is an operator error (e.g. a cloned
    # invocation whose year was edited but not its time_window) that would
    # otherwise land mislabeled embeddings and tag them permanently complete.
    window_years = {y for y, _ in config.time_window.months}
    if year not in window_years:
        raise ValueError(
            f"config.time_window ({config.time_window.window_end_label}) covers {sorted(window_years)} "
            f"but year={year} — the inference window must overlap the calendar-year slot it fills."
        )

    # Idempotent retry: a cell that already landed is done — re-running it
    # would produce a new snapshot that tag_zone_year rightly refuses to move
    # the tag to. If the crash hit between the fill commit and the tag, tag the
    # current tip now (the fill commit is an ancestor, so retention holds; see
    # mark_zone_year_empty's attribution note) instead of re-running inference.
    # Zone-year tags are write-once forever (icechunk forbids tag-name reuse
    # even after deletion); deliberate refills need a fresh tag name.
    tag = zone_year_tag(zone, year)
    if year in read_years_complete(node):
        if tag in repo.list_tags():
            snapshot = repo.lookup_tag(tag)
            log.info("Zone %s year %d already complete and tagged (%s) — nothing to do", zone, year, tag)
        else:
            snapshot = repo.lookup_branch("main")
            tag_zone_year(repo, zone, year, snapshot_id=snapshot)
            log.info("Zone %s year %d landed but was untagged (crash before tag) — tagged %s", zone, year, tag)
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
    shard_px = shard_pitch(emb)
    if config.chunk_size != shard_px:
        raise ValueError(
            f"config.chunk_size={config.chunk_size} but {zone} shards are {shard_px} px — "
            "the global write path requires 1 inference tile == 1 shard (ADR-008 D3)."
        )

    # The ingest mosaics must be on the zone grid exactly (campaign contract) —
    # and not just dimensionally: every zone in a hemisphere shares the same
    # pixel extent, so a same-shaped mosaic for the WRONG zone (or a shifted /
    # reversed grid) would pass a shape check and be written positionally,
    # silently misgeoreferencing the whole fill. Compare CRS and coordinate
    # endpoints against the seeded group, the grid authority.
    spatial = read_spatial_coords(mosaic_base)
    total_y, total_x = len(spatial.northing), len(spatial.easting)
    if (total_y, total_x) != (ny, nx):
        raise ValueError(
            f"Mosaic grid ({total_y} x {total_x}) at {mosaic_base} does not match "
            f"zone {zone}'s grid ({ny} x {nx}) — the campaign ingest must cover the zone extent exactly."
        )
    zone_crs = node.attrs.get("crs")
    if spatial.crs != zone_crs:
        raise ValueError(
            f"Mosaic CRS {spatial.crs!r} at {mosaic_base} does not match zone {zone}'s CRS "
            f"{zone_crs!r} — a wrong-zone mosaic would be written positionally and misgeoreferenced."
        )
    z_north = cast(zarr.Array, node["northing"])
    z_east = cast(zarr.Array, node["easting"])
    # Absolute half-pixel tolerance, NOT np.isclose's default relative one: at a
    # ~9.3e6 m northing the default rtol=1e-5 would admit ~93 m (≈9 px) of drift.
    # A real shift is ≥1 px (10 m); half a pixel (5 m) sits safely above float32
    # coordinate roundtrip (~1 m at this magnitude) yet below one pixel.
    atol = PIXEL_M / 2
    endpoints_match = (
        np.isclose(spatial.northing[0], z_north[0], rtol=0.0, atol=atol)
        and np.isclose(spatial.northing[-1], z_north[-1], rtol=0.0, atol=atol)
        and np.isclose(spatial.easting[0], z_east[0], rtol=0.0, atol=atol)
        and np.isclose(spatial.easting[-1], z_east[-1], rtol=0.0, atol=atol)
    )
    if not endpoints_match:
        raise ValueError(
            f"Mosaic coordinates at {mosaic_base} (northing {spatial.northing[0]}..{spatial.northing[-1]}, "
            f"easting {spatial.easting[0]}..{spatial.easting[-1]}) do not lie on zone {zone}'s grid — "
            "shifted or reversed axes would silently misgeoreference the fill."
        )

    # The land mask is this zone's coverage group in the mask repo (ADR-010):
    # registry-derived tile-liveness bitmaps, not a pixel mask. Open it with the
    # same helper as the global store and validate its identity by attrs — all
    # 60 same-hemisphere zones share one grid shape, so a wrong-zone mask would
    # otherwise be read positionally and silently misclassify tiles (permanently
    # tagging the cell empty). Guarding zone + CRS + grid_shape closes that hole.
    cov = open_store_as_zarr_group(land_mask_path, group=zone, get_credentials=get_credentials, region=s3_region)
    cov_zone = cov.attrs.get("zone")
    cov_crs = cov.attrs.get("crs")
    cov_shape = list(cast("list[int]", cov.attrs.get("grid_shape", [])))
    if cov_zone != zone or cov_crs != zone_crs or cov_shape != [ny, nx]:
        raise ValueError(
            f"Coverage group for {zone} at {land_mask_path} has zone={cov_zone!r} crs={cov_crs!r} "
            f"grid_shape={cov_shape} — expected zone={zone!r} crs={zone_crs!r} grid_shape={[ny, nx]}. "
            "A wrong-zone coverage mask would be read positionally and misclassify tiles."
        )
    tile_live = np.asarray(cast("zarr.Array", cov["tile_live_2048"]), dtype=bool)
    n_tile_rows, n_tile_cols = ny // shard_px, nx // shard_px
    if tile_live.shape != (n_tile_rows, n_tile_cols):
        raise ValueError(
            f"Coverage bitmap for {zone} has tile_live shape {tile_live.shape} but the zone's tile grid is "
            f"({n_tile_rows}, {n_tile_cols}) — the coverage build is inconsistent with the seeded grid."
        )

    # 1 inference tile == 1 shard == 1 coverage tile (ADR-008 D3), so a chunk's
    # (row, col) ARE its coverage-bitmap indices: liveness is a direct lookup,
    # not the per-tile windowed read the single-ROI pixel mask needs.
    chunks = enumerate_chunks(ny, nx, shard_px)
    live = [c for c in chunks if bool(tile_live[c.row, c.col])]
    log.info(
        "Zone %s year %d: %d/%d tiles are live in the campaign coverage mask (run %s)",
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
    writer = ZarrWriter(staging_base)

    def _cleanup() -> None:
        if cleanup_staging:
            try:
                writer.cleanup_staging(run_id, log)
            except Exception:
                log.warning("Staging cleanup failed for run %s", run_id, exc_info=True)

    def _finish_empty(**extra: float) -> dict[str, Any]:
        # A no-data cell (all-ocean, or every live tile skipped) still lands:
        # years_complete + provenance in one commit, then the zone-year tag.
        snapshot = mark_zone_year_empty(repo, zone, year, run_id=run_id, gate=gate)
        tag = tag_zone_year(repo, zone, year, snapshot_id=snapshot)
        return {**summary, "empty": True, "snapshot_id": snapshot, "tag": tag, **extra}

    if not live:
        result = _finish_empty(elapsed_sec=time.monotonic() - t0)
        log.info("Zone %s year %d has no land under the mask — marked complete empty (%s)", zone, year, result["tag"])
        return result

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

    staged_labels = writer.verify_staged_completeness(run_id, live, log=log)
    if not staged_labels:
        # Every live tile resolved to a skip marker (zero valid pixels under
        # the validity filters) — a legitimate no-data cell, same as all-ocean.
        # Mark + tag FIRST, clean up after (matching the data path): a crash
        # after cleanup but before the tag would otherwise force full
        # re-inference just to regenerate zero-byte skip markers.
        result = _finish_empty(
            skipped=sum(r["status"] == "skipped" for r in results), elapsed_sec=time.monotonic() - t0
        )
        _cleanup()
        log.info(
            "Zone %s year %d: all %d live tiles skipped — marked complete empty (%s)",
            zone,
            year,
            len(live),
            result["tag"],
        )
        return result

    n_workers = n_assembly_workers or AssemblyConfig().compute_n_workers(len(live))
    snapshot = writer.assemble_global(
        store_path,
        zone,
        year=year,
        run_id=run_id,
        n_workers=n_workers,
        gate=gate,
        staged_labels=staged_labels,
        get_credentials=get_credentials,
        s3_region=s3_region,
        log=log,
    )
    tag = tag_zone_year(repo, zone, year, snapshot_id=snapshot)
    _cleanup()

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
