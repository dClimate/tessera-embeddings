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
from dataclasses import dataclass
from typing import Any, cast

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.assembly import AssemblyConfig
from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.inference.assembly import (
    SpatialCoords,
    ZarrWriter,
    read_spatial_coords,
    read_store_spatial_coords,
)
from tessera_embeddings.inference.chunk_spec import ChunkSpec, enumerate_chunks
from tessera_embeddings.inference.data_loading import _active_orbits
from tessera_embeddings.inference.runner import run_inference
from tessera_embeddings.storage.campaign import mark_zone_year_empty, tag_zone_year, zone_year_tag
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.shard_writer import CommitGate, read_years_complete, shard_pitch
from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group, time_index_of
from tessera_embeddings.storage.zone_grid import PIXEL_M, year_timestamp


def assert_calendar_year_window(time_window: TimeWindow, year: int) -> None:
    """Reject any window that is not exactly January-December of ``year``.

    The single enforcement point for the calendar-year guarantee. Callable BEFORE
    a cluster is provisioned as well as inside planning: the request is decidable
    from config alone, and provisioning a GPU fleet (or starting ingests) for a
    window that planning will certainly reject is pure spend.

    Compares the WHOLE month set, not just "every month falls in ``year``": a
    same-year partial (e.g. Jan-Jun 2025) also has ``window_years == {year}`` but
    is only six months, and would otherwise tag the slot complete with a short
    window while the seeded ``time_bnds`` advertise the full Jan-Dec.
    """
    if set(time_window.months) != {(year, m) for m in range(1, 13)}:
        window_years = sorted({y for y, _ in time_window.months})
        raise ValueError(
            f"config.time_window ({time_window.window_end_label}, months in year(s) {window_years}) "
            f"is not the exact January-December {year} window, but the global store guarantees "
            f"calendar-year slots (time_convention='calendar_year'): year={year} requires all 12 months "
            f"January-December {year}. Rolling/offset OR same-year partial windows are not writable to "
            f"the global store — use the single-ROI `12mo_window_end` path for non-calendar windows "
            f"(or the ADR-011 windowed-variant design at zone scale)."
        )


def zone_live_tile_count(
    land_mask_path: str,
    zone: str,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> int:
    """Number of 2048-px tiles the coverage bitmap marks live for ``zone``.

    The same one-GET read as :func:`zone_has_live_tiles`, kept separate because
    the multi-zone sequential fill uses the COUNT twice at preflight: ordering
    its zones largest-first (so a shared cluster's fleet only ever shrinks) and
    clamping per-zone actor requests. This only reads the liveness bitmap;
    :func:`fill_zone_year` re-reads and attr-validates the coverage group as
    the authority, so a wrong-zone mask still fails loudly there rather than
    being trusted here.
    """
    cov = open_store_as_zarr_group(land_mask_path, group=zone, get_credentials=get_credentials, region=s3_region)
    return int(np.asarray(cast("zarr.Array", cov["tile_live_2048"]), dtype=bool).sum())


def zone_has_live_tiles(
    land_mask_path: str,
    zone: str,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> bool:
    """Whether the coverage bitmap marks ANY 2048-px tile live for ``zone``.

    A cheap one-GET preflight: an all-ocean cell (no live tiles) can be filled
    empty with no GPU work, so a caller can skip provisioning a Ray cluster for
    it. See :func:`zone_live_tile_count` for the underlying read and its
    trust boundary.
    """
    return zone_live_tile_count(land_mask_path, zone, get_credentials=get_credentials, s3_region=s3_region) > 0


def zone_year_complete(
    store_path: str,
    zone: str,
    year: int,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> bool:
    """Whether ``(zone, year)`` is already recorded in the group's ``years_complete``.

    A cheap metadata read so a caller can skip provisioning a GPU cluster for a
    crash-recovery retry of a landed-but-untagged cell (:func:`fill_zone_year`
    re-checks and, for such a cell, only re-creates the tag — no inference).
    Returns False for an unseeded zone.
    """
    repo = open_global_repo(store_path, get_credentials=get_credentials, region=s3_region)
    root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    if zone not in root:
        return False
    return year in read_years_complete(cast(zarr.Group, root[zone]))


def zone_year_on_axis(
    store_path: str,
    zone: str,
    year: int,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> bool:
    """Whether ``year`` falls on ``zone``'s pre-allocated time axis.

    A cheap metadata read (no mask, no mosaic) so a caller can decline to
    provision a GPU cluster for an off-axis / unseeded year: :func:`fill_zone_year`
    rejects such a year up front, but only after the flow would otherwise have
    stood up Ray. Returns False for an unseeded zone or an off-axis year.
    """
    repo = open_global_repo(store_path, get_credentials=get_credentials, region=s3_region)
    root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    if zone not in root:
        return False
    return time_index_of(cast(zarr.Group, root[zone]), year_timestamp(year)) is not None


@dataclass
class ZoneFillHandoff:
    """State handed from :func:`infer_zone_year` to :func:`assemble_zone_year`.

    Plain data only (labels, counts, per-tile result dicts) — no repo handles
    or zarr groups — so the assembly phase can run on a different thread than
    the inference phase (the sequential multi-zone runner trails a cell's
    assembly behind the NEXT cell's inference) and re-open its own store
    connections there.
    """

    zone: str
    year: int
    run_id: str
    t0: float
    # zone/year/run_id/tile-count fields shared by every result dict.
    summary: dict[str, Any]
    # Live tiles that went to inference — verify_staged_completeness and
    # assembly worker sizing both need the list, not just its length.
    live: list[ChunkSpec]
    results: list[dict[str, Any]]
    # Terminal result produced by the inference phase (already-complete cell,
    # or all-ocean empty fill). When set, assembly is a pass-through no-op.
    done: dict[str, Any] | None = None


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
    s3_concurrency: int | None = None,
    cleanup_staging: bool = True,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
    on_actor_retire: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fill one (zone, year): mask → inference → shard assembly → tag.

    The composition of :func:`infer_zone_year` and :func:`assemble_zone_year`,
    run back to back — the right shape for a standalone single-cell fill. The
    multi-zone sequential runner calls the two phases separately so a cell's
    assembly can trail behind the next cell's inference on a shared cluster.

    Args:
        store_path: URI of the global Icechunk repo (``BucketPaths.global_store()``).
        zone: Zone group name — UTM common name, e.g. ``"01N"``/``"60S"``.
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
        s3_concurrency: This fill's slice of the fleet S3-PUT budget, forwarded to
            ``assemble_global``; ``None`` uses the full aggregate target (a lone fill).
        cleanup_staging: Delete staged tiles after a successful fill.
        get_credentials: Optional icechunk credential callback (actors + store).
        s3_region: Optional S3 region override for the global store.
        on_actor_retire: Optional callback when a misbehaving actor is retired
            (the AWS provider injects an EC2 terminator).

    Returns:
        Summary dict: zone, year, run_id, snapshot_id, tag, tile counts,
        inference outcome counts, ``empty`` flag, and elapsed seconds.
    """
    handoff = infer_zone_year(
        store_path=store_path,
        zone=zone,
        year=year,
        land_mask_path=land_mask_path,
        mosaic_base=mosaic_base,
        staging_base=staging_base,
        config=config,
        num_actors=num_actors,
        log=log,
        run_id=run_id,
        gate=gate,
        get_credentials=get_credentials,
        s3_region=s3_region,
        on_actor_retire=on_actor_retire,
    )
    return assemble_zone_year(
        handoff,
        store_path=store_path,
        staging_base=staging_base,
        log=log,
        gate=gate,
        n_assembly_workers=n_assembly_workers,
        s3_concurrency=s3_concurrency,
        cleanup_staging=cleanup_staging,
        get_credentials=get_credentials,
        s3_region=s3_region,
    )


@dataclass
class ZonePlan:
    """Everything a (zone, year) fill knows before any GPU work starts.

    Produced by :func:`plan_zone_inference`; consumed either by
    :func:`infer_zone_year` (which runs a per-cell ``run_inference`` over
    ``live``) or by the chained multi-zone runner (which enqueues ``live``
    into a shared scheduler session and reassembles a
    :class:`ZoneFillHandoff` from the streamed results).
    """

    zone: str
    year: int
    run_id: str
    t0: float
    summary: dict[str, Any]
    live: list[ChunkSpec]
    # Terminal result (already-complete cell, or all-ocean empty fill —
    # committed and tagged during planning). When set, there is nothing to
    # infer or assemble.
    done: dict[str, Any] | None = None


def plan_zone_inference(
    *,
    store_path: str,
    zone: str,
    year: int,
    land_mask_path: str,
    mosaic_base: str,
    config: InferenceConfig,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    run_id: str | None = None,
    gate: CommitGate | None = None,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> ZonePlan:
    """Validate a (zone, year) cell and enumerate its live tiles — no GPU work.

    Everything :func:`infer_zone_year` does before ``run_inference``:
    seeded/axis/window validation, the coverage-mask read (with identity
    checks), the live-tile enumeration, and the mosaic grid asserts. Terminal
    cells — already complete (retagged here) or all-ocean (marked complete
    empty, committed and tagged here) — come back with ``ZonePlan.done`` set.
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
    # STRICT calendar-year gate: the window must be EXACTLY Jan-Dec of `year`. The
    # global store's time points are window STARTS (each slot's coordinate is Jan 1
    # of its year, fixed at seeding), so a window that merely overlaps `year` — e.g.
    # a rolling Feb-Jan — would sit at a coordinate outside its own real interval
    # (violating CF §7.1 bounds containment) and be permanently mislabeled under the
    # write-once zone-year tag; and one slot per year cannot hold two window phases
    # anyway. Label accuracy is the invariant: `time_convention="calendar_year"` is a
    # GUARANTEE, and the seeded `time_bnds` ([Jan 1 of y, Jan 1 of y+1) per slot) states each
    # slot's true interval. Non-calendar 12-month windows belong in a store whose
    # time points ARE the windows: today the single-ROI `12mo_window_end` path
    # (assemble(): time = window-end label, extendable axis); zone-scale, the
    # windowed-variant design in ADR-011 (slots declared at other start months).
    # Compare the WHOLE month set, not just "every month falls in `year`": a
    # same-year partial (e.g. Jan-Jun 2025) also has window_years == {year} but is
    # only six months, and would otherwise pass here and tag the slot complete
    # with a short window while the seeded time_bnds advertise the full Jan-Dec.
    assert_calendar_year_window(config.time_window, year)

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
        result = {
            "zone": zone,
            "year": year,
            "run_id": run_id,
            "already_complete": True,
            "snapshot_id": snapshot,
            "tag": tag,
            "elapsed_sec": time.monotonic() - t0,
        }
        return ZonePlan(zone=zone, year=year, run_id=run_id, t0=t0, summary={}, live=[], done=result)

    emb = cast(zarr.Array, node["embeddings"])
    _, ny, nx, _ = emb.shape
    shard_px = shard_pitch(emb)
    if config.chunk_size != shard_px:
        raise ValueError(
            f"config.chunk_size={config.chunk_size} but {zone} shards are {shard_px} px — "
            "the global write path requires 1 inference tile == 1 shard (ADR-008 D3)."
        )

    zone_crs = node.attrs.get("crs")

    # Read the coverage mask BEFORE the mosaic: an all-ocean cell (whose ingest
    # mosaic may never have been created) must reach the empty-cell path below
    # rather than failing on a missing reflectance store. The land mask is this
    # zone's coverage group in the mask repo (ADR-010): registry-derived
    # tile-liveness bitmaps, not a pixel mask. Validate its identity by attrs —
    # all 60 same-hemisphere zones share one grid shape, so a wrong-zone mask
    # would otherwise be read positionally and silently misclassify tiles
    # (permanently tagging the cell empty). Guarding zone + CRS + grid_shape
    # closes that hole.
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

    if not live:
        # A no-data cell (all-ocean) still lands: years_complete + provenance
        # in one commit, then the zone-year tag. Terminal here — no staging
        # exists, so there is nothing for the assembly phase to do.
        snapshot = mark_zone_year_empty(repo, zone, year, run_id=run_id, gate=gate)
        tag = tag_zone_year(repo, zone, year, snapshot_id=snapshot)
        result = {**summary, "empty": True, "snapshot_id": snapshot, "tag": tag, "elapsed_sec": time.monotonic() - t0}
        log.info("Zone %s year %d has no land under the mask — marked complete empty (%s)", zone, year, result["tag"])
        return ZonePlan(zone=zone, year=year, run_id=run_id, t0=t0, summary={}, live=[], done=result)

    # Live tiles will be inferred, so EVERY active mosaic store must sit on the
    # zone grid EXACTLY (campaign contract) — not just dimensionally. Every zone
    # in a hemisphere shares the same pixel extent, so a same-shaped store for the
    # WRONG zone (or a shifted/reversed grid) would pass a shape check and be
    # written positionally, silently misgeoreferencing the fill. The SAR stores
    # are read by positional slice (_load_sar_orbit) with no coords of their own,
    # so a stale child SAR store or a hand-provided `mosaic_base` on a different
    # grid than reflectance would otherwise slip through unchecked. Validate the
    # reflectance store AND each active SAR orbit against the seeded group, the
    # grid authority. (Read only now — an all-ocean cell already returned above
    # without touching the mosaic, which may not even exist.)
    z_north = cast(zarr.Array, node["northing"])
    z_east = cast(zarr.Array, node["easting"])
    # Absolute half-pixel tolerance, NOT np.isclose's default relative one: at a
    # ~9.3e6 m northing the default rtol=1e-5 would admit ~93 m (~9 px) of drift.
    # A real shift is >=1 px (10 m); half a pixel (5 m) sits safely above float32
    # coordinate roundtrip (~1 m at this magnitude) yet below one pixel.
    atol = PIXEL_M / 2

    def _assert_on_zone_grid(label: str, store_path: str, coords: SpatialCoords) -> None:
        total_y, total_x = len(coords.northing), len(coords.easting)
        if (total_y, total_x) != (ny, nx):
            raise ValueError(
                f"{label} grid ({total_y} x {total_x}) at {store_path} does not match "
                f"zone {zone}'s grid ({ny} x {nx}) — the campaign ingest must cover the zone extent exactly."
            )
        if coords.crs != zone_crs:
            raise ValueError(
                f"{label} CRS {coords.crs!r} at {store_path} does not match zone {zone}'s CRS "
                f"{zone_crs!r} — a wrong-zone mosaic would be written positionally and misgeoreferenced."
            )
        endpoints_match = (
            np.isclose(coords.northing[0], z_north[0], rtol=0.0, atol=atol)
            and np.isclose(coords.northing[-1], z_north[-1], rtol=0.0, atol=atol)
            and np.isclose(coords.easting[0], z_east[0], rtol=0.0, atol=atol)
            and np.isclose(coords.easting[-1], z_east[-1], rtol=0.0, atol=atol)
        )
        if not endpoints_match:
            raise ValueError(
                f"{label} coordinates at {store_path} (northing {coords.northing[0]}..{coords.northing[-1]}, "
                f"easting {coords.easting[0]}..{coords.easting[-1]}) do not lie on zone {zone}'s grid — "
                "shifted or reversed axes would silently misgeoreference the fill."
            )
        # Length, CRS and endpoints still do not pin an axis: a REORDERED or non-affine
        # interior satisfies all three. Inference writes positionally onto the seeded
        # grid, so such a mosaic would publish real pixels at the wrong coordinates with
        # nothing to signal it. Checking uniform PIXEL_M spacing closes that — together
        # with the matching length and endpoints above it makes the axis exactly the
        # seeded one, and it costs one diff over a coordinate vector already in hand.
        #
        # The campaign's own ingest cannot produce a bad axis (odc builds every load
        # against the zone geobox), but `ingest=False` accepts a mosaic the operator
        # staged, and that path is supported.
        for axis, values in (("northing", coords.northing), ("easting", coords.easting)):
            if len(values) < 2:
                continue
            diffs = np.diff(np.asarray(values, dtype="float64"))
            if not np.all(np.isclose(np.abs(diffs), PIXEL_M, rtol=0.0, atol=atol)):
                worst = int(np.argmax(np.abs(np.abs(diffs) - PIXEL_M)))
                raise ValueError(
                    f"{label} {axis} at {store_path} is not a uniform {PIXEL_M} m axis: step "
                    f"{diffs[worst]} between index {worst} and {worst + 1}. Endpoints and length "
                    f"match zone {zone}'s grid, so this is a reordered or non-affine interior — "
                    "inference writes positionally and would misgeoreference those pixels silently."
                )

    _assert_on_zone_grid(
        "Mosaic reflectance",
        f"{mosaic_base}/reflectance.zarr",
        read_spatial_coords(mosaic_base, get_credentials=get_credentials, s3_region=s3_region),
    )
    for orbit in _active_orbits(config.s1_orbit):
        sar_path = f"{mosaic_base}/sar_{orbit}.zarr"
        _assert_on_zone_grid(
            f"Mosaic SAR {orbit}",
            sar_path,
            read_store_spatial_coords(sar_path, get_credentials=get_credentials, s3_region=s3_region),
        )

    return ZonePlan(zone=zone, year=year, run_id=run_id, t0=t0, summary=summary, live=live)


def infer_zone_year(
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
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
    on_actor_retire: Callable[[str], None] | None = None,
    retire_idle_actors: bool = True,
) -> ZoneFillHandoff:
    """Inference phase of a (zone, year) fill: plan → run_inference.

    :func:`plan_zone_inference` composed with a per-cell ``run_inference``
    over the plan's live tiles. Terminal cells (already complete / all-ocean)
    come back with ``ZoneFillHandoff.done`` set.

    ``retire_idle_actors=False`` keeps idle actors alive through this cell's
    tail — a caller running cells as separate sessions passes it for every
    cell but its last so the shared cluster's instances survive to serve the
    next zone (the chained runner instead streams every zone through ONE
    session; see :mod:`.sequential_fill`).

    Raises:
        RuntimeError: If any live tile fails inference.
    """
    plan = plan_zone_inference(
        store_path=store_path,
        zone=zone,
        year=year,
        land_mask_path=land_mask_path,
        mosaic_base=mosaic_base,
        config=config,
        log=log,
        run_id=run_id,
        gate=gate,
        get_credentials=get_credentials,
        s3_region=s3_region,
    )
    return (
        complete_zone_inference(plan, results=None)
        if plan.done is not None
        else _infer_planned(
            plan,
            mosaic_base=mosaic_base,
            staging_base=staging_base,
            config=config,
            num_actors=num_actors,
            log=log,
            on_actor_retire=on_actor_retire,
            get_credentials=get_credentials,
            s3_region=s3_region,
            retire_idle_actors=retire_idle_actors,
        )
    )


def _infer_planned(
    plan: ZonePlan,
    *,
    mosaic_base: str,
    staging_base: str,
    config: InferenceConfig,
    num_actors: int,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    on_actor_retire: Callable[[str], None] | None,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None,
    s3_region: str | None,
    retire_idle_actors: bool,
) -> ZoneFillHandoff:
    """Run a per-cell inference session over a plan's live tiles."""
    results = run_inference(
        num_actors,
        config,
        plan.live,
        mosaic_base,
        staging_base,
        plan.run_id,
        plan.t0,
        log,
        on_actor_retire=on_actor_retire,
        get_credentials=get_credentials,
        s3_region=s3_region,
        retire_idle_actors=retire_idle_actors,
    )
    return complete_zone_inference(plan, results=results)


def complete_zone_inference(plan: ZonePlan, *, results: list[dict] | None) -> ZoneFillHandoff:
    """Fold per-tile results back into the handoff the assembly phase consumes.

    Shared by the per-cell path and the chained runner (which collects a
    zone's results from the streamed session instead of a dedicated
    ``run_inference`` call). Raises if any tile failed — the zone must not
    assemble partial output; its cell stays pending in the campaign ledger.
    """
    if plan.done is not None:
        return ZoneFillHandoff(
            zone=plan.zone,
            year=plan.year,
            run_id=plan.run_id,
            t0=plan.t0,
            summary={},
            live=[],
            results=[],
            done=plan.done,
        )
    assert results is not None
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        raise RuntimeError(
            f"{len(failed)} tiles failed during inference for zone {plan.zone} year {plan.year} (run {plan.run_id})"
        )
    return ZoneFillHandoff(
        zone=plan.zone,
        year=plan.year,
        run_id=plan.run_id,
        t0=plan.t0,
        summary=plan.summary,
        live=plan.live,
        results=results,
    )


def assemble_zone_year(
    handoff: ZoneFillHandoff,
    *,
    store_path: str,
    staging_base: str,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    gate: CommitGate | None = None,
    n_assembly_workers: int | None = None,
    s3_concurrency: int | None = None,
    cleanup_staging: bool = True,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> dict[str, Any]:
    """Assembly phase of a (zone, year) fill: verify staged → assemble → tag.

    Consumes an :class:`infer_zone_year` handoff: verifies staged
    completeness, assembles the staged tiles into the zone group (or marks
    the cell complete-empty when every live tile skipped), tags the landed
    commit, and cleans up staging. Opens its own repo/writer connections so
    the sequential multi-zone runner can run it on a trailing thread while
    the caller's thread starts the next cell's inference. A terminal handoff
    (``done`` set) passes straight through.
    """
    if handoff.done is not None:
        return handoff.done
    zone, year, run_id, t0 = handoff.zone, handoff.year, handoff.run_id, handoff.t0
    summary, live, results = handoff.summary, handoff.live, handoff.results
    writer = ZarrWriter(staging_base)

    def _cleanup() -> None:
        if cleanup_staging:
            try:
                writer.cleanup_staging(run_id, log)
            except Exception:
                log.warning("Staging cleanup failed for run %s", run_id, exc_info=True)

    staged_labels = writer.verify_staged_completeness(run_id, live, log=log)
    if not staged_labels:
        # Every live tile resolved to a skip marker (zero valid pixels under
        # the validity filters) — a legitimate no-data cell, same as all-ocean.
        # Mark + tag FIRST, clean up after (matching the data path): a crash
        # after cleanup but before the tag would otherwise force full
        # re-inference just to regenerate zero-byte skip markers.
        repo = open_global_repo(store_path, get_credentials=get_credentials, region=s3_region)
        snapshot = mark_zone_year_empty(repo, zone, year, run_id=run_id, gate=gate)
        tag = tag_zone_year(repo, zone, year, snapshot_id=snapshot)
        result = {
            **summary,
            "empty": True,
            "snapshot_id": snapshot,
            "tag": tag,
            "skipped": sum(r["status"] == "skipped" for r in results),
            "elapsed_sec": time.monotonic() - t0,
        }
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
        s3_concurrency=s3_concurrency,
        get_credentials=get_credentials,
        s3_region=s3_region,
        log=log,
    )
    repo = open_global_repo(store_path, get_credentials=get_credentials, region=s3_region)
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
