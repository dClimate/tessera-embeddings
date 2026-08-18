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
from tessera_embeddings.config.fault_injection import ArmedFault
from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.store_layout import GLOBAL
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.inference.assembly import (
    SpatialCoords,
    ZarrWriter,
    read_spatial_coords,
    read_store_spatial_coords,
    summarise_radar_coverage,
)
from tessera_embeddings.inference.chunk_spec import ChunkSpec, enumerate_chunks
from tessera_embeddings.inference.data_loading import _active_orbits
from tessera_embeddings.inference.runner import run_inference
from tessera_embeddings.storage.campaign import mark_zone_year_empty, tag_zone_year, zone_year_tag
from tessera_embeddings.storage.global_store import check_destination_types, open_global_repo
from tessera_embeddings.storage.shard_writer import CommitGate, read_years_complete, shard_pitch
from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group, time_index_of
from tessera_embeddings.storage.zone_grid import (
    PIXEL_M,
    canonicalize_zone,
    tile_row_latitudes,
    year_timestamp,
)
from tessera_embeddings.storage.zone_grid import zone as zone_spec


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


#: Observations per pixel-year by 20-degree latitude band — the campaign census, from
#: `context_docs/design/campaign-cost-model.md`. Radar is a CMR granule census of
#: OPERA RTC-S1 normalised by cos(lat); optical is a Sentinel-2 STAC census of distinct
#: acquisition dates weighted by mean clear fraction. The value is `S2 + S1`, which is
#: the token count per pixel and therefore taken to be proportional to inference cost.
#:
#: **That proportionality treats a radar token and an optical token as equally expensive, and
#: that is an assumption rather than a measurement.** The forward pass encodes optical and radar
#: as separate sequences, so a chunk carrying radar does work that scales with more than its
#: token total; whether the per-token cost is the same across the two is unmeasured. If it is
#: not, this weight under-values a radar-heavy zone relative to a radar-free one, and clusters
#: balanced on it finish unevenly in that direction. `CHUNK_SUMMARY` now reports the radar
#: sequence lengths per chunk, which is what would settle it — see the inference profile record.
#:
#: The bands are wide and the absolutes carry sampling error (five points per band), but
#: cluster balancing only needs the RATIOS between bands, which are the robust part —
#: both halves were counted on the same grid. The final band runs to -80 rather than -60
#: because there is negligible land below -60 and no reason for a zone to fall outside
#: the table.
TOKENS_PER_PX_BY_BAND: tuple[tuple[float, float], ...] = (
    # (band's lower latitude bound, tokens per pixel-year)
    (60.0, 208.0),
    (40.0, 176.0),
    (20.0, 168.0),
    (0.0, 120.0),
    (-20.0, 104.0),
    (-40.0, 128.0),
    (-80.0, 104.0),
)


def tokens_per_px(latitudes: np.ndarray) -> np.ndarray:
    """Tokens per pixel-year for each latitude, bucketed into :data:`TOKENS_PER_PX_BY_BAND`."""
    # Ascending by lower bound, so a higher band overwrites the ones it contains and the
    # highest match wins. Iterating the table as written would let every band above -80
    # be overwritten by the last row, which is why the order is explicit rather than
    # incidental.
    out = np.full(latitudes.shape, TOKENS_PER_PX_BY_BAND[-1][1], dtype="float64")
    for lower, tokens in sorted(TOKENS_PER_PX_BY_BAND):
        out = np.where(latitudes >= lower, tokens, out)
    return out


def zone_work_weight(
    land_mask_path: str,
    zone: str,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> float:
    """A zone's inference WORK, in tile-token units, for balancing clusters.

    The same one-GET read as :func:`zone_live_tile_count`, but weighted: each live
    tile row is multiplied by the tokens-per-pixel of its latitude band before
    summing. Returns tiles x tokens-per-px, so it is proportional to a zone-year's
    GPU-hours and directly comparable between zones.

    **Why not just count tiles.** Inference consumes one sequence per pixel, so its
    cost scales with `pixels x observations`, and observation count varies about
    twofold with latitude — a boreal tile is worth roughly twice an equatorial one.
    Balancing on tile counts therefore balances AREA and leaves the real work uneven,
    which shows up as clusters finishing at materially different times. Two zones
    with identical tile counts can differ by ~2x in work.

    **This is not a substitute for the tile count.** The chained fill still orders its
    zones by :func:`zone_live_tile_count` and still clamps actor requests to it,
    because those are properties of AREA: a zone needs one actor per tile regardless
    of how many observations each pixel has, and the ordering exists so an autoscaled
    fleet only ever shrinks. Use tiles for capacity, this for scheduling.
    """
    cov = open_store_as_zarr_group(land_mask_path, group=zone, get_credentials=get_credentials, region=s3_region)
    live = np.asarray(cast("zarr.Array", cov["tile_live_2048"]), dtype=bool)
    latitudes = tile_row_latitudes(zone_spec(canonicalize_zone(zone)), live.shape[0])
    return float((live.sum(axis=1) * tokens_per_px(latitudes)).sum())


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


def _check_assembly_workers(n: int | None) -> None:
    """Reject an assembly worker override that cannot build a pool.

    Checked at entry, not at use: the value is only consumed after the cell's whole
    GPU inference has run, and `n or default` treats 0 as "unset" while passing a
    negative through to `ProcessPoolExecutor(max_workers=<0)`. Either way the cell
    dies at assembly having already burned the expensive half of the fill.
    """
    if n is not None and n < 1:
        raise ValueError(f"n_assembly_workers must be >= 1 when set, got {n}")


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
    fault: ArmedFault | None = None,
    input_coverage: dict | None = None,
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
        input_coverage: How much of the requested window the input mosaics actually held,
            measured by the fill's preflight and recorded on the year. The mosaics are
            deleted once a cell lands, so this is the only lasting record of it.
        fault: Supervised-drill hook, forwarded to the assembly phase. Inert unless
            the run was armed for a fault this path hosts and for this cell
            (:mod:`tessera_embeddings.config.fault_injection`).

    Returns:
        Summary dict: zone, year, run_id, snapshot_id, tag, tile counts,
        inference outcome counts, ``empty`` flag, and elapsed seconds.
    """
    _check_assembly_workers(n_assembly_workers)
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
        fault=fault,
        input_coverage=input_coverage,
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

    # The store's minimum-depth rule is part of its write-once root identity, so the store —
    # not this call's config — is the authority on what rule its zones were filled under. The
    # Prefect adapter substitutes the store's value into the config before calling; this ASSERTS
    # that rather than trusting it, so the domain API is safe to call directly and the adapter's
    # substitution is verified rather than assumed. Free: `root` is already open above.
    #
    # A check and not a substitution, deliberately. Silently replacing a caller's value would
    # let a direct caller believe it had configured a line it had not, which is the same class
    # of surprise as the gap it closes.
    raw_rule = root.attrs.get("optical_min_obs")
    store_rule = int(cast("int", raw_rule)) if raw_rule is not None else None
    if config.optical_min_obs != store_rule:
        raise ValueError(
            f"This fill would apply optical_min_obs={config.optical_min_obs!r} to {zone}-{year}, but "
            f"{store_path} advertises {store_rule!r}. The rule is part of the store's write-once root "
            "identity: filling one zone under a different line than its neighbours is undetectable "
            "afterwards, because a refused pixel is indistinguishable from one with no optical input. "
            "Pass the store's value (None when it declares no rule), or seed a new store for a new rule."
        )

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

    # CAN THIS DESTINATION HOLD WHAT WE ARE ABOUT TO WRITE? One metadata read, here, rather than
    # the same answer from assembly after the fleet has run: on 2026-08-18 two fills each spent
    # their whole inference before dying on a seeded dtype the staging writer cannot produce.
    # Beside the shard-pitch check above because it is the same kind of gate — the destination's
    # shape, tested before anything is billed.
    check_destination_types(node, GLOBAL, where=f"{zone} year {year}")

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
        # nothing to signal it.
        #
        # Compare the COMPLETE coordinate vectors against the seeded axes, not a
        # spacing test. A per-step tolerance has to admit float round-trip noise, and
        # anything it admits per step an adversarial axis can accumulate: half a pixel
        # of slack per step permits a 5-to-15 m stride, and a pattern that wanders and
        # returns still matches the length and both endpoints. Comparing every element
        # against the authority has no such gap and needs no tolerance argument — it is
        # two 1-D reads per store, once per fill.
        #
        # The campaign's own ingest cannot produce a bad axis (odc builds every load
        # against the zone geobox), but `ingest=False` accepts a mosaic the operator
        # staged, and that path is supported.
        for axis, values, seeded in (
            ("northing", coords.northing, z_north),
            ("easting", coords.easting, z_east),
        ):
            got = np.asarray(values, dtype="float64")
            want = np.asarray(seeded[:], dtype="float64")
            # atol is a float round-trip allowance, NOT a geometric one: coordinates
            # round-trip through float32 at ~1 m near a 9.3e6 m northing, and any real
            # displacement is a whole pixel (10 m).
            bad = ~np.isclose(got, want, rtol=0.0, atol=1.0)
            if bad.any():
                i = int(np.argmax(bad))
                raise ValueError(
                    f"{label} {axis} at {store_path} does not match zone {zone}'s seeded axis: "
                    f"index {i} is {got[i]} where the grid says {want[i]} "
                    f"({int(bad.sum())} of {got.size} coordinates differ). Length, CRS and "
                    "endpoints all match, so this is a reordered or non-affine interior — "
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
    fault: ArmedFault | None = None,
    input_coverage: dict | None = None,
) -> dict[str, Any]:
    """Assembly phase of a (zone, year) fill: verify staged → assemble → tag.

    Consumes an :class:`infer_zone_year` handoff: verifies staged
    completeness, assembles the staged tiles into the zone group (or marks
    the cell complete-empty when every live tile skipped), tags the landed
    commit, and cleans up staging. Opens its own repo/writer connections so
    the sequential multi-zone runner can run it on a trailing thread while
    the caller's thread starts the next cell's inference. A terminal handoff
    (``done`` set) passes straight through.

    ``fault`` is the supervised-drill hook, forwarded to the assembly call on both
    the data and the all-skipped path so the drill does not depend on which one a
    cell takes. Inert unless the run was armed for a fault this path hosts and for
    this cell (:mod:`tessera_embeddings.config.fault_injection`).
    """
    _check_assembly_workers(n_assembly_workers)
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
        #
        # It still WRITES, over the whole live footprint, for the reason the mixed
        # path below writes over its skipped tiles: a year lands in two commits, so an
        # attempt that crashed between them leaves shards on a year nothing has marked,
        # and the campaign re-dispatches it. If that attempt's mosaic made tiles
        # productive where this one skips them all, marking the year empty without
        # writing would leave its embeddings readable under a completion mark and a
        # zone-year tag that both say the cell holds nothing.
        #
        # Mark + tag FIRST, clean up after (matching the data path): a crash
        # after cleanup but before the tag would otherwise force full
        # re-inference just to regenerate zero-byte skip markers.
        if live:
            # The fill write and the completion mark land together, in the one call, so
            # the year can never be marked empty over shards this run did not clear.
            snapshot = writer.assemble_global(
                store_path,
                zone,
                year=year,
                run_id=run_id,
                n_workers=n_assembly_workers or AssemblyConfig().compute_n_workers(len(live)),
                gate=gate,
                staged_labels=(),
                skipped_labels=sorted(c.label for c in live),
                s3_concurrency=s3_concurrency,
                empty=True,
                get_credentials=get_credentials,
                s3_region=s3_region,
                log=log,
                fault=fault,
                input_coverage=input_coverage,
            )
        else:
            # No live tiles at all: nothing was ever written here, so there is nothing
            # to clear and the attrs alone are the whole job.
            snapshot = mark_zone_year_empty(
                open_global_repo(store_path, get_credentials=get_credentials, region=s3_region),
                zone,
                year,
                run_id=run_id,
                gate=gate,
            )
        repo = open_global_repo(store_path, get_credentials=get_credentials, region=s3_region)
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

    # Live tiles that staged nothing because every pixel failed the validity filters.
    # Passed so assembly writes fill over them instead of leaving them alone: this year
    # may already hold shards from an attempt that crashed between the shard commit and
    # the completion attrs, and if THAT attempt's mosaic made a tile productive where
    # this one skips it, its data would survive under this run's completion mark.
    # Derived from the STAGING PREFIX (live minus what verify_staged_completeness
    # resolved as staged), never from `results`: a resumed run's skip markers were
    # written by earlier legs, which the finishing leg reports as resumed successes —
    # a results-based set would clear nothing and record zero skips over real gaps.
    # Assembly also records this set as the year's `optical_skips` provenance.
    skipped_labels = sorted({c.label for c in live} - staged_labels)
    n_workers = n_assembly_workers or AssemblyConfig().compute_n_workers(len(live))
    # Reduced from what the actors already reported, so it costs no reads. Recorded on the
    # YEAR's provenance entry: radar coverage is a property of what was acquired, so one
    # year of a zone can be radar-free where another is not.
    radar_coverage = summarise_radar_coverage(results)
    if radar_coverage:
        log.info(
            "Radar coverage %s-%d: %.1f%% of embedded pixels have NO radar, %.1f%% have "
            "fewer than %d observations (%d tile(s) wholly radar-free)",
            zone,
            year,
            radar_coverage["s1_free_pct"],
            radar_coverage["s1_thin_pct"],
            radar_coverage["s1_thin_below_obs"],
            radar_coverage["tiles_fully_s1_free"],
        )
    snapshot = writer.assemble_global(
        store_path,
        zone,
        year=year,
        run_id=run_id,
        n_workers=n_workers,
        gate=gate,
        staged_labels=staged_labels,
        skipped_labels=skipped_labels,
        s3_concurrency=s3_concurrency,
        radar_coverage=radar_coverage,
        get_credentials=get_credentials,
        s3_region=s3_region,
        log=log,
        fault=fault,
        input_coverage=input_coverage,
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
