# Cropping ingest to live tiles

**Status:** measured, design chosen, implementation pending.

## Problem

Campaign ingest mosaics a zone's **entire grid** regardless of how much of it is
land, so its cost scales with EXTENT, not content. The STAC *search* is already
narrowed — `export_zone_roi` writes a WGS84 bbox tight to the live tiles — but the
raster computation is not: the ROI mask is deliberately written on the fixed full
zone grid, so `roi.geobox` is full-extent and both sensor paths hand that straight
to `odc.stac.load`.

Observed on the first real campaign cell (zone `03S`, 2024), chosen as the cheapest
possible end-to-end test because it is the smallest land zone in the scheme — 4 live
2048-px tiles:

| | |
| --- | --- |
| Dask tasks in one graph | 119,002 |
| Worker memory | 517 GiB resident + 468 GiB spilled |
| Worker churn | 62 registered / 44 removed |
| Scheduler | healthy throughout (0.94 GiB RSS, 0.0 s loop lag, no backlog) |

Spill drove memory pressure, workers were killed, completed tasks were lost and
recomputed, and the run would likely never have finished. The scheduler was never
implicated; the graph was.

`03S` is 890,880 × 67,584 px, which at `INGEST_CHUNK_SIZE = 4096` is 218 × 17 =
**3,706 chunks per band, per date** — the exact per-band counts the dashboard
showed. One uint16 chunk at 4096² is 34 MB, so a single band-date materializes
~124 GB. Empty ocean chunks still allocate and still run a task.

## Measurement

`scripts/measure_live_chunk_cropping.py` coarsens each zone's frozen
`tile_live_2048` bitmap (ADR-010) onto the 4096-px ingest grid and costs three
cropping strategies against today's full-extent baseline. It reads a few KB per
zone from the coverage repo — no cluster, no mosaic, no writes.

All 112 land zones, chunks computed per band-date:

| strategy | chunks | vs today |
| --- | --- | --- |
| today (full extent) | 425,272 | — |
| `bbox` — one window enclosing all live chunks | 230,686 | 1.8× |
| `rows` — one window per chunk-row, spanning that row's live columns | 99,847 | **4.3×** |
| `exact` — the live chunks themselves (the floor) | 97,597 | 4.4× |

Two results decide the design:

**A single bounding box is not enough.** It gives only 1.8× campaign-wide. It is
spectacular where land is clustered (`03S`: 3,706 → 4) and worthless where a zone
holds scattered islands across a long north-south extent (`31S`: 3,706 → 1,152 by
bbox, → 3 by rows) or where land is dense (`35N`: 3,876 → 3,723, a 4% saving).

**Row-bands capture essentially the whole win.** They land within 1.0% of the exact
floor at the median and 2.3% campaign-wide. A full rectangle decomposition is
therefore not worth building: it would recover ~2% for materially more complexity.

Window counts stay modest — a median of 74 per zone and a maximum of 220 — which
matters because each window is one load and one write.

## Design

**The mosaic's declared grid stays full-extent.** The zone fill validates the mosaic
against the exact `ZoneSpec` grid and that check is load-bearing. Zarr and Icechunk
are sparse, so the grid can stay full while only live windows are computed and
written; unwritten chunks read back as fill. Zarr 3 elides all-fill chunks on write
by default (`write_empty_chunks` is False), so this costs no storage either.

The shape:

1. Derive live windows from the ROI mask, coarsened to the ingest chunk grid.
2. Seed the mosaic full-extent and all-fill — creation cost is independent of
   spatial extent, since data vars are schema-only with zero chunks written.
3. Per date, extend the time axis, then write each live window with
   `zarr_store.write_region`. One commit per date, not per window.

Snapping windows to the 4096 chunk grid is deliberate: it makes windows
chunk-disjoint, which removes the shared-boundary-chunk reconciliation that the
region-merge tier has to handle (see below).

### Interoperability with single-ROI runs

Windows are derived from **the ROI mask itself**, not from the zone coverage bitmap.
`export_zone_roi` already upsamples `tile_live_2048` onto the zone grid to produce
that mask, so reading the mask serves both callers: a campaign zone and a
single-ROI run whose mask came from `rasterize_roi_zarr`. A sparse single ROI —
scattered fields, a coastline, any footprint much smaller than its bounding box —
gets the same reduction from the same code path, and the ingest flows need no
knowledge of whether they are in a campaign.

This also settles an open question from the original write-up: the ingest does not
need `export_zone_roi` to carry the bitmap forward, because it already opens the
mask.

## Writing N windows without a commit storm

`zarr_store.write_region` writes **one region per commit**. At a median of 74
windows and roughly 50 dates that is ~3,700 commits per zone-year, against a store
whose commit pressure is already a governed constraint (ADR-008 D5/D6 caps
concurrent committers). So the per-window loop cannot use it as-is.

The same code shows the way out: `_commit_preserving_attrs` calls `to_icechunk`
and `session.commit` as separate steps, so several writes can share one session and
one commit. That is what `region_merge` already does properly — fork the session,
write chunk bytes per fork, merge the changesets, commit once — and it is the
mechanism this work adopts (below).

**This must not become `write_regions` again** — a single Dask graph spanning every
region, which was built and removed for being the bottleneck (see the warning
below). The adopted mechanism is the opposite: raw chunk writes, no graph.

## What we take from the region-writes / region-merge work, and why

`storage/region_merge.py` and its design note (PR #77), and the ROI fan-out
foundations (PR #79), solve the tier directly above this one: merging many
grid-aligned feature stores into one master. We take code from both, generalised —
neither PR is merging soon, so the generalised version here becomes the shared one
and those branches rebase onto it rather than landing a second copy.

Two things are adopted outright:

**The fork-and-merge write mechanism** (#77). Within one icechunk session every
chunk write serialises on that session's store mutex, so a thread pool plateaus
regardless of width; the only axis that scales is processes. `region_merge` forks
the session N ways, each fork writing its shard's bytes directly to storage with its
own lock and connection pool, and merges the returned changesets into one commit.
Workers use the **spawn** context so a child never inherits the parent's icechunk
runtime or credential state. This is measured, non-obvious knowledge, and it gives
us many windows in one commit with no task graph.

*Generalised:* today the byte source can only be another store. It becomes a
parameter, so the source is either a source store (the merge case) or in-memory
arrays (the ingest case). The overlay policy is likewise a parameter — blind assign
for windows that are disjoint and single-owner, fill-masked overlay
(`where(src is fill, master, src)`) where windows may share a master chunk.

**Grid-aligned window derivation** (#79). `feature_window` snaps a geometry to a
window of the master geobox with `GeoBox.enclosing` plus an `overlap_roi` clamp, so
the window is an exact pixel-subset *by construction* rather than by post-hoc
validation — region writes place data positionally and are silently wrong otherwise.

*Generalised:* geometry and bitmap are two ways of expressing the same live-cell
selection, so one helper takes either and returns grid-aligned windows. The
single-ROI path keeps its geometry entry point; the campaign gets a bitmap one.

**Forward compatibility with real merge workflows.** The store-to-store merge is a
use case we may want for its own reasons, so generalising must not foreclose it: a
source store stays a first-class byte source, fill-masked overlay stays expressible
even though the ingest path does not need it, and the date-union and
temp-store-cleanup helpers stay usable. The test for "generalised correctly" is that
`merge_stores` could be rebuilt on top of this without special-casing.

This should be cheap rather than a contortion, because the two cases differ on only
two axes: **where a block's bytes come from** (a source store's chunk, or a slice of
an in-memory array) and **how they combine with what is already there** (assign, or
overlay only non-fill pixels). Everything else — session forking, spawn context,
tiling to the master chunk grid, changeset merge, one commit, the temporal
distinct-time-chunk check, the spatial pixel-subset check — is identical and already
written. And `region_merge` already contains *both* combine behaviours internally as
an optimisation: an all-real interior block skips the master read and assigns
directly, while partial-edge blocks read-modify-write. So exposing the choice is
surfacing an existing branch, not adding one. If that turns out not to hold once the
code is in front of us, drop the merge generality rather than contort for it — the
ingest path is the thing on the critical path.

**What we deliberately do not take: the merge *workflow*.** Ingesting each window to
a temp store and merging it in would write every live byte twice. Empty chunks are
already elided, so today's write volume is live-only — doubling it doubles S3 PUTs
against a backend whose push-back is already managed by an aggregate concurrency
budget. Take the mechanism, not the pipeline.

Four further findings transfer, and one is a warning we would otherwise have walked
into.

**The warning: do not rebuild a Dask-graph region write.** `write_regions`, a
`store_dask`-based batch path, was built and then removed. Its
`O(runs × bands × spatial_chunks)` task graph, constructed single-threaded on the
flow runner before any compute began, was itself the bottleneck that made
continental merges take days. That is the same signature as the 03S incident — a
119,002-task graph whose construction and scheduling, not its data, was the
problem. It confirms the fix must shrink the **graph**, not merely make dead blocks
cheap to compute.

**The sparse-master pattern is the same problem one level up.** Region-merge's
motivating use case is "a big rectangular master where only scattered regions are
populated, the rest staying all-fill" — a zone mosaic with scattered live tiles is
exactly that. Its workflow (seed all-fill over the full grid, then write only the
populated parts) is the one adopted above, and `empty_store.create_empty_store` is
the seeding primitive it established: correct grid, chunking, dtype and attrs with
zero chunks written, so seeding a continental extent is cheap.

**Positional writes demand grid-aligned windows.** PR #79's `feature_window` snaps a
geometry to a window of the master geobox via `GeoBox.enclosing` plus an
`overlap_roi` clamp, so the window is an exact pixel-subset *by construction* rather
than by post-hoc validation — because region writes place data positionally and are
silently wrong otherwise. Our windows come from a bitmap rather than a geometry, so
the function is not directly reusable, but the invariant and the idiom are the same
and the two should not drift. If #79 lands first, unify on one helper.

**The temporal invariant is already satisfied.** Region-merge requires distinct
dates in distinct time chunks, or concurrent writers race within a chunk with no
conflict resolution. `INGEST_CHUNKS` already sets `time: 1`.

**Fill-masked overlay is the hazard we design around rather than solve.** A
rectangular window over an irregular footprint carries fill where the footprint
does not cover, so region-merge must overlay only real pixels lest one feature
overwrite a neighbour's data with its own fill. Chunk-aligned, per-row windows over
a single ROI make our windows disjoint and single-owner, so the hazard does not
arise — but it is the reason to keep windows chunk-snapped rather than tight to the
live pixels.

**Pacing, if we fan out.** PR #79's `DispatchThrottle` bounds concurrency *and*
paces launch instants, which matters against APIs that throttle on burst rate. Only
relevant if windows are ever dispatched as separate runs rather than looped
in-process; noted so it is not reinvented.

## Also from the 03S run

**A hard-cancelled ingest leaks its Dask cluster** — 23 workers and a scheduler left
running in ECS. The original write-up attributed this to `skip_cleanup: True`, but
that flag only disables dask-cloudprovider's startup sweep for debris from *prior*
runs, and the comment above it records that turning it off breaks cluster
construction outright on AWS SSO, because the sweep iterates IAM roles. The actual
cause is that `ecs_cluster` tears down in a `finally`, which a hard cancel skips,
and the ingest flows register no cancellation hook. The GPU flows already solved
this: `_ray_lifecycle.ray_cleanup_on_cancellation`, registered as both
`on_cancellation` and `on_crashed`, re-derives the cluster name from the flow-run id
and terminates by tag. Ingest needs that same treatment for ECS.

**The 50-worker ceiling was hit for a 4-tile zone**, which is a symptom of the
uncropped graph rather than a limit to raise. Once cropping lands, worker counts
should be sized from a cell's live-chunk count.

**External services were not implicated** — one store-write retry, no catalog
throttling.

## Consequences for the test programme

The ingest scaling ladder cannot measure a scheduler envelope until it stops
benchmarking ocean: as it stands the memory wall arrives long before any scheduler
limit, so it would tune the wrong bottleneck. Fargate worker-count and quota sizing
derived from full-extent mosaics likewise overstate what the campaign needs, by
roughly the 4.3× above.
