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
2. On the first passing date, seed the mosaic full-extent and all-fill with that
   date's slot (`create_empty_store` is schema-only — zero chunks, no graph — so
   creation cost is independent of spatial extent).
3. Per passing date, ONE session: extend the time axis by that date, then
   `to_icechunk(mode="r+", region=...)` each live window, then commit — one
   commit per date. The experiment (below) confirmed a session sees its own
   uncommitted append, so region indices for the new date resolve in-session.
   Explicitly NOT per-window `zarr_store.write_region` (commits per call), and
   NOT a full-extent all-fill append through dask (that rebuilds a ~44K-task
   graph of zero-work per date — extend the axis as metadata, not as data).

Dates stay discovery-as-you-go: the time axis only ever contains dates that
passed the coverage filter and were written. That single property is what keeps
today's semantics intact for every reader that treats the time axis as the
record of what was ingested (`get_existing_dates` dedupe,
`check_time_window_coverage`, the empty-timestep prunes) — see the relocation
inventory below for why a pre-seeded daily axis would have broken all three.

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

**Confirmed by local experiment (2026-07-24, report + scripts:
`ingest-live-tile-cropping-icechunk-experiment.md`):** several `to_icechunk`
calls CAN share one `writable_session` and one commit, on this exact stack
(icechunk 2.0.4 / zarr 3.2.1 / xarray 2026.4.0, dask-backed data). Twenty
disjoint chunk-aligned region writes on one session produced exactly one
snapshot with no cross-window interference, ~10 ms/write locally with no
degradation. Two further results shape the design:

- **A session sees its own uncommitted append.** `mode="a"` along time followed
  by `mode="r+"` region writes into the just-appended date commits as ONE
  snapshot — the region indices resolve against the uncommitted shape. So the
  per-date flow can stay discovery-as-you-go: append the date's slot, write its
  live windows, commit once. No pre-enumeration of dates is required.
- **Uncommitted writes are invisible to readers** on separate repo handles, so
  the per-date commit stays atomic for concurrent readers.

Caveats carried into the implementation: every experimental window was exactly
chunk-aligned. Two writes straddling the SAME chunk in one session were not
tested and stay forbidden — disjoint chunk-aligned windows are the contract,
which the 4096-snapped window derivation gives us by construction. And the
experiment ran on icechunk 2.0.4 (the venv's pin) while the campaign standard is
≥2.1.1 (ADR-008's pre-benchmark bump, the #2158 fix) — unlikely to change these
results, but re-run the report's call sequences after the bump before relying on
them at scale.

**Framing principle (standing, 2026-07-24): avoid Dask where possible.** Direct
zarr writes under an icechunk commit are cleaner than anything routed through a
task graph — `write_regions`' removal is the precedent. Dask is genuinely needed
to *compute* the mosaic (the STAC loads and resampling), so it stays for that;
it is not needed to *place* the results. So the write path prefers
`region_merge`'s write style without its process pool: compute a window (or a
chunk of one) to numpy, assign it positionally into the session's zarr group,
one commit per date. The time-axis extension is likewise a raw resize plus a
coord append on the session — metadata, not a graph.

The proven fallback, if per-session write throughput ever binds (one session's
chunk writes serialise on its store mutex; region-merge measured the plateau), is
the experiment's other result: sequential `to_icechunk(mode="r+", region=...)`
calls on the shared session, which fork internally and write from Dask workers.
Either way: no process pool, no session forking of our own.

**This must not become `write_regions` again** — a single Dask graph spanning every
region, which was built and removed for being the bottleneck (see the warning
below). Per-window graphs stay per-window; only the session and commit are shared.

## What we take from the region-writes / region-merge work, and why

`storage/region_merge.py` and its design note (PR #77), and the ROI fan-out
foundations (PR #79), solve the tier directly above this one: merging many
grid-aligned feature stores into one master. We take code from both, generalised —
neither PR is merging soon, so the generalised version here becomes the shared one
and those branches rebase onto it rather than landing a second copy.

Adopted as code:

**Grid-aligned window derivation** (#79). `feature_window` snaps a geometry to a
window of the master geobox with `GeoBox.enclosing` plus an `overlap_roi` clamp, so
the window is an exact pixel-subset *by construction* rather than by post-hoc
validation — region writes place data positionally and are silently wrong otherwise.

*Generalised:* geometry and bitmap are two ways of expressing the same live-cell
selection, so one helper takes either and returns grid-aligned windows. The
single-ROI path keeps its geometry entry point; the campaign gets a bitmap one.

**The chunk-scaled cluster sizing pattern** (upstream `roi_fanout`): workers per
cell derived from its live-chunk count, clamped to a floor and cap
(`_workers_for_chunks`, 0.5 workers/chunk in [10, 200]). This is the direct fix for
"the 50-worker ceiling was hit for a 4-tile zone" — after cropping, a cell's
live-window count is exactly the work measure the cluster should be sized from.

**NOT adopted after the experiment: `region_merge`'s fork-and-merge machinery.**
Its process pool exists because raw zarr chunk writes serialise on a session's
store mutex in-process — a constraint of the store-to-store copy, whose bytes are
re-readable by child processes (only paths and slices pickle across). Our bytes
come out of a Dask graph, and the experiment showed plain `to_icechunk` region
writes on a shared session already give one commit per date with worker-side
writes. Using the fork machinery would have meant materialising windows on the
flow runner just to ship them to a pool — the exact thing `to_icechunk` avoids.

**Forward compatibility with real merge workflows** therefore costs nothing: the
batching surface we add (one session, N region writes, one commit — with the attrs
preservation `_commit_preserving_attrs` provides) is useful to a merge tier but
does not replace it. `region_merge` stays the right tool for store-to-store merges
when they are wanted, unchanged, and nothing here forecloses porting it.

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

## Bookkeeping relocation inventory (traced 2026-07-24)

`write_dataset` and its two callers own bookkeeping that the seed-then-region-write
path must explicitly re-home. Full trace with file:line cites lives in the PR #97
thread record; the load-bearing items:

**Dissolved by keeping dates discovery-as-you-go** (the per-date append design):
- `check_time_window_coverage` proves coverage by counting timestamps
  (`data_loading.py:435`), so a pre-seeded daily axis would pass it vacuously and
  the write-once ingest marker would make a hollow mosaic permanent. A
  written-dates-only axis keeps the gate meaningful.
- `get_existing_dates` means "already ingested" to both sensors' STAC dedupe; a
  pre-seeded axis would filter every date and silently produce empty mosaics.
- The empty-timestep prunes (S2 SCL-validity, S1 `vv != 0`) — sound as long as
  the seed fill for every mosaic var is integer 0, which `_fill_for_dtype` gives.
  Pin with a test; a float-seeded var (NaN fill) would break both.

**Must be re-homed deliberately:**
- `manifest.validate_against` runs on every append today (`zarr_store.py:984`) and
  is the only gate against writing into a structurally different store. The
  batched path must validate at least once per ingest invocation; `write_region`
  does no manifest validation at all.
- `last_appended` must keep moving per write session (`update_attrs`) — it is the
  `_mosaic_identity` fallback for prebuilt mosaics, and frozen-at-seed it would
  alias two different builds.
- `baselines_applied` merges (dict union, new wins) and `doy` concatenates in
  append order (`zarr_store.py:985-996`); a naive attrs `update` clobbers instead
  of merging. `doy` has no src/ reader (inference recomputes from the time coord),
  so write-at-append-time correctness is what matters, not consumers.
- Chunk encoding + `TIME_ENCODING` are create-only today; they move wholesale to
  the seeding call, and `create_empty_store` already computes the identical clamp.
  Verify its default codecs match what `to_icechunk` picks today — `VarSpec`
  defaults to `"auto"`, and a codec mismatch between seeded and appended stores
  would be silent.
- S1 writes a whole `batch_days` batch per call (many dates), unlike S2's one;
  its dates are non-contiguous, so the S1 port loops per-date sessions rather
  than assuming the S2 shape transfers.
- Tenacity retry currently wraps only `write_dataset`; it must widen to the whole
  per-date unit (resolve + append + region writes + commit), which stays safe
  under icechunk atomicity since an uncommitted session is invisible to readers.

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
