# Cropping ingest to live tiles

**Status:** shipped, **unconditional, and no longer behind a flag** (amended 2026-07-29).
It was `IngestSettings.crop_to_live_windows`, defaulting off; the flag was removed
outright once no scenario wanted cropping disabled, and the one validation that
depended on it (`batch_dates > 1 requires crop_to_live_windows`) went with it, its
precondition now holding by construction. Validated on a live `03S` cell, which also
surfaced a second extent-scaled cost in the coverage denominator — see "What the first
cropped run found" below. Anywhere below that names the flag, or shows it being set, is
describing how a measurement was taken rather than a switch that still exists.

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

A one-off script coarsened each zone's frozen `tile_live_2048` bitmap (ADR-010)
onto the 4096-px ingest grid and costed three cropping strategies against the
full-extent baseline, reading a few KB per zone from the coverage repo — no
cluster, no mosaic, no writes. It was **deleted once the strategy was chosen**
(`scripts/measure_live_chunk_cropping.py`, removed 2026-08-11): it exists to pick
between the three rows below, and the mask it reads is frozen for the campaign's
duration. Recover it from history if a future delivery changes the coverage
enough to reopen the choice.

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

**Confirmed by local experiment (2026-07-24; the full test log is §A below):**
several `to_icechunk`
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
control flow, metadata, and commits are direct: the time-axis extension is a raw
resize plus a coord append on the session, attrs are plain zarr writes, one
commit per date is one `session.commit`.

**For the window PIXELS, volume decides — the same split `region_merge` itself
implements** (one shard → no pool, "fork once and write inline"; many → the
pool). The domain function runs on a single Dask worker, so computing windows to
numpy and assigning them directly funnels the date's live volume through that one
node. The arithmetic: sparse zones — the ones that motivated this work — are
trivial (03S: 4 chunks, well under 1 GB/date across all bands); a dense zone is
not (35N: ~2,400 live chunks ≈ 80 GB per band-date; a single wide row window is
~570 MB per band). At volume, the mechanism is the experiment's shared-session
`to_icechunk(mode="r+", region=...)` per window: placement stays on the workers
that computed the pixels, Dask's memory manager stays in charge, and the commit
shape is unchanged. That is also the faithful adaptation of region-merge's
fork/merge to graph-resident bytes — icechunk forks the session and merges
changesets *inside* `to_icechunk`, with Dask workers playing the role
region-merge gives its process pool. The pool itself does not transfer: it exists
because region-merge's bytes are re-readable from a store by child processes
(only paths and slices pickle across); ours would have to carry the pixels.

**This must not become `write_regions` again** — a single Dask graph spanning every
region, which was built and removed for being the bottleneck (see the warning
below). Per-window graphs stay per-window; only the session and commit are shared.

## What this shares with the region-merge tier

`storage/region_merge.py` and the ROI fan-out foundations solve the tier directly above this one:
merging many grid-aligned feature stores into one master. Code was taken from both and
generalised, so the generalised version here is the shared one and those branches rebase onto it
rather than landing a second copy.

The one design consequence worth carrying: **snapping windows to the chunk grid makes them
chunk-disjoint**, which removes the shared-boundary reconciliation the merge tier has to handle.
That is why this path can write N windows into one session safely and the merge tier cannot.

## Bookkeeping relocation — resolved

`write_dataset` and its two callers owned bookkeeping the seed-then-region-write path had to
re-home. The full trace with file:line cites is in the PR #97 thread record; the outcome is that
**most of it dissolved rather than moved**, because keeping dates discovery-as-you-go meant the
time axis still only ever contains dates that passed the gate and were written — which is the
property `get_existing_dates`, `check_time_window_coverage` and the empty-timestep prunes all
depend on. A pre-seeded daily axis would have broken all three, and that is the reason the design
is the way it is.

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

## What the first cropped run found: crop the coverage denominator too

The first `03S` run with the flag on confirmed the mosaic side exactly as designed
— **2 windows, 4 chunks per band-date** against 3,706, the window derivation
agreeing with an independent full-resolution scan of the same mask. But the run
still built an **8,794-task graph** and still spilled, and the dashboard showed
why: `from-zarr 3706 / 3706`, `sum 3706 / 3706`, `sum-partial 1378`. Those totals
are the whole graph, and none of it is the mosaic.

It was these two lines, which sit upstream of every window and which the original
change did not touch:

```python
roi_mask = client.persist(roi_mask)              # 3,706 chunks, ~60 GiB, pinned
roi_pixel_count = int(roi_mask.sum().compute())  # a full-extent reduce
```

The ROI-pixel total is the **denominator** of the S2 coverage check. Cropping had
already cropped the numerator (the SCL validity reduce) on the stated grounds that
the mask is False outside every window — so the identical argument applies to the
denominator, and the fix is to apply it: total the mask over the live windows, and
drop the persist, since `persist` materialises the entire grid while every
downstream consumer (that same reduce, and `apply_roi_mask`) slices to windows and
lets dask cull the reads. S1 carries the same persist and gets the same treatment;
it computes no coverage total of its own.

For `03S` this is ~8,800 tasks and ~60 GiB down to a handful of tasks and ~64 MiB.
Note the shape of the mistake, because it generalises: the residual cost was
**constant per zone** (~3,700–4,000 chunks regardless of how much land a zone
holds), so it would have been a fixed overhead on all 120 zones × 3 sensor
children — the same extent-scaled pathology as the original finding, surviving in
a place nobody was looking because the mosaic numbers looked right.

The numerator/denominator coupling is now pinned by
`tests/unit/test_s2_coverage_windows.py`, including a deliberately non-vacuous
case: windows derived from one mask, applied to a mask with land outside them,
asserting the totals DO diverge. Cropping only one side of that ratio would still
have produced a plausible-looking percentage, and the write-once ingest marker
would have made a wrongly-filtered year permanent.

## Also from the first cropped run: the profiler was blind to spill

The scheduler heartbeat recorded no worker-side memory at all, so the residual
above had to be diagnosed from a screenshot of the Dask dashboard. The heartbeat
now carries fleet totals (`wmem`/`wmanaged`/`wspill`/`wmax`) summed from the
per-worker state the scheduler already tracks, and `te-watch-scheduler` alerts on
`worker-spill`. This does not demote the scheduler: it remains the named
saturation risk at scale, and a clean fleet is the precondition for a
high-worker-count rung to measure a scheduler envelope rather than a doomed run.

## Consequences for the test programme

The ingest scaling ladder cannot measure a scheduler envelope until it stops
benchmarking ocean: as it stands the memory wall arrives long before any scheduler
limit, so it would tune the wrong bottleneck. Fargate worker-count and quota sizing
derived from full-extent mosaics likewise overstate what the campaign needs, by
roughly the 4.3× above.

---

## Grouping row bands: why window COUNT was the real limit (2026-07-25)

Cropping to row bands fixed the *area* problem and exposed a different one. Row
bands minimise computed area, and area turned out not to be what a windowed ingest
is billed for.

## The measurement

Dense zone `35N`, January 2024, a 120-worker Fargate fleet (4 vCPU each, so 480
task slots), `crop_to_live_windows=true`, 197 live windows:

| quantity | value |
|---|---|
| date `2024-01-01` | 1471.3 s |
| date `2024-01-02` | 1329.2 s |
| per window | ~7.1 s |
| sparse-zone comparison, far smaller fleet | ~6 s per window |
| mean tasks processing | 30.6 of 480 slots = **6%** |
| scheduler samples with an EMPTY graph | **16 of 44 (36%)**, twice for 2 min straight |
| worker spill | 0 GiB |
| scheduler cpu / rss / lag | 20-50% / 0.78 GiB / ~0 s |

Per-window cost is flat from a sparse zone on a small fleet to a dense zone on 120
workers. That is a fixed serial cost, not distributed work — and it is the same
phenomenon the earlier fleet-scaling rung reported as "35% efficiency", seen from
the other end. **More workers was never going to help.**

The cause is structural: `write_day_windows` loops `to_icechunk` once per window,
each call blocking. A `35N` row band is at most 17 chunks wide and only ~3 of those
hold a given day's swath, so each graph is a few dozen tasks against 480 slots, 197
times in series.

## The cost model

A matched A/B pair — identical zone, month, fleet and mask, only the window count
differing — gives two equations in two unknowns for
`cost = n_windows × F + chunk_area × V`:

| run | windows | chunk area | seconds |
|---|---|---|---|
| row bands | 197 | 2428 | 1471.3 |
| grouped (first cut) | 15 | 2563 | 194.4 |

→ **F = 7.04 s per window, V = 0.0346 s per chunk**, so one saved write is worth
**≈200 chunks** of extra computed area. The second baseline date (1329.2 s against
1471 s predicted) puts per-date variance around 10%, which is the precision this
model deserves: its value is the RATIO, not the constants.

Measured A/B outcome at 15 windows: **194.4 / 207.8 / 230.4 / 222.8 / 239.5 s** for
the first five dates against **1471.3 / 1329.2 s** — a **~6.6× speedup**. Fleet
occupancy went from 30.6 to ~407 tasks in flight (6% → 85% of slots), mean graph
size from 482 to ~13,400 tasks, spill stayed at zero, hottest worker 3.96 → 5.6 GiB
of 16.

## Why the first cut was left as a greedy heuristic, and why it was replaced

The first implementation merged greedily while the added area stayed within 25% of
the row-band baseline. With the cost model in hand that bound is simply the wrong
shape: a fixed waste *fraction* cannot express "extra area is nearly free", so it
stopped merging on sparse zones — exactly where a tiny absolute area makes merging
almost costless.

Evaluated on all 112 real zone masks, predicted per-date cost summed:

| strategy | total | vs row bands | max graph |
|---|---|---|---|
| row bands (one per live chunk-row) | 73,636 s | 100% | 1,650 tasks |
| greedy, 25% waste bound, 512-chunk cap | 13,143 s | 17.8% | 5,632 tasks |
| **cost-model DP** | **6,733 s** | **9.1%** | 21,692 tasks |
| single bounding box per zone | 8,770 s | 11.9% | 42,636 tasks |

The greedy pass left **+95%** on the table. Note the DP also beats the pure bounding
box, because it adapts per zone instead of committing to one shape; the worst greedy
zones were the sparse ones (`07S` +402%, `08S` +311%, `03N` +292%).

So `merge_bands` now minimises `n_windows × WINDOW_COST_IN_CHUNKS + total_area`
exactly, by dynamic programming over groupings of consecutive bands. O(n²) with n =
a zone's live chunk-rows (≤ ~230), and the unit tests hold it against brute-force
enumeration on small inputs.

## Choosing the graph-size cap

`MAX_CHUNKS_PER_WINDOW` exists to bound one region write's graph, not to bound cost.
Sweeping it against the DP optimum:

| area cap | max graph (tasks) | total | vs uncapped |
|---|---|---|---|
| none | 28,050 | 6,728 s | — |
| 2,560 | 28,050 | 6,728 s | +0.0% |
| **2,048** | **21,692** | **6,733 s** | **+0.07%** |
| 1,536 | 16,874 | 6,794 s | +1.0% |
| 1,024 | 11,264 | 6,909 s | +2.7% |
| 512 | 5,632 | 7,463 s | +10.9% |

2,048 is the pick: it costs 0.07%, and it holds every graph at or below 21,692 tasks
— just under the 22,812 a live run has actually driven with zero spill and a 5.6 GiB
hottest worker. Note that even the 512-cap DP (7,463 s) beats the greedy pass
(13,143 s) by 1.76×, so the win comes from optimising the right objective rather
than from taking memory risk.

## Per-zone effect

| zone | live chunks | row bands | grouped | added area |
|---|---|---|---|---|
| `03S` | 4 | 2 | 1 | +0.0% |
| `15S` | 22 | 5 | 1 | +45.5% |
| `40S` | 26 | 12 | 2 | +50.0% |
| `35N` | 2415 | 197 | 3 | +20.4% |

Median 2 windows per zone, maximum 5. Sparse zones group *harder* in relative terms
— which is the correction, not a regression: their added area is a large fraction of
a tiny total, and a tiny total is where extra area cannot matter.

## What this leaves

The empty-graph samples did not disappear — 3 of 10 after grouping against 16 of 44
before, a similar fraction of a much shorter date. Those gaps are the serial phase
*between* dates: the STAC query and graph construction, which run single-threaded on
the flow runner while the fleet waits. That is the next thing to measure, and it is
a different fix from this one.

---

## Appendix A — Icechunk multi-write-per-commit experiment (2026-07-24)

**Absorbed 2026-08-17 from `ingest-live-tile-cropping-icechunk-experiment.md`.** It was
always an appendix to this document — the only thing referencing it was the paragraph above
— and the contract it establishes is one the ingest still depends on, so it belongs beside
the design it justifies rather than one directory listing away.

Question: can multiple `icechunk.xarray.to_icechunk` calls share ONE `writable_session` + ONE `commit` (N disjoint spatial windows of one date -> one snapshot), with dask-backed xarray data?

## Versions (repo venv `/Users/rbanick/dev/tessera-embeddings/.venv/bin/python`)

- icechunk 2.0.4
- zarr 3.2.1
- xarray 2026.4.0
- dask 2026.3.0
- numpy 2.4.4
- scheduler: default threaded local dask scheduler (no distributed cluster)

## Setup (all tests)

Local filesystem icechunk repo per test. Store: dims (time=3, northing=64, easting=64), chunks (1, 16, 16), data vars `emb` float32 (fill -999.0) + `count` uint16 (fill 0), datetime64[ns] time coord. Seeded all-fill via `to_icechunk(ds, session, mode="w")` + commit (ancestry = 2 snapshots: repo-init + seed). All window datasets are dask-backed (`ds.chunk({"time":1,"northing":16,"easting":16})`, verified `dask.array.core.Array`) with coords dropped, mirroring the repo's `_drop_region_coords`. Scripts: `common.py`, `test_a.py` ... `test_f_attrs.py` in this directory.

## TEST A — two r+ region writes, one session, one commit: **PASS**

```python
session = repo.writable_session("main")
to_icechunk(win1, session, mode="r+",
            region={"time": slice(1,2), "northing": slice(0,32), "easting": slice(0,64)},
            align_chunks=True, split_every=8)
to_icechunk(win2, session, mode="r+",
            region={"time": slice(1,2), "northing": slice(32,64), "easting": slice(0,64)},
            align_chunks=True, split_every=8)
session.commit("...")
```

- Snapshots: 2 before -> 3 after (exactly 1 new).
- Read-back: window 1 = 1.0/11, window 2 = 2.0/22, time slices 0 and 2 still all-fill. Both vars correct.

## TEST B — same spatial chunk row, adjacent chunk-aligned windows: **PASS**

Regions `northing 0:16` and `16:32` (same easting chunk columns 0:64), same call sequence as A. Snapshots 2 -> 3. No interference: window 1 = 3.0/33, window 2 = 4.0/44, `northing 32:64` of that date and the other two dates still fill.

## TEST C — 20 disjoint chunk-aligned 16x16 windows, one session, one commit: **PASS**

Loop of 20 `to_icechunk(..., mode="r+", region=..., align_chunks=True, split_every=8)` on one session, then one commit. Snapshots 2 -> 3. All 20 windows read back with their distinct values; untouched chunks still fill.

- Timing: 20 writes + 1 commit = **0.20 s total (~0.010 s/write)** on local FS with tiny chunks. No per-call slowdown observed.

## TEST D — append (mode="a") THEN region write into the new date, SAME session, one commit: **PASS (same-session works; fallback not needed)**

```python
session = repo.writable_session("main")
to_icechunk(new_date_allfill_with_coords, session, mode="a", append_dim="time", align_chunks=True)
# session sees its own uncommitted append:
#   zarr.open_group(session.store)["emb"].shape == (4, 64, 64)  <- already 4
to_icechunk(win, session, mode="r+",
            region={"time": slice(3,4), "northing": slice(0,32), "easting": slice(0,64)},
            align_chunks=True, split_every=8)
session.commit("...")
```

- Snapshots: 2 before -> 3 after (append + region = ONE snapshot).
- Known wrinkle answered: **the same session DOES see its own uncommitted append** — `zarr.open_group(session.store)` showed shape (4, 64, 64) and time indices [0 1 2 3] immediately after the append, before commit, so `region={"time": slice(3,4)}` validated and wrote fine.
- Read-back: 4 dates, appended time == 2024-01-04, window 7.0/77 correct, rest of new date fill, original 3 dates untouched.
- Two-session fallback (append+commit, then region+commit) was coded but never triggered. Pre-seeding dates first is NOT mandatory.

## TEST E — uncommitted r+ writes invisible to a separate readonly reader: **PASS**

After one r+ region write on an uncommitted session: a `readonly_session(branch="main")` from a **separate** `Repository.open` handle read the entire store as all-fill, and ancestry count was unchanged. After `session.commit(...)` the same read path saw the written window (guards against a false pass). Snapshots 2 -> 3.

## TEST F (bonus) — root attrs across multiple r+ calls: preserved

Root attrs stamped in a prior commit (`{"geoemb:probe": "keep-me", "spatial:thing": 42}`) survived two `mode="r+"` region writes + commit in one session, unchanged. So the attrs-clobbering that motivated the repo's `_commit_preserving_attrs` did **not** reproduce for pure `mode="r+"` region writes on icechunk 2.0.4 / xarray 2026.4.0 (it may still apply to `mode="a"`/`"w"` or older versions — this is a data point, not a recommendation to drop the guard).

## Verdict

**Multiple to_icechunk region writes per session+commit: YES.** On icechunk 2.0.4 / zarr 3.2.1 / xarray 2026.4.0 with dask-backed data on the threaded scheduler, N sequential `to_icechunk(..., mode="r+", region=..., align_chunks=True, split_every=8)` calls against one `writable_session("main")` followed by one `session.commit()` produce exactly one snapshot, with all windows correct and no cross-window interference even when windows share a chunk row (tested up to 20 windows; ~10 ms/write locally, no degradation). The session never refused a second write, i.e. to_icechunk's internal fork/merge for dask tolerates a session that already carries uncommitted changes. **Append+region same session: YES** — `mode="a", append_dim="time"` followed by `mode="r+"` region writes into the just-appended date commits as one snapshot, because the session exposes its own uncommitted append (shape already reflects the new date before commit), so region indices for the new date resolve without an intermediate commit; the two-session pre-seed-dates-first fallback is available but not required. Uncommitted session writes are invisible to readonly readers on separate repo handles (commit atomicity holds). Caveats: all windows here were exactly chunk-aligned, so `align_chunks=True` never had to read-modify-write a boundary chunk — two writes straddling the SAME chunk within one session were not tested and should still be treated as forbidden (disjoint chunk-aligned windows remain the contract); root attrs were not clobbered by multiple r+ calls in this stack; local-FS icechunk warns it is unsafe for concurrent commits (irrelevant here — one commit).
