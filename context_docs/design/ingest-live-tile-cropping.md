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

`zarr_store.write_region` writes **one region per commit**. At a median of 74 windows and ~50 dates
that is ~3,700 commits per zone-year, against a store whose commit pressure is a governed constraint
(ADR-008 D5/D6 caps concurrent committers). So the per-window loop cannot use it as-is.

**What shipped:** one session per date — extend the time axis by that date, region-write each live
window into it, commit once. Appendix A is the verification that this is allowed, and its
chunk-alignment caveat is the contract.

**Windows are snapped to the chunk grid deliberately**, which makes them chunk-disjoint and removes
the shared-boundary reconciliation the region-merge tier has to handle. It is also what makes the
single-session write safe: two writes into the same chunk in one session are a lost update.

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

Cropping the computation left the coverage GATE measuring against the full extent, so a cropped
run's coverage ratio collapsed against a denominator that no longer described what it was computing.
Fixed by cropping the denominator with it.

**The general shape is worth more than the instance**, and it recurs: an optimisation that narrows
what is COMPUTED silently changes every ratio whose denominator was the old extent. The same error
appears as §3.9's granularity artefact in the ingest record, and as the "presence counted where
coverage was meant" family in the corrections register. **When you crop a numerator, go and find its
denominators.**

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

## Appendix A — the multi-write-per-commit contract (verified 2026-07-24)

Six tests, and what they establish is one contract the ingest still depends on:

**N `to_icechunk` region writes CAN share one `writable_session` and one commit** — twenty disjoint
chunk-aligned windows produced exactly one snapshot with no cross-window interference, ~10 ms per
write locally with no degradation. `to_icechunk`'s internal fork/merge tolerates a session that
already carries uncommitted changes.

**A session sees its own uncommitted append**, so `mode="a"` along time followed by `mode="r+"`
region writes into the just-appended date commits as ONE snapshot. That is what lets dates stay
discovery-as-you-go: append the slot, write its live windows, commit once, with no pre-enumeration.

**Uncommitted writes are invisible to readers on separate repo handles**, so commit atomicity holds.

**The caveat that is still the contract:** every window tested was exactly chunk-aligned, so
`align_chunks` never had to read-modify-write a boundary chunk. **Two writes straddling the SAME
chunk in one session remain forbidden** — disjoint chunk-aligned windows is the contract, not
"multiple writes work".

Verified on icechunk 2.0.4 / zarr 3.2.1 / xarray 2026.4.0. Full transcripts in git history.
