# Ingest optimization campaign — July 2026

> **Superseded in part.** Throughput at fleet scale was re-measured on 2026-08-04 and every
> zone ran 1.8-2.1x slower than the figures here at the same zone and the same width. The
> width conclusions below still hold (6x workers buys 3.7-4.9x, confirmed by same-zone
> pairs); the DURATION basis does not. See
> `ingest_concurrency_investigation_2026_08.md` before quoting any per-date or per-zone
> duration from this document.

Authoritative record of the campaign-ingest optimization work: what was changed, how much
each change bought, what was tried and abandoned, and the constraints future work must
respect. The empirical basis for why the ingest path looks the way it does.

Companion to `inference_gpu_saturation_profile_2026_07.md`, which is the same kind of record
for the GPU side. Read that one before touching anything the fill reads — several decisions
here were made *because* of what it measured.

Related notes: `ingest-live-tile-cropping.md` (why ingest crops at all, and how windows are
derived), `ingest-graph-and-stac-budget.md` (the living working notes this summarises),
`region-writes.md`, ADR-011 (campaign zone ingestion).

Unless a figure says otherwise, every wall-clock number here is the **same cell** — zone `35N`,
January 2024, on a 120-worker Dask-on-Fargate fleet against one frozen ROI mask — so the rows
compare directly. §10 records the finer measurement caveats and how to keep this document.

**Start with §1.** It is the whole story at a level you can act on. Detail deepens as the
document goes: §2 gives the cost model, §3 and §4 are the per-change archive, and §8 holds the
numbers of record.

---

## 1. Overview

### What the ingest actually does

For each UTM zone and year, the ingest builds a cloud-free mosaic of satellite imagery on a
fixed pixel grid. For every acquisition date it asks the satellite catalogue what images exist,
reads the ones it needs, discards cloudy pixels, and appends that date to a Zarr store. Sentinel-2
optical imagery and Sentinel-1 radar are separate stores built by separate runs, and a campaign
cell runs them at the same time.

The work is distributed with Dask on Fargate: one scheduler and a fleet of workers per run. The
scheduler breaks the work into tasks and hands them out; the workers read, reproject, mask and
write. Almost everything below is about the relationship between those two.

### Where the time goes, what still limits us, and three corrections

**Three facts explain nearly every decision in this document.** A zone is mostly ocean — a tall
thin strip of which typically a fifth is land, so computing the whole grid meant most work
produced nothing. Source reads dominate what remains: 72% of task work is fetching and resampling
COGs. And a large part of each date was never work at all but *waiting*, which is why removing a
serial term beat shrinking one.

**What still limits us.** The fleet is already full on a dense region: one date's work
oversubscribes the workers, so adding workers to a single run buys much less than it appears to —
the measured ceiling from unlimited workers is under 3×. That also disqualifies several
optimisations that look like they should help, because **there is no idle capacity for them to
fill.** Date batching (§3.16) is the clearest case.

**Three corrections to carry into any planning.**

1. **Do not size cells against the dispatch-rate model in §2.** Its shape holds, its constant is
   stale, and the advice it produced — keep cells narrow — is **withdrawn**. Aggregate throughput
   is flat within ~6% from 20 to 120 workers (§3.14).
2. **Every velocity figure measured early in a run understates the full-year cost**, because later
   dates image more of a zone's land. The same effect across seasons is what produced this
   programme's most-repeated withdrawn claim.
3. **Cost is not proportional to area.** A rectangle costs the greater of a fixed per-window term
   and its task count over the dispatch rate — `per_date ≈ C + Σ max(F, tasks/R)` — so window
   COUNT, not area, is what a windowing strategy must minimise once areas are small.

## 2. The shape of the cost — a picture and two formulas

### Where a date's time goes

```
   ONE DATE, dense zone, 60 workers  (~175 s)
   ├─────────────────────────────────────────────────────────────────┐
   │ prepare ~20 s        │ write ~150 s                    │ commit │
   │ ask the catalogue,   │ read + reproject + mask + store  │  <1 s  │
   │ build the graph,     │ every live rectangle             │        │
   │ check for cloud      │                                  │        │
   └──────────────────────┴──────────────────────────────────┴────────┘
       ^ can overlap the                ^ THE COST. Of the part that
         PREVIOUS date's write            runs on workers, ~72% is
         (this is "pipelining")           reading and resampling
                                          source imagery — real data
                                          movement, not overhead
```

Two things follow. The write dominates, so anything that does not shrink the write, overlap it,
or remove a boundary from it will not move the total. And the preparation is *hideable* — it can
run during the previous date's write — which is why the overlap changes paid and why several
later ideas did not: once preparation is fully hidden, hiding it harder buys nothing.

### How big is the work? Task count, not pixel count

```
tasks for one date  ≈  4  ×  bands (11)  ×  area in blocks
                       ^        ^                  ^
                       |        |                  how many blocks the
                       |        |                  live rectangles cover
                       |        one task set per band
                       fetch, mask, select, store
```

This predicted real graphs to within 0.5% (a 972-block window predicted 42,768 tasks and
measured 42,588). It matters because the **scheduler dispatches tasks on a single thread**, so
task count — not data volume — is what limits how much fleet one run can absorb. Every change
that worked reduced one of those three multiplicands; every change that failed either raised the
count or traded it for something dearer.

> **Dead hypothesis worth recording.** The loader emits one open-file task per source *image*
> (about 956 for one day over a 6° zone), **not** one per image per block. The per-block path was
> already lean, so there is no elementwise fat to fuse away. Anyone assuming a thousand images
> means a thousand tasks per block will optimise the wrong thing.

### Why cost is not proportional to area

```
per_date  ≈  C  +  Σ over rectangles  max( F , tasks / R )
                                      ^^^^^^^^^^^^^^^^^^^^
                          a rectangle costs the GREATER of a fixed
                          overhead and its dispatch time — never the sum
```

- **F** — fixed cost of one rectangle, about 7 s. Binds when the rectangle is small.
- **R** — the scheduler's dispatch rate. Binds when the rectangle is large.
- **C** — per-date constant: graph build, cloud check, commit.

The `max()` is the whole point, and it is why a simpler "rectangles × fixed + area × rate" model
was fitted first and **failed its first out-of-sample test** — it predicted 122 s where 194 s was
measured. Small rectangles are latency-bound and large ones dispatch-bound, and a model that adds
the two instead of taking the larger cannot reproduce either regime:

| rectangles | tasks each | regime | predicted | measured |
|---|---|---|---|---|
| 197 | ~540 | latency-bound, F binds | ~1,407 s | 1,329–1,471 s |
| 15 | ~7,500 | dispatch-bound | ~180 s | 194–242 s |
| 3 | up to 84,000 | dispatch-bound | ~199 s | 193.9 s |

> **The dispatch rate R is STALE and must not be used for planning.** It predicted a 150–190 s
> floor for a dense date; the write alone now measures 79.8 s, roughly twice as fast. The advice
> it produced — keep cells narrow to stay under the floor — is **withdrawn**. The `max()` shape
> still holds; only the constant is wrong.

### And the campaign multiplies out like this

```
campaign time  =  zone-hours per year  ×  years  ÷  cells run at once

                  ^ set by per-date cost      ^ set by the QUOTA, because a
                    and dates per year          real cell is three fleets
                                                (S2 + both S1 orbits) at
                                                ~744 vCPU, not one at ~248
```

**The divisor is set by quota, not by contention** — measured, not assumed, and now up to 20
concurrent cells. Per-window cost paired by zone: 5 → 10 cells is **1.01×**; 10 → 20 cells is
**1.24×**, against the **1.33×** predicted by the narrower fleet that quota forced at 20. Fleet
width more than accounts for the slowdown, so no contention term survives. No spill, and no task
ever waited for a worker. An earlier forecast applied a 1.04× penalty per cell — implying 2.56× at
40 cells — which is **withdrawn**.

Raw per-date medians do not compare across rungs, because each used a different set of zones and a
date's cost tracks its **rectangle count** (in the 20-cell rung, a 10-rectangle zone cost 136.7 s
and a 3-rectangle zone 25.5 s in the same run). Normalise per rectangle and pair by zone, or the
comparison measures zone composition instead of concurrency.

---
## 3. What each change bought

### 3.1 Cropping to live windows — the precondition

Ingest computed the full declared zone grid regardless of land content. On the first real
campaign cell, zone `03S` (4 live tiles of 14,355) built a **119,002-task graph and spilled
~1 TB** of worker memory before being cancelled. Measured across all 112 land zones, **~78% of
campaign ingest compute was ocean**. Cropping restricts loads and writes to chunk-aligned
windows intersecting the ROI mask. Details in `ingest-live-tile-cropping.md`.

Two follow-ups were needed before cropping actually paid:

- **The mask scan was serial** — ~3,700 sequential reads, 336 s from a laptop. Because
  all-ocean chunks are never written, the mask's stored chunk KEYS *are* the live grid, so one
  listing replaces the scan: **336 s → 0.38 s**, identical windows. Errs only toward more work
  (a written-but-empty chunk widens a window), and falls back to the pixel scan for any layout
  it does not positively recognise.
- **The coverage denominator was still full-extent**, as was the mask `persist`, and
  `apply_roi_mask` computed a per-date full-extent reduction that *both callers discarded*.
  Optical per-date graph **119,002 → ~154 tasks**; spill **468 GiB → 0**.

### 3.2 Grouping row bands — 6.2×, the largest single win

One window per live chunk-row minimises computed *area*, and area turned out not to be what a
windowed ingest is billed for. Each window is a separate **blocking** region write, and that
cost is close to fixed: **7.5 s per window** on dense `35N` across a fleet whose 480 task slots
averaged **6% busy**, with the graph completely EMPTY in **36%** of scheduler samples — twice
for two minutes straight. A sparse zone measured ~6 s per window at a fraction of the fleet.
Per-window cost was flat across an order of magnitude of fleet size, which is the signature of
a serial stall rather than distributed work.

197 windows → 15 took a date from ~1,400 s to ~225 s. Fleet occupancy went from 30.6 to ~407
tasks in flight (6% → 85% of slots); mean graph size 482 → 13,400.

This also explained, from the other direction, an earlier fleet-scaling result that showed only
**35% efficiency**: adding workers could never help while the fleet was idle between windows.

### 3.3 The masking collapse — 1.28× graph, free

Zeroing cloud-invalid pixels and zeroing outside the ROI both fill with 0, so
`x.where(A, 0).where(B, 0)` is `x.where(A & B, 0)` — and each `where` is a task per (block,
band). Measured on a real window: **99.9 → 77.9 tasks per block**.

SCL keeps only the ROI mask: it is categorical, and zeroing it by its own validity would
rewrite the class codes it carries.

### 3.4 Fusing the coverage gate into the band load

SCL is one of the eleven written bands, so the gate's separate SCL-only load re-read the same
data and paid a second client-side `odc.stac.load` build — **3.47 s of the 7.2 s** spent
single-threaded per date while all 480 slots idled. The gate now takes an already-loaded SCL
slice. A date that fails the gate has built a graph it discards, which is cheap: construction
is the same order either way and nothing was computed.

The gate had **no direct test** before this. It decides which dates are ingested at all and the
ingest marker makes that permanent, so it now has one — including a case that sets the
threshold exactly at the true percentage, so a miscounted numerator flips the verdict in either
direction.

### 3.5 Decoupling load blocks from store chunks — the key structural move

> **REMOVED 2026-07-27 — the decoupling is gone; load blocks equal store chunks again.**
> The 8192 load block capped a date's parallel width at `blocks × bands`: on a compact ROI
> (Iowa, 126 store chunks → ~27 load blocks → ~297-wide graphs against 480 slots) the write ran
> **~5× slower than its own work content**, measured against a main-code control doing 1.6× the
> executed work in half the wall clock. The dispatch saving that justified 8192 was real but is
> a *fleet-width-independent* floor (`~44 × blocks / R`), and at the campaign's chosen 40–60w
> cells it roughly coincides with the fleet-work line — §3.8's config table shows the two walls
> meeting at ~185 s at 120w, which was misread at the time as "8192 costs nothing". Densest-zone
> cells at wide fleets are the one regime that genuinely benefited; date batching (planned)
> recovers width there without re-coarsening reads. Numbers: the 2026-07-27 Iowa paired profile
> in `yield-embeddings/context_docs/measurements/`.

The original rationale, kept for the record: graph tasks scale with the number of *blocks* the
read path builds, not with pixels. So `INGEST_LOAD_CHUNK_SIZE` (8192) set the dask block size
while `INGEST_CHUNK_SIZE` (4096) kept setting the store's chunking, a whole multiple enforced
at import: the write split blocks down to store chunks, and a non-multiple would make that a
cross-block shuffle.

Windows are derived on the load-block grid by **coarsening the live grid**, not by snapping
windows afterwards. Snapping is incorrect: two windows on adjacent fine rows can snap into the
same coarse block and stop being chunk-disjoint, and the single-session per-date write depends
on disjointness. The mask stays chunked at `INGEST_CHUNK_SIZE`, so the fast key-listing path is
unaffected.

### 3.6 Manifest sharding — a cost win, not a speed win

Unsharded, every commit rewrites the whole array manifest, so bytes rewritten grow as N²/2 in
the date count: ~1.1 MB of chunk references per date means a 250-date zone-year rewrites
**~35 GB cumulatively** and ~275 MB on its final commits. Sharding by time makes growth
sawtooth instead: **~1.2 GB per zone-year, ~28× less**. Measured at 9 dates: 34.6 MB against a
34.7 MB prediction.

That is ~1–3 s per date, i.e. **worth having because it is free, not because it is fast.** An
earlier claim in these notes that it would flatten an observed per-date drift was withdrawn —
see §5.

### 3.8 Where the graph work ended, and why — the write floor

**SUPERSEDED by §3.9, and the whole section with it.** It derived a graph floor of
`store_chunks x bands` and concluded graph work was closed. The arithmetic was right and the
denominator was wrong: that is the floor for writing the WHOLE ROI, and a date writes only the
fifth its own imagery reaches. The real floor is about five times smaller and the conclusion was
premature. The working is in git history; §3.9 is what to read.

### 3.9 The per-date footprint — shipped, correct, and worth zero wall clock

**Found by counting objects, not by profiling.** A 7-date run on a dense zone wrote 44,797 chunk
objects — 582 per band-date — while its graph covered 2,992 chunks. So **19.4% of the graph
produced data**, against a Sentinel-2 revisit that predicts 20.0%. The live windows describe
where the ROI has LAND and were reused unchanged on every date, but one optical pass images only
a fraction of a 6-degree zone. Four fifths of the write tasks ran, found nothing, and wrote
nothing. **The waste was invisible precisely because the output was already correct.**

**It cannot appear on the single-ROI path**, and that is worth knowing before dismissing someone
who has not seen it: the ratio is ROI extent over what one pass covers, and a yield ROI is
smaller than one swath, so a date covers essentially all of it.

**The fix** (`live_windows.windows_for_date`) intersects the run's windows with the footprint of
that date's own items before building the graph. It provably cannot change a mosaic — the chunks
it skips are already absent, which is what 582-against-2,992 proves — so the entire safety burden
is that the footprint be CONSERVATIVE: every uncertain path returns "assume everything", rounding
goes outward, and the footprint is padded a whole cell against a curvature error measured in
metres. The coverage gate deliberately keeps the FULL window set, because its ratio asks how much
of the ROI's land a date saw and cropping the denominator would rescale every percentage.

**Result: object-for-object identical stores, 1.01× on the clock, −23% graph tasks, −35%
scheduler CPU.** The correctness claim held exactly. **The performance projection did not, and
was wrong in two independent ways worth carrying forward.**

**1. The 19.4% was a granularity artefact.** It counts 4096 px chunks; windows are built on the
8192 px grid, where one cell is 82 km and a swath edge clipping a cell dirties all of it.
Measured at the grid the ingest actually uses, a date touches **47–57% of live cells**, not 20%.
**Nothing can narrow below its own grid.**

**2. The dead area splits two ways, and an earlier version of this section stated the split
backwards** (withdrawn, §5). Over three dates at the 4096 grid: **geometric dead 55–66%** of live
chunks — no image that day, and a window strategy can skip it — against **radiometric dead
10–21%**, imaged but masked to zero, which it cannot. Geometry is the larger share, so what
limited the shipped change was the grid it operates on, not the nature of the waste.

**Kept anyway, on the scheduler rather than the clock.** The graph and CPU reductions are the
currency that buys worker count, and §3.10 shows the clock was never graph-bound. That is a
hypothesis about scale, not a measured benefit — and it is recorded as one.

### 3.10 Where a date's time actually goes — the profile that reframed the campaign

Measured with an existing hook — `perf_report_uri` is already a flow parameter, so a two-date
probe captured the Dask task stream with **no code change**.

**Source read and resample is 72.3% of task work**; mask and region write 16.3%, inter-worker
transfer 7.8%, coverage gate 2.9%. Stage shares are the durable output here; **every absolute
packing figure this section originally carried was computed from a TRUNCATED task stream and is
withdrawn — §3.14 has the corrected values**, and they are in §8.

**Three things it settles.**

*The graph was never the cost.* 72% of compute is fetching and resampling COGs against the public
archive. That is why removing 23% of the graph (§3.9) moved the clock by nothing, and why §3.8's
write-floor framing pointed at the wrong quantity.

*More workers cannot help much.* Only the packable fraction packs; serial client work, dispatch,
blocking region writes and commit do not — consistent with fitting `A + B/W` to a three-rung
worker sweep, which is worth noting because the two methods share no inputs.

*It reconciles two results that looked contradictory.* Real I/O should scale with workers and
does; a worker sweep showed scaling flattening above 60 workers, and it does, because the
residual is more than half the date. Both observations were correct.

**The lever it identifies:** the residual is the target, and shrinking it is not the only way to
remove it — **overlapping one date's serial phase with the next date's compute takes it off the
critical path entirely.** The first ingest change in this campaign proposed from a profile rather
than from a model.

### 3.11 Overlapping a date's windows — 1.59×, and the first change proposed from a profile

**Phase I: instrument, then decide.** Three permanent log lines partitioned a date over 7
sequential dates: build 10.8 s, gate 7.2 s, **write 148.0 s**, commit **0.6 s**. The per-window
times summed to **146.9 s of that 148.0 s write phase**, with the longest single window only
37.8 s. So `write_day_windows`' loop — one blocking `to_icechunk` per window — was the residual:
the fleet worked one window at a time and idled through the rest. **The commit was never the
cost.**

This was gated on a **pre-registered kill condition**: had one window been most of its date, the
overlap would have bought nothing and the plan said stop. It was not.

**Phase II: one compute for the date.** icechunk's dask path already forks a session, stores
lazily and merges; `to_icechunk` merely runs that once per call. Lifting the sequence one level —
build every window's writer, fork ONCE, collect all lazy stored arrays, run ONE merge reduction —
puts every window's loads, masks and writes in a single graph whose critical paths overlap. One
commit per date, `align_chunks`, the alignment guard and the failure contract all unchanged by
construction.

**Phase III: the A/B, identical dates and width.** Write phase **148.0 → 86.5 s (1.71×)**,
per-date **166.1 → 104.2 s (1.59×)**, peak worker memory 12.30 of 20 GiB with zero spill — and
**44,797 chunk objects on both arms, identical**. Pre-registered prediction was 90–110 s at
1.5–1.8×. **This is the only projection in this campaign that landed**; the two that did not
(§3.9, §5) were both projected from aggregates without measuring the mechanism first, which is
exactly what Phase I existed to prevent.

**Why not more.** The write phase falls to ~86 s rather than to the 37.8 s longest window: the
work still has to happen, and ~85 s is near the packing floor rather than the critical path.

**Now the default** for S2 (`overlap_window_writes=True`). `write_day_windows`' own
`parallel_windows` default stays **False** deliberately — S1 also calls it, its windows have
never been A/B'd, and a storage-layer default must not change behaviour for an unmeasured caller.

**The commit-growth concern it raised is RESOLVED — bounded, not cumulative.** Commit times
climbed monotonically across the parallel arm's 7 dates (0.5 → 1.2 s), which read as drift. A
longer run shows a **sawtooth of period exactly 8 dates**, climbing to ~1.5 s and resetting — and
8 is `INGEST_MANIFEST_SPLIT["time"]`. Each commit rewrites the current shard's manifest, which
grows as dates accumulate in it and resets when the next shard opens. **§3.6's manifest sharding
bounds commit cost as well as object count**, which was not among the reasons it was adopted.

The lesson is sharper than the finding: **a 7-date window cannot distinguish bounded periodic
cost from unbounded growth.** Both A/B arms ran 7 dates, entirely inside one 8-date shard, so the
instrument could not have seen the reset however carefully it was read.

**Robustness.** The lift uses `_XarrayDatasetWriter`, which icechunk marks private. The version is
lockfile-pinned, parity is pinned by test, and any import or signature drift falls back to the
sequential loop with a warning — so drift degrades to previously shipped behaviour, never to a
failure.

### 3.12 What the overlap did to the packing budget — and why §3.10's programme verdict survives

**Every absolute figure this section carried is withdrawn — the task stream was truncated, and
§3.14 has the corrected values.** Two conclusions survive the correction, and they are the reason
the section remains.

**The overlap REMOVED serial time; it did not convert serial time into packable work.** The
obvious reading gets this backwards. Packed task work barely moved; what collapsed was the
residual. The windows' critical paths were never *work* the fleet could have absorbed — they were
waiting, and the fix deleted the waiting rather than parallelising it.

**§3.10's programme verdict stands.** The ceiling moved, but a large fraction of a date still
cannot be compressed by any worker count, so widening a single cell remains unprofitable and the
campaign's lever is **cell concurrency, not cell width**. It would have been easy to assume that
halving the residual freed the fleet to scale; measurement says it did not.

> **The truncation caveat in this section FIRED, and how it failed is worth more than the numbers
> it got wrong.** It reasoned: "the two arms captured 182,311 and 178,123 tasks, which are neither
> round numbers nor equal, so truncation is unlikely but not excluded." Those totals include
> inter-worker transfer rectangles, which pad them. The NON-transfer counts are **100,008 and
> 100,025** — the 100,000 default cap, exactly. The instinct was right and the test was wrong:
> **a truncation check must count the records the cap applies to.**

### 3.13 The retained catalogue items — 2.45× smaller, and the first change forced by a LONG run

Everything above was found by shortening a date. This one was found by lengthening a run: a
two-month soak deadlocked at 53 of 60 dates when the worker running the ingest body crossed Dask's
memory pause threshold and never resumed.

**Why the driver is the worker that dies.** The ingest body runs as ONE task on ONE worker, so that
worker holds the streamed month plus the prefetched next month **and** its ordinary share of
compute. Per-worker memory settles near **13.45 GiB** — flat across six consecutive blocks, a real
ceiling and not a leak — while the driver sits **~4.9 GiB above** the fleet mean. Dask pauses at
14.90 GiB. It peaked at **14.89**.

**Why spilling could not save it.** **91–100% of worker memory is UNMANAGED**, so Dask has nothing
it is permitted to evict. Raising the spill threshold, spilling harder, or raising the pause
threshold all fail for the same reason: pausing frees nothing here, and Fargate has no swap, so a
higher threshold removes the guard without buying runway. **The fix had to delete memory rather
than manage it.**

**What was deleted.** Sentinel-2 L2A items carry **35 assets**; the ingest loads **11**. The rest —
previews, per-band JP2 variants, metadata documents — were retained for a whole month and never
read. Dropping unused assets and links takes an item from **86.8 to 35.4 KiB (2.45×)**, which at a
dense zone's 68,000 retained items is **~3.3 GiB off the driver**. Rehydration is a saving rather
than a cost: building from a pruned dict costs **252 µs against 810 µs**, because most of an item's
construction cost is the assets being dropped.

**A DENY-list, and a bug is the reason.** The more aggressive allow-list variant failed on first
attempt with "Failed to auto-guess CRS/resolution", because this collection carries its CRS in
`proj:code` where the list expected `proj:epsg`. That failure was loud. A subtler one — dropping
`raster:bands` scale/offset — would have **changed pixel values silently**. Dropping whole unread
assets cannot lose metadata the loader needs, and it is where 85% of the saving is. An item whose
asset names are all unrecognised is returned untouched.

**Shipped alongside: worker memory 20480 → 24576 MiB.** Both were needed; pruning alone leaves
every worker ~1.5 GiB below a hard threshold, which is the margin that just failed.

**The durable lesson, and the reason this section exists.** A 7-date A/B cannot see this — the
failure appears past ~40 dates. Worse, the first drift analysis was fitted over samples from
*after* the deadlock, when a frozen cluster reads as a plateau, and briefly concluded "bounded, not
a leak" from evidence that could not support it. **Exclude the post-failure tail before fitting any
resource trend, and size memory against a long run's asymptote rather than a short run's peak** —
20480 was itself chosen against a ~12.4 GiB peak on a short run, and the true ceiling is ~15.

### 3.14 The packing ceiling was measured off a truncated stream — it is ~2.8×

**The instrument was lying, and the check that should have caught it measured the wrong thing.**
Dask's task stream is a bounded deque, default **100,000** records — at ~25k tasks per date, about
**4 dates**. Both arms of §3.11's A/B hold **100,008 and 100,025 non-transfer rectangles**: the cap.
Each captured the run's last ~3.5 of 7 dates while the analysis divided by 7. §3.12 considered
truncation and dismissed it because the totals were "neither round numbers nor equal" — but those
totals include inter-worker transfer rectangles, which pad them.

| quantity | as recorded (§3.10, §3.12) | **corrected** |
|---|---|---|
| task work at perfect packing | 57.2 s, then 39.1 s | **64.3 s/date** |
| width-independent residual | 73.8 s (56%), then (62%) | **~37 s (~36%)** |
| ceiling, unlimited workers, ONE cell | 1.78×, then 1.60× | **~2.8× (2.0–3.0)** |

**Two independent instruments agree, which is what settles it.** The paired 60w/120w width
measurement gives `T = F + K/W` with no reference to any task stream. Fed the corrected figure it
predicts T(60w) = **166.4 s** against **167.9 s measured (−0.9%)**; fed the recorded figure it
predicted **141.2 s (−15.9%)**. And the "~30 s unexplained inside the write phase" that §3.13 named
as the next profiling target **was this arithmetic error**, not a phenomenon.

Fixed at source: `perf_budget.py` warns when the non-transfer count approaches the cap, and
`ecs_cluster` takes `diagnostic_task_stream`, wired to `perf_report_uri` so a report always covers
its whole run.

**The budget now closes.** Of a ~102 s date at 120 workers: **64.3 s of scalable task work** (71%
source COG read and resample; the profiler puts 61% of in-task time inside `rasterio._do_read`) and
**~37 s that no width touches** — commit and build 12–14 s, gate round-trip ~7 s, writer assembly
and graph submit 5–7 s, plus ~11 s of dispatch ramp and merge tail.

**On cell width, the answer is that it barely matters.** An earlier version of this section fitted
`T(W) = 36.3 + 7896/W` from **two** paired points and concluded 30–45 workers was ~20% better than
120. Two points cannot constrain a two-parameter model: a third control at 45 workers put `F`
anywhere from 11.4 to 39.3, and the three-point fit `T = 18.0 + 9391/W` makes aggregate throughput
**flat within ~6% from 20 to 120 workers**. That claim is **withdrawn**, along with a per-cell
interference figure of 1.04× that came from a single two-cell measurement — none is measurable to
20 concurrent cells (§1).

What survives: **wide cells are better than this record originally claimed and still lose per
vCPU.** 250 workers buys 1.46× per cell, 500 buys 1.89×, unlimited ~2.8×, while throughput per vCPU
falls monotonically. With flat width sensitivity and no measurable interference, the practical rule
is that **topology barely matters — pick the width easiest to run** and spend the quota on cells.

The remaining open constraint is not contention but **launch**: Fargate sustains **20 tasks/s, 100
burst**, and a 10,000 vCPU fleet is ~2,300 tasks, so about two minutes of ramp unless raised.

### 3.15 Load blocks re-unified with store chunks — width, not dispatch, was the limit

`INGEST_LOAD_CHUNK_SIZE` (§3.5) is gone; load blocks equal store chunks again. What §3.5 missed
is that a coarser load block caps a date's **parallel width** at `blocks × bands`, and that cap
binds on compact ROIs:

| workload | live 4096-chunks | read width @8192 | read width @4096 | fleet slots |
|---|---|---|---|---|
| Iowa (whole state) | 126 | ~297 | ~1,180 | 480 |
| a dense UTM zone | ~2,400 | ~6,600 | ~26,000 | 480 |

Iowa's *entire* extent is ~27 load blocks at 8192, so a date's graph could never fill 120
workers — it peaked at 0.88× of slots and spent each write draining a straggler tail of 4–7 s
tasks. A zone was never width-starved (already 14 tasks/slot at 8192), which is why the same
change is transformative for one and a rounding error for the other. Measured, 7 paired dates,
120 workers, same instrument:

| | build | gate | write | cycle |
|---|---|---|---|---|
| Iowa @8192 | 0.1 s | 4.2 s | 23.8 s | 27.9 s |
| Iowa @4096 | 0.2 s | 2.0 s | 14.9 s | **16.9 s** |
| 35N @8192 | 5.6 s | 6.9 s | 86.3 s | 98.6 s |
| 35N @4096 | 6.1 s | 10.0 s | 79.8 s | **95.5 s** |

The zone's gate got dearer (6.9 → 10.0 s) because windows are now derived on the 4096 grid, so
there are more of them (5 → 13). That is also the change's **second and larger campaign win**:
windows are chunk-aligned and kept if *any* part is live, so a finer grid hugs coastlines more
tightly. Computed offline from all 112 real zone masks (geometry only — no cluster):

**dead area 30.7% → 14.9%, i.e. an 18.6% cut in ingest write volume campaign-wide.** Sparse
zones dominate the gain (57S 65%, 60S 62%, 59S 52%, 58S 51%); dense zones see almost none.
Tool: `yield-embeddings/context_docs/measurements/window_efficiency.py`, raw output for all 112
zones alongside it.

> **A 2048 window grid is IMPOSSIBLE, and the residual has a different cause.** Three
> independent blocks: the write's alignment guard rejects windows not aligned to the 4096 store
> chunks; `coarsen_live_grid`'s factor is `window_px // chunk_px` = 0 below one chunk; and the
> mask is itself chunked at 4096 so liveness is not KNOWN any finer. The store's chunking is
> fixed (the GPU path is tuned around it). So the 14.9% residual is not granularity — it is
> **over-merging**, and §3.17 addresses it.

### 3.17 The window merge exchange rate — re-priced after overlapping

`merge_bands` groups adjacent row bands by minimising `n_windows x WINDOW_COST_IN_CHUNKS +
total_area`. The constant was 200, justified by its own docstring: "a window boundary is a serial,
blocking region write". §3.11 made that false — a date's windows now share one graph, so a
boundary costs a subgraph, a merge leaf and a changeset, order **15 chunks rather than 200**.
Priced at 200 the optimiser over-merged, trading real ocean area for a saving that no longer
existed.

Swept offline over all 112 real masks, geometry only: at **20 (shipped)** the campaign computes
**9.5% dead area against 14.9%** at 200 — **6% less area for 14% more windows**, which is the
right trade once a window boundary is nearly free.

**The general point is worth more than the constant.** A tuning parameter carries an assumption
about what is expensive, and a change elsewhere can invalidate it silently — nothing fails, the
optimiser just quietly optimises for the wrong thing. **When a cost model changes, re-derive every
constant calibrated against it.**

### 3.16 Batching dates — a win at one size, a LOSS at another, so it is sized per region

`batch_dates=k` computes k consecutive PASSING dates as one dask graph and commits them as one
snapshot.

**An earlier reading of "1.14x, adopt it globally" is SUPERSEDED.** That figure is real but it is
one point on a curve that is **not monotonic**: the same setting measured on four further regions
*loses* on two of them. Batching is therefore not a global setting — it is sized per region, and
the default is off.

**Why it can lose.** Batching trades commit count for graph size and peak memory. Where the fleet
has idle capacity the larger graph packs into it and the saved commits are free; where it does not,
the larger graph spills or stalls and the saved commits do not pay for it. Since §3.11 consumed
much of that idle capacity, **the case for batching is now weaker than when it was measured** —
noted again in §4.9's open items for S1.

**The transferable lesson: a single measurement of a tunable is a point, not a curve.** This one
was adopted globally on one favourable point and had to be withdrawn when four more regions were
tried. Sweep before generalising, or scope the setting to where it was measured.

## 4. What did not work, and why

### 4.1-4.8 What was tried and rejected — the table, so none of it is retried

Eight approaches measured and turned down. Each had its working; what a future reader needs is the
verdict and the reason, because the cost of losing these is someone re-running them.

| approach | verdict | why |
|---|---|---|
| **coarsen the STORE chunk to 8192** | rejected | the GPU side vetoes it: inference reads whole chunks, so a coarser store chunk multiplies read volume for every consumer. Measured, not argued |
| **cost-model window grouping** | rejected | over-merged into the scheduler; the optimiser traded real area for a boundary cost that had already stopped existing (§3.17) |
| **spatial manifest sharding** | rejected | a **30–50% regression** — the axis matters more than the size, and time is the right axis |
| **double load blocks again, 8192 → 16384** | not worth it | measured **1.35×**, against a prediction of 2–3×, at 537 MB per band-block |
| **several rejected without testing** | defensible | each contradicted a measured constraint already in hand; the section records which, so "untested" is not read as "unconsidered" |
| **remove the realignment** | reverted | projected 3.85× and ~5% materialised, because the census modelled the write layer instead of measuring it — and the memory claim came from the first twelve heartbeats of a run that later spilled |
| **narrow window geometry further** | rejected | three variants, all worse. The best any rectangle strategy achieves is 0.50× current area; the shipped one already achieves 0.75× |
| **worker memory back to 16 GiB** | ADOPTED, then WITHDRAWN (§8) | rejected while the driver held unpruned catalogue items, and it looked affordable once §3.13's pruning landed — but the peak it was sized against came from dates carrying half the windows of a 2019 optical date, and those denser dates paused workers at this size. Now **24576 MiB** |

**The transferable one is the realignment revert**: a projection built by modelling a layer rather
than measuring it, and a memory figure taken from the opening minutes of a run that later spilled.
Both are the same error — reading an early or synthetic sample as the steady state.

### 4.9 S1 never received the changes S2 depends on — and overlapping paid double

Sentinel-1 paid the serial penalties the S2 path had already shed. **Two of three axes are now
closed:**

| axis | S2 | S1 before | S1 now |
|---|---|---|---|
| windows within a date | one shared computation (§3.11) | **sequential** — a date costs the SUM of its windows | overlapped, by default (`12d2aed`) |
| window merge pricing | cheap-boundary rate (§3.17) | expensive rate — *correct*, given it wrote serially | cheap rate, once overlapping (§3.17) |
| dates per commit | sized per region (§3.16) | one commit per date | **unchanged, still open** |

The symptom that pointed here: on a sparse zone S1 cost ~39–59 s/date against S2's ~29–33,
**despite carrying 2 bands against S2's 11**.

**Overlapping buys 2.4–3.9×, and nothing predicts its size.** Measured over three ROIs at 23, 9
and 7 windows per date: 2.79×, 2.86×, 2.40× — **flat in window count**, with nine windows gaining
marginally more than twenty-three. A 2×2 then held it flat in fleet width too: 3.67× at 30 workers
against 3.85× at 60, inside the 5% noise floor.

> **THREE MECHANISM ACCOUNTS HAVE BEEN PROPOSED AND ALL THREE ARE REFUTED.** Sum-over-max
> (predicts the gain scales with window count — refuted, it is flat); fleet occupancy (predicts it
> grows with fleet width — refuted, doubling the fleet did not move it); and `min(query, write)`
> for the look-ahead below (predicts the gain is largest on sparse zones — refuted, it is exactly
> backwards). No fourth is offered. **Use the measured range and do not model it.**

Insensitivity to both variables tested is a *stronger* operational position than any mechanism
would have given: an effect that moves with neither can be applied across the campaign rather than
only where it was measured. Two cautions — the ratio is what shipping the overlap *and* the merge
re-pricing buys together, and the magnitude should not be extrapolated far outside 30–60 workers.

**Per-date window narrowing: shipped, 7–20%.** S2 narrows each date to the windows that date's
imagery can reach; S1 wrote the run's **full** land-window set every date. Offline against real
masks, a date's swath actually reaches **5–20% of live windows** (median 4–5 of 25 on 35N, 1 of 11
on 21N). So S1 was building **5–20× more window-writes per date than had any data in them** —
producing all-fill chunks that are never stored, so the mosaic is identical either way. Shipping it
cut windows per date six-fold (11.0 → 1.8) for **20.1% and 7.3%** of wall clock on two zones. Note
the share figures bound the *graph*, not the clock, and the conversion is nowhere near 6×.

**The dangerous failure mode was designed out.** Keying each date's footprint by solar day would
disagree with the loader wherever the offset crosses UTC midnight, narrowing to the wrong footprint
and dropping imagery silently. The join is on an **exact timestamp** instead: odc sets each slice's
time to `group[0].nominal_datetime`, so the minimum item datetime reproduces it exactly. An
unmatched slice writes every window — "reaches nothing" and "we do not know" are separate branches
and only the former skips.

> **Skipping dates that reach no live window fixes a latent correctness bug, which is not why it
> was built.** It removed 13 of 58 dates (22%) on one zone — days whose swath covers only ocean,
> carrying 483–1,113 granules each and reaching zero live windows. But some zones have an orbit
> that never reaches land at all: **40S and 24S ascending have granules on ~30 days of 2024 and
> reach zero live windows on every one.** Writing those dates created a SAR store full of fill,
> which `resolve_s1_orbit` then read as a *present* orbit — so inference consumed an all-fill band
> as real signal, the same hazard the dual-pol granule guard exists to prevent. With the skip
> nothing is written, no store is created, and the orbit is correctly downgraded.
>
> Residual case, **not observed** across eight zone-orbits: a zone with some months emptied and
> others not leaves a store missing those months, and the gate counts months present.
> `allow_partial_window` is the designed relief. Worth a census before the campaign.

**Catalogue look-ahead: shipped, ~10%, and the query hides completely.** S1 now prepares the next
batch during the current batch's writes, reusing `ingest._pipeline.pipelined` unchanged. Median
stall after the first batch is **0.0 s** against queries of 3.5–23 s. Per-date wall clock fell
**10.9%** on a 53-date arm. **One batch of look-ahead only** — deeper retention is what deadlocked
the driver (§3.13).

That is below the 14% query share, and the reason is the useful part: **a query run concurrently
gets slower**, 178 → 291 s on one zone, because the background thread contends with the write it
hides behind. The log's `hidden` field therefore overstates badly — 279.7 s claimed against 120.8 s
realised. **Treat the paired per-date difference as the result and `hidden` as an upper bound.**

The look-ahead's gain also **rises with density** (−12.5% at 4 live windows, +15.3% at 25), which
is the third refuted mechanism above. The sparsest zone loses because its look-ahead arm still
stalls: most batches held no data, so there was no preceding write to hide behind and it paid
contention for nothing. Absolute cost across the run: **1.4 seconds.** Recorded so nobody "fixes"
it.

**A sequencing rule worth generalising:** the query only became worth optimising *because* the
overlap made the write several times faster — at the sequential write speed it was 4–5% of a
batch, below the threshold for being worth building. **Speeding up one phase promotes the next, so
re-rank remaining work after every win rather than once.**

**S1 is NOT fleet-bound, so narrowing its fleet cuts cost as well as quota.** A 2.31× narrowing
(30w → 13w) cost only **1.34–1.52×** in time, not 2.31×, so worker-seconds per date fell **34–42%**
where the width-neutral forecast assumed zero. That is a large proportional cut on a base the
overlap had already shrunk to 9–16% of worker-hours, so the campaign total moves only ~4%, and
longer-running narrow fleets raise per-fleet scheduler cost enough to offset part of it. **Do not
reuse "worker-hours are width-independent"** — where a fleet is not the constraint, narrowing is
strictly cheaper on both axes, and the only way to know is to measure two widths.

**Also still open.** Date fusing is the least certain lever: by §3.16's arithmetic it wins only
where the fleet has idle capacity, and overlapping has just consumed much of exactly that, so the
case is now *weaker* than before. S1 was previously uninstrumented — it reported only "wrote N
dates", so a slow batch was indistinguishable from a slow catalogue, and those have opposite
remedies. It now emits per-date `write` and window `mode`, and per-batch `query` share.

### 4.10 A latent S1 correctness bug: credentials renewed on the wrong clock

The radar path renewed its read credential against the wrong clock, so a long leg expired mid-run.
Fixed; the mechanism and the guard are cause 1 of
[`ingest_read_failure_causes_2026_08.md`](ingest_read_failure_causes_2026_08.md), which is where that
class of failure is documented rather than duplicated here.

### 4.11-4.12 Three definitions of "a day", and they blocked four zones outright

Found by a 20-cell concurrency rung rather than by review, and on one zone of the twenty. **This
is why the solar-day offset must be applied exactly once, at the query chokepoint** — the
invariant the architecture rule now enforces.

**Layer one: which items form a day.** We grouped STAC items by **UTC calendar date**; the loader
groups by **local solar day**, shifting every timestamp by the geobox centroid's longitude
truncated to whole hours. Where the offset crosses UTC midnight the two disagree, so a group
believed to be one day arrived as TWO time slices against a cloud mask reduced to a single 2-D
slice — a dimension conflict naming nothing about its cause.

**Layer two: what the resulting slice is CALLED.** The same disagreement one level down, and it
stopped zone 56N ingesting S2 at all. It failed two different ways depending on the path, which is
how the scope became clear.

**The general shape is the thing to carry.** A "day" is defined independently in the query, in the
grouping and in the label; any two of them disagreeing produces a failure that names neither. One
definition, applied at one place, is the only arrangement that cannot drift — hence the rule.

### 4.13 Recording what an ingest EXAMINED, so an absent month is a finding

Forced by §4.9's empty-date skip. A mosaic's time axis says what was WRITTEN and nothing about what
was LOOKED AT, so a missing month meant either "examined, nothing reachable" or "the ingest never
got there" — and the coverage gate failed on both. That was tolerable while every date was written
regardless; the skip makes the first case normal.

Both ingest paths now record **`assessed_window`** on the store: the range processed in full. An
absent month inside it is a finding about the archive; an absent month outside it is a finding
about the run. **Before this, the two were indistinguishable and the gate treated both as failure.**

### 4.14 The crop flag was OFF by default, and a "tiny" zone exhausted its worker

The cropping shipped behind `crop_to_live_windows`, defaulting OFF, so the smallest land zone in the
scheme still built a full-extent graph and died. **The flag is gone** — removed once no scenario
wanted cropping disabled — and the validation that depended on it went with it, its precondition now
holding by construction (`ingest-live-tile-cropping.md`).

The lesson is about defaults rather than about this flag: a correctness-preserving optimisation
shipped OFF is a change nobody is running, and the first real cell is where you discover that.

### 4.15 An interrupted ingest is RESUMED, not rebuilt

A cancelled or crashed cell used to be CLEARED and re-ingested from scratch, so every interruption
cost a whole cell — and interruptions are expected, because the orphan sweeper cancels runs by
design. A dense zone interrupted near the end lost hours deterministically on every retry.

Three states, three answers. **Absent**: ingest. **Present and unmarked**: resume, skipping dates
already committed rather than rewriting them. **Present and marked with a different window**:
refuse, because that is a different question being asked of the same store.

### 4.16 Two writers, observed: what the guard caught and what it did not

47S/2021 was dispatched **four times** as four independent top-level runs, not retries. Generations
1→2→3 never overlapped and each resumed correctly from the previous one's commits, which is §4.15
working as designed. Generation 4 was dispatched **17 minutes into a healthy generation 3** and
collided with it.

**What the guard caught:** the concurrent commit, via the duplicate-date refusal — which is why that
error is in `CONCURRENT_WRITER_ERRORS` and means "another writer moved the branch" rather than "this
caller has a bug".

**What it did not:** anything before the commit. Two fleets ran, both paid, and only the loser found
out. The guard is a consistency guard, not an admission control, and nothing in the dispatch path
stops a second top-level run for a cell already in flight — which is what `dispatch_pending_fills`
refusing a cell a live run claims exists to address, one layer up.

## 5. Claims made and withdrawn

Recorded so they are not revived, and because the pattern is instructive.

- **"Four fifths of every date's graph can be skipped."** The 19.4% data share is real but is
  measured at the 4096 chunk grid, while windows are built on the 8192 load-block grid where a
  date touches 47–57% of live cells. The achievable prize was **2×, not 5×** (§4.7), and the
  shipped change already took three quarters of it for **zero** wall clock (§3.9).
- **"Most of the dead area is not geometric — it is cloud."** Backwards. Geometric dead is
  **55–66%** of live chunks and radiometric **10–21%** (§3.9). The claim was made to explain the
  null result and was not measured before being asserted; the real explanation is grid
  granularity plus the fact that area was never on the critical path.
- **"The sweep's timing gap came from contention with concurrent automation testing."** Asserted
  from three paired dates with no mechanism, and withdrawn: CloudWatch and ECS API calls do not
  contend with COG reads from a public bucket. An identical catalog query measured **37.6 s at
  21:45 and 33.1 s at 23:45** — external service latency drifts ~12% over two hours, which is a
  supported explanation where the contention story was not. The operative lesson is
  methodological: **timing comparisons across runs separated by tens of minutes are unreliable at
  the 20% level**, so the footprint verdict was re-based on deterministic quantities (windows per
  date, graph task count, object counts).
- **"1.41× from decoupling load blocks."** That was one cherry-picked date. On matched means it
  is **1.23×** (187.9 s against 225 s).
- **"Per-date cost drifts upward within a run."** Visible across the first six dates
  (180.2 → 195.5 mean) and **not** present over twelve: the later half held a single 289 s
  outlier and two of the fastest dates in the run. Unestablished at month scale.
- **"Manifest sharding may explain the drift."** It cannot — the arithmetic puts it at ~1–3 s
  per date. The drift claim was itself withdrawn above.
- **"`align_chunks` is ~11% faster kept."** Measured when blocks and store chunks were the same
  size, so the remap was a no-op and the difference sat inside the ~19% per-date variance since
  quantified. The correct answer at the current geometry is the opposite (§4.6).
- **"Doubling load blocks again gives 2–3×."** Measured 1.35× — see §4.4.
- **"Removing the realignment gives 3.85× and lowers peak worker memory 32%."** The graph gain
  was ~5% (the census modelled the write layer instead of measuring it) and the memory claim came
  from the first twelve heartbeats of a run that later reached 10.58 GiB and spilled. Reverted —
  see §4.6.
- **"Per-(block, item) overlap tasks dominate the graph."** They do not exist; the loader emits
  one open task per item.

---

## 6. Constraints future work must respect

**The scheduler is the known scaling wall, from prior experience as well as these
measurements.** Ingest has previously been run at **~250–300 workers with degradation
attributable primarily to the scheduler** — so the ceiling measured here at 120 workers is not
an artifact of the current quota, it is the same wall seen closer up. Any plan that assumes a
larger fleet is automatically faster should be checked against that.

**The Dask scheduler is single-core-bound.** 114 heartbeats at 120 workers: CPU 19–100% with
twelve samples at exactly 100 and **none above**, ten threads, 4 vCPU allocated. RSS peaked at
1.79 of 8 GiB. Raising vCPU cannot lift that ceiling; only fewer tasks, or more schedulers (one
per concurrent cell), can. Three signs that WOULD justify a bigger box, so it is not
re-debated each time:

1. Any sustained sample **above ~110% CPU** — auxiliary threads genuinely using cores, so vCPU
   pays. Absent at 120 workers; plausible at much larger fleets, and worth re-checking there.
2. RSS **above ~50% of the limit**, or climbing across a cell's dates. A scheduler OOM destroys
   a multi-hour cell, so memory headroom is cheap insurance.
3. **`lag` rising while CPU is below 100%** — contention or I/O, not compute. Profile; do not
   resize.

**The ingest body runs on a DASK WORKER, not the flow runner** (4 vCPU / 16 GiB). Verified from
the log stream carrying its output, and corroborated by an orphaned fleet committing a further
date seven minutes after the flow runner was hard-killed. Consequences: worker memory is the
limit that binds; an OOM there is killed and *retried*, so it loops rather than failing
cleanly; and raising flow-runner memory achieves nothing.

**Per-date variance is 19%** (SD 36 s, n=12). Resolving a 10% change needs ~14 dates per arm.
**Verify graph work by TASK COUNT** — deterministic from one date's graph, built locally with
no cluster (`dask.optimize`, then count `__dask_graph__()` keys). Several hours were spent
reading noise before adopting this.

**A local task census is only trustworthy where it counts a real graph.** The read side it
counts directly and its predictions have held. The WRITE side was modelled analytically and
over-predicted a change by ~20× (§4.6): the region write splits to store-chunk granularity
itself, so a model assuming one write task per block was simply wrong. Where a prediction
depends on what the write layer does internally, measure the live scheduler graph — do not
extrapolate from a census.

**Change one thing per run, or give each change a deterministic mechanism check.** Three
changes shipped together once; when wall clock regressed it could not be attributed from
timings and took an object count to identify.

---

## 7. Open questions

**Four conclusions carried out of questions this section opened**, kept because they are load-bearing
and cited elsewhere; the arguments are in the sections named.

- **The ingest is reproducible across the campaign and single-ROI paths** (§4.9, §4.11-4.12): 65 of
  65 shared dates, identical arrays and chunking, bit-identical pixels across all 11 bands. The
  earlier answer was no — 3.11% of pixels disagreeing — and three date-handling fixes landed between
  the two measurements.
- **The fleet-bound floor is reached** (§3.10, §3.12), so **cell concurrency is the lever and cell
  width is not**. Two caveats travel with it: the sweep's absolute values are not reproducible (§5),
  and its width curve ran pre-overlap.
- **A date's serial residual comes off the critical path**, 1.59× from overlapping (§3.11). The
  durable lesson: **removing a serial term beats shrinking it.**
- **Month-by-month streaming shipped** and is validated (§7b).

Genuinely open:

- **Is the 4096 write-window idea viable again?** §4.7 rejected it on ~17 s of serial cost per
  window; overlapping removed most of that, so the arithmetic that killed it no longer holds.
  Expected win is small and the memory cost real. **Measure the prize before building it.**
- **Is `F` fleet-invariant?** Decides whether a window cap tuned at 120 workers transfers at all.
- **Commit time versus manifest size** — the last unmeasured per-date term.
- **Does the scheduler's auxiliary-thread behaviour change at 250+ workers?**

Questions needing more workers than the quota allows are tracked as S-1..S-6 in the downstream
repo's `docs/global-tessera-test-plan.md`.

## 7b. STAC query streaming — shipped and validated

The blocker §7 was written around. The query ran once per window and retained every item, so a
zone-year needed ~27-30 GB on the 16 GiB worker the ingest body runs on. It died ~17 min in, Dask
retried it four times at ~70-75 min per cycle with the fleet idle, deterministically, on every
dispatch.

Streaming bounds retention to the month in flight plus the one buffered behind it — roughly 5 GB.
Prefetch depth **1**, deliberately: deeper retention is what deadlocked the driver (§3.13), and one
month ahead is enough to hide the query entirely.

**The cost it does carry is one extra month of retention**, which is why §3.13's item pruning was
needed alongside it rather than instead of it: streaming bounds *how many* items are held, pruning
bounds *how large each one is*, and the driver needed both to stay under the pause threshold.

## 7c. The Earth Search 502 that is NOT a depth limit, and not a page size either

Two distinct Earth Search 502s on `sentinel-2-l2a` are now on record, and they are easy to
conflate. Both are live and each needs its own remedy:

- **Response size.** A page of 250 items is refused on the FIRST request. Reproduced again
  here: the same window at `limit=250` returns 502 at page 1. Already remedied by
  `max_page_size=100` for this provider (see the comment in `config/providers.py`).
- **Individual page requests.** At `limit=100` some page requests are refused while the
  rest of the same walk is served. This section is about that one.

**Neither page size nor patience touches the second.** Halving the page halves neither the
items walked nor which requests get refused, and the `urllib3` ladder underneath has already
exhausted itself by the time the failure reaches us.

### The shape of it

```text
BEFORE — one search, one cursor chain, walked to the end

  sentinel-2-l2a | bbox = zone 34N envelope | 2019-02-28 .. 2019-04-01 | 56,558 items

   p1 --cur--> p2 --cur--> ... --> p288 --cur--> p289          ...      p566
  200         200                  200            502     (never reached)
                                                   |
                              every page's cursor comes from the page before it, so a
                              refusal is not a gap to skip -- there is no cursor for p290
                              and no offset to jump to. The walk is over, and so is the leg.


HOW IT FAILED — the remedies that look obvious, and what each one actually did

  re-ask the same page   502 502 502 502 502 502 502 502     <- urllib3 ladder, 8 deep
                          |_____ ~4 min of backoff _____|       then the leg dies
                         deterministic in the request: 6 of 6 repeats refused, 1.3 s each

  use a smaller page     no effect. Item 27,600 is item 27,600 whether that is p111 at
                         size 250 or p276 at size 100. This is the OTHER 502's remedy.

  wait longer            no effect. Latency was flat 0.6-1.6 s to the failure and the 502
                         itself returned in 1.3 s -- nothing was timing out.

  ask a shorter window   works, but ONLY if the window's LATE bound moves. See below.


WHY RE-CUTTING WORKS AT ALL — the catalogue pages newest-first, so a window's late
bound fixes its whole cursor sequence and its early bound changes nothing reached first

  03-16 ........... 03-22   refused p14  -+
  03-19 ....... 03-22       refused p14   +- same late bound, same cursors, same refusal
  03-21 ..... 03-22         refused p14  -+
  03-16 ..... 03-19         OK  53 pages -+
  03-21 . 03-21             OK  16 pages  +- different late bound, different cursors
  03-22 . 03-22             OK  15 pages -+

  So halving alone is not sufficient: the half that KEEPS the parent's late bound
  retraces the parent's cursors page for page. Measured -- 2019-02-28/2019-03-16
  completes in 213 pages while 2019-03-16/2019-04-01 is refused at p289, the same
  page and the same cursor as the whole month.


AFTER — a worklist of date windows. A refused window is RE-CUT, never re-asked.

  2019-02-28 ------------------------------------------------------- 2019-04-01
      |
      |  (1) numberMatched = 56,558 > _MAX_QUERY_ITEMS on page 1, so cut up front.
      |      Denominated in ITEMS, so a denser year yields more pieces, not deeper pages.
      v
   +---------+---------+---------+---------+---------+---------+
   | 02-28   | 03-05   | 03-11   | 03-16   | 03-22   | 03-27   |   each walked as its
   |   ..    |   ..    |   ..    |   ..    |   ..    |   ..    |   own cursor chain
   | 03-05   | 03-11   | 03-16   | 03-22   | 03-27   | 04-01   |
   +---------+---------+---------+---------+---------+---------+
       OK        OK        OK       502!       OK        OK
        ^                            |
        |  consecutive windows SHARE their boundary day -- 03-05 ends the first and starts
        |  the second -- so no acquisition can fall between them, and the id dedupe
        |  absorbs the one repeated day
                                     |
                                     |  (2) refused past page 1 -> re-cut THIS window and
                                     |      leave the other five alone. 502 is kept out of
                                     v      the retry ladder, so this starts immediately.
                        2019-03-16 .. 03-22   refused p14
                          |
                          +-- 03-16 .. 03-19   OK    53 pages   5,179 items
                          +-- 03-19 .. 03-22   refused p14
                                |
                                +-- 03-19 .. 03-21   OK   41 pages   3,944 items
                                +-- 03-21 .. 03-22   refused p14
                                      |
                                      +-- 03-21 .. 03-21  OK  16 pages  1,414 items
                                      +-- 03-22 .. 03-22  OK  15 pages  1,318 items
                                                                 ^
                                    a two-day window cannot be shortened while its halves
                                    share a boundary day, so AT THAT FLOOR ONLY they abut.
                                    Without it the recursion stalls here and the leg fails.

  The invariant that lets any of this touch a fingerprinted query:

    union of every window   ==   the input window     (a re-partition, never a narrowing)
    items keyed by id       across every search one query runs
    ==>  56,558 of 56,558 items, zero duplicates, on the query that fails in production
```

### The measurement that killed the depth hypothesis

The refusals seen during the campaign clustered at pages 276–307, which reads as a depth
ceiling. It is not one. Reproducing the exact failing query — one month of `sentinel-2-l2a`
over `bbox=2.4648,0.0000,39.7160,80.9758`, window `2019-02-28/2019-04-01`, `numberMatched`
**56,558** items i.e. **566 pages** at 100:

| observation | result |
|---|---|
| where the walk dies | page **289**, reproduced from a laptop |
| depth at failure | **51%** of the query — not the tail |
| per-page latency to failure | flat **0.60–1.57 s**; no page over 2 s; no growth with depth |
| the 502 itself | returned in **1.3 s** |
| the refused cursor, re-asked in 6 fresh sessions | **502 six times of six**, 1.2–1.4 s each |
| a shallow cursor of the same search, same minute | **200** |

Then the same query cut to `2019-03-16/2019-03-22` — a **6-day** window — was refused at
**page 14**, four times out of four. Same refusal, one twentieth of the depth.

Both refused cursors sit at the same instant:

    page 289 of the month : 2019-03-22T08:30:08.394000Z,S2A_37TCE_20190322_0_L2A,sentinel-2-l2a
    page 14 of the 6 days : 2019-03-22T08:30:27.109000Z,S2A_37TEM_20190322_0_L2A,sentinel-2-l2a

**So depth is not the variable.** The pagination token is a cursor, carried in the `next`
link's body under the key `next`, whose value is the triple
`(datetime, id, collection)` — for example
`2019-03-22T08:30:27.109000Z,S2A_37TEM_20190322_0_L2A,sentinel-2-l2a`. It has
`search_after` semantics and there is no `from`/`size` offset anywhere in the request, so the
service is not told how deep the walk has got and cannot be counting. A flat latency curve and
a 1.3-second 502 also rule out a gateway timeout and progressive deep-paging cost; a fixed
result-window limit would refuse at one page with an explicit 400.

### The cursor is not the poison: it is the (cursor, window) PAIR

Measured directly on 2026-08-22, and this supersedes the weaker "deterministic in the whole
request" framing. Walk `2019-03-21T00:00:00Z/2019-03-22T23:59:59Z` to its refusal at page 14,
then re-send **the byte-identical cursor** with only the `datetime` field narrowed:

| request | cursor | window | result |
|---|---|---|---|
| the refusal | `…08:30:27.109000Z,S2A_37TEM_20190322_0_L2A,…` | the 2-day window | **502**, 3 of 3, ~1.3 s |
| same cursor, narrowed | *identical* | the single day 03-22 | **200**, 18 items, 3 of 3, ~0.8 s |

So no cursor value is defective. A cursor the service refuses under one window it serves under
another, and the only thing that changed was the date range beside it.

**A sharper hypothesis, not yet confirmed.** In the one-day window that cursor is the LAST page
— 18 items, since the day holds 1,318 — while in the two-day window the same cursor must return
100 items, 18 from the 22nd and 82 from the 21st. The request that fails is one whose result
would cross a date boundary. That would explain the end-date property, it fits the month window
refusing on a cursor at the same instant, and it predicts that windows ending ON a date boundary
refuse less often — which the instant-boundary run's zero refusals is consistent with. It is a
hypothesis about a service we cannot see inside, on one seam, and the fix does not rest on it.

It is not the data and not the moment either: the day holding the refused cursor queries clean
on its own (15 pages, 1,318 items), and the refusal repeats identically from fresh sessions
minutes apart.

### The property that decides the fix: only the window's END matters

Walking the recursion by hand, every window that **ends** on 2019-03-22 is refused at page
14 on the identical cursor, however late it starts:

| window | days | outcome |
|---|---|---|
| 2019-03-16 .. 2019-03-22 | 6 | refused, p14 |
| 2019-03-19 .. 2019-03-22 | 4 | refused, p14 |
| 2019-03-21 .. 2019-03-22 | 2 | refused, p14 |
| 2019-03-16 .. 2019-03-19 | 3 | **OK**, 53 pages, 5,179 items |
| 2019-03-19 .. 2019-03-21 | 3 | **OK**, 41 pages, 3,944 items |
| 2019-03-21 .. 2019-03-21 | 1 | **OK**, 16 pages, 1,414 items |
| 2019-03-22 .. 2019-03-22 | 1 | **OK**, 15 pages, 1,318 items |
| 2019-03-21 .. 2019-03-23 | 3 | **OK**, 56 pages, 5,479 items |

The catalogue pages **newest-first**, so a window's late bound fixes its whole cursor
sequence; moving the early bound changes nothing the walk reaches first. **Shortening the
start does not clear the refusal — shortening the end does.**

Two consequences, both of which cost a design iteration to learn:

- **Halving a window is not sufficient.** The half that keeps the parent's late bound
  retraces the parent's cursors page for page. Measured: `2019-02-28/2019-03-16` completes
  (213 pages, 21,198 items) while `2019-03-16/2019-04-01` is refused at page **289** on the
  same cursor as the whole month.
- **The recursion must be able to reach a single day.** Splitting into windows that SHARE a
  boundary day cannot shorten a two-day window, so a naive overlap-only splitter stalls at
  `2019-03-21/2019-03-22` with a refused window and nothing left to try — verified, it fails
  the leg. At that floor only, `split_query_window` abuts instead.

### What shipped

Three things. `_MAX_QUERY_ITEMS = 10_000`, read against `numberMatched` on each window's
first page. A re-partition on any upstream-error refusal past page 1. And **502 removed from
the STAC retry ladder's `status_forcelist`**.

The ceiling is denominated in **items**, so a denser year is cut into more pieces rather than
walked further — 2019 roughly doubled 2017–18 and 2020–2025 will be denser, and that changes
the part count and nothing else.

Taking 502 out of the ladder is what makes the re-cut prompt rather than merely correct. The
ladder was pure cost on this refusal: it is deterministic in the request, so all eight
attempts re-ask a question whose answer is already known, and the re-cut that does clear it
cannot start until they are spent. 429 and 503 keep the ladder — those are the refusals where
waiting IS the remedy — and the CMR Granule query has its own ladder
(`opera_query._CMR_RETRY`), 502 included, untouched. Excluding a status the taxonomy still
classifies is deliberate and asymmetric, so a test pins the exclusion rather than only the
old containment invariant; re-adding 502 would otherwise quietly restore the backoff.

The ceiling bounds **cost, not correctness**: it is the re-partition-on-refusal that fixes the
defect, and the ceiling is what keeps a refusal from throwing away hundreds of pages of walk.

Validated end to end against the live catalogue on the query that fails in production
(`2019-02-28/2019-04-01`, 56,558 items). Every window is served, and the union of item ids
equals the whole window's `numberMatched` exactly — **56,558 of 56,558**, nothing lost and
nothing duplicated. The sub-tree under the one refused part recovers as:

    2019-03-16..03-22  refused p14
      2019-03-16..03-19  OK
      2019-03-19..03-22  refused p14
        2019-03-19..03-21  OK
        2019-03-21..03-22  refused p14
          2019-03-21..03-21  OK
          2019-03-22..03-22  OK
    => 9,166 of 9,166 items

The whole month costs **19 searches and 821 page requests**, against 566 pages for the single
walk that cannot complete — **1.45x** the page requests for a query that currently returns
nothing at all. The overhead is the pages re-walked after a proactive cut and the shared
boundary days.

### Wall clock, and where it goes

Two runs of the benchmark query from a laptop, item set identical in both
(56,558 of 56,558, zero duplicates), refusal tree identical (three refusals, each at page 14):

Three runs of the benchmark query from one laptop, all returning the same item set
(56,558 of 56,558, zero duplicates):

| | wall clock | HTTP requests | ms/item | ladder retry lines |
|---|---|---|---|---|
| re-partition as first shipped | **2,360.7 s** (39.3 min) | 821 | 41.7 | 24 (8 per refusal) |
| + 502 out of the retry ladder | **1,099.6 s** (18.3 min) | 821 | 19.4 | **0** |
| + instant boundaries | **795.2 s** (13.3 min) | **583** | 14.1 | 0 |
| a single unsplit walk, if it worked | — | 566 | — | — |
| §8 reference RATE, for scale | (316 s at this item count) | 566 | 5.58 | — |

**The request overhead is now 1.03x**, down from 1.45x. That is the whole
code-attributable regression: 583 requests against the 566 a single walk would make, the
excess being one partial last page per window plus a page-1 refetch per cascaded cut.

**The 1,261 s the ladder cost.** `total=8, backoff_factor=2` against urllib3's default
`backoff_max=120` sleeps 0+4+8+16+32+64+120+120 = **364 s per refused page request**, three
times over. Predicted 1,092 s, measured 1,261 s; the rest is per-page variance.

**The 304 s the shared boundary day cost**, and a caveat about how it was recovered. The
instant-boundary run made **zero** refusals — moving a window's end from `2019-03-22T23:59:59Z`
to `2019-03-22T00:00:00Z` changed its cursor sequence and missed the position that refuses. That
is consistent with the end-date property but it is **luck on this query, not a fix**: a different
window will land on a refusing cursor again, and the recursion is still what handles it. Do not
read "zero refusals" as "the 502 is gone".

**What remains is bandwidth, not the algorithm.** A page of 100 items is 4.5 MB, so 583 requests
move about **2.6 GB**; 795.2 s over 583 requests is 1.36 s/page, or roughly 3.3 MB/s, which is a
laptop's link and not a property of the code. The §8 reference rate implies 0.56 s/page — 8 MB/s,
which an in-region worker has. At that rate this query costs **327 s against a single walk's
316 s**, so on campaign infrastructure the re-partition should be within a few percent of the
walk it replaces. That in-region figure is a PROJECTION from §8's rate and has not been measured
on a worker.

Note that §8's "164 s/month" is a ~29,390-item month while this query is 56,558 items, so the
comparable quantity is the **rate**, 5.58 ms/item, never the duration.

**One inefficiency remains and is NOT fixed.** The proactive cut CASCADES: `_parts_for_depth`
sizes parts as `ceil(matched / _MAX_QUERY_ITEMS)` and cuts by equal day count, so where density
is uneven a child can still exceed the ceiling and be cut again — the run cut the month into 6,
then re-cut 03-22..03-27 and 03-27..04-01. Each cascade wastes one page-1 fetch, which is why it
is worth only a few requests out of 583. Sizing parts by density would fix it, but changing the
part count moves the window boundaries and therefore the walk order, so it needs the order check
above re-run rather than being treated as free.

**Cost note.** Consecutive windows share their boundary day rather than abutting it, so no
acquisition can fall between them — the catalogue is asked for whole days and expands them to
instants itself, and abutting would leave the sub-second gap at midnight. The overlap costs
one repeated day per seam and the query's existing id dedupe absorbs it. The two-day floor
described above is the one place that trade is reversed, and it is taken because the
alternative there is a failed leg.

### What the item ORDER does, which the first verification did not check

The 56,558-of-56,558 check above is a check on the item **set**. The order is a separate
property and it matters: `query_stac_items` sorts by `(solar date, cloud cover DESCENDING)`
with Python's stable sort, so items tying on both keys keep their **input** order, and that
order is what the `odc.stac` painter consumes last-write-wins per pixel.

**The walk order does change, and always did.** A single walk returns the whole window
newest-first; the worklist returns window by window in DATE order. Measured on
2019-03-05/2019-03-08 (5,034 items), the unsplit walk and every split of it diverge from
index 100 — the first page boundary. This is a property of the re-partition as first
shipped, not of the instant boundary.

**The post-sort order does not change.** Same window, through `query_stac_items` rather than
the private walk, at part counts 1 (unsplit ground truth), 2, 3 and 4:

| | item set | walk order | post-sort order | baselines |
|---|---|---|---|---|
| split 2, 3, 4 vs unsplit | identical | **differs** | **identical** | identical |

The mechanism is that each solar date's items land in exactly ONE window, in catalogue
order — a shared boundary day is deduped whole into the earlier window, and an instant
boundary does not divide a day at all — so the sort restores the same sequence a single walk
would have produced. What remains unverified is a solar day that straddles UTC midnight, in
the far-eastern and far-western zones, where one solar day CAN fall in two windows. That case
is not covered by this measurement.

### Levers that look obvious and are closed

Both measured 2026-08-22 against the live catalogue, so nobody has to test them again.

**A bigger page is refused.** The pages are the cost — a page of 100 items is **4.5 MB**, so
the benchmark query moves roughly **2.5 GB**. Halving the request count by doubling the page
would be the obvious win, and Earth Search will not serve it:

| `limit` | page 1 | median latency |
|---|---|---|
| 100 | **200**, 4,524 KB | 1.59 s |
| 150 | **502** | 1.24 s |
| 200 | **502** | 1.64 s |
| 250 | **502** | 2.22 s |
| 400 | **502** | 3.55 s |

100 is already at the ceiling; there is no headroom above it. Note that the refusal latency
RISES with the requested limit, which is what a response-size ceiling looks like — the service
assembles the answer, finds it too large, and fails. This is the same first-page refusal
`max_page_size` was lowered for, and it is why that setting cannot also be the remedy for the
deep-page refusal.

**Server-side field selection saves 4%.** Earth Search advertises
`https://api.stacspec.org/v1.0.0/item-search#fields`, and it honours it for `properties` —
asking for six properties returns exactly those six. It does **not** honour it for `assets`:
all 35 assets come back regardless, including every `-jp2` duplicate, and `exclude: [links]`
leaves `links` in place. Since the assets are what make an item 45 KB, the measured saving is
45.2 -> 43.3 KB/item. Item ids and their order are unaffected. Not worth the change.

### The seam costs a day, and an instant boundary removes it

Consecutive windows originally SHARED their boundary day, because a bare date end is expanded
by the client to `T23:59:59Z` — verified on the wire, `2019-03-05/2019-03-06` is sent as
`2019-03-05T00:00:00Z/2019-03-06T23:59:59Z` — so windows abutting on consecutive DATES leave
the last second of each seam's earlier day unasked for. That gap is real and the sharing was
the right call against it.

But the shared day is fetched twice and thrown away once. Adjacent days share **no** items at
all (measured: 2019-03-05 has 1,214, 2019-03-06 has 1,298, the two-day window has exactly
2,512, overlap zero), so every seam re-walked a full day — about **13 page requests** — for the
dedupe to discard.

The fix is to abut on an **instant** instead: interior boundaries are rendered
`T00:00:00Z`, shared by the window that ends there and the one that starts there, while the
outer bounds are the caller's own strings passed straight back. The catalogue's range is
inclusive at both ends, so the union is the input window exactly — no gap and no overhang —
and the overlap is one instant per seam rather than one day. It also removes the special case:
a shared day cost a day of length and so could not shorten a two-day window, which is why the
first version had to abut at that floor and accept the gap there.

### Rejected: subdividing the bounding box instead

Considered, because the query boxes are enclosing rectangles over scattered live tiles and
`land_mask.live_tile_block_bboxes_wgs84` already cuts a zone's grid into per-block boxes.
Rejected for this fix on three grounds:

- **It would not have fixed it.** The refusal is a function of the date window and cursor.
  Nothing measured here implicates the bbox, and every clearing change was a change of end
  date.
- **It is a narrowing, not a re-partition.** Each block box is tight to that block's own live
  rows and columns, so the union of the boxes is a strict subset of the envelope. Items
  intersecting the envelope but no block would stop being returned, which changes the item
  set — and the item set is what a mosaic's content identity rests on.
- **It is not reachable from the query.** `RoiMetadata` carries `bbox_wgs84` and nothing
  tile-shaped, while `live_tile_block_bboxes_wgs84` needs the `ZoneSpec` and the live-tile
  bitmap, which exist only at coverage-build time. Adopting it means writing a block list
  into the ROI attrs, re-delivering coverage for every zone, and plumbing it to the query.

Block boxes remain worth doing as a **separate** change: they would cut items genuinely
fetched and then discarded, which date slicing does not. That is an efficiency change with an
item-set consequence, and it should be priced and gated on its own rather than folded into a
defect fix.

## 8. Numbers of record

| quantity | value | provenance |
|---|---|---|
| tasks per (block × band) / per output block | ~4 / ~44 | measured census; culled-graph predictions within 0.5% of observed |
| gate graph | 4,834 tasks (3,876 scl + 956 open-scl, one per item) | measured census |
| `R`, scheduler dispatch rate | ~600–800 tasks/s | derived across three window configurations |
| `F`, per-window blocking cost | 7.04 s | fitted by differencing two runs (constants cancel) |
| scheduler envelope | one core of 4 saturated, never >100%; RSS peak 1.79 / 8 GiB | 114 heartbeats |
| per-date client graph build | 7.2 s (3.47 gate + 3.74 bands), one now removed | measured |
| STAC query | 5.58 ms/item; 164 s/month; ~34 min/year; 368,248 items/year | measured, linear 3→14 days |
| STAC page size ceiling | 250 (500 and 1000 both fail) | measured |
| Earth Search page-1 refusal | 502 at `limit=250`; 200 at `limit=100`, same window | measured (§7c) |
| Earth Search page-request refusal | deterministic in the **(cursor, window) pair** — the byte-identical cursor is refused with a 2-day window and served with a 1-day one, 3 of 3 each way. Seen at page **289** of a 566-page window and page **14** of a 6-day one. NOT a depth limit and NOT a bad cursor — flat 0.6-1.6 s/page throughout, 502 returned in 1.3 s | measured, 6/6, 4/4 and 3/3 (§7c) |
| what clears it | a window with a different END date; shortening the START does not | 8 windows tabulated (§7c) |
| per-search item ceiling | **10,000** items (`_MAX_QUERY_ITEMS`), from `numberMatched` on the first page | shipped; bounds cost, not correctness (§7c) |
| re-partition item-set check | union of parts = **56,558 of 56,558** on the failing query | live catalogue, end to end (§7c) |
| re-partition cost | **19 searches, 821 page requests** vs 566 for the single walk that cannot complete (**1.45x**) | same run (§7c) |
| 502 in the retry ladder | **364 s of backoff per refused page request** (`total=8`, `backoff_factor=2`, urllib3 `backoff_max=120`), bought nothing — the refusal is deterministic. Removed from `_STAC_RETRY`; `_CMR_RETRY` keeps it | arithmetic + measured, 24 retry lines to 0 (§7c) |
| benchmark query wall clock | **2,360.7 -> 1,099.6 -> 795.2 s** (39.3 -> 18.3 -> 13.3 min) for an identical item set, from a laptop. **41.7 -> 19.4 -> 14.1 ms/item** against a 5.58 ms/item reference rate | three live runs, 2026-08-22 (§7c) |
| re-partition request overhead | **821 -> 583** requests against 566 for a single unsplit walk: **1.45x -> 1.03x** | same runs (§7c) |
| page payload | **4.5 MB per 100 items** (45 KB/item); the benchmark query moves ~2.6 GB. `limit` above 100 is refused 502 on page 1, and server-side `fields` selection saves 4% because assets are not selectable | measured (§7c) |
| in-region projection | 583 requests at the reference rate is **327 s vs a single walk's 316 s**. A PROJECTION from §8's rate, never measured on a worker | open (§7c) |
| item ORDER under re-partition | walk order **differs** from an unsplit walk (from index 100); post-sort order and baselines **identical** at part counts 2, 3, 4 | measured on a 5,034-item window (§7c) |
| worker RSS per retained item | ~80 KB → ~27–30 GB/year vs 16 GiB worker | 3-point slope |
| unsharded manifest cost | ~1.1 MB refs/date → ~35 GB per zone-year | fitted, N²/2 |
| time-sharded manifest cost | ~1.2 GB per zone-year (~28× less) | predicted 34.7 MB at 9 dates, measured 34.6 |
| per-date variance | 19% SD (n=12) | measured |
| load-block census (fixed window) | 4096: 4,085 · 8192: 2,513 · 16384: 1,856 · 8192 no-align: 1,061 | measured |
| memory per band-block | 4096: 34 MB · 8192: 134 MB · 16384: 537 MB | arithmetic |
| hottest worker | 10.16 GiB of 16 at 8192 blocks; spill 0 throughout | run telemetry |
| inference baseline (do not regress) | GPU util 99% in-phase, VRAM 97% peak, host RAM 46% of a 60% ceiling, GPU-idle ~6 s/chunk | 2,352 RESOURCES samples |
| **ingest worker size (CURRENT)** | 4 vCPU / **24576 MiB** | Superseding, in order: 16384 → 30720 → 20480 → 24576 → 16384 → **24576**. **16384 is WITHDRAWN**, and with it §4.8's claim that pruning made the original size affordable: its 7.91 GiB peak over 91 dates was measured on 2017–18 optical at ~8 windows per date, and 2019 carries 14–16. The overlapped write (§3.11) holds a date's windows concurrently, so demand scales with that count — six workers on 2019 optical paused at **11.92 GiB**, 80% of the 14.90 GiB limit this size gives Dask, and never resumed, stalling four zone-years. 20480 was not chosen instead because §3.13 already records it as short: it was itself sized against a ~12.4 GiB short-run peak against a true ceiling of ~15. Whether 24576 holds at 60 concurrent cells on 2019-density data is **UNMEASURED**. With Dask task definitions PINNED the constant sets only Dask's `--memory-limit`, so the consumer's registered definition must be raised to match or the pause threshold lands above the container's hard limit. vCPU stays at 4 — the quota counts vCPU. |
| streaming retention cost | +1 month of items on the ingest worker; 1.25 GiB spill at 16 GiB | run telemetry |
| items deferred across a month boundary | 1,084 of 31,507 (one day's worth) | live cluster |
| write floor | graph ≈ store_chunks × bands; 2,992 covered vs 2,415 live (19% dead) | measured, all 7 windows |
| **per-date data share** | **582 of 2,992 covered chunks per band-date = 19.4%** at the 4096 grid; **47–57% of live CELLS** at the 8192 window grid | store object count + local footprint measurement (§3.9) |
| **dead-area split** | geometric **55–66%** of live, radiometric (cloud) **10–21%**, written 24% | 3 dates, local (§3.9) |
| **per-date time budget** | task work **57.2 s** at perfect packing of a **131 s** date; residual **73.8 s (56%)** | Dask performance report, 2-date probe (§3.10) |
| **where the residual was** | `write_day_windows` computing windows serially: per-window times summed to **146.9 s of a 148.0 s write phase**; commit only **0.6 s** | 7 dates, per-window instrumentation (§3.11) |
| **window overlap** | write phase **148.0 → 86.5 s (1.71×)**; per-date **166.1 → 104.2 s (1.59×)**; chunk objects **44,797 both arms** | A/B, identical dates and width (§3.11) |
| overlap memory cost | peak worker **12.30 of 20 GiB**, spill 0.00 GiB, no pause | parallel arm health lines (§3.11) |
| retained item size | hydrated **86.8 KiB**, deny-list pruned **35.4 KiB (2.45×)**; ~3.3 GiB off the driver at 68,000 items | shipped code, streaming, fresh process per form (§3.13) |
| item build cost | full **810 µs/item**, pruned **252 µs/item (3.2× cheaper)**; prune itself 8.7 µs | same (§3.13) |
| per-worker memory ceiling | **13.45 GiB**, flat over six consecutive blocks; **91-100% UNMANAGED**, so unspillable | 53-date soak health lines (§3.13) |
| driver excess over fleet mean | **~4.9 GiB** (the retained months); paused at 14.89 against a 14.90 threshold | same (§3.13) |
| ~~overlap packing effect~~ | **SUPERSEDED — the stream was truncated, see §3.14.** Recorded as 20% → 38% packing and a 1.26× → 1.60× ceiling; corrected to **64.3 s packed of a 102 s date (~64%)** and a **~2.8×** one-cell ceiling | §3.14 |
| **task work at perfect packing** | **64.3 s/date** at 120w; residual **~37 s (~36%)** | corrected task streams, cross-checked against the paired width fit (§3.14) |
| **ceiling, unlimited workers, ONE cell** | **~2.8× (2.0–3.0)** — not 1.78×, not 1.60× | §3.14 |
| **width model** | `T(W) = 36.3 + 7896/W` s per date, fitted from paired 60w/120w | §3.14 |
| **optimal cell width** | **flat within ~6% from 20w to 120w** — no meaningful optimum, so prefer whatever is simplest to operate. The earlier "30–45 workers, ~20% better" came from a **two-point** fit that cannot constrain two parameters; a third control at 45w put `F` anywhere from 11.4 to 39.3. **Withdrawn.** 120w was always the quota ceiling, never a choice | three-point fit `T = 18.0 + 9391/W` (§3.14) |
| **cell concurrency cost (CURRENT)** | **none measurable to 20 concurrent cells.** Per-window, paired by zone: 5→10 cells **1.01×**; 10→20 cells **1.24×** against a **1.33×** width-only prediction. The earlier **1.04× per cell** — one two-cell measurement, implying 2.56× at 40 cells — is **WITHDRAWN**. Schedule is set by the quota, not contention | 10 paired zones at N=20, 60 fleets (§1) |
| task-stream cap | **100,000 records** default ≈ 4 dates; `diagnostic_task_stream` raises it to 3,000,000 | §3.14 |
| Fargate launch rate | **20 tasks/s sustained, 100 burst** — a 10,000 vCPU fleet is ~2,300 tasks | measured quota |
| overlap contention cost | **+15% total slot-seconds** for identical output (2% fewer tasks, +25% transfer) | per-arm task streams (§3.12) |
| post-overlap residual, unexplained portion | of 65.1 s, **~18.6 s named** (build 10.4, gate 7.3, commit 0.9); **~46 s idles INSIDE the single write compute** | §3.12 |
| commit cost over a long run | **sawtooth, period 8 dates** (= `INGEST_MANIFEST_SPLIT["time"]`), 0.5 → ~1.5 s then resets — BOUNDED, not cumulative | 17+ dates, soak (§3.11) |
| **task work composition** | source read+resample **72.3%**, mask+write 16.3%, transfer 7.8%, gate 2.9% | same report (§3.10) |
| ~~ceiling, one cell — earlier values~~ | **SUPERSEDED by the ~2.8× row above.** 1.78× (pre-overlap 2-date probe), 1.26× (pre-overlap), 1.60× (post-overlap) were all measured off a TRUNCATED task stream or a stale probe. The programme verdict they supported — widen cells, no; multiply cells, yes — survives unchanged | §3.12, §3.14 |
| window-strategy bound | best any rectangle strategy achieves is **0.50×** current area; shipped achieves 0.75× | local, real footprints (§4.7) |
| external catalog latency drift | identical query **37.6 s vs 33.1 s** two hours apart (~12%) | §5 |
| worker-count scaling, dense zone | median s/date **194.8 at 120w, 232.4 at 60w, 396.7 at 30w**; doubling buys 1.71× at 30→60 and 1.19× at 60→120 | 7 dates per rung |

---

## 9. Instrumentation — use what exists

All installed as console scripts; do not rebuild these.

- `te-watch-scheduler` — live scheduler heartbeat watch with spill/lag alerting.
- `te-ingest-log-queries`, `te-ingest-report` — CloudWatch queries and reports for ingest.
- `te-observe-cluster --ram-report` — post-hoc per-worker peak RAM and GPU-util distribution
  from CloudWatch, no live cluster needed. `--start-pollers` then `--report` gives 1 s DCGM
  saturation (SMACT/TENSO) and the per-chunk phase-split table from the actors'
  `CHUNK_SUMMARY` lines.
- `te-compare-outputs` — the numerical parity gate for any change that could alter values.
- Per-run provenance belongs in `context_docs/design/inference-perf-run-ledger.md`.

The scheduler heartbeat self-reports every 30 s: CPU, RSS, memory percentage, queue lag, worker
and task counts, tasks processing, and fleet memory including spill and the hottest worker.
Read it on every rung rather than waiting to be asked.

For graph work, the cheapest and sharpest instrument is local: build one date's graph, call
`dask.optimize`, and count `__dask_graph__()` keys by prefix. No cluster, no spend, exact
answer.

---

## 10. Reading and maintaining this document

### How to compare the numbers

- **Wall clock.** Unless stated otherwise, every timing is zone `35N`, January 2024, 120 workers,
  one frozen ROI mask. Figures from other zones, widths or dates say so, and cross-zone timings do
  **not** compare — a date's cost tracks its rectangle count, which varies several-fold between
  zones.
- **Worker memory is not constant across rows**: 16 GiB up to 2026-07-25, 20 GiB after, back to
  16 GiB from 2026-07-27 (§4.8), and **24 GiB from 2026-08-21** (§8). Memory size barely moves wall
  clock, so the timing rows still compare — but a peak-memory figure only means something beside
  the limit it was measured against.
- **Graph-task counts** come from local census runs over a fixed pixel window. They compare within
  a series, not across series, and are labelled accordingly.
- **Two independent instruments** appear throughout: wall clock from logs, and packed task work
  from Dask performance reports. Where they agree the result is solid; where they disagree, the
  report has usually been truncated (§3.14).

### The rules this record is kept by

**Corrections go in place, with the superseded claim named** (§5 collects them), never as an
appended log. A reader must be able to trust any section without checking whether a later one
quietly undoes it — that property is the document's whole value, and it has been violated twice by
leaving an old number in one section while correcting it in another.

**Depth increases through the document.** §1 is the actionable summary and should stay readable by
someone who has never touched the ingest; the mechanism belongs in §2, and the full numbers in §3,
§4 and §8. Resist moving detail forward — a header dense with caveats stops people reading §1 at
all.

**Failed attempts stay, with their numbers** (§4). They are the cheapest thing here: each one is a
path a future reader would otherwise re-walk. The same goes for withdrawn claims — §5 exists so
that a plausible idea already measured and killed is not revived a third time.

**New measurements belong in §8**, one row each, with what they supersede named in the row.
