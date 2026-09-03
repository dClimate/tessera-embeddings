# Ingest: what it costs, what made it faster, and what limits it now

**The authoritative record of the campaign-ingest optimization work** — what was changed, how much
each change bought, what was tried and abandoned, and the constraints future work must respect. The
empirical basis for why the ingest path looks the way it does.

Four measurement campaigns are folded together here. The July 2026 optimization work is the spine
(§1–§10); the fleet-scale throughput investigation that corrected its duration basis is §11; the
graph and catalogue budgets that limit it now are §12; the live-tile cropping derivation that §3.1
and §3.2 summarise is §13; and the region-write primitive the whole windowed design rests on is §14.

> **The duration basis was re-measured, and the correction matters more than it looks.** A 2026-08
> reading found every zone running 1.8–2.1× slower than the figures here. **That claim is
> withdrawn** — it compared summer dates against a January baseline, and matched on all three
> conditions the gap is 1.17× (§11). What survives is that a zone-year is not twelve Januaries: a
> seasonally weighted year lands materially above the January-rate basis, so **treat every per-date
> and per-zone duration below as a January figure** and read §11 before quoting one. The width
> conclusions are unaffected — 6× workers buys 3.7–4.9×, confirmed by same-zone pairs.

Companion to [`../inference/inference-on-gpus.md`](../inference/inference-on-gpus.md), which is the
same kind of record for the GPU side. Read that one before touching anything the fill reads —
several decisions here were made *because* of what it measured. Source-read *failures*, as opposed
to source-read cost, are [`source-read-failures.md`](source-read-failures.md). See also
[ADR-011](../decisions/011-campaign-zone-ingestion.md) on campaign zone ingestion.

Unless a figure says otherwise, every wall-clock number here is the **same cell** — zone `35N`,
January 2024, on a 120-worker Dask-on-Fargate fleet against one frozen ROI mask — so the rows
compare directly. §10 records the finer measurement caveats and how to keep this document.

**Start with §1.** It is the whole story at a level you can act on. Detail deepens as the document
goes: §2 gives the cost model, §3 and §4 are the per-change archive, §8 holds the numbers of record,
and §11–§14 are the four detailed investigations.

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
windows intersecting the ROI mask. Full derivation and the per-strategy measurement: §13.

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
**35% efficiency**: adding workers could never help while the fleet was idle between windows. The
grouping algorithm, the cap sweep behind it and the per-zone effect are in §13.

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
> a *fleet-width-independent* floor (`~44 × blocks / R`, §12), and at the campaign's chosen 40–60w
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
[`source-read-failures.md`](source-read-failures.md), which is where that
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
holding by construction (`ingest-performance.md`).

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

## 7c. The Earth Search 502: one mechanism, a response-size cap of about 6 MB

**Earth Search refuses any request whose response would exceed roughly 6 MB.** That is AWS
Lambda's limit on a single synchronous response, and the search API sits behind one. Every 502
this campaign has seen from that provider is this, and nothing else.

The reason it did not look like one thing is that **items are not all the same size**. A page of
100 `sentinel-2-l2a` items averages 4.63 MB, but ranges 4.17 to 5.32 MB in practice, and
sometimes over the line. Whether a given hundred clears the cap therefore depends on WHICH
hundred, and that is selected by the pagination cursor and the date window together — which is
exactly the "deterministic in the (cursor, window) pair" behaviour §7c goes on to measure, and
why the same refusal appeared at page 289 of one window and page 14 of another.

### In plain terms, before any of the detail

Four words carry all the jargon in this section, so they are worth spending a paragraph on.

A **page** is one request and the results it returns. The catalogue will not hand over 56,000
results at once; it gives a hundred at a time. Nothing more exotic than that.

A **cursor** is a bookmark. When the catalogue hands back a page it also hands back a slip of paper
meaning "you got as far as here", and the next request has to include that slip. The important part
is what it is *not*: it is not a page number. There is no way to ask for the twentieth page directly,
and if a request fails there is no slip for the one after it, so the walk cannot step over a gap. In
this catalogue the slip is literally a timestamp and a scene name, for example
`2019-03-22T08:30:27.109000Z,S2A_37TEM_20190322_0_L2A,sentinel-2-l2a`.

A **date window** is a from-date and a to-date: everything between the 16th and the 22nd of March.

A **worklist** is a to-do list of jobs, each job being one date window. One request used to be
followed to its end. Now there is a list, and when a job turns out to be impossible it is crossed off
and two shorter jobs are written in its place. The list is what lets one date window be given up
without losing the others.

```text
WHAT WE ASK FOR
   Every Sentinel-2 scene over the UTM zone 34N envelope, 28 Feb to 1 Apr 2019.
   The catalogue answers: I have 56,558 of those.
   At a hundred per request, that is 566 requests.


THE OLD WAY -- one long chain, where the bookmark is the only way forward

  request 1 --slip--> request 2 --slip--> ... --> 288 --slip--> 289
   100 items          100 items                  100 items      REFUSED
                                                                   |
                        Each slip comes from the request before it, so when one request
                        is refused there is no slip for the next one and no page number
                        to skip to. Everything already collected is thrown away, and the
                        leg fails.


WHAT WAS ACTUALLY HAPPENING

  The catalogue refuses one specific request out of the 566, every single time, and answers
  all the others. It is not tired and it is not overloaded -- the refusal comes back in
  1.3 seconds, as fast as a successful request.

  It is not the bookmark's fault either. Take the exact bookmark it refused and hand it back
  with a SHORTER date window, or with a smaller number of results asked for:

     bookmark X + "21-22 March" + 100 results  ->  refused, 3 times out of 3
     bookmark X + "22 March"    + 100 results  ->  answered, 3 times out of 3
     bookmark X + "21-22 March" +  90 results  ->  answered, 5.60 MB of data

  The same bookmark, character for character, in all three. What the service will not serve
  is an ANSWER THAT IS TOO BIG -- about 6 MB is the ceiling. A hundred of those particular
  scenes would have been 6.2 MB; ninety of them are 5.6 MB and fit. Every request that
  succeeded in this walk carried between 4.2 and 5.3 MB, so the margin is thin and which
  hundred scenes you happen to land on decides it.


WHAT WE TRIED, AND WHAT EACH ONE ACTUALLY DID -- all measured

  ask again, and again     the same request is refused every time; the retry machinery
                           underneath spent six minutes doing this before giving up
  ask for 200 at a time    refused outright -- a bigger response is exactly the problem
  ask for 50 at a time     WORKS. The same bookmark, the same dates, half the results: served.
                           This is the tell -- the service is refusing on the SIZE of the
                           answer, not on where we are in the walk
  wait longer              nothing was timing out; every request took about 1.3 s
  ask for fewer dates      WORKS -- regrouping the results makes each answer smaller


THE PART TO WORRY ABOUT

  A hundred scenes is 4.6 MB on average against a 6 MB ceiling. That is about 30% of room,
  and it is all that stands between us and a refusal nothing in our code can route around.

  Shortening the dates works because it regroups the scenes. But the FIRST request of any
  window asks for the same first hundred however short the window is -- so if a hundred
  scenes ever averages over 6 MB, first requests start failing and there is no shorter
  window to fall back on. Anything that fattens a scene closes the gap: more files attached
  on a newer processing baseline, or a change at the provider.

  If first requests ever start failing, ask for fewer results at a time. That is the lever.


THE NEW WAY -- a to-do list of shorter date windows

  28 Feb ------------------------------------------------------------------- 1 Apr
      |  the catalogue reports the total alongside the first page, so the window is cut
      |  before the walk starts rather than 289 requests in
      v
   +--------+--------+--------+--------+--------+--------+
   |  job 1 |  job 2 |  job 3 |  job 4 |  job 5 |  job 6 |
   +--------+--------+--------+--------+--------+--------+
       ok       ok       ok    refused     ok       ok
                                  |
                                  |  cross it off, write two shorter jobs. The other five
                                  |  are untouched -- and they are still running while
                                  v  this happens.
                            16-22 March
                              +-- 16-19 March   ok
                              +-- 19-22 March   refused
                                    +-- 19-21 March   ok
                                    +-- 21-22 March   refused
                                          +-- 21 March   ok   <- a single day is as far
                                          +-- 22 March   ok      as this can go

  Two things had to be true for this to be safe.

  The jobs must add up to exactly what was asked for -- no day missed, no day added. They
  meet on a shared instant, so there is no crack between them and nothing outside them.

  The results must come back in the same ORDER as before, not merely the same set. Two
  scenes photographed on the same day with the same cloud cover are separated only by which
  one arrived first, and that decides which one is painted on top in the final image.


AND THEN IT WAS MADE FAST

  Almost all of a request is spent waiting for the catalogue to think -- 86% of it, before a
  single byte comes back. Moving our own machine closer to theirs would save almost nothing.
  But the six jobs do not depend on each other, so they can wait in parallel.

  the same query, the same 56,558 scenes, the same order, measured four times:

     as first written                    39 minutes
     stop re-asking a refused request    18 minutes
     stop re-fetching the shared day     13 minutes
     run six jobs at once                3.5 minutes
```

Everything below is the evidence for those claims, in increasing detail.

### The measurement that settles it

Walk the two-day window to its refusal at page 14, then re-send **the byte-identical cursor with
the identical date window** and change only the page size:

| `limit` | result | bytes returned | implied at 100 items |
|---|---|---|---|
| 100 | **502** | — | — |
| 90 | **200** | 5.60 MB | 6.22 MB |
| 75 | **200** | 4.58 MB | 6.11 MB |
| 50 | **200** | 3.33 MB | 6.67 MB |
| 25 | **200** | 1.80 MB | 7.22 MB |
| 10 | **200** | 0.66 MB | 6.59 MB |

The thirteen pages that WERE served in that same walk carried 4.17 to 5.32 MB. The refused
page's hundred items would have been 6.1 to 7.2 MB. Ninety of the same items, from the same
cursor, in the same window, are 5.60 MB and are served.

Bisected from the other direction: 130 items are served at 5.99 MB and 150 refused; with
unneeded assets excluded server-side the per-item cost drops from 46.3 to 33.2 KB and 150 items
are served at 4.77 MB, while 200 at a predicted 6.64 MB are refused. So the cap sits just above
5.99 MB and tracks **bytes, not the `limit` value**.

Two corroborating details. The refusal returns in **1.3 s**, as fast as a success, because the
service assembles the answer, measures it and gives up — nothing times out. And refusal latency
**rises with the requested limit** (1.24 s at 150, 3.55 s at 400), which is what "assemble, then
reject on size" looks like from outside.

### Why this is a standing risk, not just an explanation

**CORRECTED 2026-08-22:** this section previously said "largest observed 5.32 MB — roughly 30%
headroom". That 5.32 MB was the largest page in one two-day window, quoted as though it were the
ceiling. Measured across the month, the largest page actually **served** is **5.73 MB, 96% of the
cap**, and the refused page reconstructs to 5.96 MB — so the served/refused boundary sits between
5.73 and 5.96 MB and the real headroom at `limit=100` is about **4%**, not 30%.

A 100-item page averages 4.6 MB against a ~6 MB cap. Anything that makes items fatter closes it: more assets on a newer processing
baseline, or a collection change at the provider. When it closes, **first pages** start refusing,
and a first-page refusal is the one case no date-window re-cut can route around — a shorter
window asks its first page exactly the same way. `max_page_size` is the lever, and this margin is
the first thing to measure if a first-page 502 ever reappears.

### What it means for the fix

**Two levers now ship, tried in that order.**

A shorter **window** is preferred, because its halves between them walk about as many pages as
the parent would have, while a smaller page re-walks the whole window at twice the requests.

A smaller **page** is the fallback, halving to a floor of `_MIN_PAGE_SIZE = 10`. It exists
because window shortening cannot reach two refusals, and **both of them failed a leg outright
before this**:

- A **first page**. A shorter window asks its first page identically, so shortening is not a
  remedy at all. This is the failure the ~6 MB margin will eventually produce, and it is the
  one that had no answer.
- A **single day**, which is the re-cut's own floor. A day refused past page 1 had nothing
  left to try.

The friction that made this look hard is real but narrower than it appeared: `pystac_client`
bakes the limit into a search, so it cannot resume from a cursor at a different size. Neither
of the two cases above needs to. A first page has no cursor, and a single day is cheap to
re-walk from the start.

**Verified live, on the refusal class it exists for.** Asking for 250 items is over the cap, so
it is refused on the first request. The recovery stepped 250 -> 125, hit a deeper refusal at
125 and shortened the window, then stepped both halves to 62, and returned every item:

| | requests | items | distinct | post-sort id hash | baselines |
|---|---|---|---|---|---|
| `limit=100`, no refusals | 29 | 2,512 | 2,512 | `ecf847e086277035` | 2 |
| `limit=250`, recovered | 59 | 2,512 | 2,512 | `ecf847e086277035` | 2 |

Twice the requests for a query that previously failed. Note which invariant is checked: the
**post-sort** sequence, not the walk order. The walk order does differ, because the recovery
shortened windows as well as shrinking pages — and as §7c records below, the walk order changes
under any re-partition while the sequence the painter consumes does not.

**Lowering `max_page_size`** remains the separate lever for making refusals rare rather than
handled. It costs proportionally more requests, which the concurrency above largely hides.

> **CORRECTED 2026-08-22.** This section previously described two unrelated Earth Search 502s —
> a first-page refusal at `limit=250` remedied by page size, and a deep-page refusal at
> `limit=100` which it said "neither page size nor patience touches". Both readings were wrong:
> they are one refusal, and a smaller page from the refused cursor IS served. Sections below
> this one were written under the old reading; where they still assume it, they say so. The
> claim was also carried in `ingest/stac.py`, `config/providers.py` and the ingest README, and
> has been swept from all three.

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

**CONFIRMED, and it is simpler than a hypothesis about boundaries.** The reason the one-day
window serves that cursor is that it returns only 18 items there — the day holds 1,318, so the
cursor sits on its last, partial page — while the two-day window must return a full hundred. It
is the SIZE of the response, not the position. Re-sending the same cursor and the same two-day
window at `limit=90` is served, at 5.60 MB; the hundred would have been about 6.2 MB, over the
~6 MB cap. An earlier draft of this section guessed that the failing request was one whose page
crossed a date boundary; that was a coincidence of this example, and the correction at the head
of §7c has the mechanism.

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

Four runs of the benchmark query from one laptop, every one returning the same item set
(56,558 of 56,558, zero duplicates) and the last two returning the same ORDERED sequence
(SHA-256 of the comma-joined ids, `efcfdf7b6d69974c`):

| | wall clock | HTTP requests | ms/item |
|---|---|---|---|
| re-partition as first shipped | **2,360.7 s** (39.3 min) | 821 | 41.7 |
| + 502 out of the retry ladder | **1,099.6 s** (18.3 min) | 821 | 19.4 |
| + instant window seams | **795.2 s** (13.3 min) | 583 | 14.1 |
| + 6 windows walked at once | **207.6 s** (3.5 min) | 589 | **3.67** |
| a single unsplit walk, if it worked | — | 566 | — |
| §8 reference RATE, for scale | (316 s at this item count) | 566 | 5.58 |

**11.4x end to end, and the rate now BEATS the reference.** Request count is 589 against the
566 a single walk would make — 1.04x, the excess being one partial last page per window, a
page-1 refetch per cascaded cut, and one catalogue-root GET per worker thread.

**The 1,261 s the ladder cost.** `total=8, backoff_factor=2` against urllib3's default
`backoff_max=120` sleeps 0+4+8+16+32+64+120+120 = **364 s per refused page request**, three
times over. Predicted 1,092 s, measured 1,261 s; the rest is per-page variance.

**The 304 s the shared boundary day cost**, and a caveat about how it was recovered. The
instant-boundary runs make **zero** refusals — moving a window's end from `2019-03-22T23:59:59Z`
to `2019-03-22T00:00:00Z` changed its cursor sequence and missed the position that refuses. That
is consistent with the end-date property but it is **luck on this query, not a fix**: a different
window will land on a refusing cursor again, and the recursion is still what handles it. The
refusal path was therefore exercised separately, on the raw `2019-03-16/2019-03-22` window, which
refuses at page 14 three times over.

**The 588 s concurrency recovered, and why it was available.** A page request spends almost all
of its wall clock waiting on the catalogue to think:

| quantity | measured |
|---|---|
| TCP handshake to the CloudFront edge | 12-36 ms |
| TLS handshake | 25-43 ms, once per connection |
| **server think time** (`starttransfer − pretransfer`, reused connection) | **~1,150-1,240 ms** |
| body transfer | 402 ms uncompressed, **188 ms gzipped** |
| response body | 4.63 MB of JSON, **1.25 MB gzipped** (3.7x) |
| an empty-result search | 197 ms |

About **86%** of a page elapses before the first response byte. Think time scales with page
size at a fitted **7.75-7.79 ms per item** over a ~380-510 ms fixed cost, so at 100 items roughly
0.78 s of the 1.2 s is the catalogue's own per-item work.

**This retracts an earlier projection in this document.** It previously said the residual was
this laptop's bandwidth and that an in-region worker would close it, quoting 327 s. Both were
wrong. The round-trip an in-region client removes is a few percent of a page, so relocating the
client recovers **at most ~15%**, and the bytes were never the constraint either — production
negotiates gzip, so a page is 1.25 MB and the whole query moves about **740 MB**, not the 2.6 GB
an ungzipped probe suggested. Overlapping the walks is the only lever that moves this, which is
what `_QUERY_WINDOW_WORKERS` is.

**Concurrency is a threshold, not a curve, and 8 is the ceiling for this query.** At most eight
windows are ever runnable at once — four long level-one windows plus four level-two children —
and the floor is the longest single window's serial walk, about 93 pages or 128 s. Measured
speedups, drift-corrected against a ten-request canary run before and after each: 3.3x at four
workers, 3.8x at six, 6.0x at eight. Past eight there is nothing left to overlap.

**Six is shipped, not eight, because the campaign multiplies it.** Per-page latency does degrade
with width: median moved 1,272 -> 1,366 ms and p90 1,504 -> 1,840 ms (+22%) between serial and
eight-way, and total server-seconds for the identical 582 requests rose from 707.8 serial to
726.8 at four, 745.2 at six and 805.9 at eight — so eight-way costs Element 84 about 14% more of
their time for the same work, six-way about 5%. No 429, 503 or 502 was seen in over 3,500
requests across the matrix. The month streamer keeps a depth-one prefetch, so one query per cell
is in flight; at sixty cells this takes the peak from at most 60 concurrent page requests to at
most 360. Total requests and bytes are unchanged, and each query finishes about four times
sooner, so average concurrent streams rise only about 1.5x while the peak rises sixfold.

**What concurrency costs in memory.** Items are hydrated inside each window's walk, so a re-cut
window's already-walked pages are held as `Item` objects until the assembly step dedupes them.
On the whole query that is about 4%; on the refusal-heavy `2019-03-16/2019-03-22` sub-window,
where 39 of 141 pages are re-walks, peak resident memory went from 1,852 to 2,439 MB — **+32%**.
Deferring hydration to assembly was tried and REJECTED: it saves that memory but costs **208 s**
of the 207.6 s query, because hydration at ~1.1 ms/item overlaps network waiting when it happens
in the walk threads and is a bare serial tail when it does not. The cheaper fix, not taken here,
is that a re-cut window's own items are entirely redundant — its children re-fetch every one of
them — so such a node could discard them outright. That changes the walk order and so needs the
order check below re-run before it ships.

**One inefficiency remains and is NOT fixed.** The proactive cut CASCADES: `_parts_for_depth`
sizes parts as `ceil(matched / _MAX_QUERY_ITEMS)` and cuts by equal day count, so where density
is uneven a child can still exceed the ceiling and be cut again. Each cascade wastes one page-1
fetch, which is why it is worth only a few requests out of 589. Sizing parts by density would fix
it, but changing the part count moves the window boundaries and therefore the walk order.

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

### Why we ask for 100 scenes at a time and not fewer

The obvious safety move, once you know there is a 6 MB ceiling on any answer, is to ask for less
per request. We measured it and decided not to. This section is why, because the reasoning depends
on a fact about the archive that is worth knowing on its own.

**The 6 MB ceiling is closer than an average suggests.** A hundred scenes averages 4.6 MB, which
sounds like plenty of room. But the biggest single request we have seen ANSWERED carried 5.73 MB,
and the one that was refused works out to 5.96 MB. So the real gap between "fine" and "refused" is
about a quarter of a megabyte. An average is the wrong number to look at here; the biggest page is.

**Asking for fewer scenes does work.** On the window that reliably refuses:

| scenes per request | refused? | biggest request | worst case anywhere in the window |
|---|---|---|---|
| 100 | **yes, every time** | 5.73 MB (96% of the ceiling) | 6.20 MB (over it) |
| 90 | no | 5.11 MB (85%) | **5.66 MB (94%)** |
| 75 | no | 4.56 MB (76%) | 4.73 MB (79%) |
| 60 | no | 3.84 MB (64%) | 4.00 MB (67%) |

Read the last column, not the middle one. **Ninety scenes never got refused and is still 94% of the
ceiling** — it survived because of where the page boundaries happened to fall, not because it had
room. That is the difference between "we saw no failures" and "this is safe", and it is why 90 was
never a candidate.

**Why the scenes are fat, which turns out to be the whole story.** A scene's entry in the catalogue
includes the shape of the ground it covers. Usually that is a rectangle: four corners, a couple of
hundred bytes. But some entries carry the shape of where the imagery ACTUALLY is — tracing the edge
of the data instead of the tile boundary. Sentinel-2 builds each image from twelve separate
detectors, so that edge is a fine sawtooth, and drawing it takes thousands of points. One such
entry we measured runs to 2,497 points and 98 KB, against 0.2 KB for an ordinary one. The list of
files attached to the scene is about 18 KB either way, so it is the shape, not the imagery, that
makes these entries heavy.

Those entries are not scattered through the archive. They are confined to roughly **November 2018
to March 2019**, and the reason is a gap in reprocessing. Sampling one day a month across the
boundary:

| month | typical points per entry | heavy entries | processing versions available |
|---|---|---|---|
| Aug 2018 | 5 | 0 of 50 | 00.01, 02.08, 05.00 |
| Oct 2018 | 6 | 0 of 50 | 02.09 |
| **Nov 2018** | **580** | **31 of 50** | **02.11 only** |
| **Dec 2018** | **531** | **33 of 50** | **02.11 only** |
| **Feb 2019** | **579** | **30 of 50** | **02.11 only** |
| Apr 2019 | 6 | 0 of 50 | 02.11, 05.00 |
| May 2019 | 7 | 0 of 50 | 02.12, 05.00 |
| Jun 2023 | 5 | 0 of 50 | 05.09 |

The heavy months are exactly the ones where version 02.11 is the ONLY version on offer. Everywhere
else a later reprocessing exists alongside it and the catalogue serves that instead, with a simple
rectangle. So this is not something about that season, or about the satellite. It is a stretch of
the archive that was never reprocessed, and the original products happen to draw their outlines the
hard way. What we have not established is why version 02.11 in particular did that; the versions
either side of it did not, and that is an ESA processing decision we cannot see into.

Two consequences follow, and both point the same way. It cannot spread — no current processing
version produces these outlines, so no future data will bring the problem back. And it could vanish
on its own, if that stretch is ever reprocessed.

Outside the band a hundred scenes is about **2.2 MB**, roughly a third of the ceiling.

**So the problem is six months of a ten-year archive**, and that is what settles it. Asking for 75
scenes instead of 100 costs **32% more requests and 14% more time on every query in every year** —
about half a minute on a three-and-a-half-minute query — to protect a band that is a twentieth of
the timeline. Meanwhile the page-size fallback in this branch already handles a refusal by re-asking
that window at half the size, so the failure a shorter date window cannot fix now recovers by
itself, and it only pays where a refusal actually happens.

An adaptive remedy fits a concentrated problem. A global setting does not.

**What would change the decision.** If refusals inside the band turn out to be frequent rather than
occasional, the re-walks stop being cheaper than simply asking for less, and 75 becomes right. The
fallback names itself in the log every time it fires, so this is a count to read during the restart
rather than a guess to make now. Note also that editing `config/providers.py` at all moves the
mosaic fingerprint — measured, `ingcode-1739cd669dec92a2` to `ingcode-f75448a6d8841d0a` — so an
in-flight store cannot be appended to across the change.

### Levers that look obvious and are closed

Both measured 2026-08-22 against the live catalogue, so nobody has to test them again.

**A bigger page is refused, because the response cap is the whole mechanism** (see the
correction at the head of §7c — a SMALLER page from the refused cursor IS served, so page size
is a lever here, not a non-lever). Doubling the page to halve the request count is the obvious
win, and Earth Search will not serve it:

| `limit` | page 1 | median latency |
|---|---|---|
| 100 | **200**, 4,524 KB | 1.59 s |
| 150 | **502** | 1.24 s |
| 200 | **502** | 1.64 s |
| 250 | **502** | 2.22 s |
| 400 | **502** | 3.55 s |

The refusal latency RISES with the requested limit, which is what a response-size ceiling looks
like — the service assembles the answer, finds it too large, and fails.

**Bisected, it is a payload cap of roughly 6 MB, and `max_page_size=100` runs at 77% of it.**
Plain requests serve 130 items at 5.99 MB and refuse 150. With unneeded assets excluded
server-side the per-item cost drops to 33.2 KB and 150 items at 4.77 MB is *served*, while 200
items at a predicted 6.64 MB is refused. So the cap tracks bytes, not the `limit` value, and sits
between 5.99 and 6.64 MB — consistent with AWS Lambda's 6 MB synchronous response limit.

**That margin is a standing campaign risk, not just a curiosity.** A 100-item page of
`sentinel-2-l2a` is 4.63 MB against a ~6 MB cap. If items get fatter — more assets on a newer
baseline, a collection change at the provider — page one starts refusing, and that refusal is the
one no window re-cut can route around. The comment in `config/providers.py` is right about the
mechanism; this is the number, and it should be re-measured if a first-page 502 ever reappears.

**Server-side field selection is unusable, and the first reading of why was wrong.** Earth
Search advertises `https://api.stacspec.org/v1.0.0/item-search#fields`. Excluding assets **does**
work — it removes them and cuts the payload 1.40x, from 46.3 to 33.2 KB/item. What kills it is
that supplying ANY `fields` object collapses `properties` to `datetime` alone and drops
`stac_extensions`, so the pruned item is no longer equivalent: it loses `eo:cloud_cover`,
`proj:epsg` and the processing baseline, all of which the pipeline reads. The pruned dictionaries
were compared directly and they differ. (An earlier note here said `fields` did not reach assets
at all, from an include-list trial where the returned assets were unchanged. That was wrong.)

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
| `F`, per-window blocking cost | 7.04 s | fitted by differencing two runs (constants cancel). **Do not use the linear model it belongs to** — `cost = n_windows x F + area x V` failed its first out-of-sample test and is superseded by §12's dispatch-floor model. `F` survives inside that model as the per-window floor, binding only when a window is small |
| scheduler envelope | one core of 4 saturated, never >100%; RSS peak 1.79 / 8 GiB | 114 heartbeats |
| per-date client graph build | 7.2 s (3.47 gate + 3.74 bands), one now removed | measured |
| STAC query | 5.58 ms/item; 164 s/month; ~34 min/year; 368,248 items/year | measured, linear 3→14 days |
| STAC page size ceiling | 250 (500 and 1000 both fail) | measured |
| Earth Search page-1 refusal | 502 at `limit=250`; 200 at `limit=100`, same window | measured (§7c) |
| Earth Search 502, BOTH kinds | **one mechanism: a response-size cap of ~6 MB** (Lambda's synchronous limit). The byte-identical cursor and window is refused at `limit=100` and served at `limit=90`, returning 5.60 MB where the hundred would have been ~6.2 MB. Item sizes vary, so which hundred the cursor and window select decides it — which is why it looked deterministic in the (cursor, window) pair, and why it appeared at page **289** of one window and page **14** of another. NOT depth, NOT patience, NOT a bad cursor | measured (§7c); supersedes the earlier "not a page size either" reading |
| Earth Search page-size margin | largest page **served** is **5.73 MB — 96%** of the ~6 MB cap; the refused one reconstructs to 5.96 MB. Real headroom at `limit=100` is about **4%**. CORRECTS an earlier "5.32 MB, roughly 30%", which was one window's maximum quoted as the ceiling | measured (§7c) |
| why items are fat, and when | the FOOTPRINT SHAPE, not the assets. Some items trace the edge of the actual imagery — a twelve-detector sawtooth, up to 2,497 points and 98 KB, against 0.2 KB for a plain rectangle; the assets block is ~18 KB either way. Confined to roughly **Nov 2018 – Mar 2019**, which is the stretch where processing version **02.11 is the only one on offer**: elsewhere a later reprocessing exists and the catalogue serves its rectangle instead. So it is a reprocessing GAP, not a season. Cannot spread, and could vanish if that stretch is reprocessed. Outside the band a 100-item page is **2.2 MB (36%)** | measured, one day per month across the boundary (§7c). Why 02.11 in particular did this is NOT established |
| smaller page sizes, measured | `limit=90` makes zero refusals but its worst 90 consecutive items are **94%** of the cap — clean by luck. `limit=75` leaves **21%** margin against the fattest 75 anywhere in the worst week. Costs **+32% requests, +14% wall clock**. `limit=60` doubles the extra requests for 1.4 s | measured (§7c) |
| page size changes the WALK order | a window re-cut for depth appends its first page BEFORE deciding to re-cut, so each such parent hoists exactly `max_page_size` items to the head of the output — 300 items at 100, 225 at 75. Harmless ONLY because the sort key is total; it would have changed painted pixels before the `item.id` tie-breaker | measured, three hashes (§7c) |
| refusal levers, in cost order | **shorter window** first, then **smaller page** halving to `_MIN_PAGE_SIZE = 10`. The page lever covers the two refusals shortening cannot reach — a FIRST page, and a single day — both of which failed a leg outright before it | shipped (§7c) |
| page-lever live proof | `limit=250` is refused on the first request; recovery stepped 250 → 125 → 62, interleaved with a window cut, and returned 2,512 of 2,512 items with **identical post-sort order and baselines** (`ecf847e086277035`), for 59 requests against 29 | measured (§7c) |
| what clears it | a window with a different END date; shortening the START does not | 8 windows tabulated (§7c) |
| per-search item ceiling | **10,000** items (`_MAX_QUERY_ITEMS`), from `numberMatched` on the first page | shipped; bounds cost, not correctness (§7c) |
| re-partition item-set check | union of parts = **56,558 of 56,558** on the failing query | live catalogue, end to end (§7c) |
| re-partition cost | **11 searches, 589 page requests** vs 566 for the single walk that cannot complete (**1.04x**). Was 19 searches and 821 requests before instant seams | measured (§7c) |
| query window concurrency | `_QUERY_WINDOW_WORKERS` = **6**. Ceiling is 8 for this query's tree; floor is the longest single window, ~93 pages. Drift-corrected 3.3x at 4, 3.8x at 6, 6.0x at 8 | measured (§7c) |
| page latency breakdown | **86% is server think time** before the first byte; 7.75-7.79 ms/item slope on a ~380-510 ms fixed cost. Round-trip is 12-36 ms, so an in-region client recovers **at most ~15%** — this RETRACTS an earlier 327 s in-region projection | measured (§7c) |
| page payload | 4.63 MB of JSON, **1.25 MB gzipped**, which is what production sends. The whole query moves ~740 MB, not the 2.6 GB an ungzipped probe suggested | measured (§7c) |
| Earth Search response cap | roughly **6 MB**, bisected: 130 items served at 5.99 MB, 150 refused; with assets excluded 150 served at 4.77 MB. `max_page_size=100` runs at **77%** of the cap — a standing risk if items get fatter | measured (§7c) |
| 502 in the retry ladder | **364 s of backoff per refused page request** (`total=8`, `backoff_factor=2`, urllib3 `backoff_max=120`), bought nothing — the refusal is deterministic. Removed from `_STAC_RETRY`; `_CMR_RETRY` keeps it | arithmetic + measured, 24 retry lines to 0 (§7c) |
| benchmark query wall clock | **2,360.7 -> 1,099.6 -> 795.2 -> 207.6 s** for an identical item set, from a laptop: **41.7 -> 19.4 -> 14.1 -> 3.67 ms/item**, which BEATS the 5.58 ms/item reference rate. **11.4x** end to end | four live runs, 2026-08-22 (§7c) |
| serial vs concurrent item ORDER | byte-identical ordered id sequence, `efcfdf7b6d69974c` (SHA-256 of the comma-joined ids) | measured both ways (§7c) |
| item ORDER under re-partition | walk order **differs** from an unsplit walk (from index 100); post-sort order and baselines **identical** at part counts 2, 3, 4 | measured on a 5,034-item window (§7c) |
| concurrency memory cost | +4% peak RSS on the whole query, **+32%** on a refusal-heavy sub-window (1,852 -> 2,439 MB), because a re-cut window's items are held hydrated until assembly. Deferring hydration was tried and rejected: it costs 208 s | measured (§7c) |
| `fields` extension | unusable: any `fields` object collapses `properties` to `datetime` and drops `stac_extensions`, losing cloud cover, EPSG and baseline. Excluding assets DOES work and cuts payload 1.40x | measured (§7c) |
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
- Per-run provenance belongs in `context_docs/inference/inference-on-gpus.md`.

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

---

## 11. Throughput at fleet scale — what we know, and what we still need

**Status: resolved as posed.** The headline finding — a 1.8–2.1× slowdown against the July record —
**was an artefact of comparing summer dates against a January baseline, and is withdrawn.** Matched
on zone, width AND dates the gap is 1.12–1.17×. A paired A/B then measured the contention term at
**zero** (the quiet arm came out 3.3% dearer per window, not cheaper), so none of that residual is
load at the width tested.

**What is left.** Contention at 45–61 concurrent cells is unmeasured — the A/B ran at about a sixth
of campaign width — but the ladder that would measure it is on hold by decision (2026-08-05) and
blocked on the quota raise regardless. The live item is the campaign's duration basis, which needs
re-fitting on a **seasonally weighted** year rather than a January rate. That is the one correction
here which genuinely raises the ingest line.

> **Condensed from 431 lines.** The stage plan, the per-arm run narrative, the decision not to run a
> third arm, and this investigation's own list of self-corrections were the working of a question now
> answered; they are in git history. What is kept is what would otherwise be re-derived: the causes
> ruled out, the effects established, and the two methodological traps that produced the withdrawn
> claim.

### 11.1 What is ruled out

Six candidate explanations, each with the evidence that closed it. **None of these should be
re-investigated without new evidence** — that is the point of recording them.

| candidate | why it is out |
|---|---|
| **Width stops working at scale** | 6× workers buys 3.7–4.9×, against an Amdahl bound of 4.4–4.6× from the serial fraction at 10w. Width works about as well as arithmetic allows. The 2.7–4.0× that suggested otherwise was a **zone-mix artefact** — it compared different zones, and the two 10-worker zones were the two cheapest-per-chunk in the wave |
| **The orchestrator** | Direct refutation: over the window where per-date cost rose 32%, orchestrator CPU fell 13%→5.6%, requests 12.4→4.1/s, latency 146→39 ms, zero dropped events. Corroborated by the server gaining 8× capacity between waves while the *later* wave was slower |
| **The Dask schedulers** | CPU 10–40%, event-loop lag 0.0–0.03 s, graphs oversubscribing their fleets rather than starving them. Well under the ~250-worker threshold in `docs/dask-scheduler-plan.md` |
| **Capacity, quota, launch rate** | 20 live fleets, 519 workers, 2,236 vCPU of 10,000, and `no-worker=0` on every fleet — no task anywhere waiting for a worker, at 22% of quota |
| **Commit serialisation** | Commits are ~1 s; the per-date cadence leaves only 5–8 s unaccounted after build + gate + write + stall |
| **Store growth under accumulating dates** | Per-date cost rose 36% through 53N's run, but **write cost per window is flat at 16.4–18.3 s** across all four quartiles and build per window flat at ~1.2 s. Cost rose because windows per date rose 45% (7.3 → 10.6). Manifest sharding is doing its job |

> **A trap worth keeping.** ECS `describe-clusters` reported **763 pending against 67 running**
> during this investigation, which reads as catastrophic launch starvation. It is an artefact:
> cluster statistics are eventually-consistent and the account carried **204 registered ephemeral
> worker families** from a day of fleet churn. 760 genuinely pending tasks cannot coexist with seven
> fleets sitting at 56–60 workers and committing dates. **Use the schedulers' own `workers=` and
> `no-worker=` fields; they are ground truth for width.**

### 11.2 What is established

- **A per-date serial floor of 19–24% at 60 workers** (7–8% at 10w). Bounds the width benefit, and is
  worth more than per-zone width tuning: the unhidden preparation stall alone is 20–45 s median on
  writes of 160–300 s that ought to hide it.
- **Adaptive churn cost ~10–12% of effective width.** `adapt(minimum=1)` retired workers in every
  inter-date gap and relaunched them cold; one 60-slot fleet registered 1,250 distinct workers in
  5 h. **Fixed** — `min_workers` now follows each leg's derived width.
- **Per-date cost grows within a run, for a benign reason.** Later dates image more of a zone's land:
  northern-hemisphere January is snow- and cloud-limited, by May footprints are wider.
  **Consequence: every velocity figure measured early in a run UNDERSTATES the full-year cost.**
- **Width is nearly cost-neutral.** vCPU-seconds per date at 60w vs 10w averages ~1.1× over six
  same-zone pairs. Sixty workers costs about what ten does per date and finishes 4–5× sooner, so
  re-scaling `max_workers` is not where money is saved.
- **The per-tile cost gap is intrinsic, not width waste.** 53N costs ~$0.14/tile-year against 12N's
  ~$0.09, and that is 53N doing **1.38× the per-tile work** (9.7 against 7.0 worker-seconds per
  chunk-date, fewer tiles amortising the same per-date floor) — not an oversized fleet. 53N at 60
  workers is still ~80% write-bound, so it *can* use the fleet; narrowing sparse zones would recover
  single-digit percent of campaign ingest compute.

### 11.3 The withdrawn claim, and the mechanism behind it

**The claim:** every zone runs 1.8–2.1× slower than the July record at the same zone and width.
**It compared summer dates against a January baseline.**

| | zone | width | dates | windows/date | s/date |
|---|---|---|---|---:|---:|
| July record §3.10/§3.16 | 35N | 60w | **January 2024** | — | **167.9 / 175.6** |
| the withdrawn figure | 35N | 60w | **May–Sep 2021** (n=128) | 18.0 | **330.7** |
| matched re-measurement | 35N | 60w | **January 2024** (n=27) | 15.0 | **196.3** |

Matched on all three conditions the gap is **1.17×** — and the matched arm carried 17 concurrent
fleets while the July figure did not, so 1.17× is an upper bound on contention *plus* drift, not a
floor.

The mechanism is §11.2's within-run effect applied across seasons: 18.0 windows/date against 15.0 is
1.20× more work, and summer windows are individually dearer (write per window 16.7 s against 11.0 s).
Together those carry the 1.68× ratio with no appeal to a regression.

**This document's own reading instructions state the condition that was violated** — *"unless a
figure says otherwise, every timing is zone 35N, January 2024."* The claim honoured zone and width
and silently violated the third, which is the one that moves cost most.

### 11.4 What this changes: the duration basis, not the code

**The code has not regressed**, and the six ruled-out causes stay ruled out. But the July fit
(5.95 h/zone-year at 60w) is built from January-conditions measurements, and a zone-year is not
twelve Januaries. 35N at 60 workers, six of twelve months measured:

| month | windows/date | s/date | write/window |
|---|---:|---:|---:|
| Jan (2024) | 15.0 | **196.3** | 11.04 |
| Feb (2024) | 16.0 | **218.4** | 10.21 |
| Mar–Apr | — | ~250–290 *(interpolated)* | — |
| May (2021) | 18.0 | **330.5** | — |
| Jun (2021) | 18.0 | **381.6** | 18.71 |
| Jul (2021) | 18.0 | **317.6** | 16.11 |
| Aug (2021) | 18.0 | **309.6** | 14.15 |
| Sep (2021) | 18.0 | **320.6** | — |
| Oct–Dec | — | ~210–290 **(extrapolated, no data)** | — |

Two things a single multiplier hides. **Windows per date saturates at 18.0 by May and stays there** —
it plateaus once the zone is fully imaged rather than following a smooth sinusoid. And the summer
premium is mostly *not* window count: 1.20× on count against 1.68× on cost, the rest being that a
window with less cloud and snow carries more valid pixels.

Averaging gives **~280 s/date against the fit's 167.9, i.e. ~1.67×**. A quarter of that year is
extrapolated, so **treat the band and not the point.** The fix is a seasonal weighting of the basis,
not a hunt for a performance defect. Pin it with per-date covered chunks or one completed full-year
60-worker cell; the seven complete 2021 zone-years cannot serve, because they ran at 10 workers.

**The practical difference from the withdrawn claim is large: seasonality is predictable and
schedulable, so peak months can be planned around instead of hunted as a defect.**

### 11.5 The one concurrency signal left, and why it is weak

Two observations point in opposite directions: across waves, more cells looked slower; within a run,
fewer cells looked slower. The within-run one is largely the seasonal effect confounded with elapsed
time — the run progressed as the wave emptied — so **it is not a concurrency signal at all.** That
leaves the across-wave observation, and it is weak: three of its six zones showed no degradation, and
its two waves differ in server size, time of day and date range as well as cell count.

Four candidates remain and the data cannot separate them: source-read contention above ~20 cells
(July measured only to 20 and left large-count elasticity explicitly unmeasured); time-of-day load on
the public archive; the 2021 catalogue versus the 2024 dates every July figure used; and post-July
configuration drift — 35N now writes 18 windows/date against 13 in §3.15, which by the
windows-per-date arithmetic alone accounts for ~1.4× of the gap.

**Contention at campaign width was then measured directly, and is ABSENT** (55-cell rung on prod,
2026-08-06). 55 concurrent cells — 20,316 vCPU, i.e. actual campaign scale — paired against the same
zone and month at ~3,200 vCPU came out at **10.4 s/window against 13.1**. The claim this supports is
**no contention penalty at 55 cells**, NOT a 21% improvement: the two arms sit in different accounts
and per-window cost is not perfectly stable, so the sign of the residual is not interpretable — only
the absence of a penalty is. Everything around the cells held too: the orchestrator sat at 25% CPU
with zero dropped events; ECS placement was exact (5,086 tasks against ~5,115 implied); achieved
fleet widths held throughout; and `no-worker=0` on all 40 schedulers, every pass. **Nothing of ours
throttled**: every 503 in the window was upstream.

### 11.6 Two methodological rules this investigation earned

**Match on every condition that moves the number, and list them before comparing.** Three claims here
were withdrawn for the same reason: a real measurement compared against another real measurement
whose conditions differed in a dimension nobody had enumerated. Zone and width were checked; season
was not. §10's reading-instructions block names zone, width and dates as the three that must match —
consulting it would have caught the season error in one minute.

**Verify achieved width from the scheduler, not from the parameters requested.** The quiet arm of the
A/B ran on a different account — the loaded account could not host a quiet arm — so account, VPC, ECS
cluster, Prefect server and S3 bucket were all unmatched, and only a *small* difference would have
been interpretable. What made it interpretable at all was reading achieved width off the scheduler
health lines: median 57, max 60 workers with `no-worker=0` on all 48 samples, against 28
loaded-account fleets of which 14 sat at max 60. **Nominal width is a request, not a fact** — fleets
hold only 85–90% of nominal.

---

## 12. The graph budget and the serial phase — what limits ingest now

**Living reference.** Append findings with dates; correct superseded numbers in place and note the
correction.

**Why it needs a record:** graph task count has been the recurring failure mode of every iteration of
this stack, campaign and single-ROI alike. Each time it has been rediscovered from scratch. The point
of this section is that the next person inherits the budget rather than re-deriving it.

### 12.1 The budget equation

```
graph_tasks_per_date  ≈  4  ×  n_bands (11)  ×  area_in_chunks    ≈  44 × area_in_chunks
```

**Measured, not estimated.** One date's real graph was built for a zone geobox and classified by
task-key prefix. The submitted (culled) graph is ~4 tasks per (chunk × band): the `odc.stac.load`
fetch/mosaic, the ROI-mask `where`, a `getitem`, and the store write. Predictions match observed
cluster graphs almost exactly:

| window area | predicted (4 × 11 × area) | observed live graph |
|---|---|---|
| 972 chunks | 42,768 | 42,588 |
| 1,904 chunks | 83,776 | 84,054 |

**`odc.stac.load` emits ONE `open-<band>` task per source item, not per (item, chunk).** For a solar
day over a 6° zone that is ~956 open tasks against ~84,000 total — the per-chunk path is already lean
and there is no elementwise fat to fuse away. An earlier hypothesis that per-(chunk, item) overlap
tasks dominated was **wrong**; recorded here so it is not re-proposed. The coverage gate's SCL-only
graph is 4,834 tasks (3,876 `scl` + 956 `open-scl`). Cheap.

**The levers are therefore the multiplicands**, and only one has real room:

| lever | effect | cost |
|---|---|---|
| `area_in_chunks` via chunk size | ÷4 for 4096 → 8192 | store-side change reaches inference (§12.4) |
| the constant 4 → 3 (fold the mask into the load) | ~25% | second order |
| `n_bands` = 11 | fixed — contractual | — |
| window strategy | already at the frontier | spend nothing more |

### 12.2 The cost model: a dispatch floor, not a linear sum

A linear fit `cost = n_windows × F + area × V` (F = 7.04 s/window, V = 0.0346 s/chunk) was fitted
from two runs and **failed its first out-of-sample test**: it predicted 122 s/date at 3 windows and
the run measured 193.9 s. **This supersedes it, and supersedes the same fit wherever else it appears
in this document:**

```
per_date  ≈  C  +  Σ_windows  max( F , tasks_w / R )
```

- **R ≈ 600–800 tasks/s** — the single-threaded scheduler's dispatch rate.
- **F ≈ 7.04 s** — per-window blocking cost; binds only when a window is small enough that
  `tasks_w / R < F`.
- **C ≈ 15–20 s** — per-date constant on the ingest worker (client graph build + gate + commit).

Reconciliation against all three configurations of the same zone-month at 120 workers:

| windows | area | per-window tasks | regime | predicted | measured |
|---|---|---|---|---|---|
| 197 | 2,428 | ~540 | latency (F binds) | ~1,407 s | 1,329–1,471 s |
| 15 | 2,563 | ~7.5k | dispatch | ~180 s | 194–242 s |
| 3 | 2,924 | 0.7k/43k/84k | dispatch | ~199 s | 193.9 s |

**The 15↔3-window plateau is the dispatch floor.** `V` was never fleet work — it was dispatch. About
86% of a dense date is scheduler dispatch, ~4% is the single-threaded client build, and the fleet's
real fetch/resample/write work hides under both. The fleet-bound floor is still unknown and only
becomes visible once tasks shrink.

**Consequence:** two single threads (scheduler dispatch, the ingest worker) serialise ≥95% of a run,
so **widening one cell with more workers buys ≤1.1×**. Scaling comes from running more cells
concurrently — each cell has its own scheduler — not from wider cells.

### 12.3 Scheduler telemetry

114 heartbeats at 120 workers, 60 with a live graph: CPU 19–100%, **twelve samples at exactly 100 and
none above**. Ten threads, but never more than one core — the Dask event loop is single-threaded and
GIL-bound. RSS peak 1.79 GiB of an 8 GiB limit (22%), having grown from 0.33 GiB within one month.
`lag` ≤ 0.3 s throughout; worker spill 0 GiB; hottest worker 6.6 GiB of 16.

**Raising the scheduler's 4 vCPU cannot lift the ceiling.** Signs that WOULD justify a bigger box, so
this is not re-debated: any sustained sample above ~110% CPU (auxiliary threads genuinely using cores
— plausible at much larger fleets, unobserved at 120); RSS above ~50% of the limit or climbing across
a cell; `lag` rising while CPU is below 100% (contention, not compute — profile instead of resizing).
Memory is cheap insurance against a scheduler OOM killing a multi-hour cell; vCPU is not indicated.

### 12.4 Chunk size: measured variants, and why the store chunk cannot be coarsened

Task census over a FIXED pixel window (12 × 4 chunks at 4096, one date, 11 bands), graph construction
only, counting optimised graphs plus one write task per (store chunk, band):

| variant | read+mask tasks | write tasks | total | vs today | reaches inference? |
|---|---|---|---|---|---|
| load 4096 / store 4096 (today) | 2,673 | 528 | 3,201 | — | — |
| load 8192 / store 4096 | 1,749 | 528 | 2,277 | **1.41×** | **no** |
| load 8192 / store 8192 | 693 | 132 | 825 | **3.88×** | yes |

**Writes are emitted per STORE chunk, not per dask block** — the write count is unchanged at 528
across the first two variants. So coarsening only the load blocks buys 1.41×, and the full 4× requires
coarsening the store. A hypothesis that load-only coarsening might capture the whole win is therefore
**disproved**. Alignment is fine either way: `SHARD_PX` = 2048 divides 8192 exactly.

**Enlarging the store chunk would shrink the ingest graph, and the inference side vetoes it:**
inference reads whole chunks, so a coarser store chunk multiplies read volume for every consumer of
the store. That was measured rather than argued, and the measurement is what closed it. What shipped
instead was load blocks decoupled from store chunks — and that was itself later removed, for the
reason in §3.5.

**Baseline discipline for any inference-touching change:** GPU utilisation, VRAM peak, host RAM
against its ceiling, and GPU-idle per chunk, all captured before the change. A graph win that costs
read amplification is not a win, and only a paired baseline shows it.

### 12.5 The ingest worker's serial phase

**The STAC query runs once per run**, not per date, before any date is processed.

| quantity | value |
|---|---|
| rate | 5.58 ms/item; ~1.34 s per 250-item page |
| one month, one 6° zone | 164 s, 31,507 items, ~1,006 items/solar-day |
| linearity | confirmed 3 → 7 → 14 days |
| one calendar year, extrapolated | **368,248 items, 1,473 pages, ~34 min** |
| resident memory | **~80 KB per retained item** (from 484 / 809 / 1,382 MiB at 2.9k / 6.8k / 14.1k items) |
| a year, extrapolated | **~27–30 GB** |

**A year-long query as written exhausts its host before the first date processes.** This is a
feasibility blocker, not a speed problem, and it needs a longer *window* to observe — not a bigger
fleet. **Page size is capped at 250**: measured, `limit=500` and `limit=1000` both fail against
earth-search with repeated server errors while 250 succeeds. There is no free win there. (The
campaign runs at 100 for a different reason — see §7c.)

**Streaming, and why prefetch depth 1 is enough.** Stream month by month, prefetching month *m+1*'s
query on a thread while the cluster processes month *m*. That bounds retained items to one or two
months (~2.5–5 GB) and gives clean resume. Depth 1 suffices, and deeper prefetch is waste: a month's
*processing* is ~31 dates × ~194 s ≈ 100 min, while a month's *query* is 164 s — under 3% of it. Only
the FIRST month's query is exposed, and the lever for that is concurrent sub-range queries (weekly
threads → ~45 s), not more buffering. Shipped and validated; see §7b.

> **CORRECTION (2026-07-25): the memory that matters is a DASK WORKER's, not the flow runner's.**
> The whole ingest body — query, retained items, grouping, and the per-date loop — runs inside ONE
> Prefect task on a single Dask worker (verified: the "Cropping writes to N live window(s)" line
> appears in a `dask/dask-worker/...` log stream; corroborated by the orphaned fleet committing a
> further date seven minutes after the flow runner was killed). Workers are **4 vCPU / 16 GiB**. So a
> year-long query exhausts a WORKER, and Dask's response is to kill and restart it, which re-runs the
> task — a bounded retry loop rather than a clean failure. Raising flow-runner memory is irrelevant;
> raising worker memory multiplies across 120 workers. Streaming is close to the only fix.

**Per-date client-side cost.** Two `odc.stac.load` graph builds per date — 3.47 s for the gate's
SCL-only load and 3.74 s for the bands — ≈ **7.2 s per date**, single-threaded on the ingest worker
while all 480 fleet task slots idle. Over ~250 kept dates that is ~30 min per zone-year. **The gate
load is now fused into the band load** (§3.4), removing one of the two. What remains is fixable by
pipelining one date's graph build against the previous date's cluster work.

### 12.6 Results of record (2026-07-25 session)

Same zone (35N), month (January 2024), fleet (120 workers) and mask throughout, so the rows are
comparable. Per-date figures are means over the dates the run completed.

| configuration | windows | per-date | scheduler cpu mean/max | graph mean/max | hottest worker |
|---|---|---|---|---|---|
| row bands | 197 | 1471, 1329 s | — | 482 / 1,048 | — |
| grouped, greedy | 15 | 225 s (n=8) | 66% / 93% | 17,380 / 22,836 | 5.6 GiB |
| grouped, cost-model DP | 3 | 194 → 269 s, degrading | 65% / **100% pinned** | 50,875 / 84,054 | 7.2 GiB |
| **load blocks 8192, store 4096** | **7** | **187.9 s (n=12)** | **27% / 40%** | **5,069 / 6,020** | 8.0–10.3 GiB |

**Per-date variance is 19% (SD 36 s on n=12).** Detecting a 10% change needs ~14 dates per arm, so
**verify further graph work by TASK COUNT, not wall clock** — the count is deterministic and readable
from a single date's graph. Collapsing the two masking passes into one, counted on a real window's
optimised graph, gives 99.9 → **77.9** tasks per load block: **1.28× fewer graph tasks**, matching
the predicted ~25%. That is the pattern to follow for the remaining shaves.

Two claims made during that session and then withdrawn, recorded so they are not revived:

- *"1.41× from the load-block change"* — that was one cherry-picked date. On matched means it is
  **1.23×** (187.9 against the greedy configuration's 225 s).
- *"Per-date cost drifts upward within a run"* — visible in the first six dates (180.2 → 195.5) but
  **not established over twelve**: the later half contains a single 289 s outlier and two of the
  fastest dates in the run. (§11 later established the real mechanism, which is windows per date
  rising rather than a drift in per-window cost.)

### 12.7 Manifest sharding: the axis matters more than the size

Same zone, month and fleet; two dates each, so directly comparable.

| configuration | date 1 | date 2 | manifest objects | manifest bytes |
|---|---|---|---|---|
| no split | 179.5 s | 147.8 s | — | ~3.3 MB (fitted) |
| `{northing:4, easting:4, time:8}` | **245.8 s** | **281.3 s** | **10,195** | 3.85 MB |
| `{time: 8}` | 179.8 s | 146.8 s | **161** | 1.99 MB |

The spatial split cost **30–50% wall clock** — not from bytes but from object count: ~5,097 manifest
objects rewritten per commit against ~14, and the PUT latency dominates. Time-only restores no-split
speed while already writing fewer manifest bytes, and it grows LINEARLY with dates where unsharded
grows as N²/2.

**The rule, generalised:** split the axis along which a single commit is NARROW. Campaign ingest
commits one date across every live window, so it is narrow in time and wide in space — the exact
inverse of the region-write merge workload the module default was written for. Both regimes are
documented at `storage/zarr_store.py`'s split constants.

### 12.8 Task-count levers: what is left, and what is ruled out

Current: **4 tasks per (chunk × band)**, ~44 per output chunk, 77.9 per load block.

| lever | expected | LOC | effort | status |
|---|---|---|---|---|
| Load blocks 8192 → 16384 | area in blocks ÷ 4, so **~2–3×** on load-side terms | ~5 | 30 min | **next** — memory per task ×4 (134 → 536 MB per band); raising worker memory is sanctioned if the gain is real |
| Drop `align_chunks`, rely on windows already landing on load blocks | ~1 of 4 per chunk-band (**~25%**) | ~20 | 2–3 h | **after** — measure, do not assume: dropping it was 11% SLOWER when blocks and chunks matched |
| Coarsen `INGEST_CHUNKS["time"]` beyond 1 | none per date; fewer commits | ~10 | — | **REJECTED** — breaks the property that a date's time slot lands atomically with its pixels, which is what makes a crashed ingest safely retryable |
| Fold the ROI mask inside `odc.stac.load` | ~1 more per chunk-band | 80+ | 1–2 d | **REJECTED** — no clean hook; high risk for a second-order gain |

On worker memory for the 16384 experiment: the hottest worker already reached 10.3 GiB of 16 at 8192
blocks, so 16384 is expected to press it. Raising the worker size is an accepted trade if the
task-count gain is real, but the gain must be demonstrated by task count first, and spill must be
zero.

### 12.9 The optical query bbox asks for far more ground than the zone occupies — NOT yet fixed

The Sentinel-2 query area is `roi.bbox_wgs84`, the axis-aligned WGS84 envelope of a UTM zone raster. A
UTM zone is 6° of longitude, but meridians converge, so the envelope of a zone running from the
equator to ~84°N is **54.4° wide**. Most of what that box returns is discarded: measured on one
January window, the as-run 34.6° × 80.2° envelope walked 95 pages for 9,383 items, of which **81% lay
outside the zone's own longitude band**.

**The safe fix is latitude banding**, one box per band asking for the longitude the zone actually
occupies at that latitude. A fixed 6° box is NOT valid — the zone's raster genuinely widens toward the
pole, which is what inflates the envelope in the first place, so a fixed box silently drops polar
land.

Measured offline on the real zone specs, via `zone_grid.tile_range_bbox_wgs84`, as the ratio of the
single envelope's area to the summed area of N banded boxes:

| bands | 33N / 47N / 23N / 01N | 33S |
|---|---|---|
| 2 | 1.74× | 1.63× |
| 4 | 2.63× | 2.25× |
| 8 | **3.38×** | 2.68× |
| 16 | 3.81× | 2.90× |
| 32 | 3.91× | 2.94× |

It saturates by 16, so **8 bands captures most of the available saving** and is the sensible default.
This is a ground-area ratio, not an item-count claim: item density is not uniform, and the 2.5× page
saving measured on one January window had two polar bands empty from polar night — a summer window
would put data in those wide bands and save less.

**Safety verified, offline, at 8 bands on zones 33N, 01N, 33S, 47N.** 175,600 sampled points per zone
across the projected rectangle, with band seams and their immediate neighbourhoods sampled explicitly:
the count uncovered by the banded union equals the count uncovered by today's single envelope exactly
(12, 12, 20, 12 — the documented ~9.8 m densification residue at the top edge). **Banding therefore
introduces no additional loss.** Every band box is a subset of the envelope, no band is degenerate,
and the antimeridian zones are fine.

**The query layer needs almost nothing.** `stac._query_stac_items` already seeds a LIST of root boxes
and already dedupes by `item.id` across every root and node, first-occurrence-wins in tree order — put
there precisely because the antimeridian halves return overlapping items. Latitude bands overlap only
at their shared seam, which that dedupe absorbs for free.

**What blocks it is provenance, not plumbing.** The bands must come from the LIVE TILE range, the same
range that produced `bbox_wgs84`. `roi.geobox` cannot supply it: `land_mask.export_zone_roi` writes the
ROI at the **whole zone**'s shape, ocean left as fill, and crops only the `bbox_wgs84` attr. Banding a
geobox-derived rectangle would ask for a LARGER latitude range than today, i.e. a regression. So it
needs an ROI schema addition (`bbox_wgs84_bands` written beside `bbox_wgs84`, read with a fallback so
existing ROIs keep working), its validator, and the boxes threaded from the three `s2_roi.py` call
sites. Note also that `_roi_is_current` short-circuits the export when the coverage sha is unchanged,
so **already-exported ROIs would not gain the attr** without forcing a re-export.

Deliberately left out of the 403/stagger/polarisation change: it is efficiency rather than
survivability, it changes which items a query returns, and it touches an on-disk schema.

### 12.10 Open questions

Non-issue, recorded so it is not re-raised: **ingest-side commit contention across concurrent cells**,
because each `(zone, year)` writes its own repo under `…/mosaics/{zone}/{year}/`. The shared-repo
concern belongs to the global embeddings store —
[`../storage/writing-to-the-global-store.md`](../storage/writing-to-the-global-store.md).

1. **Does the 8192 store change hold the GPU duty cycle?** The decisive question for the 3.88× lever,
   and it is a duty-cycle question rather than a memory one (§12.4). Needs an end-to-end small-zone run
   measured against the 15S baseline: duty cycle 79.8%, infer-phase GPU utilisation 99%, load-phase
   host RAM 35%.
2. **Does the 1.41× load-only variant behave as the census predicts on a real run?** Task counts are a
   graph-construction prediction; confirm live. It cannot touch inference by construction.
3. **What is the fleet-bound floor?** Invisible while dispatch dominates; appears once tasks shrink.
4. **Commit time growth with manifest size** — the last unmeasured per-date term.
5. **Is `F` fleet-invariant?** Determines whether a cap tuned at 120 workers transfers.
6. **Do the scheduler's auxiliary threads use extra cores at larger fleets?** Re-check the "one core"
   finding before resizing at scale.

---

## 13. Cropping ingest to live tiles — the full derivation

**Status: shipped, unconditional, and no longer behind a flag** (amended 2026-07-29). It was
`IngestSettings.crop_to_live_windows`, defaulting off; the flag was removed outright once no scenario
wanted cropping disabled, and the one validation that depended on it (`batch_dates > 1 requires
crop_to_live_windows`) went with it, its precondition now holding by construction. Anywhere below that
names the flag is describing how a measurement was taken rather than a switch that still exists.

### 13.1 The problem

Campaign ingest mosaics a zone's **entire grid** regardless of how much of it is land, so its cost
scales with EXTENT, not content. The STAC *search* is already narrowed — `export_zone_roi` writes a
WGS84 bbox tight to the live tiles — but the raster computation is not: the ROI mask is deliberately
written on the fixed full zone grid, so `roi.geobox` is full-extent and both sensor paths hand that
straight to `odc.stac.load`.

Observed on the first real campaign cell (zone `03S`, 2024), chosen as the cheapest possible
end-to-end test because it is the smallest land zone in the scheme — 4 live 2048-px tiles:

| | |
| --- | --- |
| Dask tasks in one graph | 119,002 |
| Worker memory | 517 GiB resident + 468 GiB spilled |
| Worker churn | 62 registered / 44 removed |
| Scheduler | healthy throughout (0.94 GiB RSS, 0.0 s loop lag, no backlog) |

Spill drove memory pressure, workers were killed, completed tasks were lost and recomputed, and the
run would likely never have finished. **The scheduler was never implicated; the graph was.**

`03S` is 890,880 × 67,584 px, which at `INGEST_CHUNK_SIZE = 4096` is 218 × 17 = **3,706 chunks per
band, per date** — the exact per-band counts the dashboard showed. One uint16 chunk at 4096² is 34 MB,
so a single band-date materializes ~124 GB. **Empty ocean chunks still allocate and still run a task.**

### 13.2 The measurement that chose the strategy

A one-off script coarsened each zone's frozen `tile_live_2048` bitmap ([ADR-010](../decisions/010-landmask-registry-coverage.md))
onto the 4096-px ingest grid and costed three cropping strategies against the full-extent baseline,
reading a few KB per zone from the coverage repo — no cluster, no mosaic, no writes. It was **deleted
once the strategy was chosen** (`scripts/measure_live_chunk_cropping.py`, removed 2026-08-11): it
exists to pick between the three rows below, and the mask it reads is frozen for the campaign's
duration. Recover it from history if a future delivery changes the coverage enough to reopen the
choice.

All 112 land zones, chunks computed per band-date:

| strategy | chunks | vs today |
| --- | --- | --- |
| today (full extent) | 425,272 | — |
| `bbox` — one window enclosing all live chunks | 230,686 | 1.8× |
| `rows` — one window per chunk-row, spanning that row's live columns | 99,847 | **4.3×** |
| `exact` — the live chunks themselves (the floor) | 97,597 | 4.4× |

Two results decide the design.

**A single bounding box is not enough.** It gives only 1.8× campaign-wide. It is spectacular where
land is clustered (`03S`: 3,706 → 4) and worthless where a zone holds scattered islands across a long
north-south extent (`31S`: 3,706 → 1,152 by bbox, → 3 by rows) or where land is dense (`35N`: 3,876 →
3,723, a 4% saving).

**Row-bands capture essentially the whole win.** They land within 1.0% of the exact floor at the
median and 2.3% campaign-wide. A full rectangle decomposition is therefore not worth building: it
would recover ~2% for materially more complexity. Window counts stay modest — a median of 74 per zone
and a maximum of 220 — which matters because each window is one load and one write.

### 13.3 The design

**The mosaic's declared grid stays full-extent.** The zone fill validates the mosaic against the exact
`ZoneSpec` grid and that check is load-bearing. Zarr and Icechunk are sparse, so the grid can stay full
while only live windows are computed and written; unwritten chunks read back as fill. Zarr 3 elides
all-fill chunks on write by default, so this costs no storage either.

1. Derive live windows from the ROI mask, coarsened to the ingest chunk grid.
2. On the first passing date, seed the mosaic full-extent and all-fill with that date's slot
   (`create_empty_store` is schema-only — zero chunks, no graph — so creation cost is independent of
   spatial extent).
3. Per passing date, ONE session: extend the time axis by that date, then `to_icechunk(mode="r+",
   region=...)` each live window, then commit — one commit per date. Explicitly NOT per-window
   `zarr_store.write_region` (which commits per call), and NOT a full-extent all-fill append through
   dask (that rebuilds a ~44K-task graph of zero-work per date — extend the axis as metadata, not as
   data).

**Dates stay discovery-as-you-go**: the time axis only ever contains dates that passed the coverage
filter and were written. That single property is what keeps today's semantics intact for every reader
that treats the time axis as the record of what was ingested (`get_existing_dates` dedupe,
`check_time_window_coverage`, the empty-timestep prunes) — a pre-seeded daily axis would have broken
all three, which is why the bookkeeping `write_dataset` and its two callers owned mostly **dissolved
rather than moved**.

**Windows are snapped to the 4096 chunk grid deliberately.** That makes them chunk-disjoint, which
removes the shared-boundary-chunk reconciliation the region-merge tier has to handle, and it is what
makes the single-session write safe: two writes into the same chunk in one session are a lost update
(§14).

**Interoperability with single-ROI runs.** Windows are derived from **the ROI mask itself**, not from
the zone coverage bitmap. `export_zone_roi` already upsamples `tile_live_2048` onto the zone grid to
produce that mask, so reading the mask serves both callers: a campaign zone and a single-ROI run whose
mask came from `rasterize_roi_zarr`. A sparse single ROI — scattered fields, a coastline, any footprint
much smaller than its bounding box — gets the same reduction from the same code path, and the ingest
flows need no knowledge of whether they are in a campaign.

### 13.4 What the first cropped run found

**Crop the coverage denominator too.** Cropping the computation left the coverage GATE measuring
against the full extent, so a cropped run's coverage ratio collapsed against a denominator that no
longer described what it was computing. Fixed by cropping the denominator with it.

**The general shape is worth more than the instance**, and it recurs: an optimisation that narrows what
is COMPUTED silently changes every ratio whose denominator was the old extent. The same error appears
as §3.9's granularity artefact, and as the "presence counted where coverage was meant" family in the
corrections register. **When you crop a numerator, go and find its denominators.**

**The profiler was blind to spill.** The scheduler heartbeat recorded no worker-side memory at all, so
the residual had to be diagnosed from a screenshot of the Dask dashboard. The heartbeat now carries
fleet totals (`wmem`/`wmanaged`/`wspill`/`wmax`) summed from the per-worker state the scheduler already
tracks, and `te-watch-scheduler` alerts on `worker-spill`. This does not demote the scheduler: it
remains the named saturation risk at scale, and a clean fleet is the precondition for a
high-worker-count rung to measure a scheduler envelope rather than a doomed run.

**A hard-cancelled ingest leaks its Dask cluster** — 23 workers and a scheduler left running in ECS.
The original write-up attributed this to `skip_cleanup: True`, but that flag only disables
dask-cloudprovider's startup sweep for debris from *prior* runs, and turning it off breaks cluster
construction outright on AWS SSO because the sweep iterates IAM roles. The actual cause is that
`ecs_cluster` tears down in a `finally`, which a hard cancel skips, and the ingest flows register no
cancellation hook. The GPU flows already solved this: `_ray_lifecycle.ray_cleanup_on_cancellation`,
registered as both `on_cancellation` and `on_crashed`, re-derives the cluster name from the flow-run
id and terminates by tag. Ingest needs that same treatment for ECS.

**The 50-worker ceiling was hit for a 4-tile zone**, which is a symptom of the uncropped graph rather
than a limit to raise. Once cropping lands, worker counts should be sized from a cell's live-chunk
count. External services were not implicated — one store-write retry, no catalog throttling.

**Consequences for the test programme.** The ingest scaling ladder cannot measure a scheduler envelope
until it stops benchmarking ocean: as it stands the memory wall arrives long before any scheduler
limit, so it would tune the wrong bottleneck. Fargate worker-count and quota sizing derived from
full-extent mosaics likewise overstate what the campaign needs, by roughly the 4.3× above.

### 13.5 Grouping row bands: why window COUNT was the real limit (2026-07-25)

Cropping to row bands fixed the *area* problem and exposed a different one. Row bands minimise computed
area, and **area turned out not to be what a windowed ingest is billed for.**

Dense zone `35N`, January 2024, a 120-worker Fargate fleet (4 vCPU each, so 480 task slots), 197 live
windows:

| quantity | value |
|---|---|
| date `2024-01-01` | 1471.3 s |
| date `2024-01-02` | 1329.2 s |
| per window | ~7.1 s |
| sparse-zone comparison, far smaller fleet | ~6 s per window |
| mean tasks processing | 30.6 of 480 slots = **6%** |
| scheduler samples with an EMPTY graph | **16 of 44 (36%)**, twice for 2 min straight |
| worker spill | 0 GiB |
| scheduler cpu / rss / lag | 20–50% / 0.78 GiB / ~0 s |

Per-window cost is flat from a sparse zone on a small fleet to a dense zone on 120 workers. **That is
a fixed serial cost, not distributed work** — and it is the same phenomenon the earlier fleet-scaling
rung reported as "35% efficiency", seen from the other end. **More workers was never going to help.**

The cause is structural: `write_day_windows` loops `to_icechunk` once per window, each call blocking.
A `35N` row band is at most 17 chunks wide and only ~3 of those hold a given day's swath, so each graph
is a few dozen tasks against 480 slots, 197 times in series.

**Measured A/B outcome at 15 windows: 194.4 / 207.8 / 230.4 / 222.8 / 239.5 s for the first five dates
against 1471.3 / 1329.2 s — a ~6.6× speedup.** Fleet occupancy went from 30.6 to ~407 tasks in flight
(6% → 85% of slots), mean graph size from 482 to ~13,400 tasks, spill stayed at zero, hottest worker
3.96 → 5.6 GiB of 16.

That A/B pair is also what fitted `F = 7.04 s per window, V = 0.0346 s per chunk`, so one saved write is
worth ≈200 chunks of extra computed area. **The linear model is superseded by §12.2; the RATIO it
expresses is what the grouping algorithm still uses**, and that is the part that was ever load-bearing.

### 13.6 Why the first cut was a greedy heuristic, and why it was replaced

The first implementation merged greedily while the added area stayed within 25% of the row-band
baseline. With the cost model in hand that bound is simply the wrong shape: a fixed waste *fraction*
cannot express "extra area is nearly free", so it stopped merging on sparse zones — exactly where a tiny
absolute area makes merging almost costless.

Evaluated on all 112 real zone masks, predicted per-date cost summed:

| strategy | total | vs row bands | max graph |
|---|---|---|---|
| row bands (one per live chunk-row) | 73,636 s | 100% | 1,650 tasks |
| greedy, 25% waste bound, 512-chunk cap | 13,143 s | 17.8% | 5,632 tasks |
| **cost-model DP** | **6,733 s** | **9.1%** | 21,692 tasks |
| single bounding box per zone | 8,770 s | 11.9% | 42,636 tasks |

The greedy pass left **+95%** on the table. The DP also beats the pure bounding box, because it adapts
per zone instead of committing to one shape; the worst greedy zones were the sparse ones (`07S` +402%,
`08S` +311%, `03N` +292%).

So `merge_bands` now minimises `n_windows × WINDOW_COST_IN_CHUNKS + total_area` exactly, by dynamic
programming over groupings of consecutive bands. O(n²) with n = a zone's live chunk-rows (≤ ~230), and
the unit tests hold it against brute-force enumeration on small inputs.

**Choosing the graph-size cap.** `MAX_CHUNKS_PER_WINDOW` exists to bound one region write's graph, not
to bound cost. Sweeping it against the DP optimum:

| area cap | max graph (tasks) | total | vs uncapped |
|---|---|---|---|
| none | 28,050 | 6,728 s | — |
| 2,560 | 28,050 | 6,728 s | +0.0% |
| **2,048** | **21,692** | **6,733 s** | **+0.07%** |
| 1,536 | 16,874 | 6,794 s | +1.0% |
| 1,024 | 11,264 | 6,909 s | +2.7% |
| 512 | 5,632 | 7,463 s | +10.9% |

2,048 is the pick: it costs 0.07%, and it holds every graph at or below 21,692 tasks — just under the
22,812 a live run has actually driven with zero spill and a 5.6 GiB hottest worker. Note that even the
512-cap DP (7,463 s) beats the greedy pass (13,143 s) by 1.76×, so **the win comes from optimising the
right objective rather than from taking memory risk.**

Per-zone effect: `03S` 4 live chunks, 2 row bands → 1 window (+0.0% area); `15S` 22 → 5 → 1 (+45.5%);
`40S` 26 → 12 → 2 (+50.0%); `35N` 2,415 → 197 → 3 (+20.4%). Median 2 windows per zone, maximum 5.
**Sparse zones group *harder* in relative terms** — which is the correction, not a regression: their
added area is a large fraction of a tiny total, and a tiny total is where extra area cannot matter.

**What this leaves.** The empty-graph samples did not disappear — 3 of 10 after grouping against 16 of
44 before, a similar fraction of a much shorter date. Those gaps are the serial phase *between* dates:
the STAC query and graph construction, single-threaded on the ingest worker while the fleet waits.
That is §12.5, and it is a different fix from this one.

---

## 14. The region-write primitive and its contract

The region-write primitive lets a caller write one spatial window of an existing store without
rewriting the whole array, **which is what makes a windowed ingest possible at zone scale at all.**
`storage/zarr_store.py` is the record of what was built and its tests pin the contract. What this
section carries is the reasoning the code cannot: why the alignment problem is shaped the way it is,
what the multi-write-per-commit contract actually guarantees, and why the batch path was removed.

Verified against **icechunk 2.0.4 / zarr 3.2.1 / xarray 2026.4.0**.

### 14.1 Why `align_chunks=True` is not enough

A region that does not land on chunk boundaries forces a read-modify-write of the boundary chunks, and
`align_chunks` handles that inside one call — but **two calls writing into the SAME chunk in one
session are a lost update**, because each reads the chunk before the other's write is visible. That is
why the ingest snaps its windows to the chunk grid (§13.3): chunk-disjoint windows make the problem not
arise rather than handling it, which is also what lets N windows share one session and one commit.

The reconciliation the region-merge tier has to do — overlapping windows, shared boundary chunks — is
exactly the work this path avoids by construction. `storage/region_merge.py` and the ROI fan-out
foundations solve that tier directly above this one; code was taken from both and generalised, so the
generalised version here is the shared one.

### 14.2 The multi-write-per-commit contract (verified 2026-07-24)

Six tests, and what they establish is one contract the ingest still depends on:

**N `to_icechunk` region writes CAN share one `writable_session` and one commit** — twenty disjoint
chunk-aligned windows produced exactly one snapshot with no cross-window interference, ~10 ms per write
locally with no degradation. `to_icechunk`'s internal fork/merge tolerates a session that already
carries uncommitted changes.

**A session sees its own uncommitted append**, so `mode="a"` along time followed by `mode="r+"` region
writes into the just-appended date commits as ONE snapshot. That is what lets dates stay
discovery-as-you-go: append the slot, write its live windows, commit once, with no pre-enumeration.

**Uncommitted writes are invisible to readers on separate repo handles**, so commit atomicity holds.

**The caveat that is still the contract:** every window tested was exactly chunk-aligned, so
`align_chunks` never had to read-modify-write a boundary chunk. **Two writes straddling the SAME chunk
in one session remain forbidden** — disjoint chunk-aligned windows is the contract, not "multiple
writes work".

Without this, the per-window loop could not use `zarr_store.write_region`, which writes **one region
per commit**: at a median of 74 windows and ~50 dates that is ~3,700 commits per zone-year, against a
store whose commit pressure is a governed constraint ([ADR-008](../decisions/008-global-store-architecture.md)
D5/D6).

### 14.3 The batch path was removed

`write_regions` / `_write_regions` / `_aligned_region_sources`, built on `icechunk.dask.store_dask`,
was **removed as unused**: its `O(runs × bands × spatial_chunks)` Dask task graph — built
single-threaded on the flow runner before any compute ran — made continental merges take days. Its
replacement is a process-parallel raw-Zarr region merge with no Dask. The single-region `write_region`
path is current.

### 14.4 Gotchas that cost real time

1. **`align_chunks=True` ≠ chunk-boundary safety.** It rechunks producer-side dask blocks and avoids
   parallel races; it does **not** pad partial boundary chunks and **rejects** unaligned regions in
   `r+`. The read-modify-write pad in `_pad_region_to_chunks` is what makes arbitrary regions safe.
2. **Drop coords** — region-dim coords *and* non-region-dim coords; store coords are authoritative.
3. **Attrs are clobbered** — `to_icechunk` overwrites root attrs; snapshot and restore as
   `_write_append` already does.
4. **Read chunk sizes from the store**, not config — the store is authoritative and config may drift.
5. **Overwriting committed data.** A partial overwrite of an existing region exposes *stale real data
   mixed with new* to concurrent readers. Keep one commit per logical region; only split for memory.
   Also: "unwritten" and "real NaN" remain indistinguishable — do not infer population from contents.
6. **One session per region or slab**, committed per slab — bounds the changeset.
7. **`to_icechunk` is already distributed** on a dask-backed dataset (internal fork/merge across the
   graph). Per-region commit is the chosen strategy.
8. **Conflict detection.** Concurrent region writes to *disjoint* chunks do not conflict; overlapping
   chunks raise `ChunkDoubleUpdate`, resolvable via `BasicConflictSolver(on_chunk_conflict=UseOurs)`
   plus `rebase`. Only relevant if a future flow parallelizes uncoordinated writes to one array.
9. **No true mid-array insert is in scope** — deferred (resize + back-to-front shift + coord rewrite
   across commits; icechunk #1873 ordering hazard).
10. **Region slices are normalized and validated** — `_pad_region_to_chunks` resolves open bounds via
    `slice.indices`, rejects non-unit steps and empty slices, transposes incoming data to the store's
    dim order, and asserts each variable's shape matches the region, so a broadcastable smaller array
    cannot silently fill the whole region.
11. **`resolve_region` requires contiguous hits** — a coordinate range that straddles a gap on an
    unsorted axis is rejected rather than silently widened. Forward `get_credentials` / `s3_region` so
    the resolve and the write open the same store.
12. **Storage hang protection.** Icechunk defaults to unbounded timeouts and a single try, so a wedged
    socket blocks a write forever. `_default_repo_config` applies finite per-attempt timeouts and
    backed-off retries at every repo open — region writes inherit it for free.
