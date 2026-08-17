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

### Where the time goes, and why

Three facts explain nearly every decision in this document.

**A zone is mostly ocean.** A UTM zone is a tall, thin strip of the globe and typically only a
fifth of it is land. Computing the whole grid meant most of the work produced nothing, and the
first and largest change was to compute only the rectangles that contain land.

**A separate write costs far more than the pixels in it.** Each region write carries a
near-fixed overhead, so the *number* of rectangles mattered more than their total area. Merging
many small rectangles into fewer larger ones was worth 6.2× even though it computes more area,
including some ocean. Almost every subsequent tuning decision is a version of this trade:
how much dead area is worth avoiding one more boundary.

```
        the whole zone            only the land            land, merged
     (what we used to do)       (many rectangles)      (fewer rectangles)

    ┌──────────────────┐      ┌──────────────────┐    ┌──────────────────┐
    │~~~~~~~~~~~~~~~~~~│      │~~~~~~~~~~~~~~~~~~│    │~~~~~~~~~~~~~~~~~~│
    │~~~~~▓▓▓▓~~~~~~~~~│      │~~~~~[▓▓▓]~~~~~~~~│    │~~~~[▓▓▓▓▓▓▓▓]~~~~│
    │~~~▓▓▓▓▓▓▓▓~~~~~~~│      │~~~[▓▓▓▓▓▓▓]~~~~~~│    │~~~~[▓▓▓▓▓▓▓▓]~~~~│
    │~~~~▓▓▓▓~~~~▓▓~~~~│      │~~~~[▓▓▓]~~[▓▓]~~~│    │~~~~[▓▓▓▓▓▓▓▓]~~~~│
    │~~~~~~~~~~~~~~~~~~│      │~~~~~~~~~~~~~~~~~~│    │~~~~~~~~~~~~~~~~~~│
    └──────────────────┘      └──────────────────┘    └──────────────────┘
      computes everything;      4 boundaries to pay     2 boundaries, some
      ~78% of it is ocean       for, almost no           ocean recomputed —
      → never finished          wasted area              and it is FASTER

    ~ ocean, nothing to compute    ▓ land    [ ] one rectangle we write
```

The right-hand picture is the counter-intuitive one, and it is this campaign's central lesson:
**deliberately computing some ocean is cheaper than paying for another boundary.** Every later
tuning question — how coarsely to merge, when to stop — is that same trade re-priced as the other
costs moved.

**The scheduler is single-threaded, and it is the thing that runs out.** Its dispatch loop uses
one core no matter how many it is given, so the number of *tasks* a date creates — not the number
of pixels — sets how much fleet a single run can usefully absorb. This is why work went into
shrinking the task graph before spending money on more workers: a run that chokes its scheduler
at 120 workers cannot spend a larger allocation, and every concurrent run has its own scheduler
hitting the same wall.

### What changed, and the mechanism behind each gain

| Change | Effect | Why it worked |
|---|---|---|
| Compute only rectangles containing land | ingest completes at all | ~78% of the work was ocean |
| Merge those rectangles into fewer, larger ones | **6.2×** | a write's cost is mostly fixed, so count beat area |
| One masking pass instead of two | 1.28× smaller graph | each pass cost a task per block per band |
| Read the cloud mask as part of the band read | one fewer graph per date | it was being read twice |
| Write all of a date's rectangles as ONE computation | **1.59×** | they had been serial, so a date cost their sum |
| Prune unused metadata from retained catalogue entries | 2.45× less driver memory | most of each entry described bands we never read |
| Store the date index in shards | cost, not speed | every commit had rewritten the whole index |
| Match read blocks to stored chunks | **1.65×** on compact regions | coarse blocks capped how much could run at once |
| Re-price rectangle merging for overlapped writes | **18.6%** less written | a boundary stopped being expensive, so merge less ocean |
| Fuse several dates into one computation and commit | 1.61× to 0.71× **by region size** | amortises the commit only; see below |
| Prepare S1's next catalogue query during the current batch's writes | **~10%** on S1 | the query was serial and is now hidden |

Two things deliberately did not change: the store's chunk size, which the GPU inference path is
tuned around, and the atomicity of a commit — though fusing dates makes the atomic unit a group
of dates rather than one, which is forced by how Zarr resizes the time axis, not chosen.

### The one result that is not a straight win

Fusing several dates into a single computation **helps small regions and hurts middling ones.**
The arithmetic is that a fused write costs proportionally more and the preparation running
alongside it costs proportionally more, so the per-date cost is whichever of the two dominates,
plus one commit divided among the dates. Fusing therefore buys only the commit saving; it cannot
make the write faster, because the fleet is already the constraint. And it actively loses where
the larger write crowds out the preparation overlapping it.

Measured across four regions spanning a 127-fold size range: the smallest gained 1.61×, the next
1.12×, the middling one **lost 29%**, and the largest was unchanged. Because the relationship is
not monotonic, this ships as a size threshold rather than a fitted curve, and the threshold is
set at the top of the range where fusing was measured to win rather than at an estimated
crossover.

### What still limits us

**The fleet is already full on a dense region.** One date's work oversubscribes the workers, so
adding workers to a single run buys much less than it appears to — the measured ceiling from
unlimited workers is under 3×. More importantly, it means several optimisations that look like
they should help cannot: there is no idle capacity for them to fill.

**More than half of a date's elapsed time cannot be parallelised at all.** Client-side graph
building, dispatch, the blocking write and the commit are serial. Within the part that *is*
parallel, about 72% is reading and resampling the source imagery — real data movement, not
overhead — so there is no large inefficiency left to remove, only less work to do.

**One worker holds the retained catalogue and can deadlock the run.** The worker driving the
ingest keeps the current month of catalogue entries plus the next month prefetched. If it crosses
Dask's pause threshold it stops and never resumes, and the rest of the fleet waits forever on
data it holds. Worker memory is therefore sized against that pause threshold, not against the
container limit, and undersizing costs an entire run rather than a retry.

**A realistic cell is three fleets, not one.** A campaign cell runs Sentinel-2 and both
Sentinel-1 orbits concurrently, so its resource footprint is roughly three times a single ingest
run. Any concurrency plan costed on Sentinel-2 alone understates what the campaign needs by that
factor.

**What does NOT limit us: running many cells at once.** This was the campaign's largest open risk
and it is now measured up to **20 concurrent cells** (60 fleets, ~1,300 workers). Comparing
per-window cost and pairing each zone against itself, 5→10 cells costs **1.01×** and 10→20 costs
**1.24×** — against the **1.33×** that the narrower fleet quota forced at 20 cells predicts on its
own. Fleet width more than accounts for the slowdown, so no contention term survives. An earlier
forecast multiplied out a 1.04-per-cell penalty, implying 2.56× at 40 cells; that is **withdrawn**.
**The binding constraint on schedule is the Fargate quota, not interference between cells.**

### Three corrections worth reading before planning

> **Do not size cells against the dispatch-rate model in §2.** Its shape still holds but its
> constant is stale, and the advice it produced — keep cells narrow — is **withdrawn**. §2 has
> the detail.

> **How to get an ingest duration out of this document, because getting it wrong is easy.**
> Three tables here report per-date seconds and they answer different questions. Taking the
> wrong one cost real effort in July, so the mapping is written down:
>
> | you want | use | not |
> |---|---|---|
> | duration for a zone of known density | the **five-region k=1 column** (§3.16) | the k=4 column — that is a batching A/B |
> | how duration moves with fleet width | the **width model** `T(W) = 36.3 + 7896/W` (§3.10) | any single row, which is one width |
> | what overlapping bought S1 | the **three-ROI table** (§4.9) | anything about whole-cell duration |
>
> The five-region k=1 column fits `s/date = 10.16 + 0.06022 × live_4096_chunks`, R² 0.954, and
> those measurements sit at **~60 workers** — its 35N row (175.6 s) is within 5% of the width
> model at 60w (167.9 s), which is how the width is established rather than assumed. Two
> checks that the fit is sound: summed over the 111 real per-zone tile counts it gives **5.95
> h/zone-year at 60w against the campaign basis's 6.36, a 6.5% agreement**; and the apparent
> "~10 h versus ~21 h" disagreement between planning documents is the same dense zone at 120
> workers and at 50, via this model. **Nothing here was ever in conflict.**
>
> Two limits to carry with the fit. The intercept is a real fixed cost of about **1.0 h per
> zone-year**, so an all-but-empty zone costs an hour rather than minutes. And per-zone
> residuals run to **±35%** — 35N and 47N differ by 3 chunks in 2,418 and by 27% in per-date
> time, so area does not determine duration tightly.
>
> **The S1 table is a precondition, not an addend.** A cell runs S2 and both S1 orbits
> concurrently, so a cell's per-date cost is the MAX of the three arms, and the fit above is
> the S2 arm alone. Using it as the cell duration is legitimate *only* because §4.9 took S1's
> per-date write time down 2.4–3.9×, which is what brought S1's work to 15.5–18.1% of S2's and
> lets `s1_worker_fraction = 0.22` hide it inside S2's runtime with 20–40% to spare. Before
> that change S1 was the critical path on sparse zones and this substitution would have been
> wrong. If S1's per-date cost ever regresses, the substitution fails before the ratio does.

> **Dates per zone-year is 365, not the ~250 once assumed.** A full-height zone sees imagery
> every day. The earlier figure came from a region spanning about 3° of latitude, which
> intersects far fewer orbit passes and does not generalise to a zone.

### Why the graph was shrunk before buying concurrency

Running many cells at once is the obvious multiplier and needs no new code, so it is worth saying
why it came last. **Graph size sets the ceiling on how many workers one cell can usefully absorb**,
and the two multiply: a cell that chokes its scheduler at 120 workers cannot spend a larger
allocation however much quota arrives, and every concurrent cell has its own scheduler hitting the
same wall. Buying concurrency first would have bought the right to run twenty cells that each waste
most of their fleet.

That was known rather than guessed — ingest had previously degraded at ~250–300 workers, primarily
because of the scheduler. Cutting the graph ~3.9× moves the ceiling up by about the same factor. So
the order was **shrink the graph, raise the per-cell ceiling, then spend quota on concurrency**,
with catalogue streaming in between because year-scale cells cannot run at all without it (§7b).

---

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

### 4.1 Coarsening the STORE chunk to 8192 — rejected on the GPU side's own evidence

The largest available graph reduction (3.88×) and the one we did not take. The GPU-saturation
campaign had already measured the cost: with 4000² source chunks, "every read re-decompresses
whole storage chunks; measured fixed read ≈13 s regardless of strip size". A full-width strip
decompresses `width × chunk_edge` bytes — linear in the edge — so doubling the store chunk
roughly doubles that fixed read.

Set against what that campaign bought: GPU-idle overhead per chunk went **50–60 s → 24–34 s →
~6 s median** on prefetch-hit chunks. Adding 13+ s back roughly triples the steady-state
overhead a whole branch was spent removing, on a resource running at **99% utilisation during
inference** (VRAM peak 97%; host RAM has slack at 46% of a 60% operating ceiling, GPU does
not).

It compounds: the 5.75 GiB strip budget is sized so a full-width band load fits in ONE strip.
Coarser store chunks fit fewer rows per budget, so more chunks split into two strips, so more
two-strip co-residency — the direction of the **92–95% RAM OOM** that motivated striping. The
P3 budget raise *lowered* peak RAM by making more chunks single-strip; this pushes that lever
backwards.

**And a test design trap:** spatially-sparse chunks already read only a narrow easting window,
so a sparse zone shows less amplification than a dense one. Any chunk-geometry test must use a
DENSE area or it will look safe and not be.

### 4.2 Cost-model window grouping — over-merged into the scheduler

Fitting `F` and `V` from a matched A/B pair gave a ratio implying one saved write was worth
~200 blocks of extra area, so a dynamic program minimising `n_windows × 200 + area` took `35N`
from 15 windows to 3. **The per-date wall clock did not move** (193.9 s against 194.4 s) and
then *degraded* (233 s, 269 s) with dispatch lag reaching 2 s.

Cause: the linear model assumes area is dispatched at a fleet-limited rate. Past the
scheduler's throughput it is not, so the optimiser over-merges. Scheduler CPU went from 66%
mean / 93% max at 15 windows to **pinned at 100%** on 84,054-task graphs.

**Retained from the failure:** the window cap is now denominated in graph **tasks**
(`MAX_TASKS_PER_WINDOW`) rather than chunk area, because area silently changes meaning whenever
band count or block geometry moves — which subsequent changes did. The old 2,048-chunk cap
permitted that 84,054-task window.

### 4.3 Spatial manifest sharding — a 30–50% regression

The module's documented default splits `{northing: 4, easting: 4}`, and its comment states a
time split "would be a no-op at ≤256 dates". Following that default made ingest **30–50%
slower**: per-date went from 187.9 s to 245.8 and 281.3 s.

The default's reasoning is correct for the workload it was written for — the region-write merge
path, where "each write_region commits one compact, scattered ~3×3-chunk block". **Campaign
ingest commits one DATE covering every live window**, i.e. essentially the whole live area, so
no spatial shard is ever untouched. Measured: **~5,097 manifest objects rewritten per commit
instead of ~14**, and those PUT latencies cost far more than the bytes saved.

**The rule, now documented at the constants:** split the axis along which a single commit is
NARROW. Campaign ingest is narrow in time and wide in space — the exact inverse of the merge
workload.

### 4.4 Doubling load blocks again, 8192 → 16384 — measured, not worth it

Predicted 2–3×; **measured 1.35×**, for 4× the memory per task (134 → 537 MB per band-block) on
a fleet whose hottest worker already reached 10.3 of 16 GiB.

It flattens because the write tasks are one per (store chunk, band) and do **not** scale with
block size — they are set by the store's chunking, which is pinned. Writes were 13% of the
graph at 4096 and 28% at 16384; even a free read side would only give 3.5× more, ever. Gated on
a local task census before any spend, which is why nothing was spent.

### 4.5 Rejected without testing, and why that is defensible

- **Coarsening `INGEST_CHUNKS["time"]` beyond 1** — would batch dates per commit, breaking the
  property that a date's time slot lands atomically with its pixels. That property is what
  makes a crashed ingest safely retryable; the ingest marker makes a wrongly-recorded date
  permanent.
- **Folding the ROI mask inside `odc.stac.load`** — no clean hook exists; ~80+ LOC of
  loader-internal surgery for one task per (block, band).
- **Raising the scheduler's vCPU** — its CPU never exceeded 100% across 114 samples despite ten
  threads and 4 vCPU allocated. The event loop is single-threaded and GIL-bound; more cores
  would idle. See §6.
- **Raising the page size on the STAC query** — capped at 250. `limit=500` and `limit=1000`
  both fail against earth-search with repeated server errors.
- **More workers per cell, at the time it was proposed** — two single threads serialised ≥95%
  of a run, bounding any fleet increase at ≤1.1×. This has since changed; see §7.

---

### 4.6 Removing the realignment — reverted on spill, after a census that over-predicted

Two failures in one, and both are instructive.

**The census over-predicted by ~20×.** It counted, over one fixed window, the optimised graph
of each variable plus write tasks added *analytically* — one per store chunk with realignment,
one per block without:

| | read+mask | writes | total |
|---|---|---|---|
| with `align_chunks` | 1,985 | 528 | 2,513 |
| without | 929 | 132 | 1,061 |

That is 2.37× on the window. The live run, image digest verified, measured **mean 4,877 tasks
against 5,159 — about 5%.** The census had modelled the write layer rather than measuring it:
the region write must produce store-chunk-granular writes regardless, so it does that splitting
itself and `align_chunks` only controls whether xarray pre-rechunks. §3.8 later confirmed the
graph is essentially *all* write tasks, so there was never a large win there to find.

**Then the wall-clock gain that did exist failed its own pass condition.** Over 12 dates the
change ran **176.5 s against 184.5 s — 4.3% faster** — but worker **spill went from zero to a
3.19 GiB peak across ~30% of scheduler samples**, because each write task now carried a whole
load block (134 MB per band) instead of one store chunk (34 MB). Peak worker memory sat at
10.58 GiB against 10.16 with realignment, i.e. both configurations run near the spill threshold
and this one crossed it.

The stated pass condition was zero spill, so it was **reverted**. Spill is a hidden cost that
has scaled badly in this stack before, and 4.3% does not buy it. An untested middle path, if the
gain is ever wanted: halve the threads per worker to restore the headroom.

**Kept from the attempt:** `write_day_windows` now raises if a window is not store-chunk-aligned.
That invariant is worth having documented and enforced regardless of which side of this decision
we are on.

**A correction to an earlier claim in this document:** an interim reading said peak worker memory
*fell* from 10.16 to 6.86 GiB with the realignment removed. That came from the first twelve
heartbeats of the run; over the full run it reached 10.58 GiB and spilled. Early samples of a
memory series are not a memory measurement.

### 4.7 Window geometry beyond the shipped narrowing — three variants, all worse

Measured locally against a real date's item footprints (`scratchpad/window_geometry.py`), so none
of these cost fleet time. Area is in 8192 cells; the ideal is the live-and-imaged cells themselves.

| strategy | windows | area | vs current | verdict |
|---|---|---|---|---|
| current static (every date) | 7 | 748 | 1.00× | — |
| **narrowed + merge (shipped)** | **5** | **560** | **0.75×** | kept |
| narrowed rows, unmerged | **80** | 382 | 0.51× | rejected — 80 serial region writes |
| narrowed columns, unmerged | 8 | 531 | 0.71× | rejected — no better than shipped |
| ideal (cells themselves) | — | 372 | 0.50× | unreachable by rectangles |

**The whole prize was 2×, not 5×, and three quarters of it is already taken.** Reaching the ideal
costs 16× the window boundaries, and a window boundary is a blocking region write — the exact
regression §3.2 fixed.

Two further variants were rejected on arithmetic rather than experiment:

- **Load blocks 8192 → 4096**, to let narrowing work on a finer grid. Would recover ~1.6× more
  area, but the same measurement shows it touching **1,083 load blocks instead of 372 — 2.9× the
  load tasks**, reversing §3.5. Graph size is the scale constraint, so this trades the scarce
  thing for the abundant one.
- **Write windows at 4096 while load blocks stay 8192.** Technically sound (`align_chunks=True` is
  already on for the windowed write, so producer blocks remap), but it takes windows from 5–6 to
  **12–13 per date**. A within-run natural experiment bounds what that costs: in the footprint run
  the five-window dates averaged **158.3 s** and the six-window dates **175.8 s** — about **17 s
  per window** (n=2, wide spread, but it brackets the ~7 s of §3.2 rather than contradicting it).
  So 5 → 12 windows costs 50–120 s per date to save write area that §3.10 shows is not on the
  critical path.

### 4.8 Worker memory back to 16 GiB — a rejected size that pruning made affordable

`DEFAULT_INGEST_WORKER_MEM` 20480 → **16384 MiB** (2026-07-28). Not a new idea: 16384 was
**explicitly rejected** once, on the grounds that "demand is ~12.4 GiB; 16 GiB leaves ~0.4 GiB of
margin which is too thin". That premise no longer holds, because pruning the retained catalogue
entries (§3.13) took roughly 3.3 GiB off the driver worker.

Measured over a 91-date run spanning three monthly rollovers, 60 workers, dense zone:

| arm | peak hottest worker | spill | shape |
|---|---|---|---|
| k=1 | 6.41 GiB | 0.00 | plateaus by ~hour 3 |
| k=4 | **7.91 GiB** | 0.00 | plateaus by ~hour 3, flat for the final 2.5 h |

Both **plateau and stay flat across two further month boundaries**, which is the retained-item
accumulation question answered directly rather than by extrapolation: monthly streaming does bound
retention.

**Sized against the PAUSE threshold, not the container limit.** At 16 GiB the pause threshold is
12.8 GiB, so the worst case leaves **1.6×**. That is tighter than the 3.3× short runs suggested,
because batching costs ~1.5 GiB of peak — and batching is now mostly *not* used (§3.16), so the
realistic peak is the 6.41 GiB k=1 figure and the margin is 2×. What makes the pause threshold the
right target rather than the limit: a paused worker **does not recover**, so work waiting on data it
holds can never complete and the run DEADLOCKS with the rest of the fleet idle. Undersizing costs a
whole run, not a retry.

Still to verify before the campaign: **one full-year cell**, because twelve rollovers is four times
what this run exercised and a slow leak would compound.

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

Found under load, not by review. ASF mints AWS credentials (via Earthdata) that live about an hour,
and the S1 loop renewed them **only between time batches**. At the 30-day default a 15-day range is
ONE batch, so there was no boundary at which to renew: a batch longer than the credential could
never refresh it, and every read afterwards failed with
`CPLE_AWSError: The provided token has expired`.

It is a **duration threshold**, which is why it presented as "some runs fail and others don't"
rather than as a configuration error. In one 5-cell concurrency rung both dense zones failed (394,
355, 127 and 110 errors) while the sparse zone in the same rung finished cleanly, having completed
inside the hour. Zero S2 cells were affected.

Fixed by driving renewal from the credential's own advertised expiry with a 15-minute margin, and
checking before **every date's write** rather than per batch. The margin must exceed one date's
write, since a credential valid when a write starts must still be valid when it ends. The per-date
check is two timestamp comparisons with no I/O, so the refresh *rate* is unchanged at roughly once
per 45 minutes — what changed is how often staleness can be noticed. An absent or unparseable
expiry degrades to the old cadence rather than raising.

**The general lesson:** a renewal cadence tied to a unit of work is only safe while that unit is
shorter than the credential. Tie it to the credential's clock instead.

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

### 4.14 The crop flag was OFF by default, and that is why a "tiny" zone exhausted its workers

The campaign's own default path had never been run. Every measurement in this document passed
`crop_to_live_windows=True` **explicitly**, because they dispatched the S1/S2 deployments
directly with their own parameters. The first test to go through `ingest_zone_year` with
default `IngestSettings` — M8 — took the uncropped path, because the setting defaulted to
`False` ("until the cropped path is validated end to end") and NOTHING in production source
set it True: not `run_global_campaign`, not `fill_zones_sequential`, not `ingest_zone_year`,
which reads it straight from the settings object.

**The gap is ~420x.** On 15S land is 0.238% of the zone: 2,187 live 256-px chunks of 918,720.
Uncropped materialises the whole zone extent, so Dask spilled managed task results to a worker
holding Fargate's default 20 GiB and the spill failed with `OSError: [Errno 28]`. Worth being
exact about what spills, because it is a common confusion: Dask can only spill data it
MANAGES. **Unmanaged memory cannot be spilled at all** — and ingest is ~82% unmanaged (§8) —
so the spill traffic was the loaded full-extent arrays, not a leak.

**Every figure in this document assumes cropping.** Wall clock, cost, vCPU per cell, the
batching thresholds, the fleet widths. Run uncropped and none of them hold.

**The flag is now GONE rather than defaulted to True**, because no scenario wants it off: the
mask IS the ROI store so cropping needs no extra prerequisite; a fully-dense ROI degenerates
correctly; an all-ocean ROI correctly writes nothing; `batch_dates > 1` ALREADY required it,
so the shipped optimisations presumed it; and yield-embeddings never referenced it. Removing
it also deleted the `batch_dates > 1 requires crop_to_live_windows` validation, whose
precondition now holds by construction. TE `d2dbb8f`.

### 4.15 An interrupted ingest is RESUMED, not rebuilt

A cancelled or crashed cell used to be CLEARED and re-ingested from scratch, so every interruption
cost a whole cell — and interruptions are expected, because the orphan sweeper cancels runs by
design. A dense zone interrupted near the end lost hours deterministically on every retry.

Three states, three answers. **Absent**: ingest. **Present and unmarked**: resume, skipping dates
already committed rather than rewriting them. **Present and marked with a different window**:
refuse, because that is a different question being asked of the same store.

### 4.16 Two writers, observed: what the guard caught and what it did not

47S/2021 was dispatched **four times** as four independent top-level `ingest-zone-year` runs
(16:20, 17:46, 18:36, 18:53 UTC), each with `retries=0` and no parent — four manual dispatches,
not retries. Generations 1→2→3 never overlapped and each resumed correctly from the previous
one's commits, which is 4.15 working as designed. Generation 4 was dispatched **17 minutes into
a healthy generation 3** and collided with it.

**What the guard caught.** Generation 3 committed 2021-04-12 (ascending) at 18:56:31;
generation 4 attempted the same date at 18:56:40, nine seconds later, and was refused. On
descending, generation 4 hit `ConflictError: expected parent Q8BAVXYY, actual parent AV8HGF6Q`,
where the actual parent is generation 3's 2021-06-25 commit seven seconds earlier. All three
stores came through **clean**: no duplicate dates, strictly increasing axes, no orphan
snapshots, and generation 4 committed nothing. Its cells were wasted; nothing was corrupted.

**What the guard did not catch, and the fix.** Both retry sites retried the refusal. On a
same-date collision the retry then failed as a duplicate — so the operator sees *a date*, not
*a collision*, which is a long way from the cause and cost an hour of investigating solar-day
grouping that turned out to be correct. On a **different-date** collision the retry would have
SUCCEEDED, because it re-reads the tip the other writer moved: two writers interleaving dates
onto one axis, silently. `store_write_retrying` now excludes `icechunk.ConflictError` and the
new `DuplicateDateError` by type. It is one shared policy because it was three hand-built ones
(S1 per-date, S2 per-date, S2 per-batch) and the exclusion was missing from all three at once —
the triplication *was* the defect. A source-level test now fails if any ingest module builds
its own `Retrying` again.

**Prevention sits on the dispatch side, not in the flow.** `scripts/run_campaign_cell.py`
(yield-embeddings) refuses to dispatch when a run of the same deployment with identical
parameters is still live, `CANCELLING` included — the ECS task and its Dask cluster outlive the
state change, so a cancelled cell can still be committing. Deliberately not in
`ingest_zone_year`: a flow-level refusal would strand a crashed run that Prefect still reports
as `RUNNING`, turning a recoverable crash into a lost cell. Prevention where a human can
override it, a loud failure everywhere else.

**The misdiagnosis is worth recording too.** I asserted that 2021-04-12 was absent from the
store and built a solar-day regression hypothesis on that absence. It was present all along,
in a snapshot I could have read in one command. Walking the Icechunk ancestry is cheap and
dates every commit; it should come before any theory about how a date was derived.

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

Four questions this section opened have since been answered, and are recorded as one line
each rather than as the paragraphs that argued them — the arguments live in the sections
cited, and a list where most entries are settled stops being readable as a list of what is
open.

- **ANSWERED — is the ingest reproducible across the campaign and single-ROI paths? Yes**, on
  current code: 65 of 65 shared dates, identical arrays and chunking, bit-identical pixels
  across all 11 bands on five sampled dates, against the preserved July reference on identical
  geometry. The one extra date is the coverage threshold, i.e. configuration. This was the
  campaign's one open question about scientific soundness; the earlier answer was no (3.11% of
  pixels disagreeing), and three date-handling fixes landed between the measurements — §4.9,
  §4.11, §4.12.
- **ANSWERED — is the fleet-bound floor reached? Yes** (§3.10, §3.12). A large fraction of a
  date compresses under no worker count, so **cell concurrency is the lever and cell width is
  not**. Two caveats travel with it: the sweep's absolute values are not reproducible (§5), and
  its width curve ran pre-overlap, so post-overlap width sensitivity must be re-measured rather
  than inherited.
- **ANSWERED — can a date's serial residual come off the critical path? Yes**, 1.59× from
  overlapping (§3.11), with date pipelining taking the rest (§3.16). The durable lesson is that
  **removing a serial term beats shrinking it**.
- **ANSWERED — the month-by-month streaming blocker this section was written around** shipped
  and is validated (§7b).

Genuinely open:

- **Is the 4096 write-window idea viable again?** §4.7 rejected it because it took windows from
  5-6 to 12-13 per date at ~17 s of serial cost each; overlapping removed most of that per-window
  serial cost, so the arithmetic that killed it no longer holds. Expected win is small — it cuts
  computed area ~1.6×, and area was not on the critical path when measured (§3.9) — and the
  memory cost is real. **Measure the prize before building it**, per §3.11's lesson.
- **Is `F` fleet-invariant?** One block's fetch latency should be; scheduler round-trip would
  grow with fleet size. Decides whether a window cap tuned at 120 workers transfers at all.
- **Commit time versus manifest size** — the last unmeasured per-date term.
- **Does the scheduler's auxiliary-thread behaviour change at 250+ workers?** Re-check the
  one-core finding before resizing at scale.

Questions needing more workers than the current quota allows are tracked as S-1..S-6 in the
downstream repo's `docs/global-tessera-test-plan.md`.

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
| worker RSS per retained item | ~80 KB → ~27–30 GB/year vs 16 GiB worker | 3-point slope |
| unsharded manifest cost | ~1.1 MB refs/date → ~35 GB per zone-year | fitted, N²/2 |
| time-sharded manifest cost | ~1.2 GB per zone-year (~28× less) | predicted 34.7 MB at 9 dates, measured 34.6 |
| per-date variance | 19% SD (n=12) | measured |
| load-block census (fixed window) | 4096: 4,085 · 8192: 2,513 · 16384: 1,856 · 8192 no-align: 1,061 | measured |
| memory per band-block | 4096: 34 MB · 8192: 134 MB · 16384: 537 MB | arithmetic |
| hottest worker | 10.16 GiB of 16 at 8192 blocks; spill 0 throughout | run telemetry |
| inference baseline (do not regress) | GPU util 99% in-phase, VRAM 97% peak, host RAM 46% of a 60% ceiling, GPU-idle ~6 s/chunk | 2,352 RESOURCES samples |
| **ingest worker size (CURRENT)** | 4 vCPU / **16384 MiB** | §4.8. Peak 7.91 GiB over 91 dates and three rollovers, plateauing by hour three, zero spill → 1.6× margin to the pause threshold. Superseding, in order: 16384 → 30720 → 20480 → 24576 → **16384**; pruning the retained catalogue (§3.13) is what made the original size affordable again. vCPU stays at 4 — the quota counts vCPU. |
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
- **Worker memory is not constant across rows**: 16 GiB up to 2026-07-25, 20 GiB after, and back
  to **16 GiB from 2026-07-27** (§4.8). Memory size barely moves wall clock, so the timing rows
  still compare — but a peak-memory figure only means something beside the limit it was measured
  against.
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
