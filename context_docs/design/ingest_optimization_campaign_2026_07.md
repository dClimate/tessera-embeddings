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

> **SUPERSEDED IN PART by §3.9.** This section's arithmetic is correct but its denominator is
> not: it derives the floor for writing the WHOLE ROI, and a single date writes only the
> fifth of the ROI its own imagery reaches. The real floor is about five times smaller, and
> the conclusion drawn here — that graph work was closed — was premature. Read §3.9 before
> treating any number below as a limit.

**The graph is now essentially all write tasks, at one per (store chunk × band).** Predicted
against observed, window by window on the shipped configuration:

| blocks | store chunks | predicted (`chunks × 11`) | observed live graph |
|---|---|---|---|
| 120 | 480 | 5,280 | 5,831 |
| 120 | 480 | 5,280 | 5,684 |
| 117 | 468 | 5,148 | 5,343 |
| 100 | 400 | 4,400 | 4,246 |
| 54 | 216 | 2,376 | 2,432 |

Seven windows, seven matches. The read-and-mask side — everything §3.3 through §3.6 attacked —
has fused down into the noise. What remains is `store_chunks × n_bands`, and **both factors are
pinned by decisions taken deliberately**: 4096 chunking because the GPU path is tuned around it
(§4.1), eleven bands because that is the data.

This also retro-explains the realignment work (§4.6): there was never a large win available in the write layer, because
the write layer emits one task per store chunk regardless of what it is asked to do.

**The only remaining headroom is dead area: windows cover 2,992 store chunks against 2,415 live
— 577 dead, 19%.** So 32,912 write tasks are issued where 26,565 is the floor. That 19% comes
from a trade the realignment removal (§4.6) introduced: not realigning requires windows on the 8192 block grid, and
coarsening the live grid marks a whole block live if any quarter of it is. Recovering it means
4096-grid windows, which brings realignment back — and that measured **worse** (5,159 tasks
against 4,877).

The configuration space is therefore closed, all three measured:

| configuration | live graph per window | per-date | spill |
|---|---|---|---|
| **load 8192, 8192 windows, realigned** (shipped) | 5,159 | 184.5 s | **0** |
| load 8192, 8192 windows, not realigned | 4,877 | 176.5 s | 3.19 GiB peak — reverted (§4.6) |
| load 4096, 4096 windows | — | 187.9 s | 0 |

**Verdict: graph work is complete.** We sit within 19% of a floor set by constraints we have
chosen not to move, and the last three experiments returned 5%, 1.35%, and a 30–50% regression.
The achievement is not the per-date seconds — it is that the scheduler went from **pinned at
100% to ~25%**, which is what caps workers per cell and therefore what caps the concurrency
multiplier (§1).

Anyone reopening this should start by asking whether the store's chunking or the band count can
move. If neither can, there is nothing here worth more than 19%.

---

### 3.9 The per-date footprint — four fifths of every graph was computing nothing

**Found by counting objects, not by profiling**, and it reframes §3.8's write-floor
conclusion. A completed 7-date run on a dense zone was compared against the area its graph
covered:

| quantity | value |
|---|---|
| chunk objects in the store, 7 dates | **44,797** |
| ⇒ chunks per band-date | **582** |
| chunks covered by the run's live windows | **2,992** |
| ⇒ share of the graph that produces data | **19.4%** |
| Sentinel-2 revisit ⇒ expected daily coverage of a zone | **20.0%** |

That agreement is the finding. **The live windows describe where the ROI has LAND, and were
being reused unchanged on every date — but one optical pass images only a fraction of a wide
ROI.** Four fifths of the ~33,000 write tasks per date ran, found no data, and wrote nothing,
because an all-fill chunk is never stored. The waste was invisible precisely BECAUSE the
output was already correct.

**Why it never appeared on the single-ROI path.** The ratio is (ROI extent ÷ what one pass
covers). A yield ROI is smaller than one satellite swath, so a date covers essentially all of
it and there is nothing to skip. It only appears once the ROI is a 6-degree zone. Anyone
reasoning from single-ROI experience will not expect this, and that is not an error on their
part.

**What it explains.** §3.8 concluded the graph had reached a write floor of
`store_chunks × bands` and closed graph work. That floor was real but measured against the
wrong denominator: it is the floor for writing *the whole ROI*, and a date does not write the
whole ROI. The genuine floor is `chunks_the_date_images × bands`, five times smaller. So the
per-date fixed cost — graph construction, task dispatch, per-window region writes — was about
five times larger than the work required.

**The fix** (`live_windows.windows_for_date`): intersect the run's windows with the footprint
of that date's own items before building the graph. Both cost terms fall together, which is
the point — the graph shrinks with the area, AND windows the date misses entirely disappear,
taking their serial region writes with them. Per-date window count on a dense zone should
drop from 7 to roughly 1–2.

**Why it cannot change a mosaic.** The chunks it skips are already absent from the store —
that is what the 582-against-2,992 count proves. It removes computation whose result was
being discarded, so the output is identical by construction. The safety burden is therefore
entirely on the footprint being CONSERVATIVE:

* every uncertain path returns "assume everything" and restores the previous behaviour — an
  unreadable bbox, an unreadable geobox, a failed projection;
* all rounding goes outward, and the footprint is padded a whole cell on every side, which is
  tens of kilometres against a curvature error measured in metres;
* the coverage gate deliberately keeps the run's FULL window set, because its ratio asks how
  much of the ROI's land a date saw and cropping the denominator would rescale every
  percentage. The numerator is unaffected: there are no valid pixels outside the footprint.

Too large costs discarded area; too small silently drops imagery and nothing downstream
notices. The tests are weighted at that asymmetry rather than at the happy path.

**Per-date metadata is unaffected**, checked rather than assumed: `doy` is a scalar list with
one entry per timestep, `baselines_applied` a per-date dict, and the time slot is appended
once per date — none of them per-window. Narrowed windows stay 8192-aligned with ends clamped
to the extent, satisfying the write path's chunk-alignment guard, and remain chunk-disjoint by
construction.

**Status: SHIPPED, CORRECT, and worth ZERO wall clock.** Measured against an anchor run on
identical dates and width:

| | anchor | footprint |
|---|---|---|
| median per date | 172.5 s | **172.4 s** |
| paired speedup over 7 identical dates | — | **1.01×**, 4/7 faster, mixed directions |
| windows per date | 7 | 5.3 mean |
| graph tasks per date | ~40,000 | ~31,000 (−23%) |
| scheduler CPU | ~40% | ~26% |
| **chunk objects written** | **44,797** | **44,797 — IDENTICAL** |

The correctness claim held exactly: the two stores are object-for-object the same, so the change
provably removed computation and not data. **The performance projection was wrong**, and the
reasoning behind it was wrong in two independent ways:

**1. The 19.4% was a granularity artifact.** It counts 4096 px chunks; windows are built on the
8192 px load-block grid, where one cell is 82 km. A swath edge clipping a cell dirties the whole
cell. Measured at the grid the ingest actually uses, one date touches **47–57% of live cells**,
not 20%. Nothing can narrow below its own grid.

**2. The dead area splits two ways, and the split was stated backwards.** Measured over three
dates at the 4096 grid (live 2,415 chunks):

| | share of live |
|---|---|
| **geometric dead** — no image that day | **55–66%** — a window strategy CAN skip it |
| radiometric dead — imaged, but cloud/invalid so masking zeroed it | 10–21% — it CANNOT |
| written | 24% |

An earlier version of this section claimed the radiometric part dominated and that geometry
therefore could not help. **That is withdrawn (§5)** — geometry is the larger share. What limited
the shipped change was the grid it operates on, not the nature of the dead area. The catalog
returns ~960 items per date for this zone, roughly twice what covers it, because the zone reaches
84°N where orbits converge — so imagery is present for far more of the zone than a 5-day revisit
suggests, and the cloud share is a genuine floor no footprint can see in advance.

**Keep it anyway**, on the scheduler rather than the clock: −23% graph and −35% scheduler CPU are
the currency that buys worker count, and §3.10 shows the clock was never graph-bound. That is a
hypothesis about scale, not a measured benefit.

### 3.10 Where a date's time actually goes — the profile that reframed the campaign

Measured with the toolkit's existing hook: `perf_report_uri` is already a flow parameter, so this
needed **no code change** — a two-date probe run captures the Dask task stream to S3, and the
report's own numbers are grouped per stage (`scratchpad/perf_budget.py`).

| stage | share of task work | s/date at perfect packing |
|---|---|---|
| **source read + resample** | **72.3%** | 41.4 s |
| mask + region write | 16.3% | 9.4 s |
| inter-worker transfer | 7.8% | 4.5 s |
| coverage gate reduce | 2.9% | 1.7 s |
| total task work | 100% | **57.2 s** |

And the line that matters:

| | |
|---|---|
| measured wall clock per date | 131.0 s |
| task work at PERFECT packing | 57.2 s (44%) |
| **irreducible by adding workers** | **73.8 s (56%)** |
| **ceiling from unlimited workers** | **1.78×** |

> **EVERY PACKING FIGURE IN THIS SECTION AND §3.12 IS WRONG — read §3.14.** All of them were
> computed from a **TRUNCATED** Dask task stream (a bounded deque, default 100,000 records), so
> each report covered only its run's TAIL while the analysis divided by the full date count.
> Corrected: task work at perfect packing **64.3 s/date**, width-independent residual **~37 s
> (~36%)**, one-cell ceiling **~2.8×** — not the 1.78× here, nor §3.12's 1.60×. The qualitative
> verdict does not survive either: **most of a date DOES pack**; about a third does not.

**Three things this settles.**

*The graph was never the cost.* 72% of compute is fetching and resampling source COGs — real data
movement against the public Sentinel-2 archive. That is why removing 23% of the graph (§3.9)
moved the clock by nothing, and why §3.8's write-floor framing pointed at the wrong quantity.

*More workers cannot help much, and the number is now known.* Only the 57 s packs; the 74 s of
serial client work, dispatch, blocking region writes and commit does not. 1.78× is the ceiling on
one cell however wide the fleet — independently consistent with the 1.2–1.7× obtained by fitting
`A + B/W` to a three-rung worker sweep, which is worth noting because the two methods share no
inputs.

*It reconciles two results that looked contradictory.* Real I/O should scale with workers, and it
does — that is the 41 s. A worker sweep showed scaling flattening above 60 workers, and it does —
because the 74 s residual is more than half the date. Both observations were correct.

**The lever this identifies.** The residual is the target, and shrinking it is not the only way to
remove it: **overlapping one date's serial phase with the next date's compute** takes it off the
critical path entirely, which is worth more than any further graph work. That is the open question
in §7, and it is the first ingest change in this campaign to be proposed from a profile rather
than from a model.

### 3.11 Overlapping a date's windows — 1.59×, and the first change proposed from a profile

§3.10 left one target: 74 s of every date that no fleet width could compress. Instrumenting the
write path located it precisely, and the location was a surprise — it was none of the candidates
reasoning had offered.

**Phase I: instrument, then decide.** Three permanent log lines (per-date stage timings, per-window
timings, commit timing) partitioned a date without inference. Measured over 7 dates, sequential:

| | mean |
|---|---|
| build (client-side graph construction) | 10.8 s |
| gate (coverage compute) | 7.2 s |
| **write phase** | **148.0 s** |
| total | 166.1 s |
| **commit** | **0.6 s** |
| sum of per-window times | **146.9 s** |
| longest single window | 37.8 s |

**The windows explained 99% of the write phase** (146.9 of 148.0), and no single window dominated
(37.8 s against a 146.9 s sum). So `write_day_windows`' loop — one blocking `to_icechunk` per
window — was the residual: the fleet worked on exactly one window at a time and idled through the
rest of each date. The commit was never the cost.

This was gated on a pre-registered kill condition: had one window been most of its date, the
overlap would have bought nothing and the plan said stop. It was not.

**Phase II: one compute for the date.** icechunk's dask path already forks a session, stores
lazily and merges; `to_icechunk` merely runs that once per call. Lifting the sequence one level —
build every window's region writer, fork ONCE, collect all windows' lazy stored arrays, run ONE
merge reduction — puts every window's loads, masks and chunk writes in a single graph whose
critical paths overlap across the fleet. One commit per date, `align_chunks`, the alignment guard
and the failure contract are unchanged by construction.

**Phase III: the A/B, on identical dates and width.**

| | sequential | parallel | |
|---|---|---|---|
| **chunk objects written** | **44,797** | **44,797** | **identical** |
| write phase | 148.0 s | **86.5 s** | **1.71×** |
| per-date total | 166.1 s | **104.2 s** | **1.59×** |
| commit | 0.6 s | 0.9 s | negligible |
| peak worker memory | — | 12.30 of 20 GiB | no pause |
| worker spill | 0.00 GiB | 0.00 GiB | |

Pre-registered prediction was 90–110 s per date at 1.5–1.8×; measured **104.2 s at 1.59×**, inside
the range and independently consistent with §3.10's 1.78× packing ceiling derived from a different
instrument. **This is the only projection in this campaign that landed** — the two that did not
(§3.9, §5) were both projected from aggregates without measuring the mechanism first, which is
exactly what Phase I existed to prevent.

**Why not more.** The write phase falls to ~86 s, not to the 37.8 s longest window: the work still
has to happen, and ~85 s is near the packing floor rather than the critical path. The remaining
per-date budget is build 10.4 + gate 7.3 + write 86.5.

**It also made a date MORE responsive to fleet width, which bears directly on scaling.** Both
arms captured task streams, so each arm's packing can be derived from its own instrument:

| | sequential | parallel |
|---|---|---|
| task work at perfect packing | 34.0 s/date | 39.1 s/date |
| share of the date that packs | **20%** | **38%** |
| headroom from adding workers | **1.26×** | **1.60×** |
| total task work | 114,294 slot-s | 131,528 slot-s (**+15%**) |
| task count | 182,311 | 178,123 (−2%) |
| inter-worker transfer | 8,019 slot-s | 10,032 slot-s (+25%) |

Two things follow. **The overlap nearly doubled the fraction of a date that parallelises** (20% →
38%), so per-cell width is worth more after this change than before — a wider fleet now buys
~1.60× where it bought ~1.26×. That partially softens §3.10's conclusion that fleet width is
nearly spent: it is spent *for the serialised implementation*, less so for this one.

And **it costs ~15% more total CPU-seconds for the same output**, with 2% FEWER tasks and 25%
more inter-worker transfer — so individual tasks got slower rather than there being more work.
That is contention, the ordinary price of parallelism: 1.15× the compute for 1.59× the latency.
Worth stating because at multi-cell concurrency the aggregate CPU matters, whereas here latency
is what the deadline cares about.

**Now the default** for S2 (`overlap_window_writes=True`). `write_day_windows`' own
`parallel_windows` default stays **False** deliberately: S1 also calls it, its windows have never
been A/B'd, and a storage-layer default must not change behaviour for an unmeasured caller. S1 can
opt in when someone measures it.

**The commit-growth concern it raised is RESOLVED — the cost is bounded, not cumulative.** The
A/B's parallel arm showed commit times climbing monotonically across its 7 dates (0.5 → 1.2 s)
where the sequential arm stayed flat (~0.6 s), which read as possible cumulative drift and was
recorded as the sharpest question for the year soak. A longer run answers it: the series is a
**sawtooth with a period of exactly 8 dates**, climbing to ~1.5 s and resetting to 0.5 s —

```
0.5 0.9 0.9 0.9 1.2 1.1 1.2 1.5 │ 0.5 0.9 0.9 1.0 1.1 1.3 1.0 1.2 │ 0.5 …
└──────── dates 1-8 ────────────┘ └──────── dates 9-16 ───────────┘  └ 17
```

8 is `INGEST_MANIFEST_SPLIT["time"]`. Each commit rewrites the current shard's manifest, which
grows as dates accumulate in it and resets when the next shard opens. So over a 330-date
zone-year the commit oscillates between 0.5 and ~1.5 s rather than growing — **§3.6's manifest
sharding turns out to bound commit cost as well as object count**, which was not among the
reasons it was adopted.

Two residual facts worth keeping. The overlapped path DOES grow commit time faster *within* a
shard than the sequential one (1.2 s against 0.6 s by date 7), plausibly because one merged
changeset per date produces a larger manifest delta than several smaller merges. And the reason
the A/B could not see the reset at all is that both arms ran 7 dates — entirely inside one
8-date shard. **A 7-date window cannot distinguish bounded periodic cost from unbounded growth**;
only a run crossing a shard boundary can.

**Robustness.** The lift uses `_XarrayDatasetWriter`, which icechunk marks private. The version is
lockfile-pinned, parity is pinned by test, and any import or signature drift falls back to the
sequential loop with a warning — so drift degrades to the previously shipped behaviour, never to a
failure. The A/B confirmed the fallback did not fire and no per-window lines appeared, i.e. the new
path genuinely ran.

### 3.12 What the overlap did to the packing budget — and why §3.10's programme verdict survives

§3.11 established the overlap's win from wall clock and store contents. Re-running §3.10's
packing analysis on **both arms' performance reports** — same instrument, same 480 slots, same
7 dates — checks the mechanism from the other side, and corrects a plausible-sounding
inference about what the change did.

| | sequential | parallel |
|---|---|---|
| wall clock per date | 166.1 s | 104.2 s |
| task work at PERFECT packing | 34.0 s (20%) | 39.1 s (38%) |
| **residual no worker count can remove** | **132.1 s (80%)** | **65.1 s (62%)** |
| ceiling from unlimited workers, one cell | 1.26× | **1.60×** |
| total slot-seconds | 114,294 s | 131,528 s (**+15%**) |
| source read + resample, share of task work | 72.3% | 71.3% |

**The overlap REMOVED serial time; it did not convert serial time into packable work.** That
distinction matters and the obvious reading gets it backwards. Packed task work barely moved
(34.0 → 39.1 s); what collapsed was the residual, 132.1 → 65.1 s. The windows' critical paths
were never *work* the fleet could have absorbed — they were waiting, and the fix deleted the
waiting rather than parallelising it.

**Two independent confirmations fall out.** The +15% slot-seconds reproduces the overlap's
CPU cost from a second instrument (it was first measured from worker-time totals), and the
stage shares are stable to within a percentage point, so the change did not silently alter
what the graph computes — the same conclusion the identical 44,797 chunk-object counts reach
from store contents.

**§3.10's programme verdict stands, and this is the point of the section.** The ceiling moved
from 1.26× to 1.60×, which is real but modest: **62% of a date still cannot be compressed by
any worker count**, so widening a single cell remains unprofitable and the campaign's lever
is still cell CONCURRENCY rather than cell width. It would have been easy to assume that
halving the residual freed the fleet to scale; measurement says it did not.

**What the remaining 65.1 s is.** Roughly 18.6 s is accounted for by name — build 10.4 s
(client-side graph construction), gate 7.3 s, commit 0.9 s. The other ~46 s sits *inside* the
write phase, which is now a single dask compute: slots stand idle inside it, on dependency
structure and straggler tails rather than on anything serial that remains to be lifted. That
is the next thing worth profiling, and it is a different problem from the one §3.11 solved.

**Caveat on the absolute figures — and it FIRED. See §3.14.** This section said: "A bounded
stream would understate packed work in both arms and so understate both ceilings; the two arms
captured 182,311 and 178,123 tasks, which are neither round numbers nor equal, so truncation is
unlikely but not excluded." **The stream WAS truncated.** Those totals include inter-worker
transfer rectangles, which pad them; the NON-transfer counts are **100,008 and 100,025** — the
100,000 default cap, exactly. The instinct was right and the test was wrong, which is a more
useful lesson than the number it got wrong: a truncation check must count the records the cap
applies to. The paired DELTA does survive, as claimed.

### 3.13 The retained catalogue items — 2.45× smaller, and the first change forced by a LONG run

Everything above was found by shortening a date. This one was found by lengthening a run: a
two-month soak deadlocked at 53 of 60 dates when the worker running the ingest body crossed
Dask's memory pause threshold and never resumed. Failure record: F-19 in the test plan; options
weighed in `yield-embeddings/context_docs/decisions/driver-worker-memory-options.md`.

**Why the driver is the worker that dies.** The ingest body runs as ONE task on ONE worker, so
that worker holds the streamed month plus the prefetched next month **and** its ordinary share of
compute. Measured: per-worker memory settles near **13.45 GiB** (flat across six consecutive
measurement blocks — a real ceiling, not a leak), the driver sits **~4.9 GiB above** the fleet
mean, and 0.80 × 18.63 = **14.90 GiB** is where Dask pauses. It peaked at **14.89**.

**Why spilling could not save it.** **91–100% of worker memory is UNMANAGED** all run — Dask has
nothing it is permitted to evict. Raising the spill threshold, spilling harder, or (as was
considered) raising the pause threshold from 0.80 to 0.90 all fail for the same reason: pausing
frees nothing here, and Fargate has no swap, so a higher threshold removes the guard without
buying runway. **This is why the fix had to delete memory rather than manage it.**

**What was deleted.** Sentinel-2 L2A items from earth-search carry **35 assets**; the ingest
loads **11**. The rest — previews, per-band JP2 variants, metadata documents — were retained for
a whole month and never read. Measured with the shipped code, streaming so no arm ever holds the
unpruned form, each form in a fresh process:

| retained form | KiB/item | vs today |
|---|---|---|
| hydrated `pystac.Item` — the old behaviour | **86.8** | 1.00× |
| drop unused assets + links, keep all else verbatim | **35.4** | **2.45×** |
| also prune properties to an allow-list | 31.8 | 2.79× |

86.8 KiB/item independently reproduces §8's earlier "~80 KB per retained item", measured a
different way — so the baseline is not in doubt. At a dense zone's 68,000 retained items this is
**5.63 → 2.30 GiB, i.e. ~3.3 GiB off the driver.**

**A DENY-list, and the reason is a bug this caught.** The aggressive allow-list variant *failed
on first attempt* with "Failed to auto-guess CRS/resolution", because this collection carries its
CRS in `proj:code` where the list expected `proj:epsg`. That failure was loud. A subtler one —
dropping `raster:bands` scale/offset — would have **changed pixel values silently**. Dropping
whole unread assets cannot lose metadata the loader needs, and it is where 85% of the saving is.
An item whose asset names are all unrecognised is returned untouched, so a differently-named
collection degrades to the old behaviour rather than losing its bands.

**Rehydration is a saving, not a cost.** The query now pages `items_as_dicts()` and builds the
item *after* pruning. Building from a pruned dict costs **252 µs/item against 810 µs** for a full
one — **3.2× cheaper** — because most of an item's construction cost is the assets being dropped.
Pruning itself is 8.7 µs/item.

**Correctness is pinned structurally, not by inspection.** A pruned load is identical to a full
one on band set, dims, dtypes, CRS, transform, geobox and timestamps; nine unit tests pin the
deny-list contract. The OPERA path supplies items through `item_provider_fn` and returns before
any of this, so radar is untouched.

**Shipped alongside: worker memory 20480 → 24576 MiB**, moving the pause threshold to ~17.9 GiB.
Both were needed. Pruning alone leaves every worker ~1.5 GiB below a hard threshold — the margin
that just failed — and **where ordinary compute workers level off is still unmeasured** (mean
9.9 GiB, still rising slowly at 53 dates). Headroom covers what pruning does not reach.

**The durable methodological lesson, and it is the reason this section exists.** A 7-date A/B
cannot see this: the failure appears past ~40 dates. Worse, the first drift analysis was fitted
over samples from *after* the deadlock, when a frozen cluster reads as a plateau, and briefly
concluded "bounded, not a leak" from evidence that could not support it. **Exclude the
post-failure tail before fitting any resource trend, and size memory against a long run's
asymptote rather than a short run's peak** — the 20480 figure was itself chosen against a ~12.4
GiB peak observed on a short run, and the true ceiling is ~15.

### 3.14 The packing ceiling was measured off a truncated stream — it is ~2.8×; and how wide a cell should be

Two findings, in the order they were established, because the second depends on the first.

**The instrument was lying, and the check that should have caught it measured the wrong thing.**
Dask's task stream is a bounded deque — `distributed.scheduler.dashboard.tasks.task-stream-length`,
default **100,000**. At ~25k tasks per date that is about **4 dates**. Both arms of §3.11's A/B hold
**100,008 and 100,025 non-transfer rectangles**: the cap. So each captured the run's last ~3.5 of 7
dates and `perf_budget.py` divided by 7. §3.12 explicitly considered truncation and dismissed it
because the totals — 182,311 and 178,123 — were "neither round numbers nor equal". **Inter-worker
transfer rectangles pad those totals**; the cap applies to the non-transfer records. The caveat was
right and its test was wrong.

| quantity | as recorded (§3.10, §3.12) | **corrected** |
|---|---|---|
| task work at perfect packing | 57.2 s, then 39.1 s | **64.3 s/date** |
| width-independent residual | 73.8 s (56%), then (62%) | **~37 s (~36%)** |
| ceiling, unlimited workers, ONE cell | 1.78×, then 1.60× | **~2.8× (2.0–3.0)** |

**What settles it is two independent instruments agreeing.** The paired 60w/120w width measurement
gives `T = F + K/W` with no reference to any task stream. Fed the corrected packed figure it
predicts T(60w) = **166.4 s** against **167.9 s measured (−0.9%)**; fed the recorded figure it
predicted **141.2 s (−15.9%)**. And the "~30 s unexplained inside the write phase" that §3.13 named
as the next profiling target **was this arithmetic error**, not a phenomenon. Fixed at source:
`perf_budget.py` now warns when the non-transfer count approaches the cap, and `ecs_cluster` takes
`diagnostic_task_stream`, wired to `perf_report_uri` so a report always covers its whole run.

**The budget now closes.** Of a ~102 s date at 120 workers: **64.3 s scalable task work** (71%
source COG read and resample; the profiler puts 61% of in-task time inside `rasterio._do_read`),
and **~37 s that no width touches** — commit and build 12–14 s, gate round-trip ~7 s, writer
assembly and graph submit 5–7 s, plus ~11 s of dispatch ramp and merge tail, consistent with the
single-threaded scheduler's 600–800 tasks/s against ~25k tasks per date.

**Which makes cell WIDTH answerable, and 120 workers is not the answer.** 120 was never chosen; it
is 512 vCPU ÷ 4 minus the scheduler and runner — an artifact of the quota. Fitting the paired
points gives `T(W) = 36.3 + 7896/W`. At a fixed vCPU budget, counting the ~20 vCPU per-cell control
overhead (dask scheduler + flow runner):

| workers/cell | s/date | vCPU/cell | cells @10k | aggregate dates/h | vs 120w |
|---|---|---|---|---|---|
| 15 | 563 | 80 | 125 | 800 | 1.13× |
| **30** | 300 | 140 | 71 | **853** | **1.21×** |
| **45** | 212 | 200 | 50 | **850** | **1.21×** |
| 60 | 168 | 260 | 38 | 815 | 1.16× |
| 120 | 102 | 500 | 20 | 705 | 1.00× |
| 250 | 68 | 1020 | 9 | 477 | 0.68× |

> **CORRECTION — the table above and the two conclusions drawn from it are SUPERSEDED.** Both of
> its inputs turned out to be wrong. (1) The width curve was fitted from **two** paired points (60
> and 120), which cannot constrain a two-parameter model: a third control at 45 workers put `F`
> anywhere from 11.4 to 39.3, and the three-point fit `T = 18.0 + 9391/W` makes aggregate
> throughput **flat within ~6% from 20 to 120 workers**. There is no meaningful optimum, so the
> "30–45 workers, ~20% better than 120" claim is **withdrawn** — prefer whatever width is simplest
> to operate. (2) The per-cell interference of 1.04× came from a single two-cell measurement and is
> **withdrawn**: none is measurable to 20 concurrent cells (§1). The table is kept because its
> *shape* — throughput per vCPU falling as cells widen — survives both corrections, and because the
> reasoning is worth not repeating.

What does survive: **wide cells are better than this record originally claimed and still lose per
vCPU.** 250 workers buys 1.46× per cell, 500 buys 1.89×, unlimited ~2.8×, while throughput per vCPU
falls monotonically. Combined with flat width sensitivity and no measurable interference, the
practical rule is that **topology barely matters — pick the width that is easiest to run** and
spend the quota on more cells.

The remaining open constraint is not contention but **launch**: Fargate sustains **20 tasks/s, 100
burst**, and a 10,000 vCPU fleet is ~2,300 tasks, so about two minutes of ramp unless raised.
Aggregate source-read elasticity at large cell counts is still unmeasured.

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

### 3.17 The window merge exchange rate — priced for a cost that no longer exists

`merge_bands` groups adjacent row bands by minimising `n_windows × WINDOW_COST_IN_CHUNKS +
total_area`, and 200 was calibrated with the constant's own docstring justifying it: "a window
boundary is a serial, blocking region write". §3.11 (`overlap_window_writes`) made that false —
a date's windows now share one graph, so a boundary costs a subgraph, a merge leaf and a
changeset, order 15 chunks rather than 200. Priced at 200 the DP over-merges, trading real ocean
area for a saving that no longer exists.

Swept offline over all 112 real masks (geometry only, no cluster):

| rate | covered chunks | dead | windows | tasks/date | vs 200 |
|---|---|---|---|---|---|
| **200 (was)** | 114,705 | 14.9% | 1,177 | 5.05 M | — |
| 50 | 111,380 | 12.4% | 1,222 | 4.90 M | −2.9% area |
| **20 (shipped)** | 107,872 | 9.5% | 1,346 | 4.75 M | **−6.0% area, +14.4% windows** |
| 10 | 104,975 | 7.0% | 1,560 | 4.62 M | −8.5% |
| 1 | 99,889 | 2.3% | 3,333 | 4.40 M | −12.9% |

Total submitted tasks FALL 6% at rate 20 — area dominates window count — so this does not push
toward the dispatch saturation that §4.2 punished. Unmerged row bands cover 99,847 and the
live-only floor is 97,597, so **12.9% is the entire prize at any rate**; the knee is 10–20
(200→50 buys 74 chunks per added window, 50→20 buys 28, 20→10 buys 13.5).

**Shipped as a CALLER-OWNED parameter, not a lowered constant**, and that distinction is
load-bearing: `WINDOW_COST_IN_CHUNKS_OVERLAPPED = 20` sits alongside the 200 default, and
`live_windows_for_mask` takes the rate. A global change would hand a SEQUENTIAL writer ~1.5 extra
serial windows per zone-date — cancelling its own 6% area saving. So each caller passes the rate
that matches **how it writes**, not what it is: both paths now select on `overlap_window_writes`,
and a caller that turns overlapping off returns to the high rate in the same step.

> This coupling was initially missed on S1. Flipping `overlap_window_writes` to True (§4.9) does
> **not** move the merge rate with it — the two are separate arguments, and S1 was left paying the
> expensive rate while writing cheaply, forfeiting the ~6% area saving. Fixed by having S1 select
> the rate the same way S2 does. The lesson is the one this whole section is about: the rate must
> track the write strategy, so anything that changes the strategy has to change the rate too.

**`DEFAULT_TASKS_PER_CHUNK` is deliberately left at 200** though a chunk really costs ~44 tasks.
That 4.5× conservatism is what keeps the per-window cap at 120 chunks; "correcting" it would
raise the cap to ~545 and reinvite exactly the over-merge §4.2 regressed on. It is also why the
rate is not the binding constraint on dense geometry — 85 of 112 zones already sit within ten
chunks of the cap at rate 200, which is why dropping 200→100 moves area only 0.6%.

Unverified offline: that a window's post-overlap cost really is ~15 chunks rather than ~50, and
that more concurrent region-write buffers in one graph cost no spill (the risk that forced §4.6's
revert). Cheapest settling run: a paired 200-vs-20 A/B on **17N** (predicted −14.7% write volume,
21→28 windows) — NOT 35N, whose predicted 4.5% would drown in noise. Needs a flow parameter to
force the old rate.

**Values are NOT bit-identical across the two block sizes, and this is understood and accepted.**
On Iowa, 78% of shared-data pixels differ — but the median absolute difference is **1** on
reflectances averaging 1,859 (0.05%, p90 = 5), there is **no geometric shift** (unshifted mean
|Δ| 2.0; any one-pixel shift jumps it to 51+), and `scl` differs on only **0.09%**. Cause:
`odc.loader._rio` calls `rasterio.warp.reproject` once **per output chunk** with that chunk's own
`dst_transform`, and GDAL's approximate transformer is fitted over the region being warped — so
the block size perturbs sub-pixel source coordinates, which bilinear turns into ±1 rounding.
`scl` resamples NEAREST, hence its near-immunity: that asymmetry is the signature.

Two controls make this safe to accept: **runs with IDENTICAL config are bit-identical** (0.000%
differing, max 0, over 250k+ pixels on two dates — so the pipeline is deterministic, and the
difference is caused by the change rather than by chance), and the perturbation is two orders of
magnitude below Sentinel-2 L2A's own ~3% radiometric accuracy. Note the same code has a
**bit-exact short-circuit** (`paste_ok and read_shrink == 1`) that skips warping entirely when
source and destination grids align — a same-CRS zone workload should take it, which would make
the campaign path unaffected; that specific prediction is **untested**. Also note bit parity with
`main` was already impossible: main is at `INGEST_CHUNK_SIZE = 4000`, this branch at 4096.

### 3.16 Batching dates — a win at one size, a LOSS at another, so it is sized per region

`batch_dates=k` computes k consecutive PASSING dates as one dask graph and commits them as one
snapshot.

> **CORRECTION, 2026-07-28. The "1.14× and adopt it" reading below is SUPERSEDED.** That figure is
> real but it is one point on a curve that is **not monotonic**, and the same setting measured on
> four further regions *loses* on two of them. Batching is therefore no longer a global setting: it
> is chosen per region by a size threshold (`config.ingest.auto_batch_dates`), and `batch_dates`
> defaults to "derive". The Iowa measurement and the scheduler-CPU ceiling below stand as written;
> what changes is the conclusion drawn from them. See "The shape, and why a threshold" below.

#### The original Iowa measurement

Measured on Iowa, 120 workers, 5 dates, same instrument on both arms:

| | build | gate | write | total/date | commits |
|---|---|---|---|---|---|
| per-date | 0.4 s | 5.1 s | 14.4 s | 19.8 s | 5 |
| batched k=4 | 0.4 s | 4.1 s | 12.9 s | **17.4 s** | **2** |

**1.14×**, and output is **bit-identical** to the per-date path on real data (4 bands × 3 dates)
as well as in the parity test. The write gain is cross-date packing: one date's straggling reads
backfill with another's writes.

> **Methodology warning, and it cost a wrong headline.** The first reading of this A/B put it at
> 1.33× by comparing commit-to-commit intervals. That is invalid: the store's seeding snapshot
> lands AFTER a batch's preparation, so the batched arm's first interval excludes ~20 s of
> build+gate that the per-date arm's intervals include. Compare `Stage timings` against
> `Batch timings` — the same decomposition on both sides — never commit cadence.

**One commit per batch is forced, not chosen.** Every date's append resizes the time axis, so
per-date sessions forked from one snapshot conflict on array METADATA even though their chunk
data is disjoint. Dates are chunk-disjoint in DATA only. Consequence: a mid-batch failure commits
none of its dates and the retry re-ingests exactly the uncommitted ones; `get_existing_dates`
resume is unchanged.

**Resource cost, and where the next ceiling is:**

| | per-date | batched k=4 |
|---|---|---|
| peak worker memory (`wmax`) | 1.6 GiB | 2.81 GiB (of a 20 GiB limit) |
| spill | 0 | **0** |
| peak graph | 5,244 tasks | 14,595 tasks |
| **scheduler CPU peak** | **17%** | **48%** |
| event-loop lag | 0 | 0 |
| queue depth peak | 1.55× | 1.74× |

**The scheduler, not memory, caps k.** Memory extrapolates safely to ~k=20 (14% of limit at
k=4, zero spill), but scheduler CPU is single-threaded and scaled ~linearly: k=6 lands near 72%
and k=8 at or past saturation. The speed case for larger k is also weak — queue depth is already
1.74× at k=4, so the fleet is saturated and extra width buys nothing; only the ~0.7 s/batch of
overhead remains to amortize. **k=4–6 is the useful range; watch scheduler CPU, not memory.**

Next lever from this profile: build+gate is **20.1 s of the 4-date batch's 73.8 s (27%)**, and
the k gates still run SEQUENTIALLY as separate small graphs on an idle fleet.

**Composed with `pipeline_dates` as of 2026-07-27** — the two were refused together only while
the combined memory footprint was unmeasured. They attack different halves: batching removes the
fleet idleness WITHIN a write, pipelining removes the serial preparation BETWEEN writes, which is
that 27%. The look-ahead is sized to `batch_dates`, because a batch's write is one long consume
and the original depth-1 buffer would hide one date's preparation out of k. `pipelined()` gained
a `depth` parameter defaulting to 1, so every existing caller is unchanged (its six original
tests pass untouched); **depth buys BUFFERING, never concurrency** — preparation stays on one
worker in order, so the side-effect-free contract that makes background preparation safe is
untouched. `Batch timings` now carries prepare/hidden/stall in both modes, with the same caveat
as the per-date line: `hidden` is bounded by the SERIAL preparation cost, never by a pipelined
`prepare` figure that contention has inflated. Chosen over fusing the gates because it is a
quarter of the code, touches no existing invariant, and captures 27% rather than the gate's 17%.

Expected memory cost is much smaller than "two batches in flight" suggests: a prepared date is a
LAZY graph plus a scalar, not materialised pixels, so the real increment is the look-ahead
batch's gate computes, which touch one band (`scl`) against the write's eleven — order +10%.
That is a prediction; the 91-date 60-worker A/B on 35N is what tests it, and it is sized to span
three monthly STAC rollovers because those, not the batch buffer, are where retained-item memory
has failed before.

#### The shape, and why it ships as a threshold (2026-07-28)

Five regions, each arm **launched together** on the same dates so time-of-day and catalogue
conditions cancel in the ratio, k=1 against k=4, both arms pipelined so batching alone varies:

| region | live chunks | covered chunks | k=1 | k=4 | ratio | |
|---|---|---|---|---|---|---|
| 26S | 19 | 42 | 10.4 s | 6.5 s | **1.61×** | win |
| 59S | 188 | 493 | 32.8 s | 29.4 s | **1.12×** | win |
| 21N | 644 | 930 | 35.9 s | 50.6 s | **0.71×** | REGRESSION |
| 35N | 2,415 | 2,620 | 175.6 s | 188.6 s | **0.93×** | mild regression |
| 47N | 2,418 | 2,631 | 138.4 s | 136.6 s | 1.01× | neutral — 4 dates only, weak |

35N is the best-powered row: 91 dates per arm. It **supersedes 47N's neutral reading**, which
rested on four dates each. One caveat on 35N specifically — its two arms varied *both* batching and
pipelining (control had neither), so its 7% cannot be attributed to batching alone; what ships on a
region that size is k=1 *with* pipelining, and that combination is untested.

**The arithmetic that explains the shape.** At batch size k the write costs `k·W` and the
preparation running alongside it costs `k·P`, so per-date wall clock is `max(W, P) + commit/k`.
Batching therefore buys **only commit amortisation** — it cannot make the write faster, because the
fleet is already the constraint (§3.10) — and it **loses** wherever the larger write graph crowds
out the concurrent preparation. 21N is that case exactly: at k=1 its preparation already fitted
inside its write with zero stall, and batching disturbed an already-optimal overlap.

**Why a threshold rather than a fitted curve.** A curve through a non-monotonic relation fits
noise. The shipped threshold sits at the **top of the measured-win range**, not at an estimated
crossover, so widening it requires measuring a region in between rather than interpolating.

**Denominated in COVERED window area, and that couples it to §3.17.** Covered area is an *output*
of the window merge, so changing the merge cost moves every region along this axis without anyone
touching the threshold — a finer merge covers less, so regions drift downward and more of them
batch. The offline census in the table above was computed at the sequential merge cost and reads
about 1.25× the runtime value at the overlapped cost (21N: 930 census against 749 measured in a
run). The threshold classifies all five regions correctly either way, but **recalibrate against
runs, never an offline sweep taken at a different cost.**

**Campaign value is modest, and the headline number is misleading.** Regions under the threshold
are 31 of 111 zones — 28% by count but only **1.7% of total live chunk volume**. Weighting by wall
clock rather than volume, since sparse regions still pay fixed per-date costs, the campaign saving
is roughly **2–3%**. The 1.61× applies only to the very smallest regions. Its more valuable
function is preventing a global k, which would have cost 29% on mid-sized regions.

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

Sentinel-1 pays the serial penalties the S2 path had already shed. Two of the three are now closed:

| axis | S2 | S1 before | S1 now |
|---|---|---|---|
| windows within a date | one shared computation (§3.11) | **sequential** — a date costs the SUM of its windows | overlapped, by default (`12d2aed`) |
| window merge pricing | cheap-boundary rate (§3.17) | expensive rate — *correct*, given it wrote serially | cheap rate, once overlapping (§3.17) |
| dates per commit | sized per region (§3.16) | one commit per date | unchanged — still open, see below |

**The symptom that pointed here.** On a sparse zone S1 cost ~39–59 s/date against S2's ~29–33,
**despite carrying 2 bands against S2's 11**. A fraction of the data volume, and slower.

**What we predicted, and what we measured.** The prediction was that turning overlapping on would
transfer the S2 path's 1.59×, and that the gain would be **largest where a date has many windows**
and shrink toward nothing as window count fell. Three ROIs, each run twice at S1 = 30 workers:

| ROI | windows/date | sequential | overlapped | gain |
|---|---|---|---|---|
| 35N | 23 | 172.1 s | 61.8 s | **2.79×** |
| 21N | 9 | 61.9 s | 21.6 s | **2.86×** |
| 59S | 7 | 34.5 s | 14.3 s | **2.40×** |

Bigger than predicted, and — the part that matters — **flat in window count**. Nine windows gained
marginally *more* than twenty-three. That refutes the reasoning behind the prediction.

> **TWO MECHANISM ACCOUNTS HAVE BEEN PROPOSED AND BOTH ARE WITHDRAWN.** The gain is real and
> reproducible; why it is the size it is remains **unexplained**. Recorded in full because the
> pattern of failure is itself the useful part.
>
> 1. **Sum-over-max.** A date's sequential cost is the SUM of its windows and its overlapped cost
>    roughly the MAX, so the gain should scale with window count. **Refuted:** flat across a 3.3×
>    spread in window count (above).
> 2. **Fleet occupancy.** One window's graph is only a few tasks wide, so sequential writing
>    occupies little of the fleet however many windows queue behind it; the gain would then be
>    fleet width ÷ per-window width, which drops window count out (consistent with the table) and
>    predicts the gain should **grow with fleet width**. **Refuted:** a 2×2 on 35N, all four arms
>    launched together, gave **3.67× at 30 workers and 3.85× at 60** — a 1.05× difference, at the
>    5% noise floor. Doubling the fleet did not move it.

**What is established, and it is the operationally important part:** overlapping S1's window
writes buys **2.4–3.9×** on per-date write time, and that gain is **insensitive to both variables
tested** — window count over a 3.3× range and fleet width over a 2× range. An effect that does not
move with either is one we can apply across the campaign rather than only to the zones and widths
it was measured on. That is a stronger operational position than either mechanism would have given.

Two cautions on the numbers. The measured ratio is **what shipping the pair buys, not the overlap
alone**: the merge rate now follows the write mode, so the overlapped arms derive 25 windows to the
sequential arms' 23 and write ~6% less dead area. And because the mechanism is unknown, treat the
magnitude as an empirical range rather than something to extrapolate from — in particular, do not
assume it holds at fleet widths far outside 30–60.

One consequence survives both withdrawals: **narrowing S1's per-date windows is worth doing**
(§4.9 continues below). That case rests on the measured fraction of windows a pass actually
reaches, not on either mechanism.

**The largest remaining S1 win is per-date window narrowing, and it is bigger than expected.**
S2 narrows each date to the windows that date's own imagery can reach (`windows_for_date`); S1
writes the run's **full** land-window set on every date. How much that costs was never measured,
so we measured it offline — public CMR granule footprints against the real masks, grouped by solar
day, no cluster and no credentials needed:

| ROI | live windows | median windows a date actually reaches | share |
|---|---|---|---|
| 35N asc / desc | 25 | 4 / 5 | **16% / 20%** |
| 21N asc / desc | 11 | 1 / 0.5 | **9% / 5%** |
| 59S asc / desc | 12 | 2 / 2 | **17% / 17%** |

A single Sentinel-1 pass covers a swath, not a UTM zone, so **S1 builds and runs 5–20× more
window-writes per date than have any data in them.** Those windows produce all-fill chunks that
are never stored, so the mosaic is identical either way — this is work whose result is already
discarded, the same argument that justified narrowing on S2. Some dates reach **no** live window
at all (the 0% minima), and S1 currently commits those dates rather than skipping them.

**SHIPPED, and measured at 7–20%.** Both zones wrote **six times fewer windows per date**
(11.0 → 1.8 and 12.0 → 2.0), worth **20.1%** and **7.3%** of per-date wall clock at 13 workers.
That partly contradicts the caution written here first: the share figures do bound the *graph*
rather than the wall clock, and the conversion is nowhere near 6× — but it is not negligible
either, and the argument that S1's unsaturated scheduler would make it worthless was wrong.

**The dangerous failure mode was designed out rather than accepted.** Keying each date's
footprint by solar day would disagree with the loader wherever the offset crosses UTC midnight
and would then narrow to the wrong footprint, dropping imagery silently. Instead the join is on
an **exact timestamp**: odc sets each slice's time coordinate to `group[0].nominal_datetime`, the
earliest item's real timestamp, so the minimum item datetime in a group reproduces it exactly. An
unmatched slice writes every window; "reaches nothing" and "we do not know" are separate branches
and only the former skips.

**Dates reaching no live window are skipped unconditionally, and that mattered more than the wall
clock.** On one zone it removed **13 of 58 dates (22%)**. Verified on both sides by re-deriving
footprints from the catalogue: the eight sampled skipped dates reach **zero** live windows despite
carrying **483–1,113 granules each**, while sampled kept dates reach one to three. They are not
no-data days — they are days whose swath covers only the zone's ocean.

> **The skip fixes a latent correctness bug, which is not why it was built.** Some zones have an
> orbit that never reaches land at all: **40S and 24S ascending have granules on ~30 days of 2024
> and reach zero live windows on every one of them.** Writing those dates created a SAR store full
> of fill, which `resolve_s1_orbit` then read as a *present* orbit — so inference consumed an
> all-fill band as real signal, the same hazard the dual-pol granule guard exists to prevent. With
> the skip, nothing is written, no store is created, and the orbit is correctly downgraded to
> single-orbit; the coverage gate follows, since it derives its orbit from the stores present.
>
> Residual case, **not observed** across eight zone-orbits: a zone with some months emptied and
> others not would leave a store missing those months, and the gate counts months present.
> `allow_partial_window` is the designed relief. Worth a census before the campaign.

**The catalogue look-ahead is now SHIPPED and measured — ~10%, and the query hides completely.**
S1 used to pay its whole catalogue query before writing anything. It now prepares the next batch
during the current batch's writes, reusing `ingest._pipeline.pipelined` unchanged — the same
helper the S2 date loop uses — because the phase boundary the S1 timing logs already measured
turned out to be exactly the prepare/consume split it wants: the query holds no credential and
touches no store, so it is safe on a background thread. **One batch of look-ahead only**, because
a batch's write is one long consume and deeper retention is what deadlocked the driver (§3.13).

Measured, four arms launched together, 12 batches each: median stall on every batch after the
first is **0.0 s** against queries of 3.5–23 s, so the query hides *entirely*. Per-date wall
clock fell **10.9%** on a 53-date arm and 8.7% on a weaker 31-date one.

**That is below the 14% query share, and the reason is the useful part: a query run concurrently
gets SLOWER** — 178 → 291 s on one zone (+63%). The background thread contends with the write it
hides behind, so the log's `hidden` field overstates the saving badly (279.7 s claimed against
120.8 s realised). **Treat the paired per-date difference as the result and `hidden` as an upper
bound.** Against the honest ceiling — the *serial* arm's query — the look-ahead captured 74% of
what was available.

It also demonstrates a sequencing rule worth generalising: **the query only became worth
optimising because the overlap made the write several times faster.** At the sequential write
speed the same query was 4–5% of a batch, below the threshold for being worth building. Speeding
up one phase promotes the next, so re-rank remaining work after every win rather than once.

**The look-ahead's gain RISES with density, and the third mechanism is refuted too.** The
prediction was that the saving per batch is `min(query, write)`, so the relative gain should peak
where the two are comparable and be largest on sparse zones. Tested at a genuine 6.25× contrast
(25 live windows against 4):

| zone | live windows | serial | look-ahead | gain |
|---|---|---|---|---|
| 40S | 4 | 2.80 s | 3.15 s | **−12.5%** |
| 21N | 11 | 20.93 s | 18.65 s | +10.9% |
| 59S | 12 | 14.76 s | 13.48 s | +8.7% |
| 35N | 25 | 74.45 s | 63.05 s | **+15.3%** |

Exactly backwards: the gain rises with density and the sparsest zone loses. **Three mechanisms
have now been proposed for this effect and all three refuted** — sum-over-max, fleet occupancy,
and `min(query, write)`. No fourth is offered. Use the measured range and do not model it:
**9–15% where coverage is continuous**, negligibly negative where batches are mostly empty.

Why 40S loses is at least legible: its look-ahead arm *still stalls*, so the query never hid. It
wrote only 4 dates across 4 of 12 batches — most batches held no data, so there was no preceding
write to hide behind and it paid contention for nothing. Absolute cost across the run: **1.4
seconds**. Recorded so nobody "fixes" it.

**S1 is NOT fleet-bound, so narrowing its fleet cuts COST as well as quota.** The 13-worker width
was chosen on the standard assumption that worker-hours are width-independent — halve the fleet,
double the duration, same bill. Measured, that is false:

| zone | 30w | 13w | time ratio | worker-seconds/date |
|---|---|---|---|---|
| 21N | 18.65 s | 24.90 s | **1.34×** (not 2.31×) | 560 → **324**, −42% |
| 59S | 13.48 s | 20.50 s | **1.52×** (not 2.31×) | 404 → **267**, −34% |

A 2.31× narrowing cost only 1.34–1.52× in time, so a 30-worker S1 fleet was paying for idle
capacity. Two consequences. The re-sizing saves **34–42% of S1's compute** where the forecast
assumed zero — though that is a large proportional cut on a base the overlap had already shrunk to
9–16% of worker-hours, so the campaign total moves only ~4%, and longer-running narrow fleets raise
per-fleet scheduler and runner cost enough to offset part of it. And it leaves S1 at roughly **half**
the cell rather than the 83% the width-neutral model predicted, so 13 workers has about 2× margin.

**Do not reuse "worker-hours are width-independent."** Where a fleet is not the constraint,
narrowing is strictly cheaper on both axes, and the only way to know is to measure two widths.

**Also still open on S1.** Date fusing remains the least certain lever: by §3.16's arithmetic it
wins only where the fleet has idle capacity, and overlapping has just consumed much of exactly
that, so the case for it is now *weaker* than before.

S1 was also **uninstrumented** — it reported only "wrote N dates", so a slow batch was
indistinguishable from a slow catalogue, and those have opposite remedies. It now emits per-date
`write` and window `mode`, and per-batch `query` share.

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

### 4.11 A latent S2 correctness bug: two definitions of "a day"

Found by a 20-cell concurrency rung, not by review — and only on one zone of the twenty.

We grouped STAC items by **UTC calendar date**; the loader groups by **local solar day**,
shifting every timestamp by one longitude (its geobox extent centroid in WGS84, truncated to whole
hours). Where the solar offset is large enough to cross UTC midnight the two disagree, so a group we
believed was one day arrived as TWO time slices — against a cloud mask reduced to a single 2-D
slice. The result is a dimension conflict that names nothing about its cause:

```
ValueError: conflicting sizes for dimension 'time':
  length 2 on 'blue' and length 1 on {...}
```

```
        UTC:   ... 23:00 | 00:00  01:00 ...        ONE UTC date
   solar day:       day N |  day N+1               TWO solar days
                          ^ zone 56N images here (+10 h offset, ~00:30 UTC)
```

**Why it stayed hidden.** It is a longitude threshold, not a load or scale effect. Zone 56N spans
150–156°E and images at roughly 00:30 UTC, right on the boundary; central-longitude zones image
mid-UTC-day and can never hit it. Roughly **ten of the 111 land zones** sit in the affected band
(about 01–04 and 55–60), so the campaign would have failed on those and only those.

**The fix** derives the longitude the way the loader does — geobox extent centroid reprojected,
*not* the bbox midpoint, since those can differ by enough to fall either side of a 15° boundary and
the offset truncates to whole hours. Matching by construction is the point: any divergence reappears
as the same conflict. It degrades to the bbox midpoint and then to UTC grouping rather than raising,
because a caller with no geobox is not loading by solar day. The pre-sort uses the same key, since it
carries the painter's-algorithm contract and sorting on a different notion of "day" would silently
let a cloudier pixel win.

**The general lesson, and it is the same one as §4.10:** when two components must agree on a
derived key, derive it from the same source rather than reimplementing the definition. Both bugs
found this month are a local notion of time disagreeing with an external one — a credential's expiry
clock against a batch boundary, and a solar day against a UTC date.

### 4.12 A third definition of "a day": the slice LABEL, and it blocked four zones outright

§4.11 fixed which items form a day. This is the same disagreement one layer further down —
which day the resulting slice is *called* — and it stopped zone 56N ingesting S2 at all.

**It failed two different ways depending on the path**, which is how the scope became clear:

| path | failure |
|---|---|
| batched write | `batch dates must be strictly increasing; got ['2024-01-25', '2024-01-25', …]` |
| unbatched write | `date 2024-01-20T00:00:00 is already on the time axis; refusing a duplicate slot` |

Both **failed closed**, which is why no corrupt data exists anywhere — and the only zone-named
mosaics that exist are `03S` and `40S`, neither far-eastern nor far-western.

**Every link in the chain is individually correct, which is why it survived three fixes in this
family.** S2 sorts a solar day's items cloudiest-first so the clearest tile paints last and wins
the mosaic. `preserve_original_order=True` preserves that against odc's own re-sorting. odc then
stamps each group with `group[0].nominal_datetime` — which is now the CLOUDIEST item, whose
acquisition time is arbitrary within the day. Where the solar offset crosses UTC midnight, that
timestamp's calendar date can be the day BEFORE the solar day it belongs to.

```
   56N, +10 h offset. A solar day spans UTC 14:00 the previous day to 13:59.

   solar 2024-01-20   cloudiest item 2024-01-20T00:52   -> labelled 01-20   ok
   solar 2024-01-21   cloudiest item 2024-01-20T23:45   -> labelled 01-20   COLLIDES
                                     ^^^^^^^^^^^^^^^^
                      late-UTC acquisition, next solar day, previous calendar date
```

**The error was inconsistent, not offset, and that is the part worth remembering.** Six of
twenty-two solar days on 56N landed on the previous date — and *which* six depended on cloud
cover, because cloud cover decides which item is first. So it varied within one zone-year and
would differ on a re-ingest after the catalogue revised its cloud estimates. No adjustment of a
stored axis could have corrected it; re-ingest would have been the only remedy.

**The fix is to take the day from the GROUPING, not from the loaded slice.** The solar day is
the grouping key, and every item in a group shares it by construction, so any member yields it.
Three properties then hold by construction rather than by care: labels are unique per slice,
monotonic across slices (consecutive solar days differ by exactly one day, so the batched guard
passes without sorting), and stable against the catalogue revising cloud estimates.

**Disruption: none beyond the correction itself.** The store's time axis was ALREADY
day-granular — it normalises to midnight — so the label only ever decided WHICH day, and no
consumer could have depended on the time of day. Pixel values, the cloudiest-first ordering,
`preserve_original_order`, and which tile wins a pixel are all untouched. At mid longitudes the
solar day IS the UTC date, so nothing moves for the great majority of zones, which is exactly
why this hid until a far-eastern zone was attempted.

**S1 had the same defect** for the same reason: its items arrive in CMR order, so `group[0]` is
equally arbitrary. It already matched footprints to slices on the exact timestamp (§4.9), so the
solar day travels through that same map and stamps the written slice.

### 4.13 Recording what an ingest EXAMINED, so an absent month is a finding

Related to §4.9's empty-date skip, and forced by it. A mosaic's time axis says what was WRITTEN
and nothing about what was LOOKED AT, so a missing month meant either "examined, nothing
reachable" or "the ingest never got there" — and the coverage gate failed on both. That was
tolerable while every date was written regardless; the skip makes the first case normal.

Both ingest paths now record **`assessed_window`** on the store: the range processed in full.
The gate excuses a month lying WHOLLY inside it and still fails on anything outside, so widening
a window later cannot be excused by an older, narrower assessment, and a partially-covered month
stays an error because it could hide unexamined days.

Written by the ingest rather than by the completion marker, because the marker is written AFTER
the gate on a first ingest, and the fill re-runs the gate later from a separate process.

One attribute suffices, and the reason is worth stating: the ingest queries the catalogue across
the whole window and writes every date with reachable imagery, so absence INSIDE an assessed
range logically implies nothing was reachable. A failed ingest raises before the record is
written, so its presence also means the window completed.

**Every degraded path makes the gate STRICTER** — an absent, malformed or unparseable attribute
excuses nothing, and failing to write it is logged rather than raised. Over-excusing publishes a
mosaic with a hole in it; under-excusing costs a re-ingest. `assessed_empty_dates` rides along
for observability: it separates "sparse region" from "the footprints are wrong", which look
identical in a date count.

**Correction (`9559316`): "both ingest paths now record `assessed_window`" was FALSE for S2 for
as long as the attribute existed.** S2 passed `record_assessed_window` its `store_path` — the
mosaic PARENT directory, which holds all three child repos — where the function needs the
reflectance repo. `open_repo` raised `the repository doesn't exist`, and the catch-and-log above
swallowed it. S1 was correct (`orbit_store`), so nothing looked systematically wrong. Caught by
opening a finished 56N store and finding no attribute on it; the same wrong path had raised the
identical error a minute earlier in the verification script itself.

The failure direction was safe — no attribute means the gate falls back to strict
every-month-present — but that is exactly the behaviour this section exists to remove, so the
feature was inert on the sensor with the most dates. **The lesson is about the design, not the
typo: a deliberately non-fatal write has no failure signal, so the only way to know its path is
right is to READ THE VALUE BACK from a real store.** A completed run proves nothing about it.

**The same family bit twice more, and the second one wedged an ingest permanently.**
`open_or_create_repo` caught `icechunk.IcechunkError` wholesale on its open leg. That class is
the library's catch-all — absence, an expired credential, a throttle, an IO error and real
corruption all arrive as it — so ANY open failure was read as "not there" and sent to create,
which then trips the clean-prefix rule and surfaces as `CorruptedStoreError` naming a store
that is perfectly healthy, whose advice is to delete it. The damage is that it is
**deterministic**: callers wrap these writes in a three-attempt retry that retries every
exception, so a transient failure was already survivable, but once the repo exists every
retry's create fails identically. `is_missing_repo` already existed for this and was already
used at five call sites; this was the one place that did not honour it. TE `abc64f1`.

Then the full chain, traced from one worker's log: a Dask worker's disk filled (§4.14),
`cleanup_on_failure` fired, **the delete was DENIED with `Forbidden` and swallowed**, and the
retry created into the non-empty prefix its own failed cleanup had left. The `Forbidden` was
not an IAM policy gap: fsspec resolves the `AWS_*` environment variables, and the S1 path has
overwritten them with OPERA-scoped tokens for downloading granules, so an ambient delete
authenticated as OPERA against our own bucket. Every other S3 operation in the module threads
credentials explicitly for exactly that reason — the cleanup was the one that did not. Fixed
by threading them, by making `_write_new` ADOPT an interrupted repo rather than re-create it,
and by logging a failed cleanup at ERROR with its consequence. TE `3878827`.

**The pattern across all three: a swallowed exception plus an overloaded absence.** Each time,
something read "I could not tell" as "it is not there" and acted on the wrong one.

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

A cancelled or crashed cell used to be CLEARED and re-ingested from scratch. At campaign scale
that made every interruption cost a whole cell — and interruptions are expected, since the
orphan sweeper cancels runs by design. A dense zone interrupted near the end lost hours,
deterministically, on every retry.

Three states, three answers. **Absent**: ingest. **Present and unmarked**: resume — dates
already committed are skipped, not rewritten. **Present and marked with a DIFFERENT
fingerprint**: raise `ConfigMismatchError`, because neither automatic answer is defensible.
Resuming would append dates admitted under one configuration onto dates admitted under another
and stamp one fingerprint over the mixture; clearing would destroy a complete correct mosaic
because a parameter was mistyped. A campaign holds its inputs fixed, so reaching it means an
assumption broke and a human should choose.

**What makes resume sound is already in the format.** A date's time slot is committed
atomically WITH its pixels, so a date present is complete and a date absent was never started
— there is no partial date to repair. Both ingests already skip dates they find, so resuming
is simply NOT CLEARING: no new marker, no extra read, no extra commit. Verified live on
15S/2024 — all three stores' oldest snapshots predated the re-run, zero duplicate date
commits, and neither pre-cancel radar date was re-written. The radar counts came out at 51
ascending and 60 descending, matching a clean run exactly.

**The campaign's fixed inputs are an ASSUMPTION, documented not checked** — one land mask, one
window per year, one coverage threshold, both orbits, one partial-window policy. With one
exception, because it is the only one whose change would corrupt silently rather than raise:
`RoiManifest` records resolution, chunk size and CRS, so two ROIs built from different coverage
deliveries hash IDENTICALLY. So `IngestManifest` now carries `coverage_sha256`, read from the
ROI store `export_zone_roi` already stamps it on. The manifest is validated on EVERY write, so
a changed mask fails at the first append rather than at a comparison somebody has to remember.

**Two writers is the hazard resume makes reachable, and Icechunk guards it.** These
commits pass no `rebase_with`, so a second writer's commit fails with `ConflictError` against a
moved branch tip instead of merging. Verified directly: two sessions from one tip, the second
refused, the time axis left holding only the winner's date. **That protection is the ABSENCE
of a call**, and `storage.shard_writer.commit_with_rebase` deliberately does the opposite for
disjoint groups — so a test pins the absence, because unifying two commit paths that look
alike would silently interleave two ingests of one store. TE `3878827`, `38d5603`.

> **Correction, 2026-08-03.** This paragraph said Icechunk guarded two writers "**for free**".
> It did not. The write retry had no exception predicate, so it retried the `ConflictError` and
> re-opened the session from the moved tip — `rebase_with` by another route. Protection that is
> the absence of a call is only free if nothing else re-attempts the call. See 4.16.

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

The blocker described in §7. The query ran once per window and retained every item, so a
zone-year needed ~27-30 GB on the 16 GiB worker the ingest body executes on: it died ~17 min in,
Dask retried it four times at ~70-75 min per cycle with the fleet idle, deterministically, on
every dispatch. Streaming bounds retention to the month in flight plus the one buffered behind
it — roughly 5 GB.

**Design, and why each choice is forced rather than preferred:**

- **Partition by owned UTC calendar date.** `group_items_by_date` keys on
  `item.datetime.strftime("%Y-%m-%d")`, so the pipeline's day unit is the UTC date, and a
  calendar-month partition therefore cannot split a day group. Owned ranges tile the window
  exactly once, which means **no cross-month state is needed to deduplicate** — and that matters
  because the worker can be restarted at any instant. A partition survives a restart; an
  ID-dedupe set would not.
- **Pad the query end by one day.** A date-only interval end covers
  only that day's final second, so without the pad items in a month's last moments could fall
  outside every slice. The padded day is *owned* by the next month, so query overlap never
  becomes work overlap.

  *Superseded 2026-07-29:* the pads were originally CLAMPED to the window,
  on the reasoning that a query should never reach outside what was asked for. That was wrong,
  and it cost real imagery. The window is expressed in SOLAR days while a STAC query is bounded
  in UTC, so the two disagree at exactly the window's own edges: the first owned solar day
  includes acquisitions dated the preceding UTC day at eastern longitudes, and the last owned
  solar day acquisitions dated the following UTC day at western ones. Clamped, no slice could
  fetch them, so the first and last day of every zone-year were written with part of their
  imagery missing — invisibly, because `assessed_window` still covered them and the month-level
  coverage gate still passed. The pads are now unclamped. What keeps out-of-window dates out of
  the store is the solar-day ownership filter, which is what it always was; the query bound
  never was that guard and could not be.
- **Depth-1 prefetch on a single-thread pool.** The pool's `max_workers=1` IS the buffer: one
  query in flight, one month in the caller's hands. Depth 1 is sufficient rather than arbitrary —
  a month's query is ~2.7% of a month's processing, so deeper buffering costs memory to hide
  nothing. A thread suits pure network I/O despite the GIL, and a `Future` gives exception
  propagation without a sentinel protocol.
- **A supply-agnostic per-date closure.** Streamed and single-query paths run byte-identical
  per-date work; the per-date logic must not fork on how its items were supplied.

**Validation status:**

| check | result |
|---|---|
| partition property, 6 window shapes incl. leap year and year-crossing | 20 unit tests, no network |
| a failing month raises rather than truncating | pinned (a dropped month would be an incomplete mosaic reporting success) |
| next month genuinely in flight before the current is consumed | pinned |
| **retention bound — earlier months provably released** | pinned by weak reference, with a hoarding negative control |
| **parity vs one whole-window query, live earth-search, across a month boundary** | **12 dates, 13,024 items, identical date sets, zero per-date differences** |
| **month partition and padding, live cluster** | **31,507 items in January, 1,084 correctly deferred to February** (one day's worth) |
| **per-date cost unaffected by streaming** | **mean 173.8 s over 10 dates vs a 184.5 s baseline** — within noise |
| two month transitions with direct submit-time query evidence | PREFETCH proven (February's query submitted 13 ms after January's returned, complete before January's first date committed) |
| cumulative drift over hundreds of dates | **91 dates across three month rollovers, zero spill, peak worker memory plateauing by hour three** (§4.8). Outstanding only at full-year length: twelve rollovers is four times what this exercised. |

**Retention is bounded to two months REGARDLESS of run length**, so a three-month run tests the
memory bound exactly as well as a twelve-month one. That reframes the year soak: it is *not* the
proof of the streaming fix — a three-month run is. What a year adds is **cumulative** effects
(manifest growth, scheduler RSS drift, commit-time growth), which are a separate question.

### The cost streaming does carry: one extra month of retention

Streaming holds the month being processed **plus** the month prefetched behind it. Against a year
that is a large win (~5 GB versus ~27–30 GB); against a **single-month window it is a memory
regression** of about one month's items — and that is the comparison every test run makes.
Measured, same zone/month/fleet:

| run | realignment | streaming | peak spill |
|---|---|---|---|
| no manifest split | on | no | **0.00 GiB** |
| time-split | on | no | **0.00 GiB** |
| realignment removed | off | no | 3.19 GiB (reverted, §4.6) |
| **time-split + streaming** | on | **yes** | **1.25 GiB** |

So the spill I originally attributed solely to removing the realignment has a second source. Both
were real; the realignment revert stands on its own numbers.

**Worker memory moved 16 → 30 → 20 GiB while this was being sized, and is now back at 16 GiB
(§4.8 — read that for the current number and why it became affordable).** Sizing targets the ONE
worker that runs the ingest task, since that is where the retained items live. The vCPU stays at
4 deliberately: the quota is counted in vCPU, so doubling CPU would halve the workers a cell can
run (120 at 8 vCPU would need 960 against a 512 allowance). An attempt at 32768 MiB was **invalid
at 4 vCPU** — check Fargate CPU/memory pairings before shipping a size.

Three lessons from that detour outlived every number in it:

1. **Every worker gets the same size, so the fleet pays for one worker's working set.** 30 GiB
   across a 120-worker cell over-provisions ~119 workers by 14 GiB each. Memory is free in *quota*
   terms, which is what justified it — but it is not free in dollars, and that was not weighed.
2. **The spill was measured to cost nothing**, so eliminating spill was never worth paying for.
   Per-date mean was **173.8 s while spilling 1.25 GiB at 16 GiB** against **177.9 s with zero
   spill at 30 GiB** — the spilling configuration was marginally *faster*. That is structural, not
   luck: the spilled bytes are precisely the bytes not needed yet, because the prefetched month
   goes untouched until the boundary. The whole cost is one read-back per month.
3. **Headroom must buy distance from the PAUSE threshold, not the spill threshold.** A spilling
   worker is fine; a paused one never recovers and deadlocks the fleet. This is the rule §4.8
   sizes against.

The structural fix, still unimplemented: stop sizing a whole fleet for one worker's job. The
ingest body runs on a Dask worker; were it to run somewhere with independent sizing — the flow
runner is a single task — the cost of that memory would fall by the width of the fleet.

**A test-design correction worth keeping:** a one-month run validates nothing here. One month is
a single slice — no prefetch, no boundary crossing. The smallest useful cluster test is three
months.

`stream_stac_monthly` (default True) is the kill switch. `False` restores the single up-front
query and is a rollback path only: a year-long window cannot complete under it.

---

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
