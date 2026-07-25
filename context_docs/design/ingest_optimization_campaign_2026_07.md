# Ingest optimization campaign — July 2026

Authoritative record of the campaign-ingest optimization work: what was changed, how much
each change bought, what was tried and abandoned, and the constraints future work must
respect. The empirical basis for why the ingest path looks the way it does.

Companion to `inference_gpu_saturation_profile_2026_07.md`, which is the same kind of record
for the GPU side. Read that one before touching anything the fill reads — several decisions
here were made *because* of what it measured.

Related notes: `ingest-live-tile-cropping.md` (why ingest crops at all, and how windows are
derived), `ingest-graph-and-stac-budget.md` (the living working notes this summarises),
`region-writes.md`, ADR-011 (campaign zone ingestion).

**All wall-clock figures are the same cell** — zone `35N`, January 2024, 120-worker
Dask-on-Fargate fleet (4 vCPU / 16 GiB each), the same frozen ROI mask — so the rows compare.
Graph-task figures are from local census runs over a fixed pixel window; those compare within
a series but not across series, and are labelled accordingly.

---

## 1. Headline

Ingest of a dense zone-month went from **not completing at all** to **~170–185 s per date**,
with the graph a date submits down roughly an order of magnitude from the full-extent path and
the scheduler moved from saturated to ~25% busy.

| # | change | per-date | cumulative | what it addressed |
|---|---|---|---|---|
| 0 | full-extent mosaics | *never completed* | — | ingest computed the whole zone grid, ocean included |
| 1 | crop to live windows (row bands) | 1,400 s | baseline | ~78% of campaign ingest compute was ocean |
| 2 | group row bands into fewer windows | **225 s** | **6.2×** | per-window cost is near-fixed, so window COUNT dominated |
| 3 | cost-model grouping (3 windows) | 194→269 s | **regression** | over-merged past the scheduler's dispatch ceiling |
| 4 | one masking pass instead of two | — | 1.28× graph | two `where` passes cost a task per (chunk, band) each |
| 5 | coverage gate fused into the band load | — | one fewer graph build/date | SCL was loaded twice per date |
| 6 | decouple load blocks (8192) from store chunks (4096) | **188 s** | **7.5×** | graph scales with block count, not pixels |
| 7 | manifest sharding by time | 185 s | cost, not speed | every commit rewrote the whole manifest |
| 8 | stop realigning blocks to store chunks | ~171 s | **~5% only** (see §3.7) | a local census predicted 3.85×; the live graph barely moved |

Two things did **not** change, deliberately: the store's 4096 chunking (the GPU path is tuned
around it) and the per-date atomicity of a commit (what makes a crashed ingest retryable).

**Feasibility frame.** At ~185 s/date and ~250 kept dates, a dense zone-year is ~11 h against
the ~14 h it was, and ~70 h if it had never been cropped. Sequentially that is ~51 days for 112
zones; with ~20 concurrent cells it is under 3. But the concurrency multiplier is only worth
what each cell's fleet can absorb — which is why the graph work came first (below).

### Why graph size was attacked FIRST, ahead of streaming and cell concurrency

Cell concurrency is the campaign's obvious multiplier — each `(zone, year)` cell brings its own
scheduler, so running many cells at once scales almost linearly and needs no code. Graph size
was fixed first anyway, and deliberately, because **graph size sets the ceiling on how many
workers a single cell can usefully absorb.**

The two compose multiplicatively. Campaign throughput is roughly
`cells_in_parallel × workers_usable_per_cell × per-worker efficiency`. Cell concurrency buys the
first term and quota buys the budget, but a cell that chokes its scheduler at 120 workers cannot
spend a larger allocation no matter how much is available — and every concurrent cell has its
own scheduler hitting the same wall. Optimising concurrency first would have bought the right to
run twenty cells that each waste most of their fleet.

This is not speculative. **Ingest has previously been run at ~250–300 workers and showed
performance degradation attributable primarily to the scheduler.** Today's measurements are the
same phenomenon at the smaller fleet the current quota allows: the scheduler saturates one core,
never exceeds it, and pins at 100% once a window's graph reaches ~84k tasks. Cutting the graph
~3.9× moves that ceiling up by roughly the same factor, which is what makes the concurrency
multiplier worth having.

So the sequencing is: **shrink the graph → raise the per-cell worker ceiling → then spend quota
on concurrency**, with streaming landing in between because year-scale cells cannot execute at
all without it.

---

## 2. The one equation, and why it is the whole story

```
graph_tasks_per_date  ≈  4 × n_bands (11) × area_in_blocks
```

Measured by building a real date's graph and classifying every task by key prefix. The
submitted (culled) graph is ~4 tasks per (block × band): the `odc.stac.load` fetch/mosaic, the
mask `where`, a `getitem`, and the store write. Predictions matched observed cluster graphs to
within 0.5% — a 972-chunk window predicted 42,768 tasks and measured 42,588; a 1,904-chunk
window predicted 83,776 and measured 84,054.

Everything that worked reduced one of those three multiplicands. Everything that failed either
raised the task count or traded it for something more expensive.

**A hypothesis worth recording as dead:** `odc.stac.load` emits ONE `open-<band>` task per
source ITEM (956 for a solar day over a 6° zone), **not** per (item, block). The per-block path
was already lean; there is no elementwise fat to fuse away. Anyone who assumes ~1,000 items
means ~1,000 tasks per block will chase the wrong thing.

### The cost model: a dispatch floor, not a linear sum

```
per_date  ≈  C  +  Σ_windows  max( F , tasks_w / R )
```

- **R ≈ 600–800 tasks/s** — the single-threaded Dask scheduler's dispatch rate.
- **F ≈ 7.04 s** — per-window blocking cost; binds only when a window is small enough that
  `tasks_w / R < F`.
- **C ≈ 15–20 s** — per-date constant on the ingest worker (client graph build, gate, commit).

A linear `n_windows × F + area × V` was fitted first and **failed its first out-of-sample
test** (predicted 122 s at 3 windows; measured 193.9). The `max()` form reproduces all three
window configurations, including the plateau that killed the linear model:

| windows | area | tasks/window | regime | predicted | measured |
|---|---|---|---|---|---|
| 197 | 2,428 | ~540 | latency-bound (F binds) | ~1,407 s | 1,329–1,471 s |
| 15 | 2,563 | ~7.5k | dispatch-bound | ~180 s | 194–242 s |
| 3 | 2,924 | 0.7k/43k/84k | dispatch-bound | ~199 s | 193.9 s |

`V` was never fleet work — it was dispatch.

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

Graph tasks scale with the number of *blocks* the read path builds, not with pixels. So
`INGEST_LOAD_CHUNK_SIZE` (8192) now sets the dask block size while `INGEST_CHUNK_SIZE` (4096)
keeps setting the store's chunking. It must be a whole multiple, enforced at import: the write
splits blocks down to store chunks, and a non-multiple would make that a cross-block shuffle.

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

### 3.7 Removing the realignment — a census that over-predicted, and what it actually bought

`align_chunks` remapped producer blocks to the store's chunks before writing. A local census
predicted a large win, and **the live run did not reproduce it.** Recorded in full because the
gap between the two is the useful part.

**What the census said.** Over one fixed window, counting the optimised graph of each variable
and adding write tasks analytically (one per store chunk with realignment, one per block
without):

| | read+mask tasks | write tasks | total |
|---|---|---|---|
| with `align_chunks` | 1,985 | 528 | 2,513 |
| without | 929 | 132 | 1,061 |

That is 2.37× on the window, and 3.85× against a load-4096 baseline.

**What the run said.** Live scheduler graph, same zone/month/fleet, image digest verified as the
build containing the change: **mean 4,877 tasks per window against v4's 5,159** — about 5%.
Per-date 170.8 s against 179.8 s, also about 5%.

**Why the census over-predicted.** It modelled the write layer *analytically* — assuming
`to_icechunk` emits one write task per dask block when not realigning. It does not: the region
write must produce store-chunk-granular writes regardless, so it performs that splitting itself.
The parameter therefore controls only whether xarray pre-rechunks, and the submitted graph ends
up nearly the same either way. **The census is a sound instrument for the READ side, where it
counts a real graph, and unsound for the write side, where it counted a model.**

**What the change is still worth keeping for.** Peak worker memory fell from **10.16 GiB to
6.86 GiB** — about 32% — because the realignment was materialising intermediate store-chunk
pieces alongside the blocks they came from. That headroom is worth having on its own, and it is
what would make a future block-size increase affordable. Plus ~5% wall clock, and one fewer
graph layer.

**The invariant it introduced is a genuine gain regardless:** `write_day_windows` now raises if
any window is not store-chunk-aligned. Previously a misaligned window would have been rejected
deep in the write or, worse, straddled a chunk with a neighbouring window and made the result
depend on write order.

---

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

## 5. Claims made and withdrawn

Recorded so they are not revived, and because the pattern is instructive.

- **"1.41× from decoupling load blocks."** That was one cherry-picked date. On matched means it
  is **1.23×** (187.9 s against 225 s).
- **"Per-date cost drifts upward within a run."** Visible across the first six dates
  (180.2 → 195.5 mean) and **not** present over twelve: the later half held a single 289 s
  outlier and two of the fastest dates in the run. Unestablished at month scale.
- **"Manifest sharding may explain the drift."** It cannot — the arithmetic puts it at ~1–3 s
  per date. The drift claim was itself withdrawn above.
- **"`align_chunks` is ~11% faster kept."** Measured when blocks and store chunks were the same
  size, so the remap was a no-op and the difference sat inside the ~19% per-date variance since
  quantified. The correct answer at the current geometry is the opposite (§3.7).
- **"Doubling load blocks again gives 2–3×."** Measured 1.35% — see §4.4.
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
over-predicted a change by ~20× (§3.7): the region write splits to store-chunk granularity
itself, so a model assuming one write task per block was simply wrong. Where a prediction
depends on what the write layer does internally, measure the live scheduler graph — do not
extrapolate from a census.

**Change one thing per run, or give each change a deterministic mechanism check.** Three
changes shipped together once; when wall clock regressed it could not be attributed from
timings and took an object count to identify.

---

## 7. Open questions

- **Is the fleet-bound floor now reached?** At ~185 s/date, dispatch accounts for ~55 s and the
  client build ~15–20 s, leaving ~70 s that is plausibly real fetch-and-resample work. If so,
  the earlier ≤1.1× bound on adding workers no longer holds and per-cell width matters again.
  The discriminator is a downward worker sweep (120/60/30 at a fixed plan and fixed dates),
  which needs no quota increase.
- **Is `F` fleet-invariant?** If it is one block's fetch latency it should be; if it is
  scheduler round-trip it grows with fleet size. This decides whether a window cap tuned at 120
  workers transfers at all.
- **Commit time versus manifest size** — the last unmeasured per-date term.
- **Does the scheduler's auxiliary-thread behaviour change at 250+ workers?** Re-check the
  one-core finding before resizing at scale.

Questions genuinely requiring more workers than the current quota allows are tracked as
entries S-1..S-6 in `yield-embeddings/docs/global-tessera-test-plan.md`.

**The known blocker, independent of all of the above:** the STAC query runs once per run and
retains every item, so a zone-YEAR needs ~27–30 GB against a 16 GiB worker and dies ~17 minutes
in — then Dask retries it four times at ~70–75 min per cycle with the whole fleet provisioned
and idle, deterministically, on every dispatch. Month-by-month streaming with depth-1 prefetch
is planned (`yield-embeddings/docs/stac-streaming-implementation-plan.md`); depth 1 suffices
because a month's query is 2.7% of a month's processing.

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
- Per-run provenance belongs in `profiling/inference/RUNS.md`.

The scheduler heartbeat self-reports every 30 s: CPU, RSS, memory percentage, queue lag, worker
and task counts, tasks processing, and fleet memory including spill and the hottest worker.
Read it on every rung rather than waiting to be asked.

For graph work, the cheapest and sharpest instrument is local: build one date's graph, call
`dask.optimize`, and count `__dask_graph__()` keys by prefix. No cluster, no spend, exact
answer.
