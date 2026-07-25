# Ingest graph tasks and the STAC serial phase — living reference

**Status:** living document. Append findings with dates; correct superseded numbers in place
and note the correction. Companion to `ingest-live-tile-cropping.md` (which covers *why*
ingest crops and how windows are derived); this note covers *what limits ingest now* —
graph task count and the flow runner's single thread.

**Why it needs a record:** graph task count has been the recurring failure mode of every
iteration of this stack, campaign and single-ROI alike. Each time it has been rediscovered
from scratch. The point of this file is that the next person inherits the budget rather
than re-deriving it.

---

## 1. The budget equation

```
graph_tasks_per_date  ≈  4  ×  n_bands (11)  ×  area_in_chunks    ≈  44 × area_in_chunks
```

**Measured, not estimated.** One date's real graph was built for a zone geobox and
classified by task-key prefix. The submitted (culled) graph is ~4 tasks per
(chunk × band): the `odc.stac.load` fetch/mosaic, the ROI-mask `where`, a `getitem`, and
the store write. Predictions match observed cluster graphs almost exactly:

| window area | predicted (4 × 11 × area) | observed live graph |
|---|---|---|
| 972 chunks | 42,768 | 42,588 |
| 1,904 chunks | 83,776 | 84,054 |

**`odc.stac.load` emits ONE `open-<band>` task per source item, not per (item, chunk).**
For a solar day over a 6° zone that is ~956 open tasks against ~84,000 total — the
per-chunk path is already lean and there is no elementwise fat to fuse away. An earlier
hypothesis that per-(chunk, item) overlap tasks dominated was **wrong**; recorded here so
it is not re-proposed.

The coverage gate's SCL-only graph is 4,834 tasks (3,876 `scl` + 956 `open-scl`). Cheap.

**The levers are therefore the multiplicands**, and only one has real room:

| lever | effect | cost |
|---|---|---|
| `area_in_chunks` via chunk size | ÷4 for 4096 → 8192 | store-side change reaches inference (§4) |
| the constant 4 → 3 (fold the mask into the load) | ~25% | second order |
| `n_bands` = 11 | fixed — contractual | — |
| window strategy | already at the frontier | spend nothing more |

---

## 2. The cost model: a dispatch floor, not a linear sum

A linear fit `cost = n_windows × F + area × V` (F = 7.04 s/window, V = 0.0346 s/chunk) was
fitted from two runs and **failed its first out-of-sample test**: it predicted 122 s/date at
3 windows and the run measured 193.9 s. Superseded by:

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

**The 15↔3-window plateau is the dispatch floor.** `V` was never fleet work — it was
dispatch. About 86% of a dense date is scheduler dispatch, ~4% is the single-threaded
client build, and the fleet's real fetch/resample/write work hides under both. The
fleet-bound floor is still unknown and only becomes visible once tasks shrink.

**Consequence:** two single threads (scheduler dispatch, the ingest worker) serialise ≥95% of a
run, so **widening one cell with more workers buys ≤1.1×**. Scaling comes from running more
cells concurrently — each cell has its own scheduler — not from wider cells.

---

## 3. Scheduler telemetry

114 heartbeats at 120 workers, 60 with a live graph:

- CPU 19–100%, **twelve samples at exactly 100 and none above**. Ten threads, but never
  more than one core: the Dask event loop is single-threaded and GIL-bound.
- RSS peak 1.79 GiB of an 8 GiB limit (22%), having grown from 0.33 GiB within one month.
- `lag` ≤ 0.3 s throughout; worker spill 0 GiB; hottest worker 6.6 GiB of 16.

**Raising the scheduler's 4 vCPU cannot lift the ceiling.** Signs that WOULD justify a
bigger box, so this is not re-debated: any sustained sample above ~110% CPU (auxiliary
threads genuinely using cores — plausible at much larger fleets, unobserved at 120); RSS
above ~50% of the limit or climbing across a cell; `lag` rising while CPU is below 100%
(contention, not compute — profile instead of resizing). Memory is cheap insurance against
a scheduler OOM killing a multi-hour cell; vCPU is not indicated.

---

## 4. Chunk size: measured variants

Task census over a FIXED pixel window (12 × 4 chunks at 4096, one date, 11 bands), graph
construction only, counting optimised graphs plus one write task per (store chunk, band):

| variant | read+mask tasks | write tasks | total | vs today | reaches inference? |
|---|---|---|---|---|---|
| load 4096 / store 4096 (today) | 2,673 | 528 | 3,201 | — | — |
| load 8192 / store 4096 | 1,749 | 528 | 2,277 | **1.41×** | **no** |
| load 8192 / store 8192 | 693 | 132 | 825 | **3.88×** | yes |

**Writes are emitted per STORE chunk, not per dask block** — the write count is unchanged
at 528 across the first two variants. So coarsening only the load blocks buys 1.41×, and
the full 4× requires coarsening the store. A hypothesis that load-only coarsening might
capture the whole win is therefore **disproved**.

Alignment is fine either way: `SHARD_PX` = 2048 divides 8192 exactly (16 shards per chunk,
32 of the 256-px inner chunks).

### The inference-side constraint on the store change — already measured, and it vetoes it

The GPU-saturation campaign (`inference_gpu_saturation_profile_2026_07.md`) measured this
effect directly while tuning the strip budget:

> Budget 4.75 → 5.75 GiB (P3). Lets the whole `T≤71` full-width band load as a single strip
> instead of two, dropping a ~13 s fixed read/chunk (**source stores use 4000² chunks → every
> read re-decompresses whole storage chunks; measured fixed read ≈13 s regardless of strip
> size**).

So a **fixed per-chunk read cost of ~13 s is already attributed to storage chunk size**. For
a full-width strip the decompressed volume is `W × chunk_px` — linear in the chunk edge — so
4096 → 8192 roughly **doubles** that fixed read, to ~26 s. Tile-shaped reads amplify worse
(4× → 16× of the bytes actually used).

Set that against what the campaign bought: GPU-idle overhead per chunk went **50–60 s →
24–34 s** (striping + background write) → **~6 s median** on prefetch-hit chunks, with ~36 s
remaining only on the first-per-worker cold start. Adding 13+ s of fixed read back would
**roughly triple the steady-state overhead the campaign spent a whole branch reducing**, and
it lands squarely on the resource that is otherwise pinned at 99% utilisation.

There is a second, compounding effect. The strip budget (5.75 GiB) is sized so a full-width
band load fits in ONE strip; larger storage chunks mean fewer rows fit per budget, so more
chunks split into two strips, so more two-strip co-residency — which is precisely the
direction of the 92–95% RAM OOM that motivated striping in the first place. The P3 budget
raise *lowered* peak RAM (51% → 45–47%) by making more chunks single-strip; coarsening the
store pushes the same lever backwards.

**Conclusion: reject the store-side 8192 change on existing evidence.** It trades a 3.88×
ingest graph reduction for a likely 2–4× regression in per-chunk GPU-idle overhead plus
movement toward a known OOM regime, on the most expensive resource in the stack. The
load-only 1.41× variant is unaffected — it leaves the store at 4096, so none of the above
applies.

If the full 4× is ever wanted, the route is NOT a coarser store: it is to reduce ingest
graph tasks by some means that leaves the store's read geometry alone.

### Baseline for any inference-touching change

From the completed 15S/2024 fill (2,352 `RESOURCES` samples, 30.9 GB hosts):

| phase | samples | GPU util mean/median/max | VRAM mean/max | host RAM mean/max |
|---|---|---|---|---|
| infer | 1,876 | **99% / 100% / 100%** | 70% / 97% | 36% / 46% |
| prologue | 158 | 0% / 0% / 0% | 36% / 97% | 29% / 42% |
| load | 24 | 0% / 0% / 0% | 64% / 97% | 34% / 35% |
| (unlabelled) | 294 | 0% / 0% / 29% | 73% / 97% | 12% / 38% |

Host RAM has slack (46% peak against the 60% ceiling; 35% during load). GPU has none: VRAM
peaks at 97% and inference-phase utilisation averages 99% with 98% of samples ≥95%.

**Use the campaign's metrics, not ad-hoc ones.** `te-observe-cluster --ram-report
--log-group /ec2/global-tessera-dev/ray --since … --until …` gives per-worker peak RAM and
GPU-util distribution post-hoc with no live cluster; `--start-pollers` then `--report` gives
1 s DCGM saturation (SMACT/TENSO) and the per-chunk phase-split table from the actors'
`CHUNK_SUMMARY` lines. The decision metric is **GPU-idle overhead per chunk** (~6 s median
shipped), with peak RAM as the safety check. `te-compare-outputs` provides the numerical
parity gate, and per-run provenance goes in
`src/tessera_embeddings/profiling/inference/RUNS.md`.

**Sparse zones understate the harm.** Spatially-sparse chunks already read only a narrow
easting window (optimization C), so a sparse zone shows less amplification than a dense one.
Any inference-side chunk-geometry test must be run on a DENSE area — or a dense sub-section
of a zone — or it will look safe and not be.

### What shipped: load blocks decoupled from store chunks

`INGEST_LOAD_CHUNK_SIZE` (8192) now sets the dask block size for the read path while
`INGEST_CHUNK_SIZE` (4096) keeps setting the store's chunks. It must be a multiple, enforced
at import, because the write rechunks load blocks down to store chunks and a non-multiple
would make that a cross-block shuffle instead of a pure split.

Windows are derived on the load-block grid (`live_windows_for_mask(window_px=...)`) by
coarsening the derived live grid — NOT by snapping windows afterwards, which would be
incorrect: two windows on adjacent fine rows can snap into the same coarse block and stop
being chunk-disjoint, and the single-session per-date write depends on disjointness. The
mask stays chunked at `INGEST_CHUNK_SIZE`, so the fast key-listing path is unaffected.

The window cap is now `MAX_TASKS_PER_WINDOW` (24,000) in graph TASKS, converted to a chunk
area through a `tasks_per_chunk` estimate. Chunk area was the wrong unit: it silently changes
meaning whenever band count or block geometry moves. The old 2,048-chunk cap did exactly
that, permitting an 84,054-task window.

**Separating fixed from per-chunk cost in the census.** The ~956 per-item `open` tasks are
fixed per date, so a small census window overstates the per-chunk figure. Removing them: per
store chunk the variable cost goes **44 to 27.5 tasks (~1.6x)**; the 1.41x headline is that
same effect diluted by fixed cost on a small window, so a full date should realise closer to
1.6x.

**Falsifiable prediction for the validation run** (35N, January, 120 workers), recorded
before the run so it can be wrong: 7 windows of at most 120 blocks; ~12k tasks per window
against 43k; ~83k tasks per date against ~130k; dispatch ~119 s at R around 700/s plus C of
15-20 s, so **~135-145 s per date against 194 s measured**. Scheduler CPU should come off
the pin.

### Evidence that over-merging degrades, not merely plateaus

The 3-window configuration's later dates were **232.9 s and 268.9 s** against 193.9 s for its
first — with scheduler CPU pinned at 100% and dispatch lag reaching 1.2-2.0 s, and the
hottest worker drifting 6.6 to 7.24 GiB. So the plateau was the optimistic reading; past the
dispatch ceiling the configuration gets worse as the store's manifest grows. The 15-window
configuration (mean ~225 s over 8 dates, scheduler 66% mean) remains the best measured before
this change.

---

## 5. The ingest worker's serial phase

### The STAC query

Runs **once per run**, not per date, before any date is processed.

| quantity | value |
|---|---|
| rate | 5.58 ms/item; ~1.34 s per 250-item page |
| one month, one 6° zone | 164 s, 31,507 items, ~1,006 items/solar-day |
| linearity | confirmed 3 → 7 → 14 days |
| one calendar year, extrapolated | **368,248 items, 1,473 pages, ~34 min** |
| resident memory | **~80 KB per retained item** (from 484 / 809 / 1,382 MiB at 2.9k / 6.8k / 14.1k items) |
| a year, extrapolated | **~27–30 GB against a 16 GB flow runner** |

**A year-long query as written exhausts the flow runner before the first date processes.**
This is a feasibility blocker, not a speed problem, and it needs a longer *window* to
observe — not a bigger fleet.

**Page size is capped at 250.** Measured: `limit=500` and `limit=1000` both fail against
earth-search with repeated server errors while 250 succeeds. There is no free win here.

### Streaming, and why prefetch depth 1 is enough

Stream month by month, prefetching month *m+1*'s query on a thread while the cluster
processes month *m*. That bounds retained items to one or two months (~2.5–5 GB) and gives
clean resume.

Depth 1 suffices, and deeper prefetch is waste: a month's *processing* is ~31 dates ×
~194 s ≈ 100 min, while a month's *query* is 164 s — under 3% of it. Only the FIRST
month's query is exposed, and the lever for that is concurrent sub-range queries (weekly
threads → ~45 s), not more buffering.

**CORRECTION (2026-07-25): the memory that matters is a DASK WORKER's, not the flow
runner's.** The whole ingest body — query, retained items, grouping, and the per-date loop —
runs inside ONE Prefect task on a single Dask worker (verified: the "Cropping writes to N
live window(s)" line appears in a `dask/dask-worker/...` log stream; corroborated by the
orphaned fleet committing a further date seven minutes after the flow runner was killed).
Workers are **4 vCPU / 16 GiB**. So a year-long query exhausts a WORKER, and Dask's response
is to kill and restart it, which re-runs the task — a bounded retry loop rather than a clean
failure. Raising flow-runner memory is irrelevant; raising worker memory multiplies across
120 workers. Streaming is close to the only fix.

### Per-date client-side cost

Two `odc.stac.load` graph builds per date — 3.47 s for the gate's SCL-only load and 3.74 s
for the bands — ≈ **7.2 s per date**, single-threaded on the ingest worker while all 480 fleet
task slots idle. Over ~250 kept dates that is ~30 min per zone-year. **The gate load is now
fused into the band load** (one load per date, the gate reading SCL from it), removing one of
the two. What remains is fixable by pipelining one date's graph build against the previous
date's cluster work.

---

## 6. Campaign arithmetic

Assumption, flagged: ~250 kept dates per zone-year (January passed 8/8 at the campaign's
0.1% threshold; 365 is the ceiling).

| scenario | per zone-year | 120 zone-years, sequential |
|---|---|---|
| today (if the query didn't exhaust memory) | ~14 h | ~70 days |
| with the 1.41× safe variant | ~10 h | ~50 days |
| with the 3.88× store change | ~5 h | ~25 days |

Against ~48 days to the 2026-09-11 deadline, before inference. **Cell concurrency is the
multiplier that makes any of these fit** — at ~20 concurrent cells even today's shape is
days rather than months. That is what the pending Fargate quota raise is for; it is not for
wider cells.

---

## 7. Open questions

Tracked here; the ones that genuinely need more workers live as entries S-1..S-6 in
`yield-embeddings/docs/global-tessera-test-plan.md`.

1. **Does the 8192 store change hold the GPU duty cycle?** The decisive question for the
   3.88× lever, and it is a duty-cycle question rather than a memory one (§4). Needs an
   end-to-end small-zone run measured against the 15S baseline: duty cycle 79.8%, infer-phase
   GPU utilisation 99%, load-phase host RAM 35%.
2. **Does the 1.41× load-only variant behave as the census predicts on a real run?** Task
   counts are a graph-construction prediction; confirm live. It cannot touch inference by
   construction (the store is unchanged), so it needs an ingest-side check only.
3. **What is the fleet-bound floor?** Invisible while dispatch dominates; appears once
   tasks shrink.
4. **Commit time growth with manifest size** — the last unmeasured per-date term.
5. **Is `F` fleet-invariant?** Determines whether a cap tuned at 120 workers transfers.
6. **Do the scheduler's auxiliary threads use extra cores at larger fleets?** Re-check the
   "one core" finding before resizing at scale.

Non-issue, recorded so it is not re-raised: **ingest-side commit contention across
concurrent cells**, because each `(zone, year)` writes its own repo under
`…/mosaics/{zone}/{year}/`. The shared-repo concern belongs to the global embeddings store
on the inference side.

---

## 8. Results of record (2026-07-25 session)

Same zone (35N), month (January 2024), fleet (120 workers) and mask throughout, so the rows
are comparable. Per-date figures are means over the dates the run completed.

| configuration | windows | per-date | scheduler cpu mean/max | graph mean/max | hottest worker |
|---|---|---|---|---|---|
| row bands | 197 | 1471, 1329 s | — | 482 / 1,048 | — |
| grouped, greedy | 15 | 225 s (n=8) | 66% / 93% | 17,380 / 22,836 | 5.6 GiB |
| grouped, cost-model DP | 3 | 194 → 269 s, degrading | 65% / **100% pinned** | 50,875 / 84,054 | 7.2 GiB |
| **load blocks 8192, store 4096** | **7** | **187.9 s (n=12)** | **27% / 40%** | **5,069 / 6,020** | 8.0–10.3 GiB |

**Per-date variance is 19% (SD 36 s on n=12).** Detecting a 10% change needs ~14 dates per
arm, so **verify further graph work by TASK COUNT, not wall clock** — the count is
deterministic and readable from a single date's graph.

Two claims made during the session and then withdrawn, recorded so they are not revived:

- *"1.41× from the load-block change"* — that was one cherry-picked date. On matched means it
  is **1.23×** (187.9 against the greedy configuration's 225 s).
- *"Per-date cost drifts upward within a run"* — visible in the first six dates (180.2 → 195.5)
  but **not established over twelve**: the later half contains a single 289 s outlier and two
  of the fastest dates in the run. Unproven at month scale; a manifest-driven drift would only
  be expected at year scale anyway.

### Verified by task count rather than wall clock

Collapsing the two masking passes into one, counted on a real window's optimised graph:

| | tasks per load block |
|---|---|
| two passes (validity, then ROI) | 99.9 |
| one pass (`validity & ROI`) | **77.9** |

**1.28× fewer graph tasks**, matching the predicted ~25%. This is the pattern to follow for
the remaining shaves: the mechanism is measurable exactly, the wall-clock consequence is not.

## 9. Manifest sharding: the axis matters more than the size

Same zone, month and fleet; two dates each, so directly comparable.

| configuration | date 1 | date 2 | manifest objects | manifest bytes |
|---|---|---|---|---|
| no split | 179.5 s | 147.8 s | — | ~3.3 MB (fitted) |
| `{northing:4, easting:4, time:8}` | **245.8 s** | **281.3 s** | **10,195** | 3.85 MB |
| `{time: 8}` | 179.8 s | 146.8 s | **161** | 1.99 MB |

The spatial split cost **30-50% wall clock** — not from bytes but from object count: ~5,097
manifest objects rewritten per commit against ~14, and the PUT latency dominates. Time-only
restores no-split speed while already writing fewer manifest bytes, and it grows LINEARLY with
dates where unsharded grows as N-squared over 2.

**The rule, generalised:** split the axis along which a single commit is NARROW. Campaign
ingest commits one date across every live window, so it is narrow in time and wide in space —
the exact inverse of the region-write merge workload the module default was written for. Both
regimes are now documented at `storage/zarr_store.py`'s split constants.

Reference for the unsharded cost this removes: ~1.1 MB of chunk references per date, so a
250-date zone-year rewrites ~35 GB of manifest cumulatively and ~275 MB on its final commits.
That is ~1-3 s/date, i.e. splitting is mainly an **S3 traffic and cost** win rather than a
wall-clock one — worth having because it is free, not because it is fast.

## 10. Task-count levers: what is left, and what is ruled out

Current: **4 tasks per (chunk x band)**, ~44 per output chunk, 77.9 per load block.

| lever | expected | LOC | effort | status |
|---|---|---|---|---|
| Load blocks 8192 to 16384 | area in blocks divided by 4, so **~2-3x** on load-side terms | ~5 | 30 min | **next** — memory per task x4 (134 to 536 MB per band); raising worker memory is sanctioned if the gain is real |
| Drop `align_chunks`, rely on windows already landing on load blocks | ~1 of 4 per chunk-band (**~25%**) | ~20 | 2-3 h | **after** — measure, do not assume: dropping it was 11% SLOWER when blocks and chunks matched |
| Coarsen `INGEST_CHUNKS["time"]` beyond 1 | none per date; fewer commits | ~10 | — | **REJECTED** — breaks the property that a date's time slot lands atomically with its pixels, which is what makes a crashed ingest safely retryable |
| Fold the ROI mask inside `odc.stac.load` | ~1 more per chunk-band | 80+ | 1-2 d | **REJECTED** — no clean hook; high risk for a second-order gain |

On worker memory for the 16384 experiment: the hottest worker already reached 10.3 GiB of 16
at 8192 blocks, so 16384 is expected to press it. Raising the worker size is an accepted trade
if the task-count gain is real — the extra instance cost is small against the fleet time saved
— but the gain must be demonstrated by task count first, and spill must be zero.

## Changelog

- **2026-07-25** — Created. Graph anatomy measured (4 tasks per chunk-band); linear cost
  model superseded by the dispatch-floor model; scheduler shown single-core-bound; chunk-size
  variants measured (1.41× load-only vs 3.88× full); STAC query characterised including the
  year-scale memory blocker and the 250-item page cap; 15S fill baseline extracted — host RAM
  has slack (46% peak, 35% in the load phase, 60% ceiling) but GPU does not (VRAM 97% peak,
  infer-phase utilisation 99%), which reframes the store-change risk from memory to GPU duty
  cycle (baseline 79.8%).
