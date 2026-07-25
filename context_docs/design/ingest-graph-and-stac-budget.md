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
- **C ≈ 15–20 s** — per-date flow-runner constant (client graph build + gate + commit).

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

**Consequence:** two single threads (scheduler dispatch, flow runner) serialise ≥95% of a
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

### The inference-side constraint on the store change

Zarr fetches whole chunks, so a 2048-px tile read decompresses 4× its own bytes today and
would decompress **16×** at 8192 store chunks. Per band-timestep that is 134 MB fetched to
use 8.4 MB. Whether that binds depends on how much of a chunk each fetch actually uses.

Baseline from the completed 15S/2024 fill (2,352 `RESOURCES` samples, 30.9 GB hosts):

| phase | samples | GPU util mean/median/max | VRAM mean/max | host RAM mean/max |
|---|---|---|---|---|
| infer | 1,876 | **99% / 100% / 100%** | 70% / 97% | 36% / 46% |
| prologue | 158 | 0% / 0% / 0% | 36% / 97% | 29% / 42% |
| load | 24 | 0% / 0% / 0% | 64% / 97% | 34% / 35% |
| (unlabelled) | 294 | 0% / 0% / 29% | 73% / 97% | 12% / 38% |

**Host RAM has slack; GPU has none.** Host peaks at 46% against a 60% operating ceiling —
and only 35% during the load phase, which is where read amplification would land. The 60%
ceiling is an operational rule adopted because spikes appeared at very large scale that
small and medium tests did not show.

GPU is the opposite: VRAM peaks at 97%, and during inference **utilisation averages 99%
with 98% of samples at or above 95%**. The path is saturated by design — it has been
optimised hard because GPU workers are by far the most expensive part of the stack.

**So the risk of the store change is not memory — it is GPU duty cycle.** Read
amplification lengthens the non-inferring phases, and every second added there idles a
resource that is otherwise pinned at 100%. The metric that decides the change is therefore
the share of samples in the inference phase:

```
baseline GPU duty cycle = 1,876 / 2,352 = 79.8%
```

A store-chunk change that holds duty cycle and per-tile load time is safe; one that
depresses either is paying the most expensive resource in the stack to wait for I/O, and
should be rejected however much it saves on ingest. Memory is the secondary check, not the
primary one.

---

## 5. The flow runner's serial phase

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

**Note on which memory matters:** the query result is held by the **flow runner**, not the
Dask scheduler. Scheduler memory is irrelevant to it. Runner memory is the knob, and
streaming is preferred over raising it because the 4 vCPU Fargate shape caps at 30 GB —
zero margin against the ~29 GB estimate.

### Per-date client-side cost

`_load_scl_only` 3.47 s + `load_stac_items` 3.74 s ≈ **7.2 s per date**, single-threaded on
the runner while all 480 fleet task slots idle. Over ~250 kept dates that is ~30 min per
zone-year. Fixable by pipelining one date's graph build against the previous date's cluster
work, and by fusing the coverage gate into the band load so only one graph is built.

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

## Changelog

- **2026-07-25** — Created. Graph anatomy measured (4 tasks per chunk-band); linear cost
  model superseded by the dispatch-floor model; scheduler shown single-core-bound; chunk-size
  variants measured (1.41× load-only vs 3.88× full); STAC query characterised including the
  year-scale memory blocker and the 250-item page cap; 15S fill baseline extracted — host RAM
  has slack (46% peak, 35% in the load phase, 60% ceiling) but GPU does not (VRAM 97% peak,
  infer-phase utilisation 99%), which reframes the store-change risk from memory to GPU duty
  cycle (baseline 79.8%).
