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

**Maintaining this document.** It is the memory of record for this work: update it as findings
land, and preserve the three-depth structure above all — headline table, then mechanism, then
full detail. Corrections go **in place** with the superseded claim named (see §5), never as an
appended log; a reader must be able to trust any section without checking whether a later one
undoes it. New measurements belong in §8; failed attempts belong in §4 with their numbers, not
deleted.

**All wall-clock figures are the same cell** — zone `35N`, January 2024, 120-worker
Dask-on-Fargate fleet (4 vCPU; 16 GiB for figures up to 2026-07-25, **20 GiB after** — see §4.7), the same frozen ROI mask — so the rows compare.
Graph-task figures are from local census runs over a fixed pixel window; those compare within
a series but not across series, and are labelled accordingly.

---

## 1. Headline

Ingest of a dense zone-month went from **not completing at all** to **~170–185 s per date**,
with the graph a date submits down roughly an order of magnitude from the full-extent path and
the scheduler moved from saturated to ~25% busy.

> **THE GRAPH IS NO LONGER THE COST, and a profile now says what is (§3.10).** Of a 131 s date,
> only **57 s is task work even at perfect packing**; the other **74 s (56%)** is serial client
> work, graph dispatch, blocking region writes and the commit, which **no number of workers can
> compress**. Within the task work, **72% is reading and resampling the source imagery** — real
> data movement, not overhead. Two consequences that should be read before planning any further
> ingest work: **the ceiling from unlimited workers on one cell is 1.78×**, and the only lever
> left of that size is removing the 74 s serial residual from the critical path, not shrinking
> the graph further. A 23% graph reduction was shipped and measured at **zero** wall-clock gain
> (§3.9).
>
> **That residual has since been located and largely removed (§3.11).** It was not the commit
> (0.6 s), not graph construction (3.4 s) and not the drive loop (0.2–0.8 s between dates): it
> was `write_day_windows` computing each window to completion before starting the next, so a
> date cost the SUM of its windows' critical paths. Writing a date's windows as ONE dask compute
> took the write phase from **148.0 s → 86.5 s** and per-date from **166.1 s → 104.2 s
> (1.59×)**, with **byte-identical stores** (44,797 chunk objects both arms). Now the default.

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
| 8 | stop realigning blocks to store chunks | 176.5 s | **REVERTED** (see §4.6) | ~4% faster but it introduced worker spill |

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

Seven windows, seven matches. The read-and-mask side — everything §3.3 through §3.7 attacked —
has fused down into the noise. What remains is `store_chunks × n_bands`, and **both factors are
pinned by decisions taken deliberately**: 4096 chunking because the GPU path is tuned around it
(§4.1), eleven bands because that is the data.

This also retro-explains §3.7: there was never a large win available in the write layer, because
the write layer emits one task per store chunk regardless of what it is asked to do.

**The only remaining headroom is dead area: windows cover 2,992 store chunks against 2,415 live
— 577 dead, 19%.** So 32,912 write tasks are issued where 26,565 is the floor. That 19% comes
from a trade §3.7 introduced: not realigning requires windows on the 8192 block grid, and
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

## 5. Claims made and withdrawn

Recorded so they are not revived, and because the pattern is instructive.

- **"Four fifths of every date's graph can be skipped."** The 19.4% data share is real but is
  measured at the 4096 chunk grid, while windows are built on the 8192 load-block grid where a
  date touches 47–57% of live cells. The achievable prize was **2×, not 5×** (§4.7b), and the
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
  quantified. The correct answer at the current geometry is the opposite (§3.7).
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
over-predicted a change by ~20× (§3.7): the region write splits to store-chunk granularity
itself, so a model assuming one write task per block was simply wrong. Where a prediction
depends on what the write layer does internally, measure the live scheduler graph — do not
extrapolate from a census.

**Change one thing per run, or give each change a deterministic mechanism check.** Three
changes shipped together once; when wall clock regressed it could not be attributed from
timings and took an object count to identify.

---

## 7. Open questions

- **~~Is the fleet-bound floor now reached?~~ ANSWERED — yes, and the number is 1.78×.** The
  downward sweep ran (120/60/30, 7 dates each) and the profile settled it (§3.10): of a 131 s
  date only 57 s packs across the fleet, so unlimited workers on ONE cell buy 1.78× and most of
  that is gone by ~200 workers. Sweep medians were 194.8 / 232.4 / 396.7 s, giving 1.71× for
  30→60 and only 1.19× for 60→120 — the knee sits between 60 and 120. **Caveat on those absolute
  values: they are not reproducible** (§5, external latency drift); the SHAPE is what carries.
- **NEW, opened by §3.11: is the 4096 write-window idea now viable?** §4.7 rejected narrowing
  write windows to the 4096 grid because it took windows from 5-6 to 12-13 per date at ~17 s of
  SERIAL cost each. Overlapping the windows removes most of that per-window serial cost, so the
  arithmetic that killed it no longer holds. The remaining costs of more windows are graph size,
  merge work and memory in flight — all smaller than 17 s a window. Worth re-deriving before
  re-testing: it would cut computed area ~1.6× (§4.7's measurement), and area was NOT on the
  critical path when measured (§3.9), so the expected win is small and the memory cost real.
  Measure the prize first, per §3.11's lesson.
- **~~THE open question: can a date's serial residual be taken off the critical path?~~
  ANSWERED — yes, 1.59× (§3.11).** Retained below for the reasoning, which still applies to
  what remains: after the overlap, a date is build 10.4 s + gate 7.3 s + write 86.5 s, and the
  write is now near its packing floor rather than a sum of serial parts. 74 s
  of every 131 s date is serial client work, dispatch, blocking region writes and commit, and it
  is now the largest single target in the ingest by a wide margin. Shrinking it is one route;
  **overlapping it with the next date's compute is the better one**, because it removes the term
  rather than reducing it. Prerequisite question: the ingest is date-serial by construction
  (`_ingest_one_date` per date, one commit per date), so this is a restructuring of the drive
  loop, not a tuning change — and the commit-per-date contract must survive it.
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

## 7b. STAC query streaming — shipped, validation in progress

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
- **Pad the query end by one day, clamped to the window end.** A date-only interval end covers
  only that day's final second, so without the pad items in a month's last moments could fall
  outside every slice. The padded day is *owned* by the next month, so query overlap never
  becomes work overlap.
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
| two month transitions with direct submit-time query evidence | one transition's PREFETCH proven (February's query submitted 13 ms after January's returned, complete before January's first date committed); the CONSUMPTION handoff was not reached before the run was cancelled — the year soak crosses eleven boundaries and covers it |
| cumulative drift over hundreds of dates | outstanding — see below |

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

**Resolution: ingest worker memory 16 GiB → 30 GiB, later CORRECTED to 20 GiB.** Sized for the
ONE worker that runs the ingest task, since that is where the retained items live. The vCPU
deliberately stays at 4: the quota is counted in vCPU, so doubling CPU would halve the workers a
cell can run (120 workers at 8 vCPU would need 960 against a 512 allowance). An initial attempt at
32768 MiB was **invalid at 4 vCPU** and caught before shipping.

**The 30 GiB figure was wrong, and the reasoning behind it was the error.** It reached for the
4-vCPU ceiling to eliminate spill, on the assumption that spill is always a hidden cost worth
paying to avoid. Two things make that false here:

1. **Every worker gets the same size, so the fleet pays for one worker's working set.** 30 GiB
   across a 120-worker cell over-provisions ~119 workers by 14 GiB each. Memory is free in *quota*
   terms, which is what justified it — but it is not free in dollars, and that was not weighed.
2. **The spill was measured to cost nothing.** Per-date mean was **173.8 s while spilling 1.25 GiB
   at 16 GiB**, against **177.9 s with zero spill at 30 GiB** — the spilling configuration was
   marginally *faster*. The reason is structural: the spilled bytes are precisely the bytes not
   needed yet. The prefetched month goes untouched until the boundary, so it is the ideal eviction
   candidate and the whole cost is one read-back per month.

What the headroom actually needs to buy is distance from the **pause** threshold, not from the
spill threshold — a paused worker is a real stall where a spilling one is not. Measured demand is
~12.4 GiB, against a pause threshold of 0.8 × capacity. 16 GiB leaves ~0.4 GiB of margin, which is
too thin; **20480 MiB leaves ~4 GiB** and recovers two-thirds of the over-provisioning.

The structural fix, unimplemented: stop sizing a whole fleet for one worker's job. The ingest body
runs on a Dask worker; were it to run somewhere with independent sizing — the flow runner is a
single task — the cost of that memory would fall by the width of the fleet.

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
| ingest worker size | 4 vCPU / **20480 MiB** | 16384 spilled 1.25 GiB and sat ~0.4 GiB from the pause threshold; 30720 (the ceiling) was over-provisioning a whole fleet for one worker and bought no measured speed |
| streaming retention cost | +1 month of items on the ingest worker; 1.25 GiB spill at 16 GiB | run telemetry |
| items deferred across a month boundary | 1,084 of 31,507 (one day's worth) | live cluster |
| write floor | graph ≈ store_chunks × bands; 2,992 covered vs 2,415 live (19% dead) | measured, all 7 windows |
| **per-date data share** | **582 of 2,992 covered chunks per band-date = 19.4%** at the 4096 grid; **47–57% of live CELLS** at the 8192 window grid | store object count + local footprint measurement (§3.9) |
| **dead-area split** | geometric **55–66%** of live, radiometric (cloud) **10–21%**, written 24% | 3 dates, local (§3.9) |
| **per-date time budget** | task work **57.2 s** at perfect packing of a **131 s** date; residual **73.8 s (56%)** | Dask performance report, 2-date probe (§3.10) |
| **where the residual was** | `write_day_windows` computing windows serially: per-window times summed to **146.9 s of a 148.0 s write phase**; commit only **0.6 s** | 7 dates, per-window instrumentation (§3.11) |
| **window overlap** | write phase **148.0 → 86.5 s (1.71×)**; per-date **166.1 → 104.2 s (1.59×)**; chunk objects **44,797 both arms** | A/B, identical dates and width (§3.11) |
| overlap memory cost | peak worker **12.30 of 20 GiB**, spill 0.00 GiB, no pause | parallel arm health lines (§3.11) |
| overlap packing effect | share of a date that packs **20% → 38%**; headroom from adding workers **1.26× → 1.60×** | per-arm task streams (§3.11) |
| overlap contention cost | **+15% total slot-seconds** for identical output (2% fewer tasks, +25% transfer) | per-arm task streams (§3.11) |
| commit cost over a long run | **sawtooth, period 8 dates** (= `INGEST_MANIFEST_SPLIT["time"]`), 0.5 → ~1.5 s then resets — BOUNDED, not cumulative | 17+ dates, soak (§3.11) |
| **task work composition** | source read+resample **72.3%**, mask+write 16.3%, transfer 7.8%, gate 2.9% | same report (§3.10) |
| **ceiling from unlimited workers, one cell** | **1.78×** | from the budget; independently 1.2–1.7× from an `A + B/W` sweep fit (§3.10) |
| window-strategy bound | best any rectangle strategy achieves is **0.50×** current area; shipped achieves 0.75× | local, real footprints (§4.7b) |
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
