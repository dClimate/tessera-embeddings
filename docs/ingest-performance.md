# Ingest performance: what bounds a run, and the levers against it

An ingest is not bounded by the imagery. It is bounded by the Dask scheduler: a single process
that holds the entire task graph in memory before a worker reads a byte, and that hands out every
task through one event loop, on one CPU. Overwhelm it and it does not simply run out of memory.
That one CPU saturates, the scheduler slows at handing out work, and hundreds of machines sit
idle waiting for something to do.

## The problem, and the shape of the answer

A zone-year over a large region describes millions of tasks. At roughly 1.5 KB each the graph can
exhaust the scheduler's memory before a pixel is read. Well short of that it pins the single CPU
doing the dispatching: past some graph size, tasks go out more slowly than workers finish them,
the queue drains, and the fleet waits on a CPU-bound scheduler. **A graph that is too large buys
idleness, not throughput.**

A graph that fits can still leave the fleet idle, for an unrelated reason: write one window at a
time and every machine waits through each write.

Every lever here does one of two things:

- **Keeps the graph small,** so the scheduler can dispatch fast enough to keep the fleet fed:
  per-date iteration on optical, windowed batching on radar, and cropping to live windows, which
  on a sparse zone removes most of the extent.
- **Keeps the fleet busy** once the graph fits: overlapping a date's window writes, preparing the
  next date behind the current one, and fusing several dates into one compute.

The two interact, which is why several levers are priced rather than switched on: filling idle
slots by enlarging the graph can slow the scheduler enough to empty the fleet again. Figures are
measured, derivations in `context_docs/ingest/campaign-ingest-measurements.md`.

## What bounds the run

Three facts a reader needs before any of the levers make sense: what the scheduler actually
spends memory on, how the chunk grids line up, and what the write contract is that every lever
has to preserve.

### Background: how Dask task graphs consume scheduler RAM

"Dask is lazy" means workers do not read data until `.compute()`; it does not mean the
scheduler avoids work. Before the first worker task executes, `dask.distributed` expands the full
`HighLevelGraph` into a flat dictionary of `TaskState` objects in the scheduler process, each
holding one task's function, arguments and dependency set. The cost:

```text
    scheduler RAM used ≈ n_tasks × 1.5 KB
```

This is fully predictable and independent of data size. A graph with 1 million tasks consumes
~1.5 GB of scheduler RAM before any worker reads a single byte. Nothing is read until the Zarr
write triggers the compute, so a date's whole pipeline — load, correct, mask, write — is one
graph execution and the scheduler holds all of it at once.

**RAM is the limit that kills a run, but it is not the one that shows up first.** The process
holding the graph is also the one assigning every task, and it does that on a single CPU through
one event loop. Every assignment costs CPU time, so scheduler CPU — not worker capacity — becomes
the ceiling as the graph grows.

Past that point the fleet finishes work faster than the scheduler can hand more out. Workers go
idle while the scheduler's CPU sits at 100%, and nothing in the run is short of memory: it is
simply paying for machines that are waiting on one saturated core.

`MAX_TASKS_PER_WINDOW` and the per-date iteration below are sized against that dispatch
throughput, not against the RAM figure above.

The HLG itself is compact — it stores *layer dicts* rather than expanded objects. The
expansion to TaskStates happens only when the graph is submitted to the scheduler:

```text
  Python process (builds the HLG)          Dask distributed scheduler process
  ─────────────────────────────────         ─────────────────────────────────────
  Layer "zarr_read"                         TaskState("zarr_read",(t=0,y= 0,x= 0))
    { (t=0,y=0,x=0): read_fn, ... }         TaskState("zarr_read",(t=0,y= 0,x= 1))
  Layer "baseline_corr"                     TaskState("zarr_read",(t=0,y= 0,x= 2))
    { (t=0,y=0,x=0): corr_fn, ... }         ...
  Layer "roi_mask"                          TaskState("baseline_corr",(t=0,y=0,x=0))
    { (t=0,y=0,x=0): mask_fn, ... }         ...
  Layer "zarr_write"
    { (t=0,y=0,x=0): write_fn, ... }       one TaskState object per task,
                                            all held in scheduler RAM simultaneously
  4 compact Python dicts ≈ tens of MB      n_tasks × 1.5 KB of scheduler RAM
```

#### How task count multiplies: operations × chunk dimensions

Each Dask operation (read, transform, write) adds a new layer. Each layer has one task per
combination of *chunk coordinates* across all chunked dimensions. Ingest writes 4096×4096 px
storage chunks (`INGEST_CHUNK_SIZE`), deliberately larger than the 2048×2048 inference
read-tile size, precisely to keep this task count down. For S2 ingestion over a large ROI
(e.g. cornbelt scale: ~38×25 grid of 4096×4096 px spatial chunks), the S2 flow processes one
date at a time, so the time dimension is always 1:

```text
  Dimensions per single S2 date (ingest_s2_roi_reflectance):
    spatial chunks:     ~950  (38×25 grid of 4096×4096 px)
    dates:                 1  (one date per loop iteration)
    band variables:       10  (each S2 band is a separate xarray variable)

  Operation            tasks per layer                          notes
  ────────────────────────────────────────────────────────────────────────
  odc.stac read       950 × 1 × 10 =   9,500   one task per (chunk, band var)
  baseline corr.      950 × 1 × 10 =   9,500
  ROI mask            950 × 1 × 10 =   9,500
  zarr write          950 × 1 × 10 =   9,500
                                    ─────────
        per-date total:                38,000 tasks ≈ 0.06 GB scheduler RAM   ✓

  Full year (all ~100 S2 dates in one graph, hypothetically):
  950 × 100 × 10 × 4 = 3,800,000 tasks ≈ 5.6 GB scheduler RAM                ✗ OOM
```

Had ingest reused the 2048×2048 inference tile size, every count above would be 4× higher —
that 4× on the satellite-ingest graph is the reason storage and inference chunk sizes are
decoupled.

The two flows keep task count bounded via different mechanisms — per-date iteration for S2,
time-windowed batching for S1 — described in the sections below. The same scheduler-RAM
discipline reappears in inference assembly; see
[`inference/README.md`](../src/tessera_embeddings/inference/README.md#1-deciding-which-tiles-to-run) for the
ChunkSpec-vs-sub-chunk decoupling that makes assembly survive on the same budget.

### Chunk alignment

The ROI Zarr mask is generated with `chunk_size` matching `INGEST_CHUNKS`, so each Dask
partition maps to exactly one Zarr chunk and `da.from_zarr` reads are **zero-copy**. The same
chunk sizes go to `odc.stac.load` (after translating `northing`/`easting` to `y`/`x`), so the
band arrays and the mask share partition boundaries.

**Alignment is worth more than it sounds.** When partitions do not line up with stored chunks,
Dask has to rechunk: it reads several stored chunks to assemble each partition, allocates a new
buffer for the result, and copies the overlapping pieces in. That adds a task layer to every
graph it touches, which is exactly the cost the section above is spent avoiding, and it holds
two copies of the data — the chunks read and the partition built from them — at the moment
memory is tightest. Aligned, a partition *is* a stored chunk: nothing is reassembled, nothing is
copied, and the graph gains no layer. Misalignment costs graph size, worker memory and wall
clock together, which is why the mask writer and the loader are given the same chunk size rather
than each choosing a sensible one.

Inference is the exception, and deliberately so: it reads 2048×2048 sub-tiles out of these
4096×4096 chunks through `zarr.Array.oindex`, which needs no alignment — see
[`inference/README.md`](../src/tessera_embeddings/inference/README.md).

### Writing a date: one session, one commit

`storage.write_day_windows` owns it. A missing store is seeded all-fill (schema only — creation
cost independent of extent), then each date appends its time slot atomically WITH its windows in
**one commit**:

```text
per passing date (one writable session ── one commit)
   ├─ append time slot            (metadata-only resize; duplicate date = loud error)
   ├─ to_icechunk(region=window₁) ┐  pixels flow from the Dask workers that
   ├─ to_icechunk(region=window₂) │  computed them, never materialised on
   ├─ ...                         ┘  the flow runner
   ├─ merge attrs                 (baselines ∪, doy ++, last_appended)
   └─ commit                      (crash before here ⇒ nothing visible; retry is clean)
```

Two properties follow from that shape.

**The time axis only ever lists dates whose pixels committed.** That is what the empty-axis seed
buys: a date appears in the axis only after its write succeeded. Three things read the axis and
trust it — `get_existing_dates` (the STAC dedupe), `check_time_window_coverage`, and the
empty-timestep prunes — and all three would be wrong if the axis could name a date whose pixels
never landed.

**A failed write must not be retried if a second writer caused it.** Retrying a failed write is
normally free, because a write that did not commit changed nothing. Concurrent writers are the
one exception.

A store is meant to have exactly one writer, and the code enforces that rather than assuming it:

- These commits pass no `rebase_with`, so a commit that races another is **refused** instead of
  merged (`icechunk.ConflictError`).
- A date the other writer got to first is refused by the append guard (`DuplicateDateError`).

Retrying either refusal is what breaks it. The retry re-opens the session from the tip the other
writer just moved, so the second attempt succeeds — and the two writers interleave their dates
onto one axis. Both errors are therefore excluded by type in
`storage.zarr_store.store_write_retrying`, which is the single retry policy all three write sites
use: S1 per-date, S2 per-date, and S2 per-batch.

## Keeping the graph small

The two flows start from different baselines, and the two cropping steps then cut what either
of them has to build.

### S2: per-date iteration

`ingest_s2_roi_reflectance` queries STAC for the full date range upfront, groups items by **local
solar day** via `group_items_by_date`, then processes one day at a time in a Python loop: each
iteration builds a single-date Dask graph, calls `odc.stac.load` for that day, filters coverage,
and writes before moving on.

```text
Full year (don't build at once):
┌──────────────────────────── 365 days ────────────────────────────────┐
│ tiles × dates × bands = O(millions of tasks) → scheduler OOM         │
└──────────────────────────────────────────────────────────────────────┘

Per-date iteration (what ingest_s2_roi_reflectance actually does):
 2024-03-01    2024-03-06    2024-03-11    ...
┌────────────┐ ┌────────────┐ ┌────────────┐
│ build      │ │ build      │ │ build      │
│ SCL check  │ │ SCL check  │ │ SCL check  │
│ compute    │ │ compute    │ │ compute    │
│ write      │ │ write      │ │ write      │
│ discard ◄──┼─┼── graph freed after each date
└────────────┘ └────────────┘ └────────────┘
```

Each single-date graph is small: `spatial_chunks × bands` tasks, with no date dimension to
multiply through, and the per-date overhead of one Python loop iteration and one Zarr append is
negligible beside the Dask compute for a large spatial ROI.

The grouping key must match the loader's, per [Timestamp handling](../src/tessera_embeddings/ingest/README.md#timestamp-handling-solar_dayspy). Grouping
here by UTC calendar date lets the two disagree, and a group we believe is one day then loads as
TWO time slices against a cloud mask reduced to one:

```text
   UTC:      ... 23:00 | 00:00  01:00 ...      ONE UTC date
   solar:        day N |  day N+1              TWO solar days   (at a +10 h offset)
                       ^ far-eastern zones image right here
```

### S1: time-windowed batching

`ingest_s1_roi_sar` uses a different approach: it splits the full date range into
`batch_days`-wide windows (default 30) and runs one `build → compute → write → discard`
cycle per window, which bounds how large any one task graph gets.

```text
Batched approach (batch_days=30, ingest_s1_roi_sar only):
 Jan 1–30        Feb 1–28        Mar 1–30       ...
┌────────────┐  ┌────────────┐  ┌────────────┐
│ build      │  │ build      │  │ build      │
│ compute    │  │ compute    │  │ compute    │
│ write      │  │ write      │  │ write      │
│ discard ◄──┼──┼── graph freed
└────────────┘  └────────────┘  └────────────┘
```

A batch boundary is **not** a credential checkpoint, and treating it as one is unsafe: the STS
credential's roughly one-hour life is unrelated to how long a batch takes, so a batch that outruns
it cannot renew at its own boundary. Renewal is owned by a timer — see "Renewal runs on a timer"
above.

`batch_days` is a parameter on `ingest_s1_roi_sar` and is absent from the S2 flow. The formula
in the background section above estimates how many tasks a given window width produces; the
30-day default keeps each batch inside the scheduler's RAM budget at cornbelt scale.

Batch windows are inclusive at both ends and do not overlap: each spans `batch_days` calendar
days and the loop advances `batch_start` to the day *after* `batch_end`. Since CMR and STAC also
treat their end date as inclusive, each day is queried by exactly one batch — a boundary landing
on the next batch's start day would page it twice, wasteful where each day is many pages of
bursts.

### Cropping to live windows (unconditional)

Ingest cost scales with the **extent it computes, not the land it keeps**, and a mosaic load
covers the whole ROI grid even where the mask is entirely ocean or out of footprint. Every
load and write is restricted to the chunk-aligned windows that intersect the ROI mask,
unconditionally and with no flag: on a zone where land is 0.238% of the extent, uncropped is
~420× the array volume, which exhausts a worker's disk. Across the campaign's land zones roughly
three-quarters of the compute would otherwise go to ocean.

```text
zone / ROI extent (declared grid — UNCHANGED, the fill validates it)
┌──────────────────────────────────────────────┐
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │   · ocean / out-of-footprint:
│ · · · ┌────────────────────┐ ·  ·  ·  ·  ·   │     never loaded, never computed,
│ · · · │████████████████████│ ·  ·  ·  ·  ·   │     never written (reads back as
│ · · · └────────────────────┘ ·  ·  ·  ·  ·   │     fill — Zarr elides all-fill
│ · · · · · · ┌────────┐ ·  ·  ·  ·  ·  ·  ·   │     chunks anyway)
│ · · · · · · │████████│ ·  ·  ·  ·  ·  ·  ·   │
│ · · · · · · └────────┘ ·  ·  ·  ·  ·  ·  ·   │   █ live windows: chunk-aligned,
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │     4096 px, derived below
└──────────────────────────────────────────────┘
```

Windows are derived in **two stages**, and the second is where most of the win is. Each cell
below is one 4096-px ingest chunk; `#` is live, `.` is ocean.

**Stage 1 — row bands.** One window per live chunk-row, spanning that row's first to last live
column. A row's interior gaps are included; nothing above or below it is. This is the
minimum-*area* answer.

**Stage 2 — grouping.** Vertically adjacent bands are unioned into taller windows. That
computes some dead chunks and is still a large net win, because the two costs differ:

```text
  chunk area                          a window boundary
  ──────────                          ─────────────────
  computed in PARALLEL across the     a BLOCKING region write: the whole
  whole fleet — adding chunks         fleet waits while one completes.
  widens the graph, which is           N windows = N serial stalls.
  what the fleet wants.
```

The objective is `least n_windows × price + area`, where the price is one window expressed
as the chunk area costing the same (`WINDOW_COST_IN_CHUNKS`). That price is large, so grouping
pays whenever it is geometrically sane.

The merge runs **twice** — once over the run's live grid, once over each date's narrowed grid,
and both must use the same price, so `windows_for_date` takes it as a parameter. Priced
at the sequential default while the run used the overlapped rate, the per-date re-merge would buy
dead area back to save boundaries the write path had already made cheap.

```text
     live chunk grid        stage 1: row bands        stage 2: grouped
     c0 c1 c2 c3 c4 c5      c0 c1 c2 c3 c4 c5         c0 c1 c2 c3 c4 c5
r0    .  #  #  #  .  .       .  A  A  A  .  .          .  a  a  a  +  .
r1    .  #  #  #  #  .       .  B  B  B  B  .          .  a  a  a  a  .
r2    .  #  #  #  #  .       .  C  C  C  C  .          .  a  a  a  a  .
r3    .  .  #  #  #  .       .  .  D  D  D  .          .  +  a  a  a  .
r4    .  #  #  #  #  .       .  E  E  E  E  .          .  a  a  a  a  .
r5    .  .  .  .  .  .       .  .  .  .  .  .          .  .  .  .  .  .
r6    #  #  .  .  .  .       F  F  .  .  .  .          b  b  .  .  .  .
r7    #  #  .  .  .  .       G  G  .  .  .  .          b  b  .  .  .  .
r8    .  .  .  .  .  .       .  .  .  .  .  .          .  .  .  .  .  .

     # live chunk          7 windows, 7 stalls       2 windows, 2 stalls
     . never written       A..G one per live row     a = r0-r4 x c1-c4
                           r5/r8 have no live        b = r6-r7 x c0-c1
                           chunk, so no window       + = dead chunk the
                                                         union pulled in
```

Group `a` covers r0–r4 at the cost of two dead chunks (`+`), buying four fewer stalls. Whether
`b` joins it is the same question, and at the production price it would — real masks group harder
than this illustration suggests. Two things stop a group: the price ceasing to justify the area,
and `MAX_TASKS_PER_WINDOW`, which bounds one window's graph. That cap is expressed in TASKS and
converted to a chunk area through `DEFAULT_TASKS_PER_CHUNK`, because the scheduler dispatches on
a single-threaded event loop — past its throughput, extra area stops being cheap.

Grouping is solved **exactly**, by a dynamic program over consecutive bands rather than a greedy
rule: a heuristic bound on wasted area cannot express "extra area is nearly free", so it
under-merges precisely on the sparse ROIs where the waste is trivial. Windows stay chunk-aligned
and mutually chunk-disjoint either way, letting one session write a whole date and commit once.

Across the campaign's zones this lands every one in 2–5 windows, saving between 2× and 66× of
the stage-1 write boundaries for between 0% and 50% added area, and cutting predicted per-date
ingest cost by about **11×** summed over all land zones. Sparse ROIs group *harder* in relative
terms, which is the point: a large fraction of a tiny area is still a tiny area. Calibration of
the price, the per-zone table and the cap sweep are in
`context_docs/ingest/campaign-ingest-measurements.md` §13.

- **Windows** come from `live_windows.py`, from the boolean ROI mask that both
  `rasterize_roi_zarr` and `export_zone_roi` write. The mask is coarsened to the ingest chunk
  grid — normally from its chunk keys in one listing, else by scanning one chunk block at a
  time (~16 MB peak, no Dask) — then row-banded and grouped as above.
- **Writes** go through `storage.write_day_windows`, one commit per date — see
  *Writing a date: one session, one commit* above.
- **Reads retry, per date**, and a failed date says which date and which ROI. See
  [ingest-error-handling.md](ingest-error-handling.md#where-the-retry-sits-and-how-a-failed-date-is-attributed).
- **Each date narrows further, to the land its own imagery reaches**, via `windows_for_date`.
  See *Narrowing a date's windows, and skipping dates that reach none*.

```text
   run windows (where the ROI has land)   one date's items      that date writes
     c0 c1 c2 c3 c4                        c0 c1 c2 c3 c4        c0 c1 c2 c3 c4
r0    a  a  a  a  .                         .  F  F  .  .    r0   .  a  a  .  .
r1    a  a  a  a  .          ∩              .  F  F  F  .  = r1   .  a  a  a  .
r2    a  a  a  a  .                         .  .  .  .  .    r2   .  .  .  .  .
r3    b  b  .  .  .                         .  .  .  .  .    r3   .  .  .  .  .

     2 windows, every date                F = the swath           1 window; window b
                                            this date covers        is skipped entirely
```

- **The declared grid stays full-extent**, so the zone fill's exact-grid validation is
  unaffected — Zarr/Icechunk arrays are sparse and unwritten chunks read back as fill. That
  validation checks the declared grid COMPLETELY rather than just its corners, because matching
  length, CRS and endpoints still admit a reordered or non-affine interior, and inference writes
  positionally, so such a mosaic would publish real pixels at wrong coordinates silently. Uniform
  10 m spacing is asserted on both axes. Nothing this ingest produces can fail it, since odc
  builds every load against the zone geobox, but a fill run with `ingest=False` accepts a mosaic
  staged by hand.
- **The SCL coverage phase is cropped too**: its reduce runs over the windows only
  (identical total — the mask is False outside them) and the validity mask stays lazy,
  so no full-extent array is ever persisted.
- **Worker sizing follows**: with cropping on, `ingest-zone-year` sizes `max_workers`
  from the cell's live-chunk count (0.5 workers/chunk, clamped) instead of granting a
  4-tile zone the same fleet as a dense one.

Unconditional at every layer; the full-extent write path is gone from both modules along with
every branch that tested for it. S1 and S2 share the mechanism, differing only in that S1's
multi-date batches loop per date, since non-contiguous dates cannot share a region write, each
keeping its own atomic commit and retry scope.

### Narrowing a date's windows, and skipping dates that reach none

A run's live windows are the same on every date; one date is not, since a satellite images a
fraction of a wide ROI per pass, so most windows hold nothing for it. `windows_for_date`
intersects the run's windows with that date's own STAC footprints — reprojected onto the ingest
grid and padded one cell, so a curved reprojection cannot under-cover — then re-bands and
re-groups. Tasks over the removed windows would run, find nothing and write nothing, so this
cannot change what a mosaic contains. When a footprint cannot be determined the full window list
is returned unchanged, so the conservative path is the fallback. Both sensors do it
(`narrow_windows_per_date` on S1, always on S2): six times fewer windows per date on the S1 zones
measured, worth 7–20% of per-date wall clock.

**A date whose imagery reaches NO live window is skipped entirely**, on both paths and
unconditionally. Writing it builds a full graph to store nothing. On S1 this is not a rare case:
one zone skipped 13 of 58 dates, and some zones have an orbit that reaches land on *no* date of
the year. Skipping those creates no store, letting `resolve_s1_orbit` downgrade to single-orbit
rather than publishing a store of fill that inference would read as real signal.

**The safety rule, and it is the whole design.** A footprint that is too LARGE only costs
computed area that would have been discarded; one that is too SMALL drops imagery and nothing
downstream notices. Every uncertain path widens rather than narrows: an unreadable footprint
returns the full window set, and on S1 a time slice that cannot be matched to its items writes
everything. "Reaches nothing" and "we cannot tell" are separate branches — only the first skips.

S1's match is on an **exact timestamp** rather than a date string, because odc sets a slice's
time coordinate to its group's earliest item timestamp. Keying by solar day instead would
disagree with the loader wherever the offset crosses UTC midnight.

## Keeping the fleet busy

A graph that fits can still under-use the fleet: windows written one after another leave most
slots idle, and the client-side work between dates is time no worker spends on anything. These
three fill that, and the last of them is not a straight win.

### Overlapping a date's window writes (`overlap_window_writes`)

A date is written as several chunk-disjoint windows. Writing them one at a time dominates the
cost of a date: each window's compute completes before the next begins, so the date costs the
**sum** of the windows' critical paths while the fleet idles through most of each.

```text
Sequential windows — the fleet sees one window at a time:
 │◄─ window 1 ─►│◄─ window 2 ─►│◄─ window 3 ─►│◄ w4 ►│◄─ window 5 ─►│
 └─ the date costs the SUM of these, and most slots idle within each ─┘

Overlapped (overlap_window_writes, the default) — one graph, one commit:
 │◄─ window 1 ─►│
 │◄─ window 2 ──►│     all submitted together, so the fleet packs them and
 │◄─ window 3 ─►│      the date costs roughly the LONGEST window plus
 │◄ w4 ►│              whatever the total work itself requires
 │◄─ window 5 ──►│
 └── one merge, one commit for the date (contract unchanged) ──┘
```

Mechanism: icechunk's dask path already forks a session, stores lazily and merges changesets,
and writing per window runs that sequence once per window. Overlapping lifts it one level — fork
once, collect every window's lazy stored arrays, run one merge reduction — so every window's
loads, masks and chunk writes occupy a single graph.

The resulting store is identical either way, because the windows are chunk-disjoint: that is what
makes the merged changesets conflict-free, and the same property that lets a date commit exactly
once. Should icechunk's internals move, the write falls back to the sequential loop with a
warning.

Default **on** for both S2 and S1. `write_day_windows` itself still defaults to the
sequential path: a storage-layer default should not decide write strategy for its callers,
so each ingest path opts in explicitly.

The gain is **2.4–3.9×** on per-date write time and varies with neither window count nor fleet
width. Why it is that size is not explained — three accounts were proposed and all three refuted
by their own predictions — so rely on the measured range, do not model it, and do not extrapolate
far outside the widths measured. `context_docs/ingest/campaign-ingest-measurements.md` §3.11 and
§4.9.

### Pipelining a date's preparation (`pipeline_dates`)

A date's wall clock splits into **preparation** — building the load graph, running the coverage
gate, narrowing the footprint, constructing the masks, and the **write**. Preparation is part
client-side CPU, independent of fleet width, and part cluster compute, since the coverage gate
reads SCL on the workers. Only the client-side part is serial residual a wider fleet cannot
shrink.

**The overlap's payoff is therefore not symmetric.** Hiding the client-side part behind the write
is free; hiding the gate is not, because it is fleet work and on a saturated fleet competes for the
same slots regardless of scheduling order. The overlap pays in proportion to the spare capacity
the write leaves, making it **more** valuable on narrow fleets than wide ones.

`pipeline_dates` prepares date N+1 on one background thread while date N is written
(`ingest/_pipeline.py`). The write stays serial: icechunk commits are sequential on a branch and
one commit per date is the contract, so the store has exactly one writer either way. Preparation
must be **side-effect-free**, touching nothing but the dataset it hands back, and that is what
makes the two modes produce identical stores — pinned by a parity test including a date that fails
the coverage gate mid-run.

Depth is 1, intrinsically rather than by tuning: preparation is a small fraction of a write, so
buffering more would hold graphs in memory to hide nothing. The pipeline lives inside one `_drive`
call and drains at each streamed month boundary, leaving one unhidden preparation per month.

Each written date logs `Pipeline date=…: prepare=… hidden=… stall=…` in both modes. `stall` is
the preparation the write could not cover, and is the health metric: near zero when preparation
hides fully, rising toward the whole preparation when the gate is starved behind the write's own
tasks. Serially every date stalls for its full preparation, so the two modes compare from one
line. **`hidden` is not a saving** — pipelined, `prepare` is background-thread wall time spanning
the whole concurrent write, so it inflates with contention; take any A/B's expected saving from
the control arm's `prepare` and treat `hidden` as a contention diagnostic.

Default **off**, and the flag threads from the outer flow through the task shell to the
domain function. S1 has no coverage gate and a different batch loop; it is deliberately
untouched.

### Batching dates into one compute (`batch_dates`)

Normally each date is computed and written on its own. This fuses several consecutive dates into
one computation and one write. The number fused is `k`.

**It is not a free win, which is why the default sizes it per region.** With `batch_dates=None`
the size comes from how much live land the region covers (`config.ingest.auto_batch_dates`).
Passing an integer forces a fixed size, which is what an A/B comparison needs. Batching helps
small regions, makes little difference on large ones, and **costs about 29% on mid-sized ones**,
so no single value is right across the range.

**Where the uneven shape comes from.** Two things happen at once while a date is written: the
write itself, spread across the fleet, and the preparation for that date, running alongside it.
Whichever is slower sets the pace. On top of that, each date pays for one **commit** — the single
operation that makes it visible to readers.

```
per-date wall clock  ≈  max( W, P )  +  commit / k

    W = the batch's write, per date          k = dates fused into one graph
    P = the preparation running alongside it, per date
```

Fusing `k` dates divides the commit cost by `k`, and that is the entire gain. It cannot make the
write faster, because the machines are already fully occupied. And it can lose: a bigger write
graph leaves less room for the preparation running beside it. On a mid-sized region the
preparation already fitted inside the write with nothing left over, so there was no idle capacity
to win back and only the crowding to pay for.

So batching pays only where the fleet has spare capacity to absorb a larger graph. The threshold
is set at the largest region where that was actually measured to hold; raising it means measuring
a region in between first.

One coupling to watch. Because the size is derived from covered area, the window merge above
changes it: a finer merge covers less area, so more regions fall below the threshold and get
batched. Recalibrate against real runs rather than an offline sweep, since a different merge cost
gives a different answer. Figures in
`context_docs/ingest/campaign-ingest-measurements.md` §3.16.

**What happens when it is on.** `k` consecutive dates that pass the quality gate are computed as
one graph, and their work interleaves — while one date waits on slow reads, another's writes keep
the machines busy. The tail at the end of a computation, where the last few tasks finish and the
fleet drains, is paid once per batch instead of once per date.

**The batch has to be the commit unit; this is not a choice.** Adding a date lengthens the
store's time axis, and that axis is shared metadata. Two dates committed separately from the same
starting point would collide on it even though their pixels go to different places
(`storage.zarr_store.write_days_windows`).

The consequence is all-or-nothing: a failure part way through commits none of the batch's dates,
and a retry re-ingests exactly the ones that never landed. The resulting store is byte-for-byte
identical to the one-date-at-a-time path, held by a test that puts a gate-failing date in the
middle of a batch.

Three practical details:

- A date that fails the gate does not take up a slot, so batches stay full even in cloudy
  stretches where most dates are rejected. Whatever is left over is written at each month
  boundary.
- Logging changes. Instead of one `Stage timings` line per date there is one `Batch timings` line
  per batch. Its build and gate figures are sums of the real per-date values; the write is a
  single shared computation and cannot be split per date.
- The default is 1, which is the one-commit-per-date path, unchanged.

**Combining it with `pipeline_dates`.** The two fix different idleness. Batching fills gaps
*inside* a date's write; pipelining removes the serial preparation *between* writes. Used
together, the look-ahead is sized to the batch rather than to one date: a batch's write is one
long operation, so a buffer holding a single prepared date would hide only one date's preparation
out of `k`.

Preparation still runs on one thread at any depth, so the rule that it must not touch anything
beyond the dataset it returns is unchanged. The cost is memory — up to `k` prepared dates held
while `k` more are written. `Batch timings` reports `prepare`, `hidden` and `stall` per batch,
with the same caveat as the per-date line.

## has_new_stac_dates pre-check

**Not yet wired into any flow** — this section describes something unbuilt, kept because the
reasons are worth having written down. `has_new_stac_dates` is meant to run before provisioning a
Dask cluster: it queries the STAC catalog and checks for new dates without reading any raster data
or starting Fargate tasks, so a flow could exit early when nothing is new.

An overlapping date range is no longer a correctness problem:
each ROI ingest begins the day after the newest date its store holds, and a window wholly below
that line returns a skip without querying (see [Where a resumed run starts](ingest-error-handling.md#where-a-resumed-run-starts)). What the pre-check
would still buy is avoiding the cluster, since the skip is decided inside the ingest and the
caller reaches it only after provisioning one. Tracked in
[issue #47](https://github.com/dClimate/tessera-embeddings/issues/47); when wiring it, do not
share one OPERA `item_provider_fn` between the pre-check and the real query, because the provider
re-queries CMR on every call.

