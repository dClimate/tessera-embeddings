# Global TESSERA campaign — the plan

**The standing plan for building 9 years × 112 land zones of 10 m embeddings into one
Icechunk store.** Everything here is settled. One measurement is still outstanding before launch,
and it is item 8 of §7.

This document is the entry point and the source of truth for **operations** — what runs, with
what settings, in what order, what comes out, and what to do when it breaks. It states the
current position rather than how that position was reached: a finding that changed a decision is
folded into the section it changed, and the measurement behind it lives in the document named
beside it (§11).

[`campaign-cost-model.md`](campaign-cost-model.md) is the source of truth for **figures**:
every cost, rate, fleet size and the arithmetic behind them. Numbers appear here as results
with a pointer, never as derivations; where the two disagree, the cost model is right.

---

## Campaign overview — what runs, per zone

**A UTM zone is a 6°-wide north-south strip of the globe**, the standard way satellite imagery is
projected onto a flat grid. Each zone gets its own map projection, which is what keeps distances
and areas honest inside it. There are 120 of them, and the campaign fills the **112 that contain
land** — the other 8 are all ocean (`10S 11S 13S 14S 27S 44S 45S 46S`).

Each zone is processed independently, one calendar year at a time. A **cell** is one zone in one
year, so the campaign is 112 zones × 9 years ≈ **1,008 cells**, over 360,953 live 2048-pixel tiles.
Every per-cell cost multiplies by that 112, so the count is worth being exact about:
`scripts/rank_zones.py` reads it from the mask, and it is the authority if the mask is ever
rebuilt.

Every cell goes through the same four steps:

```
  1. INGEST (three, in parallel)          2. INFERENCE           3. ASSEMBLY        4. VALIDATION
  ┌──────────────────────────┐
  │ Sentinel-2  optical      │──┐         read the mosaics,     write the tiles    figures + a
  │ Sentinel-1  radar, asc   │──┼──▶ 3 ──▶ embed each 2048 px ──▶ into the store ──▶ machine-readable
  │ Sentinel-1  radar, desc  │──┘  mosaics  tile on a GPU        as whole shards    verdict
  └──────────────────────────┘             ↓                          ↓                 ↓
     a year of imagery, on CPU        staged tiles in S3        one zone-year of    published beside
     (Fargate/Dask)                   (the intermediate)        the final store     the data
                                                                 + a write-once tag
```

1. **Ingest** builds three mosaics for the cell — one optical and one per radar orbit — from public
   catalogues. CPU work, and the only step that touches the outside world.
2. **Inference** reads those mosaics and runs the TESSERA encoder on a GPU, tile by tile, writing
   each result to a staging area in S3.
3. **Assembly** collects the staged tiles into the published store as whole shards and tags the
   cell, which is what marks it complete. The mosaics are then deleted; they are transient.
4. **Validation** re-opens the published cell, renders figures, and writes a verdict saying whether
   it is sound. It runs as its own flow and does not hold up the next cell (§5).

The rest of this document is how 1,008 of those are scheduled, what they cost, and what happens
when one fails.

---

## 1. Shape


```
   a naive run             ingest then inference     ~12 days     5.6 PB of mosaics at once
   the campaign             ingest alongside it       ~5.1 days    ~340 TB at once
```

**~5.1 days is the campaign's wall clock.** It is 61 cells of ingest feeding 2,500 GPU actors
across 10 clusters, with every year dispatched in one batch — the settings of §3, and the only
configuration this plan describes. The campaign is **GPU-bound**: ingest finishes its 1,008
zone-years in 4.2 days and the fleet needs 5.1 to consume them, so the schedule is
`307,854 GPU-hours ÷ 2,500 actors` and nothing about ingest moves it. §6 says why those numbers.

Two design properties get it there, and neither is an optimisation the campaign could drop.

**Ingest and inference are staged, not sequential.** Ingest builds a zone-year's mosaic on
Fargate; inference consumes it on GPU and the mosaic is deleted. They use different resource
pools, so overlapping them costs neither side. And because mosaics are transient, what is held at
once is `cells in flight × ~5.6 TB` rather than the whole 5.6 PB (5.6 PB over 1,008
zone-years). Run sequentially, that full volume is held at once and storage alone becomes about
**$128,000 a month** instead of ~$3,000.

**Every year is dispatched in one batch** (`overlap_years=true`), **so there is no year barrier.**
A cluster works a multi-year zone list, which makes the makespan `total work ÷ cells` rather than
`max(longest zone, work ÷ cells)` *per year*. That is what lets cell count and GPU quota buy wall
clock at all: with the barrier, cells stop helping at about 45 and no quota purchase moves the date.
It also costs 10 `ray up` cycles instead of 90, worth ~$8,000 of the ramp line — a saving, not the
reason; the schedule is the reason.

Within that, work is organised as **10 long-lived Ray clusters**, each owning a set of UTM zones
and streaming them through one persistent set of 250 GPU actors. A cluster pays `ray up` and model
load once for its whole set, rather than once per zone-year.

```
   campaign
   └── 10 clusters x 250 actors, each opening on one of the 10 densest zones
       ├── cluster 1 ── 35N 2017…2025 → 12S 2017…2025 → 04N …   (densest zone first)
       ├── cluster 2 ── 38N 2017…2025 → …
       └── … every requested year dispatched as one batch
```

> **Why running years together is safe.** Concurrent zone-years cannot collide in the data:
> different groups rebase cleanly, and even two years of the same zone write strictly disjoint
> objects, because every chunk and shard is 1 in the time dimension.
>
> No zone-disjointness scheduler is needed: **the partition is over zones**, so a zone's every
> year lands in one cluster, where assemblies serialize on that cluster's single trailing
> thread. That is also what bounds concurrent assembly memory to one cluster's worth.
>
> **Each cell carries its own year.** The months to read travel with the work item rather than
> with the GPU actor, which is built once and reused for every cell that passes through it. That
> is what lets one cluster mix years at all: without it, a 2019 cell handed to an actor set up for
> 2025 would be embedded over the wrong twelve months, and nothing would have said so.

---

## 2. What comes out — the store a downstream user opens

**One Icechunk repository, 120 UTM-zone groups, 9 annual timesteps, a 128-dimensional
embedding per 10 m pixel.** The seed creates 120 zone groups, of which about 112 hold land, and
the whole store is 0.9–1.8 PB — destined for AWS Open Data. Read-only consumers need `icechunk` plus `zarr`
(v3); beyond that it is ordinary Zarr with named dimensions and coordinate arrays, so `xarray`
opens it with no custom reader.

```
s3://<bucket>/global/tessera.icechunk          ← one repository, branch `main`
├── (root attrs)  geoemb:dimensions=128, geoemb:data_type=int8, geoemb:gsd=10.0,
│                 geoemb:model=<encoder URL>, spatial_layout=utm_zones
├── 01N/                                       ← one group per UTM zone, 120 of them
│   ├── embeddings      (time, northing, easting, band)  int8     ← the payload
│   ├── scales          (time, northing, easting)        float32  ← the dequantizer
│   ├── s2_obs_count    (time, northing, easting)        uint16   ← optical depth
│   ├── s1_asc_obs_count / s1_desc_obs_count             uint16   ← radar depth, per orbit
│   ├── time (9) · time_bnds (9,2) · northing · easting · band    ← coordinates
│   └── (zone attrs)  crs=EPSG:32601, years_complete=[…], runs={…},
│                     time_convention=calendar_year, proj:* / spatial:*
├── 02N/  …
└── 60S/
```

**Grid.** Each zone group is its own UTM CRS on a fixed 10 m grid, snapped to the 2048-px shard
pitch, so a zone is about **933,888 × 67,584 px** (~63 Gpx per year, pole to pole and
across the 6° zone). `northing`/`easting` are coordinate arrays in projected metres and the
affine transform is on the group as `spatial:transform`. Nothing is reprojected or mosaicked
across zones: **a study area crossing a zone boundary reads two groups** and is the consumer's
join. Zones overlap slightly, which is what makes that join possible.

**Time is exactly nine calendar years**, 2017–2025, one slot each. A `time` point is Jan 1 of its
year, and `time_bnds` states the real half-open interval, because the slot holds a whole Jan–Dec
window rather than an instant. The axis is fixed at seeding: nothing appends years.

### The one thing every reader must do

`embeddings` is int8 and meaningless on its own. The real value is `embeddings * scales`, broadcast
over the band axis. There is one scale per **pixel**, taken as that pixel's largest absolute value,
so each pixel's largest dimension sits on the ±127 rail.

```python
import icechunk, xarray as xr

repo    = icechunk.Repository.open(icechunk.s3_storage(bucket=..., prefix=...))
session = repo.readonly_session(branch="main")          # or snapshot_id=<a tag's snapshot>
ds      = xr.open_zarr(session.store, group="33N", consolidated=False)

# One pixel's 128-d vector, dequantized. Two small reads, one chunk each.
px = ds.isel(time=8, northing=500_000, easting=30_000)
vector = px.embeddings.astype("float32") * px.scales                 # (band: 128)

# A 2048-px tile for one year, dequantized in one expression.
tile = ds.isel(time=8, northing=slice(0, 2048), easting=slice(0, 2048))
values = tile.embeddings.astype("float32") * tile.scales             # broadcasts over band
```

`scales` is **NaN wherever no embedding exists**: ocean, a tile with no valid observation all year,
an unfilled year. That NaN is the authoritative "no data" mask: multiplying through it
propagates, so a consumer who dequantizes never silently reads fill as zero. (`embeddings`
itself fills with 0, which is a legal value; the counts fill with 0, which is a legal count.
`scales` is the only one whose fill cannot be mistaken for data.)

### The effective API — what is cheap and what is not

Chunking decides this, and it was chosen for the per-pixel case:

| read | cost |
|---|---|
| one pixel's full 128-d vector | **one object.** The band axis is never split, so a vector is never assembled from parts |
| a 256-px chunk, all bands | one object, ~8 MB logical |
| a 2048-px tile, one year | 64 objects — a shard, which is also the unit inference wrote |
| the same pixel across 9 years | **9 objects.** Every chunk is 1 in time, so a time series costs one read per year |
| a whole zone-year | ~15,000 shards, most of which do not exist (see below) |

**Most of the grid has no objects behind it at all.** A zone is mostly ocean and nothing is written there;
Zarr v3 omits an all-fill shard entirely, so most of the grid has no objects behind it and
reads return fill. This is why the store is 0.9–1.8 PB rather than the 8.5 PB its logical
extent implies. It also means **"no object" and "no data" are the same thing** at the pixel
level, and a consumer cannot tell ocean from a tile that yielded nothing. The provenance below is what
answers that.

### How a consumer knows what is actually there

Three records, in increasing detail:

1. **`years_complete`** on each zone group — the years that finished. A year absent from this
   list may hold partial data and should not be read as published.
2. **A tag per landed cell**, `zone-<ZONE>-<YEAR>`, pinning the snapshot that cell landed in.
   Tags are write-once, so a tag is a permanent, reproducible handle:
   `repo.readonly_session(snapshot_id=repo.lookup_tag("zone-33N-2025"))` reads that cell exactly
   as it was published, regardless of what landed afterwards.
3. **`runs`** in the zone group `attrs`, one entry per year: the run id, when it was assembled, the
   input window it was built over, which live tiles produced no data at all
   (`optical_skips` — this is what separates ocean from a tile that yielded nothing), and the
   observation-depth summary (`s1_free_pct`, `s1_thin_pct`, `s2_thin_pct`).

**`optical_skips` is a real coverage record, not a diagnostic.** Skipped tiles are deterministic
and spatially clustered: independent fills of the same cell produce byte-identical skip sets. They
publish as fill, and so does ocean. So the store records them per year, and
**what ships with the data should say so** — alongside the 6.8% of pixel-years embedded without
radar, which is the other thing a reader cannot infer from the pixels (§6).

Alongside the data, each published cell has **figures** under `windows/<zone>/` and a
**machine-readable verdict** under `verdicts/<zone>/` in the same bucket. The figures are a
coverage map, sampled native-resolution windows and the per-dimension distributions; the verdict
holds the checks of §5 with the cell's own optical depth. They are separate prefixes because the
figures are looked at and the verdict is read by programs, and one folder holding both makes every
consumer of either wade through the other. The figures are the fastest way to see what a cell looks
like before reading a terabyte of it.

> **One thing a downstream reader will notice and should not chase.** The observation counts
> step at MGRS tile edges, because neighbouring tiles keep different date sets after cloud
> screening. The embeddings built from those dates show no matching discontinuity, which is
> checked per cell (§5) — it is an upstream property of the imagery, not a seam in our grid.

---

## 3. Settings

These are the values to run. **★ marks the five that differ from the shipped defaults**, which is
exactly the set an operator has to pass explicitly; everything else is already the default.

| parameter | value | why |
|---|---|---|
| `fill_strategy` | `"chained-clusters"` | one cluster per zone-set, not per zone-year |
| **`max_parallel_clusters`** | **10** ★ | 10 x 250 actors reaches the full 2,500-actor quota while keeping each cluster's assembly thread under its ~275-actor ceiling (§6). Balance holds to ~16, and each cluster still opens on one of the 10 densest zones |
| **`max_parallel_ingest`** | **61** ★ | fleet-wide cap on simultaneous zone-ingests. With every year in one batch the ingest knee is gone, so this is set by quota and by what the GPU fleet can absorb (§6) |
| `max_dispatch_rounds` | 2 | **The outer recovery.** How many times the campaign re-dispatches whatever is still missing — rounds, not a per-zone budget. It is the only thing that recovers a child run that DIED, since a killed or cancelled run takes its own retry counter with it (§8) |
| `ingest_settings.max_workers` | 60 | S2 fleet width. Shortens each cell, and so the tail of the cluster holding the densest zone (§6) |
| `ingest_settings.s1_worker_fraction` | 0.22 | → 13 workers per S1 orbit at the recommended 60w, sized to finish inside S2 |
| `ingest_settings.batch_days` | 30 | S1 batch length |
| **`num_actors`** | **250** ★ | GPU actors per cluster. 10 clusters x 250 = the 2,500-actor quota, which is 81% of what 61 cells of ingest can feed — under it by policy, so the fleet never idles (§6) |
| `s1_orbit` | `"both"` | downgrades per zone when an orbit has no imagery. `"none"` is a *resolved* value, not a request: passing it in is refused, since it would defeat `require_s1` and publish optical-only embeddings that report success |
| `cleanup_mosaics` | `true` | **required** — the storage figure depends on it |
| `allow_partial_window` | `false` | a zone-year is a full calendar year or it fails |
| **`allow_s2_only`** | **`true`** ★ | On for the global campaign. A fifth of the land has no radar for 2022–24 (§6); without this those pixels produce nothing. Cambridge validated radar-free embeddings, which cleared ADR 013's blocking follow-up |
| **`overlap_years`** | **`true`** ★ | every requested year dispatched as one batch, so a zone's later years overlap the inference of its earlier ones and the campaign boots 8 clusters rather than 72. Certified on six cells that each carry both radar orbits, including a same-zone year rollover inside one cluster |
| `attempts_per_cell_in_cluster` | 2 | **The cheap retry.** One more go at a failed cell on the cluster that is still standing, reusing its kept mosaic and staged tiles — minutes, not a fresh zone-year. Covers "the work failed but the machine is fine"; a dead run is `max_dispatch_rounds`' job (§8) |
| `force_staging_reuse` | `false` | Escape hatch, and it cannot reach staging created without it: setting it changes the prefix, so it only preserves reuse between runs that both set it. To resume a specific prefix, pass `fill-zone-year`'s explicit `run_id` (§8) |
| `force_staging_restage` | `""` | Any new token forces a fresh staging prefix, for a change the source hash cannot see (a dependency upgrade). Abandoning stale staged work is always safe, so unlike the row above this one is usable in production |
| commit limit | derived, `min(clusters, 8)` = **8** | Never set by hand. It bounds commits in flight (~1 s each), not concurrent assemblies — so at 10 clusters two may briefly queue a commit, which is the intent: the gate is a bound, not an operating point, and seconds of queueing costs nothing against zones that run for hours |
| leg retry wall-clock budget | **6 h** (`max_leg_wall_clock_s`) | **Ingest.** A zone-year's ingest is three legs (optical, plus one per radar orbit), and each retries on its own when a source misbehaves. This caps how long ONE leg may keep retrying, so a single stuck source cannot hold a whole zone (and the Dask fleet it is paying for) indefinitely. Measured in time rather than tries because one try can itself take minutes. Nothing is lost when it fires: ingest commits date by date, so everything already fetched is kept, the cell goes back on the work list, and the next dispatch carries on from where it stopped — giving up early costs latency, not work. Six hours still clears three legs of the slowest dense zone |
| Earth Search page size | 100 | Set per provider. This catalogue refuses some (area, window) pairs at 250 and answers the same query at 100 — see the note below |

The commit gate and the ingest gate are both **Prefect global concurrency limits**, because
clusters are separate flow runs on separate machines and only a server-side gate can bound them
together. **They are provisioned differently, and conflating them sends an operator hunting a problem that
is not there:**

| gate | who creates it | if it is absent |
|---|---|---|
| `tessera-global-ingests` | **the campaign, at start.** `run_global_campaign` upserts it from `max_parallel_ingest`, so that parameter is the single place the number lives | nothing to fix. Absent is the expected state on an account no campaign has run on |
| `tessera-global-commits` | **a human, before launch:** `register_work_pool.py --commit-limit N` | fills fail closed — `prefect.concurrency` does not auto-create it |
| `tessera-global-inference` | **the campaign, at start**, always at 1 | inference simply runs. It is a pause flag, not a cap, so its absence removes a lever rather than a limit |

So the ingest gate needs no pre-provisioning and **should not be created by hand**: a hand-set
value is overwritten by `max_parallel_ingest` at the next start, which reintroduces exactly the
drift the upsert exists to prevent. The registration script's asymmetry — a `--commit-limit` flag
and no ingest equivalent — is intentional, not an omission.

**Changing a gate while the campaign runs is a different matter, and it is the only way to throttle
or pause one in flight.** The campaign writes all three gates at start and never again, so a value
set by hand mid-campaign holds until the next campaign start. Lower `tessera-global-ingests` to slow
the intake, or set it to **zero to pause ingest**; set `tessera-global-inference` to **zero to pause
inference** (§8). Change the parameter as well as the gate if a new width is meant to survive a
re-dispatch.

**The inference gate is read, not acquired**, which is why it can sit inside a dispatch loop: there
is no slot to hold, no lease to renew, and nothing to release if a process dies mid-pause. **Zero
means paused and any positive value means run** — the campaign writes 1 — so its number carries no
capacity meaning, and a read that fails is treated as "not paused" rather than stopping the
campaign.

> **Page size is a per-provider setting and must never become a rule.** A deterministic catalogue
> refusal — one that defeats the whole retry ladder, because retrying an unacceptable request
> cannot help — can be a page-size sensitivity rather than missing data or our own load: Earth
> Search answers a query at 100 items per page and refuses the same one at 250, while the adjacent
> year answers at 250. Raising it for CMR-STAC made that catalogue's 500s *worse*. Tune it against
> the provider in front of you.

---

## 4. Order

Zones are dealt to clusters **longest-processing-time first**, so each of the 10 clusters
opens on one of the 10 densest zones and totals land within 0.0% of each other. Within a
cluster, zones run densest to sparsest.

That ordering does two jobs, and it is **not** a barrier: a cluster requests GPUs
as soon as *any* mosaic in its opening window lands, not its densest one. Blocking on the
densest, which is also the slowest to ingest, idles a fleet about six hours per cluster-year with
finished mosaics already on disk. The ingest adapter's executor is sized to
`1 + look_ahead`, matching what the flow primes: one thread short and a sibling queues behind the
head cell, which silently restores the barrier.

---

## 5. Validation — every cell is checked as it lands

**A cell is not finished when it is assembled. It is finished when it has been checked.** Each fill
dispatches one `validate-zone-year` run per cell the moment that cell is tagged, and **does not wait
for it**: cell N is checked while cell N+1 is still being inferred and assembled.

Three things follow from it being a separate flow rather than a step:

* **It is off the critical path entirely.** Nothing joins the validation run, so no GPU fleet waits on
  it and the campaign's throughput is untouched.
* **A failure is addressable.** One run per cell, named for the cell, so monitoring escalates a
  coordinate rather than a line buried in a multi-hour fill's log.
* **It runs on the ingestion image**, needing neither Ray nor a GPU, so the thing that judges the
  product does not inherit the inference image's release cadence.

**The cell is already tagged when the check runs, and it must be.** Withholding the tag until a
separate run agrees would make campaign progress depend on a run nothing joins — and a validation
that never ran would then look exactly like a cell that never landed. So the two facts get two
records: **the tag says the cell landed, the verdict says it is sound.** A blocking finding surfaces
as a failed validation run.

**Nothing is retried or reopened automatically.** A refill costs hours and real money, so remediation
is a decision. A failed cell stays published and flagged until someone makes it.

**Cost: about 3 minutes and under three cents on the densest cell** (8,714 shards), under a minute on
small ones. The full check runs, pixels included: coverage, placement, provenance, the input window,
seams, dimension health, the quantization invariant, and 10 native-resolution windows.

**What fails a cell**, in one line each: a shard written outside the land mask; a coverage
reconciliation that does not close; an input window with a month gap nothing accounts for; a completion
mark with no run record; an embedding seam median outside 0.80–1.25; a constant embedding dimension or
scale-setting shares far from 1.0.

**Four rules the campaign depends on:**

1. **A finding fails the cell, not the run.** One validation run per cell, so the campaign and every
   other cell keep moving.
2. **Monitoring can tell this error from any other.** It raises its own exception
   type, and the **cell validation** check is last in `campaign_health.py`'s list so its PROBLEM wins
   a verdict tie: a published-data defect outranks every throughput signal in §9.
3. **"Found a defect" and "could not run" are different.** A cell that cannot be read, or whose
   dispatch was lost, leaves **no verdict on file** — and monitoring reads published cells against the
   verdicts beside them, so such a cell reports as *unvalidated* rather than as fine. The absence is
   the record; the log line is not, because a log line does not survive retention.
4. **Figures go to `windows/<zone>/` and the verdict to `verdicts/<zone>/`** in the outputs
   bucket, and an **AI
   reviews the figures as part of the monitoring round** — 10 windows per cell, reporting named
   features and named artifacts plus plausible / cannot-tell / suspicious. Judged against that cell's
   own optical depth (published in the verdict as `s2_obs_mean`), because a noise-like window on a
   thin cell is the correct output. Suspicious **flags for a human, never blocks**.

Two readings that will appear constantly and are **not** defects: the observation-count arrays step at
tile edges on every cell (upstream of assembly, from cloud masking), and a **thin cell is not a broken
cell** — picture quality tracks optical depth and nothing else, which `OPTICAL_THIN_MAX_OBS` records
per pixel as a share of embedded area. Publish thin cells; label them.

**Cells marked empty are not validated.** Nothing was embedded, so every pixel-level check has no
subject; the coverage question that remains is settled by the fill's own `written + skipped == live`
reconciliation. All three consumers share one definition of the validated set, so the exclusion cannot
read as a missing verdict.

**A corrected cell needs a fresh tag name** — icechunk tags are write-once forever, so a refill can
never re-pin the canonical name (`scripts/reopen_zone_year.py`).

The whole design, its costs and its limits: `final-data-validation-plan.md`. A closing sweep over every
published cell runs once at the end, for final peace of mind rather than as a gate — it renders a
*second, differently sampled* set of windows so the inspectable total doubles. How the verdicts and
the figures reach a person, and which findings justify stopping the campaign rather than shelving the
zone: `campaign-monitoring-plan.md`.

---

## 6. Sizing, cost, and the two things that actually bind

**Figures below are results, not derivations.** [`campaign-cost-model.md`](campaign-cost-model.md)
is the source of truth for every number here and the arithmetic behind it; this section states
what to run and what it costs. Where the two disagree, the cost model is right.

**Costed in tokens, and both sides of the division are measured in one unit.** Inference
consumes a sequence per pixel, so cost scales with `tokens = pixels × observations`, not with
pixels. The campaign is **2.36 × 10¹⁵ combined S2+S1 tokens** — 1.363 × 10¹³ pixels at a
measured, land-weighted **173 tokens per pixel** — at a measured **2.127 M combined tok/sec**
per worker (cost model §6c). The equivalent px/s is 12.3K.

**Quote combined tok/sec, never px/s and never optical tok/sec.** Pixels-per-second mixes machine
speed with geography. An optical rate is a *different unit*: `t_kept` counts optical timesteps only,
so it cannot see the radar sequences the same forward pass encodes — on an optical basis the same
fleet reads 2.26–2.93 M tok/s radar-free, 1.60–1.79 M on one orbit and 1.23–1.62 M on both, a spread
that is an artefact of the unit rather than a real difference. On the combined basis it collapses to
1.01–1.08× within a run, so that is the only basis quoted here.

**The configuration is 61 ingest cells at 60 S2 workers each, and 2,500 GPU actors as 10 clusters
of 250.** That is **~5.1 days** and **~$700,000** (interval $594,000 – $846,000, of which inference
is $573,000; cost model §1, §6c, where the ingest line is under review upward by up to 3×). Both quotas are
applied in prod and both are the binding limit rather than slack: 22,692 of 25,000 Fargate vCPU,
and 10,000 of 10,000 G-and-VT.

**The schedule is `307,854 GPU-hours ÷ 2,500 actors` = 5.1 days**, and every other number here is
subordinate to it. Ingest delivers its 1,008 zone-years in 4.2 days at 61 cells, so the fleet is
the constraint and always finishes last.

**Three limits set that shape, and it is the only point where all three are satisfied.**

| limit | value | what it forces |
|---|---|---|
| G-and-VT quota | 10,000 vCPU = 2,500 `g6e.xlarge` actors | the total fleet, and therefore the 5.1 days |
| assembly ceiling | ~275 actors per cluster | at least 10 clusters, since 2,500 ÷ 8 = 312 crosses it |
| Fargate quota | 25,000 vCPU | at most ~67 cells; 61 is what the fleet can absorb |

**Why 61 cells and not fewer or more.** With every year in one batch there is no per-year floor,
so past about 52 cells the fleet is what the campaign waits on and the date stops moving. Fewer
cells would reach the same 5.1 days but with no ingest buffer; more would buy neither speed nor
anything else. 61 is the count that keeps the fleet fed at **81% of what ingest can supply** —
under it by policy, so a mosaic is always waiting and no GPU idles.

| cells | Fargate vCPU | ingest | fleet vs supply | campaign |
|---|---|---|---|---|
| 52 | 19,344 | 4.80 d | ~96% — no buffer | ~5.1 d |
| **61 — the campaign** | **22,692** | **4.18 d** | **81%** | **~5.1 d** |
| 66 | 24,552 | 3.86 d | 75% | ~5.1 d |

**Cost is flat across that table to within 0.5%**, because inference is the same pixels at the same
rate and ingest worker-hours are width-neutral. So the column that matters is the buffer, not cost.

**If the date ever has to move, buy actors, because nothing else touches it.** The fleet is already at
quota, so this means a quota increase: at 3,000 actors the campaign is ~4.3 days, and each 250
actors beyond that wants one more cluster to stay under the assembly ceiling.

**Provision the GPU fleet under what ingest can feed.** That is the policy, not an accident of
rounding: it keeps a standing queue of finished mosaics so the fleet is never idle, and the margin
absorbs an ingest cell failing and restarting without the GPUs noticing. The campaign sits at 81%,
which the quota chose rather than the policy — 85% would want 2,611 actors and there are 2,500. The cost
is that inference trails ingest slightly at the start.

**Fleet-feeding is not a risk to plan around: the GPUs were busy at least 97.3% of the time**,
measured on 38N-2021 at 60 actors over 9,051 chunks and 14.7 hours for $1,600, with nothing stalled.

> **Provisioning under supply also keeps the campaign under the assembly ceiling.** A cluster
> assembles on one trailing thread, and adding actors makes inference faster while assembly stays
> the same speed, so the two eventually cross: above roughly **275 actors per cluster** assembly
> becomes the critical path. At the campaign's **250**, assembly finishes in **1.10×** the time
> inference takes — off the critical path, but by 10% rather than comfortably. That is why the
> fleet is **10 clusters of 250** rather than 8 of 312: the same 2,500 actors, on the safe side of
> the ceiling.
>
> **So fleet width and assembly capacity are one decision, not two.** Anyone raising the fleet for
> throughput is spending this margin. If a wider fleet is ever wanted, the remedy is the assembly
> side rather than the fleet: the runner leaves about 39% of its CPU idle at the shipped pool
> width, and that idle capacity is exactly what a fleet above the ceiling would need.

**Assembly is the shard write.** The merge and commit are 37 seconds of a 196-minute assembly,
so there is no second phase to schedule around. What sets that duration is that the staged
intermediate is stored **uncompressed** — if assembly wall-clock ever matters, compressing it is
the only lever with real leverage.

**Width costs nothing in contention.** Measured on prod at 55 concurrent cells and 20,316 vCPU:
the orchestrator sat at 25% CPU with zero dropped events, placement was exact, and every 503 in
the window was upstream. The ingest line carries no load penalty at full campaign width.

**Cold start is a real term in any restart calculus.** At 160 requested actors the fleet stood
at 60 after 25 minutes and 151 after 60 — so widening a fleet by cancel-and-redispatch pays a
fresh `ray up`, per-worker bringup and model load, and the saving is the naive ratio minus that
ramp. It is also the standing argument for the chained-cluster strategy of §1.

**GPUs are on-demand.** Spot is excluded by decision: sustaining this many g6e instances for
days makes interruption a certainty, and a campaign that stalls on capacity is worse than one
that costs more. Settled; do not re-open.

**The model is v1.1.** v2 Large was evaluated and is not being used.

> **A fifth of the land has no radar for 2022–2024.** OPERA RTC-S1 coverage was withdrawn
> after Sentinel-1B failed in December 2021 — interior Australia and much of Siberia return
> *zero* granules for those years — and was restored when Sentinel-1C came online in 2025.
> **`allow_s2_only` is on**, so those pixels are embedded from their optical sequence with a
> neutral radar input instead of being dropped: 6.8% of pixel-years across the campaign.
> They are cheaper too, which is where part of the rate above comes from. Cambridge validated
> radar-free embeddings, so this is a share to report with the data rather than a caveat to
> defend, and every affected pixel is identifiable afterwards
> (`s1_asc_obs_count + s1_desc_obs_count == 0`).
>
> **Greenland and Arctic Canada ship optical-only in every year, for an unrelated reason.**
> Zones 23N and 24N publish tens of thousands of **HH/HV** granules and effectively no VV+VH, so
> the ingest declines them on polarisation rather than for lack of radar — about 208 live tiles.
> Those granules report `BEAM_MODE=IW`, so this is **not** EW mode, and it is a model question if
> ever revisited rather than an ingest one. Settled.

---

## 7. Before launch

1. **Quotas, both applied in prod (verified 2026-08-06).** Fargate **25,000 vCPU**, which
   covers the recommended 61 cells at 22,692. G-and-VT **10,000 vCPU**, which is 2,500
   `g6e.xlarge` actors — the full fleet the recommended shape provisions. Neither is slack: 61
   cells provisions all 2,500 actors, so the two are matched exactly, with no slack.
   Read the applied value in the account before relying on either; the request history lags
   amendments to an open case.
2. **The commit gate provisioned** — `tessera-global-commits`, via
   `register_work_pool.py --commit-limit N`. Fills fail closed without it. The ingest gate is
   NOT a pre-launch item: the campaign upserts it from `max_parallel_ingest` at start (§3).
3. **Coverage/land mask built** for all 120 zones, and its `registry_sha256` frozen. A
   mask rebuild mid-campaign invalidates every completed zone-year's fingerprint.
4. **Store seeded** — all zone groups, all 9 year slots, `geoemb:model` and `checkpoint_id`
   stamped. The seed is the only writer of the time axis.
5. **Model checkpoint staged** at `{inputs}/models/` and matching the seed.
6. **Deployments registered** for `ingest-zone-year`, the chained fill, and the campaign.
7. **Nothing to prepare for petabyte scale — done.** S3 Inventory delivers daily on both prod
   buckets and the audit path is exercised against a real manifest. What remains is the standing
   constraints of §7b, which are rules rather than tasks.
8. **A single dense zone-year end to end, at both 60 and 80 ingest workers.** The last gate, and
   the only remaining question about the schedule. The plan runs at 60 (§3), and the gate is that
   the densest zone behaves as the model says at that width. The 80-worker arm is upside only:
   the width model is fitted over roughly 30–60 workers, so 80 is an extrapolation and nothing in
   the plan depends on it.

**Prod's state:** the coverage mask is built, all 112 land-zone ROIs are exported, the campaign
deployment set is registered in its branch-scoped form, the crash-recovery automations are armed,
and the Prefect server is sized correctly.

**Operate prod from the branch, not from `main`.** Every prod deployment is the
`-global-tessera` form, and branch-scoped registration is the supported path until the global
code merges. The consequence worth knowing: management scripts that resolve a deployment by name
need `--branch global-tessera` on prod, and fail with a bare object-not-found without it.

### 7b. What petabyte scale actually constrains

**Neither bucket size nor object count is limited**, and the only object cap is 5 TB against
chunks of a few megabytes. Two things do bind, and neither is a filing:

**Request rate is per prefix, not per bucket** — roughly 3,500 writes and 5,500 reads per second
per partitioned prefix. S3 splits the keyspace as load rises, but reactively, and a `503
SlowDown` on **our own** bucket is that split not yet having happened. The store's chunk keys are
flat, single-level, high-entropy identifiers directly under one short prefix. That is the layout S3 partitions cleanly
and repeatedly; the anti-pattern is monotonic keys, which pile every write
on one end of the keyspace. **Splitting the store across buckets addresses none of this** — the
constraint is prefix rate, the key layout already answers it, and a split costs two stores to
keep consistent and a broken single-URI story for public release.

Distinguishing a `SlowDown` that is ours from one that belongs to a source bucket is what makes
this observable rather than alarming: the campaign absorbs upstream refusals continuously and by
design.

**The only ceiling we have actually hit is the delete path, and it is not a quota.** Per-prefix
request rate is a service characteristic: no Service Quotas entry, no case to file, nothing to
raise. Every own-bucket refusal observed so far was **bulk mosaic deletion** hitting the prefix
write rate — never ingest, inference or assembly. Two things make the campaign's own cleanup a
materially easier problem: it runs per cell under its own
`staging/<zone>/<year>/<run-id>/` prefix, so the traffic is distributed rather than concentrated,
and `s5cmd rm` **batches** — measured at 3,000 objects in 2.84 s with a single worker, which is
`DeleteObjects` taking up to 1,000 keys per request rather than one call per key. That was an
open question worth three orders of magnitude and it is closed. A throttled delete costs time,
not correctness; the one real harm is delete traffic contending with a live measurement.

**Every audit of the store reads S3 Inventory, never LIST.** A flat bucket of tens of millions of
objects makes `list_objects_v2` slow enough to time out, so completeness, orphan detection and the
coverage census all read the scheduled Parquet manifest — one read of a known object instead of a
walk. Both prod buckets are configured daily with size and storage class, and
`audit_inventory_vs_list.py` is the exercised path. **Do not write a new audit against LIST**: it
would be rewritten under pressure, at the worst possible moment.

**Three standing constraints on both buckets:**

* **Versioning stays OFF** With versioning on, deletes leave markers and prior
  versions, so mosaics deleted after consumption keep billing and the true footprint silently
  diverges from the intended one. The ~$3,000 storage figure depends on deletion actually
  reclaiming space.
* **Verify a deletion by listing the prefix, never by an exit code.** One pass is not reliably
  complete: a run has reported a single error while leaving residue in three prefixes, two of them
  with no error at all.

> **Two populations, opposite retention policies. Do not let a statement about one reach the
> other.**
>
> **The embeddings store is permanent.** It is the deliverable, it is destined for AWS Open Data,
> and nothing deletes from it. This is also a second, independent reason not to split it across
> buckets: a published dataset wants one stable URI.
>
> **The mosaics are transient, and deleting them is what keeps the storage bill at $3,000.** `cleanup_mosaics=true` is
> required, not preferred — deleting each zone-year's optical and radar inputs once inference has
> consumed them is what makes storage a throughput cost of a few thousand dollars instead of a
> six-figure balance. Deletion is per-cell and continuous rather than a periodic bulk sweep, but
> it still removes millions of objects over a campaign, so the listing rule above stands.

---

## 8. Failure handling

**One rule underlies all of it: what has landed is read from the store, never from what a run
reported.** Every dispatch round re-derives the work list from the store's own completion state and
tags, so a run that reported success while quietly leaving cells unattempted does not end the
campaign, and a cell that landed is never redone. Everything below is either a retry scope, a guard
that refuses a write, or the list of what is left at the end.

### What happens when something fails

Read this as the escalation path: each row is handled at the cheapest level that can handle it, and
falls to the row below only if that fails too.

| what fails | what happens | where it retries | if it keeps failing |
|---|---|---|---|
| one copy of a date's imagery is unreadable | ingest steps down to another copy of the same date, from every stage that can hit it | in place | the date's ingest fails loudly — only *unreadable* sources fall back |
| one ingest leg misbehaves (optical, or either radar orbit) | the leg retries on its own, capped at 6 h of wall clock (§3) | in the leg | the cell's ingest fails with everything fetched so far committed |
| a cell's ingest | the cell is recorded as failed; its partial mosaic is kept | in the cluster: the dead input production is dropped and re-run | the next dispatch round resumes the mosaic date by date |
| a cell's planning, inference or assembly | the cell is recorded as failed; **its mosaic and staged tiles are kept for exactly this** | in the cluster, on the fleet that is already up — usually minutes | the next dispatch round |
| many cells at once (`look_ahead + 2` failures holding mosaics) | the cluster stops admitting new cells and winds down; the unstarted ones are recorded as never attempted | **not in the cluster** — admitting more work is what that cap exists to prevent | the next dispatch round |
| the fill run itself — crash, cancellation, lost host | nothing in-process survives it | nowhere | the next dispatch round, which is the **only** recovery from a dead run |
| a whole round achieves nothing | read as a deterministic failure — a gate, a fingerprint mismatch, an unseeded group | rounds stop early for that batch rather than buying another fleet | straight to the unfilled list |
| everything, to the end | | | the cell goes on the **unfilled list**. The campaign still **succeeds** — a handful of cells failing at this scale is an expected outcome, not a failed campaign — and prints the whole list at the end under a `WARNING` banner, as well as returning it as data. That list is the second wave's work |
| a published cell fails validation | the cell stays published, flagged, with its verdict on file | **never automatically** — a refill is a human decision (§5) | it is on the list a human works through (§9) |

**Nothing is deleted on failure.** A failed cell keeps its mosaic and its staged tiles, which is
what makes the retry cheap; the only thing a failure releases is its slot in the storage budget, so
that a run full of failures cannot deadlock its own feeder.

**No ingest flow carries a Prefect-level retry — and you lose no visibility by that.** Prefect can
retry a flow run itself: on failure it puts the *same* run back to `AwaitingRetry` and runs its
function again, keeping one run record. We deliberately do not use that here, and the retries are
the campaign's own instead. Two reasons:

* **A Prefect retry re-enters a flow whose world has moved on.** An ingest run holds a Dask cluster
  and a slot in the fleet-wide ingest gate. When the run's process dies, both are gone — the
  cluster is torn down by the teardown hook and the gate slot's lease expires within about five
  minutes. Re-running the same function in the same run would rebuild a cluster nobody accounted
  for and proceed ungated, which is exactly what the fleet-wide cap exists to prevent.
* **The campaign's own retry re-decides what to do.** A Prefect retry repeats the same call with
  the same arguments. The driver instead re-reads the store, so a cell that landed in the meantime
  is not repeated, a cell that turned out empty is dropped, and the remaining cells are
  re-partitioned across clusters.

**Every attempt is a first-class Prefect flow run, so the UI shows more than a retry would, not
less.** A retried run collapses its attempts into one record; here each attempt is its own run with
its own name, id, logs and duration, all linked to the campaign run that dispatched it and tagged
with the cell. The failed attempt stays visible as FAILED rather than being overwritten by the
attempt that succeeded — which is what you want when the question is "how many times did 34N need
before it landed". The one thing to know is that a retry is a *different* run, so following one
cell across its attempts is not a matter of opening one run. Every child carries a tag naming the
campaign run that dispatched it, and the cell itself is in the run's **parameters** and in its log
lines — which is how `campaign_health.py` attributes work to cells, and the same route a person or
an agent has to take.

### What you can do to a campaign that is already running

Three levers, and it is worth knowing which one does what before needing them at 3 a.m.

| lever | how | what it does |
|---|---|---|
| **throttle** | lower `tessera-global-ingests` | fewer cells ingest at once. Work queues; nothing fails |
| **pause ingest** | set `tessera-global-ingests` to **0** | no new cell starts ingesting. Ingests in flight finish |
| **pause inference** | set `tessera-global-inference` to **0** | no new cell starts inferring. Each cluster finishes and lands the chunks it has already queued, keeps its actors, and waits |
| **stop** | cancel the runs — **tasks first** | the only thing that stops the spend. Cancel the campaign driver and its fills, and cancel the underlying tasks before stopping infrastructure, or runs sit in `CANCELLING` forever and look live to every guard |

```bash
prefect global-concurrency-limit update tessera-global-inference --limit 0  # pause inference
prefect global-concurrency-limit update tessera-global-ingests   --limit 0  # pause ingest
prefect global-concurrency-limit update tessera-global-inference --limit 1  # resume
prefect global-concurrency-limit update tessera-global-ingests   --limit 61 # resume
```

**Which one to reach for.** Pausing **inference** is the one that stops embeddings being
written, so it is the lever for "something about the output looks wrong". Pausing **ingest**
stops mosaics being built, which is the lever for "stop spending on inputs". Zero both to wind
the campaign down to the cells already in flight. The two are independent, and a campaign start
resets both to their running values, so a pause cannot be inherited by the next run.

**A pause does not stop the meter.** A cluster holds its GPU fleet for its whole multi-cell walk,
so a paused stream finishes its current cell and then idles at full width — the fleet is still
billed. Pausing buys time to decide without losing work; it is not a way to sit still cheaply.

**Never *deactivate* a gate to hold work back.** An inactive limit grants slots to everyone, so the
work it was throttling runs unthrottled — the opposite of the intent, and silent. That is true of
the inference gate too: it reads an inactive gate as running, so deactivating it un-pauses.

**The gates hold, they do not fail.** A limit at zero used to fail the next cell that reached it,
because the server refuses a request for more slots than the limit holds and that refusal is not
retried. The two capacity gates now wait on that state and log it, which is what makes zero a
pause rather than a way to lose cells. A gate that does not *exist* still fails immediately, and
must. The inference gate works differently by design — it is **read** rather than acquired, so it
holds no slot and needs no lease, and a read that fails answers "not paused" so that a wobbly API
can never stop the campaign working.

### The three retry scopes, and why they are three

| scope | setting | what it retries | what only it can fix |
|---|---|---|---|
| inside one **leg** | `max_leg_wall_clock_s` (6 h) | one of a cell's three imagery streams — optical, radar ascending, radar descending — inside the ingest that is already running | a stuck or flaky source, without failing the whole cell |
| inside one **cluster** | `attempts_per_cell_in_cluster` (2, so one retry) | one cell, on the GPU fleet that is still standing | a cell that failed while its machinery is fine and its inputs are still on disk |
| across **dispatch rounds** | `max_dispatch_rounds` (2) | the whole missing work list, as brand-new flow runs | a run that is **gone** — nothing inside a dead process can retry anything |

**A "leg" is one of the three imagery streams a cell needs.** Every cell is ingested as three
independent legs — Sentinel-2 optical, Sentinel-1 radar ascending, Sentinel-1 radar descending —
and each retries on its own when a source misbehaves. The 6-hour budget caps how long one leg may
keep retrying, so a single stuck source cannot hold the cell, or the Dask fleet it is paying for,
indefinitely.

**A "dispatch round" means the campaign starts the work again from scratch, as new flow runs.**
This is the part worth being precise about, because it is not a resumption of anything. At the end
of a round the driver re-reads the *store* to see which cells are still missing, groups them into
clusters again, and calls `run_deployment` for each — so what appears in Prefect is a **new fill
run**, with a new name and a new id, submitted to the work pool like any other and queued behind
whatever is already there. It is not the failed run resuming, and the failed run stays failed and
visible. What makes that cheap rather than wasteful is that the new run inherits everything already
on disk: committed mosaic dates, staged inference tiles, and completed cells, none of which are
recomputed (see "When a re-run is free" below).

At the shipped two rounds, the second round is the last whatever it achieves, so the
no-progress stop above only bites if that setting is ever raised.

They are separate settings because they are separate units, and one knob for two of them would
change both meanings at once. The in-cluster scope earns its keep because the driver's unit is a
whole dispatch: without it, a cell failing early in a multi-year list would wait for every cluster
to finish that list before being tried again.

**Cancelling a run takes effect after a delay, so a replacement must wait for it.** Prefect marks a
run as cancelling and a worker acts on it seconds later, during which the old run can still be
writing. A retry therefore waits for the old run to actually stop, and abandons the retry if it has
not stopped within five minutes. Losing a recoverable cell is the cheaper mistake: two runs writing
one mosaic produce a mosaic nothing downstream can detect as broken.

### The guards — what stops a resume or a restart from mixing data

Every one of these refuses a write rather than warning about it. They are what make "just re-run
it" safe.

| guard | what it refuses | what it would otherwise let through |
|---|---|---|
| **write-once zone-year tag** | re-publishing a cell that already landed; both fill paths refuse a complete cell | two versions of one cell, with the tag pointing at whichever won |
| **atomic date commits** | a half-written date existing at all | Icechunk commits a date's pixels and its time slot together, so a date present is complete and a date absent was never started. Without that, resume would have to distrust everything already there, and restarting from scratch would be the only safe option |
| **mosaic manifest identity** (`ConfigMismatchError`) | appending to a mosaic built under a different land mask, coverage threshold, orbit, pixel policy or ingest code — this is also what makes a changed parameter raise instead of continuing | one store holding dates built two different ways, indistinguishable afterwards |
| **completion marker validated in its own pass** | marking a mosaic finished when its recorded identity disagrees with the running code | a resume that appends nothing — every date already present — silently blessing a store built under different rules, after which every later run skips the cell |
| **absent-means-off fields** | a run with `allow_s2_only` on passing against a store that predates the field | both pixel policies mixed in one store, which is the exact append the field exists to prevent |
| **seeded model check** | filling a store seeded for a different encoder or checkpoint | embeddings from a new encoder under a root advertising the old one |
| **reflectance offset read from the copy actually opened** | taking the offset from the catalogue query instead | copies of one date can carry different processing baselines, and the baseline sets that offset, so the wrong constant would shift a whole date's values with nothing to show for it |
| **staged-tile completion marker** | assembling a tile whose upload was interrupted | a tile is many objects with no atomic commit, so its missing pieces read back as *fill values*, not as an error — holes that look like data |
| **staging fingerprint** | reusing staged tiles a different configuration or a different inference build produced | one cell assembled from two versions of the model's output |
| **frozen land-mask `registry_sha256`** | a mask rebuilt mid-campaign | every completed cell's identity invalidated at once, and no way to tell which side of the rebuild a cell came from |

### When a re-run is free

A fresh campaign run recomputes nothing already done. Completed work is filtered at four
independent levels, each cheaper than the one below it:

1. **the work list** — the store's completion state and tags decide what is dispatched at all;
2. **the per-zone ingest marker** — a finished mosaic is not re-ingested;
3. **date-level resume inside ingest** — a partial mosaic continues from the first missing date;
4. **staged-tile resume inside inference** — finished tiles are not re-inferred.

Nothing has to be cleaned up by hand first: an interrupted staged tile is re-inferred and
overwritten. Two exceptions: a **changed parameter** raises rather than silently mixing
configurations, and a change to the **inference code** invalidates staged tiles, because the tiles
are what that code produced.

Two consequences of how recovery actually behaves, both of which remove work rather than add it. A
fill's terminal hook tears down its GPU fleet even when the flow body is killed outright, skipping
every `finally` — so a leaked Ray fleet is not an event to plan around. And a dead committer's gate
slot returns on its own within about five minutes, so a whole cluster dying is a short stall on
commits rather than a deadlock, and there is no manual release procedure.

> **Staged work is reused only when the run would produce the same tiles.** A run's staging prefix
> is derived from the inputs, the parameters and the inference source, so a re-run reuses tiles only
> when all three match. The code component is the inference source alone, so an orchestration or
> ingest fix reuses staging rather than abandoning it.
>
> **To resume a specific staging prefix, pass `fill-zone-year`'s explicit `run_id`.** That is the
> only reliable lever; the two `force_staging_*` knobs change the prefix as a side effect of
> changing the fingerprint (§3). None of them relaxes a gate on the published store.
>
> **A change inside that fingerprint's import closure has a landing window.** It moves the
> fingerprint, so a mosaic caught mid-append refuses its next append and needs its interrupted store
> deleted by hand. `config/providers.py` and the ingest query modules are inside the closure, so
> anything touching the query, the collections or the provider settings is a between-campaigns
> change — land it only when nothing is mid-ingest: no ingest runner and no fleet up, only the
> Prefect worker service.
>
> The mechanism, the three guard properties that have each had a hole in them, and the full set of
> resume levers: [`staging-identity-and-resume.md`](staging-identity-and-resume.md).

---

## 9. Campaign monitoring

**Two things watch the campaign, and neither substitutes for the other.** Server-side Prefect
automations post the instant the orchestrator sees a run die — fast, and lossy by measurement, since
the event broker drops events under load and cannot fire when the server itself is the unwell thing.
So the reliable half is a poll: **`campaign-watch`, a flow on a five-minute cron** that runs
`campaign_health.py`, remembers what it has already said, and posts what is new to
`#alerts-global-tessera`. It grades nothing while the campaign is not running, publishes a record of
every round to `monitoring/` in the outputs bucket, and mutates nothing. Full design and posting
rules: [`campaign-monitoring-plan.md`](campaign-monitoring-plan.md).

The signals below are what those two paths are watching for.

| signal | why it matters |
|---|---|
| **Mosaic backlog** (completed, not yet inferred) | the $3,000 storage figure assumes prompt deletion; a four-week backlog is ~400 TB and ~$9,200/month, and sooner a bucket-capacity problem |
| **How busy the GPUs are** | the whole of §6 rests on a fleet that stays busy; below about 80% busy it is being paid for idling. `campaign_health.py` reports it as "GPUs kept busy" |
| **Cells actually running** vs the cap | the ingest gate is fleet-wide and the campaign sets it from `max_parallel_ingest`, so it cannot be missing — but if that parameter is too low nothing complains, and the fleet simply runs narrow |
| **Zones failing twice** | the retry loop stops on no-progress; these need a human |
| **Coverage-gate rejection rate** | 365 dates per zone-year is measured on three zones; heavy rejection changes both time and cost |
| **A cell that failed validation** (§5) | the highest-priority signal here, and the only one about the product rather than the machinery. The cell stays published and flagged — nothing retries it, because a refill is a decision — so a single failure is a note on a list. **Four cells failing the same check** is a property of the campaign rather than of those cells, and wants a human before another day is spent writing the same defect. The monitoring script keeps the running total per check, campaign to date, and reports it from the first failure onward — the tally is the only thing that tells the two cases apart |
| **A published cell with no verdict** (§5) | **should not happen, and means the validation never ran.** Every landed cell dispatches one, so a missing verdict is a failure of the validation path itself — a dispatch that never reached the API, a validator run that died before writing, or a deployment that is not registered. It is reported separately from a *failed* verdict because the two need opposite responses: a failed verdict is a finding about the data, while a missing one says nothing about the data at all and is fixed by running the validator again. It is a `WARN` rather than a fault only because the cell itself is intact and the check is repeatable — not because it is expected |
| **The AI window review's `suspicious` verdicts** (§5) | a ranking, not a gate. Read the worst few per round; ignore a thin cell rated poorly for being thin |

**Three reading rules, each because a wrong reading looked right.**

**Never read observation depth, tokens, throughput or cost off a run still in flight.** A run
sweeps its zone north to south while depth falls with latitude, so a partial run has measured only
its deepest part: zone 38N read `t_kept` 121 at one third complete and **73** at completion. The
bias is one-directional and predictable, not noise. How busy the GPUs are is the exception — that
rises legitimately as actor start-up amortises.

**Run the resume checks early in a fill, not late.** The work-list ratio (which needs the dispatch
intent) and orphaned staging (which needs none, and so catches the operator who meant to resume and
omitted `run_id`) both detect a resume that silently restarted — and that finding is only
actionable while killing the run still saves the money.

**Zeroing the ingest gate is the pause lever; deactivating a gate is not.** Setting
`tessera-global-ingests` to zero makes every stream hold before taking up its next cell — the cell
in flight finishes, nothing fails, and raising the limit resumes. Deactivating a gate does the
opposite of what it sounds like: the server grants slots against an inactive limit, so the work it
was throttling runs unthrottled. The monitoring view records a zeroed gate without grading it, since
that is a pause, and grades a deactivated one as a fault while any fill is live.

**A pause is not free, and nothing stops a fleet that is already up.** A cluster holds its GPU fleet
across its whole multi-cell walk, so a paused stream finishes its current cell and then idles at full
width and full cost. Pausing buys time to decide; only cancelling stops the meter (§8).

---

## 10. Cost and time determinants

**The figures in §6 are not derived here.** `campaign-cost-model.md` is the source of truth
for every number and the arithmetic behind it; this section is the index — what to look at
when something changes, and roughly what it moves. Anything quoted in this plan that is not
in the table below is a *chosen setting*, not a computed one.

| if this changes | it moves | look in the cost model |
|---|---|---|
| the coverage mask is rebuilt | tile count → every cost; zone durations → the wall-clock floor | §3, §4 |
| the longest zone gets faster or slower | the per-year floor, and therefore the whole schedule | §4 |
| S1 or S2 revisit changes (a satellite fails or launches) | tokens/pixel → inference cost and fleet size | §6 |
| cloud climatology or the SCL class set changes | the optical half of tokens/pixel | §6 |
| a measured `tok/sec` arrives from a real run | the inference rate directly — replaces the borrowed anchor | §6 |
| GPU or Fargate pricing | the totals, nothing structural | §3 |
| `allow_s2_only` is switched off | 6.8% of pixel-years stop producing output | §6 |

**Two uncertainties are carried rather than closed**, and together they are the widest driver of
the cost interval: an unexplained **~19% rate deficit** on the 20–45° both-orbit sample, and the
**35–50° latitude band is interpolated** rather than measured. No further measurement is planned;
if a cell of opportunity lands in that band, reading its observation counts refines the level
(cost model §9, §10 item 4b).

**Four mechanisms. If a number moves and none of these explains it, something else changed.**

1. **The GPU fleet, not ingest, sets the schedule.** The campaign is `GPU-hours ÷ actors`, so past
   ~52 cells every configuration lands near 5.1 days: extra cells buy buffer, and only extra
   actors buy time.
2. **Inference cost scales with tokens, not pixels — and the tokens are combined S2+S1.**
   Quote combined `tok/sec`. Pixels-per-second mixes machine speed with geography, and an
   *optical* tok/sec is a third unit again — `t_kept` counts optical timesteps only, so it cannot
   see the radar sequences the same forward pass encodes (cost model §6b). It is not purely
   token-bound either: a per-chunk read floor scales with pixels and bytes, and stops being hidden
   where sequences are short.
3. **Ingest worker-hours are width-neutral.** Halve the fleet, double the duration, same
   bill — so fleet width is schedule bought for free, and cost is nearly flat across every
   configuration in §6.
4. **Inference cost is invariant to the ingest configuration.** Same pixels, same rate. The
   ingest configuration decides only how fast mosaics arrive, and therefore how large a GPU
   fleet stays busy.

**Radar observation depth is regional, not latitudinal — do not model it against latitude.**
Across five measured latitude bands radar depth spans 66–147 tokens per pixel with a correlation
against latitude of **+0.009**, while optical depth over the same bands gives **+0.912**. The
deepest radar measured anywhere is the Middle East and the shallowest the Arctic, which points at
the Sentinel-1 observation plan rather than at geometry. Radar's *share* of a sequence does fall
with latitude, but only because the optical denominator rises. Planning form: a constant **90
tokens/px, range 66–147** (cost model §6b).

**Clusters are balanced on work, not on area.**
`zone_work_weight` weights live tiles by their latitude band's observation count, and by the number
of years a zone carries. At the campaign's **10 clusters** the resulting spread in true work is
**0.009%** (heaviest cluster over lightest), where balancing on tile counts alone would give
**21.8%**. Re-derive it with `scripts/cluster_work_spread.py`, which reads the current mask and runs
the campaign's own partitioner; rerun it if the mask is ever rebuilt. Both halves are load-bearing: cost
scales with area × observations, and a batch spanning years (which is the campaign setting) would
otherwise leave one cluster draining the extra years while the rest sit idle. A zone with no years
left weighs zero, which subsumes the retag-only case and skips its mask read.

Two properties to keep in mind if this is ever revisited. The imbalance is **not random** — latitude
drives it, so a cluster drawing high-latitude zones is heavy in *every* year. And clusters are
long-lived, so the last to finish sets the campaign date and a heavy one is never averaged away.
Within-cluster order still sorts on tile counts, which is correct: ordering and actor clamping are
properties of area, not of work.

## 11. Evidence

**Read this list as authority boundaries, not a bibliography.** Where two documents disagree, the
one named for that subject wins. A figure quoted outside its own document is a result with a
pointer, never a derivation.

| document | authoritative for | explicitly NOT for |
|---|---|---|
| this file | what runs, with what settings, in what order, and what to do when it breaks | any figure's derivation |
| [`campaign-cost-model.md`](campaign-cost-model.md) | every cost, rate, fleet size and GPU-hour figure, and the per-pixel source-coverage census | operational order |
| [`campaign-cluster-sizing.md`](campaign-cluster-sizing.md) | how work balances across N clusters, and that commits do not constrain the count | throughput, GPU-hours or cost — its px/s figures predate the switch to tokens |
| [`campaign_inference_profile_2026_08.md`](campaign_inference_profile_2026_08.md) | measured per-cell inference behaviour: observation depth, throughput, cost per chunk, and the radar effect on all three | anything from a run still in flight, which it now refuses to carry |
| [`radar_source_coverage_2026_08.md`](radar_source_coverage_2026_08.md) | which zones publish **no** usable radar at all, and the polarisation reason Greenland and Arctic Canada are optical-only | how much LAND has radar — that is a per-pixel question and belongs to the cost model |
| [`inference-perf-run-ledger.md`](inference-perf-run-ledger.md) | the raw per-run measurements each figure came from | conclusions |

**Those documents carry their own withdrawn claims beside the corrected ones, on purpose** — a
reviewer who sees only the final number learns nothing about how it went wrong, and
[`../corrections-register.md`](../corrections-register.md) indexes them by mechanism.

**This file carries none of them.** It states the current position; how that position was reached
lives in the document that owns the measurement. A finding that changed a decision here is folded
into the section it changed, and history appears only where it explains a decision that would
otherwise look arbitrary.

The register groups them by the mechanism that produced them rather than by file. The mechanism is the part
that keeps repeating: eight mechanisms cover every one, and most recur across documents that
do not cite each other. **Read it before publishing a figure or reusing one.**

- [`campaign-cost-model.md`](campaign-cost-model.md) — costs, GPU fleet sizing, the idle-burn
  arithmetic, and the observation-count model behind the throughput basis.
- [`campaign-cluster-sizing.md`](campaign-cluster-sizing.md) — the coverage census, how zones
  divide across N clusters, and why commit concurrency is a non-issue.
- [`ingest_optimization_campaign_2026_07.md`](ingest_optimization_campaign_2026_07.md) — every
  ingest measurement: what each change bought and what failed.
- [`../decisions/011-campaign-zone-ingestion.md`](../decisions/011-campaign-zone-ingestion.md)
  — why the campaign triggers ingestion per zone.
- [`../decisions/008-global-store-architecture.md`](../decisions/008-global-store-architecture.md)
  — the store layout, write model, and commit behaviour.
- [`../decisions/013-optional-s1-s2-only-pixels.md`](../decisions/013-optional-s1-s2-only-pixels.md)
  — what the `allow_s2_only` flag does, the neutral-input convention, and the scientific
  validation it left open. **The flag is on for this campaign** (§3); that validation is the
  **P2** rung's job.
- `tests/unit/test_cluster_balance.py` — the runnable diagnostic behind §4; rerun it rather
  than trusting the figures if the mask is rebuilt.
