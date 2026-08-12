# Global TESSERA campaign — the plan

**The standing plan for building 9 years × 112 land zones of 10 m embeddings into one
Icechunk store.** Everything here is settled unless it appears in §10, the list of decisions still open before
launch.

This document is the entry point and the source of truth for **operations** — what runs, with
what settings, in what order, what comes out, and what to do when it breaks. It states the
current position rather than how that position was reached: a finding that changed a decision is
folded into the section it changed, and the measurement behind it lives in the document named
beside it (§12).

[`campaign-cost-model.md`](campaign-cost-model.md) is the source of truth for **figures**:
every cost, rate, fleet size and the arithmetic behind them. Numbers appear here as results
with a pointer, never as derivations; where the two disagree, the cost model is right.

---

> **The zone count.** The coverage mask holds **120** UTM zone groups, of which **8 are all-ocean**
> (`10S 11S 13S 14S 27S 44S 45S 46S`) and **112 carry live land** — 360,953 live 2048-px tiles, or
> 24,202 live MGRS tiles. Every per-cell cost multiplies by that 112, so it is worth being exact:
> the authority is `scripts/rank_zones.py`, which reads the mask directly. Rerun it rather than
> trusting this sentence if the mask is rebuilt.

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

**Every year is dispatched in one batch, so there is no year barrier.** A cluster works a
multi-year zone list, which makes the makespan `total work ÷ cells` rather than
`max(longest zone, work ÷ cells)` *per year*. That is what lets cell count and GPU quota buy wall
clock at all: with the barrier, cells stop helping at about 45 and no quota purchase moves the
date. It also costs 10 `ray up` cycles instead of 90, worth ~$8,000 of the ramp line. That saving is not the
reason to do it; the schedule is.

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

**Every year is dispatched together** (`overlap_years=true`). A cluster works a multi-year zone
list, so one zone's later years overlap the inference of its earlier ones, and the campaign pays
**10 `ray up` cycles rather than 90**, worth ~$8,000 of the ramp line. The schedule is the real
reason, not that: it is what makes GPU quota buy wall clock again (§6).

> **Why this is safe.** Concurrent zone-years were never unsafe in the data: different groups
> rebase cleanly, and even two years of the same zone write strictly disjoint objects, because
> every chunk and shard is 1 in the time dimension. The only obstacle was two group
> *attributes* (`years_complete` and `runs`), which both writers rewrote in the same commit as
> their shards and which `ConflictDetector` cannot merge. Those commit separately and retry
> (`shard_writer.commit_year_attrs`).
>
> No zone-disjointness scheduler is needed: **the partition is over zones**, so a zone's every
> year lands in one cluster, where assemblies serialize on that cluster's single trailing
> thread. That is also what bounds concurrent assembly memory to one cluster's worth.
>
> **What makes it possible is that the inference window is a per-cell value** carried on every work
> item (`ZoneContext.time_window`), not read from the actor's config. An actor is built once with
> one config, so without that a cell of another year streamed through it would have been inferred
> over the wrong months — and the session's only mismatch check is on `s1_orbit`. The risk in
> threading it was bit-identity, since three loader call sites must receive identical kwargs: the
> fallback is resolved **once** into a local and that value passed to all three, so they cannot
> drift. Two tests hold it — one that passing the config's own window explicitly is byte-for-byte
> identical at `atol=0`, and one that a *different* window genuinely changes what is read, which is
> the failure the first cannot see, because **an ignored argument passes every identity test.**

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
3. **`runs`** on the zone group, one entry per year: the run id, when it was assembled, the
   input window it was built over, which live tiles produced no data at all
   (`optical_skips` — this is what separates ocean from a tile that yielded nothing), and the
   observation-depth summary (`s1_free_pct`, `s1_thin_pct`, `s2_thin_pct`).

**`optical_skips` is a real coverage record, not a diagnostic.** Skipped tiles are deterministic
and spatially clustered: independent fills of the same cell produce byte-identical skip sets. They
publish as fill, and so does ocean. So the store records them per year, and
**what ships with the data should say so**, alongside the optical-only regions of §10.

Alongside the data, each published cell has **figures and a machine-readable verdict** under
`windows/<zone>/` in the same bucket — a coverage map, sampled native-resolution windows, the
per-dimension distributions, and the checks of §5 with the cell's own optical depth. That is
the fastest way for a consumer to see what a cell looks like before reading a terabyte of it.

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
| **`allow_s2_only`** | **`true`** ★ | On for the global campaign. A fifth of the land has no radar for 2022–24 (§6); without this those pixels produce nothing. ADR 013 does not consider the combination validated — see §10 |
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

So the ingest gate needs no pre-provisioning and **should not be created by hand**: a hand-set
value is overwritten by `max_parallel_ingest` at the next start, which reintroduces exactly the
drift the upsert exists to prevent. The registration script's asymmetry — a `--commit-limit` flag
and no ingest equivalent — is intentional, not an omission.

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
4. **Figures and the verdict go to `s3://global-tessera-embeddings/windows/<zone>/`**, and an **AI
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

**Fleet-feeding is not a risk to plan around: duty cycle is ≥97.3%**, measured on 38N-2021 at
60 actors over 9,051 chunks and 14.7 hours for $1,600, with nothing stalled.

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
> They are cheaper too, which is where part of the rate above comes from. What remains is a
> quality caveat rather than a coverage hole — see §10.

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
8. **A single dense zone-year end to end** — see §10; this is the last gate.

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

* **Versioning stays OFF — DECIDED (Robert).** With versioning on, deletes leave markers and prior
  versions, so mosaics deleted after consumption keep billing and the true footprint silently
  diverges from the intended one. The ~$3,000 storage figure depends on deletion actually
  reclaiming space.
* **No lifecycle rules — DECIDED (Robert).** Correct rather than an oversight: the embeddings store
  is the published product and nothing in it should age out or change storage class, while the
  mosaics are removed explicitly by the campaign the moment they stop being needed — sooner and
  more precisely than any age rule could manage.
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
>
> **Open Data has consequences worth settling early**, since the application has lead time and the
> size figure is what it asks for: the store must be publicly readable with Requester Pays OFF, it
> needs a registry entry, and sponsorship may cover its storage and egress — which would move the
> storage line in the cost model's §3 rather than merely confirming it. The store currently sits branch-prefixed
> inside a private bucket, so the path from here to a public dataset is a migration question that
> has not been designed.

---

## 8. Failure handling

Three mechanisms, all shipped.

**Resume.** An interrupted mosaic is picked up, not cleared. Icechunk commits a date's time
slot atomically with its pixels, so a date present is complete and a date absent was never
started. Resume assumes the same land mask, which the manifest enforces on every write.

**If one copy of a date's imagery is unusable, ingest falls back to another.** Sentinel-2 dates are
often published as several copies, and an unreadable one steps down to the next — from every stage
that can hit it, including the very first read of a date, so a single bad object cannot strand a
whole zone-year while a good copy sits unused. Only an *unreadable* source falls back; every other
error still fails loudly.

**The reflectance offset is read from the copy actually opened.** Copies can carry different
processing baselines, and the baseline sets that offset — so taking it from the catalogue query
instead would correct the pixels with the wrong constant and shift a whole date's values silently.

**Retry, at two levels.**

*Inside a cluster.* A failed cell is re-attempted on the still-provisioned fleet —
`attempts_per_cell_in_cluster`, default 2, so one retry. The cluster is already up, the failed cell's
mosaic was retained, and its staged tiles resume, so this is normally minutes
rather than a fresh zone-year. **Eligibility is "kept its mosaic", not "failed"**: two paths
delete the mosaic on purpose (the orbit-mismatch deferral cap, a terminal plan) so a
systematic failure cannot stack multi-terabyte inputs, and retrying one of those would run
against nothing.

**Cancelling a run is a request, not an instant.** Prefect marks it cancelling and a worker acts
on that some seconds later, so a run being replaced can still be writing. The flow therefore waits
for the old run to actually stop before starting its replacement, and gives up on the retry if it
has not stopped within five minutes. Losing a recoverable cell is the cheaper mistake: two runs
writing one mosaic produce a mosaic nothing downstream can detect as broken.

*Across dispatches.* A failed zone does not sink the campaign. Each dispatch gets up to
`max_dispatch_rounds` rounds, and every round re-reads the *store* for what is still missing,
so a retry never repeats a zone that landed. Rounds stop early when one makes no progress —
that is what a deterministic failure looks like from the driver, and it wants a human rather
than another fleet. The campaign raises at the very end listing every unfilled cell.

**The two bounds are separate on purpose.** `attempts_per_cell_in_cluster` counts attempts per cell;
`max_dispatch_rounds` counts whole dispatches. Overloading one knob for both would silently
change what either means. The in-cluster level exists because the driver's unit is a whole
dispatch: without it, a cell failing early under `overlap_years` would wait for every cluster
to finish its entire multi-year list before being re-attempted.

**Re-run.** A fresh campaign run recomputes nothing already done: completed work is
filtered at the work list, at the per-zone ingest marker, at date-level resume inside
ingest, and at staged-tile resume inside inference. Two exceptions — a changed
parameter raises rather than silently mixing configurations, and a change to the **inference
code** invalidates staged tiles by design.

Staged-tile resume distinguishes a *finished* tile from one a crash caught mid-upload: a
staged tile is many objects with no atomic commit, and its missing pieces would read back as
fill values rather than as an error. A tile counts as done only if its completion marker
landed, so an interrupted one is re-inferred instead of assembled with silent holes. Nothing
has to be cleaned up by hand first — the re-run overwrites it.

> **The staging fingerprint is a hash of the inference source only** (`inference_code_identity`),
> so an orchestration or ingest fix reuses staged tiles. The AMI is resolved once per campaign and
> pinned into every fill, and that pin is the only thing stopping one run straddling two images. Two
> overrides exist for the judgement calls a hash cannot make (§3).

> **Three properties of these guards that are easy to reintroduce a hole in.** Each is a path
> around a check that is meant to be unavoidable, and each was one.
>
> - **The code-identity closure resolves relative imports.** A relative import's module name is the
>   tail alone (`providers` for `from .providers import …`), so a walker that follows only names
>   starting with the package stops at every `__init__` — which left `config/providers.py`, holding
>   the STAC collections, band lists, resolutions and baseline settings, *outside* the ingest
>   fingerprint. Changing which imagery a mosaic is built from then left its identity unmoved and a
>   resume was free to append across the change. The ingest closure is 30 files; enumerate it, never
>   trust a docstring's account of it.
> - **The completion marker validates identity in its own pass.** The ingest code identity is
>   checked on the *append* path, so that a code change does not declare finished mosaics
>   stale — but a resume adopting a store whose every date already landed appends nothing, and with
>   no append there is no check. The marker would land over a mosaic built under a different mask,
>   threshold or code, after which every later run skips the cell. Validating before marking is what
>   stops a disagreement on the last store leaving the earlier ones marked.
> - **`ABSENT_MEANS_OFF` names the fields where having no opinion is itself an opinion.** `allow_s2_only`
>   records "off" by being absent, for compatibility with stores predating the field — so manifest
>   validation skipping absent keys let a run with the flag on pass against a legacy store and mix both pixel
>   policies in one store, the exact append the field exists to refuse. A field *describing* a store
>   and a field stating the *policy* it was built under want opposite answers, and one rule cannot
>   express both.

> **Reaching existing staging: what works and what only looks like it does.** All three levers move
> which STAGING prefix a run uses. None of them touches the published store or relaxes a gate on it.
>
> - **`force_staging_reuse` does not reach staging fingerprinted without it.** It substitutes a
>   constant into the run-id hash, so it yields a *different* prefix from the one an unflagged run
>   staged under; it preserves reuse only between runs that both set it. **The explicit `run_id`
>   parameter is the only reliable lever.**
> - **`fill-zone-year` mints a fresh run_id when the parameter is omitted.** The deterministic
>   staging fingerprint belongs to the campaign driver, not the per-cell flow, so a bare re-dispatch
>   of the cell flow starts staging over. Pass the preserved `run_id` to resume.
> - **Re-assembling an already-complete zone-year needs `assemble_global` called directly** with the
>   preserved run id. Both flow paths refuse a complete cell; the direct call is repeatable because
>   it never moves the tag.
>
> The run_id is minted from the **effective** S2-only mode — `InferenceConfig` forces
> `allow_s2_only` when the orbit resolves to none — rather than from the requested flag, and
> assembly-only mode reads the policy off the run_id prefix rather than trusting a parameter nobody
> had to set. **The prefix is the record of what the staged pixels are.**

**No ingest flow carries a Prefect-level retry.** All of the retrying is the campaign's own, above.

**Two recovery facts the drills settled, both of which remove work rather than add it.** A hard
kill of a fill's flow body, skipping every `finally` and context manager, was recovered by the
terminal hook: it terminated all five GPU instances **in the same second**. That refutes the
belief that Prefect runs those hooks in the flow's own process, and it means the orphan-fleet
sweep's lack of an EC2 code path is real but **not on the recovery path**: do not plan around a
leaked Ray fleet as a likely event. And a dead committer's gate slot **returns by itself in about
five minutes**, so there is no manual release procedure to write and a whole-cluster death is a
five-minute stall on commits rather than a deadlock.

> **A change inside the mosaic-content fingerprint closure has a landing window, and a running
> campaign closes it.** Such a change moves the fingerprint, so a mosaic caught **mid-append**
> refuses its next append and needs its interrupted store deleted by hand; finished mosaics are
> unaffected. `config/providers.py` and the ingest query modules are inside that closure, so
> **anything touching the query, the collections or the provider settings is a between-campaigns
> change, not a mid-flight one.** The closure's own docstring can say what it covers but not
> *when* it is safe to act, so the rule lives here. Land such a change only after
> confirming nothing is mid-ingest — that no ingest runner or fleet is up, only the Prefect
> worker service.

---

## 9. What to watch

| signal | why it matters |
|---|---|
| **Mosaic backlog** (completed, not yet inferred) | the $3,000 storage figure assumes prompt deletion; a four-week backlog is ~400 TB and ~$9,200/month, and sooner a bucket-capacity problem |
| **GPU duty cycle** | the whole of §6; a fleet below ~80% busy is being paid for idling |
| **Cells actually running** vs the cap | the ingest gate is fleet-wide and the campaign sets it from `max_parallel_ingest`, so it cannot be missing — but if that parameter is too low nothing complains, and the fleet simply runs narrow |
| **Zones failing twice** | the retry loop stops on no-progress; these need a human |
| **Coverage-gate rejection rate** | 365 dates per zone-year is measured on three zones; heavy rejection changes both time and cost |
| **A cell that failed validation** (§5) | the highest-priority signal here, and the only one about the product rather than the machinery. The cell stays published and flagged — nothing retries it, because a refill is a decision — so a single failure is a note on a list. A repeat, or two cells failing the same check, is systemic and wants a human before the campaign spends another day writing the same defect |
| **A published cell with no verdict** (§5) | nothing checked it. The dispatch is best-effort, so a missing verdict is how a lost dispatch shows up; those cells need a validation run by hand |
| **The AI window review's `suspicious` verdicts** (§5) | a ranking, not a gate. Read the worst few per round; ignore a thin cell rated poorly for being thin |

**Three reading rules, each because a wrong reading looked right.**

**Never read observation depth, tokens, throughput or cost off a run still in flight.** A run
sweeps its zone north to south while depth falls with latitude, so a partial run has measured only
its deepest part: zone 38N read `t_kept` 121 at one third complete and **73** at completion. The
bias is one-directional and predictable, not noise. Duty cycle is the exception — it rises
legitimately as actor start-up amortises.

**Run the resume checks early in a fill, not late.** The work-list ratio (which needs the dispatch
intent) and orphaned staging (which needs none, and so catches the operator who meant to resume and
omitted `run_id`) both detect a resume that silently restarted — and that finding is only
actionable while killing the run still saves the money.

**A gate at zero is not a fault.** A concurrency limit set to zero while active once read as
*healthy*, reporting a total stall as health; the view now records that state without grading it,
because zeroing a gate is also a legitimate way to park the campaign.

---

## 10. Open before launch

Settled, and not to be re-opened: **the model is v1.1**, **GPUs are
on-demand**, the permanent store goes to **AWS Open Data**, and **Greenland and Arctic Canada ship
optical-only** — zones 23N and 24N publish tens of thousands of **HH/HV** granules and effectively
no VV+VH, so the ingest declines them on polarisation, not for lack of radar. That is ~208 live
tiles of optical-only land, distinct from the Sentinel-1B gap below, and **not** EW
mode (those granules report `BEAM_MODE=IW`). A model question if ever revisited, not an ingest one.

> **The inference line is settled and measured on one token unit.** Depth **173 tok/px**, rate
> **2.127 M combined tok/s**, both land-weighted: inference **$573,000**, campaign **$700,000** (§6).
> Full accounting in `campaign-cost-model.md` §6b–§6c.
>
> **Two uncertainties are carried rather than closable**, and together they are the widest driver of
> the cost interval: an unexplained **~19% rate deficit** on the 20–45° both-orbit sample, and the
> **35–50° band is interpolated** because measuring it is decided against. No further measurement is
> planned.

> **Provisioning under supply does a second job** beyond stopping a fleet idling
> on a shallow queue. It is also the only thing
> holding the campaign under an **assembly ceiling of ~275 actors per cluster**: assemblies
> serialise on one trailing thread, and inference time per tile falls as actors rise while assembly
> does not, so the two cross. **At the campaign's 250 actors per cluster, assembly finishes in 1.10×
> the time inference takes — off the critical path, but by 10% rather than comfortably.** The two
> cross at ~275. That is why the fleet is 10 clusters of 250 rather than 8 of 312: the same 2,500
> actors, on the safe side of the ceiling. **So fleet width and assembly capacity cannot be decided independently** — anyone raising
> the fleet for throughput is also spending this margin. If a wider fleet is ever wanted the remedy
> is the assembly side, not the fleet: the runner leaves ~39% of its CPU idle at the shipped pool
> width, and that idle capacity is precisely what a fleet above the ceiling would need.

1. **Report the optical-only cells in whatever ships with the data.** OPERA RTC-S1 coverage was
   withdrawn from about a fifth of the land after Sentinel-1B failed in December 2021 and largely
   restored with Sentinel-1C in 2025 — interior Australia and much of Siberia return *zero*
   granules for 2022–2024. `allow_s2_only` is on, so **6.8% of pixel-years are embedded without
   radar** rather than missing.

   Whether to produce them is settled: Cambridge validated radar-free embeddings, satisfying ADR
   013's blocking follow-up. What remains is descriptive. Every affected pixel is identifiable
   after the fact (`s1_asc_obs_count + s1_desc_obs_count == 0`) and `scripts/census_s1_coverage.py`
   maps the area in advance, so the campaign can report the share and how those embeddings compare
   — a statistic that ships with the data, not a gate on shipping it. **Attach Cambridge's study
   when available:** a cleared gate whose only evidence is a sentence in a chat log is one
   re-litigation away from being open again.

2. **Measure the densest zone at two fleet widths (60 and 80).** It decides whether the campaign is
   5.6 days or 4.5, and it is the only remaining question about the schedule. The width model is
   fitted over roughly 30–60 workers, so 80 is an extrapolation. **This is the last preflight
   gate.**

3. **Public release.** The store is 0.9–1.8 PB. AWS Open Data has lead time and the size figure is
   what the application asks for. The store currently sits branch-prefixed inside a private
   bucket, so the path to a public dataset is a migration question that has not been designed.

4. **The per-band split behind `zone_work_weight` is refuted at high latitude while its totals
   hold.** An open check rather than a live problem — the balance it produces is measured good to
   0.04% — but it is the one caveat inherited from the observation census (cost model §9 item 3).

---

## 11. What moves the numbers

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
of years a zone carries — measured over the real coverage census, true-work spread is
**0.04%** where balancing on tile counts alone gives 9.43%. Both halves are load-bearing: cost
scales with area × observations, and a batch spanning years (which is the campaign setting) would
otherwise leave one cluster draining the extra years while the rest sit idle. A zone with no years
left weighs zero, which subsumes the retag-only case and skips its mask read.

Two properties to keep in mind if this is ever revisited. The imbalance is **not random** — latitude
drives it, so a cluster drawing high-latitude zones is heavy in *every* year. And clusters are
long-lived, so the last to finish sets the campaign date and a heavy one is never averaged away.
Within-cluster order still sorts on tile counts, which is correct: ordering and actor clamping are
properties of area, not of work.

## 12. Evidence

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
