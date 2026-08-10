# Global TESSERA campaign — the plan

**Dated 2026-07-29, revised 2026-08-06.** The operational plan for building 9 years × **112 land
zones** of 10 m embeddings into one Icechunk store. Everything here is settled unless it appears in §8,
which is the list of decisions still open before launch.

This document is the entry point and the source of truth for **operations** — what runs,
with what settings, in what order, and what to do when it breaks.
[`campaign-cost-model.md`](campaign-cost-model.md) is the source of truth for **figures**:
every cost, rate, fleet size and the arithmetic behind them. Numbers appear here as results
with a pointer, never as derivations; where the two disagree, the cost model is right.

---

> **The zone count, settled 2026-08-06.** The coverage mask holds **120** UTM zone groups, of which
> **8 are all-ocean** (`10S 11S 13S 14S 27S 44S 45S 46S`) and **112 carry live land** — 360,953 live
> 2048-px tiles, or 24,202 live MGRS tiles. This document and the cost model both said **111** and
> that was one short; every per-cell cost multiplies by it. The authority is
> `scripts/rank_zones.py`, which reads the mask directly — rerun it rather than trusting this
> sentence if the mask is rebuilt.

## 1. Shape

Ingest and inference are **staged, not sequential**. Ingest builds a zone-year's mosaic on
Fargate; inference consumes it on GPU and the mosaic is deleted. The two use different
resource pools, so overlapping them costs neither side.

```
   sequential :  ingest  +  inference          ~12 days, and 5.6 PB held at once
   staged     :  max(ingest, inference) + lag  ~6.8 days, and ~330 TB held at once
```

Staging is **not an optimisation on top of the plan — it is the plan.** Without it the
full 5.6 PB of intermediate mosaics is held simultaneously and storage alone becomes about
$128,000 a month.

Within that, work is organised as **8 long-lived Ray clusters**, each owning a set of UTM
zones and streaming them through one persistent set of GPU actors. A cluster pays `ray up`
and model load once for its whole set rather than once per zone-year.

```
   campaign
   └── 8 clusters, each opening on one of the 8 densest zones
       ├── cluster 1 ── 35N 2017…2025 → 12S 2017…2025 → 04N …   (densest zone first)
       ├── cluster 2 ── 38N 2017…2025 → …
       └── … every requested year dispatched as ONE batch
```

**Every year is dispatched together** (`overlap_years=true`). A cluster works a multi-year
zone list, so one zone's later years overlap the inference of its earlier ones, and the
campaign pays **8 `ray up` cycles rather than 72**.

> **Why this is safe.** Concurrent zone-years were never unsafe in the data: different groups
> rebase cleanly, and even two years of the SAME zone write strictly disjoint objects, because
> every chunk and shard is 1 in the time dimension. The only obstacle was two group
> *attributes* — `years_complete` and `runs` — which both writers rewrote in the same commit as
> their shards, and which `ConflictDetector` cannot merge. Those commit separately and retry
> (`shard_writer.commit_year_attrs`).
>
> No zone-disjointness scheduler is needed: **the partition is over ZONES**, so a zone's every
> year lands in one cluster, where assemblies serialize on that cluster's single trailing
> thread. That is also what bounds concurrent assembly memory to one cluster's worth.

---

## 2. Settings

Defaults are in `run_global_campaign`. **★ marks the four that need changing from their
shipped defaults.**

| parameter | value | why |
|---|---|---|
| `fill_strategy` | `"chained-clusters"` | one cluster per zone-set, not per zone-year |
| `max_parallel_clusters` | 8 | balance holds to ~16; 8 keeps each cluster opening on a top-8 zone |
| **`max_parallel_ingest`** | **40 → 61** ★ | fleet-wide cap on simultaneous zone-ingests. With every year in one batch the ingest knee is gone, so this is set by quota and by what the GPU fleet can absorb (§4) |
| `max_zone_attempts` | 2 | bounded re-dispatch per zone-year, see §6 |
| **`ingest_settings.max_workers`** | **50 → 60** ★ | S2 fleet width. Shortens each cell, and so the tail of the cluster holding the densest zone (§4) |
| `ingest_settings.s1_worker_fraction` | 0.22 | → 13 workers per S1 orbit at the recommended 60w, sized to finish inside S2 |
| `ingest_settings.batch_days` | 30 | S1 batch length |
| **`num_actors`** | **20 → ~228** ★ | GPU actors per cluster: 85% of what ingest can feed, so the fleet never idles and keeps headroom for a restart (§4) |
| `s1_orbit` | `"both"` | downgrades per zone when an orbit has no imagery. `"none"` is a *resolved* value, not a request — passing it in is refused (2026-08-07), since it could defeat `require_s1` and publish optical-only embeddings that report success |
| `cleanup_mosaics` | `true` | **required** — the storage figure depends on it |
| `allow_partial_window` | `false` | a zone-year is a full calendar year or it fails |
| **`allow_s2_only`** | **`false` → `true`** ★ | ON for the global campaign. A fifth of the land has no radar for 2022–24 (§4); without this those pixels produce nothing. ADR 013 does not consider the combination validated — see §8 |
| **`overlap_years`** | **`false` → `true`** ★ | every requested year dispatched as ONE batch, so a zone's later years overlap the inference of its earlier ones and the campaign boots 8 clusters rather than 72. **Cleared by Phase 4** (§9b) |
| `max_cell_attempts` | 2 | attempts per cell WITHIN a cluster (one retry on the still-provisioned fleet). Distinct from `max_zone_attempts`, which counts whole dispatches |
| `force_staging_reuse` | `false` | escape hatch — but it does NOT reach staging fingerprinted without it; the reliable resume lever is `fill-zone-year`'s explicit `run_id`. See the staging corrections in §6 (2026-08-07) |
| `force_staging_restage` | `""` | escape hatch: any new token forces a fresh staging prefix, for a change the source hash cannot see (a library upgrade) |
| commit limit | derived, `min(clusters, 8)` | never set by hand — and it bounds commits IN FLIGHT (~1 s each), not concurrent assemblies |
| leg retry wall-clock budget | 36 h | **NEW 2026-08-10.** F7 settled that we retry expansively; this is what stops patience becoming unbounded. Only ATTEMPT counts were bounded before, and a page fetch is already 9 HTTP attempts over 364 s that every outer layer treats as one try. Checked only when STARTING an attempt, so a slow-but-succeeding leg is never why the loop stops (§9b) |
| Earth Search page size | 100 | **NEW 2026-08-10**, down from 250 and set per provider. That catalogue refuses some (area, window) pairs at 250 and answers the same query at 100 (§9b). NOT a rule about page sizes — raising it for CMR-STAC made its 500s worse |

The commit gate and the ingest gate are both **Prefect global concurrency limits**, because
clusters are separate flow runs on separate machines and only a server-side gate can bound them
together. **They are provisioned differently, and conflating them sends an operator hunting a
problem that is not there** (corrected 2026-08-07; this paragraph previously said both must
pre-exist):

| gate | who creates it | if it is absent |
|---|---|---|
| `tessera-global-ingests` | **the campaign, at start.** `run_global_campaign` upserts it from `max_parallel_ingest`, so that parameter is the single place the number lives | nothing to fix. Absent is the EXPECTED state on an account no campaign has run on |
| `tessera-global-commits` | **a human, before launch:** `register_work_pool.py --commit-limit N` | fills fail closed — `prefect.concurrency` does not auto-create it |

So the ingest gate needs no pre-provisioning and **should not be created by hand**: a hand-set
value is overwritten by `max_parallel_ingest` at the next start, which reintroduces exactly the
drift the upsert exists to prevent. The registration script's asymmetry — a `--commit-limit` flag
and no ingest equivalent — is deliberate rather than an omission.

---

## 3. Order

Zones are dealt to clusters **longest-processing-time first**, so each of the 8 clusters
opens on one of the 8 densest zones and totals land within 0.0% of each other. Within a
cluster, zones run densest to sparsest.

That ordering does two jobs and is deliberately **not** a barrier: a cluster requests GPUs
as soon as *any* mosaic in its opening window lands, not its densest one. Blocking on the
densest — which is also the slowest to ingest — idled a fleet about six hours per
cluster-year with finished mosaics already on disk.

> **Until 2026-08-07 that first-ready behaviour was partly defeated by a sizing bug.** The
> flow primes `1 + look_ahead` cells but the adapter's executor had only `look_ahead`
> threads, so a sibling sat queued behind the head cell and never started — and the fleet
> then waited on whichever zone came *first in the order* rather than on the first one
> ready, which is the entire point of priming a window. Fixed; the paragraph above now
> describes what the code does.

---

## 4. Sizing, cost, and the two things that actually bind

**Figures below are results, not derivations.** [`campaign-cost-model.md`](campaign-cost-model.md)
is the source of truth for every number here and the arithmetic behind it; this section states
what to run and what it costs. Where the two disagree, the cost model is right.

**Costed in tokens, and since 2026-08-07 both sides of the division are measured in one
unit.** Inference consumes a sequence per pixel, so cost scales with
`tokens = pixels × observations`, not with pixels. The campaign is **2.32 × 10¹⁵ combined
S2+S1 tokens** — 1.363 × 10¹³ pixels at a measured land-weighted **170 tokens per pixel** — at
a measured **2.273 M combined tok/sec** per worker (cost model §6b). The pair this replaces —
1.98 × 10¹⁵ tokens at ≈1.9 M tok/sec — divided a combined census by an *optical-only* rate;
the two errors nearly cancel, so the line moved only 2%. **Quote COMBINED tok/sec, not px/s
and not optical tok/sec**: pixels-per-second mixes machine speed with geography, and an
optical rate is a different unit that sat in the model's central division for three revisions.
The equivalent px/s here is 13.4K.

**The GPU fleet sets the schedule, not ingest.** Every year is dispatched in one batch, so a
zone's nine years are a single work list and no per-year floor exists: the longest single zone
gates only its own cluster's tail. Past ~52 cells the 2,500-actor fleet is what the campaign
waits on, so **extra cells buy buffer rather than speed**:

| cells | Fargate vCPU | ingest | GPU provisioning | **campaign** |
|---|---|---|---|---|
| 52 | 19,344 | 4.80 d | 100% | ~4.8 d |
| **61** ★ | **22,692** | **4.09 d** | **85%** | **~4.8 d** |
| 66 | 24,552 | 3.78 d | 79% | ~4.8 d |
| 80 | 29,760 | 3.12 d | 65% | ~4.8 d |

**61 cells at 60 workers is the recommended shape, and prod's applied Fargate quota of 25,000
vCPU accommodates it** (22,692 needed). It is not the fastest row — every row lands near 4.8
days — it is the row that reaches the 85% GPU provisioning the policy below calls for.

**Every configuration costs the same to within 0.5%: ~$661,000** (re-based 2026-08-07 from
~$672,000 with the inference line's one-unit correction; cost model §1, §6b). Inference is the
same pixels at the same rate, and ingest worker-hours are width-neutral. The decision is wall
clock bought with Fargate quota, not a cost trade.

**To go faster, ask for ACTORS, not cells.** 2,750 actors (67 cells) is ~4.4 d; 3,000 actors
(73 cells) ~4.0 d; 66 cells at proper 85% provisioning wants 2,704 actors, an ~8% bump. A
deeper ingest buffer is worth having — it absorbs a cell failing and restarting — but it is not
a speed-up. An earlier version of this plan asked for 71 cells and 23,000 vCPU on the theory
that cells were the lever. They are not.

**Provision the GPU fleet UNDER what ingest can feed — about 85% of it.** That is the policy,
not an accident of rounding. It keeps a standing queue of finished mosaics so the fleet is
never idle, and the 15% margin absorbs an ingest cell failing and restarting without the GPUs
noticing. Inference then trails ingest by roughly 18% of the run, which is the "slightly
slower start" that buys the guarantee. The recommended configuration wants **1,824 actors**,
comfortably inside the 2,500 quota; even 80 workers wants only 2,267.

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
> quality caveat rather than a coverage hole — see §8.

---

## 5. Before launch

1. **Quotas — both now applied in prod (verified 2026-08-06).** Fargate **25,000 vCPU**, which
   covers the recommended 61 cells at 22,692. G-and-VT **10,000 vCPU**, which is 2,500
   `g6e.xlarge` actors — the full fleet the recommended shape provisions. Neither is slack: 61
   cells provisions all 2,500 actors, so the two are matched by design rather than by margin.
   Read the APPLIED value in the account before relying on either; the request history lags
   amendments to an open case.
2. **The COMMIT gate provisioned** — `tessera-global-commits`, via
   `register_work_pool.py --commit-limit N`. Fills fail closed without it. The ingest gate is
   NOT a pre-launch item: the campaign upserts it from `max_parallel_ingest` at start (§2).
3. **Coverage/land mask built** for all 120 zones, and its `registry_sha256` frozen. A
   mask rebuild mid-campaign invalidates every completed zone-year's fingerprint.
4. **Store seeded** — all zone groups, all 9 year slots, `geoemb:model` and `checkpoint_id`
   stamped. The seed is the only writer of the time axis.
5. **Model checkpoint staged** at `{inputs}/models/` and matching the seed.
6. **Deployments registered** for `ingest-zone-year`, the chained fill, and the campaign.
7. **Storage at petabyte scale prepared — see §5b.** Not a quota; a set of operational
   capabilities that only stop working once the store is large, which is after it is too late
   to add them.
8. **A single dense zone-year end to end** — see §8; this is the last gate.

### 5b. What petabyte scale actually constrains

**Neither bucket size nor object count is limited**, and the only object cap is 5 TB against
chunks of a few megabytes. Two things do bind, and neither is a filing:

**Request rate is per PREFIX, not per bucket** — roughly 3,500 writes and 5,500 reads per second
per partitioned prefix. S3 splits the keyspace as load rises, but reactively, and a `503
SlowDown` on **our own** bucket is that split not yet having happened. The store's chunk keys are
flat, single-level, high-entropy identifiers directly under one short prefix, which is the layout
S3 partitions cleanly and repeatedly; the anti-pattern is monotonic keys, which pile every write
on one end of the keyspace. **Splitting the store across buckets addresses none of this** — the
constraint is prefix rate, the key layout already answers it, and a split costs two stores to
keep consistent and a broken single-URI story for public release.

Distinguishing a `SlowDown` that is ours from one that belongs to a source bucket is what makes
this observable rather than alarming: the campaign absorbs upstream refusals continuously and by
design.

**LIST stops being usable long before anything breaks.** A flat bucket of tens of millions of
objects makes `list_objects_v2` slow enough to time out, so every audit that scans the store —
completeness, orphan detection, coverage census — must read **S3 Inventory** instead. Inventory
is a scheduled manifest delivered as a file, so it costs one read of a known object rather than a
walk, but it takes a day to produce its first report. **That lead time is the whole reason this
is a before-launch item and not an operational note.**

Required, in the order they must happen:

1. **Enable S3 Inventory on the prod embeddings and inputs buckets**, daily, Parquet, with size
   and storage-class fields. Then **wait for the first manifest and read it** — an Inventory
   configuration that has never delivered is indistinguishable from one that works.
2. **Point at least one real audit at the manifest** rather than at LIST, and confirm it agrees
   with a LIST over a small prefix where both are cheap. An audit that has only ever run against
   LIST will be rewritten under pressure, at the worst moment.
3. **Versioning stays OFF on both buckets — DECIDED (Robert, 2026-08-06).** With versioning on,
   deletes leave markers and prior versions, so mosaics deleted after consumption keep billing and
   the true footprint silently diverges from the intended one. The ~$3,000 storage figure depends
   on deletion actually reclaiming space.
4. **No lifecycle rules — DECIDED (Robert, 2026-08-06).** Neither bucket carries one, and that is
   correct rather than an oversight: the embeddings store is the published product and nothing in
   it should age out or change storage class, while the mosaics are removed explicitly by the
   campaign at the moment they stop being needed, which is sooner and more precisely than any age
   rule could manage.
5. **Verify deletion by LISTING the prefix, never by an exit code.** One pass is not reliably
   complete: a run has reported a single error while leaving residue in three prefixes, two of
   them with no error at all.

> **Two populations, opposite retention policies. Do not let a statement about one reach the
> other.**
>
> **The embeddings store is PERMANENT.** It is the deliverable, it is destined for AWS Open Data,
> and nothing deletes from it. This is also a second, independent reason not to split it across
> buckets: a published dataset wants one stable URI.
>
> **The mosaics are TRANSIENT and their deletion is load-bearing.** `cleanup_mosaics=true` is
> required, not preferred — deleting each zone-year's optical and radar inputs once inference has
> consumed them is what makes storage a throughput cost of a few thousand dollars instead of a
> six-figure balance. Deletion is per-cell and continuous rather than a periodic bulk sweep, but
> it still removes millions of objects over a campaign, which is why item 5 stands.
>
> **Open Data has consequences worth settling early**, since the application has lead time and the
> size figure is what it asks for: the store must be publicly readable with Requester Pays OFF, it
> needs a registry entry, and sponsorship may cover its storage and egress — which would move the
> storage line in §3 rather than merely confirming it. The store currently sits branch-prefixed
> inside a private bucket, so the path from here to a public dataset is a migration question that
> has not been designed.

---

## 6. Failure handling

Three mechanisms, all shipped.

**Resume.** An interrupted mosaic is picked up, not cleared. Icechunk commits a date's time
slot atomically with its pixels, so a date present is complete and a date absent was never
started. Resume assumes the same land mask, which the manifest enforces on every write.

**Hardened 2026-08-07 — the S2 duplicate-copy machinery.** Two places a failure never reached
the recovery built for it. An unreadable preferred copy failed at the **coverage gate** — a
date's *first* compute, one stage before the duplicate ladder that would have stepped down to
an older copy — so one bad object stranded its whole zone-year identically on every retry with
a good older copy sitting unused; preparation now returns that failure into the same ladder a
write failure uses, and only an unreadable source is converted (anything else still
propagates). And the S2 processing-baseline map was built once per catalogue query, before
duplicates were pruned, and never rebuilt on a step-down — so the reflectance offset applied
could describe a copy the loader never opened, shifting every pixel of a date silently. It is
now derived per preparation from that preparation's own items and travels back on the result,
so the write records exactly what it applied.

**Retry, at two levels (the second added 2026-07-30).**

*Inside a cluster.* A failed cell is re-attempted on the still-provisioned fleet —
`max_cell_attempts`, default 2, so one retry. The cluster is already up, the failed cell's
mosaic was deliberately retained, and its staged tiles resume, so this is normally minutes
rather than a fresh zone-year. **Eligibility is "kept its mosaic", not "failed"**: two paths
delete the mosaic on purpose (the orbit-mismatch deferral cap, a terminal plan) so a
systematic failure cannot stack multi-terabyte inputs, and retrying one of those would run
against nothing.

**A cancellation is a request, not a fact (hardened 2026-08-07).** Prefect marks a run
Cancelling and a worker acts on it later, so a child being replaced can keep writing across
that gap — and the old discard dropped the run id, which also removed the run from the
shutdown sweep in the same breath. The flow now follows a discarded run to a *terminal* state
and keeps the id registered until it gets there; if the run will not confirm within five
minutes, **the retry is refused rather than raced**. That costs a recoverable cell, and it is
the cheaper mistake: two writers on one mosaic prefix produce a mosaic nothing downstream can
detect, and those commits do not rebase, so the loser's failure is terminal anyway.

*Across dispatches.* A failed zone does not sink the campaign. Each dispatch gets up to
`max_zone_attempts` rounds, and every round re-reads the *store* for what is still missing,
so a retry never repeats a zone that landed. Rounds stop early when one makes no progress —
that is what a deterministic failure looks like from the driver, and it wants a human rather
than another fleet. The campaign raises at the very end listing every unfilled cell.

**The two bounds are deliberately separate.** `max_cell_attempts` counts attempts per cell;
`max_zone_attempts` counts whole dispatches. Overloading one knob for both would silently
change what either means. The in-cluster level exists because the driver's unit is a whole
dispatch: without it, a cell failing early under `overlap_years` would wait for every cluster
to finish its entire multi-year list before being re-attempted.

**Re-run.** A fresh campaign run recomputes nothing already done: completed work is
filtered at the work list, at the per-zone ingest marker, at date-level resume inside
ingest, and at staged-tile resume inside inference. Two deliberate exceptions — a changed
parameter raises rather than silently mixing configurations, and a change to the **inference
code** invalidates staged tiles by design.

Staged-tile resume distinguishes a *finished* tile from one a crash caught mid-upload: a
staged tile is many objects with no atomic commit, and its missing pieces would read back as
fill values rather than as an error. A tile counts as done only if its completion marker
landed, so an interrupted one is re-inferred instead of assembled with silent holes. Nothing
has to be cleaned up by hand first — the re-run overwrites it.

> **Narrowed 2026-07-30.** That last exception used to fire on any change to the *build* — the
> staging fingerprint was the resolved AMI ID plus the source tarball's ETag, so re-baking the
> AMI or a hotfix anywhere in the repo abandoned every staged tile. It is now a hash of the
> inference source only (`inference_code_identity`), so an orchestration or ingest fix reuses
> staging. The AMI is still resolved once and pinned into every fill, which is now the only
> thing stopping one run straddling two images. Two overrides exist for the judgement calls the
> hash cannot make (§2).

> **Corrected 2026-08-07 — three holes in these guards, each a path around a check that was
> meant to be unavoidable.**
>
> - **The code-identity import closure missed relative imports.** The walker followed only
>   imports whose module name starts with the package, and a relative import's name is the
>   tail alone (`providers` for `from .providers import …`), so the walk stopped at every
>   package `__init__`. Measured on this tree: `config/providers.py` — the STAC collections,
>   band lists, resolutions and baseline settings — sat **outside the ingest fingerprint**, so
>   changing which imagery a mosaic is built from left that mosaic's identity unmoved and a
>   resume was free to append across the change. Fixed by resolving relative levels against
>   the importing module; the ingest closure went **29 → 30 files**, and the file it gained is
>   that one. Read any pre-2026-08-07 statement about what a fingerprint covers against this.
> - **A zero-append resume stamped its completion marker unchecked.** The ingest code identity
>   is validated on the *append* path — deliberately, so a code change does not declare
>   finished mosaics stale — but a resume that adopts a store whose every date already landed
>   appends nothing: no append, no check, and the marker lands over a mosaic built under a
>   different mask, threshold or code, after which every later run skips the cell. The same
>   validation now runs before marking, in its own pass, so a disagreement on the last store
>   cannot leave the earlier ones marked.
> - **"A changed parameter raises" had a one-way hole for `allow_s2_only`.** The flag records
>   "off" by being absent — deliberately, to stay compatible with stores predating the field —
>   but manifest validation skipped any key the store lacks, so a flag-ON run against a legacy
>   store matched nothing and passed, mixing both pixel policies in one store: the exact
>   append the field was added to refuse. `ABSENT_MEANS_OFF` now names the fields where a
>   store having no opinion IS an opinion; a field *describing* a store and a field stating
>   the *policy* it was built under want opposite answers, and one rule cannot express both.

> **Three staging-resume corrections (2026-08-07), found operating the escape hatches.**
>
> - **`force_staging_reuse` does not reach staging fingerprinted without it.** It substitutes
>   a constant into the run-id hash, so it yields a *different* prefix from the one an
>   unflagged run staged under — it preserves reuse only between runs that both set it. The
>   settings table (and an earlier version of this row) described it as "reuse staged tiles
>   across an inference-code change"; that reading is withdrawn. **The explicit `run_id`
>   parameter is the only reliable lever for reaching existing staging.**
> - **`fill-zone-year` mints a fresh run_id when the parameter is omitted.** The deterministic
>   staging fingerprint belongs to the campaign driver, not the per-cell flow, so a bare
>   re-dispatch of the cell flow starts staging over. Pass the preserved `run_id` to resume.
> - **Re-assembling an already-complete zone-year is possible only by calling
>   `assemble_global` directly with the preserved run id.** Both flow paths refuse a complete
>   cell; the direct call is repeatable because it never moves the tag.
>
> Related, same date: the staging run_id is now minted from the **effective** S2-only mode —
> `InferenceConfig` forces `allow_s2_only` when the orbit resolves to none — rather than the
> requested flag, and assembly-only mode reads the policy off the run_id prefix rather than
> trusting a parameter nobody had to set. The prefix is the record of what the staged pixels
> ARE.

There are **no Prefect-level retries** on any ingest flow, deliberately.

---

## 7. What to watch

| signal | why it matters |
|---|---|
| **Mosaic backlog** (completed, not yet inferred) | the $3,000 storage figure assumes prompt deletion; a four-week backlog is ~400 TB and ~$9,200/month, and sooner a bucket-capacity problem |
| **GPU duty cycle** | the whole of §4; a fleet below ~80% busy is being paid for idling |
| **Cells actually running** vs the cap | the ingest gate is fleet-wide and the campaign sets it from `max_parallel_ingest`, so it cannot be missing — but if that parameter is too low nothing complains, and the fleet simply runs narrow |
| **Zones failing twice** | the retry loop stops on no-progress; these need a human |
| **Coverage-gate rejection rate** | 365 dates per zone-year is measured on three zones; heavy rejection changes both time and cost |

---

## 8. Open before launch

Settled since the last revision, and not to be re-opened: **the model is v1.1**, **GPUs are
on-demand**, the permanent store goes to **AWS Open Data**, and (2026-08-06) **Greenland and Arctic
Canada ship optical-only** — they publish cross-pol rather than the VV+VH pair the ingest requires,
which is a model question if ever revisited, not an ingest one.

> ~~**NEW AND BLOCKING, 2026-08-06: the inference line's central division mixes two token
> units.**~~ **RESOLVED 2026-08-07 — the launch prerequisite is met.** The both-orbit
> measurement was taken (four cells under the new `t_s1_asc` / `t_s1_desc` telemetry) and the
> division re-based on one unit. Two errors of similar size pointed opposite ways — the rate
> rises 1.90 → 2.27 M tok/s on the combined basis while the measured combined depth rises
> 145 → 170 tokens per pixel — so the inference line lands at **$527,000, 0.98×** its prior
> value, interval $452,000 – $573,000. **Fleet sizing, the 85% provisioning policy and the
> work-hours bank are unaffected** (the capacity-planning rate moved +2.0%). Correcting only
> the named unit mismatch would have moved the line 19% the wrong way. Full accounting:
> `campaign-cost-model.md` §6b. Still open there: an unexplained 30% rate deficit on the one
> 20–45° both-orbit sample, and no both-orbit rate measured at campaign fleet width.
>
> **REVISED TWICE MORE, 2026-08-08, and the two moves point opposite ways.** The 20–45° sample
> above was a partial in-flight read, and completing that zone-year raised its rate while cutting
> its radar depth from 146.8 to 89.9 — a partial run's whole-cell summary is *biased*, not merely
> imprecise. A separate **weighting** defect then surfaced: optical depth was weighted by campaign
> land while radar depth was weighted by whichever cells we happened to measure, so their sum
> answered no single question. Land-weighting radar gives depth **173 tok/px** and rate
> **2.127 M tok/s**, putting the line at **$573,000** and the campaign total at **$707,000**.
> **Fleet sizing, the 85% policy and the work-hours bank remain unaffected** — the
> capacity-planning rate has moved only −6.2% from the superseded basis, so the near-cancellation
> survived a twelvefold increase in evidence. `campaign-cost-model.md` §6c.
>
> Two residuals, one now permanent: the rate deficit narrowed to **~19% and did not close**, and
> the **35–50° band stays interpolated** because measuring it is decided against — so that
> uncertainty is *carried* rather than closable, and it is the widest single driver of the interval.

> **NEW CONSTRAINT, 2026-08-08: the 85% provisioning policy is now load-bearing for a second
> reason.** It was adopted to stop a fleet idling on a shallow queue. It is also the only thing
> holding the campaign under an **assembly ceiling of ~275 actors per cluster**: assemblies
> serialise on one trailing thread, and inference time per tile falls as actors rise while assembly
> does not, so the two cross. At the planned 228 the assembly tail is ~11 minutes on an ~8-day
> campaign; at matched (268) it crosses, and the widest configuration the cost model lists adds 7.5
> hours. **So fleet width and assembly capacity cannot be decided independently** — anyone raising
> the fleet for throughput is also spending this margin. If a wider fleet is ever wanted the remedy
> is the assembly side, not the fleet: the runner leaves ~39% of its CPU idle at the shipped pool
> width, and that idle capacity is precisely what a fleet above the ceiling would need.

1. **Report the optical-only cells in whatever ships with the data.** OPERA RTC-S1 coverage was
   withdrawn from about a fifth of the land after Sentinel-1B failed in December 2021 and largely
   restored with Sentinel-1C in 2025 — interior Australia and much of Siberia return *zero*
   granules for 2022–2024. `allow_s2_only` is on, so **6.8% of pixel-years across the campaign
   are embedded without radar** rather than missing.

   **No longer a decision.** The Cambridge team validated radar-free embeddings (2026-08-03), so
   ADR 013's blocking follow-up is satisfied and we are not weighing whether to produce them.
   What is left is DESCRIPTIVE: every affected pixel is identifiable after the fact
   (`s1_asc_obs_count + s1_desc_obs_count == 0`) and `scripts/census_s1_coverage.py` maps the
   area in advance, so the campaign reports the share and how those embeddings compare — a
   statistic that ships with the data, not a gate on shipping it. **Attach Cambridge's study
   when available**: a cleared gate whose only evidence is a sentence in a chat log is one
   re-litigation away from being open again.
2. **Run the Phase-4 test geographies.** **Three** sites spanning 120–200 tokens per pixel,
   settling the two remaining throughput questions: whether tok/sec is flat across sequence
   length (confirmed by P2, 2026-08-04 — flat to ±1% across three geographies) and how large
   the per-chunk read floor is. It was four sites and six runs until 2026-08-03 — every site had to be dual-orbit and
   two had to run twice, because validating optical-only output meant masking radar and
   comparing the same pixels. Cambridge validated radar-free embeddings, so that constraint is
   gone and what remains needs neither dual orbits nor paired runs. Site list and rationale in
   `yield-embeddings/docs/global-tessera-test-plan.md`, the **P2 rung**.
3. **Measure the densest zone at two fleet widths (60 and 80).** It decides whether the
   campaign is 5.6 days or 4.5, and it is the only remaining question about the schedule. The
   width model is fitted over roughly 30–60 workers, so 80 is currently an extrapolation.
   This is the last preflight gate.
4. ~~**Raise the Fargate quota.**~~ **DONE 2026-08-06** — prod is at 25,000 vCPU and G-and-VT at
   10,000. Set `max_parallel_ingest` to 61 and `max_workers` to 60.
5. **Public release.** The store is 0.9–1.8 PB. AWS Open Data has lead time and the size
   figure is what the application asks for.
6. ~~**Weight the zone-to-cluster split by work rather than area.**~~ **DONE 2026-07-30.**
   `_partition_by_live_tiles` now balances on `zone_work_weight` — live tiles weighted by
   their latitude band's observation count from §9's census — at the same one-GET cost.
   Measured on the real coverage census at 8 clusters, true-work spread falls from **9.43%
   to 0.04%**. The within-cluster order is unchanged and still sorts on tile counts, which
   is correct: ordering and actor clamping are properties of area, not of work.

   **Extended 2026-08-07: weight is per-year work × the years a zone carries.** The 07-30 fix
   still weighed each zone ONCE — irrelevant while every zone owed a single year, and wrong
   the moment `overlap_years` lets a batch span years (which is now the campaign setting) or a
   repair run leaves uneven gaps: one cluster drains the extra years while the rest sit idle,
   and its finish time is the campaign's. A zone with no years left weighs zero, which
   subsumes the old `known_complete` retag-only case and skips its mask read the same way.
   One caveat inherited from the census: the per-band *split* behind `zone_work_weight` is
   refuted at high latitude while the totals hold — an open check, cost model §9 item 3.
7. **All years dispatch in one batch (`overlap_years=true`).** Worth **~4.8 days against ~6.8**
   at 61 cells and the full 2,500-actor fleet, plus 8 Ray cluster boots instead of 72 (~$8,000
   of the ramp line), and it is what makes GPU quota buy schedule again.

   The enabling change was making the inference window a **per-cell** value carried on every
   work item (`ZoneContext.time_window`) rather than read from the actor's config. That was the
   real blocker: an actor is built once with one config, so a cell of another year streamed
   through it would silently have been inferred over the wrong months, and the session's only
   mismatch check is on `s1_orbit`.

   Three pieces hold it up:

   - **The window travels with the cell.** Bit-identity was the risk, since three loader call
     sites must receive identical kwargs; the fallback is resolved ONCE into a local and that
     value passed to all three, so they cannot drift. One test asserts that passing the
     config's own window explicitly is byte-for-byte identical at `atol=0`; a second asserts
     that a *different* window genuinely changes what is read — the failure the first cannot
     see, since an ignored argument passes every identity test.
   - **A cluster takes `(zone, year)` pairs**, with windows and configs derived per year. A
     literal `time_window_end` override is refused for a multi-year list rather than applied
     to every year in it.
   - **The partition is over ZONES**, so a zone's every year lands in one cluster and its
     assemblies serialize on that cluster's single trailing thread. That is what bounds
     concurrent assembly memory and removes any need for a disjointness scheduler.

   **Fleet-validated 2026-08-06.** Phase 4's P4 rung passed all five checks, and P7 then ran
   two clusters across six both-orbit cells with a same-zone year rollover inside one of them.


---

## 9. What moves the numbers

**The figures in §4 are not derived here.** `campaign-cost-model.md` is the source of truth
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

1. **The GPU fleet, not ingest, sets the schedule.** Past ~52 cells every configuration lands
   near 4.8 days, so extra cells buy buffer and only extra ACTORS buy time.
2. **Inference cost scales with tokens, not pixels — and the tokens are COMBINED S2+S1.**
   Quote combined `tok/sec`. Pixels-per-second mixes machine speed with geography, which made
   the cost model's throughput basis wrong three times before the unit was changed; an
   *optical* tok/sec is a further different unit, and it sat in the model's central division
   until 2026-08-07 (cost model §6b). It is not purely token-bound either — a per-chunk read
   floor scales with pixels and bytes, and stops being hidden where sequences are short.
3. **Ingest worker-hours are width-neutral.** Halve the fleet, double the duration, same
   bill — so fleet width is schedule bought for free, and cost is nearly flat across every
   configuration in §4.
4. **Inference cost is invariant to the ingest configuration.** Same pixels, same rate. The
   ingest configuration decides only how fast mosaics arrive, and therefore how large a GPU
   fleet stays busy.

**Fixed 2026-07-30.** `_partition_by_live_tiles` balanced clusters on tile counts, which is
area, while cost scales with area × observations — so clusters could finish at materially
different times with equal tile counts. It now balances on work (§8 item 6). Two properties
worth keeping in mind if this is ever revisited: the imbalance was **not random**, because
latitude drives it, so a cluster drawing high-latitude zones was heavy in *every* year; and
clusters are long-lived, so the last to finish sets the campaign date and a heavy cluster is
never averaged away.

---

## 9b. Findings since 2026-08-05 that change how the campaign is run

Added 2026-08-06 from the pre-campaign test programme. Only what alters an operational decision or
retires an open question; the figures live in the documents named.

### Added 2026-08-10 — two operational RULES, not just findings

**A change inside the mosaic-content fingerprint closure has a LANDING WINDOW, and the campaign
closes it.** Such a change moves the fingerprint, so a mosaic caught **mid-append** refuses its next
append and needs its interrupted store deleted by hand. Finished mosaics are unaffected. Two such
changes were landed deliberately while nothing was mid-ingest — verified by confirming only the
Prefect worker service was running, not an ingest runner or fleet. `config/providers.py` and the
ingest query modules are inside that closure, so **anything touching the query, the collections or
the provider settings is a between-campaigns change**, not a mid-flight one. This is the operational
face of a property the identity closure already documents; it is recorded here because the closure's
docstring cannot tell you *when* it is safe to act.

**Our own S3 ceiling has already been hit, and it is the DELETE path — and it is not a quota.** The
per-prefix request rate is a service characteristic: no Service Quotas entry, no case to file, so
there is nothing to raise. Every own-bucket refusal observed so far is **bulk mosaic deletion**
hitting the prefix write rate — not ingest, not inference, not assembly. The campaign's own
`cleanup_staging` runs per cell under its own `staging/<zone>/<year>/<run-id>/` prefix, so its
traffic is distributed across prefixes and spread over the campaign's duration, which is a
materially easier problem than the concentrated manual cleanups that produced those refusals.

> **The open question, worth three orders of magnitude:** `DeleteObjects` accepts up to 1,000 keys
> per request, and the production path is `s5cmd rm --all-versions`. Whether it batches is currently
> a belief. **Confirm before launch** — if it does, the steady-state margin is enormous; if it does
> not, moving the cleanup to batched deletion dwarfs anything a quota could have granted. Either
> way a throttled delete costs time, not correctness; the one real harm is delete traffic contending
> with a live measurement.

**The retry policy now has a bound, because only attempt counts had one.** F7 settled that we back
off and retry expansively and fail a cell only under duress. Measuring the ladder showed the gap:
**wall clock was unbounded at all three layers** — a single page fetch is already nine HTTP attempts
across 364 seconds, and the leg, cell and campaign budgets each treat that as one try, so one
deterministically failing query could cost up to twelve leg dispatches. The bound is a **36-hour
wall-clock budget** on the leg-retry loop, checked only when deciding to START another attempt, so a
slow-but-succeeding leg can never be the reason the loop stops. A legitimately slow source loses its
remaining attempts in one flow run and nothing else: the cell returns to the work list, the outer
budgets re-dispatch it, and a re-dispatch resumes from committed dates.

**One upstream failure turned out to be ours to fix.** A catalogue search that refused deterministically
— defeating that whole ladder, three times hours apart — was a **page-size sensitivity**, not missing
data and not our load: the same query answers at 100 items per page and refuses at 250, while the
adjacent year answers at 250. Earth Search's page size is now 100. Note the opposite result already
recorded for CMR-STAC, where *raising* it made 500s worse — page size is a per-service property, so
this is set per provider and must not become a rule.

### Added 2026-08-08 — the failure drills, and what they retired

**Crash recovery of a GPU fleet works, and takes about three seconds.** A hard kill of the flow
body — every `finally` and context manager skipped — was recovered by the terminal hook, which
terminated all five instances at the same second. This **refutes** the belief that Prefect runs
those hooks in the flow's own process. The operational consequence: the orphan-fleet sweep's lack
of any EC2 code path is real but **not on the recovery path**, so no work is needed there. Do not
plan around a leaked Ray fleet as a likely event.

**A dead committer's slot returns by itself in about five minutes.** There is no manual release
procedure to write, and no runbook step to add. A whole-cluster death is a five-minute stall on
commits rather than a deadlock.

**Skipped tiles are real coverage gaps, deterministic and spatially clustered** — byte-identical
sets across independent fills of the same cell. They publish as fill, which is what ocean also
reads as, so the store now records `optical_skips` per year. **What ships with the data should say
so**, alongside the optical-only cells already listed in §8.

**The campaign-day view now carries twelve checks**, and two of them exist because a resume that
silently restarts is otherwise invisible: the work-list ratio (which needs the dispatch intent) and
orphaned staging (which needs none, and therefore catches the operator who meant to resume and
omitted `run_id`). **Both are worth running early in a fill rather than late** — the finding is only
actionable while killing the run still saves the money.

**Assembly is the shard write.** The merge and commit are 37 seconds of a 196-minute assembly, so
there is no second phase to plan around. The staged intermediate is stored **uncompressed**, which
is what sets the duration; if assembly wall-clock ever matters, compressing it is the only lever
with real leverage.

**One monitoring defect worth knowing about, now fixed:** a concurrency limit set to zero while
active read as *healthy* — a total stall reported as health. The view now records such a state
without grading it, because zeroing a gate is also a legitimate way to park the campaign.

**Radar costs about twice as much per chunk as optical, and the OPTICAL token metric cannot see
it.** `t_kept` is the Sentinel-2 mask's first dimension — optical timesteps only — so
`tokens = t_kept × valid_px` measures optical work and misses the radar sequences the forward
pass also encodes. Measured on completed cells: radar-free 2.26–2.93 M optical tok/s per actor,
one orbit 1.60–1.79 M, both orbits 1.23–1.62 M. Consequence for operations: an **optical**
throughput or cost figure is comparable only within one radar status, and `inference_profile.py`
prints its radar basis. **Resolved 2026-08-07: on the COMBINED (S2+S1) basis the spread
collapses** — within a run, both-orbit and one-orbit chunks run at the same combined rate to
1.01–1.08× — so quote combined tok/sec and the comparability problem disappears. Details:
`campaign_inference_profile_2026_08.md` and `campaign-cost-model.md` §6b.

**The cost model's central division mixed two token units — RESOLVED 2026-08-07.** An earlier
version of this paragraph said "not yet resolved" and named the measurement a launch
prerequisite; it has been taken (four cells under the per-chunk radar fields) and the line
re-based on one unit: **$527,000, 0.98× the prior value** — two errors of similar size pointed
opposite ways and nearly cancelled, and fleet sizing is unaffected. See §8's resolved note and
`campaign-cost-model.md` §6b, which also names what is still open (the 20–45° rate sample, and
no both-orbit rate at campaign fleet width).

**Radar observation depth is REGIONAL, not latitudinal — do not model it against latitude.**
Across five measured latitude bands radar depth spans 66–147 tokens per pixel with a
correlation against latitude of +0.009, while optical depth over the same bands gives +0.912.
The deepest radar measured anywhere is the Middle East and the shallowest the Arctic, which
points at the Sentinel-1 observation plan rather than geometry. Radar's *share* of a sequence
does fall with latitude, but only because the optical denominator rises. Planning form: a
constant 90 tokens/px with stated range 66–147 (`campaign-cost-model.md` §6b).

**No contention penalty at 55 concurrent cells — measured on prod at 20,316 vCPU
(2026-08-06).** The ingest line carries no load penalty at full campaign width; the
orchestrator sat at 25% CPU with zero dropped events, placement was exact, and every 503 in
the window was upstream. Figures and the reading rules: `campaign-cost-model.md` §4.

**Cold start is a real term in any restart calculus.** At 160 requested actors the fleet stood
at 60 after 25 minutes and 151 after 60, so widening a fleet by cancel-and-redispatch pays a
fresh `ray up` plus per-worker bringup and model load — the saving is the naive ratio MINUS
that ramp. It is also the standing argument for the chained-cluster strategy. Figures:
`campaign-cost-model.md` §8, the ramp note.

**Never read observation depth, tokens, throughput or cost per chunk off a run still in flight.** A
run sweeps its zone north to south while depth falls with latitude, so a partial run has measured
only its deepest part. Zone 38N read `t_kept` 121 at one third complete and **73** at completion.
The bias is one-directional and predictable, not noise. Duty cycle is the exception and rises
legitimately as actor start-up amortises.

**Greenland and Arctic Canada ship optical-only, and that is now a settled decision** (2026-08-06).
They are not short of radar: zones 23N and 24N publish tens of thousands of **HH/HV** granules and
effectively no VV+VH, so the ingest declines them on polarisation. This is distinct from the
Sentinel-1B gap in §8 and adds ~208 live tiles of deliberately optical-only land. Not EW mode — those
granules report `BEAM_MODE=IW`. If ever revisited it is a model question, not an ingest one.

**The duty-cycle question is answered: ≥97.3%** on 38N-2021 at 60 actors over 9,051 chunks, $1,600,
nothing stalled in 14.7 hours. Fleet-feeding is not a risk to plan around.

**`overlap_years` is cleared for campaign use and is the campaign setting** — all five multi-year
checks passed, with the per-cell-window one evidenced from the store. Its one caveat, that both
zones it ran on were radar-free, is closed by P7: six cells, every one carrying both radar orbits,
with a same-zone year rollover inside one cluster.

**Prod is ready (verified 2026-08-06).** Coverage mask built, all 112 land-zone ROIs exported, all
120 store groups seeded, the campaign deployment set registered in its branch-scoped form, the
crash-recovery automations armed, and the Prefect server resized. The seven stray unsuffixed
deployments left by an earlier registration attempt have been removed, so a dispatch to a default
name can no longer reach old code with a thinner parameter set.

**Operate prod from the branch, not from `main`.** Every prod deployment is the `-global-tessera`
form, and the branch-scoped registration is the supported path until the global code merges. A
consequence worth knowing: management scripts that resolve a deployment by name need `--branch
global-tessera` on prod, and fail with a bare object-not-found without it.

## 10. Evidence

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

**Three of those carry withdrawn claims next to the claim rather than deleted.** That is deliberate:
each was wrong in a way worth not repeating, and a reviewer who only sees the corrected number
learns nothing about how it went wrong. The withdrawals are marked in bold and dated.

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
  validation it left open. **The flag is ON for this campaign** (§2); that validation is the
  **P2** rung's job.
- `tests/unit/test_cluster_balance.py` — the runnable diagnostic behind §3; rerun it rather
  than trusting the figures if the mask is rebuilt.
