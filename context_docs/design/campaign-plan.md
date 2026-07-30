# Global TESSERA campaign — the plan

**Dated 2026-07-29.** The operational plan for building 9 years × 111 land zones of 10 m
embeddings into one Icechunk store. Everything here is settled unless it appears in §8,
which is the list of decisions still open before launch.

This document is the entry point and the source of truth for **operations** — what runs,
with what settings, in what order, and what to do when it breaks.
[`campaign-cost-model.md`](campaign-cost-model.md) is the source of truth for **figures**:
every cost, rate, fleet size and the arithmetic behind them. Numbers appear here as results
with a pointer, never as derivations; where the two disagree, the cost model is right.

---

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
   ├── year 2017 ─┬── cluster 1 ── zones: 35N → 12S → 04N → …   (densest first)
   │              ├── cluster 2 ── zones: 38N → …
   │              └── … 8 clusters, each opening on one of the 8 densest zones
   ├── year 2018 ─── (same, after 2017 completes)
   └── … 2017–2025, serial
```

**Years run serially.** Concurrent zone-years are safe — different groups rebase cleanly —
but two *years of the same zone* write the same group's attributes and cannot auto-merge.
Year-serial guarantees that more strongly than necessary; relaxing it needs a
zone-disjointness scheduler and is not planned.

---

## 2. Settings

Defaults are in `run_global_campaign`. **★ marks the four that need changing from their
shipped defaults.**

| parameter | value | why |
|---|---|---|
| `fill_strategy` | `"chained-clusters"` | one cluster per zone-set, not per zone-year |
| `max_parallel_clusters` | 8 | balance holds to ~16; 8 keeps each cluster opening on a top-8 zone |
| **`max_parallel_ingest`** | **40 → 45** ★ | fleet-wide cap on simultaneous zone-ingests. 45 is the knee *while years are serial*: past it the year barrier makes extra cells worthless. Without the barrier it becomes 61 (§4, §8) |
| `max_zone_attempts` | 2 | bounded re-dispatch **per year** — the year is the retry unit, see §6 |
| **`ingest_settings.max_workers`** | **50 → 60** ★ | S2 fleet width. Past the knee this is the ONLY thing that shortens the campaign, because the longest single zone sets the floor (§4) |
| `ingest_settings.s1_worker_fraction` | 0.22 | → 13 workers per S1 orbit at the recommended 60w, sized to finish inside S2 |
| `ingest_settings.batch_days` | 30 | S1 batch length |
| **`num_actors`** | **20 → ~228** ★ | GPU actors per cluster: 85% of what ingest can feed, so the fleet never idles and keeps headroom for a restart (§4) |
| `s1_orbit` | `"both"` | downgrades per zone when an orbit has no imagery |
| `cleanup_mosaics` | `true` | **required** — the storage figure depends on it |
| `allow_partial_window` | `false` | a zone-year is a full calendar year or it fails |
| **`allow_s2_only`** | **`false` → `true`** ★ | ON for the global campaign. A fifth of the land has no radar for 2022–24 (§4); without this those pixels produce nothing. ADR 013 does not consider the combination validated — see §8 |
| `force_staging_reuse` | `false` | escape hatch: reuse staged tiles across an inference-code change you know is output-neutral |
| `force_staging_restage` | `""` | escape hatch: any new token forces a fresh staging prefix, for a change the source hash cannot see (a library upgrade) |
| commit limit | derived, `min(clusters, 8)` | never set by hand — and it bounds commits IN FLIGHT (~1 s each), not concurrent assemblies |

The commit gate and the ingest gate are **Prefect global concurrency limits**, because
clusters are separate flow runs on separate machines and only a server-side gate can bound
them together. Both must exist before launch or the flow fails closed.

---

## 3. Order

Zones are dealt to clusters **longest-processing-time first**, so each of the 8 clusters
opens on one of the 8 densest zones and totals land within 0.0% of each other. Within a
cluster, zones run densest to sparsest.

That ordering does two jobs and is deliberately **not** a barrier: a cluster requests GPUs
as soon as *any* mosaic in its opening window lands, not its densest one. Blocking on the
densest — which is also the slowest to ingest — idled a fleet about six hours per
cluster-year with finished mosaics already on disk.

---

## 4. Sizing, cost, and the two things that actually bind

**Figures below are results, not derivations.** [`campaign-cost-model.md`](campaign-cost-model.md)
is the source of truth for every number here and the arithmetic behind it; this section states
what to run and what it costs. Where the two disagree, the cost model is right.

**Costed in tokens.** Inference consumes a sequence per pixel, so cost scales with
`tokens = pixels × observations`, not with pixels. The campaign is **1.98 × 10¹⁵ tokens** —
1.363 × 10¹³ pixels at a land-weighted 145 observations each, censused from CMR (radar) and
Sentinel-2 STAC (optical) — at a reference **≈1.9M tok/sec** per worker. **Quote tok/sec, not
px/s**: pixels-per-second mixes machine speed with geography, and the pipeline has been
logging tok/sec all along for exactly this reason. The equivalent px/s here is 13.1K.

**Years run serially, so a year cannot finish faster than its longest single zone** — about
17.3 hours at 50 workers, 15.0 at 60. That floor is what shapes everything:

```
   makespan per year = max( longest zone ,  total work / cells )
                            ^ only WIDTH     ^ only CELLS shorten
                              shortens this    this, and only up to 45
```

| | 40 × 50w<br>shipped | 45 × 50w<br>the knee | **45 × 60w**<br>**recommended** | 45 × 80w<br>target |
|---|---|---|---|---|
| Fargate vCPU | 12,640 | 14,220 | **16,740** | 22,140 |
| Ingest wall clock (9 yr) | 7.2 d | 6.5 d | **5.6 d** | 4.5 d |
| Mosaic supply | 5.76/h | 6.41/h | **7.42/h** | 9.22/h |
| GPU fleet to provision | 1,416 | 1,576 | **1,824** | 2,267 |
| — actors per cluster | 177 | 197 | **228** | 283 |
| **Campaign wall clock** | ~8.7 d | ~7.8 d | **~6.8 d** | ~5.5 d |
| **Campaign cost** | $670,000 | $671,000 | **$672,000** | $674,000 |

**Every configuration costs the same to within 0.5%.** Inference is the same pixels at the
same rate; ingest worker-hours are width-neutral. The decision is wall clock, bought with
Fargate quota — nothing here is a cost trade.

**Above 45 cells, extra Fargate quota buys nothing — *while years run serially*.** Not
schedule, not supply, not usable GPU fleet. An earlier version of this plan asked for 71 cells
and 23,000 vCPU; 45 × 60w is nearly two days faster on 16,740.

> **The knee is a consequence of the year barrier, not of ingest (established 2026-07-30).**
> Years are serial only because two years of one zone rewrite that zone group's
> `years_complete`/`runs` attrs and cannot auto-merge (§6). Nothing else requires it: every
> chunk and shard is **1 in the time dimension**, so different years of one zone write
> strictly disjoint objects. **Remove the barrier and the knee moves from ingest to the GPU
> fleet** — see §8 item 7 for the payoff and what it depends on.
>
> The second-order consequence is worth knowing before asking for quota. **Past ~52 cells the
> 2,500-actor fleet, not ingest, sets the schedule**, so extra cells buy *buffer* rather than
> speed:
>
> | cells | vCPU | ingest | provisioning | campaign |
> |---|---|---|---|---|
> | 52 | 19,344 | 4.80 d | 100% | ~4.8 d |
> | **61** | **22,692** | **4.09 d** | **85%** | **~4.8 d** |
> | 66 | 24,552 | 3.78 d | 79% | ~4.8 d |
> | 80 | 29,760 | 3.12 d | 65% | ~4.8 d |
>
> A deeper buffer is worth having — it is what absorbs a failed ingest cell — but it is not a
> speed-up. **To go faster, ask for ACTORS, not cells:** 2,750 actors (67 cells) is ~4.4 d,
> 3,000 (73 cells) ~4.0 d. 66 cells at proper 85% provisioning wants 2,704 actors, an ~8% bump.

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

1. **Fargate quota ≥ 17,000 vCPU** in us-west-2. The existing 2,500-actor GPU quota covers
   every year-serial configuration in §4 — the recommended fleet is 1,824 and even 80 workers
   wants only 2,267. **It stops being slack if the year barrier is removed** (§8 item 7): that
   configuration provisions the full 2,500 at 61 cells and wants ~22,700 vCPU, so the two quota
   asks are coupled and the barrier decision should come first. Fargate quota has lead time;
   start there.
2. **Prefect concurrency limits provisioned** — `tessera-global-ingests` and
   `tessera-global-commits`. Both gates are strict and fail closed on a missing limit.
3. **Coverage/land mask built** for all 120 zones, and its `registry_sha256` frozen. A
   mask rebuild mid-campaign invalidates every completed zone-year's fingerprint.
4. **Store seeded** — all zone groups, all 9 year slots, `geoemb:model` and `checkpoint_id`
   stamped. The seed is the only writer of the time axis.
5. **Model checkpoint staged** at `{inputs}/models/` and matching the seed.
6. **Deployments registered** for `ingest-zone-year`, the chained fill, and the campaign.
7. **A single dense zone-year end to end** — see §8; this is the last gate.

---

## 6. Failure handling

Three mechanisms, all shipped.

**Resume.** An interrupted mosaic is picked up, not cleared. Icechunk commits a date's time
slot atomically with its pixels, so a date present is complete and a date absent was never
started. Resume assumes the same land mask, which the manifest enforces on every write.

**Retry.** A failed zone does not sink the campaign. Each year gets up to
`max_zone_attempts` rounds, and every round re-reads the *store* for what is still missing,
so a retry never repeats a zone that landed.

> **The year is the retry UNIT, which is why removing the barrier (§8 item 7) is not just a
> loop change.** Drop the year and there is no unit left, and a naive whole-run round is much
> worse granularity: a cell failing on day one would not be retried until every cluster had
> finished its entire multi-year list. The intended fix is a **child-level retry pass** — the
> chained fill already records per-cell failures and retains their mosaics for staged resume,
> but does not retry; giving it one pass over its own failures keeps recovery local, and
> because a child owns its zones and works them in order that retry cannot collide with
> itself. The parent's rounds then handle only what a child could not.
>
> Note this needs two distinct bounds rather than today's one: attempts *within* a child and
> attempts *across* the run. Overloading `max_zone_attempts` for both would silently change
> what it means. Rounds stop early when one makes no progress
— that is what a deterministic failure looks like from the driver, and it wants a human
rather than another fleet. The campaign raises at the very end listing every unfilled cell.

**Re-run.** A fresh campaign run recomputes nothing already done: completed work is
filtered at the work list, at the per-zone ingest marker, at date-level resume inside
ingest, and at staged-tile resume inside inference. Two deliberate exceptions — a changed
parameter raises rather than silently mixing configurations, and a change to the **inference
code** invalidates staged tiles by design.

> **Narrowed 2026-07-30.** That last exception used to fire on any change to the *build* — the
> staging fingerprint was the resolved AMI ID plus the source tarball's ETag, so re-baking the
> AMI or a hotfix anywhere in the repo abandoned every staged tile. It is now a hash of the
> inference source only (`inference_code_identity`), so an orchestration or ingest fix reuses
> staging. The AMI is still resolved once and pinned into every fill, which is now the only
> thing stopping one run straddling two images. Two overrides exist for the judgement calls the
> hash cannot make (§2).

There are **no Prefect-level retries** on any ingest flow, deliberately.

---

## 7. What to watch

| signal | why it matters |
|---|---|
| **Mosaic backlog** (completed, not yet inferred) | the $3,000 storage figure assumes prompt deletion; a four-week backlog is ~400 TB and ~$9,200/month, and sooner a bucket-capacity problem |
| **GPU duty cycle** | the whole of §4; a fleet below ~80% busy is being paid for idling |
| **Cells actually running** vs the cap | the ingest gate is fleet-wide; if the limit is missing the flow fails closed, but if it is set too low nothing complains |
| **Zones failing twice** | the retry loop stops on no-progress; these need a human |
| **Coverage-gate rejection rate** | 365 dates per zone-year is measured on three zones; heavy rejection changes both time and cost |

---

## 8. Open before launch

Settled since the last revision, and not to be re-opened: **the model is v1.1**, **GPUs are
on-demand**, and the permanent store goes to **AWS Open Data**.

1. **Describe the optical-only cells in whatever ships with the data.** OPERA RTC-S1
   coverage was withdrawn from about a fifth of the land after Sentinel-1B failed in December
   2021 and largely restored with Sentinel-1C in 2025 — interior Australia and much of
   Siberia return *zero* granules for 2022–2024. `allow_s2_only` is on, so **6.8% of
   pixel-years across the campaign are embedded without radar** rather than missing. ADR 013
   does not consider that combination scientifically validated, so this is a documentation
   and caveat question, not a coverage hole. Every affected pixel is identifiable after the
   fact (`s1_asc_obs_count + s1_desc_obs_count == 0`), and
   `scripts/census_s1_coverage.py` maps the area in advance.
2. **Run the Phase-4 test geographies.** Four sites spanning 120–200 tokens per pixel,
   chosen so one set of runs settles three things: whether tok/sec is flat across sequence
   length (the assumption the cost model now rests on), how large the per-chunk read floor is,
   and whether optical-only embeddings are comparable — the ADR-013 gap that
   `allow_s2_only=true` makes live for 6.8% of pixel-years. Site list and rationale in
   `yield-embeddings/docs/global-tessera-test-plan.md`, the **P1b rung**.
3. **Measure the densest zone at two fleet widths (60 and 80).** It decides whether the
   campaign is 5.6 days or 4.5, and it is the only remaining question about the schedule. The
   width model is fitted over roughly 30–60 workers, so 80 is currently an extrapolation.
   This is the last preflight gate.
4. **Fargate quota to ~17,000 vCPU**, then set `max_parallel_ingest` to 45 and
   `max_workers` to 60. Quota has lead time; start it first.
5. **Public release.** The store is 0.9–1.8 PB. AWS Open Data has lead time and the size
   figure is what the application asks for.
6. ~~**Weight the zone-to-cluster split by work rather than area.**~~ **DONE 2026-07-30.**
   `_partition_by_live_tiles` now balances on `zone_work_weight` — live tiles weighted by
   their latitude band's observation count from §9's census — at the same one-GET cost.
   Measured on the real coverage census at 8 clusters, true-work spread falls from **9.43%
   to 0.04%**. The within-cluster order is unchanged and still sorts on tile counts, which
   is correct: ordering and actor clamping are properties of area, not of work.
7. **Remove the year barrier, and rework the retry loop with it.** Years are serial only
   because two years of one zone rewrite that group's `years_complete`/`runs` attrs. **Both
   attrs are keyed by year and each writer inserts only its own key**, so there is no semantic
   conflict — icechunk's `ConflictDetector` simply treats attrs as opaque. And nothing else
   requires the barrier: chunks and shards are 1 in the time dimension, so different years of
   one zone are strictly disjoint on disk.

   Payoff: the knee moves from ingest (45 cells) to the GPU fleet, giving **~4.8 days against
   ~6.8** at 61 cells and the full 2,500-actor quota, plus 8 Ray cluster boots instead of 72
   (~$8,000 of the ramp line). It also unblocks buying schedule with GPU quota again.

   Three things it depends on, and the third is the real work:

   - **Same-zone serialisation must still hold.** The intended shape gets it structurally:
     dispatch 8 children once for the whole run, each owning a zone-set and working a
     concatenated multi-year list. The partition is deterministic given the frozen mask, so
     cluster *k* owns 35N in every year and works its list in order — two years of 35N are
     never concurrent, and no storage change is needed.
   - **Optionally, make the attrs conflict-free anyway** so the constraint stops being
     something every future change must reason about. Cheapest is to split the commit: chunk
     data first (disjoint, always rebases), then a small second commit that re-reads and merges
     the attrs in its own session, retried on conflict. Two commits per zone-year instead of
     one — ~2,000 total, far under icechunk's "tens of thousands" — and it independently fixes
     a real defect: today an attr collision discards an entire assembly rather than retrying a
     one-second commit.
   - **The retry loop must be rebuilt**, because the year IS the current retry unit. See §6.


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

1. **Years are serial, so the longest single zone floors each year.** This is why cells stop
   helping at 45 and why width is the only lever past it.
2. **Inference cost scales with tokens, not pixels.** Quote `tok/sec`. Pixels-per-second
   mixes machine speed with geography, which made the cost model's throughput basis wrong
   three times before the unit was changed. It is not purely token-bound either — a
   per-chunk read floor scales with pixels and bytes, and stops being hidden where sequences
   are short.
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

## 10. Evidence

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
  P1b rung's job.
- `tests/unit/test_cluster_balance.py` — the runnable diagnostic behind §3; rerun it rather
  than trusting the figures if the mask is rebuilt.
