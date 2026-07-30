# Global TESSERA campaign — the plan

**Dated 2026-07-29.** The operational plan for building 9 years × 111 land zones of 10 m
embeddings into one Icechunk store. Everything here is settled unless it appears in §8,
which is the list of decisions still open before launch.

This document is the entry point. It states *what will run and with what settings*; the
evidence behind each choice lives in the documents linked at the bottom, and is not
repeated here.

---

## 1. Shape

Ingest and inference are **staged, not sequential**. Ingest builds a zone-year's mosaic on
Fargate; inference consumes it on GPU and the mosaic is deleted. The two use different
resource pools, so overlapping them costs neither side.

```
   sequential :  ingest  +  inference          ~13 days, and 5.6 PB held at once
   staged     :  max(ingest, inference) + lag  ~5 days,  and ~330 TB held at once
```

Staging is **not an optimisation on top of the plan — it is the plan.** Without it the
full 5.6 PB of intermediate mosaics is held simultaneously and storage alone becomes about
$128,000 a month.

Within that, work is organised as **8 long-lived Ray clusters**, each owning a set of UTM
zones and streaming them through one persistent set of GPU actors. A cluster pays `ray up`
and model load once for its whole set rather than once per zone-year.

```
   campaign
   ├── year 2016 ─┬── cluster 1 ── zones: 35N → 12S → 04N → …   (densest first)
   │              ├── cluster 2 ── zones: 38N → …
   │              └── … 8 clusters, each opening on one of the 8 densest zones
   ├── year 2017 ─── (same, after 2016 completes)
   └── … 9 years, serial
```

**Years run serially.** Concurrent zone-years are safe — different groups rebase cleanly —
but two *years of the same zone* write the same group's attributes and cannot auto-merge.
Year-serial guarantees that more strongly than necessary; relaxing it needs a
zone-disjointness scheduler and is not planned.

---

## 2. Settings

Defaults are in `run_global_campaign`. Only the starred row needs changing.

| parameter | value | why |
|---|---|---|
| `fill_strategy` | `"chained-clusters"` | one cluster per zone-set, not per zone-year |
| `max_parallel_clusters` | 8 | balance holds to ~16; 8 keeps each cluster opening on a top-8 zone |
| **`max_parallel_ingest`** | **40 → 71** ★ | fleet-wide cap on simultaneous zone-ingests; see §4 |
| `max_zone_attempts` | 2 | bounded re-dispatch per year |
| `ingest_settings.max_workers` | 50 | S2 fleet width; **already the default, do not raise to 60** |
| `ingest_settings.s1_worker_fraction` | 0.22 | → 11 workers per S1 orbit, sized to finish inside S2 |
| `ingest_settings.batch_days` | 30 | S1 batch length |
| `num_actors` | 20 → **~313 (v1.1) or ~229 (v2)** | GPU actors per cluster; match the fleet to ingest supply, see §4 |
| `s1_orbit` | `"both"` | downgrades per zone when an orbit has no imagery |
| `cleanup_mosaics` | `true` | **required** — the storage figure depends on it |
| `allow_partial_window` | `false` | a zone-year is a full calendar year or it fails |
| `allow_s2_only` | `false` | not validated for production; see the ADR |
| commit limit | derived, `min(clusters, 8)` | never set by hand |

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

## 4. Sizing, cost, and the one expensive mistake

Full derivation in [`campaign-cost-model.md`](campaign-cost-model.md).

Costed at the **14K px/s capacity-planning basis**. An earlier version of this table used
21K, a mid-density while-processing rate that this dense-weighted campaign does not earn;
the cost model's §6 explains the correction. Every fleet figure below is larger as a result.

| | shipped (40 cells) | **recommended (71 cells)** |
|---|---|---|
| Fargate vCPU | 12,640 | **22,436** |
| Ingest wall clock | 7.9 days | **4.5 days** |
| Mosaic supply | 5.24 zone-yr/h | **9.30 zone-yr/h** |
| **GPU fleet that stays busy — v1.1** | **1,419** | **2,518 — over the 2,500 quota** |
| **GPU fleet that stays busy — v2 Large** | **1,032** | **1,831** |
| Ingest cost | ~$121,000 | ~$121,000 |
| Inference cost — v1.1 | $503,000 | $503,000 |
| Inference cost — v2 Large | $366,000 | $366,000 |
| **Campaign total — v1.1** | **$629,000** | **$629,000** |
| **Campaign total — v2 Large** | **$492,000** | **$492,000** |

Both models are carried because the model choice is still open (§8) and it must be settled
before the store is seeded — a store's advertised model identity is write-once.

**GPUs are on-demand.** Spot is excluded by decision, not by oversight: sustaining ~1,700
g6e instances for days makes interruption a certainty, and a campaign that stalls on
capacity is worse than one that costs more. Settled; do not re-open.

**Size the GPU fleet to the ingest supply rate, never to the quota.** A zone-year costs
about **271 GPU-hours** on v1.1. Provisioning the full 2,500-actor quota against 40-cell
ingest runs the fleet at 57% duty and burns roughly **$407,000 of idle GPU time** — more
than three times the ingest bill.

**At 71 cells on v1.1 the matched fleet is 2,518 GPUs (~313 actors across 8 clusters), which
is above the 2,500-actor quota.** In that configuration the quota, not ingest, is the
constraint, and there is no idle burn to avoid. On v2 Large the matched fleet is 1,831
(**~229 actors**) — v2 is 1.375× faster and so needs less fleet for the same ingest, which
leaves 27% of the quota idle if it is provisioned anyway, worth **$124,000**.

Note the direction that pushes: a faster model makes an oversized fleet **worse**, because
each zone-year is consumed sooner and the fleet starves longer. The model choice and the
fleet size are coupled and must be decided together — and which of them binds depends on the
ingest cap, so **decide the cap first.**

**A matched fleet is necessary but not sufficient.** Matching balances supply against demand
*on average*; it does not stop a fleet idling at the start of a cluster-year, when the
opening window has produced only one shallow mosaic. Modelled in
`tests/unit/test_gpu_starvation.py`: booting on the first mosaic wastes about **108,400
GPU-hours (~$202,000)** across the campaign, and holding the boot until **3.25 work-hours** of
pixels are queued removes it for about **26 hours** of added schedule. One figure serves both
models, because a matched fleet's consumption rate is what the unit is denominated in.
Recommended, **not yet shipped.**

Raising the ingest cap costs nothing measurable and is worth 3.4 days and 732 GPUs of
usable fleet. It is the single highest-value setting in this document.

---

## 5. Before launch

1. **Fargate quota ≥ 23,000 vCPU** in us-west-2, and **GPU quota** for the chosen fleet —
   which on v1.1 at 71 cells means **more than the current 2,500 actors**. Quota increases
   have lead time; start here.
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
so a retry never repeats a zone that landed. Rounds stop early when one makes no progress
— that is what a deterministic failure looks like from the driver, and it wants a human
rather than another fleet. The campaign raises at the very end listing every unfilled cell.

**Re-run.** A fresh campaign run recomputes nothing already done: completed work is
filtered at the work list, at the per-zone ingest marker, at date-level resume inside
ingest, and at staged-tile resume inside inference. Two deliberate exceptions — a changed
parameter raises rather than silently mixing configurations, and a code change invalidates
staged tiles by design.

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

1. **Which model.** v2 Large runs **1.375× faster** on the branch's own per-model rate
   (22,000 against 16,000 px/s in `inference/actors.py`), which is worth
   **$122,000–$137,000** and cuts the matched fleet from 2,518 to 1,831. It also emits 128-D
   natively rather than truncating 192-D. Against that: the rate is a strategy-only
   planning constant whose calibration is not written down anywhere, and the model identity
   a store advertises is write-once — so this must be decided *before* seeding, not after.
2. **One instrumented dense-zone run.** Settles the throughput question (14K vs 21K px/s,
   worth $168,000 — and it decides whether the GPU quota binds at all) and puts a documented
   number behind the 1.375× ratio, in one run. This is the last preflight gate.
3. **Public release.** The permanent store is 0.9–1.8 PB and goes to AWS Open Data, which
   sponsors the storage — so it is not costed. It has lead time, and the size figure is
   what the application will ask for.
4. **Ingest cap 40 → 71**, conditional on the Fargate quota landing.

---

## 9. Evidence

- [`campaign-cost-model.md`](campaign-cost-model.md) — costs, GPU fleet sizing, the idle-burn
  arithmetic, and the v2 comparison.
- [`campaign-cluster-sizing.md`](campaign-cluster-sizing.md) — the coverage census, how zones
  divide across N clusters, and why commit concurrency is a non-issue.
- [`ingest_optimization_campaign_2026_07.md`](ingest_optimization_campaign_2026_07.md) — every
  ingest measurement: what each change bought and what failed.
- [`../decisions/011-campaign-zone-ingestion.md`](../decisions/011-campaign-zone-ingestion.md)
  — why the campaign triggers ingestion per zone.
- [`../decisions/008-global-store-architecture.md`](../decisions/008-global-store-architecture.md)
  — the store layout, write model, and commit behaviour.
- [`../decisions/013-optional-s1-s2-only-pixels.md`](../decisions/013-optional-s1-s2-only-pixels.md)
  — the `allow_s2_only` flag and why it is off.
- `tests/unit/test_cluster_balance.py` — the runnable diagnostic behind §3; rerun it rather
  than trusting the figures if the mask is rebuilt.
