# Global TESSERA campaign — full cost model

**Dated 2026-07-29.** Extends the July-27 ingest cost estimate — which costed **ingest
only** — to the whole campaign: ingest, inference, assembly, and the permanent output
store. Nine years × 111 land zones. The ingest measurements it rests on are recorded in
[`ingest_optimization_campaign_2026_07.md`](ingest_optimization_campaign_2026_07.md);
section references below of the form "ingest estimate §N" point at that working note.

Every input is either measured (and cited) or derived from a measured input (and marked
as derived). Two things are neither, and they are called out in §9 rather than buried.

---

## 1. The headline

| | |
|---|---|
| Ingest (Fargate) | $115,000 – $126,000 |
| **Inference (GPU, on-demand)** | **$391,000 – $542,000**, plan on **$470,000** |
| Assembly | ~$200 |
| Mosaic storage (transient) | ~$3,000 |
| Ray cluster ramp (72 boots, §4) | ~$9,000 |
| **Campaign total** | **$519,000 – $684,000**, plan on **$604,000** |

The "plan on" figure is the **15K px/s basis** (§6) with ingest at its midpoint. That rate is
now *derived* rather than borrowed: cost scales with observations per pixel, and weighting the
world's land by latitude shows the campaign is **0.87× the tokens per pixel of the Iowa ROI
every measured rate comes from** — so it runs slightly faster than that ROI, not slower. The
range spans Iowa's low end with no geographic gain (13K) to its high end with the full gain
(18K).

**The model is v1.1.** v2 Large was evaluated and is not being used. Its figures have been
removed rather than kept alongside, because carrying two columns through every table was
the main source of complexity in earlier versions of this document and the choice is now
settled. What the evaluation found is preserved in §6 for the record.

The permanent embeddings store (0.9–1.8 PB) is **not costed here**: it goes to AWS Open
Data, which sponsors the storage. Sizing it still matters for bucket planning — see §7 —
but it is not a line on this bill.

**GPUs are on-demand. Spot is not costed here and is not an option** — sustaining 1,400 to
2,500 g6e instances for days makes interruption a certainty rather than a risk, and a campaign
that stalls on capacity is worse than one that costs more. Settled; do not re-open.

**Four findings worth acting on.**

1. **Inference, not ingest, is where the money is** — three to four times the ingest bill.
   Every optimisation effort so far has gone into the cheaper half.
2. **Inference cost is invariant to the ingest scenario.** Same pixels, same throughput,
   same GPU-hours whether you run 40 cells or 71. What the ingest scenario changes is
   whether the GPU fleet has anything to do.
3. **Fleet sizing is the largest lever, worth $200,000 – $300,000** — the cost of
   provisioning to the GPU quota rather than to what ingest can actually feed (§5). The
   matched fleet never reaches the 2,500-actor quota in any configuration, because ingest is
   capped by something the quota cannot buy (finding 4). **The policy is to run
   deliberately UNDER the matched fleet**, which makes idle burn structurally zero and buys
   headroom for ingest restarts — see §5.
4. **Ingest wall clock has a floor that more cells cannot break, and it binds at about 45
   cells.** Years run serially, so a year cannot finish faster than its single longest zone —
   about 17.3 hours at 50 workers. Past ~45 concurrent cells, extra Fargate quota buys
   **nothing**: not schedule, not supply, not usable GPU fleet. Past the knee only fleet
   *width* helps, which inverts the "narrow fleets, more cells" advice that held while
   throughput was the constraint (§4). An earlier version recommended 71 cells and 23,000
   vCPU on a wall clock that ignored the year barrier.

---

## 2. Scope

**In:** Fargate ingest (S2 + both S1 orbits), GPU inference, assembly, S3 requests,
transient mosaic storage, permanent output storage.

**Out:** the coverage/land-mask build (one-off, hours); Prefect control plane; data
transfer (everything is in us-west-2, so there is no egress); engineering time.

**Two units that must not be confused**, carried over from the ingest estimate:

```
  COST       needs the SUM of fleet-hours    — concurrent fleets all burn at once
  WALL CLOCK needs the MAX within a cell     — a cell lasts as long as its longest fleet
```

---

## 3. Inputs

| input | value | source |
|---|---|---|
| Land zones | 111 (census lists 112 with land; the estimate uses 111) | ingest estimate §4 |
| Campaign years | 9 → **999 zone-years** | settled |
| Live 2048-px tiles | 360,953 per year | coverage census, `campaign-cluster-sizing.md` |
| **Pixels inferred** | **1.363 × 10¹³** | 360,953 × 2048² × 9 |
| S2 duration basis | 6,354 cell-hours at 60 workers | ingest estimate §5 |
| S2 worker-hours | 381,240 | 6,354 × 60 |
| S1 worker-hours | 39,600 – 71,100 | ingest estimate §6 |
| Fargate | $0.04048/vCPU-h, $0.004445/GB-h | ingest estimate §6 |
| Worker | 4 vCPU, 16 GiB → **$0.2330/worker-hour** | derived |
| g6e.xlarge | $1.861/h **on-demand** (spot excluded by decision) | `docs/providers/aws.md` |
| Inference throughput | **15K** planning basis; 13K / 18K as bounds | derived, §6 |
| Embedding output | int8, 128 dims | `config/store_layout.py` |

---

## 4. Ingest — the two scenarios

A cell is **three concurrent fleets**: S2 plus one S1 deployment per orbit. Each fleet
costs `workers × 4 + 4 (scheduler) + 4 (runner)` vCPU, and the parent flow adds 4. S1's
width is derived, not set: `s1_worker_fraction = 0.22` of the S2 fleet.

```
  vCPU per cell  =  (S2w × 4 + 8)  +  2 × (round(0.22 × S2w) × 4 + 8)  +  4
```

> **The shipped S2 default is 50 workers, not 60.** `IngestSettings.max_workers` is 50
> (`config/ingest.py`), which puts S1 at 11 per orbit rather than the 13 measured. The
> 60-worker figures in the July-27 estimate describe the configuration that was *measured*,
> not the one that will run. Both are given below so the comparison is honest either way;
> the duration basis of 6,354 cell-hours was measured at 60w and is scaled by 60/S2w for
> other widths, which is width-neutral and therefore the conservative direction.

> **How conservative, quantified.** Scaling by `60/W` assumes duration is inversely
> proportional to width. It is not: the canonical width model (ingest estimate §3.10) is
> `T(W) = 36.3 + 7896/W` s/date, whose **36.3 s width-independent residual** means a
> narrower fleet loses less than proportionally. `T(50)/T(60) = 1.157`, against the 1.200
> this table applies. Every wall clock below is therefore **3.6% pessimistic** and every
> supply rate 3.6% low: the 7.9-day row is really ~7.6 days and the 4.5-day row ~4.3, with
> supply 5.44 and 9.65 zone-yr/h. The table is left on `60/W` deliberately — one
> conservative scaling is easier to reason about than two — but the matched-fleet figures in
> §5 inherit the same 3.6% understatement and should not be treated as tight.

### Years are serial, so a year cannot finish faster than its longest zone

**This is the correction that matters most in this document, and it inverts the advice an
earlier version gave.** `run_global_campaign` loops years with a hard barrier — every zone
of year N completes before year N+1 dispatches. So each year's makespan has a floor:

```
   makespan per year  =  max( longest single zone-year ,  total work / cells )
                              ^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^
                              adding cells cannot           adding cells
                              shorten this                  shortens this
```

Evaluating the per-zone fit over the real census: total work is **771 cell-hours per year**
at 50 workers and the longest zone (**35N, 17.3 h**; 38N is 17.3 and 47N 17.1, so this is a
plateau, not one outlier). The two terms cross at **45 cells**:

| cells | vCPU | makespan/yr | **9-year wall clock** | supply | matched fleet |
|---|---|---|---|---|---|
| **40 × 50w — shipped** | 12,640 | 19.3 h | **7.2 days** | 5.76 zone-yr/h | 1,455 |
| **45 × 50w — the knee** | **14,220** | **17.3 h** | **6.5 days** | 6.41 | 1,619 |
| 53 × 50w | 16,748 | 17.3 h | 6.5 days | 6.41 | 1,619 |
| 71 × 50w | 22,436 | 17.3 h | **6.5 days** | 6.41 | 1,619 |
| 80 × 50w | 25,280 | 17.3 h | 6.5 days | 6.41 | 1,619 |

**Past 45 cells, additional Fargate quota buys nothing at all** — not schedule, not supply,
not usable GPU fleet. An earlier version of this document recommended 71 cells and a 23,000
vCPU quota on a wall clock of 4.5 days computed as `total ÷ cells`, which silently assumed
work could be spread across the year boundary. It cannot — and 45 × 60w reaches 5.6 days on
*less* quota than that plan asked for.

### Past the knee, only fleet WIDTH shortens the campaign

Once the longest zone is the binding constraint, the only way to go faster is to make that
zone faster, and that means more workers on it — the exact opposite of the "narrow fleets,
more cells" reasoning that held while throughput was the constraint.

| config | vCPU | longest zone | **9-year wall clock** | matched fleet |
|---|---|---|---|---|
| 45 × 50w | 14,220 | 17.3 h | 6.5 days | 1,619 |
| **45 × 60w** | **16,740** | **15.0 h** | **5.6 days** | **1,873** |
| 45 × 70w | 19,260 | 13.3 h | 5.0 days | 2,109 |
| 45 × 80w | 22,140 | 12.0 h | 4.5 days | 2,329 |
| 45 × 90w | 24,660 | 11.1 h | 4.2 days | 2,562 |

**45 cells at 80 workers is strictly better than 71 cells at 50**: slightly *less* Fargate
quota (22,140 against 22,436) and two days faster. Worker-hours are width-neutral, so this
costs nothing on the ingest bill — it is pure schedule, bought by pointing the same quota at
fewer, wider fleets.

> **The caveat that keeps this from being a firm recommendation.** `T(W) = 36.3 + 7896/W`
> is fitted over roughly 30–60 workers. The 70w and 80w rows extrapolate it, and the
> ingest estimate explicitly warns against extrapolating far outside that band. The 45 × 60w
> row is inside the measured range and already worth 0.9 days over 45 × 50w for 2,520 vCPU.
> **Treat 60w as the safe recommendation and 80w as the target to validate**, with the
> densest zone at two widths as the measurement that settles it (§10).

Two further consequences of the barrier, both currently unmodelled:

- **The Ray fleet is rebuilt 72 times, not 8.** `fill_zones_sequential_flow` takes a `year`
  parameter, so a cluster is per (cluster, year): 8 clusters × 9 years. At roughly 15 minutes
  of `ray up` plus model load per boot, that is **~$7,000–$10,000 of GPU ramp and ~18 hours
  of schedule** not counted anywhere below. The plan's description of clusters that "pay
  `ray up` once for the whole set" is true only *within* a year.
- **Nine barriers are nine chances to stall.** A single pathological zone delays its whole
  year, and the retry loop runs up to `max_zone_attempts` rounds before the year can close.

> **One measured hint that narrowing may be slightly cheaper than width-neutral.** The
> ingest ramp's 10 → 20 cell rung narrowed S2 from 60w to 45w and cost 1.24× in time
> against the 1.33× that width alone predicts — implying worker-hours fell ~7%. That
> comparison was made for a different purpose (ruling out interference) across different
> zone sets, so it is a hint rather than a curve. If it holds at 50w it is worth roughly
> $6,000, and it would make the optimal scenario cheaper as well as faster.

**The duration basis and the per-zone density fit agree, and this is worth recording because
the apparent disagreement between them wasted real effort.** The 6,354 cell-hour basis is a
campaign aggregate; the ingest estimate also carries a per-zone fit,
`s/date = 10.16 + 0.06022 × live_4096_chunks` (five regions, R² 0.954). Evaluating that fit
over the 111 real per-zone tile counts from the coverage census and summing gives **5.95
h/zone-year at 60 workers, against the basis's 6.36 — a 6.5% agreement.** They were never in
conflict, and neither was the "~10 h versus ~21 h" spread seen elsewhere in these documents:
that is one dense zone at 120 workers and at 50, via the same width model.

Two properties of the fit matter for anything that reasons about *individual* zones rather
than the aggregate, and the aggregate basis hides both:

- **A fixed floor of about 1.0 h per zone-year** (the 10.16 s/date intercept × 365 dates).
  An all-but-empty zone costs an hour, not minutes.
- **Per-zone residuals of ±35%.** Area does not determine duration tightly — 35N and 47N
  differ by 3 chunks in 2,418 and by 27% in per-date time. Any per-zone schedule built on
  this fit needs that much slack.

---

## 5. Inference — the cost is fixed; the waste is not

GPU-hours follow from pixels and throughput alone:

```
  GPU-hours  =  1.363 × 10¹³ px  ÷  (px/s/worker × 3600)
```

All costs on-demand at $1.861/GPU-hour, v1.1. The basis is derived in §6.

| basis | GPU-hours | **cost** |
|---|---|---|
| 13K px/s — Iowa's low end, no geographic gain | 291,100 | **$541,800** |
| **15K px/s — planning basis** | **252,300** | **$469,600** |
| 18K px/s — Iowa's high end with the full gain | 210,300 | **$391,300** |

**None of this varies with the ingest configuration.** The same pixels are inferred at the
same rate either way. What the ingest configuration decides is whether the fleet has work.

### Size the fleet UNDER what ingest can feed

A zone-year costs **252.6 GPU-hours** at the planning basis. The *matched* fleet — the size
at which the fleet exactly consumes what ingest produces — is `supply × 252.6`. Running at
the matched size is the wrong target for two reasons: it leaves no absorber when supply dips,
and any dip below it is billed as idle.

**The policy is to provision at about 85% of matched.** That keeps a standing queue of
finished mosaics, so the fleet is never idle and idle burn is structurally **zero** rather
than merely small; and the 15% margin absorbs an ingest cell going down and restarting
without the GPUs noticing. The cost is that inference trails ingest by roughly 18% of the
run — the "slightly slower start" that buys the guarantee.

| ingest config | Fargate vCPU | ingest | supply | matched | **provision (85%)** | actors/cluster | inference | **campaign** |
|---|---|---|---|---|---|---|---|---|
| 40 × 50w — shipped | 12,640 | 7.2 d | 5.76/h | 1,455 | 1,237 | 155 | 8.5 d | ~8.7 d |
| 45 × 50w — the knee | 14,220 | 6.5 d | 6.41/h | 1,619 | 1,376 | 172 | 7.6 d | ~7.8 d |
| **45 × 60w — recommended** | **16,740** | **5.6 d** | **7.42/h** | **1,873** | **1,592** | **199** | **6.6 d** | **~6.8 d** |
| 45 × 70w | 19,260 | 5.0 d | 8.35/h | 2,109 | 1,793 | 224 | 5.9 d | ~6.0 d |
| 45 × 80w — target, width unvalidated | 22,140 | 4.5 d | 9.22/h | 2,329 | 1,980 | 247 | 5.3 d | ~5.5 d |

**Idle burn is $0 in every row**, by construction. The number that used to sit here — up to
$304,000 of idle at a quota-sized fleet — is what this policy exists to avoid, and it is now
avoided by choosing the fleet rather than by hoping supply keeps up.

**The 2,500-actor quota is not the constraint and should not be provisioned against.** The
largest fleet any of these configurations can keep busy is 1,980, and the recommended one
wants 1,592. Requesting quota beyond about 2,000 actors buys nothing that ingest can feed.

**Do not mix throughput bases.** Sizing the fleet on a while-processing rate (21–24K) and
then running at the campaign rate leaves the fleet short by a third: GPU-hours are unchanged
so the bill does not move, but inference stops trailing ingest and starts setting the
campaign's duration.

> The GPU fleet already boots only when a finished mosaic is waiting, and the feeder takes
> whichever mosaic lands first, so the pipeline does not idle *within* a cluster. What the
> table above measures is different and coarser: whether the campaign as a whole generates
> mosaics fast enough to justify the fleet size you provisioned.

---

## 6. Throughput

**Cost the campaign at ~15K px/s.** This section has been wrong twice — first at 21K, then
at 14K — because both were arguments about which of someone else's measured rates to borrow.
It is now derived instead, from what actually drives inference cost.

### What sets the rate: observations per pixel, not pixels

The encoder consumes a **sequence per pixel**, so cost scales with total tokens, and
`tokens = pixels × observations per pixel`. Pixels-per-second is therefore not a property of
the pipeline at all — it is a property of the pipeline *and the geography*, and it falls as
observation count rises. That is why the profiling doc reports different rates for
"mid-density" and "dense" chunks: density there means observations, not land.

Observations per pixel per year, by band:

- **Optical.** Sentinel-2 A+B revisit every 5 days at the equator (73 acquisitions a year),
  rising with latitude as adjacent swaths overlap — approximately `73 / cos(lat)`. Only the
  cloud-free ones yield observations, and that fraction is strongly latitudinal: about 0.25
  in the humid tropics, 0.55 over the subtropical deserts, 0.40–0.45 at mid-latitudes.
- **Radar.** Sentinel-1 gave a 6-day repeat per orbit direction until S1B failed in December
  2021, and 12-day after. Over the campaign's nine years that averages **50.7 passes per
  direction per year**, doubled where both orbits are ingested, and scaled by the same
  `1/cos(lat)` convergence.
- Both are then quantised by the sampler's buckets (`num_obs_checkpoints`, 8…256).

Weighting by land area per 10° band (Antarctica excluded — our tile census is 151 M km²,
which is all land plus a coastal over-count, so area weighting is sound):

| band | land Mkm² | S2 acq/yr | S2 obs | S1 obs | **tokens/px** |
|---|---|---|---|---|---|
| +65 to +85 | 16.0 | 173–209 | 60–63 | 204–246 | 272–312 |
| +45 to +65 | 30.5 | 103–127 | 46–51 | 122–150 | 176–208 |
| +25 to +45 | 30.0 | 81–89 | 44–49 | 95–105 | 144–168 |
| **+5 to +25 (tropics)** | **23.5** | **73–76** | **18–26** | **86–89** | **112–128** |
| −5 to −25 | 29.0 | 73–81 | 18–44 | 86–95 | 112–144 |
| −25 to −55 | 5.3 | 89–127 | 45–51 | 105–150 | 160–208 |
| **campaign, weighted** | | | | | **167** |
| Iowa — the ROI every measured rate comes from | | 98 | 44 | 137 | **192** |

**The campaign is cheaper per pixel than the ROI our rates were measured on, not dearer.**
The ratio is **0.87**, and across 36 combinations of the three guessed inputs — dual-orbit
fraction, cloud fractions, radar cadence — it ranges **0.78 to 1.00 and never exceeds 1**.
Iowa sits at the expensive end: mid-latitude, dual-orbit, moderately cloudy. A third of the
world's land is tropical, where heavy cloud cuts optical observations to a quarter of the
acquisitions.

So the campaign should run at **Iowa's fleet-overall rate × 1.0 to 1.28**:

| | rate | GPU-hours | cost |
|---|---|---|---|
| pessimistic — Iowa's low end, no geographic gain | 13K | 291,100 | $541,800 |
| **planning basis** | **15K** | **252,300** | **$469,600** |
| optimistic — Iowa's high end × the median ratio | 18K | 210,300 | $391,300 |

**This retires the argument that was in this section**, which claimed the campaign was
"dense-weighted" and should therefore be costed *below* the fleet-overall figure. That
conflated two different meanings of dense: a zone with many live tiles (area) and a chunk
with many observations (sequence length). They are unrelated, and on the axis that actually
drives cost the campaign is lighter than the reference ROI.

**Three inputs here are estimates, and one dominates.** The dual-orbit fraction (assumed
0.70) moves the ratio from 0.78 to 0.97 across its plausible range — far more than cloud or
cadence. It is also the easiest to pin down: `resolve_s1_orbit` already records which orbits
each zone-year actually ingested. **And the whole model is replaceable by measurement**: the
store writes `s2_obs_count` and `s1_*_obs_count` for every pixel, so the first completed
zone-years give the real distribution rather than this reconstruction of it.

> **One consequence beyond cost.** `_partition_by_live_tiles` balances clusters on live-tile
> counts — that is area, not work. If tokens per pixel vary by a factor of two with latitude,
> two clusters with equal tile counts can carry substantially unequal work. The zone-to-cluster
> split should be weighted by `tiles × tokens-per-pixel` once per-zone observation counts are
> known.

### v2 Large: evaluated, not used

**The campaign runs v1.1.** v2 Large was costed and is not being taken forward. The
evaluation is kept here so it is not repeated, and because one of its conclusions is a
reasoning error worth remembering.

`feature/v2-large-model` carries a per-model planning rate in `inference/actors.py` —
`{"v1.1": 16_000.0, "v2-large": 22_000.0}`, a ratio of **1.375×** — which would have been
worth roughly **$137,000** at the capacity-planning basis and would have cut the matched
fleet by about a quarter. Two things made it a weak number to spend against: the constants
are labelled a *strategy-only estimator* ("never a correctness value"), and the commit that
added them points at a calibration in `context_docs` that does not exist on that branch.

**The reasoning error, which generalises.** An earlier version of this document derived the
ratio from the two `ModelArch` definitions — v2 Large is 0.89× the per-token arithmetic of
v1.1 — and then argued the real gain would be *smaller* still, because inference is not
tensor-bound. That was wrong twice over: wrong by a factor of 1.22 against the branch's own
figure, and wrong in its logic. Not being tensor-bound does not mean a smaller model gains
less; it means arithmetic is not what it gains on. A narrower model also moves less weight
and activation traffic, launches smaller kernels, and does less host-side work per token —
precisely the things the profiling run named as the bottleneck. **Do not use a FLOP ratio to
predict throughput on a pipeline measured at 0.12–0.26 tensor-pipe utilisation.**

---

## 7. The lines that are not compute

**Assembly — about $200, negligible.** Assembly runs on the fill flow's own runner rather
than a worker fleet. Eight runners at 4 vCPU across a five-day campaign is roughly 3,900
vCPU-hours. It is a rounding error and does not need a scenario.

**S3 requests — about $1,600, essentially all of it ingest.** The ingest estimate counts
~316M chunk writes at $5/M. Inference adds staged-tile writes and shard writes (~6.5M
PUTs, $33) and mosaic reads (~316M GETs at $0.40/M, $126).

**Transient mosaic storage — about $3,000, flat across every scenario.** More cells hold
more data for proportionally less time. This figure depends entirely on mosaics being
deleted as inference consumes them; if inference lags, it grows linearly with the backlog.

**Permanent embeddings storage — sized, but not billed to us.**

```
  1.363 × 10¹³ px  ×  128 dims  ×  1 byte (int8)   =  1.74 PB
  + observation counts (3 × uint16 per pixel)      =  0.08 PB
                                                      ────────
  uncompressed                                        1.83 PB
  at zstd ~1.4×                                       1.28 PB
  at zstd ~2×                                         0.91 PB
```

This goes to **AWS Open Data, which sponsors the storage**, so it is not in the totals.
The figure is still worth holding: it sets bucket and quota planning, it is what a
mirror or an egress-heavy consumer would cost, and it is the number to quote when the
Open Data application asks how large the dataset is.

---

## 8. Scenario summary

Ingest at its cost midpoint; inference at the **15K basis** (§6), v1.1, with the fleet
provisioned at 85% of matched so idle burn is zero (§5).

| | 40 × 50w<br>shipped | 45 × 50w<br>the knee | 45 × 60w<br>**recommended** | 45 × 80w<br>target, unvalidated |
|---|---|---|---|---|
| Fargate vCPU | 12,640 | 14,220 | **16,740** | 22,140 |
| GPU fleet to provision | 1,237 | 1,376 | **1,592** | 1,980 |
| — actors per cluster (÷8) | 155 | 172 | **199** | 247 |
| Ingest | $121,000 | $121,000 | $121,000 | $121,000 |
| Inference | $469,600 | $469,600 | $469,600 | $469,600 |
| Assembly + S3 + mosaics | $4,800 | $4,800 | $4,800 | $4,800 |
| Cluster ramp (72 boots) | ~$7,000 | ~$8,000 | ~$9,000 | ~$11,000 |
| **Total** | **$602,000** | **$603,000** | **$604,000** | **$606,000** |
| Ingest wall clock (9 yr) | 7.2 d | 6.5 d | **5.6 d** | 4.5 d |
| **Campaign wall clock** | **~8.7 d** | ~7.8 d | **~6.8 d** | ~5.5 d |
| Idle burn | $0 | $0 | **$0** | $0 |

**Every column costs the same to within 0.7%.** Inference is the same pixels at the same
rate in all four; ingest worker-hours are width-neutral; and the fleet policy removes idle
burn everywhere. The entire decision is wall clock, bought with Fargate quota.

**The recommendation is 45 cells at 60 workers.** It is the fastest configuration whose
fleet width sits inside the range the width model was fitted over, it needs *less* Fargate
quota than the 71-cell plan an earlier version recommended (16,740 against 22,436) while
finishing nearly two days sooner, and it asks for about 1,600 GPU actors rather than the
2,500-actor quota. Going to 80 workers buys another 1.3 days and is the thing worth
measuring.

At the pessimistic 13K basis add **$72,000** to every column; at the optimistic 18K subtract
**$78,000**.

**One consequence for the deadline.** All years must be validated by **2026-09-11**. Every
column is 5.5 to 8.7 days of campaign wall clock, so compute is not what threatens that
date. The schedule risk lives in the preflight gates (§10), the Fargate quota lead time, and
the nine year-barriers at which one stalled zone holds up everything behind it.

---

## 9. Assumptions and uncertainties, largest first

1. **The observation-count model rests on three estimated inputs, and the dual-orbit
   fraction dominates.** §6 derives the campaign rate as Iowa's × 1.0–1.28. Across 36
   combinations the ratio never exceeds 1.0, so the *direction* is safe — but the spread is
   worth **$150,000**, and 0.70 for dual-orbit coverage is the input carrying most of it.
   `resolve_s1_orbit` already records which orbits each zone-year ingested, so this can be
   replaced with a count rather than an assumption before the campaign starts.
2. **The wall-clock floor rests on one fitted curve extrapolated past its range.** The
   45 × 80w configuration — the only one that saturates the GPU quota — uses `T(W)` at 80
   workers, against a fit made over roughly 30–60. If the curve flattens sooner than the fit
   implies, 80 workers buys less than 12.0 h and the matched fleet stays under 2,500. The
   floor itself is robust (the three densest zones are within 1% of each other, so it is a
   plateau rather than an outlier), but its *value at width* is not. One dense zone at two
   widths settles it.
3. **Nothing prices observation count.** Cost per pixel scales with observations per pixel,
   so dual-orbit regions cost roughly twice single-orbit ones and optical revisit varies with
   latitude — plausibly a factor of two in each direction around the global mean. The model
   applies one rate to every pixel. This is the mechanism underneath uncertainty 1 rather
   than a separate risk, but it is the tractable way to resolve it: the store writes
   `s2_obs_count` and `s1_*_obs_count` per pixel, so the first completed zone-years measure
   it directly. It also means `_partition_by_live_tiles` balances clusters on **area, not
   work**, which is only sound if observation count is uncorrelated with zone — and latitude
   says it is not.
4. **Fleet-matching assumes ingest and inference stay in lockstep, and they do not.** The
   duty-cycle arithmetic treats supply as smooth. It is not: dense zones take far longer than
   sparse ones, and the campaign deals the densest first, so early supply is slower than
   average and late supply faster. **This is now modelled rather than hand-waved** —
   `tests/unit/test_gpu_starvation.py` runs the real per-zone tile counts through one
   cluster's ingest look-ahead and matched actor pool. A fleet that boots on its first mosaic
   idles about **4.8 GPU-hours per cluster-year** — **108,400 idle GPU-hours, about
   $202,000**, across the campaign. Holding the boot until the queue contains **3.25
   work-hours** of pixels removes it entirely, for **25.5 hours** of added schedule over nine
   years. It appears in all 96 combinations the model scans, so it is not
   scenario-dependent. **It is a recommendation, not shipped behaviour.**

   The scan was run against the earlier 71-cell configuration. The knee correction (§4) does
   not change its shape — the same zones arrive in the same order, and only the interval
   between them shortens — but the specific hour figures should be re-run at 45 × 60w before
   they are quoted as final.

   The 3.25 figure is a threshold, not a preference — 3.0 starves in 2 of 96 combinations and
   3.25 in none. It is denominated in **work-hours for the fleet that will consume them**,
   which is what makes it robust to fleet size: a smaller fleet takes proportionally longer
   on the same queue, so the same work-hours means the same protection. A mosaic count or a
   raw pixel threshold would each need re-deriving whenever the fleet changed.

   **Note this is now partly redundant with the fleet policy in §5.** Provisioning at 85% of
   matched already guarantees a standing queue in steady state; the work-hours bank is what
   protects the *start* of each year, when the queue is empty by construction. Both are
   wanted, and nine year-barriers mean the start case happens nine times.
5. **Ingest carries the ±10% per-date fit** from its own estimate, plus the untested
   assumption that cell interference stays flat above 20 concurrent cells. Per-zone
   durations additionally carry **±35%** (§4), which the aggregate basis hides.
6. **Pixel count.** The 2048-tile census gives 1.363 × 10¹³; the ingest estimate's
   4096-chunk census implies 1.459 × 10¹³, about 7% higher. The larger figure would add
   ~$23,000 to inference.

---

## 10. What to do about it

1. **Set `max_parallel_ingest` to 45 and `max_workers` to 60.** That is 16,740 Fargate vCPU
   — a smaller quota request than the 71-cell plan an earlier version of this document
   recommended, and nearly two days faster. Above 45 cells the year barrier makes extra cells
   worthless, and below 60 workers the longest zone gets no shorter. Both halves of that
   sentence are the same finding (§4).

2. **Provision about 1,600 GPU actors — 199 per cluster — not the 2,500-actor quota.** Sizing
   at 85% of what ingest can feed makes idle burn structurally zero and leaves 15% headroom
   for an ingest cell to fail and restart without starving the fleet (§5). Requesting quota
   beyond ~2,000 actors buys nothing this campaign can use.

3. **Count the dual-orbit zones before finalising the budget.** It is the single largest
   input to the throughput model — worth about $150,000 across its plausible range — and it
   is not an estimate that needs measuring, just a count `resolve_s1_orbit` can produce from
   the coverage already ingested (§6).

4. **Measure the densest zone at two fleet widths.** It settles whether 80 workers really
   delivers the 12.0 h that would take the campaign to 4.5 days, or whether the width curve
   flattens first — the only remaining question about the schedule, and worth 1.3 days.

5. **Weight the zone-to-cluster split by work, not area.** `_partition_by_live_tiles`
   balances on live-tile counts while cost scales with tiles × observations, and observation
   count varies about twofold with latitude. Two clusters with equal tile counts can carry
   materially unequal work (§6).

6. **Replace the whole throughput model with measurement after the first zone-years land.**
   The store writes `s2_obs_count` and `s1_*_obs_count` per pixel. That converts §6 from a
   reconstruction into arithmetic, and it costs nothing but reading what is already written.

---

## 11. Where the underlying detail lives

- [`ingest_optimization_campaign_2026_07.md`](ingest_optimization_campaign_2026_07.md) —
  the authoritative record of every ingest measurement this rests on: what each change
  bought, what failed, and the constraints future work must respect. The July-27 cost
  estimate is a working note derived from it.
- [`inference_gpu_saturation_profile_2026_07.md`](inference_gpu_saturation_profile_2026_07.md)
  — where the throughput figures and the tensor-utilisation reading come from.
- [`campaign-cluster-sizing.md`](campaign-cluster-sizing.md) — the coverage census, the
  cluster partitioning, and the wall-clock arithmetic for the GPU side.
- `config/inference.py` on `feature/v2-large-model` — the two `ModelArch` definitions §6
  compares.
