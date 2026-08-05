# Global TESSERA campaign — full cost model

**Dated 2026-07-29.** Extends the July-27 ingest cost estimate — which costed **ingest
only** — to the whole campaign: ingest, inference, assembly, and the permanent output
store. Nine years × 111 land zones. The ingest measurements it rests on are recorded in
[`ingest_optimization_campaign_2026_07.md`](ingest_optimization_campaign_2026_07.md);
section references below of the form "ingest estimate §N" point at that working note.

**This is the figures annex to [`campaign-plan.md`](campaign-plan.md).** That document owns
operations — what runs and with what settings; this one owns every number in it and the
arithmetic behind it. Every input here is either measured (and cited) or derived from a
measured input (and marked as derived); the ones that are neither are called out in §9
rather than buried.

---

## 1. The headline

| | |
|---|---|
| Ingest (Fargate) | $115,000 – $126,000 — **under review, see §4: measured velocity is 2.7–4.0× slower than the basis, which would treble this** |
| **Inference (GPU, on-demand)** | **$503,000 – $579,000**, plan on **$538,000** |
| Assembly | ~$200 |
| Mosaic storage (transient) | ~$3,000 |
| Ray cluster ramp (72 boots, §4) | ~$9,000 |
| **Campaign total** | **$638,000 – $712,000**, plan on **$672,000** — the ingest line is under review upward (§4); the inference line stands |

**Costed in tokens.** The campaign is **1.98 × 10¹⁵ tokens** — 1.363 × 10¹³ pixels at a
land-weighted 145 observations per pixel, both halves censused from public catalogues — run
at a reference **≈1.9M tok/sec** per worker. Pixels-per-second is a derived figure from here
on: it mixes machine speed with geography, which is why this document rewrote its throughput
basis three times before switching units (§6). The range carried through is the reference
ROI's own 13–15K px/s band.

**One input is now measured rather than modelled** (2026-08-04): the inference rate of ≈1.9M
tok/sec is confirmed at three geographies to within 1% (§6). Ingest velocity was also measured
and disagrees with the basis by 2.7–4.0× (§4), which is under investigation. The
tokens-per-pixel census remains the largest unvalidated headline input; §6 records how to
measure it directly and why P2 could not.

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

**Five findings worth acting on.**

1. **Inference, not ingest, is where the money is** — three to four times the ingest bill.
   Every optimisation effort so far has gone into the cheaper half.
2. **Inference cost is invariant to the ingest scenario.** Same pixels, same throughput,
   same GPU-hours whether you run 40 cells or 71. What the ingest scenario changes is
   whether the GPU fleet has anything to do.
3. **A fifth of the land has no radar for three of the nine years**, because OPERA coverage
   was withdrawn after Sentinel-1B failed in December 2021 and restored with Sentinel-1C in
   2025. **`allow_s2_only` is ON for this campaign**, so those pixels still get embeddings
   from a neutral radar input rather than being dropped — which is what turns a coverage hole
   into a quality caveat. It also makes them *cheaper*: a minimal radar sequence instead of a
   full one, worth about $15,000–$18,000 once the per-chunk read floor is allowed for (§6).
4. **Fleet sizing is the largest cost lever.** Provision UNDER what ingest can feed, which
   makes idle burn structurally zero and leaves headroom for ingest restarts (§5). The
   2,500-actor quota fits every configuration here, but only just at the widest — it is
   adequate rather than generous.
5. **Ingest wall clock has a floor that more cells cannot break, and it binds at about 45
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
| **Campaign tokens** | **1.98 × 10¹⁵** (1.363e13 px × 145 tok/px) | token census, §6 |
| **Inference rate** | **≈1.9M tok/sec** per worker (13.1K px/s equivalent) | measured, §6 |
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

### MEASURED 2026-08-04, and it is 2.7–4.0× SLOWER than the fit predicts

Three virgin zones ingested from scratch at the campaign's own 60-worker fleet, rate taken over
a 47-minute steady-state window so cluster startup is excluded. Calendar-days of optical
imagery committed per wall-clock hour, projected to a full year:

| zone | live 4096-chunks | fit s/date | fit h/zone-year | **measured h** | ratio |
|---|---:|---:|---:|---:|---:|
| 53N | 890 | 63.8 | 6.5 | **26.1** | **4.0×** |
| 12N | 1,551 | 103.6 | 10.5 | **28.5** | **2.7×** |
| 37N | 2,309 | 149.2 | 15.1 | **47.4** | **3.1×** |

Mean measured **34.0 h/zone-year** against the aggregate basis's **6.36**. **If this holds, the
ingest line is not $115,000–$126,000 but roughly three times that** — and ingest stops being
the cheap half of the campaign. It does not change the inference line, which §6 now confirms.

**Do not repoint the model on this yet.** Three confounds, in the order they could matter:

1. **Concurrency.** These ran while ~40 cells were live in one account, sharing S3, the source
   catalogue and one Prefect control plane. The fit was derived on far fewer. Per-cell slowdown
   under fleet-wide load is exactly what an aggregate basis would miss, and it is the most
   likely single explanation.
2. ~~**Width may not be usable.**~~ **INVESTIGATED, LARGELY WITHDRAWN.** The "six times the
   fleet for under twice the rate" reading compared *different zones*, and the two 10-worker
   zones happened to be the two cheapest-per-chunk zones in the wave. **Same-zone width pairs
   show 6x workers buying 3.7-4.9x** (23N 3.65x, 24N 3.77x, 25N 4.92x), against an Amdahl
   prediction of 4.4-4.6x from the 7-8% serial fraction measured at 10w. Width works about as
   well as it can.

   Two real but second-order effects were found. A **per-date serial floor** — build, unhidden
   preparation stall, gate residual — is **19-24% of every date at 60 workers**, and no width
   touches it. And `adapt(minimum=1)` costs **~10-12% of effective width**: one 60-slot fleet
   registered 1,250 distinct workers in five hours, retiring them in every inter-date gap and
   relaunching them cold into the next write. `min_workers = max_workers` recovers that for free,
   and the p2 fleets — which set a real minimum — do not churn at all.

   **Width is also nearly cost-NEUTRAL**: vCPU-seconds per date at 60w versus 10w averages ~1.1x
   over six same-zone pairs. Sixty workers costs about what ten does per date and finishes 4-5x
   sooner, so re-scaling `max_workers` is not where the money is.
3. **The fit's own basis** is five regions at R² 0.954, on code that has since changed.

**Investigation and plan: `ingest_concurrency_investigation_2026_08.md`.** Six candidate causes
are ruled out, including the orchestrator, the Dask schedulers, capacity, and store growth. Two
findings bear directly on this section. Per-date cost rises through a run because **windows per
date rise** — on 53N, write cost per window held flat at 16.4–18.3 s over 137 dates while
windows/date rose 45% — which means **every velocity figure measured early in a run understates
the full year**, so the durations above are optimistic on that count alone. (Write per window is
the better normaliser but is **not** a constant: on 35N it falls 18.71 → 14.15 s June to August
with windows/date pinned at 18.0.) The same arithmetic across seasons is what dissolves the
apparent July gap — see the next-but-one paragraph.

**The per-tile cost gap is intrinsic, not width waste.** 53N costs ~$0.14/tile-year against
12N's ~$0.09, and the investigation attributes that to 53N doing **1.38x the per-tile work**
(9.7 vs 7.0 worker-seconds per chunk-date, fewer tiles amortising the same per-date floor) — not
to an oversized fleet. 53N at 60 workers is still ~80% write-bound, so it *can* use the fleet;
narrowing sparse zones would recover single-digit percent of campaign ingest compute.

**The factor is SEASON, not width and not a regression** (corrected 2026-08-05; this paragraph
previously asserted an unexplained 1.8-2.1x slowdown, and that claim is **withdrawn**). The
comparison behind it put 35N's **May-September 2021** dates against the July record's **January
2024** baseline: 330.7 s/date against 167.9. Same zone, same width, different season — and the
July record's own reading instructions name dates as the third condition that must match. Matched
on all three, 35N at 60w costs **196.3 s/date against 167.9, i.e. 1.17x**, and that 196.3 was
measured under 17 concurrent fleets, so it bounds contention and drift together rather than
isolating either.

**What this means for the durations above: raise them, but for seasonality.** The fit is built on
January-conditions dates and a zone-year is not twelve Januaries — on one zone at one width,
summer dates cost **1.68x** January dates, carried by 18.0 windows/date against 15.0 plus dearer
windows (write per window 16.7 s against 11.0 s). A seasonally weighted year therefore lands
materially above the January-rate basis; a midpoint of the two measured anchors suggests
**~1.5-1.7x**, which is an estimate rather than a measurement — pin it with per-date covered
chunks or one completed full-year cell before committing a schedule to it. **The practical
difference from the withdrawn claim is large: seasonality is predictable and schedulable, so peak
months can be planned around instead of hunted as a defect.**

Full derivation and withdrawal: `ingest_concurrency_investigation_2026_08.md` §"E IS WITHDRAWN".

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
| 14.0K px/s — from Iowa's 15K | 269,600 | **$501,700** |
| **13.1K px/s — planning basis, from Iowa's 14K** | **288,900** | **$537,700** |
| 12.2K px/s — from Iowa's 13K | 311,100 | **$578,900** |

**None of this varies with the ingest configuration.** The same pixels are inferred at the
same rate either way. What the ingest configuration decides is whether the fleet has work.

### Size the fleet UNDER what ingest can feed

A zone-year costs **289.2 GPU-hours** at the planning basis. The *matched* fleet — the size
at which the fleet exactly consumes what ingest produces — is `supply × 289.2`. Running at
the matched size is the wrong target for two reasons: it leaves no absorber when supply dips,
and any dip below it is billed as idle.

**The policy is to provision at about 85% of matched.** That keeps a standing queue of
finished mosaics, so the fleet is never idle and idle burn is structurally **zero** rather
than merely small; and the 15% margin absorbs an ingest cell going down and restarting
without the GPUs noticing. The cost is that inference trails ingest by roughly 18% of the
run — the "slightly slower start" that buys the guarantee.

| ingest config | Fargate vCPU | ingest | supply | matched | **provision (85%)** | actors/cluster | inference | **campaign** |
|---|---|---|---|---|---|---|---|---|
| 40 × 50w — shipped | 12,640 | 7.2 d | 5.76/h | 1,666 | 1,416 | 177 | 8.5 d | ~8.7 d |
| 45 × 50w — the knee | 14,220 | 6.5 d | 6.41/h | 1,854 | 1,576 | 197 | 7.6 d | ~7.8 d |
| **45 × 60w — recommended** | **16,740** | **5.6 d** | **7.42/h** | **2,146** | **1,824** | **228** | **6.6 d** | **~6.8 d** |
| 45 × 80w — if the width holds | 22,140 | 4.5 d | 9.22/h | 2,667 | 2,267 | 283 | 5.3 d | ~5.5 d |

**Idle burn is $0 in every row**, by construction. The number that used to sit here — up to
$304,000 of idle at a quota-sized fleet — is what this policy exists to avoid, and it is now
avoided by choosing the fleet rather than by hoping supply keeps up.

### Which quota actually binds, and the answer flips when the year barrier goes

Every row above is **year-serial**, so its makespan is `max(longest zone, work / cells)` per
year and the cell count stops helping at ~45. Without the barrier the makespan is simply
`total work / cells`, which moves the constraint onto the GPU fleet:

| cells | Fargate vCPU | ingest | provisioning at 2,500 actors | **campaign** |
|---|---|---|---|---|
| 52 | 19,344 | 4.80 d | 100% — no buffer | ~4.8 d |
| **61** | **22,692** | **4.09 d** | **85% — the policy** | **~4.8 d** |
| 66 | 24,552 | 3.78 d | 79% | ~4.8 d |
| 80 | 29,760 | 3.12 d | 65% | ~4.8 d |

**Past ~52 cells the campaign is flat at ~4.8 days**, because the 2,500-actor fleet consumes at
a fixed rate no matter how fast mosaics arrive. Extra cells buy the **buffer** the 85% policy is
made of — which is worth having, since it is what absorbs a failed ingest cell — but they buy no
schedule at all. This is the easiest wrong quota request to make from this document: asking for
Fargate when the binding resource is GPU.

**To buy schedule, buy actors.** Cells shown are what keeps 85% provisioning:

| actors | cells | Fargate vCPU | **campaign** | vs 2,500 |
|---|---|---|---|---|
| 2,500 | 61 | 22,700 | ~4.8 d | 1.00× |
| 2,750 | 67 | 24,969 | **~4.4 d** | 0.91× |
| 3,000 | 73 | 27,239 | **~4.0 d** | 0.83× |
| 3,500 | 85 | 31,779 | ~3.4 d | 0.71× |

The cheapest useful ask is **2,704 actors**, which is 66 cells at proper 85% provisioning — an
~8% quota bump for about half a day.

> **Status of the barrier removal (2026-07-30): the code is shipped, the validation is not.**
> `overlap_years` exists and defaults OFF; the three pieces it needed — a per-cell inference
> window, a child flow taking `(zone, year)` pairs, and a driver that batches instead of
> looping years — are in and unit-tested. Nothing has run on a real fleet, so **cost and
> schedule planning should still use the year-serial rows above** until Phase 4's **P4** rung
> clears it. Two figures move when it does: the campaign floor from ~6.8 d to ~4.8 d at 61
> cells, and the cluster-ramp line from ~$9,000 to ~$1,000.

**The 2,500-actor quota fits every configuration here, but not by much at the widest.** The
recommended 45 × 60w provisions 1,824; even 45 × 80w provisions 2,267, inside the quota with
margin. The quota is adequate for every configuration considered here.

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

**Cost the campaign in tokens: 1.98 × 10¹⁵ of them, at ≈1.9M tok/sec per worker.** Every
earlier version of this section argued about which region's pixels-per-second to borrow. That
argument only existed because the unit was wrong.

### Cost is denominated in TOKENS, not pixels

The encoder consumes a **sequence per pixel**, so its cost scales with
`tokens = pixels × (T_s2 + T_s1)`. Pixels-per-second is therefore not a property of the
pipeline: it is a property of the pipeline *and the geography it ran over*, and it falls as
observation count rises — a sparsely-observed pixel costs roughly **10× less** than a densely
observed one.

```
   GPU-hours  =  total tokens  ÷  tok/sec
                 ^ geography       ^ machine
                 censused below    measured once, logged already
```

**The campaign is 1.98 × 10¹⁵ tokens** — 1.363 × 10¹³ pixels at a land-weighted **145 tokens
per pixel**. At the reference worker rate of **≈1.9M tok/sec** that is 289,000 GPU-hours.

### The rate is now measured at three geographies, not one — and it held

**P2 ran 2026-08-04**: three single-ROI inference runs spanning the token range (boreal
NWT/Yukon, Iowa, Amazon). Across twelve `g6e.xlarge` actors:

| | measured |
|---|---|
| tok/sec per actor | **1.90 – 1.93 M**, every maximum within 1% of 1.95 M |
| effective TFLOPS | **85**, flat across all twelve |
| px/sec per actor | **12,420 – 27,285** — a **2.2× spread** |

**The reference rate is confirmed** — the ≈1.9M this document has costed on since it switched
units is what three geographies deliver, to within about 1%. That is the single most load-
bearing input in the model and it is no longer a one-ROI figure.

**And the units argument is settled empirically.** Over the same twelve actors, tokens per
second is flat to ±1% while pixels per second varies 2.2×. A model denominated in pixels
would have been wrong by up to that factor depending on which site it borrowed from; this one
is not. The section above argued that from first principles — this is the measurement.

### Observations per pixel are measurable directly — but P2 cannot yet judge the census

The runs report the quantity. Each `CHUNK_SUMMARY` carries **`t_kept`**, the timesteps the
pipeline actually kept, and `t_kept × valid_px` reproduces the measured token rate to within
9% — so it *is* the observations-per-pixel term in `tokens = pixels × (T_s2 + T_s1)`. That is
the useful discovery here: the term is measured per chunk on every run, for free, and never
needs censusing again.

Over 200 chunks across all three P2 sites: `t_kept` spans **55–136**, area-weighted mean
**69.9**, against the census's **145**.

**Do NOT read that as the census being 2× high.** Four reasons, and the second is disqualifying:

1. **The effective sample is 3, not 200.** Chunks within one ROI share an acquisition history, a
   cloud climatology and an orbit geometry, so those 200 are heavily correlated and speak to
   three places.
2. **P2 ran a 50× stricter keep threshold than the campaign will.** The single-ROI path takes
   `DEFAULT_MIN_VALID_COVERAGE` from `ingest.roi_processing` = **5.0%**; the campaign path takes
   `IngestSettings.min_valid_coverage` = **0.1%**. A stricter gate drops more dates, so **P2's
   `t_kept` is biased LOW relative to the campaign — in precisely the direction of the apparent
   discrepancy.** On its own this disqualifies the comparison.
3. **Wrong constellation year.** P2 used a window ending December 2024. Sentinel-1B failed in
   December 2021 and 1C restored coverage during 2025, so a 2024 window carries thinner radar
   than the campaign's target years. It matters most where radar dominates: the census gives the
   boreal site 137 S1 observations against 48 S2.
4. **Three sites are not land-weighted.** They were chosen to bracket the token range, not to
   represent campaign land by area.

**What settles what.** P3 fills seven zones through the *campaign* path, so its `t_kept` removes
reason 2 — the threshold — and its seven zones weaken reason 1. Neither P3 (2021) nor P2 (2024)
addresses reason 3, so **a campaign-year comparison needs a fill in a target year**, and reason 4
needs enough zones to weight by area. Until then **145 stands as the planning figure** and this
section records only that the term is directly measurable, not that the census is wrong.

**Duty cycle, for the rung that needs it.** Twenty chunk summaries: inference 319.6 s mean
(min 237, max 358), overhead 58.1 s, prologue 46.6 s, total 377.7 s — **inference is 84.6% of
chunk wall-clock**. The remaining ~15% is prologue and staged-write overhead.

Raw figures and per-actor breakdown: `context_docs/design/inference-perf-run-ledger.md`.

> **This document quoted px/s for three revisions and rewrote the throughput basis three
> times because of it.** Each rewrite was really an argument about how to convert one
> region's px/s into another's, which is a conversion that only exists because the unit was
> wrong. `inference/README.md` already said so — "px/sec is density-dependent … so the
> summaries also log **tok/sec** … for density-neutral comparison" — and the pipeline has
> been emitting `tok/sec` and effective TFLOPS per sub-batch and per chunk the whole time.
> **Quote tok/sec going forward.** px/s figures below are retained only because the
> historical measurements were recorded that way.

### The token census

Both halves counted on the same global land grid. Radar: a CMR granule census of
`OPERA_L2_RTC-S1_V1` across five campaign years, each count normalised by `cos(lat)` to turn
granules-per-box into observations-per-pixel. Optical: a Sentinel-2 L2A STAC census over the
same points, counting **distinct acquisition dates** (not scenes — overlapping MGRS tiles
would double-count) and averaging `1 − eo:cloud_cover` per date.

| band | land Mkm² | **S2 obs/yr** | **S1 obs/yr** | tokens/px |
|---|---|---|---|---|
| +60 to +80 | 16.0 | 48 | 156 | 208 |
| +40 to +60 | 30.5 | 45 | 121 | 176 |
| +20 to +40 | 30.0 | 59 | 98 | 168 |
| 0 to +20 | 23.5 | 51 | 59 | 120 |
| −20 to 0 | 20.5 | 44 | 49 | 104 |
| −40 to −20 | 12.5 | 72 | 48 | 128 |
| −60 to −40 | 1.3 | 54 | 47 | 104 |
| **campaign, land-weighted** | | **52** | **91** | **152** |
| — after optical-only cells (below) | | | | **145** |
| **Iowa — the ROI every rate comes from** | | 70 | 61 | **136** |

Two things this makes visible that the px/s framing hid:

1. **Iowa is a SINGLE-orbit site** — hundreds of ascending granules and *zero* descending at
   every sample point, in both eras. Every throughput figure we hold comes from a site at the
   cheap end of the radar distribution.
2. **The optical half pushes back.** Iowa's 152 acquisition dates a year at 46% clear give 70
   usable optical observations, *above* the campaign's 52. Mid-latitude revisit is high and
   its cloud is moderate.

Net, the campaign is **1.07× Iowa's tokens per pixel**, so at equal tok/sec it runs at 0.94×
its px/s. That ratio is now bookkeeping rather than an assumption — it exists only to reuse
historical px/s measurements.

| | px/s equivalent | GPU-hours | cost |
|---|---|---|---|
| Iowa 15K → campaign | 14.0K | 269,600 | $501,700 |
| **Iowa 14K → campaign — planning basis** | **13.1K** | **288,900** | **$537,700** |
| Iowa 13K → campaign | 12.2K | 311,100 | $578,900 |

### Throughput is not purely token-bound, and the floor bites where tokens are fewest

There is a per-chunk cost that scales with **pixels and bytes**, not sequence length: roughly
13 s of fixed read amplification from the 4000-px storage chunking, plus the prologue. The
striping work cut the visible part from 50–60 s of GPU-idle per chunk to about **6 s median
on prefetch-hit chunks** — but it did so by *overlapping the read with compute*. Less compute
means less to hide behind, and `_strip_plan` already disables prefetch for chunks that are
wide but sparse.

So the honest form is two-term, `time = fixed_per_chunk + tokens / rate`, combined as a
maximum per strip rather than a sum. At campaign-average token counts the fixed term is
~2% of a chunk; on the optical-only cells (56 tokens/px against Iowa's 136) it rises to
perhaps 8–10%. **That is why the saving credited to `allow_s2_only` below is quoted as
$15,000–$18,000 rather than the $26,000 a pure token model gives.** Measuring the fixed term
is one of the two things the Phase-4 test geographies (the **P2** rung) are chosen to settle.

### The 2022–2024 radar gap, and what `allow_s2_only` does to it

The same census found something that is not really a cost question. **OPERA RTC-S1 coverage
was withdrawn from about a fifth of the land it served after Sentinel-1B failed in December
2021, and largely restored when Sentinel-1C came online in 2025:**

| era | covered land (area-weighted) | dual-orbit fraction |
|---|---|---|
| 2017–2021 — S1A + S1B | 100% (baseline) | 0.57 |
| **2022–2024 — S1A alone** | **81%** | **0.51** |
| 2025 — S1A + S1C | 96% | 0.57 |

Interior Australia and large parts of Siberia return **zero** OPERA granules for 2022, 2023
and 2024 having returned thousands in 2017–2021. The loss concentrates at high latitude (64%
coverage above 60°N) and in the southern mid-latitudes.

**`allow_s2_only` is ON for this campaign**, which is what keeps this from being a hole in
the output. A pixel with no radar is embedded from its optical sequence with a neutral
all-zeros radar input rather than being dropped, so those cells produce data. Across the nine
years, **6.8% of pixel-years are optical-only** on this basis.

It also makes them cheaper, though less than a pure token count suggests. A radar-less pixel
carries the smallest sequence bucket instead of a full one, pulling campaign tokens per pixel
from 152 to **145** — worth $26,000 on tokens alone, but those same cells are the ones where
the per-chunk read floor stops being hidden, which gives perhaps a third of it back. Call it
**$15,000–$18,000**, and treat it as incidental: the reason to enable the flag is coverage.

**Whether those embeddings are equivalent is settled, and not by us.** ADR 013 required a
mask-S1 comparison study before `allow_s2_only` could be used in production; **the Cambridge
team validated radar-free embeddings (2026-08-03)**, so that prerequisite is met upstream and
the flag is cleared. What the campaign still owes is a *description* rather than a defence:
6.8% of pixel-years are radar-free, every one identifiable after the fact because
`s1_asc_obs_count + s1_desc_obs_count == 0` marks it, so the share and its embedding
statistics ship alongside the data. Cambridge's study should be cited here once we have it.

---

## 7. The lines that are not compute

**Assembly — about $200, negligible.** Assembly runs on the fill flow's own runner rather
than a worker fleet. Eight runners at 4 vCPU across a five-day campaign is roughly 3,900
vCPU-hours. It is a rounding error and does not need a scenario.

**S3 requests — about $1,600, essentially all of it ingest.** The ingest estimate counts
~316M chunk writes at $5/M. (The zone-year attribute commit was split out of the shard
commit on 2026-07-30, doubling the campaign's *commit* count from ~1,100 to ~2,200. Those
are metadata commits, not chunk writes, so this figure is unmoved — noted only so the
arithmetic is followable.) Inference adds staged-tile writes and shard writes (~6.5M
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

Ingest at its cost midpoint; inference at the **13.1K measured basis** (§6), v1.1, with the
fleet provisioned at 85% of matched so idle burn is zero (§5).

| | 40 × 50w<br>shipped | 45 × 50w<br>the knee | 45 × 60w<br>**recommended** | 45 × 80w<br>if the width holds |
|---|---|---|---|---|
| Fargate vCPU | 12,640 | 14,220 | **16,740** | 22,140 |
| GPU fleet to provision | 1,416 | 1,576 | **1,824** | 2,267 |
| — actors per cluster (÷8) | 177 | 197 | **228** | 283 |
| Ingest | $121,000 | $121,000 | $121,000 | $121,000 |
| Inference | $537,700 | $537,700 | $537,700 | $537,700 |
| Assembly + S3 + mosaics | $4,800 | $4,800 | $4,800 | $4,800 |
| Cluster ramp (72 boots — see below) | ~$7,000 | ~$8,000 | ~$9,000 | ~$11,000 |
| **Total** | **$670,000** | **$671,000** | **$672,000** | **$674,000** |
| Ingest wall clock (9 yr) | 7.2 d | 6.5 d | **5.6 d** | 4.5 d |
| **Campaign wall clock** | **~8.7 d** | ~7.8 d | **~6.8 d** | ~5.5 d |
| Idle burn | $0 | $0 | **$0** | $0 |

> **The 72-boot ramp line is a cost of the YEAR BARRIER, and it is now optional.** Clusters
> are dispatched per year, so 8 clusters x 9 years is 72 `ray up` cycles plus 72 model-load
> cold starts. With `overlap_years` (shipped 2026-07-30, default off — see
> `campaign-plan.md` §8 item 7) a cluster works a multi-year list, so the campaign pays **8
> boots**, taking this line from about $9,000 to roughly $1,000. That is a small number
> against $672,000 and is NOT the reason to drop the barrier — the schedule is (§"Which
> quota actually binds") — but it is the one line item that moves, so it belongs here rather
> than being discovered later.

**Every column costs the same to within 0.5%.** Inference is the same pixels at the same
rate in all four; ingest worker-hours are width-neutral; and the fleet policy removes idle
burn everywhere. The entire decision is wall clock, bought with Fargate quota.

**The recommendation is 45 cells at 60 workers**, provisioning about 2,100 GPU actors — 264
per cluster. It is the fastest configuration whose fleet width sits inside the range the
width model was fitted over, and the widest whose matched fleet still fits under the
2,500-actor GPU quota. It also needs *less* Fargate quota than the 71-cell plan an earlier
version recommended (16,740 against 22,436) while finishing nearly two days sooner.

**80 workers is viable at the measured rate** — 2,267 provisioned against a 2,500 quota —
and buys another 1.3 days for 5,400 more vCPU. It stays a target rather than a recommendation
only because the width model is extrapolated there (§4).

At the optimistic 14.0K basis subtract **$36,000** from every column; at the pessimistic
12.2K add **$41,000**.

**One consequence for the deadline.** All years must be validated by **2026-09-11**. Every
column is 5.5 to 8.7 days of campaign wall clock, so compute is not what threatens that
date. The schedule risk lives in the preflight gates (§10), the Fargate quota lead time, and
the nine year-barriers at which one stalled zone holds up everything behind it.

---

## 9. Assumptions and uncertainties, largest first

1. **Both halves of the token model are counted, but from a sparse sample.** Five points per
   latitude band, one year for optical and five for radar. The sampling error on any single
   band is real; the land-weighted mean is steadier, and the quantity that matters is a
   *ratio* against a reference measured the same way, so systematic error largely cancels.
   The residual is worth roughly **$40,000** across the 11.6–13.4K band.

   What does *not* cancel is `eo:cloud_cover` omitting cloud shadow and dark pixels, which
   the pipeline also rejects. That inflates both sides' observation counts by perhaps 20–30%
   and could shift the ratio if shadow fraction varies systematically with latitude — it
   plausibly does, since shadow scales with solar zenith angle. **The store's own
   `s2_obs_count` retires this entirely** once the first zone-years land.
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
   it directly. It also meant `_partition_by_live_tiles` balanced clusters on **area, not
   work**, which is only sound if observation count is uncorrelated with zone — and latitude
   says it is not. **Fixed 2026-07-30:** the partition now weights each live tile row by its
   latitude band's observation count (`zone_work_weight`), which takes true-work spread across
   8 clusters from **9.43% to 0.04%**. The band table is this section's census, so the fix
   inherits its sampling error — but balancing needs only the RATIOS between bands, which is
   the robust part.
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

2. **Report the 2022–2024 optical-only cells in whatever ships with the data.**
   `allow_s2_only` is on, so they exist rather than being holes, and radar-free embeddings were
   validated upstream by Cambridge (2026-08-03) — so this is no longer a caveat to defend, just
   6.8% of pixel-years to characterise. They are identifiable from the store
   (`s1_asc_obs_count + s1_desc_obs_count == 0`), so the work is describing them, not finding
   them (§6).

3. **Provision about 2,100 GPU actors — 264 per cluster.** Sizing at 85% of what ingest can
   feed makes idle burn structurally zero and leaves 15% headroom for an ingest cell to fail
   and restart without starving the fleet (§5). That fits inside the 2,500-actor quota; do
   not go wider on ingest, because at 80 workers the matched fleet passes the quota and
   inference becomes the critical path.

4. **Read `s2_obs_count` and `s1_*_obs_count` off the first completed zone-years.** Both
   halves of the throughput model are now counted from public catalogues, but from a sparse
   sample and through a cloud proxy that omits shadow. The store measures the real thing per
   pixel, and reading it costs nothing (§9).

5. ~~**Weight the zone-to-cluster split by work, not area.**~~ **DONE 2026-07-30.** True-work
   spread across 8 clusters falls from 9.43% to 0.04%. Worth recording why it mattered more
   than "uneven finish times" suggested: clusters are long-lived so the last to finish sets the
   campaign date and a heavy cluster is never averaged away, and the imbalance was **not
   random** — latitude drives it, so a cluster drawing high-latitude zones was heavy in *every*
   year.

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
- `scripts/census_s1_coverage.py` — the radar census behind §6. Re-run it to refresh the
  observation counts, or to check whether OPERA coverage has expanded again. Unauthenticated;
  a few minutes per year queried.
