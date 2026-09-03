# Global TESSERA campaign — full cost model

**Dated 2026-07-29.** Extends the July-27 ingest cost estimate — which costed **ingest
only** — to the whole campaign: ingest, inference, assembly, and the permanent output
store. Nine years × **112 land zones**. The ingest measurements it rests on are recorded in
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
| **Inference (GPU, on-demand)** | **$472,000 – $713,000**, plan on **$573,000** — re-based 2026-08-07 on one token unit (§6b), re-measured the same day on a COMPLETED dense cell, then re-weighted onto campaign land (§6c). Was $452,000 – $573,000 / $527,000 |
| Assembly | ~$1,300 — measured (§6c) |
| S3 requests | ~$1,600 — almost all of it ingest (§7) |
| Mosaic storage (transient) | ~$3,000 |
| Ray cluster ramp | ~$1,200 — 10 boots, one per cluster. A year-serial campaign would pay one per cluster-year instead, ~$11,000 (§4) |
| **Campaign total** | **$594,000 – $846,000**, plan on **$700,000** — the sum of the lines above. The ingest line is under review upward (§4); the inference line is measured on one token unit and one completed cell (§6b, §6c) |

**The campaign is a GO and is inside budget; this document's job is now accuracy, not the
decision.** Every revision since 2026-08-07 has moved the inference line by single-digit
percentages, and none of them approaches the margin the decision was taken with. Read the
sections below as cost *control* — where the money goes and which term to watch — rather than as
a gate. The one figure that could still change the answer is the ingest line, which is under
review **upward** by up to 3x (§4).

**Costed in tokens, with both sides of the division measured in ONE unit.** The campaign is
**2.36 × 10¹⁵ combined S2+S1 tokens** — 1.363 × 10¹³ pixels at a measured, land-weighted
**173 tokens per pixel** — run at a measured **2.127 M combined tok/sec** per worker, which is
**307,854 GPU-hours** (§6b, §6c). The pair this replaces — 1.98 × 10¹⁵ tokens at ≈1.9 M
tok/sec — divided an S2+S1 census by an **optical-only** rate; both terms were wrong by a
similar factor in opposite directions, so the line itself moved little and the
capacity-planning rate is **12,294 px/s** equivalent against the prior 13,103, a change of
−6.2%. Pixels-per-second is a derived figure from here on: it mixes machine speed with
geography, which is why this document rewrote its throughput basis three times before
switching units (§6).

**The near-cancellation survived a 12× increase in evidence.** The correction's whole point
was that two errors pointed opposite ways, so the px/s the fleet is sized on barely moved.
That was +2.1% when it rested on a partial sample and is −6.2% now. It has crossed zero
without ever becoming large, which is a stronger statement than the original.

**The central division is now measured on both sides** (2026-08-07): the combined-token rate
from **2,596 both-orbit chunks** across four cells, and the optical half of the depth from
22,343 chunks covering every populated latitude band (§6b, §6c). What replaced the "largest
unvalidated input" is narrower, and the binding uncertainty has **changed identity**: the
rate is now pinned by one completed cell carrying 77% of the pooled weight, while the radar
depth's *level* rests on ten band-observations spanning 66–152 tokens per pixel and is now the
widest driver of the interval (§6c). No both-orbit rate has been measured at the planned 228
actors per cluster; the widest measured is 160. Ingest velocity was also measured and disagrees
with the basis by 2.7–4.0× (§4) — resolved as season, not width and not contention (§4).

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
| **Campaign tokens** | **2.36 × 10¹⁵ combined** (1.363e13 px × 173 tok/px, measured and land-weighted; was 2.28 × 10¹⁵ at 167 when radar was sample-weighted, and 1.98 × 10¹⁵ at the censused 145) | §6c |
| **Inference rate** | **2.127 M combined tok/sec** per worker (12,294 px/s equivalent; was 2.273 M on a partial cell, and ≈1.9M *optical* tok/sec — a different unit from the census, the mismatch §6b resolves) | measured, §6c |
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
  year, and the retry loop runs up to `max_dispatch_rounds` rounds before the year can close.

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
on all three, 35N at 60w costs **196.3 s/date against 167.9, i.e. 1.17x**.

**And none of that 1.17x is contention at the width tested.** The paired A/B completed the same day:
a quiet arm on an idle account came out **3.3% DEARER** per window than the loaded arm (11.40 against
11.04), with both verified at full achieved width. So there is no load penalty to price into the
ingest line at 17 concurrent fleets. An earlier version of this paragraph ended "contention at
campaign width is unmeasured, and Stage 3's 25/40/55 ladder is what would price it" — that is
now closed:

**Contention at campaign width: MEASURED, AND ABSENT (55-cell rung on prod, 2026-08-06).** 55
concurrent cells — 20,316 vCPU, i.e. actual campaign scale, not a fraction of it — paired
against the same zone and month at ~3,200 vCPU came out at **10.4 s/window against 13.1**. The
claim this supports is **no contention penalty at 55 cells**, NOT a 21% improvement: the two
arms sit in different accounts and per-window cost is not perfectly stable, so the sign of the
residual is not interpretable — only the absence of a penalty is. Everything around the cells
held too: the orchestrator sat at **25% CPU with zero dropped events** (36 cells once ran it to
100%, before the server resize); ECS placement was exact (5,086 tasks against ~5,115 implied);
the achieved fleet widths — 60 optical, 13 per radar orbit — held throughout; and `no-worker=0`
on all 40 schedulers, every pass. **Nothing of ours throttled**: every 503 in the window was
upstream (sentinel-cogs, ASF), which is the ceiling that belongs to Phase 5's F7, not to this
line.

### TOKENS AND COST DECOUPLE — the inference line is far less sensitive to the census than it looks

The single most useful thing measured on the campaign path (2026-08-05). Observations per pixel
and cost per chunk do **not** move together, because a token-poor chunk still pays the fixed
per-chunk overhead:

| | span across measured zones | ratio |
|---|---|---:|
| observations per pixel | 60 → 179 | **3.0x** |
| **cost per chunk** | $0.116 → $0.238 | **2.1x** |

Projected over every live tile, a token census 37% lower than assumed moves the inference line
only **10%** — 1.24 x 10¹⁵ tokens against 1.98, but **$482 k against the planned $538 k**, still
inside the costed range.

**Two consequences for how this document should be used.**

**Do not re-price inference off a revised token census.** The elasticity is roughly 0.3: a third
off the tokens buys a tenth off the cost. Any saving argued from the census alone is overstated by
about 3x. (The interval this paragraph used to cite — observations per pixel 75–139, central
106 — is superseded: land-weighted optical depth is now measured at **103.1** over 22,343
chunks covering every populated band, and the planning depth is the *combined* 170; see §6b.)

> **The 0.3 elasticity does NOT apply to the 2026-08-07 basis revision, and assuming it did
> would understate that change threefold.** The elasticity describes varying tokens at a fixed
> all-in per-chunk cost, where fixed overhead dilutes the effect. In §6b the rate is re-measured
> on the same chunks as the depth, so the two move together and the elasticity is 1.0 by
> construction — a 17% depth rise is a 17% cost rise before the rate correction offsets it. The
> fixed per-chunk term is small in the both-orbit population anyway: inference is 90.3% of
> both-orbit chunk wall-clock.

**The lever that would actually move the line is per-chunk overhead, not tokens.** At the sparse
end a chunk's cost is dominated by fixed work — model load, read setup, write — rather than by
inference over its observations. Reducing that overhead pays on every chunk, and pays most on the
majority of land, which is token-poor. Reducing tokens pays proportionally to tokens, which is
where the cost is not.

Stratified figures, the projection interval, and what remains unmeasured:
`campaign_inference_profile_2026_08.md`.

**What this means for the durations above: raise them, but for seasonality.** The fit is built on
January-conditions dates and a zone-year is not twelve Januaries — on one zone at one width,
summer dates cost **1.68x** January dates, carried by 18.0 windows/date against 15.0 plus dearer
windows (write per window 16.7 s against 11.0 s). A seasonally weighted year therefore lands
materially above the January-rate basis; a midpoint of the two measured anchors suggests
**~1.5-1.7x**, which is an estimate rather than a measurement — pin it with per-date covered
chunks or one completed full-year cell before committing a schedule to it. **The practical
difference from the withdrawn claim is large: seasonality is predictable and schedulable, so peak
months can be planned around instead of hunted as a defect.**

Full derivation and withdrawal: `ingest_concurrency_investigation_2026_08.md` §"The withdrawn
claim, and the mechanism behind it".

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

> **SUPERSEDED 2026-08-07.** The three rows below borrow Iowa's px/s anchor and an optical
> rate. The measured combined basis, completed the same day (§6c), gives **297,321 GPU-hours,
> $553,000 central**, with one-at-a-time sensitivity spanning **$456,000 – $713,000**. The rows
> are retained because the scenario tables in this section and §8 were built on the 13.1K row,
> and the capacity-planning rate moved only **−6.2%** (13,103 → 12,294 px/s) — so the fleet
> sizing, the 85% policy and the work-hours bank below all stand unchanged; only the dollar
> line moved. That the rate barely moved is the finding, not a coincidence: two errors of
> similar size pointed opposite ways (§6b).

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

| cells | Fargate vCPU | ingest | fleet vs supply | **campaign** |
|---|---|---|---|---|
| 52 | 19,344 | 4.80 d | ~96% — no buffer | ~5.1 d |
| **61** | **22,692** | **4.18 d** | **81%** | **~5.1 d** |
| 66 | 24,552 | 3.86 d | 75% | ~5.1 d |
| 80 | 29,760 | 3.18 d | 62% | ~5.1 d |

**Past ~52 cells the campaign is flat at ~5.1 days**, because the 2,500-actor fleet consumes at
a fixed rate no matter how fast mosaics arrive: `307,854 GPU-hours ÷ 2,500 = 123 h`.

> **Re-based on §6c (was ~4.8 d).** The rows above divided the OLD 283,200 GPU-hours by 2,500
> actors. The land-weighted census raised the work to 307,854 GPU-hours — +8.7% — and the campaign
> with it, from ~4.8 to ~5.1 days. Fleet SIZING is untouched, because that is set by ingest supply
> rather than by total work; it is the DURATION that moves, and the duration is what the deadline
> is measured against. The ingest column is likewise re-derived by scaling the measured 45-cell
> row, which is why 4.09 became 4.18. Extra cells buy the **buffer** the 85% policy is
made of — which is worth having, since it is what absorbs a failed ingest cell — but they buy no
schedule at all. This is the easiest wrong quota request to make from this document: asking for
Fargate when the binding resource is GPU.

**To buy schedule, buy actors.** Cells shown are what keeps 85% provisioning:

| actors | clusters at ≤275 each | **campaign** | vs 2,500 |
|---|---|---|---|
| 2,500 | 10 × 250 | ~5.1 d | 1.00× |
| 2,750 | 11 × 250 | **~4.7 d** | 0.91× |
| 3,000 | 12 × 250 | **~4.3 d** | 0.83× |
| 3,500 | 14 × 250 | ~3.7 d | 0.71× |

The cheapest useful ask is **2,704 actors**, which is 66 cells at proper 85% provisioning — an
~8% quota bump for about half a day.

> **Status of the barrier removal: CLEARED (P4 2026-08-06, radar caveat closed by P7), and
> `overlap_years=true` is the campaign setting.** An earlier version of this note said "cost
> and schedule planning should still use the year-serial rows above" while nothing had run on
> a real fleet; P4 passed all five multi-year checks and P7 ran two clusters across six
> both-orbit cells with a year rollover inside one of them. The two figures it moves are
> moved: the campaign floor is ~5.1 d at 61 cells (the barrier-free tables above), and the
> cluster-ramp line is ~$1,200 (10 boots), not ~$11,000 (90).

**The campaign uses the entire 2,500-actor quota**, as 10 clusters of 250 — the quota is what sets
the wall clock, not a margin above it. The year-serial rows above provision less (1,824 at 45 × 60w,
2,267 at 45 × 80w) because their ingest supply is smaller.

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

**Cost the campaign in tokens — 2.36 × 10¹⁵ combined S2+S1 tokens at 2.127 M combined tok/sec
per worker (measured, §6c).** This headline read "1.98 × 10¹⁵ at ≈1.9M tok/sec" until
2026-08-07; that pair divided a combined census by an optical rate, and §6b re-bases both sides
onto one unit. §6c then re-measured the rate on a completed cell the same day. Every earlier
version of this section argued about which region's pixels-per-second to borrow. That argument
only existed because the unit was wrong.

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
bearing input in the model and it is no longer a one-ROI figure. (**Read the unit, 2026-08-07:**
this 1.9 M is the *optical* rate — `t_kept × valid_px ÷ seconds` — and it remains correct as
one. The rate the model now divides by is the **combined 2.127 M** of §6c, which counts the
radar sequences too; §6b put it at 2.273 M before the 37N cell was completed.)

**And the units argument is settled empirically.** Over the same twelve actors, tokens per
second is flat to ±1% while pixels per second varies 2.2×. A model denominated in pixels
would have been wrong by up to that factor depending on which site it borrowed from; this one
is not. The section above argued that from first principles — this is the measurement.

### Observations per pixel are measurable directly

The store writes `s2_obs_count` and `s1_*_obs_count` per pixel, so depth is a direct read rather
than an inference from the catalogue — which is what §6b and §6c do, and why the census is now used
for composition only.

The caveat that matters: **a chunk-array depth counts a date if ANY pixel in the chunk kept it**, so
it is an upper bound on the per-pixel figure the model processes. That overstatement cancels only
when numerator and denominator are both chunk-array (§6b). Reading these arrays and pairing them
with a census rate would reintroduce exactly the mismatch that put the line 19% wrong.

### The token census — usable for composition, not for the split

The census counts distinct acquisition dates times clear fraction, per latitude band, from the
catalogue. **Measured against 2026-08-07, its TOTALS are usable and its SPLIT is not**: at 60N the
optical/radar split is close to inverted (censused 48/156, measured 145/65), and at 47S the total is
31% low.

So it survives in the model for one job — the *composition* of a pixel-year across dual-orbit,
single-orbit and radar-free populations — and the depth figures come from measurement (§6b, §6c). The
per-band derivation that stood here is in git history; keeping it invited exactly the pairing §6b
warns against, since a reader who found the censused numbers first would divide them by a measured
rate.

### 6b. The division re-based on one token unit — method, superseded numbers

Derived from CloudWatch `CHUNK_SUMMARY` over a 96-hour window (29,886 successful chunks, 31 streams,
13 zones, 1,633 carrying radar telemetry), through the truncation-bisecting profiler.

**Its numbers are superseded by §6c**, which measured the completed cell §6b asked for. The
progression is in one table so no reader can land on a retired figure without seeing what replaced it:

| basis | rate per actor | depth | px/s | GPU-hours | cost |
|---|---:|---:|---:|---:|---:|
| prior model | 1.90 M tok/s (optical) | 145 tok/px (censused, combined) | 13,103 | 288,940 | $537,700 |
| re-based, partial cell | 2.273 M tok/s (combined) | 170 tok/px (measured) | 13,371 | 283,200 | $527,000 |
| 37N complete, sample-weighted | 2.127 M | 167 | 12,736 | 297,321 | $553,000 |
| **current — radar land-weighted (§6c)** | **2.127 M tok/s** | **173 tok/px** | **12,294** | **307,854** | **$573,000** |

The convention-free statement, carrying no definitional risk: **74.8 microseconds of busy GPU time
per valid pixel, land-weighted.** Because the capacity-planning rate moves only −6.2% across the
whole progression, fleet sizing, the 85%-provisioning policy and the work-hours bank are unaffected
throughout; only the dollar line and the token census move.

**What survives §6b as method, and must not be re-derived:**

**The combined unit is right, empirically.** Within each cell the combined rate is the same for
both-orbit and one-orbit chunks to within 1.01–1.08× while the optical rate over the same chunks
spreads 1.24–1.35×. A within-run comparison holds geography, day, code and fleet constant, so this is
the shape of evidence that settled tokens-versus-pixels. **Do not fit separate per-token prices for
optical and radar** — the two-coefficient regression is not identifiable here: the ratio comes out
0.26× to 1.42× across cells, changing sign, and the fit's R² of 0.79 looks respectable and is not.

**Radar depth is NOT a function of latitude and must not be modelled as one.** Across five bands it
runs 66, 147, 93, 70, 91 — a 2.2× spread with Pearson r = **+0.009** against band midpoint, where
optical depth over the same bands gives **+0.912**. The variation is regional: the deepest radar
measured anywhere is 30–32°N and the shallowest the Arctic, which points at the Sentinel-1
observation plan rather than geometry. Radar's *share* of a sequence does fall with latitude, but
that is the optical denominator rising. The defensible form is a constant with a stated spread —
**90 tokens/px, range 66–147** — no curve, no extrapolation.

**Three conventions of "observations per pixel", and only one pairing is self-consistent.** The
census counts distinct acquisition dates × clear fraction; the chunk-array depth counts a date if
*any* pixel in the chunk kept it; the model processes each pixel's own count rounded up to a multiple
of 8. Chunk-array depth is an upper bound on the third, so `depth × valid_px` overstates true tokens
— and that overstatement **cancels exactly when numerator and denominator are both chunk-array**,
which is measured-depth ÷ measured-rate and nothing else. That is why the revised depth is 167 rather
than 145: most of the difference is convention, not disagreement. Pairing the census's 145 with the
measured combined rate would be a new mismatch of the same kind, one level down.

### 6c. 37N/2021 completed — the measurement §6b asked for, taken 2026-08-07

§6b named completing 37N/2021 as the highest-value next measurement, because that one cell
"separates $483,000 from $796,000". It was completed the same day. **The interval and the
central line in §6b are superseded by this section; §6b's method is not.**

The cell was filled in two legs against one preserved staging prefix, so the whole zone-year
is measured rather than extrapolated:

| leg | chunks | GPU-hours | cost | median s/chunk | optical depth |
|---|---|---|---|---|---|
| first (20 actors, 24.3 h) | 3,855 | 480.4 | $894 | 408.1 | 121 |
| second (160 actors, 3.3 h) | 4,859 | 421.7 | $785 | 271.2 | 71 |
| **whole zone-year** | **8,714** | **902.1** | **$1,679** | — | mean ≥ 99 |

**The two legs agree to 5.8% on tokens per second despite a 1.50× difference in seconds per
chunk.** That is the single best validation the token basis has: the per-chunk gap is depth,
not machine speed, and a per-chunk or per-pixel cost basis would have read it as a 50%
throughput discrepancy between two halves of one zone in one year.

**Two things the partial sample had wrong**, both now corrected in
`tests/unit/inference/test_gpu_starvation.py`:

* Its span was recorded as 30–32°. The completed cell spans **0.1–32.1°** — the run swept
  north to south, so the opening chunks were its northern extreme. A whole-cell median over a
  partial sweep is biased, not merely noisy; this is the third time that mechanism has produced
  a wrong number in this campaign.
* Its both-orbit radar depth read **146.8** tokens per pixel, "the deepest radar measured
  anywhere". Completed, it is **89.9** — in line with 60N's 76.4 and 47S's 66.3. The deepest
  radar *does* exist, but it belongs to the 32.5° band and not to the cell.

**What moved, and in which direction.** Both were measured in the same pass, and they push
opposite ways, so neither may be quoted without the other:

| | superseded | completed | effect on the line |
|---|---|---|---|
| combined rate | 2.273 M tok/s | **2.127 M tok/s** | +6.9% cost |
| campaign depth | 170 tok/px | **167 tok/px** | −1.7% cost |
| **central line** | $527,000 | **$553,000** | **+5.0%** |

**The evidence base grew 3.4×** — 762 both-orbit chunks over 5 latitude bands became 2,596
over 10 — and the pooled rate is now dominated by this one cell, which carries **77% of the
weight**. That is a property of the table and is why the unexplained deficit below is
load-bearing rather than a footnote: it is inherited by the central figure, not diluted.

**The interval's widest driver has changed identity.** One-at-a-time sensitivity, holding the
other term at its central value (never max-radar × worst-rate, which double-counts one
observation):

| driver | range | line |
|---|---|---|
| radar depth, band extremes | 66.1 – 151.8 tok/px | **$504,000 – $713,000** |
| combined rate, cell extremes | 2.038 – 2.579 M tok/s | **$472,000 – $598,000** |

While 37N was a partial sample its rate deficit set the width. Completing it *pinned* the rate
and *widened* the radar spread, so radar depth is now the wider term. **The residual uncertainty
is in radar's LEVEL, not in the rate.**

#### The weighting fix that followed, and why it is worth more than the missing band

Completing 37N left 35–50° unmeasured, and an earlier draft of this section called that gap
un-interpolable because radar depth is regional rather than latitudinal. **That was too strong,
and interpolating it exposed a real defect.** Radar depth is not a *function* of latitude, but
filling one gap between two measured neighbours is not evaluating a trend — and doing it made the
two halves of the campaign depth comparable for the first time.

**Optical depth was always land-weighted; radar depth was weighted by the chunks we happened to
measure.** Those are different questions, and the sample answered the wrong one:

| radar depth basis | value | what it answers |
|---|---|---|
| chunk-weighted over measured cells | 86.3 tok/px | how deep is radar *in our sample* |
| **live-tile-weighted over all populated bands** | **94.3 tok/px** | how deep is radar *over campaign land* |

The 9.2% gap is geography, not statistics. The deepest radar measured anywhere sits at 30–35°,
and the unmeasured 35–50° band beside it holds **19.2% of campaign land** — so a sample-weighted
figure systematically under-represents the latitudes where radar is deepest. Interpolating that
band (137, 123, 108 across its three sub-bands) and holding the polar bands above 70° flat (3.6%
of land, where flat is the only defensible form) gives the land-weighted figure, which is now the
model's.

**Effect: depth 167 → 173, line $553,000 → $573,000, +3.5%.** Interpolation narrows no
uncertainty — the 66–152 spread is untouched — it only stops the weighting from being wrong.

**Measuring that band is DECIDED AGAINST, 2026-08-08** — too expensive for what it buys, and the
decision is recorded here so it is not revived as an open question. A measured 35–50° cell would
test whether the land-weighted 94.3 is right, and it remains the widest single driver of the
interval. What makes it declinable is that interpolation is already the defensible treatment: the
band sits between two measured neighbours, and its absence is what would bias the weighting, not
its interpolation. **The campaign proceeds on the interpolated figure.**

The residual is therefore carried, not closed: if the true depth in that band sits at the top of
the measured 66–152 spread, the line is understated, and 19.2% of campaign land is the exposure.
That is inside the interval already published in §1.

**What completing it did NOT settle.** The rate deficit narrowed from ~30% to **~19%** and did
not close. Worse, the cell is now both the widest fleet measured (160 actors) and the slowest
rate, so **fleet width and geography are perfectly confounded** in the cell table. This looks
like an actor-count penalty and must not be read as one. Separating them needs a second wide
run in a different zone.

**Assembly, measured for the first time — to completion.** A 16 vCPU / 64 GiB Fargate task
reads **4.97 TB across 2.34 M objects** for a dense zone-year — staged tiles are stored
**uncompressed**, at 570.4 MB each — at $0.93/hour plus about $1 of S3 requests. Scaling by tile
count over 9 years of 112 zones gives **~$1,150**, superseding the ~$200 in §1. Utilisation was
57–79% of CPU, ~1.0 GB/s of combined network and 52% of memory, so nothing was saturated.

| | measured |
|---|---|
| shard-write phase | **195.9 min** (3.27 h) |
| merge + commit | **37 s — 0.32% of assembly** |
| total | **196.6 min** (3.28 h) |
| effective read rate | **423 MB/s** |

**Two corrections and one free result.** An in-flight estimate of 2.2–3.1 hours, extrapolated
from a ~550 MB/s instantaneous network sample, was optimistic: sustained over the run the
effective rate is **423 MB/s**, so a spot network reading is not a throughput basis. And **the
commit is negligible** — 37 seconds of 196 minutes, because the forked workers have already
written every chunk and the merge is metadata only. That is the phase split `ASSEMBLY_SUMMARY`
was added to obtain, available from two log timestamps: for planning, **assembly IS the shard
write**, with no second phase to model.

**The trailing-thread margin is thinner than a 2.5-hour assembly implied.** The runner sits at
0.02 vCPU throughout inference, so the box is free — but capacity was never the question. The
question is whether assembly finishes before the next cell's inference does:

| | assembly | inference at 250 actors | margin |
|---|---|---|---|
| dense cell (8,714 tiles) | 3.28 h | 3.61 h | **1.10×** |
| average cell (3,222 tiles) | 1.21 h | 1.33 h | **1.10×** |

**1.10× at the campaign's 250 actors per cluster**, and scale-invariant because both terms are
linear in tiles. (At 228 it was 1.21×; the two cross at **275**, which is what caps actors per
cluster and therefore why the fleet is 10 clusters rather than 8.) Assembly stays off the critical path, but by 21% rather than comfortably, and a
cluster runs ~126 cells in sequence so a persistent deficit compounds rather than averaging out.
That is what makes the **39% of CPU left idle** worth keeping: not spare performance to harvest,
but the margin's only buffer. **F8 now tests a thin ratio rather than confirming a comfortable
one.**

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

**Assembly — about $1,300, measured (§6c).** Assembly runs on the fill flow's own runner rather
than a worker fleet, so it is a runner-hours line rather than a fleet one. Small enough not to need
a scenario, but not the ~$200 a vCPU-hour estimate suggested before it was measured.

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

This goes to **AWS Open Data, which sponsors the storage**, so it is not in the totals —
written to `s3://tessera-embeddings/v1.1/dclimate.icechunk`, an existing Open Data bucket we
publish into rather than one we apply for. The figure is still worth holding: it is what a
mirror or an egress-heavy consumer would cost, and it is the number the bucket's owners
will want when they plan for it.

---

## 8. Scenario summary

The campaign runs **all years in one batch**: 61 cells at 60 workers, 10 clusters of 250 actors.
Ingest at its cost midpoint, inference at §6c's measured combined basis, fleet provisioned at 85% of
matched so idle burn is zero (§5).

| | **THE CAMPAIGN** |
|---|---|
| Fargate vCPU | 22,692 |
| GPU fleet | 2,500 — the quota |
| clusters × actors | 10 × 250 |
| Ingest | $121,000 |
| Inference (§6c) | $573,000 |
| Assembly + S3 + mosaics | $5,900 |
| Cluster ramp | ~$1,200 — 10 boots, not 90 |
| **Total** | **~$700,000** |
| Ingest wall clock (9 yr) | 4.2 d |
| **Campaign wall clock** | **~5.1 d** |
| Idle burn | $0 |

**Three year-serial variants stood beside this** (40×50w shipped, 45×50w at the knee, 45×60w) and are
cut: they cost within $2,000 of each other and of the campaign, so the comparison only ever showed
that **ingest width barely moves the bill** — which §4 establishes directly. What the year barrier
*did* cost was the ramp: 8 clusters × 9 years is 72 `ray up` cycles plus 72 model loads, against 10.
Dropping the barrier is worth ~$6,000 and about 3.6 days of wall clock, and that is the whole content
of the comparison.

## 9. Assumptions and uncertainties, largest first

Each entry is what it is NOW; the versions they superseded are in git history, because an
uncertainty list that carries its own retired entries is one nobody reads to the end of.

1. **Radar depth's LEVEL is the single widest driver.** Ten band-observations spanning 66–152,
   land-weighted to 94, worth **$504,000 to $713,000** on its own. It overtook the 37N rate deficit
   when completing that cell narrowed it. Radar depth is *regional*, not latitudinal (§6b), so more
   bands do not narrow it — more *regions* would.

2. **The ingest line is under review upward by up to 3×.** Measured velocity on three virgin zones
   came in 2.7–4.0× slower than the fit behind $121,000 (§4). Three confounds are named there and
   none is dismissed. This is the only figure that could still change the campaign's answer.

3. **No both-orbit rate has been measured at campaign fleet width.** The four cells ran at 20–95
   actors against a planned 250 per cluster. No actor-count trend appeared among them, which is
   reassuring and is not a measurement.

4. **The 37N/30–32° rate deficit is unexplained**, narrowed from ~30% to ~19% by completing the
   cell but not closed.

5. **The optical half is now the best-measured input**, not an uncertainty: 22,343 chunks across
   every populated band. The census survives for *composition* only — its totals are usable and its
   optical/radar split is not (§6b).

6. **The convention cancellation is exact only if the overstatement factor matches** between the
   corpus and the rate cells. Any residual pushes the line *down*.

## 10. What to do about it

1. **Set `max_parallel_ingest` to 45 and `max_workers` to 60.** That is 16,740 Fargate vCPU
   — a smaller quota request than the 71-cell plan an earlier version of this document
   recommended, and nearly two days faster. Above 45 cells the year barrier makes extra cells
   worthless, and below 60 workers the longest zone gets no shorter. Both halves of that
   sentence are the same finding (§4).

   **Amended once the barrier cleared:** with `overlap_years` validated (P4/P7) and prod's
   applied quotas at Fargate 25,000 vCPU and G-and-VT 10,000 vCPU (2,500 actors, both verified
   in the account 2026-08-06), the operating shape is **61 cells at 60 workers** — the 85%-
   provisioning row of §5's barrier-free table, and what `campaign-plan.md` §3 stars. The
   45-cell figure above is the year-serial fallback.

2. **Report the 2022–2024 optical-only cells in whatever ships with the data.**
   `allow_s2_only` is on, so they exist rather than being holes, and radar-free embeddings were
   validated upstream by Cambridge (2026-08-03) — so this is no longer a caveat to defend, just
   6.8% of pixel-years to characterise. They are identifiable from the store
   (`s1_asc_obs_count + s1_desc_obs_count == 0`), so the work is describing them, not finding
   them (§6).

3. **Provision 2,500 GPU actors as 10 clusters of 250.** Sizing under what ingest can feed
   makes idle burn structurally zero and leaves headroom for an ingest cell to fail and restart
   without starving the fleet (§5). The campaign sits at 81% of supply, which the 2,500-actor
   quota chose rather than the policy — 85% would want 2,611. The split into ten clusters rather
   than eight is the assembly ceiling: a cluster assembles on one trailing thread, and above about
   275 actors per cluster assembly becomes the critical path (`campaign-plan.md` §6). Do not go
   wider on ingest, because at 80 workers the matched fleet passes the quota and inference becomes
   the critical path.

4. **Read `s2_obs_count` and `s1_*_obs_count` off the first completed zone-years.** Both
   halves of the throughput model are now counted from public catalogues, but from a sparse
   sample and through a cloud proxy that omits shadow. The store measures the real thing per
   pixel, and reading it costs nothing (§9). **37N/2021 is the first cell that can answer this**
   — completed 2026-08-07 (§6c) with all five observation-count variables written.

4b. **Fill any zone-year between 35° and 50° of latitude under the radar telemetry — as a cell of
   opportunity, not a scheduled measurement.** That band is unmeasured and holds 19% of campaign
   land, and radar depth is the widest driver of the cost interval (§6c). It is currently
   **interpolated**, which is defensible: interpolating between two measured neighbours is not
   the latitude curve the band table forbids. So this refines the level rather than filling a
   hole. Prefer a zone outside Europe and the Middle East, since those are the densely-tasked
   Sentinel-1 regions and every deep observation we hold comes from there. Such a run would also
   disentangle fleet width from geography, which are perfectly confounded in the current cell
   table — but neither question is worth delaying the campaign for.

5. ~~**Weight the zone-to-cluster split by work, not area.**~~ **DONE 2026-07-30.** At the
   campaign's 10 clusters, true-work spread is **0.009%** where balancing on tile counts alone
   would give **21.8%** (re-measured 2026-08-12; see §9 item 3 for what did and did not
   reproduce). Worth recording why it mattered more
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
