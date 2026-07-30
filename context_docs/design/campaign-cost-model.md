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
| **Inference (GPU, on-demand) — v1.1** | **$294,000 – $503,000**, plan on **$503,000** |
| **Inference (GPU, on-demand) — v2 Large** | **$214,000 – $366,000**, plan on **$366,000** |
| Assembly | ~$200 |
| Mosaic storage (transient) | ~$3,000 |
| **Campaign total, v1.1** | **$412,000 – $632,000**, plan on **$629,000** |
| **Campaign total, v2 Large** | **$332,000 – $495,000**, plan on **$492,000** |

The "plan on" column is the **14K px/s capacity-planning basis** (§6). The low end of each
range is the best rate ever observed on one ROI and should not be budgeted against.

The permanent embeddings store (0.9–1.8 PB) is **not costed here**: it goes to AWS Open
Data, which sponsors the storage. Sizing it still matters for bucket planning — see §7 —
but it is not a line on this bill.

**GPUs are on-demand. Spot is not costed here and is not an option** — sustaining 1,400 to
2,500 g6e instances for days makes interruption a certainty rather than a risk, and a campaign
that stalls on capacity is worse than one that costs more. Settled; do not re-open.

**Four findings worth acting on.**

1. **Inference, not ingest, is where the money is** — two-and-a-half to four times the
   ingest bill. Every optimisation effort so far has gone into the cheaper half.
2. **Inference cost is invariant to the ingest scenario.** Same pixels, same throughput,
   same GPU-hours whether you run 40 cells or 71. What the ingest scenario changes is
   whether the GPU fleet has anything to do.
3. **Fleet sizing is the largest lever, worth up to $407,000** — the cost of provisioning to
   the GPU quota rather than to what ingest can actually feed (§5). **But it is now
   scenario-dependent:** at the corrected throughput basis and 71 ingest cells, v1.1's
   matched fleet *exceeds* the quota, so there is nothing to oversize. The lever is real at
   40 cells, and on v2 in every scenario. **Running v2 Large is the second, at
   $137,000** (§6). Both are within our control.
4. **Two throughput rates in this document are not interchangeable, and mixing them costs
   schedule rather than money.** Size the fleet on 21K and run at 14K and inference stretches
   from 4.5 days to 6.7 for the same bill. §6 says which to use and why.

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
| Land zones | 111 | ingest estimate §4 |
| Campaign years | 9 → **999 zone-years** | settled |
| Live 2048-px tiles | 360,953 per year | coverage census, `campaign-cluster-sizing.md` |
| **Pixels inferred** | **1.363 × 10¹³** | 360,953 × 2048² × 9 |
| S2 duration basis | 6,354 cell-hours at 60 workers | ingest estimate §5 |
| S2 worker-hours | 381,240 | 6,354 × 60 |
| S1 worker-hours | 39,600 – 71,100 | ingest estimate §6 |
| Fargate | $0.04048/vCPU-h, $0.004445/GB-h | ingest estimate §6 |
| Worker | 4 vCPU, 16 GiB → **$0.2330/worker-hour** | derived |
| g6e.xlarge | $1.861/h **on-demand** (spot excluded by decision) | `docs/providers/aws.md` |
| Inference throughput | **14K** capacity-planning; 15K / 21K / 24K carried as bounds | see §6 |
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

| scenario | S1/orbit | vCPU/cell | total vCPU | wall clock | supply rate | ingest cost |
|---|---|---|---|---|---|---|
| **shipped today — 40 × 50w** | 11 | 316 | 12,640 | **7.9 days** | 5.24 zone-yr/h | $115,400 – $126,000 |
| as asked — 40 × 60w | 13 | 372 | 14,880 | 6.6 days | 6.29 zone-yr/h | $113,900 – $124,000 |
| **optimal — 71 × 50w** | 11 | 316 | **22,436** | **4.5 days** | 9.30 zone-yr/h | $115,400 – $126,000 |
| quota-max — 80 × 50w | 11 | 316 | 25,280 | 4.0 days | 10.48 zone-yr/h | $115,400 – $126,000 |

A 25,000 vCPU quota fits **79 cells at 50w** but only **67 at 60w** — which is the whole
argument for the narrower fleet, since worker-hours (and therefore cost) do not care.

**The 71 × 50w scenario is very slightly more expensive than the 60w rows, not cheaper.** Worker-hours are
the billing unit and they are width-neutral — halve the fleet, double the duration, same
bill — so narrowing S2 from 60 to 50 workers buys nothing on compute. What it does is free
40 vCPU per cell, which is what lets 71 cells fit under a 25,000 vCPU quota instead of 67.
The small increase (~$1,700) is the scheduler-and-runner overhead, which is charged per
*fleet-hour* and so rises when a narrower fleet runs longer.

**Buying wall clock, not cost.** Against the current 40 × 60w setup, the optimal
configuration buys **2.1 days** of ingest wall clock for about $1,700. That is a good
trade on its own terms, and a much better one once §5 is taken into account.

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

Both models, at all three throughput bases. v2 Large's column is the v1.1 rate × 1.375
(§6); all costs are on-demand at $1.861/GPU-hour.

| basis (v1.1 rate) | v1.1 GPU-h | **v1.1 cost** | v2 rate | v2 GPU-h | **v2 cost** |
|---|---|---|---|---|---|
| 15K px/s — fleet-overall, upper end | 252,400 | **$469,700** | 20.6K | 183,600 | **$341,600** |
| **14K px/s — capacity-planning basis** | **270,400** | **$503,300** | **19.3K** | **196,700** | **$366,000** |
| 21K px/s — mid-density, while processing | 180,300 | **$335,500** | 28.9K | 131,100 | **$244,000** |
| 24K px/s — best observed | 157,800 | **$293,600** | 33.0K | 114,700 | **$213,500** |

**None of this varies with the ingest scenario.** The same pixels are inferred at the same
rate either way. It varies only with the model and with which throughput basis turns out
to be right — the two open questions §9 lists, and the reason one measured run is worth
making before committing.

### What the ingest scenario actually decides: whether GPUs sit idle

An idle GPU bills. So the question is not how many GPUs the quota allows, it is **how many
GPUs the ingest rate can keep fed**.

A zone-year costs **270.7 GPU-hours on v1.1** at the capacity-planning basis and **196.9 on
v2 Large**. Multiply by the supply rate:

```
  matched fleet  =  zone-years per hour from ingest  ×  GPU-hours per zone-year
```

| ingest scenario | supply | **matched, v1.1 @14K** | **matched, v2 @19.3K** | v1.1 @21K | v2 @28.9K |
|---|---|---|---|---|---|
| **shipped — 40 × 50w** | 5.24 zone-yr/h | **1,419** | **1,032** | 946 | 688 |
| 40 × 60w | 6.29 zone-yr/h | 1,703 | 1,239 | 1,135 | 825 |
| **optimal — 71 × 50w** | 9.30 zone-yr/h | **2,518** | **1,831** | 1,678 | 1,221 |
| 80 × 50w | 10.48 zone-yr/h | 2,837 | 2,064 | 1,891 | 1,375 |

**The correction to the throughput basis (§6) changes what this section recommends, and the
change is large enough to state plainly.** At the capacity-planning rate, the matched fleet
at 71 cells on v1.1 is **2,518 GPUs — at or slightly above the 2,500-actor quota.** The
quota is therefore *approximately correctly sized* for the configuration we intend to run,
not wildly oversized.

An earlier version of this section said provisioning to the quota was a **$550,000
mistake** in every scenario. That claim survives only in two of the four corners:

| ingest scenario | v1.1 @14K duty | v1.1 @14K idle burn | v1.1 @21K duty | v1.1 @21K idle burn |
|---|---|---|---|---|
| shipped — 40 × 50w | 57% | **$407,000** | 38% | $552,000 |
| **optimal — 71 × 50w** | **101% — quota-bound** | **$0** | 67% | $164,000 |

So the standing advice inverts by scenario. **At 40 cells, an oversized fleet is still the
largest waste in the campaign.** At 71 cells on v1.1 at the capacity-planning rate there is
no idle burn to save, because ingest can feed everything the quota allows — and the quota,
not ingest, becomes the constraint on schedule.

**On v2 the original warning survives everywhere**, because a faster model needs less fleet:
1,831 matched against a 2,500 quota is a 27% overprovision worth about **$124,000**. This is
the same coupling noted below — a faster model makes oversizing worse — now with the
correction that on v1.1 at 71 cells there is nothing left to oversize.

**Do not mix the two bases: it costs schedule, not money.** Sizing the fleet on 21K (1,678
GPUs) and then running at the real 14K rate leaves the fleet at two thirds of what the work
needs. GPU-hours are unchanged, so the bill does not move — but inference stretches from
**4.5 days to 6.7**, and the campaign becomes inference-bound rather than ingest-bound. This
is the single easiest error to make from this document, because 1,678 and 21K each look
defensible in isolation.

**Note which way v2 pushes this.** A faster model makes oversizing *worse*, not better:
each zone-year is consumed more quickly, so the same fleet starves sooner and idles
longer. Choosing v2 saves $137,000 on the work and would lose $124,000 of it back to idle if
the fleet were then left at the quota. The two decisions are coupled and must be made
together — and on v1.1 at 71 cells the coupling disappears only because the matched fleet has
grown past the quota, not because the effect stopped existing.

So the case for the optimal ingest configuration is not the $1,700 it costs, nor the 3.4
days it saves, but this: **it raises the fleet you can keep busy from 1,419 GPUs to the full
2,500 quota on v1.1, or from 1,032 to 1,831 on v2** — which is what converts GPU quota into
finished work rather than idle time. At 40 cells, more than a thousand actors of quota cannot
be used no matter how much is granted.

> The GPU fleet already boots only when a finished mosaic is waiting, and the feeder takes
> whichever mosaic lands first, so the pipeline does not idle *within* a cluster. What the
> table above measures is different and coarser: whether the campaign as a whole generates
> mosaics fast enough to justify the fleet size you provisioned.

---

## 6. Throughput, and the v2 Large model

**Cost the campaign at ~14K px/s. An earlier version of this section costed it at 21K and
that was wrong** — wrong on two independent counts, and the correction moves the matched
fleet enough to change what §5 recommends.

The profiling doc reports three rates, and they are not three estimates of one quantity:

| rate | what it measures | profiling doc's own label |
|---|---|---|
| **21–24K px/s** | while processing a **mid-density** chunk | — |
| **10–18K px/s** | while processing a **dense** chunk | — |
| **~13–15K px/s** | total pixels ÷ total GPU-hours, whole run | **"the capacity-planning number"** |

The superseded argument was that the truth lies between the while-processing and
fleet-overall figures and sits "probably nearer 21K, since cold starts amortise over a
999-zone-year run far better than over one ROI". Both halves fail:

1. **21–24K is the MID-density rate, and this campaign is dense-weighted.** Zones are dealt
   densest-first and dense zones dominate the pixel volume, so the applicable
   while-processing band is the dense one, **10–18K** — which straddles 14K rather than
   sitting above it. Cold-start amortisation cannot move a dense zone onto the mid-density
   rate; those are different axes, and the earlier argument conflated them.
2. **13–15K is the only rate derived the way a campaign consumes GPUs** — pixels delivered
   per GPU-hour paid, over a complete run. That is the quantity a bill and a schedule are
   both denominated in, which is why the profiling doc calls it the capacity-planning
   number. Costing against a while-processing rate silently assumes the fleet is never
   between chunks.

The range is still carried rather than collapsed — it is worth **$176,000** on-demand — but
the **primary basis is 14K**, with 21K retained as the optimistic bound and 24K as the
best-observed ceiling.

### v2 Large is materially faster — worth $122,000 – $137,000

`feature/v2-large-model` carries a **per-model inference rate**, in
`inference/actors.py`:

```python
_EST_PX_PER_SEC_BY_MODEL = {"v1.1": 16_000.0, "v2-large": 22_000.0}
```

**A ratio of 1.375×.** Applied to the campaign's costing rates:

| basis | v1.1 | v2 Large | inference cost, v1.1 → v2 | saving |
|---|---|---|---|---|
| **capacity-planning — 14K** | **14,000 px/s** | **19,250 px/s** | **$503,300 → $366,000** | **$137,300** |
| 15K fleet-overall, upper end | 15,000 px/s | 20,625 px/s | $469,700 → $341,600 | $128,100 |
| optimistic — 21K mid-density | 21,000 px/s | 28,875 px/s | $335,500 → $244,000 | $91,500 |

**Two caveats on that number, both real.** The constants are labelled a *strategy-only
estimator* — they exist so the striping planner can ask "will the GPU stay busy long
enough to hide this read?", explicitly "never a correctness value". And the commit that
added them says "per-model calibration lives in context_docs", but no such calibration is
on that branch: the pointer is dangling. So the **ratio** is the branch's own working
assumption and is the defensible thing to carry; the absolute 16,000 is a third basis
again, matching neither the 21–24K while-processing nor the 13–15K fleet-overall figures.

**A faster model also shrinks the fleet you need**, because the matched fleet is set by
GPU-hours per zone-year:

| | GPU-h per zone-year | matched fleet, 40 cells | matched fleet, 71 cells |
|---|---|---|---|
| **v1.1 @ 14K** | **270.7** | **1,419** | **2,518 — over quota** |
| **v2 Large @ 19.3K** | **196.9** | **1,032** | **1,831** |
| v1.1 @ 21K (optimistic) | 180.5 | 946 | 1,678 |
| v2 Large @ 28.9K (optimistic) | 131.3 | 688 | 1,221 |

### Correcting my own earlier estimate

An earlier version of this document derived the v2/v1.1 ratio from the two `ModelArch`
definitions and put it at **1.12×**, then argued that since inference is not tensor-bound
the real gain would be *smaller* still — "expect 0–11%, plan on 0". The branch's own
calibration says **1.375×**, so that was wrong by a factor of 1.22 and wrong in direction
of argument.

The reasoning error is worth keeping. Not being tensor-bound does not mean a smaller model
gains less; it means the arithmetic is not what it gains on. A narrower model also moves
less weight and activation traffic, launches smaller kernels, and does less host-side work
per token — and those are precisely the things the profiling run identified as the actual
bottleneck. A FLOP ratio was the wrong instrument for a workload measured at 0.12–0.26
tensor-pipe utilisation.

The architecture comparison is retained below because it explains *why* v2 is cheaper at
all; it is not a throughput prediction.

Reading both `ModelArch` definitions (`config/inference.py` on that branch):

| | v1.1 | v2 Large |
|---|---|---|
| `latent_dim` → `d_model` | 192 → 768 | 160 → 640 |
| `dim_feedforward` | 2048 | 2560 |
| layers × heads | 4 × 4 | 4 × 4 |
| representation | 192-D, first 128 saved | 128-D native (Matryoshka) |
| encoder params (both streams) | ~44.0M | ~39.3M |

Per-token multiply-accumulates, across every sequence bucket from T=8 to T=256:

| T | v1.1 | v2 Large | ratio |
|---|---|---|---|
| 8 | 22.07M | 19.70M | 0.893 |
| 64 | 22.41M | 19.99M | 0.892 |
| 256 | 23.59M | 20.97M | 0.889 |

**v2 Large is about 0.89× the arithmetic of v1.1.** The narrower model dimension more than
offsets the wider feed-forward. "Large" is the student-size tier within the v2 family; it
is not larger than v1.1. The measured 1.375× speedup exceeds this comfortably, which is
the evidence that arithmetic is not what the pipeline is spending its time on.

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

Ingest at its cost midpoint; inference at the **14K capacity-planning basis** (§6) with a
**matched** GPU fleet, so no idle burn in any column. All four combinations of the two
decisions still open. An earlier version of this table was built on the 21K basis; those
figures are kept as a final row so the difference is visible rather than silently replaced.

| | shipped 40×50w<br>**v1.1** | shipped 40×50w<br>**v2 Large** | optimal 71×50w<br>**v1.1** | optimal 71×50w<br>**v2 Large** |
|---|---|---|---|---|
| Fargate vCPU required | 12,640 | 12,640 | 22,436 | 22,436 |
| Matched GPU fleet | 1,419 | 1,032 | **2,518 — exceeds quota** | 1,831 |
| Ingest | $121,000 | $121,000 | $121,000 | $121,000 |
| Inference | $503,300 | $366,000 | $503,300 | $366,000 |
| Assembly + S3 + mosaics | $4,800 | $4,800 | $4,800 | $4,800 |
| **Total** | **$629,000** | **$492,000** | **$629,000** | **$492,000** |
| Ingest wall clock | 7.9 d | 7.9 d | 4.5 d | 4.5 d |
| Inference wall clock at the matched fleet | 7.9 d | 7.9 d | 4.5 d | 4.5 d |
| Campaign wall clock (staged) | ~8.1 d | ~8.1 d | ~4.6 d | ~4.6 d |
| Idle burn if you run 2,500 GPUs anyway | +$407,000 | +$538,000 | **$0 — quota-bound** | +$124,000 |
| *same row on the superseded 21K basis* | *$461,000 total* | *$370,000* | *$461,000* | *$370,000* |

Read the columns in pairs. **Choosing v2 Large is worth $137,000 at this basis and is
independent of the ingest configuration.** **Choosing 71 cells is worth 3.4 days and costs
nothing.** What has changed from the earlier version is the bottom pair of rows: at 71 cells
on v1.1 the matched fleet now *exceeds* the quota, so there is no idle burn to avoid and the
value of the 71-cell configuration is entirely schedule.

At the optimistic 21K basis instead, subtract **$168,000** from the v1.1 columns and
**$122,000** from the v2 columns.

**The two ingest scenarios cost the same to within a rounding error.** The decision between
them is about wall clock — 7.9 days against 4.5 — and about how much GPU quota you can
convert into work rather than idle: 1,419 GPUs against the full 2,500. It is not about money
spent on ingest.

**One consequence for the deadline.** All years must be validated by **2026-09-11**. Every
column above is 4.6 to 8.1 days of campaign wall clock, so the compute is not what threatens
that date; the schedule risk lives in the preflight gates (§10) and in the Fargate quota
lead time, not in the run.

---

## 9. Assumptions and uncertainties, largest first

1. **Throughput, 14K versus 21K px/s.** Worth **$168,000**, and the largest open number in
   the model. §6 argues for 14K on the profiling doc's own labelling, but that is a reading
   of someone else's measurement, not a measurement of this campaign. Resolvable with one
   instrumented dense-zone run — which also settles whether the *dense* while-processing
   band (10–18K) or the fleet-overall figure is the better predictor at campaign scale.
2. **The matched fleet at 71 cells sits within 1% of the 2,500-actor quota** (2,518). A
   figure that close to a hard limit should not be treated as showing the quota suffices:
   the supply rates feeding it are themselves 3.6% understated (§4), and per-zone ingest
   durations carry ±35%. Treat 71 cells on v1.1 as *quota-bound*, and expect to be deciding
   between raising the GPU quota and accepting a longer inference tail.
3. **No measured v2 Large throughput exists.** The 0.89× compute ratio is derived from the
   architectures, and the pipeline is not compute-bound, so it may not appear at all.
4. **Fleet-matching assumes ingest and inference stay in lockstep, and they do not.** The
   duty-cycle arithmetic treats supply as smooth. It is not: dense zones take far longer than
   sparse ones, and the campaign deals the densest first, so early supply is slower than
   average and late supply faster. **This is now modelled rather than hand-waved** —
   `tests/unit/test_gpu_starvation.py` runs the real per-zone tile counts through one
   cluster's ingest look-ahead and matched actor pool. At 71 cells, a fleet that boots on its
   first mosaic idles about **4.8 GPU-hours per cluster-year** — **108,400 idle GPU-hours,
   about $202,000**, across the campaign. Holding the boot until the queue contains **3.25
   work-hours** of pixels removes it entirely, for **25.5 hours** of added schedule over nine
   years. That is the second-largest cost lever in this document after fleet sizing itself,
   and unlike fleet sizing it is not scenario-dependent: it appears in all 96 combinations
   the model scans. **It is a recommendation, not shipped behaviour.**

   The 3.25 figure is a threshold, not a preference — 3.0 starves in 2 of 96 combinations and
   3.25 in none — and it holds for **both models on one number**, because it is denominated in
   work-hours for the fleet that will consume them. A matched fleet shrinks in inverse
   proportion to the model's rate (315 actors at 14K and 229 at 19.25K both consume 4.41
   Mpx/s, within 0.05%), so the same work-hours means the same pixels. A mosaic count or a
   pixel threshold would each need re-deriving per model.
5. **Ingest carries the ±10% per-date fit** from its own estimate, plus the untested
   assumption that cell interference stays flat above 20 concurrent cells. Per-zone
   durations additionally carry **±35%** (§4), which the aggregate basis hides.
6. **The v2 rate's provenance is undocumented.** 22,000 against 16,000 px/s is the
   branch's own planning estimator, labelled strategy-only, and the calibration its
   comment points at is not on the branch. The 1.375× ratio is worth $122,000–$137,000 and
   currently rests on a constant nobody has shown their working for.
7. **Pixel count.** The 2048-tile census gives 1.363 × 10¹³; the ingest estimate's
   4096-chunk census implies 1.459 × 10¹³, about 7% higher. The larger figure would add
   ~$23,000 to inference.

---

## 10. What to do about it

1. **Size the GPU fleet to the ingest supply rate, not to the quota** — at the
   capacity-planning basis that is **1,419 GPUs at 40 cells and 2,518 at 71**. Provisioning
   the full 2,500-actor quota against 40-cell ingest is a **$407,000** mistake, and with spot
   excluded it is the only six-figure lever left. At 71 cells on v1.1 the mistake is not
   available: the matched fleet already exceeds the quota. **Decide the ingest cap first** —
   it determines whether fleet sizing is a lever at all.
2. **Raise the Fargate quota to ~23,000 vCPU and raise `max_parallel_ingest` from 40 to
   71.** No width change is needed — 50 workers is already the shipped default. It costs
   nothing measurable, halves ingest wall clock from 7.9 days to 4.5, and raises the GPU
   fleet you can keep busy from 1,419 to the full 2,500-actor quota.
3. **Run v2 Large, and measure one dense zone end to end on it.** The branch's own rate
   makes it worth $122,000–$137,000, which is second only to fleet sizing — and the same
   run settles the 14K-versus-21K question ($168,000) and puts a documented number behind
   the 1.375× ratio. One run, three answers.

4. **Do not carry the 21K basis and the 1,678-GPU fleet together into any plan.** They came
   from the same superseded version of §6 and each looks defensible alone; together they
   under-provision by a third and stretch inference from 4.5 days to 6.7.

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
