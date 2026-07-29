# Global TESSERA campaign — full cost model

**Dated 2026-07-29.** Extends the July-27 ingest cost estimate — which costed **ingest
only** — to the whole campaign: ingest, inference, assembly, and the permanent output
store. Nine years × 111 land zones. The ingest measurements it rests on are recorded in
[`ingest_optimization_campaign_2026_07.md`](ingest_optimization_campaign_2026_07.md);
section references below of the form "ingest estimate §N" point at that working note.

Every input is either measured (and cited) or derived from a measured input (and marked
as derived). Two things are neither, and they are called out in §8 rather than buried.

---

## 1. The headline

| | |
|---|---|
| Ingest (Fargate) | $115,000 – $126,000 |
| **Inference (GPU, on-demand) — v1.1** | **$293,000 – $470,000** |
| **Inference (GPU, on-demand) — v2 Large** | **$213,000 – $342,000** |
| Assembly | ~$200 |
| Mosaic storage (transient) | ~$3,000 |
| **Campaign total, v1.1** | **$411,000 – $599,000** |
| **Campaign total, v2 Large** | **$331,000 – $471,000** |

The permanent embeddings store (0.9–1.8 PB) is **not costed here**: it goes to AWS Open
Data, which sponsors the storage. Sizing it still matters for bucket planning — see §7 —
but it is not a line on this bill.

**GPUs are on-demand. Spot is not costed here and is not an option** — sustaining ~1,700
g6e instances for days makes interruption a certainty rather than a risk, and a campaign
that stalls on capacity is worse than one that costs more. Settled; do not re-open.

**Three findings worth acting on.**

1. **Inference, not ingest, is where the money is** — two-and-a-half to four times the
   ingest bill. Every optimisation effort so far has gone into the cheaper half.
2. **Inference cost is invariant to the ingest scenario.** Same pixels, same throughput,
   same GPU-hours whether you run 40 cells or 71. What the ingest scenario changes is
   whether the GPU fleet has anything to do.
3. **Fleet sizing is the largest lever, worth up to $550,000** — the cost of provisioning
   to the GPU quota rather than to what ingest can actually feed (§5). **Running v2 Large
   is the second, at $91,000–$128,000** (§6). Both are within our control.

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
| Inference throughput | 15K / 21K / 24K px/s/worker | see §6 |
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

---

## 5. Inference — the cost is fixed; the waste is not

GPU-hours follow from pixels and throughput alone:

```
  GPU-hours  =  1.363 × 10¹³ px  ÷  (px/s/worker × 3600)
```

| throughput | GPU-hours | cost @ $1.861/h |
|---|---|---|
| 15K px/s — fleet-overall, incl. cold starts | 252,300 | $469,600 |
| **21K px/s — measured, while processing** | **180,200** | **$335,400** |
| 24K px/s — best observed | 157,700 | $293,500 |

**None of this varies with the ingest scenario.** The same pixels are inferred at the same
rate either way.

### What the ingest scenario actually decides: whether GPUs sit idle

An idle GPU bills. So the question is not how many GPUs the quota allows, it is **how many
GPUs the ingest rate can keep fed**.

At 21K px/s a zone-year costs **180.4 GPU-hours**. Multiply by the supply rate:

```
  matched fleet  =  zone-years per hour from ingest  ×  180.4 GPU-hours each
```

| ingest scenario | supply | **matched fleet** | duty at 1,280 GPUs | duty at 2,500 GPUs |
|---|---|---|---|---|
| **shipped — 40 × 50w** | 5.24 zone-yr/h | **946 GPUs** | 74% | **38%** |
| 40 × 60w | 6.29 zone-yr/h | 1,135 GPUs | 89% | 45% |
| **optimal — 71 × 50w** | 9.30 zone-yr/h | **1,678 GPUs** | 100% | **67%** |
| 80 × 50w | 10.48 zone-yr/h | 1,891 GPUs | 100% | 76% |

**This is the expensive mistake available in this campaign.** Running the full
2,500-actor GPU quota against ingest as shipped leaves the fleet **38% busy** and burns
roughly **$550,000** of idle GPU time on demand — more than four times the entire ingest
bill, and more than the inference it is trying to do. At 40 × 60w it is $404,000; against
71-cell ingest it falls to $164,000. Neither is a
throughput problem; both are a scheduling mismatch.

So the honest way to state the case for the optimal ingest configuration is not the
$1,700 it costs, nor even the two days it saves, but this: **it raises the fleet you can
keep busy from 1,135 to 1,678 GPUs**, which is what converts GPU quota into finished work
instead of idle time.

> The GPU fleet already boots only when a finished mosaic is waiting, and the feeder takes
> whichever mosaic lands first, so the pipeline does not idle *within* a cluster. What the
> table above measures is different and coarser: whether the campaign as a whole generates
> mosaics fast enough to justify the fleet size you provisioned.

---

## 6. Throughput, and the v2 Large model

**The 21K figure is a while-processing rate**; the profiling doc's 13–15K is total pixels
over total GPU-hours for a complete run, and so absorbs cold starts and the dense-chunk
mix. For a campaign average the truth is between them and probably nearer 21K, since cold
starts amortise over a 999-zone-year run far better than over one ROI. The range is
carried through rather than collapsed: it is worth **$176,000** on-demand.

### v2 Large is materially faster — worth $91,000 – $128,000

`feature/v2-large-model` carries a **per-model inference rate**, in
`inference/actors.py`:

```python
_EST_PX_PER_SEC_BY_MODEL = {"v1.1": 16_000.0, "v2-large": 22_000.0}
```

**A ratio of 1.375×.** Applied to the campaign's costing rates:

| basis | v1.1 | v2 Large | inference cost, v1.1 → v2 | saving |
|---|---|---|---|---|
| optimistic — 21K while-processing | 21,000 px/s | 28,900 px/s | $335,400 → $243,900 | **$91,500** |
| pessimistic — 15K fleet-overall | 15,000 px/s | 20,600 px/s | $469,600 → $341,500 | **$128,100** |

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
| v1.1 @ 21K | 180.4 | 946 | 1,678 |
| **v2 Large @ 28.9K** | **131.2** | **688** | **1,220** |

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

Ingest at its cost midpoint; inference at 21K px/s with a **matched** GPU fleet, so no
idle burn in either row.

| | shipped — 40 × 50w | optimal — 71 × 50w |
|---|---|---|
| Fargate vCPU required | 12,640 | 22,436 |
| Matched GPU fleet | 946 | 1,678 |
| Ingest | $121,000 | $121,000 |
| Inference | $335,400 | $335,400 |
| Assembly + S3 + mosaics | $4,800 | $4,800 |
| **Total, v1.1** | **$461,000** | **$461,000** |
| **Total, v2 Large** | **$370,000** | **$370,000** |
| Ingest wall clock | 7.9 d | 4.5 d |
| Campaign wall clock (staged) | ~8.1 d | ~4.6 d |
| Idle burn if you run 2,500 GPUs anyway | **+$550,000** | **+$164,000** |

At 15K px/s rather than 21K, add **$134,000** to both columns (**$98,000** on v2 Large).

**The two scenarios cost the same to within a rounding error.** The decision between them
is about wall clock — 7.9 days against 4.5 — and about how much GPU quota you can convert
into work rather than idle: 946 GPUs against 1,678. It is not about money spent on ingest.

---

## 9. Assumptions and uncertainties, largest first

1. **Throughput, 15K versus 21K px/s.** Worth $134,000, and now the largest open number
   in the model. Resolvable with one instrumented dense-zone run.
3. **No measured v2 Large throughput exists.** The 0.89× compute ratio is derived from the
   architectures, and the pipeline is not compute-bound, so it may not appear at all.
4. **Fleet-matching assumes ingest and inference stay in lockstep.** The duty-cycle
   arithmetic treats supply as smooth. It is not: dense zones take far longer than sparse
   ones, and the campaign deals the densest zones first, so early supply is slower than
   average and late supply faster. Expect real duty to be somewhat below the table.
5. **Ingest carries the ±10% per-date fit** from its own estimate, plus the untested
   assumption that cell interference stays flat above 20 concurrent cells.
6. **The v2 rate's provenance is undocumented.** 22,000 against 16,000 px/s is the
   branch's own planning estimator, labelled strategy-only, and the calibration its
   comment points at is not on the branch. The 1.375× ratio is worth $91,000–$128,000 and
   currently rests on a constant nobody has shown their working for.
7. **Pixel count.** The 2048-tile census gives 1.363 × 10¹³; the ingest estimate's
   4096-chunk census implies 1.459 × 10¹³, about 7% higher. The larger figure would add
   ~$23,000 to inference.

---

## 10. What to do about it

1. **Size the GPU fleet to the ingest supply rate, not to the quota** — 946 GPUs at 40
   cells, 1,678 at 71. Provisioning the full 2,500-actor quota against 40-cell ingest is
   a $550,000 mistake, and with spot excluded it is the only six-figure lever left.
2. **Raise the Fargate quota to ~23,000 vCPU and raise `max_parallel_ingest` from 40 to
   71.** No width change is needed — 50 workers is already the shipped default. It costs
   nothing measurable, halves ingest wall clock from 7.9 days to 4.5, and raises the GPU
   fleet you can keep busy from 946 to 1,678.
3. **Run v2 Large, and measure one dense zone end to end on it.** The branch's own rate
   makes it worth $91,000–$128,000, which is second only to fleet sizing — and the same
   run settles the 15K-versus-21K question ($134,000) and puts a documented number behind
   the 1.375× ratio. One run, three answers.

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
