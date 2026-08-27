# Is a different GPU worth its availability? (2026-08-27)

**The question.** Widening the GPU pool by instance SIZE is dead: every `g6e` size is the
same L40S drawing on one us-west-2 capacity pool, and all eight refused together
(`gpu-host-packing-2026_08.md`). A capacity reservation is not available. So the only
remaining lever is a different CARD — and peak VRAM turned out to be 4.6–8.3 GiB, not the
~43 GiB an earlier reading implied, which is what makes a 22.4 GiB card arguable at all.

Two cards are buyable when the L40S is not: the **L4** (`g6.*`) and the **A10G** (`g5.*`).
They have identical VRAM and are ordered **oppositely** on bandwidth and tensor compute, so
running them side by side tests a hypothesis rather than collecting a second price quote.

**Answer: no, on both cost and crash safety — and the hypothesis behind the question is
confirmed in its ordering and refuted in its magnitude.**

| deliverable | status |
|---|---|
| Vendor bandwidth figures (864 / 600 / 300 GB/s) | **verified** — all three exactly right |
| A10G vs L4 forward-pass throughput, identical shapes | **measured** — A10G 1.29–1.42× the L4 |
| Whether the ranking follows bandwidth or architecture | **bandwidth** — the older card wins |
| Whether bandwidth alone explains the magnitude | **no** — 1.41× delivered against 2.0× predicted |
| Deepest sequence a 22.4 GiB card can run | **measured** — OOM at 208; 232 with an allocator flag |
| Crash verdict per card | **DISQUALIFYING as configured** — see below |
| $ per unit of work | **measured** — both candidates cost more per unit than an L40S |
| End-to-end per-host tok/sec on one queue | **measured**, thin: capacity gave 3 A10G + 2 L4 |
| DCGM SMACT / TENSO / DRAMA per card | **measured** |
| L40S in the same run as the candidates | **NOT measured** — `g6e` refused all day |

## The two hypotheses, and why the A10G is the only card that separates them

Fleet telemetry says the L40S runs at SMACT ≈ 0.99 with TENSO only 0.42–0.47 and effective
TFLOPS flat at 85 — SMs always occupied, tensor pipes under half engaged. That reads as
memory-bandwidth bound rather than compute bound. If it is right, the ranking of cards
follows bandwidth and not architecture generation.

Vendor figures, **verified from the vendors' own documents** because AWS publishes no GPU
bandwidth at all (`ec2:describe-instance-types` returns GPU name, count and VRAM size only):

| card | VRAM | memory bandwidth | BF16 dense tensor, as quoted | source |
|---|---:|---:|---:|---|
| L40S | 45,776 MiB | **864 GB/s** | 362 TFLOPS (733 with sparsity) | nvidia.com/en-us/data-center/l40s/ |
| A10G | 22,888 MiB | **600 GB/s** | 70 TFLOPS (140 with sparsity) | NVIDIA/AWS *A10G Tensor Core GPU* datasheet, Feb 2022 |
| L4 | 22,888 MiB | **300 GB/s** | 121 TFLOPS (242 with sparsity) | nvidia.com/en-us/data-center/l4/ |

All three bandwidth figures asked about were exactly right. VRAM comes from
`ec2:describe-instance-types`; the A10G and L4 report the identical 22,888 MiB.

**Why the L4 alone could settle nothing.** Against an L40S, bandwidth predicts 300/864 =
0.35 and compute predicts 121/362 = 0.33. Indistinguishable. So the L4-vs-L40S figure the
previous attempt produced (0.30–0.36) is consistent with both readings and licenses
neither.

**Why the A10G separates them.** It has **twice** the L4's bandwidth on **0.58×** its
tensor compute. Bandwidth says the A10G beats the L4 by 2.0×; compute says the L4 beats the
A10G by 1.7×. There is no reading on which both are true.

## Method

Three instruments, because no single one answers both halves of the question.

1. **A forward-pass sweep on synthetic tensors** (`profiling/inference/forward_bench.py`),
   sequence length 8 → 256 at the production `B=7168` and bf16. Same tensors, same shapes,
   same dtype on every card: no S3 weather, no host feed, no geography, no optical-depth
   spread. This is the clean card comparison, and it is the ONLY way to reach the model's
   own 256-timestep ceiling — `iowa_epsg5070` tops out near 123 and the deepest `t_kept`
   ever recorded anywhere in the campaign is **206**, so no cell can exercise the worst
   case. A fleet run can only ever report "it did not OOM on the chunks we drew".
2. **One cluster, three arms, one work queue** — `gpu-worker-ladder =
   g6e.xlarge:3,g6.2xlarge:3,g5.2xlarge:3` on `iowa_epsg5070`, ascending-only so the cell
   is a single radar stratum. Same cell, same minutes, confounders removed by construction.
   The `g6e` arm was included precisely because it costs nothing when refused, and it was
   refused.
3. **Exactly-matched chunk pairs across runs** — the same chunk label at the same `t_kept`
   and the same `valid_px`, compared across today's three dev runs. Worth far more than two
   medians when the per-chunk spread is 2–3×: geography, depth and radar content are
   identical by construction. `temp/matched_pairs.py` in yield-embeddings does the join,
   keyed on `vram_total_gib` because the earlier runs' instances are already terminated and
   drop out of `ec2:describe-instances`.

`g5.2xlarge` and `g6.2xlarge` are the shapes to measure on: 8 vCPU and **32 GiB host RAM,
matching `g6e.xlarge` exactly**. The `xlarge` sizes of both families are 16 GiB, below the
measured per-actor requirement, and are deliberately not offered as rungs.

**One confound, named and quantified.** Both candidate shapes give 8 vCPU per GPU against
the production `g6e.xlarge`'s 4. That is a real, buyable advantage — but it flatters them
on the wall-clock measure. Measured directly on the same chunk: overhead outside inference
was **35.2 s on the A10G's 8-vCPU host against 46.9 s on the L40S's 4-vCPU host**, so the
CPU surplus is worth about 12 s per chunk. The `infer_s` figures isolate the GPU and carry
none of it.

## The forward-pass sweep: the A10G beats the L4 at every depth

`B=7168`, bf16, `t_s1 = 0.4 × t_s2` (the campaign's shape: 103 ascending against 263
optical on this cell), 3 warmup + 10 timed forwards per rung.

| t_s2 | t_s1 | A10G tok/sec | L4 tok/sec | **A10G / L4** | A10G TFLOPS | L4 TFLOPS | VRAM alloc | VRAM reserved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3 | 629,441 | 497,653 | **1.26** | 27.8 | 22.0 | 0.88 | 1.70 |
| 32 | 13 | 766,170 | 561,258 | **1.37** | 34.0 | 24.9 | 3.09 | 5.45 |
| 64 | 26 | 787,967 | 568,814 | **1.39** | 35.2 | 25.4 | 6.05 | 10.52 |
| 96 | 38 | 793,155 | 564,944 | **1.40** | 35.7 | 25.4 | 9.01 | 15.48 |
| 128 | 51 | 791,228 | 558,358 | **1.42** | 35.9 | 25.3 | 11.97 | 20.58 |
| 160 | 64 | 646,024 | 493,276 | **1.31** | 29.5 | 22.5 | 14.93 | 21.63 |
| 192 | 77 | 618,711 | 481,436 | **1.29** | 28.5 | 22.1 | 17.89 | 21.56 |
| 224 | 90 | **OOM** | **OOM** | — | — | — | 20.84 at failure | 20.96 |

**Peak VRAM is identical on the two cards at every rung, to the second decimal**, and both
refuse at exactly the same depth. That is the control the whole comparison rests on: the
working set is a property of the work, not of the card, so every difference in the table is
throughput.

### The hypothesis: confirmed in its ordering, refuted in its magnitude

**The ranking follows bandwidth.** The A10G beats the L4 by 1.26–1.42× while having 0.58×
its quoted tensor throughput and 2.0× its bandwidth. On a compute-bound workload the L4
would win by ~1.7×. It loses by ~1.4×. The older Ampere card beats the newer Ada one, which
is what the hypothesis predicted and is the opposite of what generation alone would suggest.

**Bandwidth does not explain the size of the gap.** A 2.0× bandwidth ratio delivered 1.41×.
So roughly 30% of the bandwidth advantage does not convert, and this measurement does not
say where it goes. Two candidates, both measured, neither isolated:

- **The L4 is power limited.** Under load it drew **63 W of its 72 W TDP** (87%) and its
  `clocks_throttle_reasons` bitmask read `0x4` — SwPowerCap — with the SM clock dropping to
  975 MHz against a 2040 MHz maximum. The A10G drew **218 W of 300 W** (73%). The L4 is the
  cheapest card in the comparison precisely because it is a 72 W part, and that envelope is
  binding. A card held back by power is not a card held back by bandwidth, and it means the
  L4's shortfall is over-attributed to bandwidth by this reading.
- **Neither candidate is at a hard memory wall by DCGM's own counter.** DRAMA (DRAM_ACTIVE)
  read **0.50–0.51 on the A10G and 0.50–0.59 on the L4** during inference, against SMACT
  0.99 on both. A truly bandwidth-saturated kernel sits far higher. So "bandwidth bound" is
  the right *ranking* rule here and an incomplete *mechanism*.

### Throughput drops before the card fills, not just at the wall

Both cards lose ~20% of their throughput between `t_s2` 128 and 160 — A10G 35.9 → 29.5
TFLOPS, L4 25.3 → 22.5 — and the reserved pool crosses 98% of the card in the same step
(20.58 → 21.63 GiB). So on a 22.4 GiB card the deepest chunks are **both slower and at
risk**, and the two degrade together. There is no regime where such a card runs deep work
at its own best rate.

## The crash verdict

Stated separately from throughput, because a card that runs 20% slow is a cost decision and
a card that OOMs mid-cell wastes GPU-hours.

**Both candidate cards: DISQUALIFYING as currently configured.**

Narrow sweep, default allocator and then with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **Run independently on both cards, and
the two agree at every rung to two decimal places:**

| t_s2 | VRAM needed | default allocator | with `expandable_segments` |
|---:|---:|---|---|
| 192 | 17.89 GiB | ok on both | ok on both |
| 200 | 18.63 GiB | ok on both | ok on both |
| **208** | 19.37 GiB | **OOM on both** (2.25–2.30 GiB free, wanted 6.40) | ok on both |
| 216 | 20.10 GiB | not attempted | ok on both |
| 224 | 20.84 GiB | not attempted | ok on both |
| 232 | 21.58 GiB | not attempted | ok on both |
| **240** | — | not attempted | **OOM on both** (wanted 2.46, had 1.84) |

That the A10G and the L4 refuse at the identical rung, having allocated identical bytes at
every rung below it, is the strongest form this control can take: the working set is a
property of the work and the two cards differ only in speed.

**The default configuration refuses at 208, and the deepest `t_kept` ever recorded in this
campaign is 206.** The model buckets a pixel's observation count to the next multiple of 8,
so a 206-deep cell presents 208. The margin is zero — not thin, zero — and it is reached by
a cell that has already run once (`38N`-2021).

`expandable_segments:True` moves the wall from 208 to 232 on both cards, at no measurable
throughput cost — A10G 27.2 vs 27.8 TFLOPS and L4 22.2 vs 23.2 at matched depth, inside
run-to-run noise. That is a **24-timestep margin over the deepest depth observed** — but it
is still short of the model's own `max(num_obs_checkpoints) = 256` ceiling, which no
configuration tested can reach on a 22.4 GiB card. And the flag is not set anywhere today.

So the honest statement of the crash verdict has three parts. **As configured today, both
candidate cards will OOM on the deepest cell the campaign has already run.** With one
allocator flag they clear it with a 24-timestep margin. Neither can reach the sequence
length the model declares it supports, on any setting tested — which is a statement about
the card, not about the flag.

**What did NOT go wrong**, recorded because absence of failure is also evidence: across the
live cluster run no OOM, no CUDA error, no worker replacement and no actor restart, at
`t_kept` up to 113. The GPU-side reserved pool sat at 21.71 of 22.06 GiB — **98.4% of the
card** — throughout, on both cards, and nothing failed at these depths. It is the depths a
run like this cannot reach that decide the verdict, which is exactly why the synthetic sweep
exists.

## Everything else that was watched

| metric | A10G (`g5.2xlarge`) | L4 (`g6.2xlarge`) | note |
|---|---|---|---|
| peak `max_memory_allocated` per chunk | 7.52 GiB at `t_kept` 108–113 | 7.52 GiB | identical, per chunk, reset each chunk |
| peak `max_memory_reserved` per chunk | 21.68–21.71 of 22.06 GiB | 21.68 of 22.04 | **98.4%** — no fragmentation headroom |
| host RAM peak (1 s sampler) | **16.03 GiB of 31.0 (52%)** | 15.76 GiB (52%) | never above 55%; the 60% ceiling held |
| host RAM peak (2 s in-actor, per chunk) | 13.97–15.9 GiB | 14.0 GiB | on `CHUNK_SUMMARY` as `host_ram_peak_gib` |
| GPU utilisation | 66% avg, 100% max | 76% avg, 100% max | busy fraction 0.67 / 0.77 |
| power draw | **218 W of 300 W** (73%) | **63 W of 72 W** (87%) | the L4 is power limited, the A10G is not |
| thermal throttling | none — 63–72 °C, bitmask `0x0` | none thermal; bitmask `0x4` = **SwPowerCap** | no `HwSlowdown`, no thermal slowdown on either |
| SM clock | 1710 MHz of 1710 max | 975–1695 of 2040 max | the L4's clock drops under the power cap |
| DCGM SMACT | 0.995 | 0.994–0.998 | same as the L40S fleet's 0.99 |
| DCGM TENSO | 0.334–0.340 | 0.482–0.642 | L40S fleet reads 0.42–0.47 |
| DCGM DRAMA | 0.501–0.512 | 0.503–0.587 | neither at a hard memory wall |
| PCIe link | gen4 **x8** | gen4 **x8** | half the L40S's x16 lanes |
| PCIe traffic | 3.2 GB/s in, 0.4 GB/s out | 1.3 GB/s in, 0.17 GB/s out | ~20% of a gen4 x8 link; not binding |
| S3 read concurrency | 24 sockets mean, 126 max | 10 mean, 115 max | no throttling observed |
| `get_batch` latency | not isolated in this run | not isolated | see "what could not be measured" |

**The 16 GiB host-RAM exclusion is now measured rather than inferred.** Peak host RAM was
**16.03 GiB** on a 31 GiB host. A `g5.xlarge` or `g6.xlarge` has 16 GiB total, so the
per-actor working set alone consumes the entire machine. Those sizes are the cheapest per
GPU-hour in either family, which is exactly why the rung exclusion is stated over every
offered node type rather than naming one instance.

## The economics, which is what actually decides it

$/GPU-hour from the Pricing API, us-west-2, on-demand Linux shared tenancy, 2026-08-27. A
card is worth adopting when its throughput ratio **exceeds** its price ratio.

| type | card | $/GPU-hr | price ratio (= break-even) | measured throughput ratio | $ per unit of work |
|---|---|---:|---:|---:|---:|
| `g6e.xlarge` | L40S | 1.861 | — | 1.000 | **1.00×** |
| `g6.2xlarge` | L4 | 0.978 | 0.525 | **0.294** (`infer_s`) / 0.344 (`total_s`) | **1.53–1.79×** |
| `g5.2xlarge` | A10G | 1.212 | 0.651 | **0.42–0.50** (see below) | **1.30–1.55×** |
| `g6.4xlarge` | L4 | 1.323 | 0.711 | as above | 2.07–2.42× |
| `g5.4xlarge` | A10G | 1.624 | 0.873 | as above | 1.75–2.08× |
| `g6e.2xlarge` | L40S | 2.242 | 1.205 | 1.00–1.15 (CPU feed only) | 1.05–1.21× |

**Every candidate costs more per unit of work than the shape we already run.** The A10G is
the better of the two candidates and still lands 30–55% above `g6e.xlarge`. The L4 lands
53–79% above. The `xlarge` sizes would be cheaper per GPU-hour, and cannot hold the actor.

## What could not be measured, named rather than filled in

- **An L40S beside the candidates in one run.** Every `g6e` size refused with
  `InsufficientInstanceCapacity` in all three dev AZs for the entire session, including a
  `g6e.xlarge` arm left standing in the ladder for the whole run. So the L40S reference is
  a **cross-run** comparison against arm A of the same cell earlier the same day, joined on
  exactly-matched chunks. That removes geography and depth but not the hour, and not S3
  weather.
- **A synthetic L40S forward-pass figure.** For the same reason. The L40S number used for
  the ratio comes from the production `EFFECTIVE TFLOPS` line, which is the same instrument
  — but on real data rather than synthetic tensors, so the synthetic sweep has no L40S row.
- **Where the missing 30% of the A10G's bandwidth advantage goes.** Named above as either
  the L4's power cap or a non-bandwidth limit on both; not isolated.
- **Whether the `frac_of_card_tflops` figure is a true utilisation.** A measured A10G
  reached 35.9 TFLOPS against a derived FP32-accumulate ceiling of 35.0, which cannot
  happen — so either the FP16-to-FP32-accumulate halving does not apply to this part, or
  `transformer_flops` overstates the work by counting attention at full T². `_CARD_CEILINGS`
  therefore carries the vendor's quoted dense figure, which cannot be exceeded, and the
  fraction is documented as an index.
- **`get_batch` latency per card.** The instrument exists (`TIMING` lines) but no per-card
  reading was taken; the `infer_s`/`total_s` split covers the same ground more coarsely.
- **Sustained network throughput per card.** S3 socket concurrency was recorded; bytes per
  second were not.
- **Multi-GPU sizes of either family.** `g5.12xlarge` and `g6.12xlarge` refused in all AZs
  every time they were probed, which is consistent with the earlier finding that a squeeze
  bites the 4-GPU boxes first.

## Capacity, corrected

An earlier reading in `gpu-host-packing-2026_08.md` — "the pool is the card" — is **too
strong, and is corrected here**. It came from all eight `g6e` sizes refusing at once, which
is what a hard family-wide shortage looks like. Measured across the day, size availability
within a family varies:

| type | 2a | 2b | 2c |
|---|---|---|---|
| every `g6e` size | refused | refused | refused |
| `g5.xlarge` | launched | launched | launched |
| `g5.2xlarge` | launched | refused, then launched | refused |
| `g5.4xlarge` | launched | refused | refused |
| `g5.12xlarge` | refused | refused | refused |
| `g6.xlarge` | launched | launched | refused |
| `g6.2xlarge` | refused, then launched | launched | launched |
| `g6.4xlarge` | refused | launched | launched |
| `g6.12xlarge` | refused | refused | refused |

The accurate statement: **a hard shortage takes a whole card family down together, and
under mild pressure the larger sizes go first.** The 1-GPU `xlarge` sizes were the most
available in every family and the 4-GPU sizes the least — which runs against the "fewer,
bigger boxes" instinct in both directions.

## Spend

| item | shape | approx cost |
|---|---|---|
| Capacity probes | ~40 single launches, terminated on success | ~$0.10 |
| A10G wide sweep | 1 × `g5.4xlarge`, ~20 min | ~$0.55 |
| L4 wide sweep | 1 × `g6.4xlarge`, ~28 min | ~$0.62 |
| A10G boundary sweep (± allocator flag) | 1 × `g5.4xlarge`, ~13 min | ~$0.35 |
| L4 boundary sweep | 1 × `g6.4xlarge`, ~20 min | ~$0.45 |
| Failed first dispatch | ECS task only; died in the driver before any node | ~$0.00 |
| Cluster run | head + 3–5 GPU workers, ~45 min | ~$5–7 |
| **total** | | **~$8** |

No run was allowed to complete: a full `iowa_epsg5070` pass is hours of GPU time and would
add nothing this question needs. The bench instances were terminated by the harness as each
sweep returned; the cluster was torn down once every arm had its medians.
