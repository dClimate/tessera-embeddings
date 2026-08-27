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
| Whether bandwidth alone explains the magnitude | **no** — 1.41× delivered against 2.0× predicted. **Resolved in the addendum: quoted bandwidth is the wrong predictor; DRAMA x bandwidth (achieved traffic) fits both candidates.** |
| Deepest sequence a 22.4 GiB card can run | **measured** — OOM at 208; 232 with an allocator flag |
| Crash verdict per card | **DISQUALIFYING as configured** — see below |
| $ per unit of work | **measured** — both candidates cost more per unit than an L40S |
| End-to-end per-host tok/sec on one queue | **measured** — all three cards, 33 chunks; see the addendum |
| DCGM SMACT / TENSO / DRAMA per card | **measured** — including the L40S in the same run, from the addendum |
| L40S in the same run as the candidates | **MEASURED** — the pool refilled at 18:59Z and the standing arm caught it; see the addendum. This row read "NOT measured" until 2026-08-27 19:45Z. |

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

**Quoted bandwidth does not explain the size of the gap.** A 2.0× bandwidth ratio delivered
1.41×, so roughly 30% of the advantage does not convert. **Resolved 2026-08-27 19:45Z: the
predictor was wrong, not the hypothesis.** Quoted peak bandwidth is a datasheet number; the
counter that predicts throughput is ACHIEVED memory traffic, DCGM DRAMA × quoted bandwidth.
Measured on all three cards in one run, that product predicts both candidates' throughput
ratio against the L40S inside the measured range, where neither quoted bandwidth nor quoted
compute does. Full table in the addendum. The two terms named below were the candidates
before the L40S row existed, and both survive as refinements rather than as the explanation:

- **The L4 is power limited.** Under load it drew **63 W of its 72 W TDP** (87%) and its
  `clocks_throttle_reasons` bitmask read `0x4` — SwPowerCap — with the SM clock dropping to
  975 MHz against a 2040 MHz maximum. The A10G drew **218 W of 300 W** (73%). The L4 is the
  cheapest card in the comparison precisely because it is a 72 W part, and that envelope is
  binding. **Corrected in the addendum: the cap is raised on all three cards, including the
  L40S. What separates them is what it COSTS — 60% of the L4's clock, 30-40% of the L40S's,
  and nothing at all on the A10G, which holds 1665-1710 of its 1710 MHz maximum.**
- ~~**Neither candidate is at a hard memory wall by DCGM's own counter.**~~ DRAMA (DRAM_ACTIVE)
  read **0.50–0.51 on the A10G and 0.50–0.59 on the L4** during inference, against SMACT
  0.99 on both, and this was read as "no card is near a memory wall". **Sharpened once the
  L40S was measured beside them: DRAMA is HIGHEST on the card with the most bandwidth —
  0.66-0.68 on the L40S.** The candidates are not near a wall because they cannot reach one.
  The L40S both moves more bytes per second and keeps its memory interface busy a larger
  fraction of the time, which is why "bandwidth bound" is the right ranking rule.

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
`t_kept` up to **123**. The GPU-side reserved pool sat at 21.71 of 22.06 GiB — **98.4% of the
card** — throughout, on both candidate cards, and nothing failed at these depths. **The L40S
in the same run reserved 28.1–35.8 of its 44.4 GiB, 63–81%**, on identical work: the same
allocator behaviour with headroom left, which is the whole difference between the two card
classes on the crash question. It is the depths a run like this cannot reach that decide the
verdict, which is exactly why the synthetic sweep exists.

## Everything else that was watched

| metric | A10G (`g5.2xlarge`) | L4 (`g6.2xlarge`) | note |
|---|---|---|---|
| peak `max_memory_allocated` per chunk | 7.52 GiB at `t_kept` 108–113 | 7.52 GiB | identical, per chunk, reset each chunk |
| peak `max_memory_reserved` per chunk | 21.68–21.71 of 22.06 GiB | 21.68 of 22.04 | **98.4%** — no fragmentation headroom |
| host RAM peak (1 s sampler) | **16.03 GiB of 31.0 (52%)** | 15.76 GiB (52%) | never above 55%; the 60% ceiling held |
| host RAM peak (2 s in-actor, per chunk) | 13.97–15.9 GiB | 14.0 GiB | on `CHUNK_SUMMARY` as `host_ram_peak_gib` |
| GPU utilisation | 66% avg, 100% max | 76% avg, 100% max | busy fraction 0.67 / 0.77 |
| power draw | **218 W of 300 W** (73%) | **63 W of 72 W** (87%) | ~~the L4 is power limited, the A10G is not~~ — **withdrawn**: all three cards raise SwPowerCap, and it costs the L4 60% of its clock, the L40S 30-40%, the A10G nothing. See the addendum. |
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
| `g6.2xlarge` | L4 | 0.978 | 0.525 | **0.29–0.34** (same-run: 0.292 `infer_s` / 0.318 `total_s`) | **1.53–1.79×**, same-run **1.65×** |
| `g5.2xlarge` | A10G | 1.212 | 0.651 | **0.42–0.53** (same-run: 0.444 `infer_s` / 0.459 `total_s`) | **1.30–1.55×**, same-run **1.42×** |
| `g6.4xlarge` | L4 | 1.323 | 0.711 | as above | 2.07–2.42× |
| `g5.4xlarge` | A10G | 1.624 | 0.873 | as above | 1.75–2.08× |
| `g6e.2xlarge` | L40S | 2.242 | 1.205 | 1.00–1.15 (CPU feed only) | 1.05–1.21× |

**Every candidate costs more per unit of work than the shape we already run.** The A10G is
the better of the two candidates and still lands 30–55% above `g6e.xlarge`. The L4 lands
53–79% above. The `xlarge` sizes would be cheaper per GPU-hour, and cannot hold the actor.

## What could not be measured, named rather than filled in

- ~~**An L40S beside the candidates in one run.**~~ **CLOSED 2026-08-27 19:45Z — it was
  measured.** The claim above was that every `g6e` size refused all session, so the L40S
  reference could only be a cross-run comparison. That held until **18:59Z**, when the
  `g6e.xlarge` arm left standing in the ladder for exactly this possibility caught the pool
  refilling; a second followed at 19:04Z. Both banked 5-6 chunks on the same cell, in the same
  run, in the same minutes as the candidates. The same-run figures are in the addendum and
  they **agree with the cross-run ones**, which is the reason to trust either. The cross-run
  caveat below still applies to the exactly-matched-chunk instrument, and always will:
  inside one run each chunk is processed exactly once, so no chunk can appear on two cards
  and within-run matching returns zero pairs by construction.
- **A synthetic L40S forward-pass figure.** Still not measured, but for a different reason
  now: by the time a `g6e` was launchable, two live instruments already agreed on the L40S
  ratio and a third instance-hour to re-derive it was not worth spending. The consequence is
  narrow — the synthetic sweep's card-versus-card ratio is A10G-to-L4 only, and the L40S
  comparison rests on the live run. The L40S number used in the body's ratio comes from the
  production `EFFECTIVE TFLOPS` line, the same instrument on real rather than synthetic
  tensors.
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
| every `g6e` size, 17:14–18:58 | refused | refused | refused |
| `g6e.xlarge`, 18:59 | — | — | **launched** |
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

**And the drought is intermittent, not permanent.** After refusing continuously from 17:14
to 18:58 — nearly two hours, across four probe rounds and two fleets' worth of autoscaler
retries — a `g6e.xlarge` launched at 18:59 into the measurement cluster, from an arm left
standing in the ladder for exactly that possibility. That is the single most important
capacity fact here: **the L40S pool refills.** A card that is 13–53% cheaper per unit of
work and comes back within hours is a different proposition from one that is gone.

## Spend

| item | shape | approx cost |
|---|---|---|
| Capacity probes | ~40 single launches, terminated on success | ~$0.10 |
| A10G wide sweep | 1 × `g5.4xlarge`, ~20 min | ~$0.55 |
| L4 wide sweep | 1 × `g6.4xlarge`, ~28 min | ~$0.62 |
| A10G boundary sweep (± allocator flag) | 1 × `g5.4xlarge`, ~13 min | ~$0.35 |
| L4 boundary sweep | 1 × `g6.4xlarge`, ~20 min | ~$0.45 |
| Failed first dispatch | ECS task only; died in the driver before any node | ~$0.00 |
| Cluster run, to 18:58Z | head + 3–5 GPU workers | ~$5–7 |
| Cluster run, 18:59–19:32Z | head + 8 GPU workers (3 A10G, 3 L4, **2 L40S**) at $10.30/GPU-hr | ~$6 |
| **total** | | **~$14** |

No run was allowed to complete: a full `iowa_epsg5070` pass is hours of GPU time and would
add nothing this question needs. The bench instances were terminated by the harness as each
sweep returned. **The cluster was NOT torn down when the first arms had their medians** — it
was held while the `g6e` arm, which had just become launchable after a two-hour drought,
banked enough chunks to clear the three-chunk cold-start bar. That extra ~25 minutes cost
about $4 and bought the one row the body could not fill. Torn down at 19:32Z by cancelling
the flow run, so the finalizer ran; zero GPU instances remained in dev afterwards.

---

# Addendum — the L40S measured beside the candidates (2026-08-27, 19:00–19:32Z)

Everything above was written while `g6e` was refusing, so its L40S column is a cross-run
comparison. **The pool refilled at 18:59Z**, a `g6e.xlarge` launched into the arm left standing
in the ladder for exactly that possibility, a second followed at 19:04Z, and both banked 5–6
chunks before the cluster was torn down. This addendum is that measurement. Where it
contradicts the body, the body has been corrected in place and the old claim named.

Run `s2only-7d4b56931c1b` / flow run `e91ec54b-ea60-4832-af6d-659300ce3130`
(`gpucards-l4-vs-a10g-2`), ladder `g6e.xlarge:3,g6.2xlarge:3,g5.2xlarge:3`, cell
`iowa_epsg5070`, ascending-only, all 33 chunks one-orbit. Tokens are the combined
optical-plus-radar identity; `busy` divides by `total_s`, `infer` by `infer_s`.

## Hardware confirmed from inside every host before any figure was trusted

An earlier attempt at this comparison silently measured an L4 while believing it an L40S, so
identity is asserted three ways per row — IMDS `instance-type`, `nvidia-smi --query-gpu=name`,
and the actor's own `vram_total_gib` on the `CHUNK_SUMMARY` line — cross-checked against
`ec2:describe-instances`. All eight workers agreed on all four, at 100% GPU utilisation.

## Throughput

| shape | card | hosts | chunks | tok/s busy | tok/s infer | ratio busy | ratio infer | $/GPU-hr | Mtok/$ | $ per unit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `g6e.xlarge` | **L40S** | 2 | 11 | **2.06M** | **2.38M** | 1.000 | 1.000 | 1.861 | **3975** | **1.00×** |
| `g5.2xlarge` | A10G | 3 | 15 | 0.94M | 1.06M | **0.459** | **0.444** | 1.212 | 2799 | **1.42×** |
| `g6.2xlarge` | L4 | 1 | 4 | 0.65M | 0.69M | **0.318** | **0.292** | 0.978 | 2405 | **1.65×** |

Host spread inside the L40S arm is **1.5%**, tighter than the 4.3% noise floor the single-type
control arm measured. Holding optical depth constant (`t_kept` 100–125, 22 chunks) moves the
ratios to 0.468/0.456 for the A10G and 0.330/0.306 for the L4 — i.e. **not at all**, so the
differing chunk mixes across arms were not carrying the result.

**Both candidate shapes are `.2xlarge` (8 vCPU) against the L40S's `.xlarge` (4 vCPU)**, which
flatters the candidates on `busy` and not on `infer`. Network runs the other way: `g6e.xlarge`
is "Up to 20 Gigabit" against "Up to 10 Gigabit" on both candidates. Both are burst credits.

This **confirms the cross-run estimates in the body** (A10G 0.42–0.50, L4 0.294–0.344) rather
than displacing them, and the agreement between two instruments confounded in opposite
directions is the reason to trust either.

## DCGM, all three cards on the same work at the same minute

45 one-second samples per host, 19:16Z. The body's L40S column came from the prod fleet.

| card | GRACT | SMACT | TENSO | DRAMA | quoted bandwidth | DRAMA × bandwidth |
|---|---:|---:|---:|---:|---:|---:|
| L40S (`g6e.xlarge`) ×2 | 0.995 | 0.987–0.991 | 0.419–0.440 | **0.659–0.675** | 864 GB/s | **~576 GB/s** |
| A10G (`g5.2xlarge`) | 0.998 | 0.997 | 0.336 | 0.501 | 600 GB/s | **~301 GB/s** |
| L4 (`g6.2xlarge`) | 0.998 | 0.996 | 0.515 | 0.559 | 300 GB/s | **~168 GB/s** |

The dev L40S reproduces the prod fleet's SMACT 0.99 / TENSO 0.42–0.47 almost exactly, which
is a useful check that a two-host arm sees the same regime as 338 prod hosts.

**Achieved traffic is the predictor that works:**

| predictor, as a ratio to the L40S | A10G | L4 | agrees with measurement? |
|---|---:|---:|---|
| quoted BF16 dense tensor compute | 0.193 | 0.334 | no — off by 2.4× for the A10G |
| quoted peak memory bandwidth | 0.694 | 0.347 | no — overstates the A10G by ~35% |
| **DRAMA × quoted bandwidth** | **0.523** | **0.291** | **yes, both cards** |
| measured | 0.416–0.526 | 0.276–0.342 | — |

## Power: the cap is raised on all three cards

| card | draw / limit | SM clock / max | bitmask |
|---|---|---|---|
| L4 ×3 | 71–73 W of 72 W (99–101%) | **810–840 of 2040 MHz — 40%** | `0x4` on all 3 |
| L40S ×2 | 339–341 W of 350 W (97%) | 1470–1785 of 2520 MHz — 58–71% | `0x4` on both |
| A10G ×3 | 267–282 W of 300 W (89–94%) | **1665–1710 of 1710 MHz — 97–100%** | `0x4` on 2, `0x0` on 1 |

So the body's "the L4 is power limited, the A10G is not" is **withdrawn**. The envelope binds
on every card; the A10G is simply the only one that keeps its full clock while it binds.

## Ray reaches for the right rung — and, between the two candidates, the wrong one

Ray's scorer run directly against post-autodetect resource dicts, with its autoscaler's
`functools.partial` binding, at demand from 1 to 250 bundles:

| rung | instance type | vCPU | score |
|---|---|---:|---|
| `gpu-workers-ondemand` | `g6e.xlarge` | 4 | `(True, 2, 0.0, 0.265625)` |
| `gpu-workers-ondemand-l4-2xl` | `g6.2xlarge` | 8 | `(True, 2, 0.0, 0.25390625)` |
| `gpu-workers-ondemand-a10g-2xl` | `g5.2xlarge` | 8 | `(True, 2, 0.0, 0.25390625)` |

`g6e.xlarge` wins at every demand level, because 4 vCPU per GPU fits the one-CPU bundle more
tightly than 8 and idle vCPU drags the mean down. **Opening a candidate rung therefore cannot
quietly move a healthy fleet off the L40S** — the same property
`test_the_two_single_gpu_rungs_are_ranked_by_gpu_density_not_size` pins for `g6e.2xlarge`.

**But the two candidates score byte-identically**, and `get_nodes_for` sorts `(score,
node_type)` tuples with `reverse=True`, so the tie falls to the node-type NAME descending —
`…-l4-2xl` beats `…-a10g-2xl` because `l` sorts after `a`. Run against the real template:

| ladder | what Ray launches |
|---|---|
| `g6e.xlarge:400,g5.2xlarge:400,g6.2xlarge:400` | `g6e.xlarge` — correct |
| `g5.2xlarge:400,g6.2xlarge:400` (the drought case) | **`g6.2xlarge` — the worse card** |
| `g5.2xlarge:400` | `g5.2xlarge` — correct |

In exactly the situation the fallback exists for, Ray fills the fleet with L4s: 0.318 of an
L40S against the A10G's 0.459, and 1.65× the unit cost against 1.42×. **Operating rule: open
at most ONE candidate rung, and make it `g5.2xlarge`.** A ladder is authoritative over its
whole domain, so naming only `g6e.xlarge` and `g5.2xlarge` closes the L4 rungs by
construction — no code change needed.

**Pinned in code since 2026-08-27.** A guard forbidding both candidate rungs was written and then
REMOVED: `test_a_three_card_ladder_puts_all_three_arms_on_one_cluster` refutes the premise as
baldly stated, and it passes — two SMALL equal caps give both rungs work, because the cap binds
before the score does. Both readings are true in different shapes, and measurement settled which
applies where:

| ladder | what Ray launches |
|---|---|
| `g6.2xlarge:100,g5.2xlarge:100` | 8 L4, **zero** A10G — the score decides |
| `g6.2xlarge:3,g5.2xlarge:3` | 3 of each — the cap decides |
| `g5.2xlarge:100` | 8 A10G |

So the hazard is real but narrower than "never open both": **it bites when a fallback is opened
WIDE, which is the shape a fallback is actually opened in.** Hence a rule plus two tests
(`test_an_uncapped_pair_of_fallback_cards_goes_ENTIRELY_to_the_L4` and
`test_the_a10g_rung_alone_takes_the_whole_fallback_fleet`) rather than a guard — the uncapped case
is pinned so the rule's reason cannot silently stop being true.

## Verdict, unchanged in direction and now measured rather than inferred

**As a preferred type: no, for both.** The A10G's most favourable reading (0.526) is still
well under its 0.651 break-even; the L4's (0.342) is well under its 0.526. Not marginal.

**As a capacity fallback: the A10G yes, the L4 no** — because when `g6e` refuses, the
alternative is an idle fleet at infinite cost per token, and 1.42× beats that. Gated on two
things:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — without it both 22.4 GiB cards OOM at
   `t_s2` 208, and the campaign's deepest recorded `t_kept` of 206 presents as 208, i.e. **zero
   margin**. **Corrected in place: this said "not true today". It ships in this branch**, set on
   `InferenceActor`'s `runtime_env` so it is exported before torch builds the allocator (it is read
   once, at allocator construction). It is worth having with no fallback open at all, because on the
   L40S it also stops the reserved pool drifting to ~95% of the card to hold a <9 GB working set.
2. Only the A10G rung may be open — see the tie-break above.

**The failover is manual.** Ray's scorer ignores `node_availability_summary`, so a capacity
refusal never moves it off the top-scored rung; an operator must lower `g6e.xlarge`'s
`max_workers` for a fallback rung to be reached at all. The ladder makes that one SSM key, but
nothing here measured the latency of a human in that loop.
