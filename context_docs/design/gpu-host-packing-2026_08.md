# Does per-GPU throughput hold when four actors share one host? (2026-08-27)

**The question.** Widening the GPU instance pool beyond `g6e.xlarge` was settled by
reading, with one exception: every candidate multi-GPU size puts several
`InferenceActor`s on one machine, and **nothing in either repository had ever run two
actors on one host.** No document, log or test recorded it. This is the record of the
dev-account measurement that settled it, and of several things the plan had wrong that
the measurement found on the way.

Read `yield-embeddings/temp/gpu-instance-diversity-plan.md` for the case being tested.
This document is the evidence, including the corrections.

## What is being compared, and why it must be one run

`g6e.xlarge` gives one actor 4 vCPU. `g6e.12xlarge` gives four actors 12 vCPU each on
one 48-vCPU host. GPU, VRAM and PCIe are per-device and all bulk I/O goes to S3, so the
only shared resources are host CPU and the NIC. The question is whether sharing costs
per-GPU throughput.

Two runs cannot answer it. Per-actor tok/sec on this workload varies with geography,
chunk density, optical depth, time of day and S3 weather, and the effect being looked for
is on the order of 10%. So **one cluster, two instance types, one work queue**: the same
cell in the same minutes, with the confounders removed by construction rather than
adjusted for. `gpu-worker-ladder = g6e.xlarge:16,g6e.12xlarge:4` produces exactly that
shape, and the arithmetic was checked before any money was spent
(`test_the_experiments_ladder_yields_both_arms_on_one_cluster`).

**Stratify by radar status or do not bother.** Per-actor tok/sec is 1.26–1.62M on
both-orbit chunks and 2.26–2.93M on radar-free ones. A 2.3× spread buries a 10% effect.

**Quote combined tok/sec, never px/sec.** `t_kept × valid_px` is optical work only; a
radar-bearing chunk's forward pass also encodes one or two SAR sequences, worth ~1.3× (one
orbit) to ~2.0× (both) the per-chunk time at equal optical depth. px/sec additionally mixes
machine speed with geography and hides a 4× spread in observation depth — it put a cost
line 19% wrong once.

**Acceptance:** median per-actor combined tok/sec on the 4-GPU hosts ≥ 0.95× the 1-GPU
median, same cell, same run, same radar stratum, with the control arm's host-to-host
spread under 5%. A control arm runs first, because a ratio without a noise floor under it
cannot be judged.

## Subject

`iowa_epsg5070` in `global-tessera-dev`: 34,964 × 53,383 pixels, 263 optical timesteps and
103 Sentinel-1 ascending timesteps over calendar 2024, 1.2 TiB of reflectance. Ascending
only, so the cell is a single radar stratum — which makes the comparison cleaner and means
this measurement says nothing about the both-orbit stratum.

## Corrections the work produced

These are recorded because each one was believed otherwise, and three of them change a
decision.

### 1. `CHUNK_SUMMARY` did not carry `instance_id`

The plan said "read it from telemetry that already exists: `CHUNK_SUMMARY` carries
`instance_id`. Group by it." **It did not.** `instance_id` was in the dict `process_chunk`
returns to the scheduler, never on the log line. The line is attributed by CloudWatch log
stream, and one stream is one ECS flow runner — so the whole fleet's chunks share one
bucket and nothing said which worker did which chunk.

This was invisible for a real reason: while every host held exactly one actor, the host
axis WAS the actor axis, and Ray's log prefix carries `ip=`, which separates hosts. It
stops being invisible the moment a host holds four. `instance_id` and `gpu` are now on the
line, and `inference_profile.py --by-host` groups on them.

### 2. Ray prefers the 4-GPU box, but only from three unplaced actors up

The plan computed `g6e.xlarge` at 0.266 and `g6e.12xlarge` at 1.007 and concluded the
multi-GPU rung demotes `g6e.xlarge` to fallback. The conclusion is right at campaign scale
and the comparison was not: the two scores were computed against **different demands** —
one bundle for the `xlarge`, four for the `12xlarge`. Ray's scorer scores every candidate
against the *whole* outstanding demand list, so the answer depends on how many actors are
unplaced:

| unplaced bundles | `g6e.xlarge` | `g6e.2xlarge` | `g6e.12xlarge` | winner |
|---:|---:|---:|---:|---|
| 1 | 0.2656 | 0.2539 | 0.0157 | `g6e.xlarge` |
| 2 | 0.2656 | 0.2539 | 0.1259 | `g6e.xlarge` |
| 3 | 0.2656 | 0.2539 | 0.4248 | **`g6e.12xlarge`** |
| ≥4 | 0.2656 | 0.2539 | 1.0069 | **`g6e.12xlarge`** |

A campaign asks for hundreds, so it always gets the multi-GPU rung. But a one- or
two-actor smoke test lands on `g6e.xlarge` and looks like a contradiction. Pinned by
`TestRayNodeTypePreference::test_multi_gpu_rung_is_preferred_once_three_actors_are_unplaced`.

### 3. Releasing the `g6e.2xlarge` rung, on its own, buys nothing

This one changes the plan's sequencing. The plan's step 3 was "turn on the `g6e.2xlarge`
rung in prod — carries no unmeasured risk, only the ~30% premium." It carries no risk
because **it does nothing**:

```
ladder g6e.xlarge:500, g6e.2xlarge:500   demand 250  ->  {g6e.xlarge: 250}
ladder g6e.xlarge:100, g6e.2xlarge:500   demand 250  ->  {g6e.xlarge: 100, g6e.2xlarge: 150}
```

Two reasons compound. `g6e.2xlarge` scores *below* `g6e.xlarge` (0.2539 vs 0.2656) because
its extra vCPU sits idle under a one-CPU actor bundle and drags the utilisation mean down —
so it is a fallback, never a preference. And capacity failures have no feedback at all:
Ray's scorer takes a `node_availability_summary` and never reads it, so a fleet that cannot
buy `g6e.xlarge` goes on asking for `g6e.xlarge` for ever. **Lowering `g6e.xlarge`'s
`max_workers` is what releases the second pool**, and the ladder must name both rungs. The
plan's own example value, `g6e.xlarge:100,g6e.2xlarge:150`, happens to be exactly right;
its stated rationale was not.

### 4. The `_band_read_workers` fix is correct and arithmetically inert here

The plan called this the "one-line durable fix" for host CPU contention: `_band_read_workers`
sized from `os.sched_getaffinity(0)`, so on a 48-vCPU host every actor read 48 and "spawns
10 background readers instead of 2." The fix — size from the actor's share of the host, i.e.
host cores ÷ host GPUs — is right and has landed. **It changes nothing at any `g6e`
multi-GPU size**, because the band-count cap binds first. With the background reservation of
2:

```
naive (host):      min(10, 48 - 2) = 10
fair share (12):   min(10, 12 - 2) = 10
```

The smallest per-GPU share among the family's multi-GPU sizes is `g6e.12xlarge`'s 12, and
12 − 2 = 10 sits exactly on the cap. So four actors run 10 reader threads each — 40 reader
threads on 48 vCPU — before and after. The contention mechanism, if it is real, is not
reached by any fair-share formula; only a *lower* budget reduces the pool, which is what the
new `TESSERA_BAND_READ_CPUS` override supplies and what the third arm below measures.

Also worth recording: **sizing from Ray's assigned resources would have been a regression.**
`get_assigned_resources()` returns `{"CPU": 1.0, "GPU": 1.0}` for our actor — measured, not
assumed, because nothing sets `num_cpus` and Ray's actor default is 1. That would put ONE
decompression thread on a `g6e.xlarge` where four run today.

### 5. The card is 44.7 GiB, and the "46 GB" figure was a mislabelled MiB number

`inference_gpu_saturation_profile_2026_07.md:14` recorded the L40S as "46 GB VRAM";
`inference/README.md:135` said "48 GB". Both describe **45,776 MiB = 44.7 GiB = 48.0 GB
decimal**, confirmed against `ec2:describe-instance-types` in us-west-2. 46 was the MiB
figure rounded and relabelled. The README was right; the design doc has been corrected in
place.

It matters twice. It is the denominator of every "% of the card" figure, and it changes a
conclusion about the smaller cards: the L4 and A10G hold 22,888 MiB = 22.4 GiB, so this card
is **1.94×** them, not "barely half again". The campaign's only prior VRAM reading — 97% of
the card — is therefore ~43.4 GiB against 22.4 GiB available on a 24 GB-class card. For such
a card to be viable, that reading would have to be overstating the true requirement by more
than 2×; and it IS an overstatement, because nvidia-smi reports the caching allocator's
*reserved* pool rather than what live tensors needed. How large the overstatement is, is
what `max_memory_allocated` now measures.

### 6. There are no bigger GPUs in the family

Every `g6e` size — `xlarge` through `48xlarge` — carries the same L40S at 45,776 MiB. They
differ only in vCPU and host RAM per GPU, and in how many GPUs share a host. So "scale the
workload up to a bigger GPU" is not a lever that exists here; the only thing a wider rung
changes is how well fed the same GPU is, and our own ledger bounds that at 7–15% of
GPU-hours (`inference-perf-run-ledger.md:42`), against a GPU already at 89–93% utilisation
fleet-wide with SMACT ≈ 0.99. Going bigger on batch size is separately closed: B=7168 was
measured throughput-optimal at every sequence length and is on the do-not-re-litigate list.
Asserted by `TestEc2CatalogueArithmetic::test_every_g6e_size_carries_the_same_card`.

### 7. The quota table survives the vendor's own catalogue

vCPU per GPU, from `ec2:describe-instance-types` (us-west-2, 2026-08-27):

| type | vCPU | host GiB | GPUs | vCPU/GPU | network |
|---|---:|---:|---:|---:|---|
| `g6e.xlarge` | 4 | 32 | 1 | **4** | up to 20 Gb (burst credit) |
| `g6e.2xlarge` | 8 | 64 | 1 | **8** | up to 20 Gb (burst credit) |
| `g6e.12xlarge` | 48 | 384 | 4 | **12** | 100 Gb sustained |
| `g6e.4xlarge` | 16 | 128 | 1 | 16 | 20 Gb |
| `g6e.24xlarge` | 96 | 768 | 4 | 24 | 200 Gb |
| `g6e.48xlarge` | 192 | 1536 | 8 | 24 | 400 Gb |
| `g6e.8xlarge` | 32 | 256 | 1 | 32 | 25 Gb |
| `g6e.16xlarge` | 64 | 512 | 1 | 64 | 35 Gb |

`g6e.2xlarge` is the most quota-efficient rung after `g6e.xlarge`, ahead of every multi-GPU
size — so diversification strictly reduces the GPU count a fixed vCPU quota can hold.

The network reading runs **opposite** to the plan's worry. Four actors at ~950 MB/s each need
~30 Gbps; the packed host has 100 Gbps sustained, while the `g6e.xlarge` baseline's "up to
20 Gbps" is a burst credit. It is the *unpacked* arm whose network figure is the soft one.
(No per-GPU sustained baseline for the `xlarge` has been measured, and none is claimed.)

Host RAM per GPU clears the measured ~17.7 GB per-actor requirement on every rung shipped
(32 / 64 / 96 GiB) — the term that disqualifies the 16 GiB `g5`-class sizes that OOMed the
loader before inference in an earlier run.

## Instrumentation added before the measurement, not after

1. **Per-chunk peak VRAM.** `reset_peak_memory_stats()` at chunk start;
   `vram_peak_gib` (`max_memory_allocated`), `vram_reserved_peak_gib`
   (`max_memory_reserved`) and `vram_total_gib` on the `CHUNK_SUMMARY` line. Per chunk and
   not per process, because VRAM scales with optical depth and a maximum over shallow chunks
   is not the requirement.
2. **`instance_id` and `gpu` on the line** — see correction 1.
3. **`resource_monitor._get_gpu_stats` reported the wrong GPU.** It ran `nvidia-smi
   --query-gpu` with no `-i`, and nvidia-smi ignores `CUDA_VISIBLE_DEVICES`. On a 4-GPU host
   it returned four CSV rows and the code split the whole multi-line output on commas, so six
   field names were filled from the first row plus a newline-joined boundary: every actor
   logged GPU 0 with fields out of alignment. It now passes the actor's own accelerator id
   and refuses multi-row output rather than misattributing it.

## Arms

| arm | ladder | shape | purpose |
|---|---|---|---|
| **control** | none | 12 × `g6e.xlarge` | host-to-host spread = the noise floor |
| **mixed** | `g6e.xlarge:16,g6e.12xlarge:4` | 16 × 1-GPU + 4 × 4-GPU, one queue | the ratio, and a second noise-floor reading |
| **mixed, readers capped** | same + `TESSERA_BAND_READ_CPUS=4` | as above | whether reader-thread contention is real |

The third arm exists because of correction 4: the fair-share fix cannot change the reader
pool at these sizes, so the only way to test the CPU-contention mechanism is to force a
budget below the band-count cap. It is also the cheapest possible test of "scale the
workload per instance type": the delta between arms 2 and 3 is the measured value of that
idea on the one axis where it plausibly exists.

**No run is allowed to complete.** Each arm runs long enough for every actor to pay its
~36 s cold start once and give a stable median, then the cluster is torn down. A full
`iowa_epsg5070` pass is hours of GPU time and would buy nothing this question needs.

## Results

### One card, three different capacities — name the source

Before any VRAM figure, the denominator. The same L40S reports:

| source | figure | as GiB |
|---|---:|---:|
| `ec2:describe-instance-types` | 45,776 MiB | 44.70 |
| `nvidia-smi memory.total` (live worker) | 46,068 MiB | 45.00 |
| `torch.cuda.get_device_properties().total_memory` | — | **44.39** |

All three are correct for their own definition and none is 46 GB. **"46 GB" in the earlier
design note is `nvidia-smi`'s 46,068 MiB read as MB** — that is the provenance, now
identified rather than guessed. Every percentage below uses torch's 44.39 GiB, because that
is the number the same API produces the peak figures in.

### Peak VRAM — the first real measurement, and it moves a decision

From the control run on `g6e.xlarge` (L40S, 45,776 MiB = 44.39 GiB as torch reports it),
`iowa_epsg5070`, one-orbit chunks, B=7168:

| `t_kept` | `max_memory_allocated` | % of card | `max_memory_reserved` | % of card |
|---:|---:|---:|---:|---:|
| 54 | **4.55** GiB | 10% | 22.24 GiB | 50% |
| 54 | **5.31** GiB | 12% | 28.40 GiB | 64% |
| 57 | **4.55** GiB | 10% | 25.53 GiB | 58% |
| 62 | **5.29** GiB | 12% | 39.56 GiB | 89% |
| 93 | **5.29** GiB | 12% | 22.24 GiB | 50% |
| 108–110 | **7.52** GiB | 17% | 22.19–25.53 GiB | 50–58% |
| 113–115 | **7.52–8.27** GiB | 17–19% | 25.53–39.56 GiB | 58–89% |

**The live tensor requirement is 4.6–8.3 GiB — 10% to 19% of the card — not the ~43 GiB the
campaign's only prior reading implied.** The two columns behave completely differently, and
that difference is the whole answer:

* **Allocated** tracks optical depth cleanly and is quantised by the strip planner's buffer
  sizes (4.55 / 5.3 / 7.52 / 8.27). A straight fit over the range gives about
  **1.3 GiB fixed + 0.061 GiB per optical timestep**.
* **Reserved** does NOT track depth. It tracks the actor's AGE: a 62-timestep chunk late in
  a worker's life reserved 39.56 GiB while a 93-timestep chunk early in another's reserved
  22.24 GiB. It is the caching allocator's pool drifting upward with fragmentation, and it
  runs **2.7× to 7.5×** the live requirement.

**That is where "97% of the card" came from, now confirmed live.** A `RESOURCES` line from a
worker mid-chunk reads `VRAM=42171 MiB/46068 MiB` — 91.5%. `nvidia-smi` sees the reserved
pool, so the campaign's headline VRAM figure was measuring allocator slack, not need.

**A finding worth acting on independently of any instance-type decision.** The allocator is
drifting to ~89% of a 44 GiB card to hold a working set of 8 GiB. Nothing bounds it: a
chunk deeper than any yet seen could OOM a worker that has plenty of genuinely free VRAM.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, or a periodic `empty_cache()`, would
reclaim it. Untested here; the measurement is what makes it visible.

**What it settles for smaller cards.** A 22.4 GiB card (L4, A10G) is **not disqualified by
VRAM**. At the deepest depth observed (115) the requirement is 8.27 GiB — 37% of a 22.4 GiB
card. Extrapolated to the deepest a 2024 window could hold (263 optical timesteps) it is
**~17.3 GiB, or 77%** of such a card. Two honest caveats: that extrapolation rests on
depths up to 115 and wants deeper chunks before being quoted as a bound; and the reserved
pool's behaviour on a card that physically cannot over-reserve is **untested**. The
allocator would necessarily recycle rather than hoard, but whether it does so without
fragmentation-driven OOM at 77% occupancy is exactly the thing to measure before adopting a
24 GB-class card.

**Peak VRAM was identical per actor across hosts** (7.52 GiB on every deep chunk, on three
different instances), which is the expected answer — VRAM is per device and is not shared —
and is the control for the packed-host reading.

### Capacity: the `g6e` family is ONE pool, and that voids the plan's premise

Measured by real single launches in the dev account on 2026-08-27, ~17:14–17:19 UTC. (A
first attempt used `ec2 run-instances --dry-run`, which returns `DryRunOperation` after the
permission check and **never consults capacity** — a probe that would have reported every
type as available.)

| type | card | us-west-2a | us-west-2b | us-west-2c |
|---|---|---|---|---|
| `g6e.xlarge` … `g6e.48xlarge` (all 8 sizes) | L40S | refused | refused | refused |
| `g6.xlarge` | L4 | — | — | **launched** |
| `g6.2xlarge` | L4 | refused | **launched** | **launched** |
| `g6.12xlarge` | L4 | refused | refused | **launched at 17:14, refused at 17:19** |
| `g5.xlarge` | A10G | — | — | **launched** |
| `g5.12xlarge` | A10G | — | — | refused |
| `g4dn.xlarge` | T4 | — | — | **launched** |

Every refusal was `InsufficientInstanceCapacity`, not `VcpuLimitExceeded` — the applied
G-and-VT quota in this account is 10,000 vCPU, so quota is not the constraint.

**All eight `g6e` sizes refused together, in all three AZs, at the same moment, while three
other GPU families launched.** So a different `g6e` SIZE is not a different capacity pool.
**The pool is the card.** This is exactly the plan's own kill condition — *"if another `g6e`
size draws on the SAME constrained pool in us-west-2, the premise is void and the answer is
a capacity reservation or a second region"* — and it is now measured rather than assumed.

Two corollaries:

- **The 4-GPU boxes are the scarcest thing, not the easiest.** `g6.12xlarge` and
  `g6e.12xlarge` refused when their 1-GPU siblings launched. A 4-GPU host needs four
  contiguous GPUs on one machine, so a squeeze bites it first. That runs against the
  "fewer, bigger boxes" instinct as well as against diversification.
- **A withdrawn reading, and one open question left open.** For about ten minutes I read 50
  launch failures all naming `us-west-2b` as evidence that Ray's v2 autoscaler path does not
  rotate subnets. Then a worker appeared in `us-west-2a`, so **rotation does happen and that
  reading is withdrawn**; the AZ-failover behaviour the cluster template documents is intact,
  and what I was looking at was a region-wide shortage.

  What remains genuinely unexplained: across ~140 logged `InsufficientInstanceCapacity`
  failures, **every single one named `us-west-2b`** as the zone requested, and 2b is the
  first subnet in the SSM list. Ray's v1 `_create_node` rotates `subnet_idx` on each
  `ClientError` with `max_tries = max(BOTO_CREATE_MAX_RETRIES, len(subnet_ids))` ≥ 3, and the
  exception it reports is the last attempt's — so the reported zone should vary. It does not.
  I could not settle from logs whether later attempts happen and are unreported, or whether
  the v2 path's cached `launch_config` interacts badly with `_create_node`'s
  `conf.pop("SubnetIds")` (which mutates the dict it is given). **Not investigated further —
  it is outside this measurement's scope, and it did not block anything, because every zone
  was refusing anyway.** Worth a look if a future fleet stalls in one AZ while another has
  capacity.

### The packing ratio

**Not obtained on `g6e`.** The dev account could not buy a single 4-GPU host of either
family for the duration of the attempt. The control arm reached only 3 of 12 requested
`g6e.xlarge` workers in 20 minutes, against a fleet that normally fills in about two.

Because `g6.12xlarge` is 48 vCPU and 4 GPUs — **12 vCPU per actor, identical to
`g6e.12xlarge`** — the host-sharing mechanism can be tested on the L4 family instead, with
`g6.2xlarge` (8 vCPU, 32 GiB, one L4) as the 1-GPU arm. Both arms then run the same card
and the only difference is host sharing, which is what the question is about. The peak-VRAM
measurement above is what makes that substitution legitimate: 7.5 GiB fits a 22.4 GiB card.
Absolute tok/sec on an L4 is NOT comparable to production and no such comparison is made
here.

### Control-arm noise floor

Two `g6e.xlarge` hosts with 4–5 chunks each, one-orbit stratum, median per-host combined
tok/sec of **1.79M and 1.67M** — a 7% spread on very few chunks per host. Below the sample
count the acceptance threshold needs, and above the 5% the threshold assumes; both readings
are provisional.
