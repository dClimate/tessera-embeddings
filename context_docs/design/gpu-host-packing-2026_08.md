# Does per-GPU throughput hold when four actors share one host? (2026-08-27)

**The question.** Widening the GPU instance pool beyond `g6e.xlarge` was settled by
reading, with one exception: every candidate multi-GPU size puts several
`InferenceActor`s on one machine, and **nothing in either repository had ever run two
actors on one host.** No document, log or test recorded it.

**Status: the throughput ratio is NOT measured, and why is the most important finding
here.** Every `g6e` size was unbuyable in us-west-2 for the whole attempt — the pool is
the card, not the instance size — so the packed arm never launched. What the attempt did
produce is a peak-VRAM measurement that retires a two-year-old wrong number, real
per-GPU prices that kill the multi-GPU rung on cost before throughput is asked about, a
4.3% noise floor the ratio can be judged against when it is measured, and the first
confirmation that four actors DO co-exist on one host. Plus six corrections to the plan.

Read `yield-embeddings/temp/gpu-instance-diversity-plan.md` for the case being tested.
This document is the evidence, including the corrections and two claims of my own that
are withdrawn in place.

| deliverable | status |
|---|---|
| Peak VRAM per chunk, against optical depth | **measured** — 4.6–9.0 GiB, 10–20% of the card |
| Control-arm noise floor | **measured** — 4.3% over three hosts |
| Per-actor VRAM unaffected by host sharing | **measured** |
| Four actors on one host at all | **confirmed**, with correct per-actor GPU indices |
| $/GPU-hour for every candidate size | **measured** |
| `g6e` capacity across sizes and AZs | **measured** — one pool, all refused |
| **Packed-vs-unpacked throughput ratio** | **BLOCKED on `g6e` capacity** |
| Whether the `_band_read_workers` fix was needed to pass | **unanswerable until the ratio is** |

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

## Arms — planned, and what actually ran

| arm | ladder | intended shape | outcome |
|---|---|---|---|
| **A. control** | none | 12 × `g6e.xlarge` | **ran** — reached 3 of 12 workers in 20 min; gave the noise floor and the VRAM curve |
| **B. packed** | `g6e.xlarge:16,g6e.12xlarge:4` | 16 × 1-GPU + 4 × 4-GPU, one queue | **NOT RUN — no `g6e.12xlarge` capacity** |
| **C. readers capped** | B + `TESSERA_BAND_READ_CPUS=4` | as B | not reached |

Arm C exists because of correction 4: the fair-share fix cannot change the reader pool at
these sizes, so the only way to test the CPU-contention mechanism is to force a budget below
the band-count cap. It is also the cheapest test of "scale the workload per instance type" —
the delta between B and C is the measured value of that idea on the one axis where it
plausibly exists.

**No run was allowed to complete.** Each arm ran only long enough for its actors to pay the
~36 s cold start and give a median, then the cluster was torn down. A full `iowa_epsg5070`
pass is hours of GPU time and would buy nothing this question needs.

### A substitution that was tried and then RULED OUT

With no `g6e.12xlarge` obtainable, an all-L4 pair (`g6.12xlarge` against `g6.2xlarge`) was
dispatched on the reasoning that it holds the CARD constant within the comparison and so
isolates actors-per-host — the mechanism under test. **Robert ruled it out and it was torn
down**, on grounds that stand: the production fleet is L40S, so a ratio measured on L4 does
not transfer to `g6e.12xlarge`, which is the instance the plan actually recommends. A
substitution that changes the hardware also changes what a pass or a failure would license.

What that arm did produce is recorded below under "L4 findings", explicitly **not** as the
packing answer. It ran for about 12 minutes and cost roughly $3.

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

### Price per GPU-hour — the plan's "term most likely to change the recommendation"

From the AWS Pricing API, us-west-2, on-demand Linux shared tenancy, 2026-08-27. The plan
had exactly one price ($1.861) and one inherited relative premium ("~30%").

| type | $/hr | GPUs | **$/GPU-hr** | vs `g6e.xlarge` | vCPU/GPU |
|---|---:|---:|---:|---:|---:|
| `g6e.xlarge` | 1.861 | 1 | **1.861** | — | 4 |
| `g6e.2xlarge` | 2.242 | 1 | **2.242** | **+20.5%** | 8 |
| `g6e.12xlarge` | 10.493 | 4 | **2.623** | **+40.9%** | 12 |
| `g6e.24xlarge` | 15.066 | 4 | **3.766** | +102% | 24 |
| `g6e.48xlarge` | 30.131 | 8 | **3.766** | +102% | 24 |
| `g6.2xlarge` | 0.978 | 1 | **0.978** | **−47.5%** | 8 |
| `g6.12xlarge` | 4.602 | 4 | **1.150** | **−38.2%** | 12 |

Three things follow, and two of them are decisions.

**1. `g6e.2xlarge` is cheaper than assumed: +20.5%, not ~30%.** Against 7–15% recoverable
feed that is a net +5% to +13% on hours that land there, not +13% to +21%.

**2. The multi-GPU L40S rung fails the plan's own test.** The plan wrote: *"If
`g6e.12xlarge` carries a per-GPU premium much above `g6e.2xlarge`'s ~30%, the multi-GPU rung
is not worth the unmeasured risk and the plan collapses to '`g6e.2xlarge` only'."* It is
**+40.9% against +20.5% — double the premium** — and it is simultaneously the worse option
on quota efficiency (12 vCPU/GPU against 8). So `g6e.12xlarge` is dominated by
`g6e.2xlarge` on price AND on quota, before any throughput question is asked. **Even a
packing ratio of exactly 1.000 would not make it the better rung.**

**3. The L4 is 38–47% cheaper per GPU-hour.** That turns the L4 from a substitute-hardware
convenience into a live production candidate: if an L4 delivers even ~62% of an L40S's
throughput, `g6.12xlarge` is cost-neutral, and `g6.2xlarge` breaks even at ~53%. Combined
with the capacity finding above — the L4 pool was buyable when the entire L40S family was
not — **the throughput of the Tessera model on an L4 is now the most valuable unmeasured
number in this area.** It is NOT measured here: this exercise held the card constant within
each comparison precisely so the packing result would not be confounded, and no L4-vs-L40S
throughput claim is made.

### Four actors on one host: it works, and the GPU-index fix is what makes it readable

**This one does transfer**, because it is about Ray placement and this repository's own
instrumentation, not about the card. It was observed on `g6.12xlarge`; the placement,
the per-actor GPU assignment and the logging are identical on `g6e.12xlarge`.

Confirmed at 17:37 UTC on 2026-08-27, from the actors' own ready lines:

```
2 instance i-02a0fcb4aa551767c (GPU 0)   \
2 instance i-02a0fcb4aa551767c (GPU 1)    |  g6.12xlarge — FOUR actors, one host
2 instance i-02a0fcb4aa551767c (GPU 2)    |
2 instance i-02a0fcb4aa551767c (GPU 3)   /
2 instance i-0d262da5050224468 (GPU 0..3)   the second g6.12xlarge
2 instance i-03c3640a65e503ee3 (GPU 0)   \
2 instance i-0cab65709bcdc719f (GPU 0)    |  four g6.2xlarge — one actor each
2 instance i-0dc2a0170c4076608 (GPU 0)    |
2 instance i-0ed83ea5656064e5a (GPU 0)   /
```

**This is the first time two `InferenceActor`s have run on one host in this codebase**, which
answers the plan's open question 5 directly: nobody had, and now it works — twelve actors
placed across six hosts, no OOM, no crash, chunks in flight on all of them. (The `2 ` prefix
is the double-logging of every event, deduped elsewhere.)

It also demonstrates the accelerator-id fix in the condition it was written for. Each of the
four actors on a packed host reports its OWN GPU index, 0 through 3. Before the fix all four
would have logged GPU 0 with `nvidia-smi` fields sliding out of alignment, and no reader
could have told them apart.

It also gave the first per-actor CPU reading on a packed host. From the `RESOURCES` lines
mid-chunk: the two 4-GPU hosts ran **load 11.35 and 12.40 on 48 vCPU** (24–26% of the box)
with **all four GPUs at 100% utilisation**, while the 1-GPU hosts ran load 1.04–1.27 on
8 vCPU. So a packed host was not CPU-saturated and its GPUs were not idling — but load per
actor was ~2.9 against ~1.1, which is the `_band_read_workers` asymmetry doing exactly what
correction 4 describes (10 readers per actor at a 12-vCPU share, 6 at an 8-vCPU share).
**This is suggestive and is not the ratio**; utilisation at 100% is consistent with both a
fed GPU and a GPU spending time on smaller kernels.

### Per-actor VRAM is unaffected by host sharing

Also transferable, because VRAM is a per-device resource and the mechanism is
card-independent. On a packed host an actor reported `vram_peak_gib = 5.31` at `t_kept = 54`;
single-GPU hosts at the same depth reported 4.55 and 5.31. The variation is inside what one
card shows across chunks, so **four actors sharing a host do not affect each other's VRAM** —
which is the expected answer, and worth having as a control rather than an assumption. Had it
moved, that would have been a finding in itself and a reason to stop.

The reserved pool behaves the same way per actor: 19.7 of 22.04 GiB on the packed host's
actor, against 21.7 on the singles. Each actor's allocator sizes itself to its own card.

### L4 findings — a separate question, not the packing answer

**Read this as capacity-and-cost evidence about a different card, nothing more.** It cannot
speak to `g6e.12xlarge` packing, for the reason in the ruling above.

The substitution produced a number nobody had: **the Tessera model runs on a 22.4 GiB L4**,
no OOM, at B=7168. Peak allocated is **7.52 GiB — identical to the L40S on the same chunk**,
which is the expected answer (the working set is a property of the work, not the card) and is
the control that says the VRAM figures above are real.

The reserved pool, though, went straight to **21.71 of 22.04 GiB — 98.5% of the card**. That
is the allocator doing exactly what it does on the L40S (reserve against what is available)
with no room left over. The model works; the fragmentation headroom is gone. If a 24 GB-class
card is ever adopted, `PYTORCH_CUDA_ALLOC_CONF` is not optional.

And because both runs processed the same cell, two chunks came out **exactly matched** — same
label, same `t_kept`, same `valid_px`, so same geography and same depth:

| chunk | `t_kept` | L40S `total_s` | L4 `total_s` | ratio | L40S `infer_s` | L4 `infer_s` | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `chunk_0_0` | 108 | 148.0 | 330.9 | **2.24×** | 101.1 | 299.0 | **2.96×** |
| `chunk_0_6` | 57 | 115.1 | 351.6 | **3.05×** | 86.7 | 324.9 | **3.75×** |

**The L4 is 2.2–3.1× slower per chunk at 0.525× the price per GPU-hour, so it costs 1.18× to
1.60× as much per unit of work.** The door the capacity finding appeared to open is therefore
mostly shut again — not by VRAM, which is fine, but by throughput. What the L4 offers is
capacity that can actually be bought, at an 18–60% cost premium.

Two honest limits on this. **n = 2**, though exactly-matched pairs are worth far more than two
medians would be. And the L4 arm ran with six hosts pulling from S3 concurrently against the
control's three, so some of the gap could be read contention — but a 2–3× gap is far too
large for that to be the mechanism, and the `infer_s` ratios (2.96×, 3.75×) are larger than
the `total_s` ratios, which points at the GPU rather than at I/O.

### The packing ratio: NOT MEASURED

**The number the plan asked for does not exist yet, and the reason is the finding.**

`g6e.12xlarge` could not be bought in the dev account at any point during the attempt.
**Four probe rounds spanning 17:14 to 17:49 UTC**, each covering all three AZs, refused every
`g6e` size — and from the second round on, `g6e.xlarge` and `g6e.2xlarge` refused too, so
**no `g6e` experiment of any kind was runnable**, including the 1-GPU CPU-feed comparison
that would otherwise be the fallback. The control arm's own fleet is the same evidence from
the other side: it reached 3 of 12 requested `g6e.xlarge` workers in 20 minutes, against a
fleet that normally fills in about two.

What remains to run, unchanged, the moment `g6e` capacity returns:

1. `g6e.xlarge` control arm for the noise floor — **already done at 4.3%**, and reusable.
2. `g6e.xlarge` + `g6e.12xlarge` on one queue via `gpu-worker-ladder`. The mechanism is
   proven: with the L4 pair named, one SSM key made the autoscaler ask for exactly
   `4 × gpu-workers-ondemand-l4-12xl` and `8 × gpu-workers-ondemand-l4-2xl`, with no release
   and no re-registration. The key is already set to the `g6e` shape
   (`g6e.xlarge:8,g6e.12xlarge:2`); only the launches were refused, not the configuration.
3. The same run with `TESSERA_BAND_READ_CPUS=4` on the packed arm.

Everything needed for it is shipped and verified except the hardware.

### Control-arm noise floor: 4.3%

Three `g6e.xlarge` hosts, 4–7 chunks each, one-orbit stratum, 18 successful chunks over
`t_kept` 54–123. Median per-host combined tok/sec:

| instance | chunks | median combined tok/sec |
|---|---:|---:|
| `i-03ccb860224f06194` | 7 | 1.829M |
| `i-0d67a26298df43c3d` | 4 | 1.793M |
| `i-0687d74c32bf43230` | 7 | 1.752M |

**Host-to-host spread 4.3%**, against the acceptance criterion's requirement of under 5%. So
the threshold has a floor under it, and a packed-host reading below 0.95 would be outside
the noise.

One methodological note that cost a wrong number before it was caught: an earlier reading of
the same arm gave 7.1% over two hosts, and 26.7% once a host with a single chunk was
included. **A host's first chunk carries its ~36 s cold start**, so a barely-started host
reads slow and one that drew a shallow chunk reads fast. `inference_profile.py --by-host` now
excludes hosts below three chunks and says that it has.

## What this changes about the plan

The plan's shape survives; three of its steps do not.

**1. `g6e.12xlarge` is dominated before the packing question is asked.** At **+40.9%** per
GPU-hour against `g6e.2xlarge`'s **+20.5%**, and at 12 vCPU/GPU against 8, it is worse on
both price and quota. The plan's own criterion — *"if `g6e.12xlarge` carries a per-GPU premium
much above `g6e.2xlarge`'s ~30%, the multi-GPU rung is not worth the unmeasured risk"* — is
met. Even a packing ratio of exactly 1.000 would not make it the better rung. **Measuring
the ratio is still worth doing**, because it is the only thing standing between us and ever
using a multi-GPU host, and because the 4-GPU sizes are the only way to add GPUs without
adding node count to the Ray head. But it is no longer a step on the critical path to more
capacity.

**2. "Turn on the `g6e.2xlarge` rung in prod" does nothing on its own.** Ray's scorer never
reads capacity availability, and `g6e.2xlarge` scores *below* `g6e.xlarge`, so an unrestricted
`g6e.xlarge` rung absorbs all demand. The step must be "set
`gpu-worker-ladder = g6e.xlarge:<N>,g6e.2xlarge:<M>` with N *below* today's ceiling."

**3. The premise that diversification buys capacity is void within `g6e`.** All eight sizes
share one pool. So the real options for capacity are, in order of what the evidence supports:

- **An on-demand capacity reservation on `g6e.xlarge`.** The plan already names this as
  co-equal and it now looks strictly better: it preserves 4 vCPU/GPU, every measured
  invariant, and the single-price cost basis, and it is the only option that addresses a
  card-level shortage. **Price it first.**
- **A second region.** Untouched by any of this work and the only true pool diversification.
- **A different card.** `g6`/`g5` are buyable and are no longer disqualified by VRAM — that
  objection is measured away. They are disqualified on throughput instead: the L4 is 2.2–3.1×
  slower at 0.525× the price, so ~1.18–1.60× the cost per unit work. Whether that trade is
  worth taking depends on how much a GPU we cannot rent is costing, which is a question about
  the campaign's schedule rather than about hardware.
- **`g6e.2xlarge`** remains worth having as a rung — but it buys a better-fed GPU at +20.5%,
  not capacity.

**4. The quota ask.** The pending 16,000 vCPU request was justified on instance-type
diversity. Nothing here supports that justification: within `g6e`, diversity does not reach a
different pool, and the sizes that would consume the extra quota are the least efficient per
GPU. If the ask is still wanted, the reason has to be eviction headroom or a genuinely mixed
fleet spanning card families — and it should be stated as whichever it actually is.

## To finish this: one command, when capacity returns

Everything is shipped and verified except the hardware. The SSM ladder is already set to the
minimal shape, and the deployment is already registered, so the `g6e` run is a single
dispatch:

```bash
# ladder already set: g6e.xlarge:8,g6e.12xlarge:2  (8 + 2x4 = 16 actors)
aws ssm get-parameter --profile global-tessera-dev --region us-west-2 \
  --name /global-tessera-dev/ray/gpu-worker-ladder --query Parameter.Value --output text

python scripts/run_campaign_cell.py --deployment global-tessera-dev \
  --flow tessera-embeddings --branch global-tessera-gpu-packing \
  --params-json params_g6e.json --run-name gpupack-mixed-g6e
```

with `params_g6e.json` carrying `roi_name=iowa_epsg5070`, `time_window_end="December 2024"`,
`num_actors=16`, `s1_orbit=ascending`, `require_s1=false`, `allow_s2_only=true`,
`dev_params={"skip_coverage_check": true}` and the branch-scoped `paths`. Then, once every
actor has 3+ chunks:

```bash
python scripts/inference_profile.py --deployment global-tessera-dev --hours 2 \
  --by-host --run-id <the run_id from the flow's first log line>
```

which prints the per-instance medians, the host spread within each type, and the
`g6e.12xlarge / g6e.xlarge` ratio against the 0.95 floor. Repeat with
`TESSERA_BAND_READ_CPUS=4` on the packed arm for arm C.

**Reuse arm A's noise floor (4.3%) rather than re-running it** — it is a property of the
`g6e.xlarge` population on this cell, and it is already measured.

## Spend

| item | shape | approx cost |
|---|---|---|
| Arm A (`g6e.xlarge` control) | 3 GPU workers + head, ~33 min | ~$3.0 |
| Ruled-out L4 arm | 6 × `g6.2xlarge` + 2 × `g6.12xlarge` + head, ~12 min | ~$3.2 |
| Capacity probes | ~25 single launches, most refused; successes terminated at once | ~$0.6 |
| **total** | | **~$7** |

No cluster was left idle: the control arm was torn down the moment its noise floor and VRAM
curve were in hand, the L4 arm within four minutes of the ruling, and the account was verified
at zero GPU instances afterwards. The `g6e` re-run is **blocked on capacity, holding nothing**
— a probe loop watches for `g6e` to return rather than a warm cluster waiting for it.
