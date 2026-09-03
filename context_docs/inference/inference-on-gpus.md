# Running the encoder on GPUs: throughput, cards, batch size, and what it costs

**The empirical record for the inference half of the pipeline.** What the GPU is actually doing,
what each optimisation bought, which cards the campaign may rent and why, how the batch is sized to
whichever card turns up, and what all of that measured on real campaign work rather than on a
reference region.

**Audience: anyone who wants to understand this, including people who have never worked on this
pipeline.** Every term is defined where it first appears, and the code locations are collected in an
appendix so the explanation does not depend on them.

Four measurement campaigns are folded together here, in the order a reader needs them:

| § | what it answers | when |
|---|---|---|
| 2 | what the July optimisation work changed, and what each change bought | 2026-07 |
| 3 | which levers were measured and abandoned — do not re-litigate these | 2026-07 |
| 4 | which GPU the campaign may rent when its preferred one is unavailable | 2026-08-27 |
| 5 | why the batch is computed from the card rather than fixed | 2026-08-29 |
| 6 | what the campaign path itself measured, which is what the cost model divides by | 2026-08 |

Cost figures are results here, with pointers. [`../campaign/campaign-cost-model.md`](../campaign/campaign-cost-model.md)
is the source of truth for every rate, fleet size and dollar figure; where the two disagree, the
cost model is right.

---

## 1. What the machine is actually doing

Five terms carry the whole argument.

**Pixel.** One ground location, 10 m square, with one year of satellite history attached to it. The
network's job is to turn that history into a short numeric summary — a 128-dimensional vector.

**Observation, and depth.** One usable satellite pass over that pixel. There are two independent
streams — **optical** (Sentinel-2, blocked by cloud) and **radar** (Sentinel-1, not blocked by
cloud) — and each pixel has its own count of each. A pixel's **depth** is how many observations it
carries. Depth varies enormously across the globe: a cloudy tropical pixel may have 57 usable
optical passes in a year and a clear desert pixel 206.

**Token.** The network reads each observation the way a language model reads a word: as one **token**
in a sequence. A pixel with 184 optical and 120 radar observations goes through the network as a
304-token sequence. **Tokens per pixel** is therefore optical depth plus radar depth, and it is the
unit that matters both for memory and for cost.

**Bucket.** A graphics card wants rectangular work — every sequence in one call the same length.
Pixels do not oblige, so the pipeline sorts them into **buckets** by depth. All the pixels in one
bucket are processed at the same sequence length; different buckets have different lengths. The
allowed lengths are a fixed list called the **checkpoint ladder** (`num_obs_checkpoints`), which by
default is every multiple of 8 from 8 up to **256**. A pixel is rounded up to the next rung, and —
the load-bearing part — anything deeper than the top rung is **clipped** to it. That is why an
optical depth of 206 is processed as 208, and why 300 would also be processed as 256.

**Sub-batch, and `batch_size`.** A bucket can hold millions of pixels, far more than a card can hold
at once, so it is processed in slices. One slice is a **sub-batch**, and `batch_size` is how many
pixels are in it. Its calibrated value is **7,168 pixels**.

One more piece of vocabulary, for the machinery rather than the model. An **actor** is one worker
process holding one copy of the network on one card; the pipeline runs hundreds of them at once
through the Ray framework. A **fill** is one unit of campaign work — one UTM zone for one year. A
**chunk** is a tile of pixels within a fill; a chunk that fails is retried, up to three attempts,
after which it is recorded `PERMANENTLY FAILED`.

### The memory rule, in words

When the card runs one sub-batch it must hold the intermediate results for every token in it — the
**activations**. There are `batch_size` pixels in the slice, each carrying up to its bucket's
tokens, so:

> **Memory needed ≈ (pixels in the slice) × (tokens each pixel carries).**

Both factors matter and only one of them is ours. `batch_size` is a setting. Tokens per pixel is
decided by the weather and the satellite's orbit over whatever piece of ground the fill happens to
cover. **That is the whole problem** behind §5: a batch size chosen once, on one card, against one
depth, is a bet that neither the card nor the depth will change.

The network's own weights are not the issue: the loaded checkpoint accounts for about **0.2 GiB** of
the 20.9 GiB that was in use when a card ran out. Essentially all of the memory is activations.

---

## 2. The saturation campaign — July 2026

Authoritative record of the inference optimization work behind PR #85 (branch
`perf/inference-rethink`): what was changed, how much each change bought, how correctness was
preserved. This is the empirical basis for why the pipeline looks the way it does.

Hardware: `g6e.xlarge` — 1× L40S (181 TFLOPS BF16 dense, 46 GB VRAM, 864 GB/s), 4 vCPU, 30.9 GB
usable RAM, 250 GB NVMe. Iowa reference region, 2000×2000 chunks. Raw telemetry (`TIMING` /
`RESOURCES` / `EFFECTIVE-TFLOPS` lines) survives in CloudWatch (`/ec2/yield-embeddings/ray`);
reproduce fleet stats with `te-observe-cluster` (`--report`, `--ram-report`).

### Headline numbers, final shipped state

The phase-5 run `a60550ae` (bounded cross-chunk starter prefetch — the last optimization on the
branch) against the batch-3584 baseline `a85be572e2fb` on `main`. Both single runs.

| metric | `main` baseline (batch 3584) | shipped (branch, batch 7168) |
|---|---|---|
| per-worker throughput | 1× | **~2–2.8×** |
| GPU utilization (fleet) | 48–72% avg; SMACT 0.31–0.68; TENSO 0.12–0.26 | **~89–93%; SMACT ≈0.99; TENSO 0.42–0.47** |
| px/s per worker, per-chunk-class | 9.6–13.3K | **21–24K** mid-density / 10–18K dense (rate while processing a chunk of that class; bit-identical forwards, so unchanged since striping) |
| px/s per worker, fleet-overall | — (not separately derived) | **~13–15K** (1.87B-px region ÷ ~34 GPU-hrs on `a60550ae`; includes cold starts, density mix, ramp — the capacity-planning number) |
| GPU-idle overhead per chunk | ~50–60 s (22–25% of wall) | **~6 s** median on prefetch-hit chunks; ~36 s on the unavoidable first-per-worker cold start |
| CPU batch-prep per sub-batch | 165 ms @3584 (651 ms @7168 unvectorised) | **103–160 ms** |
| peak host RAM | ~**50%** (estimated; not directly instrumented) | **~52%** (16.1 GB / 30.9 GB) |

The campaign originally measured ~2.5–3.5× at 100% GPU utilisation **with** a full cross-chunk
prologue prefetch — but that co-resided two chunks' whole working sets and OOM-killed a worker at
92–95% RAM. It was **removed** for UTM-zone-scale safety (target peak RAM ≤60%, ideally ~50%) and
replaced by the RAM-bounded striping and starter-prefetch design below; the ~1.25–1.3× interleaving
factor was deliberately traded for headroom, then partially recovered by the bounded prefetch. The
numbers above are the current, RAM-safe config.

```
 per-chunk wall anatomy (single-strip chunk), GPU busy = ▓, idle = ░

 main:        [mask][bands][build][▓▓▓▓ inference ▓▓▓▓][write]
              ░░░░░░░░░░░░░░░░░░░░  ~50-60 s idle       ~7.5 s idle

 striping:    [mask][b][build][▓▓▓▓ inference ▓▓▓▓]     write runs in background,
              ░░ ~24-34 s ░░                            overlapping next prologue

 +prefetch:   [▓▓▓▓ inference N ▓▓▓▓][▓ N+1 starter ▓][▓▓ N+1 body ▓▓]
              next chunk's mask+starter preloaded during N's last strip
              (RAM trough) → GPU idle between chunks ~6 s on a hit
```

### The optimizations, and what each bought

Five changes shipped. The per-change working — method, A/B arms, attribution table and the
correctness argument for each — is compressed here because all five are in the code and their
combined effect is the headline above. What does not live in the code is which lever was worth
pulling and why, so that is what stays:

| lever | what it bought | note |
|---|---|---|
| keep the GPU fed inside the forward loop | the largest single win | retained, bit-identical |
| hide the per-chunk prologue idle | the second | **the RAM-bounded one** — it trades host memory for idle, so it is the change that has a ceiling |
| cheaper spatially-sparse / edge chunks | modest | bit-identical |
| background staging write | modest | bit-identical |
| cost and reliability changes | none on output | |

**Four of the five are bit-identical**, which is the property that made them safe to ship together:
correctness was pinned by comparing outputs rather than by reasoning about the change. The one that
is not — the prologue-idle hiding — is bounded by host RAM, and that bound is why it has a ceiling
rather than a knob.

### The phase-0 baseline the campaign started from

2026-07-16, Iowa region, `main` at `batch_size=3584`, 4 × `g6e.xlarge`. Fleet GPU polls over ~10.5
min steady state, all four workers within a few percent:

| metric | value |
|---|---|
| avg GPU util (nvidia-smi) | 48–57% |
| busy fraction (util > 5%) | ~0.70 |
| avg power | 219–248 W / 350 W |
| DCGM SMACT during inference | 0.31–0.68 |
| DCGM TENSO (tensor pipes) during inference | **0.12–0.26** |
| DCGM DRAMA | 0.25–0.49 |

Per-chunk phase splits (12 chunks, all single-strip, 40–70% valid):

| phase | wall | GPU |
|---|---|---|
| scheduler gap + SCL mask load | ~3 s | idle |
| S2 band read (5–8 GB) | 20–35 s | idle |
| SAR read + dataset build | 9–17 s | idle |
| inference (1.6–2.8 M px) | 137–221 s @ 9.6–13.3K px/s | oscillating 21–100% |
| staging write | 7–8 s | idle |
| **total** | **204–281 s** | **~22–25% fully idle** |

Reading: two independent losses. **Structural** — ~50–60 s of GPU-idle prologue and epilogue per
chunk with no cross-chunk overlap. **Within-inference** — tensor pipes ≤26% active even mid-forward:
CPU batch prep gating (~165 ms per sub-batch), serial per-sub-batch host-to-device → forward →
device-to-host bubbles, and the launch-bound GRU loop. Note that production workers have only **4
vCPU** feeding an L40S whose tensor ceiling is ~181 TFLOPS, which is why the CPU-side and pipeline
bottlenecks dominated.

### How comparisons between runs are made honest

A staged tile is only reusable, and two runs only comparable, if something identifies the code that
made it. This was once a hand-maintained ledger mapping staging run ids to the code and config that
produced them; **that table is the argument for the automatic mechanism that replaced it** — a table
somebody has to update is a table that is wrong the first time somebody forgets. The mechanism is
`_staging_run_id`, which fingerprints inputs, config and inference-source identity, so a resume can
only ever reuse tiles produced by the same work
([`../storage/staging-identity-and-resume.md`](../storage/staging-identity-and-resume.md)).

The comparison rule it enforces is [ADR-012](../decisions/012-validated-equivalence-for-inference-outputs.md)'s:
**two runs are comparable only on identical inputs, identical geometry and identical code.** The
ledger's whole structure was that triple; the equivalence gate enforces it now rather than relying
on a reader checking a table. Two properties of that gate are worth carrying:

* **Output is NOT bit-exact across a batch-size change.** The batch-size change shimmers int8 by
  ±1–2 levels through cuBLAS. Compare with `compare_outputs.py --cross-config`, never an exact
  re-diff. Bit-exactness holds only at the same batch and config — and, measured, the phase-2 and
  phase-3 builds against each other at one batch were **100.000000% bit-identical**, max |Δ| = 0.
* **Few-valid-pixel chunks are the coverage that matters.** Cross-config checks on chunks with 3.5%
  valid pixels returned exact int8 agreement of 99.85%, max |Δ| = 2, cosine ≥ 0.99992 and **zero
  NaN-mask mismatches** — which is what confirms validity and skip semantics are unchanged under the
  prefetched-prologue path. A dense chunk cannot show that.

Per-run figures from that ledger are in git history. Anything quoted from them in a live document
carries its own provenance line, which is what the corrections register requires.

---

## 3. Dead ends — measured, and not to be re-litigated

- **FP16 fast-accumulate.** L40S GEMM microbenchmark: BF16 is already at the full dense ceiling (189
  against a 181 TFLOPS datasheet figure), so FP16 buys nothing. The GA10x half-rate penalty that
  makes FP16 attractive elsewhere is A10G-specific. BF16 stays.
- **Adaptive token-budget batching.** Real-model forward sweep: `B = 7,168` is throughput-optimal at
  *every* sequence length, down to 8 timesteps, and larger B is neutral-to-worse. Shelved — and see
  §5, alternative A, where this is what closes the case for a per-bucket memory budget.
- **Eager bucketing (P4).** Opens `dataset.py`; post-striping payoff is a sliver. Rejected.
- **GRU restructure.** The builder already fuses `CustomGRU` into cuDNN's `nn.GRU`; the restructure
  never reached production and was reverted as dead code.
- **`g6e.2xlarge` (8 vCPU).** A ~30% premium for ~7–15% of feed-recoverable time. The software route
  was preferred.

---

## 4. Which GPU, and whether a different one is worth its availability

**The question, 2026-08-27.** Widening the GPU pool by instance *size* is dead: every `g6e` size is
the same L40S drawing on one us-west-2 capacity pool, and all eight refused together. A capacity
reservation is not available. So the only remaining lever is a different **card** — and peak VRAM
turned out to be 4.6–8.3 GiB, not the ~43 GiB an earlier reading implied, which is what makes a
22.4 GiB card arguable at all.

Two cards are buyable when the L40S is not: the **L4** (`g6.*`) and the **A10G** (`g5.*`). They have
identical VRAM and are ordered **oppositely** on bandwidth and tensor compute, so running them side
by side tests a hypothesis rather than collecting a second price quote.

**Answer: no, on both cost and crash safety — and the hypothesis behind the question is confirmed in
its ordering and refuted in its magnitude.**

| deliverable | status |
|---|---|
| Vendor bandwidth figures (864 / 600 / 300 GB/s) | **verified** — all three exactly right |
| A10G vs L4 forward-pass throughput, identical shapes | **measured** — A10G 1.29–1.42× the L4 |
| Whether the ranking follows bandwidth or architecture | **bandwidth** — the older card wins |
| Whether bandwidth alone explains the magnitude | **no** — 1.41× delivered against 2.0× predicted. Quoted bandwidth is the wrong predictor; DRAMA × bandwidth (achieved traffic) fits both candidates |
| Deepest sequence a 22.4 GiB card can run | **measured** — OOM at 208; 232 with an allocator flag |
| Crash verdict per card | **DISQUALIFYING as configured** — see below |
| $ per unit of work | **measured** — both candidates cost more per unit than an L40S |
| End-to-end per-host tok/sec on one queue | **measured** — all three cards, 33 chunks |
| DCGM SMACT / TENSO / DRAMA per card | **measured** — including the L40S in the same run |

### The two hypotheses, and why the A10G is the only card that separates them

Fleet telemetry says the L40S runs at SMACT ≈ 0.99 with TENSO only 0.42–0.47 and effective TFLOPS
flat at 85 — streaming multiprocessors always occupied, tensor pipes under half engaged. That reads
as memory-bandwidth bound rather than compute bound. If it is right, the ranking of cards follows
bandwidth and not architecture generation.

Vendor figures, **verified from the vendors' own documents** because AWS publishes no GPU bandwidth
at all (`ec2:describe-instance-types` returns GPU name, count and VRAM size only):

| card | VRAM | memory bandwidth | BF16 dense tensor, as quoted | source |
|---|---:|---:|---:|---|
| L40S | 45,776 MiB | **864 GB/s** | 362 TFLOPS (733 with sparsity) | nvidia.com/en-us/data-center/l40s/ |
| A10G | 22,888 MiB | **600 GB/s** | 70 TFLOPS (140 with sparsity) | NVIDIA/AWS *A10G Tensor Core GPU* datasheet, Feb 2022 |
| L4 | 22,888 MiB | **300 GB/s** | 121 TFLOPS (242 with sparsity) | nvidia.com/en-us/data-center/l4/ |

**Why the L4 alone could settle nothing.** Against an L40S, bandwidth predicts 300/864 = 0.35 and
compute predicts 121/362 = 0.33. Indistinguishable, so an L4-versus-L40S figure is consistent with
both readings and licenses neither.

**Why the A10G separates them.** It has **twice** the L4's bandwidth on **0.58×** its tensor
compute. Bandwidth says the A10G beats the L4 by 2.0×; compute says the L4 beats the A10G by 1.7×.
There is no reading on which both are true.

### Method

Three instruments, because no single one answers both halves of the question.

1. **A forward-pass sweep on synthetic tensors** (`profiling/inference/forward_bench.py`), sequence
   length 8 → 256 at the production `B = 7,168` and bf16. Same tensors, same shapes, same dtype on
   every card: no S3 weather, no host feed, no geography, no optical-depth spread. This is the clean
   card comparison, and it is the ONLY way to reach the model's own 256-timestep ceiling — the Iowa
   region tops out near 123 and the deepest depth ever recorded anywhere in the campaign is **206**,
   so no cell can exercise the worst case. A fleet run can only ever report "it did not OOM on the
   chunks we drew".
2. **One cluster, three arms, one work queue** — `gpu-worker-ladder =
   g6e.xlarge:3,g6.2xlarge:3,g5.2xlarge:3` on the Iowa region, ascending-only so the cell is a
   single radar stratum. Same cell, same minutes, confounders removed by construction. The `g6e` arm
   was included precisely because it costs nothing when refused, and it was refused.
3. **Exactly-matched chunk pairs across runs** — the same chunk label at the same depth and the same
   valid-pixel count, compared across the day's three dev runs. Worth far more than two medians when
   the per-chunk spread is 2–3×: geography, depth and radar content are identical by construction.
   The join is keyed on `vram_total_gib`, because the earlier runs' instances are already terminated
   and drop out of `ec2:describe-instances`.

`g5.2xlarge` and `g6.2xlarge` are the shapes to measure on: 8 vCPU and **32 GiB host RAM, matching
`g6e.xlarge` exactly**. The `xlarge` sizes of both families are 16 GiB, below the measured per-actor
requirement, and are deliberately not offered as rungs.

**One confound, named and quantified.** Both candidate shapes give 8 vCPU per GPU against the
production `g6e.xlarge`'s 4. That is a real, buyable advantage — but it flatters them on the
wall-clock measure. Measured directly on the same chunk: overhead outside inference was **35.2 s on
the A10G's 8-vCPU host against 46.9 s on the L40S's 4-vCPU host**, so the CPU surplus is worth about
12 s per chunk. The `infer_s` figures isolate the GPU and carry none of it.

### The forward-pass sweep: the A10G beats the L4 at every depth

`B = 7,168`, bf16, radar length = 0.4 × optical length (the campaign's shape: 103 ascending against
263 optical on this cell), 3 warmup and 10 timed forwards per rung. **This is also the table §5's
memory law is fitted to** — `t_s2` is the optical sequence length in timesteps, `t_s1` the radar
one, and their sum is tokens per pixel:

| t_s2 | t_s1 | tokens/px | A10G tok/sec | L4 tok/sec | **A10G / L4** | A10G TFLOPS | L4 TFLOPS | VRAM alloc (GiB) | VRAM reserved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3 | 11 | 629,441 | 497,653 | **1.26** | 27.8 | 22.0 | 0.88 | 1.70 |
| 32 | 13 | 45 | 766,170 | 561,258 | **1.37** | 34.0 | 24.9 | 3.09 | 5.45 |
| 64 | 26 | 90 | 787,967 | 568,814 | **1.39** | 35.2 | 25.4 | 6.05 | 10.52 |
| 96 | 38 | 134 | 793,155 | 564,944 | **1.40** | 35.7 | 25.4 | 9.01 | 15.48 |
| 128 | 51 | 179 | 791,228 | 558,358 | **1.42** | 35.9 | 25.3 | 11.97 | 20.58 |
| 160 | 64 | 224 | 646,024 | 493,276 | **1.31** | 29.5 | 22.5 | 14.93 | 21.63 |
| 192 | 77 | 269 | 618,711 | 481,436 | **1.29** | 28.5 | 22.1 | 17.89 | 21.56 |
| 224 | 90 | 314 | **OOM** | **OOM** | — | — | — | 20.84 at failure | 20.96 |

**Peak VRAM is identical on the two cards at every rung, to the second decimal**, and both refuse at
exactly the same depth. That is the control the whole comparison rests on: the working set is a
property of the work, not of the card, so every difference in the table is throughput.

#### The hypothesis: confirmed in its ordering, refuted in its magnitude

**The ranking follows bandwidth.** The A10G beats the L4 by 1.26–1.42× while having 0.58× its quoted
tensor throughput and 2.0× its bandwidth. On a compute-bound workload the L4 would win by ~1.7×. It
loses by ~1.4×. The older Ampere card beats the newer Ada one, which is what the hypothesis
predicted and the opposite of what generation alone would suggest.

**Quoted bandwidth does not explain the size of the gap.** A 2.0× bandwidth ratio delivered 1.41×,
so roughly 30% of the advantage does not convert. **The predictor was wrong, not the hypothesis.**
Quoted peak bandwidth is a datasheet number; the counter that predicts throughput is ACHIEVED memory
traffic, DCGM DRAMA × quoted bandwidth. Measured on all three cards in one run, that product
predicts both candidates' throughput ratio against the L40S inside the measured range, where neither
quoted bandwidth nor quoted compute does (table below). Two refinements survive:

- **The L4 is power limited, and so is everything else — what differs is the cost.** The envelope
  binds on every card; the A10G is simply the only one that keeps its full clock while it binds. An
  earlier reading of "the L4 is power limited, the A10G is not" is **withdrawn**.
- **DRAMA is HIGHEST on the card with the most bandwidth** — 0.66–0.68 on the L40S against 0.50–0.51
  on the A10G. An earlier reading took the candidates' mid-range DRAMA as "no card is near a memory
  wall"; sharpened, the candidates are not near a wall because they *cannot reach* one. The L40S
  both moves more bytes per second and keeps its memory interface busy a larger fraction of the
  time, which is why "bandwidth bound" is the right ranking rule.

#### Throughput drops before the card fills, not just at the wall

Both cards lose ~20% of their throughput between optical length 128 and 160 — A10G 35.9 → 29.5
TFLOPS, L4 25.3 → 22.5 — and the reserved pool crosses 98% of the card in the same step (20.58 →
21.63 GiB). So on a 22.4 GiB card the deepest chunks are **both slower and at risk**, and the two
degrade together. There is no regime where such a card runs deep work at its own best rate.

### The crash verdict

Stated separately from throughput, because a card that runs 20% slow is a cost decision and a card
that OOMs mid-cell wastes GPU-hours.

**Both candidate cards: DISQUALIFYING as configured.**

Narrow sweep, default allocator and then with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **Run independently on both cards, and the two
agree at every rung to two decimal places:**

| t_s2 | VRAM needed | default allocator | with `expandable_segments` |
|---:|---:|---|---|
| 192 | 17.89 GiB | ok on both | ok on both |
| 200 | 18.63 GiB | ok on both | ok on both |
| **208** | 19.37 GiB | **OOM on both** (2.25–2.30 GiB free, wanted 6.40) | ok on both |
| 216 | 20.10 GiB | not attempted | ok on both |
| 224 | 20.84 GiB | not attempted | ok on both |
| 232 | 21.58 GiB | not attempted | ok on both |
| **240** | — | not attempted | **OOM on both** (wanted 2.46, had 1.84) |

That the A10G and the L4 refuse at the identical rung, having allocated identical bytes at every
rung below it, is the strongest form this control can take.

**The default configuration refuses at 208, and the deepest depth ever recorded in this campaign is
206.** The ladder buckets a pixel's observation count to the next multiple of 8, so a 206-deep cell
presents 208. The margin is zero — not thin, zero — and it is reached by a cell that has already run
once (38N/2021).

`expandable_segments:True` moves the wall from 208 to 232 on both cards, at no measurable throughput
cost (A10G 27.2 against 27.8 TFLOPS, L4 22.2 against 23.2 at matched depth, inside run-to-run
noise). That is a **24-timestep margin over the deepest depth observed** — still short of the
model's own 256-rung ceiling, which no configuration tested can reach on a 22.4 GiB card.

**The flag ships.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is on `InferenceActor`'s Ray
`runtime_env` and merged in #154, so the prerequisite for opening either 24 GB rung is met and the
crash verdict should be read with the flag in force. (An earlier version of this section ended "the
flag is not set anywhere today," which was true when written.)

So the honest statement has three parts. **Under the DEFAULT allocator, both candidate cards OOM on
the deepest cell the campaign has already run** — which is why the flag was a precondition for
opening either rung rather than a tuning nicety. **With the flag they clear that cell with a
24-timestep margin.** And neither can reach the sequence length the model declares it supports, on
any setting tested — which is a statement about the card, not about the flag.

**The residual risk, stated plainly:** a cell deeper than 232 would OOM on a fallback worker. None
has ever been observed, but the margin is three buckets, not a comfortable multiple, and it is the
reason to watch the first fills that run on a fallback rung.

**What did NOT go wrong**, recorded because absence of failure is also evidence: across the live
cluster run there was no OOM, no CUDA error, no worker replacement and no actor restart, at depths up
to **123**. The reserved pool sat at 21.71 of 22.06 GiB — **98.4% of the card** — throughout, on both
candidates, and nothing failed at these depths. **The L40S in the same run reserved 28.1–35.8 of its
44.4 GiB, 63–81%**, on identical work: the same allocator behaviour with headroom left, which is the
whole difference between the two card classes on the crash question. It is the depths a run like
this cannot reach that decide the verdict, which is exactly why the synthetic sweep exists.

### Everything else that was watched

| metric | A10G (`g5.2xlarge`) | L4 (`g6.2xlarge`) | note |
|---|---|---|---|
| peak `max_memory_allocated` per chunk | 7.52 GiB at depth 108–113 | 7.52 GiB | identical, per chunk, reset each chunk |
| peak `max_memory_reserved` per chunk | 21.68–21.71 of 22.06 GiB | 21.68 of 22.04 | **98.4%** — no fragmentation headroom |
| host RAM peak (1 s sampler) | **16.03 GiB of 31.0 (52%)** | 15.76 GiB (52%) | never above 55%; the 60% ceiling held |
| host RAM peak (2 s in-actor, per chunk) | 13.97–15.9 GiB | 14.0 GiB | on `CHUNK_SUMMARY` as `host_ram_peak_gib` |
| GPU utilisation | 66% avg, 100% max | 76% avg, 100% max | busy fraction 0.67 / 0.77 |
| thermal throttling | none — 63–72 °C, bitmask `0x0` | none thermal; bitmask `0x4` = SwPowerCap | no `HwSlowdown`, no thermal slowdown on either |
| PCIe link | gen4 **x8** | gen4 **x8** | half the L40S's x16 lanes |
| PCIe traffic | 3.2 GB/s in, 0.4 GB/s out | 1.3 GB/s in, 0.17 GB/s out | ~20% of a gen4 x8 link; not binding |
| S3 read concurrency | 24 sockets mean, 126 max | 10 mean, 115 max | no throttling observed |

**The 16 GiB host-RAM exclusion is measured rather than inferred.** Peak host RAM was **16.03 GiB**
on a 31 GiB host. A `g5.xlarge` or `g6.xlarge` has 16 GiB total, so the per-actor working set alone
consumes the entire machine. Those sizes are the cheapest per GPU-hour in either family, which is
exactly why the rung exclusion is stated over every offered node type rather than naming one
instance.

### All three cards on the same work at the same minute

The body above was written while `g6e` was refusing, so its L40S column was a cross-run comparison.
**The pool refilled at 18:59Z**, a `g6e.xlarge` launched into the arm left standing in the ladder for
exactly that possibility, a second followed at 19:04Z, and both banked 5–6 chunks before teardown.

An earlier attempt at this comparison silently measured an L4 while believing it an L40S, so
**identity is asserted three ways per row** — IMDS `instance-type`, `nvidia-smi --query-gpu=name`,
and the actor's own `vram_total_gib` on the `CHUNK_SUMMARY` line — cross-checked against
`ec2:describe-instances`. All eight workers agreed on all four, at 100% GPU utilisation.

| shape | card | hosts | chunks | tok/s busy | tok/s infer | ratio busy | ratio infer | $/GPU-hr | Mtok/$ | $ per unit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `g6e.xlarge` | **L40S** | 2 | 11 | **2.06M** | **2.38M** | 1.000 | 1.000 | 1.861 | **3975** | **1.00×** |
| `g5.2xlarge` | A10G | 3 | 15 | 0.94M | 1.06M | **0.459** | **0.444** | 1.212 | 2799 | **1.42×** |
| `g6.2xlarge` | L4 | 1 | 4 | 0.65M | 0.69M | **0.318** | **0.292** | 0.978 | 2405 | **1.65×** |

Host spread inside the L40S arm is **1.5%**, tighter than the 4.3% noise floor the single-type
control arm measured. Holding optical depth constant (100–125, 22 chunks) moves the ratios to
0.468/0.456 for the A10G and 0.330/0.306 for the L4 — i.e. **not at all**, so the differing chunk
mixes across arms were not carrying the result. This **confirms the cross-run estimates** (A10G
0.42–0.50, L4 0.294–0.344) rather than displacing them, and the agreement between two instruments
confounded in opposite directions is the reason to trust either.

DCGM, 45 one-second samples per host at 19:16Z:

| card | GRACT | SMACT | TENSO | DRAMA | quoted bandwidth | DRAMA × bandwidth |
|---|---:|---:|---:|---:|---:|---:|
| L40S (`g6e.xlarge`) ×2 | 0.995 | 0.987–0.991 | 0.419–0.440 | **0.659–0.675** | 864 GB/s | **~576 GB/s** |
| A10G (`g5.2xlarge`) | 0.998 | 0.997 | 0.336 | 0.501 | 600 GB/s | **~301 GB/s** |
| L4 (`g6.2xlarge`) | 0.998 | 0.996 | 0.515 | 0.559 | 300 GB/s | **~168 GB/s** |

The dev L40S reproduces the prod fleet's SMACT 0.99 / TENSO 0.42–0.47 almost exactly, which is a
useful check that a two-host arm sees the same regime as 338 prod hosts. **Achieved traffic is the
predictor that works:**

| predictor, as a ratio to the L40S | A10G | L4 | agrees with measurement? |
|---|---:|---:|---|
| quoted BF16 dense tensor compute | 0.193 | 0.334 | no — off by 2.4× for the A10G |
| quoted peak memory bandwidth | 0.694 | 0.347 | no — overstates the A10G by ~35% |
| **DRAMA × quoted bandwidth** | **0.523** | **0.291** | **yes, both cards** |
| measured | 0.416–0.526 | 0.276–0.342 | — |

Power, and why the cap is not the discriminator people expect:

| card | draw / limit | SM clock / max | bitmask |
|---|---|---|---|
| L4 ×3 | 71–73 W of 72 W (99–101%) | **810–840 of 2040 MHz — 40%** | `0x4` on all 3 |
| L40S ×2 | 339–341 W of 350 W (97%) | 1470–1785 of 2520 MHz — 58–71% | `0x4` on both |
| A10G ×3 | 267–282 W of 300 W (89–94%) | **1665–1710 of 1710 MHz — 97–100%** | `0x4` on 2, `0x0` on 1 |

### The economics, which is what actually decides it

$/GPU-hour from the Pricing API, us-west-2, on-demand Linux shared tenancy, 2026-08-27. A card is
worth adopting when its throughput ratio **exceeds** its price ratio.

| type | card | $/GPU-hr | price ratio (= break-even) | measured throughput ratio | $ per unit of work |
|---|---|---:|---:|---:|---:|
| `g6e.xlarge` | L40S | 1.861 | — | 1.000 | **1.00×** |
| `g6.2xlarge` | L4 | 0.978 | 0.525 | **0.29–0.34** (same-run: 0.292 infer / 0.318 total) | **1.53–1.79×**, same-run **1.65×** |
| `g5.2xlarge` | A10G | 1.212 | 0.651 | **0.42–0.53** (same-run: 0.444 infer / 0.459 total) | **1.30–1.55×**, same-run **1.42×** |
| `g6.4xlarge` | L4 | 1.323 | 0.711 | as above | 2.07–2.42× |
| `g5.4xlarge` | A10G | 1.624 | 0.873 | as above | 1.75–2.08× |
| `g6e.2xlarge` | L40S | 2.242 | 1.205 | 1.00–1.15 (CPU feed only) | 1.05–1.21× |

**Every candidate costs more per unit of work than the shape we already run.** The A10G is the
better of the two and still lands 30–55% above `g6e.xlarge`; the L4 lands 53–79% above. The `xlarge`
sizes would be cheaper per GPU-hour, and cannot hold the actor.

### Capacity: the pool refills, and a shortage takes a family down together

An earlier reading — "the pool is the card" — is **too strong, and is corrected here**. It came from
all eight `g6e` sizes refusing at once, which is what a hard family-wide shortage looks like.
Measured across the day, size availability within a family varies:

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

The accurate statement: **a hard shortage takes a whole card family down together, and under mild
pressure the larger sizes go first.** The 1-GPU `xlarge` sizes were the most available in every
family and the 4-GPU sizes the least — which runs against the "fewer, bigger boxes" instinct in both
directions.

**And the drought is intermittent, not permanent.** After refusing continuously from 17:14 to 18:58
— nearly two hours, across four probe rounds and two fleets' worth of autoscaler retries — a
`g6e.xlarge` launched at 18:59. That is the single most important capacity fact here: **the L40S
pool refills.** A card that is 13–53% cheaper per unit of work and comes back within hours is a
different proposition from one that is gone.

### Ray reaches for the right rung — and, between the two candidates, the wrong one

Ray's scorer run directly against post-autodetect resource dicts, with its autoscaler's
`functools.partial` binding, at demand from 1 to 250 bundles:

| rung | instance type | vCPU | score |
|---|---|---:|---|
| `gpu-workers-ondemand` | `g6e.xlarge` | 4 | `(True, 2, 0.0, 0.265625)` |
| `gpu-workers-ondemand-l4-2xl` | `g6.2xlarge` | 8 | `(True, 2, 0.0, 0.25390625)` |
| `gpu-workers-ondemand-a10g-2xl` | `g5.2xlarge` | 8 | `(True, 2, 0.0, 0.25390625)` |

`g6e.xlarge` wins at every demand level, because 4 vCPU per GPU fits the one-CPU bundle more tightly
than 8 and idle vCPU drags the mean down. **Opening a candidate rung therefore cannot quietly move a
healthy fleet off the L40S.**

**But the two candidates score byte-identically**, and `get_nodes_for` sorts `(score, node_type)`
tuples with `reverse=True`, so the tie falls to the node-type NAME descending — `…-l4-2xl` beats
`…-a10g-2xl` because `l` sorts after `a`. In exactly the situation the fallback exists for, Ray
fills the fleet with L4s: 0.318 of an L40S against the A10G's 0.459, and 1.65× the unit cost against
1.42×.

**Operating rule: open at most ONE candidate rung, and make it `g5.2xlarge`.** A ladder is
authoritative over its whole domain, so naming only `g6e.xlarge` and `g5.2xlarge` closes the L4
rungs by construction — no code change needed.

A guard forbidding both candidate rungs was written and then **removed**, because the premise as
baldly stated is refutable: two small equal caps give both rungs work, since the cap binds before
the score does. Both readings are true in different shapes, and measurement settled which applies
where:

| ladder | what Ray launches |
|---|---|
| `g6e.xlarge:400,g5.2xlarge:400,g6.2xlarge:400` | `g6e.xlarge` — correct |
| `g5.2xlarge:400,g6.2xlarge:400` (the drought case) | **`g6.2xlarge` — the worse card** |
| `g6.2xlarge:100,g5.2xlarge:100` | 8 L4, **zero** A10G — the score decides |
| `g6.2xlarge:3,g5.2xlarge:3` | 3 of each — the cap decides |
| `g5.2xlarge:100` or `g5.2xlarge:400` | A10G — correct |

So the hazard is real but narrower than "never open both": **it bites when a fallback is opened
WIDE, which is the shape a fallback is actually opened in.** Hence a rule plus two tests
(`test_an_uncapped_pair_of_fallback_cards_goes_ENTIRELY_to_the_L4` and
`test_the_a10g_rung_alone_takes_the_whole_fallback_fleet`) rather than a guard — the uncapped case
is pinned so the rule's reason cannot silently stop being true.

### Verdict

**As a preferred type: no, for both.** The A10G's most favourable reading (0.526) is still well
under its 0.651 break-even; the L4's (0.342) is well under its 0.526. Not marginal.

**As a capacity fallback: the A10G yes, the L4 no** — because when `g6e` refuses, the alternative is
an idle fleet at infinite cost per token, and 1.42× beats that. Gated on two things: the allocator
flag, which now ships; and only the A10G rung being open.

**The failover is not manual.** Ray's DEFAULT scorer accepts a `node_availability_summary` and never
reads it, so a capacity refusal never moves it off the top-scored rung. Ray lets you replace that
scorer (`RAY_AUTOSCALER_UTILIZATION_SCORER`), and `providers.aws.autoscaler_scorer` now does,
demoting a rung AWS last refused for want of capacity. Enabling it is
`gpu_fallback_instance_types=["g5.2xlarge"]` on the campaign, which opens the rung and installs the
scorer together.

**One case remains manual, and it is the common one.** A launch that returns FEWER instances than
requested is a success — `MinCount: 1` — so a production rung that is trickling rather than refusing
outright is never marked unavailable and the scorer never fires. Releasing demand to the fallback
under a trickle still means capping the production rung, which is one SSM key. The automatic path
covers a hard outage; the cap covers a slow one.

**Spend on the whole investigation: about $14** — 40 capacity probes at ~$0.10, four bench sweeps at
$0.35–0.62 each, and two cluster windows at ~$5–7 and ~$6. No run was allowed to complete; a full
Iowa pass is hours of GPU time and would add nothing the question needs. The cluster was **not** torn
down when the first arms had their medians — it was held ~25 more minutes, for about $4, while the
`g6e` arm banked enough chunks to clear the three-chunk cold-start bar. That bought the one row the
body could not otherwise fill.

**What could not be measured, named rather than filled in:** a synthetic L40S forward-pass figure
(two live instruments already agreed on the ratio, so a third instance-hour was not spent); where
the missing 30% of the A10G's bandwidth advantage goes; whether `frac_of_card_tflops` is a true
utilisation (a measured A10G reached 35.9 TFLOPS against a derived FP32-accumulate ceiling of 35.0,
which cannot happen — so `_CARD_CEILINGS` carries the vendor's quoted dense figure and the fraction
is documented as an index); per-card `get_batch` latency; sustained network throughput per card; and
multi-GPU sizes of either family, which refused in all availability zones every time they were
probed.

---

## 5. Why the batch is sized to the card it lands on

### What went wrong, on 29 August 2026

The 2026-08-29 global campaign was the first to run two card types at once: the **L40S** (44.4 GiB
usable) in `g6e.xlarge`, which had always been the only rung, and the **A10G** (22.06 GiB usable) in
`g5.2xlarge`, added by PR #159 as the capacity fallback §4 sanctions.

Within ten hours the fills had logged roughly **4,100 actor deaths**. The run immediately before it
— ten fills over 23 hours, L40S only — logged **zero**. Every dead actor names its host in a
`Replacing dead actor N (was on i-...)` line, and resolving those instance ids against EC2 gives an
unambiguous split:

| instance type | card | usable memory | dead-actor hosts |
|---|---|---:|---:|
| `g5.2xlarge` | A10G | 22.06 GiB | **178** |
| `g6e.xlarge` | L40S | 44.4 GiB | **0** |

That is 178 of 178. The Ray logs give the mechanism with no interpretation required:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.54 GiB.
GPU 0 has a total capacity of 22.06 GiB of which 1.17 GiB is free.
Including non-PyTorch memory, this process has 20.89 GiB memory in use.
```

The card was asked for 2.54 GiB it did not have. The process died, Ray replaced it, the replacement
re-loaded the 175 MB checkpoint and picked up the work — and, running the same batch on the same
card at the same depth, often died the same way.

### The memory rule, measured

§1 asserted that memory is proportional to `batch_size × tokens per pixel`. That does not have to be
argued from the code: the forward sweep in §4 measured it directly, at the production batch, in
bf16, with radar depth held at 0.4 × optical depth. Plotted against **tokens per pixel** — not
against optical depth alone — the VRAM-allocated column of that table is a straight line:

> **allocated GiB = 0.0660 × (tokens per pixel) + 0.14**, at `batch_size = 7,168`

| statistic | value | meaning |
|---|---:|---|
| R² | 0.999992 | the line explains essentially all of the variation |
| largest residual | 0.03 GiB | worst disagreement between line and measurement |
| slope, per token | 9.65 KiB | memory cost of one token, at this batch |
| intercept | 0.14 GiB | the fixed part: weights, workspace, allocator overhead |

Two checks that this is a law and not a curve fit — neither was used to produce the line.

**Check 1, against §4's refusal boundary.** The narrow sweep's last working rung was `t_s2 = 232`,
which is 325 tokens per pixel, recorded at 21.58 GiB. **The line predicts 21.59 GiB.** The next rung
up, 336 tokens per pixel, is predicted at 22.31 GiB against a 22.06 GiB card — and that is the rung
that refused.

**Check 2, against the production failure.** The traceback above needed 20.89 + 2.54 = **23.43 GiB**.
Inverting the line puts that at **353 tokens per pixel**. Independently: the deepest optical depth
ever recorded in this campaign is 206, which the ladder rounds to 208; the deepest radar depth
measured anywhere is 147, which rounds to 152; the sum is **360 tokens per pixel**, which the line
predicts at 23.90 GiB. **The line reproduces a production out-of-memory event it was never fitted
to, to within 2%.**

#### Why the deaths clustered in particular fills rather than spreading evenly

This looked at first like evidence against a simple memory rule. It is the opposite — it is what the
rule predicts.

At `batch_size = 7,168`, rearranging the line for a 22.06 GiB card gives a wall at **332 tokens per
pixel**. And what has to fit is a fill's **deepest** bucket, not its average pixel: the scheduler
deliberately runs the deepest bucket first, so a chunk containing any deep pixels pays its peak
immediately.

| quantity | value |
|---|---:|
| campaign land-weighted mean | 173 tokens per pixel |
| optical depth, observed range | 57 – 206 |
| radar depth, observed range | 66 – 147 |
| correlation of optical depth with latitude band | Pearson r = +0.912 |
| correlation of radar depth with latitude band | Pearson r = +0.009 |
| where `B = 7,168` fills a 22.06 GiB card | 332 tokens per pixel |

The mean is comfortably under the wall. But optical depth tracks latitude strongly and radar depth
does not track it at all, so the two go deep **independently**, and the places where both happen to
go deep are scattered and geographically clustered. Those are the fills that died. Nothing about
their imagery was anomalous; they were simply on the far side of a threshold the batch size had no
way of knowing about.

### The fix

`batch_size_for_gpu` computes the batch each actor should run, once, on the actor's own device,
before anything reads it. The calculation is a single ratio applied to the **calibrated** batch of
7,168:

> **fitted batch = 7,168 × (this actor's share of the card ÷ 44 GiB) × (512 ÷ the ladder's deepest
> tokens per pixel)**, capped at whatever the caller asked for.

44 GiB and 512 tokens per pixel are what the calibration covered. The result on the two production
cards: **the L40S is untouched at 7,168; the A10G gets 3,593.**

#### Why one number is enough — the worst case is bounded in advance

The reason a single fitted value can be safe for every bucket is the clipping in §1. A pixel's
sequence is **not** open-ended: each of the two streams is clipped to the ladder's deepest rung,
256, so **no bucket the sampler can build carries more than 512 tokens per pixel.** That is a
property of the bucketing, fixed before any imagery is read, and completely independent of
geography.

So the deepest sub-batch that can ever be presented to a card is `batch × 512` tokens:

| configuration | batch | deepest legal sub-batch | predicted VRAM | of the card |
|---|---:|---:|---:|---:|
| L40S, unchanged | 7,168 | 3,670,016 tokens | 33.9 GiB | 76% of 44.4 GiB |
| A10G, before this change | 7,168 | 3,670,016 tokens | 33.9 GiB | **154% of 22.06 GiB** |
| **A10G, fitted** | **3,593** | **1,839,616 tokens** | **17.1 GiB** | **77% of 22.06 GiB** |

Two independent ways of reading the A10G row. **Against measurement:** 1,839,616 tokens is **79% of
the 2,329,600 an A10G was measured to complete** (7,168 pixels at the 325-token rung, the deepest
that worked in §4's narrow sweep). **Against the deepest depth ever actually observed:** at 360
tokens per pixel the fitted batch needs 12.1 GiB — **55% of the card**.

The table also shows why the L40S needs no change and gets none: at 76% of its memory it cannot
reach an out-of-memory condition on any bucket the sampler can legally build. **That is a falsifiable
prediction.** If any chunk recorded `PERMANENTLY FAILED` on 2026-08-29 turns out to have died on a
`g6e.xlarge`, this line is wrong and the derivation must be re-opened.

Scaling by memory holds the deepest legal sub-batch at nearly the same fraction of every card — 76%
on the L40S, 77% on the A10G. That is not luck. It is what a linear memory law and a
memory-proportional batch necessarily produce together, and it is why one ratio generalises to cards
nobody has measured yet.

### Why this approach and not the others

Four alternatives were on the table. Each is stated as its advocate would state it.

**Alternative A — give each bucket its own budget, from its actual depth.** Stop shipping a pixel
count at all. Ship a *token* budget, and at each bucket compute `batch = budget ÷ that bucket's
tokens per pixel`. Deep buckets automatically get small slices; shallow buckets get large ones. It
is the exact expression of the memory rule rather than a conservative approximation of it.

*Rejected on two measurements.* **It solves a problem that is already solved:** the clipping bound
means the worst case is known before a tile is read, and one fitted value clears it with 21% margin
against the measured ceiling. **And its only remaining benefit is throughput, which is not there:**
adaptive token-budget batching was measured on the real model and shelved (§3) — `B = 7,168` is
throughput-optimal at *every* sequence length, and the card is already saturated at these sizes; on
the A10G, SMACT 0.995 at its full 1710 MHz clock. There is no idle capacity for a bigger
shallow-bucket batch to fill. What it would cost is a second calibration constant, a per-bucket
branch in the hot loop, and a new failure mode: a mis-set token budget becomes a per-bucket
out-of-memory rather than a whole-run one, which is harder to see and harder to reproduce. **How to
reopen it:** if a future measurement shows throughput rising with batch size at shallow depth on a
small card, this reasoning no longer holds.

**Alternative B — just halve the number for the A10G and move on.** One line: if the card is an
A10G, use 3,584. It would have stopped the 29 August failures.

*Rejected.* It only knows the cards you happened to enumerate — `fleet_mix.GPU_RUNGS` is a list that
changes, and the failure mode for "no answer" is the 7,168 default, which is the bug. It is not a
statement about anything, so it cannot be checked: the ratio version makes a claim — *hold the
deepest legal slice at a fixed fraction of whatever card is present* — that the linear memory law
makes true for cards never benchmarked, and that a test can pin. And it has nowhere to put the other
inputs below.

**Alternative C — use the smaller batch everywhere.** Set the default to 3,593 for all cards. One
number in the config, obviously safe on both.

*Rejected on measured cost.* It pays a permanent throughput tax on the card that does nearly all the
work, to protect a card only rented when the primary is unavailable. The pipeline's own baseline ran
at 3,584 before the July campaign, and the per-sub-batch CPU preparation cost is roughly **flat in
batch size** — 165 ms for a 3,584-pixel sub-batch against 103–160 ms for a 7,168-pixel one (§2).
Halving the batch therefore roughly doubles that fixed cost per pixel and doubles the number of
forward launches for the same imagery.

**Alternative D — refuse the small card.** Close the `g5.2xlarge` rung.

*Rejected.* The rung exists because of capacity, not preference (§4). Refusing it means the campaign
stalls rather than running somewhat slower on a smaller card.

### What the fit has to see, and why

"The batch needs no per-bucket term" is not the same claim as "memory is the only input". The
512-token cap is a *consequence* of two configured things — how deep the ladder goes, and how many
actors share the card — and a fit that ignored them would be safe only by coincidence. Automated
review found both, plus a third route around the calculation entirely, and they compound rather than
compete.

**A deeper ladder.** `num_obs_checkpoints` is an ordinary config field and the pipeline accepts any
positive depths. A caller passing `(512,)` doubles every sub-batch's token count. A memory-only ratio
would still hand the A10G 3,593 pixels — a 3,679,232-token sub-batch, 158% of the measured ceiling.

**A packed card.** A fractional Ray reservation (`num_gpus`) deliberately puts several actors on one
card. Each sizing to the card's *total* memory oversubscribes it by exactly the packing factor. The
share an actor actually receives is `1 ÷ floor(1 ÷ num_gpus)`, because that is how many actors Ray
fits — so `num_gpus = 0.6` packs one actor that owns the *whole* card, while `0.4` packs two that
get half each. Reading the reservation itself as the share is wrong in both directions.

**A caller asking for more than was calibrated.** `batch_size` is a settable field. Scaling *the
caller's request* by the card ratio preserves an over-ask: a request of 10,000 on an A10G came out at
5,013, a 2,566,656-token sub-batch, 110% of the measured ceiling. The fit is therefore derived from
the calibrated 7,168 and the caller's number applied only as a ceiling.

All three are handled the same way — one scale factor, still computed once in the actor — and the
demand placed on a card is then invariant across configurations that ought to be equivalent:

| ladder | reservation | actors on the card | fitted batch | tokens demanded of the card | of measured ceiling |
|---|---|---:|---:|---:|---:|
| default (deepest 256) | whole card | 1 | 3,593 | 1,839,616 | 79% |
| default (deepest 256) | 0.5 | 2 | 1,796 | 1,839,104 | 79% |
| `(512,)` | whole card | 1 | 1,796 | 1,839,104 | 79% |
| `(512,)` | 0.5 | 2 | 898 | 1,839,104 | 79% |
| default (deepest 256) | 0.1 | 10 | 359 | 1,838,080 | 79% |
| `(4096,)` | whole card | 1 | 224 | 1,835,008 | 79% |

Same card, same demand, six configurations. Before the fit took all three inputs, the last two rows
read **2,621,440 tokens (113%)** and **4,194,304 tokens (180%)**. The L40S at the default ladder on a
whole card still gets 7,168, unchanged, because its share already exceeds the calibration reference.

**Two deliberate choices inside that.**

*There is no minimum batch size.* An earlier version floored the fitted value at 512 pixels, on the
reasoning that below that the per-forward overhead dominates and a small card is starved rather than
protected. That floor is what produced the 113% and 180% rows above: it silently raised the batch
back over the bound in exactly the configurations nobody watches. A configuration whose safe batch is
224 pixels now gets 224 pixels and runs slowly. **Slow is a cost; out-of-memory is a failure.**

*A configuration is never refused.* Review's alternative was to reject configurations whose safe batch
falls below an operational floor. Refusing turns a slow run into a dead one, and the configurations in
question — a 0.1 GPU reservation, a 4,096-deep ladder — are not production settings and have no
operator waiting to fix them.

*One caveat the table does not show.* The invariant is on *activations*. Each actor also pays fixed
per-process costs — its own CUDA context and its own copy of the weights — which the token accounting
does not model. That is the fitted line's 0.14 GiB intercept, and at the default of one actor per card
it is irrelevant. At ten actors per card it is paid ten times, and the true demand is nearer 83% of the
card than 79%. Heavy packing is not a production configuration, and this is one of the reasons.

### What the margin rests on

Two assumptions. Both now fail a test rather than fail a fill.

**The token cap itself.** `test_the_fitted_batch_holds_the_deepest_bucket_the_sampler_can_build`
pins `fitted × 2 × max(checkpoints) × actors-per-card` against the measured ceiling, over every
combination of ladder depth and reservation in the table above.
`test_the_unfitted_batch_does_not_hold_it` proves that bound is not vacuous by showing the old batch
fails the same test. Dropping either the depth term or the packing term from the fit turns exactly
the affected rows red; this was verified by doing it.

**Segment-backed allocation.** Every VRAM figure here is *allocated* bytes — memory handed to live
tensors. Under PyTorch's default caching allocator the *reserved* pool runs well above that and
strands the difference: **20.58 GiB reserved for an 11.97 GiB working set**, measured. That is why
the A10G's refusal boundary sits at optical depth 208 without the flag and 232 with it (§4).
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ships on the actor's Ray `runtime_env` and is
pinned by `test_the_actor_ships_the_segment_backed_allocator`. Removing it would re-open this
failure silently, on the fallback rung only.

### What it cost while unfixed

**In the first ten fills, nothing permanent.** A chunk gets three attempts before it is recorded
`PERMANENTLY FAILED`, and a scan of every ERROR record across all ten found zero. The cost there was
GPU time: roughly 4,100 actor restarts in ten hours, each re-loading a 175 MB checkpoint before the
replacement could take work. The **7,830 chunks/hour** measured that afternoon is therefore a figure
*including* this churn, not a clean throughput number.

**Then it stopped being free.** Between 16:27Z and 16:54Z on 2026-08-29 a single fill recorded **16
chunks `PERMANENTLY FAILED`**, every one an out-of-memory on the third and final attempt. Three
attempts at the same chunk, on the same card, at the same batch size, is not three chances at all:
the sub-batch either fits or it does not, and the entire retry budget was spent on a deterministic
refusal.

**A note for whoever merges a change like this.** It touches the `inference` package, so it moves
the **staging fingerprint** — the identity the pipeline computes over its own source to decide
whether previously staged inputs can be reused. A campaign resuming previously staged tiles must be
dispatched with the existing `staging_code_identity`, exactly as the 2026-08-29 restart was. See
[`../storage/staging-identity-and-resume.md`](../storage/staging-identity-and-resume.md).

---

## 6. Measured on the campaign path

Everything above was measured on the Iowa reference region or on synthetic tensors. This section is
the first GPU profile taken on the **campaign path** — whole UTM zones, campaign thresholds,
`fill-zones-sequential` — and it is where the cost model's two headline inputs come from. Source:
`p3-chained-7zones-v2` on 20 `g6e.xlarge` actors, cells processed sequentially in one process.

### READ THIS FIRST: what this telemetry can and cannot tell you

> **`CHUNK_SUMMARY` cannot be attributed to a zone, and attempting it produced two wrong findings
> before the attribution itself was checked.** The line carries no zone. Its `label` is
> `chunk_<row>_<col>` — **grid-local**, so every cell restarts at `chunk_0_0` and labels collide
> across zones *and* across concurrently running fills, which share one log group.
>
> Attributing by time window and treating repeated labels as retries produced a confident "49S
> re-inferred 101 chunks after write failures, a 9.1% GPU tax". **Both halves were false.** There is
> not one `staging write unconfirmed` / `writer pool wedged` / `flush failed` line in the window —
> the mechanism did not fire at all. And the giveaway was in the data already: the kept-observation
> count *differed* between the supposed retry pairs (60→56, 122→58), which cannot happen when
> re-running identical data. They were two different zones sharing a label space.
>
> A second attempt, splitting cells where a label repeats, is also unsound: a genuine within-cell
> retry splits a segment, and the boundaries it produced contradict the chain's own per-cell counts.
>
> **So this section reports aggregates over ONE log stream, plus per-cell facts the chain states
> outright.** It does not report per-zone throughput, because the instrument cannot support it.

**The fix is one field.** The chain already mints a per-cell run id (`49S-2021-f1fa65fc`) and logs
it. Adding that id — or just the zone — to `CHUNK_SUMMARY` makes every figure below decomposable per
zone, retries distinguishable from collisions, and concurrent fills separable. It is the
highest-value instrumentation change available and it is additive, so it cannot break the
`campaign_progress.py` needles.

`write_confirmed` is **always `false`** in this line, by design: the write is confirmed one chunk
later via chain-confirmation, which updates an in-memory record the log never sees. It is not an
unconfirmed-write signal.

### Per-cell chunk counts, as the chain states them

Authoritative, because the chain logs them with the zone and the run id (`Zone <z> year <y>: N/<total>
tiles are live in the campaign coverage mask`):

| cell | live chunks | of grid |
|---|---:|---:|
| 49S-2021 | 943 | 14,355 |
| 48S-2021 | 774 | 14,355 |
| 17S-2021 | 767 | 14,355 |
| 32S-2021 | 386 | 14,355 |
| 58S-2021 | 354 | 14,355 |
| 02N-2021 | 245 | 15,048 |

**A useful sanity figure in its own right: live chunks run 1.6–6.6% of the grid** on these southern
and tropical zones. Fill cost scales with the live count, not the grid.

### Aggregate, one runner, cells 49S → 32S

| | |
|---|---:|
| chunk events (deduped) | 2,123 — 2,118 success, 5 skipped |
| wall span | 6.55 h |
| actors | 20 |
| GPU-hours | 131 |
| cost at $1.861/GPU-h | **$244** |
| inference seconds per chunk, median | 206.9 |
| **kept optical observations** | **median 64, p10 55, p90 122, range 34–145** |
| tokens delivered | 648.0 G |
| **tok/s per actor (wall-clock)** | **1.37 M** |
| **$/chunk** | **0.115** |

Skips are rare and benign: **5 of 2,123 (0.24%)**.

### What this run established, after four corrections

**Price in tokens, not chunks.** A fixed per-chunk price cannot describe a 2× spread whatever causes
it. This is the finding the cost model's whole structure rests on.

**Throughput stratifies by radar status, and the split is the reason.** Optical tokens per second
per actor, busy:

| population | tok/s per actor |
|---|---:|
| radar-free | 2.26 M – 2.93 M |
| one orbit | 1.60 M |
| **both orbits** | **1.26 M – 1.62 M** |

The old planning reference of ≈1.9 M sits **above every both-orbit cell measured and below every
radar-free one**, which is what made a single rate untenable. All figures here are *optical* tokens
— the unit mismatch that correction 4 is about.

**Four corrections, and what each one teaches.**

1. **"Cost per chunk tracks kept-observation count" — WITHDRAWN.** The deepest cell measured
   (23N-2021, 158 kept) is among the cheapest ($0.113) *because it is radar-free*. Depth and radar
   status are confounded across the measured set, and this document read the combination as depth
   alone.
2. **"Nothing measured is slower than the model assumes" — WITHDRAWN.** True in aggregate, false for
   the population that matters: stratified by radar status the picture reverses.
3. **Land shares from a per-ZONE presence survey, presented as coverage — WITHDRAWN.** The
   area-weighted per-pixel census in the cost model is the authority: 81% of land covered in
   2022–2024 against 100% for 2017–2021, and 6.8% of pixel-years optical-only after Sentinel-1B
   failed. Radar-free work is a **larger** share than the withdrawn figure implied, which makes the
   throughput split matter more, not less.
4. **Kept-count 145 as a planning depth — WITHDRAWN, unit-mixed.** 145 is a COMBINED census figure
   and the kept count is optical, so "inside the observed 57–158 range" compared quantities in
   different units. Measured planning depths are **103.1 optical** (land-weighted) and **170
   combined**.

**The two lessons worth more than the numbers.**

*Refusing to claim a direction was load-bearing.* With the exposure known to be a unit mismatch, two
terms pointed opposite ways: the census counts S2+S1 while the rate counted optical only (making the
line conservative), but the rate was measured at sites carrying at most one orbit and both-orbit
chunks are slower per optical token (cutting the other way). An earlier version said the reference
"looks optimistic by roughly 20–35%"; that was withdrawn as unsupported, because it netted one term
against nothing. **When the correction was finally taken with both terms measured, the line moved
0.98× — and a one-error correction would have moved it 19% the wrong way.**

*A warning does not retract the sentence it invalidates.* This document restated "cost per chunk
tracks the kept count" as a surviving conclusion *two sections after* the radar finding showed that
range mixes two populations — and then used the restatement to rule out re-basing the budget. The
warning and the claim it invalidated sat in the same file for a day. **Withdraw the sentence, not
just its neighbourhood.**

### An independent cross-check of the inference line, from the coverage mask

`scripts/rank_zones.py` reads the campaign coverage mask directly: **360,953 live tiles across 112
zones**, i.e. per campaign year. Nine years is **3.25 M chunk-years**, and this run measured cost per
delivered chunk at **$0.101–0.214** depending on depth.

| assumed land-weighted mean $/chunk | implied inference line |
|---|---:|
| $0.115 (the sparse-zone figure) | $374 k |
| $0.15 | $487 k |
| $0.18 | $585 k |

The model's **$503–579 k** sits inside that band at a mean of roughly $0.155–0.18 — a plausible place
for a land-weighted mean to land, given how far apart the two modes are. **This is a genuinely
independent route to the same number**: it multiplies a chunk census from the coverage mask by a
measured per-chunk cost, using neither the token census nor the reference rate that the model's own
derivation depends on. Two unrelated methods agreeing to within their spreads is the strongest
evidence yet that the inference line is sound.

> **Watch this figure: the equatorial cost is running higher than first measured.** At 61–68 chunks
> each, 03N and 06N are at **$0.30–0.31 per chunk** — not the ~$0.20 an earlier, smaller sample
> suggested — because their chunks take 475–480 s of inference against ~200 s in the far south. Some
> of that is cluster warm-up amortised over few chunks and will fall as they progress; how much is
> exactly what completing them answers. If the equatorial mode settles near $0.30, a land-weighted
> mean lands above $0.18 and the inference line moves toward the top of the model's range or past
> it. **Do not treat the band above as settled**, and do not read the table as narrowing the
> estimate — the mean $/chunk is exactly the unknown the 17-zone programme measures, and quoting a
> row of it as a result would be picking the answer.

### Operational notes

- **The chain deletes each cell's staging prefix and source mosaics after the fill lands**, which is
  correct and deliberate — the embeddings carry `years_complete`, so the mosaic is reclaimable. Two
  consequences: a "complete mosaic" is a transient state, and a cleaned cell is indistinguishable
  from a never-ingested one *from the mosaic side*. **Judge doneness from the embedding store's year
  tag, never from the presence of mosaics.**
- Its reclaim uses the same `s5cmd` path that, operated by hand the same night, reported success
  while leaving residue. **Verify a reclaim by listing the prefix.**
- **The first cell pays a warm-up premium** in per-chunk overhead, amortised across a chained shard
  rather than paid per zone. Quantifying it per cell needs the zone field named above.

---

## 7. Gotchas, and remaining headroom

**RAM budget is load-bearing.** Do NOT raise `_S2_STRIP_BYTE_BUDGET` or reintroduce whole-chunk
cross-chunk prefetch without re-deriving the arithmetic at the constant. The pair ceiling (2× budget)
plus the ~2 GiB prefetch stash is what keeps peak host RAM under 60%. The prefetch MUST skip
pair-budget plans (`_XCHUNK_UNSAFE_STRATEGIES`) — their last strip is not a RAM trough.

**The strip-plan estimator is strategy-only.** `_EST_*` constants pick which safe strategy is
fastest; they are NEVER a RAM bound. Every branch is RAM-safe regardless of estimate accuracy.

**Shared CloudWatch log group across runs.** `--ram-report` and log greps must be scoped tightly with
`--since` / `--until`; a broad window mixes concurrent runs. On-worker 1 s GPU poll files
(`/tmp/gpu_poll.csv`) die with the workers — capture `--report` BEFORE a run's cluster tears down.

**`min_workers: 0` defers worker capacity to the autoscaler.** `ray up` launches only the head, so a
`g6e` worker-capacity shortfall surfaces later at autoscale time, not at launch. The single-AZ pin is
chosen by least-loaded subnet, spread across concurrent clusters; it does not model spot capacity.

**Never boot a real Ray cluster in pytest.** `ray.kill` is wrapped in Ray's auto-init hook — an
unpatched call in a unit test silently starts a local Ray cluster whose init hashes the whole working
directory (multi-GB with scale-test stores present); three concurrent runs ate ~60 GB RAM once. Patch
`ray.kill` in any test that reaches `ActorPool.replace`, and run ONE targeted pytest at a time.

**A cropped chunk's SCL mask stays full-width** even when bands are cropped — strip sizing charges
the mask at `chunk.width` (`mask_width`), and SAR is read full-width so observation-count layers keep
full extent.

**Remaining headroom.** Fleet GPU utilisation sits at ~89–93%; the residual gap is almost entirely
the cold first-chunk-per-worker prologue (~36 s, ~85 chunks on `a60550ae` as the fleet autoscaled
22→30) — a chunk with no predecessor to prefetch from, which the cross-chunk prefetch structurally
cannot reach. **The next structural lever is source-store chunk geometry:** the 4000² storage
chunking drives a ~13 s fixed read amplification, and an inference-aligned geometry chosen before a
global ingestion would cut it for every future run. That is a config choice beforehand and a
re-ingest afterwards — see [`../ingest/ingest-performance.md`](../ingest/ingest-performance.md) §4 for
why the store chunk was not coarsened, which is the same trade seen from the ingest side.

**There is no CI coverage of the CUDA path and that is accepted** — there is no GPU runner and none
is coming. `TestPipelinedGpuLoop` keeps its `skipif`, and that skip is the only standing signal the
gap exists. See [`../../tests/README.md`](../../tests/README.md).

---

## Appendix: where each thing lives in the code

| what | where |
|---|---|
| the batch calculation | `config/inference.py`, `batch_size_for_gpu` |
| its constants (44 GiB, 7,168 pixels, 512 tokens) | `config/inference.py`, `TUNED_GPU_GIB`, `TUNED_BATCH_SIZE`, `TUNED_TOKENS_PER_PIXEL` |
| where an actor applies it | `inference/actors.py`, `InferenceActor.__init__` |
| reading the card's size | `inference/actors.py`, `_gpu_total_gib` |
| the allocator flag | `inference/actors.py`, the `@ray.remote(runtime_env=...)` decorator |
| the checkpoint ladder and its clipping | `inference/sampling.py`, `compute_bin_keys` |
| deepest bucket first | `inference/dataset.py`, `iter_buckets(largest_first=True)` |
| the strip plan and its RAM budget | `inference/inference.py`, `_strip_plan`, `_S2_STRIP_BYTE_BUDGET` |
| the pipelined forward loop | `inference/inference.py`, `_pipelined_gpu_loop` |
| how many actors Ray packs on a card | `inference/scheduling.py`, `FleetDemand.machines` |
| which instance rungs may be opened | `providers/aws/fleet_mix.py`, `GPU_RUNGS` |
| the capacity-aware autoscaler scorer | `providers/aws/autoscaler_scorer.py` |
| the synthetic forward sweep | `profiling/inference/forward_bench.py` |

`batch_size` is read in two different places downstream, both in `inference/inference.py` — the
sub-batch split in `run_inference`, and the pinned host buffers allocated by `_pipelined_gpu_loop`.
That is why the actor narrows it **once**, in its constructor, and stores the result on
`self.config`: scaling at either read site would leave the other on the tuned value, and the two
would disagree only on the card where it matters.
`test_the_actor_reads_no_un_narrowed_config_after_fitting_the_batch` pins that structurally.
