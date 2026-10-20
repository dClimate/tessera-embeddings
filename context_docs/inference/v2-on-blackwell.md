# v2 on Blackwell: the g7e evaluation and the fused encoder layer

Measured on 2026-10-20 in global-tessera-dev, to decide whether `g7e` (RTX PRO 6000 Blackwell) can join the
fleet for a global v2 run, and what else makes v2 faster. Outcome: once the workers run PyTorch 2.14.1, a
`g7e.2xlarge` does 1.96× an L40S's v2 work at 1.81× the price; two changes measured here add 12% on it and
6–7% on an L40S or A10G; and on the day there was no `g7e` capacity in us-west-2 at all.

## 1. Setup

| instance | card | VRAM | vCPU / RAM | power limit | max SM clock | $/h, us-west-2 on demand |
|---|---|---:|---:|---:|---:|---:|
| `g6e.xlarge` | L40S | 48 GB | 4 / 32 GiB | 350 W | 2,520 MHz | 1.861 |
| `g7e.2xlarge` | RTX PRO 6000 Blackwell Server | 96 GB | 8 / 64 GiB | 600 W | 2,430 MHz | 3.363 |
| `g7.2xlarge` | RTX PRO 4500 Blackwell Server | 32 GB | 8 / 32 GiB | 165 W | — | 2.520 |
| `g5.2xlarge` | A10G | 24 GB | 8 / 32 GiB | 300 W | 1,710 MHz | 1.212 |

`g7e.2xlarge` is the smallest `g7e`; the larger sizes cost more per GPU. Every box booted the
`dev/global-tessera-pytorch-2-14` worker image (PyTorch 2.14.1+cu130, driver 595.91). The v1.1 rounds ran
`upgrade/pytorch-2-14` at `fb0cef2b`, which predates the day-of-year table; the v2 rounds ran it merged
with `main` (`a426171d`); §3 ends with this branch's own code.

One process per card drives `InferenceActor` exactly as Ray does, minus Ray, over the same tiles of
`mosaics/iowa_epsg5070` (calendar 2024, 263 optical dates, ascending radar) and one tile of
`m10_parity_15S` (both orbits). The figure of merit is inference-loop tokens per second over identical
strips: every card cuts a tile into the same strips, so the ratio is per strip, not per run. "Steady
state" drops each run's first strip, which carries a one-time compile that a production actor pays once
per process. The `g7e` sat in us-east-1, the only region with capacity, reading us-west-2 data, so its
tile wall times include cross-region reads; in-loop figures do not. Harness and raw results:
`s3://global-tessera-inputs-dev/perf/g7e/` (`bench.py` for the pipeline, `bench_v2.py` for the forward
pass, `round4.sh` for the final round, `odcr.py` for capacity).

## 2. The g7e against the fleet's cards

**Today's workers cannot use Blackwell at all.** They install PyTorch 2.5.1+cu121, whose kernels cover
`sm_50`–`sm_90`; on both Blackwell cards the first matmul fails with `no kernel image is available for
execution on the device`. 2.14.1+cu130 ships `sm_120` and runs the pipeline unchanged. Adding the card
to the fleet also needs a `GpuRung` in `providers/aws/fleet_mix.py`: its throughput per vCPU, the
quantity the rungs are ordered by, ties the L40S's.

| | v1.1, 14 Iowa strips | v1.1, m10 tile | v2, 9 Iowa strips | cost per token vs `g6e.xlarge` |
|---|---:|---:|---:|---:|
| RTX PRO 6000 | 2.01× | 1.94× | 1.96× (1.88–2.09 per strip) | 0.90 (v1.1), 0.92 (v2) |
| L40S | 1.00 | 1.00 | 1.00 | 1.00 |
| RTX PRO 4500 | 0.72× | 0.74× | 0.73× | 1.87 |
| A10G, batch 3,593 | 0.42× | — | — | 1.56 |

**Neither a bigger batch nor a second actor helps the RTX PRO 6000.** Doubling the batch to 14,336 gave
+0.4% (v1.1, six tiles) and +0.5% (v2), at 16.5 GiB of 96. Two actors sharing the card ran 10% slower
combined than one.

**The card is full, not throttled.** It held a median 2,407 MHz of its 2,430 maximum while drawing
572–595 W of 600; the power-cap flag is set most of the time but costs about 1% of clock. The L40S, by
contrast, sits at 1,760–2,030 MHz on v1.1 and 1,460 MHz on v2 against its 350 W limit. DCGM on v1.1:

| | SMs active | tensor pipes active | DRAM active |
|---|---:|---:|---:|
| RTX PRO 6000 | 0.99 | 0.50 | 0.63 |
| L40S | 0.99 | 0.43 | 0.70 |

A sustained BF16 GEMM reaches 412 TFLOPS on the RTX PRO 6000 at 8192³, 309–361 at v1.1's layer shapes and
310–322 at v2's (L40S: 186, 131–137, 127–138); there the power cap does bite, at 2,220–2,280 MHz. The
pipeline's transformer layers reach about 190 effective TFLOPS on the RTX PRO 6000 and 100 on the L40S,
so the faster card sits further below its matmul ceiling. The difference is the work between the matmuls.

## 3. v2's forward pass: where the time goes, and the fused add + LayerNorm

GPU time of one v2 forward at a (160, 64) bucket, B = 7,168, on today's path:

| | RTX PRO 6000 | L40S |
|---|---:|---:|
| matmuls (already at the card's GEMM rate) | 67.5% | 62.9% |
| residual adds | 12.5% | 13.9% |
| attention (FlashAttention, head-dim-192 template for 160) | 9.0% | 11.6% |
| LayerNorm | 8.0% | 8.5% |
| other | ~3% | ~3% |

PyTorch runs `nn.TransformerEncoderLayer` in inference as one fused operator the compiler cannot open.
Turning that fast path off costs 28% on both cards; compiling the layers, each backbone or the whole model
with it off recovers half of that and stays 12–16% slower than today. The October attempt on v1.1 found
the same: compiling the layers was 3–5% slower
([`inference-on-gpus.md`](inference-on-gpus.md) §2).

`student_v2._encoder_layer_forward` calls the fast path's kernels one by one: one QKV projection,
`scaled_dot_product_attention`, the output projection, the FFN matmuls with bias and ReLU in the epilogue.
Alone it gives today's output (int8 100% identical on all three cards) at 1.00× on the RTX PRO 6000, 0.96–0.98×
on the L40S and **0.85× on the A10G, so it never ships without the fusion**. Joining each residual add
to its LayerNorm in one compiled kernel, with `emulate_precision_casts` so the sum rounds where eager's
does:

| | forward, six bucket shapes | real pipeline, steady state |
|---|---:|---:|
| RTX PRO 6000 | 1.087× | 1.054× |
| L40S | 1.046× | 1.067× |
| A10G, batch 3,593 | 1.063× | 1.068× (one tile) |

The kernel compiles once per process: 120 s on a fresh machine, about 2 s once the compiler's on-disk cache
is warm, the same order as v1.1's GRU step (150–200 s fresh). On the RTX PRO 6000 the pipeline
gains less than the forward pass because the faster forward leaves the card waiting on CPU prep (§4).

**This branch's code, run unpatched**, reproduced the measured arms bit for bit on the RTX PRO 6000 and the
L40S (three tiles each, 100% of int8 values identical) and ran 11.2% and 7.7% faster than today, steady
state. What the change does to outputs, and why it ships at a store boundary, is
[ADR 030](../decisions/030-forward-pass-reorderings-ship-at-a-store-boundary.md).

## 4. Prep workers by host size

Without a recurrent layer, v2 is fast enough that the RTX PRO 6000 waits on CPU batch preparation: its SMs
were active 0.93–0.94 of the inference loop on v2, against 0.98 on the L40S. Four prep workers instead of
two, on the same strips:

| | 4 workers | 4 workers + fused add + LayerNorm |
|---|---:|---:|
| RTX PRO 6000 (8 vCPU) | +4.3% (SMs active 0.97) | **+12.0%** |
| L40S (4 vCPU) | −0.6% | +6.4% |
| A10G (8 vCPU) | +0.3% | +6.8% |

Outputs are bit-identical at any depth on all three cards, since the workers change only scheduling.
`config.inference.prefetch_depth` therefore runs one worker per two vCPUs, from two to four: four is the
deepest measured, and a `g6e.xlarge` keeps its two.

## 5. Outputs across cards

Every distinct GPU model picks its own kernels, with its own tile shapes and accumulation order, so the
same BF16 forward rounds differently and int8 quantisation turns values near a boundary into one-level
changes. On one card a doubled batch changed nothing (one tile 100%, one 99.96% identical); a card change
always does. Two tiles each, the same code on both sides:

| | int8 identical | mean cosine distance | top-20 neighbours | same k-means cluster |
|---|---:|---:|---:|---:|
| v1.1: RTX PRO 6000 vs L40S | 94% | 1.6e-5 | 0.985 | 99.85% |
| v1.1: RTX PRO 4500 vs RTX PRO 6000 | 92–93% | 1.8e-5 | 0.985 | 99.85% |
| v1.1: A10G vs L40S | 88–90% | 2.6e-5 | 0.982 | 99.82% |
| v2: RTX PRO 6000 vs L40S | 97.2–97.5% | 1.1e-5 | 0.981 | 99.85% |
| v2: fused add + LayerNorm vs today, one card | 98.1–98.4% | 7.2e-6 | 0.985 | 99.9% |

Each tile is computed whole on one card, so in a mixed store these differences fall between tiles, never
inside one. The A10G fallback rung, which already writes into stores whenever `g6e` is refused, moves
outputs more than the RTX PRO 6000 does.

## 6. Capacity on the day

Sampled from 15:20 to 17:00 UTC with capacity reservations created and cancelled at once (`odcr.py`:
a reservation needs only an AZ, refusals are free) and with real launches:

| | us-west-2a/b/c/d | us-east-1 |
|---|---|---|
| `g7e.2xlarge` (and 4x, 8x) | refused, every sample, every AZ | 10 in 1b at 15:27, none after |
| `g6e.xlarge` | refused; one launch landed after 50 minutes of retries | refused |
| `g6.xlarge` | refused | — |
| `g5.xlarge` | available in two AZs | — |
| `g7.2xlarge` | 150 in each of a, b and c | — |

Spot placement scores say nothing here: the account's spot quota is 64 vCPU, so every target scores "unlikely".
The Ray VPC has no subnet in us-west-2d. One day is one sample.

## 7. Not measured

- The two changes together over a whole campaign cell, and the A10G beyond one tile.
- A compiler cache baked into the worker image, which would remove the fresh-machine compile.
- v1.1 with the fused layer. Its encoder uses the same layers, but its published store keeps its
  arithmetic, so it was left alone.
- `g7e` capacity on any other day.
