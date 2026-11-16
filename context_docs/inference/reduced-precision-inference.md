# Reduced-precision inference: FP8 on the L40S, int8 on Intel AMX

*Measured 2026-11-16, us-west-2, on Tessera v2 Large with the production checkpoint. GPU runs used
`torch 2.14.1+cu130`. CPU runs used the `tessera-inference` C++ library on oneDNN v3.13.3.*

## Bottom line

| path | numerics (ADR 012) | speed | cost per Mpx (UK) | verdict |
|---|---|---|---|---|
| **FP8 on g6e (L40S)** | per-value ✓, neighbours ✗ (0.97) | ~1.2–1.3× forward pass, only with hand-written fused quantisation; naive and `torch.compile` versions are slower than bf16 | ~$0.024–0.025 vs $0.030 | parked: real but modest win, on a card we can't get |
| **int8 on Intel (c8i, AMX)** | per-value ✓, neighbours ✗ (0.98) | ~1.45–1.6× projected over CPU bf16 | spot ~$0.041–0.046, against a g5 bar of $0.054 | viable as spot overflow, if the neighbour-gate question resolves |

- **Which bar applies.** The CPU path was to be pursued only within 1.3× of GPU cost. g6e capacity was effectively unavailable throughout, so the GPU we can actually get is the g5.2xlarge, which makes the bar **$0.054/Mpx**.
- **The neighbour gate is the open question for both paths.** ADR 012's top-20 gate (0.9940) is failed even by bf16 on a different kernel stack: the CPU library scores 0.989–0.990, and bf16 under `torch.compile` scores 0.988. The gate effectively demands identical kernels. A downstream evaluation, such as kNN classification on labelled pixels, should decide whether neighbour churn at this level matters.

## Baselines: production engine, bf16, four dense UK chunks of ~4.0M px

| instance | per chunk | throughput | on-demand $/h | $/Mpx |
|---|---|---|---|---|
| g6e.xlarge (L40S) | 225–248 s | 17.2k px/s | $1.861 | **$0.030** |
| g5.2xlarge (A10G, batch 3,593) | 456–533 s | 8.1k px/s (0.47× g6e) | $1.212 | **$0.042** |
| c8i.8xlarge, 16 cores, C++ library, bf16 | — | 3.0k px/s (188 px/s per core) | $1.50; spot $0.67–0.74 | $0.139; **spot $0.062–0.069** |
| c8i.48xlarge, 96 cores, one process | — | 12.0k px/s (125 px/s per core) | $9.00; spot $3.15–4.54 | $0.208; spot $0.073–0.105 |

- **Box size matters.** Smaller c8i instances are about 1.5× more productive per core than one 96-core process.
- **Spot.** g6e spot sat at its on-demand price, so it offered no discount. g5.2xlarge spot was $0.79–0.84.

## How both paths were tested

- **Pixels:** ADR 012's calibration sampling. A 3×3 lattice of 160×160 windows, inset 15%, on two full tiles: dense UK (both radar orbits) and Iowa (ascending radar only). Inputs were built with the production loader and dataset.
- **Reference:** the GPU production store for the same pixels. A fresh bf16 rerun of the production forward pass gives the same-model ceiling.
- **Metrics:** ADR 012's per-value envelope (int8 within ±1 on ≥ 99.5% of values, worst cosine ≥ 0.999, scale drift ≤ 3%). Plus top-20 neighbour overlap among 2,000 sampled pixels, as mean ± sd over 5 samples.
- **What was quantised:** the four transformer linears (QKV, out-proj, FFN up, FFN down), which hold nearly all the FLOPs. Everything else stays bf16.

## FP8 on the L40S (g6e)

### Parity: real `torch._scaled_mm` kernels, full 230,400-px lattice per tile

| vs production store | within ±1 | worst cosine | drift | top-20, UK | top-20, Iowa |
|---|---|---|---|---|---|
| bf16 rerun (ceiling) | 100.0% | 0.99991 | 1.4% | 0.9989 | 0.9995 |
| E4M3, row-wise | 99.82–99.84% ✓ | 0.99964–0.99968 ✓ | 1.8–1.9% ✓ | 0.9708 ✗ | 0.9723 ✗ |
| E4M3, per-tensor | 99.75–99.84% ✓ | 0.99965–0.99967 ✓ | 1.8–2.3% ✓ | 0.9702 ✗ | 0.9721 ✗ |

- **Accumulation mode doesn't matter.** Fast and exact accumulation give the same parity.
- **E5M2 ("bf8") fails even the per-value envelope** (simulated: 95.7–97.4% within ±1), so E4M3 is the only candidate.

### Speed: g6e.xlarge

| measurement | result |
|---|---|
| FP8 vs bf16 GEMM at our shapes (M = 896k) | **1.64× per layer**: FFN 1.85×, QKV 1.42×, out-proj 1.10× |
| production bf16 forward, by share of time | GEMM 66%, flash attention 14%, LayerNorm and elementwise 21% |
| ceiling if quantisation were free | ~1.35× |
| naive FP8 (separate quantise pass per linear) | **0.33–0.58×**: each pass costs 16–65 ms vs 4–20 ms per GEMM |
| whole-model `torch.compile(max-autotune)` | bf16 0.96×; FP8 row-wise **0.90×**, per-tensor **0.76×**; FP8 compiled crashes (CUDA illegal address) on varied batch shapes |
| hand-written fused quantisation, estimated | Tier A (QKV and FFN up fed from a fused LayerNorm, ~200 LoC, 2–3 days): **~1.15×**. Tier B (plus a Triton FP8 GEMM with a ReLU and FP8 output epilogue for FFN down, plus static-scale calibration, a further ~350–400 LoC, 4–6 days): **~1.25–1.3×** |

FP8 applies only to Ada and newer, so the g5's A10G gets nothing from it.

## int8 on Intel (c8i, Granite Rapids AMX)

### Why int8 and not FP8

Granite Rapids' AMX does bf16, fp16 and int8. FP8 needs Diamond Rapids, which AWS doesn't offer: Intel instances top out at Granite Rapids (c8i/m8i/r8i).

### Parity: simulated (fake-quant on CPU), 40,000 lattice px per tile

| vs production store | within ±1 | worst cosine | drift | top-20, UK | top-20, Iowa |
|---|---|---|---|---|---|
| bf16 on CPU (reference) | 99.99% | 0.99990 | 1.0–1.2% | 0.9892 | 0.9900 |
| **int8, weights per-channel + activations per-token** | **99.96–99.98%** ✓ | **0.99987** ✓ | 1.2% ✓ | **0.9797** | **0.9792** |
| int8 per-token, u8 for the FFN-down input | 99.96–99.98% | 0.99986–0.99988 | 1.2% | 0.9797 | 0.9791 |
| int8 per-tensor | 99.75–99.84% | 0.99930–0.99967 | 1.5–1.7% | 0.9679 | 0.9693 |
| FP8 E4M3 per-token (anchor) | 99.83–99.85% | 0.99965–0.99977 | 1.8–1.9% | 0.9704 | 0.9728 |

- **Per-token int8 is nearly indistinguishable from bf16 per value**, and it churns fewer neighbours than FP8. It adds about 1% on top of the CPU kernels' own churn.
- **Per-tensor int8 is clearly worse**, so per-token scaling is required.

### Speed

**`benchdnn`**, sequential oneDNN as the library runs it, at the library's per-thread shapes (M = 192–384). The ratios below are with 32 cores running at once:

| GEMM | int8 vs bf16 |
|---|---|
| QKV (640→1920) | 2.08–2.12× |
| out-proj (640→640) | 2.17–2.20× |
| FFN up (640→2560) | 1.89–1.99× |
| FFN down (2560→640) | 2.12–2.97× |

- **u8 vs s8:** u8 inputs perform the same as s8.
- **Single-core rates:** bf16 reaches 1.9–2.5 TFLOPS per core, int8 3.8–5.5 TOPS.

**Library phase profile** (bf16, Iowa pixels, 32 threads, 0.68 M tok/s):

| phase | share of time |
|---|---|
| FFN down | 29.7% |
| FFN up | 21.7% |
| QKV | 16.5% |
| out-proj | 6.3% |
| attention | 7.3% |
| idle (load imbalance) | 6.1% |
| LayerNorm | 3.3% |
| fusion head | 2.5% |
| oneDNN arg setup | 2.3% |
| embedding GEMMs | 2.1% |
| other | ~2% |

The four transformer GEMMs are **74%** of the time.

**Projection:**
- **Speed:** Amdahl gives about 1.6× with free quantisation. Allowing for per-token quantise and dequantise, which is cheap because each thread's block of 4–8 pixels is already in cache, it comes to **about 1.45–1.6× end-to-end**.
- **On spot:** that projects c8i.8xlarge spot at **$0.041–0.046/Mpx**. It clears the 1.3× g5 bar of $0.054 and is roughly at parity with g5 on-demand.
- **On-demand:** c8i on-demand stays about 2.1–2.3× the g5.

### Graviton

- **Graviton5 (c9g)** is a Neoverse V3 with no SME. Its matrix instructions are 128-bit SVE: BFMMLA peaks at about 0.42 TFLOPS per core and int8 SMMLA at about 0.84 TOPS.
- **bf16 cannot reach the bar.**
- **int8 might**, and only on spot. It needs about 63–92 px/s per core at $0.20–0.29/h for a c9g.4xlarge, against an estimated 72–95 px/s per core. The library would port by rebuilding oneDNN against Arm Compute Library.
- The unknown is efficiency at these small per-thread shapes. That is untested, and a half-day `benchdnn` run on a c9g.4xlarge would measure it.

## Why per-value parity passes but neighbour parity fails

- **Neighbours are near-ties.** A pixel's 20th neighbour sits at a median of 13°, but the cosine gap between the 20th and 21st neighbour is only **0.00024** at the median (0.00003 at p10). Fields and forests are full of near-duplicate pixels.
- **Random noise of FP8's size reproduces FP8's churn exactly.** Isotropic per-pixel noise was added to the production embeddings and re-quantised to int8:

  | noise level | top-20 overlap |
  |---|---|
  | mean cosine 0.99999 | 0.9985 (≈ a bf16 rerun) |
  | mean cosine 0.9999, worst 0.99979 | **0.9704** (≈ FP8's 0.9708) |
  | worst cosine 0.999 (ADR 012's per-value floor) | 0.945 |

- **So the reduced-precision error behaves like unstructured per-pixel noise**, and any change of that size reshuffles near-tied neighbours. The swapped neighbours are interchangeable near-duplicates, so downstream impact is plausibly nil, but untested.
- **ADR 012's two gates are not calibrated against each other.** The neighbour gate is roughly 10× stricter than the per-value floor for this kind of change.

## Next steps

1. **Downstream evaluation (in progress).** kNN classification on labelled pixels: bf16 vs int8 vs FP8. This decides whether ADR 012's neighbour gate should treat unstructured kernel noise differently.
2. **If it passes, int8 in `tessera-inference`.**
   - **Scope:** about 200–300 LoC of C++. Per-channel s8 weights at load, per-token activation quantisation inside each thread's block, and oneDNN int8 matmul with dequantise, bias and ReLU fused in.
   - **Validation:** an end-to-end c8i.8xlarge spot run against the g5 numbers.
   - **Cheap pickup:** set `TC_BLOCK_S2=8`. The profiled runs used S2 blocks of 4 pixels, because our wrapper leaves it unset.
3. **Optional Graviton check:** `benchdnn` int8 on a c9g.4xlarge with an ACL-built oneDNN, about half a day.
4. **FP8 on g6e:** only if g6e becomes the main fleet type again. Tier A fused kernels first.

## Methods and artifacts

- **Scripts** are in `s3://arbol-tessera-inputs-dev/perf/cpu-ab/scripts/`:

  | script | purpose |
  |---|---|
  | `fp8_sim.py` | FP8 and E5M2 simulation |
  | `fp8_gpu.py` | real `_scaled_mm` kernels |
  | `fp8_micro.py` | GEMM microbenchmark and forward profile |
  | `lattice_prep.py` / `lattice_gpu.py` | lattice sampling and FP8 parity |
  | `step0.py` | whole-model `torch.compile` |
  | `int8_sim.py` | int8 simulation and library profile |
  | `bench_int8.sh` | `benchdnn` |

- **Lattice inputs** are in `s3://arbol-tessera-inputs-dev/perf/fp8-lattice/`.
- **Mosaics** (calendar 2024) are at `s3://arbol-tessera-inputs-dev/mosaics/cpu-ab-{uk,iowa}`. The production stores are at `s3://arbol-tessera-embeddings-dev/cpu-ab/embeddings/`.
- **CPU library integration** is on branch `feature/cpu-tc-encoder`: `TcEncoder`, thread and affinity fixes, and the plain runner's `--skip-ingest`.
- **Caveats:**
  - The int8 numbers are a simulation plus a projection. The library has not been changed.
  - $/Mpx figures use dense UK pixels; sparser regions shift both sides.
  - Spot prices are as of 2026-11-16.
