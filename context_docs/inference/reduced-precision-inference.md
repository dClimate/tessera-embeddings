# Reduced-precision inference: FP8 on the L40S, int8 on Intel AMX

*Measured 2026-11-16, us-west-2, on Tessera v2 Large with the production checkpoint. GPU runs used
`torch 2.14.1+cu130`. CPU runs used the `tessera-inference` C++ library on oneDNN v3.13.3.*

## Bottom line

| path | numerics (ADR 012) | speed | cost per Mpx (UK) | verdict |
|---|---|---|---|---|
| **FP8 on g6e (L40S)** | per-value ✓, neighbours ✗ (0.97) | ~1.2–1.3× forward pass, only with hand-written fused quantisation; naive and `torch.compile` versions are slower than bf16 | ~$0.024–0.025 vs $0.030 | planned: g6e stays the workhorse, so this is the next speed work (see Next steps) |
| **int8 on Intel (c8i, AMX)** | per-value ✓, neighbours ✗ (0.98) | ~1.45–1.6× projected over CPU bf16 | spot ~$0.039–0.048, against a bar of $0.039 (g6e) or $0.054 (g5) | at best reaches the g6e bar; viable as spot overflow only where we fall back to g5 |

- **Fleet plan.** As of 2026-11-16 the g6e (L40S) remains the workhorse until further notice: we expect to get a substantial number of them. That makes FP8 on the g6e the path to build.
- **Which bar applies.** The CPU path was to be pursued only within 1.3× of GPU cost. Against the g6e that bar is **$0.039/Mpx**, which projected int8 on c8i spot ($0.039–0.048) reaches only at its best case; FP8 on the g6e would lower it to about $0.031–0.033. g6e capacity was effectively unavailable while these measurements were taken, and wherever that recurs the fallback GPU is the g5.2xlarge, with a bar of **$0.054/Mpx** that int8 clears.
- **The neighbour gate over-penalises, and has been re-examined.** ADR 012's top-20 gate (0.9940) is failed even by bf16 on a different kernel stack: the CPU library scores 0.989–0.990, and bf16 under `torch.compile` scores 0.988. The churn is swaps among near-tied neighbours: counted tie-tolerantly, every evaluated change keeps 0.999 or more. See [Calibrating the gate](#calibrating-the-gate).
- **Downstream accuracy is unaffected within a store, and at risk only across a change.** On labelled pixels, FP8 and int8 move kNN and linear-probe accuracy by no more than matched random noise does (within about ±0.2 points). The one consumer that loses accuracy is a classifier trained on embeddings from one side of a change and applied to the other.
- **Open question: is g5 fallback consistent enough?** If g6e capacity falls short and a g5 fills part of an FP8 store, that store mixes FP8 and bf16 pixels, which is the across-a-change case. At FP8's size the measured loss is small, but FP8's error has a shared component that the neighbour metrics do not see, and it is untested on real kernels and on more than two tiles.

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
- **On spot:** applying the 1.45–1.6× range to the measured bf16 spot cost projects c8i.8xlarge spot at **$0.039–0.048/Mpx**. It clears the 1.3× g5 bar of $0.054 and is roughly at parity with g5 on-demand, but only its best case reaches the 1.3× g6e bar of $0.039.
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

- **So the churn is what unstructured per-pixel noise of that size produces**, and any change of that size reshuffles near-tied neighbours. The swapped neighbours are interchangeable near-duplicates. The error itself is not purely unstructured, though: FP8 and int8 carry a small shift shared across pixels, which leaves neighbours alone and so does not show here (next section).
- **ADR 012's two gates are not calibrated against each other.** The neighbour gate is roughly 10× stricter than the per-value floor for this kind of change.

## Calibrating the gate

Run on the same lattice pixels, against the production store. FP8 and int8 are the simulated variants, regenerated on 10,000 pixels per tile; the strict overlaps reproduce the tables above.

### Tie-tolerant neighbours

Score the changed embeddings' top-20 in the reference space: the share that falls inside the reference top-25.

| variant (UK) | mean cosine | strict top-20 | inside reference top-25 |
|---|---|---|---|
| bf16 on CPU | 0.999987 | 0.9894 | 1.0000 |
| int8 per-token | 0.999954 | 0.9796 | 0.9999 |
| FP8 E4M3 per-token | 0.999901 | 0.9709 | 0.9994 |
| int8 per-tensor | 0.999919 | 0.9703 | 0.9993 |
| noise, cosine 0.9999 | 0.999868 | 0.9702 | 0.9995 |
| noise, cosine 0.9995 | 0.999465 | 0.9435 | 0.9934 |
| noise, cosine 0.999 | 0.998969 | 0.9234 | 0.9840 |
| noise, cosine 0.995 | 0.994999 | 0.8322 | 0.9125 |

- **Iowa agrees to within 0.002** on every real variant.
- **The width matters.** Top-40 stays at 1.0000 until noise reaches cosine 0.995, so it is too loose. Top-25 separates the evaluated changes from noise five times larger.
- **The new neighbours are barely worse.** FP8's top-20 lose 0.000013 of reference cosine on average, a twentieth of the 0.00024 gap between the 20th and 21st neighbour.

### Noise ladder on labelled pixels

- **Labels:** USDA Cropland Data Layer 2024 in Iowa (corn, soybeans, developed, forest, pasture) and ESA WorldCover 2021 in the UK (trees, grass, cropland, built-up).
- **Split:** leave-one-window-out over the nine lattice windows, 36,000 queries per tile against 64,000 training pixels per fold, so a query never meets pixels from its own field.
- **Reference accuracy:** kNN-20 87.5% (Iowa) and 89.5% (UK); linear probe 86.6% and 91.2%.

Change in accuracy, in points, when isotropic noise is added to every pixel:

| noise (mean cosine) | within ±1 | inside top-25 | kNN-20, Iowa / UK | probe across the change, Iowa / UK |
|---|---|---|---|---|
| 0.9999 (FP8's size) | 0.995–0.998 | 0.999 | −0.08 / −0.06 | −0.05 / −0.04 |
| 0.9995 | 0.81–0.84 | 0.992–0.993 | −0.07 / −0.02 | −0.37 / −0.16 |
| 0.999 | 0.64–0.68 | 0.981–0.983 | −0.05 / −0.25 | −0.57 / −0.34 |
| 0.995 | 0.32–0.35 | 0.90–0.91 | −0.04 / −0.21 | −2.50 / −1.42 |
| 0.99 | 0.23–0.25 | 0.82–0.84 | −0.06 / −0.16 | −5.19 / −2.68 |
| 0.9 | 0.08 | 0.30–0.31 | −1.24 / −0.82 | −35.4 / −21.8 |

- **kNN classification barely moves** until cosine 0.9. A probe retrained on the changed store is as flat (within about ±1 point down to 0.95, with about ±0.2 of refit jitter).
- **"Across the change" is the sensitive case:** a probe trained on reference embeddings and applied to changed ones. Its loss starts between cosine 0.9999 and 0.9995.
- **ADR 012's ±1 envelope (≥ 99.5%) trips at that same point**, and so does top-25 at 0.995. For unstructured error the per-value envelope is well placed and a neighbour gate adds nothing to it.

### The evaluated changes on the same test

| variant (10,000 px per tile) | kNN-20, Iowa / UK | probe across the change, Iowa / UK | error coherence, all pixels |
|---|---|---|---|
| bf16 on CPU | +0.02 / 0.00 | 0.00 / −0.10 | 0.01 |
| int8 per-token | +0.11 / +0.01 | −0.08 / −0.13 | 0.11–0.15 |
| FP8 E4M3 per-token | +0.14 / +0.05 | −0.04 / −0.21 | 0.27–0.28 |
| noise, cosine 0.9999 | +0.10 / −0.02 | −0.20 / −0.14 | 0.01 |

- **None is worse than matched noise.** Standard errors are 0.05–0.09 points and naive, since they ignore spatial correlation.
- **Coherence** is the length of the mean error vector over the mean error length: about 0.01 for unstructured noise, 1 for a shift every pixel shares. Within k-means clusters of about 150 pixels, where noise gives about 0.09, FP8's median is 0.38–0.39 and int8's is 0.18–0.21.
- **FP8's error is therefore about 7% shared shift by energy**, and int8's less. bf16 on a different kernel stack has none.

### Structured-error controls

Synthetic errors applied to the production embeddings, to see what each measure catches.

| control | per-value envelope | inside top-25 | downstream effect |
|---|---|---|---|
| noise in the thinnest-optical decile only, cosine 0.999 there | fails (±1: 0.96) | overall 0.997–0.998; in the decile 0.976–0.985 | none measurable |
| one shift shared by that decile, cosine 0.99 there | fails | in the decile 0.82–0.91, against 0.66–0.72 for noise of that size | probe across the change ±3 points in the decile |
| noise in the top 8 principal components, cosine 0.9999 | fails (±1: 0.98) | 0.991–0.993 | none |
| one class pushed towards another, cosine 0.9999 in that class | passes | passes: 0.9996–1.0000 overall and in the class | across the change, UK cropland loses 1.2 ± 0.4 points of kNN recall and Iowa soybeans 0.3 ± 0.1 |

- **The metric dilutes**, so it has to be scored on the worst stratum, not overall.
- **Shared shifts are under-detected** by neighbour metrics, because pixels that move together keep their neighbours.
- **Low-rank noise is over-penalised**: it fails top-25 at 0.995 with no accuracy change.
- **The class-directed shift passes every gate we have.** Only coherence separates it from noise: 0.69–0.73 over all pixels, against 0.01–0.02.

### What this supports

1. **Keep the per-value envelope as the magnitude gate.** It trips where the first measurable harm appears.
2. **Replace the strict top-20 gate with top-25 at 0.995, scored on the worst stratum.** It passes every benign change evaluated and trips with the envelope.
3. **Add a shared-shift check per cluster** for structured error, since no neighbour metric sees it. Its threshold is not set, and FP8's 0.27 needs a ruling.
4. **Harm appeared only across a change.** That supports ADR 030's store boundary, and it is why g5 fallback inside an FP8 store is an open question.

## Next steps

The g6e stays the workhorse, so the plan is FP8 on the L40S, built by extending the fusion production already has.

**Starting point.** v2's CUDA forward (`_encoder_layer_forward` in `student_v2.py`) already runs each residual add and LayerNorm as one compiled bf16 kernel, and the FFN-up bias and ReLU in the matmul's epilogue ([ADR 030](../decisions/030-forward-pass-reorderings-ship-at-a-store-boundary.md), 5–7% on the L40S and A10G). The baselines above include it, so the bf16 fusion step is done and the remaining gain is FP8's.

1. **Settle the gate.** The first downstream pass is above: on two tiles, simulated FP8 costs nothing within a store. Still needed before FP8 ships:
   - the same tests on real `_scaled_mm` output and on more than two tiles;
   - the two-card comparison scored with top-25 and coherence on one sampling. ADR 030's 0.981 uses a different sampling from the lattice, so FP8 has not been scored against it;
   - a threshold for the shared-shift check;
   - a decision record, because FP8 changes precision rather than order and ADR 030 does not cover it.
2. **Stage 1: FP8 out of the fused add + LayerNorm (Tier A, target ~1.15×).**
   - Replace the compiled add + LayerNorm with a hand-written kernel that also emits its result as E4M3. The encoder is post-norm, so each LayerNorm output feeds both the residual stream and the next GEMM: the kernel writes bf16 for the first and FP8 for the second.
   - QKV and FFN up take the FP8 output through `torch._scaled_mm`, with weights quantised once at load.
   - Hand-written rather than `torch.compile`, which was slower and crashed on varied batch shapes with FP8.
   - The g5 keeps today's bf16 path: the A10G has no FP8.
3. **Measure Stage 1 on g6e.xlarge**: production chunks for speed, the lattice for parity. The measured figure shows how much of the ~1.35× ceiling is left for Stage 2.
4. **Stage 2: FFN down (Tier B, target ~1.25–1.3×), if Stage 1's number justifies it.**
   - First check whether `_scaled_mm` can emit FP8 output with a static result scale. If so, FFN up feeds FFN down with no bf16 round trip and both stay on cuBLASLt.
   - Otherwise a Triton FP8 GEMM with a ReLU and FP8-output epilogue, which has to match cuBLASLt's FFN speed to be worth having.
   - Static scales need a calibration pass. Out-proj (1.10× per layer) is decided on measurement and may stay bf16.
5. **Ship at a store boundary**, as ADR 030 requires of the bf16 fusion.
6. **Open question: is g5 fallback consistent enough?** The A10G has no FP8, so a g5 filling part of an FP8 store writes bf16 pixels beside FP8 ones. What is known:
   - a probe across a change of FP8's size lost 0.04–0.21 points here, the same as matched noise;
   - FP8's shared shift (coherence 0.27) is the kind of error that hurt most across a change in the controls, and it is measured only in simulation;
   - mixed fleets already mix cards: ADR 030 accepts the L40S against RTX PRO 6000 difference inside one store.

   To decide: score real FP8 on a g6e against bf16 on a g5 for the same tiles with the per-stratum top-25 and the shared-shift check, and compare with the two-card baseline. If it is no further apart than two cards on identical code, fallback is fine; otherwise an FP8 store is g6e-only.

Parked while the g6e is the workhorse:

- **int8 in `tessera-inference`.** About 200–300 LoC of C++: per-channel s8 weights at load, per-token activation quantisation inside each thread's block, and oneDNN int8 matmul with dequantise, bias and ReLU fused in. Validate with an end-to-end c8i.8xlarge spot run. Independent cheap pickup: set `TC_BLOCK_S2=8`, since the profiled runs used S2 blocks of 4 pixels because our wrapper leaves it unset.
- **Graviton check:** `benchdnn` int8 on a c9g.4xlarge with an ACL-built oneDNN, about half a day.

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

- **Gate calibration scripts** are not yet uploaded to that prefix: `tie_tolerant.py` (variants and neighbour metrics), `labels.py` (labels and observation counts), `ladder.py` (noise ladder and controls), `coherence.py` (cross-change probe and coherence).
- **Labels:** CDL 2024 from CropScape's coverage endpoint, sampled at pixel centres; WorldCover 2021 v200 from `s3://esa-worldcover`.
- **Lattice inputs** are in `s3://arbol-tessera-inputs-dev/perf/fp8-lattice/`.
- **Mosaics** (calendar 2024) are at `s3://arbol-tessera-inputs-dev/mosaics/cpu-ab-{uk,iowa}`. The production stores are at `s3://arbol-tessera-embeddings-dev/cpu-ab/embeddings/`.
- **CPU library integration** is on branch `feature/cpu-tc-encoder`: `TcEncoder`, thread and affinity fixes, and the plain runner's `--skip-ingest`.
- **Caveats:**
  - The int8 numbers are a simulation plus a projection. The library has not been changed.
  - $/Mpx figures use dense UK pixels; sparser regions shift both sides.
  - Spot prices are as of 2026-11-16.
  - The gate calibration uses two tiles, one noise draw per level, simulated FP8 and int8, and synthetic controls. WorldCover labels are from 2021 against 2024 embeddings, and UK cropland is a weak class (32% kNN recall).
  - The bf16 CPU variant in the calibration ran on an Apple M2, a third kernel stack; it scored the same 0.989–0.990 as the c8i run.
