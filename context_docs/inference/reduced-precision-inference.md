# Reduced-precision inference: FP8 on the L40S, int8 on Intel AMX

*Measured 2026-11-16, us-west-2, on Tessera v2 Large with the production checkpoint. GPU runs used
`torch 2.14.1+cu130`. CPU runs used the `tessera-inference` C++ library on oneDNN v3.13.3.*

## Bottom line

| path | numerics (ADR 012) | speed | cost per Mpx (UK) | verdict |
|---|---|---|---|---|
| **FP8 on g6e (L40S)** | per-value ✓, strict neighbours ✗ (0.97), cross-change accuracy ✓ per tile except Ecuador's Stage 1 mixed store (0.2005), ✗ per class on three tiles, which would have shipped as a named exception ([ADR 043](../decisions/043-a-per-class-gate-for-precision-changes.md)) | **0.83–0.91× the bf16 forward** at production chunk sizes; even with quantisation free, at most 1.02× (Stage 1) or 1.13× (all four linears) | none | **not adopted** (2026-11-19): see [FP8 at production chunk sizes](#fp8-at-production-chunk-sizes) |
| **int8 on Intel (c8i, AMX)** | per-value ✓, neighbours ✗ (0.98) | ~1.45–1.6× projected over CPU bf16 | spot ~$0.039–0.048, against a bar of $0.039 (g6e) or $0.054 (g5) | at best reaches the g6e bar; viable as spot overflow only where we fall back to g5 |

- **Fleet plan.** As of 2026-11-16 the g6e (L40S) remains the workhorse until further notice: we expect to get a substantial number of them. That made FP8 on the g6e the path to build, and it was built and measured: since chunked graph replay ([ADR 042](../decisions/042-chunked-graph-replay-ships-above-the-two-card-yardstick.md)) the L40S multiplies 7,400-row chunks, not 896,000-row batches, and at that size FP8 is slower than bf16.
- **Which bar applies.** The CPU path was to be pursued only within 1.3× of GPU cost. Against the g6e that bar is **$0.039/Mpx**, which projected int8 on c8i spot ($0.039–0.048) reaches only at its best case; FP8 on the g6e was projected to lower it to about $0.031–0.033, which the chunk-size measurement rules out. g6e capacity was effectively unavailable while these measurements were taken, and wherever that recurs the fallback GPU is the g5.2xlarge, with a bar of **$0.054/Mpx** that int8 clears.
- **The neighbour gate over-penalises, and has been re-examined.** ADR 012's top-20 gate (0.9940) is failed even by bf16 on a different kernel stack: the CPU library scores 0.989–0.990, and bf16 under `torch.compile` scores 0.988. The churn is swaps among near-tied neighbours: counted tie-tolerantly, every evaluated change keeps 0.999 or more on the two-tile calibration, and 0.995 or more over whole tiles on the six-tile lattice. See [Calibrating the gate](#calibrating-the-gate).
- **Downstream accuracy is unaffected within a single-precision store, and at risk only across a change.** On labelled pixels, FP8 and int8 move kNN and linear-probe accuracy by no more than matched random noise does (within about ±0.2 points). The one consumer that loses accuracy is a classifier trained on embeddings from one side of a change and applied to the other. A store mixing g5 bf16 and g6e FP8 pixels contains such a change: a linear classifier crossing it loses up to 0.2 points on some tiles, and up to 1.0 point on a single class (next bullet).
- **g5 fallback inside an FP8 store would have been accepted, on bf16** ([ADR 043](../decisions/043-a-per-class-gate-for-precision-changes.md)). Across six tiles on real kernels, a linear classifier fitted on one card's pixels and applied to the other's loses nothing beyond the metric's own noise on four tiles, and at most 0.2 points where FP8's shared shift has a small real cost (Ecuador, Kansas). Per class the cost is larger: up to 1.0 point of Kansas winter wheat recall, and 0.55–0.65 on one class each in Iowa and Ecuador with all four linears, which ADR 043 accepts as a named exception. kNN is unaffected everywhere. See [The mixed-pool test](#the-mixed-pool-test) and [Per-class losses across the change](#per-class-losses-across-the-change).

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

### FP8 at production chunk sizes

*Measured 2026-11-19 on one g6e.4xlarge (L40S, PyTorch 2.14.1+cu130, the production worker environment and its
`CUBLAS_WORKSPACE_CONFIG`), with the production checkpoint.*

The figures above were taken on batches of 896,000 token rows. Since chunked graph replay, an L40S runs each
sub-batch in 64-pixel chunks, which at these bucket depths is 64 to 7,200 rows per multiply. The Stage 1 forward
was built as it would ship and timed there:

- **Code.** `F.scaled_mm` with row-wise E4M3 scales on QKV (layers 2–4) and the first FFN projection; weights
  quantised once from fp32 at model build; each activation row quantised in a separate pass (amax, divide, clamp,
  cast); bias inside the multiply; the ReLU after it. Captured in the CUDA graph without change. Commits
  `ea5d2d9e`, `1a068a3f` (the store's precision record), `3664ae55`, reverted in `614ce448`; also
  `s3://global-tessera-inputs-dev/perf/fp8/scripts/te-fp8-code.tgz`.
- **Method.** As [`v2-chunked-graph-replay.md`](v2-chunked-graph-replay.md) §2: card pre-heated 90 s, seven Iowa
  bucket shapes of 7,168 pixels, four rounds, every round timing every arm in rotating order, fastest round per
  shape. Each arm is its own model built through `build_inference_model`. The card's clock moved between 1,335 and
  2,040 MHz across rounds at its 350 W limit; the rotation spreads that over every arm.

**The whole forward:**

| bucket (S2 × S1 steps) | bf16, ms per 64-pixel chunk | FP8 Stage 1 | FP8, first layer's QKV too |
|---|---:|---:|---:|
| 48 × 56 | 1.90 | 0.900× | 0.891× |
| 80 × 56 | 2.57 | 0.897× | 0.877× |
| 88 × 56 | 2.76 | 0.901× | 0.884× |
| 40 × 32 | 1.33 | 0.848× | 0.831× |
| 80 × 32 | 2.14 | 0.895× | 0.881× |
| 80 × 1 | 1.62 | 0.827× | 0.820× |
| 112 × 48 | 3.09 | 0.914× | 0.890× |
| all seven | | **0.889×** | 0.873× |

**Each multiply, at chunk sizes** (64 pixels × T steps; one multiply, inside a CUDA graph, ms). "Free" is FP8 with
the activations already quantised, the best any fused kernel could do:

| T | QKV bf16 | QKV FP8 free | QKV FP8 shipped | FFN up bf16 | FFN up FP8 free | out-proj bf16 / FP8 free | FFN down bf16 / FP8 free | quantising one row set |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.030 | 0.029 | 0.055 | 0.044 | 0.043 | 0.016 / 0.014 | 0.047 / 0.028 | 0.026 |
| 48 | 0.049 | 0.041 | 0.074 | 0.060 | 0.059 | 0.017 / 0.016 | 0.047 / 0.031 | 0.032 |
| 80 | 0.074 | 0.060 | 0.107 | 0.089 | 0.096 | 0.029 / 0.027 | 0.088 / 0.052 | 0.045 |
| 112 | 0.096 | 0.082 | 0.143 | 0.122 | 0.140 | 0.030 / 0.030 | 0.094 / 0.062 | 0.059 |

- **Quantising a row set costs about as much as the multiply it feeds.** At 896,000 rows it cost more (16–65 ms
  against 4–20 ms, above), which is why FP8 always depended on fusing it away. What chunks change is the multiply's
  own gain: 1.42× (QKV) and 1.85× (FFN up) at 896,000 rows, at most 1.17× and none at 7,400. FP8 doubles the
  tensor cores' arithmetic rate, which pays only when a multiply is long enough for arithmetic to dominate.
- **Dropping chunks for FP8 does not help.** Chunked replay makes the bf16 forward 1.04–1.19× faster than the whole
  batch ([`v2-chunked-graph-replay.md`](v2-chunked-graph-replay.md) §3); on real buckets, under 200 tokens per
  pixel, 1.15–1.19×. Whole-batch FP8 would need about 1.24–1.29× over whole-batch bf16 to clear the 1.08× bar
  against today's forward: past the fused-LayerNorm estimate (~1.15×), and at the edge of the Tier B estimate
  (~1.25–1.3×). Both are estimates, never built.
- **The first FFN projection gains nothing from FP8 even when quantisation is free.** bf16 already runs its bias
  and ReLU inside the multiply; FP8 adds the ReLU as its own pass.
- **Ceilings, with every quantisation free.** Summing the free savings over the eight encoder layers and dividing
  by the measured bf16 forward: **1.02× for Stage 1** and **1.13× for all four linears**, where FFN down carries
  most of it. Both are upper bounds, since the two backbones run on parallel streams and overlap. Reaching the
  all-four figure needs FFN down's input to arrive already in FP8, which means FFN up emitting FP8 with fixed
  activation scales (the old plan's Tier B, about 350–400 lines of Triton plus a calibration pass), and all four
  linears is also the size that lost more on the lattice.
- **Against the bar.** The plan's bar was 1.08× on the forward, about 5% of tile time or $18,000 over the nine-year
  v2 campaign. Stage 1 cannot reach it at any amount of kernel work. All four linears could only in the best case,
  at the cost above.

**Two numerical findings, both relevant to any later FP8 work:**

- **The FP8 multiply is not position-independent.** The same rows moved 7 pixels within a batch give different
  bf16 outputs from `F.scaled_mm` at some sizes (64 × 32, 64 × 48 and 64 × 88 rows; up to 0.031), in both
  accumulation modes. The quantiser's codes and scales are identical, and bf16's multiply is position-independent
  at every size tested. A pixel's FP8 embedding would therefore depend on which chunk row it lands in, which every
  determinism check in this project assumes it does not.
- **Accuracy at chunk sizes matches the record.** FP8 Stage 1 against bf16 on random inputs at the seven shapes:
  mean per-pixel cosine 0.99991–0.99992, worst 0.99976. Replaying a captured graph equals the eager forward bit for
  bit, and a NaN or infinite input reaches the output as non-finite without changing any other pixel.

Outputs: `s3://global-tessera-inputs-dev/perf/fp8/out/` (`b2` the forward, `g1`/`g2` the multiplies), scripts
beside them in `scripts/`.

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

Run on the same lattice pixels, against the production store. FP8 and int8 are the simulated variants, regenerated on a 10,000-pixel subset per tile; the strict overlaps reproduce the tables above. The noise ladder uses a larger labelled sample than the variants do, so the variants are scored on their own subset, with a noise row recomputed there for comparison.

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

The variants exist only for the 10,000-pixel subset, so the queries are its labelled pixels (9,789 in Iowa, 9,999 in the UK), not the ladder's 36,000. kNN-20 is leave-one-window-out inside the subset. The probe is fitted on 48,000 reference pixels outside the subset, from the other window rows, and applied to the subset. The noise row is drawn on the same subset and scored the same way, so the table compares like with like. Its values are not the ladder's first row, which has its own larger sample and training set.

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
- **The class-directed shift passes the envelope, the neighbour overlaps and a tile-level bound.** Among the measures in this section only coherence separates it from noise: 0.69–0.73 over all pixels, against 0.01–0.02. The per-class gate rejects it ([Per-class losses across the change](#per-class-losses-across-the-change)).

### The mixed-pool test

Measured 2026-11-17 on the full lattice (230,400 px per tile), with real kernels, to settle g5 fallback in an FP8 store. Six tiles of calendar 2024: the dense UK and Iowa tiles above, plus Ecuador lowlands (cloudy tropics), Punjab (irrigated, arid), Kansas (winter wheat, ascending radar only) and Finland (boreal), ingested for this test. Native FP8 ran on the L40S and on an L4; bf16 on the L40S, A10G and L4. Stage 1 is FP8 on QKV and FFN up only, quantised from the fp32 LayerNorm output as the fused kernel will emit it. "Emulated" is FP8 on the A10G: E4M3 operands through an fp32 GEMM, then the scales. Labels: ESA WorldCover 2021, plus USDA CDL 2024 in Iowa and Kansas.

| pair (ranges over six tiles) | within ±1 | strict top-20 | inside top-25, overall / worst stratum | coherence | cross-change probe / kNN-20, pts |
|---|---|---|---|---|---|
| two cards, bf16 (L40S, A10G, L4) | ≥ 99.998% | 0.999–1.000 | 1.0000 / ≥ 0.9987 | 0.02–0.10 | ±0.01 / ±0.01 |
| two cards, native FP8 (L40S, L4) | ≥ 99.99% | 0.994–1.000 | ≥ 0.9999 / ≥ 0.9986 | 0.01–0.07 | ±0.01 / ±0.01 |
| FP8 change on the L40S, all four linears | 99.58–99.97% | 0.954–0.977 | 0.995–1.000 / 0.972–0.994 | 0.25–0.29 | −0.15 to +0.15 / −0.03 to +0.05 |
| FP8 change on the L40S, Stage 1 | 99.78–99.99% | 0.958–0.979 | 0.997–1.000 / 0.975–0.995 | 0.22–0.27 | −0.20 to +0.20 / −0.05 to +0.01 |
| A10G bf16 vs L40S FP8 (a mixed store) | as the FP8 rows | | | | −0.20 to +0.26 / −0.05 to +0.05 |
| A10G emulated FP8 vs L40S native FP8 | 99.90–99.998% | 0.966–0.987 | 0.998–1.000 / 0.987–0.999 | 0.006–0.009 | ±0.02 / −0.05 to +0.02 |

ADR 012's full cross-config envelope holds on every tile and variant, mixed stores included: at least 99.58% of values within ±1 (floor 99.5%), worst-pixel cosine at least 0.99940 (floor 0.999), per-pixel scale drift at most 2.6% (limit 3%). Ecuador with all four linears is the closest to all three limits.

**Noise control.** Isotropic per-pixel noise matched to FP8's mean cosine after int8 quantisation, five draws per tile and size, scored with the same folds:

| tile | FP8 all four, probe (both directions) | FP8 Stage 1, probe | matched noise, largest probe move (sd) |
|---|---|---|---|
| UK | +0.01 / 0.00 | +0.02 / −0.01 | 0.03–0.04 (0.02) |
| Iowa | +0.04 / −0.03 | +0.01 / 0.00 | 0.06–0.07 (0.03) |
| Finland | +0.02 / −0.01 | +0.03 / −0.03 | 0.05 (0.04) |
| Punjab | −0.01 / +0.04 | −0.01 / 0.00 | 0.03 (0.02) |
| Ecuador | +0.11 / −0.11 | **+0.20 / −0.20** | 0.10–0.13 (0.05–0.06) |
| Kansas | **−0.15 / +0.15** | −0.10 / +0.12 | 0.07–0.08 (0.05) |

- **The fleet's cards are nearly identical on bf16**, and the L4 matches the L40S on FP8. ADR 030's yardstick, "no further apart than two cards on identical code", therefore rejects any precision change, and a bf16 g5 in an FP8 store is exactly the FP8 change.
- **The cross-change metric has its own noise floor**, 0.03–0.13 points depending on the tile, with no change of dtype at all. A move counts as structural when it passes the largest of the five draws. On UK, Iowa and Finland, and on Ecuador and Punjab for the other size, FP8 sits inside the noise.
- **FP8's shared shift has a small real cost on a linear probe on two tiles.** On Ecuador Stage 1 and Kansas at both sizes, the probe moves about 2–4 noise standard deviations, past every draw, by roughly equal and opposite amounts in the two directions. Punjab all four passes its largest draw marginally, but as a gain (+0.04 against 0.03), not a loss. The largest loss in a mixed store is 0.20 points. kNN-20 stays within noise everywhere.
- **Emulated FP8 on the A10G removes the shared shift** (coherence at the noise floor) and holds a mixed store to ±0.02 points. ADR 043 rejects it: it adds card-specific rounding to every g5 path, and does nothing for a classifier applied across a bf16 and an FP8 store, where the same losses appear.
- **Top-25 per stratum is stricter than overall.** Inside one class or window FP8 scores as low as 0.972 (Finland's water), under the 0.995 calibrated over whole tiles.

### Per-class losses across the change

The tile-level bound misses harm to one class: the class-directed control above passes it while costing UK cropland 2 points of kNN recall, because cropland is 3% of the labels. So the same lattice was scored per class, with every threshold calibrated per tile from controls only (2026-11-17, one m7i.16xlarge, about an hour for six tiles). The resulting gate is ADR 043's decision 1.

- **Cells:** each class × consumer (probe, kNN-20) × direction (fitted on the reference and applied to the change, and the reverse) × measure (recall, precision). Loss in points, with a delete-one-block jackknife SE over 144 blocks of 40×40 px. Naive SEs are too small by a large factor because errors cluster in fields.
- **Known good:** two cards on identical bf16, and 20 matched-noise draws per FP8 size. A cell is gated when its median noise SE is at most δ/4 = 0.125. z_fam is the 95th percentile over the 40 draws of the largest loss/SE among gated cells: 2.79–4.45 over 16–40 gated cells per tile.
- **Known bad:** each class with at least 2,000 px shifted towards its most-confused class at cosine 0.9999, 0.99995 and 0.99998 inside the class; and the largest class towards the smallest such class.
- **A cell fails** when it is gated and loses more than δ = 0.5 points with loss/SE above z_fam. **It is unresolved** when it loses more than δ and is either gated with loss/SE at or under z_fam (the candidate's own SE is too large to call it) or ungated with loss/SE above z_fam. An unresolved cell needs a decision.

| tile | gated classes (cells) | z_fam | FP8 all four, worst gated cell | FP8 Stage 1, worst gated cell |
|---|---|---:|---|---|
| UK | 4 (28) | 2.79 | cropland kNN precision 0.25 ± 0.18 | cropland kNN precision 0.31 ± 0.16 |
| Iowa | 5 (39) | 2.83 | **forest probe precision 0.64 ± 0.15** | forest probe precision 0.27 ± 0.10 |
| Ecuador | 5 (35) | 4.45 | **trees probe recall 0.55 ± 0.09** | cropland probe precision 0.35 ± 0.08 |
| Punjab | 4 (26) | 3.58 | trees probe recall 0.39 ± 0.12 | trees probe recall 0.34 ± 0.14 |
| Kansas | 6 (40) | 3.16 | **winter wheat probe recall 0.96 ± 0.21** | **winter wheat probe recall 0.66 ± 0.16** |
| Finland | 2 (16) | 4.21 | trees probe precision 0.12 ± 0.03 | trees probe precision 0.08 ± 0.03 |

Bold cells fail. A mixed store (A10G bf16 against L40S FP8) fails in the same cells by the same amounts, to within 0.04.

- **No known-good variant fails:** all 12 two-card pairs and 239 of the 240 noise draws pass, and one draw is unresolved. None trips the tripwire.
- **The gate rejects most class-directed controls, and flags the rest that do harm.** At cosine 0.9999 the gate rejects the control in 21 of the 25 gated classes that have one. Of the four it does not reject, Finland water's control does no harm and passes; the other three are unresolved, flagged but not failed. Kansas soybeans' and Punjab grass's cost 1.30 ± 0.66 and 0.58 ± 0.36 points in gated cells, and Kansas developed's costs 3.23 ± 0.44 points of probe recall in cells too noisy to gate, where FP8 loses at most 0.47 ± 0.20. UK cropland is rejected at 0.9999 (kNN recall −2.00 ± 0.68, z 3.0) and 0.99995 (−1.07 ± 0.37), and unresolved at 0.99998 (−0.79 ± 0.31, z 2.5): the gate's detection limit on a weak class. Structured error carries a far larger SE than noise: Kansas sorghum's control loses 3.20 ± 1.12 in a cell whose noise SE is 0.07. The large-to-small shift fails on every tile at 0.9999, mostly through the large class's recall when fitted on the changed side, not the small class's precision as expected.
- **FP8's failures are all on the linear probe;** kNN-20 passes every cell.
- **FP8 leaves six cells unresolved,** all on the probe. With all four linears: Ecuador cropland recall 0.79 ± 0.18 (z 4.3 against 4.45), Kansas sorghum precision 0.78 ± 0.30, corn recall 0.68 ± 0.32 and, in a mixed store only, sorghum recall 0.51 ± 0.30 (own SE five times the noise's), and Punjab trees precision 0.53 ± 0.10 (z 5.2, ungated by a noise SE of 0.126 against 0.125). At Stage 1: Ecuador wetland recall 0.79 ± 0.17 (z 4.5, ungated). Punjab (all four) and Ecuador (Stage 1) therefore move from pass to unresolved. Of the known-good variants, 1 of 240 noise draws and none of the 12 two-card pairs is unresolved.
- **Cells too noisy to gate still show losses past 0.5 under z_fam,** which are reported: Finland grass recall 1.09 ± 0.30 (all four), Kansas soybean precision up to 1.30 ± 0.45.
- **The training sample matters more than FP8 for minority classes.** Ten redraws of the 48,000-px training sample on bf16 move per-class recall by a standard deviation of 0.1–0.6 points for large classes and 1–5.6 for minority ones (UK cropland 5.6 on the probe, 3.0 on kNN). Letting δ widen to 3 × that spread would set it at 2–17 points for minority classes and stop the gate rejecting the UK cropland control, so the spread is reported, not used.
- **Gap closure** (a class's mean error projected on the direction to another class's centroid, as a share of the gap) trips for FP8 on every tile except UK and Finland at Stage 1, and for no noise draw. It is reported: a trip asks for a decision, and FP8's exception already is one.

ADR 043 records the outcome: FP8 fails the gate on Kansas (both sizes), Iowa and Ecuador (all four linears), and would have shipped as a named exception that accepts those measured losses and the six unresolved cells.

### What this supports

1. **Keep the per-value envelope as the magnitude gate.** It trips where the first measurable harm appears.
2. **Replace the strict top-20 gate with a tie-tolerant measure, reported rather than gated.** Top-25 at 0.995 over a whole tile passes every benign change evaluated and trips with the envelope. Scored inside one class or window it is stricter (FP8 scores as low as 0.972), so a per-stratum threshold needs its own calibration.
3. **Report a shared-shift (coherence) check per cluster** for structured error, since no neighbour metric sees it. FP8's 0.25–0.29 costs about 0.2 points per tile at most (0.2005 on Ecuador's Stage 1 mixed store) across a change on real kernels, and up to 1.0 on a single class; coherence does not track the size of the harm, so it is reported, not gated.
4. **Harm appears only across a change, and is concentrated in single classes.** The store boundary stays (ADR 043). Cross-change accuracy is the gate, on loss, per tile and per class, with thresholds calibrated against matched noise and synthetic controls so that the metric's own noise floor is not mistaken for harm.

## Next steps

**FP8 on the L40S is closed for Stage 1** ([FP8 at production chunk sizes](#fp8-at-production-chunk-sizes)): since
chunked graph replay its ceiling is 1.02× on the forward. Two things remain open, each for a decision rather than
for work already planned:

1. **Whether to try all four linears.** Its ceiling is 1.13× with every quantisation free and needs FFN up to emit
   FP8 with fixed activation scales, a hand-written kernel and a calibration pass; all four linears also lost more
   on the lattice, and the FP8 multiply's position dependence would have to be solved or accepted first. The
   recommendation is not to.
2. **The gate.** [ADR 043](../decisions/043-a-per-class-gate-for-precision-changes.md)
   decision 1 is the bar for any precision change and does not depend on FP8 shipping. Its tables, and the FP8
   exception's bounds file generated from them (deleted with FP8), come from a linear probe that stopped fitting early
   (fixed in the gate code, commit `af33d0a3`). Re-scored with the probe fitted to convergence
   (`s3://arbol-tessera-inputs-dev/perf/fp8-rebase/rebase-20261118/record/`), the same embeddings give, at Stage 1:
   no failure on Kansas (winter wheat recall was 0.66), a failure on Ecuador wetland recall (0.84 ± 0.18, was
   unresolved at 0.79), and two unresolved cells (Kansas developed kNN precision 0.51 ± 0.17, Finland grass recall
   0.52 ± 0.09). All 12 two-card pairs and 239 of the 240 noise draws still pass. Any later candidate is scored
   with the converged probe.

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

- **Mixed-pool test** is in `s3://arbol-tessera-inputs-dev/perf/fp8-mixed/`: `scripts/mixed_pool_gpu.py` (embeds the lattice per card: bf16, FP8 all four, FP8 Stage 1; native on Ada, emulated on the A10G), `scripts/mixed_pool_score.py` (envelope, neighbours, coherence, cross-change classifiers), `scripts/noise_control.py` (the matched-noise control), `scripts/lattice_prep2.py` and `scripts/labels2.py` (lattice and labels without a reference store), the outputs `mixed_<card>_<tile>.npz`, the labels, `score6_<tile>.txt` and `nc_<tile>.txt`. The per-class scoring is `scripts/perclass_cal.py` (per-pixel predictions for every variant, one npz each under `perclass/<tile>/`), then `scripts/perclass_rule.py` (applies the gate locally from those predictions) with `bootstrap-perclass.sh` and `watch_perclass.sh`. Its reports are `perclass/rule_unres/` (the gate as decided), `perclass/rule_delta05/` (the same gate without the unresolved outcome, from `scripts/perclass_rule_delta05.py`) and `perclass/rule_delta_refit/` (δ widened by the refit spread, rejected). The four new tiles' mosaics are in `s3://arbol-tessera-inputs-dev/mosaics/fp8-{ecuador,punjab,kansas,finland}`.
- **Gate calibration scripts** are in `s3://arbol-tessera-inputs-dev/perf/cpu-ab/scripts/`: `tie_tolerant.py` (variants and neighbour metrics), `labels.py` (labels and observation counts), `ladder.py` (noise ladder and controls), `coherence.py` (cross-change probe and coherence). The variants and the ladder's reports are in `perf/cpu-ab/gate-calibration/` (`variants_<tile>.npz`, `ladder_<tile>.txt`); the neighbour and coherence tables were printed, not saved, and rerun from the variants.
- **Labels:** CDL 2024 from CropScape's coverage endpoint, sampled at pixel centres; WorldCover 2021 v200 from `s3://esa-worldcover`.
- **Lattice inputs** are in `s3://arbol-tessera-inputs-dev/perf/fp8-lattice/`.
- **Mosaics** (calendar 2024) are at `s3://arbol-tessera-inputs-dev/mosaics/cpu-ab-{uk,iowa}`. The production stores are at `s3://arbol-tessera-embeddings-dev/cpu-ab/embeddings/`.
- **FP8 at production chunk sizes** is in `s3://global-tessera-inputs-dev/perf/fp8/` (account `global-tessera-dev`):
  `scripts/bench_fp8.py` (the whole forward, per arm, with the GPU checks), `scripts/bench_gemm.py` (each multiply
  and the position probe), `scripts/te-fp8-code.tgz` (the code measured), outputs under `out/`.
- **CPU library integration** is on branch `feature/cpu-tc-encoder`: `TcEncoder`, thread and affinity fixes, and the plain runner's `--skip-ingest`.
- **Caveats:**
  - The int8 numbers are a simulation plus a projection. The library has not been changed.
  - $/Mpx figures use dense UK pixels; sparser regions shift both sides.
  - Spot prices are as of 2026-11-16.
  - The gate calibration uses two tiles, one noise draw per level, simulated FP8 and int8, and synthetic controls. WorldCover labels are from 2021 against 2024 embeddings, and UK cropland is a weak class (32% kNN recall).
  - The bf16 CPU variant in the calibration ran on an Apple M2, a third kernel stack; it scored the same 0.989–0.990 as the c8i run.
