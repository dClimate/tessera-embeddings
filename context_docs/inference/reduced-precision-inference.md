# Reduced-precision inference: FP8 on the L40S, int8 on Intel AMX

*Measured 2026-11-16 to 2026-11-19, us-west-2, on Tessera v2 Large with the production checkpoint. GPU runs used
`torch 2.14.1+cu130`; CPU runs used the `tessera-inference` C++ library on oneDNN v3.13.3.*

## Bottom line

- **FP8 on the L40S is not adopted.** Built as it would ship, the Stage 1 forward (QKV and the first FFN
  projection in E4M3) runs at **0.83–0.91× bf16** at the 64-pixel chunks production uses
  ([ADR 042](../decisions/042-chunked-graph-replay-ships-above-the-two-card-yardstick.md)). With every
  quantisation free its ceiling is 1.02×; all four linears could reach at most 1.13×. The bar was 1.08×. It also
  failed the accuracy gate it prompted ([ADR 043](../decisions/043-a-per-class-gate-for-precision-changes.md)).
- **int8 on Intel AMX is parked.** Projected c8i spot cost is $0.039–0.048/Mpx. That clears the g5's bar
  ($0.054) but reaches the g6e's ($0.039) only at its best case. It is worth revisiting only if the g6e becomes
  scarce and the g5 is the fallback.
- **Both pass ADR 012's per-value envelope and fail its top-20 neighbour gate.** The neighbour churn is
  swaps among near-tied neighbours, which matched random noise reproduces exactly. The harm that matters is
  concentrated in single classes, for a classifier fitted on one side of a change and applied to the other,
  which is what ADR 043's gate measures.

## FP8 on the L40S: why it does not pay

**The original case.** On 896,000-row batches FP8 multiplies alone were 1.64× bf16 per layer (FFN up 1.85×, QKV
1.42×), with the activations already in FP8. Multiplies were 66% of the forward, so free quantisation would
have given about 1.35×. Quantising each multiply's input in its own pass cost 16–65 ms against 4–20 ms per
multiply, so naive FP8 ran at 0.33–0.58×. FP8 therefore always depended on hand-written kernels that fold the
quantising into the preceding LayerNorm. Those kernels were estimated at ~1.15× (QKV and FFN up only) or
~1.25–1.3× (all four linears, with fixed activation scales). `torch.compile` gave no speedup, and its FP8 path
crashed on varied batch shapes.

**At production chunk sizes** (2026-11-19, one g6e.4xlarge, the production worker environment): since chunked
replay, an L40S multiplies 64 pixels × T steps, at most about 7,200 rows. The Stage 1 forward, built as it would
ship (`F.scaled_mm`, row-wise scales, weights quantised once at model build), measured 0.83–0.91× bf16 over seven
Iowa bucket shapes (0.889× overall). Quantising the first layer's QKV as well measured 0.873×. Each multiply
inside a CUDA graph, in ms ("free" = activations already quantised, the best a fused kernel could do):

| T | QKV bf16 / FP8 free | FFN up bf16 / FP8 free | out-proj bf16 / FP8 free | FFN down bf16 / FP8 free | quantising one row set |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.030 / 0.029 | 0.044 / 0.043 | 0.016 / 0.014 | 0.047 / 0.028 | 0.026 |
| 112 | 0.096 / 0.082 | 0.122 / 0.140 | 0.030 / 0.030 | 0.094 / 0.062 | 0.059 |

- **FP8's gain is a large-multiply effect.** FP8 doubles the tensor cores' arithmetic rate, which pays only when
  arithmetic dominates. At chunk sizes QKV gains at most 1.17× and FFN up nothing; bf16 already runs FFN up's
  bias and ReLU inside the multiply. Quantising a row set still costs about as much as the multiply it feeds.
- **Ceilings, with every quantisation free:** 1.02× for Stage 1 and 1.13× for all four linears, which is mostly FFN
  down. Reaching 1.13× needs FFN up to emit FP8 with fixed activation scales: a Triton kernel plus a calibration
  pass, and the size that also lost more accuracy.
- **Going back to whole batches does not help.** Chunking makes the bf16 forward 1.15–1.19× faster on real
  buckets ([`v2-chunked-graph-replay.md`](v2-chunked-graph-replay.md) §3). Whole-batch FP8 would need about
  1.24–1.29× over whole-batch bf16 to clear 1.08× against today's forward: past the fused-LayerNorm estimate,
  and at the edge of the all-four estimate.
- **The FP8 multiply is not position-independent.** At 64 × 32, 48 and 88 rows, the same rows moved within a
  batch give outputs up to 0.031 different, in both accumulation modes. bf16 does not do this. A pixel's FP8
  embedding would depend on which chunk row it lands in, and every determinism check in this project assumes it
  does not.

Graph capture, replay-equals-eager and NaN isolation all held. Mean cosine to bf16 at chunk sizes was 0.99991.

## FP8's accuracy cost

- **Per value.** E4M3 with row-wise scales: 99.82–99.84% of int8 values within ±1, worst cosine 0.99964, drift
  1.9%. Per-tensor scales are similar. E5M2 fails the envelope (95.7–97.4% within ±1).
- **Six tiles, real kernels.** ADR 012's envelope held on every tile, mixed stores included: at least 99.58%
  within ±1, worst cosine 0.99940, drift at most 2.6%. Ecuador with all four linears came closest. Across the
  change, a linear probe moved at most 0.17 points per tile, past the metric's own noise only on Ecuador and Kansas.
  kNN-20 stayed within noise.
- **Per class it fails ADR 043's gate.** At Stage 1 it fails Ecuador wetland recall (0.84 ± 0.18). It leaves
  Kansas developed kNN precision (0.51 ± 0.17) and Finland grass recall (0.52 ± 0.09) unresolved. With all four
  linears it fails Ecuador cropland recall (0.65 ± 0.15) and leaves cells unresolved on Kansas, Finland and
  Punjab (up to 1.11 ± 0.39). The tripwire fires on Kansas sorghum at both sizes. UK and Iowa pass.
- **The error is about 7% shared shift by energy** (coherence 0.25–0.29, against about 0.02 for noise). That
  shift is what costs single classes.
- **A store both cards write to** (A10G bf16 next to L40S FP8) loses what a bf16 and an FP8 store do. Emulating
  FP8 on the A10G removes the shift inside the store, but not between stores, and adds card-specific rounding
  to every g5 path.

## int8 on Intel AMX (parked)

Granite Rapids' AMX does bf16 and int8 but not FP8, which needs Diamond Rapids. AWS does not offer Diamond
Rapids.

- **Parity (simulated):** per-token activation and per-channel weight int8 keeps 99.96–99.98% within ±1, worst
  cosine 0.99987 and top-20 overlap 0.98, close to bf16 on CPU. Per-tensor int8 is clearly worse.
- **Speed:** `benchdnn` at the library's per-thread shapes gives int8 1.9–3.0× bf16 per multiply. The four
  multiplies are 74% of the library's time, so the projection is **1.45–1.6× end to end**.
- **Cost against the bar** (1.3× GPU cost per Mpx on dense UK pixels):

  | | $/Mpx |
  |---|---|
  | g6e.xlarge (L40S), on demand | $0.030 (bar $0.039) |
  | g5.2xlarge (A10G), on demand | $0.042 (bar $0.054) |
  | c8i.8xlarge bf16, spot | $0.062–0.069 |
  | c8i.8xlarge int8, spot, projected | $0.039–0.048 |

- **To pick it up:** about 200–300 lines of C++ in `tessera-inference` (s8 weights at load, per-token
  quantisation inside each thread's block, oneDNN int8 matmul with dequantise, bias and ReLU fused), then an
  end-to-end c8i spot run. Setting `TC_BLOCK_S2=8` is a cheap, separate pickup; the profiled runs used S2
  blocks of 4 pixels because the wrapper leaves it unset.
- **Graviton5 (c9g)** has no SME. bf16 cannot reach the bar, and int8 might on spot. That needs a half-day
  `benchdnn` run at these small shapes.

## What the neighbour gate measures

These findings bear on ADR 012's gates, not only on FP8:

- **Neighbours are near-ties.** A pixel's 20th and 21st neighbours differ by 0.00024 of cosine at the median.
  Isotropic noise at FP8's size (mean cosine 0.9999) gives top-20 overlap 0.9704, against FP8's 0.9708. ADR 012's
  0.9940 top-20 gate is about 10× stricter than its per-value floor, and bf16 on another kernel stack fails it
  (0.988–0.990).
- **A tie-tolerant count separates real changes from noise.** It counts the share of a pixel's new top-20 that
  stays inside the reference top-25. Every evaluated change keeps ≥ 0.999 on two tiles and ≥ 0.995 over the six;
  noise five times FP8's size scores 0.993. Inside one class or window FP8 falls to 0.972, so a per-stratum
  threshold would need its own calibration.
- **For unstructured error the per-value envelope is well placed.** On labelled pixels, a probe applied across
  the change first loses accuracy between noise cosine 0.9999 and 0.9995, which is where the ±1 envelope trips.
- **Structured error escapes every tile-level measure.** Shifts shared by many pixels keep their neighbours. A
  class pushed towards another passes the envelope, the neighbour counts and a tile bound while costing that
  class 1–2 points. This is why ADR 043 gates per class.

## Artifacts

- **FP8 at chunk sizes:** `s3://global-tessera-inputs-dev/perf/fp8/` (account `global-tessera-dev`):
  `scripts/bench_fp8.py` (the forward), `scripts/bench_gemm.py` (each multiply and the position check),
  `scripts/te-fp8-code.tgz` (the code measured), and `out/`. The code is also in this repository's history,
  reverted in `614ce448`.
- **Six-tile accuracy and the gate's calibration:** `s3://arbol-tessera-inputs-dev/perf/fp8-mixed/` (per-card
  embeddings, labels, scripts, per-class scoring under `perclass/`; `scripts/perclass_cal.py` and
  `perclass_rule.py` became `scripts/precision_gate/calibrate.py` and `apply_gate.py`, whose reports carry
  each tile's z_fam and gated cells). Re-scored with the converged probe in
  `perf/fp8-rebase/rebase-20261118/record/`. The tiles' mosaics are in `mosaics/fp8-*` and `mosaics/cpu-ab-*`.
- **Early FP8 and int8 measurements, neighbour calibration:** `s3://arbol-tessera-inputs-dev/perf/cpu-ab/`
  (`scripts/`, `gate-calibration/`). CPU library integration is on branch `feature/cpu-tc-encoder`.
- **Caveats:** the int8 figures are a simulation plus a projection, and the library is unchanged. $/Mpx uses
  dense UK pixels, and spot prices are as of 2026-11-16.
