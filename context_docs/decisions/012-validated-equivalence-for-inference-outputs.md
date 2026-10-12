# 012 — Validated equivalence, not bit-exactness, for inference outputs

**Status:** Accepted (2026-07-16). Cross-config envelope widened 2026-10-06 (repo owner).

## Context

The GPU-saturation optimization campaign (branch `perf/inference-rethink`)
includes changes that make the forward pass faster by *regrouping* its
floating-point work: fusing per-gate matmuls, padding partial sub-batches to
fixed shapes for CUDA-graph capture, and similar regroupings we may adopt.
None of these change the mathematical formula — but all of them change the
*order* in which floating-point sums are reduced, and floating-point addition
is not associative: `(a+b)+c ≠ a+(b+c)` at the last bit. In BF16 (8 mantissa
bits) that last bit is ~0.4% relative, and the model propagates such wiggles
through four transformer layers and a recurrent GRU.

(As it landed, this campaign shipped no reduction-reordering change: the
inference-loop and pipelining work is bit-identical, the positional-encoder
rewrite is bit-identical, and the GRU already runs as a cuDNN `nn.GRU` fused
in the builder. This ADR governs the reordering changes we expect to make
next — it is the policy, not a description of the current diff.)

The question this ADR answers: **must optimized code produce bit-identical
embeddings, or is a validated tolerance acceptable?**

Three facts drove the decision.

**1. How drift lands in the saved artifact.** Embeddings are stored int8
with a per-pixel scale = absmax/127, so the quantization grid step is
~0.8% of each pixel's max channel value. A BF16-level wiggle (~0.4%) is
below one grid step, but values near a rounding boundary flip by ±1 int8
level. Expected profile for a reduction-reorder change: the large majority
of int8 values bit-identical, a small percentage off by exactly ±1,
essentially none off by more, per-pixel scales drifting < 0.1%.
Quantization itself already imposes ±0.5-level rounding error on *every*
value, so ±1 flips on a few percent of values sit inside the noise floor
the artifact carries by construction.

**2. Bit-exactness was never a config-stable property.** Per-pixel results
on GPU depend on the shape of the batch they ride in: cuBLAS selects
different kernels/tilings for different matrix sizes, and those have
different reduction orders. Consequently the `batch_size` sweep on
`perf/inference-loader-and-batch` (3584 → 7168 → 14336 → 7168) already
changed int8 outputs; so does bucket occupancy varying with strip
boundaries, a torch/cuBLAS/driver upgrade, or a different GPU model
(A10G vs L40S vs the T4 FP16 fallback). Bit-exactness only ever held
within one frozen config on one software stack — a regression-testing
convenience, not a product guarantee.

**3. What genuinely is exact stays exact.** The quantizer parity contract
(`quantize_rows_torch` bit-identical to the CPU `quantize_rows` on the same
float32 input) sits *after* the forward and is untouched. The Phase-2
resampler vectorization produces identical indices and identical
per-element arithmetic and is held to bit-identity via golden-batch tests.
Adapter parity (ADR 006's "byte-equivalent output") is likewise defined
*at a fixed code version* and is unaffected.

## Decision

Forward-pass optimizations may reorder floating-point reductions if they
pass an **equivalence harness** comparing against the current BF16
reference on representative dense + sparse chunks:

| Metric | Threshold |
|---|---|
| int8 values exactly equal | ≥ 99.5% |
| max int8 deviation | ≤ 1 level |
| per-pixel scale relative drift | ≤ 0.1% |
| cosine similarity of dequantized embeddings | ≥ 0.9999 |

A change that cannot pass does not ship. Changes that can be kept
bit-identical (pipelining, memory movement, resampler vectorization) are
kept bit-identical and tested as such.

**Scope.** These gates decide changes to the forward pass and its config. A
change upstream that alters the mosaics it reads, such as the ingest hosts'
CPU architecture, is outside them: the harness measures it, and the
maintainers accept or refuse it with that measurement on record. The
Graviton ingest workers were accepted this way on 2026-10-02, at a
worst-pixel cosine of 0.99981 ([ingest measurements
§15.6](../ingest/campaign-ingest-measurements.md)).

### Cross-config comparison envelope

The table above gates a **same-config** change — optimized code vs the
reference at the *identical* batch size and software stack. Comparing two
runs that differ in **config** (batch size, strip boundaries, tile order,
library stack) is a different question: per fact 2, cuBLAS selects different
kernels for different batch shapes, so int8 values shimmer by more than one
level with no change to the model or code. Judged against the same-config
table, a config change would read as a regression; it is not one. Cross-run
diffs of this kind are therefore held to a looser envelope (implemented as
`compare_outputs.py --cross-config`):

| Metric | Same-config gate | Cross-config envelope |
|---|---|---|
| int8 within ±1 level | (max deviation ≤ 1) | ≥ 99.5%, on tiles of at least 10,000 pixels |
| max int8 deviation | ≤ 1 level | reported, not gated |
| per-pixel scale relative drift | ≤ 0.1% | ≤ 3% |
| cosine similarity of dequantized embeddings | ≥ 0.9999 | ≥ 0.999 (worst pixel) |
| footprint / obs-count layers | exact | exact |

**Where the bounds come from.** Two full Iowa years compared across configs,
788 tile pairs: the sharded strip loader against the old one on v1.1, and
the October 2026 optimisation stack against `main` on v2. Their worst tiles
reached 4 levels, 1.97% scale drift and a worst-pixel cosine of 0.999865,
and every tile of 10,000 pixels or more kept at least 99.994% of values
within one level. What a user consumes did not move: 99.98% of each pixel's
top-20 neighbours were kept on v1.1 and 99.99% on v2. A real defect sits far
outside the bounds: the `nn.GRU` swap of ADR 026 left 41% of values within
one level, 24% scale drift and a worst cosine of 0.99. The largest deviation
is reported but not gated, because it is one value among hundreds of
millions and grows with the number compared; the within-one share is not
judged below 10,000 pixels, where a handful of values decides it. The
per-pixel drift and cosine bounds apply to every tile. The first bounds,
set on three chunks of a batch-size change (≥ 99.99% within one level, at
most 3 levels, 1.6% drift, cosine 0.9999), failed both full years on their
extremes alone.

This envelope does **not** gate *exactly-equal*, which legitimately falls
to ~95% across a batch-size change, but footprint and observation-count
layers are deterministic and stay **exact** in both classes. It is for
diffing across configs, **not** a relaxation of the same-config gate, which
every shippable forward-pass reorder must still meet.

**Usability is measured before such a change reaches a store:** top-20
neighbour agreement on the same tiles, at or above the same-model ceiling of
0.9940 ([`validating-a-model-change.md`](../inference/validating-a-model-change.md)).

**Precision stays BF16.** FP16 with reduced-precision accumulation would
roughly double the matmul ceiling on GA10x-class GPUs (FP32-accumulate
runs at half rate there), but FP16's range tops out at 65504 and
activation saturation produces inf/NaN — the reason BF16 was chosen
originally. The FP16 experiment is deferred indefinitely; revisiting it
requires explicit sign-off, a feature flag, and this same harness.

## Rejected alternatives

- **Bit-exact only.** Rejected because it forbids the only levers that
  speed up the forward itself while protecting a property (cross-config
  bitwise reproducibility) the pipeline never actually had — see fact 2.
- **"Anything that holds downstream metrics."** Rejected for now: it
  unlocks INT8/TensorRT-class rewrites we are not pursuing, and embedding-
  level thresholds are cheaper to gate on than downstream task evals.
- **FP16 fast-accumulate now.** Deferred as above.

## Consequences

- Chunks written before and after a harness-gated change coexist with
  ±1-level int8 inconsistency — the same class of inconsistency the
  batch-size change already introduced. Consistency is defined at the
  model/code-version boundary (recorded in store metadata), not bitwise.
- Exact re-diff of a re-processed chunk is not a corruption check across
  code versions (it already wasn't across config changes).
- Exact-value golden tests on forward modules are replaced by
  tolerance-based equivalence tests; the quantizer parity tests stay exact.
- Embeddings are documented as reproducible to within quantization noise
  across library/config versions — documenting what was already true.
