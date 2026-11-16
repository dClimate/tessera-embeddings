# FP8 inference for Tessera v2 Large on the L40S (g6e): parity and performance

*2026-11-16. v2 Large, production checkpoint, `torch 2.14.1+cu130`, on g6e.xlarge.*

## Summary

- **Speed.** Running the transformer's linear layers in FP8 (E4M3) could make the forward pass about **1.2–1.3× faster** on g6e. That needs fused quantisation kernels, which are not built yet. A naive implementation is *slower* than bf16.
- **Per-pixel parity.** FP8 **passes** ADR 012's per-value equivalence envelope on two full tiles.
- **Neighbour parity.** FP8 **fails** ADR 012's top-20 neighbour gate: it keeps about 97% of neighbours, against a gate of ≥ 99.40%.
- The swapped neighbours are near-ties, so the change likely doesn't matter downstream. That is unverified.
- FP8 needs Ada or newer, so the g5 (A10G) part of the fleet gets nothing.

## What was tested

- **What was changed:** only the four transformer linear layers (QKV, output projection, FFN1, FFN2), using `torch._scaled_mm` E4M3 with per-tensor or row-wise scaling. These carry nearly all the FLOPs. Everything else stays bf16.
- **Reference:** the production bf16 store for the same pixels. A fresh bf16 rerun gives the same-model ceiling.
- **Pixels:** ADR 012's calibration sampling, a 3×3 lattice of 160×160 windows (230,400 px) on each of a dense UK tile (both radar orbits) and an Iowa tile (ascending radar only).
- **Neighbour metric:** top-20 neighbours among 2,000 sampled pixels, repeated over 5 random samples.

## Numerical parity

**Per-value envelope** (ADR 012 bars: int8 within ±1 on ≥ 99.5% of values, worst-pixel cosine ≥ 0.999, scale drift ≤ 3%):

| vs production store | within ±1 | worst cosine | scale drift |
|---|---|---|---|
| bf16 rerun (ceiling) | 100.0% | 0.99991 | 1.4% |
| **FP8 row-wise** | 99.82–99.84% ✓ | 0.99964–0.99968 ✓ | 1.8–1.9% ✓ |
| **FP8 per-tensor** | 99.75–99.84% ✓ | 0.99965–0.99967 ✓ | 1.8–2.3% ✓ |

**Neighbour gate** (top-20 overlap, mean ± sd over 5 samples; gate ≥ 0.9940):

| vs production store | UK | Iowa |
|---|---|---|
| bf16 rerun (ceiling) | 0.9989 | 0.9995 |
| **FP8 row-wise** | 0.9708 ± 0.0006 ✗ | 0.9723 ± 0.0013 ✗ |
| **FP8 per-tensor** | 0.9702 ± 0.0005 ✗ | 0.9721 ± 0.0010 ✗ |

- **E5M2 ("bf8")**, tested in simulation only, fails even the per-value envelope (95.7–97.4% within ±1). E4M3 is the only candidate.

## Why the two kinds of parity disagree

- **Neighbours are near-ties.** Each pixel's 20th nearest neighbour is at a median of 13°, but the gap between the 20th and 21st is a cosine of only 0.00024 at the median (0.00003 at p10).
- **FP8 moves each embedding about 1°** (cosine ≈ 0.9999). That is enough to swap near-tied neighbours across the cutoff.
- **The error behaves like random per-pixel noise.** Adding isotropic noise of that size to the production embeddings reproduces FP8's overlap exactly (0.9704 vs 0.9708).
- **ADR 012's two gates are not calibrated against each other.** A change at the per-value floor (worst cosine 0.999) would keep only about 94% of neighbours.
- **The swapped neighbours are interchangeable near-duplicates.** So the churn probably doesn't affect downstream tasks, but that hasn't been tested.

## Performance (g6e.xlarge)

| measurement | result |
|---|---|
| FP8 vs bf16 GEMM, our shapes (M = 896k) | **1.64× per layer**: FFN 1.85×, QKV 1.42×, out-proj 1.10× |
| GEMM share of the production bf16 forward | 66%; flash attention 14%, LayerNorm and elementwise 21% |
| Forward-pass ceiling if quantisation were free | ~1.35× |
| Naive FP8 (separate quantise pass per linear) | **0.33–0.58×**: each pass costs 16–65 ms vs 4–20 ms per GEMM |
| Realistic, quantisation fused into LayerNorm/ReLU | **~1.2–1.3× forward**, ~1.2× end-to-end |

**Rough value.** v2 with the October optimisation stack is costed at about $363k of GPU per nine-year campaign. A ~1.2× gain on the g6e share is worth on the order of **$50–70k per campaign** if the fleet is mostly g6e, and less if g5 carries much of it.

## What it would take

1. **A downstream check** (about 1–2 days): kNN classification accuracy on labelled pixels, FP8 vs bf16. It decides whether the 3% neighbour churn matters.
2. **A policy decision** on ADR 012's neighbour gate for unstructured per-pixel perturbations, if the downstream check shows no loss.
3. **Fused quantisation kernels** (a few days): Triton, with quantisation folded into the add+LayerNorm and ReLU epilogues. Then a speed measurement on g6e.
4. **A store boundary.** FP8 and bf16 embeddings should not share a store.

*Scripts (`fp8_sim.py`, `fp8_gpu.py`, `fp8_micro.py`, `lattice_prep.py`, `lattice_gpu.py`) and the lattice inputs are in `s3://arbol-tessera-inputs-dev/perf/cpu-ab/scripts/` and `s3://arbol-tessera-inputs-dev/perf/fp8-lattice/`.*
