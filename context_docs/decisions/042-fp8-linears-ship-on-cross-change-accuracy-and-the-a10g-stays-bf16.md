# 042 — FP8 linears ship on cross-change accuracy, at a store boundary, and the A10G stays bf16

**Status:** Proposed (2026-11-17)

## Context

FP8 is the next inference speed-up on the L40S (g6e): its four transformer linears in E4M3 are 1.64×
faster per layer, and with fused quantisation the forward pass is projected at 1.2–1.3×
([`../inference/reduced-precision-inference.md`](../inference/reduced-precision-inference.md)). It
changes precision, not the order of floating-point work, so
[ADR 030](030-forward-pass-reorderings-ship-at-a-store-boundary.md) does not cover it. Two questions
had to be settled before building the fused kernels: which gate a precision change meets, and whether
a g5 (A10G, which has no FP8) can still fill part of an FP8 store when g6e capacity runs short.

Both were measured on the unfused `torch._scaled_mm` path, which quantises with the same scale rule as
the planned fused kernels. Fusion reorders the LayerNorm's reduction, a bf16-sized difference (ADR 030)
far smaller than FP8's, but not zero, so the shipped kernels are measured again (decision 2).
Measured 2026-11-17 with production code and the production checkpoint, on ADR 012's lattice (3×3
windows of 160×160, 230,400 px per tile) over six tiles of calendar 2024: dense UK and Iowa, plus
Ecuador lowlands (cloudy tropics), Punjab (irrigated, arid), Kansas (winter wheat; ascending radar
only) and Finland (boreal). Native FP8 ran on the L40S and on an L4; bf16 ran on all three cards;
FP8 was also emulated on the A10G. FP8 used row-wise activation and per-channel weight scales, on all
four linears and on QKV and FFN up only (Stage 1). Ranges cover the six tiles:

| | int8 within ±1 | top-20 kept | inside top-25 | coherence | cross-change probe (pts) |
|---|---:|---:|---:|---:|---:|
| two cards, bf16, identical code (L40S, A10G, L4) | ≥ 99.998% | 0.999–1.000 | 1.0000 | 0.02–0.10 | ±0.01 |
| two cards, native FP8 (L40S, L4) | ≥ 99.99% | 0.994–1.000 | ≥ 0.9999 | 0.01–0.07 | ±0.01 |
| FP8 on the L40S, all four linears | 99.58–99.97% | 0.954–0.977 | 0.995–1.000 | 0.25–0.29 | −0.15 to +0.15 |
| FP8 on the L40S, Stage 1 | 99.78–99.99% | 0.958–0.979 | 0.997–1.000 | 0.22–0.27 | −0.20 to +0.20 |
| A10G bf16 vs L40S FP8 (a mixed store) | as the FP8 rows | | | | −0.20 to +0.26 |
| A10G emulated FP8 vs L40S native FP8 | 99.90–99.998% | 0.966–0.987 | 0.998–1.000 | 0.006–0.009 | ±0.02 |

ADR 012's full cross-config envelope holds on every tile and variant, mixed stores included: at least
99.58% of values within ±1 (floor 99.5%), worst-pixel cosine at least 0.99940 (floor 0.999) and
per-pixel scale drift at most 2.6% (limit 3%). Ecuador with all four linears is the closest to all
three limits.

*Coherence* is the length of the mean error vector over the mean error length: about 0.02 for
unstructured noise, 1 for a shift every pixel shares. *Cross-change probe* is the change when a linear
probe, fitted on labelled pixels from one side (ESA WorldCover 2021; USDA CDL 2024 in Iowa and
Kansas), is applied to the other side's pixels from held-out windows, against applying it to its own
side: about 230,000 labelled pixels per tile in three held-out folds, both directions. A kNN-20
classifier scored the same way moves by at most ±0.05 points on every pair.

To tell a real effect from the metric's own noise, the same test was run on isotropic per-pixel noise
matched to FP8's mean cosine after int8 quantisation, five draws per tile and size. Noise alone moves
the probe by up to 0.03–0.13 points depending on the tile (standard deviation 0.015–0.063), and kNN-20
by up to 0.05.

Four findings decide this:

- **ADR 030's yardstick cannot admit a precision change.** On identical bf16 code the fleet's cards
  agree on ≥ 99.998% of values within ±1 and keep 0.999 of top-20 neighbours, far closer than the RTX
  PRO 6000 that set ADR 030's 0.981. Any change of precision, FP8 or int8, fails it.
- **A bf16 g5 in an FP8 store is exactly the FP8 change**, because the A10G's bf16 matches the L40S's.
- **Most of that change is indistinguishable from noise.** A move counts as structural when it goes
  past the largest of the five noise draws. On UK, Iowa and Finland, and on Ecuador and Punjab for the
  other size, FP8 moves the probe no further than matched noise does, and kNN-20 never does.
- **FP8's shared shift (coherence 0.25) has a small real cost on a linear probe on two tiles.** On
  Ecuador (Stage 1, ±0.20) and Kansas (all four ±0.15, Stage 1 −0.10/+0.12) the probe moves about 2–4
  noise standard deviations, past every noise draw, by roughly equal and opposite amounts in the two
  directions, which noise never does. Punjab (all four) passes its largest draw marginally, but as a
  gain (+0.04 against 0.03), not a loss. The
  largest loss in a mixed store is 0.20 points, about one labelled pixel in 500 changing class. kNN-20
  sees nothing beyond noise.

## Decision

1. **A precision change to the forward pass is gated by ADR 012's cross-config envelope plus
   cross-change accuracy.** On every labelled tile measured, a linear probe and a kNN-20 classifier
   fitted on one side and applied to the other must lose no more than 0.2 points. The gate is on loss:
   a gain is not harm. Each tile is also scored against five draws of matched isotropic noise, and a
   move outside that spread is reported as structural. Top-20 overlap, top-25 overlap and coherence
   are reported, not gated.
2. **FP8 means E4M3, with per-channel weight and per-row activation scales.** The fused kernels that
   ship must pass the same gate on the lattice themselves, against bf16 on the A10G as well as the
   L40S. Static activation scales (Stage 2) change the values further and are held to it too.
3. **It lands at a store boundary** (ADR 025 §2). A store begun in bf16 is finished in bf16, and FP8
   stores are those started after the change.
4. **Precision is a property of the store, and inside an FP8 store an A10G runs today's bf16 forward
   unchanged.** A store's precision is recorded when the store is created and handed to model
   construction for every run that writes to it, so a bf16 store is finished in bf16 on any card. The
   card only decides whether the store's precision can run: on Ada and newer an FP8 store builds the
   FP8 encoder layers, and on an A10G it builds the bf16 ones. No other code knows which card it runs
   on.

## Consequences

- g5 fallback keeps working for FP8 stores at today's g5 cost. The g5 runs today's bf16 forward; the
  only new code is recording the store's precision and passing it to model construction.
- **What a mixed FP8 store carries:** between its g5 and g6e pixels, FP8's shared shift. A linear
  classifier fitted on one card's pixels and applied to the other's can lose up to 0.2 points on some
  tiles (measured: cloudy tropics and Kansas wheat); kNN is unaffected. This is the first thing to
  recheck if a downstream user reports a seam between regions, and the case for emulated FP8 on the g5
  if one does.
- The gate passes Stage 1 and all four linears on six tiles, with the mixed store's largest loss at
  the 0.2-point limit (Ecuador, Stage 1). A change that moves further, such as Stage 2's static
  scales, needs a decision rather than a pass.
- The tie-tolerant neighbour gate proposed in the reduced-precision record (top-25 at 0.995 on the
  worst stratum) is not adopted. Inside one class or window FP8 scores as low as 0.972 (Finland's
  water), and that threshold was calibrated over whole tiles. A per-stratum threshold needs its own
  noise ladder.
- The measurement scripts and every output are in `s3://arbol-tessera-inputs-dev/perf/fp8-mixed/`;
  the four new tiles' mosaics stay under `mosaics/fp8-*` for reruns.

## Rejected alternatives

- **A fixed 0.2-point bound on any move, in either direction.** The metric's own noise reaches 0.13
  points on these tiles, and the largest move (+0.26) is a gain, the mirror of the 0.20 loss.
- **ADR 030's two-card comparison as the bar.** The fleet's cards agree too closely for it to measure
  anything but bit-level reordering.
- **ADR 012's 0.9940 top-20 gate.** bf16 on a different kernel stack fails it, and FP8's churn is
  swaps among near-tied neighbours that cost no accuracy.
- **FP8 emulated on the A10G** (FP8 rounding on bf16 tensor cores). It removes the shared shift,
  holding a mixed store to ±0.02 points with coherence at the noise floor, but adds card-specific
  rounding to every forward path that a g5 runs, to remove a loss of at most 0.2 points on a linear
  probe.
- **g6e-only FP8 stores.** They give up g5 fallback, which g6e capacity has needed repeatedly, for the
  same 0.2 points.
