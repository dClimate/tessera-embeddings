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

Fusion changes speed, not the values written, so the unfused `torch._scaled_mm` path answers both.
Measured 2026-11-17 on ADR 012's lattice (3×3 windows of 160×160 on a dense UK tile and an Iowa tile,
230,400 px each), with production code and the production checkpoint. No g6e had capacity, so native
FP8 ran on an L4 (the same Ada FP8 hardware). bf16 ran on the L40S (the production store), the A10G
and the L4. FP8 used row-wise activation and per-channel weight scales, on all four linears and on
QKV and FFN up only (Stage 1). Ranges cover both tiles:

| | int8 within ±1 | top-20 kept | inside top-25 | coherence | cross-change accuracy (pts) |
|---|---:|---:|---:|---:|---:|
| two cards, bf16, identical code | ≥ 99.999% | 0.9992–0.9996 | 1.0000 | 0.05–0.10 | 0.00 |
| FP8, all four linears | 99.84% | 0.971–0.973 | 0.9992–0.9995 | 0.27–0.28 | −0.02 to +0.07 |
| FP8, Stage 1 | 99.90–99.92% | 0.973–0.974 | 0.9993–0.9996 | 0.25–0.27 | −0.02 to +0.03 |
| A10G bf16 vs FP8 (a mixed store) | as the FP8 rows | | | | −0.04 to +0.02 |

*Coherence* is the length of the mean error vector over the mean error length: about 0.02 for
unstructured noise, 1 for a shift every pixel shares. *Cross-change accuracy* is the change when a
linear probe or a kNN-20 classifier, fitted on labelled pixels (USDA CDL in Iowa, ESA WorldCover in
the UK) from one side, is applied to the other side's pixels from held-out windows, against applying
it to its own side: about 230,000 labelled pixels per tile, in three held-out folds.

Three findings decide this:

- **ADR 030's yardstick cannot admit a precision change.** On identical bf16 code the A10G and the
  L40S agree on 99.86–99.91% of int8 values and keep 0.9995 of top-20 neighbours, far closer than the
  RTX PRO 6000 that set ADR 030's 0.981. Any change of precision, FP8 or int8, fails it.
- **FP8's error is partly a shared shift** (coherence 0.27), the kind of error that neighbour metrics
  miss and that hurt a classifier crossing a change in the synthetic controls.
- **On real kernels that shift costs nothing measurable across the change.** Every pairing, including
  bf16 pixels from an A10G read by a classifier fitted on FP8 pixels, stays within ±0.07 points.

## Decision

1. **A precision change to the forward pass is gated by ADR 012's cross-config envelope plus
   cross-change accuracy.** On every labelled tile measured, a probe and a kNN-20 classifier fitted on
   one side and applied to the other must move by no more than 0.2 points, the refit jitter of the
   gate calibration. Top-20 overlap, top-25 overlap and coherence are reported, not gated.
2. **FP8 means E4M3, with per-channel weight and per-row activation scales.** Static activation scales
   (Stage 2) change the values and must pass the same gate before they ship.
3. **It lands at a store boundary** (ADR 025 §2). A store begun in bf16 is finished in bf16, and FP8
   stores are those started after the change.
4. **Inside an FP8 store, an A10G runs today's bf16 forward unchanged.** One capability check, made
   when the model is built, picks the FP8 encoder layers on Ada and newer. No other code knows which
   card it runs on.

## Consequences

- g5 fallback keeps working for FP8 stores at today's g5 cost, and needs no new code.
- An FP8 store filled partly on a g5 carries FP8's shared shift between its g5 and g6e pixels. That
  is measured harmless on two tiles. It is the first thing to recheck if a downstream user reports a
  seam between regions.
- **Before an FP8 store begins**, this measurement is repeated with native FP8 on an L40S and on more
  than two tiles, scored with `mixed_pool_score.py`.
- The tie-tolerant neighbour gate proposed in the reduced-precision record (top-25 at 0.995 on the
  worst stratum) is not adopted. Inside one window FP8 scores 0.992–0.994, and that threshold was
  calibrated over whole tiles. A per-stratum threshold needs its own noise ladder.

## Rejected alternatives

- **ADR 030's two-card comparison as the bar.** The fleet's two cards agree too closely for it to
  measure anything but bit-level reordering.
- **ADR 012's 0.9940 top-20 gate.** bf16 on a different kernel stack fails it, and FP8's churn is
  swaps among near-tied neighbours that cost no accuracy.
- **FP8 emulated on the A10G** (FP8 rounding on bf16 tensor cores). It removes the shared shift,
  bringing the A10G to coherence 0.006–0.007 against native FP8, but adds a third forward path and
  card-specific rounding for no measured gain.
- **g6e-only FP8 stores.** They give up the g5 fallback to avoid a difference that measures as zero.
