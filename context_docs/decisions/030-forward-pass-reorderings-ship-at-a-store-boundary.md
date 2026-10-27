# 030 — A forward-pass reordering ships on the cross-config envelope and measured usability, at a store boundary

**Status:** Accepted by merging PR #241 (proposed 2026-10-20)

## Context

[ADR 012](012-validated-equivalence-for-inference-outputs.md) lets a forward-pass optimization reorder
floating-point work only if it passes the same-config gate: at least 99.5% of int8 values exactly
equal, no value more than one level away, per-pixel scales within 0.1%, cosine at least 0.9999.
[ADR 025](025-library-upgrades-ship-on-measured-usability-at-a-store-boundary.md) dropped that gate for
library upgrades, because `main` fails it against itself: a tile's output depends on queue order, by up
to three levels and 1.5% in scale. A gate that the current code cannot pass measures noise the pipeline
already has, not the change.

The first reordering of our own arithmetic since then is v2's fused add + LayerNorm: each residual add
in v2's encoder layers and the LayerNorm after it run as one compiled kernel
([`../inference/v2-on-blackwell.md`](../inference/v2-on-blackwell.md) §3). It makes v2 inference 5–7%
faster on an L40S, an RTX PRO 6000 and an A10G. The formula is unchanged, but the kernel's LayerNorm
sums in a different order from aten's. Measured on the real pipeline, the same tiles and the same
card, against today's output; the second column is what the fleet already produces when two GPU
models run identical code:

| | fused add + LayerNorm vs today | RTX PRO 6000 vs L40S, same code |
|---|---:|---:|
| int8 values exactly equal | 98.1–98.4% | 97.2–97.5% |
| largest int8 difference | 3 levels | 3 levels |
| largest scale drift | 1.18% | 1.38% |
| ADR 012 cross-config envelope | passes, every tile on all three cards | passes |
| top-20 neighbours kept | 0.985 | 0.981 |
| same k-means cluster | 99.9% | 99.85% |
| mean per-pixel cosine distance | 7.2e-6 | 1.1e-5 |

It fails the same-config gate on exact match, largest difference and scale, as `main` does against
itself. Its neighbour agreement is also below ADR 025's 0.9940. That figure was calibrated by a
different sampling, a lattice of windows from two whole stores, while the usability script draws
2,000 queries among 20,000 candidates inside each tile. The same script gives 0.981 between two cards on
identical code, which every store filled on a mixed fleet already contains.

## Decision

1. **A forward-pass change that reorders floating-point work, without changing the formula, is gated
   by ADR 012's cross-config envelope plus measured usability**, the rule ADR 025 set for libraries.
   Usability is measured on the same tiles with the same script as a two-card comparison on identical
   code, and the change must not move top-20 neighbours, cluster assignments or the mean shift
   further than that comparison does.
2. **It lands at a store boundary** (ADR 025 §2). A store begun before the change is finished by the
   code that began it; stores started after it use the new code.
3. **A change that can be bit-identical is still held to bit-identity**, as ADR 012 requires.

## Consequences

- v2's fused add + LayerNorm ships, and a global v2 store starts on it. v1.1 is untouched: the change
  is in v2's encoder only, so the published v1.1 store and its updates keep their arithmetic.
- A v2 store begun before this change is appended from a commit that predates it.
- Card mixing becomes the yardstick for "small", because it is the variation already accepted in
  every store a mixed fleet fills.

## Rejected alternatives

- **The same-config gate for reorderings.** It forbids a change that moves outputs less than a
  fallback card does, and `main` fails it against itself.
- **ADR 025's 0.9940 as the usability bar.** Different sampling; two cards running identical code
  fail it.
- **A bit-identical fused kernel.** Rounding the sum where eager does (`emulate_precision_casts`) brings
  the kernel closer, but its LayerNorm still reduces in a different order from aten's, and no compiler
  option reproduces aten's order.
