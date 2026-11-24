# 043 — A per-class gate for precision changes to the forward pass

**Status:** Proposed (2026-11-17; revised 2026-11-19, when FP8, the change it was built for, was measured slower
than bf16 and not adopted: [reduced-precision record](../inference/reduced-precision-inference.md)).

## Context

A precision change (FP8, int8) moves outputs further than reordering the same arithmetic, so
[ADR 030](030-forward-pass-reorderings-ship-at-a-store-boundary.md)'s yardstick, no further apart than two cards
on identical code, rejects every one. [ADR 012](012-validated-equivalence-for-inference-outputs.md)'s envelope
and a tile-level accuracy bound let through the harm that matters. A synthetic shift of FP8's size (cosine
0.9999), aimed from one class towards its most-confused class, passes the envelope, the neighbour overlaps and
a 0.2-point tile bound, yet costs UK cropland 2.0 ± 0.7 points of kNN recall for a classifier fitted on one
side and applied to the other. Cropland is 3% of the labels, so the tile moves 0.06.

The gate below bounds loss per class, with every threshold calibrated per tile from controls and never from
the candidate. It was measured on six tiles with FP8 on the L40S as its first candidate:

- **Known-good variants pass:** all 12 two-card pairs on identical bf16, and 239 of 240 matched-noise draws
  (one is unresolved).
- **Known-bad variants are caught:** the class-directed control at cosine 0.9999 is rejected in 21 of the 25
  gated classes that have one. Three more are unresolved, so they are flagged but not failed. The fourth does
  no harm.
- **Structured error carries a far larger SE than noise in the same cell.** Kansas sorghum's control loses
  3.20 ± 1.12 points where the noise SE is 0.07. A gated cell can therefore hide a large loss behind the
  candidate's own SE, which is why the gate has an unresolved outcome.
- **FP8 failed the gate on Ecuador at both sizes** (Stage 1 wetland recall 0.84 ± 0.18, all four linears
  cropland recall 0.65 ± 0.15). It left cells unresolved on Kansas and Finland, and on Punjab with all four,
  and tripped the tripwire on Kansas sorghum. UK and Iowa passed.

## Decision

1. **A precision change to the forward pass passes when, on every tile of the suite:**

   The suite is six tiles of calendar 2024 on ADR 012's lattice: UK, Iowa, Ecuador, Punjab, Kansas and
   Finland. Labels are ESA WorldCover 2021, with USDA CDL 2024 in Iowa and Kansas. Inputs are under
   `s3://arbol-tessera-inputs-dev/perf/fp8-mixed/` and `fp8-lattice/`. A tile may be added; dropping or
   replacing one is a new decision.

   The candidate is the branch's production forward on an L40S, compared with today's code on the L40S, and
   with today's code on the A10G (a store both cards write to). Two cards on today's code are scored beside
   it as known-good comparisons. The tooling is `scripts/precision_gate/`, and the rule is
   `profiling.inference.precision_gate`.

   1. **Envelope:** ADR 012's cross-config envelope holds.
   2. **Tile loss:** a linear probe and a kNN-20 classifier, fitted on one side and applied to the other, lose
      no more than 0.2 points, in both directions.
   3. **Class loss:** every cell that loses more than δ = 0.5 points is sorted:

      | | loss / SE above z_fam | loss / SE at or under z_fam |
      |---|---|---|
      | **gated cell** | the change **fails** | **unresolved** |
      | **other cell** | **unresolved** | reported |

      - A *cell* is one class × consumer (probe, kNN-20) × direction × measure (recall, precision).
      - SE is the candidate's own delete-one-block jackknife over 40×40-pixel blocks.
      - A cell is *gated* when its median SE over the 40 matched-noise draws is at most δ/4.
      - z_fam is the 95th percentile, over the noise draws, of the largest loss / SE among the tile's gated
        cells.
      - Gains are not harm.
   4. **Validity:** each gated class's control, shifted towards its most-confused class at cosine 0.9999,
      should be rejected. A class whose control is only unresolved is listed as flag-only. One whose control
      passes while losing more than δ is listed as uncovered. A gated class under 2,000 labelled pixels has no
      control and is listed as unvalidated. The lists are reported beside the verdict.
   5. **Tripwire:** a class's gap closure is its mean error projected on the direction to another class's
      centroid, as a share of that gap. The tripwire is half its own control's gap closure at the dose that
      costs δ, interpolated between the three doses. A trip leaves the change unresolved.

   An unresolved result needs a decision; only a pass ships without one. Coherence and the refit spread are
   reported, not gated.
2. **A change that fails may ship as a named exception**, which is a decision of its own, recorded in an ADR that
   accepts its measured losses. The exception's bounds are each failed or unresolved cell and failed tile loss
   at its recorded loss plus one SE, and each tripped class at 1.1 × its recorded gap closure. Gap closure has
   no SE, so the tenth is set by hand: two cards on identical code move it by at most 0.05. Later runs of the
   change, such as its final kernels, are held to those bounds (`apply_gate.py --exception`). Anything the
   exception does not list, an item past its bound, or a failed envelope is a new decision. The exception
   leaves the gate unchanged for every other change.
3. **A change that passes, or ships as an exception, still lands at a store boundary**, as ADR 030 requires of any change to outputs: a
   store is finished on the forward it began with.

## How the thresholds were set

- δ = 0.5 was fixed from the controls before the six-tile per-class run: 1.0 would admit Iowa's soybean
  control (0.70 at worst).
- Three choices were made after seeing FP8's results, and each makes the gate stricter:
  - the unresolved outcome;
  - gating per cell rather than per class, because UK cropland's probe cells cannot resolve 2δ;
  - no refit term in δ.
- δ is a bar on harm, so it is not fitted to a candidate. If it changes, the new value comes from what a
  downstream user can tolerate, decided before the next candidate is measured.

## Consequences

- A gate run is four GPU boxes (today's code on the L40S, A10G and L4; the candidate on an L40S) and about an
  hour on one m7i.16xlarge with 60 single-threaded shards.
- The probe is fitted to convergence. The first FP8 scoring used a probe that stopped early, which moved
  verdicts between environments.

## Rejected alternatives

- **A tile-level bound alone.** A class-directed shift costs one class up to 2 points while the tile moves 0.06.
- **Widening δ to 3 × the refit spread.** The spread is real (1–5.6 points for minority classes), but it
  measures a user's own classifier's dependence on its training sample. Widening δ by it ungates the minority
  classes the gate exists for.
- **Raising δ to whatever a candidate needs.** That sets the bar from the candidate. Accepting a candidate
  that fails is a decision of its own, which leaves the gate unchanged.
- **Gating coherence.** It does not track the size of the harm, and two identical-bf16 cards reach 0.60 inside
  one class.
- **ADR 030's two-card yardstick, or ADR 012's 0.9940 top-20 neighbour gate.** The cards agree too closely
  to measure anything but reordering. bf16 on a different kernel stack fails the top-20 gate, because the
  churn is swaps among near-tied neighbours that cost no accuracy.
