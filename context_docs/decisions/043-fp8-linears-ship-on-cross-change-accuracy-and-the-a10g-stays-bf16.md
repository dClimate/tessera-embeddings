# 043 — A per-class gate for precision changes; FP8 linears fail it and ship as a named exception, at a store boundary, and the A10G stays bf16

**Status:** Proposed (2026-11-17). **FP8 is not adopted** (2026-11-19): at the 64-pixel chunks the L40S has run
since [ADR 042](042-chunked-graph-replay-ships-above-the-two-card-yardstick.md), the Stage 1 forward measures
0.83–0.91× bf16 and cannot exceed 1.02× even with quantisation free
([reduced-precision record](../inference/reduced-precision-inference.md#fp8-at-production-chunk-sizes)). Decision 1,
the gate, stands for any precision change; decisions 2–5 describe an FP8 that is not built. The tables below come
from a probe that stopped fitting early; the record's Next steps give the converged figures. A mixed store's gap
closure in them was measured against the L40S bf16 reference rather than its own A10G one, which is why it matches
FP8's; the gate's tooling now measures each comparison from its own reference side.

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
far smaller than FP8's, but not zero, so the shipped kernels are measured again (decision 3).
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

**A tile-level bound misses harm to one class.** A synthetic shift of FP8's size (cosine 0.9999),
aimed from one class towards its most-confused class, passes the envelope, the neighbour overlaps and a
0.2-point tile bound, yet costs UK cropland 2.0 ± 0.7 points of kNN recall across the change: cropland
is 3% of the labels, so the tile moves 0.06. The gate therefore bounds loss per class, and every
threshold in it is calibrated per tile from controls, never from the candidate:

- **Known good:** two cards on identical bf16, and 20 draws per FP8 size of isotropic noise matched to
  FP8's mean cosine after int8 quantisation (40 per tile). They set the noise SE of each cell and the
  family-wise threshold z_fam.
- **Known bad:** each class with at least 2,000 labelled pixels shifted towards its most-confused class
  at cosine 0.9999, 0.99995 and 0.99998 inside that class; and the largest class shifted towards the
  smallest such class.
- **Refit spread:** ten redraws of the 48,000-pixel training sample on bf16.

What the controls show, on the six tiles:

- **No known-good variant fails.** All 12 two-card pairs and 239 of the 240 noise draws pass; one
  draw is unresolved.
- **The gate rejects the class-directed control at cosine 0.9999 in 21 of the 25 gated classes that
  have one** (Ecuador water, at 1,709 px, is too small for a control),
  including both shifts that motivated it: UK cropland towards grass (kNN recall −2.00 ± 0.68, z 3.0
  against z_fam 2.79) and Iowa soybeans towards corn (probe recall −0.68 ± 0.10). Of the four it
  does not reject, Finland water's control does no harm (worst 0.22) and passes. The other three are
  unresolved, so the gate flags them without failing them: Kansas soybeans' and Punjab grass's cost
  1.30 ± 0.66 and 0.58 ± 0.36 points in gated cells, and **Kansas developed's** costs
  3.23 ± 0.44 points of probe recall in cells too noisy to gate (7,396 px), where FP8 loses at most
  0.47 ± 0.20. The detection limit also shows at the weakest dose: UK cropland's −0.79 ± 0.31 at
  cosine 0.99998 is unresolved, not rejected.
- **Structured error carries a far larger SE than noise in the same cell.** Kansas sorghum's control
  loses 3.20 ± 1.12 points of probe recall where the noise draws' SE is 0.07, so its loss / SE is 2.9
  against z_fam 3.16. A cell gated on the noise SE therefore does not resolve every 2δ loss, and a
  loss past δ that the candidate's own SE leaves under z_fam is unresolved (decision 1.3).
- **The large-to-small shift fails on all six tiles at 0.9999,** mostly through the large class's own
  recall when the classifier is fitted on the changed side, not through the small class's precision
  as expected.
- **The refit spread is large for minority classes:** redrawing the training sample moves their recall
  by a standard deviation of 1–5.6 points (UK cropland 5.6 on the probe, 3.0 on kNN), against 0.1–0.6
  for large classes. It is reported, not used to widen the bound (rejected alternatives).

**FP8 fails the gate on three of six tiles.** Worst gated cells, loss in points ± block-jackknife SE:

| tile | z_fam (gated cells) | FP8 all four linears | FP8 Stage 1 |
|---|---:|---|---|
| UK | 2.79 (28) | pass: cropland kNN precision 0.25 ± 0.18 | pass: cropland kNN precision 0.31 ± 0.16 |
| Iowa | 2.83 (39) | **fail: forest probe precision 0.64 ± 0.15** (z 4.2) | pass: forest probe precision 0.27 ± 0.10 |
| Ecuador | 4.45 (35) | **fail: trees probe recall 0.55 ± 0.09** (z 6.4) | pass: cropland probe precision 0.35 ± 0.08 |
| Punjab | 3.58 (26) | pass: trees probe recall 0.39 ± 0.12 | pass: trees probe recall 0.34 ± 0.14 |
| Kansas | 3.16 (40) | **fail: winter wheat probe recall 0.96 ± 0.21** (z 4.6) | **fail: winter wheat probe recall 0.66 ± 0.16** (z 4.2) |
| Finland | 4.21 (16) | pass: trees probe precision 0.12 ± 0.03 | pass: trees probe precision 0.08 ± 0.03 |

- **Every failure is on the linear probe.** kNN-20 passes every cell on every tile.
- **A mixed store fails in the same cells,** since A10G bf16 is the L40S's bf16: Kansas winter wheat
  1.00 (all four) and 0.70 (Stage 1), Iowa forest 0.65, Ecuador trees 0.55. Ecuador's Stage 1 mixed
  store also sits on the tile bound, losing 0.2005 points.
- **FP8 leaves six cells unresolved,** five with all four linears and one at Stage 1. Loss in points
  of the probe, mixed-store figure where it is larger:

  | tile | cell | loss | loss / SE against z_fam | why unresolved |
  |---|---|---|---|---|
  | Ecuador, all four | cropland recall | 0.79 ± 0.18 | 4.3 against 4.45 | gated; own SE 0.18 against noise 0.12 |
  | Kansas, all four | sorghum precision | 0.78 ± 0.30 | 2.6 against 3.16 | gated; own SE 0.30 against noise 0.06 |
  | Kansas, all four | corn recall | 0.68 ± 0.32 | 2.1 against 3.16 | gated; own SE 0.32 against noise 0.05 |
  | Kansas, all four (mixed store only) | sorghum recall | 0.51 ± 0.30 | 1.7 against 3.16 | gated; own SE 0.30 against noise 0.06 |
  | Punjab, all four | trees precision | 0.53 ± 0.10 | 5.2 against 3.58 | not gated: noise SE 0.126 against 0.125 |
  | Ecuador, Stage 1 | wetland recall | 0.79 ± 0.17 | 4.5 against 4.45 | not gated: noise SE 0.139 |

  Punjab with all four linears and Ecuador at Stage 1 fail nothing, so these cells move them from
  pass to unresolved. Known-good variants almost never land there: none of the 12 two-card pairs and
  1 of the 240 noise draws (Finland, an ungated grass cell at z 4.25 against 4.21).
- **Losses past 0.5 that are neither failed nor unresolved are reported.** They sit in cells too noisy
  to gate, under z_fam: Finland grass loses 1.09 ± 0.30 of probe recall (all four), and Kansas
  soybeans up to 1.30 ± 0.45 of probe precision (both sizes).
- **The gap-closure tripwire fires for FP8 on every tile** except UK and Finland at Stage 1, and for
  no noise draw on any tile. Gap closure in thousandths of the centroid gap, a mixed store's being
  the same:

  | tile | class | tripwire | all four | Stage 1 |
  |---|---|---:|---:|---:|
  | UK | trees | 0.82 | 1.16 | under (0.53) |
  | Iowa | pasture | 1.00 | 2.73 | 1.84 |
  | Ecuador | trees | 1.72 | 3.92 | 1.85 |
  | Ecuador | grass | 0.35 | 1.97 | 2.61 |
  | Punjab | trees | 0.73 | 1.86 | 2.07 |
  | Kansas | pasture | 0.80 | 5.24 | 4.40 |
  | Kansas | winter wheat | 2.53 | 4.34 | 2.60 |
  | Kansas | sorghum | 2.77 | 7.03 | 4.08 |
  | Finland | grass | 2.79 | 3.28 | under (1.51) |

  Two cards on identical bf16 reach 0.05 at most, and the noise draws 0.25.
- **Coherence does not track harm:** 0.75 at the weakest control dose against 0.72 at about nine times
  the probe damage, and two cards on identical bf16 reach 0.60 inside one class.

## Decision

1. **A precision change to the forward pass passes when, on every tile of the suite:**

   The suite is the six tiles measured here (UK, Iowa, Ecuador, Punjab, Kansas, Finland; calendar
   2024; ADR 012's lattice; ESA WorldCover 2021 labels, USDA CDL 2024 in Iowa and Kansas), with the
   labels, scripts and per-card embeddings stored under
   `s3://arbol-tessera-inputs-dev/perf/fp8-mixed/`; the reduced-precision record lists every input.
   A candidate is scored on all six. A tile may be added; dropping or replacing one is a new
   decision.

   1. **Envelope:** ADR 012's cross-config envelope holds.
   2. **Tile loss:** a linear probe and a kNN-20 classifier, fitted on one side and applied to the
      other, lose no more than 0.2 points, in both directions.
   3. **Class loss:** every cell that loses more than δ = 0.5 points is sorted by whether it is gated
      and whether its loss / SE is above z_fam:

      | | loss / SE above z_fam | loss / SE at or under z_fam |
      |---|---|---|
      | **gated cell** | the change **fails** | **unresolved** |
      | **other cell** | **unresolved** | reported |

      A change passes when no cell fails and none is unresolved. An unresolved cell needs a decision,
      as a tripwire does: the loss is past δ and the gate cannot call it noise.
      - A *cell* is one class × consumer (probe, kNN-20) × direction (fitted on the reference and
        applied to the change, and the reverse) × measure (recall, precision).
      - SE is the candidate's own delete-one-block jackknife over 40×40-pixel blocks.
      - A cell is *gated* when its median SE over the matched-noise draws is at most δ/4. That
        resolves a 2δ loss whose SE is the noise's. Structured error clusters in fields and can carry
        several times that SE, which is what the gated-and-under case catches.
      - z_fam is the 95th percentile, over the noise draws, of the largest loss / SE among the tile's
        gated cells.
      - Gains are not harm.
   4. **Validity:** each gated class's class-directed control at cosine 0.9999 should be rejected.
      Where it is only unresolved, the gate can flag a change of that kind in that class but cannot
      fail it, and the class is listed as flag-only on that tile. Where it passes while losing more
      than δ in a reported cell, the gate cannot see that kind of change at all, and the class is
      listed as uncovered. A gated class too small for a
      control (under 2,000 labelled pixels) is listed as unvalidated. All three lists go beside the
      verdict; the classes' cells still count.
   5. **Tripwire:** a class's gap closure is its mean error projected on the direction to another
      class's centroid, as a share of the centroid gap. Past half its own control's gap closure at the
      dose that costs δ, the change needs a decision even if accuracy passes.
      - A control's dose is its worst loss in any gated cell on the tile. The gap closure at δ is
        interpolated linearly between the three doses, and scaled in proportion from the nearest
        dose where δ lies outside them, as it does when no dose costs δ.
      - A class with no control has no tripwire. If it is gated it is already listed as unvalidated.

   Neighbour overlaps and coherence are reported, not gated.
2. **FP8 fails this gate and ships as a named exception.** It fails on Kansas winter wheat (probe
   recall, both sizes, up to 1.0 point including a mixed store), Iowa forest (probe precision, all four
   linears, 0.65) and Ecuador trees (probe recall, all four linears, 0.55); Ecuador's Stage 1 mixed
   store loses 0.2005 points at the tile level; it leaves the six cells in the context's
   unresolved-cells table unresolved; and it trips the tripwire.
   These losses are accepted for FP8 E4M3 as defined in decision 3, the unresolved cells included:
   this is their decision. The exception does not change the
   gate: any other precision change, int8 included, meets it unchanged. The fused kernels and Stage 2
   go through the gate themselves, and pass under this exception only if they fail or are unresolved
   in these cells and Ecuador's tile bound alone, by no more than the recorded loss plus one SE
   (a cell's jackknife SE; for Ecuador's tile bound, 0.2005 + 0.047, its jackknife SE over blocks),
   and trip the tripwire only in the classes of the context's gap-closure table, by no more than 1.1 times the
   gap closure recorded there. Each is held to the record for its own size, all four linears or
   Stage 1. Gap closure has no SE of its own, so the tenth is an allowance set by hand: two cards on
   identical code move it by at most 0.05, under a twentieth of the smallest recorded trip. A cell
   or class that worsens past its bound is outside the exception even if nothing new fails.
   Anything else is a new decision.
3. **FP8 means E4M3, with per-channel weight and per-row activation scales.** The fused kernels that
   ship must be measured on the lattice themselves, against bf16 on the A10G as well as the L40S.
   The exception in decision 2 is defined on these scales. Stage 2's static activation scales are a
   different quantisation: Stage 2 goes through the gate, and stays inside the exception only on
   decision 2's terms. Past them it is a new decision.
4. **It lands at a store boundary** (ADR 025 §2). A store begun in bf16 is finished on today's forward, and FP8
   stores are those started after the change.
5. **Precision is a property of the store, and inside an FP8 store an A10G runs today's bf16 forward
   unchanged.** A store's precision is recorded when the store is created and handed to model
   construction for every run that writes to it, so a bf16 store is finished on today's forward on any card. The
   card only decides whether the store's precision can run, and only the fleet's two cards may write
   to an FP8 store: the L40S builds the FP8 encoder layers, and the A10G builds the bf16 ones. Any
   other card refuses to write to an FP8 store. That covers the L4, which is not in the fleet and
   whose FP8 was measured only on the unfused path; a newer architecture with FP8, such as
   Blackwell's RTX PRO 6000, which may select different kernels; and a card without bf16, such as
   the T4, which today's forward moves to FP16. A card joins either list by being measured on the
   lattice against the L40S, on the kernels that ship. No other code knows which card it runs on.

## How the thresholds were set

- δ = 0.5 was fixed before the six-tile per-class run, from the controls alone: a 1.0 allowance was
  dropped because Iowa's soybean control costs 0.70 at worst, so 1.0 would admit it. At that point
  per-class FP8 results existed only for UK and Iowa (worst cells 0.27 and 0.17).
- Three choices were made after seeing data that included FP8. All make the gate stricter on FP8, not
  more lenient:
  - **The unresolved outcome** (added 2026-11-18, in review). Before it, a gated cell passed whenever
    the candidate's own SE kept loss / SE under z_fam, however large the loss. It moves FP8's Punjab
    (all four) and Ecuador (Stage 1) results from pass to unresolved and changes no failure. An
    ungated cell is unresolved only above z_fam because a loss past δ alone is routine there: it
    occurs in 71 of the 240 noise draws, all 40 on Finland.
  - **Gating per cell rather than per class.** UK cropland's kNN cells resolve a 2δ loss but its probe
    cells do not (noise SE 0.08–0.11 against 0.13–0.17). Gating per class would drop the class that
    motivated the gate.
  - **No refit term in δ.** The plan let δ widen to 3 × the refit standard deviation where that was
    larger. Measured, that sets δ at 2–17 points for minority classes, and the gate then no longer
    rejects the UK cropland control.
- The control does not set δ. FP8's error along the cropland-to-grass direction is about 90 times
  smaller than the control's, so a wide range of δ would separate the two; choosing among them after
  measuring FP8 would be fitting the bar to the candidate. δ is a bar on harm. If it changes, it comes
  from what a downstream consumer can tolerate, decided before the next candidate is measured.

## Consequences

- g5 fallback keeps working for FP8 stores at today's g5 cost. The g5 runs today's bf16 forward; the
  new code is recording the store's precision, passing it to model construction, and refusing an
  FP8 store on a card outside the measured three.
- **What FP8 costs a downstream user:** a linear classifier fitted on one precision and applied to the
  other, across stores or inside a mixed store, loses up to 1.0 point of winter wheat recall on Kansas
  (0.7 at Stage 1), and up to 0.65 on single minority classes elsewhere with all four linears. Those
  are the losses the gate confirms. Unresolved cells add up to 0.79 (Ecuador cropland and wetland
  recall), and cells too noisy to gate show point estimates of 1.09 ± 0.30 (Finland grass recall) and
  1.30 ± 0.45 (Kansas soybean precision). At the
  tile level that is at most 0.2 points (0.2005 for Ecuador's Stage 1 mixed store); kNN is unaffected. This is the first thing to recheck if a
  downstream user reports a seam between FP8 and bf16 stores, or between regions inside a mixed store.
- Stage 1 is the cheaper exception: it fails only on Kansas winter wheat, plus Ecuador's tile bound by
  0.0005 as a mixed store, with one unresolved cell (Ecuador wetland). All four linears adds Iowa
  forest and Ecuador trees, and five unresolved cells.
- The tie-tolerant neighbour gate proposed in the reduced-precision record (top-25 at 0.995 on the
  worst stratum) is not adopted. Inside one class or window FP8 scores as low as 0.972 (Finland's
  water), and that threshold was calibrated over whole tiles. A per-stratum threshold needs its own
  noise ladder.
- Each tile's gate costs about 70–80 scored variants (40 noise draws, controls for each class, refit
  redraws): about an hour for all six tiles on one m7i.16xlarge with 60 single-threaded shards.
- The measurement scripts and every output are in `s3://arbol-tessera-inputs-dev/perf/fp8-mixed/`,
  the per-class run under `perclass/`. The gate as decided is `scripts/perclass_rule.py`, and its
  reports are `perclass/rule_unres/`. The four new tiles' mosaics stay under `mosaics/fp8-*` for
  reruns.

## Rejected alternatives

- **A tile-level bound alone** (0.2 points). A class-directed shift of FP8's size costs one class up to
  2 points while the tile moves 0.06.
- **Widening δ to 3 × the refit spread.** The spread is real (1–5.6 points for minority classes) but
  measures how much a user's own classifier depends on its training sample, not what a change does to
  a fixed classifier. Using it ungates the minority classes the gate exists for.
- **Raising δ to 1.0, or to whatever FP8 needs.** That admits Iowa's soybean control and would set the
  bar from the candidate. FP8 is admitted by a named exception instead, which leaves the gate unchanged
  for the next change.
- **Gating coherence.** It does not track the size of the harm, and two identical-bf16 cards reach
  0.60 inside one class.
- **A fixed 0.2-point bound on any move, in either direction.** The metric's own noise reaches 0.13
  points on these tiles, and the largest move (+0.26) is a gain, the mirror of the 0.20 loss.
- **ADR 030's two-card comparison as the bar.** The fleet's cards agree too closely for it to measure
  anything but bit-level reordering.
- **ADR 012's 0.9940 top-20 gate.** bf16 on a different kernel stack fails it, and FP8's churn is
  swaps among near-tied neighbours that cost no accuracy.
- **FP8 emulated on the A10G** (E4M3-rounded operands through an fp32 GEMM, then the scales). It removes the shared shift inside
  a mixed store, holding it to ±0.02 points with coherence at the noise floor, but adds card-specific
  rounding to every forward path that a g5 runs, and does nothing for a classifier applied across a
  bf16 and an FP8 store, where the same losses appear.
- **g6e-only FP8 stores.** They give up g5 fallback, which g6e capacity has needed repeatedly, and the
  cross-store losses remain.
