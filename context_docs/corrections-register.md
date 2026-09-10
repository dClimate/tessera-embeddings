# Corrections register

Every figure this programme has published and then withdrawn, grouped by **how it went
wrong** rather than by which document it was in. Audited 2026-08-11: **83 marked retractions
across the 14 documents that carried them.** That count is a dated measurement, not a live
one — a grep for "withdrawn" now also finds the prose in this file and in the two places that
link to it.

**This does not replace the withdrawals themselves.** Those stay next to the claims they
correct, and they have to: a withdrawn number a reader can still see, with the correction
attached, cannot be quoted by accident — the same number in a register nobody opened is
just a number in a document. Lifting figures out of the context that qualified them is how
several of the entries below happened in the first place.

What this file adds is the thing proximity cannot give: **the pattern.** Individually each
withdrawal reads as bad luck. Grouped, eight mechanisms account for all of them, most
recur across documents that never cite each other, and one document caught itself
repeating a single mistake three times without noticing the same mistake in its
neighbours.

Read it before publishing a figure, before quoting one, or when a review finding smells
familiar.

**Two records already do this for themselves**, and both now sit inside
[`ingest/ingest-performance.md`](ingest/ingest-performance.md) — §5 "Claims made and withdrawn",
and the fleet-scale investigation's own self-corrections, condensed into §11. Both work, and both
were blind past their own edges: the second noticed it had repeated one mistake three times, while
the same mistake sat uncorrected in two neighbouring documents it never cited. **Merging those
documents closes exactly that gap for one subject; this file is the same habit widened to the whole
corpus.**

---

## The mechanisms

### 1. Two real measurements compared across an unlisted condition

The largest class by some distance, and the only one that has produced a headline finding
that was entirely an artefact.

- **The 1.8–2.1× "slowdown at fleet scale."** Compared May–September dates against a
  January baseline. Same zone, same width, different **season**. Matched on all three it is
  **1.17×**. Withdrawn in [`ingest/ingest-performance.md`](ingest/ingest-performance.md) §11.3;
  the correction propagated to [`campaign/campaign-cost-model.md`](campaign/campaign-cost-model.md)
  §4 and to that record's own header.
- **"Width may not be usable."** The "six times the fleet for under twice the rate" reading
  compared **different zones**, and the two narrow-fleet zones happened to be the two
  cheapest-per-chunk in the wave. Same-zone pairs show 6× workers buying 3.7–4.9×.
  (`campaign/campaign-cost-model.md`)
- **"The sweep's timing gap came from contention with concurrent automation testing."**
  Asserted from three paired dates with no mechanism. An identical catalogue query measured
  37.6 s at 21:45 and 33.1 s at 23:45 — external latency drifts ~12% over two hours on its
  own. (`ingest/ingest-performance.md` §5)
- Plus, by that investigation's own count, three more of its six corrections: zone in one,
  keep threshold in another, generalising a single zone in a third.
- **The assembly pool's "2.92 GiB per worker at both sizes", and the peaks it was built from.**
  Read from the CloudWatch metric `Maximum` over `{ClusterName, TaskDefinitionFamily}`, where
  **each statistic is aggregated across tasks independently** — so the unlisted condition was
  *which task*, and a minute's memory maximum and its CPU maximum could belong to different ones.
  They do: per task the peak-memory and peak-CPU minutes are different tasks in both families, so
  filtering minutes by CPU never restricted the memory figures to the tasks that were assembling.
  **What the route could not do was ESTABLISH a figure, which is not the same as getting one
  wrong** — of the four it produced, 93.5 GiB and 99.7% CPU are confirmed per task, while 46.7 GiB
  and 96.8% are not reproducible (40.1 GiB and 95.8% are the highest attributable) and belong to a
  run outside the log's retention. The per-worker claim is withdrawn outright: the two ratios are
  2.51 and 2.92, and their apparent agreement was a coincidence between one confirmed number and
  one that could not be checked. See
  [`assembly/what-bounds-assembly-2026-09-09.md`](assembly/what-bounds-assembly-2026-09-09.md).
- **The same pool's "41.1 GiB peak", one round earlier.** The largest of forty one-minute samples
  on a **single task**, published as the pool's peak. Note what cannot be said about it: that run
  is outside the performance log's 24-hour retention, so where the figure fell in its own run's
  distribution is unrecoverable — it is neither confirmed nor shown to be typical, only shown to
  be ungeneralisable. The generalising half of this is mechanism 6.

**The cure is written down already and it is mechanical:** before comparing two figures,
write down the conditions of each side and diff them. The July record carries a
reading-instructions block naming **zone, width and dates** as the three that must match.
Consulting it would have caught the season error in one minute. It was not consulted.

### 2. Two quantities in different units, divided or compared

- **The campaign's central cost division mixed token units for months.** The census counts
  **Sentinel-2 plus Sentinel-1** observations; the measured rate is computed from `t_kept`,
  which is the optical cloud mask's first dimension — **optical only**. So
  `GPU-hours = tokens ÷ tok/sec` divided an S2+S1 numerator by an optical-only denominator.
  (`campaign/campaign-cost-model.md` §6b, `inference/inference-on-gpus.md`)
- **"`t_kept` 145 remains a defensible planning figure."** 145 is a combined census figure
  and `t_kept` is optical, so "inside the observed 57–158 range" compared quantities in
  different units. (`inference/inference-on-gpus.md`, correction 4)
- **Pixels per second versus tokens per second.** The cluster-sizing note's wall-clock and
  GPU-hour columns rested on a px/s figure measured over one region; inference cost scales with
  tokens, and px/s is a property of the pipeline *and the geography it ran over*. Those columns are
  gone: what survives of that note is [`campaign/campaign-cost-model.md`](campaign/campaign-cost-model.md)
  §5b, which is about how work BALANCES and carries no throughput figure at all.

**The sharpest lesson in this class is about false reassurance.** The unit mismatch survived
because the two sides agreed to within 9%, which was read as validation. It was an
instrument self-check — the same optical quantity computed two ways. Radar is 91 of the
census's 143 tokens, so had `t_kept` included radar the two could not have agreed at all.
**They agreed *because* both were optical.** An agreement that a hypothesis predicts should
be impossible is evidence against the hypothesis, not for it.

- **"62% of the fleet is the cheaper card."** A share by *instantaneous count*, quoted where the
  cost argument needs a share by *card-hours*; by hours it is **57.3%**. The two are different
  measurements of a fleet whose composition changes hour to hour, and only one of them multiplies
  against a price.
  ([`campaign/campaign-cost-model.md`](campaign/campaign-cost-model.md) §12)

### 3. Presence counted where coverage was meant

Three withdrawals, one instrument, and the error is always in the same direction: an
aggregate unit too coarse to see the thing being measured.

- **"Both orbits cover 95.8–98.6% of campaign land in every year."** That figure is **per
  zone**: a zone counted as dual-orbit if each orbit had granules *anywhere* over its live
  tiles. The 2022–2024 radar loss is **sub-zonal** — interior Australia, much of Siberia,
  inside zones whose coastal tiles keep their radar — so the instrument could not see it.
  Area-weighted per pixel: **81% covered, not 96–99%**.
  (`campaign/radar-coverage-by-zone.md`)
- **The land shares in the throughput table**, from the same per-zone survey, implied
  radar-free work was **1.2%** of the campaign. Per pixel it is **6.8%** of pixel-years —
  which makes the throughput split matter *more*, not less.
  (`inference/inference-on-gpus.md`)

**The cure:** state the unit of aggregation in the claim itself. "Per zone" and "per pixel"
are different questions, and a percentage that does not say which is not yet a finding.

### 4. An in-flight measurement read as a finished one

- **The 37N/2021 rate deficit.** Read from a 203-chunk in-flight sample as a ~30% deficit;
  completed at 4,859 chunks it is ~19%. The same cell's latitude span read 30–32° partial
  and **0.1–32.1°** complete, and its radar depth 146.8 partial against **89.9** complete.
  ([`campaign/campaign-cost-model.md`](campaign/campaign-cost-model.md) §6c; originally recorded in
  a since-deleted working note, in git history)

- **Three of the campaign cost model's §12 figures, all taken on 2026-09-02 while the campaign
  was a third done.** "The whole campaign will cost **$555,000**" became **$816,901** measured;
  "the ingest containers use 964 of 25,000 vCPU, **under 4%**" became a **73% peak** on 08-27,
  because the reading was taken after the heavy ingest had finished; and "**83%** of single-card
  basis" became **86.6%**, because it multiplied a snapshot fleet count by an elapsed period
  instead of using the billed hours. The pattern is one reading, taken mid-campaign, phrased as a
  property of the campaign.
  ([`campaign/campaign-cost-model.md`](campaign/campaign-cost-model.md) §12)

- **"Per-chunk cost varies 7× by zone."** The 7× included a 0.029 reading that the very next
  sentence rejected as a partial-resumption artefact — resumed tiles cost nothing and drag a leg's
  average down. The uncontaminated spread is **2.2×**. Sizing a spread with a figure you have just
  declared invalid overstates sustainable variation, and it weakened the fleet-sizing conclusion it
  was meant to support.
  ([`campaign/campaign-cost-model.md`](campaign/campaign-cost-model.md) §5)

**A partial run's whole-cell summary is biased, not merely imprecise** — the chunks that
have finished are not a random sample of the cell, because scheduling order correlates with
geography. The profile document now refuses to carry figures from a run still in flight,
which is the right guard: precision language ("preliminary", "±") does not describe this
error, and using it invites the number to be quoted anyway.

### 5. Confounded variables read as one

- **"Cost per chunk tracks `t_kept`."** The deepest cell measured is *also among the
  cheapest*, because it is radar-free. Depth and radar status are confounded across the
  measured set, and the document read the combination as depth alone. The $0.115–$0.211
  range it produced mixes two populations.
  (`inference/inference-on-gpus.md`)
- **"Nothing measured is slower than the model assumes."** True in aggregate, false once
  stratified by radar status — and it reverses for the population that matters.

**The cure:** before reporting a correlation, name what else moves with the variable. Here
one stratification flipped the sign of the conclusion.

### 6. A model fitted on too few points, or one sample generalised

- **"30–45 workers, ~20% better than 120."** The width curve was fitted from **two** paired
  points, which cannot constrain a two-parameter model. A third control at 45 workers put
  the serial constant anywhere from 11.4 to 39.3 s, and the three-point fit makes aggregate
  throughput **flat within ~6% from 20 to 120 workers**. There is no optimum.
  (`ingest/ingest-performance.md`)
- **"Batching wins 1.14×, adopt it globally."** One point on a curve that is **not
  monotonic** — the same setting *loses* on two of four further regions. Batching is now
  chosen per region by a size threshold.
- **The 1.04× per-cell interference penalty**, from a single two-cell measurement, implied
  2.56× at 40 cells. None is measurable to 20 concurrent cells.
- **A linear cost fit from two runs failed its first out-of-sample test** — predicted 122 s at
  three windows, measured 193.9 s. Superseded by a dispatch-floor model
  ([`ingest/ingest-performance.md`](ingest/ingest-performance.md) §12.2). **The retired fit
  survived in two other documents for weeks after it was superseded**, which is mechanism 8 below
  and is the clearest single argument for merging them.

- **"The assembly pool is linear in worker count at 2.92 GiB per worker."** Two points, from
  **different runner families running different cells**, whose per-worker quotients happened to
  agree to three significant figures — and only one of the two was ever confirmed (mechanism 1
  above), so the agreement was a coincidence between a real number and one that could not be
  checked. Per task the ratios are 2.51 GiB at 16 workers and 2.92 at 32; a line through them
  implies a **negative** fixed overhead of −13 GiB, which is the arithmetic saying the two points
  cannot separate overhead from per-worker cost. **The correction is symmetric, and the first
  attempt at it was not:** whole-task peaks from different families and cells cannot establish
  non-linearity either, so the finding is that linearity is *unestablished*, not that the pool is
  non-linear. A second salvage attempt, "use the larger ratio as an upper bound", was withdrawn
  as well: nothing measured bounds a third pool size.
  ([`assembly/what-bounds-assembly-2026-09-09.md`](assembly/what-bounds-assembly-2026-09-09.md))

- **The assembly-concurrency margin of 1.10×, and the 275-actor ceiling and ten-cluster split
  derived from it.** A ratio of two modelled values on one cell, used to locate a crossover and
  then to choose a fleet shape. Both terms were later re-measured at 1.7–2.6× their modelled
  values — the same cell's assembly took **5.78 h** against the 3.28 h the ratio assumed — so the
  crossover had no supported location. The campaign then ran **25 clusters of 100**, testing
  neither the cap nor the split.
  ([`campaign/campaign-cost-model.md`](campaign/campaign-cost-model.md) §6c, §10)

**The cure:** a two-parameter model needs a third point before it is a model, and a curve
needs enough points to show it is monotonic before one of them becomes a recommendation.
**Two points agreeing is not evidence when both come from the same broken instrument.**

### 7. A mechanism asserted to explain a result, before being measured

- **The assembly pool's "1 to 1.5 GB per worker", and the 24 GB and 48 GB pool figures built on
  it.** Published in the `AssemblyConfig` docstring and in two **public** documents as sizing
  guidance, having never been measured at any pool size; a 16-worker pool measures near 40 GiB, so
  the figure was low by more than half and an operator following it would have under-provisioned a
  runner into an OOM. Filed here because an unmeasured estimate presented as a measurement is the
  same failure as an unmeasured mechanism presented as a cause. Its *propagation* is mechanism 8:
  the estimate reached `docs/configuration.md`, `docs/prefect-setup.md`, the storage record and two
  test docstrings, and the public guidance was the last of them to be corrected.
  ([`assembly/what-bounds-assembly-2026-09-09.md`](assembly/what-bounds-assembly-2026-09-09.md))
- **The campaign cost model's two smallest lines, "S3 requests ~$1,600" and "transient mosaic
  storage ~$3,000."** Neither was ever measured; the outturns are **$26,167** and **$70,821**, 16×
  and 24× low, and together $92,000. The storage estimate even names the sensitivity that broke it
  — "if inference lags, it grows linearly with the backlog" — and then does not size it. An
  estimate small enough to skip re-deriving is exactly the one nobody checks.
  ([`campaign/campaign-cost-model.md`](campaign/campaign-cost-model.md) §7, §12)
- **"Assembly is CPU-bound."** Asserted from utilisation, which shows the processor is busy at
  moments, not that it is the constraint that binds the rate of work — while Fargate publishes no
  per-task network allowance, so the competing candidate was never measurable from outside. The
  *utilisation* survives and is now attributable: 95.8% of 16 vCPU and 99.7% of 32, each one task
  in one minute. The *inference* from it does not, and the question is now stated as open, needing
  a controlled run.
  ([`assembly/what-bounds-assembly-2026-09-09.md`](assembly/what-bounds-assembly-2026-09-09.md))
- **"Most of the dead area is not geometric — it is cloud."** Backwards: geometric dead is
  55–66% of live chunks and radiometric 10–21%. *"The claim was made to explain the null
  result and was not measured before being asserted."*
  (`ingest/ingest-performance.md` §5)
- **Two mechanism accounts for the overlap gain, both refuted.** Sum-over-max predicted the
  gain should scale with window count — flat across a 3.3× spread. Fleet occupancy predicted
  it should grow with fleet width — 3.67× at 30 workers against 3.85× at 60, at the noise
  floor. **The gain is real and reproducible; why it is that size remains unexplained**, and
  the document says so rather than proposing a third story.
- **"Looks like a production backlog."** It was a satellite failure — Sentinel-1B, December
  2021. (`campaign/radar-coverage-by-zone.md`)
- **"`odc` mosaics by a painter's algorithm, so the last item wins a pixel."** Asserted for the
  whole life of the code and never measured. `odc.loader`'s default fuser writes only where the
  destination is still empty, so the FIRST valid source wins. Both ingest paths sorted
  cloudiest-first on the strength of it, handing every same-day overlap to the CLOUDIEST scene
  available. Two things kept it alive: the library's own docstring says "only fill where **src** is
  nodata" where the code masks on **dst**, so checking the docstring confirmed the error; and every
  test in the area asserted the order handed *to* the loader, never the pixels that came *out*.
  (`ingest/solar-day-fusion-order.md`)

**The cure:** an unexplained result is a publishable state. Recording "the effect is real
and the mechanism is unknown" costs nothing and blocks nothing; a mechanism invented to
close the gap becomes load-bearing for later decisions and then has to be dug out of them.

### 8. A correction applied in one place and not the others

The only class here that is a process failure rather than a measurement failure — and the
one most likely to recur, because it is invisible to whoever makes it.

- **The fix for the fusion ordering swept "descending" to "ascending" and left "last" standing.**
  `group_items_by_date`'s docstring claimed a CLOUD-DESCENDING sort one line above "the clearest
  tile comes first", and the ingest README described the per-date baseline as "the last item's, the
  clearest tile, because the query sorts cloud-ascending" — wrong twice over, since under ascending
  the last item is the cloudiest and the code now keeps the first. Both caught in review of the
  correcting change itself. (`ingest/solar-day-fusion-order.md`)
- **A profile section contradicted its own radar finding for a day**: it restated "cost per
  chunk tracks `t_kept`" as a surviving conclusion *two sections after* the finding that
  showed the range mixes two populations, then used the restatement to rule out re-basing
  the budget. In its author's words: *"I wrote the warning and then left the sentence it
  invalidates standing in the same file."*
- The ingest record notes the same failure independently: *"it has been violated twice by
  leaving an old number in one section while correcting it in another."*

- **The 1.10× withdrawal was struck in one paragraph and left running the document.** Review of
  the correcting change found the withdrawn crossover still sizing clusters at "≤275 each" in §5's
  active table and still choosing "10 clusters of 250" in §10's action list, and the invalidated
  per-zone cost still multiplying every fleet size by a single 289.2 GPU-hour average. Three
  reviewers raised it across two rounds before it was swept. **The strike-through is not the
  withdrawal; the grep is.**
- **A related one, found only by the campaign finishing: §5's durations assumed a fleet nobody had
  committed to.** Every table read "if the fleet is 2,500 actors at full basis", and none said how
  wide the fleet would actually be. It averaged 961. The arithmetic was sound — fed the measured
  work and rate it reproduces its own answer — so the miss was an unstated input, not a wrong
  calculation, which is the hardest kind to grep for.
  ([`campaign/campaign-cost-model.md`](campaign/campaign-cost-model.md) §5, §12)

**The cure is a grep, and it is the cheapest item in this file:** when a figure is
withdrawn, search every document for the number and for the phrase, not just for the
section you were editing. Both instances above would have been caught by searching for the
figure itself.

---

## Before publishing a figure

Distilled from the eight above. Each line exists because skipping it cost a withdrawal.

1. **Name the conditions** — zone, width, dates, season, fleet state. If you are comparing,
   diff both sides' conditions first.
2. **Name the unit** — optical or combined tokens; per pixel or per zone; pixels or tokens
   per second.
3. **Is the run finished?** If not, do not report a whole-cell figure from it at all.
4. **What else moves with this variable?** Stratify before reporting a correlation.
5. **How many points?** Two cannot fit two parameters, and one region is not a curve.
6. **Is the mechanism measured or assumed?** "Unexplained" is an acceptable answer.
7. **Where else does this number appear?** Grep for it before you finish.

## Why the withdrawals stay in place

Recorded here so this file is not mistaken for a proposal to consolidate them.

The withdrawals live next to the claims they correct because that is where they do their
work — a reader reaching for a figure has to read past the retraction to reach it. They are
also a small share of the text they sit in: across the documents that carry them they are
**under a tenth** of the lines, and the remainder is the derivation those documents exist
for. Extracting them would shorten nothing meaningfully and would retire no document, while
moving each correction one lookup away from the number it corrects.

[`campaign/campaign-plan.md`](campaign/campaign-plan.md) §11 states the underlying rule: the
stage records *"carry their own withdrawn claims beside the corrected ones, on purpose — a
reviewer who sees only the final number learns nothing about how it went wrong"*. The decision
records follow the same principle by being append-only — a superseded record stays and a new one
supersedes it.

This register is the index over that material, not a replacement for it.
