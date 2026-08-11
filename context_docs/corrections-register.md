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

**Two documents already do this for themselves** — `ingest_concurrency_investigation_2026_08.md`
§"Corrections this investigation made to its own earlier claims", and
`ingest_optimization_campaign_2026_07.md` §5 "Claims made and withdrawn". Both work, and
both are blind past their own edges: the first noticed it had repeated one mistake three
times, while the same mistake sat uncorrected in two neighbouring documents it never cites.
This file is that habit widened to the corpus, not a replacement for either.

---

## The mechanisms

### 1. Two real measurements compared across an unlisted condition

The largest class by some distance, and the only one that has produced a headline finding
that was entirely an artefact.

- **The 1.8–2.1× "slowdown at fleet scale."** Compared May–September dates against a
  January baseline. Same zone, same width, different **season**. Matched on all three it is
  **1.17×**. Withdrawn in `ingest_concurrency_investigation_2026_08.md` §"E IS WITHDRAWN";
  the correction propagated to `campaign-cost-model.md` and to
  `ingest_optimization_campaign_2026_07.md`'s header, which had been superseded by it.
- **"Width may not be usable."** The "six times the fleet for under twice the rate" reading
  compared **different zones**, and the two narrow-fleet zones happened to be the two
  cheapest-per-chunk in the wave. Same-zone pairs show 6× workers buying 3.7–4.9×.
  (`campaign-cost-model.md`)
- **"The sweep's timing gap came from contention with concurrent automation testing."**
  Asserted from three paired dates with no mechanism. An identical catalogue query measured
  37.6 s at 21:45 and 33.1 s at 23:45 — external latency drifts ~12% over two hours on its
  own. (`ingest_optimization_campaign_2026_07.md` §5)
- Plus, by that investigation's own count, three more of its six corrections: zone in one,
  keep threshold in another, generalising a single zone in a third.

**The cure is written down already and it is mechanical:** before comparing two figures,
write down the conditions of each side and diff them. The July record carries a
reading-instructions block naming **zone, width and dates** as the three that must match.
Consulting it would have caught the season error in one minute. It was not consulted.

### 2. Two quantities in different units, divided or compared

- **The campaign's central cost division mixed token units for months.** The census counts
  **Sentinel-2 plus Sentinel-1** observations; the measured rate is computed from `t_kept`,
  which is the optical cloud mask's first dimension — **optical only**. So
  `GPU-hours = tokens ÷ tok/sec` divided an S2+S1 numerator by an optical-only denominator.
  (`campaign-cost-model.md` §6b, `inference-perf-run-ledger.md`)
- **"`t_kept` 145 remains a defensible planning figure."** 145 is a combined census figure
  and `t_kept` is optical, so "inside the observed 57–158 range" compared quantities in
  different units. (`campaign_inference_profile_2026_08.md`, correction 4)
- **Pixels per second versus tokens per second.** The cluster-sizing document's wall-clock
  and GPU-hour columns rest on a px/s figure measured over one region; inference cost scales
  with tokens, and px/s is a property of the pipeline *and the geography it ran over*.
  (`campaign-cluster-sizing.md`)

**The sharpest lesson in this class is about false reassurance.** The unit mismatch survived
because the two sides agreed to within 9%, which was read as validation. It was an
instrument self-check — the same optical quantity computed two ways. Radar is 91 of the
census's 143 tokens, so had `t_kept` included radar the two could not have agreed at all.
**They agreed *because* both were optical.** An agreement that a hypothesis predicts should
be impossible is evidence against the hypothesis, not for it.

### 3. Presence counted where coverage was meant

Three withdrawals, one instrument, and the error is always in the same direction: an
aggregate unit too coarse to see the thing being measured.

- **"Both orbits cover 95.8–98.6% of campaign land in every year."** That figure is **per
  zone**: a zone counted as dual-orbit if each orbit had granules *anywhere* over its live
  tiles. The 2022–2024 radar loss is **sub-zonal** — interior Australia, much of Siberia,
  inside zones whose coastal tiles keep their radar — so the instrument could not see it.
  Area-weighted per pixel: **81% covered, not 96–99%**.
  (`radar_source_coverage_2026_08.md`)
- **The land shares in the throughput table**, from the same per-zone survey, implied
  radar-free work was **1.2%** of the campaign. Per pixel it is **6.8%** of pixel-years —
  which makes the throughput split matter *more*, not less.
  (`campaign_inference_profile_2026_08.md`)

**The cure:** state the unit of aggregation in the claim itself. "Per zone" and "per pixel"
are different questions, and a percentage that does not say which is not yet a finding.

### 4. An in-flight measurement read as a finished one

- **The 37N/2021 rate deficit.** Read from a 203-chunk in-flight sample as a ~30% deficit;
  completed at 4,859 chunks it is ~19%. The same cell's latitude span read 30–32° partial
  and **0.1–32.1°** complete, and its radar depth 146.8 partial against **89.9** complete.
  (`inference_cost_basis_revision_2026_08.md`, `campaign-cost-model.md` §6c)

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
  (`campaign_inference_profile_2026_08.md`)
- **"Nothing measured is slower than the model assumes."** True in aggregate, false once
  stratified by radar status — and it reverses for the population that matters.

**The cure:** before reporting a correlation, name what else moves with the variable. Here
one stratification flipped the sign of the conclusion.

### 6. A model fitted on too few points, or one sample generalised

- **"30–45 workers, ~20% better than 120."** The width curve was fitted from **two** paired
  points, which cannot constrain a two-parameter model. A third control at 45 workers put
  the serial constant anywhere from 11.4 to 39.3 s, and the three-point fit makes aggregate
  throughput **flat within ~6% from 20 to 120 workers**. There is no optimum.
  (`ingest_optimization_campaign_2026_07.md`)
- **"Batching wins 1.14×, adopt it globally."** One point on a curve that is **not
  monotonic** — the same setting *loses* on two of four further regions. Batching is now
  chosen per region by a size threshold.
- **The 1.04× per-cell interference penalty**, from a single two-cell measurement, implied
  2.56× at 40 cells. None is measurable to 20 concurrent cells.
- **A linear cost fit from two runs failed its first out-of-sample test** — predicted 122 s
  at three windows, measured 193.9 s. (`ingest-graph-and-stac-budget.md`)

**The cure:** a two-parameter model needs a third point before it is a model, and a curve
needs enough points to show it is monotonic before one of them becomes a recommendation.

### 7. A mechanism asserted to explain a result, before being measured

- **"Most of the dead area is not geometric — it is cloud."** Backwards: geometric dead is
  55–66% of live chunks and radiometric 10–21%. *"The claim was made to explain the null
  result and was not measured before being asserted."*
  (`ingest_optimization_campaign_2026_07.md` §5)
- **Two mechanism accounts for the overlap gain, both refuted.** Sum-over-max predicted the
  gain should scale with window count — flat across a 3.3× spread. Fleet occupancy predicted
  it should grow with fleet width — 3.67× at 30 workers against 3.85× at 60, at the noise
  floor. **The gain is real and reproducible; why it is that size remains unexplained**, and
  the document says so rather than proposing a third story.
- **"Looks like a production backlog."** It was a satellite failure — Sentinel-1B, December
  2021. (`radar_source_coverage_2026_08.md`)

**The cure:** an unexplained result is a publishable state. Recording "the effect is real
and the mechanism is unknown" costs nothing and blocks nothing; a mechanism invented to
close the gap becomes load-bearing for later decisions and then has to be dug out of them.

### 8. A correction applied in one place and not the others

The only class here that is a process failure rather than a measurement failure — and the
one most likely to recur, because it is invisible to whoever makes it.

- **A profile section contradicted its own radar finding for a day**: it restated "cost per
  chunk tracks `t_kept`" as a surviving conclusion *two sections after* the finding that
  showed the range mixes two populations, then used the restatement to rule out re-basing
  the budget. In its author's words: *"I wrote the warning and then left the sentence it
  invalidates standing in the same file."*
- The ingest record notes the same failure independently: *"it has been violated twice by
  leaving an old number in one section while correcting it in another."*

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

`campaign-plan.md` §10 states the underlying rule: *"each was wrong in a way worth not
repeating, and a reviewer who only sees the corrected number learns nothing about how it
went wrong."* The decision records follow the same principle by being append-only — a
superseded record stays and a new one supersedes it.

This register is the index over that material, not a replacement for it.
