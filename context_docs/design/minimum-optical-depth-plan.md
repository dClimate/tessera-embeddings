# Minimum optical depth: refuse pixels below 15 observations — decision and evidence

**Status: BUILT.** `OPTICAL_MIN_OBS = 15`, the gate applies it per pixel per year, the store stamps
it as write-once root identity, and refused shards record why. The decision is
[ADR-018](../decisions/018-refuse-pixels-below-minimum-optical-depth.md); this document is the
evidence behind it and the record of how the number moved.

> **Consolidated 2026-08-18, from 1,432 lines.** Eight work-item sections specified machinery that
> has since shipped, and the code is a better record of it than a plan is — §4-11 now says where each
> piece lives. The deferred-spend reasoning became ADR-018 and the cross-repo validation move became
> ADR-019, both of which it was already written "for". The month-plane implementation defects are
> compressed to the three things about that failure which generalise. What is kept in full is the
> measurement evidence, because that is what a future revisit of the line will need and what no code
> states.

**The line is 15**, decided by Robert and a colleague on **2026-08-17**.
**Coverage was chosen over reproducibility, knowingly.** The line keeps **94% of pixels against 79%
at 25** (pixel-weighted over 40 cells), and the accepted cost is that two independent embeddings of
the same ground agree less well. `OPTICAL_MIN_OBS` is 15 and the campaign's seeder registers 15.

**15 is not the reproducibility elbow, and that is the point of the decision rather than an oversight.**
The elbow is at 25: measured on 741 blocks of doubly-embedded ground, agreement runs three to four
times the pipeline's own floor everywhere below 25 and halves crossing into 25–30 (§11 of the
legibility report). A line at 15 therefore **admits the worst-reproducing band in the whole
measurement** — 15–19 sits at 3.83× the floor, worse even than under-15 at 2.21×. Whoever revisits
this should know the trade was made with that in front of them, not around it.

**Two things weigh on the other side, and both post-date the elbow.** Refusal is irreversible while
filtering is not: `s2_obs_count` is published per pixel, so any user can apply a stricter line at read
time and nobody can recover a pixel we refused. And **count is not a reliable proxy for temporal
spread** — 33.7% of pixels at 15–19 observations already cover ten or more months of the year, so
some of what the elbow refuses does describe a year (§1, 2026-08-17 measurement). Neither makes 15
*better* than 25 on the evidence; they are why a coverage-first reading is defensible.

**The history, because the number moved three times and the record matters more than the number.**
The original 30 rested on how the pictures looked; blind re-measurement showed legibility was tracking
rendered contrast rather than information, and spatial organisation is flat across 12 to 45
observations. That put 25 in as an explicit **placeholder** (2026-08-13) so the machinery could be
built while the evidence was gathered. 15 replaces it as a decision.

**Evidence:** [`optical_depth_census_2026_08.md`](optical_depth_census_2026_08.md) for the
pixel-vs-shard arithmetic; [`window_legibility_vs_depth_2026_08.md`](window_legibility_vs_depth_2026_08.md)
§11 for reproducibility against depth; and
[`optical_retention_per_pixel_2026_08.md`](optical_retention_per_pixel_2026_08.md) for what each
candidate line costs, counted per pixel — which withdrew the window-mean proxy every earlier cost
figure rested on.

> **Measurements below that name the 25 line stay as they are.** They are what was measured, at the
> line in force when they were taken, and several are line-independent in the way that matters (the
> chunk-versus-pixel comparison, the count-versus-spread relationship). Where a figure is specific to
> 25 it says so.

---

## 1. The rule

**A pixel is not embedded if it has fewer than 15 valid Sentinel-2 L2A observations in the
calendar year being filled.** The refused pixel is written as fill, exactly as an
out-of-ROI pixel is, and the reason is recorded.

Three decisions were taken with the rule and are **not open** for the implementer to
revisit:

| decision | ruling |
|---|---|
| **2017**, where the census brackets the line between 38.7% of shard-years below 20 and 65.7% below 30 | **Same rule, populate it anyway, warn about it.** No per-year threshold — one rule keeps the product consistent. 2017 publishes roughly a third populated, and that must be stated where a user meets it (§7.1), not left for them to discover. |
| **Temporal consistency** — a pixel can clear 15 in 2020 and fail in 2021 | **Per pixel, per year.** Holes move between years. No all-years gate, no stability mask in this iteration. |
| **Scope** | **Global campaign only.** The single-ROI path (`tessera_embeddings` flow, `plain` runner) keeps today's behaviour. |

### Why the line is where it is, and why it is safe to apply per pixel

The line was originally proposed at 30, on this argument: the repo's own quality measurement
was *crisp at 30+ observations, noisy below about 20*, across 15 cells, and 30 is the
conservative end of that band. That argument no longer stands, and what replaced it is
below.

> **That measurement has been re-taken and it does not hold (2026-08-13).** Two defects were found in
> the instrument that produced it, and since this rule is the one decision in the campaign that
> cannot be revisited later — a refused pixel does not exist, so recovery is a re-run rather than a
> filter change — the evidence was re-gathered before the rule shipped. The defects were:
>
> 1. **The figures were drawn with a per-window contrast stretch.** Each window was stretched to its
>    own 2nd–98th percentile, so a window whose ground varies little had its noise floor expanded to
>    full contrast *whatever its depth*. Some of the "noise below 20" was the renderer. The stretch
>    is now shared across a cell's windows.
> 2. **Depth was read per CELL, not per window.** A zone spans most of a hemisphere and depth falls
>    with latitude and with cloud, so a cell mean describes a mixed population. Measured: a cell of
>    mean 46.9 holds windows at 34.5 and 61.0, and its *thinnest* window is the one with the most
>    structure. Each window now records its own mean and tenth percentile.
>
> **The re-measurement has been taken and the gradient runs BACKWARDS.** Full report:
> [`window_legibility_vs_depth_2026_08.md`](window_legibility_vs_depth_2026_08.md). 713 windows from
> 40 cells sampled BY DEPTH, reviewed blind by six independent sessions over disjoint ranges with
> depth and flatness withheld in code. On the 671 fully-land windows:
>
> | depth band | n | % legible |
> |---|---:|---:|
> | under 12 | 15 | 60% |
> | 12–18 | 98 | **91%** |
> | 18–27 | 212 | 86% |
> | 27–30 | 79 | 77% |
> | 30–45 | 251 | **63%** |
>
> Legibility FALLS as depth rises. Tested rather than asserted: latitude alone explains nothing, but
> depth *within* the tropics — where 401 windows sit — runs 87% below 20 observations and 61% at
> 30–45, while mid-latitudes are flat. Depth is a proxy for climate and climate is a proxy for how
> much there is to see: the cloudiest tropics are rivers, clearings and smallholder farmland, and the
> clear tropics are uniform ground. Per-cell legibility runs from **11%** (Saharan desert, deep and
> empty) to **100%**.
>
> **Two numbers for whoever picks the line.** At every cutoff from 15 to 30, about **85% of what it
> refuses is legible**; only at 12 does that drop to 60%. And the illegible share of what a cutoff
> KEEPS rises monotonically, 18% at 12 to **30% at 30** — refusing more makes the surviving product
> proportionally worse, because the unreadable windows are mostly the deep ones.
>
> **The limit that outranks all of this: legibility is not utility.** The picture is a 3-component
> projection of 128 dimensions, and one reviewer measured seven of its illegible windows as carrying
> coherent structure at a tenth of the amplitude the stretch renders visible. The rule's argument
> rests on usefulness, which no measurement has tested and — with no downstream model in existence
> (2026-08-13) — none can.
>
> **Everything else in this plan stands.** The registry, the per-shard record, the skip-marker payload
> and the cycle model are all wanted whether the line refuses pixels or merely labels them.

> **Count is NOT a reliable proxy for temporal spread, and it is weakest exactly at the line
> (2026-08-17).** A year-long embedding is supposed to describe a year, so what plausibly matters is
> whether the year is *covered*, not how many observations there are. Count and spread are correlated
> by construction — 60 observations must span the year, 6 cannot — so the question is whether the
> correlation is tight enough that gating on count already gates on spread. Measured off the zone
> mosaics' SCL, which is the same input the gate sees: **24 cells, three 4096-px tiles each,
> 919,224,646 pixels.**
>
> | count | pixels | mean months covered | ≥10 months | all 12 | ≥2-month gap | ≥3-month gap |
> |---|---:|---:|---:|---:|---:|---:|
> | 1–9 | 16,151,138 | 5.1 | 0.0% | 0.0% | 98.9% | 72.0% |
> | 10–14 | 21,392,541 | 7.7 | 6.8% | 0.0% | 76.2% | 35.8% |
> | 15–19 | 49,196,955 | 9.0 | 33.7% | 1.3% | 54.9% | 15.2% |
> | **20–24** | 55,488,166 | 9.9 | **64.3%** | 7.8% | 38.2% | **15.1%** |
> | 25–29 | 59,745,522 | 10.7 | 85.8% | 29.3% | 19.9% | 10.2% |
> | 30–39 | 207,201,979 | 11.5 | 98.5% | 62.1% | 4.2% | 0.7% |
> | **40+** | 510,048,345 | **11.1** | **81.2%** | 65.8% | **19.0%** | 5.0% |
>
> **Two errors the rule cannot see, and both are large.** Of the pixels at 20–24 observations that
> the line refuses, **64.3% already cover ten or more months** — the refusal is discarding data that
> describes the year. And **the 40+ band is worse-distributed than the 30–39 band** (11.1 months
> against 11.5, and 19.0% with a two-month blind gap against 4.2%), so the line admits, unexamined,
> deep pixels that are blind for a season.
>
> **The 40+ inversion is real but concentrated in a third of cells**, not an artefact of one:
> 03N/2021 44.4%, 06N/2021 56.0%, 23N/2021 60.5%, 37N/2021 25.7%, 38N/2021 22.2%, 53N/2021 33.7%,
> 60N/2020 36.5% have a two-month gap on a large share of their deepest pixels (26S/2022 reads 100%
> on only 394k pixels — small n). The other sixteen cells read **0.0–4.6%**. The pattern is a strong
> clear season: many observations, all in one window.
>
> **And whether the refused near-miss population is well distributed is a per-cell property.** Among
> cells with over a million pixels at 20–24, the share covering ten or more months runs from **38.2%
> (58S/2022) to 91.1% (58S/2025)**. One global count line cannot express that; a spread rule could.
>
> **What this does NOT show, and it is the load-bearing gap:** that better-spread pixels produce
> better embeddings. It establishes only that spread is an axis *independent of* count — necessary
> for a spread rule to be worth anything, nowhere near sufficient. The experiment that would settle
> it is the zone-overlap reproducibility measurement re-stratified by spread **at fixed count**, and
> it is runnable: 57S/2022–58S/2022 and 58S/2022–59S/2022 are adjacent pairs whose mosaics are both
> still readable.
>
> **Limits.** Three tiles per cell, so three locations — within-cell variation between tiles is not
> captured. Only **24 of 40** complete cells have a readable mosaic at all; the rest kept orphaned
> chunk objects after their Icechunk metadata was deleted, and their SCL is unrecoverable, so the
> measurable subset is whatever survived cleanup rather than a random sample. "Months covered" is a
> crude spread statistic — twelve observations one per month scores the same as twelve in a fortnight
> plus scattered singletons.
> **Spread is retained as twelve labelled month planes, and they are BUILT (`s2_month_covered`).**
> A packed uint16 bitmask was rejected: the only argument for it was size, and measured that is 14%
> rather than the 6x the proposal assumed, because embeddings barely compress and the planes are a
> rounding error beside them. **Cost to the finished store: 0.025%, about 353 GiB** — cross-checked
> against `campaign-cost-model.md`'s independent 0.9–1.8 PB estimate, inside which 1.53 PB sits.
> Labelled planes also mean `cov.sel(month=7)` says July to a reader who has never seen this
> document, which a bitmask cannot.

> **A refused shard records WHY, in its own marker, and the per-shard registry is BUILT.** Three
> defects made this necessary rather than nice: a fully refused shard used to discard its counters,
> so a thin-depth refusal was indistinguishable from no optical coverage at all; "no records" and
> "every read failed" were the same empty dict; and nothing checked that the reasons partition the
> refused set. The markers are read at ASSEMBLY, which is the last moment they exist — they live with
> the staging prefix and go when it does.
>
> **The marker's PRESENCE is load-bearing**, which is the part a reporting-shaped fix would have
> missed: absent markers and absent refusals look identical downstream, so `unreadable_markers` is a
> separate signal from "nothing was refused". Only FULLY refused shards need a record; for a shard
> that was written, the per-pixel `s2_obs_count` arrays in the store are already the evidence.
>
> Design and red-team detail is in `inference/assembly.py` and `tests/unit/test_skip_registry.py`,
> which are now the record. What belongs here is why it exists.

> **OPTICAL DEPTH IS THE ONLY REFUSAL RULE — and radar was silently refusing land too
> (2026-08-18).** A DECISION (Robert), and a correction to what the campaign was actually doing. The
> per-pixel gate in `inference/dataset.py` is `has_optical & deep_enough`, then `if not
> allow_s2_only: &= has_radar`. `allow_s2_only` defaults to **False** and no deployment set it, so
> every pixel with zero S1 observations was refused — and a tile with no radar coverage had *every*
> pixel refused, wrote a skip marker, and published as fill.
>
> Measured on the overnight four-cell run:
>
> | cell | live tiles | skipped | published |
> |---|---:|---:|---:|
> | 40S/2023 | 58 | **43** | 59.7M px, ~24% of its land |
> | 40S/2022 | 58 | **43** (the same 43) | 58.6M px |
> | 02S/2023 | 76 | 34 | 173.7M px |
> | 47S/2023 | 239 | 4 | 858.9M px |
> | 15S/2023 | 59 | 0 | 225.2M px |
>
> **The identical 43 tiles in two independent years is what identified the cause**: radar orbit
> footprints are fixed geometry, so the same ground lacks radar every year. Cloud cover cannot
> reproduce that. Two other explanations were tested and refuted first — reflectance bands missing
> where the SCL is valid (0.0% of valid observations lack a red band, on surviving 16S/17S mosaics),
> and a coordinate error in the comparison (the mosaic's northing at its own index equals the store's
> at the resolved index).
>
> **Too much land to weed out, so the cost is accepted.** Radar-free pixels are embedded through the
> upstream v1.1 missing-S1 convention (an all-zeros normalised S1 slice); their embedding quality is
> unvalidated for an S1-trained checkpoint. The alternative was losing the majority of some zones.
> `CAMPAIGN_ALLOW_S2_ONLY = True` is registered on the driver AND both fill deployments — a fill
> dispatched by hand takes its own default, which is exactly how the overnight cells got the old
> policy.
>
> **The line is STRICTLY FEWER than 15.** A pixel with exactly 15 valid optical observations is
> embedded; 14 is refused. The gate reads `s2_valid_count >= optical_min_obs` and the thin count
> reads `< thin_below`, so both agree, and a test now pins 13/14/15/16 at the campaign's own value
> rather than at a stand-in line.
>
> **Three defects in the RECORD, all fixed by the per-shard registry (2026-08-18).** A fully refused
> shard discarded its counters; "no records" and "every read failed" were the same empty dict; and
> nothing checked that the reasons partition the refused set. All three are closed and pinned — see
> the registry paragraph above and `tests/unit/test_skip_registry.py`.
>
> **The month array shipped wrong once, and how it failed is worth more than the fix.** The first
> cell published twelve all-empty planes while every array beside them was correct. The cause was
> ENUMERATION rather than the array: assembly chose which staged variables to copy by a whitelist
> the new variable was not on, so it staged fine, was never copied, and the destination kept its
> fill value — which for a bool plane reads as "no pixel had coverage" rather than as "nothing was
> written". Fixed in `ac932c6`, verified in `fc16f57` on 40S/2022 and 28S/2025, and reproduced
> against the published store with the reader expressions run verbatim rather than reasoned about
> (`cov.sum("month")`, `cov.sel(month=7)`, `cov.all("month")` all work with no helper).
>
> **Three things about the failure generalise.**
>
> No test caught it because every assembly test hand-rolled its destination group, so none exercised
> the enumeration that decides what gets copied. A test that builds its own destination cannot catch
> a destination-selection bug.
>
> **A well-observed window cannot verify this array.** A dense window scores full marks whatever the
> code does; only thin windows make "100% covered" mean something, which is why verification used
> 28S's thinnest window and checked that months 10 and 11 are absent there exactly as the mosaic says.
>
> **The array is `int8` on disk with the attribute `dtype="bool"`**, because the write path cannot
> emit bool directly — and a test staging with raw zarr keeps bool, so it cannot see the production
> representation at all.
>
> Two operational notes for anyone verifying a future cell: the mosaic is the only external check and
> is deleted once a cell lands, so the snapshot has to be taken between ingest finishing and the fill
> completing — a fill cancelled *before* its cell lands keeps the mosaic. And a single-timestep probe
> for "is this window populated" wrongly concluded 28S had no populated window at all, because a
> partial-swath date reads empty over good land. Probe several.
>
> 09S/2022 keeps its empty planes: the store is write-once, so a wrong array is superseded by a fresh
> store rather than repaired.
The census that produced the campaign figures measured **shard means**, and this rule
refuses **individual pixels** — a different question, and the class of error the
corrections register calls *presence counted where coverage was meant*. It was therefore
measured directly against `s2_obs_count` as written, over **844 chunks and 55.3M populated
pixels in 15 zones**:

| line | % of PIXELS below | % of CHUNKS whose mean is below | ratio |
|---|---:|---:|---:|
| 15 | 2.27% | 2.25% | 1.01 |
| 20 | 3.73% | 3.79% | 0.98 |
| **30** | **10.90%** | **11.37%** | **0.96** |
| 40 | 21.66% | 21.45% | 1.01 |

The two views agree to within 4%, so the census's shard-level figures carry over. (The
absolute level in that sample is below the census's global 18.4% because the sampled zones
are dominated by dry 37N/38N; the **ratio** is the transferable result, not the level.)

The loss is also concentrated rather than smeared — at the 30 line, **90.4% of refused
pixels sit in chunks whose own mean is already below 30**, and chunks averaging 45+
contribute 0.1%. This is what makes a per-shard registry an honest summary: refusals
cluster, so a shard-level number is not hiding a diffuse scatter.

> **What this section does NOT give is the per-cell cost**, and the two are easy to confuse:
> a small global percentage is compatible with individual cells losing nearly everything.
> Counted per pixel over 40 cells, the median cell keeps 92% at the line and two keep under
> a tenth — [`optical_retention_per_pixel_2026_08.md`](optical_retention_per_pixel_2026_08.md).

#### The enforcement unit is settled, and the record was the open question

Applying the rule at a coarser unit than the pixel changes retention by **about one point at every
unit tested**, because retention is *bimodal* at chunk scale: a chunk is nearly all-in or nearly
all-out, so where the boundary is drawn barely matters. And the hole inside a nearly-full chunk is
**entirely a depth refusal** — no part of it is missing imagery — with the refused population
concentrated just under the line.

Two consequences, which are why the measurement was worth taking:

**Refusals cluster, they do not smear.** 90.4% of refused pixels sit in chunks already below the
line, so a later top-up can select whole shards by their own mean rather than hunting scattered
pixels. That is what makes a top-up affordable at all (ADR-018).

**So the enforcement unit was never the open question — the RECORD was**, and that is what the
per-shard registry answers.

##### Store-wide, over every complete cell

The six cells above span the depth range on purpose and are **not** a store-wide figure — they
include three of the thinnest cells there are, so they overstate the refusal. Re-run over every cell
the store calls complete, ten whole shards each: **40 of 48 cells, 940,376,064 land-live pixels.**
(The eight that contributed nothing had all ten sampled shards at the edge of the land-mask grid,
where the shard's 8×8 slice is incomplete; they are small zones, so the aggregate under-weights
those.)

Both candidate lines below come from **one pass of reads**, so the two columns describe identical
pixels and the only difference between them is the line. Note the strong/weak partition moves with
the line too: a chunk is strong when half its pixels clear *that* line, so lowering the line promotes
chunks as well as pixels.

| | line 20 | share | **line 25** | share |
|---|---:|---:|---:|---:|
| embedded under the rule | 795,839,925 | 84.63% | 708,200,433 | **75.31%** |
| thin, inside weak chunks (<50% full) | 119,958,492 | 12.76% | 209,302,978 | 22.26% |
| thin, inside strong chunks (≥50% full) | 22,950,927 | **2.44%** | 21,245,933 | **2.26%** |
| refused in total | 142,909,419 | 15.20% | 230,548,911 | 24.52% |
| land in strong chunks | 800,653,312 | 85.14% | 709,361,664 | 75.43% |
| land in weak chunks | 139,722,752 | 14.86% | 231,014,400 | 24.57% |

| | line 20 | line 25 |
|---|---:|---:|
| fill inside strong chunks | 97.1% | 97.0% |
| fill inside weak chunks | 13.0% | 8.7% |
| strong chunks' share of refused pixels | 16.1% | 9.2% |

Pixels with no optical observation at all: **0.17%** of land-live, at either line.

> **Correction.** An earlier version of this table gave embedded-under-the-line as 709,827,153
> pixels. That was `land − thin`, which credits the 1.6M pixels with no optical observation as
> embedded; they are neither embedded nor refused by the depth rule. The count is **708,200,433**.
> The percentage was unaffected at one decimal.

**The permanently-unrecoverable population is nearly invariant to the line: 2.44% at 20 against
2.26% at 25.** It rises slightly as the line *falls*, because lowering the line promotes chunks into
the strong class (85.1% of land against 75.4%) and each newly-promoted chunk brings its own
sub-line pixels with it. So **the wave-through decision is orthogonal to where the line sits** —
moving the line does not shrink the problem the proposal exists to solve, it only changes what
fraction of all refusals it represents (16.1% at line 20, 9.2% at line 25).

Both lines leave the population overwhelmingly near-miss: at line 20, 88.6% of thin-in-strong pixels
are within five observations of the line and 2.5% are under 10; at line 25, 81.7% and 1.7%.

**Cross-check:** retention against the *any-optical* denominator reads 75.4% here against the
retention study's 79.2% pixel-weighted — 3.8 points apart on entirely different sampling (ten whole
shards versus forty land-aimed inner chunks). Close enough to trust both; the gap is spatial spread.

**What the six-cell findings look like at store scale.** Bimodality holds and is if anything
sharper — a strong chunk is ~97% full at either line while a weak one is under 13%. Missing imagery
stays negligible, so the refusal really is a depth decision rather than an absence of data, and the
hole inside a strong chunk is entirely reachable by a rule change.

**The number that reframes the decision, at the 25 line: 90.8% of refused pixels are in weak
chunks.** The proposal rescues a tenth of the refused population and defers the other nine tenths to
a repair flow that does not exist. Its reach per cell runs from 0.00% (03N/2021, 23N/2021 — deep,
nothing to rescue) through a median of 2.30% to **7.28%** (30S/2023) and 6.35% (26S/2022), both
middling-depth cells. At the 20 line the same proposal covers 16.1% of a smaller refused population.

**Three things it does not settle**, all of which outrank the sizing:

1. **It trades a visible hole for an invisible one.** Today every published pixel in a strong chunk
   meets the line. Waving through makes such a chunk read 100% full with 4.2% of it below the line,
   detectable only by reading `s2_obs_count`. This is exactly what the blocking check in
   `embedding_validation_rules.py` was written to prevent, in its own words: *"not low quality, it
   is mislabelled — and unlike a thin cell, no picture of the data reveals it."* If the proposal
   ships, `optical_min_obs: 25` **must not** — the root attribute has to become `null` plus a
   separate field naming the chunk-level rule, or the store advertises a guarantee it does not keep.
2. **Permanence is a property of a repair flow that does not exist.** Until one does, every refusal
   is permanent, not 5% of them, and the argument taken literally says refuse nothing. Embedding
   every eligible pixel would cost under 8% more inference against ~1% for the targeted version, so
   cost is not what rules it out — a quality judgement about publishing near-empty chunks is.
3. **The benefit is a function of a threshold not yet chosen.** At a repair cutoff of 90% the
   rescued population shrinks; at 20% it grows. Choosing store contents now on a repair threshold
   chosen later is the weakest link in the argument.

---

## 2. Non-goals — do not build these

**No ingest-time skipping.** The only signal available before reading a byte is the
catalogue's **acquisition-date count**, which upper-bounds the observation count and is
therefore a sound, zero-false-positive skip test. It was measured and it does not pay:

| year | provably skippable pre-ingest | actually refused after masking |
|---|---:|---:|
| **2017** | **37.15%** | 65.69% |
| 2018 | 0.34% | 14.06% |
| 2019–2024 | 0.07–0.23% | 11.1–16.8% |
| 2025 | 0.07% | 7.58% |

Overall the bound catches **23% of refusals, and almost all of it is 2017**. Three further
reasons it is the wrong trade: the ingest unit is a **4096-px chunk**, so an entire 40 km
block would have to fail rather than a location; skipping ingest destroys the audit record,
because `s2_obs_count` is derived from the mask that was not built; and the mosaic is shared
with the radar legs. Changing the most fragile path in the system for a fifth of a percent
outside 2017 is not worth it.

**Also out of scope:** any change to the land mask; any change to the single-ROI path;
re-running dev cells already filled under the old rule; a per-pixel "years cleared" stability
mask (considered, deferred).

---

## 3. What already exists and must be built on, not reinvented

| thing | where | note |
|---|---|---|
| the per-pixel gate | `inference/dataset.py` ~L123, `valid_mask` | the enforcement point is one conjunct |
| per-pixel S2 depth | `s2_obs_count` array in every zone group | **written for every pixel from the mask bundle, independent of the gate** (`actors.py` ~L1229) — so a refused pixel's depth is still recorded |
| the embedded mask | `scales`, NaN except where embedded | `~isnan(scales)` IS the per-pixel embedded mask |
| per-chunk counters | `actors.py` ~L1328–1341, emitted on `CHUNK_SUMMARY` | `s1_free_px` / `s1_thin_px` / `s2_thin_px` — the pattern to copy |
| per-year roll-ups | `assembly.summarise_radar_coverage`, `assembly.summarise_optical_skips` | already answer "which live tiles published as fill" |
| the year record | `storage/shard_writer.run_provenance` | the schema's one owner |
| the resume trap | documented in `summarise_optical_skips` | a resumed leg reports synthetic successes **with no counters** — see §6 |
| skip markers | `assembly.ZarrWriter.write_skip_marker` | currently **zero-byte**; resume reads only the object NAME, so a payload can be added safely |

**One inference chunk is one shard** (`SHARD_PX = 2048` for both), so per-chunk accounting
in the actor *is* per-shard accounting. Nothing needs to be re-read to build the registry.

---

## 4-11. The build — SHIPPED, and the code is the record

Eight work-item sections stood here: the threshold recorded once and enforced, the gate, keeping
fully-refused shards auditable, the per-shard record, the registry files, reconciling the records
this contradicted, the tests, and the verification checklist. **All of it is built.** They specified
field names, file layouts, invariants and a test list, and every one of those is now settled in code
where it cannot drift from what runs:

| What it specified | Where it lives now |
|---|---|
| the threshold, recorded once | `config/inference.py` (`OPTICAL_MIN_OBS = 15`), stamped into the store root as write-once identity by `storage/global_store.py` |
| the gate | `inference/dataset.py`, applied per pixel per year |
| the store's rule is the only rule a fill may apply | `orchestration/runners/zone_fill.py` asserts it; the Prefect adapter substitutes it |
| refusal recorded per shard, with a reason | `inference/actors.py` writes the marker, `inference/assembly.py` reads it at assembly |
| the registry beside the store | `config/paths.py` (`optical_registry()`), a sibling of the Icechunk prefix rather than inside it |
| the tests | `tests/unit/test_skip_registry.py`, `test_dataset_v11.py`, `test_assembly.py` |

Kept from those sections, because the code states the mechanism and not the reason:

**The registry is a SIBLING of the store, never inside it.** Icechunk owns every key under its own
prefix — garbage collection enumerates that prefix and reconciles it against its own manifests — so a
Parquet file living there is at best unrecognised and at worst collected.

**The rule is part of the store's write-once root identity.** Not a per-run parameter, because a cell
filled under a different line than its neighbours is undetectable afterwards: a refused pixel is
indistinguishable from one that had no optical input. Moving the line therefore means a new store,
not a migration — which is the cost the decision in §1 was taken with in view.

**Three categories, not two.** Not-eligible (ocean or outside the ROI), eligible-and-embedded, and
eligible-and-refused. A percentage over the wrong denominator was the error this replaced: the land
mask extends about 11 km into the sea, so "share of the grid" and "share of eligible land" differ by
enough to change what a reader concludes.

## 12. Why the spend is deferred — see ADR-018

This section was written "for the ADR" and is now one:
[ADR-018](../decisions/018-refuse-pixels-below-minimum-optical-depth.md) carries the decision, the
deferral premium (a ~$50K deferral that costs MORE to recover later, not a saving), the top-up unit,
and the two objections that were raised against refusing rather than flagging — with the answers,
because those are what a reader of the ADR will raise.

## 13. Generations, update cycles, and what tags are actually for

Settled while reviewing this plan, and settled **now** because no cell has been tagged yet.
Once the campaign runs, its tags are write-once forever.

### The mismatch that made tag naming feel impossible

One mechanism was being asked to do five jobs:

| job | asked by | what it wants |
|---|---|---|
| idempotence — has this cell landed? | the campaign work list | a boolean per cell |
| generation — which version, how many? | the top-up plan (§12) | a counter and history, per cell |
| reproducibility — the exact store behind a result | debugging | a snapshot pin |
| release — the published product | users | a few curated names |
| audit — when, from what, under what rule | everyone | structured data per cell |

Tags are good at reproducibility and release, serve idempotence by accident, and serve
generation and audit badly. No naming scheme fixes that, because it is a mechanism mismatch
rather than a naming problem. Two further facts settle it:

- **A per-cell tag pins the WHOLE repo.** All 120 zone groups share one Icechunk repository,
  so `zone-33N-2021` is the entire store as of the moment that cell landed — a snapshot in
  which most other zones are still empty. Nobody wants to read that.
- **Per-cell reproducibility is not wanted** (repo owner, 2026-08-13): the latest state is
  what users ask for, and they think in **update cycles**, not per-cell dynamics.

### The model

**An update cycle is the user-facing unit.** One pass of the campaign — the initial fill, or
a later top-up batch once Element 84 publishes more imagery — is one cycle, labelled
`v1.0`, `v1.1`, and so on.

**Per-cell tags are dropped entirely.** Tags mark cycles and nothing else:

```
release-v1.0            <- what a user cites
release-v1.1            <- after a top-up batch
year-2021-complete      <- already exists, keep
snap-2026-09-14T0000Z   <- optional operational pins
```

Prefixed by kind so `list_tags()` filters cleanly. ISO-8601 without colons, so lexicographic
order is chronological order and shells and URLs stay unharmed. **Never encode a mutable
fact** — a count, a percentage, a coverage figure — in a name that can never be changed.

**Generation lives in the data, not in a name.** `run_provenance` currently ends
`return {**existing, str(year): record}`, which **replaces** the year's record — so a
top-up would silently erase the first fill's `run_id`, `optical_skips`, `input_coverage`
and `code`, which is exactly the evidence a top-up needs to compare against. Change
`runs[year]` to a **list**, append rather than replace, and give each record a `cycle`
field. Then:

- generation of a cell = `len(runs["2021"])`
- what changed in a top-up = the last two records
- what is in a release = every cell whose runs contain that cycle

### The contract: attrs alone must answer every user question

The registry (§8) is an **efficiency layer, never a source of truth**. Anything a data user
needs must be answerable from the store itself:

| question | answered from the store alone |
|---|---|
| what rule produced this product? | root attr `optical_min_obs` |
| which cycle is this store at? | root attr `current_cycle` |
| has this cell been filled — how often, when, by which code? | `runs[year]` list |
| which tiles in this cell were fully skipped? | `runs[year][-1].optical_skips.labels` |
| how much of this cell was refused, and why? | the optical/radar summaries in the same record |
| **is this specific pixel refused?** | `s2_obs_count < optical_min_obs`, and `isnan(scales)` |

What the registry adds is speed and reach, not facts: per-shard rows, the chunk bitmask, the
depth histogram, and cross-zone queries in one file instead of 120 group reads.

**The one honest exception**, which must be stated in the README rather than glossed: a
**fully-skipped** shard writes no arrays at all, so its per-pixel depth is not recoverable
from the store. The attrs still name it (`optical_skips.labels`, which is complete), so a
user can always tell "refused" from "ocean" — the thing that record was invented for. Only
the *distribution* of how thin it was lives solely in the registry, and that is a top-up
planning field rather than a user-facing one.

### Migrating idempotence off tags — the one real hazard

The work list currently reads
`not status.has(z, y) or zone_year_tag(z, y) not in existing_tags`
(`campaign.py` L326, `run_global_campaign.py` L859). `status.has` reads `years_complete`,
which the shard writer advances **atomically with the data**, through exactly one writer that
also writes `runs`. So the attr alone is a sound "the data landed" signal and the tag half is
belt-and-braces.

**But the two halves are not synonyms**, and this is the thing to get right: `years_complete`
means *the data landed*, while the tag means *the cell was finalised* — committed, tagged, and
its validation dispatched. A crash between those two points leaves a cell that `status.has`
calls done and that never got tagged, and today's OR re-runs it. Dropping the tag check
without replacing that distinction would silently skip such a cell.

Replace it with the cycle record, not with nothing: a cell is done **for this cycle** when
`runs[year]` contains a record whose `cycle` matches the running one, and that record is
appended at the point the tag is created today. The work list keys on that. A resume then
behaves exactly as it does now, and a top-up cycle naturally re-selects every cell it wants.

### What this means for the refill tooling (it lives in `yield-embeddings`)

An earlier draft of this section called `scripts/reopen_zone_year.py` a missing tool.
**That was wrong and is retracted** — it exists, along with `drop_run_field.py`,
`validate_zone_group.py`, `campaign_health.py` and `validate_all_cells.py`, in the sibling
**`yield-embeddings`** repository, which is where the validation instrument lives.
`final-data-validation-plan.md`'s paths are relative to that repo, not this one. Do not
rewrite any of them.

What *does* change is how much work `reopen_zone_year.py` has to do. Its docstring describes
a three-step refill whose **middle step is expected to fail**: it clears one year from
`years_complete`, the refill then commits its data and raises at the tag step because the
canonical `zone-<Z>-<Y>` name is already spent, and the operator pins the new snapshot under
a fresh name with `--pin-fresh-tag`.

Under this design that awkwardness disappears at the root rather than being worked around:

- there is no per-cell tag to collide with, so **the refill no longer fails at a tag step**;
- `runs[year]` appends, so a refill no longer erases the original fill's provenance;
- idempotence keys on the **cycle**, so a top-up cycle re-selects every cell it wants without
  anyone clearing marks by hand.

The residual need is narrower: forcing **one** cell to refill outside a cycle, after a defect.
That becomes "clear this cell's record for the current cycle", or simply a `--force` flag on
the fill — no tag surgery and no expected-failure step.

### Which repo owns what

**This is a cross-repo change**: `yield-embeddings` holds the refill and validation tooling,
and it reads the campaign's tags and `years_complete` semantics — both of which move here.
The dependency is one-way and currently clean (`tessera-embeddings` imports `yield_embeddings`
**zero** times); keep it that way.

The boundary is **deployment identity**, not domain versus operations. All five scripts import
`yield_embeddings.config.buckets` — which account, which buckets, which deployment — while
their logic is this repo's data model. The rule that follows, and this repo is **public**, so
it matters:

> **This repo owns the data model and the operations on it. The private repo owns deployment
> identity.** Domain functions take a `BucketPaths`; the private side populates it. Nothing
> carrying an account, a bucket name, or a Prefect deployment name moves here.

Applied to the refill: **do not relocate `reopen_zone_year.py` — dissolve it.** Its logic is
four tessera imports deep in this repo's own campaign model; only its wiring is Arbol's. Under
this design its job becomes a `--force` / `--reopen` parameter on the fill flow, which lives
here and already receives `BucketPaths`. The private repo keeps a thin invocation wrapper or
nothing.

The four validation scripts stay where they are: they need `yield_embeddings.domain` as well
as bucket config.

### The validation modules move here — see ADR-019

Ruled 2026-08-13: `yield_embeddings.domain.embedding_validation{,_rules}` belong in this library by
the written boundary, and the move happens AFTER the campaign.
[ADR-019](../decisions/019-validation-modules-belong-in-the-library.md) carries the reasoning, the
sizing (~4,600 lines with essentially nothing to untangle), the new optional Pillow extra, what stays
behind, and the two prose statements in that validator which the current rule makes wrong.
