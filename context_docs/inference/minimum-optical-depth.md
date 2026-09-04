# The minimum optical depth line: refuse a pixel below 15 observations

**The one data-quality rule in the campaign, and the only reason a pixel is refused.** A pixel with
fewer than fifteen clear Sentinel-2 observations in the calendar year being filled is not embedded.
It is written as fill, exactly as an out-of-ROI pixel is, and the reason is recorded.

The decision is [ADR-018](../decisions/018-refuse-pixels-below-minimum-optical-depth.md). **This
document is the evidence behind it** — four measurement campaigns, taken over six days in August
2026, that between them moved the number three times. It is one document because the four are
useless separately: each answers a question the others raise, and two of them were taken
specifically to overturn a third.

**Status: BUILT.** `OPTICAL_MIN_OBS = 15`, the gate applies it per pixel per year, the store stamps
it as write-once root identity, and refused shards record why.

| § | question | instrument | when |
|---|---|---|---|
| 2 | how much of the product is thin, and where? | catalogue census, 18,819 point queries | 2026-08-12 |
| 3 | does picture quality track depth? | 713 windows, reviewed blind | 2026-08-13 |
| 4 | where does the curve actually break? | 741 matched blocks in zone overlaps | 2026-08-13 |
| 5 | what does each candidate line cost? | 82.8 M pixels counted in the store | 2026-08-14 |
| 6 | is count even the right variable? | 919 M pixels, month coverage | 2026-08-17 |

---

## 1. The rule, and how the number was chosen

**The line is 15**, decided by Robert and a colleague on **2026-08-17**. `OPTICAL_MIN_OBS` is 15 and
the campaign's seeder registers 15. **The line is STRICTLY FEWER than 15**: a pixel with exactly
fifteen valid optical observations is embedded; fourteen is refused. The gate reads
`s2_valid_count >= optical_min_obs` and the thin count reads `< thin_below`, so both agree, and a
test pins 13/14/15/16 at the campaign's own value rather than at a stand-in line.

**Coverage was chosen over reproducibility, knowingly.** The line keeps **94% of pixels against 79%
at 25** (pixel-weighted over 40 cells), and the accepted cost is that two independent embeddings of
the same ground agree less well.

**15 is not the reproducibility elbow, and that is the point of the decision rather than an
oversight.** The elbow is at 25 (§4): agreement runs three to four times the pipeline's own floor
everywhere below 25 and halves crossing into 25–30. A line at 15 therefore **admits the
worst-reproducing band in the whole measurement** — 15–19 sits at 3.83× the floor, worse even than
under-15 at 2.21×. Whoever revisits this should know the trade was made with that in front of them,
not around it.

**Two things weigh on the other side, and both post-date the elbow.** Refusal is irreversible while
filtering is not: `s2_obs_count` is published per pixel, so any user can apply a stricter line at
read time and nobody can recover a pixel we refused. And **count is not a reliable proxy for
temporal spread** (§6) — 33.7% of pixels at 15–19 observations already cover ten or more months of
the year, so some of what the elbow refuses does describe a year. Neither makes 15 *better* than 25
on the evidence; they are why a coverage-first reading is defensible.

**The history, because the number moved three times and the record matters more than the number.**
The original 30 rested on how the pictures looked; blind re-measurement (§3) showed legibility was
tracking rendered contrast rather than information, and spatial organisation is flat across 12 to 45
observations. That put 25 in as an explicit **placeholder** (2026-08-13) so the machinery could be
built while the evidence was gathered. 15 replaces it as a decision.

> **Measurements below that name the 25 line stay as they are.** They are what was measured, at the
> line in force when they were taken, and several are line-independent in the way that matters (the
> chunk-versus-pixel comparison, the count-versus-spread relationship). Where a figure is specific
> to 25 it says so.

### Three decisions taken with the rule, not open for revisiting

| decision | ruling |
|---|---|
| **2017**, where the census brackets the line between 38.7% of shard-years below 20 and 65.7% below 30 | **Same rule, populate it anyway, warn about it.** No per-year threshold — one rule keeps the product consistent. 2017 publishes roughly a third populated, and that must be stated where a user meets it, not left for them to discover |
| **Temporal consistency** — a pixel can clear 15 in 2020 and fail in 2021 | **Per pixel, per year.** Holes move between years. No all-years gate, no stability mask in this iteration |
| **Scope** | **Global campaign only.** The single-ROI path (`tessera_embeddings` flow, `plain` runner) keeps today's behaviour |

### Optical depth is the ONLY refusal rule — and radar was silently refusing land too

A DECISION (Robert, 2026-08-18), and a correction to what the campaign was actually doing. The
per-pixel gate in `inference/dataset.py` is `has_optical & deep_enough`, then `if not
allow_s2_only: &= has_radar`. `allow_s2_only` defaulted to **False** and no deployment set it, so
every pixel with zero Sentinel-1 observations was refused — and a tile with no radar coverage had
*every* pixel refused, wrote a skip marker, and published as fill.

Measured on the overnight four-cell run:

| cell | live tiles | skipped | published |
|---|---:|---:|---:|
| 40S/2023 | 58 | **43** | 59.7M px, ~24% of its land |
| 40S/2022 | 58 | **43** (the same 43) | 58.6M px |
| 02S/2023 | 76 | 34 | 173.7M px |
| 47S/2023 | 239 | 4 | 858.9M px |
| 15S/2023 | 59 | 0 | 225.2M px |

**The identical 43 tiles in two independent years is what identified the cause**: radar orbit
footprints are fixed geometry, so the same ground lacks radar every year. Cloud cover cannot
reproduce that. Two other explanations were tested and refuted first — reflectance bands missing
where the SCL is valid (0.0% of valid observations lack a red band, on surviving 16S/17S mosaics),
and a coordinate error in the comparison.

**Too much land to weed out, so the cost is accepted.** Radar-free pixels are embedded through the
upstream v1.1 missing-S1 convention (an all-zeros normalised S1 slice); their embedding quality is
unvalidated for an S1-trained checkpoint. The alternative was losing the majority of some zones.
`CAMPAIGN_ALLOW_S2_ONLY = True` is registered on the driver AND both fill deployments — a fill
dispatched by hand takes its own default, which is exactly how the overnight cells got the old
policy.

---

## 2. How much of the product is thin, and where — 2026-08-12

The optical half of the coverage picture, measured at shard resolution so the line can be argued
about with numbers. The radar half is
[`../campaign/radar-coverage-by-zone.md`](../campaign/radar-coverage-by-zone.md); the
observation-count model this refines is
[`../campaign/campaign-cost-model.md`](../campaign/campaign-cost-model.md) §6. Instrument:
[`scripts/scoping/census_s2_coverage.py`](../../scripts/scoping/census_s2_coverage.py).

### The headline

Every live shard the campaign will write, scored against four candidate lines. **A cell is a
zone-year** (1,008 of them); a cell is *majority below* when more than half its live shards are.

| | **15** | **20** | **30** | **40** |
|---|---:|---:|---:|---:|
| **cells** majority below | **5.6%** | **9.3%** | **20.5%** | **36.5%** |
| — count, of 1,008 zone-years | 56 | 94 | 207 | 368 |
| — weighted by zone footprint | 0.6% | 4.3% | 9.8% | 20.9% |
| — excluding 2017, of 896 | 3.0% | 4.8% | 12.9% | 28.9% |
| **shard-years** below | **2.95%** | **6.76%** | **18.38%** | **34.46%** |
| — count, of 3,248,577 | 95,927 | 219,453 | 597,013 | 1,119,452 |
| — chunk-years below (256 px) | 2.83% | 6.63% | 18.15% | 34.19% |
| — excluding 2017 | 0.72% | 2.76% | 12.46% | 28.51% |
| **shards** below in ALL nine years | 0.18% | 0.70% | 5.15% | 15.23% |
| — count, of 360,953 | 645 | 2,513 | 18,586 | 54,967 |
| shards below in **2017 alone** | 20.9% | 38.7% | 65.7% | 82.1% |
| shards below in **2025 alone** | 0.5% | 1.6% | 7.6% | 20.1% |

Shards and chunks agree to within 0.3 points, so the unit does not move the answer. **Shards are the
reportable unit** — the census resolves ~110 km, and a 2.56 km chunk figure would imply a precision
the instrument does not have.

**The curve is steepest exactly where the quality evidence sits.** Moving the line 15 → 20 adds 3.8
points of flagged product; 20 → 30 adds 11.6; 30 → 40 adds 16.1. So a line anywhere in the 20–30
band is the most sensitive to the census being slightly wrong, and a line at 15 is the least.

### At the 15 line this is mostly a 2017 problem

Sentinel-2B reached routine operations partway through 2017, so that year carries roughly half the
acquisitions of every year after it and behaves like a separate dataset.

| year | <15 | <20 | <30 | <40 | shards with zero obs |
|---|---:|---:|---:|---:|---:|
| **2017** | **20.85%** | **38.70%** | **65.69%** | **82.05%** | 8,468 |
| 2018 | 0.78% | 3.24% | 14.06% | 30.73% | 842 |
| 2019 | 0.59% | 2.16% | 11.67% | 26.98% | 479 |
| 2020 | 0.76% | 2.27% | 11.10% | 26.43% | 523 |
| 2021 | 0.55% | 2.50% | 12.13% | 26.02% | 493 |
| 2022 | 0.84% | 3.64% | 14.21% | 29.76% | 238 |
| 2023 | 1.23% | 4.28% | 16.79% | 38.66% | 238 |
| 2024 | 0.48% | 2.41% | 12.15% | 29.38% | 238 |
| 2025 | 0.47% | 1.61% | 7.58% | 20.12% | 238 |

**Two thirds of the headline 2.95% is 2017 alone.** The 40 line does not decompose the same way —
dropping 2017 moves it only 34.5% → 28.5% — because there the driver is ordinary cloud climatology
over the tropics and the boreal belt, not a commissioning gap.

### Where

**Persistently below 15** (all nine years): 645 shards in 25 contiguous clusters.

| place | shards | centroid | zones |
|---|---:|---|---|
| Gabon – Republic of Congo, Ogooué basin | 151 | 2.6°S 12.2°E | 32S 33S |
| N Greenland & Canadian Arctic islands † | 107 | 82–84°N 20–80°W | 17N–27N |
| Coastal Ecuador, western Andean foothills | 96 | 1.5°S 79.0°W | 17S |
| Equatorial Guinea – northern Gabon | 96 | 0.5°N 11.5°E | 32N 33N |
| **Eastern Chukotka** *(see below)* | 67 | 67°N, at 180° | 01N 60N |
| Papua New Guinea highlands | 30 | 7.0°S 143.5°E | 54S |
| Southern Cameroon | 30 | 2.5°N 12.0°E | 32N 33N |
| **Western Aleutians** *(see below)* | 20 | 52°N 179°E | 60N |

† Not one cluster but every shard above 80°N, which the clustering splits across a dozen small
ice-cap fragments; the rest of the table is single contiguous clusters.

**Persistently below 40**: 54,967 shards, concentrated in three places that are two thirds of the
total — insular and mainland Southeast Asia (16,760), the Amazon basin (13,783), and the Congo basin
with the Gulf of Guinea (7,948). The zones carrying the most are `48N` (3,195 shards, 36% of its
footprint) and `49N` (2,652, 36%), neither of which is majority-below.

### Do not read the cell figures as area

**A cell is a unit of work, so counting cells gives every zone one vote regardless of size.** Zone
footprints run from **4** live shards to **9,132**; the 43 smallest zones are 38% of the zone count
and **3.9%** of the product. Weighted by footprint, the cell figures fall from 5.6% → 0.6% and 36.5%
→ 20.9%.

The same trap bites the zone lists. Nineteen zones are majority-below-40 in every year, but fourteen
of them hold under 500 live shards and all nineteen together are 2.8% of the product. At the 15 line
the single zone that qualifies, `31S`, has **six** live shards — four of six is not a finding.

*Quote the shard figures for how much of the product is thin, and the cell figures for how many
units of work carry the label.*

### Zones 01 and 60 are not measurable this way

earth-search stores antimeridian-crossing granules as clipped fragments, so a point query in zones
`01N/01S/60N/60S` under-reports. Those four zones hold **0.46%** of live shards but **20%** of the
persistently-below-15 population — the Chukotka and Aleutian rows above. At the 40 line the
contamination is 1%.

**This may be a real thinness rather than an artifact.** The ingest path queries the same catalogue
the same way, so if the fragments defeat a point query they may also defeat an ingest query.
Settling it is cheap: compare a `grid:code` query for an affected MGRS tile against a bbox query over
the same ground.

Separately, **211 shards (0.06%) have no Sentinel-2 acquisitions at all**, 116 of them
antimeridian-suspect. Those produce no embeddings rather than thin ones, so they are a coverage
question, not a quality one.

### Method, and what it was checked against

The quantity is the one the threshold is applied to: `s2_obs_count`, the number of dates a pixel
survives the SCL mask in a calendar year. Estimated as the cost model's optical census estimates it —
per location, the sum over **distinct acquisition dates** of `1 − eo:cloud_cover` — at ~110× its
spatial resolution.

- **18,819 point queries**, one per 1° land bin, each returning the whole 2017–2025 history in a
  single response. Bins come from the land-mask registry, so small islands get their own measurement
  instead of inheriting a mainland one.
- **Same catalogue as the ingest path** (earth-search `sentinel-2-l2a`). Item geometries there are
  true granule footprints — verified: swath-edge granules carry clipped polygons — so a point query
  returns only acquisitions that cover the point.
- **360,953 live shards** enumerated with `ingest.land_mask.build_zone_coverage`, the function the
  land-mask store is built with, so these are exactly the tiles the campaign will write. 21,439,830
  live 256 px chunks, mean 59.4 of 64 per shard.
- Each shard takes the nearest census point: median **39 km**, p95 **62 km**.

**Checked against the store, not asserted.** 1,079 chunks read from the dev global store across 22
zone-years in 15 zones from 40°S to 85°N, comparing written `s2_obs_count` with the census at the
same place:

| | store / census |
|---|---:|
| median | **1.04** |
| quartiles | 0.95 – 1.19 |
| by band, −20…0 / 0…20 / 20…40 / 40…60 / 60…85 | 1.23 / 1.04 / 1.03 / 1.07 / 1.01 |

So the census sits within a few percent of what the pipeline writes, and runs slightly **low** —
which makes every percentage above a mild over-estimate rather than an under-estimate.

The agreement is not guaranteed by construction and the direction was not predictable:
`eo:cloud_cover` is a granule average that cannot see cloud clustering; the mask drops SCL 2 and 3
(dark area, cloud shadow) which `eo:cloud_cover` does not count, pushing the store down; and it keeps
SCL 10 and 11 (thin cirrus, snow) which `eo:cloud_cover` does count as cloudy for cirrus, pushing the
store up. The measurement is what settles it.

> **Noted in passing, not fixed here:** `config/satellites.py` describes
> `S2_SCL_INVALID_CLASSES = {0, 1, 2, 3, 8, 9}` as "(nodata, saturated, cloud shadow, cloud,
> snow/ice)", but class 11 (snow) is **not** in the set and is therefore kept. The comment is wrong
> about snow; the code is what the validation above measured.

### What the census does not settle

- **The measurement resolves ~110 km, not 20 km.** Shard-level figures inherit a 1° census value and
  are not independent per shard. The aggregates are sound; a single named shard is not.
- **The orbit sidelap is aliased.** Revisit roughly doubles in the ~50 km overlap strips, visible as
  diagonal banding across the boreal belt at the 40 line. A 1° census samples that coarsely, so
  cluster *edges* there are approximate.
- **Within-shard spread is unmeasured.** The validation bounds the aggregate bias, not the
  distribution of pixels inside a shard around its granule-mean estimate.
- **This is optical only.** Radar absence alone does not visibly degrade the embeddings.

**Nothing here is fixable by re-running:** the observations do not exist, and the per-pixel counts
are already written beside every embedding. The decision is about labelling. 15 sits deep in the left
tail, where moving it changes almost nothing; 40 sits just below the mode, where the distribution is
steepest and a line is least stable. That is the quantitative form of *"a label that covers a third
of the product is not a label"* — and it measures the claim, which had not been: at 40 it is 34.5% of
shard-years, 28.5% excluding 2017.

### Pixels versus shards: the census carries over

The census measured **shard means**, and the rule refuses **individual pixels** — a different
question, and the class of error the corrections register calls *presence counted where coverage was
meant*. It was therefore measured directly against `s2_obs_count` as written, over **844 chunks and
55.3M populated pixels in 15 zones**:

| line | % of PIXELS below | % of CHUNKS whose mean is below | ratio |
|---|---:|---:|---:|
| 15 | 2.27% | 2.25% | 1.01 |
| 20 | 3.73% | 3.79% | 0.98 |
| **30** | **10.90%** | **11.37%** | **0.96** |
| 40 | 21.66% | 21.45% | 1.01 |

The two views agree to within 4%, so the census's shard-level figures carry over. (The absolute level
in that sample is below the census's global 18.4% because the sampled zones are dominated by dry
37N/38N; the **ratio** is the transferable result, not the level.)

The loss is also concentrated rather than smeared — at the 30 line, **90.4% of refused pixels sit in
chunks whose own mean is already below 30**, and chunks averaging 45+ contribute 0.1%. This is what
makes a per-shard registry an honest summary: refusals cluster, so a shard-level number is not
hiding a diffuse scatter.

---

## 3. Does picture quality track depth? A blind re-measurement — 2026-08-13

**Why this exists.** The campaign's records had carried one sentence about embedding quality since
July: *crisp at 30 or more valid observations per pixel, noisy below about 20*, measured across 15
cells. The proposal was to make that sentence into a **refusal** at 30. That decision is irreversible
— a refused pixel does not exist, so recovery is a re-run — and on 2026-08-13 two defects were found
in the instrument that produced the original sentence.

**The renderer manufactured noise.** The window figures were drawn with a **per-window** contrast
stretch — each window scaled to its own 2nd–98th percentile — so a window whose ground varies little
had its noise floor expanded to full contrast *whatever its depth*. Some of the "noise below 20" was
the picture, not the data. The stretch is now shared across a cell's windows.

**Depth was read per cell.** A zone spans most of a hemisphere and depth falls with latitude and with
cloud, so a cell mean describes a mixed population. The first cell re-measured has a mean of 46.9 and
windows at 34.5 and 61.0 — and its *thinnest* window is the one with the most structure. Each window
now records its own mean and tenth percentile.

### Method

**Two passes.** A pilot of 73 windows from 9 cells, sampled by latitude, and then the pass reported
here: **713 windows from 40 cells**, sampled by depth so the bands that decide a cutoff are populated
on purpose rather than by accident. The pilot is why the second pass exists — it had one window
separating a cutoff of 20 from one of 30.

**Blinded in code, not by instruction** (`yield-embeddings/scripts/depth_legibility_probe.py` in `yield-embeddings`).
The reviewer received opaque filenames in shuffled order and one fact per window — how much of the
frame the land mask calls land, without which an ocean window cannot be told from a failed land one.
**Depth was withheld**, and so was the flatness figure: it is computed from the same pixels, so a
reviewer handed it could answer "spread 0.12, therefore illegible" without looking, and the result
would be a correlation between two derived numbers.

**One question per window:** can you name a real ground feature — field boundaries, drainage, valley,
coastline, settlement, dune or glacial forms, a boundary between distinct covers? Answer `legible`,
`uncertain` or `illegible`, with `illegible` explicitly not a criticism, since open ocean, unbroken
desert and unbroken canopy are correctly illegible. Every verdict carries a one-line note naming what
was seen, so a verdict can be audited rather than trusted.

**The six agents agree on level**, which is what makes the pooling legitimate: on fully-land windows
they returned 69%, 74%, 75%, 78%, 79% and 81% legible against matched median depths of 25 to 28
observations. Two of them reached that agreement by different routes — one upgraded verdicts after a
per-image restretch revealed organised structure, another refused to upgrade on anything not visible
at the delivered contrast. The seven-point spread is the size of that disagreement.

### The result — 671 fully-land windows

| depth band | n | % legible |
|---|---:|---:|
| under 12 | 15 | 60% |
| 12–15 | 39 | **90%** |
| 15–18 | 59 | **92%** |
| 18–21 | 69 | 86% |
| 21–24 | 71 | 86% |
| 24–27 | 72 | 86% |
| 27–30 | 79 | 77% |
| 30–35 | 111 | **66%** |
| 35–45 | 140 | **61%** |
| 45+ | 16 | 69% |

**Legibility FALLS as optical depth rises**, from 92% in the 15–18 band to 61% at 35–45. The July
claim is not merely unsupported; the gradient in this corpus runs the other way. The one band where
refusal looks defensible is the very bottom — **under 12 observations, legibility is 60%**, the
lowest of any band and the only one where a cutoff removes roughly as much bad as good.

**Why it runs backwards, tested rather than asserted.** Not latitude: legibility by absolute latitude
is flat within noise (76%, 78%, 67%, 84%, 72% across five bands). The controlled test is depth
**within** a latitude band, and it splits:

| | under 20 obs | 20–30 | 30–45 |
|---|---:|---:|---:|
| tropics, 0–25° (n=401) | **87%** | 85% | **61%** |
| mid-latitudes, 25–55° (n=112) | 77% | 82% | 82% |
| high latitudes, 55–90° (n=43) | 75% | 67% | 67% |

So the effect is real inside the tropics and absent elsewhere, which points at **land cover
correlated with cloudiness** rather than at depth doing anything. In the tropics the wettest places
are the most persistently clouded — and they are river networks, forest-and-clearing mosaics and
smallholder agriculture, which are legible. The drier tropics are clear overhead and uniform on the
ground. **Depth is a proxy for climate, and climate is a proxy for how much there is to see.**

The per-cell spread says the same thing more bluntly. Legibility by cell runs from **11%** (23N/2021,
Saharan desert, deep and uniform) to **100%** (38N/2021, 47S/2021, 59S/2022). Which zone a window came
from predicts its legibility far better than how many observations it had.

**Two numbers for whoever picks a line.** At every cutoff from 15 to 30, about **85% of what it
refuses is legible**; only at 12 does that drop to 60%. And the illegible share of what a cutoff
KEEPS rises monotonically, 18% at 12 to **30% at 30** — refusing more makes the surviving product
proportionally worse, because the unreadable windows are mostly the deep ones.

### Why the legibility route was then abandoned

Six sections of the original report priced candidate cutoffs by legibility, and its own §7 concluded
that legibility was the wrong measure. They are cut rather than kept, because the line was decided
against them twice over:

**The limit that outranks all of it: legibility is not utility.** The picture is a three-component
projection of 128 dimensions, and one reviewer measured seven of its illegible windows as carrying
coherent structure at a tenth of the amplitude the stretch renders visible. The rule's argument rests
on usefulness, which no measurement has tested and — with no downstream model in existence — none
can.

**What does depend on observation count is a measurable bias**, not legibility: the straight-edge
artefacts visible in some windows are upstream, present in the source imagery rather than introduced
by assembly, which was checked rather than assumed.

**The reason this matters beyond this section:** the original 30 line rested on how pictures looked,
and the blind re-measurement is what dislodged it. That is why §4 abandons pictures entirely and
measures agreement between two independent embeddings of the same ground. Full working in git
history.

---

## 4. Where the curve breaks: reproducibility from zone overlaps — 2026-08-13

Adjacent UTM zones overlap by about 200 km and each embeds that ground independently: its own mosaic,
its own ingest, its own per-pixel observation count. So the overlap gives the same ground twice, and
— this is the part every earlier attempt lacked — **blocks where the two runs saw the same number of
observations measure the pipeline's own reproducibility**, which is the yardstick everything else has
to be judged against.

### The first sweep established a direction and could not locate anything

81 blocks across four adjacent pairs (`yield-embeddings/scripts/zone_overlap_drift.py`). **The noise floor: 0.00050** —
two completely independent embeddings of the same ground, at matched depth, differ by that much in
cosine distance (n=59, 90th percentile 0.00232).

Depth *of difference* moves the embedding, and two independent methods agree on the slope:

| observation difference | n | median distance | against the floor |
|---|---:|---:|---:|
| 0–1 | 59 | 0.00050 | 1.0× |
| 1–3 | 18 | 0.00140 | **2.8×** |
| 3–6 | 4 | 0.00259 | **5.2×** |

That is 0.00053 of excess distance per observation of difference. The seam measurement, which uses
Sentinel-2 swath boundaries and shares no code or data with this one, gives **0.00060**.

And depth ITSELF predicts reproducibility, measured on matched-depth blocks only, so this is not the
difference effect: under 20 observations runs 3.3× the floor, 20–25 at 2.7×, 25–30 at 3.2×, 30–40 at
0.4× and 40+ at 0.6×. **Thin ground is about five times less reproducible than deep ground**, and the
effect appears in all four pairs.

> **This was the first evidence in the whole investigation that supports the rule's premise**, and it
> arrives by a completely different route from legibility: not "thin pictures look worse" but "thin
> embeddings are not reproducible". What it could not do is choose a number — twelve blocks below 30
> observations, four per bin, is enough to establish a direction and nowhere near enough to locate
> where the curve breaks.

### 741 matched blocks: the elbow is at 25

The second sweep stratifies by depth — a block's depth is screened from the count array before either
embedding is read, and sixteen blocks are taken from each qualifying shard — so the budget goes where
the question is. **813 blocks across five adjacent zone pairs, 741 of them at matched depth.**

**The floor is 0.00040**, the median distance between two independent embeddings of the same ground
where both runs saw forty or more observations. That is as well as this pipeline ever agrees with
itself, and every figure below is against it.

| shallower side | n | median distance | against the floor |
|---|---:|---:|---:|
| under 15 | 36 | 0.00088 | 2.21× |
| 15–20 | 76 | 0.00152 | **3.83×** |
| 20–25 | 114 | 0.00135 | **3.40×** |
| **25–30** | 114 | 0.00077 | **1.93×** |
| 30–35 | 159 | 0.00057 | 1.43× |
| 35–40 | 22 | 0.00048 | 1.22× |
| 40+ | 220 | 0.00040 | 1.00× |

**The elbow is at 25.** Reproducibility runs between three and four times the floor everywhere below
25 observations, then **halves** crossing into the 25–30 band, and closes on the floor slowly after
that. The single largest improvement available from any cutoff in this range is the one at 25.

Read as a decision, cutting at each candidate:

| cutoff | blocks refused | refused side, against floor | kept side, against floor |
|---|---:|---:|---:|
| 20 | 112 | 3.11× | 1.64× |
| **25** | 226 | **3.15×** | **1.33×** |
| 30 | 340 | 2.94× | 1.22× |
| 35 | 499 | 2.36× | 1.06× |

**Choosing 30 over 25 buys little and costs roughly double.** The kept side improves from 1.33× the
floor to 1.22× while the refused population grows by half again — and by 35 the cutoff is eating data
that is already near the floor, which is what the falling refused-side column shows.

> **The line shipped is 15, not this elbow.** Coverage was chosen over reproducibility deliberately:
> 15 retains 94% of pixels against 79% at 25. A line at 15 admits the **worst-reproducing band
> measured here** — 15–19 at 3.83× the floor, worse than under-15 at 2.21× — and the decision was
> taken with that in view. **This section is not withdrawn:** it remains the best available account
> of how agreement varies with depth, and it is the reason the trade has a price at all.

**Three things this does not say.**

The bottom bin is not the worst: under 15 observations reproducibility is *better* than at 15–20
(2.21× against 3.83×). One coherent explanation is that agreement needs either the same few inputs or
enough inputs that the sample stops mattering, and the middle is where two runs most easily see
different halves of a moderate set. It is an observation from 36 blocks, not a finding.

**The effect is not universal.** Thin against deep within each pair: 3.0×, 2.8×, 2.1× — and then 1.1×
and 1.1× for the two pairs that both involve zone 48S, which are poor at *every* depth (about 0.0016
either side). Whatever is wrong there is not depth, and it is worth a look on its own.

And this is reproducibility, not accuracy. Two runs agreeing tells you the pipeline is stable at that
depth; it does not tell you the embedding is right. That question still has no measurement, and with
no downstream model in existence it cannot get one.

---

## 5. What each candidate line costs, counted per pixel — 2026-08-14

The census (§2) measures the *input* at ~110 km from the catalogue; the legibility and reproducibility
work (§3, §4) measures whether a window can be read and whether two runs agree. **Neither answers
"how much of a cell disappears"**, and that answer decides whether a published cell is worth
publishing.

At the 25 line, **the median measured cell keeps 92% of the pixels a rule-free fill would have
published**, five of forty keep under half, and two keep under a tenth.

### The instrument

Every filled cell already carries a per-pixel `s2_obs_count` array, so the retained share is a direct
count. The denominator is **pixels with any optical observation at all**, because those are the ones a
fill without the rule would embed. Retention therefore reads as *of what we would have published, how
much do we keep*.

Sampled, not exhaustive — a zone is roughly 891,000 × 68,000 px at 10 m. One 256-px chunk per sampled
shard, aimed at land by the same `window_origin` the validator uses, spread evenly over the written
shards with `np.linspace`. Forty chunks per cell, 82.8 M pixels over **40 cells in 26 zones**, read
from the dev global store.

Two corrections are part of the result:

- **The first version sampled `range(0, n, step)`, which degenerates to "the first N shards"** whenever
  the step rounds to 1. Shard order is row-major, so on a zone whose depth tracks latitude that samples
  one end of a gradient. It read 1.4% retention for 26S/2021 where the full sample reads 8.5% — a
  factor of six, in a table where every row looked plausible. Caught only by checking it against a
  figure the cell's own published validation already implied.
- **Window-mean depth, used as a proxy, is withdrawn.** A window whose mean falls below the line was
  called gutted, which conflates losing 40% of its pixels with losing 99%. It overstated the cutoff's
  cost by roughly four times, and in the other direction it called 58S/2021 thin at a mean of 25.2
  where the exact median pixel has 43 observations.

The corrected sampler was then checked against figures produced by a different code path entirely: the
mean of pixels at or above 25 in the pre-rule store reproduces the newly filled cells' independently
reported means — 29.58 against 29.75 for 26S/2021, 36.28 against 36.37 for 47S/2021. Two agreements
within 0.2 observations.

### The cost of each candidate line

| line | median cell retains | pixel-weighted mean | cells under 50% | cells under 10% | worst cell |
|---|---:|---:|---:|---:|---:|
| **15** | — | **94.1%** | **0** | **0** | — |
| 20 | 96.6% | 87.7% | 3 | 0 | 26.9% |
| **25** | **91.9%** | **79.2%** | **5** | **2** | **2.3%** |
| 30 | 86.0% | 70.9% | 10 | 4 | 0.0% |

**This is the table that chose 15.** It retains 94.1% of pixels pixel-weighted against 79.2% at 25,
and leaves **no** cell under half or under a tenth. Moving 25 → 30 costs about six points of median
retention and **doubles the number of cells that lose more than half their pixels**, from five to ten.
Moving 25 → 20 recovers five points but leaves in the band where two independent embeddings of the
same ground disagree about twice as much as the pipeline's own floor — a trade §4 prices.

### Per cell, at the 25 line

Sorted by what survives. `median obs` is the median observation count over the same sample, so a
cell's depth and its loss can be read together.

| cell | px sampled | median obs | keeps at 20 | **keeps at 25** | keeps at 30 |
|---|---:|---:|---:|---:|---:|
| 29S/2021 | 393,216 | 20 | 0.648 | **0.023** | 0.000 |
| 26S/2021 | 1,638,400 | 18 | 0.388 | **0.085** | 0.039 |
| 32S/2022 | 2,621,440 | 15 | 0.269 | **0.133** | 0.073 |
| 26S/2022 | 1,769,056 | 20 | 0.561 | **0.233** | 0.096 |
| 32S/2021 | 2,621,432 | 16 | 0.416 | **0.291** | 0.152 |
| 47S/2020 | 2,621,440 | 26 | 0.696 | 0.555 | 0.408 |
| 17S/2022 | 2,621,440 | 28 | 0.655 | 0.555 | 0.479 |
| 17S/2021 | 2,615,052 | 30 | 0.754 | 0.605 | 0.521 |
| 47S/2022 | 2,621,440 | 28 | 0.861 | 0.655 | 0.405 |
| 48S/2021 | 2,621,440 | 28 | 0.961 | 0.719 | 0.410 |
| 47S/2021 | 2,621,440 | 31 | 0.859 | 0.727 | 0.549 |
| 49S/2021 | 2,621,440 | 30 | 0.928 | 0.736 | 0.521 |
| 47S/2025 | 2,621,440 | 33 | 0.902 | 0.754 | 0.600 |
| 39S/2021 | 2,621,440 | 37 | 0.959 | 0.826 | 0.724 |
| 58S/2022 | 2,621,440 | 39 | 0.906 | 0.845 | 0.776 |
| 57S/2022 | 2,621,440 | 41 | 0.960 | 0.860 | 0.797 |
| 16S/2022 | 1,114,112 | 33 | 0.934 | 0.902 | 0.708 |
| 15S/2024 | 2,621,348 | 43 | 0.923 | 0.907 | 0.855 |
| 58S/2025 | 2,621,440 | 40 | 0.950 | 0.909 | 0.817 |
| 40S/2024 | 983,000 | 49 | 0.936 | 0.919 | 0.893 |
| 30S/2023 | 262,144 | 29 | 0.999 | 0.920 | 0.431 |
| 59S/2022 | 2,621,440 | 42 | 0.983 | 0.937 | 0.866 |
| 02N/2022 | 2,621,440 | 52 | 0.993 | 0.953 | 0.920 |
| 02N/2021 | 2,621,440 | 55 | 0.972 | 0.960 | 0.953 |
| 53N/2021 | 2,621,440 | 63 | 0.988 | 0.961 | 0.914 |
| 41S/2023 | 393,216 | 44 | 0.970 | 0.962 | 0.951 |
| 57S/2021 | 2,621,440 | 40 | 0.983 | 0.963 | 0.924 |
| 41S/2022 | 393,216 | 51 | 0.997 | 0.965 | 0.962 |
| 59S/2021 | 2,621,440 | 41 | 0.984 | 0.966 | 0.915 |
| 58S/2021 | 2,621,440 | 43 | 0.989 | 0.967 | 0.919 |
| 12S/2021 | 524,288 | 48 | 0.994 | 0.984 | 0.950 |
| 16S/2021 | 1,114,112 | 38 | 1.000 | 0.987 | 0.923 |
| 37N/2021 | 2,621,440 | 58 | 0.999 | 0.991 | 0.979 |
| 03N/2021 | 2,621,440 | 62 | 0.999 | 0.994 | 0.981 |
| 23N/2021 | 2,555,904 | 70 | 0.999 | 0.998 | 0.997 |
| 38N/2021 | 2,621,440 | 67 | 1.000 | 1.000 | 0.999 |
| 06N/2021 | 2,621,440 | 66 | 1.000 | 1.000 | 1.000 |
| 09S/2021 | 655,360 | 49 | 1.000 | 1.000 | 1.000 |
| 30S/2022 | 262,144 | 37 | 1.000 | 1.000 | 0.995 |
| 60N/2020 | 2,621,440 | 71 | 1.000 | 1.000 | 1.000 |

The loss is concentrated, not smeared. Twenty-six of forty cells keep 90% or more at the 25 line; the
entire cost sits in five cells of three zones — 26S, 29S and 32S — plus a middle group around 47S,
17S, 48S and 49S that loses a quarter to a half.

**What these figures are not.** Not a campaign-wide estimate: the store's zones were chosen
purposively for the depth study and lean thin, and the pixel-weighted refusal here at the 30 line is
29.1% against 18.4% of shard-years for the same line in the global census — so this sample is roughly
half again as thin as the world, and campaign-wide loss will be lower than every figure above. Not
exact per cell: forty chunks resolve a cell to a few points when its depth is uniform and rather less
when it is not. And not a statement about usefulness — this counts pixels.

### The operational consequence, seen in the rehearsal

26S/2021 was filled and published at 8.5% retention, and its per-cell validation passed with no
blocking finding. Tile-level coverage read 25 of 27 live tiles written — **a cell can look 93% covered
and hold a twelfth of its pixels.** Two of its seam checks came back UNAVAILABLE precisely because so
little survived that too few boundaries could be compared.

So a cell this thin is not caught by the gate, and was never meant to be: refusal is the rule working.
What the gate does not do is tell an operator that the cell is nearly empty. The verdict now carries
`coverage.embedded_fraction_of_land` for that purpose — but read it beside `written`/`live`, since it
is computed over windows drawn from **written** tiles and so cannot see a tile that was skipped whole.

---

## 6. Count is not a reliable proxy for temporal spread — 2026-08-17

A year-long embedding is supposed to describe a year, so what plausibly matters is whether the year is
*covered*, not how many observations there are. Count and spread are correlated by construction — 60
observations must span the year, 6 cannot — so the question is whether the correlation is tight enough
that gating on count already gates on spread.

Measured off the zone mosaics' SCL, which is the same input the gate sees: **24 cells, three 4096-px
tiles each, 919,224,646 pixels.**

| count | pixels | mean months covered | ≥10 months | all 12 | ≥2-month gap | ≥3-month gap |
|---|---:|---:|---:|---:|---:|---:|
| 1–9 | 16,151,138 | 5.1 | 0.0% | 0.0% | 98.9% | 72.0% |
| 10–14 | 21,392,541 | 7.7 | 6.8% | 0.0% | 76.2% | 35.8% |
| 15–19 | 49,196,955 | 9.0 | 33.7% | 1.3% | 54.9% | 15.2% |
| **20–24** | 55,488,166 | 9.9 | **64.3%** | 7.8% | 38.2% | **15.1%** |
| 25–29 | 59,745,522 | 10.7 | 85.8% | 29.3% | 19.9% | 10.2% |
| 30–39 | 207,201,979 | 11.5 | 98.5% | 62.1% | 4.2% | 0.7% |
| **40+** | 510,048,345 | **11.1** | 81.2% | 65.8% | **19.0%** | 5.0% |

**Two errors the rule cannot see, and both are large.** Of the pixels at 20–24 observations that a
line at 25 refuses, **64.3% already cover ten or more months** — the refusal discards data that
describes the year. And **the 40+ band is worse-distributed than the 30–39 band** (11.1 months against
11.5, and 19.0% with a two-month blind gap against 4.2%), so the line admits, unexamined, deep pixels
that are blind for a season.

**The 40+ inversion is real but concentrated in a third of cells**, not an artefact of one: 03N/2021
44.4%, 06N/2021 56.0%, 23N/2021 60.5%, 37N/2021 25.7%, 38N/2021 22.2%, 53N/2021 33.7%, 60N/2020 36.5%
have a two-month gap on a large share of their deepest pixels. The other sixteen cells read
**0.0–4.6%**. The pattern is a strong clear season: many observations, all in one window.

**And whether the refused near-miss population is well distributed is a per-cell property.** Among
cells with over a million pixels at 20–24, the share covering ten or more months runs from **38.2%
(58S/2022) to 91.1% (58S/2025)**. One global count line cannot express that; a spread rule could.

**What this does NOT show, and it is the load-bearing gap:** that better-spread pixels produce better
embeddings. It establishes only that spread is an axis *independent of* count — necessary for a spread
rule to be worth anything, nowhere near sufficient. The experiment that would settle it is §4's
zone-overlap measurement re-stratified by spread **at fixed count**, and it is runnable: 57S/2022–58S/2022
and 58S/2022–59S/2022 are adjacent pairs whose mosaics are both still readable.

> **This also qualifies §4's elbow.** The bands there differ in *how much of the year they see* as
> well as in how many looks they got, so the count elbow may be a spread elbow reported in count's
> units — which is also the shape that would explain why the bottom bin reproduces better than 15–20.
> **Nothing in §4 is withdrawn:** the count elbow is a real, measured feature of the data and remains
> the best available basis for a count rule. What is not established is that count is the right
> *variable*.

**Limits.** Three tiles per cell, so three locations — within-cell variation between tiles is not
captured. Only **24 of 40** complete cells have a readable mosaic at all; the rest kept orphaned chunk
objects after their Icechunk metadata was deleted, and their SCL is unrecoverable, so the measurable
subset is whatever survived cleanup rather than a random sample. "Months covered" is a crude spread
statistic — twelve observations one per month scores the same as twelve in a fortnight plus scattered
singletons.

**Spread is retained as twelve labelled month planes, and they are BUILT** (`s2_month_covered`). A
packed uint16 bitmask was rejected: the only argument for it was size, and measured that is 14% rather
than the 6× the proposal assumed, because embeddings barely compress and the planes are a rounding
error beside them. **Cost to the finished store: 0.025%, about 353 GiB** — cross-checked against the
cost model's independent 0.9–1.8 PB estimate. Labelled planes also mean `cov.sel(month=7)` says July
to a reader who has never seen this document, which a bitmask cannot.

---

## 7. The enforcement unit, and the record

### The unit was never the open question — the RECORD was

Applying the rule at a coarser unit than the pixel changes retention by **about one point at every
unit tested**, because retention is *bimodal* at chunk scale: a chunk is nearly all-in or nearly
all-out, so where the boundary is drawn barely matters. And the hole inside a nearly-full chunk is
**entirely a depth refusal** — no part of it is missing imagery — with the refused population
concentrated just under the line.

Measured store-wide over every cell the store calls complete, ten whole shards each: **40 of 48 cells,
940,376,064 land-live pixels.** (The eight that contributed nothing had all ten sampled shards at the
edge of the land-mask grid, where the shard's 8×8 slice is incomplete; they are small zones, so the
aggregate under-weights those.) Both candidate lines come from **one pass of reads**, so the two
columns describe identical pixels and the only difference between them is the line:

| | line 20 | share | **line 25** | share |
|---|---:|---:|---:|---:|
| embedded under the rule | 795,839,925 | 84.63% | 708,200,433 | **75.31%** |
| thin, inside weak chunks (<50% full) | 119,958,492 | 12.76% | 209,302,978 | 22.26% |
| thin, inside strong chunks (≥50% full) | 22,950,927 | **2.44%** | 21,245,933 | **2.26%** |
| refused in total | 142,909,419 | 15.20% | 230,548,911 | 24.52% |
| fill inside strong chunks | 97.1% | | 97.0% | |
| fill inside weak chunks | 13.0% | | 8.7% | |
| strong chunks' share of refused pixels | 16.1% | | 9.2% | |

Pixels with no optical observation at all: **0.17%** of land-live, at either line.

> **Correction.** An earlier version gave embedded-under-the-line as 709,827,153 pixels. That was
> `land − thin`, which credits the 1.6M pixels with no optical observation as embedded; they are
> neither embedded nor refused by the depth rule. The count is **708,200,433**. The percentage was
> unaffected at one decimal.

**Refusals cluster, they do not smear.** 90.4% of refused pixels sit in chunks already below the line,
so a later top-up can select whole shards by their own mean rather than hunting scattered pixels. That
is what makes a top-up affordable at all (ADR-018).

**The permanently-unrecoverable population is nearly invariant to the line: 2.44% at 20 against 2.26%
at 25.** It rises slightly as the line *falls*, because lowering the line promotes chunks into the
strong class and each newly-promoted chunk brings its own sub-line pixels with it. So a
wave-through decision is orthogonal to where the line sits.

**Cross-check:** retention against the *any-optical* denominator reads 75.4% here against §5's 79.2%
pixel-weighted — 3.8 points apart on entirely different sampling (ten whole shards versus forty
land-aimed inner chunks). Close enough to trust both; the gap is spatial spread.

### Waving through thin pixels inside strong chunks — three things it does not settle

At the 25 line, **90.8% of refused pixels are in weak chunks**, so a proposal to wave through the thin
pixels inside strong chunks rescues a tenth of the refused population and defers the other nine tenths
to a repair flow that does not exist. Its reach per cell runs from 0.00% through a median of 2.30% to
**7.28%**. At the 20 line the same proposal covers 16.1% of a smaller refused population.

1. **It trades a visible hole for an invisible one.** Today every published pixel in a strong chunk
   meets the line. Waving through makes such a chunk read 100% full with 4.2% of it below the line,
   detectable only by reading `s2_obs_count`. This is exactly what the blocking check in
   `embedding_validation_rules.py` was written to prevent, in its own words: *"not low quality, it is
   mislabelled — and unlike a thin cell, no picture of the data reveals it."* If the proposal ships,
   `optical_min_obs: 25` **must not** — the root attribute has to become `null` plus a separate field
   naming the chunk-level rule, or the store advertises a guarantee it does not keep.
2. **Permanence is a property of a repair flow that does not exist.** Until one does, every refusal is
   permanent, not 5% of them, and the argument taken literally says refuse nothing. Embedding every
   eligible pixel would cost under 8% more inference against ~1% for the targeted version, so cost is
   not what rules it out — a quality judgement about publishing near-empty chunks is.
3. **The benefit is a function of a threshold not yet chosen.** At a repair cutoff of 90% the rescued
   population shrinks; at 20% it grows. Choosing store contents now on a repair threshold chosen later
   is the weakest link in the argument.

### A refused shard records WHY

Three defects made a per-shard registry necessary rather than nice: a fully refused shard used to
discard its counters, so a thin-depth refusal was indistinguishable from no optical coverage at all;
"no records" and "every read failed" were the same empty dict; and nothing checked that the reasons
partition the refused set. All three are closed and pinned.

**The marker's PRESENCE is load-bearing**, which is the part a reporting-shaped fix would have missed:
absent markers and absent refusals look identical downstream, so `unreadable_markers` is a separate
signal from "nothing was refused". Only FULLY refused shards need a record; for a shard that was
written, the per-pixel `s2_obs_count` arrays in the store are already the evidence. The markers are
read at ASSEMBLY, which is the last moment they exist — they live with the staging prefix and go when
it does.

### The month array shipped wrong once, and how it failed is worth more than the fix

The first cell published twelve all-empty planes while every array beside them was correct. The cause
was ENUMERATION rather than the array: assembly chose which staged variables to copy by a whitelist the
new variable was not on, so it staged fine, was never copied, and the destination kept its fill value —
which for a bool plane reads as "no pixel had coverage" rather than as "nothing was written". Fixed in
`ac932c6`, verified in `fc16f57` on 40S/2022 and 28S/2025, and reproduced against the published store
with the reader expressions run verbatim rather than reasoned about.

**Three things about the failure generalise.**

No test caught it because every assembly test hand-rolled its destination group, so none exercised the
enumeration that decides what gets copied. **A test that builds its own destination cannot catch a
destination-selection bug.**

**A well-observed window cannot verify this array.** A dense window scores full marks whatever the code
does; only thin windows make "100% covered" mean something, which is why verification used 28S's
thinnest window and checked that months 10 and 11 are absent there exactly as the mosaic says.

**The array is `int8` on disk with the attribute `dtype="bool"`**, because the write path cannot emit
bool directly — and a test staging with raw zarr keeps bool, so it cannot see the production
representation at all.

Two operational notes for anyone verifying a future cell: the mosaic is the only external check and is
deleted once a cell lands, so the snapshot has to be taken between ingest finishing and the fill
completing — a fill cancelled *before* its cell lands keeps the mosaic. And a single-timestep probe for
"is this window populated" wrongly concluded 28S had no populated window at all, because a
partial-swath date reads empty over good land. Probe several. 09S/2022 keeps its empty planes: the
store is write-once, so a wrong array is superseded by a fresh store rather than repaired.

---

## 8. Non-goals — do not build these

**No ingest-time skipping.** The only signal available before reading a byte is the catalogue's
**acquisition-date count**, which upper-bounds the observation count and is therefore a sound,
zero-false-positive skip test. It was measured and it does not pay:

| year | provably skippable pre-ingest | actually refused after masking |
|---|---:|---:|
| **2017** | **37.15%** | 65.69% |
| 2018 | 0.34% | 14.06% |
| 2019–2024 | 0.07–0.23% | 11.1–16.8% |
| 2025 | 0.07% | 7.58% |

Overall the bound catches **23% of refusals, and almost all of it is 2017**. Three further reasons it
is the wrong trade: the ingest unit is a **4096-px chunk**, so an entire 40 km block would have to fail
rather than a location; skipping ingest destroys the audit record, because `s2_obs_count` is derived
from the mask that was not built; and the mosaic is shared with the radar legs. Changing the most
fragile path in the system for a fifth of a percent outside 2017 is not worth it.

**Also out of scope:** any change to the land mask; any change to the single-ROI path; re-running dev
cells already filled under the old rule; a per-pixel "years cleared" stability mask (considered,
deferred).

---

## 9. Where the rule lives, and what it rests on

Eight work-item sections once specified the machinery here. **All of it is built**, and the code is a
better record than a plan is:

| what it specified | where it lives now |
|---|---|
| the threshold, recorded once | `config/inference.py` (`OPTICAL_MIN_OBS = 15`), stamped into the store root as write-once identity by `storage/global_store.py` |
| the gate | `inference/dataset.py`, applied per pixel per year |
| the store's rule is the only rule a fill may apply | `orchestration/runners/zone_fill.py` asserts it; the Prefect adapter substitutes it |
| refusal recorded per shard, with a reason | `inference/actors.py` writes the marker, `inference/assembly.py` reads it at assembly |
| the registry beside the store | `config/paths.py` (`optical_registry()`), a sibling of the Icechunk prefix rather than inside it |
| the tests | `tests/unit/assembly/test_skip_registry.py`, `test_dataset_v11.py`, `test_assembly.py` |

What already existed and had to be built on rather than reinvented:

| thing | where | note |
|---|---|---|
| per-pixel S2 depth | `s2_obs_count` in every zone group | **written for every pixel from the mask bundle, independent of the gate** — so a refused pixel's depth is still recorded |
| the embedded mask | `scales`, NaN except where embedded | `~isnan(scales)` IS the per-pixel embedded mask |
| per-chunk counters | `actors.py`, emitted on `CHUNK_SUMMARY` | `s1_free_px` / `s1_thin_px` / `s2_thin_px` |
| per-year roll-ups | `assembly.summarise_radar_coverage`, `assembly.summarise_optical_skips` | already answer "which live tiles published as fill" |
| the year record | `storage/shard_writer.run_provenance` | the schema's one owner |
| the resume trap | documented in `summarise_optical_skips` | a resumed leg reports synthetic successes **with no counters** |

**One inference chunk is one shard** (`SHARD_PX = 2048` for both), so per-chunk accounting in the actor
*is* per-shard accounting. Nothing needs to be re-read to build the registry.

Three properties that the code states as mechanism and not as reason:

**The registry is a SIBLING of the store, never inside it.** Icechunk owns every key under its own
prefix — garbage collection enumerates that prefix and reconciles it against its own manifests — so a
Parquet file living there is at best unrecognised and at worst collected. Its schema and the defects
its first live run exposed are in
[`../storage/writing-to-the-global-store.md`](../storage/writing-to-the-global-store.md) §6.

**The rule is part of the store's write-once root identity.** Not a per-run parameter, because a cell
filled under a different line than its neighbours is undetectable afterwards: a refused pixel is
indistinguishable from one that had no optical input. **Moving the line therefore means a new store,
not a migration** — which is the cost the decision in §1 was taken with in view.

**Three categories, not two.** Not-eligible (ocean or outside the ROI), eligible-and-embedded, and
eligible-and-refused. A percentage over the wrong denominator was the error this replaced: the land
mask extends about 11 km into the sea, so "share of the grid" and "share of eligible land" differ by
enough to change what a reader concludes.

**Why the top-up spend is deferred:**
[ADR-018](../decisions/018-refuse-pixels-below-minimum-optical-depth.md) carries the decision, the
deferral premium (a ~$50K deferral that costs MORE to recover later, not a saving), the top-up unit,
and the two objections raised against refusing rather than flagging.

---

## 10. Generations, update cycles, and what tags are actually for

Settled while reviewing this rule, and settled **then** because no cell had been tagged yet. Once the
campaign runs, its tags are write-once forever.

### The mismatch that made tag naming feel impossible

One mechanism was being asked to do five jobs:

| job | asked by | what it wants |
|---|---|---|
| idempotence — has this cell landed? | the campaign work list | a boolean per cell |
| generation — which version, how many? | the top-up plan | a counter and history, per cell |
| reproducibility — the exact store behind a result | debugging | a snapshot pin |
| release — the published product | users | a few curated names |
| audit — when, from what, under what rule | everyone | structured data per cell |

Tags are good at reproducibility and release, serve idempotence by accident, and serve generation and
audit badly. No naming scheme fixes that, because it is a mechanism mismatch rather than a naming
problem. Two further facts settle it: **a per-cell tag pins the WHOLE repo** — all 120 zone groups
share one Icechunk repository, so `zone-33N-2021` is the entire store as of the moment that cell
landed, a snapshot in which most other zones are still empty. And **per-cell reproducibility is not
wanted** (repo owner, 2026-08-13): the latest state is what users ask for, and they think in **update
cycles**, not per-cell dynamics.

### The model

**An update cycle is the user-facing unit.** One pass of the campaign — the initial fill, or a later
top-up batch once Element 84 publishes more imagery — is one cycle, labelled `v1.0`, `v1.1`, and so on.

**Per-cell tags are dropped entirely.** Tags mark cycles and nothing else:

```
release-v1.0            <- what a user cites
release-v1.1            <- after a top-up batch
year-2021-complete      <- already exists, keep
snap-2026-09-14T0000Z   <- optional operational pins
```

Prefixed by kind so `list_tags()` filters cleanly. ISO-8601 without colons, so lexicographic order is
chronological order and shells and URLs stay unharmed. **Never encode a mutable fact** — a count, a
percentage, a coverage figure — in a name that can never be changed.

**Generation lives in the data, not in a name.** `run_provenance` currently ends
`return {**existing, str(year): record}`, which **replaces** the year's record — so a top-up would
silently erase the first fill's `run_id`, `optical_skips`, `input_coverage` and `code`, which is
exactly the evidence a top-up needs to compare against. Change `runs[year]` to a **list**, append
rather than replace, and give each record a `cycle` field. Then generation of a cell is
`len(runs["2021"])`, what changed in a top-up is the last two records, and what is in a release is
every cell whose runs contain that cycle.

### The contract: attrs alone must answer every user question

The registry is an **efficiency layer, never a source of truth**. Anything a data user needs must be
answerable from the store itself:

| question | answered from the store alone |
|---|---|
| what rule produced this product? | root attr `optical_min_obs` |
| which cycle is this store at? | root attr `current_cycle` |
| has this cell been filled — how often, when, by which code? | `runs[year]` list |
| which tiles in this cell were fully skipped? | `runs[year][-1].optical_skips.labels` |
| how much of this cell was refused, and why? | the optical/radar summaries in the same record |
| **is this specific pixel refused?** | `s2_obs_count < optical_min_obs`, and `isnan(scales)` |

What the registry adds is speed and reach, not facts: per-shard rows, the chunk bitmask, the depth
histogram, and cross-zone queries in one file instead of 120 group reads.

**The one honest exception**, which must be stated in the README rather than glossed: a
**fully-skipped** shard writes no arrays at all, so its per-pixel depth is not recoverable from the
store. The attrs still name it (`optical_skips.labels`, which is complete), so a user can always tell
"refused" from "ocean". Only the *distribution* of how thin it was lives solely in the registry, and
that is a top-up planning field rather than a user-facing one.

### Migrating idempotence off tags — the one real hazard

The work list currently reads `not status.has(z, y) or zone_year_tag(z, y) not in existing_tags`.
`status.has` reads `years_complete`, which the shard writer advances **atomically with the data**,
through exactly one writer that also writes `runs`. So the attr alone is a sound "the data landed"
signal and the tag half is belt-and-braces.

**But the two halves are not synonyms**, and this is the thing to get right: `years_complete` means
*the data landed*, while the tag means *the cell was finalised* — committed, tagged, and its validation
dispatched. A crash between those two points leaves a cell that `status.has` calls done and that never
got tagged, and today's OR re-runs it. Dropping the tag check without replacing that distinction would
silently skip such a cell.

Replace it with the cycle record, not with nothing: a cell is done **for this cycle** when `runs[year]`
contains a record whose `cycle` matches the running one, and that record is appended at the point the
tag is created today. The work list keys on that. A resume then behaves exactly as it does now, and a
top-up cycle naturally re-selects every cell it wants.

### What this means for the refill tooling, which lives in `yield-embeddings`

An earlier draft called `yield-embeddings/scripts/reopen_zone_year.py` a missing tool. **That was wrong and is
retracted** — it exists, along with `drop_run_field.py`, `validate_zone_group.py`, `campaign_health.py`
and `validate_all_cells.py`, in the sibling **`yield-embeddings`** repository, which is where the
validation instrument lives.

What *does* change is how much work `reopen_zone_year.py` has to do. Its docstring describes a
three-step refill whose **middle step is expected to fail**: it clears one year from `years_complete`,
the refill then commits its data and raises at the tag step because the canonical `zone-<Z>-<Y>` name
is already spent, and the operator pins the new snapshot under a fresh name with `--pin-fresh-tag`.
Under this design that awkwardness disappears at the root: there is no per-cell tag to collide with,
`runs[year]` appends so a refill no longer erases the original fill's provenance, and idempotence keys
on the **cycle**. The residual need is narrower — forcing **one** cell to refill outside a cycle, after
a defect — which becomes a `--force` flag on the fill.

**Which repo owns what.** This is a cross-repo change: `yield-embeddings` holds the refill and
validation tooling, and it reads the campaign's tags and `years_complete` semantics, both of which move
here. The dependency is one-way and currently clean (`tessera-embeddings` imports `yield_embeddings`
**zero** times); keep it that way. The boundary is **deployment identity**, not domain versus
operations — all five scripts import `yield_embeddings.config.buckets` while their logic is this repo's
data model. And this repo is **public**, so it matters:

> **This repo owns the data model and the operations on it. The private repo owns deployment
> identity.** Domain functions take a `BucketPaths`; the private side populates it. Nothing carrying an
> account, a bucket name, or a Prefect deployment name moves here.

Applied to the refill: **do not relocate `reopen_zone_year.py` — dissolve it.** Its logic is four
tessera imports deep in this repo's own campaign model; only its wiring is Arbol's. The four validation
scripts stay where they are: they need `yield_embeddings.domain` as well as bucket config.

**The validation modules move here after the campaign** —
[ADR-019](../decisions/019-validation-modules-belong-in-the-library.md) carries the reasoning, the
sizing (~4,600 lines with essentially nothing to untangle), the new optional Pillow extra, what stays
behind, and the two prose statements in that validator which the current rule makes wrong.
