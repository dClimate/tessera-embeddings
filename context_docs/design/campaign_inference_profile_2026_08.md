# Campaign inference, measured — the P3 chained run

**What this is.** The first GPU profile taken on the **campaign path** (whole UTM zones, campaign
thresholds, `fill-zones-sequential`) rather than on the Iowa reference ROI. It exists because two of
the cost model's headline inputs — tokens per pixel and the per-worker inference rate — can be read
directly off this run, and one of them is the largest unvalidated number in the campaign budget.

Companion to `inference_gpu_saturation_profile_2026_07.md` (single-ROI saturation work) and
`campaign-cost-model.md` (which this feeds). Source: `p3-chained-7zones-v2` on 20 `g6e.xlarge`
actors, cells processed sequentially in one process.

## READ THIS FIRST: what this telemetry can and cannot tell you

> **`CHUNK_SUMMARY` cannot be attributed to a zone, and attempting it produced two wrong findings
> before the attribution itself was checked.** The line carries no zone. Its `label` is
> `chunk_<row>_<col>` — **grid-local**, so every cell restarts at `chunk_0_0` and labels collide
> across zones *and* across concurrently running fills, which share one log group.
>
> Attributing by time window and treating repeated labels as retries produced a confident "49S
> re-inferred 101 chunks after write failures, a 9.1% GPU tax". **Both halves were false.** There is
> not one `staging write unconfirmed` / `writer pool wedged` / `flush failed` line in the window —
> the mechanism did not fire at all. And the giveaway was in the data already: `t_kept` *differed*
> between the supposed retry pairs (60→56, 122→58), which cannot happen when re-running identical
> data. They were two different zones sharing a label space.
>
> A second attempt, splitting cells where a label repeats, is also unsound: a genuine within-cell
> retry splits a segment, and the boundaries it produced contradict the chain's own per-cell counts.
>
> **So this document reports aggregates over ONE log stream, plus per-cell facts the chain states
> outright.** It does not report per-zone throughput, because the instrument cannot support it.

**The fix is one field.** The chain already mints a per-cell run id (`49S-2021-f1fa65fc`) and logs
it. Adding that id — or just the zone — to `CHUNK_SUMMARY` makes every figure below decomposable
per zone, retries distinguishable from collisions, and concurrent fills separable. It is the
highest-value instrumentation change available and it is additive, so it cannot break the
`campaign_progress.py` needles.

`write_confirmed` is **always `false`** in this line, by design: the write is confirmed one chunk
later via chain-confirmation, which updates an in-memory record the log never sees. It is not an
unconfirmed-write signal.

## Per-cell chunk counts, as the chain states them

Authoritative, because the chain logs them with the zone and the run id
(`Zone <z> year <y>: N/<total> tiles are live in the campaign coverage mask`):

| cell | live chunks | of grid |
|---|---:|---:|
| 49S-2021 | 943 | 14,355 |
| 48S-2021 | 774 | 14,355 |
| 17S-2021 | 767 | 14,355 |
| 32S-2021 | 386 | 14,355 |
| 58S-2021 | 354 | 14,355 |
| 02N-2021 | 245 | 15,048 |

**A useful sanity figure in its own right: live chunks run 1.6–6.6% of the grid** on these
southern/tropical zones. Fill cost scales with the live count, not the grid.

## Aggregate, one runner, cells 49S → 32S

| | |
|---|---:|
| chunk events (deduped) | 2,123 — 2,118 success, 5 skipped |
| wall span | 6.55 h |
| actors | 20 |
| GPU-hours | 131 |
| cost at $1.861/GPU-h | **$244** |
| `infer_s`/chunk, median | 206.9 |
| **`t_kept`** | **median 64, p10 55, p90 122, range 34–145** |
| tokens delivered | 648.0 G |
| **tok/s per actor (wall-clock)** | **1.37 M** |
| **$/chunk** | **0.115** |

Skips are rare and benign: **5 of 2,123 (0.24%)**.

## Two findings that bear on the cost model, and they pull opposite ways

The model costed inference at **$503–579 k**, from 1.98 × 10¹⁵ tokens (1.363 × 10¹³ pixels at a
land-weighted **145** observations per pixel) at a reference **≈1.9 M tok/sec** per worker — the
rate being *optical* tokens/sec and the census S2+S1, a unit mismatch **settled 2026-08-07**:
the line is re-based on one unit to **$527,000 (0.98×)** in `campaign-cost-model.md` §6b, and
"The radar term, measured" below carries this document's share of the evidence. The reference
rate was measured on **the same `g6e.xlarge`** at the same
wall-clock basis, so the rate comparison is like-for-like.

### `t_kept` across COMPLETED zones — and three corrections this section has had to make

This section has now been wrong three times, in different directions, on the same number. The
completed-run figures are below; the corrections are kept because each one is a distinct mistake
worth not repeating.

Re-measured 2026-08-06 with a truncation-free profiler (see "The instrument was undercounting"
below), so every row is the cell's whole population rather than whatever fitted in one query.

| cell | `t_kept` median (p10–p90, range) | $/chunk | n chunks / live |
|---|---:|---:|---:|
| 32S-2022 | 57 (48–97, 40–121) | 0.148 | 342 / 386 |
| 57S-2021 | 60 (54–118, 51–126) | 0.116 | 247 / 267 |
| 57S-2022 | 63 (58–122, 32–135) | 0.069 | 267 / 267 |
| chain aggregate (49S, 48S, 17S, 32S, 58S, 02N, 47S) | 65 (55–132, 34–197) | 0.115 | 3,637 |
| 16S-2021 + 16S-2022 (aggregate) | 71 (66–136, 61–139) | — | 34 / 34 |
| 26S-2021 | 82 (69–134, 66–137) | — | 25 / 27 |
| 47S-2022 + 02N-2022 (aggregate) | 102 (55–149, 45–189) | 0.168 | 422 |
| 59S-2021 | 118 (61–128, 4–135) | 0.179 | 562 / 577 |
| 59S-2022 | 119 (61–132, 38–137) | 0.156 | 486 / 577 |
| 38N-2021 | **73** (69–145, 50–206) | 0.177 | 9,051 / 9,100 |
| 53N-2021 | 127 (64–171, 54–220) | 0.158 | 3,189 / 3,269 |
| 03N-2021 | 143 (102–175, 92–198) | 0.194 | 612 / 614 |
| 06N-2021 | 151 (109–188, 97–202) | 0.202 | 857 / 856 |
| **23N-2021** | **158 (121–181, 29–204)** | 0.183 | 1,395 / 1,402 |

**The observed range is now 57 to 158**, on about 15,000 measured chunks across eighteen zones.

#### Correction 4 (2026-08-07) — every comparison of `t_kept` against the census's 145 in this section mixed units

`t_kept` is optical; **145 is a COMBINED S2+S1 figure** — established three ways in
`campaign-cost-model.md` §6b, including that 60N-2020's optical `t_kept` of exactly 145
coincides with a band whose measured *combined* depth is 210 against a censused 208 (a
coincidence that, read as agreement, would have inverted the correction). The like-for-like
comparisons are: measured land-weighted **optical** depth **103.1** against the census's
optical half of 52 — the census's optical half is LOW — and measured **combined** 170 against
the censused 145. Sentences in this section that read 145 as a comparator for `t_kept` ("sits
near the top of the observed range", "inside the observed range rather than above it") are
**retired as unit-mixed**, not repaired.

**Every row above is a COMPLETED cell, and that is now a stated requirement rather than a
preference.** Two rows previously carried in-flight figures and both were wrong: 38N-2021 read
`t_kept` **121** at one third complete and **73** at completion — a 40% overstatement — and its
$/chunk moved 0.171 → 0.177.

The cause is not sampling noise. A run sweeps its zone **north to south**, and observation depth
falls with latitude, so a partial run has measured only its deepest part. Splitting 38N's 9,051
chunks into time-ordered fifths shows the median chunk row index climbing 146 → 229 → 295 → 354 →
419 while median `t_kept` falls 124 → 116 → 73 → 73 → 72. **An in-flight median is therefore biased
high by construction, predictably and in one direction.**

That is also the mechanism behind corrections 1 and 2 below, which observed the effect (03N 176 →
143, 06N 181 → 151) without identifying why. **Never quote depth, tokens, throughput or cost per
chunk from a run still in flight.**

**Two claims elsewhere in this document no longer hold as written.**

1. **"The census's 145 sits just above the top of the interval."** It does not any more. **23N's
   median is 158 and 06N's is 151**, both above 145. The census figure is inside the observed
   range rather than above it, which strengthens rather than weakens it as a planning number —
   but the sentence claiming it is conservative relative to everything measured is now false.
   *(2026-08-07: this whole comparison is unit-mixed — correction 4 above.)*
2. **"About 26% of land is effectively unmeasured."** Three of the new cells are large and
   high-latitude — 23N at 1,395 chunks, 53N at 2,887, 38N at 3,242 so far — so the thin bands at
   the top of the stratified table almost certainly have far more support now. **I have not
   recomputed the stratification**, because that needs per-chunk latitude and this profile does
   not carry it. Recomputing it is the specific next step, and it is the only thing standing
   between these figures and a revised interval.

**A zone name is not a latitude, and this table deliberately no longer pretends otherwise.** The
earlier version labelled each row with a region, which is how correction 2 below happened. A UTM
zone number is a longitude band and its letter is a hemisphere, so a zone spans every latitude of
that hemisphere and its median is a median over a mixed population. Per-band attribution belongs
to the stratified table, which derives latitude per chunk.

**A continuum from 57 to 158, not two modes** — and the mechanism is almost certainly that
polar-orbiting revisit converges toward the poles, so observation depth rises with |latitude|. The
census's land-weighted **145** sits near the TOP of this observed range *(retired — unit-mixed,
correction 4)*.

#### The instrument was undercounting, and it was invisible

`inference_profile.py` asked Logs Insights for up to 10,000 rows sorted oldest-first. **Insights
caps a non-aggregating query at exactly that many and reports the truncation nowhere**, so on a
busy log group the rows it silently dropped were the NEWEST — and because a run's later chunks
are systematically different from its early ones (see correction 2: 03N read 176 at 16 chunks and
143 at 612), the loss is biased rather than random.

**How it surfaced:** a healthy 8-actor fill was reported at 251 of 577 chunks while the fill's own
progress line said 455, which read as a stall on a run that had never stalled. Nothing in the
output hinted at it.

**What it cost, checked rather than assumed.** Re-measuring every completed cell moved the CHUNK
COUNTS — 06N went 767 → 857, the whole cell — and the per-chunk dollar figures by a few percent,
because the busy-hours denominator changed. **It did not move a single `t_kept` median:** 06N
stayed at 151 and 03N at 143 to the unit. So the load-bearing finding of this document survived
its instrument being wrong, which is luck rather than design.

The profiler now bisects its window until no sub-window reaches the ceiling, and prints a warning
if a one-minute window still does.

#### Correction 1 — "the census is ~2.3× too high". WRONG: sampled only sparse southern zones.

#### Correction 2 — "it is bimodal, and the equatorial zones are 176–181, above 145". WRONG TWICE.

**A UTM zone number is a LONGITUDE band (1–60, six degrees each); the letter is the HEMISPHERE.**
A zone therefore spans *every latitude* of its hemisphere, and 02N/03N/06N are longitudes −174° to
−144° north — **Alaska and the Bering Sea, not the equator.** Calling them equatorial inverted the
mechanism: their high `t_kept` comes from dense high-latitude revisit, the opposite of the reason
given. This is the same error as reading a zone's trailing letter wrongly in the radar audit — see
`a-granule-count-is-not-coverage`: **derive a zone's geography from the convention, never from what
the number looks like.**

And the numbers themselves moved once the runs finished: **03N went 176 → 143 and 06N 181 → 151**
between 16 chunks and 612/767. Early chunks are not a random sample of a cell.

#### Correction 3 — the sample is not land-area representative, so the weighted mean is not yet in hand.

Of 16 filled zones, **13 are southern hemisphere** and the 3 northern ones are all Alaskan. **Zero
mid-latitude northern zones are filled**, and that is where most of the world's land is. The zones
that would cover it — 35N, 37N, 38N, 53N, 12N — are still ingesting.

So "17 distinct zones" is met on COUNT but not on COVERAGE. The ±20% argument assumed between-zone
spread drives the sample size; it did not assume the zones would cluster in one hemisphere.
**Finishing the northern dense zones matters more now than adding further southern ones.**

### The CHUNK-level model — and it holds out-of-sample

Zone-level modelling failed because a zone spans every latitude of its hemisphere. Each chunk
sits at one latitude, which the mask's grid geometry gives exactly (tile row → northing →
latitude), so the model belongs at chunk level: **2,759 chunks with known latitudes instead of
four zone points.**

**Latitude does real work, and this is the confound-free evidence.** Across zones, latitude and
zone are nearly collinear here — each completed zone occupies its own narrow band — so a
cross-zone fit cannot separate them. WITHIN a zone the zone is held constant by construction,
and every zone shows a positive gradient:

| zone | n | \|lat\| span | r | slope /deg |
|---|---:|---|---:|---:|
| 57S | 493 | 6.5–11.7 | +0.113 | +2.31 |
| 59S | 562 | 12.2–47.2 | +0.243 | +0.80 |
| 03N | 612 | 52.9–70.0 | +0.489 | +2.72 |
| 06N | 857 | 59.1–70.4 | +0.429 | +3.58 |
| 53N | 235 | 68.3–75.9 | +0.509 | +5.42 |

Five of five positive. The slope steepens with latitude, so the relationship is convex rather
than linear.

**The shape is a STEP function, not a smooth curve** — median `t_kept` by band, training zones:

| \|lat\| | 5–34 | 35–59 | 60–69 | 70+ |
|---|---:|---:|---:|---:|
| `t_kept` | ~60 | ~120 | ~149 | ~179 |

Three plateaus with jumps between them, which is what increasing Sentinel-2 swath overlap toward
the poles would produce — passes become available in discrete steps, not continuously.

**Held out on 53N** (never trained on, and at the top edge of the training range so partly
extrapolating): linear fit MAE 23.1 with −17 bias; **quadratic MAE 14.9 with −5.6 bias — 3%
under-predicted.** So it does predict.

### What that does to the estimate — as an INTERVAL, because a mean is misleading here

> **Superseded 2026-08-07.** The stratification below now exists at full coverage: **22,343
> chunks across 13 zones, every one of the 17 populated 5° bands measured**, land-weighted
> optical depth **103.1** (per-band p25 85.4, p75 115.6) — replacing this section's central
> 106 (interval 75–139, built on 2,779 chunks) and retiring reason 3 below, "about 26% of
> land is effectively unmeasured". The monotonic rise (74.0 at 0–5° to 175.5 at 80–85°, no
> reversal) confirms the step-function finding. The "census assumption" row in the table is a
> COMBINED figure (correction 4); the combined planning depth is **170**. Arithmetic and the
> full band table: `campaign-cost-model.md` §6b.

A single weighted mean hides the two things that decide the answer: the spread inside each
latitude band, and the global land histogram the mean is taken over. Stratified at 5°, with each
band's own measured distribution:

| \|lat\| | % of land | tiles/yr | n measured | p10 | median | p90 |
|---|---:|---:|---:|---:|---:|---:|
| 0–5 | 7.4 | 26,811 | — | — | — | — |
| 5–10 | 7.7 | 27,940 | 425 | 54 | 59 | 119 |
| 10–15 | 7.4 | 26,679 | 74 | 60 | 63 | 65 |
| 15–20 | 8.3 | 29,785 | 62 | 61 | 65 | 128 |
| 20–25 | 8.8 | 31,732 | **6** | 62 | 94 | 110 |
| 25–30 | 8.7 | 31,429 | — | — | — | — |
| 30–35 | 7.9 | 28,399 | 18 | 59 | 62 | 124 |
| 35–40 | 6.4 | 23,189 | 39 | 62 | 123 | 127 |
| 40–45 | 6.4 | 23,160 | 328 | 61 | 120 | 130 |
| 45–50 | 6.3 | 22,833 | 103 | 61 | 118 | 124 |
| 50–55 | 6.1 | 21,887 | 84 | 111 | 119 | 132 |
| 55–60 | 4.9 | 17,653 | 94 | 102 | 122 | 171 |
| 60–65 | 5.4 | 19,358 | 680 | 102 | 150 | 167 |
| 65–70 | 4.8 | 17,168 | 712 | 138 | 149 | 192 |
| 70–75 | 2.1 | 7,529 | 135 | 164 | 173 | 201 |
| 75–80 | 1.1 | 4,032 | 19 | 187 | 194 | 199 |
| 80–85 | 0.4 | 1,369 | — | — | — | — |

**Projection, pricing unmeasured bands at the full measured range rather than interpolating:**

| | `t_kept` | tokens |
|---|---:|---:|
| low | 75 | 1.02 × 10¹⁵ |
| **central** | **106** | **1.45 × 10¹⁵** |
| high | 139 | 1.89 × 10¹⁵ |
| census assumption | 145 | 1.98 × 10¹⁵ |

**This supersedes the coarse four-band figure of 91, which was too low and too confident.** The
census's 145 sits just above the top of the interval, so it is defensible as a conservative
planning figure. **Any claim that the token census should be cut is not supported.**

**Three reasons a point estimate cannot be trusted here, all visible in the table:**

1. **Within-band spread is as large as between-band.** At 5–10° the p10–p90 is 54–119, a 2.2×
   range inside one 5° band. At 35–40° the p10 is half the median. Latitude explains part of
   `t_kept` and nowhere near all of it, so a per-band median discards most of the variance the
   projection should carry.
2. **The land histogram is nearly FLAT from 0–50°**, at 6–9% per 5° band. No band dominates, so
   no single zone can settle the answer — precision requires coverage across the whole range.
   This is why the earlier "56% of land below 35°" framing was misleading: it was true, but it
   lumped seven distinct bands into one and implied one measurement could fix them.
3. **About 26% of land is effectively unmeasured** — 15.5% in bands with no data at all (0–5,
   25–30, 80–85) plus 8.8% resting on six chunks (20–25).

**So the test programme should be selected on latitude coverage, not on which zones are ready.**
Filling 0–5°, 20–30° and the thin 30–40° bands would do more for precision than any further
high-latitude zone, where 4,000+ chunks across three zones already agree.

### Radar roughly DOUBLES per-chunk inference cost, and the token metric cannot see it

Measured 2026-08-06. This is the largest single effect in this document and it is not latitude.

**The mechanism is a code fact, not an inference from two runs, and it has two halves.**

1. **The metric cannot see radar.** `t_kept` is `S2MaskBundle.mask.shape[0]` — the count of
   *optical* timesteps surviving SCL pruning. So `tokens = t_kept × valid_px` measures optical work
   only, and every tok/sec figure here is comparable within one radar status and no further.
   (On the **combined** basis, available since 2026-08-06, comparability is restored — see
   "The radar term, measured" below.)
2. **The work really is smaller, and by a defined amount.** A radar-free pixel takes the
   `allow_s2_only` branch in `MosaicChunkInferenceDataset`, which lands it in the **smallest S1
   bucket** and hands the model an all-zeros normalised S1 slice. The forward pass therefore
   encodes a minimal radar sequence rather than a ~90-timestep one. That is the cost difference:
   not the model skipping a modality, but the sequence length collapsing to the smallest bucket.

Worth stating plainly because it also settles a correctness question: **that all-zeros slice is
what the upstream reference implementation returns for missing radar** (`resample_s1_bucket`'s
zero-count rows match `ucam-eo/tessera`'s `_sample_s1_merged` zero return). A radar-free embedding
is the upstream-defined missing-radar case, not an improvised one.

**Magnitude, from two pairs that differ in radar and are matched on the things that matter:**

| pair | radar (from the STORE, not the catalogue) | `t_kept` med | median `infer_s`/chunk | tok/s per actor | ratio |
|---|---|---:|---:|---:|---:|
| 06N-2021 | both orbits (asc 38.9, desc 51.7 obs) | 151 | **334.4** | 1.62 M | — |
| 23N-2021 | **none** | 158 | **169.0** | 2.93 M | **1.98×** |
| 57S-2021 | ascending only (26.3 obs) | 60 | **157.1** | 1.60 M | — |
| 57S-2022 | **none** | 63 | **120.6** | 2.26 M | **1.30×** |

**The anchor for the both-orbit population is now 38N-2021, and it is the largest sample in the
programme.** Its own assembly recorded `0.0% of embedded pixels have NO radar, 0.5% with fewer than
12 observations, 0 tiles wholly radar-free` — so it is fully dual-orbit, on the store's own evidence
rather than a catalogue's. Over **9,051 chunks** it runs at **1.23 M tok/s per actor**, the lowest
figure measured anywhere in the programme.

That places the two populations, on completed cells only:

| population | tok/s per actor (busy) | largest sample |
|---|---:|---:|
| radar-free | 2.26 M – 2.93 M | 1,395 chunks (23N) |
| one orbit | 1.60 M – 1.79 M | 3,189 chunks (53N) |
| **both orbits** | **1.23 M – 1.62 M** | **9,051 chunks (38N)** |

The ordering is monotone in radar burden and the largest sample sits at the bottom of it. The
planning reference of ≈1.9 M tok/sec lies **above the entire both-orbit range** — and both orbits is
the majority of campaign land — 0.51–0.57 of it by the per-pixel census.

The first pair is matched on peak actors (20 and 20) and on tokens per chunk (0.636 G against
0.640 G, within 0.7%), so the geographic variable that usually dominates is controlled. The second
pair is the SAME ZONE and the same 267 live tiles, differing in year and in radar.

**The obvious confound is controlled, and it is small.** In both pairs the radar-free member ran
LATER in the day, so any unrelated improvement that landed during the day would masquerade as a
radar effect. The control is **59S-2021 against 59S-2022**: the same zone, the same 577 live tiles,
the same 562 chunks, `t_kept` 118 against 119, **both carrying both orbits**, run on the two
different days. Median `infer_s` moved 303.6 → 287.1, a **5.4%** difference. So the day-to-day
effect is roughly 5–10%, an order of magnitude below the 1.30× and 1.98× above.

**One orbit costs about a third more, two orbits about double.** Do not extrapolate that into a
per-orbit constant from two points; the radar sequence length and the optical depth differ between
the pairs.

#### What this invalidates, and what it means

**"Cost per chunk tracks `t_kept`" no longer holds as stated.** 23N-2021 is the DEEPEST cell
measured, at `t_kept` 158, and among the cheapest, at $0.113 per chunk. The recorded $0.115–$0.211
range mixes radar-free and radar-bearing cells, and the correlation with depth is partly an
artefact of that mixing.

**"Nothing measured is slower than the model assumes" needs stratifying.** The two highest
throughput figures in the whole programme — 2.93 M and 2.26 M tok/s per actor — are both
radar-free cells, i.e. cells doing less work per token than the reference assumes. The
radar-bearing cells sit at 1.26–1.62 M.

**For the cost model, two things follow.** The radar-free ~1.2% of campaign land is roughly half
price, which is small. The consequential half is the other 98%: **a token census built on optical
timesteps understates the inference work**, so a line item derived from it is optimistic by
something like the ratios above unless radar is priced separately. That is the specific next
calculation, and it needs the radar sequence length per cell, which this telemetry does not carry.
*(Made 2026-08-07 — and the premise half-fails: the census's 145 was itself COMBINED, so the
line was not optimistic by these ratios; the net correction is −2%. Cost model §6b.)*

**A concrete instrument gap to close if this is pursued:** `CHUNK_SUMMARY` should carry the radar
timestep count alongside `t_kept`. With it, the effect above becomes a within-run regression over
thousands of chunks instead of a between-run comparison over four. **CLOSED: `t_s1_asc` /
`t_s1_desc` landed 2026-08-06, and the within-run measurement exists — next section.**

### The radar term, measured — four cells under `t_s1_asc` / `t_s1_desc` (2026-08-07)

The first cells run with per-chunk radar telemetry, from a 96-hour CloudWatch corpus (29,886
successful chunks, 1,633 carrying the radar fields). This section is the per-cell record; the
re-based cost line and the full derivation live in `campaign-cost-model.md` §6b.

| run | cell | chunks / live | `t_kept` med | asc / desc med | optical tok/s | combined tok/s (both-orbit chunks) | $/chunk | radar basis |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `p4d2-47S-2020-w16` | 47S/2020 | 235 / 239 | 61 | 30 / 31 | 1.301 M | **2.499 M** | 0.118 | both on 228 of 235 |
| `p4d2b-60N-2020-w16-run2` | 60N/2020 | 527 / 620 | 145 | 22 / 30 | 1.693 M — **mixed, do not quote whole-cell** | **2.431 M** | 0.177 | 326 both, 199 one, 2 free |
| `assembly-dense-37N-2021-w160` | 37N/2021, chunks at 65–69° | 141 | — | — | 1.601 M (both-orbit chunks) | **2.579 M** | — | 45 of 141 both |
| ~~`assembly-dense-37N-2021-resume`~~ *in flight* | 37N/2021, chunks at 30–32° | 203 | — | — | 0.756 M (both-orbit chunks) | ~~1.870 M~~ | — | 163 of 203 both |
| **`assembly-dense-37N-2021-resume` COMPLETE** | **37N/2021, chunks at 0.1–32.1°** | **4,859** | 71 | 29 / 30 | 1.005 M (both-orbit chunks) | **2.038 M** | 0.161 | 1,997 both, 2,862 one, 0 free |

Reading rules, each one earned:

- **60N gets no single rate.** Its 1.69 M averages three populations (326 both-orbit, 199
  one-orbit, 2 radar-free), and the profiler refuses to quote one figure for a mixed basis.
  Decomposed, its both-orbit chunks run 1.562 M optical / 2.431 M combined and its one-orbit
  chunks 1.946 M / 2.489 M.
- **The two 37N samples are admissible per latitude band only, never as whole-cell medians** —
  one was in flight when read, and 37N spans 0–70°+, which is why its runs report `t_kept`
  medians of 72, 121 and 149 without disagreeing. (The in-flight bias this document warns
  about is a bias in *whole-cell* medians; attributing each chunk to its own latitude removes
  the mechanism.) A related run without radar telemetry,
  `assembly-baseline-37N-2021-stagekept`, was cancelled at 44% (3,855 of 8,731; optical
  1.085 M, $0.232/chunk) and its `t_kept` 121 is an in-flight figure.
- **`-resume` COMPLETED 2026-08-07, and the in-flight row it replaces was wrong about two
  things** beyond being small. Its span read 30–32° because those were the run's *opening*
  chunks and the sweep went north to south; complete, it is 0.1–32.1°. And its both-orbit radar
  depth read 146.8 tok/px, "the deepest radar measured anywhere"; complete, it is **89.9** —
  ordinary against 60N's 76.4 and 47S's 66.3. Deep radar is real but belongs to the **32.5°
  band** (151.8 over 243 chunks), not to the cell. Both errors are the same mechanism this
  reading rule names, and both survived into the cost model for several hours.
- **The two legs together price the whole zone-year.** Baseline 3,855 chunks at 480.4 GPU-hours
  plus resume 4,859 at 421.7 gives **8,714 chunks, 902.1 GPU-hours, $1,679** — the first
  complete dense both-orbit zone-year cost we hold. The legs differ 1.50× in seconds per chunk
  and agree to **5.8%** on tokens per second, which is the best validation the token basis has:
  a per-chunk or per-pixel basis would have read one zone-year as two incompatible machines.
- **A duplicate 60N fill against one mosaic put the run-to-run noise floor at 0.3%** — treat
  any effect under ~1% on this instrument as noise.

**The finding that matters: the combined rate is invariant to radar status; the optical rate is
not.** Within each of the four cells, both-orbit and one-orbit chunks agree on the combined
basis to 1.01–1.08× while spreading 1.24–1.35× on the optical basis. Geography, day, code and
fleet are held constant by the within-run comparison, so this is the empirical justification
for the combined-token unit — the same shape of evidence that settled tokens-versus-pixels.
The radar-free cells corroborate from the other end: their optical rate IS their combined rate,
and 23N (2.934 M) and 57S (2.258 M) bracket the both-orbit combined figures instead of sitting
far above them, which is what the optical basis made them look like.

**Pooled both-orbit combined rate: 2.127 M tok/s per actor** (2,596 chunks, revised 2026-08-07
from 2.273 M over 762 when `-resume` completed). Three of the four cells agree within 6%; **37N
south of 32° is 19% below and unexplained** — strip plan, bucket fragmentation and per-chunk
overhead each ruled out by measurement, and completing the cell narrowed the deficit from 30%
without closing it. That cell now carries **77% of the pooled weight**, so the pooled rate is
largely its rate and the deficit is inherited by the campaign line rather than diluted. **Do not
price optical and radar tokens separately from this data**: the two-coefficient regression is
not identifiable (the ratio changes sign across cells) despite an R² that looks respectable.

**Radar depth is regional, not latitudinal — restated on ten bands.** Radar depth runs **66–152
tokens/px with r = +0.21** against |latitude|, where optical gives **+0.91** over the same
bands; deepest is the Middle East, shallowest the equator and the Arctic, pointing at the
Sentinel-1 observation plan rather than geometry. **The decisive evidence is non-monotonicity,
not the correlation**: the minimum (66.1 at 22.5°) sits *between* 76.3 below it and 117.3 above,
so no monotone curve fits regardless of what r reads. Radar's *share* of a sequence falls with
latitude (48% → 34–37%) only because the optical denominator rises. Planning form: a constant
**94** tokens/px land-weighted, range 66–152 — a level with a spread, never a curve. (The
sample-weighted pooling of the same bands gives 86; see the weighting note below for why land is
the right weight.) One-orbit chunks carry **0.49** of a both-orbit
chunk's radar (0.39–0.58); a radar-free chunk carries **8** (the smallest S1 bucket — a code
fact). An earlier two-cell reading ("radar depth is approximately flat while optical rises") is
**withdrawn**; and the five-band reading's r = +0.009 is superseded — at ten bands it is +0.21,
so the claim must be stated as *weak and non-monotonic*, never as *zero*.

**Unmeasured and interpolated: 35–50°.** The completion filled every 5° band from the equator to
35° and the bands above 50°, leaving **35–50 empty** — 19.2% of campaign land, and the largest
unmeasured span. It is now **interpolated** between its measured neighbours (137, 123, 108 across
its three sub-bands), which is defensible and must not be mistaken for the latitude curve the
band table forbids: interpolating one gap between two neighbours is not fitting a trend.

**Doing that interpolation is what exposed a weighting defect, and that mattered more than the
gap.** Radar depth had been pooled by CHUNK over the measured cells, while optical depth was
weighted by live tiles over campaign land — two different questions, and the pair put the halves
of one depth on two different weightings. Land-weighting radar gives **94.3 tok/px against the
sample's 86.3, +9.2%**, and the difference is geographic: the deepest radar sits at 30–35° and the
unmeasured band beside it carries a fifth of campaign land, so a sample-weighted figure
under-represents exactly where radar is deepest. The land-weighted figure is now the model's.

**Measuring that band is decided against (2026-08-08): too expensive for what it buys.** The
campaign runs on the interpolated figure, and the residual is carried rather than closed — if the
true depth there sits at the top of the measured 66–152 spread the line is understated across
19.2% of land, which is inside the published interval. Recorded as a decision so it is not
revived as an open question. The paragraph below is the superseded recommendation.

~~A measured band there remains the best available refinement — preferably outside Europe and the
Middle East, the densely-tasked Sentinel-1 regions that supply every deep observation we hold —~~
but it is a refinement, not a hole, and not worth delaying a cell for.

**Two open items this telemetry surfaced.** 93 of 60N's 620 live tiles produced zero valid
pixels (valid-pixel yield 0.811 against 0.983–0.984 elsewhere); whether high-latitude cells
systematically yield less is unmeasured. And **no both-orbit rate exists at campaign fleet
width** — these cells ran at 20–95 actors against a planned 228 per cluster, with no
actor-count trend among them but no measurement either.

**Zero-valid-pixel tiles are deterministic and spatially clustered (2026-08-07).** Both open
items moved. On the width question the widest measurement is now **160 actors**, still short of
228 — but it is the *slowest* cell and it is also the one carrying the unexplained geographic
deficit, so **fleet width and geography are now perfectly confounded** in this table. That
combination looks like an actor-count penalty and must not be read as one; separating them needs
a second wide run in a different zone.

On the zero-yield tiles, comparing the skip markers left by two independent fills of the same
cell settles what they are:

| cell | skipped | identical across two fills? | connected components (8-neighbour) |
|---|---:|---|---|
| 37N/2021 | 17 of 8,731 | **yes** | 16 + 1 |
| 60N/2020 | 93 of 620 | **yes** | 62 + 15 + 9 + 6 + 1 |

Byte-for-byte the same tile sets, a day apart, in contiguous blocks. Random worker failures
scatter; reproducible regions are a property of the input data, which matches what the code
documents a skip to mean — every pixel failed the validity filter. **So the campaign is handling
these correctly rather than losing data quietly**, and "do high latitudes yield less" is the
wrong question: 60N loses 15% of its tiles in five blocks, which is a coverage fact about
specific places, not a latitude gradient.

**What is NOT handled: the store keeps no record of them.** Skipped tiles are written as *fill*
and the year is then marked complete, and the per-year provenance record carries `run_id`,
`assembled_at`, an `empty` flag for a wholly-empty year and `radar_coverage` — but **no field
for partial optical skips**. A consumer reading 37N/2021 finds zeros across ~7,100 km² and
cannot distinguish "no valid optical data" from "not land", because ocean is also zeros. For
60N/2020 that is ~39,000 km². The argument that already justifies recording `radar_coverage`
per year — coverage is a property of what was acquired, not of the terrain — applies identically
here, and the plumbing exists: `assemble_global` already holds the skipped labels and
`write_year_shards` already forwards a summary dict into the provenance record. Folded into
Phase 5.

### Assembly: the worker cap WAS the constraint at 8, and at 16 nothing is

> **SUPERSEDED IN ITS CONCLUSION, 2026-08-07 — see "Assembly re-measured at 16 workers" below.**
> This section's finding ("the box is half idle, the cap binds, raise it") was correct at
> `max_workers = 8` and drove the raise to 16. Measured at 16 on a real dense cell, the box is
> **not** half idle and no resource binds. The section is kept because the reasoning that
> justified the raise is the reasoning that predicts where the next one would help.

Measured 2026-08-06 on 38N-2021's assembly — 9,050 staged tiles, the largest attempted.

**Assembly is not a fleet.** It runs in-process on the flow runner, which forks `n_workers`
processes that each hold one staged tile slice. The runner is the `inference` family:
**16 vCPU / 64 GiB Fargate**.

| measurement | value | against |
|---|---|---|
| write rate | **248 objects/min, flat for 160 min** (2,333–2,825 per 10-min bucket, no trend) | — |
| CPU used | avg 2,403 units, **max 7,443** (≈7.3 vCPU) | **16,384 allocated** |
| memory used | avg 3.0 GB, **max 20.2 GB** | **64 GiB allocated** |
| aggregate S3 concurrency | ~4.1 PUT/s | **budget 100** |
| `n_workers` chosen | **8** | `AssemblyConfig.max_workers = 8` |

**The binding constraint is the worker cap, not the box.** `compute_n_workers` is
`min(live_tiles / 10, max_workers)`, and `max_workers` was **8** when this was measured — so a
**9,050-tile zone got the same 8 processes as a 267-tile zone.** Parallelism did not scale with
the job, which is why a dense cell's assembly ran for hours. **Decided from this measurement:
`AssemblyConfig.max_workers` is 16 (raised 2026-08-06), and 8 is dead** — 16 is what the flow
runner was explicitly sized for.

Every resource signal agrees: at most **45% of allocated vCPU** and **31% of memory**, with a
perfectly flat rate — no degradation, no backpressure, no thrashing. A flat rate at half the CPU is
the signature of a fixed worker count, not of a resource limit.

**So: raise `max_workers` before touching the task size.** — DONE, 16 shipped 2026-08-06. The
16 vCPU / 64 GiB box was *explicitly*
sized for this — `consumer_stack.py` says "16 vCPU / 64 GiB leaves headroom for n_workers=16
(~19 GiB)". Measured at 16 workers, memory sits near **19 of 64 GiB** (the 38 GB this
paragraph once predicted was high) — **memory is not a constraint** at any pool size under
consideration.

**Do NOT adopt `assembly_large` (32 vCPU / 244 GiB) yet.** It is registered and unused, and its own
comment says to adopt it only once a measurement shows 16 vCPU is insufficient. **We are not using
half of the 16 we have.** It becomes the next step only after `max_workers=16` saturates the current
box.

**One thing that could defeat the raise, and why it probably will not.** The per-fork S3 request cap
is `TARGET_AGGREGATE_S3_CONCURRENCY // n_workers`, so aggregate PUT concurrency stays at 100 whatever
the worker count — if the assembly were S3-concurrency-bound, more workers would buy nothing. It is
measured at **~4.1 PUT/s against a budget of 100**, nowhere near the ceiling, so the budget is not
what is holding it.

**The constraint set around the pool, settled 2026-08-07.** Three things bound what raising
the pool further could buy, and only the first is live:

- **The S3 budget is DIVIDED by the pool**: 16 workers get **6** concurrent PUTs each where 8
  got 12, so raising workers without raising the budget may not raise throughput — and the
  budget should not be raised casually. Its own docstring records `SlowDown` observed at
  **800** concurrent PUTs, the budget is per-RUN, and the campaign runs 8 clusters — so the
  campaign-wide figure is ALREADY ~800.
- **The northing-band count is NOT a constraint** — raised as one and withdrawn: a real zone
  has ~456 bands, far above any worker count under consideration.
- **Memory is not a constraint** (~19 of 64 GiB at 16 workers, above).

**Phase-split instrumentation now exists**: the `ASSEMBLY_SUMMARY` record reports read and
write wall-clock and CPU per worker. Compression and upload are **fused** in one call, and the
record says so rather than inventing a split.

**Why this matters at campaign scale — corrected 2026-08-07.** An earlier version of this
paragraph said a dense cell's assembly hours "land directly on the critical path, 1,008
times". On the chained path they do not: assembly runs on the cluster's dedicated **trailing
thread** while later zones keep the GPUs busy, so it is **off the critical path by design** —
it lands on the critical path only for a cluster's final zone, or if an assembly outlasts the
next zones' inference. The runner design note puts assembly at ~10–15% of a zone's inference
wall time, **which is unverified and should be treated as such** (Phase 5's F8 is the check at
width). The cost half of the old sentence stands: the GPU
fleet is released before assembly starts ("Killing 60 actors to release resource reservations"), so
these hours cost one flow-runner task rather than sixty GPUs — a wall-clock question, not a cost one.

**Caveat on the utilisation figures.** They are Container Insights metrics for the whole cluster, so
the maxima include two concurrent 35N ingest tasks. The true assembly-only CPU share is therefore
**below** the 45% quoted, which strengthens the conclusion rather than weakening it.

### Assembly re-measured at 16 workers — no resource binds, and the runner is free during inference

Measured 2026-08-07 on **37N/2021's assembly: 8,714 staged tiles**, the largest ever assembled,
on the same 16 vCPU / 64 GiB Fargate runner. Metrics filtered to the runner's own task family,
so unlike the figures above they carry no other workload.

| measurement | value | against | verdict |
|---|---|---|---|
| CPU used | avg **10.6–12.7 vCPU**, peak 13.2 | 16 vCPU allocated | 57–79% — not idle, not saturated |
| memory used | **30–34 GiB, flat from minute one** | 64 GiB allocated | ~52%, a steady working set |
| network | **~1.0 GB/s combined** (≈550 read + ≈430 write) | Fargate 16 vCPU | ~7.5 Gbps mean, 8.9 peak |
| `n_workers` | **16** | `AssemblyConfig.max_workers = 16` | one process per vCPU |

**Nothing binds, and the network question is settled by variance rather than by level.** Over 26
consecutive minutes combined throughput ran 760–1,116 MB/s: mean 935, coefficient of variation
**9.6%**, maximum 19% above the mean. A hard cap pins the maximum at the mean and drives the
variation to zero, so this is not a cap. CPU and memory are likewise mid-range. The box is
correctly sized and **`assembly_large` remains unjustified.**

**Memory now bounds the pool, and it does so before the northing bands ever would.** 16 processes
hold ~2 GiB each. That is a ceiling near **32 workers** on a 64 GiB task — the first real bound
found on the pool, and it supersedes "memory is not a constraint at any pool size under
consideration" above, which was measured when the working set read 19 GiB. The 456-band figure
remains irrelevant.

**Two corrections to how the pool was described.** The dominant write path partitions tiles
**round-robin**, not into northing bands: `write_year_shards` builds exactly `n_workers` payloads
of ~545 tiles each, balanced by count. So the load balancing is structural, the "band" language
applied only to the ROI path, and the progress reporter's `0/16 done` for hours was the designed
behaviour of a partition scheme where nothing completes until everything nearly does. Both were
fixed 2026-08-07 (per-worker progress on a timer, and the caller naming its own unit).

**The staged intermediate is uncompressed, which is what sets the duration.** Every staged tile is
byte-identical at **570.4 MB** — exactly the embeddings array plus the scales array with no
compression — so a dense zone-year is **4.97 TB across 2.34 M objects** to read. That, not CPU,
is why assembly takes hours. Compressing the staged intermediate is the one change that would
move assembly wall-clock materially.

**Measured to completion, and the write IS the assembly.** The run's own timestamps give the
phase split without the `ASSEMBLY_SUMMARY` record:

| | measured |
|---|---|
| shard-write phase | **195.9 min** (3.27 h) |
| merge + commit | **37 s — 0.32%** |
| total | **196.6 min** (3.28 h) |
| effective read rate | **423 MB/s** |

The commit is negligible because the forked workers have already written every chunk and the
merge is metadata only — so there is no second phase to model, and the 16 partitions all
completed inside a **6-minute window** at the end, which is the round-robin balance working and
also why the old reporter read `0/16` for over three hours.

**An in-flight estimate of 2.2–3.1 h, extrapolated from a ~550 MB/s instantaneous network
sample, was optimistic.** Sustained, the effective rate is 423 MB/s. A spot network reading is
not a throughput basis — the same class of error as a partial run's median.

**Cost, measured for the first time.** The runner task is $0.93/hour, so a dense cell's assembly
is **~$3.05 of compute plus ~$1 of S3 requests**. Scaled by tile count over 1,008 campaign cells
that is **~$1,150** — superseding the ~$200 previously carried in the cost model with no
measurement behind it. Assembly is a scheduling term and never a cost term.

**The trailing-thread design holds, but by 21% rather than comfortably.** The runner sits at
0.02 vCPU for the whole of GPU inference, with one ten-minute burst of 6–8 vCPU at run start, so
the box is free for a trailing assembly needing 11–12 vCPU. **Capacity was never the question**:
what matters is whether assembly finishes before the next cell's inference does.

| | assembly | inference at 228 actors | margin |
|---|---|---|---|
| dense cell (8,714 tiles) | 3.28 h | 3.96 h | **1.21×** |
| average cell (3,222 tiles) | 1.21 h | 1.46 h | **1.21×** |

Scale-invariant, because both terms are linear in tiles. A cluster runs ~126 cells in sequence,
so a persistent deficit compounds rather than averaging out — which is what makes the 39% of CPU
left idle worth keeping as the margin's buffer rather than harvesting it. The corollary about
**two concurrent assemblies contending** is moot in practice: the trailing executor is
`max_workers=1`, so a second assembly queues instead, and the queue is exactly what a margin
this thin puts at risk.

### What survives all three corrections — and what the radar finding takes back

**This section contradicted the radar section above until 2026-08-06, and the contradiction was
mine.** It restated "cost per chunk tracks `t_kept`, $0.115–$0.211" as a surviving conclusion two
sections after the radar finding showed that range mixes two populations, and then used the
restatement to rule out re-basing the budget. Retained below with the retraction attached, because
the mistake is the interesting part: I wrote the warning and then left the sentence it invalidates
standing in the same file.

**Still stands: the cost model is right to price in tokens rather than chunks.** A fixed
per-chunk price cannot describe a 2× spread whatever causes it.

~~**Still stands: `t_kept` 145 remains a defensible planning figure for observation depth.**~~
**WITHDRAWN 2026-08-07 — unit-mixed (correction 4).** 145 is a COMBINED census figure and
`t_kept` is optical, so "inside the observed 57–158 range" compared quantities in different
units. The measured planning depths are **103.1 optical** (land-weighted, all populated bands)
and **170 combined**; `campaign-cost-model.md` §6b.

**WITHDRAWN: "cost per chunk tracks `t_kept`."** The deepest cell measured (23N-2021, `t_kept` 158)
is among the cheapest ($0.113) because it is radar-free. Depth and radar status are confounded
across the measured set, and this document previously read the combination as depth alone.

**WITHDRAWN: "nothing measured is slower than the model assumes."** Stratified by radar status,
the picture reverses for the population that matters:

| population | tok/s per actor, busy | share of campaign land (2025) |
|---|---:|---:|
| radar-free | 2.26 M – 2.93 M | see below |
| one orbit | 1.60 M | see below |
| **both orbits** | **1.26 M – 1.62 M** | see below |

**The land shares that used to sit in this table are withdrawn.** They came from a per-ZONE presence
survey and were presented as coverage. The area-weighted per-pixel census in `campaign-cost-model.md`
is the authority: **81% of land covered in 2022–2024** against a 100% baseline for 2017–2021, and
**6.8% of pixel-years optical-only** across the nine years, following Sentinel-1B's failure in
December 2021. Radar-free work is therefore a much larger share of the campaign than the withdrawn
1.2% implied, which makes the throughput split below matter more, not less.

The planning reference of ≈1.9 M sits ABOVE every both-orbit cell measured and below every
radar-free one. All of these figures — the reference included — are **optical** tokens per second,
since they all come from `t_kept × valid_px`.

**Re-basing the $503–579 k line is now a live question, and this document cannot settle it, in
either direction.** *(Settled 2026-08-07 — see "TAKEN" below and cost model §6b; the reasoning
is kept because the refusal to pick a direction was load-bearing.)* The exposure turned out to
be a UNIT mismatch rather than a rate shortfall, and
the two point opposite ways:

- The cost model's numerator, the token census, counts **S2 + S1** — 52 optical plus 91 radar
  observations per pixel. Its denominator, this rate, counts **optical only**. Dividing the first by
  the second buys more GPU-seconds than the work needs, which makes the line *conservative* on that
  axis.
- But the rate itself was measured at sites carrying at most one orbit, and both-orbit chunks are
  slower per optical token, which cuts the other way.

Neither term is pinned, so **no direction is claimed here.** An earlier version of this section said
the reference "looks optimistic by roughly 20–35%"; that is withdrawn as unsupported — it netted one
term against nothing. The full accounting, including a third term (the censused optical figure of 52
against a measured 69.9), is in `campaign-cost-model.md` beside the census table.

**What settles it is one run.** With `t_s1_asc` and `t_s1_desc` on `CHUNK_SUMMARY` from 2026-08-06,
a single both-orbit cell yields the radar term per chunk, hence the true S2+S1 token count and a
rate in the SAME unit as the census. Then the division is like-for-like.

**TAKEN, 2026-08-07 — four cells, and the line is re-based at 0.98× ($527,000).** The two
errors nearly cancelled: the combined rate is higher than the optical one (pushing the line
down) and the measured combined depth is higher than the censused 145 (pushing it up).
"Neither direction is claimed" above was the right refusal — the direction that a one-error
correction would have produced was wrong by 19%. Per-cell record: "The radar term, measured"
above; arithmetic: `campaign-cost-model.md` §6b.

**One caution against over-correcting.** The 20–35% above compares whole-cell medians across
different zones, which is exactly the kind of comparison that has been wrong three times in this
document already. The `t_s1_asc` / `t_s1_desc` fields added to `CHUNK_SUMMARY` on 2026-08-06 turn
it into a within-run regression over thousands of chunks; the honest sequence is to deploy those,
re-measure one both-orbit cell, and only then move a budget line. **That sequence was followed,
and the budget line moved 2%.**

## An independent cross-check of the inference line, from the coverage mask

`scripts/rank_zones.py` reads the campaign coverage mask directly: **360,953 live tiles across
112 zones**, i.e. per campaign year. Nine years is **3.25 M chunk-years**, and this run measured
cost per delivered chunk at **$0.101–0.214** depending on `t_kept`.

| assumed land-weighted mean $/chunk | implied inference line |
|---|---:|
| $0.115 (the sparse-zone figure) | $374 k |
| $0.15 | $487 k |
| $0.18 | $585 k |

The model's **$503–579 k** sits inside that band at a mean of roughly $0.155–0.18 — which is a
plausible place for a land-weighted mean to land, given how far apart the two modes are.

> **Watch this figure: the equatorial cost is running higher than first measured.** At 61–68
> chunks each, 03N and 06N are at **$0.30–0.31 per chunk** — not the ~$0.20 an earlier, smaller
> sample suggested — because their chunks take 475–480 s of inference against ~200 s in the far
> south. Some of that is cluster warm-up amortised over few chunks and will fall as they
> progress; how much is exactly what completing them answers. If the equatorial mode settles
> near $0.30, a land-weighted mean lands above $0.18 and the inference line moves toward the top
> of the model's range or past it. **Do not treat the band above as settled.** **This is a genuinely independent route to the same number**: it multiplies a chunk
census from the coverage mask by a measured per-chunk cost, using neither the token census nor
the reference rate that the model's own derivation depends on. Two unrelated methods agreeing to
within their spreads is the strongest evidence yet that the inference line is sound.

Do not read the table as narrowing the estimate — the mean $/chunk is exactly the unknown the
17-zone programme measures, and quoting a row of it as a result would be picking the answer.

## Operational notes

- **The chain deletes each cell's staging prefix and source mosaics after the fill lands**, which is
  correct and deliberate — the embeddings carry `years_complete`, so the mosaic is reclaimable. Two
  consequences: a "complete mosaic" is a transient state, and a cleaned cell is indistinguishable
  from a never-ingested one *from the mosaic side*. **Judge doneness from the embedding store's year
  tag, never from the presence of mosaics.**
- Its reclaim uses the same `s5cmd` path that, operated by hand the same night, reported success
  while leaving residue. Verify a reclaim by listing the prefix.
- **First cell pays a warm-up premium** in per-chunk overhead, amortised across a chained shard
  rather than paid per zone. Quantifying it per cell needs the zone field above.
