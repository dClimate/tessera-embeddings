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

The model costs inference at **$503–579 k**, from 1.98 × 10¹⁵ tokens (1.363 × 10¹³ pixels at a
land-weighted **145** observations per pixel) at a reference **≈1.9 M tok/sec** per worker — the
rate being *optical* tokens/sec and the census S2+S1, which is a unit mismatch settled nowhere yet
(see "Radar roughly DOUBLES per-chunk inference cost" below) — giving
289,000 GPU-hours. The reference rate was measured on **the same `g6e.xlarge`** at the same
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
census's land-weighted **145** sits near the TOP of this observed range.

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

**A concrete instrument gap to close if this is pursued:** `CHUNK_SUMMARY` should carry the radar
timestep count alongside `t_kept`. With it, the effect above becomes a within-run regression over
thousands of chunks instead of a between-run comparison over four.

### What survives all three corrections — and what the radar finding takes back

**This section contradicted the radar section above until 2026-08-06, and the contradiction was
mine.** It restated "cost per chunk tracks `t_kept`, $0.115–$0.211" as a surviving conclusion two
sections after the radar finding showed that range mixes two populations, and then used the
restatement to rule out re-basing the budget. Retained below with the retraction attached, because
the mistake is the interesting part: I wrote the warning and then left the sentence it invalidates
standing in the same file.

**Still stands: the cost model is right to price in tokens rather than chunks.** A fixed
per-chunk price cannot describe a 2× spread whatever causes it.

**Still stands: `t_kept` 145 remains a defensible planning figure for observation depth.** It is
inside the observed 57–158 range rather than above it, which is a change from what this document
said earlier but not a reason to move it.

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
either direction.** The exposure turned out to be a UNIT mismatch rather than a rate shortfall, and
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

**One caution against over-correcting.** The 20–35% above compares whole-cell medians across
different zones, which is exactly the kind of comparison that has been wrong three times in this
document already. The `t_s1_asc` / `t_s1_desc` fields added to `CHUNK_SUMMARY` on 2026-08-06 turn
it into a within-run regression over thousands of chunks; the honest sequence is to deploy those,
re-measure one both-orbit cell, and only then move a budget line.

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
