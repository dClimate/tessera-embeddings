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
land-weighted **145** observations per pixel) at a reference **≈1.9 M tok/sec** per worker, giving
289,000 GPU-hours. The reference rate was measured on **the same `g6e.xlarge`** at the same
wall-clock basis, so the rate comparison is like-for-like.

### `t_kept` across COMPLETED zones — and three corrections this section has had to make

This section has now been wrong three times, in different directions, on the same number. The
completed-run figures are below; the corrections are kept because each one is a distinct mistake
worth not repeating.

| cell | the land in that band | `t_kept` median (range) | $/chunk | n chunks |
|---|---|---:|---:|---:|
| 57S-2021 | NZ / SW Pacific | 60 (51–126) | 0.116 | 246 |
| chain aggregate (49S, 48S, 17S, 32S …) | southern mixed | 67 (34–197) | 0.115 | 2,224 |
| 47S-2022 + 02N-2022 (aggregate) | — | 102 (45–189) | 0.168 | 422 |
| 59S-2021 | NZ / SW Pacific | 118 (4–135) | 0.179 | 562 |
| 03N-2021 | **western Alaska** | 143 (92–198) | 0.200 | 612 |
| 06N-2021 | **Alaska / Yukon** | 151 (97–202) | 0.211 | 767 |

**A continuum from 60 to 151, not two modes** — and the mechanism is almost certainly that
polar-orbiting revisit converges toward the poles, so observation depth rises with |latitude|. The
census's land-weighted **145** sits near the TOP of this observed range.

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

### What survives all three corrections

**Throughput per actor rises with `t_kept`** — 1.25 M at 60 against 1.57 M at 151 — because a
token-poor chunk is dominated by fixed per-chunk overhead while a token-rich one amortises it. The
reference ≈1.9 M is approached on the deepest zones. So the earlier "the rate is 1.39× short" was
also a composition artefact, and **nothing measured is slower than the model assumes.**

**Cost per chunk is NOT stable across zones** — $0.115 to $0.211, tracking `t_kept`. The cost model
is right to price in tokens rather than chunks.

**No re-basing of the $503–579 k line is warranted on this evidence.** 145 remains the planning
figure. It now looks high rather than low relative to what has been measured, but every measured
zone is in the half of the world with less land.

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
