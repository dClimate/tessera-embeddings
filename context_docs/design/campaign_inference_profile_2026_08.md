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
