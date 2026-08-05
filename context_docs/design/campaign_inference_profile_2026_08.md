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

### CORRECTION (2026-08-05, later the same day): both apparent findings were zone composition

An earlier version of this section reported that `t_kept` runs ~2.3× below the census's 145 and that
per-actor throughput runs 1.39× below the ≈1.9 M reference, and reasoned toward a ~$350 k inference
line. **Both were artefacts of aggregating four southern sparse zones**, and per-zone attribution —
possible for single-cell runs by log stream, see `scripts/inference_profile.py` — reverses them.

| cell | latitude band | `t_kept` median (range) | tok/s per actor |
|---|---|---:|---:|
| 57S-2021 | far south | 57 (51–125) | 1.12 M |
| 47S-2022 | far south | 56 (45–118) | 1.31 M |
| chain aggregate (49S, 48S, 17S, 32S …) | south/tropical | 64 (34–145) | 1.38 M |
| **03N-2021** | **equatorial** | **176 (128–186)** | **1.68 M** |
| **06N-2021** | **equatorial** | **178 (136–186)** | **1.85 M** |

**`t_kept` is strongly bimodal by latitude, and the equatorial zones sit ABOVE 145, not below.**
Equatorial zones are imaged on nearly every pass and screen out little, so they retain ~177
observations per pixel against ~57 in the far south. A land-weighted mean over 112 zones is exactly
the right way to combine those, and **145 is entirely plausible as that mean** — it is not a high
figure, it is a middle one. The earlier reading sampled only the low mode and mistook it for the
distribution.

**The rate shortfall was the same artefact, and it has a mechanism.** Throughput per actor rises with
`t_kept` — 1.12 M at `t_kept` 57 against 1.85 M at 178 — because a token-poor chunk is dominated by
fixed per-chunk overhead while a token-rich one amortises it. So the reference ≈1.9 M is reproduced
on token-rich zones and the low figures are what overhead-dominated sparse chunks cost. **Nothing is
slower than the model assumes; the model's rate simply applies to token-rich work.**

**Consequence for the budget: the cost model's two headline inputs both look sound, and no
re-basing is warranted.** What this run does add is that **cost per chunk is NOT a stable unit
across zones** ($0.101–0.214 here) because it scales with `t_kept` — the cost model is right to
price in tokens.

**Caveats, stated because the corrected reading is only hours old.** The equatorial figures rest on
16 successful chunks each (both fills had just started), so their medians will move; and warm-up
overhead is still in their per-chunk cost. The far-south figures rest on 58–64 chunks. **The 17-zone
land-weighted measurement remains the thing that settles it**, and it is close: 16 distinct zones
filled or in flight. Treat 145 as the planning figure until it lands — now with the expectation that
it will be confirmed rather than cut.

**The generalisable lesson, third instance in one day:** a median over a mixed population is not an
estimate of anything. Stratify by the variable that drives the spread — here latitude — *before*
comparing against a weighted reference. See `ingest_concurrency_investigation_2026_08.md`
§"Corrections", where the same error appears as zone-mix, season-mix and threshold-mix.

## An independent cross-check of the inference line, from the coverage mask

`scripts/rank_zones.py` reads the campaign coverage mask directly: **360,953 live tiles across
112 zones**, i.e. per campaign year. Nine years is **3.25 M chunk-years**, and this run measured
cost per delivered chunk at **$0.101–0.214** depending on `t_kept`.

| assumed land-weighted mean $/chunk | implied inference line |
|---|---:|
| $0.115 (the sparse-zone figure) | $374 k |
| $0.15 | $487 k |
| $0.18 | $585 k |

The model's **$503–579 k** sits inside that band at a mean of roughly $0.155–0.18 — which is
where a land-weighted mean should land, given equatorial zones cost ~$0.20/chunk and far-south
ones ~$0.11. **This is a genuinely independent route to the same number**: it multiplies a chunk
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
