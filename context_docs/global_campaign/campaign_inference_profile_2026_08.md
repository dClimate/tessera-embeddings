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

## What this run established, after four corrections

**Condensed 2026-08-17 from 818 lines.** The per-measurement working — the `t_kept` derivation and
its three corrections, the chunk-level model, the interval, the four-cell radar table, and the two
assembly worker-cap passes — was the road to figures that now live in `campaign-cost-model.md` §6b
and §6c, which is the authority for all of them. Kept here is what the cost model cannot carry:
what this instrument can and cannot see, and four things this document got wrong in ways worth not
repeating. Full working in git history.

### Still standing

**Price in tokens, not chunks.** A fixed per-chunk price cannot describe a 2× spread whatever
causes it. This is the finding the cost model's whole structure rests on.

**Throughput stratifies by radar status, and the split is the reason.** Optical tokens per second
per actor, busy:

| population | tok/s per actor |
|---|---:|
| radar-free | 2.26 M – 2.93 M |
| one orbit | 1.60 M |
| **both orbits** | **1.26 M – 1.62 M** |

The old planning reference of ≈1.9 M sits **above every both-orbit cell measured and below every
radar-free one**, which is what made a single rate untenable. All figures here are *optical*
tokens, from `t_kept × valid_px` — the unit mismatch that correction 4 is about.

### Four corrections, and what each one teaches

**1. "Cost per chunk tracks `t_kept`" — WITHDRAWN.** The deepest cell measured (23N-2021,
`t_kept` 158) is among the cheapest ($0.113) *because it is radar-free*. Depth and radar status
are confounded across the measured set, and this document read the combination as depth alone.

**2. "Nothing measured is slower than the model assumes" — WITHDRAWN.** True in aggregate,
false for the population that matters: stratified by radar status the picture reverses.

**3. Land shares from a per-ZONE presence survey, presented as coverage — WITHDRAWN.** The
area-weighted per-pixel census in the cost model is the authority: 81% of land covered in
2022–2024 against 100% for 2017–2021, and 6.8% of pixel-years optical-only after Sentinel-1B
failed. Radar-free work is a **larger** share than the withdrawn figure implied, which makes the
throughput split matter more, not less.

**4. `t_kept` 145 as a planning depth — WITHDRAWN, unit-mixed.** 145 is a COMBINED census figure
and `t_kept` is optical, so "inside the observed 57–158 range" compared quantities in different
units. Measured planning depths are **103.1 optical** (land-weighted) and **170 combined**.

### The two lessons worth more than the numbers

**Refusing to claim a direction was load-bearing.** With the exposure known to be a unit mismatch,
two terms pointed opposite ways: the census counts S2+S1 while the rate counted optical only
(making the line conservative), but the rate was measured at sites carrying at most one orbit and
both-orbit chunks are slower per optical token (cutting the other way). An earlier version said
the reference "looks optimistic by roughly 20–35%"; that was withdrawn as unsupported, because it
netted one term against nothing. **When the correction was finally taken with both terms measured,
the line moved 0.98× — and a one-error correction would have moved it 19% the wrong way.**

**A warning does not retract the sentence it invalidates.** This document restated "cost per chunk
tracks `t_kept`" as a surviving conclusion *two sections after* the radar finding showed that range
mixes two populations — and then used the restatement to rule out re-basing the budget. The warning
and the claim it invalidated sat in the same file for a day. Withdraw the sentence, not just its
neighbourhood.

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
