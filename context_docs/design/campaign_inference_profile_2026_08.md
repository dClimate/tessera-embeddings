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

**1. `t_kept` centres far below 145 — median 64 — but its DISTRIBUTION reaches 145.** p90 is 122 and
the maximum is exactly 145. That reframes the question and is the most important nuance here: the
census figure may be a faithful description of *dense* chunks while the median describes typical
ones, in which case the census is not wrong but differently weighted. Three further reasons not to
re-base the budget on this yet:

- These are southern-hemisphere and tropical zones (49S, 48S, 17S, 32S). The census is **land-area
  weighted across 112 zones**, and dense northern zones — the ones that would sit at the top of the
  distribution — are absent.
- A previous "the census is ~2× high" claim was withdrawn because its chunks ran a 50× stricter keep
  threshold, biasing `t_kept` low in exactly the direction claimed. The campaign path removes *that*
  bias, not the weighting problem.
- Per-chunk `t_kept` here spans 34–145, a 4.3× internal spread. A median from four zones is not a
  land-weighted mean.

**This is what the 17-zone fill programme exists to settle**, and it is close: 16 distinct zones are
filled or in flight. Treat 145 as the planning figure until the weighted mean lands.

**2. Per-actor throughput is BELOW the reference: 1.37 M tok/sec against ≈1.9 M.** Same instance
type, same basis — a genuine **1.39×** shortfall that pushes GPU-hours up. Untested candidates:
whole-zone work mixes strategies (`single+xstarter` dominant, `dense/prefetch+starter` for a
minority) where the reference ROI was more uniform, and the reference ran 12 actors against 20 here.

**Net direction, not a new budget line.** Naively combining a 2.3× token reduction with a 1.39× rate
shortfall lands near **$350 k** against the planned $538 k. That multiplies a well-measured rate by a
badly-weighted token census, so it is a direction only. **The weighting is worth more than the rate:
fix `t_kept` first.**

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
