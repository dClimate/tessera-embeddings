# Campaign inference, measured — the P3 chained run

**What this is.** The first per-zone GPU profile taken on the **campaign path** (whole UTM zones,
campaign thresholds, `fill-zones-sequential`) rather than on the Iowa reference ROI. It exists
because two of the cost model's headline inputs — tokens per pixel and the per-worker inference
rate — can both be read directly off this run, and one of them is the largest unvalidated number
in the campaign budget.

Companion to `inference_gpu_saturation_profile_2026_07.md` (single-ROI saturation work) and
`campaign-cost-model.md` (which this feeds). Source: `p3-chained-7zones-v2`, cells processed
sequentially in one process on 20–21 `g6e.xlarge` actors.

## How the numbers were derived, and where they can be wrong

Everything comes from the `CHUNK_SUMMARY` JSON line each actor emits per chunk. Three properties
of that instrument matter, and getting any of them wrong changes the answer:

1. **Every event is logged twice** — once `|`-delimited, once not. A naive count doubles
   everything.
2. **A failed deferred write requeues the item and the chunk is inferred AGAIN**, re-emitting the
   same `label` at a later timestamp. So *delivered* chunks and tokens must count **distinct
   labels**, while GPU-hours and cost must count **every pass**, because the rework really consumed
   GPUs. Conflating the two made 49S look 8% cheaper per chunk than 48S when they agree to within
   2% — the apparent difference was entirely its rework.
3. **`CHUNK_SUMMARY` carries no zone.** Its `label` is `chunk_<row>_<col>`, grid-local, and the
   chain runs cells sequentially in one process. Zones were therefore separated **by time**, using
   the chain's own `Deleting prefix …/mosaics/<zone>/<year>` markers as each cell's closing
   boundary. That is an inference from the log, not a field in it: a chunk adjacent to a boundary
   could fall in the wrong bucket, so per-zone counts carry ±1–2 chunks. **Adding the zone to this
   line would remove the whole class of doubt** and is the single cheapest instrumentation fix here.

`write_confirmed` is **always `false`** in this line and that is by design, not a fault — the write
is confirmed one chunk later via chain-confirmation, which updates an in-memory record the log line
never sees. Do not read it as an unconfirmed-write problem.

## Measured, three cells

| | 49S-2021 | 48S-2021 | 17S-2021 (partial) |
|---|---:|---:|---:|
| chunks delivered | 1,005 | 627 | 86+ |
| inference passes | 1,106 | 627 | 86 |
| **rework** | **101 (9.1%)** | **0** | **0** |
| wall span (h) | 3.29 | 2.02 | 0.25 |
| actors | 20 | 20 | 21 |
| GPU-hours | 65.9 | 40.5 | 5.3 |
| cost at $1.861/GPU-h | $123 | $75 | $10 |
| `infer_s`/chunk, median | 198.7 | 218.2 | 197.6 |
| overhead/chunk, median | 12.5 s (8.0%) | 6.2 s (3.5%) | 4.8 s (3.3%) |
| **`t_kept` median (range)** | **64 (49–145)** | **62 (34–138)** | **68 (50–136)** |
| tokens delivered | 316.9 G | 182.0 G | 26.7 G |
| **$/delivered chunk** | **0.122** | **0.121** | **0.114** |
| tok/s per actor (wall) | 1.34 M | 1.25 M | 1.40 M |
| skipped chunks | 1 | 4 | 0 |

**$/delivered chunk is the tight, transferable unit: $0.114–0.122 across three zones.** Per-token
cost is looser ($0.349–0.414 per G token) because `t_kept` varies more than chunk work does. Quote
the per-chunk figure for budgeting and the token figure only where tokens are the actual driver.

## Two findings that bear on the cost model, and they pull opposite ways

The model costs inference at **$503–579 k**, from 1.98 × 10¹⁵ tokens (1.363 × 10¹³ pixels at a
land-weighted **145** observations per pixel) at a reference **≈1.9 M tok/sec** per worker, giving
289,000 GPU-hours. Both inputs now have campaign-path measurements, and the reference rate was
measured on **the same `g6e.xlarge`**, so the comparison is like-for-like.

**1. `t_kept` is far below 145 — median 62–68 on three zones.** That is 2.1–2.3× lower, and it is
the dominant term. **But do not re-base the budget on it yet**, for reasons that killed the last
version of this claim:

- **n = 3 zones**, and all three are southern-hemisphere and comparatively sparse (49S, 48S, 17S).
  The census figure is **land-area weighted across 112 zones**; dense northern zones are exactly
  the ones missing here.
- **The within-zone range reaches 138–145.** So 145 may faithfully describe dense chunks while the
  median describes typical ones — in which case the census is not wrong, only differently weighted.
- A previous "the census is ~2× high" claim was withdrawn because its chunks ran a 50× stricter keep
  threshold, biasing `t_kept` low in precisely the direction claimed. These cells ran the campaign
  path, which removes *that* bias but not the weighting problem.

**This is what the 17-zone fill programme exists to settle**, and it is close: 16 distinct zones are
filled or in flight. Treat 145 as the planning figure until the weighted mean lands.

**2. Per-actor throughput is BELOW the reference: 1.25–1.40 M tok/sec against ≈1.9 M.** Same
instance type, same wall-clock basis (tokens ÷ actor-seconds), so this is a genuine 1.36–1.52×
shortfall and it pushes GPU-hours **up**. Likely candidates, none yet tested: whole-zone chunks mix
strategies (`single+xstarter` dominates here, with `dense/prefetch+starter` for 11–19%) where the
reference ROI was more uniform; and the reference was 12 actors against 20–21 here.

**Net direction, and why it is not a new budget line.** Naively combining them — tokens × 0.45,
GPU-hours ÷ rate — lands near **$350 k** against the planned $538 k. That arithmetic multiplies a
well-measured rate by a badly-weighted token census, so it is a *direction*, not a figure. The
weighting is worth more than the rate: fix `t_kept` first.

## Operational notes

- **Rework is real and uneven: 9.1% of 49S's GPU time, 0% of 48S's and 17S's.** It comes from
  deferred S3 write failures, which requeue the chunk for full re-inference rather than just
  retrying the upload. At campaign scale that is worth watching — a 9% GPU tax is ~$48 k on a
  $538 k line — and worth an eventual fix that retries the *write* rather than the inference.
- **First cell pays a visible overhead premium**: 8.0% on 49S against 3.3–3.5% after. Cold model
  load and cluster warm-up, amortised across a chained shard rather than paid per zone.
- **The chain deletes each cell's staging prefix and source mosaics after the fill lands**, which is
  correct and deliberate — the embeddings carry `years_complete`, so the mosaic is reclaimable. Two
  consequences: a "complete mosaic" is a transient state, and a cell that has been cleaned looks
  identical to a never-ingested one from the mosaic side. Judge doneness from the embedding store's
  year tag, never from the presence of mosaics.
- Its own delete left residue on one prefix while reporting success, the same failure mode seen
  operating `s5cmd` by hand. Verify a reclaim by listing the prefix.
