# Campaign cluster sizing — how the world's UTM zones divide across N Ray clusters

Authoritative basis for choosing `max_parallel_clusters` and `max_parallel_ingest`
on the chained (`fill_strategy="chained-clusters"`) campaign. Measured against the
**real** coverage bitmaps, not synthetic weights, because every conclusion here
depends on the actual distribution of land: a handful of huge continental zones
and a long tail of islands.

Reproduce any of it:

```bash
TE_CLUSTERS=8,16,24 uv run pytest tests/unit/test_cluster_balance.py -k report -s
```

The diagnostic drives the shipped `_partition_by_live_tiles` and the shipped
densest-first sort, so it reports what a campaign would actually do rather than a
model of it. Counts are snapshotted in `tests/unit/zone_density.py`; the same
module carries the recipe for refreshing them after a mask rebuild.

**Provenance of the numbers:** `s3://global-tessera-inputs-dev/masks/global.icechunk`,
built 2026-07-24 from `s3://tessera-embeddings/v1.1/global_0.1_degree_tiff_all/`,
registry sha256 `5ea80dd9…c794e`. This is the **dev** coverage store — at the time
of writing the production bucket has no mask under `masks/`. The distribution of
land does not change, but regenerate the snapshot once production is built.

## What the planet looks like

| | |
|---|---|
| UTM zones with any land | **112** of 120 |
| All-ocean zones (never dispatched) | 8 — `10S 11S 13S 14S 27S 44S 45S 46S` |
| Total live tiles (2048 px) | **360,953** |
| Largest single zone | `35N`, 9,132 tiles — **2.5%** of all land |
| Ten densest | `35N 38N 47N 48N 36N 34N 37N 33N 32N 19S` (9,132 → 7,951) |

## The three properties the design rests on

All three are *consequences* of code that never states them outright — the
longest-processing-time assignment in `_partition_by_live_tiles` and the
densest-first sort in `fill_zones_sequential`. They are asserted in
`tests/unit/test_cluster_balance.py` so a refactor cannot quietly lose them.

1. **Every cluster opens on a dense zone.** All cluster totals start at zero, so
   the N densest zones of a year are dealt one to each of the N clusters. This is
   what lets a cluster request GPUs after a *single* zone has ingested instead of
   waiting out its whole list.
2. **Every cluster runs dense → sparse.** The autoscaled fleet then only ever
   shrinks (no mid-run worker relaunch) and the cheap island zones land at the tail.
3. **Clusters finish together.** Near-even total work, so no cluster is still
   grinding after the others have released their fleets.

## At the shipped default of 8 clusters

| | |
|---|---|
| Tiles per cluster | 45,117 – 45,123 (**0.0%** spread — six tiles out of forty-five thousand) |
| Zones per cluster | 13 – 15 |
| Opening zone | 8,593 – 9,132 tiles |
| Opening zone as a share of its cluster's work | **~20%** |
| Ingest window per cluster | 5 zones (`max_parallel_ingest` 40 ÷ 8) |

One cluster in full, in the order it will work it:

```
35N:9,132  20S:7,004  50N:6,917  16N:5,805  29N:5,514  18S:3,787  50S:2,915
23N:1,402  05N:1,150  07N:850  32S:386  25S:182  40S:58  09S:10  31S:6
```

That 20% is the margin the GPU-start rule spends: inference on the opening zone
has to outlast the ingest of the five zones queued behind it. Inference is slower
than ingest in almost every case, so it does.

## Scaling the cluster count

| Clusters | Even share | Largest zone ÷ share | Balance spread | Smallest opener |
|---:|---:|---:|---:|---:|
| 1 | 360,953 | 3% | 0.0% | 9,132 |
| 2 | 180,476 | 5% | 0.0% | 9,100 |
| 4 | 90,238 | 10% | 0.0% | 8,803 |
| **8** | **45,119** | **20%** | **0.0%** | **8,593** |
| 12 | 30,079 | 30% | 0.3% | 7,808 |
| 16 | 22,560 | 40% | 0.6% | 7,004 |
| 20 | 18,048 | 51% | 0.7% | 6,694 |
| 24 | 15,040 | 61% | 7.9% | 6,327 |
| 32 | 11,280 | 81% | 2.3% | 5,590 |
| 40 | 9,024 | 101% | 5.2% | 4,715 |
| 60 | 6,016 | 152% | 108% | 1,441 |

**Headroom to 16 is free.** 0.6% imbalance, every cluster still opening on a
7,000-tile zone. Doubling the clusters roughly halves the year's wall clock at
the same total GPU-hours, so 16 is the obvious move if wall clock matters more
than blast radius.

**Degradation starts around 20–24, not at the theoretical knee.** The arithmetic
limit is at 40, where a perfectly even share equals the largest single zone and
that zone alone becomes the critical path — a cluster cannot be handed a fraction
of a zone. But balance is already visibly worse at 24 (7.9%), because the split is
being squeezed by the *several* largest zones, not just the one.

**Degradation is not monotonic.** 24 clusters (7.9%) is worse than 32 (2.3%). That
is an artefact of greedy assignment, not a bug — but it means a cluster count
picked by intuition can land somewhere worse than a larger number would.

## The small-zone fleet-fill worry is a non-issue

`runners/sequential_fill.py` carries a KNOWN LIMITATION: only `look_ahead + 2`
zones can have tiles in flight, so a cluster of zones each far smaller than the
fleet can leave actors idle. On real data this never bites.

| | |
|---|---|
| Zones smaller than a 20-actor fleet | 9 zones, 71 tiles = **0.02%** of all work |
| Zones under 100 tiles | 18 zones, 473 tiles = 0.13% |
| Zones under 500 tiles | 29 zones, 3,323 tiles = 0.92% |
| Last five zones of a typical cluster | ~1% of that cluster's work |

The taper is real and costs essentially nothing. Do not spend effort decoupling
admission from fleet-fill parallelism on this evidence.

## Known limitation: the balance is spatial, not temporal

Zones are balanced on **land area**, not on observation count. A zone with the
same tile count but twice as many usable dates costs more inference, and nothing
in the split accounts for that. The area balance at 8 clusters has six tiles of
spread, so there is a great deal of headroom to absorb it — but if clusters ever
finish at visibly different times despite this document saying they should not,
**temporal depth is the first thing to check**, not the partitioner.

## Why the diagnostic lives in `tests/`

It drives `_partition_by_live_tiles`, which is inside the Prefect layer, so it
cannot be imported from a `profiling/` console script without breaking the
no-prefect-outside-the-prefect-layer rule. Printing is its output contract, hence
the scoped `T201` exemption in `ruff.toml` alongside the profiling tools'.

## Related

- [ADR-011 — campaign-triggered per-zone ingestion](../decisions/011-campaign-zone-ingestion.md):
  the ingest cap, the GPU-start rule, and why the density ordering is load-bearing.
- [`orchestration/prefect/README.md`](../../src/tessera_embeddings/orchestration/prefect/README.md):
  the operational view of both fill strategies.
- [`runners/sequential_fill.py`](../../src/tessera_embeddings/orchestration/runners/sequential_fill.py)
  module docstring: the canonical statement of how the chained stream keeps a fleet busy.
