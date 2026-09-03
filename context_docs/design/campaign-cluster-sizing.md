# Campaign cluster sizing — how the world's UTM zones divide across N Ray clusters

Authoritative basis for choosing `max_parallel_clusters` and `max_parallel_ingest`
on the chained (`fill_strategy="chained-clusters"`) campaign. **Authoritative for how work
BALANCES and whether commits constrain the cluster count; NOT for throughput, GPU-hours or cost** —
those are `campaign-cost-model.md`'s, and the px/s figures below predate the switch to tokens. Two halves: how the
work **balances** across clusters, and whether the **commit** path to the global
Icechunk store constrains how many you can run. Measured against the **real**
coverage bitmaps, not synthetic weights, because every conclusion here depends on
the actual distribution of land: a handful of huge continental zones and a long
tail of islands.

Reproduce any of it:

```bash
TE_CLUSTERS=8,16,24 uv run pytest tests/unit/orchestration/flows/test_cluster_balance.py -k report -s
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

## The finding

**Clusters are balanced on WORK, not on area**, and the difference is worth measuring rather than
assuming: a zone's cost is its live tiles weighted by their latitude band's observation count, which
is proportional to GPU-hours, and balancing on raw tile count instead is balancing on area.

The campaign's schedule is set by the LAST cluster to finish and clusters are long-lived, so an
uneven split is not averaged away — it is added to the campaign date.

**Eight clusters is what shipped.** `scripts/cluster_work_spread.py` re-derives the spread from the
real mask through the campaign's own partitioner, so the figure can be recomputed when the mask
changes rather than being a number in a document. Run it rather than quoting this.

**The trap it exists to avoid:** balancing on tiles looks correct and is not, because tile count and
work diverge with latitude — a high-latitude zone's tiles carry far more observations each. The
comparison is the point of the script: it says what balancing on area would cost, in the only
currency that decides when a cluster finishes.
