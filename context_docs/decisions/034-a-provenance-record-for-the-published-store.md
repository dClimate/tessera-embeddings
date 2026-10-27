# 034 — A provenance record for the published store

**Status:** Proposed 2026-10-27, ingest half in PR #253 and the model and assembly half in the PR
stacked on it. The design, its sizes and its reviews are in
[`storage/registry-provenance-plan.md`](../storage/registry-provenance-plan.md); this record keeps
the decisions and what was deliberately left out.

## Context

A reader of the published store asks what satellite imagery is behind an embedding. The ingest knows
which images it fused into each day's mosaic and the model knows which dates each pixel used, but
neither wrote it down, and the mosaics are deleted once a zone-year lands. v1.1 therefore has no
provenance beyond its registry, and none can be reconstructed.

## Decision

1. **Record, beside the registry, every granule fused into each mosaic slot, which of a shard's
   dates the model used, and per pixel and date which granule supplied the pixel.** Four pieces
   under `provenance/`: `granules`, `shard_dates`, `pixel_sources` and, for holes updates,
   `update_pixels`. A run writes its own files and never another run's.
2. **Per pixel, an integer, not a bit.** Measured on a real shard-year, the integer naming the
   winning granule costs 4% more than a validity bit (1.17 MB against 1.12 MB a shard-year, about
   3.7 TB for the campaign) and is exact where footprints overlap, where a bit would leave a list of
   candidates.
3. **The published plane is `uint8`, a rank into the shard-date's `sources`; the mosaic's index is
   `uint16`, the zone-wide fusion position.** A day's fusion order is zone-wide and can exceed 255;
   a shard has a handful of contributing granules on a date, and the size above was measured on
   per-shard ranks.
4. **The index rides on the loader's own per-source hook.** The mask band (SCL, VV) is read through
   a reader that packs each granule's position above its pixels, so odc's unchanged first-wins fuser
   picks the winner and a split returns the band bit-identical. One read, no extra tasks. Loading the
   mask band per granule instead would multiply the load graph by the granules in a day.
5. **Granule rows stay per zone-day; the per-shard list is `shard_dates.sources`**, derived from the
   planes, so it names granules that supplied pixels rather than those whose footprints touch.
6. **Publication failure fails the zone-year, before the tag and the cleanup.** It is retried as
   store writes are, and a retry of a landed but untagged cell republishes from its staging before
   tagging. No marker, gate or held state.
7. **The actors and the export read the mosaics at the snapshots the plan sampled.** A commit to a
   mosaic mid-fill can change neither what the model read nor what the record says.
8. **The record is behind a `provenance` switch, off by default.** Only the public global store
   needs it; dev runs and internal stores run without it. The mosaic's ingest manifest records the
   setting as `provenance_version`, a term whose absence means off, so a resume under the other
   setting is refused both ways and a store from before the term resumes off. The model and
   assembly follow the mosaics.
9. **`zone` and `year` are partition keys, not columns**, as in the registry, whose module records
   the pyarrow type collision that carrying them as columns causes. `run_id` is a column.

## Left out

- **The granule behind a cloudy pixel.** The mosaic's index names it, but only the planes masked to
  the model's validity test are published.
- **A source index per band.** It follows the mask band. A granule with data in one band and nodata
  in another at the same pixel would break that; the products do not produce one.
- **v1.1.** Its mosaics are gone.
- **Radar quality layers** (layover, shadow, looks): a separate decision, ADR 033.
- **The copies the fallback ladder rejected.** They could be added later as rows that supplied no
  pixel without changing the schema.
- **Attribution.** The record says what went in, not how much any of it mattered.

## Consequences

- The mosaic gains a `granules` array and a `source_index` band while the switch is on; the
  ingest's code fingerprint moves, so the ingest half deploys while nothing is mid-ingest or under
  `allow_ingest_code_mismatch`.
- About 3.7 TB of planes and under 100 GB of tables are added beside the store over the whole
  campaign, about 0.2% of it.
- A pixel query joins three things: the plane, the shard-date's `sources`, and `granules`.
