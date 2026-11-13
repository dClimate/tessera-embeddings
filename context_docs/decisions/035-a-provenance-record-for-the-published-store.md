# 035 — A provenance record for the published store

**Status:** Proposed 2026-10-27, ingest half in PR #253, the model and assembly half in #255, and
`source_runs` with the reader helpers in #256. The design, its sizes and its reviews are in
[`storage/registry-provenance-plan.md`](../storage/registry-provenance-plan.md); this record keeps
the decisions and what was deliberately left out.

## Context

A reader of the published store asks what satellite imagery is behind an embedding. The ingest knows
which images it fused into each day's mosaic and the model knows which dates each pixel used, but
neither wrote it down, and the mosaics are deleted once a zone-year lands. v1.1 therefore has no
provenance beyond its registry, and none can be reconstructed.

## Decision

1. **Record, beside the registry, every granule fused into each mosaic slot and every copy the
   ingest dropped before the fuse, on which dates a shard's pixels had data, and per pixel and date
   which granule supplied the observation and whether the model used it.** Four pieces under `provenance/`: `granules`, `shard_dates`,
   `pixel_sources` and, for holes updates, `update_pixels`. A run writes its own `shard_dates`,
   planes and mask and never another run's; `granules` belongs to the mosaic (5).
2. **Per pixel, an integer, not a bit.** Measured on a real optical shard-year, a per-shard rank
   naming the winning granule cost 4% more than a validity bit (1.17 MB against 1.12 MB), and the
   published fusion order (3) costs 16–23% more again on optical. Either is exact where footprints overlap,
   where a bit would leave a list of candidates.
3. **The published plane is `int16`, the granule's signed fusion order; the mosaic's index is
   `uint16`, the same zone-wide fusion position.** A value k means `fusion_order = |k| - 1`:
   positive when the model used the observation, negative when its validity test rejected it, a
   cloud for instance, and 0 when no granule had data. A reader can therefore tell "no image" from
   "image rejected", and see which granule was cloudy. The run's group names each sensor's mosaic
   snapshot in its attributes, and the run's `mode` (`fill`, `shards` or `holes`), which tells a
   reader whether the run claims every pixel of its shards or only its `update_pixels`, so a plane value and its date are a full `granules` key: a pixel
   query joins the plane to `granules` and nothing else, and the plane does not depend on a second,
   separately written table agreeing with it on ordering. The alternative, an `int8` rank into
   each shard-date's `sources`, is smaller: the fusion order measured 16–23% larger on optical and
   34–73% on radar, 33–41% overall, on three dev shard-years in the published layout. But the rank
   costs every reader a join and caps a shard-date at 127 granules. The larger planes were
   accepted: about 1.7–2 TB more over the campaign.
4. **The index rides on the loader's own per-source hook.** The mask band (SCL, VV) is read through
   a reader that packs each granule's position above its pixels, so odc's unchanged first-wins fuser
   picks the winner and a split returns the band bit-identical. One read and an unchanged fuse; the
   split and the index's writes are the new work. Loading the mask band per granule instead would
   multiply the load graph by the granules in a day.
5. **Granule rows stay per zone-day, keyed by the mosaic snapshot they were read from; the
   per-shard list is `shard_dates.sources`.** A granule list belongs to its mosaic, so it is
   written once per snapshot, a later run reading the same snapshot checks it instead, and each
   `shard_dates` row names its snapshot: a reader joining through it never counts one mosaic's list
   twice. `sources` is derived from the planes, so it names granules that had data at the shard's
   pixels rather than those whose footprints touch. Copies the ingest dropped follow a day's fused
   granules with no fusion order and a `status`: `duplicate` for a losing copy of an image it
   fused, `unreadable` for a copy that would not read and was replaced, `missing_bands` for a
   radar granule whose catalogue entry lacks its VV or VH file. A day the ingest skipped whole is
   listed too, off the time axis in two arrays of the mosaic committed when it is skipped, every
   copy under the day's reason: `low_coverage`, `producer_conflict`, `unreadable` or
   `missing_bands`.
6. **The record is published before the run's shards commit.** A committed run always has its
   record; a failed publication fails the run with nothing committed and its staging kept, and the
   retry publishes again, every write keyed by run id. A record whose run never commits is never
   read, since readers reach runs only through the store-derived `source_runs`. No marker, gate,
   held state or republish path.
7. **The actors and the export read the mosaics at the snapshots the plan sampled.** A commit to a
   mosaic mid-fill can change neither what the model read nor what the record says. Each staged
   shard names the snapshots it was read from, and publication refuses a shard that names others.
8. **The record is behind a `provenance` switch, off by default.** Only the public global store
   needs it; dev runs and internal stores run without it. The mosaic's ingest manifest records the
   setting as `provenance_version`, a term whose absence means off, so a resume under the other
   setting is refused both ways and a store from before the term resumes off. The model and
   assembly follow the mosaics.
9. **`zone` and `year` are partition keys, not columns**, as in the registry, whose module records
   the pyarrow type collision that carrying them as columns causes. `run_id` and `snapshot_id` are
   columns.
10. **A holes update records the pixels it embedded.** Its `update_pixels` mask marks them and its
    planes are cut to it, so the mask is exactly where a reader takes the update's record. Every
    other pixel keeps the earlier run's record, which covers it: a fill or a shards update records
    every pixel it was asked to embed, refused ones included.
11. **Reader helpers answer the three questions** (`storage/provenance_reader.py`): a day's
    granules, a shard's dates and granules for one run, and a pixel's granules from a latitude and
    longitude, located on the store's own zone grid and read from the run that claims the pixel.
    Walking `source_runs` from the newest, a holes update claims the pixels its `update_pixels`
    marks and a fill or a shards update claims every pixel.

## Left out

- **A source index per band.** It follows the mask band. A granule with data in one band and nodata
  in another at the same pixel would break that; the products do not produce one.
- **v1.1.** Its mosaics are gone.
- **Radar quality layers** (layover, shadow, looks): a separate decision, ADR 033.
- **Days whose imagery reaches no land in the zone.** None of it covered a pixel the store embeds.
- **Attribution.** The record says what went in, not how much any of it mattered.

## Consequences

- The mosaic gains a `granules` array and a `source_index` band while the switch is on; the
  ingest's code fingerprint moves, so the ingest half deploys while nothing is mid-ingest or under
  `allow_ingest_code_mismatch`.
- About 6.7–7 TB of planes and under 200 GB of tables are added beside the store over the whole
  campaign, about a third of a percent of it.
- The ingest's time cost is inside run-to-run variation over seven paired runs of a dense zone;
  what the record adds for certain is one more chunk write per date, the index's, for the Dask
  scheduler, accepted (plan §6).
- A pixel query joins two things: the plane, and `granules` through the snapshot the run's group
  names; `pixel_granules` does it, after picking the run. `shard_dates` answers per-shard
  questions and is off the pixel path.
