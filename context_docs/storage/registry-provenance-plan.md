# Registry provenance: which satellite images went into every embedding

Status: plan, revision 5 (2026-10-27). It has been through two adversarial reviews and has no open
questions. The ingest half is built (§11); revision 5 records what building it changed (§13). It is aimed at the v2 campaign and assumes two pull requests
that are in review: [PR #240](https://github.com/dClimate/tessera-embeddings/pull/240), which
teaches the ingest which catalogue entries are copies of the same image, and
[PR #250](https://github.com/dClimate/tessera-embeddings/pull/250), which decides which radar
image wins where two overlap and keeps each radar image's identifiers. §12 lists what the reviews
changed and §13 the decisions taken since.

## 1. The problem

TESSERA builds one year of embeddings for one UTM zone at a time. The ingest downloads every
Sentinel-2 (optical) and Sentinel-1 (radar) image over the zone for the year and fuses each day's
images into a single zone-wide picture, called a mosaic. The model then reads each pixel's whole
year of mosaic values and writes one embedding per pixel into the published store. Once the
embeddings are committed, the mosaics are deleted, because at 14 PB they cannot be kept.

A reader of the published store will reasonably ask what satellite imagery is behind an
embedding. Today nothing answers that. The ingest knows which images it fused into each day's
mosaic, but only in memory, and the one thing it writes down is the processing baseline of the
day's first image. The model's worker knows which dates each shard used and forgets them once the
shard is staged. The v1.1 store therefore has no provenance beyond its registry, and none can be
reconstructed, because the mosaics it was built from are gone. This record starts with the first
v2 zone-year.

The ingest also stamps a fingerprint of its own code on every mosaic, and a mosaic can only be
resumed by the code that started it. The ingest half of this change moves that fingerprint, so
it has to be deployed while no ingest or update is running, as #240 and #250 do.

The record is for the public global store. It is behind one switch, off by default, so dev runs
and Arbol's own internal stores run without it and a dev store seeded before this change stays
usable (§5.4).

### Words used throughout

| word | meaning |
|---|---|
| zone group | one UTM zone and hemisphere, such as `33N`; the published store has 120 of them |
| shard | one 2048 by 2048 pixel block of a zone group, about 20 km on a side. The store writes a shard at a time, the model processes a shard at a time, and the registry has one row per shard per year. The registry's `tile` column and the code's `chunk_<row>_<col>` label both name a shard |
| chunk | a smaller block inside a shard, the unit of compression |
| granule | one catalogue entry for one image: a Sentinel-2 scene or a Sentinel-1 burst. Where the catalogue holds several copies of one image, #240 picks one |
| solar day | the local calendar day an image belongs to. The ingest groups images by solar day and makes one mosaic slot per day |
| fusion order | the order the day's granules are handed to the mosaic builder. The first granule with data at a pixel supplies that pixel; later ones only fill gaps. Optical goes clearest scene first; radar goes most-often-acquired burst first (#250) |
| run | one fill of one zone-year under one set of inputs, with an id such as `33N-2025-1a2b3c4d`. The first run is the campaign fill; a later targeted update that re-ingests and re-infers part of the zone-year is another run |
| holes update | a targeted update that embeds only the pixels a fill left empty and keeps everything else |
| actor | the worker that runs the model on one shard and stages the result |
| staging | the holding area where each shard waits until the whole zone-year is committed to the published store |
| assembly | the step after the model: commit the staged shards, publish the registry rows, tag the zone-year, delete staging and mosaics |
| registry | the Parquet tables beside the store. `master/` has one row per shard per year describing the store as it is; `parts/` has one file per run |
| snapshot | an Icechunk store version. Each committed mosaic day is a snapshot |

## 2. What the record contains

Four new pieces, all under `provenance/` beside the registry, plus one new column on the
registry's existing rows.

**`granules`: which images went into each day's mosaic.** One row per granule per solar day, in
fusion order, carrying the granule's identifiers, its acquisition time, its cloud cover, its
footprint, its catalogue link and the exact object URLs the ingest read. It is written once per
zone-day, because the mosaic is built zone-wide once per day. A Sentinel-2 scene covers about
thirty shards, so repeating its row for each shard would make the table thirty times larger with
nothing new in it.

**`shard_dates`: what each shard got.** One row per shard, sensor and solar day the model used,
saying how many of the shard's pixels were valid that day, how many the run embedded, and which
of the day's granules supplied at least one pixel of the shard. That last column is the
per-shard granule list, and it is exact: it names the granules that supplied pixels, not every
granule whose footprint touches the shard.

**`pixel_sources`: which image supplied each pixel on each date.** For each pixel and each date,
one small integer. Zero means the model did not see the pixel that day. A positive number k means
it did, and the k-th granule in the shard-date's `sources` list supplied it. This is what makes the
pixel-level answer exact: a reader looks up a pixel's column of numbers and joins each one, through
`shard_dates`, to `granules`. It is stored as a zarr array per sensor per run, in an Icechunk store
of its own.

**`update_pixels`: which pixels a holes update wrote.** A holes update rewrites only some pixels
of a shard. Its mask is kept so a reader knows which run's imagery applies to which pixel.

**`source_runs`, a new registry column.** The list of runs whose pixels a shard currently holds,
oldest first. For most shards it is one run. After a holes update it is two.

```
  <store-stem>.registry/
  ├── README.md
  ├── master/        zone=<Z>/year=<Y>/part-0.parquet       one row per shard per year; gains `source_runs`
  ├── parts/         zone=<Z>/year=<Y>/<run_id>.parquet     one row per shard per run; gains `source_runs`
  └── provenance/
      ├── granules/      zone=<Z>/year=<Y>/<run_id>.parquet one row per granule per solar day per sensor
      ├── shard_dates/   zone=<Z>/year=<Y>/<run_id>.parquet one row per shard, sensor and date a run used
      ├── update_pixels/ zone=<Z>/year=<Y>/<run_id>.zarr    holes updates only: the pixels the run embedded
      └── pixel_sources.icechunk/                           one Icechunk repository with a group per run
          └── <Z>/<Y>/<run_id>/{s2,s1_asc,s1_desc}         (date, northing, easting) uint8: per pixel and
                                                           date, the granule that supplied it
```

Every provenance row carries `zone`, `year` and `run_id` as ordinary columns, as well as in the
folder names, so any Parquet reader can join across zones and runs without parsing paths. A run's
files are written once and never changed; a later run adds its own files beside them.

## 3. What a reader can ask

| about | the answer | from |
|---|---|---|
| a shard-year | the dates each run used for the shard, per sensor, how many pixels were valid on each, and which granules supplied them | `shard_dates`, for each run in the shard's `source_runs` |
| a date | the granules fused into that day's mosaic, in fusion order, with identifiers, cloud cover, footprint, catalogue link and the exact objects read | `granules` |
| a pixel | the dates the model saw it on, per sensor, and the one granule that supplied it on each | `pixel_sources` joined to `granules` |

A worked example. A reader has a pixel near Rome and wants to know what the 2025 embedding was
built from. Rome is in zone group `33N`, and the pixel falls in shard `chunk_120_14`. The reader
opens the registry master row for that shard and year and reads its `source_runs`: one run,
`33N-2025-1a2b3c4d`, the campaign fill. They open that run's `pixel_sources` group, take the
optical array, and read the one column of numbers at the pixel's position. It has 130 entries, one
per solar day on the mosaic's time axis; 48 of them are non-zero. Each non-zero entry is a date
the model actually saw this pixel, and its value points, through the shard's `shard_dates` row
for that date, at a row of `granules`: for 19 June the value is 2, so the second granule in that
row's `sources` list supplied the pixel, and its `granules` row gives its scene id, its 3% cloud cover, its footprint, its catalogue link and the exact object
URLs the ingest read. If the reader only wanted the shard-level picture, `shard_dates` for the
same run lists the same 130 days with the count of valid pixels and the granules that supplied
any part of the shard, with no need to touch the array.

Three things it does not tell a reader, and the registry README should say so.

- How much any date or granule mattered. The model has no attribution; the record says what went
  in.
- The pixel values the model saw. The mosaics are deleted. A reader who wants the imagery
  re-reads the recorded URLs. If a catalogue entry has moved since, the dead URL is itself
  information.
- Anything about v1.1.

For comparison, [OlmoEarth's provenance](https://source.coop/olmoearth/olmoearth-v1-3-embeddings/provenance)
stops at a 2048-pixel window and a 30-day period, lists the candidate scenes rather than the one
that won, and its model mixes neighbouring pixels within a 480-pixel crop. TESSERA's model reads
one pixel at a time, so there is no neighbour effect to account for, and this record names the
granule per pixel and date.

## 4. How it is produced

Three stages already exist. Each gains a small job.

```
  INGEST (per zone-year)                              ASSEMBLY (per zone-year, after the model)
  ──────────────────────                              ─────────────────────────────────────────
  for each day: fuse the granules ─▶ mosaic slot      commit the staged shards to the store
      + the day's granule list     (same commit)       ─▶ publish the registry part (+ source_runs)
      + the per-pixel source index                     ─▶ publish provenance:
                                                              shard_dates, update_pixels and pixel_sources
  MODEL (per shard)                                               from the staged shards
  ────────────────                                              granules from the mosaics, at the
  the actor reads the mosaics, runs the model,                    snapshots the run pinned
      and stages with the shard: the dates it kept,           check everything (§7)
      pixel counts, and the source index masked to            a failure here fails the zone-year
      what the model saw                                  ─▶ merge the master, tag, delete staging and mosaics
```

**The ingest** already builds each day's granule list in memory before fusing. It writes that
list, as JSON, into a `granules` array in the mosaic store, one entry per day, in the same
Icechunk commit as the day's pixels. It also writes a `source_index` band, `uint16`: for each
pixel, the position in the day's fusion order of the granule the fuser took the pixel from, plus
one, or zero where no granule had data. The mosaic builder (odc's fuser) cannot report which
source it used, so one band, SCL for optical and VV for radar, is read through a reader that packs
each granule's position into the high bits of the pixels it returns, zero where the granule has
no data. odc's unchanged fuser fills only where the destination is still zero, so it keeps the
first granule with data as it always has, and splitting the fused result gives back the band,
bit-identical to an unpacked load, and the index. The band is still read once, the index and the
mask share one definition, and nothing is added to the task graph. The other bands are fused by
odc as today.

**The actor** already records, for each shard, how many observations each pixel had. It will also
keep the list of dates it used per sensor, with the count of valid pixels on each, and the count
of pixels it embedded. For `pixel_sources`, it reads its shard's `source_index` at those dates,
sets it to zero wherever its own validity test fails (so the published value means "the model saw
this"), and for a holes update also zeroes everything outside the pixels it was asked to embed.
All of this is staged with the shard, about a megabyte per shard, in the same place the shard's
existing coverage record goes, so every resume path that reads coverage back carries it too.

**Assembly** gains a publication step after the registry part is published: write `shard_dates`
and `update_pixels` from the staged records, write the run's `pixel_sources` group shard by shard
as the shards are committed, export `granules` from the three mosaic stores, and run the checks in
§7. The step is idempotent, so running it twice rewrites the same files.

Three rules govern this.

1. **The granule list is written in the same commit as the pixels.** The ingest writes a day in
   one Icechunk commit, and both of its write paths run inside one session. The list goes into
   that session, so a day that did not commit has no slot and no list, and a day that committed
   has the list its own write carried. A retry, a batch broken up and written one day at a time,
   or a day skipped on its second attempt leaves nothing behind. A resumed ingest skips days
   already on the time axis, so a list is written once and never touched again.
2. **The model and the export read the same mosaic version.** When a fill resumes, the campaign
   already samples each mosaic's latest snapshot id and hashes it into the run id. It does not yet
   pin what the actors read, which open the moving `main` branch. The snapshot ids will be
   threaded into every actor store open and into the export, so a commit to a mosaic during its
   own fill, which nothing in the pipeline makes but an out-of-band repair could, can change
   neither what the model read nor what the record says. This costs nothing at read time: a
   branch open resolves to a snapshot id anyway, and the chunks fetched are the same.
3. **If publication fails, the zone-year fails before anything is deleted.** Publication runs
   under the same retry policy as the store writes. If it still fails, the zone-year is not tagged
   and no cleanup runs, exactly as any other post-commit step fails today. The staging and mosaics
   the next attempt needs are still there, and that attempt republishes. There is deliberately no
   separate marker, gate or held state. The step is three Parquet writes, a small zarr and one
   store group written the way the embeddings are, and the ordinary retry is enough.

Where the code changes:

- Loader (`stac.py`, `ingest/provenance.py`): each collection names its source-index band
  (`source_index_band`: `scl`, `0_VV`). `load_stac_items(source_index=True)` stamps every item it
  loads with its position in its day's fusion order, which is the loader's input order after any
  step-down. `SourceParser` reports that band to odc in a packed dtype with nodata 0 and hands
  each of its sources the position and the band's native dtype and nodata; `_SourceReader` reads
  natively and packs, and a split after the load restores the band and yields `source_index`. The
  BOA offset parser and reader are the same classes, so the offset and the index ride one hook.
  The coverage gate reads the unpacked SCL, so it is still built once.
- Sentinel-2 ingest: each prepared date carries its granule list, built from the positions its
  own load stamped, and both write paths pass it to `write_days_windows`, which writes it in the
  slot's commit. A fallback copy re-prepares the date and so re-lists it; `stepped_down` marks the
  swapped-in copy. The item's catalogue link, pruned at query time, is rebuilt from catalogue URL,
  collection and id. `baselines_applied` stays.
- Sentinel-1 ingest: #250 keeps track, burst, sub-swath and acquisition time on each item and
  sorts each day. Each batch lists its days from the items its load stamped, and each date's write
  carries its own list.
- Actor: `_coverage_record` (`actors.py:215`) gains the per-sensor date lists; the planes are
  written as arrays on the staged shard beside the embeddings; both travel with the `coverage`
  attribute (`assembly.py:1424`) that `read_staged_coverage`, the resumed-shard top-up, the
  coverage-only records for refused shards and the skip records already read. The actors open the
  mosaics through `readonly_session(snapshot_id=…)`, which the code already uses elsewhere.
- Assembly: `publish_provenance` runs after `publish_registry_part` on both the fill and update
  paths. The already-landed retry path needs nothing, since a tagged zone-year has published.
- Switch: a `provenance` field on `IngestSettings` for the campaign flows and a flat parameter
  on the base ingest flows, forwarded beside `allow_ingest_code_mismatch` through
  `tasks/ingest.py` to `s2_roi` and `s1_roi`. `ingest-zone-year` passes it to every leg, to the
  manifest check before marking, and into the completion fingerprint when on. The global
  campaign's deployment in `yield-embeddings` sets it on. The model and assembly half has no
  switch; it follows the mosaics (§5.4).
- Dead code: `inference/shard_record.py` is test-only and is deleted.
- Docs: a Provenance section in `docs/registry.md`; the ingest and inference READMEs say where
  their half is produced; `registry-master.md` gains the publication step;
  `staging-identity-and-resume.md` gains the snapshot pinning; an ADR records §8 and §13.

## 5. The pieces in detail

### 5.1 `granules`

| column | type | meaning | source |
|---|---|---|---|
| `zone`, `year`, `run_id` | string, int32, string | the zone-year and the run that read the mosaic | assembly |
| `sensor` | string | `s2`, `s1_asc` or `s1_desc`, the mosaic store the row describes | the store |
| `solar_day` | date | the day, as on the mosaic's time axis | the slot |
| `fusion_order` | int16 | the granule's position in the day's fusion order, 0 first | the write's item list |
| `granule_id` | string | the catalogue id of the copy read | `item.id` |
| `image_key` | string | which image this is a copy of: S2 `"<datatake>/<datastrip sensing start>"` (#240), S1 the burst id `T<track>-<burst>-<subswath>` (#250) | item properties |
| `platform` | string | `S2A`, `S1C` and so on | `properties["platform"]`; the title for S1 |
| `acquired_at` | timestamp (UTC) | the real acquisition instant | `properties["datetime"]`, which solar-day grouping never overwrites (#250 fixes this for S1) |
| `cloud_cover` | float64 | scene-level cloud percentage; null for radar | `eo:cloud_cover` |
| `processing_baseline` | string | S2 only, for example `05.11` | `item_baselines` |
| `product_uri` | string | ESA SAFE product name; S2 only | `s2:product_uri` |
| `provider`, `collection` | string | `earth-search`/`sentinel-2-l2a`; `cmr-asf`/`OPERA_L2_RTC-S1_V1` | provider config |
| `stac_item_url` | string | the item's public catalogue URL | rebuilt from catalogue URL, collection and id |
| `assets` | map<string,string> | band to the exact URL read, after any bucket rewrite | the item's read set |
| `stepped_down` | bool | this copy replaced one that would not read | the fallback ladder |
| `geometry`, `bbox` | GeoParquet, WGS84 | the footprint as published, split at ±180° | `item.geometry` |

The file's metadata records, once per mosaic store, its snapshot id, the ingest code fingerprint,
the window the ingest assessed, and `provenance_version`. The snapshot id is stored here once and
never on rows.

Inside the mosaic, the list is a one-dimensional `granules` array along `time` with one
variable-length string per slot, which zarr 3.2 supports (verified on the repo's version). JSON is
used because any zarr client can read it; it would only be packed if a test showed a reason to.
Assembly reads the three stores' arrays at the pinned snapshots and writes the Parquet file
without filtering: every slot has a list, and a slot without one fails the checks.

### 5.2 `shard_dates`

| column | type | meaning | source |
|---|---|---|---|
| `zone`, `year`, `run_id` | | as in `granules` | assembly |
| `tile` | string | the shard, `chunk_<row>_<col>` | `ChunkSpec.label` |
| `sensor` | string | `s2`, `s1_asc`, `s1_desc` | |
| `solar_day` | date | a date on that store's time axis the shard kept | the actor's kept indices (`S2MaskBundle.abs_indices`; the SAR `kept` indices) |
| `valid_px` | int32 | pixels of the shard valid on that date, counted over the pixels this run was asked to embed | the actor's mask |
| `embedded_px` | int32 | pixels this run embedded in the shard: the whole live shard for a fill, the holes it filled for a holes update | the actor's refusal accounting |
| `sources` | list<int16> | the `fusion_order` values that supplied at least one pixel of the shard on that date, ascending | the distinct non-zero values of the shard's mosaic `source_index`, less one, over the pixels the model saw |

A refused shard still gets rows, because imagery was seen even though no embedding came of it.
`valid_px` is the non-zero count of the matching `pixel_sources` plane, and the plane's values are
positions in `sources`, so the table and the array cannot disagree.

### 5.3 `pixel_sources`

One Icechunk repository with a group per run, `<zone>/<year>/<run_id>`, holding one array per
sensor of shape `(date, northing, easting)`, dtype uint8, chunked one date by one shard
`(1, 2048, 2048)`, compressed with Blosc zstd and bit shuffle (§9 explains the codec). The `date`
coordinate is the mosaic's time axis at the pinned snapshot; `northing` and `easting` are the
zone group's.

A value k greater than zero means the model received this pixel on this date and the granule with
`fusion_order = sources[k − 1]`, from the shard-date's `shard_dates` row, supplied it. Zero means
the model did not receive it, either because no granule had data there or because the cloud mask
removed it; the footprints in `granules` tell those two apart. The value is a rank within the
shard-date rather than the zone-wide fusion order because a day's fusion order is zone-wide and
can exceed 255, while a shard rarely has more than a handful of contributing granules on one
date; the rank keeps the plane `uint8` and its values small, which is what §9 measured. Only the shards a run evaluated are written, only at the dates they kept, and a
holes update writes only the pixels it was asked to embed, so the store is as sparse as the
embeddings.

Why an integer rather than a bit per pixel: it costs 4% more bytes (§9) and names the winning
granule wherever two footprints overlap, where a bit would leave the reader a list of candidates.
The actor refuses a shard-date with more than 255 contributing granules rather than wrap.
Why a store of its own rather than arrays in the published zone groups: each run has its own list
of dates, whose length differs by zone-year and by run, whereas the published store's groups share
one fixed grid. Keeping it separate also leaves the published store, its metadata conventions and
its checks untouched, and a reader who never wants it never sees it.

### 5.4 The switch, and versioning

The record is for the public global store. Dev runs and Arbol's own internal stores do not need
it, and a dev store seeded before this change has to stay usable: `allow_ingest_code_mismatch`
already lets a run resume such a store across ingest code changes, and a hard version refusal
would have taken that away. So the whole record is behind one switch, `provenance`, off by
default.

The switch is an ingest setting: a field on `IngestSettings` for the campaign flows and a flat
parameter on the single-ROI ingest flows, forwarded to `s2_roi` and `s1_roi` beside
`allow_ingest_code_mismatch`. The global campaign's deployment in `yield-embeddings` sets it on.
With it off the ingest writes no `granules` array and no `source_index` band, and the mosaic is
exactly today's.

The mosaic records the choice. Its ingest manifest gains `provenance_version` (1) when it is seeded
with the switch on, and nothing when off, and the manifest validates it on every write like its
other terms. The term is one whose absence means off (`ABSENT_MEANS_OFF`, the rule the embedding
manifest already applies to `allow_s2_only`). A run with the switch on refuses a store without the
version; such a store, seeded with the switch off or before the field existed, is re-seeded
rather than patched. A run with the switch off refuses a store that has the version, so it can
never leave a granule list short. A store from before the field resumes under off. The seed
creates the `granules` array and the `source_index` band in the same commit as the manifest, and
every date writes both, so a store with the version always has both and no separate check is
needed. The zone-year completion fingerprint carries `provenance: true` when the switch is on, so
a finished mosaic marked without the record is refused by a run that wants it.

The model and assembly half has no switch of its own; it follows the mosaics. The actor stages
planes when its mosaic has `source_index`. Assembly publishes when all three of the zone-year's
mosaics carry the version, publishes nothing when none does, and fails the zone-year when they
disagree, since one dispatch seeds all three and a mix can only be made by hand. A zone-year
written with the switch off has no files under `provenance/` and null `source_runs`, which is
how a reader tells "off" from "nothing seen".

## 6. Targeted updates

A holes update (ADR 027) embeds only the pixels a fill left empty, keeps every other pixel's
embedding and observation counts, and has its registry row re-derived from the merged shard
(`measured_by = "rebuild"`). A shard's imagery is then a composition: the fill's dates for the
pixels the fill wrote, the update's dates for the holes. The record keeps both sides.

- The update publishes its own `shard_dates` file and its own `pixel_sources` group, counted and
  written over the holes it filled. Today the holes path (`zone_fill._assemble_update` and
  `_publish_merged_rows`) publishes no per-shard records, so this is wired explicitly.
- `update_pixels/<run_id>.zarr` holds one boolean array per shard the update wrote, its embed
  mask at the time it ran. An Icechunk diff would name the chunks an update changed but not the
  pixels within them, so the mask is the only exact record. At one bit per pixel it compresses to
  a few kilobytes per shard. Fills publish none, because they wrote every pixel their
  `shard_dates` rows describe.
- `source_runs` is derived from the store by every registry merge and every rebuild, never copied
  from an older row. The rule: the fill's run id from the zone group's `runs[year]` attribute,
  then each entry of `updates[year]` whose `written` list names the shard, in order. A `shards`
  update, which rewrites whole shards, restarts the list at itself. The run that publishes a part
  computes it the same way onto its own rows, so `parts/` describes itself and the newest-run-wins
  merge that rebuilds the master from parts cannot lose it. The registry's own `run_id` column
  cannot play this role, because a rebuild replaces it with a `rebuild-…` id. Rows from before
  the change, all of v1.1, and rows of a zone-year written with the switch off (§5.4), hold
  null.

A reader's shard-year answer is the union over `source_runs`: the per-run rows say which run
contributed which dates, `update_pixels` says which pixels, and each run's `pixel_sources` group
says which granule on which date.

To continue the example from §3: suppose a holes update, run `33N-2025-9e8f7a6b`, later filled
the cloudy corner of shard `chunk_120_14` with imagery from a longer window. The shard's
`source_runs` becomes `[33N-2025-1a2b3c4d, 33N-2025-9e8f7a6b]`. The update's `update_pixels`
array marks the corner's pixels. For a pixel in that corner the reader uses the update run's
`pixel_sources` and `granules`; for every other pixel of the shard they use the fill's, as before.
Each run's files describe only the pixels that run wrote, so neither contradicts the other.

## 7. Checks

Run at publication, where a failure fails the step (§4, rule 3), and again in the registry
census, where a failure names the zone-year for repair.

- Every slot on each mosaic's time axis has a non-empty granule list.
- Every sensor and solar day in `shard_dates` is a slot.
- Every shard in the registry part has `shard_dates` rows for every sensor present, or a stated
  reason such as coverage-only or skipped.
- Every `shard_dates` row's `valid_px` equals the non-zero count of its `pixel_sources` plane,
  the plane's distinct non-zero values are exactly 1 to the length of its `sources` list, and every
  `sources` entry is a position within the day's granule list.
- Every holes-update shard has an `update_pixels` array whose count equals `embedded_px`.
- Every embedded master row's `source_runs` names only runs with published files and groups.
- A zone-year written with the switch off (§5.4) has no files and null `source_runs`; the census
  passes it over.

## 8. Left out, and why

- **The granule behind a cloudy pixel.** The mosaic's `source_index` names it, but only the
  masked planes are published. Nobody has asked which granule a cloud came from, and the mosaic
  holds the answer until it is deleted.
- **A source index per band.** The fuser works per band, so in principle a reflectance band could
  come from a different granule than the mask at the same pixel. That would need a granule with
  data in one band and nodata in another at the same pixel, which the products do not produce.
  The index follows the band the ingest fuses itself (SCL, or VV for radar), and the dev test's
  reflectance check (§11) would catch a counter-example.
- **v1.1.** Its mosaics are gone; it keeps `master/` only.
- **Radar quality layers** such as layover, shadow and number of looks. A separate decision,
  recorded in ADR 033 (#250).
- **The copies the fallback ladder rejected.** They could be added later as rows that touched no
  pixel (`fusion_order = null`) without changing the schema.

## 9. Sizes

**`granules`.** One month of Sentinel-2 items over 12–18°E, 36–60°N (June 2024) came to 2,602
items on 30 solar days, with footprints of 5 vertices at the median and 15 at most. Written as zstd
Parquet in the §5.1 schema that is 121 bytes a row and 0.3 MB for the month; 86 bytes without
geometry and 46 without geometry or asset URLs. Simplifying the polygons saved nothing, so they are
stored as published. That query covered the land latitudes of one zone, not the whole zone, and
the campaign's figure of 368,248 catalogue entries a year came from a wider query that counts
entries before duplicate removal (`campaign-ingest-measurements.md` §12.9). Neither is a true
full-zone count, so the dev test will measure one, but even at 368,248 rows a zone-year is under
50 MB. Radar is the larger table: #250's census counted 100,179 ascending and 115,142 descending
bursts for zone 35N in 2024, about 26 MB a zone-year at the same bytes per row. In all, under
100 MB a zone-year and under 100 GB for the campaign.

**`pixel_sources`.** Measured on one real shard-year, zone 33N, `chunk_209_14`, 2024, with the
Sentinel-2 items in clearest-first order. It had 61 solar days with any valid pixel and 31 valid
observations per pixel at the mean; 56 of the days had one candidate granule and 5 had two.

| encoding of one uint8 plane per date | MB a shard-year | campaign, 3.2 M shard-years |
|---|---|---|
| Blosc zstd level 3 with bit shuffle | **1.17** | **3.7 TB** |
| two bits a pixel, zstd level 3 (would cap candidates at three) | 1.38 | 4.4 TB |
| zstd level 3, no shuffle | 2.83 | 9.1 TB |
| a bare validity bit a pixel, zstd level 3, for comparison | 1.12 | 3.6 TB |

Naming the winner costs 4% over a bare bit, and the bit shuffle is what makes that possible:
without it zstd does not see the mostly constant planes as runs. The campaign total is about 0.2%
of the store, on the order of $1,000 a year. The alternative the integer replaces, guessing the
granule from footprints, named the wrong one for 0.10% of this shard's valid pixel-dates. This is
one temperate optical shard; radar and tropical or arid optical shards are unmeasured, so treat
the figure as good to a factor of two until the dev test measures all three sensors over several
climates. Packing bits along time and XOR-delta planes were also tried and lost to the per-date
plane, which is why the chunking is one date by one shard.

`shard_dates` is about 3,000 shards by around 130 dates by 3 sensors for a dense zone-year,
roughly a million rows and 10 MB, so around 10 GB campaign-wide. `update_pixels` is a few
kilobytes per updated shard. The mosaic's `source_index` band adds two bytes per pixel per date
before compression, about a tenth of a Sentinel-2 mosaic's raw bytes, and compresses to far less
because it is constant over each granule's ground; it exists only as long as the mosaic.

## 10. Example queries

The granules behind one shard on one date, in fusion order, for every run that wrote the shard,
by footprint:

```sql
SELECT g.run_id, g.fusion_order, g.granule_id, g.platform, g.acquired_at, g.cloud_cover, g.stepped_down, g.stac_item_url
FROM granules g, master m
WHERE m.zone = '33N' AND m.year = 2025 AND m.tile = 'chunk_120_14'
  AND g.zone = m.zone AND g.year = m.year AND list_contains(m.source_runs, g.run_id)
  AND g.sensor = 's2' AND g.solar_day = DATE '2025-06-19'
  AND ST_Intersects(g.geometry, ST_MakeEnvelope(m.bbox_west, m.bbox_south, m.bbox_east, m.bbox_north))
ORDER BY g.run_id, g.fusion_order;
```

The granules that actually supplied pixels of one shard on each date it kept, with no geometry:

```sql
SELECT d.solar_day, g.fusion_order, g.granule_id, g.platform, g.cloud_cover, g.stac_item_url
FROM shard_dates d, UNNEST(d.sources) AS s(fusion_order)
JOIN granules g USING (zone, year, run_id, sensor, fusion_order)
WHERE d.zone = '33N' AND d.year = 2025 AND d.run_id = '33N-2025-1a2b3c4d'
  AND d.tile = 'chunk_120_14' AND d.sensor = 's2' AND g.solar_day = d.solar_day
ORDER BY d.solar_day, g.fusion_order;
```

One pixel: its column of `pixel_sources` gives one number per date on the run's time axis, and
each non-zero number is a row of `granules`.

```python
run = pixel_sources[f"{zone}/{year}/{run_id}"]          # the run's group, opened read-only
k = run["s2"][:, northing_index, easting_index]         # one uint8 per date
seen = k > 0
picks = pl.DataFrame({"solar_day": run["date"][:][seen], "rank": k[seen] - 1})
rows = shard_dates.filter(zone=zone, year=year, run_id=run_id, tile=shard, sensor="s2")
picks = picks.join(rows, on="solar_day").with_columns(
    fusion_order=pl.col("sources").list.get(pl.col("rank")))
picks.join(granules.filter(zone=zone, year=year, run_id=run_id, sensor="s2"),
           on=["solar_day", "fusion_order"])             # one granule per date the model saw
```

A shard with several `source_runs` is read once per run, and `update_pixels` says which run's
numbers apply to the pixel. All three queries will be run against a dev dataset holding several
zones, years and an update run before the registry README publishes them.

## 11. Sequence and effort

1. #240 and #250 land.
2. The ingest half ships as one stack on #250: the switch, the granule list, the source index,
   `provenance_version` and the fingerprint note, for both sensors. Built: real-raster tests pin
   that the fused band is bit-identical with the index on, for an integer band, for Sentinel-2
   through the BOA-correcting parser, and for radar through the dB transform.
3. The model and assembly half ships as a second stack: the actor's date lists, snapshot pinning,
   the embed mask and the masked planes; then publication, `pixel_sources`, the checks,
   `source_runs` on parts and master, and the docs. The registry schema changes and the
   `_common_metadata` rewrite go with it.
4. A dev test on the standard chain (tiny, then Iowa, then 35N) asserts: every slot has a granule
   list; every recorded asset URL returns 200 in-region; `valid_px` equals the mask-valid count in
   the mosaic for a sampled shard-date; a resumed zone-year publishes tables identical to an
   uninterrupted one; a zone-year interrupted between the shard commit and publication is left
   untagged with its staging and mosaics intact, and its retry publishes, tags and cleans; a
   commit to a mosaic mid-fill that leaves the time axis unchanged changes nothing the actors read
   or the export records; a holes update on a filled zone-year adds a second `shard_dates` file and
   an `update_pixels` array, `source_runs` recomputes to both runs after a second master merge and
   after a rebuild, and a `shards` update resets it; and for a sampled shard-date, the plane's
   non-zero count equals `valid_px`, its distinct values are 1 to the length of `sources`, every
   pixel with value k lies inside the footprint of the granule at fusion order `sources[k − 1]`,
   and the mosaic's reflectance at a sample of those pixels equals that granule's own. The same
   run measures the retained `granules` rows for a full zone-year per sensor, the `pixel_sources`
   planes for all three sensors, and the ingest's time per day against #250, which should be level
   since no band is read twice. The chain runs with the switch on. A tiny leg then runs with it off and asserts
   that the mosaics, staging and registry are today's, that a store seeded before the change
   resumes under it, and that a run with the switch on refuses that store with a reason.
5. It ships with the first v2 zone-year, and `yield-embeddings` refreshes its lock after each
   merge.

The estimate is four to five weeks with no infrastructure change. Revision 1 said two to three.
Revision 3 added the snapshot threading through the actors, the update composition and the
derived `source_runs`; revision 4 added the source index in both loaders, the plane staging and
the `pixel_sources` store.

## 12. What the reviews changed

| finding | before | now |
|---|---|---|
| a Parquet file written before the commit can name the wrong granules for a committed day | a file beside the mosaic, filtered by the time axis | an array in the mosaic store, appended in the day's own commit, one session for both write paths (§4 rule 1) |
| a holes update cannot describe a shard with one run's dates | one file per run | per-run files, `update_pixels` and `source_runs` (§6) |
| the master's `run_id` is replaced by a rebuild id, and a master-only column is dropped when the master is rebuilt from parts | lookup by `run_id`, then `source_runs` on the master only | `source_runs` derived from the store at every merge and rebuild, and written on the part rows (§6) |
| fail-soft publication followed by cleanup is irreparable, and a mosaic-only gate still lost the staging that holds the only per-shard records | log and continue; then a mosaic-only gate; then a marker gating staging, mosaics and the sweep | no gate: publication is a landing step that fails the zone-year before tag and cleanup, and the retry republishes (§4 rule 3) |
| stores from before the change would fail to resume indefinitely | unaddressed | a `provenance` switch, off by default; the mosaic records it as `provenance_version`, and resume refuses a mismatch with a reason (§5.4) |
| the snapshot id is sampled at resume but not pinned at read | "pinned" | snapshot ids threaded into the actors and the export (§4 rule 2) |
| footprint order does not identify the winner, and the fallback ladder leaves the order unsorted | "exact except near edges" | fusion order is the loader's input after step-down, and the winner is named per pixel (§5.3) |
| `run_id` was only a filename and file metadata, so the documented joins could not run, and the queries lacked zone and year | | `zone`, `year` and `run_id` columns on every row; queries rewritten (§2, §10) |
| sizes were extrapolated | 368k rows, a 20–45 TB bitmask | measured (§9); the per-pixel record taken up (§13) |

## 13. Decisions since the reviews

- **The per-pixel record is in** (2026-10-26). It was ruled out in revision 1 on an estimate of
  20–45 TB for a bit per pixel per date. Measured, the bit is 3.6 TB and the integer that names
  the winner is 3.7 TB (§9), so it was taken up as the integer: for 4% more it is exact where
  footprints overlap and a reader needs no geometry. It lives in a store of its own beside the
  registry rather than as arrays in the published zone groups (§5.3).
- **Granule rows stay per zone-day; the per-shard list is a column of `shard_dates`** derived
  from `pixel_sources` (§5.2), so it names the granules that supplied pixels rather than those
  whose footprints overlap.
- **Publication failure fails the zone-year and is retried.** No gate, marker or held state
  (§4 rule 3).
- **Snapshot pinning in the actors is in scope**, at no read-time cost (§4 rule 2).
- **The mask band is fused by the ingest from per-granule slices, not read twice** (2026-10-26).
  The second read would have cost about 1% of a day's bytes, so the reason is that one read gives
  one definition shared by the index and the mask, and no measurement step.
- **The record is behind a switch, off by default** (2026-10-26). It is for the public global
  store; dev runs and Arbol's internal stores do not need it, and a hard version refusal would
  have stranded the dev stores that `allow_ingest_code_mismatch` keeps usable today. The ingest
  owns the switch, the mosaics carry it, and the model and assembly half follows the mosaics
  (§5.4).
- **The index rides on the reader, not on a per-granule load** (2026-10-27, building the ingest
  half). Loading the mask band per granule (`groupby="id"`) makes one load task per granule per
  chunk: tens of granules a day for optical, hundreds of bursts for radar, each a full chunk of
  memory, where today there is one task per chunk. Packing the position into the pixels on the
  read keeps one task per chunk and lets odc's own fuser pick the winner. odc mishandles a
  per-band dtype override when chunks are given, so the packed dtype is declared through the
  parser's band metadata instead.
- **The mosaic's index is `uint16`; the published plane stays `uint8` as a rank into the
  shard-date's `sources`** (2026-10-27). A day's fusion order is zone-wide, and a long land zone
  can have more than 255 scenes or bursts in a day, so the mosaic needs sixteen bits. A shard has
  a handful of contributing granules on a date, and §9's measurement was taken on per-shard ranks,
  so ranking within `sources` keeps the measured size and the `uint8` plane, at the cost of one
  more join for a pixel query (§10).
- **Vocabulary:** shard, never tile, for the 2048-pixel unit; `granules`, never `scenes`, for the
  table.
