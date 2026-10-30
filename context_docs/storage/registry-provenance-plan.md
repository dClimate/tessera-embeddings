# Recording which satellite images went into every TESSERA embedding

**Status.** Built behind a switch that is off by default, and tested end to end on the dev
deployment on 2026-10-27 ([`provenance-dev-verification.md`](provenance-dev-verification.md)).
Not merged: it is three draft pull requests, [#253](https://github.com/dClimate/tessera-embeddings/pull/253)
(ingest), [#255](https://github.com/dClimate/tessera-embeddings/pull/255) (model and publication)
and [#256](https://github.com/dClimate/tessera-embeddings/pull/256) (one registry column and the
reader helpers), stacked on two in review, [#240](https://github.com/dClimate/tessera-embeddings/pull/240)
and [#250](https://github.com/dClimate/tessera-embeddings/pull/250). It is meant to start with the
first zone-year of the v2 campaign. The decisions are recorded in
[ADR 035](../decisions/035-a-provenance-record-for-the-published-store.md).

**What we would like from a reviewer.** Whether the record answers the questions users will ask
(§2), whether its costs are worth paying (§6), and views on the choices in §8 and the open
questions in §10. The appendices are reference material and can be skipped.

## Summary

- **The gap.** An embedding is built from a year of satellite images, but the images are fused
  into daily mosaics and the mosaics are deleted once the embeddings are written. Today nobody can
  say which images went into a given embedding, and for the v1.1 store nobody ever will.
- **The record.** For every pixel, and every date on which its observation was valid for the
  model, the record names the one satellite image (the *granule*) that supplied that pixel, with
  the granule's id, acquisition time, cloud cover, footprint and the exact file URLs that were read.
  It sits beside the store's existing registry.
- **Three questions it answers,** each with a reader helper: what went into this day's mosaic,
  into this 20 km shard, and which granule supplied this pixel on each date?
- **Cost.** About 4.7 TB of storage over the campaign, a quarter of a percent of the store. Its
  ingest time cost is inside run-to-run variation, measured over seven paired runs of a dense zone
  (§6); what it certainly adds is one more chunk write per date for the Dask scheduler.
  Correctness checks run at publication and fail the zone-year if the record is wrong.
- **Scope.** Only the public global store needs it. Internal and dev stores leave it off and are
  unchanged.

## Words used

| word | meaning |
|---|---|
| zone | one UTM zone and hemisphere, such as `33N`. The store is filled one zone-year at a time |
| shard | a 2048 × 2048 pixel block of a zone, about 20 km on a side. The model works a shard at a time, and the registry has one row per shard per year. For historical reasons a shard is named `chunk_<row>_<col>` and the registry's column for it is `tile`; both mean shard |
| sensor | `s2` (Sentinel-2, optical), `s1_asc` and `s1_desc` (Sentinel-1 radar, ascending and descending orbits) |
| granule | one catalogue entry for one image: a Sentinel-2 scene (optical) or a Sentinel-1 burst (radar) |
| mosaic | the zone-wide picture the ingest builds for one sensor on one day by fusing that day's granules. It is deleted when the zone-year is published |
| mosaic snapshot | the version of a mosaic a run read, an Icechunk snapshot id. A run that ingests builds its own mosaics, so it reads its own snapshots |
| solar day | the local calendar day a granule belongs to; the mosaic has one slot per solar day |
| fusion order | the order a day's granules are handed to the mosaic builder: optical clearest first, radar most-often-acquired burst first (#250). Where granules overlap, the first one with data supplies the pixel |
| valid for the model | an observation that passes the model's own test: a clear scene class for optical, data in VV or VH for radar |
| refused | a pixel a run was asked to embed but did not, for example because it had fewer clear optical observations than the minimum. A refused pixel has no embedding |
| run | one pass that writes pixels into a zone-year: the first fill; a *holes update*, which re-runs the model on refused pixels only; or a *shards update*, which rewrites whole shards |
| registry | the Parquet tables published beside the store, one row per shard per year, saying what each shard holds |

## 1. Why this is needed

```text
                     INGEST                       MODEL                    PUBLISHED STORE
  Sentinel-2 ──┐   for each day, fuse the     for each pixel, read its     one embedding
  scenes       ├─▶ day's granules into one ─▶ year of mosaic values,   ─▶  per pixel, plus
  Sentinel-1 ──┘   mosaic per sensor          drop invalid dates, embed    the registry
  bursts
  (across one zone, about 1,000 optical granules and 900–2,600 radar bursts per orbit, every day)

                   mosaics are deleted after publication: about 14 PB for the campaign
```

A user looking at an embedding will reasonably ask what imagery is behind it: which dates, from
which satellite, how cloudy the scene was, and where to fetch the original files. The ingest knows
which granules it fused each day, and the model which dates were valid for each pixel, but neither
writes it down, and once the mosaics are deleted the answer is gone, as it is for the published
v1.1 store. This plan closes the gap from the first v2 zone-year onward.

## 2. What a reader can find out

| question | answer | helper | read from |
|---|---|---|---|
| What went into one day's mosaic? | every granule fused that day, in fusion order, with ids, acquisition time, cloud cover, footprint, catalogue link and file URLs | `day_granules` | `granules` |
| What went into one shard? | per sensor, the dates on which any of its pixels had a valid observation, how many did, and which granules supplied them | `shard_granules` | `shard_dates`, then `granules` |
| What is behind one pixel? | per sensor, every date its observation was valid for the model, and the one granule that supplied it on each | `pixel_granules` | `pixel_sources`, then `shard_dates` and `granules` |

The helpers are in `storage/provenance_reader.py` and return polars tables (Appendix B).

**A worked example, with real values from the dev test.** The pixel at 45.5071°N, 14.3761°E, on
the Slovenia–Croatia border, as `pixel_granules` finds it:

```text
  45.5071°N 14.3761°E ──▶ zone 33N, shard chunk_209_14, row 1908, column 70
  registry row, 2024:    source_runs = [fill 33N-2024-9a1fc94c, update 33N-2024-e892e33b]
  newest run first:      the update's update_pixels marks the pixel, so the update's record answers (§5)
  that record's planes:  one small number per date, non-zero on 27 optical, 61 ascending and 28
                         descending radar dates
```

Each non-zero number leads to one granule. The dev test traced those steps on a second pixel of
the same shard, row 1822, column 96 (45.5149°N, 14.3794°E), which the fill embedded:

```text
  pixel_sources (fill, s2), 2024-01-20:  1     (a 0 would mean the observation was not valid for
        │                                       the model: no granule had data, or cloud masked it)
        ▼
  shard_dates (fill, chunk_209_14, s2, 2024-01-20):  sources = [0], so 1 means fusion order 0
        ▼
  granules (the snapshot that row names, 2024-01-20, fusion order 0):
        S2B_33TVL_20240120_0_L2A · Sentinel-2B · 10:08 UTC · 14.6% cloud,
        its catalogue link, footprint, and the URLs of the 11 band files read
```

That pixel has 40 such dates for optical. Radar shows why the extra step through `sources`
exists. On 1 January its ascending number is 2 and the shard's `sources` is `[0, 2, 4]`, so it came
from fusion order 2, burst `T044-092628-IW3`: three bursts covered parts of the shard that day, and
the record names the one that covered this pixel.

**How this compares.** OlmoEarth publishes provenance per 2048-pixel window and 30-day period,
listing the candidate scenes rather than the one used, and its model mixes neighbouring pixels.
TESSERA's model reads one pixel at a time, so a per-pixel answer is exact.

## 3. What gets recorded

Four new pieces under `provenance/` beside the registry, and one new registry column.

```text
  <store>.registry/
  ├── master/  parts/          the existing registry; both gain a `source_runs` column
  └── provenance/
      ├── granules/      zone=33N/year=2024/<sensor>-<snapshot>.parquet   per mosaic snapshot: its granules
      ├── shard_dates/   zone=33N/year=2024/<run>.parquet    per run: one row per shard, sensor and date
      ├── update_pixels/ zone=33N/year=2024/<run>.zarr       holes updates only: the pixels they embedded
      └── pixel_sources.icechunk/  33N/2024/<run>/{s2, s1_asc, s1_desc}
                                   per run: one small integer per pixel per date
```

The Parquet folders use hive `key=value` names because Parquet readers turn them into columns;
the Icechunk groups use plain names because Zarr readers do not. `parts/` is to be renamed `runs/`
in a separate pull request after this stack.

**`granules`: what went into each day's mosaic.** One row per granule per solar day, per zone
rather than per shard because one Sentinel-2 scene covers about thirty shards. A list belongs to
the mosaic it was fused into, so the file is keyed by mosaic snapshot and written once, by the first
run that reads the snapshot; a later run reading the same snapshot, such as a retry, checks it
instead. A run that re-ingests builds new mosaics and writes new files. A real row, from the
fill's optical snapshot: 2 January 2024, fusion order 0, `S2A_33TVL_20240102_0_L2A`, Sentinel-2A,
09:58:07 UTC, 94.9% cloud, with its catalogue link, its footprint as GeoParquet, and the URL of
each of the 11 files read, such as `…/S2A_33TVL_20240102_0_L2A/B04.tif` (Appendix A).

**`shard_dates`: what went into each shard.** Per run, one row per shard, sensor and date on which
at least one of the shard's pixels had a valid observation:

| run_id | tile | sensor | snapshot_id | solar_day | valid_px | embedded_px | sources |
|---|---|---|---|---|---|---|---|
| 33N-2024-9a1fc94c | chunk_209_14 | s1_desc | the fill's descending snapshot | 2024-01-12 | 4,194,294 | 15,668 | [0, 1, 2] |

`valid_px` counts the shard's pixels with a valid observation that date; `embedded_px` the pixels
the run embedded in the whole shard, repeated on each of its rows; `sources` the fusion orders of
the granules that supplied at least one pixel, so footprints that merely touch the shard are not in
it. `snapshot_id` names the mosaic the run read, and is how a row finds its granules.

**`pixel_sources`: which granule supplied each pixel.** Per run and sensor, an array of one byte
per pixel per date. Zero means the observation was not valid for the model that date; k means the
k-th entry of that shard-date's `sources`. The planes compress to about a megabyte per shard-year
(§6). They live in an Icechunk store of their own because each run has its own dates, while the
published store's arrays all share one grid.

**`update_pixels`, and which pixels the record covers.** A fill or a shards update records every
pixel it was asked to embed, refused ones included, so a refused shard still has rows and a reader
can see what imagery a refused pixel had (§8). A holes update records only the pixels it embedded:
its `update_pixels` marks them, one boolean mask per shard, and its planes are cut to the same
mask. Every other pixel keeps the record of the run before it (§5). Whether a pixel has an
embedding is in the store and the registry.

**`source_runs`: which runs a shard's pixels came from.** A new registry column listing them
oldest first: the fill, then each update that wrote or cleared the shard, in order. A holes update
appends itself; a shards update restarts the list at itself. It is null unless every run in the
list carries a record, and it is recomputed from the store every time the registry is merged or
rebuilt, never copied from an old row.

## 4. How it is made

The pipeline already has three stages; each gains one job.

```text
  INGEST, per zone-year                 MODEL, per shard                 ASSEMBLY, per zone-year
  ─────────────────────                 ────────────────                 ───────────────────────
  for each day and sensor:              read the mosaics, run the       commit the shards to the store
   fuse the granules into the mosaic     model, stage the shard          publish the registry rows
   + NEW: write the day's granule list   + NEW: stage, per date, the   + NEW: publish the record,
   + NEW: write a source_index band        index, zeroed where the         check it, and fail the
     (which granule supplied each          observation was not valid       zone-year if it is wrong
     pixel), in the same commit            for the model, with counts    then tag, delete staging and mosaics
```

**How the ingest knows which granule supplied a pixel.** The mosaic builder, odc's fuser, takes
the day's granules in fusion order and keeps, at each pixel, the first one that has data:

```text
  pixels along one row            1   2   3   4   5   6   7   8
  granule, fusion order 0         A   A   A   ·   ·   ·   ·   ·      · = no data at this pixel
  granule, fusion order 1         ·   ·   B   B   B   B   ·   ·
  granule, fusion order 2         ·   ·   ·   ·   ·   C   C   ·
  ───────────────────────────────────────────────────────────
  mosaic (first with data)        A   A   A   B   B   B   C   ·
  source_index (order + 1)        1   1   1   2   2   2   3   0
```

The fuser does not report which granule it used, so one band per sensor, the scene
classification (SCL) for optical and VV for radar, is read through a reader that writes the
granule's position into the high bits of every pixel it returns. The fuser runs unchanged and
still keeps the first granule with data; afterwards the value splits back into the band, bit for
bit what a plain read gives, and the index. No band is read twice and the fuse is unchanged; the
index is one more array to write. The model's stage zeroes the index wherever its own cloud and
validity test rejects the observation, so a published non-zero means the observation was valid for
the model, and renumbers each shard-date's granules 1, 2, 3 so a byte suffices (a day's zone-wide
fusion order runs past 255; a shard rarely has more than a handful on one date).

**Three rules keep the record and the embeddings consistent.**

1. *The granule list is written in the same commit as the day's pixels.* A day that did not
   commit has neither, so the list can never describe a day the mosaic lacks.
2. *The model and the record read the same version of the mosaics.* The fill notes each mosaic's
   snapshot when it plans and every later read uses it, so a change to a mosaic during a fill
   changes neither the embeddings nor the record, which names those snapshots.
3. *If publishing the record fails, the zone-year fails, before anything is deleted.* The staging
   and mosaics a retry needs are still there, and the retry publishes the same record.

**The switch.** One setting, `provenance`, off by default, is set in the ingest. Each mosaic
records whether it was built with the record, and a run that disagrees with an existing mosaic is
refused with a reason rather than producing a half-recorded year. The model and publication
follow the mosaics. A zone-year built with the switch off has no `provenance/` files and a null
`source_runs`, which is how a reader tells "not recorded" from "nothing valid". **A known danger:**
a mosaic ingested with the switch off is unrecorded for good, since a run that wants the record
refuses it and only a re-ingest replaces it, so the campaign's deployment must turn the switch on
before its first ingest.

## 5. Updates

A *holes update* re-runs the model only on refused pixels, often after lowering the minimum number
of clear observations, and leaves every other pixel untouched. It re-ingests the year, so it reads
mosaics of its own, and a shard's pixels then come from two runs. The dev test did exactly this:
its fill required a minimum of 40 clear optical observations, so in shard `chunk_209_14` it
embedded 15,668 pixels and refused the rest; an update with a minimum of one embedded the remaining
4,178,636.

```text
  shard chunk_209_14, 2024              source_runs = [33N-2024-9a1fc94c, 33N-2024-e892e33b]
  ┌──────────────────────────────┐
  │ ░ embedded by the fill       │   15,668 px    → not in the update's update_pixels;
  │                              │                  read the fill's record
  │ ▓ embedded by the update     │   4,178,636 px → marked in the update's update_pixels;
  │   (refused by the fill)      │                  read the update's record
  └──────────────────────────────┘
```

§2's worked pixel, row 1908, column 70, had 27 clear optical dates: too few for the fill, enough
for the update, whose `update_pixels` marks it and whose record names its dates and granules.

**The reader's rule, for any number of runs.** Walk the shard's `source_runs` from the newest: a
holes update claims a pixel only where its `update_pixels` marks it, and a fill or a shards update
claims every pixel. The first run that claims the pixel holds its record; the helpers apply this.

## 6. What it costs

**Storage,** measured on the dev test's shards and on one week of a whole zone:

| piece | measured | campaign |
|---|---|---|
| `pixel_sources` | optical 1.1–1.2 MB a shard-year; radar 0.07–0.2 MB per orbit | about 4.7 TB, a quarter of a percent of the store, around $1,300 a year |
| `granules` | about 160 MB for a land-dense zone-year (zone 35N) | at most about 170 GB for the fills, plus a set per update that re-ingests |
| `shard_dates`, `update_pixels` | about 10 MB a zone-year; kilobytes per updated shard | about 10 GB |
| the granule lists and index inside the mosaics | about 150 MB and 0.1–0.3% of a mosaic | none: deleted with the mosaics |

The ~150 MB inside the mosaics and the ~160 MB `granules` table are the same lists in two
encodings; the mosaics' copy is deleted at publication. The planes are chunked one date by one
shard, so reading one pixel's year fetches one object per date, 40–130 per sensor (§10).

**Ingest time.** Measured on zone 35N, 1–7 January 2024, over two days, each pair's two arms run
side by side on their own clusters. Per-date write time with the record on, divided by off:

| pairs | on ÷ off | workers | CPU |
|---|---|---|---|
| optical, three pairs at 60 workers | 1.12, 1.01, 0.87 | 60 | 30–48% |
| radar ascending, four pairs at 13 workers | 1.31, 1.30, 0.87, 1.37 | 13 | 31–42% |

The same arm, same code, varied as much between runs as any pair did between its arms: the plain
radar ingest took 230 s one run and 165 s the next, a 28% swing, and the dense optical ingest ran
level in one pair and 13% faster with the record on in another. Workers in every arm sat at
30–48% CPU, so the ingest is bound by its reads, not by the record's arithmetic, which a laptop
puts at about 40 ms per 4,096² chunk against about 1.4 s to load it. The record's ingest time
cost is therefore not distinguishable from zero at this fleet; the first measurement, one radar
pair at +30%, was one draw from that spread.

What the record does add, every time: one more chunk write per date, the index's, which is one more
task for the Dask scheduler per chunk-date (+8% tasks on an optical graph of 12 bands, +20% on a
radar graph of two), and more CPU on workers that have it to spare. That is accepted: it costs no
ingest time, and if the scheduler ever becomes the limit the lever is fewer, larger chunks for the
index, not a cheaper index. Dropping the planes would not escape it, since `sources` is derived
from the per-pixel index. Writing the index inside its band's task was built and is correct, but
it does not reduce the task count on the cluster:
distributed Dask never fuses a two-input task with its parents, so the pairing step only moved the
boundary. It is not shipped. Fusing the write graph before the compute, which distributed Dask
otherwise skips, folds the index's cast into its write and is shipped: 7% fewer tasks on a recorded
radar graph, none on a plain one. Measurements: [`provenance-dev-verification.md`](provenance-dev-verification.md).

**Publication** took about 16 seconds for two shards, including the registry part. It copies
planes eight shards at a time, for three sensors; if a batch of eight takes as long as that pair, a
zone-year of about 3,000 shards takes of the order of an hour or two, a risk to examine separately.

## 7. How it was tested

| what | result |
|---|---|
| bands built with the switch on against off | bit-identical |
| the published record against the mosaics, sampled shard-dates and pixels | every plane matched; every sampled optical pixel equals the named granule's own file |
| a run whose switch disagrees with an existing mosaic | refused with a reason, both ways |
| a fill killed between its commit and publication, then retried | the retry's record equals an uninterrupted one |
| a mosaic changed mid-fill | the record and embeddings read the snapshot planned |
| a holes update, a registry rebuild and a shards update | each run's record correct; `source_runs` follows |

The dev test ran an earlier layout: a `granules` file per run, and an `update_pixels` of the
pixels a holes update was asked to fill. Today's layout and meaning have unit tests and wait for
the next dev run. Numbers: [`provenance-dev-verification.md`](provenance-dev-verification.md).

## 8. Choices worth questioning

The last column is the first draft's choice, where reviews of the plan or the build changed it.

| choice | alternative | why this one | first draft |
|---|---|---|---|
| a granule per pixel per date | a yes/no bit per pixel per date, with the granule guessed from footprints | 4% more storage, and exact: guessing named the wrong granule for 0.10% of one shard's pixel-dates, wherever footprints overlap | nothing per pixel, ruled out at an estimated 20–45 TB (measured: 3.7 TB for optical); the granule guessed from footprint order |
| a per-shard granule list derived from the pixels | every granule whose footprint touches the shard | names only granules that supplied a pixel | |
| the index written by the ingest while it fuses | reading the mask band a second time, or once per granule | one read, one definition of "valid"; per granule would multiply the load by the day's granules, hundreds of radar bursts | each granule loaded separately |
| the granule list inside the mosaic, in the day's own commit | a file written beside the mosaic before the commit | a file written first can describe a day that never committed | the file beside |
| `granules` keyed by mosaic snapshot, written once | a `granules` file per run | the list belongs to the mosaic: one answer per mosaic, and a reader joining through `snapshot_id` never counts a list twice | a file per run |
| a record per run, with `update_pixels` and `source_runs` | one record per shard-year | a holes update mixes two runs in one shard | one per shard-year |
| `update_pixels` marks what a holes update embedded, and its planes are cut to it | the pixels it was asked to embed | one meaning: exactly where a reader takes the update's record; its refused pixels keep the earlier run's record, which covers them | the pixels asked for |
| a fill or shards update records refused pixels too | only embedded pixels | a refused shard still says what imagery it had; the store says what was embedded | |
| `source_runs` recomputed from the store at every merge and rebuild, on part rows too | kept on the master only | a rebuild replaces the master's run id and rebuilds the master from parts | the master only |
| a separate Icechunk store for the planes | arrays inside the published store | each run has its own dates; the published store and its conventions stay untouched | |
| failure to publish fails the zone-year | publish best-effort and log | a zone-year can never be published without its record, and the retry repairs it | best-effort, then a gate |
| one switch, off by default, recorded in the mosaic, mismatches refused | always on | internal and dev stores neither need it nor pay its ingest cost, and dev stores from before the change stay usable | always on, no ingest cost expected |

## 9. Left out

- **Attribution.** The model has none, so the record says what went in, not how much it weighed.
- **The pixel values.** The mosaics are gone; a reader who wants the imagery re-reads the URLs.
- **The granule behind an invalid observation,** a cloudy pixel for instance. The mosaic knows,
  but only observations valid for the model are published.
- **A separate index per band.** The index follows the band the validity test reads. A granule with
  data in one band and none in another at the same pixel would break this; the products do not
  produce one, and the dev test's pixel checks found none.
- **v1.1**, whose mosaics are gone.
- **Radar quality layers** such as layover and shadow, a separate decision (ADR 033).
- **Copies of an image the ingest rejected** after a failed read. They could be added later as
  rows that supplied no pixel, without changing the schema.

## 10. Open questions

1. **Plane chunking.** A one-pixel read fetches 40–130 objects per sensor. The next dev run will
   compare chunks of all dates × 256 × 256 and all dates × 128 × 128 against the current one date
   × one shard on a real shard-year, for size and for the time to read one pixel.
2. **Rank or fusion order in the plane.** The rank (plane value k → `sources[k − 1]` → fusion
   order) exists to fit a byte. Storing the fusion order itself as `uint16` with bit-shuffled zstd
   would remove `sources` and one hop; the next dev run will measure its size on a real shard-year.
3. **Publication time at zone scale,** of the order of an hour or two per zone-year if it grows
   linearly (§6), to be examined separately.
4. **Radar file URLs need Earthdata credentials to fetch.** Is it enough to record them, or should
   the record also point to a public mirror where one exists?
5. **Is anything missing** that a user would need to trust or reproduce an embedding?

## Appendix A. Table schemas

`zone` and `year` are folder names (hive partitions), not stored in the files, since pyarrow
refuses a dataset whose partition keys are also columns. The mosaics are deleted after publication,
so a `snapshot_id` is a join key and a value for the publication checks, not a way to open one.

**`granules`**, one file per sensor and mosaic snapshot; its metadata names the zone, year,
sensor, snapshot and the run that first published it.

| column | type | meaning |
|---|---|---|
| `snapshot_id` | string | the mosaic snapshot the lists were read from |
| `sensor` | string | `s2`, `s1_asc` or `s1_desc` |
| `solar_day` | date | the mosaic slot |
| `fusion_order` | int16 | position in the day's fusion order, 0 first |
| `granule_id` | string | the catalogue id of the copy read |
| `image_key` | string | which image it is a copy of: Sentinel-2 datatake and datastrip (#240), Sentinel-1 burst id (#250) |
| `platform` | string | `S2A`, `S1C` and so on |
| `acquired_at` | timestamp, UTC | the acquisition instant |
| `cloud_cover`, `processing_baseline`, `product_uri` | float64, string, string | Sentinel-2 only |
| `provider`, `collection`, `stac_item_url` | string | where the entry came from, and its catalogue link |
| `assets` | map<string, string> | band to the exact URL read |
| `stepped_down` | bool | this copy replaced one that would not read |
| `geometry`, `bbox_west` … `bbox_north` | WKB (GeoParquet 1.1), float64 | the footprint as published, WGS84 |

**`shard_dates`**, one file per run: `run_id`, `tile`, `sensor`, `snapshot_id`, `solar_day`,
`valid_px` (int32), `embedded_px` (int32, per shard, repeated on each of its rows) and `sources`
(list of int16 fusion orders, ascending). Its metadata names the zone, year, run, the snapshot of
each mosaic read, and the refused shards that staged no record.

**`pixel_sources`**: one group per run, `<zone>/<year>/<run_id>`, holding per sensor a
`(date, northing, easting)` uint8 array on the zone's grid, chunked one date by one shard, and a
`<sensor>_date` array of its dates; the group records the grid shape and shard size. The codec was
chosen on one real optical shard-year (33N `chunk_209_14`, 2024): Blosc zstd level 3 with bit
shuffle, 1.17 MB, against 1.38 MB for two bits a pixel (which would cap a shard-date at three
granules), 2.83 MB for zstd without the shuffle, which then misses the mostly constant planes' runs,
and 1.12 MB for a bare validity bit. Bits packed along time and XOR-delta planes also lost.

**`update_pixels`**: one boolean `(2048, 2048)` array per shard, named for the shard: the pixels
the holes update embedded.

## Appendix B. Reading the record

The helpers in `storage/provenance_reader.py` answer §2's questions and apply §5's rule; OlmoEarth's
documented helpers for its own provenance are the bar they are meant to clear. `REGISTRY` is the
registry's root, a local path or an `s3://` URI.

```python
import datetime as dt
from tessera_embeddings.storage.provenance_reader import day_granules, locate, pixel_granules, shard_granules

FILL, UPDATE = "33N-2024-9a1fc94c", "33N-2024-e892e33b"

# A day: the optical granules of 9 May 2024 in the mosaic the fill read, in fusion order.
day_granules(REGISTRY, "33N", 2024, "s2", dt.date(2024, 5, 9), run_id=FILL)
# A shard: per sensor and date the fill recorded, every granule that supplied a pixel.
shard_granules(REGISTRY, "33N", 2024, "chunk_209_14", FILL)
# A pixel: §2's, from the update that claims it; a row per sensor and date (27 optical, 61 and 28
# radar) with run_id, granule_id, platform, acquired_at, cloud_cover and the asset URLs.
locate(45.5071, 14.3761)              # PixelLocation(zone='33N', tile='chunk_209_14', row=1908, col=70)
pixel_granules(REGISTRY, 45.5071, 14.3761, 2024)
```

The same in raw polars, for one run, here the update. Two rules from the helpers apply to any
reader: a shard's pixels may come from several runs, listed oldest first in the registry's
`source_runs`, and a pixel belongs to the newest run that claims it, where a holes update claims a
pixel only where its `update_pixels` mask marks it and a fill or a shards update claims every
pixel (§5); and every join from `shard_dates` to `granules` goes through the snapshot the run
read, `snapshot_id`, never through `run_id`, because a granule list is written once per mosaic
snapshot and shared by every run that read it:

```python
import numpy as np
import polars as pl
import zarr
from tessera_embeddings.storage.zarr_store import open_repo

table = f"{REGISTRY}/provenance/{{}}/zone=33N/year=2024/*.parquet"
granules, key = pl.scan_parquet(table.format("granules")), ["sensor", "snapshot_id", "solar_day", "fusion_order"]
# The shard: the rows one run wrote for it, one per sensor and date; `sources` lists the fusion
# orders of the granules that supplied at least one of its pixels, so exploding it and joining
# through the snapshot gives the shard's exact granule list.
rows = pl.scan_parquet(table.format("shard_dates")).filter(pl.col("run_id") == UPDATE, pl.col("tile") == "chunk_209_14")
rows.explode("sources").rename({"sources": "fusion_order"}).join(granules, on=key).collect()      # the shard

# The pixel: its plane values are ranks, one per date on the run's time axis; 0 means the date
# was not valid for the model there, and k > 0 names `sources[k - 1]` of that shard-date's row.
run = zarr.open_group(open_repo(f"{REGISTRY}/provenance/pixel_sources.icechunk").readonly_session("main").store,
                      path=f"33N/2024/{UPDATE}", mode="r")
k = np.asarray(run["s2"][:, 209 * 2048 + 1908, 14 * 2048 + 70])     # the pixel's ranks, one per date
days = np.asarray(run["s2_date"][:]).astype("datetime64[ns]").astype("datetime64[D]")
(pl.LazyFrame({"solar_day": days[k > 0], "rank": k[k > 0].astype(np.int64) - 1, "sensor": "s2"})
 .join(rows, on=["sensor", "solar_day"])
 .with_columns(fusion_order=pl.col("sources").list.get(pl.col("rank")))
 .join(granules, on=key)
 .collect())                                                         # the pixel: 27 optical rows
```

For a shard with several `source_runs`, pick the run by §5's rule first, as `pixel_granules` does.

The same in DuckDB, for readers who prefer SQL. The tables are plain Parquet, so `read_parquet`
takes the same paths (add `INSTALL httpfs; LOAD httpfs` for `s3://`). The pixel's ranks still come
from the Icechunk planes, read as above and handed to DuckDB as the small `seen` frame:

```python
import duckdb

con = duckdb.connect()
con.sql(f"""
-- A day: every granule fused into 9 May's optical mosaic, in fusion order. One row set per
-- mosaic snapshot that recorded the day; a fill and an update that read the same mosaic share it.
SELECT snapshot_id, fusion_order, granule_id, platform, acquired_at, cloud_cover
FROM read_parquet('{table.format("granules")}')
WHERE sensor = 's2' AND solar_day = DATE '2024-05-09'
ORDER BY snapshot_id, fusion_order""")

con.sql(f"""
-- The shard: each sensor and date the update used in chunk_209_14 and every granule that supplied
-- a pixel of it. `sources` holds fusion orders; UNNEST it and join through the snapshot the run read.
SELECT d.sensor, d.solar_day, d.valid_px, g.fusion_order, g.granule_id, g.platform, g.acquired_at, g.cloud_cover
FROM read_parquet('{table.format("shard_dates")}') AS d, UNNEST(d.sources) AS s(fusion_order)
JOIN read_parquet('{table.format("granules")}') AS g
  ON g.sensor = d.sensor AND g.snapshot_id = d.snapshot_id AND g.solar_day = d.solar_day AND g.fusion_order = s.fusion_order
WHERE d.run_id = '{UPDATE}' AND d.tile = 'chunk_209_14'
ORDER BY d.sensor, d.solar_day, g.fusion_order""")

con.register("seen", seen)  # the pixel's (sensor, solar_day, rank) from the planes, as above
con.sql(f"""
-- The pixel: rank k names the k-th entry of that shard-date's `sources` (DuckDB lists are 1-based,
-- so rank + 1), which is the fusion order of the granule that supplied the pixel.
SELECT p.sensor, p.solar_day, g.fusion_order, g.granule_id, g.platform, g.acquired_at, g.cloud_cover, g.assets
FROM seen AS p
JOIN read_parquet('{table.format("shard_dates")}') AS d
  ON d.run_id = '{UPDATE}' AND d.tile = 'chunk_209_14' AND d.sensor = p.sensor AND d.solar_day = p.solar_day
JOIN read_parquet('{table.format("granules")}') AS g
  ON g.sensor = d.sensor AND g.snapshot_id = d.snapshot_id AND g.solar_day = d.solar_day
 AND g.fusion_order = list_extract(d.sources, p.rank + 1)
ORDER BY p.solar_day""")
```

Both forms were run against the record the unit tests build and return the same rows.

## Appendix C. For implementers

**Where the code is.** The switch and the ingest half: `docs/ingest/provenance.md`, with
`ingest/provenance.py` and the per-source reader hook in `stac.py`. The actor's planes:
`inference/provenance.py`. Publication and its checks: `inference/provenance_publish.py`, called
from `runners/zone_fill.py` on the fill and update paths and on retry of a landed but untagged
cell. Tables and layout: `storage/provenance.py`. `source_runs`: `published_store.source_runs`,
applied by every registry merge and rebuild. The reader helpers: `storage/provenance_reader.py`.

**Checks at publication**, each failing the zone-year: every mosaic slot has a granule list; the
three mosaics agree on the switch; every embedded shard staged a record; each plane's non-zero
count equals its row's `valid_px` and its values are exactly 1 to the length of `sources`; every
`sources` entry is a granule of that day in the snapshot the row names; a snapshot's `granules`
file, if an earlier run published it, holds the lists read now; a holes update's mask counts
exactly its `embedded_px` and its planes are zero outside it. Re-running these checks from the
registry census, so a later repair can find a broken zone-year, is not built yet.

**Deploying.** The ingest stamps a fingerprint of its own code on every mosaic, and this change
moves it, so the ingest half deploys while nothing is mid-ingest, as #240 and #250 do. The
campaign's deployment in yield-embeddings turns the switch on, before its first ingest (§4).
