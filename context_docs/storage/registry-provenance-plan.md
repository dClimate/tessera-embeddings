# Recording which satellite images went into every TESSERA embedding

**Status.** Built behind a switch that is off by default, and tested end to end on the dev
deployment on 2026-10-27 ([`provenance-dev-verification.md`](provenance-dev-verification.md)).
Not merged: it is three draft pull requests, [#253](https://github.com/dClimate/tessera-embeddings/pull/253)
(ingest), [#255](https://github.com/dClimate/tessera-embeddings/pull/255) (model and publication)
and [#256](https://github.com/dClimate/tessera-embeddings/pull/256) (one registry column), stacked
on two in review, [#240](https://github.com/dClimate/tessera-embeddings/pull/240) and
[#250](https://github.com/dClimate/tessera-embeddings/pull/250). It is meant to start with the
first zone-year of the v2 campaign. The decisions are recorded in
[ADR 035](../decisions/035-a-provenance-record-for-the-published-store.md).

**What we would like from a reviewer.** Whether the record answers the questions users will ask
(§2), whether its costs are worth paying (§6), and views on the choices in §8 and the open
questions in §10. The appendices are reference material and can be skipped.

## Summary

- **The gap.** An embedding is built from a year of satellite images, but the images are fused
  into daily mosaics and the mosaics are deleted once the embeddings are written. Today nobody can
  say which images went into a given embedding, and for the v1.1 store nobody ever will.
- **The record.** For every pixel and every date the model used, the record names the one
  satellite image (the *granule*) that supplied that pixel, with the granule's id, acquisition
  time, cloud cover, footprint and the exact file URLs that were read. It sits beside the store's
  existing registry and is written once per run.
- **Three questions it answers.** What went into this day's mosaic? Which dates and granules did
  this 20 km shard use? Which granule supplied this pixel on each date the model saw it?
- **Cost.** About 4.7 TB of storage over the campaign, a quarter of a percent of the store, and
  about 6–20% more ingest worker time. Correctness checks run at publication and fail the zone-year if the record
  is wrong.
- **Scope.** Only the public global store needs it. Internal and dev stores leave it off and are
  unchanged.

## Words used

| word | meaning |
|---|---|
| zone | one UTM zone and hemisphere, such as `33N`. The store is filled one zone-year at a time |
| shard | a 2048 × 2048 pixel block of a zone, about 20 km on a side, named `chunk_<row>_<col>`. The model works a shard at a time, and the registry has one row per shard per year |
| granule | one catalogue entry for one image: a Sentinel-2 scene (optical) or a Sentinel-1 burst (radar) |
| mosaic | the zone-wide picture the ingest builds for one sensor on one day by fusing that day's granules. It is deleted when the zone-year is published |
| solar day | the local calendar day a granule belongs to; the mosaic has one slot per solar day |
| fusion order | the order a day's granules are handed to the mosaic builder: optical clearest first, radar most-often-acquired burst first (#250). Where granules overlap, the first one with data supplies the pixel |
| run | one pass that writes pixels into a zone-year: the first fill, or a later *update* that redoes some of them |
| registry | the Parquet tables published beside the store, one row per shard per year, saying what each shard holds |

## 1. Why this is needed

```text
                     INGEST                       MODEL                    PUBLISHED STORE
  Sentinel-2 ──┐   for each day, fuse the     for each pixel, read its     one embedding
  scenes       ├─▶ day's granules into one ─▶ year of mosaic values,   ─▶  per pixel, plus
  Sentinel-1 ──┘   mosaic per sensor          drop cloudy dates, embed     the registry
  bursts
  (across one zone, about 1,000 optical granules and 900–2,600 radar bursts per orbit, every day)

                   mosaics are deleted after publication: about 14 PB for the campaign
```

A user looking at an embedding will reasonably ask what imagery is behind it: which dates, from
which satellite, how cloudy the scene was, and where to fetch the original files. The ingest
knows which granules it fused each day, but only in memory, and the model knows which dates each
pixel used, but forgets them once the shard is written. Once the mosaics are deleted the answer is
gone. That is the situation for the published v1.1 store, and this plan closes the gap from the
first v2 zone-year onward.

## 2. What a reader can find out

| question | answer | read from |
|---|---|---|
| What went into one day's mosaic? | every granule fused that day, in fusion order, with ids, acquisition time, cloud cover, footprint, catalogue link and file URLs | `granules` |
| What did one shard use? | per sensor, the dates the model used, how many of the shard's pixels were valid on each, and which granules supplied them | `shard_dates` |
| What is behind one pixel? | per sensor, every date the model saw the pixel, and the one granule that supplied it on each | `pixel_sources`, then `shard_dates` and `granules` |

**A worked example, with real values from the dev test.** A pixel at 45.5149°N, 14.3794°E, on
the Slovenia–Croatia border. It lies in zone `33N`, shard `chunk_209_14`, row 1822, column 96. The
registry row for that shard and 2024 says which runs wrote it; take the first fill. That run's
`pixel_sources` holds, for the optical sensor, one small number per date on the year's mosaic:

```text
  date          2024-01-10   2024-01-15   2024-01-20   …   2024-12-30
  number             1            0            1       …        1
                     │            │            │
                     │       0: the model did not see this pixel that day
                     │          (no granule had data there, or the cloud mask removed it)
                     ▼
  shard_dates row (chunk_209_14, optical, 2024-01-20):   sources = [0]
          number 1 means "the 1st entry of sources", fusion order 0
                     ▼
  granules row (optical, 2024-01-20, fusion order 0):
          S2B_33TVL_20240120_0_L2A · Sentinel-2B · 10:08 UTC · 14.6% cloud
          catalogue link, footprint, and the URLs of the 11 band files that were read
```

The pixel has 40 such dates for optical, 61 for ascending radar and 28 for descending radar.
Radar shows why the extra step through `sources` exists. On 1 January the ascending number is 2
and the shard's `sources` is `[0, 2, 4]`, so the pixel came from fusion order 2, burst
`T044-092628-IW3`: three bursts covered parts of the shard that day, and the record names the one
that covered this pixel.

**What the record does not say.** How much any date or granule mattered to the embedding: the
model has no attribution, so the record says what went in, not what it weighed. Nor the pixel
values themselves: the mosaics are gone, and a reader who wants the imagery re-reads the recorded
URLs. Nothing about v1.1, whose mosaics are gone.

**How this compares.** OlmoEarth publishes provenance per 2048-pixel window and 30-day period,
listing the candidate scenes rather than the one used, and its model mixes neighbouring pixels.
TESSERA's model reads one pixel at a time, so a per-pixel answer is exact.

## 3. What gets recorded

Four new pieces under `provenance/` beside the registry, and one new registry column. Each run
writes its own files and never changes another run's.

```text
  <store>.registry/
  ├── master/  parts/          the existing registry; both gain a `source_runs` column
  └── provenance/
      ├── granules/      zone=33N/year=2024/<run>.parquet   one row per granule per day
      ├── shard_dates/   zone=33N/year=2024/<run>.parquet   one row per shard, sensor and date
      ├── update_pixels/ zone=33N/year=2024/<run>.zarr      holes updates only: which pixels they wrote
      └── pixel_sources.icechunk/  33N/2024/<run>/{s2, s1_asc, s1_desc}
                                   one small integer per pixel per date
```

**`granules`: what went into each day's mosaic.** One row per granule per solar day. It is
written per zone-day rather than per shard because one Sentinel-2 scene covers about thirty
shards. A real row:

| run_id | sensor | solar_day | fusion_order | granule_id | platform | acquired_at | cloud_cover |
|---|---|---|---|---|---|---|---|
| 33N-2024-9a1fc94c | s2 | 2024-01-02 | 0 | S2A_33TVL_20240102_0_L2A | S2A | 09:58:07 UTC | 94.9 |

It also carries the processing baseline, the ESA product name, the catalogue link, the footprint
as GeoParquet, and a map from each band to the URL read, here 11 files such as
`…/S2A_33TVL_20240102_0_L2A/B04.tif`.

**`shard_dates`: what each shard used.** One row per shard, sensor and date the model used:

| run_id | tile | sensor | solar_day | valid_px | embedded_px | sources |
|---|---|---|---|---|---|---|
| 33N-2024-9a1fc94c | chunk_209_14 | s1_desc | 2024-01-12 | 4,194,294 | 15,668 | [0, 1, 2] |

`valid_px` is how many of the shard's pixels had a valid observation that date, `embedded_px` how
many pixels the run embedded in the shard, and `sources` the fusion orders of the granules that
supplied at least one pixel. That makes `sources` the exact per-shard granule list: granules whose
footprint merely touches the shard are not in it.

**`pixel_sources`: which granule supplied each pixel.** Per run and sensor, an array of one byte
per pixel per date. Zero means the model did not see the pixel that date; k means the k-th entry
of that shard-date's `sources`. The planes are mostly constant, so they compress to about a
megabyte per shard-year (§6). It lives in an Icechunk store of its own rather than inside the
published store, because each run has its own list of dates and the published store's arrays all
share one grid.

**`update_pixels`: which pixels a holes update wrote.** A holes update writes only some pixels of
a shard (§5), and this boolean mask per shard says which.

**`source_runs`: which runs a shard's pixels came from.** A new registry column listing, oldest
first, the runs whose pixels a shard holds now: one run for most shards, two after a holes update.
It is recomputed from the store every time the registry is merged or rebuilt, never copied from an
old row.

**Which pixels the record covers.** A run's record covers every pixel the run was asked to embed,
including pixels it then refused, for example for having too few clear observations. Whether a
pixel ended up with an embedding is in the store and the registry. That is how a refused shard
still has rows, and it lets a reader see what a refused pixel had (§8).

## 4. How it is made

The pipeline already has three stages; each gains one job.

```text
  INGEST, per zone-year                 MODEL, per shard                 ASSEMBLY, per zone-year
  ─────────────────────                 ────────────────                 ───────────────────────
  for each day and sensor:              read the mosaics, run the       commit the shards to the store
   fuse the granules into the mosaic     model, stage the shard          publish the registry rows
   + NEW: write the day's granule list   + NEW: stage, per date, the   + NEW: publish the record,
   + NEW: write a source_index band        index masked to what the        check it, and fail the
     (which granule supplied each          model saw, with its             zone-year if it is wrong
     pixel), in the same commit            granule list and counts       then tag, delete staging and mosaics
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
bit what a plain read gives, and the index. No band is read twice and no step is added to the
fuse. The model's stage then sets the index to zero wherever its own cloud and validity test
rejects the observation, so a published non-zero means the observation was used, and renumbers
each shard-date's granules 1, 2, 3 so a byte suffices (a day's zone-wide fusion order runs past
255; a shard rarely has more than a handful on one date).

**Three rules keep the record and the embeddings consistent.**

1. *The granule list is written in the same commit as the day's pixels.* A day that did not
   commit has neither, so the list can never describe a day the mosaic lacks.
2. *The model and the record read the same version of the mosaics.* The fill notes each mosaic's
   version when it plans, and every read after that uses it, so a change to a mosaic during a
   fill changes neither the embeddings nor the record. The dev test made such a change and the
   record was unaffected.
3. *If publishing the record fails, the zone-year fails, before anything is deleted.* The staging
   and mosaics a retry needs are still there, and the retry publishes the same record. The dev
   test killed a fill at that point and the retry's record matched an uninterrupted one exactly.

**The switch.** One setting, `provenance`, off by default, is set in the ingest. Each mosaic
records whether it was built with the record, and a run that disagrees with an existing mosaic is
refused with a reason rather than producing a half-recorded year. The model and publication
follow the mosaics. A zone-year built with the switch off has no `provenance/` files and an empty
`source_runs`, which is how a reader tells "not recorded" from "nothing seen".

## 5. Updates

A *holes update* re-runs the model only on pixels a fill left empty, often after lowering the
minimum number of clear observations, and leaves every other pixel untouched. A shard's pixels
then come from two runs, and the record keeps both. The dev test did exactly this. Its fill
used a line of 40 clear optical observations, so in shard `chunk_209_14` it embedded 15,668 pixels
and refused the rest; an update at a line of one embedded the remaining 4,178,636.

```text
  shard chunk_209_14, 2024              source_runs = [33N-2024-9a1fc94c, 33N-2024-e892e33b]
  ┌──────────────────────────────┐
  │ ░ embedded by the fill       │   15,668 px    → read the fill's pixel_sources and granules
  │ ▓ embedded by the update     │   4,178,636 px → marked in the update's update_pixels;
  │   (the fill's holes)         │                  read the update's pixel_sources and granules
  └──────────────────────────────┘
```

The pixel at 45.5071°N, 14.3761°E, row 1908, column 70, had 27 clear optical dates: too few for
the fill, enough for the update. The update's `update_pixels` marks it, and the update's own
record names its 27 optical, 61 ascending and 28 descending dates and their granules. A *shards
update*, which rewrites whole shards, restarts `source_runs` at itself.

## 6. What it costs

**Storage,** measured on the dev test's shards and on one week of a whole zone:

| piece | measured | campaign |
|---|---|---|
| `pixel_sources` | optical 1.1–1.2 MB a shard-year; radar 0.07–0.2 MB per orbit | about 4.7 TB, a quarter of a percent of the store, around $1,300 a year |
| `granules` | about 160 MB for a land-dense zone-year (zone 35N) | at most about 170 GB |
| `shard_dates`, `update_pixels` | about 10 MB a zone-year; kilobytes per updated shard | about 10 GB |
| the granule lists and index inside the mosaics | about 150 MB and 0.1–0.3% of a mosaic | none: deleted with the mosaics |

The integer per pixel costs 4% more than a bare yes/no bit per pixel would, measured on a real
shard-year, because both compress to almost nothing over runs of identical values.

**Ingest time.** With the record on, each date of a whole zone took 5–17% longer for optical and
10–30% longer for radar, over two paired runs; about 6–20% more ingest worker time in all, which
on the campaign cost model's ingest line of about $121,000 is roughly $7,000–24,000. The cost is
CPU, not reading: packing and fusing the wider band, and writing the index as one more array. A
small area showed no difference, because its workers had time to spare.

**Publication** took about 16 seconds for two shards. A whole zone-year is about 3,000 shards,
copied eight at a time, and has not been measured.

## 7. How it was tested

| what | result |
|---|---|
| bands built with the switch on against off, Iowa | bit-identical |
| the published record against the mosaics, 36 sampled shard-dates and 48 sampled pixels | every plane matched; every sampled optical pixel equals the named granule's own file |
| a run whose switch disagrees with an existing mosaic | refused with a reason, both ways |
| a fill killed between its commit and publication | left unpublished and untagged with its inputs intact; the retry's record equals an uninterrupted one |
| a mosaic changed mid-fill | the record and embeddings read the version planned |
| a holes update, a registry rebuild and a shards update | each run's record correct; `source_runs` follows |
| ingest time, Iowa and a week of zone 35N, paired | §6 |

Details, numbers and the checks run at every publication are in
[`provenance-dev-verification.md`](provenance-dev-verification.md) and Appendix C.

## 8. Choices worth questioning

| choice | alternative | why this one |
|---|---|---|
| a granule per pixel per date | a yes/no bit per pixel per date, with the granule guessed from footprints | 4% more storage, and exact: guessing named the wrong granule for 0.10% of one shard's pixel-dates, wherever footprints overlap |
| a per-shard granule list derived from the pixels | every granule whose footprint touches the shard | names only granules that supplied a pixel |
| the record covers refused pixels | only embedded pixels | a refused shard still says what imagery it had; the store says what was embedded |
| the index written by the ingest while it fuses | reading the mask band a second time, or once per granule | one read, one definition of "valid"; per granule would multiply the work by the granules in a day, hundreds of radar bursts |
| a separate Icechunk store for the planes | arrays inside the published store | each run has its own dates; the published store and its conventions stay untouched |
| failure to publish fails the zone-year | publish best-effort and log | a zone-year can never be published without its record, and the retry repairs it |
| off by default, on for the public store only | always on | internal and dev stores neither need it nor pay its ingest cost |

## 9. Left out

- **The granule behind a cloudy pixel.** The mosaic knows, but only what the model used is
  published.
- **A separate index per band.** The index follows the band the validity test reads. A granule with
  data in one band and none in another at the same pixel would break this; the products do not
  produce one, and the dev test's pixel checks found none.
- **v1.1**, whose mosaics are gone.
- **Radar quality layers** such as layover and shadow, a separate decision (ADR 033).
- **Copies of an image the ingest rejected** after a failed read. They could be added later as
  rows that supplied no pixel, without changing the schema.

## 10. Open questions for reviewers

1. **Is 6–20% more ingest time acceptable for the public store?** Or should it be reduced before
   the campaign? The cost is spread over packing a wider band and writing one more array; both
   could be made cheaper, at some complexity.
2. **Is per-pixel, per-date detail what users need?** The shard-level tables alone cost almost
   nothing; the per-pixel planes are most of the storage.
3. **Should refused pixels stay in the record?** (§3, §8.)
4. **Radar file URLs need Earthdata credentials to fetch.** Is it enough to record them, or should
   the record also point to a public mirror where one exists?
5. **Is anything missing** that a user would need to trust or reproduce an embedding?

## Appendix A. Table schemas

`zone` and `year` are folder names (hive partitions), which every Parquet reader turns into
columns; they are not stored in the files, because pyarrow refuses a dataset whose partition keys
are also columns. Every file's metadata records the zone, year, run and the snapshot id of each
mosaic it was read from.

**`granules`**

| column | type | meaning |
|---|---|---|
| `run_id` | string | the run that read the mosaic |
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

**`shard_dates`**: `run_id`, `tile`, `sensor`, `solar_day`, `valid_px` (int32), `embedded_px`
(int32) and `sources` (list of int16 fusion orders, ascending), as in §3.

**`pixel_sources`**: one group per run, `<zone>/<year>/<run_id>`, holding per sensor a
`(date, northing, easting)` uint8 array chunked one date by one shard, beside a `<sensor>_date`
array giving each date. Only the shards and dates a run used are written. The codec was chosen on
one real optical shard-year (33N `chunk_209_14`, 2024):

| encoding of one plane per date | MB a shard-year |
|---|---|
| Blosc zstd level 3 with bit shuffle, the one used | 1.17 |
| two bits a pixel, zstd level 3 (would cap a shard-date at three granules) | 1.38 |
| zstd level 3, no shuffle | 2.83 |
| a bare validity bit, zstd level 3, for comparison | 1.12 |

Without the bit shuffle zstd does not see the mostly constant planes as runs. Packing bits along
time and XOR-delta planes were also tried and lost to the per-date plane.

**`update_pixels`**: one boolean `(2048, 2048)` array per shard, named for the shard.

## Appendix B. Reading the record

Each example ran against the dev test's record. `REGISTRY` is the registry's root.

```python
import datetime as dt
import numpy as np
import polars as pl
import zarr
from tessera_embeddings.storage.zarr_store import open_repo

granules = pl.scan_parquet(f"{REGISTRY}/provenance/granules/**/*.parquet", hive_partitioning=True)
shard_dates = pl.scan_parquet(f"{REGISTRY}/provenance/shard_dates/**/*.parquet", hive_partitioning=True)
master = pl.scan_parquet(f"{REGISTRY}/master/**/*.parquet", hive_partitioning=True)

# A day: the optical granules fused into 9 May 2024 in zone 33N, in fusion order.
granules.filter(pl.col("zone") == "33N", pl.col("year") == 2024, pl.col("sensor") == "s2",
                pl.col("solar_day") == dt.date(2024, 5, 9)).sort("run_id", "fusion_order").collect()

# A shard: the runs that wrote it, then the granules that supplied it on each date.
runs = master.filter(pl.col("zone") == "33N", pl.col("year") == 2024, pl.col("tile") == "chunk_209_14") \
             .select("source_runs").collect().item()
supplied = (shard_dates.filter(pl.col("zone") == "33N", pl.col("year") == 2024, pl.col("tile") == "chunk_209_14",
                               pl.col("sensor") == "s2", pl.col("run_id") == runs[0])
            .explode("sources").rename({"sources": "fusion_order"})
            .join(granules, on=["zone", "year", "run_id", "sensor", "solar_day", "fusion_order"])
            .collect())

# A pixel: row 1908, column 70 of the shard, as a position in the zone's grid.
row, col = 209 * 2048 + 1908, 14 * 2048 + 70
run = zarr.open_group(open_repo(f"{REGISTRY}/provenance/pixel_sources.icechunk").readonly_session("main").store,
                      path=f"33N/2024/{runs[0]}", mode="r")
k = np.asarray(run["s2"][:, row, col])                                 # one number per date
days = np.asarray(run["s2_date"][:]).astype("datetime64[ns]").astype("datetime64[D]")
seen = pl.DataFrame({"solar_day": days[k > 0], "rank": k[k > 0].astype(np.int32) - 1})
(seen.lazy()
 .join(shard_dates.filter(pl.col("run_id") == runs[0], pl.col("tile") == "chunk_209_14", pl.col("sensor") == "s2"),
       on="solar_day")
 .with_columns(fusion_order=pl.col("sources").list.get(pl.col("rank")))
 .join(granules.filter(pl.col("sensor") == "s2"), on=["zone", "year", "run_id", "sensor", "solar_day", "fusion_order"])
 .collect())                                                           # one granule per date seen
```

For a shard with two `source_runs`, a pixel marked in the later run's `update_pixels` is read from
that run, and every other pixel from the first.

## Appendix C. For implementers

**Where the code is.**

| part | code |
|---|---|
| the switch | `IngestSettings.provenance` and a flat `provenance` parameter on the ingest flows; the mosaic manifest's `provenance_version`, a term whose absence means off |
| granule lists and the index | `ingest/provenance.py` (fusion positions, packing, granule lists); `stac.SourceParser` and `stac._SourceReader`, the per-source reader hook the Sentinel-2 offset correction also uses; `zarr_store.write_days_windows(granules=)` |
| the actor's planes | `inference/provenance.py`, read where each sensor's mask is read, ranked on the actor's background writer |
| publication and checks | `inference/provenance_publish.py`, called from `runners/zone_fill.py` on the fill and update paths, and on retry of a landed but untagged cell |
| tables and layout | `storage/provenance.py` |
| `source_runs` | `published_store.source_runs`, applied by every registry merge and rebuild |

**Checks at publication**, each failing the zone-year: every mosaic slot has a granule list; the
three mosaics agree on the switch; every embedded shard staged a record; each plane's non-zero
count equals its row's `valid_px`, its values are exactly 1 to the length of `sources`, and every
`sources` entry is a granule of that day; a holes update's mask counts its `embedded_px` and its
planes are zero outside it. Re-running these checks from the registry census, so a later repair
can find a broken zone-year, is not built yet.

**Deploying.** The ingest stamps a fingerprint of its own code on every mosaic, and this change
moves it, so the ingest half deploys while nothing is mid-ingest, as #240 and #250 do. The
campaign's deployment in yield-embeddings turns the switch on.

## Appendix D. How the design changed

Two adversarial reviews and the build changed the first draft:

| first draft | now | why |
|---|---|---|
| a granule file written beside the mosaic before its commit | a list inside the mosaic, written in the day's own commit | a file written first can describe a day that never committed |
| one record per shard-year | a record per run, `update_pixels` and `source_runs` | a holes update mixes two runs in one shard |
| `source_runs` kept on the master only | recomputed from the store at every merge and rebuild, also on part rows | a rebuild replaces the master's run id and rebuilds the master from parts |
| publication best-effort, then a gate | a failure fails the zone-year before cleanup | best-effort followed by cleanup cannot be repaired |
| no way to resume older stores | a switch, recorded in the mosaic, with mismatches refused | dev stores from before the change must stay usable |
| the granule guessed from footprint order | the index, per pixel | footprint order does not identify the winner where granules overlap |
| a bit per pixel ruled out at an estimated 20–45 TB | an integer per pixel, measured at 3.7 TB for optical | the estimate was an order of magnitude high |
| each granule loaded separately to find the winner | a reader that packs the position into the pixels | per-granule loads multiply the work by hundreds of radar bursts a day |
| no ingest cost expected | 6–20% of ingest worker time, measured | packing, fusing and writing cost CPU even with no extra read |
