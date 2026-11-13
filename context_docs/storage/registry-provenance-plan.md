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
- **The record.** For every pixel, and every date on which a satellite image had data there, the
  record names that image (the *granule*) and says whether the model used the observation or
  rejected it, a cloud for instance, with the granule's id, acquisition time, cloud cover,
  footprint and the catalogue's URLs of the files that were read. It also lists the copies the ingest dropped
  before building each day's mosaic, and why. It sits beside the store's existing registry.
- **Three questions it answers,** each with a reader helper: what went into this day's mosaic,
  into this 20 km shard, and which granule supplied this pixel on each date?
- **Cost.** About 6.7–7 TB of storage over the campaign, a third of a percent of the store. Its
  ingest time cost is inside run-to-run variation, measured over seven paired runs of a dense zone
  (§6); what it certainly adds is one more chunk write per date for the Dask scheduler.
  Correctness checks run at publication and fail the zone-year if the record is wrong.
- **Scope.** Only the public global store needs it. Internal and dev stores leave it off and are
  unchanged.

## Words used

| word | meaning |
|---|---|
| zone | a group of the store: on v1.1's layout a UTM zone and hemisphere, such as `33N`, which the dev runs below used; on v2's (ADR 037) a whole UTM column, `utm33`, filled as two halves, `utm33n` and `utm33s` (ADR 038). The record, like the registry, is keyed by the group, so both halves' runs publish under `zone=utm33` |
| shard | a 2048 × 2048 pixel block of a zone, about 20 km on a side. The model works a shard at a time, and the registry has one row per shard per year. For historical reasons a shard is named `chunk_<row>_<col>` and the registry's column for it is `tile`; both mean shard |
| sensor | `s2` (Sentinel-2, optical), `s1_asc` and `s1_desc` (Sentinel-1 radar, ascending and descending orbits) |
| granule | one catalogue entry for one image: a Sentinel-2 scene (optical) or a Sentinel-1 burst (radar) |
| mosaic | the zone-wide picture the ingest builds for one sensor on one day by fusing that day's granules. It is deleted when the zone-year is published |
| mosaic snapshot | the version of a mosaic a run read, an Icechunk snapshot id. A run that ingests builds its own mosaics, so it reads its own snapshots |
| solar day | the local calendar day a granule belongs to; the mosaic has one slot per solar day |
| fusion order | the order a day's granules are handed to the mosaic builder: optical clearest first, radar most-often-acquired burst first (#250). Where granules overlap, the first one with data supplies the pixel |
| valid for the model | an observation that passes the model's own test: a clear scene class for optical, data in VV or VH for radar |
| rejected | an observation a granule supplied that fails the model's test, a cloudy pixel for instance. The record keeps its granule |
| dropped | a catalogue copy the ingest left out of a day's mosaic: a losing duplicate of an image it fused, or a copy that would not read and was replaced by another |
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
| What went into one day's mosaic? | every granule fused that day, in fusion order, with ids, acquisition time, cloud cover, footprint, catalogue link and file URLs; then the copies dropped, and why | `day_granules` | `granules` |
| What went into one shard? | per sensor, the dates on which any granule had data at its pixels, how many observations the model used, and which granules covered them | `shard_granules` | `shard_dates`, then `granules` |
| What is behind one pixel? | per sensor, every date a granule had data at it, that granule, and whether the model used the observation or rejected it | `pixel_granules` | `pixel_sources`, then `shard_dates` and `granules` |

The helpers are the methods of `ProvenanceRecord` in `storage/provenance_reader.py`, and return polars tables (Appendix B).

**A worked example, with real values from the dev test.** The pixel at 45.5071°N, 14.3761°E, on
the Slovenia–Croatia border, as `pixel_granules` finds it:

```text
  45.5071°N 14.3761°E ──▶ zone 33N, shard chunk_209_14, row 1908, column 70
  registry row, 2024:    source_runs = [fill 33N-2024-9a1fc94c, update 33N-2024-e892e33b]
  newest run first:      the update's update_pixels marks the pixel, so the update's record answers (§5)
  that record's planes:  one small number per date, positive (used by the model) on 27 optical,
                         61 ascending and 28 descending radar dates
```

Each non-zero number leads to one granule; its sign says whether the model used it. The dev test traced those steps on a second pixel of
the same shard, row 1822, column 96 (45.5149°N, 14.3794°E), which the fill embedded:

```text
  pixel_sources (fill, s2), 2024-01-20:  1     (−1 would mean the same granule covered the pixel
        │                                       and the model rejected it; 0, that no granule had data)
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
the record names the one that covered this pixel. The dev test ran before rejected observations
were recorded, so its planes hold 0 on the cloudy dates where a run now records a negative.

**How this compares.** OlmoEarth publishes provenance per 2048-pixel window and 30-day period,
listing the candidate scenes rather than the one used, and its model mixes neighbouring pixels.
TESSERA's model reads one pixel at a time, so a per-pixel answer is exact.

## 3. What gets recorded

Four new pieces under `provenance/` beside the registry, and one new registry column.

```text
  <store>.registry/
  ├── master/  runs/           the existing registry; `master/` gains a `source_runs` column
  └── provenance/
      ├── granules/      zone=33N/year=2024/<sensor>-<snapshot>.parquet   per mosaic snapshot: its granules
      ├── shard_dates/   zone=33N/year=2024/<run>.parquet    per run: one row per shard, sensor and date
      ├── update_pixels/ zone=33N/year=2024/<run>.zarr       holes updates only: the pixels they embedded
      └── pixel_sources.icechunk/  33N/2024/<run>/{s2, s1_asc, s1_desc}
                                   per run: one small integer per pixel per date
```

The Parquet folders use hive `key=value` names because Parquet readers turn them into columns;
the Icechunk groups use plain names because Zarr readers do not. `runs/` holds one file per run
per zone-year, as `shard_dates/` does, so the registry and the record are both keyed by run
([ADR 036](../decisions/036-the-registry-folder-is-runs.md)).

**`granules`: what went into each day's mosaic.** One row per granule per solar day, per zone
rather than per shard because one Sentinel-2 scene covers about thirty shards. A list belongs to
the mosaic it was fused into, so the file is keyed by mosaic snapshot and written once, by the first
run that reads the snapshot; a later run reading the same snapshot, such as a retry, checks it
instead. A run that re-ingests builds new mosaics and writes new files. A real row, from the
fill's optical snapshot: 2 January 2024, fusion order 0, `S2A_33TVL_20240102_0_L2A`, Sentinel-2A,
09:58:07 UTC, 94.9% cloud, with its catalogue link, its footprint as GeoParquet, and the URL of
each of the 11 files read, such as `…/S2A_33TVL_20240102_0_L2A/B04.tif` (Appendix A). After a
day's fused granules come the copies the ingest dropped before fusing it, with no fusion order and
their reason in `status`: `duplicate`, a losing copy of an image it fused; `unreadable`, a copy that
would not read and was replaced by another, which `stepped_down` marks; or `missing_bands`, a radar
granule whose catalogue entry lacks its VV or VH file. A day the ingest skipped whole has no mosaic
and no fused granules, only its copies, each dropped for the day's reason unless already dropped for
its own: `low_coverage`, too little
clear ground for the coverage gate; `producer_conflict`, no single offset correction fitting the
day's images; `unreadable`, every copy of some image failed to read; or `missing_bands`, every radar
granule of the day lacked VV or VH, so the loader had nothing to fuse.

**`shard_dates`: what went into each shard.** Per run, one row per shard, sensor and date on which
a granule had data at at least one of the shard's pixels:

| run_id | tile | sensor | snapshot_id | solar_day | valid_px | embedded_px | sources |
|---|---|---|---|---|---|---|---|
| 33N-2024-9a1fc94c | chunk_209_14 | s1_desc | the fill's descending snapshot | 2024-01-12 | 4,194,294 | 15,668 | [0, 1, 2] |

`valid_px` counts the shard's pixels whose observation the model used that date; `embedded_px` the
pixels the run embedded in the whole shard, repeated on each of its rows; `sources` the fusion orders
of the granules that had data at at least one pixel, used or rejected, so footprints that merely
touch the shard are not in it. `snapshot_id` names the mosaic the run read, and is how a row finds its granules.

**`pixel_sources`: which granule supplied each pixel.** Per run and sensor, an array of one signed
16-bit integer per pixel per date. A value k is the granule's fusion order plus one, `|k| - 1`:
positive when the model used the observation, negative when it rejected it, a cloud for instance,
and 0 when no granule had data. At a pixel with more valid observations than the model's deepest
bucket (256 per sensor) the model reads an evenly spaced subset of the positive values
(`build_resample_indices_v2`), which is deterministic from the count, so the plane keeps every
valid one rather than repeating that selection. The run's group names each sensor's mosaic snapshot, so the value
and its date find the granule in `granules` without going through `shard_dates`. A fully cloudy
date keeps its plane. The planes compress to about one to one and a half megabytes per optical
shard-year (§6). They live in an Icechunk store of their own because each run has its own dates, while the
published store's arrays all share one grid.

**`update_pixels`, and which pixels the record covers.** A fill or a shards update records every
pixel it was asked to embed, refused ones included, so a refused shard still has rows and a reader
can see what imagery a refused pixel had (§8). A holes update records only the pixels it embedded:
its `update_pixels` marks them, one boolean mask per shard, and its planes are cut to the same
mask. Every other pixel keeps the record of the run before it (§5). Whether a pixel has an
embedding is in the store and the registry.

**`source_runs`: which runs a shard's pixels came from.** A new column on the registry's master
listing them oldest first, from the store's shard commits that wrote or cleared the shard, in the
order they landed. A fill or a shards update restarts the list at itself; a holes update appends
itself. So a refill drops the updates before it, and of two concurrent refills the one whose shards
landed last owns the shard. A shard a refill left untouched keeps the fill before it: the zone
group's `runs` record keeps a replaced fill in `previous`, with its `provenance_version` when it
carried a record. It is null unless every run in the list carries a record, or when a commit's
shards can no longer be read (a run from before commits were tagged, or an expired commit). It is
recomputed from the store every time the master is merged or rebuilt, never copied from an old
row, and `rebuild_registry_measurements.py verify` checks every row of it against the store.

## 4. How it is made

The pipeline already has three stages; each gains one job.

```text
  INGEST, per zone-year                 MODEL, per shard                 ASSEMBLY, per zone-year
  ─────────────────────                 ────────────────                 ───────────────────────
  for each day and sensor:              read the mosaics, run the       + NEW: publish the record,
   fuse the granules into the mosaic     model, stage the shard            check it, and fail the
   + NEW: write the day's granule list   + NEW: stage, per date, the       run if it is wrong
   + NEW: write a source_index band        index, negated where the      commit the shards to the store
     (which granule supplied each          observation was not valid     publish the registry rows
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
index is one more array to write. The model's stage negates the index wherever its own cloud and
validity test rejects the observation, so a published positive value means the model used the
observation and a negative one that the granule had data there and was rejected. It reads the
optical index on every date, a fully cloudy one included, although the model itself skips those.
The published value is that signed index as `int16`: a day's zone-wide fusion order runs past
255, so a byte would not hold it, and keeping the fusion order itself lets a pixel join `granules`
directly.

**How the ingest records what it dropped.** Before fusing a day, the Sentinel-2 ingest keeps one
copy of each image and holds the others back, in case the kept copy will not read. It marks each
copy it holds back as a duplicate, and a kept copy that fails to read and is swapped for one held
back as unreadable. Radar fuses every copy of a burst, but its catalogue query sets aside a granule
whose entry lacks its VV or VH file, and marks it missing bands. The day's granule list names all
of these after the fused granules.

**How the ingest records a day it skipped whole.** A skipped day has no slot, so no list beside its
pixels. The ingest commits its list to two arrays of the mosaic off the time axis, the skipped days
and their lists, in a commit of its own when it skips the day; while the mosaic does not exist yet,
the day waits for the leg's first write to create it. Skipped days never move where a resumed
ingest restarts, so a crash before the commit is recovered by the retry, which reaches the day
again, and a list recorded twice is replaced by the latest. Days whose imagery reaches no land in
the zone are not recorded: none of it covered a pixel the store embeds.

**Three rules keep the record and the embeddings consistent.**

1. *The granule list is written in the same commit as the day's pixels.* A day that did not
   commit has neither, so the list can never describe a day the mosaic lacks.
2. *The model and the record read the same version of the mosaics.* The fill notes each mosaic's
   snapshot when it plans and every later read uses it, so a change to a mosaic during a fill
   changes neither the embeddings nor the record, which names those snapshots. Each staged shard
   names the snapshots it was read from, and publication refuses one that names others.
3. *The record is published before the run's shards commit.* A committed run therefore always has
   its record, and a failed publication fails the run with nothing committed, its staging kept
   for the retry, which publishes again. A record whose run never commits is never read: a reader
   reaches a run only through `source_runs`, which the store's own commits derive.

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
holes update, as its run group's `mode` attribute states, claims a pixel only where its
`update_pixels` marks it, and a fill or a shards update claims every pixel. The first run that claims the pixel holds its record; the helpers apply this.

**A master that has not caught up is refused.** `source_runs` is derived by the master merge that
follows each run's commit, and a merge can lag or fail. Each merge stamps the master with the very
list `source_runs` was derived from, every commit that wrote shards into the year as its run and
its time (`store_runs`), and the pixel helpers build that list from the store again, from its
commit history and run records, by the same function; a difference raises `LookupError` rather
than name an older run's images. So shards committed before their run record, a retry under the
same run id, and a lagging or failed merge are all refused, and a run that wrote nothing (a null
commit) changes neither list. A commit's time is the one its run record keeps, so expiring old
snapshots leaves the list unchanged. A failed
merge so shows as an error until a later merge, or `rebuild_registry_measurements.py repair`, lands.

## 6. What it costs

**Storage,** measured on the dev test's shards and on one week of a whole zone:

| piece | measured | campaign |
|---|---|---|
| `pixel_sources` | optical 0.8–1.3 MB a shard-year; radar 0.16–0.73 MB per orbit, and up to 4.6 MB on one of the three measured | about 6.7–7 TB, a third of a percent of the store, around $1,900 a year |
| `granules` | about 160 MB for a land-dense zone-year (zone 35N) | at most about 170 GB for the fills, plus a set per update that re-ingests |
| `shard_dates`, `update_pixels` | about 10 MB a zone-year; kilobytes per updated shard | about 10 GB |
| the granule lists and index inside the mosaics | about 150 MB and 0.1–0.3% of a mosaic | none: deleted with the mosaics |

The ~150 MB inside the mosaics and the ~160 MB `granules` table are the same lists in two
encodings; the mosaics' copy is deleted at publication. Recording rejected observations, so
that cloudy pixels are non-zero and fully cloudy shard-dates keep their planes, makes the planes
about 4% larger on optical and 0–5% on radar. The planes are chunked like the v2 store, 512 × 512
pixels in shards of 2048 × 2048, each chunk holding every date of the run, so reading one pixel's
year is one fetch (§8).

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

**Publication** took 16 seconds for two shards and 31 seconds for twelve, including the registry
run file. It copies planes eight shards at a time, for three sensors, so twelve shards are two batches
of about 15 seconds each; at that rate a zone-year of about 3,000 shards takes about an hour and a
half, a risk to examine separately (§10).

## 7. How it was tested

| what | result |
|---|---|
| bands built with the switch on against off | bit-identical |
| the published record against the mosaics, sampled shard-dates and pixels | every plane matched; every sampled optical pixel equals the named granule's own file |
| a run whose switch disagrees with an existing mosaic | refused with a reason, both ways |
| a fill killed between its commit and publication, then retried | the retry's record equals an uninterrupted one |
| a mosaic changed mid-fill | the record and embeddings read the snapshot planned |
| a holes update, a registry rebuild and a shards update | each run's record correct; `source_runs` follows |
| signed planes, fully rejected shard-dates and days skipped for coverage, on twelve shards | every sampled plane equals the mosaic's index under the validity test, sign included; each skipped day is published with its copies, and only days off the time axis are |

The holes update, the crash and the mid-fill commit ran on an earlier layout (a `granules` file
per run; an `update_pixels` of the pixels asked for); the twelve-shard run tested today's. Copies
set aside as duplicate, unreadable or missing bands, and days skipped for a producer conflict or a
failed read, did not occur in that area and year, so those paths have unit tests only. Numbers: [`provenance-dev-verification.md`](provenance-dev-verification.md).

## 8. Choices worth questioning

The last column is the first draft's choice, where reviews of the plan or the build changed it.

| choice | alternative | why this one | first draft |
|---|---|---|---|
| a granule per pixel per date | a yes/no bit per pixel per date, with the granule guessed from footprints | 4% more storage as a per-shard rank and 16–23% more again on optical as the fusion order, and exact: guessing named the wrong granule for 0.10% of one shard's pixel-dates, wherever footprints overlap | nothing per pixel, ruled out at an estimated 20–45 TB (measured: 3.7 TB for optical); the granule guessed from footprint order |
| a rejected observation keeps its granule, as its fusion order negated | zero for anything the model did not use | a reader can tell "no image had data" from "the model rejected this image here", and see which granule was cloudy; a fully cloudy date keeps its plane | zero |
| planes hold the granule's signed fusion order, `int16` | a signed `int8` rank into each shard-date's `sources` | a pixel joins `granules` directly, with no second table that must agree with the plane on ordering and no cap of 127 granules per shard-date; 33–41% larger overall, about 1.7–2 TB over the campaign, accepted | the rank |
| the copies the ingest dropped listed after the fused granules, with a `status` | the fused granules only | a reader sees the duplicates, unreadable copies and incomplete radar granules the ingest set aside, beside what it used | the fused only |
| a day skipped whole committed off the time axis, alone, when it is skipped | carried on the next day's commit | no state held back between days and nothing to flush when the year ends; the time axis the model reads is untouched | not recorded |
| planes chunked every date × 512 × 512, in 2048 × 2048 shards | one date × one shard | one pixel's year is one fetch instead of 28–130 (0.2 s against 1.9 s, measured at 256 × 256), 7–12% smaller, and a plane chunk covers the same pixels as a v2 store chunk | one date × one shard |
| a per-shard granule list derived from the pixels | every granule whose footprint touches the shard | names only granules that supplied a pixel | |
| the index written by the ingest while it fuses | reading the mask band a second time, or once per granule | one read, one definition of "valid"; per granule would multiply the load by the day's granules, hundreds of radar bursts | each granule loaded separately |
| the granule list inside the mosaic, in the day's own commit | a file written beside the mosaic before the commit | a file written first can describe a day that never committed | the file beside |
| `granules` keyed by mosaic snapshot, written once | a `granules` file per run | the list belongs to the mosaic: one answer per mosaic, and a reader joining through `snapshot_id` never counts a list twice | a file per run |
| a record per run, with `update_pixels` and `source_runs` | one record per shard-year | a holes update mixes two runs in one shard | one per shard-year |
| `update_pixels` marks what a holes update embedded, and its planes are cut to it | the pixels it was asked to embed | one meaning: exactly where a reader takes the update's record; its refused pixels keep the earlier run's record, which covers them | the pixels asked for |
| a fill or shards update records refused pixels too | only embedded pixels | a refused shard still says what imagery it had; the store says what was embedded | |
| `source_runs` on the master only, recomputed at every merge and rebuild from the store's shard commits in the order they landed | also on run file rows, derived from the zone group's `runs` and `updates` records | the records keep a refill's earlier updates and name whichever refill recorded itself last, so they do not say which runs' shards a tile holds; a run file's copy would be a second account, and every merge rewrites the master's anyway | the master only |
| a separate Icechunk store for the planes | arrays inside the published store | each run has its own dates; the published store and its conventions stay untouched | |
| failure to publish fails the zone-year | publish best-effort and log | a zone-year can never be published without its record, and the retry repairs it | best-effort, then a gate |
| one switch, off by default, recorded in the mosaic, mismatches refused | always on | internal and dev stores neither need it nor pay its ingest cost, and dev stores from before the change stay usable | always on, no ingest cost expected |

## 9. Left out

- **Attribution.** The model has none, so the record says what went in, not how much it weighed.
- **The pixel values.** The mosaics are gone; a reader who wants the imagery re-reads the URLs.
- **A separate index per band.** The index follows the band the validity test reads. A granule with
  data in one band and none in another at the same pixel would break this; the products do not
  produce one, and the dev test's pixel checks found none.
- **v1.1**, whose mosaics are gone.
- **Radar quality layers** such as layover and shadow, a separate decision (ADR 033).
- **Days whose imagery reaches no land in the zone.** None of it covered a pixel the store embeds.

## 10. Open questions

1. **Publication time at zone scale,** about an hour and a half per zone-year at the measured rate
   (§6), to be examined separately.
2. **Radar file URLs need Earthdata credentials to fetch.** Is it enough to record them, or should
   the record also point to a public mirror where one exists?
3. **Is anything missing** that a user would need to trust or reproduce an embedding?

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
| `fusion_order` | int16 | position in the day's fusion order, 0 first; null for a dropped copy |
| `granule_id` | string | the catalogue id of the copy |
| `status` | string | `fused`; for a copy dropped before the fuse, `duplicate`, `unreadable` or `missing_bands`; for every copy of a day skipped whole, `low_coverage`, `producer_conflict`, `unreadable` or `missing_bands` |
| `image_key` | string | which image it is a copy of: Sentinel-2 datatake and datastrip (#240), Sentinel-1 burst id (#250) |
| `platform` | string | `S2A`, `S1C` and so on |
| `acquired_at` | timestamp, UTC | the acquisition instant |
| `cloud_cover`, `processing_baseline`, `product_uri` | float64, string, string | Sentinel-2 only |
| `provider`, `collection`, `stac_item_url` | string | where the entry came from, and its catalogue link |
| `assets` | map<string, string> | band to the URL of the file read, as the catalogue published it: radar is read through S3 URIs or signed CloudFront links, and the signed ones carry a credential, so the record keeps the catalogue's (`CATALOG_ASSETS`) |
| `stepped_down` | bool | this copy replaced one that would not read |
| `geometry`, `bbox_west` … `bbox_north` | WKB (GeoParquet 1.1), float64 | the footprint as published, WGS84 |

**`shard_dates`**, one file per run: `run_id`, `tile`, `sensor`, `snapshot_id`, `solar_day`,
`valid_px` (int32, observations the model used), `embedded_px` (int32, per shard, repeated on each
of its rows) and `sources` (list of int16 fusion orders of the granules with data, ascending). Its metadata names the zone, year, run, the snapshot of
each mosaic read, and the refused shards that staged no record.

**`pixel_sources`**: one group per run, `<zone>/<year>/<run_id>`, holding per sensor a
`(date, northing, easting)` int16 array on the zone's grid, chunked every date of the run by 512 × 512 pixels (the v2 store's chunk) in 2048 × 2048 shards, and a
`<sensor>_date` array of its dates; the group records the grid shape, the shard size and, as
`snapshot_<sensor>`, the mosaic snapshot each sensor's planes describe. A value k is the
granule's `fusion_order` plus one, positive when the model used the observation and negative when
it rejected it; 0 means no granule had data. The codec was chosen, before the sign was added, on one
real optical shard-year (33N `chunk_209_14`, 2024): Blosc zstd level 3 with bit
shuffle, 1.17 MB, against 1.38 MB for two bits a pixel (which would cap a shard-date at three
granules), 2.83 MB for zstd without the shuffle, which then misses the mostly constant planes' runs,
and 1.12 MB for a bare validity bit. Bits packed along time and XOR-delta planes also lost. The
sign adds about 4% on optical and 0–5% on radar. In the published layout, the signed fusion
order as `int16` is 16–23% larger than a signed `int8` rank into each shard-date's `sources` on
optical and 34–73% on radar, 33–41% overall, over three shard-years; it was chosen anyway, for one
join fewer on every pixel query (ADR 035, decision 3).

**`update_pixels`**: one boolean `(2048, 2048)` array per shard, named for the shard: the pixels
the holes update embedded.

## Appendix B. Reading the record

The reading guide for users is [`docs/registry.md`](../../docs/registry.md), "Which imagery is
behind an embedding": the helpers, the rule for which run holds a pixel, and the same lookups by
hand in polars and in DuckDB. It is uploaded as the published registry's README, so this plan
does not repeat it.

The helpers are methods of `ProvenanceRecord` in `storage/provenance_reader.py`, plus `locate`;
they answer §2's questions and apply §5's rule. OlmoEarth's documented helpers for its own
provenance are the bar they are meant to clear. A pixel's lookup reads the plane, takes each
value's fusion order and the snapshot its run's group names, and joins `granules` on
`(sensor, snapshot_id, solar_day, fusion_order)`; `shard_dates` serves only the per-shard
question. The guide's polars and DuckDB examples were run against the record the unit tests build,
a fill and a holes update over it, and return the same rows as the helpers for a pixel each run
holds.

## Appendix C. For implementers

**Where the code is.** The switch and the ingest half: `docs/ingest/provenance.md`, with
`ingest/provenance.py`, the per-source reader hook in `stac.py`, and the marks on dropped copies in
`s2_roi.py`. The actor's planes:
`inference/provenance.py`. Publication and its checks: `inference/provenance_publish.py`, called
from `runners/zone_fill.py` on the fill and update paths, before each shard commit. Tables and layout: `storage/provenance.py`. `source_runs`: `published_store.source_runs`,
applied to the master by every registry merge and rebuild. The reader helpers: `storage/provenance_reader.py`.

**Checks at publication**, each failing the run before it commits: every mosaic slot has a granule list; the
three mosaics agree on the switch; every embedded shard staged a record; every staged shard was
read from the snapshots the run publishes; each plane's positive count equals its row's `valid_px`
and its distinct absolute values, less one, are its `sources`;
every `sources` entry is a fused granule of that day in the snapshot the row names; a snapshot's `granules`
file, if an earlier run published it, holds the lists read now; a holes update's mask counts
exactly its `embedded_px` and its planes are zero outside it. Re-running these checks from the
registry census, so a later repair can find a broken zone-year, is not built yet.

**Deploying.** The ingest stamps a fingerprint of its own code on every mosaic, and this change
moves it, so the ingest half deploys while nothing is mid-ingest, as #240 and #250 do. The
campaign's deployment in yield-embeddings turns the switch on, before its first ingest (§4).
