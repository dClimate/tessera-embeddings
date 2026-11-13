# Choosing the areas for testing changes

**Changes are tested on six boxes in France, Ethiopia and Ontario, chosen because they hold the
most public crop labels for the least area. For the v2 Large model, testing a version in one year
per area costs about $190, and in all nine years about $1,700.** The steps are in
[`docs/evaluation.md`](../../docs/evaluation.md). This page records why these areas, why the
results are stored as they are, and how the costs were estimated. The decisions were made on
2026-11-11.

## Why label density decides everything

The cost of a run is set by how many 2048-pixel tiles (about 20 km square) it covers, not by how
many labelled points they hold: a tile with one label costs the same as a tile with a thousand.
A good test area therefore has many labels per tile. Most published benchmarks are built the other
way, with labels spread thinly over a continent or the world, so that a model is tested everywhere.

## The labels considered

**The AlphaEarth evaluation data** ([Zenodo 16585402](https://zenodo.org/records/16585402)) holds
fifteen sets of labels. To compare them, each set's points were counted in squares of 0.185°,
about the size of one tile:

| labels | points | squares holding points | points per square | squares holding half the points |
|---|---|---|---|---|
| `ethiopia_crops` | 2,530 | 185 | 13.7 | 17 |
| `lucas_lu` | 226,858 | 16,937 | 13.4 | 4,550 |
| `lucas_lc` | 203,569 | 16,905 | 12.0 | 4,479 |
| `canada_crops_coarse` | 16,079 | 1,535 | 10.5 | 262 |
| `canada_crops_fine` | 14,566 | 1,501 | 9.7 | 263 |
| `us_trees` | 45,382 | 12,021 | 3.8 | 1,955 |
| `openet_ensemble` | 35,683 | 14,595 | 2.4 | 3,989 |
| `descals` | 17,477 | 9,139 | 1.9 | 2,760 |
| `lcmap_lc`, `lcmap_lu` | ~26,500 | ~16,400 | 1.6 | ~5,100 |
| `glance` | 34,885 | 28,056 | 1.2 | 10,614 |
| `aster_ged`, `africa_crop_mask`, `lcmap_lcc`, `lcmap_luc` | 991–17,636 | — | 1.0–1.1 | — |

Only three of these can be tested cheaply:

- **`ethiopia_crops`** is compact: half its points lie in 17 squares, all in UTM zone 37. It is
  82% wheat, and most of the maize and teff points lie away from the densest squares.
- **`canada_crops_coarse`** has one dense patch: 1,314 points in 81–79.5°W, 43–44°N, in
  southwestern Ontario, across nine classes and every year from 2017 to 2022. It is the only set
  with labels in six different years within one small area.
- **LUCAS** is too spread out to choose an area by, but 2,473 of its land-cover points and 2,692 of
  its land-use points fall inside the French boxes, which are run anyway.

**PASTIS-R**, supplied as `pastis_r_roi.geojson`, holds 2,433 patches of 128 × 128 pixels with the
crop type of every pixel, all for 2019. They were cut from four Sentinel-2 image tiles (T30UXV,
T31TFJ, T31TFM and T32ULU). The patches touch 135 of the 144 tiles under those image tiles, so
restricting the run to the patches would save almost nothing; the whole image tiles are used.

## Decisions

- **Only v2 and later models are run.** The published dataset already holds v1.1 for every area and
  year.
- **French Guiana is left out for now.** It came with no labels, so it could only be judged by eye.
  It can return if a reference map, such as land cover or canopy height, is chosen for it.
- **Ethiopia is one box around every point** (882 tiles), rather than only the tiles that hold
  points (about 185). The box costs about $1,000 more for all nine years. In return, every area is
  a plain box that anyone can reproduce from the coordinates in the steps, and the highlands are
  covered without gaps, which helps when looking at maps and when measuring year-to-year change.
- **Each version is run in one year per area**, the year the area's labels mostly fall in. Running
  all nine years is optional and not expected as normal practice: France has crop labels only for
  2019 and LUCAS points for 2017 and 2018, and nearly all of Ethiopia's labels are for 2020, so most
  of the nine years can show only how much embeddings change, not whether they are right, at nine
  times the cost.
- **Each version gets its own store, laid out like the global dataset.** Three layouts were
  considered:
  - *One store holding every version, each under its own name:* the flows that write the global
    dataset name groups only by UTM zone, so this would need code changes. Turned down as too much
    code for a narrow need.
  - *A separate single-area store for each area:* six stores per version, each needing a grid
    aligned by hand to the zone's pixels so that it lines up with the baseline. The single-area
    runner also skips pixels without radar and drops more cloudy dates by default (it keeps a date
    only if 5% of the area is clear, against 0.1% for the global run), and its end-to-end path has
    no automated tests. It would test a path a global run does not use.
  - *One store per version, laid out like the global dataset (chosen):* tiles outside the mask are
    never written, so a store holding six small boxes stays small. The runs use the same flows as a
    global run, and both models' embeddings are read from the same zone groups and coordinates.
- **Each store's location is set with `global_store_uri`**, which keeps the `global/` directory out
  of its path without any code change.
- **Model weights are copied** into the areas' own input location, because the flows look for them
  only under `<inputs>/models/`. Adding a separate setting for where models live was turned down as
  code for a narrow need.
- **These runs wait for the `utm01`–`utm60` zone names**
  ([ADR 037](../decisions/037-v2-stores-use-the-utm-zones-layout.md)), which every model from v2
  onward uses. A store created with the older zone names would test a layout no global v2 run will
  use.

## How the costs were estimated

None of these areas has been run yet, so every figure scales a cost measured on the global run.

**Producing embeddings.** The estimate for a nine-year v2 global run
([cost model §13](../campaign/campaign-cost-model.md)) is about $560,000 for 3.25 million tiles
counted once per year, everything included: $0.172 per tile per year.

| run | tiles × years | cost |
|---|---|---|
| one year per area | 1,099 | ~$190 |
| all nine years | 9,891 | ~$1,700 |
| Ethiopia alone, nine years, as one box | 7,938 | ~$1,370 |
| Ethiopia alone, nine years, only tiles holding points | ~1,665 | ~$290 |

The tile counts come from the published land-cell list: each box's land cells, placed on the UTM
zone's grid, counting every 20,480 m tile a cell touches. The global rate is an average over the
whole world, and these areas differ from it in both directions: each of the five zones pays a fixed
cost to start, while every area is entirely land. Treat the figures as accurate to about ±30%.

**Keeping the downloaded imagery between runs.** The global run downloaded 14.04 PB of imagery for
its 3.25 million tile-years ([§12c](../campaign/campaign-cost-model.md)), about 4.3 GB each, and
downloading it cost about $0.038 per tile per year.

| | one year per area | all nine years |
|---|---|---|
| imagery | ~4.7 TB | ~42 TB |
| keeping it, per month | ~$110 | ~$980 |
| downloading it again, per run | ~$42 | ~$375 |

Downloading again is cheaper unless more than two or three versions are tested a month, and a
version that changes how imagery is downloaded needs a fresh download anyway. The steps therefore
keep the flows' default of deleting imagery after each run.

## The labels' licences

Every set of labels may be used, changed and shared, including commercially, as long as its source
is credited. Checked on 2026-11-12 against the Zenodo records and the licence section of each
AlphaEarth label set's README.

| labels | licence |
|---|---|
| PASTIS-R ([Zenodo 5735646](https://zenodo.org/records/5735646)) | CC BY 4.0 |
| The AlphaEarth evaluation data as a whole ([Zenodo 16585402](https://zenodo.org/records/16585402)) | CC BY 4.0 (its code: Apache 2.0) |
| `ethiopia_crops` (Ethiopian Crop Type 2020) | CC BY 4.0 |
| `canada_crops_coarse` (AAFC Annual Crop Inventory ground truth) | Open Government Licence – Canada |
| `lucas_lc`, `lucas_lu` (LUCAS Harmonised, JRC) | CC BY 4.0 |

The steps link to the labels rather than copying them, so this repository redistributes none of
them. The LUCAS points are the survey's planned locations, not where the surveyor stood; at 10 m
a few of them will fall in the neighbouring field.

## Radar over the areas

Every area has Sentinel-1 radar in every year from 2017 to 2025, but only France has it from both
orbit directions. Checked on 2026-11-12 by asking NASA's catalogue (CMR) for OPERA RTC-S1
granules with the filters the ingest sends the catalogue: the same collection, orbit direction and
VV polarisation. The ingest also needs VH, which it checks itself and this count does not. Each
area was split into 1° squares, and an orbit was counted only with at least 12 granules in the
year. A granule covers one strip of one pass, so this shows radar in each square, not at every
pixel.

| area | 2017–2025 |
|---|---|
| Normandy, Provence, Burgundy, Alsace | both orbits in every square and year |
| Ethiopia | in 2018–2023, only descending passes in about 28 of 35 squares, both orbits in the rest; most squares have both in 2017, 2024 and 2025 |
| Ontario | ascending passes only, every year (about 1,300 granules a year, no descending) |

So the six areas also test how a version handles radar from one orbit direction, which is common
across the world: Ontario has only ascending passes, and most of the Ethiopian box has only
descending ones in its tested year, 2020. No area is without radar, so none tests a place made
from Sentinel-2 alone, though single pixels without radar may still occur within them.

## First run: Ontario, 2018

On 2026-11-13 the steps were run on dev for Ontario alone in 2018, the 42 tiles of zone `utm17n`,
with the v2 Large model, and the store was scored against itself.

- **The steps worked as written.** The mask came out at 150 cells and 42 tiles, the store was
  created at the location given by `global_store_uri`, and the fill wrote there and nowhere else.
  The fill took 31 minutes, 25 of them for the cell itself.
- **The first attempt found a bug in the fill.** `fill-zones-sequential` refused every v2 store,
  because the run ID it gives its shared inference session lacked the v2 prefix. It failed after
  42 minutes of downloading imagery. PR #271 fixed it.
- **Scoring the store against itself** gave every difference as exactly zero and a pass, and 100%
  coverage. Reading the 316 labels from both stores took under two minutes from a laptop.
- **The labels land on the right pixels.** With the `canada_crops_coarse` labels, v2 scores a
  balanced accuracy of 0.924 against 0.333 for chance. The published v1.1 dataset, read at the
  same points, scores 0.903.
- **The pass rule fails a worse version.** Scrambling the embeddings across points gives 0.353,
  and moving every point 1 km east gives 0.358. Both fail, with the 975th smallest difference
  at −0.51 and −0.49.

**Ontario in one year is a weak test.** Of its 316 points in 2018, the AlphaEarth split puts 69
in training and 247 in testing. Only 3 of its 7 classes have the 10 training points needed to be
kept, which leaves 49 training points and 240 test points. It can catch a large drop in quality but not a small one.

## Not yet settled

- **No public code computes the scores.** The steps describe what to compute; the TESSERA team's
  implementation lives with its private deployment code.
- **159 Ethiopian points are labelled for 2021**, not 2020, so only the nine-year run scores them.
