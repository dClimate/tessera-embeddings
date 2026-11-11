# Choosing the evaluation suite's areas

**The suite is six boxes in France, Ethiopia and Ontario, chosen because they hold the densest
public crop labels that small areas can carry; a gate run costs about $190 and a full nine-year
run about $1,700 per variant on v2 Large.** The procedure is in
[`docs/evaluation.md`](../../docs/evaluation.md). This page records why these areas, why the suite
is shaped as it is, and how the costs were estimated, as decided on 2026-11-11.

## The constraint that decided everything

Cost scales with 2048-pixel tiles (about 20 km square), not with labelled points: a tile holding
one label costs the same as a tile holding a thousand. A useful evaluation area is therefore one
where labels are **dense per tile**, and most published benchmarks are built the other way, spread
thinly over a continent or the globe so that a model is tested everywhere.

## Candidates

**The AlphaEarth evaluation set** ([Zenodo 16585402](https://zenodo.org/records/16585402), fifteen
label sets) was measured by binning each set's points into squares of 0.185° (about one tile):

| label set | points | squares holding points | points per square | squares holding half the points |
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

Only three are usable at small-area cost:

- **`ethiopia_crops`** is compact: half its points sit in 17 squares, all in UTM zone 37. It is
  82% wheat, and maize and teff lie mostly outside the dense core.
- **`canada_crops_coarse`** has one dense pocket: 1,314 points in 81–79.5°W, 43–44°N
  (southwestern Ontario), nine classes, labelled in every year from 2017 to 2022. It is the only
  label set with labels in six different years of one small area.
- **LUCAS** is too thin to choose an area by, but it falls inside the PASTIS boxes for free: 2,473
  land-cover and 2,692 land-use points.

**PASTIS-R**, supplied as `pastis_r_roi.geojson`, holds 2,433 patches of 128 × 128 pixels with
crop types for every pixel, all from 2019, cut from four Sentinel-2 tiles (T30UXV, T31TFJ, T31TFM,
T32ULU). The patches touch 135 of the 144 tiles under those four Sentinel-2 tiles, so cutting the
area down to the patches saves nothing; the suite takes the whole Sentinel-2 tiles.

## Decisions

- **v2 only.** v1.1 already exists for every area and year in the published store, so the suite
  generates v2 and later models only.
- **French Guiana is held out.** No labels came with it, so it could only test the cloudy tropics
  by inspection. It returns if a reference such as a land-cover or canopy-height map is chosen.
- **Ethiopia is one bounding box around every point** (882 tiles), rather than only the tiles
  holding points (about 185). The box costs about $1,000 more per full run, and in exchange every
  area is a plain box that anyone can reproduce from the coordinates in the runbook, with
  continuous coverage across the highlands for maps and the stability check.
- **Label years first.** Every candidate gets a gate run of each area's one label year; only a
  candidate that passes gets the full nine years. France and Ethiopia have labels in only one year,
  so the other years test stability, not accuracy.
- **One store per variant, laid out like the global store.** Three layouts were considered:
  - *One store holding all variants as named groups:* the global flows write only UTM-zone groups,
    so this needs code changes, ruled out for a need this narrow.
  - *Separate single-area stores per area:* six per variant, each needing a grid hand-snapped to the zone's pixel
    lattice to line up with the reference. Worse, the single-area runner drops pixels without radar
    and keeps dates at 5% rather than 0.1% valid coverage by default, and it has no flow-level
    tests, so it would validate a path a global run does not use.
  - *A global-layout store per variant (chosen):* tiles the mask does not reach are never written,
    so a store covering six small boxes is small. The fill path is the one a global run uses, and
    a reader finds both models by the same `utmNN` group and coordinates.
- **The store is addressed with `global_store_uri`**, which drops the `global/` directory from its
  path with no code change.
- **Checkpoints are copied** into the suite's `inputs` root, because the fill flows read
  `<inputs>/models/<filename>` and nothing else. Adding a separate models location to the path
  settings was turned down as code for a narrow need.
- **The suite waits for the `utm01`–`utm60` zone layout** ([ADR 037](../decisions/037-v2-stores-use-the-utm-zones-layout.md)),
  which every model from v2 onward uses. A store seeded before that layout ships would test a
  layout no global v2 run will use.

## How the costs were estimated

No part of the suite has been run, so every figure scales a measured campaign rate.

**Embeddings.** The v2 campaign estimate
([cost model §13](../campaign/campaign-cost-model.md)) is about $560,000 for 3.25 million
tile-years, all lines included: $0.172 per tile-year.

| run | tile-years | cost |
|---|---|---|
| gate (one label year per area) | 1,099 | ~$190 |
| full (2017–2025) | 9,891 | ~$1,700 |
| Ethiopia alone, full, as a box | 7,938 | ~$1,370 |
| Ethiopia alone, full, tiles with points only | ~1,665 | ~$290 |

Tile counts come from the land registry: each box's land cells, projected into the fill zone's
UTM grid, counting every 20,480 m tile a cell touches. The campaign rate averages over the whole
globe, and the suite departs from that average in ways that push both directions: five zones each
pay a fixed start-up cost, while all six areas are fully land. Treat the figures as ±30%.

**Keeping mosaics between runs.** The campaign ingested 14.04 PB of mosaics for 3.25 million
tile-years ([§12c](../campaign/campaign-cost-model.md)), about 4.3 GB per tile-year, and ingest
(containers and requests) came to about $0.038 per tile-year.

| | gate | full |
|---|---|---|
| mosaics | ~4.7 TB | ~42 TB |
| storing them, per month | ~$110 | ~$980 |
| rebuilding them, per run | ~$42 | ~$375 |

Rebuilding is cheaper unless more than two or three candidates are evaluated a month, and a
candidate that changes ingest needs fresh mosaics anyway. The runbook keeps the default of
deleting them.

## Not yet settled

- **Scoring has no implementation.** The runbook defines the procedure; the TESSERA team's
  implementation will live with its deployment code.
- **Licences of the label sets** were not checked; the suite links to them rather than copying
  them.
- **Radar coverage of the areas** was not checked. An area without Sentinel-1 in some year scores
  on optical-only pixels there.
- **159 Ethiopian points are labelled in 2021**, outside the gate year. They are scored only in a
  full run.
