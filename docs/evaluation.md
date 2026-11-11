# Testing a change on small areas before a global run

Producing embeddings for the whole world costs hundreds of thousands of dollars, so a change to the
model or the pipeline should first be shown to work on a few small areas. This page describes a
fixed set of six areas for that purpose. Each area has public ground-truth labels, such as the crop
grown in a field. You produce embeddings for the six areas with the version you want to test, do
the same with the version you already trust, and compare how well each one's embeddings predict
the labels.

The set is meant to be shared: two teams that follow this page get numbers they can compare, and
anyone can propose a better area or a better way to score.

Why these areas were chosen, what each costs, and what was considered and turned down is in
[`context_docs/evaluation/area-selection.md`](../context_docs/evaluation/area-selection.md).

## Words used on this page

| word | meaning |
|---|---|
| version | what is being tested: a branch of the code together with the model it runs (the `model_version` setting, such as `v2-large`) |
| baseline | the version a new one is compared against, normally your main branch. When a tested version is adopted, it becomes the next baseline |
| store | where embeddings are kept: a Zarr store inside an Icechunk repository. Each version gets its own store, holding the six areas and nothing else |
| tile | a 2048 × 2048-pixel square of a store, about 20 km on a side. The cost of a run is roughly proportional to the number of tiles it covers, multiplied by the number of years |
| UTM zone | one of the 60 strips of longitude, each 6° wide, that the store is divided into. The flows name the northern half of zone 31 `utm31n` |
| land cell | a 0.1° square from the published list of land cells (`registry.txt`). The cells you list decide which tiles are processed |
| mosaics | the satellite imagery the pipeline downloads and stacks before running the model |

## What you need

- **The four Prefect flows that produce the global dataset**, set up in your own cloud account:
  `build-land-mask-coverage`, `seed-global-store`, `ingest-zone-year` and `fill-zones-sequential`.
  [`prefect-setup.md`](prefect-setup.md) and [`providers/aws.md`](providers/aws.md) explain how.
  These areas are run through the same flows as the global dataset, not through the single-area
  runner from the quickstart, because a test should exercise the code a global run will use, and
  the two embed different pixels by default ([`single-vs-global.md`](single-vs-global.md)).
- **A version of this library that names zones `utm01` to `utm60`.** Every model from v2 onward
  uses those names, and the steps below rely on them.
- **The labels**, downloaded once. None are stored in this repository.

| labels | where to get them | files used |
|---|---|---|
| PASTIS-R crop types, France | the [PASTIS benchmark](https://github.com/VSainteuf/pastis-benchmark), on Zenodo | `metadata.geojson` (where each patch is, and which group it belongs to) and `ANNOTATIONS/TARGET_<id_patch>.npy` (the crop type of every pixel) |
| Ethiopian crop types | the AlphaEarth evaluation data, [Zenodo record 16585402](https://zenodo.org/records/16585402) | `ethiopia_crops/ethiopia_crops.csv` |
| Canadian crop types | the same record | `canada_crops/canada_crops_coarse/canada_crops_coarse.csv` |
| European land cover and land use (LUCAS) | the same record | `lucas/lucas_lc/lucas_lc.csv`, `lucas/lucas_lu/lucas_lu.csv` |

Each AlphaEarth file has one row per labelled point: its longitude `x` and latitude `y`, its
`label`, whether it is for training or testing (`split`), and the dates the label applies to. Only
the points inside the six areas are used.

## The six areas

Each area is a box of longitude and latitude. The land cells whose centres fall inside a box are
the ones processed. The four French boxes enclose the four Sentinel-2 image tiles that the PASTIS
labels were cut from.

| area | box: west, south, east, north | zone | land cells | tiles | years with labels | year for the first check |
|---|---|---|---|---|---|---|
| Normandy (PASTIS T30UXV) | −1.7, 48.6, −0.1, 49.7 | `utm30n` | 144 | 38 | PASTIS 2019; LUCAS 2017–2018 | 2019 |
| Provence (PASTIS T31TFJ) | 4.2, 43.2, 5.7, 44.3 | `utm31n` | 151 | 44 | the same | 2019 |
| Burgundy (PASTIS T31TFM) | 4.3, 45.9, 5.8, 47.0 | `utm31n` | 165 | 44 | the same | 2019 |
| Alsace (PASTIS T32ULU) | 6.2, 47.7, 7.8, 48.8 | `utm32n` | 176 | 49 | the same | 2019 |
| Ethiopian highlands | 36.2, 5.9, 40.8, 12.0 | `utm37n` | 2,806 | 882 | 2020, and a few points in 2021 | 2020 |
| Southwestern Ontario | −81.0, 43.0, −79.5, 44.0 | `utm17n` | 150 | 42 | every year 2017–2022 | 2018 |
| **all six** | | 5 zones | 3,592 | 1,099 | | |

A version is tested in two steps:

1. **The first check** runs each area for one year, the one in the last column, and compares
   predictions. That is 1,099 tiles. Every version gets this check.
2. **The full check** runs every area for every year from 2017 to 2025, about 9,900 tiles in
   all. Only a version that passes the first check gets it.

## Where everything is kept

These areas get their own input and output locations. Land masks, mosaics and temporary files are
named by zone, not by store, so sharing a location with any other run over the same zones would
mix the two runs' files. The examples use the TESSERA team's locations; replace them with yours.

```
s3://global-tessera-inputs-dev/evaluation/          paths.inputs
├── masks/evaluation.icechunk                        which tiles to process, built once
├── masks/evaluation-registry.txt                    the land cells that was built from
├── models/v2_student_large.pt                       a copy of each model's weights
└── mosaics/, rois/                                  imagery, downloaded again for each run

s3://global-tessera-embeddings-dev/evaluation/      paths.outputs
├── v2-large.icechunk                                the baseline's store
├── v2-large-<change>.icechunk                       one store per version tested
└── staging/                                         temporary, emptied as each zone and year finishes
```

**Every run against a store names it with `global_store_uri`.** Without it, the flows put the store
at `<outputs>/global/tessera.icechunk`. A run that leaves it out writes to that default store
instead, so build the `paths` setting from the version's name in one place:

```json
{"inputs": "s3://global-tessera-inputs-dev/evaluation",
 "outputs": "s3://global-tessera-embeddings-dev/evaluation",
 "global_store_uri": "s3://global-tessera-embeddings-dev/evaluation/v2-large.icechunk"}
```

## Set up the areas (once)

**1. List the land cells.** Copy the lines of the published land-cell list that fall inside the
six boxes, unchanged:

```python
import fsspec
from tessera_embeddings.ingest.land_mask import parse_cell_name

BOXES = [(-1.7, 48.6, -0.1, 49.7), (4.2, 43.2, 5.7, 44.3), (4.3, 45.9, 5.8, 47.0),
         (6.2, 47.7, 7.8, 48.8), (36.2, 5.9, 40.8, 12.0), (-81.0, 43.0, -79.5, 44.0)]

def inside(line: str) -> bool:
    lon, lat = parse_cell_name(line.split()[0])
    return any(w < lon < e and s < lat < n for w, s, e, n in BOXES)

registry = "s3://tessera-embeddings/v1.1/global_0.1_degree_tiff_all/registry.txt"
with fsspec.open(registry, "rt", anon=True) as f:
    keep = [line for line in f if inside(line)]
assert len(keep) == 3592, len(keep)
with fsspec.open("s3://global-tessera-inputs-dev/evaluation/masks/evaluation-registry.txt", "wt") as f:
    f.writelines(keep)
```

**2. Build the land mask** from those cells, for the five zones only. `verify=false` skips checking
each cell against the original image files, which is safe because the lines were copied unchanged
from a list that was checked when it was published:

```bash
prefect deployment run 'build-land-mask-coverage/<deployment>' -p paths='<paths>' \
    -p name=evaluation -p zone_layout=utm_zones -p verify=false \
    -p registry_uri=s3://global-tessera-inputs-dev/evaluation/masks/evaluation-registry.txt \
    -p zones='["utm17n", "utm30n", "utm31n", "utm32n", "utm37n"]'
```

If the areas ever change, build the new mask under a new name. A store records only the name of the
mask it was made with, so two different masks under one name cannot be told apart afterwards.

**3. Copy each model's weights** into `<inputs>/models/`, where the flows look for them:

```bash
aws s3 cp s3://global-tessera-inputs-dev/models/v2_student_large.pt \
          s3://global-tessera-inputs-dev/evaluation/models/v2_student_large.pt
```

## Test a version

**1. Create its store.** The model is chosen here and cannot be changed afterwards.
`optical_min_obs=1` embeds every pixel with at least one clear Sentinel-2 observation in the year,
as the global dataset does:

```bash
prefect deployment run 'seed-global-store/<deployment>' -p paths='<paths>' \
    -p model_version=v2-large -p optical_min_obs=1
```

A version that changes the code but keeps the model's weights still uses `model_version=v2-large`.
New weights first need their own `model_version` in this library, as any new model does before a
global run.

**2. Run the first check** from the Prefect deployment of the version's code branch.
`allow_s2_only=true` embeds pixels that have Sentinel-2 observations but no Sentinel-1 radar, as
the global dataset does:

```bash
prefect deployment run 'fill-zones-sequential/<deployment>' -p paths='<paths>' \
    -p mask_name=evaluation -p allow_s2_only=true \
    -p cells='[["utm30n", 2019], ["utm31n", 2019], ["utm32n", 2019], ["utm37n", 2020], ["utm17n", 2018]]'
```

**3. Score it** against the baseline, as described below. The first time, the baseline needs this
check run on its own store too.

**4. If it passes, run the full check:** the same command, with `cells` listing each of the five
zones with every year from 2017 to 2025. Then score again.

**Test one version at a time.** Versions share the downloaded imagery, and each run deletes a zone's
imagery for a year once that zone and year are finished (the `cleanup_mosaics` setting, on by
default). Downloading it again for every run costs less than keeping it, unless you test more than
two or three versions a month; the figures are in the context document.

## Scoring

This repository does not include code for these steps. They are written down so that every
implementation computes the same thing.

**Read each label's embedding** from the pixel that contains its point, in the year the label
applies to, from both stores. [`global-store.md`](global-store.md) shows how to open a store and turn
its stored integers back into embedding values. Zone `utm31n` is in the store's group `utm31`, with
`x` and `y` in the map projection EPSG:32631. Leave out any label whose pixel has no embedding in
either store, and report how many were left out.

- **PASTIS:** use pixels from each patch's `TARGET` file, leaving out class 0 (background) and
  class 19 (unlabelled). Take at most 50 pixels per patch, chosen at random with a fixed seed.
  PASTIS splits its patches into five groups (`fold`): train on groups 1–3 and test on group 5.
- **The AlphaEarth files:** use every point inside the areas, with the `split` it comes with.

**Score each set of labels** by training a logistic-regression classifier on the training points'
embeddings and measuring it on the test points. Report the balanced accuracy, which is the average
over classes of the share of each class's test points predicted correctly. Leave out any class with
fewer than 10 training points inside the areas.

**A version passes the first check** if, for every set of labels, its balanced accuracy is not
clearly lower than the baseline's. To decide "clearly", repeat the comparison 1,000 times, each
time on test points drawn at random with replacement, scoring both versions on the same draw. Sort
the 1,000 differences (new version minus baseline). If the 25th smallest is below zero, the drop is
within chance and the version passes.

**Check how many pixels have embeddings.** For each area and year, report the share of pixels
inside the mask that hold an embedding, in both stores. If a version embeds more than one
percentage point fewer pixels than the baseline, find out why before adopting it, even if its
accuracy passes.

**The full check adds two things.** Score every set of labels in every year it has labels. And for
each area, report how similar each pixel's embedding is from one year to the next (the median
cosine similarity between consecutive years), for both versions. A large difference from the
baseline is worth explaining, but it is not by itself a failure.

**The published v1.1 dataset is a useful extra comparison**, though not a pass mark, since a
different model is expected to score differently. Its
[Source Coop copy](global-store.md#without-icechunk) uses the same `utmNN` groups and `x` and `y`
coordinates as the stores made here.

## What it costs

For the v2 Large model, the first check costs about $190 and the full check about $1,700, mostly
for GPU time. These are estimates from the global run's measured cost per tile and year, not from
running these areas; the context document shows the working.

## What this does not tell you

- **The areas are all farmland**, in temperate Europe, temperate North America and the Ethiopian
  highlands. Rainforest, desert, snow and cities are not tested.
- **2017 has much less imagery.** The second Sentinel-2 satellite was not yet imaging routinely, so
  every version scores lower in 2017. Compare 2017 only with 2017.
- **Most years have no labels in France and Ethiopia.** France has crop labels only for 2019 and
  LUCAS points for 2017 and 2018; nearly all of Ethiopia's labels are for 2020. In the other years,
  the full check can show only how much the embeddings change, not whether they are right.
- **It does not test the single-area runner**, which these steps do not use.
