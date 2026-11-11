# Evaluating a change before a global run

A global run costs hundreds of thousands of dollars, so a change to the model or the pipeline
should prove itself on a small area first. This page describes a fixed **evaluation suite**: six
small areas with public ground-truth labels, each embedded by a candidate and scored against the
same areas embedded by a reference. The suite is meant to be shared, so that two teams who run it
get numbers they can compare, and so that anyone can propose a better area or a better score.

Why these areas, what each costs, and what was considered and turned down is in
[`context_docs/evaluation/area-selection.md`](../context_docs/evaluation/area-selection.md).

## Words used on this page

| word | meaning |
|---|---|
| variant | one version of the pipeline being evaluated: a code branch together with the model it runs (`model_version`) |
| reference | the variant every candidate is scored against, normally your main branch. The variant that ships becomes the next reference |
| evaluation store | one Icechunk store per variant, laid out like the global store, holding embeddings for the evaluation areas only |
| area | one box of the suite, in longitude and latitude |
| fill zone | the half of a UTM zone the global flows fill as one unit, such as `utm31n` |
| tile | a 2048 × 2048-pixel square of a store, about 20 km on a side at 10 m pixels. Cost scales with tiles |
| gate run | an evaluation of each area in its one label year. Every candidate gets one |
| full run | an evaluation of every area in every year from 2017 to 2025. Only candidates that pass the gate get one |

## What you need

- **A deployment of the global flows**: `build-land-mask-coverage`, `seed-global-store`,
  `ingest-zone-year` and `fill-zones-sequential`. The suite runs on the global path, not the
  single-area runner, because the global path is what a global run uses, and the two embed
  different pixels by default ([`single-vs-global.md`](single-vs-global.md)). Setting one up is
  covered in [`prefect-setup.md`](prefect-setup.md) and [`providers/aws.md`](providers/aws.md).
- **A deployment that writes the `utm01` to `utm60` zone layout.** Every model from v2 onward uses
  it, and the fill zones below are named in it.
- **The labels**, downloaded once. None are stored in this repository.

| label set | where to get it | files used |
|---|---|---|
| PASTIS-R crop types, France | the [PASTIS benchmark](https://github.com/VSainteuf/pastis-benchmark), on Zenodo | `metadata.geojson` (patch footprints and folds) and `ANNOTATIONS/TARGET_<id_patch>.npy` (crop type per pixel) |
| Ethiopia crop types | the AlphaEarth evaluation set, [Zenodo record 16585402](https://zenodo.org/records/16585402) | `ethiopia_crops/ethiopia_crops.csv` |
| Canada crop types | the same record | `canada_crops/canada_crops_coarse/canada_crops_coarse.csv` |
| LUCAS land cover and land use, Europe | the same record | `lucas/lucas_lc/lucas_lc.csv`, `lucas/lucas_lu/lucas_lu.csv` |

Every AlphaEarth file holds `x` and `y` in degrees, a `label`, a train or test `split`, and the
period the label is valid for. Only the points inside the areas below are used.

## The areas

Each box selects the land cells (0.1° squares in the global land registry) whose centres fall
inside it. The PASTIS boxes enclose the four Sentinel-2 tiles the PASTIS patches were cut from.

| area | box: west, south, east, north | fill zone | land cells | tiles | labels | gate year |
|---|---|---|---|---|---|---|
| PASTIS T30UXV, Normandy | −1.7, 48.6, −0.1, 49.7 | `utm30n` | 144 | 38 | PASTIS 2019; LUCAS 2017–18 | 2019 |
| PASTIS T31TFJ, Provence | 4.2, 43.2, 5.7, 44.3 | `utm31n` | 151 | 44 | the same | 2019 |
| PASTIS T31TFM, Burgundy | 4.3, 45.9, 5.8, 47.0 | `utm31n` | 165 | 44 | the same | 2019 |
| PASTIS T32ULU, Alsace | 6.2, 47.7, 7.8, 48.8 | `utm32n` | 176 | 49 | the same | 2019 |
| Ethiopian highlands | 36.2, 5.9, 40.8, 12.0 | `utm37n` | 2,806 | 882 | Ethiopia crops 2020–21 | 2020 |
| Southwestern Ontario | −81.0, 43.0, −79.5, 44.0 | `utm17n` | 150 | 42 | Canada crops 2017–22 | 2018 |
| **all** | | 5 fill zones | 3,592 | 1,099 | | |

A gate run is therefore about 1,100 tile-years and a full run about 9,900.

## Where everything lives

The suite gets its own `inputs` and `outputs` roots. Land masks, mosaics and staged tiles are
named by zone, not by store, so a shared root would mix the suite's files with any other run over
the same zones. The examples use the paths the TESSERA team uses; replace them with your own.

```
s3://global-tessera-inputs-dev/evaluation/          paths.inputs
├── masks/evaluation.icechunk                        the suite's land mask, built once
├── masks/evaluation-registry.txt                    the land cells it was built from
├── models/v2_student_large.pt                       a copy of each model's checkpoint
└── mosaics/, rois/                                  imagery, rebuilt for each run

s3://global-tessera-embeddings-dev/evaluation/      paths.outputs
├── v2-large.icechunk                                the reference's evaluation store
├── v2-large-<change>.icechunk                       one store per candidate
└── staging/                                         transient, emptied as each zone-year lands
```

**Each store is addressed with `global_store_uri`**, which replaces the derived
`<outputs>/global/tessera.icechunk` with the URI you give. Every run against a store must pass the
same value, so build `paths` from the variant's name in one place:

```json
{"inputs": "s3://global-tessera-inputs-dev/evaluation",
 "outputs": "s3://global-tessera-embeddings-dev/evaluation",
 "global_store_uri": "s3://global-tessera-embeddings-dev/evaluation/v2-large.icechunk"}
```

## Set up the suite (once)

**1. Write the land cells.** Copy the lines of the global land registry whose cells fall inside the
boxes, unchanged:

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

**2. Build the land mask** from those cells, for the five fill zones only. `verify=false` because
the lines are copied unchanged from a registry that was verified when it was published:

```bash
prefect deployment run 'build-land-mask-coverage/<deployment>' -p paths='<paths>' \
    -p name=evaluation -p zone_layout=utm_zones -p verify=false \
    -p registry_uri=s3://global-tessera-inputs-dev/evaluation/masks/evaluation-registry.txt \
    -p zones='["utm17n", "utm30n", "utm31n", "utm32n", "utm37n"]'
```

Never rebuild a mask under the same name with different cells: a store records only the mask's
name, so a changed mask cannot be told apart afterwards. A changed suite gets a new mask name.

**3. Copy each model's checkpoint** into `<inputs>/models/`. The fill flows read the checkpoint
from there, under the model's fixed filename:

```bash
aws s3 cp s3://global-tessera-inputs-dev/models/v2_student_large.pt \
          s3://global-tessera-inputs-dev/evaluation/models/v2_student_large.pt
```

## Evaluate a variant

**1. Seed its store.** The model is chosen here and cannot change afterwards, and the optical
line matches the campaign's:

```bash
prefect deployment run 'seed-global-store/<deployment>' -p paths='<paths>' \
    -p model_version=v2-large -p optical_min_obs=1
```

A variant that changes code but not weights keeps `model_version=v2-large`. New weights first need
their own model entry in the library, as any new model does before a global run.

**2. Run the gate** on the deployment of the variant's branch, with the campaign's setting for
pixels that have no radar:

```bash
prefect deployment run 'fill-zones-sequential/<deployment>' -p paths='<paths>' \
    -p mask_name=evaluation -p allow_s2_only=true \
    -p cells='[["utm30n", 2019], ["utm31n", 2019], ["utm32n", 2019], ["utm37n", 2020], ["utm17n", 2018]]'
```

**3. Score it** against the reference's gate run (see below). The reference needs a gate run of
its own the first time.

**4. If it passes, run the full nine years** with the same command, listing each of the five fill
zones with every year from 2017 to 2025, and score again.

**Run one variant at a time.** Variants share the mosaics under `inputs`, and each fill deletes
the mosaics of a zone-year once that zone-year lands (`cleanup_mosaics`, on by default). Rebuilding
them each run costs less than storing them, unless you evaluate more than two or three candidates
a month; the comparison is in the context doc.

## Scoring

No implementation of this procedure ships in this repository; it is written down here so that
every implementation computes the same thing.

**Read each label's embedding** from the pixel that contains its point, in the year its label is
valid for, from the candidate's store and from the reference's. Open a store and dequantise as in
[`global-store.md`](global-store.md); fill zone `utm31n` lives in group `utm31`, with `x` and `y`
in EPSG:32631. A label whose pixel has no embedding in either store is left out of both, and
counted.

- **PASTIS:** pixels from each patch's `TARGET` raster, leaving out class 0 (background) and
  class 19 (void), at most 50 per patch, drawn with a fixed seed. Folds 1–3 train and fold 5 tests.
- **The AlphaEarth sets:** every point inside the areas, with the `split` it comes with.

**Score each label set** with a logistic-regression classifier fitted on the train points' embeddings
and evaluated on the test points, reported as balanced accuracy (the mean of the per-class recalls).
Leave out classes with fewer than 10 train points inside the areas.

**A candidate passes the gate** when, for every label set scored in the gate years, its balanced
accuracy is not lower than the reference's by more than the 95% interval of the difference. The
interval comes from a paired bootstrap over test points: resample them with replacement, score both
variants on each resample, and take the 2.5th percentile of the candidate's score minus the
reference's. Below zero is a pass.

**Check coverage as well.** For each area and year, report the share of the mask's pixels holding
an embedding, for both variants. A drop of more than one percentage point needs an explanation
before the candidate ships, even when its accuracy passes.

**A full run adds change over time.** Score every label set in every year it has labels. For
each area, report the median cosine similarity between each pixel's embeddings in consecutive years,
for both variants; a large difference from the reference is a finding to explain, not a failure.

**The published v1.1 store gives a further comparison**, not a pass line: a different model
embeds differently. Its [Source Coop copy](global-store.md#without-icechunk) uses the same `utmNN`
groups and `x` and `y` coordinates as an evaluation store.

## What it costs

On v2 Large, a gate run costs roughly $190 and a full run roughly $1,700, mostly graphics cards.
Both figures come from the campaign's measured cost per tile-year rather than a run of the suite;
the derivation is in the context doc.

## What the suite does not tell you

- **It covers cropland in temperate Europe, temperate North America and the East African
  highlands.** Rainforest, desert, snow and cities are not scored.
- **2017 is thin.** Sentinel-2B was not yet routinely imaging, so 2017 has far fewer clear
  observations than later years, and every variant scores lower for it. Compare 2017 with 2017.
- **Most of the nine years carry no labels** in France and Ethiopia, so there the full run shows
  only stability, not accuracy.
- **It says nothing about the single-area runner**, which this suite does not use.
