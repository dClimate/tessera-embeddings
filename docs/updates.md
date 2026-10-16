# Updating part of a store

An **update** runs the model again on chosen tiles of an embeddings store you have already filled,
and writes nothing else. Use one when new imagery has arrived for a period the store already
covers, when some pixels were left empty, or when a fix means some tiles must be redone. Every tile
you did not choose is left exactly as it was.

This page covers both kinds of store:

- a **single-area store**, made by the single-area runner from a config file, as in the
  [quickstart](quickstart.md);
- the **global store**, made by the global campaign's Prefect flows
  ([`global-store.md`](global-store.md)). You can only update a global store your own campaign
  wrote: the store dClimate publishes is read-only to everyone else.

Why updates work the way they do is in
[ADR 027](../context_docs/decisions/027-targeted-updates.md), an architecture decision record.

## Words used on this page

| word | meaning |
|---|---|
| store | where the embeddings are kept: a Zarr store inside an Icechunk repository. Like git, Icechunk keeps past versions as snapshots, and a **tag** is a name pinned to one snapshot |
| tile | a 2048 × 2048-pixel square of a store, named `chunk_<row>_<column>`. Row and column count from 0 at the top-left corner of the area (single-area store) or of the UTM zone (global store). Code and messages also call a tile a *shard* or a *chunk* |
| embedding arrays | the embedding itself and the seven arrays stored beside it: its scale, and the observation counts and month coverage it was built from |
| hole | a pixel with no embedding, often because it had too few satellite observations |
| filled | holding embeddings: a single-area store once its first run finishes, a zone-year of the global store once the campaign has run it |
| time window | the 12 months a single-area run covers, ending in the config's `time_window_end` month. An update redoes a window the store already holds; to cover a new period, run the config without `update` |
| zone-year | one UTM zone in one calendar year, the unit the global campaign fills. `18S` is zone 18, southern hemisphere |
| mosaics | the satellite imagery the pipeline downloads and stacks before running the model |
| registry | a table with one row per tile and year, which the global campaign writes beside the store, recording what each tile holds ([`registry.md`](registry.md)) |

## Before you start

- **Run every command from the repository root, with its virtual environment active**
  (`source .venv/bin/activate`; see [`environment-setup.md`](environment-setup.md)). Use `python`
  directly rather than `uv run`, as the quickstart explains.
- **On AWS, sign in with a named profile or single sign-on** (`AWS_PROFILE`). This library opens
  stores with those credentials and ignores AWS keys exported as environment variables.
- **Find your paths.** Every location on this page comes from the run's `paths` setting:
  `paths.inputs` holds imagery, land masks and update plans, and `paths.outputs` holds stores. For a
  single-area run they are in the config file. For the global flows they are the deployment's
  `paths` parameter: `prefect deployment inspect 'fill-zones-sequential/<deployment>'` prints it if
  it was set when the deployment was registered, and otherwise you pass it when you run a flow, as
  `-p paths='{"inputs": "s3://my-inputs", "outputs": "s3://my-outputs"}'`. `prefect deployment ls`
  lists the deployment names.
- **Replace the example values** below (`s3://my-inputs`, `amazon-thin-2024`, `18S`, and so on) with
  your own.

## The update's name ties everything together

Every update has a **name** that you choose, such as `more-imagery-2025` or `fix-tile-edges`. A
name may use lowercase letters, digits and hyphens, must start with a letter or a digit, and can be
up to 63 characters long.

**An update runs once per name.** Running the same name again finishes it if it stopped part way,
and otherwise does nothing. To run another update, choose a new name.

| | single-area store | global store |
|---|---|---|
| you choose the name in | `update.name`, in the config file | `--name`, when you write the plan |
| you start the update with | the same config file | `update=<name>` on a global flow. This is only the name: the plan holds everything else |
| the tiles and mode are set in | the same `update` block | the plan file, `{paths.inputs}/updates/<name>/plan.json` |
| afterwards the store has | a record named after the update in its `updates` attribute, and the tag `update-<name>-base`: the store as it was before | the same record in each zone it updated, the same tag, and `update-<name>-complete` once every planned zone-year is done |
| afterwards the registry has | nothing: single-area stores have no registry | the name in the `update_name` column of every tile the update rewrote |

## Choose a mode: holes or shards

| mode | what runs on the GPU | what is written | use it for |
|---|---|---|---|
| `holes` (the default) | only the holes | the holes, in all the embedding arrays; every other pixel keeps its exact values | new imagery, an outage, pixels a lower depth rule now admits (see "To embed pixels the depth rule left empty" below) |
| `shards` | every pixel of each chosen tile | each chosen tile, whole | a fix to ingest or inference, where existing values are wrong |

If nothing new can be embedded in a chosen tile, that tile is left as it was.

**Name the tiles for a `shards` update.** Without a tile list it rewrites every tile: the whole
area, or every tile of each zone-year.

## Update a single-area store

You need the config file of the run that made the store, and the store itself, at
`{paths.outputs}/embeddings/<roi.name>.zarr`.

1. **Decide whether you need new imagery.** The update reads the mosaics already on disk, in
   `{paths.inputs}/<roi.name>/`. To take in imagery published since your first run, for the same
   window, delete that folder. The run then downloads the whole window again, which takes as long
   as the first run's download.
2. **Add an `update` block to the config file:**

   ```yaml
   update:
     name: more-imagery-2025   # your update's name
     mode: holes               # or shards; holes when left out
     tiles: [chunk_0_1]        # optional for holes; leave it out to update every tile
   ```

   In `holes` mode, leave out `tiles`: only holes are inferred anyway. In `shards` mode, list the
   tiles to rewrite. Keep every other setting as it was; the store refuses settings that would
   change what its embeddings mean, such as the radar settings.
3. **Run the same command as before,** with your config file:

   ```bash
   python -m tessera_embeddings.orchestration.runners.plain path/to/config.yaml
   ```

4. **Check the result.** The store now holds a record of the update, and a tag holds its state from
   before:

   ```python
   import zarr
   from tessera_embeddings.storage.global_store import open_global_repo  # opens single-area stores too

   repo = open_global_repo("file:///tmp/tessera/outputs/embeddings/quickstart_denver_co.zarr")  # your store
   root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
   print(root.attrs["updates"])  # {"2024": [{"update": "more-imagery-2025", "written": [...], "kept": [...], ...}]}
   print(repo.list_tags())       # includes "update-more-imagery-2025-base"
   ```

   `written` lists the tiles that changed and `kept` the tiles that had nothing to update.
5. **Delete the tag once you are satisfied:** `repo.delete_tag("update-more-imagery-2025-base")`.
   Until then the store keeps the old version of every tile the update rewrote.

## Update the global store

You need:

- **a global store your campaign filled.** Unless the deployment sets `paths.global_store_uri`, the
  store is `{paths.outputs}/global/tessera.icechunk` and its registry
  `{paths.outputs}/global/tessera.registry`;
- **the deployment of the global flows** that filled it (`run-global-campaign` and
  `fill-zones-sequential`), on a Prefect server you can reach
  ([`prefect-setup.md`](prefect-setup.md));
- **AWS access** that can read the store, its registry and the land mask, write under
  `{paths.inputs}/updates/`, and write to the store itself for clean-up.

Open the store this way for every check on this page:

```python
import zarr
from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials
from tessera_embeddings.storage.global_store import open_global_repo

repo = open_global_repo(
    "s3://my-outputs/global/tessera.icechunk", get_credentials=iam_icechunk_credentials, region="us-west-2"
)
root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
print(root["18S"].attrs["years_complete"])  # the years of zone 18S that are filled, e.g. [2023, 2024]
```

### 1. Choose the tiles and write the plan

`scripts/update/plan_update.py` writes the plan. It has two ways to choose tiles.

**From the registry**, the tiles that left pixels empty:

```bash
python scripts/update/plan_update.py from-registry \
    --name amazon-thin-2024 --mode holes \
    --inputs s3://my-inputs --registry s3://my-outputs/global/tessera.registry --region us-west-2 \
    --refused thin --zones 18S,19S --years 2024 --aoi=-75,-15,-50,5
```

| option | what it does |
|---|---|
| `--inputs` | the deployment's `paths.inputs`. The plan is written under it, where the flows look for it |
| `--refused` | keep tiles that left pixels empty for this reason: `thin` (too few optical observations), `no_optical` (none at all), `no_radar` (no radar observation), or `any`. Without it, every tile is kept |
| `--zones`, `--years` | keep only these zones and years, comma-separated |
| `--aoi=W,S,E,N` | keep tiles that overlap this box of longitudes and latitudes. Write it with `=`, or a negative first number is read as an option. It may not cross the 180° meridian; plan each side separately |
| `--registry` | the registry's location: the folder that holds `master/` |
| `--region` | the S3 region of the registry and the land mask (default `us-west-2`) |
| `--anonymous` | read a public registry without credentials |

**From a CSV of tiles**, for any other choice. Build the file however you like; each registry row
gives a tile's `zone`, `year`, `tile` and its bounding box (`bbox_west`, `bbox_south`, `bbox_east`,
`bbox_north`), which is how to find the tiles covering a place:

```text
zone,year,tile
18S,2024,chunk_28_1
18S,2024,chunk_30_3
19S,2024,
```

A blank `tile` means every tile of that zone-year.

```bash
python scripts/update/plan_update.py from-cells --name fix-tile-edges --mode shards \
    --inputs s3://my-inputs --cells tiles.csv
```

Both commands print a dry run and write nothing:

```text
  18S/2024      42 planned      42 live       0 dead in the mask
  19S/2024     310 planned     305 live       5 dead in the mask
2 zone-year(s), 347 live tile(s) to update
Optical depth rule: the store's own
Dry run: pass --write to save the plan to s3://my-inputs/updates/amazon-thin-2024/plan.json
```

A **live** tile holds land in the campaign's land mask, the map of where land is; a **dead** one
holds none and is skipped. When the counts look right, run the same command with `--write`. **A
written plan cannot be changed:** to change it, write a new plan under a new name.

Two more options: `--mask` names a land mask other than the campaign's `global`, and
`--on-skip clear` makes a `shards` update blank a chosen tile where nothing embeds any more, instead
of keeping it.

**To embed pixels the depth rule left empty**, give the update a lower rule of its own. A global
store has a depth rule, `optical_min_obs`: the minimum number of valid optical (Sentinel-2)
observations a pixel needs in its year to be embedded. Pixels below it were left as holes, and the
registry counts them as `thin`. Plan a `holes` update of those tiles with `--refused thin` and
`--optical-min-obs` set to the new minimum; `1` embeds every pixel with any optical observation:

```bash
python scripts/update/plan_update.py from-registry \
    --name amazon-thin-2024 --mode holes --optical-min-obs 1 \
    --inputs s3://my-inputs --registry s3://my-outputs/global/tessera.registry --region us-west-2 \
    --refused thin --zones 18S,19S --years 2024
```

The lower rule applies to this update's tiles only, and must be lower than the store's. A later
update of the same zone-years must use that rule or a lower one, or it is refused. The store's
own `optical_min_obs` keeps its value, so afterwards it understates the zone-years you updated; each
one's update record states the rule its update used. The registry rows of the updated tiles show the
new `optical_min_obs` and `refused_thin_px = 0`, and the same plan command then selects nothing.

### 2. Run it

Start one of the global flows with the update's name. `run-global-campaign` spreads the plan over
several GPU clusters and suits many zone-years; `fill-zones-sequential` uses one cluster and suits a
few. With Prefect:

```bash
prefect deployment run 'run-global-campaign/<deployment>' -p update=amazon-thin-2024 -p allow_s2_only=true
prefect deployment run 'fill-zones-sequential/<deployment>' -p update=amazon-thin-2024 -p allow_s2_only=true
```

- **A `holes` update needs `allow_s2_only=true`**, which embeds pixels that have optical
  (Sentinel-2) observations but no radar (Sentinel-1) ones, as the campaign does. Without it the
  run refuses before doing any work.
- **Parameters you leave out keep the deployment's values.** Any parameter the deployment does
  not set, such as `paths` in some setups, must be passed with `-p` too. To run part of the plan, add
  `-p zones='["18S"]' -p years='[2024]'` to `run-global-campaign`, or `-p cells='[["18S", 2024]]'` to
  `fill-zones-sequential`.
- **It starts GPU machines in your cloud account,** so it costs money while it runs; see "How long
  it takes" below.
- **Follow it in the Prefect UI** or with `prefect flow-run ls`. It is done when the flow run ends
  `Completed` and the store has the tag `update-<name>-complete`.
- **To resume after a failure, run the same command again.** Zone-years that are done are skipped.
- **Zone-years the campaign never filled are not updated.** `run-global-campaign` skips them with a
  warning, and `fill-zones-sequential` refuses them.

### 3. Check it

With `repo` opened as above:

```python
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.storage.update import load_update, update_states

name = "amazon-thin-2024"
plan = load_update(BucketPaths(inputs="s3://my-inputs", outputs=""), name)  # only inputs is read
print(update_states(repo, name, plan.cell_list()))  # every zone-year should be "done"
print(f"update-{name}-complete" in repo.list_tags())  # True

# Every tile that changed should be in the plan. Zones planned with a blank tile are left out.
planned = {(key.split("/")[0], tile) for key, tiles in plan.cells.items() for tile in tiles or ()}
whole = {key.split("/")[0] for key, tiles in plan.cells.items() if tiles is None}
diff = repo.diff(from_tag=f"update-{name}-base", to_tag=f"update-{name}-complete")
changed = {(path.split("/")[1], f"chunk_{c[1]}_{c[2]}") for path, chunks in diff.updated_chunks.items() for c in chunks}
print({c for c in changed if c[0] not in whole} - planned)  # passes when it prints set()
```

Then check that the registry agrees with the store:

```bash
python scripts/maintenance/rebuild_registry_measurements.py verify \
    --registry s3://my-outputs/global/tessera.registry --store s3://my-outputs/global/tessera.icechunk \
    --region us-west-2 --zones 18S,19S
```

The registry rows of the tiles the update wrote now carry its name in `update_name`.

If the tile check prints any tile, something outside the plan changed between the two tags: either
another writer committed to the store meanwhile, or the update wrote outside its plan, which is a
bug to report.

### 4. Clean up

1. **Delete the update's working files,** keeping `plan.json`, which keeps the name reserved:

   ```bash
   aws s3 rm s3://my-inputs/updates/amazon-thin-2024/ --recursive --exclude plan.json
   ```

2. **Delete the base tag once you are satisfied:** `repo.delete_tag("update-amazon-thin-2024-base")`.
   Until then the store keeps the old version of every tile the update rewrote, and you pay to store
   it.
3. **Reclaim the space,** if you want it back now. This permanently deletes every version of the
   store older than the update's completion that no tag holds, not only this update's, and there is
   no preview: `dry_run=True` skips deleting the old versions and reports only data nothing uses
   already, so it understates. Before running it, check that every version you want to keep has a
   tag (`print(repo.list_tags())`), and run it only while nothing is writing to the store:

   ```python
   from tessera_embeddings.storage.campaign import expire_and_gc

   cutoff = repo.lookup_snapshot(repo.lookup_tag("update-amazon-thin-2024-complete")).written_at
   print(expire_and_gc(repo, older_than=cutoff))  # what it expired and deleted
   ```

There is no undo for one update: going back to the base tag would also discard everything written
to the store after it.

## If something goes wrong

| what you see | what it means | what to do |
|---|---|---|
| `already recorded — nothing to do` | this name has already run on this window or zone-year | choose a new name |
| `holds no <window> window to update` | the single-area store never filled this time window | run the config without `update` first |
| `lists no tiles` | `tiles: []` in the config | leave `tiles` out to update every tile |
| `names tiles outside the area` | a tile in `tiles` is not part of the area | check the tile names |
| `selects no tiles` | no registry row matched the options | loosen the options, or check `--registry` |
| `is not in the mask` | a zone has no land in the land mask | leave the zone out |
| `No AWS credentials found (checked all providers except env vars)` | your AWS keys are environment variables | sign in with a profile or single sign-on |
| `FileExistsError` when writing a plan | a plan with that name exists | choose a new name |
| `FileNotFoundError` when the run starts | no plan with that name under the deployment's `paths.inputs` | write the plan with `--inputs` set to the deployment's `paths.inputs` |
| `needs allow_s2_only=True` | a `holes` update ran with radar required | add `-p allow_s2_only=true` |
| the run stopped part way | some zone-years are not done | run the same command again |
| `could not be re-derived` or `registry master NOT merged` in the log, or `verify` names a zone-year | the store is correct, but the registry is behind for that zone-year | `python scripts/maintenance/rebuild_registry_measurements.py repair --registry ... --store ... --zones <zone> --write` |
| `registry part ... could not be published` in the log | the store is correct, but the update's registry rows were never written, and `repair` cannot rebuild them | fix the cause (usually permissions), then run a `shards` update of the same tiles under a new name |

## How long it takes

On the reference deployment's GPUs, running the model over a whole tile takes 1 to 5 minutes of one
GPU's time, depending on how much imagery the tile has. A `holes` update sends only the holes to the
GPU but still reads the imagery around them, so a tile with few holes takes about 20 to 40 seconds,
and a tile that is mostly holes takes close to a full run. Each GPU cluster writes its zone-years
one at a time, so an update of many tiles with few holes finishes sooner with more clusters than
with more GPUs per cluster. Measurements:
[`update-dev-verification.md`](../context_docs/campaign/update-dev-verification.md).

## For developers

To drive updates from an orchestrator other than Prefect, see
[`orchestrator-swap.md`](orchestrator-swap.md): load the plan with
`storage.update.load_update(paths, name)` and pass each zone-year's
`plan.target(zone, year, root=paths.update_root(name))` as `update=` to the ingest and to
`runners.zone_fill.fill_zone_year`. An AI agent can follow this page through the
[`run-update`](../.agents/skills/run-update/SKILL.md) skill.
