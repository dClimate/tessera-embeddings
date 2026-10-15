# Updating part of a store

An **update** re-runs inference on chosen tiles of an embeddings store you have already filled, and
writes nothing else. Use one when new imagery has arrived, when some pixels were left empty, or when
a fix means some tiles must be redone. Every tile you did not choose is left exactly as it was.

This page covers both kinds of store:

- a **single-area store**, written by the plain runner from a config file
  ([`quickstart.md`](quickstart.md));
- the **global store**, written by the campaign flows ([`global-store.md`](global-store.md)).

The design and its reasons are in [ADR 027](../context_docs/decisions/027-targeted-updates.md).

## Words used on this page

| word | meaning |
|---|---|
| tile | a square of the output, 2048 × 2048 pixels, named `chunk_<row>_<column>` (row and column counted from the top-left corner, starting at 0) |
| hole | a pixel with no embedding, including one left empty because it had too few observations |
| zone-year | one UTM zone in one calendar year, such as `18S` in 2024: what the global campaign fills at a time |
| registry | the per-tile table the global campaign writes beside the store (`tessera.registry` next to `tessera.icechunk`), recording what each tile holds ([`registry.md`](registry.md)) |

## The update's name ties everything together

Every update has a **name** that you choose, such as `more-imagery-2025` or `fix-tile-edges`. The
name is how you start the update, how the store records it, and how you find it again afterwards.

A name may use lowercase letters, digits and hyphens, must start with a letter or a digit, and can be
up to 63 characters long. **An update runs once per name.** Running the same name again finishes it
if it stopped part way, and otherwise does nothing. To run another update, choose a new name.

Where you write the name, and where it appears afterwards:

| | single-area store | global store |
|---|---|---|
| you give the name | `update.name` in the run's config file | `--name` when you plan the update, then `update=<name>` when you run it |
| the tiles and mode are set | in the same `update` block | in the plan file, `{inputs}/updates/<name>/plan.json` |
| the store records it | in its root attribute `updates`, under the year: a record whose `update` field is the name | in each zone's attribute `updates`, the same way |
| tags in the store | `update-<name>-base`: the store before the update | `update-<name>-base`, and `update-<name>-complete` once every planned zone-year is done |
| the registry shows it | single-area stores have no registry | the `update_name` column on every tile the update rewrote |

So the `update` parameter of the global flows is not a description of the work. It is only the name
of a plan you have already written; the plan holds everything else.

## Choose a mode: holes or shards

| mode | what is re-inferred | what is written | use it for |
|---|---|---|---|
| `holes` (the default) | only the holes | the holes, in every array; every other pixel keeps its exact values | new imagery, an outage, pixels a looser rule now admits |
| `shards` | every pixel of each chosen tile | each chosen tile, whole | a fix to ingest or inference, where existing values are wrong |

If nothing new can be embedded in a chosen tile, that tile is left as it was.

## Update a single-area store

You need the config file you ran before, and the store it wrote. An update only changes a time
window the store already holds.

1. **Decide whether you need new imagery.** The update reads the mosaics already on disk, in
   `{paths.inputs}/<roi name>/`. To take in imagery published since your first run, delete that
   folder; the run rebuilds it.
2. **Add an `update` block to the config file:**

   ```yaml
   update:
     name: more-imagery-2025   # your update's name
     mode: holes               # or shards; holes when left out
     tiles: [chunk_0_1]        # optional; leave it out to update every tile
   ```

   Leave out `tiles` unless you want only some tiles: in `holes` mode only holes are inferred
   anyway. Keep every other setting as it was. The store refuses settings that would change what
   its embeddings mean, such as its radar settings.
3. **Run the same command as before:**

   ```bash
   python -m tessera_embeddings.orchestration.runners.plain config.yaml
   ```

4. **Check the result.** The store now records the update, and the tag holds its state from before:

   ```python
   import zarr
   from tessera_embeddings.storage.global_store import open_global_repo

   repo = open_global_repo("file:///tmp/tessera/outputs/embeddings/quickstart_denver_co.zarr")
   root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
   print(root.attrs["updates"])  # {"2024": [{"update": "more-imagery-2025", "written": [...], "kept": [...], ...}]}
   print(repo.list_tags())       # includes "update-more-imagery-2025-base"
   ```

   `written` lists the tiles that changed and `kept` the tiles that had nothing to update.

## Update the global store

You need:

- zone-years the campaign has already filled;
- the registry the campaign wrote beside the store, if you want to choose tiles from it;
- a deployment of the global flows (`run-global-campaign` and `fill-zones-sequential`);
- credentials that can read the registry and write to the deployment's inputs location (`{inputs}`,
  its `BucketPaths.inputs`).

### 1. Choose the tiles and write the plan

`scripts/update/plan_update.py` writes the plan. Run it from the repository root. It has two ways
to choose tiles.

**From the registry**, the tiles that left pixels empty:

```bash
python scripts/update/plan_update.py from-registry \
    --name amazon-thin-2024 --mode holes \
    --inputs s3://my-inputs --registry s3://my-bucket/tessera.registry --region us-west-2 \
    --refused thin --zones 18S,19S --years 2024 --aoi=-75,-15,-50,5
```

| option | what it does |
|---|---|
| `--refused` | keep tiles that left pixels empty for this reason: `thin` (too few optical observations), `no_optical` (none at all), `no_radar` (no radar observation), or `any`. Without it, every tile is kept |
| `--zones`, `--years` | keep only these zones and years, comma-separated |
| `--aoi=W,S,E,N` | keep tiles that overlap this longitude and latitude box. Write it with `=`, or a negative first number is read as an option. It may not cross the 180° meridian; plan each side separately |
| `--anonymous` | read a public registry without credentials |
| `--region` | the registry's S3 region |

**From a CSV of tiles**, for any other choice. Build the file however you like:

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

Both print a dry run and write nothing:

```text
  18S/2024      42 planned      42 live       0 dead in the mask
  19S/2024     310 planned     305 live       5 dead in the mask
2 cell(s), 347 live shard(s) to update
Dry run: pass --write to save the plan to s3://my-inputs/updates/amazon-thin-2024/plan.json
```

A **dead** tile holds no land in the campaign's land mask and is skipped. When the counts look
right, run the same command with `--write`. **A written plan cannot be changed:** to change it,
write a new plan under a new name.

Two more options: `--mask` names a land mask other than the campaign's `global`, and
`--on-skip clear` makes a `shards` update blank a chosen tile where nothing embeds any more, instead
of keeping it.

### 2. Run it

Pass the name to one of the global flows. With Prefect:

```bash
# across several GPU clusters
prefect deployment run 'run-global-campaign/<deployment>' -p update=amazon-thin-2024 -p allow_s2_only=true

# or on one cluster
prefect deployment run 'fill-zones-sequential/<deployment>' -p update=amazon-thin-2024 -p allow_s2_only=true
```

- **A `holes` update needs `allow_s2_only=true`**, unless the deployment already sets it. The run
  refuses before doing any work otherwise.
- **Every other parameter is your deployment's usual one.** `run-global-campaign` also takes `zones`
  and `years`, and `fill-zones-sequential` takes `cells`, to run part of the plan.
- **To resume, run the same command again.** Zone-years that are done are skipped.
- **Zone-years the campaign never filled are not updated.** `run-global-campaign` skips them with a
  warning, and `fill-zones-sequential` refuses them.

For each zone-year, the run ingests imagery for the chosen tiles only, re-infers them, writes them
into the store, records the update in the zone's `updates` attribute, and updates the registry rows
of the tiles it wrote. It leaves the zone-year's original records (`years_complete`, `runs`) as they
were.

Using an orchestrator other than Prefect: see
[`orchestrator-swap.md`](orchestrator-swap.md). The flows load the plan with
`storage.update.load_update(paths, name)` and pass each zone-year's
`plan.target(zone, year, root=paths.update_root(name))` to the ingest and to
`runners.zone_fill.fill_zone_year` as `update=`.

### 3. Check it

```python
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.update import load_update, update_states

name = "amazon-thin-2024"
repo = open_global_repo("s3://my-bucket/tessera.icechunk")
plan = load_update(BucketPaths(inputs="s3://my-inputs", outputs=""), name)

print(update_states(repo, name, plan.cell_list()))  # every zone-year should be "done"
print(f"update-{name}-complete" in repo.list_tags())

diff = repo.diff(from_tag=f"update-{name}-base", to_tag=f"update-{name}-complete")
print(diff.updated_chunks)  # per array, the (year index, row, column, ...) of each tile that changed
```

- **Only the plan's tiles changed.** Each entry of `diff.updated_chunks` should be one of the plan's
  tiles in one of its years. Anything else another writer committed in the meantime appears here too.
- **The registry agrees with the store:**

  ```bash
  python scripts/maintenance/rebuild_registry_measurements.py verify \
      --registry s3://my-bucket/tessera.registry --store s3://my-bucket/tessera.icechunk --zones 18S,19S
  ```

  The registry rows of the tiles the update wrote carry its name in `update_name`.

### Filling pixels a looser rule now admits

A global store has a depth rule, `optical_min_obs`: a pixel needs at least that many valid optical
observations in its year to be embedded. Pixels below the rule were left empty, and the registry
counts them as `thin`. To embed them, lower the rule first, then run a `holes` update over the tiles
with thin pixels.

1. **Lower the rule**, for the whole store or for some zones. It can only go down, and each value
   given up is kept in the store's history:

   ```bash
   python scripts/update/loosen_optical_min_obs.py --store s3://my-bucket/tessera.icechunk --line 1 --zones 18S,19S
   python scripts/update/loosen_optical_min_obs.py --store s3://my-bucket/tessera.icechunk --line 1 --zones 18S,19S --write
   ```

   The first command is a dry run that prints each rule before and after. Lower it while no fill is
   running in those zones: a fill already running there stops at its next zone-year, because the
   rule it started with no longer matches the store.
2. **Plan with `--refused thin` and the same `--zones`** (step 1). A zone still at the old rule would
   re-infer its holes and leave them empty again.
3. **Run and check it** as in steps 2 and 3. The registry rows of the updated tiles then show the new
   `optical_min_obs` and `refused_thin_px = 0`, and the same plan command selects nothing.

### 4. Clean up

- **Delete the update's working files:** everything under `{inputs}/updates/<name>/` except
  `plan.json`, which keeps the name reserved.
- **Delete the base tag once you are satisfied:**
  `repo.delete_tag(f"update-{name}-base")`. Until then the store keeps every tile's old version.
  The space comes back only when the store's old snapshots are expired and garbage-collected
  (`storage.campaign.expire_and_gc`), which must never run while anything is writing to the store.
- **There is no undo for one update.** Resetting the store to the base tag would also discard
  everything written after it.

## If something goes wrong

| what you see | what it means | what to do |
|---|---|---|
| `already recorded — nothing to do` | this name has already run on this window or zone-year | choose a new name |
| `holds no <window> window to update` | the single-area store never wrote this time window | run the config without `update` first |
| `lists no tiles` | `tiles: []` | leave `tiles` out to update every tile |
| `names tiles outside the area` | a tile in `tiles` is not part of the area | check the tile names |
| `selects no tiles` | no registry row matched the options | loosen the options, or check the registry path |
| `is not in the mask` | a zone has no land in the land mask | leave the zone out |
| `FileExistsError` when writing a plan | a plan with that name exists | choose a new name |
| `FileNotFoundError` when the run starts | no plan with that name under the deployment's `{inputs}` | write the plan with the deployment's own `--inputs` |
| `needs allow_s2_only=True` | a `holes` update ran with radar required | pass `allow_s2_only=true` |
| the run stopped part way | some zone-years are not done | run the same command again |
| `could not be re-derived` in the log, after a commit | the store is correct, but some registry rows were not written | `rebuild_registry_measurements.py repair --registry ... --store ... --zones <zone> --write` |

## How long it takes

A `holes` update sends only the holes to the GPU, but still reads the imagery around them. A tile
with few holes costs about a tenth of a full re-run of the tile; a tile that is mostly holes costs
close to a full run. Each GPU cluster writes its zone-years one at a time, so an update of many tiles
with few holes finishes sooner with more clusters than with more GPUs per cluster. Measurements:
[`update-dev-verification.md`](../context_docs/campaign/update-dev-verification.md).

An agent can follow this page through the
[`run-update`](../.agents/skills/run-update/SKILL.md) skill.
