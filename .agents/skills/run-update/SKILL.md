---
name: run-update
description: Run a targeted update of a tessera-embeddings store — re-infer chosen tiles of windows or zone-years the store already holds, either filling only the pixels that have no embedding (holes) or rewriting whole tiles (shards). Use when asked to fill holes, take in new imagery, redo tiles after a fix, or update part of a single-area or global store.
---

# Running an update

An **update** re-infers chosen tiles of data a store already holds and writes nothing else. It is
named, recorded on the store, and runs once: re-running it resumes, and once recorded it does
nothing. The design is [ADR 027](../../../context_docs/decisions/027-targeted-updates.md).

## Rules

- **Only for data the store holds.** A window or zone-year the store never filled is refused; fill
  it first.
- **One update per zone-year at a time.** Two updates of the same tiles conflict on their chunks.
- **Names and plans are permanent.** A global plan cannot be edited, and a name already recorded on
  a store's year does nothing when run again. To change anything, use a new name.
- **The store's identity still rules.** An update runs under the store's model and depth rule; it
  cannot change either. The depth rule can only be loosened, in a step of its own (§2b).
- **Dry-run, then write, then check.** Every writing script is a dry run without `--write`, and
  nothing is released until §4 passes.

## 1. Choose the mode

A **hole** is a pixel with no embedding (`scales` is NaN), including pixels refused for too few
observations.

| you want to | mode |
|---|---|
| fill holes: new imagery, an outage, pixels a loosened depth rule now admits | `holes` |
| redo whole tiles after an ingest or inference fix | `shards` |
| the same, and blank tiles where nothing embeds any more | `shards` with `on_skip: clear` |
| redo whole zone-years | `shards`, every tile |

- **`holes`** infers only the holes. Every pixel that holds an embedding keeps its bytes in all eight
  arrays; every hole gets the new imagery's observation counts, even where it still does not embed.
  It needs the run's `allow_s2_only=True`, and is refused before ingest otherwise.
- **`shards`** re-infers each target tile in full and writes it whole.
- **A tile where nothing new embeds is kept as it was**, unless the plan says `clear`.

## 2a. Update a single-area store

The single-area runner takes an `update` block in the run's config:

```yaml
update:
  name: more-imagery   # lowercase letters, digits and hyphens
  mode: holes          # or shards
  tiles: [chunk_0_1]   # optional: every tile when omitted; an empty list is refused
```

1. **To take in new imagery, delete the area's mosaics first.** Ingest resumes mosaics already on
   disk, and the update infers from them as they are.
2. **Run the original run's config with the block added:**
   `python -m tessera_embeddings.orchestration.runners.plain <config.yaml>`. The store refuses an
   update whose radar policy differs from the run it updates.
3. **Check it** (§4): the store's root `updates` attr holds a record for the window's year naming the
   update, with the tiles it `written` and `kept`, and `update-<name>-base` pins the store as it was.

## 2b. Update the global store

### Loosen the depth rule first, when that is the goal

To embed pixels the store refused for too few optical observations, move its minimum
(`optical_min_obs`) down, for the whole store or for named zones, then plan a `holes` update over the
tiles with thin refusals. The rule only ever moves down, and every line given up stays in the store's
history ([ADR 028](../../../context_docs/decisions/028-embed-every-pixel-with-an-optical-observation.md)):

```bash
python scripts/update/loosen_optical_min_obs.py --store <store-uri> --line 1 --zones 18N,18S   # dry run
python scripts/update/loosen_optical_min_obs.py --store <store-uri> --line 1 --zones 18N,18S --write
```

Then plan with `--refused thin`, over the zones you loosened: a zone left at the old
line re-infers its holes, refuses them again and keeps its tiles.

### Plan

`scripts/update/plan_update.py` writes the plan to `{inputs}/updates/<name>/plan.json`, from the
registry beside the store or from a list of tiles:

```bash
# shards whose registry master/ rows refused pixels, in an area, zones and years
python scripts/update/plan_update.py from-registry --name <name> --mode holes \
    --inputs <inputs-uri> --registry <registry-uri> --region <s3-region> \
    --refused thin --aoi=<west>,<south>,<east>,<north> --zones 18N,18S --years 2024

# a CSV with a zone,year,tile header; a blank tile means every live shard of that zone-year
python scripts/update/plan_update.py from-cells --name <name> --mode shards \
    --inputs <inputs-uri> --cells targets.csv
```

- **The registry** sits beside the store: `<path>/<store>.icechunk` has `<path>/<store>.registry`
  ([`docs/registry.md`](../../../docs/registry.md)); add `--anonymous` for a public registry.
- **`--refused`** keeps the tiles that refused pixels for one reason (`thin`, `no_optical`,
  `no_radar`) or for `any`; without it every tile in the zones, years and area is planned. Any other
  selection is a CSV, built however you like.
- **Read the dry run's report** before writing. Per zone-year it gives the shards planned, those the
  land mask holds live, and those it holds dead, which the fill drops. A zone missing from the mask
  is refused, and a `clear` plan cannot target dead shards.
- **The ingest is relaxed on purpose.** A plan admits a date with one valid pixel and a month with
  no data, because an update ingests only the blocks around its targets, and stricter settings would
  drop dates and months the zone-wide fill kept.
- `--mask` names the land mask under `{inputs}/masks/` (default `global`); `--on-skip clear` sets the
  plan's `on_skip`.
- **Then rerun with `--write`.**

### Run

The update is one parameter on the global fill flows, `update=<name>`. Every other parameter is
your deployment's usual one; a `holes` update also needs `allow_s2_only=true` unless the deployment
already sets it.

```bash
# several clusters, balanced on the plan's shards; zones= and years= narrow it
prefect deployment run 'run-global-campaign/<deployment>' -p update=<name> -p allow_s2_only=true
# one cluster; cells default to the plan's, and a subset splits the update across runs
prefect deployment run 'fill-zones-sequential/<deployment>' -p update=<name> -p allow_s2_only=true
```

For each zone-year, the run:
1. refuses it if never filled, and skips it if the update is already recorded;
2. pins `update-<name>-base` before the update's first write;
3. in `holes` mode, writes each target tile's hole mask and keeps the tiles that have none;
4. ingests only the 4096-pixel blocks holding targets, under `{inputs}/updates/<name>/`;
5. infers the targets, then merges holes into the existing shards or rewrites them whole;
6. records itself in the zone's `updates[year]`, leaving `years_complete`, `runs` and the zone-year
   tags as the fill left them;
7. publishes registry rows naming the update for the tiles it wrote.

Once every planned zone-year holds the record, the run pins `update-<name>-complete`. Re-running
the same command resumes. `storage.update.update_states(repo, name, plan.cell_list())` reports each
zone-year as `done`, `pending` or `not_complete`.

**Another orchestrator.** The Prefect flows are the reference layer
([`docs/orchestrator-swap.md`](../../../docs/orchestrator-swap.md)). Load the plan with
`storage.update.load_update(paths, name)`, take the work list from `update_states`, and pass each
zone-year's `plan.target(zone, year, root=paths.update_root(name))` as `update=` to the ingest and
to `runners.zone_fill.fill_zone_year`. `finish_update` pins the complete tag.

## 3. Size it

Figures are from the reference deployment
([`update-dev-verification.md`](../../../context_docs/campaign/update-dev-verification.md)); measure
your own hardware's.

- **A `holes` tile costs at least its reads.** The GPU skips every pixel outside the mask, and the
  read skips strips with no hole, but a strip with one hole is read whole: about 20 s for a sparse
  tile against 170–330 s for a full fill.
- **Merging costs assembly memory.** A `holes` merge reads each shard back: about 3 s and 0.9 GiB more
  per assembly worker.
- **Sparse `holes` updates are paced by assembly.** Each cluster assembles its zone-years one at a
  time, about one tile a second at 32 workers, so beyond a few tens of GPU workers per cluster the
  GPUs finish early and wait for nothing. Add clusters, not GPUs.
- **Old shards stay in storage.** The base tag holds every replaced shard until it is deleted (§5).

## 4. Check it

Compare the store before and after, with the store opened as you read it:

```python
diff = repo.diff(from_tag=f"update-{name}-base", to_tag=f"update-{name}-complete")  # single area: to_branch="main"
```

- **Only the targets changed.** `diff.updated_chunks` maps `/<zone>/<array>` to shard indices
  `(year_index, row, col, …)`, which must be the plan's tiles (`chunk_<row>_<col>`) in its years. A
  writer that committed between the two tags shows up here too.
- **Only the record changed in the attrs.** The zone groups in `diff.updated_groups` differ only in
  `updates`.
- **In `holes` mode, embedded pixels are untouched.** On each written tile, every pixel with a
  non-NaN `scales` at the base tag is byte-identical afterwards in all eight arrays, and the hole
  count fell.
- **The registry agrees with the store:**
  `python scripts/maintenance/rebuild_registry_measurements.py verify --registry <registry-uri> --store <store-uri> --zones <zones>`,
  and the `master/` rows of written tiles carry `update_name`.

## 5. Release it

- **Delete the update's inputs:** everything under `{inputs}/updates/<name>/` except `plan.json`,
  which reserves the name, and anything left under `{outputs}/staging/updates/<name>/`.
- **Once satisfied, delete the base tag** (`repo.delete_tag(f"update-{name}-base")`). Its replaced
  shards are reclaimed only when the store's history is expired and garbage-collected
  (`storage.campaign.expire_and_gc`), which must never run while anything is writing.
- **There is no per-update undo.** Resetting a branch to the base tag also discards every other
  commit made since.

## Recovery

| what happened | what to do |
|---|---|
| the run died before a zone-year's shards committed | re-run; staged tiles and hole masks are reused |
| it died after the shards committed, before the record | re-run; the same hole masks merge the same pixels again |
| registry rows failed after the commit (the log names the repair) | `rebuild_registry_measurements.py repair --registry <registry-uri> --store <store-uri> --zones <zone> --write` |
| a `master/` file is missing for an updated zone-year | re-run; finding the update recorded, it merges the file back without a commit |
| the plan was wrong | write a new plan under a new name |

## Traps

- **`--aoi=` needs the equals sign.** A negative west longitude is otherwise read as a flag.
- **Pass `--region` for an S3 registry.** Without it the scan asks a default region, and S3 answers
  with a redirect.
- **A recorded name runs silently.** Re-running a single-area update under a name already recorded
  for that year does nothing, without an error.
- **Single-area updates read the mosaics on disk.** Delete them first to take in new imagery.
