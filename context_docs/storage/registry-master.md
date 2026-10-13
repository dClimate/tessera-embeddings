# The registry master: one registry, kept current by every fill

The published registry answers "is my area covered, and how well" without opening the store. Its
`master/` prefix is that answer: **one file per zone-year and one row per tile, describing what the
store holds now.** Every fill merges its rows into its zone-year's file as it lands, tile by tile, so
the master never waits on a compaction, never needs a reader to know which run is current, and never
drops a tile that a later run did not touch. The store stands behind it: a fill updates the master
only when the store's own record of who wrote the zone-year accounts for every row. This doc is the
design, the checks that hold it, how to repair it, and how to roll it back.

The v1.1 backfill that first filled `master/` is a separate record:
[`registry-backfill.md`](registry-backfill.md).

## Two prefixes, and only one is for reading

```
  <store-stem>.registry/
  ├── README.md the readers' guide, a copy of docs/registry.md
  ├── parts/    zone=<Z>/year=<Y>/<run_id>.parquet
  │             each run's raw output, one part per zone-year per run. A re-run under the
  │             same run id replaces its own part; no run touches another's. Internal — the
  │             record the master is built from and checked against.
  └── master/   zone=<Z>/year=<Y>/part-0.parquet  + _common_metadata
                one row per (zone, year, tile), one schema, and two provenance columns:
                measured_by  "fill" | "rebuild" | null   who measured the numbers
                filled_at    the fill stamp of the run the row describes
```

`master/` is written with one schema and a dataset-level `_common_metadata`, so a reader without this
package takes the schema from the dataset. `parts/` needs `registry.dataset_schema()` stated, or
`pyarrow` infers it from the first file and silently drops any column added later.

`README.md` is a copy of [`docs/registry.md`](../../docs/registry.md), uploaded by hand, so a
change that alters what readers are told is re-uploaded with it. The writer role trusts only the
runner task's role, so the upload runs as a one-off task of the flow-runner task definition, under a
session policy granting that one key. It is served as `text/plain; charset=utf-8` so a browser
shows it rather than downloading it. Nothing reads it, and it sits outside both prefixes, so no
`ds.dataset` over either one sees it.

## How a fill keeps it current

```
  assemble_global ── commit the shards, tagged with the run ──▶ record the run in the zone's runs attr
        │         ──▶ publish parts/…/<run_id>.parquet, stamped with that commit's timestamp
        │
  zone_fill ──▶ _update_registry_master
                  │  read the master file's version, then the file and all of the cell's parts
                  │  then the store's record of the year's runs, and check the parts against it
                  │  merge tile by tile, check the invariants
                  │  replace master/zone=Z/year=Y/part-0.parquet if it is still at that version
                  │    (one conditional PUT; refused → read and merge again, up to 8 times)
                  │  compare embedded tiles with the store's live shards
                  │  on any failure: ERROR log, cell left stale, fill carries on
                  ▼
              tag the zone-year

  a retry of a cell that already landed ──▶ _update_registry_master, with no run id
```

`registry_master.update_cell` does the work, called from both assembled branches of
`orchestration/runners/zone_fill.py`, the ordinary one and the all-refused `empty=True` one. It
rewrites `_common_metadata` when it is absent or lacks a column of the current schema, so a column
added later reaches readers who take the schema from the dataset. It never rewrites one that has a
column the current schema lacks, so a worker still on an older release cannot take a newer column
away mid-deploy. That write is conditional too, so of two workers that both found it stale, the
second is refused and checks again. It runs before the tag because a deliberate refill fails at the tag step by design
(`reopen_zone_year.py`), after its data and part have landed, and the master must still describe it.

- **Two writers of one zone-year cannot lose each other's rows.** An update reads the file's
  version, its S3 ETag, before anything the new file is built from, and writes on condition that
  the version is unchanged. Of two fills, or a fill and a repair, that read the same version, only
  the first lands; the second is refused, and reads and merges again. Different zone-years never
  contend, and a reader sees a cell's old file or its new one, never a mixture. A local registry,
  for rehearsals, checks a content hash under a file lock instead.
- **It is idempotent.** Re-running it over the same inputs writes the same bytes.
- **It merges all of the cell's parts, not only the new one.** That costs a few small reads and
  means a cell whose earlier update failed is healed by its next fill.
- **It refuses a run whose own part is missing.** Publishing a part is best-effort, and merging
  the older parts alone would rewrite the cell as it was and report success.
- **A retry of a cell that already landed runs it again.** A process that dies between the commit
  and the update leaves no tag, and its retry finds the cell complete and skips the fill, so
  `assemble_zone_year` runs the update on that path too, with no run id. The store check below then
  refuses the cell if the dead run's part never published. A cell with neither a part nor a master
  file, an all-ocean one, is left alone.
- **It never fails a committed cell.** The cell is committed by then. Raising would undo nothing
  and keep the cell from being tagged, which is `publish_registry_part`'s rationale too. A failure
  is logged at ERROR with the repair command, and the stale cell is visible to the checks below.

## The merge rule: a tile's row comes from the newest run that wrote it

A run writes only the tiles it staged or refused and leaves the year's other shards as they were
(`assemble_global`'s refill caveat; `reopen_zone_year.py` deletes no data). So the store holds each
tile as the most recently committed run that wrote it left it, and the master follows the store:

| what lands | effect on the master |
|---|---|
| a new year | its tiles are added |
| a full refill | every tile it wrote supersedes the old row |
| a partial run, such as an infill | only the tiles it wrote are replaced; every other tile keeps its row |
| a re-derived measurement | fills a row the fill left null, for the run that row describes |

`filled_at` is what names the run. A part is stamped with its run's shard commit timestamp, taken
from the store, and Icechunk refuses a commit older than its parent, so the newest `filled_at` is the
run whose pixels the store holds, whichever run published first. The clock at publication would not
be: a run that commits first and publishes last would win tiles another run has since overwritten.
A measurement re-derived against a superseded run loses to the newer run's row, measured or not.
Within the winning run the precedence is not latest-wins:

1. a fill row carrying measurements beats everything;
2. otherwise a re-derived row carrying measurements, one that measured radar ahead of an
   optical-only one, then the most recent;
3. otherwise the most recent row of any kind, so a tile nothing has measured stays as a null row.

Timestamps are compared parsed and timezone-aware, and a stamp that will not parse stops the merge
rather than letting text order pick a run. Remaining ties fall to `run_id`, so every merge of the
same rows writes the same file.

**A run's row must describe the whole tile as that run leaves it.** Per-tile latest-wins makes the
newest run's row the tile's only row. An infill that rewrites some of a tile's pixels must therefore
publish measurements for the whole tile after the infill, not for the infilled pixels alone, or the
master will describe a fraction of the tile as if it were all of it. This is a requirement on the
future infill, not something this code can check.

## The store stands behind the parts

Newest-part-wins is right only if the parts account for every commit that wrote the year. Publishing
a part is best-effort, so a run can commit without one. Its tiles then keep the previous run's rows,
and once any later run lands, nothing in the registry shows it. So the store keeps the account, and
keeps it where no crash can separate it from the pixels: **every shard commit is tagged with its
run.** `write_year_shards` puts `{"shard_write": {run_id, zone, year}}` in the commit's own Icechunk
metadata, in the same commit as the shards.

```
  main:  … ─ shard commit  metadata.shard_write = {run_id, zone, year}   ← the authority
           ─ attrs commit  runs[year] = {run_id, assembled_at, commit: {snapshot, committed_at},
                                         ..., previous: [{run_id, assembled_at, commit, empty}, ...]}
```

- **The tagged commits are read from `main`'s history** (`published_store.shard_writes`). A
  spec-version-2 repository keeps every snapshot record, metadata included, in its one `repo`
  object, so the whole history is one read: 3,002 commits took 10 ms locally. A run that died
  between its two commits has no `runs` record and no part, but its tagged commit is there.
- **Every commit counts on its own, a retry's included.** A part describes the one commit it is
  stamped with. Any other commit by the same run, such as an earlier attempt that died after its
  shard commit, is checked like a run with no part: its tiles are the store's until something
  rewrites them, so a retry that rewrote every one clears it, and one that skipped a tile does not.
- **Runs from before commits were tagged**, all of v1.1, are known only from the `runs` record: an
  entry with no `commit` key, the year's own or one a refill kept in `previous`. One marked `empty`
  left no embeddings, so it needs no part. `commit` is null for a year marked without writing
  shards, which wrote no tile.
- **Snapshot expiry drops shard commits from the history** (`campaign.expire_and_gc` keeps tagged
  snapshots, and a zone-year's tag is on its attrs commit). A recorded run whose commit has gone is
  then known from its `runs` entry, like a v1.1 run: its part still describes it, and without one
  nothing bounds its tiles, so the zone-year blocks. Run `verify` before expiring.
- **Which tiles a commit wrote** comes from Icechunk's diff of that commit against its parent, the
  store's own transaction log (`published_store.commit_tiles`), in every per-tile array rather
  than `scales` alone: a refused tile writes its observation counts and month coverage and no
  embeddings, so re-refusing a tile changes nothing in `scales`. It is read only for a commit no
  part describes, so the run records carry no tile lists.

`registry_master.store_disagreements` refuses a cell, and writes nothing, when:

| the parts | why that is wrong |
|---|---|
| include a run the store records no commit by | the run is not in the store; its part must be deleted |
| include a part stamped at none of its run's commits | the merge would order it by the wrong clock |
| describe no commit that was the last to write some tile | that tile's row describes an older run |
| lack a run from before commits were tagged, other than one that marked the year empty | nothing bounds which tiles it wrote |

A commit no part describes whose every tile a later commit wrote over describes nothing the store
holds, so it does not block. **That is also how a blocked cell recovers.** Rewrite the tiles the refusal names
under a run that publishes its part, such as a refill of the zone-year, and the cell's next update
goes through. Every update and repair runs this check, and so does `verify`. An update reads the
store's record after the parts, on every attempt: a run publishes its part only after its commit, so
a fill that lands mid-update, or mid-way through an hours-long repair, is merged rather than refused.

## What guards it

- **Every update checks its invariants before writing**: one row per tile-year, every tile any part
  names present, and `measured_by` agreeing with the numbers. A breach raises and writes nothing.
- **The store's live shards are compared with the master's embedded tiles**, from the manifests
  alone (`published_store.live_shards`). No run removes a tile, so a shard an operator cleared by hand
  would otherwise stay in the master as embedded. A disagreement is logged at ERROR; the master is
  still written, because it describes the registry faithfully.
- **A re-derived row must name its fill.** `as_master_rows` refuses one without `filled_at`, which
  would otherwise pass for a newer run and beat the fill's measurement.
- **An existing master row survives only while its run still has a part** (`still_published`).
  A run whose part is deleted would otherwise keep winning through the master's copy of
  its rows, which carry the newest `filled_at`. And only its re-derived rows are taken from the
  existing file: a fill row always comes from its part, so a damaged copy cannot outlive a repair.
- **The parts are checked against the store's record of the year's runs**, as above. Then every
  row's tile holds the pixels of the run that row names, so a cell built from several runs is
  re-derivable tile by tile, each row under its own depth rule.

## Checking and repairing it

`verify` in `scripts/maintenance/rebuild_registry_measurements.py`, and the completeness section of
`scripts/diagnostic/published_registry_census.py`, check `master/` itself, cell by cell, against
`parts/` (`registry_master.audit`); the census fails outright when `master/` is absent. `verify`
also checks `parts/` against the store, which the census does not, in every cell with a part, a
master file or a tagged shard commit, so a first fill whose part never published is found though
its cell has no part at all. `--parts-only` skips the store for a registry copy with none to read.

| check | catches |
|---|---|
| one row per tile-year; `measured_by` agrees with the numbers | a duplicated or mislabelled row |
| every tile in `parts/` present, and none that no part names | a cell whose update failed, or a stray row |
| each row from its tile's latest run | a refill or infill whose update failed |
| fill-measured rows equal `parts/` for the latest run: measurements, `run_id` and `assembled_at` | a fill row the master altered |
| `embedded`, the bounding box, the depth rule and the build equal the latest run's fill row | a row, re-derived or not, carrying another row's fields |
| no unmeasured rows | a fill that published nulls |
| the master's embedded tiles against the store's live shards (census `--verify-zones` only) | a shard cleared by hand, which no part records |
| `parts/` against the store's run record (`verify` only) | a run with no part that holds a tile, a part out of commit order, a withdrawn run's part |

Both name every failing cell. `repair --zones <Z,...> --write` rebuilds those cells from three
inputs, merged by the rule above: the cell's parts, its existing master rows, and a fresh
re-derivation from the store of every row its fills left null. The parts supply every tile and every
fill measurement; the existing rows keep a radar column an optical-only repair would not re-measure.
A cell whose re-derivation or store check is blocked is not written, so a repair never replaces
measurements with nulls. Repair visits the same cells `verify` does, so a master file whose last
part was deleted is emptied and a tagged commit with no part is reported. Repair writes each cell through `update_cell`, the fill's own path, which
re-reads that cell's parts and master and writes conditionally, so a fill landing mid-repair is
merged in rather than overwritten. `gate` proves the re-derivation reproduces measured rows before
any repair; the method and its proof are in [`registry-backfill.md`](registry-backfill.md).

**The fill does not re-derive inline.** A fill records every tile's coverage record on object
storage (refused tiles in their skip markers, embedded tiles on the staged tile; see the
[inference README](../../src/tessera_embeddings/inference/README.md)), and a resumed fill reads it
back, so a fill publishes null only when one of those reads fails. That case is rare, logged with
a count, named by the checks above, and repaired by `repair`. Re-deriving inside every fill would
add store reads and a gate to each cell's tail to cover it.

**No per-cell validator check is added.** The per-cell validation flow belongs to the consumer that
deploys it (`flows/_cell_validation.py` only dispatches it), so the census is where this repository
detects a stale master.

## Rolling back: S3 versioning, a bucket-owner action

The master is overwritten in place, so its history lives in S3 object versions. **This needs the
bucket owner to configure the bucket; this repository does not and cannot do it.**

Versioning is bucket-wide, not per prefix, and `tessera-embeddings` also holds the Icechunk store,
where deletes and ref overwrites would otherwise keep billable versions — the reason our own buckets
keep versioning off ([`campaign-plan.md`](../campaign/campaign-plan.md), "Three standing
constraints"). Lifecycle rules make the store effectively unversioned while the registry keeps a
year of history:

```
aws s3api put-bucket-versioning --bucket tessera-embeddings \
    --versioning-configuration Status=Enabled
aws s3api put-bucket-lifecycle-configuration --bucket tessera-embeddings \
    --lifecycle-configuration file://lifecycle.json
```

```json
{"Rules": [
  {"ID": "store-unversioned", "Status": "Enabled",
   "Filter": {"Prefix": "v1.1/dclimate.icechunk/"},
   "NoncurrentVersionExpiration": {"NoncurrentDays": 1},
   "Expiration": {"ExpiredObjectDeleteMarker": true}},
  {"ID": "registry-history", "Status": "Enabled",
   "Filter": {"Prefix": "v1.1/dclimate.registry/"},
   "NoncurrentVersionExpiration": {"NoncurrentDays": 365},
   "Expiration": {"ExpiredObjectDeleteMarker": true}}
]}
```

Every other prefix in the bucket keeps its noncurrent versions indefinitely unless it has a rule
too; a one-day rule per prefix keeps it behaving as it does today. A rule cannot exclude a prefix,
and where two expiration rules overlap S3 applies the shorter, so a bucket-wide one-day rule would
also cut the registry's history to a day. Versioning, once enabled, can be suspended but not
removed. The registry's cost is small: each fill overwrites one file of a few hundred kilobytes,
and `parts/` only gains a file when a run under a new id lands.

To roll a zone-year's master file back to a prior version:

```
KEY=v1.1/dclimate.registry/master/zone=16S/year=2021/part-0.parquet
aws s3api list-object-versions --bucket tessera-embeddings --prefix "$KEY" \
    --query 'Versions[].[VersionId,LastModified,IsLatest]' --output table
aws s3api copy-object --bucket tessera-embeddings --key "$KEY" \
    --copy-source "tessera-embeddings/$KEY?versionId=<VersionId>"
```

The copy becomes the current version and the replaced one stays recoverable. The master is merged
from `parts/`, though, so a rollback that disagrees with the parts is flagged by `verify` and undone
by the cell's next fill.

**A run cannot be withdrawn from the registry alone**, because the master describes what the store
holds. To replace a run's data, refill the cell: the refill's part supersedes it tile by tile.
Deleting a part (also versioned) and then running `repair` is only for a part that describes nothing
the store holds, such as a run whose commit was itself rolled back, which rolls back its record
too; if it was the cell's last part, `repair` empties the cell's master file. The store check refuses the cell while the store still records the run as the last writer of
any tile, and refuses a part the store has no record of until it is deleted.

## Which code identity it moves

`inference/assembly.py`, `storage/registry.py` and `storage/shard_writer.py` are inside
`inference_code_identity`'s closure, which is everything `inference/` and `config/inference.py`
import, transitively, so editing any of them re-stages every in-progress tile. This design edits two:
assembly persists the coverage record on the staged tile and stamps the part with the commit, and
the shard writer records the commit and the runs before it. So it moves the inference identity, and
a tile staged under the old code is re-staged after a deploy. The ingest identity, and so every
mosaic, is unchanged. `registry_master.py`, `registry_rebuild.py`, `published_store.py` and the
zone-fill runner are outside both closures, so the merge and its checks can change without touching
staged work or mosaics.

## The switch to this design changed nothing on v1.1

Every v1.1 zone-year has one run, so the per-tile rule must reproduce the published master exactly.
Checked on 2026-10-12 against a local copy of `parts/`, `master/` and the re-derived rows that went
into it, writing nowhere but scratch space:

| check | result |
|---|---|
| rows a repair would re-derive vs rows the v1.1 backfill re-derived | the same 1,519,045 tile-years |
| full repair from `parts/` plus those re-derived rows, no master | identical to the published master, row for row and column for column |
| repair with the published master also present | identical |
| every cell's fill update over the published master and `parts/`, written to disk | identical rows in all 994 files; 23 files differ only in row order, now sorted by tile |
| `audit` and `verify` of the published master | sound in every cell; exit 0 |
| every cell's update with the conditional write, over a fresh copy of `parts/` and `master/` (2026-10-13) | identical to the published master in all 994 cells, 3,247,410 rows |
| `audit` with the fill fields compared too, over the published master (2026-10-13) | sound in every cell, 3,247,410 rows |

The store check passes on v1.1 without a run: each v1.1 record names one run and no commit, and the
check then requires only that the cell's parts are that run's. The backfill's own store check
required the same of all 994 cells, and every one matched.
