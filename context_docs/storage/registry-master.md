# The registry master: one registry, kept current by every fill

The published registry answers "is my area covered, and how well" without opening the store. Its
`master/` prefix is that answer: **one file per zone-year and one row per tile, describing what the
store holds now.** Every fill merges its rows into its zone-year's file as it lands, tile by tile, so
the master never waits on a compaction, never needs a reader to know which run is current, and never
drops a tile that a later run did not touch. This doc is the design, the checks that hold it, how to
repair it, and how to roll it back.

The v1.1 backfill that first filled `master/` is a separate record:
[`registry-backfill.md`](registry-backfill.md).

## Two prefixes, and only one is for reading

```
  <store-stem>.registry/
  ├── parts/    zone=<Z>/year=<Y>/<run_id>.parquet
  │             each fill's raw output: one part per zone-year per run, append-only, never
  │             rewritten. Internal — the record the master is built from and checked against.
  └── master/   zone=<Z>/year=<Y>/part-0.parquet  + _common_metadata
                one row per (zone, year, tile), one schema, and two provenance columns:
                measured_by  "fill" | "rebuild" | null   who measured the numbers
                filled_at    the fill stamp of the run the row describes
```

`master/` is written with one schema and a dataset-level `_common_metadata`, so a reader without this
package takes the schema from the dataset. `parts/` needs `registry.dataset_schema()` stated, or
`pyarrow` infers it from the first file and silently drops any column added later.

## How a fill keeps it current

```
  assemble_global ── commit the zone-year ──▶ publish parts/…/<run_id>.parquet
        │
  zone_fill ── tag the zone-year ──▶ _update_registry_master
                                       │  read the cell's master file + all of its parts
                                       │  merge tile by tile, check the invariants
                                       │  replace master/zone=Z/year=Y/part-0.parquet  (one PUT)
                                       │  compare embedded tiles with the store's live shards
                                       ▼
                                     on any failure: ERROR log, cell left stale, fill succeeds
```

`registry_master.update_cell` does the work, called from both assembled branches of
`orchestration/runners/zone_fill.py`, the ordinary one and the all-refused `empty=True` one. It
writes `_common_metadata` only if absent.

- **Zone-years are independent.** Each update reads and replaces one object, so concurrent fills
  of different cells never contend, and a reader sees a cell's old file or its new one, never a
  mixture.
- **It is idempotent.** Re-running it over the same inputs writes the same bytes.
- **It merges all of the cell's parts, not only the new one.** That costs a few small reads and
  means a cell whose earlier update failed is healed by its next fill.
- **It never fails a committed cell.** The cell is committed and tagged by then. Raising would undo
  nothing and lose the tag's record, which is `publish_registry_part`'s rationale too. A failure is
  logged at ERROR with the repair command, and the stale cell is visible to the checks below.

## The merge rule: a tile's row comes from the newest run that wrote it

A run writes only the tiles it staged or refused and leaves the year's other shards as they were
(`assemble_global`'s refill caveat; `reopen_zone_year.py` deletes no data). So the store holds each
tile as the most recently assembled run that wrote it left it, and the master follows the store:

| what lands | effect on the master |
|---|---|
| a new year | its tiles are added |
| a full refill | every tile it wrote supersedes the old row |
| a partial run, such as an infill | only the tiles it wrote are replaced; every other tile keeps its row |
| a re-derived measurement | fills a row the fill left null, for the run that row describes |

`filled_at` is what names the run: the newest parsed `filled_at` wins, so a measurement re-derived
against a superseded run loses to the newer run's row, measured or not. Within the winning run the
precedence is not latest-wins:

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

## What guards it

- **Every update checks its invariants before writing**: one row per tile-year, every tile any part
  names present, and `measured_by` agreeing with the numbers. A breach raises and writes nothing.
- **The store's live shards are compared with the master's embedded tiles**, from the manifests
  alone (`published_store.live_shards`). No run removes a tile, so a shard an operator cleared by hand
  would otherwise stay in the master as embedded. A disagreement is logged at ERROR; the master is
  still written, because it describes the registry faithfully.
- **A re-derived row must name its fill.** `as_master_rows` refuses one without `filled_at`, which
  would otherwise pass for a newer run and beat the fill's measurement.
- **Re-derivation checks the store holds the cell's newest run.** The zone group's `runs` attribute
  names the last run that wrote each year, and it must be the newest run in the cell's parts. Then
  every row's tile holds the pixels of the run that row names, so a cell built from several runs is
  re-derivable tile by tile. A run that committed but never published its part blocks the cell.

## Checking and repairing it

`verify` in `scripts/maintenance/rebuild_registry_measurements.py`, and the completeness section of
`scripts/diagnostic/published_registry_census.py`, check `master/` itself, cell by cell, against
`parts/` (`registry_master.audit`):

| check | catches |
|---|---|
| one row per tile-year; `measured_by` agrees with the numbers | a duplicated or mislabelled row |
| every tile in `parts/` present, and none that no part names | a cell whose update failed, or a stray row |
| each row from its tile's latest run | a refill or infill whose update failed |
| fill-measured values equal `parts/` for the latest run | a measurement the master altered |
| no unmeasured rows | a fill that published nulls |

Both name every failing cell. `repair --zones <Z,...> --write` rebuilds those cells from three
inputs, merged by the rule above: the cell's parts, its existing master rows, and a fresh
re-derivation from the store of every row its fills left null. The parts supply every tile and every
fill measurement; the existing rows keep a radar column an optical-only repair would not re-measure.
A cell whose re-derivation is blocked is not written, so a repair never replaces measurements with
nulls. `gate` proves the re-derivation reproduces measured rows before any repair; the method and
its proof are in [`registry-backfill.md`](registry-backfill.md).

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
and `parts/` is append-only.

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
by the cell's next fill. To withdraw a run, delete its part (also versioned) and then `repair` the
cell.

## Why none of this moves a code identity

`storage/registry.py` and `inference/assembly.py` are inside `inference_code_identity`'s closure,
which is everything `inference/` and `config/inference.py` import, transitively; editing either would
re-stage every in-progress tile. `registry_master.py`, `registry_rebuild.py`, `published_store.py`
and the zone-fill runner are outside both identities' closures, so the master's design can change
without touching staged work or mosaics.

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
