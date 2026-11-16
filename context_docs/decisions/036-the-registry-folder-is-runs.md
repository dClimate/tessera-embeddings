# 036 — The registry's per-run folder is `runs/`

**Status:** Proposed 2026-10-28 in PR #259, stacked on #256. Moving a registry written before it
is a deploy step: `scripts/maintenance/rename_registry_parts_to_runs.py`.

## Context

The registry beside a store holds two Parquet folders. `master/` has one file per zone-year and is
what consumers read. The other has one file per run per zone-year, `<run_id>.parquet`, which the
master is merged from and checked against. That folder was called `parts/`, after the part files of
a partitioned dataset, and the name told a reader nothing: `docs/registry.md` had to explain that a
part is a run's file, and the code called the same file a part, a run's rows or a fill's output
depending on where you looked.

The provenance record ([ADR 035](035-a-provenance-record-for-the-published-store.md)) keys its
per-run pieces by run ID as well (`shard_dates/…/<run_id>.parquet`, the `<run_id>` groups of
`pixel_sources.icechunk`), so the run is the unit the whole registry folder is organised by.

## Decision

1. **The folder is `runs/`, and a file in it is a run file.** The path below the folder is
   unchanged: `runs/zone=<Z>/year=<Y>/<run_id>.parquet`. The code, its log lines and the docs say
   "run file" throughout, so `part_uri`, `write_registry_part`, `ZarrWriter.publish_registry_part`
   and the rebuild script's `--parts-only` become `run_file_uri`, `write_run_file`,
   `publish_registry_run_file` and `--runs-only`.
2. **The code reads and writes `runs/` only.** A registry written before is moved once by
   `rename_registry_parts_to_runs.py`: a dry run by default, every copy checked before any source
   is deleted, resumable after a crash, and reversible with `--rollback`.
3. **The master's `part-0.parquet` keeps its name.** It is the file consumers' own queries name
   (the DuckDB example in `docs/registry.md` globs it), so renaming it would be a second migration,
   of the files people actually read, for a name that sits inside a folder that already says what
   it holds.

## Rejected alternatives

- **Keep `parts/`.** Free to do, but every reader pays: the word needs defining wherever it
  appears, and with the provenance record keyed by run the registry would have two words for one
  unit.
- **Read both folders for a while.** The merge, `verify`, `repair` and the census would each need a
  union read and a rule for a run present in both, all to cover a one-off move of about a thousand
  small objects that takes minutes.
- **A lock around the move.** The registry has no lock, and a deploy already stops fills; building
  one for a single move is more code than the window it closes. The script's help, the scripts
  index and the deploy steps say that nothing may write the registry while it runs.

## Consequences

- **A registry must be moved before fills run on this code against it.** A fill on this code
  against an unmoved registry writes its run file to `runs/`, and its master merge is then refused,
  because the store records runs whose files are still under `parts/`
  (`registry_master.store_disagreements`). The master stays stale rather than wrong until the move
  and a `repair --zones`. The safe order is to stop fills, move, run `verify`, then deploy.
- `storage/registry.py`, `inference/assembly.py`, `storage/shard_writer.py` and
  `storage/published_store.py` are inside `inference_code_identity`'s import closure, so the change
  moves the inference identity and a tile staged before the deploy is re-staged after it. The
  ingest identity is unchanged.
- A copy of a registry taken before its move still has `parts/`. `docs/registry.md` says so in one
  sentence and points at the script, which moves a local copy the same way.
