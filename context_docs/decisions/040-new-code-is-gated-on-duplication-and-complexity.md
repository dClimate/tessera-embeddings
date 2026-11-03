# 040 — New code is gated on duplication and complexity

**Status:** Proposed 2026-11-03.

## Context

Much of the code arriving in this repository is written by AI agents. Their characteristic
failures are ones a reviewer reading the PR description will not see: a near-copy of an existing
function with a few lines changed rather than a call to it, and a new function that keeps
growing branches rather than being split. Both make the codebase larger and harder to change, and
both are cheap to detect mechanically.

A gate on the whole codebase would first need the existing debt fixed or calibrated away. A gate
on what a pull request *adds* needs neither, and it is the signal that steers the next change.

## Decision

Two checks fail a pull request, each on new code only.

**Duplication: jscpd against the base branch** (`.github/workflows/duplication.yml`, settings in
`.jscpd.json`). jscpd scans `src/` and `tests/` and builds a baseline from the base branch's tree
with the same settings; the job fails on any clone absent from it. Three kinds are checked:

- **Exact copies** of at least 50 tokens and 5 lines (jscpd's defaults).
- **Gap clones** (`maxGapLines: 3`): two copies separated by at most three inserted or changed
  lines are merged into one finding. This is the copy-then-tweak pattern.
- **Similar functions** (`similarity: 0.8`): functions whose syntax trees match after local
  names and literals are normalised. jscpd leaves test files out of this kind.

A clone's fingerprint is its content, so moving code within a file does not report it as new;
editing a line inside an existing clone does, which is deliberate, since that PR touched
duplicated code.

**Complexity: ruff `C901` at 15** (`ruff.toml`), McCabe complexity per function, run by the
existing lint job. The 19 functions already over 15, mostly flow and runner entry points, carry
`# noqa: C901`, so the gate binds new functions and functions newly pushed past the limit.

## What was measured (2026-11-03, `main` at `cc0dec14`)

- `src/` has 28 exact clones, 0.46% of lines. With gaps and similar functions on, 32. `tests/`
  has 149 exact or gap clones, 2.2% of lines. None of these fail the gate.
- **Controls.** A 100-line function copied into a new module with two lines inserted and a
  variable renamed was reported as one new gap clone at 0.99 similarity, and the job failed.
  The same file with 30 lines added at its top reported nothing new. A new 17-branch function
  failed `C901`.
- A full run against the base branch takes under half a second.
- Complexity thresholds: 40 functions in `src/` exceed 10, 19 exceed 15, 13 exceed 20. None in
  `tests/` exceed 15. 15 grandfathers a manageable list while still catching the
  large-function habit early.

## Rejected

**jscpd's complexity report.** It ranks whole files, with no per-function figure, no threshold
and no comparison against the base branch, and the ranking tracks file size. It cannot screen a
pull request. Ruff's per-function `C901` can.

**Gating jscpd's dead-code analysis.** With orchestration, providers, profiling and package
`__init__` files declared as entry points (in `.jscpd.json`, so they count as live and their
imports count as uses), it reports 30 findings, 15 at confidence 85 or above. Only one was
dead code, `config/ingest.py::auto_batch_dates` and its threshold constant, now removed. Removing
them moved both code identities (`config/ingest.py` is in both closures), accepted as a one-off:
a run resumed across the change re-infers its staged tiles, and its mosaics need
`allow_ingest_code_mismatch=true`. Of the rest:

- used by `scripts/` operator tools: four `storage/registry_rebuild.py` exports;
- used by yield-embeddings, which a scan of this repo cannot see: `daily_times`,
  `mgrs_tile_to_bbox`, `quantize_rows`;
- used only by tests: `inference/shard_record.py`, `quantize_embeddings`,
  `dequantize_embeddings`, `mgrs_tile_to_utm_epsg`;
- wrong: `_equal` in `registry_rebuild.py` is called in its own file, and `ConfigMismatchError`
  is a deliberate re-export.

Below confidence 85 most findings were wrong. Its `ignore` takes file globs only, so an export
cannot be exempted individually, and it has no baseline, so it cannot be limited to new code.
It stays configured for an on-demand run (`uv run jscpd --dead-code`) and is not in CI.

**A duplication percentage threshold.** It would fail on existing code until calibrated, then
let a large copy through as long as the total stays under the line.

## Cross-repository duplication

jscpd over this repo's `src/` and yield-embeddings' `src/` and `scripts/` found six clones
between the two. Three are the same logic maintained twice, where yield-embeddings should be
importing from here:

- `yield_embeddings/domain/coarsen.py::_slab_array_from_zarr` is a copy of
  `storage/region_writes.py::_slab_array_from_zarr`, and they have already diverged: this repo
  enumerates read blocks on the shard grid (`zarr_arr.shards or zarr_arr.chunks`), the copy on
  the inner chunk grid, so on a sharded store it issues one read task per chunk instead of per
  shard.
- `yield_embeddings/domain/roi_merge.py` re-applies the Icechunk timeout and retry settings of
  `storage/zarr_store.py`'s repository config, constants included, because `open_repo` takes no
  config to extend.
- `yield_embeddings/config/coarsen.py::compute_n_workers` repeats
  `config/assembly.py::compute_n_workers`.

The other three (two URI one-liners and a six-line flow fragment) are incidental.

## Consequences

- An agent that copies code gets a failing check naming both locations, which points it at the
  call it should have made.
- The grandfathered `# noqa: C901` list is a visible record of the functions most worth splitting.
- A deliberate duplicate (a fixture, a golden reference) is excluded in `.jscpd.json`'s
  `ignore`, which is reviewed like any other change.

## Related

- [ADR 039](039-architecture-rules-run-on-import-linter-and-ruff.md) — the import and
  banned-API rules, the other deterministic gate
