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

**A ratchet** (`.github/workflows/quality-ratchet.yml`, rules in
`.github/scripts/quality_ratchet.py`) fails a pull request that writes a copy of existing code or
makes a function worse than in the commit it merges into. It runs on every PR, stacked ones
included, against the merge commit's first parent rather than the base branch, which can move
while the job is queued.

**Duplication: code a PR writes may not be a copy.** jscpd (settings in `.jscpd.json`) runs once,
on the head, over `src/` and `tests/`, and finds three kinds of clone:

- **Exact copies** of at least 50 tokens and 5 lines (jscpd's defaults).
- **Gap clones** (`maxGapLines: 3`): two copies separated by at most three inserted or changed
  lines are merged into one finding. This is the copy-then-tweak pattern.
- **Similar functions** (`similarity: 0.8`): functions whose syntax trees match after local
  names and literals are normalised. jscpd leaves test files out of this kind.

`git diff` from the base gives the lines the PR added or changed, following renames. A clone side
holding 5 or more of those lines, or at least half of a side 5 or more lines long, is a copy the
PR made, and the job names it and the code it copies. Five is jscpd's minimum clone length, and
catches an existing copy extended by five or more lines. The half catches a short copy that `git
diff` aligns with the lines it replaced: a shared `return` or blank line reads as unchanged, and a
10-line copy over a function with the same skeleton showed only 6 lines written. A side under five
lines is the short end of a gap or similar-function clone and is not judged on its own. jscpd
scans from the repository root (`"path": ["."]` with a `pattern`), so the names it reports are the
repository paths `git diff` uses.

Only lines the PR writes are judged, so how jscpd pairs code the PR did not touch cannot fail it:
deleting a copy never fails, and neither does a jscpd upgrade or a settings change. Why the
duplication is not compared with the base is under Rejected.

**Per function** (ruff's `C901`, `PLR0913` and `PLR0911` scores, read by the script). Existing
debt carries no suppression comment.

| Measure | Limit | Binds | Over it on `main` when set | What it signals |
|---|---|---|---|---|
| McCabe complexity | 15 | every function: none may newly cross it, none over it may grow | 19 | a function too branchy to review |
| Arguments | 5 | new functions only | 191 (124 in `src/`) | a missing abstraction, or a function doing too much |
| Return statements | 3 | new functions only | 58 (42 in `src/`) | tangled control flow |

**Arguments and returns bind new functions only**: a function whose qualified name is not in
the base, so a renamed or moved one counts as new. Existing functions are left alone on these
two measures, whatever their counts, until a deliberate clean-up. Holding them at 7 and 4 was
measured first: 88 functions over 7 arguments and 27 over 4 returns across the repository,
thirty of them five or more arguments over, and those are the pipeline's entry points
(`assemble`, `ingest_tile`, `run_inference` and the like). Bringing them in is a refactor of
configuration objects across ingest, inference and assembly, not a clean-up, so it is deferred.

A new function that needs more than three exits should combine its guards or split, not nest
branches. **Prefect `@flow` functions are exempt from the argument limit**: a flow's
parameters are its interface, the one deployments and the Prefect UI expose. Everything else in
a flow module, its closures and helpers included, is held to it.

Ten of the 19 functions over 15 when the ratchet landed were split under it, each by extracting
a self-contained block. The nine left are the campaign and ingest orchestration functions
(`fill_zones_sequential`, `run_global_campaign`, `ingest_s2_roi_reflectance`,
`_process_chunks_work_stealing`, `fill_zones_sequential_flow`, `ingest_zone_year`,
`ingest_s1_roi_sar`) and two closures inside them (`_feed`, `_drive`). Their branches share
state through `nonlocal` closures, so splitting one moves branches between closures of the same
function; getting them under 15 means turning those closures into classes, a redesign of the
paths the campaign depends on rather than a cleanup. The ratchet holds each at its current score.

**Intended mirrors are marked, not refactored.** A block that must repeat another, such as a
Prefect flow's parameter docs (Prefect shows each flow's own), an explicit keyword signature
that forwards to another, or v2 model code that reproduces the v1.1 encoder as trained, is
wrapped in `# jscpd:ignore-start: <reason>` and `# jscpd:ignore-end`. For a similar-function
finding the markers must enclose the whole function. With logic duplicates extracted and
mirrors marked, `src/` reports no clones.

## What was measured (2026-11-03, `main` at `cc0dec14`)

- `src/` has 28 exact clones, 0.46% of lines. With gaps and similar functions on, 32. `tests/`
  has 149 exact or gap clones, 2.2% of lines. None of these fail the gate.
- **The duplication rule replayed over the last 30 merged PRs** (2026-11-04) flags 3, none of
  them the large clean-ups (#221, #263, #264). The one inspected, #209, is a new test repeating
  the previous test's four setup lines. With four copies of one function, jscpd pairs the new one
  against the first, so a new copy is always reported.
- **Controls.** A 100-line function copied into a new module with two lines inserted and a
  variable renamed raised two files' duplicated lines by 101 each, and the job failed. A new
  17-branch function failed, and so did one added branch in a function already at 19.
- Each jscpd run takes about a tenth of a second.
- Complexity thresholds: 40 functions in `src/` exceed 10, 19 exceed 15, 13 exceed 20. None in
  `tests/` exceed 15. 15 grandfathers a manageable list while still catching the
  large-function habit early.

## Rejected

**jscpd's complexity report.** It ranks whole files, with no per-function figure, no threshold
and no comparison against the base branch, and the ranking tracks file size. It cannot screen a
pull request. Ruff's per-function `C901` scores can.

**Comparing each file's duplicated lines between the base and the head**, the rule this ADR
first adopted. Nothing identifies the same clone across two jscpd runs: jscpd reports clones in
pairs, each new copy against the first it saw, and re-pairs them whenever a copy is deleted or
changed. A rule comparing the two runs' totals has to choose which errors to make, and review
found one after another while porting it to yield-model (Arbol-Project/yield-model#72):

- Summing each file's clone spans failed a clean-up. Deleting one of three copies pairs the
  other two against one file, counted once per partner: an untouched file "rose from 16 to 75".
- Counting each line once hid an edit that made an already-duplicated span copy one more file.
- Judging only files the PR changed fixed the clean-ups (an untouched file had "risen from 41 to
  47" when a deletion surfaced lines jscpd had folded into the deleted pair) but hid an edit that
  swapped a copy of one file for a copy of another, leaving the edited file's total unchanged.

**Failing on any clone absent from the base** (jscpd's own `--baseline-from-ref
--fail-on-new-clones`), the first version. jscpd identifies a clone by its exact extent, so an
edit that moves a clone's boundary makes it new: a `# noqa` added to a function's first line,
or a refactor that SHRINKS an existing clone. The test clean-up that cut test clones from 140 to
119 reported 21 of them as new. A rule that fails changes for reducing duplication cannot steer
toward less of it.

**One repository-wide total.** Simpler, but a PR that removes duplication in one place could
add a fresh copy elsewhere and pass.

**`# noqa: C901` on each function already over the limit**, the first version for complexity.
A suppression is permanent: the function could then grow without limit and the gate would
never fire.

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
- used only by tests: `mgrs_tile_to_utm_epsg`;
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

## Scope

The gate catches habits, not evasion. Every pull request is reviewed by a person, and the
gate's own escape, an ignore marker, is open by design, so a dodge such as a non-Prefect
decorator named `flow` or a parameter renamed to slip a count shows up in the diff, where it
belongs. The script compares the tools' output between base and head and takes their semantics
as they are rather than re-deriving them:

- Ruff's `PLR0913` skips methods decorated with `@override`, whose signature is the parent's
  and is checked there, and parameters it treats as unused (`_`-prefixed names).
- Lines a PR writes inside an existing clone count toward it. Rewriting five or more lines of a
  block that stays duplicated fails, as do moving a still-duplicated function to another file,
  reverting a commit that removed one, and a partial clean-up that rewrites a surviving copy. The
  job names the clone: merge it or mark it. None of these occurred in the replay.
- A copy completed mostly from lines already in the file, a copy split across several small PRs,
  and anything jscpd cannot see (under its minimum size, or outside `src/` and `tests/`) pass.
- A pull request is measured when it is opened, pushed or retargeted, not when its base branch
  advances. As for every other check, keeping a pull request current with its base is a
  repository setting (required up-to-date branches or a merge queue), not this workflow's job.

## Consequences

- An agent that copies code gets a failing check naming both locations, which points it at the
  call it should have made.
- A renamed file counts only the lines that changed in it. A function renamed while over a
  limit counts as new; the job's output says which, and the fix is to reduce it or split the
  rename into its own PR.
- A deliberate duplicate is marked at the site with its reason, or, for a whole directory such
  as the vendored upstream fixtures, excluded in `.jscpd.json`'s `ignore`. Both are reviewed
  like any other change.

## Related

- [ADR 039](039-architecture-rules-run-on-import-linter-and-ruff.md) — the import and
  banned-API rules, the other deterministic gate
