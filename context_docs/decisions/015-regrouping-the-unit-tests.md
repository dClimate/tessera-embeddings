# 015 — Regrouping `tests/unit/` into subject directories

**Status:** Executed (2026-09-03). Accepted 2026-08-17.

## What executing it corrected in this record

Two statements below were true when written and had stopped being true by the time the move
happened. Both are left in place rather than edited, because the reasoning around them is still
the reasoning that was followed; read them with these corrections attached.

**"Import paths inside tests are unaffected — tests import from `tessera_embeddings`, not from
each other" (Consequences) is false.** Six test files import three shared helper modules —
`zone_density.py`, `mosaic_stores.py`, `coverage_repo.py` — by absolute package path, and
`test_source_coverage.py` imported one relatively. Anyone following the consequence as written
would have moved those helpers and broken six importers. **Resolution:** the three helpers stay at
`tests/unit/` alongside `conftest.py`, for the same reason `conftest.py` does; the one relative
import was made absolute in its own commit beforehand.

**"No content edits in a move commit" (recipe step 1) could not be followed literally.** Ten files
located the source tree, the repo root or the fixtures directory by counting levels up from their
own file, so descending a directory silently changed what they resolved to. Three of those fail
*silently* rather than loudly when the anchor is wrong — an `rglob` over a nonexistent root asserts
nothing and passes, and a `parametrize` fed from one is reported as skipped rather than failed.
**Resolution:** `tests/_paths.py` anchors on the directory holding `pyproject.toml` instead of on a
level count, landed in its own commit *before* any move, which let every move commit stay the pure
rename the recipe asks for. The recipe's intent held; only its literal reading failed.

Neither belongs in `corrections-register.md`: that file is scoped to published figures that were
withdrawn, and these are stale design claims.

## What was actually built

**This tree is the layout, not the "Proposed grouping" block below it** — that one is kept as the
historical proposal and differs in three ways.

```
tests/unit/
├── conftest.py                                      fixtures for every subdirectory
├── zone_density.py  mosaic_stores.py  coverage_repo.py
│                                                    shared helpers, imported by absolute path
├── test_imports.py  test_public_api.py  test_context_docs_index.py
├── test_architecture_rules.py  test_prefect_layer.py  test_properties.py
│                                                    no single subject
├── config/                  10 files
├── ingest/                  36 files
├── inference/               18 files
├── assembly/                 7 files
├── storage/                 13 files
├── orchestration/
│   ├── flows/               14 files
│   └── runners/              4 files
├── providers/                5 files
└── profiling/                5 files
```

112 of 123 files moved; the eleven listed at the root stay there. The three helpers must not move —
six tests import them as `tests.unit.<name>`, which is the correction at the top of this record.

**Placement rule: primary subject** — what the file's docstring and test names say it tests, not
which module it imports most often. A test importing eight modules still has one subject.

Three differences from the proposal:

1. **Nine directories, not seven.** `profiling/` was added because it exists in `src/` and the
   proposal omitted it. `orchestration/` was split into `flows/` and `runners/`, mirroring
   `src/orchestration/prefect/flows/` and `src/orchestration/runners/`, which also separates
   `test_fill_zone_year_flow.py` from `test_zone_fill.py` — different subjects whose names sat
   adjacent and read as near-duplicates.
2. **Two of the proposal's hints were overridden by the placement rule.** It lists `campaign` and
   `scheduling` under `orchestration/`; their source modules are `storage/campaign.py` and
   `inference/scheduling.py`, so their tests went to `storage/` and `inference/`. The prose hints
   in that block are indicative only — the rule above decides.
3. **`assembly/` was kept as its own subject**, as proposed, rather than folded into `inference/`.
   It deliberately crosses the `inference`/`storage` boundary: `inference/assembly.py`,
   `storage/shard_writer.py` and the four provenance-record tests are one theme — what a published
   zone-year records about itself — and a strict mirror of `src/` would split it in two.

Executed in PR #171, thirteen commits: four content changes and nine pure renames, each reporting
zero insertions and zero deletions. Verified by three gates rather than by a green suite, because
three of the tests involved fail silently when their path anchor is wrong — identical pass/skip
counts (3712/1), an identical covered-source-line set (11,674, none lost) and an identical
`filename::testname` set (3,713, none gained or lost).

## Context

`tests/` holds 138 files and 41,766 lines against 45,045 lines of source. **107 of those
files sit flat in `tests/unit/`.** The largest are `test_assembly.py` (2,563 lines),
`test_scheduling.py` (1,831), `test_run_global_campaign.py` (1,334), `test_gpu_starvation.py`
(1,196) and `test_fill_zones_sequential_flow.py` (1,152).

A flat directory of that size has one concrete cost, and it is not aesthetic: **there is no
way to run the tests for one subsystem.** `pytest tests/unit -k ingest` selects on a naming
convention nobody has enforced, so the practical choice is one file or all 107.

Two things a flat layout is often blamed for turned out not to apply here, and are recorded
so nobody re-litigates them:

- **Duplication is mild.** Scanning module-level private helpers, the most repeated name
  appears in four files (`_item`), then three (`_snapshots`, `_seed`, `_run`, `_day_ds`).
  That is a normal amount of local test scaffolding, not copy-paste sprawl.
- **The suite is not oversized.** 0.93 lines of test per line of source is healthy.

So this is a navigation problem, and the fix should be scoped to navigation.

## Decision

Regroup `tests/unit/` into subject directories mirroring the package, in a **separate PR
after the current branch merges**.

The downstream repository already uses this layout — `tests/unit/coarsen/`,
`roi_grid/`, `monitoring/`, `scripts/`, `embedding_validation/` — and it works. Copy it
rather than invent one.

Proposed grouping, following `src/tessera_embeddings/` — **superseded; see "What was actually
built" above for the layout that exists.** Nine directories were built rather than these seven,
and two of the hints below (`campaign`, `scheduling`) name directories their tests did not go to:

```
tests/unit/
├── ingest/          stac, opera, duplicates, solar days, land mask, catalogue refusal
├── inference/       dataset, strip loop, model, gpu starvation
├── assembly/        assembly, shard writer, seams
├── storage/         zarr store, store manifest, zone grid, open-or-create
├── orchestration/   campaign, scheduling, zone fill, sequential fill, the flows
├── providers/       aws dask, aws ray, credentials
└── config/          inference config, fault injection
```

## Why it is deferred rather than done now

A directory move is the change most likely to conflict with in-flight work: every moved file
is a delete plus an add, so any concurrent edit to a moved test conflicts as a whole file
rather than as a hunk. Two models have been committing into this checkout, and the branch
carries 531 commits ahead of `main`. Doing it during a merge run buys nothing and risks a
resolution that silently drops a test.

## How to execute it, so the diff stays reviewable

1. **One commit per directory**, `git mv` only. No content edits in a move commit — a rename
   that also changes a line stops being reviewable as a rename.
2. **Add the `__init__.py`** each new directory needs before moving into it, in its own
   commit, so the move commits are pure.
3. **Run the full suite after each directory**, not at the end. The failure mode is a fixture
   in `tests/unit/conftest.py` that a moved file can no longer see; catching that one
   directory at a time names the cause.
4. **Do not split the large files while moving them.** Splitting `test_assembly.py` is a
   separate judgement about what it tests, and bundling it into a move makes both harder to
   review.
5. **Leave `tests/parity/`, `tests/architecture/`, `tests/integration/`, `tests/slow/` and
   `tests/gpu/` alone.** They are already grouped by *kind*, which is a different axis and a
   correct one.

## Rejected alternatives

**Split the oversized files first.** Tempting, because five files hold 8,000 lines. But file
size is not what makes the suite hard to navigate — a 2,563-line `test_assembly.py` is
findable precisely because it is named for its subject. Splitting first also multiplies the
number of files a later move has to touch.

**Group by test kind (unit / integration / slow) all the way down.** Already the top-level
axis. Adding it a second level down would produce `tests/unit/ingest/unit/`.

**Enforce a marker convention instead of directories** (`pytest -m ingest`). Markers must be
applied per test and drift silently when someone forgets; a directory cannot be forgotten,
because a file has to be somewhere.

## Consequences

- Selecting a subsystem's tests becomes `pytest tests/unit/ingest`, which is what makes an
  edit to one area cheap to verify.
- `tests/unit/conftest.py` stays at the `unit/` level and remains visible to every
  subdirectory. Fixtures used by exactly one subject may move down beside it later; that is
  a follow-up and not part of the move.
- Any open branch touching a moved test file will conflict. Land this immediately after a
  merge, when the fewest branches are outstanding.
- Import paths inside tests are unaffected — tests import from `tessera_embeddings`, not
  from each other.

## Related

- [ADR 014](014-architecture-rules-scan-the-package.md) — the other deferred structural item
