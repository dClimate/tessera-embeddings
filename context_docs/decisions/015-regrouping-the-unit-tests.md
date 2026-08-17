# 015 — Regrouping `tests/unit/` into subject directories

**Status:** Accepted, not yet executed (2026-08-17)

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

Proposed grouping, following `src/tessera_embeddings/`:

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
