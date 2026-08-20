# 016 — Operator scripts: shared modules move into the package before any subdirectory layout

**Status:** Accepted, not yet executed (2026-08-17)

## Context

Both repositories keep operator tooling in a flat `scripts/` directory. This one holds 28
files and 4,895 lines; the downstream consumer holds 50 files and 16,984. Neither is scanned
by the architecture rules (ADR 014), and in the downstream repository the unscanned tree is
now larger than the scanned one.

The obvious tidy-up — group the scripts by function into subdirectories — **cannot be done
as a rename**, and the reason is worth stating precisely because it is invisible until you
try it.

Downstream, the scripts import one another by bare module name:

| Import | Files doing it |
|---|---|
| `from _deployments import …` | ~35 |
| `from _git import current_git_branch` | 13 |
| `from _prefect_ops import …` | 4 |
| `from _prefect_api import …` | 4 |
| `from _branch_infra import aws_session` | 3 |

Those resolve only because every script sits in one directory and Python puts a script's own
directory on `sys.path`. Move a script one level down and its imports stop resolving. The 28
test files under `tests/unit/scripts/` import the same modules and break identically.

This repository has the same pattern in miniature: `scripts/scale_tests/` is already a
package and imports work there, but the standalone scripts share nothing, so the coupling is
latent rather than active.

## Decision

**Move the shared, underscore-prefixed modules into the installed package first. Only then
introduce subdirectories.**

Downstream, `_deployments`, `_git`, `_prefect_api`, `_prefect_ops` and `_branch_infra` — about
800 lines — become a module inside `yield_embeddings`. Scripts then import them absolutely,
from any depth, and the subdirectory layout becomes an ordinary `git mv`.

Three things this buys beyond the layout:

1. **The coupling goes away permanently** rather than being re-expressed. A subdirectory
   layout held together by `sys.path` manipulation is more fragile than the flat one it
   replaced.
2. **800 lines move from the unscanned tree into the scanned one** (ADR 014), which is the
   only mechanism that actually shrinks that gap.
3. **The tests stop depending on directory adjacency.** `tests/unit/scripts/` currently
   imports modules that are importable only by accident of layout.

The target layout, once the prerequisite is done:

```
scripts/
├── README.md          which tier each script is in
├── deploy/            registration, task defs, work pools, locking
├── monitor/           the campaign-day tools
├── validate/          per-cell and sweep validation
├── campaign/          dispatch and cell operations
└── investigations/    kept-for-reference instruments, explicitly unmaintained
```

## The tier distinction is the part that matters most

Both repositories now carry a `scripts/README.md` listing **every** script in one of two
tiers: supported tooling, or a kept-for-reference instrument that is not maintained.

This exists because of a specific near-miss. On 2026-08-17 three scripts were shortlisted for
deletion on the evidence that nothing referenced them. Reading them showed all three were
still useful — one a deliberately reusable surgical tool with a name that reads like a
migration, one already documented at length, one cited by name inside a live test.
**Reference counting identifies undocumented code, not dead code.** An unlisted script is
indistinguishable from an abandoned one, which makes the index a safety mechanism rather
than a courtesy.

The `investigations/` tier is the other half. Instruments written to settle one question, kept
because their findings are published and anyone re-opening a finding wants the thing that
produced it, and explicitly not maintained — so a failure there raises the question of
whether to keep it, not how to repair it.

## Rejected alternatives

**`git mv` into subdirectories and add a `sys.path` shim per directory.** Works, and
preserves the fragility in a new location. It also makes each script's import behaviour
depend on a file next to it, which is harder to reason about than the flat version.

**Make `scripts/` a real package and invoke as `python -m scripts.monitor.campaign_health`.**
Correct, and it changes every invocation in every runbook, CI step and habit. The
package-module move achieves the same decoupling without touching a single command line.

**Leave the flat layout permanently.** Defensible at 28 files, not at 50 with five shared
modules and a test directory depending on their location.

**Delete the unmaintained instruments instead of tiering them.** They are the evidence base
for a threshold decision still being implemented. Deleting an instrument while its finding is
being built against is the worst available timing.

## Consequences

- The subdirectory reorganisation is blocked on the module move, and neither happens before
  the current branch merges.
- Until then, `scripts/README.md` is the mechanism that distinguishes live tooling from kept
  instruments. **A script added without a README row is the failure this ADR exists to
  prevent** — add the row in the same commit.
- The moved modules become subject to the architecture rules, mypy, and `ruff format` in CI.
  Expect the move to surface type errors that the unscanned tree never reported.
- Downstream, this supersedes the standalone plan in that repository's
  `context_docs/monitoring/toolkit-into-package-plan.md`, which describes the same move for
  the monitoring half only.

## Related

- [ADR 014](014-architecture-rules-scan-the-package.md) — why the unscanned tree exists
- [ADR 015](015-regrouping-the-unit-tests.md) — the other deferred structural move
