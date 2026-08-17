# 014 — The architecture rules scan the package, not the repository

**Status:** Accepted (2026-08-17)

## Context

`.github/workflows/architecture.yml` runs the rule checker as:

```
python -m tessera_embeddings.architecture_tests --source src/tessera_embeddings/
```

Seven rules are enforced: `solar-offset-applied-only-in-solar-days`,
`no-prefect-outside-prefect-layer`, `no-prefect-context-helpers-outside-prefect-layer`,
`no-dask-distributed-get_client-in-domain`, `no-boto3-outside-aws-provider`,
`no-botocore-outside-aws-provider`, and `no-profiling-imports-outside-profiling`.

**`scripts/` is not scanned.** That is 28 files and 4,895 lines, of which
`scripts/scale_tests/` is roughly two thirds. Downstream, the same shape is far more
pronounced: `yield-embeddings` scans `src/yield_embeddings/` (12,458 lines) while its
`scripts/` holds **16,984 lines** — 36% larger than the tree the rules cover — plus 6,033
lines of CDK infrastructure. Sixteen files there import `prefect` or `boto3` from paths
that match neither the permitted `orchestration/prefect/` nor `providers/aws/` prefix.

This was not a decision anyone took. It is the default that follows from pointing
`--source` at the package, and it went unstated long enough to be mistaken for coverage.

## Decision

**The scanned boundary is the installed package, and that is now written down rather than
implied.** Code in `scripts/` is outside the rules and is expected to be.

The reasoning is that the rules encode *library layering* — which import may appear in which
layer of a package that other people install. An operator script is not a layer. It is
allowed to reach for boto3, Prefect and a Dask client in one file, because that is what
composing the library from outside looks like, and a rule forbidding it would be enforcing
a constraint that does not apply.

**What follows from this, and is the actual cost:** anything that genuinely belongs to the
library and happens to live in `scripts/` is unprotected, and nothing announces that. The
remedy is not to widen the scan. It is to move such code into the package, where the rules
already apply.

## The one rule where this was worth checking rather than assuming

`solar-offset-applied-only-in-solar-days` is the rule with correctness rather than tidiness
consequences: the solar-day offset must be applied exactly once, at the query chokepoint, and
a second application silently shifts every acquisition date. A second call site inside an
unscanned `scripts/` would be invisible to CI.

**Verified 2026-08-17: it does not happen.** The only non-test occurrences of
`solar_day_offset_seconds` are `ingest/solar_days.py` and the rule definition itself. The
gap is structural rather than currently breached — which is precisely why it is recorded
here instead of being left as a thing someone happens to know.

## Rejected alternatives

**Point `--source` at the repository root.** Every operator script would fail on day one,
and the honest fix for each would be an allowlist entry. An allowlist with one entry per
script is a list of exemptions that means nothing, and it would train readers to add a line
rather than ask a question.

**Extend the scan and fix the violations.** In this repository that is nearly free; in
`yield-embeddings` it is sixteen files, and those files import each other by bare module
name because they share one directory, so moving them is a prerequisite rather than a
consequence. See ADR 016.

**Say nothing.** The status quo. Rejected because a rule set that appears to cover a
repository and covers half of it is worse than one whose edges are stated: the first invites
the assumption that CI would have caught it.

## Consequences

- The `--source` argument in `architecture.yml` is load-bearing documentation. Changing it
  changes what the project claims to enforce, and should be accompanied by an ADR.
- Reviewers cannot infer from a green Architecture job that a change respects the layering,
  unless the change is inside `src/`.
- Downstream repositories extend the rules through a TOML allowlist and scan their own
  package the same way. The allowlist expresses *deviations within* the scanned tree; it has
  nothing to say about code outside it.
- The follow-up that actually shrinks this gap is moving shared operator modules into the
  package — tracked in ADR 016 and, downstream, in that repo's
  `context_docs/monitoring/toolkit-into-package-plan.md`.

## Related

- [ADR 016](016-scripts-layout-and-shared-modules.md) — the move that reduces the unscanned
  tree
- [`src/tessera_embeddings/architecture_tests/`](../../src/tessera_embeddings/architecture_tests/)
  — the rules and the allowlist parser
