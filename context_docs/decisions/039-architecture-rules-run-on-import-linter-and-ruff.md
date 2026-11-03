# 039 — The architecture rules run on import-linter and ruff

**Status:** Accepted 2026-11-03 (Robert). Replaces the checker ADR 014 describes; ADR 014's
scope decision stands.

## Context

Seven architecture rules were enforced by `tessera_embeddings.architecture_tests`, an AST
walker of our own: about 410 lines in the package plus 300 lines of tests. It shipped as a
public module with a CLI and a TOML allowlist so that downstream repos could run the same
rules against their own trees.

The walker saw only the import written in a file. A domain module importing
`providers.aws.credentials`, which imports botocore, passed every rule. Several of its
lines existed to resolve relative imports and to find the package under a `src/` wrapper,
which is work any import-graph tool already does. The three-layer architecture itself was
never checked; only specific packages were banned from specific paths.

## Decision

**Module rules run on [import-linter](https://import-linter.readthedocs.io/); name rules run
on ruff's banned-API rule (`TID251`).** The custom checker, its CLI, its allowlist format
and the `architecture-tests` extra are deleted. Contracts live in `.importlinter`; bans in
`ruff.toml`.

| Former rule | Now |
|---|---|
| `no-prefect-outside-prefect-layer` | `prefect` forbidden contract, transitive |
| `no-prefect-context-helpers-outside-prefect-layer` (`get_run_logger`) | Subsumed: the function cannot be reached without importing Prefect |
| `no-boto3-outside-aws-provider`, `no-botocore-outside-aws-provider` | `aws-domain` (transitive, domain packages) and `aws-orchestration` (direct only, since flows compose `providers.aws`) |
| `no-profiling-imports-outside-profiling` | `profiling` is the top of the `layers` contract, so nothing may import it |
| `no-dask-distributed-get_client-in-domain` | `TID251` on `dask.distributed.get_client` and `distributed.get_client`; the two legitimate imports carry `# noqa: TID251` |
| `solar-offset-applied-only-in-solar-days` | `TID251` on `solar_days.solar_day_offset_seconds` |

New: a `layers` contract (`profiling` > `orchestration` > `providers` >
`ingest : inference : storage` > `config` > `errors | utils`), exhaustive so a new top-level
subpackage must be placed. It held on the tree as it stood.

**Why two tools.** import-linter models a third-party package as its top-level module, so it
cannot name `dask.distributed.get_client`, and it checks imports, not calls. Ruff resolves
names through its semantic model and flags `from x import name`, `module.name(...)` and
relative imports alike.

**Exemptions are per line, not per file.** A per-file ignore of `TID251` would lift every
ban in that file at once, so `ingest/auth.py` could then apply the solar offset. Tests are
exempted by file, matching the old scope, because they exercise the banned names directly.

## Verification

- All four contracts and `ruff check src/ tests/` pass on the tree.
- **Control:** a planted `storage/_plant.py` importing `providers.aws.credentials` failed
  `aws-domain` and printed the chain down to botocore. The old checker passes that file.
- In a scratch tree, `TID251` flagged all five usage forms tried (a direct import, an
  attribute call through `dask.distributed`, an attribute call through an aliased
  `distributed`, an import and an attribute call of the solar offset) and left a file
  under the exempted path alone. A relative `from .solar_days import ...` was flagged too.
- `lint-imports` runs in about a second.

## Consequences

- **Downstream repos copy configuration rather than import rules.** `yield-embeddings` now
  carries its own `.importlinter` and `banned-api` table; its one deviation (in-package
  `scripts/` may import Prefect) is expressed by leaving that module out of the Prefect
  contract. Changing a rule here no longer propagates; that is the price of using standard
  tools, and the rules change rarely.
- The `get_client` ban is narrower and more accurate than before. The old rule matched any
  call named `get_client`, which is why the whole Prefect layer was exempt; Prefect's own
  `prefect.client.orchestration.get_client` is now simply not banned.
- Rules 3–5 of the README (pydantic config, fsspec storage, secrets at flow entry) were never
  machine-checked and still are not; the README now says so.

## Related

- [ADR 014](014-architecture-rules-scan-the-package.md) — why the package, not the
  repository, is scanned
- [`.importlinter`](../../.importlinter), [`ruff.toml`](../../ruff.toml)
