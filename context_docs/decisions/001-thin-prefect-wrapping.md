# 001 — Thin Prefect wrapping

**Status:** Accepted (v0.1.0)

## Context

The reference repo's pipeline is built on Prefect — `@flow` and
`@task` decorators, Blocks for secrets, the work-pool deployment
model. When porting to OSS, we had to decide how Prefect-coupled
the public package should be:

* **Option A: thin wrapping** — Prefect is the *reference*
  orchestrator, but the domain layer doesn't know about it.
* **Option B: native Prefect package** — embrace the dependency,
  optimise the API for Prefect-shaped consumers.
* **Option C: orchestrator-agnostic abstraction** — design a
  generic interface that any orchestrator can implement.

## Decision

**Option A.** Prefect lives in `orchestration/prefect/`. Everything
below that — `ingest/`, `inference/`, `storage/`, `providers/` — is
Prefect-free. The architecture-tests check enforces this with a
hard rule: `import prefect` is forbidden outside the allowed
subtree.

The Prefect-using flows are 100% real — they're not facades, they
are the production deployment. But the domain layer below them is a
plain Python library that can be imported and called from a
Dagster job, an Airflow task, a Jupyter notebook, or a shell
script.

## Rejected alternatives

**Option B (native Prefect package):** Lock-in is the product. It
also forces every downstream user — including ourselves in private
deployments — onto Prefect. Closes off Airflow, Dagster, k8s-only,
and laptop-only adoption paths.

**Option C (abstraction):** A `Runner` / `Orchestrator` /
`Workflow` interface that Prefect, Airflow, Dagster all implement.
Every attempt at this in the broader OSS ecosystem (kedro, prefect
v1's `BaseRunner`, dvc pipelines) has produced an abstraction so
thin it's harder to implement than the underlying API, or so thick
it can't evolve with the orchestrators it wraps. We'd be
maintaining the abstraction *and* a Prefect implementation; the
LOC increase isn't worth it.

## Consequences

- **Pro:** the domain layer is genuinely independent of Prefect.
  `runners/plain.py` proves it on every CI run.
- **Pro:** community adapters (Dagster, Airflow) are real
  contributions, not toy facades — they get the same domain
  functions our own production flows use.
- **Con:** there's a thin pass-through layer in
  `orchestration/prefect/tasks/` that exists just to bridge
  Prefect's context-manager API to our function-call API. ~20 LOC
  per task; we accept the duplication.
- **Con:** if a non-Prefect orchestrator user wants the same
  features Prefect ships (retries, caching, observability), they
  have to implement them in their own task shells or use their
  orchestrator's equivalents.

## Related

- [`docs/orchestrator-swap.md`](../../docs/orchestrator-swap.md) —
  worked example of swapping in Dagster.
- The broader framing is ports-and-adapters (hexagonal architecture): the domain
  declares the interfaces it needs and infrastructure implements them. ADR-004
  (duck-typed providers) and ADR-006 (adapter policy) are where that is actually
  decided for this repo.
