# 002 — Shape C: thin task shells over domain functions

**Status:** Accepted (v0.1.0)

## Context

Given [001](001-thin-prefect-wrapping.md) — Prefect lives only in
`orchestration/prefect/` — we still need to decide how the
Prefect-using `@task` shells relate to the underlying domain
functions. Three shapes were considered:

* **Shape A: domain function = task body.** The domain function
  takes Prefect context as a parameter; the `@task` is a one-line
  decorator over it.
* **Shape B: task wraps domain, returns prefect-typed result.** The
  task does business logic + Prefect-shaped return values
  (PrefectFutures, structured logs).
* **Shape C: task is a thin pass-through.** The task pulls
  context-bound objects (logger, Dask client) once at the boundary,
  hands them off as plain function parameters to the domain
  function, converts the result to a dict.

## Decision

**Shape C.** Each `@task` shell is ~20 LOC:

```python
@task(name="process-roi-reflectance")
def process_roi_reflectance(*, roi_zarr_path, ...):
    result = ingest_s2_roi_reflectance(
        roi_zarr_path=roi_zarr_path,
        client=get_client(),         # ← pulls from Dask context
        log=get_run_logger(),        # ← pulls from Prefect context
        ...
    )
    return asdict(result)            # ← dataclass → dict at the boundary
```

Domain functions never call `get_run_logger()`, `get_client()`, or
read Prefect/Dask context in any other way. They take `client` and
`log` as keyword parameters.

## Rejected alternatives

**Shape A:** Domain functions become Prefect-aware. Violates
[001](001-thin-prefect-wrapping.md) — the architecture-tests check
would fail because `get_run_logger` would appear in domain modules.
And the "domain function" testability advantage evaporates: every
unit test now needs a Prefect runtime to call the function.

**Shape B:** Tasks own business logic. Means the same logic exists
twice (once in the task, once in the plain runner) and they drift
in subtle ways. Tested with prototypes; the drift was measurable
within weeks.

## Consequences

- **Pro:** `runners/plain.py` and the Prefect flows call the
  identical domain function. The parity test
  (`assert_zarr_equivalent`) is a meaningful contract.
- **Pro:** task shells are short enough to be obviously correct on
  inspection. Code review for new tasks is fast.
- **Con:** there's an unavoidable ~20 LOC of boilerplate per task.
  Worth it.
- **Con:** dataclass→dict conversion happens at the boundary
  because Prefect's UI prefers dicts. The dataclass is the
  source-of-truth shape; the dict is just for display.

## Related

- [`docs/orchestrator-swap.md`](../../docs/orchestrator-swap.md) —
  applying Shape C to non-Prefect orchestrators.
- [`design/orchestration_infra_leakage_audit.md`](../design/orchestration_infra_leakage_audit.md) —
  the audit that informed this decision.
