# Running without Prefect

Prefect is the **reference orchestrator** for `tessera_embeddings`,
not a requirement. The architecture is designed so the flow layer
can be replaced without touching domain code. This doc walks through
exactly what that means in practice, using **Dagster as the worked
example**.

## The contract

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 3:   Orchestrator-specific definitions                 │
│            (Prefect @flow / Dagster job / Airflow DAG / …)   │
│  --- swap this layer ---                                     │
└──────────────────────────────────┬───────────────────────────┘
                                   │ delegates to
┌──────────────────────────────────▼───────────────────────────┐
│ Layer 2:   Thin task shells                                  │
│            Pull `client` and `log` from orchestrator context, │
│            call domain function, convert result to dict.     │
│  --- rewrite this layer (~20 LOC per task) ---               │
└──────────────────────────────────┬───────────────────────────┘
                                   │ delegates to
┌──────────────────────────────────▼───────────────────────────┐
│ Layer 1:   Domain functions                                  │
│            Plain Python; client/log as parameters.           │
│  --- you don't touch this ---                                │
└──────────────────────────────────────────────────────────────┘
```

Layers 1 and 2 are unchanged across orchestrators. Only Layer 3
needs rewriting.

## The Prefect reference

The S2 ingest flow looks like:

```python
# src/tessera_embeddings/orchestration/prefect/flows/ingest_s2_roi_reflectance.py
from prefect import flow, get_run_logger
from prefect_dask import DaskTaskRunner

from tessera_embeddings.providers.aws.dask import ecs_cluster
from tessera_embeddings.orchestration.prefect.tasks.ingest import (
    process_roi_reflectance,
)


@flow(name="ingest_s2_roi_impl")
def _impl(*, roi_zarr_path, start_date, end_date, store_path, ...):
    future = process_roi_reflectance.submit(
        roi_zarr_path=roi_zarr_path,
        start_date=start_date,
        end_date=end_date,
        store_path=store_path,
        ...
    )
    return future.result()


@flow(name="ingest_s2_roi_reflectance")
def ingest_s2_roi_reflectance(...):
    log = get_run_logger()
    with ecs_cluster(log, min_workers=1, max_workers=50) as cluster:
        runner = DaskTaskRunner(address=cluster.scheduler_address)
        return _impl.with_options(task_runner=runner)(...)
```

And the @task shell it submits:

```python
# src/tessera_embeddings/orchestration/prefect/tasks/ingest.py
from dask.distributed import get_client
from prefect import task, get_run_logger

from tessera_embeddings.ingest.s2_roi import ingest_s2_roi_reflectance


@task(name="process-roi-reflectance")
def process_roi_reflectance(*, roi_zarr_path, start_date, end_date, store_path, ...):
    result = ingest_s2_roi_reflectance(   # ← the domain function
        roi_zarr_path=roi_zarr_path,
        start_date=start_date,
        end_date=end_date,
        store_path=store_path,
        client=get_client(),              # ← from Dask context
        log=get_run_logger(),             # ← from Prefect context
        ...
    )
    return asdict(result)
```

## The Dagster equivalent

Same domain function, different glue. Sketch:

```python
# my_dagster_project/orchestration/dagster/ops/ingest.py
from dagster import In, Nothing, OpExecutionContext, op
from dask.distributed import Client

from tessera_embeddings.ingest.s2_roi import ingest_s2_roi_reflectance


@op(required_resource_keys={"dask_client"})
def process_roi_reflectance(
    context: OpExecutionContext,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
) -> dict:
    """Dagster op shell. Pulls client + logger from context;
    delegates to domain.
    """
    client: Client = context.resources.dask_client
    result = ingest_s2_roi_reflectance(   # ← same domain function
        roi_zarr_path=roi_zarr_path,
        start_date=start_date,
        end_date=end_date,
        store_path=store_path,
        client=client,                    # ← from Dagster resource
        log=context.log,                  # ← from Dagster context
    )
    return asdict(result)
```

```python
# my_dagster_project/orchestration/dagster/jobs/ingest.py
from contextlib import contextmanager

from dagster import job, op, resource

from tessera_embeddings.providers.aws.dask import ecs_cluster


@resource
@contextmanager
def aws_dask_client(init_context):
    """Dagster resource wrapping the AWS Dask provider."""
    log = init_context.log
    with ecs_cluster(log, min_workers=1, max_workers=50) as cluster:
        client = Client(cluster.scheduler_address)
        try:
            yield client
        finally:
            client.close()


@job(resource_defs={"dask_client": aws_dask_client})
def ingest_s2_roi_reflectance_job():
    process_roi_reflectance()
```

## What changed

```
                              Prefect              Dagster
                              ─────────            ─────────
flow definition               @flow                @job
task definition               @task                @op
context fetching:
  client                      get_client()         context.resources.dask_client
  logger                      get_run_logger()     context.log
cluster lifecycle:
  ctx manager                 inside @flow body    @resource (also a ctx manager)
task runner binding:
  Dask                        prefect_dask.DaskTaskRunner   resource_defs={"dask_client": ...}
parameter passing             function args        op inputs / config
result conversion             dataclass→dict       dataclass→dict
```

Note what **didn't** change:

* The domain function (`ingest_s2_roi_reflectance` from
  `tessera_embeddings.ingest.s2_roi`) is identical.
* The provider (`ecs_cluster` from
  `tessera_embeddings.providers.aws.dask`) is identical.
* The result dataclass (`IngestResult`) is identical.
* The error handling, the retry policy (tenacity, inside the domain
  function), the credential refresh logic — all unchanged.

## How to verify it works

The **parity contract**: your adapter must produce identical output
to `runners/plain.py` for the same inputs.

```
┌──────────────────────┐       ┌──────────────────────┐
│   plain.py           │       │   your adapter       │
│  (no orchestrator)   │       │  (Dagster job)       │
│                      │       │                      │
│  identical inputs    │       │  identical inputs    │
│         │            │       │         │            │
│         ▼            │       │         ▼            │
│  identical domain    │       │  identical domain    │
│  function call       │       │  function call       │
│         │            │       │         │            │
│         ▼            │       │         ▼            │
│  output Zarr A       │  ===  │  output Zarr B       │
└──────────────────────┘       └──────────────────────┘
                  │                    │
                  └────────┬───────────┘
                           ▼
              assert_zarr_equivalent(A, B)
              from tests/parity/helpers.py
```

To accept a community adapter, we require this test to pass in CI.
See [`tests/parity/adapter_template/`](../tests/parity/adapter_template/)
for the starter template.

## Other orchestrators

Same pattern works for:

| Orchestrator | Layer-3 idiom | Layer-2 fetch idiom |
|---|---|---|
| Prefect | `@flow` + `@task` | `get_run_logger()`, `get_client()` |
| Dagster | `@job` + `@op` | `context.log`, `context.resources.<name>` |
| Airflow | `DAG` + `PythonOperator` | `**context['ti'].kwargs`, custom hooks |
| Flyte | `@workflow` + `@task` | `flytekit.current_context()` |
| Argo Workflows | YAML + container steps | env vars / mounted secrets |
| Kubeflow Pipelines | `@dsl.pipeline` + `@func_to_container_op` | passed-in args |

The Layer-1 domain functions don't care which one you pick.

## What we accept upstream

Community-maintained orchestrator adapters land in
`src/tessera_embeddings/orchestration/<adapter_name>/` with a parity
test under `tests/parity/<adapter_name>/`. Maintenance commitment is
required — see the README's "Contributing" section.

The architecture-tests check
([`docs/public-api.md`](public-api.md)) auto-generates an allowlist
entry for any future orchestrator subdirectory, so adapter PRs don't
trip the "no orchestrator imports outside the orchestration layer"
rule.
