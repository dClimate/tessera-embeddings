# Prefect setup

This package ships **flows**, not the **Prefect server** they run
against. Standing up the server is your responsibility — same
pattern as the Ray cluster: we tell you what to build, you bring the
infrastructure.

## What you need

```
                           ┌──────────────────────┐
                           │   Prefect server     │   self-hosted or Prefect Cloud
                           │                      │
                           │   - API              │
                           │   - Work pool        │
                           │   - Blocks           │
                           └──────────┬───────────┘
                                      │ (1) prefect deploy
                                      │ (2) prefect deployment run
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Flow runner        │   ECS Fargate task,
                           │                      │   k8s pod, or local
                           │   - imports          │   process — anywhere
                           │     tessera_embeddings   the worker can run
                           │   - calls providers/ │
                           └──────────┬───────────┘
                                      │ (3) provisions cluster
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Ray / Dask         │
                           │   on AWS (or your    │
                           │   provider of choice)│
                           └──────────────────────┘
```

## Server options

| Path | Hosted? | When |
|---|---|---|
| **Prefect Cloud** | Hosted by Prefect | Fastest to start. No infra to run. Watch costs at scale. |
| **Self-hosted** | You run it (PostgreSQL + API + UI) | Production-grade control. CDK/Terraform examples in the Prefect docs. |
| **Local** | `prefect server start` | Dev only. Single-process. Don't put production deployments here. |

This package doesn't care which you choose — it talks to whichever
API is set in `PREFECT_API_URL`.

## Work pool

Each `@flow` runs in a worker pulled from a work pool. The work
pool defines:

* **Substrate** — ECS Fargate, Kubernetes, Docker, Process, etc.
* **Default job template** — env vars, IAM, network, image.
* **Concurrency** — max simultaneous flow runs.

For the reference deployment, we use an **ECS Fargate work pool**
with a job template that sets:

```yaml
# Conceptual — the real template is your CDK/Terraform.
env:
  - name: VPC_ID
    value: vpc-...
  - name: PRIVATE_SUBNETS
    value: subnet-aaa,subnet-bbb
  - name: SECURITY_GROUP_ID
    value: sg-...
  - name: ECS_CLUSTER_ARN
    value: arn:aws:ecs:...
  - name: ECS_EXECUTION_ROLE_ARN
    value: arn:aws:iam::...
  - name: DASK_TASK_ROLE_ARN
    value: arn:aws:iam::...
  - name: DASK_ECR_IMAGE_URI
    value: <account>.dkr.ecr.us-west-2.amazonaws.com/tessera-embeddings:inference-latest
  - name: CLOUDWATCH_LOG_GROUP
    value: /ecs/tessera/dask
  - name: EARTHDATA_USERNAME
    valueFrom: <secret-arn>
  - name: EARTHDATA_PASSWORD
    valueFrom: <secret-arn>
```

These env vars are the **AWS Dask provider's contract** — they're
read by `providers/aws/dask.py::get_fargate_config` at flow start to
build the Fargate cluster. Document them in your IaC; they don't
appear anywhere in this package's code.

## Deployments

Each flow becomes a Prefect deployment. The pattern:

```bash
# Deploy a single flow
prefect deploy --name ingest-s2-roi-reflectance \
    src/tessera_embeddings/orchestration/prefect/flows/ingest_s2_roi_reflectance.py:ingest_s2_roi_reflectance

# Trigger a run
prefect deployment run \
    'ingest_s2_roi_reflectance/ingest-s2-roi-reflectance' \
    --param roi_zarr_path=s3://my-bucket/rois/zarrs/test.zarr \
    --param start_date=2024-07-01 \
    --param end_date=2024-08-01 \
    --param store_path=s3://my-bucket/mosaics/test
```

For the master pipeline, deploy `tessera_full_pipeline.py` and pass
the names of the four child deployments via the
`PipelineDeployments` pydantic parameter:

```python
prefect deployment run 'tessera-full-pipeline/master' \
    --param paths='{"inputs": "s3://...", "outputs": "s3://..."}' \
    --param time_window_end='June 2025' \
    --param tile_names='14TPK' \
    --param ami_ssm_name='/tessera/ray/ami-id'
```

The plan was deliberately **caller-supplied deployment names**, not
hardcoded — same flow can drive multiple environments (prod, dev,
staging) without code changes.

## Blocks

Prefect Blocks are the recommended way to ship secrets to flows.
**They enter at flow boundary only** — the domain functions never
load Blocks themselves (architecture rule #5). The pattern:

```python
@flow(name="ingest_s1_roi_sar")
def ingest_s1_roi_sar(...):
    edl = Secret.load("edl-credentials").get()  # at flow entry
    edl_env = {
        "EARTHDATA_USERNAME": edl["username"],
        "EARTHDATA_PASSWORD": edl["password"],
    }
    # ...inject edl_env into workers via extra_worker_env
```

A `Secret` block stores `{"username": ..., "password": ...}`. The
flow loads it once, in the flow body, and injects the values into
the worker environment. Domain code never sees `Block.load`.

> Note: the flow reads secrets in its **own body** rather than
> accepting an injected callable. Flows are launched as Prefect
> deployments whose parameters must be JSON-serializable, so a
> callable could never cross that boundary anyway — `_default_edl_env`
> is the env-var default; swap in a `Secret.load` here when needed.

## Common gotchas

### `PREFECT_API_URL` leaks into tests

Symptom: parity test fails with
`RuntimeError: Failed to reach API at https://your-staging.../api/`.

Cause: a developer's shell exports `PREFECT_API_URL` and the test
inherits it. Even with `flow.fn(...)` to bypass the outer flow, the
inner `@flow` body still calls a real Prefect runtime.

Fix: the `tests/parity/` suite wraps everything in
`prefect.testing.utilities.prefect_test_harness`, which spins up an
in-memory SQLite-backed Prefect runtime scoped to the test session.
Any future test that runs a real `@flow` should be inside parity, or
must explicitly use the harness.

### Dask scheduler timeout during graph build

Symptom: flow logs hundreds of warnings about heartbeat timeouts
while "Building Dask graph"; eventually fails.

Cause: chunk size too small → graph too big. See
[`README.md`](../README.md) §"Why chunk size dominates everything".
The package defaults to `DEFAULT_CHUNK_SIZE = 2000` — change with
care.

### ECS task definition diff between dev branches

Not relevant in this OSS package; the closed-source `yield_modeling`
deployment system uses per-branch ECS task definition revisions to
let dev branches use a different image without polluting the prod
family. If you've forked from a downstream that does this, the
machinery lives outside the OSS scope.

### Why two `@flow`s per flow file?

```
@flow ingest_s2_roi_reflectance(...)         ← outer: provisions cluster
  with ecs_cluster(...) as cluster:
    runner = DaskTaskRunner(cluster.scheduler_address)
    _ingest_s2_roi_impl.with_options(        ← inner: runs against runner
        task_runner=runner,
    )(...)
```

Prefect requires `task_runner=` to be set at flow-definition time or
via `.with_options()` on a callable, which requires an inner
`@flow`. Trying to collapse this into one decorator results in
"task runner not bound" errors. Don't refactor it. The docstring at
the top of each flow file states this explicitly.

## Cancellation

Each long-running flow registers an `on_cancellation` hook. For
`tessera_embeddings.py`, that hook tears down the Ray cluster (via
`ray down` if the resolved YAML is available, falling back to
EC2-tag-based termination). Don't disable it — orphaned GPU
instances are expensive.

The hook is a Prefect-specific concern, so it lives **inside** the
flow file, not in the AWS provider. The provider exposes the
helpers; the flow wires them into Prefect's lifecycle.

## What we don't ship

- **CDK / Terraform / Pulumi** for the Prefect server itself.
- **The work pool job template** as a file — it's deployment-specific
  (your VPC, your IAM, your image registry).
- **A CI/CD pipeline for deploying flows** — your call.

We document the env-var contract; you bring the infrastructure.
That keeps the OSS surface narrow without forcing every adopter
through our specific Prefect setup.
