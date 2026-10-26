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

Every Ray flow also takes `ami_ssm_name_by_model`, such as
`{"v1.1": "/tessera/ray/ami-id-torch-2-5-1"}`. A run of a listed model provisions from
that image and any other model from `ami_ssm_name`. **A v1.1 run requires an entry** and is
refused without one, because v1.1 never runs on the default image. Set it on the deployment, so each
run picks its worker image from the model it actually runs rather than from an
override someone has to remember (ADR 025 keeps v1.1 on its own image). A fill's model is
the one its store was seeded for.

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
inherits it. Even with `flow.fn(...)` to bypass the flow engine, an
ingest flow's task shell still runs as a real Prefect task run.

Fix: the `tests/parity/` suite wraps everything in
`prefect.testing.utilities.prefect_test_harness`, which spins up an
in-memory SQLite-backed Prefect runtime scoped to the test session.
Any future test that runs a real `@flow` should be inside parity, or
must explicitly use the harness.

### Dask scheduler timeout during graph build

Symptom: flow logs hundreds of warnings about heartbeat timeouts
while "Building Dask graph"; eventually fails.

Cause: chunk size too small → graph too big. See
[`single-vs-global.md`](single-vs-global.md#why-chunk-size-dominates-everything)
on why chunk size dominates everything.
Only *ingest* uses Dask (`INGEST_CHUNK_SIZE = 4096` storage chunks);
assembly runs as local worker processes on the flow runner — size that
container for the **full 64 GiB** (`AssemblyConfig` caps **16** workers, and a
16-worker pool was measured peaking near 40 GiB) plus the coordinator and the
commit. Anything smaller risks an OOM in the shard-write
tail rather than a clean failure. Don't reach for
`INFERENCE_CHUNK_SIZE` here: it drives GPU read-tiling, not Dask, and
must stay equal to the 2048-px shard pitch for the global store (D3).

### How an ingest flow reaches its Dask cluster

```
@flow ingest_s2_roi_reflectance(...)          ← one flow: the deployment's
  with ecs_cluster(...) as cluster:           ← provisions the fleet
    run_on_cluster(cluster.scheduler_address, ← PrefectDaskClient on that scheduler
                   process_roi_reflectance, ...)
      └── @task process_roi_reflectance       ← runs on a Dask worker, as a task
                                                run of the same flow run
```

The cluster's address exists only once the flow body has provisioned it,
so a `DaskTaskRunner` set on `@flow(task_runner=...)` has nothing to point
at. The flow instead submits its task shell through a `PrefectDaskClient`
(`flows/_dask_lifecycle.run_on_cluster`). The task's logs land on the
deployment's own flow run. See
[ADR 029](../context_docs/decisions/029-one-flow-per-ingest.md).

## Cancellation

Each long-running flow registers `on_cancellation` (and `on_crashed`)
hooks. For `tessera_embeddings.py`, the hook terminates the Ray
cluster's EC2 instances by tag: the cluster name is derived
deterministically from the Prefect flow-run id, so the hook can
recompute it even though Prefect runs hooks in a freshly imported copy
of the flow module (the flow child process — and any module state it
set — is already dead by then; `ray down` via the resolved YAML is
attempted only as an opportunistic fast path). Don't disable it —
orphaned GPU instances are expensive, and the autoscaler idle timeout
never terminates the head node.

Normally-completed runs are covered separately: the provider's
`ray_cluster` finalizer falls back to the same tag-based termination
when its `ray down` exits nonzero, so a run that finishes but can't
reach the head doesn't leave the fleet billing either. If a launch
fails in every AZ before any config resolves, the finalizer skips
`ray down` altogether (running it against the unresolved template
could match the base cluster name and tear down an unrelated cluster)
and goes straight to exact-tag termination for the generated name.

The hook is a Prefect-specific concern, so it lives **inside** the
flow file, not in the AWS provider. The provider exposes the
helpers; the flow wires them into Prefect's lifecycle.

## What we don't ship

- **CDK / Terraform / Pulumi** for the Prefect server itself.
- **The work pool job template** as a file — it's deployment-specific
  (your VPC, your IAM, your image registry).
- **A CI/CD pipeline for deploying flows** — your call. That includes anything that
  gives a dev branch its own container image or its own task-definition revision:
  deployment names are caller-supplied (see Deployments above), so the library never
  needs to know which environment it is in.

We document the env-var contract; you bring the infrastructure.
That keeps the OSS surface narrow without forcing every adopter
through our specific Prefect setup.
