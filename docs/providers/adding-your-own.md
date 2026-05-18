# Adding your own provider

This package's "provider" is the glue between a concrete cloud (or
Kubernetes, or on-prem cluster) and the orchestration layer's two
substrates: **Ray** for GPU inference, and **Dask** for ingest +
assembly.

The AWS reference at
[`src/tessera_embeddings/providers/aws/`](../../src/tessera_embeddings/providers/aws/)
is the worked example. This doc walks through what to do for a
different target, using **GCP as the running example**.

## What a provider is, and isn't

```
                              ┌──────────────────────┐
                              │   Flow / runner      │
                              │                      │
                              │   with X_cluster(    │
                              │       log,           │
                              │       …)             │
                              │   as cluster:        │
                              │       …              │
                              └──────────┬───────────┘
                                         │ "X_cluster" is your provider's
                                         │ context manager. Returns whatever
                                         │ your substrate's API expects.
                                         ▼
                              ┌──────────────────────┐
   provider boundary          │   Your provider:     │
   (everything below is       │   - reads YOUR cloud │
    your responsibility)      │     config           │
                              │   - calls YOUR cloud │
                              │     APIs             │
                              │   - yields a handle  │
                              │   - tears down on    │
                              │     exit             │
                              └──────────────────────┘
```

The **provider boundary** is a context manager with the same shape
as the AWS one. Above the boundary, the orchestration layer is
substrate-agnostic. Below the boundary, it's your cloud's idioms.

We don't ship a `Provider` base class. Providers are **duck-typed**
— if your context manager looks like the AWS one to a caller, the
caller doesn't care what's underneath. See
[`context_docs/decisions/004-duck-typed-providers.md`](../../context_docs/decisions/004-duck-typed-providers.md)
for the reasoning.

## Step 1: create the directory

```
src/tessera_embeddings/providers/gcp/
├── __init__.py
├── ray.py                   ray_cluster(...) context manager
├── dask.py                  gke_cluster(...) or dataproc_cluster(...) ctx
├── diagnostics.py           Stackdriver-backed fetch_events for inference.diagnostics
├── cluster.yaml.template    Ray autoscaler template for GCP, INJECT-commented
└── gotchas.md               operational doc
```

Lay it out flat, like AWS. No nested subpackages.

## Step 2: implement Ray

Compare the AWS shape:

```python
# providers/aws/ray.py
@contextlib.contextmanager
def ray_cluster(
    log,
    *,
    ami_ssm_name: str,            # SSM parameter holding the GPU AMI ID
    cluster_yaml: str | None = None,
    cluster_name: str | None = None,
    region: str = "us-west-2",
    ssm_prefix: str = "/tessera/ray/",
    instance_tags: list[dict] | None = None,
    sync_source_path: Path | None = None,
    code_bucket: str | None = None,
    code_suffix: str = "",
    cloudwatch_log_group: str = ...,
    cloudwatch_template: Path = ...,
) -> Iterator[str | None]:
    ...
```

Your GCP version:

```python
# providers/gcp/ray.py
@contextlib.contextmanager
def ray_cluster(
    log,
    *,
    image_uri: str,                       # GCE image instead of AMI
    cluster_yaml: str | None = None,
    cluster_name: str | None = None,
    region: str = "us-central1",
    secret_manager_prefix: str = "...",   # Secret Manager instead of SSM
    instance_labels: dict | None = None,  # GCE labels not EC2 tags
    sync_source_path: Path | None = None,
    code_bucket: str | None = None,
    code_suffix: str = "",
    logging_log_name: str = "tessera-ray", # Cloud Logging instead of CloudWatch
    ...
) -> Iterator[str | None]:
    ...
```

Same **shape** (context manager, yields a path-or-None for the
on-cancellation hook to clean up); different **arguments** (because
GCP's resource model is different from AWS's).

Don't try to share kwargs across providers. A `provider.ray_cluster(
ami_ssm_name=...)` makes no sense on GCP. Forcing every provider
through a shared kwarg list creates dead parameters and
NotImplementedErrors.

## Step 3: implement Dask

Same pattern. The AWS `ecs_cluster` reads env vars; your GCP
equivalent might read GCE metadata, or take everything as kwargs:

```python
# providers/gcp/dask.py — sketch
@contextlib.contextmanager
def gke_cluster(
    log,
    *,
    project: str,
    namespace: str,
    image: str,
    min_workers: int = 1,
    max_workers: int = 50,
    ...
) -> Iterator[GKEClusterHandle]:
    """Provision a Dask cluster on Google Kubernetes Engine."""
    ...
```

The handle needs to expose `cluster.scheduler_address` for the
flow's `DaskTaskRunner` wiring. That's the only contract — the
underlying `Cluster` class can be `dask_kubernetes.KubeCluster`,
`dask_cloudprovider.gcp.GoogleComputeCluster`, or anything else.

## Step 4: provide the diagnostics shim

The pure-domain `inference.diagnostics` module accepts a
`fetch_events` callable. AWS supplies one backed by CloudWatch:

```python
# providers/aws/diagnostics.py
def make_cloudwatch_fetcher(*, log_group, region, ...):
    client = boto3.client("logs", region_name=region)
    def _fetch(instance_id: str) -> list[dict]:
        ...   # query CloudWatch
    return _fetch
```

For GCP, write `make_stackdriver_fetcher(...)` that does the
equivalent against Cloud Logging. The diagnostics module doesn't
care which one you pass.

## Step 5: write the cluster YAML

Copy `providers/aws/cluster.yaml.template`, change `provider.type:
aws` to `provider.type: gcp` (Ray's GCP provider lives in
`ray-cloud-providers`), and re-do the INJECT comments for GCP's
node config (zone, machine type, image, labels, scopes).

Mark every GCP-account-bound field with a `# INJECT: <what>` comment
so future you can see at a glance what your provider's
`_resolve_ray_config` writes in.

## Step 6: write a gotchas.md

Document what you wished you'd known before you shipped. The AWS
gotchas covers:

- AMI bake pattern + cost rationale
- Code-sync modes (baked vs runtime)
- CloudWatch agent wiring
- Three-layer teardown defence
- Connection modes (managed / attach)

Replace AWS-specific bits with GCP equivalents but keep the
structure — it's a known-good pattern for explaining a cloud
provider's quirks.

## Step 7: wire up the flows

Two options:

**Option A: parameterise the existing flows.** Extend each flow's
`use_local: bool` toggle to a tri-state (or use Python pattern-match
on a `provider: str` parameter):

```python
@flow(name="ingest_s2_roi_reflectance")
def ingest_s2_roi_reflectance(..., provider: str = "aws"):
    if provider == "local":
        from tessera_embeddings.providers.local.dask import local_cluster
        ctx = local_cluster()
    elif provider == "gcp":
        from tessera_embeddings.providers.gcp.dask import gke_cluster
        ctx = gke_cluster(log, ...)
    else:
        from tessera_embeddings.providers.aws.dask import ecs_cluster
        ctx = ecs_cluster(log, ...)
    with ctx as cluster:
        ...
```

**Option B: fork the flow file.** Copy
`orchestration/prefect/flows/ingest_s2_roi_reflectance.py` to a
sibling, change the import, ship it. This is the "rewrite 200 lines
for your stack" promise from the README.

Option B is what the architecture is designed for — it keeps the
canonical AWS flow file from accumulating provider-selection
branches over time.

## Step 8: write the parity test

Required for any community-contributed provider:

```
tests/parity/gcp/
├── conftest.py
└── test_ingest_s2_roi_parity.py
```

Pattern is identical to
[`tests/parity/test_ingest_s2_roi_parity.py`](../../tests/parity/test_ingest_s2_roi_parity.py)
— call your GCP-routed Prefect flow and the plain-runner domain
function on the same inputs, then `assert_zarr_equivalent` the
outputs.

## Step 9: run the architecture tests

```bash
uv run python -m tessera_embeddings.architecture_tests \
    --source src/tessera_embeddings/ \
    --allowlist your-arch-allowlist.toml
```

Your allowlist permits Google SDK imports under `providers/gcp/`:

```toml
# your-arch-allowlist.toml
[allowed_imports."no-google-cloud-outside-gcp-provider"]
paths = ["providers/gcp/"]
```

(The bundled rules don't ship a `no-google-cloud-...` rule because
GCP isn't supported core. Add your own; the architecture-tests
module accepts arbitrary forbidden-import patterns.)

## Maintenance commitment

If you contribute a provider upstream (new cloud, new substrate),
the README's "Contributing" section spells out the requirements:

* Maintainer named in the provider's README.
* Parity test passing in CI.
* "Community-maintained, not core-supported" labelling.

Unmaintained providers move to `archived/` rather than getting
deleted.

## See also

- [`docs/orchestrator-swap.md`](../orchestrator-swap.md) — the
  same forking pattern, but for the orchestration layer (Prefect →
  Dagster, Airflow, Flyte).
- [`src/tessera_embeddings/providers/aws/gotchas.md`](../../src/tessera_embeddings/providers/aws/gotchas.md) —
  the operational doc the AWS provider ships, as a template.
- [`context_docs/decisions/004-duck-typed-providers.md`](../../context_docs/decisions/004-duck-typed-providers.md) —
  why no abstract `Provider` interface.
