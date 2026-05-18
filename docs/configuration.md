# Configuration

`tessera_embeddings` uses **pydantic models** for everything that
spans flow / task / domain boundaries, plus a couple of frozen
dataclasses for purely-internal recipes. There is no environment-based
config-by-magic — every config object enters the system at the flow
boundary as an explicit parameter.

## The configuration tree

```
config/
├── paths.py                BucketPaths   ─── pydantic, deployment-supplied storage URIs
├── inference.py            InferenceConfig ── frozen dataclass: model, sampling, Ray-actor
│                           TimeWindow      ── 12-month rolling window
│                           DEFAULT_CHUNK_SIZE = 2000
│                           EMBEDDING_DIM = 128
│                           checkpoint_filename(quantized=True)
├── time_windows.py         parse_time_window(s)  ── "Month YYYY" → TimeWindow
├── dask.py                 AssemblyConfig   ── frozen: chunks_per_worker scaling
│                           compute_pipeline_cluster_sizing
├── providers.py            STAC PROVIDERS registry: Earth Search, CMR-STAC, PC
├── satellites.py           Band lists, baseline thresholds, SCL classes
└── environment.py          configure_gdal_environment() — call before rasterio import
```

## The contract: caller-supplied URIs, never env-derived

The reference repo had a `dev: bool` toggle that selected hardcoded
S3 buckets. The OSS package has `BucketPaths` instead — a pydantic
model the deployment fills in at flow entry:

```python
from tessera_embeddings import BucketPaths

paths = BucketPaths(
    inputs="s3://my-org-tessera-inputs",
    outputs="s3://my-org-tessera-outputs",
    preprocessed="s3://my-org-tessera-mosaics",
)
```

Every domain function that touches storage takes `paths` (or one of
its derived URIs) as a keyword. There's no `os.environ.get(...)`
anywhere in the domain layer — the architecture-tests check enforces
this.

```
                 ┌─────────────────────────────┐
caller / config  │  BucketPaths                │
─────────────────┤    inputs: <uri>            │ pydantic: validates at boundary
                 │    outputs: <uri>           │
                 │    preprocessed: <uri>      │
                 └────────────┬────────────────┘
                              │
                              ▼
                 ┌─────────────────────────────┐
flow             │  ingest_s2_roi_reflectance(│
                 │    store_path=paths.store_for("my-roi", "reflectance"),
                 │    client=client,           │
                 │    ...)                     │
                 └────────────┬────────────────┘
                              │
                              ▼
                 ┌─────────────────────────────┐
domain function  │  uses store_path + an       │
                 │  fsspec FS to read/write    │
                 │  — no env reads             │
                 └─────────────────────────────┘
```

## InferenceConfig

Frozen dataclass. Holds the model architecture (must match the
checkpoint), inference parameters, and Ray actor resource
reservation. Construct via the helper rather than the raw class so
defaults stay consistent:

```python
from tessera_embeddings import (
    InferenceConfig, TimeWindow, parse_time_window, checkpoint_filename,
)

config = InferenceConfig(
    time_window=parse_time_window("June 2025"),
    s1_orbit="ascending",
    checkpoint_path=f"{paths.inputs}/models/{checkpoint_filename(quantized=True)}",
    inputs_bucket=paths.inputs,
    output_bucket=paths.outputs,
    num_gpus=1,                # 0 for CPU runs
    repeat_times=3,
    sample_size_s2=40,
    sample_size_s1=40,
)
```

The model-architecture fields (`latent_dim`, `nhead`,
`num_encoder_layers`, etc.) default to the production checkpoint's
shape. Only override if you're loading a different checkpoint.

## Secrets enter at the flow boundary

Hard rule #5 from the architecture contract: secrets are
caller-supplied, never read inside a domain function.

For S1 ingest, this looks like:

```
  flow runner / shell
        │
        │  reads EARTHDATA_USERNAME / PASSWORD from env / Prefect Block
        │
        ▼
  ingest_s1_roi_sar.py @flow
        │
        │  edl_credentials_fn = lambda: {"AccessKeyId": ..., ...}
        │  apply_credentials_fn = lambda creds: set_s3_credentials(creds)
        │
        ▼
  ingest_s1_roi_sar() domain function
        │
        │  uses the callbacks; refreshes every 30 min;
        │  never reads os.environ for AWS_* itself
        ▼
```

`set_s3_credentials` (from `ingest/auth.py`) is the substrate-aware
half: it sets env vars on the local process and registers a Dask
`WorkerPlugin` so workers see the same creds. The domain function
just calls the callbacks on schedule.

## TimeWindow

A 12-month window, parsed from a `"Month YYYY"` string. Used by
both ingest (date range for STAC queries) and inference (which
months of the mosaic to sample from):

```python
from tessera_embeddings import parse_time_window

window = parse_time_window("June 2025")
# → 12 months ending at and including June 2025: July 2024 … June 2025
```

The string form is the only public input — the underlying
`TimeWindow` shape is implementation detail.

## AssemblyConfig

Frozen dataclass that encodes "how big a cluster do I need to
assemble N ROI chunks?" — used by the master pipeline's
auto-sizing logic and by `assemble_embeddings_task`.

```python
from tessera_embeddings import AssemblyConfig

cfg = AssemblyConfig(chunks_per_worker=40, max_workers=200)
n_workers = cfg.compute_n_workers(n_live_chunks=850)  # → 22
```

Calibration: ~850 live chunks → 20 workers → 200 workers ceiling for
dense ROIs. Override `chunks_per_worker` if your workload profile
differs.

## What's NOT pydantic

- Cluster YAML templates (`providers/aws/cluster.yaml.template`) —
  YAML, not pydantic. They're configuration for an external tool
  (`ray up`), not Python-internal config.
- AWS resource IDs (security groups, subnets, IAM, key pairs) — they
  live in **SSM Parameter Store**, not in a pydantic model. The AWS
  Ray provider reads them at flow start. See
  [`docs/providers/aws.md`](providers/aws.md) for the SSM key list.

## Validating config at flow entry

Every flow entry validates its parameters via pydantic. This catches
mistakes once, at the obvious boundary, instead of letting them
propagate as runtime errors deep inside the pipeline. If you see a
`pydantic.ValidationError` on flow start, the message points at
exactly which field was bad.

## Public API

The pydantic models and helpers documented above are part of the
public API surface (see [`docs/public-api.md`](public-api.md)). The
internal helpers (e.g. `_chunkscaledClusterConfig` base class) are
not — depend only on what's listed.
