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
│                           INFERENCE_CHUNK_SIZE = 2048
│                           EMBEDDING_DIM = 128
│                           DEFAULT_NUM_OBS_CHECKPOINTS = tuple(range(8, 257, 8))
│                           checkpoint_filename(norm_source="mpc"|"aws") → str
├── time_windows.py         parse_time_window(s)  ── "Month YYYY" → TimeWindow
├── assembly.py             AssemblyConfig   ── frozen: worker-process pool sizing
├── dask.py                 compute_pipeline_cluster_sizing ── ingest cluster caps
├── providers.py            STAC PROVIDERS registry: Earth Search, CMR-STAC, PC
├── satellites.py           Band lists, baseline thresholds, SCL classes
└── environment.py          configure_gdal_environment() — call before rasterio import
                            configure_logging() — package logger level + fallback handler;
                            spawned worker processes call it at entry (they inherit none)
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
                 │                             │
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
    s1_orbit="both",               # "ascending", "descending", or "both"
    checkpoint_path=f"{paths.inputs}/models/{checkpoint_filename()}",
    inputs_bucket=paths.inputs,
    output_bucket=paths.outputs,
    num_gpus=1,                    # 0 for CPU runs
    # v1.1 sampling — bucketed deterministic resampling:
    num_obs_checkpoints=tuple(range(8, 257, 8)),  # default; multiples of 8 up to 256
    norm_source="mpc",             # "aws" for the AWS-normalised encoder checkpoint
)
```

Key architecture fields default to the v1.1 production checkpoint shape:
`latent_dim=192`, `dim_feedforward=2048`, `representation_dim=192`.
Only override if loading a different checkpoint. `batch_size=7168` controls
per-GPU pixels per forward pass within a bucket.

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

Frozen dataclass that encodes "how many worker *processes* does
assembling N ROI chunks need?" — assembly runs as a local process
pool driving raw-zarr fork/merge writes (no Dask cluster), sized by
`assemble_embeddings_task` and the plain runner.

```python
from tessera_embeddings import AssemblyConfig

cfg = AssemblyConfig()                                # chunks_per_worker=10, max_workers=16
n_workers = cfg.compute_n_workers(25)                 # → 3
```

The default cap of 16 keeps the pool around **~24 GB** — one staged-tile slice
in flight per worker, ~1–1.5 GB at a 2048-px full-band tile — which fits inside
the flow runner's 64 GiB and measured 20 GB peak in practice. Size a custom
flow-runner container against 24 GB, not 12: assembly is where the peak is. Override `chunks_per_worker` if your workload
profile differs; raise `max_workers` only with the RAM and S3 budgets
in view.

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
public API surface (see [`docs/public-api.md`](public-api.md)).
Anything underscore-prefixed is not — depend only on what is listed
there.
