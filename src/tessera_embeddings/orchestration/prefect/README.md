# Prefect orchestration layer

The reference orchestration for `tessera_embeddings`. Every file
under this subtree is allowed to `import prefect`; nothing outside
this subtree is. The architecture-tests in `tests/architecture/`
enforce that rule.

```
orchestration/prefect/
├── flows/                  Layer 3: @flow definitions
│   ├── generate_roi.py
│   ├── ingest_s2_roi_reflectance.py
│   ├── ingest_s1_roi_sar.py
│   ├── tessera_embeddings.py        # ROI → mosaic → inference → assembly
│   └── tessera_full_pipeline.py     # master: chains the four above via run_deployment
├── tasks/                  Layer 2: thin @task wrappers (~20 LOC each)
│   ├── ingest.py                    # process_roi_reflectance, process_roi_sar
│   └── inference.py                 # run_inference_task, assemble_embeddings_task
└── _dask_runner.py         internal helper: prefect_dask DaskTaskRunner factory
```

## Master pipeline

`tessera_full_pipeline.py` is the recommended entry point for end-to-end
runs. It dispatches the four child flows via `arun_deployment` so
cancelling the master cancels all running children:

```
generate_roi  →  ingest_s1_roi_sar       \
              →  ingest_s2_roi_reflectance ─→  tessera_embeddings
```

The two ingestion stages run concurrently. Cluster sizes are
auto-derived from ROI chunk count via
`config.dask.compute_pipeline_cluster_sizing`; pass explicit
`ingest_min_workers` / `ingest_max_workers` / `num_actors` to
override.

## Individual flows

| File | What it does |
|---|---|
| `generate_roi.py` | Rasterise a GeoJSON or S2-tile-footprint AOI into a chunked Zarr ROI mask. No cluster — runs entirely on the flow runner. Idempotent. |
| `ingest_s2_roi_reflectance.py` | Two-phase S2 L2A ingest: load only SCL for cloud screening, then load reflectance bands for days that pass coverage. Writes mosaicked `reflectance.zarr`. |
| `ingest_s1_roi_sar.py` | OPERA RTC-S1 ingest with EDL auth, batched windows, amplitude-to-dB conversion, orbit filtering. Writes `sar_<orbit>.zarr`. |
| `tessera_embeddings.py` | Distributed GPU inference: spin up Ray cluster, work-stealing dispatch across actors, Dask-based assembly. On-cancellation hook tears down EC2 instances. |
| `tessera_full_pipeline.py` | Async master flow chaining the four above via `arun_deployment`. |

## The two-flow pattern

You'll notice each flow file has an outer `@flow` and an inner
`@flow` invoked via `.with_options(task_runner=...)`. This is a
deliberate Prefect idiom — `task_runner=` must be set at flow
definition time or via `.with_options()` on a callable, which
requires a separate inner `@flow`. Don't try to collapse it. The
docstring at the top of each flow file states this explicitly.

```
ingest_s2_roi_reflectance(...)            ← outer @flow: provisions the cluster
  with ecs_cluster(...) as cluster:
    runner = DaskTaskRunner(cluster.scheduler_address)
    _ingest_s2_roi_impl.with_options(     ← inner @flow: runs against that runner
        task_runner=runner,
    )(...)
      └── process_roi_reflectance.submit(...)   ← @task: pulls client + log from context
```

## Task shells

Each `@task` shell is ~20 LOC. The pattern:

```python
@task(name="...")
def some_task(*, ...):
    """Pull client + log from context, delegate to a domain function,
    convert the dataclass result to a dict at the boundary."""
    result = some_domain_function(
        client=get_client(),
        log=get_run_logger(),
        ...
    )
    return asdict(result)
```

Domain functions never reach for context — they take `client` and
`log` as explicit parameters. That keeps them testable without a
running orchestrator (see `tests/parity/`).

## Provider selection

Each flow accepts `use_local: bool = False`. When True, the flow
imports `providers.local.dask` / `providers.local.ray` instead of
`providers.aws.*` and runs against single-machine clusters. This is
how `runners.plain` and the parity tests exercise the same code path
without AWS.

To run on a non-AWS cloud, fork the relevant flow file and replace
the `from tessera_embeddings.providers.aws.dask import ecs_cluster`
import with your provider. The domain functions don't change. See
[`docs/providers/adding-your-own.md`](../../../../docs/providers/adding-your-own.md).

## Running

```bash
# Unit-test parity (no Prefect server needed — uses prefect_test_harness):
uv run pytest tests/parity -m parity

# Local end-to-end (Local Dask + Local Ray):
uv run python -m tessera_embeddings.orchestration.runners.plain \
    examples/quickstart/config.yaml

# Production (Prefect deployment):
prefect deploy --name tessera-embeddings flows/tessera_embeddings.py
prefect deployment run 'tessera-embeddings/tessera-embeddings'
```

For Prefect-server provisioning (work pool, Blocks, deployment
templates), see [`docs/prefect-setup.md`](../../../../docs/prefect-setup.md).
