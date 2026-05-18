# Parity tests

The parity contract is the load-bearing test category for the
orchestrator-decoupling story:

> For every flow, the Prefect-orchestrated path and the plain runner
> path produce **identical outputs** for the same inputs.

If a parity test fails, one of two things happened:

1. A regression separated the two paths.
2. A non-determinism leaked in (random seeds, timestamps, ordering).

Either way, parity tests block merge until fixed. They are not
"check this still works" tests — they are "the OSS contract".

## Running

```bash
uv run pytest -m parity                 # everything except the slow ones
uv run pytest -m "parity and slow"      # full pipeline parity
```

The default `pytest` invocation excludes `parity` so contributors
don't accidentally run the slow ones.

## Adding a parity test

1. Bring up a `LocalCluster` Dask + a local Ray runtime via the
   shared `conftest.py` fixtures.
2. Run the Prefect flow against a temp output path. Use
   `flow.fn(**kwargs)` to bypass the Prefect runtime when the flow's
   shape allows; otherwise call the flow normally and let Prefect
   record a flow run (slow but unavoidable for some flows).
3. Run the plain runner / domain function against a different temp
   output path with the same inputs.
4. Compare via :func:`tests.parity.helpers.assert_zarr_equivalent`.

`assert_zarr_equivalent` tolerates run IDs, timestamps, and other
metadata that legitimately differs between runs; it asserts identical
data arrays and critical metadata (CRS, chunks, manifest fields).

## Adapter contract

Community adapter PRs (Dagster, Airflow, …) must include a parity
test that runs their adapter against the plain runner on the same
input. Drop your test in `tests/parity/<adapter_name>/` following the
template in `adapter_template/`.
