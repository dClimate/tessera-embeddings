# Parity tests

The parity contract is the load-bearing test category for the
orchestrator-decoupling story:

> For every flow, the Prefect-orchestrated path and the plain runner
> path produce **identical outputs** for the same inputs.

**Read "every flow" literally: this tier compares one stage at a time.** Each test pairs a single
Prefect flow against the domain function beneath it. Nothing here runs the two paths over the whole
pipeline at once, and nothing here invokes `run_plain` — that comparison was a deleted stub, and the
end-to-end path is now verified by running the quickstart by hand
([ADR 023](../../context_docs/decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md)).
So the contract holds per stage; it is not evidence that a full run of the two paths agrees.

**And one of the seven real comparisons is not running, for two unrelated reasons at once.**
`test_ingest_s1_roi_parity.py` carries a credentials `skipif` — Earthdata Login, which many
contributor laptops cannot basic-auth against — so without `EARTHDATA_TOKEN` it never starts. Set
the token and it starts and then `xfail`s, for a separate reason: its committed cassette predates
the native CMR granule query ([ADR 009](../../context_docs/decisions/009-native-cmr-granule-query.md),
issue #45) and no longer matches on replay. **Neither reason is the other's fix.**

So `-m parity` collects eight — seven comparisons plus `adapter_template/`, which is
`@pytest.mark.skip`ped on purpose as a template to copy — and a clean local run reports
**`6 passed, 2 skipped`**. That is the healthy result here, not a fully-verified one: read it as six
of seven comparisons made, with the template accounting for the other skip.

If a parity test fails, one of two things happened:

1. A regression separated the two paths.
2. A non-determinism leaked in (random seeds, timestamps, ordering).

Either way, parity tests block merge until fixed. They are not
"check this still works" tests — they are "the OSS contract".

## Running

```bash
uv run pytest -m parity                 # all of them
```

There is no `-m "parity and slow"` target. It used to select a full-pipeline stub, which is
deleted — the end-to-end path is verified by running the quickstart by hand
([ADR 023](../../context_docs/decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md)).

The default `pytest` invocation excludes `parity` so contributors
don't accidentally run the slow ones.

## A second kind of parity: performance flags

Some parity tests compare two configurations of the *same* path rather than two
orchestrators. A performance flag meant to change only *when* or *how
concurrently* work happens — `pipeline_dates`, in
`test_ingest_s2_roi_pipeline_parity.py` — is safe only if the store is identical
with the flag on and off, which is what lets its default be flipped on evidence.

Those tests take the same shape (two temp stores, one `assert_zarr_equivalent`)
and carry the same weight, but a failure means a third thing beyond the two
above: the optimisation is not semantics-preserving. That is a STOP rather than a
fix-forward — the flag stays off until the stores match.

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
