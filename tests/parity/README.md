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
`test_ingest_s1_roi_parity.py` carries a credentials `skipif` that accepts **either** supported
Earthdata Login form — `EARTHDATA_TOKEN`, or `EARTHDATA_USERNAME` **and** `EARTHDATA_PASSWORD`
together — so it skips only when none of them is set. (The token is the one to prefer: many
contributor accounts are SAML- or MFA-linked and cannot basic-auth at all.) Supply either and the
test starts, and then `xfail`s for a separate reason: its committed cassette predates the native CMR
granule query ([ADR 009](../../context_docs/decisions/009-native-cmr-granule-query.md), issue #45)
and no longer matches on replay. **Neither reason is the other's fix**, so what you see depends on
your environment — a skip with no credentials, an xfail with any of them.

So `-m parity` collects ten — nine comparisons plus `adapter_template/`, which is
`@pytest.mark.skip`ped on purpose as a template to copy. **With no Earthdata credentials in the
environment** a clean serial run reports **`8 passed, 2 skipped` in about 215 seconds** (measured,
cassette replay; the real-imagery tests also read live COGs, so the network sets most of it) — the
template and S1 being the two skips. **With credentials set** the same run reports
`8 passed, 1 skipped, 1 xfailed` — the same eight comparisons and a different label on S1, because
it fails on the cassette mismatch before doing real work. Either way, read it as eight of nine
comparisons made, not as a full verification.

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

The scheduling *engine* is the same kind of flag. `test_ingest_s2_roi_frisky_parity.py`
runs the S2 domain ingest on Dask and on Frisky (`use_frisky`, see
`context_docs/ingest/frisky-experiment.md`) over the pipeline test's toy dates, which
must match byte for byte, and over the Denver cassette's real imagery. Each Frisky run
also asserts that Frisky scheduled the work, so a silent fall-back to Dask cannot pass.
It skips without the `frisky` extra.

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
