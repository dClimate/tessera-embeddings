# Integration tests

Tests that hit external surfaces — moto-mocked AWS, VCR-recorded STAC
responses, LocalCluster Dask, Ray local mode — but stay deterministic.

Marked `@pytest.mark.integration`. Skipped by default; opt in via
`pytest -m integration`.

## What goes here

* **Provider tests** that exercise the full ctx-manager lifecycle
  (resolve config, "spin up" a moto cluster, tear down). The lighter
  unit-level provider tests under `tests/unit/` only check pure
  helpers.
* **Cassette-backed STAC tests** for the ingest helpers. The cassette
  workflow is documented in `tests/fixtures/stac_cassettes/README.md`.
* **Plain-runner partials** like `--skip-inference` end-to-end against
  cassettes.

## What does NOT go here

* Anything that requires real AWS credentials. Use moto or skip
  cleanly via `@pytest.mark.skipif(no_aws_creds)`.
* Live STAC. Cassettes always.
* Anything > 2 minutes. Move to `tests/slow/`.
