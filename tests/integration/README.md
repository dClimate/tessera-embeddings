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

## The one live-service exception, and why it is one

`test_published_store_access.py` reads the real published store over the network. The
cassette rule is right for a third-party API, where what matters is that our parsing
still works and a recording proves it. Here the subject under test **is** the live
artifact: the claim is "somebody with no AWS account can open the published store and
get data out of it", and a replayed recording of our own request stays green through
every way that claim can break — a tightened bucket policy, a reset branch, a
reorganisation that moves the zone groups.

It needs no credentials and clears every AWS variable first, so passing with a profile
in the environment and failing without one would be a finding rather than a flaw. It is
opt-in twice — the `integration` marker plus `TESSERA_TEST_PUBLISHED_STORE=1`, which
nothing in CI sets — because a marker alone has previously been enough for a test to run
where nobody wanted it.
