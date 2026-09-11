# STAC cassettes

This directory holds VCR.py recordings of STAC API responses. We use
`pytest-recording` (which wraps `vcrpy`) to make ingest tests
deterministic and fast — the alternative is querying live STAC, which
is slow, flaky, and gives a different number of items each time data
is reprocessed upstream.

## How a cassette works

1. First run with ``--record-mode=once``: VCR observes every HTTP
   request the test makes and records the response to a YAML file in
   this directory.
2. Subsequent runs: VCR intercepts the same request URLs and replays
   the recorded responses. No network. ~100 ms per request, fully
   deterministic.

## When to record

* **Adding a new ingest parity test.** Run ``pytest --record-mode=once
  tests/parity/test_<your>_parity.py``. Inspect the resulting YAML to
  confirm only the API calls you expect were captured. Commit.
* **Quarterly refresh.** Cassettes encode the STAC API's response
  shape. If a provider changes its response schema, replay starts
  failing. Re-record by deleting the cassette + running with
  ``--record-mode=once`` again.
* **Never re-record on every run.** ``--record-mode=new_episodes``
  will silently extend cassettes during normal runs and turn the
  parity test into a regression-quietener.

## File naming

**pytest-recording names the file for the test that recorded it**, so most
files here are ``<ClassName>.<test_name>.yaml`` and there is nothing to
choose. That is the naming to expect, and it means each test costs its own
full copy of the response — a reason to keep the number of tests sharing a
window small.

A test may additionally load a SHARED cassette by name with
``@pytest.mark.vcr("<name>")``, which is how the parity tests reuse
``test_s1_roi_parity.yaml`` and ``test_s2_roi_parity.yaml``. Only name a
cassette that exists: the mark is silently inert against a missing file, so a
stale name reads like coverage and provides none.

The examples this section used to list — ``s2_l2a_story_county_jul2024.yaml``
and two siblings — were never recorded, and the ``<provider>_<collection>_
<aoi-tag>_<date-range>`` convention they illustrated is not what any file here
uses.

## Filtering credentials

Earthdata Login responses can include short-lived bearer tokens. The
``conftest.py`` for any test that uses credentials must filter them
out before VCR commits to disk::

    @pytest.fixture(scope="module")
    def vcr_config():
        return {
            "filter_headers": ["authorization", "x-amz-security-token"],
            "filter_query_parameters": ["X-Amz-Signature"],
        }

Cassettes are committed to git, so any leaked token would persist in
history forever. Always inspect a freshly recorded cassette by hand
before committing.

## Why this directory is empty at v0.1.0

Cassettes are recorded against live STAC and add ~50–500 KB of YAML
per scenario. The OSS contract is to land them in a dedicated PR with
a reviewer who can verify (a) the recorded URLs are sensible, (b) no
credentials leaked, (c) the cassette replays deterministically.

The parity-test skeletons (``test_ingest_s*_roi_parity.py``) reference
the cassette filenames they will use; recording is a Phase 10
follow-up.
