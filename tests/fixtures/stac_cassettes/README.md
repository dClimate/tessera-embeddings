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

**pytest-recording names the file for the test that recorded it**, so a
plain ``@pytest.mark.vcr`` writes ``<ClassName>.<test_name>.yaml`` and each
test costs its own full copy of the response.

A test or class may instead load a SHARED cassette by name with
``@pytest.mark.vcr("<name>")``. ``test_s2_roi_parity.yaml`` is shared this
way by the S2 parity test. ``TestRecordedProducerSplit.yaml`` is shared by
every test in that integration class through
``@pytest.mark.default_cassette("<name>")`` plus ``@pytest.mark.vcr``, which
also makes a re-record write to that one file — one recording rather than
one copy per test. Only name a cassette that exists: the mark is silently inert against a
missing file, so a stale name reads like coverage and provides none.

Current files:

| Cassette | Replayed by |
| --- | --- |
| `test_s2_roi_parity.yaml` | `tests/parity/test_ingest_s2_roi_parity.py` (recorded by `scripts/build/record_stac_cassettes.py`) |
| `TestRecordedProducerSplit.yaml` | `tests/integration/test_baseline_producer_cassette.py::TestRecordedProducerSplit` |
| `TestARealItemIsOwedTheCorrection.test_the_version_does_not_decide_which_images_keep_the_offset.yaml` | the same file's `TestARealItemIsOwedTheCorrection` |

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
