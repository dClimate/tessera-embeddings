#!/usr/bin/env bash
# Record the VCR cassettes that back the parity + integration tests.
#
# This is a one-shot command, run by a human, against live STAC. It
# captures the HTTP request/response pairs that pystac-client makes
# during a normal ingest, into YAML files under
# tests/fixtures/stac_cassettes/. The cassettes are then committed
# to the repo, after which the same tests replay deterministically
# without network access.
#
# IMPORTANT: this script does NOT scrub credentials by itself. The
# scrubbing is done by pytest-recording's `filter_headers` config
# inside each test. We re-check every cassette for leaks via the
# tests/architecture/test_cassette_safety.py guard, but you should
# also eyeball the YAMLs by hand before committing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Pre-flight checks ─────────────────────────────────────────────

if [[ -z "${EARTHDATA_USERNAME:-}" || -z "${EARTHDATA_PASSWORD:-}" ]]; then
    cat >&2 <<EOF
ERROR: Earthdata Login credentials are required to record OPERA
       cassettes. Export both env vars before running this script:

           export EARTHDATA_USERNAME=<your-edl-username>
           export EARTHDATA_PASSWORD=<your-edl-password>

       Get an EDL account at https://urs.earthdata.nasa.gov/.
EOF
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is required. Install via https://docs.astral.sh/uv/." >&2
    exit 1
fi

# ── Activate env ──────────────────────────────────────────────────
# Need dev (pytest, pytest-recording) AND inference (ray, torch).
# tessera_embeddings/__init__.py re-exports run_inference, which
# imports ray at module load — collection crashes without it even
# though the parity tests don't actually invoke Ray.

uv sync --frozen --group dev --group inference

# ── Record ────────────────────────────────────────────────────────

CASSETTE_DIR="tests/fixtures/stac_cassettes"
mkdir -p "$CASSETTE_DIR"

echo
echo "Recording cassettes against live STAC. This will hit:"
echo "  - https://earth-search.aws.element84.com/v1  (S2 L2A)"
echo "  - https://cmr.earthdata.nasa.gov/stac/ASF     (OPERA RTC search)"
echo "  - https://cmr.earthdata.nasa.gov/search/granules.json (OPERA orbit filter)"
echo

# Run the cassette-recording tests. These are the integration + parity
# tests that have been written to use pytest-recording. --record-mode=once
# means: only record what's missing; never re-record an existing cassette
# (use --record-mode=rewrite to refresh).
RECORDING_EXIT=0
uv run pytest \
    tests/integration/test_stac_query_cassette.py \
    tests/parity/test_ingest_s2_roi_parity.py \
    tests/parity/test_ingest_s1_roi_parity.py \
    -m "integration or parity" \
    --record-mode=once \
    -v \
    || RECORDING_EXIT=$?

# ── Inspection checklist ──────────────────────────────────────────

if (( RECORDING_EXIT != 0 )); then
    cat >&2 <<EOF

────────────────────────────────────────────────────────────────────
Recording exited with status $RECORDING_EXIT before all cassettes
were captured.

Common causes:
- Collection error (ImportError): a dep is missing from the env. Run
  'uv sync --frozen --group dev --group inference' and try again.
- Network error reaching STAC: retry; if it persists, check whether
  Earth Search or CMR is degraded.
- Test failure after cassette write: VCR may have written a partial
  cassette — DO NOT COMMIT IT. Delete tests/fixtures/stac_cassettes/*.yaml
  and re-run.

Skipping the inspection checklist. Fix the failure and re-run this
script before committing anything.
────────────────────────────────────────────────────────────────────
EOF
    exit "$RECORDING_EXIT"
fi

cat <<EOF

────────────────────────────────────────────────────────────────────
Cassette recording complete (or partially complete — review above).

NEXT STEPS — DO NOT COMMIT THE CASSETTES UNTIL THIS CHECKLIST PASSES:

1. Run the safety guard:

       uv run pytest tests/architecture/test_cassette_safety.py -v

   Should pass. If it fails, your auth filtering missed something —
   inspect the offending YAML, fix the test's vcr_config, re-record.

2. Inspect each cassette by hand for surprises. Focus on:

       grep -in -E '(authorization|cookie|token|signature|bearer)' \\
           $CASSETTE_DIR/*.yaml

   Should return zero matches (the safety guard checks the obvious
   ones, but a surprising header field in a 30x redirect can slip
   through).

3. Verify item counts match expectations. As of last check
   (2024-07, Story County, IA):
     - s2_l2a:    ~12 items
     - opera_rtc: ~6 ascending granules

       grep -c '"id":' $CASSETTE_DIR/*.yaml

4. Verify file sizes are reasonable. Each cassette should be
   50–500 KB. Anything > 1 MB suggests an unexpected payload was
   captured; investigate before committing.

       du -h $CASSETTE_DIR/*.yaml

5. Run the parity tests in replay mode (no --record-mode):

       uv run pytest tests/parity/ -m parity -v

   Should pass without network access. If a test fails with a
   "could not find cassette" error, the request URL changed between
   recording and replay — check for nondeterminism in the test
   inputs (timestamps, randomised IDs).

6. Once all checks pass, commit only the cassette YAMLs:

       git add $CASSETTE_DIR/*.yaml
       git commit -m 'cassettes: record S2 + OPERA STAC fixtures (Story County IA, Jul 2024)'

────────────────────────────────────────────────────────────────────
EOF

exit "${RECORDING_EXIT:-0}"
