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
#
# Local cassette recording is allowed to deviate from production auth
# in two ways. These deviations live ONLY in this script:
#
#   1. EARTHDATA_TOKEN (Bearer) is preferred over EARTHDATA_USERNAME +
#      EARTHDATA_PASSWORD for the preflight. Some local EDL accounts
#      can't basic-auth against ASF (SAML/Launchpad-linked, MFA, etc.)
#      even when the same credentials work in production. Bearer
#      tokens from urs.earthdata.nasa.gov/profile sidestep this.
#
#   2. AWS_PROFILE may need to point at a us-west-2 SSO profile (e.g.
#      "cyclops") so the test phase's COG reads route through an
#      in-region identity. Production runs in us-west-2 natively.
#      Local runs from a laptop may need this hint.
#
# Production paths are unchanged: the package's auth.py uses basic
# auth via EARTHDATA_USERNAME / EARTHDATA_PASSWORD, period. Both
# deviations are recording-time conveniences only.

# 1. EDL credentials: token preferred, basic auth as fallback.
if [[ -n "${EARTHDATA_TOKEN:-}" ]]; then
    AUTH_MODE="bearer"
elif [[ -n "${EARTHDATA_USERNAME:-}" && -n "${EARTHDATA_PASSWORD:-}" ]]; then
    AUTH_MODE="basic"
else
    cat >&2 <<EOF
ERROR: EDL credentials required to record OPERA cassettes.

Set ONE of:

  EARTHDATA_TOKEN          (preferred — works around local basic-auth
                            quirks on SAML/MFA-linked accounts)

  EARTHDATA_USERNAME +
  EARTHDATA_PASSWORD       (legacy basic auth — same form production uses)

Get a token at: https://urs.earthdata.nasa.gov/profile → "Generate Token"
Get an EDL account at: https://urs.earthdata.nasa.gov/
EOF
    exit 1
fi
echo "Using EDL auth mode: $AUTH_MODE"

# 2. AWS profile hint. Some ASF endpoints behave differently for
#    requests originating outside us-west-2; if the user has a local
#    SSO profile that lands them in us-west-2, exporting AWS_PROFILE
#    before running this script may unblock the test phase. The
#    user-specified profile name lives in the AWS_PROFILE env var
#    (which we don't override here — just a reminder).
if [[ -z "${AWS_PROFILE:-}" ]]; then
    cat >&2 <<EOF
NOTE: AWS_PROFILE is not set. If the test phase fails on COG reads
      with HTTP 401/403 from ASF, try exporting an SSO profile that
      authenticates from us-west-2:

          export AWS_PROFILE=<your-us-west-2-profile>

      For dClimate users, that's typically:

          export AWS_PROFILE=cyclops

      Recording-time only — production paths don't need it.
EOF
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is required. Install via https://docs.astral.sh/uv/." >&2
    exit 1
fi

# Probe ASF with whichever credential mode the user gave us. A small
# ranged GET (instead of HEAD) avoids a CloudFront-signed-URL bug
# where HEAD against a signed URL returns 403 IncompleteSignatureException
# even when credentials are fine.
echo "Verifying EDL credentials against ASF..."
EDL_PROBE_URL="https://datapool.asf.alaska.edu/RTC/OPERA-S1/OPERA_L2_RTC-S1_T063-133417-IW3_20240701T001410Z_20240701T044329Z_S1A_30_v1.0_VV.tif"
if [[ "$AUTH_MODE" == "bearer" ]]; then
    EDL_HTTP_CODE=$(
        curl --silent --output /dev/null --location-trusted \
            --header "Authorization: Bearer $EARTHDATA_TOKEN" \
            --range 0-1023 \
            --max-time 30 \
            --write-out '%{http_code}' \
            "$EDL_PROBE_URL" \
        || echo "curl_failed"
    )
else
    EDL_HTTP_CODE=$(
        curl --silent --output /dev/null --location-trusted \
            --user "$EARTHDATA_USERNAME:$EARTHDATA_PASSWORD" \
            --range 0-1023 \
            --max-time 30 \
            --write-out '%{http_code}' \
            "$EDL_PROBE_URL" \
        || echo "curl_failed"
    )
fi

case "$EDL_HTTP_CODE" in
    200|206)
        echo "  EDL credentials OK."
        ;;
    401)
        cat >&2 <<EOF
ERROR: EDL returned HTTP 401 for the ASF datapool.

Most common causes:
  1. You have not approved the "Alaska Satellite Facility Data
     Access" application in your EDL profile. Visit
     https://urs.earthdata.nasa.gov/profile → "Authorized Apps"
     and approve it.
  2. (basic-auth mode only) Your account uses SAML / Launchpad
     SSO / MFA and basic auth against ASF doesn't work even with
     the app approved. Switch to a Bearer token:
         export EARTHDATA_TOKEN=<token from urs.earthdata.nasa.gov/profile>
     and re-run this script.
EOF
        exit 1
        ;;
    curl_failed)
        echo "ERROR: Could not reach $EDL_PROBE_URL (network or DNS issue)." >&2
        exit 1
        ;;
    *)
        echo "ERROR: EDL probe returned HTTP $EDL_HTTP_CODE — expected 200 or 206." >&2
        echo "       Check ASF status (https://asf.alaska.edu/) and retry." >&2
        exit 1
        ;;
esac

# ── Activate env ──────────────────────────────────────────────────
# We sync all four groups (dev + inference + prefect + aws) because
# pytest collection imports modules transitively across the package:
#   - dev:        pytest, pytest-recording, hypothesis
#   - inference:  ray (re-exported from tessera_embeddings.__init__)
#   - prefect:    parity tests import the Prefect flow modules
#   - aws:        providers/aws/dask.py imports dask_cloudprovider, and
#                 tests/unit/test_imports.py asserts every provider
#                 submodule imports cleanly during collection.
# Recording itself only hits Earth Search + CMR-STAC, but partial
# environments cause ImportError during collection — see the failure
# mode in the first run of this script.

uv sync --frozen --group dev --group inference --group prefect --group aws

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
Cassette recording complete. All tests passed.

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
   (2024-07, Story County, IA): ~12 S2 items, ~6 OPERA ascending
   granules. Each STAC item has a unique \`id\` field; cassettes are
   YAML, so the JSON body shows up double-escaped:

       grep -c -E '\\\\"id\\\\"' $CASSETTE_DIR/*.yaml

   Numbers will be approximate — STAC items contain nested 'id'
   fields too (assets, etc.). What you're checking for is
   "non-zero" and "in the right order of magnitude."

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

exit 0
