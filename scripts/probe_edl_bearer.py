#!/usr/bin/env -S uv run python
"""Diagnose whether a Bearer EDL token survives the ASF redirect chain.

Run from the repo root::

    EARTHDATA_TOKEN=<token> uv run python scripts/probe_edl_bearer.py

Prints the final HTTP status code and the full redirect chain. Used
to debug ``tests/parity/test_ingest_s1_roi_parity.py`` failures where
the Bearer-token monkeypatch reaches EDL but EDL still returns 401:

* Final status 200 → the Bearer header survived; the test failure is
  somewhere else (monkeypatch not in effect, GDAL config, etc.).
* Final status 401 → the Bearer header got stripped during one of
  the cross-domain hops. The chain output identifies which hop, so
  the fix can re-add the header for exactly that host.

This is a diagnostic-only script. It makes one live request to
NASA / ASF using your real EDL token. No cassettes, no test
runtime.
"""

from __future__ import annotations

import os
import sys

from tessera_embeddings.ingest import auth as auth_module

# A known-good OPERA RTC tile that we use elsewhere as a probe target.
# Story County, IA, July 2024 — same as the bundled quickstart fixture.
PROBE_URL = (
    "https://datapool.asf.alaska.edu/RTC/OPERA-S1/"
    "OPERA_L2_RTC-S1_T063-133417-IW3_20240701T001410Z_"
    "20240701T044329Z_S1A_30_v1.0_VV.tif"
)


def main() -> int:
    """Run the probe; return process exit code."""
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        print("ERROR: EARTHDATA_TOKEN not set in environment.", file=sys.stderr)
        print(
            "Get a token at https://urs.earthdata.nasa.gov/profile -> 'Generate Token'.",
            file=sys.stderr,
        )
        return 1

    print(f"Probe URL: {PROBE_URL[:80]}...")
    print(f"Token length: {len(token)} chars")
    print()

    session = auth_module._EDLSession()  # type: ignore[attr-defined]
    session.headers["Authorization"] = f"Bearer {token}"

    resp = session.get(PROBE_URL, allow_redirects=True, stream=True)
    try:
        print(f"Final status: {resp.status_code}")
        print(f"Final URL:    {resp.url[:120]}")
        print()
        print("Redirect chain (oldest first):")
        for h in resp.history:
            print(f"  {h.status_code}  {h.url[:120]}")
        print(f"  {resp.status_code}  {resp.url[:120]}")
        print()

        if resp.status_code == 200:
            print("SUCCESS: Bearer header survived the redirect chain.")
            print("If the S1 parity test still 401s, the monkeypatch is not")
            print("being applied. Check tests/parity/test_ingest_s1_roi_parity.py")
            print("and confirm EARTHDATA_TOKEN is set when pytest starts.")
            return 0

        if resp.status_code == 401:
            print("FAIL: Bearer header got stripped somewhere in the chain.")
            print("Look at the chain above. The hop that 401s is the one that")
            print("dropped the header. Fix the monkeypatch in")
            print("tests/parity/test_ingest_s1_roi_parity.py to re-add the")
            print("Authorization header for that specific host.")
            return 2

        print(f"UNEXPECTED status {resp.status_code}.")
        print("If it's a 5xx, retry — could be ASF transient.")
        return 3
    finally:
        resp.close()


if __name__ == "__main__":
    sys.exit(main())
