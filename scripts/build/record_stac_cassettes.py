#!/usr/bin/env -S uv run python
"""Standalone cassette recorder.

Hits ONLY the STAC HTTP endpoints (Earth Search, CMR-STAC, CMR
Granule Search) and writes the request/response pairs to VCR
cassettes. Skips the parity tests, the Dask cluster, the Prefect
runtime, and the COG-pixel reads — none of those are needed to
record a STAC cassette.

This bypasses the cross-process patching problem in
``tests/parity/test_ingest_s1_roi_parity.py``: VCR captures HTTP
inside this process; we only need STAC search responses, not the
full parity test path.

Usage::

    # Records all three cassettes (or just the missing ones):
    EARTHDATA_TOKEN=<...> ./scripts/build/record_stac_cassettes.py

    # Force re-record (delete first):
    rm tests/fixtures/stac_cassettes/*.yaml
    ./scripts/build/record_stac_cassettes.py

After recording, inspect each YAML, then commit. The parity tests
will replay these cassettes when run against the unchanged URLs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import vcr  # vcrpy

# AOI matches examples/quickstart/roi.geojson — Denver, CO.
# Denver has both ascending and descending OPERA RTC-S1 coverage in July 2024.
BBOX = (-105.020, 39.740, -105.010, 39.750)
DATETIME = "2024-07-01/2024-07-31"
START_DATE = "2024-07-01"
END_DATE = "2024-07-31"

REPO_ROOT = Path(__file__).resolve().parents[1]
CASSETTE_DIR = REPO_ROOT / "tests" / "fixtures" / "stac_cassettes"

VCR_CONFIG = {
    "filter_headers": ["authorization", "cookie", "set-cookie", "x-amz-security-token"],
    "filter_query_parameters": ["X-Amz-Signature", "X-Amz-Security-Token", "X-Amz-Credential"],
    "decode_compressed_response": True,
    "ignore_localhost": True,
}


def _record_s2_search() -> Path:
    """Record an S2 L2A search against Earth Search."""
    from pystac_client import Client

    cassette_path = CASSETTE_DIR / "test_s2_roi_parity.yaml"
    print(f"\n[1/3] Recording S2 search → {cassette_path.name}")

    with vcr.VCR(record_mode="once", **VCR_CONFIG).use_cassette(str(cassette_path)):
        client = Client.open("https://earth-search.aws.element84.com/v1")
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=BBOX,
            datetime=DATETIME,
            max_items=200,
        )
        items = list(search.items())
        print(f"  Captured {len(items)} S2 items")
    return cassette_path


def _record_s2_integration_smoke() -> Path:
    """Record the integration smoke test cassette."""
    from pystac_client import Client

    cassette_path = CASSETTE_DIR / "test_s2_stac_search_against_denver.yaml"
    print(f"\n[2/3] Recording S2 integration smoke → {cassette_path.name}")

    with vcr.VCR(record_mode="once", **VCR_CONFIG).use_cassette(str(cassette_path)):
        es = Client.open("https://earth-search.aws.element84.com/v1")
        search = es.search(
            collections=["sentinel-2-l2a"],
            bbox=BBOX,
            datetime=DATETIME,
            max_items=200,
        )
        items = list(search.items())
        print(f"  Captured {len(items)} items (integration test)")
    return cassette_path


def _record_s1_search_and_orbit() -> Path:
    """Record both the OPERA STAC search AND the CMR Granule orbit query.

    The S1 parity test calls both endpoints — VCR intercepts both
    inside this single ``use_cassette`` block so they land in one
    cassette file, exactly as the parity test expects.
    """
    from pystac_client import Client

    from tessera_embeddings.ingest.opera_query import _query_cmr_granules

    cassette_path = CASSETTE_DIR / "test_s1_roi_parity.yaml"
    print(f"\n[3/3] Recording OPERA search + orbit filter → {cassette_path.name}")

    with vcr.VCR(record_mode="once", **VCR_CONFIG).use_cassette(str(cassette_path)):
        # Part A: STAC search for OPERA RTC items
        client = Client.open("https://cmr.earthdata.nasa.gov/stac/ASF")
        search = client.search(
            collections=["OPERA_L2_RTC-S1_V1_1"],
            bbox=BBOX,
            datetime=DATETIME,
            max_items=200,
        )
        opera_items = list(search.items())
        print(f"  Captured {len(opera_items)} OPERA items")

        # Part B: CMR Granule Search API for ascending orbit filter
        ascending_items = _query_cmr_granules(BBOX, START_DATE, END_DATE, "ascending")
        print(f"  Captured {len(ascending_items)} ascending OPERA items from CMR")
    return cassette_path


def main() -> int:
    """Record any missing cassettes; print inspection checklist."""
    if not os.environ.get("EARTHDATA_TOKEN") and not (
        os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD")
    ):
        print(
            "WARNING: No EDL credentials in env. STAC searches don't strictly "
            "need them (Earth Search and CMR-STAC are public), but if any "
            "endpoint requires auth the recording will partial-record.",
            file=sys.stderr,
        )

    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)

    # Each call short-circuits if the cassette already exists (record_mode=once).
    paths = [
        _record_s2_search(),
        _record_s2_integration_smoke(),
        _record_s1_search_and_orbit(),
    ]

    print()
    print("=" * 70)
    print("Recording complete.")
    print()
    print("Cassette files:")
    for p in paths:
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name:<60s}  {size_kb:>6.1f} KB")
        else:
            print(f"  {p.name:<60s}  MISSING")
    print()
    print("BEFORE COMMITTING:")
    print()
    print("  1. Run the safety guard:")
    print("       uv run pytest tests/architecture/test_cassette_safety.py -v")
    print()
    print("  2. Manual scan for credential markers:")
    print("       grep -in -E '(authorization|cookie|token|signature|bearer)' \\")
    print(f"           {CASSETTE_DIR}/*.yaml")
    print("     Should return zero matches.")
    print()
    print("  3. Verify file sizes are 50-500 KB each.")
    print()
    print("  4. Commit:")
    print(f"       git add {CASSETTE_DIR}/*.yaml")
    print("       git commit -m 'cassettes: record STAC fixtures'")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
