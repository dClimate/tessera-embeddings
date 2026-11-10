#!/usr/bin/env -S uv run python
"""Standalone cassette recorder.

Hits ONLY the Earth Search STAC endpoint and writes the request/response pairs to
``tests/fixtures/stac_cassettes/test_s2_roi_parity.yaml``, the cassette the S2 parity
test replays. Skips the parity test itself, the Dask cluster, the Prefect runtime, and
the COG-pixel reads — none of those are needed to record a STAC cassette. The
integration tests' per-test cassettes are recorded by running them with
``--record-mode=once`` (see ``tests/fixtures/stac_cassettes/README.md``).

Usage::

    ./scripts/build/record_stac_cassettes.py

    # Force re-record (delete first):
    rm tests/fixtures/stac_cassettes/test_s2_roi_parity.yaml
    ./scripts/build/record_stac_cassettes.py

After recording, inspect the YAML, then commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import vcr  # vcrpy

# AOI matches examples/quickstart/roi.geojson — Denver, CO.
BBOX = (-105.020, 39.740, -105.010, 39.750)
DATETIME = "2024-07-01/2024-07-31"


def _repo_root() -> Path:
    """The directory holding ``pyproject.toml``, found by walking up from this file.

    Anchored on a marker rather than a parent count, which is what ``tests/_paths.py``
    does and for the same reason: a count encodes this file's depth, so moving the script
    one directory down silently retargets everything derived from it. That happened —
    ``parents[1]`` pointed at ``scripts/`` once this moved into ``scripts/build/``, and the
    recorder would have written cassettes to ``scripts/tests/fixtures/`` and reported
    success while the tests kept reading the real ones.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"no pyproject.toml above {__file__}; cannot locate the repo root")


REPO_ROOT = _repo_root()
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
    print(f"\nRecording S2 search → {cassette_path.name}")

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


def main() -> int:
    """Record any missing cassettes; print inspection checklist."""
    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)

    # Each call short-circuits if the cassette already exists (record_mode=once).
    paths = [_record_s2_search()]

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
