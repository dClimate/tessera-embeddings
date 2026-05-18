"""Integration test: STAC query against a recorded cassette.

This exercises the cassette infrastructure end-to-end and is the
canonical example for any future cassette-backed ingest test.

Run with::

    uv run pytest tests/integration/test_stac_query_cassette.py -m integration

To record (do this once, then commit the cassette)::

    uv run pytest tests/integration/test_stac_query_cassette.py \
        -m integration --record-mode=once
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    """VCR configuration: filter Earthdata bearer tokens out of cassettes."""
    return {
        "filter_headers": ["authorization", "x-amz-security-token", "cookie"],
        "filter_query_parameters": ["X-Amz-Signature", "X-Amz-Security-Token"],
        "decode_compressed_response": True,
    }


@pytest.mark.integration
@pytest.mark.vcr  # cassette: tests/fixtures/stac_cassettes/test_s2_stac_search_against_story_county.yaml
def test_s2_stac_search_against_story_county() -> None:
    """A pystac-client search against Earth Search returns ~12 items.

    Inputs match the bundled quickstart AOI (Story County, IA) so the
    cassette captures the realistic case. Item count is the
    invariant the test asserts; specific item IDs are too volatile.
    """
    from pystac_client import Client

    es = Client.open("https://earth-search.aws.element84.com/v1")
    search = es.search(
        collections=["sentinel-2-l2a"],
        bbox=[-93.65, 42.00, -93.55, 42.10],
        datetime="2024-07-01/2024-07-31",
        max_items=200,
    )
    items = list(search.items())
    assert 5 <= len(items) <= 30, f"unexpected item count: {len(items)}"
