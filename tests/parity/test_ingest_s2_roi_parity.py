"""Parity skeleton: S2 reflectance ingestion.

This test will run the Prefect S2 flow (with ``use_local=True``) and
the domain function :func:`ingest_s2_roi_reflectance` against a
LocalCluster, against the same VCR-cassette-backed STAC responses,
and assert the two output Zarr stores match via
:func:`assert_zarr_equivalent`.

Status: **xfail until cassettes land** (see
``tests/fixtures/stac_cassettes/README.md``). The skeleton documents
the expected shape; remove the xfail once the cassette is recorded.
"""

from __future__ import annotations

import pytest


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.xfail(reason="Awaiting S2 STAC cassette recording (Phase 10.4 follow-up)", strict=True)
def test_s2_roi_parity() -> None:
    """Domain function and Prefect flow produce identical S2 reflectance Zarrs.

    See ``test_generate_roi_parity.py`` for the canonical shape; the
    only differences for S2 are:

    * Both calls receive a ``client`` from the shared LocalCluster
      fixture (or the flow gets ``use_local=True`` which sets up its
      own).
    * STAC HTTP responses are intercepted by ``pytest-recording``;
      the cassette path is
      ``tests/fixtures/stac_cassettes/s2_l2a_story_county_jul2024.yaml``.
    * The output store name is ``reflectance.zarr`` so
      :func:`assert_zarr_equivalent` walks the group's data variables.
    """
    raise NotImplementedError("Implement once the S2 STAC cassette is recorded.")
