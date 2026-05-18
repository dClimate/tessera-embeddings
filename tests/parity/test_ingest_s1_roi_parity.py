"""Parity skeleton: S1 OPERA RTC SAR ingestion.

Same shape as the S2 parity test (see
``test_ingest_s2_roi_parity.py``), with two extra concerns:

* OPERA orbit filtering hits the CMR Granule Search API in addition
  to STAC. **Two cassettes** are required: one for the STAC search
  and one for the granule-orbit query.
* EDL credentials are required for S3 direct access. Tests pass
  ``use_s3_direct=False`` so the cassette captures CloudFront URLs
  instead of the in-region S3 path.

Status: **xfail until cassettes land**.
"""

from __future__ import annotations

import pytest


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.xfail(reason="Awaiting OPERA STAC + CMR cassettes (Phase 10.4 follow-up)", strict=True)
def test_s1_roi_parity() -> None:
    """Domain function and Prefect flow produce identical S1 SAR Zarrs.

    Implementation notes for whoever lands this:

    * Cassette files:
        - ``tests/fixtures/stac_cassettes/opera_rtc_story_county_jul2024.yaml``
        - ``tests/fixtures/stac_cassettes/cmr_granule_orbit_story_county_jul2024.yaml``
    * Pass ``use_s3_direct=False`` to both call sites.
    * Pass ``edl_credentials_fn=None`` and ``apply_credentials_fn=None``
      — the cassette has the credentials baked into the recorded
      requests already.
    * Output store name is ``sar_ascending.zarr``.
    """
    raise NotImplementedError("Implement once OPERA cassettes are recorded.")
