"""Parity: S1 OPERA RTC SAR ingestion (Prefect flow vs domain function).

Same shape as ``test_ingest_s2_roi_parity.py``, with two extra
concerns specific to OPERA:

* OPERA orbit filtering hits the CMR Granule Search API in addition
  to STAC. ``pytest-recording`` captures both endpoints into the
  single cassette ``opera_rtc_story_county_jul2024.yaml``.
* EDL credentials are required for COG reads (rasterio → GDAL curl
  bypasses VCR). On us-west-2 production the package's auth.py uses
  basic auth via EARTHDATA_USERNAME / EARTHDATA_PASSWORD. Many
  contributor laptops can't basic-auth against EDL (SAML / Launchpad
  / MFA-linked accounts), so this test accepts an EARTHDATA_TOKEN
  and monkey-patches the auth session to use Bearer for the test
  duration. The patch is local — auth.py and production behaviour
  are unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import requests
from dask.distributed import Client

from tessera_embeddings.ingest import auth as auth_module
from tessera_embeddings.ingest import opera_query as opera_query_module
from tessera_embeddings.ingest.roi import rasterize_roi_zarr
from tessera_embeddings.ingest.s1_roi import ingest_s1_roi_sar
from tessera_embeddings.orchestration.prefect.flows import ingest_s1_roi_sar as flow_module
from tests.parity.helpers import assert_zarr_equivalent

CASSETTE_NAME = "opera_rtc_story_county_jul2024"

# Story County, IA. July 2024 has 6 ascending granules per the live
# probe in the AOI selection commit.
STORY_COUNTY_DATES = ("2024-07-01", "2024-07-31")
FORCE_CRS = "EPSG:32615"


def _stage_quickstart_roi(tmp_path: Path, roi_geojson: Path) -> Path:
    """Rasterise the quickstart GeoJSON to a Zarr ROI under ``tmp_path``."""
    roi_zarr = tmp_path / "roi.zarr"
    rasterize_roi_zarr(
        output_path=str(roi_zarr),
        resolution=10.0,
        chunk_size=2000,
        force_crs=FORCE_CRS,
        input_path=str(roi_geojson),
    )
    return roi_zarr


def _bearer_session_factory(token: str) -> requests.Session:
    """Build a Bearer-auth requests session for cassette-recording use.

    Sets the ``Authorization`` header at the **session** level so it's
    applied to every request the session sends — including across
    redirects. ``requests`` does not strip session-level headers the
    way it strips per-request ``prepared_request.headers`` on
    cross-domain redirects.

    Verified working against the full ASF redirect chain by
    ``scripts/probe_edl_bearer.py``: a single GET reaches
    ``cloudfront.net`` with HTTP 200 across 5 redirect hops
    (datapool → cumulus → URS → cumulus/login → cumulus/RTC → cloudfront).

    We intentionally do NOT subclass ``auth_module._EDLSession``: that
    class overrides ``rebuild_auth`` to re-inject ``session.auth`` on
    URS hops (the basic-auth path). For bearer, the base
    ``requests.Session.rebuild_auth`` is correct — it strips
    *prepared-request* auth on cross-domain hops but leaves
    *session-level* headers alone, which is exactly what we want.
    """
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return session


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.vcr(CASSETTE_NAME)
@pytest.mark.skipif(
    not (
        os.environ.get("EARTHDATA_TOKEN")
        or (os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"))
    ),
    reason="EDL credentials required: set EARTHDATA_TOKEN (preferred) or "
    "EARTHDATA_USERNAME + EARTHDATA_PASSWORD",
)
def test_s1_roi_parity(
    tmp_path: Path,
    fixture_quickstart_roi: Path,
    parity_cluster: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Domain function and Prefect flow produce identical S1 SAR Zarrs."""
    monkeypatch.setattr(flow_module, "get_run_logger", lambda: logging.getLogger("parity-s1"))

    # Bearer-token override: when EARTHDATA_TOKEN is set, swap the
    # auth.py session factory for one that uses Bearer auth for the
    # duration of the test only. auth.py itself is unchanged.
    #
    # We patch BOTH auth_module.get_edl_session AND opera_query_module.get_edl_session
    # because opera_query.py uses `from tessera_embeddings.ingest.auth import
    # get_edl_session` (a from-import that binds the symbol into opera_query's
    # namespace at import time). Patching only auth_module wouldn't reach
    # opera_query's already-bound reference. If a future module also
    # from-imports get_edl_session, add it here.
    if token := os.environ.get("EARTHDATA_TOKEN"):
        bearer_factory = lambda: _bearer_session_factory(token)  # noqa: E731
        monkeypatch.setattr(auth_module, "get_edl_session", bearer_factory)
        monkeypatch.setattr(opera_query_module, "get_edl_session", bearer_factory)

    roi_zarr = _stage_quickstart_roi(tmp_path, fixture_quickstart_roi)

    domain_store = tmp_path / "domain"
    flow_store = tmp_path / "flow"

    # Domain path. use_s3_direct=False so the test routes through
    # CloudFront HTTPS (which the cassette captures); no STS. The
    # credential callbacks stay None — the cassette-replayed CMR /
    # STAC responses don't need fresh creds, and the COG pixel reads
    # use GDAL's environment-based EDL handling outside VCR.
    ingest_s1_roi_sar(
        roi_zarr_path=str(roi_zarr),
        start_date=STORY_COUNTY_DATES[0],
        end_date=STORY_COUNTY_DATES[1],
        store_path=str(domain_store),
        client=parity_cluster,
        orbit="ascending",
        use_s3_direct=False,
        edl_credentials_fn=None,
        apply_credentials_fn=None,
        log=logging.getLogger("parity-s1-domain"),
    )

    # Flow path via .fn bypass — same reasoning as the S2 parity test.
    flow_module.ingest_s1_roi_sar.fn(  # type: ignore[attr-defined]
        roi_zarr_path=str(roi_zarr),
        start_date=STORY_COUNTY_DATES[0],
        end_date=STORY_COUNTY_DATES[1],
        store_path=str(flow_store),
        orbit="ascending",
        use_s3_direct=False,
        use_local=True,
    )

    domain_zarr = domain_store / "sar_ascending.zarr"
    flow_zarr = flow_store / "sar_ascending.zarr"
    assert domain_zarr.exists(), f"Domain output missing: {domain_zarr}"
    assert flow_zarr.exists(), f"Flow output missing: {flow_zarr}"

    # SAR amplitude → dB conversion has the same float-ULP variance
    # discussion as S2; tolerance is appropriate.
    assert_zarr_equivalent(domain_zarr, flow_zarr, rtol=1e-6, atol=1e-6)
