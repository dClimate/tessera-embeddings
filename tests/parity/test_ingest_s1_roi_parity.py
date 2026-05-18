"""Parity: S1 OPERA RTC SAR ingestion (Prefect flow vs domain function).

Same shape as ``test_ingest_s2_roi_parity.py``, with two extra
concerns specific to OPERA:

* OPERA orbit filtering hits the CMR Granule Search API in addition
  to STAC. ``pytest-recording`` captures both endpoints into the
  single cassette ``opera_rtc_story_county_jul2024.yaml``.
* EDL credentials are required for OPERA S3 direct access. We pass
  ``use_s3_direct=False`` so the test routes through CloudFront-signed
  HTTPS URLs (no STS). EDL credentials still appear in the
  Authorization header on the redirect chain — pytest-recording's
  ``filter_headers`` config (in conftest.py) strips them, and the
  ``test_cassette_safety.py`` guard backs that up.

The COG pixel reads (rasterio → GDAL curl) are not VCR-mockable; we
need EDL credentials in the test environment for them to succeed.
``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` env vars are
required.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from dask.distributed import Client

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


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.vcr(CASSETTE_NAME)
@pytest.mark.skipif(
    not os.environ.get("EARTHDATA_USERNAME") or not os.environ.get("EARTHDATA_PASSWORD"),
    reason="EARTHDATA_USERNAME and EARTHDATA_PASSWORD required for COG reads",
)
def test_s1_roi_parity(
    tmp_path: Path,
    fixture_quickstart_roi: Path,
    parity_cluster: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Domain function and Prefect flow produce identical S1 SAR Zarrs."""
    monkeypatch.setattr(flow_module, "get_run_logger", lambda: logging.getLogger("parity-s1"))

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
