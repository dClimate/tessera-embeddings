"""Parity: S2 reflectance ingestion (Prefect flow vs domain function).

Both call sites share a session-scoped LocalCluster; STAC HTTP
responses are intercepted by ``pytest-recording`` against the
cassette ``tests/fixtures/stac_cassettes/s2_l2a_story_county_jul2024.yaml``.

The actual COG pixel reads (``odc.stac.load`` → rasterio → GDAL curl)
are NOT mocked — VCR sits at the Python HTTP layer; GDAL bypasses
it. Earth Search COGs are public and unauthenticated, so the reads
work without credentials. Tests fail if Earth Search is unreachable;
that's tolerated because S2 ingest fundamentally depends on tile
pixel access.

The cassette only captures the STAC search response (item discovery);
the ~50 KB JSON is what makes this fast and deterministic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from dask.distributed import Client

from tessera_embeddings.ingest.roi import rasterize_roi_zarr
from tessera_embeddings.ingest.s2_roi import ingest_s2_roi_reflectance
from tessera_embeddings.orchestration.prefect.flows import ingest_s2_roi_reflectance as flow_module
from tests.parity.helpers import assert_zarr_equivalent

CASSETTE_NAME = "s2_l2a_story_county_jul2024"

# Story County, IA. The fixture covers July 2024 — non-trivial S2
# cloud coverage that exercises the painter's-algorithm sort.
STORY_COUNTY_DATES = ("2024-07-01", "2024-07-31")
FORCE_CRS = "EPSG:32615"  # UTM zone 15N covers Story County


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
def test_s2_roi_parity(
    tmp_path: Path,
    fixture_quickstart_roi: Path,
    parity_cluster: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Domain function and Prefect flow produce identical S2 reflectance Zarrs."""
    monkeypatch.setattr(flow_module, "get_run_logger", lambda: logging.getLogger("parity-s2"))

    roi_zarr = _stage_quickstart_roi(tmp_path, fixture_quickstart_roi)

    domain_store = tmp_path / "domain"
    flow_store = tmp_path / "flow"

    # Domain path. Capture the result so a "skipped" status (zero
    # dates passing the SCL coverage check, etc.) surfaces in the
    # test output instead of silently producing no Zarr store.
    domain_log = logging.getLogger("parity-s2-domain")
    domain_log.setLevel(logging.INFO)
    domain_log.addHandler(logging.StreamHandler())
    domain_result = ingest_s2_roi_reflectance(
        roi_zarr_path=str(roi_zarr),
        start_date=STORY_COUNTY_DATES[0],
        end_date=STORY_COUNTY_DATES[1],
        store_path=str(domain_store),
        client=parity_cluster,
        log=domain_log,
    )
    assert domain_result.status == "success", (
        f"Domain ingest produced no output (result={domain_result}). "
        f"This usually means every date got rejected by the SCL coverage "
        f"check, or the domain call hit an exception that was swallowed."
    )

    # Flow path. We bypass the @flow wrapper via .fn so tests don't
    # need a running Prefect server; the wrapper around our domain
    # function is one concrete provider-selection branch (use_local)
    # plus a dispatch to .with_options(task_runner=...). Both paths
    # eventually land in ingest_s2_roi_reflectance, so .fn-bypass is
    # faithful to the production code path.
    flow_module.ingest_s2_roi_reflectance.fn(  # type: ignore[attr-defined]
        roi_zarr_path=str(roi_zarr),
        start_date=STORY_COUNTY_DATES[0],
        end_date=STORY_COUNTY_DATES[1],
        store_path=str(flow_store),
        use_local=True,
    )

    # Both stores must materialise
    domain_zarr = domain_store / "reflectance.zarr"
    flow_zarr = flow_store / "reflectance.zarr"
    assert domain_zarr.exists(), f"Domain output missing: {domain_zarr}"
    assert flow_zarr.exists(), f"Flow output missing: {flow_zarr}"

    # Compare with float32-ULP tolerance — odc.stac.load can produce
    # bit-different floats across runs due to threading-order effects
    # in baseline corrections. Strict equality is too tight for a
    # parity check; correctness is what we're after.
    assert_zarr_equivalent(domain_zarr, flow_zarr, rtol=1e-6, atol=1e-6)
