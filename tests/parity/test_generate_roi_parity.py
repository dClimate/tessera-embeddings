"""Parity: ``generate_roi`` Prefect flow ↔ direct domain function.

The simplest parity case in the suite — rasterise an ROI with no
Dask, no Ray, no STAC. Calling :func:`rasterize_roi_zarr` directly
must produce the same on-disk Zarr as invoking the Prefect flow.

We invoke the Prefect flow via ``flow.fn(**kwargs)`` to bypass the
Prefect runtime; for ``generate_roi`` this is sound because the body
only calls domain functions plus :func:`prefect.get_run_logger`,
which we replace with the stdlib logger via a small monkey-patch.

This file is the worked example for adding parity tests; harder
flows (S2/S1/full-pipeline) follow the same shape with VCR cassettes
and a LocalCluster fixture.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from tessera_embeddings.ingest.roi import rasterize_roi_zarr
from tessera_embeddings.orchestration.prefect.flows import generate_roi as flow_module
from tests.parity.helpers import assert_zarr_equivalent


@pytest.mark.parity
def test_generate_roi_parity(tmp_path: Path, fixture_quickstart_roi: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Domain function and Prefect flow produce identical ROI Zarrs.

    The flow is invoked via ``.fn`` (Prefect's bypass for testing) so
    we don't need a full Prefect runtime; ``get_run_logger`` is
    replaced with the stdlib logger so the bypass actually works.
    """
    # Replace get_run_logger so flow.fn doesn't try to read a Prefect run context
    monkeypatch.setattr(flow_module, "get_run_logger", lambda: logging.getLogger("parity"))

    bucket_a = tmp_path / "domain"
    bucket_b = tmp_path / "flow"
    # Flow appends a CRS suffix to the output filename when force_crs is set;
    # mirror that exactly on the domain-call side so the two outputs are at
    # paths the same shape.
    out_a = bucket_a / "zarrs" / "test_epsg32613.zarr"
    out_b = bucket_b / "zarrs" / "test_epsg32613.zarr"

    # The Prefect flow reads the geojson from {bucket}/geojsons/{name}.geojson,
    # so stage the fixture there before invoking it.
    geojson_dir = bucket_b / "geojsons"
    geojson_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture_quickstart_roi, geojson_dir / "test.geojson")

    # The GeoJSON ships with OGC:CRS84 (geographic, degrees). Force a
    # UTM CRS so rasterisation works in metres. Denver, CO falls
    # in UTM zone 13N.
    force_crs = "EPSG:32613"

    # Domain function path
    rasterize_roi_zarr(
        output_path=str(out_a),
        resolution=10.0,
        chunk_size=2000,
        force_crs=force_crs,
        input_path=str(fixture_quickstart_roi),
    )

    # Prefect flow path. .fn drops the @flow wrapper so we don't need
    # a runtime; the flow body still does its argument-validation,
    # path-derivation, and call into rasterize_roi_zarr.
    flow_module.generate_roi.fn(  # type: ignore[attr-defined]
        roi_bucket=str(bucket_b),
        roi_name="test",
        resolution=10.0,
        chunk_size=2000,
        force_crs=force_crs,
    )

    # Both stores must materialise
    assert out_a.exists(), f"Domain output missing: {out_a}"
    assert out_b.exists(), f"Flow output missing: {out_b}"

    assert_zarr_equivalent(out_a, out_b)
