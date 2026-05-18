"""Adapter parity test template.

Copy this file to ``tests/parity/<your_adapter>/test_<your_adapter>_parity.py``
and replace the ``run_via_your_adapter`` placeholder with the adapter's
actual entry point.

The contract is identical for every adapter: run the adapter against
the same inputs as the plain runner / domain function, then assert
the output Zarr stores are byte-equivalent (modulo timestamps and run
IDs — see :func:`assert_zarr_equivalent`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# from tessera_embeddings.orchestration.<your_adapter>.flows import generate_roi as adapter_flow
from tessera_embeddings.ingest.roi import rasterize_roi_zarr
from tests.parity.helpers import assert_zarr_equivalent


@pytest.mark.parity
@pytest.mark.skip(reason="Template — copy and customise for a real adapter")
def test_your_adapter_parity(tmp_path: Path, fixture_quickstart_roi: Path) -> None:
    """Replace the body with your adapter's flow invocation."""
    out_a = tmp_path / "domain.zarr"
    out_b = tmp_path / "adapter.zarr"

    # Reference path: the bundled domain function or plain runner.
    rasterize_roi_zarr(
        output_path=str(out_a),
        resolution=10.0,
        chunk_size=2000,
        force_crs="EPSG:32615",
        input_path=str(fixture_quickstart_roi),
    )

    # Adapter path: invoke the adapter's flow / DAG / pipeline.
    # run_via_your_adapter(
    #     input_path=str(fixture_quickstart_roi),
    #     output_path=str(out_b),
    #     resolution=10.0,
    #     chunk_size=2000,
    #     force_crs="EPSG:32615",
    # )

    assert_zarr_equivalent(out_a, out_b)
