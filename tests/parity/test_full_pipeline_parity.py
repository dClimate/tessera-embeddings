"""Parity skeleton: full pipeline (ROI → ingest → inference → assembly).

Slowest parity test in the suite. Marked ``slow`` in addition to
``parity`` so it's only run on the nightly CI workflow.

Status: **xfail until upstream parity tests pass** (this depends on
S2, S1, and inference parity all working).
"""

from __future__ import annotations

import pytest


@pytest.mark.parity
@pytest.mark.slow
@pytest.mark.xfail(reason="Awaiting upstream S2/S1/inference parity (Phase 10 follow-up)", strict=True)
def test_full_pipeline_parity() -> None:
    """Plain runner and master Prefect flow produce identical embeddings.

    * Plain runner: ``run_plain(quickstart_config)`` end-to-end.
    * Prefect flow: ``tessera_embeddings`` flow against the same
      inputs, with ``use_local=True`` propagated to the inner Dask
      and Ray providers.
    * Compare the final ``embeddings/<roi>.zarr`` stores via
      :func:`assert_zarr_equivalent` with a small ``atol`` tolerance
      for floating-point determinism.
    """
    raise NotImplementedError("Implement once S2 + S1 parity pass.")
