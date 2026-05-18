"""Parity-test shared fixtures.

* :func:`fixture_quickstart_roi` — path to the bundled
  ``examples/quickstart/roi.geojson``. Tests rasterise it on demand
  rather than checking in a pre-rasterised store (the AOI is small,
  rasterisation is fast, and avoids a binary fixture in git).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def fixture_quickstart_roi() -> Path:
    """The bundled Story-County-IA quickstart GeoJSON."""
    path = REPO_ROOT / "examples" / "quickstart" / "roi.geojson"
    assert path.exists(), f"Missing quickstart fixture at {path}"
    return path
