"""Fixtures shared by the ingest subject's tests."""

import pytest

from tessera_embeddings.ingest import live_windows, s1_roi


@pytest.fixture
def s1_leg_offline(monkeypatch):
    """Stub every read the S1 leg makes that a domain test does not assert on.

    The ROI mask and its metadata, the run's live windows (one 1x1 window), the item provider
    and the ROI store's provenance. What a test is about (the catalogue, the writes, the
    store's existing dates) it patches itself.
    """
    monkeypatch.setattr(s1_roi, "apply_roi_mask", lambda data, *a, **k: data)
    monkeypatch.setattr(s1_roi, "read_roi_mask", lambda *a, **k: object())
    monkeypatch.setattr(
        s1_roi,
        "read_roi_metadata",
        lambda _p, **_k: type("M", (), {"geobox": object(), "native_crs": "EPSG:32635", "bbox_wgs84": (0, 0, 1, 1)})(),
    )
    monkeypatch.setattr(
        live_windows, "live_windows_for_mask", lambda *a, **k: [type("W", (), {"y0": 0, "y1": 1, "x0": 0, "x1": 1})()]
    )
    monkeypatch.setattr(s1_roi, "make_s1_item_provider", lambda *a, **k: lambda: [])
    monkeypatch.setattr(s1_roi.IngestManifest, "from_roi_store", classmethod(lambda _c, _p, **_k: object()))
