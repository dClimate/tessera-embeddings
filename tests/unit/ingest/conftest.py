"""Fixtures shared by the ingest subject's tests."""

import numpy as np
import pytest
import zarr

from tessera_embeddings.ingest import live_windows, opera_query, provenance, s1_roi


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
    # The burst census: nothing recorded on the store, and the catalogue's counts answered at once.
    monkeypatch.setattr(s1_roi, "get_root_attr", lambda _store, _key, **_kw: None)
    monkeypatch.setattr(opera_query, "burst_acquisitions", lambda *_a, **_k: {"T001-000001-IW1": 2})
    # The store is faked, so the record of days given up whole is too.
    monkeypatch.setattr(provenance, "record_skipped_days", lambda *_a, **_k: None)
    monkeypatch.setattr(s1_roi.IngestManifest, "from_roi_store", classmethod(lambda _c, _p, **_k: object()))


@pytest.fixture
def mask_store(tmp_path):
    """Write a bool mask as a Zarr ROI store in 4-pixel chunks and return its path."""

    def write(mask: np.ndarray) -> str:
        path = str(tmp_path / "roi.zarr")
        z = zarr.open(path, mode="w", shape=mask.shape, chunks=(4, 4), dtype="bool")
        z[:] = mask
        return path

    return write
