"""Unit tests for ingest/roi_processing.py — ROI mask application."""

from unittest.mock import patch

import dask.array as da
import numpy as np
import xarray as xr

from tessera_embeddings.ingest.roi_processing import apply_roi_mask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(data_vars: dict[str, np.ndarray], times: list[str] | None = None) -> xr.Dataset:
    """Create a lazy xarray Dataset with (time, northing, easting) dimensions.

    Args:
        data_vars: Mapping of variable name to 3D numpy array (time, northing, easting).
        times: ISO date strings for the time coordinate.
    """
    first = next(iter(data_vars.values()))
    nt, ny, nx = first.shape
    if times is None:
        times = [f"2024-06-{i + 1:02d}" for i in range(nt)]
    coords = {
        "time": [np.datetime64(t) for t in times],
        "northing": np.arange(ny),
        "easting": np.arange(nx),
    }
    ds_vars = {}
    for name, arr in data_vars.items():
        ds_vars[name] = (["time", "northing", "easting"], da.from_array(arr, chunks=(1, ny, nx)))
    return xr.Dataset(ds_vars, coords=coords)


# ---------------------------------------------------------------------------
# apply_roi_mask
# ---------------------------------------------------------------------------


def test_applies_mask_to_all_variables():
    """Pixels outside ROI should be zeroed for every variable."""
    band = np.array([[[10, 20], [30, 40]]], dtype=np.int16)
    ds = _make_dataset({"red": band.copy(), "green": band.copy()})
    mask_2d = da.from_array(np.array([[True, False], [False, True]]), chunks=(2, 2))

    result = apply_roi_mask(ds, mask_2d)

    for name in ("red", "green"):
        np.testing.assert_array_equal(result[name].compute().values[0], [[10, 0], [0, 40]])


def test_masking_is_lazy():
    """Nothing may compute here: the radar leg calls this once per date.

    Counts computes rather than inspecting the return value, so eagerness fails
    wherever it is reintroduced.
    """
    ds = _make_dataset({"band": np.ones((1, 4, 4), dtype=np.int16)})
    mask_2d = da.from_array(np.ones((4, 4), dtype=bool), chunks=(4, 4))

    computes = []
    with patch.object(da.Array, "compute", lambda self, **kw: computes.append(1)):
        out = apply_roi_mask(ds, mask_2d)

    assert computes == [], f"apply_roi_mask computed eagerly {len(computes)} time(s)"
    assert isinstance(out, xr.Dataset)
