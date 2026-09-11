"""Helpers for parity tests.

The canonical helper is :func:`assert_zarr_equivalent`, which
compares two stores while ignoring metadata that legitimately
differs between runs (timestamps, run IDs).

Most stores in this package are **icechunk repos**, not plain zarr
groups — ``write_dataset`` writes via icechunk for transactional
commits. ``zarr.open(path)`` cannot read these directly. The helper
detects icechunk repos (via :func:`open_store`) and falls back to
:func:`zarr.open` for the only non-icechunk case (the ROI mask,
which is a single zarr array written by ``rasterize_roi_zarr``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE
from tessera_embeddings.ingest.roi import rasterize_roi_zarr
from tessera_embeddings.storage.zarr_store import open_store

# Attribute keys that are *expected* to differ between runs and are
# excluded from the strict-equality check.
_VOLATILE_ATTRS = frozenset(
    {
        "created_at",
        "last_appended",
        "_run_id",
        "run_id",
        "run_started_at",
    }
)


def _looks_like_icechunk_repo(path: Path) -> bool:
    """Heuristic: an icechunk repo has a ``config.yaml`` at its root.

    A plain zarr store has ``zarr.json`` (v3) or ``.zarray``/``.zgroup`` (v2).
    The two formats are mutually exclusive on disk.
    """
    return (path / "config.yaml").exists() or any(path.glob("snapshots/*"))


def assert_zarr_equivalent(
    actual: Path | str,
    expected: Path | str,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    extra_volatile_attrs: Iterable[str] = (),
) -> None:
    """Assert two stores hold equivalent data + critical metadata.

    Detects whether each path is an icechunk repo or a plain zarr
    store and reads accordingly:

    * **icechunk** — opened via :func:`open_store`, which returns an
      :class:`xarray.Dataset`. Comparison walks data variables.
    * **plain zarr array/group** — opened via :func:`zarr.open`.
      Used for the ROI mask (a single zarr array on disk).

    Args:
        actual: Path to the candidate store.
        expected: Path to the reference store.
        rtol: Relative tolerance for ``np.testing.assert_allclose``.
        atol: Absolute tolerance. Both default to 0 (byte-equal).
        extra_volatile_attrs: Additional attribute keys to ignore on
            top of the default volatile set (timestamps, run IDs).
    """
    actual_path = Path(str(actual))
    expected_path = Path(str(expected))

    a_is_icechunk = _looks_like_icechunk_repo(actual_path)
    b_is_icechunk = _looks_like_icechunk_repo(expected_path)
    assert a_is_icechunk == b_is_icechunk, (
        f"Mixed store types: actual is "
        f"{'icechunk' if a_is_icechunk else 'plain zarr'}, "
        f"expected is {'icechunk' if b_is_icechunk else 'plain zarr'}"
    )

    if a_is_icechunk:
        _compare_icechunk_datasets(
            open_store(str(actual_path)),
            open_store(str(expected_path)),
            rtol=rtol,
            atol=atol,
            extra_volatile_attrs=extra_volatile_attrs,
        )
    else:
        _compare_zarr_path(actual_path, expected_path, rtol=rtol, atol=atol, extra_volatile_attrs=extra_volatile_attrs)


def _compare_icechunk_datasets(
    a: xr.Dataset,
    b: xr.Dataset,
    *,
    rtol: float,
    atol: float,
    extra_volatile_attrs: Iterable[str],
) -> None:
    """Compare two xarray Datasets read from icechunk repos."""
    try:
        a_vars = sorted(a.data_vars)
        b_vars = sorted(b.data_vars)
        assert a_vars == b_vars, f"Variable lists differ: {a_vars} vs {b_vars}"
        for name in a_vars:
            arr_a = a[name].values
            arr_b = b[name].values
            _compare_arrays(arr_a, arr_b, name=str(name), rtol=rtol, atol=atol)

        volatile = _VOLATILE_ATTRS | frozenset(extra_volatile_attrs)
        a_attrs = {k: v for k, v in a.attrs.items() if k not in volatile}
        b_attrs = {k: v for k, v in b.attrs.items() if k not in volatile}
        assert a_attrs == b_attrs, f"Attrs differ:\n  actual:   {a_attrs}\n  expected: {b_attrs}"
    finally:
        a.close()
        b.close()


def _compare_zarr_path(
    actual: Path,
    expected: Path,
    *,
    rtol: float,
    atol: float,
    extra_volatile_attrs: Iterable[str],
) -> None:
    """Compare two plain zarr stores (single-array ROI masks)."""
    a = zarr.open(str(actual), mode="r")
    b = zarr.open(str(expected), mode="r")

    if isinstance(a, zarr.Array) or isinstance(b, zarr.Array):
        assert isinstance(a, zarr.Array) and isinstance(b, zarr.Array), (
            "One side is a Group, the other is an Array; cannot compare."
        )
        _compare_arrays(np.asarray(a[:]), np.asarray(b[:]), name="<root>", rtol=rtol, atol=atol)
    else:
        a_vars = sorted(_iter_array_names(a))
        b_vars = sorted(_iter_array_names(b))
        assert a_vars == b_vars, f"Variable lists differ: {a_vars} vs {b_vars}"
        for name in a_vars:
            arr_a = np.asarray(a[name][:])  # type: ignore[index]
            arr_b = np.asarray(b[name][:])  # type: ignore[index]
            _compare_arrays(arr_a, arr_b, name=name, rtol=rtol, atol=atol)

    volatile = _VOLATILE_ATTRS | frozenset(extra_volatile_attrs)
    a_attrs = {k: v for k, v in dict(a.attrs).items() if k not in volatile}
    b_attrs = {k: v for k, v in dict(b.attrs).items() if k not in volatile}
    assert a_attrs == b_attrs, f"Attrs differ:\n  actual:   {a_attrs}\n  expected: {b_attrs}"


def _compare_arrays(arr_a: np.ndarray, arr_b: np.ndarray, *, name: str, rtol: float, atol: float) -> None:
    assert arr_a.shape == arr_b.shape, f"{name}: shape {arr_a.shape} vs {arr_b.shape}"
    assert arr_a.dtype == arr_b.dtype, f"{name}: dtype {arr_a.dtype} vs {arr_b.dtype}"
    if rtol == 0.0 and atol == 0.0:
        np.testing.assert_array_equal(arr_a, arr_b, err_msg=f"data differs for {name}")
    else:
        np.testing.assert_allclose(arr_a, arr_b, rtol=rtol, atol=atol, err_msg=f"data differs for {name}")


def _iter_array_names(group: zarr.Group) -> list[str]:
    """List array names in a Zarr group (one level deep)."""
    return [name for name, _ in group.arrays()]


#: UTM zone 13N, which covers the Denver quickstart AOI. Both ROI-parity tests rasterise
#: against it, and a mismatch between them would compare two different projections.
FORCE_CRS = "EPSG:32613"


def stage_quickstart_roi(tmp_path: Path, roi_geojson: Path) -> Path:
    """Rasterise the quickstart GeoJSON to a Zarr ROI under ``tmp_path``.

    Shared because the S1 and S2 ROI-parity tests staged it with byte-identical code. Both
    compare a Prefect flow against the plain runner over the SAME staged ROI, so the staging
    is setup common to them rather than anything either one is testing.
    """
    roi_zarr = tmp_path / "roi.zarr"
    rasterize_roi_zarr(
        output_path=str(roi_zarr),
        resolution=10.0,
        chunk_size=INGEST_CHUNK_SIZE,
        force_crs=FORCE_CRS,
        input_path=str(roi_geojson),
    )
    return roi_zarr
