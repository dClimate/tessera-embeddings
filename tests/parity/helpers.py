"""Helpers for parity tests.

The canonical helper is :func:`assert_zarr_equivalent`, which
compares two Zarr stores while ignoring metadata that legitimately
differs between runs (timestamps, run IDs).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import zarr

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


def assert_zarr_equivalent(
    actual: Path | str,
    expected: Path | str,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    extra_volatile_attrs: Iterable[str] = (),
) -> None:
    """Assert two Zarr stores hold equivalent data + critical metadata.

    Args:
        actual: Path to the candidate store.
        expected: Path to the reference store.
        rtol: Relative tolerance for ``np.testing.assert_allclose``.
            Use 0.0 for byte-identical data.
        atol: Absolute tolerance.
        extra_volatile_attrs: Additional attribute keys to ignore on
            top of the default volatile set (timestamps, run IDs).

    Raises:
        AssertionError: If the data arrays or critical metadata differ.
    """
    a = zarr.open(str(actual), mode="r")
    b = zarr.open(str(expected), mode="r")

    if isinstance(a, zarr.Array) or isinstance(b, zarr.Array):
        # Single-array store (e.g. ROI zarr). The other side must
        # also be a single-array store.
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

    # Compare root attrs after stripping volatile keys
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
