"""The destination's types are checked BEFORE a fill spends a fleet, not after.

Assembly already refuses a staged tile whose dtype the destination cannot hold, and that guard is
what caught the defect this preflight exists for. But it fires at the END: on 2026-08-18 two fills
each ran their whole inference — 13 and 14 minutes across 20 GPU actors — and then died because
``s2_month_covered`` had been seeded ``bool`` while the staging writer emits ``int8``. On a campaign
that is one wasted cell per attempt, and a schema change mid-campaign makes it every cell.

The same comparison costs one metadata read at fill start. What it must NOT do is refuse a store that
is merely older: a variable joining an existing store takes that store's geometry by design, so
geometry is deliberately not compared.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.store_layout import CARRIED_VARS, GLOBAL, MONTH_COVERED_VAR, REQUIRED_VARS
from tessera_embeddings.storage.global_store import check_destination_types


def _group(**arrays: tuple[str, dict]) -> zarr.Group:
    """A bare group holding the named arrays as (dtype, attrs)."""
    g = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    for name, (dtype, attrs) in arrays.items():
        a = g.create_array(name, shape=(1, 8, 8), chunks=(1, 8, 8), dtype=dtype, fill_value=0)
        a.attrs.update(attrs)
    return g


def _as_layout_declares(var: str) -> tuple[str, dict]:
    want = GLOBAL.for_var(var)
    return want.dtype, dict(want.attrs)


def test_a_correctly_seeded_group_passes() -> None:
    """Every variable the layout declares, at the dtype and attrs it declares."""
    g = _group(**{v: _as_layout_declares(v) for v in (*REQUIRED_VARS, *CARRIED_VARS) if v in GLOBAL.arrays})
    check_destination_types(g, GLOBAL)  # must not raise


def test_the_bool_month_array_that_cost_two_fills_is_refused() -> None:
    """The exact 2026-08-18 case: seeded bool, staged int8, discovered at assembly.

    `Dataset.to_zarr` stores a bool array as int8 with attrs dtype="bool" and ignores an encoding
    dtype asking for bool, so a bool destination can never match what the writer produces.
    """
    g = _group(**{MONTH_COVERED_VAR: ("bool", {})})
    with pytest.raises(ValueError, match="Refusing to fill") as exc:
        check_destination_types(g, GLOBAL, where="09S year 2022")
    assert "s2_month_covered: dtype bool on disk, layout declares int8" in str(exc.value)
    assert "09S year 2022" in str(exc.value)
    assert "write-once" in str(exc.value), "the remedy is a fresh store, and the message must say so"


def test_a_missing_bool_attr_is_refused_because_it_is_part_of_the_type() -> None:
    """int8 with no `dtype="bool"` attr reads back as 0/1 integers, not booleans — a different
    published type, and one no reader of the docs would expect."""
    g = _group(**{MONTH_COVERED_VAR: ("int8", {})})
    with pytest.raises(ValueError, match="attr dtype=None on disk, layout declares 'bool'"):
        check_destination_types(g, GLOBAL)


def test_a_store_that_predates_an_array_is_not_wrong() -> None:
    """Absence is not a mismatch: assembly writes only what BOTH sides have, so a store seeded
    before an array existed must fill normally rather than be refused."""
    g = _group(**{v: _as_layout_declares(v) for v in REQUIRED_VARS})
    check_destination_types(g, GLOBAL)  # must not raise


def test_geometry_is_deliberately_not_compared() -> None:
    """A variable joining an EXISTING store takes that store's chunking rather than the preset's,
    by design. Comparing geometry would refuse every legitimately older store."""
    g = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    want = GLOBAL.for_var("s2_obs_count")
    a = g.create_array("s2_obs_count", shape=(1, 64, 64), chunks=(1, 16, 16), dtype=want.dtype, fill_value=0)
    a.attrs.update(dict(want.attrs))
    check_destination_types(g, GLOBAL)  # must not raise on the odd chunking


def test_every_mismatch_is_named_at_once() -> None:
    """One report, not one per run: a reader fixing a seeded store needs the whole list."""
    g = _group(**{MONTH_COVERED_VAR: ("bool", {}), "s2_obs_count": ("int32", {})})
    with pytest.raises(ValueError) as exc:
        check_destination_types(g, GLOBAL)
    assert "s2_month_covered" in str(exc.value) and "s2_obs_count" in str(exc.value)


def test_an_array_the_layout_does_not_declare_is_not_this_gates_business() -> None:
    """`embedding_std` is single-ROI only. A global store carrying an extra variable is a schema
    question, not a reason to refuse a fill."""
    g = _group(**{v: _as_layout_declares(v) for v in REQUIRED_VARS})
    extra = g.create_array("embedding_std", shape=(1, 8, 8), chunks=(1, 8, 8), dtype="float32", fill_value=np.nan)
    assert extra is not None
    check_destination_types(g, GLOBAL)  # must not raise
