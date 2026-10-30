"""Synthetic in-memory mosaic stores for the inference read path.

`load_chunk` and everything above it read three sibling Icechunk stores —
``reflectance``, ``sar_ascending``, ``sar_descending`` — and are exercised by
patching the store opener. Building those groups and dispatching on the path is
mechanical, identical everywhere it is needed, and was copied into two test
modules six times over; it lives here so a change to the mosaic schema (a band,
a coordinate, a dtype) is made once rather than found one failing module at a
time.

The seeds are fixed and DIFFERENT per store so the three are distinguishable in
an assertion — a test that accidentally reads ascending where it meant
descending fails rather than passing on identical noise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import zarr

from tessera_embeddings.config.inference import S2_BAND_ORDER
from tessera_embeddings.config.ingest import SOURCE_INDEX_VAR
from tessera_embeddings.inference.chunk_spec import ChunkSpec

#: Per-store seeds. Distinct so the three stores never hold identical data.
S2_SEED, ASC_SEED, DESC_SEED = 10, 20, 30


def _add_source_index(root: zarr.Group, rng: np.random.Generator, shape: tuple[int, int, int]) -> None:
    """A ``source_index`` band, as a provenance ingest writes it: 0 for no granule, else position + 1."""
    vals = rng.integers(0, 4, size=shape).astype(np.uint16)
    root.create_array(SOURCE_INDEX_VAR, shape=shape, dtype=np.uint16, chunks=shape)[:] = vals


def make_s2_group(n_t: int, h: int, w: int, seed: int = S2_SEED, *, source_index: bool = False) -> zarr.Group:
    """An S2 reflectance group: every band in ``S2_BAND_ORDER``, plus ``scl`` and ``time``.

    SCL mixes valid and invalid classes (0/4/5/8) so date pruning has something
    to prune; without that, tests of the coverage gate pass trivially.
    """
    rng = np.random.default_rng(seed)
    root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    for band in S2_BAND_ORDER:
        vals = rng.integers(100, 5000, size=(n_t, h, w)).astype(np.uint16)
        root.create_array(band, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)[:] = vals
    scl = rng.choice([0, 4, 5, 8], size=(n_t, h, w)).astype(np.uint8)
    root.create_array("scl", shape=scl.shape, dtype=scl.dtype, chunks=scl.shape)[:] = scl
    if source_index:
        _add_source_index(root, rng, (n_t, h, w))
    _write_time(root, n_t, freq="5D")
    return root


def make_sar_group(n_t: int, h: int, w: int, seed: int = ASC_SEED, *, source_index: bool = False) -> zarr.Group:
    """A single-orbit SAR group: ``0_VV``, ``0_VH`` and ``time``."""
    rng = np.random.default_rng(seed)
    root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    for name in ("0_VV", "0_VH"):
        vals = rng.integers(1000, 8000, size=(n_t, h, w)).astype(np.uint16)
        root.create_array(name, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)[:] = vals
    if source_index:
        _add_source_index(root, rng, (n_t, h, w))
    _write_time(root, n_t, freq="12D")
    return root


def _write_time(root: zarr.Group, n_t: int, *, freq: str) -> None:
    """Write the int64-nanosecond ``time`` coordinate the readers decode."""
    ns = pd.date_range("2024-01-01", periods=n_t, freq=freq).values.astype("datetime64[ns]").astype("int64")
    root.create_array("time", shape=ns.shape, dtype=np.int64, chunks=ns.shape)[:] = ns


def store_opener(
    chunk: ChunkSpec, *, n_t_s2: int = 10, n_t_sar: int = 5, source_index: bool = False
) -> Callable[..., zarr.Group]:
    """A patch target for ``open_store_as_zarr_group`` serving all three stores.

    Sized to ``chunk`` so the loaded arrays cover it exactly. Dispatches on the
    path the way the real layout does, and raises on anything unrecognised
    rather than returning a default — a test that asks for a store this fixture
    does not model should say so, not silently read S2.

    ``region`` is accepted and ignored: the credential/region threading passes
    it through, and a stub that rejected it would fail for the wrong reason.
    ``source_index`` builds mosaics a provenance ingest wrote.
    """
    h, w = chunk.height, chunk.width
    stores = {
        "reflectance": make_s2_group(n_t_s2, h, w, seed=S2_SEED, source_index=source_index),
        "ascending": make_sar_group(n_t_sar, h, w, seed=ASC_SEED, source_index=source_index),
        "descending": make_sar_group(n_t_sar, h, w, seed=DESC_SEED, source_index=source_index),
    }

    def _open_store(path: str, region: str | None = None, **_: object) -> zarr.Group:
        for key, group in stores.items():
            if key in path:
                return group
        raise ValueError(f"Unexpected store path: {path}")

    return _open_store
