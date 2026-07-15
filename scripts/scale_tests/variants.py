"""Store-layout variants under test — the single source of truth for T1/T2/T3.

Variant names are stable join keys in the metrics, so this registry is exactly
the five entries from test-plan §3; extend it rather than forking per-test
layouts. Each variant fixes only the chunk/shard geometry; dtype, serializer,
fill, and dim order are constant across variants (below) because those are
decided (ADR D2) and not what the variants probe.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from tessera_embeddings.config.inference import EMBEDDING_DIM
from tessera_embeddings.config.store_layout import clamp_chunks_and_shards, pcodec_serializer

#: Canonical dimension order for embedding arrays (ADR D2: band varies fastest).
DIMS: tuple[str, str, str, str] = ("time", "northing", "easting", "band")
#: Embedding channel count.
BAND: int = EMBEDDING_DIM


@dataclasses.dataclass(frozen=True)
class Variant:
    """One store layout: chunk geometry and optional sharding.

    ``chunks`` and ``shards`` are in element units over :data:`DIMS`. ``shards``
    is ``None`` for the unsharded layouts; when set it must be a whole multiple
    of ``chunks`` on every axis (Zarr v3 requirement).
    """

    name: str
    chunks: tuple[int, int, int, int]
    shards: tuple[int, int, int, int] | None

    def __post_init__(self) -> None:
        """Validate shard/chunk alignment so a bad registry entry fails loudly."""
        if self.shards is not None:
            for axis, (shard, chunk) in enumerate(zip(self.shards, self.chunks, strict=True)):
                if shard % chunk != 0:
                    raise ValueError(f"{self.name}: shard {shard} not a multiple of chunk {chunk} on axis {axis}")


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in [
        Variant("c500_band4", (1, 500, 500, 4), None),  # current layout, baseline
        Variant("c500_full", (1, 500, 500, 128), None),  # 32 MB chunks
        Variant("c384_full", (1, 384, 384, 128), None),  # ~19 MB
        Variant("c256_full", (1, 256, 256, 128), None),  # 8.4 MB, AlphaEarth-like
        Variant("c256_sharded", (1, 256, 256, 128), (1, 2048, 2048, 128)),  # ~0.5 GB objects
    ]
}


def selected(name: str | None) -> list[Variant]:
    """Return all variants, or just the one named (for ``--variant``)."""
    if name is None:
        return list(VARIANTS.values())
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; choices: {', '.join(VARIANTS)}")
    return [VARIANTS[name]]


def embeddings_array_kwargs(variant: Variant, shape: tuple[int, int, int, int]) -> dict:
    """Return ``create_array`` kwargs for the int8 ``embeddings`` array.

    Serializer/compressor match the *library*: PCodec cannot encode int8
    (numcodecs raises "Unsupported data type"), so — exactly as
    ``inference.assembly`` does — only the float arrays get PCodec; ``embeddings``
    uses the Zarr v3 default bytes codec + default (zstd) compressor via
    ``"auto"``. Chunks are clamped to the shape so a store smaller than one
    nominal chunk still creates cleanly (the library ingest path clamps too).
    """
    chunks, shards = clamp_chunks_and_shards(shape, variant.chunks, variant.shards)
    kwargs: dict = {
        "shape": shape,
        "chunks": chunks,
        "dtype": np.dtype("int8"),
        "fill_value": 0,
        "dimension_names": DIMS,
        "serializer": "auto",
        "compressors": "auto",
    }
    if shards is not None:
        kwargs["shards"] = shards
    return kwargs


def scales_array_kwargs(variant: Variant, shape: tuple[int, int, int]) -> dict:
    """Return ``create_array`` kwargs for the float32 + NaN ``scales`` array.

    ``scales`` carries the "never written" sentinel (NaN, ADR D1) and is
    co-chunked with the embeddings *spatial* chunks (so one embeddings spatial
    chunk maps to exactly one scales chunk — keeping ref-counting clean). Left
    unsharded: sharding is an ``embeddings``-only question here.
    """
    spatial = (variant.chunks[0], variant.chunks[1], variant.chunks[2])
    chunks, _ = clamp_chunks_and_shards(shape, spatial, None)
    return {
        "shape": shape,
        "chunks": chunks,
        "dtype": np.dtype("float32"),
        "fill_value": np.float32("nan"),
        "dimension_names": DIMS[:3],
        "serializer": pcodec_serializer(),
        "compressors": None,
    }


def band_chunks(variant: Variant) -> int:
    """Number of band-axis chunks (embeddings refs per written spatial chunk)."""
    return math.ceil(BAND / variant.chunks[3])
