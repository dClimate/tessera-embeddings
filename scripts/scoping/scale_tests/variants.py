"""Store-layout variants under test — the single source of truth for T1/T2/T3/T8/T9.

Variant names are stable join keys in the metrics, so the registry holds the five v1.1 entries the
read benchmark was specified over plus the four v2 geometries T9 scopes; extend it rather than
forking per-test layouts. The v1.1 entries fix only chunk/shard geometry; the v2 ones may also vary
the int8 codec, a band-planar transpose, the inner-chunk order within a shard and the `scales`
geometry, which T9 derives as arms with :func:`arm`. dtype, fill and dim order stay constant.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import zarr
from zarr.codecs import BloscCodec, ShardingCodec, TransposeCodec, ZstdCodec

from tessera_embeddings.config.inference import EMBEDDING_DIM
from tessera_embeddings.config.store_layout import clamp_chunks_and_shards, pcodec_serializer

#: Canonical dimension order for embedding arrays (ADR D2: band varies fastest).
DIMS: tuple[str, str, str, str] = ("time", "northing", "easting", "band")
#: Embedding channel count.
BAND: int = EMBEDDING_DIM

#: int8 ``embeddings`` compressor arms; the float arrays always use PCodec. ``"auto"`` is zarr's
#: default zstd, which is what the library writes today.
COMPRESSORS: dict[str, object] = {
    "zstd": "auto",
    "zstd1": (ZstdCodec(level=1),),
    "raw": None,
    "blosc_bitshuffle": (BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),),
}
#: Inner-chunk orders within a shard (zarr >= 3.4). Write-time only: not persisted in metadata.
WRITE_ORDERS: tuple[str, ...] = ("morton", "lexicographic", "colexicographic")


@dataclasses.dataclass(frozen=True)
class Variant:
    """One store layout: chunk geometry, optional sharding, and the v2 arms' codec choices.

    ``chunks`` and ``shards`` are in element units over :data:`DIMS`. ``shards``
    is ``None`` for the unsharded layouts; when set it must be a whole multiple
    of ``chunks`` on every axis (Zarr v3 requirement). ``scales_chunks`` of ``None``
    co-chunks ``scales`` with the embeddings' spatial chunk.
    """

    name: str
    chunks: tuple[int, int, int, int]
    shards: tuple[int, int, int, int] | None
    codec: str = "zstd"
    transpose: bool = False
    write_order: str = "morton"
    scales_chunks: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        """Validate shard/chunk alignment and arm names so a bad entry fails loudly."""
        if self.shards is not None:
            for axis, (shard, chunk) in enumerate(zip(self.shards, self.chunks, strict=True)):
                if shard % chunk != 0:
                    raise ValueError(f"{self.name}: shard {shard} not a multiple of chunk {chunk} on axis {axis}")
        if self.codec not in COMPRESSORS or self.write_order not in WRITE_ORDERS:
            raise ValueError(f"{self.name}: unknown codec {self.codec!r} or write order {self.write_order!r}")


_V2_SHARD = (1, 2048, 2048, BAND)

VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in [
        Variant("c500_band4", (1, 500, 500, 4), None),  # current layout, baseline
        Variant("c500_full", (1, 500, 500, 128), None),  # 32 MB chunks
        Variant("c384_full", (1, 384, 384, 128), None),  # ~19 MB
        Variant("c256_full", (1, 256, 256, 128), None),  # 8.4 MB, AlphaEarth-like
        Variant("c256_sharded", (1, 256, 256, 128), (1, 2048, 2048, 128)),  # ~0.5 GB objects
        # v2 (T9): the v1.1 geometry as control, then band-16 inner chunks in the same shard, and
        # band-split shards (8 objects per tile) as the fallback rung.
        Variant("v2_c0", (1, 256, 256, 128), _V2_SHARD),
        Variant("v2_c1", (1, 256, 256, 16), _V2_SHARD),
        Variant("v2_c2", (1, 512, 512, 16), _V2_SHARD),
        Variant("v2_b2", (1, 512, 512, 16), (1, 2048, 2048, 16)),
    ]
}


def selected(name: str | None) -> list[Variant]:
    """Return the v1.1 variants, or just the one named (for ``--variant``)."""
    if name is None:
        return [v for v in VARIANTS.values() if not v.name.startswith("v2_")]
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; choices: {', '.join(VARIANTS)}")
    return [VARIANTS[name]]


def arm(base: Variant, **changes: object) -> Variant:
    """Derive a T9 arm from a registry geometry, naming it by every field that differs."""
    v = dataclasses.replace(base, **changes)  # type: ignore[arg-type]
    parts = [base.name, v.codec, "T" if v.transpose else "", v.write_order if v.write_order != "morton" else ""]
    if v.scales_chunks is not None:
        parts.append(f"s{v.scales_chunks[1]}")
    return dataclasses.replace(v, name="_".join(p for p in parts if p))


def embeddings_array_kwargs(variant: Variant, shape: tuple[int, int, int, int]) -> dict:
    """Return ``create_array`` kwargs for the int8 ``embeddings`` array.

    PCodec cannot encode int8 (numcodecs raises "Unsupported data type"), so — exactly as
    ``inference.assembly`` does — only the float arrays get PCodec. The control ``"zstd"`` arm is
    the Zarr v3 default bytes codec + default zstd compressor via ``"auto"``, which is what the
    library writes. ``transpose`` stores each inner chunk band-planar (time, band, y, x) through
    the core ``transpose`` codec; the logical dim order is unchanged. Chunks are clamped to the
    shape so a store smaller than one nominal chunk still creates cleanly.
    """
    chunks, shards = clamp_chunks_and_shards(shape, variant.chunks, variant.shards)
    kwargs: dict = {
        "shape": shape,
        "chunks": chunks,
        "dtype": np.dtype("int8"),
        "fill_value": 0,
        "dimension_names": DIMS,
        "serializer": "auto",
        "compressors": COMPRESSORS[variant.codec],
    }
    if variant.transpose:
        kwargs["filters"] = (TransposeCodec(order=(0, 3, 1, 2)),)
    if shards is not None:
        kwargs["shards"] = shards
    return kwargs


def scales_array_kwargs(variant: Variant, shape: tuple[int, int, int]) -> dict:
    """Return ``create_array`` kwargs for the float32 + NaN ``scales`` array.

    ``scales`` carries the "never written" sentinel (NaN, ADR D1). By default it is co-chunked
    with the embeddings *spatial* chunks and co-SHARDED with them whenever the variant shards
    (matching ``GLOBAL``, where unsharded scales would cap the object-count win — D3); a v2 arm's
    ``scales_chunks`` overrides the inner chunk only.
    """
    spatial = variant.scales_chunks or (variant.chunks[0], variant.chunks[1], variant.chunks[2])
    spatial_shards = (variant.shards[0], variant.shards[1], variant.shards[2]) if variant.shards else None
    chunks, shards = clamp_chunks_and_shards(shape, spatial, spatial_shards)
    kwargs: dict = {
        "shape": shape,
        "chunks": chunks,
        "dtype": np.dtype("float32"),
        "fill_value": np.float32("nan"),
        "dimension_names": DIMS[:3],
        "serializer": pcodec_serializer(),
        "compressors": None,
    }
    if shards is not None:
        kwargs["shards"] = shards
    return kwargs


def apply_write_order(array: zarr.Array, order: str) -> None:
    """Set the inner-chunk order an opened sharded array writes with (zarr >= 3.4).

    The order is a ``ShardingCodec`` constructor argument with no config key and is not persisted,
    so reopening an array reverts it to morton; a writer has to re-apply it to every handle. The
    codec is a frozen dataclass, hence ``object.__setattr__``.
    """
    if order == "morton":
        return
    codec = next(c for c in array.metadata.codecs if isinstance(c, ShardingCodec))
    if not hasattr(codec, "subchunk_write_order"):
        raise RuntimeError(f"write order {order!r} needs zarr >= 3.4; {zarr.__version__} would write morton")
    object.__setattr__(codec, "subchunk_write_order", order)


def band_chunks(variant: Variant) -> int:
    """Number of band-axis chunks (embeddings refs per written spatial chunk)."""
    return math.ceil(BAND / variant.chunks[3])
