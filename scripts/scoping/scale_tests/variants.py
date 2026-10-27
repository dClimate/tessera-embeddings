"""Store-layout variants under test — the single source of truth for T1/T2/T3/T8/T9.

Variant names are stable join keys in the metrics, so the registry holds the five v1.1 entries the
read benchmark was specified over plus the v2 geometries T9 and T10 scope; extend it rather than
forking per-test layouts. The v1.1 entries fix only chunk/shard geometry; the v2 ones may also vary
the int8 codec, a band-planar transpose, the inner-chunk order within a shard, the `scales`
geometry, the dimension order and variable-width band shards, which T9 and T10 derive as arms with
:func:`arm`. dtype and fill stay constant.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import zarr
from zarr.codecs import BloscCodec, ShardingCodec, TransposeCodec, ZstdCodec

from tessera_embeddings.config.inference import EMBEDDING_DIM
from tessera_embeddings.config.store_layout import clamp_chunks_and_shards, pcodec_serializer

#: Variable-width band shards (a rectilinear chunk grid) are opt-in in zarr, for reading as for writing. Switched on
#: here because every script and cold subprocess imports this module; regular arrays are unaffected.
zarr.config.set({"array.rectilinear_chunks": True})

#: Canonical dimension order for embedding arrays (ADR D2: band varies fastest).
DIMS: tuple[str, str, str, str] = ("time", "northing", "easting", "band")
#: Embedding channel count.
BAND: int = EMBEDDING_DIM
#: Axis permutation from :data:`DIMS` to the band-first order (time, band, northing, easting).
BAND_FIRST: tuple[int, int, int, int] = (0, 3, 1, 2)

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
    co-chunks ``scales`` with the embeddings' spatial chunk. ``band_first`` stores the array as
    (time, band, northing, easting); ``chunks`` and ``shards`` are still written over :data:`DIMS`.
    ``band_edges`` makes the shard grid variable-width along the band axis: each shard's band extent
    in turn, summing to 128, each a whole number of inner chunks. ``shards`` then sets the other axes.
    """

    name: str
    chunks: tuple[int, int, int, int]
    shards: tuple[int, int, int, int] | None
    codec: str = "zstd"
    transpose: bool = False
    write_order: str = "morton"
    scales_chunks: tuple[int, int, int] | None = None
    band_first: bool = False
    band_edges: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate shard/chunk alignment and arm names so a bad entry fails loudly."""
        if self.band_edges is not None and (
            self.shards is None or sum(self.band_edges) != BAND or any(e % self.chunks[3] for e in self.band_edges)
        ):
            raise ValueError(f"{self.name}: band edges {self.band_edges} must cover {BAND} bands in whole inner chunks")
        if self.shards is not None:
            for axis, (shard, chunk) in enumerate(zip(self.shards, self.chunks, strict=True)):
                if shard % chunk != 0 and not (axis == 3 and self.band_edges):
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
        # Band-first storage (time, band, northing, easting): a full vector is stacked from contiguous
        # band blocks rather than interleaved per pixel, and every chunk is band-planar by construction.
        Variant("v2_c2f", (1, 512, 512, 16), _V2_SHARD, band_first=True),
        Variant("v2_b2f", (1, 512, 512, 16), (1, 2048, 2048, 16), band_first=True),
        # Narrower band chunks for mobile readers (ADR 031 §6), all band-first. Band-split shards of 8 or 4 bands, at
        # 512² or 256² inner chunks; 16-band shards holding 8- or 4-band inner chunks; and variable-width shards whose
        # band extents follow the Matryoshka prefixes, with inner chunks as narrow as the narrowest shard (zarr
        # requires one inner shape that divides every shard).
        Variant("v2_b2f8", (1, 512, 512, 8), (1, 2048, 2048, 8), band_first=True),
        Variant("v2_b2f4", (1, 512, 512, 4), (1, 2048, 2048, 4), band_first=True),
        Variant("v2_b1f8", (1, 256, 256, 8), (1, 2048, 2048, 8), band_first=True),
        Variant("v2_b1f4", (1, 256, 256, 4), (1, 2048, 2048, 4), band_first=True),
        Variant("v2_b2f_i8", (1, 512, 512, 8), (1, 2048, 2048, 16), band_first=True),
        Variant("v2_b2f_i4", (1, 512, 512, 4), (1, 2048, 2048, 16), band_first=True),
        Variant("v2_m2f", (1, 512, 512, 4), _V2_SHARD, band_first=True, band_edges=(4, 4, 8, 16, 32, 64)),
        Variant("v2_m1f", (1, 256, 256, 4), _V2_SHARD, band_first=True, band_edges=(4, 4, 8, 16, 32, 64)),
        Variant("v2_m2f_x4", (1, 512, 512, 4), _V2_SHARD, band_first=True, band_edges=(4, 4, 8, 112)),
        Variant("v2_m2f_x3", (1, 512, 512, 8), _V2_SHARD, band_first=True, band_edges=(8, 8, 112)),
    ]
}


def selected(name: str | None) -> list[Variant]:
    """Return the v1.1 variants, or just the one named (for ``--variant``).

    T1 to T8 index ``embeddings`` band-last, so a band-first variant is refused rather than read along
    the wrong axis.
    """
    if name is None:
        return [v for v in VARIANTS.values() if not v.name.startswith("v2_")]
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; choices: {', '.join(VARIANTS)}")
    if VARIANTS[name].band_first:
        raise SystemExit(f"variant {name!r} is band-first, which only T9 and T10 read")
    return [VARIANTS[name]]


def arm(base: Variant, **changes: object) -> Variant:
    """Derive a T9 arm from a registry geometry, naming it by every field that differs."""
    v = dataclasses.replace(base, **changes)  # type: ignore[arg-type]
    parts = [base.name, "bandfirst" if v.band_first else "", v.codec, "T" if v.transpose else ""]
    parts.append(v.write_order if v.write_order != "morton" else "")
    if v.scales_chunks is not None:
        parts.append(f"s{v.scales_chunks[1]}")
    return dataclasses.replace(v, name="_".join(p for p in parts if p))


#: ADR 031's accepted layout, as T9 and T10 build it: band-first, band-split 16-band shards, zstd at level 1.
ACCEPTED = arm(VARIANTS["v2_b2f"], codec="zstd1", scales_chunks=(1, 256, 256))


def _band_first(arr: zarr.Array) -> bool:
    """Whether ``arr`` is stored band-first, read from its dimension names (ADR 031)."""
    return arr.metadata.dimension_names[1] == "band"


def window_key(
    arr: zarr.Array, t: int | slice, ys: slice | int, xs: slice | int, trailing: slice = slice(None)
) -> tuple:
    """The index of ``arr[t]`` over a window, in the array's own dimension order.

    One index serves band-last and band-first stores alike, and arrays with no band axis, for reads
    (:func:`read_window`) and for writes of a block put in order by :func:`to_storage_order`.
    """
    return (t, trailing, ys, xs) if _band_first(arr) else (t, ys, xs, trailing)[: arr.ndim]


def read_window(arr: zarr.Array, t: int, ys: slice | int, xs: slice | int, trailing: slice = slice(None)) -> np.ndarray:
    """``arr[t]`` over a window as ``(y, x[, band])``, whatever the array's dimension order."""
    out = np.asarray(arr[window_key(arr, t, ys, xs, trailing)])
    return np.moveaxis(out, 0, -1) if _band_first(arr) else out


def to_storage_order(arr: zarr.Array, block: np.ndarray) -> np.ndarray:
    """A ``(y, x, band)`` tile in ``arr``'s own dimension order, contiguous.

    Called outside any timer: a band-first pipeline stages its tiles band-first, so a write's cost
    excludes the reorder.
    """
    return np.ascontiguousarray(block.transpose(2, 0, 1)) if _band_first(arr) else block


def dask_chunks(arr: zarr.Array) -> dict[str, int]:
    """Dask chunks of one task per shard footprint and band chunk, keyed by the array's dimension names.

    Read from the array, as :func:`read_window` reads its order, so each arm is chunked to its own
    geometry: the control's 128-band chunk stays whole and a band-16 arm gets one task per band chunk.
    The footprint is the first shard's extent on each axis, from ``write_chunk_sizes``, which unlike
    ``.shards`` is defined for a variable-width shard grid.
    """
    names = arr.metadata.dimension_names
    footprint = {name: sizes[0] for name, sizes in zip(names, arr.write_chunk_sizes, strict=True)}
    return footprint | {"band": dict(zip(names, arr.chunks, strict=True))["band"]}


def embeddings_array_kwargs(variant: Variant, shape: tuple[int, int, int, int]) -> dict:
    """Return ``create_array`` kwargs for the int8 ``embeddings`` array.

    PCodec cannot encode int8 (numcodecs raises "Unsupported data type"), so — exactly as
    ``inference.assembly`` does — only the float arrays get PCodec. The control ``"zstd"`` arm is
    the Zarr v3 default bytes codec + default zstd compressor via ``"auto"``, which is what the
    library writes. ``transpose`` stores each inner chunk band-planar (time, band, y, x) through
    the core ``transpose`` codec; the logical dim order is unchanged. Chunks are clamped to the
    shape so a store smaller than one nominal chunk still creates cleanly.

    A variable-width shard grid rounds the spatial extent up to whole inner chunks, so the array can
    be larger than ``shape``: icechunk parses such a grid only with every axis listed edge by edge,
    refusing the specification's bare-integer shorthand, and zarr then requires the edges to sum to
    the extent with every edge a whole number of inner chunks. The margin is fill, never stored.
    """
    if variant.band_edges is not None:
        shape = tuple(math.ceil(n / c) * c for n, c in zip(shape, variant.chunks, strict=True))  # type: ignore[assignment]
    chunks, shards = clamp_chunks_and_shards(shape, variant.chunks, variant.shards)
    order = BAND_FIRST if variant.band_first else (0, 1, 2, 3)
    kwargs: dict = {
        "shape": tuple(shape[i] for i in order),
        "chunks": tuple(chunks[i] for i in order),
        "dtype": np.dtype("int8"),
        "fill_value": 0,
        "dimension_names": tuple(DIMS[i] for i in order),
        "serializer": "auto",
        "compressors": COMPRESSORS[variant.codec],
    }
    if variant.transpose:
        kwargs["filters"] = (TransposeCodec(order=(0, 3, 1, 2)),)
    if variant.band_edges is not None and shards is not None:
        # Every axis edge by edge (see above); a last spatial edge is the remainder, in whole inner chunks.
        edges = [[size] * (n // size) + [n % size] * bool(n % size) for n, size in zip(shape, shards, strict=True)]
        edges[3] = list(variant.band_edges)
        kwargs["shards"] = tuple(edges[i] for i in order)
    elif shards is not None:
        kwargs["shards"] = tuple(shards[i] for i in order)
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
