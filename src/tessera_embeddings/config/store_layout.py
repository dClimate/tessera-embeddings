"""Store-layout presets: the on-disk geometry (chunks, shards, codecs) per array.

A ``StoreLayout`` names, per data variable, the inner-chunk shape, optional
shard shape, dtype, fill value, and codec. It is the single source of truth that
both the empty-store seeder and the shard writer consult, so seeding and writing
can never disagree about geometry.

``SINGLE`` and ``GLOBAL`` share ONE geometry (ADR-008 D2/D3), built from one
definition so it cannot drift: ``(1, 256, 256, 128)`` full-band int8 inner chunks
in ``(1, 2048, 2048, 128)`` shards for ``embeddings``, with the 3-D companion
arrays (``scales``, obs counts) on the same 2048² spatial shards.

They differ in ONE thing, and only in which variables exist: **``embedding_std``
is single-ROI only and is never part of the global store.** v1.1's sampling is
deterministic, so there is no spread to record, and an array that is created in
every zone group and never written is a schema surprise for every downstream
reader of the published product. The geometry is still shared, so the two cannot
diverge in chunking — which is what the shared definition exists to prevent.

**Both ``embeddings`` and ``scales`` are sharded** — leaving ``scales`` unsharded
caps the object-count win (d3v2 E4). PCodec composes with the sharding codec
(verified: round-trip + fill on partial shards).

One shard is one inference tile is a quarter of one ingest chunk, so the chain
divides evenly end to end and nothing rechunks.

A layout is read only when a store — or a variable missing from one — is CREATED,
never to reshape an array that already exists. A variable joining an EXISTING store
takes that store's chunk and shard geometry rather than the preset's (see
``inference.assembly._layout_matching_store``): every data variable must agree on a
write granularity, so a store tiled by an older preset stays appendable.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from zarr.codecs.numcodecs import PCodec as _PCodecZarr3

from tessera_embeddings.config.inference import EMBEDDING_DIM

# PCodec is intentionally a Zarr v3 *serializer* (array->bytes), not a
# bytes->bytes compressor; silence the "not in the Zarr v3 spec" warning it
# emits on construction. This module is the codec's canonical home
# (inference.assembly and the scale-test harness import it from here).
warnings.filterwarnings("ignore", message="Numcodecs codecs are not in the Zarr version 3")

DIMS_4D: tuple[str, str, str, str] = ("time", "northing", "easting", "band")
DIMS_3D: tuple[str, str, str] = ("time", "northing", "easting")

#: obs-count variables carried through inference (canonical definition;
#: inference.assembly re-exports it).
OBS_COUNT_VARS: tuple[str, ...] = ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count")

# codec keys
_ZSTD = "zstd"  # default bytes codec + default (zstd) compressor
_PCODEC = "pcodec"  # PCodec serializer, no bytes-compressor
_RAW = "raw"  # default bytes codec, no compressor


def pcodec_serializer() -> _PCodecZarr3:
    """Return a fresh PCodec serializer instance (Zarr v3 array->bytes codec)."""
    return _PCodecZarr3()


def clamp_chunks_and_shards(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Clamp nominal chunk/shard sizes to an array's ``shape``.

    Chunks are clamped to the shape; shards are clamped to the shape and then
    floored to a whole multiple of the clamped chunks (Zarr v3 requires shards
    to be exact chunk multiples), never below one chunk — so a store smaller
    than one nominal chunk/shard still creates cleanly. The single
    implementation of this load-bearing geometry math; both
    :meth:`ArrayLayout.create_kwargs` and the empty-store seeder call it.
    """
    # max(..., 1): a zero-extent axis is legal (a mosaic seeded with an EMPTY
    # time axis, appended per date), but a zero chunk edge is not — keep the
    # nominal chunk there so the axis grows into its normal chunking.
    clamped = tuple(max(min(c, s), 1) for c, s in zip(chunks, shape, strict=True))
    if shards is None:
        return clamped, None
    clamped_shards = tuple(max(c, (min(sh, s) // c) * c) for sh, s, c in zip(shards, shape, clamped, strict=True))
    return clamped, clamped_shards


@dataclasses.dataclass(frozen=True)
class ArrayLayout:
    """On-disk geometry for one data variable.

    ``chunks``/``shards`` are element counts over ``dims``. ``codec`` is one of
    ``"zstd"`` (default serializer + zstd), ``"pcodec"`` (PCodec serializer, no
    compressor — floats only), or ``"raw"`` (default serializer, no compressor).
    """

    dims: tuple[str, ...]
    chunks: tuple[int, ...]
    dtype: str
    fill_value: float | int
    codec: str
    shards: tuple[int, ...] | None = None

    def create_kwargs(self, shape: tuple[int, ...]) -> dict:
        """Build ``zarr.Group.create_array`` kwargs for an array of ``shape``.

        Chunks (and shards, kept a whole multiple of the clamped chunks) are
        clamped to ``shape`` so a store smaller than one nominal chunk/shard
        still creates cleanly.
        """
        if len(shape) != len(self.dims):
            raise ValueError(f"shape {shape} has {len(shape)} dims, layout expects {len(self.dims)} ({self.dims})")
        chunks, shards = clamp_chunks_and_shards(shape, self.chunks, self.shards)
        kwargs: dict = {
            "shape": shape,
            "chunks": chunks,
            "dtype": np.dtype(self.dtype),
            "fill_value": self.fill_value,
            "dimension_names": self.dims,
        }
        if shards is not None:
            kwargs["shards"] = shards
        if self.codec == _PCODEC:
            kwargs["serializer"] = pcodec_serializer()
            kwargs["compressors"] = None
        elif self.codec == _RAW:
            kwargs["serializer"] = "auto"
            kwargs["compressors"] = None
        else:  # _ZSTD
            kwargs["serializer"] = "auto"
            kwargs["compressors"] = "auto"
        return kwargs


@dataclasses.dataclass(frozen=True)
class StoreLayout:
    """The geometry of every data variable in a store, keyed by variable name."""

    name: str
    arrays: dict[str, ArrayLayout]

    def for_var(self, var: str) -> ArrayLayout:
        """Return the layout for a data variable, or raise if unknown."""
        try:
            return self.arrays[var]
        except KeyError as exc:
            raise KeyError(f"{self.name} has no layout for variable {var!r} (known: {sorted(self.arrays)})") from exc


def _obs(chunks: tuple[int, ...], shards: tuple[int, ...] | None, codec: str) -> dict[str, ArrayLayout]:
    """Build the three obs-count array layouts (identical geometry)."""
    layout = ArrayLayout(dims=DIMS_3D, chunks=chunks, dtype="uint16", fill_value=0, codec=codec, shards=shards)
    return dict.fromkeys(OBS_COUNT_VARS, layout)


#: The 2048-px shard pitch — also the inference read-tile size, so one tile is
#: exactly one shard (ADR-008 D3; ``config.inference.INFERENCE_CHUNK_SIZE`` is
#: pinned to this by a test, since importing it here would be circular) — and
#: the 256-px inner-chunk size. The single numeric source both presets are built
#: from, so the constants and the presets cannot drift.
SHARD_PX: int = 2048
INNER_PX: int = 256

_INNER_4D = (1, INNER_PX, INNER_PX, EMBEDDING_DIM)
_SHARD_4D = (1, SHARD_PX, SHARD_PX, EMBEDDING_DIM)
_INNER_3D = (1, INNER_PX, INNER_PX)
_SHARD_3D = (1, SHARD_PX, SHARD_PX)


def _sharded_arrays(*, include_std: bool) -> dict[str, ArrayLayout]:
    """The embedding-store geometry: 256-px full-band inner chunks in 2048² shards.

    8x8 = 64 inner chunks per shard; the band axis is never split (D2), so a
    reader gets a pixel's whole 128-dimensional embedding from one object.

    ``include_std`` adds ``embedding_std``, which is 4-D and mirrors ``scales``'
    float32 + PCodec treatment. It is for the SINGLE-ROI path only: v1.1's
    deterministic sampling forces ``compute_std=False``, so in the global store it
    would be created in all 120 zone groups and never written — a schema surprise
    for every downstream reader of a published petabyte. The write path already
    treats it as optional in the destination (``assembly`` filters on presence), so
    its absence needs no other change.

    A fresh dict per call: ``StoreLayout.arrays`` is mutable, and two presets
    sharing one instance would let a mutation of either reach both.
    """
    arrays = {
        "embeddings": ArrayLayout(DIMS_4D, _INNER_4D, "int8", 0, _ZSTD, shards=_SHARD_4D),
        "scales": ArrayLayout(DIMS_3D, _INNER_3D, "float32", float("nan"), _PCODEC, shards=_SHARD_3D),
        **_obs(_INNER_3D, _SHARD_3D, _ZSTD),
    }
    if include_std:
        arrays["embedding_std"] = ArrayLayout(DIMS_4D, _INNER_4D, "float32", float("nan"), _PCODEC, shards=_SHARD_4D)
    return arrays


#: Single-ROI output. Its own name because callers pass a preset explicitly and
#: the name lands in the store's creating commit message.
SINGLE = StoreLayout(name="single", arrays=_sharded_arrays(include_std=True))

#: The global campaign (ADR-008 D3). No ``embedding_std``, ever — see
#: :func:`_sharded_arrays`.
GLOBAL = StoreLayout(name="global", arrays=_sharded_arrays(include_std=False))
