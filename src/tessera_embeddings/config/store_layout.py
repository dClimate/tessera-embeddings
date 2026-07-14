"""Store-layout presets: the on-disk geometry (chunks, shards, codecs) per array.

A ``StoreLayout`` names, per data variable, the inner-chunk shape, optional
shard shape, dtype, fill value, and codec. It is the single source of truth that
both the empty-store seeder and the shard writer consult, so seeding and writing
can never disagree about geometry.

Two presets (ADR-008 D2/D3):

* ``LEGACY`` — today's single-ROI output: ``(1, 500, 500, 4)`` unsharded
  embeddings, PCodec floats. Default for existing single-ROI entry points so
  vanilla users are unaffected (D8).
* ``GLOBAL_V1`` — the global campaign: ``(1, 256, 256, 128)`` full-band int8
  inner chunks in ``(1, 2048, 2048, 128)`` shards for ``embeddings``; the 3-D
  companion arrays (``scales``, ``embedding_std``, obs counts) share the same
  2048² spatial shards. **Both `embeddings` and `scales` are sharded** — leaving
  `scales` unsharded caps the object-count win (d3v2 E4). PCodec composes with
  the sharding codec (verified: round-trip + fill on partial shards).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from zarr.codecs.numcodecs import PCodec as _PCodecZarr3

from tessera_embeddings.config.inference import EMBEDDING_DIM

# PCodec is intentionally a Zarr v3 *serializer* (array->bytes), not a
# bytes->bytes compressor; silence the "not in the Zarr v3 spec" warning it
# emits on construction. Mirrors inference.assembly (kept local to avoid a
# config -> inference import).
warnings.filterwarnings("ignore", message="Numcodecs codecs are not in the Zarr version 3")

DIMS_4D: tuple[str, str, str, str] = ("time", "northing", "easting", "band")
DIMS_3D: tuple[str, str, str] = ("time", "northing", "easting")

#: obs-count variables carried through inference (kept local to avoid importing
#: from inference.assembly).
OBS_COUNT_VARS: tuple[str, ...] = ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count")

# codec keys
_ZSTD = "zstd"  # default bytes codec + default (zstd) compressor
_PCODEC = "pcodec"  # PCodec serializer, no bytes-compressor
_RAW = "raw"  # default bytes codec, no compressor


def _pcodec() -> _PCodecZarr3:
    """Return a fresh PCodec serializer instance."""
    return _PCodecZarr3()


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
        chunks = tuple(min(c, s) for c, s in zip(self.chunks, shape, strict=True))
        kwargs: dict = {
            "shape": shape,
            "chunks": chunks,
            "dtype": np.dtype(self.dtype),
            "fill_value": self.fill_value,
            "dimension_names": self.dims,
        }
        if self.shards is not None:
            kwargs["shards"] = tuple(
                max(c, (min(sh, s) // c) * c) for sh, s, c in zip(self.shards, shape, chunks, strict=True)
            )
        if self.codec == _PCODEC:
            kwargs["serializer"] = _pcodec()
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


# Today's single-ROI output, reproduced exactly (D8 — vanilla users unaffected).
# Band split into 4 (== EMBEDDING_DIM // 32), unsharded; PCodec floats; raw obs.
# ``embedding_std`` is 4-D (per-band std, mirroring ``embeddings``) as the
# historical engine wrote it; it is never produced under v1.1 (deterministic
# sampling forces ``compute_std=False``) but the schema must stay faithful.
LEGACY = StoreLayout(
    name="legacy",
    arrays={
        "embeddings": ArrayLayout(DIMS_4D, (1, 500, 500, EMBEDDING_DIM // 32), "int8", 0, _ZSTD),
        "scales": ArrayLayout(DIMS_3D, (1, 500, 500), "float32", float("nan"), _PCODEC),
        "embedding_std": ArrayLayout(DIMS_4D, (1, 500, 500, EMBEDDING_DIM // 32), "float32", float("nan"), _PCODEC),
        **_obs((1, 500, 500), None, _RAW),
    },
)

# The global campaign: 256-px full-band inner chunks in 2048² shards; scales
# sharded the same way (D3). 8x8 = 64 inner chunks per shard. ``embedding_std``
# mirrors ``scales``' treatment (float32 + PCodec, same spatial shards) on its
# natural per-band 4-D dims; never produced under v1.1 (see LEGACY note).
_INNER_4D = (1, 256, 256, EMBEDDING_DIM)
_SHARD_4D = (1, 2048, 2048, EMBEDDING_DIM)
_INNER_3D = (1, 256, 256)
_SHARD_3D = (1, 2048, 2048)
GLOBAL_V1 = StoreLayout(
    name="global_v1",
    arrays={
        "embeddings": ArrayLayout(DIMS_4D, _INNER_4D, "int8", 0, _ZSTD, shards=_SHARD_4D),
        "scales": ArrayLayout(DIMS_3D, _INNER_3D, "float32", float("nan"), _PCODEC, shards=_SHARD_3D),
        "embedding_std": ArrayLayout(DIMS_4D, _INNER_4D, "float32", float("nan"), _PCODEC, shards=_SHARD_4D),
        **_obs(_INNER_3D, _SHARD_3D, _ZSTD),
    },
)

#: The 2048-px shard pitch (also the aligned inference tile size).
SHARD_PX: int = 2048
INNER_PX: int = 256
