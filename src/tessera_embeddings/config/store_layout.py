"""Store-layout presets: the on-disk geometry (chunks, shards, codecs) per array.

A ``StoreLayout`` names, per data variable, the inner-chunk shape, optional shard shape, dtype,
fill value, and codec. It is the single source of truth both the empty-store seeder and the shard
writer consult, so seeding and writing can never disagree about geometry.

``SINGLE`` and ``GLOBAL`` share ONE geometry (ADR-008 D2/D3), built from one definition so it
cannot drift: ``(1, 256, 256, 128)`` full-band int8 inner chunks in ``(1, 2048, 2048, 128)``
shards for ``embeddings``, with the 3-D companion arrays (``scales``, obs counts) on the same
2048² spatial shards. They differ only in WHICH variables exist: **``embedding_std`` is
single-ROI only and is never part of the global store**, because v1.1's sampling is deterministic
(no spread to record) and an array created in every zone group and never written is a schema
surprise for every downstream reader of the published product.

**Both ``embeddings`` and ``scales`` are sharded** — leaving ``scales`` unsharded caps the
object-count win (d3v2 E4). PCodec composes with the sharding codec (verified: round-trip + fill
on partial shards).

One shard is one inference tile is a quarter of one ingest chunk, so the chain divides evenly end
to end and nothing rechunks.

A layout is read only when a store — or a variable missing from one — is CREATED, never to
reshape an array that already exists. A variable joining an EXISTING store takes that store's
chunk and shard geometry rather than the preset's (see
``inference.assembly._layout_matching_store``): every data variable must agree on a write
granularity, so a store tiled by an older preset stays appendable.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from zarr.codecs.numcodecs import PCodec as _PCodecZarr3

from tessera_embeddings.config.inference import EMBEDDING_DIM

# PCodec is intentionally a Zarr v3 *serializer* (array->bytes), not a bytes->bytes compressor; silence the "not in
# the Zarr v3 spec" warning it emits on construction. This module is the codec's canonical home (inference.assembly
# and the scale-test harness import it from here).
warnings.filterwarnings("ignore", message="Numcodecs codecs are not in the Zarr version 3")

DIMS_4D: tuple[str, str, str, str] = ("time", "northing", "easting", "band")
DIMS_3D: tuple[str, str, str] = ("time", "northing", "easting")
#: Trailing axis of :data:`MONTH_COVERED_VARS` — twelve calendar months, not bands.
DIMS_4D_MONTH: tuple[str, str, str, str] = ("time", "northing", "easting", "month")

#: The three observation sources a pixel can be seen by, as the prefix their per-sensor arrays are named from. ONE
#: list, because every per-sensor array is a pair — a count and a month mask — derived from a single validity mask per
#: sensor. Spelling the sensors once and deriving both name lists means a fourth source, or a renamed one, cannot
#: arrive in only half the arrays.
SENSORS: tuple[str, ...] = ("s2", "s1_asc", "s1_desc")

#: obs-count variables carried through inference (canonical definition; inference.assembly re-exports it).
OBS_COUNT_VARS: tuple[str, ...] = tuple(f"{sensor}_obs_count" for sensor in SENSORS)

#: Months in the coverage axis, and the coordinate values written for it (1 = January), so a reader can say
#: ``cov.sel(month=7)`` and mean July rather than having to know the axis is 0-based.
MONTHS_IN_YEAR = 12
MONTH_COORD = tuple(range(1, MONTHS_IN_YEAR + 1))

#: Per-pixel record of WHICH months a pixel was seen in, as twelve booleans, PER SENSOR.
#:
#: The ``*_obs_count`` arrays say how many usable observations a pixel had; these say how they were distributed. Only
#: the second distinguishes a pixel seen twenty times in July from one seen twelve times, once a month — for a
#: year-long embedding, a partial season versus a whole one.
#:
#: **Per sensor, because the sensors fail differently and a merged mask would hide it.** Optical gaps are weather and
#: seasonal: a cloudy monsoon removes the same months every year. Radar gaps are orbital, and the S1B failure left
#: whole regions with one orbit direction for years, so an ascending and a descending gap over the same pixel mean
#: different things about what the embedding saw. Or-ing the three would report a pixel as covered in a month it was
#: seen only by the sensor a given reader cannot use.
#:
#: **They gate nothing.** ``config.inference.OPTICAL_MIN_OBS`` remains the only rule deciding whether a pixel is
#: embedded. These let a reader apply their own view of sufficiency without re-deriving it from imagery we do not
#: publish — the mosaics behind them are deleted after a fill, so it is captured here or nowhere.
#:
#: Presence, not counts, and each mask partitions ITS OWN count: a month is covered when at least one timestep that
#: month passed the same validity test the paired ``*_obs_count`` totals — SCL classes for optical, a non-zero
#: backscatter sample for radar. Both come from one validity mask per sensor
#: (``data_loading.coverage_from_validity``), so "how many" and "which months" cannot disagree. ``False`` also reads
#: for an unwritten pixel, exactly as a count of 0.
#:
#: Stored as ``int8`` carrying the attribute ``dtype="bool"``, xarray's own representation of a boolean array, so a
#: reader gets booleans back while the on-disk type is the one the staging writer can produce (``Dataset.to_zarr``
#: stores bool as int8 whatever the encoding says, and assembly reads staged tiles with raw zarr).
MONTH_COVERED_VARS: tuple[str, ...] = tuple(f"{sensor}_month_covered" for sensor in SENSORS)

#: The month mask paired with each obs-count array, so a caller holding one can name the other without slicing a name
#: apart.
MONTH_COVERED_FOR_OBS: dict[str, str] = dict(zip(OBS_COUNT_VARS, MONTH_COVERED_VARS, strict=True))

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

    Chunks are clamped to the shape; shards are clamped to the shape and then floored to a whole
    multiple of the clamped chunks (Zarr v3 requires shards to be exact chunk multiples), never
    below one chunk, so a store smaller than one nominal chunk/shard still creates cleanly. The
    single implementation of this geometry math: both :meth:`ArrayLayout.create_kwargs` and the
    empty-store seeder call it.
    """
    # max(..., 1): a zero-extent axis is legal (a mosaic seeded with an EMPTY time axis, appended per date), but a
    # zero chunk edge is not — keep the nominal chunk so the axis grows into it.
    clamped = tuple(max(min(c, s), 1) for c, s in zip(chunks, shape, strict=True))
    if shards is None:
        return clamped, None
    clamped_shards = tuple(max(c, (min(sh, s) // c) * c) for sh, s, c in zip(shards, shape, clamped, strict=True))
    return clamped, clamped_shards


@dataclasses.dataclass(frozen=True)
class ArrayLayout:
    """On-disk geometry for one data variable.

    ``chunks``/``shards`` are element counts over ``dims``. ``codec`` is one of ``"zstd"``
    (default serializer + zstd), ``"pcodec"`` (PCodec serializer, no compressor — floats only),
    or ``"raw"`` (default serializer, no compressor).

    ``attrs`` are array attributes written at creation, as pairs so the layout stays hashable and
    two presets cannot share a mutable dict. Needed because a dtype is not always the whole story:
    xarray represents a boolean array as ``int8`` plus the attribute ``dtype="bool"``, and a
    reader gets booleans back only if the attribute is there.
    """

    dims: tuple[str, ...]
    chunks: tuple[int, ...]
    dtype: str
    fill_value: float | int
    codec: str
    shards: tuple[int, ...] | None = None
    attrs: tuple[tuple[str, str], ...] = ()

    def create_kwargs(self, shape: tuple[int, ...]) -> dict:
        """Build ``zarr.Group.create_array`` kwargs for an array of ``shape``.

        Chunks (and shards, kept a whole multiple of the clamped chunks) are clamped to ``shape``
        so a store smaller than one nominal chunk/shard still creates cleanly.
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


#: The 2048-px shard pitch — also the inference read-tile size, so one tile is exactly one shard (ADR-008 D3;
#: ``config.inference.INFERENCE_CHUNK_SIZE`` is pinned to this by a test, since importing it here would be circular) —
#: and the 256-px inner-chunk size. The single numeric source both presets are built from, so the constants and the
#: presets cannot drift.
SHARD_PX: int = 2048
INNER_PX: int = 256

_INNER_4D = (1, INNER_PX, INNER_PX, EMBEDDING_DIM)
_SHARD_4D = (1, SHARD_PX, SHARD_PX, EMBEDDING_DIM)
_INNER_3D = (1, INNER_PX, INNER_PX)
_SHARD_3D = (1, SHARD_PX, SHARD_PX)
# The month axis is never split, for the same reason the band axis is not: a reader gets a pixel's whole year from one
# object, so "was this pixel covered every month of the growing season" is one read rather than twelve.
_INNER_4D_MONTH = (1, INNER_PX, INNER_PX, MONTHS_IN_YEAR)
_SHARD_4D_MONTH = (1, SHARD_PX, SHARD_PX, MONTHS_IN_YEAR)


def _sharded_arrays(*, include_std: bool) -> dict[str, ArrayLayout]:
    """The embedding-store geometry: 256-px full-band inner chunks in 2048² shards.

    8x8 = 64 inner chunks per shard; the band axis is never split (D2), so a reader gets a pixel's
    whole 128-dimensional embedding from one object.

    ``include_std`` adds ``embedding_std``, 4-D and mirroring ``scales``' float32 + PCodec
    treatment. SINGLE-ROI only: v1.1's deterministic sampling forces ``compute_std=False``, so in
    the global store it would be created in all 120 zone groups and never written — a schema
    surprise for every downstream reader of a published petabyte. The write path already treats it
    as optional in the destination (``assembly`` filters on presence).

    A fresh dict per call: ``StoreLayout.arrays`` is mutable, and two presets sharing one instance
    would let a mutation of either reach both.
    """
    arrays = {
        "embeddings": ArrayLayout(DIMS_4D, _INNER_4D, "int8", 0, _ZSTD, shards=_SHARD_4D),
        "scales": ArrayLayout(DIMS_3D, _INNER_3D, "float32", float("nan"), _PCODEC, shards=_SHARD_3D),
        **_obs(_INNER_3D, _SHARD_3D, _ZSTD),
        # Twelve labelled planes rather than a packed integer, and zstd rather than PCodec. Measured on real coverage:
        # the planes cost 1.14x a packed 12-bit mask once compressed (0.24 bits a pixel against 0.21), because month
        # coverage is spatially smooth and a plane compresses ~400x while packed bits look like noise to the
        # compressor. The 6x uncompressed never reaches disk, and a plane per month needs no helper library to read.
        #
        # int8 + attrs dtype="bool" is xarray's OWN representation of a boolean array, forced here by the write path:
        # staged tiles are written with `Dataset.to_zarr`, which stores bool as int8 whatever the encoding asks, while
        # assembly reads those tiles with RAW zarr and sees int8 — so a bool destination refused every staged month
        # tile on the dtype guard. Matching the destination to what the writer can express keeps the guard intact, and
        # the attribute is what makes an xarray reader see booleans rather than 0/1, so `cov.sel(month=7)` is still a
        # boolean mask. Size is unchanged: numpy bool is 1 byte.
        **{
            var: ArrayLayout(
                DIMS_4D_MONTH,
                _INNER_4D_MONTH,
                "int8",
                0,
                _ZSTD,
                shards=_SHARD_4D_MONTH,
                attrs=(("dtype", "bool"),),
            )
            for var in MONTH_COVERED_VARS
        },
    }
    if include_std:
        arrays["embedding_std"] = ArrayLayout(DIMS_4D, _INNER_4D, "float32", float("nan"), _PCODEC, shards=_SHARD_4D)
    return arrays


#: Single-ROI output. Its own name because callers pass a preset explicitly and the name lands in the store's creating
#: commit message.
SINGLE = StoreLayout(name="single", arrays=_sharded_arrays(include_std=True))

#: The arrays every staged tile must carry and every destination must hold.
REQUIRED_VARS: tuple[str, str] = ("embeddings", "scales")

#: The per-pixel arrays BESIDES :data:`REQUIRED_VARS` that inference stages and assembly copies into the destination.
#:
#: DERIVED from the layout, so a new array is carried by construction. Enumerated by hand at the write path's four
#: decision points instead, adding an array to :func:`_sharded_arrays` is silently insufficient: the array is created
#: and seeded, tiles stage real values into it, and the destination copy drops it — publishing fill over a whole
#: zone-year. Assembly still filters on presence in BOTH the staged tile and the destination, so a store that predates
#: an array, or a run that stages nothing for one, is unaffected. Built from :data:`SINGLE` because it is the superset
#: (it alone has ``embedding_std``); the global path's filter drops what its zones do not hold.
CARRIED_VARS: tuple[str, ...] = tuple(v for v in SINGLE.arrays if v not in REQUIRED_VARS)


def trailing_extent(var: str, embedding_dim: int) -> int | None:
    """The trailing-axis extent a 4-D *var* must have, or ``None`` if it is 2-D per timestep.

    Read off the layout's own dims rather than a list of variable names, because the two 4-D
    trailing axes differ in width: ``band`` is as wide as the embedding, ``month`` is twelve. A
    single "4-D means band" rule rejects a correct month tile and shapes a cleared one wrongly.
    ``None`` for an unknown var, or for a band axis when *embedding_dim* is 0 (check disabled).
    """
    dims = SINGLE.arrays[var].dims if var in SINGLE.arrays else ()
    if len(dims) != 4:
        return None
    return MONTHS_IN_YEAR if dims[3] == "month" else (embedding_dim or None)


#: The global campaign (ADR-008 D3). No ``embedding_std``, ever — see :func:`_sharded_arrays`.
GLOBAL = StoreLayout(name="global", arrays=_sharded_arrays(include_std=False))
