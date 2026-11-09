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

from tessera_embeddings.config.inference import EMBEDDING_DIM, model_family
from tessera_embeddings.config.store_geometry import LEVEL0, PRESET_SPATIAL_DIMS, clamp_chunks_and_shards

# PCodec is intentionally a Zarr v3 *serializer* (array->bytes), not a bytes->bytes compressor; silence the "not in
# the Zarr v3 spec" warning it emits on construction. This module is the codec's canonical home (inference.assembly
# and the scale-test harness import it from here).
warnings.filterwarnings("ignore", message="Numcodecs codecs are not in the Zarr version 3")

DIMS_4D: tuple[str, str, str, str] = ("time", *PRESET_SPATIAL_DIMS, "band")
DIMS_3D: tuple[str, str, str] = ("time", *PRESET_SPATIAL_DIMS)
#: Trailing axis of :data:`MONTH_COVERED_VARS` — twelve calendar months, not bands.
DIMS_4D_MONTH: tuple[str, str, str, str] = ("time", *PRESET_SPATIAL_DIMS, "month")
#: v2's ``embeddings`` order: each chunk holds whole image planes, one per band (ADR 031 §3).
DIMS_4D_BAND_FIRST: tuple[str, str, str, str] = ("time", "band", *PRESET_SPATIAL_DIMS)

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
#: **They gate nothing.** The fill's optical depth rule, the store root's ``optical_min_obs`` or an update's lower
#: line, decides whether a pixel is embedded. These let a reader apply their own view of sufficiency without
#: re-deriving it from imagery we do not publish — the mosaics behind them are deleted after a fill, so it is captured
#: here or nowhere.
#:
#: Presence, not counts, and each mask partitions ITS OWN count: a month is covered when at least one timestep that
#: month passed the same validity test the paired ``*_obs_count`` totals — SCL classes for optical, a non-zero
#: backscatter sample for radar. Both come from one validity mask per sensor
#: (``data_loading.coverage_from_validity``), so "how many" and "which months" cannot disagree. ``False`` also reads
#: for an unwritten pixel, exactly as a count of 0.
#:
#: Stored as ``int8`` carrying the attribute ``dtype="bool"``, xarray's own representation of a boolean array, so a
#: reader gets booleans back while the on-disk type is the one the staging writer produces (it stages the masks as
#: int8 explicitly, ``ZarrWriter._stage_month_masks``, and assembly reads staged tiles with raw zarr).
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
    """The geometry of every data variable in a store, keyed by variable name.

    ``bands`` is the extent of the band axis, which the band shards must tile exactly. ``level``
    names the child group the arrays sit in, ``"0"`` for a v2 store (ADR 031 §1), or ``None`` for
    one that keeps them in the zone group or store root, as v1.1 does.
    """

    name: str
    arrays: dict[str, ArrayLayout]
    bands: int = EMBEDDING_DIM
    level: str | None = None

    def for_var(self, var: str) -> ArrayLayout:
        """Return the layout for a data variable, or raise if unknown."""
        try:
            return self.arrays[var]
        except KeyError as exc:
            raise KeyError(f"{self.name} has no layout for variable {var!r} (known: {sorted(self.arrays)})") from exc

    @property
    def band_first(self) -> bool:
        """Whether ``embeddings`` puts its band axis before the spatial pair, as v2 does (ADR 031 §3)."""
        return self.for_var("embeddings").dims[1] == "band"

    @property
    def spatial_dims(self) -> tuple[str, str]:
        """This layout's Y and X dimension names, read off the required ``scales`` array."""
        _, y, x = self.for_var("scales").dims
        return y, x

    def with_spatial_dims(self, spatial_dims: tuple[str, str]) -> StoreLayout:
        """This layout with its Y and X dimensions named *spatial_dims*; geometry unchanged.

        The presets spell them ``northing``/``easting``. A new store takes its model's names
        (``ModelFamily.spatial_dims``) and an existing one keeps its own
        (:func:`~tessera_embeddings.config.store_geometry.spatial_dims_of`).
        """
        rename = dict(zip(self.spatial_dims, spatial_dims, strict=True))
        arrays = {
            var: dataclasses.replace(array, dims=tuple(rename.get(d, d) for d in array.dims))
            for var, array in self.arrays.items()
        }
        return dataclasses.replace(self, arrays=arrays)


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
        # int8 + attrs dtype="bool" is xarray's OWN representation of a boolean array, and what the staging writer
        # stages (explicitly: from xarray 2026.9, `to_zarr` would store bool natively). Assembly reads staged tiles
        # with RAW zarr and compares dtypes, so destination and staged tile must agree — a bool destination refused
        # every staged month tile on the dtype guard. Matching the destination to what the writer stages keeps the
        # guard intact, and the attribute is what makes an xarray reader see booleans rather than 0/1, so
        # `cov.sel(month=7)` is still a boolean mask. Size is unchanged: numpy bool is 1 byte.
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


#: The global campaign (ADR-008 D3). No ``embedding_std``, ever — see :func:`_sharded_arrays`.
GLOBAL = StoreLayout(name="global", arrays=_sharded_arrays(include_std=False))


#: v2's shards and chunks along the band axis: 16 dimensions to a shard, the shortest prefix v2 was trained on, cut
#: into chunks of 4, so a phone or browser fetches only the dimensions it reads (ADR 031 §4).
V2_BANDS_PER_SHARD = 16
V2_BANDS_PER_CHUNK = 4
#: v2's chunk edge in pixels. 256² builds four times the Dask tasks at default chunks; 1,024² makes a full vector
#: cost 134 MB.
V2_CHUNK_PX = 512


def _band_first(arrays: dict[str, ArrayLayout]) -> dict[str, ArrayLayout]:
    """*arrays* with the band-axis arrays in v2's band-first geometry; every other array unchanged."""
    band_first = {
        "chunks": (1, V2_BANDS_PER_CHUNK, V2_CHUNK_PX, V2_CHUNK_PX),
        "shards": (1, V2_BANDS_PER_SHARD, SHARD_PX, SHARD_PX),
        "dims": DIMS_4D_BAND_FIRST,
    }
    return {
        var: dataclasses.replace(array, **band_first) if array.dims == DIMS_4D else array
        for var, array in arrays.items()
    }


#: The v2 stores: ``embeddings`` (and the single-area ``embedding_std``) band-first, in 16-band shards of 4-band
#: chunks, every array in level group ``0`` (ADR 031).
SINGLE_V2 = StoreLayout(name="single-v2", arrays=_band_first(_sharded_arrays(include_std=True)), level=LEVEL0)
GLOBAL_V2 = StoreLayout(name="global-v2", arrays=_band_first(_sharded_arrays(include_std=False)), level=LEVEL0)


def layout_for(model_version: str, *, single: bool) -> StoreLayout:
    """The preset a NEW store takes from its model: v1.1's geometry, or v2's (ADR 031 §5).

    An existing store keeps the geometry it was created with whatever the model, which the write
    path reads from the store itself.
    """
    if model_family(model_version).generation == "v2":
        return SINGLE_V2 if single else GLOBAL_V2
    return SINGLE if single else GLOBAL
