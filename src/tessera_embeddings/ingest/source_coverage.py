"""Whether the optical source publishes anything for a cell's LIVE land, before we spend on it.

Zone 31S year 2021 failed nine minutes into a run with ``IcechunkError: the repository doesn't
exist``. The cause was upstream of us: most of 31S's live land is covered by MGRS tiles that
publish nothing for 2021, so too little imagery reached the live windows to produce a
reflectance store, and the first thing to open one reported a storage symptom instead.

**Why this deserves a preflight when the radar equivalent does not.** A cell with no radar has
a defined answer — optical-only embeddings, recorded and flagged. A cell with no optical has no
fallback at all: that land gets no embedding. And the difference is invisible until a fleet has
been paid for.

**Live land is the whole question.** A zone's bounding box is not a useful proxy: 31S spans 1°S
to 55°S because its live tiles are two unrelated islands, and a neighbouring MGRS tile that
publishes every year (31MHU) sits inside that box while covering only part of the land. A count
of items in the box would have said "plenty" and been wrong.

**Two corrections make the answer trustworthy, and without them it is worse than useless.**

1. *A square a point falls in is often not a square ESA publishes.* Near a UTM zone boundary the
   land is carried by a tile of the ADJACENT zone, so geometric attribution names squares that
   have no data at all — measured at **599 of 3,150** derived squares, 19%. Treating those as
   coverage gaps would fire on every zone. A tile publishing **no year whatsoever** is therefore
   read as a naming artefact, not a gap; a tile publishing some years but not the requested one
   is the real signal, and there were **24** of those against 599 artefacts.
2. *Tiles must be weighted by the land they carry.* One remote-island tile missing a year is not
   the same as the tile under most of a zone's land missing it. Each MGRS tile is weighted by how
   many live 2048-px tiles fall in it, so the fraction reported is of land rather than of names.

This module holds the geometry and the verdict; the listing is a cloud concern and lives in
:mod:`tessera_embeddings.providers.aws.sentinel_cogs`, injected as ``years_fn`` so this stays
provider-agnostic (ADR-004).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import cast, final

import numpy as np
import zarr
from pyproj import Transformer

from tessera_embeddings.ingest.roi import read_roi_metadata

logger = logging.getLogger(__name__)

#: Years-per-tile lookup, as :func:`providers.aws.sentinel_cogs.published_years` provides it.
YearsFn = Callable[[Iterable[str]], dict[str, frozenset[str]]]

#: Below this share of live land covered, a cell is reported as barely covered.
#:
#: Set from the measured distribution rather than chosen: land-weighted coverage across the
#: fifteen zones surveyed on 2026-08-05 runs 9.7 to 12.0 months out of 12 for every zone except
#: 31S, which sits at 2.0 for 2021 and 7.2 for 2022. So anything under about a third of live
#: land is far outside the normal range, and nothing real sits near the boundary. It is a
#: WARNING threshold, not a refusal — see :func:`assert_optical_is_published` for why only the
#: fully-empty case refuses.
BARELY_COVERED_FRACTION = 0.33


@final
@dataclass(frozen=True)
class OpticalCoverage:
    """What the source publishes for one cell's live land.

    Attributes:
        zone: UTM zone label, e.g. ``"31S"``.
        year: Calendar year asked about.
        live_weight: MGRS tile → number of live 2048-px tiles falling in it.
        tiles_with_data: Tiles that publish ``year``.
        tiles_never_published: Tiles that publish NO year at all — read as geometric
            attribution artefacts rather than gaps, and excluded from the verdict.
    """

    zone: str
    year: int
    live_weight: dict[str, int]
    tiles_with_data: tuple[str, ...]
    tiles_never_published: tuple[str, ...]

    @property
    def tiles_considered(self) -> tuple[str, ...]:
        """Live-land tiles that publish something, so their year-by-year record means something."""
        artefacts = set(self.tiles_never_published)
        return tuple(sorted(t for t in self.live_weight if t not in artefacts))

    @property
    def tiles_missing(self) -> tuple[str, ...]:
        """Tiles that publish other years but not this one — the real gaps."""
        have = set(self.tiles_with_data)
        return tuple(t for t in self.tiles_considered if t not in have)

    @property
    def fraction_land_covered(self) -> float:
        """Share of live land whose tile publishes this year, weighted by live tiles.

        The denominator is the land under tiles that publish SOMETHING, so attribution
        artefacts neither inflate nor deflate it. Returns 1.0 when there is nothing to judge.
        """
        considered = self.tiles_considered
        total = sum(self.live_weight[t] for t in considered)
        if total == 0:
            return 1.0
        return sum(self.live_weight[t] for t in self.tiles_with_data) / total

    @property
    def is_empty(self) -> bool:
        """No live-land tile publishes this year, so there is nothing to ingest.

        Deliberately distinct from a partial gap: a cell losing one tile of forty has a hole in
        its mosaic, while a cell losing every tile would fail after provisioning.
        """
        return bool(self.tiles_considered) and not self.tiles_with_data

    @property
    def is_unattributable(self) -> bool:
        """Every live-land tile publishes nothing ever, so the attribution itself is suspect.

        Distinguished from :attr:`is_empty` because the remedy differs: an empty year is an
        upstream gap to skip, whereas this is more likely a tile-naming mismatch and wants
        looking at before anything is concluded about the source.
        """
        return bool(self.live_weight) and not self.tiles_considered


class NoPublishedOpticalError(RuntimeError):
    """The optical source publishes nothing for any tile carrying this cell's live land.

    Raised at preflight rather than surfacing later as a missing store, because the two read
    completely differently to whoever is on call: this names the source and the tiles, and
    costs seconds instead of a provisioned cluster.
    """


def live_mgrs_tiles(
    zone: str,
    *,
    land_mask_path: str,
    roi_bucket: str,
    get_credentials: object = None,
    s3_region: str | None = None,
) -> dict[str, int]:
    """MGRS tiles carrying ``zone``'s live land, mapped to how many live tiles each carries.

    Tile names are COMPUTED from the tile centre's latitude/longitude, not looked up in a
    footprint index. That is deliberate: the index is a per-deployment asset which is simply
    absent in some accounts, and the computation needs nothing but the coordinate.

    Tile CENTRES are used rather than footprints, which is adequate because a live 2048-px tile
    is 20.5 km against an MGRS square's 100 km. The weight is a proxy for land, not an area.

    **What this cannot see, and why it is the safe direction.** Sentinel-2 tiles are 110 km on a
    100 km grid, so they overlap and the square a point falls in is often not the square ESA
    publishes for that land. The computation returns only the canonical square, so a point whose
    square is unpublished but whose neighbour is will look unattributable. Those tiles publish no
    year at all, so the verdict excludes them (see :class:`OpticalCoverage`) and their land drops
    out of the denominator — which neither refuses valid work nor claims coverage we do not have.

    Args:
        zone: UTM zone label.
        land_mask_path: Coverage store holding one group per zone with ``tile_live_2048``.
        roi_bucket: Base URI for ROI storage — the per-zone grids live under it.
        get_credentials: Icechunk credential callback for the coverage store.
        s3_region: Region for the coverage store.
    """
    import mgrs

    from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group

    cov = open_store_as_zarr_group(land_mask_path, group=zone, get_credentials=get_credentials, region=s3_region)
    live = np.asarray(cast("zarr.Array", cov["tile_live_2048"]), dtype=bool)
    rows, cols = np.nonzero(live)
    if len(rows) == 0:
        return {}
    tile_px = int(cast("dict", cov.attrs)["tile_px"])

    # The zone's own affine, so a tile index becomes a real coordinate rather than an estimate.
    md = read_roi_metadata(f"{roi_bucket.rstrip('/')}/zarrs/zone_{zone}.zarr")
    affine = md.geobox.affine
    xs, ys = [], []
    for r, c in zip(rows, cols, strict=True):
        x, y = affine * ((c + 0.5) * tile_px, (r + 0.5) * tile_px)
        xs.append(x)
        ys.append(y)
    lons, lats = Transformer.from_crs(md.native_crs, "EPSG:4326", always_xy=True).transform(xs, ys)

    converter = mgrs.MGRS()
    weight: dict[str, int] = {}
    for lon, lat in zip(lons, lats, strict=True):
        # Precision 0 is the 100 km square, which is exactly a Sentinel-2 tile id.
        try:
            tile = converter.toMGRS(lat, lon, MGRSPrecision=0)
        except Exception:
            # Outside the MGRS domain (beyond 84N/80S) — no Sentinel-2 tile exists there, so
            # counting it would put land in the denominator that can never be imaged.
            logger.debug("Zone %s: live tile at %.4f, %.4f is outside the MGRS domain", zone, lat, lon)
            continue
        weight[tile] = weight.get(tile, 0) + 1
    logger.info("Zone %s: %d live tile(s) fall in %d MGRS square(s)", zone, len(rows), len(weight))
    return weight


def optical_coverage(
    zone: str,
    year: int,
    *,
    years_fn: YearsFn,
    land_mask_path: str,
    roi_bucket: str,
    get_credentials: object = None,
    s3_region: str | None = None,
) -> OpticalCoverage:
    """What the optical source publishes for ``(zone, year)``'s live land."""
    weight = live_mgrs_tiles(
        zone,
        land_mask_path=land_mask_path,
        roi_bucket=roi_bucket,
        get_credentials=get_credentials,
        s3_region=s3_region,
    )
    per_tile = years_fn(weight.keys()) if weight else {}
    never = tuple(sorted(t for t in weight if not per_tile.get(t)))
    with_data = tuple(sorted(t for t in weight if str(year) in per_tile.get(t, frozenset())))
    return OpticalCoverage(
        zone=zone,
        year=year,
        live_weight=weight,
        tiles_with_data=with_data,
        tiles_never_published=never,
    )


def assert_optical_is_published(coverage: OpticalCoverage) -> None:
    """Refuse a cell with no published optical imagery; warn about a partial or thin one.

    An all-ocean zone (no live tiles at all) is NOT refused — it has no land to image, the
    campaign screens it separately, and refusing it would conflate "nothing to do" with
    "something is missing".

    **Only the fully-empty case refuses.** A thin cell still produces a real, if partial,
    mosaic, and the threshold that would catch it sits in a range no measured zone occupies —
    so making it a refusal would be acting on a boundary nothing has tested. Thin cells warn
    loudly instead, which is what makes them findable in a fleet-wide log.
    """
    if coverage.is_unattributable:
        raise NoPublishedOpticalError(
            f"Cannot judge optical coverage for {coverage.zone} year {coverage.year}: all "
            f"{len(coverage.live_weight)} MGRS tile(s) derived for this zone's live land publish "
            f"nothing in ANY year ({', '.join(coverage.tiles_never_published[:8])}). That points "
            f"at the tile attribution rather than at the source — near a UTM zone boundary land "
            f"is carried by a tile of the adjacent zone — so check the mapping before concluding "
            f"anything about the imagery."
        )
    if coverage.is_empty:
        raise NoPublishedOpticalError(
            f"The optical source publishes no imagery for {coverage.zone} year {coverage.year}: "
            f"none of the {len(coverage.tiles_considered)} MGRS tile(s) carrying this zone's live "
            f"land has data for that year ({', '.join(coverage.tiles_considered[:8])}), though "
            f"they do publish other years. This is an upstream gap, not a failure of this run — "
            f"the satellite may have acquired it, but the collection does not publish it, so "
            f"there is nothing to ingest and no retry changes that. Skip this zone-year, or "
            f"source its imagery elsewhere."
        )
    fraction = coverage.fraction_land_covered
    if fraction < BARELY_COVERED_FRACTION:
        logger.warning(
            "BARELY COVERED: %s year %d has published optical over only %.0f%% of its live land "
            "(%d of %d tile(s) publish this year; missing %s). The ingest will proceed, but "
            "expect a sparse mosaic and possibly no committed dates at all.",
            coverage.zone,
            coverage.year,
            100 * fraction,
            len(coverage.tiles_with_data),
            len(coverage.tiles_considered),
            ", ".join(coverage.tiles_missing[:8]) + (" …" if len(coverage.tiles_missing) > 8 else ""),
        )
    elif coverage.tiles_missing:
        logger.warning(
            "Optical coverage HOLE in %s year %d: %d of %d live-land MGRS tile(s) publish nothing "
            "for this year (%s), covering %.0f%% of live land. The ingest proceeds and that land "
            "has no imagery, so the mosaic is complete only over the tiles that do publish.",
            coverage.zone,
            coverage.year,
            len(coverage.tiles_missing),
            len(coverage.tiles_considered),
            ", ".join(coverage.tiles_missing[:8]) + (" …" if len(coverage.tiles_missing) > 8 else ""),
            100 * fraction,
        )
    if coverage.tiles_never_published:
        logger.info(
            "%s: %d of %d derived MGRS tile(s) publish nothing in any year and were excluded as "
            "attribution artefacts (land near a zone boundary is carried by the adjacent zone's "
            "tile). Measured at ~19%% of derived squares, so this is expected, not a finding.",
            coverage.zone,
            len(coverage.tiles_never_published),
            len(coverage.live_weight),
        )
