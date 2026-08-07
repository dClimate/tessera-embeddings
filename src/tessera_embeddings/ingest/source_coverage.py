"""Whether the optical source publishes anything for a cell's LIVE land, before we spend on it.

A cell whose optical source publishes nothing for its window is not buildable: unlike a
radar gap — which has a defined optical-only answer — an optical gap has no fallback, and
without a preflight it surfaces only after the ingest has run and a GPU fleet is up,
as a coverage-gate failure or a missing reflectance store. This module answers the
question from catalogue metadata alone, in seconds, so a driver can refuse the cell
before provisioning anything.

**The verdict is three-valued, and only one value refuses.** A preflight that skips a
buildable cell silently loses campaign coverage, which is strictly worse than the wasted
cluster it exists to save — so "could not determine" and "confirmed absent" are distinct
values, never a boolean, and every failure of the preflight itself (catalogue error,
exhausted budget, unreadable mask) resolves to INCONCLUSIVE, which passes the cell
through to fail exactly where it does today.

**Why a miss is decisive and a hit is not.** The live tiles are cut into blocks and each
block's WGS84 envelope is asked one limit-1 question against the same catalogue the
ingest queries. Each envelope CONTAINS its block's live tiles and the blocks jointly
cover every live tile, so a clean miss on every block is a positive finding: no
catalogued item's footprint reaches any live land, and the ingest — whose own searches
are bounded by the ROI's live-tile envelope, a superset of these blocks — must load
nothing. A hit proves much less: an envelope is a superset of the land it stands for, so
an item intersecting it may still reach no live pixel. Hence any hit ends the sweep as
PRESENT, a provisional answer that leaves the fill's own gates the authority. The blocks
are sized near a source tile so an envelope is dominated by its own live land — a
whole-zone envelope is the degenerate case of this trade-off, mostly water and land the
zone does not own, which is why the sweep is per block rather than one box.

**Date convention.** Mosaic dates are SOLAR days; the catalogue is bounded in UTC. A
solar day lies within its UTC date ±1 day, so the probe range is the window's date range
padded one UTC day on each side (``solar_days.whole_window_range``) — the union of the
padded ranges the ingest itself queries. Refusing on an unpadded range would miss
granules keyed to the adjacent UTC day that the fill would legitimately use.

**Interaction with ``allow_partial_window``.** The refusal condition — zero catalogued
items reaching live land in the padded window — implies even the RELAXED coverage gate
("non-empty") must fail. The preflight therefore needs no knowledge of which gate mode
the caller runs: it refuses only cells neither mode could build.

The catalogue probe is injected (``probe``), so the verdict logic stays
provider-agnostic; the default probe targets the same provider/collection the S2 ingest
reads. It is built HERE rather than in :mod:`~tessera_embeddings.ingest.stac`, and that
placement is load-bearing: the ingest leg entry points seed the mosaic-content
fingerprint (``config.ingest``), which hashes everything they import — so adding even
inert code to a module inside that closure re-fingerprints every in-flight mosaic and
forces it to re-ingest on the next deploy. This module is outside the closure; only the
pieces that must not drift are shared by import (the provider registry and the
antimeridian split).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast, final

import numpy as np
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO

from tessera_embeddings.config import PROVIDERS
from tessera_embeddings.ingest._http import make_logging_retry
from tessera_embeddings.ingest.land_mask import live_tile_block_bboxes_wgs84
from tessera_embeddings.ingest.solar_days import whole_window_range
from tessera_embeddings.ingest.stac import split_antimeridian_bbox
from tessera_embeddings.storage import zone_grid
from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group

if TYPE_CHECKING:
    import icechunk
    import zarr

    from tessera_embeddings.config.time_windows import TimeWindow

logger = logging.getLogger(__name__)

#: ``probe(bbox_wgs84, start_date, end_date) -> True`` iff the catalogue holds any item
#: intersecting the box in the UTC date range. Must raise on failure rather than guess —
#: the caller maps a raised probe to INCONCLUSIVE, and a probe that returned ``False``
#: on error would convert an outage into a refusal.
ExistenceProbe = Callable[[tuple[float, float, float, float], str, str], bool]

#: Tile-grid block edge for the per-block sweep. Sized so a block is on the order of a
#: source tile: small enough that a block's envelope is dominated by its own live land
#: (an envelope much larger than a source tile re-admits the neighbour problem the sweep
#: exists to remove), large enough that a fully-live zone stays at tens of probes.
BLOCK_TILES = 8

#: Wall-clock budget for one cell's probes. The preflight must stay cheap relative to
#: what it saves; a cell it cannot settle within the budget passes through INCONCLUSIVE
#: rather than holding up triage — pass-through costs only what the run costs today.
PROBE_BUDGET_S = 45.0

#: Shorter-fused than the ingest's own catalogue hardening, deliberately: an exhausted
#: probe costs a pass-through (today's behaviour), so patience buys nothing here, while
#: the budget above bounds the whole sweep.
_PROBE_RETRY = make_logging_retry(
    "STAC preflight",
    total=3,
    backoff_factor=1,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]),
    respect_retry_after_header=True,
)
#: (connect, read) seconds. Without an explicit timeout a stalled connection blocks
#: past any budget check, which only runs between probes.
_PROBE_TIMEOUT = (10, 30)


class SourceFinding(Enum):
    """What the catalogue said about a cell's live land, with unequal weights.

    ``CONFIRMED_ABSENT`` is the only value a caller may act on by refusing work: it
    asserts a positive finding — every probe answered, and none found an item. The
    other two both mean "do not refuse": ``PRESENT`` is provisional (an item
    intersecting live land is necessary for a build, not sufficient), and
    ``INCONCLUSIVE`` means the preflight could not earn an answer.
    """

    CONFIRMED_ABSENT = "confirmed-absent"
    INCONCLUSIVE = "inconclusive"
    PRESENT = "present"


@final
@dataclass(frozen=True)
class OpticalPreflight:
    """One cell's preflight verdict, with the evidence a refusal must carry.

    Attributes:
        zone: UTM common name the verdict is about.
        finding: See :class:`SourceFinding`; only ``CONFIRMED_ABSENT`` may refuse.
        reason: Human-readable grounds — a refusal names what was probed and missed,
            so an operator can re-check it without re-deriving the geometry.
        probes: Catalogue probes performed (cost accounting, and evidence that a
            verdict was earned rather than defaulted).
    """

    zone: str
    finding: SourceFinding
    reason: str
    probes: int


def _make_default_probe(provider: str, collection: str) -> ExistenceProbe:
    """One catalogue client behind a reusable limit-1 existence probe.

    Existence is the cheapest question a catalogue answers — one search capped at a
    single item, no assets touched. The probe intersects the item's GEOMETRY with the
    box (the STAC bbox contract), so ``False`` means no item's footprint reaches the
    box; a server matching item bboxes instead can only widen the answer toward
    ``True``, never manufacture a ``False``. An antimeridian-crossing box
    (``west > east``) is split exactly as the ingest's own query path splits it, so the
    probe and the ingest cannot disagree about what a crossing box means. The client is
    opened lazily on the first probe and reused; the returned callable shares one HTTP
    session and is not thread-safe. Errors propagate — a failed probe is "could not
    determine", which is not an answer this function may turn into one.
    """
    provider_config = PROVIDERS[provider]
    collection_id = provider_config.collections[collection].collection_id
    client: Client | None = None

    def probe(bbox: tuple[float, float, float, float], start_date: str, end_date: str) -> bool:
        nonlocal client
        if client is None:
            client = Client.open(
                provider_config.catalog_url, stac_io=StacApiIO(max_retries=_PROBE_RETRY, timeout=_PROBE_TIMEOUT)
            )
        for sub_bbox in split_antimeridian_bbox(bbox):
            search = client.search(
                collections=[collection_id],
                datetime=f"{start_date}/{end_date}",
                bbox=sub_bbox,
                limit=1,
                max_items=1,
            )
            if next(search.items_as_dicts(), None) is not None:
                return True
        return False

    return probe


def preflight_optical_source(
    zone: str,
    window: TimeWindow,
    *,
    land_mask_path: str,
    provider: str = "earth-search",
    collection: str = "sentinel-2-l2a",
    block_tiles: int = BLOCK_TILES,
    budget_s: float = PROBE_BUDGET_S,
    probe: ExistenceProbe | None = None,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> OpticalPreflight:
    """Ask the optical catalogue whether ``zone``'s live land has anything in ``window``.

    Never raises: every failure mode of the preflight itself — an unreadable coverage
    bitmap, a catalogue outage, an exhausted budget — returns INCONCLUSIVE, because the
    preflight's one hard obligation is to refuse only on a positive finding of absence.
    Blocks are probed densest-first, so where the source publishes anything at all the
    sweep typically ends at its first probe. See the module docstring for why a miss is
    decisive and a hit is provisional.

    Args:
        zone: UTM common name (e.g. ``"31S"``).
        window: The cell's inference window; probed as its date range padded one UTC
            day per the solar-day convention.
        land_mask_path: Coverage repo holding the zone's ``tile_live_2048`` bitmap.
        provider: STAC provider key — must be the one the S2 ingest reads, since the
            decisiveness of a miss rests on probing the ingest's own catalogue.
        collection: Collection alias within the provider, same constraint.
        block_tiles: Block edge for the per-block sweep (see :data:`BLOCK_TILES`).
        budget_s: Wall-clock probe budget (see :data:`PROBE_BUDGET_S`).
        probe: Injectable catalogue probe; ``None`` builds the default from
            ``provider``/``collection``.
        get_credentials: Icechunk credential callback for the coverage repo.
        s3_region: Region of the coverage repo, when not the default.
        log: Logger for probe failures and the verdict trail.
    """
    log = log or logger
    try:
        spec = zone_grid.zone(zone)
        cov = open_store_as_zarr_group(land_mask_path, group=zone, get_credentials=get_credentials, region=s3_region)
        tile_live = np.asarray(cast("zarr.Array", cov["tile_live_2048"]), dtype=bool)
        if not tile_live.any():
            # All-ocean: there is no live land for the source to reach, so absence is
            # not a finding about the source. Callers settle these cells elsewhere.
            return OpticalPreflight(zone, SourceFinding.INCONCLUSIVE, "no live tiles — nothing to preflight", 0)
        rng = whole_window_range(*window.to_date_range())
        blocks = live_tile_block_bboxes_wgs84(spec, tile_live, block_tiles=block_tiles)
        # Densest block first: where anything is published it shows up soonest over
        # the most land, so the common case settles in one probe.
        blocks.sort(key=lambda entry: entry[1], reverse=True)
        if probe is None:
            probe = _make_default_probe(provider, collection)
    except Exception as exc:
        log.warning("Optical preflight for %s could not be set up — passing the cell through", zone, exc_info=True)
        return OpticalPreflight(zone, SourceFinding.INCONCLUSIVE, f"preflight setup failed: {exc!r}", 0)

    deadline = time.monotonic() + budget_s
    probes = 0
    window_label = f"{rng.query_start}..{rng.query_end}"

    failed_blocks = 0
    for bbox, _weight in blocks:
        if time.monotonic() > deadline:
            return OpticalPreflight(
                zone,
                SourceFinding.INCONCLUSIVE,
                f"probe budget ({budget_s:.0f}s) exhausted after {probes} probe(s) with no verdict",
                probes,
            )
        try:
            probes += 1
            if probe(bbox, rng.query_start, rng.query_end):
                return OpticalPreflight(
                    zone,
                    SourceFinding.PRESENT,
                    f"catalogue has items intersecting a live-land block for {window_label} "
                    f"(provisional — the fill's own coverage gates remain the authority)",
                    probes,
                )
        except Exception:
            failed_blocks += 1
            log.warning("Optical preflight for %s: block probe failed", zone, exc_info=True)

    if failed_blocks:
        # Some blocks went unanswered and none hit: absence over live land is possible
        # but was not POSITIVELY established, and that distinction is the whole gate.
        return OpticalPreflight(
            zone,
            SourceFinding.INCONCLUSIVE,
            f"{failed_blocks} of {len(blocks)} block probe(s) failed and none found items",
            probes,
        )
    return OpticalPreflight(
        zone,
        SourceFinding.CONFIRMED_ABSENT,
        f"catalogue has no items intersecting any of the zone's {len(blocks)} live-land block(s) for {window_label}",
        probes,
    )
