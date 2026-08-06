"""What the public Sentinel-2 COG bucket actually publishes, per MGRS tile and year.

The catalogue and the bucket can disagree with what a reader assumes: a tile may be absent
for whole years. Zone 31S has its only land under MGRS tile 31MGU, which publishes nothing
before 2022 — so an ingest of 2021 built no reflectance store and the run died several
minutes in, reporting a missing repository rather than a missing source.

**Answering "is anything published here" is a LISTING, not a search.** One
``list_objects_v2`` at a tile's prefix returns every year that tile holds, in about 120 ms,
against a public bucket needing no credentials. That is cheap enough to run as a preflight
before provisioning anything, which is the whole point: the alternative is discovering it
after a fleet has been paid for.

This module knows only about the bucket. Deciding which tiles a zone's live land needs, and
what to do about a gap, belongs to :mod:`tessera_embeddings.ingest.source_coverage`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore import UNSIGNED
from botocore.config import Config

logger = logging.getLogger(__name__)

#: The public bucket and prefix Element 84 publishes Sentinel-2 L2A COGs under.
COGS_BUCKET = "sentinel-cogs"
COGS_PREFIX = "sentinel-s2-l2a-cogs"

#: An MGRS tile id as the tile index spells it: zone number, latitude band, 100 km square.
#: The band letters exclude I and O by MGRS convention, so accepting them would mean
#: accepting an id that cannot exist.
_TILE_RE = re.compile(r"^(?P<zone>\d{1,2})(?P<band>[C-HJ-NP-X])(?P<square>[A-Z]{2})$")


def tile_prefix(tile: str) -> str:
    """The bucket prefix holding one MGRS tile's years.

    **The zone number is UNPADDED**, and that is not cosmetic: the bucket has ``1/C/CV/`` and
    no ``01/C/CV/`` at all, so a zero-padded lookup returns an empty listing for every tile in
    zones 1 through 9. It would not error — it would report those tiles as publishing nothing,
    which is exactly the answer that makes a preflight refuse valid work.
    """
    match = _TILE_RE.match(tile.removeprefix("MGRS-").strip().upper())
    if match is None:
        raise ValueError(
            f"Not an MGRS tile id: {tile!r}. Expected e.g. '14TPK' or 'MGRS-14TPK' — "
            "one or two digits, a latitude band letter (I and O excluded), then two letters."
        )
    # int() then str() rather than lstrip("0"), so a padded input is normalised rather than
    # turned into an empty string by a hypothetical "00".
    return f"{COGS_PREFIX}/{int(match['zone'])}/{match['band']}/{match['square']}"


def _client() -> object:
    """An UNSIGNED S3 client. The bucket is public, and signing it with our own role would
    make the answer depend on credentials that have nothing to do with what is published.
    """
    return boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))


def published_years(tiles: Iterable[str], *, max_workers: int = 16, client: object = None) -> dict[str, frozenset[str]]:
    """Which years each MGRS tile publishes, as ``{tile: {"2021", "2022", ...}}``.

    One listing per tile, run concurrently. A tile that publishes nothing maps to an empty
    set — the answer the caller acts on, so it must be distinguishable from an error, which
    raises instead.

    Args:
        tiles: MGRS tile ids, with or without the ``MGRS-`` prefix. Deduplicated internally.
        max_workers: Concurrent listings. Modest by default: the win here is already three
            orders of magnitude over a per-year catalogue search, so there is nothing to buy
            by hammering a shared public bucket.
        client: Injectable S3 client, for tests. Must be safe to share across threads —
            botocore clients are.
    """
    s3 = client if client is not None else _client()
    wanted = sorted({t.removeprefix("MGRS-").strip().upper() for t in tiles})
    if not wanted:
        return {}

    def years_for(tile: str) -> tuple[str, frozenset[str]]:
        prefix = f"{tile_prefix(tile)}/"
        response = s3.list_objects_v2(Bucket=COGS_BUCKET, Prefix=prefix, Delimiter="/")  # type: ignore[attr-defined]
        # Years arrive as CommonPrefixes; a tile with no data has none at all.
        found = {p["Prefix"].rstrip("/").rsplit("/", 1)[-1] for p in response.get("CommonPrefixes", ())}
        # Keep only what looks like a year. The bucket has at least one stray prefix at this
        # level, and letting it through would make a tile appear to publish a year it cannot.
        return tile, frozenset(y for y in found if len(y) == 4 and y.isdigit())

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = dict(pool.map(years_for, wanted))

    empty = [t for t, years in results.items() if not years]
    if empty:
        logger.info(
            "%d of %d MGRS tile(s) publish nothing at all: %s",
            len(empty),
            len(results),
            ", ".join(empty[:10]) + (" …" if len(empty) > 10 else ""),
        )
    return results
