"""Where an item's assets live, and what that implies.

Two questions are asked of an asset's location, and **they are different claims that happen
to have the same answer today**. Keeping them apart is the point of this module:

* **Is the read cheap?** A property of the bucket's REGION. In-region reads go through the
  VPC's S3 gateway endpoint; anything else egresses through NAT, which is metered, slower
  and a shared ceiling.
* **Has the reflectance offset already been removed?** A property of the PRODUCER. Element 84
  harmonises its COGs — it subtracts the post-baseline-04.00 offset for you — while ESA's
  originals carry it.

Element 84's COG buckets satisfy both, so one constant would work now and would be wrong the
first time anyone mirrors *unharmonised* data in region: the mirror would inherit "already
corrected" from a fact about its location. Two constants make that impossible to conflate.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from tessera_embeddings.config.providers import S2_L2A_BANDS

#: Buckets in the same region as our compute — reads against these reach the S3 gateway
#: endpoint. An unrecognised bucket is REMOTE rather than assumed local: under-claiming
#: locality costs a lost tiebreak, over-claiming it routes bulk reads through NAT believing
#: they are free.
PREFERRED_ASSET_BUCKETS: frozenset[str] = frozenset({"sentinel-cogs", "e84-earth-search-sentinel-data"})

#: Buckets whose Sentinel-2 surface reflectance already has the post-baseline-04.00 offset
#: subtracted. An unrecognised bucket is treated as NOT harmonised, so the correction is
#: applied: over-correcting harmonised data and under-correcting raw data are both wrong, but
#: only one of them is discoverable — a doubled correction shifts values by a visible 1000,
#: while a skipped one leaves plausible-looking pixels that are silently 1000 too high.
HARMONISED_ASSET_BUCKETS: frozenset[str] = frozenset({"sentinel-cogs", "e84-earth-search-sentinel-data"})

#: The asset keys an S2 ingest actually reads: the configured bands plus the scene
#: classification layer. Every question here is asked of THESE and not of every asset, because
#: a real Element 84 item carries the original JP2s as extras alongside its COG bands — so
#: judging all of them answers about assets that are never fetched.
READ_ASSET_KEYS: tuple[str, ...] = (*S2_L2A_BANDS, "scl")


def asset_href(asset: Any) -> str | None:  # noqa: ANN401 — pystac Asset or a plain dict
    """An asset's href, whether it arrives as a pystac ``Asset`` or a raw dict."""
    href = getattr(asset, "href", None)
    if href is None and isinstance(asset, dict):
        href = asset.get("href")
    return href if isinstance(href, str) and href else None


def asset_bucket(href: str) -> str | None:
    """The S3 bucket an href addresses, or ``None`` if it does not address one.

    Parsed rather than substring-matched, because a substring test answers yes to things that
    are not the bucket: ``s3://sentinel-cogs-backup/...`` and
    ``https://elsewhere.example/sentinel-cogs/...`` both contain a known bucket's name while
    being somewhere else entirely. Returning ``None`` for anything unrecognised is what keeps
    every unlisted-is-the-cautious-answer rule in this module honest.

    Handles the three forms the catalogues emit: ``s3://bucket/key``, virtual-hosted
    ``https://bucket.s3.<region>.amazonaws.com/key``, and path-style
    ``https://s3.<region>.amazonaws.com/bucket/key``.
    """
    parsed = urllib.parse.urlparse(href)
    if parsed.scheme == "s3":
        return parsed.netloc or None
    host = parsed.netloc.split("@")[-1].split(":")[0].lower()
    if not host.endswith(".amazonaws.com"):
        return None
    labels = host.split(".")
    if len(labels) > 3 and labels[1] == "s3":
        return labels[0] or None
    if labels[0] == "s3":
        first_segment = parsed.path.lstrip("/").split("/", 1)[0]
        return first_segment or None
    return None


def read_asset_buckets(item: Any, keys: tuple[str, ...] = READ_ASSET_KEYS) -> list[str | None]:  # noqa: ANN401
    """The bucket of each asset this ingest would READ. ``None`` for anything unrecognised."""
    assets = getattr(item, "assets", None) or {}
    return [
        asset_bucket(href) for key in keys if (asset := assets.get(key)) is not None and (href := asset_href(asset))
    ]


def item_is_in_preferred_location(
    item: Any,  # noqa: ANN401 — any STAC-like item
    buckets: frozenset[str] = PREFERRED_ASSET_BUCKETS,
) -> bool:
    """Whether every asset this ingest would read sits in a preferred bucket.

    All of the read set, not any: an item whose bands straddle two buckets cannot deliver the
    locality being claimed. An item exposing none of them is remote — absence of evidence is
    not locality.
    """
    found = read_asset_buckets(item)
    return bool(found) and all(bucket in buckets for bucket in found)


def item_is_pre_harmonised(
    item: Any,  # noqa: ANN401 — any STAC-like item
    buckets: frozenset[str] = HARMONISED_ASSET_BUCKETS,
) -> bool:
    """Whether this item's reflectance already has the baseline-04.00 offset removed.

    All of the read set, not any, and for a sharper reason than locality: the correction is
    applied per DATE to every band at once, so a partly-harmonised item has no single right
    answer and the safe reading is "not harmonised" — under-correcting one band is a smaller
    error than over-correcting the rest.

    An item exposing none of the read bands is likewise not harmonised. Absence of evidence
    must not buy an exemption from a correction.
    """
    found = read_asset_buckets(item)
    return bool(found) and all(bucket in buckets for bucket in found)
