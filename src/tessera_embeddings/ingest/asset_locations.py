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

import enum
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

#: Everything an S2 ingest actually reads: the configured bands plus the scene classification
#: layer. Used for LOCALITY, because every one of these is fetched and so every one costs egress.
#: Asked of these and not of every asset, because a real Element 84 item carries the original
#: JP2s as extras alongside its COG bands — judging all of them answers about assets that are
#: never fetched at all.
READ_ASSET_KEYS: tuple[str, ...] = (*S2_L2A_BANDS, "scl")

#: The subset the BOA offset is applied to. Used for HARMONISATION, and deliberately EXCLUDES
#: ``scl``: the scene classification layer is categorical and never baseline-corrected —
#: subtracting 1000 from a class label is meaningless — so which producer served it cannot make
#: the reflectance correction ambiguous. Including it let an item whose reflectance is uniformly
#: harmonised be classified MIXED because of a layer that is never touched, and a MIXED item can
#: refuse an entire date.
REFLECTANCE_ASSET_KEYS: tuple[str, ...] = tuple(S2_L2A_BANDS)


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


class Harmonisation(enum.Enum):
    """Whether an item's reflectance already has the baseline-04.00 offset removed.

    Three states and not a boolean, because ``MIXED`` needs a different response from either
    answer rather than a default. The correction is applied per DATE to every band at once, so
    an item whose bands straddle two producers has no correct date-wide answer: exempting it
    leaves the raw band 1000 high, correcting it drops 1000 from every harmonised band. A
    boolean forces one of those silently.
    """

    HARMONISED = "harmonised"
    RAW = "raw"
    MIXED = "mixed"


def item_harmonisation(
    item: Any,  # noqa: ANN401 — any STAC-like item
    buckets: frozenset[str] = HARMONISED_ASSET_BUCKETS,
    keys: tuple[str, ...] = REFLECTANCE_ASSET_KEYS,
) -> Harmonisation:
    """Which producer served the REFLECTANCE bands — the only ones the offset touches.

    Judged over the reflectance bands alone, for two separate reasons. A real Element 84 item
    carries the original JP2s as extra assets beside its COG bands, so judging *every* asset
    reports ``MIXED`` for an item that is wholly harmonised where it matters. And ``scl`` is
    excluded even though it IS read: it is categorical and never corrected, so its producer
    cannot make the reflectance decision ambiguous.

    An item exposing none of the reflectance bands is ``RAW``, and so is one served from a bucket
    nobody has listed. **Absence of evidence must not buy an exemption from a correction**, and
    of the two mistakes only one is discoverable: a doubled correction shifts values by a
    visible 1000, while a skipped one leaves plausible pixels that are quietly wrong.
    """
    found = read_asset_buckets(item, keys)
    if not found:
        return Harmonisation.RAW
    harmonised = [bucket in buckets for bucket in found]
    if all(harmonised):
        return Harmonisation.HARMONISED
    if any(harmonised):
        return Harmonisation.MIXED
    return Harmonisation.RAW
