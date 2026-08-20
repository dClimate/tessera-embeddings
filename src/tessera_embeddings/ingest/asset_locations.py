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

import dataclasses
import enum
import logging
import urllib.parse
from typing import Any

from tessera_embeddings.config.providers import S2_L2A_BANDS

logger = logging.getLogger(__name__)

#: Buckets in the same region as our compute — reads against these reach the S3 gateway
#: endpoint. An unrecognised bucket is REMOTE rather than assumed local: under-claiming
#: locality costs a lost tiebreak, over-claiming it routes bulk reads through NAT believing
#: they are free.
PREFERRED_ASSET_BUCKETS: frozenset[str] = frozenset({"sentinel-cogs", "e84-earth-search-sentinel-data"})

#: The ESA archive that a Sentinel-2 catalogue may point at INSTEAD of a harmonised mirror.
#: Named because reaching it is the surprising route: Earth Search was believed to serve only
#: its own harmonised COGs, so a correction owed on data from here is the case worth announcing.
#: Other unharmonised producers (Planetary Computer, on Azure) are corrected as a matter of
#: course and need no announcement.
RAW_ARCHIVE_BUCKETS: frozenset[str] = frozenset({"sentinel-s2-l2a"})

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
    try:
        parsed = urllib.parse.urlparse(href)
    except ValueError:
        # A malformed authority raises. Unrecognised means remote, so one bad href cannot
        # abort a whole selection.
        logger.debug("Unparseable asset href %r", href)
        return None
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


@dataclasses.dataclass(frozen=True)
class AssetSources:
    """Where every REQUESTED asset of an item lives, including the ones that are not there.

    Carries the missing keys rather than dropping them, so no caller has to re-derive whether the
    set it asked for was complete.
    """

    #: The bucket serving each requested key that resolved. ``None`` for an href that is not a
    #: recognised S3 URL, which is a distinct answer from a key that is absent.
    buckets: dict[str, str | None]
    #: The requested keys with no resolvable href.
    missing: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Whether every requested key resolved."""
        return not self.missing

    @property
    def empty(self) -> bool:
        """Whether nothing at all resolved."""
        return not self.buckets

    def all_in(self, buckets: frozenset[str]) -> bool:
        """Whether the complete requested set is served from ``buckets``.

        False for an incomplete set: a claim about every asset cannot be made from a subset.
        """
        return self.complete and all(bucket in buckets for bucket in self.buckets.values())

    def any_in(self, buckets: frozenset[str]) -> bool:
        """Whether any resolved key is served from ``buckets``."""
        return any(bucket in buckets for bucket in self.buckets.values())


def read_asset_sources(item: Any, keys: tuple[str, ...] = READ_ASSET_KEYS) -> AssetSources:  # noqa: ANN401
    """Resolve every key in ``keys`` against the item's assets, reporting what is missing."""
    assets = getattr(item, "assets", None) or {}
    found: dict[str, str | None] = {}
    missing: list[str] = []
    for key in keys:
        asset = assets.get(key)
        href = asset_href(asset) if asset is not None else None
        if href is None:
            missing.append(key)
        else:
            found[key] = asset_bucket(href)
    return AssetSources(buckets=found, missing=tuple(missing))


def read_set_is_complete(item: Any, keys: tuple[str, ...] = READ_ASSET_KEYS) -> bool:  # noqa: ANN401
    """Whether every asset this ingest would READ resolves to an href.

    Answers "can this copy be read at all", where :func:`item_is_in_preferred_location` answers
    "from where". Ranking needs the weaker claim separately, so that a copy which cannot be read
    loses to one that can even when locality ties.

    Items carrying none of :data:`READ_ASSET_KEYS` are uniformly incomplete, so the term ties
    across such a set and changes no ordering.
    """
    return read_asset_sources(item, keys).complete


def item_is_in_preferred_location(
    item: Any,  # noqa: ANN401 — any STAC-like item
    buckets: frozenset[str] = PREFERRED_ASSET_BUCKETS,
    keys: tuple[str, ...] = READ_ASSET_KEYS,
) -> bool:
    """Whether every asset this ingest would READ sits in a preferred bucket.

    All of ``keys``, not any. An item whose bands straddle two buckets cannot deliver the locality
    being claimed, and an incomplete item cannot deliver the read at all. An item exposing none of
    them is remote: absence of evidence is not locality.
    """
    return read_asset_sources(item, keys).all_in(buckets)


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
    UNKNOWN = "unknown"


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

    An item served from a bucket nobody has listed is ``RAW``: **absence of evidence must not buy
    an exemption from a correction**, and of the two mistakes only one is discoverable — a doubled
    correction shifts values by a visible 1000, while a skipped one leaves plausible pixels that
    are quietly wrong.

    An item that does not expose EVERY reflectance band under the configured names is ``UNKNOWN``
    rather than ``RAW``. Nothing here can resolve the alias table mapping a band name to an asset
    key, so a band absent under the requested name may still be served under a native one, and
    calling such an item raw would subtract the offset from pixels that may already be harmonised.
    The caller refuses on ``UNKNOWN`` rather than guessing.
    """
    sources = read_asset_sources(item, keys)
    if not sources.complete:
        return Harmonisation.UNKNOWN
    if sources.all_in(buckets):
        return Harmonisation.HARMONISED
    if sources.any_in(buckets):
        return Harmonisation.MIXED
    return Harmonisation.RAW


def item_is_from_raw_archive(
    item: Any,  # noqa: ANN401 — any STAC-like item
    buckets: frozenset[str] = RAW_ARCHIVE_BUCKETS,
) -> bool:
    """Whether this item's reflectance comes from the ESA archive named in ``buckets``.

    Narrower than "not harmonised", and deliberately: every Planetary Computer item is
    unharmonised too, and correcting those is routine rather than notable. Only the archive a
    harmonised-COG catalogue has started pointing at is a surprise worth a warning.
    """
    return read_asset_sources(item, REFLECTANCE_ASSET_KEYS).any_in(buckets)
