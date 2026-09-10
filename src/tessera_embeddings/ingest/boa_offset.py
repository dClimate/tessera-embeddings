"""Whether one reflectance source is owed the Sentinel-2 BOA offset correction.

ESA's baseline 04.00 (January 2022) added a fixed ``+1000`` to surface reflectance so that
negative values — routine over water and deep shadow — survive an unsigned type. Element 84
subtracts it from its own COGs; ESA's originals carry it. Which of those served a given band is a
property of **where that band's object lives**, so the question is asked of a single asset rather
than of an item, a date, or a collection.

**One atom.** :func:`source_decision` is the only place the question is answered, and it knows
nothing of *items*. Its production caller is the load path's metadata parser, which stamps the
answer onto each source as odc opens it; duplicate selection reaches the same evidence through
:mod:`asset_locations` when it ranks a candidate copy. Two item-level derivations of this drifted
once and offered the fallback ladder a spare the correction path then refused — the ladder recovers
from a read failure but not from a refusal, so a single unreadable object aborted a whole ingest.

Deliberately imports no odc: the GDAL environment must be configured before ``odc.stac`` is
imported (see :mod:`tessera_embeddings.config.environment` and the import ordering in
:mod:`tessera_embeddings.ingest.stac`). Callers hand in a bucket and a baseline; resolving an asset
key to a bucket is :mod:`tessera_embeddings.ingest.asset_locations`' job and resolving a band name
to an asset is odc's.
"""

from __future__ import annotations

import enum

from tessera_embeddings.ingest.asset_locations import (
    HARMONISED_ASSET_BUCKETS,
    UNHARMONISED_ASSET_BUCKETS,
    Harmonisation,
    SettledProducer,
)


class OffsetDecision(enum.Enum):
    """What is owed to one reflectance source.

    Three states rather than a boolean: correcting already-harmonised pixels and leaving raw ones
    alone are both wrong by exactly the offset and both silent, so a source nobody can classify
    must not be handled as though it had been.
    """

    #: The offset is present and must be removed.
    OWED = "owed"
    #: Nothing is owed — either the producer already removed it, or the baseline predates it.
    EXEMPT = "exempt"
    #: Neither answer can be justified from the evidence. The caller refuses rather than guessing.
    UNDECIDABLE = "undecidable"


def source_decision(
    bucket: str | None,
    baseline: int | None,
    threshold: int,
    known_harmonisation: SettledProducer | None = None,
) -> OffsetDecision:
    """What is owed to the source served from ``bucket`` by an item declaring ``baseline``.

    Asked per ASSET, which is what lets an item whose bands straddle two producers be corrected
    band by band. Asked of the whole item, such a copy has no single correct answer and refuses —
    and a refusal decided per solar day takes every other tile of that day with it.

    Args:
        bucket: the S3 bucket serving this source, from
            :func:`~tessera_embeddings.ingest.asset_locations.asset_bucket`. ``None`` for an href
            that addresses no recognised bucket, which is a *distinct* answer from an absent key
            and is treated the same as an unclassified one: neither is evidence of a producer.
        baseline: the item's declared processing baseline as an integer hundredth, from
            :func:`~tessera_embeddings.ingest.item_baselines.processing_baseline`. ``None`` when it
            declares none or declares something unreadable.
        threshold: baselines at or above this carry the offset (``400``, i.e. ESA's ``04.00``).
        known_harmonisation: the producer for EVERY source, where the collection's configuration
            settles it and the assets have nothing to add. Planetary Computer serves the whole
            archive unharmonised and keys its assets natively, so ``bucket`` there describes a
            location nobody has classified and must not be consulted — see
            :func:`~tessera_embeddings.ingest.stac.collection_harmonisation`. Typed
            :data:`~tessera_embeddings.ingest.asset_locations.SettledProducer`, so ``MIXED`` and
            ``UNKNOWN`` cannot be passed: an answer naming no producer cannot decide a correction,
            and making it unrepresentable is cheaper than detecting it.

    Returns:
        The decision for this one source.
    """
    # The collection's answer REPLACES the asset's rather than supplementing it: where a provider
    # serves bands under native asset keys the bucket is unclassifiable by construction, so
    # consulting it would make a fully-described collection refuse every date.
    if known_harmonisation is not None:
        producer: Harmonisation | None = known_harmonisation
    elif bucket in HARMONISED_ASSET_BUCKETS:
        producer = Harmonisation.HARMONISED
    elif bucket in UNHARMONISED_ASSET_BUCKETS:
        producer = Harmonisation.RAW
    else:
        # Unlisted is UNDETERMINED, never RAW. Membership of the unharmonised set is what makes a
        # correction safe to apply; inferring it from "not in the harmonised set" would correct the
        # first mirror anyone stands up, silently and by exactly the offset.
        producer = None

    # Harmonised pixels are owed nothing whatever the baseline says, so an unreadable baseline is
    # harmless here. Ordered before the baseline checks for that reason: penalising such a copy on
    # a baseline it does not need would hand its tile-date to an older raw reprocessing.
    if producer is Harmonisation.HARMONISED:
        return OffsetDecision.EXEMPT

    # Below the threshold no producer changes a pixel, so an unclassified bucket is harmless: this
    # gate is why an unrecognised mirror of PRE-04.00 data costs nothing, where refusing would lose
    # real imagery over an ambiguity with no consequence.
    if baseline is not None and baseline < threshold:
        return OffsetDecision.EXEMPT

    # At or above the threshold — or with no readable baseline at all — the producer decides a
    # pixel, so not knowing it is a refusal.
    if producer is None:
        return OffsetDecision.UNDECIDABLE

    # Unharmonised, and the baseline is what says whether the offset was ever added. A missing or
    # malformed value parses as nothing rather than as zero: read as zero it exempts a date whose
    # pixels may carry the offset; corrected anyway it strips the offset from pre-04.00 pixels that
    # never had it. Both are wrong by the same amount in opposite directions, and both silent.
    if baseline is None:
        return OffsetDecision.UNDECIDABLE

    # Unharmonised and at or above the threshold: the offset is there and comes off. Nothing else
    # reaches this line — a harmonised producer returned above, an unclassified one refused, and
    # `SettledProducer` stops an answer naming no producer from arriving at all.
    return OffsetDecision.OWED
