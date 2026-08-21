"""Whether one reflectance source is owed the Sentinel-2 BOA offset correction.

ESA's baseline 04.00 (January 2022) added a fixed ``+1000`` to surface reflectance so that
negative values — routine over water and deep shadow — survive an unsigned type. Element 84
subtracts it from its own COGs; ESA's originals carry it. Which of those served a given band is a
property of **where that band's object lives**, so the question is asked of a single asset rather
than of an item, a date, or a collection.

**One atom, three callers.** :func:`source_decision` is the only place the question is answered.
The load path builds its per-source correction table from it, the load path's refusal guard reads
it, and duplicate selection asks it of a candidate copy. Those three used to be separate
derivations of an item-level answer, and they drifted: a spare that the correction path would
refuse was offered to the fallback ladder, which recovers from a read failure and not from a
refusal, so a single unreadable object aborted a whole ingest. Keeping the atom here — and keeping
it free of any notion of *item* — is what makes that class of drift unrepresentable rather than
merely tested for.

Deliberately imports no odc. The GDAL environment has to be configured before ``odc.stac`` is
imported (see :mod:`tessera_embeddings.config.environment` and the import ordering in
:mod:`tessera_embeddings.ingest.stac`), so the module holding the decision must not be the module
that pulls odc in. Callers hand in a bucket and a baseline; resolving an asset key to a bucket is
:mod:`tessera_embeddings.ingest.asset_locations`' job and resolving a band name to an asset is
odc's.
"""

from __future__ import annotations

import enum

from tessera_embeddings.ingest.asset_locations import (
    HARMONISED_ASSET_BUCKETS,
    UNHARMONISED_ASSET_BUCKETS,
    Harmonisation,
)


class OffsetDecision(enum.Enum):
    """What is owed to one reflectance source.

    Three states rather than a boolean, because "we cannot tell" needs a response that neither
    answer gives. Correcting already-harmonised pixels and leaving raw ones alone are both wrong
    by exactly the offset and both silent, so a source nobody can classify must not be handled as
    though it had been classified.
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
    known_harmonisation: Harmonisation | None = None,
) -> OffsetDecision:
    """What is owed to the source served from ``bucket`` by an item declaring ``baseline``.

    Asked per ASSET, which is what lets an item whose bands straddle two producers be corrected
    band by band. Asked of the whole item instead, such a copy has no single correct answer and
    refuses — and because the correction used to be decided per solar day, that refusal took every
    other tile of the day with it.

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
            :func:`~tessera_embeddings.ingest.stac.collection_harmonisation`.

    Returns:
        The decision for this one source.
    """
    # The collection's answer replaces the asset's rather than supplementing it. Where a provider
    # serves its bands under native asset keys, the bucket is unclassifiable by construction, so
    # consulting it would turn a collection the configuration fully describes into one that refuses
    # every date.
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

    # Below the threshold no producer changes a pixel, so an unclassified bucket is harmless there.
    # This gate is why an unrecognised mirror of PRE-04.00 data costs nothing: the correction is
    # not owed either way, and refusing would lose real imagery for an ambiguity with no
    # consequence.
    if baseline is not None and baseline < threshold:
        return OffsetDecision.EXEMPT

    # At or above the threshold — or with no readable baseline at all — the producer decides a
    # pixel, so not knowing it is a refusal.
    if producer is None:
        return OffsetDecision.UNDECIDABLE

    # Unharmonised, and the baseline is what says whether the offset was ever added. A missing or
    # malformed value parses as nothing rather than as zero: reading it as zero exempts a date
    # whose pixels may carry the offset, and correcting it takes the offset off pre-04.00 pixels
    # that never had it. Both are wrong by the same amount in opposite directions, and both silent.
    if baseline is None:
        return OffsetDecision.UNDECIDABLE

    return OffsetDecision.OWED
