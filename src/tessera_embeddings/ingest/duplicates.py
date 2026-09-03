"""Choosing between duplicate catalogue items for one tile and one solar day.

Element 84's Sentinel-2 catalogue holds **more than one item per tile-date** whenever a granule
has been reprocessed: same tile, same acquisition, different processing baseline, distinguished by
``s2:sequence``. It is common — a single MGRS tile can have dozens of duplicated dates in a season.

**The loader fuses them.** ``odc.stac.load`` groups by solar day and mosaics every item in the
group, so both copies are read and one unreadable copy fails the whole date. Choosing one copy up
front is what makes a fallback possible at all: while both are in the group there is nothing to
fall back *to*. Reading two copies of one photograph is also pure waste, and costs the egress of
whichever copy sits in another region. (The reflectance offset is removed per IMAGE as it is read,
before anything is fused, so a date mixing two baselines is not mis-corrected.)

**The order lives in one place.** :func:`_preference_key` states it term by term, with the reason
for each position; add a term there and nowhere else. Two invariants govern where a new term may
sit, and neither is recoverable from the list itself:

**The losing copy is a fallback, never a default.** Among copies that are all usable, a newer
reprocessing is the better data in the overwhelming majority of cases; corruption is rare and is
not a property of being newer — sampling duplicate pairs chosen independently of failure found
every copy intact. So callers step down only on a demonstrated read failure, never pre-emptively.

**Nothing below "will this work at all" may buy its preference with an older reprocessing.** Only
the terms deciding whether a copy can be read, and whether its correction can be decided, outrank
the baseline. Everything under it is either a claim about the PIXEL — an already-harmonised copy is
floored by its producer before resampling where one we correct is floored after, and a copy owing
nothing cannot be wrong by the offset at all — or a claim about COST, since in-region assets reach
the VPC's S3 gateway endpoint while anything else egresses through NAT. Neither may buy what it
wants with an older pixel, so both sit below the baseline, the pixel claim above the cost one.
"""

from __future__ import annotations

import collections
import datetime
import functools
import itertools
import logging
import re
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any, assert_never

from tessera_embeddings.config.satellites import S2_BASELINE_THRESHOLD
from tessera_embeddings.ingest.asset_locations import (
    READ_ASSET_KEYS,
    Harmonisation,
    item_harmonisation,
    item_is_in_preferred_location,
    read_set_is_complete,
)
from tessera_embeddings.ingest.item_baselines import processing_baseline as item_processing_baseline
from tessera_embeddings.ingest.solar_days import solar_day_of

logger = logging.getLogger(__name__)

#: Trailing ``_<sequence>_L2A`` of an Element 84 item id, e.g. ``S2B_34WFA_20210908_1_L2A``.
_ID_SEQUENCE_RE = re.compile(r"_([0-9]+)_[A-Z0-9]+$")

#: A DECLARED sequence: ASCII digits, nothing else. Matched rather than coerced — see
#: `item_baselines` for why validating the shape beats validating a parsed number.
_SEQUENCE_VALUE_RE = re.compile(r"[0-9]+")

#: The MGRS tile in an Element 84 item id — the second underscore-separated field.
_ID_TILE_RE = re.compile(r"^[A-Z0-9]+_([0-9]{1,2}[A-Z]{3})_")

#: A bare MGRS grid square: UTM zone, latitude band, 100 km square. Matched rather than
#: accepted as any string, because the property it reads is only ever this.
_MGRS_SQUARE_RE = re.compile(r"[0-9]{1,2}[A-Z]{3}")


def item_tile(item: Any) -> str | None:  # noqa: ANN401 — any STAC-like item
    """The MGRS tile an item covers, or ``None`` if it cannot be determined.

    Three sources, in order: ``grid:code`` as Earth Search names it, ``s2:mgrs_tile`` as
    Planetary Computer names it, then the item id. All three are returned in one canonical
    form, so nothing downstream has to know which one answered. Planetary Computer needs its own
    property rather than the id fallback — its ids carry the tile as ``_T33TWM_`` in a later
    field, which the Element 84 pattern does not match, so without it every one of its items is
    unkeyable and duplicate selection a no-op for the whole provider.

    ``None`` means "do not treat this item as having duplicates", which is the safe answer: a
    wrong tile key would either merge two different tiles (dropping real imagery) or split one
    tile's duplicates (defeating the point).
    """
    code = item.properties.get("grid:code")
    if isinstance(code, str) and code:
        return code
    square = item.properties.get("s2:mgrs_tile")
    if isinstance(square, str) and _MGRS_SQUARE_RE.fullmatch(square.strip()):
        return f"MGRS-{square.strip()}"
    match = _ID_TILE_RE.match(str(getattr(item, "id", "")))
    return f"MGRS-{match.group(1)}" if match else None


def item_sequence(item: Any) -> int | None:  # noqa: ANN401 — any STAC-like item
    """The reprocessing sequence of an item, or ``None`` if it cannot be determined.

    Higher is newer. ``None`` is ordered *below* every known sequence, so an item whose
    sequence cannot be read never displaces one whose can — an unreadable sequence must not
    win a comparison it cannot participate in.
    """
    raw = item.properties.get("s2:sequence")
    if raw is not None:
        # Matched, not coerced (see `item_baselines`): `int()` truncates 1.9 to 1, and a numeric
        # type also accepts `-1`, `1e3` and Unicode digits — so a non-sequence would order the
        # copies instead of deferring to the one encoded in the item id.
        if _SEQUENCE_VALUE_RE.fullmatch(str(raw).strip()):
            return int(raw)
        logger.debug("Not a sequence: s2:sequence %r on %s", raw, getattr(item, "id", "?"))
    match = _ID_SEQUENCE_RE.search(str(getattr(item, "id", "")))
    return int(match.group(1)) if match else None


def _producer_of(
    item: Any,  # noqa: ANN401
    read_keys: tuple[str, ...],
    known_harmonisation: Harmonisation | None,
) -> Harmonisation | None:
    """The producer state a term may rank this copy on, or ``None`` where none may.

    ``known_harmonisation`` is the collection's own answer, for a collection whose producer cannot
    vary between items; where given it is used directly, because the assets have nothing to add and
    — for a provider keying its assets natively — nothing to say.

    Otherwise the answer comes from the assets, and ``None`` means it cannot be had: an empty read
    set means the collection's configured names are not its asset keys, and an incomplete one means
    what IS found describes a subset. Either way
    :func:`~tessera_embeddings.ingest.asset_locations.item_harmonisation` answers ``UNKNOWN`` for
    every copy, so a term counting that would rank them all as producer-problems and hand the
    tile-date to an older pre-threshold copy — the opposite of the usable-first rule.
    """
    if known_harmonisation is not None:
        return known_harmonisation
    if not read_keys or not read_set_is_complete(item, read_keys):
        return None
    return item_harmonisation(item)


def refuses_its_date(
    item: Any,  # noqa: ANN401
    read_keys: tuple[str, ...] = READ_ASSET_KEYS,
    known_harmonisation: Harmonisation | None = None,
) -> bool:
    """Whether loading this copy makes its date REFUSE rather than fail to read.

    Two shapes, both raised by :class:`~tessera_embeddings.ingest.stac.BoaOffsetParser` because no
    offset decision is correct for one of the copy's own sources: an unharmonised copy declaring no
    readable baseline, and a copy served from a bucket nobody has classified, at or above the
    correction threshold.

    A refusal is not a read failure, so the fallback ladder cannot recover from one — which is why
    such a copy is both ranked last and excluded from the ladder entirely. That makes
    ``known_harmonisation`` load-bearing rather than an optimisation: on a collection whose
    producer it settles, the correction path refuses an unreadable baseline whatever the assets
    look like, so a spare judged only on visible assets would be offered to the ladder and abort
    the ingest when the ladder reached it.

    A copy whose bands straddle a harmonised and a raw producer is NOT a refusal: the offset is
    decided per SOURCE (ADR-021), so it is corrected band by band. Only the two cases where the
    evidence itself is missing refuse.
    """
    # Answers "we can see that it refuses" and never "we cannot see anything" — an incomplete copy
    # fails to READ, which the ladder does handle.
    harmonisation = _producer_of(item, read_keys, known_harmonisation)
    if harmonisation is None or harmonisation is Harmonisation.HARMONISED:
        return False
    baseline = item_processing_baseline(item)
    # Ordered ABOVE the producer test on purpose: an unharmonised copy declaring no readable
    # baseline refuses whatever its producer turns out to be, including a MIXED one. Folding this
    # into the tuple below would let such a copy through.
    if baseline is None:
        return True
    return baseline >= S2_BASELINE_THRESHOLD and harmonisation is Harmonisation.UNKNOWN


def _would_refuse_its_date(
    baseline: int | None,
    harmonisation: Harmonisation | None,
) -> bool:
    """Whether choosing this otherwise-usable copy makes its date refuse rather than load.

    :class:`~tessera_embeddings.ingest.stac.BoaOffsetParser` refuses a source at or above the
    threshold whose producer cannot be identified, because no offset decision is correct for it.
    Nothing retries a refusal — the read-failure ladder steps down on a read error, not on this —
    so a copy that causes one must lose to a copy that does not, even an older one. A copy whose
    bands straddle two KNOWN producers is not among them: each source is decided on its own bucket.

    Only asked of copies whose read set is complete. An incomplete copy is already ranked last, and
    it reports an undecidable producer *because* it is incomplete, so counting it here would say the
    same thing twice and invert the baseline preference among copies that are all incomplete.
    """
    if baseline is None or baseline < S2_BASELINE_THRESHOLD:
        return False
    return harmonisation is Harmonisation.UNKNOWN


def _owes_a_correction(
    baseline: int | None,
    harmonisation: Harmonisation | None,
) -> bool:
    """Whether choosing this copy puts an offset decision on its date.

    A strictly wider question than :func:`_would_refuse_its_date`: a raw copy owes a correction
    without being ambiguous about it. Gated identically — on the threshold, because below it no
    producer changes a pixel, and on the producer being known at all, because a copy nobody can
    judge must not be ranked as though it had been judged and found wanting.

    Owing a correction withholds a copy from nothing; it costs a ranking position and no more, and
    that position is BELOW the baseline, because the two reasons to prefer a copy owing nothing are
    both about pixel quality and a quality preference may not buy a better pixel with an older
    reprocessing. See :func:`_preference_key` term 6.

    **Asked only where the producer was determined per ITEM**, which is the caller's job to
    enforce: this term compares producers, and where the collection settles the producer there is
    nothing to compare — it would then discriminate on the threshold alone and, ranked below the
    baseline, tie among every copy that reaches it.
    """
    if baseline is None or baseline < S2_BASELINE_THRESHOLD or harmonisation is None:
        return False
    return harmonisation is not Harmonisation.HARMONISED


def _acquisition_is_known(item: Any) -> bool:  # noqa: ANN401 — any STAC-like item
    """Whether this copy says which observation it came from.

    True on either evidence :func:`_by_acquisition` clusters by. A copy with neither was attached
    to a cluster arbitrarily, so nothing may be concluded from which cluster it landed in.
    """
    return acquisition_identity(item) is not None or acquisition_instant(item) is not None


def _preference_key(
    item: Any,  # noqa: ANN401
    read_keys: tuple[str, ...] = READ_ASSET_KEYS,
    known_harmonisation: Harmonisation | None = None,
) -> tuple[int, int, int, int, float, int, int, int, int, str]:
    """How much we would rather read this copy than another. Lower sorts first.

    **The canonical statement of the ranking order.** The module docstring holds the two invariants
    that constrain it, and the operator-facing summary in :func:`log_duplicate_selection` must track
    this list. Nothing else states the order.

    **Context-free:** no term is relative to the set being sorted, so the same key orders copies
    of one acquisition and copies from different acquisitions. That is what lets one key serve
    both, and it must hold for any term added here.

    The fields, in order:

    1. **Read-set completeness**, over ``read_keys`` — the assets THIS load will request. First,
       because a copy missing one cannot deliver the tile-date at any baseline, and a missing band
       is not one of the read failures the fallback ladder recognises, so an incomplete winner
       fails the acquisition outright.
    2. **Whether the producer is decidable**, where it would matter. A copy from a bucket nobody
       has classified cannot have its correction decided, so it refuses its date at or above the
       correction threshold — and neither the generic path nor the read-failure ladder retries a
       refusal. A copy whose own bands span a harmonised and a raw producer is NOT one of these:
       each source is decided on its own bucket. Inert below the threshold, where the producer
       changes no pixel.
    3. **Whether this copy demonstrably belongs to the acquisition it is ranked in.** A copy naming
       neither an observation nor an instant was attached to a cluster arbitrarily, so it must not
       displace one that does say which pass it came from, whatever its baseline. Above the
       baseline terms deliberately: a known member at an older baseline still represents that pass,
       while a possibly-unrelated copy at a newer one may duplicate another pass and drop this
       one's coverage entirely.
    4. **Whether the baseline is readable — for producers whose correction depends on it.** A copy
       declaring nothing cannot have its correction decided, so it refuses its date. Above the next
       term because a refusal has no recovery while owing a correction merely gets one. An
       already-harmonised copy is the exception: its pixels need no correction whatever the
       baseline says, so penalising it here would hand a tile-date to an OLDER raw reprocessing.
    5. **The baseline, descending, by value** — not by "is it the best", so every rung of the
       fallback ladder stays in descending baseline order.
    6. **Whether the copy owes an offset correction at all**, where the producer is an item's own
       property. Below the baseline, because both reasons to prefer a copy owing nothing are about
       the PIXEL rather than about coverage:

       - An already-harmonised copy had its floor applied by the producer *before* any resampling.
         A copy we correct ourselves is floored *after*, because the read and the reprojection are
         one step — so on the very dark population the two disagree, and the harmonised copy is the
         better of the two. See the limit recorded in
         ``context_docs/decisions/020-boa-offset-applies-to-every-valid-dn.md``.
       - A copy owing nothing cannot be wrong by the offset. One we correct is right only if the
         bucket lists and the item's declared baseline are both honest, and nothing detects it when
         they are not.

       Both are quality claims, and a quality preference must not buy a better pixel with an older
       reprocessing — which is why locality sits below the baseline too. This sits above locality
       only because a pixel argument outranks a cost one. Inert below the threshold, and inert
       without a read set, where every copy reports an undecidable producer and the term ties.
    7. **Locality, only where the baseline is readable.** Below the baseline so it cannot buy
       cheaper egress with an older pixel, and inert for unreadable baselines so it cannot decide
       a comparison the baseline could not enter.
    8. **Sequence, descending, then id.** The id makes the order total, so the choice is
       independent of catalogue response order and a rerun cannot produce a different mosaic.
    """
    baseline = item_processing_baseline(item)
    sequence = item_sequence(item)
    # Resolved once and shared by every term that asks about the producer, so they cannot disagree.
    harmonisation = _producer_of(item, read_keys, known_harmonisation)
    baseline_matters = harmonisation is not Harmonisation.HARMONISED
    # Inert where the producer came from the COLLECTION rather than the item — see
    # `_owes_a_correction` for why a term that compares producers has nothing to compare there.
    owes = known_harmonisation is None and _owes_a_correction(baseline, harmonisation)
    return (
        0 if read_set_is_complete(item, read_keys) else 1,
        1 if _would_refuse_its_date(baseline, harmonisation) else 0,
        0 if _acquisition_is_known(item) else 1,
        0 if baseline is not None or not baseline_matters else 1,
        -(baseline or 0.0),
        1 if owes else 0,
        (0 if item_is_in_preferred_location(item, keys=read_keys) else 1) if baseline is not None else 0,
        0 if sequence is not None else 1,
        -(sequence or 0),
        str(getattr(item, "id", "")),
    )


def _rank_copies(
    copies: list[Any],
    read_keys: tuple[str, ...] = READ_ASSET_KEYS,
    known_harmonisation: Harmonisation | None = None,
) -> list[Any]:
    """Order copies of one acquisition, PREFERRED first.

    An unreadable baseline ranks last rather than suspending the comparison: such a copy refuses
    its date in :class:`~tessera_embeddings.ingest.stac.BoaOffsetParser`, and the
    read-failure ladder recovers from a read error but not from a refusal. A reprocessing that can
    be corrected beats a newer one that cannot be processed at all.
    """
    return sorted(
        copies,
        key=functools.partial(_preference_key, read_keys=read_keys, known_harmonisation=known_harmonisation),
    )


#: How far apart two catalogue timestamps must be to be different acquisitions rather than
#: reprocessings of one, used only where :func:`acquisition_identity` cannot answer. The closest
#: genuinely distinct pair observed on the live catalogue is a successive orbit ~50 minutes later.
#: It is NOT a safe margin against reprocessings, which is why identity is tried first: copies of
#: one granule at baselines 02.06 and 05.00 are timestamped more than three minutes apart.
_SAME_ACQUISITION_S = 120.0

#: ``s2:datatake_id``, e.g. ``GS2B_20171219T095409_004109_N05.00``: mission, sensing start,
#: absolute orbit, then the processing baseline. Only the last field changes between
#: reprocessings, so the head of it names the observation itself.
_DATATAKE_RE = re.compile(r"(G[A-Z0-9]{2,}_[0-9]{8}T[0-9]{6}_[0-9]+)_N[0-9]{1,2}\.[0-9]{1,2}")


def acquisition_identity(item: Any) -> str | None:  # noqa: ANN401 — any STAC-like item
    """A stable name for the OBSERVATION, shared by every reprocessing of it.

    Preferred over :func:`acquisition_instant` for deciding whether two copies are the same
    acquisition: the catalogue ``datetime`` is per-copy and reprocessings do not agree on it, so no
    tolerance around it separates "two reprocessings" from "two passes" without getting one wrong.
    Mission, sensing start and orbit are identical across a reprocessing pair and different across
    passes, so no tolerance is needed at all.

    ``None`` where the property is absent or not this shape, which sends the caller to the
    instant. Matched as a whole rather than split on underscores, so a value that is not a
    datatake id is rejected instead of yielding a plausible-looking wrong key.
    """
    raw = item.properties.get("s2:datatake_id")
    if not isinstance(raw, str):
        return None
    match = _DATATAKE_RE.fullmatch(raw.strip())
    if match is None:
        logger.debug("Not a datatake id: %r on %s", raw, getattr(item, "id", "?"))
        return None
    return match.group(1)


def acquisition_instant(item: Any) -> datetime.datetime | None:  # noqa: ANN401 — any STAC-like item
    """The instant an item was ACQUIRED, or ``None`` if it cannot be read.

    Read from ``properties["datetime"]`` and never from ``item.datetime``, because by
    the time duplicates are selected the latter has been overwritten with the canonical
    noon-UTC solar-day stamp (:func:`normalize_to_solar_day`) and every copy of a day
    therefore carries an identical value.

    A per-copy field: reprocessings of one granule do not agree on it, so it is the FALLBACK
    for placing a copy and :func:`acquisition_identity` is the primary.
    """
    raw = item.properties.get("datetime")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("Unparseable datetime %r on %s", raw, getattr(item, "id", "?"))
        return None
    # No offset means UNREADABLE, not assumed UTC: guessing would place the acquisition up to a
    # day out, and a naive value alongside aware ones makes the sort in `_by_acquisition` raise.
    if parsed.tzinfo is None:
        logger.debug("Timezone-less datetime %r on %s", raw, getattr(item, "id", "?"))
        return None
    return parsed


def _by_acquisition(copies: list[Any]) -> list[list[Any]]:
    """Split one tile-date's copies into one list per distinct acquisition.

    **Identity first, timestamps only where there is no identity.** Copies naming the same
    observation (:func:`acquisition_identity`) are one cluster however far apart their catalogue
    timestamps are — which is how reprocessings of one granule actually present themselves, so
    grouping by timestamp fuses nothing and leaves both copies to be loaded and mosaicked together.
    The remainder falls back to the timestamps, clustered within :data:`_SAME_ACQUISITION_S`.

    **A copy without an identity JOINS an identified acquisition its timestamp places it in**,
    rather than always starting a cluster of its own. One reprocessing declaring the datatake while
    its sibling omits it would otherwise never be compared, however close their timestamps, so both
    would survive to be fused — the same defect identity exists to fix, by a different route.

    **A copy with neither JOINS a cluster rather than forming its own**, so it competes for one slot
    instead of adding one — that is what stops two copies of a pass being fused at two processing
    baselines. Collapsing the whole tile-date would prevent the same fusion by discarding coverage
    from passes that identified themselves perfectly well, which this function exists to preserve.
    Which cluster such a copy joins is arbitrary, which is why it cannot WIN one
    (:func:`_preference_key` ranks a known acquisition above data vintage) and why attributed
    recovery will not place one (:func:`_first_for_failed_acquisition`).

    Splitting on a real acquisition protects real coverage — see
    :func:`select_preferred_duplicates` for what keying on (tile, solar day) alone discarded.
    """
    # Sorted so the cluster ORDER is a property of the items and not of catalogue response order.
    identified: dict[str, list[Any]] = {}
    known: list[tuple[datetime.datetime, Any]] = []
    unplaceable: list[Any] = []
    for item in copies:
        identity = acquisition_identity(item)
        if identity is not None:
            identified.setdefault(identity, []).append(item)
            continue
        instant = acquisition_instant(item)
        if instant is None:
            unplaceable.append(item)
        else:
            known.append((instant, item))

    clusters: list[list[Any]] = [group for _, group in sorted(identified.items())]

    # A timestamp-only copy joins an identified acquisition it is close to, before it is allowed to
    # start one. Matched against ANY member: members of one acquisition do not agree on the
    # timestamp (that disagreement is why identity is the primary key), so being close to any of
    # them is the whole of the available evidence.
    orphans: list[tuple[datetime.datetime, Any]] = []
    for instant, item in sorted(known, key=lambda pair: pair[0]):
        joined = next(
            (
                cluster
                for cluster in clusters
                if any(
                    (sibling_instant := acquisition_instant(sibling)) is not None
                    and abs((instant - sibling_instant).total_seconds()) <= _SAME_ACQUISITION_S
                    for sibling in cluster
                )
            ),
            None,
        )
        if joined is None:
            orphans.append((instant, item))
        else:
            joined.append(item)

    if orphans:
        clusters.append([orphans[0][1]])
        for previous, current in itertools.pairwise(orphans):
            if (current[0] - previous[0]).total_seconds() > _SAME_ACQUISITION_S:
                clusters.append([current[1]])
            else:
                clusters[-1].append(current[1])

    if not clusters:
        return [copies]
    clusters[0].extend(unplaceable)
    return clusters


def select_preferred_duplicates(
    items: Sequence[Any],
    read_keys: tuple[str, ...] = READ_ASSET_KEYS,
    known_harmonisation: Harmonisation | None = None,
) -> tuple[list[Any], dict[tuple[str, str], list[Any]]]:
    """Keep one item per (tile, solar day); return the survivors and the alternates.

    Items must already be normalised to solar days — the grouping key is the solar day, so
    an un-normalised item would be grouped under the wrong one. :func:`solar_day_of` raises
    rather than guessing, which is what enforces the ordering.

    ``read_keys`` is the asset set THIS load will request, extra bands included. Readability and
    locality are judged over it, so a copy missing a band the caller asked for loses to one that
    has it; judging a fixed set instead lets a preferred copy lack a requested extra asset and
    fail at load.

    ``known_harmonisation`` is the producer state the COLLECTION settles, for a collection whose
    items cannot disagree about it. Supplying it is what lets a provider serving its bands under
    native asset keys be judged at all: without it every copy reports an undecidable producer, and
    a spare that will refuse its date is offered to the fallback ladder, which cannot recover from
    a refusal.

    Items whose tile cannot be determined are passed through untouched and never grouped:
    the radar path has no such duplicates, and an item we cannot key is safer kept than
    silently dropped.

    **One per ACQUISITION, not one per day.** A tile-date can hold two genuinely different
    acquisitions as well as reprocessings of one — at high latitude, successive orbits revisit the
    same tile the same day. Those are separate imagery, mosaicked together by the loader, and
    collapsing them to one copy silently discards coverage. Measured against the live catalogue
    over eight tiles and 2021: keying on (tile, solar day) alone dropped **493 of 2,733 items** as
    duplicates when they were distinct acquisitions — nothing at all in the mid-latitude tiles,
    196 of 500 at 33XVG and 297 of 500 at 22XER. So copies are split by acquisition first, and
    only true reprocessings of one acquisition compete on ``s2:sequence``.

    Returns:
        ``(kept, alternates)``. ``kept`` preserves the input order of the survivors, so a
        caller's own sort still decides fusion order, and holds one item per acquisition, so a
        tile-date with two acquisitions keeps both. ``alternates`` maps each ``(tile, solar_day)``
        that had a rejected copy to those copies, most preferred first — the ladder a caller steps
        down when a chosen copy cannot be read. It stays keyed by tile-date rather than by
        acquisition because that is the granularity a read failure is attributed at.
    """
    groups: dict[tuple[str, str], list[Any]] = {}
    ungrouped: list[Any] = []
    for item in items:
        tile = item_tile(item)
        if tile is None:
            ungrouped.append(item)
            continue
        groups.setdefault((tile, solar_day_of(item)), []).append(item)

    # Identity, not equality: two STAC items for one tile-date differ only in fields an
    # __eq__ may not compare, so a value-based membership test could drop the wrong copy.
    survivors: set[int] = {id(it) for it in ungrouped}
    alternates: dict[tuple[str, str], list[Any]] = {}
    for key, copies in groups.items():
        rejected: list[Any] = []
        for acquisition in _by_acquisition(copies):
            ranked = _rank_copies(acquisition, read_keys, known_harmonisation)
            survivors.add(id(ranked[0]))
            rejected.extend(ranked[1:])
        if rejected:
            # ONE order over every rejected copy of the tile-date, not a concatenation of
            # per-acquisition ladders. The unattributed recovery consumes `copies[0]` on each
            # retry, so every position is a choice and the whole list has to be ranked.
            #
            # Only copies that would REFUSE ON THEIR OWN ACCOUNT are dropped rather than offered.
            # The ladder steps down on a read failure; a refusal is not one, so handing one over
            # escapes as `HeterogeneousProducerError` instead of trying the next usable copy or
            # recording the date as lost. A spare that merely OWES the correction IS offered: the
            # offset is decided per image (ADR-021), so swapping one in changes nothing beyond its
            # own tile.
            usable = [it for it in rejected if not refuses_its_date(it, read_keys, known_harmonisation)]
            if len(usable) != len(rejected):
                logger.debug(
                    "%s: %d of %d spare copies excluded from the fallback ladder — their own "
                    "producer or baseline cannot be determined, so they would refuse rather than "
                    "fail to read",
                    key,
                    len(rejected) - len(usable),
                    len(rejected),
                )
            if usable:
                alternates[key] = sorted(
                    usable,
                    key=functools.partial(
                        _preference_key, read_keys=read_keys, known_harmonisation=known_harmonisation
                    ),
                )

    kept = [it for it in items if id(it) in survivors]
    return kept, alternates


def _contested_key(item: Any) -> tuple[str, str] | None:  # noqa: ANN401 — any STAC-like item
    """An item's ``(tile, solar_day)`` key, or ``None`` if it cannot be keyed.

    Returns ``None`` rather than raising: the only caller is a log line, and a logging call must
    not be able to abort an ingest.
    """
    tile = item_tile(item)
    if tile is None:
        return None
    try:
        return (tile, solar_day_of(item))
    except Exception:
        logger.debug("Could not key %s for the duplicate audit line", getattr(item, "id", "?"))
        return None


def log_duplicate_selection(
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    roi: str,
    alternates: dict[tuple[str, str], list[Any]],
    kept: Iterable[Any] = (),
    read_keys: tuple[str, ...] = READ_ASSET_KEYS,
    items: Iterable[Any] = (),
) -> None:
    """Record that duplicates were pruned, at a level that survives a fleet-wide log.

    Summary only, and deliberately: duplicates are routine, so a line per pruned copy would
    bury the outcomes that are not. The tile-dates carrying alternates are named only when a
    fallback actually fires, where the identity matters.

    **Reports WHERE the surviving copies came from.** This is the only audit trail for that
    decision: where two copies carry the same baseline their pixels are identical, so nothing
    downstream can show which was read.

    Counted over the CONTESTED tile-dates only. ``kept`` is every survivor on the ROI, most of
    which had no duplicate, so counting them all would describe the supply rather than the choices.

    ``read_keys`` must be the set selection ranked on, or the line reports a locality the decision
    did not use — a winner serving every requested band locally reads as remote for lacking an
    asset nobody asked for.
    """
    # Counted from what selection DID, not from the ladder: copies that would refuse their date are
    # excluded from `alternates`, so it does not answer "what was pruned".
    supplied, survivors = list(items), list(kept)
    contested_keys: set[tuple[str, str]] | None = None
    if supplied:
        pruned = len(supplied) - len(survivors)
        # Keys whose multiplicity DECREASED. A tile-date that loses a copy keeps its key, so a
        # difference of cardinalities cannot see it; and a tile-date holding several genuinely
        # distinct same-day passes has multiplicity above one while losing nothing, so counting
        # every multi-item key inflates the figure with dates no choice was made on.
        before = collections.Counter(k for it in supplied if (k := _contested_key(it)) is not None)
        after = collections.Counter(k for it in survivors if (k := _contested_key(it)) is not None)
        decreased = {key for key, n in before.items() if after[key] < n}
        contested_keys = decreased
        contested = len(decreased)
    else:
        # Nothing to compare against, so report the ladder — the answer for a caller that passes
        # only `alternates`.
        pruned = sum(len(v) for v in alternates.values())
        contested = len(alternates)
    if pruned <= 0:
        return
    recoverable = sum(len(v) for v in alternates.values())
    # Which tile-dates the source breakdown covers, taken from what pruning DID rather than from the
    # recovery ladder. A tile-date whose every spare was excluded as a refusal risk has no entry in
    # `alternates`, so keying off that drops its winner and reports source totals for the
    # recoverable subset alone while still labelling them as the winners.
    audited = set(alternates) if contested_keys is None else contested_keys
    winners = [it for it in kept if _contested_key(it) in audited]
    where = ""
    if winners:
        local = sum(1 for it in winners if item_is_in_preferred_location(it, keys=read_keys))
        where = f"; winners by source: {local} in-region, {len(winners) - local} remote"

    # Must track `_preference_key`'s docstring, which is the canonical statement of this order. An
    # operator reads this line to explain a selected pixel, so a stale order gives them the wrong
    # explanation.
    how = (
        "complete read set, then a decidable producer, then a known acquisition, then a readable "
        "baseline, then newest baseline, then owing no offset correction, then in-region, then "
        "newest sequence"
    )
    log.info(
        "Duplicate catalogue items pruned roi=%s: %d tile-date(s) had more than one copy, "
        "%d rejected, %d of those available as a fallback. Preference: %s%s",
        roi,
        contested,
        pruned,
        recoverable,
        how,
        where,
    )


def alternates_for(
    alternates: dict[tuple[str, str], list[Any]],
    items: Iterable[Any],
    only: Iterable[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], list[Any]]:
    """The subset of ``alternates`` belonging to the tile-dates present in ``items``.

    A date's fallback ladder must not reach into another date's copies: the tile-dates are
    what identify a source object, and stepping down the wrong one would swap imagery for a
    day that read perfectly well.

    ``only`` narrows further, to the tile-dates a failure was ATTRIBUTED to; ``None`` means it
    could not be, and yields every duplicated tile-date in the date. An EMPTY ``only`` is
    meaningful and distinct from ``None``: the failure was attributed, and to nothing here. See
    :func:`step_down_copies` for what stepping an unattributed date costs.

    Exhausted entries are omitted rather than returned empty, so a caller can treat "no
    alternates" and "every alternate already tried" as the one condition they are.
    """
    wanted = {(tile, solar_day_of(it)) for it in items if (tile := item_tile(it)) is not None}
    if only is not None:
        wanted &= set(only)
    return {key: copies for key, copies in alternates.items() if key in wanted and copies}


#: Signatures of a source object that cannot be READ, as distinct from a transient failure.
#:
#: Matched on the whole exception chain's TEXT: the informative part is the cause — rasterio's
#: ``WarpOperationError('Chunk and warp failed')`` discards the reason, and the reason (a codec
#: that cannot inflate a tile) is what says the object is broken rather than briefly unavailable.
#: Text rather than types because the type NAMES survive the hop to the orchestrator however
#: faithfully the classes are rebuilt, and one string can name a condition several unrelated
#: layers report. See ``ingest/loader_failures.py`` for what the chain arrives with and what has
#: to be installed on the reader for it to arrive at all.
#:
#: Only signatures naming a CODEC OR FORMAT OPERATION failing, and the minimality is the point:
#: `True` here means "give up this date", and it is the DEFAULT for anything the transient lists
#: do not recognise, so every gap in those lists becomes a silent data-loss verdict. That is how a
#: 429, and separately a DNS failure, a refused connection and a TLS error, each came to be
#: recorded as permanently bad imagery (``context_docs/design/ingest_read_failure_causes_2026_08.md``).
#:
#: So the generic wrappers are deliberately absent: `Chunk and warp failed`, `Read failed. See
#: previous exception` and `IReadBlock failed` are what GDAL raises when a block read fails FOR ANY
#: REASON, transport failures included. They say a read failed, never that the bytes are bad, and
#: matching them made the dangerous verdict the fallback for the unknown. A refusal wrapped in one
#: of them is still caught, because the CAUSE carries the codec name and the whole chain is
#: matched; anything unrecognised re-raises and the leg retries in order. A genuinely truncated
#: object naming no codec therefore fails the leg loudly instead of quietly holing the store.
_UNREADABLE_MARKERS = (
    "ZIPDecode",
    "TIFFReadEncodedTile",
)

#: Signatures of a source object that is NOT THERE — which none of the markers above can see,
#: because every one of them is emitted by a BLOCK READ and a missing object fails at open,
#: before any block is requested. A catalogue item can name an href the provider never
#: published, and no retry publishes it.
#:
#: Several forms, because several layers can surface the condition and two GDAL drivers are in
#: play: an `s3://` href goes through the S3 driver and says `ObjectNotFound`, the S3 API answers
#: with the `NoSuchKey` code and its body carries the sentence, while the plain `https://` hrefs
#: the optical assets actually carry go through the HTTP driver and say only `HTTP response code:
#: 404`. Matching one form and not the other is why the optical step-down would not have fired on
#: the path it was written for. Each still needs an independent
#: :data:`_SOURCE_READER_MARKERS` corroboration, so the status alone claims nothing.
_MISSING_OBJECT_MARKERS = (
    "ObjectNotFound",
    "NoSuchKey",
    # 410 Gone alongside 404: both say the object is not there and no retry publishes it. Named
    # here so the un-wrapped form gets the same corroboration 404 gets.
    "HTTP response code: 410",
    "The specified key does not exist",
    "HTTP response code: 404",
)


class ReadFailure(StrEnum):
    """What one read failure IS — exactly one member, so no two verdicts can both apply.

    Closed because the alternative failed repeatedly: with correctness resting on nine marker
    tuples being jointly COMPLETE and mutually EXCLUSIVE and nothing checking either, gaps recurred
    one status at a time (429, then 401, then every 5xx) and an overlap let call ORDER decide
    whether a date was skipped or a leg retried. A function returning one member cannot overlap,
    and a member nobody handles is a decision somebody has to make rather than a silence.

    Ordered by what the evidence is ABOUT, and that order is the classification:

    * :attr:`OUR_CREDENTIAL` — repairable here, and no waiting or copy fixes it.
    * :attr:`PROVIDER_REFUSED` — the service said no; only time resolves it.
    * :attr:`REFUSAL_UNATTRIBUTED` — refusal words with nothing tying them to the source reader.
      Every string in the refusal markers belongs to S3, and the destination store speaks S3 too,
      so unattributed it may be our own bucket answering and must claim nothing.
    * :attr:`CLIENT_ERROR` — the request is wrong; every copy sends the same kind of request.
    * :attr:`UNREADABLE` — these bytes will not yield, on this attempt or any other.
    * :attr:`ABSENT` — the object was never published.
    * :attr:`UNDECIDABLE` — no evidence either way, which is what a cause destroyed crossing the
      worker boundary looks like. It earns nothing: no wait on suspicion, and no date given up.
    """

    OUR_CREDENTIAL = "our-credential"
    PROVIDER_REFUSED = "provider-refused"
    REFUSAL_UNATTRIBUTED = "refusal-unattributed"
    CLIENT_ERROR = "client-error"
    UNREADABLE = "unreadable"
    ABSENT = "absent"
    UNDECIDABLE = "undecidable"


def _means_the_copy_is_lost(verdict: ReadFailure) -> bool:
    """Whether ``verdict`` says THIS COPY will not yield its bytes.

    An exhaustive match rather than a set, so the type checker enforces what the closed set is for:
    add a member and mypy reports this function as reachable past its cases, naming the one it does
    not handle. A set would silently answer ``False`` for anything new — the quiet wrong answer
    this design exists to remove, and the direction that costs a store.
    """
    match verdict:
        case ReadFailure.UNREADABLE | ReadFailure.ABSENT:
            return True
        case (
            ReadFailure.OUR_CREDENTIAL
            | ReadFailure.PROVIDER_REFUSED
            | ReadFailure.REFUSAL_UNATTRIBUTED
            | ReadFailure.CLIENT_ERROR
            | ReadFailure.UNDECIDABLE
        ):
            return False
    assert_never(verdict)


def classify_read_failure(exc: BaseException) -> ReadFailure:
    """The single verdict for ``exc``, from which both public predicates are derived.

    Text over ``isinstance`` for the reason :func:`_exception_chain_text` gives. So this is the
    READING of the chain, and :func:`classify_read_failure_in` is the whole of the judgement.
    """
    return classify_read_failure_in(_exception_chain_text(exc))


def classify_read_failure_in(text: str) -> ReadFailure:
    """The single verdict named by ``text``, whatever produced those words.

    Takes the WORDS rather than an exception, because not every caller has one: a leg running as
    its own deployment reports through a state message, which the exception object does not cross
    at all (see the leg retry ladder in ``orchestration/prefect/flows/ingest_zone_year.py``). One
    classifier reads both, so a verdict cannot depend on which side of a boundary the caller is on.

    The ORDER is the policy, each step a claim about what outranks what: our own fault first,
    because no amount of waiting or stepping down repairs it; then the service refusing, because
    that outranks every statement about bytes made in the same chain; then the request being wrong,
    since no copy is fetched differently; and only then anything about the data itself.
    """
    if any(m in text for m in _OWN_CREDENTIAL_MARKERS):
        return ReadFailure.OUR_CREDENTIAL
    if _names_a_transient_refusal(text):
        if not any(m in text for m in _SOURCE_READER_MARKERS):
            return ReadFailure.REFUSAL_UNATTRIBUTED
        return ReadFailure.PROVIDER_REFUSED
    statuses = _http_statuses(text)
    # Every 4xx that is neither absence nor a reason to wait. Both exceptions are named by their
    # own sets rather than one by a set and the other by a literal, so a status can be moved
    # between the two by editing one line and cannot end up in both.
    if any(400 <= n < 500 and n not in (_ABSENT_4XX | _TRANSIENT_4XX) for n in statuses):
        return ReadFailure.CLIENT_ERROR
    if any(m in text for m in _UNREADABLE_MARKERS):
        return ReadFailure.UNREADABLE
    if any(m in text for m in _MISSING_OBJECT_MARKERS) and any(m in text for m in _SOURCE_READER_MARKERS):
        return ReadFailure.ABSENT
    return ReadFailure.UNDECIDABLE


def _names_a_transient_refusal(text: str) -> bool:
    """Whether the chain says the SERVICE refused, by name or by status RANGE.

    The one expression both public predicates read, which is what makes them disjoint by
    construction rather than by two lists kept in step. Kept in step is what failed: with the
    markers naming 403 and 500 as strings while the ranges covered every 5xx,
    `HTTP response code: 503` was a refusal to neither predicate, and radar drew the ordinary
    attempt limit for the commonest shape of outage there is.

    Ranged for the reason the ranges exist: a status nobody enumerated must not change the verdict.
    """
    if any(m in text for m in _PROVIDER_REFUSAL_MARKERS):
        return True
    return any(s >= 500 or s in _TRANSIENT_4XX for s in _http_statuses(text))


def is_unreadable_source(exc: BaseException) -> bool:
    """Whether ``exc`` says a source object could not be read.

    Deliberately NARROW, and it fails closed: anything unrecognised is re-raised by the caller
    rather than treated as bad data. Stepping down is a response to data that will never read, not
    to a bad minute — falling back to an older copy on a credential expiry or a throttle would
    silently swap in worse imagery to work around a fault a retry fixes. An expired credential is
    excluded for that reason, even though it surfaces through the same wrapper.

    An object the provider never published qualifies, and reads no differently from a corrupt one:
    no retry produces it, and another copy of the tile-date may have it. It has to be recognised as
    the SOURCE reader's failure, though, which the not-found text alone cannot say — see
    :data:`_SOURCE_READER_MARKERS`.

    NOT matched: a whole bucket or prefix being gone, which is systemic by definition and must fail
    the run on the first date rather than be skipped date by date.
    """
    # Derived, so this cannot disagree with `is_provider_refusal`: one classifier assigns one
    # verdict and each predicate reports which verdicts it owns.
    return unreadable_source_in(_exception_chain_text(exc))


def unreadable_source_in(text: str) -> bool:
    """Whether ``text`` says a source object could not be read.

    :func:`is_unreadable_source` for a caller holding words instead of an exception: same verdicts,
    same closed set, same fail-closed direction. Only ``UNREADABLE`` and ``ABSENT`` answer
    ``True``; a refusal, a credential fault and a chain carrying no evidence all answer ``False``.
    The docstring above has the exclusions, which are the load-bearing part.
    """
    return _means_the_copy_is_lost(classify_read_failure_in(text))


def is_source_read_failure(exc: BaseException) -> bool:
    """Whether ``exc``'s chain says the SOURCE READER is what failed, whatever it failed with.

    Not a verdict about the failure — :func:`classify_read_failure` is that — but about whose
    failure it is. The one caller is the code that copies a refusal out of GDAL's log onto an
    exception: that evidence is about a READ, and attaching it to anything else would let a
    concurrent read's refusal explain a store conflict or a duplicate date. Same marker set as
    every other corroboration here, for the reason :data:`_SOURCE_READER_MARKERS` gives.

    Admits the flattened shape too, whose whole message is the reader's wrapper repr — that is
    the shape this evidence exists to rescue, so excluding it would defeat the point.
    """
    return any(m in _exception_chain_text(exc) for m in _SOURCE_READER_MARKERS)


def cause_was_flattened(exc: BaseException) -> bool:
    """Whether ``exc`` is a read failure that arrived with its cause destroyed.

    The shape is Dask's substitute for an exception it could not serialise: a plain ``Exception``
    whose whole message is the repr of the real one, and no chain behind it. Every predicate here
    reads the chain, so this shape carries no evidence to classify at all, and the fail-closed
    verdict it gets is the absence of a decision rather than one.

    ``ingest/loader_failures.py`` stops it being produced, but that rescue is installed per process
    and best effort, so a worker it did not reach still yields this shape and the difference is
    invisible in the verdict. The detector exists to make the residue LOUD — the one signal that a
    re-raise happened for want of evidence, not for want of a matching marker.
    """
    return (
        type(exc) is Exception
        and exc.__cause__ is None
        and exc.__context__ is None
        and any(m in str(exc) for m in _SOURCE_READER_MARKERS)
    )


def _exception_chain_text(exc: BaseException) -> str:
    """The chain's type names and messages, joined, for text classification.

    Text rather than ``isinstance`` because the chain crosses the Dask boundary and the classes
    that reach it there are rebuilt by tblib, whose reconstruction is only as faithful as the
    reducer each class exposes; the type NAMES survive that regardless. See
    ``ingest/loader_failures.py`` for what keeps the chain itself intact. The WHOLE chain, because
    the wrapper discards the reason: ``WarpOperationError('Chunk and warp failed')`` says nothing.

    A chain's NOTES are part of its text. Not all of a read failure's reason is ever in the
    exception: GDAL states some of it only in its own log and raises something else entirely, so
    the reader holding that log attaches what it found as a note — see
    :func:`tessera_embeddings.ingest.loader_failures.carry_logged_refusal`, the only writer of
    notes in this pipeline. Without this line that evidence would be gathered and then ignored.
    """
    seen: list[str] = []
    current: BaseException | None = exc
    links = 0
    # Links, not entries: a note must not shorten how far down the chain this reads.
    while current is not None and links < 20:
        seen.append(f"{type(current).__name__}: {current}")
        seen.extend(str(note) for note in getattr(current, "__notes__", ()))
        current = current.__cause__ or current.__context__
        links += 1
    return " | ".join(seen)


#: Signatures of the PROVIDER refusing a read — an authorization refusal, a throttle, a server
#: error. All statements about the SERVICE rather than the bytes: the same object read moments
#: before and reads again once the service recovers.
#:
#: Read two ways. As an EXCLUSION by :func:`is_unreadable_source`, and positively by
#: :func:`is_provider_refusal`, whose whole purpose is to earn the failure a longer WAIT.
#:
#: **A positive verdict may never give up the date.** Giving up a date and then committing a later
#: one puts the earlier date permanently below the store's append-only maximum, so the re-run meant
#: to recover it is refused instead. A refusal is transient, so the response is to wait it out and
#: — if it will not clear — to fail the leg with the axis unmoved.
#:
#: The exclusion needs no :data:`_SOURCE_READER_MARKERS` corroboration; the positive verdict does.
#: Declining is the safe direction either way, but spending a long wait is not: an exception nobody
#: can attribute to the source reader must fail on its first date rather than hold a fleet idle.
_PROVIDER_REFUSAL_MARKERS = (
    "AccessDenied",
    "SlowDown",
    "ServiceUnavailable",
    "InternalError",
    "HTTP response code: 403",  # see the range note below: a provider judgement, not an HTTP one
    "HTTP response code: 500",
    # Gateway statuses (502, 504) are NOT named here — the `>= 500` range in
    # `_names_a_transient_refusal` claims them. They have to be claimed by SOMETHING: the optical
    # assets are plain `https://` hrefs, so a gateway failure arrives as a bare status from GDAL's
    # HTTP driver, and unclaimed a source outage wrapped in `Chunk and warp failed` is recorded as
    # permanently unreadable DATA — telling an operator to look for corruption while they should
    # be waiting for the provider.
    "TooManyRequests",
    # Transport-level failures, which carry no status at all. A connection dropped mid-transfer is
    # a statement about the link, never about the bytes — the same object reads on the next
    # attempt. Unnamed, a throttling provider's dropped reads are claimed by `is_unreadable_source`
    # through whichever block-read wrapper GDAL happens to raise, and a recoverable date is
    # recorded as permanent loss. The list covers the WHOLE class: while half of it was named and
    # half was not, a dropped connection was declined while a refused one was read as bad data —
    # the same link failing, classified two ways depending on which word GDAL happened to use.
    "Connection reset",
    "Broken pipe",
    "timed out",
    "Connection aborted",
    "RequestTimeout",
    "Could not resolve host",
    "Connection refused",
    "Empty reply from server",
    "handshake failed",
)

#: GDAL's HTTP driver reports a bare status, so the numeric forms are classified by RANGE rather
#: than enumerated. Enumeration is what let 429 — the most explicit "slow down" a provider can send
#: — be read as unreadable data: the list held 403 and 500/502/503/504 and simply missed it, and a
#: list that must be complete to be correct will be incomplete again.
#:
#: The ranges are exhaustive and each is decidable without knowing the provider:
#:
#: * **Any 5xx is transient.** It is a statement about the server, never about the bytes. That
#:   covers 507 and 509 (a CDN bandwidth throttle) without either being named.
#: * **408 and 429 are transient** — the two 4xx that mean "not now" rather than "not you".
#: * **404 and 410 are absence**, and are corroborated separately by
#:   :data:`_MISSING_OBJECT_MARKERS`.
#: * **403 is transient HERE, deliberately**, and is the one entry that is a judgement about a
#:   provider rather than about HTTP: Earth Search answers a rate block with it. Elsewhere a 403
#:   is about who is asking, which is why the catalogue layer treats it per provider too.
#: * **Every other 4xx is neither** — a malformed request or a rejected credential is not the
#:   data's fault and is not fixed by waiting, so it re-raises and fails the leg on its first
#:   date instead of being skipped date by date.
#:
#: GDAL states a status in four wordings, and all of them reach this classifier as
#: ``rasterio._env`` records at WARNING:
#:
#: * ``CPLE_AppDefined in HTTP response code on <url>: 403``
#: * ``CPLE_AppDefined in HTTP error code: 503 - <url>. Retrying again in 0.5 secs``
#: * ``CPLE_AppDefined in HTTP error code for <url> range <first>-<last>: 503. Retrying again ...``
#: * ``CPLE_AppDefined in Request for <url> range <first>-<last> failed with response_code=403``
#:
#: The first three spell the status after ``code`` and differ only in what sits between: the
#: object's URL, nothing at all, or the URL and the byte range. The fourth spells it
#: ``response_code=`` instead — what the ranged reader states when it STOPS retrying, where the
#: third is what it says on each retry. A pattern anchored on ``HTTP response code:`` reads a
#: status out of NONE of them, and a line carrying no status is not a refusal, so the refusal each
#: states goes unseen and the codec failure beneath it keeps its unreadable verdict.
#:
#: What sits between the phrase and the status is an OBJECT ADDRESS, matched as one — ``\S+``,
#: since a URL holds no whitespace — which keeps the phrase from binding to some unrelated number
#: later on a flattened traceback line. No length cap: a CloudFront-signed OPERA href runs past 900
#: characters once its policy and signature are on it, the shape the radar path uses whenever
#: ``use_s3_direct=False``.
#:
#: **The separator must be a colon followed by a SPACE, and that is what keeps a URL port out.**
#: GDAL formats the pair as ``%s: %d``; an authority writes ``host:443`` with no space. Without it
#: the shortest match in ``... on https://host:443/object.tif: 403`` is the PORT — read as a 4xx
#: that is neither absence nor a reason to wait, so the real refusal is dropped and the codec
#: failure keeps its unreadable verdict and costs its date. A ``:503`` port went the other way and
#: invented a refusal. The trailing guard rejects a longer port such as ``:8443``.
#:
#: **The fourth is matched on its own key**, not folded into the alternation: ``response_code=``
#: separates with ``=`` and no space, so it is unambiguous and needs none of the port guard the
#: ``code:`` forms do. It comes from ``Request for %s range %s failed with response_code=%ld``.
#:
#: ``\d{3}`` and not ``\d+`` keeps ``response_code=0`` out. GDAL prints that when the request never
#: completed: it is the absence of a status, and read as one it would be neither absence nor a
#: reason to wait — the verdict that re-raises.
_HTTP_STATUS_RE = re.compile(
    r"(?:HTTP (?:response|error) code(?: on \S+| for \S+ range \S+)?:[ \t]+|response_code=)(\d{3})(?![\d/])"
)

#: 4xx statuses that mean WAIT rather than stop.
#:
#: **403 belongs here and not only in** :data:`_PROVIDER_REFUSAL_MARKERS`. A marker is a wording;
#: a status is a fact. The literal ``HTTP response code: 403`` matches what GDAL says when it
#: cannot OPEN an object — and that failure carries the words into the exception anyway, where the
#: marker was always going to catch them. It does NOT match what GDAL says when a chunk READ is
#: refused mid-transfer, which is the shape a lost date arrives in and whose exception says only
#: ``Chunk and warp failed``. The same split cost us 5xx once already: with the markers naming 403
#: and 500 as strings and only the ranges covering every 5xx, ``HTTP response code: 503`` was a
#: refusal to neither predicate.
_TRANSIENT_4XX = frozenset({403, 408, 429})
_ABSENT_4XX = frozenset({404, 410})


def _http_statuses(text: str) -> set[int]:
    """Every HTTP status the exception chain names."""
    return {int(m) for m in _HTTP_STATUS_RE.findall(text)}


#: What makes a failure the SOURCE reader's: GDAL is the layer that reported it.
#:
#: Required alongside :data:`_MISSING_OBJECT_MARKERS` and :data:`_PROVIDER_REFUSAL_MARKERS` by
#: :func:`is_unreadable_source`, for one reason shared by both: every string in those two tuples
#: belongs to S3, and every S3 client in this process speaks S3. Icechunk's error enum carries
#: ``ObjectNotFound`` and ``NoSuchKey`` verbatim, our own store write answers ``AccessDenied`` when
#: icechunk picks up the OPERA-scoped token instead of the role (``storage/zarr_store.py``
#: documents exactly that), and a throttle on the DESTINATION bucket says ``SlowDown`` in the same
#: words the source does. The read and the write both happen inside ``write_day_windows``, so the
#: text is all there is to separate them.
#:
#: GDAL's own vocabulary separates them, because GDAL reads source imagery here and nothing else —
#: the destination is Zarr and Icechunk, and the ROI mask is read through ``da.from_zarr``. Neither
#: goes through GDAL. Unpaired, the exception is re-raised: a hole or a fault on the destination
#: fails the leg on its FIRST date, instead of being written onto the store as source data loss or
#: absorbed as somebody else's outage one date at a time.
#:
#: Deliberately only the reader's OWN vocabulary: a marker that also appears in
#: :data:`_PROVIDER_REFUSAL_MARKERS` would let one message be its own corroboration, which
#: `HTTP response code: 503` did until it was removed here.
_SOURCE_READER_MARKERS = ("RasterioIOError", "WarpOperationError", "CPLE_")

#: Refusals that are NOT the provider's to answer for, though they surface identically. Our own
#: credential is repairable here and no waiting fixes it, so these are matched FIRST and the
#: caller re-raises — failing a leg on its first date rather than its tenth.
_OWN_CREDENTIAL_MARKERS = (
    "ExpiredToken",
    "token has expired",
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
)

#: Everything neither predicate may claim, because none of it is the DATA being at fault: our own
#: credential, and a throttle. ONE list because two drifted — a chain carrying `InvalidAccessKeyId`
#: was refused by the refusal predicate of the time and accepted by `is_unreadable_source`, so the
#: same credential fault was skipped as bad data on one path and raised on the other.
_NOT_THE_DATAS_FAULT = (*_OWN_CREDENTIAL_MARKERS, "AccessDenied", "SlowDown")


def is_provider_refusal(exc: BaseException) -> bool:
    """Whether ``exc`` says the source PROVIDER refused the read.

    A refusal is a statement about the service and not about the bytes, so the only thing that
    resolves it is time — never a different copy of the object, which is what
    :func:`is_unreadable_source` gates and why that predicate declines everything matched here.
    The two are disjoint by construction, so no call order decides a verdict.

    Sits beside a retry policy rather than beside a skip. A caller passes it as that policy's
    ``wait_out`` and the failure buys patience with it; nothing may spend it on giving up a date.

    Fails closed three ways, each closed door costing only the ordinary attempt limit: our own
    credential fault is excluded first, being repairable here and unfixed by waiting; a refusal
    nothing attributes to the source reader is excluded (see :data:`_SOURCE_READER_MARKERS`),
    because every string in :data:`_PROVIDER_REFUSAL_MARKERS` belongs to S3 and the destination
    store speaks S3 too; and anything unrecognised is excluded, which is what a failure that
    reached the driver with its cause stripped looks like — it gets the short ladder and then
    fails the leg, rather than silently drawing the long wait.

    Args:
        exc: The exception a source read failed with.

    Returns:
        ``True`` when the chain names a refusal the provider owns.
    """
    return classify_read_failure(exc) is ReadFailure.PROVIDER_REFUSED


def step_down_copies(
    alternates: dict[tuple[str, str], list[Any]],
    items: Sequence[Any],
    only: Iterable[tuple[str, str]] | None = None,
    implicated: Sequence[Any] = (),
) -> tuple[list[Any], set[tuple[str, str]]] | None:
    """Swap tile-dates' copies for their next alternates, or ``None`` if none is left.

    ``alternates`` is CONSUMED: each key it steps loses the copy it hands out, so a caller
    walking the ladder repeatedly cannot re-offer a copy that already failed. Pass the same
    dict across a date's whole recovery, and never one shared with a date whose copies must
    stay untouched.

    ``only`` is the set of tile-dates a failure was ATTRIBUTED to. ``None`` means the failing
    object could not be identified, and then every duplicated tile-date in ``items`` steps together
    — the only option without attribution, and expensive twice over: on a wide ROI most of the
    date's tiles carry alternates, so one bad object downgrades hundreds that read perfectly well,
    and a bad object with no alternate of its own still walks every rung of every other tile's
    ladder before the date can be given up. An EMPTY ``only`` returns ``None`` rather than stepping
    everything: the failure was attributed, and to nothing here.

    ``implicated`` is the failed ITEMS, when the caller could identify them. A tile-date can hold
    several distinct acquisitions, and without this the rung taken is whichever alternate ranks
    highest across the whole tile-date — possibly one belonging to an acquisition that read
    perfectly well. Stepping that downgrades healthy imagery to an older baseline and leaves the
    unreadable copy selected, so the next rung has to step again: two rungs spent, one acquisition
    needlessly older. With it the ladder steps the acquisition that actually failed.

    Returns the new item list and the keys that were stepped, so a caller can name what
    changed rather than the whole date.
    """
    remaining = alternates_for(alternates, items, only=only)
    if not remaining:
        return None

    # Survivors grouped by the same key the alternates use, so an alternate can be matched
    # to the ONE item it is an alternate for.
    survivors: dict[tuple[str, str], list[Any]] = {}
    for item in items:
        tile = item_tile(item)
        if tile is not None:
            survivors.setdefault((tile, solar_day_of(item)), []).append(item)

    # BY IDENTITY, not by key. A tile-date can hold several distinct acquisitions
    # (`_by_acquisition`), and swapping on the key replaces every one of them with the same
    # alternate — turning `[a_new, b_new]` into `[a_old, a_old]`, which duplicates one acquisition
    # and silently drops the other, feeding the loader the same granule twice.
    #
    # Bucketed by the SAME key the alternates use. `implicated` carries the failed items from
    # every tile in the date, and acquisition matching is by INSTANT — so neighbouring MGRS tiles
    # imaged on one pass share an instant and cross-match. Unfiltered, a failure in tile B picks
    # tile A's spare, downgrading A while leaving B's own failure selected.
    failed_by_key: dict[tuple[str, str], list[Any]] = {}
    for item in implicated:
        tile = item_tile(item)
        if tile is not None:
            failed_by_key.setdefault((tile, solar_day_of(item)), []).append(item)

    swap: dict[int, Any] = {}
    stepped: set[tuple[str, str]] = set()
    for key, copies in remaining.items():
        alternate = _first_for_failed_acquisition(copies, failed_by_key.get(key, ()))
        if alternate is None:
            # The failed acquisition has no placeable spare. Leave every copy available and step
            # nothing: swapping a healthy acquisition cannot fix the failure and costs a re-read.
            continue
        target = _alternate_for(alternate, survivors.get(key, ()), taken=swap)
        if target is None:
            continue
        # Consumed from the ladder only once it is actually used. Removing it before knowing that
        # would drop a copy from a tile-date nothing stepped.
        alternates[key] = [c for c in copies if c is not alternate]
        swap[id(target)] = alternate
        stepped.add(key)
    if not swap:
        return None
    # The tile-dates actually STEPPED, not every one considered. The caller labels its attempt from
    # this set and records it as `copies_tried`, so naming an untouched tile-date claims a copy was
    # tried that never was — during exactly the investigation that record exists for.
    return [swap.get(id(item), item) for item in items], stepped


def _first_for_failed_acquisition(copies: list[Any], implicated: Sequence[Any]) -> Any | None:  # noqa: ANN401
    """The best-ranked alternate belonging to an acquisition that FAILED, or ``None``.

    ``copies`` is already preference-ordered, so the plain answer is ``copies[0]`` — which is what
    this returns when nothing was attributed.

    **When attribution DID name the failing objects, only candidates whose own acquisition is
    known are considered, and the answer is ``None`` if none of them belongs to the failing
    acquisition.** Preferring an alternate that shares an acquisition with a failed item is what
    keeps every other acquisition on its newest copy: otherwise a tile-date holding two
    acquisitions steps whichever has the higher-ranked spare, which can be the one that read fine,
    dropping it to an older baseline for no reason while the broken copy stays selected. And
    falling back to the best overall alternate when none matches re-read the whole date once per
    unrelated spare, failing the same way each time, before recording the loss. Nothing to step
    means nothing to step.
    """
    if not implicated:
        return copies[0]
    for candidate in copies:
        # A candidate naming neither an observation nor an instant is attached to a cluster
        # ARBITRARILY, and the two calls that need its acquisition see different item sets: this
        # one clusters it against the implicated copies, `_alternate_for` against the survivors.
        # So it could be chosen as the spare for the acquisition that FAILED and then swapped onto
        # a healthy one — consuming the spare, leaving the failure selected, recovering nothing.
        # Attributed recovery therefore only considers candidates whose acquisition is a fact.
        if not _acquisition_is_known(candidate):
            continue
        for cluster in _by_acquisition([*implicated, candidate]):
            if any(it is candidate for it in cluster) and any(any(it is bad for bad in implicated) for it in cluster):
                return candidate
    return None


def _alternate_for(alternate: Any, survivors: Iterable[Any], *, taken: dict[int, Any]) -> Any | None:  # noqa: ANN401
    """The surviving item ``alternate`` is a fallback FOR: the one sharing its acquisition.

    Decided by :func:`_by_acquisition` rather than by a second notion of sameness, so the ladder
    can only ever step a copy down onto the acquisition it came from. Falls back to the first
    un-swapped survivor when no copy names its acquisition at all, which is correct for the
    single-acquisition case that is the only way to reach it.
    """
    free = [it for it in survivors if id(it) not in taken]
    if not free:
        return None
    for cluster in _by_acquisition([*free, alternate]):
        if any(it is alternate for it in cluster):
            sibling = [it for it in cluster if it is not alternate]
            return sibling[0] if sibling else None
    return free[0]


#: How many tiles a label names before it summarises the rest. A label is read by a human
#: deciding what happened to one date; a wide ROI's date carries over a thousand tiles, and
#: naming them all costs tens of kilobytes of log per attempt while telling that reader
#: nothing they can act on.
_LABEL_CAP = 12


def copies_label(items: Iterable[Any], only: Iterable[tuple[str, str]] | None = None) -> str:
    """A compact record of which copy was used per duplicated tile, for the log.

    Sequences rather than full ids: the tile is already named alongside, and what the reader
    needs is which reprocessing was tried, on which attempt.

    ``only`` restricts the label to particular tile-dates — the ones a step-down actually
    swapped, rather than every tile in the date, which is the difference between a label a
    reader can use and one they have to diff against the previous attempt to interpret.
    """
    keys = None if only is None else set(only)
    parts = sorted(
        f"{tile}#{item_sequence(it)}"
        for it in items
        if (tile := item_tile(it)) is not None and (keys is None or (tile, solar_day_of(it)) in keys)
    )
    if not parts:
        return "none"
    if len(parts) > _LABEL_CAP:
        return ",".join(parts[:_LABEL_CAP]) + f",+{len(parts) - _LABEL_CAP} more"
    return ",".join(parts)
