"""Choosing between duplicate catalogue items for one tile and one solar day.

Element 84's Sentinel-2 catalogue holds **more than one item per tile-date** whenever a
granule has been reprocessed: same tile, same acquisition, different processing baseline,
distinguished by ``s2:sequence``. This is common rather than exceptional — a single MGRS
tile can have dozens of duplicated dates in a single season.

Two things make that a problem rather than a curiosity.

**The loader fuses them.** ``odc.stac.load`` groups by solar day and mosaics every item in
the group, so both copies are read and one unreadable copy fails the whole date. Choosing
one copy up front is what makes a fallback possible at all: while both are in the group
there is nothing to fall back *to*.

**Their baselines differ.** One copy can sit either side of the reflectance-offset threshold
from the other, so a fused date can hold pixels processed both ways. The offset itself is now
removed from each image as it is read, before anything is fused, so a mixed date is no longer
mis-corrected — but reading two copies of one photograph to blend them into one pixel stack is
still waste, and still costs the egress of whichever copy sits in another region.

**The order lives in one place.** :func:`_preference_key` states it term by term, with the reason
for each position; add a term there and nowhere else. Two invariants govern where a new term may
sit, and neither is recoverable from the list itself.

**The losing copy is a fallback, never a default.** Among copies that are all usable, a newer
reprocessing is the better data in the overwhelming majority of cases; corruption in one is rare and
is not a property of being newer, since sampling duplicate pairs chosen independently of failure
found every copy intact. So callers step down to an alternate only on a demonstrated read failure,
never pre-emptively.

**Nothing below "will this work at all" may buy its preference with an older reprocessing.** Only
the terms deciding whether a copy can be read, and whether its correction can be decided, outrank
the baseline. Everything under it is either a claim about the PIXEL — an already-harmonised copy is
floored by its producer before resampling where one we correct is floored after, and a copy owing
nothing cannot be wrong by the offset at all — or a claim about COST, since in-region assets reach
the VPC's S3 gateway endpoint while anything else egresses through NAT. Neither kind may buy what it
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
from typing import Any

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
    form, so nothing downstream has to know which one answered.

    Planetary Computer needs its own property rather than the id fallback: its ids carry the
    tile as ``_T33TWM_`` in a later field, which the Element 84 pattern does not match, so
    without this every one of its items was unkeyable and duplicate selection was a no-op for
    the whole provider.

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
        # Matched as a sequence rather than coerced, for the reason set out in `item_baselines`:
        # `int()` truncates 1.9 to 1, and a numeric type also accepts `-1`, `1e3` and Unicode
        # digits — so something that is not a sequence would order the copies instead of deferring
        # to the one encoded in the item id.
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
    vary between items. Where it is given it is used directly, because the assets have nothing to
    add and — for a provider keying its assets natively — nothing to say.

    Otherwise the answer comes from the assets, and ``None`` means it cannot be had: an empty read
    set means the collection's configured names are not its asset keys, so nothing is found under
    them, and an incomplete read set means what IS found describes a subset. Either way
    :func:`~tessera_embeddings.ingest.asset_locations.item_harmonisation` answers ``UNKNOWN`` for
    every copy, so a term that counted that would rank them all as producer-problems and hand the
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

    **Narrower than it was.** A copy whose bands straddle a harmonised and a raw producer used to
    refuse too, because one correction was applied to a whole fused date and no single answer fitted
    such an item. The offset is now decided per SOURCE, so that copy is corrected band by band and
    is perfectly loadable. What is left is the two cases where the evidence itself is missing.
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
    so a copy that causes one must lose to a copy that does not, even an older one.

    A copy whose bands straddle two KNOWN producers is no longer among them: each of its sources is
    decided on its own bucket, so it is corrected band by band rather than refused.

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

    Owing a correction is no longer a reason to withhold a copy from anything — it costs a
    ranking position and nothing more. It ranks BELOW the baseline, because the two reasons to
    prefer a copy owing nothing are both about pixel quality and this module does not let a quality
    preference buy a better pixel with an older reprocessing. See :func:`_preference_key` term 6.

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
       because a copy missing one of them cannot deliver the tile-date at any baseline, and the
       generic paths have no recovery for it: a missing band is not one of the read failures the
       fallback ladder recognises, so an incomplete winner fails the acquisition outright.
    2. **Whether the producer is decidable**, where it would matter. A copy served from a bucket
       nobody has classified cannot have its correction decided, so it refuses its date at or above
       the correction threshold — and neither the generic path nor the read-failure ladder retries a
       refusal. A copy whose own bands span a harmonised and a raw producer is NOT one of these:
       each of its sources is decided on its own bucket, so it is corrected band by band. Below the
       threshold the term is inert, because there the producer changes no pixel.
    3. **Whether this copy demonstrably belongs to the acquisition it is ranked in.** A copy that
       names neither an observation nor an instant was attached to a cluster arbitrarily — nothing
       says which pass it came from — so it must not displace one that does say, whatever its
       baseline. Above the baseline terms deliberately: a known member at an older baseline still
       represents that pass, while a possibly-unrelated copy at a newer one may duplicate another
       pass and drop this one's coverage entirely.
    4. **Whether the baseline is readable — for producers whose correction depends on it.** A copy
       declaring nothing is one whose correction cannot be decided, so it refuses its date. Above
       the next term because a refusal has no recovery while owing a correction merely gets one.
       An already-harmonised copy is the exception: its pixels need no correction whatever the
       baseline says, so penalising it here would hand a tile-date to an OLDER raw reprocessing,
       which is the opposite of the usable-first rule this key starts with.
    5. **The baseline, descending, by value** — not by "is it the best", so every rung of the
       fallback ladder stays in descending baseline order.
    6. **Whether the copy owes an offset correction at all**, where the producer is an item's own
       property. Below the baseline, because the two reasons to prefer a copy owing nothing are both
       about the PIXEL rather than about coverage:

       - An already-harmonised copy had its floor applied by the producer *before* any resampling.
         A copy we correct ourselves is floored *after*, because the read and the reprojection are
         one step — so on the very dark population the two disagree, and the harmonised copy is the
         better of the two. See the limit recorded in
         ``context_docs/decisions/020-boa-offset-applies-to-every-valid-dn.md``.
       - A copy owing nothing cannot be wrong by the offset. One we correct is right only if the
         bucket lists and the item's declared baseline are both honest, and nothing detects it when
         they are not.

       Those are quality claims, and the module's rule is that a quality preference must not buy a
       better pixel with an older reprocessing — which is why locality sits below the baseline too.
       This sits above locality only because a pixel argument outranks a cost one. Inert below the
       threshold, where no producer changes a pixel, and inert without a read set, where every copy
       reports an undecidable producer and the term therefore ties.
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
#: It is NOT a safe margin against reprocessings, which is why identity is tried first: an earlier
#: version of this comment claimed reprocessings agree to sub-second precision, and copies of one
#: granule at baselines 02.06 and 05.00 are timestamped more than three minutes apart.
_SAME_ACQUISITION_S = 120.0

#: ``s2:datatake_id``, e.g. ``GS2B_20171219T095409_004109_N05.00``: mission, sensing start,
#: absolute orbit, then the processing baseline. Only the last field changes between
#: reprocessings, so the head of it names the observation itself.
_DATATAKE_RE = re.compile(r"(G[A-Z0-9]{2,}_[0-9]{8}T[0-9]{6}_[0-9]+)_N[0-9]{1,2}\.[0-9]{1,2}")


def acquisition_identity(item: Any) -> str | None:  # noqa: ANN401 — any STAC-like item
    """A stable name for the OBSERVATION, shared by every reprocessing of it.

    Preferred over :func:`acquisition_instant` for deciding whether two copies are the same
    acquisition. The catalogue ``datetime`` is a per-copy field and reprocessings do not agree on
    it, so a tolerance around it cannot separate "two reprocessings" from "two passes" without
    getting one of them wrong. Mission, sensing start and orbit are identical across a
    reprocessing pair and different across passes, so no tolerance is needed at all.

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
    grouping them by timestamp fused nothing and left both copies of a granule to be loaded and
    mosaicked together. The remainder falls back to the timestamps, clustered within
    :data:`_SAME_ACQUISITION_S`.

    **A copy without an identity JOINS an identified acquisition its timestamp places it in**,
    rather than always starting a cluster of its own. One reprocessing declaring the datatake while
    its sibling omits it would otherwise never be compared, however close their timestamps, so both
    would survive to be fused — the same defect identity was introduced to fix, by a different
    route.

    **A copy with neither JOINS a cluster rather than forming its own.** That makes it compete for
    one slot instead of adding one, which is what stops two copies of a pass being fused at two
    processing baselines. Collapsing the whole tile-date would prevent the same fusion by
    discarding coverage from passes that identified themselves perfectly well — this function
    exists to preserve those, so it does not.

    Which cluster such a copy joins is arbitrary, which is why it cannot WIN one
    (:func:`_preference_key` ranks a known acquisition above data vintage) and why attributed
    recovery will not place one (:func:`_first_for_failed_acquisition`).

    Splitting on a real acquisition protects real coverage: keying on (tile, solar day) alone
    dropped 493 of 2,733 items as duplicates when they were distinct acquisitions.
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
    # start one. Matched against ANY member, because members of one acquisition do not agree on the
    # timestamp — that disagreement is why identity is the primary key — so being close to any of
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
    has it; judging a fixed set instead let a preferred copy lack a requested extra asset and fail
    at load.

    ``known_harmonisation`` is the producer state the COLLECTION settles, for a collection whose
    items cannot disagree about it. Supplying it is what lets a provider serving its bands under
    native asset keys be judged at all: without it every copy reports an undecidable producer, and
    a spare that will refuse its date is offered to the fallback ladder, which cannot recover from
    a refusal.

    Items whose tile cannot be determined are passed through untouched and never grouped:
    the radar path has no such duplicates, and an item we cannot key is safer kept than
    silently dropped.

    **One per ACQUISITION, not one per day.** A tile-date can hold two genuinely different
    acquisitions as well as reprocessings of one — at high latitude, successive orbits
    revisit the same tile the same day. Those are separate imagery, mosaicked together by
    the loader, and collapsing them to one copy silently discards coverage. Measured
    against the live catalogue over eight tiles and 2021: keying on (tile, solar day) alone
    dropped **493 of 2,733 items** as duplicates when they were distinct acquisitions —
    nothing at all in the mid-latitude tiles, 196 of 500 at 33XVG and 297 of 500 at 22XER.
    So copies are split by acquisition instant first, and only true reprocessings of one
    acquisition compete on ``s2:sequence``.

    Returns:
        ``(kept, alternates)``. ``kept`` preserves the input order of the survivors, so a
        caller's own sort still decides fusion order — and now holds one item per
        acquisition, so a tile-date with two acquisitions keeps both. ``alternates`` maps
        each ``(tile, solar_day)`` that had a rejected copy to those copies, most preferred
        first — the ladder a caller steps down when a chosen copy cannot be read. It stays
        keyed by tile-date rather than by acquisition because that is the granularity a
        read failure is attributed at.
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
            # recording the date as lost.
            #
            # A spare that merely OWES the correction is now offered. It used to be withheld,
            # because the offset was decided per solar day and introducing a copy that owed one
            # could make the whole day refuse — so a swap this function could not evaluate, over
            # tiles it cannot see, was the cheaper mistake to avoid. The offset is decided per
            # image now, so such a swap changes nothing beyond its own tile and the spare is simply
            # usable. That returns the recoverable dates the exclusion was costing.
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
    bury the outcomes that are not routine. The tile-dates that carry alternates are named
    only when a fallback actually fires, where the identity matters.

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
        # difference of cardinalities cannot see it — and a tile-date holding several genuinely
        # distinct same-day passes has multiplicity above one while losing nothing, so counting
        # every multi-item key inflates the figure with dates no choice was made on.
        before = collections.Counter(k for it in supplied if (k := _contested_key(it)) is not None)
        after = collections.Counter(k for it in survivors if (k := _contested_key(it)) is not None)
        decreased = {key for key, n in before.items() if after[key] < n}
        contested_keys = decreased
        contested = len(decreased)
    else:
        # Nothing to compare against, so report the ladder — what callers passing only
        # `alternates` have always received.
        pruned = sum(len(v) for v in alternates.values())
        contested = len(alternates)
    if pruned <= 0:
        return
    recoverable = sum(len(v) for v in alternates.values())
    # Which tile-dates the source breakdown covers, taken from what pruning DID rather than from the
    # recovery ladder. A tile-date whose every spare was excluded as a refusal risk has no entry in
    # `alternates`, so keying off that dropped its winner and reported source totals for the
    # recoverable subset alone — while still labelling them as the winners.
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

    ``only`` narrows further, to the tile-dates a failure was actually ATTRIBUTED to. Without
    it the answer is every duplicated tile-date in the date, which is what a caller that
    cannot name the failing object has to step down — on a wide ROI that is most of the date,
    so one bad object downgrades hundreds of tiles that read fine, and a bad object with no
    alternate still walks every rung of every other tile's ladder before the date is given
    up. Pass the attributed keys and both costs disappear. Passing an EMPTY ``only`` is
    meaningful and distinct from ``None``: it says the failure was attributed to nothing in
    this date, so there is nothing here to step down.

    Exhausted entries are omitted rather than returned empty, so a caller can treat "no
    alternates" and "every alternate already tried" as the one condition they are.
    """
    wanted = {(tile, solar_day_of(it)) for it in items if (tile := item_tile(it)) is not None}
    if only is not None:
        wanted &= set(only)
    return {key: copies for key, copies in alternates.items() if key in wanted and copies}


#: Signatures of a source object that cannot be READ, as distinct from a transient failure.
#:
#: Matched as TEXT on the whole chain, because the read runs on a Dask worker and the failure
#: reaches the orchestrating worker through tblib, which cannot reconstruct rasterio's GDAL-backed
#: classes. What arrives is a plain ``Exception`` carrying the outer exception's **repr**.
#:
#: That repr does not include the cause, and the generic wrappers were once removed from this list
#: on the assumption that it did. It does not: measured in prod on 2026-08-24, `ZIPDecode:Decoding
#: error at scanline 0` appeared only in `dask/dask-worker` logs while the orchestrator saw exactly
#: `Exception: RasterioIOError('Read failed. See previous exception for details.')`. So the wrapper
#: is the ONLY evidence that survives the hop, and declining it did not make an unreadable object
#: fail loudly — it disabled the alternate-copy step-down that :func:`step_down_copies` exists for,
#: stranding the whole zone-year on one bad copy that a good duplicate could have replaced.
#:
#: The safety that made removing them attractive is still here, and it sits EARLIER: every refusal
#: marker and every HTTP status range is tested before this list, so a cause that IS visible still
#: decides the verdict. What remains is the genuinely blind case, and two things bound it — the
#: read has already failed :data:`SOURCE_READ_ATTEMPTS` times across about 61 s, so a transient is
#: mostly gone by here, and this `True` first tries another copy rather than giving up a date.
_UNREADABLE_MARKERS = (
    "ZIPDecode",
    "TIFFReadEncodedTile",
    # Generic block-read wrappers. Ambiguous by nature, but see above: after the hop they are all
    # that is left, and treating them as transient costs the whole cell instead of one date.
    "Chunk and warp failed",
    "Read failed. See previous exception",
    "IReadBlock failed",
)

#: Signatures of a source object that is NOT THERE — which none of the markers above can see,
#: because every one of them is emitted by a BLOCK READ and a missing object fails at open,
#: before any block is requested. A catalogue item can name an href the provider never
#: published, and no retry publishes it.
#:
#: Three forms because three layers can be the one that surfaces the condition: GDAL and
#: rasterio raise ``ObjectNotFound``, the S3 API answers with the ``NoSuchKey`` code, and the
#: response body carries the sentence.
#: How a source object that was never published READS. Two vocabularies, because two GDAL drivers
#: are in play: an `s3://` href goes through the S3 driver and says `ObjectNotFound`, while the
#: plain `https://` hrefs the optical assets actually carry go through the HTTP driver and say
#: only `HTTP response code: 404`. Matching one form and not the other is why the optical
#: step-down would not have fired on the path it was written for. Each still needs an independent
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


def is_unreadable_source(exc: BaseException) -> bool:
    """Whether ``exc`` says a source object could not be read.

    Deliberately NARROW, and it fails closed: anything unrecognised is re-raised by the
    caller rather than treated as bad data. Falling back to an older copy on a credential
    expiry or a throttle would silently swap in worse imagery to work around a fault that
    a retry fixes — and the whole point of the ladder is that stepping down is a response to
    data that will never read, not to a bad minute.

    An expired credential is explicitly excluded for that reason, even though it surfaces
    through the same wrapper.

    An object the provider never published qualifies too, and reads no differently from a
    corrupt one: no retry produces it, and another copy of the tile-date may have it. It has to
    be recognised as the SOURCE reader's failure, though, which the not-found text alone cannot
    say — see :data:`_SOURCE_READER_MARKERS`.

    Note what is NOT matched: a whole bucket or prefix being gone, which is systemic by
    definition and must fail the run on the first date rather than be skipped date by date.
    """
    text = _exception_chain_text(exc)
    if any(m in text for m in _NOT_THE_DATAS_FAULT):
        return False
    # A PROVIDER REFUSAL IS NEVER BAD DATA, whatever wrapper it arrives in. This predicate and
    # `is_provider_refusal` overlap otherwise: a refusal reaches us through a block-read wrapper,
    # so the chain carries a refusal marker AND an unreadable one, and whichever predicate the
    # caller happens to ask first decides. The radar path asks about refusals first and the
    # optical path never asks at all, so the same failure was transient for one sensor and
    # permanent data loss for the other. Excluding them here makes the two disjoint by
    # construction, so neither call order nor a caller that knows only one predicate can
    # misclassify. See context_docs/monitoring/incident-2026-08-24-nonmonotonic-append.md.
    if any(m in text for m in _PROVIDER_REFUSAL_MARKERS):
        return False
    statuses = _http_statuses(text)
    # Ranged, so a status nobody enumerated cannot be read as bad data. See the note on
    # `_HTTP_STATUS_RE` for why each range is decidable.
    if any(s >= 500 or s in _TRANSIENT_4XX for s in statuses):
        return False
    if any(400 <= s < 500 and s not in _ABSENT_4XX and s != 403 for s in statuses):
        return False
    if any(m in text for m in _UNREADABLE_MARKERS):
        return True
    return any(m in text for m in _MISSING_OBJECT_MARKERS) and any(m in text for m in _SOURCE_READER_MARKERS)


def _exception_chain_text(exc: BaseException) -> str:
    """The chain's type names and messages, joined, for text classification.

    Text rather than ``isinstance`` because the read runs on a Dask worker and is re-raised on
    the driver through tblib, which cannot rebuild rasterio's GDAL-backed classes — what arrives
    is a plain exception carrying the original's repr. The whole chain, because the wrapper
    discards the reason: ``WarpOperationError('Chunk and warp failed')`` says nothing on its own.
    """
    seen: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(seen) < 20:
        seen.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(seen)


#: Signatures of the PROVIDER refusing a read — an authorization refusal, a throttle, a server
#: error. All statements about the SERVICE rather than the bytes: the same object read moments
#: before and reads again once the service recovers.
#:
#: Consumed as an EXCLUSION by :func:`is_unreadable_source`, which is the only reader. There was
#: once a companion predicate answering "is this a refusal?" positively, so a caller could give up
#: the date and record it; nothing does that any more, because giving up a date and then
#: committing a later one puts the earlier date permanently below the store's append-only
#: maximum. A refusal is transient, so the response is to retry and — if it will not clear — to
#: fail the leg with the axis unmoved.
#:
#: The exclusion needs no :data:`_SOURCE_READER_MARKERS` corroboration, unlike the positive
#: verdict it replaced: declining is the safe direction, and a failure nothing claims is
#: re-raised either way.
_PROVIDER_REFUSAL_MARKERS = (
    "AccessDenied",
    "SlowDown",
    "ServiceUnavailable",
    "InternalError",
    "HTTP response code: 403",  # see the range note below: a provider judgement, not an HTTP one
    "HTTP response code: 500",
    # 502 and 504 for the same reason 404 is in the missing-object list: the optical assets are
    # plain `https://` hrefs, so a gateway failure reaches us as a bare status from GDAL's HTTP
    # driver. Without these it is claimed by neither predicate, and a source outage wrapped in
    # `Chunk and warp failed` is then recorded as permanently unreadable DATA — which tells an
    # operator to look for corruption when they should be waiting for the provider.
    "TooManyRequests",
    # Transport-level failures, which carry no status at all. A connection dropped mid-transfer
    # is a statement about the link, never about the bytes — the same object reads on the next
    # attempt. Without these a throttling provider's dropped reads are claimed by
    # `is_unreadable_source` through whichever block-read wrapper GDAL happens to raise, and a
    # recoverable date is recorded as permanent loss.
    "Connection reset",
    "Broken pipe",
    "timed out",
    "Connection aborted",
    "RequestTimeout",
    # The statusless rest of that class. These matter more now the block-read wrappers are claimed
    # again: a named transport failure has to be declined HERE, before the wrapper can speak for it.
    "Could not resolve host",
    "Connection refused",
    "Empty reply from server",
    "handshake failed",
)

#: GDAL's HTTP driver reports a bare status, so the numeric forms are classified by RANGE rather
#: than enumerated. Enumeration is what let 429 — the single most explicit "slow down" a provider
#: can send — be read as unreadable data: the list held 403 and 500/502/503/504 and simply missed
#: it, and a list that must be complete to be correct will be incomplete again.
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
_HTTP_STATUS_RE = re.compile(r"HTTP response code:\s*(\d{3})")
_TRANSIENT_4XX = frozenset({408, 429})
_ABSENT_4XX = frozenset({404, 410})


def _http_statuses(text: str) -> set[int]:
    """Every HTTP status the exception chain names."""
    return {int(m) for m in _HTTP_STATUS_RE.findall(text)}


#: What makes a failure the SOURCE reader's: GDAL is the layer that reported it.
#:
#: Required alongside :data:`_MISSING_OBJECT_MARKERS` by :func:`is_unreadable_source`, and
#: alongside :data:`_PROVIDER_REFUSAL_MARKERS` by :func:`is_provider_refusal`, for one reason
#: shared by both. Every string in those two tuples belongs to S3, and every S3 client in this
#: process speaks S3 — icechunk's error enum carries ``ObjectNotFound`` and ``NoSuchKey``
#: verbatim, our own store write answers ``AccessDenied`` when icechunk picks up the
#: OPERA-scoped token instead of the role (``storage/zarr_store.py`` documents exactly that),
#: and a throttle on the DESTINATION bucket says ``SlowDown`` in the same words the source does.
#: The read and the write both happen inside ``write_day_windows``, so the text is all there is
#: to separate them.
#:
#: GDAL's own vocabulary separates them, because GDAL reads source imagery here and nothing
#: else — the destination is Zarr and Icechunk, and the ROI mask is read through
#: ``da.from_zarr``. Neither goes through GDAL. Unpaired, the exception is re-raised: a hole or
#: a fault on the destination fails the leg on its FIRST date, instead of being written onto the
#: store as source data loss or absorbed as somebody else's outage one date at a time. That is
#: what happened before these marker classes existed, and it is the safe direction to fail in.
#: Names that say the SOURCE READER is what failed, as opposed to anything else in the pipeline
#: that can raise the same words. Deliberately only the reader's own vocabulary: a marker that
#: also appears in :data:`_PROVIDER_REFUSAL_MARKERS` would let one message be its own
#: corroboration, and `HTTP response code: 503` did exactly that until it was removed here.
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

#: Everything neither predicate may claim, because none of it is the DATA being at fault: our
#: own credential, and a throttle. One list, used by both, because two lists drifted — a chain
#: carrying `InvalidAccessKeyId` was refused by `is_provider_refusal` and accepted by
#: `is_unreadable_source`, so the same credential fault was skipped as bad data on one path and
#: raised on the other.
_NOT_THE_DATAS_FAULT = (*_OWN_CREDENTIAL_MARKERS, "AccessDenied", "SlowDown")


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
    object could not be identified, and then every duplicated tile-date in ``items`` steps
    together — the only option available without attribution, and expensive in two ways: on a
    wide ROI most of the date's tiles carry alternates, so one bad object downgrades hundreds
    of tiles that read perfectly well, and a bad object with no alternate of its own still
    walks every rung of every other tile's ladder before the date can be given up. An EMPTY
    ``only`` returns ``None`` rather than stepping everything: the failure was attributed,
    and to nothing here.

    ``implicated`` is the failed ITEMS, when the caller could identify them. A tile-date can
    hold several distinct acquisitions, and without this the rung taken is whichever
    alternate ranks highest across the whole tile-date — which may belong to an acquisition
    that read perfectly well. Stepping it downgrades healthy imagery to an older baseline and
    leaves the unreadable copy selected, so the next rung has to step again: two rungs spent,
    one acquisition needlessly older. With it, the ladder steps the acquisition that actually
    failed and every other acquisition keeps its newest copy.

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
    # (:func:`_by_acquisition`), and swapping on the key replaced every one of them with the
    # same alternate — turning ``[a_new, b_new]`` into ``[a_old, a_old]``, which duplicates
    # one acquisition and silently drops the other. That is worse than the coverage loss the
    # acquisition split was added to fix, because it feeds the loader the same granule twice.
    # Bucketed by the SAME key the alternates use. `implicated` carries the failed items from
    # every tile in the date, and acquisition matching is by INSTANT — so neighbouring MGRS
    # tiles imaged on one pass share an instant and cross-match. Unfiltered, a failure in tile B
    # picks tile A's spare, downgrading A while leaving B's own failure selected: the exact
    # defect this attribution was added to remove, moved one level sideways.
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

    ``copies`` is already preference-ordered, so the plain answer is ``copies[0]`` — and that
    is what this returns when nothing was attributed, which is the pre-existing behaviour.

    **When attribution DID name the failing objects, only candidates whose own acquisition is
    known are considered, and the answer is ``None`` if none of them belongs to the failing
    acquisition.** Falling back to the best overall alternate there swapped
    a healthy acquisition down to an older copy while leaving the known-bad one selected, then
    rebuilt and re-read the whole date only to fail the same way — once per unrelated spare before
    recording the loss. Nothing to step means nothing to step.

    When the caller identified the failing objects, prefer an alternate sharing an acquisition
    with one of them. Otherwise a tile-date holding two acquisitions steps whichever has the
    higher-ranked spare, which can be the one that read fine: that acquisition drops to an
    older baseline for no reason, the broken one stays selected, and the next rung has to step
    again. Preferring the failed acquisition keeps every other one on its newest copy.
    """
    if not implicated:
        return copies[0]
    for candidate in copies:
        # A candidate naming neither an observation nor an instant is attached to a cluster
        # ARBITRARILY, and the two
        # calls that need its acquisition see different item sets: this one clusters it against the
        # implicated copies, while `_alternate_for` clusters it against the survivors and lands on
        # the earliest. So it could be chosen as the spare for the acquisition that FAILED and then
        # swapped onto a healthy one — consuming the spare, leaving the failure selected, and
        # recovering nothing. Attributed recovery therefore only considers candidates whose
        # acquisition is a fact rather than a guess.
        if not _acquisition_is_known(candidate):
            continue
        for cluster in _by_acquisition([*implicated, candidate]):
            if any(it is candidate for it in cluster) and any(any(it is bad for bad in implicated) for it in cluster):
                return candidate
    return None


def _alternate_for(alternate: Any, survivors: Iterable[Any], *, taken: dict[int, Any]) -> Any | None:  # noqa: ANN401
    """The surviving item ``alternate`` is a fallback FOR: the one sharing its acquisition.

    Decided by :func:`_by_acquisition` rather than by a second notion of sameness, so the
    ladder can only ever step a copy down onto the acquisition it came from. Falls back to
    the first un-swapped survivor when no copy names its acquisition at all, which is the
    pre-acquisition-split behaviour and correct for the single-acquisition case that produced it.
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
