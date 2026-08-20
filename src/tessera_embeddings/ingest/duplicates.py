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

**Their baselines differ, and the correction is keyed by date.** One copy can sit either
side of the reflectance-offset threshold from the other, so a fused date can hold pixels
processed both ways while exactly one correction is applied to the result — decided by item
order rather than by which copy supplied the pixels.

The preference is **newest baseline first, then in-region, then newest sequence**. A newer
reprocessing is the better data in the overwhelming majority of cases, so the baseline term
leads and nothing below it may override it. Corruption in a newer copy is rare: sampling
duplicate pairs chosen independently of failure found every copy intact. It is also not a
property of being newer — so the losing copy is a *fallback*, never a default. Callers step
down the alternates only on a demonstrated read failure.

**Locality breaks ties, and only ties.** A catalogue can index the same granule from more
than one region; when it does, the copies carry the same baseline and the same acquisition
instant, so the data is identical and the only difference is what the read costs. In-region
assets go through the VPC's S3 gateway endpoint while anything else egresses through NAT.
Ranking locality *below* the baseline is what keeps this a cost decision rather than a
quality one — it can never buy cheaper egress with a worse pixel.
"""

from __future__ import annotations

import datetime
import itertools
import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any

from tessera_embeddings.ingest.asset_locations import (
    item_is_in_preferred_location,
    read_set_is_complete,
)
from tessera_embeddings.ingest.item_baselines import processing_baseline as item_processing_baseline
from tessera_embeddings.ingest.solar_days import solar_day_of

logger = logging.getLogger(__name__)

#: Trailing ``_<sequence>_L2A`` of an Element 84 item id, e.g. ``S2B_34WFA_20210908_1_L2A``.
_ID_SEQUENCE_RE = re.compile(r"_(\d+)_[A-Z0-9]+$")

#: The MGRS tile in an Element 84 item id — the second underscore-separated field.
_ID_TILE_RE = re.compile(r"^[A-Z0-9]+_([0-9]{1,2}[A-Z]{3})_")


def item_tile(item: Any) -> str | None:  # noqa: ANN401 — any STAC-like item
    """The MGRS tile an item covers, or ``None`` if it cannot be determined.

    Prefers the ``grid:code`` property and falls back to the item id. ``None`` means "do not
    treat this item as having duplicates", which is the safe answer: a wrong tile key would
    either merge two different tiles (dropping real imagery) or split one tile's duplicates
    (defeating the point).
    """
    code = item.properties.get("grid:code")
    if isinstance(code, str) and code:
        return code
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
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.debug("Unparseable s2:sequence %r on %s", raw, getattr(item, "id", "?"))
    match = _ID_SEQUENCE_RE.search(str(getattr(item, "id", "")))
    return int(match.group(1)) if match else None


def _sequence_key(item: Any) -> tuple[int, int, int, str]:  # noqa: ANN401 — any STAC-like item
    """Readable read set, then newest sequence, then a deterministic tiebreak. No locality.

    Used where the baseline comparison is suspended, so it deliberately omits both the baseline
    and the locality terms — but NOT completeness, which is orthogonal to either. A copy missing
    an asset the ingest would read cannot deliver the tile-date whatever its sequence says, and
    letting a higher-sequence incomplete copy win guaranteed the eager ``No such band/alias``
    failure. That failure names no object, so the recovery steps every duplicated tile-date of
    the day rather than this one, and can downgrade healthy imagery before ever reaching the
    complete alternate sitting right behind it.

    An item whose sequence cannot be read never displaces one whose can — an unreadable
    sequence must not win a comparison it cannot participate in. The id tiebreak makes the
    choice independent of catalogue response order, so a rerun cannot silently produce a
    different mosaic.
    """
    sequence = item_sequence(item)
    has_sequence = 0 if sequence is None else 1
    incomplete = 0 if read_set_is_complete(item) else 1
    return (incomplete, -has_sequence, -(sequence or 0), str(getattr(item, "id", "")))


def _preference_key(item: Any) -> tuple[int, float, int, int, int, int, str]:  # noqa: ANN401
    """How much we would rather read this copy than another. Lower sorts first.

    One key for both regimes, because it is CONTEXT-FREE: no term means "best in my group", so
    the same tuple orders two copies of one acquisition and two copies from different ones. An
    earlier version ranked against the group's own best baseline, which made a cross-acquisition
    comparison meaningless and needed a second key to work around — and every fix then had to be
    applied twice, which is how the read-set term reached one key and not the other.

    The fields, in order:

    1. **Whether the baseline is readable at all.** Unknown sorts last rather than being treated
       as a tie: an absent baseline is an absence of evidence, and ``_extract_baseline`` maps it
       to 0 downstream, so a copy that declares nothing is also the copy whose offset correction
       will silently be skipped.
    2. **The baseline, descending.** By VALUE, not by "is it the best": collapsing every
       non-best baseline into one tier let a read failure skip a 04.00 copy and hand out a 03.00
       one because it happened to be in region.
    3. **Locality, and only where the baseline is readable.** Ranked below the baseline so it can
       never buy cheaper egress with an older pixel, and suspended entirely for unreadable
       baselines so it cannot decide a comparison the baseline could not enter — among copies
       that all declare nothing, sequence is the only evidence there is.
    4. **Read-set completeness.** A copy missing an asset we would read cannot deliver the
       tile-date, so it loses to one that can. Below locality because locality already implies
       completeness; it is here to break the tie locality leaves, which the id term would
       otherwise settle alphabetically.
    5. **Sequence, descending, then id.** The id keeps the choice independent of catalogue
       response order, so a rerun cannot silently produce a different mosaic.
    """
    baseline = item_processing_baseline(item)
    sequence = item_sequence(item)
    return (
        0 if baseline is not None else 1,
        -(baseline or 0.0),
        (0 if item_is_in_preferred_location(item) else 1) if baseline is not None else 0,
        0 if read_set_is_complete(item) else 1,
        0 if sequence is not None else 1,
        -(sequence or 0),
        str(getattr(item, "id", "")),
    )


def _rank_copies(copies: list[Any]) -> list[Any]:
    """Order copies of one tile-date, PREFERRED first.

    **An unknown baseline suspends the locality preference entirely** and restores
    sequence-first ordering for the whole set. An absent baseline is not a tie: it is an absence
    of evidence, and treating it as one let an in-region copy with no baseline displace a remote
    copy at 05.00. That is wrong twice over — it can select an older reprocessing, and
    ``_extract_baseline`` maps the missing value to 0 downstream, so the post-04.00 offset
    correction is skipped and the reflectance is wrong with nothing raising.

    Locality therefore applies only where baselines are readable and can be compared, which is
    the overwhelming majority of real items and every case this preference was built for.
    """
    if any(item_processing_baseline(item) is None for item in copies):
        return sorted(copies, key=_sequence_key)
    return sorted(copies, key=_preference_key)


#: How far apart two acquisition instants must be to be different acquisitions rather
#: than reprocessings of one. Reprocessings carry the same instant to sub-second
#: precision; the closest genuinely distinct pair observed on the live catalogue is a
#: successive orbit ~50 minutes later, so this sits more than an order of magnitude
#: clear of both populations and no plausible catalogue jitter reaches it.
_SAME_ACQUISITION_S = 120.0


def acquisition_instant(item: Any) -> datetime.datetime | None:  # noqa: ANN401 — any STAC-like item
    """The instant an item was ACQUIRED, or ``None`` if it cannot be read.

    Read from ``properties["datetime"]`` and never from ``item.datetime``, because by
    the time duplicates are selected the latter has been overwritten with the canonical
    noon-UTC solar-day stamp (:func:`normalize_to_solar_day`) and every copy of a day
    therefore carries an identical value. The property is the only surviving record of
    which acquisition a copy came from.
    """
    raw = item.properties.get("datetime")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("Unparseable datetime %r on %s", raw, getattr(item, "id", "?"))
        return None


def _by_acquisition(copies: list[Any]) -> list[list[Any]]:
    """Split one tile-date's copies into one list per distinct acquisition.

    Copies whose instant cannot be read stay together in a single group, which is the
    pre-existing behaviour and the conservative one: without an instant there is no
    evidence they are distinct, and treating them as distinct would keep duplicate
    reprocessings and hand the loader exactly the fused-baseline problem this module
    exists to prevent.
    """
    dated = [(inst, it) for it in copies if (inst := acquisition_instant(it)) is not None]
    undated = [it for it in copies if acquisition_instant(it) is None]
    if not dated:
        return [copies]

    dated.sort(key=lambda pair: pair[0])
    clusters: list[list[Any]] = [[dated[0][1]]]
    for previous, current in itertools.pairwise(dated):
        if (current[0] - previous[0]).total_seconds() > _SAME_ACQUISITION_S:
            clusters.append([current[1]])
        else:
            clusters[-1].append(current[1])
    if undated:
        clusters.append(undated)
    return clusters


def select_preferred_duplicates(
    items: Sequence[Any],
) -> tuple[list[Any], dict[tuple[str, str], list[Any]]]:
    """Keep one item per (tile, solar day); return the survivors and the alternates.

    Items must already be normalised to solar days — the grouping key is the solar day, so
    an un-normalised item would be grouped under the wrong one. :func:`solar_day_of` raises
    rather than guessing, which is what enforces the ordering.

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
            ranked = _rank_copies(acquisition)
            survivors.add(id(ranked[0]))
            rejected.extend(ranked[1:])
        if rejected:
            # ONE global order over every rejected copy, by a context-free key.
            #
            # `_rank_copies` cannot build this list. It suspends the baseline comparison whenever
            # any baseline in its input is unreadable, which is right for picking an
            # acquisition's winner and wrong here: applied to a whole tile-date, one malformed
            # item reverted every acquisition's ladder to sequence order, so a read failure could
            # hand out a 03.00 copy while a 04.00 one sat further down.
            #
            # Ranking each acquisition separately and concatenating is also wrong, and by a
            # subtler route: it orders the ladder heads and then flattens, so with A holding
            # 05.00 and 01.00 spares and B holding 04.00 it yields [A05, A01, B04]. The
            # unattributed recovery in `s2_roi` consumes `copies[0]` on each retry, so the second
            # retry takes A01 and skips the better B04 — degrading imagery to avoid nothing.
            #
            # A single sort by `_preference_key` is the priority-preserving merge those
            # comments ask for, with the same comparator applied at every position rather than
            # only at the heads. Nothing is suspended, so no acquisition can leak its metadata
            # into another's order, and an unknown baseline sorts LAST instead of suspending the
            # comparison — which addresses the same risk more directly, since a copy whose
            # baseline cannot be read is the one whose correction will silently be skipped.
            alternates[key] = sorted(rejected, key=_preference_key)

    kept = [it for it in items if id(it) in survivors]
    return kept, alternates


def _contested_key(item: Any) -> tuple[str, str] | None:  # noqa: ANN401 — any STAC-like item
    """An item's ``(tile, solar_day)`` key, or ``None`` if it cannot be keyed.

    Returns ``None`` rather than raising because the only caller is a log line, and
    :func:`solar_day_of` raises on an item that has not been normalised. A logging call that
    can abort an ingest is worse than a log line that omits a figure.
    """
    tile = item_tile(item)
    if tile is None:
        return None
    try:
        return (tile, solar_day_of(item))
    except Exception:
        logger.debug("Could not key %s for the duplicate audit line", getattr(item, "id", "?"))
        return None


def _keys_ranked_by_sequence_only(
    alternates: dict[tuple[str, str], list[Any]],
    winners: Iterable[Any],
) -> set[tuple[str, str]]:
    """The contested tile-dates whose ranking fell back to sequence order.

    :func:`_rank_copies` suspends the baseline and locality terms for an acquisition in which any
    copy's baseline is unreadable, so the audit line cannot claim a baseline-first preference for
    those dates. Recomputed here rather than threaded out of the selector, because it is needed
    only to word one log line and a return-value change would reach every caller.
    """
    unreadable: set[tuple[str, str]] = set()
    for key, copies in alternates.items():
        if any(item_processing_baseline(it) is None for it in copies):
            unreadable.add(key)
    for winner in winners:
        if item_processing_baseline(winner) is None and (winner_key := _contested_key(winner)) is not None:
            unreadable.add(winner_key)
    return unreadable & set(alternates)


def log_duplicate_selection(
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    roi: str,
    alternates: dict[tuple[str, str], list[Any]],
    kept: Iterable[Any] = (),
) -> None:
    """Record that duplicates were pruned, at a level that survives a fleet-wide log.

    Summary only, and deliberately: duplicates are routine, so a line per pruned copy would
    bury the outcomes that are not routine. The tile-dates that carry alternates are named
    only when a fallback actually fires, where the identity matters.

    **Reports WHERE the surviving copies came from, because the preference is no longer purely
    "newest".** This line said "newest kept" while the ranking preferred an in-region copy over
    a higher-sequence remote one — a log misreporting its own behaviour. It is also the only
    audit trail for that decision: where two copies carry the same baseline their pixels are
    identical, so nothing downstream can show which was read, and the choice is otherwise
    invisible in the store, the mosaic and the metrics.

    **Counted over the CONTESTED tile-dates only.** ``kept`` is every survivor on the ROI, and
    on a wide one almost none of them had a duplicate at all. Counting them all reported the
    composition of the whole supply — "1 in-region, 100 remote" — while saying nothing about
    the choices this line exists to record, and the one figure that matters was rounded away by
    the hundred dates that were never in question.
    """
    if not alternates:
        return
    pruned = sum(len(v) for v in alternates.values())
    winners = [it for it in kept if _contested_key(it) in alternates]
    where = ""
    if winners:
        local = sum(1 for it in winners if item_is_in_preferred_location(it))
        where = f"; winners by source: {local} in-region, {len(winners) - local} remote"

    # Name the ranking that actually ran. `_rank_copies` abandons BOTH the baseline and the
    # locality terms for an acquisition where any copy declares no readable baseline, and
    # chooses on sequence alone — so on exactly the uncertain-metadata dates this line exists to
    # explain, stating the baseline-then-locality preference described the wrong mechanism.
    unreadable = _keys_ranked_by_sequence_only(alternates, winners)
    how = "newest baseline, then in-region, then newest sequence"
    if unreadable:
        how += (
            f" — except {len(unreadable)} tile-date(s) where a copy declared no readable "
            f"baseline, ranked on sequence alone"
        )
    log.info(
        "Duplicate catalogue items pruned roi=%s: %d tile-date(s) had more than one copy, "
        "%d rejected. Preference: %s (rejected copies stay available as a fallback)%s",
        roi,
        len(alternates),
        pruned,
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
#: Matched on the whole exception chain's text because the informative part is the cause:
#: rasterio's ``WarpOperationError('Chunk and warp failed')`` is a wrapper that discards the
#: reason, and the reason — a codec that cannot inflate a tile — is what says the object is
#: broken rather than briefly unavailable.
_UNREADABLE_MARKERS = (
    "ZIPDecode",
    "TIFFReadEncodedTile",
    "IReadBlock failed",
    "Chunk and warp failed",
    "Read failed. See previous exception",
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
    """
    seen: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(seen) < 20:
        seen.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    text = " | ".join(seen)
    if any(m in text for m in ("ExpiredToken", "token has expired", "AccessDenied", "SlowDown")):
        return False
    return any(m in text for m in _UNREADABLE_MARKERS)


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
    for key, copies in remaining.items():
        alternate = _first_for_failed_acquisition(copies, failed_by_key.get(key, ()))
        alternates[key] = [c for c in copies if c is not alternate]
        target = _alternate_for(alternate, survivors.get(key, ()), taken=swap)
        if target is not None:
            swap[id(target)] = alternate
    if not swap:
        return None
    return [swap.get(id(item), item) for item in items], set(remaining)


def _first_for_failed_acquisition(copies: list[Any], implicated: Sequence[Any]) -> Any:  # noqa: ANN401
    """The best-ranked alternate belonging to an acquisition that FAILED, else the best overall.

    ``copies`` is already preference-ordered, so the plain answer is ``copies[0]`` — and that
    is what this returns when nothing was attributed, which is the pre-existing behaviour.

    When the caller identified the failing objects, prefer an alternate sharing an acquisition
    with one of them. Otherwise a tile-date holding two acquisitions steps whichever has the
    higher-ranked spare, which can be the one that read fine: that acquisition drops to an
    older baseline for no reason, the broken one stays selected, and the next rung has to step
    again. Preferring the failed acquisition keeps every other one on its newest copy.
    """
    if not implicated:
        return copies[0]
    for candidate in copies:
        for cluster in _by_acquisition([*implicated, candidate]):
            if any(it is candidate for it in cluster) and any(any(it is bad for bad in implicated) for it in cluster):
                return candidate
    return copies[0]


def _alternate_for(alternate: Any, survivors: Iterable[Any], *, taken: dict[int, Any]) -> Any | None:  # noqa: ANN401
    """The surviving item ``alternate`` is a fallback FOR: the one sharing its acquisition.

    Decided by :func:`_by_acquisition` rather than by a second notion of sameness, so the
    ladder can only ever step a copy down onto the acquisition it came from. Falls back to
    the first un-swapped survivor when the acquisition instants cannot be read at all, which
    is the pre-acquisition-split behaviour and correct for the single-acquisition case that
    produced it.
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
