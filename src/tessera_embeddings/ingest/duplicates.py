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


def _would_refuse_its_date(
    item: Any,  # noqa: ANN401
    baseline: int | None,
    read_keys: tuple[str, ...],
) -> bool:
    """Whether choosing this otherwise-usable copy makes its date refuse rather than load.

    :func:`~tessera_embeddings.ingest.stac.dates_exempt_from_correction` refuses a date at or above
    the threshold whose item straddles two producers, or whose producer cannot be identified,
    because no date-wide offset decision is correct for it. Nothing retries a refusal — the
    read-failure ladder steps down on a read error, not on this — so a copy that causes one must
    lose to a copy that does not, even an older one.

    Only asked of copies whose read set is complete. An incomplete copy is already ranked last, and
    it reports an undecidable producer *because* it is incomplete, so counting it here would say the
    same thing twice and invert the baseline preference among copies that are all incomplete.
    """
    if baseline is None or baseline < S2_BASELINE_THRESHOLD:
        return False
    if not read_set_is_complete(item, read_keys):
        return False
    return item_harmonisation(item) in (Harmonisation.MIXED, Harmonisation.UNKNOWN)


def _preference_key(
    item: Any,  # noqa: ANN401
    read_keys: tuple[str, ...] = READ_ASSET_KEYS,
) -> tuple[int, int, int, float, int, int, int, str]:
    """How much we would rather read this copy than another. Lower sorts first.

    **Context-free:** no term is relative to the set being sorted, so the same key orders copies
    of one acquisition and copies from different acquisitions. That is what lets one key serve
    both, and it must hold for any term added here.

    The fields, in order:

    1. **Read-set completeness**, over ``read_keys`` — the assets THIS load will request. First,
       because a copy missing one of them cannot deliver the tile-date at any baseline, and the
       generic paths have no recovery for it: a missing band is not one of the read failures the
       fallback ladder recognises, so an incomplete winner fails the acquisition outright.
    2. **Whether the producer is decidable**, where it would matter. A copy whose own reflectance
       bands span a harmonised and a raw producer, or whose producer cannot be identified at all,
       refuses its date at or above the correction threshold — and neither the generic path nor the
       read-failure ladder retries a refusal. Below the threshold the term is inert, because there
       the producer changes no pixel.
    3. **Whether the baseline is readable.** Unknown sorts last, for the same reason: a copy that
       declares nothing is one whose correction cannot be decided, so it refuses its date too.
    4. **The baseline, descending, by value** — not by "is it the best", so every rung of the
       fallback ladder stays in descending baseline order.
    5. **Locality, only where the baseline is readable.** Below the baseline so it cannot buy
       cheaper egress with an older pixel, and inert for unreadable baselines so it cannot decide
       a comparison the baseline could not enter.
    6. **Sequence, descending, then id.** The id makes the order total, so the choice is
       independent of catalogue response order and a rerun cannot produce a different mosaic.
    """
    baseline = item_processing_baseline(item)
    sequence = item_sequence(item)
    return (
        0 if read_set_is_complete(item, read_keys) else 1,
        1 if _would_refuse_its_date(item, baseline, read_keys) else 0,
        0 if baseline is not None else 1,
        -(baseline or 0.0),
        (0 if item_is_in_preferred_location(item, keys=read_keys) else 1) if baseline is not None else 0,
        0 if sequence is not None else 1,
        -(sequence or 0),
        str(getattr(item, "id", "")),
    )


def _rank_copies(copies: list[Any], read_keys: tuple[str, ...] = READ_ASSET_KEYS) -> list[Any]:
    """Order copies of one acquisition, PREFERRED first.

    An unreadable baseline ranks last rather than suspending the comparison: such a copy refuses
    its date in :func:`~tessera_embeddings.ingest.stac.dates_exempt_from_correction`, and the
    read-failure ladder recovers from a read error but not from a refusal. A reprocessing that can
    be corrected beats a newer one that cannot be processed at all.
    """
    return sorted(copies, key=functools.partial(_preference_key, read_keys=read_keys))


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

    **If ANY copy's instant cannot be read, the whole tile-date is one acquisition.** An
    unreadable instant is no evidence that the copy is a distinct pass, and the two errors are not
    symmetric: giving it its own group leaves two copies to be fused at two processing baselines,
    which is what this module prevents, while collapsing at worst discards a genuinely distinct
    pass.

    Where instants ARE readable, splitting on them protects real coverage: keying on (tile, solar
    day) alone dropped 493 of 2,733 items as duplicates when they were distinct acquisitions.
    """
    instants = [acquisition_instant(it) for it in copies]
    if any(instant is None for instant in instants):
        return [copies]

    # Narrowed by the guard above; restated for the type checker rather than cast.
    known: list[tuple[datetime.datetime, Any]] = [
        (instant, item) for instant, item in zip(instants, copies, strict=True) if instant is not None
    ]
    dated = sorted(known, key=lambda pair: pair[0])
    clusters: list[list[Any]] = [[dated[0][1]]]
    for previous, current in itertools.pairwise(dated):
        if (current[0] - previous[0]).total_seconds() > _SAME_ACQUISITION_S:
            clusters.append([current[1]])
        else:
            clusters[-1].append(current[1])
    return clusters


def select_preferred_duplicates(
    items: Sequence[Any],
    read_keys: tuple[str, ...] = READ_ASSET_KEYS,
) -> tuple[list[Any], dict[tuple[str, str], list[Any]]]:
    """Keep one item per (tile, solar day); return the survivors and the alternates.

    Items must already be normalised to solar days — the grouping key is the solar day, so
    an un-normalised item would be grouped under the wrong one. :func:`solar_day_of` raises
    rather than guessing, which is what enforces the ordering.

    ``read_keys`` is the asset set THIS load will request, extra bands included. Readability and
    locality are judged over it, so a copy missing a band the caller asked for loses to one that
    has it; judging a fixed set instead let a preferred copy lack a requested extra asset and fail
    at load.

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
            ranked = _rank_copies(acquisition, read_keys)
            survivors.add(id(ranked[0]))
            rejected.extend(ranked[1:])
        if rejected:
            # ONE order over every rejected copy of the tile-date, not a concatenation of
            # per-acquisition ladders. The unattributed recovery consumes `copies[0]` on each
            # retry, so every position is a choice and the whole list has to be ranked.
            alternates[key] = sorted(rejected, key=functools.partial(_preference_key, read_keys=read_keys))

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
    if not alternates:
        return
    pruned = sum(len(v) for v in alternates.values())
    winners = [it for it in kept if _contested_key(it) in alternates]
    where = ""
    if winners:
        local = sum(1 for it in winners if item_is_in_preferred_location(it, keys=read_keys))
        where = f"; winners by source: {local} in-region, {len(winners) - local} remote"

    how = "newest baseline, then in-region, then complete read set, then newest sequence"
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
