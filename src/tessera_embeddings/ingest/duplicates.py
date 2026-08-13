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

The preference is **newest first**, which is the newer reprocessing and the better data in
the overwhelming majority of cases. Corruption in a newer copy is rare: sampling duplicate
pairs chosen independently of failure found every copy intact. It is also not a property of
being newer — so the older copy is a *fallback*, never a default. Callers step down the
alternates only on a demonstrated read failure.
"""

from __future__ import annotations

import datetime
import itertools
import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any

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


def _preference_key(item: Any) -> tuple[int, int, str]:  # noqa: ANN401 — any STAC-like item
    """Sort key placing the PREFERRED copy first: newest sequence, then a stable tiebreak.

    The id tiebreak is only there to make the choice deterministic when two copies claim the
    same sequence; without it the preferred copy would depend on catalogue response order,
    and a rerun could pick differently and silently produce a different mosaic.
    """
    sequence = item_sequence(item)
    known = 0 if sequence is None else 1
    return (-known, -(sequence or 0), str(getattr(item, "id", "")))


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
            ranked = sorted(acquisition, key=_preference_key)
            survivors.add(id(ranked[0]))
            rejected.extend(ranked[1:])
        if rejected:
            alternates[key] = sorted(rejected, key=_preference_key)

    kept = [it for it in items if id(it) in survivors]
    return kept, alternates


def log_duplicate_selection(
    log: logging.Logger,
    roi: str,
    alternates: dict[tuple[str, str], list[Any]],
) -> None:
    """Record that duplicates were pruned, at a level that survives a fleet-wide log.

    Summary only, and deliberately: duplicates are routine, so a line per pruned copy would
    bury the outcomes that are not routine. The tile-dates that carry alternates are named
    only when a fallback actually fires, where the identity matters.
    """
    if not alternates:
        return
    pruned = sum(len(v) for v in alternates.values())
    log.info(
        "Duplicate catalogue items pruned roi=%s: %d tile-date(s) had more than one copy, "
        "%d rejected, newest kept (rejected copies stay available as a fallback)",
        roi,
        len(alternates),
        pruned,
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

    Returns the new item list and the keys that were stepped, so a caller can name what
    changed rather than the whole date.
    """
    remaining = alternates_for(alternates, items, only=only)
    if not remaining:
        return None
    swap: dict[tuple[str, str], Any] = {}
    for key, copies in remaining.items():
        swap[key] = copies[0]
        alternates[key] = copies[1:]
    out: list[Any] = []
    for item in items:
        tile = item_tile(item)
        key = (tile, solar_day_of(item)) if tile is not None else None
        out.append(swap.get(key, item) if key is not None else item)
    return out, set(remaining)


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
