"""Duplicate catalogue items: prefer the newest copy, fall back only on unreadable data.

Element 84's Sentinel-2 catalogue carries more than one item per tile-date whenever a
granule has been reprocessed, and it is common — dozens of duplicated dates in one season
for a single tile. The loader FUSES a solar-day group, so both copies are read and one
unreadable copy fails the whole date; reducing to one copy up front is what makes a
fallback possible at all.

Preference is newest-first because that is the newer reprocessing and, in an unbiased
sample of duplicate pairs, every copy read fine. Corruption is not a property of being
newer, so the older copy is a fallback and never a default.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pystac
import pytest

from tessera_embeddings.ingest.asset_locations import (
    PREFERRED_ASSET_BUCKETS,
    READ_ASSET_KEYS,
    asset_bucket,
    item_is_in_preferred_location,
    read_set_is_complete,
)
from tessera_embeddings.ingest.duplicates import (
    _preference_key,
    alternates_for,
    copies_label,
    is_unreadable_source,
    item_processing_baseline,
    item_sequence,
    item_tile,
    select_preferred_duplicates,
    step_down_copies,
)
from tessera_embeddings.ingest.solar_days import normalize_to_solar_day


class _Item:
    """Stands in for a pystac Item, carrying only what the selector reads."""

    def __init__(self, ident: str, tile: str | None = None, sequence: str | None = None, **extra: object) -> None:
        self.id = ident
        # Every item reaching the selector has passed normalize_to_solar_day, whose canonical
        # stamp is noon UTC; solar_day_of RAISES on anything else, so the fixture must match.
        self.datetime = datetime(2021, 9, 8, 12, 0, 0, tzinfo=UTC)
        self.properties: dict[str, object] = dict(extra)
        if tile is not None:
            self.properties["grid:code"] = tile
        if sequence is not None:
            self.properties["s2:sequence"] = sequence


def _pair() -> tuple[_Item, _Item]:
    """The real shape: two copies of one tile-date, older first as the catalogue returns."""
    old = _Item("S2B_34WFA_20210908_0_L2A", "MGRS-34WFA", "0", **{"s2:processing_baseline": "03.01"})
    new = _Item("S2B_34WFA_20210908_1_L2A", "MGRS-34WFA", "1", **{"s2:processing_baseline": "05.00"})
    return old, new


class TestAgainstRealPystacItems:
    """The claim that `normalize_to_solar_day` destroys the acquisition instant, settled.

    Reviewers have raised this four times: that assigning `pystac.Item.datetime` writes
    through to `properties["datetime"]`, so `acquisition_instant` reads canonical noon for
    every copy and distinct same-day passes collapse into one reprocessing group — real
    imagery silently dropped at high latitudes, where successive orbits overlap.

    It is a good failure to worry about and it is not what pystac does: `datetime` is a
    plain instance attribute with no descriptor behind it, so the assignment leaves
    `properties` untouched. Every other test in this file uses a double, which is exactly
    the objection someone could raise next — so this one drives REAL `pystac.Item`s
    through the real `normalize_to_solar_day` and asserts on the outcome, not the mechanism.
    """

    @staticmethod
    def _real(ident: str, tile: str, sequence: str, acquired: str) -> pystac.Item:
        return pystac.Item(
            id=ident,
            geometry=None,
            bbox=None,
            datetime=datetime.fromisoformat(acquired.replace("Z", "+00:00")),
            properties={"grid:code": tile, "s2:sequence": sequence, "datetime": acquired},
        )

    def test_normalising_leaves_the_acquisition_instant_in_properties(self) -> None:
        item = self._real("S2A_33XVG_20210908_0_L2A", "MGRS-33XVG", "0", "2021-09-08T10:20:31.024000Z")
        normalize_to_solar_day([item], mid_longitude=15.0)

        assert item.datetime.hour == 12, "the solar-day stamp must be applied to .datetime"
        assert item.properties["datetime"] == "2021-09-08T10:20:31.024000Z", (
            "pystac.Item.datetime is a plain attribute — assigning it must NOT write through"
        )

    def test_two_real_same_day_acquisitions_both_survive_the_whole_pipeline(self) -> None:
        """End to end on real objects: normalise, then select. Neither may be dropped."""
        first = self._real("S2A_33XVG_20210908_0_L2A", "MGRS-33XVG", "0", "2021-09-08T10:20:31.024000Z")
        second = self._real("S2B_33XVG_20210908_0_L2A", "MGRS-33XVG", "0", "2021-09-08T11:10:14.512000Z")
        normalize_to_solar_day([first, second], mid_longitude=15.0)

        kept, alternates = select_preferred_duplicates([first, second])
        assert kept == [first, second], "two distinct passes were collapsed into one"
        assert alternates == {}

    def test_real_reprocessings_of_one_acquisition_still_reduce(self) -> None:
        """The other direction, so the test above cannot pass by never grouping anything."""
        old = self._real("S2B_33XVG_20210908_0_L2A", "MGRS-33XVG", "0", "2021-09-08T10:20:31.024000Z")
        new = self._real("S2B_33XVG_20210908_1_L2A", "MGRS-33XVG", "1", "2021-09-08T10:20:31.024000Z")
        normalize_to_solar_day([old, new], mid_longitude=15.0)

        kept, alternates = select_preferred_duplicates([old, new])
        assert kept == [new]
        assert alternates == {("MGRS-33XVG", "2021-09-08"): [old]}


class TestIdentifyingACopy:
    """Keying a copy: which tile it covers, and which reprocessing it is."""

    def test_the_tile_comes_from_the_property(self) -> None:
        assert item_tile(_Item("x", "MGRS-34WFA", "1")) == "MGRS-34WFA"

    def test_the_tile_falls_back_to_the_id(self) -> None:
        """A provider that omits grid:code must not silently defeat the grouping."""
        assert item_tile(_Item("S2B_34WFA_20210908_1_L2A")) == "MGRS-34WFA"

    def test_an_unkeyable_item_returns_none(self) -> None:
        """``None`` means "no duplicates here" — the safe answer, since a wrong key either
        merges two tiles or splits one tile's copies.
        """
        assert item_tile(_Item("not-a-sentinel-id")) is None

    def test_the_sequence_is_read_as_an_integer(self) -> None:
        """The property is a STRING in the real catalogue, so ordering it as text would put
        '10' before '9'.
        """
        assert item_sequence(_Item("x", "MGRS-1ABC", "10")) == 10

    def test_the_sequence_falls_back_to_the_id(self) -> None:
        assert item_sequence(_Item("S2B_34WFA_20210908_2_L2A")) == 2

    def test_an_unreadable_sequence_is_none(self) -> None:
        assert item_sequence(_Item("no-sequence-here", "MGRS-1ABC")) is None


class TestChoosingBetweenCopies:
    """Reducing a tile-date to one copy, deterministically and without losing imagery."""

    def test_the_newest_copy_is_kept(self) -> None:
        old, new = _pair()
        kept, alternates = select_preferred_duplicates([old, new])
        assert [it.id for it in kept] == [new.id]
        assert [it.id for it in alternates[("MGRS-34WFA", "2021-09-08")]] == [old.id]

    def test_catalogue_order_does_not_decide(self) -> None:
        """The newest must win whichever order the search returned them in."""
        old, new = _pair()
        for order in ([old, new], [new, old]):
            kept, _ = select_preferred_duplicates(order)
            assert [it.id for it in kept] == [new.id]

    def test_a_third_copy_orders_the_whole_ladder(self) -> None:
        """Fallback must step down in preference order, not arbitrarily.

        `_pair()` declares baselines 03.01 and 05.00; the third copy declares none, so it now
        ranks BELOW both. A copy whose baseline cannot be read refuses its whole date downstream,
        and the read-failure ladder cannot recover from a refusal — so an older reprocessing that
        can be corrected correctly beats a newer one that cannot be processed at all.
        """
        old, new = _pair()
        no_baseline = _Item("S2B_34WFA_20210908_2_L2A", "MGRS-34WFA", "2")
        kept, alternates = select_preferred_duplicates([old, new, no_baseline])
        assert [it.id for it in kept] == [new.id]
        assert [it.id for it in alternates[("MGRS-34WFA", "2021-09-08")]] == [old.id, no_baseline.id]

    def test_different_tiles_are_never_merged(self) -> None:
        """Two tiles imaged the same day are not duplicates — merging them drops imagery."""
        a = _Item("S2B_34WFA_20210908_0_L2A", "MGRS-34WFA", "0")
        b = _Item("S2B_34WFB_20210908_0_L2A", "MGRS-34WFB", "0")
        kept, alternates = select_preferred_duplicates([a, b])
        assert {it.id for it in kept} == {a.id, b.id}
        assert alternates == {}

    def test_a_lone_copy_produces_no_alternates(self) -> None:
        """The overwhelmingly common case must add no state to step down."""
        _, new = _pair()
        kept, alternates = select_preferred_duplicates([new])
        assert [it.id for it in kept] == [new.id]
        assert alternates == {}

    def test_input_order_is_preserved_among_survivors(self) -> None:
        """The caller sorts cloudiest-first for the painter's-algorithm mosaic; the selector
        must not reorder that or the clearest tile stops painting last.
        """
        first = _Item("S2B_11VPD_20210908_0_L2A", "MGRS-11VPD", "0")
        old, new = _pair()
        last = _Item("S2B_34WFB_20210908_0_L2A", "MGRS-34WFB", "0")
        kept, _ = select_preferred_duplicates([first, old, new, last])
        assert [it.id for it in kept] == [first.id, new.id, last.id]

    def test_an_unkeyable_item_survives_untouched(self) -> None:
        """The radar path has no such duplicates; an item we cannot key must not be dropped."""
        odd = _Item("something-else")
        kept, alternates = select_preferred_duplicates([odd])
        assert [it.id for it in kept] == [odd.id]
        assert alternates == {}

    def test_an_unreadable_sequence_never_displaces_a_known_one(self) -> None:
        """An item that cannot state its sequence must not win a comparison it can't enter."""
        known = _Item("S2B_34WFA_20210908_1_L2A", "MGRS-34WFA", "1")
        unknown = _Item("weird", "MGRS-34WFA")
        kept, _ = select_preferred_duplicates([unknown, known])
        assert [it.id for it in kept] == [known.id]


class TestTheFallbackLadder:
    """What a caller steps down when the chosen copy will not read."""

    def test_alternates_are_scoped_to_the_date_being_retried(self) -> None:
        """Stepping down another date's copies would swap imagery for a day that read fine."""
        old, new = _pair()
        alternates = {
            ("MGRS-34WFA", "2021-09-08"): [old],
            ("MGRS-99ZZZ", "2021-09-08"): [_Item("other", "MGRS-99ZZZ", "0")],
        }
        scoped = alternates_for(alternates, [new])
        assert list(scoped) == [("MGRS-34WFA", "2021-09-08")]

    def test_an_exhausted_ladder_is_reported_as_absent(self) -> None:
        """An exhausted ladder and an absent one are ONE condition for the caller."""
        _, new = _pair()
        assert alternates_for({("MGRS-34WFA", "2021-09-08"): []}, [new]) == {}

    def test_the_label_names_the_copy_per_tile(self) -> None:
        """The log needs which reprocessing was tried on which attempt."""
        old, new = _pair()
        assert copies_label([new]) == "MGRS-34WFA#1"
        assert copies_label([old]) == "MGRS-34WFA#0"


class TestWhenToStepDown:
    """The predicate decides whether a failure means "this data will never read".

    Falling back on a *transient* failure would silently swap in older imagery to work around
    something a retry fixes. So it is narrow and fails closed: anything unrecognised is
    re-raised by the caller rather than treated as bad data.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "ZIPDecode:Decoding error at scanline 0",
            "B02.tif, band 1: IReadBlock failed at X offset 6, Y offset 9: TIFFReadEncodedTile() failed.",
            "Read failed. See previous exception for details.",
        ],
    )
    def test_a_decode_failure_means_step_down(self, message: str) -> None:
        assert is_unreadable_source(RuntimeError(message))

    def test_the_cause_chain_is_searched_not_just_the_message(self) -> None:
        """The exception that propagates is a wrapper that DISCARDS the reason.

        ``WarpOperationError('Chunk and warp failed')`` says nothing about why; the decode
        error is its cause. A predicate reading only the top message would classify every
        cause identically.
        """
        try:
            try:
                raise ValueError("ZIPDecode:Decoding error at scanline 2048")
            except ValueError as cause:
                raise RuntimeError("something opaque") from cause
        except RuntimeError as exc:
            assert is_unreadable_source(exc)

    @pytest.mark.parametrize(
        "message",
        [
            "An error occurred (ExpiredToken) when calling the GetObject operation",
            "The provided token has expired.",
            "An error occurred (AccessDenied)",
            "An error occurred (SlowDown) when calling the PutObject operation",
        ],
    )
    def test_a_credential_or_throttle_failure_must_not_step_down(self, message: str) -> None:
        """These are transient and fixed by a retry. Stepping down here would degrade the
        data to route around a fault that was about to clear.
        """
        assert not is_unreadable_source(RuntimeError(message))

    def test_a_credential_failure_wrapped_in_the_decode_wrapper_still_does_not_step_down(self) -> None:
        """The dangerous case: an expired credential surfaces through the SAME wrapper as
        corruption, because the loader hands the codec a buffer the failed read never filled.
        Exclusion has to beat the marker, or every credential blip degrades a date's imagery.
        """
        try:
            try:
                raise ValueError("An error occurred (ExpiredToken) when calling the GetObject operation")
            except ValueError as cause:
                raise RuntimeError("Chunk and warp failed") from cause
        except RuntimeError as exc:
            assert not is_unreadable_source(exc)

    def test_an_unrelated_failure_does_not_step_down(self) -> None:
        assert not is_unreadable_source(ValueError("some unrelated bug"))


class TestDistinctAcquisitionsOnOneDay:
    """A tile-date can hold two ACQUISITIONS, not just two copies of one.

    At high latitude successive orbits revisit a tile the same day. Those are separate
    imagery that the loader mosaics together, so collapsing them to one copy discards
    coverage silently. Measured on the live catalogue over eight tiles across 2021,
    keying on (tile, solar day) alone discarded 493 of 2,733 items as duplicates when
    they were distinct acquisitions — none in the mid-latitude tiles, 196 of 500 at
    33XVG and 297 of 500 at 22XER.
    """

    @staticmethod
    def _at(ident: str, acquired: str, sequence: str = "0") -> _Item:
        """A copy carrying its ACQUISITION instant, as the catalogue supplies it.

        ``properties["datetime"]`` and not ``item.datetime``: normalize_to_solar_day has
        already overwritten the latter with the canonical noon stamp by this point, so the
        property is the only surviving record of which acquisition a copy came from.
        """
        return _Item(ident, "MGRS-33XVG", sequence, **{"datetime": acquired})

    def test_two_acquisitions_on_one_day_both_survive(self) -> None:
        first = self._at("S2A_33XVG_20210908_0_L2A", "2021-09-08T10:20:31.024000Z")
        second = self._at("S2B_33XVG_20210908_0_L2A", "2021-09-08T11:10:14.512000Z")
        kept, alternates = select_preferred_duplicates([first, second])
        assert kept == [first, second]
        assert alternates == {}

    def test_reprocessings_of_one_acquisition_still_reduce_to_one(self) -> None:
        """The behaviour the module exists for, unchanged: same instant, different
        sequence, newest kept and the older offered as the fallback.
        """
        old = self._at("S2B_33XVG_20210908_0_L2A", "2021-09-08T10:20:31.024000Z", "0")
        new = self._at("S2B_33XVG_20210908_1_L2A", "2021-09-08T10:20:31.024000Z", "1")
        kept, alternates = select_preferred_duplicates([old, new])
        assert kept == [new]
        assert alternates == {("MGRS-33XVG", "2021-09-08"): [old]}

    def test_each_acquisition_is_deduplicated_on_its_own(self) -> None:
        """Both at once — two acquisitions, each reprocessed. One survivor per
        acquisition, and both rejected copies reachable through the ladder.
        """
        a_old = self._at("S2A_33XVG_20210908_0_L2A", "2021-09-08T10:20:31.024000Z", "0")
        a_new = self._at("S2A_33XVG_20210908_1_L2A", "2021-09-08T10:20:31.024000Z", "1")
        b_old = self._at("S2B_33XVG_20210908_0_L2A", "2021-09-08T11:10:14.512000Z", "0")
        b_new = self._at("S2B_33XVG_20210908_1_L2A", "2021-09-08T11:10:14.512000Z", "1")
        kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
        assert kept == [a_new, b_new]
        assert set(alternates[("MGRS-33XVG", "2021-09-08")]) == {a_old, b_old}

    def test_sub_second_jitter_is_one_acquisition(self) -> None:
        """Reprocessings differ in the microseconds of an otherwise identical instant.
        Splitting on that would keep every reprocessing and defeat the module.
        """
        old = self._at("S2B_33XVG_20210908_0_L2A", "2021-09-08T10:20:31.024000Z", "0")
        new = self._at("S2B_33XVG_20210908_1_L2A", "2021-09-08T10:20:31.026000Z", "1")
        kept, _ = select_preferred_duplicates([old, new])
        assert kept == [new]

    def test_copies_without_an_instant_keep_the_old_behaviour(self) -> None:
        """No instant is no evidence of distinctness, so they compete as before —
        which is what keeps a catalogue that stops publishing the field from
        silently retaining every reprocessing.
        """
        old, new = _pair()
        kept, alternates = select_preferred_duplicates([old, new])
        assert kept == [new]
        assert alternates == {("MGRS-34WFA", "2021-09-08"): [old]}


class TestTheLadderStepsOneAcquisitionAtATime:
    """The fallback ladder must not swap an acquisition's sibling out from under it.

    `select_preferred_duplicates` keeps one copy per ACQUISITION, but `alternates` is keyed
    by tile-date because that is the granularity a read failure is attributed at. Swapping
    on that key replaced every surviving acquisition with the same alternate — turning
    `[a_new, b_new]` into `[a_old, a_old]`, which duplicates one acquisition and drops the
    other. That is worse than the coverage loss the acquisition split fixed, because the
    loader is then handed the same granule twice.
    """

    @staticmethod
    def _four() -> tuple[_Item, _Item, _Item, _Item]:
        """Two acquisitions on one tile-date, each with a reprocessing."""
        at = "2021-09-08T10:20:31.024000Z"
        bt = "2021-09-08T11:10:14.512000Z"
        return (
            _Item("S2A_33XVG_20210908_0_L2A", "MGRS-33XVG", "0", **{"datetime": at}),
            _Item("S2A_33XVG_20210908_1_L2A", "MGRS-33XVG", "1", **{"datetime": at}),
            _Item("S2B_33XVG_20210908_0_L2A", "MGRS-33XVG", "0", **{"datetime": bt}),
            _Item("S2B_33XVG_20210908_1_L2A", "MGRS-33XVG", "1", **{"datetime": bt}),
        )

    def test_stepping_down_keeps_the_other_acquisition(self) -> None:
        a_old, a_new, b_old, b_new = self._four()
        kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
        assert kept == [a_new, b_new]

        stepped, _ = step_down_copies(alternates, kept, only=None)
        assert stepped == [a_old, b_new], "one acquisition steps; its sibling is untouched"
        assert len({id(i) for i in stepped}) == 2, "no acquisition may be duplicated"

    def test_the_ladder_then_steps_the_second_acquisition(self) -> None:
        a_old, a_new, b_old, b_new = self._four()
        kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
        first, _ = step_down_copies(alternates, kept, only=None)
        second, _ = step_down_copies(alternates, first, only=None)
        assert second == [a_old, b_old]
        assert step_down_copies(alternates, second, only=None) is None, "ladder exhausted"

    def test_a_single_acquisition_steps_exactly_as_before(self) -> None:
        """The behaviour the ladder was built for, unchanged."""
        old, new = _pair()
        kept, alternates = select_preferred_duplicates([old, new])
        stepped, keys = step_down_copies(alternates, kept, only=None)
        assert stepped == [old]
        assert keys == {("MGRS-34WFA", "2021-09-08")}


def _two_acquisitions_each_with_a_spare() -> tuple[_Item, _Item, _Item, _Item]:
    """One high-latitude tile-date: two ORBITS, each reprocessed once.

    Successive orbits revisit the same tile the same day above ~60 degrees, so a tile-date
    holds genuinely different imagery as well as reprocessings of it. The acquisition split
    keeps one copy per orbit; the spares are the ladder.
    """
    a_old = _Item("A_old", "MGRS-34WFA", "0")
    a_new = _Item("A_new", "MGRS-34WFA", "9")  # ranks first across the whole tile-date
    b_old = _Item("B_old", "MGRS-34WFA", "0")
    b_new = _Item("B_new", "MGRS-34WFA", "1")
    # Distinct acquisition instants, far enough apart to be separate orbits.
    for it in (a_old, a_new):
        it.properties["datetime"] = "2021-09-08T10:00:00Z"
    for it in (b_old, b_new):
        it.properties["datetime"] = "2021-09-08T11:00:00Z"
    return a_old, a_new, b_old, b_new


def test_the_ladder_steps_the_acquisition_that_failed_not_the_best_ranked_spare():
    """A tile-date with two acquisitions must downgrade only the one that could not be read.

    Ranking is per tile-date, so without attribution the rung taken is whichever spare sorts
    first — which can belong to the acquisition that read perfectly well. That downgrades
    healthy imagery to an older baseline, leaves the unreadable copy still selected, and
    forces a second rung: two spent, one acquisition needlessly older.

    Here A's spare outranks B's, and B is the one that failed.
    """
    a_old, a_new, b_old, b_new = _two_acquisitions_each_with_a_spare()
    kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
    assert set(kept) == {a_new, b_new}, "one survivor per acquisition"

    stepped = step_down_copies(dict(alternates), kept, implicated=[b_new])
    assert stepped is not None
    items, _keys = stepped
    assert a_new in items, "the acquisition that read fine must keep its newest copy"
    assert b_old in items, "the acquisition that failed is the one that steps down"
    assert b_new not in items


def test_without_attribution_the_ladder_still_steps_the_best_ranked_spare():
    """The pre-existing behaviour, kept: with nothing attributed there is nothing to prefer.

    Pinned so the new preference cannot quietly become the only path — a caller that cannot
    identify the failing object must still make progress.
    """
    a_old, a_new, b_old, b_new = _two_acquisitions_each_with_a_spare()
    kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
    stepped = step_down_copies(dict(alternates), kept)
    assert stepped is not None
    items, _keys = stepped
    assert a_old in items, "highest-ranked spare belongs to A, so A steps"


def test_a_failure_in_one_tile_does_not_downgrade_a_neighbouring_tile():
    """Acquisition matching is by INSTANT, and neighbouring MGRS tiles share one pass.

    `implicated` carries the failed items from every tile in the date, so without filtering to
    the tile-day being stepped, tile B's failure matches tile A's acquisition by time alone —
    downgrading A while leaving B's own failed copy selected. That is the defect the attribution
    was added to remove, moved one level sideways.
    """
    a_old = _Item("A_old", "MGRS-34WFA", "0")
    a_new = _Item("A_new", "MGRS-34WFA", "1")
    b_old = _Item("B_old", "MGRS-34WFB", "0")
    b_new = _Item("B_new", "MGRS-34WFB", "1")
    for it in (a_old, a_new, b_old, b_new):  # one pass — the same instant on both tiles
        it.properties["datetime"] = "2021-09-08T10:00:00Z"

    kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
    assert set(kept) == {a_new, b_new}

    # Only tile B failed.
    stepped = step_down_copies(dict(alternates), kept, only=[("MGRS-34WFB", "2021-09-08")], implicated=[b_new])
    assert stepped is not None
    items, keys = stepped
    assert b_old in items, "the tile that failed steps down"
    assert a_new in items, "the neighbouring tile keeps its newest copy"
    assert keys == {("MGRS-34WFB", "2021-09-08")}


# The two hosts that matter, spelled out rather than taken from the constant, so a change to
# the constant has to be a deliberate edit here too.
_IN_REGION = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/33/T/WM/2017/1"
_REMOTE = "s3://sentinel-s2-l2a/tiles/33/T/WM/2017/1/20/1"


def _bands_at(host_root: str, *, extra: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
    """Every asset key the ingest READS, served from one host — plus any extras.

    Keyed by the real band names because the predicate looks up exactly those. The first
    version of these fixtures used made-up keys (`b0`, `b1`), which the predicate found none
    of, so it answered "remote" for every fixture and the tests passed while agreeing with
    each other and with nothing real. Same class of mistake as the bug they exist to cover.
    """
    assets = {key: {"href": f"{host_root}/{key}"} for key in READ_ASSET_KEYS}
    assets.update(extra or {})
    return assets


def _with_assets(item: _Item, assets: dict[str, dict[str, str]]) -> _Item:
    item.assets = assets
    return item


def _copy(ident: str, *, sequence: str, baseline: str, host_root: str, extra: dict | None = None) -> _Item:
    """One copy of a tile-date, at a given baseline/sequence, served from a given host."""
    it = _Item(ident, "MGRS-33TWM", sequence, **{"s2:processing_baseline": baseline})
    return _with_assets(it, _bands_at(host_root, extra=extra))


class TestInRegionPreference:
    """Locality decides between copies the catalogue indexed from two regions.

    Written from the real pair measured on the live catalogue: tile 35QKC on 2017-01-20 is
    indexed twice, both at baseline 00.01 and the SAME acquisition instant, with the remote
    copy carrying the HIGHER sequence. A sequence-first key therefore preferred the remote
    copy on every duplicated date.
    """

    def test_equal_baseline_prefers_the_in_region_copy(self) -> None:
        """The measured case. Same baseline and instant, so the pixels are identical and the
        only difference is whether the read egresses through NAT.
        """
        remote = _copy("S2A_33TWM_20170120_1_L2A", sequence="1", baseline="00.01", host_root=_REMOTE)
        local = _copy("S2A_33TWM_20170120_0_L2A", sequence="0", baseline="00.01", host_root=_IN_REGION)
        kept, _ = select_preferred_duplicates([remote, local])
        assert [i.id for i in kept] == [local.id], "the in-region copy must survive"

    def test_the_remote_copy_stays_available_as_a_fallback(self) -> None:
        """Preferring in-region must not DISCARD the remote copy — it is still the ladder a
        caller steps down when the chosen copy cannot be read.
        """
        remote = _copy("S2A_33TWM_20170120_1_L2A", sequence="1", baseline="00.01", host_root=_REMOTE)
        local = _copy("S2A_33TWM_20170120_0_L2A", sequence="0", baseline="00.01", host_root=_IN_REGION)
        _kept, alternates = select_preferred_duplicates([remote, local])
        assert [i.id for i in next(iter(alternates.values()))] == [remote.id]

    def test_locality_never_outranks_a_newer_baseline(self) -> None:
        """The complement, and the reason locality sits BELOW the baseline term: a cheaper
        read must never be bought with a worse pixel.
        """
        local_old = _copy("S2A_33TWM_20170120_0_L2A", sequence="0", baseline="02.06", host_root=_IN_REGION)
        remote_new = _copy("S2A_33TWM_20170120_1_L2A", sequence="1", baseline="05.00", host_root=_REMOTE)
        kept, _ = select_preferred_duplicates([local_old, remote_new])
        assert [i.id for i in kept] == [remote_new.id], "a newer baseline must win despite egress"

    def test_sequence_still_decides_when_baseline_and_locality_tie(self) -> None:
        """No regression: with nothing to separate them but sequence, the old rule stands."""
        low = _copy("S2A_33TWM_20170120_0_L2A", sequence="0", baseline="05.00", host_root=_IN_REGION)
        high = _copy("S2A_33TWM_20170120_1_L2A", sequence="1", baseline="05.00", host_root=_IN_REGION)
        kept, _ = select_preferred_duplicates([low, high])
        assert [i.id for i in kept] == [high.id]

    def test_an_all_remote_date_still_selects_one(self) -> None:
        """Most backfilled dates have NO in-region copy. They must still reduce to one item
        rather than being dropped or raising — the majority case by a wide margin.
        """
        a = _copy("S2A_33TWM_20170120_0_L2A", sequence="0", baseline="02.06", host_root=_REMOTE)
        b = _copy("S2A_33TWM_20170120_1_L2A", sequence="1", baseline="02.06", host_root=_REMOTE)
        kept, _ = select_preferred_duplicates([a, b])
        assert [i.id for i in kept] == [b.id], "all-remote falls through to sequence"


class TestLocalityPredicate:
    """`item_is_in_preferred_location` — ALL assets, not any."""

    def test_bands_all_in_a_preferred_host_are_local(self) -> None:
        assert item_is_in_preferred_location(_with_assets(_Item("x"), _bands_at(_IN_REGION))) is True

    def test_extra_assets_elsewhere_do_not_make_it_remote(self) -> None:
        """THE REGRESSION THIS EXISTS FOR. A real Element 84 item carries its COG bands and
        the original JP2s as extra assets — 35 assets over two buckets on the measured pair.
        Judging all of them marked that item remote and silently disabled the whole
        preference: a live-catalogue check then found 0 mixed tile-dates where 5 existed.
        The extras are never fetched, so they cannot make a read expensive.
        """
        item = _with_assets(
            _Item("x"),
            _bands_at(_IN_REGION, extra={"aot": {"href": f"{_REMOTE}/AOT.jp2"}, "wvp": {"href": f"{_REMOTE}/WVP.jp2"}}),
        )
        assert item_is_in_preferred_location(item) is True

    def test_bands_straddling_two_hosts_are_not_local(self) -> None:
        """All of the READ set, not any: an item whose bands are split cannot deliver the
        locality being claimed.
        """
        assets = _bands_at(_IN_REGION)
        assets["red"] = {"href": f"{_REMOTE}/B04.jp2"}
        assert item_is_in_preferred_location(_with_assets(_Item("x"), assets)) is False

    def test_an_item_exposing_none_of_the_read_bands_is_not_local(self) -> None:
        """Absence of evidence is not locality. This is also the case every pre-existing
        fixture in this file hits, which is why they keep their old relative order.
        """
        assert item_is_in_preferred_location(_Item("x")) is False
        assert item_is_in_preferred_location(_with_assets(_Item("x"), {"thumbnail": {"href": _IN_REGION}})) is False

    def test_the_remote_bucket_is_not_in_the_preferred_set(self) -> None:
        """Guards the constant itself: if `sentinel-s2-l2a` were ever added to
        PREFERRED_ASSET_BUCKETS the whole fix would silently invert.
        """
        assert "sentinel-s2-l2a" not in PREFERRED_ASSET_BUCKETS


class TestAssetBucketParsing:
    """The bucket is PARSED, because a substring test answers yes to the wrong things."""

    @pytest.mark.parametrize(
        ("href", "expected"),
        [
            ("s3://sentinel-cogs/sentinel-s2-l2a-cogs/33/T/WM/B04.tif", "sentinel-cogs"),
            ("https://sentinel-cogs.s3.us-west-2.amazonaws.com/x/B04.tif", "sentinel-cogs"),
            ("https://sentinel-cogs.s3.amazonaws.com/x/B04.tif", "sentinel-cogs"),
            ("https://s3.us-west-2.amazonaws.com/sentinel-cogs/x/B04.tif", "sentinel-cogs"),
            ("s3://sentinel-s2-l2a/tiles/33/T/WM/B04.jp2", "sentinel-s2-l2a"),
        ],
    )
    def test_the_three_forms_the_catalogues_emit(self, href: str, expected: str) -> None:
        assert asset_bucket(href) == expected

    @pytest.mark.parametrize(
        "href",
        [
            "s3://sentinel-cogs-backup/x/B04.tif",
            "https://sentinel-cogs-backup.s3.us-west-2.amazonaws.com/x/B04.tif",
            "https://elsewhere.example/sentinel-cogs/x/B04.tif",
            "https://sentinel-cogs.attacker.example/x/B04.tif",
        ],
    )
    def test_lookalikes_are_not_the_preferred_bucket(self, href: str) -> None:
        """Each of these CONTAINS a preferred bucket's name while being somewhere else. A
        substring test called them local and would have routed the reads through NAT believing
        they were free.
        """
        assert asset_bucket(href) not in PREFERRED_ASSET_BUCKETS

    def test_a_non_s3_href_has_no_bucket(self) -> None:
        assert asset_bucket("https://example.com/data/B04.tif") is None


class TestUnknownBaselineSuspendsLocality:
    """An absent baseline is not a tie, and treating it as one is wrong twice over."""

    def test_an_unknown_baseline_does_not_win_on_locality(self) -> None:
        """The reported P1. A local copy with NO baseline must not displace a remote copy at
        05.00: that can select an older reprocessing, AND `_extract_baseline` maps the missing
        value to 0 downstream, so the post-04.00 offset correction is skipped and the
        reflectance is wrong with nothing raising.
        """
        local_unknown = _Item("S2A_33TWM_20220107_0_L2A", "MGRS-33TWM", "0")
        _with_assets(local_unknown, _bands_at(_IN_REGION))
        remote_best = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="05.00", host_root=_REMOTE)
        kept, _ = select_preferred_duplicates([local_unknown, remote_best])
        assert [i.id for i in kept] == [remote_best.id], "sequence must decide when a baseline is unknown"

    def test_a_usable_baseline_beats_a_higher_sequence_without_one(self) -> None:
        """REVERSED deliberately, and the reason is worth reading.

        This used to assert that a newest copy omitting the property still wins, on the grounds
        that suspending locality must not become demoting the unknown. That was right while an
        unreadable baseline merely meant "no evidence". It now means the date is REFUSED, because
        whether those pixels carry the +1000 offset cannot be determined — and the duplicate
        recovery ladder only steps down on a read failure, never on a refusal. So preferring the
        unreadable copy does not risk an older pixel, it discards the date while a usable copy
        sits directly behind it.
        """
        local_old = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="05.00", host_root=_IN_REGION)
        remote_unknown = _Item("S2A_33TWM_20220107_1_L2A", "MGRS-33TWM", "1")
        _with_assets(remote_unknown, _bands_at(_REMOTE))
        kept, alternates = select_preferred_duplicates([local_old, remote_unknown])
        assert [i.id for i in kept] == [local_old.id]
        assert [i.id for i in alternates[("MGRS-33TWM", "2021-09-08")]] == [remote_unknown.id]


class TestFallbackLadderKeepsBaselineOrder:
    """Every rung stays in descending baseline order, not just the top one."""

    def test_a_middle_baseline_is_not_skipped_for_a_local_older_one(self) -> None:
        """The reported MEDIUM. With 05.00 selected, a remote 04.00 and a local 03.00, a read
        failure must step to 04.00. Collapsing every non-best baseline into one tier let
        locality hand out the 03.00 copy and needlessly degrade the imagery.
        """
        best = _copy("S2A_33TWM_20220107_2_L2A", sequence="2", baseline="05.00", host_root=_REMOTE)
        middle = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="04.00", host_root=_REMOTE)
        worst_local = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="03.00", host_root=_IN_REGION)
        kept, alternates = select_preferred_duplicates([worst_local, middle, best])
        assert [i.id for i in kept] == [best.id]
        ladder = [i.id for i in next(iter(alternates.values()))]
        assert ladder == [middle.id, worst_local.id], "the ladder skipped a newer reprocessing"


class TestTheDuplicateLogIsAnAuditTrail:
    """The only record of which copy won, so it has to be accurate."""

    def _emit(self, caplog, kept, alternates):
        log = logging.getLogger("dup-audit-test")
        with caplog.at_level(logging.INFO, logger="dup-audit-test"):
            copies_label  # noqa: B018 — keep the import honest
            from tessera_embeddings.ingest.duplicates import log_duplicate_selection

            log_duplicate_selection(log, "roi-x", alternates, kept=kept)
        return " ".join(r.getMessage() for r in caplog.records)

    def test_it_states_the_actual_preference(self, caplog) -> None:
        """It said "newest kept" while the ranking preferred an in-region copy over a
        higher-sequence remote one — a line misreporting its own behaviour.
        """
        local = _copy("a", sequence="0", baseline="00.01", host_root=_IN_REGION)
        remote = _copy("b", sequence="1", baseline="00.01", host_root=_REMOTE)
        msg = self._emit(caplog, [local], {("MGRS-33TWM", "2021-09-08"): [remote]})
        assert "newest baseline, then in-region, then complete read set, then newest sequence" in msg
        assert "newest kept" not in msg, "the stale claim must not come back"

    def test_it_names_where_the_survivors_came_from(self, caplog) -> None:
        """THE AUDIT TRAIL. Where two copies share a baseline their pixels are identical, so
        nothing downstream can show which was read — not the store, not the mosaic, not the
        metrics. Without this the decision is unobservable in production.
        """
        local = _copy("a", sequence="0", baseline="00.01", host_root=_IN_REGION)
        remote = _copy("b", sequence="1", baseline="00.01", host_root=_REMOTE)
        msg = self._emit(caplog, [local], {("MGRS-33TWM", "2021-09-08"): [remote]})
        assert "1 in-region, 0 remote" in msg

    def test_it_reports_a_remote_survivor_as_remote(self, caplog) -> None:
        """The complement: the line must be capable of saying the preference did NOT apply,
        or it is decoration rather than evidence.
        """
        remote = _copy("b", sequence="1", baseline="00.01", host_root=_REMOTE)
        msg = self._emit(caplog, [remote], {("MGRS-33TWM", "2021-09-08"): [remote]})
        assert "0 in-region, 1 remote" in msg

    def test_it_counts_only_the_contested_tile_dates(self, caplog) -> None:
        """THE FINDING. `kept` is every survivor on the ROI, and on a wide one almost none of
        them had a duplicate. Counting them all described the composition of the whole supply
        rather than the choices this line exists to record.
        """
        contested_winner = _copy("a", sequence="0", baseline="00.01", host_root=_IN_REGION)
        rejected = _copy("b", sequence="1", baseline="00.01", host_root=_REMOTE)
        # A hundred tile-dates that were never in question, all served from elsewhere.
        untouched = [_with_assets(_Item(f"u{i}", f"MGRS-33TW{i:02d}", "0"), _bands_at(_REMOTE)) for i in range(100)]
        msg = self._emit(
            caplog,
            [contested_winner, *untouched],
            {("MGRS-33TWM", "2021-09-08"): [rejected]},
        )
        assert "1 in-region, 0 remote" in msg, "the 100 uncontested dates drowned out the decision"
        assert "winners by source" in msg

    def test_an_unkeyable_survivor_does_not_break_the_line(self, caplog) -> None:
        """`solar_day_of` raises on an un-normalised item, and a log call must not abort an
        ingest. The figure is omitted for that item rather than propagated.
        """
        winner = _copy("a", sequence="0", baseline="00.01", host_root=_IN_REGION)
        stray = _with_assets(_Item("stray", "MGRS-33TWM", "0"), _bands_at(_REMOTE))
        stray.datetime = datetime(2021, 9, 8, 7, 30, 0, tzinfo=UTC)
        msg = self._emit(caplog, [winner, stray], {("MGRS-33TWM", "2021-09-08"): [stray]})
        assert "1 in-region, 0 remote" in msg

    def test_it_states_one_ranking_because_there_is_only_one(self, caplog) -> None:
        """This clause used to be conditional, naming a sequence-only fallback for acquisitions
        where a copy declared no readable baseline. There is no fallback any more — such a copy
        simply ranks last — so the line states the ranking unconditionally, and it is the truth
        on an uncertain-metadata date as much as on any other.
        """
        winner = _copy("a", sequence="1", baseline="00.01", host_root=_IN_REGION)
        unreadable = _with_assets(_Item("b", "MGRS-33TWM", "0"), _bands_at(_REMOTE))
        msg = self._emit(caplog, [winner], {("MGRS-33TWM", "2021-09-08"): [unreadable]})
        assert "newest baseline, then in-region, then complete read set, then newest sequence" in msg
        assert "sequence alone" not in msg, "a mode that cannot happen must not be reported"

    def test_no_duplicates_logs_nothing(self, caplog) -> None:
        assert self._emit(caplog, [], {}) == ""


def _copy_at(ident: str, *, acquired: str, sequence: str, baseline: str | None, host_root: str = _IN_REGION) -> _Item:
    """A copy pinned to a given ACQUISITION instant, for cross-acquisition ordering tests."""
    extra: dict[str, object] = {"datetime": acquired}
    if baseline is not None:
        extra["s2:processing_baseline"] = baseline
    return _with_assets(_Item(ident, "MGRS-33TWM", sequence, **extra), _bands_at(host_root))


class TestLocalityNeedsTheCompleteReadSet:
    """An asset-incomplete item cannot claim locality. Regression: review on PR #107.

    The predicate skipped keys that were absent or href-less and then asserted over what
    remained, so a single local band satisfied it. That copy reads as fully in-region,
    displaces a COMPLETE remote copy on a baseline tie, and then fails at load — sending
    the recovery ladder stepping down every duplicated tile-date of that day.
    """

    def test_one_local_band_is_not_a_local_item(self) -> None:
        assert (
            item_is_in_preferred_location(_with_assets(_Item("x"), {"blue": {"href": f"{_IN_REGION}/B02.tif"}}))
            is False
        )

    def test_the_complete_read_set_is_local(self) -> None:
        assert item_is_in_preferred_location(_with_assets(_Item("x"), _bands_at(_IN_REGION))) is True

    def test_a_read_key_missing_from_an_otherwise_local_item_is_not_local(self) -> None:
        assets = _bands_at(_IN_REGION)
        dropped = assets.pop(READ_ASSET_KEYS[-1])
        assert dropped is not None, "the fixture must actually have carried that key"
        assert item_is_in_preferred_location(_with_assets(_Item("x"), assets)) is False

    def test_a_read_key_present_but_href_less_is_not_local(self) -> None:
        assets = _bands_at(_IN_REGION)
        assets[READ_ASSET_KEYS[0]] = {}
        assert item_is_in_preferred_location(_with_assets(_Item("x"), assets)) is False

    def test_an_incomplete_local_copy_does_not_displace_a_complete_remote_one(self) -> None:
        """The consequence the reviewer named, asserted on the selector and not the predicate.

        Sequence is what decides here once locality ties, so pre-fix and post-fix disagree:
        locality outranks sequence, so a "local" partial copy beat the complete copy that
        carried the newer sequence.
        """
        complete_remote = _copy("remote", sequence="1", baseline="05.00", host_root=_REMOTE)
        partial_local = _with_assets(
            _Item("local", "MGRS-33TWM", "0", **{"s2:processing_baseline": "05.00"}),
            {"blue": {"href": f"{_IN_REGION}/B02.tif"}},
        )
        kept, _ = select_preferred_duplicates([partial_local, complete_remote])
        assert [it.id for it in kept] == ["remote"]

    def test_an_incomplete_copy_loses_an_otherwise_total_tie(self) -> None:
        """Completeness is its own rank field, so the id tiebreak cannot hand it the date.

        `local` sorts before `remote`, so with baseline, locality and sequence all tied the
        alphabetical tiebreak used to award the tile-date to the copy that cannot be read.
        """
        complete = _copy("remote", sequence="0", baseline="05.00", host_root=_REMOTE)
        partial = _with_assets(
            _Item("local", "MGRS-33TWM", "0", **{"s2:processing_baseline": "05.00"}),
            {"blue": {"href": f"{_IN_REGION}/B02.tif"}},
        )
        kept, _ = select_preferred_duplicates([partial, complete])
        assert [it.id for it in kept] == ["remote"]

    def test_the_completeness_predicate_answers_readability_not_location(self) -> None:
        """A complete REMOTE read set is complete — the two predicates must not conflate."""
        assert read_set_is_complete(_with_assets(_Item("x"), _bands_at(_REMOTE))) is True
        assert item_is_in_preferred_location(_with_assets(_Item("x"), _bands_at(_REMOTE))) is False

    def test_an_item_with_no_assets_is_incomplete(self) -> None:
        """Radar copies and non-S2 products tie here rather than being ordered by it."""
        assert read_set_is_complete(_Item("x")) is False


class TestNonFiniteBaselinesAreUnknown:
    """`float()` accepts "NaN" and "Infinity". Neither is a baseline. Review on PR #107."""

    @pytest.mark.parametrize("raw", ["NaN", "nan", "Infinity", "inf", "-inf", "-Infinity"])
    def test_a_non_finite_baseline_reads_as_unknown(self, raw: str) -> None:
        assert item_processing_baseline(_Item("x", "MGRS-33TWM", "0", **{"s2:processing_baseline": raw})) is None

    def test_a_real_baseline_still_reads(self) -> None:
        """So the test above cannot pass by rejecting everything.

        Reported as an integer hundredth (500, not 5.0) — the scale the correction threshold is
        expressed in. There used to be two parsers on two scales, and unifying them is what let
        the numeric edge cases be fixed once instead of twice.
        """
        assert item_processing_baseline(_Item("x", "MGRS-33TWM", "0", **{"s2:processing_baseline": "05.00"})) == 500

    def test_infinity_does_not_outrank_a_real_baseline(self) -> None:
        infinite = _copy("infinite", sequence="0", baseline="Infinity", host_root=_IN_REGION)
        real = _copy("real", sequence="1", baseline="05.00", host_root=_IN_REGION)
        kept, _ = select_preferred_duplicates([infinite, real])
        # Unknown suspends the baseline comparison, so sequence decides and 1 beats 0.
        assert [it.id for it in kept] == ["real"]

    def test_a_nan_baseline_does_not_leave_the_order_response_dependent(self) -> None:
        """Every NaN comparison is false, so a NaN in the ladder preserved the given order."""

        def _run(order: list[str]) -> list[str]:
            by_id = {
                "nan": _copy("nan", sequence="0", baseline="NaN", host_root=_IN_REGION),
                "real": _copy("real", sequence="2", baseline="04.00", host_root=_IN_REGION),
            }
            kept, _ = select_preferred_duplicates([by_id[i] for i in order])
            return [it.id for it in kept]

        assert _run(["nan", "real"]) == _run(["real", "nan"]) == ["real"]


class TestUnknownBaselineStaysScopedToItsAcquisition:
    """One acquisition's unreadable baseline must not reorder another's. Review on PR #107.

    `_rank_copies` reverts to sequence ordering when ANY baseline in the list it is given is
    unknown — correct within an acquisition, where it stops locality promoting a copy over a
    possibly-newer one. The alternates list was built by ranking a whole tile-date in one
    call, so a single malformed item leaked that suspension across acquisition boundaries.
    """

    _A = "2021-09-08T10:00:00Z"
    _B = "2021-09-08T14:00:00Z"

    def test_a_neighbours_unknown_baseline_keeps_this_ladder_baseline_first(self) -> None:
        """The unknown copy must be a REJECTED one — that is what puts it in the shared list.

        Acquisition A holds an unreadable baseline on its LOSING copy, so `rejected` spans
        both acquisitions and contains an unknown. Ranking that whole list in one call
        suspends the baseline term for B as well.
        """
        items = [
            # A: the unknown loses on sequence, so it lands in the shared rejected list.
            _copy_at("a-keep", acquired=self._A, sequence="5", baseline="04.00"),
            _copy_at("a-unknown", acquired=self._A, sequence="0", baseline=None),
            # B: readable throughout, so its ladder must stay in descending baseline order.
            _copy_at("b-best", acquired=self._B, sequence="0", baseline="05.00"),
            _copy_at("b-mid", acquired=self._B, sequence="1", baseline="04.00"),
            _copy_at("b-newer-seq", acquired=self._B, sequence="9", baseline="03.00"),
        ]
        kept, alternates = select_preferred_duplicates(items)

        assert sorted(it.id for it in kept) == ["a-keep", "b-best"]

        ladder = [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]]
        assert "a-unknown" in ladder, "the fixture must put the unknown into the shared list"
        # Pre-fix the shared call fell back to sequence ordering for the WHOLE tile-date, which
        # put b-newer-seq (sequence 9, baseline 03.00) ahead of b-mid (sequence 1, baseline
        # 04.00) — a read failure would then have handed out the older reprocessing.
        assert [i for i in ladder if i.startswith("b-")] == ["b-mid", "b-newer-seq"]

    def test_the_best_spare_overall_is_still_first(self) -> None:
        """`_first_for_failed_acquisition` falls back to copies[0] when nothing is attributed."""
        items = [
            _copy_at("a-win", acquired=self._A, sequence="0", baseline="05.00"),
            _copy_at("a-spare", acquired=self._A, sequence="0", baseline="03.00"),
            _copy_at("b-win", acquired=self._B, sequence="0", baseline="05.00"),
            _copy_at("b-spare", acquired=self._B, sequence="0", baseline="04.00"),
        ]
        _, alternates = select_preferred_duplicates(items)
        ladder = [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]]
        # b-spare (04.00) outranks a-spare (03.00) despite belonging to the later acquisition.
        assert ladder == ["b-spare", "a-spare"]


class TestTheLadderKeepsGlobalPriorityPastItsHead:
    """`s2_roi`'s unattributed recovery consumes `copies[0]` on each retry, so EVERY position in
    the ladder is a choice, not only the first. Raised on PR #107 against a per-acquisition
    ranking that ordered the ladder heads and then flattened whole ladders behind them.
    """

    _A = "2021-09-08T10:00:00Z"
    _B = "2021-09-08T14:00:00Z"

    def test_a_second_retry_does_not_skip_a_better_copy_in_another_acquisition(self) -> None:
        items = [
            _copy_at("a-win", acquired=self._A, sequence="9", baseline="05.10"),
            _copy_at("a-05", acquired=self._A, sequence="2", baseline="05.00"),
            _copy_at("a-01", acquired=self._A, sequence="1", baseline="01.00"),
            _copy_at("b-win", acquired=self._B, sequence="9", baseline="05.10"),
            _copy_at("b-04", acquired=self._B, sequence="2", baseline="04.00"),
        ]
        _, alternates = select_preferred_duplicates(items)
        ladder = [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]]
        # Concatenating whole ladders behind their sorted heads gave [a-05, a-01, b-04], so the
        # second retry downgraded to baseline 01.00 while a 04.00 copy was still untried.
        assert ladder == ["a-05", "b-04", "a-01"]

    def test_every_position_is_in_descending_baseline_order(self) -> None:
        """The general property, over three acquisitions interleaved at every rung."""
        items = [
            _copy_at(f"{tag}-win", acquired=acq, sequence="9", baseline="05.10")
            for tag, acq in (("a", self._A), ("b", self._B))
        ] + [
            _copy_at("a-hi", acquired=self._A, sequence="3", baseline="05.00"),
            _copy_at("a-lo", acquired=self._A, sequence="1", baseline="02.00"),
            _copy_at("b-mid", acquired=self._B, sequence="3", baseline="04.00"),
            _copy_at("b-low", acquired=self._B, sequence="1", baseline="03.00"),
        ]
        _, alternates = select_preferred_duplicates(items)
        ladder = alternates[("MGRS-33TWM", "2021-09-08")]
        baselines = [item_processing_baseline(it) for it in ladder]
        assert baselines == sorted(baselines, reverse=True), f"ladder out of order: {baselines}"
        assert [it.id for it in ladder] == ["a-hi", "b-mid", "b-low", "a-lo"]

    def test_an_unreadable_baseline_sorts_last_rather_than_suspending_the_order(self) -> None:
        """The alternative — suspending the comparison — is what leaked one acquisition's
        malformed metadata into another's ladder. A copy whose baseline cannot be read is also
        the one whose correction will silently be skipped, so it belongs at the end.
        """
        items = [
            _copy_at("a-win", acquired=self._A, sequence="9", baseline="05.10"),
            _copy_at("a-unknown", acquired=self._A, sequence="5", baseline=None),
            _copy_at("a-04", acquired=self._A, sequence="1", baseline="04.00"),
            _copy_at("b-win", acquired=self._B, sequence="9", baseline="05.10"),
            _copy_at("b-03", acquired=self._B, sequence="1", baseline="03.00"),
        ]
        _, alternates = select_preferred_duplicates(items)
        ladder = [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]]
        assert ladder == ["a-04", "b-03", "a-unknown"]


class TestLocalityDoesNotDecideBetweenUnknownBaselines:
    """In the ladder key, locality was ranked ahead of sequence for copies that declare no
    readable baseline — so a local sequence-1 spare came out before a remote sequence-2 one,
    buying cheaper egress with an older reprocessing. Raised on PR #107.
    """

    _A = "2021-09-08T10:00:00Z"

    def test_sequence_decides_between_two_unknown_baseline_spares(self) -> None:
        items = [
            _copy_at("winner", acquired=self._A, sequence="3", baseline=None),
            _copy_at("remote-seq2", acquired=self._A, sequence="2", baseline=None, host_root=_REMOTE),
            _copy_at("local-seq1", acquired=self._A, sequence="1", baseline=None, host_root=_IN_REGION),
        ]
        kept, alternates = select_preferred_duplicates(items)
        assert [it.id for it in kept] == ["winner"]
        ladder = [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]]
        assert ladder == ["remote-seq2", "local-seq1"], "locality outranked the newer sequence"

    def test_locality_still_decides_between_two_equal_known_baselines(self) -> None:
        """The complement: neutralising locality everywhere would pass the test above."""
        items = [
            _copy_at("winner", acquired=self._A, sequence="3", baseline="05.00"),
            _copy_at("remote", acquired=self._A, sequence="1", baseline="04.00", host_root=_REMOTE),
            _copy_at("local", acquired=self._A, sequence="1", baseline="04.00", host_root=_IN_REGION),
        ]
        _, alternates = select_preferred_duplicates(items)
        ladder = [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]]
        assert ladder == ["local", "remote"]


class TestAMalformedHrefDoesNotAbortTheSelection:
    """`urlparse` raises on some malformed authorities, and locality is evaluated for every copy
    including singletons — so one bad catalogue href aborted the whole selection. PR #107.
    """

    @pytest.mark.parametrize("href", ["https://[oops/B02.tif", "https://[::1/x", "s3://[/key"])
    def test_an_unparseable_href_reads_as_remote(self, href: str) -> None:
        assert asset_bucket(href) is None

    def test_a_singleton_group_with_a_malformed_href_still_returns(self) -> None:
        item = _with_assets(_Item("only", "MGRS-33TWM", "0"), {"blue": {"href": "https://[oops/B02.tif"}})
        kept, alternates = select_preferred_duplicates([item])
        assert [it.id for it in kept] == ["only"]
        assert alternates == {}


class TestThePreferenceKeyHoldsItsInvariants:
    """One key orders copies within an acquisition and across acquisitions, so the properties it
    must hold are worth asserting directly rather than only through the outcomes above.

    This class exists because there used to be TWO keys expressing the same preference, and every
    fix had to be applied to both — which is how the read-set term reached one and not the other.
    """

    @staticmethod
    def _item(ident: str, *, baseline: str | None, sequence: str | None, local: bool, complete: bool) -> _Item:
        extra: dict[str, object] = {}
        if baseline is not None:
            extra["s2:processing_baseline"] = baseline
        item = _Item(ident, "MGRS-33TWM", sequence, **extra)
        keys = READ_ASSET_KEYS if complete else READ_ASSET_KEYS[:3]
        root = _IN_REGION if local else _REMOTE
        return _with_assets(item, {k: {"href": f"{root}/{k}"} for k in keys})

    def _pool(self) -> list[_Item]:
        out, n = [], 0
        for baseline in ("05.00", "04.00", None):
            for sequence in ("0", "2", None):
                for local in (True, False):
                    for complete in (True, False):
                        out.append(
                            self._item(f"i{n}", baseline=baseline, sequence=sequence, local=local, complete=complete)
                        )
                        n += 1
        return out

    def test_a_readable_baseline_beats_an_unreadable_one_among_equally_complete_copies(self) -> None:
        """Qualified by completeness, which ranks ahead of it: a copy that cannot be read is no
        use at any baseline, and the generic paths have no recovery for a missing band.
        """
        pool = [i for i in self._pool() if read_set_is_complete(i)]
        known = [i for i in pool if item_processing_baseline(i) is not None]
        unknown = [i for i in pool if item_processing_baseline(i) is None]
        assert known and unknown
        for k in known:
            for u in unknown:
                assert _preference_key(k) < _preference_key(u), f"{u.id} outranked {k.id}"

    def test_completeness_outranks_the_baseline(self) -> None:
        """An incomplete 05.00 copy must lose to a complete 04.00 one."""
        incomplete_new = self._item("a", baseline="05.00", sequence="9", local=True, complete=False)
        complete_old = self._item("z", baseline="04.00", sequence="0", local=False, complete=True)
        assert _preference_key(complete_old) < _preference_key(incomplete_new)

    def test_locality_never_outranks_a_higher_baseline(self) -> None:
        """The P1 this ordering exists to prevent: cheaper egress must not buy an older pixel."""
        remote_new = self._item("remote-new", baseline="05.00", sequence="0", local=False, complete=True)
        local_old = self._item("local-old", baseline="04.00", sequence="9", local=True, complete=True)
        assert _preference_key(remote_new) < _preference_key(local_old)

    def test_locality_does_not_participate_when_the_baseline_is_unreadable(self) -> None:
        local = self._item("local", baseline=None, sequence="1", local=True, complete=True)
        remote = self._item("remote", baseline=None, sequence="2", local=False, complete=True)
        assert _preference_key(remote) < _preference_key(local), "locality decided a baseline-free comparison"

    def test_an_incomplete_copy_loses_to_an_otherwise_identical_complete_one(self) -> None:
        complete = self._item("z-complete", baseline="05.00", sequence="0", local=False, complete=True)
        partial = self._item("a-partial", baseline="05.00", sequence="0", local=False, complete=False)
        assert _preference_key(complete) < _preference_key(partial), "the id tiebreak decided readability"

    def test_the_key_is_a_total_order_with_no_ties_between_distinct_items(self) -> None:
        """Ties would make the sort depend on catalogue response order, so a rerun could produce
        a different mosaic. The id term is what rules them out.
        """
        keys = [_preference_key(i) for i in self._pool()]
        assert len(set(keys)) == len(keys)

    def test_it_is_context_free(self) -> None:
        """The property that lets ONE key serve both regimes: an item's rank cannot depend on
        which other copies it is being compared against.
        """
        pool = self._pool()
        alone = {i.id: _preference_key(i) for i in pool}
        for subset in (pool[:5], pool[3:11], list(reversed(pool))):
            for item in subset:
                assert _preference_key(item) == alone[item.id]


class TestCompletenessSurvivesTheSequenceOnlyFallback:
    """When any copy of an acquisition declares no readable baseline, `_rank_copies` drops to
    `_sequence_key` — which omitted the read-set term, so a higher-sequence copy missing an asset
    beat a complete lower-sequence one. That guarantees the eager `No such band/alias` failure,
    which names no object, so the recovery steps every duplicated tile-date of the day rather
    than this one. Raised on PR #107.
    """

    def _copy(self, ident: str, *, sequence: str, baseline: str | None, complete: bool) -> _Item:
        extra: dict[str, object] = {}
        if baseline is not None:
            extra["s2:processing_baseline"] = baseline
        keys = READ_ASSET_KEYS if complete else READ_ASSET_KEYS[:3]
        return _with_assets(
            _Item(ident, "MGRS-33TWM", sequence, **extra),
            {k: {"href": f"{_IN_REGION}/{k}"} for k in keys},
        )

    def test_a_complete_copy_beats_a_higher_sequence_incomplete_one(self) -> None:
        # The unreadable baseline is what forces the sequence-only path for the whole group.
        unreadable = self._copy("unreadable", sequence="0", baseline=None, complete=True)
        incomplete_newer = self._copy("incomplete-newer", sequence="9", baseline="05.00", complete=False)
        complete_older = self._copy("complete-older", sequence="1", baseline="05.00", complete=True)
        kept, _ = select_preferred_duplicates([unreadable, incomplete_newer, complete_older])
        assert [it.id for it in kept] == ["complete-older"]

    def test_sequence_still_decides_between_two_complete_copies(self) -> None:
        """The complement: completeness must not have displaced the sequence preference."""
        unreadable = self._copy("unreadable", sequence="0", baseline=None, complete=True)
        newer = self._copy("newer", sequence="9", baseline="05.00", complete=True)
        older = self._copy("older", sequence="1", baseline="05.00", complete=True)
        kept, _ = select_preferred_duplicates([unreadable, newer, older])
        assert [it.id for it in kept] == ["newer"]


class TestRankingJudgesTheAssetsThisLoadWillRequest:
    """Readability is judged over the set the caller asked for, extra bands included.

    A fixed read set let a preferred copy lack a requested extra asset while a rejected copy had
    it; `odc.stac.load` then asked for that band and either failed or returned missing coverage.
    """

    def _copy(self, ident: str, *, sequence: str, extra: bool) -> _Item:
        assets = _bands_at(_IN_REGION)
        if extra:
            assets["qa"] = {"href": f"{_IN_REGION}/qa.tif"}
        return _with_assets(_Item(ident, "MGRS-33TWM", sequence, **{"s2:processing_baseline": "05.00"}), assets)

    def test_a_copy_missing_a_requested_extra_band_loses(self) -> None:
        without = self._copy("newer-without-qa", sequence="9", extra=False)
        with_qa = self._copy("older-with-qa", sequence="1", extra=True)
        kept, _ = select_preferred_duplicates([without, with_qa], (*READ_ASSET_KEYS, "qa"))
        assert [it.id for it in kept] == ["older-with-qa"]

    def test_the_same_pair_prefers_the_newer_copy_when_the_band_is_not_requested(self) -> None:
        """The complement: the extra band must only matter when the caller asks for it."""
        without = self._copy("newer-without-qa", sequence="9", extra=False)
        with_qa = self._copy("older-with-qa", sequence="1", extra=True)
        kept, _ = select_preferred_duplicates([without, with_qa])
        assert [it.id for it in kept] == ["newer-without-qa"]

    def test_locality_is_judged_over_the_requested_set_too(self) -> None:
        """A remote extra band makes the copy remote, since that read costs egress as well."""
        local = _with_assets(
            _Item("x", "MGRS-33TWM", "0", **{"s2:processing_baseline": "05.00"}),
            {**_bands_at(_IN_REGION), "qa": {"href": f"{_REMOTE}/qa.jp2"}},
        )
        assert item_is_in_preferred_location(local) is True, "not requested: irrelevant"
        assert item_is_in_preferred_location(local, keys=(*READ_ASSET_KEYS, "qa")) is False


class TestTheAuditUsesTheSameReadSetAsSelection:
    """The audit line is the only record of which source won, so it must judge locality over the
    set selection ranked on. Raised on PR #107.
    """

    def _emit(self, caplog, kept, alternates, read_keys=None) -> str:
        log = logging.getLogger("dup-audit-readkeys")
        with caplog.at_level(logging.INFO, logger="dup-audit-readkeys"):
            from tessera_embeddings.ingest.duplicates import log_duplicate_selection

            kwargs = {} if read_keys is None else {"read_keys": read_keys}
            log_duplicate_selection(log, "roi-x", alternates, kept=kept, **kwargs)
        return " ".join(r.getMessage() for r in caplog.records)

    def test_a_winner_is_local_when_the_missing_asset_was_not_requested(self, caplog) -> None:
        bands = tuple(k for k in READ_ASSET_KEYS if k != "scl")
        winner = _with_assets(
            _Item("a", "MGRS-33TWM", "0", **{"s2:processing_baseline": "05.00"}),
            {k: {"href": f"{_IN_REGION}/{k}"} for k in bands},
        )
        rejected = _copy("b", sequence="1", baseline="05.00", host_root=_REMOTE)
        alternates = {("MGRS-33TWM", "2021-09-08"): [rejected]}
        assert "1 in-region, 0 remote" in self._emit(caplog, [winner], alternates, read_keys=bands)
        caplog.clear()
        # With the fixed default it lacks `scl` and is reported remote, contradicting the decision.
        assert "0 in-region, 1 remote" in self._emit(caplog, [winner], alternates)
