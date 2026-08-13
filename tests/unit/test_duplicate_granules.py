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

from datetime import UTC, datetime

import pytest

from tessera_embeddings.ingest.duplicates import (
    alternates_for,
    copies_label,
    is_unreadable_source,
    item_sequence,
    item_tile,
    select_preferred_duplicates,
)


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
        """Fallback must step down in preference order, not arbitrarily."""
        old, new = _pair()
        newest = _Item("S2B_34WFA_20210908_2_L2A", "MGRS-34WFA", "2")
        kept, alternates = select_preferred_duplicates([old, new, newest])
        assert [it.id for it in kept] == [newest.id]
        assert [it.id for it in alternates[("MGRS-34WFA", "2021-09-08")]] == [new.id, old.id]

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
        sequence, newest kept and the older offered as the fallback."""
        old = self._at("S2B_33XVG_20210908_0_L2A", "2021-09-08T10:20:31.024000Z", "0")
        new = self._at("S2B_33XVG_20210908_1_L2A", "2021-09-08T10:20:31.024000Z", "1")
        kept, alternates = select_preferred_duplicates([old, new])
        assert kept == [new]
        assert alternates == {("MGRS-33XVG", "2021-09-08"): [old]}

    def test_each_acquisition_is_deduplicated_on_its_own(self) -> None:
        """Both at once — two acquisitions, each reprocessed. One survivor per
        acquisition, and both rejected copies reachable through the ladder."""
        a_old = self._at("S2A_33XVG_20210908_0_L2A", "2021-09-08T10:20:31.024000Z", "0")
        a_new = self._at("S2A_33XVG_20210908_1_L2A", "2021-09-08T10:20:31.024000Z", "1")
        b_old = self._at("S2B_33XVG_20210908_0_L2A", "2021-09-08T11:10:14.512000Z", "0")
        b_new = self._at("S2B_33XVG_20210908_1_L2A", "2021-09-08T11:10:14.512000Z", "1")
        kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
        assert kept == [a_new, b_new]
        assert set(alternates[("MGRS-33XVG", "2021-09-08")]) == {a_old, b_old}

    def test_sub_second_jitter_is_one_acquisition(self) -> None:
        """Reprocessings differ in the microseconds of an otherwise identical instant.
        Splitting on that would keep every reprocessing and defeat the module."""
        old = self._at("S2B_33XVG_20210908_0_L2A", "2021-09-08T10:20:31.024000Z", "0")
        new = self._at("S2B_33XVG_20210908_1_L2A", "2021-09-08T10:20:31.026000Z", "1")
        kept, _ = select_preferred_duplicates([old, new])
        assert kept == [new]

    def test_copies_without_an_instant_keep_the_old_behaviour(self) -> None:
        """No instant is no evidence of distinctness, so they compete as before —
        which is what keeps a catalogue that stops publishing the field from
        silently retaining every reprocessing."""
        old, new = _pair()
        kept, alternates = select_preferred_duplicates([old, new])
        assert kept == [new]
        assert alternates == {("MGRS-34WFA", "2021-09-08"): [old]}
