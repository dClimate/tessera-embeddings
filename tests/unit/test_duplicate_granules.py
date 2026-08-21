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
    Harmonisation,
    asset_bucket,
    item_is_in_preferred_location,
    read_set_is_complete,
)
from tessera_embeddings.ingest.duplicates import (
    _first_for_failed_acquisition,
    _preference_key,
    acquisition_identity,
    acquisition_instant,
    alternates_for,
    copies_label,
    is_unreadable_source,
    item_processing_baseline,
    item_sequence,
    item_tile,
    refuses_its_date,
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


class TestKeyingACopyAcrossProviders:
    """The tile and the acquisition, read from fields each catalogue actually populates."""

    def test_the_planetary_computer_tile_property_is_read(self) -> None:
        """Its ids carry the tile as `_T33TWM_` in a later field, which the Element 84 pattern
        does not match — so without the property every one of its items was unkeyable and
        duplicate selection was a no-op for the whole provider.
        """
        item = _Item("S2A_MSIL2A_20171219T095409_R079_T33TWM_20230807T060445", **{"s2:mgrs_tile": "33TWM"})
        assert item_tile(item) == "MGRS-33TWM"

    def test_the_planetary_computer_id_alone_is_not_enough(self) -> None:
        """The fixture above must be earning its keep, not passing on the id fallback."""
        assert item_tile(_Item("S2A_MSIL2A_20171219T095409_R079_T33TWM_20230807T060445")) is None

    def test_both_providers_key_one_tile_the_same_way(self) -> None:
        """The two properties name the tile in different forms, so they are canonicalised — a
        grouping key that depended on which catalogue answered would split one tile's copies.
        """
        earth_search = _Item("x", "MGRS-33TWM")
        planetary = _Item("y", **{"s2:mgrs_tile": "33TWM"})
        assert item_tile(earth_search) == item_tile(planetary)

    def test_a_value_that_is_not_a_grid_square_is_not_accepted(self) -> None:
        assert item_tile(_Item("not-a-sentinel-id", **{"s2:mgrs_tile": "whatever"})) is None

    def test_the_datatake_names_the_observation_without_its_baseline(self) -> None:
        """The processing baseline is the only field a reprocessing changes, so the head of the
        value is the observation itself.
        """
        item = _Item("x", **{"s2:datatake_id": "GS2B_20171219T095409_004109_N05.00"})
        assert acquisition_identity(item) == "GS2B_20171219T095409_004109"

    @pytest.mark.parametrize(
        "raw", ["", "GS2B_20171219T095409_004109", "not a datatake", "GS2B_20171219T095409_004109_N05.00.1", 5.0]
    )
    def test_anything_that_is_not_a_datatake_id_is_refused(self, raw: object) -> None:
        """Matched as a whole rather than split on underscores, so a value of another shape sends
        the caller to the timestamp instead of yielding a plausible-looking wrong key.
        """
        assert acquisition_identity(_Item("x", **{"s2:datatake_id": raw})) is None

    def test_an_item_without_the_property_falls_back(self) -> None:
        assert acquisition_identity(_Item("x")) is None


class TestReprocessingsAreOneAcquisitionHoweverTheyAreTimestamped:
    """The reported MEDIUM, taken from the committed 2017-12-19 cassette.

    Two copies of one granule — same sensing time, orbit and tile, baselines 02.06 and 05.00 —
    carry catalogue timestamps 209 seconds apart. Clustering on those timestamps put them in
    separate acquisitions, so both survived selection and were mosaicked together, and a
    post-threshold version of the same shape would have refused the date as mixed-producer.
    """

    @staticmethod
    def _reprocessing(sequence: str, baseline: str, acquired: str, host_root: str) -> _Item:
        item = _Item(
            f"S2B_33TWM_20171219_{sequence}_L2A",
            "MGRS-33TWM",
            sequence,
            **{
                "s2:processing_baseline": baseline,
                "s2:datatake_id": f"GS2B_20171219T095409_004109_N{baseline}",
                "datetime": acquired,
            },
        )
        return _with_assets(item, _bands_at(host_root))

    def _cassette_pair(self) -> tuple[_Item, _Item]:
        old = self._reprocessing("0", "02.06", "2017-12-19T09:54:10.457000Z", _REMOTE)
        new = self._reprocessing("1", "05.00", "2017-12-19T09:57:39.063000Z", _IN_REGION)
        return old, new

    def test_the_fixture_really_straddles_the_timestamp_window(self) -> None:
        """Otherwise this would pass on the old timestamp clustering and prove nothing."""
        old, new = self._cassette_pair()
        skew = abs((acquisition_instant(new) - acquisition_instant(old)).total_seconds())
        assert skew > 120, f"the copies are only {skew}s apart — the fixture no longer bites"

    def test_they_reduce_to_one_copy(self) -> None:
        old, new = self._cassette_pair()
        kept, alternates = select_preferred_duplicates([old, new])
        assert [i.id for i in kept] == [new.id], "two reprocessings of one granule were both loaded"
        assert [i.id for i in next(iter(alternates.values()))] == [old.id]

    def test_two_genuine_passes_are_still_kept_apart(self) -> None:
        """The complement, and the reason this cannot simply collapse the tile-date: successive
        orbits revisit a high-latitude tile the same day, and those are separate imagery.
        """
        first = self._reprocessing("0", "05.00", "2017-12-19T09:54:10.457000Z", _IN_REGION)
        first.properties["s2:datatake_id"] = "GS2B_20171219T095409_004109_N05.00"
        second = self._reprocessing("0", "05.00", "2017-12-19T10:44:10.457000Z", _IN_REGION)
        second.id = "S2B_33TWM_20171219_9_L2A"
        second.properties["s2:datatake_id"] = "GS2B_20171219T104409_004110_N05.00"
        kept, _ = select_preferred_duplicates([first, second])
        assert len(kept) == 2, "two distinct passes were collapsed into one"

    def test_a_copy_without_a_datatake_still_clusters_on_its_timestamp(self) -> None:
        """The fallback has to keep working: an item naming no observation is placed by its
        timestamp, exactly as before.
        """
        old, new = self._cassette_pair()
        del old.properties["s2:datatake_id"]
        del new.properties["s2:datatake_id"]
        kept, _ = select_preferred_duplicates([old, new])
        assert len(kept) == 2, "the timestamp fallback stopped clustering"


class TestASpareThatWillRefuseIsKeptOffTheLadderWithoutVisibleAssets:
    """The reported P2/MEDIUM. Two changes on this branch interlock to create it.

    Reading `s2:mgrs_tile` made Planetary Computer items keyable, so that provider has a fallback
    ladder for the first time; deriving the correction from the items made its unreadable
    baselines refuse. Together, a raw spare declaring no baseline could enter the ladder and abort
    the ingest when a read failure stepped down to it — the recovery loop handles a read failure,
    not a refusal.

    The producer cannot be read from that provider's assets, so the collection has to say it.
    """

    @staticmethod
    def _native_key_copy(ident: str, sequence: str, baseline: str | None) -> _Item:
        """A copy whose assets are keyed natively, so nothing can be looked up by band name."""
        extra: dict[str, object] = {"datetime": "2022-01-07T10:20:31.024000Z"}
        if baseline is not None:
            extra["s2:processing_baseline"] = baseline
        item = _Item(ident, "MGRS-33TWM", sequence, **extra)
        item.assets = {k: {"href": f"https://x.blob.core.windows.net/{k}"} for k in ("B02", "SCL")}
        return item

    def test_without_the_collection_answer_the_spare_looks_harmless(self) -> None:
        """The state that produced the finding, asserted so the fix below is not vacuous."""
        spare = self._native_key_copy("no-baseline", "1", None)
        assert refuses_its_date(spare, ()) is False

    def test_with_it_the_spare_is_recognised_as_refusing(self) -> None:
        spare = self._native_key_copy("no-baseline", "1", None)
        assert refuses_its_date(spare, (), Harmonisation.RAW) is True

    def test_such_a_spare_is_excluded_from_the_ladder(self) -> None:
        best = self._native_key_copy("S2A_33TWM_20220107_2_L2A", "2", "05.00")
        unreadable = self._native_key_copy("S2A_33TWM_20220107_1_L2A", "1", None)
        kept, alternates = select_preferred_duplicates([unreadable, best], (), Harmonisation.RAW)
        assert [i.id for i in kept] == [best.id]
        assert alternates == {}, "a spare that will refuse the date must not be offered"

    def test_a_usable_spare_is_still_offered(self) -> None:
        """The complement, or the exclusion above could be excluding everything."""
        best = self._native_key_copy("S2A_33TWM_20220107_2_L2A", "2", "05.00")
        older = self._native_key_copy("S2A_33TWM_20220107_1_L2A", "1", "04.00")
        kept, alternates = select_preferred_duplicates([older, best], (), Harmonisation.RAW)
        assert [i.id for i in kept] == [best.id]
        assert [i.id for i in next(iter(alternates.values()))] == [older.id]

    def test_a_spare_is_kept_when_the_copy_it_replaces_already_owes(self) -> None:
        """The exclusion applies only to a swap that CHANGES the day's decision.

        If the copy being replaced already owes the correction, the day already contains one that
        does, so a spare that also owes it introduces nothing. Withholding it there cost a
        recoverable date on an all-raw day for no gain.
        """
        remote_scl = {"scl": {"href": f"{_REMOTE}/scl"}}
        winner = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="05.10", host_root=_REMOTE)
        spare = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="05.00", host_root=_REMOTE, extra=remote_scl)
        kept, alternates = select_preferred_duplicates([spare, winner])
        assert [i.id for i in kept] == [winner.id]
        assert [i.id for i in next(iter(alternates.values()))] == [spare.id], (
            "both copies owe the correction, so swapping cannot change the day — keep the ladder"
        )

    def test_a_spare_is_still_withheld_when_the_winner_owes_nothing(self) -> None:
        """The complement, or the test above would pass with the exclusion deleted."""
        harmonised = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="04.00", host_root=_IN_REGION)
        raw_spare = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="05.00", host_root=_REMOTE)
        kept, alternates = select_preferred_duplicates([raw_spare, harmonised])
        assert [i.id for i in kept] == [harmonised.id]
        assert alternates == {}, "swapping in a copy that owes the offset would refuse the day"

    def test_the_collection_answer_does_not_invert_the_baseline_preference(self) -> None:
        """The hole this must not reopen: with the producer known raw, every copy at or above the
        threshold owes a correction, so the term ties among them rather than deranking the newest.
        """
        new = self._native_key_copy("S2A_33TWM_20220107_1_L2A", "1", "05.00")
        old = self._native_key_copy("S2A_33TWM_20220107_0_L2A", "0", "03.00")
        kept, _ = select_preferred_duplicates([old, new], (), Harmonisation.RAW)
        assert [i.id for i in kept] == [new.id], "a pre-threshold copy won on owing no correction"

    def test_the_driver_supplies_it_for_the_provider_that_needs_it(self) -> None:
        """The wiring, so the fix cannot be correct in the library and absent at the call site."""
        from tessera_embeddings.ingest.s2_roi import _known_harmonisation, _read_asset_keys

        assert _read_asset_keys("planetary-computer", "sentinel-2-l2a") == ()
        assert _known_harmonisation("planetary-computer", "sentinel-2-l2a") is Harmonisation.RAW
        # Where the producer varies by item the assets ARE the evidence, so there is no
        # collection-wide answer to give.
        assert _known_harmonisation("earth-search", "sentinel-2-l2a") is None
        # And a collection owed no offset at all has no producer worth naming.
        assert _known_harmonisation("earth-search", "sentinel-2-l1c") is None


class TestATimestampOnlyCopyJoinsAnIdentifiedAcquisition:
    """The reported MEDIUM. Identity clusters were built first and timestamp-only copies always
    started their own, so one reprocessing declaring the datatake while its sibling omitted it were
    never compared — and both survived to be fused, which is the defect identity was added to fix.
    """

    @staticmethod
    def _copy_of_one_granule(sequence: str, baseline: str, acquired: str, *, datatake: bool) -> _Item:
        extra: dict[str, object] = {"s2:processing_baseline": baseline, "datetime": acquired}
        if datatake:
            extra["s2:datatake_id"] = f"GS2B_20171219T095409_004109_N{baseline}"
        item = _Item(f"S2B_33TWM_20171219_{sequence}_L2A", "MGRS-33TWM", sequence, **extra)
        return _with_assets(item, _bands_at(_IN_REGION))

    def test_a_sibling_without_a_datatake_still_reduces(self) -> None:
        identified = self._copy_of_one_granule("1", "05.00", "2017-12-19T09:54:10.457000Z", datatake=True)
        anonymous = self._copy_of_one_granule("0", "02.06", "2017-12-19T09:54:41.000000Z", datatake=False)
        kept, alternates = select_preferred_duplicates([anonymous, identified])
        assert [i.id for i in kept] == [identified.id], "the copies were never compared"
        assert [i.id for i in next(iter(alternates.values()))] == [anonymous.id]

    def test_a_distant_timestamp_still_forms_its_own_acquisition(self) -> None:
        """The complement: joining must be decided by the timestamp, not granted to anything
        lacking a datatake, or a genuine second pass would be swallowed.
        """
        identified = self._copy_of_one_granule("1", "05.00", "2017-12-19T09:54:10.457000Z", datatake=True)
        other_pass = self._copy_of_one_granule("0", "05.00", "2017-12-19T10:44:10.457000Z", datatake=False)
        other_pass.id = "S2B_33TWM_20171219_9_L2A"
        kept, _ = select_preferred_duplicates([identified, other_pass])
        assert len(kept) == 2, "a distinct pass was absorbed into an identified acquisition"

    def test_it_is_matched_against_any_member_of_the_acquisition(self) -> None:
        """Members of one acquisition do NOT agree on the timestamp — that disagreement is why
        identity is primary — so closeness to any member is the whole of the evidence.
        """
        early = self._copy_of_one_granule("0", "02.06", "2017-12-19T09:54:10.457000Z", datatake=True)
        late = self._copy_of_one_granule("1", "05.00", "2017-12-19T09:57:39.063000Z", datatake=True)
        # Close to `late` only, and 209s from `early`.
        anonymous = self._copy_of_one_granule("2", "04.00", "2017-12-19T09:57:50.000000Z", datatake=False)
        kept, _ = select_preferred_duplicates([early, late, anonymous])
        assert [i.id for i in kept] == [late.id], "all three are one observation"


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

        Both copies are below the correction threshold, so neither owes an offset correction and
        the term that DOES outrank the baseline is inert. That is what leaves locality alone with
        the baseline, which is the pair this test is about.
        """
        local_old = _copy("S2A_33TWM_20170120_0_L2A", sequence="0", baseline="02.06", host_root=_IN_REGION)
        remote_new = _copy("S2A_33TWM_20170120_1_L2A", sequence="1", baseline="03.01", host_root=_REMOTE)
        kept, _ = select_preferred_duplicates([local_old, remote_new])
        assert [i.id for i in kept] == [remote_new.id], "a newer baseline must win despite egress"

    def test_a_copy_owing_no_correction_outranks_a_newer_one_that_does(self) -> None:
        """The one term that IS allowed to cost a reprocessing, and the reason it may.

        The offset correction is decided per solar day over every tile fused into it, so a single
        raw copy at or above the threshold can refuse the whole day. Avoiding that is a coverage
        decision, not a cost one, so unlike locality it ranks above the baseline.

        Unobservable on the archive as indexed — every copy served from the ESA bucket is at a
        pre-04.00 baseline and so owes nothing either way — which is why the case this pins is
        the one the ingest logs at WARNING as unexpected.
        """
        harmonised_old = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="04.00", host_root=_IN_REGION)
        raw_new = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="05.00", host_root=_REMOTE)
        kept, alternates = select_preferred_duplicates([harmonised_old, raw_new])
        assert [i.id for i in kept] == [harmonised_old.id]
        # And the raw copy is NOT offered as a fallback. It reads fine, but swapping it in beside
        # the harmonised tiles of the same solar day makes that day refuse, and the recovery ladder
        # steps down on a read failure rather than on a refusal — see `risks_refusing_its_date`.
        assert alternates == {}, "a spare that risks refusing the date must not be on the ladder"

    def test_below_the_threshold_the_correction_term_is_inert(self) -> None:
        """The complement of the test above, and what stops it degrading the whole archive.

        A pre-04.00 copy owes no correction whichever bucket serves it, so the term ties and the
        baseline decides. This is the case that actually occurs.
        """
        harmonised_old = _copy("S2A_33TWM_20170120_0_L2A", sequence="0", baseline="02.06", host_root=_IN_REGION)
        raw_new = _copy("S2A_33TWM_20170120_1_L2A", sequence="1", baseline="03.01", host_root=_REMOTE)
        kept, _ = select_preferred_duplicates([harmonised_old, raw_new])
        assert [i.id for i in kept] == [raw_new.id]

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
        """The reported P1, posed where it applies: BOTH copies raw, so the missing baseline is
        the only difference.

        A raw copy declaring no baseline cannot be corrected or exempted on evidence, so its date
        is refused — and a refusal is not a read failure, so nothing recovers from it. The copy
        with a usable baseline must win however its sequence compares.
        """
        raw_unknown = _Item("S2A_33TWM_20220107_1_L2A", "MGRS-33TWM", "1")
        _with_assets(raw_unknown, _bands_at(_REMOTE))
        raw_best = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="05.00", host_root=_REMOTE)
        kept, _ = select_preferred_duplicates([raw_unknown, raw_best])
        assert [i.id for i in kept] == [raw_best.id], "a usable baseline must beat an unreadable one"

    def test_a_harmonised_copy_needs_no_baseline_to_be_usable(self) -> None:
        """The other side, and it does NOT reverse: an unreadable baseline costs a harmonised copy
        nothing, because no offset decision rests on it. Penalising it here would hand the
        tile-date to a raw copy whose date may then be refused outright.
        """
        harmonised_unknown = _Item("S2A_33TWM_20220107_0_L2A", "MGRS-33TWM", "0")
        _with_assets(harmonised_unknown, _bands_at(_IN_REGION))
        raw_best = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="05.00", host_root=_REMOTE)
        kept, _ = select_preferred_duplicates([harmonised_unknown, raw_best])
        assert [i.id for i in kept] == [harmonised_unknown.id]

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
        # And it is not offered as a fallback either: a refusal is not a read failure, so handing
        # it to the ladder would escape rather than step down.
        assert alternates == {}


class TestFallbackLadderKeepsBaselineOrder:
    """Every rung stays in descending baseline order, not just the top one."""

    def test_a_middle_baseline_is_not_skipped_for_a_local_older_one(self) -> None:
        """The reported MEDIUM. With 05.00 selected, a less-local 04.00 and a fully local 03.00, a
        read failure must step to 04.00. Collapsing every non-best baseline into one tier let
        locality hand out the 03.00 copy and needlessly degrade the imagery.

        All three are harmonised, and locality is varied through ``scl`` alone — an asset this
        ingest reads but never corrects. That is what separates the two claims the bucket lists
        make: the copies agree about the producer and disagree about egress, so the term under
        test is locality and nothing else.
        """
        remote_scl = {"scl": {"href": f"{_REMOTE}/scl"}}
        best = _copy("S2A_33TWM_20220107_2_L2A", sequence="2", baseline="05.00", host_root=_IN_REGION, extra=remote_scl)
        middle = _copy(
            "S2A_33TWM_20220107_1_L2A", sequence="1", baseline="04.00", host_root=_IN_REGION, extra=remote_scl
        )
        worst_local = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="03.00", host_root=_IN_REGION)
        assert item_is_in_preferred_location(worst_local)
        assert not item_is_in_preferred_location(best), "the fixture must actually differ in locality"
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

    def test_a_winner_is_audited_even_when_no_spare_survived(self, caplog) -> None:
        """The source breakdown must describe every contested tile-date, not the recoverable ones.

        A tile-date whose every spare is excluded as a refusal risk has no entry in `alternates`.
        Keying the breakdown off that dropped its winner and reported totals for the recoverable
        subset while still labelling them "winners by source" — a silently partial audit of the
        very decision this line exists to record.
        """
        from tessera_embeddings.ingest.duplicates import log_duplicate_selection

        winner = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="04.00", host_root=_IN_REGION)
        raw_spare = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="05.00", host_root=_REMOTE)
        kept, alternates = select_preferred_duplicates([winner, raw_spare])
        assert [i.id for i in kept] == [winner.id]
        assert alternates == {}, "the fixture must produce an empty ladder, or this proves nothing"

        log = logging.getLogger("dup-audit-empty-ladder")
        with caplog.at_level(logging.INFO, logger="dup-audit-empty-ladder"):
            log_duplicate_selection(log, "roi-x", alternates, kept=kept, items=[winner, raw_spare])
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "1 tile-date(s) had more than one copy" in msg
        assert "winners by source: 1 in-region, 0 remote" in msg, msg

    def test_it_states_the_actual_preference(self, caplog) -> None:
        """It said "newest kept" while the ranking preferred an in-region copy over a
        higher-sequence remote one — a line misreporting its own behaviour.
        """
        local = _copy("a", sequence="0", baseline="00.01", host_root=_IN_REGION)
        remote = _copy("b", sequence="1", baseline="00.01", host_root=_REMOTE)
        msg = self._emit(caplog, [local], {("MGRS-33TWM", "2021-09-08"): [remote]})
        assert "complete read set, then a decidable producer, then a known acquisition" in msg
        assert (
            "then a readable baseline, then owing no offset correction, then newest baseline, "
            "then in-region, then newest sequence"
        ) in msg
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
        assert "complete read set, then a decidable producer, then a known acquisition" in msg
        assert (
            "then a readable baseline, then owing no offset correction, then newest baseline, "
            "then in-region, then newest sequence"
        ) in msg
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
        # Sequence decides the winner among copies that all declare nothing readable.
        assert [it.id for it in kept] == ["winner"]
        # The RAW spare is dropped from the ladder — unharmonised with no readable baseline refuses
        # the date, and a refusal is not something the ladder can step down on. The harmonised one
        # stays: it is owed nothing whatever its baseline says, so it cannot refuse.
        assert [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]] == ["local-seq1"]

    def test_locality_still_decides_between_two_equal_known_baselines(self) -> None:
        """The complement: neutralising locality everywhere would pass the test above.

        Posed BELOW the correction threshold so that neither spare owes an offset — above it a raw
        spare is kept off the ladder entirely (`risks_refusing_its_date`), which would empty the
        ladder and prove nothing about locality.
        """
        items = [
            _copy_at("winner", acquired=self._A, sequence="3", baseline="03.01"),
            _copy_at("remote", acquired=self._A, sequence="1", baseline="02.06", host_root=_REMOTE),
            _copy_at("local", acquired=self._A, sequence="1", baseline="02.06", host_root=_IN_REGION),
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
        """Qualified twice, by the two terms that rank ahead of it.

        By completeness, because a copy that cannot be read is no use at any baseline and the
        generic paths have no recovery for a missing band. And by producer, because a harmonised
        copy owes no correction and so is compared on different terms from a raw one — the
        readable-baseline rule holds within each producer, which is where it is claimed.
        """
        pool = [i for i in self._pool() if read_set_is_complete(i)]
        for producer in (True, False):
            same = [i for i in pool if item_is_in_preferred_location(i) is producer]
            known = [i for i in same if item_processing_baseline(i) is not None]
            unknown = [i for i in same if item_processing_baseline(i) is None]
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
        """The P1 this ordering exists to prevent: cheaper egress must not buy an older pixel.

        Both baselines are below the correction threshold, so neither copy owes an offset and the
        one term that does outrank the baseline is inert. Locality is then alone with the
        baseline, which is what this asserts.
        """
        remote_new = self._item("remote-new", baseline="03.01", sequence="0", local=False, complete=True)
        local_old = self._item("local-old", baseline="02.06", sequence="9", local=True, complete=True)
        assert _preference_key(remote_new) < _preference_key(local_old)

    def test_locality_does_not_participate_when_the_baseline_is_unreadable(self) -> None:
        """Compared like with like: both raw, so the baseline-readable term ties and only locality
        and sequence are left. A harmonised copy would not tie here — it needs no correction, so an
        unreadable baseline costs it nothing.
        """
        local = self._item("local", baseline=None, sequence="1", local=False, complete=True)
        remote = self._item("remote", baseline=None, sequence="2", local=False, complete=True)
        assert _preference_key(remote) < _preference_key(local), "locality decided a baseline-free comparison"

    def test_a_harmonised_copy_is_not_penalised_on_the_readable_term(self) -> None:
        """It needs no correction whatever the baseline says, so an unreadable baseline is not
        evidence against it the way it is against a raw copy — which refuses its date.

        Note the narrower claim: this is about the READABLE term, not the baseline VALUE. A
        harmonised copy declaring nothing still loses to a raw copy declaring 03.00, because an
        absent baseline is no vintage information at all and this module does not let absence of
        evidence win a comparison. That is consistent rather than accidental.
        """
        harmonised = self._item("harmonised", baseline=None, sequence="0", local=True, complete=True)
        raw_same = self._item("raw-no-baseline", baseline=None, sequence="9", local=False, complete=True)
        assert _preference_key(harmonised) < _preference_key(raw_same)

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


class TestACopyThatWouldRefuseItsDateLosesToOneThatWouldNot:
    """A copy whose reflectance bands span a harmonised and a raw producer refuses its date at or
    above the correction threshold, and nothing retries a refusal — the fallback ladder steps down
    on a read error, not on this. So it must lose to a homogeneous copy, even an older one.
    Raised on PR #107.
    """

    @staticmethod
    def _copy_at_baseline(ident: str, baseline: str, *, mixed: bool, sequence: str = "0") -> _Item:
        assets = _bands_at(_IN_REGION)
        if mixed:
            assets["red"] = {"href": f"{_REMOTE}/B04.jp2"}
        return _with_assets(_Item(ident, "MGRS-33TWM", sequence, **{"s2:processing_baseline": baseline}), assets)

    def test_a_homogeneous_older_copy_beats_a_mixed_newer_one(self) -> None:
        mixed_new = self._copy_at_baseline("mixed-05", "05.00", mixed=True)
        clean_old = self._copy_at_baseline("clean-04", "04.00", mixed=False)
        kept, alternates = select_preferred_duplicates([mixed_new, clean_old])
        assert [it.id for it in kept] == ["clean-04"]
        # The mixed copy is excluded from the ladder as well as losing the selection: offering it
        # would raise `HeterogeneousProducerError`, which is not a read failure.
        assert alternates == {}

    def test_below_the_threshold_the_baseline_still_decides(self) -> None:
        """The term is inert there: a mixed producer changes no pixel that gets no offset."""
        mixed_new = self._copy_at_baseline("mixed-03", "03.00", mixed=True)
        clean_old = self._copy_at_baseline("clean-02", "02.00", mixed=False)
        kept, _ = select_preferred_duplicates([mixed_new, clean_old])
        assert [it.id for it in kept] == ["mixed-03"]

    def test_two_homogeneous_copies_are_still_ranked_on_baseline(self) -> None:
        """The complement, or the term could be penalising everything."""
        new = self._copy_at_baseline("clean-05", "05.00", mixed=False)
        old = self._copy_at_baseline("clean-04", "04.00", mixed=False)
        kept, _ = select_preferred_duplicates([new, old])
        assert [it.id for it in kept] == ["clean-05"]

    def test_an_incomplete_copy_is_not_penalised_twice(self) -> None:
        """It is already last on completeness, and reports an undecidable producer BECAUSE it is
        incomplete — counting that again inverted the baseline preference among incomplete copies.
        """
        assets = {k: {"href": f"{_IN_REGION}/{k}"} for k in READ_ASSET_KEYS[:3]}
        newer = _with_assets(_Item("partial-05", "MGRS-33TWM", "0", **{"s2:processing_baseline": "05.00"}), assets)
        older = _with_assets(
            _Item("partial-04", "MGRS-33TWM", "0", **{"s2:processing_baseline": "04.00"}), dict(assets)
        )
        kept, _ = select_preferred_duplicates([newer, older])
        assert [it.id for it in kept] == ["partial-05"]


class TestTheLadderOnlyOffersCopiesItCanRecoverWith:
    """A refusal is not a read failure, so the ladder cannot step down on one.

    `step_down_copies` exists for a source object that will not read. Handing it a copy that makes
    `dates_exempt_from_correction` raise means the exception escapes instead of the next usable
    copy being tried or the date being recorded as lost. Raised on PR #107.
    """

    def _copy(
        self, ident: str, *, baseline: str | None, sequence: str, mixed: bool = False, raw: bool = False
    ) -> _Item:
        extra: dict[str, object] = {}
        if baseline is not None:
            extra["s2:processing_baseline"] = baseline
        # A HARMONISED copy is owed nothing whatever its baseline says, so it never refuses — the
        # unreadable-baseline case only bites an unharmonised one.
        assets = _bands_at(_REMOTE if raw else _IN_REGION)
        if mixed:
            assets["red"] = {"href": f"{_REMOTE}/B04.jp2"}
        return _with_assets(_Item(ident, "MGRS-33TWM", sequence, **extra), assets)

    def test_a_harmonised_spare_with_no_baseline_is_still_offered(self) -> None:
        """It is owed nothing, so it cannot refuse, so it remains a usable fallback."""
        winner = self._copy("win", baseline="05.00", sequence="9")
        spare = self._copy("harmonised-no-baseline", baseline=None, sequence="1")
        _, alternates = select_preferred_duplicates([winner, spare])
        assert [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]] == ["harmonised-no-baseline"]

    def test_a_usable_spare_is_still_offered(self) -> None:
        """The complement first: the ladder must not have been emptied wholesale."""
        winner = self._copy("win", baseline="05.00", sequence="9")
        spare = self._copy("spare", baseline="04.00", sequence="1")
        _, alternates = select_preferred_duplicates([winner, spare])
        assert [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]] == ["spare"]

    def test_a_spare_with_an_unreadable_baseline_is_excluded(self) -> None:
        winner = self._copy("win", baseline="05.00", sequence="9")
        unusable = self._copy("no-baseline", baseline=None, sequence="1", raw=True)
        _, alternates = select_preferred_duplicates([winner, unusable])
        assert alternates == {}

    def test_a_mixed_producer_spare_is_excluded_above_the_threshold(self) -> None:
        winner = self._copy("win", baseline="05.00", sequence="9")
        unusable = self._copy("mixed", baseline="05.00", sequence="1", mixed=True)
        _, alternates = select_preferred_duplicates([winner, unusable])
        assert alternates == {}

    def test_a_mixed_producer_spare_is_kept_below_the_threshold(self) -> None:
        """Nothing is owed there, so the date does not refuse and the copy is a usable fallback."""
        winner = self._copy("win", baseline="03.00", sequence="9")
        spare = self._copy("mixed", baseline="03.00", sequence="1", mixed=True)
        _, alternates = select_preferred_duplicates([winner, spare])
        assert [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]] == ["mixed"]

    def test_the_usable_spares_survive_alongside_an_excluded_one(self) -> None:
        winner = self._copy("win", baseline="05.10", sequence="9")
        usable = self._copy("usable", baseline="05.00", sequence="2")
        unusable = self._copy("no-baseline", baseline=None, sequence="1", raw=True)
        _, alternates = select_preferred_duplicates([winner, usable, unusable])
        assert [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]] == ["usable"]


class TestKnownAcquisitionsSurviveAnUndatedSibling:
    """Two readable passes plus one undated copy must keep BOTH passes.

    Raised four times in review, and I twice reported it fixed when it was not — a later edit had
    reinstated the collapse while leaving the docstring describing the replacement. These assert
    the outcome rather than the mechanism, so a revert fails here instead of in a fifth review.
    """

    _A = "2021-09-08T10:00:00Z"
    _B = "2021-09-08T14:00:00Z"

    def _at(self, ident: str, acquired: str | None, sequence: str) -> _Item:
        extra: dict[str, object] = {"s2:processing_baseline": "05.00"}
        if acquired is not None:
            extra["datetime"] = acquired
        return _with_assets(_Item(ident, "MGRS-33TWM", sequence, **extra), _bands_at(_IN_REGION))

    def test_both_readable_passes_survive(self) -> None:
        items = [
            self._at("pass-a", self._A, "0"),
            self._at("pass-b", self._B, "0"),
            self._at("undated", None, "1"),
        ]
        kept, _ = select_preferred_duplicates(items)
        assert sorted(it.id for it in kept) == ["pass-a", "pass-b"], "a readable pass was discarded"

    def test_the_undated_copy_competes_rather_than_surviving_alongside(self) -> None:
        """It must not add a third survivor — that is the fusion this module prevents."""
        items = [self._at("pass-a", self._A, "0"), self._at("undated", None, "1")]
        kept, alternates = select_preferred_duplicates(items)
        assert len(kept) == 1, "the undated copy survived as its own acquisition"
        assert alternates[("MGRS-33TWM", "2021-09-08")], "the loser must remain a fallback"

    def test_three_passes_and_two_undated_keep_three_survivors(self) -> None:
        items = [
            self._at("pass-a", self._A, "0"),
            self._at("pass-b", self._B, "0"),
            self._at("pass-c", "2021-09-08T18:00:00Z", "0"),
            self._at("undated-1", None, "1"),
            self._at("undated-2", None, "2"),
        ]
        kept, _ = select_preferred_duplicates(items)
        assert len(kept) == 3, f"expected one per readable pass, got {[i.id for i in kept]}"

    def test_all_undated_still_reduces_to_one(self) -> None:
        """With no readable instant anywhere there is no evidence of distinctness at all."""
        items = [self._at("u1", None, "0"), self._at("u2", None, "1")]
        kept, _ = select_preferred_duplicates(items)
        assert len(kept) == 1


class TestTheAuditReportsWhatSelectionDidNotWhatSurvivedTheFilter:
    """The ladder filter drops copies that would refuse, so the audit could not be derived from it.

    On a date whose every spare was unusable the INFO line vanished entirely, and elsewhere it
    undercounted while claiming every rejected copy was still available. Raised on PR #107.
    """

    def _emit(self, caplog, supplied, kept, alternates) -> str:
        log = logging.getLogger("dup-audit-counts")
        with caplog.at_level(logging.INFO, logger="dup-audit-counts"):
            from tessera_embeddings.ingest.duplicates import log_duplicate_selection

            log_duplicate_selection(log, "roi-x", alternates, kept=kept, items=supplied)
        return " ".join(r.getMessage() for r in caplog.records)

    def _copy(self, ident: str, *, baseline: str | None, raw: bool = False) -> _Item:
        extra: dict[str, object] = {}
        if baseline is not None:
            extra["s2:processing_baseline"] = baseline
        return _with_assets(_Item(ident, "MGRS-33TWM", "0", **extra), _bands_at(_REMOTE if raw else _IN_REGION))

    def test_the_line_appears_and_separates_rejected_from_recoverable(self, caplog) -> None:
        winner = self._copy("win", baseline="05.00")
        refuser = self._copy("refuser", baseline=None, raw=True)
        supplied = [winner, refuser]
        kept, alternates = select_preferred_duplicates(supplied)
        assert alternates == {}, "the fixture must have had its only spare filtered out"
        msg = self._emit(caplog, supplied, kept, alternates)
        assert msg, "the audit vanished on a date whose spare was unusable"
        assert "1 rejected, 0 of those available as a fallback" in msg
        # Counted by multiplicity: the tile-date keeps its key while losing a copy, so a set
        # difference reported 0 contested dates beside 1 rejected copy — a self-contradiction.
        assert "1 tile-date(s) had more than one copy" in msg

    def test_a_usable_spare_is_reported_as_recoverable(self, caplog) -> None:
        winner = self._copy("win", baseline="05.10")
        usable = self._copy("usable", baseline="05.00")
        supplied = [winner, usable]
        kept, alternates = select_preferred_duplicates(supplied)
        msg = self._emit(caplog, supplied, kept, alternates)
        assert "1 rejected, 1 of those available as a fallback" in msg

    def test_a_multi_pass_date_that_lost_nothing_is_not_counted_as_contested(self, caplog) -> None:
        """A tile-date holding two genuinely distinct same-day passes has multiplicity above one
        while losing nothing, so counting every multi-item key inflated the figure with dates no
        choice was made on.
        """
        two_passes = [
            _with_assets(
                _Item(f"pass-{tag}", "MGRS-33TWA", "0", **{"s2:processing_baseline": "05.00", "datetime": acq}),
                _bands_at(_IN_REGION),
            )
            for tag, acq in (("a", "2021-09-08T10:00:00Z"), ("b", "2021-09-08T14:00:00Z"))
        ]
        # ...and one real duplicate on a DIFFERENT tile, so the line is emitted at all.
        win = self._copy("win", baseline="05.10")
        spare = self._copy("spare", baseline="05.00")
        supplied = [*two_passes, win, spare]
        kept, alternates = select_preferred_duplicates(supplied)
        assert len(kept) == 3, "both distinct passes must survive alongside the duplicate's winner"
        msg = self._emit(caplog, supplied, kept, alternates)
        assert "1 tile-date(s) had more than one copy" in msg, f"multi-pass date inflated it: {msg}"
        assert "1 rejected" in msg

    def test_nothing_pruned_logs_nothing(self, caplog) -> None:
        only = self._copy("only", baseline="05.00")
        assert self._emit(caplog, [only], [only], {}) == ""


class TestTheLadderStepsNothingWhenTheFailedAcquisitionHasNoSpare:
    """Attribution named the failing objects and no copy belongs to their acquisition.

    Falling back to the best overall alternate there swapped a HEALTHY acquisition down to an older
    copy while leaving the known-bad one selected, then rebuilt and re-read the whole date only to
    fail identically — once per unrelated spare before recording the loss. Raised on PR #107.
    """

    _A = "2021-09-08T10:00:00Z"
    _B = "2021-09-08T14:00:00Z"

    def _at(self, ident: str, acquired: str, sequence: str, baseline: str = "05.00") -> _Item:
        return _with_assets(
            _Item(ident, "MGRS-33TWM", sequence, **{"s2:processing_baseline": baseline, "datetime": acquired}),
            _bands_at(_IN_REGION),
        )

    def test_nothing_is_stepped_when_only_a_healthy_acquisition_has_a_spare(self) -> None:
        bad = self._at("bad-a", self._A, "1")  # the one that failed; no spare of its own
        healthy = self._at("healthy-b", self._B, "1")  # a different acquisition...
        spare_b = self._at("spare-b", self._B, "0")  # ...which DOES have a spare
        kept, alternates = select_preferred_duplicates([bad, healthy, spare_b])
        assert sorted(i.id for i in kept) == ["bad-a", "healthy-b"]

        stepped = step_down_copies(alternates, kept, implicated=[bad])
        assert stepped is None, "a healthy acquisition was downgraded for a failure it did not have"

    def test_the_failed_acquisition_is_stepped_when_it_does_have_a_spare(self) -> None:
        """The complement, or returning None unconditionally would pass the test above."""
        bad = self._at("bad-a", self._A, "1")
        spare_a = self._at("spare-a", self._A, "0")
        healthy = self._at("healthy-b", self._B, "1")
        kept, alternates = select_preferred_duplicates([bad, spare_a, healthy])
        stepped = step_down_copies(alternates, kept, implicated=[bad])
        assert stepped is not None
        swapped, _keys = stepped
        assert "spare-a" in [i.id for i in swapped], "the failed acquisition was not stepped"
        assert "healthy-b" in [i.id for i in swapped], "the healthy acquisition must be untouched"

    def test_unattributed_failure_still_steps_the_best_spare(self) -> None:
        """With nothing attributed there is no acquisition to match, and the pre-existing
        behaviour — take the best-ranked alternate — is still right.
        """
        first = self._at("first", self._A, "1")
        spare = self._at("spare", self._A, "0")
        kept, alternates = select_preferred_duplicates([first, spare])
        stepped = step_down_copies(alternates, kept, implicated=[])
        assert stepped is not None
        assert "spare" in [i.id for i in stepped[0]]


class TestAnUndatedCopyCannotDisplaceADatedPass:
    """`_by_acquisition` attaches an undated copy to a cluster ARBITRARILY, so it must not win it.

    The acquisition-instant term used to sit below baseline and locality, so an undated 05.10 copy
    beat that cluster's dated 05.00 pass — representing a real pass with an item that may belong to
    a different one, and potentially duplicating another pass while dropping this one's coverage.
    Raised on PR #107.
    """

    _A = "2021-09-08T10:00:00Z"
    _B = "2021-09-08T14:00:00Z"

    def _at(self, ident: str, acquired: str | None, baseline: str, sequence: str = "0") -> _Item:
        extra: dict[str, object] = {"s2:processing_baseline": baseline}
        if acquired is not None:
            extra["datetime"] = acquired
        return _with_assets(_Item(ident, "MGRS-33TWM", sequence, **extra), _bands_at(_IN_REGION))

    def test_a_dated_pass_beats_an_undated_copy_at_a_higher_baseline(self) -> None:
        dated = self._at("dated-05.00", self._A, "05.00")
        undated = self._at("undated-05.10", None, "05.10")
        kept, _ = select_preferred_duplicates([dated, undated])
        assert [i.id for i in kept] == ["dated-05.00"], "an arbitrary attachment displaced a real pass"

    def test_a_dated_pass_beats_an_undated_copy_in_region(self) -> None:
        """Locality sits below this too, so cheaper egress cannot buy an unknown pass either."""
        dated = _with_assets(
            _Item("dated-remote", "MGRS-33TWM", "0", **{"s2:processing_baseline": "05.00", "datetime": self._A}),
            _bands_at(_REMOTE),
        )
        undated = self._at("undated-local", None, "05.00")
        kept, _ = select_preferred_duplicates([dated, undated])
        assert [i.id for i in kept] == ["dated-remote"]

    def test_two_dated_copies_are_still_ranked_on_baseline(self) -> None:
        """The complement: the new term must only separate dated from undated."""
        newer = self._at("dated-05.10", self._A, "05.10")
        older = self._at("dated-05.00", self._A, "05.00")
        kept, _ = select_preferred_duplicates([newer, older])
        assert [i.id for i in kept] == ["dated-05.10"]

    def test_both_passes_still_survive_with_an_undated_copy_present(self) -> None:
        """And the coverage guarantee still holds: one survivor per readable pass."""
        items = [
            self._at("pass-a", self._A, "05.00"),
            self._at("pass-b", self._B, "05.00"),
            self._at("undated", None, "05.10", sequence="9"),
        ]
        kept, _ = select_preferred_duplicates(items)
        assert sorted(i.id for i in kept) == ["pass-a", "pass-b"]


class TestAttributedRecoveryOnlyUsesCopiesItCanPlace:
    """A spare with no readable instant has an arbitrary — and UNSTABLE — acquisition.

    `_first_for_failed_acquisition` clusters a candidate against the implicated copies, while
    `_alternate_for` clusters it against the survivors and lands on the earliest. So an undated
    spare could be chosen for the acquisition that failed and then swapped onto a healthy one:
    the spare is consumed, the failure stays selected, and the date is not recovered. Raised on
    PR #107.
    """

    _A = "2021-09-08T10:00:00Z"
    _B = "2021-09-08T14:00:00Z"

    def _at(self, ident: str, acquired: str | None, sequence: str = "0") -> _Item:
        extra: dict[str, object] = {"s2:processing_baseline": "05.00"}
        if acquired is not None:
            extra["datetime"] = acquired
        return _with_assets(_Item(ident, "MGRS-33TWM", sequence, **extra), _bands_at(_IN_REGION))

    def test_an_undated_spare_is_not_offered_for_an_attributed_failure(self) -> None:
        undated = self._at("undated", None, "9")
        failed = self._at("failed-b", self._B, "1")
        assert _first_for_failed_acquisition([undated], [failed]) is None

    def test_a_dated_spare_of_the_failed_acquisition_is_offered(self) -> None:
        """The complement, or excluding everything would pass the test above."""
        spare = self._at("spare-b", self._B, "0")
        failed = self._at("failed-b", self._B, "1")
        chosen = _first_for_failed_acquisition([spare], [failed])
        assert chosen is not None and chosen.id == "spare-b"

    def test_an_undated_spare_is_still_used_when_nothing_is_attributed(self) -> None:
        """With no acquisition to match there is nothing to place it against, and the
        pre-existing behaviour — take the best-ranked spare — remains right.
        """
        undated = self._at("undated", None, "9")
        chosen = _first_for_failed_acquisition([undated], [])
        assert chosen is not None and chosen.id == "undated"

    def test_the_healthy_acquisition_is_not_swapped_end_to_end(self) -> None:
        """The consequence the reviewer named, through the real entry point."""
        pass_a = self._at("pass-a", self._A, "1")
        pass_b = self._at("pass-b", self._B, "1")
        undated = self._at("undated-spare", None, "9")
        kept, alternates = select_preferred_duplicates([pass_a, pass_b, undated])
        assert sorted(i.id for i in kept) == ["pass-a", "pass-b"]
        stepped = step_down_copies(alternates, kept, implicated=[pass_b])
        if stepped is not None:
            swapped = [i.id for i in stepped[0]]
            assert "pass-a" in swapped, "the healthy acquisition was replaced by an unplaceable spare"


class TestASequenceIsMatchedNotCoerced:
    """`s2:sequence` is validated as a sequence, the same way a baseline is validated as a version.

    `int()` accepted a long list of things a sequence is not: it truncates 1.9 to 1, and a numeric
    type also takes `-1`, `1e3` and Unicode digits, while an infinity raised an uncaught
    OverflowError. Any of those would order the copies instead of deferring to the sequence encoded
    in the item id. Organised by category, so a newly found malformed input goes in the list rather
    than into a new guard.
    """

    #: The id carries sequence 7, so anything unreadable must fall back to that.
    _ID = "S2B_34WFA_20210908_7_L2A"

    def _item(self, raw: object) -> _Item:
        item = _Item(self._ID, "MGRS-34WFA")
        if raw is not None:
            item.properties["s2:sequence"] = raw
        return item

    @pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("3", 3), ("12", 12), (3, 3)])
    def test_a_real_sequence_parses(self, raw: object, expected: int) -> None:
        assert item_sequence(self._item(raw)) == expected

    @pytest.mark.parametrize(
        ("label", "raw"),
        [
            ("truncating-float", 1.9),
            ("truncating-string", "1.9"),
            ("infinity", float("inf")),
            ("nan", float("nan")),
            ("negative", "-1"),
            ("signed", "+1"),
            ("exponent", "1e3"),
            ("hex", "0x1"),
            ("arabic-indic", chr(0x664)),
            ("empty", ""),
            ("spaces", "   "),
            ("word", "one"),
            ("list", []),
            ("bool", True),
        ],
        ids=lambda v: v if isinstance(v, str) and v.replace("-", "").isalpha() else None,
    )
    def test_anything_else_defers_to_the_id(self, label: str, raw: object) -> None:
        assert item_sequence(self._item(raw)) == 7, label

    def test_an_absent_property_defers_to_the_id(self) -> None:
        assert item_sequence(self._item(None)) == 7

    def test_an_unreadable_sequence_with_no_id_sequence_is_none(self) -> None:
        """Ordered below every known sequence, rather than guessed at."""
        item = _Item("no-sequence-here", "MGRS-34WFA")
        item.properties["s2:sequence"] = "1.9"
        assert item_sequence(item) is None


class TestTheSteppedSetNamesOnlyWhatWasStepped:
    """The caller labels its retry from this set and records it as `copies_tried`.

    Returning every tile-date considered claims a copy was tried on dates nothing touched — during
    exactly the data-loss investigation that record exists for. Raised on PR #107.
    """

    _A = "2021-09-08T10:00:00Z"

    def _at(self, ident: str, tile: str, acquired: str, sequence: str) -> _Item:
        return _with_assets(
            _Item(ident, tile, sequence, **{"s2:processing_baseline": "05.00", "datetime": acquired}),
            _bands_at(_IN_REGION),
        )

    def test_only_the_tile_date_with_a_usable_spare_is_reported(self) -> None:
        # Tile A: the failure has a spare of its own acquisition, so it steps.
        a_bad = self._at("a-bad", "MGRS-33TWA", self._A, "1")
        a_spare = self._at("a-spare", "MGRS-33TWA", self._A, "0")
        # Tile B: also failed, but its only spare belongs to a DIFFERENT acquisition.
        b_bad = self._at("b-bad", "MGRS-33TWB", self._A, "1")
        b_other = self._at("b-other", "MGRS-33TWB", "2021-09-08T14:00:00Z", "1")
        b_other_spare = self._at("b-other-spare", "MGRS-33TWB", "2021-09-08T14:00:00Z", "0")

        kept, alternates = select_preferred_duplicates([a_bad, a_spare, b_bad, b_other, b_other_spare])
        stepped = step_down_copies(alternates, kept, implicated=[a_bad, b_bad])
        assert stepped is not None
        _swapped, keys = stepped
        assert keys == {("MGRS-33TWA", "2021-09-08")}, f"reported a tile-date it never stepped: {keys}"

    def test_a_spare_is_not_consumed_from_a_tile_date_that_did_not_step(self) -> None:
        """Removing it from the ladder before knowing it would be used loses a copy for nothing."""
        b_bad = self._at("b-bad", "MGRS-33TWB", self._A, "1")
        b_other = self._at("b-other", "MGRS-33TWB", "2021-09-08T14:00:00Z", "1")
        b_other_spare = self._at("b-other-spare", "MGRS-33TWB", "2021-09-08T14:00:00Z", "0")
        a_bad = self._at("a-bad", "MGRS-33TWA", self._A, "1")
        a_spare = self._at("a-spare", "MGRS-33TWA", self._A, "0")

        kept, alternates = select_preferred_duplicates([a_bad, a_spare, b_bad, b_other, b_other_spare])
        before = [i.id for i in alternates[("MGRS-33TWB", "2021-09-08")]]
        step_down_copies(alternates, kept, implicated=[a_bad, b_bad])
        after = [i.id for i in alternates[("MGRS-33TWB", "2021-09-08")]]
        assert after == before, "a spare was consumed from a tile-date nothing stepped"


class TestTheProducerTermIsInertWithoutAReadSet:
    """An empty read set means the collection's configured names are not its asset keys.

    `item_harmonisation` then finds nothing and reports UNKNOWN for every copy, so a term that
    treats UNKNOWN as a refusal condemns them all — demoting every post-threshold copy and stripping
    it from the fallback ladder, which hands the tile-date to an older pre-threshold one. Planetary
    Computer is exactly that case: it serves `B02`/`SCL` and its correction is decided at collection
    level, so this term has nothing to say there. Raised on PR #107.
    """

    def _native(self, ident: str, baseline: str, sequence: str) -> _Item:
        item = _Item(ident, "MGRS-33TWM", sequence, **{"s2:processing_baseline": baseline})
        item.properties["datetime"] = "2024-06-05T10:20:31.024000Z"
        # Native keys, as Planetary Computer serves them — none of the configured names resolve.
        return _with_assets(item, {k: {"href": f"https://x.blob.core.windows.net/{k}"} for k in ("B02", "B03")})

    def test_a_post_threshold_copy_does_not_read_as_refusing(self) -> None:
        assert refuses_its_date(self._native("new", "05.10", "1"), ()) is False

    def test_the_newer_copy_still_wins(self) -> None:
        """The consequence: without this, the older pre-threshold copy took the tile-date."""
        newer = self._native("new-05.10", "05.10", "1")
        older = self._native("old-03.00", "03.00", "0")
        assert _preference_key(newer, ()) < _preference_key(older, ())

    def test_the_spare_stays_on_the_fallback_ladder(self) -> None:
        newer = self._native("new-05.10", "05.10", "1")
        older = self._native("old-03.00", "03.00", "0")
        kept, alternates = select_preferred_duplicates([newer, older], ())
        assert [i.id for i in kept] == ["new-05.10"]
        assert [i.id for i in alternates[("MGRS-33TWM", "2021-09-08")]] == ["old-03.00"]

    def test_the_term_still_fires_where_the_names_are_the_keys(self) -> None:
        """The complement, or gating it would have disabled the guard everywhere."""
        mixed = _Item("mixed", "MGRS-33TWM", "0", **{"s2:processing_baseline": "05.00"})
        mixed.properties["datetime"] = "2024-06-05T10:20:31.024000Z"
        assets = _bands_at(_IN_REGION)
        assets["red"] = {"href": f"{_REMOTE}/B04.jp2"}
        _with_assets(mixed, assets)
        assert refuses_its_date(mixed, READ_ASSET_KEYS) is True
