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
from rasterio.errors import RasterioIOError, WarpOperationError

from tessera_embeddings.ingest.asset_locations import (
    PREFERRED_ASSET_BUCKETS,
    READ_ASSET_KEYS,
    Harmonisation,
    asset_bucket,
    item_is_in_preferred_location,
    read_set_is_complete,
)
from tessera_embeddings.ingest.duplicates import (
    _preference_key,
    alternates_for,
    cause_was_flattened,
    copies_label,
    image_identity,
    item_processing_baseline,
    item_sequence,
    item_tile,
    refuses_its_date,
    select_preferred_duplicates,
    step_down_copies,
)
from tessera_embeddings.ingest.solar_days import normalize_to_solar_day


def _image(pass_no: int = 0, strip_no: int = 0) -> dict[str, str]:
    """The identifiers of one image: doubles built from the same numbers are copies of one image."""
    return {
        "s2:datatake_id": f"GS2B_20210908T{100000 + pass_no:06d}_{pass_no:06d}_N05.00",
        "s2:datastrip_id": f"S2B_OPER_MSI_L2A_DS_TEST_20210908T120000_S20210908T{110000 + strip_no:06d}_N05.00",
    }


class _Item:
    """Stands in for a pystac Item, carrying only what the selector reads.

    A copy of one image unless the test says otherwise (:func:`_image`), because selection only
    compares copies of one image; pass ``None`` for an identifier to make a copy that names none.
    """

    def __init__(self, ident: str, tile: str | None = None, sequence: str | None = None, **extra: object) -> None:
        self.id = ident
        # Every item reaching the selector has passed normalize_to_solar_day, whose canonical
        # stamp is noon UTC; solar_day_of RAISES on anything else, so the fixture must match.
        self.datetime = datetime(2021, 9, 8, 12, 0, 0, tzinfo=UTC)
        self.properties: dict[str, object] = {**_image(), **extra}
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
    """Selection on real `pystac.Item`s after the real `normalize_to_solar_day`, not on doubles."""

    @staticmethod
    def _real(ident: str, sequence: str, image: dict[str, str]) -> pystac.Item:
        return pystac.Item(
            id=ident,
            geometry=None,
            bbox=None,
            datetime=datetime(2021, 9, 8, 10, 20, 31, tzinfo=UTC),
            properties={"grid:code": "MGRS-33XVG", "s2:sequence": sequence, **image},
        )

    def test_copies_of_one_image_reduce_and_another_image_survives(self) -> None:
        old = self._real("S2B_33XVG_20210908_0_L2A", "0", _image(0))
        new = self._real("S2B_33XVG_20210908_1_L2A", "1", _image(0))
        other = self._real("S2A_33XVG_20210908_0_L2A", "0", _image(1))
        normalize_to_solar_day([old, new, other], mid_longitude=15.0)

        kept, alternates = select_preferred_duplicates([old, new, other])
        assert kept == [new, other]
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


class TestChoosingBetweenCopies:
    """Reducing a tile-date to one copy, deterministically and without losing imagery."""

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
        """The caller sorts clearest-first for the fusion order; the selector must not reorder
        that or the clearest tile stops winning the overlap.
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


class TestKeyingACopyAcrossProviders:
    """The tile and the image, read from fields each catalogue actually populates."""

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

    def test_an_image_is_its_pass_and_datastrip_without_the_baseline(self) -> None:
        """A reprocessing changes the baseline and the processing fields and nothing else."""
        item = _Item(
            "x",
            **{
                "s2:datatake_id": "GS2B_20171219T095409_004109_N05.00",
                "s2:datastrip_id": "S2B_OPER_MSI_L2A_DS_S2RP_20230527T013126_S20171219T095412_N05.00",
            },
        )
        assert image_identity(item) == ("GS2B_20171219T095409_004109", "20171219T095412")

    @pytest.mark.parametrize("field", ["s2:datatake_id", "s2:datastrip_id"])
    @pytest.mark.parametrize("raw", [None, "GS2B_20171219T095409_004109"])
    def test_an_unreadable_identifier_names_no_image(self, field: str, raw: object) -> None:
        """Matched as a whole rather than split on underscores, so a value of another shape is
        unknown instead of a plausible-looking wrong key.
        """
        assert image_identity(_Item("x", **{field: raw})) is None


def _granule(ident: str, baseline: str, datatake: str, strip: str) -> _Item:
    """A real catalogue item, reduced to what selection reads; ``strip`` is the datastrip id's tail."""
    _, tile, _, sequence, _ = ident.split("_")
    return _Item(
        ident,
        f"MGRS-{tile}",
        sequence,
        **{
            "s2:processing_baseline": baseline,
            "s2:datatake_id": f"{datatake}_N{baseline}",
            "s2:datastrip_id": f"{ident[:3]}_OPER_MSI_L2A_DS_{strip}_N{baseline}",
        },
    )


class TestADatastripSplitIsTwoImages:
    """One pass crossing a datastrip boundary is published as one granule per datastrip.

    Both share the datatake and baseline, so naming an image by the datatake alone made them
    compete and dropped the loser's ground. Values are as Earth Search publishes them; see
    ``context_docs/ingest/split-datastrips.md``.
    """

    # 18NUN: `_0_` covers the whole tile, `_1_` a sliver at its south edge.
    NUN = (
        _granule(
            "S2B_18NUN_20250619_0_L2A", "05.11", "GS2B_20250619T153629_043280", "2BPS_20250619T191325_S20250619T153625"
        ),
        _granule(
            "S2B_18NUN_20250619_1_L2A", "05.11", "GS2B_20250619T153629_043280", "2BPS_20250619T204702_S20250619T154121"
        ),
    )
    # 33UUP: the split survived reprocessing, two datastrips at 02.13 and the same two at 05.00.
    UUP = tuple(
        _granule(f"S2A_33UUP_20191021_{seq}_L2A", base, "GS2A_20191021T101041_022613", strip)
        for seq, base, strip in (
            ("0", "02.13", "MPS__20191021T105520_S20191021T101549"),
            ("1", "02.13", "EPAE_20191021T115008_S20191021T101730"),
            ("2", "05.00", "S2RP_20230527T013126_S20191021T101549"),
            ("3", "05.00", "S2RP_20230705T190725_S20191021T101730"),
        )
    )
    # 17QRE: the 05.00 reprocessing published only one of the two 00.01 datastrips.
    QRE = tuple(
        _granule(f"S2B_17QRE_20180622_{seq}_L2A", base, "GS2B_20180622T155529_006758", strip)
        for seq, base, strip in (
            ("0", "00.01", "SHIT_20200930T000431_S20180622T155524"),
            ("1", "00.01", "SHIT_20200713T180011_S20180622T155654"),
            ("2", "05.00", "S2RP_20230629T091808_S20180622T155524"),
        )
    )

    @staticmethod
    def _ids(items: list[_Item]) -> set[str]:
        return {it.id for it in items}

    def test_a_copy_whose_image_cannot_be_read_is_kept_on_its_own(self) -> None:
        """Merged only on proof: it could be a copy of either half or of neither, and keeping a
        duplicate costs a read where merging two images loses ground.
        """
        unread = _Item(
            "S2B_18NUN_20250619_2_L2A", "MGRS-18NUN", "2", **{**self.NUN[0].properties, "s2:datastrip_id": None}
        )
        kept, alternates = select_preferred_duplicates([*self.NUN, unread])
        assert self._ids(kept) == {*self._ids(list(self.NUN)), unread.id}
        assert alternates == {}

    def test_a_copy_whose_image_cannot_be_read_has_no_spare(self) -> None:
        """So its read failure steps nothing, rather than taking another image's spare."""
        old, _, new, _ = self.UUP
        unread = _Item("S2A_33UUP_20191021_4_L2A", "MGRS-33UUP", "4", **{**new.properties, "s2:datatake_id": None})
        kept, alternates = select_preferred_duplicates([old, new, unread])
        assert self._ids(kept) == {new.id, unread.id}
        assert step_down_copies(alternates, kept, implicated=[unread]) is None

    def test_both_granules_of_a_split_pass_are_kept(self) -> None:
        kept, alternates = select_preferred_duplicates(list(self.NUN))
        assert self._ids(kept) == {it.id for it in self.NUN}, "half of the pass was dropped"
        assert alternates == {}

    def test_a_reprocessed_split_keeps_the_newest_copy_of_each_half(self) -> None:
        kept, alternates = select_preferred_duplicates(list(self.UUP))
        assert self._ids(kept) == {self.UUP[2].id, self.UUP[3].id}
        assert self._ids(next(iter(alternates.values()))) == {self.UUP[0].id, self.UUP[1].id}

    def test_a_datastrip_the_reprocessing_omits_is_kept_at_its_older_baseline(self) -> None:
        kept, alternates = select_preferred_duplicates(list(self.QRE))
        assert self._ids(kept) == {self.QRE[1].id, self.QRE[2].id}, "coverage was traded for one baseline"
        assert self._ids(next(iter(alternates.values()))) == {self.QRE[0].id}

    def test_a_granule_published_twice_is_still_one_copy(self) -> None:
        first = self.NUN[0]
        again = _Item("S2B_18NUN_20250619_2_L2A", "MGRS-18NUN", "2", **first.properties)
        kept, _ = select_preferred_duplicates([first, again])
        assert len(kept) == 1

    def test_a_read_failure_steps_down_only_the_half_that_failed(self) -> None:
        kept, alternates = select_preferred_duplicates(list(self.UUP))
        stepped = step_down_copies(alternates, kept, implicated=[self.UUP[2]])
        assert stepped is not None
        assert self._ids(stepped[0]) == {self.UUP[0].id, self.UUP[3].id}, "the healthy half was stepped down"


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
        extra: dict[str, object] = {}
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

    def test_an_all_raw_tile_date_keeps_its_spare(self) -> None:
        """Two raw copies, so the winner and the spare owe the same correction.

        The ladder's exclusion is per copy and asks only whether that copy would REFUSE. Merely
        owing the offset is not a refusal, so the spare stays available. Withholding it cost a
        recoverable date on an all-raw day for no gain.
        """
        remote_scl = {"scl": {"href": f"{_REMOTE}/scl"}}
        winner = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="05.10", host_root=_REMOTE)
        spare = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="05.00", host_root=_REMOTE, extra=remote_scl)
        kept, alternates = select_preferred_duplicates([spare, winner])
        assert [i.id for i in kept] == [winner.id]
        assert [i.id for i in next(iter(alternates.values()))] == [spare.id], (
            "merely owing the offset is not a refusal, so the spare belongs on the ladder"
        )

    def test_a_spare_that_merely_owes_the_offset_is_now_offered(self) -> None:
        """REVERSED, and this is limit 3 of ADR 020 dissolving.

        This asserted that a raw spare lost to a harmonised copy at an OLDER baseline, and was
        withheld from the ladder as well. Both followed from the offset being decided per solar
        day: introducing a copy that owed a correction could make the whole day refuse, over tiles
        this function cannot see, and a refusal is not something the ladder can step down on.

        The offset is decided per image now. Such a swap changes nothing beyond its own tile, so
        the spare is simply usable — and the newer reprocessing wins outright, because owing a
        correction no longer outranks the baseline. See ADR 021.
        """
        harmonised = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="04.00", host_root=_IN_REGION)
        raw_spare = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="05.00", host_root=_REMOTE)
        kept, alternates = select_preferred_duplicates([raw_spare, harmonised])
        assert [i.id for i in kept] == [raw_spare.id], "the newer baseline wins now"
        assert [i.id for i in alternates[("MGRS-33TWM", "2021-09-08")]] == [harmonised.id]

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


class TestTheLadderStepsOneImageAtATime:
    """The fallback ladder must not swap an image's sibling out from under it.

    `select_preferred_duplicates` keeps one copy per ACQUISITION, but `alternates` is keyed
    by tile-date because that is the granularity a read failure is attributed at. Swapping
    on that key replaced every surviving image with the same alternate — turning
    `[a_new, b_new]` into `[a_old, a_old]`, which duplicates one image and drops the
    other. That is worse than the coverage loss the image split fixed, because the
    loader is then handed the same granule twice.
    """

    @staticmethod
    def _four() -> tuple[_Item, _Item, _Item, _Item]:
        """Two images on one tile-date, each with a reprocessing."""
        return (
            _Item("S2A_33XVG_20210908_0_L2A", "MGRS-33XVG", "0", **_image(1)),
            _Item("S2A_33XVG_20210908_1_L2A", "MGRS-33XVG", "1", **_image(1)),
            _Item("S2B_33XVG_20210908_0_L2A", "MGRS-33XVG", "0", **_image(2)),
            _Item("S2B_33XVG_20210908_1_L2A", "MGRS-33XVG", "1", **_image(2)),
        )

    def test_stepping_down_keeps_the_other_image(self) -> None:
        a_old, a_new, b_old, b_new = self._four()
        kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
        assert kept == [a_new, b_new]

        stepped, _ = step_down_copies(alternates, kept, only=None)
        assert stepped == [a_old, b_new], "one image steps; its sibling is untouched"
        assert len({id(i) for i in stepped}) == 2, "no image may be duplicated"

    def test_the_ladder_then_steps_the_second_image(self) -> None:
        a_old, a_new, b_old, b_new = self._four()
        kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
        first, _ = step_down_copies(alternates, kept, only=None)
        second, _ = step_down_copies(alternates, first, only=None)
        assert second == [a_old, b_old]
        assert step_down_copies(alternates, second, only=None) is None, "ladder exhausted"

    def test_a_single_image_steps_exactly_as_before(self) -> None:
        """The behaviour the ladder was built for, unchanged."""
        old, new = _pair()
        kept, alternates = select_preferred_duplicates([old, new])
        stepped, keys = step_down_copies(alternates, kept, only=None)
        assert stepped == [old]
        assert keys == {("MGRS-34WFA", "2021-09-08")}


def _two_images_each_with_a_spare() -> tuple[_Item, _Item, _Item, _Item]:
    """One high-latitude tile-date: two ORBITS, each reprocessed once.

    Successive orbits revisit the same tile the same day above ~60 degrees, so a tile-date
    holds genuinely different imagery as well as reprocessings of it. The image split
    keeps one copy per orbit; the spares are the ladder.
    """
    a_old = _Item("A_old", "MGRS-34WFA", "0", **_image(1))
    a_new = _Item("A_new", "MGRS-34WFA", "9", **_image(1))  # ranks first across the whole tile-date
    b_old = _Item("B_old", "MGRS-34WFA", "0", **_image(2))
    b_new = _Item("B_new", "MGRS-34WFA", "1", **_image(2))
    return a_old, a_new, b_old, b_new


def test_the_ladder_steps_the_image_that_failed_not_the_best_ranked_spare():
    """A tile-date with two images must downgrade only the one that could not be read.

    Ranking is per tile-date, so without attribution the rung taken is whichever spare sorts
    first — which can belong to the image that read perfectly well. That downgrades
    healthy imagery to an older baseline, leaves the unreadable copy still selected, and
    forces a second rung: two spent, one image needlessly older.

    Here A's spare outranks B's, and B is the one that failed.
    """
    a_old, a_new, b_old, b_new = _two_images_each_with_a_spare()
    kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
    assert set(kept) == {a_new, b_new}, "one survivor per image"

    stepped = step_down_copies(dict(alternates), kept, implicated=[b_new])
    assert stepped is not None
    items, _keys = stepped
    assert a_new in items, "the image that read fine must keep its newest copy"
    assert b_old in items, "the image that failed is the one that steps down"
    assert b_new not in items


def test_without_attribution_the_ladder_still_steps_the_best_ranked_spare():
    """The pre-existing behaviour, kept: with nothing attributed there is nothing to prefer.

    Pinned so the new preference cannot quietly become the only path — a caller that cannot
    identify the failing object must still make progress.
    """
    a_old, a_new, b_old, b_new = _two_images_each_with_a_spare()
    kept, alternates = select_preferred_duplicates([a_old, a_new, b_old, b_new])
    stepped = step_down_copies(dict(alternates), kept)
    assert stepped is not None
    items, _keys = stepped
    assert a_old in items, "highest-ranked spare belongs to A, so A steps"


def test_a_failure_in_one_tile_does_not_downgrade_a_neighbouring_tile():
    """An image identity names a pass and datastrip but not a tile, so neighbouring tiles share it.

    `implicated` carries the failed items from every tile in the date, so without filtering to
    the tile-day being stepped, tile B's failure matches tile A's image —
    downgrading A while leaving B's own failed copy selected. That is the defect the attribution
    was added to remove, moved one level sideways.
    """
    a_old = _Item("A_old", "MGRS-34WFA", "0")
    a_new = _Item("A_new", "MGRS-34WFA", "1")
    b_old = _Item("B_old", "MGRS-34WFB", "0")
    b_new = _Item("B_new", "MGRS-34WFB", "1")
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
    indexed twice, both at baseline 00.01 and the SAME image, with the remote
    copy carrying the HIGHER sequence. A sequence-first key therefore preferred the remote
    copy on every duplicated date.
    """

    def test_equal_baseline_prefers_the_in_region_copy(self) -> None:
        """The measured case. Same baseline and image, so the pixels are identical and the
        only difference is whether the read egresses through NAT.
        """
        remote = _copy("S2A_33TWM_20170120_1_L2A", sequence="1", baseline="00.01", host_root=_REMOTE)
        local = _copy("S2A_33TWM_20170120_0_L2A", sequence="0", baseline="00.01", host_root=_IN_REGION)
        kept, alternates = select_preferred_duplicates([remote, local])
        assert [i.id for i in kept] == [local.id], "the in-region copy must survive"
        assert [i.id for i in next(iter(alternates.values()))] == [remote.id], "and the remote one is the fallback"

    def test_locality_never_outranks_a_newer_baseline(self) -> None:
        """The complement, and the reason locality sits BELOW the baseline term: a cheaper
        read must never be bought with a worse pixel.

        Both copies are below the correction threshold, so both producer terms are inert — the
        owes term, which sits below the baseline, and the refusal term, which sits above it. That
        is what leaves locality alone with the baseline, which is the pair this test is about.
        """
        local_old = _copy("S2A_33TWM_20170120_0_L2A", sequence="0", baseline="02.06", host_root=_IN_REGION)
        remote_new = _copy("S2A_33TWM_20170120_1_L2A", sequence="1", baseline="03.01", host_root=_REMOTE)
        kept, _ = select_preferred_duplicates([local_old, remote_new])
        assert [i.id for i in kept] == [remote_new.id], "a newer baseline must win despite egress"

    def test_it_still_decides_between_copies_at_one_baseline(self) -> None:
        """Demoted, not deleted — the two pixel-quality reasons above still hold.

        Among copies the baseline cannot separate, the one owing no correction is preferred: its
        floor was applied before resampling, and it cannot be wrong by the offset.
        """
        harmonised = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="05.00", host_root=_IN_REGION)
        raw = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="05.00", host_root=_REMOTE)
        kept, _ = select_preferred_duplicates([raw, harmonised])
        assert [i.id for i in kept] == [harmonised.id]

    @pytest.mark.parametrize("host_root", [_IN_REGION, _REMOTE], ids=["in-region", "all-remote"])
    def test_sequence_still_decides_when_baseline_and_locality_tie(self, host_root: str) -> None:
        """With nothing to separate them but sequence, sequence decides — including the
        all-remote date, which is most backfilled dates.
        """
        low = _copy("S2A_33TWM_20170120_0_L2A", sequence="0", baseline="05.00", host_root=host_root)
        high = _copy("S2A_33TWM_20170120_1_L2A", sequence="1", baseline="05.00", host_root=host_root)
        kept, _ = select_preferred_duplicates([low, high])
        assert [i.id for i in kept] == [high.id]


class TestLocalityPredicate:
    """`item_is_in_preferred_location` — ALL assets, not any."""

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

    def test_a_harmonised_copy_is_still_not_penalised_for_its_baseline(self) -> None:
        """An unreadable baseline costs a harmonised copy nothing on the READABILITY term.

        What it does cost is the baseline VALUE term, which is why the newer raw copy wins here
        now. That is not the term under test misfiring: a raw copy at 05.00 declaring its baseline
        is fully decidable and gets corrected per image, so preferring it is preferring a newer
        reprocessing over an older one — exactly the rule. The fear this test used to encode, that
        the raw copy's date would be refused outright, no longer applies to a copy like that.

        The readability term itself is pinned below, where the baselines are equal.
        """
        harmonised_unknown = _Item("S2A_33TWM_20220107_0_L2A", "MGRS-33TWM", "0")
        _with_assets(harmonised_unknown, _bands_at(_IN_REGION))
        raw_best = _copy("S2A_33TWM_20220107_1_L2A", sequence="1", baseline="05.00", host_root=_REMOTE)
        kept, alternates = select_preferred_duplicates([harmonised_unknown, raw_best])
        assert [i.id for i in kept] == [raw_best.id]
        # And the harmonised copy stays available: it is owed nothing, so it cannot refuse.
        assert [i.id for i in alternates[("MGRS-33TWM", "2021-09-08")]] == [harmonised_unknown.id]

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

        # The spare has to be one the ladder genuinely cannot use, and "merely owes the offset" is
        # no longer such a case — per-image correction made those usable again. What still refuses
        # is a copy whose own evidence is missing: raw bands, no readable baseline.
        winner = _copy("S2A_33TWM_20220107_0_L2A", sequence="0", baseline="04.00", host_root=_IN_REGION)
        raw_spare = _with_assets(_Item("S2A_33TWM_20220107_1_L2A", "MGRS-33TWM", "1"), _bands_at(_REMOTE))
        kept, alternates = select_preferred_duplicates([winner, raw_spare])
        assert [i.id for i in kept] == [winner.id]
        assert alternates == {}, "the fixture must produce an empty ladder, or this proves nothing"

        log = logging.getLogger("dup-audit-empty-ladder")
        with caplog.at_level(logging.INFO, logger="dup-audit-empty-ladder"):
            log_duplicate_selection(log, "roi-x", alternates, kept=kept, items=[winner, raw_spare])
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "1 tile-date(s) had more than one copy" in msg
        assert "winners by source: 1 in-region, 0 remote" in msg, msg

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

    def test_no_duplicates_logs_nothing(self, caplog) -> None:
        assert self._emit(caplog, [], {}) == ""


def _copy_at(
    ident: str, *, image: dict[str, str], sequence: str, baseline: str | None, host_root: str = _IN_REGION
) -> _Item:
    """A copy of a given IMAGE, for cross-image ordering tests."""
    extra: dict[str, object] = {**image}
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


class TestNonFiniteBaselinesAreUnknown:
    """`float()` accepts "NaN" and "Infinity". Neither is a baseline. Review on PR #107."""

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


class TestUnknownBaselineStaysScopedToItsImage:
    """One image's unreadable baseline must not reorder another's. Review on PR #107.

    `_rank_copies` reverts to sequence ordering when ANY baseline in the list it is given is
    unknown — correct within an image, where it stops locality promoting a copy over a
    possibly-newer one. The alternates list was built by ranking a whole tile-date in one
    call, so a single malformed item leaked that suspension across image boundaries.
    """

    _A = _image(1)
    _B = _image(2)

    def test_a_neighbours_unknown_baseline_keeps_this_ladder_baseline_first(self) -> None:
        """The unknown copy must be a REJECTED one — that is what puts it in the shared list.

        Image A holds an unreadable baseline on its LOSING copy, so `rejected` spans
        both images and contains an unknown. Ranking that whole list in one call
        suspends the baseline term for B as well.
        """
        items = [
            # A: the unknown loses on sequence, so it lands in the shared rejected list.
            _copy_at("a-keep", image=self._A, sequence="5", baseline="04.00"),
            _copy_at("a-unknown", image=self._A, sequence="0", baseline=None),
            # B: readable throughout, so its ladder must stay in descending baseline order.
            _copy_at("b-best", image=self._B, sequence="0", baseline="05.00"),
            _copy_at("b-mid", image=self._B, sequence="1", baseline="04.00"),
            _copy_at("b-newer-seq", image=self._B, sequence="9", baseline="03.00"),
        ]
        kept, alternates = select_preferred_duplicates(items)

        assert sorted(it.id for it in kept) == ["a-keep", "b-best"]

        ladder = [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]]
        assert "a-unknown" in ladder, "the fixture must put the unknown into the shared list"
        # Pre-fix the shared call fell back to sequence ordering for the WHOLE tile-date, which
        # put b-newer-seq (sequence 9, baseline 03.00) ahead of b-mid (sequence 1, baseline
        # 04.00) — a read failure would then have handed out the older reprocessing.
        assert [i for i in ladder if i.startswith("b-")] == ["b-mid", "b-newer-seq"]


class TestTheLadderKeepsGlobalPriorityPastItsHead:
    """`s2_roi`'s unattributed recovery consumes `copies[0]` on each retry, so EVERY position in
    the ladder is a choice, not only the first. Raised on PR #107 against a per-image
    ranking that ordered the ladder heads and then flattened whole ladders behind them.
    """

    _A = _image(1)
    _B = _image(2)

    def test_every_position_is_in_descending_baseline_order(self) -> None:
        """The general property, over three images interleaved at every rung."""
        items = [
            _copy_at(f"{tag}-win", image=acq, sequence="9", baseline="05.10")
            for tag, acq in (("a", self._A), ("b", self._B))
        ] + [
            _copy_at("a-hi", image=self._A, sequence="3", baseline="05.00"),
            _copy_at("a-lo", image=self._A, sequence="1", baseline="02.00"),
            _copy_at("b-mid", image=self._B, sequence="3", baseline="04.00"),
            _copy_at("b-low", image=self._B, sequence="1", baseline="03.00"),
        ]
        _, alternates = select_preferred_duplicates(items)
        ladder = alternates[("MGRS-33TWM", "2021-09-08")]
        baselines = [item_processing_baseline(it) for it in ladder]
        assert baselines == sorted(baselines, reverse=True), f"ladder out of order: {baselines}"
        assert [it.id for it in ladder] == ["a-hi", "b-mid", "b-low", "a-lo"]

    def test_an_unreadable_baseline_sorts_last_rather_than_suspending_the_order(self) -> None:
        """The alternative — suspending the comparison — is what leaked one image's
        malformed metadata into another's ladder. A copy whose baseline cannot be read is also
        the one whose correction will silently be skipped, so it belongs at the end.
        """
        items = [
            _copy_at("a-win", image=self._A, sequence="9", baseline="05.10"),
            _copy_at("a-unknown", image=self._A, sequence="5", baseline=None),
            _copy_at("a-04", image=self._A, sequence="1", baseline="04.00"),
            _copy_at("b-win", image=self._B, sequence="9", baseline="05.10"),
            _copy_at("b-03", image=self._B, sequence="1", baseline="03.00"),
        ]
        _, alternates = select_preferred_duplicates(items)
        ladder = [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]]
        assert ladder == ["a-04", "b-03", "a-unknown"]


class TestLocalityDoesNotDecideBetweenUnknownBaselines:
    """In the ladder key, locality was ranked ahead of sequence for copies that declare no
    readable baseline — so a local sequence-1 spare came out before a remote sequence-2 one,
    buying cheaper egress with an older reprocessing. Raised on PR #107.
    """

    _A = _image(1)

    def test_sequence_decides_between_two_unknown_baseline_spares(self) -> None:
        items = [
            _copy_at("winner", image=self._A, sequence="3", baseline=None),
            _copy_at("remote-seq2", image=self._A, sequence="2", baseline=None, host_root=_REMOTE),
            _copy_at("local-seq1", image=self._A, sequence="1", baseline=None, host_root=_IN_REGION),
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
        spare is kept off the ladder entirely (`refuses_its_date`), which would empty the
        ladder and prove nothing about locality.
        """
        items = [
            _copy_at("winner", image=self._A, sequence="3", baseline="03.01"),
            _copy_at("remote", image=self._A, sequence="1", baseline="02.06", host_root=_REMOTE),
            _copy_at("local", image=self._A, sequence="1", baseline="02.06", host_root=_IN_REGION),
        ]
        _, alternates = select_preferred_duplicates(items)
        ladder = [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]]
        assert ladder == ["local", "remote"]


class TestAMalformedHrefDoesNotAbortTheSelection:
    """`urlparse` raises on some malformed authorities, and locality is evaluated for every copy
    including singletons — so one bad catalogue href aborted the whole selection. PR #107.
    """

    def test_an_unparseable_href_reads_as_remote(self) -> None:
        assert asset_bucket("https://[oops/B02.tif") is None

    def test_a_singleton_group_with_a_malformed_href_still_returns(self) -> None:
        item = _with_assets(_Item("only", "MGRS-33TWM", "0"), {"blue": {"href": "https://[oops/B02.tif"}})
        kept, alternates = select_preferred_duplicates([item])
        assert [it.id for it in kept] == ["only"]
        assert alternates == {}


class TestThePreferenceKeyHoldsItsInvariants:
    """One key orders copies within an image and across images, so the properties it
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


class TestCompletenessOutranksTheSequenceWhenABaselineIsUnreadable:
    """A copy missing an asset must lose to a complete one even when a sibling cannot be dated.

    Read-set completeness is term 1 of `_preference_key` while an unreadable baseline is only
    term 4, so one undatable copy cannot put the group on the sequence alone. A higher-sequence
    incomplete winner guarantees the eager `No such band/alias` failure, which names no object, so
    recovery steps every duplicated tile-date of the day rather than this one. Raised on PR #107.
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
        # Present to put a copy nobody can date into the group alongside the two being compared.
        unreadable = self._copy("unreadable", sequence="0", baseline=None, complete=True)
        incomplete_newer = self._copy("incomplete-newer", sequence="9", baseline="05.00", complete=False)
        complete_older = self._copy("complete-older", sequence="1", baseline="04.00", complete=True)
        kept, _ = select_preferred_duplicates([unreadable, incomplete_newer, complete_older])
        assert [it.id for it in kept] == ["complete-older"]


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


class TestAMixedProducerCopyIsRankedOnItsBaselineLikeAnyOther:
    """A copy whose reflectance bands span a harmonised and a raw producer is ordinary.

    Each of its sources is decided on its own bucket, so it is corrected band by band rather than
    refused, and the refusal term is therefore inert for it at every baseline. What is left to
    decide these comparisons is the baseline — the module's rule holding: nothing below "will this
    work at all" may buy its preference with an older reprocessing. Raised on PR #107.
    """

    @staticmethod
    def _copy_at_baseline(ident: str, baseline: str, *, mixed: bool, sequence: str = "0") -> _Item:
        assets = _bands_at(_IN_REGION)
        if mixed:
            assets["red"] = {"href": f"{_REMOTE}/B04.jp2"}
        return _with_assets(_Item(ident, "MGRS-33TWM", sequence, **{"s2:processing_baseline": baseline}), assets)

    @pytest.mark.parametrize(("new", "old"), [("05.00", "04.00"), ("03.00", "02.00")], ids=["above", "below"])
    def test_a_mixed_producer_copy_is_no_longer_a_problem_at_all(self, new: str, old: str) -> None:
        """A copy whose bands span two KNOWN producers is corrected band by band, so it is
        ordinary on either side of the threshold, and the newer baseline wins.
        """
        mixed_new = self._copy_at_baseline("mixed-new", new, mixed=True)
        clean_old = self._copy_at_baseline("clean-old", old, mixed=False)
        kept, alternates = select_preferred_duplicates([mixed_new, clean_old])
        assert [it.id for it in kept] == ["mixed-new"]
        assert [it.id for it in next(iter(alternates.values()))] == ["clean-old"]

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
    the correction path raise means the exception escapes instead of the next usable
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

    @pytest.mark.parametrize("baseline", ["05.00", "03.00"], ids=["above", "below"])
    def test_a_mixed_producer_spare_is_offered_at_every_baseline(self, baseline: str) -> None:
        """Per-source correction handles a mixed-producer spare, so it is a usable fallback on
        either side of the threshold.
        """
        winner = self._copy("win", baseline=baseline, sequence="9")
        spare = self._copy("mixed", baseline=baseline, sequence="1", mixed=True)
        _, alternates = select_preferred_duplicates([winner, spare])
        assert [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]] == ["mixed"]

    def test_the_usable_spares_survive_alongside_an_excluded_one(self) -> None:
        winner = self._copy("win", baseline="05.10", sequence="9")
        usable = self._copy("usable", baseline="05.00", sequence="2")
        unusable = self._copy("no-baseline", baseline=None, sequence="1", raw=True)
        _, alternates = select_preferred_duplicates([winner, usable, unusable])
        assert [it.id for it in alternates[("MGRS-33TWM", "2021-09-08")]] == ["usable"]


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
                _Item(f"pass-{tag}", "MGRS-33TWA", "0", **{"s2:processing_baseline": "05.00", **image}),
                _bands_at(_IN_REGION),
            )
            for tag, image in (("a", _image(1)), ("b", _image(2)))
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


class TestTheLadderStepsNothingWhenTheFailedImageHasNoSpare:
    """Attribution named the failing objects and no copy belongs to their image.

    Falling back to the best overall alternate there swapped a HEALTHY image down to an older
    copy while leaving the known-bad one selected, then rebuilt and re-read the whole date only to
    fail identically — once per unrelated spare before recording the loss. Raised on PR #107.
    """

    _A = _image(1)
    _B = _image(2)

    def _at(self, ident: str, image: dict[str, str], sequence: str, baseline: str = "05.00") -> _Item:
        return _with_assets(
            _Item(ident, "MGRS-33TWM", sequence, **{"s2:processing_baseline": baseline, **image}),
            _bands_at(_IN_REGION),
        )

    def test_nothing_is_stepped_when_only_a_healthy_image_has_a_spare(self) -> None:
        bad = self._at("bad-a", self._A, "1")  # the one that failed; no spare of its own
        healthy = self._at("healthy-b", self._B, "1")  # a different image...
        spare_b = self._at("spare-b", self._B, "0")  # ...which DOES have a spare
        kept, alternates = select_preferred_duplicates([bad, healthy, spare_b])
        assert sorted(i.id for i in kept) == ["bad-a", "healthy-b"]

        stepped = step_down_copies(alternates, kept, implicated=[bad])
        assert stepped is None, "a healthy image was downgraded for a failure it did not have"


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

    @pytest.mark.parametrize(("raw", "expected"), [("12", 12), (3, 3)])
    def test_a_real_sequence_parses(self, raw: object, expected: int) -> None:
        assert item_sequence(self._item(raw)) == expected

    @pytest.mark.parametrize(
        ("label", "raw"),
        [
            ("truncating-float", 1.9),
            ("infinity", float("inf")),
            ("negative", "-1"),
            ("exponent", "1e3"),
            ("arabic-indic", chr(0x664)),
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

    _A = _image(1)

    def _at(self, ident: str, tile: str, image: dict[str, str], sequence: str) -> _Item:
        return _with_assets(
            _Item(ident, tile, sequence, **{"s2:processing_baseline": "05.00", **image}),
            _bands_at(_IN_REGION),
        )

    def test_only_the_tile_date_with_a_usable_spare_is_reported(self) -> None:
        # Tile A: the failure has a spare of its own image, so it steps.
        a_bad = self._at("a-bad", "MGRS-33TWA", self._A, "1")
        a_spare = self._at("a-spare", "MGRS-33TWA", self._A, "0")
        # Tile B: also failed, but its only spare belongs to a DIFFERENT image.
        b_bad = self._at("b-bad", "MGRS-33TWB", self._A, "1")
        b_other = self._at("b-other", "MGRS-33TWB", _image(2), "1")
        b_other_spare = self._at("b-other-spare", "MGRS-33TWB", _image(2), "0")

        kept, alternates = select_preferred_duplicates([a_bad, a_spare, b_bad, b_other, b_other_spare])
        stepped = step_down_copies(alternates, kept, implicated=[a_bad, b_bad])
        assert stepped is not None
        _swapped, keys = stepped
        assert keys == {("MGRS-33TWA", "2021-09-08")}, f"reported a tile-date it never stepped: {keys}"

    def test_a_spare_is_not_consumed_from_a_tile_date_that_did_not_step(self) -> None:
        """Removing it from the ladder before knowing it would be used loses a copy for nothing."""
        b_bad = self._at("b-bad", "MGRS-33TWB", self._A, "1")
        b_other = self._at("b-other", "MGRS-33TWB", _image(2), "1")
        b_other_spare = self._at("b-other-spare", "MGRS-33TWB", _image(2), "0")
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
        # Native keys, as Planetary Computer serves them — none of the configured names resolve.
        return _with_assets(item, {k: {"href": f"https://x.blob.core.windows.net/{k}"} for k in ("B02", "B03")})

    def test_the_spare_stays_on_the_fallback_ladder(self) -> None:
        newer = self._native("new-05.10", "05.10", "1")
        older = self._native("old-03.00", "03.00", "0")
        kept, alternates = select_preferred_duplicates([newer, older], ())
        assert [i.id for i in kept] == ["new-05.10"]
        assert [i.id for i in alternates[("MGRS-33TWM", "2021-09-08")]] == ["old-03.00"]

    def test_the_term_still_fires_where_the_names_are_the_keys(self) -> None:
        """The complement, or gating it would have disabled the guard everywhere.

        Posed on an UNCLASSIFIED bucket, which is the case that still refuses. A copy whose bands
        span two KNOWN producers no longer does — each band is decided on its own bucket — so this
        used to be posed on one of those and would now pass for the wrong reason.
        """
        unlisted = _Item("unlisted", "MGRS-33TWM", "0", **{"s2:processing_baseline": "05.00"})
        _with_assets(unlisted, _bands_at("s3://nobody-has-classified-this/33/T/WM"))
        assert refuses_its_date(unlisted, READ_ASSET_KEYS) is True


class TestRecognisingAFailureThatArrivedWithNoEvidence:
    """The shape Dask substitutes when it cannot serialise a read failure.

    ``ingest/loader_failures.py`` stops it being produced, and the tests that prove that run
    the exception through Dask's own error path. This is the other half: recognising the shape
    if it turns up anyway, because that rescue is installed per process and best effort. Every
    predicate here reads the cause chain, so this shape does not get a wrong verdict — it gets
    no verdict, and the point of naming it is to make that difference visible in a log.
    """

    @staticmethod
    def _flattened(outer: BaseException) -> Exception:
        """The substitution, built the way Dask builds it: ``Exception(repr(outer))``.

        From the real class rather than a copied string, so the message cannot drift from what
        rasterio actually produces. A hand-made ``RuntimeError`` stood here once and is why the
        defect shipped — it had the same shape without having the same origin.
        """
        return Exception(repr(outer))

    def test_the_production_shape_is_recognised(self) -> None:
        flattened = self._flattened(RasterioIOError("Read failed. See previous exception for details."))
        assert flattened.__cause__ is None
        assert cause_was_flattened(flattened) is True

    def test_the_bare_warp_wrapper_is_recognised(self) -> None:
        """The radar form: the outer exception names no codec and no status, so its repr is all
        that arrives and it says only that a read failed.
        """
        assert cause_was_flattened(self._flattened(WarpOperationError("Chunk and warp failed"))) is True

    def test_a_chain_that_arrived_intact_is_not_flattened(self) -> None:
        """The negative that matters: once the cause survives, this must stay silent."""
        wrapper = Exception("RasterioIOError: Read failed. See previous exception for details.")
        wrapper.__cause__ = Exception("CPLE_AppDefinedError: ZIPDecode:Decoding error at scanline 0")
        assert cause_was_flattened(wrapper) is False

    def test_a_context_counts_as_a_chain(self) -> None:
        """``raise`` inside an ``except`` attaches a CONTEXT and no cause, and the classifier
        reads either. A failure carrying one has evidence, so it is not the flattened shape.
        """
        try:
            try:
                raise ValueError("ZIPDecode:Decoding error at scanline 0")
            except ValueError:
                raise Exception("RasterioIOError('Read failed.')")  # noqa: B904
        except Exception as exc:
            assert exc.__context__ is not None
            assert cause_was_flattened(exc) is False

    def test_a_typed_exception_is_not_the_flattened_shape(self) -> None:
        """The substitution is always a bare ``Exception``. A subclass arrived as itself, which
        means its class survived and so did anything under it.
        """
        assert cause_was_flattened(RasterioIOError("Read failed. See previous exception.")) is False

    def test_an_unrelated_failure_is_not_claimed(self) -> None:
        """A bug of ours is not a read whose evidence was lost, and must not be reported as one."""
        assert cause_was_flattened(Exception("KeyError('northing')")) is False
        assert cause_was_flattened(Exception("some unrelated bug")) is False
