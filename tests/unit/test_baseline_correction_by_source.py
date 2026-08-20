"""Whether the BOA offset is corrected is a per-ITEM question, not a per-collection one.

ESA changed processing baseline 04.00 in January 2022: from that baseline on, surface
reflectance carries a +1000 offset that has to be subtracted. Element 84 harmonises its own
COGs — it subtracts it for you — so the Earth Search collection config used to leave
`baseline_threshold` unset with a comment saying no correction was needed.

That stopped being true when the same collection began indexing items whose assets point at
ESA's originals, which carry the offset. Reading the collection alone exempts, or corrects,
both kinds together, and one of those is always wrong. Nothing raises either way.

The threshold is therefore set, and the exemption is decided per item from where its assets
live. These tests pin both directions, because each is a silent corruption in its own
direction: a skipped correction leaves plausible pixels 1000 too high, a doubled one shifts
every value by 1000.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from tessera_embeddings.config.providers import PROVIDERS
from tessera_embeddings.config.satellites import S2_BASELINE_OFFSET, S2_BASELINE_THRESHOLD
from tessera_embeddings.ingest.asset_locations import (
    HARMONISED_ASSET_BUCKETS,
    READ_ASSET_KEYS,
    REFLECTANCE_ASSET_KEYS,
    Harmonisation,
    item_harmonisation,
)
from tessera_embeddings.ingest.stac import (
    HeterogeneousProducerError,
    _extract_baseline,
    dates_exempt_from_correction,
    extract_baselines,
)

_HARMONISED = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/33/T/WM/2022/1"
_RAW_ESA = "s3://sentinel-s2-l2a/tiles/33/T/WM/2022/1/7/0"


class _Item:
    """A STAC-like item carrying only what the baseline decision reads."""

    def __init__(self, host_root: str | None, baseline: str | None, *, extra: dict | None = None) -> None:
        self.id = "S2A_33TWM_20220107_0_L2A"
        self.datetime = datetime(2022, 1, 7, 12, 0, 0, tzinfo=UTC)
        self.properties: dict[str, object] = {}
        if baseline is not None:
            self.properties["s2:processing_baseline"] = baseline
        self.assets: dict[str, dict[str, str]] = {}
        if host_root is not None:
            self.assets = {key: {"href": f"{host_root}/{key}"} for key in READ_ASSET_KEYS}
        self.assets.update(extra or {})


class TestTheCollectionNowAsksForCorrection:
    """The threshold has to be SET, or the per-item decision is never consulted."""

    def test_earth_search_requires_baseline_correction(self) -> None:
        """`requires_baseline_correction` gates the whole correction block in `stac.py`. While
        it was False, no Earth Search item was ever corrected — whatever its own baseline and
        whatever bucket served it.
        """
        config = PROVIDERS["earth-search"].collections["sentinel-2-l2a"]
        assert config.requires_baseline_correction is True
        assert config.baseline_threshold == S2_BASELINE_THRESHOLD
        assert config.baseline_offset == S2_BASELINE_OFFSET


class TestPerItemExemption:
    """`_extract_baseline` reports 0 — no correction due — for already-harmonised pixels."""

    def test_a_raw_esa_item_over_the_threshold_is_corrected(self) -> None:
        """THE POINT OF THIS CHANGE. Raw ESA pixels at baseline 05.00 carry the offset, and
        before this they were silently left 1000 too high.
        """
        item = _Item(_RAW_ESA, "05.00")
        assert _extract_baseline(item) == 500
        assert dates_exempt_from_correction([item]) == set(), "raw pixels are owed the correction"

    def test_a_harmonised_item_over_the_threshold_is_left_alone(self) -> None:
        """The safety direction, and the bug this change could introduce. Element 84's COGs
        already had the offset removed, so correcting them again would shift every value by
        1000 — a regression on the majority of the archive.
        """
        item = _Item(_HARMONISED, "05.09")
        assert _extract_baseline(item) == 509, "the parser reports what the item declares"
        assert extract_baselines([item]) == {"2022-01-07": 509}, "provenance keeps the real baseline"
        assert dates_exempt_from_correction([item]) == {"2022-01-07"}, "but no correction is owed"

    def test_a_raw_esa_item_below_the_threshold_reports_its_baseline(self) -> None:
        """Reporting the real baseline is what lets the threshold decide. 301 is under 400, so
        no correction is applied — but by the threshold, not by an exemption.
        """
        assert _extract_baseline(_Item(_RAW_ESA, "03.01")) == 301

    @pytest.mark.parametrize("baseline", [None, "", "not-a-number"])
    def test_an_unreadable_baseline_reports_zero(self, baseline: str | None) -> None:
        """Pre-existing behaviour, unchanged."""
        assert _extract_baseline(_Item(_RAW_ESA, baseline)) == 0

    def test_two_dates_from_different_producers_get_different_answers(self) -> None:
        """Derived over the items a preparation actually loads, so one call must separate them."""
        raw = _Item(_RAW_ESA, "05.00")
        harmonised = _Item(_HARMONISED, "05.00")
        harmonised.datetime = datetime(2022, 1, 8, 12, 0, 0, tzinfo=UTC)
        items = [raw, harmonised]
        assert extract_baselines(items) == {"2022-01-07": 500, "2022-01-08": 500}, "provenance is untouched"
        assert dates_exempt_from_correction(items) == {"2022-01-08"}


class TestProvenanceIsNotOverwritten:
    """The reported baseline reaches the store's `baselines_applied` and must survive."""

    def test_a_harmonised_item_still_reports_its_real_baseline(self) -> None:
        """Encoding the exemption as a baseline of 0 made the store misreport its own vintage —
        worse than the error it avoided, because it cannot be recovered after the fact.
        """
        assert extract_baselines([_Item(_HARMONISED, "05.10")]) == {"2022-01-07": 510}


class TestMixedProducerDates:
    """Mixed-producer days are REAL, so the question is what is owed — not who disagrees."""

    def test_a_mixed_date_owing_nothing_is_exempt_not_refused(self) -> None:
        """THE REAL CASE, and the one an earlier version of this got wrong by refusing.

        A full census of four tile-years found 7 mixed days in 522. After duplicate selection
        the survivors pair a harmonised COG with a raw item at an OLD baseline — measured on
        33TWM 2017-12-19, a COG at 05.00 beside a raw item at 02.06, kept apart because their
        instants are 209 s apart and so count as distinct acquisitions. Nothing there is owed a
        correction, so exempting is right and refusing would lose a real date.
        """
        harmonised = _Item(_HARMONISED, "05.00")
        raw_old = _Item(_RAW_ESA, "02.06")
        assert dates_exempt_from_correction([harmonised, raw_old]) == {"2022-01-07"}

    def test_a_mixed_date_that_genuinely_owes_a_correction_raises(self) -> None:
        """The unobserved case that has no right answer: a raw item at or above the threshold
        fused with a harmonised one. Exempting leaves the raw tiles 1000 high, correcting drops
        1000 from the harmonised ones, and the correction is date-wide.
        """
        with pytest.raises(HeterogeneousProducerError, match="fuses a raw item"):
            dates_exempt_from_correction([_Item(_HARMONISED, "05.00"), _Item(_RAW_ESA, "05.00")])

    def test_an_all_raw_date_over_the_threshold_is_corrected(self) -> None:
        """No ambiguity when every item is raw — correct the whole date."""
        assert dates_exempt_from_correction([_Item(_RAW_ESA, "05.00"), _Item(_RAW_ESA, "05.00")]) == set()

    def test_a_mixed_band_item_below_the_threshold_is_not_refused(self) -> None:
        """Straddling bands only matter if something is actually owed. Below the threshold there
        is nothing to get wrong, so refusing would be gratuitous.
        """
        item = _Item(_HARMONISED, "02.06")
        item.assets["red"] = {"href": f"{_RAW_ESA}/B04.jp2"}
        assert dates_exempt_from_correction([item]) == {"2022-01-07"}

    def test_a_mixed_band_item_over_the_threshold_raises(self) -> None:
        """Same reasoning one level down: no date-wide answer is correct for a single item whose
        own bands come from both producers.
        """
        item = _Item(_HARMONISED, "05.00")
        item.assets["red"] = {"href": f"{_RAW_ESA}/B04.jp2"}
        with pytest.raises(HeterogeneousProducerError, match="read bands span"):
            dates_exempt_from_correction([item])

    def test_a_homogeneous_multi_tile_date_is_fine(self) -> None:
        """The common case must NOT refuse: many tiles per date is the norm, and they agree."""
        tiles = [_Item(_HARMONISED, "05.00") for _ in range(4)]
        assert dates_exempt_from_correction(tiles) == {"2022-01-07"}


class TestHarmonisationPredicate:
    """`item_harmonisation` — three states, because MIXED needs its own response."""

    def test_a_harmonised_bucket_is_recognised(self) -> None:
        assert item_harmonisation(_Item(_HARMONISED, "05.00")) is Harmonisation.HARMONISED

    def test_raw_esa_is_not(self) -> None:
        assert item_harmonisation(_Item(_RAW_ESA, "05.00")) is Harmonisation.RAW

    def test_extra_assets_elsewhere_do_not_change_the_answer(self) -> None:
        """Only the bands we read matter; a real Element 84 item carries the original JP2s
        alongside its COGs and they are never fetched.
        """
        item = _Item(_HARMONISED, "05.00", extra={"aot": {"href": f"{_RAW_ESA}/AOT.jp2"}})
        assert item_harmonisation(item) is Harmonisation.HARMONISED

    def test_bands_straddling_two_producers_are_not_harmonised(self) -> None:
        """The correction is applied per DATE to every band at once, so a partly-harmonised
        item has no single right answer. Under-correcting one band beats over-correcting the
        rest.
        """
        item = _Item(_HARMONISED, "05.00")
        item.assets["red"] = {"href": f"{_RAW_ESA}/B04.jp2"}
        assert item_harmonisation(item) is Harmonisation.MIXED, "MIXED, not silently one or the other"

    def test_an_item_with_no_read_bands_is_not_harmonised(self) -> None:
        """Absence of evidence must not buy an exemption from a correction."""
        assert item_harmonisation(_Item(None, "05.00")) is Harmonisation.RAW

    def test_an_unrecognised_bucket_is_not_harmonised(self) -> None:
        """A future mirror nobody has told us about gets corrected rather than exempted."""
        assert item_harmonisation(_Item("s3://some-new-mirror/tiles/33/T/WM", "05.00")) is Harmonisation.RAW

    def test_the_raw_archive_is_not_in_the_harmonised_set(self) -> None:
        """Guards the constant: adding `sentinel-s2-l2a` here would silently reinstate the bug."""
        assert "sentinel-s2-l2a" not in HARMONISED_ASSET_BUCKETS


class TestSclDoesNotDecideHarmonisation:
    """`scl` is read, so it counts for egress — and never corrected, so it cannot make the
    reflectance correction ambiguous.
    """

    def test_scl_is_read_but_not_reflectance(self) -> None:
        """Guards the split itself. Collapsing these back into one tuple reinstates the bug."""
        assert "scl" in READ_ASSET_KEYS, "scl IS fetched, so it counts toward locality"
        assert "scl" not in REFLECTANCE_ASSET_KEYS, "scl is categorical and never offset-corrected"

    def test_raw_scl_beside_harmonised_reflectance_is_still_harmonised(self) -> None:
        """THE REPORTED CASE. Subtracting 1000 from a class label is meaningless, so the
        producer of `scl` cannot make the reflectance decision ambiguous — and while it could,
        a single raw `scl` classified the item MIXED and refused the whole date.
        """
        item = _Item(_HARMONISED, "05.00")
        item.assets["scl"] = {"href": f"{_RAW_ESA}/scl"}
        assert item_harmonisation(item) is Harmonisation.HARMONISED
        assert dates_exempt_from_correction([item]) == {"2022-01-07"}

    def test_a_raw_reflectance_band_still_makes_it_mixed(self) -> None:
        """The complement: excluding scl must not weaken the reflectance check itself."""
        item = _Item(_HARMONISED, "05.00")
        item.assets["red"] = {"href": f"{_RAW_ESA}/B04.jp2"}
        assert item_harmonisation(item) is Harmonisation.MIXED


class TestRawDatesStraddlingTheThreshold:
    """One date carries one baseline, so raw items on opposite sides cannot both be served."""

    def test_raw_items_across_the_threshold_raise(self) -> None:
        """`extract_baselines` is last-wins by construction, so the date-wide baseline serves one
        side or the other: correcting shifts the pre-threshold pixels down 1000, not correcting
        leaves the post-threshold ones 1000 high.
        """
        over, under = _Item(_RAW_ESA, "05.00"), _Item(_RAW_ESA, "03.01")
        with pytest.raises(HeterogeneousProducerError, match="straddle the correction threshold"):
            dates_exempt_from_correction([over, under])

    def test_raw_items_all_under_the_threshold_are_fine(self) -> None:
        """The common case for the backfill, which is entirely pre-04.00: nothing owed, no
        conflict, no refusal.
        """
        assert dates_exempt_from_correction([_Item(_RAW_ESA, "02.06"), _Item(_RAW_ESA, "03.01")]) == {"2022-01-07"}

    def test_raw_items_all_over_the_threshold_are_corrected(self) -> None:
        assert dates_exempt_from_correction([_Item(_RAW_ESA, "05.00"), _Item(_RAW_ESA, "05.09")]) == set()


class TestTheCorrectionPathAnnouncesItself:
    """Tier 5: the assumption this change rests on, made observable at runtime.

    Every raw item measured on the live catalogue reports a pre-04.00 baseline, so the
    correction has never actually run on real data. Sampling cannot prove the combination never
    appears — a 100-item page of a 146-item year said it did not — so the honest close is a
    signal on the day it does, rather than an inference from a sample.
    """

    def test_a_routinely_unharmonised_producer_does_not_warn(self, caplog) -> None:
        """FOUND BY THE REGRESSION SWEEP. Planetary Computer serves unharmonised data on Azure,
        so every one of its dates is corrected as a matter of course. Warning on each would fire
        on every date of an MPC ingest AND would assert something untrue — that the combination
        had not been seen before. Narrowed to the ESA archive, which is the route a
        harmonised-COG catalogue was not expected to take.
        """
        azure = _Item("https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/33/T/TG/x", "05.10")
        with caplog.at_level(logging.WARNING, logger="tessera_embeddings.ingest.stac"):
            assert dates_exempt_from_correction([azure]) == set(), "it must still be corrected"
        assert not [r for r in caplog.records if "correction ACTIVE" in r.message]

    def test_it_warns_when_a_raw_item_is_actually_corrected(self, caplog) -> None:
        """Not an error: correcting a raw item over the threshold is exactly right. The warning
        says the path has gone live, so its output gets verified instead of assumed.
        """
        with caplog.at_level(logging.WARNING, logger="tessera_embeddings.ingest.stac"):
            assert dates_exempt_from_correction([_Item(_RAW_ESA, "05.00")]) == set()
        assert any("correction ACTIVE on ESA-archive data" in r.message for r in caplog.records)
        assert any("500" in str(r.args) for r in caplog.records), "the baseline must be named"

    def test_it_stays_quiet_below_the_threshold(self, caplog) -> None:
        """The entire backfill is pre-04.00, so this is the common path. A signal that fires on
        every historical date is noise and would be filtered out before the day it matters.
        """
        with caplog.at_level(logging.WARNING, logger="tessera_embeddings.ingest.stac"):
            dates_exempt_from_correction([_Item(_RAW_ESA, "02.06")])
        assert not [r for r in caplog.records if "correction ACTIVE" in r.message]

    def test_it_stays_quiet_for_harmonised_data(self, caplog) -> None:
        """Harmonised items are exempt, so nothing is corrected and nothing is announced."""
        with caplog.at_level(logging.WARNING, logger="tessera_embeddings.ingest.stac"):
            dates_exempt_from_correction([_Item(_HARMONISED, "05.10")])
        assert not [r for r in caplog.records if "correction ACTIVE" in r.message]
