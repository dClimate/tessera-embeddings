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
    _declared_baseline,
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

    def test_an_item_with_no_read_bands_is_undetermined_not_raw(self) -> None:
        """Absence of evidence must not buy an exemption — and must not buy a CORRECTION either.

        This returned RAW, which reads as "we know it is unharmonised" when what we know is
        nothing. Nothing here can see the alias table, so a band absent under the configured name
        may be served under a native one — and calling that raw subtracts 1000 from pixels that
        may already be harmonised. The caller refuses on UNKNOWN instead of guessing.
        """
        assert item_harmonisation(_Item(None, "05.00")) is Harmonisation.UNKNOWN

    def test_an_item_serving_bands_under_native_keys_is_undetermined(self) -> None:
        """The concrete alias case, which `_prune_item_dict` in `stac.py` exists to preserve."""
        item = _Item(None, "05.00")
        item.assets = {f"B{n:02d}": {"href": f"{_HARMONISED}/B{n:02d}.tif"} for n in (2, 3, 4, 8)}
        assert item_harmonisation(item) is Harmonisation.UNKNOWN

    def test_the_live_catalogue_serves_the_configured_alias_keys(self) -> None:
        """Why UNKNOWN does not fire in production: real Element 84 items key their assets by
        the configured band names, so they classify HARMONISED on the normal path.
        """
        assert item_harmonisation(_Item(_HARMONISED, "05.00")) is Harmonisation.HARMONISED

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


class TestTheThresholdFollowsTheBaselineTheCorrectionWillApply:
    """The exemption must be decided on the caller's map, not on re-parsed metadata.

    `load_stac_items` builds a per-date baseline map, hands it to the correction, and asked
    this function separately — which re-read each item instead. The two normally agree, since
    the map was built from these items. Where they do not, the correction applies the map's
    value while the exemption was decided on the item's, so the decision described a baseline
    that was never used. Raised three times in review of PR #108.
    """

    def test_a_supplied_baseline_rescues_an_item_with_unreadable_metadata(self) -> None:
        """THE FAILURE NAMED. Malformed metadata parses as 0, which is under the threshold, so
        the date was exempted and its pixels stayed 1000 too high while the caller had
        correctly supplied 500.
        """
        item = _Item(_RAW_ESA, "not-a-number")
        assert _extract_baseline(item) == 0, "the fixture must actually be unreadable"

        assert dates_exempt_from_correction([item]) == {"2022-01-07"}, "no map: nothing to go on"
        assert dates_exempt_from_correction([item], {"2022-01-07": 500}) == set()

    def test_a_missing_baseline_property_is_also_rescued(self) -> None:
        item = _Item(_RAW_ESA, None)
        assert dates_exempt_from_correction([item], {"2022-01-07": 500}) == set()

    def test_the_map_is_read_in_the_scaled_space_the_threshold_uses(self) -> None:
        """500 is baseline 05.00, not 500.00. A map read at the wrong scale would exempt
        every date, since 5.0 is under a threshold of 400.
        """
        assert S2_BASELINE_THRESHOLD == 400
        item = _Item(_RAW_ESA, None)
        assert dates_exempt_from_correction([item], {"2022-01-07": 500}) == set()
        assert dates_exempt_from_correction([item], {"2022-01-07": 399}) == {"2022-01-07"}
        assert dates_exempt_from_correction([item], {"2022-01-07": 400}) == set(), "the threshold is inclusive"

    def test_a_readable_item_is_judged_on_its_own_baseline(self) -> None:
        """The map is a FALLBACK, not a replacement. `extract_baselines` is last-wins, so it
        carries one arbitrary item's value; letting it override a readable declaration would put
        the decision back under the caller's sort order.
        """
        item = _Item(_RAW_ESA, "05.00")
        assert dates_exempt_from_correction([item]) == set()
        assert dates_exempt_from_correction([item], {"2022-01-07": 300}) == set()

    def test_a_last_wins_map_does_not_exempt_a_post_threshold_item(self) -> None:
        """THE REGRESSION. With raw 05.00 and raw 02.06 on one day, the last-wins map supplies
        02.06 for the date. Applying that to both items read the whole day as pre-threshold,
        exempted it, and left the 05.00 pixels 1000 too high.
        """
        over = _Item(_RAW_ESA, "05.00")
        under = _Item(_RAW_ESA, "02.06")
        with pytest.raises(HeterogeneousProducerError, match="straddle the correction threshold"):
            dates_exempt_from_correction([over, under], {"2022-01-07": 206})

    def test_a_last_wins_map_does_not_refuse_a_date_that_needs_nothing(self) -> None:
        """The mirror image, from the recorded 2017-12-19 case: cloud sorting left the harmonised
        05.00 item last, so applying the date's baseline to the raw 02.06 item read it as
        post-threshold and refused a date where nothing was owed at all.
        """
        harmonised = _Item(_HARMONISED, "05.00")
        raw_old = _Item(_RAW_ESA, "02.06")
        # The map holds the HARMONISED item's 500, because it sorted last.
        assert dates_exempt_from_correction([harmonised, raw_old], {"2022-01-07": 500}) == {"2022-01-07"}

    def test_the_decision_does_not_depend_on_item_order(self) -> None:
        """The general property both regressions violated."""
        harmonised = _Item(_HARMONISED, "05.00")
        raw_old = _Item(_RAW_ESA, "02.06")
        forward = dates_exempt_from_correction([harmonised, raw_old], {"2022-01-07": 500})
        reverse = dates_exempt_from_correction([raw_old, harmonised], {"2022-01-07": 206})
        assert forward == reverse == {"2022-01-07"}

    def test_a_date_the_map_does_not_carry_falls_back_to_the_item(self) -> None:
        """A partial map must not exempt the dates it happens to omit."""
        item = _Item(_RAW_ESA, "05.00")
        assert dates_exempt_from_correction([item], {"1999-01-01": 300}) == set()

    def test_an_unreadable_item_does_not_refuse_the_date_it_was_rescued_for(self) -> None:
        """The straddle guard reads each item's OWN baseline. Counting an unreadable one as 0
        would raise for exactly the date the map is there to rescue, turning a silent wrong
        answer into a lost date rather than a correct one.
        """
        readable = _Item(_RAW_ESA, "05.00")
        unreadable = _Item(_RAW_ESA, None)
        assert dates_exempt_from_correction([readable, unreadable], {"2022-01-07": 500}) == set()

    def test_a_genuine_straddle_still_refuses(self) -> None:
        """So the test above cannot pass by having disabled the guard."""
        over = _Item(_RAW_ESA, "05.00")
        under = _Item(_RAW_ESA, "03.00")
        with pytest.raises(HeterogeneousProducerError, match="straddle the correction threshold"):
            dates_exempt_from_correction([over, under], {"2022-01-07": 500})


class TestTheActivationWarningDescribesWhatHappened:
    """The warning must not claim a correction that a refusal then prevented. PR #108 review."""

    def _records(self, caplog, items, baselines=None) -> list[logging.LogRecord]:
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tessera_embeddings.ingest.stac"):
            try:
                dates_exempt_from_correction(items, baselines)
            except HeterogeneousProducerError:
                pass
        return [r for r in caplog.records if "Baseline correction" in r.getMessage()]

    def test_a_date_that_refuses_logs_no_correction(self, caplog) -> None:
        """Emitted before the checks, the line announced a correction on a date that then
        raised and was never loaded at all.
        """
        raw_over = _Item(_RAW_ESA, "05.00")
        harmonised = _Item(_HARMONISED, "05.00")
        records = self._records(caplog, [raw_over, harmonised])
        assert records == [], f"a refused date announced a correction: {[r.getMessage() for r in records]}"

    def test_a_straddling_date_logs_no_correction(self, caplog) -> None:
        raw_over = _Item(_RAW_ESA, "05.00")
        raw_under = _Item(_RAW_ESA, "03.00")
        assert self._records(caplog, [raw_over, raw_under]) == []

    def test_a_date_that_is_actually_corrected_still_warns(self, caplog) -> None:
        """The complement: silencing the warning entirely would pass every test above."""
        records = self._records(caplog, [_Item(_RAW_ESA, "05.00")])
        assert [r.levelno for r in records] == [logging.WARNING]
        assert "ESA-archive" in records[0].getMessage()


class TestTheBaselineParserReportsUnknown:
    """`_extract_baseline` maps a missing baseline to 0, so it cannot say "unknown"."""

    @pytest.mark.parametrize("raw", [None, "", "not-a-number", "NaN", "Infinity", "-Infinity"])
    def test_an_undeclared_or_nonsense_baseline_is_unknown(self, raw) -> None:
        assert _declared_baseline(_Item(_RAW_ESA, raw)) is None

    @pytest.mark.parametrize(("raw", "scaled"), [("04.00", 400), ("05.10", 510), ("00.01", 1)])
    def test_a_real_baseline_is_scaled_by_a_hundred(self, raw: str, scaled: int) -> None:
        assert _declared_baseline(_Item(_RAW_ESA, raw)) == scaled

    def test_an_infinite_baseline_does_not_raise_out_of_the_parser(self) -> None:
        """`round(inf)` raises OverflowError, which the parser's except clause did not catch —
        so a single malformed catalogue value aborted the whole preparation.
        """
        assert _extract_baseline(_Item(_RAW_ESA, "Infinity")) == 0

    def test_a_declared_zero_is_distinguishable_from_an_absent_one(self) -> None:
        assert _declared_baseline(_Item(_RAW_ESA, "00.00")) == 0
        assert _declared_baseline(_Item(_RAW_ESA, None)) is None
        assert _extract_baseline(_Item(_RAW_ESA, "00.00")) == _extract_baseline(_Item(_RAW_ESA, None)) == 0


class TestTheCorrectorIsSkippedWhenNothingIsOwed:
    """Setting the threshold for Earth Search made the already-harmonised path the COMMON case.

    The corrector is a no-op on a date owed nothing, but not a free one: it clips, casts, adds
    and `xr.where`s every reflectance band into the graph before deciding to change nothing.
    Raised on PR #108 — before this change that branch was unreachable, so the cost never
    landed on a real ingest.
    """

    @staticmethod
    def _run(monkeypatch, items, baselines) -> list[dict]:
        import numpy as np
        import xarray as xr

        from tessera_embeddings.ingest import stac as stac_module

        data = xr.Dataset(
            {"B02": (("time", "y", "x"), np.ones((1, 2, 2), dtype="uint16"))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )
        monkeypatch.setattr(stac_module, "_load_from_stac", lambda *a, **k: data)

        calls: list[dict] = []

        def _spy(_data, corrections, **kwargs):
            calls.append({"corrections": corrections, **kwargs})
            return _data

        monkeypatch.setattr(stac_module, "_apply_baseline_corrections_by_date", _spy)
        stac_module.load_stac_items(items, "earth-search", "sentinel-2-l2a", baselines=baselines)
        return calls

    def test_an_all_harmonised_date_does_not_enter_the_corrector(self, monkeypatch) -> None:
        calls = self._run(monkeypatch, [_Item(_HARMONISED, "05.10")], {"2022-01-07": 510})
        assert calls == [], "the corrector ran for a date whose correction baseline is 0"

    def test_a_pre_threshold_date_does_not_enter_the_corrector(self, monkeypatch) -> None:
        calls = self._run(monkeypatch, [_Item(_RAW_ESA, "03.00")], {"2022-01-07": 300})
        assert calls == []

    def test_a_date_that_is_owed_the_offset_still_enters_the_corrector(self, monkeypatch) -> None:
        """The complement. Skipping unconditionally would pass both tests above."""
        calls = self._run(monkeypatch, [_Item(_RAW_ESA, "05.00")], {"2022-01-07": 500})
        assert len(calls) == 1
        assert calls[0]["corrections"] == {"2022-01-07": 500}

    def test_the_provenance_map_is_untouched_by_the_skip(self, monkeypatch) -> None:
        """The store's `baselines_applied` records the vintage of what was loaded, so the
        harmonised item's real 510 must survive a load that corrected nothing.
        """
        baselines = {"2022-01-07": 510}
        self._run(monkeypatch, [_Item(_HARMONISED, "05.10")], baselines)
        assert baselines == {"2022-01-07": 510}


class TestAnUndeterminedProducerRefusesRatherThanGuessing:
    """UNKNOWN is refused where the answer matters, and ignored where it does not. PR #108."""

    def test_a_post_threshold_undetermined_item_refuses_the_date(self) -> None:
        item = _Item(None, "05.00")
        with pytest.raises(HeterogeneousProducerError, match="none of the reflectance bands"):
            dates_exempt_from_correction([item])

    def test_a_pre_threshold_undetermined_item_is_exempt_not_refused(self) -> None:
        """Nothing is owed, so which producer served it cannot change any pixel."""
        item = _Item(None, "03.00")
        assert dates_exempt_from_correction([item]) == {"2022-01-07"}

    def test_a_normal_harmonised_date_is_unaffected(self) -> None:
        """The guard must not fire on the common path."""
        assert dates_exempt_from_correction([_Item(_HARMONISED, "05.10")]) == {"2022-01-07"}


class TestTheExemptionGroupsByTheDayTheLoaderFuses:
    """`odc.stac.load` fuses a SOLAR day. Checking UTC dates checked different sets. PR #108."""

    def test_an_unnormalised_item_is_refused_rather_than_grouped_by_utc_date(self) -> None:
        item = _Item(_RAW_ESA, "05.00")
        item.datetime = datetime(2022, 1, 7, 23, 40, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match=r"canonical\s+noon-UTC solar-day timestamp"):
            dates_exempt_from_correction([item])

    def test_a_normalised_item_groups_on_its_solar_day(self) -> None:
        """The canonical noon stamp IS the solar day, so the common path is unchanged."""
        assert dates_exempt_from_correction([_Item(_HARMONISED, "05.10")]) == {"2022-01-07"}
