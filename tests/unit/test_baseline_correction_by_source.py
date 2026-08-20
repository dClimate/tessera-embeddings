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

import numpy as np
import pytest
import xarray as xr

from tessera_embeddings.config.providers import PROVIDERS
from tessera_embeddings.config.satellites import S2_BASELINE_OFFSET, S2_BASELINE_THRESHOLD
from tessera_embeddings.ingest import stac as stac_module
from tessera_embeddings.ingest.asset_locations import (
    HARMONISED_ASSET_BUCKETS,
    PREFERRED_ASSET_BUCKETS,
    READ_ASSET_KEYS,
    REFLECTANCE_ASSET_KEYS,
    Harmonisation,
    item_harmonisation,
    item_is_in_preferred_location,
    read_asset_sources,
)
from tessera_embeddings.ingest.duplicates import select_preferred_duplicates
from tessera_embeddings.ingest.stac import (
    HeterogeneousProducerError,
    _apply_baseline_corrections_by_date,
    _declared_baseline,
    _extract_baseline,
    correction_baselines_by_date,
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

    def test_a_partly_aliased_item_is_undetermined(self) -> None:
        """The subset case, and the one `_prune_item_dict` is specifically written for: asked for
        `blue` and `scl`, carrying `B02` and `scl`. Classifying from the visible band alone let a
        harmonised alias speak for a hidden native-keyed band that may be raw, and the date-wide
        decision then corrupts the subset nothing here could see.
        """
        item = _Item(None, "05.00")
        item.assets = {"blue": {"href": f"{_HARMONISED}/blue.tif"}}
        for n in (3, 4, 8):
            item.assets[f"B{n:02d}"] = {"href": f"{_RAW_ESA}/B{n:02d}.jp2"}
        assert item_harmonisation(item) is Harmonisation.UNKNOWN

    def test_one_missing_reflectance_band_is_enough_to_be_undetermined(self) -> None:
        """The boundary: a complete-but-one set is still an incomplete set."""
        item = _Item(_HARMONISED, "05.00")
        del item.assets[REFLECTANCE_ASSET_KEYS[-1]]
        assert item_harmonisation(item) is Harmonisation.UNKNOWN

    def test_a_missing_scl_does_not_make_the_producer_undetermined(self) -> None:
        """`scl` is not a reflectance band, so it cannot affect this predicate — the complement
        of the completeness rule, and the reason the two key sets stay separate.
        """
        item = _Item(_HARMONISED, "05.00")
        del item.assets["scl"]
        assert item_harmonisation(item) is Harmonisation.HARMONISED

    def test_the_live_catalogue_serves_the_configured_alias_keys(self) -> None:
        """Why UNKNOWN does not fire in production: real Element 84 items key their assets by
        the configured band names, so they classify HARMONISED on the normal path.
        """
        assert item_harmonisation(_Item(_HARMONISED, "05.00")) is Harmonisation.HARMONISED

    def test_an_unrecognised_bucket_is_not_harmonised(self) -> None:
        """A future mirror nobody has told us about gets corrected rather than exempted."""
        assert item_harmonisation(_Item("s3://some-new-mirror/tiles/33/T/WM", "05.00")) is Harmonisation.UNKNOWN

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
        # An unidentified producer is UNDETERMINED, so it refuses rather than warning — the
        # warning's claim is specifically about the ESA archive.
        azure = _Item("https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/33/T/TG/x", "05.10")
        assert item_harmonisation(azure) is Harmonisation.UNKNOWN
        with caplog.at_level(logging.WARNING, logger="tessera_embeddings.ingest.stac"):
            assert dates_exempt_from_correction([_Item(_RAW_ESA, "05.00")]) == set()
        assert [r for r in caplog.records if "correction ACTIVE" in r.message], "the archive must warn"

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


class TestAnUnreadableBaselineRefusesRatherThanBeingGuessed:
    """An unharmonised item that declares no readable baseline is ambiguous, and refused.

    This went through three rounds of review. Reading the item naively exempted the date, because
    a malformed `s2:processing_baseline` parses as 0 and 0 is under the threshold. Taking the
    answer from the caller's per-date map instead was tried and is worse: the production path
    builds that map with `extract_baselines(day_items)` from these very items, so a lone
    unreadable item supplies the same parsed zero and is rescued from nothing, while on a
    multi-item date it inherits an arbitrary last item's value — leaving post-04.00 pixels 1000
    too high or taking 1000 off pre-04.00 ones. A figure derived from the items cannot be
    evidence about an item.
    """

    def test_a_malformed_baseline_refuses_the_date(self) -> None:
        item = _Item(_RAW_ESA, "not-a-number")
        assert _extract_baseline(item) == 0, "the fixture must actually be unreadable"
        with pytest.raises(HeterogeneousProducerError, match="no readable processing baseline"):
            dates_exempt_from_correction([item])

    def test_a_missing_baseline_property_refuses_the_date(self) -> None:
        with pytest.raises(HeterogeneousProducerError, match="no readable processing baseline"):
            dates_exempt_from_correction([_Item(_RAW_ESA, None)])

    def test_a_harmonised_item_with_no_baseline_is_harmless(self) -> None:
        """It is owed nothing whatever its baseline says, so the ambiguity cannot bite."""
        assert dates_exempt_from_correction([_Item(_HARMONISED, None)]) == {"2022-01-07"}

    def test_a_readable_item_is_judged_on_its_own_baseline(self) -> None:
        assert dates_exempt_from_correction([_Item(_RAW_ESA, "05.00")]) == set()
        assert dates_exempt_from_correction([_Item(_RAW_ESA, "03.00")]) == {"2022-01-07"}

    def test_the_threshold_is_read_in_the_scaled_space(self) -> None:
        """500 is baseline 05.00, not 500.00. A comparison at the wrong scale would exempt
        everything, since 5.0 is under a threshold of 400.
        """
        assert S2_BASELINE_THRESHOLD == 400
        assert dates_exempt_from_correction([_Item(_RAW_ESA, "04.00")]) == set(), "inclusive"
        assert dates_exempt_from_correction([_Item(_RAW_ESA, "03.99")]) == {"2022-01-07"}

    def test_the_decision_does_not_depend_on_item_order(self) -> None:
        """The property the map-based version violated: a last-wins figure made the answer a
        function of the caller's sort.
        """
        harmonised = _Item(_HARMONISED, "05.00")
        raw_old = _Item(_RAW_ESA, "02.06")
        assert dates_exempt_from_correction([harmonised, raw_old]) == {"2022-01-07"}
        assert dates_exempt_from_correction([raw_old, harmonised]) == {"2022-01-07"}

    def test_a_genuine_straddle_still_refuses(self) -> None:
        with pytest.raises(HeterogeneousProducerError, match="straddle the correction threshold"):
            dates_exempt_from_correction([_Item(_RAW_ESA, "05.00"), _Item(_RAW_ESA, "03.00")])


class TestTheActivationWarningDescribesWhatHappened:
    """The warning must not claim a correction that a refusal then prevented. PR #108 review."""

    def _records(self, caplog, items) -> list[logging.LogRecord]:
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tessera_embeddings.ingest.stac"):
            try:
                dates_exempt_from_correction(items)
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


class TestThePerItemCheckIsScopedToTheCollectionThatNeedsIt:
    """Planetary Computer keys its assets NATIVELY, so a per-item read of asset locations finds
    nothing there. Raised on PR #107 and confirmed against the live catalogue: real PC items at
    baseline 05.10 expose `B02`..`B12`, `SCL` and none of the configured common names.

    Running the per-item check anyway classified every modern PC item UNKNOWN and refused every
    date at baseline >= 04.00 — a working provider broken by a guard meant for another one. PC
    serves ESA's values unharmonised throughout, so its answer belongs to the collection.
    """

    #: Exactly what the live API returns, minus the assets the ingest never reads.
    _PC_NATIVE_KEYS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL")

    def _pc_item(self, baseline: str = "05.10") -> _Item:
        item = _Item(None, baseline)
        item.assets = {
            k: {"href": f"https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/{k}.tif"}
            for k in self._PC_NATIVE_KEYS
        }
        return item

    def test_a_native_keyed_item_reads_as_undetermined_on_its_own(self) -> None:
        """The predicate genuinely cannot answer for these items — which is why the CALLER, not
        the predicate, is what had to change.
        """
        assert item_harmonisation(self._pc_item()) is Harmonisation.UNKNOWN

    def test_the_planetary_computer_collection_does_not_use_the_per_item_check(self) -> None:
        config = PROVIDERS["planetary-computer"].collections["sentinel-2-l2a"]
        assert config.requires_baseline_correction is True, "PC data is raw and IS owed the offset"
        assert config.harmonisation_varies_by_item is False

    def test_earth_search_does_use_the_per_item_check(self) -> None:
        """The complement: turning the flag off everywhere would pass the test above."""
        assert PROVIDERS["earth-search"].collections["sentinel-2-l2a"].harmonisation_varies_by_item is True

    def test_a_native_keyed_item_is_corrected_rather_than_refused(self, monkeypatch) -> None:
        """End to end through the documented entry point: the PC path must reach the corrector
        with its declared baseline, not raise.
        """
        data = xr.Dataset(
            {"B02": (("time", "y", "x"), np.ones((1, 2, 2), dtype="uint16"))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )
        monkeypatch.setattr(stac_module, "_load_from_stac", lambda *a, **k: data)
        calls: list[dict] = []
        monkeypatch.setattr(
            stac_module,
            "_apply_baseline_corrections_by_date",
            lambda d, corrections, **kw: calls.append({"corrections": corrections}) or d,
        )
        stac_module.load_stac_items(
            [self._pc_item()], "planetary-computer", "sentinel-2-l2a", baselines={"2022-01-07": 510}
        )
        assert calls == [{"corrections": {"2022-01-07": 510}}], "PC data must still be corrected"

    def test_the_earth_search_path_still_refuses_an_undetermined_item(self, monkeypatch) -> None:
        """The guard must survive being scoped — it still fires where it is meant to."""
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.ones((1, 2, 2), dtype="uint16"))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )
        monkeypatch.setattr(stac_module, "_load_from_stac", lambda *a, **k: data)
        with pytest.raises(HeterogeneousProducerError, match="none of the reflectance bands"):
            stac_module.load_stac_items(
                [self._pc_item()], "earth-search", "sentinel-2-l2a", baselines={"2022-01-07": 510}
            )


class TestTheTwoBucketSetsAreIndependent:
    """`PREFERRED_ASSET_BUCKETS` and `HARMONISED_ASSET_BUCKETS` hold the same values today and
    mean different things: one is "cheap to read from here", the other is "the offset is already
    subtracted". Nothing may rely on the coincidence — the day someone mirrors UNHARMONISED data
    in region, the sets diverge and code that conflated them silently skips a correction.
    """

    def test_locality_reads_only_the_locality_set(self) -> None:
        item = _Item(None, "05.00")
        item.assets = {k: {"href": f"s3://a-new-in-region-mirror/{k}"} for k in READ_ASSET_KEYS}
        assert item_is_in_preferred_location(item, buckets=frozenset({"a-new-in-region-mirror"})) is True
        # The same bucket, asked the harmonisation question, is NOT harmonised.
        assert item_harmonisation(item) is Harmonisation.UNKNOWN

    def test_harmonisation_reads_only_the_harmonisation_set(self) -> None:
        item = _Item(None, "05.00")
        item.assets = {k: {"href": f"s3://some-harmonised-mirror/{k}"} for k in READ_ASSET_KEYS}
        assert item_harmonisation(item, buckets=frozenset({"some-harmonised-mirror"})) is Harmonisation.HARMONISED
        assert item_is_in_preferred_location(item) is False

    def test_the_raw_archive_is_absent_from_both(self) -> None:
        """Adding it to either would silently reinstate a different bug."""
        assert "sentinel-s2-l2a" not in HARMONISED_ASSET_BUCKETS
        assert "sentinel-s2-l2a" not in PREFERRED_ASSET_BUCKETS


class TestCorrectedAndExemptDatesShareOneStorageDtype:
    """A ROI store's arrays take their dtype from the FIRST date's dataset
    (`zarr_store.py`: `var_dtypes={v: day_ds[v].dtype ...}`), and existing production stores are
    unsigned. So a corrected date and an exempt date must come out of the corrector with the same
    dtype, or the store's dtype depends on which date happened to land first and every date of
    the other kind is cast on the way in. Raised on PR #107.

    The failure is silent and severe in both directions: a negative corrected pixel written to an
    unsigned store reads as roughly 65535, and an uncorrected bright pixel written to a signed
    store wraps negative. Either looks like extreme reflectance to inference.
    """

    @staticmethod
    def _dataset(values: np.ndarray, date: str) -> xr.Dataset:

        return xr.Dataset(
            {"blue": (("time", "y", "x"), values)},
            coords={"time": [np.datetime64(f"{date}T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )

    def test_a_corrected_date_keeps_the_unsigned_dtype_it_arrived_with(self) -> None:

        ds = self._dataset(np.array([[[1500, 2000], [3000, 4000]]], dtype=np.uint16), "2022-01-07")
        out = _apply_baseline_corrections_by_date(ds, baselines={"2022-01-07": 500}, bands=["blue"])
        assert out["blue"].dtype == np.uint16
        np.testing.assert_array_equal(out["blue"].values, np.array([[[500, 1000], [2000, 3000]]], dtype=np.uint16))

    def test_an_exempt_date_keeps_the_same_dtype(self) -> None:

        ds = self._dataset(np.array([[[1500, 2000], [3000, 4000]]], dtype=np.uint16), "2022-01-07")
        out = _apply_baseline_corrections_by_date(ds, baselines={"2022-01-07": 0}, bands=["blue"])
        assert out["blue"].dtype == np.uint16

    def test_no_corrected_pixel_can_go_negative(self) -> None:
        """What makes the unsigned dtype safe: the subtraction only touches pixels already at or
        above the offset, so the floor is 0 rather than -1000.
        """
        ds = self._dataset(np.array([[[0, 999], [1000, 1001]]], dtype=np.uint16), "2022-01-07")
        out = _apply_baseline_corrections_by_date(ds, baselines={"2022-01-07": 500}, bands=["blue"])
        # 0 and 999 are below the offset and untouched; 1000 and 1001 lose exactly 1000.
        np.testing.assert_array_equal(out["blue"].values, np.array([[[0, 999], [0, 1]]], dtype=np.uint16))
        assert out["blue"].values.min() >= 0

    def test_a_bright_pixel_does_not_wrap(self) -> None:
        """`clip(max=64535) - 1000` is 63535, which does not fit int16 — so the previous cast
        wrapped bright pixels even without any dtype mixing.
        """
        ds = self._dataset(np.array([[[60000, 65535], [40000, 33000]]], dtype=np.uint16), "2022-01-07")
        out = _apply_baseline_corrections_by_date(ds, baselines={"2022-01-07": 500}, bands=["blue"])
        assert out["blue"].values.min() >= 0, f"a bright pixel wrapped: {out['blue'].values}"
        np.testing.assert_array_equal(out["blue"].values, np.array([[[59000, 64535], [39000, 32000]]], dtype=np.uint16))

    def test_the_signed_mode_is_opt_in(self) -> None:
        """It is still available and still returns int16 — it is just no longer the default, since
        it cannot round-trip into the unsigned store this repo writes.
        """
        ds = self._dataset(np.array([[[500, 800], [1000, 1500]]], dtype=np.uint16), "2022-01-07")
        out = _apply_baseline_corrections_by_date(
            ds, baselines={"2022-01-07": 500}, bands=["blue"], preserve_low_values=False
        )
        assert out["blue"].dtype == np.int16
        assert out["blue"].values.min() < 0


class TestScalingCannotOverflowThePicker:
    """`math.isfinite` was checked BEFORE the multiplication, so a finite but enormous value
    became infinity when scaled and `round()` raised `OverflowError` — aborting the whole batch
    over one item's metadata. Raised on PR #107 after the earlier non-finite fix.
    """

    @pytest.mark.parametrize("raw", ["1e308", "-1e308", "9" * 400])
    def test_a_value_that_overflows_when_scaled_reads_as_unknown(self, raw: str) -> None:
        assert _declared_baseline(_Item(_RAW_ESA, raw)) is None
        assert _extract_baseline(_Item(_RAW_ESA, raw)) == 0

    def test_a_large_but_scalable_value_still_parses(self) -> None:
        """So the guard rejects only what genuinely cannot be scaled."""
        assert _declared_baseline(_Item(_RAW_ESA, "1e10")) == 10**12

    def test_the_batch_is_not_aborted_by_one_such_item(self) -> None:
        """The consequence: `extract_baselines` must survive it rather than raising."""
        good = _Item(_HARMONISED, "05.10")
        bad = _Item(_HARMONISED, "1e308")
        assert extract_baselines([good, bad])["2022-01-07"] in (510, 0)


class TestAssetSourcesCannotLoseAKey:
    """The primitive that generated the same defect three times, now typed so it cannot.

    Its predecessor returned a bare list of the buckets it managed to resolve and dropped the
    rest, so every caller had to remember to compare the length against what it asked for. Three
    did not: locality was granted from one local band, a producer was named from one visible band,
    and a subset was read as evidence about the whole set.
    """

    def test_a_complete_set_reports_complete(self) -> None:
        sources = read_asset_sources(_Item(_HARMONISED, "05.00"))
        assert sources.complete is True
        assert sources.missing == ()
        assert set(sources.buckets) == set(READ_ASSET_KEYS)

    def test_a_missing_key_is_named_rather_than_dropped(self) -> None:
        item = _Item(_HARMONISED, "05.00")
        del item.assets[READ_ASSET_KEYS[0]]
        sources = read_asset_sources(item)
        assert sources.complete is False
        assert sources.missing == (READ_ASSET_KEYS[0],)

    def test_an_href_less_key_counts_as_missing(self) -> None:
        item = _Item(_HARMONISED, "05.00")
        item.assets[READ_ASSET_KEYS[0]] = {}
        assert read_asset_sources(item).missing == (READ_ASSET_KEYS[0],)

    def test_an_unrecognised_bucket_is_a_real_answer_not_a_missing_key(self) -> None:
        """The distinction the flat list could not express: "served from somewhere we have not
        listed" is evidence, "not served at all" is the absence of evidence.
        """
        item = _Item(None, "05.00")
        item.assets = {k: {"href": f"https://elsewhere.example/{k}"} for k in READ_ASSET_KEYS}
        sources = read_asset_sources(item)
        assert sources.complete is True
        assert set(sources.buckets.values()) == {None}

    def test_all_in_is_false_for_an_incomplete_set(self) -> None:
        """The property every one of those three defects needed and none of them checked."""
        item = _Item(_HARMONISED, "05.00")
        for key in READ_ASSET_KEYS[1:]:
            del item.assets[key]
        sources = read_asset_sources(item)
        assert sources.any_in(HARMONISED_ASSET_BUCKETS) is True, "the one visible band IS harmonised"
        assert sources.all_in(HARMONISED_ASSET_BUCKETS) is False, "but that says nothing about the rest"

    def test_an_item_with_no_assets_is_empty_and_incomplete(self) -> None:
        sources = read_asset_sources(_Item(None, "05.00"))
        assert sources.empty is True
        assert sources.complete is False


class TestTheCorrectionValueComesFromTheSameEvidenceAsTheDecision:
    """The exemption decided from the items while the value came from the caller's map, so where
    they disagreed the correction quietly did nothing. Raised on PR #107.
    """

    def test_a_date_the_items_show_is_owed_gets_a_usable_correction_baseline(self) -> None:
        assert correction_baselines_by_date([_Item(_RAW_ESA, "05.00")]) == {"2022-01-07": 500}

    def test_an_exempt_date_is_reported_as_zero(self) -> None:
        assert correction_baselines_by_date([_Item(_HARMONISED, "05.10")]) == {"2022-01-07": 0}

    def test_a_pre_threshold_raw_date_is_reported_as_zero(self) -> None:
        """It is exempt — nothing in it is owed the offset — and this map answers "correct at
        what", not "what was declared". Provenance is `extract_baselines`' job and still reports
        the real 300.
        """
        assert correction_baselines_by_date([_Item(_RAW_ESA, "03.00")]) == {"2022-01-07": 0}
        assert extract_baselines([_Item(_RAW_ESA, "03.00")]) == {"2022-01-07": 300}

    def test_the_highest_declared_baseline_of_a_date_is_used(self) -> None:
        """They cannot meaningfully disagree — a straddling date refuses — so the max is only
        ever picking between equals above the threshold.
        """
        items = [_Item(_RAW_ESA, "05.00"), _Item(_RAW_ESA, "05.10")]
        assert correction_baselines_by_date(items) == {"2022-01-07": 510}

    def test_a_harmonised_item_does_not_contribute_a_correction_value(self) -> None:
        """The mixed-producer day: the raw item is pre-threshold so nothing is owed, and the
        harmonised item's 05.10 must not become the date's correction baseline.
        """
        items = [_Item(_HARMONISED, "05.10"), _Item(_RAW_ESA, "02.06")]
        assert correction_baselines_by_date(items) == {"2022-01-07": 0}

    def test_the_end_to_end_path_corrects_a_date_the_caller_map_omits(self, monkeypatch) -> None:
        """THE FAILURE. The corrector reads a missing entry as 0, so an owed date was skipped."""
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.full((1, 2, 2), 3000, dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )
        monkeypatch.setattr(stac_module, "_load_from_stac", lambda *a, **k: data)
        calls: list[dict] = []
        monkeypatch.setattr(
            stac_module,
            "_apply_baseline_corrections_by_date",
            lambda d, corrections, **kw: calls.append({"corrections": corrections}) or d,
        )
        stac_module.load_stac_items(
            [_Item(_RAW_ESA, "05.00")], "earth-search", "sentinel-2-l2a", baselines={"1999-01-01": 300}
        )
        assert calls == [{"corrections": {"2022-01-07": 500}}], "an owed date was skipped"

    def test_provenance_is_not_touched(self) -> None:
        """`extract_baselines` still reports what each item declared, including for exempt dates,
        because it is what reaches the store's `baselines_applied`.
        """
        assert extract_baselines([_Item(_HARMONISED, "05.10")]) == {"2022-01-07": 510}


class TestTheGenericEntryPointPrunesAndTheRoiLadderSurvives:
    """Duplicate selection belongs to the layer that owns the fallback ladder — and `ingest_tile`
    is the layer that has none.

    Three findings interlock here. Only `s2_roi` used to prune, so the generic entry point handed
    the producer check an unpruned set and a harmonised COG beside a raw reprocessing of one
    acquisition refused a date that selecting one copy resolves. Pruning inside the shared
    `query_stac_items` fixed that and broke two other things: the caller's baseline map had already
    been built from the unpruned list, so a REJECTED copy sorting last supplied the recorded
    baseline for pixels the selected copy provided — and `s2_roi` runs its own selection after that
    query and KEEPS the rejected copies as the ladder `step_down_copies` walks, so pruning upstream
    left it with nothing to step down to and one unreadable object lost the date.
    """

    @staticmethod
    def _pair() -> list[_Item]:
        """One acquisition, two producers, DIFFERENT baselines, the rejected copy sorted last."""
        harmonised = _Item(_HARMONISED, "05.10")
        harmonised.id = "S2A_33TWM_20220107_1_L2A"
        harmonised.properties["s2:sequence"] = "1"
        raw = _Item(_RAW_ESA, "02.06")
        raw.id = "S2A_33TWM_20220107_0_L2A"
        raw.properties["s2:sequence"] = "0"
        for item in (harmonised, raw):
            item.properties["grid:code"] = "MGRS-33TWM"
            item.properties["datetime"] = "2022-01-07T10:20:31.024000Z"
        return [harmonised, raw]

    def test_the_shared_query_leaves_duplicates_for_the_roi_driver(self) -> None:
        """It must NOT prune: `s2_roi` selects afterwards and needs the rejected copies as its
        fallback ladder. An empty ladder turns one unreadable object into a lost date.
        """
        items, _ = stac_module.query_stac_items(
            provider="earth-search",
            collection="sentinel-2-l2a",
            tile_id="33TWM",
            start_date="2022-01-01",
            end_date="2022-01-31",
            mid_longitude=15.0,
            item_provider_fn=lambda **_: self._pair(),
        )
        assert len(items) == 2, "the shared query pruned copies the ROI ladder needs"
        _, alternates = select_preferred_duplicates(items)
        assert alternates, "the ROI driver would have had nothing to step down to"

    def test_the_generic_entry_point_prunes(self, monkeypatch) -> None:
        loaded: list[str] = []
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )

        def fake_load(items, *a, **k):
            loaded.extend(i.id for i in items)
            return data

        monkeypatch.setattr(stac_module, "_load_from_stac", fake_load)
        monkeypatch.setattr(stac_module, "_apply_baseline_corrections_by_date", lambda d, *a, **k: d)
        _, baselines = stac_module.ingest_tile(
            provider="earth-search",
            collection="sentinel-2-l2a",
            tile_id="33TWM",
            start_date="2022-01-01",
            end_date="2022-01-31",
            mid_longitude=15.0,
            item_provider_fn=lambda **_: self._pair(),
        )
        assert loaded == ["S2A_33TWM_20220107_1_L2A"], "the generic path refused instead of selecting"
        # And provenance describes the copy that was kept, not the one that sorted last.
        assert baselines == {"2022-01-07": 510}


class TestEverySentinel2PathHasASelectionOwner:
    """Three entry points, three reasons to select, and one path that had none.

    `s2_roi` selects to build its fallback ladder, `ingest_tile` selects before extracting
    provenance — and the documented `query_stac_items` -> `load_stac_items` workflow passed through
    neither, so an unpruned pair of producers for one acquisition reached the correction decision as
    a genuine conflict. The loader selects too, idempotently.
    """

    @staticmethod
    def _pair() -> list[_Item]:
        harmonised = _Item(_HARMONISED, "05.10")
        harmonised.id = "S2A_33TWM_20220107_1_L2A"
        harmonised.properties["s2:sequence"] = "1"
        raw = _Item(_RAW_ESA, "05.00")
        raw.id = "S2A_33TWM_20220107_0_L2A"
        raw.properties["s2:sequence"] = "0"
        for item in (harmonised, raw):
            item.properties["grid:code"] = "MGRS-33TWM"
            item.properties["datetime"] = "2022-01-07T10:20:31.024000Z"
        return [harmonised, raw]

    def test_the_split_workflow_loads_rather_than_refusing(self, monkeypatch) -> None:
        loaded: list[str] = []
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )

        def fake_load(items, *a, **k):
            loaded.extend(i.id for i in items)
            return data

        monkeypatch.setattr(stac_module, "_load_from_stac", fake_load)
        monkeypatch.setattr(stac_module, "_apply_baseline_corrections_by_date", lambda d, *a, **k: d)
        # Exactly the documented two-step: the query does not prune, so the loader must.
        items, baselines = stac_module.query_stac_items(
            provider="earth-search",
            collection="sentinel-2-l2a",
            tile_id="33TWM",
            start_date="2022-01-01",
            end_date="2022-01-31",
            mid_longitude=15.0,
            item_provider_fn=lambda **_: self._pair(),
        )
        assert len(items) == 2, "the query must leave duplicates for the ROI ladder"
        stac_module.load_stac_items(items, "earth-search", "sentinel-2-l2a", baselines=baselines)
        assert loaded == ["S2A_33TWM_20220107_1_L2A"]

    def test_selecting_twice_changes_nothing(self) -> None:
        """What makes it safe to select in more than one place."""
        once, _ = select_preferred_duplicates(self._pair())
        twice, alternates = select_preferred_duplicates(once)
        assert [i.id for i in twice] == [i.id for i in once]
        assert alternates == {}


class TestRankingUsesTheRequestedAssetsNotThePruningSet:
    """`_loadable_assets` is a pruning set and deliberately generous — it keeps `scl` for any
    collection that has one, whether or not the call asks for it. Ranking on that penalises a copy
    for lacking an asset the load never reads. Raised on PR #107.
    """

    def test_scl_is_not_requested_unless_asked_for(self) -> None:
        config = PROVIDERS["earth-search"].collections["sentinel-2-l2a"]
        assert config.has_scl is True
        assert "scl" in stac_module._loadable_assets(config), "the pruning set keeps it"
        assert "scl" not in stac_module._requested_assets(config), "the read set does not"

    def test_an_explicit_extra_band_is_requested(self) -> None:
        config = PROVIDERS["earth-search"].collections["sentinel-2-l2a"]
        assert "scl" in stac_module._requested_assets(config, ["scl"])

    def test_the_requested_set_does_not_duplicate_a_configured_band(self) -> None:
        config = PROVIDERS["earth-search"].collections["sentinel-2-l2a"]
        requested = stac_module._requested_assets(config, ["blue", "qa"])
        assert requested.count("blue") == 1
        assert requested[-1] == "qa"

    def test_a_copy_lacking_scl_is_not_penalised_when_scl_is_not_read(self) -> None:
        """The consequence: an equal-baseline local copy losing to a remote one over an asset this
        call never reads, which is an unnecessary cross-region read.
        """
        config = PROVIDERS["earth-search"].collections["sentinel-2-l2a"]
        keys = stac_module._requested_assets(config)
        local_no_scl = _Item(None, "05.00")
        local_no_scl.id = "local"
        local_no_scl.assets = {b: {"href": f"{_HARMONISED}/{b}.tif"} for b in config.bands}
        assert item_is_in_preferred_location(local_no_scl, keys=keys) is True
        assert item_is_in_preferred_location(local_no_scl) is False, "the fixed set demands scl"


class TestTheSplitWorkflowProvenanceFollowsTheSelectedCopy:
    """The loader selects duplicates, so the caller's baseline map goes stale.

    `query_stac_items` builds that map from the unpruned list and the caller keeps it for
    `baselines_applied`, so a rejected copy that sorted last could describe pixels the selected
    copy provided. Raised on PR #107.
    """

    @staticmethod
    def _pair() -> list[_Item]:
        """Different baselines, and the copy that will be REJECTED sorts last."""
        winner = _Item(_HARMONISED, "05.10")
        winner.id = "S2A_33TWM_20220107_1_L2A"
        winner.properties["s2:sequence"] = "1"
        loser = _Item(_HARMONISED, "02.06")
        loser.id = "S2A_33TWM_20220107_0_L2A"
        loser.properties["s2:sequence"] = "0"
        for item in (winner, loser):
            item.properties["grid:code"] = "MGRS-33TWM"
            item.properties["datetime"] = "2022-01-07T10:20:31.024000Z"
        return [winner, loser]

    def test_the_supplied_map_is_realigned_with_the_survivor(self, monkeypatch) -> None:
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )
        monkeypatch.setattr(stac_module, "_load_from_stac", lambda *a, **k: data)
        items = self._pair()
        # What `query_stac_items` hands back: last-wins over the UNPRUNED list.
        baselines = extract_baselines(items)
        assert baselines == {"2022-01-07": 206}, "the fixture must start with the rejected copy's value"

        stac_module.load_stac_items(items, "earth-search", "sentinel-2-l2a", baselines=baselines)
        assert baselines == {"2022-01-07": 510}, "provenance still described the rejected copy"

    def test_a_map_is_untouched_when_there_are_no_duplicates(self, monkeypatch) -> None:
        """It must only realign what selection actually changed."""
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )
        monkeypatch.setattr(stac_module, "_load_from_stac", lambda *a, **k: data)
        baselines = {"2022-01-07": 510}
        stac_module.load_stac_items([self._pair()[0]], "earth-search", "sentinel-2-l2a", baselines=baselines)
        assert baselines == {"2022-01-07": 510}


class TestTheThemesThatGeneratedTheReviewStayClosed:
    """Guards against the SHAPES of defect this branch was reviewed for, not just the instances.

    Grouping the review findings, most fell into a few repeating shapes: a set judged over a
    subset of itself, one fact with two implementations that drift, and a guard whose position is
    load-bearing but incidental. These assert the invariants that keep each shape closed, so an
    unreported instance fails here rather than in a later review.
    """

    def test_there_is_one_definition_of_what_a_load_reads(self) -> None:
        """The driver and the generic path must not define their read set twice."""
        from tessera_embeddings.ingest.s2_roi import _LOADED_EXTRA_BANDS, _READ_ASSET_KEYS

        config = PROVIDERS["earth-search"].collections["sentinel-2-l2a"]
        assert stac_module._requested_assets(config, _LOADED_EXTRA_BANDS) == _READ_ASSET_KEYS

    def test_the_reflectance_set_matches_the_collection_it_judges(self) -> None:
        """`item_harmonisation` defaults to a module constant while the decision is made for a
        specific collection. They agree today; if a collection's bands change, this fails rather
        than silently judging the producer over the wrong bands.
        """
        for provider, collection in (("earth-search", "sentinel-2-l2a"), ("planetary-computer", "sentinel-2-l2a")):
            config = PROVIDERS[provider].collections[collection]
            if config.harmonisation_varies_by_item:
                assert set(REFLECTANCE_ASSET_KEYS) == set(config.bands), f"{provider}/{collection}"

    def test_scl_is_read_but_never_judged_for_harmonisation(self) -> None:
        """The two key sets differ by exactly `scl`, and that difference is the point."""
        assert set(READ_ASSET_KEYS) - set(REFLECTANCE_ASSET_KEYS) == {"scl"}

    def test_every_refusal_is_gated_on_the_date_being_owed_a_correction(self) -> None:
        """A pre-threshold date must never refuse, whatever else is wrong with it: which producer
        served it, or whether they disagree, cannot change a pixel that gets no offset.
        """
        pre_threshold = [
            [_Item(_HARMONISED, "03.00"), _Item(_RAW_ESA, "03.00")],  # mixed producers
            [_Item(None, "03.00")],  # producer undeterminable
            [_Item(_RAW_ESA, "03.00"), _Item(_RAW_ESA, "02.06")],  # differing raw baselines
        ]
        for items in pre_threshold:
            assert dates_exempt_from_correction(items) == {"2022-01-07"}, [i.assets for i in items]

    def test_the_same_situations_refuse_once_the_date_is_owed_a_correction(self) -> None:
        """The complement, or the test above would pass with the refusals deleted."""
        for items in (
            [_Item(_HARMONISED, "05.00"), _Item(_RAW_ESA, "05.00")],
            [_Item(None, "05.00")],
            [_Item(_RAW_ESA, "05.00"), _Item(_RAW_ESA, "02.06")],
        ):
            with pytest.raises(HeterogeneousProducerError):
                dates_exempt_from_correction(items)

    def test_provenance_is_never_the_correction_input(self) -> None:
        """Two maps, two purposes: what each item declared, and what to correct at. Collapsing
        them is what made a store report baseline 0 for imagery processed at 05.10.
        """
        harmonised = [_Item(_HARMONISED, "05.10")]
        assert extract_baselines(harmonised) == {"2022-01-07": 510}
        assert correction_baselines_by_date(harmonised) == {"2022-01-07": 0}


class TestTheCorrectionDoesNotDependOnTheCallerSupplyingAMap:
    """Where producers can disagree the correction is derived from the ITEMS, so gating the whole
    block on the caller's map being truthy left a raw post-04.00 item uncorrected. Raised twice.
    """

    @staticmethod
    def _run(monkeypatch, items, baselines) -> list[dict]:
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.full((1, 2, 2), 3000, dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )
        monkeypatch.setattr(stac_module, "_load_from_stac", lambda *a, **k: data)
        calls: list[dict] = []
        monkeypatch.setattr(
            stac_module,
            "_apply_baseline_corrections_by_date",
            lambda d, corrections, **kw: calls.append({"corrections": corrections}) or d,
        )
        stac_module.load_stac_items(items, "earth-search", "sentinel-2-l2a", baselines=baselines)
        return calls

    @pytest.mark.parametrize("baselines", [None, {}])
    def test_a_raw_post_threshold_item_is_corrected_with_no_map(self, monkeypatch, baselines) -> None:
        calls = self._run(monkeypatch, [_Item(_RAW_ESA, "05.00")], baselines)
        assert calls == [{"corrections": {"2022-01-07": 500}}], "the offset was left in the pixels"

    @pytest.mark.parametrize("baselines", [None, {}])
    def test_a_harmonised_item_is_still_not_corrected(self, monkeypatch, baselines) -> None:
        """The complement: running the block unconditionally must not correct what is exempt."""
        assert self._run(monkeypatch, [_Item(_HARMONISED, "05.10")], baselines) == []

    def test_a_collection_without_the_per_item_decision_still_needs_its_map(self, monkeypatch) -> None:
        """Planetary Computer's answer IS the caller's map, so with no map there is nothing to
        apply and the block must stay out of the way.
        """
        item = _Item(None, "05.10")
        item.assets = {k: {"href": f"https://x.blob.core.windows.net/{k}"} for k in ("B02", "B03")}
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )
        monkeypatch.setattr(stac_module, "_load_from_stac", lambda *a, **k: data)
        calls: list[dict] = []
        monkeypatch.setattr(stac_module, "_apply_baseline_corrections_by_date", lambda d, c, **kw: calls.append(c) or d)
        stac_module.load_stac_items([item], "planetary-computer", "sentinel-2-l2a", baselines=None)
        assert calls == []


class TestProvenanceKeepsTheDatesSelectionDidNotTouch:
    """`query_stac_items` returns entries for every queried date, including ones whose items were
    filtered out as already present. Replacing the map dropped those. Raised on PR #107.
    """

    def test_an_existing_date_survives_the_realignment(self, monkeypatch) -> None:
        contested = _Item(_HARMONISED, "05.10")
        contested.id = "S2A_33TWM_20220107_1_L2A"
        contested.properties.update({"s2:sequence": "1", "grid:code": "MGRS-33TWM"})
        contested.properties["datetime"] = "2022-01-07T10:20:31.024000Z"
        rejected = _Item(_HARMONISED, "02.06")
        rejected.id = "S2A_33TWM_20220107_0_L2A"
        rejected.properties.update({"s2:sequence": "0", "grid:code": "MGRS-33TWM"})
        rejected.properties["datetime"] = "2022-01-07T10:20:31.024000Z"

        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )
        monkeypatch.setattr(stac_module, "_load_from_stac", lambda *a, **k: data)
        monkeypatch.setattr(stac_module, "_apply_baseline_corrections_by_date", lambda d, *a, **k: d)
        # An existing date whose items were filtered out keeps its provenance entry.
        monkeypatch.setattr(
            stac_module,
            "query_stac_items",
            lambda **_: ([contested, rejected], {"2022-01-07": 206, "2021-12-25": 500}),
        )
        _, baselines = stac_module.ingest_tile(
            provider="earth-search",
            collection="sentinel-2-l2a",
            tile_id="33TWM",
            start_date="2022-01-01",
            end_date="2022-01-31",
            mid_longitude=15.0,
        )
        assert baselines["2022-01-07"] == 510, "the contested date must follow the survivor"
        assert baselines["2021-12-25"] == 500, "an already-present date lost its provenance"


class TestAnUnlistedBucketIsUndeterminedNotRaw:
    """`RAW` means "we have identified this producer as unharmonised". An unlisted bucket is not
    that: were a harmonised mirror to move behind a new bucket or CDN, calling it raw would
    subtract the offset from pixels that already had it removed. Raised on PR #107.
    """

    def test_a_complete_set_from_an_unlisted_bucket_is_undetermined(self) -> None:
        item = _Item(None, "05.00")
        item.assets = {k: {"href": f"s3://some-new-mirror/{k}"} for k in READ_ASSET_KEYS}
        assert item_harmonisation(item) is Harmonisation.UNKNOWN

    def test_the_named_raw_archive_is_still_raw(self) -> None:
        """The complement: a producer we HAVE identified must still be corrected."""
        assert item_harmonisation(_Item(_RAW_ESA, "05.00")) is Harmonisation.RAW

    def test_a_harmonised_producer_is_still_harmonised(self) -> None:
        assert item_harmonisation(_Item(_HARMONISED, "05.00")) is Harmonisation.HARMONISED

    def test_an_unlisted_bucket_refuses_rather_than_double_correcting(self) -> None:
        item = _Item(None, "05.00")
        item.assets = {k: {"href": f"s3://some-new-mirror/{k}"} for k in READ_ASSET_KEYS}
        with pytest.raises(HeterogeneousProducerError, match="cannot be determined"):
            dates_exempt_from_correction([item])

    def test_a_pre_threshold_unlisted_bucket_is_exempt(self) -> None:
        """Gated on the date being owed a correction, like every other refusal."""
        item = _Item(None, "03.00")
        item.assets = {k: {"href": f"s3://some-new-mirror/{k}"} for k in READ_ASSET_KEYS}
        assert dates_exempt_from_correction([item]) == {"2022-01-07"}


class TestTheCorrectionDoesNotCorruptTheBrightestCodes:
    """Clamping before subtraction cost the top `abs(offset)` input codes their range: 65535 was
    clipped to 64535 and then reduced to 63535 rather than 64535. Raised on PR #107.
    """

    @staticmethod
    def _corrected(values: np.ndarray) -> np.ndarray:
        ds = xr.Dataset(
            {"blue": (("time", "y", "x"), values)},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0], "x": list(range(values.shape[2]))},
        )
        out = stac_module._apply_baseline_corrections_by_date(ds, baselines={"2022-01-07": 500}, bands=["blue"])
        return out["blue"].values

    def test_the_top_codes_lose_exactly_the_offset(self) -> None:
        got = self._corrected(np.array([[[65535, 65000, 64535, 30000]]], dtype=np.uint16))
        np.testing.assert_array_equal(got, np.array([[[64535, 64000, 63535, 29000]]], dtype=np.uint16))

    def test_nothing_wraps_or_saturates(self) -> None:
        values = np.array([[[65535, 60000, 1000, 999, 0]]], dtype=np.uint16)
        got = self._corrected(values)
        assert got.dtype == np.uint16
        assert got.min() >= 0
        # Every eligible pixel loses exactly the offset; the two below it are untouched.
        np.testing.assert_array_equal(got, np.array([[[64535, 59000, 0, 999, 0]]], dtype=np.uint16))


class TestRawRequiresEveryBandFromAKnownProducer:
    """One raw band beside bands from an unlisted bucket is not a raw item.

    `any_in` classified such a set as RAW, and the date-wide corrector then subtracted the offset
    from bands whose producer is unknown and may already have had it removed. Raised on PR #107.
    """

    def test_raw_plus_unlisted_is_undetermined(self) -> None:
        item = _Item(None, "05.00")
        item.assets = {k: {"href": f"s3://some-new-mirror/{k}"} for k in READ_ASSET_KEYS}
        item.assets["red"] = {"href": f"{_RAW_ESA}/B04.jp2"}
        assert item_harmonisation(item) is Harmonisation.UNKNOWN

    def test_every_band_from_the_known_archive_is_raw(self) -> None:
        """The complement: an item genuinely served by the archive must still be corrected."""
        assert item_harmonisation(_Item(_RAW_ESA, "05.00")) is Harmonisation.RAW

    def test_raw_plus_harmonised_is_still_mixed(self) -> None:
        """The three-way distinction survives: straddling KNOWN producers stays MIXED."""
        item = _Item(_HARMONISED, "05.00")
        item.assets["red"] = {"href": f"{_RAW_ESA}/B04.jp2"}
        assert item_harmonisation(item) is Harmonisation.MIXED

    def test_the_undetermined_set_refuses_rather_than_correcting(self) -> None:
        item = _Item(None, "05.00")
        item.assets = {k: {"href": f"s3://some-new-mirror/{k}"} for k in READ_ASSET_KEYS}
        item.assets["red"] = {"href": f"{_RAW_ESA}/B04.jp2"}
        with pytest.raises(HeterogeneousProducerError):
            dates_exempt_from_correction([item])
