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
    asset_bucket,
    item_harmonisation,
    item_is_in_preferred_location,
    read_asset_sources,
)
from tessera_embeddings.ingest.boa_offset import OffsetDecision, source_decision
from tessera_embeddings.ingest.duplicates import select_preferred_duplicates
from tessera_embeddings.ingest.stac import (
    _declared_baseline,
    _extract_baseline,
    collection_harmonisation,
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


def _decisions(item: object, keys: tuple[str, ...] = REFLECTANCE_ASSET_KEYS) -> set[OffsetDecision]:
    """What is owed to each of an item's reflectance sources.

    The date-scoped equivalents of these assertions are gone: the deleted date-wide check asked
    whether a whole solar day was owed the offset, and per-image correction makes that question
    unnecessary. What survives is the per-source decision, so these tests assert on the set of
    answers an item's own bands produce.
    """
    assets = getattr(item, "assets", {})
    baseline = _declared_baseline(item)
    out: set[OffsetDecision] = set()
    for key in keys:
        asset = assets.get(key)
        href = asset.get("href") if isinstance(asset, dict) else None
        out.add(source_decision(asset_bucket(href) if href else None, baseline, S2_BASELINE_THRESHOLD))
    return out


def _is_exempt(item: object) -> bool:
    """Whether nothing at all is owed to this item — the per-source form of "the date is exempt"."""
    return _decisions(item) == {OffsetDecision.EXEMPT}


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
        assert _decisions(item) == {OffsetDecision.OWED}, "raw pixels are owed the correction"

    def test_a_harmonised_item_over_the_threshold_is_left_alone(self) -> None:
        """The safety direction, and the bug this change could introduce. Element 84's COGs
        already had the offset removed, so correcting them again would shift every value by
        1000 — a regression on the majority of the archive.
        """
        item = _Item(_HARMONISED, "05.09")
        assert _extract_baseline(item) == 509, "the parser reports what the item declares"
        assert extract_baselines([item]) == {"2022-01-07": 509}, "provenance keeps the real baseline"
        assert _is_exempt(item), "but no correction is owed"

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
        assert _decisions(items[0]) == {OffsetDecision.OWED}, "the raw item is owed the offset"
        assert _is_exempt(items[1]), "the harmonised one is not"


class TestProvenanceIsNotOverwritten:
    """The reported baseline reaches the store's `baselines_applied` and must survive."""

    def test_a_harmonised_item_still_reports_its_real_baseline(self) -> None:
        """Encoding the exemption as a baseline of 0 made the store misreport its own vintage —
        worse than the error it avoided, because it cannot be recovered after the fact.
        """
        assert extract_baselines([_Item(_HARMONISED, "05.10")]) == {"2022-01-07": 510}


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
        """No single ITEM-level answer fits a partly-harmonised item, so neither may be reported.

        The correction is decided per asset, which is what gives such an item a right answer band
        by band. This function's only job is to refuse to flatten it to one.
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
        assert _is_exempt(item)

    def test_a_raw_reflectance_band_still_makes_it_mixed(self) -> None:
        """The complement: excluding scl must not weaken the reflectance check itself."""
        item = _Item(_HARMONISED, "05.00")
        item.assets["red"] = {"href": f"{_RAW_ESA}/B04.jp2"}
        assert item_harmonisation(item) is Harmonisation.MIXED


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
        """A native-keyed item is CORRECTED on the collection's answer, never refused for its bucket.

        Its assets sit in an Azure container that `asset_bucket` cannot name and nobody has
        classified, so consulting the bucket would refuse every Planetary Computer date at baseline
        >= 04.00 — a working provider broken by a guard meant for another one. The collection's own
        answer replaces the bucket entirely.

        Asserted on the decision rather than through a load, because these fixtures are plain
        objects and odc's alias table needs real items. The load-level equivalent is in
        `test_boa_offset.py`.
        """
        item = self._pc_item()
        bucket = asset_bucket(item.assets["B02"]["href"])
        assert bucket is None, "the fixture must be unclassifiable, or this proves nothing"
        assert source_decision(bucket, _declared_baseline(item), 400, Harmonisation.RAW) is OffsetDecision.OWED, (
            "the collection's answer decides it"
        )

    def test_the_same_item_is_undecidable_where_the_producer_varies(self) -> None:
        """The complement, or scoping the guard would have disabled it everywhere.

        The very same item, judged for a collection whose producer varies BETWEEN items, has to be
        read from its assets — and they say nothing. Correcting and exempting are wrong by the same
        amount in opposite directions, so it refuses.
        """
        item = self._pc_item()
        bucket = asset_bucket(item.assets["B02"]["href"])
        assert source_decision(bucket, _declared_baseline(item), 400, None) is OffsetDecision.UNDECIDABLE


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


class TestScalingCannotOverflowThePicker:
    """`math.isfinite` was checked BEFORE the multiplication, so a finite but enormous value
    became infinity when scaled and `round()` raised `OverflowError` — aborting the whole batch
    over one item's metadata. Raised on PR #107 after the earlier non-finite fix.
    """

    @pytest.mark.parametrize("raw", ["1e308", "-1e308", "9" * 400])
    def test_a_value_that_overflows_when_scaled_reads_as_unknown(self, raw: str) -> None:
        assert _declared_baseline(_Item(_RAW_ESA, raw)) is None
        assert _extract_baseline(_Item(_RAW_ESA, raw)) == 0

    def test_a_high_but_plausible_version_still_parses(self) -> None:
        """The bound must leave headroom above ESA's 05.x rather than pinning today's value."""
        assert _declared_baseline(_Item(_RAW_ESA, "99.99")) == 9999

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
    """Two entry points, two reasons to select, and one path that had none.

    `s2_roi` selects to build its fallback ladder, and the loader selects because the documented
    `query_stac_items` -> `load_stac_items` workflow passed through neither — an unpruned pair of
    producers for one acquisition reached the correction decision as a genuine conflict.
    `ingest_tile` leaves it to the loader: same gate, same read keys, same collection answer, and
    the loader realigns provenance in the very dict `ingest_tile` returns.
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

    def test_no_collection_can_reach_the_mixed_or_unknown_refusal(self) -> None:
        """The MIXED/UNKNOWN arm of `source_decision` is UNREACHABLE, and this is what says so.

        That arm refuses a source whose producer the COLLECTION could not settle. It cannot fire
        today: the only production caller passes `collection_harmonisation(config)`, and that
        returns `Harmonisation.RAW` or `None` by construction. `MIXED` and `UNKNOWN` come from
        `item_harmonisation`, which feeds duplicate RANKING and never the correction.

        Pinned rather than deleted. Removing the arm would send an unresolved answer to OWED — a
        silent 1000 DN error — the first time a caller supplies one. Asserting it is unreachable
        is what lets the arm stay without being an untested branch: it is closed by the type of
        its only input, and this fails the moment that stops being true.
        """
        reachable = {
            collection_harmonisation(cfg) for provider in PROVIDERS.values() for cfg in provider.collections.values()
        }
        assert reachable <= {Harmonisation.RAW, None}, (
            f"a collection now reports {reachable - {Harmonisation.RAW, None}} to `source_decision`, "
            "which makes its MIXED/UNKNOWN refusal live. It has never run: decide deliberately "
            "whether refusing is right for that collection before this ships."
        )

    def test_there_is_one_definition_of_what_a_load_reads(self) -> None:
        """The driver and the generic path must not define their read set twice."""
        from tessera_embeddings.ingest.s2_roi import _LOADED_EXTRA_BANDS, _read_asset_keys

        config = PROVIDERS["earth-search"].collections["sentinel-2-l2a"]
        assert stac_module._requested_assets(config, _LOADED_EXTRA_BANDS) == _read_asset_keys(
            "earth-search", "sentinel-2-l2a"
        )

    def test_the_read_set_is_empty_where_the_names_are_not_the_asset_keys(self) -> None:
        """Planetary Computer serves `B02`/`SCL` and relies on the loader's alias table, so looking
        the configured names up directly reports every copy incomplete and remote. An empty set
        makes both terms tie rather than mislead.
        """
        from tessera_embeddings.ingest.s2_roi import _read_asset_keys

        config = PROVIDERS["planetary-computer"].collections["sentinel-2-l2a"]
        assert config.band_names_are_asset_keys is False
        assert _read_asset_keys("planetary-computer", "sentinel-2-l2a") == ()

    def test_a_name_based_check_is_only_run_where_names_are_keys(self) -> None:
        """Producer classification, completeness and locality all look an asset up BY NAME, so the
        two flags cannot disagree in the direction that would run such a check blind.
        """
        for provider in ("earth-search", "planetary-computer"):
            config = PROVIDERS[provider].collections["sentinel-2-l2a"]
            if config.harmonisation_varies_by_item:
                assert config.band_names_are_asset_keys, provider

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

    def test_every_refusal_is_gated_on_the_source_being_owed_a_correction(self) -> None:
        """A pre-threshold item must never refuse, whatever else is wrong with it: which producer
        served it, or whether its bands disagree, cannot change a pixel that gets no offset.
        """
        pre_threshold = [
            [_Item(_HARMONISED, "03.00"), _Item(_RAW_ESA, "03.00")],  # mixed producers
            [_Item(None, "03.00")],  # producer undeterminable
            [_Item(_RAW_ESA, "03.00"), _Item(_RAW_ESA, "02.06")],  # differing raw baselines
        ]
        for items in pre_threshold:
            for item in items:
                assert _is_exempt(item), item.assets

    def test_the_one_situation_that_still_refuses_once_a_correction_is_owed(self) -> None:
        """The complement, NARROWED — and the narrowing is the substance of ADR 021.

        Three situations used to refuse at or above the threshold. Two of them were consequences of
        one correction being applied to a whole fused solar day, and per-image correction dissolves
        both rather than isolating them:

        - a harmonised copy beside a raw one owed the offset: each is now decided on its own bucket;
        - raw copies on opposite sides of the threshold: each is now decided on its own baseline.

        Both are asserted as LOADING in `test_boa_offset.py`, over real pixels, because "it no
        longer refuses" is only worth anything if the pixels are also right.

        What still refuses is the case where the evidence itself is missing — a source whose
        producer cannot be determined at all. That is a property of one asset, not of a day, and no
        rearrangement of the load fixes it. Kept here, so deleting the guard still fails a test.
        """
        undetermined = _Item(None, "05.00")
        assert _decisions(undetermined) == {OffsetDecision.UNDECIDABLE}
        # And it is still gated on a correction being owed at all.
        assert _is_exempt(_Item(None, "02.06"))

    def test_provenance_is_never_the_correction_input(self) -> None:
        """Two maps, two purposes: what each item declared, and what to correct at. Collapsing
        them is what made a store report baseline 0 for imagery processed at 05.10.
        """
        harmonised = [_Item(_HARMONISED, "05.10")]
        assert extract_baselines(harmonised) == {"2022-01-07": 510}
        assert _is_exempt(harmonised[0]), "and the correction input owes it nothing"


class TestPruningReachesEveryCollectionThatMakesAnOffsetDecision:
    """The reported P2. Both generic entry points gated pruning on the producer VARYING by item,
    so a collection-wide raw provider was never pruned — and once its items became keyable, two
    reprocessings of one acquisition reached the correction decision together.

    Measured on the live Planetary Computer catalogue over six tiles: 1,000 redundant copies of one
    observation fused, and 12 solar days whose copies straddle the 04.00 threshold and would refuse
    outright for an ambiguity that selecting one copy resolves.
    """

    @staticmethod
    def _pc_pair() -> list[_Item]:
        """One acquisition, two reprocessings straddling the threshold, natively keyed assets."""
        copies = []
        for sequence, baseline in (("1", "04.00"), ("0", "03.00")):
            item = _Item(None, baseline)
            item.id = f"S2A_MSIL2A_20220107T100411_R122_T33TWM_2022010{sequence}T060906"
            item.assets = {k: {"href": f"https://x.blob.core.windows.net/{k}"} for k in ("B02", "SCL")}
            item.properties.update(
                {
                    "s2:mgrs_tile": "33TWM",
                    "s2:sequence": sequence,
                    "s2:datatake_id": f"GS2A_20220107T100411_039301_N{baseline}",
                    "datetime": "2022-01-07T10:04:11.024000Z",
                }
            )
            copies.append(item)
        return copies

    def test_the_unpruned_pair_straddles_the_threshold(self) -> None:
        """The condition that used to refuse the date, asserted so the rest is not vacuous.

        Two copies of one acquisition on opposite sides of 04.00 were fused into a single pixel
        stack and given ONE correction, so the date had no correct answer and was refused. Per
        source, each copy is corrected on its own baseline — but reading both to blend them is
        still waste, which is what pruning is for and what this class tests.
        """
        pair = self._pc_pair()
        decisions = {source_decision(None, _declared_baseline(i), 400, Harmonisation.RAW) for i in pair}
        assert decisions == {OffsetDecision.OWED, OffsetDecision.EXEMPT}, "the pair really does straddle"

    def test_selecting_one_copy_leaves_one_answer(self) -> None:
        """And pruning still reduces it to one copy, which is the point of the gate."""
        kept, _ = select_preferred_duplicates(self._pc_pair(), (), Harmonisation.RAW)
        assert len(kept) == 1
        assert source_decision(None, _declared_baseline(kept[0]), 400, Harmonisation.RAW) is OffsetDecision.OWED

    def test_the_loader_prunes_for_a_collection_wide_raw_provider(self, monkeypatch) -> None:
        loaded: list[str] = []
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )

        def fake_load(items, *a, **k):
            loaded.extend(i.id for i in items)
            return data

        monkeypatch.setattr(stac_module, "_load_from_stac", fake_load)
        stac_module.load_stac_items(self._pc_pair(), "planetary-computer", "sentinel-2-l2a", baselines=None)
        assert len(loaded) == 1, f"the pair reached the loader unpruned: {loaded}"

    def test_the_generic_entry_point_prunes_too(self, monkeypatch) -> None:
        loaded: list[str] = []
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )

        def fake_load(items, *a, **k):
            loaded.extend(i.id for i in items)
            return data

        monkeypatch.setattr(stac_module, "_load_from_stac", fake_load)
        stac_module.ingest_tile(
            provider="planetary-computer",
            collection="sentinel-2-l2a",
            tile_id="33TWM",
            start_date="2022-01-01",
            end_date="2022-01-31",
            mid_longitude=15.0,
            item_provider_fn=lambda **_: self._pc_pair(),
        )
        assert len(loaded) == 1, f"the pair reached the loader unpruned: {loaded}"

    def test_the_read_set_rule_has_one_definition(self) -> None:
        """The driver's read set and the generic path's must not be derived twice."""
        from tessera_embeddings.ingest.s2_roi import _LOADED_EXTRA_BANDS, _read_asset_keys

        for provider in ("earth-search", "planetary-computer"):
            config = PROVIDERS[provider].collections["sentinel-2-l2a"]
            assert stac_module.selection_read_keys(config, _LOADED_EXTRA_BANDS) == _read_asset_keys(
                provider, "sentinel-2-l2a"
            )

    def test_the_gate_stops_at_collections_owed_no_offset(self) -> None:
        """The scope of the change, asserted because it was chosen rather than assumed.

        Removing the gate entirely would reach any collection with duplicates — but Landsat items
        key perfectly well here (`grid:code` is of the form `WRS2-190028`), so it would start
        reducing those too. That is a change nothing in this branch has measured, so it is owed
        separately rather than taken as a side effect of this one.
        """
        for provider, collection in (("earth-search", "landsat-c2-l2"), ("earth-search", "sentinel-1-grd")):
            config = PROVIDERS[provider].collections[collection]
            assert config.requires_baseline_correction is False, f"{provider}/{collection}"


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
        assert _decisions(item) == {OffsetDecision.UNDECIDABLE}

    def test_a_pre_threshold_unlisted_bucket_is_exempt(self) -> None:
        """Gated on the source being owed a correction, like every other refusal."""
        item = _Item(None, "03.00")
        item.assets = {k: {"href": f"s3://some-new-mirror/{k}"} for k in READ_ASSET_KEYS}
        assert _is_exempt(item)


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
        # The unlisted bands are undecidable; the one raw band is simply owed the offset. Per
        # source, those coexist — where one date-wide answer had to cover both and could not.
        assert _decisions(item) == {OffsetDecision.UNDECIDABLE, OffsetDecision.OWED}


class TestMixedNeedsBothKnownProducers:
    """MIXED means both known producers are present, so it must not be reported where one of them
    is only an unclassified bucket. Raised on PR #107.

    A harmonised band beside a band from an unlisted bucket is UNKNOWN, not MIXED. Nothing in such
    an item is known to be raw, so the actionable remedy is to classify the bucket — and
    classifying it as harmonised may remove the refusal altogether. Reporting MIXED sends the
    operator after a disagreement between producers that may not exist.
    """

    @staticmethod
    def _harmonised_plus_unlisted() -> _Item:
        item = _Item(_HARMONISED, "05.00")
        item.assets["red"] = {"href": "s3://some-new-mirror/B04.tif"}
        return item

    def test_harmonised_plus_unlisted_is_undetermined(self) -> None:
        assert item_harmonisation(self._harmonised_plus_unlisted()) is Harmonisation.UNKNOWN

    def test_the_unlisted_band_is_the_only_one_that_refuses(self) -> None:
        """The remedy is to classify the bucket, and only that band is affected.

        This used to assert the wording of a date-wide refusal. The refusal is per source now, so
        what there is to assert is that the harmonised bands are unaffected by their neighbour —
        which is the substance of the same point: nothing here needs the producers loaded
        separately, it needs one bucket classified.
        """
        item = self._harmonised_plus_unlisted()
        assert _decisions(item, ("red",)) == {OffsetDecision.UNDECIDABLE}
        assert _decisions(item, ("blue", "green")) == {OffsetDecision.EXEMPT}

    def test_straddling_two_known_producers_is_still_mixed(self) -> None:
        """The complement, or this would pass with MIXED deleted."""
        item = _Item(_HARMONISED, "05.00")
        item.assets["red"] = {"href": f"{_RAW_ESA}/B04.jp2"}
        assert item_harmonisation(item) is Harmonisation.MIXED
        # And MIXED no longer refuses anything: each band is decided on its own bucket.
        assert _decisions(item, ("red",)) == {OffsetDecision.OWED}
        assert _decisions(item, ("blue",)) == {OffsetDecision.EXEMPT}


class TestProvenanceRealignsWheneverSelectionPruned:
    """Gated on whether selection changed the items, not on whether a fallback survived.

    `select_preferred_duplicates` drops rejected copies that would refuse their date, so a pruned
    tile-date can come back with NO alternates — and gating the realignment on alternates left the
    caller's map describing a copy that is not being loaded. Raised on PR #107.
    """

    @staticmethod
    def _pair() -> list[_Item]:
        """The winner is harmonised; the rejected copy is raw with no readable baseline, so the
        ladder filter removes it and `alternates` comes back empty.
        """
        winner = _Item(_HARMONISED, "05.10")
        winner.id = "S2A_33TWM_20220107_1_L2A"
        winner.properties["s2:sequence"] = "1"
        loser = _Item(_RAW_ESA, None)
        loser.id = "S2A_33TWM_20220107_0_L2A"
        loser.properties["s2:sequence"] = "0"
        for item in (winner, loser):
            item.properties["grid:code"] = "MGRS-33TWM"
            item.properties["datetime"] = "2022-01-07T10:20:31.024000Z"
        return [winner, loser]

    def test_the_fixture_really_yields_no_alternates(self) -> None:
        """Otherwise this would pass through the old gate and prove nothing."""
        kept, alternates = select_preferred_duplicates(self._pair())
        assert [i.id for i in kept] == ["S2A_33TWM_20220107_1_L2A"]
        assert alternates == {}, "the ladder filter must have emptied it"

    def test_the_map_is_realigned_even_with_no_alternates(self, monkeypatch) -> None:
        data = xr.Dataset(
            {"blue": (("time", "y", "x"), np.ones((1, 2, 2), dtype=np.uint16))},
            coords={"time": [np.datetime64("2022-01-07T12:00:00")], "y": [0, 1], "x": [0, 1]},
        )
        monkeypatch.setattr(stac_module, "_load_from_stac", lambda *a, **k: data)
        items = self._pair()
        # What the split workflow hands over: last-wins over the UNPRUNED list, so the rejected
        # copy's absent baseline reads as 0.
        baselines = extract_baselines(items)
        assert baselines == {"2022-01-07": 0}, "the fixture must start describing the rejected copy"

        stac_module.load_stac_items(items, "earth-search", "sentinel-2-l2a", baselines=baselines)
        assert baselines == {"2022-01-07": 510}, "provenance still described a copy not being loaded"
