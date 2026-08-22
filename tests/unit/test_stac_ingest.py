"""Unit tests for ingest/stac.py - STAC ingestion with provider/collection config."""

import numpy as np
import pytest
from pystac_client.item_search import ItemSearch

from tessera_embeddings.config import (
    LANDSAT_C2_BANDS,
    S1_GRD_BANDS,
    S2_L2A_BANDS,
    CollectionConfig,
    STACProvider,
)
from tessera_embeddings.ingest import stac as stac_module
from tessera_embeddings.ingest.stac import (
    _build_stac_query,
    _extract_baseline,
    _filter_existing_dates,
    _get_collection_config,
    _get_provider_config,
    _load_from_stac,
    _loadable_assets,
    _prune_item_dict,
    _query_stac_items,
    correct_boa_dn,
    extract_baselines,
    ingest_tile,
    split_antimeridian_bbox,
    split_query_window,
)


class TestBaselineCorrection:
    """Tests for `correct_boa_dn` — the offset arithmetic, and nothing else.

    The DECISION of which pixels are owed the offset is no longer here. It is made per source, at
    parse time, by `BoaOffsetParser`, and asserted against `source_decision` in
    `test_boa_offset.py`. This class is only the arithmetic that runs once that decision is yes,
    which is why it takes an array and an offset rather than a date and a baseline map.
    """

    def test_subtracts_the_offset(self):
        """The plain case: a bright pixel loses exactly the offset."""
        values = np.array([[1500, 2000], [3000, 4000]], dtype=np.uint16)
        result = correct_boa_dn(values, -1000)
        np.testing.assert_array_equal(result, np.array([[500, 1000], [2000, 3000]], dtype=np.uint16))

    def test_the_input_dtype_survives(self):
        """The store's arrays are unsigned, and a signed result would be cast on the way in."""
        for dtype in (np.uint16, np.int16, np.int32):
            result = correct_boa_dn(np.array([[2000]], dtype=dtype), -1000)
            assert result.dtype == dtype, dtype

    def test_a_dark_pixel_is_corrected_rather_than_left_alone(self):
        """The offset applies to every valid DN, not only to those above it.

        ESA adds it across the whole reflectance range precisely so that negative surface
        reflectance is representable, so leaving DN 1-999 alone left them up to 999 too high.
        """
        result = correct_boa_dn(np.array([[500, 800, 999]], dtype=np.uint16), -1000)
        np.testing.assert_array_equal(result, np.array([[1, 1, 1]], dtype=np.uint16))

    def test_a_valid_dark_pixel_does_not_become_nodata(self):
        """The floor is 1, not 0, and this is the part no amount of reading settles.

        Zero means "no observation". Flooring a real dark measurement there turns it into a gap
        and every downstream mask drops it. Element 84 floors at 1; established by measuring their
        COGs rather than by argument.
        """
        result = correct_boa_dn(np.array([[1000]], dtype=np.uint16), -1000)
        assert result[0, 0] == 1

    def test_no_corrected_pixel_can_wrap(self):
        """A uint16 input cannot underflow, which is what the int32 widening is for.

        Adding a negative Python int to a uint16 array raises under numpy 2, and doing the
        arithmetic in the input dtype would wrap DN 500 to about 64536.
        """
        result = correct_boa_dn(np.zeros((1, 1), dtype=np.uint16), -1000)
        assert result[0, 0] == 1, "even DN 0 floors rather than wrapping"

    def test_the_brightest_codes_lose_exactly_the_offset(self):
        """Nothing saturates at the top of the range either."""
        result = correct_boa_dn(np.array([[65535, 65000]], dtype=np.uint16), -1000)
        np.testing.assert_array_equal(result, np.array([[64535, 64000]], dtype=np.uint16))

    def test_an_empty_array_is_a_no_op(self):
        """A zero-size array, which odc returns on a tolerated read failure, is untouched."""
        result = correct_boa_dn(np.zeros((0, 0), dtype=np.uint16), -1000)
        assert result.shape == (0, 0)
        assert result.dtype == np.uint16


class TestDateFiltering:
    """Tests for filtering STAC items by existing dates."""

    def test_filter_existing_dates_removes_already_processed_items(self, mock_stac_item):
        """Items with dates in existing_dates should be filtered out."""
        items = [
            mock_stac_item("2024-01-01"),
            mock_stac_item("2024-01-06"),
            mock_stac_item("2024-01-11"),
            mock_stac_item("2024-01-16"),
            mock_stac_item("2024-01-21"),
        ]
        existing_dates = {"2024-01-01", "2024-01-11"}

        result = _filter_existing_dates(items, existing_dates)

        assert len(result) == 3
        result_dates = {item.datetime.strftime("%Y-%m-%d") for item in result}
        assert result_dates == {"2024-01-06", "2024-01-16", "2024-01-21"}

    def test_filter_existing_dates_returns_all_when_none_exist(self, mock_stac_item):
        """When existing_dates is empty, all items should be returned."""
        items = [mock_stac_item("2024-01-01"), mock_stac_item("2024-01-06")]

        result = _filter_existing_dates(items, set())

        assert len(result) == 2

    def test_filter_existing_dates_with_empty_items_returns_empty(self):
        """Empty items list should return empty list gracefully."""
        result = _filter_existing_dates([], {"2024-01-01", "2024-01-06"})

        assert result == []
        assert isinstance(result, list)


class TestBaselineExtraction:
    """Tests for parsing baseline version from STAC item properties."""

    def test_extract_baseline_parses_400_correctly(self, mock_stac_item):
        """Baseline "04.00" should parse to integer 400."""
        item = mock_stac_item("2024-01-01", baseline="04.00")
        assert _extract_baseline(item) == 400

    def test_extract_baseline_parses_510_correctly(self, mock_stac_item):
        """Baseline "05.10" should parse to integer 510."""
        item = mock_stac_item("2024-01-01", baseline="05.10")
        assert _extract_baseline(item) == 510

    def test_extract_baseline_returns_zero_when_missing(self, mock_stac_item):
        """Missing baseline property should return 0."""
        item = mock_stac_item("2024-01-01", baseline=None)
        # Remove the baseline property
        del item.properties["s2:processing_baseline"]
        assert _extract_baseline(item) == 0

    def test_extract_baselines_returns_dict_mapping_dates_to_baselines(self, mock_stac_item):
        """extract_baselines should return dict of date -> baseline."""
        items = [
            mock_stac_item("2024-01-01", baseline="04.00"),
            mock_stac_item("2024-01-06", baseline="05.10"),
        ]

        result = extract_baselines(items)

        assert result == {"2024-01-01": 400, "2024-01-06": 510}


class TestProviderConfiguration:
    """Tests for STAC provider and collection configuration."""

    @pytest.mark.parametrize(
        "provider_name, expected_catalog_url",
        [
            ("earth-search", "https://earth-search.aws.element84.com/v1"),
            ("planetary-computer", "https://planetarycomputer.microsoft.com/api/stac/v1"),
        ],
    )
    def test_get_provider_config_returns_expected_provider(self, provider_name, expected_catalog_url):
        """Known providers resolve to the correct STACProvider catalog + collections."""
        provider = _get_provider_config(provider_name)

        assert isinstance(provider, STACProvider)
        assert provider.catalog_url == expected_catalog_url
        assert "sentinel-2-l2a" in provider.collections

    def test_get_provider_config_raises_on_unknown_provider(self):
        """Unknown provider should raise ValueError with available options."""
        with pytest.raises(ValueError, match="Unknown provider"):
            _get_provider_config("nonexistent-provider")

    @pytest.mark.parametrize(
        "provider_name,collection_alias,expected_bands",
        [
            ("earth-search", "sentinel-2-l2a", S2_L2A_BANDS),
            ("earth-search", "sentinel-1-grd", S1_GRD_BANDS),
            ("earth-search", "landsat-c2-l2", LANDSAT_C2_BANDS),
            ("planetary-computer", "sentinel-2-l2a", S2_L2A_BANDS),
        ],
    )
    def test_get_collection_config_returns_correct_bands(self, provider_name, collection_alias, expected_bands):
        """Collection configs should return appropriate bands for each dataset."""
        config = _get_collection_config(provider_name, collection_alias)

        assert config.bands == expected_bands

    @pytest.mark.parametrize(
        "provider_name,collection_alias,expected_correction",
        [
            # Earth Search S2 COGs are already BOA-corrected; its S1 threshold
            # constant is None — so both are False.
            ("earth-search", "sentinel-2-l2a", True),
            ("earth-search", "sentinel-1-grd", False),
            ("earth-search", "landsat-c2-l2", False),
            # Planetary Computer S2 sets a baseline_threshold (=400) → True.
            # This proves the flag can be True, not just hardcoded False.
            ("planetary-computer", "sentinel-2-l2a", True),
            ("planetary-computer", "landsat-c2-l2", False),
        ],
    )
    def test_baseline_correction_flag_set_correctly(self, provider_name, collection_alias, expected_correction):
        """requires_baseline_correction reflects whether baseline_threshold is set."""
        config = _get_collection_config(provider_name, collection_alias)

        assert config.requires_baseline_correction == expected_correction

    def test_get_collection_config_raises_on_unknown_collection(self):
        """Unknown collection should raise ValueError with available options."""
        with pytest.raises(ValueError, match="Unknown collection"):
            _get_collection_config("earth-search", "nonexistent-collection")


class TestSTACQueryBuilder:
    """Tests for STAC query parameter construction."""

    @pytest.mark.parametrize(
        "provider,collection,tile_id,expected_property,expected_value",
        [
            ("earth-search", "sentinel-2-l2a", "33UUP", "grid:code", "MGRS-33UUP"),
            ("earth-search", "landsat-c2-l2", "042", "landsat:wrs_path", "042"),
            ("planetary-computer", "sentinel-2-l2a", "33UUP", "s2:mgrs_tile", "33UUP"),
        ],
    )
    def test_build_stac_query_uses_correct_tile_property(
        self, provider, collection, tile_id, expected_property, expected_value
    ):
        """Query builder produces correct tile property and prefix per provider/collection."""
        config = _get_collection_config(provider, collection)
        query = _build_stac_query(config, tile_id, "2024-01-01", "2024-01-31")

        assert query["collections"] == [config.collection_id]
        assert query["datetime"] == "2024-01-01/2024-01-31"
        assert expected_property in query["query"]
        assert query["query"][expected_property]["eq"] == expected_value


class TestIngestTile:
    """Tests for the high-level ingest_tile function."""

    def test_ingest_tile_returns_none_when_all_dates_exist(self, mock_stac_item, sample_reflectance_data, monkeypatch):
        """When all queried dates already exist, returns (None, baselines)."""
        # Mock the STAC query to return items we control
        items = [
            mock_stac_item("2024-01-01", baseline="04.00"),
            mock_stac_item("2024-01-06", baseline="04.00"),
        ]

        def mock_query(*args, **kwargs):
            return items

        monkeypatch.setattr("tessera_embeddings.ingest.stac._query_stac_items", mock_query)

        # All dates already exist
        existing_dates = {"2024-01-01", "2024-01-06"}

        result_data, result_baselines = ingest_tile(
            provider="earth-search",
            collection="sentinel-2-l2a",
            tile_id="33UUP",
            start_date="2024-01-01",
            end_date="2024-01-10",
            existing_dates=existing_dates,
            # 33UUP is UTM zone 33 (central meridian 15E). Required now: solar-day grouping
            # refuses a call carrying no geometry rather than stamping UTC dates silently.
            mid_longitude=15.0,
        )

        assert result_data is None
        assert result_baselines == {"2024-01-01": 400, "2024-01-06": 400}

    def test_ingest_tile_applies_baseline_correction(self, mock_stac_item, sample_reflectance_data, monkeypatch):
        """Data from baseline >= 400 should have correction applied."""
        # Mock STAC query
        items = [mock_stac_item("2024-01-01", baseline="04.00")]

        def mock_query(*args, **kwargs):
            return items

        monkeypatch.setattr("tessera_embeddings.ingest.stac._query_stac_items", mock_query)

        # Mock load_from_stac to return controlled data
        # Values of 2000 should stay same
        raw_data = sample_reflectance_data(["2024-01-01"], height=32, width=32, seed=42)
        # Set known values for verification (use "blue" - common name for B02)
        raw_data["blue"].values[:] = 2000
        captured: dict[str, object] = {}

        def mock_load(*args, **kwargs):
            captured["driver"] = kwargs.get("driver")
            return raw_data

        monkeypatch.setattr("tessera_embeddings.ingest.stac._load_from_stac", mock_load)

        result_data, result_baselines = ingest_tile(
            provider="earth-search",
            collection="sentinel-2-l2a",
            tile_id="33UUP",
            start_date="2024-01-01",
            end_date="2024-01-10",
            existing_dates=None,
            # 33UUP is UTM zone 33 (central meridian 15E). Required now: solar-day grouping
            # refuses a call carrying no geometry rather than stamping UTC dates silently.
            mid_longitude=15.0,
        )

        assert result_data is not None
        # The DRIVER is the assertion now, not the pixel. The offset is removed inside the read,
        # per image, before resampled sources are fused — so a test that replaces the loader can
        # only check that the correction was handed to it. What the driver then does to real pixels
        # is asserted in `TestTheOffsetIsRemovedPerImage`, which drives a real `odc.stac.load`.
        assert captured["driver"] is not None, "a collection owed the offset must load through the corrector"
        assert result_baselines == {"2024-01-01": 400}, "provenance still records what each item declared"

    def test_ingest_tile_leaves_a_collection_owed_no_offset_alone(
        self, mock_stac_item, sample_reflectance_data, monkeypatch
    ):
        """A collection with no correction threshold gets odc's own driver, not ours.

        Sentinel-1 and Landsat carry no BOA offset, so wrapping their reads would add a decision
        where there is no question — and `requires_baseline_correction` is the gate that keeps the
        whole mechanism off those paths, including OPERA's, which shares this entry point.
        """
        items = [mock_stac_item("2024-01-01", baseline="04.00")]
        monkeypatch.setattr("tessera_embeddings.ingest.stac._query_stac_items", lambda *a, **k: items)
        captured: dict[str, object] = {}

        def mock_load(*args, **kwargs):
            captured["driver"] = kwargs.get("driver")
            return sample_reflectance_data(["2024-01-01"], height=32, width=32, seed=42)

        monkeypatch.setattr("tessera_embeddings.ingest.stac._load_from_stac", mock_load)
        ingest_tile(
            provider="earth-search",
            collection="landsat-c2-l2",
            tile_id="33UUP",
            start_date="2024-01-01",
            end_date="2024-01-10",
            existing_dates=None,
            mid_longitude=15.0,
        )
        assert captured["driver"] is None

    def test_ingest_tile_does_not_correct_extra_bands(self, mock_stac_item, sample_reflectance_data, monkeypatch):
        """Extra bands (like SCL) should NOT have baseline correction applied."""
        items = [mock_stac_item("2024-01-01", baseline="04.00")]

        def mock_query(*args, **kwargs):
            return items

        monkeypatch.setattr("tessera_embeddings.ingest.stac._query_stac_items", mock_query)

        # Create data with an extra "scl" band
        raw_data = sample_reflectance_data(["2024-01-01"], height=32, width=32, seed=42)
        raw_data["blue"].values[:] = 2000
        scl_values = np.full((1, 32, 32), 4, dtype=np.uint8)  # 4 = vegetation
        raw_data["scl"] = (["time", "northing", "easting"], scl_values)
        captured: dict[str, object] = {}

        def mock_load(*args, **kwargs):
            driver = kwargs.get("driver")
            captured["reflectance_assets"] = driver.md_parser.reflectance_assets
            return raw_data

        monkeypatch.setattr("tessera_embeddings.ingest.stac._load_from_stac", mock_load)

        result_data, _ = ingest_tile(
            provider="earth-search",
            collection="sentinel-2-l2a",
            tile_id="33UUP",
            start_date="2024-01-01",
            end_date="2024-01-10",
            existing_dates=None,
            extra_bands=["scl"],
            # 33UUP is UTM zone 33 (central meridian 15E). Required now: solar-day grouping
            # refuses a call carrying no geometry rather than stamping UTC dates silently.
            mid_longitude=15.0,
        )

        assert result_data is not None
        # `scl` is excluded STRUCTURALLY now, not by a list of band names the corrector was told to
        # skip. The offset decision is stamped onto reflectance sources only, so the scene
        # classification layer carries no decision at all and the reader has nothing to apply.
        # Asserted on the resolved asset keys, because that is the thing that could go wrong.
        assert "scl" not in captured["reflectance_assets"]
        assert set(captured["reflectance_assets"]) == set(S2_L2A_BANDS)
        assert result_data["scl"].values[0, 0, 0] == 4

    def test_ingest_tile_applies_post_load_fn(self, mock_stac_item, sample_reflectance_data, monkeypatch):
        """post_load_fn should be applied after loading and baseline correction."""
        items = [mock_stac_item("2024-01-01", baseline="02.00")]  # No baseline correction

        def mock_query(*args, **kwargs):
            return items

        monkeypatch.setattr("tessera_embeddings.ingest.stac._query_stac_items", mock_query)

        raw_data = sample_reflectance_data(["2024-01-01"], height=32, width=32, seed=42)
        raw_data["blue"].values[:] = 100

        def mock_load(*args, **kwargs):
            return raw_data

        monkeypatch.setattr("tessera_embeddings.ingest.stac._load_from_stac", mock_load)

        # Post-load function that doubles all blue values
        def double_blue(ds):
            return ds.assign(blue=ds["blue"] * 2)

        result_data, _ = ingest_tile(
            provider="earth-search",
            collection="sentinel-2-l2a",
            tile_id="33UUP",
            start_date="2024-01-01",
            end_date="2024-01-10",
            existing_dates=None,
            post_load_fn=double_blue,
            # 33UUP is UTM zone 33 (central meridian 15E). Required now: solar-day grouping
            # refuses a call carrying no geometry rather than stamping UTC dates silently.
            mid_longitude=15.0,
        )

        assert result_data is not None
        assert result_data["blue"].values[0, 0, 0] == 200

    def test_ingest_tile_applies_item_filter_fn(self, mock_stac_item, sample_reflectance_data, monkeypatch):
        """item_filter_fn should filter items before date filtering and loading."""
        items = [
            mock_stac_item("2024-01-01", baseline="04.00"),
            mock_stac_item("2024-01-06", baseline="04.00"),
            mock_stac_item("2024-01-11", baseline="04.00"),
        ]

        def mock_query(*args, **kwargs):
            return items

        monkeypatch.setattr("tessera_embeddings.ingest.stac._query_stac_items", mock_query)

        raw_data = sample_reflectance_data(["2024-01-01"], height=32, width=32, seed=42)

        def mock_load(*args, **kwargs):
            return raw_data

        monkeypatch.setattr("tessera_embeddings.ingest.stac._load_from_stac", mock_load)

        # Filter that keeps only the first item
        def keep_first(items_list):
            return items_list[:1]

        result_data, result_baselines = ingest_tile(
            provider="earth-search",
            collection="sentinel-2-l2a",
            tile_id="33UUP",
            start_date="2024-01-01",
            end_date="2024-01-15",
            existing_dates=None,
            item_filter_fn=keep_first,
            # 33UUP is UTM zone 33 (central meridian 15E). Required now: solar-day grouping
            # refuses a call carrying no geometry rather than stamping UTC dates silently.
            mid_longitude=15.0,
        )

        assert result_data is not None
        # Baselines should only contain the filtered item's date
        # (baselines are extracted AFTER item_filter_fn)
        assert "2024-01-01" in result_baselines

    def test_ingest_tile_item_filter_fn_returns_none_when_empty(self, mock_stac_item, monkeypatch):
        """When item_filter_fn removes all items, returns (None, {})."""
        items = [mock_stac_item("2024-01-01", baseline="04.00")]

        def mock_query(*args, **kwargs):
            return items

        monkeypatch.setattr("tessera_embeddings.ingest.stac._query_stac_items", mock_query)

        result_data, result_baselines = ingest_tile(
            provider="earth-search",
            collection="sentinel-2-l2a",
            tile_id="33UUP",
            start_date="2024-01-01",
            end_date="2024-01-10",
            existing_dates=None,
            item_filter_fn=lambda items: [],  # Remove all items
            # 33UUP is UTM zone 33 (central meridian 15E). Required now: solar-day grouping
            # refuses a call carrying no geometry rather than stamping UTC dates silently.
            mid_longitude=15.0,
        )

        assert result_data is None
        assert result_baselines == {}


class TestSolarDayPainterOrdering:
    """Which of two overlapping same-day scenes wins a pixel, and it is decided by ORDER.

    `ingest_tile` loads with `preserve_original_order=True` and `groupby="solar_day"`, and odc.stac's
    painter keeps the LAST item written. So the list handed to the loader must end with the clearest
    scene. It used to be sorted clearest-FIRST, with a comment saying that was for SCL mosaicking, so
    the cloudiest scene won every overlap — silently, because the output has the right shape and the
    right dates. The campaign's own S2 path reverses the order and says why; this generic API
    disagreed with it.
    """

    def test_the_clearest_scene_of_a_solar_day_is_loaded_last(self, mock_stac_item, monkeypatch):
        seen: dict = {}

        def mock_query(*args, **kwargs):
            # Deliberately delivered clearest-first, which is what query_stac_items produces.
            return [
                mock_stac_item("2024-01-01T10:00:00", cloud_cover=2.0),
                mock_stac_item("2024-01-01T10:05:00", cloud_cover=55.0),
                mock_stac_item("2024-01-01T10:10:00", cloud_cover=90.0),
            ]

        monkeypatch.setattr("tessera_embeddings.ingest.stac._query_stac_items", mock_query)

        def stamp_noon(items, mid_longitude=None):
            """What the real function does to `.datetime`, without the longitude offset.

            A bare pass-through is no longer a faithful stub: every date derived downstream goes
            through `solar_day_of`, which refuses an item that has not been stamped. The offset is
            what this test is isolating away from, not the stamping.
            """
            for item in items:
                item.datetime = item.datetime.replace(hour=12, minute=0, second=0, microsecond=0)
            return items

        monkeypatch.setattr("tessera_embeddings.ingest.stac.normalize_to_solar_day", stamp_noon)

        def fake_load(items, **kwargs):
            seen["order"] = [it.properties["eo:cloud_cover"] for it in items]
            seen["preserve"] = kwargs.get("preserve_original_order")
            seen["groupby"] = kwargs.get("groupby")
            raise _StopLoadError

        import odc.stac

        monkeypatch.setattr(odc.stac, "load", fake_load)

        with pytest.raises(_StopLoadError):
            ingest_tile(
                provider="earth-search",
                collection="sentinel-2-l2a",
                tile_id="33UUP",
                start_date="2024-01-01",
                end_date="2024-01-02",
                # 33UUP is UTM zone 33 (central meridian 15E). Required now: solar-day grouping
                # refuses a call carrying no geometry rather than stamping UTC dates silently.
                mid_longitude=15.0,
            )

        assert seen["preserve"] is True, "the premise: order decides, so order must be correct"
        assert seen["groupby"] == "solar_day"
        assert seen["order"] == [90.0, 55.0, 2.0], "cloudiest first, so the clearest paints last"


class _StopLoadError(Exception):
    """Cuts `ingest_tile` off at the load call — the ordering is what this asserts."""


class TestBuildStacQueryBboxFallback:
    """Tests for _build_stac_query bbox fallback behavior."""

    def test_uses_tile_property_when_available(self):
        """Collections with tile_id_property should use property-based query."""
        config = _get_collection_config("earth-search", "sentinel-2-l2a")
        query = _build_stac_query(config, "33UUP", "2024-01-01", "2024-01-31")

        assert "query" in query
        assert "bbox" not in query

    def test_falls_back_to_bbox_when_no_tile_property(self):
        """Collections with tile_id_property=None should use bbox query."""
        config = CollectionConfig(
            collection_id="OPERA_L2_RTC-S1_V1_1",
            bands=["0_VV", "0_VH"],
            resolution=10,
            tile_id_property=None,
        )
        bbox = (11.0, 48.0, 12.0, 49.0)
        query = _build_stac_query(config, "33UUP", "2024-01-01", "2024-01-31", bbox=bbox)

        assert "bbox" in query
        assert query["bbox"] == bbox

    def test_raises_when_no_tile_property_and_no_bbox(self):
        """Collections with tile_id_property=None and no bbox should raise."""
        config = CollectionConfig(
            collection_id="OPERA_L2_RTC-S1_V1_1",
            bands=["0_VV", "0_VH"],
            resolution=10,
            tile_id_property=None,
        )

        with pytest.raises(ValueError, match="no tile_id_property"):
            _build_stac_query(config, "33UUP", "2024-01-01", "2024-01-31")

    def test_tile_id_none_with_bbox_uses_bbox_query(self):
        """tile_id=None with bbox should use bbox-based query, even when
        collection has a tile_id_property (ROI-based ingestion).
        """
        config = _get_collection_config("earth-search", "sentinel-2-l2a")
        bbox = (-95.0, 45.0, -94.0, 46.0)
        query = _build_stac_query(config, None, "2024-01-01", "2024-01-31", bbox=bbox)

        assert "bbox" in query
        assert query["bbox"] == bbox

    def test_tile_id_none_without_bbox_raises(self):
        """tile_id=None without bbox should raise ValueError."""
        config = _get_collection_config("earth-search", "sentinel-2-l2a")

        with pytest.raises(ValueError, match="no bbox was provided"):
            _build_stac_query(config, None, "2024-01-01", "2024-01-31")

    def test_tile_id_with_tile_property_unchanged(self):
        """Existing behavior: tile_id + tile_id_property → property-based query."""
        config = _get_collection_config("earth-search", "sentinel-2-l2a")
        query = _build_stac_query(config, "33UUP", "2024-01-01", "2024-01-31")

        assert "query" in query
        assert "bbox" not in query
        assert query["query"]["grid:code"]["eq"] == "MGRS-33UUP"


class TestAntimeridianBboxSplit:
    """Zones 01 and 60 emit a west>east bbox; the catalog search must not see it.

    The land mask writes the GeoJSON/STAC crossing convention (west > east) for the
    UTM zones that snap just past +/-180, because plain min/max there would produce a
    box spanning nearly the globe. Sending that tuple to a catalog is a bet on the
    server reading it as a crossing; the two ways that bet loses are opposite — no
    dates at all, or an unbounded item set — so the query is split instead.
    """

    def test_ordinary_bbox_is_passed_through_untouched(self):
        bbox = (-95.0, 45.0, -94.0, 46.0)
        assert split_antimeridian_bbox(bbox) == [bbox]

    def test_absent_bbox_still_yields_one_query(self):
        # Tile-id queries carry no bbox; callers loop over the result unconditionally.
        assert split_antimeridian_bbox(None) == [None]

    def test_crossing_bbox_splits_at_the_antimeridian(self):
        assert split_antimeridian_bbox((179.5, -18.0, -179.5, -16.0)) == [
            (179.5, -18.0, 180.0, -16.0),
            (-180.0, -18.0, -179.5, -16.0),
        ]

    def test_query_runs_both_halves_and_dedupes_straddling_items(self, monkeypatch):
        """A granule crossing +/-180 is returned by BOTH searches — load it once."""
        searched: list = []

        def _raw(id_: str) -> dict:
            """A minimal valid STAC item dict.

            Dicts rather than stubs because the query pages as dicts and hydrates AFTER
            pruning, so this also covers that a pruned item is still something
            ``Item.from_dict`` accepts.
            """
            return {
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": id_,
                "geometry": {"type": "Point", "coordinates": [179.9, -17.0]},
                "bbox": [179.9, -17.0, 179.9, -17.0],
                "properties": {"datetime": "2024-01-05T00:00:00Z"},
                "links": [{"rel": "self", "href": f"https://example/{id_}"}],
                "assets": {"blue": {"href": f"s3://b/{id_}-blue.tif"}},
            }

        class _Search:
            def __init__(self, ids):
                self._ids = ids

            def pages_as_dicts(self):
                # The query walks PAGES, so a failure can name the page ordinal it died
                # on. `items_as_dicts` is defined upstream as exactly this, flattened over
                # each page's "features", so the two differ only in attribution.
                yield {"features": [_raw(i) for i in self._ids]}

        class _Client:
            def search(self, **kw):
                searched.append(kw["bbox"])
                # The straddling granule "S" comes back from each half.
                return _Search(["S", "E"] if kw["bbox"][2] == 180.0 else ["S", "W"])

        monkeypatch.setattr("tessera_embeddings.ingest.stac.Client.open", lambda *a, **k: _Client())
        config = _get_collection_config("earth-search", "sentinel-2-l2a")
        provider = _get_provider_config("earth-search")
        items = _query_stac_items(
            provider, config, None, "2024-01-01", "2024-01-31", bbox=(179.5, -18.0, -179.5, -16.0)
        )

        assert searched == [(179.5, -18.0, 180.0, -16.0), (-180.0, -18.0, -179.5, -16.0)]
        assert [i.id for i in items] == ["S", "E", "W"]


class TestItemPruning:
    """Tests for _prune_item_dict / _loadable_assets.

    Retained items dominate the ingest driver's memory, and pruning them is only safe if
    it cannot drop metadata the loader needs. These tests pin the deny-list contract: kept
    assets and all other item fields survive byte-for-byte.
    """

    @staticmethod
    def _item(assets: dict | None = None) -> dict:
        return {
            "type": "Feature",
            "stac_version": "1.0.0",
            "stac_extensions": ["https://stac-extensions.github.io/projection/v1.1.0/schema.json"],
            "id": "S2A_TEST_20240603",
            "collection": "sentinel-2-l2a",
            "bbox": [24.0, 36.0, 25.0, 37.0],
            "geometry": {"type": "Polygon", "coordinates": [[[24, 36], [25, 36], [25, 37], [24, 36]]]},
            "properties": {
                "datetime": "2024-06-03T10:04:10.274000Z",
                # The CRS lives here for this collection. An allow-list built expecting
                # `proj:epsg` silently dropped it, which is why pruning is a deny-list.
                "proj:code": "EPSG:32634",
                "s2:processing_baseline": "05.10",
                "eo:cloud_cover": 4.66,
            },
            "assets": assets
            if assets is not None
            else {
                "blue": {
                    "href": "s3://b/blue.tif",
                    "proj:shape": [10980, 10980],
                    "raster:bands": [{"nodata": 0, "scale": 1}],
                },
                "scl": {"href": "s3://b/scl.tif", "proj:shape": [5490, 5490]},
                "visual": {"href": "s3://b/visual.tif"},
                "thumbnail": {"href": "s3://b/thumb.jpg"},
                "granule_metadata": {"href": "s3://b/meta.xml"},
            },
            "links": [{"rel": "self", "href": "https://example/item"}],
        }

    def test_loadable_assets_is_bands_plus_scl(self):
        cfg = CollectionConfig(collection_id="c", bands=["blue", "green"], resolution=10, has_scl=True)
        assert _loadable_assets(cfg) == frozenset({"blue", "green", "scl"})

    def test_loadable_assets_omits_scl_when_absent(self):
        cfg = CollectionConfig(collection_id="c", bands=["vv", "vh"], resolution=10, has_scl=False)
        assert _loadable_assets(cfg) == frozenset({"vv", "vh"})

    def test_drops_unread_assets_and_links(self):
        pruned = _prune_item_dict(self._item(), frozenset({"blue", "scl"}))
        assert set(pruned["assets"]) == {"blue", "scl"}
        assert pruned["links"] == []

    def test_kept_assets_and_other_fields_survive_verbatim(self):
        """The deny-list contract: nothing inside a kept asset, and no other field, changes."""
        src = self._item()
        pruned = _prune_item_dict(src, frozenset({"blue", "scl"}))
        assert pruned["assets"]["blue"] == src["assets"]["blue"]
        assert pruned["assets"]["scl"] == src["assets"]["scl"]
        for key in ("id", "collection", "type", "stac_version", "stac_extensions", "bbox", "geometry", "properties"):
            assert pruned[key] == src[key], key

    def test_a_partial_asset_match_retains_the_whole_item(self):
        """The alias case: matching SOME requested names is not enough to prune safely.

        Ask for `blue` and `scl` against an item carrying `B02` and `scl` — one name matches,
        so the old `if not kept` guard let the prune run, keeping `scl` and deleting `B02`.
        The loader then asks for `blue`, whose only source has just been removed, and the
        load fails having passed every check before it. Nothing here can see the alias table
        that maps a band name to an asset key, so an absent name and an aliased one look
        identical — and the costs are not symmetric: retaining spends memory, wrongly
        pruning loses the band.
        """
        item = self._item()
        item["assets"]["B02"] = item["assets"].pop("blue")  # native key, not the requested name

        pruned = _prune_item_dict(item, frozenset({"blue", "scl"}))

        assert pruned is item, "a partial match must leave the item untouched"
        assert "B02" in pruned["assets"], "the asset backing the requested band must survive"

    def test_a_complete_match_still_prunes(self):
        """The guard must not have disabled pruning in the ordinary case, which is the point
        of the function — 35 assets down to the handful the loader reads.
        """
        pruned = _prune_item_dict(self._item(), frozenset({"blue", "scl"}))
        assert set(pruned["assets"]) == {"blue", "scl"}

    def test_does_not_mutate_the_input(self):
        src = self._item()
        _prune_item_dict(src, frozenset({"blue"}))
        assert set(src["assets"]) == {"blue", "scl", "visual", "thumbnail", "granule_metadata"}
        assert src["links"] != []

    def test_unrecognised_asset_names_leave_the_item_untouched(self):
        """A collection whose assets are named differently must keep its bands, not lose them."""
        src = self._item(assets={"B02": {"href": "s3://b/B02.tif"}})
        assert _prune_item_dict(src, frozenset({"blue", "scl"})) is src

    def test_item_without_assets_is_returned_unchanged(self):
        src = self._item(assets={})
        assert _prune_item_dict(src, frozenset({"blue"})) is src

    def test_pruned_dict_still_builds_a_pystac_item(self):
        """The pruned form is hydrated immediately, so it must remain a valid STAC Item."""
        from pystac import Item

        item = Item.from_dict(_prune_item_dict(self._item(), frozenset({"blue", "scl"})))
        assert item.id == "S2A_TEST_20240603"
        assert set(item.assets) == {"blue", "scl"}
        assert item.properties["proj:code"] == "EPSG:32634"
        assert str(item.datetime)[:10] == "2024-06-03"


def test_requested_extra_bands_survive_item_pruning(monkeypatch):
    """Pruning runs at QUERY time, before odc.stac.load resolves bands.

    An asset dropped there cannot be loaded later, so a caller asking for a QA or
    visualisation band would get a missing-band error instead of the band —
    `ingest_tile` and `load_stac_items` both still document that option.
    """
    config = _get_collection_config("earth-search", "sentinel-2-l2a")
    assert "aot" not in _loadable_assets(config)
    assert "aot" in _loadable_assets(config, ["aot"])
    # The collection's own bands are never dropped by asking for an extra one.
    assert set(config.bands) <= _loadable_assets(config, ["aot"])

    # ...and the request has to REACH the pruning, which happens two calls below
    # `ingest_tile`. Keeping only the `_loadable_assets` half of this correct still
    # loses the band, because the query never hears about it.
    seen: dict = {}

    def capture(*_a, extra_bands=None, **_k):
        seen["extra_bands"] = extra_bands
        return []

    monkeypatch.setattr("tessera_embeddings.ingest.stac._query_stac_items", capture)
    ingest_tile(
        provider="earth-search",
        collection="sentinel-2-l2a",
        tile_id="33UUP",
        start_date="2024-01-01",
        end_date="2024-01-10",
        extra_bands=["aot"],
        # 33UUP is UTM zone 33 (central meridian 15E). Required now: solar-day grouping
        # refuses a call carrying no geometry rather than stamping UTC dates silently.
        mid_longitude=15.0,
    )
    assert seen["extra_bands"] == ["aot"]


def test_the_loader_refuses_a_grouping_it_cannot_honour():
    """`groupby="time"` is a lie by the time items reach the loader.

    `query_stac_items` stamps every item to noon of its solar day — the package's one
    application of the offset — so no exact acquisition timestamp survives to group on.
    Grouping by "time" would collapse a day's separate acquisitions exactly as
    "solar_day" does while reporting a different convention, which is worse than
    refusing: the caller would believe it had exact timestamps.
    """
    config = _get_collection_config("earth-search", "sentinel-2-l2a")
    with pytest.raises(ValueError, match="groupby='time' is not supported"):
        _load_from_stac([object()], config, groupby="time")


class _FakeCatalogue:
    """A catalogue that honours a date window, pages, reports its match count, and counts serves.

    Built as an ORACLE rather than a stub: the same fake, asked for one window or for
    several covering the same days, must hand back the same items. A stub replaying a
    fixed page list could not tell those two apart, so it could not fail.
    """

    def __init__(self, per_day: dict[str, list[str]], page_size: int):
        self.per_day = per_day
        self.page_size = page_size
        self.windows: list[str] = []
        self.served_by_window: dict[str, set[str]] = {}

    def raw(self, item_id: str, day: str) -> dict:
        return {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": item_id,
            "geometry": {"type": "Point", "coordinates": [5.0, 45.0]},
            "bbox": [5.0, 45.0, 5.0, 45.0],
            "properties": {"datetime": f"{day}T10:00:00Z"},
            "links": [],
            "assets": {"blue": {"href": f"s3://b/{item_id}.tif"}},
        }

    def search(self, **kw):
        start, end = kw["datetime"].split("/")
        self.windows.append(kw["datetime"])
        matched = [
            (item_id, day) for day, ids in sorted(self.per_day.items()) if start <= day <= end for item_id in ids
        ]
        return _FakeSearch(self, kw["datetime"], matched)


class _FakeSearch:
    def __init__(self, catalogue: _FakeCatalogue, window: str, matched: list[tuple[str, str]]):
        self.catalogue = catalogue
        self.window = window
        self.matched = matched

    def pages_as_dicts(self):
        size = self.catalogue.page_size
        for offset in range(0, max(len(self.matched), 1), size):
            chunk = self.matched[offset : offset + size]
            served = self.catalogue.served_by_window.setdefault(self.window, set())
            served.update(item_id for item_id, _ in chunk)
            yield {
                "numberMatched": len(self.matched),
                "features": [self.catalogue.raw(item_id, day) for item_id, day in chunk],
            }


def _catalogue_over(days: int, per_day_count: int, page_size: int):
    per_day = {f"2024-03-{day:02d}": [f"S2_{day:02d}_{n}" for n in range(per_day_count)] for day in range(1, days + 1)}
    every_id = {item_id for ids in per_day.values() for item_id in ids}
    return _FakeCatalogue(per_day, page_size), every_id


class TestADeepWindowIsQueriedAsShorterOnes:
    """A window the catalogue will not page to the end of is re-cut, not failed.

    Earth Search refuses certain individual page requests deterministically, as a function
    of the request rather than of how deep the walk has got, so retrying the same window
    cannot succeed. The only thing re-cutting may change is how many searches run — never
    which items come back, because this module is inside the mosaic content fingerprint.
    """

    @staticmethod
    def _run(catalogue, monkeypatch, ceiling, start="2024-03-01", end="2024-03-30"):
        provider = _get_provider_config("earth-search")
        collection = _get_collection_config("earth-search", "sentinel-2-l2a")
        monkeypatch.setattr(stac_module, "_MAX_QUERY_ITEMS", ceiling)
        monkeypatch.setattr(stac_module.Client, "open", lambda *a, **k: catalogue)
        return _query_stac_items(provider, collection, None, start, end, bbox=(4.0, 44.0, 6.0, 46.0))

    def test_the_item_set_is_what_an_unsliced_query_would_have_returned(self, monkeypatch):
        """The comparison that matters: sliced against unsliced, one fake catalogue."""
        page = _get_provider_config("earth-search").max_page_size
        whole, every_id = _catalogue_over(days=30, per_day_count=20, page_size=page)
        sliced, _ = _catalogue_over(days=30, per_day_count=20, page_size=page)

        unsliced_items = self._run(whole, monkeypatch, ceiling=10_000_000)
        sliced_items = self._run(sliced, monkeypatch, ceiling=250)

        assert len(whole.windows) == 1, "the control must NOT have been sliced, or it proves nothing"
        assert len(sliced.windows) > 1, "the subject must have been sliced, or it proves nothing"
        assert {i.id for i in sliced_items} == {i.id for i in unsliced_items} == every_id
        assert len(sliced_items) == len(every_id), "a re-partition must not duplicate an item"

    def test_an_item_on_a_boundary_day_shared_by_two_windows_is_returned_once(self, monkeypatch):
        """Adjacent windows share their boundary day on purpose; the dedupe keeps it single.

        The shorter windows are identified by EXCLUDING the input window, so a re-serve by
        the abandoned parent walk cannot be mistaken for the sibling overlap this is about.
        """
        page = _get_provider_config("earth-search").max_page_size
        catalogue, every_id = _catalogue_over(days=8, per_day_count=5, page_size=page)

        items = self._run(catalogue, monkeypatch, ceiling=20, start="2024-03-01", end="2024-03-08")

        shorter = [ids for w, ids in catalogue.served_by_window.items() if w != "2024-03-01/2024-03-08"]
        overlaps = [a & b for i, a in enumerate(shorter) for b in shorter[i + 1 :] if a & b]
        assert overlaps, f"no two shorter windows shared an item, so the overlap was never exercised: {shorter}"
        assert len(items) == len(every_id), "an item two windows both matched was returned twice"
        assert {i.id for i in items} == every_id


class TestSplitQueryWindow:
    """The re-partition itself: it must cover the input, and it must terminate."""

    @staticmethod
    def _sent(start: str, end: str) -> tuple[str, str]:
        """The instants the catalogue is actually asked for, per the client's own expansion.

        Read out of ``pystac_client`` rather than restated here, because the whole point of
        an instant boundary is what the client does to a bare date: expand it to
        ``T23:59:59Z`` and leave the last second of that day unasked for. A change to that
        expansion upstream must fail this, not be agreed with by a local copy of it.
        """
        expanded = ItemSearch(url="http://example.invalid/search", datetime=f"{start}/{end}")
        low, _, high = expanded.get_parameters()["datetime"].partition("/")
        return low, high

    def test_the_parts_cover_every_day_of_the_input(self):
        parts = split_query_window("2024-03-01", "2024-03-30", 4)
        assert parts[0][0] == "2024-03-01"
        assert parts[-1][1] == "2024-03-30"
        # Each window picks up where the last left off, SHARING that boundary, so no
        # acquisition can fall between them.
        assert [later[0] for later in parts[1:]] == [earlier[1] for earlier in parts[:-1]]

    def test_the_union_is_the_input_exactly_with_no_gap_and_no_overhang(self):
        """The property the whole re-partition rests on, checked on the wire form.

        Two failures are possible and both are silent. A GAP drops items the unsplit window
        would have returned, which reads downstream as missing data. An OVERHANG returns
        items it would not have, which changes a mosaic's content for the same request. The
        interior boundaries must therefore be shared instants, and the outer bounds must be
        the caller's own strings.
        """
        for start, end, parts in [
            ("2024-03-01", "2024-03-30", 4),
            ("2019-02-28", "2019-04-01", 6),
            ("2019-03-21", "2019-03-22", 2),
            ("2019-03-16", "2019-03-19T00:00:00Z", 2),
        ]:
            windows = split_query_window(start, end, parts)
            assert windows, (start, end, parts)
            asked = [self._sent(*w) for w in windows]
            whole_low, whole_high = self._sent(start, end)
            assert asked[0][0] == whole_low
            assert asked[-1][1] == whole_high
            # Consecutive requests must meet on the SAME instant: earlier means a gap,
            # later means the seam was covered twice by a whole day.
            assert [low for low, _ in asked[1:]] == [high for _, high in asked[:-1]]

    def test_a_single_day_cannot_be_cut(self):
        assert split_query_window("2024-03-01", "2024-03-01", 2) == []

    def test_a_window_ending_on_a_boundary_instant_can_still_be_cut(self):
        """The recursion re-cuts its OWN output, so it has to parse the instants it emits.

        A window whose end is a boundary instant is exclusive of that instant's day, so a
        one-day window in that form is at the floor and must stop rather than emit a part
        it cannot shorten again.
        """
        assert split_query_window("2019-03-16", "2019-03-19T00:00:00Z", 2) == [
            ("2019-03-16", "2019-03-17T00:00:00Z"),
            ("2019-03-17T00:00:00Z", "2019-03-19T00:00:00Z"),
        ]
        assert split_query_window("2019-03-21", "2019-03-22T00:00:00Z", 2) == []

    def test_two_days_split_into_one_day_each(self):
        """The floor of the recursion, and it must not stall there.

        Stopping here would leave a refused window with nothing to try, which is exactly
        where the live failure ended up. An instant boundary costs no length, so a two-day
        window cuts cleanly without the shared-day fallback the first fix needed.
        """
        assert split_query_window("2019-03-21", "2019-03-22", 2) == [
            ("2019-03-21", "2019-03-22T00:00:00Z"),
            ("2019-03-22T00:00:00Z", "2019-03-22"),
        ]

    def test_a_seam_does_not_repeat_a_whole_day(self):
        """What the instant boundary buys: adjacent windows share an instant, not a date.

        A shared boundary DAY made every seam re-fetch that day in full — measured at
        Earth Search as roughly 1,250 items and 13 page requests per seam, all of them
        discarded by the id dedupe.
        """
        windows = split_query_window("2024-03-01", "2024-03-03", 2)
        assert windows == [
            ("2024-03-01", "2024-03-02T00:00:00Z"),
            ("2024-03-02T00:00:00Z", "2024-03-03"),
        ]
        assert windows[0][1][:10] == windows[1][0][:10], "the seam is one instant, on one day"
        assert "T" in windows[0][1], "a bare date here would drop that day's last second"

    def test_fewer_than_two_parts_is_no_split(self):
        assert split_query_window("2024-03-01", "2024-03-30", 1) == []
