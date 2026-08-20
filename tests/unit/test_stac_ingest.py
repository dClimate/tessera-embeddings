"""Unit tests for ingest/stac.py - STAC ingestion with provider/collection config."""

import numpy as np
import pytest
import xarray as xr

from tessera_embeddings.config import (
    LANDSAT_C2_BANDS,
    S1_GRD_BANDS,
    S2_L2A_BANDS,
    CollectionConfig,
    STACProvider,
)
from tessera_embeddings.ingest.stac import (
    _apply_baseline_corrections_by_date,
    _build_stac_query,
    _extract_baseline,
    _filter_existing_dates,
    _get_collection_config,
    _get_provider_config,
    _load_from_stac,
    _loadable_assets,
    _prune_item_dict,
    _query_stac_items,
    extract_baselines,
    ingest_tile,
    split_antimeridian_bbox,
)


class TestBaselineCorrection:
    """Tests for _apply_baseline_corrections_by_date function."""

    def _make_dataset(self, bands_data, dates):
        """Create a dataset with time dimension from {band: 2d_array} dict."""
        coords = {
            "time": [np.datetime64(d, "ns") for d in dates],
            "northing": np.arange(bands_data[next(iter(bands_data))].shape[-2]),
            "easting": np.arange(bands_data[next(iter(bands_data))].shape[-1]),
        }
        data_vars = {band: (["time", "northing", "easting"], arr) for band, arr in bands_data.items()}
        return xr.Dataset(data_vars, coords=coords)

    def test_subtracts_offset_when_baseline_gte_400(self):
        """Date with baseline>=400 should have 1000 subtracted."""
        values = np.array([[[1500, 2000], [3000, 4000]]], dtype=np.uint16)
        data = self._make_dataset(
            {"B02": values.copy(), "B03": values.copy()},
            ["2024-01-01"],
        )

        result = _apply_baseline_corrections_by_date(data, baselines={"2024-01-01": 400})

        expected = np.array([[[500, 1000], [2000, 3000]]], dtype=np.int16)
        np.testing.assert_array_equal(result["B02"].values, expected)
        np.testing.assert_array_equal(result["B03"].values, expected)

    def test_low_values_go_negative_after_correction(self):
        """Values below 1000 become negative after offset is applied."""
        values = np.array([[[500, 800], [1000, 1500]]], dtype=np.uint16)
        data = self._make_dataset({"B02": values}, ["2024-01-01"])

        result = _apply_baseline_corrections_by_date(data, baselines={"2024-01-01": 400})

        # 500-1000=-500, 800-1000=-200, 1000-1000=0, 1500-1000=500
        np.testing.assert_array_equal(
            result["B02"].values,
            np.array([[[-500, -200], [0, 500]]], dtype=np.int16),
        )

    def test_skips_correction_when_baseline_lt_400(self):
        """Dates with baseline < 400 should not have any correction applied."""
        original = np.array([[[1500, 2000], [3000, 4000]]], dtype=np.uint16)
        data = self._make_dataset({"B02": original.copy()}, ["2024-01-01"])

        result = _apply_baseline_corrections_by_date(data, baselines={"2024-01-01": 399})

        # Values should be unchanged (cast to int16 but same magnitude)
        np.testing.assert_array_equal(
            result["B02"].values,
            original.astype(np.int16),
        )

    def test_only_corrects_specified_bands(self):
        """When bands parameter is provided, only those bands are corrected."""
        values = np.array([[[2000]]], dtype=np.uint16)
        data = self._make_dataset(
            {"B02": values.copy(), "B03": values.copy()},
            ["2024-01-01"],
        )

        result = _apply_baseline_corrections_by_date(data, baselines={"2024-01-01": 400}, bands=["B02"])

        assert result["B02"].values[0, 0, 0] == 1000
        assert result["B03"].values[0, 0, 0] == 2000

    def test_corrects_only_dates_above_threshold(self):
        """Only dates with baseline >= threshold get corrected."""
        values = np.array([[[2000]], [[2000]], [[2000]]], dtype=np.uint16)
        data = self._make_dataset(
            {"B02": values.copy()},
            ["2024-01-01", "2024-01-06", "2024-01-11"],
        )

        result = _apply_baseline_corrections_by_date(
            data,
            baselines={
                "2024-01-01": 400,  # corrected
                "2024-01-06": 399,  # not corrected
                "2024-01-11": 510,  # corrected
            },
        )

        assert result["B02"].values[0, 0, 0] == 1000  # corrected
        assert result["B02"].values[1, 0, 0] == 2000  # unchanged
        assert result["B02"].values[2, 0, 0] == 1000  # corrected


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
            ("earth-search", "sentinel-2-l2a", False),
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

        def mock_load(*args, **kwargs):
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
        # blue (B02) values should stay same
        assert result_data["blue"].values[0, 0, 0] == 2000
        assert result_baselines == {"2024-01-01": 400}

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

        def mock_load(*args, **kwargs):
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
        # blue doesn't need to be corrected
        assert result_data["blue"].values[0, 0, 0] == 2000
        # scl should NOT be corrected (still 4)
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
        monkeypatch.setattr(
            "tessera_embeddings.ingest.stac.normalize_to_solar_day",
            lambda items, mid_longitude=None: items,
        )

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
