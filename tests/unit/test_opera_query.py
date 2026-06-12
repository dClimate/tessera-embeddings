"""Unit tests for opera_query.py - OPERA RTC-S1 query utilities."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from tessera_embeddings.ingest.opera_query import (
    _query_cmr_granule_ids,
    make_s1_item_rewriter,
    mgrs_tile_to_bbox,
    mgrs_tile_to_utm_epsg,
    normalize_opera_timestamps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stac_item(hour: int, minute: int = 0) -> MagicMock:
    """Create a mock STAC item with a specific UTC acquisition hour."""
    item = MagicMock()
    item.datetime = datetime(2024, 6, 1, hour, minute, 0, tzinfo=UTC)
    return item


# ---------------------------------------------------------------------------
# mgrs_tile_to_bbox
# ---------------------------------------------------------------------------


class TestMgrsTileToBbox:
    """Tests for mgrs_tile_to_bbox()."""

    def test_returns_four_element_tuple(self):
        bbox = mgrs_tile_to_bbox("33UUP")
        assert len(bbox) == 4

    def test_bbox_is_valid_wgs84(self):
        west, south, east, north = mgrs_tile_to_bbox("33UUP")
        assert -180 <= west <= 180
        assert -90 <= south <= 90
        assert -180 <= east <= 180
        assert -90 <= north <= 90
        assert west < east
        assert south < north

    def test_tile_33uup_bbox_in_expected_range(self):
        """Tile 33UUP is in Austria/Czech Republic area."""
        west, south, east, north = mgrs_tile_to_bbox("33UUP")
        assert 12.0 < west < 14.0
        assert 47.0 < south < 49.0
        assert 13.0 < east < 15.0
        assert 48.0 < north < 50.0

    def test_southern_hemisphere_tile(self):
        """Tile 55HBE is in New Zealand (southern hemisphere)."""
        _, south, _, _ = mgrs_tile_to_bbox("55HBE")
        assert south < 0


# ---------------------------------------------------------------------------
# mgrs_tile_to_utm_epsg — parameterized
# ---------------------------------------------------------------------------


class TestMgrsTileToUtmEpsg:
    """Tests for mgrs_tile_to_utm_epsg()."""

    @pytest.mark.parametrize(
        "tile_id, expected_epsg",
        [
            ("33UUP", "EPSG:32633"),  # Zone 33, band U (northern)
            ("55HBE", "EPSG:32755"),  # Zone 55, band H (southern)
            ("10SEG", "EPSG:32610"),  # Zone 10, band S (northern)
        ],
        ids=["zone33-north", "zone55-south", "zone10-north"],
    )
    def test_known_tiles(self, tile_id, expected_epsg):
        assert mgrs_tile_to_utm_epsg(tile_id) == expected_epsg

    @pytest.mark.parametrize(
        "tile_id, epsg_prefix",
        [
            ("33NXX", "EPSG:326"),  # Band N = first northern band
            ("33MXX", "EPSG:327"),  # Band M = last southern band
        ],
        ids=["band-N-is-north", "band-M-is-south"],
    )
    def test_hemisphere_boundary(self, tile_id, epsg_prefix):
        """Bands N-X are northern, C-M are southern."""
        assert mgrs_tile_to_utm_epsg(tile_id).startswith(epsg_prefix)


# ---------------------------------------------------------------------------
# normalize_opera_timestamps
# ---------------------------------------------------------------------------


class TestNormalizeOperaTimestamps:
    """Tests for normalize_opera_timestamps()."""

    def test_bursts_on_same_date_get_same_timestamp(self):
        items = [
            _make_stac_item(5, 30),
            _make_stac_item(5, 45),
            _make_stac_item(5, 50),
        ]
        normalize_opera_timestamps(items)
        assert all(item.datetime == items[0].datetime for item in items)
        assert items[0].datetime == datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)

    def test_different_dates_stay_distinct(self):
        item_a = _make_stac_item(5)
        item_b = MagicMock()
        item_b.datetime = datetime(2024, 6, 3, 5, 30, 0, tzinfo=UTC)

        normalize_opera_timestamps([item_a, item_b])

        assert item_a.datetime.day == 1
        assert item_b.datetime.day == 3
        assert item_a.datetime.hour == item_b.datetime.hour == 12

    def test_empty_input(self):
        assert normalize_opera_timestamps([]) == []

    def test_modifies_in_place_and_returns_same_list(self):
        items = [_make_stac_item(5)]
        result = normalize_opera_timestamps(items)
        assert result is items


# ---------------------------------------------------------------------------
# _query_cmr_granule_ids
# ---------------------------------------------------------------------------


def _cmr_response(entries, search_after=None):
    """Build a mock requests.Response for CMR granule search."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"feed": {"entry": entries}}
    resp.headers = {"CMR-Search-After": search_after} if search_after else {}
    return resp


class TestQueryCmrOrbitIds:
    """Tests for _query_cmr_granule_ids()."""

    BBOX = (-122.0, 37.0, -121.0, 38.0)

    def test_returns_producer_granule_ids(self):
        entries = [
            {"producer_granule_id": "OPERA_L2_RTC-S1_T035-073245-IW3_20240101_v1.0"},
            {"producer_granule_id": "OPERA_L2_RTC-S1_T035-073246-IW2_20240101_v1.0"},
        ]
        with patch("tessera_embeddings.ingest.opera_query.requests.get", return_value=_cmr_response(entries)):
            ids = _query_cmr_granule_ids(self.BBOX, "2024-01-01", "2024-01-15", "ascending")

        assert ids == {
            "OPERA_L2_RTC-S1_T035-073245-IW3_20240101_v1.0",
            "OPERA_L2_RTC-S1_T035-073246-IW2_20240101_v1.0",
        }

    def test_passes_attribute_filter(self):
        with patch("tessera_embeddings.ingest.opera_query.requests.get", return_value=_cmr_response([])) as mock_get:
            _query_cmr_granule_ids(self.BBOX, "2024-01-01", "2024-01-15", "descending")

        params = mock_get.call_args.kwargs["params"]
        assert params["attribute[]"] == "string,ASCENDING_DESCENDING,DESCENDING"

    def test_paginates_with_cmr_search_after(self):
        page1 = _cmr_response(
            [{"producer_granule_id": f"id-{i}"} for i in range(2000)],
            search_after="token123",
        )
        page2 = _cmr_response([{"producer_granule_id": "id-final"}])
        with patch("tessera_embeddings.ingest.opera_query.requests.get", side_effect=[page1, page2]) as mock_get:
            ids = _query_cmr_granule_ids(self.BBOX, "2024-01-01", "2024-12-31", "ascending")

        assert "id-final" in ids
        _, kwargs2 = mock_get.call_args_list[1]
        assert kwargs2["headers"]["CMR-Search-After"] == "token123"

    def test_empty_response(self):
        with patch("tessera_embeddings.ingest.opera_query.requests.get", return_value=_cmr_response([])):
            ids = _query_cmr_granule_ids(self.BBOX, "2024-01-01", "2024-01-15", "ascending")

        assert ids == set()


# ---------------------------------------------------------------------------
# make_s1_item_rewriter
# ---------------------------------------------------------------------------


class TestMakeS1ItemRewriter:
    """Tests for make_s1_item_rewriter()."""

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match=r"ascending.*descending"):
            make_s1_item_rewriter("invalid", (-122, 37, -121, 38), "2024-01-01", "2024-01-15")

    def test_filters_items_by_orbit_ids(self):
        """Items not in the CMR orbit ID set are excluded."""
        items = [MagicMock(id="match-1"), MagicMock(id="no-match"), MagicMock(id="match-2")]
        for item in items:
            item.datetime = datetime(2024, 6, 1, 5, 0, 0, tzinfo=UTC)
            item.assets = {}

        orbit_ids = {"match-1", "match-2"}
        with (
            patch("tessera_embeddings.ingest.opera_query._query_cmr_granule_ids", return_value=orbit_ids),
            patch("tessera_embeddings.ingest.opera_query.rewrite_assets_to_s3"),
        ):
            fn = make_s1_item_rewriter("ascending", (-122, 37, -121, 38), "2024-01-01", "2024-01-15")
            result = fn(items)

        assert len(result) == 2
        assert all(item.id in orbit_ids for item in result)
