"""Unit tests for opera_query.py - OPERA RTC-S1 query utilities."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from tessera_embeddings.ingest.opera_query import (
    _granule_to_item,
    _query_cmr_granules,
    make_s1_item_provider,
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
# _granule_to_item / _query_cmr_granules
# ---------------------------------------------------------------------------

# One IW1 ring (lat lon pairs) lifted from a real OPERA granule near Ireland.
_RING = "55.15054 -9.61626 55.30706 -10.9955 55.14258 -11.05041 54.98632 -9.67693 55.15054 -9.61626"
_HREF_BASE = "https://datapool.asf.alaska.edu/RTC/OPERA-S1"


def _granule_entry(granule_id, bands=("VV", "VH"), with_ring=True):
    """Build one CMR granule ``feed.entry`` dict with the given band data links."""
    links = [
        {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": f"{_HREF_BASE}/{granule_id}_{b}.tif"}
        for b in bands
    ]
    # Mirror the real response: each data link is paired with an s3# link we ignore.
    links += [
        {"rel": "http://esipfed.org/ns/fedsearch/1.1/s3#", "href": f"s3://bucket/{granule_id}_{b}.tif"} for b in bands
    ]
    entry = {
        "title": granule_id,
        "time_start": "2024-06-02T07:03:52.000Z",
        "links": links,
    }
    if with_ring:
        entry["polygons"] = [[_RING]]
    return entry


def _cmr_response(entries, search_after=None):
    """Build a mock requests.Response for CMR granule search."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"feed": {"entry": entries}}
    resp.headers = {"CMR-Search-After": search_after} if search_after else {}
    return resp


def _patch_cmr_session(**get_kwargs):
    """Patch ``_cmr_session`` so its session's ``.get`` is a controllable mock.

    The query helper issues requests via ``_cmr_session().get(...)`` (a
    retry-mounted Session), not the module-level ``requests.get``. This patches
    the session factory and returns ``(patcher, mock_get)`` so callers can make
    assertions on ``mock_get`` while the context manager is active.
    """
    mock_get = MagicMock(**get_kwargs)
    session = MagicMock()
    session.get = mock_get
    patcher = patch("tessera_embeddings.ingest.opera_query._cmr_session", return_value=session)
    return patcher, mock_get


class TestGranuleToItem:
    """Tests for _granule_to_item()."""

    def test_maps_band_links_to_opera_asset_keys(self):
        item = _granule_to_item(_granule_entry("OPERA_L2_RTC-S1_T035-073245-IW3_20240101_v1.0"))
        assert item is not None
        assert set(item.assets) == {"0_VV", "0_VH"}
        assert item.assets["0_VV"].href.endswith("_VV.tif")
        assert item.assets["0_VH"].href.endswith("_VH.tif")
        assert item.assets["0_VV"].roles == ["data"]

    def test_id_and_datetime_from_granule(self):
        item = _granule_to_item(_granule_entry("granule-123"))
        assert item.id == "granule-123"
        assert item.datetime == datetime(2024, 6, 2, 7, 3, 52, tzinfo=UTC)

    def test_geometry_and_bbox_from_polygon(self):
        item = _granule_to_item(_granule_entry("g1"))
        assert item.geometry["type"] == "Polygon"
        # GeoJSON is [lon, lat]; first ring vertex was lat=55.15054 lon=-9.61626.
        assert item.geometry["coordinates"][0][0] == [-9.61626, 55.15054]
        west, south, east, north = item.bbox
        assert west < east and south < north
        assert west == pytest.approx(-11.05041)

    def test_missing_band_links_returns_none(self):
        """A granule lacking a VV or VH data link is skipped."""
        assert _granule_to_item(_granule_entry("g1", bands=("VV",))) is None

    def test_handles_missing_polygon(self):
        item = _granule_to_item(_granule_entry("g1", with_ring=False))
        assert item is not None
        assert item.geometry is None
        assert item.bbox is None


class TestQueryCmrGranules:
    """Tests for _query_cmr_granules()."""

    BBOX = (-122.0, 37.0, -121.0, 38.0)

    def test_returns_built_items(self):
        entries = [_granule_entry("g-1"), _granule_entry("g-2")]
        patcher, _ = _patch_cmr_session(return_value=_cmr_response(entries))
        with patcher:
            items = _query_cmr_granules(self.BBOX, "2024-01-01", "2024-01-15", "ascending")

        assert {item.id for item in items} == {"g-1", "g-2"}
        assert all(set(item.assets) == {"0_VV", "0_VH"} for item in items)

    def test_passes_attribute_filter(self):
        patcher, mock_get = _patch_cmr_session(return_value=_cmr_response([]))
        with patcher:
            _query_cmr_granules(self.BBOX, "2024-01-01", "2024-01-15", "descending")

        params = mock_get.call_args.kwargs["params"]
        assert params["attribute[]"] == "string,ASCENDING_DESCENDING,DESCENDING"

    def test_paginates_with_cmr_search_after(self):
        page1 = _cmr_response(
            [_granule_entry(f"g-{i}") for i in range(2000)],
            search_after="token123",
        )
        page2 = _cmr_response([_granule_entry("g-final")])
        patcher, mock_get = _patch_cmr_session(side_effect=[page1, page2])
        with patcher:
            items = _query_cmr_granules(self.BBOX, "2024-01-01", "2024-12-31", "ascending")

        assert any(item.id == "g-final" for item in items)
        _, kwargs2 = mock_get.call_args_list[1]
        assert kwargs2["headers"]["CMR-Search-After"] == "token123"

    def test_normal_bbox_queries_once(self):
        """A west<east bbox issues a single CMR query (no antimeridian split)."""
        with patch(
            "tessera_embeddings.ingest.opera_query._query_cmr_granules_one", return_value=[]
        ) as one:
            _query_cmr_granules(self.BBOX, "2024-01-01", "2024-01-15", "ascending")
        one.assert_called_once()

    def test_antimeridian_bbox_splits_and_dedupes(self):
        """A west>east (antimeridian) bbox is split at ±180 into two CMR-valid
        boxes and their granules merged with dedup — CMR reads bounding_box as
        lower-left/upper-right, so a single west>east query would match nothing.
        """
        calls: list[tuple] = []

        def fake_one(bbox, start, end, orbit):
            calls.append(bbox)
            # West half returns g1; east half returns g1 (shared burst) + g2.
            return [MagicMock(id="g1")] if bbox[0] >= 170.0 else [MagicMock(id="g1"), MagicMock(id="g2")]

        with patch("tessera_embeddings.ingest.opera_query._query_cmr_granules_one", side_effect=fake_one):
            items = _query_cmr_granules((170.0, -10.0, -170.0, 10.0), "2024-01-01", "2024-01-15", "ascending")

        assert calls == [(170.0, -10.0, 180.0, 10.0), (-180.0, -10.0, -170.0, 10.0)]
        assert {i.id for i in items} == {"g1", "g2"}  # shared granule deduped across halves

    def test_empty_response(self):
        patcher, _ = _patch_cmr_session(return_value=_cmr_response([]))
        with patcher:
            items = _query_cmr_granules(self.BBOX, "2024-01-01", "2024-01-15", "ascending")

        assert items == []


# ---------------------------------------------------------------------------
# make_s1_item_provider
# ---------------------------------------------------------------------------


class TestMakeS1ItemProvider:
    """Tests for make_s1_item_provider()."""

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match=r"ascending.*descending"):
            make_s1_item_provider("invalid", (-122, 37, -121, 38), "2024-01-01", "2024-01-15")

    def test_builds_rewrites_and_normalizes_items(self):
        """The provider queries CMR, rewrites assets to S3, and normalizes timestamps."""
        items = [_granule_to_item(_granule_entry("g-1")), _granule_to_item(_granule_entry("g-2"))]

        with (
            patch("tessera_embeddings.ingest.opera_query._query_cmr_granules", return_value=items),
            patch("tessera_embeddings.ingest.opera_query.rewrite_assets_to_s3") as mock_rewrite,
        ):
            provide = make_s1_item_provider("ascending", (-122, 37, -121, 38), "2024-01-01", "2024-01-15")
            result = provide()

        assert len(result) == 2
        # Assets were rewritten once per item.
        assert mock_rewrite.call_count == 2
        # Same-date bursts share the canonical noon-UTC timestamp.
        assert all(item.datetime == datetime(2024, 6, 2, 12, 0, 0, tzinfo=UTC) for item in result)
