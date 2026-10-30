"""Which of two overlapping sources of one solar day actually supplies a pixel.

Asserting the ORDER handed to `odc.stac.load` is only half the question. The other half is what odc
does with that order: its default fuser is `nodata_fuser`, which writes only where the destination
is still empty, so the FIRST valid source of a group supplies a pixel and later ones fill its gaps.

**These tests therefore refuse to mock the loader**, because a mock would assert the half that was
never in doubt. Real GeoTIFFs on one grid, real items on one solar day, through the production load
path. The assertion is on the PIXELS that come out: the clearest optical scene, and the radar burst
acquired most often over the window (ADR 033).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import rasterio
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox
from pystac import Item
from rasterio.transform import from_origin

from tessera_embeddings.config.ingest import SOURCE_INDEX_VAR
from tessera_embeddings.config.providers import CollectionConfig
from tessera_embeddings.ingest import stac as stac_module
from tessera_embeddings.ingest.opera_query import BURST_ACQUISITIONS, BURST_ID, _granule_to_item
from tessera_embeddings.ingest.provenance import FUSION_POSITION
from tessera_embeddings.ingest.stac import _load_from_stac, ingest_tile, load_stac_items
from tessera_embeddings.ingest.transforms import amplitude_to_db

CLOUDY_DN = 100
CLEAR_DN = 200
NODATA = 0
SIZE = 8
RES = 10.0
EAST0, NORTH1 = 500000.0, 5000000.0
EPSG = 32633


def _write(
    path: pathlib.Path, value: float, *, left_half_only: bool = False, dtype: str = "uint16", nodata: float = NODATA
) -> None:
    pixels = np.full((SIZE, SIZE), value, dtype=dtype)
    if left_half_only:
        # A hole on the right: what an off-swath or masked region looks like to the fuser.
        pixels[:, SIZE // 2 :] = nodata
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=SIZE,
        height=SIZE,
        count=1,
        dtype=dtype,
        crs=f"EPSG:{EPSG}",
        transform=from_origin(EAST0, NORTH1, RES, RES),
        nodata=nodata,
    ) as dst:
        dst.write(pixels, 1)


def _item(item_id: str, href: str, cloud: float) -> Item:
    return Item.from_dict(
        {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": item_id,
            # Must actually contain the raster's ground, or odc drops the item before reading it.
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[14.9, 45.0], [15.2, 45.0], [15.2, 45.3], [14.9, 45.3], [14.9, 45.0]]],
            },
            "bbox": [14.9, 45.0, 15.2, 45.3],
            "properties": {
                # Both on one solar day, stamped to noon exactly as `normalize_to_solar_day` leaves
                # them — the form the loader actually receives.
                "datetime": "2024-06-15T12:00:00Z",
                "eo:cloud_cover": cloud,
                "proj:epsg": EPSG,
                "proj:shape": [SIZE, SIZE],
                "proj:transform": [RES, 0.0, EAST0, 0.0, -RES, NORTH1, 0.0, 0.0, 1.0],
            },
            "links": [],
            "assets": {
                "blue": {
                    "href": href,
                    "type": "image/tiff; application=geotiff",
                    "roles": ["data"],
                    "raster:bands": [{"nodata": NODATA, "data_type": "uint16"}],
                    "eo:bands": [{"name": "B02", "common_name": "blue"}],
                }
            },
            "stac_extensions": [],
        }
    )


GBOX = GeoBox.from_bbox(
    BoundingBox(EAST0, NORTH1 - SIZE * RES, EAST0 + SIZE * RES, NORTH1, crs=f"EPSG:{EPSG}"),
    resolution=RES,
)
CHUNKS = {"northing": SIZE, "easting": SIZE}


def _load(items: list[Item], *, fusion_order: str | None = "clearest_scene") -> np.ndarray:
    # A collection naming a fusion order is one whose order THIS package decides, which is what
    # `_load_from_stac` keys `preserve_original_order` on.
    cfg = CollectionConfig(collection_id="test", bands=["blue"], resolution=int(RES), fusion_order=fusion_order)
    ds = _load_from_stac(items, cfg, geobox=GBOX, chunks=CHUNKS)
    assert ds.sizes["time"] == 1, "both items must land in ONE solar-day slice or this proves nothing"
    return np.asarray(ds["blue"].compute().values[0])


@pytest.fixture
def scenes(tmp_path: pathlib.Path):
    """A cloudy and a clear scene covering the same ground, distinguishable by pixel value."""
    cloudy_tif, clear_tif = tmp_path / "cloudy.tif", tmp_path / "clear.tif"
    _write(cloudy_tif, CLOUDY_DN)
    _write(clear_tif, CLEAR_DN)
    return (
        _item("CLOUDY", str(cloudy_tif), cloud=90.0),
        _item("CLEAR", str(clear_tif), cloud=5.0),
    )


def test_the_first_source_wins_rather_than_the_last(scenes) -> None:
    """The mechanism, stated as the thing it is: position decides, and it is the FIRST position.

    Says nothing about clouds — it pins that odc fuses first-wins, so reversing the sort is known
    to reverse the outcome rather than assumed to.
    """
    cloudy, clear = scenes
    assert np.unique(_load([cloudy, clear])).tolist() == [CLOUDY_DN]
    assert np.unique(_load([clear, cloudy])).tolist() == [CLEAR_DN]


def test_a_hole_in_the_clearest_scene_falls_through_to_the_next(scenes, tmp_path) -> None:
    """Why first-wins plus clearest-first is the RIGHT pairing, not merely a self-consistent one.

    The clear scene wins the ground it covers, and where it has no observation the cloudier scene
    still fills the gap — so the day keeps its coverage instead of trading it for clarity.
    """
    cloudy, _ = scenes
    holed = tmp_path / "clear_holed.tif"
    _write(holed, CLEAR_DN, left_half_only=True)
    fused = _load([_item("CLEAR", str(holed), cloud=5.0), cloudy])

    assert np.unique(fused[:, : SIZE // 2]).tolist() == [CLEAR_DN], "the clear scene keeps what it covers"
    assert np.unique(fused[:, SIZE // 2 :]).tolist() == [CLOUDY_DN], "its hole is filled, not left empty"


def test_a_collection_this_package_does_not_sort_fuses_the_same_either_way(scenes) -> None:
    """Where no fusion order is named, odc's own `(time, id)` order decides — and that is the point.

    Without odc's default, the order the provider returned would pick the winner of every overlap.
    """
    cloudy, clear = scenes
    assert np.array_equal(_load([cloudy, clear], fusion_order=None), _load([clear, cloudy], fusion_order=None)), (
        "the fused result must not depend on the order the items were supplied in"
    )


#: Two bursts of adjacent ascending tracks, as overlap at high latitude: different look geometries.
#: The lower burst id is the one a title-ordered load would pick.
LOW_TRACK, HIGH_TRACK = "T029-061234-IW2", "T131-279876-IW1"
AMPLITUDE = {LOW_TRACK: 0.1, HIGH_TRACK: 0.2}


def _burst(tmp_path: pathlib.Path, burst: str, *, holed: bool = False) -> Item:
    """One OPERA granule over the test ground, built from a CMR entry exactly as the query builds it."""
    title = f"OPERA_L2_RTC-S1_{burst}_20240615T163000Z_20240616T000000Z_S1A_30_v1.0"
    links = []
    for band in ("VV", "VH"):
        _write(
            tmp_path / f"{title}_{band}.tif",
            AMPLITUDE[burst],
            dtype="float32",
            nodata=np.nan,
            left_half_only=holed,
        )
        links.append(
            {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": str(tmp_path / f"{title}_{band}.tif")}
        )
    entry = {
        "title": title,
        "time_start": "2024-06-15T16:30:00.000Z",
        "links": links,
        "polygons": [["45.0 14.9 45.0 15.2 45.3 15.2 45.3 14.9 45.0 14.9"]],
    }
    item = _granule_to_item(entry)
    assert item is not None
    return item


@pytest.mark.parametrize("favoured", [LOW_TRACK, HIGH_TRACK])
def test_the_most_acquired_burst_wins_a_contested_radar_pixel(tmp_path, favoured) -> None:
    """Swap which burst the census favours and the winner swaps, whatever the supply order or ids.

    Supplied high track first, so the low track winning shows the sort moved it, and the high
    track winning shows the count outranks the burst id. Through `ingest_tile`, which sorts on the
    collection's fusion order and loads with the order kept.
    """
    items = [_burst(tmp_path, HIGH_TRACK), _burst(tmp_path, LOW_TRACK)]
    for item in items:
        # What the item provider stamps from the window's census.
        item.properties[BURST_ACQUISITIONS] = 31 if item.properties[BURST_ID] == favoured else 29

    ds, _ = ingest_tile(
        provider="cmr-asf",
        collection="opera-rtc-s1",
        tile_id=None,
        start_date="2024-06-14",
        end_date="2024-06-16",
        bbox=(14.9, 45.0, 15.2, 45.3),
        geobox=GBOX,
        chunks=CHUNKS,
        item_provider_fn=lambda: items,
        mid_longitude=15.0,
    )

    assert ds is not None and ds.sizes["time"] == 1, "both bursts must land in ONE solar-day slice"
    np.testing.assert_allclose(ds["0_VV"].compute().values[0], AMPLITUDE[favoured])


def _load_indexed(items: list[Item], monkeypatch: pytest.MonkeyPatch, *, source_index: bool = True):
    """Load through `load_stac_items`, with the test band as the collection's source-index band."""
    cfg = CollectionConfig(
        collection_id="test",
        bands=["blue"],
        resolution=int(RES),
        fusion_order="clearest_scene",
        source_index_band="blue",
    )
    monkeypatch.setattr(stac_module, "_get_collection_config", lambda *_: cfg)
    ds = load_stac_items(
        items, provider="test", collection="test", geobox=GBOX, chunks=CHUNKS, source_index=source_index
    )
    return ds.compute()


def test_the_source_index_names_the_granule_the_fuser_kept(scenes, tmp_path, monkeypatch) -> None:
    """The index is the fuser's own choice, and asking for it leaves the fused band untouched.

    The clear scene goes first with a hole, so it supplies the left half and the cloudy scene the
    right. Positions are 0-based in fusion order; the index stores position + 1.
    """
    cloudy, _ = scenes
    holed = tmp_path / "clear_holed.tif"
    _write(holed, CLEAR_DN, left_half_only=True)
    items = [_item("CLEAR", str(holed), cloud=5.0), cloudy]

    indexed = _load_indexed(items, monkeypatch)
    plain = _load_indexed(items, monkeypatch, source_index=False)

    np.testing.assert_array_equal(indexed["blue"].values, plain["blue"].values)
    assert indexed["blue"].dtype == plain["blue"].dtype == np.uint16
    index = indexed[SOURCE_INDEX_VAR].values[0]
    assert index.dtype == np.uint16
    assert np.unique(index[:, : SIZE // 2]).tolist() == [1], "the first granule supplies what it covers"
    assert np.unique(index[:, SIZE // 2 :]).tolist() == [2], "the second fills its hole"
    assert [item.properties[FUSION_POSITION] for item in items] == [0, 1]


def test_a_pixel_no_granule_covers_has_index_zero_and_the_fill(tmp_path, monkeypatch) -> None:
    """Zero means no granule had data, and the band keeps the nodata an unpacked load leaves."""
    left = tmp_path / "left.tif"
    _write(left, CLEAR_DN, left_half_only=True)
    indexed = _load_indexed([_item("ONLY", str(left), cloud=5.0)], monkeypatch)
    assert np.unique(indexed[SOURCE_INDEX_VAR].values[0][:, SIZE // 2 :]).tolist() == [0]
    assert np.unique(indexed["blue"].values[0][:, SIZE // 2 :]).tolist() == [NODATA]


def test_the_radar_source_index_follows_the_burst_order(tmp_path) -> None:
    """Float radar, NaN nodata, through `ingest_tile`: the favoured burst's hole falls to the other.

    The dB transform runs over the bands only; the index is added after it, unscaled.
    """
    items = [_burst(tmp_path, HIGH_TRACK), _burst(tmp_path, LOW_TRACK, holed=True)]
    for item in items:
        item.properties[BURST_ACQUISITIONS] = 31 if item.properties[BURST_ID] == LOW_TRACK else 29

    def load(source_index: bool):
        ds, _ = ingest_tile(
            provider="cmr-asf",
            collection="opera-rtc-s1",
            tile_id=None,
            start_date="2024-06-14",
            end_date="2024-06-16",
            bbox=(14.9, 45.0, 15.2, 45.3),
            geobox=GBOX,
            chunks=CHUNKS,
            item_provider_fn=lambda: items,
            mid_longitude=15.0,
            post_load_fn=amplitude_to_db,
            source_index=source_index,
        )
        assert ds is not None
        return ds.compute()

    indexed, plain = load(True), load(False)
    for band in ("0_VV", "0_VH"):
        np.testing.assert_array_equal(indexed[band].values, plain[band].values)
    index = indexed[SOURCE_INDEX_VAR].values[0]
    assert np.unique(index[:, : SIZE // 2]).tolist() == [1], "the most-acquired burst is first"
    assert np.unique(index[:, SIZE // 2 :]).tolist() == [2], "its hole falls to the other burst"
