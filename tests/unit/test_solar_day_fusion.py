"""Which of two overlapping scenes of one solar day actually supplies a pixel.

Asserting the ORDER handed to `odc.stac.load` is only half the question. The other half is what odc
does with that order: its default fuser is `nodata_fuser`, which writes only where the destination
is still empty, so the FIRST valid source of a group supplies a pixel and later ones fill its gaps.

**This test therefore refuses to mock the loader**, because a mock would assert the half that was
never in doubt. Two real single-band GeoTIFFs on one grid, two real STAC items on one solar day,
through the production `_load_from_stac` with the production load kwargs. The assertion is on the
PIXELS that come out.
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

from tessera_embeddings.config.providers import CollectionConfig
from tessera_embeddings.ingest.stac import _load_from_stac

CLOUDY_DN = 100
CLEAR_DN = 200
NODATA = 0
SIZE = 8
RES = 10.0
EAST0, NORTH1 = 500000.0, 5000000.0
EPSG = 32633


def _write(path: pathlib.Path, value: int, *, left_half_only: bool = False) -> None:
    pixels = np.full((SIZE, SIZE), value, dtype="uint16")
    if left_half_only:
        # A hole on the right: what an off-swath or masked region looks like to the fuser.
        pixels[:, SIZE // 2 :] = NODATA
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=SIZE,
        height=SIZE,
        count=1,
        dtype="uint16",
        crs=f"EPSG:{EPSG}",
        transform=from_origin(EAST0, NORTH1, RES, RES),
        nodata=NODATA,
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


def _load(items: list[Item]) -> np.ndarray:
    gbox = GeoBox.from_bbox(
        BoundingBox(EAST0, NORTH1 - SIZE * RES, EAST0 + SIZE * RES, NORTH1, crs=f"EPSG:{EPSG}"),
        resolution=RES,
    )
    cfg = CollectionConfig(collection_id="test", bands=["blue"], resolution=int(RES))
    ds = _load_from_stac(items, cfg, geobox=gbox, chunks={"northing": SIZE, "easting": SIZE})
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


def test_the_clearest_scene_supplies_the_overlap(scenes) -> None:
    """THE rule. Clearest first, and the clearest is what comes out.

    Handed the other way round the cloudy scene wins, which is what makes the sort in
    `query_stac_items` load-bearing rather than cosmetic.
    """
    cloudy, clear = scenes
    assert np.unique(_load([clear, cloudy])).tolist() == [CLEAR_DN]


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
