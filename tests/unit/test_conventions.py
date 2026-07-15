"""Tests for GeoZarr convention attribute builders."""

from importlib.metadata import version as _dist_version

import numpy as np
import pytest

from tessera_embeddings.inference.conventions import ENCODER_VERSION, build_convention_attrs, tile_id_to_epsg

_PKG_VERSION = _dist_version("tessera_embeddings")


class TestTileIdToEpsg:
    """Tests for MGRS tile ID → EPSG code derivation."""

    @pytest.mark.parametrize(
        "tile_id, expected",
        [
            ("37PBM", "EPSG:32637"),
            ("33UWP", "EPSG:32633"),
            ("01NBJ", "EPSG:32601"),
            ("60NUG", "EPSG:32660"),
            # Southern hemisphere
            ("56HKH", "EPSG:32756"),
            ("19MCR", "EPSG:32719"),
            ("36MYE", "EPSG:32736"),
        ],
    )
    def test_valid_tiles(self, tile_id: str, expected: str) -> None:
        assert tile_id_to_epsg(tile_id) == expected

    @pytest.mark.parametrize(
        "tile_id",
        [
            "",
            "AB",
            "kenya_highlands",
            "00NBC",
            "61NBC",
            "37ABM",
            None,
        ],
    )
    def test_invalid_tiles_return_none(self, tile_id: str | None) -> None:
        assert tile_id_to_epsg(tile_id) is None  # type: ignore[arg-type]


class TestBuildConventionAttrs:
    """Tests for the full convention attribute builder."""

    @pytest.fixture()
    def projected_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """10m UTM coordinates for a small 100x100 grid."""
        y = np.arange(6200000.0, 6199000.0, -10.0)  # 100 pixels, descending
        x = np.arange(500000.0, 501000.0, 10.0)  # 100 pixels, ascending
        return y, x

    def test_full_attrs_with_tile_id(self, projected_coords: tuple[np.ndarray, np.ndarray]) -> None:
        y_coords, x_coords = projected_coords
        attrs = build_convention_attrs(
            tile_id="33UWP",
            total_y=100,
            total_x=100,
            embedding_dim=128,
            y_coords=y_coords,
            x_coords=x_coords,
            model_version="1.1",
        )

        # zarr_conventions should contain all three
        conventions = attrs["zarr_conventions"]
        names = [c["name"] for c in conventions]
        assert "proj:" in names
        assert "spatial:" in names
        assert "geoemb:" in names
        # Each convention has a UUID
        for conv in conventions:
            assert "uuid" in conv

        # proj: — all fields derived from tile_id EPSG
        assert attrs["proj:code"] == "EPSG:32633"
        assert "proj:wkt2" in attrs
        assert "proj:projjson" in attrs

        # spatial:
        assert attrs["spatial:dimensions"] == ["northing", "easting"]
        assert attrs["spatial:transform_type"] == "affine"
        assert attrs["spatial:transform"] == [10.0, 0.0, 500000.0, 0.0, -10.0, 6200000.0]
        assert attrs["spatial:shape"] == [100, 100]
        assert attrs["spatial:registration"] == "pixel"
        # bbox should be [xmin, ymin, xmax, ymax], extends half-pixel beyond coord centres
        bbox = attrs["spatial:bbox"]
        assert bbox[0] == pytest.approx(500000.0 - 5.0)  # xmin = first x - half pixel
        assert bbox[2] == pytest.approx(500990.0 + 5.0)  # xmax = last x + half pixel
        assert bbox[0] < bbox[2]  # xmin < xmax
        assert bbox[1] < bbox[3]  # ymin < ymax

        # geoemb: — required fields per the convention schema
        assert attrs["geoemb:type"] == "pixel"
        assert attrs["geoemb:dimensions"] == 128
        assert attrs["geoemb:model"] == "https://geotessera.org/model/1.1"  # encoder version
        assert attrs["geoemb:source_data"] == ["s3://sentinel-cogs", "https://datapool.asf.alaska.edu/RTC/OPERA-S1"]
        assert attrs["geoemb:data_type"] == "int8"
        assert attrs["geoemb:gsd"] == 10.0  # derived from the 10 m (metre CRS) coordinate spacing
        # spatial_layout is omitted by default (single-ROI store, no utmNN groups)
        assert "geoemb:spatial_layout" not in attrs
        assert attrs["geoemb:build_version"] == _PKG_VERSION  # software/package version, not the encoder
        quant = attrs["geoemb:quantization"]
        assert quant["method"] == "per_pixel_scale"
        assert quant["original_dtype"] == "float32"
        assert quant["quantized_dtype"] == "int8"
        assert quant["scale"] == {"type": "array", "array_name": "scales", "nodata": "NaN"}

    def test_no_tile_id_no_coords_skips_proj_spatial(self) -> None:
        attrs = build_convention_attrs(
            total_y=10,
            total_x=10,
            embedding_dim=128,
        )
        assert "proj:code" not in attrs
        assert "spatial:dimensions" not in attrs
        # geoemb: should still be present (its required fields don't need CRS/coords)
        assert attrs["geoemb:type"] == "pixel"
        assert attrs["geoemb:dimensions"] == 128
        names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj:" not in names
        assert "spatial:" not in names
        assert "geoemb:" in names

    def test_non_mgrs_tile_id_omits_proj(self) -> None:
        """When tile_id isn't an MGRS tile, proj: fields are omitted."""
        y = np.arange(100.0, 0.0, -10.0)
        x = np.arange(0.0, 100.0, 10.0)
        attrs = build_convention_attrs(
            tile_id="kenya_highlands",
            total_y=10,
            total_x=10,
            embedding_dim=128,
            y_coords=y,
            x_coords=x,
        )
        assert "proj:code" not in attrs
        names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj:" not in names
        # spatial: still present since coords are provided
        assert "spatial:" in names

    def test_model_uses_encoder_version_build_uses_package_version(self) -> None:
        """geoemb:model is versioned by the encoder checkpoint; geoemb:build_version
        is the software/package version (they are distinct).
        """
        attrs = build_convention_attrs(
            total_y=10,
            total_x=10,
            embedding_dim=128,
            model_version=None,
        )
        assert attrs["geoemb:model"] == f"https://geotessera.org/model/{ENCODER_VERSION}"
        assert attrs["geoemb:build_version"] == _PKG_VERSION
        assert attrs["geoemb:build_version"] != ENCODER_VERSION

    def test_spatial_layout_omitted_by_default_included_when_set(self) -> None:
        """spatial_layout is optional: omitted for a root-only single-ROI store,
        present when a multi-group caller (e.g. the campaign) sets it.
        """
        assert "geoemb:spatial_layout" not in build_convention_attrs(total_y=10, total_x=10, embedding_dim=128)
        with_layout = build_convention_attrs(total_y=10, total_x=10, embedding_dim=128, spatial_layout="utm_zones")
        assert with_layout["geoemb:spatial_layout"] == "utm_zones"

    def test_gsd_derived_from_coordinate_spacing(self) -> None:
        """Gsd reflects the actual pixel size, not the nominal default."""
        y = np.arange(1000.0, 800.0, -20.0)  # 20 m spacing
        x = np.arange(0.0, 200.0, 20.0)
        attrs = build_convention_attrs(
            tile_id="33UWP", total_y=10, total_x=10, embedding_dim=128, y_coords=y, x_coords=x
        )
        assert attrs["geoemb:gsd"] == 20.0

    def test_gsd_not_derived_from_geographic_crs_spacing(self) -> None:
        """A geographic CRS (EPSG:4326, degrees) must NOT have its 0.1° spacing
        mislabelled as metres — gsd falls back to the explicit value.
        """
        y = np.arange(10.0, 9.0, -0.1)  # 0.1 degree spacing
        x = np.arange(0.0, 1.0, 0.1)
        attrs = build_convention_attrs(
            epsg_code="EPSG:4326", total_y=10, total_x=10, embedding_dim=128, y_coords=y, x_coords=x, gsd=10.0
        )
        assert attrs["geoemb:gsd"] == 10.0  # nominal fallback, not 0.1

    def test_epsg_code_overrides_tile_id(self) -> None:
        """When both tile_id and epsg_code are provided, epsg_code wins."""
        attrs = build_convention_attrs(
            tile_id="33UWP",
            epsg_code="EPSG:5070",
            total_y=10,
            total_x=10,
            embedding_dim=128,
        )
        assert attrs["proj:code"] == "EPSG:5070"
        assert "proj:wkt2" in attrs
        assert "proj:projjson" in attrs
        # Verify the CRS content references EPSG:5070 (Conus Albers)
        assert "5070" in attrs["proj:wkt2"] or "Conus Albers" in attrs["proj:wkt2"]

    def test_epsg_code_without_tile_id(self) -> None:
        """epsg_code alone (no tile_id) populates proj fields."""
        attrs = build_convention_attrs(
            epsg_code="EPSG:5070",
            total_y=10,
            total_x=10,
            embedding_dim=128,
        )
        assert attrs["proj:code"] == "EPSG:5070"
        assert "proj:wkt2" in attrs
        assert "proj:projjson" in attrs
        names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj:" in names

    def test_single_pixel_coords_skips_spatial(self) -> None:
        """spatial: requires at least 2 coordinate values to derive a transform."""
        attrs = build_convention_attrs(
            tile_id="33UWP",
            total_y=1,
            total_x=1,
            embedding_dim=128,
            y_coords=np.array([6200000.0]),
            x_coords=np.array([500000.0]),
        )
        assert "spatial:dimensions" not in attrs
        # proj: should still work
        assert attrs["proj:code"] == "EPSG:32633"
