"""Tests for GeoZarr convention attribute builders."""

import numpy as np
import pytest

from tessera_embeddings.inference.conventions import build_convention_attrs, tile_id_to_epsg


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
            model_version="best_model_fsdp_20250608",
            n_tiles=24,
        )

        # zarr_conventions should contain all three
        conventions = attrs["zarr_conventions"]
        names = [c["name"] for c in conventions]
        assert "proj:" in names
        assert "spatial:" in names
        assert "tessera:" in names
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

        # tessera:
        assert attrs["tessera:dataset_version"] == "v1"
        assert attrs["tessera:n_bands"] == 128
        assert attrs["tessera:quantization_method"] == "absmax_per_pixel"
        assert attrs["tessera:model_version"] == "best_model_fsdp_20250608"
        assert attrs["tessera:n_tiles"] == 24

    def test_no_tile_id_no_coords_skips_proj_spatial(self) -> None:
        attrs = build_convention_attrs(
            total_y=10,
            total_x=10,
            embedding_dim=128,
        )
        assert "proj:code" not in attrs
        assert "spatial:dimensions" not in attrs
        # tessera: should still be present
        assert attrs["tessera:dataset_version"] == "v1"
        names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj:" not in names
        assert "spatial:" not in names
        assert "tessera:" in names

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

    def test_optional_fields_omitted_when_none(self) -> None:
        attrs = build_convention_attrs(
            total_y=10,
            total_x=10,
            embedding_dim=128,
            model_version=None,
            n_tiles=None,
        )
        assert "tessera:model_version" not in attrs
        assert "tessera:n_tiles" not in attrs

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
