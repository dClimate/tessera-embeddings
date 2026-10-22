"""Tests for GeoZarr convention attribute builders."""

from importlib.metadata import version as _dist_version
from typing import ClassVar

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.inference import encoder_url
from tessera_embeddings.storage.conventions import (
    _SPATIAL_CONVENTION,
    assert_encoder_matches,
    build_convention_attrs,
    build_geoemb_root_attrs,
    stamp_array_conventions,
    tile_id_to_epsg,
)

_PKG_VERSION = _dist_version("tessera_embeddings")
_V11_URL = encoder_url("v1.1")


def _attrs(**kwargs: object) -> dict:
    """``build_convention_attrs`` for a v1.1 store, the model these layout tests don't vary."""
    return build_convention_attrs(**{"model_version": "v1.1", **kwargs})  # type: ignore[arg-type]


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
        attrs = _attrs(
            tile_id="33UWP",
            total_y=100,
            total_x=100,
            embedding_dim=128,
            y_coords=y_coords,
            x_coords=x_coords,
            checkpoint_id="1.1",
        )

        # zarr_conventions should contain all three
        conventions = attrs["zarr_conventions"]
        names = [c["name"] for c in conventions]
        assert "proj" in names
        assert "spatial" in names
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
        # The origin is the OUTER CORNER of the first pixel, half a pixel back from the first
        # centre on each axis — `c` the western edge, `f` the northern edge — as the convention
        # requires and as GDAL/rasterio write it.
        assert attrs["spatial:transform"] == [10.0, 0.0, 499995.0, 0.0, -10.0, 6200005.0]
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
        assert attrs["geoemb:model"] == _V11_URL  # public encoder ref
        assert attrs["checkpoint_id"] == "1.1"  # a checkpoint id is provenance, not the public URL
        assert attrs["geoemb:source_data"] == [
            "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a",
            "https://doi.org/10.5067/SNWG/OPERA_L2_RTC-S1_V1",
        ]
        assert attrs["geoemb:data_type"] == "int8"
        assert attrs["geoemb:gsd"] == 10.0  # derived from the 10 m (metre CRS) coordinate spacing
        assert "geoemb:spatial_layout" not in attrs
        assert attrs["geoemb:build_version"] == _PKG_VERSION  # software/package version, not the encoder
        quant = attrs["geoemb:quantization"]
        assert quant["method"] == "per_pixel_scale"
        assert quant["original_dtype"] == "float32"
        assert quant["quantized_dtype"] == "int8"
        assert quant["scale"] == {"type": "array", "array_name": "scales", "nodata": "NaN"}

    def test_no_tile_id_no_coords_skips_proj_spatial(self) -> None:
        attrs = _attrs(
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
        assert "proj" not in names
        assert "spatial" not in names
        assert "geoemb:" in names

    def test_non_mgrs_tile_id_omits_proj(self) -> None:
        """When tile_id isn't an MGRS tile, proj: fields are omitted."""
        y = np.arange(100.0, 0.0, -10.0)
        x = np.arange(0.0, 100.0, 10.0)
        attrs = _attrs(
            tile_id="kenya_highlands",
            total_y=10,
            total_x=10,
            embedding_dim=128,
            y_coords=y,
            x_coords=x,
        )
        assert "proj:code" not in attrs
        names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj" not in names
        # spatial: still present since coords are provided
        assert "spatial" in names

    def test_model_is_public_ref_build_is_package_checkpoint_is_provenance(self) -> None:
        """geoemb:model is the PUBLIC encoder reference (the model's URL), NOT the
        supplied checkpoint id; build_version is the package version; the checkpoint
        id (an internal filename in production) is recorded as checkpoint_id.
        """
        attrs = _attrs(
            total_y=10,
            total_x=10,
            embedding_dim=128,
            checkpoint_id="best_model_fsdp_20250608_220648_QAT",  # a checkpoint stem, as prod passes
        )
        assert attrs["geoemb:model"] == _V11_URL  # NOT the checkpoint
        assert attrs["checkpoint_id"] == "best_model_fsdp_20250608_220648_QAT"
        assert attrs["geoemb:build_version"] == _PKG_VERSION
        # No checkpoint id supplied -> no checkpoint_id attr.
        assert "checkpoint_id" not in _attrs(total_y=10, total_x=10, embedding_dim=128)

    def test_model_url_tracks_the_encoder_family(self) -> None:
        """A v2 store must not advertise v1.1's public URL.

        The checkpoint stem and the public URL are separate provenance, so the
        stem alone cannot select the URL — the family has to be threaded.
        """
        v2 = _attrs(
            total_y=10, total_x=10, embedding_dim=128, checkpoint_id="v2_student_large", model_version="v2-large"
        )
        assert v2["geoemb:model"] == encoder_url("v2-large")
        assert v2["geoemb:model"] != encoder_url("v1.1")
        assert v2["checkpoint_id"] == "v2_student_large"  # stem stays separate provenance
        # An explicit URL still wins over the family lookup.
        explicit = _attrs(
            total_y=10, total_x=10, embedding_dim=128, model_version="v2-large", model_url="https://example/x"
        )
        assert explicit["geoemb:model"] == "https://example/x"
        # Provenance is never guessed: no model, or an unknown one, errors rather than defaulting.
        with pytest.raises(ValueError, match="needs a model_version"):
            build_convention_attrs(total_y=10, total_x=10, embedding_dim=128)
        with pytest.raises(ValueError, match="Invalid model_version"):
            _attrs(total_y=10, total_x=10, embedding_dim=128, model_version="v2_student_large")

    def test_gsd_derived_from_coordinate_spacing(self) -> None:
        """Gsd reflects the actual pixel size, not the nominal default."""
        y = np.arange(1000.0, 800.0, -20.0)  # 20 m spacing
        x = np.arange(0.0, 200.0, 20.0)
        attrs = _attrs(tile_id="33UWP", total_y=10, total_x=10, embedding_dim=128, y_coords=y, x_coords=x)
        assert attrs["geoemb:gsd"] == 20.0

    def test_gsd_omitted_for_geographic_crs(self) -> None:
        """A geographic CRS (EPSG:4326, degrees) with no explicit gsd OMITS the
        field entirely — never a false metre value from degree spacing.
        """
        y = np.arange(10.0, 9.0, -0.1)  # 0.1 degree spacing
        x = np.arange(0.0, 1.0, 0.1)
        attrs = _attrs(epsg_code="EPSG:4326", total_y=10, total_x=10, embedding_dim=128, y_coords=y, x_coords=x)
        assert "geoemb:gsd" not in attrs  # not 0.1, not a nominal default — absent

    def test_gsd_omitted_without_coords_or_explicit_value(self) -> None:
        """No coords and no explicit gsd → omit (no trustworthy metric value)."""
        assert "geoemb:gsd" not in _attrs(total_y=10, total_x=10, embedding_dim=128)

    def test_gsd_uses_explicit_value_when_supplied(self) -> None:
        """An explicit gsd the caller vouches for IS emitted (a trustworthy value)."""
        attrs = _attrs(total_y=10, total_x=10, embedding_dim=128, gsd=30.0)
        assert attrs["geoemb:gsd"] == 30.0

    def test_model_url_overrides_derived_public_ref(self) -> None:
        """A caller can pass the exact public model URI for its encoder; otherwise
        the URL derives from the model (never the checkpoint id).
        """
        explicit = _attrs(total_y=10, total_x=10, embedding_dim=128, model_url="https://geotessera.org/model/1.0")
        assert explicit["geoemb:model"] == "https://geotessera.org/model/1.0"
        derived = _attrs(total_y=10, total_x=10, embedding_dim=128, checkpoint_id="ckpt_stem")
        assert derived["geoemb:model"] == _V11_URL

    def test_epsg_code_overrides_tile_id(self) -> None:
        """When both tile_id and epsg_code are provided, epsg_code wins."""
        attrs = _attrs(
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
        attrs = _attrs(
            epsg_code="EPSG:5070",
            total_y=10,
            total_x=10,
            embedding_dim=128,
        )
        assert attrs["proj:code"] == "EPSG:5070"
        assert "proj:wkt2" in attrs
        assert "proj:projjson" in attrs
        names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "proj" in names

    def test_single_pixel_coords_skips_spatial(self) -> None:
        """spatial: requires at least 2 coordinate values to derive a transform."""
        attrs = _attrs(
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


class TestTransformAndBboxAgree:
    """The transform origin and the bbox describe the SAME grid edges, at every shape.

    The published store shipped with an origin at the first pixel's CENTRE while its bbox was
    already edge-based, so the two attrs disagreed by half a pixel on all 120 zone groups and
    every consumer trusting the transform placed the imagery half a cell to the south-east.
    Pinning the transform's expected value alone did not catch it — the wrong value was pinned.
    What catches it is the RELATIONSHIP, which holds whatever the resolution or the sign of the
    axis, so these assertions do not need updating when either changes.
    """

    #: Absolute, and far tighter than a half pixel at any resolution here. `approx`'s default is
    #: RELATIVE (1e-6), which on a 6.2 M metre northing is ±6.2 m — so the half-pixel error this
    #: class exists to catch would pass every assertion below unnoticed.
    TOL: ClassVar[dict[str, float]] = {"abs": 1e-6, "rel": 0.0}

    @staticmethod
    def _attrs(y: np.ndarray, x: np.ndarray) -> dict:
        return _attrs(
            epsg_code="EPSG:32633",
            total_y=len(y),
            total_x=len(x),
            embedding_dim=128,
            y_coords=y,
            x_coords=x,
        )

    @pytest.mark.parametrize(
        "res, y0, x0, n",
        [
            (10.0, 6200000.0, 500000.0, 100),  # 10 m UTM, the campaign's grid
            (30.0, 4000000.0, 300000.0, 7),  # coarser, and an odd count
            (0.5, 100.25, 200.25, 4),  # sub-metre, where a half-pixel is 0.25
        ],
    )
    def test_origin_is_the_bbox_corner_not_the_first_centre(self, res: float, y0: float, x0: float, n: int) -> None:
        """``c`` is the western bbox edge and ``f`` the northern one, for a north-up grid."""
        y = y0 - np.arange(n) * res  # descending: row 0 is the top
        x = x0 + np.arange(n) * res
        attrs = self._attrs(y, x)
        transform, bbox = attrs["spatial:transform"], attrs["spatial:bbox"]

        assert transform[2] == pytest.approx(bbox[0], **self.TOL), "transform c must be the bbox's western edge"
        assert transform[5] == pytest.approx(bbox[3], **self.TOL), "transform f must be the bbox's northern edge"
        # And each sits exactly half a pixel off the first CENTRE, which is what the store
        # shipped. Asserted as the offset rather than as "not equal", so the test states where
        # the origin IS and not merely one place it is not.
        assert float(x[0]) - transform[2] == pytest.approx(res / 2, **self.TOL)
        assert transform[5] - float(y[0]) == pytest.approx(res / 2, **self.TOL)

    def test_bbox_spans_exactly_shape_times_resolution_from_the_origin(self) -> None:
        """Walking ``shape`` pixels from the origin lands on the far bbox edge, to the float.

        The check an outside consumer would make, and the one that fails on a half-pixel error
        at either end: an off-by-half origin leaves the far edge half a pixel short.
        """
        y = 6200000.0 - np.arange(40) * 10.0
        x = 500000.0 + np.arange(25) * 10.0
        attrs = self._attrs(y, x)
        a, _, c, _, e, f = attrs["spatial:transform"]
        height, width = attrs["spatial:shape"]
        xmin, ymin, xmax, ymax = attrs["spatial:bbox"]

        assert (c, f) == pytest.approx((xmin, ymax), **self.TOL)
        assert c + a * width == pytest.approx(xmax, **self.TOL)
        assert f + e * height == pytest.approx(ymin, **self.TOL)

    def test_a_south_up_grid_steps_back_along_its_own_sign(self) -> None:
        """An ascending Y axis puts the origin BELOW the first centre, not above it.

        The half-pixel is taken along each axis's own signed resolution rather than as an
        absolute value, so a grid stored bottom-row-first lands on its own leading edge instead
        of half a pixel inside the data.
        """
        y = 6199000.0 + np.arange(20) * 10.0  # ascending
        x = 500000.0 + np.arange(20) * 10.0
        attrs = self._attrs(y, x)
        transform, bbox = attrs["spatial:transform"], attrs["spatial:bbox"]

        assert transform[4] == pytest.approx(10.0, **self.TOL)  # positive e: not north-up
        assert transform[5] == pytest.approx(6199000.0 - 5.0, **self.TOL)  # below the first centre
        assert transform[5] == pytest.approx(bbox[1], **self.TOL)  # which for this grid is ymin


class TestConventionRegistration:
    """The registration block a consumer follows to find the spec."""

    def test_registered_urls_are_the_ones_each_schema_requires_and_resolve(self) -> None:
        """proj/spatial pin ``v0.1``, the tag their schemas name; geoemb carries no URL at all.

        Offline by design — this asserts the pin, not that GitHub is up. geoemb's schema names a
        ``v1`` tag upstream has not cut, so any URL there either 404s or fails the schema's
        ``const``; the uuid alone registers it until the tag exists.
        """
        attrs = _attrs(
            epsg_code="EPSG:32633",
            total_y=10,
            total_x=10,
            embedding_dim=128,
            y_coords=np.arange(6200000.0, 6199900.0, -10.0),
            x_coords=np.arange(500000.0, 500100.0, 10.0),
        )
        registered = {c["name"]: c for c in attrs["zarr_conventions"]}
        for name in ("proj", "spatial"):
            for url_field in ("schema_url", "spec_url"):
                url = registered[name][url_field]
                assert f"zarr-conventions/{name}/" in url and "/v0.1/" in url, f"{name} {url_field}"
        assert {"schema_url", "spec_url"}.isdisjoint(registered["geoemb:"])
        assert registered["geoemb:"]["uuid"] == "61c12cc5-0e28-4056-999a-480cf3fb7e4c"


class TestMultiGroupPlacement:
    """Root-vs-per-zone geoemb placement for the utm_zones campaign store."""

    def test_include_geoemb_false_omits_geoemb(self) -> None:
        """A per-zone call emits only proj:/spatial: — geoemb: is stated on the root."""
        y = np.arange(6200000.0, 6199000.0, -10.0)
        x = np.arange(500000.0, 501000.0, 10.0)
        attrs = _attrs(
            epsg_code="EPSG:32633",
            total_y=100,
            total_x=100,
            embedding_dim=128,
            y_coords=y,
            x_coords=x,
            include_geoemb=False,
        )
        assert "geoemb:type" not in attrs
        names = [c["name"] for c in attrs["zarr_conventions"]]
        assert "geoemb:" not in names
        assert {"proj", "spatial"} <= set(names)
        assert attrs["proj:code"] == "EPSG:32633"

    def test_root_attrs_are_geoemb_only(self) -> None:
        """The root builder emits geoemb: (no proj:/spatial:) with an explicit gsd."""
        attrs = build_geoemb_root_attrs(
            embedding_dim=128,
            gsd=10.0,
            model_version="v1.1",
            checkpoint_id="best_model_fsdp_20250608",
        )
        assert attrs["geoemb:type"] == "pixel"
        assert attrs["geoemb:dimensions"] == 128
        assert attrs["geoemb:gsd"] == 10.0
        assert attrs["geoemb:model"] == _V11_URL
        assert attrs["checkpoint_id"] == "best_model_fsdp_20250608"  # provenance, not the URL
        assert [c["name"] for c in attrs["zarr_conventions"]] == ["geoemb:"]
        assert "proj:code" not in attrs
        assert "spatial:dimensions" not in attrs


# ── the multi-group root must state the same family the single-store root does ──


def test_the_global_root_stamps_the_encoder_family_it_was_seeded_for() -> None:
    """A campaign store states its encoder ONCE, at the root. Left unversioned it stamped v1.1's
    URL over a v2 seed — wrong provenance, and the cross-family gate then refuses the very append
    the store was seeded for, which is the failure being self-inflicted rather than caught.
    """
    v2 = build_geoemb_root_attrs(embedding_dim=128, model_version="v2-large")
    v11 = build_geoemb_root_attrs(embedding_dim=128, model_version="v1.1")
    assert v2["geoemb:model"] == encoder_url("v2-large")
    assert v11["geoemb:model"] == _V11_URL
    assert v2["geoemb:model"] != v11["geoemb:model"], "the two families publish different URLs"
    with pytest.raises(ValueError, match="needs a model_version"):
        build_geoemb_root_attrs(embedding_dim=128)


def test_one_comparison_serves_every_reader_of_the_encoder_identity() -> None:
    """Assembly and the pre-flight both gate on this. Two copies is how they come to disagree
    about one store, so the shared function is asserted directly.
    """
    assert_encoder_matches(encoder_url("v1.1"), model_version="v1.1", where="s")
    assert_encoder_matches(None, model_version="v1.1", where="s")  # absence dates the store
    assert_encoder_matches(encoder_url("v2-large"), model_version="v2-large", where="s")

    for published, wanted in ((encoder_url("v1.1"), "v2-large"), (None, "v2-large"), (encoder_url("v2-large"), "v1.1")):
        with pytest.raises(ValueError, match="Refusing to append"):
            assert_encoder_matches(published, model_version=wanted, where="s")


@pytest.mark.parametrize(
    ("epsg", "real_coords", "labelled"),
    [("EPSG:32633", True, True), ("EPSG:32633", False, False), ("EPSG:4326", True, False), (None, True, False)],
)
def test_cf_coord_attrs_only_claim_metres_for_real_metre_coords(
    epsg: str | None, real_coords: bool, labelled: bool
) -> None:
    """Labelled only when the group declares real coordinates (``spatial:dimensions``) in a metre
    ``proj:code``: pixel indices under a tile-derived CRS, a degree CRS, or no CRS stay unlabelled.
    """
    node = zarr.group(store=zarr.storage.MemoryStore())
    if epsg:
        node.attrs["proj:code"] = epsg
    if real_coords:
        node.attrs["spatial:dimensions"] = ["northing", "easting"]
    for name in ("northing", "easting"):
        node.create_array(name, data=np.arange(3.0), dimension_names=(name,))
    stamp_array_conventions(node)
    assert dict(node["easting"].attrs) == (
        {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"} if labelled else {}
    )
    assert dict(node["northing"].attrs).get("axis") == ("Y" if labelled else None)


def test_spatial_arrays_repeat_the_group_spatial_dimensions() -> None:
    """Arrays spanning both spatial dims register ``spatial`` beside any registration they hold."""
    node = zarr.group(store=zarr.storage.MemoryStore())
    node.attrs["spatial:dimensions"] = ["northing", "easting"]
    scales = node.create_array("scales", shape=(1, 2, 2), dtype="f4", dimension_names=("time", "northing", "easting"))
    scales.attrs["zarr_conventions"] = [{"uuid": "other"}]
    node.create_array("time", data=np.arange(1.0), dimension_names=("time",))
    stamp_array_conventions(node)
    stamp_array_conventions(node)  # idempotent: no duplicate registration
    assert node["scales"].attrs["spatial:dimensions"] == ["northing", "easting"]
    assert [c["uuid"] for c in node["scales"].attrs["zarr_conventions"]] == ["other", _SPATIAL_CONVENTION["uuid"]]
    assert dict(node["time"].attrs) == {}
