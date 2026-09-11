"""Reader-side audit of a global store: layout conformance, shard coverage, live-pixel sampling."""

from __future__ import annotations

import icechunk
import numpy as np
import pytest
import zarr

from tessera_embeddings.config.store_layout import GLOBAL, SHARD_PX
from tessera_embeddings.storage import global_store, published_store, zarr_store
from tessera_embeddings.storage.zone_grid import ZoneSpec

# Two shards of easting by four of northing, so a shard grid exists without a large store.
_ZONE = ZoneSpec("32601", "N", 1, (0.0, 2 * SHARD_PX * 10.0), (0.0, 4 * SHARD_PX * 10.0))
_YEARS = (2024, 2025)


@pytest.fixture
def seeded(tmp_path):
    """A two-year, one-zone global store with nothing written into its data arrays."""
    path = str(tmp_path / "global.icechunk")
    repo = global_store.create_global_repo(path)
    global_store.seed_zone_groups(repo, [_ZONE], years=_YEARS)
    return path


def _writable_group(path: str, zone: str) -> tuple:
    """A writable session and one zone group on it — the session must outlive the group."""
    session = global_store.open_global_repo(path).writable_session("main")
    return session, zarr.open_group(session.store, mode="r+")[zone]


def _fill_shard(path: str, zone: str, time_index: int, shard: tuple[int, int], *, live_fraction: float = 1.0) -> None:
    """Write one shard of `scales` and `embeddings`, leaving `1 - live_fraction` of it at fill."""
    session, group = _writable_group(path, zone)
    scales, embeddings = group["scales"], group["embeddings"]
    y0, y1, x0, x1 = published_store.shard_pixel_window(shard, scales.shape)
    block = np.full((y1 - y0, x1 - x0), np.nan, dtype="float32")
    live_rows = int((y1 - y0) * live_fraction)
    block[:live_rows, :] = 0.5
    scales[time_index, y0:y1, x0:x1] = block
    embeddings[time_index, y0:y1, x0:x1, :] = 7
    session.commit(f"fill {zone} t{time_index} shard {shard}")


class TestLayoutDepartures:
    """Comparing a live zone group against the declared global layout."""

    def test_a_seeded_zone_conforms_to_the_global_layout(self, seeded):
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        assert published_store.layout_departures(group) == []

    def test_every_declared_array_is_actually_checked(self, seeded):
        # Or a conforming verdict means only "found none".
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        assert set(GLOBAL.arrays) <= set(dict(group.arrays()))

    def test_a_wrong_dtype_is_reported(self, seeded):
        _, group = _writable_group(seeded, "01N")
        group.create_array("scales", shape=group["scales"].shape, dtype="float64", overwrite=True)
        assert any("scales" in d and "dtype" in d for d in published_store.layout_departures(group))

    def test_an_unsharded_array_is_reported(self, seeded):
        _, group = _writable_group(seeded, "01N")
        group.create_array("scales", shape=group["scales"].shape, dtype="float32", chunks=(1, 256, 256), overwrite=True)
        assert any("scales" in d and "shards" in d for d in published_store.layout_departures(group))

    def test_a_missing_array_is_reported(self, seeded):
        _, group = _writable_group(seeded, "01N")
        del group["s2_obs_count"]
        assert any("s2_obs_count" in d and "absent" in d for d in published_store.layout_departures(group))

    def test_coordinate_arrays_are_not_departures(self, seeded):
        # The six coordinate arrays are in the group and in no layout; reporting them would bury
        # real findings under six per zone.
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        assert "time" in dict(group.arrays())
        assert published_store.layout_departures(group) == []


class TestLiveShards:
    """Enumerating which shards of a zone-year hold data."""

    def test_a_seeded_store_has_no_live_shards(self, seeded):
        repo = global_store.open_global_repo(seeded)
        assert published_store.live_shards(repo.readonly_session(branch="main"), "01N") == {}

    def test_written_shards_are_reported_per_time_index(self, seeded):
        _fill_shard(seeded, "01N", 0, (1, 0))
        _fill_shard(seeded, "01N", 0, (2, 1))
        _fill_shard(seeded, "01N", 1, (1, 0))
        repo = global_store.open_global_repo(seeded)
        coverage = published_store.live_shards(repo.readonly_session(branch="main"), "01N")
        assert coverage == {0: frozenset({(1, 0), (2, 1)}), 1: frozenset({(1, 0)})}

    def test_the_coordinates_are_shard_grid_not_inner_chunk_grid(self, seeded):
        # One 2048-px shard holds 64 inner chunks, so an inner-chunk enumeration would return 64
        # coordinates with indices up to (8*sy + 7).
        _fill_shard(seeded, "01N", 0, (3, 1))
        repo = global_store.open_global_repo(seeded)
        coverage = published_store.live_shards(repo.readonly_session(branch="main"), "01N")
        assert coverage == {0: frozenset({(3, 1)})}

    def test_embeddings_reports_the_same_shard_set_as_scales(self, seeded):
        _fill_shard(seeded, "01N", 0, (2, 0))
        repo = global_store.open_global_repo(seeded)
        session = repo.readonly_session(branch="main")
        assert published_store.live_shards(session, "01N", "embeddings") == published_store.live_shards(
            session, "01N", "scales"
        )


class TestShardPixelWindow:
    """Converting a shard coordinate back to a pixel window."""

    def test_a_window_is_the_shard_pitch_square(self):
        assert published_store.shard_pixel_window((2, 3), (9, 100_000, 100_000)) == (
            2 * SHARD_PX,
            3 * SHARD_PX,
            3 * SHARD_PX,
            4 * SHARD_PX,
        )
        # (y0, y1, x0, x1): the y window is rows 2*pitch..3*pitch, the x window 3*pitch..4*pitch.

    def test_an_edge_shard_is_clamped_to_the_array(self):
        y0, y1, x0, x1 = published_store.shard_pixel_window((1, 1), (9, SHARD_PX + 700, SHARD_PX + 300))
        assert (y0, y1) == (SHARD_PX, SHARD_PX + 700)
        assert (x0, x1) == (SHARD_PX, SHARD_PX + 300)


class TestSampleLivePixels:
    """Choosing probe pixels that provably hold embeddings."""

    def test_no_shards_yields_no_pixels(self, seeded):
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        assert published_store.sample_live_pixels(group, 0, [], 10) == []

    def test_sampled_pixels_all_hold_data(self, seeded):
        _fill_shard(seeded, "01N", 0, (1, 0))
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        pixels = published_store.sample_live_pixels(group, 0, [(1, 0)], 8, seed=3)
        assert len(pixels) == 8
        scales = group["scales"]
        assert all(np.isfinite(scales[0, y, x]) for y, x in pixels)

    def test_pixels_land_inside_the_requested_shard(self, seeded):
        _fill_shard(seeded, "01N", 0, (2, 1))
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        pixels = published_store.sample_live_pixels(group, 0, [(2, 1)], 6, seed=5)
        assert pixels
        for y, x in pixels:
            assert 2 * SHARD_PX <= y < 3 * SHARD_PX
            assert 1 * SHARD_PX <= x < 2 * SHARD_PX

    def test_fill_inside_a_live_shard_is_filtered_out(self, seeded):
        # THE reason this function exists: a candidate on fill is read without leaving the
        # process, so a latency benchmark built on them reports free reads as real ones.
        _fill_shard(seeded, "01N", 0, (1, 1), live_fraction=0.5)
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        pixels = published_store.sample_live_pixels(group, 0, [(1, 1)], 12, seed=11, oversample=8)
        assert pixels, "the live half of the shard should still yield pixels"
        assert all(y < SHARD_PX + SHARD_PX // 2 for y, _ in pixels)

    def test_the_sample_is_reproducible_for_a_seed(self, seeded):
        _fill_shard(seeded, "01N", 0, (1, 0))
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        first = published_store.sample_live_pixels(group, 0, [(1, 0)], 5, seed=7)
        second = published_store.sample_live_pixels(group, 0, [(1, 0)], 5, seed=7)
        assert first == second


class TestAnonymousStorage:
    """Opening an S3 store with no credentials — the path an outside consumer reads on."""

    def test_anonymous_asks_icechunk_for_anonymous_credentials(self, monkeypatch):
        captured: dict[str, object] = {}
        monkeypatch.setattr(zarr_store.icechunk, "s3_storage", lambda **kw: captured.update(kw))
        zarr_store._create_storage("s3://bucket/prefix", anonymous=True)
        assert captured["anonymous"] is True
        # ABSENT, not merely unused: passing both leaves the choice up to icechunk.
        assert "get_credentials" not in captured

    def test_the_default_path_forwards_the_registered_provider_and_never_asks_for_anonymous(self, monkeypatch):
        captured: dict[str, object] = {}
        provider = object()
        monkeypatch.setattr(zarr_store.icechunk, "s3_storage", lambda **kw: captured.update(kw))
        monkeypatch.setattr(zarr_store, "_default_credentials_provider", provider)
        zarr_store._create_storage("s3://bucket/prefix")
        assert captured.get("get_credentials") is provider
        assert "anonymous" not in captured

    def test_with_no_registered_provider_icechunks_own_chain_is_left_to_answer(self, monkeypatch):
        # The shape a bare test process sees, and it must not be mistaken for the anonymous path.
        captured: dict[str, object] = {}
        monkeypatch.setattr(zarr_store.icechunk, "s3_storage", lambda **kw: captured.update(kw))
        monkeypatch.setattr(zarr_store, "_default_credentials_provider", None)
        zarr_store._create_storage("s3://bucket/prefix")
        assert "get_credentials" not in captured
        assert "anonymous" not in captured

    def test_anonymous_with_a_credential_callback_is_refused(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            zarr_store._create_storage("s3://bucket/prefix", get_credentials=lambda: None, anonymous=True)

    def test_a_local_path_ignores_anonymous(self, tmp_path):
        # `anonymous` is an S3 concept; a local store must not be refused or altered by it.
        storage = zarr_store._create_storage(str(tmp_path / "local.icechunk"), anonymous=True)
        assert storage is not None


class TestLayoutDimensionNames:
    """Dimension names are part of the contract, and the one departure the geometry checks miss."""

    def test_transposed_spatial_axes_are_reported(self, seeded):
        # Why names are checked at all: `easting, northing` has the right rank, the right dtype
        # and — both spatial chunk sizes being 256 — the right chunk geometry.
        _, group = _writable_group(seeded, "01N")
        shape = group["scales"].shape
        group.create_array(
            "scales",
            shape=shape,
            dtype="float32",
            chunks=(1, 256, 256),
            shards=(1, 2048, 2048),
            dimension_names=("time", "easting", "northing"),
            overwrite=True,
        )
        departures = published_store.layout_departures(group)
        assert any("scales" in d and "dimension names" in d for d in departures)
        assert not any("chunks" in d or "shards" in d or "dtype" in d for d in departures), (
            "the point of this test is that only the NAMES are wrong"
        )

    def test_unnamed_dimensions_are_reported(self, seeded):
        _, group = _writable_group(seeded, "01N")
        group.create_array(
            "scales",
            shape=group["scales"].shape,
            dtype="float32",
            chunks=(1, 256, 256),
            shards=(1, 2048, 2048),
            overwrite=True,
        )
        assert any("no dimension names" in d for d in published_store.layout_departures(group))


class TestLayoutFillAndAttributes:
    """Fill value, declared attributes, and extents — the checks geometry alone cannot make."""

    def test_a_finite_fill_on_scales_is_reported(self, seeded):
        # Every coverage question reads a finite `scales` value as written data, so a finite fill
        # hides never-written pixels behind the right dtype, chunks and shards.
        _, group = _writable_group(seeded, "01N")
        group.create_array(
            "scales",
            shape=group["scales"].shape,
            dtype="float32",
            chunks=(1, 256, 256),
            shards=(1, 2048, 2048),
            dimension_names=("time", "northing", "easting"),
            fill_value=0.0,
            overwrite=True,
        )
        departures = published_store.layout_departures(group)
        assert any("scales" in d and "fill value" in d for d in departures)
        assert not any("chunks" in d or "shards" in d or "dimension names" in d for d in departures)

    def test_nan_fill_is_not_reported_as_a_departure(self, seeded):
        # float("nan") != float("nan"), and zarr returns a numpy scalar rather than a Python float,
        # so a naive comparison calls every conforming store broken.
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        assert not any("fill value" in d for d in published_store.layout_departures(group))

    def test_a_missing_declared_attribute_is_reported(self, seeded):
        # `dtype="bool"` on an int8 array is how xarray presents booleans; without it a labelled
        # reader gets 0 and 1.
        _, group = _writable_group(seeded, "01N")
        covered = group["s2_month_covered"]
        group.create_array(
            "s2_month_covered",
            shape=covered.shape,
            dtype="int8",
            chunks=covered.chunks,
            shards=covered.shards,
            dimension_names=("time", "northing", "easting", "month"),
            fill_value=covered.fill_value,
            overwrite=True,
        )
        departures = published_store.layout_departures(group)
        assert any("s2_month_covered" in d and "dtype" in d and "bool" in d for d in departures)

    def test_an_array_shorter_than_its_coordinate_is_reported(self, seeded):
        # Expectations come from the array's own shape, so truncating it by a whole shard passes
        # every geometry check; the coordinate arrays are the independent oracle.
        _, group = _writable_group(seeded, "01N")
        scales = group["scales"]
        short = (scales.shape[0], scales.shape[1] - SHARD_PX, scales.shape[2])
        group.create_array(
            "scales",
            shape=short,
            dtype="float32",
            chunks=(1, 256, 256),
            shards=(1, 2048, 2048),
            dimension_names=("time", "northing", "easting"),
            fill_value=float("nan"),
            overwrite=True,
        )
        departures = published_store.layout_departures(group)
        assert any("northing" in d and "coordinate array" in d for d in departures)


class TestLayoutCoordinatesAndCodec:
    """Two more departures the geometry checks cannot see: a lost coordinate, and a wrong codec."""

    def test_a_missing_coordinate_array_is_reported(self, seeded):
        # Otherwise losing `northing` only skips that dimension's extent check, so the audit
        # reports nothing while every geospatial read of the zone is incomplete.
        _, group = _writable_group(seeded, "01N")
        del group["northing"]
        assert any("northing" in d and "does not have it" in d for d in published_store.layout_departures(group))

    def test_a_missing_month_coordinate_is_reported(self, seeded):
        # The two easiest to leave out of a hand-written "required coordinates" set, which is why
        # the audit asks the seeder's own definition.
        _, group = _writable_group(seeded, "01N")
        del group["month"]
        assert any("month" in d and "does not have it" in d for d in published_store.layout_departures(group))

    def test_scales_without_its_declared_codec_is_reported(self, seeded):
        _, group = _writable_group(seeded, "01N")
        group.create_array(
            "scales",
            shape=group["scales"].shape,
            dtype="float32",
            chunks=(1, 256, 256),
            shards=(1, 2048, 2048),
            dimension_names=("time", "northing", "easting"),
            fill_value=float("nan"),
            compressors=None,
            serializer="auto",
            overwrite=True,
        )
        departures = published_store.layout_departures(group)
        assert any("scales" in d and "codec" in d for d in departures)


class TestAnonymousWithAnOverride:
    """`anonymous` must not be quietly satisfied by an installed S3 override."""

    def test_an_installed_override_refuses_an_anonymous_read(self, monkeypatch):
        # The override carries credentials and an endpoint, so honouring it would make an
        # anonymous-access check pass while authenticating — the one thing it rules out.
        class _Override:
            def make_storage(self, prefix_override=None):
                raise AssertionError("the override must not be consulted for an anonymous read")

        monkeypatch.setattr(zarr_store, "_s3_config_override", _Override())
        with pytest.raises(ValueError, match="anonymous"):
            zarr_store._create_storage("s3://bucket/prefix", anonymous=True)

    def test_an_installed_override_still_serves_a_credentialed_read(self, monkeypatch):
        sentinel = object()

        class _Override:
            def make_storage(self, prefix_override=None):
                return sentinel

        monkeypatch.setattr(zarr_store, "_s3_config_override", _Override())
        assert zarr_store._create_storage("s3://bucket/prefix") is sentinel


class TestCoordinateDepartures:
    """Whether the grid a zone is laid on puts its pixels where the CRS says they are."""

    def test_a_seeded_zone_matches_its_zone_grid(self, seeded):
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        assert published_store.coordinate_departures(group, _ZONE) == []

    def test_a_reversed_northing_axis_is_reported(self, seeded):
        # Northing descends; ascending is the same values, length and spacing with every pixel in
        # the wrong place.
        _, group = _writable_group(seeded, "01N")
        northing = np.asarray(group["northing"][:])
        group["northing"][:] = northing[::-1]
        departures = published_store.coordinate_departures(group, _ZONE)
        assert any("northing" in d for d in departures)

    def test_an_easting_shifted_by_one_pixel_is_reported(self, seeded):
        _, group = _writable_group(seeded, "01N")
        group["easting"][:] = np.asarray(group["easting"][:]) + 10.0
        assert any("easting" in d for d in published_store.coordinate_departures(group, _ZONE))

    def test_a_wrong_pixel_spacing_is_reported(self, seeded):
        _, group = _writable_group(seeded, "01N")
        easting = np.asarray(group["easting"][:])
        group["easting"][:] = easting[0] + (np.arange(easting.size) + 0.5) * 20.0
        assert any("easting" in d for d in published_store.coordinate_departures(group, _ZONE))

    def test_a_missing_axis_is_reported_rather_than_raising(self, seeded):
        _, group = _writable_group(seeded, "01N")
        del group["easting"]
        assert any("easting" in d and "absent" in d for d in published_store.coordinate_departures(group, _ZONE))


class TestMonthCoordinate:
    """The `month` axis is compared by value: `sel(month=7)` has to mean July."""

    def test_a_zero_based_month_axis_is_reported(self, seeded):
        _, group = _writable_group(seeded, "01N")
        group["month"][:] = np.arange(12, dtype="int16")
        assert any("month" in d for d in published_store.coordinate_departures(group, _ZONE))

    def test_a_reordered_month_axis_is_reported(self, seeded):
        _, group = _writable_group(seeded, "01N")
        months = np.asarray(group["month"][:])
        group["month"][:] = months[::-1]
        assert any("month" in d for d in published_store.coordinate_departures(group, _ZONE))

    def test_the_seeded_month_axis_is_not_a_departure(self, seeded):
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        assert not any("month" in d for d in published_store.coordinate_departures(group, _ZONE))


class TestSavedManifestPreload:
    """The switch writers use: it changes the store, so every reader inherits the change."""

    def test_a_new_store_is_created_with_preloading_on(self, seeded):
        # On for a fill, which is about to touch those manifests anyway.
        repo = global_store.open_global_repo(seeded)
        assert repo.config.manifest.preload.max_total_refs > 0

    def test_switching_it_off_is_visible_to_a_reader_that_passes_no_config(self, seeded):
        global_store.set_saved_manifest_preload(seeded, enabled=False)
        preload = icechunk.Repository.open(zarr_store._create_storage(seeded)).config.manifest.preload
        assert preload.max_total_refs == 0
        assert preload.max_arrays_to_scan == 0

    def test_switching_it_back_on_restores_the_writers_budget(self, seeded):
        global_store.set_saved_manifest_preload(seeded, enabled=False)
        global_store.set_saved_manifest_preload(seeded, enabled=True)
        expected = zarr_store.global_store_config().manifest.preload.max_total_refs
        assert global_store.open_global_repo(seeded).config.manifest.preload.max_total_refs == expected

    def test_manifest_splitting_and_storage_settings_survive_the_switch(self, seeded):
        # Splitting describes the manifests already on disk; building a fresh config to change the
        # preload is how it gets dropped.
        before = global_store.open_global_repo(seeded).config
        global_store.set_saved_manifest_preload(seeded, enabled=False)
        after = global_store.open_global_repo(seeded).config
        assert repr(after.manifest.splitting) == repr(before.manifest.splitting)
        assert repr(after.storage) == repr(before.storage)

    def test_tags_and_the_branch_tip_survive_the_switch(self, seeded):
        # The write rebuilds the object that holds them, so this is what must not move.
        repo = global_store.open_global_repo(seeded)
        session = repo.writable_session("main")
        zarr.open_group(session.store, mode="r+")["01N"].attrs["years_complete"] = [2025]
        snapshot = session.commit("mark 01N year 2025 complete")
        repo.create_tag("zone-01N-2025", snapshot)
        before = ({t: str(repo.lookup_tag(t)) for t in repo.list_tags()}, str(repo.lookup_branch("main")))
        global_store.set_saved_manifest_preload(seeded, enabled=False)
        after_repo = global_store.open_global_repo(seeded)
        assert {t: str(after_repo.lookup_tag(t)) for t in after_repo.list_tags()} == before[0]
        assert str(after_repo.lookup_branch("main")) == before[1]

    def test_it_raises_rather_than_returning_quietly_if_the_switch_does_not_take(self, seeded, monkeypatch):
        monkeypatch.setattr(icechunk.Repository, "save_config", lambda self: None)
        with pytest.raises(RuntimeError, match="preload did not change"):
            global_store.set_saved_manifest_preload(seeded, enabled=False)


class TestOpenInheritsTheStoredConfig:
    """`open_global_repo` passes no config, which is what makes the switch above reach everyone."""

    def test_opening_does_not_override_what_the_store_saved(self, seeded):
        global_store.set_saved_manifest_preload(seeded, enabled=False)
        # If this handed Icechunk `global_store_config()`, the preload would read back on.
        assert global_store.open_global_repo(seeded).config.manifest.preload.max_total_refs == 0
