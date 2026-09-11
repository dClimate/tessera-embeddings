"""Reader-side audit of a global store: layout conformance, shard coverage, live-pixel sampling."""

from __future__ import annotations

import icechunk
import numpy as np
import pytest
import zarr

from tessera_embeddings.config.store_layout import GLOBAL, SHARD_PX
from tessera_embeddings.storage import global_store, published_store, zarr_store
from tessera_embeddings.storage.zone_grid import PIXEL_M, ZoneSpec

# Two shards of easting by four of northing, so a shard grid exists without a large store.
_ZONE = ZoneSpec("32601", "N", 1, (0.0, 2 * SHARD_PX * PIXEL_M), (0.0, 4 * SHARD_PX * PIXEL_M))
_YEARS = (2024, 2025)


@pytest.fixture
def seeded(tmp_path):
    """A two-year, one-zone global store with nothing written into its data arrays."""
    path = str(tmp_path / "global.icechunk")
    repo = global_store.create_global_repo(path)
    global_store.seed_zone_groups(repo, [_ZONE], years=_YEARS)
    return path


@pytest.fixture
def writable_zone(seeded):
    """The seeded zone group on a writable session, for a test that has to break it.

    Yielded rather than returned so the session stays alive for the duration of the test: dropping
    it invalidates the group built on its store.
    """
    session = global_store.open_global_repo(seeded).writable_session("main")
    yield zarr.open_group(session.store, mode="r+")["01N"]


def _fill_shard(path: str, zone: str, time_index: int, shard: tuple[int, int], *, live_fraction: float = 1.0) -> None:
    """Write one shard of `scales` and `embeddings`, leaving `1 - live_fraction` of it at fill."""
    session = global_store.open_global_repo(path).writable_session("main")
    group = zarr.open_group(session.store, mode="r+")[zone]
    scales, embeddings = group["scales"], group["embeddings"]
    y0, y1, x0, x1 = published_store.shard_pixel_window(shard, scales.shape)
    block = np.full((y1 - y0, x1 - x0), np.nan, dtype="float32")
    live_rows = int((y1 - y0) * live_fraction)
    block[:live_rows, :] = 0.5
    scales[time_index, y0:y1, x0:x1] = block
    embeddings[time_index, y0:y1, x0:x1, :] = 7
    session.commit(f"fill {zone} t{time_index} shard {shard}")


def _recreate(group: zarr.Group, var: str, **overrides) -> None:
    """Recreate `var` exactly as the layout would, with `overrides` applied.

    Everything the layout declares comes from the writer's own ``create_kwargs``, so a case below
    breaks the ONE property it names and nothing else. Array attributes are not part of
    ``create_kwargs`` — the seeder applies them separately — so passing no overrides at all
    reproduces an array created without them.
    """
    kwargs = GLOBAL.for_var(var).create_kwargs(tuple(group[var].shape))
    group.create_array(var, overwrite=True, **{**kwargs, **overrides})


#: One property of one array broken per case, against the array the seeder actually wrote.
#: ``None`` overrides delete the array instead of recreating it.
_LAYOUT_CASES = (
    # Why the names are checked at all: `easting, northing` has the right rank, the right dtype
    # and — both spatial chunk sizes being 256 — the right chunk geometry, so a labelled read
    # comes back transposed with every other check passing.
    ("transposed-dimension-names", "scales", {"dimension_names": ("time", "easting", "northing")}, "dimension names"),
    ("no-dimension-names", "scales", {"dimension_names": None}, "no dimension names"),
    ("wrong-dtype", "scales", {"dtype": "float64"}, "dtype is float64"),
    # Every coverage question reads a finite `scales` value as written data, so a finite fill hides
    # never-written pixels behind the right dtype, chunks and shards.
    ("finite-fill", "scales", {"fill_value": 0.0}, "fill value is"),
    # The codec decides the store's size and read speed.
    ("wrong-codec", "scales", {"serializer": "auto"}, "codec is 'raw'"),
    ("unsharded", "scales", {"shards": None}, "shards are None"),
    # `dtype="bool"` on an int8 array is how xarray presents booleans; without it a labelled reader
    # gets 0 and 1. Hence no overrides: the layout's attrs are applied outside `create_kwargs`.
    ("missing-declared-attribute", "s2_month_covered", {}, "attribute dtype=None"),
    ("declared-array-deleted", "s2_obs_count", None, "absent from the group"),
    # Losing `northing` or `month` would otherwise only skip that dimension's extent check, so the
    # audit reports nothing while a labelled read of the zone is incomplete. These two are the
    # easiest to leave out of a hand-written "required coordinates" set, which is why the audit
    # asks the seeder's own definition.
    ("coordinate-deleted", "northing", None, "does not have it"),
    ("month-coordinate-deleted", "month", None, "does not have it"),
)


class TestLayoutDepartures:
    """Comparing a live zone group against the declared global layout."""

    def test_a_seeded_zone_conforms_to_the_global_layout(self, seeded):
        # Also what pins that the six COORDINATE arrays, which are in the group and in no layout,
        # are not reported: six spurious departures per zone would bury every real finding.
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        assert published_store.layout_departures(group) == []

    @pytest.mark.parametrize(
        ("var", "overrides", "phrase"),
        [pytest.param(var, over, phrase, id=name) for name, var, over, phrase in _LAYOUT_CASES],
    )
    def test_one_broken_property_is_reported_and_nothing_else_is(self, writable_zone, var, overrides, phrase):
        if overrides is None:
            del writable_zone[var]
        else:
            _recreate(writable_zone, var, **overrides)
        departures = published_store.layout_departures(writable_zone)
        assert any(phrase in d for d in departures), f"nothing mentioned {phrase!r}: {departures}"
        assert all(var in d for d in departures), f"only {var} was broken, yet: {departures}"

    def test_an_array_shorter_than_its_coordinate_is_reported(self, writable_zone):
        # Expectations come from the array's own shape, so truncating it by a whole shard passes
        # every geometry check; the coordinate arrays are the independent oracle.
        time, northing, easting = writable_zone["scales"].shape
        _recreate(writable_zone, "scales", shape=(time, northing - SHARD_PX, easting))
        departures = published_store.layout_departures(writable_zone)
        assert any("northing" in d and "coordinate array" in d for d in departures), departures


class TestLiveShards:
    """Enumerating which shards of a zone-year hold data."""

    def test_a_seeded_store_has_no_live_shards(self, seeded):
        repo = global_store.open_global_repo(seeded)
        assert published_store.live_shards(repo.readonly_session(branch="main"), "01N") == {}

    def test_written_shards_are_reported_per_time_index_on_the_shard_grid(self, seeded):
        # The values pin the SHARD grid, not the inner-chunk grid: one 2048-px shard holds 64 inner
        # chunks, so an inner-chunk enumeration would answer (8*sy + k, 8*sx + k) instead.
        _fill_shard(seeded, "01N", 0, (1, 0))
        _fill_shard(seeded, "01N", 0, (2, 1))
        _fill_shard(seeded, "01N", 1, (1, 0))
        repo = global_store.open_global_repo(seeded)
        coverage = published_store.live_shards(repo.readonly_session(branch="main"), "01N")
        assert coverage == {0: frozenset({(1, 0), (2, 1)}), 1: frozenset({(1, 0)})}


class TestShardPixelWindow:
    """Converting a shard coordinate back to a pixel window."""

    @pytest.mark.parametrize(
        ("shard", "shape", "expected"),
        [
            # (y0, y1, x0, x1): the y window is rows 2*pitch..3*pitch, the x window 3..4*pitch.
            pytest.param(
                (2, 3),
                (9, 100_000, 100_000),
                (2 * SHARD_PX, 3 * SHARD_PX, 3 * SHARD_PX, 4 * SHARD_PX),
                id="an-interior-shard-is-the-pitch-square",
            ),
            pytest.param(
                (1, 1),
                (9, SHARD_PX + 700, SHARD_PX + 300),
                (SHARD_PX, SHARD_PX + 700, SHARD_PX, SHARD_PX + 300),
                id="an-edge-shard-is-clamped-to-the-array",
            ),
        ],
    )
    def test_the_window_of_a_shard(self, shard, shape, expected):
        assert published_store.shard_pixel_window(shard, shape) == expected


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

    def test_fill_inside_a_live_shard_is_filtered_out(self, seeded):
        # THE reason this function exists: a candidate on fill is read without leaving the
        # process, so a latency benchmark built on them reports free reads as real ones.
        _fill_shard(seeded, "01N", 0, (1, 1), live_fraction=0.5)
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        pixels = published_store.sample_live_pixels(group, 0, [(1, 1)], 12, seed=11, oversample=8)
        assert pixels, "the live half of the shard should still yield pixels"
        dead = [(y, x) for y, x in pixels if y >= SHARD_PX + SHARD_PX // 2]
        assert not dead, f"sampled pixels in the shard's fill half: {dead}"

    def test_the_sample_is_reproducible_for_a_seed(self, seeded):
        _fill_shard(seeded, "01N", 0, (1, 0))
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        first = published_store.sample_live_pixels(group, 0, [(1, 0)], 5, seed=7)
        second = published_store.sample_live_pixels(group, 0, [(1, 0)], 5, seed=7)
        assert first == second


#: Stand-in for the credential callback the AWS provider registers process-wide.
_PROVIDER = object()
#: The two keys that decide WHOSE identity the read is made with. Absence is as load-bearing as
#: presence — passing both leaves the choice up to icechunk.
_IDENTITY_KEYS = ("anonymous", "get_credentials")


@pytest.fixture
def s3_kwargs(monkeypatch):
    """What `_create_storage` hands `icechunk.s3_storage`, instead of building real storage."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(zarr_store.icechunk, "s3_storage", lambda **kw: captured.update(kw))
    return captured


class TestAnonymousStorage:
    """Opening an S3 store with no credentials — the path an outside consumer reads on."""

    @pytest.mark.parametrize(
        ("provider", "anonymous", "expected"),
        [
            pytest.param(None, True, {"anonymous": True}, id="anonymous-asks-icechunk-for-anonymous"),
            pytest.param(_PROVIDER, False, {"get_credentials": _PROVIDER}, id="a-registered-provider-is-forwarded"),
            # The shape a bare test process sees, and it must not be mistaken for the anonymous path.
            pytest.param(None, False, {}, id="no-provider-leaves-icechunks-own-chain-to-answer"),
        ],
    )
    def test_which_identity_reaches_icechunk(self, s3_kwargs, monkeypatch, provider, anonymous, expected):
        monkeypatch.setattr(zarr_store, "_default_credentials_provider", provider)
        zarr_store._create_storage("s3://bucket/prefix", anonymous=anonymous)
        assert {k: v for k, v in s3_kwargs.items() if k in _IDENTITY_KEYS} == expected

    def test_anonymous_with_a_credential_callback_is_refused(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            zarr_store._create_storage("s3://bucket/prefix", get_credentials=lambda: None, anonymous=True)

    def test_a_local_path_ignores_anonymous(self, tmp_path):
        # `anonymous` is an S3 concept; a local store must not be refused or altered by it.
        storage = zarr_store._create_storage(str(tmp_path / "local.icechunk"), anonymous=True)
        assert storage is not None

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


#: One property of one spatial or calendar axis broken per case. ``None`` deletes the axis; the rest
#: rewrite its values from what the seeder wrote.
_COORDINATE_CASES = (
    # Northing descends; ascending is the same values, length and spacing with every pixel in the
    # wrong place.
    ("northing-reversed", "northing", lambda values: values[::-1], "northing: starts"),
    ("easting-shifted-one-pixel", "easting", lambda values: values + PIXEL_M, "easting: starts"),
    (
        "easting-at-twice-the-pixel-spacing",
        "easting",
        lambda values: values[0] + np.arange(values.size) * 2 * PIXEL_M,
        "easting: starts",
    ),
    ("easting-deleted", "easting", None, "easting: absent"),
    # `month` is compared by VALUE: a zero-based or reordered axis has the right rank and extent
    # while `sel(month=7)` selects the wrong plane.
    ("month-zero-based", "month", lambda values: values - 1, "month: holds"),
    ("month-reversed", "month", lambda values: values[::-1], "month: holds"),
)


class TestCoordinateDepartures:
    """Whether the grid a zone is laid on puts its pixels where the CRS says they are."""

    def test_a_seeded_zone_matches_its_zone_grid(self, seeded):
        group = zarr_store.open_store_as_zarr_group(seeded, group="01N")
        assert published_store.coordinate_departures(group, _ZONE) == []

    @pytest.mark.parametrize(
        ("axis", "rewrite", "phrase"),
        [pytest.param(axis, rewrite, phrase, id=name) for name, axis, rewrite, phrase in _COORDINATE_CASES],
    )
    def test_a_misplaced_axis_is_reported_rather_than_raising(self, writable_zone, axis, rewrite, phrase):
        if rewrite is None:
            del writable_zone[axis]
        else:
            array = writable_zone[axis]
            array[:] = rewrite(np.asarray(array[:]))
        departures = published_store.coordinate_departures(writable_zone, _ZONE)
        assert any(phrase in d for d in departures), f"nothing mentioned {phrase!r}: {departures}"


class TestSavedManifestPreload:
    """The switch writers use: it changes the store, so every reader inherits the change."""

    def test_a_new_store_is_created_with_preloading_on(self, seeded):
        # On for a fill, which is about to touch those manifests anyway.
        repo = global_store.open_global_repo(seeded)
        assert repo.config.manifest.preload.max_total_refs > 0

    def test_switching_it_off_reaches_the_store_and_this_packages_opener(self, seeded):
        global_store.set_saved_manifest_preload(seeded, enabled=False)
        # Raw icechunk first: the zeroes are IN THE STORE, not just in a handle we built.
        saved = icechunk.Repository.open(zarr_store._create_storage(seeded)).config.manifest.preload
        assert (saved.max_total_refs, saved.max_arrays_to_scan) == (0, 0)
        # And `open_global_repo` hands Icechunk no config of its own — handing it
        # `global_store_config()` is how readers used to get the writer's preload back.
        assert global_store.open_global_repo(seeded).config.manifest.preload.max_total_refs == 0

    def test_switching_it_back_on_restores_the_writers_budget(self, seeded):
        global_store.set_saved_manifest_preload(seeded, enabled=False)
        global_store.set_saved_manifest_preload(seeded, enabled=True)
        expected = zarr_store.global_store_config().manifest.preload.max_total_refs
        assert global_store.open_global_repo(seeded).config.manifest.preload.max_total_refs == expected

    def test_tags_and_the_branch_tip_survive_the_switch(self, seeded):
        # The write rebuilds the object that holds them, so this is what must not move. No mutation
        # of this package turns it red — Icechunk is what preserves them — and it is kept anyway,
        # as the guard that an Icechunk upgrade has not stopped doing so.
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
