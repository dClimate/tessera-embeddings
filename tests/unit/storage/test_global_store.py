"""Group-aware seeding + opens for the global store (W2)."""

from __future__ import annotations

import dataclasses

import icechunk
import numpy as np
import pytest
import zarr

from tessera_embeddings.config.inference import EMBEDDING_DIM, MODEL_FAMILIES, checkpoint_filename
from tessera_embeddings.config.store_layout import GLOBAL
from tessera_embeddings.storage import global_store, published_store, zarr_store
from tessera_embeddings.storage.zone_grid import ZoneSpec

# Tiny stand-in zones (2048x4096 px = a couple shards) so coord arrays stay small.
_ZA = ZoneSpec("32601", "N", 1, (0.0, 20_480.0), (0.0, 40_960.0))
_ZB = ZoneSpec("32701", "S", 1, (0.0, 20_480.0), (1_105_920.0, 1_146_880.0))


@pytest.fixture
def seeded(seeded_store):
    """Both stand-in zones, three years: a private copy per test."""
    return seeded_store([_ZA, _ZB], years=(2023, 2024, 2025))


@pytest.mark.parametrize(("model", "names"), [("v1.1", ("northing", "easting")), ("v2-large", ("y", "x"))])
def test_each_model_seeds_its_own_spatial_names(tmp_path, model, names):
    """v1.1 keeps northing/easting, v2 takes y/x, and the published-store audit accepts both."""
    repo = global_store.create_global_repo(str(tmp_path / "global.icechunk"))
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,), model_version=model)
    g = zarr_store.open_store_as_zarr_group(str(tmp_path / "global.icechunk"), group="01N")
    y, x = names
    assert (g[y].shape, g[x].shape) == ((4096,), (2048,))
    assert g["embeddings"].metadata.dimension_names == ("time", y, x, "band")
    assert g.attrs["spatial:dimensions"] == [y, x] and g[y].attrs["axis"] == "Y"
    assert published_store.layout_departures(g) == []
    assert published_store.coordinate_departures(g, _ZA) == []


def test_extending_a_store_keeps_its_names(tmp_path, monkeypatch):
    """A zone added to a store seeded with other names takes the store's, never mixing two."""
    store = str(tmp_path / "global.icechunk")
    repo = global_store.create_global_repo(store)
    v2 = MODEL_FAMILIES["v2-large"]
    monkeypatch.setitem(MODEL_FAMILIES, "v2-large", dataclasses.replace(v2, spatial_dims=("northing", "easting")))
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,), model_version="v2-large")
    MODEL_FAMILIES["v2-large"] = v2
    global_store.seed_zone_groups(repo, [_ZB], years=(2025,), model_version="v2-large")
    g = zarr_store.open_store_as_zarr_group(store, group="01S")
    assert g["scales"].metadata.dimension_names == ("time", "northing", "easting")
    assert "y" not in g


def test_data_vars_are_metadata_only(seeded):
    # embeddings/scales must be all-fill (no chunk objects) — cost-free seeding.
    store = seeded
    g = zarr_store.open_store_as_zarr_group(store, group="01N")
    assert g["embeddings"].nchunks_initialized == 0
    assert g["scales"].nchunks_initialized == 0
    assert g["embeddings"].shape == (3, 4096, 2048, EMBEDDING_DIM)


def test_coords_and_attrs(seeded):
    store = seeded
    g = zarr_store.open_store_as_zarr_group(store, group="01N")
    assert g["easting"].shape == (2048,)
    assert g["northing"].shape == (4096,)
    assert g["time"].shape == (3,)
    assert g["band"].shape == (EMBEDDING_DIM,)  # band coord present for xarray consumers
    assert g["northing"].attrs["standard_name"] == "projection_y_coordinate"  # CF axis attrs for CF-only readers
    assert g["easting"].attrs["axis"] == "X"
    # CF time bounds: (time, 2), linked via time.attrs["bounds"], seeded once to each
    # slot's calendar year, half-open so consecutive years leave no gap.
    assert g["time_bnds"].shape == (3, 2)
    assert g["time"].attrs["bounds"] == "time_bnds"
    yrs = np.asarray(g["time"]).astype("datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
    bnds = np.asarray(g["time_bnds"]).astype("datetime64[ns]").astype("datetime64[D]").astype(str)
    for i, y in enumerate(yrs):
        assert list(bnds[i]) == [f"{y}-01-01", f"{y + 1}-01-01"]
    # Contiguous: each slot's upper bound IS the next slot's lower bound.
    assert [b[1] for b in bnds[:-1]] == [b[0] for b in bnds[1:]]
    attrs = dict(g.attrs)
    assert attrs["crs"] == "EPSG:32601"
    assert attrs["years_complete"] == []
    assert attrs["zone_scheme"] == "utm_6deg_nominal"
    assert attrs["time_convention"] == "calendar_year"  # a GUARANTEE (zone-fill gate enforces Jan-Dec)
    assert attrs["proj:code"] == "EPSG:32601"  # per-zone proj: merged in
    # geoemb: lives once on the root (utm_zones layout), not per zone.
    assert "geoemb:type" not in attrs
    zone_conventions = [c["name"] for c in attrs["zarr_conventions"]]
    assert "geoemb:" not in zone_conventions
    assert {"proj", "spatial"} <= set(zone_conventions)


def test_root_carries_geoemb_convention(seeded):
    """Encoder/quantization provenance sits once on the root group;
    zones carry only their own proj:/spatial:.
    """
    store = seeded
    attrs = dict(zarr_store.open_store_as_zarr_group(store).attrs)
    assert attrs["geoemb:type"] == "pixel"
    assert attrs["geoemb:dimensions"] == EMBEDDING_DIM
    assert "geoemb:spatial_layout" not in attrs  # both schema values misdescribe 120 hemisphere groups
    assert attrs["geoemb:gsd"] == 10.0  # fixed 10 m grid across all zones
    assert attrs["geoemb:data_type"] == "int8"
    assert "geoemb:build_version" in attrs
    assert [c["name"] for c in attrs["zarr_conventions"]] == ["geoemb:"]
    # proj:/spatial: are per-zone, not on the root.
    assert "proj:code" not in attrs


def test_a_utm_zones_store_declares_its_layout_and_refuses_the_other(tmp_path):
    """Columns are named utm01..utm60 in the northern CRS, and only that store claims utm_zones."""
    store = str(tmp_path / "global.icechunk")
    repo = global_store.create_global_repo(store)
    column = ZoneSpec("32601", None, 1, (0.0, 20_480.0), (-20_480.0, 20_480.0))
    global_store.seed_zone_groups(repo, [column], years=(2025,), model_version="v2-large")
    root = zarr_store.open_store_as_zarr_group(store)
    assert root.attrs["geoemb:spatial_layout"] == "utm_zones"
    assert list(root.group_keys()) == ["utm01"] and root["utm01"].attrs["proj:code"] == "EPSG:32601"
    with pytest.raises(ValueError, match="one store holds one layout"):
        global_store.seed_zone_groups(repo, [_ZA], years=(2025,), model_version="v2-large")
    # A store seeded before the root carried an identity declares its layout when it is stamped.
    session = repo.writable_session("main")
    unstamped = zarr.open_group(session.store, mode="a")
    for key in list(unstamped.attrs):
        del unstamped.attrs[key]
    session.commit("strip the identity")
    global_store.stamp_root_identity(repo, model_version="v2-large")
    assert zarr_store.open_store_as_zarr_group(store).attrs["geoemb:spatial_layout"] == "utm_zones"


def test_group_aware_open_store_reads_one_zone(seeded):
    store = seeded
    ds = zarr_store.open_store(store, chunks=None, group="01S")
    assert "embeddings" in ds.data_vars
    assert ds.sizes["time"] == 3
    # time axis decodes to calendar years
    years = ds["time"].dt.year.values
    assert list(years) == [2023, 2024, 2025]
    ds.close()


def test_an_unstamped_store_keeps_its_identity_when_nothing_is_created(seeded_store):
    """The stamp is written to the SESSION, so a call that creates nothing must still commit it.

    A reseed of already-complete zones on a store predating the root identity added `_root_attrs` to
    the session and then returned the branch tip without committing — so the call reported success,
    the root stayed unstamped, and the fill gates (which pass on an ABSENT attr) then accepted
    anything while the operator believed the store had been stamped.
    """
    store = seeded_store([_ZA], years=(2025,))
    repo = global_store.open_global_repo(store)

    # A store predating the root identity. Simulated by removing the attr, since every seed since
    # then stamps one — which is exactly why such a store can only arrive from an older build.
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    del root.attrs["geoemb:model"]
    session.commit("drop the root identity, as a store predating it would have")
    assert "geoemb:model" not in zarr_store.open_store_as_zarr_group(store).attrs, "the premise"

    # The same zone again — complete, so nothing is created — but now asking for an identity.
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,), optical_min_obs=15)

    root = zarr_store.open_store_as_zarr_group(store)
    assert "geoemb:model" in root.attrs, "the stamp must have been committed, not discarded"
    assert root.attrs["checkpoint_id"] == checkpoint_filename(model_version="v1.1"), "the model's own checkpoint"
    assert root.attrs["optical_min_obs"] == 15


def test_a_group_missing_a_later_schema_array_is_refused_not_skipped(seeded_store):
    """The completeness check must name every array the seeder writes.

    `month` and `time_bnds` were absent from it, so a group holding the layout arrays and the four
    other coordinates counted as complete without them — and the reseed reported idempotent success
    while leaving `s2_month_covered` with no calendar labels and the time axis with no CF bounds.
    """
    store = seeded_store([_ZA], years=(2025,))
    repo = global_store.open_global_repo(store)

    # Simulate a group seeded by a schema that predates `month`.
    session = repo.writable_session("main")
    zarr.open_group(session.store, mode="a")[_ZA.group_name].__delitem__("month")
    session.commit("drop month, as an older seeder would have left it")

    with pytest.raises(ValueError, match=r"missing.*month"):
        global_store.seed_zone_groups(repo, [_ZA], years=(2025,))


@pytest.mark.parametrize("model", ["v1.1", "v2-large"])
def test_the_store_names_the_model_that_fills_it(seeded_store, model):
    """Every fill takes its model from the store's root, and runs that model's checkpoint."""
    store = seeded_store([_ZA], years=(2025,), model_version=model)
    assert global_store.store_model_version(store) == model


@pytest.mark.parametrize(
    ("attr", "value", "match"),
    [
        ("geoemb:model", "https://example.org/custom", "names no known model"),
        ("checkpoint_id", "tessera_v1_1_mpc_encoder.pt", "checkpoint"),
    ],
)
def test_a_store_no_fill_can_run_is_refused_before_anything_is_provisioned(seeded_store, attr, value, match):
    """A root naming an unknown model, or a checkpoint other than its model's, is one no fill runs."""
    store = seeded_store([_ZA], years=(2025,))
    session = global_store.open_global_repo(store).writable_session("main")
    zarr.open_group(session.store, mode="a").attrs[attr] = value
    session.commit("a root no fill can run")
    with pytest.raises(ValueError, match=match):
        global_store.store_model_version(store)


@pytest.mark.parametrize(
    ("seeded_with", "reseeded_with"),
    [
        # Or the root would advertise the second model over already-seeded zones.
        pytest.param({"model_version": "v1.1"}, {"model_version": "v2-large"}, id="different-model"),
        # Zones filled under one minimum depth must not sit beside zones filled under another, under a
        # root advertising only the second: moving the line is a new store, not a migration.
        pytest.param({"optical_min_obs": 30}, {"optical_min_obs": 20}, id="different-minimum-depth"),
        # Every store seeded so far has no rule; stamping one on retrospectively would claim its
        # existing zones were filled under a line never applied to them.
        pytest.param({}, {"optical_min_obs": 30}, id="rule-added-to-a-store-without-one"),
    ],
)
def test_the_root_identity_is_write_once(seeded_store, seeded_with, reseeded_with):
    """A matching incremental seed adds its group; one carrying a different identity is refused."""
    repo = global_store.open_global_repo(seeded_store([_ZA], years=(2025,), **seeded_with))
    global_store.seed_zone_groups(repo, [_ZB], years=(2025,), **seeded_with)  # same identity: adds the group
    with pytest.raises(ValueError, match="write-once"):
        global_store.seed_zone_groups(repo, [_ZB], years=(2025,), **reseeded_with)


def test_reseed_with_different_axis_rejected(seeded_store):
    """The time axis is fixed + uniform across groups (ADR-008 D1): a direct
    incremental seed with a different `years` than already-seeded groups is
    rejected at the helper (not only in the seed_global_store flow).
    """
    store = seeded_store([_ZA], years=(2024, 2025))
    repo = global_store.open_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZB], years=(2024, 2025))  # same axis: fine
    with pytest.raises(ValueError, match="existing axis"):
        global_store.seed_zone_groups(repo, [_ZB], years=(2025,))


_EMB = GLOBAL.arrays["embeddings"]


@pytest.mark.parametrize(
    ("embeddings", "match"),
    [
        pytest.param(dataclasses.replace(_EMB, shards=(1, 1024, 1024, EMBEDDING_DIM)), "shards: have", id="shards"),
        # Geometry alone is not the schema: a float32 zone passes a pitch check, then the int8
        # staging writer refuses to fill it.
        pytest.param(dataclasses.replace(_EMB, dtype="float32"), "dtype: have", id="dtype"),
    ],
)
def test_reseed_with_a_different_layout_rejected(seeded_store, embeddings, match):
    """Layout is fixed across groups for the same reason the axis is.

    `plan_zone_inference` pins the inference tile to the group's shard pitch, so a zone seeded with
    another layout is unfillable — and only rejected at fill time, long after seeding committed it.
    Refuse the mixed seed instead.
    """
    repo = global_store.open_global_repo(seeded_store([_ZA], years=(2025,)))
    layout = dataclasses.replace(GLOBAL, arrays={**GLOBAL.arrays, "embeddings": embeddings})
    with pytest.raises(ValueError, match=match):
        global_store.seed_zone_groups(repo, [_ZB], years=(2025,), layout=layout)


def test_seeding_rejects_duplicate_or_unordered_years(tmp_path):
    """The axis is fixed at seeding, so a malformed `years` is unrepairable.

    Duplicates are the dangerous shape: `time_index_of` always resolves to the first
    of two identical coordinates, so the second slot is never written while
    `years_complete` reports the pair done — permanently empty and invisible.
    """
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    for bad in [(2025, 2025), (2025, 2024), ()]:
        with pytest.raises(ValueError, match="strictly increasing"):
            global_store.seed_zone_groups(repo, [_ZA], years=bad)


def test_a_store_with_no_rule_carries_no_attr_rather_than_zero(seeded_store):
    """Absent and zero are different statements. Zero is a threshold that refuses nothing;
    absent is a store that never had a rule, which is every store seeded before 2026-08-13.
    Recording zero for the second would let a later reader believe a rule was applied.
    """
    store = seeded_store([_ZA], years=(2025,))
    root = zarr_store.open_store_as_zarr_group(store)
    assert "optical_min_obs" not in root.attrs


def test_a_rule_that_refuses_nothing_is_refused(tmp_path):
    """Zero would be stamped as a configured rule on a store that has none, permanently, and
    the write-once identity means it could never be corrected. Caught at the seeder instead.
    """
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    with pytest.raises(ValueError, match="refuses nothing"):
        global_store.seed_zone_groups(repo, [_ZA], years=(2025,), optical_min_obs=0)


def test_seeding_a_zone_that_already_exists_is_a_no_op(seeded_store):
    """A matching reseed is allowed by the identity and layout checks, then used to fail.

    `require_group` returns the existing group and `create_array` raises on its first array —
    and it raises PART-WAY through `specs`, so a direct incremental call carrying both an old
    and a new zone seeded neither the old one (correctly) nor the remaining new one (a loss).

    So seeding the same zone twice must be a no-op, and a call carrying a new zone alongside
    an existing one must still seed the new one.
    """
    store = seeded_store([_ZA], years=(2025,))
    repo = global_store.open_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA, _ZB], years=(2025,))
    root = zarr_store.open_store_as_zarr_group(store)
    assert set(root.group_keys()) >= {"01N", "01S"}, "the new zone must land beside the existing one"

    # And a request with NOTHING new must not fail. Skipping complete groups means no writes,
    # and icechunk refuses an empty commit unless asked — so committing regardless turned the
    # idempotent reseed this skip exists to allow into a hard failure.
    snapshot = global_store.seed_zone_groups(repo, [_ZA, _ZB], years=(2025,))
    assert isinstance(snapshot, str) and snapshot, "an all-seeded reseed returns the current tip"


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

    def test_it_raises_rather_than_returning_quietly_if_the_switch_does_not_take(self, seeded, monkeypatch):
        monkeypatch.setattr(icechunk.Repository, "save_config", lambda self: None)
        with pytest.raises(RuntimeError, match="preload did not change"):
            global_store.set_saved_manifest_preload(seeded, enabled=False)
