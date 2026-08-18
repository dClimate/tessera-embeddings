"""Group-aware seeding + opens for the global store (W2)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tessera_embeddings.config.inference import EMBEDDING_DIM
from tessera_embeddings.config.store_layout import GLOBAL
from tessera_embeddings.storage import global_store, zarr_store
from tessera_embeddings.storage.zone_grid import ZoneSpec

# Tiny stand-in zones (2048x4096 px = a couple shards) so coord arrays stay small.
_ZA = ZoneSpec("32601", "N", 1, (0.0, 20_480.0), (0.0, 40_960.0))
_ZB = ZoneSpec("32701", "S", 1, (0.0, 20_480.0), (1_105_920.0, 1_146_880.0))


def _seed(tmp_path):
    store = str(tmp_path / "global.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA, _ZB], years=(2023, 2024, 2025))
    return store


def test_sibling_groups_seeded_without_clobber(tmp_path):
    store = _seed(tmp_path)
    root = zarr_store.open_store_as_zarr_group(store)
    assert set(root.group_keys()) >= {"01N", "01S"}


def test_data_vars_are_metadata_only(tmp_path):
    # embeddings/scales must be all-fill (no chunk objects) — cost-free seeding.
    store = _seed(tmp_path)
    g = zarr_store.open_store_as_zarr_group(store, group="01N")
    assert g["embeddings"].nchunks_initialized == 0
    assert g["scales"].nchunks_initialized == 0
    assert g["embeddings"].shape == (3, 4096, 2048, EMBEDDING_DIM)


def test_embeddings_is_sharded(tmp_path):
    store = _seed(tmp_path)
    g = zarr_store.open_store_as_zarr_group(store, group="01N")
    emb = g["embeddings"]
    assert emb.shards == (1, 2048, 2048, EMBEDDING_DIM)
    assert emb.chunks == (1, 256, 256, EMBEDDING_DIM)
    assert g["scales"].shards == (1, 2048, 2048)


def test_coords_and_attrs(tmp_path):
    store = _seed(tmp_path)
    g = zarr_store.open_store_as_zarr_group(store, group="01N")
    assert g["easting"].shape == (2048,)
    assert g["northing"].shape == (4096,)
    assert g["time"].shape == (3,)
    assert g["band"].shape == (EMBEDDING_DIM,)  # band coord present for xarray consumers
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
    assert {"proj:", "spatial:"} <= set(zone_conventions)


def test_root_carries_geoemb_convention(tmp_path):
    """Encoder/quantization provenance sits once on the root group (utm_zones layout);
    zones carry only their own proj:/spatial:.
    """
    store = _seed(tmp_path)
    attrs = dict(zarr_store.open_store_as_zarr_group(store).attrs)
    assert attrs["geoemb:type"] == "pixel"
    assert attrs["geoemb:dimensions"] == EMBEDDING_DIM
    assert attrs["geoemb:spatial_layout"] == "utm_zones"
    assert attrs["geoemb:gsd"] == 10.0  # fixed 10 m grid across all zones
    assert attrs["geoemb:data_type"] == "int8"
    assert "geoemb:build_version" in attrs
    assert [c["name"] for c in attrs["zarr_conventions"]] == ["geoemb:"]
    # proj:/spatial: are per-zone, not on the root.
    assert "proj:code" not in attrs


def test_group_aware_open_store_reads_one_zone(tmp_path):
    store = _seed(tmp_path)
    ds = zarr_store.open_store(store, chunks=None, group="01S")
    assert "embeddings" in ds.data_vars
    assert ds.sizes["time"] == 3
    # time axis decodes to calendar years
    years = ds["time"].dt.year.values
    assert list(years) == [2023, 2024, 2025]
    ds.close()


def test_second_seed_adds_groups_without_clobbering_first(tmp_path):
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,))
    global_store.seed_zone_groups(repo, [_ZB], years=(2025,))
    root = zarr_store.open_store_as_zarr_group(store)
    assert set(root.group_keys()) >= {"01N", "01S"}


def test_reseed_with_different_model_rejected(tmp_path):
    """Root encoder provenance is write-once: a matching reseed is a no-op, but a
    partial reseed carrying a DIFFERENT encoder is rejected — otherwise the root
    would advertise the new model and the fill-time model gate would permit mixing
    it with already-seeded zones.
    """
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,), model_version="v1")
    # Same model on a later incremental seed: fine (adds the group, provenance no-op).
    global_store.seed_zone_groups(repo, [_ZB], years=(2025,), model_version="v1")
    # A different model is rejected before any group work.
    with pytest.raises(ValueError, match="write-once"):
        global_store.seed_zone_groups(repo, [_ZB], years=(2025,), model_version="v2")


def test_reseed_with_different_axis_rejected(tmp_path):
    """The time axis is fixed + uniform across groups (ADR-008 D1): a direct
    incremental seed with a different `years` than already-seeded groups is
    rejected at the helper (not only in the seed_global_store flow).
    """
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA], years=(2024, 2025))
    global_store.seed_zone_groups(repo, [_ZB], years=(2024, 2025))  # same axis: fine
    with pytest.raises(ValueError, match="existing axis"):
        global_store.seed_zone_groups(repo, [_ZB], years=(2025,))


def test_reseed_with_different_shard_geometry_rejected(tmp_path):
    """Layout is fixed across groups for the same reason the axis is.

    `plan_zone_inference` pins the inference tile to the group's shard pitch, so zones
    seeded at a different pitch are unfillable — and they are only rejected at fill
    time, long after seeding committed them. Refuse the mixed seed instead.
    """
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,))
    global_store.seed_zone_groups(repo, [_ZB], years=(2025,))  # same layout: fine

    emb = GLOBAL.arrays["embeddings"]
    halved = dataclasses.replace(
        GLOBAL, arrays={**GLOBAL.arrays, "embeddings": dataclasses.replace(emb, shards=(1, 1024, 1024, EMBEDDING_DIM))}
    )
    with pytest.raises(ValueError, match="shards: have"):
        global_store.seed_zone_groups(repo, [_ZB], years=(2025,), layout=halved)


def test_reseed_with_same_geometry_but_different_dtype_rejected(tmp_path):
    """Geometry alone is not the schema. Matching chunks and shards with a different
    dtype passes a pitch check and then creates zones the int8 staging writer refuses
    to fill — a heterogeneous store that only shows up at fill time.
    """
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,))

    emb = GLOBAL.arrays["embeddings"]
    as_float = dataclasses.replace(
        GLOBAL, arrays={**GLOBAL.arrays, "embeddings": dataclasses.replace(emb, dtype="float32")}
    )
    with pytest.raises(ValueError, match="dtype: have"):
        global_store.seed_zone_groups(repo, [_ZB], years=(2025,), layout=as_float)


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


def test_global_store_config_has_time_split_and_preload():
    cfg = zarr_store.global_store_config()
    assert cfg.manifest is not None
    assert cfg.manifest.splitting is not None
    assert cfg.manifest.preload is not None


def test_the_minimum_depth_rule_is_stamped_on_the_root(tmp_path):
    """A user must be able to ask "what rule produced this dataset" without reading provenance
    per cell, and a fill must be able to check the rule it is about to apply against the one the
    store advertises. Both need it on the root.
    """
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,), optical_min_obs=25)
    root = zarr_store.open_store_as_zarr_group(store)
    assert root.attrs["optical_min_obs"] == 25


def test_a_store_with_no_rule_carries_no_attr_rather_than_zero(tmp_path):
    """Absent and zero are different statements. Zero is a threshold that refuses nothing;
    absent is a store that never had a rule, which is every store seeded before 2026-08-13.
    Recording zero for the second would let a later reader believe a rule was applied.
    """
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,))
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


def test_reseed_with_a_different_minimum_depth_rejected(tmp_path):
    """The consequence of putting the rule in the write-once identity, and the reason it is
    there: zones filled under one minimum depth must not end up beside zones filled under
    another, under a root advertising only the second. It also means the rule can never be
    changed for this store — moving the line is a new store, not a migration.
    """
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,), optical_min_obs=30)
    # The same rule on a later incremental seed: fine, provenance is a no-op.
    global_store.seed_zone_groups(repo, [_ZB], years=(2025,), optical_min_obs=30)
    with pytest.raises(ValueError, match="write-once"):
        global_store.seed_zone_groups(repo, [_ZB], years=(2025,), optical_min_obs=20)


def test_adding_a_rule_to_a_store_seeded_without_one_is_rejected(tmp_path):
    """The direction that matters most in practice: every store seeded so far has no rule, and
    stamping one onto it retrospectively would claim its existing zones were filled under a
    line that was never applied to them. A re-stamp is a deliberate act on a fresh store.
    """
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_ZA], years=(2025,))
    with pytest.raises(ValueError, match="write-once"):
        global_store.seed_zone_groups(repo, [_ZB], years=(2025,), optical_min_obs=30)
