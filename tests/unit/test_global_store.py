"""Group-aware seeding + opens for the global store (W2)."""

from __future__ import annotations

from tessera_embeddings.config.inference import EMBEDDING_DIM
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
    assert set(root.group_keys()) >= {"32601", "32701"}


def test_data_vars_are_metadata_only(tmp_path):
    # embeddings/scales must be all-fill (no chunk objects) — cost-free seeding.
    store = _seed(tmp_path)
    g = zarr_store.open_store_as_zarr_group(store, group="32601")
    assert g["embeddings"].nchunks_initialized == 0
    assert g["scales"].nchunks_initialized == 0
    assert g["embeddings"].shape == (3, 4096, 2048, EMBEDDING_DIM)


def test_embeddings_is_sharded(tmp_path):
    store = _seed(tmp_path)
    g = zarr_store.open_store_as_zarr_group(store, group="32601")
    emb = g["embeddings"]
    assert emb.shards == (1, 2048, 2048, EMBEDDING_DIM)
    assert emb.chunks == (1, 256, 256, EMBEDDING_DIM)
    assert g["scales"].shards == (1, 2048, 2048)


def test_coords_and_attrs(tmp_path):
    store = _seed(tmp_path)
    g = zarr_store.open_store_as_zarr_group(store, group="32601")
    assert g["easting"].shape == (2048,)
    assert g["northing"].shape == (4096,)
    assert g["time"].shape == (3,)
    assert g["band"].shape == (EMBEDDING_DIM,)  # band coord present for xarray consumers
    attrs = dict(g.attrs)
    assert attrs["crs"] == "EPSG:32601"
    assert attrs["years_complete"] == []
    assert attrs["zone_scheme"] == "utm_6deg_nominal"
    assert attrs["time_convention"] == "calendar_year"
    assert attrs["proj:code"] == "EPSG:32601"  # GeoZarr conventions merged in


def test_group_aware_open_store_reads_one_zone(tmp_path):
    store = _seed(tmp_path)
    ds = zarr_store.open_store(store, chunks=None, group="32701")
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
    assert set(root.group_keys()) >= {"32601", "32701"}


def test_global_store_config_has_time_split_and_preload():
    cfg = zarr_store.global_store_config()
    assert cfg.manifest is not None
    assert cfg.manifest.splitting is not None
    assert cfg.manifest.preload is not None
