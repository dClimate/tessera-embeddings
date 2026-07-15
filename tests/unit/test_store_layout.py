"""StoreLayout presets: create_array kwargs + a sharded round-trip."""

from __future__ import annotations

import icechunk
import numpy as np
import pytest
import zarr

from tessera_embeddings.config.inference import EMBEDDING_DIM
from tessera_embeddings.config.store_layout import GLOBAL, OBS_COUNT_VARS, SINGLE


def test_global_embeddings_geometry():
    k = GLOBAL.for_var("embeddings").create_kwargs((9, 4096, 4096, EMBEDDING_DIM))
    assert k["chunks"] == (1, 256, 256, EMBEDDING_DIM)
    assert k["shards"] == (1, 2048, 2048, EMBEDDING_DIM)
    assert k["dtype"] == np.dtype("int8")
    assert k["fill_value"] == 0
    assert k["dimension_names"] == ("time", "northing", "easting", "band")


def test_global_scales_is_sharded_pcodec():
    k = GLOBAL.for_var("scales").create_kwargs((9, 4096, 4096))
    assert k["chunks"] == (1, 256, 256)
    assert k["shards"] == (1, 2048, 2048)
    assert k["compressors"] is None
    assert type(k["serializer"]).__name__ == "PCodec"
    assert np.isnan(k["fill_value"])


def test_single_embeddings_unsharded():
    k = SINGLE.for_var("embeddings").create_kwargs((1, 500, 500, EMBEDDING_DIM))
    assert k["chunks"] == (1, 500, 500, EMBEDDING_DIM // 32)
    assert "shards" not in k


def test_obs_vars_resolve_and_unknown_raises():
    for var in OBS_COUNT_VARS:
        assert GLOBAL.for_var(var).dtype == "uint16"
    with pytest.raises(KeyError):
        GLOBAL.for_var("nonsense")


def test_chunks_and_shards_clamp_to_small_shape():
    # A store smaller than one nominal shard must still create cleanly.
    k = GLOBAL.for_var("embeddings").create_kwargs((1, 300, 300, EMBEDDING_DIM))
    assert k["chunks"] == (1, 256, 256, EMBEDDING_DIM)
    # shard clamped to a whole multiple of the (clamped) chunk within shape
    assert k["shards"][1] % k["chunks"][1] == 0
    assert k["shards"][1] <= 300


def test_sharded_roundtrip_and_fill(tmp_path):
    # embeddings (int8, sharded) + scales (pcodec float32, sharded) round-trip,
    # and unwritten inner chunks read back as fill.
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(tmp_path / "r")))
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    emb_kwargs = GLOBAL.for_var("embeddings").create_kwargs((1, 2048, 2048, EMBEDDING_DIM))
    emb = root.create_array("embeddings", **emb_kwargs)
    scl = root.create_array("scales", **GLOBAL.for_var("scales").create_kwargs((1, 2048, 2048)))
    edata = np.random.default_rng(0).integers(-127, 128, size=(1, 256, 256, EMBEDDING_DIM), dtype="int8")
    sdata = np.random.default_rng(1).random((1, 256, 256), dtype="float32")
    emb[0:1, 0:256, 0:256, :] = edata
    scl[0:1, 0:256, 0:256] = sdata
    session.commit("write one inner chunk of one shard")

    rs = repo.readonly_session(branch="main")
    r = zarr.open_group(rs.store, mode="r")
    assert np.array_equal(r["embeddings"][0, 0:256, 0:256, :], edata[0])
    assert np.allclose(r["scales"][0, 0:256, 0:256], sdata[0])
    assert (r["embeddings"][0, 300:400, 300:400, :] == 0).all()  # unwritten -> fill
    assert np.isnan(r["scales"][0, 300:400, 300:400]).all()
