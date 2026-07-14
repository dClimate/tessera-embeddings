"""Shard-aligned land-masked writer + commit discipline (W3)."""

from __future__ import annotations

import numpy as np
import zarr

from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.storage import global_store, zarr_store
from tessera_embeddings.storage.shard_writer import (
    SemaphoreCommitGate,
    commit_with_rebase,
    write_year_shards,
)
from tessera_embeddings.storage.zone_grid import ZoneSpec

_BAND = 8  # small band for a light test
_SHARD = 512
_CHUNK = 256

# Small layout: 512^2 shards of 256^2 inner chunks - whole-shard blocks are a few MB.
_EMB = ArrayLayout(DIMS_4D, (1, _CHUNK, _CHUNK, _BAND), "int8", 0, "zstd", shards=(1, _SHARD, _SHARD, _BAND))
_SCL = ArrayLayout(DIMS_3D, (1, _CHUNK, _CHUNK), "float32", float("nan"), "pcodec", shards=(1, _SHARD, _SHARD))
SMALL = StoreLayout(name="small", arrays={"embeddings": _EMB, "scales": _SCL})

# Zone spanning 2 shards tall x 1 wide (1024 x 512 px).
_ZONE = ZoneSpec("32601", "N", 1, (0.0, 5_120.0), (0.0, 10_240.0))
_ZONE_B = ZoneSpec("32701", "S", 1, (0.0, 5_120.0), (1_105_920.0, 1_116_160.0))


class _OneInnerChunkSource:
    """Writes shard (0,0): one 256^2 inner chunk of real data, the rest fill."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def live_shards(self):
        return [(0, 0)]

    def load(self, shard):
        rng = np.random.default_rng(self.seed)
        emb = np.zeros((1, _SHARD, _SHARD, _BAND), dtype="int8")
        scl = np.full((1, _SHARD, _SHARD), np.nan, dtype="float32")
        emb[0, 0:_CHUNK, 0:_CHUNK, :] = rng.integers(-127, 128, size=(_CHUNK, _CHUNK, _BAND), dtype="int8")
        scl[0, 0:_CHUNK, 0:_CHUNK] = rng.random((_CHUNK, _CHUNK), dtype="float32")
        return {"embeddings": emb, "scales": scl}


def _seed(tmp_path, zones=(_ZONE,)):
    # band is derived from the SMALL layout's embeddings band chunk (_BAND).
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, zones, years=(2023, 2024, 2025), layout=SMALL)
    return store, repo


def test_writes_one_shard_land_masked(tmp_path):
    store, repo = _seed(tmp_path)
    write_year_shards(repo, "32601", year_index=2, source=_OneInnerChunkSource(seed=1), n_workers=1, shard_px=_SHARD)

    g = zarr_store.open_store_as_zarr_group(store, group="32601")
    expected = np.random.default_rng(1).integers(-127, 128, size=(_CHUNK, _CHUNK, _BAND), dtype="int8")
    assert np.array_equal(g["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :], expected)
    # ocean within the written shard is elided -> reads as fill
    assert (g["embeddings"][2, 300:512, 300:512, :] == 0).all()
    assert np.isnan(g["scales"][2, 300:512, 300:512]).all()
    # the OTHER shard (rows 512:1024) was never written
    assert (g["embeddings"][2, 512:1024, 0:_CHUNK, :] == 0).all()
    # exactly one 512^2 shard touched (== 4 inner-chunk positions); the second
    # shard's 4 inner chunks stay uninitialized. (Ocean bytes within the written
    # shard are elided by the codec - verified at scale in d3v2 E4.)
    assert g["embeddings"].nchunks_initialized == 4


def test_years_complete_updated_in_commit(tmp_path):
    store, repo = _seed(tmp_path)
    write_year_shards(repo, "32601", year_index=2, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)
    g = zarr_store.open_store_as_zarr_group(store, group="32601")
    assert g.attrs["years_complete"] == [2025]


def test_other_years_stay_empty(tmp_path):
    store, repo = _seed(tmp_path)
    write_year_shards(repo, "32601", year_index=2, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)
    g = zarr_store.open_store_as_zarr_group(store, group="32601")
    assert (g["embeddings"][0, 0:_CHUNK, 0:_CHUNK, :] == 0).all()  # 2023 untouched


def test_commit_with_rebase_resolves_concurrent_disjoint_commits(tmp_path):
    store, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
    # Two sessions from the same tip write disjoint groups; both must commit.
    s1 = repo.writable_session("main")
    s2 = repo.writable_session("main")
    zarr.open_group(s1.store, mode="a")["32601"]["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :] = 1
    zarr.open_group(s2.store, mode="a")["32701"]["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :] = 2
    id1 = commit_with_rebase(s1, "write zone A")
    id2 = commit_with_rebase(s2, "write zone B")  # tip moved -> auto-rebase
    assert id1 and id2 and id1 != id2


def test_semaphore_gate_allows_and_bounds():
    gate = SemaphoreCommitGate(max_concurrent=1)
    with gate:
        assert not gate._sem.acquire(blocking=False)  # cap reached
    assert gate._sem.acquire(blocking=False)  # released


def test_write_year_shards_behind_gate(tmp_path):
    store, repo = _seed(tmp_path)
    gate = SemaphoreCommitGate(max_concurrent=2)
    sid = write_year_shards(
        repo, "32601", year_index=2, source=_OneInnerChunkSource(), n_workers=1, gate=gate, shard_px=_SHARD
    )
    assert isinstance(sid, str) and sid
