"""Shard-aligned land-masked writer + commit discipline (W3)."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.storage import global_store, zarr_store
from tessera_embeddings.storage.shard_writer import (
    _await_forks,
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
    write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(seed=1), n_workers=1, shard_px=_SHARD)

    g = zarr_store.open_store_as_zarr_group(store, group="01N")
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
    write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)
    g = zarr_store.open_store_as_zarr_group(store, group="01N")
    assert g.attrs["years_complete"] == [2025]


def test_other_years_stay_empty(tmp_path):
    store, repo = _seed(tmp_path)
    write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)
    g = zarr_store.open_store_as_zarr_group(store, group="01N")
    assert (g["embeddings"][0, 0:_CHUNK, 0:_CHUNK, :] == 0).all()  # 2023 untouched


def test_commit_with_rebase_resolves_concurrent_disjoint_commits(tmp_path):
    store, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
    # Two sessions from the same tip write disjoint groups; both must commit.
    s1 = repo.writable_session("main")
    s2 = repo.writable_session("main")
    zarr.open_group(s1.store, mode="a")["01N"]["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :] = 1
    zarr.open_group(s2.store, mode="a")["01S"]["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :] = 2
    id1 = commit_with_rebase(s1, "write zone A")
    id2 = commit_with_rebase(s2, "write zone B")  # tip moved -> auto-rebase
    assert id1 and id2 and id1 != id2


def test_write_year_shards_behind_gate(tmp_path):
    store, repo = _seed(tmp_path)
    gate = threading.Semaphore(2)
    sid = write_year_shards(
        repo, "01N", year_index=2, source=_OneInnerChunkSource(), n_workers=1, gate=gate, shard_px=_SHARD
    )
    assert isinstance(sid, str) and sid


class TestForkProgressReporting:
    """A forked write reports what is outstanding, on a timer, without reordering forks.

    Exercises :func:`_await_forks` against plain futures rather than a real process
    pool: the behaviour under test is the coordinator's waiting policy, and driving it
    with resolvable futures makes the timing deterministic instead of dependent on how
    fast a spawned interpreter starts.
    """

    @staticmethod
    def _resolved(value):
        future: Future = Future()
        future.set_result(value)
        return future

    @staticmethod
    def _progress_lines(caplog):
        return [r.getMessage() for r in caplog.records if "Assembly progress" in r.getMessage()]

    def test_results_follow_submission_order_not_completion_order(self):
        # The slow fork is submitted FIRST but finishes LAST; merge order must still be
        # the caller's band order, so a completion-ordered collect would fail here.
        slow: Future = Future()
        quick = self._resolved("second")
        threading.Timer(0.05, lambda: slow.set_result("first")).start()
        assert _await_forks([slow, quick], 10.0) == ["first", "second"]

    def test_progress_is_reported_on_a_timer_not_only_on_completions(self, caplog):
        # ONE outstanding fork, so there is no completion to report until the very end.
        # Per-completion reporting would leave this silent — which is the failure mode
        # the timer exists to remove.
        outstanding: Future = Future()
        threading.Timer(0.15, lambda: outstanding.set_result("done")).start()
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.storage.shard_writer"):
            assert _await_forks([outstanding], 0.01) == ["done"]
        lines = self._progress_lines(caplog)
        assert len(lines) >= 2, f"a single long band reported {len(lines)} progress line(s)"
        assert "1 outstanding" in lines[0]

    def test_nothing_outstanding_reports_nothing(self, caplog):
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.storage.shard_writer"):
            assert _await_forks([self._resolved(1), self._resolved(2)], 0.01) == [1, 2]
        assert self._progress_lines(caplog) == []

    def test_a_failed_fork_still_raises(self):
        # Waiting must not swallow a band failure into a partial merge.
        failed: Future = Future()
        failed.set_exception(RuntimeError("band 0 failed"))
        with pytest.raises(RuntimeError, match="band 0 failed"):
            _await_forks([failed, self._resolved("ok")], 10.0)
