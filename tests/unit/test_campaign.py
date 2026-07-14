"""Campaign operations: tags, snapshot hygiene, progress reader (W5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.storage import campaign, global_store
from tessera_embeddings.storage.shard_writer import write_year_shards
from tessera_embeddings.storage.zone_grid import ZoneSpec

_BAND = 8
_SHARD = 512
_CHUNK = 256

_EMB = ArrayLayout(DIMS_4D, (1, _CHUNK, _CHUNK, _BAND), "int8", 0, "zstd", shards=(1, _SHARD, _SHARD, _BAND))
_SCL = ArrayLayout(DIMS_3D, (1, _CHUNK, _CHUNK), "float32", float("nan"), "pcodec", shards=(1, _SHARD, _SHARD))
SMALL = StoreLayout(name="small", arrays={"embeddings": _EMB, "scales": _SCL})

_ZONE_A = ZoneSpec("32601", "N", 1, (0.0, 5_120.0), (0.0, 10_240.0))
_ZONE_B = ZoneSpec("32602", "N", 2, (0.0, 5_120.0), (0.0, 10_240.0))
_YEARS = (2023, 2024, 2025)


class _OneInnerChunkSource:
    """Writes shard (0,0): one 256^2 inner chunk of real data, the rest fill."""

    def live_shards(self):
        return [(0, 0)]

    def load(self, shard):
        emb = np.zeros((1, _SHARD, _SHARD, _BAND), dtype="int8")
        scl = np.full((1, _SHARD, _SHARD), np.nan, dtype="float32")
        emb[0, 0:_CHUNK, 0:_CHUNK, :] = 1
        scl[0, 0:_CHUNK, 0:_CHUNK] = 0.5
        return {"embeddings": emb, "scales": scl}


def _seed(tmp_path, zones=(_ZONE_A, _ZONE_B)):
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, zones, years=_YEARS, layout=SMALL)
    return store, repo


def _fill(repo, zone, year_index):
    write_year_shards(repo, zone, year_index=year_index, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)


# --- campaign_status -------------------------------------------------------


def test_status_empty_after_seed(tmp_path):
    _, repo = _seed(tmp_path)
    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.zones_seeded == 2
    assert status.zone_years_done == 0
    assert status.zones == {"32601": (), "32602": ()}


def test_status_tracks_landed_years(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "32601", year_index=0)  # 2023
    _fill(repo, "32601", year_index=2)  # 2025
    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.zones["32601"] == (2023, 2025)
    assert status.zones["32602"] == ()
    assert status.zone_years_done == 2
    assert status.has("32601", 2023) and not status.has("32601", 2024)


def test_status_pending_lists_gaps(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "32601", year_index=0)  # 2023
    status = campaign.campaign_status(repo, years=_YEARS)
    pending = status.pending(expected_zones=("32601", "32602"))
    assert ("32601", 2023) not in pending
    assert ("32601", 2024) in pending
    assert ("32602", 2023) in pending  # seeded-but-unfilled zone: every year pending
    assert len(pending) == 5  # 6 cells - 1 filled


def test_status_pending_counts_unseeded_zone(tmp_path):
    _, repo = _seed(tmp_path, zones=(_ZONE_A,))
    status = campaign.campaign_status(repo, years=_YEARS)
    # An entirely unseeded zone contributes all years to the work list.
    pending = status.pending(expected_zones=("32601", "32602"))
    assert sum(1 for z, _ in pending if z == "32602") == len(_YEARS)


def test_years_fully_complete(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "32601", year_index=0)  # 2023
    _fill(repo, "32602", year_index=0)  # 2023
    _fill(repo, "32601", year_index=1)  # 2024 only in one zone
    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.years_fully_complete(expected_zones=("32601", "32602")) == [2023]


# --- tags ------------------------------------------------------------------


def test_tag_zone_year_at_head(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "32601", year_index=0)
    tag = campaign.tag_zone_year(repo, "32601", 2023)
    assert tag == "zone-32601-2023"
    assert tag in repo.list_tags()
    assert repo.lookup_tag(tag) == repo.lookup_branch("main")


def test_tag_zone_year_idempotent(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "32601", year_index=0)
    first = campaign.tag_zone_year(repo, "32601", 2023)
    again = campaign.tag_zone_year(repo, "32601", 2023)  # no-op, same snapshot
    assert first == again


def test_tag_zone_year_refuses_to_move(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "32601", year_index=0)
    campaign.tag_zone_year(repo, "32601", 2023)
    _fill(repo, "32601", year_index=1)  # HEAD advances
    with pytest.raises(ValueError, match="refusing to move"):
        campaign.tag_zone_year(repo, "32601", 2023)  # would point at a new HEAD


def test_tag_year_complete_requires_all_zones(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "32601", year_index=0)  # only one of two zones has 2023
    with pytest.raises(ValueError, match="have not landed it"):
        campaign.tag_year_complete(repo, 2023, expected_zones=("32601", "32602"))


def test_tag_year_complete_when_all_land(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "32601", year_index=0)
    _fill(repo, "32602", year_index=0)
    tag = campaign.tag_year_complete(repo, 2023, expected_zones=("32601", "32602"))
    assert tag == "year-2023-complete"
    assert tag in repo.list_tags()


# --- expire_and_gc ---------------------------------------------------------


def test_expire_and_gc_rejects_naive_cutoff(tmp_path):
    _, repo = _seed(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware"):
        campaign.expire_and_gc(repo, older_than=datetime(2020, 1, 1))


def test_expire_and_gc_rejects_future_cutoff(tmp_path):
    _, repo = _seed(tmp_path)
    future = datetime.now(UTC) + timedelta(days=1)
    with pytest.raises(ValueError, match="in the past"):
        campaign.expire_and_gc(repo, older_than=future)


def test_expire_and_gc_dry_run_does_not_mutate(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "32601", year_index=0)
    before = list(repo.ancestry(branch="main"))
    cutoff = datetime.now(UTC) - timedelta(seconds=1)
    result = campaign.expire_and_gc(repo, older_than=cutoff, dry_run=True)
    assert result.expired_snapshots == frozenset()
    # dry-run leaves history untouched
    assert [s.id for s in repo.ancestry(branch="main")] == [s.id for s in before]


def test_expire_and_gc_keeps_tagged_snapshot(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "32601", year_index=0)
    tagged = repo.lookup_branch("main")
    campaign.tag_zone_year(repo, "32601", 2023)  # protects `tagged`
    _fill(repo, "32601", year_index=1)  # advance HEAD past the tagged snapshot
    # Expire anything strictly older than "now": the tagged snapshot is a
    # candidate for expiry by age but must survive because the tag pins it.
    cutoff = datetime.now(UTC) - timedelta(microseconds=1)
    campaign.expire_and_gc(repo, older_than=cutoff)
    # the tagged snapshot is reachable via the tag even though HEAD moved on
    assert tagged in {s.id for s in repo.ancestry(tag="zone-32601-2023")}
