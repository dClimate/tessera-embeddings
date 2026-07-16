"""Campaign operations: tags, snapshot hygiene, progress reader (W5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.storage import campaign, global_store
from tessera_embeddings.storage.shard_writer import write_year_shards
from tessera_embeddings.storage.zone_grid import ZONES, ZoneSpec

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
    assert status.zones == {"01N": (), "02N": ()}


def test_status_tracks_landed_years(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)  # 2023
    _fill(repo, "01N", year_index=2)  # 2025
    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.zones["01N"] == (2023, 2025)
    assert status.zones["02N"] == ()
    assert status.zone_years_done == 2
    assert status.has("01N", 2023) and not status.has("01N", 2024)


def test_status_pending_lists_gaps(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)  # 2023
    status = campaign.campaign_status(repo, years=_YEARS)
    pending = status.pending(expected_zones=("01N", "02N"))
    assert ("01N", 2023) not in pending
    assert ("01N", 2024) in pending
    assert ("02N", 2023) in pending  # seeded-but-unfilled zone: every year pending
    assert len(pending) == 5  # 6 cells - 1 filled


def test_status_pending_counts_unseeded_zone(tmp_path):
    _, repo = _seed(tmp_path, zones=(_ZONE_A,))
    status = campaign.campaign_status(repo, years=_YEARS)
    # An entirely unseeded zone contributes all years to the work list.
    pending = status.pending(expected_zones=("01N", "02N"))
    assert sum(1 for z, _ in pending if z == "02N") == len(_YEARS)


# --- campaign_work_list (tag-aware driver work list) -----------------------


def test_work_list_filters_to_requested_zones(tmp_path):
    """The `zones` filter restricts the fill chain to the requested zones only."""
    _, repo = _seed(tmp_path)
    status = campaign.campaign_status(repo, years=_YEARS)
    work = campaign.campaign_work_list(status, set(), expected_zones=("01N",), years=_YEARS)
    assert {z for z, _ in work} == {"01N"}
    assert len(work) == len(_YEARS)  # every year of the one requested zone


def test_work_list_skips_completed_and_tagged_zones(tmp_path):
    """A default re-run of a partially-complete year skips the finished zones."""
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)  # 2023 lands
    campaign.tag_zone_year(repo, "01N", 2023)  # ...and is tagged -> DONE
    status = campaign.campaign_status(repo, years=_YEARS)
    work = campaign.campaign_work_list(status, set(repo.list_tags()), expected_zones=("01N", "02N"), years=_YEARS)
    assert ("01N", 2023) not in work  # finished zone-year skipped
    assert ("01N", 2024) in work  # same zone, unfinished year still runs
    assert ("02N", 2023) in work  # untouched zone still runs


def test_work_list_includes_complete_but_untagged(tmp_path):
    """A crash between the fill commit and the tag leaves a complete-but-untagged
    cell; it stays in the work list so the runner's idempotent retag path runs.
    """
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)  # lands in years_complete, NOT tagged
    status = campaign.campaign_status(repo, years=_YEARS)
    work = campaign.campaign_work_list(status, set(repo.list_tags()), expected_zones=("01N",), years=_YEARS)
    assert ("01N", 2023) in work  # complete but untagged -> still dispatched (retag)


def test_work_list_defaults_to_all_120_zones(tmp_path):
    """`expected_zones=None` drives the whole globe (an unseeded zone needs every year)."""
    _, repo = _seed(tmp_path)  # only 2 zones seeded, but the default list spans all 120
    status = campaign.campaign_status(repo, years=_YEARS)
    work = campaign.campaign_work_list(status, set(), years=_YEARS)  # expected_zones=None
    assert len(work) == len(ZONES) * len(_YEARS)
    assert ("60N", 2023) in work  # an unseeded zone still needs every year


def test_years_fully_complete(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)  # 2023
    _fill(repo, "02N", year_index=0)  # 2023
    _fill(repo, "01N", year_index=1)  # 2024 only in one zone
    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.years_fully_complete(expected_zones=("01N", "02N")) == [2023]


# --- tags ------------------------------------------------------------------


def test_tag_zone_year_at_head(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)
    tag = campaign.tag_zone_year(repo, "01N", 2023)
    assert tag == "zone-01N-2023"
    assert tag in repo.list_tags()
    assert repo.lookup_tag(tag) == repo.lookup_branch("main")


def test_tag_zone_year_idempotent(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)
    first = campaign.tag_zone_year(repo, "01N", 2023)
    again = campaign.tag_zone_year(repo, "01N", 2023)  # no-op, same snapshot
    assert first == again


def test_tag_zone_year_refuses_to_move_explicit(tmp_path):
    """An EXPLICIT snapshot that disagrees with the existing tag is refused."""
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)
    campaign.tag_zone_year(repo, "01N", 2023)
    _fill(repo, "01N", year_index=1)  # HEAD advances
    with pytest.raises(ValueError, match="refusing to move"):
        campaign.tag_zone_year(repo, "01N", 2023, snapshot_id=repo.lookup_branch("main"))


def test_tag_zone_year_default_is_idempotent_after_head_moves(tmp_path):
    """Default-snapshot re-tagging after main advanced is a no-op success (sweep-safe)."""
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)
    campaign.tag_zone_year(repo, "01N", 2023)
    pinned = repo.lookup_tag("zone-01N-2023")
    _fill(repo, "01N", year_index=1)  # HEAD advances
    assert campaign.tag_zone_year(repo, "01N", 2023) == "zone-01N-2023"
    assert repo.lookup_tag("zone-01N-2023") == pinned, "existing pin must not move"


def test_tag_year_complete_requires_all_zones(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)  # only one of two zones has 2023
    with pytest.raises(ValueError, match="have not landed it"):
        campaign.tag_year_complete(repo, 2023, expected_zones=("01N", "02N"))


def test_tag_year_complete_when_all_land(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)
    _fill(repo, "02N", year_index=0)
    tag = campaign.tag_year_complete(repo, 2023, expected_zones=("01N", "02N"))
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
    _fill(repo, "01N", year_index=0)
    before = list(repo.ancestry(branch="main"))
    cutoff = datetime.now(UTC) - timedelta(seconds=1)
    result = campaign.expire_and_gc(repo, older_than=cutoff, dry_run=True)
    assert result.expired_snapshots == frozenset()
    # dry-run leaves history untouched
    assert [s.id for s in repo.ancestry(branch="main")] == [s.id for s in before]


def test_expire_and_gc_keeps_tagged_snapshot(tmp_path):
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)
    tagged = repo.lookup_branch("main")
    campaign.tag_zone_year(repo, "01N", 2023)  # protects `tagged`
    _fill(repo, "01N", year_index=1)  # advance HEAD past the tagged snapshot
    # Expire anything strictly older than "now": the tagged snapshot is a
    # candidate for expiry by age but must survive because the tag pins it.
    cutoff = datetime.now(UTC) - timedelta(microseconds=1)
    campaign.expire_and_gc(repo, older_than=cutoff)
    # the tagged snapshot is reachable via the tag even though HEAD moved on
    assert tagged in {s.id for s in repo.ancestry(tag="zone-01N-2023")}


# --- mark_zone_year_empty ----------------------------------------------------


def test_mark_zone_year_empty_advances_completion(tmp_path):
    """An all-ocean cell lands in years_complete + runs without any shard data."""
    _, repo = _seed(tmp_path)
    snapshot = campaign.mark_zone_year_empty(repo, "01N", 2024, run_id="runE")

    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.has("01N", 2024)
    assert not status.has("02N", 2024)
    assert snapshot == repo.lookup_branch("main")

    node = zarr.open_group(repo.readonly_session("main").store, mode="r")["01N"]
    assert node.attrs["runs"]["2024"] == {**node.attrs["runs"]["2024"], "run_id": "runE", "empty": True}


def test_mark_zone_year_empty_idempotent(tmp_path):
    """Re-marking an already-complete year commits nothing new — even with a run_id.

    A retry of a completed empty cell must return the same snapshot so the
    existing zone-year tag still matches (tag_zone_year refuses to move tags).
    """
    _, repo = _seed(tmp_path)
    first = campaign.mark_zone_year_empty(repo, "01N", 2024, run_id="runA")
    again = campaign.mark_zone_year_empty(repo, "01N", 2024, run_id="runB")
    assert first == again == repo.lookup_branch("main")
    node = zarr.open_group(repo.readonly_session("main").store, mode="r")["01N"]
    assert node.attrs["runs"]["2024"]["run_id"] == "runA", "original provenance must be preserved"


def test_mark_zone_year_empty_off_axis_year_raises(tmp_path):
    """A year outside the pre-allocated axis must never enter years_complete (D1)."""
    _, repo = _seed(tmp_path)
    with pytest.raises(ValueError, match="not on 01N's pre-allocated time axis"):
        campaign.mark_zone_year_empty(repo, "01N", 1999)
    status = campaign.campaign_status(repo, years=_YEARS)
    assert not status.has("01N", 1999)


def test_work_list_dedupes_duplicate_inputs(tmp_path):
    """Duplicate zones/years collapse to one cell (else two concurrent same-cell fills)."""
    _, repo = _seed(tmp_path)
    status = campaign.campaign_status(repo, years=_YEARS)
    work = campaign.campaign_work_list(status, set(), expected_zones=("01N", "01N"), years=(2023, 2023))
    assert work == [("01N", 2023)]
