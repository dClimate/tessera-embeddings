"""Campaign operations: tags, snapshot hygiene, progress reader (W5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import icechunk
import numpy as np
import pytest
import zarr

import tessera_embeddings.storage.shard_writer as shard_writer
from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.storage import campaign, global_store
from tessera_embeddings.storage.shard_writer import commit_with_rebase, write_year_shards
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


def test_tag_year_complete_requires_all_zones(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ZONES", ("01N", "02N"))
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)  # only one of two zones has 2023
    with pytest.raises(ValueError, match="have not landed it"):
        campaign.tag_year_complete(repo, 2023)


def test_tag_year_complete_when_all_land(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ZONES", ("01N", "02N"))
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)
    _fill(repo, "02N", year_index=0)
    tag = campaign.tag_year_complete(repo, 2023)
    assert tag == "year-2023-complete"
    assert tag in repo.list_tags()


def test_year_complete_tag_cannot_be_minted_from_a_zone_subset(tmp_path):
    """The campaign-wide tag takes no scope argument, so a subset cannot mint it.

    Icechunk tags are write-once forever and `_ensure_tag` treats an existing tag as an
    idempotent success, so a `year-2023-complete` stamped after two zones landed could
    never be corrected by the real 120-zone campaign. The guarantee is structural —
    there is no parameter to pass — so this asserts the signature, not a branch: the
    subset question is answerable only through the helper that does not tag.
    """
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", year_index=0)
    _fill(repo, "02N", year_index=0)

    with pytest.raises(TypeError):
        campaign.tag_year_complete(repo, 2023, expected_zones=("01N", "02N"))  # type: ignore[call-arg]

    # Both seeded zones HAVE landed 2023, so the subset check is satisfied...
    assert campaign.missing_zones_for_year(repo, 2023, expected_zones=("01N", "02N")) == ()
    # ...and no tag was created by asking.
    assert "year-2023-complete" not in repo.list_tags()
    # Against the real campaign scope the year is nowhere near done, and the refusal
    # names how far off it is rather than tagging.
    with pytest.raises(ValueError, match="have not landed it"):
        campaign.tag_year_complete(repo, 2023)
    assert "year-2023-complete" not in repo.list_tags()


def test_missing_zones_for_year_reports_in_scope_order(tmp_path):
    """The report is the scope minus what landed, so a caller can name what is left."""
    _, repo = _seed(tmp_path)
    _fill(repo, "02N", year_index=0)
    assert campaign.missing_zones_for_year(repo, 2023, expected_zones=("01N", "02N")) == ("01N",)
    assert campaign.missing_zones_for_year(repo, 2024, expected_zones=("02N", "01N")) == ("02N", "01N")


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


def test_year_tag_refuses_an_empty_completion_scope(tmp_path):
    """An empty scope verifies nothing, and the tag it would write is write-once.

    `None` means "all 120 zones". An empty list means the caller computed a scope and
    got nothing back — a bug in the caller, not a finished year. Tagging on it would
    stamp a permanent completion marker that cannot be corrected under its own name.
    """
    _, repo = _seed(tmp_path)
    with pytest.raises(ValueError, match="expected_zones is empty"):
        campaign.missing_zones_for_year(repo, _YEARS[0], expected_zones=[])


# --- concurrent same-zone, different-year writes (the year barrier's whole content) ---


def test_a_stale_group_attr_commit_really_does_conflict(tmp_path):
    """The precondition the whole design rests on, pinned rather than assumed.

    Two sessions forked from one tip, each inserting its own year, the second raises
    RebaseFailedError — icechunk's ConflictDetector treats group attributes as an opaque
    value and will not merge them. This is what made the campaign's years serial, and it is
    worth a test because the alternative behaviour (a silent last-writer-wins clobber) would
    make `commit_year_attrs`'s retry loop pointless and the lost update invisible.

    NOT tested with threads: icechunk warns that local-filesystem storage is unsafe for
    concurrent commits, so a threaded test here would be measuring that rather than our
    logic. Forking two sessions from one tip reproduces the same race deterministically.
    """
    _, repo = _seed(tmp_path)
    s1, s2 = repo.writable_session("main"), repo.writable_session("main")
    zarr.open_group(s1.store, mode="a")["01N"].attrs["years_complete"] = [_YEARS[0]]
    zarr.open_group(s2.store, mode="a")["01N"].attrs["years_complete"] = [_YEARS[1]]
    commit_with_rebase(s1, "writer A")
    with pytest.raises(icechunk.RebaseFailedError):
        commit_with_rebase(s2, "writer B")


def test_commit_year_attrs_retries_a_conflict_and_produces_the_union(tmp_path, monkeypatch):
    """The retry is correct, not merely bounded: the loser must re-read and re-apply.

    Because each writer inserts only its OWN year's key, re-reading the winner's value and
    adding your own yields exactly what both writers intended. A retry that re-applied a
    STALE read would instead drop the winner's year, which is the silent failure this
    replaces — so the assertion is on the union, not just on the absence of an exception.
    """
    _, repo = _seed(tmp_path)
    shard_writer.commit_year_attrs(repo, "01N", _YEARS[0])  # the "winner" lands first

    real = shard_writer.commit_with_rebase
    calls: list[int] = []

    def flaky(session, message, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise icechunk.RebaseFailedError("synthetic", [])
        return real(session, message, **kw)

    monkeypatch.setattr(shard_writer, "commit_with_rebase", flaky)
    shard_writer.commit_year_attrs(repo, "01N", _YEARS[1])
    assert len(calls) == 2, "the first attempt must have been retried"

    node = zarr.open_group(repo.readonly_session("main").store, mode="r")["01N"]
    assert shard_writer.read_years_complete(node) == [_YEARS[0], _YEARS[1]]


def test_commit_year_attrs_gives_up_after_its_bound(tmp_path, monkeypatch):
    """A persistent conflict must surface, not spin — a defect is not a race."""
    _, repo = _seed(tmp_path)
    monkeypatch.setattr(
        shard_writer,
        "commit_with_rebase",
        lambda *a, **k: (_ for _ in ()).throw(icechunk.RebaseFailedError("always", [])),
    )
    with pytest.raises(icechunk.RebaseFailedError):
        shard_writer.commit_year_attrs(repo, "01N", _YEARS[0], tries=3)


def test_two_years_of_one_zone_both_land_with_neither_mark_lost(tmp_path):
    """End to end through the real fill path: the union survives, both years are complete."""
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", 0)
    _fill(repo, "01N", 1)
    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.has("01N", _YEARS[0]) and status.has("01N", _YEARS[1])


def test_a_no_land_mark_composes_with_a_real_fill_of_another_year(tmp_path):
    """The other attr writer takes the same path, so the two compose."""
    _, repo = _seed(tmp_path)
    _fill(repo, "01N", 0)
    campaign.mark_zone_year_empty(repo, "01N", _YEARS[1])
    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.has("01N", _YEARS[0]) and status.has("01N", _YEARS[1])


def test_the_attr_commit_is_separate_from_the_shard_commit(tmp_path):
    """Two commits per fill, not one — the mechanism the tests above rely on.

    Pinned because collapsing them back into one commit would silently restore the
    year barrier, and the concurrency tests above are timing-dependent enough that they
    might not catch it.
    """
    _, repo = _seed(tmp_path)
    before = len(list(repo.ancestry(branch="main")))
    _fill(repo, "01N", 0)
    after = len(list(repo.ancestry(branch="main")))
    assert after - before == 2, f"expected a shard commit and an attr commit, got {after - before}"
