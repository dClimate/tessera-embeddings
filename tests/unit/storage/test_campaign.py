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
from tessera_embeddings.storage.zone_grid import ZoneSpec, fill_zones

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

    def pixels(self, shard):
        return None

    def load(self, shard):
        emb = np.zeros((1, _SHARD, _SHARD, _BAND), dtype="int8")
        scl = np.full((1, _SHARD, _SHARD), np.nan, dtype="float32")
        emb[0, 0:_CHUNK, 0:_CHUNK, :] = 1
        scl[0, 0:_CHUNK, 0:_CHUNK] = 0.5
        return {"embeddings": emb, "scales": scl}


@pytest.fixture
def seed(seeded_store):
    """``seed(zones=...)`` -> (path, repo) of a private copy of the seeded SMALL-layout store."""

    def seed(zones=(_ZONE_A, _ZONE_B)):
        store = seeded_store(zones, name="g.icechunk", years=_YEARS, layout=SMALL)
        return store, global_store.open_global_repo(store)

    return seed


def _fill(repo, zone, year_index):
    """Land a year through the one writer of ``years_complete`` without writing shards.

    Status, tag, work-list and GC tests only need the year marked; the shard path itself is
    exercised by the tests that call :func:`_write_shards`.
    """
    campaign.mark_zone_year_empty(repo, zone, _YEARS[year_index])


def _write_shards(repo, zone, year_index):
    write_year_shards(repo, zone, year_index=year_index, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)


# --- campaign_status -------------------------------------------------------


def test_status_empty_after_seed(seed):
    _, repo = seed()
    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.zones_seeded == 2
    assert status.zone_years_done == 0
    assert status.zones == {"01N": (), "02N": ()}


def test_status_tracks_landed_years(seed):
    _, repo = seed()
    _fill(repo, "01N", year_index=0)  # 2023
    _fill(repo, "01N", year_index=2)  # 2025
    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.zones["01N"] == (2023, 2025)
    assert status.zones["02N"] == ()
    assert status.zone_years_done == 2
    assert status.has("01N", 2023) and not status.has("01N", 2024)


# --- campaign_work_list (tag-aware driver work list) -----------------------


def test_work_list_filters_to_requested_zones(seed):
    """The `zones` filter restricts the fill chain to the requested zones only."""
    _, repo = seed()
    status = campaign.campaign_status(repo, years=_YEARS)
    work = campaign.campaign_work_list(status, set(), expected_zones=("01N",), years=_YEARS)
    assert {z for z, _ in work} == {"01N"}
    assert len(work) == len(_YEARS)  # every year of the one requested zone


def test_work_list_skips_completed_and_tagged_zones(seed):
    """A default re-run of a partially-complete year skips the finished zones."""
    _, repo = seed()
    _fill(repo, "01N", year_index=0)  # 2023 lands
    campaign.tag_zone_year(repo, "01N", 2023)  # ...and is tagged -> DONE
    status = campaign.campaign_status(repo, years=_YEARS)
    work = campaign.campaign_work_list(status, set(repo.list_tags()), expected_zones=("01N", "02N"), years=_YEARS)
    assert ("01N", 2023) not in work  # finished zone-year skipped
    assert ("01N", 2024) in work  # same zone, unfinished year still runs
    assert ("02N", 2023) in work  # untouched zone still runs


def test_work_list_includes_complete_but_untagged(seed):
    """A crash between the fill commit and the tag leaves a complete-but-untagged
    cell; it stays in the work list so the runner's idempotent retag path runs.
    """
    _, repo = seed()
    _fill(repo, "01N", year_index=0)  # lands in years_complete, NOT tagged
    status = campaign.campaign_status(repo, years=_YEARS)
    work = campaign.campaign_work_list(status, set(repo.list_tags()), expected_zones=("01N",), years=_YEARS)
    assert ("01N", 2023) in work  # complete but untagged -> still dispatched (retag)


@pytest.mark.parametrize(
    ("hemisphere", "layout", "unseeded"), [("N", "hemispheres", ("60N", 2023)), (None, "utm_zones", ("utm60s", 2023))]
)
def test_work_list_defaults_to_every_zone_of_the_store_layout(seed, hemisphere, layout, unseeded):
    """`expected_zones=None` drives the whole globe in the store's own layout (ADR 037), by halves (ADR 038)."""
    specs = [ZoneSpec(f"3260{n}", hemisphere, n, (0.0, 5_120.0), (0.0, 10_240.0)) for n in (1, 2)]
    _, repo = seed(zones=specs)  # only 2 zones seeded, but the default spans the layout
    status = campaign.campaign_status(repo, years=_YEARS)
    work = campaign.campaign_work_list(status, set(), years=_YEARS)  # expected_zones=None
    assert len(work) == len(fill_zones(layout)) * len(_YEARS)
    assert unseeded in work  # an unseeded zone still needs every year


def test_a_column_year_completes_with_its_second_half(seed):
    """ADR 038: each half marks only itself; the year enters ``years_complete`` with the second."""
    _, repo = seed(zones=(ZoneSpec("32601", None, 1, (0.0, 5_120.0), (-10_240.0, 10_240.0)),))
    write_year_shards(repo, "utm01", 0, _OneInnerChunkSource(), n_workers=1, shard_px=_SHARD, run_id="north", half="n")
    status = campaign.campaign_status(repo, years=_YEARS)
    assert (status.has("utm01n", 2023), status.has("utm01s", 2023), status.zones["utm01"]) == (True, False, ())
    assert status.zone_years_done == 1, "a landed half is data the seed must not stamp over"
    assert campaign.campaign_work_list(status, set(), years=(2023,))[:2] == [("utm01n", 2023), ("utm01s", 2023)]

    campaign.mark_zone_year_empty(repo, "utm01s", 2023, run_id="south")
    status = campaign.campaign_status(repo, years=_YEARS)
    assert (status.has("utm01s", 2023), status.zones["utm01"]) == (True, (2023,))
    runs = zarr.open_group(repo.readonly_session("main").store, mode="r")["utm01"].attrs["runs"]
    assert [r["run_id"] for r in shard_writer.year_runs(runs, 2023)] == ["north", "south"]


# --- tags ------------------------------------------------------------------


def test_tag_zone_year_at_head(seed):
    _, repo = seed()
    _fill(repo, "01N", year_index=0)
    tag = campaign.tag_zone_year(repo, "01N", 2023)
    assert tag == "zone-01N-2023"
    assert tag in repo.list_tags()
    assert repo.lookup_tag(tag) == repo.lookup_branch("main")


def test_tag_zone_year_refuses_to_move_explicit(seed):
    """An EXPLICIT snapshot that disagrees with the existing tag is refused."""
    _, repo = seed()
    _fill(repo, "01N", year_index=0)
    campaign.tag_zone_year(repo, "01N", 2023)
    _fill(repo, "01N", year_index=1)  # HEAD advances
    with pytest.raises(ValueError, match="refusing to move"):
        campaign.tag_zone_year(repo, "01N", 2023, snapshot_id=repo.lookup_branch("main"))


def test_tag_zone_year_default_is_idempotent_after_head_moves(seed):
    """Default-snapshot re-tagging after main advanced is a no-op success (sweep-safe)."""
    _, repo = seed()
    _fill(repo, "01N", year_index=0)
    campaign.tag_zone_year(repo, "01N", 2023)
    pinned = repo.lookup_tag("zone-01N-2023")
    _fill(repo, "01N", year_index=1)  # HEAD advances
    assert campaign.tag_zone_year(repo, "01N", 2023) == "zone-01N-2023"
    assert repo.lookup_tag("zone-01N-2023") == pinned, "existing pin must not move"


def test_tag_year_complete_requires_all_zones(seed, monkeypatch):
    monkeypatch.setattr(campaign, "fill_zones", lambda layout: dict.fromkeys(("01N", "02N")))
    _, repo = seed()
    _fill(repo, "01N", year_index=0)  # only one of two zones has 2023
    with pytest.raises(ValueError, match="have not landed it"):
        campaign.tag_year_complete(repo, 2023)


def test_tag_year_complete_when_all_land(seed, monkeypatch):
    monkeypatch.setattr(campaign, "fill_zones", lambda layout: dict.fromkeys(("01N", "02N")))
    _, repo = seed()
    _fill(repo, "01N", year_index=0)
    _fill(repo, "02N", year_index=0)
    tag = campaign.tag_year_complete(repo, 2023)
    assert tag == "year-2023-complete"
    assert tag in repo.list_tags()


def test_missing_zones_for_year_reports_in_scope_order(seed):
    """The report is the scope minus what landed, so a caller can name what is left."""
    _, repo = seed()
    _fill(repo, "02N", year_index=0)
    assert campaign.missing_zones_for_year(repo, 2023, expected_zones=("01N", "02N")) == ("01N",)
    assert campaign.missing_zones_for_year(repo, 2024, expected_zones=("02N", "01N")) == ("02N", "01N")


# --- expire_and_gc ---------------------------------------------------------


def test_expire_and_gc_rejects_naive_cutoff(seed):
    _, repo = seed()
    with pytest.raises(ValueError, match="timezone-aware"):
        campaign.expire_and_gc(repo, older_than=datetime(2020, 1, 1))


def test_expire_and_gc_rejects_future_cutoff(seed):
    _, repo = seed()
    future = datetime.now(UTC) + timedelta(days=1)
    with pytest.raises(ValueError, match="in the past"):
        campaign.expire_and_gc(repo, older_than=future)


def test_expire_and_gc_dry_run_does_not_mutate(seed):
    _, repo = seed()
    _fill(repo, "01N", year_index=0)
    before = list(repo.ancestry(branch="main"))
    cutoff = datetime.now(UTC) - timedelta(seconds=1)
    result = campaign.expire_and_gc(repo, older_than=cutoff, dry_run=True)
    assert result.expired_snapshots == frozenset()
    # dry-run leaves history untouched
    assert [s.id for s in repo.ancestry(branch="main")] == [s.id for s in before]


def test_expire_and_gc_keeps_tagged_snapshot(seed):
    _, repo = seed()
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


def test_mark_zone_year_empty_advances_completion(seed):
    """An all-ocean cell lands in years_complete + runs without any shard data."""
    _, repo = seed()
    snapshot = campaign.mark_zone_year_empty(repo, "01N", 2024, run_id="runE")

    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.has("01N", 2024)
    assert not status.has("02N", 2024)
    assert snapshot == repo.lookup_branch("main")

    node = zarr.open_group(repo.readonly_session("main").store, mode="r")["01N"]
    assert node.attrs["runs"]["2024"] == {**node.attrs["runs"]["2024"], "run_id": "runE", "empty": True}


def test_mark_zone_year_empty_idempotent(seed):
    """Re-marking an already-complete year commits nothing new — even with a run_id.

    A retry of a completed empty cell must return the same snapshot so the
    existing zone-year tag still matches (tag_zone_year refuses to move tags).
    """
    _, repo = seed()
    first = campaign.mark_zone_year_empty(repo, "01N", 2024, run_id="runA")
    again = campaign.mark_zone_year_empty(repo, "01N", 2024, run_id="runB")
    assert first == again == repo.lookup_branch("main")
    node = zarr.open_group(repo.readonly_session("main").store, mode="r")["01N"]
    assert node.attrs["runs"]["2024"]["run_id"] == "runA", "original provenance must be preserved"


def test_mark_zone_year_empty_off_axis_year_raises(seed):
    """A year outside the pre-allocated axis must never enter years_complete (D1)."""
    _, repo = seed()
    with pytest.raises(ValueError, match="not on 01N's pre-allocated time axis"):
        campaign.mark_zone_year_empty(repo, "01N", 1999)
    status = campaign.campaign_status(repo, years=_YEARS)
    assert not status.has("01N", 1999)


def test_work_list_dedupes_duplicate_inputs(seed):
    """Duplicate zones/years collapse to one cell (else two concurrent same-cell fills)."""
    _, repo = seed()
    status = campaign.campaign_status(repo, years=_YEARS)
    work = campaign.campaign_work_list(status, set(), expected_zones=("01N", "01N"), years=(2023, 2023))
    assert work == [("01N", 2023)]


def test_year_tag_refuses_an_empty_completion_scope(seed):
    """An empty scope verifies nothing, and the tag it would write is write-once.

    `None` means "all 120 zones". An empty list means the caller computed a scope and
    got nothing back — a bug in the caller, not a finished year. Tagging on it would
    stamp a permanent completion marker that cannot be corrected under its own name.
    """
    _, repo = seed()
    with pytest.raises(ValueError, match="expected_zones is empty"):
        campaign.missing_zones_for_year(repo, _YEARS[0], expected_zones=[])


# --- concurrent same-zone, different-year writes (the year barrier's whole content) ---


def test_a_stale_group_attr_commit_really_does_conflict(seed):
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
    _, repo = seed()
    s1, s2 = repo.writable_session("main"), repo.writable_session("main")
    zarr.open_group(s1.store, mode="a")["01N"].attrs["years_complete"] = [_YEARS[0]]
    zarr.open_group(s2.store, mode="a")["01N"].attrs["years_complete"] = [_YEARS[1]]
    commit_with_rebase(s1, "writer A")
    with pytest.raises(icechunk.RebaseFailedError):
        commit_with_rebase(s2, "writer B")


def test_commit_year_attrs_retries_a_conflict_and_produces_the_union(seed, monkeypatch):
    """The retry is correct, not merely bounded: the loser must re-read and re-apply.

    Because each writer inserts only its OWN year's key, re-reading the winner's value and
    adding your own yields exactly what both writers intended. A retry that re-applied a
    STALE read would instead drop the winner's year, which is the silent failure this
    replaces — so the assertion is on the union, not just on the absence of an exception.
    """
    _, repo = seed()
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


def test_commit_year_attrs_gives_up_after_its_bound(seed, monkeypatch):
    """A persistent conflict must surface, not spin — a defect is not a race."""
    _, repo = seed()
    monkeypatch.setattr(
        shard_writer,
        "commit_with_rebase",
        lambda *a, **k: (_ for _ in ()).throw(icechunk.RebaseFailedError("always", [])),
    )
    with pytest.raises(icechunk.RebaseFailedError):
        shard_writer.commit_year_attrs(repo, "01N", _YEARS[0], tries=3)


def test_two_years_of_one_zone_both_land_with_neither_mark_lost(seed):
    """End to end through the real fill path: the union survives, both years are complete.

    Each fill is two commits, shards then attrs. Collapsing them into one would silently restore
    the year barrier, which the timing-dependent concurrency tests might not catch.
    """
    _, repo = seed()
    before = len(list(repo.ancestry(branch="main")))
    _write_shards(repo, "01N", 0)
    _write_shards(repo, "01N", 1)
    assert len(list(repo.ancestry(branch="main"))) - before == 4, "a shard commit and an attr commit per fill"
    status = campaign.campaign_status(repo, years=_YEARS)
    assert status.has("01N", _YEARS[0]) and status.has("01N", _YEARS[1])
