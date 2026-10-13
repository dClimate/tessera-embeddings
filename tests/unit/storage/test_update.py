"""Targeted-update plans and the store records they leave (storage/update.py)."""

from __future__ import annotations

import icechunk
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.inference.chunk_spec import enumerate_chunks, read_pixel_mask
from tessera_embeddings.storage import campaign
from tessera_embeddings.storage import update as upd
from tessera_embeddings.storage.campaign import pin_tag
from tessera_embeddings.storage.published_store import Commit, committed_runs
from tessera_embeddings.storage.zarr_store import open_or_create_repo


def _plan(**overrides) -> upd.UpdatePlan:
    fields = {"name": "fix-1", "mode": "shards", "cells": {"14N/2019": ["chunk_1_2"], "15S/2020": None}}
    return upd.UpdatePlan(**{**fields, **overrides})


def test_a_plan_round_trips_and_normalises_its_cells():
    plan = _plan(cells={"14n/2019": ["chunk_1_2", "chunk_0_1", "chunk_1_2"], "15S/2020": None})
    assert plan.cells == {"14N/2019": ("chunk_0_1", "chunk_1_2"), "15S/2020": None}
    assert upd.UpdatePlan.from_json(plan.to_json()) == plan
    assert plan.cell_list(zones=["14N"]) == [("14N", 2019)]
    assert plan.target("15S", 2020, root="s3://b/updates/fix-1").holes == "s3://b/updates/fix-1/holes/15S/2020"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"name": "Not_Valid"}, "must match"),
        ({"mode": "pixels"}, "mode must be"),
        ({"mode": "holes", "on_skip": "clear"}, "nothing to clear"),
        ({"min_valid_coverage": 0}, "percentage"),
        ({"cells": {}}, "at least one cell"),
        ({"cells": {"14N/2019": []}}, "lists no shards"),
        ({"cells": {"14N/2019": ["tile_1_2"]}}, "chunk_<row>_<col>"),
        ({"cells": {"14N-2019": None}}, "<zone>/<year>"),
        ({"cells": {"7S/2024": None, "07S/2024": ["chunk_1_2"]}}, "more than once"),
    ],
)
def test_a_plan_the_flows_could_not_run_is_refused(overrides, match):
    with pytest.raises(ValueError, match=match):
        _plan(**overrides)


def test_a_plan_is_written_once(tmp_path):
    paths = BucketPaths(inputs=str(tmp_path / "in"), outputs=str(tmp_path / "out"))
    uri = upd.plan_uri(paths.update_root("fix-1"))
    upd.save_plan(_plan(), uri)
    with pytest.raises(FileExistsError):
        upd.save_plan(_plan(source="a second operator"), uri)
    assert upd.load_update(paths, "fix-1") == _plan(created_at=upd.load_plan(uri).created_at)
    with pytest.raises(ValueError, match="planned under mask"):
        upd.load_update(paths, "fix-1", mask_name="other")
    upd.save_plan(_plan(), upd.plan_uri(paths.update_root("copied")))
    with pytest.raises(ValueError, match="is for update 'fix-1'"):
        upd.load_update(paths, "copied")


def _master(tmp_path, rows_by_cell: dict[tuple[str, int], list[dict]]) -> str:
    """A miniature registry ``master/``: hive partitions of tile rows with the columns queried."""
    root = tmp_path / "registry"
    for (zone, year), rows in rows_by_cell.items():
        part = root / "master" / f"zone={zone}" / f"year={year}"
        part.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), part / "part-0.parquet")
    return str(root)


def _row(tile: str, refused: int, box: tuple[float, float, float, float]) -> dict:
    west, south, east, north = box
    return {
        "tile": tile,
        "refused_thin_px": refused,
        "bbox_west": west,
        "bbox_south": south,
        "bbox_east": east,
        "bbox_north": north,
    }


@pytest.mark.parametrize(
    ("where", "aoi", "cells"),
    [
        ("refused_thin_px > 0", None, {"14N/2019": ("chunk_0_1",), "60N/2019": ("chunk_0_0",)}),
        (None, (-100.0, 40.0, -99.0, 41.0), {"14N/2019": ("chunk_0_0", "chunk_0_1")}),
        # Zone 60's antimeridian row (west > east) intersects a box just east of -180.
        (None, (-179.9, 10.0, -179.0, 11.0), {"60N/2019": ("chunk_0_0",)}),
        # And an AOI that itself crosses the antimeridian still finds the crossing row.
        (None, (179.0, 10.0, -179.0, 11.0), {"60N/2019": ("chunk_0_0",)}),
    ],
)
def test_a_plan_can_come_from_the_registry(tmp_path, where, aoi, cells):
    root = _master(
        tmp_path,
        {
            ("14N", 2019): [
                _row("chunk_0_0", 0, (-99.8, 40.2, -99.6, 40.4)),
                _row("chunk_0_1", 5, (-99.6, 40.2, -99.4, 40.4)),
            ],
            # A cell file written after a column was added: the scan must still read the others.
            ("60N", 2019): [_row("chunk_0_0", 3, (179.9, 10.2, -179.8, 10.4)) | {"update_name": "earlier"}],
        },
    )
    plan = upd.plan_from_registry(root, name="fill-thin", mode="holes", where=where, aoi=aoi)
    assert plan.cells == cells


def test_a_registry_query_matching_nothing_is_refused(tmp_path):
    root = _master(tmp_path, {("14N", 2019): [_row("chunk_0_0", 0, (0.0, 0.0, 1.0, 1.0))]})
    with pytest.raises(ValueError, match="No registry rows"):
        upd.plan_from_registry(root, name="none", mode="shards", where="refused_thin_px > 0")


@pytest.fixture
def zone_repo(tmp_path) -> icechunk.Repository:
    """A repo with one zone group filled for 2024 by run ``fill``."""
    repo, _ = open_or_create_repo(str(tmp_path / "store.icechunk"))
    session = repo.writable_session("main")
    node = zarr.open_group(session.store, mode="a").require_group("01N")
    node.create_array("scales", shape=(2, 8, 8), chunks=(1, 4, 4), dtype="float32", fill_value=float("nan"))
    node.attrs.update(
        {"years_complete": [2024], "runs": {"2024": {"run_id": "fill", "assembled_at": "2026-01-01T00:00:00+00:00"}}}
    )
    session.commit("seed")
    return repo


def _target(name: str = "fix-1") -> upd.UpdateTarget:
    return upd.UpdateTarget(name=name, mode="shards", on_skip="keep", mask_name="global", tiles=None, holes="")


def _node(repo: icechunk.Repository) -> zarr.Group:
    return zarr.open_group(repo.readonly_session(branch="main").store, mode="r")["01N"]


def test_an_update_record_appends_and_leaves_the_fill_alone(zone_repo):
    record = upd.update_record(_target(), run_id="u-run", written=["chunk_0_0"], kept=[], optical_min_obs=1)
    commit = {"snapshot": "s-1", "committed_at": "2026-10-13T00:00:00+00:00"}
    upd.commit_update_record(zone_repo, "01N", 2024, commit, record=record)
    tip = zone_repo.lookup_branch("main")
    upd.commit_update_record(zone_repo, "01N", 2024, commit, record=record)  # idempotent per update

    node = _node(zone_repo)
    assert zone_repo.lookup_branch("main") == tip
    assert [(r["update"], r["commit"], r["optical_min_obs"]) for r in upd.read_updates(node, 2024)] == [
        ("fix-1", commit, 1)
    ]
    assert node.attrs["years_complete"] == [2024] and node.attrs["runs"]["2024"]["run_id"] == "fill"
    assert upd.update_states(zone_repo, "fix-1", [("01N", 2024), ("01N", 2025)]) == {
        ("01N", 2024): "done",
        ("01N", 2025): "not_complete",
    }


def test_an_update_whose_shard_commit_expired_is_still_known_to_the_store_check(zone_repo):
    """An update records itself in ``updates[year]``, not ``runs``, so after snapshot expiry that record is
    the store's only account of the write; without it the update's registry part names a run the store lacks.
    """
    commit = {"snapshot": "expired", "committed_at": "2026-10-13T00:00:00+00:00"}
    wrote = upd.update_record(_target(), run_id="u-run", written=["chunk_0_0"], kept=[], optical_min_obs=1)
    upd.commit_update_record(zone_repo, "01N", 2024, commit, record=wrote)
    nothing = upd.update_record(_target("noop"), run_id="noop", written=[], kept=["chunk_0_0"], optical_min_obs=1)
    upd.commit_update_record(zone_repo, "01N", 2024, record=nothing)
    assert committed_runs(_node(zone_repo).attrs, 2024, []) == [
        Commit("fill", "2026-01-01T00:00:00+00:00", None),
        Commit("u-run", commit["committed_at"], None),
    ]


def test_an_update_record_retries_a_concurrent_attr_commit(zone_repo, monkeypatch):
    """Attrs cannot be merged by icechunk, so a lost race re-reads and re-applies — exactly once."""
    real = upd.commit_with_rebase
    calls = {"n": 0}

    def flaky(session, message):
        calls["n"] += 1
        if calls["n"] == 1:
            raise icechunk.RebaseFailedError("lost-the-race", [])
        return real(session, message)

    monkeypatch.setattr(upd, "commit_with_rebase", flaky)
    upd.commit_update_record(
        zone_repo, "01N", 2024, record=upd.update_record(_target(), run_id="r", written=[], kept=[], optical_min_obs=1)
    )
    assert calls["n"] == 2 and len(upd.read_updates(_node(zone_repo), 2024)) == 1


def test_the_complete_tag_waits_for_every_planned_cell(zone_repo):
    plan = upd.UpdatePlan(name="fix-1", mode="shards", cells={"01N/2024": None, "01N/2025": None})
    assert upd.finish_update(zone_repo, plan) is None
    upd.commit_update_record(
        zone_repo, "01N", 2024, record=upd.update_record(_target(), run_id="r", written=[], kept=[], optical_min_obs=1)
    )
    assert upd.finish_update(zone_repo, plan) is None, "2025 was never filled, so the update is not complete"
    single = upd.UpdatePlan(name="fix-1", mode="shards", cells={"01N/2024": None})
    assert upd.finish_update(zone_repo, single) == upd.complete_tag("fix-1")


def test_a_base_tag_pins_the_first_snapshot_only(zone_repo):
    first = zone_repo.lookup_branch("main")
    pin_tag(zone_repo, upd.base_tag("fix-1"))
    upd.commit_update_record(
        zone_repo, "01N", 2024, record=upd.update_record(_target(), run_id="r", written=[], kept=[], optical_min_obs=1)
    )
    pin_tag(zone_repo, upd.base_tag("fix-1"))
    assert zone_repo.lookup_tag(upd.base_tag("fix-1")) == first


def test_hole_masks_cover_the_holes_and_are_reused(tmp_path, zone_repo):
    """Tiles with holes get a mask; a hole-free tile is left out; a retry reads the first mask back."""
    session = zone_repo.writable_session("main")
    scales = zarr.open_group(session.store, mode="a")["01N"]["scales"]
    scales[0] = np.ones((8, 8), dtype="float32")
    scales[0, :2, :2] = np.nan
    session.commit("one hole in tile (0, 0)")
    chunks = enumerate_chunks(8, 8, 4)
    prefix = str(tmp_path / "holes")

    masks = upd.stage_hole_masks(_node(zone_repo), year_index=0, chunks=chunks, prefix=prefix)

    assert list(masks) == ["chunk_0_0"]
    assert read_pixel_mask(masks["chunk_0_0"]).sum() == 4
    session = zone_repo.writable_session("main")
    zarr.open_group(session.store, mode="a")["01N"]["scales"][0, :2, :2] = 1.0
    session.commit("the update filled it")
    assert upd.stage_hole_masks(_node(zone_repo), year_index=0, chunks=chunks, prefix=prefix) == masks


def test_a_base_tag_race_lost_to_another_cluster_still_succeeds(zone_repo, monkeypatch):
    """Two clusters plan cells of one update at once; the one whose create collides finds the tag pinned."""

    def collide(repo, tag, sid, *, explicit):
        repo.create_tag(tag, sid)
        raise icechunk.IcechunkError("tag already exists")

    monkeypatch.setattr(campaign, "_ensure_tag", collide)
    assert pin_tag(zone_repo, upd.base_tag("fix-1")) == upd.base_tag("fix-1")


@pytest.mark.parametrize(
    ("mode", "allow_s2_only", "refused"), [("holes", False, True), ("holes", True, False), ("shards", False, False)]
)
def test_a_holes_update_needs_the_campaigns_radar_policy(mode, allow_s2_only, refused):
    plan = _plan(mode=mode)
    if refused:
        with pytest.raises(ValueError, match="allow_s2_only"):
            upd.check_radar_policy(plan, allow_s2_only=allow_s2_only)
    else:
        upd.check_radar_policy(plan, allow_s2_only=allow_s2_only)
