"""Targeted updates through the zone-fill runner: only target shards change, nothing is re-marked."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest
import zarr

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.paths import store_key
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.orchestration.runners import zone_fill
from tessera_embeddings.storage import global_store
from tessera_embeddings.storage.campaign import zone_year_tag
from tessera_embeddings.storage.update import UpdateTarget, base_tag
from tests.unit.orchestration.runners.test_zone_fill import (
    _BAND,
    _TILE,
    _ZONE,
    _config,
    _fill,
    _make_mask,
    _make_mosaic,
    _seed_global,
    _staging_inference_stub,
)

_LIVE = [(0, 0), (0, 1), (1, 1)]
#: The campaign's radar policy, which a holes update must share (`check_radar_policy`).
_S2_ONLY = InferenceConfig(time_window=_config().time_window, chunk_size=_TILE, num_gpus=0, allow_s2_only=True)
_YEAR_INDEX = 1  # 2025 on the (2024, 2025) axis


def _stub(seed: int, *, skip: tuple[str, ...] = ()):
    """A run_inference stand-in staging fresh values from ``seed``, and skip markers for ``skip``."""

    def stub(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        rng = np.random.default_rng(seed)
        writer = ZarrWriter(staging_base, embedding_dim=_BAND)
        results = []
        for chunk in chunks:
            if chunk.label in skip:
                writer.write_skip_marker(chunk, run_id)
                results.append({"chunk": chunk.label, "status": "skipped", "valid_pixels": 0, "elapsed_sec": 0.0})
                continue
            emb = rng.integers(1, 100, size=(chunk.height, chunk.width, _BAND)).astype(np.int8)
            writer.write_chunk(chunk, emb, run_id, scales=rng.random((chunk.height, chunk.width)).astype(np.float32))
            results.append({"chunk": chunk.label, "status": "success", "valid_pixels": 1, "elapsed_sec": 0.0})
        return results

    return stub


def _year(store: str) -> tuple[zarr.Group, dict[str, np.ndarray]]:
    """The zone group on ``main`` and its 2025 slice of every array."""
    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[_ZONE]
    return node, {v: np.asarray(node[v][_YEAR_INDEX]) for v in ("embeddings", "scales")}


def _tile(values: np.ndarray, row: int, col: int) -> np.ndarray:
    return values[row * _TILE : (row + 1) * _TILE, col * _TILE : (col + 1) * _TILE]


@pytest.fixture
def filled(tmp_path, monkeypatch):
    """A 2025 cell the campaign has filled and tagged, with a hole punched into tile (0, 0)."""
    store = _seed_global(tmp_path)
    mosaic = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, _LIVE)
    monkeypatch.setattr(zone_fill, "run_inference", _staging_inference_stub({}))
    registry = str(tmp_path / "registry")
    _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic, registry_root=registry, run_id="fill1")
    repo = global_store.open_global_repo(store)
    session = repo.writable_session("main")
    node = zarr.open_group(session.store, mode="a")[_ZONE]
    node["scales"][_YEAR_INDEX, :8, :8] = np.nan
    node["embeddings"][_YEAR_INDEX, :8, :8] = 0
    session.commit("punch a hole")
    return {"store": store, "mosaic": mosaic, "mask": mask, "registry": registry, "tmp": tmp_path}


def _update(filled, monkeypatch, target: UpdateTarget, *, seed: int = 11, skip: tuple[str, ...] = (), run_id="upd1"):
    monkeypatch.setattr(zone_fill, "run_inference", _stub(seed, skip=skip))
    return _fill(
        filled["tmp"],
        filled["store"],
        land_mask_path=filled["mask"],
        mosaic_base=filled["mosaic"],
        staging_base=str(filled["tmp"] / "update-staging"),
        registry_root=filled["registry"],
        config=_S2_ONLY,
        run_id=run_id,
        update=target,
    )


def _target(filled, mode="shards", tiles=("chunk_0_1",), on_skip="keep") -> UpdateTarget:
    return UpdateTarget(
        name="u1",
        mode=mode,
        on_skip=on_skip,
        mask_name="global",
        tiles=frozenset(tiles) if tiles is not None else None,
        holes=str(filled["tmp"] / "holes"),
    )


def test_a_shards_update_rewrites_its_target_and_nothing_else(filled, monkeypatch):
    """The target shard is rewritten whole; every other shard, mark, record and tag stays put."""
    before_node, before = _year(filled["store"])
    before_runs, before_complete = dict(before_node.attrs["runs"]), list(before_node.attrs["years_complete"])
    repo = global_store.open_global_repo(filled["store"])
    tip_before, tag_before = repo.lookup_branch("main"), repo.lookup_tag(zone_year_tag(_ZONE, 2025))

    summary = _update(filled, monkeypatch, _target(filled))

    node, after = _year(filled["store"])
    assert summary["written"] == 1
    assert not np.array_equal(_tile(after["embeddings"], 0, 1), _tile(before["embeddings"], 0, 1))
    for row, col in [(0, 0), (1, 1), (1, 0)]:
        for var in after:
            np.testing.assert_array_equal(_tile(after[var], row, col), _tile(before[var], row, col))
    assert dict(node.attrs["runs"]) == before_runs
    assert list(node.attrs["years_complete"]) == before_complete
    [record] = node.attrs["updates"]["2025"]
    assert (record["update"], record["written"], record["kept"]) == ("u1", ["chunk_0_1"], [])
    repo = global_store.open_global_repo(filled["store"])
    assert repo.lookup_tag(zone_year_tag(_ZONE, 2025)) == tag_before
    assert repo.lookup_tag(base_tag("u1")) == tip_before


def test_a_holes_update_fills_only_the_holes(filled, monkeypatch):
    """Hole pixels take the new run's values; every embedded pixel, and a hole-free tile, is untouched."""
    _, before = _year(filled["store"])

    summary = _update(filled, monkeypatch, _target(filled, mode="holes", tiles=("chunk_0_0", "chunk_1_1")))

    node, after = _year(filled["store"])
    hole = np.zeros((_TILE, _TILE), dtype=bool)
    hole[:8, :8] = True
    assert summary["written"] == 1
    assert not np.isnan(_tile(after["scales"], 0, 0)[hole]).any()
    for var in after:
        np.testing.assert_array_equal(_tile(after[var], 0, 0)[~hole], _tile(before[var], 0, 0)[~hole])
        np.testing.assert_array_equal(_tile(after[var], 1, 1), _tile(before[var], 1, 1))
    [record] = node.attrs["updates"]["2025"]
    assert (record["mode"], record["written"], record["kept"]) == ("holes", ["chunk_0_0"], ["chunk_1_1"])
    # The masks are filed under the destination store, so a plan run against another store reads its own.
    [store_dir] = list((filled["tmp"] / "holes").iterdir())
    assert store_dir.name == store_key(filled["store"])
    assert sorted(p.name for p in store_dir.iterdir()) == ["chunk_0_0.npz", "chunk_1_1.npz"]
    # The merged tile's row is now the update's, stamped with its shard commit so the store stands
    # behind it; the rest stay the fill's.
    master = pq.read_table(f"{filled['registry']}/master/zone={_ZONE}/year=2025/part-0.parquet").to_pylist()
    assert {r["tile"]: (r["run_id"], r["update_name"]) for r in master} == {
        "chunk_0_0": ("upd1", "u1"),
        "chunk_0_1": ("fill1", None),
        "chunk_1_1": ("fill1", None),
    }


@pytest.mark.parametrize(("on_skip", "cleared"), [("keep", False), ("clear", True)])
def test_a_target_that_embeds_nothing_is_kept_unless_the_plan_clears_it(filled, monkeypatch, on_skip, cleared):
    """A skipped target keeps its data by default; a clearing update blanks it, for defect fixes."""
    _, before = _year(filled["store"])

    _update(filled, monkeypatch, _target(filled, tiles=("chunk_0_1",), on_skip=on_skip), skip=("chunk_0_1",))

    node, after = _year(filled["store"])
    if cleared:
        assert np.isnan(_tile(after["scales"], 0, 1)).all()
    else:
        np.testing.assert_array_equal(_tile(after["scales"], 0, 1), _tile(before["scales"], 0, 1))
    [record] = node.attrs["updates"]["2025"]
    assert (record["kept"], record["cleared"]) == (([], ["chunk_0_1"]) if cleared else (["chunk_0_1"], []))


def test_a_holes_update_refuses_the_radar_rule(filled, monkeypatch):
    """Its registry rows are re-derived from the store, which cannot reproduce a radar refusal."""
    monkeypatch.setattr(zone_fill, "run_inference", lambda *a, **k: pytest.fail("inferred under the radar rule"))
    with pytest.raises(ValueError, match="allow_s2_only"):
        _fill(filled["tmp"], filled["store"], land_mask_path=filled["mask"], update=_target(filled, mode="holes"))


def test_an_update_already_recorded_does_nothing(filled, monkeypatch):
    """Re-running an update is free: the cell is recorded, so no inference and no new snapshot."""
    _update(filled, monkeypatch, _target(filled))
    tip = global_store.open_global_repo(filled["store"]).lookup_branch("main")
    monkeypatch.setattr(zone_fill, "run_inference", lambda *a, **k: pytest.fail("a recorded update re-inferred"))

    summary = _fill(
        filled["tmp"],
        filled["store"],
        land_mask_path=filled["mask"],
        mosaic_base=filled["mosaic"],
        run_id="upd2",
        update=_target(filled),
    )

    assert summary["already_recorded"] is True
    assert global_store.open_global_repo(filled["store"]).lookup_branch("main") == tip


def test_an_update_refuses_a_year_the_campaign_never_filled(filled, monkeypatch):
    """An update inserts into existing data; filling an empty year is the campaign's job."""
    with pytest.raises(ValueError, match="has not been filled"):
        _fill(
            filled["tmp"],
            filled["store"],
            year=2024,
            config=_config(2024),
            land_mask_path=filled["mask"],
            mosaic_base=filled["mosaic"],
            update=_target(filled),
        )


def test_an_update_whose_shards_are_dead_records_itself_without_writing(filled, monkeypatch):
    """No live target is still an answer: the cell is recorded done and nothing is written or marked."""
    tip = global_store.open_global_repo(filled["store"]).lookup_branch("main")
    monkeypatch.setattr(zone_fill, "run_inference", lambda *a, **k: pytest.fail("a dead target was inferred"))

    summary = _fill(
        filled["tmp"],
        filled["store"],
        land_mask_path=filled["mask"],
        mosaic_base=filled["mosaic"],
        update=_target(filled, tiles=("chunk_1_0",)),
    )

    node, _ = _year(filled["store"])
    assert summary["live_tiles"] == 0
    assert node.attrs["updates"]["2025"][0]["written"] == []
    repo = global_store.open_global_repo(filled["store"])
    assert repo.list_tags() >= {base_tag("u1")} and repo.lookup_branch("main") != tip


def test_an_out_of_grid_shard_is_refused_before_inference(filled, monkeypatch):
    """A shard list for another grid fails loudly instead of being silently dropped."""
    monkeypatch.setattr(zone_fill, "run_inference", lambda *a, **k: pytest.fail("inferred an off-grid plan"))
    with pytest.raises(ValueError, match="outside the zone's"):
        _fill(
            filled["tmp"],
            filled["store"],
            land_mask_path=filled["mask"],
            mosaic_base=filled["mosaic"],
            update=_target(filled, tiles=("chunk_9_9",)),
        )
