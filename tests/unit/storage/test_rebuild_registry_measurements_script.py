"""The registry repair script's ``repair`` and ``verify``, driven through its own ``main`` over a local tree.

``repair`` rewrites what a consumer reads, and could put a re-derived number where the fill
published a measured one. It is exercised here against a real Parquet tree and a real (small) Zarr
store on disk rather than against its internals, because the thing most likely to be wrong is the
plumbing — which rows are re-derived, which survive, and whether a blocked cell is left alone. The
arithmetic of a re-derivation is covered in ``test_registry_rebuild.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import zarr

from tessera_embeddings.storage import published_store
from tessera_embeddings.storage.published_store import Commit
from tessera_embeddings.storage.registry import registry_schema
from tessera_embeddings.storage.registry_master import (
    MEASURED_BY_FILL,
    MEASURED_BY_REBUILD,
    as_master_rows,
    cell_version,
    master_schema,
    merge,
    write_cell,
)
from tessera_embeddings.storage.shard_writer import run_provenance

_SCRIPT = Path(__file__).parents[3] / "scripts" / "maintenance" / "rebuild_registry_measurements.py"
FILLED = "2026-08-01T00:00:00+00:00"
LOCAL = fsspec.filesystem("file")
#: The store's record of cell 33N/2017 as v1.1 left it: one run, recorded before commits were.
V11_RUNS = {"2017": {"run_id": "fill-1", "assembled_at": FILLED}}


@pytest.fixture(scope="module")
def script() -> Any:
    """The script as an importable module, loaded from its path like an operator invokes it."""
    spec = importlib.util.spec_from_file_location("rebuild_registry_measurements", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(tile: str, *, measured: bool, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {field.name: None for field in master_schema()}
    row |= {"zone": "33N", "year": 2017, "tile": tile, "run_id": "fill-1", "assembled_at": FILLED}
    row |= {"embedded": True, "optical_min_obs": 15}
    if measured:
        row |= {
            "eligible_px": 4194304,
            "chunk_px": 4194304,
            "refused_px": 10,
            "refused_no_optical_px": 4,
            "refused_thin_px": 6,
            "refused_no_radar_px": 0,
            "px_with_any_optical": 4194300,
            "obs_max": 50,
            "median_obs_where_any": 40.0,
            "radar_rule_enforced": False,
        }
    return row | extra


#: One cell as v1.1 published it: two tiles the fill measured, two it left null.
FILL_ROWS = [
    _row("chunk_0_0", measured=True),
    _row("chunk_0_1", measured=True),
    _row("chunk_1_0", measured=False),
    _row("chunk_1_1", measured=False),
]


def _seed(root: Path, rows: list[dict[str, Any]] = FILL_ROWS, run: str = "fill-1") -> None:
    cell = root / "runs" / "zone=33N" / "year=2017"
    cell.mkdir(parents=True, exist_ok=True)
    trimmed = [{field.name: row.get(field.name) for field in registry_schema()} for row in rows]
    pq.write_table(pa.Table.from_pylist(trimmed, schema=registry_schema()), cell / f"{run}.parquet")


def _seed_master(root: Path, *, radar: int | None) -> None:
    """The cell's master as a previous repair left it: the null tiles re-derived."""
    rebuilt = [
        _row(t, measured=True, run_id="rebuild-0", filled_at=FILLED, eligible_px=None, px_with_any_radar=radar)
        for t in ("chunk_1_0", "chunk_1_1")
    ]
    table = pa.Table.from_pylist(rebuilt, schema=master_schema())
    fill = as_master_rows(pa.Table.from_pylist(FILL_ROWS, schema=master_schema()), MEASURED_BY_FILL)
    write_cell(LOCAL, str(root), "33N", 2017, merge(fill, as_master_rows(table, MEASURED_BY_REBUILD)), version=None)


@pytest.fixture
def store(script: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> zarr.Group:
    """A one-zone store that holds run ``fill-1`` for 2017, nothing for 2018, and never imaged anything."""
    group = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    group.create_array("time", data=np.array(["2017-01-01", "2018-01-01"], dtype="datetime64[ns]").astype("int64"))
    for name in ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count"):
        group.create_array(name, shape=(2, 4096, 4096), dtype="uint16", chunks=(1, 2048, 2048))
    group.attrs["runs"] = V11_RUNS
    monkeypatch.setattr(script, "_open_store", lambda *_args, **_kwargs: (None, {"33N": group}, dict))
    return group


def _master(root: Path) -> dict[str, dict[str, Any]]:
    table = ds.dataset(str(root / "master"), partitioning="hive", schema=master_schema()).to_table()
    return {row["tile"]: row for row in table.to_pylist()}


def _repair(script: Any, root: Path, *flags: str) -> int:
    return script.main(["repair", "--registry", str(root), "--no-in-cell-gate", *flags])


class TestRepair:
    """Rebuilding a cell from its run files, its master rows and a fresh re-derivation."""

    def test_a_dry_run_writes_nothing(self, script: Any, store: zarr.Group, tmp_path: Path) -> None:
        """Every writing subcommand is a dry run without ``--write`` — the default has to be safe."""
        _seed(tmp_path)
        assert _repair(script, tmp_path) == 0
        assert not (tmp_path / "master").exists()

    def test_null_rows_are_re_derived_and_fill_measurements_kept(
        self, script: Any, store: zarr.Group, tmp_path: Path
    ) -> None:
        _seed(tmp_path)
        assert _repair(script, tmp_path, "--write") == 0
        assert _repair(script, tmp_path, "--write") == 0, "and running it again replaces rather than accumulates"
        tiles = _master(tmp_path)
        assert sorted(tiles) == ["chunk_0_0", "chunk_0_1", "chunk_1_0", "chunk_1_1"]
        assert (tiles["chunk_0_0"]["measured_by"], tiles["chunk_0_0"]["refused_thin_px"]) == (MEASURED_BY_FILL, 6)
        rebuilt = tiles["chunk_1_0"]
        assert (rebuilt["measured_by"], rebuilt["refused_no_optical_px"]) == (MEASURED_BY_REBUILD, 2048 * 2048)
        assert (rebuilt["eligible_px"], rebuilt["filled_at"]) == (None, FILLED)
        assert pq.read_schema(tmp_path / "master" / "_common_metadata").names == master_schema().names
        assert script.main(["verify", "--registry", str(tmp_path)]) == 0

    def test_an_optical_only_repair_cannot_undo_a_radar_measurement(
        self, script: Any, store: zarr.Group, tmp_path: Path
    ) -> None:
        _seed(tmp_path)
        _seed_master(tmp_path, radar=123)
        assert _repair(script, tmp_path, "--skip-radar", "--write") == 0
        assert _master(tmp_path)["chunk_1_0"]["px_with_any_radar"] == 123

    def test_a_master_file_whose_last_run_file_was_withdrawn_is_emptied(
        self, script: Any, store: zarr.Group, tmp_path: Path
    ) -> None:
        """No run file is left to list the cell, so the pass has to find it from the master file itself."""
        _seed(tmp_path)
        withdrawn = pa.Table.from_pylist([_row("chunk_0_0", measured=True, year=2018)], schema=master_schema())
        write_cell(LOCAL, str(tmp_path), "33N", 2018, as_master_rows(withdrawn, MEASURED_BY_FILL), version=None)
        assert script.main(["verify", "--registry", str(tmp_path)]) == 1
        assert _repair(script, tmp_path, "--write") == 0
        assert pq.read_table(tmp_path / "master" / "zone=33N" / "year=2018" / "part-0.parquet").num_rows == 0
        assert script.main(["verify", "--registry", str(tmp_path)]) == 0

    @pytest.mark.parametrize("null_rows", [True, False], ids=["with-rows-to-re-derive", "fully-measured"])
    def test_a_cell_the_store_cannot_stand_behind_is_left_as_it_was(
        self, script: Any, store: zarr.Group, tmp_path: Path, null_rows: bool
    ) -> None:
        """A store holding a run the registry has no run file for: the run files no longer describe the cell.

        Rewriting it would drop the measurements it holds, or, when nothing is null, report success on
        rows the store has moved past.
        """
        _seed(tmp_path, FILL_ROWS if null_rows else [_row(row["tile"], measured=True) for row in FILL_ROWS])
        _seed_master(tmp_path, radar=None)
        before = (tmp_path / "master" / "zone=33N" / "year=2017" / "part-0.parquet").read_bytes()
        store.attrs["runs"] = {"2017": {"run_id": "a-refill-the-registry-never-saw", "assembled_at": FILLED}}
        assert _repair(script, tmp_path, "--write") == 1
        assert (tmp_path / "master" / "zone=33N" / "year=2017" / "part-0.parquet").read_bytes() == before


@pytest.mark.parametrize("command", ["gate", "repair", "verify"])
def test_a_registry_without_run_files_fails_rather_than_passing_vacuously(
    script: Any, tmp_path: Path, command: str
) -> None:
    """A mistyped ``--registry`` must not read as an empty registry that every check passes."""
    with pytest.raises(SystemExit, match="runs does not exist"):
        script.main([command, "--registry", str(tmp_path / "typo")])


@pytest.mark.parametrize("command", ["gate", "repair"])
def test_a_zone_the_registry_does_not_hold_fails(script: Any, store: zarr.Group, tmp_path: Path, command: str) -> None:
    """A mistyped zone in a zone-split run would otherwise select nothing and report success."""
    _seed(tmp_path)
    with pytest.raises(SystemExit, match="99X"):
        script.main([command, "--registry", str(tmp_path), "--zones", "33N,99X"])


def test_repair_reaches_a_zone_with_no_run_file_left(
    script: Any, store: zarr.Group, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every run file of zone 34N was withdrawn, so only its master file names it; ``--zones`` must still find it."""
    _seed(tmp_path)
    withdrawn = pa.Table.from_pylist([_row("chunk_0_0", measured=True, zone="34N")], schema=master_schema())
    write_cell(LOCAL, str(tmp_path), "34N", 2017, as_master_rows(withdrawn, MEASURED_BY_FILL), version=None)
    zone = zarr.open_group(str(tmp_path / "34N.zarr"), mode="w")
    zone.create_array("time", data=store["time"][:])
    monkeypatch.setattr(script, "_open_store", lambda *_args, **_kwargs: (None, {"33N": store, "34N": zone}, dict))
    assert _repair(script, tmp_path, "--zones", "34N", "--write") == 0
    assert pq.read_table(tmp_path / "master" / "zone=34N" / "year=2017" / "part-0.parquet").num_rows == 0


@pytest.mark.parametrize(
    ("refill_wrote", "expected"),
    [({"chunk_1_1"}, 0), ({"chunk_0_0"}, 1)],
    ids=["refilled-tiles-since-rewritten-repair", "refilled-tile-still-held-blocks"],
)
def test_a_refill_that_never_published_blocks_only_while_it_holds_a_tile(
    script: Any,
    store: zarr.Group,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refill_wrote: set[str],
    expected: int,
) -> None:
    """A refill that died after its shard commit, so it published no run file and never reached the year's
    ``runs`` record, then an update that landed whole. Only the refill's tagged commit in the store's
    history shows it at all, and only its diff says whether a tile it wrote is still its own.
    """
    update_at = "2026-08-09T00:00:00+00:00"
    _seed(tmp_path)
    _seed(tmp_path, [_row("chunk_1_1", measured=True, run_id="update", assembled_at=update_at)], run="update")
    _seed_master(tmp_path, radar=None)
    store.attrs["runs"] = run_provenance(
        V11_RUNS, 2017, "update", commit={"snapshot": "s-c", "committed_at": update_at}
    )
    writes = {("33N", 2017): [Commit("refill", "2026-08-05T00:00:00+00:00", "s-b"), Commit("update", update_at, "s-c")]}
    monkeypatch.setattr(script, "_open_store", lambda *_args, **_kwargs: (None, {"33N": store}, lambda: writes))
    wrote = {"s-b": refill_wrote, "s-c": {"chunk_1_1"}}
    monkeypatch.setattr(published_store, "commit_tiles", lambda _repo, _zone, _t, snapshot: frozenset(wrote[snapshot]))
    master = tmp_path / "master" / "zone=33N" / "year=2017" / "part-0.parquet"
    before = master.read_bytes()
    assert _repair(script, tmp_path, "--write") == expected
    assert script.main(["verify", "--registry", str(tmp_path)]) == expected
    assert (master.read_bytes() == before) == bool(expected), "a blocked cell is left exactly as it was"


class TestVerify:
    """The master itself, checked against the run files it should describe."""

    def test_a_sound_master_verifies(self, script: Any, store: zarr.Group, tmp_path: Path) -> None:
        _seed(tmp_path)
        _seed_master(tmp_path, radar=None)
        assert script.main(["verify", "--registry", str(tmp_path)]) == 0

    @pytest.mark.parametrize("zones", [[], ["--zones", "35N"]], ids=["every-zone", "only-that-zone"])
    def test_a_first_fill_whose_run_file_never_published_fails(
        self,
        script: Any,
        store: zarr.Group,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
        zones: list[str],
    ) -> None:
        """Its zone has no run file and no master file, so only the store's tagged commit shows it, named or not."""
        _seed(tmp_path)
        _seed_master(tmp_path, radar=None)
        zone = zarr.open_group(str(tmp_path / "35N.zarr"), mode="w")
        zone.create_array("time", data=store["time"][:])
        writes = {("35N", 2017): [Commit("first-fill", FILLED, "s-a")]}
        root = {"33N": store, "35N": zone}
        monkeypatch.setattr(script, "_open_store", lambda *_args, **_kwargs: (None, root, lambda: writes))
        monkeypatch.setattr(published_store, "commit_tiles", lambda *_args: frozenset({"chunk_0_0"}))
        assert script.main(["verify", "--registry", str(tmp_path), *zones]) == 1
        assert "35N/2017: the store cannot stand behind its run files: run 'first-fill'" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "damage",
        [
            lambda rows: [row | {"refused_thin_px": 999} if row["tile"] == "chunk_0_0" else row for row in rows],
            lambda rows: [row for row in rows if row["tile"] != "chunk_1_1"],
        ],
        ids=["an-altered-fill-measurement", "a-missing-tile"],
    )
    def test_a_damaged_master_fails(self, script: Any, store: zarr.Group, tmp_path: Path, damage: Any) -> None:
        _seed(tmp_path)
        _seed_master(tmp_path, radar=None)
        rows = damage(list(_master(tmp_path).values()))
        damaged = pa.Table.from_pylist(rows, schema=master_schema())
        assert write_cell(
            LOCAL, str(tmp_path), "33N", 2017, damaged, version=cell_version(LOCAL, str(tmp_path), "33N", 2017)
        )
        assert script.main(["verify", "--registry", str(tmp_path)]) == 1

    def test_zones_narrow_what_is_checked(self, script: Any, store: zarr.Group, tmp_path: Path) -> None:
        """A zone-split verification must pass or fail on its own zones, and a mistyped one must not pass."""
        _seed(tmp_path)
        _seed_master(tmp_path, radar=None)
        stray = pa.Table.from_pylist([_row("chunk_0_0", measured=True, zone="34N")], schema=master_schema())
        write_cell(LOCAL, str(tmp_path), "34N", 2017, as_master_rows(stray, MEASURED_BY_FILL), version=None)
        assert script.main(["verify", "--registry", str(tmp_path)]) == 1
        assert script.main(["verify", "--registry", str(tmp_path), "--zones", "33N"]) == 0
        with pytest.raises(SystemExit, match="99X"):
            script.main(["verify", "--registry", str(tmp_path), "--zones", "33N,99X"])
