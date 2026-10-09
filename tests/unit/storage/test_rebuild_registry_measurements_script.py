"""The registry backfill script's compaction, driven through its own ``main`` over a local tree.

The compaction is the only step that decides what a consumer reads, and the only one that could
put a re-derived number where the fill published a measured one. It is exercised here against a
real Parquet tree on disk rather than against its internals, because the thing most likely to be
wrong is the plumbing between the prefixes — which schema is stated where, which rows a dataset
read picks up, and whether a missing prefix reads as empty or as a failure.

The rebuild subcommand is not driven here: it opens the published Icechunk store, and the
arithmetic it performs is covered against a real (small) Zarr group in
``test_registry_rebuild.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from tessera_embeddings.storage.registry import registry_schema
from tessera_embeddings.storage.registry_master import master_schema
from tessera_embeddings.storage.registry_rebuild import rebuild_schema

_SCRIPT = Path(__file__).parents[3] / "scripts" / "maintenance" / "rebuild_registry_measurements.py"


@pytest.fixture(scope="module")
def script() -> Any:
    """The script as an importable module, loaded from its path like an operator invokes it."""
    spec = importlib.util.spec_from_file_location("rebuild_registry_measurements", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _measured() -> dict[str, Any]:
    return {
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


def _row(schema: pa.Schema, tile: str, *, measured: bool, stamp: str, run: str, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {field.name: None for field in schema}
    row |= {"tile": tile, "run_id": run, "assembled_at": stamp, "embedded": True, "optical_min_obs": 15}
    if measured:
        row |= _measured()
    row |= extra
    return row


def _seed(root: Path, *, with_rebuild: bool = True) -> None:
    """A one-cell registry: two tiles the fill measured, two it left null, and a rebuild for both."""
    parts = [
        _row(registry_schema(), "chunk_0_0", measured=True, stamp="2026-08-01T00:00:00+00:00", run="fill-1"),
        _row(registry_schema(), "chunk_0_1", measured=True, stamp="2026-08-01T00:00:00+00:00", run="fill-1"),
        _row(registry_schema(), "chunk_1_0", measured=False, stamp="2026-08-01T00:00:00+00:00", run="fill-1"),
        _row(registry_schema(), "chunk_1_1", measured=False, stamp="2026-08-01T00:00:00+00:00", run="fill-1"),
    ]
    cell = root / "parts" / "zone=33N" / "year=2017"
    cell.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(parts, schema=registry_schema()), cell / "fill-1.parquet")

    if not with_rebuild:
        return
    rebuilt = [
        _row(
            rebuild_schema(),
            tile,
            measured=True,
            stamp="2026-09-21T00:00:00+00:00",
            run="rebuild-1",
            eligible_px=None,
            refused_thin_px=5,
            refused_px=7,
            filled_at="2026-08-01T00:00:00+00:00",
        )
        for tile in ("chunk_1_0", "chunk_1_1")
    ]
    cell = root / "rebuild" / "zone=33N" / "year=2017"
    cell.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rebuilt, schema=rebuild_schema()), cell / "rebuild-1.parquet")


def _master(root: Path) -> pa.Table:
    return ds.dataset(str(root / "master"), partitioning="hive", schema=master_schema()).to_table()


class TestCompact:
    """Merging the prefixes into the view a consumer reads."""

    def test_a_dry_run_writes_nothing(self, script: Any, tmp_path: Path) -> None:
        """Every writing subcommand is a dry run without ``--write`` — the default has to be safe."""
        _seed(tmp_path)
        assert script.main(["compact", "--registry", str(tmp_path)]) == 0
        assert not (tmp_path / "master").exists()

    def test_the_master_holds_one_row_per_tile_year_with_provenance(self, script: Any, tmp_path: Path) -> None:
        _seed(tmp_path)
        assert script.main(["compact", "--registry", str(tmp_path), "--write"]) == 0
        master = _master(tmp_path)
        assert master.num_rows == 4
        by_tile = {row["tile"]: row for row in master.to_pylist()}
        assert by_tile["chunk_0_0"]["measured_by"] == "fill"
        assert by_tile["chunk_1_0"]["measured_by"] == "rebuild"

    def test_a_fill_measurement_is_not_replaced_by_a_rebuilt_one(self, script: Any, tmp_path: Path) -> None:
        """The rebuild rows here carry 5; the fill's 6 must survive the merge."""
        _seed(tmp_path)
        script.main(["compact", "--registry", str(tmp_path), "--write"])
        by_tile = {row["tile"]: row for row in _master(tmp_path).to_pylist()}
        assert by_tile["chunk_0_0"]["refused_thin_px"] == 6
        assert by_tile["chunk_1_0"]["refused_thin_px"] == 5

    def test_a_rebuilt_row_leaves_eligible_px_null_and_keeps_the_fills_timestamp(
        self, script: Any, tmp_path: Path
    ) -> None:
        _seed(tmp_path)
        script.main(["compact", "--registry", str(tmp_path), "--write"])
        row = {r["tile"]: r for r in _master(tmp_path).to_pylist()}["chunk_1_0"]
        assert row["eligible_px"] is None
        assert row["filled_at"] == "2026-08-01T00:00:00+00:00"
        assert row["assembled_at"] == "2026-09-21T00:00:00+00:00"

    def test_a_missing_rebuild_prefix_is_the_normal_state_not_a_failure(self, script: Any, tmp_path: Path) -> None:
        """Before the first rebuild pass there is nothing under ``rebuild/`` and compacting must work."""
        _seed(tmp_path, with_rebuild=False)
        assert script.main(["compact", "--registry", str(tmp_path), "--write"]) == 0
        master = _master(tmp_path)
        assert master.num_rows == 4
        assert sorted({r["measured_by"] for r in master.to_pylist()}, key=str) == [None, "fill"]

    def test_a_common_metadata_is_written_beside_the_data(self, script: Any, tmp_path: Path) -> None:
        """Its absence is what forces every consumer to state the schema from a part they chose."""
        _seed(tmp_path)
        script.main(["compact", "--registry", str(tmp_path), "--write"])
        assert (tmp_path / "master" / "_common_metadata").exists()
        assert pq.read_schema(tmp_path / "master" / "_common_metadata").names == master_schema().names

    def test_recompacting_replaces_rather_than_accumulates(self, script: Any, tmp_path: Path) -> None:
        """A second pass must not leave the first's files beside it, which would duplicate rows."""
        _seed(tmp_path)
        script.main(["compact", "--registry", str(tmp_path), "--write"])
        script.main(["compact", "--registry", str(tmp_path), "--write"])
        assert _master(tmp_path).num_rows == 4


@pytest.mark.parametrize("command", ["gate", "rebuild", "compact", "verify"])
def test_a_registry_without_parts_fails_rather_than_passing_vacuously(
    script: Any, tmp_path: Path, command: str
) -> None:
    """A mistyped ``--registry`` must not read as an empty registry that every check passes."""
    with pytest.raises(SystemExit, match="parts does not exist"):
        script.main([command, "--registry", str(tmp_path / "typo")])


class TestVerify:
    """The invariants, checked against the published parts rather than against the merge rule."""

    def test_a_clean_master_verifies(self, script: Any, tmp_path: Path) -> None:
        _seed(tmp_path)
        script.main(["compact", "--registry", str(tmp_path), "--write"])
        assert script.main(["verify", "--registry", str(tmp_path)]) == 0

    def test_a_master_with_an_altered_fill_row_fails(self, script: Any, tmp_path: Path) -> None:
        """Tampering with a published measurement has to be caught by reading the parts, not the merge."""
        _seed(tmp_path)
        script.main(["compact", "--registry", str(tmp_path), "--write"])
        table = _master(tmp_path)
        rows = table.to_pylist()
        for row in rows:
            if row["tile"] == "chunk_0_0":
                row["refused_thin_px"] = 999
        target = next((tmp_path / "master").rglob("*.parquet"))
        pq.write_table(pa.Table.from_pylist(rows, schema=master_schema()), target)
        assert script.main(["verify", "--registry", str(tmp_path)]) == 1

    def test_a_missing_row_fails(self, script: Any, tmp_path: Path) -> None:
        _seed(tmp_path)
        script.main(["compact", "--registry", str(tmp_path), "--write"])
        table = _master(tmp_path)
        rows = [row for row in table.to_pylist() if row["tile"] != "chunk_1_1"]
        target = next((tmp_path / "master").rglob("*.parquet"))
        pq.write_table(pa.Table.from_pylist(rows, schema=master_schema()), target)
        assert script.main(["verify", "--registry", str(tmp_path)]) == 1
