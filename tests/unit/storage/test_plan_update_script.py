"""``scripts/update/plan_update.py``: a plan is checked against its mask before it is written."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.storage.update import load_update
from tests._paths import REPO_ROOT
from tests.unit.coverage_repo import make_coverage

_SCRIPT = REPO_ROOT / "scripts" / "update" / "plan_update.py"


@pytest.fixture(scope="module")
def script() -> Any:
    spec = importlib.util.spec_from_file_location("plan_update", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv(tmp_path: Path, rows: list[str]) -> str:
    path = tmp_path / "cells.csv"
    path.write_text("zone,year,tile\n" + "\n".join(rows) + "\n")
    return str(path)


def test_a_plan_from_cells_is_checked_then_written(script: Any, tmp_path: Path, capsys) -> None:
    inputs = tmp_path / "in"
    make_coverage(inputs / "masks", "31N", [(10, 5), (11, 5)])  # -> masks/coverage.icechunk
    cells = _csv(tmp_path, ["31N,2024,chunk_10_5", "31N,2024,chunk_12_5", "31N,2025,"])
    argv = ["from-cells", "--name", "u1", "--mode", "shards", "--inputs", str(inputs), "--mask", "coverage"]

    assert script.main([*argv, "--cells", cells]) == 0
    assert "Dry run" in capsys.readouterr().out
    assert script.main([*argv, "--cells", cells, "--write"]) == 0

    out = capsys.readouterr().out
    assert "31N/2024       2 planned       1 live       1 dead" in out
    assert "31N/2025       2 planned       2 live" in out
    plan = load_update(BucketPaths(inputs=str(inputs), outputs=""), "u1", mask_name="coverage")
    assert plan.cells == {"31N/2024": ("chunk_10_5", "chunk_12_5"), "31N/2025": None}


def test_a_plan_for_another_grid_is_refused(script: Any, tmp_path: Path) -> None:
    inputs = tmp_path / "in"
    make_coverage(inputs / "masks", "31N", [(10, 5)])
    argv = ["from-cells", "--name", "u1", "--mode", "shards", "--inputs", str(inputs), "--mask", "coverage"]
    with pytest.raises(ValueError, match="outside the zone's"):
        script.main([*argv, "--cells", _csv(tmp_path, ["31N,2024,chunk_9999_0"])])
