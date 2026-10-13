"""``scripts/update/loosen_optical_min_obs.py``: a dry run refuses what a write would, as does an empty ``--zones``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from tests._paths import REPO_ROOT
from tests.unit.storage.test_global_store import _root, _ruled

_SCRIPT = REPO_ROOT / "scripts" / "update" / "loosen_optical_min_obs.py"


@pytest.fixture(scope="module")
def script() -> Any:
    spec = importlib.util.spec_from_file_location("loosen_optical_min_obs", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("zones", ["", ", "])
def test_an_empty_zone_list_is_refused_not_read_as_the_whole_store(script: Any, tmp_path: Path, zones: str) -> None:
    """A root loosen cannot be undone, so ``--zones "$UNSET"`` must not become one."""
    repo = _ruled(tmp_path)
    with pytest.raises(SystemExit):
        script.main(["--store", str(tmp_path / "g.icechunk"), "--line", "1", "--zones", zones, "--write"])
    assert _root(repo).attrs["optical_min_obs"] == 15


@pytest.mark.parametrize(("line", "code"), [(1, 0), (0, 1), (20, 1)])
def test_a_dry_run_refuses_what_a_write_would_and_commits_nothing(
    script: Any, tmp_path: Path, line: int, code: int
) -> None:
    """A loosen passes, a line of zero and a tightening are refused, and none of them moves the store."""
    repo = _ruled(tmp_path)
    tip = repo.lookup_branch("main")
    assert script.main(["--store", str(tmp_path / "g.icechunk"), "--line", str(line)]) == code
    assert repo.lookup_branch("main") == tip
