"""The registry rename script, driven through its own ``main`` over a local registry tree.

It deletes the only copy of every run file once the copies check, so what is tested is that it
never deletes before then and never overwrites what it has not looked at: a dry run, a move, a
rerun, a damaged copy and a rollback, each over a real directory tree through the same fsspec calls
an S3 run makes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import fsspec
import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "maintenance" / "rename_registry_parts_to_runs.py"
#: Run files as a fill leaves them, by path below the folder.
FILES = {
    "zone=33N/year=2017/fill-1.parquet": b"first fill",
    "zone=33N/year=2018/fill-2.parquet": b"second fill",
    "zone=34S/year=2017/update-1.parquet": b"an update",
}
MASTER = Path("master") / "zone=33N" / "year=2017" / "part-0.parquet"


@pytest.fixture(scope="module")
def script() -> Any:
    """The script as an importable module, loaded from its path like an operator invokes it."""
    spec = importlib.util.spec_from_file_location("rename_registry_parts_to_runs", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(root: Path, folder: str, files: dict[str, bytes]) -> None:
    for key, data in files.items():
        (root / folder / key).parent.mkdir(parents=True, exist_ok=True)
        (root / folder / key).write_bytes(data)


def _tree(root: Path, folder: str) -> dict[str, bytes]:
    base = root / folder
    return {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*") if p.is_file()}


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    """A registry from before the rename: its run files under ``parts/``, and a master file beside them."""
    _write(tmp_path, "parts", FILES)
    _write(tmp_path, ".", {MASTER.as_posix(): b"master"})
    return tmp_path


def _run(script: Any, registry: Path, *flags: str) -> int:
    return script.main(["--registry", str(registry), *flags])


def test_a_dry_run_changes_nothing_and_lists_the_move(script: Any, registry: Path, capsys: Any) -> None:
    assert _run(script, registry) == 0
    assert _tree(registry, "parts") == FILES
    assert not (registry / "runs").exists()
    assert (
        "parts/zone=33N/year=2017/fill-1.parquet  ->  runs/zone=33N/year=2017/fill-1.parquet" in capsys.readouterr().out
    )


def test_apply_moves_every_file_and_a_rerun_is_a_no_op(script: Any, registry: Path, capsys: Any) -> None:
    assert _run(script, registry, "--apply") == 0
    assert _tree(registry, "runs") == FILES
    assert not (registry / "parts").exists(), "the emptied folder goes too"
    assert (registry / MASTER).read_bytes() == b"master", "master/ is not touched"
    assert f"verify --registry {registry}" in capsys.readouterr().out, "and it says what to run next"
    assert _run(script, registry, "--apply") == 0
    assert "already migrated" in capsys.readouterr().out
    assert _tree(registry, "runs") == FILES


def test_a_rerun_after_a_crash_keeps_the_matching_copies_and_finishes(script: Any, registry: Path) -> None:
    """Killed after one copy and before any delete: the copy matches, so it stays and the rest move."""
    _write(registry, "runs", {"zone=33N/year=2017/fill-1.parquet": b"first fill"})
    assert _run(script, registry, "--apply") == 0
    assert _tree(registry, "runs") == FILES
    assert not (registry / "parts").exists()


@pytest.mark.parametrize("tampered", [b"short", b"SECOND FILL"], ids=["a-different-size", "the-same-size"])
def test_a_destination_that_differs_is_refused_before_anything_is_written(
    script: Any, registry: Path, capsys: Any, tampered: bytes
) -> None:
    key = "zone=33N/year=2018/fill-2.parquet"
    _write(registry, "runs", {key: tampered})
    assert _run(script, registry, "--apply") == 1
    assert f"runs/{key}" in capsys.readouterr().out, "the refusal names the file"
    assert _tree(registry, "parts") == FILES
    assert _tree(registry, "runs") == {key: tampered}


def test_a_copy_that_does_not_check_leaves_every_source(script: Any, registry: Path, monkeypatch: Any) -> None:
    """A copy that lands short must stop the move before a single source is deleted."""
    fs = fsspec.filesystem("file", auto_mkdir=True, skip_instance_cache=True)

    def truncating(source: str, dest: str, **_: Any) -> None:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(Path(source).read_bytes()[:-1])

    monkeypatch.setattr(fs, "cp_file", truncating)
    monkeypatch.setattr(script, "_filesystem", lambda *_args, **_kwargs: fs)
    assert _run(script, registry, "--apply") == 1
    assert _tree(registry, "parts") == FILES


def test_rollback_puts_parts_back(script: Any, registry: Path) -> None:
    assert _run(script, registry, "--apply") == 0
    assert _run(script, registry, "--rollback") == 0
    assert _tree(registry, "runs") == FILES, "a rollback is a dry run too without --apply"
    assert _run(script, registry, "--rollback", "--apply") == 0
    assert _tree(registry, "parts") == FILES
    assert not (registry / "runs").exists()


def test_a_registry_with_neither_folder_is_refused(script: Any, tmp_path: Path) -> None:
    """A mistyped ``--registry`` must not read as a registry already migrated."""
    assert _run(script, tmp_path / "typo", "--apply") == 1


def test_an_apply_to_the_published_bucket_without_its_writer_role_is_refused_before_any_read(
    script: Any, monkeypatch: Any
) -> None:
    """Our own identity cannot write that bucket, so the move must stop before it lists anything."""
    monkeypatch.delenv("PUBLISHED_STORE_WRITER_ROLE_ARN", raising=False)
    monkeypatch.setattr(script, "_filesystem", lambda *_args, **_kwargs: pytest.fail("opened the bucket"))
    assert script.main(["--registry", script.DEFAULT_REGISTRY, "--apply"]) == 1
