"""The quality ratchet's duplication rule (``.github/scripts/quality_ratchet.py``).

Code a PR writes may not be a copy; code it did not write is never judged, however jscpd pairs it.
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "quality_ratchet.py"
_spec = importlib.util.spec_from_file_location("quality_ratchet", SCRIPT)
ratchet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ratchet)


def _clone(a: tuple[str, int, int], b: tuple[str, int, int]) -> dict:
    return {
        "firstFile": {"name": a[0], "start": a[1], "end": a[2]},
        "secondFile": {"name": b[0], "start": b[1], "end": b[2]},
    }


def _lines(first: int, last: int) -> set[int]:
    return set(range(first, last + 1))


@pytest.mark.parametrize(
    ("clones", "written", "flagged"),
    [
        # Deleting one of three copies re-pairs the survivors; the PR wrote none of them.
        ([_clone(("a.py", 1, 10), ("c.py", 1, 10))], {}, []),
        # A new file copying an existing one.
        ([_clone(("b.py", 1, 10), ("new.py", 1, 10))], {"new.py": _lines(1, 10)}, ["new.py"]),
        # An edited file swaps a copy of c.py for a copy of untouched b.py.
        ([_clone(("a.py", 1, 10), ("b.py", 1, 10))], {"a.py": _lines(1, 10)}, ["a.py"]),
        # Extending an existing copy by 8 lines: under half the side, over the minimum.
        ([_clone(("a.py", 1, 19), ("b.py", 1, 19))], {"a.py": _lines(12, 19)}, ["a.py"]),
        # A short copy git aligned with the lines it replaced: 4 of 7 written.
        ([_clone(("a.py", 1, 7), ("b.py", 1, 7))], {"a.py": {1, 3, 4, 6}}, ["a.py"]),
        # Renaming a variable inside an existing copy touches 3 of its 11 lines.
        ([_clone(("a.py", 1, 11), ("b.py", 1, 11))], {"a.py": {1, 4, 10}}, []),
        # The 3-line short end of a gap clone is not judged on its own.
        ([_clone(("a.py", 1, 3), ("b.py", 1, 8))], {"a.py": _lines(1, 3)}, []),
    ],
    ids=["deletion", "new-file", "swap", "extend", "aligned-short-copy", "rename", "short-side"],
)
def test_a_copy_is_a_clone_side_the_pr_wrote(clones, written, flagged):
    assert [line.split(":")[0] for line in ratchet.copies(clones, written)] == flagged


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=x@x", "-c", "user.name=x", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_written_lines_are_head_line_numbers_and_follow_renames(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "old.py").write_text("".join(f"line_{i} = {i}\n" for i in range(20)))
    (tmp_path / "edited.py").write_text("a = 1\nb = 2\nc = 3\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "old.py").rename(tmp_path / "renamed.py")
    (tmp_path / "edited.py").write_text("a = 1\nb = 20\nc = 3\nd = 4\n")
    (tmp_path / "new.py").write_text("x = 1\ny = 2\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "head")

    assert ratchet.written_lines(tmp_path, "HEAD~1") == {"edited.py": {2, 4}, "new.py": {1, 2}}
