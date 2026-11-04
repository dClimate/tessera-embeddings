"""The quality ratchet's duplication count (``.github/scripts/quality_ratchet.py``).

A clean-up that deletes a copy must never be reported as adding duplication.
"""

import importlib.util
from collections import Counter
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


def _grew(before: list[dict], after: list[dict], changed: set[str], tmp_path: Path) -> list[str]:
    """``grew`` over two trees in which only ``changed`` files differ."""
    base, head = tmp_path / "base", tmp_path / "head"
    names = {side["name"] for c in before + after for side in (c["firstFile"], c["secondFile"])}
    for tree in (base, head):
        tree.mkdir()
        for name in names:
            (tree / name).write_text(f"{name} {tree.name if name in changed else ''}")
    return ratchet.grew(ratchet.duplicated_lines(before), ratchet.duplicated_lines(after), base, head)


def test_deleting_the_shared_copy_does_not_fail(tmp_path):
    """The remaining copies are re-paired against one untouched file, raising its count."""
    x, a, c, d = (("x.py", 1, 10), ("a.py", 1, 10), ("c.py", 1, 10), ("d.py", 1, 10))
    assert _grew([_clone(x, a), _clone(x, c), _clone(x, d)], [_clone(a, c), _clone(a, d)], set(), tmp_path) == []


def test_an_edit_that_copies_one_more_file_fails(tmp_path):
    """An already-duplicated span in an edited file that now also copies an untouched file."""
    a, b, c = (("a.py", 1, 10), ("b.py", 1, 10), ("c.py", 1, 10))
    assert _grew([_clone(a, c)], [_clone(a, c), _clone(a, b)], {"a.py"}, tmp_path) == ["a.py"]


def test_each_side_counts_its_own_span():
    """A gap clone's sides differ in length."""
    found = [_clone(("a.py", 1, 10), ("b.py", 1, 13))]
    assert ratchet.duplicated_lines(found) == {"a.py": 10, "b.py": 13}


@pytest.mark.parametrize(
    ("base_text", "head_text", "judged"),
    [("x = 1\n", "x = 1\n", False), ("x = 1\n", "x = 2\n", True), (None, "x = 1\n", True)],
    ids=["untouched", "modified", "added"],
)
def test_only_files_the_pr_wrote_are_judged(tmp_path, base_text, head_text, judged):
    """An untouched file's count can rise when a deletion makes jscpd re-pair its clones."""
    base, head = tmp_path / "base", tmp_path / "head"
    for tree, text in ((base, base_text), (head, head_text)):
        tree.mkdir()
        if text is not None:
            (tree / "f.py").write_text(text)
    expected = ["f.py"] if judged else []
    assert ratchet.grew(Counter({"f.py": 5}), Counter({"f.py": 9}), base, head) == expected
