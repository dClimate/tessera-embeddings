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


def test_deleting_the_shared_copy_does_not_raise_a_survivor():
    """The remaining copies are re-paired against one file, whose lines must count once."""
    x, a, c, d = (("x.py", 1, 10), ("a.py", 1, 10), ("c.py", 1, 10), ("d.py", 1, 10))
    before = ratchet.duplicated_lines([_clone(x, a), _clone(x, c), _clone(x, d)])
    after = ratchet.duplicated_lines([_clone(a, c), _clone(a, d)])
    assert after["a.py"] <= before["a.py"]


def test_each_side_counts_its_own_span():
    """A gap clone's sides differ in length; overlapping spans in one file are a union."""
    found = [_clone(("a.py", 1, 10), ("b.py", 1, 13)), _clone(("a.py", 5, 14), ("c.py", 1, 10))]
    assert ratchet.duplicated_lines(found) == {"a.py": 14, "b.py": 13, "c.py": 10}


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
