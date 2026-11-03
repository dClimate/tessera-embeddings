"""Fail a pull request that adds duplication to a file or complexity to a function.

Usage: ``python quality_ratchet.py BASE_TREE HEAD_TREE`` from the head checkout, with jscpd and
ruff on PATH (``uv run``). Both trees are measured with the HEAD's settings, so a base that
predates them is measured the same way.

Each measure is compared with the base rather than with a fixed limit, so existing debt needs
no calibration or suppression, and a change that reduces it always passes:

- **Duplication, per file.** A file's duplicated lines (jscpd, ``.jscpd.json``) may not rise.
  Per file rather than per clone because jscpd identifies a clone by its exact extent, so a
  refactor that shrinks an existing clone would otherwise report it as new.
- **Complexity, per function.** A function's McCabe complexity may not rise once it is over
  ``MAX_COMPLEXITY``, and a function may not newly cross it.

Rationale: context_docs/decisions/040-new-code-is-gated-on-duplication-and-complexity.md.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

MAX_COMPLEXITY = 15


def clones(tree: Path, config: Path) -> list[dict]:
    """Every clone jscpd reports in ``tree`` under ``config``.

    jscpd resolves a config's paths and ignore globs against the config file's directory, so
    the config is copied INTO ``tree`` (the base is a throwaway worktree) rather than pointed at.
    """
    if config.parent != tree:
        shutil.copy(config, tree / ".jscpd.json")
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(["jscpd", "--reporters", "silent,json", "--output", out], cwd=tree, check=True)
        return json.loads((Path(out) / "jscpd-report.json").read_text())["duplicates"]


def duplicated_lines(found: list[dict]) -> Counter[str]:
    """Duplicated lines per file, counting each side of each clone."""
    lines: Counter[str] = Counter()
    for clone in found:
        for side in ("firstFile", "secondFile"):
            lines[clone[side]["name"]] += clone["lines"]
    return lines


def complexity(tree: Path) -> dict[tuple[str, str], int]:
    """``{(file, function): score}`` for every function in ``tree/src`` and ``tests`` over the limit."""
    result = subprocess.run(
        ["ruff", "check", "--isolated", "--ignore-noqa", "--select", "C901", "--output-format", "json"]
        + ["--config", f"lint.mccabe.max-complexity = {MAX_COMPLEXITY}", "src", "tests"],
        cwd=tree,
        capture_output=True,
        text=True,
    )
    scores: dict[tuple[str, str], int] = {}
    for finding in json.loads(result.stdout):
        name, score = re.match(r"`(.+)` is too complex \((\d+) >", finding["message"]).groups()
        key = (Path(finding["filename"]).relative_to(tree.resolve()).as_posix(), name)
        scores[key] = max(scores.get(key, 0), int(score))
    return scores


def main(base: Path, head: Path) -> int:
    """Print every regression and return the exit code."""
    failures: list[str] = []
    head_clones = clones(head, head / ".jscpd.json")
    before, after = duplicated_lines(clones(base, head / ".jscpd.json")), duplicated_lines(head_clones)
    for name in sorted(n for n in after if after[n] > before[n]):
        failures.append(f"{name}: duplicated lines rose from {before[name]} to {after[name]}")
        for c in head_clones:
            if name in (c["firstFile"]["name"], c["secondFile"]["name"]):
                a, b = c["firstFile"], c["secondFile"]
                failures.append(f"    {c['lines']} lines: {a['name']}:{a['start']} ~ {b['name']}:{b['start']}")
    old, new = complexity(base), complexity(head)
    for (path, fn), score in sorted(new.items()):
        was = old.get((path, fn))
        if was is None:
            failures.append(f"{path}: {fn} has complexity {score}, over {MAX_COMPLEXITY}")
        elif score > was:
            failures.append(f"{path}: {fn} grew from complexity {was} to {score}")
    print("\n".join(failures) or "No file gained duplication and no function gained complexity.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
