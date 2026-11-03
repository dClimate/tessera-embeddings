"""Fail a pull request that adds duplication to a file or complexity to a function.

Usage: ``python quality_ratchet.py BASE_TREE HEAD_TREE`` from the head checkout, with jscpd and
ruff on PATH (``uv run``). Both trees are measured with the HEAD's settings, so a base that
predates them is measured the same way.

Each measure is compared with the base rather than with a fixed limit, so existing debt needs
no calibration or suppression, and a change that reduces it always passes:

- **Duplication, per file.** A file's duplicated lines (jscpd, ``.jscpd.json``) may not rise.
  Per file rather than per clone because jscpd identifies a clone by its exact extent, so a
  refactor that shrinks an existing clone would otherwise report it as new.
- **Per function: complexity, arguments and return statements** (ruff's ``C901``, ``PLR0913``,
  ``PLR0911``). A function over a limit in ``LIMITS`` may not get worse on that measure, and no
  function may newly cross one. Prefect ``@flow`` functions are exempt from the argument limit.

Rationale: context_docs/decisions/040-new-code-is-gated-on-duplication-and-complexity.md.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


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


#: ``{metric: (ruff rule, setting, limit)}``. A function over a limit may not get worse.
LIMITS = {
    "complexity": ("C901", "lint.mccabe.max-complexity", 15),
    "arguments": ("PLR0913", "lint.pylint.max-args", 8),
    "return statements": ("PLR0911", "lint.pylint.max-returns", 6),
}


def _flow_lines(path: Path) -> set[int]:
    """The ``def`` lines of the Prefect ``@flow`` functions in ``path``.

    A flow's parameters are its interface, the one a deployment and the Prefect UI expose, so
    flows are exempt from the argument limit. Nothing else about them is.
    """
    return {
        node.lineno
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            (d.func if isinstance(d, ast.Call) else d).id == "flow"
            for d in node.decorator_list
            if isinstance(d.func if isinstance(d, ast.Call) else d, ast.Name)
        )
    }


def function_scores(tree: Path, rule: str, setting: str, limit: int) -> dict[tuple[str, str], int]:
    """``{(file, function): score}`` for every function in ``tree/src`` and ``tests`` over ``limit``."""
    result = subprocess.run(
        ["ruff", "check", "--isolated", "--ignore-noqa", "--select", rule, "--output-format", "json"]
        + ["--config", f"{setting} = {limit}", "src", "tests"],
        cwd=tree,
        capture_output=True,
        text=True,
    )
    scores: dict[tuple[str, str], int] = {}
    for finding in json.loads(result.stdout):
        path, row, col = Path(finding["filename"]), finding["location"]["row"], finding["location"]["column"]
        if rule == "PLR0913" and row in _flow_lines(path):
            continue
        # Ruff reports at the function's name, which only C901's message repeats.
        name = re.match(r"\w+", path.read_text().splitlines()[row - 1][col - 1 :]).group()
        score = int(re.search(r"\((\d+) >", finding["message"]).group(1))
        key = (path.relative_to(tree.resolve()).as_posix(), name)
        scores[key] = max(scores.get(key, 0), score)
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
    for metric, (rule, setting, limit) in LIMITS.items():
        old, new = function_scores(base, rule, setting, limit), function_scores(head, rule, setting, limit)
        for (path, fn), score in sorted(new.items()):
            was = old.get((path, fn))
            if was is None:
                failures.append(f"{path}: {fn} has {score} {metric}, over {limit}")
            elif score > was:
                failures.append(f"{path}: {fn} grew from {was} to {score} {metric}")
    print("\n".join(failures) or "No file gained duplication and no function got worse.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
