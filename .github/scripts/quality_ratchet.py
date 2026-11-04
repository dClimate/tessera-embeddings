"""Fail a pull request that adds duplication to a file or complexity to a function.

Usage: ``python quality_ratchet.py BASE_TREE HEAD_TREE`` from the head checkout, with jscpd and
ruff on PATH (``uv run``). Both trees are measured with the HEAD's settings, so a base that
predates them is measured the same way.

Each measure is compared with the base rather than with a fixed limit, so existing debt needs
no calibration or suppression, and a change that reduces it always passes:

- **Duplication, per file.** A file's duplicated lines (jscpd, ``.jscpd.json``) may not rise.
  Per file rather than per clone because jscpd identifies a clone by its exact extent, so a
  refactor that shrinks an existing clone would otherwise report it as new.
- **Per function** (ruff's ``C901``, ``PLR0913``, ``PLR0911``; limits in ``LIMITS``). No function
  may newly cross the complexity limit, and one already over it may not grow. Arguments and
  return statements bind NEW functions only (``NEW_ONLY``): a function whose qualified name is
  absent from the base, so a renamed or moved one counts as new. Prefect ``@flow`` functions are
  exempt from the argument limit.

It catches habits, not deliberate evasion, and adopts each tool's semantics as they are (see the
ADR's Scope). Rationale: context_docs/decisions/040-new-code-is-gated-on-duplication-and-complexity.md.
"""

from __future__ import annotations

import ast
import functools
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
    """Duplicated lines per file, counting each side of each clone over its own span.

    jscpd's ``lines`` is the FIRST side's span; a gap or similar-function clone's sides can
    differ in length.
    """
    lines: Counter[str] = Counter()
    for clone in found:
        for side in (clone["firstFile"], clone["secondFile"]):
            lines[side["name"]] += side["end"] - side["start"] + 1
    return lines


#: ``{metric: (ruff rule, setting, limit)}``. A function over a limit may not get worse.
LIMITS = {
    "complexity": ("C901", "lint.mccabe.max-complexity", 15),
    "arguments": ("PLR0913", "lint.pylint.max-args", 5),
    "return statements": ("PLR0911", "lint.pylint.max-returns", 3),
}

#: Measures that bind NEW functions only. An existing function is left alone on these, whatever
#: its score, until a deliberate clean-up; complexity binds existing functions too.
NEW_ONLY = {"arguments", "return statements"}


@functools.cache
def _functions(path: Path) -> dict[int, tuple[str, bool]]:
    """``{def line: (qualified name, is a Prefect @flow)}`` for every function in ``path``.

    Qualified (``ActorPool.__init__``, ``ingest_zone_year._run_leg``) so two same-named functions
    in one file are compared separately. A flow's parameters are its interface, the one a
    deployment and the Prefect UI expose, so flows are exempt from the argument limit.
    """
    found: dict[int, tuple[str, bool]] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    decorators = [d.func if isinstance(d, ast.Call) else d for d in child.decorator_list]
                    is_flow = any(
                        (isinstance(d, ast.Name) and d.id == "flow")
                        or (isinstance(d, ast.Attribute) and d.attr == "flow")
                        for d in decorators
                    )
                    found[child.lineno] = (name, is_flow)
                visit(child, f"{name}.")
            else:
                visit(child, prefix)

    visit(ast.parse(path.read_text()), "")
    return found


def all_functions(tree: Path) -> set[tuple[str, str]]:
    """``{(file, function)}`` for every function in ``tree/src`` and ``tests``."""
    return {
        (path.relative_to(tree).as_posix(), name)
        for root in ("src", "tests")
        for path in (tree / root).rglob("*.py")
        for name, _ in _functions(path).values()
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
        path = Path(finding["filename"])
        # Ruff reports on the function's def line, which is where ast places it.
        name, is_flow = _functions(path)[finding["location"]["row"]]
        if rule == "PLR0913" and is_flow:
            continue
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
    existing = all_functions(base)
    for metric, (rule, setting, limit) in LIMITS.items():
        old, new = function_scores(base, rule, setting, limit), function_scores(head, rule, setting, limit)
        for (path, fn), score in sorted(new.items()):
            if metric in NEW_ONLY and (path, fn) in existing:
                continue
            was = old.get((path, fn))
            if was is None:
                failures.append(f"{path}: {fn} has {score} {metric}, over {limit}")
            elif score > was:
                failures.append(f"{path}: {fn} grew from {was} to {score} {metric}")
    print("\n".join(failures) or "No file gained duplication and no function got worse.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
