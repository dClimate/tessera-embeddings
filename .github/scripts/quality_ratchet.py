"""Fail a pull request that writes a copy of existing code or adds complexity to a function.

Usage: ``python quality_ratchet.py BASE_TREE HEAD_TREE`` from the head checkout, with jscpd and
ruff on PATH (``uv run``). BASE_TREE is a worktree of the commit the PR merges into, so ``git``
can diff the two.

- **Duplication: code the PR writes may not be a copy.** jscpd (``.jscpd.json``) runs once, on
  the head, and ``git diff`` gives the lines the PR added or changed. A clone side holding at
  least ``MIN_WRITTEN`` of those lines, or at least half of a side that long, is a copy the PR
  made. Only written lines are judged, so how jscpd pairs copies the PR did not touch (it
  re-pairs them whenever one is deleted) cannot fail a PR, and deleting a copy never does.
- **Per function** (ruff's ``C901``, ``PLR0913``, ``PLR0911``; limits in ``LIMITS``), compared
  with the base. No function may newly cross the complexity limit, and one already over it may
  not grow. Arguments and return statements bind NEW functions only (``NEW_ONLY``): a function
  whose qualified name is absent from the base, so a renamed or moved one counts as new. Prefect
  ``@flow`` functions are exempt from the argument limit.

It catches habits, not deliberate evasion, and adopts each tool's semantics as they are (see the
ADR's Scope). Rationale: context_docs/decisions/040-new-code-is-gated-on-duplication-and-complexity.md.
"""

from __future__ import annotations

import ast
import functools
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

#: A clone side holding this many lines the PR wrote, or at least half of a side this long, is a
#: copy it made. jscpd's own minimum clone length. The half catches a short copy that ``git diff``
#: aligns with lines it replaced (a shared ``return`` or blank line reads as unchanged); a side
#: shorter than this is the short end of a gap or similar-function clone and is not judged alone.
MIN_WRITTEN = 5


def clones(tree: Path) -> list[dict]:
    """Every clone jscpd reports in ``tree``, under the ``.jscpd.json`` there."""
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(["jscpd", "--reporters", "silent,json", "--output", out], cwd=tree, check=True)
        return json.loads((Path(out) / "jscpd-report.json").read_text())["duplicates"]


def written_lines(head: Path, base_commit: str) -> dict[str, set[int]]:
    """``{file: head line numbers}`` the PR added or changed since ``base_commit``.

    Renames are followed (``-M``), so a renamed file counts only the lines that changed in it.
    """
    diff = subprocess.run(
        ["git", "diff", "--unified=0", "-M", "--no-color", "--no-ext-diff", base_commit, "HEAD"],
        cwd=head,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    written: dict[str, set[int]] = {}
    name = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            name = None if line == "+++ /dev/null" else line.removeprefix("+++ b/")
        elif line.startswith("@@") and name:
            start, _, count = re.search(r"\+(\d+)(,(\d+))?", line).groups()
            first = int(start)
            written.setdefault(name, set()).update(range(first, first + int(count or 1)))
    return written


def copies(found: list[dict], written: dict[str, set[int]]) -> list[str]:
    """One line for each clone side the PR wrote, naming the code it copies."""
    made = []
    for clone in found:
        sides = (clone["firstFile"], clone["secondFile"])
        for side, other in (sides, sides[::-1]):
            span = side["end"] - side["start"] + 1
            lines = range(side["start"], side["end"] + 1)
            n = len(written.get(side["name"], set()).intersection(lines))
            if n >= MIN_WRITTEN or (span >= MIN_WRITTEN and 2 * n >= span):
                made.append(
                    f"{side['name']}:{side['start']}: wrote {n} of these {span} lines,"
                    f" a copy of {other['name']}:{other['start']}"
                )
    return sorted(set(made))


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
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=base, capture_output=True, text=True, check=True
    ).stdout.strip()
    failures = copies(clones(head), written_lines(head, base_commit))
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
    print("\n".join(failures) or "The PR wrote no copies, and no function got worse.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
