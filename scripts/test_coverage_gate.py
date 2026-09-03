#!/usr/bin/env python
r"""Compare two coverage runs and fail if any source line lost its last covering test.

The gate behind ``context_docs/test-suite-streamlining.md``. Restructuring a test
suite is safe exactly insofar as it changes no source line's coverage; the percentage cannot
show that, because a file can gain covered lines and lose others while the ratio holds.
This compares the SETS.

It is a floor, not a proof. A line that still runs is not a line anything still asserts on —
see §4 of the plan for a worked case where coverage was identical and the tests were not
interchangeable. Pair this with a mutation spot-check wherever tests are deleted or merged.

Usage::

    uv run pytest tests/unit -n auto -q \\
        --cov=src/tessera_embeddings --cov-report=json:before.json
    # ... change the tests ...
    uv run pytest tests/unit -n auto -q \\
        --cov=src/tessera_embeddings --cov-report=json:after.json
    uv run python scripts/test_coverage_gate.py before.json after.json

Exit status is 1 if any line was lost, 0 otherwise, so it drops into a CI step unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

#: How many lost lines to print in full before summarising the rest by file. A regression
#: usually clusters in one or two files, and the first few are enough to name the cause.
_MAX_LISTED = 40


def executed_lines(report: Path) -> set[tuple[str, int]]:
    """The (file, line) pairs a coverage JSON report records as executed."""
    try:
        files = json.loads(report.read_text())["files"]
    except FileNotFoundError:
        sys.exit(f"no such coverage report: {report}")
    except (KeyError, json.JSONDecodeError) as exc:
        sys.exit(f"{report} is not a coverage JSON report ({exc}); pass --cov-report=json:PATH")
    return {(name, line) for name, data in files.items() for line in data["executed_lines"]}


def main() -> int:
    """Diff the two reports named on the command line; exit 1 if any line lost coverage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path, help="coverage JSON from before the change")
    parser.add_argument("after", type=Path, help="coverage JSON from after the change")
    args = parser.parse_args()

    before, after = executed_lines(args.before), executed_lines(args.after)
    if not before:
        sys.exit(f"{args.before} records no executed lines at all; the baseline run did not work")

    lost = sorted(before - after)
    gained = after - before

    if not lost:
        print(f"OK — {len(before)} lines covered before, {len(after)} after, none lost.")
        if gained:
            print(f"     {len(gained)} newly covered.")
        return 0

    by_file: dict[str, list[int]] = defaultdict(list)
    for name, line in lost:
        by_file[name].append(line)

    print(f"BLOCKED — {len(lost)} source lines lost their last covering test, across {len(by_file)} file(s).\n")
    for name, lines in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
        shown = ", ".join(str(line) for line in lines[:_MAX_LISTED])
        more = f" (+{len(lines) - _MAX_LISTED} more)" if len(lines) > _MAX_LISTED else ""
        print(f"  {name}\n    lines {shown}{more}")
    print("\nEither restore coverage of these lines, or justify each one in the PR description.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
