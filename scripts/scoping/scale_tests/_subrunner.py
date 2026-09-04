"""Subprocess entrypoint for cold-cache measurements (see ``harness.run_cold``).

Invoked as ``python -m scale_tests._subrunner <dotted.func.path>`` with a JSON
payload on stdin. Resolves the function, calls it with the decoded payload, and
prints ``RESULT:<json>`` on stdout. Kept tiny and dependency-light so the fresh
interpreter starts fast and shares no in-process cache with the caller.
"""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any


def _resolve(dotpath: str) -> Any:  # noqa: ANN401 — returns an arbitrary callable
    """Resolve ``pkg.module.func`` to the callable it names."""
    module_path, _, attr = dotpath.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def main() -> int:
    """Run the requested function against the stdin JSON payload."""
    if len(sys.argv) != 2:
        print("usage: python -m scale_tests._subrunner <dotted.func.path>", file=sys.stderr)
        return 2
    func = _resolve(sys.argv[1])
    payload = json.loads(sys.stdin.read() or "{}")
    result = func(payload)
    print("RESULT:" + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
