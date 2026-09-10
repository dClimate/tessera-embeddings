"""CLI entry point for ``tessera_embeddings.architecture_tests``.

Invoke via ``python -m tessera_embeddings.architecture_tests --source path/to/src
[--allowlist file.toml]``. Exits ``0`` with no violations, or ``1`` after writing each
violation to stderr on its own line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tessera_embeddings.architecture_tests.allowlist import load_allowlist
from tessera_embeddings.architecture_tests.rules import run


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="tessera_embeddings architecture-rule checker")
    parser.add_argument("--source", type=Path, required=True, help="Root directory to scan")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Optional TOML allowlist that extends per-rule allowed paths",
    )
    args = parser.parse_args(argv)

    extra: dict[str, tuple[str, ...]] | None = None
    if args.allowlist is not None:
        extra = load_allowlist(args.allowlist)

    violations = run(args.source, extra_allowed_paths=extra)
    if violations:
        for v in violations:
            print(str(v), file=sys.stderr)
        print(f"\n{len(violations)} architecture violation(s) found", file=sys.stderr)
        return 1
    print(f"No architecture violations found under {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
