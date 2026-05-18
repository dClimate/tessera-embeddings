"""Reusable architecture-rule checker.

The ``tessera_embeddings`` OSS contract is enforced by a small set of
"hard rules" — e.g. *no* ``import prefect`` outside
``orchestration/prefect/``. This module makes those rules executable
against any source tree.

Use cases:

* The OSS repo's CI runs the bundled rules against
  ``src/tessera_embeddings/`` (``tests/architecture/test_layer_rules.py``).
* Downstream consumers (closed-source repos, community-adapter PRs)
  point the same checker at *their* source tree, optionally with a
  TOML allowlist that adds extra paths where a rule is permitted.

Public API:

* :class:`Rule` — one architecture rule.
* :class:`Violation` — a single rule violation with file, line, and detail.
* :func:`run` — execute every rule against a source path.
* :data:`DEFAULT_RULES` — the OSS contract's bundled rules.

CLI: ``python -m tessera_embeddings.architecture_tests --source path/ [--allowlist file.toml]``.
"""

from tessera_embeddings.architecture_tests.allowlist import load_allowlist
from tessera_embeddings.architecture_tests.rules import (
    DEFAULT_RULES,
    Rule,
    Violation,
    run,
)

__all__ = [
    "DEFAULT_RULES",
    "Rule",
    "Violation",
    "load_allowlist",
    "run",
]
