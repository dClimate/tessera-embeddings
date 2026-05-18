"""Architecture-rule enforcement.

Runs the bundled rules from
:mod:`tessera_embeddings.architecture_tests` against the OSS source
tree on every PR. The rules themselves are tested separately under
``tests/unit/test_architecture_rules.py``.

Failing this test means the OSS architecture contract was violated.
Look at the violation messages to find the offending file + line.
"""

from __future__ import annotations

from pathlib import Path

from tessera_embeddings.architecture_tests import run

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "tessera_embeddings"


def test_oss_source_tree_has_no_architecture_violations() -> None:
    """No bundled rule fires on ``src/tessera_embeddings/``."""
    violations = run(SRC_ROOT)
    assert not violations, "Architecture violations:\n" + "\n".join(str(v) for v in violations)
