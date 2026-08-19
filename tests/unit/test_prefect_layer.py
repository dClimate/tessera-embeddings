"""Architecture + smoke tests for the Prefect orchestration layer.

These tests enforce the architectural invariant that ``import prefect``
appears **only** under ``orchestration/prefect/``. Domain modules
(``ingest/``, ``inference/``, ``storage/``, ``providers/``,
``orchestration/concurrency.py``) must stay orchestrator-agnostic so
they remain testable without a Prefect runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path("src/tessera_embeddings")
PREFECT_ALLOWED_PREFIX = SRC_ROOT / "orchestration" / "prefect"


def test_flow_modules_import() -> None:
    """Every Prefect flow module imports cleanly."""
    from tessera_embeddings.orchestration.prefect.flows import (  # noqa: F401
        build_land_mask,
        fill_zone_year,
        generate_roi,
        ingest_s1_roi_sar,
        ingest_s2_roi_reflectance,
        run_global_campaign,
        seed_global_store,
        tessera_embeddings,
        tessera_full_pipeline,
    )


def test_task_modules_import() -> None:
    """Every Prefect task shell module imports cleanly."""
    from tessera_embeddings.orchestration.prefect.tasks import (  # noqa: F401
        inference,
        ingest,
        land_mask,
    )


def test_prefect_imports_confined_to_orchestration_subtree() -> None:
    """``import prefect`` may only appear under ``orchestration/prefect/``.

    Walks every Python file under ``src/tessera_embeddings/`` and uses
    AST inspection (so a docstring mentioning ``prefect`` does not trip
    the check). Any import of a name starting with ``prefect`` outside
    the allowed subtree is a violation of the architectural rule.
    """
    violations: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        # Skip files inside the allowed subtree
        if PREFECT_ALLOWED_PREFIX in path.parents or path == PREFECT_ALLOWED_PREFIX:
            continue

        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("prefect"):
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("prefect"):
                violations.append(f"{path}: from {node.module}")

    assert not violations, "Prefect imports leaked outside orchestration/prefect/:\n" + "\n".join(violations)


def test_inference_orchestration_helpers_no_prefect() -> None:
    """``inference/orchestration_helpers.py`` is domain-pure (no Prefect)."""
    path = SRC_ROOT / "inference" / "orchestration_helpers.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("prefect"), f"{path} imports {alias.name!r}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("prefect"), f"{path} imports from {node.module!r}"
