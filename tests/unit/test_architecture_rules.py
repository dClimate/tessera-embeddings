"""Unit tests for the architecture-tests module.

The module is the engine the OSS contract depends on, so it gets its
own tight test suite. These tests construct synthetic source trees on
disk and assert the rule runner reports the right violations.
"""

from __future__ import annotations

from pathlib import Path

from tessera_embeddings.architecture_tests import (
    DEFAULT_RULES,
    Rule,
    load_allowlist,
    run,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_no_violations_on_clean_tree(tmp_path: Path) -> None:
    """A tree with no forbidden imports / calls reports no violations."""
    _write(tmp_path / "domain.py", "from collections import Counter\n")
    assert run(tmp_path) == []


def test_forbidden_import_outside_allowed_path_flags(tmp_path: Path) -> None:
    """``import prefect`` outside the allowed prefix is flagged."""
    _write(tmp_path / "domain" / "leak.py", "import prefect\n")
    violations = run(tmp_path)
    assert len(violations) == 1
    v = violations[0]
    assert v.rule == "no-prefect-outside-prefect-layer"
    assert v.path.name == "leak.py"
    assert "import prefect" in v.detail


def test_allowed_path_suppresses_violation(tmp_path: Path) -> None:
    """A forbidden import under the allowed prefix is fine."""
    _write(tmp_path / "orchestration" / "prefect" / "tasks.py", "import prefect\n")
    assert run(tmp_path) == []


def test_from_import_subpackage_flags(tmp_path: Path) -> None:
    """``from prefect.foo import bar`` matches the prefix rule too."""
    _write(tmp_path / "domain.py", "from prefect.deployments import arun_deployment\n")
    violations = run(tmp_path)
    assert len(violations) == 1
    assert violations[0].detail == "from prefect.deployments"


def test_forbidden_call_name_flags(tmp_path: Path) -> None:
    """A bare call to ``get_run_logger()`` is flagged."""
    _write(
        tmp_path / "domain.py",
        "def f():\n    log = get_run_logger()\n    return log\n",
    )
    violations = run(tmp_path)
    rules_fired = {v.rule for v in violations}
    assert "no-prefect-context-helpers-outside-prefect-layer" in rules_fired


def test_library_code_may_not_import_the_profiling_subpackage(tmp_path: Path) -> None:
    """Only profiling/ may import profiling/.

    The boto3 exemption for profiling/ is conditional on the subpackage being
    unreachable from library code — one import from a flow or domain module would
    pull boto3 into a plain ``import tessera_embeddings``. Both directions are
    pinned: flagged from outside, allowed within (the tools import each other).
    """
    _write(tmp_path / "inference" / "actors.py", "from tessera_embeddings.profiling.ingest import report\n")
    violations = [v for v in run(tmp_path) if v.rule == "no-profiling-imports-outside-profiling"]
    assert len(violations) == 1
    assert violations[0].path.name == "actors.py"

    _write(tmp_path / "profiling" / "ingest" / "report.py", "from tessera_embeddings.profiling import _cloudwatch\n")
    assert not [
        v for v in run(tmp_path) if v.rule == "no-profiling-imports-outside-profiling" and v.path.name == "report.py"
    ]


def test_relative_imports_are_resolved_before_matching(tmp_path: Path) -> None:
    """A package-relative import of profiling/ is flagged like an absolute one.

    ``ast.ImportFrom.module`` holds only the text after the dots, so
    ``from ..profiling.ingest import report`` reports ``profiling.ingest`` and matches
    no absolute prefix. Unresolved, the rule passes on an import doing exactly what it
    forbids — pulling module-scope boto3 into a base install. The scan resolves the
    level against the file's own package first; the detail shows both forms.
    """
    pkg = tmp_path / "tessera_embeddings"  # the rule's prefix is package-qualified
    _write(pkg / "inference" / "actors.py", "from ..profiling.ingest import report\n")
    violations = [v for v in run(pkg) if v.rule == "no-profiling-imports-outside-profiling"]
    assert len(violations) == 1
    assert violations[0].path.name == "actors.py"
    assert "tessera_embeddings.profiling.ingest" in violations[0].detail

    # Still allowed within profiling/ — the tools import each other relatively too.
    _write(pkg / "profiling" / "ingest" / "report.py", "from .. import _cloudwatch\n")
    assert not [
        v for v in run(pkg) if v.rule == "no-profiling-imports-outside-profiling" and v.path.name == "report.py"
    ]


def test_relative_import_climbing_out_of_the_tree_is_not_attributed(tmp_path: Path) -> None:
    """A relative import above the scanned root names no package we can judge."""
    pkg = tmp_path / "tessera_embeddings"
    _write(pkg / "domain.py", "from ...elsewhere import thing\n")
    assert [v for v in run(pkg) if v.rule != "parse-error"] == []


def test_extra_allowed_paths_extend_per_rule(tmp_path: Path) -> None:
    """``extra_allowed_paths`` permits the import in additional subtrees."""
    _write(tmp_path / "yield_modeling" / "iac.py", "import boto3\n")
    # Without the allowlist, it's a violation:
    assert any(v.rule == "no-boto3-outside-aws-provider" for v in run(tmp_path))
    # With the allowlist, it's fine:
    extra = {"no-boto3-outside-aws-provider": ("yield_modeling/",)}
    assert run(tmp_path, extra_allowed_paths=extra) == []


def test_load_allowlist_round_trip(tmp_path: Path) -> None:
    """TOML allowlist parses into the dict shape ``run`` accepts."""
    toml_path = tmp_path / "allowlist.toml"
    toml_path.write_text(
        '[allowed_imports."no-boto3-outside-aws-provider"]\n'
        'paths = ["yield_modeling/iac/", "yield_modeling/providers/aws/"]\n'
    )
    allow = load_allowlist(toml_path)
    assert allow == {
        "no-boto3-outside-aws-provider": (
            "yield_modeling/iac/",
            "yield_modeling/providers/aws/",
        )
    }


def test_default_rules_cover_six_hard_rules() -> None:
    """The bundled rules include the six-hard-rules contract.

    Hard rule list (from the implementation plan §0):
        1. No prefect imports outside orchestration/prefect/
        2. Stdlib logging in domain modules
        3. Pydantic for config (not enforced as a forbidden import; structural)
        4. fsspec for storage (structural)
        5. Secrets enter at flow boundary (structural)
        6. Clients passed in, never reached for in context (covers
           get_client / get_run_logger and boto3 reach-around)

    The structural rules (2, 3, 4, 5) are enforced by code review and
    tests rather than the AST-level checker. This test verifies that
    the *enforceable* rules are wired into DEFAULT_RULES.
    """
    rule_names = {r.name for r in DEFAULT_RULES}
    assert "no-prefect-outside-prefect-layer" in rule_names
    assert "no-prefect-context-helpers-outside-prefect-layer" in rule_names
    assert "no-dask-distributed-get_client-in-domain" in rule_names
    assert "no-boto3-outside-aws-provider" in rule_names
    assert "no-botocore-outside-aws-provider" in rule_names
    # Guards the condition the boto3 exemption for profiling/ was granted on.
    assert "no-profiling-imports-outside-profiling" in rule_names


def test_rule_dataclass_is_immutable() -> None:
    """``Rule`` is frozen — mutation raises."""
    r = DEFAULT_RULES[0]
    try:
        r.name = "tampered"  # type: ignore[misc]
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__
        return
    raise AssertionError("Rule should be immutable but mutation succeeded")


def test_custom_rule_runs(tmp_path: Path) -> None:
    """Callers can pass a custom rule list."""
    _write(tmp_path / "code.py", "import sneaky\n")
    rule = Rule(name="no-sneaky", forbidden_import_prefix="sneaky", allowed_path_prefixes=())
    violations = run(tmp_path, rules=[rule])
    assert len(violations) == 1
    assert violations[0].rule == "no-sneaky"


def test_importing_the_subpackage_by_name_is_flagged(tmp_path: Path) -> None:
    """`from tessera_embeddings import profiling` imports the forbidden subpackage.

    The source module resolves to `tessera_embeddings`, which matches no forbidden
    prefix — so matching the module alone let the exact import the rule exists to
    prevent through. The imported NAMES are checked too.
    """
    pkg = tmp_path / "tessera_embeddings"
    _write(pkg / "inference" / "actors.py", "from tessera_embeddings import profiling\n")
    _write(pkg / "inference" / "relative.py", "from .. import profiling\n")
    flagged = {v.path.name for v in run(pkg) if v.rule == "no-profiling-imports-outside-profiling"}
    assert flagged == {"actors.py", "relative.py"}

    # An unrelated name from the same package is still fine.
    _write(pkg / "inference" / "ok.py", "from tessera_embeddings import storage\n")
    assert "ok.py" not in {v.path.name for v in run(pkg) if v.rule == "no-profiling-imports-outside-profiling"}


def test_scanning_a_src_wrapper_descends_into_the_package(tmp_path: Path) -> None:
    """A `src/` wrapper is scanned by descending into it, not by refusing it.

    The scanned root's NAME is what relative imports resolve against, so scanning `src/`
    itself turns `from ..profiling import x` inside `src/pkg/inference/` into
    `src.pkg.profiling` — matching no absolute forbidden prefix, so every relative-import
    rule passes on the imports it exists to forbid. A checker that approves by accident is
    worse than no checker.

    This used to be handled by REFUSING a single-package wrapper, which made
    `--source src/` — the command the adapter template documents — unusable. Each package
    beneath the wrapper is now its own scan root, so the imports resolve and the rule fires.
    """
    src = tmp_path / "src"
    _write(src / "tessera_embeddings" / "__init__.py", "")
    _write(
        src / "tessera_embeddings" / "inference" / "actors.py",
        "from ..profiling.ingest import report\n",
    )

    flagged = {v.rule for v in run(src)}
    assert "no-profiling-imports-outside-profiling" in flagged, (
        "scanning the wrapper must find the violation, not pass it or refuse to look"
    )

    # And the same answer as pointing at the package directly, which is the whole point.
    assert flagged == {v.rule for v in run(src / "tessera_embeddings")}


def test_a_src_wrapper_holding_several_packages_is_scanned_whole(tmp_path: Path) -> None:
    """The silent half of the old behaviour: more than one package fell straight through.

    The refusal only triggered on a SINGLE nested package. With two, the old code scanned
    with the root named `src` and every relative-import rule passed on the imports it
    forbids — no error, no finding, exit 0.
    """
    src = tmp_path / "src"
    for pkg in ("tessera_embeddings", "second_pkg"):
        _write(src / pkg / "__init__.py", "")
    _write(
        src / "tessera_embeddings" / "inference" / "actors.py",
        "from ..profiling.ingest import report\n",
    )

    assert "no-profiling-imports-outside-profiling" in {v.rule for v in run(src)}


def test_a_namespace_package_under_a_wrapper_is_still_found(tmp_path: Path) -> None:
    """PEP 420 packages have no `__init__.py`, and requiring one reopened the silent pass.

    The first version of the wrapper handling looked for `__init__.py` to decide what was a
    package. A namespace package has none, so nothing was found, the wrapper was scanned as
    itself, and every relative-import rule passed again — the same hole, reached by a
    different route.
    """
    src = tmp_path / "src"
    # No __init__.py anywhere: this is the namespace layout.
    _write(
        src / "tessera_embeddings" / "inference" / "actors.py",
        "from ..profiling.ingest import report\n",
    )

    assert "no-profiling-imports-outside-profiling" in {v.rule for v in run(src)}


def test_a_root_holding_its_own_modules_is_scanned_as_itself(tmp_path: Path) -> None:
    """A plain tree is not a wrapper, and descending into it would change what resolves."""
    root = tmp_path / "tessera_embeddings"
    _write(root / "__init__.py", "")
    _write(root / "inference" / "actors.py", "from ..profiling.ingest import report\n")

    assert "no-profiling-imports-outside-profiling" in {v.rule for v in run(root)}
