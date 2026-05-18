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
