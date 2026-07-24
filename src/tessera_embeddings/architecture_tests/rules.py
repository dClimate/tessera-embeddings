"""Architecture rule definitions and AST-based runner.

Each :class:`Rule` describes a forbidden import (or call) plus a list
of subtree-relative path prefixes where the import is allowed. The
runner walks every ``.py`` file under a source root, parses to AST,
and reports violations.

AST-based instead of grep so docstrings and comments don't trigger
false positives.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    """One architecture rule.

    Attributes:
        name: Human-readable rule identifier (used in violation
            messages).
        forbidden_import_prefix: Module-name prefix that, if imported,
            counts as a violation. e.g. ``"prefect"`` flags
            ``import prefect``, ``import prefect.deployments``,
            ``from prefect import flow``, etc.
        forbidden_call_names: Function or method names whose call
            sites flag as violations. e.g. ``{"get_run_logger"}``.
        allowed_path_prefixes: Subpaths (relative to the source root)
            where the rule does NOT apply. Match against
            POSIX-formatted relative paths.
    """

    name: str
    forbidden_import_prefix: str | None = None
    forbidden_call_names: frozenset[str] = field(default_factory=frozenset)
    allowed_path_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Violation:
    """A single rule violation."""

    rule: str
    path: Path
    lineno: int
    detail: str

    def __str__(self) -> str:
        """One-line human-readable rendering."""
        return f"{self.path}:{self.lineno}: {self.rule}: {self.detail}"


# The OSS contract's bundled rules. Downstream consumers can extend the
# allowed-path tuples via a TOML allowlist; see :func:`run`.
DEFAULT_RULES: tuple[Rule, ...] = (
    Rule(
        name="no-prefect-outside-prefect-layer",
        forbidden_import_prefix="prefect",
        allowed_path_prefixes=("orchestration/prefect/",),
    ),
    Rule(
        name="no-prefect-context-helpers-outside-prefect-layer",
        forbidden_call_names=frozenset({"get_run_logger"}),
        allowed_path_prefixes=("orchestration/prefect/",),
    ),
    Rule(
        name="no-dask-distributed-get_client-in-domain",
        forbidden_call_names=frozenset({"get_client"}),
        # The Prefect task shells legitimately call get_client to fetch
        # the Dask client from context; the auth module needs it to
        # register a worker plugin. Domain code never reaches.
        allowed_path_prefixes=(
            "orchestration/prefect/",
            "ingest/auth.py",
        ),
    ),
    Rule(
        name="no-boto3-outside-aws-provider",
        forbidden_import_prefix="boto3",
        # boto3 is the AWS SDK; it must stay confined to providers/aws/
        # so closed-source clouds and on-prem deployments don't transitively
        # pull AWS in.
        #
        # profiling/ is the one other exception: operator CLIs that read
        # CloudWatch/ECS/EC2 to profile a live run. The invariant this rule
        # protects still holds — no library code imports profiling/, so the
        # subpackage is only loaded when someone runs a profiling command, and
        # boto3 stays in the optional `aws` extra. Confining these tools to
        # providers/aws/ instead would split each harness across two trees for
        # no gain in isolation.
        allowed_path_prefixes=("providers/aws/", "profiling/"),
    ),
    Rule(
        name="no-botocore-outside-aws-provider",
        forbidden_import_prefix="botocore",
        allowed_path_prefixes=("providers/aws/",),
    ),
)


def _path_to_subtree(root: Path, path: Path) -> str:
    """Return ``path`` as a POSIX-formatted relative path under ``root``."""
    return path.relative_to(root).as_posix()


def _is_allowed(rel: str, allowed_prefixes: Iterable[str]) -> bool:
    """True if ``rel`` starts with any allowed prefix."""
    return any(rel.startswith(prefix) for prefix in allowed_prefixes)


def _scan_file(path: Path, rel: str, rules: Iterable[Rule]) -> list[Violation]:
    """Scan one file against every rule and return any violations."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:
        return [Violation(rule="parse-error", path=path, lineno=exc.lineno or 0, detail=str(exc))]

    violations: list[Violation] = []
    for node in ast.walk(tree):
        for rule in rules:
            if _is_allowed(rel, rule.allowed_path_prefixes):
                continue
            v = _check_node(node, rule, path)
            if v is not None:
                violations.append(v)
    return violations


def _check_node(node: ast.AST, rule: Rule, path: Path) -> Violation | None:
    """Return a Violation if ``node`` violates ``rule``, else ``None``."""
    if rule.forbidden_import_prefix is not None:
        prefix = rule.forbidden_import_prefix
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == prefix or alias.name.startswith(f"{prefix}."):
                    return Violation(
                        rule=rule.name,
                        path=path,
                        lineno=node.lineno,
                        detail=f"import {alias.name}",
                    )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == prefix or node.module.startswith(f"{prefix}."))
        ):
            return Violation(
                rule=rule.name,
                path=path,
                lineno=node.lineno,
                detail=f"from {node.module}",
            )

    if rule.forbidden_call_names and isinstance(node, ast.Call):
        fn = node.func
        name: str | None = None
        if isinstance(fn, ast.Name):
            name = fn.id
        elif isinstance(fn, ast.Attribute):
            name = fn.attr
        if name and name in rule.forbidden_call_names:
            return Violation(
                rule=rule.name,
                path=path,
                lineno=node.lineno,
                detail=f"call to {name}()",
            )
    return None


def run(
    source_path: Path,
    *,
    rules: Iterable[Rule] = DEFAULT_RULES,
    extra_allowed_paths: dict[str, tuple[str, ...]] | None = None,
) -> list[Violation]:
    """Run every rule against every ``.py`` file under ``source_path``.

    Args:
        source_path: Root directory whose ``.py`` files are scanned.
            Path matching for allowed prefixes is relative to this
            root and uses POSIX separators.
        rules: Rules to enforce. Defaults to :data:`DEFAULT_RULES`.
        extra_allowed_paths: Per-rule additional allowed prefixes,
            keyed by :attr:`Rule.name`. Loaded from a TOML allowlist
            via :func:`load_allowlist`.

    Returns:
        A list of :class:`Violation` instances, sorted by path then
        line number.
    """
    extra = extra_allowed_paths or {}
    expanded_rules = tuple(
        Rule(
            name=r.name,
            forbidden_import_prefix=r.forbidden_import_prefix,
            forbidden_call_names=r.forbidden_call_names,
            allowed_path_prefixes=tuple(r.allowed_path_prefixes) + tuple(extra.get(r.name, ())),
        )
        for r in rules
    )

    source_path = source_path.resolve()
    violations: list[Violation] = []
    for path in sorted(source_path.rglob("*.py")):
        rel = _path_to_subtree(source_path, path)
        violations.extend(_scan_file(path, rel, expanded_rules))

    violations.sort(key=lambda v: (str(v.path), v.lineno))
    return violations
