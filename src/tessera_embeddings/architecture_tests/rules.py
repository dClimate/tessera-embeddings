"""Architecture rule definitions and AST-based runner.

Each :class:`Rule` describes a forbidden import (or call) plus the subtree-relative path
prefixes where it is allowed; the runner walks every ``.py`` file under a source root,
parses to AST, and reports violations. AST rather than grep, so docstrings and comments
don't trigger false positives.
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
        name: Human-readable rule identifier, used in violation messages.
        forbidden_import_prefix: Module-name prefix whose import is a violation, e.g.
            ``"prefect"`` flags ``import prefect``, ``import prefect.deployments`` and
            ``from prefect import flow``.
        forbidden_call_names: Function or method names whose call sites are violations,
            e.g. ``{"get_run_logger"}``.
        allowed_path_prefixes: Subpaths (relative to the source root, POSIX-formatted)
            where the rule does NOT apply.
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
        name="solar-offset-applied-only-in-solar-days",
        forbidden_call_names=frozenset({"solar_day_offset_seconds"}),
        # The UTC-to-solar offset must be applied EXACTLY ONCE per item, by
        # `solar_days.normalize_to_solar_day`, at the catalogue chokepoint. After that an
        # item's timestamp IS its solar day and every date derivation is a plain strftime.
        #
        # Applying it per site was tried and drifted: of six sites, the cloud pre-sort and
        # the baseline map keyed on the UTC date while the loader grouped by solar day, so
        # on a day straddling UTC midnight the group was not sorted as intended and half its
        # baseline entries never matched — both invisible in the output. A second
        # application is the opposite bug: it shifts an already-shifted timestamp. Confining
        # the call to the module that owns the concept makes both impossible.
        allowed_path_prefixes=("ingest/solar_days.py",),
    ),
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
        # The Prefect task shells legitimately call get_client for the Dask client from
        # context, and the auth module needs it to register a worker plugin. Domain code
        # never does.
        allowed_path_prefixes=(
            "orchestration/prefect/",
            "ingest/auth.py",
        ),
    ),
    Rule(
        name="no-boto3-outside-aws-provider",
        forbidden_import_prefix="boto3",
        # boto3 is the AWS SDK; confined to providers/aws/ so other clouds and on-prem
        # deployments don't transitively pull AWS in.
        #
        # profiling/ is the one other exception: operator CLIs that read CloudWatch/ECS/EC2
        # to profile a live run. The invariant holds because no library code imports
        # profiling/ (enforced below), so boto3 loads only when someone runs a profiling
        # command and stays in the optional `aws` extra. Moving these tools under
        # providers/aws/ would split each harness across two trees for no gain in isolation.
        allowed_path_prefixes=("providers/aws/", "profiling/"),
    ),
    Rule(
        name="no-botocore-outside-aws-provider",
        forbidden_import_prefix="botocore",
        allowed_path_prefixes=("providers/aws/",),
    ),
    Rule(
        # Enforces the condition the boto3 exemption above rests on: profiling/ stays
        # unreachable from library code. The first domain or flow module to import it would
        # pull boto3 transitively into an ordinary `import tessera_embeddings` and silently
        # undo the isolation the exemption was granted against.
        name="no-profiling-imports-outside-profiling",
        forbidden_import_prefix="tessera_embeddings.profiling",
        allowed_path_prefixes=("profiling/",),
    ),
)


def _path_to_subtree(root: Path, path: Path) -> str:
    """Return ``path`` as a POSIX-formatted relative path under ``root``."""
    return path.relative_to(root).as_posix()


def _is_allowed(rel: str, allowed_prefixes: Iterable[str]) -> bool:
    """True if ``rel`` starts with any allowed prefix."""
    return any(rel.startswith(prefix) for prefix in allowed_prefixes)


def _resolved_module(node: ast.ImportFrom, rel: str, package: str) -> str | None:
    """The absolute dotted module an ``ImportFrom`` names, resolving relative levels.

    ``node.module`` alone is the text after the dots, so ``from ..profiling.ingest.watch
    import main`` reports ``profiling.ingest.watch``, matches no absolute forbidden prefix,
    and the rule silently passes on an import that does exactly what it forbids. Level 1 is
    the file's own package and each extra level walks one parent up; ``None`` for a relative
    import that climbs out of the scanned tree, which cannot be attributed to a package.
    """
    if not node.level:
        return node.module
    parts = [package, *rel.split("/")[:-1]]
    if node.level - 1 > len(parts) - 1:
        return None
    base = parts[: len(parts) - (node.level - 1)]
    return ".".join([*base, node.module]) if node.module else ".".join(base)


def _scan_file(path: Path, rel: str, package: str, rules: Iterable[Rule]) -> list[Violation]:
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
            v = _check_node(node, rule, path, rel, package)
            if v is not None:
                violations.append(v)
    return violations


def _check_node(node: ast.AST, rule: Rule, path: Path, rel: str, package: str) -> Violation | None:
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
        elif isinstance(node, ast.ImportFrom):
            module = _resolved_module(node, rel, package)
            if module is None:
                return None
            # The imported NAMES matter too, not just the source module: `from
            # tessera_embeddings import profiling` resolves to `tessera_embeddings`, which
            # matches no forbidden prefix, yet imports precisely the subpackage the rule
            # keeps out of library code. So each alias is appended and matched as well.
            candidates = [module, *(f"{module}.{a.name}" for a in node.names)]
            if any(c == prefix or c.startswith(f"{prefix}.") for c in candidates):
                return Violation(
                    rule=rule.name,
                    path=path,
                    lineno=node.lineno,
                    detail=f"from {'.' * node.level}{node.module or ''}" + (f" ({module})" if node.level else ""),
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
    # The scanned root's NAME is what relative imports resolve against, so scanning a `src/`
    # wrapper as itself resolves `from ..profiling` inside `src/pkg/inference/` to
    # `src.pkg.profiling`, which matches no absolute forbidden prefix — every
    # relative-import rule then passes on precisely the imports it forbids, and a checker
    # that approves by accident is worse than no checker.
    #
    # So DESCEND rather than refuse: `--source src/` is the command the adapter template
    # documents, and rejecting it made the checker unusable where it is meant to be adopted.
    # Each package beneath the wrapper becomes its own scan root, which is what makes those
    # imports resolve. A root carrying modules of its own is somebody scanning a plain tree:
    # supported, already resolving correctly, left alone.
    if not any(p.suffix == ".py" for p in source_path.iterdir() if p.is_file()):
        nested = sorted(p for p in source_path.iterdir() if p.is_dir() and (p / "__init__.py").exists())
        if not nested and source_path.name == "src":
            # PEP 420 namespace packages carry no `__init__.py`, so the marker above finds
            # nothing and the wrapper gets scanned as itself — the same silent pass, reached
            # by another route. Keyed on the directory being named `src` because that is the
            # only signal: without `__init__.py` nothing distinguishes a namespace package
            # from an ordinary subpackage, and treating every module-bearing directory as a
            # root would descend into the subpackages of a normal package root whose own
            # modules all live one level down.
            nested = sorted(p for p in source_path.iterdir() if p.is_dir() and next(p.rglob("*.py"), None))
        if nested:
            found = [v for pkg in nested for v in _scan_root(pkg, expanded_rules)]
            found.sort(key=lambda v: (str(v.path), v.lineno))
            return found

    return _scan_root(source_path, expanded_rules)


def _scan_root(source_path: Path, expanded_rules: tuple[Rule, ...]) -> list[Violation]:
    """Every violation under one PACKAGE root, whose name relative imports resolve against."""
    violations: list[Violation] = []
    for path in sorted(source_path.rglob("*.py")):
        rel = _path_to_subtree(source_path, path)
        violations.extend(_scan_file(path, rel, source_path.name, expanded_rules))

    violations.sort(key=lambda v: (str(v.path), v.lineno))
    return violations
