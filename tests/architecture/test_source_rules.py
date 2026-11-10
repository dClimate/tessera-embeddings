"""Rules about the package's source that no linter states: checked on the parsed AST."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "tessera_embeddings"


def test_every_assembly_commit_goes_through_the_traced_helper():
    """No bare ``session.commit`` anywhere an assembly can reach.

    Every assembly commit can stall, and a commit outside a tracing scope stalls silently: our last
    line is printed before the call and the next only after it returns. So the property pinned is
    not "these commits are wrapped" but "none is left bare", which also catches the next one added.
    """
    bare = []
    for path in (SRC / "inference" / "assembly.py", SRC / "storage" / "shard_writer.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "commit"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"session", "fork"}
            ):
                bare.append(f"{path.name}:{node.lineno}")
    assert not bare, f"commit(s) outside a tracing scope: {bare}; use traced_commit"


# --- fault injection: the drill gate has no second way in ---------------------------------------

FAULT_MODULE = SRC / "config" / "fault_injection.py"
_ENV_READS = {"environ", "environb", "getenv"}


def _call_sites(name: str) -> list[Path]:
    """The file of every call of ``name`` (bare or attribute) across the package."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                fn = node.func
                called = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if called == name:
                    found.append(path)
    return found


def test_only_arm_constructs_an_armed_fault():
    """An injection point accepts nothing but an ``ArmedFault``, so if ``arm`` is its only source,
    every fault that fires has passed the deployment allowlist and the hosted-fault check. A
    second constructor anywhere retires that guarantee.
    """
    assert _call_sites("ArmedFault") == [FAULT_MODULE]


def test_the_package_hard_exits_in_exactly_one_place():
    """A hard exit skips every handler by design, which makes it a faithful death and an
    unacceptable thing to have anywhere but the one site behind the drill gate.
    """
    assert _call_sites("_exit") == [FAULT_MODULE]


def test_nothing_about_a_fault_is_read_from_the_environment():
    """An environment variable can be left behind in a task definition or an image and inherited
    by an ordinary run; a flow parameter is stated per dispatch and stored on the run.
    """
    tree = ast.parse(FAULT_MODULE.read_text())
    reads = [
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr in _ENV_READS)
        or (isinstance(node, ast.Name) and node.id in _ENV_READS)
    ]
    assert not reads, f"fault_injection.py reads the environment at line(s) {reads}"
