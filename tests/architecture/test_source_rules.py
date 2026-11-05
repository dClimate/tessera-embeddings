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
