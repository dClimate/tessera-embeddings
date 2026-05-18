"""Public-API surface tests.

Two invariants:

1. ``tessera_embeddings.__init__`` does not re-export underscore-prefixed
   names. Everything in ``__all__`` is meant to be public.
2. Every name in ``__all__`` has an entry in ``docs/public-api.md`` so
   the docs and the code can't drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import tessera_embeddings

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_all_contains_no_private_names() -> None:
    """No underscore-prefixed names in ``__all__``."""
    private = [name for name in tessera_embeddings.__all__ if name.startswith("_")]
    assert not private, f"Underscore-prefixed names in __all__: {private}"


def test_init_does_not_import_private_names() -> None:
    """``__init__.py`` itself avoids ``from X import _name``."""
    init_path = REPO_ROOT / "src" / "tessera_embeddings" / "__init__.py"
    text = init_path.read_text()
    matches = re.findall(r"from\s+\S+\s+import\s+_\w+", text)
    assert not matches, f"Private imports in __init__.py: {matches}"


def test_public_api_doc_lists_every_exported_name() -> None:
    """Every name in ``__all__`` appears verbatim in ``docs/public-api.md``."""
    public_api = (REPO_ROOT / "docs" / "public-api.md").read_text()
    missing = [name for name in tessera_embeddings.__all__ if name not in public_api]
    assert not missing, f"Public symbols missing from docs/public-api.md: {missing}"


def test_every_exported_name_resolves() -> None:
    """Every name in ``__all__`` is actually present on the package object."""
    missing = [name for name in tessera_embeddings.__all__ if not hasattr(tessera_embeddings, name)]
    assert not missing, f"Names in __all__ not found on package: {missing}"
