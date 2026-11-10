"""The package's public names resolve.

Ruff's F822 cannot check this: it skips ``__init__.py``, and it cannot see names the package
resolves lazily through its module-level ``__getattr__``.
"""

from __future__ import annotations

import tessera_embeddings


def test_every_exported_name_resolves() -> None:
    """Every name in ``__all__`` is actually present on the package object."""
    missing = [name for name in tessera_embeddings.__all__ if not hasattr(tessera_embeddings, name)]
    assert not missing, f"Names in __all__ not found on package: {missing}"
