"""Repo paths for tests that read the source tree or the fixtures directory.

A handful of tests assert on the SHAPE of the source — that no module outside one place applies
the solar-day offset, that every caller of ``apply_roi_mask`` supplies the mask, that the docs
index names nothing that is gone. Those read ``.py`` and ``.md`` files off disk, so they need to
know where the repo is.

Each of them used to work that out by counting directory levels up from its own file
(``Path(__file__).resolve().parents[2]``), which silently means something different the moment a
test file moves to a different depth. Three of them fail SILENTLY when it is wrong rather than
loudly: an ``rglob`` over a nonexistent root yields nothing, so a test that asserts "no offenders"
passes while checking nothing, and a ``parametrize`` fed from one is reported as skipped rather
than failed.

Anchoring on a marker file instead of a level count removes the whole class. This module sits at
``tests/`` and is the only thing that has to stay put; every test imports the constants from here
and can live at any depth.
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    """The directory holding ``pyproject.toml``, found by walking up from this file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"no pyproject.toml above {__file__}; cannot locate the repo root")


#: The repository root — the directory holding ``pyproject.toml``, ``src/`` and ``context_docs/``.
REPO_ROOT = _repo_root()

#: The package source tree. Tests that scan modules for a structural invariant read this.
SRC_ROOT = REPO_ROOT / "src" / "tessera_embeddings"

#: Golden inputs: ``stac_cassettes/``, ``checkpoints/``, recorded AWS API responses.
FIXTURES = REPO_ROOT / "tests" / "fixtures"
