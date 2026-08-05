"""The staging fingerprint must move when tile CONTENT-determining code moves.

The campaign namespaces its staging prefix by this digest, so a retry reuses tiles
whenever it is unchanged. That makes the two failure directions wildly asymmetric: an
identity that moves too readily costs a re-inference, and one that fails to move
assembles tiles from two code versions into a single write-once zone-year, silently.

So these tests pin the closure, not a value. What matters is which files can change the
digest — a hand-maintained list of them goes stale the first time an import is added,
and goes stale in the direction that loses data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera_embeddings.config.code_identity import first_party_import_closure
from tessera_embeddings.config.inference import (
    _STAGED_OUTPUT_SOURCES,
    inference_code_identity,
)

_ROOT = Path(__file__).resolve().parents[2] / "src" / "tessera_embeddings"


def _covered() -> set[str]:
    """Package-relative paths of every file the fingerprint hashes."""
    seed: list[Path] = []
    for entry in _STAGED_OUTPUT_SOURCES:
        target = _ROOT / entry
        if target.is_dir():
            seed.extend(p for p in target.rglob("*.py") if "__pycache__" not in p.parts)
        else:
            seed.append(target)
    return {str(p.relative_to(_ROOT)) for p in first_party_import_closure(seed, _ROOT)}


@pytest.mark.parametrize(
    ("module", "why"),
    [
        ("storage/zarr_store.py", "compute_doy builds a model input from the timestamps"),
        ("config/time_windows.py", "TimeWindow decides which observations are used"),
        ("inference/dataset.py", "the valid-pixel gate decides which pixels are embedded"),
        ("inference/sampling.py", "the sampler decides which observations reach the model"),
        ("config/inference.py", "the checkpoint, batching and band order live here"),
    ],
)
def test_content_determining_modules_are_covered(module: str, why: str) -> None:
    """Each of these can change a staged tile's contents, so each must move the digest.

    The first two are the ones a hand-listed identity missed: they are reached through
    an import from `inference/data_loading.py` rather than named directly.
    """
    assert module in _covered(), f"{module} not fingerprinted, but {why}"


def test_the_closure_is_reached_transitively_not_listed() -> None:
    """The seed names far less than the fingerprint covers — that gap IS the fix."""
    seeded = set(_STAGED_OUTPUT_SOURCES)
    covered = _covered()

    beyond_the_seed = {c for c in covered if not (c.startswith("inference/") or c in seeded)}
    assert beyond_the_seed, "the closure added nothing — it has stopped following imports"
    assert "storage/zarr_store.py" in beyond_the_seed


def test_orchestration_and_providers_stay_out() -> None:
    """The narrowing has to still narrow, or a hotfix anywhere abandons every tile.

    That is what the old whole-build identity did, and why it was replaced. Inference
    does not import these, so they cannot change what a staged tile contains.
    """
    covered = _covered()
    assert not [c for c in covered if c.startswith(("orchestration/", "providers/", "profiling/"))]
    assert len(covered) < len(list(_ROOT.rglob("*.py"))), "the closure has swallowed the package"


def test_the_digest_is_stable_across_calls() -> None:
    """Nothing in the walk may depend on iteration or filesystem order."""
    assert inference_code_identity() == inference_code_identity()
    assert inference_code_identity().startswith("infcode-")


def test_an_edit_to_a_transitive_dependency_moves_the_digest(tmp_path: Path) -> None:
    """The end-to-end property: touch a reached file, get a different prefix.

    Built as a miniature package rather than by editing the real tree, so the test
    cannot corrupt the checkout it is verifying.
    """
    root = tmp_path / "pkg"
    (root / "inference").mkdir(parents=True)
    (root / "storage").mkdir()
    (root / "inference" / "runner.py").write_text("from tessera_embeddings.storage.helper import doy\n")
    helper = root / "storage" / "helper.py"
    helper.write_text("def doy(x):\n    return x\n")

    seed = [root / "inference" / "runner.py"]
    reached = first_party_import_closure(seed, root)
    assert helper in reached, "an imported module must be reached"

    def digest(paths: set[Path]) -> tuple[str, ...]:
        return tuple(sorted(p.read_text() for p in paths))

    before = digest(reached)
    helper.write_text("def doy(x):\n    return x + 1\n")  # a different model input
    assert digest(first_party_import_closure(seed, root)) != before
