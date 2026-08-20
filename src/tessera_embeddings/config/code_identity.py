"""Fingerprinting the source that decides a stored artifact's CONTENT.

Two pipeline stages need the same answer to the same question: *was the data already
in this store produced by the code running now?* Inference asks it of a staging prefix
before resuming into it; ingest asks it of an interrupted mosaic before appending to it.
Both have the same failure if they guess wrong — one artifact holding output from two
code versions, published under a single completion mark, with nothing recording that it
happened.

A **source hash over an import closure** is the answer both use, rather than a build
identity (an AMI id, an image digest). A build identity is correct but far too wide:
it moves on every re-bake and every hotfix anywhere in the repo, so it abandons work
that is perfectly reusable. Hashing only the code that determines the artifact's
content keeps a hotfix cheap while still catching the changes that matter.

**What this deliberately does not cover: dependency drift.** A source hash is blind to
a new library under the same source. Callers bound that gap their own way — the fill
pins one AMI for a whole campaign, so a single run cannot straddle two images — and a
deliberate mid-campaign upgrade wants an explicit force-new token rather than a silent
reuse.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

_PACKAGE = __name__.split(".")[0]


def first_party_import_closure(seed: list[Path], root: Path) -> set[Path]:
    """Every in-package module reachable from *seed*, following imports transitively.

    The seed names the code that OBVIOUSLY produces the artifact; this finds the code it
    delegates to. Without it the identity misses exactly the dependencies that matter
    most — inference's ``data_loading`` calls ``compute_doy`` from ``storage.zarr_store``
    to build a model input, and reads ``TimeWindow`` from ``config.time_windows`` to
    choose which observations are used. A change to either alters the output while
    leaving a hand-listed identity unmoved.

    A closure rather than a longer list, because a list only covers the dependencies
    someone remembered: it goes stale the first time an import is added, and it goes
    stale silently, in the direction that loses data.

    Parses rather than imports — this runs on a flow runner that has no torch, and
    importing to inspect would execute module bodies for a hash. ``from x import y``
    contributes both ``x`` and ``x.y`` as candidates, since either may name the module.
    Non-package imports and names that resolve to no file are skipped.
    """

    def module_file(dotted: str) -> Path | None:
        rel = dotted.removeprefix(_PACKAGE + ".").replace(".", "/")
        for candidate in (root / f"{rel}.py", root / rel / "__init__.py"):
            if candidate.exists():
                return candidate
        return None

    def absolute_module(node: ast.ImportFrom, path: Path) -> str | None:
        """The dotted name ``node`` imports from, resolved against ``path`` if relative.

        A relative import's ``node.module`` is the tail alone — ``"providers"`` for
        ``from .providers import PROVIDERS`` — so a first-party filter on the package
        prefix rejects it. That mattered: package ``__init__`` files re-export with
        relative imports, so the closure stopped at every one of them. ``config`` was
        reached and ``config/providers.py`` was not, leaving the STAC collections, band
        lists, resolutions and baseline settings outside the fingerprint of the code
        that ingests with them.
        """
        if not node.level:
            return node.module if node.module and node.module.startswith(_PACKAGE) else None
        # `level` counts dots. One means the importing module's own package — which is the
        # containing directory for a plain module AND for an ``__init__.py``, since a
        # package's ``__init__`` IS that package. Each further dot climbs one more.
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        try:
            rel = base.relative_to(root)
        except ValueError:  # pragma: no cover - a level that climbs out of the package
            return None
        parts = [_PACKAGE, *rel.parts, *([node.module] if node.module else [])]
        return ".".join(parts)

    def imported_modules(path: Path) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names if a.name.startswith(_PACKAGE))
            elif isinstance(node, ast.ImportFrom) and (dotted := absolute_module(node, path)):
                found.add(dotted)
                found.update(f"{dotted}.{a.name}" for a in node.names)
        return found

    reached, stack = set(seed), list(seed)
    while stack:
        for dotted in imported_modules(stack.pop()):
            path = module_file(dotted)
            if path is not None and path not in reached:
                reached.add(path)
                stack.append(path)
    return reached


def source_identity(sources: tuple[str, ...], prefix: str) -> str:
    """A ``<prefix>-<digest>`` fingerprint of *sources* and everything they import.

    ``sources`` are package-relative paths — a directory contributes every ``.py`` under
    it, a file contributes itself — and each is required to exist. A moved or renamed
    module must fail loudly here rather than silently fingerprint fewer files, because
    fingerprinting fewer files is how two code versions come to share one artifact.

    Contents are hashed with their package-relative paths, sorted, so the digest is
    stable across machines and checkouts.
    """
    root = Path(__file__).resolve().parent.parent
    seed: list[Path] = []
    for entry in sources:
        target = root / entry
        if target.is_dir():
            seed.extend(p for p in target.rglob("*.py") if "__pycache__" not in p.parts)
        elif target.is_file():
            seed.append(target)
        else:  # pragma: no cover - a rename must fail loudly, not fingerprint nothing
            raise FileNotFoundError(
                f"{target} is listed as a content source but does not exist. A moved or "
                f"renamed module must update that list — silently fingerprinting fewer files "
                f"would let two code versions share one artifact."
            )
    digest = hashlib.sha256()
    for path in sorted(first_party_import_closure(seed, root)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    return f"{prefix}-{digest.hexdigest()[:16]}"
