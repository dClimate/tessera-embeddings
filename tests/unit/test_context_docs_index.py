"""`context_docs/README.md` must agree with what is actually on disk.

An index is only worth having while it is true, and a stale one is worse than none: it sends
a reader looking for a document that has moved, and it hides a document nobody listed. Both
failures are silent, and both have happened in this programme — a docstring in the downstream
repo claimed "no test guards this" long after a test did, and an external reviewer read the
comment and reported a gap that was already closed.

The rule this file enforces is small enough to state in one line: **every markdown file under
``context_docs/`` is named in the index, and every path the index names exists.**

Kept deliberately mechanical. It does not check that a description is accurate or that the
grouping is sensible — those need a reader. It checks the one property that decays on its own
whenever someone adds or removes a file and forgets the index.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._paths import REPO_ROOT

REPO = REPO_ROOT
CONTEXT_DOCS = REPO / "context_docs"
INDEX = CONTEXT_DOCS / "README.md"

#: Any ``word.md`` token, wherever it appears — the index names files inside a fenced layout
#: block, in prose, and in markdown links, and all three count as listing it.
_MARKDOWN_NAME = re.compile(r"[\w./-]+\.md")


def _index_text() -> str:
    return INDEX.read_text(encoding="utf-8")


def _tracked_docs() -> list[Path]:
    """Every markdown file under ``context_docs/``, excluding the index itself."""
    return sorted(p for p in CONTEXT_DOCS.rglob("*.md") if p != INDEX)


def test_the_index_exists() -> None:
    """Asserted separately so its absence reports as itself, not as 43 missing entries."""
    assert INDEX.is_file(), f"{INDEX.relative_to(REPO)} is the entry point to this tree"


@pytest.mark.parametrize("doc", _tracked_docs(), ids=lambda p: str(p.relative_to(CONTEXT_DOCS)))
def test_every_document_is_listed(doc: Path) -> None:
    """A document the index does not name is indistinguishable from an abandoned one.

    Parametrised per document so a failure names the file that was added without an index
    row, rather than reporting a set difference the author then has to read.
    """
    name = doc.name
    assert name in _index_text(), (
        f"{doc.relative_to(REPO)} is not named in context_docs/README.md. "
        "Add it to the layout block in the same commit that adds the file."
    )


def _repo_markdown_names() -> set[str]:
    """Markdown filenames anywhere in the working tree, excluding hidden directories.

    The exclusion is load-bearing rather than tidiness. ``.venv`` holds vendored docs, and
    ``.claude/worktrees/`` holds whole checkouts of other branches — so a file deleted on this
    branch is still on disk under a worktree, and counting it here makes the staleness check
    below pass on a developer machine and fail in CI. That happened: nine documents were removed
    from the index's tree and the guard stayed green locally because old worktrees still held
    them.
    """
    return {p.name for p in REPO.rglob("*.md") if not any(part.startswith(".") for part in p.relative_to(REPO).parts)}


def test_the_index_names_nothing_that_is_gone() -> None:
    """The other direction: a row surviving its file sends readers after a dead path.

    Only names that look like a path *into* this tree are checked. The index also cites docs
    elsewhere in the repo and on the web, and those are not this test's business.
    """
    on_disk = {p.name for p in _tracked_docs()} | {INDEX.name}
    named = {Path(m).name for m in _MARKDOWN_NAME.findall(_index_text())}
    # Names that exist somewhere else in the repo are legitimate cross-references.
    elsewhere = _repo_markdown_names()
    stale = sorted(named - on_disk - elsewhere)
    assert not stale, (
        f"context_docs/README.md names {stale}, which no longer exist. "
        "Remove the row, or point it at whatever absorbed the document."
    )
