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
import subprocess
from pathlib import Path, PurePosixPath

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
    """Markdown filenames the REPOSITORY holds, asked of git rather than of the filesystem.

    The staleness check below excuses an index row whose file exists somewhere else in the
    repo, and "somewhere else in the repo" has to mean *tracked*. Walking the working tree
    answers a different question, and gets it wrong twice over.

    **Wrong answer.** ``.venv`` holds vendored documentation and ``.claude/worktrees/`` holds
    whole checkouts of other branches, so a file deleted on this branch is still on disk and
    the excuse fires for it. That happened: nine documents were removed and this guard stayed
    green locally, while CI — which has neither directory — would have failed on all nine.

    **Slow answer.** ``rglob("*.md")`` walks 2,975 files in ~9.5 s on a checkout carrying
    worktrees, of which 66 are the repository's, and that made this the slowest test in the
    unit suite by a factor of two. ``git ls-files`` returns the same 66 in ~8 ms.

    Excluding hidden directories fixes the correctness half and none of the speed, and it
    goes on guessing at the question instead of asking it.
    """
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO, capture_output=True, text=True, check=True)
    return {PurePosixPath(line).name for line in out.stdout.splitlines() if line}


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
