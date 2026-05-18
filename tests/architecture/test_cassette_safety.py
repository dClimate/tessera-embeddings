"""Cassette safety: no leaked credentials in committed VCR fixtures.

Every YAML under ``tests/fixtures/stac_cassettes/`` is grep-checked
for known credential markers. A failure means a cassette was
recorded with auth filters misconfigured and must NOT be merged
without scrubbing.

Defence in depth: ``pytest-recording`` is configured with
``filter_headers`` per-test, but a misconfiguration there would
otherwise commit the leak silently. This test fires before any
cassette ships.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CASSETTE_DIR = REPO_ROOT / "tests" / "fixtures" / "stac_cassettes"

# Substrings (case-insensitive) that should never appear inside a
# committed cassette. Each indicates a leaked or unfiltered credential.
_FORBIDDEN_SUBSTRINGS = (
    "authorization:",
    "x-amz-security-token",
    "x-amz-signature",
    "set-cookie:",
    "cookie:",
    "edl-token",
    "earthdatalogin",
    " bearer ",  # space-padded so we don't flag legitimate "Bearer" docs
)

# Patterns that are stricter than substring matching. Useful for
# tokens that follow a known shape (e.g. AWS access key ID format).
_FORBIDDEN_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),  # long-lived AWS access key id
    re.compile(r"ASIA[0-9A-Z]{16}"),  # short-lived AWS access key id (STS)
)


def test_cassettes_directory_exists() -> None:
    """The cassette directory exists with at minimum a README.

    A missing directory is a setup error. An empty directory (no .yaml)
    is fine — Phase 10 ships before any cassette is recorded.
    """
    assert CASSETTE_DIR.is_dir(), f"Missing cassette dir: {CASSETTE_DIR}"
    assert (CASSETTE_DIR / "README.md").exists(), "Cassette dir must include README.md"


def test_no_credentials_in_cassettes() -> None:
    """Every committed cassette is free of known credential markers."""
    cassettes = sorted(CASSETTE_DIR.glob("*.yaml"))
    if not cassettes:
        # No cassettes yet → nothing to check. Test passes trivially.
        # Phase 10 lands the directory; Phase 13 (this file) lands the
        # guard; the cassettes themselves arrive in a separate PR.
        return

    failures: list[str] = []
    for cassette in cassettes:
        text = cassette.read_text()
        lower = text.lower()
        for marker in _FORBIDDEN_SUBSTRINGS:
            if marker in lower:
                failures.append(f"{cassette.name}: contains forbidden marker {marker!r}")
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(text):
                failures.append(f"{cassette.name}: matches forbidden pattern {pattern.pattern!r}")

    assert not failures, "Cassette credential leaks detected:\n" + "\n".join(failures)
