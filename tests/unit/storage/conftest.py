"""Seeded global stores, built once per worker and copied into each test.

Seeding a zone group writes every layout array's metadata plus its coordinate arrays, which costs
a few hundred milliseconds per test. A local Icechunk repository is a plain directory, so each
distinct seeding is built once per xdist worker and ``shutil.copytree``'d into the test's own
``tmp_path``: every test still mutates a private store.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from tessera_embeddings.storage import global_store
from tessera_embeddings.storage.zone_grid import ZoneSpec


@pytest.fixture(scope="session")
def _seed_templates(tmp_path_factory) -> Callable[..., Path]:
    built: dict[str, Path] = {}

    def template(zones: tuple[ZoneSpec, ...], **seed_kwargs) -> Path:
        key = repr((zones, sorted(seed_kwargs.items())))
        if key not in built:
            path = tmp_path_factory.mktemp("seed") / "store.icechunk"
            global_store.seed_zone_groups(global_store.create_global_repo(str(path)), zones, **seed_kwargs)
            built[key] = path
        return built[key]

    return template


@pytest.fixture
def seeded_store(tmp_path, _seed_templates) -> Callable[..., str]:
    """``seeded_store(zones, name=..., **seed_zone_groups kwargs)`` -> path of a private seeded copy."""

    def copy(zones: Iterable[ZoneSpec], *, name: str = "global.icechunk", **seed_kwargs) -> str:
        dest = tmp_path / name
        shutil.copytree(_seed_templates(tuple(zones), **seed_kwargs), dest)
        return str(dest)

    return copy
