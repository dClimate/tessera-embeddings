"""A synthetic land-mask coverage repo, for the zone-ROI export path.

`export_zone_roi` and everything that validates its output read one artifact: a
per-zone ``tile_live_2048`` bitmap with a ``registry_sha256`` on the group. Both
the domain tests and the flow tests need to control which tiles are live, and
building that repo is the same fifteen lines either way — so it lives here, and
a change to the coverage schema is made once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage import zone_grid
from tessera_embeddings.storage.zarr_store import open_or_create_repo

#: Stamped on the coverage group. Real deliveries carry a registry digest here and
#: the exported ROI records the one it was built from, so the two can be compared.
COVERAGE_SHA = "test-coverage-sha"


def make_coverage(tmp_path: Path, zone: str, live_tiles: list[tuple[int, int]]) -> str:
    """A coverage repo whose ``zone`` group marks exactly ``live_tiles`` as land.

    ``live_tiles`` are ``(tile_row, tile_col)`` on the zone's 2048-px tile grid;
    an empty list is a legitimate all-ocean zone. Returns the repo path.
    """
    spec = zone_grid.zone(zone)
    nty, ntx = spec.height // SHARD_PX, spec.width // SHARD_PX
    path = str(tmp_path / "coverage.icechunk")
    repo, _ = open_or_create_repo(path)
    session = repo.writable_session("main")
    node = zarr.open_group(session.store, mode="a").require_group(zone)
    tl = np.zeros((nty, ntx), dtype=bool)
    for r, c in live_tiles:
        tl[r, c] = True
    node.create_array("tile_live_2048", data=tl, chunks=(nty, ntx), dimension_names=("tile_row", "tile_col"))
    node.attrs["registry_sha256"] = COVERAGE_SHA
    session.commit("seed coverage")
    return path
