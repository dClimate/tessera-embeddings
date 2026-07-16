"""Seed the 120 UTM-zone groups of the global embeddings store (ADR-008 D1).

Metadata-only and cluster-less — like :mod:`generate_roi`, it runs entirely on
the flow runner (creating a zone group is a handful of schema arrays with no
chunk data, so cost is independent of extent). Idempotent: it seeds only the
groups not already present, so a re-run after a partial seed finishes the job
without touching (and thus without corrupting) the zones already landed.

This is the first step of the global campaign — run it once before any
:mod:`fill_zone_year` runs; the fill runner treats the seeded group as the grid
authority and creates nothing.
"""

from __future__ import annotations

from typing import Any, cast

import icechunk
import zarr
from prefect import flow, get_run_logger

from tessera_embeddings.config.inference import checkpoint_filename
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.storage.campaign import campaign_status
from tessera_embeddings.storage.global_store import create_global_repo, open_global_repo, seed_zone_groups
from tessera_embeddings.storage.zarr_store import read_time_values
from tessera_embeddings.storage.zone_grid import CAMPAIGN_YEARS, ZONES, year_of


@flow(name="seed-global-store")
def seed_global_store(
    *,
    paths: BucketPaths,
    name: str = "tessera",
    years: tuple[int, ...] = CAMPAIGN_YEARS,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Create the global-store repo (if absent) and seed every unseeded zone group.

    Args:
        paths: Deployment storage contract; the repo is ``paths.global_store(name)``.
        name: Global-store repo basename (default ``"tessera"``).
        years: Campaign year axis to pre-allocate on each group (fixed at seeding).
        model_version: Model-identity attr stamped as ``checkpoint_id`` on the root.
            Defaults to :func:`checkpoint_filename` so the fill's model gate can
            distinguish the concrete checkpoint (the ``aws``/``mpc`` v1.1 checkpoints
            share one ``geoemb:model`` URL); without it the checkpoint gate is a
            no-op. Override to record a custom identity.

    Returns:
        Summary: store path, zones seeded this run, zones already present, total.
    """
    log = get_run_logger()
    store_path = paths.global_store(name)
    # Default the recorded checkpoint identity to the build's checkpoint filename so
    # the fill's checkpoint gate is effective by default (it only compares when the
    # store carries a checkpoint_id). geoemb:model alone can't tell aws from mpc.
    model_version = model_version or checkpoint_filename()

    # Open-or-create with the global config (manifest split + preload). A missing
    # repo raises on open; create_global_repo persists the config via save_config.
    try:
        repo = open_global_repo(store_path)
        seeded = set(campaign_status(repo, years=years).zones)
    except (FileNotFoundError, icechunk.IcechunkError):
        log.info("Creating global store %s", store_path)
        repo = create_global_repo(store_path)
        seeded = set()

    # A retry after a partial seed must use the SAME year axis as the groups already
    # landed — the axis is fixed at seeding (ADR-008 D1), so seeding the remainder
    # with a different `years` would silently leave the store with mixed axes.
    if seeded:
        root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
        probe = cast("zarr.Group", root[sorted(seeded)[0]])
        existing_years = tuple(year_of(t) for t in read_time_values(probe))
        if existing_years != tuple(years):
            raise ValueError(
                f"years {tuple(years)} differ from the store's existing axis {existing_years} "
                f"({len(seeded)} zone(s) already seeded) — reseeding with a different axis would corrupt the store."
            )

    todo = [spec for zone_name, spec in ZONES.items() if zone_name not in seeded]
    if not todo:
        log.info("All %d zone groups already seeded in %s", len(ZONES), store_path)
        return {"store_path": store_path, "seeded_now": 0, "already_seeded": len(seeded), "total": len(ZONES)}

    snapshot = seed_zone_groups(
        repo, todo, years=years, model_version=model_version, commit_msg=f"seed {len(todo)} zone group(s)"
    )
    log.info("Seeded %d zone group(s) into %s (%s)", len(todo), store_path, snapshot)
    return {
        "store_path": store_path,
        "seeded_now": len(todo),
        "already_seeded": len(seeded),
        "total": len(ZONES),
        "snapshot_id": snapshot,
    }
