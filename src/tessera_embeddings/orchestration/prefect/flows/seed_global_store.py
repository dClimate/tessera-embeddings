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
from tessera_embeddings.storage.zarr_store import is_missing_repo, read_time_values
from tessera_embeddings.storage.zone_grid import CAMPAIGN_YEARS, ZONES, year_of


@flow(name="seed-global-store")
def seed_global_store(
    *,
    paths: BucketPaths,
    name: str = "tessera",
    years: tuple[int, ...] = CAMPAIGN_YEARS,
    model_version: str | None = None,
    s3_region: str | None = None,
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
        s3_region: Optional S3 region for the global store, forwarded to
            open/create — like the campaign and fill paths, so a non-default-region
            deployment can seed at all.

    Returns:
        Summary: store path, zones seeded this run, zones already present, total.
    """
    log = get_run_logger()

    # Lazy AWS import so the flow file imports on non-AWS machines (arch tests). This
    # mandatory FIRST step opens/creates the repo, so a callback-only or non-default-
    # region deployment needs the credential callback + region here too — not just in
    # the campaign and fill flows.
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials

    store_path = paths.global_store(name)
    # Default the recorded checkpoint identity to the build's checkpoint filename so
    # the fill's checkpoint gate is effective by default (it only compares when the
    # store carries a checkpoint_id). geoemb:model alone can't tell aws from mpc.
    model_version = model_version or checkpoint_filename()

    # Open-or-create with the global config (manifest split + preload). A missing
    # repo raises on open; create_global_repo persists the config via save_config.
    seeded: set[str]
    try:
        repo = open_global_repo(store_path, get_credentials=iam_icechunk_credentials, region=s3_region)
    except FileNotFoundError:
        log.info("Creating global store %s", store_path)
        repo = create_global_repo(store_path, get_credentials=iam_icechunk_credentials, region=s3_region)
        seeded = set()
    except icechunk.IcechunkError as exc:
        # Only a genuinely-missing repo means "create it". Auth/throttle/timeout/
        # corruption errors must NOT fall into the create path (it would fail against
        # the live repo if the transient error clears, and buries the real cause) —
        # re-raise them. Mirrors the ingest/campaign is_missing_repo handling.
        if not is_missing_repo(exc):
            raise
        log.info("Creating global store %s", store_path)
        repo = create_global_repo(store_path, get_credentials=iam_icechunk_credentials, region=s3_region)
        seeded = set()
    else:
        # The repo exists — but may still be UNSEEDED: a prior run can create the
        # repo (and persist its config) then crash BEFORE the first seed_zone_groups
        # commit writes the root group. campaign_status then reads a rootless store
        # and raises GroupNotFoundError; treat that as "nothing seeded yet" so this
        # retry seeds the store (the flow's advertised idempotency) instead of
        # propagating and wedging every retry on the half-created repo.
        try:
            seeded = set(campaign_status(repo, years=years).zones)
        except zarr.errors.GroupNotFoundError:
            log.info("Store %s exists but has no root group yet — treating as unseeded", store_path)
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
