"""Seed the 120 UTM-zone groups of the global embeddings store (ADR-008 D1).

Metadata-only and cluster-less: it runs entirely on the flow runner, because a zone
group is a handful of schema arrays with no chunk data and so costs the same at any
extent. Idempotent — it seeds only the groups not already present, so a re-run after a
partial seed finishes the job without touching (and thus corrupting) the zones already
landed.

The first step of the global campaign: run it once before any :mod:`fill_zone_year`,
which treats the seeded group as the grid authority and creates nothing.
"""

from __future__ import annotations

from typing import Any, cast

import icechunk
import zarr
from prefect import flow, get_run_logger

from tessera_embeddings.config.inference import checkpoint_filename
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.store_layout import GLOBAL
from tessera_embeddings.storage.campaign import campaign_status
from tessera_embeddings.storage.global_store import (
    _check_layout_matches,
    check_root_identity,
    create_global_repo,
    missing_seeded_arrays,
    open_global_repo,
    seed_zone_groups,
    stamp_root_identity,
)
from tessera_embeddings.storage.time_axis import CAMPAIGN_YEARS, read_time_values, year_of
from tessera_embeddings.storage.zarr_store import is_missing_repo
from tessera_embeddings.storage.zone_grid import ZONES


@flow(name="seed-global-store")
def seed_global_store(
    *,
    paths: BucketPaths,
    name: str = "tessera",
    years: tuple[int, ...] = CAMPAIGN_YEARS,
    model_version: str | None = None,
    optical_min_obs: int | None = None,
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
            share one ``geoemb:model`` URL); without it that gate is a no-op.
        optical_min_obs: Minimum valid optical observations for a pixel to be embedded,
            stamped on the root as part of its write-once identity. **Deliberately has no
            default and is not read from the config constant**: it decides what the product
            contains, can never be changed for this store afterwards, and a value inherited
            from whatever a module happened to hold is not a decision anyone took. ``None``
            seeds a store with no such rule, as older stores have.
        s3_region: Optional S3 region for the global store, forwarded to open/create — like
            the campaign and fill paths, so a non-default-region deployment can seed at all.

    Returns:
        Summary: store path, zones seeded this run, zones already present, total.
    """
    log = get_run_logger()

    # Lazy AWS import so the flow file imports on non-AWS machines (arch tests). This
    # mandatory FIRST step opens/creates the repo, so a callback-only or non-default-region
    # deployment needs the credential callback + region here too, not just in the campaign
    # and fill flows.
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials, icechunk_credentials_for

    store_path = paths.global_store(name)
    # CHOSEN FROM THE DESTINATION, not hard-wired. When the store lives in the partner-owned
    # published bucket and a writer role is configured, the task role cannot create or open it
    # and the seed fails with AccessDenied before anything exists. `icechunk_credentials_for`
    # returns the task-role provider unchanged for our own buckets, so the substitution only
    # happens where it is required.
    store_credentials = icechunk_credentials_for(store_path, iam_icechunk_credentials)
    # Default the recorded checkpoint identity to the build's checkpoint filename so the fill's
    # checkpoint gate is effective by default (it only compares when the store carries a
    # checkpoint_id). geoemb:model alone can't tell aws from mpc.
    model_version = model_version or checkpoint_filename()

    # Open-or-create with the global config (manifest split + preload). A missing repo raises
    # on open; create_global_repo persists the config via save_config.
    seeded: set[str]
    try:
        repo = open_global_repo(store_path, get_credentials=store_credentials, region=s3_region)
    except FileNotFoundError:
        log.info("Creating global store %s", store_path)
        repo = create_global_repo(store_path, get_credentials=store_credentials, region=s3_region)
        seeded = set()
        # A just-created store holds nothing; initialised because the unstamped-store refusal
        # below reads it on EVERY path, not only the every-zone-exists one.
        cells_landed = 0
    except icechunk.IcechunkError as exc:
        # Only a genuinely-missing repo means "create it". Auth/throttle/timeout/corruption
        # errors must NOT fall into the create path: it would fail against the live repo once
        # the transient error clears, and buries the real cause. Mirrors the ingest/campaign
        # is_missing_repo handling.
        if not is_missing_repo(exc):
            raise
        log.info("Creating global store %s", store_path)
        repo = create_global_repo(store_path, get_credentials=store_credentials, region=s3_region)
        seeded = set()
        # A just-created store holds nothing; initialised because the unstamped-store refusal
        # below reads it on EVERY path, not only the every-zone-exists one.
        cells_landed = 0
    else:
        # The repo exists but may still be UNSEEDED: a prior run can create it (and persist
        # its config) then crash BEFORE the first seed_zone_groups commit writes the root
        # group. campaign_status then reads a rootless store and raises GroupNotFoundError;
        # treat that as "nothing seeded yet" so this retry seeds the store (the flow's
        # advertised idempotency) instead of wedging every retry on the half-created repo.
        try:
            status = campaign_status(repo, years=years)
        except zarr.errors.GroupNotFoundError:
            log.info("Store %s exists but has no root group yet — treating as unseeded", store_path)
            seeded = set()
            cells_landed = 0
        else:
            # Kept whole, not reduced to its zone names: the every-zone-exists path below
            # needs to know whether any CELL has landed before it may stamp an identity.
            seeded = set(status.zones)
            cells_landed = status.zone_years_done

    # A retry after a partial seed must use the SAME year axis as the groups already
    # landed — the axis is fixed at seeding (ADR-008 D1), so seeding the remainder
    # with a different `years` would silently leave the store with mixed axes.
    if seeded:
        root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
        probe_name = sorted(seeded)[0]
        probe = cast("zarr.Group", root[probe_name])
        existing_years = tuple(year_of(t) for t in read_time_values(probe))
        if existing_years != tuple(years):
            raise ValueError(
                f"years {tuple(years)} differ from the store's existing axis {existing_years} "
                f"({len(seeded)} zone(s) already seeded) — reseeding with a different axis would corrupt the store."
            )
        # And the LAYOUT, alongside the axis it belongs with. `seed_zone_groups` runs this
        # check too, but only while it is CREATING something — so a rerun against a store whose
        # 120 groups all exist reported "seeded successfully" without ever applying it, and an
        # older-schema store passed a seed the helper itself would have refused, failing later
        # at fill or read against a store an operator was told was fine. One group suffices:
        # every group shares one axis and one layout, the same invariant the helper's own
        # single-group check rests on.
        _check_layout_matches(probe, probe_name, GLOBAL)
        # And COMPLETENESS, which the layout check does not cover: it iterates `layout.arrays`,
        # the six data arrays, while a seed also writes six coordinates beside them. An older
        # store missing `month` or `time_bnds` therefore passed the layout check and still
        # failed at fill or read — the exact case this block exists to catch.
        if missing := missing_seeded_arrays(probe, GLOBAL):
            raise ValueError(
                f"Zone group {probe_name!r} in {store_path} is missing {sorted(missing)}. The store "
                f"was seeded by an older schema or a crashed run, so reporting this reseed as "
                f"successful would hand the campaign a store that fails at fill or read instead. "
                f"Reseed into a fresh store, or remove the incomplete groups and seed them again."
            )

    def _refuse_stamping_over_landed_cells(root_attrs: dict, how_many_groups: str) -> None:
        """Refuse to stamp a write-once identity onto a store that already holds cells.

        Called on BOTH seeding paths, because both stamp: the every-zone-exists path through
        `stamp_root_identity`, the incremental path as a side effect of `seed_zone_groups`,
        which writes the root identity whenever it is absent while adding groups. A store with
        SOME zones seeded, no root identity and landed cells would otherwise have those cells
        attributed to an encoder and a depth rule nobody verified — and the stamp is write-once
        and read by every later fill to decide what may write, so later fills would then mix
        under that false identity.
        """
        if "geoemb:model" in root_attrs or not cells_landed:
            return
        raise ValueError(
            f"Store {store_path} has {how_many_groups} and {cells_landed} landed cell(s) but no root "
            f"identity. Refusing to stamp one: this cannot know which encoder or minimum-depth rule "
            f"those cells were filled under, and the stamp is write-once and read by every later "
            f"fill. Verify the existing data's provenance and stamp it deliberately, or fill a "
            f"freshly seeded store."
        )

    todo = [spec for zone_name, spec in ZONES.items() if zone_name not in seeded]
    if not todo:
        # seed_zone_groups is where the write-once root identity is checked, and it is not
        # reached when there is nothing to create — so a rerun asking for a different checkpoint
        # or minimum-depth rule would report a clean seed, leave the old identity standing and
        # let the campaign follow the OLD rule. Checked here too, on the path that skips the seed.
        root_attrs = dict(zarr.open_group(repo.readonly_session(branch="main").store, mode="r").attrs)
        # GLOBAL is what the seed below uses (seed_zone_groups' default), so the identity
        # compared here is the identity that a seed would have written.
        check_root_identity(root_attrs, layout=GLOBAL, model_version=model_version, optical_min_obs=optical_min_obs)
        if "geoemb:model" not in root_attrs:
            # UNSTAMPED, and no group is created here to stamp it as a side effect. Seeding a
            # store predating the root identity would otherwise report success having written
            # neither the checkpoint nor the depth rule asked for — and the fill gates pass on
            # an ABSENT attr (`if seeded is not None`, `optical_min_obs` compared to None), so
            # the store would accept anything while the operator was told it was seeded.
            _refuse_stamping_over_landed_cells(root_attrs, f"all {len(ZONES)} zone groups")
            snapshot = stamp_root_identity(
                repo, layout=GLOBAL, model_version=model_version, optical_min_obs=optical_min_obs
            )
            log.info(
                "Store %s was seeded without a root identity and holds no data — stamped it (%s); "
                "minimum optical depth %s",
                store_path,
                snapshot,
                optical_min_obs if optical_min_obs is not None else "no rule",
            )
        log.info("All %d zone groups already seeded in %s", len(ZONES), store_path)
        return {"store_path": store_path, "seeded_now": 0, "already_seeded": len(seeded), "total": len(ZONES)}

    # BEFORE the incremental seed, because that seed stamps the root identity as a side effect
    # of adding groups: a partially seeded store predating the identity and holding cells would
    # otherwise have them attributed to an identity this call invented.
    if seeded:
        _refuse_stamping_over_landed_cells(
            dict(zarr.open_group(repo.readonly_session(branch="main").store, mode="r").attrs),
            f"{len(seeded)} of {len(ZONES)} zone groups",
        )
    snapshot = seed_zone_groups(
        repo,
        todo,
        years=years,
        model_version=model_version,
        optical_min_obs=optical_min_obs,
        commit_msg=f"seed {len(todo)} zone group(s)",
    )
    log.info(
        "Seeded %d zone group(s) into %s (%s); minimum optical depth %s",
        len(todo),
        store_path,
        snapshot,
        optical_min_obs if optical_min_obs is not None else "no rule",
    )
    return {
        "store_path": store_path,
        "seeded_now": len(todo),
        "already_seeded": len(seeded),
        "total": len(ZONES),
        "snapshot_id": snapshot,
    }
