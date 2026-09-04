"""T6 — GC, expiration, rollback (T6; settles ADR-008 D7).

Builds a realistic mess (multi-commit history + superseded chunks from an
overwrite), then exercises the snapshot-hygiene runbook: tag -> expire -> GC
(dry-run then real), recording objects/s (the LIST-bound cost that extrapolates
to the campaign's 10^8+ objects) and bytes reclaimed, plus a ``reset_branch``
rollback drill.

Self-contained (builds its own mess) so it runs independently; on a full
campaign run it would instead target the store T2-T5 left behind. Run from
``scripts/``::

    uv run python -m scale_tests.t6_gc_bench --run-id dev --backend local --scale tiny
"""

from __future__ import annotations

import argparse
import datetime
import logging
import time

import numpy as np
import zarr

from scripts.scoping.scale_tests import harness
from scripts.scoping.scale_tests import store_builder as SB
from scripts.scoping.scale_tests import variants as V
from scripts.scoping.scale_tests.seeding import embedding_group_spec, seed_groups
from scripts.scoping.scale_tests.zone_geometry import YEARS, MockZone

logger = logging.getLogger("scale_tests.t6")

TEST = "t6"
GROUP = "zone"
VARIANT = V.VARIANTS["c256_full"]
ZONE = MockZone(1_024, 1_024)
STORE = "t6_gc"
SEED = 0


def phase_build_mess(cfg: harness.RunConfig) -> None:
    """Fill 3 years then overwrite one — leaving superseded chunks as garbage."""
    harness.reset_store(cfg, STORE)
    repo = harness.create_repo(cfg, STORE)
    years = YEARS[-3:]  # 2023, 2024, 2025
    seed_groups(repo, {GROUP: embedding_group_spec(ZONE, VARIANT, years=years)}, commit_msg="seed 3 years")
    land = SB.land_chunks(ZONE, VARIANT, fraction=0.7, seed=SEED)
    for yi in range(len(years)):
        SB.fill_year(cfg, STORE, GROUP, VARIANT, ZONE, yi, land, n_workers=2, seed=SEED)

    # Overwrite year 0's chunks with new data -> the original chunk objects become
    # unreferenced once this commit's snapshot supersedes the prior one.
    _, cy, cx, _ = VARIANT.chunks
    session = harness.open_repo(cfg, STORE).writable_session("main")
    emb = zarr.open_group(session.store, mode="a")[GROUP]["embeddings"]
    for yc, xc in land:
        y0, x0 = yc * cy, xc * cx
        y1, x1 = min(y0 + cy, ZONE.height), min(x0 + cx, ZONE.width)
        emb[0:1, y0:y1, x0:x1, :] = np.full((1, y1 - y0, x1 - x0, V.BAND), 9, "int8")
    session.commit("overwrite year 0 (creates garbage)")

    count, total = harness.object_stats(harness.store_uri(cfg, STORE))
    harness.emit_metric(cfg, TEST, "build_mess", "objects_listed", count, "count", when="after_build")
    logger.info("built mess: %d objects, %d bytes across history", count, total)


def phase_expire_and_gc(cfg: harness.RunConfig) -> None:
    """Tag the tip, expire old snapshots, then GC (dry-run then real)."""
    repo = harness.open_repo(cfg, STORE)
    tip = repo.lookup_branch("main")
    repo.create_tag("keep-tip", tip)

    before_count, _ = harness.object_stats(harness.store_uri(cfg, STORE))
    harness.emit_metric(cfg, TEST, "expire_and_gc", "objects_listed", before_count, "count", when="before_gc")

    cutoff = datetime.datetime.now(datetime.UTC)
    expired = repo.expire_snapshots(older_than=cutoff)
    logger.info("expired %d snapshots (tagged tip protected)", len(expired))
    if "keep-tip" not in repo.list_tags():
        raise SystemExit("tag 'keep-tip' vanished after expiry — tags must protect snapshots")

    dry = repo.garbage_collect(cutoff, dry_run=True)
    logger.info(
        "GC dry-run would delete: chunks=%d manifests=%d snapshots=%d bytes=%d",
        dry.chunks_deleted,
        dry.manifests_deleted,
        dry.snapshots_deleted,
        dry.bytes_deleted,
    )

    t0 = time.monotonic()
    summary = repo.garbage_collect(cutoff, dry_run=False)
    gc_wall = time.monotonic() - t0

    after_count, _ = harness.object_stats(harness.store_uri(cfg, STORE))
    deleted = (
        summary.chunks_deleted
        + summary.manifests_deleted
        + summary.snapshots_deleted
        + summary.transaction_logs_deleted
    )
    objs_per_s = deleted / gc_wall if gc_wall > 0 else 0.0
    harness.emit_metric(cfg, TEST, "expire_and_gc", "gc_wall_s", gc_wall, "s")
    harness.emit_metric(cfg, TEST, "expire_and_gc", "objects_deleted", deleted, "count", per_s=round(objs_per_s, 1))
    harness.emit_metric(cfg, TEST, "expire_and_gc", "bytes_reclaimed", summary.bytes_deleted, "bytes")
    harness.emit_metric(cfg, TEST, "expire_and_gc", "objects_listed", after_count, "count", when="after_gc")
    logger.info(
        "GC real: deleted %d objects (%.1f/s), reclaimed %d bytes; store %d->%d objects",
        deleted,
        objs_per_s,
        summary.bytes_deleted,
        before_count,
        after_count,
    )


def phase_rollback(cfg: harness.RunConfig) -> None:
    """reset_branch back one commit, read at the old snapshot, then re-write."""
    repo = harness.open_repo(cfg, STORE)
    history = list(repo.ancestry(branch="main"))
    if len(history) < 2:
        logger.warning("history too short to roll back (%d snapshots) — skipping", len(history))
        return
    current, target = history[0], history[1]

    repo.reset_branch("main", target.id, from_snapshot_id=current.id)

    # Old snapshot stays readable by id after the branch moves — read it
    # post-move, or the drill only proves it was readable as the branch tip.
    old_session = repo.readonly_session(snapshot_id=current.id)
    _ = zarr.open_group(old_session.store, mode="r")[GROUP]["embeddings"].shape
    if repo.lookup_branch("main") != target.id:
        raise SystemExit("reset_branch did not move the branch tip")

    # Re-write onto the rolled-back tip (a fresh commit).
    session = repo.writable_session("main")
    _, cy, cx, _ = VARIANT.chunks
    zarr.open_group(session.store, mode="a")[GROUP]["embeddings"][0:1, 0:cy, 0:cx, :] = np.full(
        (1, cy, cx, V.BAND), 1, "int8"
    )
    session.commit("re-write after rollback")
    harness.emit_metric(cfg, TEST, "rollback", "retries", 0.0, "count", outcome="rolled_back_and_rewrote")
    logger.info("rollback drill ok: %s -> %s -> new commit", current.id[:8], target.id[:8])


def main() -> int:
    """Parse args and run T6 phases in order."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    cfg = harness.config_from_args(parser.parse_args())
    harness.configure_logging()

    harness.run_phase(cfg, TEST, "build_mess", lambda: phase_build_mess(cfg))
    # Rollback drill first, on full history — expiry below collapses intermediate
    # snapshots, so a post-expiry rollback would only reach the empty root.
    harness.run_phase(cfg, TEST, "rollback", lambda: phase_rollback(cfg))
    harness.run_phase(cfg, TEST, "expire_and_gc", lambda: phase_expire_and_gc(cfg))
    logger.info("T6 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
