"""T2 — write path, commit scaling, manifest behaviour (T2).

Feeds ADR D4 (split config) and D6 (batch size, refs/commit ceiling), and probes
the open upstream bugs directly: commit-time RSS vs refs (~400 B/ref; icechunk
#1558 panics ~7e7 — this stays <=2e7), and whether per-year commit time stays
bounded by the touched manifest window or grows with cumulative refs (#1600).

Conceptually consumes T3's pre-allocated store; kept independently runnable by
seeding its own. Run from ``scripts/``::

    uv run python -m scripts.scoping.scale_tests.t2_write_bench --run-id dev --backend local --scale tiny
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import zarr

from scripts.scoping.scale_tests import harness
from scripts.scoping.scale_tests import store_builder as SB
from scripts.scoping.scale_tests import variants as V
from scripts.scoping.scale_tests.seeding import embedding_group_spec, seed_groups
from scripts.scoping.scale_tests.zone_geometry import YEARS, zone_for

logger = logging.getLogger("scale_tests.t2")

TEST = "t2"
GROUP = "zone"
VARIANT = V.VARIANTS["c256_full"]
LAND_FRACTION = 0.7
SEED = 0

# Chunk counts per fill for the refs sweep (x refs_per_spatial ~ refs/commit).
# 1e6+ refs needs a larger zone than the bench mock; bump the zone on the server
# to push higher — the sweep logs when a target is capped by available chunks.
SWEEP_CHUNKS_TINY = (10, 50, 150)
SWEEP_CHUNKS_BENCH = (500, 5000, 20000)
# Hard guard: never approach the ~7e7-ref single-commit panic (#1558).
MAX_REFS_PER_COMMIT = 20_000_000

SPLIT_CONFIGS = {
    "none": None,
    "time1": {"time": 1},
    "time1_spatial": {"time": 1, "northing": 8, "easting": 8},
}


def _config_for(split: dict[str, int] | None):  # noqa: ANN202 — icechunk.RepositoryConfig
    """Layered repo config for a split spec (or the plain default)."""
    return harness.layered_config(split_sizes=split) if split else harness.layered_config()


def _manifest_stats(cfg: harness.RunConfig, store: str) -> tuple[int, int]:
    """(object count, total bytes) under the store's ``manifests/`` prefix."""
    return harness.object_stats(harness.store_uri(cfg, store) + "/manifests")


def _snapshot_bytes(cfg: harness.RunConfig, store: str) -> int:
    """Bytes of the newest object under the store's ``snapshots/`` prefix."""
    return harness.newest_object_bytes(harness.store_uri(cfg, store) + "/snapshots")


def phase_refs_sweep(cfg: harness.RunConfig) -> None:
    """Fill increasing chunk counts in one commit; measure commit wall + RSS."""
    zone = zone_for(cfg.scale)
    counts = SWEEP_CHUNKS_TINY if cfg.is_tiny else SWEEP_CHUNKS_BENCH
    n_workers = 2 if cfg.is_tiny else 8
    for n_chunks in counts:
        refs_est = n_chunks * SB.refs_per_spatial_chunk(VARIANT)
        if refs_est > MAX_REFS_PER_COMMIT:
            logger.warning("skipping %d chunks (~%d refs) > MAX_REFS_PER_COMMIT", n_chunks, refs_est)
            continue
        store = f"t2_sweep_{n_chunks}"
        harness.reset_store(cfg, store)
        repo = harness.create_repo(cfg, store)
        seed_groups(repo, {GROUP: embedding_group_spec(zone, VARIANT, years=(YEARS[-1],))}, commit_msg="seed")
        chunk_list = SB.first_n_chunks(zone, VARIANT, n_chunks)
        with harness.rss_sampler(cfg, TEST, "refs_sweep", variant=VARIANT.name):
            res = SB.fill_year(cfg, store, GROUP, VARIANT, zone, 0, chunk_list, n_workers=n_workers, seed=SEED)
        tags = {"variant": VARIANT.name, "n_chunks": res.n_chunks, "refs": res.refs_committed}
        harness.emit_metric(cfg, TEST, "refs_sweep", "refs_committed", res.refs_committed, "count", **tags)
        harness.emit_metric(cfg, TEST, "refs_sweep", "commit_wall_s", res.commit_wall_s, "s", **tags)
        harness.emit_metric(cfg, TEST, "refs_sweep", "merge_wall_s", res.merge_wall_s, "s", **tags)


def phase_year_fill_trend(cfg: harness.RunConfig) -> None:
    """Fill 2025->2017 sequentially (split time@1); watch per-year commit cost.

    Flat per-year commit time confirms bounded rewrite; a rising trend reproduces
    icechunk #1600 (append cost growing with cumulative refs despite splitting).
    """
    store = "t2_trend"
    zone = zone_for(cfg.scale)
    config = _config_for({"time": 1})
    land = SB.land_chunks(zone, VARIANT, fraction=LAND_FRACTION, seed=SEED)
    n_workers = 2 if cfg.is_tiny else 8
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store, config=config)
    seed_groups(repo, {GROUP: embedding_group_spec(zone, VARIANT)}, commit_msg="seed full axis")

    for year_idx in reversed(range(len(YEARS))):  # 2025 -> 2017
        res = SB.fill_year(
            cfg, store, GROUP, VARIANT, zone, year_idx, land, n_workers=n_workers, seed=SEED, repo_config=config
        )
        man_count, man_bytes = _manifest_stats(cfg, store)
        tags = {"year": YEARS[year_idx], "refs": res.refs_committed}
        harness.emit_metric(cfg, TEST, "year_fill_trend", "commit_wall_s", res.commit_wall_s, "s", **tags)
        harness.emit_metric(cfg, TEST, "year_fill_trend", "manifest_bytes", man_bytes, "bytes", **tags)
        harness.emit_metric(cfg, TEST, "year_fill_trend", "manifest_count", man_count, "count", **tags)
        harness.emit_metric(
            cfg, TEST, "year_fill_trend", "snapshot_bytes", _snapshot_bytes(cfg, store), "bytes", **tags
        )


def phase_split_config_ab(cfg: harness.RunConfig) -> None:
    """Compare no-split / time@1 / time@1+spatial on fill + patch-write cost."""
    zone = zone_for(cfg.scale)
    land = SB.land_chunks(zone, VARIANT, fraction=LAND_FRACTION, seed=SEED)
    n_workers = 2 if cfg.is_tiny else 8
    _, cy, cx, _ = VARIANT.chunks

    for name, split in SPLIT_CONFIGS.items():
        store = f"t2_split_{name}"
        config = _config_for(split)
        harness.reset_store(cfg, store)
        repo = harness.create_repo(cfg, store, config=config)
        seed_groups(repo, {GROUP: embedding_group_spec(zone, VARIANT)}, commit_msg="seed")
        # Fill two years so a manifest spans more than one timestep.
        for year_idx in (len(YEARS) - 1, len(YEARS) - 2):
            res = SB.fill_year(
                cfg, store, GROUP, VARIANT, zone, year_idx, land, n_workers=n_workers, seed=SEED, repo_config=config
            )
            harness.emit_metric(
                cfg,
                TEST,
                "split_config_ab",
                "commit_wall_s",
                res.commit_wall_s,
                "s",
                split=name,
                kind="fill",
                year=YEARS[year_idx],
            )
        man_count, _ = _manifest_stats(cfg, store)
        harness.emit_metric(cfg, TEST, "split_config_ab", "manifest_count", man_count, "count", split=name)
        harness.emit_metric(
            cfg, TEST, "split_config_ab", "snapshot_bytes", _snapshot_bytes(cfg, store), "bytes", split=name
        )

        # Patch-write cost: overwrite one chunk in an already-filled year.
        yc, xc = land[0]
        y0, x0 = yc * cy, xc * cx
        # Clamp to the chunk's real extent (bench zone is not a 256 multiple).
        ch, cw = min(cy, zone.height - y0), min(cx, zone.width - x0)
        session = harness.open_repo(cfg, store, config=config).writable_session("main")
        grp = zarr.open_group(session.store, mode="a")[GROUP]
        grp["embeddings"][len(YEARS) - 1 : len(YEARS), y0 : y0 + ch, x0 : x0 + cw, :] = np.full(
            (1, ch, cw, V.BAND), 5, "int8"
        )
        with harness.timer() as t:
            session.commit("patch one chunk")
        harness.emit_metric(cfg, TEST, "split_config_ab", "commit_wall_s", t.seconds, "s", split=name, kind="patch")
        logger.info("split %s: manifest_count=%d patch_commit=%.3fs", name, man_count, t.seconds)


def main() -> int:
    """Parse args and run T2 phases in order."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    cfg = harness.config_from_args(parser.parse_args())
    harness.configure_logging()

    harness.run_phase(cfg, TEST, "refs_sweep", lambda: phase_refs_sweep(cfg))
    harness.run_phase(cfg, TEST, "year_fill_trend", lambda: phase_year_fill_trend(cfg))
    harness.run_phase(cfg, TEST, "split_config_ab", lambda: phase_split_config_ab(cfg))
    logger.info("T2 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
