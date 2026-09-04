"""T4 — 120-group repo metadata scale (T4; settles ADR-008 D5).

Feeds the snapshot/open-time arm of the one-repo go/no-go: seed groups
incrementally and watch (a) snapshot-file growth and constant-write commit cost
vs group count (every commit re-serializes the whole snapshot), and (b)
single-group vs whole-repo open time, default vs tuned manifest preload
(``max_arrays_to_scan`` default 50 < our node count — icechunk #1464/#1462).

Run from ``scripts/``::

    uv run python -m scripts.scoping.scale_tests.t4_group_scale --run-id dev --backend local --scale tiny
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from scripts.scoping.scale_tests import harness
from scripts.scoping.scale_tests import store_builder as SB
from scripts.scoping.scale_tests import variants as V
from scripts.scoping.scale_tests.seeding import embedding_group_spec, seed_groups
from scripts.scoping.scale_tests.zone_geometry import YEARS, MockZone, zone_for

logger = logging.getLogger("scale_tests.t4")

TEST = "t4"
VARIANT = V.VARIANTS["c256_full"]
STORE = "t4_groups"
GROUP_COUNTS_TINY = (4, 12)
GROUP_COUNTS_BENCH = (12, 60, 120)


def _group_zone(cfg: harness.RunConfig) -> MockZone:
    """Zone geometry for the group-scale seed (tall+narrow; coords are the cost)."""
    return zone_for(cfg.scale, full_height=True) if not cfg.is_tiny else MockZone(50_000, 1_024)


def _group_name(i: int) -> str:
    """UTM-zone-like group name."""
    return f"z{i:03d}"


def cold_open_group(payload: dict[str, Any]) -> dict[str, Any]:
    """Cold-open one group (optionally with tuned preload); return open wall."""
    import icechunk
    import zarr

    from scripts.scoping.scale_tests import harness as _h
    from tessera_embeddings.storage import zarr_store

    config = _h.layered_config(
        preload_max_arrays_to_scan=payload.get("preload_arrays"),
        preload_max_total_refs=payload.get("preload_refs"),
    )
    t0 = time.monotonic()
    repo = icechunk.Repository.open(zarr_store._create_storage(payload["store_uri"]), config=config)
    session = repo.readonly_session(branch="main")
    grp = zarr.open_group(session.store, mode="r")[payload["group"]]
    _ = dict(grp.attrs)  # force group metadata resolution
    _ = grp["embeddings"].shape
    return {"open_wall_s": time.monotonic() - t0}


def phase_scale_groups(cfg: harness.RunConfig) -> None:
    """Seed groups up to each checkpoint; measure snapshot size + one-group commit."""
    zone = _group_zone(cfg)
    counts = GROUP_COUNTS_TINY if cfg.is_tiny else GROUP_COUNTS_BENCH
    harness.reset_store(cfg, STORE)
    harness.create_repo(cfg, STORE)

    seeded = 0
    for target in counts:
        repo = harness.open_repo(cfg, STORE)
        batch = {
            _group_name(i): embedding_group_spec(zone, VARIANT, extra_attrs={"crs": f"EPSG:{32601 + i}"})
            for i in range(seeded, target)
        }
        with harness.timer() as seed_t:
            seed_groups(repo, batch, commit_msg=f"seed groups {seeded}..{target}")
        seeded = target

        snap = harness.newest_object_bytes(harness.store_uri(cfg, STORE) + "/snapshots")
        harness.emit_metric(cfg, TEST, "scale_groups", "snapshot_bytes", snap, "bytes", n_groups=target)
        harness.emit_metric(
            cfg, TEST, "scale_groups", "wall_s", seed_t.seconds, "s", n_groups=target, kind="seed_batch"
        )

        # Constant one-chunk fill into the first group: does commit cost grow with
        # group count (snapshot re-serialization)?
        one_chunk = SB.first_n_chunks(zone, VARIANT, 1)
        res = SB.fill_year(cfg, STORE, _group_name(0), VARIANT, zone, len(YEARS) - 1, one_chunk, n_workers=1, seed=0)
        harness.emit_metric(
            cfg, TEST, "scale_groups", "commit_wall_s", res.commit_wall_s, "s", n_groups=target, kind="one_group_fill"
        )
        logger.info("groups=%d: snapshot_bytes=%d one_group_commit=%.3fs", target, snap, res.commit_wall_s)


def phase_open_and_preload(cfg: harness.RunConfig) -> None:
    """Single-group open (default vs tuned preload) and whole-repo datatree open."""
    store_uri = harness.store_uri(cfg, STORE)
    group = _group_name(0)

    for label, arrays, refs in (("default", None, None), ("tuned", 1000, 1_000_000)):
        payload = {"store_uri": store_uri, "group": group, "preload_arrays": arrays, "preload_refs": refs}
        cold = harness.run_cold("scripts.scoping.scale_tests.t4_group_scale.cold_open_group", payload)
        harness.emit_metric(
            cfg, TEST, "open_and_preload", "open_wall_s", cold["open_wall_s"], "s", preload=label, scope="single_group"
        )
        logger.info("single-group open (preload=%s): %.3fs", label, cold["open_wall_s"])

    _measure_datatree(cfg, store_uri)


def _measure_datatree(cfg: harness.RunConfig, store_uri: str) -> None:
    """Time a whole-repo ``open_datatree`` (records failure rather than crashing)."""
    import icechunk
    import xarray as xr

    from tessera_embeddings.storage import zarr_store

    repo = icechunk.Repository.open(zarr_store._create_storage(store_uri), config=harness.layered_config())
    session = repo.readonly_session(branch="main")
    try:
        t0 = time.monotonic()
        xr.open_datatree(session.store, engine="zarr", consolidated=False)
        wall = time.monotonic() - t0
        harness.emit_metric(cfg, TEST, "open_and_preload", "open_wall_s", wall, "s", scope="datatree")
        logger.info("open_datatree over all groups: %.3fs", wall)
    except Exception as exc:
        logger.warning("open_datatree failed (record + continue): %s", exc)


def main() -> int:
    """Parse args and run T4 phases in order."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    cfg = harness.config_from_args(parser.parse_args())
    harness.configure_logging()

    harness.run_phase(cfg, TEST, "scale_groups", lambda: phase_scale_groups(cfg))
    harness.run_phase(cfg, TEST, "open_and_preload", lambda: phase_open_and_preload(cfg))
    logger.info("T4 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
