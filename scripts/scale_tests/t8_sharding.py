"""T8 — settle D3: shard or not shard (see design/d3-sharding-plan.md).

Run 1's T1 sweep showed sharded reads tie/beat unsharded and cut object count
~64x, at a 2.8x write cost — but that cost came from unaligned (chunk-by-chunk)
writes into 2048² shards. T8 runs the four experiments the D3 decision rule
hinges on:

* **E1 `bytes_on_wire`** — real bytes off the NIC per point read (is the 16.5 MB
  analytic figure an artifact? does sharding do partial reads?).
* **E2 `write_alignment`** — build wall for full vs sharded-chunkwise vs
  sharded-**aligned**; does aligning writes shrink the penalty toward 1x?
* **E3 `scattered_reads`** — cold point p50/p95 under a shard-scattered access
  pattern (does the p50 win survive when points don't share shards?).
* **E4 `object_count`** — actual objects + manifest bytes, sharded vs full.

Run from ``scripts/`` (bench, in-region)::

    uv run python -m scale_tests.t8_sharding --run-id d3 --backend s3 --scale bench \
      --bucket arbol-tessera-embeddings-dev --store-root s3://arbol-tessera-embeddings-dev/global-embeddings
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

from scale_tests import harness
from scale_tests import store_builder as SB
from scale_tests import variants as V
from scale_tests.seeding import embedding_group_spec, seed_groups
from scale_tests.t1_read_bench import _open_group, _point_read
from scale_tests.zone_geometry import YEARS, MockZone

logger = logging.getLogger("scale_tests.t8")

TEST = "t8"
GROUP = "zone"
FULL = V.VARIANTS["c256_full"]
SHARDED = V.VARIANTS["c256_sharded"]
LAND_FRACTION = 0.7
SEED = 0
YEAR = 0  # stores are seeded with a single-year axis

STORE_FULL = "t8_full"
STORE_SHARDED = "t8_sharded"  # chunkwise (matches T1)
STORE_ALIGNED = "t8_sharded_aligned"

N_POINTS_TINY = 100
N_POINTS_BENCH = 500


def _d3_zone(cfg: harness.RunConfig) -> MockZone:
    """A few-shard zone: cheap dense shard writes, still multi-shard for reads.

    tiny -> sub-shard (fast laptop smoke); bench -> 3x3 of the 2048² shards.
    """
    return MockZone(512, 512) if cfg.is_tiny else MockZone(6144, 6144)


def _n_workers(cfg: harness.RunConfig) -> int:
    """Worker count for fills (small — this test is about mechanics, not scale)."""
    return 2 if cfg.is_tiny else 8


# ── E2 — write alignment (also builds the stores E1/E3/E4 read) ───────────────


def _seed(cfg: harness.RunConfig, store: str, variant: V.Variant, zone: MockZone) -> None:
    """Reset and seed a single-year store for ``variant``."""
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store)
    seed_groups(
        repo, {GROUP: embedding_group_spec(zone, variant, years=(YEARS[-1],))}, commit_msg=f"seed {variant.name}"
    )


def _chunk_bytes(cfg: harness.RunConfig, store: str) -> tuple[int, int]:
    """(object count, total bytes) under the store's chunk data."""
    return harness.object_stats(harness.store_uri(cfg, store) + "/chunks")


def phase_write_alignment(cfg: harness.RunConfig) -> None:
    """Build full / sharded-chunkwise / sharded-aligned; compare build + bytes."""
    zone = _d3_zone(cfg)
    nw = _n_workers(cfg)
    land = SB.land_chunks(zone, SHARDED, fraction=LAND_FRACTION, seed=SEED)

    # (a) unsharded reference
    _seed(cfg, STORE_FULL, FULL, zone)
    with harness.timer() as t:
        res = SB.fill_year(
            cfg,
            STORE_FULL,
            GROUP,
            FULL,
            zone,
            YEAR,
            SB.land_chunks(zone, FULL, fraction=LAND_FRACTION, seed=SEED),
            n_workers=nw,
            seed=SEED,
        )
    _emit_build(cfg, "full", t.seconds, res.commit_wall_s, _chunk_bytes(cfg, STORE_FULL))

    # (b) sharded, chunkwise (the T1 path)
    _seed(cfg, STORE_SHARDED, SHARDED, zone)
    with harness.timer() as t:
        res = SB.fill_year(cfg, STORE_SHARDED, GROUP, SHARDED, zone, YEAR, land, n_workers=nw, seed=SEED)
    _emit_build(cfg, "sharded_chunkwise", t.seconds, res.commit_wall_s, _chunk_bytes(cfg, STORE_SHARDED))

    # (c) sharded, shard-aligned (one full-shard write per shard)
    _seed(cfg, STORE_ALIGNED, SHARDED, zone)
    shards = SB.shards_for_chunks(SHARDED, land)
    with harness.timer() as t:
        res = SB.fill_year_shard_aligned(
            cfg, STORE_ALIGNED, GROUP, SHARDED, zone, YEAR, shards, n_workers=nw, seed=SEED
        )
    _emit_build(cfg, "sharded_aligned", t.seconds, res.commit_wall_s, _chunk_bytes(cfg, STORE_ALIGNED))

    logger.info("write_alignment built full / sharded-chunkwise / sharded-aligned (%d shards)", len(shards))


def _emit_build(
    cfg: harness.RunConfig, mode: str, build_wall: float, commit_wall: float, chunks: tuple[int, int]
) -> None:
    """Emit build wall, commit wall, chunk-object count, and bytes written."""
    n_obj, n_bytes = chunks
    harness.emit_metric(cfg, TEST, "write_alignment", "wall_s", build_wall, "s", mode=mode, kind="build")
    harness.emit_metric(cfg, TEST, "write_alignment", "commit_wall_s", commit_wall, "s", mode=mode)
    harness.emit_metric(cfg, TEST, "write_alignment", "bytes_written", n_bytes, "bytes", mode=mode)
    harness.emit_metric(cfg, TEST, "write_alignment", "objects_listed", n_obj, "count", mode=mode, where="chunks")


# ── E1 — bytes on the wire (cold subprocess) ─────────────────────────────────


def cold_bytes_on_wire(payload: dict[str, Any]) -> dict[str, Any]:
    """Measure real bytes received per point read (NIC delta / N) + latencies."""
    import psutil

    group = _open_group(payload["store_uri"], payload["group"], payload["async_concurrency"])
    variant = V.VARIANTS[payload["variant_name"]]
    points = _sample_points(
        tuple(payload["zone_hw"]), variant, payload["n_points"], payload["sample_seed"], scattered=False
    )
    net0 = psutil.net_io_counters().bytes_recv
    lat = _point_read(group, payload["year_index"], points)
    net1 = psutil.net_io_counters().bytes_recv
    arr = np.array(lat) * 1e3
    return {
        "bytes_per_point": (net1 - net0) / max(1, len(points)),
        "read_p50_ms": float(np.percentile(arr, 50)),
        "read_p95_ms": float(np.percentile(arr, 95)),
        "n_points": len(points),
    }


def phase_bytes_on_wire(cfg: harness.RunConfig) -> None:
    """E1: real bytes/point for full vs sharded (resolves the analytic overestimate)."""
    zone = _d3_zone(cfg)
    n = N_POINTS_TINY if cfg.is_tiny else N_POINTS_BENCH
    for variant, store in ((FULL, STORE_FULL), (SHARDED, STORE_SHARDED)):
        payload = {
            "store_uri": harness.store_uri(cfg, store),
            "group": GROUP,
            "year_index": YEAR,
            "async_concurrency": 128,
            "variant_name": variant.name,
            "zone_hw": [zone.height, zone.width],
            "n_points": n,
            "sample_seed": 1,
        }
        r = harness.run_cold("scale_tests.t8_sharding.cold_bytes_on_wire", payload)
        harness.emit_metric(
            cfg,
            TEST,
            "bytes_on_wire",
            "bytes_fetched",
            r["bytes_per_point"],
            "bytes",
            variant=variant.name,
            method="wire",
        )
        harness.emit_metric(
            cfg, TEST, "bytes_on_wire", "read_p95_ms", r["read_p95_ms"], "ms", variant=variant.name, method="wire"
        )
        logger.info(
            "E1 %s: %.2f MB/point on the wire, p95 %.1f ms", variant.name, r["bytes_per_point"] / 1e6, r["read_p95_ms"]
        )


# ── E3 — scattered reads ─────────────────────────────────────────────────────


def cold_scattered_read(payload: dict[str, Any]) -> dict[str, Any]:
    """Cold point reads under a shard-scattered pattern; return p50/p95."""
    group = _open_group(payload["store_uri"], payload["group"], payload["async_concurrency"])
    variant = V.VARIANTS[payload["variant_name"]]
    points = _sample_points(
        tuple(payload["zone_hw"]), variant, payload["n_points"], payload["sample_seed"], scattered=True
    )
    arr = np.array(_point_read(group, payload["year_index"], points)) * 1e3
    return {"read_p50_ms": float(np.percentile(arr, 50)), "read_p95_ms": float(np.percentile(arr, 95))}


def phase_scattered_reads(cfg: harness.RunConfig) -> None:
    """E3: does sharded's p50 win survive a cache-hostile (scattered) pattern?"""
    zone = _d3_zone(cfg)
    n = N_POINTS_TINY if cfg.is_tiny else N_POINTS_BENCH
    for variant, store in ((FULL, STORE_FULL), (SHARDED, STORE_SHARDED)):
        payload = {
            "store_uri": harness.store_uri(cfg, store),
            "group": GROUP,
            "year_index": YEAR,
            "async_concurrency": 128,
            "variant_name": variant.name,
            "zone_hw": [zone.height, zone.width],
            "n_points": n,
            "sample_seed": 7,
        }
        r = harness.run_cold("scale_tests.t8_sharding.cold_scattered_read", payload)
        harness.emit_metric(
            cfg,
            TEST,
            "scattered_reads",
            "read_p50_ms",
            r["read_p50_ms"],
            "ms",
            variant=variant.name,
            pattern="scattered",
        )
        harness.emit_metric(
            cfg,
            TEST,
            "scattered_reads",
            "read_p95_ms",
            r["read_p95_ms"],
            "ms",
            variant=variant.name,
            pattern="scattered",
        )
        logger.info("E3 %s scattered: p50 %.1f ms, p95 %.1f ms", variant.name, r["read_p50_ms"], r["read_p95_ms"])


# ── E4 — object + manifest count ─────────────────────────────────────────────


def phase_object_count(cfg: harness.RunConfig) -> None:
    """E4: actual objects + manifest bytes, full vs sharded (chunkwise + aligned)."""
    for mode, store in (("full", STORE_FULL), ("sharded_chunkwise", STORE_SHARDED), ("sharded_aligned", STORE_ALIGNED)):
        n_obj, _ = _chunk_bytes(cfg, store)
        _, man_bytes = harness.object_stats(harness.store_uri(cfg, store) + "/manifests")
        harness.emit_metric(cfg, TEST, "object_count", "objects_listed", n_obj, "count", mode=mode, where="chunks")
        harness.emit_metric(cfg, TEST, "object_count", "manifest_bytes", man_bytes, "bytes", mode=mode)
        logger.info("E4 %s: %d chunk objects, %d manifest bytes", mode, n_obj, man_bytes)


# ── point sampling ───────────────────────────────────────────────────────────


def _sample_points(
    zone_hw: tuple[int, int], variant: V.Variant, n: int, seed: int, *, scattered: bool
) -> list[tuple[int, int]]:
    """Sample ``n`` pixels in written land chunks.

    ``scattered=False`` mirrors T1 (random land chunk per point — points cluster
    into shared shards). ``scattered=True`` spreads points across *distinct*
    shards first (cache-hostile), cycling once every shard is used.
    """
    zone = MockZone(zone_hw[0], zone_hw[1])
    _, cy, cx, _ = variant.chunks
    land = SB.land_chunks(zone, variant, fraction=LAND_FRACTION, seed=SEED)
    rng = np.random.default_rng(seed)
    if scattered and variant.shards is not None:
        _, sh_y, sh_x, _ = variant.shards
        by_shard: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for yc, xc in land:
            by_shard.setdefault((yc * cy // sh_y, xc * cx // sh_x), []).append((yc, xc))
        shard_keys = sorted(by_shard)
        chunks = [by_shard[shard_keys[i % len(shard_keys)]][0] for i in range(n)]
    else:
        chunks = [land[int(rng.integers(len(land)))] for _ in range(n)]
    pts: list[tuple[int, int]] = []
    for yc, xc in chunks:
        y = min(yc * cy + int(rng.integers(cy)), zone_hw[0] - 1)
        x = min(xc * cx + int(rng.integers(cx)), zone_hw[1] - 1)
        pts.append((y, x))
    return pts


def main() -> int:
    """Parse args and run E1-E4 (write_alignment builds the stores first)."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    cfg = harness.config_from_args(parser.parse_args())
    harness.configure_logging()

    harness.run_phase(cfg, TEST, "write_alignment", lambda: phase_write_alignment(cfg))
    harness.run_phase(cfg, TEST, "bytes_on_wire", lambda: phase_bytes_on_wire(cfg))
    harness.run_phase(cfg, TEST, "scattered_reads", lambda: phase_scattered_reads(cfg))
    harness.run_phase(cfg, TEST, "object_count", lambda: phase_object_count(cfg))
    logger.info("T8 complete — feed report.py, then apply the D3 decision rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
