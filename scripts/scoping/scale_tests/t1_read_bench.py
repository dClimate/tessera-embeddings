"""T1 — retrieval benchmark across chunk/shard variants (T1).

Builds one ~70%-land store per variant (one timestep) and measures the read
workloads that decide the chunk shape (ADR D2) and whether sharding earns its
keep (ADR D3): point-vector, patch, tile, band-subset, bulk-slab, and open time,
cold (fresh process) and warm, swept over ``async.concurrency``.

Run from ``scripts/``::

    uv run python -m scale_tests.t1_read_bench --run-id dev --backend local --scale tiny --variant c256_full
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import zarr

from scripts.scoping.scale_tests import harness
from scripts.scoping.scale_tests import store_builder as SB
from scripts.scoping.scale_tests import variants as V
from scripts.scoping.scale_tests.seeding import embedding_group_spec, seed_groups
from scripts.scoping.scale_tests.zone_geometry import YEARS, MockZone, zone_for

logger = logging.getLogger("scale_tests.t1")

TEST = "t1"
GROUP = "zone"
YEAR_INDEX = len(YEARS) - 1  # 2025 — the single filled timestep
LAND_FRACTION = 0.7
BUILD_SEED = 0

# Concurrency sweep and workload sizing differ by scale to keep tiny fast.
CONCURRENCIES_TINY = (10, 128)
CONCURRENCIES_BENCH = (10, 64, 128)
N_POINTS_TINY = 200
N_POINTS_BENCH = 1000

# (label, y_extent, x_extent, n_bands) — extents clamped to the zone.
WORKLOADS = (
    ("patch", 100, 100, V.BAND),
    ("tile", 1000, 1000, V.BAND),
    ("band_subset", 512, 512, 8),
    ("bulk", 4096, 4096, V.BAND),
)


# ── shared read primitives (identical code for cold + warm) ──────────────────


def _open_group(store_uri: str, group: str, async_concurrency: int) -> zarr.Group:
    """Open a store read-only as a raw zarr group at a given async concurrency."""
    from scripts.scoping.scale_tests._workers import _open_repo

    zarr.config.set({"async.concurrency": async_concurrency})
    session = _open_repo(store_uri).readonly_session(branch="main")
    return zarr.open_group(session.store, mode="r")[group]


def _point_read(group: zarr.Group, year: int, points: list[tuple[int, int]]) -> list[float]:
    """Read every band at each ``(y, x)`` point; return per-point latencies (s)."""
    emb = group["embeddings"]
    latencies: list[float] = []
    for y, x in points:
        t0 = time.monotonic()
        _ = emb[year, y, x, :]
        latencies.append(time.monotonic() - t0)
    return latencies


def _region_read(group: zarr.Group, year: int, y0: int, x0: int, dy: int, dx: int, n_bands: int) -> int:
    """Read a region; return elements read (for throughput)."""
    emb = group["embeddings"]
    block = emb[year, y0 : y0 + dy, x0 : x0 + dx, 0:n_bands]
    return int(block.size)


# ── land-aware sampling ──────────────────────────────────────────────────────


def _variant(name: str) -> V.Variant:
    """Look up a variant by name (workers pass names, not objects)."""
    return V.VARIANTS[name]


def _land_origins(zone_hw: tuple[int, int], variant: V.Variant, seed: int) -> list[tuple[int, int]]:
    """Pixel origins (y0, x0) of the variant's land chunks over the zone."""
    zone = MockZone(zone_hw[0], zone_hw[1])
    _, cy, cx, _ = variant.chunks
    return [(yc * cy, xc * cx) for yc, xc in SB.land_chunks(zone, variant, fraction=LAND_FRACTION, seed=seed)]


def _land_pixels(
    zone_hw: tuple[int, int], variant: V.Variant, seed: int, n: int, sample_seed: int
) -> list[tuple[int, int]]:
    """Sample ``n`` pixels uniformly within land chunks (so reads hit real data)."""
    origins = _land_origins(zone_hw, variant, seed)
    _, cy, cx, _ = variant.chunks
    rng = np.random.default_rng(sample_seed)
    pts: list[tuple[int, int]] = []
    for _ in range(n):
        oy, ox = origins[rng.integers(len(origins))]
        y = min(oy + int(rng.integers(cy)), zone_hw[0] - 1)
        x = min(ox + int(rng.integers(cx)), zone_hw[1] - 1)
        pts.append((y, x))
    return pts


# ── cold entrypoints (called in a fresh process via harness.run_cold) ────────


def cold_open(payload: dict[str, Any]) -> dict[str, Any]:
    """Open the store and return open wall time (cold cache)."""
    t0 = time.monotonic()
    _open_group(payload["store_uri"], payload["group"], payload["async_concurrency"])
    return {"open_wall_s": time.monotonic() - t0}


def cold_point_read(payload: dict[str, Any]) -> dict[str, Any]:
    """Cold point-vector read: p50/p95 latency over sampled land pixels."""
    group = _open_group(payload["store_uri"], payload["group"], payload["async_concurrency"])
    variant = _variant(payload["variant_name"])
    points = _land_pixels(tuple(payload["zone_hw"]), variant, BUILD_SEED, payload["n_points"], payload["sample_seed"])
    lat = _point_read(group, payload["year_index"], points)
    return _point_stats(lat, variant, payload["mean_chunk_bytes"])


def cold_region_read(payload: dict[str, Any]) -> dict[str, Any]:
    """Cold region read (patch/tile/band-subset/bulk): wall + throughput."""
    group = _open_group(payload["store_uri"], payload["group"], payload["async_concurrency"])
    origins = _land_origins(tuple(payload["zone_hw"]), _variant(payload["variant_name"]), BUILD_SEED)
    y0, x0 = origins[0]
    dy = min(payload["dy"], payload["zone_hw"][0] - y0)
    dx = min(payload["dx"], payload["zone_hw"][1] - x0)
    t0 = time.monotonic()
    n_elem = _region_read(group, payload["year_index"], y0, x0, dy, dx, payload["n_bands"])
    wall = time.monotonic() - t0
    return {"wall_s": wall, "n_elem": n_elem, "throughput_mbps": (n_elem / 1e6) / wall if wall > 0 else 0.0}


def _point_stats(latencies: list[float], variant: V.Variant, mean_chunk_bytes: float) -> dict[str, Any]:
    """Summarize point-read latencies + analytic bytes fetched."""
    arr = np.array(latencies) * 1e3  # ms
    chunks_touched = V.band_chunks(variant)  # one pixel column spans all band chunks of 1 spatial chunk
    return {
        "read_p50_ms": float(np.percentile(arr, 50)),
        "read_p95_ms": float(np.percentile(arr, 95)),
        "chunks_touched": chunks_touched,
        "bytes_fetched": chunks_touched * mean_chunk_bytes,
        "n_points": len(latencies),
    }


# ── orchestration ────────────────────────────────────────────────────────────


def _chunk_mean_bytes(cfg: harness.RunConfig, store: str) -> float:
    """Mean on-disk bytes per chunk object (for analytic ``bytes_fetched``)."""
    uri = harness.store_uri(cfg, store)
    fs, path = harness.fs_and_path(uri)
    if not fs.exists(path):
        return 0.0
    entries = fs.find(path, detail=True)
    chunk_sizes = [
        int(m.get("size") or 0) for p, m in entries.items() if "/chunks/" in p or p.rstrip("/").endswith("chunks")
    ]
    return float(np.mean(chunk_sizes)) if chunk_sizes else 0.0


def build_variant(cfg: harness.RunConfig, variant: V.Variant) -> None:
    """Seed a store for ``variant`` and fill its single (2025) timestep."""
    store = f"t1_{variant.name}"
    zone = zone_for(cfg.scale)
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store)
    seed_groups(
        repo,
        {GROUP: embedding_group_spec(zone, variant, years=(YEARS[YEAR_INDEX],))},
        commit_msg=f"seed {variant.name}",
    )
    land = SB.land_chunks(zone, variant, fraction=LAND_FRACTION, seed=BUILD_SEED)
    n_workers = 2 if cfg.is_tiny else 8
    # The seeded store has a single-year axis, so fill year index 0 here.
    SB.fill_year(cfg, store, GROUP, variant, zone, 0, land, n_workers=n_workers, seed=BUILD_SEED)


def read_variant(cfg: harness.RunConfig, variant: V.Variant) -> None:
    """Run all read workloads for ``variant`` across the concurrency sweep."""
    store = f"t1_{variant.name}"
    store_uri = harness.store_uri(cfg, store)
    zone = zone_for(cfg.scale)
    zone_hw = [zone.height, zone.width]
    mean_bytes = _chunk_mean_bytes(cfg, store)
    concurrencies = CONCURRENCIES_TINY if cfg.is_tiny else CONCURRENCIES_BENCH
    n_points = N_POINTS_TINY if cfg.is_tiny else N_POINTS_BENCH
    # T1 stores are filled at year index 0 (single-year axis).
    year = 0
    phase = f"read_{variant.name}"  # matches the run_phase name so resume clears cleanly

    for conc in concurrencies:
        base = {
            "store_uri": store_uri,
            "group": GROUP,
            "year_index": year,
            "async_concurrency": conc,
            "variant_name": variant.name,
            "zone_hw": zone_hw,
        }

        # open (cold + warm)
        otags = {"variant": variant.name, "concurrency": conc}
        cold = harness.run_cold("scale_tests.t1_read_bench.cold_open", base)
        harness.emit_metric(cfg, TEST, phase, "open_wall_s", cold["open_wall_s"], "s", cache="cold", **otags)
        wt0 = time.monotonic()
        _open_group(store_uri, GROUP, conc)
        harness.emit_metric(cfg, TEST, phase, "open_wall_s", time.monotonic() - wt0, "s", cache="warm", **otags)

        # point-vector (cold + warm)
        pp = {**base, "n_points": n_points, "sample_seed": 1, "mean_chunk_bytes": mean_bytes}
        cold = harness.run_cold("scale_tests.t1_read_bench.cold_point_read", pp)
        _emit_point(cfg, phase, variant, conc, "cold", cold)
        warm = _warm_point_read(store_uri, variant, year, zone_hw, n_points, mean_bytes, conc)
        _emit_point(cfg, phase, variant, conc, "warm", warm)

        # region workloads (cold + warm)
        for label, dy, dx, n_bands in WORKLOADS:
            rp = {**base, "dy": dy, "dx": dx, "n_bands": n_bands}
            cold = harness.run_cold("scale_tests.t1_read_bench.cold_region_read", rp)
            _emit_region(cfg, phase, variant, conc, "cold", label, cold)
            warm = _warm_region_read(store_uri, variant, year, zone_hw, dy, dx, n_bands, conc)
            _emit_region(cfg, phase, variant, conc, "warm", label, warm)


def _warm_point_read(
    store_uri: str, variant: V.Variant, year: int, zone_hw: list[int], n_points: int, mean_bytes: float, conc: int
) -> dict:
    """Open once, read points twice, return the (warm) second measurement."""
    group = _open_group(store_uri, GROUP, conc)
    points = _land_pixels(tuple(zone_hw), variant, BUILD_SEED, n_points, 1)
    _point_read(group, year, points)  # warm the cache
    lat = _point_read(group, year, points)
    return _point_stats(lat, variant, mean_bytes)


def _warm_region_read(
    store_uri: str, variant: V.Variant, year: int, zone_hw: list[int], dy: int, dx: int, n_bands: int, conc: int
) -> dict:
    """Open once, read the region twice, return the (warm) second measurement."""
    group = _open_group(store_uri, GROUP, conc)
    origins = _land_origins(tuple(zone_hw), variant, BUILD_SEED)
    y0, x0 = origins[0]
    dy = min(dy, zone_hw[0] - y0)
    dx = min(dx, zone_hw[1] - x0)
    _region_read(group, year, y0, x0, dy, dx, n_bands)
    t0 = time.monotonic()
    n_elem = _region_read(group, year, y0, x0, dy, dx, n_bands)
    wall = time.monotonic() - t0
    return {"wall_s": wall, "n_elem": n_elem, "throughput_mbps": (n_elem / 1e6) / wall if wall > 0 else 0.0}


def _emit_point(cfg: harness.RunConfig, phase: str, variant: V.Variant, conc: int, cache: str, r: dict) -> None:
    """Emit point-read metrics under the variant's read phase."""
    tags = {"variant": variant.name, "cache": cache, "concurrency": conc}
    harness.emit_metric(cfg, TEST, phase, "read_p50_ms", r["read_p50_ms"], "ms", **tags)
    harness.emit_metric(cfg, TEST, phase, "read_p95_ms", r["read_p95_ms"], "ms", **tags)
    harness.emit_metric(cfg, TEST, phase, "bytes_fetched", r["bytes_fetched"], "bytes", workload="point", **tags)


def _emit_region(
    cfg: harness.RunConfig, phase: str, variant: V.Variant, conc: int, cache: str, label: str, r: dict
) -> None:
    """Emit region-read metrics under the variant's read phase."""
    tags = {"variant": variant.name, "cache": cache, "concurrency": conc, "workload": label}
    harness.emit_metric(cfg, TEST, phase, "throughput_mbps", r["throughput_mbps"], "MB/s", **tags)


def main() -> int:
    """Parse args and run T1 build + read phases for each selected variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    cfg = harness.config_from_args(parser.parse_args())
    harness.configure_logging()

    for variant in V.selected(cfg.variant):
        harness.run_phase(
            cfg, TEST, f"build_{variant.name}", lambda v=variant: build_variant(cfg, v), variant=variant.name
        )
        harness.run_phase(
            cfg, TEST, f"read_{variant.name}", lambda v=variant: read_variant(cfg, v), variant=variant.name
        )
    logger.info("T1 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
