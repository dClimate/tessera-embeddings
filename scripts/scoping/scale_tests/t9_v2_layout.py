"""T9 — v2 store layout: band-16 inner chunks, int8 codecs, shard orders, measured without AWS.

Builds each arm as a plain local Zarr store of 2x2 shards from real v2 tiles, then records what
the layout decision needs that does not depend on the network:

* **write cost** — stored bytes and encode CPU per shard, through the same whole-shard assignment
  the assembly writer makes;
* **request plans** — the exact requests and bytes every read workload makes, through a counting
  store, on whichever zarr version runs the phase. They carry over to S3 because they depend only
  on zarr's sharding codec; latency does not, which is what T1 measures;
* **decode CPU** per workload, and the **Dask graph** a reader's default chunking builds.

Stores are written by the library's own environment. The ``requests`` phase is recorded under the
zarr version that runs it, so the same run can be repeated by an old client (zarr 3.2.1 +
icechunk 2.1.1) against the same stores; ``--phase requests`` or ``--phase dask`` selects the
running client's. Run from the REPOSITORY ROOT::

    python -m scripts.scoping.scale_tests.t9_v2_layout --run-id v2a --tiles <dir>
    <old-venv>/bin/python -m scripts.scoping.scale_tests.t9_v2_layout --run-id v2a --tiles <dir>

``--tiles`` holds ``emb_<n>.npy`` (2048, 2048, 128) int8 and ``scales_<n>.npy`` (2048, 2048)
float32, written by ``--phase fetch --source <store> --profile <aws profile>``. Without it the
tiny scale smokes on synthetic data.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr
from zarr.abc.store import Store
from zarr.storage import LocalStore, WrapperStore

from scripts.scoping.scale_tests import harness, synth
from scripts.scoping.scale_tests import variants as V

logger = logging.getLogger("scale_tests.t9")

TEST = "t9"
TILE = 2048
DIMS_READ = (16, 64, 128)
CLIENT = f"zarr{'.'.join(zarr.__version__.split('.')[:2])}"

_GEOMETRIES = [V.VARIANTS[n] for n in ("v2_c0", "v2_c1", "v2_c2", "v2_b2")]
_CODEC_ARMS = (("zstd1", False), ("raw", False), ("blosc_bitshuffle", False), ("zstd", True), ("zstd1", True))
ARMS: list[V.Variant] = [
    *_GEOMETRIES,
    *(V.arm(g, codec=c, transpose=t) for g in _GEOMETRIES for c, t in _CODEC_ARMS),
    *(V.arm(V.VARIANTS[g], transpose=True, write_order=o) for g in ("v2_c1", "v2_c2") for o in V.WRITE_ORDERS[1:]),
    *(V.arm(V.VARIANTS[g], transpose=True, scales_chunks=(1, 512, 512)) for g in ("v2_c1", "v2_c2")),
]

# (label, y0, x0, extent): non-aligned windows inside shard (0, 0), then aligned whole shards.
REGIONS = (("patch", 700, 900, 100), ("tile", 500, 700, 1000), ("shard", 0, 0, TILE), ("bulk", 0, 0, 2 * TILE))


class CountingStore(WrapperStore):
    """Record every request a read makes, on zarr 3.2 and 3.4 alike.

    zarr >= 3.3 reads a partial shard through ``Store.get_ranges``, which ``WrapperStore`` forwards
    to the wrapped store, and zarr 3.4 also has a synchronous path it forwards the same way — either
    would bypass a counter on ``get``. So this opts out of sync IO (icechunk's store is async-only,
    so that is also the path an S3 reader takes) and runs the base merged-range reader over its own
    ``get``. Metadata reads are not counted.
    """

    def __init__(self, store: Store) -> None:
        """Wrap ``store`` with an empty request log."""
        super().__init__(store)
        self.requests: list[int] = []

    @property
    def _supports_sync_io(self) -> bool:
        """Force zarr's async read path."""
        return False

    async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:  # noqa: ANN401
        """Fetch through the wrapped store and log the returned length."""
        buf = await self._store.get(key, prototype, byte_range)
        if not key.endswith("zarr.json"):
            self.requests.append(len(buf) if buf is not None else 0)
        return buf

    if hasattr(Store, "get_ranges"):

        async def get_ranges(self, key: str, byte_ranges: Any, **kwargs: Any) -> AsyncIterator[Any]:  # noqa: ANN401
            """Merge ranges with zarr's own coalescer, fetching through :meth:`get`."""
            async for group in Store.get_ranges(self, key, byte_ranges, **kwargs):
                yield group


# ── tiles ────────────────────────────────────────────────────────────────────


def _tile(cfg: harness.RunConfig, tiles: Path | None, n: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return tile ``n`` (cycling) as ``(embeddings (h, w, 128) int8, scales (h, w) float32)``."""
    if tiles is None:
        if not cfg.is_tiny:
            raise SystemExit("--tiles is required at --scale bench")
        return synth.embedding_block((h, w, V.BAND), seed=0, block_index=(n,)), synth.scales_block(
            (h, w), seed=0, block_index=(n,)
        )
    names = sorted(tiles.glob("emb_*.npy"))
    emb_path = names[n % len(names)]
    emb = np.load(emb_path, mmap_mode="r")[:h, :w]
    scales = np.load(emb_path.with_name(emb_path.name.replace("emb_", "scales_")), mmap_mode="r")[:h, :w]
    return np.ascontiguousarray(emb), np.ascontiguousarray(scales)


def _side(cfg: harness.RunConfig) -> int:
    """Store side in pixels: 2x2 shards at bench, one partial shard at tiny."""
    return 1024 if cfg.is_tiny else 2 * TILE


def phase_fetch(source: str, profile: str, tiles: Path, n_tiles: int) -> None:
    """Copy ``n_tiles`` whole 2048-px tiles of a v2 store into ``tiles`` as ``.npy``.

    The tiles are spread evenly, in grid order, over every tile holding any embedding, rather than
    taken fullest-first: in a uniform area the fullest tiles cluster, and a coast's partial tiles
    are what the elision and compression of a real zone depend on. A band-first store's tiles are
    transposed to the ``(y, x, band)`` every later phase reads. Opens the store with static
    credentials from ``profile``: Icechunk's provider chain skips environment credentials, so the
    library's own opener fails from a laptop.
    """
    import boto3
    import icechunk

    bucket, _, prefix = source.removeprefix("s3://").partition("/")
    creds = boto3.Session(profile_name=profile).get_credentials().get_frozen_credentials()
    storage = icechunk.s3_storage(
        bucket=bucket,
        prefix=prefix,
        region="us-west-2",
        access_key_id=creds.access_key,
        secret_access_key=creds.secret_key,
        session_token=creds.token,
    )
    root = zarr.open_group(icechunk.Repository.open(storage).readonly_session(branch="main").store, mode="r")
    scales, emb = root["scales"], root["embeddings"]
    _, ny, nx = scales.shape
    windows = [(y, x) for y in range(0, ny - TILE + 1, TILE) for x in range(0, nx - TILE + 1, TILE)]
    valid = {(y, x): int(np.isfinite(scales[0, y : y + TILE : 64, x : x + TILE : 64]).sum()) for y, x in windows}
    live = [w for w in windows if valid[w]]
    picks = [live[int(i)] for i in np.linspace(0, len(live) - 1, min(n_tiles, len(live)))]
    tiles.mkdir(parents=True, exist_ok=True)
    # Later phases read every emb_*/scales_* pair in the directory, so a re-fetch of fewer tiles
    # must not leave an earlier fetch's higher-numbered pairs behind.
    for stale in (*tiles.glob("emb_*.npy"), *tiles.glob("scales_*.npy")):
        stale.unlink()
    for i, (y, x) in enumerate(picks):
        np.save(tiles / f"emb_{i:03d}.npy", V.read_window(emb, 0, slice(y, y + TILE), slice(x, x + TILE)))
        np.save(tiles / f"scales_{i:03d}.npy", np.asarray(scales[0, y : y + TILE, x : x + TILE]))
        logger.info("fetched tile %d at (%d, %d), %d%% valid", i, y, x, 100 * valid[(y, x)] // (TILE // 64) ** 2)


# ── build ────────────────────────────────────────────────────────────────────


def _store_dir(cfg: harness.RunConfig, arm: V.Variant) -> Path:
    """Local directory holding one arm's store."""
    return Path(cfg.store_root) / "t9" / arm.name


def _dir_bytes(path: Path) -> int:
    """Total bytes of every file under ``path``."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def phase_build(cfg: harness.RunConfig, tiles: Path | None) -> None:
    """Write every arm from the same tiles, one whole-shard assignment per shard, timing each."""
    side = _side(cfg)
    for arm in ARMS:
        path = _store_dir(cfg, arm)
        harness.reset_store(cfg, f"t9/{arm.name}")
        group = zarr.open_group(LocalStore(path), mode="w")
        emb = group.create_array("embeddings", **V.embeddings_array_kwargs(arm, (1, side, side, V.BAND)))
        scl = group.create_array("scales", **V.scales_array_kwargs(arm, (1, side, side)))
        V.apply_write_order(emb, arm.write_order)
        V.apply_write_order(scl, arm.write_order)
        step = min(TILE, side)
        for n, (y0, x0) in enumerate((y, x) for y in range(0, side, step) for x in range(0, side, step)):
            block, scales = _tile(cfg, tiles, n, step, step)
            cpu0, wall0 = time.process_time(), time.monotonic()
            emb[0:1, y0 : y0 + step, x0 : x0 + step] = block[None]
            cpu = time.process_time() - cpu0
            harness.emit_metric(cfg, TEST, "build", "encode_cpu_s", cpu, "s", variant=arm.name, shard=n)
            harness.emit_metric(cfg, TEST, "build", "wall_s", time.monotonic() - wall0, "s", variant=arm.name, shard=n)
            scl[0:1, y0 : y0 + step, x0 : x0 + step] = scales[None]
        for name in ("embeddings", "scales"):
            stored = _dir_bytes(path / name / "c") if (path / name / "c").exists() else 0
            harness.emit_metric(cfg, TEST, "build", "bytes_written", stored, "bytes", variant=arm.name, array=name)
        logger.info("built %s: %.1f MB embeddings", arm.name, _dir_bytes(path / "embeddings") / 1e6)


# ── reads ────────────────────────────────────────────────────────────────────


Measured = tuple[float, float, float, float]


def _measure(store: CountingStore, read: Callable[..., object], *args: object) -> Measured:
    """Run ``read(*args)`` once; return (requests, bytes, wall_s, cpu_s)."""
    store.requests.clear()
    cpu0, wall0 = time.process_time(), time.monotonic()
    read(*args)
    return len(store.requests), sum(store.requests), time.monotonic() - wall0, time.process_time() - cpu0


def _emit_read(cfg: harness.RunConfig, phase: str, arm: str, result: Measured, **tags: object) -> None:
    """Emit one read's requests, bytes, wall and decode CPU."""
    for metric, value, unit in zip(
        ("get_requests", "bytes_fetched", "wall_s", "decode_cpu_s"), result, ("count", "bytes", "s", "s"), strict=True
    ):
        harness.emit_metric(cfg, TEST, phase, metric, value, unit, variant=arm, client=CLIENT, **tags)


def _windows(side: int) -> list[tuple[str, tuple[slice, slice]]]:
    """Each REGIONS window as (label, (northing slice, easting slice)), clamped to the store."""
    out = []
    for label, y0, x0, ext in REGIONS:
        ext = min(ext, side - y0, side - x0)
        out.append((label, (slice(y0, y0 + ext), slice(x0, x0 + ext))))
    return out


def phase_requests(cfg: harness.RunConfig, phase: str, n_points: int) -> None:
    """Every workload at every prefix width, on every arm, through the counting store."""
    side = _side(cfg)
    points = [(int(y), int(x)) for y, x in np.random.default_rng(1).integers(0, side, size=(n_points, 2))]
    for arm in ARMS:
        store = CountingStore(LocalStore(_store_dir(cfg, arm), read_only=True))
        group = zarr.open_group(store, mode="r")
        emb, scl = group["embeddings"], group["scales"]
        for dims in DIMS_READ:
            per_point = [_measure(store, emb.__getitem__, (0, y, x, slice(0, dims))) for y, x in points]
            n, nbytes, _, cpu = (float(np.mean(col)) for col in zip(*per_point, strict=True))
            wall = float(np.median([p[2] for p in per_point]))
            _emit_read(cfg, phase, arm.name, (n, nbytes, wall, cpu), workload="point", dims=dims, array="embeddings")
            for label, (ys, xs) in _windows(side):
                result = _measure(store, emb.__getitem__, (0, ys, xs, slice(0, dims)))
                _emit_read(cfg, phase, arm.name, result, workload=label, dims=dims, array="embeddings")
        result = _measure(store, scl.__getitem__, (0, *points[0]))
        _emit_read(cfg, phase, arm.name, result, workload="point", dims=0, array="scales")
        for label, (ys, xs) in _windows(side):
            _emit_read(
                cfg,
                phase,
                arm.name,
                _measure(store, scl.__getitem__, (0, ys, xs)),
                workload=label,
                dims=0,
                array="scales",
            )
        logger.info("[%s] read %s", CLIENT, arm.name)


def phase_dask(cfg: harness.RunConfig, phase: str) -> None:
    """Task count and requests for the bulk 16-d pull through xarray, default vs shard-aligned chunks."""
    import dask

    side = _side(cfg)
    aligned = {"time": 1, "northing": min(TILE, side), "easting": min(TILE, side), "band": 16}
    for arm in (a for a in ARMS if a.codec == "zstd" and a.write_order == "morton" and a.scales_chunks is None):
        store = CountingStore(LocalStore(_store_dir(cfg, arm), read_only=True))
        for label, chunks in (("default", {}), ("shard_aligned", aligned)):
            ds = xr.open_zarr(store, consolidated=False, chunks=chunks)
            lazy = ds["embeddings"].isel(time=0, band=slice(0, 16)).data
            tasks = len(lazy.__dask_graph__())
            harness.emit_metric(
                cfg, TEST, phase, "dask_tasks", tasks, "count", variant=arm.name, client=CLIENT, chunks=label
            )
            with dask.config.set(scheduler="synchronous"):
                _emit_read(
                    cfg,
                    phase,
                    arm.name,
                    _measure(store, lazy.compute),
                    workload="bulk",
                    dims=16,
                    array="embeddings",
                    chunks=label,
                )


def main() -> int:
    """Parse args and run the T9 phases; build and dask only on the library's own zarr."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    parser.add_argument("--tiles", type=Path, default=None, help="Directory of real v2 tiles (.npy).")
    parser.add_argument("--source", default=None, help="v2 store to fetch tiles from (--phase fetch).")
    parser.add_argument("--profile", default=None, help="AWS profile for --source.")
    parser.add_argument("--n-tiles", type=int, default=4)
    parser.add_argument("--points", type=int, default=100)
    args = parser.parse_args()
    cfg = harness.config_from_args(args)
    harness.configure_logging()
    if cfg.is_s3:
        raise SystemExit("T9 measures local stores only; latency on S3 is T1's job")

    if cfg.phase == "fetch":
        phase_fetch(args.source, args.profile, args.tiles, args.n_tiles)
        return 0
    if tuple(int(p) for p in zarr.__version__.split(".")[:2]) >= (3, 4):
        harness.run_phase(cfg, TEST, "build", lambda: phase_build(cfg, args.tiles))
        harness.run_phase(cfg, TEST, f"dask_{CLIENT}", lambda: phase_dask(cfg, f"dask_{CLIENT}"))
    harness.run_phase(cfg, TEST, f"requests_{CLIENT}", lambda: phase_requests(cfg, f"requests_{CLIENT}", args.points))
    logger.info("T9 complete under %s.", CLIENT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
