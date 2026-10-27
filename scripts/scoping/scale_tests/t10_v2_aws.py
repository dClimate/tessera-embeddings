"""T10 — the v2 layout finalists on AWS: cold read latency from S3, and the assembly code's write cost.

T9 settled what does not depend on the network. This measures the two things that do, each on its
own in-region EC2 box, in the dev account, into throwaway repos under the run's store root:

* ``build`` then ``read_<client>`` (x86, the read benchmark's r7i.4xlarge). Every finalist is built as
  an Icechunk repo of 10 x 10 shards cycled from real v2 tiles, so no cache holds the working set.
  Each workload then runs cold in a fresh process, arms interleaved, in two passes of opposite arm
  order. Records wall time, requests, store bytes and wire bytes (psutil). Run ``read`` once per
  client: the library's environment, then one pinned to zarr 3.2.1 + icechunk 2.1.1.
* ``assemble`` (Graviton, like the assembly runner). The same staged v2 tiles are written into each
  finalist by the library's own assembly code, ``shard_writer.write_year_shards`` over a
  ``StagedShardSource`` — every array a fill writes, not just the two that change — in two passes of
  opposite order. Records worker CPU (total, read, compress-and-upload), wall time and stored bytes.
* ``mobile`` (the read box, under ``tc netem`` shaping). The mobile reads of ADR 031 §6 through a
  browser-like client: no merged requests, 6 in flight, no split downloads, on a phone network
  profile (:data:`NETWORKS`) the phase measures before it runs.

``--study`` picks the arms and workloads, as in T9: ``band16`` (the default) is ADR 031's scoping,
``narrow`` its 4-, 8- and variable-width band chunks for mobile readers.

Run from the REPOSITORY ROOT, with ``T10="-m scripts.scoping.scale_tests.t10_v2_aws --run-id v2b
--backend s3 --bucket <bucket>/<prefix> --scale bench"``::

    python $T10 --tiles <dir> --phase build
    python $T10 --tiles <dir> --phase read        # once per client
    python $T10 --tiles <dir> --phase tune        # full-vector pulls under reader settings
    python $T10 --phase assemble --staging s3://<staged run dir> --extent 34964,53383 --n-tiles 64 --workers 16
    python $T10 --study narrow --phase mobile --network poor   # after `tc` shapes the box's link
    python $T10 --phase verify                    # the built stores hold C0's values
    python $T10 --phase verify --staging s3://<staged run dir> --n-tiles 64   # the assembled stores do
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import xarray as xr
import zarr

from scripts.scoping.scale_tests import harness
from scripts.scoping.scale_tests import variants as V
from scripts.scoping.scale_tests._workers import _open_repo
from scripts.scoping.scale_tests.t9_v2_layout import CLIENT, DIMS_READ, DIRECT_READ, TILE, CountingStore
from tessera_embeddings.storage import shard_writer

logger = logging.getLogger("scale_tests.t10")

TEST = "t10"
#: Store prefixes: ``build`` writes one repo per arm under BUILT, ``assemble`` one per arm and pass under ASSEMBLED.
BUILT, ASSEMBLED = "t10", "t10asm"
_S256 = (1, 256, 256)
FINALISTS: list[V.Variant] = [
    V.VARIANTS["v2_c0"],
    V.arm(V.VARIANTS["v2_c2"], codec="zstd1", scales_chunks=_S256),
    V.arm(V.VARIANTS["v2_c2"], codec="zstd1", transpose=True, scales_chunks=_S256),
    V.arm(V.VARIANTS["v2_b2"], codec="zstd1", scales_chunks=_S256),
    V.arm(V.VARIANTS["v2_c2f"], codec="zstd1", scales_chunks=_S256),
    V.arm(V.VARIANTS["v2_b2f"], codec="zstd1", scales_chunks=_S256),
]
POINTS = 300
PASSES = 2
#: Phone networks for the mobile phase (ADR 031 §6): added round trip in ms, downlink in Mbit/s. The box's link is
#: shaped to one with `tc netem` + `ifb` before the phase runs; "poor" is Lighthouse's mobile profile.
NETWORKS: dict[str, tuple[int, float]] = {"good": (40, 20.0), "poor": (150, 1.6)}
#: The browser-like client of the mobile phase: zarrita.js's one request per inner chunk, a browser's 6 connections
#: per host, and each object fetched in one GET.
MOBILE_CLIENT: dict[str, Any] = {"zarr_config": DIRECT_READ, "max_in_flight": 6, "split_large_gets": False}
#: Workloads that read codes alone: a preview renders them, and cosine similarity ignores the per-pixel scale.
SCALE_FREE = ("maptile", "viewport")


def _side(cfg: harness.RunConfig) -> int:
    """Bench zone side in shards: 10 at bench, 1 for a tiny smoke."""
    return 1 if cfg.is_tiny else 10


def _regions(side: int) -> dict[str, tuple[slice, slice]]:
    """The region workloads: a 1,000-px tile inside one shard, a 2x2-shard bulk, an 8x8-shard pull."""
    k = min(side, 8)
    off = TILE * min(3, side - 1)
    corner = TILE * max(1, min(4, side - 1))  # a shard corner, for a viewport that spans four shards
    return {
        "tile": (slice(off + 500, off + 1500), slice(off + 700, off + 1700)),
        "bulk": (slice(TILE * min(2, side - 1), TILE * min(4, side)),) * 2,
        "pull8": (slice(0, TILE * k),) * 2,
        # The mobile windows, placed as in T9: a 256² map tile on the 256 grid, a 1,024 x 2,048 phone viewport.
        "maptile": (slice(off + 512, off + 768), slice(off + 768, off + 1024)),
        "viewport": (slice(corner - 512, corner + 512), slice(corner - 1024, corner + 1024)),
    }


# ── build ────────────────────────────────────────────────────────────────────


def _write_tiles(payload: dict[str, Any]) -> tuple[Any, float, int]:
    """Spawned worker: write whole shards into the fork in the arm's order; return (fork, write CPU, shards).

    A band-first arm's tiles are put in band-first order BEFORE the timer: a band-first pipeline
    stages them that way, so the write cost excludes any transpose.
    """
    fork, arm, tiles = payload["fork"], V.Variant(**payload["arm"]), Path(payload["tiles"])
    grp = zarr.open_group(fork.store, mode="a")["zone"]
    emb, scl = grp["embeddings"], grp["scales"]
    V.apply_write_order(emb, arm.write_order)
    V.apply_write_order(scl, arm.write_order)
    cpu = 0.0
    for sy, sx, n in payload["shards"]:
        ys, xs = slice(sy * TILE, (sy + 1) * TILE), slice(sx * TILE, (sx + 1) * TILE)
        block = V.to_storage_order(emb, np.load(tiles / f"emb_{n:03d}.npy"))
        scales = np.load(tiles / f"scales_{n:03d}.npy")
        c0 = time.process_time()
        emb[V.window_key(emb, slice(0, 1), ys, xs)] = block[None]
        scl[0:1, ys, xs] = scales[None]
        cpu += time.process_time() - c0
    return fork, cpu, len(payload["shards"])


def phase_build(cfg: harness.RunConfig, arms: list[V.Variant], tiles: Path, workers: int) -> None:
    """Build every finalist as an Icechunk repo of side x side shards, tiles cycled over the grid."""
    side, n_tiles = _side(cfg), len(list(tiles.glob("emb_*.npy")))
    jobs = [(sy, sx, (sy * side + sx) % n_tiles) for sy in range(side) for sx in range(side)]
    for arm in arms:
        name = f"{BUILT}/{arm.name}"
        harness.reset_store(cfg, name)
        repo = harness.create_repo(cfg, name)
        session = repo.writable_session("main")
        grp = zarr.open_group(session.store, mode="a").require_group("zone")
        grp.create_array("embeddings", **V.embeddings_array_kwargs(arm, (1, side * TILE, side * TILE, V.BAND)))
        grp.create_array("scales", **V.scales_array_kwargs(arm, (1, side * TILE, side * TILE)))
        session.commit("seed")
        session = repo.writable_session("main")
        fork = session.fork()
        payloads = [
            {"fork": fork, "arm": dataclasses.asdict(arm), "tiles": str(tiles), "shards": part}
            for part in shard_writer.partition_round_robin(jobs, workers)
        ]
        t0 = time.monotonic()
        with ProcessPoolExecutor(len(payloads), mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(_write_tiles, payloads))
        session.merge(*(fork for fork, _, _ in results))
        session.commit(f"build {arm.name}")
        n_obj, n_bytes = harness.object_stats(harness.store_uri(cfg, name) + "/chunks")
        harness.emit_metric(cfg, TEST, "build", "wall_s", time.monotonic() - t0, "s", variant=arm.name)
        cpu_per_shard = sum(c for _, c, _ in results) / sum(n for _, _, n in results)
        harness.emit_metric(cfg, TEST, "build", "encode_cpu_s", cpu_per_shard, "s", variant=arm.name, kind="write")
        harness.emit_metric(cfg, TEST, "build", "bytes_written", n_bytes, "bytes", variant=arm.name, objects=n_obj)
        # A direct reader fetches an array's manifest on its first chunk read; it grows with the objects referenced.
        _, m_bytes = harness.object_stats(harness.store_uri(cfg, name) + "/manifests")
        harness.emit_metric(cfg, TEST, "build", "manifest_bytes", m_bytes, "bytes", variant=arm.name, objects=n_obj)
        logger.info("built %s: %d objects, %.1f GB", arm.name, n_obj, n_bytes / 1e9)


# ── read ─────────────────────────────────────────────────────────────────────


def cold_read(payload: dict[str, Any]) -> dict[str, Any]:
    """One workload on one arm in a fresh process: wall, requests, store bytes, wire bytes.

    A ``client`` payload (:data:`MOBILE_CLIENT`) reads as a browser does, and first reads one pixel of
    a distant shard, timed as ``first_read_s``: that read fetches the manifest, which every cold
    workload would otherwise carry in its own time.
    """
    import psutil

    client = payload.get("client") or {}
    zarr.config.set({**(payload.get("zarr_config") or {}), **client.get("zarr_config", {})})
    t_open = time.monotonic()
    repo = _open_repo(
        payload["store_uri"],
        max_concurrent_requests=client.get("max_in_flight"),
        split_large_gets=client.get("split_large_gets", True),
    )
    store = CountingStore(repo.readonly_session(branch="main").store, max_in_flight=client.get("max_in_flight"))
    grp = zarr.open_group(store, mode="r")["zone"]
    emb, scl = grp["embeddings"], grp["scales"]
    out: dict[str, Any] = {"open_wall_s": time.monotonic() - t_open}
    workload, dims = payload["workload"], payload["dims"]
    if client:
        t_first = time.monotonic()
        _, ny, nx = scl.shape
        V.read_window(emb, 0, ny - 1, nx - 1, slice(0, 1))
        out["first_read_s"] = time.monotonic() - t_first
        store.requests.clear()

    regions = {k: tuple(slice(*s) for s in v) for k, v in payload["regions"].items()}
    net0, t0 = psutil.net_io_counters().bytes_recv, time.monotonic()
    if workload == "point":
        lat = []
        for y, x in payload["points"]:
            p0 = time.monotonic()
            V.read_window(emb, 0, y, x, slice(0, dims)) * scl[0, y, x]  # a dequantised prefix, as a reader reads
            lat.append(time.monotonic() - p0)
        ms = 1e3 * np.asarray(lat)
        # A 95% bootstrap interval on the median: a gate fails only if its interval clears the threshold.
        boots = np.median(np.random.default_rng(0).choice(ms, (2000, ms.size)), axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out |= {"read_p50_ms": float(np.median(ms)), "read_p95_ms": float(np.percentile(ms, 95))}
        out |= {"read_p50_lo_ms": float(lo), "read_p50_hi_ms": float(hi)}
    elif workload.startswith("dask"):
        chunks = {} if workload == "dask_default" else V.dask_chunks(emb)
        ds = xr.open_zarr(store, group="zone", consolidated=False, chunks=chunks)
        ys, xs = regions["pull8"]
        ds["embeddings"].isel(time=0, northing=ys, easting=xs, band=slice(0, dims)).data.compute()
    else:
        ys, xs = regions[workload]
        V.read_window(emb, 0, ys, xs, slice(0, dims))
        if workload not in SCALE_FREE:
            scl[0, ys, xs]
    # Per read, as T9 records it: a point workload is many reads, one per probe.
    reads = len(payload["points"]) if workload == "point" else 1
    out |= {
        "wall_s": (time.monotonic() - t0) / reads,
        "requests": len(store.requests) / reads,
        "bytes": sum(store.requests) / reads,
        "wire_bytes": (psutil.net_io_counters().bytes_recv - net0) / reads,
    }
    return out


def _points(cfg: harness.RunConfig, tiles: Path) -> list[list[int]]:
    """POINTS pixels holding an embedding, spread over every shard of the zone."""
    side, n_tiles = _side(cfg), len(list(tiles.glob("emb_*.npy")))
    rng, pts = np.random.default_rng(7), []
    valid = [np.argwhere(np.isfinite(np.load(tiles / f"scales_{n:03d}.npy"))) for n in range(n_tiles)]
    for i in range(POINTS):
        sy, sx = divmod(i % (side * side), side)
        yx = valid[(sy * side + sx) % n_tiles][rng.integers(len(valid[(sy * side + sx) % n_tiles]))]
        pts.append([int(sy * TILE + yx[0]), int(sx * TILE + yx[1])])
    return pts


#: The read sweep: (workload, dims, zarr config). ``tune`` re-runs the full-vector pulls with a wider
#: fetch concurrency and band-aligned Dask chunks, to see how much of their cost a reader can recover.
READ_WORKLOADS: list[tuple[str, int, dict]] = [
    ("point", 16, {}),
    ("point", 128, {}),
    ("bulk", 128, {}),
    *(("pull8", d, {}) for d in DIMS_READ),
    ("dask_default", 16, {}),
    ("dask_default", 128, {}),
]
POINT_WORKLOADS: list[tuple[str, int, dict]] = [("point", 16, {}), ("point", 128, {})]
TUNE_WORKLOADS: list[tuple[str, int, dict]] = [
    ("pull8", 128, {}),
    ("pull8", 128, {"async.concurrency": 64}),
    ("dask_aligned", 128, {}),
    ("bulk", 128, {"async.concurrency": 64}),
    ("pull8", 16, {"async.concurrency": 64}),
]


def phase_read(
    cfg: harness.RunConfig,
    phase: str,
    tiles: Path,
    workloads: list,
    arms: list[V.Variant],
    client: dict[str, Any] | None = None,
) -> None:
    """Every workload, cold, arms interleaved, PASSES passes with the arm order reversed each pass."""
    regions = {k: [(s.start, s.stop) for s in v] for k, v in _regions(_side(cfg)).items()}
    points = _points(cfg, tiles) if any(w == "point" for w, _, _ in workloads) else []
    for p in range(PASSES):
        for workload, dims, zarr_config in workloads:
            for arm in arms if p % 2 == 0 else arms[::-1]:
                payload = {
                    "store_uri": harness.store_uri(cfg, f"{BUILT}/{arm.name}"),
                    "workload": workload,
                    "dims": dims,
                    "regions": regions,
                    "points": points if workload == "point" else [],
                    "zarr_config": zarr_config,
                    "client": client,
                }
                r = harness.run_cold("scripts.scoping.scale_tests.t10_v2_aws.cold_read", payload)
                tags = {"variant": arm.name, "client": CLIENT, "workload": workload, "dims": dims, "pass": p}
                if zarr_config:
                    tags["zarr_config"] = ",".join(f"{k}={v}" for k, v in zarr_config.items())
                harness.emit_metric(cfg, TEST, phase, "wall_s", r["wall_s"], "s", **tags)
                harness.emit_metric(cfg, TEST, phase, "get_requests", r["requests"], "count", **tags)
                harness.emit_metric(cfg, TEST, phase, "bytes_fetched", r["bytes"], "bytes", method="store", **tags)
                harness.emit_metric(cfg, TEST, phase, "bytes_fetched", r["wire_bytes"], "bytes", method="wire", **tags)
                for k in ("read_p50_ms", "read_p95_ms", "read_p50_lo_ms", "read_p50_hi_ms"):
                    if k in r:
                        harness.emit_metric(cfg, TEST, phase, k, r[k], "ms", **tags)
                for k in ("open_wall_s", "first_read_s"):
                    if k in r:
                        harness.emit_metric(cfg, TEST, phase, k, r[k], "s", **tags)
                logger.info(
                    "[%s] pass %d %s/%d %s: %.2fs %d req",
                    phase,
                    p,
                    workload,
                    dims,
                    arm.name,
                    r["wall_s"],
                    r["requests"],
                )


class Study(NamedTuple):
    """The arms a study builds, control first, and the workloads each read phase runs."""

    arms: list[V.Variant]
    read: list[tuple[str, int, dict]]
    points: list[tuple[str, int, dict]]
    tune: list[tuple[str, int, dict]]
    mobile: list[tuple[str, int, dict]]


#: The mobile reads (ADR 031 §6), at the prefixes a preview or an on-device comparison takes.
MOBILE_WORKLOADS: list[tuple[str, int, dict]] = [(w, d, {}) for w in SCALE_FREE for d in (4, 8)]
#: The narrow finalists, fixed before T9 ran: the control, the accepted layout, band-split 8 and 4, 16-band shards
#: with 4-band inner chunks, and two variable-width grids.
NARROW_FINALISTS: list[V.Variant] = [
    V.VARIANTS["v2_c0"],
    V.ACCEPTED,
    *(
        V.arm(V.VARIANTS[g], codec="zstd1", scales_chunks=_S256)
        for g in ("v2_b2f8", "v2_b2f4", "v2_b2f_i4", "v2_m2f", "v2_m2f_x4")
    ),
]
STUDIES: dict[str, Study] = {
    "band16": Study(FINALISTS, READ_WORKLOADS, POINT_WORKLOADS, TUNE_WORKLOADS, mobile=[]),
    "narrow": Study(
        NARROW_FINALISTS, READ_WORKLOADS + MOBILE_WORKLOADS, POINT_WORKLOADS, TUNE_WORKLOADS, MOBILE_WORKLOADS
    ),
}
assert all(study.arms[0].name == "v2_c0" for study in STUDIES.values()), "every study runs the control first"


def _probe_network(cfg: harness.RunConfig, phase: str, arm: V.Variant, network: str) -> None:
    """Measure the round trip and downlink to the stores, and refuse to run unless ``network``'s shaping holds."""
    rtt_want, mbps_want = NETWORKS[network]
    fs, path = harness.fs_and_path(harness.store_uri(cfg, f"{BUILT}/{arm.name}") + "/chunks")
    key, meta = max(fs.find(path, detail=True).items(), key=lambda kv: int(kv[1].get("size") or 0))
    rtts = []
    for _ in range(20):
        t0 = time.monotonic()
        fs.cat_file(key, start=0, end=1)
        rtts.append(1e3 * (time.monotonic() - t0))
    n = min(int(meta["size"]), 2 << 20)
    t0 = time.monotonic()
    fs.cat_file(key, start=0, end=n)
    rtt, mbps = float(np.median(rtts)), 8 * n / 1e6 / (time.monotonic() - t0)
    harness.emit_metric(cfg, TEST, phase, "probe_rtt_ms", rtt, "ms", network=network)
    harness.emit_metric(cfg, TEST, phase, "probe_mbps", mbps, "Mbit/s", network=network)
    logger.info("network %s: %.0f ms round trip, %.1f Mbit/s down", network, rtt, mbps)
    if rtt < rtt_want / 2 or mbps > 2 * mbps_want:
        raise SystemExit(
            f"network {network!r} wants about {rtt_want} ms and {mbps_want} Mbit/s, measured {rtt:.0f} ms and "
            f"{mbps:.1f} Mbit/s: shape the link with tc first"
        )


def phase_mobile(cfg: harness.RunConfig, phase: str, study: Study, tiles: Path, network: str) -> None:
    """The mobile reads through the browser-like client, on the shaped link (a local smoke skips the probe)."""
    if cfg.is_s3:
        _probe_network(cfg, phase, study.arms[0], network)
    phase_read(cfg, phase, tiles, study.mobile, study.arms, client=MOBILE_CLIENT)


# ── assemble ─────────────────────────────────────────────────────────────────


def _picks(staging: str, n_tiles: int) -> list[str]:
    """The staged tiles assembly writes: ``n_tiles`` labels spread evenly over the staged run."""
    from tessera_embeddings.profiling.inference.compare_outputs import _staged_labels

    labels = sorted(_staged_labels(staging)[0])
    return [labels[int(i)] for i in np.linspace(0, len(labels) - 1, min(n_tiles, len(labels)))]


def phase_assemble(
    cfg: harness.RunConfig, staging: str, extent: tuple[int, int], n_tiles: int, workers: int, arms: list[V.Variant]
) -> None:
    """Write the same staged tiles into each arm with the library's assembly code; time the workers."""
    from tessera_embeddings.config.store_layout import CARRIED_VARS, GLOBAL, REQUIRED_VARS
    from tessera_embeddings.inference.assembly import StagedShardSource, _staged_storage_options
    from tessera_embeddings.inference.chunk_spec import parse_chunk_label
    from tessera_embeddings.storage.shard_writer import write_year_shards
    from tessera_embeddings.storage.time_axis import TIME_ENCODING, calendar_year_times

    picks = _picks(staging, n_tiles)
    probe_path = f"{staging}/{picks[0]}.zarr"
    probe = zarr.open_group(probe_path, mode="r", storage_options=_staged_storage_options(probe_path))
    variables = tuple(v for v in (*REQUIRED_VARS, *CARRIED_VARS) if v in probe and v in GLOBAL.arrays)
    dtypes = tuple((v, str(probe[v].dtype)) for v in variables)
    base, run_id = staging.rstrip("/").rsplit("/", 1)
    ny, nx = extent
    for p in range(PASSES):
        for arm in arms if p % 2 == 0 else arms[::-1]:
            name = f"{ASSEMBLED}/p{p}/{arm.name}"
            harness.reset_store(cfg, name)
            repo = harness.create_repo(cfg, name)
            session = repo.writable_session("main")
            grp = zarr.open_group(session.store, mode="a").require_group("zone")
            for v in variables:
                layout = GLOBAL.arrays[v]
                shape = (1, ny, nx, *layout.chunks[3:])[: len(layout.dims)]
                if v == "embeddings":
                    kwargs = V.embeddings_array_kwargs(arm, (1, ny, nx, V.BAND))
                elif v == "scales":
                    kwargs = V.scales_array_kwargs(arm, (1, ny, nx))
                else:
                    kwargs = layout.create_kwargs(shape)
                grp.create_array(v, **kwargs).attrs.update(dict(layout.attrs))
            times = calendar_year_times((2025,)).astype("datetime64[ns]").astype("int64")
            grp.create_array("time", data=times, chunks=(1,), dimension_names=("time",)).attrs.update(TIME_ENCODING)
            session.commit("seed")
            source = StagedShardSource(
                staging_base=base,
                run_id=run_id,
                shards=tuple(parse_chunk_label(label) for label in picks),
                variables=variables,
                shard_px=TILE,
                dtypes=dtypes,
                embedding_dim=V.BAND,
                extent=(ny, nx),
            )
            telemetry: dict[str, Any] = {}
            t0 = time.monotonic()
            write_year_shards(
                repo,
                "zone",
                0,
                source,
                n_workers=workers,
                telemetry=telemetry,
                commit_attrs=lambda _repo, _group, _year, commit: commit["snapshot"],
            )
            wall = time.monotonic() - t0
            stats = telemetry["workers"]
            tags = {"variant": arm.name, "pass": p, "tiles": len(picks), "workers": workers}
            for kind, key in (("worker", "cpu_s"), ("write", "write_cpu_s"), ("read", "read_cpu_s")):
                total = sum(float(w.get(key, 0.0)) for w in stats)
                harness.emit_metric(cfg, TEST, "assemble", "encode_cpu_s", total / len(picks), "s", kind=kind, **tags)
            harness.emit_metric(cfg, TEST, "assemble", "wall_s", wall, "s", **tags)
            _, n_bytes = harness.object_stats(harness.store_uri(cfg, name) + "/chunks")
            harness.emit_metric(cfg, TEST, "assemble", "bytes_written", n_bytes, "bytes", **tags)
            logger.info(
                "assembled %s pass %d: %d tiles in %.0fs, %.1f GB", arm.name, p, len(picks), wall, n_bytes / 1e9
            )


# ── verify ───────────────────────────────────────────────────────────────────


def phase_verify(
    cfg: harness.RunConfig, phase: str, prefix: str, shards: list[tuple[int, int]], arms: list[V.Variant]
) -> None:
    """Every arm under ``prefix`` holds exactly C0's values in every array of every listed shard.

    Layouts must be lossless: a band-first array is read back and reordered before the comparison, and
    floats compare with NaN equal to NaN. Any mismatch raises after the counts are recorded, so the
    phase is never marked done and a rerun checks again.
    """

    def zone(arm: V.Variant) -> zarr.Group:
        repo = _open_repo(harness.store_uri(cfg, f"{prefix}/{arm.name}"))
        return cast(zarr.Group, zarr.open_group(repo.readonly_session(branch="main").store, mode="r")["zone"])

    ref = zone(FINALISTS[0])
    others = [(a, zone(a)) for a in arms if a.name != FINALISTS[0].name]
    mismatched = dict.fromkeys((a.name for a, _ in others), 0)
    for sy, sx in shards:
        ys, xs = slice(sy * TILE, (sy + 1) * TILE), slice(sx * TILE, (sx + 1) * TILE)
        for var, ref_arr in ref.arrays():
            if ref_arr.ndim < 3:
                continue
            want = V.read_window(ref_arr, 0, ys, xs)
            for arm, grp in others:
                got = V.read_window(cast(zarr.Array, grp[var]), 0, ys, xs)
                if not np.array_equal(want, got, equal_nan=want.dtype.kind == "f"):
                    mismatched[arm.name] += 1
                    logger.warning("verify %s: %s differs from C0 in shard (%d, %d)", arm.name, var, sy, sx)
    for name, n in mismatched.items():
        harness.emit_metric(
            cfg, TEST, phase, "mismatched_arrays", n, "count", variant=name, prefix=prefix, shards=len(shards)
        )
        logger.info("verify %s under %s: %d mismatched array-shards over %d shards", name, prefix, n, len(shards))
    if failed := {name: n for name, n in mismatched.items() if n}:
        raise RuntimeError(f"layouts are not lossless under {prefix}: mismatched array-shards {failed}")


def main() -> int:
    """Parse args and run the one phase asked for."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    parser.add_argument("--tiles", type=Path, default=None, help="Directory of real v2 tiles (.npy), for build/read.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--staging", default=None, help="A staged run directory, for assemble and its verify.")
    parser.add_argument("--extent", default=None, help="The staged run's ROI extent 'ny,nx', for assemble.")
    parser.add_argument("--n-tiles", type=int, default=64)
    parser.add_argument("--study", choices=STUDIES, default="band16", help="Which arms and workloads to run.")
    parser.add_argument("--network", choices=NETWORKS, default=None, help="The shaped link, for --phase mobile.")
    args = parser.parse_args()
    cfg = harness.config_from_args(args)
    harness.configure_logging()
    phase, cfg = cfg.phase, dataclasses.replace(cfg, phase=None)  # run_phase filters on the name it is given
    study = STUDIES[args.study]
    # Each phase runs the arms it can. The library's assembly writes zarr's default order, and writes a band-first
    # array only where its shard writer says it can (the band-first prototype of ADR 031 sets WRITES_BAND_FIRST);
    # the other arms' write cost comes from `build`.
    band_first_ok = getattr(shard_writer, "WRITES_BAND_FIRST", False)
    assembled = [a for a in study.arms if a.write_order == "morton" and (band_first_ok or not a.band_first)]
    if phase == "build":
        harness.run_phase(cfg, TEST, "build", lambda: phase_build(cfg, study.arms, args.tiles, args.workers))
    elif phase == "read":
        harness.run_phase(
            cfg,
            TEST,
            f"read_{CLIENT}",
            lambda: phase_read(cfg, f"read_{CLIENT}", args.tiles, study.read, study.arms),
        )
    elif phase == "tune":
        arms = [a for a in study.arms if a.write_order == "morton"]
        harness.run_phase(
            cfg, TEST, f"tune_{CLIENT}", lambda: phase_read(cfg, f"tune_{CLIENT}", args.tiles, study.tune, arms)
        )
    elif phase == "points":
        harness.run_phase(
            cfg,
            TEST,
            f"points_{CLIENT}",
            lambda: phase_read(cfg, f"points_{CLIENT}", args.tiles, study.points, study.arms),
        )
    elif phase == "mobile":
        if not study.mobile or args.network is None:
            raise SystemExit("--phase mobile needs a study with mobile workloads (--study narrow) and --network")
        name = f"mobile_{args.network}_{CLIENT}"
        harness.run_phase(cfg, TEST, name, lambda: phase_mobile(cfg, name, study, args.tiles, args.network))
    elif phase == "assemble":
        extent = tuple(int(v) for v in args.extent.split(","))
        harness.run_phase(
            cfg,
            TEST,
            "assemble",
            lambda: phase_assemble(cfg, args.staging, extent, args.n_tiles, args.workers, assembled),
        )
    elif phase == "verify":
        from tessera_embeddings.inference.chunk_spec import parse_chunk_label

        # Each set names its own stores, shards and arms, so a check can never open the other set's stores.
        if args.staging:  # the assembled set's first pass: the tiles assembly wrote, in the arms it wrote them to
            prefix, arms = f"{ASSEMBLED}/p0", assembled
            shards = [parse_chunk_label(label) for label in _picks(args.staging, args.n_tiles)]
        else:  # the built set: two full cycles of the tiles, each tile in two places
            prefix, arms = BUILT, study.arms
            shards = [divmod(i, _side(cfg)) for i in range(min(16, _side(cfg) ** 2))]
        name = f"verify_{prefix.replace('/', '_')}"
        harness.run_phase(cfg, TEST, name, lambda: phase_verify(cfg, name, prefix, shards, arms))
    else:
        raise SystemExit("--phase must be build, read, points, tune, mobile, assemble or verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
