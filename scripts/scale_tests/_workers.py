"""Picklable multiprocessing worker entrypoints (spawn-safe).

These live in an importable module (never ``__main__``) so pickling-by-qualified
-name works under the ``spawn`` start method — the only safe one for icechunk
(``fork`` deadlocks its tokio runtime). Workers take plain dict payloads (paths,
indices, scalars), never closures, and return plain dicts.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any

import icechunk
import numpy as np
import zarr

from scale_tests import synth
from tessera_embeddings.storage import zarr_store


def _open_repo(store_uri: str, max_concurrent_requests: int | None = None) -> icechunk.Repository:
    """Open a repo directly from a URI (workers have no ``RunConfig``)."""
    return icechunk.Repository.open(
        zarr_store._create_storage(store_uri),
        config=zarr_store._default_repo_config(max_concurrent_requests),
    )


def write_fork(payload: dict[str, Any]) -> Any:  # noqa: ANN401 — returns an icechunk ForkSession
    """Write assigned chunks into a forked session and return it for merge.

    The cooperative-write worker (the T1/T2/T3 store builder): receives a
    pickled ``ForkSession``, writes its share of spatial chunks (all bands) for
    one year into ``group/embeddings`` and ``group/scales`` using deterministic
    synth data, and returns the fork so the coordinator can ``merge`` it.

    Payload keys: ``fork`` (ForkSession), ``group``, ``year_index``,
    ``chunks`` (list of ``[yc, xc]`` spatial chunk indices), ``chunk_yx``
    (``[cy, cx]`` chunk size), ``zone_hw`` (``[ny, nx]`` clamp bounds),
    ``band``, ``seed``.
    """
    fork = payload["fork"]
    group = payload["group"]
    year = int(payload["year_index"])
    cy, cx = payload["chunk_yx"]
    ny, nx = payload["zone_hw"]
    band = int(payload["band"])
    seed = int(payload["seed"])

    root = zarr.open_group(fork.store, mode="a")
    emb = root[group]["embeddings"]
    scl = root[group]["scales"]
    for yc, xc in payload["chunks"]:
        y0, x0 = yc * cy, xc * cx
        y1, x1 = min(y0 + cy, ny), min(x0 + cx, nx)
        h, w = y1 - y0, x1 - x0
        idx = (year, yc, xc)
        emb[year : year + 1, y0:y1, x0:x1, :] = synth.embedding_block((1, h, w, band), seed=seed, block_index=idx)
        scl[year : year + 1, y0:y1, x0:x1] = synth.scales_block((1, h, w), seed=seed, block_index=idx)
    return fork


def write_fork_shards(payload: dict[str, Any]) -> Any:  # noqa: ANN401 — returns a ForkSession
    """Shard-aligned variant of :func:`write_fork` (ADR-008 D3).

    One full shard-sized block per assigned shard in a single array assignment,
    so the Zarr sharding codec emits each shard object once with no
    read-modify-write. Two modes:

    * **dense** (``land`` absent) — writes synth data across the whole shard
      region (ocean included); the faithful "write everything" comparison point.
    * **land-masked** (``land`` = list of ``[yc, xc]`` land inner-chunk indices,
      with ``chunk_yx``) — writes synth data only into land inner chunks and
      leaves ocean inner chunks at the fill value, so the codec elides them.
      This is the production-representative writer: one lean shard object per
      shard, no dense nodata.

    Payload keys mirror :func:`write_fork` plus ``shards`` (``[sy, sx]``),
    ``shard_yx`` (``[shard_y, shard_x]``), and optionally ``land`` + ``chunk_yx``.
    """
    fork = payload["fork"]
    group = payload["group"]
    year = int(payload["year_index"])
    sh_y, sh_x = payload["shard_yx"]
    ny, nx = payload["zone_hw"]
    band = int(payload["band"])
    seed = int(payload["seed"])
    land = payload.get("land")
    land_set = {tuple(c) for c in land} if land is not None else None
    cy, cx = payload.get("chunk_yx", (0, 0))

    root = zarr.open_group(fork.store, mode="a")
    emb = root[group]["embeddings"]
    scl = root[group]["scales"]
    for sy, sx in payload["shards"]:
        y0, x0 = sy * sh_y, sx * sh_x
        y1, x1 = min(y0 + sh_y, ny), min(x0 + sh_x, nx)
        h, w = y1 - y0, x1 - x0
        if land_set is None:
            idx = (year, sy, sx)
            emb_block = synth.embedding_block((1, h, w, band), seed=seed, block_index=idx)
            scl_block = synth.scales_block((1, h, w), seed=seed, block_index=idx)
        else:
            # Fill (0 / NaN) everywhere, then paint land inner chunks; all-fill
            # ocean inner chunks are elided by the sharding codec on write.
            emb_block = np.zeros((1, h, w, band), dtype="int8")
            scl_block = np.full((1, h, w), np.float32("nan"), dtype="float32")
            for yc, xc in land_set:
                if yc * cy // sh_y != sy or xc * cx // sh_x != sx:
                    continue  # inner chunk not in this shard
                iy0, ix0 = yc * cy - y0, xc * cx - x0
                ih, iw = min(cy, h - iy0), min(cx, w - ix0)
                bidx = (year, yc, xc)
                emb_block[0, iy0 : iy0 + ih, ix0 : ix0 + iw, :] = synth.embedding_block(
                    (1, ih, iw, band), seed=seed, block_index=bidx
                )[0]
                scl_block[0, iy0 : iy0 + ih, ix0 : ix0 + iw] = synth.scales_block(
                    (1, ih, iw), seed=seed, block_index=bidx
                )[0]
        emb[year : year + 1, y0:y1, x0:x1, :] = emb_block
        scl[year : year + 1, y0:y1, x0:x1] = scl_block
    return fork


def _solver(kind: str) -> icechunk.ConflictSolver:
    """Map a payload solver name to an icechunk conflict solver."""
    if kind == "detector":
        return icechunk.ConflictDetector()
    if kind == "useours":
        return icechunk.BasicConflictSolver(on_chunk_conflict=icechunk.VersionSelection.UseOurs)
    raise ValueError(f"unknown solver {kind!r}")


def _slices(region: list[list[int]]) -> tuple[slice, ...]:
    """Turn ``[[start, stop], ...]`` into a tuple of slices."""
    return tuple(slice(a, b) for a, b in region)


def commit_to_group(payload: dict[str, Any]) -> dict[str, Any]:
    """Write one region into ``group/array`` and commit with a rebase-retry loop.

    Independent-session (uncooperative) write: the worker opens its own repo and
    session, writes deterministic synth data into the region, and commits to
    ``main``, manually rebasing on :class:`icechunk.ConflictError` so the retry
    count is observable. Returns a result dict (never raises for an expected
    conflict outcome).

    Payload keys: ``store_uri``, ``group``, ``array``, ``region`` (list of
    ``[start, stop]`` per dim), ``seed``, ``rebase_tries``, ``solver``
    (``"detector"``/``"useours"``), ``jitter_s``, ``max_concurrent_requests``,
    and optionally ``barrier`` — a ``Manager().Barrier`` proxy that all workers
    ``wait`` on after writing but before committing, forcing real commit-time
    contention (the wall-clock alternative loses to staggered spawn latency).
    """
    group = payload["group"]
    array = payload["array"]
    region = _slices(payload["region"])
    solver = _solver(payload.get("solver", "detector"))
    rebase_tries = int(payload.get("rebase_tries", 100))
    jitter_s = float(payload.get("jitter_s", 0.05))

    repo = _open_repo(payload["store_uri"], payload.get("max_concurrent_requests"))
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    arr = root[group][array]

    shape = tuple(sl.stop - sl.start for sl in region)
    block_index = tuple(sl.start for sl in region)
    data = synth.embedding_block(shape, seed=int(payload["seed"]), block_index=block_index)
    arr[region] = data

    # All workers reach here with sessions based on the same seed snapshot, then
    # release together so their commits genuinely race the branch-tip CAS.
    barrier = payload.get("barrier")
    if barrier is not None:
        try:
            barrier.wait(timeout=120)
        except threading.BrokenBarrierError:
            pass

    retries = 0
    unresolvable = False
    error = ""
    snapshot_id = ""
    t0 = time.monotonic()
    while True:
        try:
            snapshot_id = session.commit(f"{group} region {block_index}")
            break
        except icechunk.ConflictError:
            retries += 1
            if retries > rebase_tries:
                error = "exceeded rebase_tries"
                break
            try:
                session.rebase(solver)
            except icechunk.RebaseFailedError as exc:
                unresolvable = True
                error = str(exc)
                break
            time.sleep(random.uniform(0, jitter_s))
    commit_wall_s = time.monotonic() - t0

    return {
        "group": group,
        "ok": bool(snapshot_id) and not unresolvable,
        "retries": retries,
        "commit_wall_s": commit_wall_s,
        "snapshot_id": snapshot_id,
        "unresolvable": unresolvable,
        "error": error,
    }
