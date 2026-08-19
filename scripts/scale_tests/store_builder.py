"""Cooperative (fork/merge) store filler shared by T1, T2, and T3.

One commit per (zone, year): the coordinator forks a writable session, ships the
fork to N spawned workers that each write a disjoint share of the year's land
chunks, then merges the returned forks and commits once (ADR D6). This is the
production write shape the campaign will use, exercised at test scale.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import zarr

from scale_tests import harness
from scale_tests import variants as V
from scale_tests._workers import write_fork, write_fork_shards
from scale_tests.synth import land_mask
from scale_tests.zone_geometry import YEARS, MockZone
from tessera_embeddings.storage.shard_writer import partition_round_robin as _partition

logger = logging.getLogger("scale_tests.builder")


@dataclasses.dataclass(frozen=True)
class ChunkGrid:
    """The spatial/band chunk grid of a variant over a zone."""

    n_y: int
    n_x: int
    n_band: int
    cy: int
    cx: int

    @property
    def n_spatial(self) -> int:
        """Total spatial chunks (before land masking)."""
        return self.n_y * self.n_x


def chunk_grid(zone: MockZone, variant: V.Variant) -> ChunkGrid:
    """Compute the chunk grid for ``variant`` over ``zone``."""
    _, cy, cx, _ = variant.chunks
    return ChunkGrid(
        n_y=math.ceil(zone.height / cy),
        n_x=math.ceil(zone.width / cx),
        n_band=V.band_chunks(variant),
        cy=cy,
        cx=cx,
    )


def refs_per_spatial_chunk(variant: V.Variant) -> int:
    """Chunk refs one written spatial chunk materializes: band chunks + 1 scales."""
    return V.band_chunks(variant) + 1


def land_chunks(zone: MockZone, variant: V.Variant, *, fraction: float, seed: int) -> list[tuple[int, int]]:
    """Return the ``(yc, xc)`` chunk indices that fall on coherent "land"."""
    grid = chunk_grid(zone, variant)
    mask = land_mask(grid.n_y, grid.n_x, fraction=fraction, seed=seed)
    return [(int(yc), int(xc)) for yc, xc in zip(*mask.nonzero(), strict=True)]


def first_n_chunks(zone: MockZone, variant: V.Variant, n: int) -> list[tuple[int, int]]:
    """Return the first ``n`` spatial chunk positions row-major (clamped, logged).

    Decouples the T2 refs sweep from land coverage: it lets a phase request an
    exact chunk count. If the zone has fewer spatial chunks than ``n``, returns
    all of them and logs the shortfall (no silent cap).
    """
    grid = chunk_grid(zone, variant)
    positions = [(yc, xc) for yc in range(grid.n_y) for xc in range(grid.n_x)]
    if n > len(positions):
        logger.warning("requested %d chunks but zone yields only %d; capping", n, len(positions))
    return positions[:n]


def chunks_for_refs(
    zone: MockZone,
    variant: V.Variant,
    *,
    refs_target: int,
    fraction: float,
    seed: int,
) -> list[tuple[int, int]]:
    """Return enough land chunks to materialize ~``refs_target`` chunk refs.

    Fewer are returned if the zone does not contain enough land chunks; the
    caller logs the shortfall rather than silently under-filling.
    """
    per = refs_per_spatial_chunk(variant)
    n_spatial = max(1, math.ceil(refs_target / per))
    land = land_chunks(zone, variant, fraction=fraction, seed=seed)
    return land[:n_spatial]


@dataclasses.dataclass
class FillResult:
    """Outcome of one cooperative year fill."""

    refs_committed: int
    merge_wall_s: float
    commit_wall_s: float
    snapshot_id: str
    n_chunks: int
    n_workers: int


def fill_year(
    cfg: harness.RunConfig,
    store: str,
    group: str,
    variant: V.Variant,
    zone: MockZone,
    year_index: int,
    chunk_list: list[tuple[int, int]],
    *,
    n_workers: int,
    seed: int = 0,
    repo_config=None,  # noqa: ANN001 — optional icechunk.RepositoryConfig
) -> FillResult:
    """Fill one year's ``chunk_list`` into ``group`` via fork/merge, one commit.

    The merge and commit run in this (coordinator) process, so wrapping the call
    in :func:`harness.rss_sampler` captures the coordinator RSS that validates
    the ~400 B/ref model (test plan T2).
    """
    if not chunk_list:
        raise ValueError("chunk_list is empty; nothing to fill")

    repo = harness.open_repo(cfg, store, config=repo_config)
    session = repo.writable_session("main")
    fork = session.fork()

    parts = _partition(chunk_list, n_workers)
    payloads = [
        {
            "fork": fork,
            "group": group,
            "year_index": year_index,
            "chunks": part,
            "chunk_yx": [variant.chunks[1], variant.chunks[2]],
            "zone_hw": [zone.height, zone.width],
            "band": V.BAND,
            "seed": seed,
        }
        for part in parts
    ]

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as ex:
        forks = list(ex.map(write_fork, payloads))

    merge_start = time.monotonic()
    session.merge(*forks)
    merge_wall = time.monotonic() - merge_start

    # Advance years_complete in the same commit as the data (ADR D1).
    root = zarr.open_group(session.store, mode="a")
    grp = root[group]
    done = list(grp.attrs.get("years_complete", []))
    year_label = YEARS[year_index]
    if year_label not in done:
        done.append(year_label)
        grp.attrs["years_complete"] = done

    commit_start = time.monotonic()
    snapshot_id = session.commit(f"fill {group} year {year_label} ({len(chunk_list)} chunks)")
    commit_wall = time.monotonic() - commit_start

    refs = len(chunk_list) * refs_per_spatial_chunk(variant)
    logger.info(
        "filled %s/%s year %d: %d chunks, ~%d refs, merge %.2fs commit %.2fs",
        store,
        group,
        year_label,
        len(chunk_list),
        refs,
        merge_wall,
        commit_wall,
    )
    return FillResult(
        refs_committed=refs,
        merge_wall_s=merge_wall,
        commit_wall_s=commit_wall,
        snapshot_id=snapshot_id,
        n_chunks=len(chunk_list),
        n_workers=len(payloads),
    )


def shards_for_chunks(variant: V.Variant, chunk_list: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the ``(sy, sx)`` shard indices that contain >=1 of ``chunk_list``."""
    if variant.shards is None:
        raise ValueError(f"{variant.name} is not sharded")
    _, sh_y, sh_x, _ = variant.shards
    _, cy, cx, _ = variant.chunks
    shards = {(yc * cy // sh_y, xc * cx // sh_x) for yc, xc in chunk_list}
    return sorted(shards)


def fill_year_shard_aligned(
    cfg: harness.RunConfig,
    store: str,
    group: str,
    variant: V.Variant,
    zone: MockZone,
    year_index: int,
    shard_list: list[tuple[int, int]],
    *,
    n_workers: int,
    seed: int = 0,
    land_chunks: list[tuple[int, int]] | None = None,
    repo_config=None,  # noqa: ANN001 — optional icechunk.RepositoryConfig
) -> FillResult:
    """Fill one year **shard-aligned**: one full-shard-block write per shard.

    The D3/E2 counterpart to :func:`fill_year` — each shard object is emitted
    once by the sharding codec (no read-modify-write). Requires a sharded
    ``variant``. With ``land_chunks`` given, writes synth data only into those
    inner chunks and leaves ocean inner chunks at fill (elided by the codec) —
    the production-representative writer (lean shards, no dense nodata). Without
    it, writes dense shards (the E2 "write everything" comparison point).
    """
    if variant.shards is None:
        raise ValueError(f"{variant.name} is not sharded; use fill_year")
    if not shard_list:
        raise ValueError("shard_list is empty; nothing to fill")

    repo = harness.open_repo(cfg, store, config=repo_config)
    session = repo.writable_session("main")
    fork = session.fork()
    parts = _partition(shard_list, n_workers)
    land = [list(c) for c in land_chunks] if land_chunks is not None else None
    payloads = [
        {
            "fork": fork,
            "group": group,
            "year_index": year_index,
            "shards": part,
            "shard_yx": [variant.shards[1], variant.shards[2]],
            "chunk_yx": [variant.chunks[1], variant.chunks[2]],
            "zone_hw": [zone.height, zone.width],
            "band": V.BAND,
            "seed": seed,
            "land": land,
        }
        for part in parts
    ]

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as ex:
        forks = list(ex.map(write_fork_shards, payloads))

    merge_start = time.monotonic()
    session.merge(*forks)
    merge_wall = time.monotonic() - merge_start

    # Advance years_complete in the same commit as the data (ADR D1), mirroring
    # fill_year so both fill paths expose identical completion metadata.
    root = zarr.open_group(session.store, mode="a")
    grp = root[group]
    done = list(grp.attrs.get("years_complete", []))
    if YEARS[year_index] not in done:
        grp.attrs["years_complete"] = sorted([*done, YEARS[year_index]])

    commit_start = time.monotonic()
    snapshot_id = session.commit(f"shard-aligned fill {group} year {YEARS[year_index]} ({len(shard_list)} shards)")
    commit_wall = time.monotonic() - commit_start

    logger.info(
        "shard-aligned filled %s/%s: %d shards, merge %.2fs commit %.2fs",
        store,
        group,
        len(shard_list),
        merge_wall,
        commit_wall,
    )
    return FillResult(
        refs_committed=len(shard_list),  # one shard object per shard
        merge_wall_s=merge_wall,
        commit_wall_s=commit_wall,
        snapshot_id=snapshot_id,
        n_chunks=len(shard_list),
        n_workers=len(payloads),
    )
