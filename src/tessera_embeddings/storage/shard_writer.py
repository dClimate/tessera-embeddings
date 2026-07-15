"""Shard-aligned, land-masked writer for the global store (ADR-008 D3/D6).

Fills one (zone, year) with whole shards in a single Icechunk commit. The
d3v2-verified write path: each worker reads its assigned shard *sources* and
writes each shard's region in one raw-zarr assignment - real data in land inner
chunks, fill (elided by the sharding codec) elsewhere - so every shard object is
emitted once, no read-modify-write, no dense nodata.

Cooperative fork/merge: the coordinator forks the session, workers write into
their fork, the coordinator merges and makes **one commit per (zone, year)**,
updating ``years_complete`` in the same commit (D1). Commits go behind a
:class:`CommitGate` so no more than a handful land at once - uncoordinated
concurrent commits storm (run-1 T5). Fleet-wide gating is the orchestrator's job
(Prefect global concurrency limit); this module enforces it within a process.

A :class:`ShardSource` decouples the writer from *where* shard data comes from
(staged inference files in production; synthetic in tests), and must be picklable
so it can be shipped to spawned workers.
"""

from __future__ import annotations

import multiprocessing
import threading
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import Any, Protocol, cast, runtime_checkable

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.store_layout import SHARD_PX


@runtime_checkable
class ShardSource(Protocol):
    """Supplies the shard data for one (zone, year) fill.

    Implementations must be picklable (a frozen dataclass) so the writer can ship
    them to spawned workers.
    """

    def live_shards(self) -> Iterable[tuple[int, int]]:
        """Return the ``(sy, sx)`` shard indices that have data to write."""
        ...

    def load(self, shard: tuple[int, int]) -> dict[str, np.ndarray]:
        """Return ``{var_name: block}`` for a shard - each block covers the whole
        (edge-clamped) shard region with ocean inner chunks at the array fill
        value. Return ``{}`` to skip a shard entirely.
        """
        ...


class CommitGate(Protocol):
    """Context manager limiting how many commits proceed at once."""

    def __enter__(self) -> Any: ...  # noqa: ANN401, D105
    def __exit__(self, *exc: object) -> None: ...  # noqa: D105


class SemaphoreCommitGate:
    """In-process :class:`CommitGate` backed by a semaphore (default cap 6).

    6 is the middle of the run-1-mandated 4-8 simultaneous committers. Share one
    instance across the zone-year fills a single process drives; the campaign's
    cross-machine gate is a Prefect global concurrency limit (ADR-008 D6, Q5).
    """

    def __init__(self, max_concurrent: int = 6) -> None:
        """Create a gate allowing ``max_concurrent`` commits at once."""
        self._sem = threading.Semaphore(max_concurrent)

    def __enter__(self) -> SemaphoreCommitGate:
        """Acquire a commit slot (blocks if the cap is reached)."""
        self._sem.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the commit slot."""
        self._sem.release()


def commit_with_rebase(session: icechunk.Session, message: str, *, tries: int = 1000) -> str:
    """Commit, auto-rebasing on a moved branch tip; return the snapshot id.

    Uses icechunk's built-in rebase loop with a :class:`ConflictDetector` - enough
    for our write model, where concurrent commits touch disjoint groups/regions
    and always rebase cleanly (run-1 T0/T5: zero unresolvable conflicts). A real
    chunk conflict surfaces as ``RebaseFailedError`` rather than being masked.
    """
    return session.commit(message, rebase_with=icechunk.ConflictDetector(), rebase_tries=tries)


def _group_node(store: Any, group: str) -> zarr.Group:  # noqa: ANN401 — icechunk store handle
    """Open a repo's zarr root and return the named group node (typed as Group)."""
    return cast(zarr.Group, zarr.open_group(store, mode="a")[group])


def _year_label(node: zarr.Group, year_index: int) -> int:
    """Read the calendar year at ``year_index`` from a group's time coordinate."""
    time_arr = cast(zarr.Array, node["time"])
    t = np.asarray(time_arr[year_index]).astype("datetime64[ns]")
    return int(t.astype("datetime64[Y]").astype(int)) + 1970


def _partition(items: list, n: int) -> list[list]:
    """Round-robin partition ``items`` into up to ``n`` non-empty lists."""
    parts = [items[i::n] for i in range(n)]
    return [p for p in parts if p]


def _write_shards_worker(payload: dict[str, Any]) -> Any:  # noqa: ANN401 - returns a ForkSession
    """Write assigned shards into a forked session and return it for merge."""
    fork = payload["fork"]
    group = payload["group"]
    year = int(payload["year_index"])
    shard_px = int(payload["shard_px"])
    source: ShardSource = payload["source"]

    node = _group_node(fork.store, group)
    for sy, sx in payload["shards"]:
        blocks = source.load((sy, sx))
        for var, block in blocks.items():
            arr = cast(zarr.Array, node[var])
            y0, x0 = sy * shard_px, sx * shard_px
            h, w = block.shape[1], block.shape[2]
            if arr.ndim == 4:
                arr[year : year + 1, y0 : y0 + h, x0 : x0 + w, :] = block
            else:
                arr[year : year + 1, y0 : y0 + h, x0 : x0 + w] = block
    return fork


def write_year_shards(
    repo: icechunk.Repository,
    group: str,
    year_index: int,
    source: ShardSource,
    *,
    n_workers: int = 1,
    gate: CommitGate | None = None,
    shard_px: int = SHARD_PX,
    commit_msg: str | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> str:
    """Fill one (zone, year) with whole shards from ``source`` in one commit.

    Forks the session, writes the source's live shards across ``n_workers``
    (in-process when 1, else spawned processes), merges, advances
    ``years_complete``, and commits behind ``gate`` via :func:`commit_with_rebase`.
    ``extra_attrs`` (e.g. per-fill run provenance) is merged into the group's
    attrs in the same commit.

    Concurrency contract: concurrent commits to *different* groups rebase
    cleanly (disjoint nodes), but two concurrent fills of the *same* group both
    rewrite its ``years_complete``/``extra_attrs`` and
    :class:`icechunk.ConflictDetector` cannot auto-merge attribute conflicts —
    the loser fails with ``RebaseFailedError`` (loud, retriable) rather than
    silently dropping an update. Orchestrate one fill per zone at a time.
    Returns the commit snapshot id.
    """
    session = repo.writable_session("main")
    fork = session.fork()
    shards = list(source.live_shards())
    if not shards:
        raise ValueError(f"source has no live shards for {group} year_index={year_index}")

    parts = _partition(shards, max(1, n_workers))
    payloads = [
        {"fork": fork, "group": group, "year_index": year_index, "shards": part, "source": source, "shard_px": shard_px}
        for part in parts
    ]
    if len(parts) == 1:
        forks = [_write_shards_worker(payloads[0])]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(parts), mp_context=ctx) as ex:
            forks = list(ex.map(_write_shards_worker, payloads))

    session.merge(*forks)

    node = _group_node(session.store, group)
    year_label = _year_label(node, year_index)
    raw = node.attrs.get("years_complete", [])
    done: list[int] = [int(y) for y in raw] if isinstance(raw, list) else []
    if year_label not in done:
        node.attrs["years_complete"] = sorted([*done, year_label])
    if extra_attrs:
        node.attrs.update(extra_attrs)

    with gate if gate is not None else nullcontext():
        return commit_with_rebase(session, commit_msg or f"fill {group} year {year_label}")
