"""Shard-aligned, land-masked writer for the global store (ADR-008 D3/D6).

Fills one (zone, year) with whole shards in a single Icechunk commit. The
d3v2-verified write path: each worker reads its assigned shard *sources* and
writes each shard's region in one raw-zarr assignment - real data in land inner
chunks, fill (elided by the sharding codec) elsewhere - so every shard object is
emitted once, no read-modify-write, no dense nodata.

Cooperative fork/merge: the coordinator forks the session, workers write into
their fork, the coordinator merges and makes **one commit per (zone, year)**,
updating ``years_complete`` in the same commit (D1). Commits go behind a
``gate`` (any context manager — e.g. ``threading.Semaphore(DEFAULT_COMMIT_CAP)``)
so no more than a handful land at once - uncoordinated concurrent commits storm
(run-1 T5). Fleet-wide gating is the orchestrator's job (Prefect global
concurrency limit); the gate enforces it within a process.

A :class:`ShardSource` decouples the writer from *where* shard data comes from
(staged inference files in production; synthetic in tests), and must be picklable
so it can be shipped to spawned workers. :func:`run_forked` is the shared
fork → parallel-write → merge scaffolding; the single-ROI assembly engine
(:mod:`tessera_embeddings.inference.assembly`) drives it with a different
worker body.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from typing import Any, Protocol, cast

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage.zarr_store import read_time_values
from tessera_embeddings.storage.zone_grid import year_of

#: Any context manager works as a commit gate (``threading.Semaphore`` is the
#: canonical in-process one — acquire on enter, release on exit).
CommitGate = AbstractContextManager

#: Default in-process commit cap: the middle of the run-1-mandated 4-8
#: simultaneous committers (ADR-008 D6). Share one
#: ``threading.Semaphore(DEFAULT_COMMIT_CAP)`` across the zone-year fills a
#: single process drives; the campaign's cross-machine gate is a Prefect
#: global concurrency limit (Q5).
DEFAULT_COMMIT_CAP = 6


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


def commit_with_rebase(
    session: icechunk.Session,
    message: str,
    *,
    tries: int = 1000,
    gate: CommitGate | None = None,
) -> str:
    """Commit, auto-rebasing on a moved branch tip; return the snapshot id.

    Uses icechunk's built-in rebase loop with a :class:`ConflictDetector` - enough
    for our write model, where concurrent commits touch disjoint groups/regions
    and always rebase cleanly (run-1 T0/T5: zero unresolvable conflicts). A real
    chunk conflict surfaces as ``RebaseFailedError`` rather than being masked.
    When ``gate`` is given the commit proceeds inside it, bounding how many
    commits are in flight at once.
    """
    with gate if gate is not None else nullcontext():
        return session.commit(message, rebase_with=icechunk.ConflictDetector(), rebase_tries=tries)


def shard_pitch(arr: zarr.Array) -> int:
    """An array's northing write granularity: shard height if sharded, else chunk height."""
    return (arr.shards or arr.chunks)[1]


def run_forked(
    session: icechunk.Session,
    worker_fn: Callable[[dict[str, Any]], Any],
    payloads: list[dict[str, Any]],
) -> None:
    """Fork ``session``, run ``worker_fn`` over ``payloads``, merge the forks back.

    The shared coordinator scaffolding for cooperative writes: each payload is
    shipped to ``worker_fn`` with a ``"fork"`` key added (a pickled copy per
    spawned worker; caller dicts are not mutated), the worker writes into its
    fork and returns it, and the coordinator merges. One payload
    runs in-process; more spawn a process pool (``spawn`` context — workers must
    be module-level functions and payloads picklable).
    """
    fork = session.fork()
    # Copies, not mutation: callers keep their payload dicts fork-free.
    payloads = [{**payload, "fork": fork} for payload in payloads]
    if len(payloads) == 1:
        forks = [worker_fn(payloads[0])]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as ex:
            forks = list(ex.map(worker_fn, payloads))
    session.merge(*forks)


def _group_node(store: Any, group: str) -> zarr.Group:  # noqa: ANN401 — icechunk store handle
    """Open a repo's zarr root and return the named group node (typed as Group)."""
    return cast(zarr.Group, zarr.open_group(store, mode="a")[group])


def read_years_complete(node: zarr.Group) -> list[int]:
    """A group's ``years_complete`` attr as a sorted list of ints (the one parser)."""
    raw = node.attrs.get("years_complete", [])
    return sorted(int(y) for y in raw) if isinstance(raw, list) else []


def run_provenance(
    existing: object, year: int, run_id: str, *, empty: bool = False, window_bounds: tuple[str, str] | None = None
) -> dict:
    """Merge a per-year run record into a group's ``runs`` attr (the schema's one owner).

    Both fill paths use this — the shard write (:func:`write_year_shards`) and
    the no-data marking (``campaign.mark_zone_year_empty``) — so the provenance
    record shape can only change in one place.

    ``window_bounds`` is the ``(start, end)`` ISO date range the slot was actually
    filled from (``TimeWindow.to_date_range()`` — first day of the start month, last
    day of the end month, e.g. ``("2025-01-01", "2025-12-31")`` for a calendar year).
    It is recorded here as human-readable provenance; the authoritative machine-readable
    form is the group's ``time_bnds`` CF-bounds variable (written in the same commit).
    The store advertises ``time_convention="calendar_year"`` as the DEFAULT, not a
    guarantee (``time_convention_strict=False``): a fill may write a non-calendar
    12-month window, so these make the true window legible rather than a silent
    mislabel under the calendar-year ``time`` point.
    """
    record: dict = {"run_id": run_id, "assembled_at": datetime.now(UTC).isoformat()}
    if window_bounds is not None:
        record["window"] = list(window_bounds)
    if empty:
        record["empty"] = True
    return {**(dict(existing) if isinstance(existing, dict) else {}), str(year): record}


def write_time_bounds(node: zarr.Group, year_index: int, window_bounds: tuple[str, str]) -> None:
    """Write a slot's real ``[start, end]`` date range into the group's ``time_bnds``.

    No-op when the group has no ``time_bnds`` (e.g. a non-global store). Encodes to the
    same int64-ns representation as the ``time`` coordinate (:data:`TIME_ENCODING`).
    """
    if "time_bnds" not in node:
        return
    start_ns = np.datetime64(window_bounds[0], "ns").astype("int64")
    end_ns = np.datetime64(window_bounds[1], "ns").astype("int64")
    cast("zarr.Array", node["time_bnds"])[year_index] = [start_ns, end_ns]


def _year_label(node: zarr.Group, year_index: int) -> int:
    """Read the calendar year at ``year_index`` from a group's time coordinate.

    Decodes via :func:`read_time_values` so a foreign store with non-TIME_ENCODING
    units errors loudly instead of yielding an epoch-adjacent bogus year.
    """
    return year_of(read_time_values(node)[year_index])


def partition_round_robin(items: list, n: int) -> list[list]:
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
    arrays: dict[str, zarr.Array] = {}
    for sy, sx in payload["shards"]:
        blocks = source.load((sy, sx))
        for var, block in blocks.items():
            arr = arrays.get(var)
            if arr is None:
                arr = arrays[var] = cast(zarr.Array, node[var])
            y0, x0 = sy * shard_px, sx * shard_px
            h, w = block.shape[1], block.shape[2]
            # Trailing dims (band) not indexed are written in full, so one
            # assignment covers both the 3-D and 4-D arrays.
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
    run_id: str | None = None,
    window_bounds: tuple[str, str] | None = None,
) -> str:
    """Fill one (zone, year) with whole shards from ``source`` in one commit.

    Forks the session, writes the source's live shards across ``n_workers``
    (in-process when 1, else spawned processes), merges, advances
    ``years_complete``, and commits behind ``gate`` via :func:`commit_with_rebase`.
    When ``run_id`` is given, per-year run provenance (:func:`run_provenance`)
    is merged into the group's ``runs`` attr in the same commit — read and
    written inside THIS writable session, so a commit landing between a
    caller's earlier probe and this write cannot be silently clobbered.

    Concurrency contract: concurrent commits to *different* groups rebase
    cleanly (disjoint nodes), but two concurrent fills of the *same* group both
    rewrite its ``years_complete``/``runs`` attrs and
    :class:`icechunk.ConflictDetector` cannot auto-merge attribute conflicts —
    the loser fails with ``RebaseFailedError`` (loud, retriable) rather than
    silently dropping an update. Orchestrate one fill per zone at a time.
    Returns the commit snapshot id.
    """
    session = repo.writable_session("main")
    shards = list(source.live_shards())
    if not shards:
        raise ValueError(f"source has no live shards for {group} year_index={year_index}")

    payloads: list[dict[str, Any]] = [
        {"group": group, "year_index": year_index, "shards": part, "source": source, "shard_px": shard_px}
        for part in partition_round_robin(shards, max(1, n_workers))
    ]
    run_forked(session, _write_shards_worker, payloads)

    node = _group_node(session.store, group)
    year_label = _year_label(node, year_index)
    done = read_years_complete(node)
    if year_label not in done:
        node.attrs["years_complete"] = sorted([*done, year_label])
    if window_bounds is not None:
        write_time_bounds(node, year_index, window_bounds)
    if run_id is not None:
        node.attrs["runs"] = run_provenance(node.attrs.get("runs"), year_label, run_id, window_bounds=window_bounds)

    return commit_with_rebase(session, commit_msg or f"fill {group} year {year_label}", gate=gate)
