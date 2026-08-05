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
    existing: object,
    year: int,
    run_id: str,
    *,
    empty: bool = False,
    radar_coverage: dict | None = None,
) -> dict:
    """Merge a per-year run record into a group's ``runs`` attr (the schema's one owner).

    Both fill paths use this — the shard write (:func:`write_year_shards`) and
    the no-data marking (``campaign.mark_zone_year_empty``) — so the provenance
    record shape can only change in one place. The record carries no window: the
    store GUARANTEES calendar-year slots (the zone-fill gate rejects any window
    that is not exactly Jan-Dec of the slot's year), and each slot's true interval
    is stated by the seeded ``time_bnds`` CF-bounds variable.

    ``radar_coverage`` records how much of the year's embedded area had no radar, or
    little of it. It belongs PER YEAR rather than per zone because radar coverage is a
    property of what was acquired, not of the terrain: one year of a zone can be
    radar-free where another is not, so a zone-level figure would be wrong for at least
    one of them. Exact per-pixel counts already live in the store's
    ``s1_asc_obs_count``/``s1_desc_obs_count`` arrays; this is the summary that makes the
    question answerable without reading a zone-sized grid.
    """
    record: dict = {"run_id": run_id, "assembled_at": datetime.now(UTC).isoformat()}
    if empty:
        record["empty"] = True
    if radar_coverage:
        record["radar_coverage"] = dict(radar_coverage)
    return {**(dict(existing) if isinstance(existing, dict) else {}), str(year): record}


def commit_year_attrs(
    repo: icechunk.Repository,
    group: str,
    year_label: int,
    *,
    run_id: str | None = None,
    empty: bool = False,
    radar_coverage: dict | None = None,
    gate: CommitGate | None = None,
    tries: int = 8,
    skip_if_marked: bool = False,
) -> str:
    """Advance one year's ``years_complete``/``runs`` in its own small commit, retrying.

    The single writer of those two attrs, and the reason concurrent fills of the SAME
    zone group are safe.

    **Why a separate commit.** Chunk data for different years of one zone is strictly
    disjoint — every chunk and shard is 1 in the time dimension — so those writes always
    rebase cleanly. The only thing that ever collided was these two attrs, because
    icechunk's :class:`~icechunk.ConflictDetector` treats attributes as an opaque value
    and cannot merge them. Bundling them into the shard commit meant a collision threw
    away the whole assembly; here it throws away a sub-second commit.

    **Why re-reading and retrying is CORRECT rather than hopeful.** Both attrs are keyed
    by year and each writer only ever inserts its OWN key — ``years_complete`` is a set
    union, ``runs`` a per-year dict insert. So there is no semantic conflict to resolve:
    a loser that re-reads the winner's value and re-applies its own key produces exactly
    the state both writers intended, in either order. That is what makes this a plain
    optimistic-concurrency loop rather than a lossy merge. Each attempt opens a FRESH
    session, so it cannot re-apply onto a stale read.

    **The two callers want different idempotency, and the difference is deliberate.**
    ``skip_if_marked=True`` returns the branch tip untouched when the year is already in
    ``years_complete``, EVEN IF a different ``run_id`` was passed — that is what
    ``mark_zone_year_empty`` needs, because a re-mark of an empty cell must not mint a new
    snapshot: :func:`~tessera_embeddings.storage.campaign.tag_zone_year` refuses to move a
    tag, so a new snapshot would leave the existing zone-year tag pointing at an ancestor
    and the original provenance is the one worth keeping. The default (``False``) records
    the new run, which is what a genuine refill through :func:`write_year_shards` means:
    shards were rewritten, so the provenance should say by which run.
    """
    for attempt in range(1, tries + 1):
        session = repo.writable_session("main")
        node = _group_node(session.store, group)
        done = read_years_complete(node)
        if year_label in done and (run_id is None or skip_if_marked):
            return repo.lookup_branch("main")
        if year_label not in done:
            node.attrs["years_complete"] = sorted([*done, year_label])
        if run_id is not None:
            node.attrs["runs"] = run_provenance(
                node.attrs.get("runs"), year_label, run_id, empty=empty, radar_coverage=radar_coverage
            )
        try:
            return commit_with_rebase(session, f"mark {group} year {year_label} complete", gate=gate)
        except icechunk.RebaseFailedError:
            if attempt == tries:
                raise
            # Another year of THIS group committed between our read and our commit. Re-read
            # and re-apply; the loop is bounded so a genuine defect still surfaces.
            continue
    raise AssertionError("unreachable")  # pragma: no cover


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
    radar_coverage: dict | None = None,
) -> str:
    """Fill one (zone, year) with whole shards from ``source`` in one commit.

    Forks the session, writes the source's live shards across ``n_workers``
    (in-process when 1, else spawned processes), merges, advances
    ``years_complete``, and commits behind ``gate`` via :func:`commit_with_rebase`.
    When ``run_id`` is given, per-year run provenance (:func:`run_provenance`)
    is merged into the group's ``runs`` attr in the same commit — read and
    written inside THIS writable session, so a commit landing between a
    caller's earlier probe and this write cannot be silently clobbered.

    Concurrency contract (RELAXED 2026-07-30): concurrent fills of different groups
    rebase cleanly, and concurrent fills of the SAME group but different years are
    now safe too. This used to require "one fill per zone at a time", which is what
    made the campaign's years serial. Two changes to that reasoning:

    * Chunk data was never the problem — every chunk and shard is 1 in the time
      dimension, so different years of one zone write strictly disjoint objects.
    * The two group attrs that DID collide now commit separately and retry, via
      :func:`commit_year_attrs`, which is correct because each writer only inserts
      its own year's key.

    So this issues TWO commits: the shards, then the year's attrs. A consequence
    worth knowing: if the shard commit lands and the attr commit then exhausts its
    retries, the year holds data but is not marked complete. The work list reads the
    marks, so that cell simply looks pending and a retry re-writes the same shards
    (a whole-shard overwrite) and re-marks. That is strictly better than the previous
    behaviour, where a collision discarded the shards as well.

    Returns the ATTR commit's snapshot id — a tag must point at a state where the
    year is both written and marked.
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

    year_label = _year_label(_group_node(session.store, group), year_index)
    commit_with_rebase(session, commit_msg or f"fill {group} year {year_label}", gate=gate)
    # The per-year attrs go in their OWN commit (see `commit_year_attrs`), so a same-zone
    # collision costs a sub-second retry instead of this whole assembly. Return that
    # snapshot rather than the shard one: a tag must point at a state where the year is
    # both written AND marked.
    return commit_year_attrs(repo, group, year_label, run_id=run_id, radar_coverage=radar_coverage, gate=gate)
