"""Repair the registry's ``master/``: rebuild cells from their parts and re-derive what fills left null.

Every fill merges its own cell into ``master/`` (``registry_master.merge_cell``, called by the
zone-fill runner). This script is for what that cannot do: re-deriving rows a fill published null,
and rebuilding a cell whose update failed. Its first use repaired the v1.1 campaign, whose resumed
fills published 1,519,045 of 3,247,410 rows (47%) null in every measurement column
(``context_docs/storage/registry-backfill.md``).

Every column is derivable from the store, so this is a re-derivation rather than an estimate — see
:mod:`tessera_embeddings.storage.registry_rebuild` for the proof that the fill's optical test
reduces to ``s2_obs_count > 0`` on the delivered product, and for the one column
(``eligible_px``) deliberately left null.

**``repair`` rebuilds a cell from three inputs**: its parts, its existing master rows, and a fresh
re-derivation of every row its fills left null, merged tile by tile. The parts supply every tile and
every measurement a fill took, which the merge never overrides; the existing master keeps a radar
column an optical-only pass would not re-measure. ``parts/`` is never touched, and a cell whose
re-derivation or store check is blocked is not written at all. Every cell with a part, a master
file or a shard commit is visited: a master file whose last part was deleted is emptied, and a
commit with no part is reported. Each cell is written through the
fill's own ``registry_master.merge_cell``, conditionally, so a fill landing mid-repair is merged in
rather than overwritten. Every writing subcommand is a DRY RUN unless given ``--write``.

Run from the REPOSITORY ROOT::

    # 1. prove the re-derivation reproduces measurements that already exist. Writes nothing, ever.
    uv run python scripts/maintenance/rebuild_registry_measurements.py gate --sample 50000

    # 2. rebuild master/ (every cell, or --zones). --skip-radar is the cheap optical-only pass.
    uv run python scripts/maintenance/rebuild_registry_measurements.py repair --write

    # 3. check master/ itself, cell by cell, against parts/, and parts/ against the store
    uv run python scripts/maintenance/rebuild_registry_measurements.py verify

**An ``s3://`` registry needs ``s3fs``**, from the ``aws`` extra (``uv sync --all-extras`` installs it):
the master write is a conditional PUT, which pyarrow's own S3 filesystem cannot make.

**Rehearsing needs no credentials.** ``--anonymous`` reads the public store and registry unsigned,
and a non-``s3://`` ``--registry`` is the local filesystem, so a downloaded copy of the registry takes
every subcommand end to end — ``repair --write`` included — without touching the bucket.

**One process saturates at three to four cores**, because the per-tile work holds the GIL for part
of its time. A full pass on a bigger host runs several processes over disjoint ``--zones`` lists;
``--zones`` is applied before rows are materialised, so each holds only its own share in memory.

**The radar column can be a second pass.** ``px_with_any_radar`` needs both Sentinel-1
observation-count arrays, which triples the bytes moved for one informational column. ``repair
--skip-radar`` lands the optical answer; a later full ``repair`` replaces it with rows that measured
radar too, and a later optical-only one cannot undo that.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import functools
import json
import random
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fsspec
import icechunk
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import zarr
from fsspec import AbstractFileSystem

from tessera_embeddings.storage import published_store, registry_rebuild
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.registry import dataset_schema
from tessera_embeddings.storage.registry_master import (
    MEASURED_BY_FILL,
    MEASURED_BY_REBUILD,
    MEASURED_SENTINEL,
    as_master_rows,
    audit,
    cell_tables,
    invariant_failures,
    master_schema,
    merge,
    merge_cell,
    store_disagreements,
    tile_years,
)
from tessera_embeddings.storage.registry_rebuild import rebuild_run_id, rebuilt_row

DEFAULT_STORE = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
DEFAULT_REGISTRY = "s3://tessera-embeddings/v1.1/dclimate.registry"
DEFAULT_REGION = "us-west-2"

#: Measured tiles re-derived per cell before that cell's null tiles are written. Only tiles that saw
#: some optical count: in 122 of the 162 partly measured cells every measured tile is a wholly
#: refused one, often never imaged, whose shard is absent and "reproduces" trivially. 87 partly
#: measured cells hold an uncropped tile worth checking; the other 75, and the 299 with no measured
#: tile at all, lean on `gate`.
IN_CELL_GATE_TILES = 64

#: Concurrent tile reads per process. Each tile is one GET of ~0.5 MB per array; from a laptop the
#: link is the limit, in-region the CPU is (see the module docstring on splitting by --zones).
DEFAULT_CONCURRENCY = 32


def _filesystem(root: str, region: str, *, anonymous: bool = False) -> AbstractFileSystem:
    """The filesystem for ``root`` — S3 for an ``s3://`` registry, local for anything else.

    Local is not a convenience: rehearsing a sync against a downloaded copy of the registry is how
    the invariants get checked before anything is published, and a subcommand that only knows
    how to talk to the real bucket cannot be rehearsed at all. fsspec rather than pyarrow's own,
    because a master write is conditional and pyarrow's filesystems cannot make one.
    """
    if root.startswith("s3://"):
        return fsspec.filesystem("s3", anon=anonymous, client_kwargs={"region_name": region})
    return fsspec.filesystem("file")


def _registry_table(
    fs: AbstractFileSystem, root: str, prefix: str, schema: pa.Schema, *, optional: bool = False
) -> pa.Table:
    """One of the registry's prefixes as a table, schema STATED rather than inferred.

    Inferring it from the first file in sorted path order is how a column added mid-campaign
    silently disappears from a whole-dataset read; the registry's own docstring says so and this is
    the script that would be most damaged by it.

    Only an ``optional`` prefix may be absent — ``master/`` before the first repair of a fresh
    registry — and then it reads as an empty table. Anything else that cannot be read raises: an
    empty ``parts/`` from a mistyped ``--registry`` or a failed read would let every subcommand pass
    having checked nothing.
    """
    path = root.removeprefix("s3://").rstrip("/") + "/" + prefix
    if not fs.exists(path):
        if optional:
            return pa.Table.from_pylist([], schema=schema)
        raise SystemExit(f"{root.rstrip('/')}/{prefix} does not exist — is --registry right?")
    table = ds.dataset(path, filesystem=fs, partitioning="hive", schema=schema).to_table()
    if not optional and not table.num_rows:
        raise SystemExit(f"{root.rstrip('/')}/{prefix} holds no rows — refusing to treat that as a registry")
    return table


#: The published store as the checks read it: its repository, its root group read-only, and a
#: read of every tagged shard commit on ``main`` by zone-year (``published_store.shard_writes``),
#: made afresh on each call so a repair sees the commits of fills that land while it runs.
_Store = tuple[icechunk.Repository, zarr.Group, Callable[[], dict[tuple[str, int], list[published_store.Commit]]]]


def _open_store(store_uri: str, region: str, *, anonymous: bool) -> _Store:
    """The published store, opened once for however many cells a subcommand checks."""
    repo = open_global_repo(store_uri, region=region, anonymous=anonymous)
    root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    return repo, root, lambda: published_store.shard_writes(repo)


def _one_read(store: _Store) -> _Store:
    """``store`` with its commit history read once, for a pass that checks a single snapshot."""
    return store[0], store[1], functools.cache(store[2])


def _only_zones(table: pa.Table, zones: list[str]) -> pa.Table:
    """``table`` narrowed to ``zones`` (all of it when none are given), in Arrow.

    Applied before rows become Python dicts: the whole registry as dicts is ~6.7 GB of memory, so a
    run split across processes by ``--zones`` holds only its own share.
    """
    if not zones:
        return table
    table = table.filter(pc.is_in(table.column("zone"), value_set=pa.array(zones)))
    # A mistyped zone would otherwise select nothing and take a no-work success path, leaving that
    # share of a zone-split run unmeasured while every later step passes.
    absent = sorted(set(zones) - set(table.column("zone").unique().to_pylist()))
    if absent:
        raise SystemExit(f"--zones names zones the registry does not hold: {', '.join(absent)}")
    return table


def _cells(parts: pa.Table, zones: list[str]) -> dict[tuple[str, int], pa.Table]:
    """Each ``(zone, year)``'s parts, every run's rows: the unit a fill and a repair work in."""
    parts = _only_zones(parts, zones).sort_by([("zone", "ascending"), ("year", "ascending")])
    return {(zone, year): rows for zone, year, rows in cell_tables(parts)}


def _every_cell(
    fs: AbstractFileSystem, root: str, store: _Store, cells: dict[tuple[str, int], pa.Table], zones: list[str]
) -> dict[tuple[str, int], pa.Table]:
    """``cells`` plus every cell with a master file or a shard commit but no part, narrowed to ``zones``.

    A cell with no part has no rows. A master file outlives its last part when that run is
    withdrawn, and a fill whose part never published leaves only its commit; neither is in
    ``parts/`` for a pass over it to find, so a zone named in ``zones`` need only be in one of them.
    """
    base = root.removeprefix("s3://").rstrip("/")
    held = {
        (match["zone"], int(match["year"]))
        for path in fs.glob(f"{base}/master/zone=*/year=*/part-0.parquet")
        if (match := re.search(r"zone=(?P<zone>[^/]+)/year=(?P<year>\d+)/", path))
    }
    empty = dataset_schema().empty_table()
    every = {
        cell: cells.get(cell, empty) for cell in sorted({*cells, *held, *store[2]()}) if not zones or cell[0] in zones
    }
    if absent := sorted(set(zones) - {zone for zone, _ in every}):
        raise SystemExit(f"--zones names zones neither the registry nor the store holds: {', '.join(absent)}")
    return every


def _current(parts: pa.Table) -> pa.Table:
    """Each tile's current fill row: from the newest run that wrote it, the run the store holds there.

    So a refilled tile is neither gated against an older run's numbers nor re-derived under its rule.
    """
    return merge(as_master_rows(parts, MEASURED_BY_FILL))


def _store_slot(
    store: _Store, zone: str, year: int, parts: pa.Table
) -> tuple[int, Callable[[], published_store.StoreYear]] | str:
    """The cell's time index and a read of its store record, or why the store cannot stand behind its parts.

    The store's record of the year must account for every part, and a part for every tile a run
    last wrote (``registry_master.store_disagreements``). Then every current row's tile holds the
    pixels of the run that row names, because a run writes only its own tiles — so a cell built
    from several runs, an update over a first fill, is re-derivable tile by tile. A run that committed
    but never published its part breaks that, and re-deriving rows from its pixels would publish a
    mixture neither run produced.
    """
    repo, root, writes = store
    if zone not in root:
        return "no such zone in the store"
    group = root[zone]
    years = published_store.calendar_years(group)
    if year not in years:
        return "no time slot in the store"

    def record() -> published_store.StoreYear:
        return published_store.store_year(repo, group, zone, year, writes())

    if found := store_disagreements(parts, record()):
        return "; ".join(found)
    return years.index(year), record


def _rebuild_many(
    group: zarr.Group,
    *,
    time_index: int,
    rows: list[dict[str, Any]],
    with_radar: bool,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    """Re-derive measurements for ``rows``, keyed by tile label. Concurrent; order is irrelevant.

    Each row under its own depth rule, since a cell built from several runs holds each run's rows.
    """

    def one(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return row["tile"], registry_rebuild.rebuild_tile(
            group,
            time_index=time_index,
            tile=row["tile"],
            optical_min_obs=int(row["optical_min_obs"]),
            with_radar=with_radar,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return dict(pool.map(one, rows))


def _gate_rows(
    group: zarr.Group,
    *,
    time_index: int,
    rows: list[dict[str, Any]],
    with_radar: bool,
    concurrency: int,
) -> dict[str, Any]:
    """Re-derive rows that ALREADY carry measurements and compare. The only verdict that passes is ``match``."""
    rebuilt = _rebuild_many(group, time_index=time_index, rows=rows, with_radar=with_radar, concurrency=concurrency)
    verdicts: dict[str, int] = defaultdict(int)
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        verdict, differences = registry_rebuild.compare_row(rebuilt[row["tile"]], row)
        verdicts[verdict] += 1
        if verdict == "mismatch":
            mismatches.append(
                {"zone": row["zone"], "year": row["year"], "tile": row["tile"], "differences": differences}
            )
    return {"checked": len(rows), "verdicts": dict(verdicts), "mismatches": mismatches}


def cmd_gate(args: argparse.Namespace) -> int:
    """Prove the rebuild reproduces measurements the fill already took. Writes nothing."""
    fs = _filesystem(args.registry, args.region, anonymous=args.anonymous)
    table = _registry_table(fs, args.registry, "parts", dataset_schema())
    violating, checked_rows = registry_rebuild.basis_violations(table)
    print(
        f"basis: {checked_rows - violating:,} of {checked_rows:,} measured rows satisfy "
        "refused_no_optical_px == eligible_px - px_with_any_optical"
    )
    if violating:
        print(f"\nFAILED — {violating:,} measured rows break the identity every re-derivation rests on")
        return 1
    cells = _cells(table, args.zones)
    current = {cell: _current(parts) for cell, parts in sorted(cells.items())}
    measured = [
        (cell, rows.filter(pc.is_valid(rows.column(MEASURED_SENTINEL))).to_pylist()) for cell, rows in current.items()
    ]
    measured = [(cell, rows) for cell, rows in measured if rows]
    if not measured:
        # Opening the store costs a session and a credential; a registry with nothing to check
        # against does not need one, and a local rehearsal often has no store at all.
        print("gate: no rows carry measurements, so there is nothing to re-derive against")
        return 0
    rng = random.Random(args.seed)
    # Spread the sample over cells rather than over rows: the thing that could differ is a cell's
    # geometry or its rule, and a row-uniform sample would spend most of its budget in the largest
    # zone-years and check one latitude.
    per_cell = max(1, args.sample // max(1, len(measured)))
    store = _one_read(_open_store(args.store, args.region, anonymous=args.anonymous))
    root = store[1]

    totals: dict[str, int] = defaultdict(int)
    mismatches: list[dict[str, Any]] = []
    checked = 0
    unusable: list[str] = []
    for (zone, year), rows in measured:
        slot = _store_slot(store, zone, year, cells[zone, year])
        if isinstance(slot, str):
            print(f"  {zone}/{year}: FAILED — {slot}", file=sys.stderr)
            unusable.append(f"{zone}/{year}")
            continue
        sample = rows if len(rows) <= per_cell else rng.sample(rows, per_cell)
        result = _gate_rows(
            root[zone], time_index=slot[0], rows=sample, with_radar=not args.skip_radar, concurrency=args.concurrency
        )
        checked += result["checked"]
        for verdict, count in result["verdicts"].items():
            totals[verdict] += count
        mismatches.extend(result["mismatches"])
        print(f"  {zone}/{year}: {result['checked']:>5} checked  {dict(result['verdicts'])}")

    print(f"\ngate: {checked} rows re-derived across {len(measured)} cells")
    for verdict in sorted(totals):
        print(f"  {verdict:<11} {totals[verdict]}")
    if mismatches:
        print(f"\nFAILED — {len(mismatches)} rows the rebuild claims to derive exactly do not agree:")
        for entry in mismatches[:20]:
            print(f"  {entry['zone']}/{entry['year']}/{entry['tile']}: {'; '.join(entry['differences'])}")
    if args.json_out:
        with Path(args.json_out).open("w") as handle:
            json.dump({"checked": checked, "verdicts": dict(totals), "mismatches": mismatches}, handle, indent=2)
    if unusable:
        print(f"\nFAILED — {len(unusable)} cells the store cannot stand behind: {', '.join(unusable[:20])}")
    return 1 if mismatches or unusable else 0


def _rederive(
    group: zarr.Group,
    time_index: int,
    every: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
    run_id: str,
) -> pa.Table | str:
    """Fresh master rows for the cell's unmeasured rows, or why the in-cell gate refused them.

    Self-checks first: a cell holding measured tiles with optical in them proves the re-derivation
    reproduces its own geometry and rule before a single re-derived value is accepted.
    """
    targets = [row for row in every if row.get(MEASURED_SENTINEL) is None]
    known = [
        row
        for row in every
        if row.get(MEASURED_SENTINEL) is not None
        and row.get("eligible_px") == row.get("chunk_px")
        and row.get("px_with_any_optical")
    ]
    if known and not args.no_in_cell_gate:
        sample = known if len(known) <= IN_CELL_GATE_TILES else rng.sample(known, IN_CELL_GATE_TILES)
        check = _gate_rows(
            group, time_index=time_index, rows=sample, with_radar=not args.skip_radar, concurrency=args.concurrency
        )
        if check["mismatches"]:
            first = check["mismatches"][0]
            return (
                f"in-cell gate failed on {len(check['mismatches'])} rows, e.g. {first['tile']}: {first['differences']}"
            )
    measurements = _rebuild_many(
        group, time_index=time_index, rows=targets, with_radar=not args.skip_radar, concurrency=args.concurrency
    )
    stamp = datetime.datetime.now(datetime.UTC).isoformat()
    rebuilt = [rebuilt_row(row, measurements[row["tile"]], run_id=run_id, assembled_at=stamp) for row in targets]
    return as_master_rows(pa.Table.from_pylist(rebuilt, schema=master_schema()), MEASURED_BY_REBUILD)


def _repair_cell(
    fs: AbstractFileSystem,
    store: _Store,
    zone: str,
    year: int,
    parts: pa.Table,
    args: argparse.Namespace,
    rng: random.Random,
    run_id: str,
) -> int:
    """Rebuild one cell's master through ``merge_cell``; returns the rows re-derived.

    Raises ``ValueError`` with the reason when the cell is blocked, writing nothing.
    """
    # Checked for EVERY cell, not only one with rows to re-derive: a run the store committed that
    # the parts do not account for means they no longer describe the cell, whatever they hold.
    slot = _store_slot(store, zone, year, parts)
    if isinstance(slot, str):
        raise ValueError(slot)
    current = _current(parts)
    extra = []
    if pc.any(pc.is_null(current.column(MEASURED_SENTINEL))).as_py():
        fresh = _rederive(store[1][zone], slot[0], current.to_pylist(), args, rng, run_id)
        if isinstance(fresh, str):
            raise ValueError(fresh)
        extra.append(fresh)
    # The cell's one writer, conditional on its file as it is now: a fill landing mid-repair is
    # merged in, not overwritten, and the parts and master rows are re-read for this cell alone.
    merge_cell(fs, args.registry, zone, year, store=slot[1], extra=extra, write=args.write)
    return sum(table.num_rows for table in extra)


def cmd_repair(args: argparse.Namespace) -> int:
    """Rebuild each cell's master file from its parts, its master rows and a fresh re-derivation."""
    fs = _filesystem(args.registry, args.region, anonymous=args.anonymous)
    parts = _cells(_registry_table(fs, args.registry, "parts", dataset_schema()), [])
    store = _open_store(args.store, args.region, anonymous=args.anonymous)
    cells = _every_cell(fs, args.registry, store, parts, args.zones)
    run_id = args.run_id or rebuild_run_id(suffix="optical" if args.skip_radar else "full")
    rng = random.Random(args.seed)

    written = rederived = blocked = 0
    for (zone, year), parts in sorted(cells.items()):
        try:
            rederived += _repair_cell(fs, store, zone, year, parts, args, rng, run_id)
        except ValueError as error:
            blocked += 1
            print(f"  {zone}/{year}: BLOCKED, left as it is — {error}", file=sys.stderr)
            continue
        written += 1
    action = "written" if args.write else "would be written (dry run — pass --write to publish)"
    print(f"\nrepair {run_id}: {written} cell(s) {action}, {rederived} rows re-derived, {blocked} cells blocked")
    return 1 if blocked else 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Check ``master/`` itself: its invariants, and every cell against its parts."""
    fs = _filesystem(args.registry, args.region, anonymous=args.anonymous)
    parts = _registry_table(fs, args.registry, "parts", dataset_schema())
    master = _registry_table(fs, args.registry, "master", master_schema())
    if args.zones:
        # With the store checked, `_every_cell` checks the zones against its commits too, below.
        held = set(parts.column("zone").to_pylist()) | set(master.column("zone").to_pylist())
        if args.parts_only and (absent := sorted(set(args.zones) - held)):
            raise SystemExit(f"--zones names zones the registry does not hold: {', '.join(absent)}")
        parts, master = (
            table.filter(pc.is_in(table.column("zone"), pa.array(args.zones))) for table in (parts, master)
        )
    expected = tile_years(parts)
    failures = invariant_failures(master, expected_rows=expected)
    problems = audit(master, parts)
    if not args.parts_only:
        # The parts themselves against the store's record of who wrote each year: a run with no
        # part leaves the parts, and so the master, describing its tiles by the run before it.
        store = _one_read(_open_store(args.store, args.region, anonymous=args.anonymous))
        for (zone, year), cell in _every_cell(fs, args.registry, store, _cells(parts, []), args.zones).items():
            if isinstance(slot := _store_slot(store, zone, year, cell), str):
                problems.setdefault(f"{zone}/{year}", []).append(f"the store cannot stand behind its parts: {slot}")
        problems = dict(sorted(problems.items()))
    origins = master.column("measured_by").to_pylist()
    print(f"master {master.num_rows} rows against {expected} distinct tile-years in parts/")
    print(f"  measured by fill {origins.count(MEASURED_BY_FILL)}, by rebuild {origins.count(MEASURED_BY_REBUILD)}")
    for failure in failures:
        print(f"  INVARIANT: {failure}")
    for cell, found in list(problems.items())[:20]:
        print(f"  {cell}: {'; '.join(found)}")
    if problems:
        zones = ",".join(sorted({cell.split("/")[0] for cell in problems}))
        print(f"  {len(problems)} cell(s) need repair: repair --zones {zones} --write")
    return 1 if failures or problems else 0


def main(argv: list[str] | None = None) -> int:
    """Parse the subcommand and run it; the exit status is the subcommand's."""
    # Every option lives on the SUBCOMMANDS rather than on the top-level parser, so a flag written
    # after the subcommand works — which is where anybody types it, and where a global-only
    # `--write` silently becomes an argparse error instead.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", default=DEFAULT_REGISTRY, help="registry root; a non-s3 path reads locally")
    common.add_argument("--store", default=DEFAULT_STORE)
    common.add_argument("--region", default=DEFAULT_REGION)
    common.add_argument(
        "--anonymous", action="store_true", help="read with no credentials (published bucket allows it)"
    )
    common.add_argument("--zones", default="", help="comma-separated zones to limit the pass to")
    common.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument(
        "--skip-radar", action="store_true", help="leave px_with_any_radar null — the optical-only pass"
    )
    common.add_argument(
        "--write", action="store_true", help="actually publish; without it every subcommand is a dry run"
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", parents=[common], help="re-derive rows that already carry measurements and compare")
    gate.add_argument("--sample", type=int, default=50_000)
    gate.add_argument("--json", dest="json_out")
    gate.set_defaults(func=cmd_gate)

    repair = sub.add_parser("repair", parents=[common], help="rebuild master/ cells, re-deriving rows fills left null")
    repair.add_argument("--run-id", default="")
    repair.add_argument("--no-in-cell-gate", action="store_true", help="skip each cell's self-check (not advised)")
    repair.set_defaults(func=cmd_repair)

    verify = sub.add_parser(
        "verify", parents=[common], help="check master/ against parts/, and parts/ against the store's run record"
    )
    verify.add_argument(
        "--parts-only", action="store_true", help="skip the store check, for a registry copy with no store to read"
    )
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    args.zones = [z.strip() for z in args.zones.split(",") if z.strip()]
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
