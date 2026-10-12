"""Repair the registry: re-derive measurements a fill left null, and re-sync ``master/``.

Every fill keeps its own cell's ``master/`` file current (``registry_master.update_cell``, called
by the zone-fill runner). This script is for what that cannot do on its own: re-deriving rows a
fill published null, and rebuilding master files a failed update left stale. Its first use repaired
the v1.1 campaign, whose resumed fills published 1,519,045 of 3,247,410 rows (47%) null in every
measurement column (``context_docs/storage/registry-backfill.md``).

Every column is derivable from the store, so this is a re-derivation rather than an estimate — see
:mod:`tessera_embeddings.storage.registry_rebuild` for the proof that the fill's optical test
reduces to ``s2_obs_count > 0`` on the delivered product, and for the one column
(``eligible_px``) deliberately left null.

**Nothing here rewrites published input.** ``parts/`` is never touched; rebuilt rows land under a
``rebuild/`` prefix of their own, and ``sync`` rewrites only ``master/``, which is derived from the
other two. Every writing subcommand is a DRY RUN unless given ``--write``.

Run from the REPOSITORY ROOT::

    # 1. prove the rebuild reproduces measurements that already exist. Writes nothing, ever.
    uv run python scripts/maintenance/rebuild_registry_measurements.py gate --sample 50000

    # 2. re-derive the missing rows. --skip-radar is the cheap optical-only first pass.
    uv run python scripts/maintenance/rebuild_registry_measurements.py rebuild --write

    # 3. rebuild master/ from parts/ + rebuild/ (every cell, or --zones), and check it
    uv run python scripts/maintenance/rebuild_registry_measurements.py sync --write
    uv run python scripts/maintenance/rebuild_registry_measurements.py verify

**Rehearsing needs no credentials.** ``--anonymous`` reads the public store and registry unsigned,
and a non-``s3://`` ``--registry`` is the local filesystem, so a downloaded copy of the registry takes
every subcommand end to end — ``rebuild --write`` and ``sync --write`` included — without touching
the bucket.

**One process saturates at three to four cores**, because the per-tile work holds the GIL for part
of its time. A full pass on a bigger host runs several processes over disjoint ``--zones`` lists;
``--zones`` is applied before rows are materialised, so each holds only its own share in memory.

**The radar column is a second pass by design.** ``px_with_any_radar`` needs both Sentinel-1
observation-count arrays, which triples the bytes moved for one informational column. Run
``rebuild --skip-radar`` first to land the optical answer, then ``rebuild`` in full; the later
pass's rows supersede the earlier ones in the merge, by ``assembled_at``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs
import zarr

from tessera_embeddings.storage import published_store, registry_rebuild
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.registry import dataset_schema
from tessera_embeddings.storage.registry_master import (
    MEASURED_BY_FILL,
    cell_tables,
    current_fill_rows,
    fill_rows_unchanged,
    invariant_failures,
    master_schema,
    merge,
    stale_cells,
    tile_years,
    write_cell,
    write_common_metadata,
)
from tessera_embeddings.storage.registry_rebuild import (
    REBUILT_COLUMNS,
    rebuild_dataset_schema,
    rebuild_run_id,
    rebuilt_row,
    write_rebuild_part,
)

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


def _filesystem(root: str, region: str, *, anonymous: bool = False) -> pyarrow.fs.FileSystem:
    """The filesystem for ``root`` — S3 for an ``s3://`` registry, local for anything else.

    Local is not a convenience: rehearsing a sync against a downloaded copy of the registry is how
    the invariants get checked before anything is published, and a subcommand that only knows
    how to talk to the real bucket cannot be rehearsed at all.
    """
    if root.startswith("s3://"):
        return pyarrow.fs.S3FileSystem(region=region, anonymous=anonymous)
    return pyarrow.fs.LocalFileSystem()


def _registry_table(
    fs: pyarrow.fs.FileSystem, root: str, prefix: str, schema: pa.Schema, *, optional: bool = False
) -> pa.Table:
    """One of the registry's prefixes as a table, schema STATED rather than inferred.

    Inferring it from the first file in sorted path order is how a column added mid-campaign
    silently disappears from a whole-dataset read; the registry's own docstring says so and this is
    the script that would be most damaged by it.

    Only an ``optional`` prefix may be absent — ``rebuild/`` before the first pass — and then it
    reads as an empty table. Anything else that cannot be read raises: an empty ``parts/`` from a
    mistyped ``--registry`` or a failed read would let every subcommand pass having checked nothing.
    """
    path = root.removeprefix("s3://").rstrip("/") + "/" + prefix
    if fs.get_file_info(path).type == pyarrow.fs.FileType.NotFound:
        if optional:
            return pa.Table.from_pylist([], schema=schema)
        raise SystemExit(f"{root.rstrip('/')}/{prefix} does not exist — is --registry right?")
    table = ds.dataset(path, filesystem=fs, partitioning="hive", schema=schema).to_table()
    if not optional and not table.num_rows:
        raise SystemExit(f"{root.rstrip('/')}/{prefix} holds no rows — refusing to treat that as a registry")
    return table


def _open_zone(store_uri: str, region: str, *, anonymous: bool) -> zarr.Group:
    """The published store's root group, read-only."""
    session = open_global_repo(store_uri, region=region, anonymous=anonymous).readonly_session(branch="main")
    return zarr.open_group(session.store, mode="r")


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


def _cells(table: pa.Table, zones: list[str]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Registry rows grouped by ``(zone, year)``, which is the unit a fill — and a rebuild — works in.

    Only each tile's current row is kept — from the newest run that wrote it, the run the store
    holds there and the master keeps — so a refilled tile is neither gated against an older run's
    numbers nor rebuilt under its rule.
    """
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in current_fill_rows(_only_zones(table, zones)).to_pylist():
        grouped[(row["zone"], int(row["year"]))].append(row)
    return dict(grouped)


def _store_slot(group: zarr.Group, year: int, rows: list[dict[str, Any]]) -> int | str:
    """The cell's time index in the store, or why the store cannot stand behind these registry rows.

    The zone group records the last run that wrote each year, which must be the newest run among
    the cell's rows. Then every row's tile holds the pixels of the run that row names, because a
    run writes only its own tiles — so a cell built from several runs, an infill over a fill, is
    re-derivable tile by tile. A run that committed to the store but never published its registry
    part breaks that, and re-deriving rows from its pixels would publish a mixture neither run
    produced.
    """
    years = published_store.calendar_years(group)
    if year not in years:
        return "no time slot in the store"
    newest = max((datetime.datetime.fromisoformat(row["assembled_at"]), row["run_id"]) for row in rows)[1]
    stored = ((dict(group.attrs).get("runs") or {}).get(str(year)) or {}).get("run_id")
    if stored != newest:
        return f"the store holds run {stored!r}, the registry's newest is {newest!r}"
    return years.index(year)


def _rebuild_many(
    group: zarr.Group,
    *,
    time_index: int,
    rows: list[dict[str, Any]],
    optical_min_obs: int,
    with_radar: bool,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    """Re-derive measurements for ``rows``, keyed by tile label. Concurrent; order is irrelevant."""

    def one(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return row["tile"], registry_rebuild.rebuild_tile(
            group,
            time_index=time_index,
            tile=row["tile"],
            optical_min_obs=optical_min_obs,
            with_radar=with_radar,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return dict(pool.map(one, rows))


def _gate_rows(
    group: zarr.Group,
    *,
    time_index: int,
    rows: list[dict[str, Any]],
    optical_min_obs: int,
    with_radar: bool,
    concurrency: int,
) -> dict[str, Any]:
    """Re-derive rows that ALREADY carry measurements and compare. The only verdict that passes is ``match``."""
    rebuilt = _rebuild_many(
        group,
        time_index=time_index,
        rows=rows,
        optical_min_obs=optical_min_obs,
        with_radar=with_radar,
        concurrency=concurrency,
    )
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
    measured = [
        (cell, [row for row in rows if row.get("chunk_px") is not None]) for cell, rows in sorted(cells.items())
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
    root = _open_zone(args.store, args.region, anonymous=args.anonymous)

    totals: dict[str, int] = defaultdict(int)
    mismatches: list[dict[str, Any]] = []
    checked = 0
    unusable: list[str] = []
    for (zone, year), rows in measured:
        group = root[zone]
        time_index = _store_slot(group, year, cells[zone, year])
        if isinstance(time_index, str):
            print(f"  {zone}/{year}: FAILED — {time_index}", file=sys.stderr)
            unusable.append(f"{zone}/{year}")
            continue
        sample = rows if len(rows) <= per_cell else rng.sample(rows, per_cell)
        result = _gate_rows(
            group,
            time_index=time_index,
            rows=sample,
            optical_min_obs=int(sample[0]["optical_min_obs"]),
            with_radar=not args.skip_radar,
            concurrency=args.concurrency,
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


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Re-derive the null rows and write one part per cell under ``rebuild/``."""
    fs = _filesystem(args.registry, args.region, anonymous=args.anonymous)
    table = _registry_table(fs, args.registry, "parts", dataset_schema())
    cells = _cells(table, args.zones)
    run_id = args.run_id or rebuild_run_id(suffix="optical" if args.skip_radar else "full")
    rebuild_root = args.registry.rstrip("/") + "/rebuild"
    if not any(row.get("chunk_px") is None for rows in cells.values() for row in rows):
        print("rebuild: every row already carries measurements, so there is nothing to re-derive")
        return 0
    root = _open_zone(args.store, args.region, anonymous=args.anonymous)
    rng = random.Random(args.seed)

    written = blocked = 0
    for (zone, year), rows in sorted(cells.items()):
        targets = [row for row in rows if row.get("chunk_px") is None]
        if not targets:
            continue
        group = root[zone]
        time_index = _store_slot(group, year, rows)
        if isinstance(time_index, str):
            blocked += 1
            print(f"  {zone}/{year}: BLOCKED — {time_index}", file=sys.stderr)
            continue
        rule = int(targets[0]["optical_min_obs"])

        # SELF-CHECK FIRST. A cell that holds measured tiles with optical in them can prove the
        # rebuild reproduces its own geometry and rule before a single re-derived row is written.
        known = [
            row
            for row in rows
            if row.get("chunk_px") is not None
            and row.get("eligible_px") == row.get("chunk_px")
            and row.get("px_with_any_optical")
        ]
        if known and not args.no_in_cell_gate:
            sample = known if len(known) <= IN_CELL_GATE_TILES else rng.sample(known, IN_CELL_GATE_TILES)
            check = _gate_rows(
                group,
                time_index=time_index,
                rows=sample,
                optical_min_obs=rule,
                with_radar=not args.skip_radar,
                concurrency=args.concurrency,
            )
            if check["mismatches"]:
                blocked += 1
                print(
                    f"  {zone}/{year}: BLOCKED — in-cell gate failed on {len(check['mismatches'])} rows",
                    file=sys.stderr,
                )
                for entry in check["mismatches"][:3]:
                    print(f"      {entry['tile']}: {'; '.join(entry['differences'])}", file=sys.stderr)
                continue

        measurements = _rebuild_many(
            group,
            time_index=time_index,
            rows=targets,
            optical_min_obs=rule,
            with_radar=not args.skip_radar,
            concurrency=args.concurrency,
        )
        stamp = datetime.datetime.now(datetime.UTC).isoformat()
        part = [rebuilt_row(row, measurements[row["tile"]], run_id=run_id, assembled_at=stamp) for row in targets]
        uri = registry_rebuild.part_uri(rebuild_root, zone, year, run_id)
        print(f"  {zone}/{year}: {len(part):>5} rows -> {uri}{'' if args.write else '  (dry run)'}")
        if args.write:
            fs.create_dir(uri.removeprefix("s3://").rsplit("/", 1)[0])
            write_rebuild_part(
                uri,
                part,
                open_output=lambda target: fs.open_output_stream(target.removeprefix("s3://")),
                zone=zone,
                year=year,
                extra_metadata={"rebuilt_columns": ",".join(REBUILT_COLUMNS), "with_radar": str(not args.skip_radar)},
            )
        written += len(part)

    print(
        f"\nrebuild {run_id}: {written} rows {'written' if args.write else 'would be written'}, {blocked} cells blocked"
    )
    return 1 if blocked else 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Rebuild every cell's ``master/`` file (or ``--zones``'s) from ``parts/`` and ``rebuild/``.

    The per-cell update every fill runs, for all cells at once: the repair for a cell whose update
    failed, and for a rebuild pass that has landed in ``rebuild/`` but not yet in the master.
    """
    fs = _filesystem(args.registry, args.region, anonymous=args.anonymous)
    parts = _only_zones(_registry_table(fs, args.registry, "parts", dataset_schema()), args.zones)
    rebuilt = _registry_table(fs, args.registry, "rebuild", rebuild_dataset_schema(), optional=True)
    if args.zones:
        rebuilt = rebuilt.filter(pc.is_in(rebuilt.column("zone"), value_set=pa.array(args.zones)))
    master = merge(parts, rebuilt)

    failures = invariant_failures(master, expected_rows=tile_years(parts))
    origins = defaultdict(int)
    for tag in master.column("measured_by").to_pylist():
        origins[tag or "unmeasured"] += 1
    print(f"parts {parts.num_rows} rows, rebuild {rebuilt.num_rows} rows -> master {master.num_rows} rows")
    print(f"  measured_by: {dict(origins)}")
    if failures:
        print("\nFAILED — the merged master breaks its own invariants:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    if not args.write:
        print("  (dry run — pass --write to publish)")
        return 0
    # One object per cell, each replaced whole: a reader mid-sync sees every cell either before or
    # after, never half-written, and an interrupted sync is repaired by running it again.
    cells = [write_cell(fs, args.registry, zone, year, rows) for zone, year, rows in cell_tables(master)]
    # Rewritten here, where a fill only writes it when absent: a sync is how a schema change lands.
    write_common_metadata(fs, args.registry)
    print(f"  {len(cells)} cell(s) written to {args.registry.rstrip('/')}/master")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Check the published master against its invariants, against ``parts/``, and against a fresh merge."""
    fs = _filesystem(args.registry, args.region, anonymous=args.anonymous)
    parts = _registry_table(fs, args.registry, "parts", dataset_schema())
    master = _registry_table(fs, args.registry, "master", master_schema())
    rebuilt = _registry_table(fs, args.registry, "rebuild", rebuild_dataset_schema(), optional=True)
    expected = tile_years(parts)
    failures = invariant_failures(master, expected_rows=expected)
    altered = fill_rows_unchanged(master, parts, REBUILT_COLUMNS)
    stale = stale_cells(master, merge(parts, rebuilt))
    if stale:
        failures.append(
            f"{len(stale)} cell(s) differ from a fresh merge of parts/ and rebuild/: {', '.join(stale[:20])} "
            "— re-run sync with --zones"
        )
    print(f"master {master.num_rows} rows against {parts.num_rows} part rows ({expected} distinct tile-years)")
    unmeasured = sum(1 for tag in master.column("measured_by").to_pylist() if tag is None)
    from_fill = sum(1 for tag in master.column("measured_by").to_pylist() if tag == MEASURED_BY_FILL)
    print(f"  measured by fill {from_fill}, still unmeasured {unmeasured}")
    for failure in failures:
        print(f"  INVARIANT: {failure}")
    for difference in altered[:20]:
        print(f"  ALTERED: {difference}")
    if len(altered) > 20:
        print(f"  ... and {len(altered) - 20} more altered rows")
    return 1 if failures or altered else 0


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

    rebuild = sub.add_parser("rebuild", parents=[common], help="re-derive the null rows into rebuild/")
    rebuild.add_argument("--run-id", default="")
    rebuild.add_argument("--no-in-cell-gate", action="store_true", help="skip each cell's self-check (not advised)")
    rebuild.set_defaults(func=cmd_rebuild)

    sync = sub.add_parser("sync", parents=[common], help="rebuild master/ from parts/ and rebuild/, per cell")
    sync.set_defaults(func=cmd_sync)

    verify = sub.add_parser("verify", parents=[common], help="check master/ against its invariants and against parts/")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    args.zones = [z.strip() for z in args.zones.split(",") if z.strip()]
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
