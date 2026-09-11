"""Audit the published registry: is it written as designed, and can a consumer navigate it?

The registry is the Parquet dataset beside the store that answers "is my area covered, and how
well" without opening a petabyte. This script checks the three ways that promise can fail.

**Is it shaped as designed?** Parts should land at ``parts/zone=<ZONE>/year=<YEAR>/<run>.parquet``,
one per cell, keyed by run so a refill adds a part instead of overwriting one. A cell with two
parts is therefore expected and reported rather than flagged, but a cell with none, or a part where
no cell is complete, is a real disagreement with the store.

**Can the whole dataset be read?** A campaign crossing code versions leaves older parts missing a
column newer ones have, and ``pyarrow`` infers a dataset's schema from the first file it finds in
sorted path order — so a column added mid-campaign is silently dropped from every whole-dataset
read if zone 01N was written before the change. The only safe read states the schema explicitly.
This script reads both ways and reports whether the answers differ, which is the difference between
a consumer who follows the documentation and one who does not.

**Does it agree with the store?** Every column is derivable from the store, so the registry is a
convenience layer and never a second source of truth — which is only true if it actually agrees.
``--verify-zones`` reads live shards for the named zones and checks, per cell, that the count of
rows with ``embedded`` true equals the number of shards holding embeddings. A mismatch means one of
the two is wrong about published coverage, and that is worth knowing before a consumer trusts
either.

Run from the REPOSITORY ROOT::

    uv run python scripts/diagnostic/published_registry_census.py
    uv run python scripts/diagnostic/published_registry_census.py --verify-zones 16S,33N --aoi=-93.8,41.9,-93.4,42.2

``--aoi`` takes ``west,south,east,north`` in WGS84 degrees and must be written with an equals sign,
because a western longitude starts with a minus and would otherwise be read as a flag.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

if TYPE_CHECKING:
    import pyarrow.fs

from tessera_embeddings.storage import published_store
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.registry import dataset_schema, registry_schema

DEFAULT_STORE = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
DEFAULT_REGISTRY = "s3://tessera-embeddings/v1.1/dclimate.registry"
DEFAULT_REGION = "us-west-2"


def _filesystem(region: str, *, anonymous: bool = False) -> pyarrow.fs.FileSystem:
    """An S3 filesystem for the registry — anonymous, or on the ambient credential chain.

    ``anonymous`` matters for more than convenience: the registry sits in a bucket whose policy
    grants public reads, so a consumer without an AWS account reads it that way, and a diagnostic
    that only ever signs its requests cannot tell whether that consumer's path works.
    """
    from pyarrow.fs import S3FileSystem

    return S3FileSystem(region=region, anonymous=anonymous)


def _parts(fs: pyarrow.fs.FileSystem, root: str) -> list[dict[str, Any]]:
    """Every Parquet part with its parsed partition keys, from one listing of the dataset.

    Parsed from the PATH because the path is the authority — ``zone`` and ``year`` are hive
    partition keys and deliberately not columns, so a part opened alone learns what it describes
    from where it sits (and from its own key-value metadata, which this script checks separately).
    """
    from pyarrow.fs import FileSelector

    prefix = root.removeprefix("s3://").rstrip("/") + "/parts"
    out: list[dict[str, Any]] = []
    for info in fs.get_file_info(FileSelector(prefix, recursive=True)):
        if not info.path.endswith(".parquet"):
            continue
        segments = info.path.split("/")
        zone = next((s.removeprefix("zone=") for s in segments if s.startswith("zone=")), None)
        year = next((s.removeprefix("year=") for s in segments if s.startswith("year=")), None)
        out.append(
            {
                "path": info.path,
                "zone": zone,
                "year": int(year) if year and year.isdigit() else None,
                "run_id": Path(info.path).stem,
                "size": info.size,
            }
        )
    return out


def _siblings(fs: pyarrow.fs.FileSystem, root: str) -> list[str]:
    """Names of everything sitting directly under the registry root, ``parts`` included."""
    from pyarrow.fs import FileSelector

    selector = FileSelector(root.removeprefix("s3://").rstrip("/"), recursive=False, allow_not_found=True)
    return sorted(Path(info.path).name for info in fs.get_file_info(selector))


def _schema_audit(fs: pyarrow.fs.FileSystem, parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether every part carries the declared columns, and what each part's own metadata says.

    Reads footers only — no row groups — so the cost is one small range request per part.
    """
    import pyarrow.parquet as pq

    declared = {field.name: str(field.type) for field in registry_schema()}
    missing: dict[str, list[str]] = {}
    extra: dict[str, list[str]] = {}
    retyped: dict[str, list[str]] = {}
    without_identity: list[str] = []
    for part in parts:
        schema = pq.read_schema(part["path"], filesystem=fs)
        present = {field.name: str(field.type) for field in schema}
        if absent := sorted(set(declared) - set(present)):
            missing[part["path"]] = absent
        if surplus := sorted(set(present) - set(declared)):
            extra[part["path"]] = surplus
        if wrong := sorted(n for n in set(declared) & set(present) if declared[n] != present[n]):
            retyped[part["path"]] = wrong
        metadata = {k.decode(): v.decode() for k, v in (schema.metadata or {}).items() if k != b"pandas"}
        if metadata.get("zone") != part["zone"] or metadata.get("year") != str(part["year"]):
            without_identity.append(part["path"])
    return {
        "parts_missing_declared_columns": missing,
        "parts_with_undeclared_columns": extra,
        "parts_with_wrong_column_types": retyped,
        "parts_whose_metadata_disagrees_with_its_path": without_identity,
    }


def _read_dataset(fs: pyarrow.fs.FileSystem, root: str, *, with_schema: bool) -> dict[str, Any]:
    """Read the whole dataset, with the declared schema or with an inferred one, and time it."""
    import pyarrow.dataset as ds

    prefix = root.removeprefix("s3://").rstrip("/") + "/parts"
    started = time.monotonic()
    kwargs: dict[str, Any] = {"filesystem": fs, "partitioning": "hive", "format": "parquet"}
    if with_schema:
        kwargs["schema"] = dataset_schema()
    dataset = ds.dataset(prefix, **kwargs)
    table = dataset.to_table()
    return {
        "schema_stated": with_schema,
        "wall_s": round(time.monotonic() - started, 2),
        "rows": table.num_rows,
        "columns": sorted(table.column_names),
    }


def _aoi_query(
    fs: pyarrow.fs.FileSystem, root: str, aoi: tuple[float, float, float, float], year: int
) -> dict[str, Any]:
    """Time the question the registry exists to answer: is this box covered in this year?

    Filters on the row bounding boxes with a plain overlap test rather than the containment test a
    reader might reach for first — a tile overlapping the area of interest is part of the answer
    even when it is not inside it. The antimeridian is left out on purpose: rows in zones 01 and 60
    have ``bbox_west > bbox_east``, so an overlap test written this way drops them, and the
    honest thing for a diagnostic is to say so rather than to appear to handle it.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    west, south, east, north = aoi
    prefix = root.removeprefix("s3://").rstrip("/") + "/parts"
    dataset = ds.dataset(prefix, filesystem=fs, partitioning="hive", schema=dataset_schema())
    started = time.monotonic()
    table = dataset.to_table(
        filter=(pc.field("year") == year)
        & (pc.field("bbox_west") <= east)
        & (pc.field("bbox_east") >= west)
        & (pc.field("bbox_south") <= north)
        & (pc.field("bbox_north") >= south),
        columns=["tile", "embedded", "refused_px", "eligible_px", "median_obs_where_thin"],
    )
    embedded = table.column("embedded").to_pylist()
    refused = [v for v in table.column("refused_px").to_pylist() if v is not None]
    return {
        "aoi": list(aoi),
        "year": year,
        "wall_s": round(time.monotonic() - started, 2),
        "tiles_overlapping": table.num_rows,
        "tiles_embedded": sum(1 for v in embedded if v),
        "tiles_not_embedded": sum(1 for v in embedded if not v),
        "refused_px_total": sum(refused),
        "note": "antimeridian zones 01/60 are not handled by this overlap test",
    }


def _latest_per_tile(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep one row per tile — the one with the latest ``assembled_at``.

    The registry's latest-wins rule, applied rather than assumed away. A tie keeps the row already
    held: arbitrary but stable, and a tie means two runs stamped the same instant, which nothing in
    the data can order.

    **Compared as strings, which is only valid while every part writes the same timestamp format.**
    ``assembled_at`` is a column of strings, and the writer fills it from ``datetime.isoformat()``
    with a ``+00:00`` offset, so lexical order is chronological order. A future writer emitting
    ``Z`` instead, or a local offset, would sort wrongly and silently — so this parses rather than
    trusting the ordering, and says so if a value does not match the expected shape.
    """
    latest: dict[str, tuple[datetime, dict[str, Any]]] = {}
    for row in rows:
        stamp = datetime.fromisoformat(str(row["assembled_at"]))
        held = latest.get(row["tile"])
        if held is None or stamp > held[0]:
            latest[row["tile"]] = (stamp, row)
    return {tile: row for tile, (_, row) in latest.items()}


def _verify_against_store(
    fs: pyarrow.fs.FileSystem,
    root: str,
    store_uri: str,
    region: str,
    zones: list[str],
    *,
    anonymous: bool = False,
) -> list[dict[str, Any]]:
    """Per cell, compare the registry's embedded-tile count with the store's live shard count."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    prefix = root.removeprefix("s3://").rstrip("/") + "/parts"
    dataset = ds.dataset(prefix, filesystem=fs, partitioning="hive", schema=dataset_schema())
    session = open_global_repo(store_uri, region=region, anonymous=anonymous).readonly_session(branch="main")
    root_group = zarr.open_group(session.store, mode="r")

    findings: list[dict[str, Any]] = []
    for zone in zones:
        group = root_group[zone]
        years = [int(y) for y in group.attrs.get("years_complete", [])]
        stamps = group["time"][:]
        calendar = _calendar_years(stamps)
        coverage = published_store.live_shards(session, zone)
        table = dataset.to_table(
            filter=pc.field("zone") == zone,
            columns=["year", "embedded", "tile", "refused_px", "run_id", "assembled_at"],
        )
        rows = table.to_pylist()
        for year in sorted(set(calendar)):
            time_index = calendar.index(year)
            all_rows = [r for r in rows if r["year"] == year]
            # LATEST RUN PER TILE, not every row. A refill deliberately writes a NEW part rather
            # than overwriting the original, so the registry holds one complete tile set per run —
            # and summing them all would count a two-run cell's tiles twice against a store that
            # holds one shard each, reporting a correct cell as a disagreement. `assembled_at` is
            # the clock the registry provides for exactly this decision; a run id is not one.
            registry_rows = list(_latest_per_tile(all_rows).values())
            findings.append(
                {
                    "zone": zone,
                    "year": year,
                    "marked_complete": year in years,
                    "registry_rows": len(registry_rows),
                    "registry_rows_before_dedup": len(all_rows),
                    "runs_present": sorted({r["run_id"] for r in all_rows}),
                    "registry_embedded": sum(1 for r in registry_rows if r["embedded"]),
                    "registry_not_embedded": sum(1 for r in registry_rows if not r["embedded"]),
                    "store_live_shards": len(coverage.get(time_index, ())),
                }
            )
    return findings


def _calendar_years(stamps: np.ndarray) -> list[int]:
    """Calendar years from an int64-nanosecond time coordinate; see the read bench on the cast."""
    decoded = np.asarray(stamps).astype("datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
    years = [int(y) for y in decoded]
    if not all(1970 <= y <= 2200 for y in years):
        raise ValueError(f"time coordinate did not decode to plausible years: {years[:4]}")
    return years


def main(argv: list[str] | None = None) -> int:
    """Audit the registry; return 0 when nothing disagrees and 1 when something does."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--verify-zones", default="", help="comma-separated zones to cross-check against the store")
    parser.add_argument("--aoi", default="", help="west,south,east,north in WGS84 degrees, to time a coverage query")
    parser.add_argument("--aoi-year", type=int, default=2025)
    parser.add_argument("--skip-schema-audit", action="store_true", help="skip the per-part footer read")
    parser.add_argument(
        "--anonymous",
        action="store_true",
        help="read with no credentials (the published bucket grants public reads)",
    )
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args(argv)

    fs = _filesystem(args.region, anonymous=args.anonymous)
    started = time.monotonic()
    parts = _parts(fs, args.registry)
    listing_s = round(time.monotonic() - started, 2)

    # Unparsable paths are separated BEFORE the cell map is built. A key of `(None, None)` sorted
    # alongside `("01N", 2017)` raises TypeError in Python 3, so the diagnostic would crash on
    # exactly the malformed paths it exists to report, before reporting them.
    unparsed = [part["path"] for part in parts if part["zone"] is None or part["year"] is None]
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for part in parts:
        if part["zone"] is None or part["year"] is None:
            continue
        cells.setdefault((part["zone"], part["year"]), []).append(part)
    refilled = {f"{z}/{y}": len(v) for (z, y), v in sorted(cells.items()) if len(v) > 1}

    print(f"registry:      {args.registry} ({args.region})")
    print(f"parts:         {len(parts)} in {listing_s}s, {sum(p['size'] for p in parts) / 1e6:.1f} MB")
    print(f"cells covered: {len(cells)}")
    print(f"cells with more than one part (a refill): {len(refilled)} {list(refilled)[:8]}")
    print(f"parts whose path does not parse:          {len(unparsed)} {unparsed[:4]}")

    # A compacted master and a dataset-level `_common_metadata` are what a consumer WITHOUT this
    # package needs: the first so one schema covers the whole read, the second so a reader can take
    # the schema from the dataset instead of from a part it has to know is current. Their absence is
    # a finding about who can read this dataset, not a defect in it.
    siblings = _siblings(fs, args.registry)
    print(f"siblings beside parts/: {[s for s in siblings if s != 'parts'] or 'none'}")

    report: dict[str, Any] = {
        "registry": args.registry,
        "region": args.region,
        "parts": len(parts),
        "bytes": sum(p["size"] for p in parts),
        "listing_s": listing_s,
        "cells": len(cells),
        "cells_with_multiple_parts": refilled,
        "parts_with_unparsable_path": unparsed,
        "siblings": siblings,
    }

    if not args.skip_schema_audit:
        started = time.monotonic()
        report["schema_audit"] = _schema_audit(fs, parts)
        report["schema_audit"]["wall_s"] = round(time.monotonic() - started, 2)
        audit = report["schema_audit"]
        print(f"\nschema audit ({audit['wall_s']}s over {len(parts)} footers):")
        for key in (
            "parts_missing_declared_columns",
            "parts_with_undeclared_columns",
            "parts_with_wrong_column_types",
            "parts_whose_metadata_disagrees_with_its_path",
        ):
            print(f"  {key}: {len(audit[key])}")

    # Run BOTH ways to see which columns an inferring reader loses. The two wall times are NOT a
    # comparison of the two methods: the first read warms the object store's and the host's caches
    # for the second, and swapping the order swaps which one looks fast.
    print("\nwhole-dataset reads (wall times are sequential, so not comparable to each other):")
    for with_schema in (True, False):
        result = _read_dataset(fs, args.registry, with_schema=with_schema)
        report.setdefault("dataset_reads", []).append(result)
        label = "schema stated" if with_schema else "schema inferred"
        print(f"  {label:<16} {result['rows']:>9,} rows, {len(result['columns'])} columns, {result['wall_s']}s")
    stated, inferred = report["dataset_reads"]
    dropped = sorted(set(stated["columns"]) - set(inferred["columns"]))
    print(f"  columns an inferring reader loses: {dropped or 'none'}")
    report["columns_lost_by_inference"] = dropped

    if args.aoi:
        west, south, east, north = (float(v) for v in args.aoi.split(","))
        report["aoi_query"] = _aoi_query(fs, args.registry, (west, south, east, north), args.aoi_year)
        query = report["aoi_query"]
        print(f"\narea-of-interest query ({args.aoi}, {args.aoi_year}): {query['wall_s']}s")
        print(
            f"  tiles overlapping {query['tiles_overlapping']}, embedded {query['tiles_embedded']}, "
            f"not embedded {query['tiles_not_embedded']}, refused px {query['refused_px_total']:,}"
        )

    disagreements: list[dict[str, Any]] = []
    if args.verify_zones:
        zones = args.verify_zones.split(",")
        findings = _verify_against_store(fs, args.registry, args.store, args.region, zones, anonymous=args.anonymous)
        report["store_cross_check"] = findings
        print(f"\ncross-check against the store ({len(zones)} zone(s)):")
        print(f"  {'cell':<12} {'complete':>8} {'rows':>6} {'embedded':>9} {'shards':>7}  verdict")
        for finding in findings:
            counts_agree = finding["registry_embedded"] == finding["store_live_shards"]
            # A cell holding data and NOT marked complete is the half-published state this audit
            # records `marked_complete` to catch: a fill that wrote its shards and its registry
            # part, then died before adding the year to `years_complete`. Its counts agree, so
            # counting alone would call it healthy.
            populated = bool(finding["registry_embedded"] or finding["store_live_shards"])
            unmarked = populated and not finding["marked_complete"]
            verdict = "agrees" if counts_agree and not unmarked else ("UNMARKED" if unmarked else "DISAGREES")
            if not counts_agree or unmarked:
                disagreements.append({**finding, "verdict": verdict})
            print(
                f"  {finding['zone']}/{finding['year']:<7} {finding['marked_complete']!s:>8} "
                f"{finding['registry_rows']:>6} {finding['registry_embedded']:>9} "
                f"{finding['store_live_shards']:>7}  {verdict}"
            )
        report["disagreements"] = disagreements

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.json_out}")

    # EVERY category the schema audit collects, not the two that looked most serious. A part whose
    # own metadata contradicts its partition path carries two conflicting identities, and a part
    # holding a column the schema does not declare means the writer and this checker disagree about
    # the schema — automation exiting 0 on either has accepted a registry it should not have.
    audit = report.get("schema_audit", {})
    broken = any(
        audit.get(key)
        for key in (
            "parts_missing_declared_columns",
            "parts_with_undeclared_columns",
            "parts_with_wrong_column_types",
            "parts_whose_metadata_disagrees_with_its_path",
        )
    )
    return 1 if (unparsed or broken or disagreements) else 0


if __name__ == "__main__":
    sys.exit(main())
