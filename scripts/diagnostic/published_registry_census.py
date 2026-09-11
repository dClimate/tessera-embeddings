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
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

if TYPE_CHECKING:
    import pyarrow.fs

from tessera_embeddings.storage import published_store, zone_grid
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


#: The campaign's preallocated time axis. A partition key outside it names a cell the store has no
#: slot for, however well-formed the part under it is.
CAMPAIGN_YEARS = tuple(range(2017, 2026))

#: The three refusal reasons `refused_px` is the sum of. Mirrored from the registry module so this
#: check and the writer name the same three.
_REFUSAL_REASONS = ("refused_no_optical_px", "refused_thin_px", "refused_no_radar_px")


def _refusals_add_up(row: dict[str, Any]) -> bool:
    """Whether ``refused_px`` equals the sum of its three reason columns.

    True when any of the four is null, because null means "not measured" and a comparison against
    an unmeasured total asserts something nobody recorded. Only rows carrying all four are checked.
    """
    total = row.get("refused_px")
    parts = [row.get(reason) for reason in _REFUSAL_REASONS]
    if total is None or any(part is None for part in parts):
        return True
    return int(total) == sum(int(part) for part in parts)


def _rows_with_null_bbox(fs: pyarrow.fs.FileSystem, prefix: str, year: int) -> list[str]:
    """Tiles in ``year`` whose bounding box has a null component, so no area query can find them."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(prefix, filesystem=fs, partitioning="hive", schema=dataset_schema())
    missing = pc.field("bbox_west").is_null()
    for field in ("bbox_south", "bbox_east", "bbox_north"):
        missing = missing | pc.field(field).is_null()
    table = dataset.to_table(filter=(pc.field("year") == year) & missing, columns=["zone", "tile"])
    return sorted(f"{r['zone']}/{r['tile']}" for r in table.to_pylist())


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

    # Type AND nullability. `str(field.type)` alone accepts a part that declares `tile`, `run_id`,
    # `assembled_at` or `embedded` as nullable — and a null identity there breaks the latest-wins
    # selection or groups unrelated rows under a null tile.
    declared = {field.name: (str(field.type), field.nullable) for field in registry_schema()}
    missing: dict[str, list[str]] = {}
    extra: dict[str, list[str]] = {}
    retyped: dict[str, list[str]] = {}
    without_identity: list[str] = []
    for part in parts:
        schema = pq.read_schema(part["path"], filesystem=fs)
        present = {field.name: (str(field.type), field.nullable) for field in schema}
        if absent := sorted(set(declared) - set(present)):
            missing[part["path"]] = absent
        if surplus := sorted(set(present) - set(declared)):
            extra[part["path"]] = surplus
        if wrong := sorted(n for n in set(declared) & set(present) if declared[n] != present[n]):
            retyped[part["path"]] = wrong
        metadata = {k.decode(): v.decode() for k, v in (schema.metadata or {}).items() if k != b"pandas"}
        # Run id as well as zone and year. Parts are keyed BY RUN — that is what makes a refill add
        # a part instead of overwriting one — so a footer whose run id contradicts its own filename
        # has two provenances, and the latest-wins selection has no way to tell which is the row's.
        if (
            metadata.get("zone") != part["zone"]
            or metadata.get("year") != str(part["year"])
            or metadata.get("run_id") != part["run_id"]
        ):
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
        columns=[
            "zone",
            "tile",
            "embedded",
            "refused_px",
            *_REFUSAL_REASONS,
            "eligible_px",
            "median_obs_where_thin",
            "assembled_at",
        ],
    )
    # LATEST RUN PER TILE, for the same reason the store cross-check needs it: a refill leaves the
    # original part in place, so a tile filled twice would be counted twice here and its refused
    # pixels added together — inflating the very answer a consumer came for.
    raw = table.to_pylist()
    # Per (zone, year), because an area of interest can span both — and within each, the newest
    # RUN as a whole for the reason in `_newest_run`.
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in raw:
        by_cell.setdefault((str(row["zone"]), int(row["year"] or year)), []).append(row)
    rows = [r for cell_rows in by_cell.values() for r in _newest_run(cell_rows)]
    embedded = [r["embedded"] for r in rows]
    refused = [r["refused_px"] for r in rows if r["refused_px"] is not None]
    # Malformed timestamps have to reach the REPORT, not just the run selection. `_newest_run`
    # cannot order a run whose timestamps will not parse, so it never selects it — and an AOI-only
    # run would otherwise print a plausible answer built on the older rows and exit 0.
    bad_stamps = unparsable_stamps(raw)
    # A null bounding box makes every Arrow comparison in the filter above evaluate to null, so the
    # row is dropped from EVERY area query — silently undercounting the coverage this dataset
    # exists to report. The schema permits null, which is why it has to be checked rather than
    # assumed.
    null_bbox = _rows_with_null_bbox(fs, prefix, year)
    # `refused_px` is defined as the sum of the three reason columns, and nothing else checks it.
    inconsistent = [r["tile"] for r in rows if not _refusals_add_up(r)]
    return {
        "aoi": list(aoi),
        "year": year,
        "wall_s": round(time.monotonic() - started, 2),
        "tiles_overlapping": len(rows),
        "rows_before_dedup": table.num_rows,
        "zones_overlapping": sorted({str(r["zone"]) for r in rows}),
        "tiles_embedded": sum(1 for v in embedded if v),
        "tiles_not_embedded": sum(1 for v in embedded if not v),
        "refused_px_total": sum(refused),
        "rows_with_unparsable_assembled_at": bad_stamps[:10],
        "rows_with_a_null_bounding_box": null_bbox[:10],
        "rows_whose_refusal_reasons_do_not_sum": inconsistent[:10],
        "note": "antimeridian zones 01/60 are not handled by this overlap test",
    }


#: Registry tile labels are ``chunk_<shard_y>_<shard_x>`` — the same shard-grid coordinates
#: :func:`~tessera_embeddings.storage.published_store.live_shards` returns, which is what makes the
#: two directly comparable rather than only countable.
_TILE_LABEL = re.compile(r"^chunk_(\d+)_(\d+)$")


def _tile_coordinate(label: str) -> tuple[int, int] | None:
    """The ``(shard_y, shard_x)`` a tile label names, or None if it does not parse.

    Returns None rather than raising so an unrecognised label is REPORTED as a finding instead of
    aborting the audit — a renamed label scheme is exactly the sort of drift this exists to notice,
    and crashing on it would hide every other cell's verdict.
    """
    match = _TILE_LABEL.match(str(label))
    return (int(match.group(1)), int(match.group(2))) if match else None


def unparsable_stamps(rows: list[dict[str, Any]]) -> list[str]:
    """Tiles among ``rows`` whose ``assembled_at`` will not parse or compare.

    **Collected BEFORE deduplication**, over every row rather than the survivors. A malformed row
    is ordered last, so a tile that also has a valid older row loses the malformed one entirely —
    and asking the deduplicated set afterwards would report nothing while the stale coverage it
    silently selected is exactly the problem.
    """
    return sorted({f"{row.get('zone', '')}/{row['tile']}" for row in rows if _parse_stamp(row["assembled_at"]) is None})


def _parse_stamp(value: object) -> datetime | None:
    """``assembled_at`` as an aware datetime, or None when it will not parse or compare.

    Offset-NAIVE is treated as unparsable rather than assumed to be UTC: comparing a naive against
    an aware datetime raises, and guessing a zone would silently reorder runs. Current writers emit
    an offset, so a naive value is itself the finding.
    """
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo is not None else None


def _optical_skip_labels(group: zarr.Group, year: int) -> list[str]:
    """Tile labels the year's provenance records as skipped for want of usable optical imagery.

    The third oracle, and the only one that can see a refusal NEITHER array witnesses: a tile with
    no usable optical observations has no ``scales`` shard and no ``s2_obs_count`` shard either, so
    comparing those two cannot tell a missing registry row from a tile that was never land.
    ``runs[<year>].optical_skips.labels`` is the store's own list of them.

    Returns an empty list when the year has no provenance or records no labels — absence here is
    not evidence of a problem, because a cell that refused nothing legitimately records none.
    """
    runs = group.attrs.get("runs", {})
    entry = runs.get(str(year)) if isinstance(runs, dict) else None
    if not isinstance(entry, dict):
        return []
    skips = entry.get("optical_skips")
    labels = skips.get("labels") if isinstance(skips, dict) else None
    return [str(label) for label in labels] if isinstance(labels, list) else []


def _newest_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only the rows of the most recently assembled run among ``rows``.

    A registry part is one complete run of one cell, so "latest wins" is a choice between RUNS,
    not between rows. Picking per tile would build a union of runs — newer rows for the tiles a
    refill touched, older rows for any it dropped — which no run produced and which the store
    therefore cannot match.

    A run whose timestamps will not parse cannot be ordered, so it is never selected as the newest;
    :func:`unparsable_stamps` reports those separately, over every row rather than the survivors.
    """
    newest: tuple[datetime, str] | None = None
    for row in rows:
        stamp = _parse_stamp(row["assembled_at"])
        if stamp is None:
            continue
        candidate = (stamp, str(row["run_id"]))
        if newest is None or candidate > newest:
            newest = candidate
    if newest is None:
        return list(rows)
    return [r for r in rows if str(r["run_id"]) == newest[1]]


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
        # Shards holding OBSERVATION COUNTS, which is a different question from shards holding
        # embeddings. A tile that was imaged and then wholly refused has counts and no embeddings,
        # so this is the only independent record of the refused half of the registry — without it,
        # deleting every `embedded=False` row would leave the embedded-tile comparison unchanged
        # and the audit would still say "agrees".
        observed = published_store.live_shards(session, zone, "s2_obs_count")
        table = dataset.to_table(
            filter=pc.field("zone") == zone,
            columns=[
                "year",
                "embedded",
                "tile",
                "refused_px",
                *_REFUSAL_REASONS,
                "run_id",
                "assembled_at",
            ],
        )
        rows = table.to_pylist()
        # The UNION of the store's calendar and the registry's own years. Iterating only the store's
        # would load a part for a year the store has no slot for — `year=2026`, say — and never look
        # at it, so a registry advertising a cell that cannot exist would pass every check.
        registry_years = {int(r["year"]) for r in rows if r["year"] is not None}
        for year in sorted(set(calendar) | registry_years):
            time_index = calendar.index(year) if year in calendar else None
            all_rows = [r for r in rows if r["year"] == year]
            # LATEST RUN PER TILE, not every row. A refill deliberately writes a NEW part rather
            # than overwriting the original, so the registry holds one complete tile set per run —
            # and summing them all would count a two-run cell's tiles twice against a store that
            # holds one shard each, reporting a correct cell as a disagreement. `assembled_at` is
            # the clock the registry provides for exactly this decision; a run id is not one.
            # Duplicates WITHIN one run, before latest-wins hides them. Cross-run duplication is
            # by design; the same `(run_id, tile)` twice is not, and a consumer reading the Parquet
            # directly gets both rows and doubles that tile's coverage and refusal counts.
            within_run: dict[tuple[str, str], int] = {}
            for row in all_rows:
                key = (str(row["run_id"]), str(row["tile"]))
                within_run[key] = within_run.get(key, 0) + 1
            duplicated = sorted(f"{run}/{tile}" for (run, tile), n in within_run.items() if n > 1)
            # The newest RUN as a whole, not the newest row per tile. A part is a complete run of
            # a cell, so taking newer rows for some tiles and older rows for others would
            # synthesise a union no run ever produced — and if a refill legitimately dropped a tile
            # the earlier fill had, the stale row would survive and be compared against a store
            # that no longer holds it.
            registry_rows = _newest_run(all_rows)
            # The COORDINATES, not the count. One embedded tile missing and one wrongly marked
            # embedded leaves the cardinalities equal, and a registry whose whole job is to say
            # WHERE coverage is would then be pointing consumers at the wrong tiles while this
            # audit said "agrees". `live_shards` already has the exact pairs.
            registry_tiles = {
                _tile_coordinate(r["tile"]) for r in registry_rows if r["embedded"] and _tile_coordinate(r["tile"])
            }
            store_tiles = coverage.get(time_index, frozenset()) if time_index is not None else frozenset()
            observed_tiles = observed.get(time_index, frozenset()) if time_index is not None else frozenset()
            refused_tiles = {
                _tile_coordinate(r["tile"]) for r in registry_rows if not r["embedded"] and _tile_coordinate(r["tile"])
            }
            # A tile with observation counts and no embeddings was evaluated and refused, so the
            # registry must carry a not-embedded row for it. The converse does NOT hold: a tile
            # refused for having NO imagery at all has no counts either — neither array witnesses
            # it, so a missing row for that case would be invisible here. The store records those
            # separately, in the year's own provenance, and they are folded in below.
            skipped = {
                coord for label in _optical_skip_labels(group, year) if (coord := _tile_coordinate(label)) is not None
            }
            must_be_refused = ((observed_tiles - store_tiles) | skipped) - store_tiles
            unrecorded_refusals = sorted(must_be_refused - refused_tiles)
            unparsed_tiles = sorted({r["tile"] for r in registry_rows if not _tile_coordinate(r["tile"])})
            bad_stamps = unparsable_stamps(all_rows)
            findings.append(
                {
                    "zone": zone,
                    "year": year,
                    "in_store_time_axis": time_index is not None,
                    "marked_complete": year in years,
                    "registry_rows": len(registry_rows),
                    "registry_rows_before_dedup": len(all_rows),
                    "runs_present": sorted({r["run_id"] for r in all_rows}),
                    "registry_embedded": sum(1 for r in registry_rows if r["embedded"]),
                    "registry_not_embedded": sum(1 for r in registry_rows if not r["embedded"]),
                    "store_live_shards": len(store_tiles),
                    "in_registry_only": sorted(registry_tiles - store_tiles)[:20],
                    "in_store_only": sorted(store_tiles - registry_tiles)[:20],
                    "tiles_disagreeing": len(registry_tiles ^ store_tiles),
                    "refused_tiles_the_registry_omits": unrecorded_refusals[:20],
                    "unparsable_tile_labels": unparsed_tiles[:10],
                    "unparsable_assembled_at": bad_stamps[:10],
                    "duplicated_within_a_run": duplicated[:10],
                    "refusal_reasons_do_not_sum": [r["tile"] for r in registry_rows if not _refusals_add_up(r)][:10],
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
    # A zone key that parses but names no real zone is as much a defect as one that does not parse:
    # `zone=61N` is syntactically fine, passes the schema audit, and advertises coverage in a zone
    # the published store cannot contain. Checked against the zone grid, the same authority the
    # store's own group names come from.
    unparsed = [
        part["path"]
        for part in parts
        if part["zone"] is None
        or part["year"] is None
        or part["zone"] not in zone_grid.ZONES
        or part["year"] not in CAMPAIGN_YEARS
    ]
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for part in parts:
        if (
            part["zone"] is None
            or part["year"] is None
            or part["zone"] not in zone_grid.ZONES
            or part["year"] not in CAMPAIGN_YEARS
        ):
            continue
        cells.setdefault((part["zone"], part["year"]), []).append(part)
    refilled = {f"{z}/{y}": len(v) for (z, y), v in sorted(cells.items()) if len(v) > 1}

    print(f"registry:      {args.registry} ({args.region})")
    print(f"parts:         {len(parts)} in {listing_s}s, {sum(p['size'] for p in parts) / 1e6:.1f} MB")
    print(f"cells covered: {len(cells)}")
    print(f"cells with more than one part (a refill): {len(refilled)} {list(refilled)[:8]}")
    print(f"parts whose path does not parse or names no real zone: {len(unparsed)} {unparsed[:4]}")

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
            f"  tiles overlapping {query['tiles_overlapping']} in zones {query['zones_overlapping']}, "
            f"embedded {query['tiles_embedded']}, not embedded {query['tiles_not_embedded']}, "
            f"refused px {query['refused_px_total']:,}"
        )
        print(
            f"  rows with an unparsable timestamp {len(query['rows_with_unparsable_assembled_at'])}, "
            f"a null bounding box {len(query['rows_with_a_null_bounding_box'])}, "
            f"refusal reasons that do not sum {len(query['rows_whose_refusal_reasons_do_not_sum'])}"
        )

    disagreements: list[dict[str, Any]] = []
    if args.verify_zones:
        zones = args.verify_zones.split(",")
        findings = _verify_against_store(fs, args.registry, args.store, args.region, zones, anonymous=args.anonymous)
        report["store_cross_check"] = findings
        print(f"\ncross-check against the store ({len(zones)} zone(s)):")
        print(f"  {'cell':<12} {'complete':>8} {'rows':>6} {'embedded':>9} {'shards':>7}  verdict")
        for finding in findings:
            # Coordinate equality, which implies count equality and catches what it cannot.
            counts_agree = (
                finding["tiles_disagreeing"] == 0
                and not finding["unparsable_tile_labels"]
                and not finding["unparsable_assembled_at"]
                and not finding["duplicated_within_a_run"]
                and not finding["refusal_reasons_do_not_sum"]
                and not finding["refused_tiles_the_registry_omits"]
            )
            # A cell holding data and NOT marked complete is the half-published state this audit
            # records `marked_complete` to catch: a fill that wrote its shards and its registry
            # part, then died before adding the year to `years_complete`. Its counts agree, so
            # counting alone would call it healthy.
            # ANY registry row counts as populated, not only an embedded one. A cell whose part
            # holds nothing but refused tiles has zero embedded rows and zero shards, so counting
            # those alone calls it empty — while the registry is still advertising a record for a
            # cell nothing marks complete.
            populated = bool(finding["registry_rows"] or finding["store_live_shards"])
            unmarked = populated and not finding["marked_complete"]
            outside = not finding["in_store_time_axis"]
            verdict = "agrees"
            if outside:
                verdict = "NOT IN THE STORE'S TIME AXIS"
            elif unmarked:
                verdict = "UNMARKED"
            elif not counts_agree:
                verdict = "DISAGREES"
            if verdict != "agrees":
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
    aoi = report.get("aoi_query", {})
    aoi_broken = any(
        aoi.get(key)
        for key in (
            "rows_with_unparsable_assembled_at",
            "rows_with_a_null_bounding_box",
            "rows_whose_refusal_reasons_do_not_sum",
        )
    )
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
    return 1 if (unparsed or broken or disagreements or aoi_broken) else 0


if __name__ == "__main__":
    sys.exit(main())
