r"""Write a targeted update's plan, checked against its land mask. Dry run unless ``--write``.

An update re-runs chosen shards of zone-years the campaign has already filled
(``src/tessera_embeddings/storage/update.py``, ADR 027). This writes the plan the flows read —
``fill-zones-sequential`` and ``run-global-campaign`` both take ``update=<name>`` — from either:

* ``from-registry``: the shards whose ``master/`` rows match a SQL ``--where`` and/or an ``--aoi``
  box, e.g. every shard with thin refusals in a cloudy region::

      plan_update.py from-registry --name amazon-thin-2026-10 --mode holes \
          --registry s3://tessera-embeddings/v1.1/dclimate.registry --anonymous \
          --where "refused_thin_px > 0" --aoi -75,-15,-50,5 --years 2021,2022 \
          --inputs s3://global-tessera-inputs-prod

* ``from-cells``: a CSV of ``zone,year,tile`` rows, a blank ``tile`` meaning every live shard.

Before anything is written the plan is checked against its mask: a zone the mask lacks, or a shard
outside a zone's grid, refuses; shards the mask holds dead are reported, since the fill drops them.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import icechunk
import numpy as np
import polars as pl

from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.ingest.land_mask import narrow_tile_live
from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials
from tessera_embeddings.storage.update import UpdatePlan, cell_key, plan_from_registry, plan_uri, save_plan
from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group
from tessera_embeddings.storage.zone_grid import canonicalize_zone


def _credentials(uri: str) -> Callable[[], icechunk.S3StaticCredentials] | None:
    """Our buckets' credential callback for an S3 mask, none for a local one."""
    return iam_icechunk_credentials if uri.startswith("s3://") else None


def check_against_mask(plan: UpdatePlan, mask_path: str, *, region: str | None = None) -> list[dict[str, Any]]:
    """Per cell: shards planned, and how many the mask holds live. Raises on a plan for another grid."""
    report = []
    for zone, year in plan.cell_list():
        try:
            group = open_store_as_zarr_group(
                mask_path, group=zone, get_credentials=_credentials(mask_path), region=region
            )
        except (KeyError, FileNotFoundError) as exc:
            raise SystemExit(f"Zone {zone} is not in the mask at {mask_path}") from exc
        tile_live = np.asarray(group["tile_live_2048"], dtype=bool)
        tiles = plan.cells[cell_key(zone, year)]
        live = int(narrow_tile_live(tile_live, tiles).sum())
        planned = len(tiles) if tiles is not None else int(tile_live.sum())
        report.append({"cell": f"{zone}/{year}", "planned": planned, "live": live, "dead": planned - live})
    return report


def _cells_from_csv(path: str) -> dict[str, tuple[str, ...] | None]:
    """``zone,year,tile`` rows grouped by cell; a blank tile asks for every live shard.

    Zones are canonicalised before grouping, so ``7S`` and ``07S`` rows land in one cell.
    """
    rows = pl.read_csv(path, schema={"zone": pl.String, "year": pl.Int64, "tile": pl.String})
    rows = rows.with_columns(pl.col("zone").map_elements(canonicalize_zone, return_dtype=pl.String))
    cells: dict[str, tuple[str, ...] | None] = {}
    for (zone, year), group in rows.group_by("zone", "year"):
        tiles = group["tile"].drop_nulls().to_list()
        cells[cell_key(str(zone), int(year))] = None if group["tile"].null_count() else tuple(tiles)
    return cells


def main(argv: list[str] | None = None) -> int:
    """Build, check and (with ``--write``) save one plan."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["from-registry", "from-cells"])
    parser.add_argument("--name", required=True, help="update name: lowercase letters, digits and hyphens")
    parser.add_argument("--mode", required=True, choices=["holes", "shards"])
    parser.add_argument("--inputs", required=True, help="the deployment's inputs URI (BucketPaths.inputs)")
    parser.add_argument("--mask", default="global", help="land-mask name (BucketPaths.land_mask_store)")
    parser.add_argument("--on-skip", default="keep", choices=["keep", "clear"])
    parser.add_argument("--min-valid-coverage", type=float, default=None, help="percent of the ROI; default 1e-9")
    parser.add_argument("--strict-window", action="store_true", help="require a valid date in every month")
    parser.add_argument("--registry", help="from-registry: the registry root (beside the store)")
    parser.add_argument("--anonymous", action="store_true", help="from-registry: read a public registry unsigned")
    parser.add_argument("--where", help="from-registry: SQL over master/ columns")
    parser.add_argument("--aoi", help="from-registry: west,south,east,north in WGS84 degrees")
    parser.add_argument("--zones", help="comma-separated zones")
    parser.add_argument("--years", help="comma-separated years")
    parser.add_argument("--cells", help="from-cells: CSV of zone,year,tile")
    parser.add_argument("--region", default=None, help="S3 region for the mask read")
    parser.add_argument("--write", action="store_true", help="save the plan; without it this is a dry run")
    args = parser.parse_args(argv)

    fields: dict[str, Any] = {
        "mask_name": args.mask,
        "on_skip": args.on_skip,
        "allow_partial_window": not args.strict_window,
    }
    if args.min_valid_coverage is not None:
        fields["min_valid_coverage"] = args.min_valid_coverage
    if args.source == "from-registry":
        if not args.registry:
            parser.error("from-registry needs --registry")
        aoi = None
        if args.aoi:
            west, south, east, north = (float(v) for v in args.aoi.split(","))
            aoi = (west, south, east, north)
        plan = plan_from_registry(
            args.registry,
            name=args.name,
            mode=args.mode,
            zones=args.zones.split(",") if args.zones else None,
            years=[int(y) for y in args.years.split(",")] if args.years else None,
            where=args.where,
            aoi=aoi,
            storage_options={"skip_signature": "true", "aws_region": args.region or "us-west-2"}
            if args.anonymous
            else None,
            **fields,
        )
    else:
        if not args.cells:
            parser.error("from-cells needs --cells")
        plan = UpdatePlan(
            name=args.name, mode=args.mode, cells=_cells_from_csv(args.cells), source=args.cells, **fields
        )

    paths = BucketPaths(inputs=args.inputs, outputs="")
    report = check_against_mask(plan, paths.land_mask_store(plan.mask_name), region=args.region)
    for row in report:
        print(
            f"{row['cell']:>10}  {row['planned']:>6} planned  {row['live']:>6} live  {row['dead']:>6} dead in the mask"
        )
    print(f"{len(report)} cell(s), {sum(r['live'] for r in report)} live shard(s) to update")
    # The runner refuses to clear a tile the mask has no land in, and a written plan cannot be redone.
    if plan.on_skip == "clear" and (dead := [r["cell"] for r in report if r["dead"]]):
        raise SystemExit(f"A clearing plan cannot target shards dead in the mask, as in {dead}; leave them out")
    uri = plan_uri(paths.update_root(plan.name))
    if args.write:
        save_plan(plan, uri)
        print(f"Wrote {uri}")
    else:
        print(f"Dry run: pass --write to save the plan to {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
