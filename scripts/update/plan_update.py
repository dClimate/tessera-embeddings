r"""Write a targeted update's plan, checked against its land mask. Dry run unless ``--write``.

An update re-runs chosen shards of zone-years the campaign has already filled
(``src/tessera_embeddings/storage/update.py``, ADR 027). This writes the plan the flows read —
``fill-zones-sequential`` and ``run-global-campaign`` both take ``update=<name>`` — from either:

* ``from-registry``: the shards whose ``master/`` rows refused pixels (``--refused``, for one reason
  or any), narrowed by ``--zones``, ``--years`` and an ``--aoi`` box, e.g. every shard with thin
  refusals in a cloudy region::

      plan_update.py from-registry --name amazon-thin-2026-10 --mode holes \
          --registry s3://tessera-embeddings/v1.1/dclimate.registry --anonymous \
          --refused thin --aoi=-75,-15,-50,5 --years 2021,2022 \
          --inputs s3://global-tessera-inputs-prod

* ``from-cells``: a CSV of ``zone,year,tile`` rows, a blank ``tile`` meaning every live shard. Any
  other selection is a CSV, built however you like.

``--optical-min-obs`` gives the update a minimum optical depth looser than the store's, for its own
cells only (ADR 028): ``--mode holes --refused thin --optical-min-obs 1`` embeds every pixel an older
line refused as thin. The store's attrs keep its own rule; each cell's update record states this one.

Before anything is written the plan is checked against its mask: a zone the mask lacks, or a shard
outside a zone's grid, refuses; shards the mask holds dead are reported, since the fill drops them.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, cast, get_args

import icechunk
import numpy as np
import polars as pl

from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.ingest.land_mask import narrow_tile_live
from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials
from tessera_embeddings.storage.update import (
    Refused,
    UpdatePlan,
    cell_key,
    plan_from_registry,
    plan_from_tiles,
    plan_uri,
    save_plan,
)
from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group


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


def main(argv: list[str] | None = None) -> int:
    """Build, check and (with ``--write``) save one plan."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["from-registry", "from-cells"])
    parser.add_argument("--name", required=True, help="update name: lowercase letters, digits and hyphens")
    parser.add_argument("--mode", required=True, choices=["holes", "shards"])
    parser.add_argument("--inputs", required=True, help="the deployment's inputs URI (BucketPaths.inputs)")
    parser.add_argument("--mask", default="global", help="land-mask name (BucketPaths.land_mask_store)")
    parser.add_argument("--on-skip", default="keep", choices=["keep", "clear"])
    parser.add_argument(
        "--optical-min-obs", type=int, help="a depth rule looser than the store's, for this update only"
    )
    parser.add_argument("--registry", help="from-registry: the registry root (beside the store)")
    parser.add_argument("--anonymous", action="store_true", help="from-registry: read a public registry unsigned")
    parser.add_argument(
        "--refused", choices=get_args(Refused), help="from-registry: keep tiles that refused pixels for this reason"
    )
    parser.add_argument("--aoi", help="from-registry: west,south,east,north in WGS84 degrees, as --aoi=W,S,E,N")
    parser.add_argument("--zones", help="comma-separated zones")
    parser.add_argument("--years", help="comma-separated years")
    parser.add_argument("--cells", help="from-cells: CSV of zone,year,tile")
    parser.add_argument("--region", default=None, help="S3 region of the registry and the mask (default us-west-2)")
    parser.add_argument("--write", action="store_true", help="save the plan; without it this is a dry run")
    args = parser.parse_args(argv)

    fields: dict[str, Any] = {"mask_name": args.mask, "on_skip": args.on_skip, "optical_min_obs": args.optical_min_obs}
    if args.source == "from-registry":
        if not args.registry:
            parser.error("from-registry needs --registry")
        plan = plan_from_registry(
            args.registry,
            name=args.name,
            mode=args.mode,
            refused=args.refused,
            zones=args.zones.split(",") if args.zones else None,
            years=[int(y) for y in args.years.split(",")] if args.years else None,
            aoi=cast("tuple[float, float, float, float]", tuple(map(float, args.aoi.split(",")))) if args.aoi else None,
            # The region always: without it the scan asks us-east-1 and S3 answers with a redirect.
            storage_options={
                "aws_region": args.region or "us-west-2",
                **({"skip_signature": "true"} if args.anonymous else {}),
            },
            **fields,
        )
    else:
        if not args.cells:
            parser.error("from-cells needs --cells")
        tiles = pl.read_csv(args.cells, schema={"zone": pl.String, "year": pl.Int64, "tile": pl.String})
        plan = plan_from_tiles(tiles, name=args.name, mode=args.mode, source=args.cells, **fields)

    paths = BucketPaths(inputs=args.inputs, outputs="")
    report = check_against_mask(plan, paths.land_mask_store(plan.mask_name), region=args.region)
    for row in report:
        print(
            f"{row['cell']:>10}  {row['planned']:>6} planned  {row['live']:>6} live  {row['dead']:>6} dead in the mask"
        )
    print(f"{len(report)} zone-year(s), {sum(r['live'] for r in report)} live tile(s) to update")
    print("Optical depth rule:", plan.optical_min_obs or "the store's own")
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
