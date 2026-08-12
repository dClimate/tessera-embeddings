#!/usr/bin/env python
"""How evenly N Ray clusters divide the campaign's work, measured from the real mask.

The campaign's schedule is set by the LAST cluster to finish, and clusters are long-lived, so an
uneven split is not averaged away — it is added to the campaign date. This is the measurement
behind the "clusters are balanced on work, not on area" claim in `campaign-plan.md` §10, and it is
a script rather than a paragraph so the figure can be re-derived when the mask changes.

It reads each zone's live-tile bitmap once (one GET per zone, metadata and one small array), scores
each zone two ways, then runs the campaign's OWN partitioner:

* **work** — live tiles weighted by their latitude band's observation count (``zone_work_weight``),
  which is proportional to GPU-hours;
* **tiles** — a raw live-tile count, which is what balancing on area would use.

Both splits are then scored by TRUE WORK, because that is what decides when a cluster finishes.
The comparison is the point: it says what balancing on area would cost, in the only currency that
matters.

Usage::

    AWS_PROFILE=global-tessera-prod uv run python scripts/cluster_work_spread.py
    AWS_PROFILE=global-tessera-prod uv run python scripts/cluster_work_spread.py --clusters 8 10 16

Takes a few minutes: 112 land zones x 2 reads, and the reads are remote.
"""

from __future__ import annotations

import argparse

from tessera_embeddings.orchestration.prefect.flows import run_global_campaign as campaign
from tessera_embeddings.orchestration.runners.zone_fill import zone_live_tile_count, zone_work_weight
from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials
from tessera_embeddings.storage.zone_grid import ZONES

DEFAULT_MASK = "s3://global-tessera-inputs/masks/global.icechunk"


def _read(mask: str, region: str | None) -> tuple[dict[str, float], dict[str, float]]:
    """Per-zone work weight and live-tile count, for every zone that holds land."""
    work: dict[str, float] = {}
    tiles: dict[str, float] = {}
    for i, zone in enumerate(ZONES):
        weight = zone_work_weight(mask, zone, get_credentials=iam_icechunk_credentials, s3_region=region)
        if weight <= 0:
            continue  # all-ocean
        work[zone] = weight
        tiles[zone] = float(
            zone_live_tile_count(mask, zone, get_credentials=iam_icechunk_credentials, s3_region=region)
        )
        if i % 20 == 0:
            print(f"  ...{i}/{len(ZONES)} zones read", flush=True)
    return work, tiles


def _spread(n: int, balance_on: dict[str, float], score_by: dict[str, float]) -> tuple[float, list[float]]:
    """Run the real partitioner on one weighting, then score the result by another.

    The partitioner reads the mask itself, so its reader is substituted with the values already
    read — the SPLIT is the production one either way, which is the whole point of not
    reimplementing the dealing here.
    """
    real = campaign.zone_work_weight
    campaign.zone_work_weight = lambda _mask, zone, **_kw: float(balance_on[zone])
    try:
        groups = campaign._partition_by_live_tiles(sorted(balance_on), n, land_mask_path="<in-memory>")
    finally:
        campaign.zone_work_weight = real
    totals = [sum(score_by[z] for z in group) for group in groups]
    return max(totals) / min(totals) - 1, totals


def main() -> None:
    """Read the mask once, then report the spread at each requested cluster count."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask", default=DEFAULT_MASK, help="land-mask store (default: %(default)s)")
    ap.add_argument("--region", default=None, help="S3 region for the mask store")
    ap.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[10],
        help="cluster counts to report; the campaign runs 10 (default: %(default)s)",
    )
    args = ap.parse_args()

    print(f"reading {args.mask}")
    work, tiles = _read(args.mask, args.region)
    print(f"{len(work)} zones hold land; total work {sum(work.values()):,.0f} tile-token units\n")

    print(f"{'clusters':>8}  {'balanced on WORK':>18}  {'balanced on TILES':>18}")
    for n in args.clusters:
        by_work, _ = _spread(n, work, work)
        by_tiles, _ = _spread(n, tiles, work)
        print(f"{n:>8}  {by_work:>17.4%}  {by_tiles:>17.4%}")
    print("\nBoth columns are the spread in TRUE WORK (heaviest cluster over lightest, minus one).")


if __name__ == "__main__":
    main()
