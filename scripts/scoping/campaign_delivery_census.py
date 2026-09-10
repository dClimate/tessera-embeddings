"""What the global campaign delivered, read from the published store and the land mask.

Produces the delivery half of ``context_docs/campaign/campaign-cost-model.md`` section 12: how
many cells and tile-years were published, how many were refused, and the delivery curve over
time.

    python scripts/scoping/campaign_delivery_census.py
    python scripts/scoping/campaign_delivery_census.py --daily --json out.json

Both sides of the fraction come from primary data. The land mask carries each zone's live-tile
count as a stored attribute, which gives the roster; each zone group in the published store
carries ``years_complete`` and a per-year ``runs`` entry, which gives what landed. Neither is
taken from a document.

THREE DISTINCTIONS THAT DECIDE WHETHER THE PERCENTAGE MEANS ANYTHING:

* **A cell published as ``empty`` is counted separately, never folded in.** It is a genuine
  completion of the roster carrying no land, so counting it as delivered inflates the cell
  count while contributing no tiles, and dropping it silently makes the roster look unfinished.
  Both are reported, and land cells published empty are distinguished from landless ones --
  the first mean every tile was refused, which is a finding; the second is just ocean.
* **Every roster cell is reconciled into exactly one bucket**, and the script says so. Without
  that check a completion percentage is an assertion about the numerator only.
* **The delivery curve is reported per day, not as an average.** A campaign that is relaunched
  inherits work already computed and staged, so publication can drain a backlog at a rate no
  compute could sustain; a single mean over the whole span hides that and reads as throughput.
  Section 12 records a day that published 505 cells for exactly this reason.

Read-only. Needs credentials for the mask bucket and read access to the published store.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials
from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group

MASK_URI = "s3://global-tessera-inputs/masks/global.icechunk"
STORE_URI = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
YEARS = tuple(range(2017, 2026))
REGION = "us-west-2"


def read_mask() -> dict[str, int]:
    """Per-zone live-tile counts from the mask's own stored attribute."""
    mask = open_store_as_zarr_group(MASK_URI, get_credentials=iam_icechunk_credentials, region=REGION)
    return {z: int(dict(mask[z].attrs).get("n_live_tiles") or 0) for z in mask.group_keys()}


def read_store(land: dict[str, int]) -> tuple[dict, dict]:
    """(published_with_data, published_empty), each keyed ``zone|year``."""
    store = open_store_as_zarr_group(STORE_URI, get_credentials=iam_icechunk_credentials, region=REGION)
    published: dict[str, dict] = {}
    empty: dict[str, dict] = {}
    for zone in store.group_keys():
        attrs = dict(store[zone].attrs)
        runs = attrs.get("runs") or {}
        for year in attrs.get("years_complete") or []:
            run = runs.get(str(year)) or {}
            record = {"assembled_at": run.get("assembled_at"), "tiles": land.get(zone, 0)}
            (empty if run.get("empty") else published)[f"{zone}|{year}"] = record
    return published, empty


def main(argv: list[str] | None = None) -> int:
    """Print the delivery census. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daily", action="store_true", help="also print the per-day delivery curve")
    ap.add_argument("--json", dest="json_path", help="write the raw census to this path")
    args = ap.parse_args(argv)

    land = read_mask()
    with_land = {z: n for z, n in land.items() if n > 0}
    per_year = sum(with_land.values())
    roster_tiles = per_year * len(YEARS)
    print(f"mask: {len(land)} zones, {len(with_land)} with land, {per_year:,} live tiles per year")
    print(f"roster: {per_year:,} x {len(YEARS)} years = {roster_tiles:,} tile-years")

    published, empty = read_store(land)
    delivered = sum(v["tiles"] for v in published.values())

    roster_land = {f"{z}|{y}" for z in with_land for y in YEARS}
    roster_none = {f"{z}|{y}" for z, n in land.items() if n == 0 for y in YEARS}
    empty_land = set(empty) & roster_land
    empty_none = set(empty) & roster_none
    missing = roster_land - set(published) - set(empty)
    missing_none = roster_none - set(published) - set(empty)

    print(f"\n{'bucket':<44} {'cells':>6} {'tile-years':>13}")
    print(f"{'published with data':<44} {len(published):>6} {delivered:>13,}")
    print(f"{'published empty — landless zone':<44} {len(empty_none):>6} {0:>13}")
    print(
        f"{'published empty — land, every tile refused':<44} {len(empty_land):>6} "
        f"{sum(land[k.split('|')[0]] for k in empty_land):>13,}"
    )
    print(
        f"{'not published':<44} {len(missing) + len(missing_none):>6} "
        f"{sum(land[k.split('|')[0]] for k in missing):>13,}"
    )
    total_cells = len(published) + len(empty) + len(missing) + len(missing_none)
    roster_cells = len(roster_land) + len(roster_none)
    print(f"{'ROSTER':<44} {roster_cells:>6} {roster_tiles:>13,}")

    # The reconciliation is the point: without it the percentage describes the numerator alone.
    if total_cells != roster_cells:
        stray = (set(published) | set(empty)) - roster_land - roster_none
        print(f"\nNOT RECONCILED: buckets sum to {total_cells}, roster is {roster_cells}", file=sys.stderr)
        if stray:
            print(f"  published but not in the roster: {sorted(stray)[:10]}", file=sys.stderr)
        return 1
    print(f"\nreconciled: every one of {roster_cells} roster cells in exactly one bucket")
    print(f"delivered {delivered:,} of {roster_tiles:,} tile-years = {delivered / roster_tiles * 100:.2f}%")

    if missing:
        by_zone = collections.Counter(k.split("|")[0] for k in missing)
        print(f"\nnot published, by zone: {dict(sorted(by_zone.items()))}")

    stamps = sorted(dt.datetime.fromisoformat(v["assembled_at"]) for v in published.values() if v["assembled_at"])
    if len(stamps) < 2:
        print("\ntoo few timestamps for a delivery curve")
    else:
        span_h = (stamps[-1] - stamps[0]).total_seconds() / 3600
        print(
            f"\ndelivery span {stamps[0]:%Y-%m-%d %H:%M}Z .. {stamps[-1]:%Y-%m-%d %H:%M}Z"
            f" = {span_h:.1f} h ({span_h / 24:.2f} days)"
        )
        print(
            f"end-to-end rate {delivered / span_h:,.0f} tile-years/h"
            f"  — includes ramp, restarts and tail; NOT a compute rate"
        )
        if args.daily:
            tiles_by_day: collections.Counter = collections.Counter()
            cells_by_day: collections.Counter = collections.Counter()
            for v in published.values():
                if v["assembled_at"]:
                    day = dt.datetime.fromisoformat(v["assembled_at"]).date()
                    tiles_by_day[day] += v["tiles"]
                    cells_by_day[day] += 1
            print(f"\n{'day':<12} {'tile-years':>12} {'cells':>6} {'rate/h':>9}")
            for day in sorted(tiles_by_day):
                print(f"{day!s:<12} {tiles_by_day[day]:>12,} {cells_by_day[day]:>6} {tiles_by_day[day] / 24:>9,.0f}")

    if args.json_path:
        with Path(args.json_path).open("w") as fh:
            json.dump({"land": land, "published": published, "empty": empty, "roster": roster_tiles}, fh)
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
