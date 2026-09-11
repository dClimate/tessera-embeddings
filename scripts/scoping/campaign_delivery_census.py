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

* **A tile published as FILL is not a tile carrying data**, and the two are reported separately.
  Each run records ``optical_skips.tiles_skipped``: tiles whose every date the optical preflight
  refused, which assembly still writes over the cell's whole live footprint so the array carries no
  holes. Crediting a published cell's entire land footprint as delivered counts those, and they are
  a real fraction -- 4.2% of 2017. So the roster percentage and the *tile-years carrying embeddings*
  are two different numbers, and a token or throughput figure wants the second.
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

**It prints the snapshot ID of each store it read**, because a census of a live store is an
as-of and the reader needs to know which as-of. Both stores keep committing after a campaign
ends, so a later re-run reporting a higher percentage is a NEWER answer rather than a
contradiction -- and the snapshot IDs are what let someone tell those two apart.

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
from tessera_embeddings.storage.zarr_store import open_store_group_and_tip
from tessera_embeddings.storage.zone_grid import ZONES

MASK_URI = "s3://global-tessera-inputs/masks/global.icechunk"
STORE_URI = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
YEARS = tuple(range(2017, 2026))
REGION = "us-west-2"


def read_mask() -> tuple[dict[str, int], str]:
    """(per-zone live-tile counts, mask snapshot ID), from the mask's own stored attribute.

    A missing or unreadable ``n_live_tiles`` RAISES rather than reading as zero. Zero is a
    meaningful value here -- it means an ocean-only zone, whose nine cells are then expected to
    publish as empty -- so treating an absent attribute as zero would silently drop that zone's
    tiles from the denominator *and* reclassify its cells as landless, which the roster
    reconciliation cannot catch because both sides move together. The percentage this script
    exists to produce would come out too high, and nothing would say so.
    """
    mask, snapshot = open_store_group_and_tip(MASK_URI, get_credentials=iam_icechunk_credentials, region=REGION)
    # The zone roster is checked against the grid definition rather than taken from whatever the
    # mask happens to hold. A zone absent from the mask would otherwise drop out of the denominator
    # while the reconciliation still passed, because both sides of that check are built from this
    # same dictionary -- the identical failure shape as a missing ``n_live_tiles``, one level up.
    present = set(mask.group_keys())
    if present != set(ZONES):
        missing, extra = sorted(set(ZONES) - present), sorted(present - set(ZONES))
        raise SystemExit(f"mask zone roster does not match the grid: missing={missing} extra={extra}")
    land: dict[str, int] = {}
    for zone in mask.group_keys():
        raw = dict(mask[zone].attrs).get("n_live_tiles")
        if raw is None:
            raise SystemExit(f"mask zone {zone} has no n_live_tiles attribute — the roster cannot be derived")
        n = int(raw)
        if n < 0:
            raise SystemExit(f"mask zone {zone} has n_live_tiles={raw!r} — not a tile count")
        land[zone] = n
    return land, snapshot


def read_store(land: dict[str, int]) -> tuple[dict, dict, str]:
    """(published_with_data, published_empty, store snapshot ID), each keyed ``zone|year``.

    Each published record carries both tile counts: the cell's live footprint, and how much of it
    the optical preflight refused so assembly wrote it as fill. ``tiles_live`` comes from the run's
    own record rather than from the mask, which makes it an independent check ON the mask -- and
    the two agree exactly, to the tile, in all nine years.
    """
    store, snapshot = open_store_group_and_tip(STORE_URI, get_credentials=iam_icechunk_credentials, region=REGION)
    published: dict[str, dict] = {}
    empty: dict[str, dict] = {}
    for zone in store.group_keys():
        attrs = dict(store[zone].attrs)
        runs = attrs.get("runs") or {}
        for year in attrs.get("years_complete") or []:
            run = runs.get(str(year)) or {}
            skips = run.get("optical_skips") or {}
            record = {
                "assembled_at": run.get("assembled_at"),
                "tiles": land.get(zone, 0),
                "tiles_live_recorded": skips.get("tiles_live"),
                "skipped": int(skips.get("tiles_skipped") or 0),
            }
            (empty if run.get("empty") else published)[f"{zone}|{year}"] = record
    return published, empty, snapshot


def main(argv: list[str] | None = None) -> int:
    """Print the delivery census. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daily", action="store_true", help="also print the per-day delivery curve")
    ap.add_argument("--json", dest="json_path", help="write the raw census to this path")
    args = ap.parse_args(argv)

    land, mask_snapshot = read_mask()
    with_land = {z: n for z, n in land.items() if n > 0}
    per_year = sum(with_land.values())
    roster_tiles = per_year * len(YEARS)
    print(f"mask: {len(land)} zones, {len(with_land)} with land, {per_year:,} live tiles per year")
    print(f"roster: {per_year:,} x {len(YEARS)} years = {roster_tiles:,} tile-years")

    published, empty, store_snapshot = read_store(land)
    delivered = sum(v["tiles"] for v in published.values())
    skipped = sum(v["skipped"] for v in published.values())
    with_data = delivered - skipped

    # The run's own live-tile count against the mask's. Two instruments, no shared code: if they
    # disagree, one of the two footprints is wrong and the roster percentage means nothing.
    disagree = {
        k: (v["tiles"], v["tiles_live_recorded"])
        for k, v in published.items()
        if v["tiles_live_recorded"] is not None and int(v["tiles_live_recorded"]) != v["tiles"]
    }
    if disagree:
        print(f"\nMASK AND RUN RECORDS DISAGREE on the live footprint of {len(disagree)} cells:", file=sys.stderr)
        for k, (m, r) in sorted(disagree.items())[:10]:
            print(f"  {k}: mask {m:,} vs run record {r:,}", file=sys.stderr)
        return 1

    roster_land = {f"{z}|{y}" for z in with_land for y in YEARS}
    roster_none = {f"{z}|{y}" for z, n in land.items() if n == 0 for y in YEARS}
    empty_land = set(empty) & roster_land
    empty_none = set(empty) & roster_none
    missing = roster_land - set(published) - set(empty)
    missing_none = roster_none - set(published) - set(empty)

    print(f"\n{'bucket':<44} {'cells':>6} {'tile-years':>13}")
    print(f"{'published, cell carries data':<44} {len(published):>6} {delivered:>13,}")
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
    print(f"roster completion {delivered:,} of {roster_tiles:,} tile-years = {delivered / roster_tiles * 100:.2f}%")
    # The second number, and the one a token or throughput figure needs. The first says the cell
    # published; this says the tile holds an embedding.
    print(
        f"of which {skipped:,} tiles were published as FILL (optical preflight refused every date), "
        f"so\ntile-years CARRYING EMBEDDINGS {with_data:,} = {with_data / roster_tiles * 100:.2f}% of the roster"
    )
    by_year_skip: collections.Counter = collections.Counter()
    by_year_live: collections.Counter = collections.Counter()
    for k, v in published.items():
        by_year_skip[int(k.split("|")[1])] += v["skipped"]
        by_year_live[int(k.split("|")[1])] += v["tiles"]
    print(
        "  fill by year: "
        + "  ".join(
            f"{y}:{by_year_skip[y]:,}({by_year_skip[y] / by_year_live[y] * 100:.2f}%)" for y in sorted(by_year_live)
        )
    )

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

    # Last, so it is the line nearest the figures a reader copies out.
    print(f"\nread at mask snapshot {mask_snapshot}, store snapshot {store_snapshot}")
    print("Both stores keep committing. Quote the snapshot with the percentage, or the percentage")
    print("dates silently.")

    if args.json_path:
        with Path(args.json_path).open("w") as fh:
            json.dump(
                {
                    "land": land,
                    "published": published,
                    "empty": empty,
                    "roster": roster_tiles,
                    "delivered": delivered,
                    "skipped": skipped,
                    "with_data": with_data,
                    "mask_snapshot": mask_snapshot,
                    "store_snapshot": store_snapshot,
                },
                fh,
            )
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
