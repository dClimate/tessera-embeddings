"""Audit the published global store from a consumer's seat: does it open, and is it as planned?

Answers four questions a consumer or an operator asks about a finished campaign, in the order
they matter:

1. **Does it open, and what does a reader inherit?** The tuned repository configuration is saved
   into the store, so a consumer who opens it with no configuration of their own still gets the
   manifest splitting and preload settings the writer chose. This prints what came back, because
   "the reader inherits the tuning" is a claim that should be checked rather than assumed.
2. **Is every zone group shaped the way the architecture promised?** Per-array dtype, inner chunks
   and shards, compared against the declared global layout
   (:data:`~tessera_embeddings.config.store_layout.GLOBAL`) by
   :func:`~tessera_embeddings.storage.published_store.layout_departures`.
3. **Which zone-years are complete?** Each group's ``years_complete`` is the authority — slots for
   all nine years are preallocated at seed, so an unfilled year opens without error and reads back
   as fill. A year deliberately left empty because the zone has no qualifying land IS in the list;
   a year that never landed is absent from it. The two are indistinguishable from the data.
4. **Does the tag record agree?** Every completed cell is tagged ``zone-<ZONE>-<YEAR>``, written in
   a different operation from the attribute. Reconciling the two is the cheapest check that no cell
   was half-recorded, and the script exits non-zero if they disagree.

``--shards`` adds per-zone-year shard coverage, which costs one manifest enumeration per zone and
is the only part of this script that is not effectively instant.

Run from the REPOSITORY ROOT::

    uv run python scripts/diagnostic/published_store_census.py --zones 16S,33N
    uv run python scripts/diagnostic/published_store_census.py --shards --json census.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import icechunk
import zarr

from tessera_embeddings.storage import published_store
from tessera_embeddings.storage.global_store import open_global_repo

#: The published store and the region its bucket lives in. Both overridable so the script can be
#: pointed at a staging copy, which is the only way to rehearse it without reading the real thing.
DEFAULT_URI = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
DEFAULT_REGION = "us-west-2"

#: Campaign time axis, preallocated at seed (ADR 008 D1). Used only to report which years a zone is
#: missing; the store's own `time` coordinate is the authority for what the slots mean.
CAMPAIGN_YEARS = tuple(range(2017, 2026))


def _cell_tags(repo: icechunk.Repository) -> set[tuple[str, int]]:
    """Every ``(zone, year)`` the store carries a completion tag for.

    Tag names that do not parse as ``zone-<ZONE>-<YEAR>`` are ignored rather than guessed at: the
    store also carries ``year-<YEAR>-complete`` tags, and a future scheme may add more.
    """
    cells: set[tuple[str, int]] = set()
    for tag in repo.list_tags():
        parts = tag.split("-")
        if len(parts) == 3 and parts[0] == "zone" and parts[2].isdigit():
            cells.add((parts[1], int(parts[2])))
    return cells


def _describe_config(repo: icechunk.Repository) -> dict[str, Any]:
    """The parts of the persisted repository config a reader's performance depends on."""
    manifest = repo.config.manifest
    preload = getattr(manifest, "preload", None)
    splitting = getattr(manifest, "splitting", None)
    return {
        "manifest_preload_max_total_refs": getattr(preload, "max_total_refs", None),
        "manifest_preload_max_arrays_to_scan": getattr(preload, "max_arrays_to_scan", None),
        "manifest_splitting_configured": splitting is not None,
        "caching": repr(repo.config.caching),
        "storage_concurrency": repr(getattr(repo.config.storage, "concurrency", None)),
    }


def _zone_report(root: zarr.Group, zone: str, *, with_shards: bool, session: icechunk.Session) -> dict[str, Any]:
    """Everything this audit records about one zone group."""
    opened = time.monotonic()
    group = root[zone]
    attrs = dict(group.attrs)
    years = [int(y) for y in attrs.get("years_complete", [])]
    report: dict[str, Any] = {
        "zone": zone,
        "crs": attrs.get("crs"),
        "years_complete": years,
        "years_missing": [y for y in CAMPAIGN_YEARS if y not in years],
        "layout_departures": published_store.layout_departures(group),
        "open_s": round(time.monotonic() - opened, 3),
        # A year in `runs` but not in `years_complete` would mean a fill that wrote data and never
        # marked itself, which is the one inconsistency the two-commit write model could leave.
        "years_with_provenance": sorted(int(y) for y in attrs.get("runs", {})),
    }
    if with_shards:
        started = time.monotonic()
        coverage = published_store.live_shards(session, zone)
        report["live_shards_by_time_index"] = {str(k): len(v) for k, v in coverage.items()}
        report["live_shards_total"] = sum(len(v) for v in coverage.values())
        report["shard_enumeration_s"] = round(time.monotonic() - started, 2)
    return report


def main(argv: list[str] | None = None) -> int:
    """Audit the store; return 0 when everything reconciles and 1 when anything does not."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=DEFAULT_URI, help="Icechunk store URI")
    parser.add_argument("--region", default=DEFAULT_REGION, help="bucket region")
    parser.add_argument("--zones", default="all", help="comma-separated zone list, or 'all'")
    parser.add_argument("--shards", action="store_true", help="also enumerate live shards per zone-year")
    parser.add_argument(
        "--anonymous",
        action="store_true",
        help="read with no credentials (the published bucket grants public reads)",
    )
    parser.add_argument("--json", dest="json_out", help="write the full report to this path")
    args = parser.parse_args(argv)

    timings: dict[str, float] = {}
    started = time.monotonic()
    repo = open_global_repo(args.uri, region=args.region, anonymous=args.anonymous)
    timings["repository_open_s"] = round(time.monotonic() - started, 3)

    started = time.monotonic()
    session = repo.readonly_session(branch="main")
    timings["readonly_session_s"] = round(time.monotonic() - started, 3)

    started = time.monotonic()
    root = zarr.open_group(session.store, mode="r")
    timings["root_group_open_s"] = round(time.monotonic() - started, 3)

    zones = sorted(k for k, _ in root.groups()) if args.zones == "all" else args.zones.split(",")
    print(f"store:     {args.uri} ({args.region})")
    print(f"snapshot:  {session.snapshot_id}")
    print(f"opens:     {timings}")
    print(f"config:    {json.dumps(_describe_config(repo), indent=13)[1:-1].strip()}")
    print(f"root keys: {sorted(root.attrs)}")
    print(f"groups:    {len(list(root.groups()))} (auditing {len(zones)})")

    tags = _cell_tags(repo)
    zone_reports = [_zone_report(root, z, with_shards=args.shards, session=session) for z in zones]

    print(f"\n{'zone':<6} {'crs':<12} {'complete':>8} {'missing years':<30} departures")
    for report in zone_reports:
        missing = ",".join(str(y) for y in report["years_missing"]) or "-"
        departures = "; ".join(report["layout_departures"]) or "none"
        complete = len(report["years_complete"])
        print(f"{report['zone']:<6} {report['crs'] or '?':<12} {complete:>8} {missing:<30} {departures}")

    # Reconciliation. Only meaningful over the zones actually audited, so a `--zones` run compares
    # the tag record for those zones and nothing else.
    audited = {r["zone"] for r in zone_reports}
    from_attrs = {(r["zone"], y) for r in zone_reports for y in r["years_complete"]}
    from_tags = {(z, y) for z, y in tags if z in audited}
    only_attrs = sorted(from_attrs - from_tags)
    only_tags = sorted(from_tags - from_attrs)
    unmarked_provenance = sorted(
        (r["zone"], y) for r in zone_reports for y in r["years_with_provenance"] if y not in r["years_complete"]
    )
    departures = {r["zone"]: r["layout_departures"] for r in zone_reports if r["layout_departures"]}

    print(f"\ncells complete (attrs):   {len(from_attrs)}")
    print(f"cells tagged:             {len(from_tags)}")
    print(f"cells never filled:       {len(audited) * len(CAMPAIGN_YEARS) - len(from_attrs)}")
    if args.shards:
        print(f"live shards total:        {sum(r['live_shards_total'] for r in zone_reports):,}")
    print(f"zones departing layout:   {len(departures)}")
    print(f"marked complete, untagged:{len(only_attrs)} {only_attrs[:6]}")
    print(f"tagged, not marked:       {len(only_tags)} {only_tags[:6]}")
    print(f"has provenance, unmarked: {len(unmarked_provenance)} {unmarked_provenance[:6]}")

    if args.json_out:
        with Path(args.json_out).open("w") as handle:
            json.dump(
                {
                    "uri": args.uri,
                    "region": args.region,
                    "snapshot_id": session.snapshot_id,
                    "timings": timings,
                    "config": _describe_config(repo),
                    "root_attrs": {k: root.attrs[k] for k in sorted(root.attrs)},
                    "zones": zone_reports,
                    "reconciliation": {
                        "marked_complete_untagged": only_attrs,
                        "tagged_not_marked": only_tags,
                        "has_provenance_unmarked": unmarked_provenance,
                        "layout_departures": departures,
                    },
                },
                handle,
                indent=2,
                default=str,
            )
        print(f"\nwrote {args.json_out}")

    return 1 if (only_attrs or only_tags or unmarked_provenance or departures) else 0


if __name__ == "__main__":
    sys.exit(main())
