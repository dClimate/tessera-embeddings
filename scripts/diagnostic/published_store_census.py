"""Audit the published global store from a consumer's seat: does it open, and is it as planned?

Checks four things and exits non-zero on any disagreement:

1. **What a reader inherits.** The repository configuration is saved into the store, so a consumer
   who passes none of their own gets what the writer left. Printed rather than assumed.
2. **Layout.** Per-array dtype, inner chunks and shards against
   :data:`~tessera_embeddings.config.store_layout.GLOBAL`, plus the CRS and the spatial grid.
3. **Completion.** Each group's ``years_complete`` is the authority — all nine slots are
   preallocated at seed, so an unfilled year opens without error and reads back as fill. A year
   deliberately left empty IS in the list; a year that never landed is absent from it.
4. **The tag record.** Cells are tagged ``zone-<ZONE>-<YEAR>`` in a different operation from the
   attribute, so reconciling the two is the cheapest check that no cell was half-recorded.

``--shards`` adds per-zone-year shard coverage, one manifest enumeration per zone and the only part
that is not effectively instant. Run from the REPOSITORY ROOT::

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

from tessera_embeddings.config.inference import EMBEDDING_DIM
from tessera_embeddings.storage import published_store, zone_grid
from tessera_embeddings.storage.global_store import open_global_repo

#: Overridable so the script can be pointed at a staging copy, which is the only way to rehearse it.
DEFAULT_URI = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
DEFAULT_REGION = "us-west-2"

#: Campaign time axis, preallocated at seed (ADR 008 D1). Used only to report which years a zone is
#: missing; the store's own `time` coordinate is the authority for what the slots mean.
CAMPAIGN_YEARS = tuple(range(2017, 2026))

#: Checked as a SET and before any zone is audited: an audit that iterates whatever groups it finds
#: cannot report a missing one, and with `--zones all` the tag reconciliation is filtered to the
#: audited groups too — so a store that lost a whole zone would reconcile against its smaller self.
EXPECTED_ZONES = tuple(f"{n:02d}{hemisphere}" for n in range(1, 61) for hemisphere in ("N", "S"))

#: Root attributes a consumer needs to interpret the product at all. In the ``utm_zones`` layout the
#: encoder and quantization provenance is stated ONCE at the root, so a zone group cannot make up
#: for a root that lost it.
_REQUIRED_ROOT_VALUES = {
    "geoemb:type": "pixel",
    "geoemb:dimensions": EMBEDDING_DIM,
    "geoemb:data_type": "int8",
    "geoemb:spatial_layout": "utm_zones",
}
#: Present-and-non-empty is all that can be asked of these: the model URL and build version depend
#: on the run, so a value check would only pin whatever the last run happened to write.
_REQUIRED_ROOT_KEYS = ("geoemb:model", "geoemb:build_version", "geoemb:source_data", "geoemb:quantization")


def _root_departures(attrs: dict[str, Any]) -> list[str]:
    """Every way the root's provenance falls short of what a consumer needs to interpret the data.

    Printed AND counted: without the quantization block nobody can turn int8 back into
    reflectance-space values, so a store missing it is unreadable however complete its arrays are.
    """
    out: list[str] = []
    for key, expected in _REQUIRED_ROOT_VALUES.items():
        if attrs.get(key) != expected:
            out.append(f"root: {key}={attrs.get(key)!r}, expected {expected!r}")
    for key in _REQUIRED_ROOT_KEYS:
        if not attrs.get(key):
            out.append(f"root: {key} is missing or empty")
    quantization = attrs.get("geoemb:quantization")
    if isinstance(quantization, dict):
        # The dequantization recipe: which array holds the per-pixel factors, and what a missing
        # factor looks like there. Without both, the embeddings can be read but not used.
        scale = quantization.get("scale")
        if quantization.get("method") != "per_pixel_scale":
            out.append(f"root: quantization method is {quantization.get('method')!r}, expected 'per_pixel_scale'")
        if not isinstance(scale, dict) or scale.get("array_name") != "scales":
            out.append(f"root: quantization scale does not name the `scales` array: {scale!r}")
        elif str(scale.get("nodata")).lower() != "nan":
            out.append(f"root: quantization scale nodata is {scale.get('nodata')!r}, expected NaN")
    elif quantization is not None:
        out.append(f"root: geoemb:quantization is {type(quantization).__name__}, expected a mapping")
    declared = {c.get("name") for c in attrs.get("zarr_conventions", []) if isinstance(c, dict)}
    if "geoemb:" not in declared:
        out.append(f"root: zarr_conventions does not declare the geoemb: convention (declares {sorted(declared)})")
    return out


def _storage_for(args: argparse.Namespace) -> icechunk.Storage:
    """Storage for the audited URI, with no repository config attached.

    So the config the store SAVED is fetched independently of the handle the audit reads through: a
    config handed to ``Repository.open`` replaces the saved one, and anything read back off such a
    handle is only an echo of what was passed in. Takes a local path too, for rehearsals.
    """
    if not args.uri.startswith("s3://"):
        return icechunk.local_filesystem_storage(args.uri.removeprefix("file://"))
    bucket, _, prefix = args.uri.removeprefix("s3://").partition("/")
    return icechunk.s3_storage(
        bucket=bucket,
        prefix=prefix,
        region=args.region,
        anonymous=True if args.anonymous else None,
        from_env=None if args.anonymous else True,
    )


def _cell_tags(repo: icechunk.Repository) -> set[tuple[str, int]]:
    """Every ``(zone, year)`` the store carries a completion tag for.

    Names that do not parse as ``zone-<ZONE>-<YEAR>`` are ignored rather than guessed at: the store
    also carries ``year-<YEAR>-complete`` tags, and a future scheme may add more.
    """
    cells: set[tuple[str, int]] = set()
    for tag in repo.list_tags():
        parts = tag.split("-")
        if len(parts) == 3 and parts[0] == "zone" and parts[2].isdigit():
            cells.add((parts[1], int(parts[2])))
    return cells


def _describe_config(config: icechunk.RepositoryConfig | None) -> dict[str, Any]:
    """The parts of a repository config a reader's performance depends on."""
    if config is None:
        return {"saved_config": None}
    manifest = config.manifest
    preload = getattr(manifest, "preload", None)
    splitting = getattr(manifest, "splitting", None)
    return {
        "manifest_preload_max_total_refs": getattr(preload, "max_total_refs", None),
        "manifest_preload_max_arrays_to_scan": getattr(preload, "max_arrays_to_scan", None),
        "manifest_splitting_configured": splitting is not None,
        "caching": repr(config.caching),
        "storage_concurrency": repr(getattr(config.storage, "concurrency", None)),
    }


def _zone_report(root: zarr.Group, zone: str, *, with_shards: bool, session: icechunk.Session) -> dict[str, Any]:
    """Everything this audit records about one zone group."""
    opened = time.monotonic()
    group = root[zone]
    attrs = dict(group.attrs)
    years = [int(y) for y in attrs.get("years_complete", [])]
    departures = published_store.layout_departures(group)
    # The CRS is CHECKED, not just reported: another zone's EPSG code georeferences every array
    # wrongly while the shapes stay right and the attribute stays plausible. Expected value from
    # `zone_grid`, so this and the seeder cannot disagree.
    spec = zone_grid.zone(zone) if zone in zone_grid.ZONES else None
    if spec is None:
        departures.append(f"{zone}: not a known UTM zone, so neither its CRS nor its grid can be checked")
    else:
        if attrs.get("crs") != spec.crs:
            departures.append(f"{zone}: declares crs {attrs.get('crs')!r}, the zone grid says {spec.crs!r}")
        departures += [f"{zone}: {d}" for d in published_store.coordinate_departures(group, spec)]
    expected_crs = spec.crs if spec else None
    # A completion record pointing at no time slot. Unchecked it also makes the "never filled"
    # arithmetic wrong, and can turn it negative.
    unexpected_years = [y for y in years if y not in CAMPAIGN_YEARS]
    # Sorted and unique is how the writer persists it and what every count below assumes: a
    # duplicate survives into `from_attrs` as one element, so the set-based reconciliation cannot
    # see it and the reported cell counts quietly stop adding up.
    if years != sorted(set(years)):
        departures.append(f"{zone}: years_complete is {years}, not a sorted unique list")
    # The group's OWN time axis, decoded — not just the attribute against a constant. A shifted or
    # duplicated axis passes every attribute and tag check while a labelled reader asking for 2021
    # gets another year, because `years_complete` has no link to the coordinate it describes.
    calendar = published_store.calendar_years(group)
    if tuple(calendar) != CAMPAIGN_YEARS:
        departures.append(f"{zone}: time coordinate decodes to {calendar}, the campaign axis is {list(CAMPAIGN_YEARS)}")
    report: dict[str, Any] = {
        "zone": zone,
        "crs": attrs.get("crs"),
        "expected_crs": expected_crs,
        "years_complete": years,
        "years_missing": [y for y in CAMPAIGN_YEARS if y not in years],
        "years_unexpected": unexpected_years,
        "time_axis": calendar,
        "layout_departures": departures,
        "open_s": round(time.monotonic() - opened, 3),
        # A year in `runs` but not in `years_complete` means a fill that wrote data and never marked
        # itself — the one inconsistency the two-commit write model can leave.
        "years_with_provenance": sorted(int(y) for y in attrs.get("runs", {})),
    }
    if with_shards:
        started = time.monotonic()
        coverage = published_store.live_shards(session, zone)
        report["live_shards_by_time_index"] = {str(k): len(v) for k, v in coverage.items()}
        report["live_shards_total"] = sum(len(v) for v in coverage.values())
        report["shard_enumeration_s"] = round(time.monotonic() - started, 2)
        # THE half-published state the write model can produce: shards are committed before the
        # year's attributes, so a crash between the two leaves real data in a year nothing records
        # as complete — and a reader trusting `years_complete` will never look at it.
        report["live_shards_in_incomplete_years"] = {
            str(calendar[index]): len(shards)
            for index, shards in sorted(coverage.items())
            if index < len(calendar) and calendar[index] not in years and shards
        }
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
    saved_config = icechunk.Repository.fetch_config(_storage_for(args))

    started = time.monotonic()
    session = repo.readonly_session(branch="main")
    timings["readonly_session_s"] = round(time.monotonic() - started, 3)

    started = time.monotonic()
    root = zarr.open_group(session.store, mode="r")
    timings["root_group_open_s"] = round(time.monotonic() - started, 3)

    present = sorted(k for k, _ in root.groups())
    missing_groups = sorted(set(EXPECTED_ZONES) - set(present))
    unexpected_groups = sorted(set(present) - set(EXPECTED_ZONES))
    # A full audit walks the EXPECTED zones that are present: an auxiliary or corrupt extra group
    # has no `time` array, so `_zone_report` would raise while decoding the calendar and the census
    # would never print its reconciliation, write its JSON or return the non-zero status that group
    # had earned. Extras are reported as `unexpected_groups`.
    zones = [z for z in present if z in set(EXPECTED_ZONES)] if args.zones == "all" else args.zones.split(",")
    if absent := sorted(set(zones) - set(present)):
        parser.error(f"the store has no group(s) {absent}; it holds {len(present)}")
    print(f"store:     {args.uri} ({args.region})")
    print(f"snapshot:  {session.snapshot_id}")
    print(f"opens:     {timings}")
    print(f"config:    {json.dumps(_describe_config(saved_config), indent=13)[1:-1].strip()}")
    root_departures = _root_departures(dict(root.attrs))
    print(f"root keys: {sorted(root.attrs)}")
    for departure in root_departures:
        print(f"           {departure}")
    print(f"groups:    {len(present)} of {len(EXPECTED_ZONES)} expected (auditing {len(zones)})")
    if missing_groups:
        print(f"           MISSING GROUPS: {missing_groups}")
    if unexpected_groups:
        print(f"           UNEXPECTED GROUPS: {unexpected_groups}")

    tags = _cell_tags(repo)
    zone_reports = [_zone_report(root, z, with_shards=args.shards, session=session) for z in zones]

    print(f"\n{'zone':<6} {'crs':<12} {'complete':>8} {'missing years':<30} departures")
    for report in zone_reports:
        missing = ",".join(str(y) for y in report["years_missing"]) or "-"
        departures = "; ".join(report["layout_departures"]) or "none"
        complete = len(report["years_complete"])
        print(f"{report['zone']:<6} {report['crs'] or '?':<12} {complete:>8} {missing:<30} {departures}")

    audited = {r["zone"] for r in zone_reports}
    from_attrs = {(r["zone"], y) for r in zone_reports for y in r["years_complete"]}
    # A FULL audit compares every parsed tag; a subset audit only the zones asked for. Filtering
    # unconditionally is how an orphan tag — `zone-61N-2025`, or one left by a deleted group —
    # disappears from the comparison and lets the census exit 0 on it.
    from_tags = tags if args.zones == "all" else {(z, y) for z, y in tags if z in audited}
    only_attrs = sorted(from_attrs - from_tags)
    only_tags = sorted(from_tags - from_attrs)
    unmarked_provenance = sorted(
        (r["zone"], y) for r in zone_reports for y in r["years_with_provenance"] if y not in r["years_complete"]
    )
    # And the other direction: a year marked complete with no `runs` entry is a published cell whose
    # run id and input coverage nobody can look up, and a one-sided check certifies exactly the half
    # that is missing.
    provenance_missing = sorted(
        (r["zone"], y) for r in zone_reports for y in r["years_complete"] if y not in r["years_with_provenance"]
    )
    departures = {r["zone"]: r["layout_departures"] for r in zone_reports if r["layout_departures"]}
    unexpected_years = sorted((r["zone"], y) for r in zone_reports for y in r["years_unexpected"])
    unmarked_shards = {
        f"{r['zone']}/{year}": count
        for r in zone_reports
        for year, count in r.get("live_shards_in_incomplete_years", {}).items()
    }

    print(f"\ncells complete (attrs):   {len(from_attrs)}")
    print(f"cells tagged:             {len(from_tags)}")
    print(f"cells never filled:       {len(audited) * len(CAMPAIGN_YEARS) - len(from_attrs)}")
    if args.shards:
        print(f"live shards total:        {sum(r['live_shards_total'] for r in zone_reports):,}")
    print(f"zones departing layout:   {len(departures)}")
    print(f"marked complete, untagged:{len(only_attrs)} {only_attrs[:6]}")
    print(f"tagged, not marked:       {len(only_tags)} {only_tags[:6]}")
    print(f"has provenance, unmarked: {len(unmarked_provenance)} {unmarked_provenance[:6]}")
    print(f"complete, no provenance:  {len(provenance_missing)} {provenance_missing[:6]}")
    print(f"groups missing/unexpected:{len(missing_groups)} / {len(unexpected_groups)}")
    print(f"years outside the campaign:{len(unexpected_years)} {unexpected_years[:6]}")
    print(f"root provenance departures: {len(root_departures)}")
    if args.shards:
        print(f"shards in unmarked years:  {len(unmarked_shards)} {list(unmarked_shards.items())[:4]}")

    if args.json_out:
        with Path(args.json_out).open("w") as handle:
            json.dump(
                {
                    "uri": args.uri,
                    "region": args.region,
                    "snapshot_id": session.snapshot_id,
                    "timings": timings,
                    "config": _describe_config(saved_config),
                    "root_attrs": {k: root.attrs[k] for k in sorted(root.attrs)},
                    "zones": zone_reports,
                    "reconciliation": {
                        "marked_complete_untagged": only_attrs,
                        "tagged_not_marked": only_tags,
                        "has_provenance_unmarked": unmarked_provenance,
                        "complete_without_provenance": provenance_missing,
                        "layout_departures": departures,
                        "missing_groups": missing_groups,
                        "unexpected_groups": unexpected_groups,
                        "years_outside_the_campaign": unexpected_years,
                        "root_departures": root_departures,
                        "live_shards_in_unmarked_years": unmarked_shards,
                    },
                },
                handle,
                indent=2,
                default=str,
            )
        print(f"\nwrote {args.json_out}")

    problems = (only_attrs, only_tags, unmarked_provenance, provenance_missing, departures)
    groups_and_years = (missing_groups, unexpected_groups, unexpected_years, unmarked_shards, root_departures)
    return 1 if any(problems) or any(groups_and_years) else 0


if __name__ == "__main__":
    sys.exit(main())
