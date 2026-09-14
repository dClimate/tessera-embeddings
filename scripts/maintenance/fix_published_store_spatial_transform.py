"""Repair the published store's ``spatial:transform`` origin, and its dead convention URLs.

The store shipped with each zone group's ``spatial:transform`` origin at the CENTRE of the first
pixel. The ``spatial:`` convention puts it at the outer CORNER — ``c`` is the western-most
coordinate of the X axis, ``f`` the northern-most of the Y axis, with array index ``(0, 0)`` at the
top-left corner of the top-left pixel — so every consumer reading the transform placed the imagery
half a pixel (5 m) to the south-east. The same groups' ``spatial:bbox`` was already correct and
edge-based, so the store also contradicted itself. This walks the zone groups and moves the origin
half a pixel back along each axis, leaving every other attribute untouched.

It also re-pins the ``proj:`` and ``spatial:`` entries in ``zarr_conventions``. Both pointed at
``refs/tags/v1``, a tag neither convention has cut, so all four URLs a consumer might follow
returned 404. ``v0.1`` is the tag both repositories carry.

**Nothing is recomputed from a table.** Each group's corrected origin is derived from that group's
OWN coordinate arrays, then cross-checked against that group's OWN ``spatial:bbox`` — an attribute
written by different code at seeding time — and the write is refused unless the two agree
independently. Only ``spatial:transform`` and the two registration entries may change; the script
diffs the whole attribute dict before and after and refuses on any other difference.

Safe to re-run: a group already holding the corrected origin and the pinned URLs is skipped, and a
run with nothing to do commits nothing. Dry run unless ``--apply``.

    uv run python scripts/maintenance/fix_published_store_spatial_transform.py            # dry run
    uv run python scripts/maintenance/fix_published_store_spatial_transform.py --zone 33N # rehearse
    AWS_PROFILE=... PUBLISHED_STORE_WRITER_ROLE_ARN=... uv run python \
        scripts/maintenance/fix_published_store_spatial_transform.py --apply

Rollback is Icechunk's: the pre-repair snapshot stays on ``main``'s history and the run prints its
id, so ``repo.reset_branch("main", <snapshot>)`` puts the old attrs back. ``--record`` also writes
the before/after values to a local JSON file, so the change is auditable without the store.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import zarr

from tessera_embeddings.providers.aws.credentials import icechunk_credentials_for, published_store_writer_role
from tessera_embeddings.storage.conventions import _PROJ_CONVENTION, _SPATIAL_CONVENTION
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.zarr_store import _create_storage

DEFAULT_URI = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
DEFAULT_REGION = "us-west-2"

#: What the published store holds, asserted rather than discovered. A store that has drifted from
#: this shape is one nobody gathered the safety evidence against, so it is refused, not adapted to.
EXPECTED_GROUPS = 120
EXPECTED_TAGS = 1070

#: The registration entries as the fixed source now emits them, keyed by convention name. Imported
#: from `conventions` rather than restated, so the store and the code that writes new stores cannot
#: drift apart: correcting one here without the other is exactly the failure being repaired.
WANTED_CONVENTIONS: dict[str, dict] = {
    _PROJ_CONVENTION["name"]: _PROJ_CONVENTION,
    _SPATIAL_CONVENTION["name"]: _SPATIAL_CONVENTION,
}

#: How far two coordinates may differ and still count as the same place, in CRS units (metres
#: here). Far below the 5 m half-pixel being corrected and far above float64 noise on a 10 M metre
#: northing, so it separates "the same edge computed two ways" from "a different edge".
TOL_M = 1e-6

#: Attrs this script is permitted to change. Anything else differing between the before and after
#: dict aborts the run with nothing written.
WRITABLE_KEYS = frozenset({"spatial:transform", "zarr_conventions"})


class RefusedError(Exception):
    """A precondition failed. Raised, not returned, so no caller can proceed past one."""


def _corner_origin(centres: np.ndarray) -> tuple[float, float]:
    """``(resolution, leading edge)`` for one axis of pixel-CENTRE coordinates.

    The edge is half a pixel back from the first centre along the axis's own signed resolution, so
    a descending (north-up) axis moves UP to its northern edge and an ascending one moves DOWN.
    Median spacing, matching how the source computes it, so one ragged step cannot move the answer.
    """
    res = float(np.median(np.diff(centres)))
    return res, float(centres[0]) - res / 2


def _inspect(group: zarr.Group, name: str) -> dict[str, Any]:
    """What one zone group holds now, what it should hold, and whether it may be written.

    Every check that could make a write unsafe lives here and raises :class:`RefusedError`. The caller
    only ever sees a group that is already correct, or one that is in exactly the known-bad state
    and whose corrected value has been confirmed against evidence written by other code.
    """
    attrs = dict(group.attrs)

    registration = attrs.get("spatial:registration")
    if registration != "pixel":
        raise RefusedError(
            f"{name}: spatial:registration is {registration!r}, not 'pixel' — the half-pixel rule differs"
        )
    if attrs.get("spatial:transform_type", "affine") != "affine":
        raise RefusedError(f"{name}: spatial:transform_type is {attrs.get('spatial:transform_type')!r}, not 'affine'")

    transform = list(attrs.get("spatial:transform") or [])
    if len(transform) != 6:
        raise RefusedError(f"{name}: spatial:transform has {len(transform)} elements, expected 6")
    a, b, c, d, e, f = (float(v) for v in transform)
    if b or d:
        raise RefusedError(f"{name}: transform is rotated (b={b}, d={d}) — the half-pixel step assumes axis-aligned")

    north = np.asarray(group["northing"][:], dtype="float64")
    east = np.asarray(group["easting"][:], dtype="float64")
    if north.size < 2 or east.size < 2:
        raise RefusedError(f"{name}: needs at least two coordinates per axis to derive a resolution")

    res_x, edge_x = _corner_origin(east)
    res_y, edge_y = _corner_origin(north)
    if abs(res_x - a) > TOL_M or abs(res_y - e) > TOL_M:
        raise RefusedError(f"{name}: transform scale ({a}, {e}) disagrees with the coordinates ({res_x}, {res_y})")

    shape = [int(v) for v in attrs.get("spatial:shape") or []]
    if shape != [int(north.size), int(east.size)]:
        raise RefusedError(f"{name}: spatial:shape {shape} disagrees with the arrays {[north.size, east.size]}")

    # The independent cross-check. `spatial:bbox` was written from the same coordinates by a
    # different function that got the half-pixel right, so agreeing with it means two separately
    # derived answers land on the same edge. Without this the script would only be asserting its
    # own arithmetic back to itself.
    bbox = [float(v) for v in attrs.get("spatial:bbox") or []]
    if len(bbox) != 4:
        raise RefusedError(f"{name}: spatial:bbox has {len(bbox)} elements, expected 4")
    want_x = bbox[0] if res_x > 0 else bbox[2]
    want_y = bbox[3] if res_y < 0 else bbox[1]
    if abs(edge_x - want_x) > TOL_M or abs(edge_y - want_y) > TOL_M:
        raise RefusedError(
            f"{name}: corrected origin ({edge_x}, {edge_y}) does not match the stored bbox edges "
            f"({want_x}, {want_y}) — the bbox is not the corroboration this assumed"
        )

    wanted = [a, b, edge_x, d, e, edge_y]
    already_corner = abs(c - edge_x) <= TOL_M and abs(f - edge_y) <= TOL_M
    if not already_corner:
        # The only other state this script knows how to reason about. A third value is neither the
        # bug being fixed nor the fix, so it is somebody else's change and must not be overwritten.
        centre_x, centre_y = float(east[0]), float(north[0])
        if abs(c - centre_x) > TOL_M or abs(f - centre_y) > TOL_M:
            raise RefusedError(
                f"{name}: origin ({c}, {f}) is neither the pixel centre ({centre_x}, {centre_y}) "
                f"nor the corner ({edge_x}, {edge_y}) — refusing to overwrite an unknown value"
            )

    registered = {entry.get("name"): entry for entry in attrs.get("zarr_conventions", []) if isinstance(entry, dict)}
    urls_stale = any(registered.get(n) != want for n, want in WANTED_CONVENTIONS.items() if n in registered)

    return {
        "zone": name,
        "before": transform,
        "after": wanted,
        "shift": [wanted[2] - c, wanted[5] - f],
        "transform_stale": not already_corner,
        "urls_stale": urls_stale,
        "conventions_before": attrs.get("zarr_conventions", []),
    }


def _repaired_conventions(entries: list) -> list:
    """*entries* with the ``proj:``/``spatial:`` registrations re-pinned, order preserved.

    Rebuilt positionally rather than filtered and re-appended: the list is what a consumer reads to
    find the spec, and reordering it would be a second, gratuitous change to diff against.
    """
    out = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else None
        out.append(dict(WANTED_CONVENTIONS[name]) if name in WANTED_CONVENTIONS else entry)
    return out


def _apply_to_group(group: zarr.Group, plan: dict[str, Any]) -> None:
    """Write one group's corrected attrs, refusing if anything outside :data:`WRITABLE_KEYS` moved.

    The guard is the point: `zarr`'s attrs are replaced wholesale, so a mistake here silently drops
    provenance, run records and the depth rule. Diffing before against after and refusing on any
    unexpected key means the blast radius is checked rather than assumed.
    """
    before = dict(group.attrs)
    after = dict(before)
    if plan["transform_stale"]:
        after["spatial:transform"] = plan["after"]
    if plan["urls_stale"]:
        after["zarr_conventions"] = _repaired_conventions(before.get("zarr_conventions", []))

    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if not changed <= WRITABLE_KEYS:
        raise RefusedError(f"{plan['zone']}: would change {sorted(changed - WRITABLE_KEYS)}, which this must not touch")
    if not changed:
        return
    group.attrs.update(after)


def _verify(group: zarr.Group, name: str) -> None:
    """Re-read one group and assert the invariant a consumer depends on, or raise.

    Asserted against the bbox already in the store rather than against what was just written, so
    this is a check and not an echo.
    """
    attrs = dict(group.attrs)
    transform = [float(v) for v in attrs["spatial:transform"]]
    bbox = [float(v) for v in attrs["spatial:bbox"]]
    want_x = bbox[0] if transform[0] > 0 else bbox[2]
    want_y = bbox[3] if transform[4] < 0 else bbox[1]
    if abs(transform[2] - want_x) > TOL_M or abs(transform[5] - want_y) > TOL_M:
        raise RefusedError(f"{name}: after the write the origin still disagrees with the bbox")
    registered = {e.get("name"): e for e in attrs.get("zarr_conventions", []) if isinstance(e, dict)}
    for conv_name, want in WANTED_CONVENTIONS.items():
        if conv_name in registered and registered[conv_name] != want:
            raise RefusedError(f"{name}: {conv_name} registration was not re-pinned")


def _open_for_read(uri: str, region: str) -> icechunk.Repository:
    """Open with no config, anonymously where the store allows it — as the sibling script does."""
    try:
        return icechunk.Repository.open(_create_storage(uri, anonymous=True, region=region))
    except icechunk.IcechunkError:
        return icechunk.Repository.open(_create_storage(uri, region=region))


def main(argv: list[str] | None = None) -> int:
    """Report, or apply, the repair. 0 on success or a clean no-op."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--apply", action="store_true", help="write; without it this is a dry run")
    parser.add_argument("--zone", action="append", help="limit to these zone groups (repeatable); default all")
    parser.add_argument("--record", help="write the before/after values to this JSON file")
    parser.add_argument(
        "--expect-tags", type=int, default=EXPECTED_TAGS, help="refuse on a different count; 0 disables"
    )
    parser.add_argument(
        "--expect-groups", type=int, default=EXPECTED_GROUPS, help="refuse on a different count; 0 disables"
    )
    args = parser.parse_args(argv)

    repo = _open_for_read(args.uri, args.region)
    tags, branches = sorted(repo.list_tags()), sorted(repo.list_branches())
    before_snapshot = repo.lookup_branch("main")
    root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    groups = sorted(name for name, _ in root.groups())

    print(f"store:  {args.uri} ({args.region})")
    print(f"state:  branches {branches}, {len(tags)} tags, {len(groups)} groups, main at {before_snapshot}")

    if branches != ["main"]:
        print(f"REFUSING: expected one branch named main, found {branches}")
        return 1
    if args.expect_tags and len(tags) != args.expect_tags:
        print(f"REFUSING: expected {args.expect_tags} tags, found {len(tags)}")
        return 1
    if args.expect_groups and len(groups) != args.expect_groups:
        print(f"REFUSING: expected {args.expect_groups} groups, found {len(groups)}")
        return 1

    targets = groups
    if args.zone:
        unknown = sorted(set(args.zone) - set(groups))
        if unknown:
            print(f"REFUSING: no such group(s): {unknown}")
            return 1
        targets = sorted(set(args.zone))

    # Every group is inspected BEFORE anything is written, so a refusal on the last group leaves
    # the store untouched rather than half-repaired.
    try:
        plans = [_inspect(root[name], name) for name in targets]
    except RefusedError as refusal:
        print(f"REFUSING: {refusal}")
        return 1

    todo = [p for p in plans if p["transform_stale"] or p["urls_stale"]]
    shifts = {(round(p["shift"][0], 9), round(p["shift"][1], 9)) for p in plans if p["transform_stale"]}
    print(f"\ninspected {len(plans)} group(s): {len(todo)} need a change, {len(plans) - len(todo)} already correct")
    if todo:
        print(f"  transform origin stale: {sum(p['transform_stale'] for p in plans)}")
        print(f"  registration URLs stale: {sum(p['urls_stale'] for p in plans)}")
        print(f"  distinct origin shifts (x, y): {sorted(shifts)}")
        example = todo[0]
        print(f"  e.g. {example['zone']}: {example['before']}")
        print(f"       {' ' * len(example['zone'])}  -> {example['after']}")

    if args.record:
        with Path(args.record).open("w") as handle:
            json.dump({"uri": args.uri, "snapshot": str(before_snapshot), "groups": plans}, handle, indent=1)
        print(f"  wrote {args.record}")

    if not todo:
        print("\nnothing to do")
        return 0
    if not args.apply:
        print("\ndry run — pass --apply to write")
        return 0
    if published_store_writer_role() is None and args.uri == DEFAULT_URI:
        print("\nREFUSING: writing the published store needs PUBLISHED_STORE_WRITER_ROLE_ARN; our own")
        print("identity is read-only there by design.")
        return 1

    writable = open_global_repo(args.uri, get_credentials=icechunk_credentials_for(args.uri), region=args.region)
    session = writable.writable_session("main")
    node = zarr.open_group(session.store, mode="a")
    try:
        for plan in todo:
            _apply_to_group(node[plan["zone"]], plan)
    except RefusedError as refusal:
        print(f"REFUSING mid-write, nothing committed: {refusal}")
        return 1
    snapshot = session.commit(
        f"Move spatial:transform origin to the pixel corner and re-pin proj:/spatial: "
        f"registration URLs ({len(todo)} group(s))"
    )
    print(f"\ncommitted {snapshot}")

    after = _open_for_read(args.uri, args.region)
    after_root = zarr.open_group(after.readonly_session(branch="main").store, mode="r")
    try:
        for name in targets:
            _verify(after_root[name], name)
    except RefusedError as refusal:
        print(f"VERIFICATION FAILED: {refusal}")
        print(f"roll back with repo.reset_branch('main', '{before_snapshot}')")
        return 1
    surviving, branches_after = sorted(after.list_tags()), sorted(after.list_branches())
    if len(surviving) != len(tags) or branches_after != branches:
        print(f"VERIFICATION FAILED: refs changed — {len(surviving)} tags, branches {branches_after}")
        return 1
    print(f"verified {len(targets)} group(s); {len(surviving)} tags and branches {branches_after} intact")
    print(f"roll back if needed with repo.reset_branch('main', '{before_snapshot}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
