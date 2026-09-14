"""Repair the published store's ``spatial:transform`` origin, and its dead convention URLs.

The store shipped with each zone group's ``spatial:transform`` origin at the CENTRE of the first
pixel. The ``spatial:`` convention puts it at the outer CORNER — ``c`` is the western-most
coordinate of the X axis, ``f`` the northern-most of the Y axis, with array index ``(0, 0)`` at the
top-left corner of the top-left pixel — so every consumer reading the transform placed the imagery
half a pixel (5 m) to the south-east. The same groups' ``spatial:bbox`` was already correct and
edge-based, so the store also contradicted itself. This walks the zone groups and moves the origin
half a pixel back along each axis, leaving every other attribute untouched.

It also replaces the ``proj``/``spatial`` entries in ``zarr_conventions`` with the exact
registration objects those conventions' ``v0.1`` schemas require. The published ones pointed at
``refs/tags/v1``, a tag neither convention has cut, so all four URLs a consumer might follow
returned 404; they also carried ``proj:``/``spatial:`` as the ``name`` where both schemas pin the
bare word, and one named a repository that has since moved. Entries are matched by ``uuid``,
because the repair changes the name.

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

#: The registration entries as the fixed source now emits them, keyed by **UUID**. Imported from
#: `conventions` rather than restated, so the store and the code that writes new stores cannot drift
#: apart: correcting one here without the other is exactly the failure being repaired.
#:
#: Keyed by uuid and not by `name` because the repair CHANGES the name — the `v0.1` schemas require
#: the bare `"proj"`/`"spatial"` where the store holds `"proj:"`/`"spatial:"` — so a name key would
#: fail to match the very entries being migrated. The uuid is the one field all three conventions
#: describe as permanently identifying them.
WANTED_CONVENTIONS: dict[str, dict] = {
    _PROJ_CONVENTION["uuid"]: _PROJ_CONVENTION,
    _SPATIAL_CONVENTION["uuid"]: _SPATIAL_CONVENTION,
}

#: The exact registration objects the published store SHIPPED with, which are the only stale state
#: this script knows how to replace. A third shape — an extra field, a different uuid, a later
#: version — is somebody else's change, and overwriting it wholesale would silently delete metadata
#: this script never examined. Recognise, then replace; never "differs, therefore mine now".
SHIPPED_CONVENTIONS: dict[str, dict] = {
    entry["uuid"]: {
        "schema_url": f"https://raw.githubusercontent.com/{repo}/refs/tags/v1/schema.json",
        "spec_url": f"https://github.com/{repo}/blob/v1/README.md",
        "uuid": entry["uuid"],
        "name": f"{entry['name']}:",
        "description": entry["description"],
    }
    for entry, repo in (
        (_PROJ_CONVENTION, "zarr-experimental/geo-proj"),
        (_SPATIAL_CONVENTION, "zarr-conventions/spatial"),
    )
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


def _require_finite(name: str, **values: np.ndarray | list[float]) -> None:
    """Refuse *name* if any of *values* holds a NaN or an infinity.

    Separate and called early because every other guard here is a tolerance comparison, and those
    are silently satisfied by NaN: ``abs(nan - x) > TOL`` is False, so a non-finite value passes
    each check by failing to be comparable at all.
    """
    for label, value in values.items():
        if not np.all(np.isfinite(np.asarray(value, dtype="float64"))):
            raise RefusedError(f"{name}: {label} holds a non-finite value, which no tolerance check can judge")


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

    # Before ANY tolerance comparison. `abs(x - y) > TOL` is False when either side is NaN, so a
    # single non-finite coordinate would sail through every guard below, be written as a NaN origin,
    # and pass the post-write verification for the same reason — the one failure mode where these
    # checks report success precisely because they cannot see.
    _require_finite(name, transform=transform, northing=north, easting=east)

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
    _require_finite(name, bbox=bbox)
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

    urls_stale = _registration_state(attrs.get("zarr_conventions", []), name)

    return {
        "zone": name,
        "before": transform,
        "after": wanted,
        "shift": [wanted[2] - c, wanted[5] - f],
        "transform_stale": not already_corner,
        "urls_stale": urls_stale,
        "conventions_before": attrs.get("zarr_conventions", []),
    }


def _registration_state(entries: list, name: str) -> bool:
    """Whether the ``proj``/``spatial`` registrations need replacing. Refuses any third state.

    Three demands, and a missing registration fails the first of them:

    * **Both must be present.** Absent, the group promises a consumer no way to look up what its
      attributes mean, which is an unrecognised store state rather than "already correct" — and
      treating it as correct is how a group would be reported clean and silently skipped.
    * **Each must equal EITHER the shipped object or the wanted one.** Recognise, then replace.
      Anything else — an added field, a changed uuid, a later version somebody pinned deliberately —
      is a change this script never examined, and replacing it wholesale would delete it silently.
    * **They must agree with each other**, so a half-migrated group is repaired rather than reported
      clean on the strength of whichever entry happens to be current.
    """
    found = {entry.get("uuid"): entry for entry in entries if isinstance(entry, dict)}
    states = set()
    for uuid, wanted in WANTED_CONVENTIONS.items():
        entry = found.get(uuid)
        if entry is None:
            raise RefusedError(f"{name}: zarr_conventions has no entry for {wanted['name']} ({uuid})")
        if entry == wanted:
            states.add(False)
        elif entry == SHIPPED_CONVENTIONS[uuid]:
            states.add(True)
        else:
            raise RefusedError(
                f"{name}: the {wanted['name']} registration is neither the one published nor the one "
                f"wanted — refusing to overwrite metadata this has not examined: {entry}"
            )
    if len(states) != 1:
        raise RefusedError(f"{name}: one registration is repaired and the other is not — refusing a half-migration")
    return states.pop()


def _repaired_conventions(entries: list) -> list:
    """*entries* with the ``proj``/``spatial`` registrations replaced, order preserved.

    Rebuilt positionally rather than filtered and re-appended: the list is what a consumer reads to
    find the spec, and reordering it would be a second, gratuitous change to diff against. Safe to
    replace whole entries because :func:`_registration_state` has already established that each is
    byte-for-byte the object we published, so nothing unexamined can be lost.
    """
    out = []
    for entry in entries:
        uuid = entry.get("uuid") if isinstance(entry, dict) else None
        out.append(dict(WANTED_CONVENTIONS[uuid]) if uuid in WANTED_CONVENTIONS else entry)
    return out


def _apply_to_group(group: zarr.Group, plan: dict[str, Any]) -> None:
    """Write one group's corrected attrs, refusing if anything outside :data:`WRITABLE_KEYS` moved.

    The guard is the point: `zarr`'s attrs are replaced wholesale, so a mistake here silently drops
    provenance, run records and the depth rule. Diffing before against after and refusing on any
    unexpected key means the blast radius is checked rather than assumed.
    """
    before = dict(group.attrs)
    # The plan was made against a read-only snapshot; this runs against the writable session. The
    # branch tip is compared once in `main`, and this re-checks the one value being overwritten, so
    # a group that moved underneath the plan is refused rather than written from stale evidence.
    if [float(v) for v in before.get("spatial:transform") or []] != [float(v) for v in plan["before"]]:
        raise RefusedError(
            f"{plan['zone']}: its transform changed between inspection and the write "
            f"({plan['before']} -> {before.get('spatial:transform')}) — re-run the inspection"
        )
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
    # Ahead of the comparison, for the same reason `_inspect` checks it: a NaN origin would satisfy
    # every tolerance test below by being incomparable, so the verification would confirm a wreck.
    _require_finite(name, transform=transform, bbox=bbox)
    want_x = bbox[0] if transform[0] > 0 else bbox[2]
    want_y = bbox[3] if transform[4] < 0 else bbox[1]
    if abs(transform[2] - want_x) > TOL_M or abs(transform[5] - want_y) > TOL_M:
        raise RefusedError(f"{name}: after the write the origin still disagrees with the bbox")
    # Required to be PRESENT and exactly right, not merely "not wrong if present" — a verification
    # that skips what it cannot find confirms nothing about the group that lost its registration.
    registered = {e.get("uuid"): e for e in attrs.get("zarr_conventions", []) if isinstance(e, dict)}
    for uuid, want in WANTED_CONVENTIONS.items():
        if registered.get(uuid) != want:
            raise RefusedError(f"{name}: the {want['name']} registration is missing or was not re-pinned")


def _rollback_hint(before: object, repair: object) -> str:
    """The undo command, as a compare-and-swap that refuses once someone else has built on it.

    ``from_snapshot_id`` is the point: a bare ``reset_branch`` would also discard any commit that
    landed after this repair, so an undo run a week later would silently take real work with it.
    Pinned to the repair's own snapshot, the reset succeeds only while it is still the tip.
    """
    return f"roll back if needed with repo.reset_branch('main', '{before}', from_snapshot_id='{repair}')"


def _tag_targets(repo: icechunk.Repository) -> dict[str, str]:
    """Every tag and the snapshot it points at, as the sibling reader-config repair records them."""
    return {tag: str(repo.lookup_tag(tag)) for tag in repo.list_tags()}


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
    # Where each tag POINTS, not how many there are. A count survives a tag being deleted and
    # another added, or any tag being retargeted, so it cannot support the claim the run makes.
    before_tags = _tag_targets(repo)
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
    # The plans describe `before_snapshot`. If `main` moved between the inspection and here, this
    # session is based on newer content the plans never saw, and Icechunk will not conflict on it
    # because the write is a valid edit of whatever it finds. Refusing is the only safe answer —
    # and it must be checked on the SESSION's base, which is what the write will actually build on.
    if str(session.snapshot_id) != str(before_snapshot):
        print(f"\nREFUSING: main moved from {before_snapshot} to {session.snapshot_id} since the inspection.")
        print("Nothing written. Re-run: the plans describe a snapshot that is no longer the tip.")
        return 1
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
        print(_rollback_hint(before_snapshot, snapshot))
        return 1
    after_tags, branches_after = _tag_targets(after), sorted(after.list_branches())
    if after_tags != before_tags or branches_after != branches:
        moved = sorted(set(after_tags.items()) ^ set(before_tags.items()))
        print(f"VERIFICATION FAILED: refs changed — branches {branches_after}, tags differing: {moved[:5]}")
        return 1
    print(f"verified {len(targets)} group(s); {len(after_tags)} tags all on their original snapshots")
    print(_rollback_hint(before_snapshot, snapshot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
