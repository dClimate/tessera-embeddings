r"""Loosen a global store's minimum optical depth rule, for the whole store or named zones. Dry run unless ``--write``.

The rule (``optical_min_obs``) only ever moves down (ADR 028): every pixel already embedded stays
compliant with the new line, and every pixel it newly admits is a hole that a ``holes`` update
then fills (``scripts/update/plan_update.py``, ADR 027)::

    loosen_optical_min_obs.py --store s3://tessera-embeddings/v1.1/dclimate.icechunk --line 1 --zones 18N,19N
    loosen_optical_min_obs.py --store ... --line 1 --zones 18N,19N --write

Without ``--zones`` the store root moves, and any zone's own rule at or above the new line goes with
it. Prints the root rule, each affected zone's effective rule and every history, before and after.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import cast

import icechunk
import zarr

from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials, icechunk_credentials_for
from tessera_embeddings.storage.global_store import (
    OPTICAL_RULE_HISTORY_ATTR,
    effective_optical_min_obs,
    loosen_optical_min_obs,
    open_global_repo,
)
from tessera_embeddings.storage.zone_grid import canonicalize_zone


def _report(repo: icechunk.Repository, zones: Sequence[str] | None) -> zarr.Group:
    """Print the root's rule and history, then each named zone's (or each overriding zone's); return the root."""
    root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    print(f"  root: optical_min_obs={root.attrs.get('optical_min_obs')!r}")
    print(f"        history={root.attrs.get(OPTICAL_RULE_HISTORY_ATTR, [])}")
    shown = zones if zones is not None else [g for g in sorted(root.group_keys()) if "optical_min_obs" in root[g].attrs]
    for zone in shown:
        if zone not in root:
            print(f"  {zone}: not seeded")
            continue
        node = cast(zarr.Group, root[zone])
        print(
            f"  {zone}: effective={effective_optical_min_obs(root, zone)!r} own={node.attrs.get('optical_min_obs')!r}"
        )
        print(f"        history={node.attrs.get(OPTICAL_RULE_HISTORY_ATTR, [])}")
    return root


def _planned(root: zarr.Group, line: int, zones: Sequence[str] | None) -> None:
    """What ``--write`` would do to each target, by the rules :func:`loosen_optical_min_obs` enforces."""
    raw = root.attrs.get("optical_min_obs")
    targets = (
        [("store", None if raw is None else int(cast(int, raw)))]
        if zones is None
        else [(z, effective_optical_min_obs(root, z) if z in root else None) for z in zones]
    )
    for where, current in targets:
        if current is None:
            verdict = "refused: no rule (or not seeded)"
        elif line > current:
            verdict = f"refused: {current} -> {line} is a tightening"
        else:
            verdict = "unchanged" if line == current else f"{current} -> {line}"
        print(f"  {where}: {verdict}")


def main(argv: list[str] | None = None) -> int:
    """Report the rule, and with ``--write`` loosen it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True, help="the global store's URI (s3:// or a local path)")
    parser.add_argument("--line", required=True, type=int, help="the new, looser minimum valid optical observations")
    parser.add_argument("--zones", help="comma-separated zones to loosen; omit to loosen the whole store")
    parser.add_argument("--region", default=None, help="S3 region of the store")
    parser.add_argument("--write", action="store_true", help="commit the change; without it this is a dry run")
    args = parser.parse_args(argv)

    zones = sorted({canonicalize_zone(z) for z in args.zones.split(",")}) if args.zones else None
    credentials = (
        icechunk_credentials_for(args.store, iam_icechunk_credentials) if args.store.startswith("s3://") else None
    )
    repo = open_global_repo(args.store, get_credentials=credentials, region=args.region)
    print(f"Before ({args.store}):")
    root = _report(repo, zones)
    if not args.write:
        print(f"Dry run: pass --write to loosen {'the store' if zones is None else ', '.join(zones)} to {args.line}")
        _planned(root, args.line, zones)
        return 0
    snapshot = loosen_optical_min_obs(repo, args.line, zones=zones)
    print(f"Committed {snapshot}" if snapshot else "Nothing to change: every target is already at that line")
    print("After:")
    _report(repo, zones)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
