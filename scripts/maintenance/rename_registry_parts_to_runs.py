"""Move a registry's run files from ``parts/`` to ``runs/``, the folder's name since ADR 036.

Each run's file keeps its path below the folder, ``zone=<Z>/year=<Y>/<run_id>.parquet``. The code
reads and writes ``runs/`` only, so a registry written under the old name has to be moved once:
until it is, a fill's merge, ``verify`` and the registry census see no run files there. ``master/``,
``provenance/`` and the README beside them are not touched.

**Nothing else may write the registry while this runs.** It has no lock, and a fill or update that
lands mid-move writes its run file into whichever folder its code knows and merges ``master/``
against it. Stop every fill and update against this registry first.

**How the move stays safe.** Both folders are listed first. A file already at its destination that
differs from its source is a refusal, named, before anything is written; one that matches is kept,
so a rerun after a crash finishes the move. Every other file is copied, server-side on S3, and
every destination is then checked against its source: the size, then the ETag where both carry the
same one, and otherwise the bytes, since an ETag is not always the content's hash. Only once every
file checks are the sources deleted, and only those, so a file that appears in the source folder
mid-move is reported rather than lost. A registry with nothing left in the source folder and files
in the destination is reported as already migrated. A non-``s3://`` ``--registry`` is a local copy
moved through the same calls, which is how a move is rehearsed.

Dry run unless ``--apply``; ``--rollback`` moves ``runs/`` back to ``parts/`` under the same checks.
An S3 move uses the ambient AWS credentials, as ``rebuild_registry_measurements.py`` does, so run
it under a profile that can write the registry's bucket; ``--anonymous`` lists a public registry
for a dry run. Run from the REPOSITORY ROOT::

    uv run python scripts/maintenance/rename_registry_parts_to_runs.py --registry s3://<bucket>/<store>.registry
    uv run python scripts/maintenance/rename_registry_parts_to_runs.py --registry ... --apply
    uv run python scripts/maintenance/rename_registry_parts_to_runs.py --registry ... --rollback --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import fsspec
from fsspec import AbstractFileSystem

DEFAULT_REGISTRY = "s3://tessera-embeddings/v1.1/dclimate.registry"
DEFAULT_REGION = "us-west-2"
OLD, NEW = "parts", "runs"


def _filesystem(root: str, region: str, *, anonymous: bool) -> AbstractFileSystem:
    """S3 for an ``s3://`` registry, the local filesystem otherwise, behind the same calls.

    ``auto_mkdir`` because a local copy needs its ``zone=``/``year=`` folders made before a file
    lands in them, where S3 has no folders to make.
    """
    if root.startswith("s3://"):
        return fsspec.filesystem("s3", anon=anonymous, client_kwargs={"region_name": region})
    return fsspec.filesystem("file", auto_mkdir=True)


def _base(root: str) -> str:
    """``root`` as the filesystem names its paths: ``bucket/key`` on S3, an absolute path locally."""
    return root.removeprefix("s3://").rstrip("/") if root.startswith("s3://") else str(Path(root).resolve())


def _listing(fs: AbstractFileSystem, base: str, folder: str) -> dict[str, dict[str, Any]]:
    """Every file under ``folder``, keyed by its path below it, listed afresh rather than from a cache.

    An S3 folder marker, a key ending in ``/``, is not a file; the final clean-up removes it.
    """
    fs.invalidate_cache()
    prefix = f"{base}/{folder}/"
    found = fs.find(prefix.rstrip("/"), detail=True)
    return {
        path.removeprefix(prefix): info
        for path, info in found.items()
        if info["type"] == "file" and not path.endswith("/")
    }


def _same(fs: AbstractFileSystem, paths: tuple[str, str], infos: tuple[dict[str, Any], dict[str, Any]]) -> bool:
    """Whether two files hold the same bytes: unequal sizes say no, equal ETags say yes, else compare."""
    if infos[0]["size"] != infos[1]["size"]:
        return False
    etags = [info.get("ETag") for info in infos]
    return bool(etags[0] and etags[0] == etags[1]) or fs.cat_file(paths[0]) == fs.cat_file(paths[1])


def _differing(
    fs: AbstractFileSystem, base: str, folders: tuple[str, str], keys: list[str], listings: tuple[dict, dict]
) -> list[str]:
    """The ``keys`` whose file in the second folder is missing or differs from the one in the first."""
    source, dest = folders
    return [
        key
        for key in keys
        if key not in listings[1]
        or not _same(fs, (f"{base}/{source}/{key}", f"{base}/{dest}/{key}"), (listings[0][key], listings[1][key]))
    ]


def _megabytes(listing: dict[str, dict[str, Any]], keys: list[str] | None = None) -> str:
    return f"{sum(listing[key]['size'] for key in (listing if keys is None else keys)) / 1e6:,.1f} MB"


def _next_step(registry: str, folder: str) -> None:
    """What to run once the registry is in ``folder``."""
    if folder == NEW:
        print("\nnext, audit master/ against runs/ (add --store for a registry beside another store):")
        print(f"  uv run python scripts/maintenance/rebuild_registry_measurements.py verify --registry {registry}")
    else:
        print("\nparts/ is back: this checkout reads runs/, so verify from a checkout from before ADR 036")


def main(argv: list[str] | None = None) -> int:
    """Report, or make, the move. 0 when done or already done, 1 on a refusal."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="registry root; a non-s3 path is a local copy")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--anonymous", action="store_true", help="list with no credentials; dry run only")
    parser.add_argument("--apply", action="store_true", help="move; without it this is a dry run")
    parser.add_argument("--rollback", action="store_true", help="move runs/ back to parts/")
    args = parser.parse_args(argv)
    if args.anonymous and args.apply:
        parser.error("--anonymous cannot write; drop it to apply")
    source, dest = (NEW, OLD) if args.rollback else (OLD, NEW)

    fs = _filesystem(args.registry, args.region, anonymous=args.anonymous)
    base = _base(args.registry)
    have, held = _listing(fs, base, source), _listing(fs, base, dest)
    print(f"registry: {args.registry}")
    print(f"  {source}/ {len(have)} file(s), {_megabytes(have)}; {dest}/ {len(held)} file(s)")
    print("  nothing else may write this registry while this runs: stop every fill and update against it")
    if not have:
        if not held:
            print(f"\nREFUSING: neither {source}/ nor {dest}/ holds a file here. Is --registry right?")
            return 1
        done = "already rolled back" if args.rollback else "already migrated"
        print(f"\n{done}: {source}/ holds no file and {dest}/ holds {len(held)}")
        _next_step(args.registry, dest)
        return 0

    # Refusals before any write: a differing file at the destination is one nobody has looked at.
    present = sorted(key for key in have if key in held)
    if clashes := _differing(fs, base, (source, dest), present, (have, held)):
        print(f"\nREFUSING: {len(clashes)} file(s) in {dest}/ differ from their source, so nothing was written:")
        for key in clashes[:20]:
            print(f"  {dest}/{key}")
        return 1
    copy = sorted(key for key in have if key not in held)
    print(f"\n{len(copy)} file(s) to copy, {_megabytes(have, copy)}; {len(present)} already copied and identical:")
    for key in copy:
        print(f"  {source}/{key}  ->  {dest}/{key}")
    print(f"then delete all {len(have)} file(s) from {source}/, once every copy checks")
    if extra := len(held.keys() - have.keys()):
        print(f"{dest}/ also holds {extra} file(s) {source}/ lacks; they stay as they are")
    if not args.apply:
        print("\ndry run — pass --apply to move")
        return 0

    for done, key in enumerate(copy, 1):
        fs.cp_file(f"{base}/{source}/{key}", f"{base}/{dest}/{key}")
        if done % 100 == 0:
            print(f"  copied {done}/{len(copy)}")
    if bad := _differing(fs, base, (source, dest), sorted(have), (have, _listing(fs, base, dest))):
        print(f"\nFAILED: {len(bad)} copy(ies) missing or not matching their source; nothing was deleted, rerun:")
        for key in bad[:20]:
            print(f"  {dest}/{key}")
        return 1
    fs.rm([f"{base}/{source}/{key}" for key in sorted(have)])
    if left := _listing(fs, base, source):
        print(f"\nFAILED: {len(left)} file(s) appeared in {source}/ during the move and were left there:")
        for key in sorted(left)[:20]:
            print(f"  {source}/{key}")
        return 1
    if fs.exists(f"{base}/{source}"):
        fs.rm(f"{base}/{source}", recursive=True)  # empty now: a local copy's folders, or an S3 folder marker
    print(f"\nmoved {len(have)} file(s) from {source}/ to {dest}/")
    _next_step(args.registry, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
