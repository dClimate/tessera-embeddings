"""Switch the published store's SAVED manifest preload off, so readers stop inheriting it.

**What this changes and why.** Icechunk persists a repository's configuration inside the store, so
every consumer who opens it without supplying their own inherits whatever the writer used. The
campaign wrote with a manifest preload sized for a fill — a million refs across 2,400 arrays — and
that costs a reader about 2.5 s of every open and returns nothing measurable: point latency, region
throughput and bytes on the wire are unchanged without it, in-region and cross-region, including
for a reader touching many zones, which is the one population the setting could plausibly have
still been helping. Measured in ``context_docs/storage/reading-the-published-store.md`` §4.3.

**What it costs, which is not nothing.** On a spec-version-2 repository ``save_config`` rewrites the
single ``repo`` object, and that object holds the branch pointers, every tag, the deleted-tag list,
every snapshot record, the metadata, the feature flags and the status. Icechunk rebuilds all of it
and puts it back. On the published store that is the ``main`` pointer and 1,070 completion tags —
the record of what the campaign delivered.

That write was verified safe before this script was written, not assumed. A battery run against a
throwaway clone of this store's own reference state — the real ``repo`` object, all 1,070 tags —
showed every tag, the branch pointer, the spec version, the manifest splitting and the storage
settings surviving a rewrite and a rollback unchanged, with the previous object copied to
``overwritten/`` first and nothing deleted. The evidence is recorded in
``context_docs/storage/reading-the-published-store.md``; the one-off battery that produced it was
removed once the question was settled.

**Nothing in our own code is affected either way.** Both places the package opens Icechunk pass a
configuration explicitly, which replaces the saved one, and forked assembly workers are handed the
coordinator's live session rather than re-opening storage. The saved configuration is consumed only
by a consumer who passes none — which is exactly the population this helps. A future fill is
unaffected, because it asks for the preload by name.

**Dry run by default.** It prints what it would change and exits. ``--apply`` writes, and then
immediately re-reads the store and verifies that nothing but the configuration moved; if anything
else did, it says so and names the backup object to restore from. ``--rollback`` puts the writer's
configuration back.

    uv run python scripts/maintenance/set_published_store_reader_config.py                  # dry run
    PUBLISHED_STORE_WRITER_ROLE_ARN=arn:aws:iam::601791338954:role/... \
        uv run python scripts/maintenance/set_published_store_reader_config.py --apply
    PUBLISHED_STORE_WRITER_ROLE_ARN=... \
        uv run python scripts/maintenance/set_published_store_reader_config.py --rollback
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import icechunk
import zarr

from tessera_embeddings.providers.aws.credentials import icechunk_credentials_for, published_store_writer_role
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.zarr_store import _create_storage, global_store_config

DEFAULT_URI = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
DEFAULT_REGION = "us-west-2"

#: Preferred zone for the post-write read-back, because it is small and so costs about a second.
#: NOT required to exist: naming a zone only the published store has made this script report a
#: failure against every other store, which is the same defect as hard-coding anonymous reads — a
#: check that can only run against production cannot be rehearsed before production.
PREFERRED_SPOT_CHECK_ZONE = "16S"


def _open_for_read(uri: str, region: str) -> icechunk.Repository:
    """Open for reading with no configuration, anonymously if the store allows it."""
    try:
        return icechunk.Repository.open(_create_storage(uri, anonymous=True, region=region))
    except icechunk.IcechunkError:
        return icechunk.Repository.open(_create_storage(uri, region=region))


def read_state(uri: str, region: str) -> dict[str, Any]:
    """The state a configuration write must not disturb, read with no configuration supplied.

    **Opened with no config**, so what comes back is what the STORE holds rather than anything this
    process handed it — which is the whole point, since `open_global_repo` supplies
    `global_store_config()` and would mask a store that had lost its own.

    Tries anonymously first and falls back to the ambient credential chain, because each mode alone
    breaks a case that matters. Forcing anonymous made the script impossible to rehearse anywhere
    but the published store — a throwaway failed with `AccessDenied` before reaching the write it
    was meant to be testing. Forcing the ambient chain then broke the published store for an
    operator with no credentials loaded, which is how its own dry run is most naturally invoked.
    Credentials are orthogonal to what this function is for, which is passing no CONFIGURATION.
    """
    repo = _open_for_read(uri, region)
    tags = sorted(repo.list_tags())
    return {
        "spec_version": str(repo.spec_version),
        "branches": sorted(repo.list_branches()),
        "branch_tips": {b: str(repo.lookup_branch(b)) for b in sorted(repo.list_branches())},
        "tag_count": len(tags),
        "tag_targets": {t: str(repo.lookup_tag(t)) for t in tags},
        "config": repr(repo.config),
        "preload_refs": repo.config.manifest.preload.max_total_refs,
        "preload_arrays": repo.config.manifest.preload.max_arrays_to_scan,
        "splitting": repr(repo.config.manifest.splitting),
        "storage_settings": repr(repo.config.storage),
    }


def backups(uri: str, region: str) -> set[str]:
    """Keys under the store's ``overwritten/`` prefix — where the previous ``repo`` object lands."""
    import boto3

    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    client = boto3.client("s3", region_name=region)
    keys: set[str] = set()
    token = None
    while True:
        page = client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{prefix.rstrip('/')}/overwritten/",
            **({"ContinuationToken": token} if token else {}),
        )
        keys |= {item["Key"] for item in page.get("Contents", [])}
        if not page.get("IsTruncated"):
            return keys
        token = page["NextContinuationToken"]


def refuse_unless_expected(state: dict[str, Any], want_preload: bool, expect_tags: int | None) -> str | None:
    """Why this store must not be written, or None if it is in the state we verified against.

    Every one of these is a reason to stop rather than a reason to adapt. The safety evidence was
    gathered against a spec-version-2 store whose saved configuration is exactly what this package
    writes; a store that has drifted from that is a store nobody has tested this on.
    """
    if state["spec_version"] != "SpecVersion.v2 (current)":
        return f"spec version is {state['spec_version']}; the safety check covered v2 only"
    if state["branches"] != ["main"]:
        return f"expected one branch named main, found {state['branches']}"
    expected = repr(global_store_config(preload_manifests=not want_preload))
    if state["config"] == repr(global_store_config(preload_manifests=want_preload)):
        return "ALREADY_DONE"
    if state["config"] != expected:
        return (
            "the store's saved configuration is neither what this package writes nor the state "
            "this script produces — somebody has changed it, so stop and look"
        )
    if expect_tags is not None and state["tag_count"] != expect_tags:
        return f"expected {expect_tags} tags, found {state['tag_count']}"
    return None


def main(argv: list[str] | None = None) -> int:
    """Report, or apply, the configuration change. 0 on success or a clean no-op; 1 otherwise."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--apply", action="store_true", help="actually write; without it this is a dry run")
    parser.add_argument("--rollback", action="store_true", help="put the writer's preload back")
    parser.add_argument(
        "--expect-tags",
        type=int,
        default=1070,
        help="refuse unless the store carries exactly this many tags (0 disables the guard)",
    )
    args = parser.parse_args(argv)
    if args.apply and args.rollback:
        parser.error("--apply and --rollback are opposites; pass one")

    # `--rollback` restores the writer's preload, `--apply` removes it.
    want_preload = bool(args.rollback)
    expect_tags = args.expect_tags or None

    print(f"store:  {args.uri} ({args.region})")
    before = read_state(args.uri, args.region)
    print(f"state:  spec {before['spec_version']}, branches {before['branches']}, {before['tag_count']} tags")
    print(f"        preload refs {before['preload_refs']}, arrays {before['preload_arrays']}")

    refusal = refuse_unless_expected(before, want_preload, expect_tags)
    if refusal == "ALREADY_DONE":
        print(f"\nnothing to do: the saved configuration already has preload {'on' if want_preload else 'off'}")
        return 0
    if refusal:
        print(f"\nREFUSING: {refusal}")
        return 1

    action = "restore the writer's manifest preload" if want_preload else "switch the manifest preload off"
    print(f"\nwould {action}, leaving splitting, timeouts and retries as they are")
    if not (args.apply or args.rollback):
        print("dry run — pass --apply to write, or --rollback to undo")
        return 0

    role = published_store_writer_role()
    if role is None and args.uri == DEFAULT_URI:
        print(
            "\nREFUSING: writing the published store needs PUBLISHED_STORE_WRITER_ROLE_ARN set to the "
            "role in the owner's account. Our own identity is read-only there by design."
        )
        return 1

    known_backups = backups(args.uri, args.region)
    print(f"\nwriting... ({len(known_backups)} existing backup object(s))")
    repo = open_global_repo(
        args.uri,
        get_credentials=icechunk_credentials_for(args.uri),
        region=args.region,
        preload_manifests=want_preload,
    )
    repo.save_config()

    after = read_state(args.uri, args.region)
    new_backups = sorted(backups(args.uri, args.region) - known_backups)
    moved = [k for k in before if k not in {"config", "preload_refs", "preload_arrays"} and before[k] != after[k]]

    print(f"  preload refs now {after['preload_refs']}, arrays {after['preload_arrays']}")
    print(f"  backup of the previous repo object: {new_backups[0] if new_backups else 'NONE FOUND'}")
    print(f"  tags {after['tag_count']}, branch tip {after['branch_tips'].get('main')}")

    ok = True
    # Compared against what this package WRITES for the requested direction, not against the value
    # the store happened to hold before. The earlier form asked whether the new value equalled the
    # OLD one whenever restoring, which is false exactly when a rollback succeeds — a spurious
    # failure at the one moment somebody is undoing something and least wants to doubt the tool.
    expected_refs = global_store_config(preload_manifests=want_preload).manifest.preload.max_total_refs
    if after["preload_refs"] != expected_refs:
        print(f"  PROBLEM: the configuration did not take — refs {after['preload_refs']}, wanted {expected_refs}")
        ok = False
    if moved:
        print(f"  PROBLEM: something other than the configuration changed: {moved}")
        ok = False
    if not new_backups:
        print("  PROBLEM: no backup object appeared, so there is nothing to restore from")
        ok = False

    # One real read, because a store whose metadata reconciles can still have broken references.
    try:
        session = _open_for_read(args.uri, args.region).readonly_session(branch="main")
        root = zarr.open_group(session.store, mode="r")
        present = sorted(name for name, _ in root.groups())
        zone = PREFERRED_SPOT_CHECK_ZONE if PREFERRED_SPOT_CHECK_ZONE in present else (present or [None])[0]
        if zone is None:
            print("  spot check: the store holds no zone groups, so there is nothing to read back")
        else:
            years = list(root[zone].attrs["years_complete"])
            print(f"  spot check: {zone} still opens and reports {len(years)} complete years")
    except Exception as exc:
        print(f"  PROBLEM: the store no longer reads: {type(exc).__name__}: {exc}")
        ok = False

    if not ok:
        print(
            f"\nFAILED. Restore by copying {new_backups[0] if new_backups else '<the newest overwritten/repo.*>'} "
            f"back over {args.uri.removeprefix('s3://')}/repo, or re-run with the opposite flag."
        )
        return 1
    print(f"\ndone. Undo with --rollback, or by restoring {new_backups[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
