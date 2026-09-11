"""Is rewriting a published store's saved configuration safe? Prove it against a real store.

**Why this needs proving rather than reasoning about.** On a spec-version-2 repository,
``Repository.save_config`` does not write a small config file off to one side. It rewrites the
single ``repo`` object — and that object holds the branch pointers, every tag, the deleted-tag
list, every snapshot record, the repository metadata, the feature flags and the status. Icechunk
rebuilds the whole thing from its parts (``RepoInfo::set_config``) and puts it back. So a
configuration change is a full rewrite of the store's entire reference state, and on the published
global store that state includes the ``main`` pointer and 1,066 completion tags. Losing it would
lose the record of what the campaign delivered.

Reading the implementation says it should be safe: the previous object is copied to
``overwritten/repo.<time>.<id>`` first, both the copy and the put are conditional on the version the
caller read, a lost race surfaces as ``RepoInfoUpdated`` and retries, and a single object PUT is
atomic in S3 so there is no torn state to observe. That is an argument, not evidence. This script
is the evidence.

**Run it against a throwaway store, never against the published one.** It mutates what it is
pointed at. The point is to establish the behaviour somewhere expendable and then trust it.

    uv run python scripts/diagnostic/repo_config_write_safety.py --store /tmp/probe.icechunk
    uv run python scripts/diagnostic/repo_config_write_safety.py \
        --store s3://global-tessera-embeddings/diagnostics/config-write-probe.icechunk

Exits non-zero if any check fails. Every check is stated as what would have to be true for the
operation to be unsafe, so a pass is a statement about the failure mode and not just a green tick.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from typing import Any

import icechunk
import numpy as np
import zarr

from tessera_embeddings.storage.global_store import create_global_repo, seed_zone_groups
from tessera_embeddings.storage.zarr_store import _create_storage, global_store_config
from tessera_embeddings.storage.zone_grid import ZoneSpec

#: Two small stand-in zones, a couple of shards each, so the fixture builds in seconds while still
#: exercising the multi-group layout the published store uses.
_ZONES = (
    ZoneSpec("32601", "N", 1, (0.0, 40_960.0), (0.0, 40_960.0)),
    ZoneSpec("32701", "S", 1, (0.0, 40_960.0), (1_105_920.0, 1_146_880.0)),
)
_YEARS = (2023, 2024, 2025)


def _storage(uri: str) -> icechunk.Storage:
    """Storage for a local path or an S3 URI, with no repository config attached."""
    return _create_storage(uri)


def clone_reference_state(source: str, destination: str) -> None:
    """Copy ``source``'s ``repo`` object into ``destination``, and nothing else.

    That one object holds the entire reference state, and `set_config` rebuilds it from itself
    rather than from the snapshots it names — so a copy of it alone is enough to rewrite a real
    store's tags and branch pointers at real scale, without touching a byte of anybody's data. The
    source is opened read-only; only the destination is written.
    """
    import boto3

    for uri in (source, destination):
        if not uri.startswith("s3://"):
            raise ValueError(f"cloning a reference state needs two S3 URIs, got {uri!r}")
    src_bucket, _, src_prefix = source.removeprefix("s3://").partition("/")
    dst_bucket, _, dst_prefix = destination.removeprefix("s3://").partition("/")
    print(f"cloning the reference state of {source}")
    boto3.client("s3").copy_object(
        Bucket=dst_bucket,
        Key=f"{dst_prefix.rstrip('/')}/repo",
        CopySource={"Bucket": src_bucket, "Key": f"{src_prefix.rstrip('/')}/repo"},
    )


def capture(uri: str) -> dict[str, Any]:
    """Everything about the store that a configuration write must not change.

    Read through a FRESH open each time, with no configuration passed, so what comes back is what
    the store itself holds rather than anything this process supplied.
    """
    repo = icechunk.Repository.open(_storage(uri))
    state: dict[str, Any] = {
        "spec_version": str(repo.spec_version),
        "branches": sorted(repo.list_branches()),
        "tags": sorted(repo.list_tags()),
        "tag_targets": {tag: str(repo.lookup_tag(tag)) for tag in sorted(repo.list_tags())},
        "branch_tips": {b: str(repo.lookup_branch(b)) for b in sorted(repo.list_branches())},
        "config": repr(repo.config),
        "preload_refs": repo.config.manifest.preload.max_total_refs,
        "splitting": repr(repo.config.manifest.splitting),
        "storage_settings": repr(repo.config.storage),
    }
    # A cloned reference state names snapshots whose objects were not copied, so there is nothing to
    # open — and nothing that needs opening, because what is under test is the reference state
    # itself. Where the data IS present it is hashed: metadata can survive a rewrite that has
    # broken the chunk references, so a check that never reads a value is not a check that the
    # store still works.
    try:
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(session.store, mode="r")
        state["snapshot_id"] = str(session.snapshot_id)
        state["ancestry"] = [(str(s.id), s.message) for s in repo.ancestry(branch="main")]
        state["root_attrs"] = {k: repr(root.attrs[k]) for k in sorted(root.attrs)}
        state["groups"] = sorted(name for name, _ in root.groups())
        state["data_sha256"] = {
            zone: hashlib.sha256(np.asarray(root[zone]["scales"][:]).tobytes()).hexdigest() for zone in state["groups"]
        }
        state["years_complete"] = {zone: sorted(root[zone].attrs.get("years_complete", [])) for zone in state["groups"]}
    except Exception as exc:
        state["data_unreadable"] = type(exc).__name__
    return state


def build_fixture(uri: str) -> None:
    """Create a small multi-group store with data, several commits and several tags."""
    repo = create_global_repo(uri)
    seed_zone_groups(repo, list(_ZONES), years=_YEARS)
    for index, year in enumerate(_YEARS):
        session = repo.writable_session("main")
        group = zarr.open_group(session.store, mode="r+")["01N"]
        group["scales"][index, :2048, :2048] = np.full((2048, 2048), 0.5 + index, dtype="float32")
        group.attrs["years_complete"] = list(_YEARS[: index + 1])
        snapshot = session.commit(f"fill 01N year {year}")
        repo.create_tag(f"zone-01N-{year}", snapshot)
    repo.create_tag("year-2025-complete", repo.lookup_branch("main"))


def _objects(uri: str) -> set[str]:
    """Object keys under the store prefix, for an S3 store; empty for a local one."""
    if not uri.startswith("s3://"):
        return set()
    import boto3

    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    client = boto3.client("s3")
    keys: set[str] = set()
    token = None
    while True:
        page = client.list_objects_v2(Bucket=bucket, Prefix=prefix, **({"ContinuationToken": token} if token else {}))
        keys |= {item["Key"] for item in page.get("Contents", [])}
        if not page.get("IsTruncated"):
            return keys
        token = page["NextContinuationToken"]


def _diff(before: dict[str, Any], after: dict[str, Any], ignoring: set[str]) -> list[str]:
    """Keys whose value changed, excluding the ones a config write is allowed to change."""
    return [k for k in before if k not in ignoring and before[k] != after[k]]


def main(argv: list[str] | None = None) -> int:
    """Run the battery; return 0 only if a config write is provably non-destructive here."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True, help="THROWAWAY store URI — this script mutates it")
    parser.add_argument("--skip-build", action="store_true", help="reuse an existing fixture")
    parser.add_argument(
        "--clone-reference-state-from",
        metavar="URI",
        help=(
            "instead of building a fixture, copy that store's `repo` object into the throwaway "
            "prefix and rewrite THAT. The repo object carries the whole reference state — every "
            "tag, the branch pointers, the snapshot records — so this exercises the real thing at "
            "real scale, which a small fixture cannot. Reads the source store only."
        ),
    )
    args = parser.parse_args(argv)
    if "dclimate.icechunk" in args.store:
        parser.error("refusing to run against the published store; point this at a throwaway")

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{f' — {detail}' if detail else ''}")
        if not ok:
            failures.append(name)

    if args.clone_reference_state_from:
        clone_reference_state(args.clone_reference_state_from, args.store)
    elif not args.skip_build:
        print(f"building fixture at {args.store}")
        build_fixture(args.store)

    before = capture(args.store)
    objects_before = _objects(args.store)
    shape = (
        f"{len(before['groups'])} groups, {len(before['ancestry'])} snapshots"
        if "groups" in before
        else f"reference state only ({before['data_unreadable']} on the data, as a clone expects)"
    )
    print(f"\nsubject: {shape}, {len(before['tags'])} tags, preload refs {before['preload_refs']}")

    print("\n1. writing a new configuration (preload zeroed)")
    repo = icechunk.Repository.open(_storage(args.store), config=global_store_config(preload_manifests=False))
    repo.save_config()
    after = capture(args.store)
    objects_after = _objects(args.store)

    check("the saved configuration changed", after["preload_refs"] == 0, f"refs now {after['preload_refs']}")
    check(
        "no other recorded state changed",
        not _diff(before, after, {"config", "preload_refs"}),
        ", ".join(_diff(before, after, {"config", "preload_refs"})) or "nothing else moved",
    )
    check("the branch tip is unmoved", after["branch_tips"] == before["branch_tips"])
    check(
        "every tag survived",
        after["tag_targets"] == before["tag_targets"],
        f"{len(after['tags'])} of {len(before['tags'])}",
    )
    check(
        "no snapshot was created",
        len(after.get("ancestry", [])) == len(before.get("ancestry", [])),
        f"{len(after.get('ancestry', []))} snapshots",
    )
    check("the data still reads back identically", after.get("data_sha256") == before.get("data_sha256"))
    check("manifest splitting survived", after["splitting"] == before["splitting"])
    check("storage timeouts and retries survived", after["storage_settings"] == before["storage_settings"])

    if objects_before:
        added = objects_after - objects_before
        removed = objects_before - objects_after
        backups = {k for k in added if "overwritten/" in k}
        check("the previous repo object was backed up", bool(backups), f"{len(backups)} backup object(s)")
        check("nothing was deleted", not removed, ", ".join(sorted(removed)[:3]) or "nothing removed")
        check(
            "only the repo object and its backup were touched",
            all("overwritten/" in k or k.endswith("/repo") for k in added),
            ", ".join(sorted(added)[:3]),
        )

    print("\n2. rolling the configuration back")
    repo = icechunk.Repository.open(_storage(args.store), config=global_store_config())
    repo.save_config()
    restored = capture(args.store)
    check("the original configuration is byte-identical after rollback", restored["config"] == before["config"])
    check(
        "state is still intact after two writes",
        not _diff(before, restored, set()),
        ", ".join(_diff(before, restored, set())) or "identical",
    )

    print("\n3. a stale writer must not silently clobber")
    # Two handles read the same version; the first write moves it, so the second is working from a
    # version that no longer exists. Icechunk must either detect that and retry onto the new state,
    # or refuse — what it must not do is overwrite blind and lose the first writer's change.
    stale = icechunk.Repository.open(_storage(args.store), config=global_store_config(preload_manifests=False))
    fresh = icechunk.Repository.open(_storage(args.store), config=global_store_config(preload_manifests=False))
    fresh.save_config()
    try:
        stale.save_config()
        outcome = "retried onto the new version and succeeded"
        clobbered = False
    except Exception as exc:
        outcome = f"refused with {type(exc).__name__}"
        clobbered = False
    contested = capture(args.store)
    check(
        "a contested write neither lost state nor corrupted it",
        not _diff(before, contested, {"config", "preload_refs"}) and not clobbered,
        outcome,
    )
    check("every tag survived the contested write", contested["tag_targets"] == before["tag_targets"])

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
