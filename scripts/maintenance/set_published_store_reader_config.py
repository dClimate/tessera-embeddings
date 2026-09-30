"""Switch the published store's saved manifest preload off, so readers stop inheriting it.

Preloading manifests helps a fill and costs a reader about two seconds per open for nothing, so it
belongs on during a campaign and off afterwards. A campaign already does this at its end; this
script is for doing it by hand, or undoing it.

The write rewrites the store's ``repo`` object, which holds the branch pointers and every tag —
safe, verified against a clone of this store's real reference state, and described in
``context_docs/storage/reading-the-published-store.md`` §4.3a. Dry run unless ``--apply``.

    uv run python scripts/maintenance/set_published_store_reader_config.py            # dry run
    AWS_PROFILE=... PUBLISHED_STORE_WRITER_ROLE_ARN=... uv run python \
        scripts/maintenance/set_published_store_reader_config.py --apply
"""

from __future__ import annotations

import argparse
import sys

import icechunk

from tessera_embeddings.providers.aws.credentials import icechunk_credentials_for, published_store_writer_role
from tessera_embeddings.storage.global_store import set_saved_manifest_preload
from tessera_embeddings.storage.zarr_store import _create_storage

DEFAULT_URI = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
DEFAULT_REGION = "us-west-2"


def _open_for_read(uri: str, region: str) -> icechunk.Repository:
    """Open with no config, anonymously where the store allows it.

    No config, so what comes back is what the store holds. Anonymous first because the published
    store's dry run is naturally invoked with no credentials loaded; the fallback is what lets this
    be rehearsed against a private throwaway.
    """
    try:
        return icechunk.Repository.open(_create_storage(uri, anonymous=True, region=region))
    except icechunk.IcechunkError:
        return icechunk.Repository.open(_create_storage(uri, region=region))


def main(argv: list[str] | None = None) -> int:
    """Report, or apply, the change. 0 on success or a clean no-op."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--apply", action="store_true", help="write; without it this is a dry run")
    parser.add_argument("--rollback", action="store_true", help="put the writer's preload back")
    parser.add_argument("--expect-tags", type=int, default=1070, help="refuse on a different count; 0 disables")
    args = parser.parse_args(argv)
    if args.apply and args.rollback:
        parser.error("--apply and --rollback are opposites; pass one")
    enabled = bool(args.rollback)

    repo = _open_for_read(args.uri, args.region)
    tags, branches = sorted(repo.list_tags()), sorted(repo.list_branches())
    refs = repo.config.manifest.preload.max_total_refs
    print(f"store:  {args.uri} ({args.region})")
    print(f"state:  {repo.spec_version}, branches {branches}, {len(tags)} tags, preload refs {refs}")

    # Refusals, not adaptations: the safety evidence was gathered against a store in this shape,
    # and one that has drifted is one nobody has tested this on.
    if bool(refs) == enabled:
        print(f"nothing to do: preload is already {'on' if enabled else 'off'}")
        return 0
    if branches != ["main"]:
        print(f"REFUSING: expected one branch named main, found {branches}")
        return 1
    if args.expect_tags and len(tags) != args.expect_tags:
        print(f"REFUSING: expected {args.expect_tags} tags, found {len(tags)}")
        return 1

    print(f"\nwould switch the manifest preload {'on' if enabled else 'off'}")
    if not (args.apply or args.rollback):
        print("dry run — pass --apply to write, or --rollback to undo")
        return 0

    if published_store_writer_role() is None and args.uri == DEFAULT_URI:
        print("\nREFUSING: writing the published store needs PUBLISHED_STORE_WRITER_ROLE_ARN; our own")
        print("identity is read-only there by design.")
        return 1

    # `set_saved_manifest_preload` raises unless the preload changed and every tag and branch
    # survived, so reaching the next line is the verification.
    set_saved_manifest_preload(
        args.uri, enabled=enabled, get_credentials=icechunk_credentials_for(args.uri), region=args.region
    )
    after = _open_for_read(args.uri, args.region)
    print(f"done: preload refs now {after.config.manifest.preload.max_total_refs}, {len(after.list_tags())} tags")
    print("undo by re-running with the opposite flag, or restore the newest object under overwritten/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
