"""Tear down a scale-test run's stores (and optionally results).

Deletes the run's store prefixes so a throwaway bucket returns to $0, and
verifies the prefix is empty. Results are kept by default (they are the run's
product); pass ``--purge-results`` to remove them too.

Run from the REPOSITORY ROOT::

    uv run python -m scripts.scoping.scale_tests.teardown --run-id dev --backend local
    uv run python -m scripts.scoping.scale_tests.teardown --run-id dev --backend s3 --bucket my-throwaway
"""

from __future__ import annotations

import argparse
import logging

import fsspec

from scripts.scoping.scale_tests import harness
from tessera_embeddings.storage.object_store import delete_prefix

logger = logging.getLogger("scale_tests.teardown")


def _rm_prefix(uri: str) -> tuple[int, int]:
    """Delete everything under ``uri`` (all versions on S3); return (before, after)."""
    before, _ = harness.object_stats(uri)
    fs, path = harness.fs_and_path(uri)
    if fs.exists(path):
        if uri.startswith("s3://"):
            # All-version delete: on a versioned bucket `fs.rm(recursive=True)` only
            # writes delete markers, leaving the (still-billed) object versions behind
            # even though object_stats then reports the prefix empty. delete_prefix
            # uses `s5cmd rm --all-versions` (fsspec fallback) so it truly empties.
            delete_prefix(uri, log=logger, all_versions=True)
        else:
            fs.rm(path, recursive=True)
    after, _ = harness.object_stats(uri)
    return before, after


def main() -> int:
    """Parse args and remove the run's store (and optionally results) prefixes."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    parser.add_argument("--purge-results", action="store_true", help="Also delete local + S3 results.")
    parser.add_argument(
        "--force", action="store_true", help="Delete a store root even when it is not scoped by the run id."
    )
    args = parser.parse_args()
    cfg = harness.config_from_args(args)
    harness.configure_logging()

    # The run_id must be the FINAL prefix component of a non-bucket-root store
    # root — mere membership isn't enough (`--run-id my-bucket` would "scope"
    # `s3://my-bucket/anything`), and a bucket root with no key prefix must never
    # be recursively deleted. So require leaf == run_id AND a key beyond the bucket.
    root = cfg.store_root.rstrip("/")
    rest = root.split("://", 1)[1] if "://" in root else root
    parts = [p for p in rest.split("/") if p]
    leaf = parts[-1] if parts else ""
    has_key_prefix = len(parts) > 1  # bucket + at least one key component
    if not args.force and (leaf != cfg.run_id or not has_key_prefix):
        raise SystemExit(
            f"store root {cfg.store_root!r} is not scoped by run id {cfg.run_id!r} (run id must be the "
            "final key component, and the root must not be a bare bucket) — a recursive delete could take "
            "out other runs or a whole shared prefix. Pass --force to override."
        )

    before, after = _rm_prefix(cfg.store_root)
    logger.info("stores: removed %d objects under %s (%d remain)", before - after, cfg.store_root, after)
    # A count of what was removed is not a statement about what is left, and this command's
    # whole advertised outcome is return-to-$0. Exiting zero with a warning meant an
    # automated caller — or an operator reading `$?` — recorded a successful teardown while
    # a benchmark store kept billing. The warning stays for the human; the status is what
    # the machine reads.
    residue = after
    if residue:
        logger.error("store prefix NOT empty after teardown: %d object(s) remain under %s", residue, cfg.store_root)

    if args.purge_results:
        if cfg.results_dir.exists():
            fsspec.filesystem("file").rm(str(cfg.results_dir), recursive=True)
            logger.info("results: removed local %s", cfg.results_dir)
        if cfg.is_s3 and cfg.bucket:
            # Same prefix-aware root the mirror writes to (RunConfig.s3_results_root),
            # so a prefix-scoped run's mirrored results are actually reclaimed.
            s3_results = f"s3://{cfg.s3_results_root}"
            b, a = _rm_prefix(s3_results)
            logger.info("results: removed %d S3 objects under %s", b - a, s3_results)
    else:
        logger.info("results kept at %s (pass --purge-results to remove)", cfg.results_dir)
    return 1 if residue else 0


if __name__ == "__main__":
    raise SystemExit(main())
