"""Tear down a scale-test run's stores (and optionally results).

Deletes the run's store prefixes so a throwaway bucket returns to $0, and
verifies the prefix is empty. Results are kept by default (they are the run's
product); pass ``--purge-results`` to remove them too.

Run from ``scripts/``::

    uv run python -m scale_tests.teardown --run-id dev --backend local
    uv run python -m scale_tests.teardown --run-id dev --backend s3 --bucket my-throwaway
"""

from __future__ import annotations

import argparse
import logging

import fsspec

from scale_tests import harness

logger = logging.getLogger("scale_tests.teardown")


def _rm_prefix(uri: str) -> tuple[int, int]:
    """Delete everything under ``uri``; return (objects_before, objects_after)."""
    before, _ = harness.object_stats(uri)
    fs, path = harness.fs_and_path(uri)
    if fs.exists(path):
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

    if cfg.run_id not in cfg.store_root and not args.force:
        raise SystemExit(
            f"store root {cfg.store_root!r} is not scoped by run id {cfg.run_id!r} — a recursive "
            "delete could take out other runs (or a whole shared prefix). Pass --force to override."
        )

    before, after = _rm_prefix(cfg.store_root)
    logger.info("stores: removed %d objects under %s (%d remain)", before - after, cfg.store_root, after)
    if after != 0:
        logger.warning("store prefix not empty after teardown: %d objects remain", after)

    if args.purge_results:
        if cfg.results_dir.exists():
            fsspec.filesystem("file").rm(str(cfg.results_dir), recursive=True)
            logger.info("results: removed local %s", cfg.results_dir)
        if cfg.is_s3 and cfg.bucket:
            s3_results = f"s3://{cfg.bucket}/results/{cfg.run_id}"
            b, a = _rm_prefix(s3_results)
            logger.info("results: removed %d S3 objects under %s", b - a, s3_results)
    else:
        logger.info("results kept at %s (pass --purge-results to remove)", cfg.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
