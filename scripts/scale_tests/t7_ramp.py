"""T7 — fresh-bucket PUT ramp (test plan §4 T7, optional).

Deliberately provokes S3 ``503 SlowDown`` on a cold bucket to calibrate the
campaign warm-up and confirm/adjust ``TARGET_AGGREGATE_S3_CONCURRENCY``. Raw
boto3 PUTs of fixed-size objects at ramped concurrency; no icechunk involved.

**S3 only** — a no-op on ``--backend local``. Run it LAST (or against a second
throwaway bucket) since it stresses the bucket. Run from ``scripts/``::

    uv run python -m scale_tests.t7_ramp --run-id dev --backend s3 --bucket my-throwaway
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from scale_tests import harness

logger = logging.getLogger("scale_tests.t7")

TEST = "t7"
OBJECT_BYTES = 8 * 1024 * 1024  # 8 MB, S3 range-GET sweet-spot size
CONCURRENCY_RAMP_TINY = (20, 50)
CONCURRENCY_RAMP_BENCH = (50, 100, 200, 400)
OBJECTS_PER_LEVEL = 400


def _put_many(bucket: str, prefix: str, concurrency: int, n_objects: int) -> tuple[int, float]:
    """PUT ``n_objects`` at the given concurrency; return (503 count, wall_s)."""
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    # botocore's default pool is 10 connections; without this the high ramp
    # levels would serialize on the pool and never actually hit S3 at the
    # requested concurrency.
    client = boto3.client("s3", config=Config(max_pool_connections=max(10, concurrency)))
    body = b"\0" * OBJECT_BYTES
    slowdowns = 0

    def put(i: int) -> int:
        try:
            client.put_object(Bucket=bucket, Key=f"{prefix}/obj_{concurrency}_{i}", Body=body)
            return 0
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("SlowDown", "503", "ServiceUnavailable"):
                return 1
            # Anything else (AccessDenied, NoSuchBucket, KMS, ...) is a broken
            # run, not throttling — counting it as a clean PUT would corrupt
            # the ramp evidence with failed-request "throughput".
            raise

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        slowdowns = sum(ex.map(put, range(n_objects)))
    return slowdowns, time.monotonic() - t0


def phase_ramp(cfg: harness.RunConfig) -> None:
    """Ramp PUT concurrency, recording 503 rate and throughput per level."""
    ramp = CONCURRENCY_RAMP_TINY if cfg.is_tiny else CONCURRENCY_RAMP_BENCH
    n = OBJECTS_PER_LEVEL
    for concurrency in ramp:
        # Under the configured store root (which is run_id-scoped), so
        # teardown.py's recursive delete actually removes the ramp objects.
        uri = harness.store_uri(cfg, f"t7_ramp/c{concurrency}")
        bucket, _, prefix = uri.removeprefix("s3://").partition("/")
        slowdowns, wall = _put_many(bucket, prefix, concurrency, n)
        puts_per_s = n / wall if wall > 0 else 0.0
        harness.emit_metric(cfg, TEST, "ramp", "slowdown_503_count", slowdowns, "count", concurrency=concurrency, n=n)
        harness.emit_metric(cfg, TEST, "ramp", "puts_per_s", puts_per_s, "count/s", concurrency=concurrency)
        logger.info("concurrency=%d: %d/%d SlowDown, %.0f PUT/s", concurrency, slowdowns, n, puts_per_s)


def main() -> int:
    """Parse args; run the ramp only on S3, else cleanly skip."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    cfg = harness.config_from_args(parser.parse_args())
    harness.configure_logging()

    if not cfg.is_s3:
        logger.info("T7 is S3-only; nothing to do on --backend local. Skipping.")
        return 0
    harness.run_phase(cfg, TEST, "ramp", lambda: phase_ramp(cfg))
    logger.info("T7 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
