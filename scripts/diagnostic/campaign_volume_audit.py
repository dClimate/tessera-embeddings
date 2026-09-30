"""Measure how many bytes a campaign actually wrote, from S3 Inventory via Athena.

Peak resident storage is what the bill charges for. It is not what the campaign moved: a
mosaic is deleted as soon as inference has consumed it, so a bucket that peaked at 10 PiB
can have carried 14. This counts every object the bucket ever held, exactly once.

It needs daily S3 Inventory with ``Size`` and ``LastModifiedDate`` in Parquet, and it is
only a total-since-the-beginning if the bucket was empty when the inventory started ---
``--check-empty-before`` verifies that against CloudWatch rather than assuming it.

Run via::

    uv run python scripts/diagnostic/campaign_volume_audit.py
        --inventory s3://dest-bucket/prefix/source-bucket/config-id/hive/
        --results s3://scratch-bucket/athena/ --bucket source-bucket
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta

import boto3

DATABASE = "tessera_volume_audit"
PB = 10**15
PIB = 2**50


def _run(athena: boto3.client, sql: str, results: str, *, database: str = DATABASE) -> list[list[str]]:
    """Run one Athena statement and return its rows, header first."""
    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": results},
        # Athena rejects an empty context, so the CREATE DATABASE call omits it entirely.
        **({"QueryExecutionContext": {"Database": database}} if database else {}),
    )["QueryExecutionId"]
    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if status["State"] == "SUCCEEDED":
            break
        if status["State"] in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"{status['State']}: {status.get('StateChangeReason')}")
        time.sleep(4)
    rows, token = [], None
    while True:
        page = athena.get_query_results(QueryExecutionId=qid, **({"NextToken": token} if token else {}))
        rows += [[c.get("VarCharValue", "") for c in r["Data"]] for r in page["ResultSet"]["Rows"]]
        token = page.get("NextToken")
        if not token:
            return rows


def _resident_before(bucket: str, when: datetime, region: str) -> float | None:
    """The most the bucket held on any day of the week before `when`, or None if unknown.

    Exactly zero is the wrong test: a bucket carrying a few code tarballs is empty for
    this purpose. The caller compares this against the volume it measured.
    """
    points = boto3.client("cloudwatch", region_name=region).get_metric_statistics(
        Namespace="AWS/S3",
        MetricName="BucketSizeBytes",
        Dimensions=[
            {"Name": "BucketName", "Value": bucket},
            {"Name": "StorageType", "Value": "StandardStorage"},
        ],
        StartTime=when - timedelta(days=7),
        EndTime=when,
        Period=86400,
        Statistics=["Maximum"],
    )["Datapoints"]
    return max(p["Maximum"] for p in points) if points else None


def main() -> int:
    """Build the table, count the bytes, print the totals and the daily curve."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, help="s3:// URI of the inventory hive/ prefix")
    ap.add_argument("--results", required=True, help="s3:// URI for Athena query results")
    ap.add_argument("--bucket", required=True, help="the bucket the inventory describes")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--table", default="inv")
    ap.add_argument("--key-prefix", default="", help="restrict to keys under this prefix, e.g. mosaics/")
    ap.add_argument(
        "--check-empty-before",
        metavar="YYYY-MM-DD",
        help="confirm the bucket was empty before this date, so the total is a lifetime total",
    )
    args = ap.parse_args()
    # Each Athena stage takes about a minute. Unbuffered, so a watcher can tell a slow query
    # from a hang.
    sys.stdout.reconfigure(line_buffering=True)

    athena = boto3.client("athena", region_name=args.region)
    _run(athena, f"CREATE DATABASE IF NOT EXISTS {DATABASE}", args.results, database="")
    _run(
        athena,
        f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {args.table} (
          bucket string, key string, size bigint,
          last_modified_date timestamp, e_tag string,
          storage_class string, is_multipart_uploaded boolean)
        PARTITIONED BY (dt string)
        ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
        STORED AS INPUTFORMAT 'org.apache.hadoop.hive.ql.io.SymlinkTextInputFormat'
        OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
        LOCATION '{args.inventory}'
        """,
        args.results,
    )
    _run(athena, f"MSCK REPAIR TABLE {args.table}", args.results)

    where = f"WHERE key LIKE '{args.key_prefix}%'" if args.key_prefix else ""
    # One row per object ever seen. `last_modified_date` joins the grouping so that a key
    # overwritten in place counts both writes, which is what "bytes written" means.
    deduped = f"""
        SELECT key, last_modified_date AS lmd, max(size) AS size
        FROM {args.table} {where} GROUP BY key, last_modified_date
    """

    total = _run(
        athena,
        f"SELECT sum(size), count(*), min(lmd), max(lmd) FROM ({deduped})",
        args.results,
    )[1]
    written, objects = int(total[0]), int(total[1])
    print(f"bucket      {args.bucket}{'/' + args.key_prefix if args.key_prefix else ''}")
    print(f"written     {written:,} bytes = {written / PB:.3f} PB = {written / PIB:.3f} PiB")
    print(f"objects     {objects:,}, mean {written / objects / 10**6:.2f} MB")
    print(f"window      {total[2]} -> {total[3]}")

    if args.check_empty_before:
        cutoff = datetime.fromisoformat(args.check_empty_before).replace(tzinfo=UTC)
        before = _resident_before(args.bucket, cutoff, args.region)
        if before is None:
            print(f"lifetime    UNKNOWN - no bucket metrics before {cutoff:%Y-%m-%d}")
        else:
            share = before / written
            verdict = (
                "negligible, so the figure above is a lifetime total"
                if share <= 0.001
                else f"{share:.1%} of the total, so the figure above is a FLOOR"
            )
            print(f"lifetime    {before:,.0f} B resident before {cutoff:%Y-%m-%d} - {verdict}")

    # The cross-check. Reading, for each clock hour, the most any one snapshot ever saw
    # written in that hour is cheap and cannot be fooled by inventory reporting lag; the
    # dedupe above is exact but pays to scan every key. They should agree.
    cross = _run(
        athena,
        f"""
        WITH per_snap AS (
          SELECT dt, date_trunc('hour', last_modified_date) AS wh, sum(size) AS b
          FROM {args.table} {where} GROUP BY dt, date_trunc('hour', last_modified_date))
        SELECT sum(b) FROM (SELECT wh, max(b) AS b FROM per_snap GROUP BY wh)
        """,
        args.results,
    )[1]
    hourly = int(cross[0])
    print(
        f"cross-check {hourly / PB:.3f} PB by hourly maximum, {abs(hourly - written) / written:.2%} apart", flush=True
    )

    print("\nday            objects            PB")
    for day, n, b in _run(
        athena,
        f"SELECT date(lmd), count(*), sum(size) FROM ({deduped}) GROUP BY date(lmd) ORDER BY 1",
        args.results,
    )[1:]:
        print(f"{day}  {int(n):>15,}  {int(b) / PB:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
