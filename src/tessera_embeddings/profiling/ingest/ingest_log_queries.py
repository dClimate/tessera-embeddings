"""Run-scoped CloudWatch Logs Insights queries over an ingest run.

The ``watch_scheduler`` tool answers "is the *scheduler* saturated?". This tool
answers the other half — "is the bottleneck the scheduler, or something
external?" — by grepping across the thousands of Dask worker streams at once.
Each named query is parameterized only by the run's time window + log group
(nothing hard-coded per run) and returns JSON rows, so the same pack works for
any ingest on any account.

The patterns key off real log markers in the ingest path:

- **HTTP catalog retries** — ``tessera_embeddings.ingest._http`` logs every
  retry as ``<service> retry: <method> <url> — HTTP <status>`` (service is
  ``CMR`` or ``STAC``); ``status_forcelist`` is (429, 500, 502, 503, 504), so
  429 throttling and 503 SlowDown both surface here, attributable to CMR vs
  earth-search/element84 by service and host.
- **S1 granule-download retries** — ``ingest/s1_roi.py`` retries ASF downloads
  via tenacity ``before_sleep_log`` → ``Retrying ... as it raised ...``.
- **S3 SlowDown** — 503s from the object store (icechunk / botocore).
- **Worker lifecycle** — distributed/nanny worker exit/restart/removal lines,
  with reasons, to separate "workers dying" from "scheduler falling behind".

Usage::

    te-ingest-log-queries --profile global-tessera-dev \
        --since 2026-07-24T18:00 --until 2026-07-24T20:30            # all queries -> JSON
    te-ingest-log-queries ... --query http_retries_by_service
    te-ingest-log-queries --list    # names + descriptions

Only needs CloudWatch Logs read access. The log group is shared across ingest
runs — scope the window tightly to the run of interest.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time

import boto3

DEFAULT_INGEST_LOG_GROUP = "/ecs/tessera/dask"

# name -> (description, Insights query string). The run window + log group are
# passed to start_query separately, so the query strings carry no per-run state.
QUERIES: dict[str, tuple[str, str]] = {
    "http_retries_by_service": (
        "Catalog HTTP retry count by service (CMR vs STAC/earth-search) — the "
        "primary external-throttling discriminator.",
        r"fields @message"
        r" | filter @message like /retry:/"
        r" | parse @message /(?<service>\S+) retry:/"
        r" | stats count(*) as retries by service"
        r" | sort retries desc",
    ),
    "http_retry_status": (
        "Catalog HTTP retries broken down by service and status code (429 "
        "throttle vs 503 SlowDown vs 5xx).",
        r"fields @message"
        r" | filter @message like /retry:/ and @message like /HTTP/"
        r" | parse @message /(?<service>\S+) retry:.* HTTP (?<status>\d+)/"
        r" | stats count(*) as n by service, status"
        r" | sort n desc",
    ),
    "http_retries_over_time": (
        "Catalog retry storms: retries per 5-minute bin (rising bins = an "
        "external API pushing back harder over the run).",
        r"filter @message like /retry:/"
        r" | stats count(*) as retries by bin(5m)"
        r" | sort bin(5m) asc",
    ),
    "s1_download_retries": (
        "S1 (ASF) granule-download retries per 5-minute bin (tenacity "
        "before_sleep_log: 'Retrying ... as it raised ...').",
        r"filter @message like /Retrying/ and @message like /as it raised/"
        r" | stats count(*) as retries by bin(5m)"
        r" | sort bin(5m) asc",
    ),
    "s3_slowdown": (
        "S3 503 SlowDown occurrences per 5-minute bin (object-store push-back "
        "on mosaic/store writes).",
        r"filter @message like /SlowDown/ or @message like /HTTP 503/"
        r" | stats count(*) as slowdowns by bin(5m)"
        r" | sort bin(5m) asc",
    ),
    "worker_lifecycle_counts": (
        "Worker lifecycle event counts by type — separates 'workers dying' "
        "(exit/restart/removed) from a scheduler that is merely behind.",
        r"fields @message"
        r" | filter @message like /(Worker process .* exited|Remove worker|Removing worker"
        r"|Register worker|Registered worker|Nanny .* (restart|closing)|lost all workers"
        r"|worker failed|Killed worker|Unexpected worker)/"
        r" | parse @message /(?<event>Worker process .* exited|Remove worker|Removing worker"
        r"|Register worker|Registered worker|Nanny.*restart|Nanny.*closing|lost all workers"
        r"|worker failed|Killed worker|Unexpected worker)/"
        r" | stats count(*) as n by event"
        r" | sort n desc",
    ),
    "worker_exit_reasons": (
        "Raw recent worker exit/kill/removal lines with reasons and timestamps "
        "(the forensic view behind the lifecycle counts).",
        r"fields @timestamp, @logStream, @message"
        r" | filter @message like /(Worker process .* exited|Killed worker|worker failed"
        r"|Unexpected worker|Removing worker)/"
        r" | sort @timestamp desc"
        r" | limit 200",
    ),
    "error_rate_over_time": (
        "General worker error/exception rate per 5-minute bin — a catch-all so "
        "a novel failure mode still shows up as a rising error curve.",
        r"filter @message like /ERROR/ or @message like /Traceback/ or @message like /Exception/"
        r" | stats count(*) as errors by bin(5m)"
        r" | sort bin(5m) asc",
    ),
}


def _parse_ts(s: str) -> int:
    """Parse an ISO8601 UTC timestamp (or 'YYYY-MM-DDTHH:MM') to epoch seconds."""
    dt = datetime.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return int(dt.timestamp())


def _insights_query(logs: object, log_group: str, query: str, start_epoch: int, end_epoch: int) -> list[dict] | None:
    """Run one CloudWatch Insights query and return its rows (None on failure).

    Bounded poll, mirroring src/tessera_embeddings/profiling/inference/observe_cluster.py: break
    on any terminal status and cap total wait so a stuck query can't spin forever.
    """
    qid = logs.start_query(
        logGroupName=log_group, startTime=start_epoch, endTime=end_epoch, queryString=query, limit=10000
    )["queryId"]
    deadline = time.monotonic() + 180
    res = logs.get_query_results(queryId=qid)
    while res["status"] not in ("Complete", "Failed", "Cancelled", "Timeout", "Unknown"):
        if time.monotonic() > deadline:
            print("CloudWatch query did not finish within 180s", file=sys.stderr)
            return None
        time.sleep(2)
        res = logs.get_query_results(queryId=qid)
    if res["status"] != "Complete":
        print(f"CloudWatch query {res['status']}", file=sys.stderr)
        return None
    return [{c["field"]: c["value"] for c in row} for row in res["results"]]


def run_queries(
    session: boto3.session.Session,
    log_group: str,
    start_epoch: int,
    end_epoch: int,
    names: list[str],
) -> dict:
    """Run the named queries and return a JSON-able dict keyed by query name.

    A query that fails hard (permission/timeout) records ``rows: null``; a
    query that simply matched nothing records ``rows: []`` — the caller can
    tell "couldn't ask" from "asked, nothing there".
    """
    logs = session.client("logs")
    out: dict = {
        "log_group": log_group,
        "window": {"start": _iso(start_epoch), "end": _iso(end_epoch)},
        "queries": {},
    }
    for name in names:
        description, query = QUERIES[name]
        rows = _insights_query(logs, log_group, query, start_epoch, end_epoch)
        out["queries"][name] = {"description": description, "rows": rows}
    return out


def _iso(epoch: float) -> str:
    return f"{datetime.datetime.fromtimestamp(epoch, datetime.UTC):%Y-%m-%dT%H:%M:%SZ}"


def main(argv: list[str] | None = None) -> int:
    """Parse args and run the requested Insights queries (or --list)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="global-tessera-dev", help="AWS profile (default: %(default)s)")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--log-group", default=DEFAULT_INGEST_LOG_GROUP)
    parser.add_argument("--since", help="run start (ISO8601 UTC)")
    parser.add_argument("--until", help="run end (ISO8601 UTC)")
    parser.add_argument("--query", action="append", choices=sorted(QUERIES), help="run only this query (repeatable)")
    parser.add_argument("--list", action="store_true", help="list query names + descriptions and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, (desc, _) in QUERIES.items():
            print(f"{name}\n    {desc}")
        return 0

    if not args.since or not args.until:
        parser.error("--since and --until are required (unless --list)")

    names = args.query or list(QUERIES)
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    result = run_queries(session, args.log_group, _parse_ts(args.since), _parse_ts(args.until), names)
    print(json.dumps(result, indent=2))
    # Nonzero if every requested query failed hard (never mask a broken run as clean).
    if all(q["rows"] is None for q in result["queries"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
