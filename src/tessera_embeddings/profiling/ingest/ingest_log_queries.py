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
  429 throttling and 503s from the catalog both surface here, attributable to
  CMR vs earth-search/element84 by service and host.
- **Store-write retries** — both ``ingest/s1_roi.py`` and ``ingest/s2_roi.py``
  wrap ``write_dataset`` in tenacity with ``before_sleep_log``
  (``Retrying ... as it raised ...``), so this measures icechunk/GDAL write
  pressure across both sensors.
- **S3 SlowDown** — 503s from the object store, keyed on the error code and
  excluding catalog retry lines (which also carry ``HTTP 503``).
- **Worker lifecycle** — distributed/nanny exit/kill/restart/removal counts by
  category, to separate "workers dying" from "scheduler falling behind".

Known gap: **ASF granule downloads are not separately instrumented.** Nothing in
the download path emits a stable retry marker today, so external download
throttling is not directly observable here — it would show up indirectly as
worker errors or stalled batches. Adding a marker to the download path is a
follow-up, not something this query pack can infer.

Usage::

    export AWS_PROFILE=...              # or pass --profile on each call
    te-ingest-log-queries --since 2026-07-24T18:00 --until 2026-07-24T20:30  # all -> JSON
    te-ingest-log-queries ... --query http_retries_by_service
    te-ingest-log-queries --list    # names + descriptions

Only needs CloudWatch Logs read access, resolved from the ambient AWS credential
chain unless ``--profile`` names one. The log group is shared across ingest runs
— scope the window tightly to the run of interest. A query that cannot be
answered records ``rows: null`` and never aborts the others.
"""

from __future__ import annotations

import argparse
import json

import boto3

from tessera_embeddings.profiling._cloudwatch import insights_query, iso, parse_ts
from tessera_embeddings.profiling.ingest import DEFAULT_INGEST_LOG_GROUP

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
        "Catalog HTTP retries broken down by service and status code (429 throttle vs 503 SlowDown vs 5xx).",
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
        r" | stats count(*) as retries by bin(5m) as period"
        r" | sort period asc",
    ),
    "store_write_retries": (
        "Mosaic store-write retries per 5-minute bin. BOTH s1_roi.py and "
        "s2_roi.py wrap write_dataset in tenacity with before_sleep_log, so this "
        "counts icechunk/GDAL write failures under parallelism across both "
        "sensors — write pressure, NOT catalog or granule-download trouble.",
        r"filter @message like /Retrying/ and @message like /as it raised/"
        r" | stats count(*) as retries by bin(5m) as period"
        r" | sort period asc",
    ),
    "s3_slowdown": (
        "S3 503 SlowDown per 5-minute bin (object-store push-back on mosaic/store "
        "writes). Keyed on the S3 error code alone, with catalog retry lines "
        "excluded: CMR/STAC retries also log 'HTTP 503', and counting those here "
        "would blame the object store when the catalog is at fault (those live in "
        "http_retry_status).",
        r"filter @message like /SlowDown/ and @message not like /retry:/"
        r" | stats count(*) as slowdowns by bin(5m) as period"
        r" | sort period asc",
    ),
    "worker_lifecycle_counts": (
        "Worker lifecycle counts by CATEGORY — separates 'workers dying' "
        "(exit/kill/restart/removal) from a scheduler that is merely behind. "
        "Returns one row of fixed columns.",
        # Counted with per-category boolean flags summed into fixed columns rather
        # than `parse ... | stats by event`: the lifecycle markers embed variable
        # text (a PID in "Worker process 1234 exited", an address in "Nanny at
        # 'tcp://...' restarting"), so capturing the match would group by that
        # text and yield one bucket per worker — turning a restart storm into
        # thousands of rows of 1 instead of a single visible count.
        r"fields"
        r" (@message like /Worker process .* exited/) as f_proc_exited,"
        r" (@message like /Killed worker/) as f_killed,"
        r" (@message like /worker failed/) as f_failed,"
        r" (@message like /Unexpected worker/) as f_unexpected,"
        r" (@message like /Remov(e|ing) worker/) as f_removed,"
        r" (@message like /Regist(er|ered) worker/) as f_registered,"
        r" (@message like /Nanny.*(restart|closing)/) as f_nanny,"
        r" (@message like /lost all workers/) as f_lost_all"
        r" | stats sum(f_proc_exited) as worker_process_exited, sum(f_killed) as killed_worker,"
        r" sum(f_failed) as worker_failed, sum(f_unexpected) as unexpected_worker,"
        r" sum(f_removed) as worker_removed, sum(f_registered) as worker_registered,"
        r" sum(f_nanny) as nanny_restart_or_close, sum(f_lost_all) as lost_all_workers",
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
        r" | stats count(*) as errors by bin(5m) as period"
        r" | sort period asc",
    ),
}


def run_queries(
    session: boto3.session.Session,
    log_group: str,
    start_epoch: int,
    end_epoch: int,
    names: list[str],
) -> dict:
    """Run the named queries and return a JSON-able dict keyed by query name.

    A query that fails hard (rejected/denied/timed out) records ``rows: null``; a
    query that simply matched nothing records ``rows: []`` — the caller can tell
    "couldn't ask" from "asked, nothing there". One failing query never aborts the
    rest: the whole point is to collect whatever IS available for the dossier, so
    every query is attempted and reported independently (see ``_insights``).
    ``truncated`` flags a result that hit the Insights row cap.
    """
    logs = session.client("logs")
    out: dict = {
        "log_group": log_group,
        "window": {"start": iso(start_epoch), "end": iso(end_epoch)},
        "queries": {},
    }
    for name in names:
        description, query = QUERIES[name]
        rows, truncated = insights_query(logs, log_group, query, start_epoch, end_epoch)
        out["queries"][name] = {"description": description, "rows": rows, "truncated": truncated}
    return out


def main(argv: list[str] | None = None) -> int:
    """Parse args and run the requested Insights queries (or --list)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--profile", default=None, help="AWS profile; default uses the ambient credential chain / AWS_PROFILE"
    )
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
    result = run_queries(session, args.log_group, parse_ts(args.since), parse_ts(args.until), names)
    print(json.dumps(result, indent=2))
    # Nonzero if every requested query failed hard (never mask a broken run as clean).
    if all(q["rows"] is None for q in result["queries"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
