"""Shared CloudWatch Logs Insights runner for the ingest profiling tools.

``watch_scheduler --report`` and ``ingest_log_queries`` both run Insights
queries, and both need the same two guarantees, so the runner lives here rather
than in two copies that can drift apart:

1. **A failed query returns ``None``, it doesn't abort the command.** Insights
   rejects a query for reasons that are per-query and recoverable-by-skipping
   (``MalformedQueryException``), or environmental (``LimitExceededException``
   when too many concurrent queries are running, throttling, denied
   permissions). Letting one of those propagate would discard every row already
   collected in the same run — the dossier's whole point is to gather what IS
   available, and "couldn't ask" (``rows: null``) is distinguishable from
   "asked, nothing matched" (``rows: []``).
2. **Truncation is never silent.** Insights caps a result set, so a wide window
   or a prefix spanning several streams can return a partial series. Presenting
   that as a full-run profile would hide the actual peak or saturation onset, so
   the cap is reported both on stderr and as a ``truncated`` flag in the JSON.
"""

from __future__ import annotations

import sys
import time

#: Insights' own per-query result ceiling.
INSIGHTS_MAX_RESULTS = 10_000

#: How long to wait for one query to reach a terminal state.
POLL_DEADLINE_S = 180

#: Insights statuses that mean the query has stopped moving.
_TERMINAL = ("Complete", "Failed", "Cancelled", "Timeout", "Unknown")


def insights_query(
    logs: object,
    log_group: str,
    query: str,
    start_epoch: int,
    end_epoch: int,
    *,
    limit: int = INSIGHTS_MAX_RESULTS,
) -> tuple[list[dict] | None, bool]:
    """Run one Insights query.

    Returns ``(rows, truncated)``. ``rows`` is ``None`` when the query could not
    be answered at all (rejected, timed out, denied, unreachable) — never an
    empty list, so a caller can tell that apart from a query that legitimately
    matched nothing. ``truncated`` is True when the result hit ``limit`` and the
    series is therefore incomplete.

    Bounded poll: breaks on any terminal status and caps total wait so a query
    stuck Scheduled/Running can't spin forever.
    """
    try:
        qid = logs.start_query(
            logGroupName=log_group,
            startTime=start_epoch,
            endTime=end_epoch,
            queryString=query,
            limit=limit,
        )["queryId"]
        deadline = time.monotonic() + POLL_DEADLINE_S
        res = logs.get_query_results(queryId=qid)
        while res["status"] not in _TERMINAL:
            if time.monotonic() > deadline:
                print(f"CloudWatch query did not finish within {POLL_DEADLINE_S}s", file=sys.stderr)
                return None, False
            time.sleep(2)
            res = logs.get_query_results(queryId=qid)
        if res["status"] != "Complete":
            print(f"CloudWatch query {res['status']}", file=sys.stderr)
            return None, False
        rows = [{c["field"]: c["value"] for c in row} for row in res["results"]]
    except Exception as e:
        # Broad by design: boto3 generates its client exceptions dynamically
        # (logs.exceptions.MalformedQueryException et al) and botocore adds
        # credential/endpoint errors, so there is no useful narrow tuple. The
        # documented contract is that one bad query yields rows=null and the
        # command still emits everything else it gathered.
        print(f"CloudWatch query failed: {e}", file=sys.stderr)
        return None, False

    truncated = len(rows) >= limit
    if truncated:
        print(
            f"WARNING: query hit the {limit}-row Insights cap — the result is PARTIAL. "
            "Narrow --since/--until or the stream prefix; peaks and onsets may be missing.",
            file=sys.stderr,
        )
    return rows, truncated
