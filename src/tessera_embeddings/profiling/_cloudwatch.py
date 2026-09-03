"""CloudWatch plumbing shared by every profiling CLI, for both compute stages.

The ingest tools (``watch_scheduler``, ``ingest_log_queries``) and the inference tool
(``observe_cluster``) all run Logs-Insights queries over a ``--since / --until`` window.
One copy of the query loop and the timestamp parsing keeps the two harnesses from drifting
apart on what a failed query means.

The Insights runner guarantees:

1. **A failed query returns ``None``, it doesn't abort the command.** Rejections are
   per-query (``MalformedQueryException``) or environmental (``LimitExceededException``,
   throttling, denied permissions); propagating one would discard every row already
   collected in the same run. The dossier's point is to gather what IS available, so
   "couldn't ask" (``rows: null``) stays distinguishable from "asked, nothing matched"
   (``rows: []``).
2. **Truncation is never silent.** Insights caps a result set, so a wide window or a
   multi-stream prefix can return a partial series that would hide the real peak or
   saturation onset. The cap is reported on stderr and as a ``truncated`` flag in the JSON,
   which ``report.py`` carries into the dossier.
3. **An abandoned query is cancelled, not orphaned.** Concurrent queries per account are
   limited, and walking away from one at the deadline leaves it holding a slot — which
   surfaces later as ``LimitExceededException`` on an unrelated query, i.e. guarantee 1
   firing for no visible reason, cumulatively across repeated probe runs.
"""

from __future__ import annotations

import contextlib
import datetime
import sys
import time
from typing import Any

#: Insights' own per-query result ceiling.
INSIGHTS_MAX_RESULTS = 10_000

#: How long to wait for one query to reach a terminal state.
POLL_DEADLINE_S = 180

#: Insights statuses that mean the query has stopped moving.
_TERMINAL = ("Complete", "Failed", "Cancelled", "Timeout", "Unknown")


def parse_ts(s: str) -> int:
    """Parse an ISO8601 UTC timestamp (or 'YYYY-MM-DDTHH:MM') to epoch seconds.

    A naive timestamp is read as UTC, because every window these tools take is a
    CloudWatch window and CloudWatch is UTC.
    """
    dt = datetime.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return int(dt.timestamp())


def iso(epoch: float) -> str:
    """Render epoch seconds as a UTC ISO8601 timestamp."""
    return f"{datetime.datetime.fromtimestamp(epoch, datetime.UTC):%Y-%m-%dT%H:%M:%SZ}"


def insights_query(
    # A boto3 ``logs`` client: generated at runtime, so there is no static type to name —
    # which is what lets the tests pass a stub with only the two methods used here.
    logs: Any,  # noqa: ANN401 — boto3 client, no static type exists
    log_group: str,
    query: str,
    start_epoch: int,
    end_epoch: int,
    *,
    limit: int = INSIGHTS_MAX_RESULTS,
    deadline_s: float = POLL_DEADLINE_S,
) -> tuple[list[dict] | None, bool]:
    """Run one Insights query.

    Returns ``(rows, truncated)``. ``rows`` is ``None`` when the query could not be answered
    at all (rejected, timed out, denied, unreachable) — never an empty list, so a caller can
    tell that apart from a query that legitimately matched nothing. ``truncated`` is True
    when the result hit ``limit`` and the series is therefore incomplete.

    Bounded poll: breaks on any terminal status and caps total wait so a query stuck
    Scheduled/Running can't spin forever. ``deadline_s`` exists so a test can drive the
    give-up path without patching the clock or sleeping.
    """
    try:
        qid = logs.start_query(
            logGroupName=log_group,
            startTime=start_epoch,
            endTime=end_epoch,
            queryString=query,
            limit=limit,
        )["queryId"]
        deadline = time.monotonic() + deadline_s
        res = logs.get_query_results(queryId=qid)
        while res["status"] not in _TERMINAL:
            if time.monotonic() > deadline:
                print(f"CloudWatch query did not finish within {deadline_s}s", file=sys.stderr)
                # Cancel rather than orphan it (guarantee 3). Suppressed because a query
                # that went terminal between the poll and here rejects the cancel.
                with contextlib.suppress(Exception):
                    logs.stop_query(queryId=qid)
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
        # credential/endpoint errors, so there is no useful narrow tuple. Guarantee 1.
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
