"""CloudWatch plumbing shared by every profiling CLI, for both compute stages.

The ingest tools (``watch_scheduler``, ``ingest_log_queries``) and the inference
tool (``observe_cluster``) all run Logs-Insights queries over a ``--since /
--until`` window. That was three copies of the same query loop and three copies
of the same timestamp parsing; one copy here keeps the contract below honest for
all of them, and the ingest and inference harnesses can no longer drift apart on
what a failed query means.

The Insights runner guarantees:

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
   the cap is reported both on stderr and as a ``truncated`` flag in the JSON,
   which ``report.py`` carries into the dossier.
3. **An abandoned query is cancelled, not orphaned.** Insights allows a limited
   number of concurrent queries per account, and walking away from one at the
   deadline leaves it holding a slot — which then surfaces as
   ``LimitExceededException`` on an unrelated later query, i.e. as guarantee 1
   firing for no visible reason. Repeated probe runs make that cumulative.
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
    # A boto3 ``logs`` client. boto3 generates its clients at runtime, so there is
    # no static type to name; the tests pass a stub with just the two methods used
    # here, which is the point of not pinning a concrete type.
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

    Returns ``(rows, truncated)``. ``rows`` is ``None`` when the query could not
    be answered at all (rejected, timed out, denied, unreachable) — never an
    empty list, so a caller can tell that apart from a query that legitimately
    matched nothing. ``truncated`` is True when the result hit ``limit`` and the
    series is therefore incomplete.

    Bounded poll: breaks on any terminal status and caps total wait so a query
    stuck Scheduled/Running can't spin forever. ``deadline_s`` exists so a test
    can drive the give-up path without patching the clock or sleeping.
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
                # Cancel rather than orphan it: an abandoned query keeps holding
                # one of the account's concurrent-query slots, and the symptom
                # surfaces later as an unrelated query being rejected. Suppressed
                # because a query that reached a terminal state between the poll
                # and here rejects the cancel, which is not worth reporting.
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
