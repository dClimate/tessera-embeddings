"""Watch and profile the Dask *scheduler* during an ingest run.

The ingest scheduler is a single event-loop process that builds every task
graph and routes every task; at UTM scale it — not the workers — is the named
saturation risk. :class:`SchedulerResourceLogger`
(``tessera_embeddings.providers.aws.dask``) already ships a 30 s ``scheduler
health:`` heartbeat to the ``dask-scheduler`` CloudWatch stream. This tool turns
that heartbeat into something a machine can act on:

1. ``--live`` — tail the scheduler stream in near real time, parse each health
   line into a rolling window, derive rates (task-count change, backlog slope,
   worker churn), evaluate saturation thresholds, and emit one machine-readable
   JSON snapshot per line to **stdout** (with ALERT objects when a threshold
   trips). A compact human line goes to **stderr**, so stdout stays a clean
   JSON stream an agent can consume live.
2. ``--report --since --until`` — post-hoc full-run profile from CloudWatch: the
   complete metric time series, per-metric peaks, saturation-onset timestamps,
   a simple cpu/backlog co-occurrence summary, and derived worker join/exit
   events — emitted as one JSON object to stdout (consumed by ``report.py``),
   with ``--markdown`` adding a paste-ready section to stderr.

Both modes only need CloudWatch Logs read access, resolved from the ambient AWS
credential chain (``AWS_PROFILE``, instance role, …) unless ``--profile`` names
one. Scope ``--report`` tightly: the log group is shared across ingest runs, so
an overlapping run's scheduler stream lands in the same group (disambiguate with
``--stream-prefix``).

Usage::

    export AWS_PROFILE=...              # or pass --profile on each call
    # live, during a run (Ctrl-C to stop)
    te-watch-scheduler --live
    # post-hoc profile of a finished run
    te-watch-scheduler --report --since 2026-07-24T18:00 --until 2026-07-24T20:30 --markdown
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass

import boto3

from tessera_embeddings.profiling._cloudwatch import insights_query, iso, parse_ts
from tessera_embeddings.profiling.ingest import DEFAULT_INGEST_LOG_GROUP

# dask-cloudprovider names streams ``<prefix>/<container>/<task-id>``;
# ``ecs_cluster`` sets prefix "dask" and the scheduler container is
# "dask-scheduler" (see providers/aws/dask.py), so scheduler health lines land
# under "dask/dask-scheduler/...".
DEFAULT_SCHEDULER_STREAM_PREFIX = "dask/dask-scheduler"

# The substring CloudWatch filters on server-side, and the marker the parser
# keys off — kept identical to the logger's format string.
HEALTH_MARKER = "scheduler health:"

# How often --live polls CloudWatch. The heartbeat itself is every 30 s, so a
# 20 s poll keeps latency under one heartbeat without hammering the API.
DEFAULT_POLL_INTERVAL_S = 20.0

# How far back --live rewinds its cursor each poll, so a heartbeat that CloudWatch
# surfaces out of order (common when the prefix spans several scheduler streams)
# is still picked up instead of being skipped forever. eventId dedupe makes the
# replay free; two minutes covers ingestion lag at a 30 s heartbeat.
REPLAY_OVERLAP_MS = 120_000


@dataclass(frozen=True)
class Thresholds:
    """Saturation rules evaluated against the rolling window.

    Defaults encode the operating envelope we care about; ``--cpu-threshold``
    etc. override them (the smoke test forces them low to prove alerts fire).
    """

    # Sustained scheduler CPU: the GIL-bound loop pinned near 100% precedes
    # event-loop stalls. Trip when >= cpu_pct for cpu_intervals consecutive
    # samples.
    cpu_pct: float = 90.0
    cpu_intervals: int = 3
    # Process RSS as a percent of the container limit: the OOM predictor.
    mem_pct: float = 80.0
    # Backlog (no-worker / unrunnable) rising for backlog_intervals consecutive
    # samples: the scheduler is falling behind assigning work.
    backlog_intervals: int = 3
    # Event-loop lag (seconds) in a single sample: the direct stall signal.
    lag_s: float = 5.0
    # Worker-count change between two samples that counts as a churn spike.
    churn_delta: int = 25


# One health line, e.g.:
#   scheduler health: cpu=100% rss=1.50GiB mem=18% lag=5.0s fds=42 threads=9 \
#   workers=2 tasks=42 processing=8 no-worker=7
# mem may be "nan" when the container limit is unknown; counts may be -1 when
# the scheduler ref is briefly unavailable. Parse defensively.
_NUM = r"-?\d+(?:\.\d+)?"
_HEALTH_RE = re.compile(
    rf"scheduler health: cpu=(?P<cpu>{_NUM})% rss=(?P<rss>{_NUM})GiB "
    rf"mem=(?P<mem>nan|{_NUM})% lag=(?P<lag>{_NUM})s "
    rf"fds=(?P<fds>-?\d+) threads=(?P<threads>-?\d+) workers=(?P<workers>-?\d+) "
    rf"tasks=(?P<tasks>-?\d+) processing=(?P<processing>-?\d+) no-worker=(?P<no_worker>-?\d+)"
)


def parse_health_line(message: str) -> dict | None:
    """Parse a ``scheduler health:`` log message into a metrics dict, or None.

    Returns None for any line that isn't a well-formed health line, so callers
    can feed raw CloudWatch messages in and drop non-matches silently.

    ``mem`` is ``None`` when the scheduler could not read its container limit (it
    logs ``mem=nan%``). Deliberately None rather than ``float("nan")``: these
    dicts go through ``json.dumps``, which renders NaN as the bare token ``NaN``
    — not valid JSON — so a strict consumer of the advertised JSON/JSONL output
    would break in exactly the unknown-memory case. ``None`` serializes to null.
    """
    m = _HEALTH_RE.search(message)
    if not m:
        return None
    g = m.groupdict()
    return {
        "cpu": float(g["cpu"]),
        "rss_gib": float(g["rss"]),
        "mem": None if g["mem"] == "nan" else float(g["mem"]),
        "lag": float(g["lag"]),
        "fds": int(g["fds"]),
        "threads": int(g["threads"]),
        "workers": int(g["workers"]),
        "tasks": int(g["tasks"]),
        "processing": int(g["processing"]),
        "no_worker": int(g["no_worker"]),
    }


def _rate(cur: float, prev: float, dt_s: float) -> float | None:
    """Per-second change between two samples, or None if the interval is 0."""
    if dt_s <= 0:
        return None
    return (cur - prev) / dt_s


def evaluate_alerts(window: list[dict], thresholds: Thresholds) -> list[str]:
    """Return the alert keys tripped by the newest sample in ``window``.

    ``window`` is time-ordered; the last element is the sample under test.
    Sustained rules look back over the tail of the window, so callers should
    keep at least ``max(cpu_intervals, backlog_intervals)`` samples.
    """
    if not window:
        return []
    latest = window[-1]
    alerts: list[str] = []

    # cpu sustained: last cpu_intervals samples all >= cpu_pct.
    tail = window[-thresholds.cpu_intervals :]
    if len(tail) >= thresholds.cpu_intervals and all(s["cpu"] >= thresholds.cpu_pct for s in tail):
        alerts.append("cpu-sustained")

    # memory: single sample over the limit fraction (an unknown limit never trips).
    if latest["mem"] is not None and latest["mem"] >= thresholds.mem_pct:
        alerts.append("mem-high")

    # backlog: no-worker strictly rising over the last backlog_intervals steps.
    btail = window[-(thresholds.backlog_intervals + 1) :]
    if len(btail) >= thresholds.backlog_intervals + 1 and all(
        b["no_worker"] > a["no_worker"] for a, b in itertools.pairwise(btail)
    ):
        alerts.append("backlog-growth")

    # event-loop lag: direct stall signal, single sample.
    if latest["lag"] >= thresholds.lag_s:
        alerts.append("loop-lag")

    # worker churn: |Δworkers| between the last two samples over the delta.
    if len(window) >= 2 and abs(latest["workers"] - window[-2]["workers"]) >= thresholds.churn_delta:
        alerts.append("worker-churn")

    return alerts


def _snapshot(sample: dict, epoch: int, prev: dict | None, prev_epoch: int | None, alerts: list[str]) -> dict:
    """Assemble the per-interval JSON snapshot emitted on stdout."""
    snap: dict = {"ts": iso(epoch), "epoch": epoch, **sample, "alerts": alerts}
    if prev is not None and prev_epoch is not None:
        dt = epoch - prev_epoch
        snap["d_tasks_per_s"] = _rate(sample["tasks"], prev["tasks"], dt)
        snap["d_no_worker_per_s"] = _rate(sample["no_worker"], prev["no_worker"], dt)
        snap["d_workers_per_s"] = _rate(sample["workers"], prev["workers"], dt)
    return snap


def _human_line(snap: dict) -> str:
    mem = "?" if snap["mem"] is None else f"{snap['mem']:.0f}"
    return (
        f"{snap['ts']}  cpu={snap['cpu']:.0f}% mem={mem}% lag={snap['lag']:.1f}s "
        f"workers={snap['workers']} tasks={snap['tasks']} "
        f"proc={snap['processing']} no-worker={snap['no_worker']}"
        + (f"  ALERT[{','.join(snap['alerts'])}]" if snap["alerts"] else "")
    )


def watch_live(
    session: boto3.session.Session,
    log_group: str,
    stream_prefix: str,
    poll_interval_s: float,
    thresholds: Thresholds,
    lookback_s: int = 300,
) -> int:
    """Tail scheduler health lines and emit JSON snapshots until interrupted.

    Uses ``filter_log_events`` (not Insights) for low-latency incremental
    tailing. stdout = one JSON object per health line; stderr = a human line and
    ALERT banners.

    The cursor is advanced to ``newest seen - REPLAY_OVERLAP_MS`` rather than
    past the newest event: when the prefix spans several scheduler streams,
    CloudWatch can make a line from one stream visible only after a
    later-timestamped line from another has already been consumed, and a cursor
    parked past the newest timestamp would skip that heartbeat forever (a missed
    sample can mean a missed alert). Re-fetching a small overlap each poll is
    harmless because ``seen`` dedupes on eventId.
    """
    logs = session.client("logs")
    start_ms = int((time.time() - lookback_s) * 1000)
    seen: set[str] = set()
    # Bounded to what the longest sustained rule looks back over, +1 for the
    # rising-backlog comparison and +1 of slack. The deque length IS the window
    # every rule sees, so no caller-side slicing is needed.
    window: deque[dict] = deque(maxlen=max(thresholds.cpu_intervals, thresholds.backlog_intervals) + 2)
    prev: dict | None = None
    prev_epoch: int | None = None

    print(f"# watching {log_group} [{stream_prefix}] — Ctrl-C to stop", file=sys.stderr)
    try:
        while True:
            kwargs = {
                "logGroupName": log_group,
                "logStreamNamePrefix": stream_prefix,
                "startTime": start_ms,
                "filterPattern": f'"{HEALTH_MARKER}"',
            }
            events: list[dict] = []
            while True:
                resp = logs.filter_log_events(**kwargs)
                events.extend(resp.get("events", []))
                token = resp.get("nextToken")
                if not token:
                    break
                kwargs["nextToken"] = token
            for ev in sorted(events, key=lambda e: e["timestamp"]):
                if ev["eventId"] in seen:
                    continue
                seen.add(ev["eventId"])
                sample = parse_health_line(ev["message"])
                if sample is None:
                    continue
                epoch = ev["timestamp"] // 1000
                window.append(sample)
                alerts = evaluate_alerts(list(window), thresholds)
                snap = _snapshot(sample, epoch, prev, prev_epoch, alerts)
                print(json.dumps(snap), flush=True)
                print(_human_line(snap), file=sys.stderr, flush=True)
                prev, prev_epoch = sample, epoch
                start_ms = max(start_ms, ev["timestamp"] - REPLAY_OVERLAP_MS)
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        print("# stopped", file=sys.stderr)
        return 0


def _series_from_rows(rows: list[dict]) -> list[dict]:
    """Parse Insights (@timestamp,@message) rows into a time-ordered series."""
    series: list[dict] = []
    for r in rows:
        sample = parse_health_line(r.get("@message", ""))
        if sample is None:
            continue
        # Insights renders @timestamp as "YYYY-MM-DD HH:MM:SS.mmm" (space-separated,
        # no zone); parse_ts reads that and treats the naive value as UTC, which is
        # what CloudWatch means. Skip a row whose timestamp doesn't parse rather
        # than lose the whole series to one malformed value.
        try:
            epoch = parse_ts(r.get("@timestamp", ""))
        except ValueError:
            continue
        series.append({"epoch": epoch, "ts": iso(epoch), **sample})
    series.sort(key=lambda s: s["epoch"])
    return series


def profile_run(series: list[dict], thresholds: Thresholds) -> dict:
    """Compute a post-hoc profile (peaks, onsets, co-occurrence, churn) from a series."""
    if not series:
        return {"samples": 0}

    def _known(key: str) -> list[float]:
        """Values for ``key`` across the series, skipping unknown (None) samples."""
        return [s[key] for s in series if s[key] is not None]

    peaks = {
        "cpu": max(s["cpu"] for s in series),
        # None (JSON null) when the container limit was unknown for every sample.
        "mem": max(_known("mem"), default=None),
        "lag": max(s["lag"] for s in series),
        "no_worker": max(s["no_worker"] for s in series),
        "workers": max(s["workers"] for s in series),
        "tasks": max(s["tasks"] for s in series),
    }

    # Onsets: first timestamp each sustained/single rule trips, replaying the
    # same evaluator the live path uses so live and post-hoc agree.
    onsets: dict[str, str | None] = dict.fromkeys(("cpu-sustained", "mem-high", "backlog-growth", "loop-lag"))
    window: list[dict] = []
    cooccur = 0
    for i, s in enumerate(series):
        window.append(s)
        alerts = evaluate_alerts(window, thresholds)
        for a in alerts:
            if a in onsets and onsets[a] is None:
                onsets[a] = s["ts"]
        # cpu-high AND backlog-rising in the same step: the saturation signature.
        if i > 0 and s["cpu"] >= thresholds.cpu_pct and s["no_worker"] > series[i - 1]["no_worker"]:
            cooccur += 1

    # Worker join/exit events derived from count changes between samples.
    events = []
    for before, after in itertools.pairwise(series):
        if after["workers"] != before["workers"]:
            events.append(
                {
                    "ts": after["ts"],
                    "from": before["workers"],
                    "to": after["workers"],
                    "delta": after["workers"] - before["workers"],
                }
            )

    return {
        "samples": len(series),
        "window": {"start": series[0]["ts"], "end": series[-1]["ts"]},
        "peaks": peaks,
        "onsets": onsets,
        "cpu_backlog_cooccurrence": cooccur,
        "worker_events": events,
    }


def _markdown(profile: dict, log_group: str, *, truncated: bool = False) -> str:
    if not profile.get("samples"):
        return f"### Scheduler profile\n\n_No `scheduler health:` lines found in {log_group}._\n"
    p, w, pk = profile, profile["window"], profile["peaks"]
    mem = "unknown" if pk["mem"] is None else f"{pk['mem']:.0f}%"
    lines = [
        "### Scheduler profile",
        "",
    ]
    if truncated:
        lines += [
            "> **PARTIAL — the query hit the Insights row cap.** Peaks and onsets "
            "below are lower bounds; narrow the window or stream prefix.",
            "",
        ]
    lines += [
        f"- Window: {w['start']} → {w['end']} ({p['samples']} samples)",
        f"- Peaks: cpu **{pk['cpu']:.0f}%**, mem **{mem}**, lag **{pk['lag']:.1f}s**, "
        f"no-worker backlog **{pk['no_worker']}**, workers **{pk['workers']}**, tasks **{pk['tasks']}**",
        f"- cpu-high ∧ backlog-rising intervals: **{p['cpu_backlog_cooccurrence']}**",
        "- Saturation onsets: " + (", ".join(f"{k} @ {v}" for k, v in p["onsets"].items() if v) or "_none tripped_"),
        f"- Worker join/exit events: **{len(p['worker_events'])}**",
    ]
    return "\n".join(lines) + "\n"


def report_run(
    session: boto3.session.Session,
    log_group: str,
    stream_prefix: str,
    start_epoch: int,
    end_epoch: int,
    thresholds: Thresholds,
    markdown: bool,
) -> int:
    """Post-hoc profile: pull health lines via Insights, emit JSON (+ optional md)."""
    logs = session.client("logs")
    # Match on the FULL stream prefix, so --stream-prefix can actually separate
    # overlapping ingests in the shared log group as documented (truncating to the
    # last path component would merge two runs' series into one bogus profile).
    # Insights `like "..."` is a literal substring match — no regex escaping of
    # the slashes, and no chance of a prefix character being read as a metachar.
    query = (
        "fields @timestamp, @message "
        f'| filter @logStream like "{stream_prefix}" '
        f'| filter @message like "{HEALTH_MARKER}" '
        "| sort @timestamp asc"
    )
    rows, truncated = insights_query(logs, log_group, query, start_epoch, end_epoch)
    if rows is None:
        return 1
    series = _series_from_rows(rows)
    profile = profile_run(series, thresholds)
    out = {
        "log_group": log_group,
        "stream_prefix": stream_prefix,
        "window": {"start": iso(start_epoch), "end": iso(end_epoch)},
        # True when the row cap was hit: the series is PARTIAL, so the peaks and
        # onsets below are lower bounds, not the full run (insights_query also
        # warns on stderr). Never present a capped series as a full-run profile.
        "truncated": truncated,
        "profile": profile,
        "series": series,
    }
    print(json.dumps(out, indent=2))
    if markdown:
        print(_markdown(profile, log_group, truncated=truncated), file=sys.stderr)
    return 0


def _build_thresholds(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        cpu_pct=args.cpu_threshold,
        cpu_intervals=args.cpu_intervals,
        mem_pct=args.mem_threshold,
        backlog_intervals=args.backlog_intervals,
        lag_s=args.lag_threshold,
        churn_delta=args.churn_delta,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse args and run the live or post-hoc scheduler-watch mode."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--profile", default=None, help="AWS profile; default uses the ambient credential chain / AWS_PROFILE"
    )
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--log-group", default=DEFAULT_INGEST_LOG_GROUP)
    parser.add_argument(
        "--stream-prefix",
        default=DEFAULT_SCHEDULER_STREAM_PREFIX,
        help="scheduler stream prefix, to disambiguate concurrent ingests (default: %(default)s)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="tail the scheduler stream and emit JSON snapshots")
    mode.add_argument("--report", action="store_true", help="post-hoc profile over --since/--until")
    parser.add_argument("--since", help="report start (ISO8601 UTC)")
    parser.add_argument("--until", help="report end (ISO8601 UTC)")
    parser.add_argument("--markdown", action="store_true", help="report: also print a paste-ready md section to stderr")
    parser.add_argument(
        "--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S, help="live: seconds between polls"
    )
    parser.add_argument("--cpu-threshold", type=float, default=Thresholds.cpu_pct)
    parser.add_argument("--cpu-intervals", type=int, default=Thresholds.cpu_intervals)
    parser.add_argument("--mem-threshold", type=float, default=Thresholds.mem_pct)
    parser.add_argument("--backlog-intervals", type=int, default=Thresholds.backlog_intervals)
    parser.add_argument("--lag-threshold", type=float, default=Thresholds.lag_s)
    parser.add_argument("--churn-delta", type=int, default=Thresholds.churn_delta)
    args = parser.parse_args(argv)

    thresholds = _build_thresholds(args)
    session = boto3.Session(profile_name=args.profile, region_name=args.region)

    if args.live:
        return watch_live(session, args.log_group, args.stream_prefix, args.poll_interval, thresholds)

    if not args.since or not args.until:
        parser.error("--report requires --since and --until")
    return report_run(
        session,
        args.log_group,
        args.stream_prefix,
        parse_ts(args.since),
        parse_ts(args.until),
        thresholds,
        args.markdown,
    )


if __name__ == "__main__":
    raise SystemExit(main())
