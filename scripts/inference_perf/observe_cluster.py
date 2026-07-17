"""Observe GPU utilization + per-chunk phase splits on a live Ray inference cluster.

Phase-0 measurement tool for the GPU-saturation campaign. Against a running
``ray-tessera-inference-*`` cluster it can:

1. ``--start-pollers`` — start 1 s ``nvidia-smi`` and DCGM (``dcgmi dmon``:
   GRACT/SMACT/TENSO/DRAMA/PCIe) captures on every GPU worker via SSM.
2. ``--report`` — fetch a fleet report: per-worker GPU-poll summary (avg/max
   util, avg power, busy fraction) plus a per-chunk phase-split table parsed
   from the actor logs (gap+mask / band read / SAR+build / inference / write),
   the wall-clock anatomy that shows how much GPU time each phase wastes.
3. ``--ram-report`` — per-worker peak host RAM + GPU-util distribution pulled
   from the workers' ``ResourceMonitor`` ``RESOURCES`` lines in CloudWatch.
   Unlike ``--report`` (which SSHes live workers), this works AFTER a run has
   torn down, so it answers "how close to the RAM ceiling did we get?" and
   "how much GPU idle is left to recover?" post hoc. Scope with --since/--until.

Usage::

    python scripts/inference_perf/observe_cluster.py --profile yield --start-pollers
    # ...let the run process a few chunks...
    python scripts/inference_perf/observe_cluster.py --profile yield --report
    # after a run (cluster gone): peak RAM + fleet util from CloudWatch
    python .../observe_cluster.py --profile yield --ram-report \
        --since 2026-07-16T22:50 --until 2026-07-17T03:00

--report/--start-pollers require workers reachable via SSM (the production AMI
runs the agent). --ram-report only needs CloudWatch Logs read access. All modes
use an AWS profile with the relevant permissions.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time

import boto3

# CloudWatch log group the workers' ResourceMonitor lines ship to (yield deploy).
DEFAULT_RAM_LOG_GROUP = "/ec2/yield-embeddings/ray"

# Kill any prior pollers before starting new ones. Match the actual long-lived
# processes (the `nvidia-smi --query-gpu ... -l` and `dcgmi dmon` command lines),
# not just the redirect-filename wrapper — otherwise re-running --start-pollers
# leaves the old samplers writing concurrently to the same files and corrupts the
# measurements.
POLLER_COMMANDS = [
    "pkill -f 'nvidia-smi --query-gpu' 2>/dev/null; pkill -f 'dcgmi dmon' 2>/dev/null; true",
    (
        "setsid nohup bash -c 'nvidia-smi --query-gpu=timestamp,utilization.gpu,"
        "utilization.memory,power.draw,clocks.sm,memory.used --format=csv,noheader -l 1 "
        "> /tmp/gpu_poll.csv 2>&1' </dev/null >/dev/null 2>&1 &"
    ),
    (
        "if command -v dcgmi >/dev/null; then setsid nohup bash -c "
        "'dcgmi dmon -e 1001,1002,1004,1005,1009,1010 -d 1000 > /tmp/dcgm_poll.txt 2>&1' "
        "</dev/null >/dev/null 2>&1 & fi"
    ),
    "sleep 2; pgrep -af 'nvidia-smi --query|dcgmi dmon' | head -4",
]

GPU_SUMMARY_CMD = (
    "awk -F', ' '{gsub(/ %| W| MiB| MHz/,\"\"); n++; u+=$2; if($2>mu)mu=$2; p+=$4; "
    'if($2>5)busy++} END {if(n>0) printf "samples=%d avg_util=%.1f%% max_util=%d%% '
    "avg_power=%.0fW busy_frac(>5%%)=%.2f\\n\", n,u/n,mu,p/n,busy/n}' /tmp/gpu_poll.csv"
)

# Runs on the worker: parse actor .err logs into per-chunk phase rows. The
# format string targets _configure_actor_logging's basicConfig layout.
PHASE_PARSER = r"""
import glob, re
from datetime import datetime
PAT = {
 "pref": re.compile(r"^(\S+ \S+) \S+ INFO Using prefetched prologue for (\S+)"),
 "tkept": re.compile(r"^(\S+ \S+) \S+ INFO Chunk (\S+): T_kept=(\d+) -> strip_h=(\d+) -> (\d+) strip"),
 "ds": re.compile(r"^(\S+ \S+) \S+ INFO MosaicChunkInferenceDataset: (\d+) valid pixels"
                  r" out of \d+ total \(([\d.]+)%\) in (\d+) buckets"),
 "start": re.compile(r"^(\S+ \S+) \S+ INFO Starting v1.1 inference"),
 "idone": re.compile(r"^(\S+ \S+) \S+ INFO Inference complete: .* ([\d.]+)s total, (\d+) px/sec"),
 "cdone": re.compile(r"^(\S+ \S+) \S+ INFO Chunk (\S+) complete: (\d+) valid pixels, ([\d.]+)s"),
}
def ts(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")
events = []
for f in glob.glob("/tmp/ray/session_latest/logs/worker-*.err"):
    for line in open(f, errors="replace"):
        for kind, pat in PAT.items():
            m = pat.match(line.strip())
            if m: events.append((ts(m.group(1)), kind, m.groups()[1:]))
events.sort(key=lambda e: e[0])
# Overhead accounting: overhead_s = total_s - infer_s - write_s is the honest
# per-chunk non-inference cost in BOTH modes — it's derived from the wall-clock
# total (measured from the top of _process_chunk, before any prologue work) minus
# the two GPU-adjacent phases, so it captures cold load regardless of where the
# load lines fall. prologue_s (T_kept-log -> inference-start) is main-thread-only
# and UNDERSTATES cold chunks: the "Chunk T_kept" line is emitted after the
# prologue's SCL/band read, and with cross-chunk prefetch (Phase 1) the band/SAR
# loads log during the PREVIOUS chunk's window entirely. Read overhead_s for the
# real figure; prologue_s only isolates main-thread build. pref=Y marks chunks
# that consumed a prefetched stash (expect their overhead_s ~= write_s).
# A split chunk calls run_inference once per strip, so ds/start/idone lines
# repeat within one chunk. Accumulate: t_start = FIRST strip's inference start,
# infer_s = SUM of every strip's reported inference seconds, t_idone = LAST
# strip's completion. valid_px comes from the actor's chunk-total "complete: N
# valid pixels" (cdone) line — NOT the per-strip ds sum, which under-counts
# multi-strip chunks (a strip with no valid pixels emits no ds line, and the ds
# count is per-strip-within-crop); px/s is the honest END-TO-END rate
# valid_px / total_s. ds lines are still summed as a fallback if cdone is absent.
rows, cur, prev_done, pending_pref = [], None, None, set()
for t, kind, g in events:
    if kind == "pref":
        pending_pref.add(g[0])
    elif kind == "tkept":
        gap = (t - prev_done).total_seconds() if prev_done else None
        cur = {"label": g[0], "tkept": g[1], "strips": g[3], "gap_s": gap, "t_mask": t,
               "pref": "Y" if g[0] in pending_pref else "N", "infer_s": 0.0, "px": 0}
        pending_pref.discard(g[0])
    elif cur is None: continue
    elif kind == "ds":
        if "buckets" not in cur: cur["buckets"] = g[2]  # first strip's bucket count
        cur["px"] += int(g[0])
    elif kind == "start":
        if "t_start" not in cur: cur["t_start"] = t
    elif kind == "idone":
        cur["t_idone"] = t  # last strip wins
        cur["infer_s"] += float(g[0])
    elif kind == "cdone":
        cur["t_cdone"], cur["cdone_px"], cur["total_s"] = t, int(g[1]), g[2]
        rows.append(cur); prev_done = t; cur = None
print("label\tTkept\tstrips\tpref\tbuckets\tvalid_px\tprologue_s\tinfer_s\twrite_s\toverhead_s\ttotal_s\tpx/s")
for r in rows:
    try:
        prologue = (r["t_start"] - r["t_mask"]).total_seconds()
        write = (r["t_cdone"] - r["t_idone"]).total_seconds()
        infer = r["infer_s"]
        total = float(r["total_s"])
        overhead = total - infer - write
        vpx = r.get("cdone_px", r["px"])  # authoritative chunk-total; ds-sum fallback
        pxs = (vpx / total) if total > 0 else 0  # honest END-TO-END px/s
        print(f"{r['label']}\t{r['tkept']}\t{r['strips']}\t{r['pref']}\t{r.get('buckets','?')}\t{vpx}\t{prologue:.1f}\t{infer:.1f}\t{write:.1f}\t{overhead:.1f}\t{r['total_s']}\t{pxs:.0f}")
    except KeyError as e:
        print(f"{r['label']}\tINCOMPLETE({e})")
"""

REPORT_COMMANDS = [
    "echo '=== GPU POLL SUMMARY ==='",
    GPU_SUMMARY_CMD,
    "echo '=== DCGM tail ==='",
    "tail -6 /tmp/dcgm_poll.txt 2>/dev/null || echo no-dcgm",
    "echo '=== PHASE SPLITS ==='",
    f"python3 - << 'PYEOF'\n{PHASE_PARSER}\nPYEOF",
]


def find_workers(session: boto3.session.Session, name_prefix: str) -> list[str]:
    """Return running, SSM-registered GPU workers whose Name tag matches the prefix.

    Freshly-launched autoscaler workers take a minute to register with SSM, and
    one unregistered instance in a ``send_command`` batch fails the whole call —
    so intersect the EC2 listing with SSM's registered set and report the rest.
    """
    ec2 = session.client("ec2")
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "instance-state-name", "Values": ["running"]},
            {"Name": "tag:Name", "Values": [f"{name_prefix}*worker*"]},
        ]
    )
    running = [inst["InstanceId"] for res in resp["Reservations"] for inst in res["Instances"]]
    if not running:
        return []

    ssm = session.client("ssm")
    registered: set[str] = set()
    paginator = ssm.get_paginator("describe_instance_information")
    for page in paginator.paginate(Filters=[{"Key": "InstanceIds", "Values": running}]):
        registered.update(info["InstanceId"] for info in page["InstanceInformationList"])

    skipped = sorted(set(running) - registered)
    if skipped:
        print(f"Skipping {len(skipped)} worker(s) not yet SSM-registered: {', '.join(skipped)}")
    return [iid for iid in running if iid in registered]


# SSM SendCommand caps InstanceIds at 50; the GPU fleet can reach the 80-actor
# cap (config.dask.NUM_ACTORS_CAP), so send in batches of 50.
_SSM_MAX_INSTANCES = 50


def run_on_workers(session: boto3.session.Session, instance_ids: list[str], commands: list[str]) -> dict[str, str]:
    """Run shell commands on all workers via SSM; return {instance_id: stdout}."""
    ssm = session.client("ssm")
    cmd_ids: dict[str, list[str]] = {}  # cmd_id -> its instance ids
    for i in range(0, len(instance_ids), _SSM_MAX_INSTANCES):
        batch = instance_ids[i : i + _SSM_MAX_INSTANCES]
        cmd_id = ssm.send_command(
            InstanceIds=batch,
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": commands},
        )["Command"]["CommandId"]
        cmd_ids[cmd_id] = batch

    outputs: dict[str, str] = {}
    for cmd_id, batch in cmd_ids.items():
        # Round-robin across the batch's instances so the ~60s wait is shared,
        # not serialized: a single slow/unresponsive instance mustn't delay
        # collecting results from the rest (sequential per-instance waits would
        # be up to 60s x len(batch) in the worst case).
        pending = set(batch)
        for _ in range(30):
            if not pending:
                break
            time.sleep(2)
            for iid in list(pending):
                inv = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=iid)
                if inv["Status"] in ("Success", "Failed", "TimedOut", "Cancelled"):
                    outputs[iid] = inv["StandardOutputContent"] + (
                        f"\n[stderr] {inv['StandardErrorContent']}" if inv["StandardErrorContent"].strip() else ""
                    )
                    pending.remove(iid)
        for iid in pending:
            outputs[iid] = "[timed out waiting for SSM invocation]"
    return outputs


def _parse_ts(s: str) -> int:
    """Parse an ISO8601 UTC timestamp (or 'YYYY-MM-DDTHH:MM') to epoch seconds."""
    dt = datetime.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return int(dt.timestamp())


def ram_util_report(
    session: boto3.session.Session, log_group: str, start_epoch: int, end_epoch: int
) -> int:
    """Per-worker peak host RAM + GPU-util distribution from CloudWatch.

    Sources the ``ResourceMonitor`` ``RESOURCES`` lines (30 s cadence) that the
    workers ship to ``log_group``, so it works AFTER a run's cluster has torn
    down — unlike the SSM ``--report`` path, which needs live workers.

    Reports, per log stream (≈ per worker): peak host RAM (GB and % of node),
    mean/min GPU utilization, and sample count; plus fleet rollups. Peak RAM is
    the 30 s-polled peak — it can miss an instantaneous OOM spike (the OOM
    killer catches those; see the flow-runner memory-monitor diagnostics).

    NOTE: the log group is shared across runs, so scope the window tightly to
    the run of interest; streams from an overlapping run share the same group.
    """
    logs = session.client("logs")
    query = (
        "fields @logStream, @message "
        "| filter @message like /RESOURCES/ "
        '| parse @message "RAM=*/* GB (*%)" as ram_gb, ram_total, ram_pct '
        '| parse @message "GPU=*%" as gpu_pct '
        "| stats max(ram_gb) as peak_ram_gb, max(ram_pct) as peak_ram_pct, "
        "max(ram_total) as node_gb, avg(gpu_pct) as avg_gpu, min(gpu_pct) as min_gpu, "
        "count(*) as samples by @logStream "
        "| sort peak_ram_pct desc"
    )
    qid = logs.start_query(
        logGroupName=log_group, startTime=start_epoch, endTime=end_epoch, queryString=query, limit=10000
    )["queryId"]
    while True:
        res = logs.get_query_results(queryId=qid)
        if res["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
        time.sleep(2)
    if res["status"] != "Complete":
        print(f"CloudWatch query {res['status']}", file=sys.stderr)
        return 1

    rows = [{c["field"]: c["value"] for c in row} for row in res["results"]]
    if not rows:
        print("No RESOURCES lines in the window (check --log-group / --since / --until).", file=sys.stderr)
        return 1

    win = f"{datetime.datetime.fromtimestamp(start_epoch, datetime.UTC):%Y-%m-%d %H:%M}"
    win += f" → {datetime.datetime.fromtimestamp(end_epoch, datetime.UTC):%Y-%m-%d %H:%M} UTC"
    print(f"RAM + GPU-util from {log_group}  [{win}]  ({len(rows)} worker streams)\n")
    print("stream\tpeak_ram_gb\tpeak_ram_%\tavg_gpu_%\tmin_gpu_%\tsamples")
    peak_gb = peak_pct = 0.0
    gpu_avgs, weighted_gpu, total_n, node_gb = [], 0.0, 0, 0.0
    for r in rows:
        pg, pp = float(r["peak_ram_gb"]), float(r["peak_ram_pct"])
        ag, n = float(r["avg_gpu"]), int(float(r["samples"]))
        node_gb = max(node_gb, float(r.get("node_gb", 0) or 0))
        peak_gb, peak_pct = max(peak_gb, pg), max(peak_pct, pp)
        gpu_avgs.append(ag)
        weighted_gpu += ag * n
        total_n += n
        stream = r["@logStream"].rsplit("/", 1)[-1][:32]
        print(f"{stream}\t{pg:.1f}\t{pp:.0f}\t{ag:.1f}\t{float(r['min_gpu']):.0f}\t{n}")

    fleet_gpu = weighted_gpu / total_n if total_n else 0.0
    print(
        f"\nFLEET: peak host RAM = {peak_gb:.1f} GB ({peak_pct:.0f}% of ~{node_gb:.1f} GB node) | "
        f"sample-weighted avg GPU util = {fleet_gpu:.1f}% across {len(rows)} workers | "
        f"idle-recovery ceiling if all → 100% ≈ {100 - fleet_gpu:.0f}%"
    )
    print(
        "(peak RAM is the 30s-polled peak; instantaneous OOM spikes show only in the "
        "flow-runner memory-monitor diagnostics. GPU util is nvidia-smi 'busy-or-not', "
        "an upper bound on useful-work fraction — not all the sub-100% gap is CPU-feed.)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="yield", help="AWS profile (default: yield)")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--name-prefix", default="ray-tessera-inference")
    parser.add_argument("--start-pollers", action="store_true", help="Start 1s GPU pollers on all workers")
    parser.add_argument(
        "--report", action="store_true", help="Fetch GPU summaries + per-chunk phase splits (live workers)"
    )
    parser.add_argument(
        "--ram-report",
        action="store_true",
        help="Per-worker peak host RAM + GPU-util distribution from CloudWatch RESOURCES lines "
        "(works post-run, no live cluster needed). Scope with --since/--until.",
    )
    parser.add_argument("--log-group", default=DEFAULT_RAM_LOG_GROUP, help="CloudWatch log group for --ram-report")
    parser.add_argument("--since", help="--ram-report window start, ISO8601 UTC (e.g. 2026-07-16T22:50)")
    parser.add_argument("--until", help="--ram-report window end, ISO8601 UTC")
    parser.add_argument(
        "--hours", type=float, default=6.0, help="--ram-report lookback when --since omitted (default 6h)"
    )
    args = parser.parse_args(argv)

    if not (args.start_pollers or args.report or args.ram_report):
        parser.error("nothing to do: pass --start-pollers, --report, and/or --ram-report")

    session = boto3.session.Session(profile_name=args.profile, region_name=args.region)

    if args.ram_report:
        end = _parse_ts(args.until) if args.until else int(time.time())
        start = _parse_ts(args.since) if args.since else end - int(args.hours * 3600)
        rc = ram_util_report(session, args.log_group, start, end)
        if not (args.start_pollers or args.report):
            return rc

    workers = find_workers(session, args.name_prefix)
    if not workers:
        print(f"No running workers matching {args.name_prefix}*worker*", file=sys.stderr)
        return 1
    print(f"Found {len(workers)} worker(s): {', '.join(workers)}\n")

    if args.start_pollers:
        for iid, out in run_on_workers(session, workers, POLLER_COMMANDS).items():
            print(f"###### {iid} (pollers)\n{out}")

    if args.report:
        for iid, out in run_on_workers(session, workers, REPORT_COMMANDS).items():
            print(f"###### {iid}\n{out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
