"""Observe GPU utilization + per-chunk phase splits on a live Ray inference cluster.

Phase-0 measurement tool for the GPU-saturation campaign. Against a running
``ray-tessera-inference-*`` cluster it can:

1. ``--start-pollers`` — start 1 s ``nvidia-smi`` and DCGM (``dcgmi dmon``:
   GRACT/SMACT/TENSO/DRAMA/PCIe) captures on every GPU worker via SSM.
2. ``--report`` — fetch a fleet report: per-worker GPU-poll summary (avg/max
   util, avg power, busy fraction) plus a per-chunk phase-split table parsed
   from the actor logs (gap+mask / band read / SAR+build / inference / write),
   the wall-clock anatomy that shows how much GPU time each phase wastes.

Usage::

    python scripts/inference_perf/observe_cluster.py --profile yield --start-pollers
    # ...let the run process a few chunks...
    python scripts/inference_perf/observe_cluster.py --profile yield --report

Requires workers reachable via SSM (the production AMI runs the agent) and an
AWS profile with ec2:DescribeInstances + ssm:SendCommand.
"""

from __future__ import annotations

import argparse
import sys
import time

import boto3

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
    "if($2>5)busy++} END {if(n>0) printf \"samples=%d avg_util=%.1f%% max_util=%d%% "
    "avg_power=%.0fW busy_frac(>5%%)=%.2f\\n\", n,u/n,mu,p/n,busy/n}' /tmp/gpu_poll.csv"
)

# Runs on the worker: parse actor .err logs into per-chunk phase rows. The
# format string targets _configure_actor_logging's basicConfig layout.
PHASE_PARSER = r'''
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
rows, cur, prev_done, pending_pref = [], None, None, set()
for t, kind, g in events:
    if kind == "pref":
        pending_pref.add(g[0])
    elif kind == "tkept":
        gap = (t - prev_done).total_seconds() if prev_done else None
        cur = {"label": g[0], "tkept": g[1], "strips": g[3], "gap_s": gap, "t_mask": t,
               "pref": "Y" if g[0] in pending_pref else "N"}
        pending_pref.discard(g[0])
    elif cur is None: continue
    elif kind == "ds" and "t_ds" not in cur:
        cur["t_ds"], cur["px"], cur["valid_pct"], cur["buckets"] = t, g[0], g[1], g[2]
    elif kind == "start" and "t_start" not in cur: cur["t_start"] = t
    elif kind == "idone" and "t_idone" not in cur:
        cur["t_idone"], cur["infer_s"], cur["pxs"] = t, g[0], g[1]
    elif kind == "cdone":
        cur["t_cdone"], cur["total_s"] = t, g[2]; rows.append(cur); prev_done = t; cur = None
print("label\tTkept\tstrips\tpref\tbuckets\tvalid_px\tvalid%\tprologue_s\tinfer_s\twrite_s\toverhead_s\ttotal_s\tpx/s")
for r in rows:
    try:
        prologue = (r["t_start"] - r["t_mask"]).total_seconds()
        write = (r["t_cdone"] - r["t_idone"]).total_seconds()
        overhead = float(r["total_s"]) - float(r["infer_s"]) - write
        print(f"{r['label']}\t{r['tkept']}\t{r['strips']}\t{r['pref']}\t{r.get('buckets','?')}\t{r.get('px','?')}\t{r.get('valid_pct','?')}\t{prologue:.1f}\t{r.get('infer_s','?')}\t{write:.1f}\t{overhead:.1f}\t{r['total_s']}\t{r.get('pxs','?')}")
    except KeyError as e:
        print(f"{r['label']}\tINCOMPLETE({e})")
'''

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
        for iid in batch:
            for _ in range(30):
                time.sleep(2)
                inv = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=iid)
                if inv["Status"] in ("Success", "Failed", "TimedOut", "Cancelled"):
                    outputs[iid] = inv["StandardOutputContent"] + (
                        f"\n[stderr] {inv['StandardErrorContent']}" if inv["StandardErrorContent"].strip() else ""
                    )
                    break
            else:
                outputs[iid] = "[timed out waiting for SSM invocation]"
    return outputs


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="yield", help="AWS profile (default: yield)")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--name-prefix", default="ray-tessera-inference")
    parser.add_argument("--start-pollers", action="store_true", help="Start 1s GPU pollers on all workers")
    parser.add_argument("--report", action="store_true", help="Fetch GPU summaries + per-chunk phase splits")
    args = parser.parse_args(argv)

    if not (args.start_pollers or args.report):
        parser.error("nothing to do: pass --start-pollers and/or --report")

    session = boto3.session.Session(profile_name=args.profile, region_name=args.region)
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
