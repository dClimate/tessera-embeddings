"""Observe GPU/RAM utilization + per-chunk phase splits on a Ray inference cluster.

General-purpose profiling tool for any Tessera Ray inference deployment (built
during the GPU-saturation campaign; equally aimed at global-tessera runs).
Workers are discovered by the Ray autoscaler's own tags (``ray-cluster-name``
+ ``ray-node-type=worker``) — the same tags teardown uses — so any cluster is
reachable via ``--cluster``/``--cluster-prefix`` regardless of naming scheme.

1. ``--start-pollers`` — start 1 s pollers on every GPU worker via SSM:
   ``nvidia-smi``, DCGM (``dcgmi dmon``: GRACT/SMACT/TENSO/DRAMA/PCIe), and a
   host-RAM sampler (used/avail/pct + top-3 process RSS each second). The RAM
   sampler writes where the CloudWatch agent's dedicated ``ram_poll`` entry
   ships it — so 1 s RAM data SURVIVES cluster teardown (the GPU/DCGM CSVs
   stay in /tmp and die with the worker; summarize them live).
2. ``--report`` — fetch a fleet report from live workers: per-worker GPU-poll
   summary (avg/max util, avg power, busy fraction), a 1 s RAM summary (peak,
   time ≥55%/60%, top spikes), OOM forensics (kernel OOM-killer + Ray memory
   monitor events), and a per-chunk phase-split table. The table prefers the
   actors' machine-readable ``CHUNK_SUMMARY`` JSON lines and falls back to
   legacy prose-log regex parsing for runs from older code.
3. ``--ram-report`` — post-hoc, from CloudWatch (no live cluster needed):
   per-worker peak host RAM + GPU-util distribution from the 30 s RESOURCES
   lines, plus — when the 1 s RAM poller ran — a per-worker 1 s rollup
   (peak/p99) and the top spike samples with timestamps. Scope with
   --since/--until.

Usage::

    # any deployment: pass the profile/region/log-group for that account
    te-observe-cluster --profile yield --start-pollers
    # ...let the run process a few chunks...
    te-observe-cluster --profile yield --report
    # after a run (cluster gone): peak/spike RAM + fleet util from CloudWatch
    python .../observe_cluster.py --profile yield --ram-report \
        --log-group /ec2/yield-embeddings/ray \
        --since 2026-07-16T22:50 --until 2026-07-17T03:00
    # scope to one run's fleet when several clusters share the account
    python .../observe_cluster.py --profile yield --cluster tessera-inference-a60550ae --report

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

# CloudWatch log group the workers' ResourceMonitor lines ship to (yield deploy
# default — pass --log-group for other deployments).
DEFAULT_RAM_LOG_GROUP = "/ec2/yield-embeddings/ray"

# 1 s host-RAM sampler, uploaded to each worker and run with nohup. Writes to
# the Ray session log dir when it exists: the CloudWatch agent ships that exact
# path via a DEDICATED collect_list entry (stream ``<instance>/ram_poll``; see
# providers/aws/cloudwatch-agent.json.tpl), so the 1 s samples survive teardown
# and feed --ram-report's spike analysis. A dedicated entry is required — the
# agent tails only the NEWEST file matching a wildcard entry, so relying on the
# ``**/*.log`` catch-all would both drop samples and let this once-a-second
# file displace every other log behind that glob (it is blacklisted there for
# the same reason). Clusters launched from AMIs/templates predating the entry
# keep the data worker-local — the --report path still summarizes it live.
# /tmp is the fallback when no Ray session exists yet (worker-local only).
# Reads /proc directly (no psutil dependency on the host python).
RAM_POLLER_PY = r"""
import os, time
from datetime import datetime, timezone

def logs_dir():
    d = "/tmp/ray/session_latest/logs"
    return d if os.path.isdir(d) else "/tmp"

def meminfo():
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            m[k.strip()] = int(v.split()[0])  # kB
    total, avail = m["MemTotal"], m["MemAvailable"]
    used = total - avail
    return used / 1048576, avail / 1048576, 100.0 * used / total

def top_rss(n=3):
    procs = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            comm = rss = None
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("Name:"):
                        comm = line.split(None, 1)[1].strip()
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
                        break
            if rss:
                procs.append((rss, comm))
        except OSError:
            continue
    procs.sort(reverse=True)
    return ",".join(f"{c}:{r / 1048576:.1f}" for r, c in procs[:n])

# Resolve the target ONCE and truncate it at startup. The GPU/DCGM pollers
# start with fresh files (`>`); this must too, or re-running --start-pollers on
# a live worker leaves the prior session's samples appended — a stale spike
# would then be reported as the current run's peak. Fixing the path here also
# keeps every append in one file even if the Ray session dir appears later.
path = os.path.join(logs_dir(), "ram_poll.log")
open(path, "w").close()
while True:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        used, avail, pct = meminfo()
        line = f"RAMPOLL ts={ts} used_gb={used:.2f} avail_gb={avail:.2f} pct={pct:.0f} top={top_rss()}"
        with open(path, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    time.sleep(1)
"""

# Kill any prior pollers before starting new ones. Match the actual long-lived
# processes (the `nvidia-smi --query-gpu ... -l`, `dcgmi dmon`, and ram_poll.py
# command lines), not just the redirect-filename wrapper — otherwise re-running
# --start-pollers leaves the old samplers writing concurrently to the same
# files and corrupts the measurements. The `[n]`/`[d]`/`[r]` bracket idiom
# keeps the pattern from matching any shell whose own command line contains it
# (e.g. an SSM wrapper), which pkill -f would otherwise kill before the
# poller-start commands run.
POLLER_COMMANDS = [
    (
        "pkill -f '[n]vidia-smi --query-gpu' 2>/dev/null; pkill -f '[d]cgmi dmon' 2>/dev/null; "
        "pkill -f '[r]am_poll.py' 2>/dev/null; true"
    ),
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
    f"cat > /tmp/ram_poll.py << 'RAMPYEOF'\n{RAM_POLLER_PY}\nRAMPYEOF",
    "setsid nohup python3 /tmp/ram_poll.py </dev/null >/dev/null 2>&1 &",
    "sleep 2; pgrep -af 'nvidia-smi --query|dcgmi dmon|ram_poll' | head -6",
]

GPU_SUMMARY_CMD = (
    "awk -F', ' '{gsub(/ %| W| MiB| MHz/,\"\"); n++; u+=$2; if($2>mu)mu=$2; p+=$4; "
    'if($2>5)busy++} END {if(n>0) printf "samples=%d avg_util=%.1f%% max_util=%d%% '
    "avg_power=%.0fW busy_frac(>5%%)=%.2f\\n\", n,u/n,mu,p/n,busy/n}' /tmp/gpu_poll.csv"
)

# 1 s RAM poll summary. Field layout of a RAMPOLL line:
#   RAMPOLL ts=<iso> used_gb=<g> avail_gb=<g> pct=<p> top=<comm:gb,...>
# ($5 is pct=<p>.) One compound command so $F persists across the pipeline.
RAM_SUMMARY_CMD = (
    'F=/tmp/ray/session_latest/logs/ram_poll.log; [ -f "$F" ] || F=/tmp/ram_poll.log; '
    'awk \'/^RAMPOLL/{n++; split($5,p,"="); v=p[2]+0; s+=v; if(v>mx){mx=v; ts=$2} '
    "if(v>=55)h55++; if(v>=60)h60++} "
    'END{if(n>0) printf "samples=%d mean=%.1f%% peak=%.0f%% at %s | sec>=55%%: %d | sec>=60%%: %d\\n", '
    'n, s/n, mx, ts, h55, h60; else print "no ram_poll data (pollers not started?)"}\' "$F"; '
    "echo '--- top 5 RAM spike samples ---'; "
    "grep -h '^RAMPOLL' \"$F\" 2>/dev/null | sort -t= -k5 -rn | head -5; true"
)

# Runs on the worker: build the per-chunk phase table. Preferred source is the
# actors' machine-readable CHUNK_SUMMARY JSON lines (see actors.py
# _chunk_summary_line — stable keys, immune to prose-wording drift). The legacy
# prose-regex path below remains as a fallback so the tool still works against
# runs from code that predates the summary lines.
PHASE_PARSER = r"""
import glob, json, os, re
from datetime import datetime

LOGS = os.environ.get("TESSERA_RAY_LOGS", "/tmp/ray/session_latest/logs")

summaries = []
for f in glob.glob(f"{LOGS}/worker-*.err"):
    for line in open(f, errors="replace"):
        i = line.find("CHUNK_SUMMARY: ")
        if i != -1:
            try:
                summaries.append(json.loads(line[i + len("CHUNK_SUMMARY: "):]))
            except ValueError:
                pass

if summaries:
    print("label\tTkept\tstrips\trung\tvalid_px\tprologue_s\tinfer_s\toverhead_s\ttotal_s\tpx/s\tstatus")
    for s in summaries:
        total = float(s.get("total_s") or 0)
        px = int(s.get("valid_px") or 0)
        pxs = px / total if total > 0 else 0
        print("\t".join(str(x) for x in (
            s.get("label"), s.get("t_kept", "?"), s.get("strips", "?"), s.get("rung", "?"),
            px, s.get("prologue_s", "?"), s.get("infer_s", "?"), s.get("overhead_s", "?"),
            s.get("total_s", "?"), f"{pxs:.0f}", s.get("status"),
        )))
else:
    # ------- legacy prose-log parsing (pre-CHUNK_SUMMARY code) -------
    PAT = {
     "pref": re.compile(r"^(\S+ \S+) \S+ INFO xchunk prefetch: hit \([^)]+\) for (\S+)"),
     "tkept": re.compile(r"^(\S+ \S+) \S+ INFO Chunk (\S+): T_kept=(\d+) -> strip_h=(\d+) -> (\d+) strip"),
     "ds": re.compile(r"^(\S+ \S+) \S+ INFO MosaicChunkInferenceDataset: (\d+) valid pixels"
                      r" out of \d+ total \(([\d.]+)%\) in (\d+) buckets"),
     "start": re.compile(r"^(\S+ \S+) \S+ INFO Starting v1.1 inference"),
     "idone": re.compile(r"^(\S+ \S+) \S+ INFO Inference complete: .* ([\d.]+)s total, (\d+) px/sec"),
     "cdone": re.compile(r"^(\S+ \S+) \S+ INFO Chunk (\S+) complete: (\d+) valid pixels, ([\d.]+)s"),
    }
    def ts(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")
    events = []
    for f in glob.glob(f"{LOGS}/worker-*.err"):
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
    "echo '=== RAM POLL SUMMARY (1s) ==='",
    RAM_SUMMARY_CMD,
    "echo '=== OOM / MEMORY EVENTS ==='",
    (
        "{ sudo -n dmesg -T 2>/dev/null || dmesg -T 2>/dev/null; } "
        "| grep -iE 'out of memory|oom[-_]?kill' | tail -5; true"
    ),
    (
        "grep -hiE 'node memory usage|low on memory|killed.*memory' "
        "/tmp/ray/session_latest/logs/raylet* 2>/dev/null | tail -3; true"
    ),
    "echo '=== PHASE SPLITS ==='",
    f"python3 - << 'PYEOF'\n{PHASE_PARSER}\nPYEOF",
]


def find_workers(session: boto3.session.Session, cluster: str | None, cluster_prefix: str) -> list[str]:
    """Return running, SSM-registered GPU workers of the target cluster(s).

    Discovery keys on the Ray autoscaler's own EC2 tags — ``ray-cluster-name``
    (the same tag teardown terminates by) and ``ray-node-type=worker`` — so it
    works for any deployment regardless of instance-naming scheme. ``cluster``
    scopes to one run's fleet exactly; otherwise ``cluster_prefix`` matches all
    clusters whose name starts with it.

    Freshly-launched autoscaler workers take a minute to register with SSM, and
    one unregistered instance in a ``send_command`` batch fails the whole call —
    so intersect the EC2 listing with SSM's registered set and report the rest.
    """
    ec2 = session.client("ec2")
    tag_values = [cluster] if cluster else [f"{cluster_prefix}*"]
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "instance-state-name", "Values": ["running"]},
            {"Name": "tag:ray-node-type", "Values": ["worker"]},
            {"Name": "tag:ray-cluster-name", "Values": tag_values},
        ]
    )
    running = [inst["InstanceId"] for res in resp["Reservations"] for inst in res["Instances"]]
    if not running:
        return []

    ssm = session.client("ssm")
    registered: set[str] = set()
    paginator = ssm.get_paginator("describe_instance_information")
    # The InstanceIds filter caps Values at 50; the fleet can exceed that
    # (cluster.yaml.template allows max_workers=500), so query in batches or a
    # >50-worker run raises ValidationException here. _SSM_MAX_INSTANCES (=50)
    # is defined below for the same cap on SendCommand.
    for i in range(0, len(running), _SSM_MAX_INSTANCES):
        batch = running[i : i + _SSM_MAX_INSTANCES]
        for page in paginator.paginate(Filters=[{"Key": "InstanceIds", "Values": batch}]):
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
                try:
                    inv = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=iid)
                except ssm.exceptions.InvocationDoesNotExist:
                    # SSM Run Command is eventually consistent — the invocation
                    # can be briefly invisible right after send_command. Leave the
                    # instance pending and retry on the next poll.
                    continue
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


def _insights_query(
    logs: object, log_group: str, query: str, start_epoch: int, end_epoch: int
) -> list[dict[str, str]] | None:
    """Run one CloudWatch Insights query and return its rows (None on failure).

    Bounded poll: Insights statuses are Scheduled/Running/Complete/Failed/
    Cancelled/Timeout/Unknown. Break on any terminal state (incl. Unknown) and
    cap total wait so a query stuck Scheduled/Running/Unknown can't spin forever.
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


def _worker_label(stream: str) -> str:
    """Per-worker label from a ``<instance-id>/<suffix>`` CloudWatch stream name.

    The instance id is the discriminating part — the suffix (``actors``,
    ``ram_poll``, …) is the SAME for every worker, so labelling by the suffix
    (a trailing ``rsplit('/', 1)[-1]``) collapses all rows to one name.
    """
    return stream.split("/", 1)[0][:20]


def ram_util_report(session: boto3.session.Session, log_group: str, start_epoch: int, end_epoch: int) -> int:
    """Per-worker peak host RAM + GPU-util distribution from CloudWatch.

    Sources two line families the workers ship to ``log_group``, so it works
    AFTER a run's cluster has torn down — unlike the SSM ``--report`` path:

    - ``RESOURCES`` (30 s, always on): per-stream peak RAM + GPU-util rollup.
      30 s cadence MISSES sub-30 s spikes — treat its peak as a floor.
    - ``RAMPOLL`` (1 s, present when ``--start-pollers`` ran): per-stream 1 s
      peak/p99 rollup plus the top spike samples with timestamps and the top
      RSS processes at that instant — the spike-forensics view.

    NOTE: the log group is shared across runs, so scope the window tightly to
    the run of interest; streams from an overlapping run share the same group.
    """
    logs = session.client("logs")
    resources_q = (
        "fields @logStream, @message "
        "| filter @message like /RESOURCES/ "
        '| parse @message "RAM=*/* GB (*%)" as ram_gb, ram_total, ram_pct '
        '| parse @message "GPU=*%" as gpu_pct '
        "| stats max(ram_gb) as peak_ram_gb, max(ram_pct) as peak_ram_pct, "
        "max(ram_total) as node_gb, avg(gpu_pct) as avg_gpu, min(gpu_pct) as min_gpu, "
        "count(*) as samples by @logStream "
        "| sort peak_ram_pct desc"
    )
    win = f"{datetime.datetime.fromtimestamp(start_epoch, datetime.UTC):%Y-%m-%d %H:%M}"
    win += f" → {datetime.datetime.fromtimestamp(end_epoch, datetime.UTC):%Y-%m-%d %H:%M} UTC"
    print(f"RAM + GPU-util from {log_group}  [{win}]")

    # The two sources are INDEPENDENT: a short run (or a worker OOM-killed
    # before its first 30 s RESOURCES sample) may have only RAMPOLL — the very
    # 1 s evidence this feature targets — so an empty RESOURCES set must not
    # short-circuit the RAMPOLL section. `_insights_query` returns None on a
    # HARD failure (permission/timeout/failed status), [] for "no such lines":
    # None → return nonzero (never mask a failed query as "no data"); [] →
    # note it and try the other source; fail only if NEITHER produced data.

    # --- 30 s RESOURCES rollup (always-on monitor) ---
    rows = _insights_query(logs, log_group, resources_q, start_epoch, end_epoch)
    if rows is None:
        return 1
    if rows:
        print(f"\n30s RESOURCES rollup ({len(rows)} worker streams):")
        print("worker\tpeak_ram_gb\tpeak_ram_%\tavg_gpu_%\tmin_gpu_%\tsamples")
        peak_gb = peak_pct = 0.0
        weighted_gpu, total_n, node_gb = 0.0, 0, 0.0
        for r in rows:
            pg, pp = float(r["peak_ram_gb"]), float(r["peak_ram_pct"])
            ag, n = float(r["avg_gpu"]), int(float(r["samples"]))
            node_gb = max(node_gb, float(r.get("node_gb", 0) or 0))
            peak_gb, peak_pct = max(peak_gb, pg), max(peak_pct, pp)
            weighted_gpu += ag * n
            total_n += n
            print(f"{_worker_label(r['@logStream'])}\t{pg:.1f}\t{pp:.0f}\t{ag:.1f}\t{float(r['min_gpu']):.0f}\t{n}")
        fleet_gpu = weighted_gpu / total_n if total_n else 0.0
        print(
            f"\nFLEET: peak host RAM = {peak_gb:.1f} GB ({peak_pct:.0f}% of ~{node_gb:.1f} GB node) | "
            f"sample-weighted avg GPU util = {fleet_gpu:.1f}% across {len(rows)} workers | "
            f"idle-recovery ceiling if all → 100% ≈ {100 - fleet_gpu:.0f}%"
        )
        print(
            "(peak RAM above is the 30s-polled peak — a FLOOR for the true instantaneous peak. "
            "GPU util is nvidia-smi 'busy-or-not', an upper bound on useful-work fraction.)"
        )
    else:
        print("\n(no 30s RESOURCES lines in window)")

    # --- 1 s RAMPOLL rollup + spike table (present when --start-pollers ran) ---
    poll_q = (
        "fields @logStream, @message "
        "| filter @message like /RAMPOLL/ "
        '| parse @message "used_gb=* avail_gb=* pct=* top=*" as used_gb, avail_gb, pctv, topv '
        "| stats max(used_gb) as peak_gb, max(pctv) as peak_pct, pct(pctv, 99) as p99_pct, "
        "count(*) as samples by @logStream "
        "| sort peak_pct desc"
    )
    poll_rows = _insights_query(logs, log_group, poll_q, start_epoch, end_epoch)
    if poll_rows is None:
        return 1
    if poll_rows:
        print(f"\n1s RAM POLL rollup ({len(poll_rows)} worker streams):")
        print("worker\tpeak_gb\tpeak_%\tp99_%\tsamples")
        for r in poll_rows:
            print(
                f"{_worker_label(r['@logStream'])}\t{float(r['peak_gb']):.1f}\t{float(r['peak_pct']):.0f}"
                f"\t{float(r['p99_pct']):.0f}\t{int(float(r['samples']))}"
            )
        spike_q = (
            "fields @logStream, @message "
            "| filter @message like /RAMPOLL/ "
            '| parse @message "pct=* top=*" as pctv, topv '
            "| sort pctv desc "
            "| limit 10"
        )
        spikes = _insights_query(logs, log_group, spike_q, start_epoch, end_epoch)
        if spikes is None:
            return 1
        if spikes:
            print("\nTop 10 RAM spike samples (1s):")
            for r in spikes:
                msg = r.get("@message", "").strip()
                print(f"{_worker_label(r['@logStream'])}\t{msg[msg.find('RAMPOLL') :][:160]}")
    else:
        print("\n(no 1s RAMPOLL lines in window — --start-pollers not run, or pre-tooling code)")

    if not rows and not poll_rows:
        print(
            "No RAM data (RESOURCES or RAMPOLL) in the window (check --log-group / --since / --until).",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="yield", help="AWS profile (default: yield)")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--cluster", help="Exact ray-cluster-name tag — scope to one run's fleet")
    parser.add_argument(
        "--cluster-prefix",
        "--name-prefix",  # backward-compat alias (previous versions matched the EC2 Name)
        dest="cluster_prefix",
        default="tessera-inference",
        help="ray-cluster-name tag prefix to match when --cluster is not given (default: tessera-inference)",
    )
    parser.add_argument("--start-pollers", action="store_true", help="Start 1s GPU + host-RAM pollers on all workers")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Fetch GPU/RAM summaries, OOM events, and per-chunk phase splits (live workers)",
    )
    parser.add_argument(
        "--ram-report",
        action="store_true",
        help="Per-worker peak host RAM + GPU-util distribution from CloudWatch RESOURCES lines, "
        "plus 1s spike analysis when the RAM poller ran (works post-run, no live cluster "
        "needed). Scope with --since/--until.",
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

    # Tolerate the old Name-style prefix (instance Names are "ray-<cluster>-...",
    # the tag value is the bare cluster name).
    prefix = args.cluster_prefix.removeprefix("ray-")
    workers = find_workers(session, args.cluster, prefix)
    if not workers:
        target = args.cluster or f"{prefix}*"
        print(f"No running SSM-registered workers with ray-cluster-name={target}", file=sys.stderr)
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
