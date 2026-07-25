"""Assemble a per-run ingest dossier from the profiling JSON outputs.

The dossier is the deliverable of every at-scale ingest rung: it merges the
scheduler profile (``watch_scheduler.py --report``), the external-service
aggregates (``ingest_log_queries.py``), and — when captured — the deep-profile
``performance_report`` artifact link, into one markdown skeleton. This tool
does the mechanical merge and leaves a clearly-marked **Interpretation** section
for the operator (Claude) to fill in with the bottleneck verdict and the
recommended next rung.

It is pure: it reads JSON files already produced by the other tools and touches
no AWS, so it is deterministic and unit-testable, and the raw JSON stays the
system of record behind the prose.

Usage::

    te-watch-scheduler ... --report > sched.json
    te-ingest-log-queries ... > logs.json
    te-ingest-report \
        --scheduler sched.json --logs logs.json \
        --run-id abc123 --zone 32633 --year 2025 --max-workers 500 \
        --perf-report s3://.../perf.html --out dossier.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: str | None) -> dict | None:
    """Read one profiling JSON file, or None when the flag wasn't given.

    Raises :class:`ValueError` for an unreadable or malformed file so ``main`` can
    turn it into a one-line message: an operator assembling a dossier has usually
    just mistyped a path or piped a tool's stderr over its stdout, and a traceback
    is a poor way to say so.
    """
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text())
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON ({e}) — is it a profiling tool's stdout?") from e


def _md_table(rows: list[dict], limit: int = 25) -> str:
    """Render Insights rows (list of {col: val}) as a markdown table."""
    if not rows:
        return "_no matching log lines_\n"
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    head = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join("---" for _ in cols) + " |\n"
    body = ""
    for r in rows[:limit]:
        body += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    if len(rows) > limit:
        body += f"\n_({len(rows) - limit} more rows in the raw JSON)_\n"
    return head + sep + body


#: Marker appended to any section built from a capped (partial) result set. The
#: producing tools flag truncation in their JSON (``truncated``); the dossier is
#: what an operator actually reads, so it must carry the flag through rather than
#: present a capped series as a full run — the peaks and onsets below a cap are
#: lower bounds, and a saturation onset may be missing entirely.
_PARTIAL = "**PARTIAL — hit the Insights row cap; figures below are lower bounds.**"


def _gib(value: float | None) -> str:
    """Render a fleet-memory peak, distinguishing "not measured" from zero.

    Mirrors ``watch_scheduler``'s formatter rather than importing it: that module
    imports boto3 at module scope, and this assembler is deliberately pure over
    the JSON — no cloud SDK, so it runs anywhere the files do.
    """
    return "not recorded" if value is None else f"{value:.2f} GiB"


def _truncation_banner(sched: dict | None, logs: dict | None) -> list[str]:
    """A top-of-dossier warning naming every capped input, or [] when all complete.

    Repeated at the top because the per-section markers are easy to skim past,
    and a reader who misses the cap draws conclusions ("peaked at 98% CPU, never
    tripped backlog-growth") from a series that simply stopped early.
    """
    capped: list[str] = []
    if sched is not None and sched.get("truncated"):
        capped.append("the scheduler health series")
    for name, q in (logs or {}).get("queries", {}).items():
        if q.get("truncated"):
            capped.append(f"log query `{name}`")
    if not capped:
        return []
    return [
        f"> ⚠️ {_PARTIAL}",
        ">",
        f"> Capped inputs: {', '.join(capped)}. Re-run those over a narrower window "
        "(or a tighter `--stream-prefix`) before treating this dossier as a full-run profile.",
        "",
    ]


def _scheduler_section(sched: dict | None) -> str:
    if sched is None:
        return "### Scheduler\n\n_No scheduler profile supplied (`--scheduler`)._\n"
    profile = sched.get("profile", {})
    if not profile.get("samples"):
        return (
            "### Scheduler\n\n"
            f"_No `scheduler health:` lines in {sched.get('log_group', '?')} for the window "
            "— was the heartbeat plugin attached, and the stream-prefix right?_\n"
        )
    pk = profile["peaks"]
    w = profile["window"]
    mem = "unknown" if pk["mem"] is None else f"{pk['mem']:.0f}%"
    tripped = ", ".join(f"`{k}` @ {v}" for k, v in profile["onsets"].items() if v)
    onsets = tripped or "_none tripped_"
    lines = [
        "### Scheduler",
        "",
    ]
    if sched.get("truncated"):
        lines += [_PARTIAL, ""]
    lines += [
        f"- Window: {w['start']} → {w['end']} ({profile['samples']} health samples)",
        f"- Peak CPU: **{pk['cpu']:.0f}%** · peak mem: **{mem}** · peak loop-lag: **{pk['lag']:.1f}s**",
        f"- Peak backlog (no-worker): **{pk['no_worker']}**",
        f"- Peak workers: **{pk['workers']}** · peak tasks: **{pk['tasks']}**",
        f"- Peak FLEET memory: **{_gib(pk.get('worker_mem_gib'))}** · spilled "
        f"**{_gib(pk.get('worker_spill_gib'))}** · hottest worker "
        f"**{_gib(pk.get('worker_max_gib'))}**"
        + ("" if pk.get("worker_spill_gib") else "  _(no spill — the graph fit the fleet)_"),
        f"- cpu-high ∧ backlog-rising intervals: **{profile['cpu_backlog_cooccurrence']}**",
        f"- Saturation onsets: {onsets}",
        f"- Worker join/exit events: **{len(profile['worker_events'])}**",
    ]
    return "\n".join(lines) + "\n"


def _logs_section(logs: dict | None) -> str:
    if logs is None:
        return "### External services & workers\n\n_No log-query output supplied (`--logs`)._\n"
    out = ["### External services & workers", ""]
    for name, q in logs.get("queries", {}).items():
        out.append(f"**{name}** — {q['description']}")
        out.append("")
        rows = q.get("rows")
        if rows is None:
            out.append("_query failed (permission/timeout) — see stderr from the run_\n")
        else:
            if q.get("truncated"):
                out += [_PARTIAL, ""]
            out.append(_md_table(rows))
        out.append("")
    return "\n".join(out)


def build_dossier(args: argparse.Namespace, sched: dict | None, logs: dict | None) -> str:
    """Merge the scheduler + log-query JSON into the dossier markdown skeleton."""
    title = args.title or f"Ingest run {args.run_id or '(unlabeled)'}"
    meta = [
        f"# {title}",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| run id | {args.run_id or '—'} |",
        f"| zone / year | {args.zone or '—'} / {args.year or '—'} |",
        f"| worker bounds | {args.min_workers or '—'} … {args.max_workers or '—'} |",
    ]
    win = (sched or {}).get("window") or (logs or {}).get("window")
    if win:
        meta.append(f"| window | {win['start']} → {win['end']} |")
    if args.perf_report:
        meta.append(f"| deep-profile artifact | [{args.perf_report}]({args.perf_report}) |")
    meta.append("")

    interpretation = [
        "## Interpretation",
        "",
        "> _Fill in from the sections below. Answer explicitly:_",
        "> - **Bottleneck verdict** — scheduler-bound, worker/external-bound, or headroom left?",
        "> - **Limiting factor** — which signal capped this rung (CPU-sustained, backlog growth, "
        "worker churn, FLEET memory / spill, an external 429/503 curve, S3 SlowDown)?",
        "> - **Next rung** — go higher, hold, or step back; and any config change to try first.",
        "",
    ]

    # Above the Interpretation prompts on purpose: the operator must know a figure
    # is a lower bound BEFORE writing a verdict against it.
    banner = _truncation_banner(sched, logs)

    return "\n".join(
        [
            "\n".join(meta),
            *(["\n".join(banner)] if banner else []),
            "\n".join(interpretation),
            _scheduler_section(sched),
            "",
            _logs_section(logs),
            "---",
            "",
            "_Raw JSON is the system of record: "
            + ", ".join(
                p for p in (f"`{args.scheduler}`" if args.scheduler else "", f"`{args.logs}`" if args.logs else "") if p
            )
            + "._",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Parse args, load the profiling JSON, and emit the dossier."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scheduler", help="watch_scheduler.py --report JSON file")
    parser.add_argument("--logs", help="ingest_log_queries.py JSON file")
    parser.add_argument("--perf-report", help="S3 URI of the distributed performance_report HTML (if captured)")
    parser.add_argument("--run-id")
    parser.add_argument("--zone")
    parser.add_argument("--year")
    parser.add_argument("--min-workers")
    parser.add_argument("--max-workers")
    parser.add_argument("--title")
    parser.add_argument("--out", help="write the dossier here (default: stdout)")
    args = parser.parse_args(argv)

    if not args.scheduler and not args.logs:
        parser.error("supply at least one of --scheduler / --logs")

    try:
        sched, logs = _load_json(args.scheduler), _load_json(args.logs)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    dossier = build_dossier(args, sched, logs)
    if args.out:
        Path(args.out).write_text(dossier)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(dossier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
