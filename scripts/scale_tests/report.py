"""Collate scale-test metrics into a decision-oriented markdown report.

Reads every ``*.jsonl`` under ``<results>/<run-id>/`` and renders the test-plan
§5 decision matrix — one section per ADR-008 decision — plus per-test metric
tables. Missing tests render as "no data (not run)" rather than failing, so a
partial run still produces a useful report.

Run from ``scripts/``::

    uv run python -m scale_tests.report --run-id dev --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger("scale_tests.report")


def _load_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Load every metric row under ``results_dir`` (recursively)."""
    rows: list[dict[str, Any]] = []
    for jsonl in sorted(results_dir.rglob("*.jsonl")):
        for line in jsonl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sel(
    rows: list[dict],
    *,
    test: str,
    phase: str | None = None,
    metric: str | None = None,
    **param_eq: Any,  # noqa: ANN401 — arbitrary param filters matched against row params
) -> list[dict]:
    """Filter rows by top-level test/phase/metric, plus required param equality.

    ``test``/``phase``/``metric`` are top-level row fields; any other keyword is
    matched against the row's ``params`` dict.
    """
    out = []
    for r in rows:
        if r["test"] != test:
            continue
        if phase is not None and r["phase"] != phase:
            continue
        if metric is not None and r["metric"] != metric:
            continue
        params = r.get("params", {})
        if all(params.get(k) == v for k, v in param_eq.items()):
            out.append(r)
    return out


def _fmt(x: float) -> str:
    """Format a metric value compactly."""
    if x == 0:
        return "0"
    if abs(x) >= 1e6 or abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:.3f}"


def _header(rows: list[dict], run_id: str) -> list[str]:
    """Render the run provenance header."""
    stamp = rows[0] if rows else {}
    ic = stamp.get("icechunk", "?")
    zr = stamp.get("zarr", "?")
    sha = stamp.get("git_sha", "?")
    return [
        f"# Scale-test report - run `{run_id}`",
        "",
        f"- backend: **{stamp.get('backend', '?')}**, scale: **{stamp.get('scale', '?')}**",
        f"- icechunk **{ic}**, zarr **{zr}**, git `{sha}`",
        f"- rows: {len(rows)}",
        "",
        "> Contention/latency numbers are only load-bearing on `--backend s3`.",
        "",
    ]


def _per_test_tables(rows: list[dict]) -> list[str]:
    """Render a compact table of every metric grouped by test/phase/metric."""
    out = ["## Per-test metrics", ""]
    tests = sorted({r["test"] for r in rows})
    for test in tests:
        out.append(f"### {test}")
        out.append("")
        out.append("| phase | metric | n | min | median | max | unit |")
        out.append("|---|---|---|---|---|---|---|")
        trows = [r for r in rows if r["test"] == test]
        keys = sorted({(r["phase"], r["metric"], r["unit"]) for r in trows})
        for phase, metric, unit in keys:
            vals = [r["value"] for r in trows if r["phase"] == phase and r["metric"] == metric]
            out.append(
                f"| {phase} | {metric} | {len(vals)} | {_fmt(min(vals))} | "
                f"{_fmt(statistics.median(vals))} | {_fmt(max(vals))} | {unit} |"
            )
        out.append("")
    return out


def _decision_matrix(rows: list[dict]) -> list[str]:
    """Render the ADR-008 decision matrix from measured values."""
    out = ["## Decision matrix (ADR-008)", "", "| Decision | Evidence | Measured | Notes |", "|---|---|---|---|"]

    # D1 — pre-allocate (T3)
    seed = _sel(rows, test="t3", phase="seed_full_axis", metric="objects_listed")
    conflict = _sel(rows, test="t3", phase="conflict_probe")
    d1_measured = "no data"
    if seed:
        outcome = conflict[0]["params"].get("outcome") if conflict else "?"
        d1_measured = f"data chunks=0 confirmed; shift-vs-write conflict={outcome}"
    out.append(f"| D1 pre-alloc | data-chunks==0; shift conflict unresolvable | {d1_measured} | escape hatch only |")

    # D2 — chunk shape (T1 point p95 + throughput by variant)
    d2 = _t1_variant_summary(rows)
    out.append(f"| D2 chunk shape | T1 point p95 by variant | {d2} | pick best weighted variant |")

    # D3 — sharding (T1 sharded vs full)
    out.append(f"| D3 sharding | sharded vs full point p95 | {_t1_shard_vs_full(rows)} | adopt only if reads win |")

    # D4 — split config (T2 year-fill trend)
    out.append(f"| D4 manifest split | T2 per-year commit trend | {_t2_trend(rows)} | rising => icechunk #1600 |")

    # D5 — one repo (T4 snapshot growth + T5 contention at N)
    out.append(
        f"| D5 one repo | T4 snapshot growth + T5 contention | {_t4_t5_summary(rows)} | kill: >2x serial or storms |"
    )

    # D7 — GC (T6)
    out.append(f"| D7 GC/hygiene | T6 GC objects/s + reclaimed | {_t6_summary(rows)} | extrapolate to 10^8 objects |")

    out.append("")
    return out


def _t1_variant_summary(rows: list[dict]) -> str:
    """Best cold point-read p95 per variant (lower is better)."""
    pts = _sel(rows, test="t1", metric="read_p95_ms", cache="cold")
    if not pts:
        return "no data"
    by_variant: dict[str, list[float]] = {}
    for r in pts:
        by_variant.setdefault(r["variant"], []).append(r["value"])
    parts = [f"{v}={_fmt(min(vals))}ms" for v, vals in sorted(by_variant.items())]
    return "; ".join(parts)


def _t1_shard_vs_full(rows: list[dict]) -> str:
    """Compare sharded vs full point p95 if both present."""

    def best(variant: str) -> float | None:
        vals = [
            r["value"] for r in _sel(rows, test="t1", metric="read_p95_ms", cache="cold") if r["variant"] == variant
        ]
        return min(vals) if vals else None

    sharded, full = best("c256_sharded"), best("c256_full")
    if sharded is None or full is None:
        return "no data (need both variants)"
    verdict = "sharded wins" if sharded < full else "full wins"
    return f"sharded={_fmt(sharded)}ms vs full={_fmt(full)}ms -> {verdict}"


def _t2_trend(rows: list[dict]) -> str:
    """Report the per-year commit-wall trend (first vs last year filled)."""
    trend = _sel(rows, test="t2", phase="year_fill_trend", metric="commit_wall_s")
    if not trend:
        return "no data"
    ordered = sorted(trend, key=lambda r: r["params"].get("year", 0))
    first, last = ordered[0]["value"], ordered[-1]["value"]
    direction = "rising" if last > first * 1.5 else "flat"
    return f"commit {_fmt(first)}s->{_fmt(last)}s ({direction})"


def _t4_t5_summary(rows: list[dict]) -> str:
    """Snapshot growth (T4) and worst contention wall/retries (T5)."""
    snaps = _sel(rows, test="t4", phase="scale_groups", metric="snapshot_bytes")
    snap_str = "no T4"
    if snaps:
        ordered = sorted(snaps, key=lambda r: r["params"].get("n_groups", 0))
        lo, hi = _fmt(ordered[0]["value"]), _fmt(ordered[-1]["value"])
        ng = ordered[-1]["params"].get("n_groups")
        snap_str = f"snapshot {lo}->{hi}B @{ng}g"
    # T5 emits per-N phases (contention_n2, ...); select by test across all of them.
    retries = _sel(rows, test="t5", metric="retries")
    ret_str = "no T5"
    if retries:
        ret_str = f"max retries={_fmt(max(r['value'] for r in retries))}"
    return f"{snap_str}; {ret_str}"


def _t6_summary(rows: list[dict]) -> str:
    """GC objects/s and bytes reclaimed."""
    deleted = _sel(rows, test="t6", phase="expire_and_gc", metric="objects_deleted")
    reclaimed = _sel(rows, test="t6", phase="expire_and_gc", metric="bytes_reclaimed")
    if not deleted:
        return "no data"
    per_s = deleted[0]["params"].get("per_s", "?")
    bytes_r = reclaimed[0]["value"] if reclaimed else 0
    return f"{int(deleted[0]['value'])} objs @ {per_s}/s, {_fmt(bytes_r)}B reclaimed"


def build_report(results_dir: Path, run_id: str) -> str:
    """Assemble the full markdown report string."""
    rows = _load_rows(results_dir)
    if not rows:
        return f"# Scale-test report — run `{run_id}`\n\nNo metrics found under {results_dir}.\n"
    lines = _header(rows, run_id) + _decision_matrix(rows) + _per_test_tables(rows)
    return "\n".join(lines)


def main() -> int:
    """Parse args, build the report, write ``report.md`` under the results dir."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", default=None, help="Local results root (default ./scale_test_results).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    base = Path(args.results_dir) if args.results_dir else Path.cwd() / "scale_test_results"
    results_dir = base / args.run_id
    report = build_report(results_dir, args.run_id)
    out_path = results_dir / "report.md"
    out_path.write_text(report)
    logger.info("Wrote %s (%d chars)", out_path, len(report))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
