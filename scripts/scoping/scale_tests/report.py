"""Collate scale-test metrics into a decision-oriented markdown report.

Reads every ``*.jsonl`` under ``<results>/<run-id>/`` and renders the
decision matrix — one section per ADR-008 decision — plus per-test metric
tables. Missing tests render as "no data (not run)" rather than failing, so a
partial run still produces a useful report.

Run from the REPOSITORY ROOT::

    uv run python -m scripts.scoping.scale_tests.report --run-id dev --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

from scripts.scoping.scale_tests.variants import ACCEPTED

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
    """Render the run provenance header: one line per environment, naming its phases when there are several.

    A run can be repeated from another client's environment under the same run id, so the
    provenance is each row's, never the first row's for all.
    """
    stamp = rows[0] if rows else {}
    envs: dict[tuple[str, str, str], set[str]] = {}
    for r in rows:
        envs.setdefault((r.get("icechunk", "?"), r.get("zarr", "?"), r.get("git_sha", "?")), set()).add(r["phase"])
    lines = [
        f"- icechunk **{ic}**, zarr **{zr}**, git `{sha}`" + (f": {', '.join(sorted(ph))}" if len(envs) > 1 else "")
        for (ic, zr, sha), ph in sorted(envs.items())
    ]
    return [
        f"# Scale-test report - run `{run_id}`",
        "",
        f"- backend: **{stamp.get('backend', '?')}**, scale: **{stamp.get('scale', '?')}**",
        *lines,
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

    # D2 — chunk shape (T1 point p95+p50 ranked by variant)
    d2 = _t1_variant_summary(rows)
    out.append(f"| D2 chunk shape | T1 point p95/p50 by variant | {d2} | 256+full band; smaller=faster/point |")

    # D3 — sharding: prefer the T8 settlement (write ratio, wire bytes, scattered
    # p95) and apply the decision rule; fall back to the T1 sweep if T8 absent.
    out.append(f"| D3 sharding | T8 write/wire/scattered (else T1) | {_d3_summary(rows)} | see ADR-008 D3 |")

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


def _t1_best_cold(rows: list[dict], variant: str, metric: str) -> float | None:
    """Best (min) cold-cache value of ``metric`` for ``variant`` in T1."""
    vals = [r["value"] for r in _sel(rows, test="t1", metric=metric, cache="cold") if r["variant"] == variant]
    return min(vals) if vals else None


def _phase_wall(rows: list[dict], test: str, phase: str) -> float | None:
    """The ``wall_s`` recorded for a (test, phase) — e.g. a build phase."""
    vals = [r["value"] for r in _sel(rows, test=test, phase=phase, metric="wall_s")]
    return min(vals) if vals else None


def _t1_variant_summary(rows: list[dict]) -> str:
    """Rank variants by cold point-read p95 (with p50), lowest-latency first.

    p95 is the conservative headline; p50 exposes the common-case gap (sharding's
    large shards keep nearby points warm). Read amplification and ref-count are
    weighed in the ADR, not squeezed into this cell.
    """
    variants = sorted({r["variant"] for r in _sel(rows, test="t1", metric="read_p95_ms", cache="cold")})
    if not variants:
        return "no data"
    scored = [(_t1_best_cold(rows, v, "read_p95_ms"), v, _t1_best_cold(rows, v, "read_p50_ms")) for v in variants]
    scored.sort(key=lambda t: t[0] if t[0] is not None else float("inf"))
    parts = [f"{v} p95={_fmt(p95)}/p50={_fmt(p50)}ms" for p95, v, p50 in scored]
    return f"best={scored[0][1]}; " + "; ".join(parts)


def _t1_shard_vs_full(rows: list[dict]) -> str:
    """Compare c256_sharded vs c256_full on p95, p50, and write cost.

    The p95-only "full wins" call is misleading (the gap is noise); this reports
    the fuller picture so the D3 decision isn't made on the tail latency alone.
    """
    p95_s, p95_f = _t1_best_cold(rows, "c256_sharded", "read_p95_ms"), _t1_best_cold(rows, "c256_full", "read_p95_ms")
    p50_s, p50_f = _t1_best_cold(rows, "c256_sharded", "read_p50_ms"), _t1_best_cold(rows, "c256_full", "read_p50_ms")
    if p95_s is None or p95_f is None:
        return "no data (need both variants)"
    build_s = _phase_wall(rows, "t1", "build_c256_sharded")
    build_f = _phase_wall(rows, "t1", "build_c256_full")
    write = f"write {build_s / build_f:.1f}x" if build_s and build_f else "write ?x"
    return (
        f"p95 {_fmt(p95_s)} vs {_fmt(p95_f)}ms (~tie); "
        f"p50 {_fmt(p50_s)} vs {_fmt(p50_f)}ms; {write} slower; ~64x fewer objects"
    )


def _t8_build_wall(rows: list[dict], mode: str) -> float | None:
    """T8 write_alignment build wall for a mode (full / sharded_* )."""
    hits = _sel(rows, test="t8", phase="write_alignment", metric="wall_s", kind="build", mode=mode)
    return hits[0]["value"] if hits else None


def _t8_by_variant(rows: list[dict], phase: str, metric: str, variant: str) -> float | None:
    """T8 value for a metric where the row's (top-level) variant matches."""
    hits = [r for r in _sel(rows, test="t8", phase=phase, metric=metric) if r.get("variant") == variant]
    return hits[0]["value"] if hits else None


def _d3_summary(rows: list[dict]) -> str:
    """Apply the D3 decision rule to T8 (write <=1.5x and scattered p95 <=1.2x).

    Prefers the land-masked aligned writer; falls back to the dense aligned
    build, then to the T1 sweep if T8 is absent entirely.
    """
    build_full = _t8_build_wall(rows, "full")
    mode = "sharded_masked" if _t8_build_wall(rows, "sharded_masked") is not None else "sharded_aligned"
    build_aligned = _t8_build_wall(rows, mode)
    if build_full is None or build_aligned is None:
        return _t1_shard_vs_full(rows)  # T8 not run — fall back to the T1 sweep

    wire_s = _t8_by_variant(rows, "bytes_on_wire", "bytes_fetched", "c256_sharded")
    wire_f = _t8_by_variant(rows, "bytes_on_wire", "bytes_fetched", "c256_full")
    scat_s = _t8_by_variant(rows, "scattered_reads", "read_p95_ms", "c256_sharded")
    scat_f = _t8_by_variant(rows, "scattered_reads", "read_p95_ms", "c256_full")

    ratio = build_aligned / build_full
    write_ok = ratio <= 1.5
    read_ok = scat_s is not None and scat_f is not None and scat_s <= 1.2 * scat_f
    # E1 wire-bytes is a mandatory gate, fail-closed: sharding must not fetch more
    # bytes than full, and a run missing the E1 measurement can't earn ADOPT.
    wire_ok = wire_s is not None and wire_f is not None and wire_s <= wire_f
    verdict = "ADOPT sharding" if (write_ok and read_ok and wire_ok) else "ship c256_full"

    parts = []
    if wire_s is not None and wire_f is not None:
        parts.append(f"wire {_fmt(wire_s / 1e6)} vs {_fmt(wire_f / 1e6)}MB/pt")
    parts.append(f"{mode.replace('sharded_', '')}-write {ratio:.2f}x full")
    if scat_s is not None and scat_f is not None:
        parts.append(f"scattered p95 {_fmt(scat_s)} vs {_fmt(scat_f)}ms")
    return "; ".join(parts) + f" -> {verdict}"


def _t2_trend(rows: list[dict]) -> str:
    """Report the per-year commit-wall trend (first vs last year filled)."""
    trend = _sel(rows, test="t2", phase="year_fill_trend", metric="commit_wall_s")
    if not trend:
        return "no data"
    # Fills are emitted 2025 -> 2017 (campaign order), so sort DESCENDING to keep
    # emission order: first == first-filled (2025), last == last-filled (2017).
    # An ascending sort would swap the endpoints and mislabel a rising
    # accumulate-refs commit-time trend as flat.
    ordered = sorted(trend, key=lambda r: r["params"].get("year", 0), reverse=True)
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


#: T9 read columns: (label, workload, dims). Each cell is requests / MB for the embeddings array; a column renders
#: only where the run measured it, so each study's tables carry its own widths and windows.
_T9_READS = (
    ("point 4", "point", 4),
    ("point 8", "point", 8),
    ("maptile 4", "maptile", 4),
    ("maptile 8", "maptile", 8),
    ("viewport 4", "viewport", 4),
    ("viewport 8", "viewport", 8),
    ("point 16", "point", 16),
    ("point 128", "point", 128),
    ("tile 16", "tile", 16),
    ("shard 16", "shard", 16),
    ("bulk 16", "bulk", 16),
    ("bulk 64", "bulk", 64),
    ("bulk 128", "bulk", 128),
)


def _t9_value(rows: list[dict], metric: str, variant: str, **params: Any) -> float | None:  # noqa: ANN401
    """Median of one T9 metric for one arm, or ``None``."""
    vals = [r["value"] for r in _sel(rows, test="t9", metric=metric, **params) if r.get("variant") == variant]
    return statistics.median(vals) if vals else None


def _t9_tables(rows: list[dict]) -> list[str]:
    """Render T9's write table, one request-plan table per client, and the Dask table, each from its phase's rows."""
    if not _sel(rows, test="t9"):
        return []
    arms = sorted({r["variant"] for r in _sel(rows, test="t9", phase="build", metric="encode_cpu_s")})
    out = ["## T9 — v2 layout arms"]

    def stored(arm: str) -> float:
        rs = _sel(rows, test="t9", metric="bytes_written", array="embeddings")
        return sum(r["value"] for r in rs if r["variant"] == arm)

    # A compression ratio needs the uncompressed sibling of the same geometry; a study without one is compared
    # with the control and, when it ran, the accepted layout instead.
    raws = {a: f"{'_'.join(a.split('_')[:2])}_raw" for a in arms}
    has_raw, has_accepted = any(r in arms for r in raws.values()), ACCEPTED.name in arms
    cols = ["embeddings MB", *(["ratio"] if has_raw else []), "vs C0", *(["vs B2f"] if has_accepted else [])]
    if arms:
        out += ["", "| arm | " + " | ".join([*cols, "encode CPU s/shard"]) + " |", "|---|" + "---|" * (len(cols) + 1)]
    for a in arms:
        cells = [f"{stored(a) / 1e6:.0f}"]
        if has_raw:
            cells.append(f"{stored(raws[a]) / stored(a):.2f}" if stored(a) and stored(raws[a]) else "?")
        cells.append(_vs(stored(a), stored("v2_c0")))
        if has_accepted:
            cells.append(_vs(stored(a), stored(ACCEPTED.name)))
        out.append(f"| {a} | " + " | ".join([*cells, _fmt(_t9_value(rows, "encode_cpu_s", a) or 0)]) + " |")
    # One table per request-plan phase: each client, and the direct-read client of the narrow study.
    for phase in sorted({r["phase"] for r in _sel(rows, test="t9", metric="get_requests")} - {""}):
        if not phase.startswith("requests_"):
            continue
        ran = {(r["params"].get("workload"), r["params"].get("dims")) for r in _sel(rows, test="t9", phase=phase)}
        reads = [c for c in _T9_READS if (c[1], c[2]) in ran]
        out += [
            "",
            f"### Requests / MB fetched, {phase.removeprefix('requests_')}",
            "",
            "| arm | " + " | ".join(c[0] for c in reads) + " |",
            "|---|" + "---|" * len(reads),
        ]
        for a in sorted({r["variant"] for r in _sel(rows, test="t9", phase=phase, metric="get_requests")}):
            cells = []
            for _, workload, dims in reads:
                # The Dask phase records the same workload; the request plan is the requests phase's.
                kw = {"phase": phase, "workload": workload, "dims": dims, "array": "embeddings"}
                n = _t9_value(rows, "get_requests", a, **kw)
                b = _t9_value(rows, "bytes_fetched", a, **kw)
                cells.append("—" if n is None or b is None else f"{n:.0f} / {b / 1e6:.1f}")
            out.append(f"| {a} | " + " | ".join(cells) + " |")
    for phase in sorted({r["phase"] for r in _sel(rows, test="t9", metric="dask_tasks")}):
        out += [
            "",
            f"### Dask, bulk 16-d pull, {phase.removeprefix('dask_')}",
            "",
            "| arm | tasks default | tasks shard-aligned | requests default | requests shard-aligned |",
            "|---|---|---|---|---|",
        ]
        for a in sorted({r["variant"] for r in _sel(rows, test="t9", phase=phase, metric="dask_tasks")}):
            t = [_t9_value(rows, "dask_tasks", a, phase=phase, chunks=c) for c in ("default", "shard_aligned")]
            q = [_t9_value(rows, "get_requests", a, phase=phase, chunks=c) for c in ("default", "shard_aligned")]
            out.append(f"| {a} | " + " | ".join("—" if v is None else f"{v:.0f}" for v in (*t, *q)) + " |")
    return [*out, ""]


#: T10 read columns: (label, workload, dims). Points report the p50 in ms; the rest, wall seconds.
_T10_READS = (
    ("point 16 p50", "point", 16),
    ("point 128 p50", "point", 128),
    ("tile 16", "tile", 16),
    ("bulk 16", "bulk", 16),
    ("bulk 128", "bulk", 128),
    ("8x8 pull 16", "pull8", 16),
    ("8x8 pull 64", "pull8", 64),
    ("8x8 pull 128", "pull8", 128),
    ("dask default", "dask_default", 16),
    ("dask aligned", "dask_aligned", 16),
    ("maptile 4", "maptile", 4),
    ("maptile 8", "maptile", 8),
    ("viewport 4", "viewport", 4),
    ("viewport 8", "viewport", 8),
)


def _t10_value(rows: list[dict], metric: str, variant: str, **params: Any) -> float | None:  # noqa: ANN401
    """Median over passes of one T10 metric for one arm, or ``None``."""
    vals = [r["value"] for r in _sel(rows, test="t10", metric=metric, **params) if r.get("variant") == variant]
    return statistics.median(vals) if vals else None


def _unreadable(rows: list[dict], phase: str) -> list[dict]:
    """A phase's ``unreadable`` rows: a workload this client could not open on an arm, recorded instead of a time."""
    return [r for r in _sel(rows, test="t10", phase=phase, metric="unreadable") if r["variant"]]


def _is_unreadable(unread: list[dict], arm: str, workload: str, dims: int, config: str | None = None) -> bool:
    """Whether ``arm``'s ``workload`` at ``dims`` (and reader ``config``, when given) was recorded unreadable."""
    return any(
        r["variant"] == arm
        and (r["params"]["workload"], r["params"]["dims"]) == (workload, dims)
        and (config is None or r["params"].get("zarr_config", "") == config)
        for r in unread
    )


def _unreadable_notes(unread: list[dict]) -> list[str]:
    """One line under a table per recorded reason, naming the arms it stopped."""
    by_reason: dict[str, set[str]] = {}
    for r in unread:
        by_reason.setdefault(r["params"]["reason"], set()).add(r["variant"])
    notes = [f"> **unreadable** ({', '.join(sorted(arms))}): {reason}" for reason, arms in sorted(by_reason.items())]
    return [*notes, ""] if notes else []


def _vs(value: float, base: float | None) -> str:
    """``value / base`` to two places, or a dash when the control did not run, never a ratio to 1."""
    return f"{value / base:.2f}" if base else "—"


def _t10_tables(rows: list[dict]) -> list[str]:
    """Render T10's assembly table (ratios to the control) and one cold-read table per client."""
    out: list[str] = []
    built = sorted({r["variant"] for r in _sel(rows, test="t10", phase="build", metric="encode_cpu_s") if r["variant"]})
    if built:
        cpu0, gb0 = (_t10_value(rows, m, "v2_c0", phase="build") for m in ("encode_cpu_s", "bytes_written"))
        # Objects and the manifest a direct reader downloads first, where the run recorded them.
        manifests = bool(_sel(rows, test="t10", phase="build", metric="manifest_bytes"))
        extra = " objects | manifest bytes per object |" if manifests else ""
        out += [
            "## T10 — bench store build, encode CPU per shard",
            "",
            "| arm | encode CPU s | vs C0 | stored GB | vs C0 |" + extra,
            "|---|---|---|---|---|" + "---|---|" * manifests,
        ]
        for a in built:
            cpu = _t10_value(rows, "encode_cpu_s", a, phase="build") or 0
            gb = _t10_value(rows, "bytes_written", a, phase="build") or 0
            line = f"| {a} | {cpu:.2f} | {_vs(cpu, cpu0)} | {gb / 1e9:.1f} | {_vs(gb, gb0)} |"
            if manifests:
                manifest_rows = _sel(rows, test="t10", phase="build", metric="manifest_bytes")
                m = next(r for r in manifest_rows if r["variant"] == a)
                objects = m["params"]["objects"]
                line += f" {objects} | {m['value'] / objects:.0f} |"
            out.append(line)
        out.append("")
    arms = sorted({r["variant"] for r in _sel(rows, test="t10", phase="assemble", metric="wall_s") if r["variant"]})
    if arms:
        base = {k: _t10_value(rows, "encode_cpu_s", "v2_c0", phase="assemble", kind=k) for k in ("worker", "write")}
        out += [
            "## T10 — assembly code: CPU per staged tile, wall per run (median of passes)",
            "",
            "| arm | worker CPU s | vs C0 | write CPU s | vs C0 | wall s, whole run | stored GB |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in arms:
            w, wr = (_t10_value(rows, "encode_cpu_s", a, phase="assemble", kind=k) or 0 for k in ("worker", "write"))
            wall = _t10_value(rows, "wall_s", a, phase="assemble") or 0
            gb = (_t10_value(rows, "bytes_written", a, phase="assemble") or 0) / 1e9
            out.append(
                f"| {a} | {w:.2f} | {_vs(w, base['worker'])} | {wr:.2f} | {_vs(wr, base['write'])} "
                f"| {wall:.0f} | {gb:.1f} |"
            )
        out.append("")
    reads = [r for r in _sel(rows, test="t10", metric="get_requests") if r["phase"].startswith("read_")]
    clients = sorted({r["params"]["client"] for r in reads})  # a points- or tune-only run has no read table
    for client in clients:
        phase = f"read_{client}"
        # A workload this client could not open is a result too, so its rows count as run.
        unread = _unreadable(rows, phase)
        measured = [*_sel(rows, test="t10", phase=phase, metric="get_requests"), *unread]
        read_arms = sorted({r["variant"] for r in measured})
        # Only the workloads this run's sweep scheduled: the sweep has been trimmed between runs.
        ran = {(r["params"].get("workload"), r["params"].get("dims")) for r in measured}
        cols = [c for c in _T10_READS if (c[1], c[2]) in ran]
        out += [
            f"## T10 — cold reads from S3, {client} (median of passes; wire MB in brackets)",
            "",
            "| arm | " + " | ".join(c[0] for c in cols) + " |",
            "|---|" + "---|" * len(cols),
        ]
        for a in read_arms:
            cells = []
            for _, workload, dims in cols:
                kw = {"phase": phase, "workload": workload, "dims": dims}
                v = _t10_value(rows, "read_p50_ms" if workload == "point" else "wall_s", a, **kw)
                wire = _t10_value(rows, "bytes_fetched", a, method="wire", **kw)
                unit = "ms" if workload == "point" else "s"
                cells.append(
                    ("unreadable" if _is_unreadable(unread, a, workload, dims) else "—")
                    if v is None
                    else f"{v:.0f} {unit} ({(wire or 0) / 1e6:.0f})"
                    if unit == "ms"
                    else f"{v:.1f} s ({(wire or 0) / 1e6:.0f})"
                )
            out.append(f"| {a} | " + " | ".join(cells) + " |")
        out.append("")
        out += _unreadable_notes(unread)
        out += _vs_accepted(rows, phase, read_arms, cols)
    out += _mobile_tables(rows)
    for phase in sorted({r["phase"] for r in _sel(rows, test="t10", metric="read_p50_lo_ms")}):
        # Every phase that bootstrapped its point reads, `points` and the narrow study's `read`. One interval
        # per pass: each pass bootstraps its own probes, and the median of two passes' endpoints is not an
        # interval for the passes combined.
        passes = sorted({r["params"]["pass"] for r in _sel(rows, test="t10", phase=phase, metric="read_p50_ms")})
        out += [
            f"## T10 — point reads, p50 [95% interval] by pass ({', '.join(map(str, passes))}), "
            f"{phase.removeprefix('points_')}",
            "",
            "| arm | point 16 ms | point 128 ms | open s |",
            "|---|---|---|---|",
        ]
        for a in sorted({r["variant"] for r in _sel(rows, test="t10", phase=phase, metric="read_p50_ms")}):
            cells = []
            for dims in (16, 128):
                per_pass = []
                for p in passes:
                    kw: dict[str, Any] = {"phase": phase, "dims": dims, "pass": p}
                    p50, lo, hi = (
                        _t10_value(rows, m, a, **kw) for m in ("read_p50_ms", "read_p50_lo_ms", "read_p50_hi_ms")
                    )
                    per_pass.append(f"{p50:.0f} [{lo:.0f}, {hi:.0f}]" if p50 and lo and hi else "—")
                cells.append(" / ".join(per_pass))
            open_s = _t10_value(rows, "open_wall_s", a, phase=phase)
            out.append(f"| {a} | {' | '.join(cells)} | {open_s or 0:.2f} |")
        out.append("")
    verified = _sel(rows, test="t10", metric="mismatched_arrays")
    if verified:
        out += ["## T10 — lossless check against C0", "", "| arm | stores | shards | mismatched array-shards |"]
        out.append("|---|---|---|---|")
        for r in sorted(verified, key=lambda r: (r["params"]["prefix"], r["variant"])):
            out.append(f"| {r['variant']} | {r['params']['prefix']} | {r['params']['shards']} | {r['value']:.0f} |")
        out.append("")

    def setting(r: dict) -> tuple:
        return r["params"]["workload"], r["params"]["dims"], r["params"].get("zarr_config", "")

    # One table per client, as for reads and points: the zarr versions read shards differently.
    for phase in sorted({r["phase"] for r in rows if r["test"] == "t10" and r["phase"].startswith("tune_")}):
        tune = [r for r in _sel(rows, test="t10", phase=phase, metric="wall_s") if r["variant"]]
        unread = _unreadable(rows, phase)
        configs = sorted({setting(r) for r in [*tune, *unread]})
        out += [
            f"## T10 — reader settings, {phase.removeprefix('tune_')} (wall s, median of passes)",
            "",
            "| arm | " + " | ".join(f"{w} {d} {c or 'default'}" for w, d, c in configs) + " |",
            "|---|" + "---|" * len(configs),
        ]
        for a in sorted({r["variant"] for r in [*tune, *unread]}):
            cells = []
            for config in configs:
                vals = [r["value"] for r in tune if r["variant"] == a and setting(r) == config]
                missing = "unreadable" if _is_unreadable(unread, a, config[0], config[1], config[2]) else "—"
                cells.append(f"{statistics.median(vals):.1f}" if vals else missing)
            out.append(f"| {a} | " + " | ".join(cells) + " |")
        out.append("")
        out += _unreadable_notes(unread)
    return out


def _vs_accepted(rows: list[dict], phase: str, arms: list[str], cols: list[tuple[str, str, int]]) -> list[str]:
    """A narrow-study read phase's times as ratios to the accepted layout, run in the same session.

    The narrow study is the one that reads the mobile windows; earlier runs that held the accepted
    layout as one finalist among several render as they were archived.
    """
    if ACCEPTED.name not in arms or not any(w in ("maptile", "viewport") for _, w, _ in cols):
        return []

    def time(arm: str, workload: str, dims: int) -> float | None:
        metric = "read_p50_ms" if workload == "point" else "wall_s"
        return _t10_value(rows, metric, arm, phase=phase, workload=workload, dims=dims)

    out = [
        f"### The same, as a ratio to the accepted layout ({ACCEPTED.name})",
        "",
        "| arm | " + " | ".join(c[0] for c in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for a in arms:
        cells = []
        for _, workload, dims in cols:
            v = time(a, workload, dims)
            cells.append("—" if v is None else _vs(v, time(ACCEPTED.name, workload, dims)))
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    return [*out, ""]


def _mobile_tables(rows: list[dict]) -> list[str]:
    """One table per shaped network: each mobile read's wall, requests and MB, under the link the probe measured."""
    out: list[str] = []
    for phase in sorted({r["phase"] for r in rows if r["test"] == "t10" and r["phase"].startswith("mobile_")}):
        rtt = _sel(rows, test="t10", phase=phase, metric="probe_rtt_ms")
        mbps = _sel(rows, test="t10", phase=phase, metric="probe_mbps")
        link = f"{rtt[0]['value']:.0f} ms, {mbps[0]['value']:.1f} Mbit/s measured" if rtt and mbps else "unshaped"
        tagged = [r for r in _sel(rows, test="t10", phase=phase, metric="wall_s") if r["variant"]]  # not the phase's
        unread = _unreadable(rows, phase)
        cols = sorted({(r["params"]["workload"], r["params"]["dims"]) for r in [*tagged, *unread]})
        out += [
            f"## T10 — mobile reads, {phase.removeprefix('mobile_')} "
            f"({link}; wall s / requests / MB, median of passes)",
            "",
            "| arm | " + " | ".join(f"{w} {d}" for w, d in cols) + " | first read s |",
            "|---|" + "---|" * (len(cols) + 1),
        ]
        for a in sorted({r["variant"] for r in [*tagged, *unread]}):
            cells = []
            for workload, dims in cols:
                kw = {"phase": phase, "workload": workload, "dims": dims}
                wall, n = _t10_value(rows, "wall_s", a, **kw), _t10_value(rows, "get_requests", a, **kw)
                mb = (_t10_value(rows, "bytes_fetched", a, method="store", **kw) or 0) / 1e6
                missing = "unreadable" if _is_unreadable(unread, a, workload, dims) else "—"
                cells.append(missing if wall is None else f"{wall:.2f} / {n or 0:.0f} / {mb:.1f}")
            first = _t10_value(rows, "first_read_s", a, phase=phase)
            out.append(f"| {a} | " + " | ".join(cells) + f" | {first or 0:.2f} |")
        out.append("")
        out += _unreadable_notes(unread)
    return out


def build_report(results_dir: Path, run_id: str) -> str:
    """Assemble the full markdown report string."""
    rows = _load_rows(results_dir)
    if not rows:
        return f"# Scale-test report — run `{run_id}`\n\nNo metrics found under {results_dir}.\n"
    # The ADR-008 matrix and the generic per-test stats describe T0-T8; a v2 layout run carries its own tables.
    v1 = [r for r in rows if r["test"] not in ("t9", "t10")]
    lines = _header(rows, run_id) + (_decision_matrix(v1) if v1 else []) + _t9_tables(rows) + _t10_tables(rows)
    return "\n".join(lines + (_per_test_tables(v1) if v1 else []))


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
