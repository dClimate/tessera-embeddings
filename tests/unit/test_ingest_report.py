"""Offline tests for the ingest dossier assembler (scripts/profiling/ingest).

``build_dossier`` is a pure merge over the profiling JSON; these tests pin that
it renders each section, tolerates a missing input, and surfaces the operator's
Interpretation prompts — no AWS involved.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "profiling" / "ingest" / "report.py"


def _load():
    spec = importlib.util.spec_from_file_location("ingest_report", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rep = _load()

_SCHED = {
    "log_group": "/ecs/tessera/dask",
    "window": {"start": "2026-07-24T18:00:00Z", "end": "2026-07-24T20:00:00Z"},
    "profile": {
        "samples": 40,
        "window": {"start": "2026-07-24T18:00:00Z", "end": "2026-07-24T20:00:00Z"},
        "peaks": {"cpu": 98.0, "mem": 74.0, "lag": 3.2, "no_worker": 220, "workers": 500, "tasks": 90000},
        "onsets": {
            "cpu-sustained": "2026-07-24T19:10:00Z",
            "mem-high": None,
            "backlog-growth": "2026-07-24T19:12:00Z",
            "loop-lag": None,
        },
        "cpu_backlog_cooccurrence": 7,
        "worker_events": [{"ts": "2026-07-24T18:05:00Z", "from": 1, "to": 250, "delta": 249}],
    },
}

_LOGS = {
    "log_group": "/ecs/tessera/dask",
    "window": {"start": "2026-07-24T18:00:00Z", "end": "2026-07-24T20:00:00Z"},
    "queries": {
        "http_retries_by_service": {
            "description": "retries by service",
            "rows": [{"service": "CMR", "retries": "418"}, {"service": "STAC", "retries": "12"}],
        },
        "s3_slowdown": {"description": "503s per bin", "rows": []},
        "worker_exit_reasons": {"description": "raw exits", "rows": None},
    },
}


def _args(**over):
    base = dict(
        scheduler="sched.json", logs="logs.json", perf_report="s3://b/perf.html",
        run_id="abc123", zone="32633", year="2025", min_workers="1", max_workers="500", title=None, out=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_full_dossier_has_all_sections():
    md = rep.build_dossier(_args(), _SCHED, _LOGS)
    assert "# Ingest run abc123" in md
    assert "## Interpretation" in md and "Bottleneck verdict" in md
    # scheduler peaks + onsets rendered
    assert "Peak CPU: **98%**" in md
    assert "`cpu-sustained` @ 2026-07-24T19:10:00Z" in md
    assert "Worker join/exit events: **1**" in md
    # log query tables + states rendered
    assert "http_retries_by_service" in md and "| CMR | 418 |" in md
    assert "_no matching log lines_" in md  # s3_slowdown rows == []
    assert "query failed" in md  # worker_exit_reasons rows == None
    # perf-report link + zone metadata
    assert "s3://b/perf.html" in md and "32633 / 2025" in md


def test_dossier_tolerates_missing_scheduler():
    md = rep.build_dossier(_args(scheduler=None), None, _LOGS)
    assert "_No scheduler profile supplied" in md
    assert "http_retries_by_service" in md


def test_dossier_notes_empty_scheduler_series():
    empty = {"log_group": "/ecs/tessera/dask", "profile": {"samples": 0}}
    md = rep.build_dossier(_args(logs=None), empty, None)
    assert "No `scheduler health:` lines" in md
