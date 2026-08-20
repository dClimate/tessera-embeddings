"""Offline tests for the ingest dossier assembler.

``build_dossier`` is a pure merge over the profiling JSON; these tests pin that
it renders each section, tolerates a missing input, and surfaces the operator's
Interpretation prompts — no AWS involved.
"""

from __future__ import annotations

import argparse

from tessera_embeddings.profiling.ingest import report as rep

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
        scheduler="sched.json",
        logs="logs.json",
        perf_report="s3://b/perf.html",
        run_id="abc123",
        zone="32633",
        year="2025",
        min_workers="1",
        max_workers="500",
        title=None,
        out=None,
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


class TestBadInputFiles:
    """A mistyped path or a stderr-polluted file is a message, not a traceback."""

    def test_missing_file_exits_nonzero_with_a_message(self, tmp_path, capsys):
        rc = rep.main(["--scheduler", str(tmp_path / "nope.json")])
        assert rc == 1
        assert "cannot read" in capsys.readouterr().err

    def test_malformed_json_names_the_likely_cause(self, tmp_path, capsys):
        bad = tmp_path / "sched.json"
        bad.write_text("WARNING: query hit the cap\n{}")
        rc = rep.main(["--scheduler", str(bad)])
        assert rc == 1
        assert "not valid JSON" in capsys.readouterr().err


class TestTruncationIsNeverSilent:
    """A capped input must never be presented as a full-run profile.

    The producing tools flag the Insights row cap in their JSON, but the dossier
    is what an operator actually reads and writes a verdict against — so if it
    dropped the flag, a series that merely stopped early would be read as "peaked
    at 98% CPU, never tripped backlog-growth". Both the banner and the
    per-section markers are pinned here.
    """

    def test_untruncated_dossier_has_no_partial_warning(self):
        md = rep.build_dossier(_args(), _SCHED, _LOGS)
        assert "PARTIAL" not in md

    def test_truncated_scheduler_series_is_flagged(self):
        md = rep.build_dossier(_args(logs=None), {**_SCHED, "truncated": True}, None)
        assert "PARTIAL" in md
        assert "the scheduler health series" in md  # named in the banner
        # Marked at the figures themselves, not only in the banner.
        assert md.index("PARTIAL") < md.index("## Interpretation")
        assert md.count("PARTIAL") >= 2

    def test_truncated_log_query_is_flagged_and_named(self):
        logs = {
            **_LOGS,
            "queries": {
                "http_retries_by_service": {
                    **_LOGS["queries"]["http_retries_by_service"],
                    "truncated": True,
                },
                "s3_slowdown": {**_LOGS["queries"]["s3_slowdown"], "truncated": False},
            },
        }
        md = rep.build_dossier(_args(scheduler=None), None, logs)
        assert "log query `http_retries_by_service`" in md
        assert "s3_slowdown" in md and "log query `s3_slowdown`" not in md

    def test_missing_truncated_key_is_treated_as_complete(self):
        """Older JSON (written before the flag existed) must still render."""
        assert "PARTIAL" not in rep.build_dossier(_args(), _SCHED, _LOGS)


class TestFleetMemoryReachesTheDossier:
    """Worker memory is the SECOND failure mode; the dossier must show it.

    The heartbeat carries fleet memory and watch_scheduler's own --markdown
    reports it, but the dossier is what an operator reads to write a bottleneck
    verdict — and the campaign cell that motivated the metric died of worker
    memory with every scheduler signal nominal. A dossier without it invites
    exactly the wrong verdict.
    """

    def _sched(self, **peak_overrides):
        peaks = {**_SCHED["profile"]["peaks"], **peak_overrides}
        return {**_SCHED, "profile": {**_SCHED["profile"], "peaks": peaks}}

    def test_spilling_run_reports_the_fleet_peaks(self):
        sched = self._sched(worker_mem_gib=517.0, worker_spill_gib=468.0, worker_max_gib=31.5)
        md = rep.build_dossier(_args(logs=None), sched, None)
        assert "Peak FLEET memory: **517.00 GiB**" in md
        assert "spilled **468.00 GiB**" in md
        assert "hottest worker **31.50 GiB**" in md
        assert "no spill" not in md

    def test_healthy_run_says_the_graph_fit(self):
        """Zero spill is a measured result, not missing data — say so."""
        sched = self._sched(worker_mem_gib=40.0, worker_spill_gib=0.0, worker_max_gib=14.0)
        md = rep.build_dossier(_args(logs=None), sched, None)
        assert "_(no spill — the graph fit the fleet)_" in md
        assert "Peak FLEET memory: **40.00 GiB**" in md

    def test_older_profile_without_fleet_keys_renders(self):
        """A profile from before the heartbeat carried fleet memory must not crash,
        and must read "not recorded" rather than implying a healthy zero.
        """
        md = rep.build_dossier(_args(logs=None), _SCHED, None)  # fixture has no fleet keys
        assert "Peak FLEET memory: **not recorded**" in md
