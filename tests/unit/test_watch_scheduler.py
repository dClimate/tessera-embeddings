"""Offline tests for the ingest scheduler watcher (scripts/profiling/ingest).

The watcher's value at scale rests on two pure functions that never touch AWS:
``parse_health_line`` (must track the SchedulerResourceLogger format byte for
byte) and ``evaluate_alerts`` / ``profile_run`` (the saturation rules). These
are exercised here without CloudWatch so a format drift or threshold-logic
regression fails fast in CI, mirroring test_observe_cluster_parser.py's
load-by-path pattern.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "profiling" / "ingest" / "watch_scheduler.py"


def _load():
    spec = importlib.util.spec_from_file_location("watch_scheduler", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves cls.__module__ via sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ws = _load()

# A real line as SchedulerResourceLogger emits it, with the CloudWatch/logging
# prefix the parser must tolerate (it uses re.search, not match). Kept in sync
# with the format string in src/tessera_embeddings/providers/aws/dask.py.
_HEALTHY = (
    "2026-07-24 18:00:00,000 distributed.scheduler - INFO - "
    "scheduler health: cpu=42% rss=1.50GiB mem=18% lag=0.1s "
    "fds=42 threads=9 workers=120 tasks=8000 processing=480 no-worker=12"
)


def test_parse_health_line_full():
    s = ws.parse_health_line(_HEALTHY)
    assert s == {
        "cpu": 42.0,
        "rss_gib": 1.5,
        "mem": 18.0,
        "lag": 0.1,
        "fds": 42,
        "threads": 9,
        "workers": 120,
        "tasks": 8000,
        "processing": 480,
        "no_worker": 12,
    }


def test_parse_health_line_nan_mem():
    line = (
        "scheduler health: cpu=100% rss=2.00GiB mem=nan% lag=6.0s "
        "fds=1 threads=1 workers=1 tasks=1 processing=1 no-worker=1"
    )
    s = ws.parse_health_line(line)
    assert s is not None
    assert math.isnan(s["mem"])
    assert s["cpu"] == 100.0


def test_parse_health_line_negatives():
    # The logger emits -1 counts when the scheduler ref is briefly unavailable.
    line = (
        "scheduler health: cpu=0% rss=0.10GiB mem=1% lag=0.0s "
        "fds=5 threads=5 workers=-1 tasks=-1 processing=-1 no-worker=-1"
    )
    s = ws.parse_health_line(line)
    assert s is not None and s["workers"] == -1 and s["no_worker"] == -1


def test_parse_health_line_rejects_non_health():
    assert ws.parse_health_line("2026-07-24 registering worker tcp://10.0.0.1") is None
    assert ws.parse_health_line("") is None


def _sample(**over):
    base = {
        "cpu": 10.0, "rss_gib": 1.0, "mem": 20.0, "lag": 0.0, "fds": 10,
        "threads": 5, "workers": 100, "tasks": 1000, "processing": 50, "no_worker": 0,
    }
    base.update(over)
    return base


def test_alert_cpu_sustained():
    th = ws.Thresholds()  # cpu_pct=90, cpu_intervals=3
    # two hot samples: not yet sustained
    assert "cpu-sustained" not in ws.evaluate_alerts([_sample(cpu=95), _sample(cpu=95)], th)
    # three hot: trips
    assert "cpu-sustained" in ws.evaluate_alerts([_sample(cpu=95)] * 3, th)
    # a cool sample in the last three breaks it
    assert "cpu-sustained" not in ws.evaluate_alerts([_sample(cpu=95), _sample(cpu=50), _sample(cpu=95)], th)


def test_alert_mem_and_lag():
    th = ws.Thresholds()
    assert "mem-high" in ws.evaluate_alerts([_sample(mem=85)], th)
    assert "mem-high" not in ws.evaluate_alerts([_sample(mem=nanmem())], th)  # nan never trips
    assert "loop-lag" in ws.evaluate_alerts([_sample(lag=6.0)], th)


def nanmem():
    return math.nan


def test_alert_backlog_growth():
    th = ws.Thresholds()  # backlog_intervals=3 -> needs 4 strictly-rising samples
    rising = [_sample(no_worker=n) for n in (1, 2, 3, 4)]
    assert "backlog-growth" in ws.evaluate_alerts(rising, th)
    flat = [_sample(no_worker=n) for n in (1, 2, 2, 3)]  # one non-increase
    assert "backlog-growth" not in ws.evaluate_alerts(flat, th)


def test_alert_worker_churn():
    th = ws.Thresholds()  # churn_delta=25
    assert "worker-churn" in ws.evaluate_alerts([_sample(workers=100), _sample(workers=140)], th)
    assert "worker-churn" not in ws.evaluate_alerts([_sample(workers=100), _sample(workers=110)], th)


def test_profile_run_peaks_onsets_events():
    # Build a series that ramps CPU into sustained saturation with a rising
    # backlog and a worker join, then compute the profile.
    series = []
    for i, (cpu, nw, workers) in enumerate(
        [(20, 0, 100), (95, 1, 100), (96, 2, 150), (97, 3, 150), (98, 4, 150)]
    ):
        series.append(ws._snapshot(_sample(cpu=cpu, no_worker=nw, workers=workers), 1000 + i * 30, None, None, []))
    prof = ws.profile_run(series, ws.Thresholds())
    assert prof["samples"] == 5
    assert prof["peaks"]["cpu"] == 98.0
    assert prof["peaks"]["no_worker"] == 4
    # cpu-sustained onset is the 3rd hot sample (index 3, epoch 1090)
    assert prof["onsets"]["cpu-sustained"] is not None
    # backlog rose strictly for the last 4 -> tripped
    assert prof["onsets"]["backlog-growth"] is not None
    # one worker join event (100 -> 150)
    assert len(prof["worker_events"]) == 1 and prof["worker_events"][0]["delta"] == 50


def test_profile_run_empty():
    assert ws.profile_run([], ws.Thresholds()) == {"samples": 0}
