"""Lightweight background resource monitor for inference workers.

Periodically logs CPU, memory, and GPU utilization so we can diagnose
performance bottlenecks from CloudWatch logs without SSM-ing into the box.

Usage:
    monitor = ResourceMonitor(interval_sec=30)
    monitor.start()
    # ... do work ...
    monitor.stop()
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_gpu_stats() -> dict[str, str] | None:
    """Query nvidia-smi for GPU utilization and memory."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 6:
            return {
                "gpu_util": f"{parts[0]}%",
                "mem_util": f"{parts[1]}%",
                "mem_used": f"{parts[2]} MiB",
                "mem_total": f"{parts[3]} MiB",
                "temp": f"{parts[4]}C",
                "power": f"{parts[5]}W",
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_cpu_mem_stats() -> dict[str, str]:
    """Get CPU and memory stats from /proc (Linux only)."""
    stats: dict[str, str] = {}

    # CPU usage from /proc/stat (snapshot — shows cumulative, but useful for trends)
    try:
        with Path("/proc/loadavg").open() as f:
            parts = f.read().strip().split()
            stats["load_avg"] = f"{parts[0]} {parts[1]} {parts[2]}"
    except OSError:
        pass

    # Memory from /proc/meminfo
    try:
        with Path("/proc/meminfo").open() as f:
            meminfo = {}
            for line in f:
                key, val = line.split(":", 1)
                meminfo[key.strip()] = int(val.strip().split()[0])  # kB
            total_gb = meminfo.get("MemTotal", 0) / 1048576
            avail_gb = meminfo.get("MemAvailable", 0) / 1048576
            used_gb = total_gb - avail_gb
            stats["ram"] = f"{used_gb:.1f}/{total_gb:.1f} GB ({100 * used_gb / max(total_gb, 0.1):.0f}%)"
    except (OSError, ValueError, KeyError):
        pass

    return stats


class ResourceMonitor:
    """Background thread that logs system resource usage at a fixed interval.

    Args:
        interval_sec: Seconds between log lines. Default 30.
    """

    def __init__(self, interval_sec: float = 30) -> None:
        self._interval = interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the monitor thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="resource-monitor")
        self._thread.start()
        logger.info("ResourceMonitor started (interval=%ds)", self._interval)

    def stop(self) -> None:
        """Stop the monitor thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("ResourceMonitor stopped")

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            parts = []

            cpu_mem = _get_cpu_mem_stats()
            if "load_avg" in cpu_mem:
                parts.append(f"load={cpu_mem['load_avg']}")
            if "ram" in cpu_mem:
                parts.append(f"RAM={cpu_mem['ram']}")

            gpu = _get_gpu_stats()
            if gpu:
                parts.append(f"GPU={gpu['gpu_util']}")
                parts.append(f"VRAM={gpu['mem_used']}/{gpu['mem_total']}")
                parts.append(f"temp={gpu['temp']}")
                parts.append(f"power={gpu['power']}")

            if parts:
                logger.info("RESOURCES: %s", " | ".join(parts))
