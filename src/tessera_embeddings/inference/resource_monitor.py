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


#: The six `--query-gpu` fields, in the order nvidia-smi emits them for the query below.
#: Named so the parse maps field to name by POSITION IN THIS LIST rather than by an
#: index literal buried in a dict, which is what let a misaligned parse go unnoticed.
_GPU_FIELDS = ("gpu_util", "mem_util", "mem_used", "mem_total", "temp", "power")


def _get_gpu_stats(gpu_index: str | None = None) -> dict[str, str] | None:
    """Query nvidia-smi for ONE GPU's utilization and memory.

    Args:
        gpu_index: The host-level GPU index to query, as ``nvidia-smi -i`` takes
            it. ``None`` queries whatever nvidia-smi defaults to, which is every
            GPU — correct only on a one-GPU host.

    **Why the index has to be passed in.** ``nvidia-smi`` does not honour
    ``CUDA_VISIBLE_DEVICES``; it reports every GPU on the host regardless of what
    the calling process can see. Without ``-i`` this returned one CSV ROW PER GPU,
    and the old parse split the whole multi-line output on commas — so on a 4-GPU
    host the six names above were filled from the first row's six fields plus the
    newline-joined boundary, i.e. every actor reported GPU 0 with fields sliding
    out of alignment. On a one-GPU host there is one row and the bug is invisible,
    which is why it survived: the code was never wrong anywhere it had run.

    Returns:
        One GPU's stats, or ``None`` if nvidia-smi is absent, times out, exits
        non-zero, or returns anything other than exactly one parseable row.
    """
    cmd = ["nvidia-smi"]
    if gpu_index is not None:
        cmd += ["-i", str(gpu_index)]
    cmd += [
        "--query-gpu="
        + ",".join(
            ("utilization.gpu", "utilization.memory", "memory.used", "memory.total", "temperature.gpu", "power.draw")
        ),
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    # Split on LINES first and require exactly one. Several lines means the -i
    # filter did not apply (an unset index, or a future nvidia-smi that ignores
    # it), and the honest answer to "which GPU is this?" is then no answer —
    # returning the first row's numbers under this actor's name is how the
    # original defect misreported a whole packed host.
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    if len(lines) != 1:
        logger.warning(
            "nvidia-smi returned %d rows for gpu_index=%s; not attributing GPU stats to this actor",
            len(lines),
            gpu_index,
        )
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) != len(_GPU_FIELDS):
        return None
    raw = dict(zip(_GPU_FIELDS, parts, strict=True))
    return {
        "gpu_util": f"{raw['gpu_util']}%",
        "mem_util": f"{raw['mem_util']}%",
        "mem_used": f"{raw['mem_used']} MiB",
        "mem_total": f"{raw['mem_total']} MiB",
        "temp": f"{raw['temp']}C",
        "power": f"{raw['power']}W",
    }


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
        gpu_index: This process's own host-level GPU index, for ``nvidia-smi -i``
            (see :func:`_get_gpu_stats`). ``None`` — the default, and correct on a
            one-GPU host — leaves the query unfiltered. On a host where several
            actors share the box, passing it is what keeps each actor's GPU line
            about its own GPU; it is also stamped on the RESOURCES line, so a
            reader can tell four actors' samples apart instead of seeing four
            identical ones.
    """

    def __init__(self, interval_sec: float = 30, gpu_index: str | None = None) -> None:
        self._interval = interval_sec
        self._gpu_index = gpu_index
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Named context slots appended to every RESOURCES line so post-hoc RAM
        # analysis can attribute a sample to what the process was doing. Slots
        # (not one string) because two threads legitimately report at once: the
        # actor's main thread ("work" — chunk/phase) and its background writer
        # ("write" — a staging upload overlapping the next chunk's prologue).
        self._contexts: dict[str, str] = {}
        self._ctx_lock = threading.Lock()

    def set_context(self, slot: str, value: str | None) -> None:
        """Set (or clear, with ``None``) one named context slot.

        Thread-safe; each caller owns its slot so concurrent phases don't
        clobber each other. Shows up as ``ctx=slot:value ...`` on the next
        RESOURCES line.
        """
        with self._ctx_lock:
            if value is None:
                self._contexts.pop(slot, None)
            else:
                self._contexts[slot] = value

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
            self._emit_once()

    def _emit_once(self) -> None:
        """Sample once and log a RESOURCES line (factored out for testing)."""
        parts = []

        cpu_mem = _get_cpu_mem_stats()
        if "load_avg" in cpu_mem:
            parts.append(f"load={cpu_mem['load_avg']}")
        if "ram" in cpu_mem:
            parts.append(f"RAM={cpu_mem['ram']}")

        gpu = _get_gpu_stats(self._gpu_index)
        if gpu:
            if self._gpu_index is not None:
                parts.append(f"gpu_idx={self._gpu_index}")
            parts.append(f"GPU={gpu['gpu_util']}")
            parts.append(f"VRAM={gpu['mem_used']}/{gpu['mem_total']}")
            parts.append(f"temp={gpu['temp']}")
            parts.append(f"power={gpu['power']}")

        with self._ctx_lock:
            if self._contexts:
                ctx = " ".join(f"{slot}:{val}" for slot, val in sorted(self._contexts.items()))
                parts.append(f"ctx={ctx}")

        if parts:
            logger.info("RESOURCES: %s", " | ".join(parts))
