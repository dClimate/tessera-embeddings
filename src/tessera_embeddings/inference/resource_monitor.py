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
    ``CUDA_VISIBLE_DEVICES``; it reports every GPU on the host regardless of what the calling
    process can see. Without ``-i`` it emits one CSV ROW PER GPU, so on a packed host a parse that
    splits the whole output on commas fills the six names above from the first row plus the
    newline-joined boundary — every actor reporting GPU 0 with fields out of alignment. A one-GPU
    host has one row and the defect is invisible there.

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
    # Split on LINES first and require exactly one. Several means the -i filter did not apply (an
    # unset index, or a future nvidia-smi that ignores it), and the honest answer to "which GPU is
    # this?" is then no answer — returning the first row's numbers under this actor's name is what
    # misreported a whole packed host.
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


#: Fields queried in a SECOND nvidia-smi call (see :func:`_get_gpu_extra_stats`).
_GPU_EXTRA_FIELDS = ("sm_clock", "pcie_gen", "pcie_width")

#: Candidate names for the throttle bitmask, newest first. nvidia-smi renamed
#: `clocks_throttle_reasons.*` to `clocks_event_reasons.*` and rejects the whole
#: query with a non-zero exit when a field name is unknown — so the name has to be
#: discovered rather than assumed, and the winner cached.
_THROTTLE_FIELD_CANDIDATES = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")

#: Which candidate answered, once one has. `None` = not yet tried; `""` = none work.
_throttle_field: str | None = None


def _query_gpu(gpu_index: str | None, fields: tuple[str, ...]) -> list[str] | None:
    """Run one `nvidia-smi --query-gpu` and return exactly one row's values.

    Returns ``None`` on any failure — a missing binary, a timeout, a non-zero exit
    (which is what an unknown FIELD NAME produces), or anything other than exactly
    one row. Callers treat ``None`` as "this metric is unavailable on this card",
    never as zero.
    """
    cmd = ["nvidia-smi"]
    if gpu_index is not None:
        cmd += ["-i", str(gpu_index)]
    cmd += ["--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    return [v.strip() for v in lines[0].split(",")]


def _get_gpu_extra_stats(gpu_index: str | None = None) -> dict[str, str]:
    """Clock, PCIe link and throttle state for ONE GPU. ``{}`` for anything unavailable.

    A separate subprocess call from :func:`_get_gpu_stats` deliberately: nvidia-smi rejects an
    ENTIRE query when one field name is unknown, so folding these in would let an unfamiliar driver
    cost us utilisation, VRAM, temperature AND power. A separate call degrades to ``{}`` instead.

    These answer "is the card held back by something other than the work?" on a GPU we have never
    run before:

    * ``sm_clock`` against the card's boost clock, with ``throttle``'s bitmask, distinguishes a
      thermally or power limited card from one simply given small kernels. A non-zero bitmask with
      only bit 0 (``GpuIdle``) set is normal; ``SwPowerCap`` (0x4) and ``HwSlowdown`` (0x8) /
      ``HwThermalSlowdown`` (0x40) are not.
    * ``pcie_gen`` / ``pcie_width`` are the negotiated link, NOT traffic — a card that came up at
      Gen1 x8 would explain a feed problem that looks like a slow GPU. PCIe throughput needs
      ``nvidia-smi dmon`` or DCGM and is not claimed here.
    """
    global _throttle_field
    stats: dict[str, str] = {}
    base = _query_gpu(gpu_index, ("clocks.sm", "pcie.link.gen.gpucurrent", "pcie.link.width.current"))
    if base is not None and len(base) == len(_GPU_EXTRA_FIELDS):
        raw = dict(zip(_GPU_EXTRA_FIELDS, base, strict=True))
        stats["sm_clock"] = f"{raw['sm_clock']}MHz"
        stats["pcie"] = f"gen{raw['pcie_gen']}x{raw['pcie_width']}"
    if _throttle_field is None:
        for candidate in _THROTTLE_FIELD_CANDIDATES:
            if _query_gpu(gpu_index, (candidate,)) is not None:
                _throttle_field = candidate
                break
        else:
            _throttle_field = ""
            logger.info("nvidia-smi knows no throttle-reason field; throttle state unavailable")
    if _throttle_field:
        row = _query_gpu(gpu_index, (_throttle_field,))
        if row is not None:
            stats["throttle"] = row[0]
    return stats


def _get_cpu_mem_stats() -> dict[str, str]:
    """Get CPU and memory stats from /proc (Linux only)."""
    stats: dict[str, str] = {}

    # Load average from /proc/loadavg (1/5/15-minute figures, not instantaneous CPU)
    try:
        with Path("/proc/loadavg").open() as f:
            parts = f.read().strip().split()
            stats["load_avg"] = f"{parts[0]} {parts[1]} {parts[2]}"
    except OSError:
        pass

    # Memory from /proc/meminfo
    ram = read_host_ram_gib()
    if ram is not None:
        used_gb, total_gb = ram
        stats["ram"] = f"{used_gb:.1f}/{total_gb:.1f} GB ({100 * used_gb / max(total_gb, 0.1):.0f}%)"

    return stats


def read_host_ram_gib() -> tuple[float, float] | None:
    """``(used, total)`` host RAM in GiB from /proc/meminfo, or ``None`` off Linux.

    Used is ``MemTotal - MemAvailable``, the figure the per-actor budget at
    :data:`~tessera_embeddings.inference.read_plan._S2_STRIP_BYTE_BUDGET` is sized against.
    Whole-HOST, not per-process: where several actors share a box this is their sum plus the
    system's — the right denominator for "did we fit in this instance size", the wrong one for
    "how much did THIS actor use".
    """
    try:
        with Path("/proc/meminfo").open() as f:
            meminfo = {}
            for line in f:
                key, val = line.split(":", 1)
                meminfo[key.strip()] = int(val.strip().split()[0])  # kB
    except (OSError, ValueError):
        return None
    total_gb = meminfo.get("MemTotal", 0) / 1048576
    avail_gb = meminfo.get("MemAvailable", 0) / 1048576
    return total_gb - avail_gb, total_gb


class ResourceMonitor:
    """Background thread that logs system resource usage at a fixed interval.

    Args:
        interval_sec: Seconds between log lines. Default 30.
        gpu_index: This process's own host-level GPU index, for ``nvidia-smi -i``
            (see :func:`_get_gpu_stats`). ``None`` — the default, correct on a one-GPU
            host — leaves the query unfiltered. Where several actors share a box, passing
            it is what keeps each actor's GPU line about its own GPU; it is also stamped
            on the RESOURCES line so a reader can tell their samples apart.
    """

    def __init__(self, interval_sec: float = 30, gpu_index: str | None = None, sample_sec: float = 2.0) -> None:
        self._interval = interval_sec
        self._sample_sec = min(sample_sec, interval_sec)
        self._gpu_index = gpu_index
        # Host-RAM high-water mark since the last reset. Sampled at `sample_sec`, NOT at
        # `interval_sec`: the per-actor RAM budget leaves ~0.9 GB under a 60% ceiling for spikes
        # shorter than the 30-second emit cadence, so the emitted instantaneous figure sits
        # systematically below the peak it is meant to police. A 2-second sampler is still no upper
        # bound (a sub-second spike escapes it) but it is the difference between "34% average" and
        # a usable peak.
        self._peak_ram_gb = 0.0
        self._ram_total_gb = 0.0
        self._ram_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Named context slots appended to every RESOURCES line so post-hoc RAM analysis can
        # attribute a sample to what the process was doing. Slots rather than one string because
        # two threads legitimately report at once: the actor's main thread ("work" — chunk/phase)
        # and its background writer ("write" — a staging upload overlapping the next prologue).
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

    def peak_host_ram_gib(self) -> tuple[float, float] | None:
        """``(peak_used, total)`` host RAM in GiB since the last reset, or ``None``.

        ``None`` means nothing has been sampled yet — off Linux, or before the
        first sampler tick. A caller must not read that as zero.
        """
        with self._ram_lock:
            if self._ram_total_gb == 0.0:
                return None
            return self._peak_ram_gb, self._ram_total_gb

    def reset_peak_host_ram(self) -> None:
        """Drop the high-water mark so the next read covers only what follows.

        Called at chunk start, because host RAM scales with a chunk's optical
        depth and a maximum taken over a whole worker's life is a property of its
        deepest chunk, not of the one being reported.
        """
        with self._ram_lock:
            self._peak_ram_gb = 0.0
            # Cleared TOO, because `peak_host_ram_gib` reports "not sampled" as a zero TOTAL.
            # Left set, a just-reset monitor returns `(0.0, total)`, which `_host_fields` publishes
            # as a measured-looking `host_ram_peak_gib: 0.0` — a chunk that finished inside one 2 s
            # sampling interval writes a false zero into the per-chunk RAM record.
            self._ram_total_gb = 0.0

    def _sample_ram(self) -> None:
        ram = read_host_ram_gib()
        if ram is None:
            return
        used_gb, total_gb = ram
        with self._ram_lock:
            self._ram_total_gb = total_gb
            self._peak_ram_gb = max(self._peak_ram_gb, used_gb)

    def _run(self) -> None:
        elapsed = 0.0
        while not self._stop_event.wait(self._sample_sec):
            self._sample_ram()
            elapsed += self._sample_sec
            if elapsed >= self._interval:
                elapsed = 0.0
                self._emit_once()

    def _emit_once(self) -> None:
        """Sample once and log a RESOURCES line (factored out for testing)."""
        parts = []

        cpu_mem = _get_cpu_mem_stats()
        if "load_avg" in cpu_mem:
            parts.append(f"load={cpu_mem['load_avg']}")
        if "ram" in cpu_mem:
            parts.append(f"RAM={cpu_mem['ram']}")
        peak = self.peak_host_ram_gib()
        if peak is not None:
            parts.append(f"RAMpeak={peak[0]:.1f}/{peak[1]:.1f} GB ({100 * peak[0] / max(peak[1], 0.1):.0f}%)")

        # OUTSIDE the `if gpu` below: a transient nvidia-smi timeout or malformed response
        # returns None, and dropping the index there would make four packed actors' CPU/RAM
        # lines indistinguishable during exactly the interval a reader needs to tell them apart.
        if self._gpu_index is not None:
            parts.append(f"gpu_idx={self._gpu_index}")
        gpu = _get_gpu_stats(self._gpu_index)
        if gpu:
            parts.append(f"GPU={gpu['gpu_util']}")
            parts.append(f"memio={gpu['mem_util']}")
            parts.append(f"VRAM={gpu['mem_used']}/{gpu['mem_total']}")
            parts.append(f"temp={gpu['temp']}")
            parts.append(f"power={gpu['power']}")
            for key, value in _get_gpu_extra_stats(self._gpu_index).items():
                parts.append(f"{key}={value}")

        with self._ctx_lock:
            if self._contexts:
                ctx = " ".join(f"{slot}:{val}" for slot, val in sorted(self._contexts.items()))
                parts.append(f"ctx={ctx}")

        if parts:
            logger.info("RESOURCES: %s", " | ".join(parts))
