"""Shared harness: config, metrics, phase markers, timers, RSS, object stats.

Every ``tN_*`` script imports from here. The contracts pinned in this module
(CLI shape, ``MetricRow`` schema, marker discipline, the fixed metric-name
vocabulary) are the *interlock* that lets ``report.py`` join results across
tests — do not diverge from them per-script.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import fsspec
import icechunk
import zarr

from tessera_embeddings.storage import zarr_store

logger = logging.getLogger("scale_tests")

# ── Fixed metric-name vocabulary (report.py joins on these) ──────────────────
# Improvising a name here silently drops the row from the decision tables.
METRICS = frozenset(
    {
        "wall_s",
        "commit_wall_s",
        "commit_wall_p95_s",
        "merge_wall_s",
        "peak_rss_bytes",
        "read_p50_ms",
        "read_p95_ms",
        "bytes_fetched",
        "throughput_mbps",
        "open_wall_s",
        "retries",
        "refs_committed",
        "manifest_count",
        "manifest_bytes",
        "bytes_written",
        "snapshot_bytes",
        "objects_listed",
        "objects_deleted",
        "bytes_reclaimed",
        "gc_wall_s",
        "slowdown_503_count",
        "puts_per_s",
    }
)

_TINY = "tiny"
_BENCH = "bench"
_LOCAL = "local"
_S3 = "s3"


# ── Run configuration ────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Resolved CLI configuration shared by every scale-test script."""

    run_id: str
    backend: str  # "local" | "s3"
    scale: str  # "tiny" | "bench"
    bucket: str | None
    results_dir: Path
    store_root: str  # local dir path or "s3://bucket/prefix"
    real_sample: Path | None
    variant: str | None  # restrict to one variant, or None for all
    phase: str | None  # run one phase, or None for all

    @property
    def is_s3(self) -> bool:
        """True when writing to S3 rather than the local filesystem."""
        return self.backend == _S3

    @property
    def is_tiny(self) -> bool:
        """True for the fast laptop-scale run (as opposed to the bench run)."""
        return self.scale == _TINY


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register the CLI flags every ``tN`` script accepts (impl-spec §2.1)."""
    parser.add_argument("--run-id", required=True, help="Groups all artifacts of one run.")
    parser.add_argument("--backend", choices=[_LOCAL, _S3], default=_LOCAL)
    parser.add_argument("--scale", choices=[_TINY, _BENCH], default=_TINY)
    parser.add_argument(
        "--bucket",
        default=None,
        help=(
            "S3 bucket for --backend s3. Accepts a bare name, 'bucket/prefix', or "
            "'s3://bucket/prefix' — a prefix acts as the default --store-root."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Local results root (default ./scale_test_results).",
    )
    parser.add_argument(
        "--store-root",
        default=None,
        help="Override the store root URI (default: <bucket>/scale_tests or ./scale_test_stores).",
    )
    parser.add_argument("--real-sample", default=None, help="Path to a real-embedding .npy sample.")
    parser.add_argument("--variant", default=None, help="Restrict to a single variant name.")
    parser.add_argument("--phase", default=None, help="Run a single phase by name.")


def _parse_bucket_arg(raw: str) -> tuple[str, str | None]:
    """Split a ``--bucket`` value into ``(bare_bucket_name, prefix_or_None)``.

    Accepts a bare name (``arbol-tessera-embeddings-dev``), a ``bucket/prefix``
    path, or a full ``s3://bucket/prefix`` URI (with or without trailing slash).
    Returns the *bare* bucket name — which is what ``boto3``/T7 and the results
    mirror require — plus any prefix, used to derive the default store root.

    Raises ``SystemExit`` with a clear message on a malformed value, rather than
    letting a mangled name reach S3 as a cryptic ``InvalidBucketName``.
    """
    s = raw.strip()
    if s.startswith("s3://"):
        s = s[len("s3://") :]
    s = s.strip("/")
    bucket, _, prefix = s.partition("/")
    # boto3's own bucket-name rule; catches the double-scheme mangling early.
    if not re.fullmatch(r"[a-zA-Z0-9.\-_]{1,255}", bucket):
        raise SystemExit(
            f"--bucket {raw!r} does not resolve to a valid S3 bucket name (got {bucket!r}). "
            "Pass a bare name, 'bucket/prefix', or 's3://bucket/prefix'."
        )
    return bucket, (prefix.strip("/") or None)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build a :class:`RunConfig` from parsed args, validating S3 requirements.

    ``--bucket`` is normalized to a bare name (a prefix or ``s3://`` scheme is
    accepted and, absent an explicit ``--store-root``, the prefix becomes the
    store root). ``--store-root``, when given, always wins.
    """
    bucket: str | None = None
    bucket_prefix: str | None = None
    if args.backend == _S3:
        if not args.bucket:
            raise SystemExit("--bucket is required with --backend s3")
        bucket, bucket_prefix = _parse_bucket_arg(args.bucket)

    results_dir = Path(args.results_dir) if args.results_dir else Path.cwd() / "scale_test_results"
    results_dir = results_dir / args.run_id

    if args.store_root:
        store_root = args.store_root  # explicit override always wins
    elif args.backend == _S3:
        # A prefix from --bucket is treated exactly as if passed to --store-root;
        # a bare bucket gets the harness-managed scale_tests/<run-id> layout.
        store_root = f"s3://{bucket}/{bucket_prefix}" if bucket_prefix else f"s3://{bucket}/scale_tests/{args.run_id}"
    else:
        store_root = str((Path.cwd() / "scale_test_stores" / args.run_id).resolve())

    return RunConfig(
        run_id=args.run_id,
        backend=args.backend,
        scale=args.scale,
        bucket=bucket,
        results_dir=results_dir,
        store_root=store_root,
        real_sample=Path(args.real_sample) if args.real_sample else None,
        variant=args.variant,
        phase=args.phase,
    )


def store_uri(cfg: RunConfig, name: str) -> str:
    """Return the store URI for a named store under the run's store root."""
    if cfg.store_root.startswith("s3://"):
        return f"{cfg.store_root.rstrip('/')}/{name}"
    return str(Path(cfg.store_root) / name)


# ── Repository helpers (wrap the library so tests inherit its hardening) ──────


def layered_config(
    *,
    split_sizes: dict[str, int] | None = None,
    preload_max_total_refs: int | None = None,
    preload_max_arrays_to_scan: int | None = None,
    max_concurrent_requests: int | None = None,
) -> icechunk.RepositoryConfig:
    """Build a RepositoryConfig on top of the library's timeout/retry hardening.

    Reuses :func:`zarr_store._default_repo_config` (finite timeouts + backed-off
    retries) and layers manifest splitting and/or preload tuning on top. All
    arguments are optional; with none set this is exactly the library default.
    """
    config = zarr_store._default_repo_config(max_concurrent_requests)
    splitting = zarr_store._manifest_splitting_config(split_sizes) if split_sizes else None
    preload = None
    if preload_max_total_refs is not None or preload_max_arrays_to_scan is not None:
        kwargs: dict[str, int] = {}
        if preload_max_total_refs is not None:
            kwargs["max_total_refs"] = preload_max_total_refs
        if preload_max_arrays_to_scan is not None:
            kwargs["max_arrays_to_scan"] = preload_max_arrays_to_scan
        preload = icechunk.ManifestPreloadConfig(**kwargs)
    if splitting is not None or preload is not None:
        config.manifest = icechunk.ManifestConfig(splitting=splitting, preload=preload)
    return config


def create_repo(cfg: RunConfig, name: str, config: icechunk.RepositoryConfig | None = None) -> icechunk.Repository:
    """Create a fresh Icechunk repository for a named store."""
    return icechunk.Repository.create(
        zarr_store._create_storage(store_uri(cfg, name)),
        config=config or layered_config(),
    )


def open_repo(cfg: RunConfig, name: str, config: icechunk.RepositoryConfig | None = None) -> icechunk.Repository:
    """Open an existing Icechunk repository for a named store."""
    return icechunk.Repository.open(
        zarr_store._create_storage(store_uri(cfg, name)),
        config=config or layered_config(),
    )


def reset_store(cfg: RunConfig, name: str) -> None:
    """Delete a named store so a phase can re-seed from a clean prefix.

    Icechunk refuses to create a repo in a non-empty prefix, so a phase that
    seeds must clear any partial store left by an earlier crashed attempt.
    """
    uri = store_uri(cfg, name)
    if uri.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        path = uri[len("s3://") :]
    else:
        fs = fsspec.filesystem("file")
        path = uri
    if fs.exists(path):
        fs.rm(path, recursive=True)


# ── Metrics ────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class MetricRow:
    """One metric observation. ``metric`` must be in :data:`METRICS`."""

    run_id: str
    test: str
    variant: str | None
    phase: str
    metric: str
    value: float
    unit: str
    params: dict[str, Any] = dataclasses.field(default_factory=dict)


_STAMP: dict[str, str] | None = None


def _stamp() -> dict[str, str]:
    """Return (memoized) version/git provenance stamped onto every metric row."""
    global _STAMP
    if _STAMP is None:
        _STAMP = {
            "icechunk": icechunk.__version__,
            "zarr": zarr.__version__,
            "git_sha": git_sha(),
        }
    return _STAMP


def git_sha() -> str:
    """Return the short git SHA of the working tree, or ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _metrics_file(cfg: RunConfig, test: str, phase: str) -> Path:
    """Local JSONL path for one (test, phase)'s metric rows."""
    d = cfg.results_dir / test
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{phase}.jsonl"


def emit(cfg: RunConfig, row: MetricRow) -> None:
    """Append one metric row to its (test, phase) JSONL file.

    Unknown metric names are rejected loudly — a typo must not silently vanish
    from the decision tables.
    """
    if row.metric not in METRICS:
        raise ValueError(f"Unknown metric {row.metric!r}; add it to harness.METRICS first.")
    record = {
        **dataclasses.asdict(row),
        **_stamp(),
        "scale": cfg.scale,
        "backend": cfg.backend,
        "ts": _utcnow(),
    }
    with _metrics_file(cfg, row.test, row.phase).open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def emit_metric(
    cfg: RunConfig,
    test: str,
    phase: str,
    metric: str,
    value: float,
    unit: str,
    *,
    variant: str | None = None,
    **params: Any,  # noqa: ANN401 — arbitrary metric tags, serialized as-is
) -> None:
    """Convenience wrapper around :func:`emit` for a single scalar."""
    emit(
        cfg,
        MetricRow(
            run_id=cfg.run_id,
            test=test,
            variant=variant,
            phase=phase,
            metric=metric,
            value=float(value),
            unit=unit,
            params=params,
        ),
    )


def _utcnow() -> str:
    """UTC timestamp string (kept isolated so it is easy to find/replace)."""
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat()


# ── Phase markers + orchestration ────────────────────────────────────────────


def _marker(cfg: RunConfig, test: str, phase: str) -> Path:
    """Local success-marker path for one (test, phase)."""
    d = cfg.results_dir / test
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{phase}.done"


def phase_done(cfg: RunConfig, test: str, phase: str) -> bool:
    """True if this (test, phase) already completed in a prior run."""
    return _marker(cfg, test, phase).exists()


def mark_phase_done(cfg: RunConfig, test: str, phase: str) -> None:
    """Record a (test, phase) as complete and best-effort mirror to S3."""
    _marker(cfg, test, phase).write_text(_utcnow())
    if cfg.is_s3:
        _mirror_to_s3(cfg, test)


def _mirror_to_s3(cfg: RunConfig, test: str) -> None:
    """Best-effort upload of a test's local result files to the S3 results prefix."""
    if not cfg.bucket:
        return
    try:
        fs = fsspec.filesystem("s3")
        local = cfg.results_dir / test
        dest = f"{cfg.bucket}/results/{cfg.run_id}/{test}"
        for f in local.glob("*"):
            fs.put_file(str(f), f"{dest}/{f.name}")
    except Exception as exc:
        logger.warning("S3 results mirror failed for %s: %s", test, exc)


def run_phase(
    cfg: RunConfig,
    test: str,
    name: str,
    fn: Callable[[], None],
    *,
    variant: str | None = None,
) -> None:
    """Run one idempotent phase: skip if done, time it, emit ``wall_s``, mark done.

    ``fn`` takes no arguments and reconstructs whatever it needs by reopening
    stores — phases must be resumable, so nothing is threaded through in memory.
    Honors ``--phase`` (run only the matching phase) and ``--variant``.
    """
    if cfg.phase and cfg.phase != name:
        return
    if phase_done(cfg, test, name):
        logger.info("[%s/%s] already done — skipping", test, name)
        return
    logger.info("[%s/%s] starting", test, name)
    # Clear any rows from a prior interrupted attempt so a resume doesn't append
    # duplicates (phases emit under a phase name matching this ``name``).
    metrics_file = _metrics_file(cfg, test, name)
    metrics_file.unlink(missing_ok=True)
    start = time.monotonic()
    fn()
    elapsed = time.monotonic() - start
    emit_metric(cfg, test, name, "wall_s", elapsed, "s", variant=variant)
    mark_phase_done(cfg, test, name)
    logger.info("[%s/%s] done in %.1fs", test, name, elapsed)


# ── Timers + RSS sampler ─────────────────────────────────────────────────────


@dataclasses.dataclass
class _Elapsed:
    """Mutable holder so a ``with timer() as t`` block can read ``t.seconds``."""

    seconds: float = 0.0


@contextlib.contextmanager
def timer() -> Iterator[_Elapsed]:
    """Context manager yielding an :class:`_Elapsed` with ``.seconds`` on exit."""
    holder = _Elapsed()
    start = time.monotonic()
    try:
        yield holder
    finally:
        holder.seconds = time.monotonic() - start


@contextlib.contextmanager
def rss_sampler(
    cfg: RunConfig,
    test: str,
    phase: str,
    *,
    variant: str | None = None,
    include_children: bool = True,
    interval_s: float = 1.0,
) -> Iterator[None]:
    """Sample this process (and optionally children) RSS at ``interval_s``.

    Emits ``peak_rss_bytes`` on exit. Used to validate the ~400 B/ref commit
    memory model (test plan T2) and to watch fork/merge coordinator RSS.
    """
    try:
        import psutil
    except ImportError as exc:
        raise SystemExit("psutil is required for scale tests: add the scale-tests dep group") from exc

    proc = psutil.Process()
    peak = 0
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            total = proc.memory_info().rss
            if include_children:
                for child in proc.children(recursive=True):
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        total += child.memory_info().rss
            peak = max(peak, total)
            stop.wait(interval_s)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval_s * 2)
        emit_metric(cfg, test, phase, "peak_rss_bytes", peak, "bytes", variant=variant)


# ── Object statistics (backend-agnostic via fsspec) ──────────────────────────


def object_stats(uri: str) -> tuple[int, int]:
    """Return ``(object_count, total_bytes)`` under a store/prefix URI.

    Works for both local paths and ``s3://`` URIs. Used for ``objects_listed``,
    ``manifest_bytes`` (point at ``.../manifests``), and ``snapshot_bytes``.
    """
    if uri.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        path = uri[len("s3://") :]
    else:
        fs = fsspec.filesystem("file")
        path = uri
    if not fs.exists(path):
        return (0, 0)
    entries = fs.find(path, detail=True)
    count = len(entries)
    total = sum(int(meta.get("size") or 0) for meta in entries.values())
    return (count, total)


def newest_object_bytes(uri: str) -> int:
    """Return the byte size of the most recently modified object under ``uri``.

    Used to size the just-written snapshot file (``snapshot_bytes``) after a
    commit, since each commit writes a fresh snapshot object.
    """
    if uri.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        path = uri[len("s3://") :]
    else:
        fs = fsspec.filesystem("file")
        path = uri
    if not fs.exists(path):
        return 0
    entries = fs.find(path, detail=True)
    if not entries:
        return 0
    newest = max(entries.values(), key=lambda m: m.get("LastModified") or m.get("mtime") or 0)
    return int(newest.get("size") or 0)


# ── Cold reads via a fresh subprocess ────────────────────────────────────────


def run_cold(func_dotpath: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run ``func_dotpath(payload)`` in a fresh process and return its dict result.

    Used for cold-cache read measurements: a new interpreter means no cached
    manifests, sessions, or repositories. ``func_dotpath`` is a dotted path like
    ``scale_tests.t1_read_bench.cold_point_read`` resolving to a function that
    takes one dict and returns a JSON-serializable dict.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "scale_tests._subrunner", func_dotpath],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cold subprocess {func_dotpath} failed:\n{proc.stderr}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:") :])
    raise RuntimeError(f"cold subprocess {func_dotpath} produced no RESULT line:\n{proc.stdout}")


# ── Logging ──────────────────────────────────────────────────────────────────


def configure_logging(level: int = logging.INFO) -> None:
    """Configure stderr logging for a scale-test run (idempotent)."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("scale_tests").setLevel(level)
