"""Bulk object-store prefix deletion, shared by staging + mosaic cleanup.

On a versioned S3 bucket a plain delete only writes delete markers, so old
object versions accumulate and keep costing storage. :func:`delete_prefix`
removes a prefix with ``s5cmd rm --all-versions`` (the production path — s5cmd
parallelizes far past fsspec's serial per-key DELETEs), falling back to fsspec's
recursive ``rm`` if s5cmd is unavailable. The fallback is best-effort: fsspec
cannot remove non-current versions, so a warning is logged when it is used on a
versioned bucket.

Subprocess + fsspec only (no boto3), so this stays outside ``providers/aws``.
"""

from __future__ import annotations

import logging
import subprocess

import fsspec

logger = logging.getLogger(__name__)

_Log = logging.Logger | logging.LoggerAdapter


def _s5cmd_rm(uri: str, log: _Log, *, all_versions: bool) -> None:
    """Delete everything under *uri* with ``s5cmd rm`` (``--all-versions`` opt-in).

    Raises:
        FileNotFoundError: If the s5cmd binary is not on PATH.
        RuntimeError: If s5cmd exits non-zero.
    """
    cmd = ["s5cmd", "rm"]
    if all_versions:
        cmd.append("--all-versions")
    cmd.append(f"{uri.rstrip('/')}/*")
    log.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        msg = "s5cmd binary not found — install it or add it to PATH"
        raise FileNotFoundError(msg) from None
    if result.returncode != 0:
        msg = f"s5cmd failed (rc={result.returncode}): {result.stderr.strip()}"
        raise RuntimeError(msg)
    n = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
    log.info("s5cmd deleted %d object(s) from %s", n, uri)


def delete_prefix(uri: str, *, log: _Log | None = None, all_versions: bool = True, strict: bool = False) -> None:
    """Delete every object under *uri* (a directory-like prefix).

    S3 uses ``s5cmd`` (all versions by default) with an fsspec fallback; other
    schemes use fsspec directly.

    Args:
        uri: Prefix to remove (e.g. ``s3://bucket/mosaics/33N/2025``).
        log: Optional logger; falls back to the module logger.
        all_versions: Pass ``--all-versions`` to s5cmd so a versioned bucket does
            not accumulate non-current versions (the default; the reason this
            helper exists).
        strict: When True, RAISE if the delete does not succeed. Best-effort
            (default) is right for post-success cleanup (staging, tagged mosaics);
            strict is for callers that must not proceed onto un-cleared data (e.g.
            rebuilding a stale mosaic before re-ingest).
    """
    log = log or logger
    log.info("Deleting prefix: %s", uri)

    if fsspec.utils.get_protocol(uri) == "s3":
        try:
            _s5cmd_rm(uri, log, all_versions=all_versions)
            return
        except (FileNotFoundError, RuntimeError) as exc:
            versions_note = " (non-current versions may remain)" if all_versions else ""
            log.warning("s5cmd rm of %s failed (%s) — falling back to fsspec%s", uri, exc, versions_note)

    try:
        fs = fsspec.filesystem(fsspec.utils.get_protocol(uri))
        if fs.exists(uri):
            fs.rm(uri, recursive=True)
    except Exception:
        if strict:
            raise
        log.warning("Failed to delete prefix %s", uri, exc_info=True)
