"""Bulk object-store prefix deletion, shared by staging + mosaic cleanup.

:func:`delete_prefix` removes a prefix with ``s5cmd rm`` (the production path — s5cmd
parallelizes far past fsspec's serial per-key DELETEs), falling back to fsspec's
recursive ``rm`` if s5cmd is unavailable.

``--all-versions`` is OPT-IN, not the default. On a versioned bucket a plain delete only
writes delete markers, so old versions accumulate and keep costing storage — that is what
the flag is for. But the campaign's buckets are UNVERSIONED, and there the flag is not a
free precaution: it raises the required permission to ``s3:DeleteObjectVersion`` and
enumerates through ``ListObjectVersions``. Callers on a versioned bucket pass it
explicitly. The fsspec fallback cannot remove non-current versions at all, so a warning is
logged when it is used with the flag on.

**The delete is verified, not reported.** A tool's count of what it removed says
nothing about what is left, and a prefix that keeps some of its objects is worse
than one that keeps all of them: chunks with no store read as a corrupted mosaic
to whatever next tries to write there.

Subprocess + fsspec only (no boto3), so this stays outside ``providers/aws``.
"""

from __future__ import annotations

import logging
import subprocess
import time

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


#: How many times a prefix is re-deleted before its survivors are reported. A delete that
#: reports success can still leave objects behind, so one extra pass is worth taking; more
#: than a couple means something other than a lost listing page is keeping them alive, and
#: repeating will not find it.
_DELETE_PASSES = 3

#: Seconds to wait before a retry, multiplied by the attempt number. Both failure modes here
#: are rate pressure — S3 answered `SlowDown` outright on one prefix and silently dropped
#: objects from another — and rate pressure wants time rather than another immediate attempt.
_DELETE_BACKOFF_S = 5.0


class DeleteUnverifiedError(RuntimeError):
    """The delete ran, and whether it worked could not be established.

    Distinct from :class:`PrefixNotEmptyError`, which asserts a fact: objects ARE still
    there. This one asserts the absence of a fact, and the two want different responses —
    survivors mean the work did not take, an unverifiable prefix means we do not know.
    Only ``strict`` callers see it, because only they have said they cannot proceed on
    an unverified prefix.
    """


class PrefixNotEmptyError(RuntimeError):
    """A delete ran to completion and objects are still there.

    Distinct from the s5cmd failures beside it because it needs the opposite response: those
    mean the tool could not run and fsspec should try instead, this means the work was done
    and did not take. Retrying it another way finds the same survivors.
    """


def _survivors(uri: str, log: _Log) -> list[str] | None:
    """Objects still under *uri*, or None if the prefix could not be listed.

    ``None`` is not "empty": a listing that failed proves nothing, and reporting it as a
    clean prefix is how an unverified delete comes to look verified.

    THE LIMIT OF THIS CHECK. `fs.find` lists CURRENT keys only. On a versioned bucket it
    cannot see non-current versions or delete markers, so an `--all-versions` delete that
    exited zero while leaving history behind would read back as verified-empty — and the
    caller would report reclaimed storage that is still billed. Measured 2026-08-18: every
    bucket this runs against (`arbol-tessera-{embeddings,inputs}` and their `-dev` pair)
    answers `get-bucket-versioning` with no Status, i.e. versioning has never been enabled,
    and the CDK construct creates buckets with it off because Icechunk carries its own
    versioning. So the gap is unreachable today. It becomes reachable the moment versioning
    is turned on anywhere, and nothing here would notice: enable it and this must move to a
    version-aware listing (`list_object_versions`) first.
    """
    try:
        fs = fsspec.filesystem(fsspec.utils.get_protocol(uri))
        fs.invalidate_cache(uri)  # the delete just changed what a cached listing would say
        return list(fs.find(uri))
    except Exception:  # an unreadable prefix is unknown, never clean
        log.warning("Could not list %s to confirm the delete", uri, exc_info=True)
        return None


def _s5cmd_rm_verified(uri: str, log: _Log, *, all_versions: bool) -> None:
    """``s5cmd rm`` the prefix, then READ IT BACK, retrying while objects survive.

    s5cmd reported deleting 145,195 objects from one cell's mosaics and exited zero, and 807 of
    them were still there — hours older than the delete, so not a race with a writer. A count of
    what a tool says it removed is not a statement about what is left, and the residue is not
    inert: the next ingest of that cell finds a prefix holding chunks and no store, which is
    indistinguishable from a corrupted mosaic.

    Raises:
        FileNotFoundError, RuntimeError: from :func:`_s5cmd_rm` — the tool could not run.
        PrefixNotEmptyError: the tool ran and the prefix is still not empty.
    """
    for attempt in range(1, _DELETE_PASSES + 1):
        if attempt > 1:
            # A throttle needs TIME, not another immediate attempt.
            time.sleep(_DELETE_BACKOFF_S * (attempt - 1))
        try:
            _s5cmd_rm(uri, log, all_versions=all_versions)
        except RuntimeError as exc:
            # The expected failure at this scale, and the one that produced the residue:
            # `SlowDown: Please reduce your request rate` on a quarter-million-object prefix,
            # observed live on 48S/2022. A throttle is transient by definition, so it earns the
            # same retry budget as a pass that left survivors — the previous version fell
            # straight through to the serial fsspec sweep, which faces the same rate limit with
            # none of s5cmd's parallelism.
            #
            # NOT FileNotFoundError: a missing binary will still be missing next time, and the
            # fallback is the whole answer for it.
            log.warning("s5cmd rm of %s failed on pass %d (%s)", uri, attempt, exc)
            if attempt == _DELETE_PASSES:
                raise
            continue
        left = _survivors(uri, log)
        if left == []:
            return  # verified empty — the only outcome that is a success
        if left is None:
            # The listing failed, so the prefix is UNKNOWN rather than clean. Retrying the
            # delete would not help (the delete is not what failed) and would manufacture an
            # endless loop, so stop here and let the caller's own bar decide: best-effort
            # returns, strict raises. Reporting it as clean is what `_survivors` documents as
            # the way an unverified delete comes to look verified.
            raise DeleteUnverifiedError(uri)
        log.warning(
            "%d object(s) survived pass %d of the delete of %s (e.g. %s)",
            len(left),
            attempt,
            uri,
            left[0],
        )
    # Only reachable when the LAST pass ran and left objects — a last pass that raised
    # re-raised inside the loop, so the fsspec fallback still gets its turn.
    raise PrefixNotEmptyError(uri)


def delete_prefix(uri: str, *, log: _Log | None = None, all_versions: bool = True, strict: bool = False) -> None:
    """Delete every object under *uri* (a directory-like prefix).

    S3 uses ``s5cmd`` with an fsspec fallback; other schemes use fsspec directly.

    Args:
        uri: Prefix to remove (e.g. ``s3://bucket/mosaics/33N/2025``).
        log: Optional logger; falls back to the module logger.
        all_versions: Pass ``--all-versions`` to s5cmd. **Defaults ON, and the default is
            the safe one rather than the cheap one**, because getting it wrong on a
            VERSIONED bucket is not a cost mistake but a FALSE SUCCESS:

            * the delete removes current keys and leaves delete markers;
            * :func:`_survivors` reads back with ``fs.find``, which lists CURRENT keys
              only, so it sees an empty prefix;
            * an empty read-back is "verified clean", so even ``strict=True`` returns
              success — and a caller that releases a storage-budget slot on the strength
              of that answer does so while every non-current version is still stored and
              still billed.

            Passing ``False`` is therefore an assertion by the caller that the bucket has
            versioning OFF. It is worth making on a bucket you know: the flag buys nothing
            there, it raises the required permission from ``s3:DeleteObject`` to
            ``s3:DeleteObjectVersion``, and it makes s5cmd enumerate through
            ``ListObjectVersions`` rather than ``ListObjectsV2`` — a heavier request
            pattern against a prefix holding hundreds of thousands of objects.

            Note the standing asymmetry even with the flag ON: s5cmd deletes every
            version, but the read-back still confirms only current ones. A *verified*
            all-versions delete needs :func:`_survivors` moved to a version-aware listing
            first, which is the prerequisite for offering this as a flow-level option.
        strict: When True, RAISE if the delete does not succeed — including when it ran
            and left objects behind, which is the case a returned success used to hide.
            Best-effort (default) is right for post-success cleanup (staging, tagged
            mosaics); strict is for callers that must not proceed onto un-cleared data
            (e.g. rebuilding a stale mosaic before re-ingest).
    """
    log = log or logger
    log.info("Deleting prefix: %s", uri)

    if fsspec.utils.get_protocol(uri) == "s3":
        try:
            _s5cmd_rm_verified(uri, log, all_versions=all_versions)
        # BEFORE the RuntimeError arm, which is its base class: except clauses match in order,
        # so the general one would swallow every survivor into the fsspec fallback.
        except DeleteUnverifiedError:
            # The delete ran; the read-back could not confirm it. fsspec would face the same
            # unreadable prefix, so there is nothing further to try.
            if strict:
                raise
            log.error("Could not verify the delete of %s — treat the prefix as unknown, not clean", uri)
            return
        except PrefixNotEmptyError:
            # The delete ran and did not finish the job. Falling back to fsspec would re-run the
            # same failing work serially, so report and let the caller's own bar decide.
            if strict:
                raise
            log.error(
                "%s still holds objects after %d delete passes — a later ingest there will find "
                "residue, not a clean prefix",
                uri,
                _DELETE_PASSES,
            )
            return
        except (FileNotFoundError, RuntimeError) as exc:
            if strict and all_versions:
                # fsspec's rm removes CURRENT objects; on a versioned bucket it leaves every
                # non-current version in place (and adds a delete marker). So the fallback
                # cannot honour `all_versions`, and returning success from it tells a strict
                # caller a prefix is reclaimed when terabytes of old versions are still billed
                # — `_InputRetention.cleanup` then releases the cell's storage-budget slot and
                # admits another ingest. Best-effort callers still get the fallback; a caller
                # that said it cannot proceed on an unclean prefix gets told.
                msg = (
                    f"s5cmd could not delete {uri} ({exc}) and the fsspec fallback cannot remove "
                    "non-current versions, so an all-versions delete cannot be honoured here. "
                    "Install s5cmd on this runner, or call with all_versions=False if current "
                    "objects alone are what the caller needs."
                )
                raise DeleteUnverifiedError(msg) from exc
            versions_note = " (non-current versions may remain)" if all_versions else ""
            log.warning("s5cmd rm of %s failed (%s) — falling back to fsspec%s", uri, exc, versions_note)
        else:
            return

    try:
        fs = fsspec.filesystem(fsspec.utils.get_protocol(uri))
        if fs.exists(uri):
            fs.rm(uri, recursive=True)
    except Exception:
        if strict:
            raise
        log.warning("Failed to delete prefix %s", uri, exc_info=True)
        return

    if not strict:
        return
    # READ THE FALLBACK BACK TOO. `strict` promises to raise when the delete "ran and left objects
    # behind", and `fs.rm` returning without error is not that guarantee — it reports per call, not
    # per prefix. The s5cmd path has always verified; this one did not, and it did not have to:
    # `all_versions` defaulted ON, so `strict and all_versions` above failed closed before fsspec
    # was ever tried. Turning that default off made this path reachable for a strict caller, which
    # is the caller whose next move is to release a retention slot on the strength of the answer.
    remaining = _survivors(uri, log)
    if remaining is None:
        raise DeleteUnverifiedError(
            f"deleted {uri} with the fsspec fallback but could not list it to confirm — a prefix "
            "that cannot be read is unknown, not clean"
        )
    if remaining:
        raise PrefixNotEmptyError(f"{uri} still holds {len(remaining)} object(s) after the fsspec fallback delete")
