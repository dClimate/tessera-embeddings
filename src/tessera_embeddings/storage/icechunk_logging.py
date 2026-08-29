"""Ownership of icechunk's process-global log filter.

Its own module, and deliberately a light one: it imports nothing from this package, so a
caller that only wants to register a filter — the test suite's conftest, for one — does not
drag in the writer, its logging configuration, or anything else. Putting it in
``shard_writer`` did exactly that and broke ten unrelated log-capture tests.

The filter is global and write-only — icechunk offers a setter and no getter — so "put it back
how it was" is only possible if exactly one place tracks what "it was" is. That place is here.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress

import icechunk

#: What icechunk's own Rust tracing is raised to while a commit runs. Both modules, because
#: the two answer different halves of "where did it stop": ``session`` names the commit phase,
#: ``asset_manager`` names the storage operation inside it.
COMMIT_LOG_FILTER = "icechunk::session=debug,icechunk::asset_manager=debug"

#: THE ONE OWNER of icechunk's process-global log filter. It has a setter and no getter, so a
#: filter installed by anyone else is invisible to everyone else — which makes "put it back
#: how it was" impossible unless exactly one place tracks what "it was" is. Seeded from the
#: environment because that is icechunk's own entry point, read once at its import.
_base_log_filter: str | None = os.environ.get("ICECHUNK_LOG") or None
#: Overlapping commits share the one global filter, so the LAST one out restores it, not the
#: first. `plan()` commits an all-ocean cell on the feeder thread while the trailing thread
#: commits an assembly, so this overlap is a real schedule and not a hypothetical one.
_tracing_depth = 0
_tracing_lock = threading.Lock()


def set_base_logs_filter(directive: str | None) -> None:
    """Install a standing icechunk log filter, and record it as the one to return to.

    Call this instead of :func:`icechunk.set_logs_filter`. Going direct is not wrong so much
    as unrecoverable: the filter is process-global and write-only, so a directive set behind
    this module's back is one it cannot see, cannot preserve, and will overwrite the next time
    a commit finishes.

    Args:
        directive: A tracing directive, or ``None`` for icechunk's default.
    """
    global _base_log_filter
    with _tracing_lock:
        _base_log_filter = directive
        setter = getattr(icechunk, "set_logs_filter", None)
        if setter is not None:
            with suppress(Exception):
                setter(directive)


@contextmanager
def commit_tracing() -> Iterator[None]:
    """Raise icechunk's internal tracing for the duration of one commit.

    On 2026-08-29 seven of eleven assemblies reached this call and never returned — no
    exception, no CPU, no outstanding request, for hours. The commit is the one step in the
    pipeline whose insides we cannot see: everything before it reports progress, and it does
    not. Nothing in our own logs can distinguish a commit waiting on the session lock from one
    waiting inside a storage request, because our last line is printed BEFORE the call and the
    next only after it returns.

    icechunk instruments itself with ``tracing`` and exposes the filter, so the lines exist —
    they are simply switched off. Raised here, the last line before a stall names the function
    it entered and never left, with file and line.

    Scoped to the commit rather than set process-wide because the filter is GLOBAL and the
    fork-write phase would otherwise log a line per chunk for hours. A commit normally takes
    about a second, so the volume is nil and the value only appears when one does not.

    **A configured filter wins outright.** If anything has set one, this does nothing at all:
    it does not replace it, and it does not append to it. Appending would be worse than it
    sounds — ``EnvFilter`` resolves a target to its MOST SPECIFIC directive, so adding
    ``icechunk::session=debug`` beside an operator's ``icechunk=trace`` would quietly DOWNGRADE
    the very module they asked to see, during the very operation they are investigating. An
    operator who wants both can name both in ``ICECHUNK_LOG``.

    Reference-counted, because overlapping commits share the one global filter and the last
    one out must restore it. Best-effort: a version without the hook, or a failed set, must
    never cost a commit.
    """
    global _tracing_depth
    setter = getattr(icechunk, "set_logs_filter", None)
    # Bound only once the raise has actually succeeded, so the restore path cannot run
    # against a hook that was missing or that threw — and so its type is settled.
    restore: Callable[[str | None], None] | None = None
    with _tracing_lock:
        if setter is not None and _base_log_filter is None:
            try:
                if _tracing_depth == 0:
                    setter(COMMIT_LOG_FILTER)
            except Exception:  # diagnostics must never fail a commit
                restore = None
            else:
                _tracing_depth += 1
                restore = setter
    if restore is None:
        yield
        return
    try:
        yield
    finally:
        with _tracing_lock:
            _tracing_depth -= 1
            if _tracing_depth == 0:
                with suppress(Exception):
                    restore(_base_log_filter)
