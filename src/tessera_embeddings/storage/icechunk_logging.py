"""Ownership of icechunk's process-global log filter.

Its own module, and deliberately a light one: it imports nothing from this package, so a
caller that only wants to register a filter — the test suite's conftest, for one — does not
drag in the writer, its logging configuration, or anything else. Putting it in
``shard_writer`` did exactly that and broke ten unrelated log-capture tests.

The filter is global and write-only — icechunk offers a setter and no getter — so "put it back
how it was" is only possible if exactly one place tracks what "it was" is. That place is here.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress

import icechunk

_log = logging.getLogger(__name__)

#: What icechunk's own Rust tracing is raised to while a commit runs. EVERY icechunk module,
#: not a chosen few: the point is to know where a stall stopped, and naming modules in advance
#: assumes we already know which one that is. A commit normally takes about a second, so the
#: volume is nil, and the one time it is not nil is the time we need it.
COMMIT_LOG_FILTER = "icechunk=debug"

#: How long a commit may run before the alarm starts. Every commit ever measured finished in
#: 0.6-1.6 s, so five minutes is roughly two hundred times the worst observed and cannot fire
#: on a healthy one.
COMMIT_ALARM_S = 300.0

#: Dump every thread's Python stack on the first alarm and then every tenth, so a stall that
#: lasts hours keeps proving it is still stuck without writing 140 stacks every five minutes.
STACK_DUMP_EVERY = 10

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

    **Installed first, recorded only if that succeeded, and errors are NOT swallowed.** The
    order matters: recording a directive icechunk then rejected would leave every later
    :func:`commit_tracing` believing an operator filter was in force, so it would skip the
    commit diagnostics — on the strength of a filter that was never installed, and without
    the caller hearing that their configuration failed. Unlike :func:`commit_tracing`, which
    must never cost a commit, this is a configuration call made at start-up: a malformed
    directive is a caller's mistake and is worth hearing about.

    Args:
        directive: A tracing directive, or ``None`` for icechunk's default.

    Raises:
        Exception: Whatever icechunk raises for a directive it will not accept. The
            previously recorded base is left untouched.
    """
    global _base_log_filter
    with _tracing_lock:
        setter = getattr(icechunk, "set_logs_filter", None)
        if setter is not None:
            setter(directive)
        _base_log_filter = directive


@contextmanager
def _commit_alarm(what: str) -> Iterator[None]:
    """Shout, repeatedly, for as long as a commit has not returned — and dump the stacks.

    A stalled commit is silent by construction: our own last line is printed before the call
    and the next only after it returns, so hours of nothing look exactly like a commit that
    finished. On 2026-08-29 that silence is what made seven stalled assemblies take a day to
    even localise. An alarm that repeats is the difference between "this stopped at 09:04"
    and "we noticed at 16:00".

    **The stack dump is the part that could not be got any other way.** `py-spy` needs
    `CAP_SYS_PTRACE`, which Fargate does not grant, and `/proc/<pid>/syscall` and
    `/proc/<pid>/stack` are gated the same way — all three were tried against a live stalled
    task and all three refused. `faulthandler` runs INSIDE the process, so it needs no
    permission at all, and it prints every thread's Python stack. That names the exact call
    the assembly thread is parked in, and shows what the other threads are doing beside it.

    Dumped on the first alarm and every tenth after, because a stall lasting hours should keep
    proving it is still stuck without writing 140 stacks every five minutes.

    Best-effort throughout, on its own daemon thread: an alarm that can end the work it is
    watching, or that dies on one failed emission and lets the silence back in, is worse than
    no alarm at all.
    """
    returned = threading.Event()

    def _watch() -> None:
        rings = 0
        while not returned.wait(timeout=COMMIT_ALARM_S):
            rings += 1
            with suppress(Exception):
                _log.critical(
                    "ASSEMBLY COMMIT STALLED: %r has not returned after %.0f minutes. This fill "
                    "publishes nothing further until it does. icechunk's own tracing is raised "
                    "for this commit — its last line names where it stopped.",
                    what,
                    rings * COMMIT_ALARM_S / 60.0,
                )
            if rings == 1 or rings % STACK_DUMP_EVERY == 0:
                with suppress(Exception):
                    _log.critical("ASSEMBLY COMMIT STALLED: thread stacks follow (ring %d)", rings)
                    faulthandler.dump_traceback()

    threading.Thread(target=_watch, name="commit-alarm", daemon=True).start()
    try:
        yield
    finally:
        returned.set()


def traced_commit(
    session: icechunk.Session,
    message: str,
    *,
    rebase_with: icechunk.ConflictSolver | None = None,
    rebase_tries: int = 1_000,
) -> str:
    """Commit with icechunk's tracing raised — the ONE way an assembly reaches a commit.

    A helper rather than a rule to remember, because the value only shows up on the one
    occasion nobody is watching. Every commit in an assembly can stall the same way, and a
    commit that stalls outside a tracing scope produces exactly the silence that made the
    2026-08-29 incident take a day to localise. Routing them all through here means a commit
    added later is instrumented by default rather than by whoever remembers.

    Args:
        session: The icechunk session to commit.
        message: The commit message.
        rebase_with: Conflict solver for the fill's two commits; ``None`` for the schema
            and time-axis commits, which are the only writer on a fresh work branch.
        rebase_tries: How many times to rebase and retry, when a solver is given.

    Returns:
        The new snapshot id.
    """
    with commit_tracing(), _commit_alarm(message):
        return session.commit(message, rebase_with=rebase_with, rebase_tries=rebase_tries)


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
