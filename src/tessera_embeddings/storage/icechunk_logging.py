"""Ownership of icechunk's process-global log filter.

Deliberately a light module: it imports nothing from this package, so a caller that only wants to
register a filter — the test suite's conftest, for one — does not drag in the writer or its logging
configuration. Living in ``shard_writer`` did exactly that and broke ten unrelated log-capture
tests.

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

#: What icechunk's own Rust tracing is raised to while a commit runs. EVERY icechunk module, not a
#: chosen few: naming modules in advance assumes we already know which one a stall stops in.
#:
#: THE BARE ``warn`` IS LOAD-BEARING. ``set_logs_filter`` replaces the whole filter rather than
#: adding to it, so a directive naming only icechunk would switch every OTHER target down for the
#: duration — warnings and errors included, and on a stall that duration is forever.
COMMIT_LOG_FILTER = "warn,icechunk=debug"

#: How long a commit may run before the alarm starts. Every commit ever measured finished in
#: 0.6-1.6 s, so five minutes is roughly two hundred times the worst observed and cannot fire
#: on a healthy one.
COMMIT_ALARM_S = 300.0

#: Dump every thread's Python stack on the first alarm and then every tenth, so a stall that
#: lasts hours keeps proving it is still stuck without writing 140 stacks every five minutes.
STACK_DUMP_EVERY = 10

#: THE ONE OWNER of icechunk's process-global log filter (setter, no getter — a filter installed
#: behind this module's back is invisible and unrestorable). Seeded from the environment because
#: that is icechunk's own entry point, read once at its import.
_base_log_filter: str | None = os.environ.get("ICECHUNK_LOG") or None
#: Overlapping commits share the one global filter, so the LAST one out restores it, not the
#: first. `plan()` commits an all-ocean cell on the feeder thread while the trailing thread
#: commits an assembly, so this overlap is a real schedule and not a hypothetical one.
_tracing_depth = 0
_tracing_lock = threading.Lock()


def set_base_logs_filter(directive: str | None) -> None:
    """Install a standing icechunk log filter, and record it as the one to return to.

    Call this instead of :func:`icechunk.set_logs_filter`: the filter is process-global and
    write-only, so a directive set behind this module's back is one it cannot see, cannot
    preserve, and will overwrite the next time a commit finishes.

    **Installed first, recorded only if that succeeded, and errors are NOT swallowed.** Recording
    a directive icechunk then rejected would leave every later :func:`commit_tracing` believing an
    operator filter was in force and skipping the commit diagnostics, on the strength of a filter
    that was never installed. This is a start-up configuration call, not a commit path: a malformed
    directive is worth raising.

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

    A stalled commit is silent by construction: our last line is printed before the call and the
    next only after it returns, so hours of nothing look exactly like a commit that finished. On
    2026-08-29 that silence made seven stalled assemblies take a day to even localise.

    **The stack dump could not be got any other way.** `py-spy` needs `CAP_SYS_PTRACE`, which
    Fargate does not grant, and `/proc/<pid>/syscall` and `/proc/<pid>/stack` are gated the same
    way — all three were tried against a live stalled task and refused. `faulthandler` runs INSIDE
    the process, needs no permission, and prints every thread's Python stack, naming the exact call
    the assembly thread is parked in.

    Best-effort throughout, on its own daemon thread: an alarm that can end the work it watches, or
    that dies on one failed emission, is worse than no alarm.
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

    try:
        threading.Thread(target=_watch, name="commit-alarm", daemon=True).start()
    except Exception:  # a diagnostic must never be the reason a commit does not happen
        # `Thread.start` raises `RuntimeError` when the OS will not give out another thread.
        # Letting that escape would abort the commit for want of an alarm about it — the
        # exact inversion this whole module exists to avoid. Commit unwatched instead.
        _log.exception("Could not start the commit alarm; committing without it")
        yield
        return
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

    A helper rather than a rule to remember: every commit in an assembly can stall the same way,
    and one that stalls outside a tracing scope produces exactly the silence that made the
    2026-08-29 incident take a day to localise. Routing them all through here instruments a
    commit added later by default.

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

    On 2026-08-29 seven of eleven assemblies reached this call and never returned — no exception,
    no CPU, no outstanding request, for hours. The commit is the one pipeline step whose insides we
    cannot see, and nothing in our own logs distinguishes a commit waiting on the session lock from
    one waiting inside a storage request. icechunk instruments itself with ``tracing`` and exposes
    the filter, so those lines exist and are merely switched off; raised here, the last line before
    a stall names the function it entered and never left, with file and line.

    Scoped to the commit rather than set process-wide because the filter is GLOBAL and the
    fork-write phase would otherwise log a line per chunk for hours.

    **A configured filter wins outright** — this neither replaces nor appends to it. Appending is
    the trap: ``EnvFilter`` resolves a target to its MOST SPECIFIC directive, so adding
    ``icechunk::session=debug`` beside an operator's ``icechunk=trace`` would quietly DOWNGRADE the
    module they asked to see. An operator who wants both names both in ``ICECHUNK_LOG``.

    Reference-counted, because overlapping commits share the one global filter and the last one out
    must restore it. Best-effort: a missing hook or a failed set must never cost a commit.
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
