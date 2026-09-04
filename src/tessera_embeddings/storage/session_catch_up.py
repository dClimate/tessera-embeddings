"""Keeping an assembly's coordinator session current while its forks write.

**The invariant this module exists to hold:** a session's base may only move past commits
that have been checked against the group being written, and it must never be left moved
without that check having happened.

**Why any of it.** A fill opens a session, forks it, writes for hours, then commits. Every cell
that publishes in those hours puts another snapshot between the session's base and the branch
tip, and the commit has to walk all of them inside ``rebase``. That walk is where seven of nine
assemblies stopped dead on 2026-08-29: ``asset_manager::fetch_snapshot`` reached from
``session::rebase`` never returned, and only ever when the walk was more than one cell long.
Catching up while the writing is still going on keeps every walk short.

**Why it is dangerous to do naively, which is most of the code here.** A conflict is only ever
detected against commits made AFTER a session's base, so moving the base past a commit is also a
decision to stop checking it. Measured on dev with a real collision: an unguarded catch-up
committed over a competing writer in silence, where stock icechunk correctly raises. Hence
:func:`_diff_touches` before moving, a re-check of what the move actually crossed, and
:func:`catch_up_best_effort`, which treats a failure as best-effort ONLY when the base did not
move.

Measurements and the full incident record are in
``context_docs/storage/writing-to-the-global-store.md``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import icechunk

_log = logging.getLogger(__name__)

#: The members of :class:`icechunk.Diff` that name NODES. Module-level so a test can assert the
#: list against a real diff — the enumeration is the guard's coverage, and an entry that
#: silently stopped existing would narrow what is checked with nothing to notice it.
#: ``updated_chunks`` and ``moved_nodes`` are shaped differently and are read separately.
_DIFF_NODE_FIELDS = (
    "new_groups",
    "updated_groups",
    "deleted_groups",
    "new_arrays",
    "updated_arrays",
    "deleted_arrays",
)


#: How long the timer is given to notice it has been asked to stop. `stop.set()` wakes a
#: healthy ticker immediately and a catch-up in flight takes about a second, so this only
#: matters when a catch-up is HUNG — which is the failure this whole module exists to bound.
CATCH_UP_STOP_TIMEOUT_S = 30.0


class CatchUpAbortedTheWaitError(RuntimeError):
    """A periodic catch-up failed, so waiting on the remaining forks is pointless.

    Raised only to END THE WAIT; the real cause is the tick's own exception, which
    :func:`ticking` re-raises when its block exits. This one stops an assembly spending another
    three hours and sixteen workers on a fill that can no longer commit.
    """


class CatchUpDidNotStopError(RuntimeError):
    """The timer thread was still inside a catch-up after being asked to stop.

    Raised rather than carried on with, because the session is not ours any more. The caller next
    merges the forks in and commits, and icechunk guards a session with a lock: a merge against a
    session a hung rebase still holds either blocks on that lock — one stuck call becomes a stuck
    fill — or, if the catch-up later returns, moves the base AFTER the write was merged, which is
    the expensive replay this module exists to avoid.
    """


class CaughtUpPastAConflictError(RuntimeError):
    """A catch-up advanced past a commit that had touched this group, so the eventual commit
    can no longer detect a collision with it.

    Its own type because it is the ONE failure in :func:`catch_up_to_branch` that must not be
    treated as best-effort: every other failure there costs an optimisation, while this leaves the
    session in a state where a genuine conflict would be overwritten in silence.
    """


def _diff_touches(diff: icechunk.Diff, group: str) -> bool:
    """Did anything in ``diff`` change inside ``group``?

    The safety half of :func:`catch_up_to_branch`. Moving a session's base past a commit is also
    a decision to stop looking at it: harmless for a commit that touched some other zone,
    unacceptable for one that touched ours.

    Deliberately GROUP-wide rather than chunk-exact. The campaign's concurrent writers are other
    zones, so a group-level test blocks essentially nothing in practice, while a chunk-exact test
    would have to model shard geometry to stay correct — precision bought with a second thing to
    get wrong, in the direction that fails silently.
    """
    own = f"/{group.strip('/')}"
    prefix = f"{own}/"
    paths: set[str] = set()
    # EVERY member of Diff, by DIRECT attribute access: `getattr(diff, name, default)` hid the
    # omission of `moved_nodes` below, and a future icechunk that renamed a field would silently
    # stop checking that whole class of change rather than raising.
    for attr in _DIFF_NODE_FIELDS:
        paths |= {str(node) for node in getattr(diff, attr)}
    paths |= {str(node) for node in diff.updated_chunks}
    # A rename populates `moved_nodes` and NOTHING else — no chunks, no arrays, no groups — so
    # omitting it made a commit that renamed our own array read as untouched. Both ends count:
    # moving a node out of our group changes it, and so does moving one in.
    for source, destination in diff.moved_nodes:
        paths |= {str(source), str(destination)}
    return any(path == own or path.startswith(prefix) for path in paths)


def catch_up_to_branch(
    repo: icechunk.Repository,
    session: icechunk.Session,
    group: str,
    *,
    branch: str = "main",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> str:
    """Move an idle coordinator session up to the branch tip. Returns what it did.

    Why an assembly needs it, and the 2026-08-29 hang that motivates it, are in the module
    docstring.

    **Why repeatedly, and not once at the end.** A single catch-up just before the commit would
    walk exactly the same distance the commit would have, through exactly the same call. The value
    is not in moving the base, it is in never letting the gap grow.

    **Why this is cheap and not a second commit.** During the fork phase the coordinator's session
    holds no changes of its own — the forks have not been merged yet — so there is nothing to
    replay. This moves a base pointer.

    **Why it can refuse.** See :func:`_diff_touches`: a commit that touched our own group is never
    skipped. The base stays where it is, the gap stops closing, and the commit's own rebase then
    behaves as it always has, including raising ``RebaseFailedError`` on a real collision.

    Returns:
        ``"current"`` — already at the tip, nothing to do.
        ``"advanced"`` — the base moved up to the tip.
        ``"blocked"`` — another writer has touched this group, so the base was left alone.
    """
    _logger = log or _log
    tip = repo.lookup_branch(branch)
    base = session.snapshot_id
    if tip == base:
        return "current"
    if _diff_touches(repo.diff(from_snapshot_id=base, to_snapshot_id=tip), group):
        _logger.info(
            "Not catching up %s: another writer has touched this group since %s, so the "
            "commit must still compare against it.",
            group,
            base,
        )
        return "blocked"
    session.rebase(icechunk.ConflictDetector())
    landed = session.snapshot_id
    if landed != tip and _diff_touches(repo.diff(from_snapshot_id=tip, to_snapshot_id=landed), group):
        # THE RACE, CLOSED AS FAR AS IT CAN BE. `rebase` takes no target snapshot — it goes to
        # whatever the tip is when it runs — so a commit landing between the check above and this
        # call is skipped without ever being vetted. That cannot be undone, but it must not pass
        # unnoticed: an empty session's rebase never raises whatever it walks over, so the commit
        # would overwrite the other writer in silence. Verified against a real repository.
        raise CaughtUpPastAConflictError(
            f"catch-up for {group} advanced from {base} to {landed}, past the vetted tip {tip}, "
            f"and a commit in that gap touched {group}; this session can no longer detect a "
            f"collision with it"
        )
    _logger.info("Caught up %s from %s to branch tip %s while its forks write.", group, base, landed)
    return "advanced"


def catch_up_best_effort(
    repo: icechunk.Repository,
    session: icechunk.Session,
    group: str,
    *,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> str:
    """:func:`catch_up_to_branch`, but a failure costs the optimisation rather than the fill.

    The catch-up is an optimisation; the commit's own rebase is the correctness mechanism and is
    untouched. It runs on a timer for three hours and then once more AFTER every worker has
    succeeded and its fork is in hand, so an exception at that last call would discard a finished
    multi-hour write over a failed housekeeping call. The triggers are real and not all transient:
    a node rename anywhere in the store makes ``rebase`` raise even on an empty session, and a
    reset branch makes ``diff`` raise.

    :class:`CaughtUpPastAConflictError` is deliberately NOT caught: it reports that the session can
    no longer detect a collision, not a failed optimisation.

    Returns:
        Whatever :func:`catch_up_to_branch` returned, or ``"failed"``.
    """
    before = session.snapshot_id
    try:
        return catch_up_to_branch(repo, session, group, log=log)
    except CaughtUpPastAConflictError:
        raise
    except Exception as exc:
        # THE TEST IS WHETHER THE BASE MOVED, not which exception was raised. `rebase` advances
        # incrementally and can leave the session advanced when a later step throws, and the
        # post-rebase verification is itself a network call, so a handler keyed on exception TYPE
        # swallows exactly the case that matters. Base unmoved: a lost optimisation. Base moved:
        # unvetted commits were crossed, and merging the forks onto that base would put a
        # same-group collision beyond the commit's conflict detection.
        if session.snapshot_id != before:
            raise CaughtUpPastAConflictError(
                f"catch-up for {group} failed after advancing {before} -> {session.snapshot_id}; "
                f"the commits it crossed were never checked against this group"
            ) from exc
        (log or _log).warning(
            "Catch-up for %s failed with the base unmoved at %s; the commit's own rebase still "
            "applies, so this costs speed and not correctness.",
            group,
            before,
            exc_info=True,
        )
        return "failed"


#: How often an idle coordinator catches up while its forks write. Its OWN cadence, not the
#: progress timer's five minutes, which would couple a housekeeping decision to a reporting one
#: and let the gap grow for a full reporting period before anything looked at it.
#:
#: FIVE SECONDS, set by two bounds. A cell publishes TWO snapshots — the fill and the completion
#: mark — so depth counts in twos, and two publications inside one interval is the depth that
#: hangs (2026-08-31: depth 2 fine 42+ times, depth 4 wedged 10 for 10). Bounded BELOW by the
#: 0-1 s gap within one publication: shorter starts landing a tick between a cell's own fill and
#: mark, an odd depth nothing has tested. Bounded ABOVE by the closest independent pair observed,
#: 11 s.
#:
#: Shorter also means MORE rebases (a close pair becomes two shallow hops rather than one deep
#: one), which is the trade, because the hazard tracks depth and not call count. A no-op tick is
#: one branch-tip ref read returning ``"current"`` before any diff or rebase, so the count of
#: EXPENSIVE calls tracks how many cells publish, not how often we look: measured 6 rebases
#: against 437 no-op ticks over a 222-minute fork phase.
CATCH_UP_INTERVAL_S = 5.0


def rehome_after_a_wedged_catch_up(
    repo: icechunk.Repository,
    group: str,
    *,
    base: str,
    branch: str = "main",
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> icechunk.Session:
    """Open a fresh session for finished forks whose own session is no longer safe to touch.

    **Why a fresh session rather than a stopped thread.** A wedged catch-up cannot be stopped: it
    is blocked inside a pyo3 call into Rust where no Python mechanism reaches it — not a timeout,
    not a signal, not ``PyThreadState_SetAsyncExc`` — and on 2026-08-31 the two that wedged never
    returned. The session it holds can never be declared free, so the only move left is to stop
    sharing state with it.

    **Why the finished work survives that.** :func:`~...shard_writer.run_forked` raises from the
    timer's exit, which runs after its body, so every worker's fork is already in hand and
    undamaged. And a fork is not bound to the session that produced it: the normal path already
    merges forks made at one base into a session that has since rebased to the tip, which eight
    published cells did on 2026-08-31 across one to seven rebases. Verified against icechunk 2.1.1
    on a real repository — a fork merged into a session opened later commits, its data lands, and
    the commits it skipped keep both their chunks and their attrs.

    **What this must check, and why it is not optional.** Abandoning the old session also abandons
    the conflict detection its commit would have done over ``base..tip``. A fresh session's own
    rebase only covers what lands after it opens, so this walks the skipped range itself and
    refuses if anything in it touched ``group`` — the same rule, and reason, as
    :func:`catch_up_to_branch`. The walk is a read-only diff, which stayed at 0.1-0.4 s throughout
    every stall yet observed, so it is not on the path that wedges.

    Args:
        repo: The repository.
        group: The group being written, whose collisions must still be detectable.
        base: The abandoned session's base, captured BEFORE the fork phase — never read off
            the poisoned session, which another thread is inside.
        branch: Branch to open the fresh session on.
        log: Where to say that this happened; it should never pass unremarked.

    Returns:
        A fresh writable session at the branch tip, safe to merge the finished forks into.

    Raises:
        CaughtUpPastAConflictError: A commit in the skipped range touched ``group``, so
            re-homing would put a same-group collision beyond conflict detection. The cell
            fails, which is the cheap outcome.
    """
    fresh = repo.writable_session(branch)
    landed = fresh.snapshot_id
    if base != landed and _diff_touches(repo.diff(from_snapshot_id=base, to_snapshot_id=landed), group):
        raise CaughtUpPastAConflictError(
            f"cannot re-home {group} onto a fresh session: a commit between the abandoned base "
            f"{base} and the tip {landed} touched {group}, so the re-homed commit could not "
            f"detect a collision with it"
        )
    (log or _log).warning(
        "Re-homing %s onto a fresh session at %s: its catch-up wedged and cannot be stopped, so "
        "the finished forks are merged onto a session no other thread holds. The abandoned "
        "session was based at %s.",
        group,
        landed,
        base,
    )
    return fresh


@contextmanager
def ticking(
    interval_s: float,
    tick: Callable[[], None] | None,
    *,
    abort: threading.Event | None = None,
) -> Iterator[None]:
    """Run ``tick`` every ``interval_s`` for the duration of the block.

    ONE timer for the whole fork phase, whichever way the payloads run. Hooking the waiting loop
    instead would reach only the multi-process branch — a single payload runs inline and never
    enters it, so that path would get exactly the one deep catch-up at the end this exists to
    avoid.

    **The tick's exceptions are re-raised on the CALLER's thread.** A daemon thread's exception is
    discarded, and a tick can raise :class:`CaughtUpPastAConflictError`, which must fail the fill
    rather than vanish.

    ``abort`` is set the moment a tick fails, so a caller waiting on hours of work can stop early
    instead of finishing a fill already known to be uncommittable. Without it the failure surfaces
    only when this block exits — for an assembly, up to three hours and sixteen workers' worth of
    object-store writes later.

    **Only a caller that can act on it will.** ``run_forked`` always passes one, but its
    single-payload branch runs the worker inline and never looks at it. That branch is a one-shard
    cell or ``n_workers=1``, a short write by construction, so the abort would arrive at a fill
    about to end anyway; interrupting it would need a second cancellation mechanism that is not
    worth carrying for that case.
    """
    if tick is None:
        yield
        return
    stop = threading.Event()
    failure: list[BaseException] = []

    def _loop() -> None:
        while not stop.wait(interval_s):
            try:
                tick()
            except BaseException as exc:  # re-raised on the caller's thread; never swallowed
                failure.append(exc)
                if abort is not None:
                    abort.set()
                return

    ticker = threading.Thread(target=_loop, name="catch-up", daemon=True)
    ticker.start()
    try:
        yield
    finally:
        stop.set()
        ticker.join(timeout=CATCH_UP_STOP_TIMEOUT_S)
        if ticker.is_alive():
            # A HUNG CATCH-UP IS THE WHOLE POINT OF THIS MODULE, so the stop path must not assume
            # the thread noticed: `join` with a timeout returns either way, and continuing would
            # hand the session to the merge while another thread is still inside it.
            raise CatchUpDidNotStopError(
                f"a catch-up was still running {CATCH_UP_STOP_TIMEOUT_S:.0f}s after being asked "
                f"to stop; the session cannot be merged or committed while it is in use"
            )
        if failure:
            raise failure[0]
