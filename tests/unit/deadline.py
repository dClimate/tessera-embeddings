"""Run a call under a deadline, so a HANG is a red test rather than a stuck runner.

Several of the assembly safeguards exist because a call never returned: a fork worker that
would not finish, a pool shutdown that would not return, an assembly backlog drain that parked
forever. A test that simply calls such a subject can only hang when the fix is absent, and no
test runner reports a hang as a failure — it reports nothing at all until someone kills it.

:func:`run_under_deadline` turns the hang itself into the assertion: the subject runs on a
daemon thread, the test thread joins with a timeout, and a thread still alive at the deadline
fails. Exceptions are re-raised on the test thread so ``pytest.raises`` still works around it.

It lives here rather than in one test module because both the fork-phase watchdog
(``tests/unit/assembly/test_shard_writer.py``) and the backlog drain
(``tests/unit/orchestration/runners/test_sequential_fill.py``) need exactly this, and two
copies of a test primitive drift into two different notions of what a hang means.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


def run_under_deadline(seconds: float, fn: Callable[[], Any], *, what: str = "the call") -> Any:
    """Call ``fn`` and fail the test if it has not returned within ``seconds``.

    Returns whatever ``fn`` returned; re-raises whatever it raised, on the caller's thread.
    """
    box: dict[str, Any] = {}

    def _go() -> None:
        try:
            box["ret"] = fn()
        except BaseException as exc:  # re-raised on the test thread below
            box["exc"] = exc

    thread = threading.Thread(target=_go, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"{what} did not return within {seconds}s: it hung"
    if "exc" in box:
        raise box["exc"]
    return box["ret"]
