"""Run a call under a deadline, so a HANG is a red test rather than a stuck runner.

No test runner reports a hang as a failure, so the subject runs on a daemon thread and a thread
still alive at the deadline fails. Shared because both the fork-phase watchdog and the assembly
backlog drain tests need it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


def run_under_deadline(seconds: float, fn: Callable[[], Any], *, what: str = "the call") -> Any:
    """Call ``fn``, failing if it has not returned in ``seconds``; re-raises on this thread."""
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
