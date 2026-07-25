"""Which Prefect execution site is running a terminal-state hook.

Cancellation hooks were observed executing **twice** for a single ``Cancelling``
transition (campaign ingest, 2026-07-25): two entries into the sweep 1 ms apart,
two completions 72 ms apart. Harmless there — both teardown hooks are idempotent
— but it doubles their cost and API calls at campaign scale, and it constrains
what the hooks may safely do later.

Prefect 3.7 has exactly two sites that run a flow's ``on_cancellation`` /
``on_crashed`` lists, and they are *meant* to be mutually exclusive:

- the flow engine, **inside the run's own process** (``flow_engine.py``), gated
  on ``_engine_owns_cancellation_and_crash_handling()``; and
- the runner that observes the state change, in a **separate process**
  (``runner.py`` → ``_run_on_cancellation_hooks``).

The gate is the env var ``PREFECT__ENABLE_CANCELLATION_AND_CRASHED_HOOKS``,
which runner-managed subprocesses set to ``"false"`` so only the runner fires.
It defaults to ``"true"``, and nothing in ``prefect_aws`` sets it — so a child
launched by an ECS *worker* (a different mechanism from ``Runner``) would have
the engine claim ownership while an observer also sweeps. That is the leading
hypothesis, but it is not yet confirmed against a live cancel, and a competing
explanation (the observed cancel set ``Cancelling`` on both child and parent)
has not been ruled out either.

:func:`hook_invocation_site` records what a single line of a hook's own output
can settle: whether the two executions come from one process or two, and whether
the engine believed it owned the transition. Cheap, and it means the next
cancellation names the cause instead of costing another investigation round.
"""

from __future__ import annotations

import os
import threading


def hook_invocation_site() -> str:
    """A short tag identifying this hook execution's process and ownership state.

    ``pid``/``thread`` distinguish "one process ran it twice" from "two processes
    each ran it once"; ``engine_owns`` reports the env gate above, so a doubled
    pair showing ``engine_owns=True`` on one line and the runner's process on the
    other confirms the split-ownership hypothesis outright.
    """
    owns = os.environ.get("PREFECT__ENABLE_CANCELLATION_AND_CRASHED_HOOKS", "true").lower() == "true"
    return f"pid={os.getpid()} thread={threading.get_ident()} engine_owns={owns}"
