"""Cancel a flow's child deployment runs when the parent is cancelled or crashes.

A flow that dispatches work with ``run_deployment``/``arun_deployment`` gets an
independently scheduled child run. Killing the parent does not touch it: the child
keeps going, keeps writing, and keeps its Dask or Ray fleet billing. Prefect's
subflow link does not help here either — several of these dispatches happen from
worker threads, so there is no parent-run relationship to walk.

The mechanism is a deterministic TAG derived from the parent's flow-run id alone.
Prefect runs terminal hooks in a FRESH import of the module after killing the flow
process, so nothing in memory survives to tell the hook what it started; a tag it
can re-derive from its ``flow_run`` argument is the only thing that does. The
parent stamps the tag on every child it dispatches, and the hook asks the server
for live runs carrying it.

Register the hook as BOTH ``on_cancellation`` and ``on_crashed`` — a crashed
parent orphans children exactly like a cancelled one — and keep it idempotent:
cancelling a parent and a child together delivers the transition twice.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from prefect.client.orchestration import get_client
from prefect.states import Cancelling

#: Runs fetched per page when sweeping children by tag. Prefect's server-side default
#: is 200; asking explicitly is what makes "a short page means the last page" true.
_PAGE = 200


def child_run_tag(prefix: str, flow_run_id: object) -> str | None:
    """The tag a parent stamps on its children, or ``None`` outside a flow run.

    ``None`` means there is no id to derive from (a direct ``.fn`` call in tests),
    in which case children simply go untagged and the hook has nothing to sweep.
    """
    return f"{prefix}:{flow_run_id}" if flow_run_id else None


def make_child_cancel_hook(prefix: str, what: str) -> Callable[..., None]:
    """A cancellation/crash hook that cancels live children tagged with ``prefix``.

    ``what`` names the children in log messages ("child ingest run", "campaign
    child run") — the operator reading a cancelled run's logs needs to know which
    fleet was swept.

    Returned as ``Callable[..., None]`` rather than a concrete three-argument type:
    Prefect's ``FlowStateHook`` is declared with ``*args, **kwargs``, and a precise
    signature here fails to match it at the registration site.

    Every failure here is logged and swallowed. This runs while the flow is already
    terminating, and a hook that raises would mask the flow's own terminal state;
    the log line names the tag so the sweep can be finished by hand.
    """

    def _hook(flow: object, flow_run: object, state: object) -> None:  # noqa: ARG001
        log = logging.getLogger(__name__)
        tag = child_run_tag(prefix, getattr(flow_run, "id", None))
        if not tag:
            log.warning("No flow-run id — cannot sweep %ss by tag. Check the Prefect UI manually.", what)
            return
        try:
            from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterTags

            with get_client(sync_client=True) as client:
                # PAGE. `limit=None` takes the server default (200), and a campaign
                # dispatches one child per (zone, year) — thousands over its life, all
                # carrying this tag long after they finish. A live child outside the
                # first page would never be cancelled, and its Ray fleet would keep
                # billing and keep writing the mosaic a retry is about to rebuild.
                # Filtering to live states server-side would be the cheaper query, but
                # the state filter is what decides which runs a page contains, and
                # getting the terminal-state set wrong here fails silently in exactly
                # the same way — so page over everything and judge finality locally,
                # where `is_final()` is the authority.
                live: list[Any] = []
                offset = 0
                while True:
                    page = client.read_flow_runs(
                        flow_run_filter=FlowRunFilter(tags=FlowRunFilterTags(all_=[tag])),
                        limit=_PAGE,
                        offset=offset,
                    )
                    live.extend(r for r in page if r.state is None or not r.state.is_final())
                    if len(page) < _PAGE:
                        break
                    offset += len(page)
                if live:
                    log.warning("Cancelling %d live %s(s) tagged %r", len(live), what, tag)
                for r in live:
                    try:
                        client.set_flow_run_state(r.id, state=Cancelling())
                    except Exception:
                        log.warning("Could not cancel %s %s", what, r.id, exc_info=True)
        except Exception:
            log.warning("%s sweep failed — check the Prefect UI for tag %r", what.capitalize(), tag, exc_info=True)

    return _hook
