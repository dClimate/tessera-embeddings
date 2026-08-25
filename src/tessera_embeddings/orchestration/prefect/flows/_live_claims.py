"""Which ``(zone, year)`` cells a LIVE flow run is already writing.

The store cannot answer this. A fill in flight has not written its year tag yet, so a
tag-only check calls a cell that something is currently filling "eligible" — and
dispatching on that answer puts two writers on one cell. The only place the answer
exists is the orchestrator's own list of live runs.

**Three ways a run claims a cell, and all three are needed.** A cell is named
differently by each kind of run that writes it:

1. a ``zone`` plus a ``year`` parameter — how the per-cell fill and the zone ingest
   name it;
2. an entry in a ``cells`` list — how a chained fill names its whole shard in one run;
3. **any parameter value containing a ``mosaics/<zone>/<year>`` path** — how a
   directly dispatched ROI ingest names it, because it takes a store path and carries
   no zone or year at all.

The rule that generalises them: match on the parameter that names the RESOURCE, not on
the parameters a particular caller happens to pass. Form 3 exists because a writer that
named its cell only in a path once read as idle, and a fill was dispatched against a
mosaic that was still growing.

**The polarity is deliberate and load-bearing.** Anything that cannot be shown to be
finished counts as a claim:

* every non-terminal state counts as live, ``CANCELLING`` included — a cancellation is
  a REQUEST, and the run keeps writing until a worker acts on it;
* the zone form is matched permissively, so an unfamiliar spelling over-claims rather
  than under-claims;
* no run is exempted by flow or deployment name. An unrecognised writer therefore
  reads as a claim. The alternative — an allow-list of known writers — fails silently
  in the one direction that matters, because a writer missing from the list reads as
  absence of a claim, which is indistinguishable from the cell being free.

A claim on a cell the caller was not going to dispatch is simply irrelevant: the caller
intersects this set with its own intended cells, so a read-only run that happens to
name a landed cell costs nothing.

Failure is not an empty answer. A caller that cannot reach the orchestrator has learned
nothing about who is writing, so this raises rather than returning an empty set, and the
caller must treat that as "claimed".
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterState, FlowRunFilterStateType
from prefect.client.schemas.objects import StateType

from tessera_embeddings.storage.zone_grid import canonicalize_zone

#: Runs fetched per page. The server-side default is 200; asking for it explicitly is
#: what makes "a short page is the last page" true rather than assumed.
_PAGE = 200

#: Every state in which a run may still be writing. ``CANCELLING`` belongs here: the
#: state means the cancellation has been requested, not that anything has stopped.
_LIVE_STATES = (
    StateType.RUNNING,
    StateType.PENDING,
    StateType.SCHEDULED,
    StateType.CANCELLING,
    StateType.PAUSED,
)

#: The written mosaic prefix — form 3. Permissive in the zone segment on purpose: an
#: unparseable zone must over-claim, never under-claim.
_MOSAIC_PREFIX = re.compile(r"mosaics/(?P<zone>[^/]+)/(?P<year>\d{4})")


def _zone(raw: object) -> str:
    """Canonical zone id, or the raw text when it will not parse.

    Keeping an unparseable spelling rather than dropping it holds the polarity: the
    claim set records everything it saw, and the caller's intersection decides what
    is relevant.
    """
    try:
        return canonicalize_zone(str(raw))
    except ValueError:
        return str(raw)


def claims_in(parameters: Mapping[str, Any] | None) -> set[tuple[str, int]]:
    """The cells one run's parameters claim, by all three naming forms.

    Split out from the query so the forms can be tested without an orchestrator.
    """
    claimed: set[tuple[str, int]] = set()
    params = parameters or {}
    zone, year = params.get("zone"), params.get("year")
    if isinstance(zone, str) and isinstance(year, int):
        claimed.add((_zone(zone), year))
    for cell in params.get("cells") or []:
        if isinstance(cell, list | tuple) and len(cell) == 2:
            try:
                claimed.add((_zone(cell[0]), int(cell[1])))
            except (TypeError, ValueError):
                continue
    # Form 3, scanned across every parameter VALUE rather than a named key, because the
    # run that needs it does not have a key for the cell — only a path that contains it.
    for value in params.values():
        for found in _MOSAIC_PREFIX.finditer(str(value)):
            claimed.add((_zone(found.group("zone")), int(found.group("year"))))
    return claimed


async def zone_years_claimed_by_live_runs() -> set[tuple[str, int]]:
    """Every cell any non-terminal flow run currently claims.

    Filtered to live states SERVER-side, so the query is bounded by what is running
    rather than by the campaign's history — a campaign accumulates thousands of
    terminal runs, and paging those to discard them would make this cost grow with
    the run's age.

    Raises:
        Exception: whatever the client raises. An unanswerable query means the caller
            has learned nothing, which must not read as "nothing is claimed".
    """
    claimed: set[tuple[str, int]] = set()
    state = FlowRunFilterState(type=FlowRunFilterStateType(any_=list(_LIVE_STATES)))
    async with get_client() as client:
        offset = 0
        while True:
            page = await client.read_flow_runs(flow_run_filter=FlowRunFilter(state=state), limit=_PAGE, offset=offset)
            for run in page:
                claimed |= claims_in(getattr(run, "parameters", None))
            if len(page) < _PAGE:
                return claimed
            offset += len(page)
