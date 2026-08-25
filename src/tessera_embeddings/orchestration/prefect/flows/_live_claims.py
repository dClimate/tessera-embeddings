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
from prefect.client.schemas.objects import TERMINAL_STATES
from prefect.client.schemas.sorting import FlowRunSort

from tessera_embeddings.storage.zone_grid import canonicalize_zone

#: Runs fetched per page. The server-side default is 200; asking for it explicitly is
#: what makes "a short page is the last page" true rather than assumed.
_PAGE = 200

#: The states a run may be in and be FINISHED, taken from Prefect rather than listed
#: here. Everything else — running, pending, scheduled, cancelling, paused, and any
#: state a future Prefect adds — is treated as live.
#:
#: **The direction of this enumeration is the safety property.** Listing the LIVE states
#: instead would put the dangerous verdict in the fallback: a state absent from the list
#: reads as "not live", which is indistinguishable from the cell being free, and a state
#: added by a future release would silently join it. Listing the terminal states puts the
#: safe verdict in the fallback — anything unrecognised reads as a claim. ``CANCELLING``
#: is the case that makes this concrete: the cancellation has been REQUESTED, and the run
#: keeps writing until a worker acts on it.
_FINISHED_STATES = tuple(TERMINAL_STATES)

#: How many times the scan may be repeated in search of a stable answer before it gives
#: up. Bounded in BOTH directions on purpose: an unbounded stability loop inside a
#: dispatch decision is its own outage, and running out of passes must DECLINE rather
#: than accept the last thing it saw. Small, because two back-to-back queries disagreeing
#: repeatedly means the cluster is churning hard enough that waiting for the round is the
#: right answer anyway.
_STABILITY_PASSES = 4

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


class UnstableClaimScanError(RuntimeError):
    """The live-run set would not hold still long enough to be read consistently.

    Its own type because the caller must treat it exactly as it treats an unreachable
    server — as "nothing was learned", never as "nothing is claimed".
    """


async def _one_pass(client: Any, state: FlowRunFilterState) -> tuple[set[str], set[tuple[str, int]]]:  # noqa: ANN401
    """One offset walk: the run ids seen, and the cells they claim.

    Sorted by ID, which is the only key here that a run cannot change. Sorting by a
    TIME would let a run move between pages as it starts, so a record could cross the
    cursor and never be read; an id is assigned once and never moves.
    """
    ids: set[str] = set()
    claimed: set[tuple[str, int]] = set()
    offset = 0
    while True:
        page = await client.read_flow_runs(
            flow_run_filter=FlowRunFilter(state=state),
            sort=FlowRunSort.ID_DESC,
            limit=_PAGE,
            offset=offset,
        )
        for run in page:
            ids.add(str(getattr(run, "id", "")))
            claimed |= claims_in(getattr(run, "parameters", None))
        if len(page) < _PAGE:
            return ids, claimed
        offset += len(page)


async def zone_years_claimed_by_live_runs() -> set[tuple[str, int]]:
    """Every cell a run that is not finished currently claims.

    Filtered SERVER-side to runs that are not in a terminal state, so the cost is
    bounded by what is running rather than by the campaign's history — which
    accumulates thousands of finished runs, and paging those only to discard them would
    make this grow with the campaign's age rather than its width.

    **An offset walk over a live set is not a snapshot, and the failure is silent.** A
    run that finishes between two pages shifts every later record one place towards the
    front, so the next offset steps over one — and a stepped-over run's cell reads as
    unclaimed, which is the one wrong answer this function must never give. So the walk
    is REPEATED until two consecutive passes see the same set of run ids, and the cells
    are UNIONED across every pass: a record missed by one walk is caught by the next,
    and a union can only ever over-claim.

    Repetition also covers a run whose state row has not been written yet. Such a run is
    invisible to any state filter — the server compiles both the inclusive and the
    exclusive form to SQL that drops NULL — but it is only ever milliseconds from having
    a state, so a later pass sees it, and the id sets disagree until it does.

    Bounded at :data:`_STABILITY_PASSES`, and running out RAISES. A cluster churning
    faster than it can be read is a cluster whose caller should wait for the round.

    Raises:
        UnstableClaimScanError: the run set would not settle within the pass budget.
        Exception: whatever the client raises. An unanswerable query means the caller has
            learned nothing, which must not read as "nothing is claimed".
    """
    state = FlowRunFilterState(type=FlowRunFilterStateType(not_any_=list(_FINISHED_STATES)))
    async with get_client() as client:
        claimed: set[tuple[str, int]] = set()
        previous: set[str] | None = None
        for _ in range(_STABILITY_PASSES):
            ids, seen = await _one_pass(client, state)
            claimed |= seen
            if previous is not None and ids == previous:
                return claimed
            previous = ids
    raise UnstableClaimScanError(
        f"the set of live runs changed on every one of {_STABILITY_PASSES} consecutive scans, so no "
        f"consistent view of what is being written could be taken"
    )
