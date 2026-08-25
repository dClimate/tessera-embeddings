"""Which cells a live run claims — the admission control an immediate re-dispatch needs.

The store cannot answer "is something writing this cell": a fill in flight has not written
its tag yet. Only the orchestrator's list of live runs can, and every rule here exists so
that an incomplete understanding of who writes errs towards REFUSING a dispatch.
"""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from prefect.client.schemas.objects import TERMINAL_STATES, StateType
from prefect.client.schemas.sorting import FlowRunSort

from tessera_embeddings.orchestration.prefect.flows import _live_claims as mod


class TestTheThreeClaimForms:
    """A cell is named differently by each kind of run that writes it, and all three
    namings must register — the rule being to match the parameter that names the
    RESOURCE, not the parameters one caller happens to pass.
    """

    def test_a_zone_and_year_pair(self):
        assert mod.claims_in({"zone": "33N", "year": 2021}) == {("33N", 2021)}

    def test_a_chained_runs_whole_shard(self):
        params = {"cells": [["33N", 2021], ["34N", 2020]]}
        assert mod.claims_in(params) == {("33N", 2021), ("34N", 2020)}

    def test_a_written_mosaic_path_in_any_parameter(self):
        """The only naming a directly dispatched ROI ingest carries.

        It takes a store path and no zone or year at all, so a check that looked at named
        keys would read it as idle while it was still committing.
        """
        params = {"store_path": "s3://in/mosaics/47S/2021/reflectance"}
        assert mod.claims_in(params) == {("47S", 2021)}

    def test_a_run_claiming_nothing_claims_nothing(self):
        assert mod.claims_in({"paths": {"inputs": "s3://in"}, "num_actors": 250}) == set()
        assert mod.claims_in(None) == set()


class TestThePolarityIsTowardsRefusal:
    """Every ambiguity resolves to "claimed", because the opposite error dispatches a
    second writer onto a cell something is still building.
    """

    def test_an_unpadded_zone_matches_the_canonical_one(self):
        """Otherwise a claim spelled one way silently fails to block a cell spelled the
        other, and the caller's intersection finds nothing.
        """
        assert mod.claims_in({"zone": "7S", "year": 2021}) == {("07S", 2021)}
        assert mod.claims_in({"store_path": "s3://in/mosaics/7s/2021/"}) == {("07S", 2021)}

    def test_an_unparseable_zone_is_still_recorded(self):
        """Recorded rather than dropped: the set reports everything it saw, and the caller
        decides what is relevant. Dropping it would be the one direction that loses a claim.
        """
        assert mod.claims_in({"zone": "not-a-zone", "year": 2021}) == {("not-a-zone", 2021)}

    def test_a_malformed_cell_entry_does_not_lose_the_rest(self):
        params = {"cells": [["33N"], ["34N", "not-a-year"], ["35N", 2020]]}
        assert mod.claims_in(params) == {("35N", 2020)}

    def test_the_decision_cannot_depend_on_who_is_running(self):
        """No allow-list of known writers, and structurally so: the decision sees a run's
        PARAMETERS and nothing else — no flow name, no deployment.

        An allow-list fails silently in the one direction that matters, because a writer
        missing from it reads as an absence of claims, which is indistinguishable from the
        cell being free. Having no identity to consult is what makes that impossible rather
        than merely avoided.
        """
        assert list(inspect.signature(mod.claims_in).parameters) == ["parameters"]

    def test_the_finished_set_is_prefects_own(self):
        """Not a list maintained here. A state type Prefect adds later must not need an
        edit in this file to be treated correctly, and the only way to guarantee that is
        to name the terminal set by reference.
        """
        assert set(mod._FINISHED_STATES) == set(TERMINAL_STATES)

    def test_the_enumeration_names_the_finished_states_not_the_live_ones(self):
        """The DIRECTION of the list is the safety property.

        Enumerating live states puts the dangerous verdict in the fallback: anything not
        listed reads as "not live", which is indistinguishable from the cell being free.
        Enumerating finished states puts the safe verdict there — anything unrecognised
        reads as a claim. This asserts the consequence rather than the spelling: every
        non-terminal state, including one this test invents, must count as live.
        """
        finished = set(mod._FINISHED_STATES)
        assert StateType.CANCELLING not in finished, "a cancellation is a request, not a stop"
        assert StateType.PAUSED not in finished
        assert not finished - {StateType.COMPLETED, StateType.FAILED, StateType.CRASHED, StateType.CANCELLED}


def _client(scans):
    """A stand-in Prefect client serving one list of pages per SCAN.

    A scan is identified by its `offset=0` request, so the fake can answer the repeated
    walks the stability loop makes with a different world each time. The last scan is
    repeated once the list runs out.
    """
    calls: list[dict] = []
    where = {"scan": -1}

    class _C:
        async def read_flow_runs(self, *, flow_run_filter=None, sort=None, limit=None, offset=None):
            if offset == 0:
                where["scan"] += 1
            calls.append(
                {"filter": flow_run_filter, "sort": sort, "limit": limit, "offset": offset, "scan": where["scan"]}
            )
            pages = scans[min(where["scan"], len(scans) - 1)]
            index = offset // limit if limit else 0
            return pages[index] if index < len(pages) else []

    @asynccontextmanager
    async def _get_client():
        yield _C()

    return _get_client, calls


def _run(rid="r", **parameters):
    return SimpleNamespace(id=rid, parameters=parameters)


@pytest.mark.asyncio
async def test_the_query_unions_every_live_runs_claims(monkeypatch):
    scan = [[_run("a", zone="33N", year=2021), _run("b", cells=[["34N", 2020]])]]
    get_client, _calls = _client([scan, scan])
    monkeypatch.setattr(mod, "get_client", get_client)
    assert await mod.zone_years_claimed_by_live_runs() == {("33N", 2021), ("34N", 2020)}


@pytest.mark.asyncio
async def test_the_query_excludes_the_finished_rather_than_including_the_live(monkeypatch):
    """Asserted on the wire, because this is where the polarity is actually spent.

    An inclusive filter would have to name every live state, and a state it failed to
    name would come back as an absence of claims.
    """
    get_client, calls = _client([[[]], [[]]])
    monkeypatch.setattr(mod, "get_client", get_client)
    await mod.zone_years_claimed_by_live_runs()
    state_type = calls[0]["filter"].state.type
    assert state_type.any_ is None, "an inclusive live-state filter cannot fail safe"
    assert set(state_type.not_any_) == set(TERMINAL_STATES)


@pytest.mark.asyncio
async def test_the_query_sorts_by_a_key_a_run_cannot_change(monkeypatch):
    """Sorting by a TIME would let a run move between pages as it starts, so a record
    could cross the cursor and never be read at all. An id is assigned once.
    """
    get_client, calls = _client([[[]], [[]]])
    monkeypatch.setattr(mod, "get_client", get_client)
    await mod.zone_years_claimed_by_live_runs()
    assert calls[0]["sort"] == FlowRunSort.ID_DESC


@pytest.mark.asyncio
async def test_the_query_pages_past_the_first_page(monkeypatch):
    """A full page is never the last one. A live campaign runs more than one page of
    ingests, and a claim beyond the first page would read as no claim at all.
    """
    scan = [[_run(f"f{i}", zone="33N", year=2021) for i in range(mod._PAGE)], [_run("z", zone="59S", year=2019)]]
    get_client, calls = _client([scan, scan])
    monkeypatch.setattr(mod, "get_client", get_client)
    assert await mod.zone_years_claimed_by_live_runs() == {("33N", 2021), ("59S", 2019)}
    assert [c["offset"] for c in calls if c["scan"] == 0] == [0, mod._PAGE]


@pytest.mark.asyncio
async def test_a_run_visible_only_on_a_later_pass_is_still_claimed(monkeypatch):
    """The reason the scan repeats at all, and it covers two failures with one mechanism.

    An offset walk over a live set is not a snapshot: a run that finishes between two
    pages shifts later records forward and the next offset steps over one. And a run
    whose state row has not been written yet is invisible to ANY state filter, because
    the server compiles both the inclusive and the exclusive form to SQL that drops
    NULL. Either way the run is missing from one pass and present in the next, and the
    union is what makes the miss harmless.
    """
    first = [[_run("a", zone="33N", year=2021)]]
    later = [[_run("a", zone="33N", year=2021), _run("b", zone="47S", year=2021)]]
    get_client, _calls = _client([first, later, later])
    monkeypatch.setattr(mod, "get_client", get_client)
    assert await mod.zone_years_claimed_by_live_runs() == {("33N", 2021), ("47S", 2021)}


@pytest.mark.asyncio
async def test_a_set_that_never_settles_declines_rather_than_guessing(monkeypatch):
    """Bounded in BOTH directions: a hard cap on passes, and running out must DECLINE.

    An unbounded stability loop inside a dispatch decision is its own outage; accepting
    the last thing seen would be the fail-open answer this whole module exists to avoid.
    """
    scans = [[[_run(f"gen{i}", zone="33N", year=2021)]] for i in range(mod._STABILITY_PASSES + 2)]
    get_client, calls = _client(scans)
    monkeypatch.setattr(mod, "get_client", get_client)
    with pytest.raises(mod.UnstableClaimScanError, match="consistent view"):
        await mod.zone_years_claimed_by_live_runs()
    assert len({c["scan"] for c in calls}) == mod._STABILITY_PASSES, "the pass budget was not honoured"
