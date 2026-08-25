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
from prefect.client.schemas.objects import StateType

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

    def test_cancelling_counts_as_live(self):
        """A cancellation is a REQUEST. The run keeps writing until a worker acts on it, so
        treating CANCELLING as finished is exactly the window a prompt retry would race.
        """
        assert StateType.CANCELLING in mod._LIVE_STATES

    def test_no_terminal_state_counts_as_live(self):
        assert not {StateType.COMPLETED, StateType.FAILED, StateType.CRASHED, StateType.CANCELLED} & set(
            mod._LIVE_STATES
        )

    def test_the_decision_cannot_depend_on_who_is_running(self):
        """No allow-list of known writers, and structurally so: the decision sees a run's
        PARAMETERS and nothing else — no flow name, no deployment.

        An allow-list fails silently in the one direction that matters, because a writer
        missing from it reads as an absence of claims, which is indistinguishable from the
        cell being free. Having no identity to consult is what makes that impossible rather
        than merely avoided.
        """
        assert list(inspect.signature(mod.claims_in).parameters) == ["parameters"]


def _client(pages):
    """A stand-in Prefect client whose ``read_flow_runs`` serves ``pages`` in order."""
    calls: list[dict] = []

    class _C:
        async def read_flow_runs(self, *, flow_run_filter=None, limit=None, offset=None):
            calls.append({"filter": flow_run_filter, "limit": limit, "offset": offset})
            index = offset // limit if limit else 0
            return pages[index] if index < len(pages) else []

    @asynccontextmanager
    async def _get_client():
        yield _C()

    return _get_client, calls


def _run(**parameters):
    return SimpleNamespace(parameters=parameters)


@pytest.mark.asyncio
async def test_the_query_unions_every_live_runs_claims(monkeypatch):
    get_client, _calls = _client([[_run(zone="33N", year=2021), _run(cells=[["34N", 2020]])]])
    monkeypatch.setattr(mod, "get_client", get_client)
    assert await mod.zone_years_claimed_by_live_runs() == {("33N", 2021), ("34N", 2020)}


@pytest.mark.asyncio
async def test_the_query_filters_to_live_states_server_side(monkeypatch):
    """Server-side, so the cost is bounded by what is RUNNING rather than by the
    campaign's history — a long campaign accumulates thousands of terminal runs.
    """
    get_client, calls = _client([[]])
    monkeypatch.setattr(mod, "get_client", get_client)
    await mod.zone_years_claimed_by_live_runs()
    asked = calls[0]["filter"].state.type.any_
    assert set(asked) == set(mod._LIVE_STATES)


@pytest.mark.asyncio
async def test_the_query_pages_past_the_first_page(monkeypatch):
    """A full page is never the last one. A live campaign runs more than one page of
    ingests, and a claim beyond the first page would read as no claim at all.
    """
    full = [_run(zone="33N", year=2021)] * mod._PAGE
    get_client, calls = _client([full, [_run(zone="59S", year=2019)]])
    monkeypatch.setattr(mod, "get_client", get_client)
    assert await mod.zone_years_claimed_by_live_runs() == {("33N", 2021), ("59S", 2019)}
    assert [c["offset"] for c in calls] == [0, mod._PAGE]
