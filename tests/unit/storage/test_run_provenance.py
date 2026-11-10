"""A zone group's per-year ``runs`` record (``shard_writer.run_provenance``, the schema's one owner).

The optional summaries — radar coverage, optical skips, input coverage, build identity — are per
YEAR because each is a property of what that year's run acquired or ran, not of the terrain: one
year of a zone can be radar-free, or lose tiles, where another does not. Each lands as given, and
an absent one is recorded as nothing: an empty field would read as a measured zero.
"""

from __future__ import annotations

import pytest

from tessera_embeddings.storage import shard_writer
from tessera_embeddings.storage.shard_writer import run_provenance

_SUMMARIES = {
    "radar_coverage": ({"s1_free_pct": 100.0}, {"s1_free_pct": 0.0}),
    "optical_skips": ({"tiles_skipped": 1, "labels": ["chunk_0_0"]}, {"tiles_skipped": 0, "labels": []}),
    "input_coverage": ({"window_months": 12, "relaxed": False}, {"window_months": 12, "relaxed": True}),
    "code": ({"version": "0.1.0", "commit": "abc123"}, {"version": "0.2.0", "commit": "def456"}),
}


@pytest.mark.parametrize("field", _SUMMARIES)
def test_each_year_keeps_its_own_summary(field: str) -> None:
    first, second = _SUMMARIES[field]
    runs = run_provenance(None, 2021, "a", **{field: first})
    runs = run_provenance(runs, 2022, "b", **{field: second})
    assert (runs["2021"][field], runs["2022"][field]) == (first, second)
    assert runs["2021"]["run_id"] == "a" and "assembled_at" in runs["2021"]


@pytest.mark.parametrize("field", _SUMMARIES)
def test_an_absent_summary_adds_no_key(field: str, monkeypatch) -> None:
    """``code`` is resolved from the process when not passed, so its absence is the process's."""
    monkeypatch.setattr(shard_writer, "code_identity", lambda: None)
    assert field not in run_provenance(None, 2021, "a", **{field: None})["2021"]


def test_an_empty_year_records_the_flag_instead_of_the_skip_labels() -> None:
    """``empty`` already states the whole live footprint is fill, which is why the label list
    needs no size cap: the one case where it could span a zone is the case it is dropped in.
    """
    record = run_provenance(
        None, 2021, "a", empty=True, optical_skips={"tiles_skipped": 620, "tiles_live": 620, "labels": ["chunk_0_0"]}
    )["2021"]
    assert record["empty"] is True
    assert "optical_skips" not in record


def test_re_recording_a_year_replaces_its_entry() -> None:
    """The campaign re-dispatches cells, so an appending merge would grow the label list without
    bound across retries of one cell.
    """
    summary = {"tiles_skipped": 1, "tiles_live": 4, "labels": ["chunk_0_1"]}
    runs = run_provenance(None, 2021, "a", optical_skips=summary)
    runs = run_provenance(runs, 2021, "b", optical_skips=summary)
    assert list(runs) == ["2021"]
    assert runs["2021"]["optical_skips"]["labels"] == ["chunk_0_1"]
    assert runs["2021"]["run_id"] == "b"


def test_a_retry_of_the_same_run_leaves_no_history_and_another_run_leaves_a_compact_one() -> None:
    """A cell is retried under its own run id routinely, so only a different run may grow the history,
    and by its identity alone: the replaced run's labels would grow it by a land mask per refill.
    """
    summary = {"tiles_skipped": 1, "tiles_live": 4, "labels": ["chunk_0_1"]}
    runs = run_provenance(None, 2021, "a", optical_skips=summary, commit={"snapshot": "s", "committed_at": "t"})
    runs = run_provenance(runs, 2021, "a", optical_skips=summary)
    assert "previous" not in runs["2021"]
    runs = run_provenance(runs, 2021, "b", optical_skips=summary)
    (replaced,) = runs["2021"]["previous"]
    assert set(replaced) == {"run_id", "assembled_at", "commit"} and replaced["run_id"] == "a"
