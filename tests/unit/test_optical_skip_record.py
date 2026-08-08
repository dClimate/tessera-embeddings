"""Recording which live tiles a year published as fill because no pixel was valid.

A skipped tile — every pixel failed the validity filter, so nothing was staged — is
written as the seeded fill value and the year is then marked complete. Ocean is fill
too, so without a record a consumer of a completed zone-year cannot tell "no valid
optical data" from "not land". The information exists at fill time and was discarded
at publication; these tests pin the record that keeps it.

Three properties matter more than the arithmetic. It is recorded **per year**, on the
same argument that already places ``radar_coverage`` there: a skip is a property of what
the year's acquisitions yielded, not of the terrain, so a zone-level figure would be
wrong for at least one year. It carries the **labels** as well as the count, because
only the labels say WHERE — a count alone still leaves the area unmaskable and
indistinguishable from ocean. And it is built from the **staging prefix's** resolution
of the live set rather than from one leg's own work, because skip markers outlive the
leg that wrote them.
"""

from __future__ import annotations

from tessera_embeddings.inference.assembly import summarise_optical_skips
from tessera_embeddings.storage.shard_writer import run_provenance


class TestSummarisingAYear:
    """What the summary states, from the two halves the staged scan resolves."""

    def test_the_count_and_the_labels_are_both_recorded(self) -> None:
        """A count answers how much was lost; only the labels answer where, and without
        where a consumer can neither mask the area nor tell it from ocean.
        """
        summary = summarise_optical_skips(staged=["chunk_0_0"], skipped=["chunk_1_0", "chunk_0_1"])
        assert summary["tiles_skipped"] == 2
        assert summary["labels"] == ["chunk_0_1", "chunk_1_0"]

    def test_the_live_total_the_count_is_against_is_recorded_with_it(self) -> None:
        """Self-describing: a reader interprets the count without fetching the land mask."""
        summary = summarise_optical_skips(staged=["chunk_0_0", "chunk_0_1"], skipped=["chunk_1_0"])
        assert summary["tiles_skipped"] == 1
        assert summary["tiles_live"] == 3

    def test_no_skips_summarises_as_a_zero_not_as_nothing(self) -> None:
        """A resolved live set with no skips is a positive fact — every live tile staged
        data — and is distinct from a caller that resolved no live set at all.
        """
        summary = summarise_optical_skips(staged=["chunk_0_0"], skipped=[])
        assert summary["tiles_skipped"] == 0
        assert summary["tiles_live"] == 1
        assert summary["labels"] == []

    def test_the_labels_are_sorted_so_two_runs_of_one_cell_compare(self) -> None:
        """Listing order is backend-dependent; a stable order is what makes the field
        diffable across two fills of the same cell.
        """
        assert summarise_optical_skips(staged=[], skipped=["chunk_1_0", "chunk_0_1", "chunk_0_0"])["labels"] == [
            "chunk_0_0",
            "chunk_0_1",
            "chunk_1_0",
        ]


class TestWhereItIsRecorded:
    """The provenance entry is per year, and the summary rides on it."""

    def test_the_summary_lands_on_the_year_s_entry(self) -> None:
        runs = run_provenance(None, 2021, "abc123", optical_skips={"tiles_skipped": 17, "labels": ["chunk_0_1"]})
        assert runs["2021"]["optical_skips"]["tiles_skipped"] == 17
        assert runs["2021"]["optical_skips"]["labels"] == ["chunk_0_1"]

    def test_each_year_keeps_its_own_summary(self) -> None:
        """The property this exists for: one year loses tiles where another does not."""
        runs = run_provenance(None, 2021, "a", optical_skips={"tiles_skipped": 1, "labels": ["chunk_0_0"]})
        runs = run_provenance(runs, 2022, "b", optical_skips={"tiles_skipped": 0, "labels": []})
        assert runs["2021"]["optical_skips"]["labels"] == ["chunk_0_0"]
        assert runs["2022"]["optical_skips"]["labels"] == []

    def test_no_summary_adds_no_key(self) -> None:
        """A caller that resolved no live set must not appear to have measured a zero."""
        assert "optical_skips" not in run_provenance(None, 2021, "abc123")
        assert "optical_skips" not in run_provenance(None, 2021, "abc123", optical_skips=None)

    def test_it_rides_alongside_the_radar_summary_not_instead_of_it(self) -> None:
        """Two independent coverage facts about one year; recording one must not cost the
        other, and neither may displace the record's identity fields.
        """
        record = run_provenance(
            None,
            2021,
            "abc123",
            radar_coverage={"s1_free_pct": 12.5},
            optical_skips={"tiles_skipped": 3, "labels": []},
        )["2021"]
        assert record["radar_coverage"]["s1_free_pct"] == 12.5
        assert record["optical_skips"]["tiles_skipped"] == 3
        assert record["run_id"] == "abc123"
        assert "assembled_at" in record

    def test_an_empty_year_records_the_flag_instead_of_the_labels(self) -> None:
        """The all-skipped case is already stated by ``empty``, which is why the label
        list needs no size cap: the one situation where it could span a whole zone is
        the situation this branch suppresses it in. Restating it would reproduce the
        land mask at zone size on a group attribute.
        """
        record = run_provenance(
            None,
            2021,
            "abc123",
            empty=True,
            optical_skips={"tiles_skipped": 620, "tiles_live": 620, "labels": ["chunk_0_0"]},
        )["2021"]
        assert record["empty"] is True
        assert "optical_skips" not in record


class TestIdempotency:
    """Re-recording a year replaces its entry; it never appends or accumulates."""

    def test_re_recording_the_same_year_leaves_one_entry(self) -> None:
        summary = {"tiles_skipped": 2, "tiles_live": 5, "labels": ["chunk_0_1", "chunk_1_0"]}
        runs = run_provenance(None, 2021, "abc123", optical_skips=summary)
        runs = run_provenance(runs, 2021, "abc123", optical_skips=summary)
        assert list(runs) == ["2021"]
        assert runs["2021"]["optical_skips"] == summary

    def test_re_recording_does_not_accumulate_labels(self) -> None:
        """The label list is the field's only variable-length content, so an appending
        merge would show up here first — and would grow without bound across retries of
        a cell, which the campaign performs routinely.
        """
        summary = {"tiles_skipped": 1, "tiles_live": 4, "labels": ["chunk_0_1"]}
        runs = run_provenance(None, 2021, "a", optical_skips=summary)
        runs = run_provenance(runs, 2021, "b", optical_skips=summary)
        assert runs["2021"]["optical_skips"]["labels"] == ["chunk_0_1"]
        assert runs["2021"]["run_id"] == "b"
