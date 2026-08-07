"""Recording how much of a year's embedded area had no radar, or little of it.

Exact per-pixel counts already live in the store's ``s1_asc_obs_count`` and
``s1_desc_obs_count`` arrays, so "where" is available at full precision. What was missing
was the ability to ASK — answering it meant reducing a zone-sized grid. These tests pin the
summary that makes it a metadata read.

Two properties matter more than the arithmetic. It is recorded **per year**, because radar
coverage is a property of what was acquired rather than of the terrain: one year of a zone
can be radar-free where another is not, so a zone-level figure would be wrong for at least
one of them. And the percentages are of **embedded** area, not of the zone, because a zone
is mostly ocean and mostly unembedded — a fraction of the grid would be dominated by area
no radar was ever expected over.
"""

from __future__ import annotations

from tessera_embeddings.config.inference import RADAR_LIGHT_MAX_OBS
from tessera_embeddings.inference.assembly import summarise_radar_coverage
from tessera_embeddings.storage.shard_writer import run_provenance


def _chunk(valid: int, free: int, light: int = 0, status: str = "success") -> dict:
    return {
        "chunk": "chunk_0_0",
        "status": status,
        "valid_pixels": valid,
        "s1_free_pixels": free,
        "s1_light_pixels": light,
    }


class TestSummarisingAYear:
    """Reducing per-chunk counts the actors already reported."""

    def test_a_wholly_radar_free_year_reads_as_one_hundred_percent(self) -> None:
        summary = summarise_radar_coverage([_chunk(100, 100)])
        assert summary is not None
        assert summary["s1_free_pct"] == 100.0
        assert summary["s1_free_px"] == 100

    def test_a_fully_covered_year_reads_as_zero(self) -> None:
        summary = summarise_radar_coverage([_chunk(100, 0)])
        assert summary is not None
        assert summary["s1_free_pct"] == 0.0

    def test_the_percentage_is_of_embedded_area_not_of_the_zone(self) -> None:
        """Half the EMBEDDED pixels, not half the grid — most of a zone is never embedded."""
        summary = summarise_radar_coverage([_chunk(1000, 0), _chunk(1000, 1000)])
        assert summary is not None
        assert summary["embedded_px"] == 2000
        assert summary["s1_free_pct"] == 50.0

    def test_radar_light_is_reported_separately_from_radar_free(self) -> None:
        """A pixel the radar barely saw is not the same as one it never saw, and the
        embedding exposes neither — both simply exist.
        """
        summary = summarise_radar_coverage([_chunk(1000, 100, light=250)])
        assert summary is not None
        assert summary["s1_free_pct"] == 10.0
        assert summary["s1_light_pct"] == 25.0

    def test_the_light_threshold_is_recorded_with_the_figure(self) -> None:
        """A percentage whose threshold is not stated cannot be compared across builds."""
        summary = summarise_radar_coverage([_chunk(10, 0, light=1)])
        assert summary is not None
        assert summary["s1_light_below_obs"] == RADAR_LIGHT_MAX_OBS

    def test_failed_chunks_are_excluded_from_the_denominator(self) -> None:
        """A failed chunk embedded nothing, so counting it would dilute the fraction."""
        summary = summarise_radar_coverage([_chunk(100, 100), _chunk(0, 0, status="failed")])
        assert summary is not None
        assert summary["embedded_px"] == 100
        assert summary["s1_free_pct"] == 100.0

    def test_a_run_that_reported_no_counts_records_no_summary_at_all(self) -> None:
        """An older build reported no counts. Recording zeros would be a WRONG answer —
        indistinguishable from a fully radar-covered year — so absence must stay absence.
        """
        assert summarise_radar_coverage([{"status": "success", "valid_pixels": 100}]) is None

    def test_an_empty_run_records_no_summary(self) -> None:
        assert summarise_radar_coverage([]) is None

    def test_a_year_that_embedded_nothing_records_no_summary(self) -> None:
        """Zero embedded pixels has no meaningful percentage; a division would be the bug."""
        assert summarise_radar_coverage([_chunk(0, 0)]) is None


class TestLocatingTheGapCoarsely:
    """Where, to the extent a summary can say it without storing a per-tile grid."""

    def test_wholly_radar_free_tiles_are_counted(self) -> None:
        """Distinguishes a CONCENTRATED absence — whole tiles, e.g. an ice margin — from a
        diffuse one, such as a swath edge clipping many tiles.
        """
        summary = summarise_radar_coverage([_chunk(100, 100), _chunk(100, 100), _chunk(100, 5)])
        assert summary is not None
        assert summary["tiles_fully_s1_free"] == 2
        assert summary["tiles_reporting"] == 3

    def test_a_partly_radar_free_tile_is_not_counted_as_wholly_free(self) -> None:
        summary = summarise_radar_coverage([_chunk(100, 99)])
        assert summary is not None
        assert summary["tiles_fully_s1_free"] == 0

    def test_a_tile_with_no_radar_free_pixels_is_not_counted(self) -> None:
        """Guards the zero-vs-zero case: 0 free of 0 embedded must not read as 'wholly free'."""
        summary = summarise_radar_coverage([_chunk(100, 0), _chunk(50, 0)])
        assert summary is not None
        assert summary["tiles_fully_s1_free"] == 0


class TestWhereItIsRecorded:
    """The provenance entry is per year, and the summary rides on it."""

    def test_the_summary_lands_on_the_year_s_entry(self) -> None:
        runs = run_provenance(None, 2021, "abc123", radar_coverage={"s1_free_pct": 100.0})
        assert runs["2021"]["radar_coverage"]["s1_free_pct"] == 100.0

    def test_each_year_keeps_its_own_summary(self) -> None:
        """The property this exists for: one year radar-free, another not, in one zone."""
        runs = run_provenance(None, 2021, "a", radar_coverage={"s1_free_pct": 100.0})
        runs = run_provenance(runs, 2022, "b", radar_coverage={"s1_free_pct": 0.0})
        assert runs["2021"]["radar_coverage"]["s1_free_pct"] == 100.0
        assert runs["2022"]["radar_coverage"]["s1_free_pct"] == 0.0

    def test_no_summary_adds_no_key(self) -> None:
        """An absent summary must not appear as an empty one, which would read as measured."""
        assert "radar_coverage" not in run_provenance(None, 2021, "abc123")
        assert "radar_coverage" not in run_provenance(None, 2021, "abc123", radar_coverage=None)

    def test_recording_a_summary_does_not_disturb_the_rest_of_the_record(self) -> None:
        record = run_provenance(None, 2021, "abc123", radar_coverage={"s1_free_pct": 1.0})["2021"]
        assert record["run_id"] == "abc123"
        assert "assembled_at" in record


def test_a_resume_records_no_summary_rather_than_one_for_the_part_it_redid():
    """A resumed tile is a synthetic success carrying no counters.

    Dropping it from both sides of the ratio still leaves a figure — over the tiles this
    run happened to redo, which is whatever the previous attempt failed to finish. That
    set has no relationship to the year's radar coverage, so the number would be wrong by
    an unknowable margin and stored as the year's with nothing saying so.
    """
    mixed = [
        {"status": "success", "valid_pixels": 100, "s1_free_pixels": 10, "s1_light_pixels": 20},
        {"status": "success", "valid_pixels": 100},  # resumed: staged by an earlier attempt
    ]
    assert summarise_radar_coverage(mixed) is None


def test_a_skipped_tile_does_not_suppress_the_summary():
    """A tile with no valid pixels reports no counts because it embedded nothing — that
    is a complete answer, not a missing one, and must not cost the year its summary.
    """
    results = [
        {"status": "success", "valid_pixels": 100, "s1_free_pixels": 10, "s1_light_pixels": 20},
        {"status": "skipped", "valid_pixels": 0},
    ]
    summary = summarise_radar_coverage(results)
    assert summary is not None
    assert summary["s1_free_pct"] == 10.0
