"""The compacted registry master: precedence, provenance, and the invariants that guard both.

The compaction is the only place a published measurement could be overwritten by a re-derived one,
so most of these tests are about what it refuses to do.
"""

from __future__ import annotations

import pyarrow as pa

from tessera_embeddings.storage.registry import dataset_schema
from tessera_embeddings.storage.registry_master import (
    MEASURED_BY_FILL,
    MEASURED_BY_REBUILD,
    fill_rows_unchanged,
    invariant_failures,
    master_schema,
    merge,
)
from tessera_embeddings.storage.registry_rebuild import rebuild_schema


def _row(tile: str, *, zone: str = "33N", year: int = 2017, measured: bool, stamp: str, **extra: object) -> dict:
    row: dict = {field.name: None for field in master_schema()}
    row |= {
        "tile": tile,
        "zone": zone,
        "year": year,
        "run_id": extra.pop("run_id", "run-a"),
        "assembled_at": stamp,
        "embedded": True,
        "optical_min_obs": 15,
    }
    if measured:
        row |= {
            "chunk_px": 4194304,
            "refused_px": 10,
            "refused_no_optical_px": 4,
            "refused_thin_px": 6,
            "refused_no_radar_px": 0,
            "px_with_any_optical": 4194300,
            "obs_max": 50,
            "median_obs_where_any": 40.0,
            "radar_rule_enforced": False,
        }
    row |= extra
    return row


def _table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    trimmed = [{field.name: row.get(field.name) for field in schema} for row in rows]
    return pa.Table.from_pylist(trimmed, schema=schema)


def _parts(rows: list[dict]) -> pa.Table:
    return _table(rows, dataset_schema())


def _rebuilt(rows: list[dict]) -> pa.Table:
    schema = pa.schema([*rebuild_schema(), pa.field("zone", pa.string()), pa.field("year", pa.int32())])
    return _table(rows, schema)


class TestPrecedence:
    """Which row wins when a tile-year appears more than once."""

    def test_a_fill_measurement_beats_a_rebuild_of_the_same_tile(self) -> None:
        """A rebuild re-derives what a fill measured; where the fill measured, its number stands.

        The rebuild is the LATER row, so a naive latest-wins would take it. That would make the
        registry's numbers depend on when it was last compacted.
        """
        parts = _parts([_row("chunk_0_0", measured=True, stamp="2026-08-01T00:00:00+00:00", refused_thin_px=6)])
        rebuilt = _rebuilt([_row("chunk_0_0", measured=True, stamp="2026-09-21T00:00:00+00:00", refused_thin_px=999)])
        master = merge(parts, rebuilt)
        assert master.num_rows == 1
        assert master.column("refused_thin_px").to_pylist() == [6]
        assert master.column("measured_by").to_pylist() == [MEASURED_BY_FILL]

    def test_a_rebuild_fills_a_row_the_fill_left_null(self) -> None:
        parts = _parts([_row("chunk_0_0", measured=False, stamp="2026-08-01T00:00:00+00:00")])
        rebuilt = _rebuilt([_row("chunk_0_0", measured=True, stamp="2026-09-21T00:00:00+00:00", refused_thin_px=7)])
        master = merge(parts, rebuilt)
        assert master.column("refused_thin_px").to_pylist() == [7]
        assert master.column("measured_by").to_pylist() == [MEASURED_BY_REBUILD]

    def test_a_later_rebuild_supersedes_an_earlier_one(self) -> None:
        """This is how the radar second pass replaces the cheap optical-only first pass."""
        parts = _parts([_row("chunk_0_0", measured=False, stamp="2026-08-01T00:00:00+00:00")])
        rebuilt = _rebuilt(
            [
                _row("chunk_0_0", measured=True, stamp="2026-09-21T00:00:00+00:00", px_with_any_radar=None),
                _row("chunk_0_0", measured=True, stamp="2026-09-22T00:00:00+00:00", px_with_any_radar=123),
            ]
        )
        master = merge(parts, rebuilt)
        assert master.num_rows == 1
        assert master.column("px_with_any_radar").to_pylist() == [123]

    def test_a_cell_filled_twice_resolves_to_its_later_fill(self) -> None:
        """Parts are keyed by run so a refill ADDS a part; dedup was always the compaction's job."""
        parts = _parts(
            [
                _row("chunk_0_0", measured=True, stamp="2026-08-01T00:00:00+00:00", refused_thin_px=6, run_id="run-a"),
                _row("chunk_0_0", measured=True, stamp="2026-08-09T00:00:00+00:00", refused_thin_px=8, run_id="run-b"),
            ]
        )
        master = merge(parts)
        assert master.num_rows == 1
        assert master.column("refused_thin_px").to_pylist() == [8]
        assert master.column("run_id").to_pylist() == ["run-b"]

    def test_a_tile_nothing_has_measured_stays_in_the_master_as_a_null_row(self) -> None:
        """Dropping it would turn "nobody measured this" into "this ground does not exist"."""
        master = merge(_parts([_row("chunk_0_0", measured=False, stamp="2026-08-01T00:00:00+00:00")]))
        assert master.num_rows == 1
        assert master.column("measured_by").to_pylist() == [None]
        assert master.column("chunk_px").to_pylist() == [None]

    def test_tiles_from_different_cells_do_not_collide(self) -> None:
        """The key is (zone, year, tile) — a tile label repeats in every zone and every year."""
        parts = _parts(
            [
                _row("chunk_0_0", zone="33N", year=2017, measured=True, stamp="2026-08-01T00:00:00+00:00"),
                _row("chunk_0_0", zone="33N", year=2018, measured=True, stamp="2026-08-01T00:00:00+00:00"),
                _row("chunk_0_0", zone="16S", year=2017, measured=True, stamp="2026-08-01T00:00:00+00:00"),
            ]
        )
        assert merge(parts).num_rows == 3


class TestProvenance:
    """What ``measured_by`` and ``filled_at`` say, and what they refuse to say."""

    def test_measured_by_is_null_where_nothing_was_measured(self) -> None:
        """It names who measured the numbers, not which pass wrote the row.

        Stamping an unmeasured fill row ``"fill"`` would make a reader filtering on it believe a
        measurement was taken.
        """
        master = merge(_parts([_row("chunk_0_0", measured=False, stamp="2026-08-01T00:00:00+00:00")]))
        assert master.column("measured_by").to_pylist() == [None]

    def test_the_fills_own_timestamp_survives_onto_a_rebuilt_row(self) -> None:
        parts = _parts([_row("chunk_0_0", measured=False, stamp="2026-08-01T00:00:00+00:00")])
        rebuilt = _rebuilt(
            [_row("chunk_0_0", measured=True, stamp="2026-09-21T00:00:00+00:00", filled_at="2026-08-01T00:00:00+00:00")]
        )
        master = merge(parts, rebuilt)
        assert master.column("filled_at").to_pylist() == ["2026-08-01T00:00:00+00:00"]
        assert master.column("assembled_at").to_pylist() == ["2026-09-21T00:00:00+00:00"]

    def test_the_master_schema_is_the_registrys_plus_provenance(self) -> None:
        names = [field.name for field in master_schema()]
        assert names[: len(dataset_schema())] == [field.name for field in dataset_schema()]
        assert names[-2:] == ["measured_by", "filled_at"]


class TestInvariants:
    """The checks that stand between a compaction and a silently wrong master."""

    def _master(self, rows: list[dict]) -> pa.Table:
        return _table(rows, master_schema())

    def test_a_clean_master_reports_nothing(self) -> None:
        master = merge(_parts([_row("chunk_0_0", measured=True, stamp="2026-08-01T00:00:00+00:00")]))
        assert invariant_failures(master, expected_rows=1) == []

    def test_a_duplicate_tile_year_is_caught(self) -> None:
        """One tile-year twice reads as coverage counted twice, and nothing else would notice."""
        duplicated = self._master(
            [
                _row("chunk_0_0", measured=True, stamp="a", measured_by=MEASURED_BY_FILL),
                _row("chunk_0_0", measured=True, stamp="b", measured_by=MEASURED_BY_FILL),
            ]
        )
        assert any("duplicate" in failure for failure in invariant_failures(duplicated))

    def test_a_lost_row_is_caught(self) -> None:
        master = merge(_parts([_row("chunk_0_0", measured=True, stamp="a")]))
        assert any("expected 2" in failure for failure in invariant_failures(master, expected_rows=2))

    def test_a_label_that_disagrees_with_the_measurements_is_caught(self) -> None:
        mislabelled = self._master([_row("chunk_0_0", measured=False, stamp="a", measured_by=MEASURED_BY_FILL)])
        assert any("disagree" in failure for failure in invariant_failures(mislabelled))

    def test_a_measured_by_value_naming_no_pass_is_caught(self) -> None:
        odd = self._master([_row("chunk_0_0", measured=True, stamp="a", measured_by="guessed")])
        assert any("name no pass" in failure for failure in invariant_failures(odd))


class TestFillRowsUnchanged:
    """The compaction must never alter a measurement the fill published."""

    COLUMNS = ("refused_thin_px", "refused_px")

    def test_an_untouched_fill_row_reports_nothing(self) -> None:
        parts = _parts([_row("chunk_0_0", measured=True, stamp="a", refused_thin_px=6)])
        assert fill_rows_unchanged(merge(parts), parts, self.COLUMNS) == []

    def test_an_altered_fill_row_is_reported(self) -> None:
        """The compaction must never change a published measurement, and "never" is worth testing."""
        parts = _parts([_row("chunk_0_0", measured=True, stamp="a", refused_thin_px=6)])
        tampered = _table(
            [_row("chunk_0_0", measured=True, stamp="a", refused_thin_px=999, measured_by=MEASURED_BY_FILL)],
            master_schema(),
        )
        differences = fill_rows_unchanged(tampered, parts, self.COLUMNS)
        assert any("refused_thin_px" in difference for difference in differences)

    def test_a_rebuilt_row_is_not_compared_against_the_fill(self) -> None:
        """It is a re-derivation of a row the fill left null; there is nothing to differ from."""
        parts = _parts([_row("chunk_0_0", measured=False, stamp="a")])
        rebuilt = _rebuilt([_row("chunk_0_0", measured=True, stamp="b", refused_thin_px=7)])
        assert fill_rows_unchanged(merge(parts, rebuilt), parts, self.COLUMNS) == []
