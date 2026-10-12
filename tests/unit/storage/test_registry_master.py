"""The registry master: the per-tile merge, provenance, the audit, and the in-place cell update.

The merge is the only place a published measurement could be overwritten by a re-derived one, or a
tile dropped by a run that never touched it, so most of these tests are about what it refuses to do.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from pyarrow.fs import LocalFileSystem

from tessera_embeddings.storage.registry import dataset_schema, registry_schema
from tessera_embeddings.storage.registry_master import (
    MEASURED_BY_FILL,
    MEASURED_BY_REBUILD,
    as_master_rows,
    audit,
    invariant_failures,
    master_schema,
    merge,
    update_cell,
    write_cell,
)

FILLED = "2026-08-01T00:00:00+00:00"
LATER = "2026-08-09T00:00:00+00:00"
REBUILT = "2026-09-21T00:00:00+00:00"


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
    """Rows as a fill part reads back from ``parts/``."""
    return _table(rows, dataset_schema())


def _fill(rows: list[dict]) -> pa.Table:
    return as_master_rows(_parts(rows), MEASURED_BY_FILL)


def _rebuilt(rows: list[dict], *, filled_at: str = FILLED) -> pa.Table:
    """Re-derived rows, against the fill stamped ``filled_at`` unless a row says otherwise."""
    rows = [row | {"filled_at": row["filled_at"] or filled_at} for row in rows]
    return as_master_rows(_table(rows, master_schema()), MEASURED_BY_REBUILD)


def _by_tile(master: pa.Table) -> dict[str, dict]:
    return {row["tile"]: row for row in master.to_pylist()}


class TestPrecedence:
    """Which row wins when a tile-year appears more than once."""

    def test_a_fill_measurement_beats_a_rebuild_of_the_same_tile(self) -> None:
        """A rebuild re-derives what a fill measured; where the fill measured, its number stands.

        The rebuild is the LATER row, so a naive latest-wins would take it. That would make the
        registry's numbers depend on when it was last repaired.
        """
        fill = _fill([_row("chunk_0_0", measured=True, stamp=FILLED, refused_thin_px=6)])
        rebuilt = _rebuilt([_row("chunk_0_0", measured=True, stamp=REBUILT, refused_thin_px=999)])
        master = merge(fill, rebuilt)
        assert master.column("refused_thin_px").to_pylist() == [6]
        assert master.column("measured_by").to_pylist() == [MEASURED_BY_FILL]

    def test_a_rebuild_fills_a_row_the_fill_left_null(self) -> None:
        fill = _fill([_row("chunk_0_0", measured=False, stamp=FILLED)])
        master = merge(fill, _rebuilt([_row("chunk_0_0", measured=True, stamp=REBUILT, refused_thin_px=7)]))
        assert master.column("refused_thin_px").to_pylist() == [7]
        assert master.column("measured_by").to_pylist() == [MEASURED_BY_REBUILD]

    @pytest.mark.parametrize(
        ("first", "second", "kept"),
        [(None, 123, 123), (123, None, 123)],
        ids=["the-radar-pass-supersedes-optical-only", "a-later-optical-only-pass-cannot-undo-it"],
    )
    def test_a_rebuild_that_measured_radar_wins(self, first: int | None, second: int | None, kept: int) -> None:
        fill = _fill([_row("chunk_0_0", measured=False, stamp=FILLED)])
        rebuilt = _rebuilt(
            [
                _row("chunk_0_0", measured=True, stamp=REBUILT, px_with_any_radar=first),
                _row("chunk_0_0", measured=True, stamp="2026-09-22T00:00:00+00:00", px_with_any_radar=second),
            ]
        )
        assert merge(fill, rebuilt).column("px_with_any_radar").to_pylist() == [kept]

    def test_a_refill_supersedes_the_older_run_and_its_rebuilds(self) -> None:
        """A newer run that left a tile null must not inherit the older run's numbers, nor a rebuild of it.

        The store holds the newer run's pixels for the tile, so the older run's measurement — and a
        re-derivation made against it — describe pixels that are no longer there.
        """
        fill = _fill(
            [
                _row("chunk_0_0", measured=True, stamp=FILLED, refused_thin_px=6, run_id="run-a"),
                _row("chunk_0_0", measured=False, stamp=LATER, run_id="run-b"),
            ]
        )
        stale = _rebuilt([_row("chunk_0_0", measured=True, stamp=REBUILT, refused_thin_px=7)])
        master = merge(fill, stale)
        assert master.column("run_id").to_pylist() == ["run-b"]
        assert master.column("measured_by").to_pylist() == [None]

    def test_a_partial_run_supersedes_only_the_tiles_it_wrote(self) -> None:
        """An infill writes some of a cell's tiles; every tile it did not touch keeps its run and rebuild.

        Superseding the whole cell would drop ``chunk_0_1``, and no check would notice if every check
        applied the same rule. The store keeps the older run's shard there, so the master must too.
        """
        held = merge(
            _fill([_row("chunk_0_0", measured=True, stamp=FILLED), _row("chunk_0_1", measured=False, stamp=FILLED)]),
            _rebuilt([_row("chunk_0_1", measured=True, stamp=REBUILT, refused_thin_px=7)]),
        )
        infill = _fill([_row("chunk_0_0", measured=True, stamp=LATER, refused_thin_px=1, run_id="infill")])
        tiles = _by_tile(merge(infill, held))
        assert (tiles["chunk_0_0"]["run_id"], tiles["chunk_0_0"]["refused_thin_px"]) == ("infill", 1)
        untouched = tiles["chunk_0_1"]
        assert (untouched["measured_by"], untouched["refused_thin_px"], untouched["filled_at"]) == (
            MEASURED_BY_REBUILD,
            7,
            FILLED,
        )

    @pytest.mark.parametrize("step", [1, -1], ids=["read-a-first", "read-b-first"])
    def test_two_runs_stamped_the_same_instant_resolve_by_run_id(self, step: int) -> None:
        """A whole-registry repair and a one-cell update must agree, whatever order rows are read in."""
        rows = [
            _row("chunk_0_0", measured=True, stamp=FILLED, refused_thin_px=n, run_id=run)
            for n, run in ((6, "run-a"), (8, "run-b"))
        ]
        assert merge(_fill(rows[::step])).column("run_id").to_pylist() == ["run-b"]

    @pytest.mark.parametrize("stamp", ["not-a-time", "2026-08-09T00:00:00"], ids=["malformed", "naive"])
    def test_a_run_whose_stamp_cannot_be_ordered_is_refused(self, stamp: str) -> None:
        """Text order would let an arbitrary string decide which run the store holds."""
        fill = _fill(
            [
                _row("chunk_0_0", measured=True, stamp=FILLED, run_id="run-a"),
                _row("chunk_0_0", measured=True, stamp=stamp, run_id="run-b"),
            ]
        )
        with pytest.raises(ValueError, match="cannot be ordered"):
            merge(fill)

    def test_a_tile_nothing_has_measured_stays_in_the_master_as_a_null_row(self) -> None:
        """Dropping it would turn "nobody measured this" into "this ground does not exist"."""
        master = merge(_fill([_row("chunk_0_0", measured=False, stamp=FILLED)]))
        assert master.num_rows == 1
        assert master.column("measured_by").to_pylist() == [None]

    def test_tiles_from_different_cells_do_not_collide(self) -> None:
        """The key is (zone, year, tile) — a tile label repeats in every zone and every year."""
        fill = _fill(
            [
                _row("chunk_0_0", zone="33N", year=2017, measured=True, stamp=FILLED),
                _row("chunk_0_0", zone="33N", year=2018, measured=True, stamp=FILLED),
                _row("chunk_0_0", zone="16S", year=2017, measured=True, stamp=FILLED),
            ]
        )
        assert merge(fill).num_rows == 3


class TestProvenance:
    """What ``measured_by`` and ``filled_at`` say, and what they refuse to say."""

    def test_measured_by_names_who_measured_not_which_pass_wrote_the_row(self) -> None:
        """Stamping an unmeasured fill row ``"fill"`` would tell a reader a measurement was taken."""
        assert merge(_fill([_row("chunk_0_0", measured=False, stamp=FILLED)])).column("measured_by").to_pylist() == [
            None
        ]

    def test_filled_at_names_the_run_on_fill_and_rebuilt_rows_alike(self) -> None:
        fill = _fill([_row("chunk_0_0", measured=False, stamp=FILLED), _row("chunk_0_1", measured=True, stamp=FILLED)])
        tiles = _by_tile(merge(fill, _rebuilt([_row("chunk_0_0", measured=True, stamp=REBUILT)])))
        assert (tiles["chunk_0_0"]["filled_at"], tiles["chunk_0_0"]["assembled_at"]) == (FILLED, REBUILT)
        assert tiles["chunk_0_1"]["filled_at"] == FILLED

    def test_the_master_schema_is_the_registrys_plus_provenance(self) -> None:
        names = [field.name for field in master_schema()]
        assert names[: len(dataset_schema())] == [field.name for field in dataset_schema()]
        assert names[-2:] == ["measured_by", "filled_at"]


class TestInvariants:
    """The checks that stand between a merge and a silently wrong master."""

    def test_a_clean_master_reports_nothing(self) -> None:
        assert invariant_failures(merge(_fill([_row("chunk_0_0", measured=True, stamp=FILLED)])), expected_rows=1) == []

    @pytest.mark.parametrize(
        ("rows", "expected_rows", "message"),
        [
            ([("chunk_0_0", True, FILLED, "fill"), ("chunk_0_0", True, LATER, "fill")], None, "duplicate"),
            ([("chunk_0_0", True, FILLED, "fill")], 2, "expected 2"),
            ([("chunk_0_0", False, FILLED, "fill")], None, "disagree"),
            ([("chunk_0_0", True, FILLED, "guessed")], None, "name no pass"),
        ],
        ids=["a-duplicate-tile-year", "a-lost-row", "a-label-the-numbers-contradict", "a-label-naming-no-pass"],
    )
    def test_a_broken_master_is_caught(self, rows: list[tuple], expected_rows: int | None, message: str) -> None:
        master = _table(
            [_row(tile, measured=measured, stamp=stamp, measured_by=tag) for tile, measured, stamp, tag in rows],
            master_schema(),
        )
        assert any(message in failure for failure in invariant_failures(master, expected_rows=expected_rows))


#: One cell's parts for the audit: a tile the fill measured, and one it left null.
_AUDITED = [
    _row("chunk_0_0", measured=True, stamp=FILLED, refused_thin_px=6),
    _row("chunk_0_1", measured=False, stamp=FILLED),
]


def _sound() -> pa.Table:
    """The master those parts should produce, with the null tile re-derived."""
    return merge(_fill(_AUDITED), _rebuilt([_row("chunk_0_1", measured=True, stamp=REBUILT)]))


def _altered(master: pa.Table) -> pa.Table:
    index = master.schema.get_field_index("refused_thin_px")
    return master.set_column(index, "refused_thin_px", pa.array([999] * master.num_rows, pa.int64()))


class TestAudit:
    """The master checked against the parts it should describe, as ``verify`` and the census run it."""

    def test_a_sound_master_reports_nothing(self) -> None:
        assert audit(_sound(), _parts(_AUDITED)) == {}

    @pytest.mark.parametrize(
        ("damage", "problem"),
        [
            (lambda m: m.filter(pc.equal(m.column("tile"), "chunk_0_0")), "tile(s) missing"),
            (lambda m: merge(m, _fill([_row("chunk_9_9", measured=True, stamp=FILLED)])), "no part names"),
            (_altered, "fill measurement(s) altered"),
            (lambda _: merge(_fill(_AUDITED)), "unmeasured"),
        ],
        ids=["a-dropped-tile", "a-tile-no-part-names", "an-altered-fill-measurement", "an-unmeasured-row"],
    )
    def test_damage_is_reported_against_its_cell(self, damage: Callable[[pa.Table], pa.Table], problem: str) -> None:
        found = audit(damage(_sound()), _parts(_AUDITED))
        assert list(found) == ["33N/2017"]
        assert any(problem in entry for entry in found["33N/2017"])

    def test_a_row_from_a_superseded_run_is_reported(self) -> None:
        """A fill whose master update failed leaves the cell describing the run before it."""
        refill = [*_AUDITED, _row("chunk_0_0", measured=True, stamp=LATER, run_id="run-b")]
        assert any("latest run" in entry for entry in audit(_sound(), _parts(refill))["33N/2017"])


def _publish(root: Path, run: str, rows: list[dict]) -> None:
    """One part of cell 33N/2017, written the way the fill writes its own."""
    cell = root / "parts" / "zone=33N" / "year=2017"
    cell.mkdir(parents=True, exist_ok=True)
    pq.write_table(_table(rows, registry_schema()), cell / f"{run}.parquet")


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    """One cell after a fill of two tiles, whose null tile was then re-derived into the master."""
    fill = [_row("chunk_0_0", measured=True, stamp=FILLED), _row("chunk_0_1", measured=False, stamp=FILLED)]
    _publish(tmp_path, "run-a", fill)
    rebuilt = _rebuilt([_row("chunk_0_1", measured=True, stamp=REBUILT, run_id="rebuild-1")])
    write_cell(LocalFileSystem(), str(tmp_path), "33N", 2017, merge(_fill(fill), rebuilt))
    return tmp_path


class TestUpdateCell:
    """The update every fill runs: the cell's parts merged into its existing master file, in place."""

    MASTER = Path("master/zone=33N/year=2017/part-0.parquet")

    def test_an_infill_replaces_its_tiles_and_keeps_every_other_row(self, registry: Path) -> None:
        _publish(registry, "infill", [_row("chunk_0_0", measured=True, stamp=LATER, run_id="infill")])
        assert update_cell(LocalFileSystem(), str(registry), "33N", 2017) == []
        master = pq.read_table(registry / self.MASTER)
        assert master.column("tile").to_pylist() == ["chunk_0_0", "chunk_0_1"]
        assert master.column("run_id").to_pylist() == ["infill", "rebuild-1"], "the re-derived row survives"
        assert pq.read_schema(registry / "master" / "_common_metadata").names == master_schema().names

    def test_running_it_again_writes_the_same_file(self, registry: Path) -> None:
        """A fill retried after its update, and a repair after a fill, must both leave the cell as it was."""
        update_cell(LocalFileSystem(), str(registry), "33N", 2017)
        first = (registry / self.MASTER).read_bytes()
        update_cell(LocalFileSystem(), str(registry), "33N", 2017)
        assert (registry / self.MASTER).read_bytes() == first

    def test_a_disagreement_with_the_store_is_reported_and_the_master_still_written(self, registry: Path) -> None:
        """No run removes a tile, so a shard cleared by hand is caught here or nowhere."""
        findings = update_cell(LocalFileSystem(), str(registry), "33N", 2017, live={(0, 1), (5, 5)})
        assert len(findings) == 2
        assert "chunk_0_0" in findings[0] and "chunk_5_5" in findings[1]
        assert (registry / self.MASTER).exists()

    def test_a_cell_with_no_part_raises_and_writes_nothing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no registry part"):
            update_cell(LocalFileSystem(), str(tmp_path), "33N", 2017)
        assert not (tmp_path / "master").exists()
