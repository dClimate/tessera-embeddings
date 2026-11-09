"""The tile registry's rows and Parquet parts (``storage.registry``).

A registry is ~1,000 parts read as one dataset, so a part that is valid alone and unmergeable with its
siblings is worse than a broken part: nothing notices until the whole thing is read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera_embeddings.storage.registry import part_uri, registry_rows, registry_schema, write_registry_part


def _record(label: str, *, thin: int = 0, eligible: int = 64) -> dict:
    """A wholly refused shard's skip record."""
    return {
        "label": label,
        "refused": {"no_optical": 0, "thin": thin, "no_radar": 0},
        "eligible_px": eligible,
        "s2_obs": {"px_with_any": 0, "max": 0, "mean_where_any": 0.0},
    }


def _write_part(root: str, cell: tuple[str, int, str], rows: list[dict], *, rule: int | None = None) -> str:
    """Write one ``(zone, year, run)`` part where the registry keeps it; *rule* is stamped in the key-value block."""
    zone, year, run = cell
    uri = part_uri(root, zone, year, run)
    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    # The lambda is not a leak: `write_registry_part` consumes it as `with open_output(uri) as f`.
    write_registry_part(
        uri,
        rows,
        open_output=lambda u: Path(u).open("wb"),  # noqa: SIM115
        zone=zone,
        year=year,
        extra_metadata=None if rule is None else {"optical_min_obs": str(rule)},
    )
    return uri


class TestTheDatasetIsActuallyReadable:
    """The two defects a red-team pass found, both of which every individual part survived.

    A registry is 1,008 parts read as one dataset, so a part that is valid alone and unmergeable with
    its siblings is worse than a broken part: nothing notices until the whole thing is read.
    """

    def test_a_cell_that_refused_nothing_still_merges_with_one_that_did(self, tmp_path: Path) -> None:
        """The schema must be DECLARED. Inferred, a cell with no refusals types every refusal column
        `null`, and concatenating it with a cell that did refuse something raises ArrowInvalid — and
        most cells refuse nothing, so the compaction would have broken on almost any pair.
        """
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        root = str(tmp_path / "reg")
        clean = _write_part(root, ("32S", 2021, "r1"), registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[]))
        dirty = _write_part(
            root,
            ("09S", 2021, "r2"),
            registry_rows(
                "r2", "t", embedded=[], refused=["chunk_1_1"], records={"chunk_1_1": _record("chunk_1_1", thin=64)}
            ),
        )

        merged = pa.concat_tables([pq.read_table(clean), pq.read_table(dirty)])
        assert merged.num_rows == 2
        assert merged.schema.field("refused_thin_px").type == pa.int64(), "typed even where all-null"

    def test_the_whole_dataset_reads_with_hive_partitioning(self, tmp_path: Path) -> None:
        """Carrying zone/year as columns as well as partition keys made this raise
        `ArrowTypeError: Field year has incompatible types: int64 vs int32` — the dataset was
        unreadable as a dataset while every part opened fine on its own.
        """
        ds = pytest.importorskip("pyarrow.dataset")
        root = str(tmp_path / "reg")
        _write_part(root, ("32S", 2021, "r1"), registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[]))
        _write_part(root, ("09S", 2022, "r2"), registry_rows("r2", "t", embedded=["chunk_1_1"], refused=[]))

        table = ds.dataset(f"{root}/parts", partitioning="hive").to_table()
        assert table.num_rows == 2
        assert set(table.column("zone").to_pylist()) == {"32S", "09S"}, "from the path, not a column"
        assert set(table.column("year").to_pylist()) == {2021, 2022}

    def test_an_update_names_itself_on_its_rows_and_a_fill_leaves_it_null(self, tmp_path: Path) -> None:
        """The master keeps each tile's newest row, so this is how a tile says which update its data came from."""
        pq = pytest.importorskip("pyarrow.parquet")
        root = str(tmp_path / "reg")
        fill = _write_part(root, ("32S", 2021, "fill"), registry_rows("fill", "t1", embedded=["chunk_0_0"], refused=[]))
        update = _write_part(
            root, ("32S", 2021, "upd"), registry_rows("upd", "t2", embedded=["chunk_0_0"], refused=[], update_name="u1")
        )
        assert pq.read_table(fill).column("update_name").to_pylist() == [None]
        assert pq.read_table(update).column("update_name").to_pylist() == ["u1"]

    def test_a_refill_adds_a_part_rather_than_replacing_one(self, tmp_path: Path) -> None:
        """Keyed by run so the registry holds both fills of a cell. A fixed name would silently
        replace the original, and dedup is the compaction's decision to make, not a write's.
        """
        ds = pytest.importorskip("pyarrow.dataset")
        root = str(tmp_path / "reg")
        _write_part(root, ("32S", 2021, "first"), registry_rows("first", "t1", embedded=["chunk_0_0"], refused=[]))
        _write_part(root, ("32S", 2021, "refill"), registry_rows("refill", "t2", embedded=["chunk_0_0"], refused=[]))

        table = ds.dataset(f"{root}/parts", partitioning="hive").to_table()
        assert table.num_rows == 2, "both fills are on the record"
        assert set(table.column("run_id").to_pylist()) == {"first", "refill"}
        assert set(table.column("assembled_at").to_pylist()) == {"t1", "t2"}, "and a clock to pick by"


class TestAPartlyRefusedTileIsNotFullyCovered:
    """The largest hole this registry had: a tile the depth gate partly refused was recorded as
    embedded with no refusals, which reads as covered ground when part of it is holes.

    The actor accumulates refusal reasons on the SUCCESS path too, so the number always existed; only
    the wholly-refused branch wrote it down. Those partial refusals are the bulk of what a revisit
    campaign would fill, so this is most of the infill work list rather than a refinement of it.
    """

    @staticmethod
    def _partial(thin: int = 40) -> dict:
        return {
            "refused": {"thin": thin, "no_optical": 0, "no_radar": 0},
            "eligible_px": 100,
            "chunk_px": 100,
            "s2_obs": {"px_with_any": 95, "max": 14, "median_where_any": 11.0},
            "px_with_any_radar": 100,
            "radar_rule_enforced": False,
        }

    def test_an_embedded_tile_reports_what_the_gate_removed(self) -> None:
        rows = registry_rows(
            "r1",
            "t",
            embedded=["chunk_0_0"],
            refused=[],
            embedded_records={"chunk_0_0": self._partial()},
            optical_min_obs=15,
        )
        row = rows[0]
        assert row["embedded"] is True, "it does hold embeddings"
        assert row["refused_px"] == 40, "and 40 of its 100 eligible pixels are not there"
        assert row["refused_thin_px"] == 40
        assert row["eligible_px"] == 100

    def test_zero_refusals_is_written_as_zero_not_null(self) -> None:
        """A measured zero and an unmeasured tile must be distinguishable: the first says the gate
        removed nothing, the second says nobody looked. Both used to publish as null.
        """
        rows = registry_rows(
            "r1",
            "t",
            embedded=["chunk_0_0", "chunk_0_1"],
            refused=[],
            embedded_records={"chunk_0_0": self._partial(thin=0)},
            optical_min_obs=15,
        )
        measured = next(r for r in rows if r["tile"] == "chunk_0_0")
        unmeasured = next(r for r in rows if r["tile"] == "chunk_0_1")
        assert measured["refused_px"] == 0
        assert unmeasured["refused_px"] is None

    def test_both_kinds_of_row_are_measured_in_the_same_terms(self) -> None:
        """Comparability is the point — an infill planner ranks a 60%-refused embedded tile against
        a wholly refused one — so the two record sources must fill the same columns identically.
        """
        rec = self._partial()
        rows = registry_rows(
            "r1",
            "t",
            embedded=["chunk_0_0"],
            refused=["chunk_1_1"],
            embedded_records={"chunk_0_0": rec},
            records={"chunk_1_1": rec},
            optical_min_obs=15,
        )
        a, b = (next(r for r in rows if r["tile"] == t) for t in ("chunk_0_0", "chunk_1_1"))
        shared = [c for c in a if c not in ("tile", "embedded")]
        assert {c: a[c] for c in shared} == {c: b[c] for c in shared}
        assert a["embedded"] is True and b["embedded"] is False

    def test_a_marker_wins_over_a_result_for_the_same_label(self) -> None:
        """A label in both sources refused everything at the end, so its marker is the later word."""
        rows = registry_rows(
            "r1",
            "t",
            embedded=[],
            refused=["chunk_0_0"],
            embedded_records={"chunk_0_0": self._partial(thin=1)},
            records={"chunk_0_0": self._partial(thin=99)},
            optical_min_obs=15,
        )
        assert rows[0]["refused_thin_px"] == 99

    def test_a_refused_tile_with_no_record_is_null_not_zero(self, tmp_path: Path) -> None:
        """A marker that could not be read leaves the reason unknown. Reporting zero refusals for a
        tile that is demonstrably empty would be a self-contradicting row.
        """
        rows = registry_rows("r1", "t", embedded=[], refused=["chunk_0_0"], records={})
        assert rows[0]["embedded"] is False
        assert rows[0]["refused_px"] is None
        assert rows[0]["obs_max"] is None


class TestTheDepthRuleTheRowsWereJudgedAgainst:
    """`obs_max` and `median_obs_where_any` are distances from a threshold, so a row without the
    threshold is unreadable: 14 is one scene short of a line at 15 and nowhere near one at 30.

    Cell-level policy, not a per-tile measurement, which is why it is stamped on EVERY row rather
    than pulled from a record — an all-refused cell and a fully-embedded one must both state it.
    """

    def test_every_row_carries_it_including_embedded_ones(self) -> None:
        """The trap this pins: the value is applied at TWO sites — the embedded loop and the refused
        loop — so setting it in one leaves half the rows unreadable, and the half that is fine is the
        half most cells are made of.
        """
        rows = registry_rows(
            "r1",
            "t",
            embedded=["chunk_0_0", "chunk_0_1"],
            refused=["chunk_1_0", "chunk_1_1"],
            records={"chunk_1_0": _record("chunk_1_0", thin=64)},
            optical_min_obs=15,
        )
        assert len(rows) == 4
        assert [r["optical_min_obs"] for r in rows] == [15, 15, 15, 15], "embedded AND refused, recorded AND not"

    def test_no_rule_stays_null_rather_than_becoming_zero(self) -> None:
        """A store that declares no depth rule is not a store with a rule of zero: the first refuses
        nothing by policy, the second is the value `optical_min_obs` validation rejects outright.
        """
        rows = registry_rows("r1", "t", embedded=["chunk_0_0"], refused=["chunk_1_1"], optical_min_obs=None)
        assert all(r["optical_min_obs"] is None for r in rows)

    def test_two_cells_filled_under_different_lines_stay_distinguishable(self, tmp_path: Path) -> None:
        """The reason this belongs in the rows and not only in a docstring: a store is write-once on
        its rule, but the registry outlives any one store and a compaction may span two. Reading the
        merged dataset must not average across two different definitions of "thin" silently.
        """
        ds = pytest.importorskip("pyarrow.dataset")
        root = str(tmp_path / "reg")
        for zone, rule in (("32S", 15), ("09S", 30)):
            rows = registry_rows("r", "t", embedded=["chunk_0_0"], refused=[], optical_min_obs=rule)
            _write_part(root, (zone, 2021, f"r-{zone}"), rows, rule=rule)

        table = ds.dataset(f"{root}/parts", partitioning="hive").to_table()
        assert table.num_rows == 2
        by_zone = dict(zip(table.column("zone").to_pylist(), table.column("optical_min_obs").to_pylist(), strict=True))
        assert by_zone == {"32S": 15, "09S": 30}

    def test_a_part_read_alone_still_states_the_rule(self, tmp_path: Path) -> None:
        """Also in the key-value block, for a reader that opens one file without the partitioning."""
        pq = pytest.importorskip("pyarrow.parquet")
        root = str(tmp_path / "reg")
        rows = registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[], optical_min_obs=15)
        uri = _write_part(root, ("32S", 2021, "r1"), rows, rule=15)
        md = {k.decode(): v.decode() for k, v in (pq.read_schema(uri).metadata or {}).items()}
        assert md["optical_min_obs"] == "15"
        assert md["zone"] == "32S" and md["year"] == "2021"


class TestPartsWrittenByDifferentBuilds:
    """A nine-year campaign crosses code versions, so parts will disagree about their columns.

    The failure is silent and order-dependent, which is what makes it worth a test rather than a
    note: `ds.dataset` infers its schema from the FIRST file in sorted path order, so an older part
    under `zone=01N` hides a column that a newer part under `zone=60S` carries — and hides it by
    omission, which reads as "never recorded" rather than as an error.
    """

    @staticmethod
    def _write_with_schema(root: str, zone: str, schema, rows: list[dict]) -> None:
        pq = pytest.importorskip("pyarrow.parquet")
        pa = pytest.importorskip("pyarrow")
        trimmed = [{k: v for k, v in r.items() if k in schema.names} for r in rows]
        uri = part_uri(root, zone, 2021, f"run-{zone}")
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(trimmed, schema=schema), uri, compression="zstd")

    def _two_builds(self, tmp_path: Path, older_zone: str, newer_zone: str) -> str:
        pa = pytest.importorskip("pyarrow")
        newer = registry_schema()
        older = pa.schema([f for f in newer if f.name != "median_obs_where_thin"])
        rec = {
            "refused": {"thin": 40, "no_optical": 0, "no_radar": 0},
            "eligible_px": 100,
            "chunk_px": 100,
            "s2_obs": {"px_with_any": 95, "max": 60, "median_where_any": 50.0, "median_where_thin": 11.0},
            "px_with_any_radar": 0,
            "radar_rule_enforced": False,
        }
        root = str(tmp_path / "reg")
        for zone, schema in ((older_zone, older), (newer_zone, newer)):
            self._write_with_schema(
                root,
                zone,
                schema,
                registry_rows(
                    "r",
                    "t",
                    embedded=["chunk_0_0"],
                    refused=[],
                    embedded_records={"chunk_0_0": rec},
                    optical_min_obs=15,
                ),
            )
        return root

    def test_stating_the_schema_recovers_it_and_nulls_the_older_part(self, tmp_path: Path) -> None:
        """The prescribed read: the column is present, and the run that never measured it says null
        rather than claiming a value.
        """
        ds = pytest.importorskip("pyarrow.dataset")
        from tessera_embeddings.storage.registry import dataset_schema

        root = self._two_builds(tmp_path, older_zone="01N", newer_zone="60S")
        table = ds.dataset(f"{root}/parts", schema=dataset_schema(), partitioning="hive").to_table()
        by_zone = dict(
            zip(table.column("zone").to_pylist(), table.column("median_obs_where_thin").to_pylist(), strict=True)
        )
        assert by_zone == {"01N": None, "60S": 11.0}


class TestRankingAnInfillByTheRightDepth:
    """`median_obs_where_any` is the wrong number to rank a revisit by, and 09S/2021 showed why.

    Observed 2026-08-19: tiles with a median depth of 50 against a line of 15, each still refusing
    several thousand pixels. The median is over the whole evaluated footprint, so on a partly refused
    tile it describes the pixels that PASSED — ranking by it puts a few marginal pixels in deep
    imagery above land that is uniformly thin.
    """

    @staticmethod
    def _rec(median_any: float, median_thin: float | None, thin: int) -> dict:
        return {
            "refused": {"thin": thin, "no_optical": 0, "no_radar": 0},
            "eligible_px": 4194304,
            "chunk_px": 4194304,
            "s2_obs": {
                "px_with_any": 4194304,
                "max": 60,
                "median_where_any": median_any,
                "median_where_thin": median_thin,
            },
            "px_with_any_radar": 0,
            "radar_rule_enforced": False,
        }

    def test_the_two_medians_describe_different_populations(self) -> None:
        rows = registry_rows(
            "r1",
            "t",
            embedded=["chunk_132_19"],
            refused=[],
            embedded_records={"chunk_132_19": self._rec(50.0, 11.0, 5504)},
            optical_min_obs=15,
        )
        row = rows[0]
        assert row["median_obs_where_any"] == 50.0, "the footprint, dominated by what passed"
        assert row["median_obs_where_thin"] == 11.0, "the pixels a revisit would fix"

    def test_a_tile_with_no_thin_pixels_is_null_not_zero(self) -> None:
        """Null says the tile has no pixels below the line. Zero would say it has some, at depth
        zero — which is the never-imaged case `refused_no_optical_px` counts instead.
        """
        rows = registry_rows(
            "r1",
            "t",
            embedded=["clean"],
            refused=[],
            embedded_records={"clean": self._rec(50.0, None, 0)},
            optical_min_obs=15,
        )
        assert rows[0]["refused_px"] == 0
        assert rows[0]["median_obs_where_thin"] is None


class TestWhichBuildRefusedIt:
    """A revisit campaign has two reasons to re-examine a refusal: more imagery now exists, or the
    code that refused it has since been fixed. Several refusal-path defects were fixed in the days
    around the first campaign, so "produced before commit X" is a query someone will actually run.
    """

    def test_every_row_names_the_build(self) -> None:
        rows = registry_rows(
            "r1",
            "t",
            embedded=["chunk_0_0"],
            refused=["chunk_1_1"],
            code={"version": "0.1.0", "commit": "8171f0186ec954f63c43b00ae5e9ed6d253b267e"},
        )
        assert {r["code_commit"] for r in rows} == {"8171f0186ec954f63c43b00ae5e9ed6d253b267e"}
        assert {r["code_version"] for r in rows} == {"0.1.0"}

    @pytest.mark.parametrize(
        ("code", "version"), [({"version": "0.1.0", "editable": True}, "0.1.0"), (None, None)], ids=["wheel", "none"]
    )
    def test_a_build_with_no_commit_is_null_not_blank(self, code, version) -> None:
        """`code_identity` reports a version and no commit for a non-VCS install, and None when its
        own metadata is unreadable. An empty string would read as a known-blank build; null says the
        build did not record one — and no fill may fail over it.
        """
        rows = registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[], code=code)
        assert rows[0]["code_version"] == version
        assert rows[0]["code_commit"] is None


class TestWhereTheTileIs:
    """The bounding box the access request promises, and the two ways a consumer misreads it."""

    def test_every_row_gets_the_box_for_its_own_tile(self) -> None:
        boxes = {
            "chunk_0_0": (-90.0, -0.05, -89.8, 0.13),
            "chunk_5_3": (-89.5, -0.98, -89.3, -0.79),
        }
        rows = registry_rows("r1", "t", embedded=["chunk_0_0"], refused=["chunk_5_3"], bboxes=boxes, optical_min_obs=15)
        by_tile = {r["tile"]: r for r in rows}
        assert by_tile["chunk_0_0"]["bbox_west"] == -90.0
        assert by_tile["chunk_0_0"]["bbox_north"] == 0.13
        assert by_tile["chunk_5_3"]["bbox_south"] == -0.98, "the refused row too, not only embedded"

    def test_a_tile_with_no_box_is_null_rather_than_a_guess(self) -> None:
        """Nulls point nowhere; a guessed box points a consumer at the wrong ground."""
        rows = registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[], bboxes={})
        assert rows[0]["bbox_west"] is None and rows[0]["bbox_east"] is None
