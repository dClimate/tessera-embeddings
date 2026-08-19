"""A refused shard records WHY, because the reason is not recoverable later.

A fully refused shard used to write a zero-byte marker, so a thin-depth refusal was
indistinguishable from land that was never imaged. The dataset computes three reasons per strip and
deliberately keeps them apart; the actor sums them over the chunk; then all of it was discarded. What
survived was a count of "optical skips", which on 2026-08-18 named the wrong cause for 43 of 40S's 58
live shards — every one refused for having no RADAR.

**Why only FULLY refused shards need this.** For a shard that WAS written the evidence is already
published: its ``s2_obs_count`` is real, so a pixel refused for depth is identifiable as
``0 < obs < optical_min_obs`` with a NaN scale. A fully refused shard's mosaic is deleted when the
cell lands, so its reason is the only one that is unrecoverable afterwards.

**Corrected 2026-08-18: a fully refused shard no longer "writes nothing at all".** That was true when
this registry was built, and it was half the problem rather than the justification for it — the obs
counts and month coverage of a refused chunk are real, measured before the gate refused it, and
publishing them as fill made a tile's provenance depend on whether some NEIGHBOURING pixel happened
to embed. A refused chunk now stages a coverage-only tile beside its marker, and assembly copies
those variables while filling the embeddings. The marker still carries the reasons, which no array
can.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

import tessera_embeddings.inference.assembly as _assembly_mod
from tessera_embeddings.config.store_layout import MONTH_COVERED_VAR, MONTHS_IN_YEAR, OBS_COUNT_VARS
from tessera_embeddings.inference.assembly import (
    REFUSAL_REASONS,
    AllChunksSkippedError,
    StagedShardSource,
    ZarrWriter,
    read_skip_records,
    summarise_optical_skips,
)
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.storage.registry import REASONS, part_uri, registry_rows, write_registry_part


def _chunk(row: int, col: int) -> ChunkSpec:
    return ChunkSpec(row=row, col=col, y_start=0, y_stop=8, x_start=0, x_stop=8)


def _record(
    label: str, *, no_optical: int = 0, thin: int = 0, no_radar: int = 0, obs_max: int = 0, eligible: int = 64
) -> dict:
    return {
        "label": label,
        "refused": {"no_optical": no_optical, "thin": thin, "no_radar": no_radar},
        "eligible_px": eligible,
        "s2_obs": {"px_with_any": 64 if obs_max else 0, "max": obs_max, "mean_where_any": float(obs_max)},
    }


class TestTheMarker:
    """Writing and reading one refused shard's record."""

    def test_it_round_trips_its_record(self, tmp_path: Path) -> None:
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(20, 9)
        writer.write_skip_marker(chunk, "run1", _record(chunk.label, thin=64, obs_max=9))
        got, unreadable = read_skip_records(str(tmp_path / "staging"), "run1", [chunk.label])
        assert got[chunk.label]["refused"]["thin"] == 64
        assert got[chunk.label]["s2_obs"]["max"] == 9
        assert unreadable == 0

    def test_an_unserialisable_record_costs_the_record_and_not_the_marker(self, tmp_path: Path) -> None:
        """THE MARKER'S PRESENCE IS LOAD-BEARING; its content is not.

        verify_staged_completeness tells a legitimate skip from a crashed worker by this file
        existing. So a record that cannot be serialised — one careless missing ``int()`` around a
        numpy scalar — must degrade the provenance entry, never turn a benign skip into a failed
        chunk that wedges the cell on every retry.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(4, 4)
        path = writer.write_skip_marker(chunk, "run1", {"refused": {"thin": np.int64(3)}, "bad": object()})
        assert Path(path).exists(), "the marker must exist even when its record does not serialise"
        assert Path(path).read_bytes() == b""
        got, unreadable = read_skip_records(str(tmp_path / "staging"), "run1", [chunk.label])
        assert got == {} and unreadable == 0, "an empty marker is ordinary, not a read failure"

    def test_a_zero_byte_marker_from_before_the_registry_is_not_an_error(self, tmp_path: Path) -> None:
        """A run resuming across the change must still assemble."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(1, 1)
        writer.write_skip_marker(chunk, "run1")
        assert read_skip_records(str(tmp_path / "staging"), "run1", [chunk.label]) == ({}, 0)

    def test_a_corrupt_marker_is_counted_rather_than_silently_dropped(self, tmp_path: Path) -> None:
        """Absent records and failed reads are the same empty dict, and mean opposite things."""
        d = tmp_path / "staging" / "run1"
        d.mkdir(parents=True)
        (d / "chunk_2_2.skipped").write_bytes(b"not json at all")
        got, unreadable = read_skip_records(str(tmp_path / "staging"), "run1", ["chunk_2_2"])
        assert got == {}
        assert unreadable == 1

    def test_a_half_written_multibyte_marker_is_counted_not_raised(self, tmp_path: Path) -> None:
        """A partial write that splits a multibyte character must not abort assembly.

        `json.loads` decodes bytes itself, so a truncated UTF-8 sequence raises
        UnicodeDecodeError — which is NOT a JSONDecodeError, so catching only the
        latter let it escape `pool.map` and kill the whole run over one bad object.
        The bytes below are a real record cut mid-character, not a synthetic blob.
        """
        d = tmp_path / "staging" / "run1"
        d.mkdir(parents=True)
        # ensure_ascii=False so the bytes really are multibyte — the default escapes
        # non-ASCII to \uXXXX and a truncation would only ever be a JSONDecodeError.
        whole = json.dumps({"label": "chunk_2_2", "note": "réfusé"}, ensure_ascii=False).encode()
        (d / "chunk_2_2.skipped").write_bytes(whole[:-3])  # lands mid-sequence
        with pytest.raises(UnicodeDecodeError):
            json.loads(whole[:-3])  # the premise, pinned

        got, unreadable = read_skip_records(str(tmp_path / "staging"), "run1", ["chunk_2_2"])
        assert got == {}
        assert unreadable == 1

    def test_a_missing_marker_is_not_a_read_failure(self, tmp_path: Path) -> None:
        (tmp_path / "staging" / "run1").mkdir(parents=True)
        assert read_skip_records(str(tmp_path / "staging"), "run1", ["chunk_7_7"]) == ({}, 0)

    def test_reading_nothing_asks_for_no_filesystem_at_all(self) -> None:
        """A cell with no skips must not pay a filesystem resolution to learn that."""
        assert read_skip_records("s3://nonexistent-bucket-xyz/staging", "run1", []) == ({}, 0)


class TestTheSummary:
    """Folding the records into the year's provenance."""

    def test_a_thin_refusal_is_distinguishable_from_land_never_imaged(self) -> None:
        """THE POINT OF THE REGISTRY. Both shards hold no data; only the record separates them."""
        records = {
            "chunk_20_9": _record("chunk_20_9", thin=64, obs_max=9),
            "chunk_21_9": _record("chunk_21_9", no_optical=64, obs_max=0),
        }
        out = summarise_optical_skips(staged=["chunk_1_1"], skipped=list(records), records=records)
        assert out["shards_by_reason"] == {"no_optical": ["chunk_21_9"], "thin": ["chunk_20_9"]}
        assert out["refused_px_by_reason"] == {"no_optical": 64, "thin": 64, "no_radar": 0}
        assert out["s2_obs_at_refused"]["max"] == 9, "how thin, not merely that it was thin"

    def test_the_radar_case_that_was_misattributed_is_named_correctly(self) -> None:
        """40S/2023: 43 shards refused for NO RADAR, recorded as "optical skips" with no reason."""
        records = {f"chunk_{r}_9": _record(f"chunk_{r}_9", no_radar=64, obs_max=50) for r in range(20, 27)}
        out = summarise_optical_skips(staged=[], skipped=list(records), records=records)
        assert list(out["shards_by_reason"]) == ["no_radar"]
        assert out["refused_px_by_reason"]["no_radar"] == 7 * 64
        assert out["refused_px_by_reason"]["thin"] == 0, "deep optical must not be reported as thin"
        assert out["s2_obs_at_refused"]["max"] == 50, "the land was deeply imaged; radar is what was absent"

    def test_units_are_in_the_field_names(self) -> None:
        """`tiles_skipped` counts TILES and the refusal totals count PIXELS. The earlier field said
        neither, and a reader comparing them would be comparing two different censuses.
        """
        records = {"chunk_1_1": _record("chunk_1_1", thin=64)}
        out = summarise_optical_skips(staged=[], skipped=["chunk_1_1"], records=records)
        assert "refused_px_by_reason" in out and "by_reason" not in out
        assert out["tiles_skipped"] == 1 and out["refused_px_by_reason"]["thin"] == 64

    def test_the_shard_lists_partition_the_shards(self) -> None:
        """A shard sits under the reason that took the MOST of its pixels, so the list lengths sum to
        the recorded total and nothing is counted twice.
        """
        records = {
            "a": _record("a", thin=40, no_radar=24),
            "b": _record("b", no_radar=64),
            "c": _record("c", no_optical=64),
        }
        out = summarise_optical_skips(staged=[], skipped=list(records), records=records)
        assert sum(len(v) for v in out["shards_by_reason"].values()) == 3
        assert out["shards_by_reason"]["thin"] == ["a"]
        assert out["shards_mixed"] == 1, "a bare dominant label would hide that `a` had two reasons"

    def test_a_shard_with_no_record_is_listed_as_unrecorded_not_as_zero(self) -> None:
        records = {"chunk_1_1": _record("chunk_1_1", thin=64, obs_max=8)}
        out = summarise_optical_skips(staged=[], skipped=["chunk_1_1", "chunk_9_9"], records=records)
        assert out["unrecorded"] == ["chunk_9_9"]
        assert out["tiles_skipped"] == 2, "the count still covers every skipped shard"

    def test_a_record_whose_parts_do_not_sum_to_its_eligible_pixels_is_flagged(self) -> None:
        """THE INVARIANT. The three reasons partition the shard's pixels by construction, so on a
        FULLY refused shard they must account for every eligible pixel. A mismatch means strips did
        not cover the chunk or a count was double-added — either way the pixel totals are wrong, and
        saying so is worth more than a plausible number.
        """
        records = {"chunk_5_5": _record("chunk_5_5", thin=10, eligible=64)}
        out = summarise_optical_skips(staged=[], skipped=["chunk_5_5"], records=records)
        assert out["inconsistent"] == ["chunk_5_5"]

    def test_a_consistent_record_is_not_flagged(self) -> None:
        records = {"chunk_5_5": _record("chunk_5_5", thin=64, eligible=64)}
        out = summarise_optical_skips(staged=[], skipped=["chunk_5_5"], records=records)
        assert "inconsistent" not in out

    def test_a_record_that_refuses_nothing_cannot_explain_an_empty_shard(self) -> None:
        """It means the producer and this summary disagree, which is a defect in one of them — and it
        must not read as a shard whose reason merely went unrecorded.
        """
        records = {"chunk_6_6": _record("chunk_6_6")}
        out = summarise_optical_skips(staged=[], skipped=["chunk_6_6"], records=records)
        assert out["inconsistent"] == ["chunk_6_6"]
        assert out["shards_by_reason"] == {}

    def test_the_old_summary_shape_survives_without_records(self) -> None:
        """A resume across the change must not write a half-populated registry."""
        out = summarise_optical_skips(staged=["a"], skipped=["b"])
        assert out == {"tiles_skipped": 1, "tiles_live": 2, "labels": ["b"]}

    def test_every_reason_the_dataset_computes_has_a_place_in_the_record(self) -> None:
        """A reason the dataset counts but the registry cannot express would be silently dropped."""
        from tessera_embeddings.inference import dataset as ds_mod

        src = Path(ds_mod.__file__).read_text()
        for reason in REFUSAL_REASONS:
            assert f"refused_{reason}" in src, f"the dataset has no refused_{reason} to record"


def test_the_record_stays_small_enough_for_a_zarr_attribute() -> None:
    """It lands in the zone group's attrs, which every reader of that zone pays on every open.

    Organised by reason rather than by shard for this reason as much as for legibility: a nested
    record per shard cost ~200 bytes each, and the largest zones hold 556 live tiles.
    """
    import json

    records = {
        f"chunk_{r}_{c}": _record(f"chunk_{r}_{c}", no_radar=64, obs_max=40) for r in range(24) for c in range(24)
    }
    out = summarise_optical_skips(staged=[], skipped=list(records), records=records)
    size = len(json.dumps(out))
    assert out["tiles_skipped"] == 576
    assert size < 30_000, f"{size} bytes for 576 refused shards is too much to carry in an attribute"


@pytest.mark.parametrize("n", [1, 5, 50])
def test_reading_many_markers_returns_every_one(tmp_path: Path, n: int) -> None:
    """The reader is concurrent; a pool that dropped or duplicated a result would be invisible in a
    single-marker test.
    """
    writer = ZarrWriter(str(tmp_path / "staging"))
    labels = []
    for i in range(n):
        chunk = _chunk(i, 0)
        writer.write_skip_marker(chunk, "runP", _record(chunk.label, thin=i + 1))
        labels.append(chunk.label)
    got, unreadable = read_skip_records(str(tmp_path / "staging"), "runP", labels)
    assert unreadable == 0
    assert len(got) == n
    assert {lbl: got[lbl]["refused"]["thin"] for lbl in labels} == {lbl: i + 1 for i, lbl in enumerate(labels)}


class TestCoverageOnlyStaging:
    """A refused chunk publishes the coverage it measured, not zeros.

    The counts and the month mask are accumulated per strip regardless of validity, so a chunk that
    embeds nothing has still measured its inputs. Filling those alongside the embeddings made a
    tile's published provenance depend on whether a neighbouring pixel happened to embed — a mixed
    tile kept real counts for its refused pixels while a fully refused one reported zero.
    """

    @staticmethod
    def _coverage(h: int = 8, w: int = 8):
        obs = {
            "s2_obs_count": np.full((h, w), 7, dtype=np.uint16),
            "s1_asc_obs_count": np.full((h, w), 3, dtype=np.uint16),
            "s1_desc_obs_count": np.zeros((h, w), dtype=np.uint16),
        }
        months = np.zeros((MONTHS_IN_YEAR, h, w), dtype=bool)
        months[:4] = True
        return obs, months

    def test_it_stages_what_write_chunk_would_have(self, tmp_path: Path) -> None:
        """The equivalence that keeps two code paths in step.

        `write_coverage_only` restates the encoding `write_chunk` uses — axis order, chunking, and
        the bool-becomes-int8 conversion — because assembly reads both with RAW zarr and compares
        against what is on disk. Asserting the two agree is cheaper than trusting them to.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(0, 0)
        obs, months = self._coverage()

        cov = writer.write_coverage_only(chunk, "run1", obs, months)
        emb, scales = np.zeros((8, 8, 128), dtype=np.int8), np.zeros((8, 8), dtype=np.float32)
        full = writer.write_chunk(chunk, emb, run_id="run2", scales=scales, obs_counts=obs, month_covered=months)

        cov_g, full_g = zarr.open_group(cov, mode="r"), zarr.open_group(full, mode="r")
        for var in (*OBS_COUNT_VARS, MONTH_COVERED_VAR):
            assert cov_g[var].dtype == full_g[var].dtype, var
            assert cov_g[var].shape == full_g[var].shape, var
            np.testing.assert_array_equal(cov_g[var][:], full_g[var][:], err_msg=var)
        assert "embeddings" not in cov_g, "the point is to NOT stage a tile of fill embeddings"

    def test_the_listing_does_not_mistake_it_for_a_chunk(self, tmp_path: Path) -> None:
        """It ends in `.zarr`, so the listing would read "<label>.coverage" as a chunk label of its
        own, find no `.done` beside it, and report every refused chunk as an interrupted write.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(0, 0)
        obs, months = self._coverage()
        writer.write_coverage_only(chunk, "run1", obs, months)
        writer.write_skip_marker(chunk, "run1", _record(chunk.label, no_optical=64))

        listing = writer._list_staged("run1")
        assert listing.complete == []
        assert listing.interrupted == [], "the coverage tile must not read as a half-landed pair"
        assert listing.skipped == [chunk.label]

    def test_a_later_successful_write_clears_it(self, tmp_path: Path) -> None:
        """A stale coverage tile carries real counts, so leaving one beside a successful rewrite
        would have assembly publish a footprint that attempt has just replaced.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(0, 0)
        obs, months = self._coverage()
        cov = writer.write_coverage_only(chunk, "run1", obs, months)
        assert Path(cov).exists()

        emb, scales = np.zeros((8, 8, 128), dtype=np.int8), np.zeros((8, 8), dtype=np.float32)
        writer.write_chunk(chunk, emb, run_id="run1", scales=scales)
        assert not Path(cov).exists()

    def test_assembly_reads_the_real_counts_instead_of_fill(self, tmp_path: Path) -> None:
        """The whole point, end to end: a skipped position whose coverage was staged publishes the
        measured counts, and everything it genuinely lacks still publishes as fill.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(0, 0)
        obs, months = self._coverage()
        writer.write_coverage_only(chunk, "run1", obs, months)

        source = StagedShardSource(
            staging_base=str(tmp_path / "staging"),
            run_id="run1",
            shards=(),
            variables=("embeddings", "s2_obs_count", "s1_asc_obs_count", MONTH_COVERED_VAR),
            shard_px=8,
            dtypes=(
                ("embeddings", "int8"),
                ("s2_obs_count", "uint16"),
                ("s1_asc_obs_count", "uint16"),
                (MONTH_COVERED_VAR, "int8"),
            ),
            embedding_dim=128,
            cleared=((0, 0),),
            fill_values=(("embeddings", 0), ("s2_obs_count", 0), ("s1_asc_obs_count", 0), (MONTH_COVERED_VAR, 0)),
        )
        block = source.load((0, 0))

        assert block["s2_obs_count"].max() == 7, "measured, not filled"
        assert block["s1_asc_obs_count"].max() == 3
        assert block[MONTH_COVERED_VAR][..., :4].all(), "the months it did see"
        assert not block[MONTH_COVERED_VAR][..., 4:].any()
        assert block["embeddings"].shape == (1, 8, 8, 128)
        assert not block["embeddings"].any(), "the embeddings are what it failed to produce — fill"

    def test_a_position_with_no_coverage_tile_still_fills(self, tmp_path: Path) -> None:
        """Absence is ordinary and silent: an ocean position never staged coverage, and neither did a
        refused chunk from a run predating this. Both must fill exactly as before.
        """
        source = StagedShardSource(
            staging_base=str(tmp_path / "staging"),
            run_id="run1",
            shards=(),
            variables=("s2_obs_count",),
            shard_px=8,
            dtypes=(("s2_obs_count", "uint16"),),
            cleared=((0, 0),),
            fill_values=(("s2_obs_count", 0),),
        )
        block = source.load((0, 0))
        assert block["s2_obs_count"].shape == (1, 8, 8)
        assert not block["s2_obs_count"].any()


class TestTheGridAnAllSkippedRunWasStagedOn:
    """An assembly-only resume of a run where everything skipped has no tile to measure.

    Falling back to the CURRENT configured size re-enumerates the chunk grid under different labels,
    after which verification reports the valid old markers as unexpected and its own new labels as
    missing — defeating the all-skipped assembly path. The default moved from 2000 to 2048, so any
    run staged before that is affected.
    """

    def test_the_size_is_recovered_from_the_markers(self, tmp_path: Path) -> None:
        writer = ZarrWriter(str(tmp_path / "staging"))
        for col in (0, 1):
            chunk = ChunkSpec(row=0, col=col, y_start=0, y_stop=2000, x_start=0, x_stop=2000)
            record = _record(chunk.label, no_optical=4_000_000, eligible=4_000_000)
            record["chunk_side_px"] = 2000
            writer.write_skip_marker(chunk, "run1", record)

        assert writer.detect_staged_chunk_size("run1") == 2000

    def test_markers_predating_the_field_still_report_unknown(self, tmp_path: Path) -> None:
        """The caller's configured size is the only answer left, and it must still be asked for."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(0, 0)
        writer.write_skip_marker(chunk, "run1", _record(chunk.label, no_optical=64))

        with pytest.raises(AllChunksSkippedError):
            writer.detect_staged_chunk_size("run1")

    def test_markers_that_disagree_report_unknown(self, tmp_path: Path) -> None:
        """A heterogeneous stage no single size describes. Guessing one of them would re-enumerate
        half the grid, which is the failure this recovery exists to prevent.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        for col, side in ((0, 2000), (1, 2048)):
            chunk = ChunkSpec(row=0, col=col, y_start=0, y_stop=side, x_start=0, x_stop=side)
            record = _record(chunk.label, no_optical=side * side, eligible=side * side)
            record["chunk_side_px"] = side
            writer.write_skip_marker(chunk, "run1", record)

        with pytest.raises(AllChunksSkippedError):
            writer.detect_staged_chunk_size("run1")

    def test_a_run_that_never_existed_still_raises(self, tmp_path: Path) -> None:
        """A typo must not become a silent re-run."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        with pytest.raises(FileNotFoundError):
            writer.detect_staged_chunk_size("no-such-run")


class TestWhetherTheRadarRuleWasEvenOn:
    """`no_radar: 0` means two different things and the count cannot say which.

    `allow_s2_only` defaults to FALSE in the library — refusing radar-free land is the default — and
    the global campaign registers True in order to embed it. So under campaign settings that count is
    zero by construction, and a reader who took it for a measurement would conclude the campaign found
    radar everywhere. The summary therefore reports the rule's state beside the count.
    """

    @staticmethod
    def _rec(label: str, *, enforced: bool | None) -> dict:
        record = _record(label, no_optical=64)
        if enforced is not None:
            record["radar_rule_enforced"] = enforced
        return record

    def test_a_campaign_run_says_the_rule_was_disabled(self) -> None:
        summary = summarise_optical_skips(staged=[], skipped=["a"], records={"a": self._rec("a", enforced=False)})
        assert summary["radar_refusal_rule"] == "disabled"
        assert summary["refused_px_by_reason"]["no_radar"] == 0, "and the zero is now readable"

    def test_a_run_that_did_enforce_it_says_so(self) -> None:
        summary = summarise_optical_skips(staged=[], skipped=["a"], records={"a": self._rec("a", enforced=True)})
        assert summary["radar_refusal_rule"] == "enforced"

    def test_records_that_disagree_are_reported_as_mixed(self) -> None:
        """A per-cell orbit downgrade forces the rule off for that cell, so one year can legitimately
        hold both — and averaging them into a single answer would misdescribe every shard.
        """
        summary = summarise_optical_skips(
            staged=[],
            skipped=["a", "b"],
            records={"a": self._rec("a", enforced=True), "b": self._rec("b", enforced=False)},
        )
        assert summary["radar_refusal_rule"] == "mixed"

    def test_records_predating_the_field_say_nothing(self) -> None:
        """Unknown, not assumed: a run from before this field cannot be described either way."""
        summary = summarise_optical_skips(staged=[], skipped=["a"], records={"a": self._rec("a", enforced=None)})
        assert "radar_refusal_rule" not in summary


class TestThePartIsPublishedOnlyAfterTheCellCommits:
    """The registry is a sibling of the store, and nothing reconciles the two.

    So a part written while the assembly was still running advertises embedded and refused tiles for
    a zone-year that may never land — a failed worker, a failed merge, an injected fault — and the
    claim is permanent, to consumers who are explicitly told they need not open the store. The write
    therefore happens after the commit, and these tests pin that rather than the writing.
    """

    def test_the_summary_does_not_write_anything(self, tmp_path: Path) -> None:
        """`_skip_summary` runs as an ARGUMENT to the shard write, so it must not publish. It reads
        the markers — the last moment they exist — and hands them on.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(0, 0)
        writer.write_skip_marker(chunk, "run1", _record(chunk.label, thin=64, obs_max=14))

        summary = writer._skip_summary("run1", [], [chunk.label])

        assert summary["refused_px_by_reason"]["thin"] == 64, "the pooled summary is unchanged"
        assert "registry_part" not in summary, "and it promises no part"
        assert not list((tmp_path / "registry").glob("**/*.parquet")), "nothing was written"

    def test_publishing_writes_a_row_per_live_tile(self, tmp_path: Path) -> None:
        pq = pytest.importorskip("pyarrow.parquet")
        writer = ZarrWriter(str(tmp_path / "staging"))
        root = str(tmp_path / "reg")
        for col in (0, 1):
            chunk = _chunk(0, col)
            writer.write_skip_marker(chunk, "run1", _record(chunk.label, thin=64, obs_max=14))
        writer._skip_summary("run1", ["chunk_9_9"], ["chunk_0_0", "chunk_0_1"])

        uri = writer.publish_registry_part(
            root, "32S", 2021, "run1", embedded=["chunk_9_9"], refused=["chunk_0_0", "chunk_0_1"]
        )

        assert uri == part_uri(root, "32S", 2021, "run1")
        by_tile = {r["tile"]: r for r in pq.read_table(uri).to_pylist()}
        assert set(by_tile) == {"chunk_9_9", "chunk_0_0", "chunk_0_1"}, "every live tile"
        assert by_tile["chunk_9_9"]["embedded"] is True
        assert by_tile["chunk_9_9"]["refused_px"] is None, "nothing counted this, so nothing is claimed"
        assert by_tile["chunk_0_0"]["refused_thin_px"] == 64
        assert by_tile["chunk_0_0"]["obs_max"] == 14, "how close it came, per tile"
        # zone/year are the partition keys, not columns — the identity is in the file metadata.
        assert "year" not in by_tile["chunk_0_0"]
        meta = pq.read_schema(uri).metadata
        assert meta[b"zone"] == b"32S" and meta[b"year"] == b"2021"

    def test_a_failed_publish_never_costs_a_committed_cell(self, tmp_path: Path, monkeypatch) -> None:
        """The cell has already landed by the time this runs. Raising here would fail a complete cell
        over its index, and every column is derivable from the store, so a lost part is rebuildable.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(0, 0)
        writer.write_skip_marker(chunk, "run1", _record(chunk.label, thin=64))
        writer._skip_summary("run1", [], [chunk.label])

        def _boom(*_a, **_k):
            raise OSError("no such bucket")

        monkeypatch.setattr(_assembly_mod, "_fs_for", _boom)
        assert (
            writer.publish_registry_part(str(tmp_path / "reg"), "32S", 2021, "run1", embedded=[], refused=[chunk.label])
            is None
        ), "reported as not published rather than raised"


def test_an_unopenable_staging_filesystem_does_not_fail_the_cell(tmp_path: Path, monkeypatch) -> None:
    """Resolving the filesystem can fail on its own — bad credentials, a malformed URI — and that
    sat outside every guard, so it would have raised out of the year's provenance construction and
    failed a cell at ASSEMBLY, after all its inference was paid for.

    Reported as every marker unreadable, which the caller surfaces loudly, rather than as no reasons
    recorded: those two are the same empty mapping and mean opposite things.
    """

    def _boom(*_a, **_k):
        raise OSError("credentials not found")

    monkeypatch.setattr(_assembly_mod, "_fs_for", _boom)
    records, unreadable = read_skip_records(str(tmp_path), "run1", ["chunk_0_0", "chunk_0_1"])

    assert records == {}
    assert unreadable == 2, "every marker, so the caller can say so rather than imply nothing existed"


class TestTheDatasetIsActuallyReadable:
    """The two defects a red-team pass found, both of which every individual part survived.

    A registry is 1,008 parts read as one dataset, so a part that is valid alone and unmergeable with
    its siblings is worse than a broken part: nothing notices until the whole thing is read.
    """

    @staticmethod
    def _write(root: str, zone: str, year: int, run: str, rows: list[dict]) -> str:
        uri = part_uri(root, zone, year, run)
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        # The lambda is not a leak: `write_registry_part` consumes it as
        # `with open_output(uri) as f`, which is the contract the parameter exists to express.
        write_registry_part(
            uri,
            rows,
            open_output=lambda u: Path(u).open("wb"),  # noqa: SIM115
            zone=zone,
            year=year,
        )
        return uri

    def test_a_cell_that_refused_nothing_still_merges_with_one_that_did(self, tmp_path: Path) -> None:
        """The schema must be DECLARED. Inferred, a cell with no refusals types every refusal column
        `null`, and concatenating it with a cell that did refuse something raises ArrowInvalid — and
        most cells refuse nothing, so the compaction would have broken on almost any pair.
        """
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        root = str(tmp_path / "reg")
        clean = self._write(root, "32S", 2021, "r1", registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[]))
        dirty = self._write(
            root,
            "09S",
            2021,
            "r2",
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
        self._write(root, "32S", 2021, "r1", registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[]))
        self._write(root, "09S", 2022, "r2", registry_rows("r2", "t", embedded=["chunk_1_1"], refused=[]))

        table = ds.dataset(f"{root}/parts", partitioning="hive").to_table()
        assert table.num_rows == 2
        assert set(table.column("zone").to_pylist()) == {"32S", "09S"}, "from the path, not a column"
        assert set(table.column("year").to_pylist()) == {2021, 2022}

    def test_a_refill_adds_a_part_rather_than_replacing_one(self, tmp_path: Path) -> None:
        """Keyed by run so the registry holds both fills of a cell. A fixed name would silently
        replace the original, and dedup is the compaction's decision to make, not a write's.
        """
        ds = pytest.importorskip("pyarrow.dataset")
        root = str(tmp_path / "reg")
        self._write(root, "32S", 2021, "first", registry_rows("first", "t1", embedded=["chunk_0_0"], refused=[]))
        self._write(root, "32S", 2021, "refill", registry_rows("refill", "t2", embedded=["chunk_0_0"], refused=[]))

        table = ds.dataset(f"{root}/parts", partitioning="hive").to_table()
        assert table.num_rows == 2, "both fills are on the record"
        assert set(table.column("run_id").to_pylist()) == {"first", "refill"}
        assert set(table.column("assembled_at").to_pylist()) == {"t1", "t2"}, "and a clock to pick by"

    def test_an_unmeasured_tile_claims_no_refusal_measurement(self, tmp_path: Path) -> None:
        """Null, not zero, when no coverage record reached the row — a resumed success carries none.

        Zero would assert that something counted this tile's refusals and found none, which is the
        misreading the whole registry exists to prevent.
        """
        rows = registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[])
        assert rows[0]["embedded"] is True
        assert rows[0]["refused_px"] is None
        assert all(rows[0][f"refused_{r}_px"] is None for r in REASONS)


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
            "r1", "t", embedded=["chunk_0_0"], refused=[],
            embedded_records={"chunk_0_0": self._partial()}, optical_min_obs=15,
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
            "r1", "t", embedded=["chunk_0_0", "chunk_0_1"], refused=[],
            embedded_records={"chunk_0_0": self._partial(thin=0)}, optical_min_obs=15,
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
            "r1", "t", embedded=["chunk_0_0"], refused=["chunk_1_1"],
            embedded_records={"chunk_0_0": rec}, records={"chunk_1_1": rec}, optical_min_obs=15,
        )
        a, b = (next(r for r in rows if r["tile"] == t) for t in ("chunk_0_0", "chunk_1_1"))
        shared = [c for c in a if c not in ("tile", "embedded")]
        assert {c: a[c] for c in shared} == {c: b[c] for c in shared}
        assert a["embedded"] is True and b["embedded"] is False

    def test_a_marker_wins_over_a_result_for_the_same_label(self) -> None:
        """A label in both sources refused everything at the end, so its marker is the later word."""
        rows = registry_rows(
            "r1", "t", embedded=[], refused=["chunk_0_0"],
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

    @staticmethod
    def _write(root: str, zone: str, year: int, run: str, rows: list[dict], rule: int | None) -> str:
        uri = part_uri(root, zone, year, run)
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        write_registry_part(
            uri,
            rows,
            open_output=lambda u: Path(u).open("wb"),  # noqa: SIM115
            zone=zone,
            year=year,
            extra_metadata={"optical_min_obs": "" if rule is None else str(rule)},
        )
        return uri

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

    def test_a_row_states_the_line_its_own_refusal_was_measured_against(self) -> None:
        """The pairing that makes a refusal interpretable: how deep it got, and what it needed."""
        rows = registry_rows(
            "r1",
            "t",
            embedded=[],
            refused=["chunk_1_1"],
            records={"chunk_1_1": _record("chunk_1_1", thin=64)},
            optical_min_obs=15,
        )
        row = rows[0]
        assert row["optical_min_obs"] == 15
        assert row["obs_max"] is not None, "a distance needs both ends"

    def test_two_cells_filled_under_different_lines_stay_distinguishable(self, tmp_path: Path) -> None:
        """The reason this belongs in the rows and not only in a docstring: a store is write-once on
        its rule, but the registry outlives any one store and a compaction may span two. Reading the
        merged dataset must not average across two different definitions of "thin" silently.
        """
        ds = pytest.importorskip("pyarrow.dataset")
        root = str(tmp_path / "reg")
        self._write(root, "32S", 2021, "r1", registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[], optical_min_obs=15), 15)
        self._write(root, "09S", 2021, "r2", registry_rows("r2", "t", embedded=["chunk_0_0"], refused=[], optical_min_obs=30), 30)

        table = ds.dataset(f"{root}/parts", partitioning="hive").to_table()
        assert table.num_rows == 2
        by_zone = dict(zip(table.column("zone").to_pylist(), table.column("optical_min_obs").to_pylist(), strict=True))
        assert by_zone == {"32S": 15, "09S": 30}

    def test_a_part_read_alone_still_states_the_rule(self, tmp_path: Path) -> None:
        """Also in the key-value block, for a reader that opens one file without the partitioning."""
        pq = pytest.importorskip("pyarrow.parquet")
        root = str(tmp_path / "reg")
        uri = self._write(
            root, "32S", 2021, "r1", registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[], optical_min_obs=15), 15
        )
        md = {k.decode(): v.decode() for k, v in (pq.read_schema(uri).metadata or {}).items()}
        assert md["optical_min_obs"] == "15"
        assert md["zone"] == "32S" and md["year"] == "2021"


class TestWhichBuildRefusedIt:
    """A revisit campaign has two reasons to re-examine a refusal: more imagery now exists, or the
    code that refused it has since been fixed. Several refusal-path defects were fixed in the days
    around the first campaign, so "produced before commit X" is a query someone will actually run.
    """

    def test_every_row_names_the_build(self) -> None:
        rows = registry_rows(
            "r1", "t", embedded=["chunk_0_0"], refused=["chunk_1_1"],
            code={"version": "0.1.0", "commit": "8171f0186ec954f63c43b00ae5e9ed6d253b267e"},
        )
        assert {r["code_commit"] for r in rows} == {"8171f0186ec954f63c43b00ae5e9ed6d253b267e"}
        assert {r["code_version"] for r in rows} == {"0.1.0"}

    def test_a_wheel_install_with_no_commit_is_null_not_blank(self) -> None:
        """`code_identity` reports a version and no commit for a non-VCS install. An empty string
        would read as a known-blank build; null says the build did not record one.
        """
        rows = registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[], code={"version": "0.1.0", "editable": True})
        assert rows[0]["code_version"] == "0.1.0"
        assert rows[0]["code_commit"] is None

    def test_no_build_information_at_all_is_null(self) -> None:
        """`code_identity` returns None rather than raising when its own metadata is unreadable, and
        no fill may fail over that."""
        rows = registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[], code=None)
        assert rows[0]["code_version"] is None and rows[0]["code_commit"] is None


class TestWhereTheTileIs:
    """The bounding box the access request promises, and the two ways a consumer misreads it."""

    def test_every_row_gets_the_box_for_its_own_tile(self) -> None:
        boxes = {
            "chunk_0_0": (-90.0, -0.05, -89.8, 0.13),
            "chunk_5_3": (-89.5, -0.98, -89.3, -0.79),
        }
        rows = registry_rows(
            "r1", "t", embedded=["chunk_0_0"], refused=["chunk_5_3"], bboxes=boxes, optical_min_obs=15
        )
        by_tile = {r["tile"]: r for r in rows}
        assert by_tile["chunk_0_0"]["bbox_west"] == -90.0
        assert by_tile["chunk_0_0"]["bbox_north"] == 0.13
        assert by_tile["chunk_5_3"]["bbox_south"] == -0.98, "the refused row too, not only embedded"

    def test_a_tile_with_no_box_is_null_rather_than_a_guess(self) -> None:
        """Nulls point nowhere; a guessed box points a consumer at the wrong ground."""
        rows = registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[], bboxes={})
        assert rows[0]["bbox_west"] is None and rows[0]["bbox_east"] is None

    def test_an_antimeridian_row_keeps_the_west_greater_than_east_convention(self, tmp_path: Path) -> None:
        """Zones 01 and 60 straddle +/-180. The registry must carry the crossing as GeoJSON does —
        west > east — rather than "helpfully" ordering the pair, which would widen the box to most
        of the globe and report coverage the tile does not have.
        """
        pytest.importorskip("pyproj")
        from tessera_embeddings.storage import zone_grid

        spec = zone_grid.zone("01N")
        west, south, east, north = zone_grid.tile_range_bbox_wgs84(spec, 0, 1, 0, 1)
        rows = registry_rows("r1", "t", embedded=["chunk_0_0"], refused=[], bboxes={"chunk_0_0": (west, south, east, north)})
        row = rows[0]
        assert row["bbox_west"] == west and row["bbox_east"] == east
        assert row["bbox_south"] < row["bbox_north"], "latitude is always ordered"
        if west > east:
            assert west - east > 0, "a crossing row is preserved, not normalised"

    def test_the_box_matches_the_shard_the_label_names(self) -> None:
        """The label's row/col index the SHARD grid — the same reading assembly uses to place a
        tile — so a box must step by one tile pitch per row, not by a pixel or by a chunk of some
        other size. A pitch error yields boxes that look plausible and describe the wrong ground.
        """
        pytest.importorskip("pyproj")
        from tessera_embeddings.config.store_layout import SHARD_PX
        from tessera_embeddings.storage.zone_grid import PIXEL_M, tile_range_bbox_wgs84, zone

        spec = zone("16S")
        top = tile_range_bbox_wgs84(spec, 0, 1, 0, 1)
        next_row = tile_range_bbox_wgs84(spec, 1, 2, 0, 1)
        deg_per_m = 1.0 / 111_320.0

        # One tile south, and by one tile PITCH: ~20.48 km of latitude per row.
        assert next_row[3] < top[3] and next_row[1] < top[1], "row 1 is south of row 0"
        assert abs((top[3] - top[1]) - SHARD_PX * PIXEL_M * deg_per_m) < 0.01

        # Adjacent boxes OVERLAP rather than abut, and that is the containment guarantee showing
        # through: each is the WGS84 envelope of its own projected quad, so along the northing line
        # the two tiles share, one box takes that curve's maximum latitude and the other its
        # minimum. What must never happen is a GAP — land in neither box.
        assert next_row[3] >= top[1], "no gap between vertically adjacent tiles"
        assert (next_row[3] - top[1]) < 10.0 * deg_per_m, "and the overlap stays inside one pixel"


class TestAPartialCoverageTileCannotWedgeACell:
    """The defect a reviewer found in the coverage tile, twice, from both ends.

    `write_coverage_only` failing AFTER `to_zarr` created the array metadata but BEFORE
    `staged_complete` was set leaves a tile that reads back as fill and that `_open_staged_tile`
    refuses. The skip marker written straight afterwards makes every resume omit the chunk, so
    nothing repairs it — and under the stable, input-fingerprinted run id that refusal repeats on
    every retry and wedges the cell until someone deletes the prefix by hand.
    """

    def test_assembly_treats_an_incomplete_coverage_tile_as_absent(self, tmp_path: Path) -> None:
        """Absent and half-written are the same answer for an OPTIONAL artifact: fill, and publish.

        Refusing here would fail assembly over PROVENANCE for a cell whose data is fine.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(0, 0)
        obs = {"s2_obs_count": np.full((8, 8), 7, dtype=np.uint16)}
        writer.write_coverage_only(chunk, "run1", obs, None)
        # Strip the completion attribute, which is exactly what a crash mid-write leaves behind.
        group = zarr.open_group(writer._coverage_path("run1", chunk), mode="a")
        del group.attrs["staged_complete"]

        source = StagedShardSource(
            staging_base=str(tmp_path / "staging"),
            run_id="run1",
            shards=(),
            variables=("s2_obs_count",),
            shard_px=8,
            dtypes=(("s2_obs_count", "uint16"),),
            cleared=((0, 0),),
            fill_values=(("s2_obs_count", 0),),
        )
        block = source.load((0, 0))
        assert not block["s2_obs_count"].any(), "filled, not raised"

    def test_a_partial_tile_is_removed_when_its_write_fails(self, tmp_path: Path) -> None:
        """The other half: the producer deletes what it left, so nothing is there to tolerate."""
        writer = ZarrWriter(str(tmp_path / "staging"))
        chunk = _chunk(0, 0)
        obs = {"s2_obs_count": np.full((8, 8), 7, dtype=np.uint16)}
        path = writer.write_coverage_only(chunk, "run1", obs, None)
        assert Path(path).exists()

        writer.discard_coverage(chunk, "run1")
        assert not Path(path).exists()

    def test_discarding_a_tile_that_is_not_there_is_silent(self, tmp_path: Path) -> None:
        """It runs on a path that is already failing, so it must never raise: another exception there
        would replace a degraded provenance entry with a lost chunk.
        """
        writer = ZarrWriter(str(tmp_path / "staging"))
        writer.discard_coverage(_chunk(4, 4), "run1")  # must not raise
