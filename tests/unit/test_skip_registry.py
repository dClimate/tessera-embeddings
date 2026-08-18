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
