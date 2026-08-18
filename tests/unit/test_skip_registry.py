"""A refused shard records WHY, because the reason is not recoverable later.

A fully refused shard used to write a zero-byte marker, so a thin-depth refusal was
indistinguishable from land that was never imaged. The dataset computes three reasons per strip and
deliberately keeps them apart; the actor sums them over the chunk; then all of it was discarded. What
survived was a count of "optical skips", which on 2026-08-18 named the wrong cause for 43 of 40S's 58
live shards — every one refused for having no RADAR.

**Why only FULLY refused shards need this.** For a shard that WAS written the evidence is already
published: its ``s2_obs_count`` is real, so a pixel refused for depth is identifiable as
``0 < obs < optical_min_obs`` with a NaN scale. A fully refused shard writes nothing at all, its obs
counts read back as fill, and its mosaic is deleted when the cell lands — so that is the only case
where the reason is unrecoverable, and the only case this registry has to cover.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tessera_embeddings.inference.assembly import (
    REFUSAL_REASONS,
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
