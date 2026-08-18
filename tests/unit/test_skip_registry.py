"""A refused shard records WHY, because the reason is not recoverable later.

A fully refused shard used to write a zero-byte marker, so a thin-depth refusal was
indistinguishable from land that was never imaged. The dataset computes three reasons per strip and
deliberately keeps them apart; the actor sums them over the chunk; then all of it was discarded. What
survived was a count of "optical skips", which on 2026-08-18 named the wrong cause for 43 of 40S's 58
live shards — every one refused for having no RADAR — and gave no way to tell those from unimaged
land. The cell is write-once and its mosaic is deleted when it lands, so this is the only moment the
reason exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from tessera_embeddings.inference.assembly import (
    REFUSAL_REASONS,
    ZarrWriter,
    read_skip_records,
    summarise_optical_skips,
)
from tessera_embeddings.inference.chunk_spec import ChunkSpec


def _chunk(row: int, col: int) -> ChunkSpec:
    return ChunkSpec(row=row, col=col, y_start=0, y_stop=8, x_start=0, x_stop=8)


def _record(label: str, *, no_optical: int = 0, thin: int = 0, no_radar: int = 0, obs_max: int = 0) -> dict:
    return {
        "label": label,
        "refused": {"no_optical": no_optical, "thin": thin, "no_radar": no_radar},
        "eligible_px": 64,
        "s2_obs": {"px_with_any": 64 if obs_max else 0, "max": obs_max, "mean_where_any": float(obs_max)},
    }


def test_the_marker_round_trips_its_record(tmp_path: Path) -> None:
    writer = ZarrWriter(str(tmp_path / "staging"))
    chunk = _chunk(20, 9)
    writer.write_skip_marker(chunk, "run1", _record(chunk.label, thin=64, obs_max=9))
    got = read_skip_records(str(tmp_path / "staging"), "run1", [chunk.label])
    assert got[chunk.label]["refused"]["thin"] == 64
    assert got[chunk.label]["s2_obs"]["max"] == 9


def test_a_zero_byte_marker_from_before_the_registry_is_not_an_error(tmp_path: Path) -> None:
    """A run resuming across the change must still assemble. Absent is not zero — the caller has to
    be able to say "reason not recorded", which is a different claim from "nothing was refused"."""
    writer = ZarrWriter(str(tmp_path / "staging"))
    chunk = _chunk(1, 1)
    writer.write_skip_marker(chunk, "run1")  # the old form
    assert read_skip_records(str(tmp_path / "staging"), "run1", [chunk.label]) == {}


def test_an_unreadable_marker_is_skipped_rather_than_raising(tmp_path: Path) -> None:
    d = tmp_path / "staging" / "run1"
    d.mkdir(parents=True)
    (d / "chunk_2_2.skipped").write_bytes(b"not json at all")
    assert read_skip_records(str(tmp_path / "staging"), "run1", ["chunk_2_2"]) == {}


def test_a_thin_refusal_is_distinguishable_from_land_never_imaged() -> None:
    """THE POINT OF THE REGISTRY. Both shards hold no data; only the record separates them."""
    records = {
        "chunk_20_9": _record("chunk_20_9", thin=64, obs_max=9),  # imaged, too thin
        "chunk_21_9": _record("chunk_21_9", no_optical=64, obs_max=0),  # never imaged
    }
    out = summarise_optical_skips(staged=["chunk_1_1"], skipped=list(records), records=records)
    assert out["shards"]["chunk_20_9"]["reason"] == "thin"
    assert out["shards"]["chunk_21_9"]["reason"] == "no_optical"
    assert out["shards"]["chunk_20_9"]["s2_obs"]["max"] == 9, "how thin, not merely that it was thin"
    assert out["by_reason"] == {"no_optical": 64, "thin": 64, "no_radar": 0}


def test_the_radar_case_that_was_misattributed_is_named_correctly() -> None:
    """40S/2023: 43 shards refused for NO RADAR, recorded as "optical skips" with no reason."""
    records = {f"chunk_{r}_9": _record(f"chunk_{r}_9", no_radar=64, obs_max=50) for r in range(20, 27)}
    out = summarise_optical_skips(staged=[], skipped=list(records), records=records)
    assert {sh["reason"] for sh in out["shards"].values()} == {"no_radar"}
    assert out["by_reason"]["no_radar"] == 7 * 64
    assert out["by_reason"]["thin"] == 0, "deep optical must not be reported as thin"


def test_a_mixed_shard_is_named_by_what_dominates_it() -> None:
    records = {"chunk_3_3": _record("chunk_3_3", thin=10, no_radar=54, obs_max=20)}
    out = summarise_optical_skips(staged=[], skipped=["chunk_3_3"], records=records)
    assert out["shards"]["chunk_3_3"]["reason"] == "no_radar"
    assert out["shards"]["chunk_3_3"]["refused"] == {"no_optical": 0, "thin": 10, "no_radar": 54}


def test_a_shard_with_no_record_is_listed_as_unrecorded_not_as_zero() -> None:
    """Folding it into a zero would say "nothing was refused here", which is the
    absence-as-evidence error the whole record exists to avoid."""
    records = {"chunk_1_1": _record("chunk_1_1", thin=64, obs_max=8)}
    out = summarise_optical_skips(staged=[], skipped=["chunk_1_1", "chunk_9_9"], records=records)
    assert out["unrecorded"] == ["chunk_9_9"]
    assert "chunk_9_9" not in out["shards"]
    assert out["tiles_skipped"] == 2, "the count still covers every skipped shard"


def test_the_old_summary_shape_survives_without_records() -> None:
    """Callers that pass no records keep exactly the previous record, so a resume across the change
    does not write a half-populated registry."""
    out = summarise_optical_skips(staged=["a"], skipped=["b"])
    assert out == {"tiles_skipped": 1, "tiles_live": 2, "labels": ["b"]}


def test_every_reason_the_dataset_computes_has_a_place_in_the_record() -> None:
    """A reason the dataset counts but the registry cannot express would be silently dropped."""
    from tessera_embeddings.inference import dataset as ds_mod

    src = Path(ds_mod.__file__).read_text()
    for reason in REFUSAL_REASONS:
        assert f"refused_{reason}" in src, f"the dataset has no refused_{reason} to record"
