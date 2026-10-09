"""The rebuild that re-derives registry measurements from the store.

The claim this module rests on is that a rebuild is a RE-DERIVATION and not an estimate, so the
tests that matter are the ones that would catch it drifting away from the fill's own arithmetic:
the drift guard against ``actors._coverage_record``, and the gate's verdicts.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import zarr

from tessera_embeddings.inference.actors import _coverage_record
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.storage import registry_rebuild as rb

RULE = 15


def _obs(values: list[list[int]]) -> np.ndarray:
    return np.array(values, dtype=np.uint16)


class TestMeasurementsFromObs:
    """The pure recompute."""

    def test_partitions_the_tile_into_zero_thin_and_deep(self) -> None:
        """no_optical counts zeros, thin counts 1..rule-1, and neither claims a deep pixel."""
        got = rb.measurements_from_obs(_obs([[0, 0, 3], [14, 15, 40]]), optical_min_obs=RULE, radar_rule_enforced=False)
        assert got["refused_no_optical_px"] == 2
        assert got["refused_thin_px"] == 2  # 3 and 14
        assert got["refused_px"] == 4
        assert got["px_with_any_optical"] == 4
        assert got["chunk_px"] == 6

    def test_eligible_px_is_null_because_the_crop_is_not_in_the_store(self) -> None:
        """The read plan's x crop is not derivable here, and null is the registry's word for that.

        Writing ``chunk_px`` into it would claim the fill evaluated the whole tile, which for a
        cropped chunk is exactly the false statement the column exists to prevent.
        """
        got = rb.measurements_from_obs(_obs([[20]]), optical_min_obs=RULE, radar_rule_enforced=False)
        assert got["eligible_px"] is None

    def test_median_where_thin_is_none_rather_than_zero_when_nothing_is_thin(self) -> None:
        """A tile with no thin pixels is not a tile whose thin pixels sit at zero."""
        got = rb.measurements_from_obs(_obs([[20, 30]]), optical_min_obs=RULE, radar_rule_enforced=False)
        assert got["median_obs_where_thin"] is None
        assert got["median_obs_where_any"] == 25.0

    def test_median_where_any_is_zero_on_a_tile_nothing_imaged(self) -> None:
        """Mirrors the fill: no observed pixel gives 0.0, not NaN and not null."""
        got = rb.measurements_from_obs(_obs([[0, 0]]), optical_min_obs=RULE, radar_rule_enforced=False)
        assert got["median_obs_where_any"] == 0.0
        assert got["obs_max"] == 0

    def test_radar_presence_needs_both_orbits_or_neither(self) -> None:
        """One orbit alone counts half a sensor and would understate presence silently."""
        with pytest.raises(ValueError, match="both orbits or neither"):
            rb.measurements_from_obs(_obs([[20]]), optical_min_obs=RULE, radar_rule_enforced=False, s1_asc=_obs([[1]]))

    def test_radar_presence_is_null_when_not_measured(self) -> None:
        """An optical-only pass must not publish a zero it never counted."""
        got = rb.measurements_from_obs(_obs([[20]]), optical_min_obs=RULE, radar_rule_enforced=False)
        assert got["px_with_any_radar"] is None

    def test_radar_presence_counts_either_orbit(self) -> None:
        got = rb.measurements_from_obs(
            _obs([[20, 20, 20]]),
            optical_min_obs=RULE,
            radar_rule_enforced=False,
            s1_asc=_obs([[1, 0, 0]]),
            s1_desc=_obs([[0, 2, 0]]),
        )
        assert got["px_with_any_radar"] == 2

    def test_the_radar_rule_being_in_force_is_refused_rather_than_guessed(self) -> None:
        """The store records radar PRESENCE, never which pixels the radar rule refused.

        Every cell of the published campaign ran with the rule off, so a rebuild that accepted
        ``True`` here would have to invent ``refused_no_radar_px``.
        """
        with pytest.raises(ValueError, match="cannot derive"):
            rb.measurements_from_obs(_obs([[20]]), optical_min_obs=RULE, radar_rule_enforced=True)


class TestDriftAgainstTheFill:
    """The rebuild and the fill must produce the same numbers on the same pixels."""

    def test_optical_statistics_match_the_actors_coverage_record(self) -> None:
        """Two implementations of one record is how two registry rows stop being comparable.

        ``_coverage_record`` is what the fill wrote; this is what a rebuild writes. They are
        driven here over the same array, including the rounding, because a tenth of a decimal
        place apart is enough to fail the gate every backfilled row depends on.
        """
        rng = np.random.default_rng(0)
        s2 = rng.integers(0, 60, size=(64, 64)).astype(np.uint16)
        s2[:8] = 0  # a band of never-imaged pixels, so no_optical is exercised
        asc = rng.integers(0, 3, size=(64, 64)).astype(np.uint16)
        desc = rng.integers(0, 3, size=(64, 64)).astype(np.uint16)

        spec = ChunkSpec(row=0, col=0, y_start=0, y_stop=64, x_start=0, x_stop=64)
        fill = _coverage_record(
            spec,
            refused={"no_optical": 0, "thin": 0, "no_radar": 0},
            radar_rule_enforced=False,
            obs_buffers={"s2_obs_count": s2, "s1_asc_obs_count": asc, "s1_desc_obs_count": desc},
            x_sub=None,
            optical_min_obs=RULE,
        )
        rebuilt = rb.measurements_from_obs(
            s2, optical_min_obs=RULE, radar_rule_enforced=False, s1_asc=asc, s1_desc=desc
        )
        assert rebuilt["chunk_px"] == fill["chunk_px"]
        assert rebuilt["px_with_any_optical"] == fill["s2_obs"]["px_with_any"]
        assert rebuilt["obs_max"] == fill["s2_obs"]["max"]
        assert rebuilt["median_obs_where_any"] == fill["s2_obs"]["median_where_any"]
        assert rebuilt["median_obs_where_thin"] == fill["s2_obs"]["median_where_thin"]
        assert rebuilt["px_with_any_radar"] == fill["px_with_any_radar"]

    def test_the_refusal_split_matches_the_datasets_rule(self) -> None:
        """``has_optical`` is ``s2_obs_count > 0`` on the delivered product — the whole basis here.

        Asserted against the expressions ``inference/dataset.py`` gates on, so a change to the
        rule that made the store insufficient to reproduce it fails here rather than silently
        producing rows that disagree with the fill's.
        """
        rng = np.random.default_rng(1)
        s2 = rng.integers(0, 40, size=(32, 32)).astype(np.uint16)
        has_optical = s2 > 0
        deep_enough = s2 >= RULE
        rebuilt = rb.measurements_from_obs(s2, optical_min_obs=RULE, radar_rule_enforced=False)
        assert rebuilt["refused_no_optical_px"] == int((~has_optical).sum())
        assert rebuilt["refused_thin_px"] == int((has_optical & ~deep_enough).sum())


class TestCompareRow:
    """The gate's verdicts."""

    def _recorded(self, **overrides: object) -> dict:
        row = {
            "eligible_px": 6,
            "chunk_px": 6,
            "refused_px": 4,
            "refused_no_optical_px": 2,
            "refused_thin_px": 2,
            "refused_no_radar_px": 0,
            "px_with_any_optical": 4,
            "obs_max": 40,
            "median_obs_where_any": 14.5,
            "median_obs_where_thin": 8.5,
            "px_with_any_radar": None,
            "radar_rule_enforced": False,
        }
        row.update(overrides)
        return row

    def _rebuilt(self) -> dict:
        return rb.measurements_from_obs(
            _obs([[0, 0, 3], [14, 15, 40]]), optical_min_obs=RULE, radar_rule_enforced=False
        )

    def test_a_faithful_rebuild_matches(self) -> None:
        verdict, differences = rb.compare_row(self._rebuilt(), self._recorded())
        assert (verdict, differences) == ("match", [])

    def test_a_cropped_row_is_excluded_rather_than_failed(self) -> None:
        """A whole-tile rebuild of a tile the fill only partly evaluated is EXPECTED to differ."""
        verdict, differences = rb.compare_row(self._rebuilt(), self._recorded(eligible_px=4))
        assert (verdict, differences) == ("cropped", [])

    def test_a_row_with_no_measurements_is_the_population_being_filled(self) -> None:
        verdict, _ = rb.compare_row(self._rebuilt(), self._recorded(chunk_px=None))
        assert verdict == "unmeasured"

    def test_a_disagreement_is_reported_per_column(self) -> None:
        verdict, differences = rb.compare_row(self._rebuilt(), self._recorded(refused_thin_px=3, refused_px=5))
        assert verdict == "mismatch"
        assert any("refused_thin_px" in d for d in differences)
        assert any("refused_px" in d for d in differences)

    def test_eligible_px_alone_never_fails_the_gate(self) -> None:
        """The rebuild leaves it null on purpose; comparing it would fail every row."""
        verdict, _ = rb.compare_row(self._rebuilt(), self._recorded())
        assert verdict == "match"

    def test_an_optical_only_pass_is_not_failed_for_the_radar_column(self) -> None:
        verdict, _ = rb.compare_row(self._rebuilt(), self._recorded(px_with_any_radar=99))
        assert verdict == "match"

    def test_a_measured_radar_column_is_compared(self) -> None:
        rebuilt = rb.measurements_from_obs(
            _obs([[0, 0, 3], [14, 15, 40]]),
            optical_min_obs=RULE,
            radar_rule_enforced=False,
            s1_asc=_obs([[0, 0, 0], [0, 0, 0]]),
            s1_desc=_obs([[0, 0, 0], [0, 0, 0]]),
        )
        verdict, differences = rb.compare_row(rebuilt, self._recorded(px_with_any_radar=99))
        assert verdict == "mismatch"
        assert any("px_with_any_radar" in d for d in differences)


@pytest.mark.parametrize(
    ("no_optical", "violating"),
    [(2, 0), (1, 1)],
    ids=["optical-test-is-obs-gt-0", "reflectance-term-removed-a-pixel"],
)
def test_basis_violations_counts_measured_rows_that_break_the_identity(no_optical: int, violating: int) -> None:
    """The registry-only check the gate runs before reading a pixel; an unmeasured row is not a violation."""
    table = pa.table(
        {
            "chunk_px": [6, None],
            "eligible_px": [6, None],
            "px_with_any_optical": [4, None],
            "refused_no_optical_px": [no_optical, None],
        }
    )
    assert rb.basis_violations(table) == (violating, 1)


class TestTileAddressing:
    """A tile label names one shard; addressing the wrong one is silently plausible."""

    def test_label_round_trips_to_a_shard_window(self) -> None:
        assert rb.tile_window("chunk_1_2", (9, 8192, 8192)) == (2048, 4096, 4096, 6144)

    def test_the_window_clamps_at_a_zone_margin(self) -> None:
        """A zone's extent is a whole number of shards only because the grid was built that way."""
        assert rb.tile_window("chunk_1_0", (9, 3000, 2048)) == (2048, 3000, 0, 2048)

    def test_a_label_that_is_not_a_tile_raises_rather_than_addressing_something_else(self) -> None:
        with pytest.raises(ValueError, match="chunk_<row>_<col>"):
            rb.parse_tile_label("tile_1_2")


class TestReadingTheStore:
    """Reading a real (small) store, so the window and year arithmetic is exercised end to end."""

    def test_rebuild_tile_reads_the_right_window_and_year(self, tmp_path) -> None:
        """A tile is one shard of one year; reading the wrong slice would be silently plausible."""
        root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
        for name in ("s2_obs_count", *rb.RADAR_OBS_VARS):
            root.create_array(name, shape=(2, 4096, 4096), dtype="uint16", chunks=(1, 2048, 2048))
        root["s2_obs_count"][1, 2048:4096, 0:2048] = 20
        root["s2_obs_count"][1, 2048:4096, 0:1024] = 4  # half the tile thin
        got = rb.rebuild_tile(root, time_index=1, tile="chunk_1_0", optical_min_obs=RULE, with_radar=False)
        assert got["chunk_px"] == 2048 * 2048
        assert got["refused_thin_px"] == 2048 * 1024
        assert got["refused_no_optical_px"] == 0
        assert got["px_with_any_radar"] is None

    def test_a_year_the_tile_holds_nothing_in_reads_as_never_imaged(self, tmp_path) -> None:
        root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
        root.create_array("s2_obs_count", shape=(2, 2048, 2048), dtype="uint16", chunks=(1, 2048, 2048))
        got = rb.rebuild_tile(root, time_index=0, tile="chunk_0_0", optical_min_obs=RULE, with_radar=False)
        assert got["refused_no_optical_px"] == 2048 * 2048
        assert got["median_obs_where_any"] == 0.0


class TestRebuiltRow:
    """The row a rebuild publishes: carried identity, re-derived measurements."""

    def test_identity_is_carried_and_measurements_are_replaced(self) -> None:
        """A rebuild does not become the build that refused the tile."""
        source = {
            "tile": "chunk_0_0",
            "embedded": True,
            "optical_min_obs": 15,
            "bbox_west": 1.0,
            "bbox_south": 2.0,
            "bbox_east": 3.0,
            "bbox_north": 4.0,
            "code_version": "0.1.0",
            "code_commit": "abc123",
            "assembled_at": "2026-08-01T00:00:00+00:00",
            "run_id": "fill-run",
        }
        measurements = rb.measurements_from_obs(_obs([[20]]), optical_min_obs=15, radar_rule_enforced=False)
        row = rb.rebuilt_row(source, measurements, run_id="rebuild-1", assembled_at="2026-09-21T00:00:00+00:00")
        assert row["code_commit"] == "abc123"
        assert row["run_id"] == "rebuild-1"
        assert row["filled_at"] == "2026-08-01T00:00:00+00:00"
        assert row["chunk_px"] == 1

    def test_a_run_id_says_it_is_a_rebuild(self) -> None:
        assert rb.rebuild_run_id(suffix="optical").startswith("rebuild-")
