"""Tests for the profiling instrumentation + its parser.

Covers the three pieces that must stay in sync for the profiling tools to
work against any run:

- ``ResourceMonitor`` context slots (the ``ctx=`` attribution on RESOURCES lines)
- ``actors._chunk_summary_line`` (the machine-readable per-chunk contract)
- ``observe_cluster.PHASE_PARSER`` consuming those lines (and its legacy
  prose-log fallback for runs from older code)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

from tessera_embeddings.inference.actors import _chunk_summary_line
from tessera_embeddings.inference.resource_monitor import ResourceMonitor

REPO_ROOT = Path(__file__).resolve().parents[2]
OBSERVE_CLUSTER = REPO_ROOT / "scripts" / "inference_perf" / "observe_cluster.py"
COMPARE_COARSENED = REPO_ROOT / "scripts" / "inference_perf" / "compare_coarsened_stores.py"


def _load_script(path: Path, name: str):
    """Import a scripts/ module by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves field types via
    # sys.modules[cls.__module__], which is absent for a bare exec_module.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_phase_parser() -> str:
    """Import observe_cluster.py by path (scripts/ is not a package)."""
    return _load_script(OBSERVE_CLUSTER, "observe_cluster").PHASE_PARSER


def _run_parser(logs_dir: Path) -> str:
    """Run PHASE_PARSER as workers do (a python subprocess), against logs_dir."""
    result = subprocess.run(
        [sys.executable, "-c", _load_phase_parser()],
        capture_output=True,
        text=True,
        env={"TESSERA_RAY_LOGS": str(logs_dir), "PATH": "/usr/bin:/bin"},
        check=True,
    )
    return result.stdout


class TestResourceMonitorContext:
    """ctx= attribution slots on RESOURCES lines."""

    def test_context_slots_appear_sorted_on_resources_line(self, caplog) -> None:
        mon = ResourceMonitor()
        mon.set_context("work", "chunk_1_2:s1/3")
        mon.set_context("write", "chunk_1_1")
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.inference.resource_monitor"):
            mon._emit_once()
        line = next(r.message for r in caplog.records if "RESOURCES" in r.message)
        # Sorted slot order → stable "work" after "write".
        assert "ctx=work:chunk_1_2:s1/3" in line
        assert "write:chunk_1_1" in line

    def test_clearing_a_slot_removes_it(self, caplog) -> None:
        mon = ResourceMonitor()
        mon.set_context("work", "chunk_0_0:prologue")
        mon.set_context("work", None)
        mon.set_context("write", "chunk_0_0")
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.inference.resource_monitor"):
            mon._emit_once()
        line = next(r.message for r in caplog.records if "RESOURCES" in r.message)
        assert "work:" not in line
        assert "ctx=write:chunk_0_0" in line


class TestChunkSummaryLine:
    """The machine-readable per-chunk contract."""

    def test_round_trips_as_json_with_stable_prefix(self) -> None:
        line = _chunk_summary_line(label="chunk_0_0", status="success", valid_px=123, total_s=4.5)
        assert line.startswith("CHUNK_SUMMARY: ")
        payload = json.loads(line[len("CHUNK_SUMMARY: ") :])
        assert payload == {"label": "chunk_0_0", "status": "success", "valid_px": 123, "total_s": 4.5}

    def test_keys_are_sorted_for_stable_diffs(self) -> None:
        line = _chunk_summary_line(z=1, a=2)
        assert line.index('"a"') < line.index('"z"')


class TestPhaseParser:
    """observe_cluster's on-worker phase-table builder."""

    def test_prefers_chunk_summary_lines(self, tmp_path: Path) -> None:
        """The parser consumes actor-emitted CHUNK_SUMMARY lines end-to-end."""
        ok = _chunk_summary_line(
            label="chunk_0_0",
            status="success",
            valid_px=1000,
            total_s=20.0,
            prologue_s=3.0,
            infer_s=15.0,
            overhead_s=5.0,
            strips=2,
            strip_h=1024,
            strategy="dense/prefetch",
            t_kept=50,
            rung="starter",
            x_crop_w=None,
        )
        skip = _chunk_summary_line(
            label="chunk_0_1",
            status="skipped",
            valid_px=0,
            total_s=2.0,
            prologue_s=2.0,
            infer_s=0.0,
            overhead_s=2.0,
            strips=1,
            strip_h=2000,
            strategy="single",
            t_kept=8,
            rung="serial",
            x_crop_w=120,
        )
        (tmp_path / "worker-abc.err").write_text(
            f"2026-07-21 10:00:00,000 X INFO {ok}\n"
            "2026-07-21 10:00:01,000 X INFO some unrelated line\n"
            f"2026-07-21 10:00:02,000 X INFO {skip}\n"
        )
        out = _run_parser(tmp_path)
        assert out.splitlines()[0].startswith("label\tTkept\tstrips\trung")
        row = next(ln for ln in out.splitlines() if ln.startswith("chunk_0_0"))
        # label, t_kept, strips, rung, valid_px ... px/s = 1000/20 = 50
        assert row.split("\t") == [
            "chunk_0_0",
            "50",
            "2",
            "starter",
            "1000",
            "3.0",
            "15.0",
            "5.0",
            "20.0",
            "50",
            "success",
        ]
        assert any(ln.startswith("chunk_0_1") and ln.endswith("skipped") for ln in out.splitlines())

    def test_falls_back_to_legacy_prose_parsing(self, tmp_path: Path) -> None:
        """Runs from pre-CHUNK_SUMMARY code still produce a phase table."""
        (tmp_path / "worker-old.err").write_text(
            "2026-07-21 10:00:00,000 X INFO Chunk chunk_0_0: T_kept=50 -> strip_h=2000 -> 1 strip(s)\n"
            "2026-07-21 10:00:05,000 X INFO MosaicChunkInferenceDataset: 1000 valid pixels"
            " out of 4000000 total (0.03%) in 3 buckets\n"
            "2026-07-21 10:00:06,000 X INFO Starting v1.1 inference\n"
            "2026-07-21 10:00:16,000 X INFO Inference complete: strip done 10.0s total, 100 px/sec\n"
            "2026-07-21 10:00:17,000 X INFO Chunk chunk_0_0 complete: 1000 valid pixels, 17.0s\n"
        )
        out = _run_parser(tmp_path)
        assert "pref\tbuckets" in out.splitlines()[0]  # legacy header
        row = next(ln for ln in out.splitlines() if ln.startswith("chunk_0_0"))
        fields = row.split("\t")
        assert fields[1] == "50"  # T_kept
        assert fields[5] == "1000"  # valid_px from the cdone line


def _make_coarse_ds(emb: object, *, with_time: bool = True):
    """Minimal coarsened-format xr.Dataset from an (northing, easting, band) array."""
    import numpy as np
    import xarray as xr

    n, e, b = emb.shape
    coords = {"northing": np.arange(n), "easting": np.arange(e), "band": np.arange(b)}
    dims = ("northing", "easting", "band")
    obs_dims = ("northing", "easting")
    data = emb.astype("float32")
    obs = np.ones((n, e), dtype="uint32")
    if with_time:
        coords["time"] = [0]
        dims, obs_dims = ("time", *dims), ("time", *obs_dims)
        data, obs = data[None], obs[None]
    return xr.Dataset(
        {
            "embeddings": (dims, data),
            "s2_obs_count": (obs_dims, obs),
            "s1_asc_obs_count": (obs_dims, obs.copy()),
            "s1_desc_obs_count": (obs_dims, obs.copy()),
        },
        coords=coords,
    )


class TestCoarsenedCompare:
    """compare_coarsened_stores: chunk-aligned sampling, dim-aware selection, structure gate."""

    def test_row_slabs_sampling_is_chunk_aligned(self) -> None:
        mod = _load_script(COMPARE_COARSENED, "compare_coarsened_stores")
        slabs = mod._row_slabs(1000, sample_rows=10)  # ROW_BLOCK=384 → chunks 0,1,2
        starts = [s.start for s, _ in slabs]
        assert starts == sorted(set(starts))  # each backing chunk read at most once
        assert all(st % mod.ROW_BLOCK == 0 for st in starts)
        for s, sel in slabs:  # selections are slab-relative offsets within the chunk
            assert sel is not None
            assert all(0 <= o < mod.ROW_BLOCK and s.start + o < 1000 for o in sel)

    def test_sampled_compare_selects_northing_rows(self) -> None:
        import numpy as np

        mod = _load_script(COMPARE_COARSENED, "compare_coarsened_stores")
        base = np.arange(8 * 4 * 2, dtype="float32").reshape(8, 4, 2)
        rows = mod._row_slabs(8, sample_rows=3)  # → rows 0, 3, 7 (one slab)
        ref = _make_coarse_ds(base)

        differ_unsampled = base.copy()
        differ_unsampled[5] += 1.0  # row 5 is NOT in {0,3,7}
        assert mod.compare_variable(ref, _make_coarse_ds(differ_unsampled), "embeddings", rows).bit_identical

        differ_sampled = base.copy()
        differ_sampled[3] += 1.0  # row 3 IS sampled → must be caught
        assert not mod.compare_variable(ref, _make_coarse_ds(differ_sampled), "embeddings", rows).bit_identical

    def test_sampled_compare_dim_aware_without_time_axis(self) -> None:
        """A time-less store still selects northing rows (positional indexing would not)."""
        import numpy as np

        mod = _load_script(COMPARE_COARSENED, "compare_coarsened_stores")
        base = np.arange(8 * 4 * 2, dtype="float32").reshape(8, 4, 2)
        rows = mod._row_slabs(8, sample_rows=3)
        ref = _make_coarse_ds(base, with_time=False)
        differ = base.copy()
        differ[3] += 1.0  # sampled northing row
        assert not mod.compare_variable(ref, _make_coarse_ds(differ, with_time=False), "embeddings", rows).bit_identical

    def test_var_structure_flags_dtype_mismatch(self) -> None:
        import numpy as np

        mod = _load_script(COMPARE_COARSENED, "compare_coarsened_stores")
        base = np.zeros((4, 4, 2), dtype="float32")
        ref = _make_coarse_ds(base)
        test = _make_coarse_ds(base)
        # Same values, narrower obs dtype — bit-view compare would miss it.
        test["s2_obs_count"] = test["s2_obs_count"].astype("uint16")
        problems = mod.compare_var_structure(ref, test)
        assert any("s2_obs_count" in p and "dtype" in p for p in problems)

    def test_non_spatial_variable_compared_not_crashed(self) -> None:
        """A var without a northing dim reads whole instead of crashing on isel(northing=…)."""
        import numpy as np

        mod = _load_script(COMPARE_COARSENED, "compare_coarsened_stores")
        base = np.zeros((4, 4, 2), dtype="float32")
        ref = _make_coarse_ds(base)
        ref["aux"] = 5.0  # scalar (0-d) data var — no northing dim
        test = _make_coarse_ds(base)
        test["aux"] = 5.0
        rows = mod._row_slabs(4, sample_rows=None)
        cmp = mod.compare_variable(ref, test, "aux", rows)  # must not raise
        assert cmp.n == 1
        assert cmp.bit_identical

    def test_finite_mask_excludes_infinities(self) -> None:
        """inf-inf=NaN must not poison the abs-diff stats when a store has infinities."""
        import numpy as np

        mod = _load_script(COMPARE_COARSENED, "compare_coarsened_stores")
        a = np.array([[1.0, np.inf, 3.0]], dtype="float32")
        b = np.array([[1.0, np.inf, 3.5]], dtype="float32")
        cmp = mod.VarComparison(name="x", dtype="float32")
        mod._accumulate_float(cmp, a, b)
        # The +inf pair is excluded, not folded in as inf-inf=NaN.
        assert np.isfinite(cmp.max_abs_diff) and abs(cmp.max_abs_diff - 0.5) < 1e-6
        assert np.isfinite(cmp.mean_abs_diff)
        assert cmp.n_finite_pairs == 2  # the two finite pairs only
