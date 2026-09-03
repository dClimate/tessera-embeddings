"""What `run_plain` does with a config, and what the CLI does with its flags.

The runner's own logic had no coverage: the existing tests mock `_run_ingest` and check whether
S1 is called once or twice, which is a fact about `_active_orbits`, not about the runner. Nothing
exercised config parsing, the staging identity, cleanup, or the argument wiring — and a
`--min-valid-coverage` flag sat parsed-but-never-read for as long as it had existed because of it.

Everything below mocks the domain calls. This file is about the wiring between reading the YAML
and invoking them; the domain functions have their own tests and the parity suite.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.ingest.roi_processing import DEFAULT_MIN_VALID_COVERAGE
from tessera_embeddings.orchestration.runners import plain

_MOD = "tessera_embeddings.orchestration.runners.plain"


def _config(tmp_path: Path, **overrides) -> Path:
    """A minimal but complete plain-runner config on disk."""
    cfg = {
        "paths": {"inputs": f"file://{tmp_path}/in", "outputs": f"file://{tmp_path}/out"},
        "roi": {"name": "test_roi", "resolution": 10.0, "chunk_size": 2048},
        "time_range": {"start": "2024-07-01", "end": "2024-08-01"},
        "time_window_end": "August 2024",
        "s1_orbit": "ascending",
        "n_workers": 2,
    }
    cfg.update(overrides)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


class TestTheCoverageThresholdReachesIngest:
    """The defect this file exists for: a flag parsed and then dropped."""

    def test_the_cli_flag_reaches_the_ingest_call(self, tmp_path):
        """``--min-valid-coverage`` must arrive at the S2 domain function.

        It was declared in the parser, documented in the help text, given a default, and never
        read: ``run_plain`` did not take the parameter. A user passing it got the default
        threshold and no warning that their argument had been ignored.
        """
        with patch(f"{_MOD}._run_ingest") as run_ingest:
            plain.run_plain(_config(tmp_path), skip_inference=True, min_valid_coverage=42.5)
        assert run_ingest.call_args.kwargs["min_valid_coverage"] == 42.5

    def test_the_config_key_is_used_when_the_flag_is_absent(self, tmp_path):
        with patch(f"{_MOD}._run_ingest") as run_ingest:
            plain.run_plain(_config(tmp_path, min_valid_coverage=33.0), skip_inference=True)
        assert run_ingest.call_args.kwargs["min_valid_coverage"] == 33.0

    def test_the_flag_beats_the_config_key(self, tmp_path):
        with patch(f"{_MOD}._run_ingest") as run_ingest:
            plain.run_plain(_config(tmp_path, min_valid_coverage=33.0), skip_inference=True, min_valid_coverage=9.0)
        assert run_ingest.call_args.kwargs["min_valid_coverage"] == 9.0

    def test_neither_falls_back_to_the_domain_default(self, tmp_path):
        with patch(f"{_MOD}._run_ingest") as run_ingest:
            plain.run_plain(_config(tmp_path), skip_inference=True)
        assert run_ingest.call_args.kwargs["min_valid_coverage"] == DEFAULT_MIN_VALID_COVERAGE

    def test_main_passes_the_parsed_flag_through(self, tmp_path):
        """Covers the CLI itself — the layer where the flag was being dropped."""
        cfg = _config(tmp_path)
        with patch(f"{_MOD}.run_plain") as run_plain:
            plain.main([str(cfg), "--skip-inference", "--min-valid-coverage", "17.5"])
        assert run_plain.call_args.kwargs["min_valid_coverage"] == 17.5
        assert run_plain.call_args.kwargs["skip_inference"] is True

    def test_main_passes_none_when_the_flag_is_omitted(self, tmp_path):
        """None, not the default — otherwise the CLI would always mask the config key."""
        with patch(f"{_MOD}.run_plain") as run_plain:
            plain.main([str(_config(tmp_path))])
        assert run_plain.call_args.kwargs["min_valid_coverage"] is None


class TestTheStagingIdentityRepeats:
    """A single run must be able to resume; the campaign's fingerprint is not wanted here."""

    @staticmethod
    def _cfg(**over):
        base = dict(
            s1_orbit="ascending",
            time_window=parse_time_window("August 2024"),
            checkpoint_path="/models/ckpt.pt",
        )
        base.update(over)
        return MagicMock(**base)

    def test_the_same_run_gets_the_same_prefix(self):
        """This is what makes an interrupted run resume instead of re-inferring."""
        a = plain._staging_run_id(roi_name="roi", config=self._cfg())
        b = plain._staging_run_id(roi_name="roi", config=self._cfg())
        assert a == b

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("time_window", parse_time_window("July 2024")),
            ("s1_orbit", "descending"),
            ("checkpoint_path", "/models/other.pt"),
        ],
    )
    def test_a_different_product_gets_a_different_prefix(self, field, value):
        """Each term changes the OUTPUT, so reusing tiles across it would publish a blend."""
        base = plain._staging_run_id(roi_name="roi", config=self._cfg())
        assert plain._staging_run_id(roi_name="roi", config=self._cfg(**{field: value})) != base

    def test_a_different_roi_gets_a_different_prefix(self):
        base = plain._staging_run_id(roi_name="roi", config=self._cfg())
        assert plain._staging_run_id(roi_name="other", config=self._cfg()) != base

    def test_the_identity_is_not_the_campaign_fingerprint(self):
        """It must NOT depend on code identity — that is the safeguard we deliberately omit.

        The campaign mixes `inference_code_identity()` in so a mid-campaign code change cannot
        blend two versions into one write-once zone-year. Here it would mean re-inferring a whole
        ROI because a comment moved, which is the wrong default for a minutes-long single run.
        """
        with patch("tessera_embeddings.config.inference.inference_code_identity") as identity:
            plain._staging_run_id(roi_name="roi", config=self._cfg())
        identity.assert_not_called()


def _apply(stack, patches):
    return [stack.enter_context(p) for p in patches]


class TestStagingLifecycle:
    """Cleanup on success, retention on failure — the pair that makes resume work."""

    def _run(self, tmp_path, *, inference_results, expect_raise):
        patches = [
            patch(f"{_MOD}.resolve_s1_orbit", return_value="ascending"),
            # a real window label: `_staging_run_id` joins it into the identity string, so a
            # bare MagicMock here fails inside the code under test rather than in the setup
            patch(
                f"{_MOD}.build_inference_config",
                return_value=MagicMock(
                    time_window=parse_time_window("August 2024"),
                    s1_orbit="ascending",
                    checkpoint_path="/models/ckpt.pt",
                    chunk_size=2048,
                ),
            ),
            patch(f"{_MOD}.enumerate_mosaic_chunks", return_value=([MagicMock()], 2048, 2048)),
            patch(f"{_MOD}.filter_chunks_by_roi_mask", return_value=[MagicMock()]),
            patch(f"{_MOD}.ray_cluster"),
            patch(f"{_MOD}.run_inference", return_value=inference_results),
            patch(f"{_MOD}.checkpoint_to_version", return_value="v1.1"),
            patch(f"{_MOD}.read_upstream_manifests", return_value=[]),
            patch(f"{_MOD}.EmbeddingManifest"),
            patch(f"{_MOD}.ZarrWriter"),
            patch(f"{_MOD}.delete_prefix"),
        ]
        with contextlib.ExitStack() as stack:
            applied = _apply(stack, patches)
            delete = applied[-1]
            kwargs = dict(
                roi_path=f"file://{tmp_path}/roi.zarr",
                roi_name="test_roi",
                paths=MagicMock(inputs=f"file://{tmp_path}/in", outputs=f"file://{tmp_path}/out"),
                time_window_end="August 2024",
                s1_orbit="ascending",
                checkpoint_dir=None,
                checkpoint_url="/models/ckpt.pt",
                num_gpus=0,
                log=logging.getLogger("test"),
            )
            if expect_raise:
                with pytest.raises(RuntimeError, match="failed during inference"):
                    plain._run_inference_and_assemble(**kwargs)
            else:
                plain._run_inference_and_assemble(**kwargs)
            return delete

    def test_a_successful_run_removes_its_staging_prefix(self, tmp_path):
        """On the quickstart the staged prefix is larger than the product it produced."""
        delete = self._run(tmp_path, inference_results=[{"status": "success"}], expect_raise=False)
        delete.assert_called_once()
        assert "/staging/" in delete.call_args.args[0]

    def test_a_failed_run_keeps_its_staging_prefix(self, tmp_path):
        """The retention that makes the resume possible: a failed run's tiles must survive."""
        delete = self._run(tmp_path, inference_results=[{"status": "failed"}], expect_raise=True)
        delete.assert_not_called()

    def test_a_cleanup_failure_does_not_fail_a_finished_pipeline(self, tmp_path):
        """The embeddings are committed by then; housekeeping must not turn success into failure."""
        patches = [
            patch(f"{_MOD}.resolve_s1_orbit", return_value="ascending"),
            # a real window label: `_staging_run_id` joins it into the identity string, so a
            # bare MagicMock here fails inside the code under test rather than in the setup
            patch(
                f"{_MOD}.build_inference_config",
                return_value=MagicMock(
                    time_window=parse_time_window("August 2024"),
                    s1_orbit="ascending",
                    checkpoint_path="/models/ckpt.pt",
                    chunk_size=2048,
                ),
            ),
            patch(f"{_MOD}.enumerate_mosaic_chunks", return_value=([MagicMock()], 2048, 2048)),
            patch(f"{_MOD}.filter_chunks_by_roi_mask", return_value=[MagicMock()]),
            patch(f"{_MOD}.ray_cluster"),
            patch(f"{_MOD}.run_inference", return_value=[{"status": "success"}]),
            patch(f"{_MOD}.checkpoint_to_version", return_value="v1.1"),
            patch(f"{_MOD}.read_upstream_manifests", return_value=[]),
            patch(f"{_MOD}.EmbeddingManifest"),
            patch(f"{_MOD}.ZarrWriter"),
            patch(f"{_MOD}.delete_prefix", side_effect=OSError("bucket said no")),
        ]
        with contextlib.ExitStack() as stack:
            _apply(stack, patches)
            summary = plain._run_inference_and_assemble(
                roi_path=f"file://{tmp_path}/roi.zarr",
                roi_name="test_roi",
                paths=MagicMock(inputs=f"file://{tmp_path}/in", outputs=f"file://{tmp_path}/out"),
                time_window_end="August 2024",
                s1_orbit="ascending",
                checkpoint_dir=None,
                checkpoint_url="/models/ckpt.pt",
                num_gpus=0,
                log=logging.getLogger("test"),
            )
        assert summary["roi_name"] == "test_roi"


class TestDeviceResolution:
    """`device` is the one config key with a runtime probe behind it."""

    def test_cpu_is_zero_gpus(self):
        assert plain._resolve_num_gpus("cpu") == 0

    def test_an_unknown_device_is_refused(self):
        with pytest.raises(ValueError, match="device must be"):
            plain._resolve_num_gpus("tpu")

    def test_cuda_without_a_device_is_refused_rather_than_silently_downgraded(self):
        with patch("torch.cuda.is_available", return_value=False), pytest.raises(RuntimeError, match="no CUDA device"):
            plain._resolve_num_gpus("cuda")

    def test_auto_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert plain._resolve_num_gpus("auto") == 0
