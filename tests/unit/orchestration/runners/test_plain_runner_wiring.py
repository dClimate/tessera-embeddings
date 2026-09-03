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


def _inference_config(**over) -> MagicMock:
    """A stand-in ``InferenceConfig`` with a REAL time window.

    ``_staging_run_id`` joins ``window_end_label`` into the identity string, so a bare MagicMock
    fails inside the code under test rather than in the setup.
    """
    base = dict(
        time_window=parse_time_window("August 2024"),
        s1_orbit="ascending",
        checkpoint_path="/models/ckpt.pt",
        chunk_size=2048,
    )
    base.update(over)
    return MagicMock(**base)


class TestTheCoverageThresholdReachesIngest:
    """The defect this file exists for: a flag parsed and then dropped.

    ``--min-valid-coverage`` was declared in the parser, documented in the help text and given a
    default — and ``run_plain`` did not accept the parameter, so it was never read. A user passing
    it got the default threshold with no warning that their argument had been ignored.
    """

    @pytest.mark.parametrize(
        ("flag", "config_key", "expected"),
        [
            (42.5, None, 42.5),
            (None, 33.0, 33.0),
            (9.0, 33.0, 9.0),
            (None, None, DEFAULT_MIN_VALID_COVERAGE),
        ],
        ids=["flag-only", "config-only", "flag-beats-config", "neither-uses-default"],
    )
    def test_the_threshold_precedence(self, tmp_path, flag, config_key, expected):
        """Flag beats config key beats domain default, and all three actually arrive."""
        over = {} if config_key is None else {"min_valid_coverage": config_key}
        with patch(f"{_MOD}._run_ingest") as run_ingest:
            plain.run_plain(_config(tmp_path, **over), skip_inference=True, min_valid_coverage=flag)
        assert run_ingest.call_args.kwargs["min_valid_coverage"] == expected

    @pytest.mark.parametrize(
        ("argv_extra", "expected"),
        [
            (["--min-valid-coverage", "17.5"], 17.5),
            # None rather than the default: the CLI must not mask a config key just by existing.
            ([], None),
        ],
        ids=["flag-given", "flag-omitted"],
    )
    def test_the_cli_hands_over_what_it_parsed(self, tmp_path, argv_extra, expected):
        """Covers ``main`` itself — the layer where the flag was being dropped."""
        with patch(f"{_MOD}.run_plain") as run_plain:
            plain.main([str(_config(tmp_path)), *argv_extra])
        assert run_plain.call_args.kwargs["min_valid_coverage"] == expected


class TestTheStagingIdentityRepeats:
    """A single run must be able to resume; the campaign's fingerprint is not wanted here."""

    def test_the_same_run_gets_the_same_prefix(self):
        """This is what makes an interrupted run resume instead of re-inferring."""
        first = plain._staging_run_id(roi_name="roi", config=_inference_config())
        second = plain._staging_run_id(roi_name="roi", config=_inference_config())
        assert first == second

    @pytest.mark.parametrize(
        ("roi_name", "config_over"),
        [
            ("roi", {"time_window": parse_time_window("July 2024")}),
            ("roi", {"s1_orbit": "descending"}),
            ("roi", {"checkpoint_path": "/models/other.pt"}),
            ("other_roi", {}),
        ],
        ids=["window", "orbit", "checkpoint", "roi"],
    )
    def test_a_different_product_gets_a_different_prefix(self, roi_name, config_over):
        """Every term changes the OUTPUT, so sharing tiles across one would publish a blend."""
        base = plain._staging_run_id(roi_name="roi", config=_inference_config())
        assert plain._staging_run_id(roi_name=roi_name, config=_inference_config(**config_over)) != base

    def test_the_identity_is_not_the_campaign_fingerprint(self):
        """It must NOT depend on code identity — the safeguard we deliberately omit.

        The campaign mixes ``inference_code_identity()`` in so a mid-campaign code change cannot
        blend two versions into one write-once zone-year. Here it would mean re-inferring a whole
        ROI because a comment moved, which is the wrong default for a minutes-long single run.
        This test exists so a future change to "harden" it fails loudly instead of silently making
        every single-ROI run start from scratch.
        """
        with patch("tessera_embeddings.config.inference.inference_code_identity") as identity:
            plain._staging_run_id(roi_name="roi", config=_inference_config())
        identity.assert_not_called()


class TestStagingLifecycle:
    """Cleanup on success, retention on failure — the pair that makes resume work."""

    @staticmethod
    @contextlib.contextmanager
    def _pipeline(*, inference_results, delete_raises: Exception | None = None):
        """Everything between the ROI and the assemble call, neutralised. Yields the delete mock."""
        with (
            patch(f"{_MOD}.resolve_s1_orbit", return_value="ascending"),
            patch(f"{_MOD}.build_inference_config", return_value=_inference_config()),
            patch(f"{_MOD}.enumerate_mosaic_chunks", return_value=([MagicMock()], 2048, 2048)),
            patch(f"{_MOD}.filter_chunks_by_roi_mask", return_value=[MagicMock()]),
            patch(f"{_MOD}.ray_cluster"),
            patch(f"{_MOD}.run_inference", return_value=inference_results),
            patch(f"{_MOD}.checkpoint_to_version", return_value="v1.1"),
            patch(f"{_MOD}.read_upstream_manifests", return_value=[]),
            patch(f"{_MOD}.EmbeddingManifest"),
            patch(f"{_MOD}.ZarrWriter"),
            patch(f"{_MOD}.delete_prefix", side_effect=delete_raises) as delete,
        ):
            yield delete

    @staticmethod
    def _call(tmp_path):
        return plain._run_inference_and_assemble(
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

    def test_a_successful_run_removes_its_staging_prefix(self, tmp_path):
        """On the quickstart the staged prefix is larger than the product it produced."""
        with self._pipeline(inference_results=[{"status": "success"}]) as delete:
            self._call(tmp_path)
        delete.assert_called_once()
        assert "/staging/" in delete.call_args.args[0]

    def test_a_failed_run_keeps_its_staging_prefix(self, tmp_path):
        """The retention that makes the resume possible: a failed run's tiles must survive."""
        with (
            self._pipeline(inference_results=[{"status": "failed"}]) as delete,
            pytest.raises(RuntimeError, match="failed during inference"),
        ):
            self._call(tmp_path)
        delete.assert_not_called()

    def test_a_cleanup_failure_does_not_fail_a_finished_pipeline(self, tmp_path):
        """The embeddings are committed by then; housekeeping must not turn success into failure."""
        with self._pipeline(inference_results=[{"status": "success"}], delete_raises=OSError("bucket said no")):
            summary = self._call(tmp_path)
        assert summary["roi_name"] == "test_roi"
