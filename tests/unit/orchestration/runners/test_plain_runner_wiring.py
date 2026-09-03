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


def _run_id(roi_name: str = "roi", *, upstream: dict | None = None, **config_over) -> str:
    """`_staging_run_id` with the boilerplate folded away."""
    return plain._staging_run_id(
        roi_name=roi_name,
        config=_inference_config(**config_over),
        upstream={"reflectance": {"roi_manifest_hash": "roi-1"}} if upstream is None else upstream,
    )


def _inference_config(**over) -> MagicMock:
    """A stand-in ``InferenceConfig`` with a REAL time window.

    ``_staging_run_id`` joins ``window_end_label`` into the identity string, so a bare MagicMock
    fails inside the code under test rather than in the setup.
    """
    base = dict(
        time_window=parse_time_window("August 2024"),
        s1_orbit="ascending",
        checkpoint_path="/models/ckpt.pt",
        allow_s2_only=False,
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

    @pytest.mark.parametrize("bad", [0.0, -1.0, 100.1, 1000.0], ids=["zero", "negative", "just-over", "way-over"])
    def test_an_out_of_range_threshold_is_refused(self, tmp_path, bad):
        """The flow path validates through pydantic (gt=0, le=100); this path did not.

        The domain test is `coverage >= min_valid_coverage`, so 0 admits a date with no valid
        pixels and 101 rejects every date -- both silently, as a config that looks like it worked.
        """
        with pytest.raises(ValueError, match=r"min_valid_coverage must be in \(0, 100\]"):
            plain.run_plain(_config(tmp_path), skip_inference=True, min_valid_coverage=bad)

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
        assert _run_id() == _run_id()

    @pytest.mark.parametrize(
        ("kwargs", "why"),
        [
            ({"time_window": parse_time_window("July 2024")}, "a different window is a different product"),
            ({"s1_orbit": "descending"}, "a different orbit reads different mosaics"),
            ({"checkpoint_path": "/models/other.pt"}, "a different model is a different embedding"),
            ({"roi_name": "other_roi"}, "a different ROI is a different footprint"),
            ({"allow_s2_only": True}, "decides whether radar-free pixels are embedded at all"),
            (
                {"upstream": {"reflectance": {"roi_manifest_hash": "roi-2"}}},
                "a re-rasterised ROI is a different footprint",
            ),
            (
                {"upstream": {"reflectance": {"min_valid_coverage": 50.0}}},
                "a re-ingest at another threshold admitted different dates",
            ),
        ],
        ids=["window", "orbit", "checkpoint", "roi", "allow_s2_only", "roi_hash", "coverage"],
    )
    def test_a_different_product_gets_a_different_prefix(self, kwargs, why):
        """`run_inference` resumes purely on this id, so a missing term is a silently mixed store."""
        roi_name = kwargs.pop("roi_name", "roi")
        assert _run_id(roi_name, **kwargs) != _run_id(), why

    def test_manifest_key_order_does_not_change_the_identity(self):
        """The upstream digest must be a fact about content, not about dict construction order."""
        one = _run_id(upstream={"reflectance": {"a": 1, "b": 2}, "sar_ascending": {"c": 3}})
        two = _run_id(upstream={"sar_ascending": {"c": 3}, "reflectance": {"b": 2, "a": 1}})
        assert one == two

    def test_ingest_code_identity_is_excluded_from_the_digest(self):
        """An INGEST source change must not re-infer every ROI.

        The upstream manifests carry `ingest_code_identity`, so digesting them wholesale would
        reintroduce the campaign's safeguard by the side door: a comment edit in the ingest path
        moves that value, which would move this identity, which would re-infer the whole ROI.
        Redoing work has to be deliberate, not the normal case.
        """
        base = _run_id(upstream={"reflectance": {"roi_manifest_hash": "roi-1"}})
        moved_code = _run_id(
            upstream={"reflectance": {"roi_manifest_hash": "roi-1", "ingest_code_identity": "ingcode-DIFFERENT"}}
        )
        assert moved_code == base

    def test_the_identity_is_not_the_campaign_fingerprint(self):
        """It must NOT depend on code identity — the safeguard we deliberately omit.

        The campaign mixes ``inference_code_identity()`` in so a mid-campaign code change cannot
        blend two versions into one write-once zone-year. Here it would mean re-inferring a whole
        ROI because a comment moved, which is the wrong default for a minutes-long single run.
        This test exists so a future change to "harden" it fails loudly instead of silently making
        every single-ROI run start from scratch.
        """
        with patch("tessera_embeddings.config.inference.inference_code_identity") as identity:
            _run_id()
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
            patch(f"{_MOD}.read_upstream_manifests", return_value={"reflectance": {"x": 1}}),
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
        # Without strict, delete_prefix swallows an unverified delete and a non-empty prefix and
        # returns, so the outer except never fires and we log success over surviving staging --
        # which the deterministic run_id would then resume onto.
        assert delete.call_args.kwargs["strict"] is True
        # all_versions would make s5cmd a hard requirement of the orchestrator-free path: on s3
        # the fsspec fallback cannot honour it, so strict turns a missing binary into a leak.
        assert delete.call_args.kwargs["all_versions"] is False

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
