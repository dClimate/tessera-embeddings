"""fill_zone_year_flow: the pre-Ray temporal-coverage gate.

Exercises the flow body via ``.fn`` with every external touchpoint mocked, so no
Prefect engine, Ray cluster, or S3 runs — the point is that a partial/absent
mosaic fails BEFORE a cluster is provisioned.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import tessera_embeddings.orchestration.prefect.flows.fill_zone_year as mod
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.conventions import expected_model_url

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


@pytest.mark.parametrize("allow_partial", [False, True])
def test_coverage_gate_fails_before_ray(monkeypatch, allow_partial):
    """A coverage-check failure raises before Ray, and threads the flag + creds."""
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-fill"))
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.credentials.iam_icechunk_credentials", object(), raising=False
    )
    ray_calls: list = []
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.ray.ray_cluster",
        lambda *a, **k: ray_calls.append((a, k)),
        raising=False,
    )
    # needs_cluster = on_axis and not complete and has_live -> True
    monkeypatch.setattr(mod, "zone_year_complete", lambda *a, **k: False)
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: True)
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "resolve_s1_orbit", lambda *a, **k: "both")
    monkeypatch.setattr(mod, "build_inference_config", lambda **k: SimpleNamespace(time_window="W"))
    # Model guard also runs pre-Ray; stub it so this test isolates the coverage gate.
    monkeypatch.setattr(mod, "_assert_seeded_model_matches", lambda *a, **k: None)

    captured: dict = {}

    def _coverage(mosaic_base, window, *, s1_orbit, skip_coverage_check, get_credentials):
        captured.update(mosaic_base=mosaic_base, skip=skip_coverage_check, creds=get_credentials)
        raise InsufficientCoverageError("reflectance store missing months")

    monkeypatch.setattr(mod, "check_time_window_coverage", _coverage)

    with pytest.raises(InsufficientCoverageError):
        mod.fill_zone_year_flow.fn(
            zone="33n",  # canonicalized to 33N by the flow
            year=2025,
            paths=_PATHS,
            ami_ssm_name="ami",
            allow_partial_window=allow_partial,
        )

    assert ray_calls == []  # gate blocked BEFORE provisioning Ray
    assert captured["mosaic_base"] == "s3://in/mosaics/33N/2025"  # per-year, canonicalized zone
    assert captured["skip"] is allow_partial  # allow_partial_window threaded through
    assert captured["creds"] is not None  # credential callback threaded through


def test_model_guard_rejects_encoder_mismatch(monkeypatch):
    """A store seeded for a different encoder than this build is rejected — a
    mid-campaign model upgrade would otherwise mix encoders under one store.
    """
    monkeypatch.setattr(
        mod, "open_store_as_zarr_group", lambda *a, **k: SimpleNamespace(attrs={"geoemb:model": "https://x/OLD"})
    )
    with pytest.raises(ValueError, match="was seeded for encoder"):
        mod._assert_seeded_model_matches(
            "s3://in/store", build_checkpoint="ckpt.pt", allow_model_mismatch=False, get_credentials=None
        )


def test_model_guard_allows_match_override_and_missing(monkeypatch):
    """The guard passes when the encoder matches, when the override flag is set,
    and when the store advertises no model at all (legacy/unseeded root).
    """
    monkeypatch.setattr(
        mod, "open_store_as_zarr_group", lambda *a, **k: SimpleNamespace(attrs={"geoemb:model": expected_model_url()})
    )
    mod._assert_seeded_model_matches(
        "s3://in/store", build_checkpoint="ckpt.pt", allow_model_mismatch=False, get_credentials=None
    )  # matches

    monkeypatch.setattr(
        mod, "open_store_as_zarr_group", lambda *a, **k: SimpleNamespace(attrs={"geoemb:model": "https://x/OLD"})
    )
    mod._assert_seeded_model_matches(
        "s3://in/store", build_checkpoint="ckpt.pt", allow_model_mismatch=True, get_credentials=None
    )  # override

    monkeypatch.setattr(mod, "open_store_as_zarr_group", lambda *a, **k: SimpleNamespace(attrs={}))
    mod._assert_seeded_model_matches(
        "s3://in/store", build_checkpoint="ckpt.pt", allow_model_mismatch=False, get_credentials=None
    )  # no attr


def test_model_guard_rejects_checkpoint_mismatch(monkeypatch):
    """Same encoder URL but a different concrete checkpoint (norm source) recorded in
    checkpoint_id is rejected — geoemb:model versions only the encoder, so the
    aws/mpc v1.1 checkpoints would otherwise be silently mixed under one URL.
    """
    monkeypatch.setattr(
        mod,
        "open_store_as_zarr_group",
        lambda *a, **k: SimpleNamespace(
            attrs={"geoemb:model": expected_model_url(), "checkpoint_id": "tessera_v1_1_mpc_encoder.pt"}
        ),
    )
    with pytest.raises(ValueError, match="seeded for checkpoint"):
        mod._assert_seeded_model_matches(
            "s3://in/store",
            build_checkpoint="tessera_v1_1_aws_encoder.pt",
            allow_model_mismatch=False,
            get_credentials=None,
        )
    # A matching checkpoint_id passes; the override tolerates a mismatch.
    mod._assert_seeded_model_matches(
        "s3://in/store",
        build_checkpoint="tessera_v1_1_mpc_encoder.pt",
        allow_model_mismatch=False,
        get_credentials=None,
    )
    mod._assert_seeded_model_matches(
        "s3://in/store", build_checkpoint="tessera_v1_1_aws_encoder.pt", allow_model_mismatch=True, get_credentials=None
    )


def test_ray_path_wires_actor_terminator(monkeypatch):
    """The Ray branch wires make_instance_terminator into on_actor_retire, so idle
    GPU nodes are terminated mid-fill instead of held until the cluster teardown.
    """
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-fill"))
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.credentials.iam_icechunk_credentials", object(), raising=False
    )
    # needs_cluster = on_axis and not complete and has_live -> True
    monkeypatch.setattr(mod, "zone_year_complete", lambda *a, **k: False)
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: True)
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "resolve_s1_orbit", lambda *a, **k: "both")
    monkeypatch.setattr(mod, "build_inference_config", lambda **k: SimpleNamespace(time_window="W"))
    # Pre-Ray gates pass so we reach the terminator wiring + Ray branch.
    monkeypatch.setattr(mod, "_assert_seeded_model_matches", lambda *a, **k: None)
    monkeypatch.setattr(mod, "check_time_window_coverage", lambda *a, **k: None)

    sentinel = object()
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.ray.make_instance_terminator", lambda **k: sentinel, raising=False
    )

    @contextmanager
    def fake_ray_cluster(*a, **k):
        yield None  # no resolved yaml → the cluster-name probe is skipped

    monkeypatch.setattr("tessera_embeddings.providers.aws.ray.ray_cluster", fake_ray_cluster, raising=False)

    captured: dict = {}
    monkeypatch.setattr(mod, "fill_zone_year", lambda **kw: captured.update(kw) or {"tag": "t"})

    mod.fill_zone_year_flow.fn(zone="33n", year=2025, paths=_PATHS, ami_ssm_name="ami")
    assert captured["on_actor_retire"] is sentinel


def test_staging_base_scoped_to_zone_year(monkeypatch):
    """The runner is handed a (zone, year)-scoped staging_base so a reused run_id
    can't cross-contaminate another cell's staged tiles.
    """
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-fill"))
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.credentials.iam_icechunk_credentials", object(), raising=False
    )
    monkeypatch.setattr("tessera_embeddings.providers.aws.ray.ray_cluster", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(mod, "zone_year_complete", lambda *a, **k: True)  # retag-only → no cluster, no gate
    monkeypatch.setattr(mod, "build_inference_config", lambda **k: SimpleNamespace(time_window="W"))
    captured: dict = {}
    monkeypatch.setattr(mod, "fill_zone_year", lambda **kw: captured.update(kw) or {"tag": "t"})

    mod.fill_zone_year_flow.fn(zone="33n", year=2025, paths=_PATHS, ami_ssm_name="ami")
    assert captured["staging_base"] == "s3://out/staging/33N/2025"
