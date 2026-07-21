"""fill_zones_sequential_flow: triage, ordering, and the pre-Ray gates.

Exercises the flow body via ``.fn`` with every external touchpoint mocked
(mirroring test_fill_zone_year_flow), so no Prefect engine, Ray cluster, or
S3 runs. The load-bearing behaviors: cheap cells (retag / all-ocean) are
settled without a cluster, live cells are ordered largest-first with clamped
actor requests, the model guard fires before Ray, and the shared cluster is
provisioned once with the idle-timeout override.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import tessera_embeddings.orchestration.prefect.flows.fill_zones_sequential as mod
from tessera_embeddings.config.paths import BucketPaths

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


@pytest.fixture()
def wired(monkeypatch):
    """Mock every external touchpoint; return a record of what the flow did."""
    rec: dict = {"no_cluster_fills": [], "ray_kwargs": None, "seq_kwargs": None, "order": []}
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-seq-flow"))
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.credentials.iam_icechunk_credentials", object(), raising=False
    )
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.ray.make_instance_terminator",
        lambda log: lambda iid: None,
        raising=False,
    )

    @contextmanager
    def fake_ray_cluster(log, **kwargs):
        rec["ray_kwargs"] = kwargs
        rec["order"].append("ray_up")
        yield None  # no resolved YAML → the hook state stays unset

    monkeypatch.setattr("tessera_embeddings.providers.aws.ray.ray_cluster", fake_ray_cluster, raising=False)

    # Default triage: nothing complete, everything on-axis, 10 live tiles.
    monkeypatch.setattr(mod, "zone_year_complete", lambda *a, **k: False)
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: True)
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda *a, **k: 10)

    def fake_no_cluster_fill(**kwargs):
        rec["no_cluster_fills"].append(kwargs)
        return {"zone": kwargs["zone"], "run_id": kwargs["run_id"]}

    monkeypatch.setattr(mod, "fill_zone_year", fake_no_cluster_fill)
    monkeypatch.setattr(mod, "_assert_seeded_model_matches", lambda *a, **k: rec["order"].append("model_guard"))
    monkeypatch.setattr(mod, "build_inference_config", lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(mod, "resolve_s1_orbit", lambda *a, **k: "both")
    monkeypatch.setattr(mod, "check_time_window_coverage", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_staging_run_id", lambda z, y, **k: f"{z}-{y}-fingerprint")

    def fake_sequential(**kwargs):
        rec["seq_kwargs"] = kwargs
        return {"cells": len(kwargs["cells"]), "succeeded": len(kwargs["cells"]), "failed": 0}

    monkeypatch.setattr(mod, "fill_zones_sequential", fake_sequential)
    return rec


def _run(**overrides):
    kwargs = {"zones": ["33N"], "year": 2025, "paths": _PATHS, "ami_ssm_name": "ami"}
    kwargs.update(overrides)
    return mod.fill_zones_sequential_flow.fn(**kwargs)


def test_retag_and_ocean_cells_never_touch_the_cluster(wired, monkeypatch):
    monkeypatch.setattr(mod, "zone_year_complete", lambda store, zone, year, **k: zone == "01N")
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda mask, zone, **k: 0)  # the rest: all-ocean
    result = _run(zones=["01N", "02N"])
    # Both cells settled by the no-cluster fill path with the right run_ids...
    run_ids = {f["zone"]: f["run_id"] for f in wired["no_cluster_fills"]}
    assert run_ids == {"01N": "01N-2025-retag", "02N": "02N-2025-empty"}
    # ...and no cluster was ever provisioned.
    assert wired["ray_kwargs"] is None and result["sequential"] is None
    assert result["retagged"] == 1 and result["empty"] == 1 and result["live"] == 0


def test_off_axis_year_fails_the_run(wired, monkeypatch):
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: False)
    with pytest.raises(ValueError, match="pre-allocated axis"):
        _run()


def test_live_cells_ordered_by_clamped_fleet_desc(wired, monkeypatch):
    counts = {"01N": 3, "02N": 500, "03N": 12}
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda mask, zone, **k: counts[zone])
    _run(zones=["01N", "02N", "03N"], num_actors=20)
    cells = wired["seq_kwargs"]["cells"]
    assert [(c.zone, c.num_actors) for c in cells] == [("02N", 20), ("03N", 12), ("01N", 3)]


def test_model_guard_runs_before_ray(wired):
    _run()
    assert wired["order"] == ["model_guard", "ray_up"]


def test_model_mismatch_prevents_cluster(wired, monkeypatch):
    def guard(*a, **k):
        raise ValueError("seeded for a different encoder")

    monkeypatch.setattr(mod, "_assert_seeded_model_matches", guard)
    with pytest.raises(ValueError, match="different encoder"):
        _run()
    assert wired["ray_kwargs"] is None


def test_cluster_gets_idle_timeout_and_code_params(wired):
    _run(idle_timeout_minutes=15, code_bucket="cb", code_suffix="-branch")
    assert wired["ray_kwargs"]["idle_timeout_minutes"] == 15
    assert wired["ray_kwargs"]["code_bucket"] == "cb"
    assert wired["ray_kwargs"]["code_suffix"] == "-branch"


def test_ingest_false_passes_no_inputs_adapter(wired):
    _run(ingest=False)
    assert wired["seq_kwargs"]["inputs"] is None


def test_ingest_true_wires_the_deployment_adapter(wired):
    _run(ingest=True, look_ahead=3)
    inputs = wired["seq_kwargs"]["inputs"]
    assert isinstance(inputs, mod._DeploymentCellInputs)
    assert wired["seq_kwargs"]["look_ahead"] == 3
    inputs.shutdown()


def test_prepare_fingerprints_after_resolving_orbit(wired):
    _run()
    prepare = wired["seq_kwargs"]["prepare"]
    prep = prepare(SimpleNamespace(zone="33N", year=2025, num_actors=5))
    assert prep.run_id == "33N-2025-fingerprint"
    assert prep.mosaic_base == "s3://in/mosaics/33N/2025"
    assert prep.staging_base == "s3://out/staging/33N/2025"
    # The per-cell config carries the resolved orbit and the shard chunk size.
    assert prep.config.s1_orbit == "both"


def test_duplicate_and_lowercase_zones_are_canonicalized(wired):
    _run(zones=["33n", "33N"])
    assert [c.zone for c in wired["seq_kwargs"]["cells"]] == ["33N"]


def test_stream_contract_wired(wired):
    """The runner receives the stream contract: plan/session/infer_single +
    the session orbit the feeder gates zone deferral on.
    """
    _run(s1_orbit="both")
    kw = wired["seq_kwargs"]
    assert kw["session_s1_orbit"] == "both"
    assert callable(kw["plan"]) and callable(kw["session"]) and callable(kw["infer_single"])
    assert "infer" not in kw and "max_consecutive_failures" not in kw
