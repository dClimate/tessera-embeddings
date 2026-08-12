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

from tessera_embeddings.orchestration.prefect import _fleet_gate as fleet_gate

import tessera_embeddings.orchestration.prefect.flows.fill_zone_year as mod
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.conventions import expected_model_url

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


@pytest.fixture()
def wired(monkeypatch):
    """Stub every touchpoint the flow reaches BEFORE the thing under test.

    Defaults put the flow on the live-cell path — on-axis, not complete, has land —
    because that is the only path where the pre-Ray gates and the Ray branch run at
    all. Both gates pass by default; a test that is about one of them re-patches it.
    Nothing here provisions Ray, so a test that wants the Ray branch must supply its
    own ``ray_cluster``.
    """
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-fill"))
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.credentials.iam_icechunk_credentials", object(), raising=False
    )
    monkeypatch.setattr(mod, "zone_year_complete", lambda *a, **k: False)
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: True)
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "resolve_s1_orbit", lambda *a, **k: "both")
    monkeypatch.setattr(mod, "build_inference_config", lambda **k: SimpleNamespace(time_window="W"))
    monkeypatch.setattr(mod, "_assert_seeded_model_matches", lambda *a, **k: None)
    monkeypatch.setattr(mod, "check_time_window_coverage", lambda *a, **k: None)
    return monkeypatch


def _stop_before_ray(monkeypatch):
    """Make the coverage gate raise, so a test can assert on pre-Ray behaviour."""

    def _raise(*a, **k):
        raise InsufficientCoverageError("stop before Ray")

    monkeypatch.setattr(mod, "check_time_window_coverage", _raise)


@pytest.mark.parametrize("allow_partial", [False, True])
def test_coverage_gate_fails_before_ray(wired, allow_partial):
    """A coverage-check failure raises before Ray, and threads the flag + creds."""
    ray_calls: list = []
    wired.setattr(
        "tessera_embeddings.providers.aws.ray.ray_cluster", lambda *a, **k: ray_calls.append((a, k)), raising=False
    )
    captured: dict = {}

    def _coverage(mosaic_base, window, *, s1_orbit, skip_coverage_check, get_credentials, s3_region=None):
        captured.update(mosaic_base=mosaic_base, skip=skip_coverage_check, creds=get_credentials)
        raise InsufficientCoverageError("reflectance store missing months")

    wired.setattr(mod, "check_time_window_coverage", _coverage)

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


@pytest.mark.parametrize("flag", [False, True])
def test_allow_s2_only_reaches_inference_config(wired, flag):
    """The flow's allow_s2_only lands in build_inference_config — the single
    chokepoint through which it reaches the dataset's per-pixel gate. The
    zone-level gates are NOT relaxed by it (the coverage gate still raises here).
    """
    captured: dict = {}
    wired.setattr(mod, "build_inference_config", lambda **kw: captured.update(kw) or SimpleNamespace(time_window="W"))
    _stop_before_ray(wired)

    with pytest.raises(InsufficientCoverageError):
        mod.fill_zone_year_flow.fn(zone="33N", year=2025, paths=_PATHS, ami_ssm_name="ami", allow_s2_only=flag)
    assert captured["allow_s2_only"] is flag


_AWS_CKPT = "tessera_v1_1_aws_encoder.pt"
_MPC_CKPT = "tessera_v1_1_mpc_encoder.pt"


def _guard(monkeypatch, attrs: dict, *, build_checkpoint: str = "ckpt.pt", allow_mismatch: bool = False) -> None:
    """Run the seeded-model guard against a store root carrying ``attrs``."""
    monkeypatch.setattr(mod, "open_store_as_zarr_group", lambda *a, **k: SimpleNamespace(attrs=attrs))
    mod._assert_seeded_model_matches(
        "s3://in/store",
        build_checkpoint=build_checkpoint,
        allow_model_mismatch=allow_mismatch,
        get_credentials=None,
    )


@pytest.mark.parametrize(
    "attrs,build_checkpoint,match",
    [
        # A store seeded for a different ENCODER than this build embeds with. Filling
        # it would mix encoders under one store and tag the result permanently.
        pytest.param({"geoemb:model": "https://x/OLD"}, "ckpt.pt", "was seeded for encoder", id="encoder"),
        # Same encoder URL, different concrete CHECKPOINT. geoemb:model versions only
        # the encoder, so the aws/mpc v1.1 checkpoints share a URL and would otherwise
        # be mixed silently; checkpoint_id is what tells them apart.
        pytest.param(
            {"geoemb:model": expected_model_url(), "checkpoint_id": _MPC_CKPT},
            _AWS_CKPT,
            "seeded for checkpoint",
            id="checkpoint",
        ),
    ],
)
def test_model_guard_rejects_a_mismatched_store(monkeypatch, attrs, build_checkpoint, match):
    with pytest.raises(ValueError, match=match):
        _guard(monkeypatch, attrs, build_checkpoint=build_checkpoint)


@pytest.mark.parametrize(
    "attrs,build_checkpoint,allow_mismatch,why",
    [
        ({"geoemb:model": expected_model_url()}, "ckpt.pt", False, "encoder matches"),
        ({}, "ckpt.pt", False, "store advertises no model (legacy/unseeded root)"),
        ({"geoemb:model": expected_model_url(), "checkpoint_id": _MPC_CKPT}, _MPC_CKPT, False, "checkpoint matches"),
        ({"geoemb:model": "https://x/OLD"}, "ckpt.pt", True, "encoder override"),
        (
            {"geoemb:model": expected_model_url(), "checkpoint_id": _MPC_CKPT},
            _AWS_CKPT,
            True,
            "checkpoint override",
        ),
    ],
    ids=["encoder-match", "no-attr", "checkpoint-match", "encoder-override", "checkpoint-override"],
)
def test_model_guard_admits_a_compatible_store(monkeypatch, attrs, build_checkpoint, allow_mismatch, why):
    """Each case must pass the guard — the assertion is that it does not raise."""
    _guard(monkeypatch, attrs, build_checkpoint=build_checkpoint, allow_mismatch=allow_mismatch)


def _fake_ray_cluster(captured: dict | None = None):
    """A ``ray_cluster`` stand-in that provisions nothing.

    Yields ``None`` rather than a resolved YAML path, which is what makes the
    flow's cluster-name probe skip — the tests here are about what the Ray branch
    WIRES, not about the cluster.
    """

    @contextmanager
    def _cm(*a, **k):
        if captured is not None:
            captured.update(k)
        yield None

    return _cm


def test_ray_path_wires_actor_terminator(wired):
    """The Ray branch wires make_instance_terminator into on_actor_retire, so idle
    GPU nodes are terminated mid-fill instead of held until the cluster teardown.
    """
    sentinel = object()
    wired.setattr("tessera_embeddings.providers.aws.ray.make_instance_terminator", lambda **k: sentinel, raising=False)
    wired.setattr("tessera_embeddings.providers.aws.ray.ray_cluster", _fake_ray_cluster(), raising=False)
    captured: dict = {}
    wired.setattr(mod, "fill_zone_year", lambda **kw: captured.update(kw) or {"tag": "t"})

    mod.fill_zone_year_flow.fn(zone="33n", year=2025, paths=_PATHS, ami_ssm_name="ami")
    assert captured["on_actor_retire"] is sentinel


def test_pinned_ami_id_reaches_ray_cluster(wired):
    """A campaign-pinned ami_id is threaded into ray_cluster so the fleet boots the
    exact image the staging fingerprint recorded, not whatever ami_ssm_name resolves
    to at provisioning time.
    """
    wired.setattr("tessera_embeddings.providers.aws.ray.make_instance_terminator", lambda **k: object(), raising=False)
    wired.setattr(mod, "fill_zone_year", lambda **kw: {"tag": "t"})
    captured: dict = {}
    wired.setattr("tessera_embeddings.providers.aws.ray.ray_cluster", _fake_ray_cluster(captured), raising=False)

    mod.fill_zone_year_flow.fn(zone="33N", year=2025, paths=_PATHS, ami_ssm_name="ami", ami_id="ami-pinned-01")
    assert captured["ami_id"] == "ami-pinned-01"


def test_staging_base_scoped_to_zone_year(wired):
    """The runner is handed a (zone, year)-scoped staging_base so a reused run_id
    can't cross-contaminate another cell's staged tiles.
    """
    wired.setattr(mod, "zone_year_complete", lambda *a, **k: True)  # retag-only: no cluster, no gate
    wired.setattr("tessera_embeddings.providers.aws.ray.ray_cluster", lambda *a, **k: None, raising=False)
    captured: dict = {}
    wired.setattr(mod, "fill_zone_year", lambda **kw: captured.update(kw) or {"tag": "t"})

    mod.fill_zone_year_flow.fn(zone="33n", year=2025, paths=_PATHS, ami_ssm_name="ami")
    assert captured["staging_base"] == "s3://out/staging/33N/2025"


class TestPerCellValidation:
    """The landed cell is handed to its validation deployment on BOTH return paths.

    Two paths, one dispatch each, and they are easy to leave half-wired: the flow returns
    from the no-cluster branch early, so a dispatch added only after the Ray branch would
    silently skip every retag-only recovery — the cells most likely never to have been
    validated at all.
    """

    @pytest.fixture()
    def dispatched(self, wired):
        calls: list[dict] = []
        wired.setattr(
            mod,
            "dispatch_cell_validation",
            lambda deployment, **kw: calls.append({"deployment": deployment, **kw}) or "validation-run",
        )
        wired.setattr(mod, "fill_zone_year", lambda **kw: {"tag": "zone-33N-2025"})
        return calls

    def test_the_ray_path_validates_the_cell(self, wired, dispatched):
        wired.setattr(
            "tessera_embeddings.providers.aws.ray.make_instance_terminator", lambda **k: object(), raising=False
        )
        wired.setattr("tessera_embeddings.providers.aws.ray.ray_cluster", _fake_ray_cluster(), raising=False)
        mod.fill_zone_year_flow.fn(zone="33n", year=2025, paths=_PATHS, ami_ssm_name="ami", validation_deployment="v/v")
        (call,) = dispatched
        assert call["deployment"] == "v/v"
        assert (call["zone"], call["year"]) == ("33N", 2025)
        assert call["parameters"]["store_name"] == "tessera"

    def test_the_no_cluster_path_validates_a_retagged_cell(self, wired, dispatched):
        wired.setattr(mod, "zone_year_complete", lambda *a, **k: True)
        mod.fill_zone_year_flow.fn(zone="33n", year=2025, paths=_PATHS, ami_ssm_name="ami", validation_deployment="v/v")
        assert [(c["zone"], c["year"]) for c in dispatched] == [("33N", 2025)]

    def test_the_summary_reaches_the_caller_unchanged(self, wired, dispatched):
        """The dispatch is a side effect; the fill's own summary is the return value."""
        wired.setattr(mod, "zone_year_complete", lambda *a, **k: True)
        summary = mod.fill_zone_year_flow.fn(
            zone="33N", year=2025, paths=_PATHS, ami_ssm_name="ami", validation_deployment="v/v"
        )
        assert summary == {"tag": "zone-33N-2025"}


def test_commit_gate_is_thread_safe(monkeypatch):
    """The chained fill shares ONE _PrefectCommitGate between its feeder thread
    (terminal plans commit inside plan()) and its trailing-assembly thread. The
    gate must pair each exit with the SAME thread's entered context — an
    instance slot let a concurrent enter overwrite the other thread's context
    and release the wrong Prefect concurrency slot.
    """
    import threading

    class _RecordingCM:
        def __init__(self):
            self.entered = 0
            self.exited = 0

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *a):
            self.exited += 1
            return False

    cms: list[_RecordingCM] = []

    def fake_concurrency(name, occupy=1, strict=True, **kw):
        cm = _RecordingCM()
        cms.append(cm)
        return cm

    monkeypatch.setattr(fleet_gate, "concurrency", fake_concurrency)
    gate = mod._PrefectCommitGate("limit")
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker():
        try:
            for _ in range(50):
                gate.__enter__()
                barrier.wait(timeout=10)  # force overlapping occupancy each round
                gate.__exit__(None, None, None)
        except BaseException as exc:  # surface thread failures to the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    # Every context entered exactly once and exited exactly once — no context
    # was double-exited (stolen by the other thread) or leaked (never exited).
    assert len(cms) == 100  # 2 threads x 50 rounds
    assert all(cm.entered == 1 and cm.exited == 1 for cm in cms)


def test_commit_gate_is_reentrant_within_a_thread(monkeypatch):
    entered: list[str] = []

    class _CM:
        def __init__(self, tag):
            self.tag = tag

        def __enter__(self):
            entered.append(f"enter:{self.tag}")

        def __exit__(self, *a):
            entered.append(f"exit:{self.tag}")
            return False

    tags = iter(["outer", "inner"])
    monkeypatch.setattr(fleet_gate, "concurrency", lambda name, occupy=1, strict=True, **kw: _CM(next(tags)))
    gate = mod._PrefectCommitGate("limit")
    with gate, gate:
        pass
    # LIFO pairing: the inner exit releases the inner context, not the outer.
    assert entered == ["enter:outer", "enter:inner", "exit:inner", "exit:outer"]
