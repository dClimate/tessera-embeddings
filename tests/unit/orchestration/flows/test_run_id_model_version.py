"""A staging run must not be resumed under the other model version.

Scoped deliberately narrowly. The hazard exists *because* this PR introduces a
second model version: staged chunks are (H, W, 128) int8 whichever student wrote
them, `model_version` defaults to v1.1, so a resume that omits it would finish a
v2 run with the v1.1 encoder and stamp the store v1.1.

The mechanism mirrors `main`'s `S2_ONLY_RUN_PREFIX`, which encodes the S2-only
mode in the run_id for the same reason — one function, so the two reconcile
trivially when this branch merges. General resume/staging hardening (concurrent
claims, unclaimed legacy staging, ROI and time-window identity) is out of scope
here and belongs with the failure-case work.
"""

from __future__ import annotations

import logging

import pytest

from tessera_embeddings.orchestration.prefect.flows.tessera_embeddings import (
    V2_RUN_PREFIX,
    _resolve_run_id,
)


def _run_id(previous_run_id, *, model_version):
    """``_resolve_run_id`` with the S2-only axis held at its default.

    That axis is exercised in ``test_s2_only_flow_wiring``; pinning it here keeps every
    assertion below a statement about the ENCODER prefix alone.
    """
    return _resolve_run_id(previous_run_id, model_version=model_version, allow_s2_only=False, assembly_only=False)


def test_fresh_v11_run_keeps_the_historical_bare_uuid():
    """The single-model path must be untouched by this change."""
    run_id = _run_id(None, model_version="v1.1")
    assert not run_id.startswith(V2_RUN_PREFIX)
    assert len(run_id) == 12


def test_fresh_v2_run_is_namespaced():
    assert _run_id(None, model_version="v2-large").startswith(V2_RUN_PREFIX)


def test_fresh_run_ids_are_unique():
    ids = {_run_id(None, model_version="v2-large") for _ in range(50)}
    assert len(ids) == 50


@pytest.mark.parametrize("model_version", ["v1.1", "v2-large"])
def test_resuming_with_the_matching_model_is_allowed(model_version):
    original = _run_id(None, model_version=model_version)
    assert _run_id(original, model_version=model_version) == original


def test_resuming_a_v2_run_as_v11_is_refused():
    """The defect: v2 staging finished by the v1.1 encoder and stamped v1.1."""
    v2_run = _run_id(None, model_version="v2-large")
    with pytest.raises(ValueError, match="staged by v2"):
        _run_id(v2_run, model_version="v1.1")


def test_resuming_a_v11_run_as_v2_is_refused():
    v11_run = _run_id(None, model_version="v1.1")
    with pytest.raises(ValueError, match=r"staged by v1\.1"):
        _run_id(v11_run, model_version="v2-large")


def test_error_names_both_versions_so_the_fix_is_obvious():
    v2_run = _run_id(None, model_version="v2-large")
    with pytest.raises(ValueError) as exc:
        _run_id(v2_run, model_version="v1.1")
    msg = str(exc.value)
    assert v2_run in msg
    assert "v1.1" in msg and "v2" in msg


# ── the encoder identity is enforced at every boundary, not only where the id is minted ──


def test_the_runner_refuses_a_run_id_from_the_other_encoder() -> None:
    """``inference.runner`` is a documented public entry point, so the flow's guard is not
    the only door. A caller arriving with a v2 staging id and a v1.1 config would otherwise
    reuse those chunks and publish a mix stamped with one encoder.
    """
    from tessera_embeddings.config.inference import assert_run_id_matches_model

    with pytest.raises(ValueError, match="came from the other encoder"):
        assert_run_id_matches_model(f"{V2_RUN_PREFIX}abc123", "v1.1")
    with pytest.raises(ValueError, match="came from the other encoder"):
        assert_run_id_matches_model("abc123def456", "v2-large")


def test_a_matching_run_id_and_a_fresh_run_both_pass() -> None:
    from tessera_embeddings.config.inference import assert_run_id_matches_model

    assert_run_id_matches_model(f"{V2_RUN_PREFIX}abc123", "v2-large")
    assert_run_id_matches_model("abc123def456", "v1.1")
    assert_run_id_matches_model(None, "v2-large")  # nothing staged yet


def test_the_guard_sees_through_a_composed_prefix() -> None:
    """The two prefixes compose (``v2-s2only-…``), so the encoder marker is not always the
    whole prefix — only the front of it.
    """
    from tessera_embeddings.config.inference import assert_run_id_matches_model

    assert_run_id_matches_model(f"{V2_RUN_PREFIX}s2only-abc123", "v2-large")
    with pytest.raises(ValueError, match="came from the other encoder"):
        assert_run_id_matches_model(f"{V2_RUN_PREFIX}s2only-abc123", "v1.1")


# ── minting and checking must agree at EVERY site, not just the one that had a guard ──


def test_every_minting_site_prefixes_v2_and_leaves_v11_alone() -> None:
    """The regression this pins: the runner's guard reads an unprefixed id as v1.1, so a
    minting site that forgot the prefix made v2 unusable — it raised before an actor started.
    Relaxing the guard for fresh runs would have hidden that while leaving the real defect,
    since the bare id is then misread by every LATER resume of the same run.
    """
    from tessera_embeddings.config.inference import assert_run_id_matches_model, run_id_prefix

    for model in ("v1.1", "v2-large"):
        minted = run_id_prefix(model) + "abc123def456"
        assert_run_id_matches_model(minted, model)  # must not raise at any site

    assert run_id_prefix("v1.1") == "", "v1.1 ids are unchanged, so no existing run is disturbed"
    assert run_id_prefix("v2-large") == V2_RUN_PREFIX


def test_the_plain_runner_mints_an_id_its_own_guard_accepts() -> None:
    """End-to-end at the seam that actually broke: the plain runner is where a YAML
    `model_version: v2-large` enters, and its minted id has to survive `run_inference`.
    Asserted on the runner's real minting expression rather than a reconstruction of it.
    """
    import uuid

    from tessera_embeddings.config.inference import assert_run_id_matches_model, run_id_prefix

    for model in ("v1.1", "v2-large"):
        run_id = run_id_prefix(model) + uuid.uuid4().hex[:12]
        assert_run_id_matches_model(run_id, model)


# ── an id that records no encoder must not be mistaken for no id ──


def test_an_empty_run_id_is_refused_rather_than_read_as_absent() -> None:
    """The guard tested `if run_id and ...`, so `""` was read as "nothing staged yet" and walked
    straight through — while `"   "`, which means the same thing, was refused as unprefixed. The
    inconsistency between two equivalent values is what marks it as a truthiness accident rather
    than a decision.

    It matters because an empty id still names a staging prefix. `_staging_path` interpolates it
    into `staging_base/<run_id>/<label>.zarr`, so `""` yields `staging_base//<label>.zarr`: one
    prefix shared by every run that passes an empty id, whichever model it ran. A v2 run stages
    there, a later v1.1 run scans the same prefix, finds the tiles marked done, resumes over them
    and publishes v2 pixels stamped v1.1 — the precise mixing the prefix exists to prevent, and a
    failure that has already happened once on this pipeline.
    """
    from tessera_embeddings.config.inference import assert_run_id_matches_model

    for empty in ("", "   ", "\t"):
        for model in ("v1.1", "v2-large"):
            with pytest.raises(ValueError, match="is empty"):
                assert_run_id_matches_model(empty, model)

    # None still means "nothing staged yet" — that case is real and must keep passing.
    assert_run_id_matches_model(None, "v1.1")
    assert_run_id_matches_model(None, "v2-large")


def test_an_empty_run_id_shares_one_staging_prefix_across_models() -> None:
    """Why the check above is not pedantry, asserted on the path builder itself rather than on
    prose: two different models handed an empty id address the same object.
    """
    from tessera_embeddings.config.inference import run_id_prefix
    from tessera_embeddings.inference.assembly import ZarrWriter
    from tessera_embeddings.inference.chunk_spec import ChunkSpec

    writer = ZarrWriter("s3://bucket/staging")
    chunk = ChunkSpec(0, 0, 0, 8, 0, 8)

    # Minting is `run_id_prefix(model) + <digest>`. With a real digest the two models address
    # different objects, which is the entire purpose of the prefix.
    digest = "abc123def456"
    assert writer._staging_path(run_id_prefix("v2-large") + digest, chunk) != writer._staging_path(
        run_id_prefix("v1.1") + digest, chunk
    )

    # An empty id is not a namespace at all: the run-id segment collapses, so the tile sits
    # directly under staging_base and carries no model marker anywhere in its path.
    empty_path = writer._staging_path("", chunk)
    assert empty_path == "s3://bucket/staging//chunk_0_0.zarr"
    assert run_id_prefix("v2-large") not in empty_path, "nothing in the path records the encoder"
    assert empty_path.replace("//chunk", "/chunk") == "s3://bucket/staging/chunk_0_0.zarr", (
        "on a local filesystem the doubled separator collapses, so the tile lands loose in "
        "staging_base beside every other run's directory"
    )


@pytest.mark.parametrize("model", ["v1.1", "v2-large"])
@pytest.mark.parametrize("empty", ["", "   "])
def test_the_public_runner_refuses_an_empty_run_id_for_either_model(inference_config, empty, model) -> None:
    """Checked at the public boundary too, and separately from the encoder rule: an empty id is a
    broken staging namespace for v1.1 as much as for v2, so the message a v1.1 caller gets talks
    about staging rather than about encoders.

    Called for real rather than read out of the source. It raises before Ray is touched, which is
    the property under test — the refusal has to land before anything is provisioned.
    """
    from tessera_embeddings.config.inference import InferenceConfig
    from tessera_embeddings.config.time_windows import parse_time_window
    from tessera_embeddings.inference.runner import run_inference

    # Built fresh per model rather than by `replace`: the v2 arch fields and the refusal of a
    # resolved `norm_source` are applied in `__post_init__`, so re-running it over an
    # already-resolved v1.1 config is rejected for a reason that has nothing to do with run_id.
    config = (
        inference_config
        if model == "v1.1"
        else InferenceConfig(
            model_version="v2-large",
            num_obs_checkpoints=(4, 8),
            checkpoint_path="dummy",
            time_window=parse_time_window("June 2025"),
        )
    )

    with pytest.raises(ValueError, match="namespaces the staging prefix"):
        run_inference(
            num_actors=1,
            config=config,
            chunks=[],
            mosaic_base="s3://bucket/mosaics",
            staging_base="s3://bucket/staging",
            run_id=empty,
            t0=0.0,
            log=logging.getLogger("test"),
        )
