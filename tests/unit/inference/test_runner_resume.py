"""Resuming a zone: the run_id it resumes under, and what the previous attempt actually did.

`run_inference` skips tiles a prior attempt already staged. Two different artifacts mean
"do not re-infer": a staged Zarr, which produced pixels, and a skip marker, which recorded
that the tile had none. The resume path used the label-set scan, which merges them, and then
synthesised every restored label as a success.

That misstates the run in a way the year's provenance depends on. `summarise_radar_coverage`
returns None when tiles report no counters, so a resumed zone's tally disagreeing with the
same zone's tally on a fresh run can suppress the radar-coverage summary entirely.
"""

from __future__ import annotations

import logging

import pytest

from tessera_embeddings.inference.runner import _resumed_result, run_inference


def test_a_restored_skip_reports_as_skipped():
    """The status must mirror what the actor reported at the time, not a default."""
    assert _resumed_result("32601/2025/r0_c0", skipped=True)["status"] == "skipped"


def test_a_restored_staged_tile_reports_as_success():
    assert _resumed_result("32601/2025/r0_c1", skipped=False)["status"] == "success"


def test_both_are_marked_resumed_and_carry_no_invented_counters():
    """`resumed` is how a caller tells a restored outcome from a measured one.

    The counters stay zero rather than being invented: this run measured nothing, and the
    radar summary already refuses to report a year whose tiles carry no counts rather than
    averaging over the subset that happened to be redone.
    """
    for skipped in (True, False):
        r = _resumed_result("32601/2025/r1_c0", skipped=skipped)
        assert r["resumed"] is True
        assert r["valid_pixels"] == 0
        assert r["elapsed_sec"] == 0.0
        assert r["chunk"] == "32601/2025/r1_c0"


def test_the_public_runner_refuses_an_empty_run_id(inference_config) -> None:
    """An empty id is a broken staging namespace for v1.1 as much as for v2, so the refusal talks
    about staging rather than encoders, and lands before Ray is touched.
    """
    with pytest.raises(ValueError, match="namespaces the staging prefix"):
        run_inference(
            num_actors=1,
            config=inference_config,
            chunks=[],
            mosaic_base="s3://bucket/mosaics",
            staging_base="s3://bucket/staging",
            run_id="",
            t0=0.0,
            log=logging.getLogger("test"),
        )
