"""Resuming a zone must report what the previous attempt actually did.

`run_inference` skips tiles a prior attempt already staged. Two different artifacts mean
"do not re-infer": a staged Zarr, which produced pixels, and a skip marker, which recorded
that the tile had none. The resume path used the label-set scan, which merges them, and then
synthesised every restored label as a success.

That misstates the run in a way the year's provenance depends on. `summarise_radar_coverage`
returns None when tiles report no counters, so a resumed zone's tally disagreeing with the
same zone's tally on a fresh run can suppress the radar-coverage summary entirely.
"""

from __future__ import annotations

from tessera_embeddings.inference.runner import _resumed_result


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


def test_the_two_outcomes_are_distinguishable():
    """The whole point: a resumed zone's skips must not read as successes."""
    skipped = _resumed_result("a", skipped=True)
    success = _resumed_result("a", skipped=False)
    assert skipped["status"] != success["status"]
