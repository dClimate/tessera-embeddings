"""The v2 student is resampled by the rule it was trained under, not v1.1's.

**Why this file exists rather than another golden test.** The model-only golden tests feed
already-sized tensors, so they cannot see the step that decides WHICH observations fill a bucket.
A wrong padding rule produces a tensor of exactly the right shape and dtype containing a sequence
the model was never trained on — there is no error, no shape mismatch, and no downstream check
that could question it. Preprocessing needs its own parity, against upstream's own code.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera_embeddings.config.inference import DEFAULT_NUM_OBS_CHECKPOINTS
from tessera_embeddings.inference.sampling import (
    build_resample_indices,
    build_resample_indices_v2,
    resampler_for,
)
from tests.fixtures.upstream.v2_pad_pattern import pad_pattern

#: Every bucket this pipeline can choose, against counts either side of each one.
_CASES = [(n, b) for b in DEFAULT_NUM_OBS_CHECKPOINTS for n in range(1, 2 * b + 1)]


def test_v2_matches_upstream_for_every_reachable_count() -> None:
    """Exhaustive over every (count, bucket) this pipeline can produce.

    Checked in ONE test rather than parametrised into thousands. The coverage is identical —
    every pair is compared — but a suite total that jumps by 8,451 reports a number nobody can
    read, and that total is a signal worth keeping legible. The diagnostic parametrising would
    have given is reproduced by hand: the failure says how many pairs diverged and shows the
    first few, which is more useful than one arbitrary pair surfacing as its own test id.
    """
    mismatches = [
        (n, bucket, build_resample_indices_v2(n, bucket).tolist(), pad_pattern(n, bucket).tolist())
        for n, bucket in _CASES
        if not np.array_equal(build_resample_indices_v2(n, bucket), pad_pattern(n, bucket))
    ]
    assert not mismatches, (
        f"{len(mismatches)} of {len(_CASES)} (count, bucket) pairs diverge from upstream. First: "
        + "; ".join(f"n={n} B={b}: ours {o} vs upstream {u}" for n, b, o, u in mismatches[:3])
    )


def test_the_two_rules_really_do_disagree() -> None:
    """The guard on the guard: if a refactor ever made v2 delegate to v1.1, every parity
    assertion above would still pass against a reference that had also been "helpfully"
    aligned. This pins the disagreement itself.
    """
    differing = sum(
        1
        for n, b in _CASES
        if len(build_resample_indices(n, b)) != len(build_resample_indices_v2(n, b))
        or not np.array_equal(build_resample_indices(n, b), build_resample_indices_v2(n, b))
    )
    assert differing > len(_CASES) // 2, "v1.1 and v2 padding differ on most inexact counts"


def test_v11_is_untouched_by_the_addition() -> None:
    """The path that must NOT change. The campaign runs v1.1, and a shifted index vector would
    silently alter every published embedding, so selection is the only thing this change does
    to it.
    """
    for n, bucket in _CASES:
        np.testing.assert_array_equal(
            resampler_for("v1.1")(n, bucket), build_resample_indices(n, bucket), err_msg=f"{n}->{bucket}"
        )


def test_an_unknown_model_version_is_refused_rather_than_defaulted() -> None:
    """Defaulting a new student to v1.1's rule is the exact failure this versioning prevents,
    and it would produce a correctly-shaped tensor nobody could question.
    """
    with pytest.raises(ValueError, match="No resampler for model_version"):
        resampler_for("v3-enormous")


# ── the wiring, which is where a correct algorithm still fails silently ──


def test_the_dataset_resamples_with_the_version_it_was_given() -> None:
    """An algorithm nobody reaches is worth nothing, and this is the seam that would fail
    quietly: the tensors are the same shape under either rule, so a dataset left on the
    default would look entirely healthy while feeding v2 the wrong sequence.
    """
    from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
    from tests.unit.test_dataset_v11 import CKPS, _make_chunk_data

    chunk_data = _make_chunk_data(rng=np.random.default_rng(3))
    v11 = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS)
    v2 = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS, model_version="v2-large")

    assert v11.model_version == "v1.1", "the default must stay v1.1 — every existing caller relies on it"
    assert v2.model_version == "v2-large"

    # Same pixels, same buckets, different padding rule → the gathered sequences must differ
    # somewhere. If they never do, the version is not reaching the resampler.
    differed = False
    for key, pixels in v2.iter_buckets():
        a = v11.get_bucket_batch(key, 0, len(pixels))
        b = v2.get_bucket_batch(key, 0, len(pixels))
        for name in ("s2", "s1"):
            assert a[name].shape == b[name].shape, "shape is NOT the signal — both rules fill the bucket"
            if not np.array_equal(a[name], b[name]):
                differed = True
    assert differed, "v2 produced v1.1's sequences — the model_version never reached the resampler"


def test_the_actor_passes_its_configured_version_to_the_dataset() -> None:
    """The last link. `stats` and `model_version` are resolved at the same call site, and an
    actor that versioned one but not the other would normalise as v2 and resample as v1.1.
    """
    import inspect

    import tessera_embeddings.inference.actors as actors_mod

    src = inspect.getsource(actors_mod)
    assert "model_version=self.config.model_version," in src, (
        "the actor must hand its configured model version to the dataset, beside the band stats"
    )
