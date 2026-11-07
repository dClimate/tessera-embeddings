"""The v2 student is resampled by the rule it was trained under, not v1.1's.

**Why this file exists rather than another golden test.** The model-only golden tests feed
already-sized tensors, so they cannot see the step that decides WHICH observations fill a bucket.
A wrong padding rule produces a tensor of exactly the right shape and dtype containing a sequence
the model was never trained on — there is no error, no shape mismatch, and no downstream check
that could question it. Preprocessing needs its own parity, against upstream's own code.
"""

from __future__ import annotations

import numpy as np

from tessera_embeddings.config.inference import DEFAULT_NUM_OBS_CHECKPOINTS, band_stats
from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
from tessera_embeddings.inference.sampling import build_resample_indices_v2
from tests.fixtures.upstream.v2_pad_pattern import pad_pattern
from tests.unit.inference.test_dataset_v11 import CKPS, _make_chunk_data

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


# ── the wiring, which is where a correct algorithm still fails silently ──


def test_the_dataset_resamples_with_the_version_it_was_given() -> None:
    """An algorithm nobody reaches is worth nothing, and this is the seam that would fail
    quietly: the tensors are the same shape under either rule, so a dataset left on the
    default would look entirely healthy while feeding v2 the wrong sequence.
    """
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


def test_normalisation_and_resampling_cannot_describe_different_students() -> None:
    """The mismatch is removed rather than validated.

    `stats` was a separate argument, so v2's `model_version` could be paired with v1.1's
    statistics — normalised by one student, resampled by the other, and the output is a
    correctly-shaped array of plausible floats either way. The statistics are now derived from
    the version, so the contradiction has no way to be expressed: there is one input, not two
    halves of one identity that a caller has to keep in step.
    """
    chunk_data = _make_chunk_data(rng=np.random.default_rng(5))
    v2 = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS, model_version="v2-large")
    np.testing.assert_array_equal(v2.s2_band_mean, np.array(band_stats("v2-large")["s2_mean"], dtype=np.float32))

    v11 = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS)
    assert not np.array_equal(v11.s2_band_mean, v2.s2_band_mean), "the two students normalise differently"
