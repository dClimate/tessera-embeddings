"""Property-based tests for the most generic invariants in the package.

We use Hypothesis where the input space is genuinely combinatorial
(many-shapes-of-jobs, many-bucket-paths) and where the property is
better expressed as "for all X: P(X)" than as a curated list of
inputs.

* :func:`sliding_window_submit` — max-concurrent invariant + completeness.
* :class:`BucketPaths.store_for` — output URI is deterministic and
  uniquely keyed by ``(roi_name, kind)``.
* :class:`RoiManifest`, :class:`EmbeddingManifest` — ``to_dict ↔ from_dict``
  round-trips identity, and ``hash()`` is stable under that identity.

Hypothesis configuration is in ``tests/conftest.py``.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from hypothesis import given
from hypothesis import strategies as st

from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.orchestration.concurrency import sliding_window_submit
from tessera_embeddings.storage.manifest import (
    EmbeddingManifest,
    IngestManifest,
    RoiManifest,
)

# ---------------------------------------------------------------------------
# sliding_window_submit
# ---------------------------------------------------------------------------


@given(
    n_jobs=st.integers(min_value=0, max_value=50),
    max_concurrent=st.integers(min_value=1, max_value=8),
)
def test_sliding_window_returns_one_pair_per_job(n_jobs: int, max_concurrent: int) -> None:
    """Every job appears exactly once in the returned list, regardless of size."""
    with ThreadPoolExecutor(max_workers=max_concurrent + 2) as pool:
        jobs = [(i, {"x": i}) for i in range(n_jobs)]
        completed = sliding_window_submit(
            submit_fn=lambda kw: pool.submit(lambda x: x, **kw),
            jobs=jobs,
            max_concurrent=max_concurrent,
        )
    assert sorted(k for k, _ in completed) == list(range(n_jobs))


@given(
    n_jobs=st.integers(min_value=4, max_value=20),
    max_concurrent=st.integers(min_value=1, max_value=4),
)
def test_sliding_window_respects_concurrency_cap(n_jobs: int, max_concurrent: int) -> None:
    """In-flight count never exceeds ``max_concurrent``, regardless of timing."""
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def slow(_: int) -> int:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.005)
        with lock:
            in_flight -= 1
        return 0

    # Pool capacity is intentionally larger than max_concurrent so that
    # the only concurrency limiter is sliding_window_submit itself.
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        jobs = [(i, {"_": i}) for i in range(n_jobs)]
        sliding_window_submit(
            submit_fn=lambda kw: pool.submit(slow, **kw),
            jobs=jobs,
            max_concurrent=max_concurrent,
        )
    assert peak <= max_concurrent


# ---------------------------------------------------------------------------
# BucketPaths.store_for
# ---------------------------------------------------------------------------


_KINDS = ["reflectance", "sar_ascending", "sar_descending", "embeddings", "staging", "roi"]


@st.composite
def _bucket_paths(draw: st.DrawFn) -> BucketPaths:
    schemes = draw(st.sampled_from(["s3://my-bucket", "gs://demo", "file:///tmp"]))
    return BucketPaths(
        inputs=f"{schemes}/inputs",
        outputs=f"{schemes}/outputs",
    )


_ROI_NAMES = st.text(
    alphabet=st.characters(min_codepoint=0x30, max_codepoint=0x7A, blacklist_characters="/\\"),
    min_size=1,
    max_size=20,
)


@given(paths=_bucket_paths(), roi_name=_ROI_NAMES, kind=st.sampled_from(_KINDS))
def test_store_for_is_deterministic(paths: BucketPaths, roi_name: str, kind: str) -> None:
    """``store_for(roi, kind)`` returns the same URI on every call."""
    a = paths.store_for(roi_name, kind)
    b = paths.store_for(roi_name, kind)
    assert a == b


@given(paths=_bucket_paths(), roi_name=_ROI_NAMES)
def test_store_for_distinguishes_kinds(paths: BucketPaths, roi_name: str) -> None:
    """Different ``kind`` arguments produce different URIs."""
    uris = {kind: paths.store_for(roi_name, kind) for kind in _KINDS}
    assert len(set(uris.values())) == len(_KINDS)


# ---------------------------------------------------------------------------
# Manifest serialization round-trip
# ---------------------------------------------------------------------------


@given(
    resolution=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    chunk_size=st.integers(min_value=1, max_value=10_000),
    crs=st.one_of(st.none(), st.sampled_from(["EPSG:4326", "EPSG:32615", "EPSG:3857"])),
)
def test_roi_manifest_round_trip(resolution: float, chunk_size: int, crs: str | None) -> None:
    """``to_dict`` then ``from_dict`` preserves field values."""
    original = RoiManifest(resolution=resolution, chunk_size=chunk_size, crs=crs)
    restored = RoiManifest.from_dict(original.to_dict())
    assert restored == original
    # Hash is stable under serialise/restore
    assert restored.hash() == original.hash()


@given(
    model_checkpoint=st.text(min_size=1, max_size=80),
    num_obs_checkpoints=st.lists(st.integers(min_value=1, max_value=256), min_size=1, max_size=32).map(
        lambda xs: tuple(sorted(set(xs)))
    ),
    reflectance_manifest_hash=st.one_of(st.none(), st.text(min_size=1, max_size=64)),
    sar_manifest_hash=st.one_of(st.none(), st.text(min_size=1, max_size=64)),
)
def test_embedding_manifest_round_trip(
    model_checkpoint: str,
    num_obs_checkpoints: tuple[int, ...],
    reflectance_manifest_hash: str | None,
    sar_manifest_hash: str | None,
) -> None:
    """Embedding manifest fields survive a to_dict / from_dict cycle.

    Verifies that the tuple→list→tuple coercion through JSON zarr-attrs
    preserves field equality and hash stability.
    """
    original = EmbeddingManifest(
        model_checkpoint=model_checkpoint,
        num_obs_checkpoints=num_obs_checkpoints,
        reflectance_manifest_hash=reflectance_manifest_hash,
        sar_manifest_hash=sar_manifest_hash,
    )
    restored = EmbeddingManifest.from_dict(original.to_dict())
    # from_dict re-coerces lists back to tuples via the dataclass field type
    assert restored == original
    assert restored.hash() == original.hash()


@given(roi_manifest_hash=st.one_of(st.none(), st.text(min_size=1, max_size=64)))
def test_ingest_manifest_round_trip(roi_manifest_hash: str | None) -> None:
    """Ingest manifest survives a to_dict / from_dict cycle."""
    original = IngestManifest(roi_manifest_hash=roi_manifest_hash)
    restored = IngestManifest.from_dict(original.to_dict())
    assert restored == original
    assert restored.hash() == original.hash()
