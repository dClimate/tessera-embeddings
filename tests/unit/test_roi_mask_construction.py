"""What the mask's new construction must not change, and what it costs.

``read_roi_mask`` moved from ``da.from_zarr`` to ``da.map_blocks`` over a closure so the
credential resolves when a block is read rather than when the graph is built
(``test_roi_mask_credential_expiry.py`` covers the credential itself). The closure builds a
DIFFERENT graph, so the risks this file guards have nothing to do with credentials:

* **The pixels.** The closure derives each block's slice from ``block_info``. A one-pixel
  disagreement at a ragged edge would change what gets embedded, silently, so the mask is
  checked against the numpy array that was written.
* **The laziness.** The old call read no pixels when it built the graph. A closure that
  read eagerly would fetch every ROI whether the leg needed it or not.
* **The serialization.** The old graph carried a zarr array; the new one carries a function
  with captured variables. Only a process-based scheduler exercises that.

The per-block request cost of the change, and the fact that the graph is cloudpickle-only,
live in ``context_docs/decisions/022-resolve-the-roi-mask-credential-at-read-time.md``
rather than in assertions here. The cost is an exact count that a zarr or dask release can
move without anything being wrong, and the pickle limitation could only be asserted AS a
limitation — a test that fails the day someone lifts it.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import functools
from collections.abc import Iterator
from pathlib import Path

import boto3
import numpy as np
import pytest
import s3fs
import zarr
import zarr.storage

from tessera_embeddings.ingest.roi import read_roi_mask

_SRC = Path(__file__).resolve().parents[2] / "src" / "tessera_embeddings"

#: Deliberately awkward: neither dimension is a multiple of the chunk size, and the two axes
#: have different remainders. Ragged edge blocks are where a hand-rolled ``block_info`` slice
#: is likeliest to go wrong, so even the fixture the S3 tests share has them.
SHAPE = (37, 53)
CHUNK = 16


# ---------------------------------------------------------------------------
# Recording what actually goes over the wire
# ---------------------------------------------------------------------------


class _S3Ops:
    """S3 calls s3fs makes, split into metadata and chunk bytes, plus zarr store opens."""

    _METADATA_KEYS = ("zarr.json", ".zarray", ".zattrs", ".zgroup")

    def __init__(self) -> None:
        self.calls: collections.Counter[tuple[str, str | None]] = collections.Counter()
        self.opens = 0

    @property
    def metadata_gets(self) -> int:
        """Requests for one of zarr's metadata keys."""
        return sum(n for (_, key), n in self.calls.items() if key and key.endswith(self._METADATA_KEYS))

    @property
    def chunk_gets(self) -> int:
        """Requests for chunk BYTES: everything that is not metadata."""
        return sum(self.calls.values()) - self.metadata_gets


@contextlib.contextmanager
def _recording() -> Iterator[_S3Ops]:
    """Count S3 calls at s3fs' single choke point, and store opens at zarr's."""
    ops = _S3Ops()
    real_call = s3fs.S3FileSystem._call_s3
    real_open = zarr.open_array

    async def counting_call(self, method, *args, **kwargs):
        ops.calls[(method, kwargs.get("Key"))] += 1
        return await real_call(self, method, *args, **kwargs)

    def counting_open(*args, **kwargs):
        ops.opens += 1
        return real_open(*args, **kwargs)

    s3fs.S3FileSystem._call_s3 = counting_call  # type: ignore[method-assign]
    zarr.open_array = counting_open  # type: ignore[assignment]
    try:
        yield ops
    finally:
        s3fs.S3FileSystem._call_s3 = real_call  # type: ignore[method-assign]
        zarr.open_array = real_open  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def roi_on_s3(moto_server) -> tuple[str, dict, np.ndarray]:
    """An ROI mask on a real S3 protocol stack, with a known pseudo-random pattern.

    A pattern rather than all-True: an all-True mask cannot distinguish a correct read
    from one that returns the fill value for every block it fails to locate.
    """
    options = {
        "key": "testing",
        "secret": "testing",
        "client_kwargs": {"endpoint_url": moto_server, "region_name": "us-east-1"},
    }
    boto3.client(
        "s3",
        endpoint_url=moto_server,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
    ).create_bucket(Bucket="roi-construction")

    truth = np.random.default_rng(20260822).random(SHAPE) > 0.5
    url = "s3://roi-construction/mask.zarr"
    store = zarr.storage.FsspecStore.from_url(url, storage_options=options)
    zarr.create_array(store=store, shape=SHAPE, chunks=(CHUNK, CHUNK), dtype="bool", overwrite=True)[:] = truth
    return url, options, truth


@pytest.fixture
def s3_chunks() -> dict[str, int]:
    return {"northing": CHUNK, "easting": CHUNK}


def _local_roi(tmp_path: Path, shape: tuple[int, int], chunk: int) -> tuple[str, np.ndarray]:
    truth = np.random.default_rng(shape[0] * 1000 + shape[1]).random(shape) > 0.4
    path = tmp_path / "mask.zarr"
    zarr.create_array(store=str(path), shape=shape, chunks=(chunk, chunk), dtype="bool")[:] = truth
    return str(path), truth


def _expected_blocks(length: int, chunk: int) -> tuple[int, ...]:
    """The block sizes along one axis, worked out here rather than asked of dask."""
    whole, remainder = divmod(length, chunk)
    return (chunk,) * whole + ((remainder,) if remainder else ())


# ---------------------------------------------------------------------------
# 1. The mask must be the pixels that were written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "chunk"),
    [
        pytest.param((37, 53), 16, id="ragged-both-axes"),
        pytest.param((32, 32), 16, id="exact-multiple"),
        pytest.param((10, 10), 64, id="chunk-larger-than-array"),
        pytest.param((1, 40), 16, id="single-row"),
        pytest.param((40, 1), 16, id="single-column"),
        pytest.param((17, 17), 1, id="one-pixel-blocks"),
    ],
)
def test_the_mask_is_the_array_that_was_written(tmp_path, shape, chunk) -> None:
    """Same pixels, dtype, shape and block layout as the store on disk.

    The biggest risk in this change and the one with nothing to do with credentials. The
    closure slices each block out of the store using ``block_info``, so an off-by-one at a
    ragged edge is what to look for: it would change what gets embedded, silently, which is
    why the edge cases are parametrised rather than trusted.

    Both sides of the comparison are independent of the reader — the numpy array that was
    written, and a block layout computed here — so this says the mask is CORRECT rather
    than merely unchanged.
    """
    path, written = _local_roi(tmp_path, shape, chunk)

    mask = read_roi_mask(path, {"northing": chunk, "easting": chunk})

    assert mask.shape == shape
    assert mask.dtype == np.bool_
    assert mask.chunks == (_expected_blocks(shape[0], chunk), _expected_blocks(shape[1], chunk)), (
        "the block layout is not the one asked for; every downstream graph rechunks"
    )
    assert np.array_equal(mask.compute(), written), "the mask is not the array that was written"


# ---------------------------------------------------------------------------
# 2. It must still be lazy
# ---------------------------------------------------------------------------


def test_construction_reads_metadata_and_never_chunk_bytes(roi_on_s3, s3_chunks) -> None:
    """Building the graph learns the array's shape and reads none of its pixels.

    Not zero requests: opening a zarr array to learn its shape and dtype costs a metadata
    probe either way, and how many requests that probe takes is zarr's business, so the
    assertions here are on the KIND of request and not the count. What matters is that no
    chunk BYTES are fetched — a closure that read eagerly would pull every ROI whether the
    leg needed it or not, and at zone scale that is tens of GiB per graph build.
    """
    url, options, _ = roi_on_s3
    s3fs.S3FileSystem.clear_instance_cache()
    with _recording() as ops:
        array = read_roi_mask(url, s3_chunks, lambda: dict(options))
        assert ops.chunk_gets == 0, f"construction fetched chunk bytes: {dict(ops.calls)}"
        assert ops.metadata_gets > 0, "construction read no metadata, so it cannot know the shape"
        assert ops.opens == 1, f"construction opened the store {ops.opens} times, expected 1"
    assert array.shape == SHAPE  # the metadata really was read, so the counts mean something


# ---------------------------------------------------------------------------
# 3. The closure must survive being sent to a Dask worker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_options",
    [
        pytest.param(lambda opts: opts, id="dict"),
        pytest.param(lambda opts: functools.partial(dict, **opts), id="provider-callable"),
    ],
)
def test_the_closure_survives_a_process_based_scheduler(roi_on_s3, s3_chunks, make_options) -> None:
    """PROCESSES, not threads: the only scheduler that actually serializes the graph.

    The old graph shipped a zarr array holding a filesystem object. The new one ships a
    nested function plus its captured variables, which plain pickle cannot do at all — it
    survives only because distributed falls back to cloudpickle. A threaded or synchronous
    scheduler shares memory and would prove nothing about that.

    Run with both storage-option forms: a dict, and a callable provider, since the provider
    has to arrive on the worker still callable for the credential to be resolved there
    rather than here.
    """
    distributed = pytest.importorskip("distributed")
    url, options, truth = roi_on_s3

    with (
        distributed.LocalCluster(
            n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None, silence_logs=50
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        mask = read_roi_mask(url, s3_chunks, make_options(dict(options)))
        computed = client.compute(mask, sync=True)

    assert np.array_equal(computed, truth), "the mask read on a worker process differs from what was written"


# ---------------------------------------------------------------------------
# 4. Pin the fallback shut
# ---------------------------------------------------------------------------


def test_every_caller_of_apply_roi_mask_supplies_the_mask() -> None:
    """``apply_roi_mask``'s fallback reads the mask with NO credentials at all.

    ``mask_2d = roi_mask if roi_mask is not None else read_roi_mask(roi_zarr_path, spatial_chunks)``
    — no ``storage_options``, so fsspec resolves the process environment, which on the
    radar path holds the SOURCE's OPERA-scoped token. That read would be refused on our
    own bucket. It is unreachable only because every production caller passes
    ``roi_mask=``, which is a property of the call sites and nothing enforces it.

    This test is the enforcement: a new caller that omits the mask fails here rather
    than in a leg.
    """
    offenders = []
    for module in sorted(_SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(module.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "apply_roi_mask":
                continue
            if not any(keyword.arg == "roi_mask" for keyword in node.keywords):
                offenders.append(f"{module.relative_to(_SRC)}:{node.lineno}")
    assert not offenders, (
        "apply_roi_mask called without roi_mask= at "
        f"{offenders} — that path reads the mask with no storage_options, resolving the "
        "environment. Pass the mask, or give the fallback a storage_options parameter."
    )


# ---------------------------------------------------------------------------
# 5. Both sensors (source-level half; the behavioural half is in the expiry file)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["s1_roi.py", "s2_roi.py"])
def test_each_sensor_reads_through_the_one_fixed_reader(module: str) -> None:
    """Neither sensor may build a mask any other way.

    The fix lives in one function precisely so both sensors inherit it. A sensor that
    grew its own ``da.from_zarr`` would be back to a credential frozen at graph build,
    and every other test here would still pass.
    """
    tree = ast.parse((_SRC / "ingest" / module).read_text())
    direct = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "from_zarr"
    ]
    assert not direct, f"{module} builds a zarr-backed array directly at line(s) {direct}"

    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "read_roi_mask"
    ]
    assert reads, f"{module} no longer reads the ROI mask; this test is checking nothing"


@pytest.mark.parametrize("module", ["s1_roi.py", "s2_roi.py"])
def test_no_sensor_freezes_its_mask_by_persisting_it(module: str) -> None:
    """``persist()`` on the mask would resolve every block's credential at persist time.

    That is the same defect one layer out: the reads would happen once, early, and the
    materialised result would then be reused for the rest of the leg. Both sensors
    currently document the choice to stay lazy for a memory reason; this pins it for the
    credential reason as well.
    """
    tree = ast.parse((_SRC / "ingest" / module).read_text())
    masks = {
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "read_roi_mask"
    }
    assert masks, f"{module} has no mask assignment to check"
    frozen = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "persist"
        and getattr(node.func.value, "id", None) in masks
    ]
    assert not frozen, f"{module} persists the ROI mask at line(s) {frozen}, refreezing the credential"
