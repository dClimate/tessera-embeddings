"""What the mask's new construction must not change, and what it costs.

``read_roi_mask`` moved from ``da.from_zarr`` to ``da.map_blocks`` over a closure so the
credential resolves when a block is read rather than when the graph is built
(``test_roi_mask_credential_expiry.py`` covers the credential itself). The two build
DIFFERENT graphs, so the risk this file carries is not credentials at all:

* **The pixels.** A mask that differs anywhere changes what gets embedded. Every test
  here comparing the two constructions is a REGRESSION GUARD — it passes on ``main`` by
  construction, because ``main`` is one of the two things being compared.
* **The laziness.** The old call read nothing at build time. A closure that reads eagerly
  would fetch every ROI whether the leg needs it or not.
* **The cost.** One ``zarr.open_array`` per block is one metadata round trip per block
  where there used to be one for the whole array. The tests below MEASURE that rather
  than asserting a bound, and the measured figures are recorded in their docstrings.
* **The serialization.** The old graph carried a zarr array; the new one carries a
  function with captured variables. Only a process-based scheduler exercises that.

The old construction is kept here verbatim as :func:`_main_read_roi_mask`. It is a fossil
on purpose: a comparison against a re-derivation of what ``main`` did would drift.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import functools
from collections.abc import Callable, Iterator
from pathlib import Path

import boto3
import dask
import dask.array as da
import numpy as np
import pytest
import s3fs
import zarr
import zarr.storage

from tessera_embeddings.ingest.roi import read_roi_mask, resolve_storage_options

_SRC = Path(__file__).resolve().parents[2] / "src" / "tessera_embeddings"

#: Deliberately awkward: neither dimension is a multiple of the chunk size, and the two
#: axes have different remainders. Ragged edge blocks are where a hand-rolled
#: ``block_info`` slice and ``da.from_zarr``'s own chunking are most likely to disagree.
SHAPE = (37, 53)
CHUNK = 16


def _main_read_roi_mask(
    roi_path: str,
    chunks: dict[str, int],
    storage_options: dict | Callable[[], dict | None] | None = None,
) -> da.Array:
    """``main``'s construction, verbatim, as the arm to compare against.

    Copied rather than imported because the point is to compare against what shipped
    before this change; a re-derivation would drift with the code it is checking.
    """
    return da.from_zarr(
        roi_path,
        chunks=(chunks["northing"], chunks["easting"]),
        storage_options=resolve_storage_options(storage_options),
    )


BOTH_CONSTRUCTIONS = pytest.mark.parametrize(
    "build",
    [pytest.param(_main_read_roi_mask, id="old-from_zarr"), pytest.param(read_roi_mask, id="new-map_blocks")],
)


# ---------------------------------------------------------------------------
# Recording what actually goes over the wire
# ---------------------------------------------------------------------------


class _S3Ops:
    """Every S3 API call s3fs makes, and every zarr store open, by key."""

    def __init__(self) -> None:
        self.calls: collections.Counter[tuple[str, str | None]] = collections.Counter()
        self.opens = 0

    @property
    def total(self) -> int:
        return sum(self.calls.values())

    def keys_matching(self, *suffixes: str) -> int:
        """Requests whose S3 key ends in any of ``suffixes`` — metadata vs chunk bytes."""
        return sum(n for (_, key), n in self.calls.items() if key and key.endswith(suffixes))

    @property
    def chunk_gets(self) -> int:
        """Requests for chunk BYTES: everything that is not one of zarr's metadata keys."""
        return self.total - self.keys_matching("zarr.json", ".zarray", ".zattrs", ".zgroup")

    def reset(self) -> None:
        self.calls.clear()
        self.opens = 0


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


# ---------------------------------------------------------------------------
# 1. The mask must be bit-identical
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
def test_the_new_construction_is_bit_identical_to_the_old(tmp_path, shape, chunk) -> None:
    """REGRESSION GUARD (passes on ``main``): same pixels, dtype, shape and chunk layout.

    The biggest risk in this change and the one with nothing to do with credentials.
    ``da.from_zarr`` derives its blocks from the zarr array; the closure derives its
    slices from ``block_info``. A one-pixel disagreement at a ragged edge would change
    what gets embedded, silently, so the edge cases are parametrised rather than trusted.

    Checked against BOTH the old construction and the numpy array that was written —
    an independent oracle, so the two constructions agreeing on something wrong fails too.
    """
    path, truth = _local_roi(tmp_path, shape, chunk)
    chunks = {"northing": chunk, "easting": chunk}

    old = _main_read_roi_mask(path, chunks)
    new = read_roi_mask(path, chunks)

    assert new.shape == old.shape == truth.shape
    assert new.dtype == old.dtype == np.bool_
    assert new.chunks == old.chunks, "the block layout changed; every downstream graph rechunks"
    old_values, new_values = old.compute(), new.compute()
    assert np.array_equal(new_values, old_values), "the two constructions disagree pixel for pixel"
    assert np.array_equal(new_values, truth), "both constructions disagree with what was written"


def test_the_new_construction_is_bit_identical_over_s3(roi_on_s3, s3_chunks) -> None:
    """REGRESSION GUARD: the same equality over the real S3 protocol stack.

    A local path and an fsspec URL take different code inside zarr, and the closure only
    ever runs against S3 in production.
    """
    url, options, truth = roi_on_s3
    old = _main_read_roi_mask(url, s3_chunks, lambda: dict(options))
    new = read_roi_mask(url, s3_chunks, lambda: dict(options))

    assert new.chunks == old.chunks
    assert new.dtype == old.dtype
    old_values, new_values = old.compute(), new.compute()
    assert np.array_equal(new_values, old_values)
    assert np.array_equal(new_values, truth)


# ---------------------------------------------------------------------------
# 2. It must still be lazy
# ---------------------------------------------------------------------------


@BOTH_CONSTRUCTIONS
def test_construction_reads_metadata_and_never_chunk_bytes(roi_on_s3, s3_chunks, build) -> None:
    """MEASURED, both arms: 4 metadata GETs and 0 chunk GETs to build the graph.

    Not zero requests — opening a zarr array to learn its shape and dtype costs a probe
    either way, and zarr 3 probes both layouts (``zarr.json`` twice, plus the v2
    ``.zarray`` and ``.zattrs``). What matters is that no chunk BYTES are fetched: a
    closure that read eagerly would pull every ROI whether the leg needed it or not, and
    at zone scale that is tens of GiB per graph build.

    Both arms measure identically, so this change costs nothing at construction.
    """
    url, options, _ = roi_on_s3
    s3fs.S3FileSystem.clear_instance_cache()
    with _recording() as ops:
        array = build(url, s3_chunks, lambda: dict(options))
        assert ops.chunk_gets == 0, f"construction fetched chunk bytes: {dict(ops.calls)}"
        assert ops.total == 4, f"construction request count moved from 4 to {ops.total}: {dict(ops.calls)}"
        assert ops.opens == 1, f"construction opened the store {ops.opens} times, expected 1"
    assert array.shape == SHAPE  # the metadata really was read, so the count means something


@BOTH_CONSTRUCTIONS
def test_nothing_is_computed_until_compute_is_called(tmp_path, build) -> None:
    """Both arms: the returned array is a real dask graph, not a materialised result.

    Uses the store's own disappearance as the proof — if construction had read the
    pixels, deleting the store afterwards could not break the later compute.
    """
    path, truth = _local_roi(tmp_path, (37, 53), 16)
    array = build(path, {"northing": 16, "easting": 16})
    assert isinstance(array, da.Array)
    assert array.npartitions == 12

    for chunk_file in sorted(Path(path).glob("c/**/*")):
        if chunk_file.is_file():
            chunk_file.unlink()
    computed = array.compute()
    assert not np.array_equal(computed, truth), (
        "the pixels survived deleting every chunk on disk, so construction had already read them"
    )


# ---------------------------------------------------------------------------
# 3. How many times the store is opened for one full mask read
# ---------------------------------------------------------------------------


def test_the_store_is_opened_once_per_block_where_it_used_to_be_once(roi_on_s3, s3_chunks) -> None:
    """MEASURED COST OF THE CHANGE, and it is a real regression, not a rounding error.

    For this 12-block mask, reading all of it:

    ======================  ==========  ==========
    ..                      old         new
    ======================  ==========  ==========
    store opens (build)     1           1
    store opens (compute)   0           12
    S3 requests (compute)   24          72
    ...of which metadata    0           48
    ...of which chunk       24          24
    ======================  ==========  ==========

    So the new construction re-opens the store once per block and pays that open's FOUR
    metadata GETs each time: 6 requests per block against the old 2, a flat **3x** on the
    mask read whatever the block count. The chunk reads themselves are unchanged.

    In absolute terms this is small — metadata GETs are a few hundred bytes, fsspec hands
    every block the same cached filesystem instance, and the mask is a rounding error
    beside a date's source imagery. It is recorded here as an exact figure rather than a
    bound so that a later change making it worse has to come and edit this table.

    Note the PR description's "one or two metadata GETs" per block understates it: zarr 3
    probes both the v3 and v2 layouts, so it is four.
    """
    url, options, _ = roi_on_s3
    measured = {}
    for label, build in (("old", _main_read_roi_mask), ("new", read_roi_mask)):
        s3fs.S3FileSystem.clear_instance_cache()
        with _recording() as ops:
            array = build(url, s3_chunks, lambda: dict(options))
            build_opens = ops.opens
            ops.reset()
            array.compute()
            measured[label] = {
                "build_opens": build_opens,
                "compute_opens": ops.opens,
                "compute_requests": ops.total,
                "compute_chunk_gets": ops.chunk_gets,
            }

    assert array.npartitions == 12, "the table below is written for a 12-block mask"
    assert measured["old"] == {
        "build_opens": 1,
        "compute_opens": 0,
        "compute_requests": 24,
        "compute_chunk_gets": 24,
    }, measured["old"]
    assert measured["new"] == {
        "build_opens": 1,
        "compute_opens": 12,
        "compute_requests": 72,
        "compute_chunk_gets": 24,
    }, measured["new"]
    assert measured["new"]["compute_opens"] == array.npartitions, "one open per block is the design"
    assert measured["new"]["compute_chunk_gets"] == measured["old"]["compute_chunk_gets"], (
        "the change must not alter how many chunk bytes are fetched, only the metadata around them"
    )


def test_a_partial_read_opens_the_store_only_for_the_blocks_it_needs(roi_on_s3, s3_chunks) -> None:
    """The per-block cost is culled with the blocks, which is what makes it affordable.

    Every production consumer slices the mask to live windows, so a zone's 3,876-block
    grid is never opened 3,876 times. Measured: one block read opens the store once.
    """
    url, options, _ = roi_on_s3
    s3fs.S3FileSystem.clear_instance_cache()
    with _recording() as ops:
        mask = read_roi_mask(url, s3_chunks, lambda: dict(options))
        ops.reset()
        mask[0:CHUNK, 0:CHUNK].compute()
        assert ops.opens == 1, f"a one-block read opened the store {ops.opens} times"


# ---------------------------------------------------------------------------
# 4. The closure must survive being sent to a Dask worker
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
    nested function plus its captured variables, which plain pickle cannot do at all —
    it survives only because distributed falls back to cloudpickle. A threaded or
    synchronous scheduler shares memory and would prove nothing about that.

    Run with both storage-option forms: a dict, and a callable provider, since the
    provider has to arrive on the worker still callable for the credential to be
    resolved there rather than here.
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


def test_the_graph_needs_cloudpickle_and_plain_pickle_will_not_do(tmp_path) -> None:
    """MEASURED CONSTRAINT the change introduces: the graph is no longer plain-picklable.

    ``pickle.dumps`` on the new array raises
    ``Can't get local object 'read_roi_mask.<locals>._read_block'`` — a nested function
    cannot be pickled by reference, which is the exact mistake
    ``AssumedRoleIcechunkCredentials`` exists to document for icechunk's callback. The old
    ``da.from_zarr`` graph carried a zarr array and pickled fine.

    It is safe today because distributed falls back to cloudpickle and nothing in ``src``
    plain-pickles a graph (checked: no ``pickle.dumps`` anywhere, no custom dask
    serializers configured). Recorded here so that anyone who later hands this array to a
    plain-pickle boundary — a multiprocessing pool, a cache, an icechunk callback — finds
    out from this test instead of from a leg.
    """
    import pickle

    cloudpickle = pytest.importorskip("cloudpickle")
    path, truth = _local_roi(tmp_path, (37, 53), 16)
    chunks = {"northing": 16, "easting": 16}

    old = _main_read_roi_mask(path, chunks)
    pickle.loads(pickle.dumps(old))  # the old construction round-trips under plain pickle

    new = read_roi_mask(path, chunks)
    with pytest.raises((AttributeError, TypeError, pickle.PicklingError)):
        pickle.dumps(new)
    assert np.array_equal(cloudpickle.loads(cloudpickle.dumps(new)).compute(), truth)


# ---------------------------------------------------------------------------
# 6. Pin the fallback shut
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


def test_the_fallback_read_still_declares_no_identity() -> None:
    """Records the fallback as it stands, so closing it has to update this test too.

    The PR found this read and deliberately left it, on the argument that closing it
    means a parameter no caller needs. That decision is only safe while the census above
    holds, and this test names the line the census is protecting.
    """
    source = (_SRC / "ingest" / "roi_processing.py").read_text()
    assert "else read_roi_mask(roi_zarr_path, spatial_chunks)" in source, (
        "the fallback changed shape — if it now takes storage_options, delete this test and relax the census above"
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


def test_the_synchronous_scheduler_agrees_with_the_threaded_one(tmp_path) -> None:
    """The closure must not depend on which scheduler runs it.

    ``block_info`` is supplied by dask, and a scheduler that passed it differently would
    give a correct-looking array of wrong pixels.
    """
    path, truth = _local_roi(tmp_path, (37, 53), 16)
    chunks = {"northing": 16, "easting": 16}
    results = {}
    for scheduler in ("synchronous", "threads"):
        with dask.config.set(scheduler=scheduler):
            results[scheduler] = read_roi_mask(path, chunks).compute()
    assert np.array_equal(results["synchronous"], truth)
    assert np.array_equal(results["threads"], truth)
