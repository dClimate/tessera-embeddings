"""The ROI mask: its credentials, its construction, and its reads after a credential expires.

The mask lives in our own bucket, which our role can always read — but the radar path injects
the SOURCE's short-lived token into the process environment for GDAL, so any fsspec read that
resolves credentials from the environment picks up the wrong one. The mask's storage options
are therefore a *provider*, resolved when a block is READ: ``read_roi_mask`` returns a lazy
array built with ``da.map_blocks`` over a closure, and the write path computes it hours later.

A. **Credentials.** The provider is invoked on every call, and the callers pass the provider
   rather than its frozen result.
B. **Construction.** The closure builds a different graph from ``da.from_zarr``, so it must
   still be the pixels that were written (``block_info`` slicing at ragged edges), still lazy,
   and still survive a process-based scheduler.
C. **Expiry.** Against a real S3 (moto behind a front door that records which access key
   signed each request and refuses a nominated one as S3 refuses an expired token), a
   credential that died after the graph was built must not fail the read.

The per-block request cost of the closure, and the fact that the graph is cloudpickle-only,
live in ``context_docs/decisions/022-resolve-the-roi-mask-credential-at-read-time.md``.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import functools
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import boto3
import dask.array as da
import numpy as np
import pytest
import s3fs
import xarray as xr
import zarr
import zarr.storage
from affine import Affine
from odc.geo.geobox import GeoBox

from tessera_embeddings.ingest.roi import read_roi_mask, resolve_storage_options
from tessera_embeddings.ingest.roi_processing import apply_roi_mask
from tessera_embeddings.providers.aws.credentials import _resolve_iam_credentials, iam_s3_storage_options
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import write_day_windows
from tests._paths import SRC_ROOT

_SRC = SRC_ROOT


# ===========================================================================
# A. Credentials: a provider, resolved per call
# ===========================================================================


def test_a_provider_is_invoked_on_every_call_never_memoised() -> None:
    """The whole point: two reads far apart must not share one resolution.

    A provider that were called once and remembered would pass any test that only checked
    the first value, and would reproduce exactly the expiry this change exists to fix.
    """
    seen: list[int] = []

    def provider() -> dict:
        seen.append(len(seen))
        return {"key": f"cred-{len(seen)}"}

    first = resolve_storage_options(provider)
    second = resolve_storage_options(provider)
    assert first == {"key": "cred-1"}
    assert second == {"key": "cred-2"}, "the provider must be re-invoked, not memoised"
    assert len(seen) == 2


class TestTheCallersPassAProviderAndReResolve:
    """Source-level checks: the wiring is what regressed, not the helper."""

    def test_the_task_layer_passes_the_callable_not_its_result(self) -> None:
        """``iam_s3_storage_options()`` — with parentheses — is the frozen snapshot."""
        src = (_SRC / "orchestration" / "prefect" / "tasks" / "ingest.py").read_text()
        assert "storage_options = iam_s3_storage_options\n" in src, (
            "the task must pass the provider itself, so each read resolves afresh"
        )
        assert "storage_options = iam_s3_storage_options()" not in src, (
            "calling it here freezes the credential for the whole leg — the original defect"
        )

    def test_the_optical_path_rebuilds_its_mask_per_date(self) -> None:
        """One graph reused across a leg carries one credential across a leg.

        Checked by CONTAINMENT: the read must be inside the per-date function, not merely
        present in the file, since the leg-entry read for the pixel total is legitimate.
        """
        tree = ast.parse((_SRC / "ingest" / "s2_roi.py").read_text())
        prepare = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_prepare_date")
        )
        reads = [
            node
            for node in ast.walk(prepare)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "read_roi_mask"
        ]
        assert reads, "the per-date path must build its own mask graph, not reuse the leg's"
        for call in reads:
            assert any(kw.arg == "storage_options" for kw in call.keywords), (
                f"the per-date mask read at line {call.lineno} must pass storage_options — "
                "without them fsspec resolves the environment, which holds the SOURCE token"
            )

    def test_no_mask_read_omits_storage_options(self) -> None:
        """A read with no options resolves the poisoned environment and eventually fails."""
        offenders = []
        for module in ("s1_roi.py", "s2_roi.py"):
            tree = ast.parse((_SRC / "ingest" / module).read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                if node.func.id != "read_roi_mask":
                    continue
                if not any(kw.arg == "storage_options" for kw in node.keywords):
                    offenders.append(f"{module}:{node.lineno}")
        assert not offenders, f"mask read(s) with no storage_options: {offenders}"


# ===========================================================================
# B. Construction: same pixels, still lazy, survives a process scheduler
# ===========================================================================


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
        pytest.param((10, 10), 64, id="chunk-larger-than-array"),
        pytest.param((1, 40), 16, id="single-row"),
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


@pytest.fixture
def process_client():
    """A single-worker PROCESS cluster, closed after its test.

    Function-scoped on purpose: while a Client is open it is dask's default scheduler, so a
    cluster outliving its test would run section C's computes on a worker that never saw that
    test's monkeypatched credential environment.
    """
    distributed = pytest.importorskip("distributed")
    with (
        distributed.LocalCluster(
            n_workers=1, threads_per_worker=1, processes=True, dashboard_address=None, silence_logs=50
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        yield client


def test_the_closure_survives_a_process_based_scheduler(process_client, roi_on_s3, s3_chunks) -> None:
    """PROCESSES, not threads: the only scheduler that actually serializes the graph.

    The old graph shipped a zarr array holding a filesystem object. The new one ships a
    nested function plus its captured variables, which plain pickle cannot do at all — it
    survives only because distributed falls back to cloudpickle. A threaded or synchronous
    scheduler shares memory and would prove nothing about that.

    Run with a callable provider, the stricter storage-option form: it has to arrive on the
    worker still callable for the credential to be resolved there rather than here.
    """
    url, options, truth = roi_on_s3
    mask = read_roi_mask(url, s3_chunks, functools.partial(dict, **options))
    computed = process_client.compute(mask, sync=True)

    assert np.array_equal(computed, truth), "the mask read on a worker process differs from what was written"


# ---------------------------------------------------------------------------
# 4. Both sensors (source-level half; the behavioural half is in section C)
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


# ===========================================================================
# C. Expiry: a credential that died after graph build
# ===========================================================================


#: Shaped like the source's STS credentials the radar path puts in ``os.environ`` for GDAL.
SOURCE_KEY = "ASIASOURCETOKEN00000"

_CREDENTIAL = re.compile(r"Credential=([^/]+)/")
_EXPIRED_TOKEN = (
    b'<?xml version="1.0" encoding="UTF-8"?><Error><Code>ExpiredToken</Code>'
    b"<Message>The provided token has expired.</Message></Error>"
)

GEOBOX = GeoBox((8, 8), Affine(10.0, 0.0, 0.0, 0.0, -10.0, 80.0), "EPSG:32601")
CHUNKS = {"time": 1, "northing": 4, "easting": 4}


class _Roi:
    geobox = GEOBOX
    height = 8
    width = 8


class _FrontDoor:
    """An S3 front door that records the signing key and can refuse chosen ones."""

    def __init__(self, upstream: str) -> None:
        self.signed_by: list[str] = []
        self.expired: set[str] = set()
        door = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:  # keep pytest output clean
                pass

            def _relay(self) -> None:
                match = _CREDENTIAL.search(self.headers.get("Authorization", "") or "")
                key = match.group(1) if match else "<unsigned>"
                door.signed_by.append(key)
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                if key in door.expired:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/xml")
                    self.send_header("Content-Length", str(len(_EXPIRED_TOKEN)))
                    self.end_headers()
                    self.wfile.write(_EXPIRED_TOKEN)
                    return
                request = urllib.request.Request(
                    upstream + self.path,
                    data=body,
                    method=self.command,
                    headers={k: v for k, v in self.headers.items() if k.lower() != "host"},
                )
                try:
                    with urllib.request.urlopen(request) as response:
                        payload, status, headers = response.read(), response.status, response.headers
                except urllib.error.HTTPError as exc:
                    payload, status, headers = exc.read(), exc.code, exc.headers
                self.send_response(status)
                for name, value in headers.items():
                    if name.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_PUT = do_POST = do_HEAD = do_DELETE = _relay  # noqa: N815 — http.server's names

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        self._server.shutdown()


@pytest.fixture
def role_only_s3(moto_server, monkeypatch, tmp_path):
    """A bucket holding an ROI mask, reachable only as the role, behind the front door.

    The role's credentials rotate and expire in seconds, so every resolution is visibly a
    different key: what a read presents says exactly when it resolved.
    """
    issuer = tmp_path / "issue_role_credential.sh"
    # UNIQUE BY CONSTRUCTION, not by a counter file: several dask threads ask for a credential
    # at once, and a shared read-modify-write counter races. Seconds plus this invocation's PID
    # are unique across concurrent invocations, which is all the assertions need. A shell script,
    # not Python: botocore refreshes a credential this close to expiry on every request, and an
    # interpreter start per request made this file the suite's slowest for its size.
    issuer.write_text(
        "#!/bin/sh\n"
        "t=$(date +%s); n=$(( (t * 100000 + $$) % 1000000000000 )); e=$((t + 2))\n"
        'exp=$(date -u -d "@$e" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "$e" +%Y-%m-%dT%H:%M:%SZ)\n'
        'printf \'{"Version": 1, "AccessKeyId": "AKIAROLE%012d", "SecretAccessKey": "role-secret", '
        '"SessionToken": "role-token-%d", "Expiration": "%s"}\\n\' "$n" "$n" "$exp"\n'
    )
    issuer.chmod(0o755)
    config = tmp_path / "aws-config"
    config.write_text(f"[default]\ncredential_process = {issuer}\n")

    # The moto server is module-scoped, so this runs against a bucket a sibling test may
    # already have made; both steps are written to be repeatable.
    setup = boto3.client(
        "s3",
        endpoint_url=moto_server,
        aws_access_key_id="setup",
        aws_secret_access_key="setup",
        region_name="us-east-1",
    )
    if "test-bucket" not in {b["Name"] for b in setup.list_buckets()["Buckets"]}:
        setup.create_bucket(Bucket="test-bucket")
    store = zarr.storage.FsspecStore.from_url(
        "s3://test-bucket/roi.zarr",
        storage_options={"key": "setup", "secret": "setup", "client_kwargs": {"endpoint_url": moto_server}},
    )
    zarr.create_array(store=store, shape=(8, 8), chunks=(4, 4), dtype="bool", overwrite=True)[:] = True

    door = _FrontDoor(moto_server)
    # Credentials only: stripping the ``env`` provider does not hide the endpoint or region.
    monkeypatch.setenv("AWS_ENDPOINT_URL", door.url)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-such-credentials"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", SOURCE_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "source-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "source-token")
    _resolve_iam_credentials.cache_clear()
    s3fs.S3FileSystem.clear_instance_cache()
    try:
        yield door
    finally:
        door.stop()
        _resolve_iam_credentials.cache_clear()
        s3fs.S3FileSystem.clear_instance_cache()


def _masked_day(mask: da.Array) -> xr.Dataset:
    """One lazy date whose every variable depends on the mask read."""
    shape = (1, 8, 8)
    day = xr.Dataset(
        {
            "band": (("time", "northing", "easting"), da.full(shape, 7, dtype=np.uint16, chunks=(1, 4, 4))),
            "scl": (("time", "northing", "easting"), da.full(shape, 4, dtype=np.uint8, chunks=(1, 4, 4))),
        },
        coords={"time": np.array([np.datetime64("2024-06-01", "ns")])},
    )
    for var in ("band", "scl"):
        day[var] = day[var].where(mask[np.newaxis, :, :], other=0)
    return day


def _write(store: str, day: xr.Dataset) -> None:
    write_day_windows(
        store,
        day,
        [(0, 4, 0, 8)],
        roi=_Roi(),
        manifest=IngestManifest(roi_manifest_hash="abc"),
        baselines={},
        tile_id="roi.zarr",
        crs="EPSG:32601",
        chunks=CHUNKS,
        parallel_windows=True,
    )


def test_the_write_signs_as_the_role_never_as_the_environment(role_only_s3, tmp_path) -> None:
    """The source's token is in the environment and must never reach our own bucket.

    A REGRESSION GUARD rather than a proof of this change: it passes on ``main`` too,
    because threading the provider into the leg was already enough to keep the environment
    out. It is here so that a later change cannot quietly go back to resolving it.

    The source key is refused as expired, so picking it up is a failed write rather than a
    silent difference — the same refusal the leg met, from the same place.
    """
    door = role_only_s3
    door.expired.add(SOURCE_KEY)
    mask = read_roi_mask("s3://test-bucket/roi.zarr", {"northing": 4, "easting": 4}, iam_s3_storage_options)

    door.signed_by.clear()
    _write(str(tmp_path / "mosaic.zarr"), _masked_day(mask))

    assert door.signed_by, "the write path made no S3 request; the test proves nothing"
    assert SOURCE_KEY not in door.signed_by
    assert all(key.startswith("AKIAROLE") for key in door.signed_by), door.signed_by


# ---------------------------------------------------------------------------
# One test per sensor: both paths read through this function, so both must survive
# a credential that died after their graph was built. The consumers below are the
# real ones — the radar path's masking step and the optical path's window total.
# ---------------------------------------------------------------------------


def _expire_everything_resolved_so_far(door: _FrontDoor) -> None:
    """Kill every credential the graph build presented, then start recording afresh."""
    assert door.signed_by, "building the graph must have read the store's metadata"
    door.expired.update(door.signed_by)
    door.signed_by.clear()


def test_the_radar_masking_step_survives_a_credential_that_expired_after_graph_build(role_only_s3) -> None:
    """SENTINEL-1. ``apply_roi_mask`` is what the radar leg does with ``batch_mask``.

    ``s1_roi._prepare_batch`` builds one mask per 30-day batch and hands it to
    ``apply_roi_mask``, whose output is computed once per date for as long as the batch's
    writes take. This is that shape: build, expire, compute.
    """
    door = role_only_s3
    mask = read_roi_mask("s3://test-bucket/roi.zarr", {"northing": 4, "easting": 4}, iam_s3_storage_options)
    day = xr.Dataset(
        {"band": (("time", "northing", "easting"), da.full((1, 8, 8), 7, dtype=np.uint16, chunks=(1, 4, 4)))},
        coords={"time": np.array([np.datetime64("2024-06-01", "ns")])},
    )
    masked = apply_roi_mask(day, mask)

    _expire_everything_resolved_so_far(door)
    assert int(masked["band"].sum().compute()) == 7 * 64, "the mask read returned the wrong pixels"
    assert door.signed_by, "the compute made no S3 request; the test proves nothing"
    assert not door.expired & set(door.signed_by), "the radar masking step presented a graph-build credential"


def test_a_second_compute_of_one_graph_re_resolves_again(role_only_s3) -> None:
    """The radar shape exactly: ONE graph, computed date after date, expiring in between.

    The batch mask is not rebuilt between dates, so it is not enough that the first
    compute after expiry works — every later one must re-resolve too. A closure that
    cached its opened store would pass the single-compute tests above and still die on
    date two.
    """
    door = role_only_s3
    mask = read_roi_mask("s3://test-bucket/roi.zarr", {"northing": 4, "easting": 4}, iam_s3_storage_options)

    for date in range(3):
        _expire_everything_resolved_so_far(door)
        # NO cache_clear here, deliberately. Botocore refreshes the cached credential
        # underneath — the issuer's two-second expiry puts every read inside the mandatory
        # refresh window — which is what production does, and it serialises the refresh
        # behind botocore's own lock. Clearing the cache instead forces a COLD resolve in
        # every dask thread at once, which is a property of the test, not of the code.
        assert mask.sum().compute() == 64, f"date {date} read the wrong pixels"
        assert door.signed_by, f"date {date} made no S3 request"
        assert not door.expired & set(door.signed_by), f"date {date} presented an expired credential"
