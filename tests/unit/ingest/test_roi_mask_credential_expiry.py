"""The mask read inside a write's compute must resolve credentials when it reads.

``read_roi_mask`` returns a LAZY array. The write path computes it later, so a credential
resolved when the graph was built is the one the read actually presents — and once it
expires the read fails on a bucket our role can always read, which is how a radar leg died
mid-write.

These tests run the real path against a real S3: a moto server behind a front door that
records which access key signed each request and can refuse a nominated one exactly as S3
refuses an expired token. The two identities are separated at the botocore level the
credential helpers already rely on — the environment holds source-shaped credentials, and
the role is reachable only with the ``env`` provider removed.
"""

from __future__ import annotations

import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

from tessera_embeddings.ingest.roi import read_roi_mask
from tessera_embeddings.ingest.roi_processing import apply_roi_mask
from tessera_embeddings.ingest.s2_roi import _sum_over_windows
from tessera_embeddings.providers.aws.credentials import (
    _resolve_iam_credentials,
    iam_s3_storage_options,
)
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import write_day_windows

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
    issuer = tmp_path / "issue_role_credential.py"
    # UNIQUE BY CONSTRUCTION, not by a counter file. This process can run concurrently —
    # several dask threads reading blocks each ask for a credential — and a shared
    # read-modify-write counter races: one invocation creates the file, another reads it
    # empty and dies on `int('')`. A nanosecond clock needs no shared state and is still
    # monotonic, which is all the rotation assertions want.
    issuer.write_text(
        "import json, time\n"
        "from datetime import UTC, datetime, timedelta\n"
        "n = time.time_ns() % 10**12\n"
        "print(json.dumps({'Version': 1, 'AccessKeyId': f'AKIAROLE{n:012d}',\n"
        "                  'SecretAccessKey': 'role-secret', 'SessionToken': f'role-token-{n}',\n"
        "                  'Expiration': (datetime.now(UTC) + timedelta(seconds=2))\n"
        "                                .isoformat().replace('+00:00', 'Z')}))\n"
    )
    config = tmp_path / "aws-config"
    config.write_text(f"[default]\ncredential_process = {sys.executable} {issuer}\n")

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


def test_a_credential_that_expires_after_the_graph_is_built_does_not_fail_the_write(role_only_s3, tmp_path) -> None:
    """Graph build and compute are hours apart on a real leg; only the read's own moment counts."""
    door = role_only_s3
    mask = read_roi_mask("s3://test-bucket/roi.zarr", {"northing": 4, "easting": 4}, iam_s3_storage_options)
    assert door.signed_by, "building the graph must have read the store's metadata"
    # Everything valid while the graph was built has since expired.
    door.expired.update(door.signed_by)

    door.signed_by.clear()
    _write(str(tmp_path / "mosaic.zarr"), _masked_day(mask))

    assert door.signed_by, "the write path made no S3 request; the test proves nothing"
    assert not door.expired & set(door.signed_by), "the write presented a credential from graph-build time"


def test_the_mask_still_reads_the_right_pixels(role_only_s3) -> None:
    """A guard that corrupts the mask would be worse than the expiry it prevents."""
    mask = read_roi_mask("s3://test-bucket/roi.zarr", {"northing": 4, "easting": 4}, iam_s3_storage_options)
    assert mask.shape == (8, 8)
    assert mask.chunks == ((4, 4), (4, 4))
    assert mask.dtype == np.bool_
    assert mask.compute().all()


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
    masked = apply_roi_mask(day, "s3://test-bucket/roi.zarr", {"northing": 4, "easting": 4}, roi_mask=mask)

    _expire_everything_resolved_so_far(door)
    assert int(masked["band"].sum().compute()) == 7 * 64, "the mask read returned the wrong pixels"
    assert door.signed_by, "the compute made no S3 request; the test proves nothing"
    assert not door.expired & set(door.signed_by), "the radar masking step presented a graph-build credential"


def test_the_optical_window_total_survives_a_credential_that_expired_after_graph_build(role_only_s3) -> None:
    """SENTINEL-2. ``_sum_over_windows`` is the optical leg's own mask consumer.

    The optical path rebuilds its mask per date and reduces it over the live windows for
    the coverage gate — the first compute of every date, and the one that failed on the
    radar side for the same reason.
    """
    door = role_only_s3
    mask = read_roi_mask("s3://test-bucket/roi.zarr", {"northing": 4, "easting": 4}, iam_s3_storage_options)
    total = _sum_over_windows(mask, [(0, 4, 0, 8), (4, 8, 0, 8)])

    _expire_everything_resolved_so_far(door)
    assert int(total.compute()) == 64, "the window total read the wrong pixels"
    assert door.signed_by, "the compute made no S3 request; the test proves nothing"
    assert not door.expired & set(door.signed_by), "the optical window total presented a graph-build credential"


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
