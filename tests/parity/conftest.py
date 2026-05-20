"""Parity-test shared fixtures.

Fixtures here are session-scoped where possible (LocalCluster +
prefect_test_harness are expensive to set up). Each parity test
gets isolation via per-function ``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from dask.distributed import Client, LocalCluster
from prefect.testing.utilities import prefect_test_harness

REPO_ROOT = Path(__file__).resolve().parents[2]
CASSETTE_DIR = REPO_ROOT / "tests" / "fixtures" / "stac_cassettes"


@pytest.fixture(scope="session", autouse=True)
def _isolated_prefect_runtime() -> Iterator[None]:
    """Run every parity-test flow against an in-memory Prefect server.

    Without this, Prefect uses the contributor's shell-configured
    ``PREFECT_API_URL`` (Cloud, staging, prod, …) — leaking external
    infrastructure into a test that's supposed to be hermetic.

    ``prefect_test_harness`` overrides settings with a temp SQLite
    database for the lifetime of the context manager. ``autouse=True``
    + session scope means every parity test inherits it without
    extra fixture wiring.
    """
    with prefect_test_harness():
        yield


@pytest.fixture(scope="session")
def fixture_quickstart_roi() -> Path:
    """Denver, CO GeoJSON used by the parity-test cassettes."""
    path = REPO_ROOT / "examples" / "quickstart" / "roi.geojson"
    assert path.exists(), f"Missing quickstart fixture at {path}"
    return path


@pytest.fixture(scope="session")
def parity_cluster() -> Iterator[Client]:
    """Session-scoped LocalCluster + connected client for parity tests.

    One worker per test is enough — parity tests are correctness
    checks, not performance checks. Keeping the cluster small avoids
    OOM on CI runners and keeps test wall-clock predictable.
    """
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        memory_limit="2GB",
        dashboard_address=None,  # off — saves an aiohttp dep at test time
    )
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


# ── pytest-recording configuration ────────────────────────────────


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    """Tell pytest-recording where to read/write cassettes.

    Default is alongside the test file; we centralise under
    ``tests/fixtures/stac_cassettes/`` so cassettes are reviewable as
    a unit and the safety guard
    (``tests/architecture/test_cassette_safety.py``) has one place to
    look.
    """
    return str(CASSETTE_DIR)


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    """Filter known credential headers + query params from cassettes.

    Belt-and-braces with the safety guard: even if a leak slips
    through here, the architecture test catches it before merge.

    ``ignore_hosts`` excludes hosts from cassette capture entirely.
    Two reasons we do this:

    * **Prefect API** — :func:`prefect_test_harness` spins up an
      ephemeral Prefect server on ``127.0.0.1:<random>``. Without
      ``ignore_localhost`` + ``ignore_hosts``, every Prefect API
      call (admin/version checks, flow-run state writes, etc.) would
      get baked into the cassette. Then on replay the random port
      doesn't match and the cassette rejects the request.
    * **Contributors with PREFECT_API_URL set** — recordings made
      with a shell-configured Prefect URL would bake that URL into
      the cassette. ``ignore_hosts`` keeps it out regardless of
      whether the harness or the env var is in effect.

    The cassette captures only the STAC HTTP we care about
    (Earth Search, CMR-STAC, CMR Granule Search). Everything else
    flows through normally.
    """
    return {
        "filter_headers": [
            "authorization",
            "cookie",
            "set-cookie",
            "x-amz-security-token",
        ],
        "filter_query_parameters": [
            "X-Amz-Signature",
            "X-Amz-Security-Token",
            "X-Amz-Credential",
        ],
        "decode_compressed_response": True,
        "ignore_localhost": True,  # never capture 127.0.0.1 (Prefect harness, Dask)
        "ignore_hosts": [
            # Catch Prefect deployments that shell-set PREFECT_API_URL to
            # an external host so cassettes stay portable across contributors.
            "cyclops-prefect-staging.up.railway.app",
            "cyclops-prefect.up.railway.app",
            "api.prefect.cloud",
        ],
    }
