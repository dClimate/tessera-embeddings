"""The cause's survival, measured across a real Dask cluster rather than simulated in-process.

Every other test for this flattens the exception itself and then proves the rescue undoes its own
flattening. That is the shape of the defect it exists to prevent: the original regression shipped
because a hand-made ``RuntimeError`` stood in for what production actually sends. These tests
raise the real classes on a spawned worker and read what the client receives.

Spawned workers are the point. A spawned interpreter cannot inherit a patched class from the
driver, which is the condition of a production container — under ``fork`` the rescue would appear
to work while proving nothing about whether it is installed where reads happen.

Marked ``integration``: the clusters cost seconds rather than milliseconds
and are excluded from the default run.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest

ASF_DENIED = (
    "AccessDenied: User: arn:aws:sts::510296831643:assumed-role/"
    "asf-cumulus-prod-thin-egress-app-2-DownloadRoleInRegion/arbol_global_tessera@ is not "
    "authorized to perform: s3:GetObject on resource: "
    '"arn:aws:s3:::asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/OPERA_L2_RTC-S1_T124.tif"'
)


def _raise_with(text: str) -> None:
    """Raise what a failing read raises: a rasterio wrapper over a GDAL cause."""
    from rasterio._err import CPLE_AppDefinedError
    from rasterio.errors import RasterioIOError

    try:
        raise CPLE_AppDefinedError(3, 5, text)
    except CPLE_AppDefinedError as exc:
        raise RasterioIOError("Read failed. See previous exception for details.") from exc


def _chain_text(exc: BaseException) -> str:
    parts, seen, cur = [], set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " | ".join(parts)


@contextlib.contextmanager
def _spawned_client() -> Iterator[Any]:
    """A cluster whose workers are fresh processes, so nothing is inherited from this one."""
    from dask.distributed import Client, LocalCluster

    with (
        LocalCluster(
            n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None, silence_logs=50
        ) as cluster,
        Client(cluster) as client,
    ):
        assert set(client.run(lambda: __import__("multiprocessing").get_start_method()).values()) == {"spawn"}, (
            "workers must be spawned, or an inherited patch makes these tests vacuous"
        )
        yield client


@pytest.fixture
def spawned_cluster() -> Iterator[Any]:
    """A fresh cluster with no rescue installed, for the A/B."""
    with _spawned_client() as client:
        yield client


@pytest.fixture(scope="module")
def rescued_cluster() -> Iterator[Any]:
    """One spawned cluster with the rescue installed, shared by the verdict cases."""
    from tessera_embeddings.ingest.loader_failures import install_capture_everywhere

    with _spawned_client() as client:
        install_capture_everywhere(client)
        yield client


@pytest.mark.integration
def test_the_cause_is_destroyed_without_the_rescue_and_survives_with_it(spawned_cluster) -> None:
    """The A/B that establishes the defect before crediting the fix.

    Without the rescue the client receives a plain ``Exception`` carrying the wrapper's repr and
    no cause — the exact shape observed in production. If this half ever stops reproducing, the
    other half proves nothing and this test says so rather than passing quietly.
    """
    from tessera_embeddings.ingest.loader_failures import install_capture_everywhere

    codec = "ZIPDecode:Decoding error at scanline 0, unknown compression method"

    with pytest.raises(BaseException) as before:
        spawned_cluster.submit(_raise_with, codec, pure=False).result()
    assert codec not in _chain_text(before.value), (
        "the cause survived without the rescue, so this environment does not reproduce the defect"
    )

    install_capture_everywhere(spawned_cluster)

    with pytest.raises(BaseException) as after:
        spawned_cluster.submit(_raise_with, codec, pure=False).result()
    chain = _chain_text(after.value)
    assert codec in chain, chain
    assert "CPLE_AppDefinedError" in chain, chain


@pytest.mark.integration
@pytest.mark.parametrize(
    ("text", "gives_up_the_date"),
    [
        ("ZIPDecode:Decoding error at scanline 0, unknown compression method", True),
        (ASF_DENIED, False),
    ],
)
def test_restoring_the_cause_does_not_widen_what_gets_skipped(
    rescued_cluster, text: str, gives_up_the_date: bool
) -> None:
    """The control that matters, because this change inverts the risk.

    A missing cause used to BLOCK the destructive verdict: too conservative, cells stranded, but
    nothing lost. Restoring it means a skip can fire, so the question is not whether corrupt bytes
    now skip — it is whether a transient still refuses to. ``ASF_DENIED`` is the verbatim refusal
    seen in production, which arrived as an undecidable wrapper at the time. The verdict for each
    message is tested in ``tests/unit/ingest/test_read_failure_verdict.py``; here the cluster only
    has to deliver the cause for one skip and one refusal.
    """
    from tessera_embeddings.ingest.duplicates import is_unreadable_source

    with pytest.raises(BaseException) as raised:
        rescued_cluster.submit(_raise_with, text, pure=False).result()
    assert is_unreadable_source(raised.value) is gives_up_the_date, _chain_text(raised.value)
