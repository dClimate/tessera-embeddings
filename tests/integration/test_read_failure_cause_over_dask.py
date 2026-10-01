"""The cause's survival, measured across a real Dask cluster rather than simulated in-process.

And across Frisky loaded onto the same cluster (``providers/frisky.py``), which loses the cause by
a different route and needs its own rescue on top of this one.

Every other test for this flattens the exception itself and then proves the rescue undoes its own
flattening. That is the shape of the defect it exists to prevent: the original regression shipped
because a hand-made ``RuntimeError`` stood in for what production actually sends. These tests
raise the real classes on a spawned worker and read what the client receives.

Spawned workers are the point. A spawned interpreter cannot inherit a patched class from the
driver, which is the condition of a production container — under ``fork`` the rescue would appear
to work while proving nothing about whether it is installed where reads happen.

Marked ``integration``: each test builds a cluster, so they cost seconds rather than milliseconds
and are excluded from the default run.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from dask.distributed import Client

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


@pytest.fixture
def spawned_cluster():
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


@pytest.fixture(params=["dask", "frisky"])
def engine_client(request: pytest.FixtureRequest, spawned_cluster: Client) -> Iterator[Client]:
    """The ingest's client on the spawned cluster: Dask's own, or Frisky's after a hijack."""
    if request.param == "dask":
        yield spawned_cluster
        return
    pytest.importorskip("frisky")
    from tessera_embeddings.providers.frisky import connect, hijack

    hijack(spawned_cluster)
    with connect(spawned_cluster) as client:
        yield client


@pytest.mark.integration
def test_frisky_destroys_the_cause_until_the_worker_plugin_is_installed(spawned_cluster: Client) -> None:
    """Frisky's own A/B, and the test that says when its workaround can go.

    Frisky returns a task's exception by plain pickle, which drops ``__cause__`` even with this
    module's rescue in place. If the first half stops reproducing, an upgrade fixed it, and
    the tblib step of ``_MatchDaskWorker`` in ``providers/frisky.py`` is a candidate for deletion.
    """
    frisky = pytest.importorskip("frisky")
    from tessera_embeddings.ingest.loader_failures import install_capture_everywhere
    from tessera_embeddings.providers.frisky import _MatchDaskWorker, connect

    codec = "ZIPDecode:Decoding error at scanline 0, unknown compression method"
    install_capture_everywhere(spawned_cluster)
    frisky.hijack(spawned_cluster, connect_client=False)

    with connect(spawned_cluster) as client:
        with pytest.raises(BaseException) as before:
            client.submit(_raise_with, codec).result()
        assert codec not in _chain_text(before.value), "Frisky now keeps the cause: the tblib step may be removable"

        spawned_cluster.register_plugin(_MatchDaskWorker())
        with pytest.raises(BaseException) as after:
            client.submit(_raise_with, codec).result()
    chain = _chain_text(after.value)
    assert codec in chain, chain
    assert "CPLE_AppDefinedError" in chain, chain


@pytest.mark.integration
@pytest.mark.parametrize(
    ("text", "gives_up_the_date"),
    [
        ("ZIPDecode:Decoding error at scanline 0, unknown compression method", True),
        ("TIFFReadEncodedTile() failed", True),
        (ASF_DENIED, False),
        ("HTTP response code: 503", False),
        ("HTTP response code: 429", False),
        ("HTTP response code: 401", False),
        ("Connection reset by peer", False),
    ],
)
def test_restoring_the_cause_does_not_widen_what_gets_skipped(
    engine_client: Client, text: str, gives_up_the_date: bool
) -> None:
    """The control that matters, because this change inverts the risk.

    A missing cause used to BLOCK the destructive verdict: too conservative, cells stranded, but
    nothing lost. Restoring it means a skip can fire, so the question is not whether corrupt bytes
    now skip — it is whether a transient still refuses to. ``ASF_DENIED`` is the verbatim refusal
    seen in production, which arrived as an undecidable wrapper at the time.

    Run on both engines, because the verdict must not depend on which one carried the failure.
    Frisky's ``submit`` takes no ``pure`` and keys every call uniquely, so neither passes it; each
    case submits once to a fresh cluster.
    """
    from tessera_embeddings.ingest.duplicates import is_unreadable_source
    from tessera_embeddings.ingest.loader_failures import install_capture_everywhere

    install_capture_everywhere(engine_client)

    with pytest.raises(BaseException) as raised:
        engine_client.submit(_raise_with, text).result()
    assert is_unreadable_source(raised.value) is gives_up_the_date, _chain_text(raised.value)
