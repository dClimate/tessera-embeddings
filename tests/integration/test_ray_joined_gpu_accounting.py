"""The Ray resource semantics the progress line's GPU-hours figure rests on.

``_joined_gpu_count`` reports fleet GPU-hours from ``ray.cluster_resources()``.
That is only the billed capacity if the total does NOT shrink as actors claim
GPUs — otherwise the figure would collapse towards zero exactly when the fleet
is busiest, and the whole approach would be wrong. The sibling call,
``available_resources()``, behaves the opposite way and is the easy mistake to
make here.

Neither property is ours to define, so the unit suite cannot pin them: it mocks
every Ray call, and a mock asserting the semantics we assumed would prove
nothing. This test runs a real Ray instance instead.

No GPU is needed. ``num_gpus`` is a scheduling resource that Ray will hand out
on a machine with no accelerator at all, and it is the accounting we are
checking, not any device.

The GPU is claimed with a PLACEMENT GROUP rather than an actor. A placement group
is a reservation the raylet satisfies on its own, so nothing has to start a
Python worker process — which is the step that has to succeed for an actor, and
the step a small runner cannot always manage. The reservation exercises the same
accounting, with strictly fewer moving parts.

What is deliberately NOT here: the created-but-unplaced actor. Locally the only
way to make an actor unplaceable is to claim every declared GPU, which is not
the shape of the real case (there the node does not exist yet), and an actor
that can never be placed blocks ``ray.shutdown`` — a hang, in a suite where one
would burn the whole job. That case is pinned in the unit suite instead, where
the pool's requested-slot count and the cluster's GPU count can be set apart
freely.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import ray
from ray.util.placement_group import placement_group, remove_placement_group

from tessera_embeddings.inference.scheduling import _joined_gpu_count

DECLARED_GPUS = 4


@pytest.fixture
def ray_instance() -> Iterator[None]:
    """A local Ray instance declaring GPUs that no hardware here provides.

    The temp dir is short and outside any home directory on purpose: Ray puts
    its plasma-store socket under it, and a UNIX socket path over 103 bytes
    fails to bind. It is removed with the session, and the instance is shut
    down explicitly — a leaked Ray session leaves a ``gcs_server`` behind.
    """
    tmp = tempfile.mkdtemp(prefix="rt-", dir="/tmp")
    try:
        ray.init(
            num_cpus=2,
            num_gpus=DECLARED_GPUS,
            include_dashboard=False,
            log_to_driver=False,
            _temp_dir=tmp,
        )
        try:
            yield
        finally:
            ray.shutdown()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        assert not Path(tmp).exists()


def _wait_for_available_gpus_below(total: float, timeout: float = 30.0) -> bool:
    """Poll until the unclaimed GPU count falls below ``total``.

    ``available_resources`` is eventually consistent: a claim registered by a
    returned actor call is not necessarily reflected the moment the call
    returns, so asserting on it directly is a race. Polling is not a
    convenience here — it is the difference between pinning the distinction and
    shipping a flaky test.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ray.available_resources().get("GPU", 0) < total:
            return True
        time.sleep(0.25)
    return False


@pytest.mark.integration
def test_cluster_total_holds_while_gpus_are_claimed(ray_instance: None) -> None:
    """The total is billed capacity; the available remainder is not.

    If ``cluster_resources`` ever started reporting the unclaimed remainder,
    the progress line would under-report a busy fleet — so this pins the total
    as flat across claiming, and pins ``available_resources`` as the call that
    does move, since confusing the two is the whole risk.
    """
    assert _joined_gpu_count(0.0, 1.0) == DECLARED_GPUS
    assert ray.available_resources().get("GPU", 0) == DECLARED_GPUS

    group = placement_group([{"GPU": 1}, {"GPU": 1}], strategy="PACK")
    try:
        ray.get(group.ready(), timeout=60)

        assert _joined_gpu_count(0.0, 1.0) == DECLARED_GPUS, "a claimed GPU is still a joined GPU"
        assert _wait_for_available_gpus_below(DECLARED_GPUS), (
            "available_resources must be the one that drops — if it stops dropping, the two calls "
            "have converged and the distinction this rests on is gone"
        )
        # And the total is still flat once the claim HAS registered, which is the
        # property being pinned: the two calls disagree, and we read the stable one.
        assert _joined_gpu_count(0.0, 1.0) == DECLARED_GPUS
    finally:
        remove_placement_group(group)
