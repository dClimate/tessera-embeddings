"""Does launch pacing actually cost less quota, and can it still grow a fleet?

Both questions need the same apparatus and neither can be answered by reading the
resolved YAML: what matters is how many launch requests reach the account's request
bucket and how many instances come back. So this drives **Ray's real AWS node
provider** — its client, its retry configuration, its subnet-rotating retry loop — at
a local stand-in for EC2 that enforces the real bucket (see
:mod:`._launch_pacing_sim`), once with the pacing environment and once without.

**The control is the point.** Every claim here is a comparison between two arms run
back to back on the same machine against the same simulated bucket, differing only in
the environment :data:`LAUNCH_PACING_ENV` sets. A single arm's numbers would say
nothing: they are as much a property of this laptop as of the change.

**What would sink the change.** Any of: the paced arm placing fewer instances than the
control, the paced arm not reducing throttled-per-instance, or the paced arm falling
silent while the bucket is shut. The third is the one to fear — a fleet that is
throttled retries, while a fleet that stops asking is stuck — so it is asserted
directly, against a bucket held closed, rather than argued from the code.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

from tessera_embeddings.providers.aws.ray import LAUNCH_PACING_ENV

from ._launch_pacing_sim import BUCKET_REFILL_PER_SEC

pytestmark = pytest.mark.integration

SIM = str(Path(__file__).parent / "_launch_pacing_sim.py")

#: Enough clusters to contend for one bucket, and enough nodes each that the control
#: arm's retries have room to pile up. Kept small: the separation is large, and this
#: has to stay a test rather than a benchmark.
CLUSTERS = 3
NODES_PER_CLUSTER = 75

#: How long the "bucket shut" arm refuses everything before reopening.
CLOSED_WINDOW_S = 6.0

#: botocore floors a paced client's send rate, so a fleet cannot be paced into silence.
#: The floor is per client and every cluster runs its own, so the aggregate floor scales
#: with the cluster count. Compared against with slack for the ramp before pacing arms
#: itself and for scheduling noise on a shared machine.
_FLOOR_PER_CLIENT_PER_S = 0.5
_FLOOR_SLACK = 0.6


@dataclass(frozen=True)
class Arm:
    """One run of the simulation."""

    calls: int
    accepted: int
    throttled: int
    placed: int
    wall_s: float
    closed_window_calls: int

    @property
    def throttled_per_instance(self) -> float:
        """Refusals paid per instance actually placed — the quota cost of a fleet.

        Per instance rather than per accepted call, because a call that places 25
        instances and a call that places 5 are not the same purchase.
        """
        assert self.placed, "an arm that placed nothing has no cost per instance"
        return self.throttled / self.placed


def _run_arm(*, paced: bool, closed_for: float = 0.0) -> Arm:
    """Run one arm: start the shared endpoint, then one process per cluster."""
    server = subprocess.Popen(
        [sys.executable, SIM, "serve"],
        env={"SIM_BUCKET_CLOSED": "1" if closed_for else "0", "PATH": "/usr/bin:/bin"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert server.stdout is not None
        port = json.loads(server.stdout.readline())["port"]
        env = {
            "SIM_PORT": str(port),
            "SIM_NODES": str(NODES_PER_CLUSTER),
            "PATH": "/usr/bin:/bin",
            **(dict(LAUNCH_PACING_ENV) if paced else {}),
        }
        started = time.monotonic()
        clusters = [
            subprocess.Popen(
                [sys.executable, SIM, "cluster"],
                env={**env, "SIM_INDEX": str(i)},
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for i in range(CLUSTERS)
        ]
        if closed_for:
            time.sleep(closed_for)
            urllib.request.urlopen(f"http://127.0.0.1:{port}/open", timeout=5).read()

        placed = 0
        for c in clusters:
            out, _ = c.communicate(timeout=300)
            placed += json.loads(out.strip().splitlines()[-1])["placed"] if out.strip() else 0
        wall = time.monotonic() - started

        stats = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=10).read())
        return Arm(
            calls=int(stats["calls"]),
            accepted=int(stats["accepted"]),
            throttled=int(stats["throttled"]),
            placed=placed,
            wall_s=wall,
            closed_window_calls=sum(1 for at, _ in stats["log"] if at <= closed_for) if closed_for else 0,
        )
    finally:
        if server.stdin is not None:
            server.stdin.close()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


@pytest.fixture(scope="module")
def arms() -> dict[str, Arm]:
    """Every arm, run once. Each is minutes of subprocesses; the assertions are cheap."""
    return {
        "control": _run_arm(paced=False),
        "paced": _run_arm(paced=True),
        "paced_shut_bucket": _run_arm(paced=True, closed_for=CLOSED_WINDOW_S),
    }


def test_pacing_costs_less_quota_per_instance_placed(arms: dict[str, Arm]) -> None:
    """The claim, stated as the ratio between the two arms rather than as one number."""
    control, paced = arms["control"], arms["paced"]
    assert paced.throttled_per_instance < control.throttled_per_instance / 2, (
        f"paced {paced.throttled_per_instance:.3f} vs control "
        f"{control.throttled_per_instance:.3f} refusals per instance placed"
    )


def test_pacing_makes_fewer_launch_requests_for_more_fleet(arms: dict[str, Arm]) -> None:
    """Fewer calls AND more instances. Either alone would be a trade rather than a win.

    Separate from the ratio above because the ratio could improve while total traffic
    grew, and total traffic is what the account's other callers actually feel.
    """
    control, paced = arms["control"], arms["paced"]
    assert paced.calls < control.calls, f"paced made {paced.calls} calls, control {control.calls}"
    assert paced.placed >= control.placed, (
        f"paced placed {paced.placed} instances, control {control.placed} — pacing must not cost fleet"
    )


def test_a_paced_fleet_keeps_asking_while_the_bucket_is_shut(arms: dict[str, Arm]) -> None:
    """The failure to fear, checked rather than reasoned about.

    Backoff that is too deep is worse than throttling: a refused request is retried,
    while a fleet that has stopped asking waits for nobody. botocore floors a paced
    client's send rate for exactly this reason, and each cluster runs its own client,
    so the aggregate floor scales with the cluster count.
    """
    shut = arms["paced_shut_bucket"]
    floor = _FLOOR_PER_CLIENT_PER_S * CLUSTERS * CLOSED_WINDOW_S * _FLOOR_SLACK
    assert shut.closed_window_calls >= floor, (
        f"{shut.closed_window_calls} calls in {CLOSED_WINDOW_S}s with the bucket shut, "
        f"expected at least {floor:.1f} — a paced fleet must not fall silent"
    )
    assert shut.placed > 0, "the fleet placed nothing after the bucket reopened"


def test_the_paced_arm_settles_near_the_buckets_refill_rate(arms: dict[str, Arm]) -> None:
    """Pacing should converge on the quota, not on something far below it.

    The upper bound is what says the clusters are not pacing each other into the
    ground; the lower bound is what says they are still using the quota they have.
    Loose on both sides — each cluster's limiter converges independently, so the
    aggregate lands above the refill rate rather than on it.
    """
    paced = arms["paced"]
    rate = paced.calls / paced.wall_s
    assert rate <= BUCKET_REFILL_PER_SEC * CLUSTERS * 2, f"paced arm called at {rate:.2f}/s"
    assert rate >= _FLOOR_PER_CLIENT_PER_S, f"paced arm called at {rate:.2f}/s, below one client's floor"
