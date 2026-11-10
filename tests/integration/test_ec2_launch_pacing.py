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
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

from tessera_embeddings.providers.aws.ray import (
    LAUNCH_PACING_AUTOSCALER_ENV,
    LAUNCH_PACING_CLIENT_ENV,
    LAUNCH_PACING_ENV,
)

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
    """Every arm, run once. Each is minutes of subprocesses; the assertions are cheap.

    Opt-in through ``TESSERA_TEST_LAUNCH_PACING=1``: pacing is off by default, and minutes
    of subprocesses on every source change is past this directory's two-minute rule.
    """
    if not os.environ.get("TESSERA_TEST_LAUNCH_PACING"):
        pytest.skip("set TESSERA_TEST_LAUNCH_PACING=1 to run the launch-pacing simulation")
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


#: The name under test, so the baseline below removes exactly the right key.
_INTERVAL = "AUTOSCALER_UPDATE_INTERVAL_S"


def _ray_autoscaler_interval(override: str | None) -> int:
    """What RAY resolves the interval to in a fresh interpreter: ``override``, or its own default.

    A subprocess because the constant is bound at module import: reading it in-process after
    monkeypatching the environment would report the value this test session imported with, not
    the value a freshly-started autoscaler would see. The autoscaler IS a fresh process, so this
    reproduces its conditions rather than approximating them.

    **The baseline STRIPS the key rather than inheriting the environment**, which is why this
    takes an override rather than a dict. A developer machine or CI runner configured with launch
    pacing already exports this name, and an inherited environment would then measure that
    override and call it Ray's default — failing the comparison below on precisely the hosts the
    setting exists for. Correct code, red test.
    """
    # Imports ONE constants module and prints an int. No `ray.init()`, no `ray start`, no
    # cluster — this repo has stranded a Ray cluster from a test before and the RAM cost was
    # felt for days, so the subprocess is deliberately the narrowest thing that can answer.
    code = "from ray.autoscaler._private.constants import AUTOSCALER_UPDATE_INTERVAL_S as v; print(v)"
    env = {k: v for k, v in os.environ.items() if k != _INTERVAL}
    if override is not None:
        env[_INTERVAL] = override
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    return int(out.stdout.strip())


def test_ray_actually_reads_the_interval_from_the_environment() -> None:
    """The setting is only pacing if RAY reads it, and that is Ray's contract to keep, not ours.

    Asserting our own dict against a default copied out of Ray proves nothing about Ray — it
    passes whether or not the name is environment-backed. So this asks Ray, in a fresh
    interpreter, and compares the answer with and without the variable set.

    **This is the regression that would make the whole setting inert**: if a future Ray pins the
    interval as a plain module constant, the value we export would be silently ignored while the
    export still reads like tuning. That failure is invisible everywhere else.
    """
    ours = int(LAUNCH_PACING_AUTOSCALER_ENV[_INTERVAL])
    default = _ray_autoscaler_interval(None)
    applied = _ray_autoscaler_interval(str(ours))
    assert applied == ours, (
        f"Ray resolved AUTOSCALER_UPDATE_INTERVAL_S to {applied} with the environment set to "
        f"{ours}; the setting is not environment-backed in this Ray and paces nothing"
    )
    assert ours > default, (
        f"our value {ours} is at or below Ray's own default {default}, so it slows nothing even though Ray reads it"
    )
    assert _INTERVAL not in LAUNCH_PACING_CLIENT_ENV, (
        "`ray up` runs no autoscaler loop, so the name is inert in the one-shot bootstrap"
    )
