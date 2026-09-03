"""The scheduler-plugin registration wiring, against a real scheduler.

The unit tests for :class:`SchedulerResourceLogger` drive it through a fake scheduler object,
so they pin what the plugin LOGS but not that a Dask scheduler will actually run it. The gap
between those two is the registration path — ``register_plugin`` and the ``PeriodicCallback``
the plugin starts on the scheduler's own event loop — and it is not observable from a stub.

Lives here rather than beside the unit tests because it stands up a ``LocalCluster``. It was
in ``tests/unit/providers/test_provider_aws_dask.py`` carrying ``@pytest.mark.integration``, where the
root ``addopts`` deselected it and no workflow ran ``tests/unit`` with the marker enabled —
so it had never executed in CI. Moved 2026-08-25, unchanged apart from this docstring; see
``context_docs/design/test-suite-streamlining-plan.md`` §2.5.
"""

from __future__ import annotations

import logging
import time

import pytest
from distributed import Client, LocalCluster

from tessera_embeddings.providers.aws.dask import SchedulerResourceLogger


@pytest.mark.integration
class TestSchedulerResourceLoggerOnCluster:
    """End-to-end: registering the plugin on a real (local) scheduler makes it
    log on its event loop. Guards the registration/start wiring that the unit
    tests stub out.
    """

    def test_registered_plugin_emits_on_real_scheduler(self, caplog) -> None:
        with (
            caplog.at_level(logging.INFO, logger="distributed.scheduler"),
            LocalCluster(n_workers=1, dashboard_address=None, processes=False) as cluster,
            Client(cluster) as client,
        ):
            client.register_plugin(
                SchedulerResourceLogger(interval_s=0.3),
                name=SchedulerResourceLogger.name,
            )
            client.gather(client.map(lambda x: x * x, range(50)))
            time.sleep(1.0)  # allow a couple of PeriodicCallback ticks
            # Snapshot INSIDE the cluster context, so only heartbeats from a RUNNING scheduler
            # are considered. Read after the `with` exits, the last line can be one the
            # PeriodicCallback emitted while the worker was already unregistering — a correct
            # heartbeat reporting workers=0 — and the assertion below would fail on teardown
            # timing rather than on the wiring it is meant to guard.
            health = [r.getMessage() for r in caplog.records if r.getMessage().startswith("scheduler health:")]

        assert health, "expected at least one health line from the running scheduler"
        # ANY line rather than the last: the plugin may also tick once before the worker has
        # registered. What this test claims is that the registered plugin ran on the
        # scheduler's own event loop and saw the configured worker, not that every tick did.
        assert any("workers=1" in line for line in health), health
