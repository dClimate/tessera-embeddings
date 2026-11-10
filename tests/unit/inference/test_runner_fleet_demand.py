"""The inference runner publishes its fleet demand before the actor wait and clears it on exit.

The demand feeds the AWS fleet mix (``providers/aws/fleet_mix.py``); this file tests the
runner's half of that contract.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference import runner as runner_mod


def _config() -> InferenceConfig:
    """A minimal InferenceConfig; only the actor-request knobs matter here."""
    return InferenceConfig(
        time_window=parse_time_window("December 2024"),
        checkpoint_path="s3://ckpt/model.pt",
        actor_request_batch_size=4,
    )


class TestTheRunnerLifecycle:
    """Publish before the actor wait, and clear on every exit path."""

    @staticmethod
    def _run(published: list[int], *, chunks: list | None = None, chained: bool = False) -> None:
        def _wait(*_a: object, **_k: object) -> tuple:
            raise RuntimeError("Only 0 / 8 actors initialized within 600s")

        with (
            patch.object(runner_mod, "InferenceActor") as actor,
            patch.object(runner_mod, "wait_for_actors", side_effect=_wait),
            patch.object(runner_mod, "ZarrWriter") as writer,
            patch("tessera_embeddings.inference.scheduling.time.sleep"),
            # WITHOUT THIS, THIS TEST BOOTS A REAL LOCAL RAY CLUSTER. `wait_for_actors`
            # raises above, but the actor factory has already handed back stand-ins, so
            # `run_inference`'s `finally` reaches `ray.kill(actor)` — and `ray.kill` is
            # wrapped in Ray's auto-init hook, which calls `ray.init()`. That init hashes and
            # uploads the whole working directory; the same hazard once ate ~60 GB of RAM
            # across three concurrent runs, and here it cost ~3 s, making this the
            # second-slowest test in the unit suite. `test_scheduling.py` patches `ray.kill`
            # in an autouse fixture for exactly this reason; this file needs it too.
            patch.object(runner_mod.ray, "kill"),
        ):
            writer.return_value.scan_existing_staged_artifacts.return_value = SimpleNamespace(done=set(), skipped=set())
            actor.options.return_value.remote.return_value = object()
            with pytest.raises(RuntimeError):
                runner_mod.run_inference(
                    8,
                    _config(),
                    chunks or [],
                    "m",
                    "s",
                    "r",
                    0.0,
                    logging.getLogger("test-runner-fleet-demand"),
                    on_fleet_demand=published.append,
                    more_work=(lambda: None) if chained else None,
                )

    def test_a_drought_that_kills_the_actor_wait_has_still_asked(self) -> None:
        """Without this the run reaches its hours-long timeout having never once asked for
        the fallback — precisely the starvation the feature exists to prevent.
        """
        published: list[int] = []
        self._run(published, chained=True)
        # The FIRST BATCH, not the 8-actor target: the batching policy must survive.
        assert published[0] == _config().initial_actor_request(8) == 4

    def test_a_static_run_asks_only_for_the_work_it_holds(self) -> None:
        published: list[int] = []
        self._run(published, chunks=[object(), object()])
        assert published[0] == 2

    def test_the_request_is_cleared_when_the_run_ends(self) -> None:
        """Machines a request holds are exempt from idle termination, so a floor left
        standing pins an idle GPU fleet through the assembly that follows.
        """
        published: list[int] = []
        self._run(published, chunks=[object(), object()])
        assert published[-1] == 0
