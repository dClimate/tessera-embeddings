"""Unit tests for tessera_embeddings/inference/scheduling.py.

Covers all modularized pieces (ActorPool, _poll_tracker) in isolation.
No Ray cluster is required — all Ray calls are mocked.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

import tessera_embeddings.inference.scheduling as _sched_mod
from tessera_embeddings.inference.scheduling import (
    ActorPool,
    _batch_actors_to_request,
    _poll_tracker,
    _process_chunks_work_stealing,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(n: int = 3, **kwargs) -> ActorPool:
    actors = [MagicMock(name=f"actor_{i}") for i in range(n)]
    config = MagicMock()
    config.checkpoint_path = "s3://bucket/ckpt.pt"
    return ActorPool(actors, [f"i-{i:04d}" for i in range(n)], config, logging.getLogger("test"), **kwargs)  # type: ignore[arg-type]


def _fake_chunk(label: str) -> MagicMock:
    c = MagicMock()
    c.label = label
    return c


def _poll(progress: dict, *, stall_threshold: float = 300.0, max_stalls: int = 3) -> None:
    tracker = MagicMock()
    tracker.get_all.remote.return_value = MagicMock()
    with patch.object(_sched_mod.ray, "get", return_value=progress):
        _poll_tracker(tracker, 0, 10, stall_threshold, max_stalls, logging.getLogger("test"))


def _do_replace(pool: ActorPool, actor_idx: int = 0) -> None:
    mock_actor = MagicMock()
    mock_actor.get_instance_id.remote.return_value = MagicMock()
    with patch.object(_sched_mod, "InferenceActor") as cls:
        cls.remote.return_value = mock_actor
        pool.replace(actor_idx, "i-dead")


# ===========================================================================
# _poll_tracker
# ===========================================================================


class TestPollTracker:
    """Tests for _poll_tracker."""

    def test_no_op_on_empty_progress(self) -> None:
        _poll({})

    def test_no_stall_below_threshold(self) -> None:
        _poll({"c": (5, 10, 100.0, "inference")}, stall_threshold=300.0)

    @pytest.mark.parametrize(
        "phase,expected_level",
        [("inference", logging.ERROR), ("loading", logging.WARNING)],
    )
    def test_stall_log_level(self, caplog: pytest.LogCaptureFixture, phase: str, expected_level: int) -> None:
        with caplog.at_level(logging.WARNING, logger="test"):
            _poll({"chunk_A": (1, 5, 400.0, phase)}, stall_threshold=300.0, max_stalls=10)
        assert "STALL" in caplog.text
        assert any(r.levelno == expected_level for r in caplog.records)

    @pytest.mark.parametrize("count", [3, 4])  # exactly at threshold, and above
    def test_systemic_stall_raises(self, count: int) -> None:
        progress = {f"c_{i}": (1, 5, 400.0, "inference") for i in range(count)}
        with pytest.raises(RuntimeError, match="stalled simultaneously"):
            _poll(progress, stall_threshold=300.0, max_stalls=3)

    def test_tracker_error_swallowed(self) -> None:
        tracker = MagicMock()
        tracker.get_all.remote.return_value = MagicMock()
        with patch.object(_sched_mod.ray, "get", side_effect=ConnectionError("dead")):
            _poll_tracker(tracker, 0, 10, 300.0, 3, logging.getLogger("test"))

    def test_phase_summary_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        progress = {"c_A": (3, 10, 10.0, "inference"), "c_B": (1, 5, 5.0, "loading")}
        with caplog.at_level(logging.INFO, logger="test"):
            _poll(progress, stall_threshold=300.0, max_stalls=10)
        assert "inference" in caplog.text
        assert "loading" in caplog.text
        assert "active" in caplog.text


# ===========================================================================
# ActorPool — properties
# ===========================================================================


class TestActorPoolProperties:
    """Tests for ActorPool property accessors."""

    def test_busy_actors_property(self) -> None:
        pool = _make_pool(3)
        assert pool.busy_actors == set()
        ref = MagicMock()
        pool.pending[ref] = ("chunk_A", 1)
        assert pool.busy_actors == {1}

    def test_live_count_decrements_on_retire(self) -> None:
        pool = _make_pool(3)
        assert pool.live_count == 3
        pool._retired.add(0)
        assert pool.live_count == 2


# ===========================================================================
# ActorPool.resolve_iid
# ===========================================================================


class TestResolveIid:
    """Tests for ActorPool.resolve_iid."""

    def test_no_op_when_no_pending_ref(self) -> None:
        _make_pool(2).resolve_iid(0)

    def test_resolves_when_ref_ready(self) -> None:
        pool = _make_pool(2)
        ref = MagicMock()
        pool._pending_iid_refs[0] = ref
        pool._initializing.add(0)
        with (
            patch.object(_sched_mod.ray, "wait", return_value=([ref], [])),
            patch.object(_sched_mod.ray, "get", return_value="i-resolved"),
        ):
            pool.resolve_iid(0)
        assert pool.actor_instance_ids[0] == "i-resolved"
        assert 0 not in pool._pending_iid_refs
        assert 0 not in pool._initializing

    def test_fetch_failure_falls_back_to_unknown(self) -> None:
        pool = _make_pool(2)
        ref = MagicMock()
        pool._pending_iid_refs[0] = ref
        pool._initializing.add(0)
        with (
            patch.object(_sched_mod.ray, "wait", return_value=([ref], [])),
            patch.object(_sched_mod.ray, "get", side_effect=RuntimeError("timeout")),
        ):
            pool.resolve_iid(0)
        assert "unknown" in pool.actor_instance_ids[0]
        assert 0 not in pool._pending_iid_refs

    def test_non_blocking_leaves_ref_pending(self) -> None:
        pool = _make_pool(2)
        ref = MagicMock()
        pool._pending_iid_refs[0] = ref
        with patch.object(_sched_mod.ray, "wait", return_value=([], [ref])):
            pool.resolve_iid(0, timeout=0)
        assert 0 in pool._pending_iid_refs


# ===========================================================================
# ActorPool.submit / seed / dispatch_idle
# ===========================================================================


class TestSubmit:
    """Tests for ActorPool.submit."""

    def test_submit_records_pending_and_tracks_attempts(self) -> None:
        pool = _make_pool(2)
        chunk = _fake_chunk("c0")
        ref = MagicMock()
        pool.actors[0].process_chunk.remote.return_value = ref
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.submit(0, chunk, "s3://mosaic", "s3://stage", "run1", tracker=None)
            assert pool.pending[ref] == ("c0", 0)
            assert pool.chunk_attempts["c0"] == 1
            pool.submit(0, chunk, "s3://mosaic", "s3://stage", "run1", tracker=None)
            assert pool.chunk_attempts["c0"] == 2


class TestSeed:
    """Tests for ActorPool.seed."""

    def test_dispatches_one_per_actor(self) -> None:
        pool = _make_pool(3)
        for actor in pool.actors:
            actor.process_chunk.remote.return_value = MagicMock()
        queue: deque = deque([_fake_chunk(f"c{i}") for i in range(5)])
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.seed(queue, "m", "s", "r", None)
        assert len(pool.pending) == 3
        assert len(queue) == 2

    def test_handles_fewer_chunks_than_actors(self) -> None:
        pool = _make_pool(4)
        for actor in pool.actors:
            actor.process_chunk.remote.return_value = MagicMock()
        queue: deque = deque([_fake_chunk(f"c{i}") for i in range(2)])
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.seed(queue, "m", "s", "r", None)
        assert len(pool.pending) == 2
        assert len(queue) == 0


class TestDispatchIdle:
    """Tests for ActorPool.dispatch_idle."""

    def test_dispatches_to_all_idle_actors(self) -> None:
        pool = _make_pool(2)
        for actor in pool.actors:
            actor.process_chunk.remote.return_value = MagicMock()
        queue: deque = deque([_fake_chunk("c0"), _fake_chunk("c1")])
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.dispatch_idle(queue, "m", "s", "r", None)
        assert len(pool.pending) == 2
        assert len(queue) == 0

    def test_skips_busy_actors(self) -> None:
        pool = _make_pool(2)
        ref = MagicMock()
        pool.pending[ref] = ("existing", 0)
        pool.actors[1].process_chunk.remote.return_value = MagicMock()
        queue: deque = deque([_fake_chunk("c0")])
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.dispatch_idle(queue, "m", "s", "r", None)
        assert {aidx for _, aidx in pool.pending.values()} == {0, 1}


# ===========================================================================
# ActorPool.retire_idle
# ===========================================================================


class TestRetireIdle:
    """Tests for ActorPool.retire_idle."""

    def test_grace_period_not_elapsed_no_kill(self) -> None:
        pool = _make_pool(2, idle_grace_sec=120)
        pool._idle_since[0] = time.monotonic()
        with patch.object(_sched_mod.ray, "kill") as mock_kill:
            pool.retire_idle(outstanding_work=0)
        mock_kill.assert_not_called()

    def test_floor_check_prevents_kill(self) -> None:
        pool = _make_pool(2, idle_grace_sec=1)
        pool._idle_since[0] = time.monotonic() - 200
        with patch.object(_sched_mod.ray, "kill") as mock_kill:
            pool.retire_idle(outstanding_work=2)  # live-1=1 < 2
        mock_kill.assert_not_called()

    def test_idle_actor_killed_after_grace(self) -> None:
        pool = _make_pool(3, idle_grace_sec=1)
        pool._idle_since[0] = time.monotonic() - 200
        with patch.object(_sched_mod.ray, "kill") as mock_kill:
            pool.retire_idle(outstanding_work=1)  # live-1=2 >= 1
        mock_kill.assert_called_once_with(pool.actors[0])
        assert 0 in pool._retired

    @pytest.mark.parametrize("exclusion", ["busy", "initializing", "retired"])
    def test_excluded_actor_not_killed(self, exclusion: str) -> None:
        pool = _make_pool(2, idle_grace_sec=1)
        pool._idle_since[0] = time.monotonic() - 200
        ref = MagicMock()
        if exclusion == "busy":
            pool.pending[ref] = ("chunk", 0)
        elif exclusion == "initializing":
            pool._initializing.add(0)
        else:
            pool._retired.add(0)
        with patch.object(_sched_mod.ray, "kill") as mock_kill:
            pool.retire_idle(outstanding_work=0)
        mock_kill.assert_not_called()
        if exclusion == "busy":
            assert 0 not in pool._idle_since  # timer cleared for active actors

    def test_kill_exception_suppressed(self) -> None:
        pool = _make_pool(2, idle_grace_sec=1)
        pool._idle_since[0] = time.monotonic() - 200
        with patch.object(_sched_mod.ray, "kill", side_effect=RuntimeError("already dead")):
            pool.retire_idle(outstanding_work=0)
        assert 0 in pool._retired

    def test_on_retire_called_with_instance_id(self) -> None:
        """on_retire callback fires with the real EC2 instance ID after kill."""
        callback = MagicMock()
        pool = _make_pool(3, idle_grace_sec=1, on_retire=callback)
        pool._idle_since[0] = time.monotonic() - 200
        with patch.object(_sched_mod.ray, "kill"):
            pool.retire_idle(outstanding_work=1)
        callback.assert_called_once_with("i-0000")

    def test_on_retire_skipped_for_placeholder_ids(self) -> None:
        """on_retire is NOT called when the instance ID is a placeholder."""
        callback = MagicMock()
        pool = _make_pool(2, idle_grace_sec=1, on_retire=callback)
        pool.actor_instance_ids[0] = "pending-init"
        pool._idle_since[0] = time.monotonic() - 200
        with patch.object(_sched_mod.ray, "kill"):
            pool.retire_idle(outstanding_work=0)
        assert 0 in pool._retired
        callback.assert_not_called()

    def test_on_retire_exception_suppressed(self) -> None:
        """Exceptions from on_retire don't crash the scheduler."""
        callback = MagicMock(side_effect=RuntimeError("EC2 API error"))
        pool = _make_pool(3, idle_grace_sec=1, on_retire=callback)
        pool._idle_since[0] = time.monotonic() - 200
        with patch.object(_sched_mod.ray, "kill"):
            pool.retire_idle(outstanding_work=1)
        assert 0 in pool._retired
        callback.assert_called_once()

    def test_on_retire_not_called_when_none(self) -> None:
        """No error when on_retire is None (default)."""
        pool = _make_pool(2, idle_grace_sec=1)
        pool._idle_since[0] = time.monotonic() - 200
        with patch.object(_sched_mod.ray, "kill"):
            pool.retire_idle(outstanding_work=0)
        assert 0 in pool._retired

    def test_resolve_iid_called_before_kill(self) -> None:
        """resolve_iid picks up a lazily-resolved instance ID before retirement."""
        callback = MagicMock()
        pool = _make_pool(2, idle_grace_sec=1, on_retire=callback)
        # Simulate an actor whose instance ID hasn't been resolved yet
        pool.actor_instance_ids[0] = "pending-init"
        pool._idle_since[0] = time.monotonic() - 200
        ref = MagicMock()
        pool._pending_iid_refs[0] = ref
        # resolve_iid will find the ready ref and update the instance ID
        with (
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod.ray, "wait", return_value=([ref], [])),
            patch.object(_sched_mod.ray, "get", return_value="i-newly-resolved"),
        ):
            pool.retire_idle(outstanding_work=0)
        assert pool.actor_instance_ids[0] == "i-newly-resolved"
        callback.assert_called_once_with("i-newly-resolved")

    def test_grace_period_always_applies(self) -> None:
        """Actors are never retired without the grace period elapsing, even with no queued work."""
        pool = _make_pool(2, idle_grace_sec=9999)
        ref = MagicMock()
        pool.pending[ref] = ("chunk", 1)  # actor 1 is busy, actor 0 is idle
        with patch.object(_sched_mod.ray, "kill") as mock_kill:
            pool.retire_idle(outstanding_work=1)
        mock_kill.assert_not_called()


# ===========================================================================
# ActorPool.mark_initializing
# ===========================================================================


class TestMarkInitializing:
    """Tests for ActorPool.mark_initializing."""

    def test_sets_up_state(self) -> None:
        pool = _make_pool(3)
        pool.mark_initializing(1)
        assert 1 in pool._initializing
        assert 1 in pool._pending_iid_refs
        assert pool.actor_instance_ids[1] == "pending-init"

    def test_custom_placeholder(self) -> None:
        pool = _make_pool(2)
        pool.mark_initializing(0, placeholder_iid="pending-replacement-of-i-dead")
        assert pool.actor_instance_ids[0] == "pending-replacement-of-i-dead"

    def test_seed_skips_initializing_actors(self) -> None:
        """seed() skips initializing actors — their chunks stay queued.

        Avoids sending work to actors whose EC2 instances may never
        provision. dispatch_idle assigns work once they're confirmed alive.
        """
        pool = _make_pool(3)
        pool.mark_initializing(2)  # actor 2 is still starting up
        for actor in pool.actors:
            actor.process_chunk.remote.return_value = MagicMock()
        queue: deque = deque([_fake_chunk(f"c{i}") for i in range(3)])
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.seed(queue, "m", "s", "r", None)
        # Only actors 0 and 1 got work; actor 2 was skipped, chunk stays queued
        assert len(pool.pending) == 2
        actor_indices = {aidx for _, aidx in pool.pending.values()}
        assert actor_indices == {0, 1}
        assert len(queue) == 1

    def test_dispatch_idle_skips_initializing_actors(self) -> None:
        """dispatch_idle() never dispatches to initializing actors."""
        pool = _make_pool(2)
        pool.mark_initializing(0)  # actor 0 still starting
        pool.actors[1].process_chunk.remote.return_value = MagicMock()
        queue: deque = deque([_fake_chunk("c0"), _fake_chunk("c1")])
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.dispatch_idle(queue, "m", "s", "r", None)
        # Actor 1 got one chunk; actor 0 (initializing) was skipped, second chunk stays queued
        assert len(pool.pending) == 1
        assert {aidx for _, aidx in pool.pending.values()} == {1}
        assert len(queue) == 1


# ===========================================================================
# ActorPool.resolve_initializing
# ===========================================================================


class TestResolveInitializing:
    """Tests for ActorPool.resolve_initializing."""

    def test_moves_resolved_actor_out_of_initializing(self) -> None:
        pool = _make_pool(3)
        pool.mark_initializing(1)
        ref = pool._pending_iid_refs[1]
        with (
            patch.object(_sched_mod.ray, "wait", return_value=([ref], [])),
            patch.object(_sched_mod.ray, "get", return_value="i-resolved"),
        ):
            resolved = pool.resolve_initializing()
        assert resolved == 1
        assert 1 not in pool._initializing
        assert pool.actor_instance_ids[1] == "i-resolved"

    def test_leaves_unresolved_actors_in_initializing(self) -> None:
        pool = _make_pool(3)
        pool.mark_initializing(1)
        pool.mark_initializing(2)
        ref1 = pool._pending_iid_refs[1]
        ref2 = pool._pending_iid_refs[2]

        # Actor 1 resolves, actor 2 does not (simulating unprovisionable instance)
        def fake_wait(refs, timeout=0):
            if refs[0] is ref1:
                return ([ref1], [])
            return ([], [ref2])

        with (
            patch.object(_sched_mod.ray, "wait", side_effect=fake_wait),
            patch.object(_sched_mod.ray, "get", return_value="i-resolved"),
        ):
            resolved = pool.resolve_initializing()
        assert resolved == 1
        assert 1 not in pool._initializing
        assert 2 in pool._initializing

    def test_newly_ready_actor_receives_work_via_dispatch_idle(self) -> None:
        """Full flow: initializing → resolve → dispatch."""
        pool = _make_pool(3)
        pool.mark_initializing(2)
        # Actors 0 and 1 are busy (simulating seeded work)
        pool.pending[MagicMock()] = ("c_a", 0)
        pool.pending[MagicMock()] = ("c_b", 1)
        pool.actors[2].process_chunk.remote.return_value = MagicMock()
        ref = pool._pending_iid_refs[2]
        queue: deque = deque([_fake_chunk("c0")])
        with (
            patch.object(_sched_mod.ray, "wait", return_value=([ref], [])),
            patch.object(_sched_mod.ray, "get", return_value="i-new"),
        ):
            pool.resolve_initializing()
            pool.dispatch_idle(queue, "m", "s", "r", None)
        # Actor 2 is now ready and got the queued chunk
        assert 2 not in pool._initializing
        assert 2 in {aidx for _, aidx in pool.pending.values()}
        assert len(queue) == 0


# ===========================================================================
# ActorPool.replace
# ===========================================================================


class TestReplace:
    """Tests for ActorPool.replace."""

    @pytest.mark.parametrize(
        "n_deaths,pool_size,expected_level",
        [
            (1, 4, logging.WARNING),  # < 50 %
            (2, 4, logging.ERROR),  # >= 50 %
            (4, 4, logging.CRITICAL),  # >= 100 %
        ],
    )
    def test_death_log_level(
        self, caplog: pytest.LogCaptureFixture, n_deaths: int, pool_size: int, expected_level: int
    ) -> None:
        pool = _make_pool(pool_size)
        with caplog.at_level(logging.WARNING, logger="test"):
            for _ in range(n_deaths):
                _do_replace(pool)
        assert any(r.levelno == expected_level for r in caplog.records)

    def test_replace_creates_new_actor_and_marks_initializing(self) -> None:
        pool = _make_pool(2)
        original = pool.actors[0]
        mock_new = MagicMock()
        mock_new.get_instance_id.remote.return_value = MagicMock()
        with patch.object(_sched_mod, "InferenceActor") as cls:
            cls.remote.return_value = mock_new
            pool.replace(0, "i-dead")
        assert pool.actors[0] is not original
        assert pool.actors[0] is mock_new
        assert 0 in pool._initializing
        assert "pending-replacement-of-i-dead" in pool.actor_instance_ids[0]


# ===========================================================================
# ActorPool.add_actors
# ===========================================================================


class TestAddActors:
    """Tests for ActorPool.add_actors (growable pool)."""

    def test_appends_and_marks_initializing(self) -> None:
        pool = _make_pool(2)
        new = [MagicMock(name="new_0"), MagicMock(name="new_1")]
        for a in new:
            a.get_instance_id.remote.return_value = MagicMock()
        pool.add_actors(new)
        assert len(pool.actors) == 4
        assert len(pool.actor_instance_ids) == 4
        # New slots are initializing with placeholder IDs until resolved.
        assert pool._initializing == {2, 3}
        assert pool.actor_instance_ids[2] == "pending-init"
        assert pool.actor_instance_ids[3] == "pending-init"

    def test_max_actor_deaths_tracks_growth(self) -> None:
        """The systemic-death threshold scales as batches are added."""
        pool = _make_pool(2)
        assert pool.max_actor_deaths == 2
        extra = [MagicMock(), MagicMock(), MagicMock()]
        for a in extra:
            a.get_instance_id.remote.return_value = MagicMock()
        pool.add_actors(extra)
        assert pool.max_actor_deaths == 5

    def test_added_actor_gets_work_via_dispatch_idle(self) -> None:
        """An appended actor receives queued work once it resolves."""
        pool = _make_pool(1)
        pool.pending[MagicMock()] = ("seeded", 0)  # original actor busy
        new = MagicMock(name="new")
        new.get_instance_id.remote.return_value = MagicMock()
        new.process_chunk.remote.return_value = MagicMock()
        pool.add_actors([new])
        ref = pool._pending_iid_refs[1]
        queue: deque = deque([_fake_chunk("c0")])
        with (
            patch.object(_sched_mod.ray, "wait", return_value=([ref], [])),
            patch.object(_sched_mod.ray, "get", return_value="i-new"),
        ):
            pool.resolve_initializing()
            pool.dispatch_idle(queue, "m", "s", "r", None)
        assert 1 in {aidx for _, aidx in pool.pending.values()}
        assert len(queue) == 0


# ===========================================================================
# _batch_actors_to_request
# ===========================================================================


class TestBatchActorsToRequest:
    """Tests for the pure batch-decision function."""

    def _call(self, **overrides) -> tuple[int, bool]:
        kwargs = dict(
            requested=50,
            target=200,
            outstanding=200,
            alive_gpu_nodes=50,
            secs_since_last_batch=1.0,
            placement_timeout_sec=300.0,
            batch_size=50,
        )
        kwargs.update(overrides)
        return _batch_actors_to_request(**kwargs)

    def test_requests_next_batch_when_placed(self) -> None:
        n, timed_out = self._call(alive_gpu_nodes=50)
        assert n == 50
        assert timed_out is False

    def test_no_request_when_target_reached(self) -> None:
        assert self._call(requested=200, target=200) == (0, False)

    def test_no_request_when_prior_batch_not_placed(self) -> None:
        # Only 40 of the requested 50 instances have joined the cluster.
        assert self._call(requested=50, alive_gpu_nodes=40) == (0, False)

    def test_straggler_within_tolerance_still_placed(self) -> None:
        # 48/50 placed is within the default tolerance of 2.
        n, _ = self._call(requested=50, alive_gpu_nodes=48)
        assert n == 50

    def test_timeout_forces_request_despite_no_placement(self) -> None:
        n, timed_out = self._call(alive_gpu_nodes=10, secs_since_last_batch=301.0)
        assert n == 50
        assert timed_out is True

    def test_final_batch_clamped_to_target(self) -> None:
        n, _ = self._call(requested=180, target=200, alive_gpu_nodes=180, outstanding=200)
        assert n == 20

    def test_no_request_when_pool_covers_outstanding_work(self) -> None:
        # Only 30 chunks left but 50 actors already requested — don't over-provision.
        assert self._call(requested=50, outstanding=30) == (0, False)


# ===========================================================================
# _process_chunks_work_stealing — loop condition
# ===========================================================================


class TestWorkStealingLoopCondition:
    """Verify the main loop doesn't exit while chunks are queued for initializing actors."""

    def test_queued_chunks_processed_after_all_ready_actors_die(self) -> None:
        """Regression: when the only ready actor dies and its chunk is re-queued,
        the loop must wait for the replacement (initializing) actor to come
        online and process remaining chunks — not exit with pending empty.
        """
        actor = MagicMock(name="actor_0")
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        chunk = _fake_chunk("c0")
        result_ok = {"chunk": "c0", "status": "ok"}

        # Track state across loop iterations for ray.wait side effect
        call_count = 0
        seed_ref = MagicMock(name="seed_ref")
        retry_ref = MagicMock(name="retry_ref")
        iid_ref = MagicMock(name="iid_ref")

        # actor.process_chunk.remote returns seed_ref first, then retry_ref
        actor.process_chunk.remote.side_effect = [seed_ref, retry_ref]
        actor.get_instance_id.remote.return_value = iid_ref

        replacement_actor = MagicMock(name="replacement_actor")
        replacement_actor.process_chunk.remote.return_value = retry_ref
        replacement_actor.get_instance_id.remote.return_value = iid_ref

        def fake_wait(refs, num_returns=1, timeout=60):
            nonlocal call_count
            call_count += 1
            # Iteration 1: seed_ref completes (will raise to simulate death)
            if seed_ref in refs:
                return ([seed_ref], [r for r in refs if r is not seed_ref])
            # Iteration 2: no pending, loop is waiting for initializing actor
            # (ray.wait not called when pending is empty — sleep path instead)
            # Iteration 3+: retry_ref completes successfully
            if retry_ref in refs:
                return ([retry_ref], [r for r in refs if r is not retry_ref])
            return ([], refs)

        def fake_get(ref, *args, **kwargs):
            if ref is seed_ref:
                raise RuntimeError("actor died")
            if ref is retry_ref:
                return result_ok
            if ref is iid_ref:
                return "i-replacement"
            return MagicMock()

        def fake_wait_iid(refs, timeout=0):
            """For resolve_iid / resolve_initializing calls."""
            if iid_ref in refs:
                return ([iid_ref], [])
            return ([], refs)

        with (
            patch.object(_sched_mod.ray, "wait") as mock_wait,
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod, "InferenceActor") as cls,
            patch.object(_sched_mod, "log_worker_failure_diagnostic"),
            patch.object(_sched_mod.time, "sleep"),
        ):
            # ray.wait dispatches differently for the main loop vs resolve_iid
            mock_wait.side_effect = lambda refs, **kw: (
                fake_wait_iid(refs, timeout=kw.get("timeout", 0))
                if kw.get("timeout", 60) == 0
                else fake_wait(refs, **kw)
            )
            cls.remote.return_value = replacement_actor

            results = _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=[chunk],
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                t0=time.monotonic(),
                log=logging.getLogger("test"),
                max_chunk_retries=2,
            )

        # The chunk must have been processed despite the actor dying
        assert len(results) == 1
        assert results[0]["status"] == "ok"


# ===========================================================================
# _process_chunks_work_stealing — actor batching
# ===========================================================================


class TestWorkStealingBatching:
    """Verify the loop requests later actor batches as the pool drains work."""

    def test_second_batch_requested_after_first_placed(self) -> None:
        """Starting with one actor and a batch size of 2, the loop requests the
        remaining batch via the factory (placement always satisfied here) and
        grows the pool until the target is reached.
        """
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        config.actor_request_batch_size = 2

        chunks = [_fake_chunk(f"c{i}") for i in range(4)]

        def _make_actor(iid: str) -> MagicMock:
            a = MagicMock()
            iid_ref = MagicMock()
            iid_ref._iid = iid
            a.get_instance_id.remote.return_value = iid_ref
            a.process_chunk.remote.side_effect = lambda *args, **kwargs: MagicMock()
            return a

        factory_calls: list[int] = []

        def factory(n: int) -> list[MagicMock]:
            factory_calls.append(n)
            return [_make_actor("i-new") for _ in range(n)]

        actor0 = _make_actor("i-0000")

        def fake_nodes():
            return [{"Alive": True, "Resources": {"GPU": 1}}] * 16

        def fake_get(ref, *args, **kwargs):
            if getattr(ref, "_iid", None):
                return ref._iid
            return {"chunk": "x", "status": "ok"}

        def fake_wait(refs, num_returns=1, timeout=60):
            # timeout=0 → resolve_iid / resolve_initializing: report all done.
            if timeout == 0:
                return (list(refs), [])
            # Main loop: complete one pending chunk per iteration.
            if refs:
                return ([refs[0]], list(refs[1:]))
            return ([], [])

        with (
            patch.object(_sched_mod.ray, "wait", side_effect=fake_wait),
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "nodes", side_effect=fake_nodes),
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod.time, "sleep"),
        ):
            results = _process_chunks_work_stealing(
                actors=[actor0],
                actor_instance_ids=["i-0000"],
                chunks=chunks,
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                t0=time.monotonic(),
                log=logging.getLogger("test"),
                actor_factory=factory,
                total_actors_target=3,
                placement_timeout_sec=300.0,
            )

        # The factory was invoked for the remaining 2 actors (target 3, started with 1).
        assert factory_calls == [2]
        # Every chunk was processed.
        assert len(results) == 4

    def test_no_batching_when_factory_absent(self) -> None:
        """With no actor_factory the loop never tries to grow the pool."""
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        config.actor_request_batch_size = 2  # set, but no factory → disabled

        actor0 = MagicMock()
        actor0.process_chunk.remote.side_effect = lambda *a, **k: MagicMock()

        def fake_wait(refs, num_returns=1, timeout=60):
            if timeout == 0:
                return (list(refs), [])
            return ([refs[0]], list(refs[1:])) if refs else ([], [])

        with (
            patch.object(_sched_mod.ray, "wait", side_effect=fake_wait),
            patch.object(_sched_mod.ray, "get", return_value={"chunk": "x", "status": "ok"}),
            patch.object(_sched_mod.ray, "nodes", side_effect=AssertionError("nodes() must not be polled")),
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod.time, "sleep"),
        ):
            results = _process_chunks_work_stealing(
                actors=[actor0],
                actor_instance_ids=["i-0000"],
                chunks=[_fake_chunk("c0")],
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                t0=time.monotonic(),
                log=logging.getLogger("test"),
            )
        assert len(results) == 1
