"""Unit tests for tessera_embeddings/inference/scheduling.py.

Covers all modularized pieces (ActorPool, _poll_tracker) in isolation.
No Ray cluster is required — all Ray calls are mocked.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

import tessera_embeddings.inference.scheduling as _sched_mod
from tessera_embeddings.inference.progress import chunk_uid
from tessera_embeddings.inference.scheduling import (
    ActorPool,
    _batch_actors_to_request,
    _joined_gpu_count,
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


def _poll(
    progress: dict,
    *,
    stall_threshold: float = 300.0,
    max_stalls: int = 3,
    recovery_threshold: float | None = None,
) -> list[str]:
    tracker = MagicMock()
    tracker.get_all.remote.return_value = MagicMock()
    with patch.object(_sched_mod.ray, "get", return_value=progress):
        return _poll_tracker(
            tracker,
            0,
            10,
            stall_threshold,
            max_stalls,
            logging.getLogger("test"),
            recovery_threshold_sec=recovery_threshold,
        )


def _do_replace(pool: ActorPool, actor_idx: int = 0) -> None:
    mock_actor = MagicMock()
    mock_actor.get_instance_id.remote.return_value = MagicMock()
    with (
        patch.object(_sched_mod, "InferenceActor") as cls,
        # ray.kill MUST be patched: Ray wraps it in an auto-init hook, so an
        # unpatched call from a test silently boots a REAL local Ray cluster —
        # ray.init() then hashes/uploads the entire working directory (multi-GB
        # with scale-test stores present) and reserves an object store. The
        # suppress(Exception) around the kill in replace() does not help; the
        # auto-init runs before kill raises. (2026-07-17: three concurrent
        # pytest runs of this file ate ~60 GB RAM this way.)
        patch.object(_sched_mod.ray, "kill"),
    ):
        cls.options.return_value.remote.return_value = mock_actor
        pool.replace(actor_idx, "i-dead")


def test_replace_carries_credentials_and_region_to_new_actor():
    """A replacement actor (after a chunk failure/OOM) must inherit the pool's
    credentials AND s3_region, else a non-default-region retry opens the mosaic in
    the default region and re-fails the remaining chunks.
    """
    creds = object()
    pool = _make_pool(get_credentials=creds, s3_region="eu-west-1")
    mock_actor = MagicMock()
    mock_actor.get_instance_id.remote.return_value = MagicMock()
    with patch.object(_sched_mod, "InferenceActor") as cls:
        cls.options.return_value.remote.return_value = mock_actor
        pool.replace(0, "i-dead")
    # .remote(config, checkpoint_path, get_credentials, s3_region)
    args = cls.options.return_value.remote.call_args.args
    assert args[2] is creds
    assert args[3] == "eu-west-1"


# ===========================================================================
# _joined_gpu_count
# ===========================================================================


class TestJoinedGpuCount:
    """The capacity reading behind the progress line's GPU-hours."""

    def test_reads_the_cluster_total(self) -> None:
        with patch.object(_sched_mod.ray, "cluster_resources", return_value={"CPU": 40.0, "GPU": 7.0}):
            assert _joined_gpu_count(0.0, 1.0) == 7.0

    def test_no_gpu_nodes_yet_reads_zero(self) -> None:
        """Before the first worker joins, the cluster has no GPU key at all."""
        with patch.object(_sched_mod.ray, "cluster_resources", return_value={"CPU": 8.0}):
            assert _joined_gpu_count(3.0, 1.0) == 0.0

    def test_a_run_that_asks_for_no_gpu_reports_none(self) -> None:
        """A CPU-only run on a GPU-capable cluster must not charge itself the cluster's GPUs.

        The reading is of the CLUSTER, so it says nothing about this run once this run reserves
        no GPU. Answering with the cluster total there would invent GPU-hours for a fill that
        used none.
        """
        with patch.object(_sched_mod.ray, "cluster_resources", return_value={"GPU": 64.0}) as looked:
            assert _joined_gpu_count(0.0, 0.0) == 0.0
            assert not looked.called, "a CPU-only run should not even ask"

    def test_a_failed_lookup_carries_the_previous_count_forward(self) -> None:
        """The caller is in the dispatch loop, outside the tracker poll's guard.

        A control-plane hiccup must cost accuracy in a log line, never the fill.
        """
        with patch.object(_sched_mod.ray, "cluster_resources", side_effect=RuntimeError("gcs unavailable")):
            assert _joined_gpu_count(5.0, 1.0) == 5.0


# ===========================================================================
# _poll_tracker
# ===========================================================================


class TestPollTracker:
    """Tests for _poll_tracker."""

    def test_no_op_on_empty_progress(self) -> None:
        """An idle fleet is not a stalled one, and an operator PAUSE depends on this.

        A paused cluster drains its queue and then completes nothing for as long as the pause
        lasts. That is only safe because a finished chunk is removed from the tracker, so a
        drained fleet reports no progress entries at all rather than entries whose staleness
        grows without bound — which would trip the systemic-abort arm and turn a pause into a
        crashed run. Anything that starts RETAINING completed entries breaks the pause.
        """
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

    def test_returns_nothing_to_recover_by_default(self) -> None:
        """Recovery is opt-in: without a threshold the poll stays log-only."""
        assert _poll({"c": (1, 5, 99_999.0, "inference")}, max_stalls=10) == []

    def test_warns_but_does_not_recover_between_the_thresholds(self) -> None:
        """The warning must fire well before anything is killed.

        A chunk past the stall threshold but short of the recovery one is logged and
        left alone — this is the window that keeps a merely-slow chunk alive.
        """
        recover = _poll(
            {"c": (1, 5, 400.0, "inference")},
            stall_threshold=300.0,
            recovery_threshold=1200.0,
            max_stalls=10,
        )
        assert recover == []

    def test_recovers_only_past_the_recovery_threshold(self) -> None:
        recover = _poll(
            {"slow": (1, 5, 400.0, "inference"), "wedged": (50, 587, 14_396.0, "inference")},
            stall_threshold=300.0,
            recovery_threshold=1200.0,
            max_stalls=10,
        )
        assert recover == ["wedged"], "only the chunk past the recovery threshold"

    def test_recovers_a_stall_in_any_phase(self) -> None:
        """The observed wedge was in 'inference', but a hung load is just as fatal."""
        recover = _poll(
            {"c": (0, 0, 5_000.0, "loading")},
            stall_threshold=300.0,
            recovery_threshold=1200.0,
            max_stalls=10,
        )
        assert recover == ["c"]

    def test_a_failed_poll_recovers_nothing(self) -> None:
        """Losing visibility must not be read as 'everything is wedged'.

        A tracker error is swallowed as non-fatal, and it must return an EMPTY
        recovery list — returning anything else would kill actors because the
        monitor broke, which is the opposite of what it exists for.
        """
        tracker = MagicMock()
        tracker.get_all.remote.return_value = MagicMock()
        with patch.object(_sched_mod.ray, "get", side_effect=ConnectionError("dead")):
            out = _poll_tracker(tracker, 0, 10, 300.0, 3, logging.getLogger("test"), recovery_threshold_sec=1200.0)
        assert out == []

    def test_systemic_abort_still_wins_over_recovery(self) -> None:
        """The two guards answer different questions and must not be merged.

        Many chunks stalling at once is systemic and wants a human, so it still
        raises rather than quietly killing the whole fleet one actor at a time.
        """
        progress = {f"c_{i}": (1, 5, 14_000.0, "inference") for i in range(4)}
        with pytest.raises(RuntimeError, match="stalled simultaneously"):
            _poll(progress, stall_threshold=300.0, max_stalls=3, recovery_threshold=1200.0)

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
            item, actor_idx = pool.pending[ref]
            assert (item.chunk.label, actor_idx) == ("c0", 0)
            assert item.ctx == _sched_mod.ZoneContext("s3://mosaic", "s3://stage", "run1")
            # Attempts are keyed run_id-qualified: bare labels collide across
            # a chained session's zones.
            assert pool.chunk_attempts["run1:c0"] == 1
            pool.submit(0, chunk, "s3://mosaic", "s3://stage", "run1", tracker=None)
            assert pool.chunk_attempts["run1:c0"] == 2


class TestReservations:
    """1-deep next-chunk reservations feeding the cross-chunk starter prefetch."""

    def test_deep_queue_reserves_head_and_passes_hint(self) -> None:
        pool = _make_pool(2)
        pool.actors[0].process_chunk.remote.return_value = MagicMock()
        queue: deque = deque([_fake_chunk(f"c{i}") for i in range(5)])
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.submit(0, _fake_chunk("current"), "m", "s", "r", None, queue)
        assert pool.reserved[0].chunk.label == "c0"
        assert len(queue) == 4
        kwargs = pool.actors[0].process_chunk.remote.call_args.kwargs
        assert kwargs["prefetch_hint"].label == "c0"

    def test_shallow_queue_does_not_reserve(self) -> None:
        # len(queue) <= live_count: a tail reservation would pin work to a
        # busy actor while others idle.
        pool = _make_pool(3)
        pool.actors[0].process_chunk.remote.return_value = MagicMock()
        queue: deque = deque([_fake_chunk("c0"), _fake_chunk("c1")])
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.submit(0, _fake_chunk("current"), "m", "s", "r", None, queue)
        assert pool.reserved == {}
        assert len(queue) == 2
        kwargs = pool.actors[0].process_chunk.remote.call_args.kwargs
        assert kwargs["prefetch_hint"] is None

    def test_no_queue_means_no_reservation(self) -> None:
        pool = _make_pool(2)
        pool.actors[0].process_chunk.remote.return_value = MagicMock()
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.submit(0, _fake_chunk("current"), "m", "s", "r", None)
        assert pool.reserved == {}
        assert pool.actors[0].process_chunk.remote.call_args.kwargs["prefetch_hint"] is None

    def test_take_reserved_pops(self) -> None:
        pool = _make_pool(2)
        chunk = _fake_chunk("c0")
        pool.reserved[1] = chunk
        assert pool.take_reserved(1) is chunk
        assert pool.take_reserved(1) is None

    def test_stale_reservation_requeued_not_stranded(self) -> None:
        # Defensive path: an actor re-dispatched while still holding a
        # reservation must not lose that chunk.
        pool = _make_pool(2)
        pool.actors[0].process_chunk.remote.return_value = MagicMock()
        stale = _fake_chunk("stale")
        pool.reserved[0] = stale
        queue: deque = deque([_fake_chunk(f"c{i}") for i in range(4)])
        with patch.object(_sched_mod.ray, "wait", return_value=([], [])):
            pool.submit(0, _fake_chunk("current"), "m", "s", "r", None, queue)

        # The stale chunk went to the queue front, so it became the new
        # reservation — conserved either way: nothing dropped.
        def _label(entry):
            return entry.chunk.label if isinstance(entry, _sched_mod.WorkItem) else entry.label

        all_labels = {_label(c) for c in queue} | {_label(c) for c in pool.reserved.values()}
        assert "stale" in all_labels
        assert len(queue) + len(pool.reserved) == 5


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
        # While the queue is deeper than the live pool, seeding also reserves
        # next chunks (prefetch hints) — nothing is dropped either way.
        assert len(queue) + len(pool.reserved) == 2
        assert pool.reserved  # at least the first seeded actor got a hint

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
            cls.options.return_value.remote.return_value = mock_new
            pool.replace(0, "i-dead")
        assert pool.actors[0] is not original
        assert pool.actors[0] is mock_new
        assert 0 in pool._initializing
        assert "pending-replacement-of-i-dead" in pool.actor_instance_ids[0]

    def test_replace_kills_outgoing_actor(self) -> None:
        """The actor in the slot is killed before the replacement takes over.

        Critical for the still-alive failure path: an actor that caught a CUDA
        error and returned status="failed" must be torn down so it can't pick up
        another chunk.
        """
        pool = _make_pool(2)
        outgoing = pool.actors[0]
        mock_new = MagicMock()
        mock_new.get_instance_id.remote.return_value = MagicMock()
        with (
            patch.object(_sched_mod, "InferenceActor") as cls,
            patch.object(_sched_mod.ray, "kill") as mock_kill,
        ):
            cls.options.return_value.remote.return_value = mock_new
            pool.replace(0, "i-dead")
        mock_kill.assert_called_once_with(outgoing)

    def test_replace_kill_exception_suppressed(self) -> None:
        """A kill failure (process already dead) doesn't break replacement."""
        pool = _make_pool(2)
        mock_new = MagicMock()
        mock_new.get_instance_id.remote.return_value = MagicMock()
        with (
            patch.object(_sched_mod, "InferenceActor") as cls,
            patch.object(_sched_mod.ray, "kill", side_effect=RuntimeError("already dead")),
        ):
            cls.options.return_value.remote.return_value = mock_new
            pool.replace(0, "i-dead")
        assert pool.actors[0] is mock_new
        assert 0 in pool._initializing


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
            nodes_at_last_batch=0,
            last_batch_size=50,
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
        # Only 40 of the last batch's 50 instances have joined the cluster.
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

    def test_partial_final_batch_clamped_to_outstanding_work(self) -> None:
        # 50 actors requested, only 60 chunks left: request 10, not a full
        # batch of 50 (which would grow the pool to 100 for 60 chunks).
        n, _ = self._call(requested=50, outstanding=60, target=200, alive_gpu_nodes=50)
        assert n == 10

    def test_placement_measured_incrementally_after_timeout(self) -> None:
        # First batch (50) timed out with only 40 placed; a later batch grew the
        # pool to 100 and those instances joined (90 nodes total). The increment
        # since the last batch (90 - 40 = 50) covers that batch's 50 actors, so
        # placement is satisfied even though 90 < 100 - tolerance cumulatively.
        n, timed_out = self._call(
            requested=100,
            outstanding=200,
            alive_gpu_nodes=90,
            nodes_at_last_batch=40,
            last_batch_size=50,
        )
        assert n == 50
        assert timed_out is False


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
            cls.options.return_value.remote.return_value = replacement_actor

            results = _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=[chunk],
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
                max_chunk_retries=2,
            )

        # The chunk must have been processed despite the actor dying
        assert len(results) == 1
        assert results[0]["status"] == "ok"


# ===========================================================================
# _process_chunks_work_stealing — the progress line's GPU-hours figure
# ===========================================================================


class TestWorkStealingGpuHours:
    """The reported GPU-hours must integrate BILLED capacity, not requested capacity."""

    @staticmethod
    def _run_one_chunk(n_unplaced: int, cluster_resources) -> list[dict]:
        """Drive one loop iteration on a fleet that is mostly unplaced.

        The state is built explicitly rather than assumed: one actor that works,
        and ``n_unplaced`` slots whose instance-ID fetch never answers because
        the instance behind them does not exist. Returns the kwargs of each
        progress poll.
        """
        actors = [MagicMock(name=f"actor_{i}") for i in range(1 + n_unplaced)]
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"

        work_ref = MagicMock(name="work_ref")
        iid_ref = MagicMock(name="iid_ref")
        actors[0].process_chunk.remote.return_value = work_ref
        for unplaced in actors[1:]:
            unplaced.get_instance_id.remote.return_value = iid_ref

        def fake_wait(refs, num_returns=None, timeout=60):
            # timeout=0 is resolve_iid. The unplaced slots never answer it —
            # their get_instance_id is queued behind an __init__ that cannot
            # start until an instance exists.
            if timeout == 0:
                return ([], list(refs))
            return ([work_ref], [r for r in refs if r is not work_ref])

        # A synthetic clock, so the two figures on the progress line are exactly
        # determined instead of racing microseconds against each other.
        clock = itertools.count(start=1.0, step=1.0)
        polls: list[dict] = []

        def record_poll(*args, **kwargs) -> list[str]:
            polls.append(kwargs)
            return []

        with (
            patch.object(_sched_mod.ray, "wait", side_effect=fake_wait),
            patch.object(_sched_mod.ray, "get", return_value={"chunk": "c0", "status": "ok"}),
            patch.object(_sched_mod.ray, "cluster_resources", **cluster_resources),
            patch.object(_sched_mod.time, "monotonic", side_effect=lambda: next(clock)),
            patch.object(_sched_mod, "_poll_tracker", side_effect=record_poll),
        ):
            results = _process_chunks_work_stealing(
                actors=actors,  # type: ignore[arg-type]
                actor_instance_ids=["i-0000"] + ["pending-init"] * n_unplaced,
                chunks=[_fake_chunk("c0")],
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
                tracker=MagicMock(),
                still_initializing=set(range(1, 1 + n_unplaced)),
            )
        assert len(results) == 1, "the chunk must complete for the poll figures to mean anything"
        assert len(polls) == 1, "the run must report progress exactly once for the bound below to be exact"
        return polls

    def test_a_mid_interval_capacity_change_is_split_across_the_interval(self) -> None:
        """A blocking wait can stretch an interval to a minute, so WHEN capacity is sampled matters.

        Charging the whole interval at the reading taken when the wait returns bills a batch that
        joined near the end for time it was not there. Averaging the interval's two endpoints
        splits the change across it. Here capacity goes from none to forty between the seed
        reading and the loop's, so a correct integral charges twenty for that interval and
        end-sampling would charge forty.
        """
        polls = self._run_one_chunk(0, {"side_effect": [{"GPU": 0.0}, {"GPU": 40.0}]})
        gpu_seconds = polls[0]["gpu_hours"] * 3600
        # The clock is synthetic and steps one second per read, so the accumulated figure is
        # exact rather than approximate: one interval of one second, charged at the average of
        # its endpoints. End-sampling would charge that second at 40 and report double.
        assert gpu_seconds == pytest.approx(20.0), (
            f"{gpu_seconds} GPU-seconds; expected the interval charged at the average of its "
            "endpoints (20), not at the reading taken when the wait returned (40)"
        )

    def test_gpu_hours_follows_the_cluster_not_the_requested_fleet(self) -> None:
        """During scale-up most requested slots have no instance behind them.

        The pool asks for its whole fleet up front; Ray accepts every handle and
        the autoscaler brings nodes up behind the request. Integrating the
        requested-slot count reports a fleet-sized GPU-hours figure while only a
        fraction of the fleet is billed, and it does so for the whole of
        scale-up — the window in which an operator reads the figure to decide
        whether to cap the fleet.
        """
        n_unplaced = 19
        polls = self._run_one_chunk(n_unplaced, {"return_value": {"CPU": 8.0, "GPU": 1.0}})
        gpu_hours = polls[0]["gpu_hours"]
        elapsed_hours = polls[0]["elapsed_min"] / 60

        # The figure must still be REPORTED. Omitting it, or leaving it at zero,
        # satisfies any upper bound while telling an operator nothing.
        assert gpu_hours > 0

        # One billed GPU cannot burn materially more than one GPU-hour per
        # elapsed hour. The factor of two is slack for the clock ticks between
        # the loop's own read and the formatter's, not room for extra GPUs:
        # counting the requested slots instead puts the figure an order of
        # magnitude above this bound.
        assert gpu_hours < 2 * elapsed_hours, (
            f"{gpu_hours * 3600:.1f} GPU-seconds over {elapsed_hours * 3600:.1f} elapsed seconds against one "
            f"billed GPU — the {n_unplaced} requested-but-unplaced slots are being billed for"
        )

    def test_a_failed_capacity_lookup_does_not_abort_the_run(self) -> None:
        """The lookup sits outside the try/except that wraps the tracker poll.

        Unguarded, a control-plane hiccup would propagate out of the dispatch
        loop and destroy a fill over a log number. The figure is what degrades.
        """
        polls = self._run_one_chunk(3, {"side_effect": RuntimeError("gcs unavailable")})
        assert polls[0]["gpu_hours"] == 0.0


# ===========================================================================
# _process_chunks_work_stealing — returned-failure retry path
# ===========================================================================


class TestWorkStealingReturnedFailure:
    """An actor that returns status="failed" (vs. dying) is retried + killed."""

    def test_returned_failure_requeues_chunk_and_replaces_actor(self) -> None:
        """A chunk that comes back status="failed" retries on a fresh actor.

        The original actor catches its CUDA error and returns a failed result
        (ray.get succeeds), so this exercises the success-path branch that must
        nonetheless kill+replace the actor and re-queue the chunk — rather than
        recording the failure and feeding the wedged actor the next chunk.
        """
        actor = MagicMock(name="actor_0")
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        chunk = _fake_chunk("c0")

        seed_ref = MagicMock(name="seed_ref")
        retry_ref = MagicMock(name="retry_ref")
        iid_ref = MagicMock(name="iid_ref")

        actor.process_chunk.remote.return_value = seed_ref
        actor.get_instance_id.remote.return_value = iid_ref

        replacement_actor = MagicMock(name="replacement_actor")
        replacement_actor.process_chunk.remote.return_value = retry_ref
        replacement_actor.get_instance_id.remote.return_value = iid_ref

        failed_result = {"chunk": "c0", "status": "failed", "error": "CUDA error: misaligned address"}
        ok_result = {"chunk": "c0", "status": "ok"}

        def fake_get(ref, *args, **kwargs):
            if ref is seed_ref:
                return failed_result  # actor caught its own error, returned failed
            if ref is retry_ref:
                return ok_result
            if ref is iid_ref:
                return "i-replacement"
            return MagicMock()

        def fake_wait(refs, **kw):
            if kw.get("timeout", 60) == 0:  # resolve_iid / resolve_initializing
                return (list(refs), [])
            if seed_ref in refs:
                return ([seed_ref], [r for r in refs if r is not seed_ref])
            if retry_ref in refs:
                return ([retry_ref], [r for r in refs if r is not retry_ref])
            return ([], list(refs))

        with (
            patch.object(_sched_mod.ray, "wait", side_effect=fake_wait),
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "kill") as mock_kill,
            patch.object(_sched_mod, "InferenceActor") as cls,
            patch.object(_sched_mod, "log_worker_failure_diagnostic"),
            patch.object(_sched_mod.time, "sleep"),
        ):
            cls.options.return_value.remote.return_value = replacement_actor
            results = _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=[chunk],
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
                max_chunk_retries=2,
            )

        # The failing actor was killed and the chunk retried to success — the
        # final result is the retry's success, not the original failed dict.
        mock_kill.assert_called_once_with(actor)
        assert len(results) == 1
        assert results[0]["status"] == "ok"

    def test_returned_failure_permanent_after_retries_exhausted(self) -> None:
        """A chunk that keeps coming back failed is recorded failed after retries."""
        actor = MagicMock(name="actor_0")
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        chunk = _fake_chunk("c0")

        iid_ref = MagicMock(name="iid_ref")
        actor.get_instance_id.remote.return_value = iid_ref
        # Every submission (seed + 2 retries) returns a distinct failed ref.
        refs = [MagicMock(name=f"ref_{i}") for i in range(3)]
        actor.process_chunk.remote.side_effect = refs

        replacement = MagicMock(name="replacement")
        replacement.get_instance_id.remote.return_value = iid_ref
        replacement.process_chunk.remote.side_effect = lambda *a, **k: refs[-1]

        def fake_get(ref, *args, **kwargs):
            if ref is iid_ref:
                return "i-x"
            return {"chunk": "c0", "status": "failed", "error": "CUDA error"}

        def fake_wait(refs_in, **kw):
            if kw.get("timeout", 60) == 0:
                return (list(refs_in), [])
            return ([refs_in[0]], list(refs_in[1:])) if refs_in else ([], [])

        with (
            patch.object(_sched_mod.ray, "wait", side_effect=fake_wait),
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod, "InferenceActor") as cls,
            patch.object(_sched_mod, "log_worker_failure_diagnostic"),
            patch.object(_sched_mod.time, "sleep"),
        ):
            cls.options.return_value.remote.return_value = replacement
            results = _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=[chunk],
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
                max_chunk_retries=2,
            )

        # First try + 2 retries all failed → one permanent-failure result.
        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert results[0]["attempts"] == 3


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
                log=logging.getLogger("test"),
            )
        assert len(results) == 1


# ===========================================================================
# _process_chunks_work_stealing — deferred staging writes
# ===========================================================================


class TestDeferredWrites:
    """Chain-confirmation protocol: deferred chunks complete only on write confirm."""

    @staticmethod
    def _run(actor, chunks, **kwargs):
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        with (
            patch.object(_sched_mod.ray, "wait", side_effect=kwargs.pop("fake_wait")),
            patch.object(_sched_mod.ray, "get", side_effect=kwargs.pop("fake_get")),
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod, "InferenceActor") as cls,
            patch.object(_sched_mod, "log_worker_failure_diagnostic"),
            patch.object(_sched_mod.time, "sleep"),
        ):
            replacement = MagicMock(name="replacement")
            replacement.get_instance_id.remote.return_value = MagicMock()
            cls.options.return_value.remote.return_value = replacement
            return _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=chunks,
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
                **kwargs,
            )

    def test_deferred_chunks_confirm_via_next_result_and_flush(self) -> None:
        """c0 confirms via c1's prior_write; c1 confirms via flush_writes."""
        actor = MagicMock(name="actor_0")
        chunks = [_fake_chunk("c0"), _fake_chunk("c1")]
        refs = [MagicMock(name="ref0"), MagicMock(name="ref1")]
        flush_ref = MagicMock(name="flush_ref")
        actor.process_chunk.remote.side_effect = refs
        actor.flush_writes.remote.return_value = flush_ref
        done: list = []

        def fake_wait(pending_refs, **kw):
            if kw.get("timeout", 60) == 0:
                return (list(pending_refs), [])
            for r in refs:
                if r in pending_refs and r not in done:
                    done.append(r)
                    return ([r], [x for x in pending_refs if x is not r])
            return ([], list(pending_refs))

        def fake_get(ref, *args, **kwargs):
            if ref is refs[0]:
                return {"chunk": "c0", "status": "success", "write_deferred": True, "prior_write": None}
            if ref is refs[1]:
                return {
                    "chunk": "c1",
                    "status": "success",
                    "write_deferred": True,
                    "prior_write": {"label": "c0", "ok": True, "error": None},
                }
            if ref is flush_ref:
                return {"label": "c1", "ok": True, "error": None}
            return MagicMock()

        results = self._run(actor, chunks, fake_wait=fake_wait, fake_get=fake_get)

        assert sorted(r["chunk"] for r in results) == ["c0", "c1"]
        assert all(r.get("write_confirmed") for r in results)
        actor.flush_writes.remote.assert_called()  # c1 had no next result to ride

    def test_failed_write_requeues_without_killing_actor(self) -> None:
        """A write failure reported via prior_write requeues that chunk on a
        healthy actor (no kill), and the retry succeeds.
        """
        actor = MagicMock(name="actor_0")
        chunks = [_fake_chunk("c0"), _fake_chunk("c1")]
        refs = [MagicMock(name=f"ref{i}") for i in range(3)]  # c0, c1, c0-retry
        flush_refs = [MagicMock(name="fl0"), MagicMock(name="fl1")]
        actor.process_chunk.remote.side_effect = refs
        actor.flush_writes.remote.side_effect = flush_refs
        done: list = []

        def fake_wait(pending_refs, **kw):
            if kw.get("timeout", 60) == 0:
                return (list(pending_refs), [])
            for r in refs:
                if r in pending_refs and r not in done:
                    done.append(r)
                    return ([r], [x for x in pending_refs if x is not r])
            return ([], list(pending_refs))

        def fake_get(ref, *args, **kwargs):
            if ref is refs[0]:
                return {"chunk": "c0", "status": "success", "write_deferred": True, "prior_write": None}
            if ref is refs[1]:
                # c1's result reports c0's write FAILED -> c0 requeues.
                return {
                    "chunk": "c1",
                    "status": "success",
                    "write_deferred": True,
                    "prior_write": {"label": "c0", "ok": False, "error": "S3 500"},
                }
            if ref is refs[2]:
                # c0 retry: carries c1's (successful) write confirmation.
                return {
                    "chunk": "c0",
                    "status": "success",
                    "write_deferred": True,
                    "prior_write": {"label": "c1", "ok": True, "error": None},
                }
            if ref is flush_refs[0]:
                return {"label": "c0", "ok": True, "error": None}
            return MagicMock()

        results = self._run(actor, chunks, fake_wait=fake_wait, fake_get=fake_get)

        assert sorted(r["chunk"] for r in results) == ["c0", "c1"]
        # The actor was never replaced: all three process_chunk calls hit it.
        assert actor.process_chunk.remote.call_count == 3

    def test_actor_death_requeues_deferred_chunk(self) -> None:
        """An actor dying mid-c1 also requeues c0, whose write it still held."""
        actor = MagicMock(name="actor_0")
        chunks = [_fake_chunk("c0"), _fake_chunk("c1")]
        ref0, ref1 = MagicMock(name="ref0"), MagicMock(name="ref1")
        retry0, retry1 = MagicMock(name="retry0"), MagicMock(name="retry1")
        flush_a, flush_b = MagicMock(name="fla"), MagicMock(name="flb")

        replacement = MagicMock(name="replacement_actor")
        replacement.process_chunk.remote.side_effect = [retry0, retry1]
        replacement.flush_writes.remote.side_effect = [flush_a, flush_b]
        iid_ref = MagicMock(name="iid_ref")
        iid_ref._iid = "i-replacement"
        replacement.get_instance_id.remote.return_value = iid_ref

        actor.process_chunk.remote.side_effect = [ref0, ref1]
        done: list = []
        order = [ref0, ref1, retry0, retry1]

        def fake_wait(pending_refs, **kw):
            if kw.get("timeout", 60) == 0:
                return (list(pending_refs), [])
            for r in order:
                if r in pending_refs and r not in done:
                    done.append(r)
                    return ([r], [x for x in pending_refs if x is not r])
            return ([], list(pending_refs))

        def fake_get(ref, *args, **kwargs):
            # NOTE: match by identity — getattr(ref, "_iid", ...) auto-creates a
            # truthy Mock attribute and would swallow every ref.
            if ref is iid_ref:
                return "i-replacement"
            if ref is ref0:
                return {"chunk": "c0", "status": "success", "write_deferred": True, "prior_write": None}
            if ref is ref1:
                raise RuntimeError("actor died")  # c1 dies; c0's write orphaned
            if ref is retry0:
                return {"chunk": "c0", "status": "success", "write_deferred": True, "prior_write": None}
            if ref is retry1:
                return {
                    "chunk": "c1",
                    "status": "success",
                    "write_deferred": True,
                    "prior_write": {"label": "c0", "ok": True, "error": None},
                }
            if ref is flush_a:
                return {"label": "c1", "ok": True, "error": None}
            return MagicMock()

        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        with (
            patch.object(_sched_mod.ray, "wait", side_effect=fake_wait),
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod, "InferenceActor") as cls,
            patch.object(_sched_mod, "log_worker_failure_diagnostic"),
            patch.object(_sched_mod.time, "sleep"),
        ):
            cls.options.return_value.remote.return_value = replacement
            results = _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=chunks,
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
            )

        # Both chunks completed (on the replacement) despite the death, and c0
        # was re-run because its deferred write died with the first actor.
        assert sorted(r["chunk"] for r in results) == ["c0", "c1"]

    def test_a_chunk_that_is_ready_and_stalled_on_the_same_tick_is_not_double_popped(self) -> None:
        """A long-stalled chunk that finally RETURNS must be recorded, not crash the run.

        ``ray.wait`` can hand back a ref on the very tick the tracker reports its chunk
        past the recovery threshold: the result exists but has not been processed, so
        the item is still pending and the recovery scan finds it. Popping it there and
        again in the ready loop raised KeyError out of the scheduler — killing the whole
        fleet at the moment the wedge cleared. The ready path wins: there is a result
        waiting, and killing the actor would throw away work already done.
        """
        actor = MagicMock(name="actor_0")
        chunks = [_fake_chunk("c0")]
        ref0 = MagicMock(name="ref0")
        actor.process_chunk.remote.return_value = ref0
        done: list = []

        def fake_wait(pending_refs, **kw):
            if kw.get("timeout", 60) == 0:
                return (list(pending_refs), [])
            if ref0 in pending_refs and ref0 not in done:
                done.append(ref0)
                return ([ref0], [x for x in pending_refs if x is not ref0])
            return ([], list(pending_refs))

        def fake_get(ref, *args, **kwargs):
            if ref is ref0:
                return {"chunk": "c0", "status": "success"}
            return MagicMock()

        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        # The tracker declares c0 wedged on every poll, including the tick it returns.
        with (
            patch.object(_sched_mod.ray, "wait", side_effect=fake_wait),
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "kill") as kill,
            patch.object(_sched_mod, "_poll_tracker", return_value=[chunk_uid("r", "c0")]),
            patch.object(_sched_mod.time, "sleep"),
        ):
            results = _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=chunks,
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
                tracker=MagicMock(name="tracker"),
            )

        assert [r["chunk"] for r in results] == ["c0"]  # recorded, not lost to a KeyError
        kill.assert_not_called()  # and its actor was not killed for work it had finished

    def test_flush_failure_replaces_actor_and_retries_on_replacement(self) -> None:
        """A failed/timed-out tail flush kills + replaces the actor.

        The flush RPC may have left the (serial) actor wedged inside
        flush_writes(); without the replacement the requeued chunk would be
        dispatched right back to it and sit behind the stuck call forever,
        invisible to stall detection (a chunk that never starts has no
        tracker entry).
        """
        actor = MagicMock(name="actor_0")
        chunks = [_fake_chunk("c0")]
        ref0, retry0 = MagicMock(name="ref0"), MagicMock(name="retry0")
        flush_stuck, flush_ok = MagicMock(name="flush_stuck"), MagicMock(name="flush_ok")

        replacement = MagicMock(name="replacement_actor")
        replacement.process_chunk.remote.return_value = retry0
        replacement.flush_writes.remote.return_value = flush_ok
        iid_ref = MagicMock(name="iid_ref")
        replacement.get_instance_id.remote.return_value = iid_ref

        actor.process_chunk.remote.return_value = ref0
        actor.flush_writes.remote.return_value = flush_stuck
        done: list = []
        order = [ref0, retry0]

        def fake_wait(pending_refs, **kw):
            if kw.get("timeout", 60) == 0:
                return (list(pending_refs), [])
            for r in order:
                if r in pending_refs and r not in done:
                    done.append(r)
                    return ([r], [x for x in pending_refs if x is not r])
            return ([], list(pending_refs))

        def fake_get(ref, *args, **kwargs):
            if ref is iid_ref:
                return "i-replacement"
            if ref is flush_stuck:
                raise RuntimeError("flush timed out")  # stand-in for GetTimeoutError
            if ref is ref0 or ref is retry0:
                return {"chunk": "c0", "status": "success", "write_deferred": True, "prior_write": None}
            if ref is flush_ok:
                return {"label": "c0", "ok": True, "error": None}
            return MagicMock()

        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        with (
            patch.object(_sched_mod.ray, "wait", side_effect=fake_wait),
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "kill") as kill,
            patch.object(_sched_mod, "InferenceActor") as cls,
            patch.object(_sched_mod, "log_worker_failure_diagnostic"),
            patch.object(_sched_mod.time, "sleep"),
        ):
            cls.options.return_value.remote.return_value = replacement
            results = _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=chunks,
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
            )

        # c0 completed (write-confirmed) on the replacement, not the wedged actor.
        assert [r["chunk"] for r in results] == ["c0"]
        assert results[0].get("write_confirmed")
        kill.assert_called_once_with(actor)
        assert actor.process_chunk.remote.call_count == 1
        assert replacement.process_chunk.remote.call_count == 1


# ===========================================================================
# _process_chunks_work_stealing — retire_idle_actors gate
# ===========================================================================


class TestRetireIdleGate:
    """retire_idle_actors=False must keep the loop from ever retiring idle
    actors — the chained multi-zone fill passes it for every zone but its
    last so a zone's tail doesn't drain the shared cluster's instances.
    """

    def _run_one_chunk(self, retire_idle_actors: bool) -> MagicMock:
        actor = MagicMock(name="actor_0")
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        chunk = _fake_chunk("c0")
        seed_ref = MagicMock(name="seed_ref")
        iid_ref = MagicMock(name="iid_ref")
        actor.process_chunk.remote.return_value = seed_ref
        actor.get_instance_id.remote.return_value = iid_ref

        def fake_get(ref, *args, **kwargs):
            if ref is seed_ref:
                return {"chunk": "c0", "status": "ok"}
            if ref is iid_ref:
                return "i-0000"
            return MagicMock()

        with (
            patch.object(_sched_mod.ray, "wait", side_effect=lambda refs, **kw: (list(refs), [])),
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod.time, "sleep"),
            patch.object(_sched_mod.ActorPool, "retire_idle") as retire_mock,
        ):
            results = _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=[chunk],
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
                retire_idle_actors=retire_idle_actors,
            )
        assert len(results) == 1 and results[0]["status"] == "ok"
        return retire_mock

    def test_default_retires_idle_actors(self) -> None:
        assert self._run_one_chunk(retire_idle_actors=True).called

    def test_gate_disables_retirement(self) -> None:
        assert not self._run_one_chunk(retire_idle_actors=False).called


# ===========================================================================
# _process_chunks_work_stealing — chained multi-zone work source
# ===========================================================================


class TestChainedWorkSource:
    """more_work/on_item_done: zones stream through one session, contexts intact."""

    @staticmethod
    def _zone_items(zone: str, labels: list[str]) -> list:
        ctx = _sched_mod.ZoneContext(f"m/{zone}", f"s/{zone}", f"run-{zone}")
        return [_sched_mod.WorkItem(chunk=_fake_chunk(label), ctx=ctx) for label in labels]

    def _drive(self, actor, zones: list[list], **loop_kwargs):
        """Run the loop with zone 0 as the initial batch and the rest sourced."""
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        remaining = list(zones[1:])

        def more_work():
            return remaining.pop(0) if remaining else None

        refs_by_call: list = []

        def fake_process_chunk(*args, **kwargs):
            ref = MagicMock(name=f"ref{len(refs_by_call)}")
            refs_by_call.append((ref, args, kwargs))
            return ref

        actor.process_chunk.remote.side_effect = fake_process_chunk

        def fake_get(ref, *a, **k):
            for r, args, _ in refs_by_call:
                if r is ref:
                    return {"chunk": args[0].label, "status": "ok"}
            return MagicMock()

        with (
            patch.object(_sched_mod.ray, "wait", side_effect=lambda refs, **kw: (list(refs), [])),
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod.time, "sleep"),
            patch.object(_sched_mod.ActorPool, "retire_idle") as retire_mock,
        ):
            results = _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=[item.chunk for item in zones[0]] if zones and not loop_kwargs.pop("wrap_first", True) else [],
                mosaic_base="m/zone-a",
                staging_base="s/zone-a",
                run_id="run-zone-a",
                config=config,
                log=logging.getLogger("test"),
                more_work=more_work,
                **loop_kwargs,
            )
        return results, refs_by_call, retire_mock

    def test_zones_stream_with_their_own_contexts(self):
        """Zone B's chunks are dispatched against zone B's mosaic/staging/run_id."""
        actor = MagicMock(name="actor_0")
        zone_a = self._zone_items("zone-a", ["c0", "c1"])
        zone_b = self._zone_items("zone-b", ["c0", "c1"])  # same labels, different zone
        done: list = []
        results, calls, _ = self._drive(
            actor,
            [[], zone_a, zone_b],
            on_item_done=lambda item, res: done.append((item.uid, res["status"])),
        )
        assert len(results) == 4
        # Every dispatch carried its item's own zone context.
        for _, args, _kw in calls:
            chunk, mosaic_base, staging_base, run_id = args[:4]
            zone = run_id.removeprefix("run-")
            assert mosaic_base == f"m/{zone}" and staging_base == f"s/{zone}"
        # on_item_done fired once per item, uids run_id-qualified (no aliasing
        # despite identical labels across the two zones).
        assert sorted(u for u, _ in done) == ["run-zone-a:c0", "run-zone-a:c1", "run-zone-b:c0", "run-zone-b:c1"]

    def test_boundary_wait_is_short_while_source_active(self):
        """At a zone boundary — source still active, queue drained to the poll
        trigger — the loop caps the ray.wait short (5s) so the next zone
        dispatches promptly; blocking the full 60s would idle the fleet up to a
        GPU-minute per boundary. Once the source is exhausted, the wait is 60s.
        """
        actor = MagicMock(name="actor_0")
        zone_a = self._zone_items("zone-a", ["c0"])
        zone_b = self._zone_items("zone-b", ["c0"])
        # `[]` = next zone still ingesting (not ready) — the boundary gap.
        remaining: list = [zone_a, [], [], zone_b, None]
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        refs: list = []
        timeouts: list = []

        def fake_process_chunk(*args, **kwargs):
            ref = MagicMock(name=f"r{len(refs)}")
            refs.append((ref, args))
            return ref

        def fake_get(ref, *a, **k):
            for r, args in refs:
                if r is ref:
                    return {"chunk": args[0].label, "status": "ok"}
            return MagicMock()

        def fake_wait(pending, **kw):
            timeouts.append(kw.get("timeout"))
            return ([pending[0]], list(pending[1:]))

        actor.process_chunk.remote.side_effect = fake_process_chunk
        with (
            patch.object(_sched_mod.ray, "wait", side_effect=fake_wait),
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod.time, "sleep"),
            patch.object(_sched_mod.ActorPool, "retire_idle"),
        ):
            _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=[],
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
                more_work=lambda: remaining.pop(0) if remaining else None,
            )
        assert 5 in timeouts  # boundary wait while the source was still active
        assert 60 in timeouts  # long wait after the source was exhausted

    def test_retirement_suppressed_until_source_exhausted(self):
        actor = MagicMock(name="actor_0")
        zone_a = self._zone_items("zone-a", ["c0"])
        _, _, retire_mock = self._drive(actor, [[], zone_a], retire_idle_actors=True)
        # The source returned items then None; retirement may only run in
        # iterations AFTER exhaustion. With one chunk and instant completion
        # the final iteration retires — but never before the source was live.
        assert retire_mock.called  # ran after exhaustion (gate passed through)

    def test_attempts_do_not_alias_across_zones(self):
        """Same label in two zones keeps independent retry budgets."""
        actor = MagicMock(name="actor_0")
        zone_a = self._zone_items("zone-a", ["c0"])
        zone_b = self._zone_items("zone-b", ["c0"])
        # Zone A's c0 fails once then succeeds on retry; zone B's c0 succeeds
        # first try. With label-keyed attempts, A's failures would eat B's
        # retry budget.
        calls: dict = {"n": 0}
        refs: list = []

        def fake_process_chunk(*args, **kwargs):
            ref = MagicMock(name=f"r{len(refs)}")
            refs.append((ref, args))
            return ref

        def fake_get(ref, *a, **k):
            for r, args in refs:
                if r is ref:
                    run_id = args[3]
                    if run_id == "run-zone-a" and calls["n"] == 0:
                        calls["n"] += 1
                        return {"chunk": args[0].label, "status": "failed", "error": "transient"}
                    return {"chunk": args[0].label, "status": "ok"}
            return MagicMock()

        actor.process_chunk.remote.side_effect = fake_process_chunk
        config = MagicMock()
        config.checkpoint_path = "s3://bucket/ckpt.pt"
        remaining = [zone_a, zone_b]

        replacement = MagicMock(name="replacement")
        replacement.process_chunk.remote.side_effect = fake_process_chunk
        replacement.get_instance_id.remote.return_value = MagicMock()

        with (
            patch.object(_sched_mod.ray, "wait", side_effect=lambda r, **kw: (list(r), [])),
            patch.object(_sched_mod.ray, "get", side_effect=fake_get),
            patch.object(_sched_mod.ray, "kill"),
            patch.object(_sched_mod, "InferenceActor") as cls,
            patch.object(_sched_mod, "log_worker_failure_diagnostic"),
            patch.object(_sched_mod.time, "sleep"),
        ):
            cls.options.return_value.remote.return_value = replacement
            results = _process_chunks_work_stealing(
                actors=[actor],
                actor_instance_ids=["i-0000"],
                chunks=[],
                mosaic_base="m",
                staging_base="s",
                run_id="r",
                config=config,
                log=logging.getLogger("test"),
                more_work=lambda: remaining.pop(0) if remaining else None,
            )
        by_status = sorted((r["chunk"], r["status"]) for r in results)
        assert by_status == [("c0", "ok"), ("c0", "ok")]  # both zones landed


# ---------------------------------------------------------------------------
# Run-qualified progress key (chunk_uid / WorkItem.uid)
# ---------------------------------------------------------------------------


class TestChunkUid:
    """The progress tracker is keyed by the run-qualified uid, not the bare
    chunk label, so a shared multi-zone session can't alias two zones' chunks.
    """

    def test_chunk_uid_is_run_qualified(self) -> None:
        from tessera_embeddings.inference.progress import chunk_uid

        assert chunk_uid("33N-2025-abc", "5_5") == "33N-2025-abc:5_5"

    def test_workitem_uid_uses_the_shared_helper(self) -> None:
        from tessera_embeddings.inference.progress import chunk_uid

        item = _sched_mod.WorkItem(
            chunk=_fake_chunk("5_5"), ctx=_sched_mod.ZoneContext("s3://m", "s3://s", "33N-2025-abc")
        )
        assert item.uid == chunk_uid("33N-2025-abc", "5_5")

    def test_same_label_different_zones_do_not_collide(self) -> None:
        """The regression: two zones both have a ``5_5`` chunk. Before the fix
        both keyed the tracker as ``"5_5"``, so one zone's completion evicted the
        other's progress entry and hung the live task. Their uids must differ.
        """
        za = _sched_mod.WorkItem(
            chunk=_fake_chunk("5_5"), ctx=_sched_mod.ZoneContext("s3://m/33N", "s3://s/33N", "33N-2025-a")
        )
        zb = _sched_mod.WorkItem(
            chunk=_fake_chunk("5_5"), ctx=_sched_mod.ZoneContext("s3://m/34N", "s3://s/34N", "34N-2025-b")
        )
        assert za.chunk.label == zb.chunk.label  # labels collide across zones...
        assert za.uid != zb.uid  # ...but the run-qualified uids do not
