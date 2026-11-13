"""Keeping an idle coordinator session current during a long fork phase (``storage.session_catch_up``).

Everything but the path predicate runs against a real repository, because the whole question is
what icechunk does with a session whose base has moved while its forks were outstanding.
"""

from __future__ import annotations

import icechunk
import pytest
import zarr

from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.storage import global_store, session_catch_up
from tessera_embeddings.storage.shard_writer import commit_year_attrs
from tessera_embeddings.storage.zone_grid import ZoneSpec

_CHUNK = 256
_SHARD = 512
# 512² shards of 256² inner chunks, 8 bands: a chunk write is a few hundred KB.
_SMALL = StoreLayout(
    name="small",
    arrays={
        "embeddings": ArrayLayout(DIMS_4D, (1, _CHUNK, _CHUNK, 8), "int8", 0, "zstd", shards=(1, 512, 512, 8)),
        "scales": ArrayLayout(DIMS_3D, (1, _CHUNK, _CHUNK), "float32", float("nan"), "pcodec", shards=(1, 512, 512)),
    },
    bands=8,
)
_ZONES = (
    ZoneSpec("32601", "N", 1, (0.0, 5_120.0), (0.0, 10_240.0)),
    ZoneSpec("32701", "S", 1, (0.0, 5_120.0), (1_105_920.0, 1_116_160.0)),
)


def _seed(tmp_path):
    """A global repo with zone groups 01N and 01S."""
    repo = global_store.create_global_repo(str(tmp_path / "g.icechunk"))
    global_store.seed_zone_groups(repo, _ZONES, years=(2025,), layout=_SMALL)
    return repo


def _commit_to(repo, group, value):
    """Another writer's commit to *group*, landing on the branch tip."""
    session = repo.writable_session("main")
    zarr.open_group(session.store, mode="a")[group]["embeddings"][0, 0:_CHUNK, 0:_CHUNK, :] = value
    return session.commit(f"fill {group}")


class _FakeDiff:
    """A duck-typed :class:`icechunk.Diff` for the pure-function cases.

    Used ONLY for :func:`_diff_touches`, which is a predicate over path strings and has no
    icechunk behaviour to get wrong. Everything about `catch_up_to_branch` is exercised
    against a real repository below, because a stub of the thing under test cannot falsify it.
    """

    def __init__(self, *, chunks=(), groups=(), arrays=(), moved=()):
        # A chunk is a path, written at coordinate (0, 0, 0), or a ``(path, coords)`` pair.
        self.updated_chunks = dict(c if isinstance(c, tuple) else (c, [[0, 0, 0]]) for c in chunks)
        self.updated_groups = set(groups)
        self.new_groups = set()
        self.deleted_groups = set()
        self.new_arrays = set(arrays)
        self.updated_arrays = set()
        self.deleted_arrays = set()
        # EIGHT members, matching icechunk.Diff exactly. The stub used to have seven, which is
        # precisely how `moved_nodes` went unchecked in the code as well — a stub that is a
        # subset of the real type makes an incomplete guard look complete.
        self.moved_nodes = list(moved)


@pytest.mark.parametrize(
    ("diff", "touched"),
    [
        (_FakeDiff(chunks=["/01N/embeddings"]), True),
        # `mark <zone> year N complete` writes group attrs and no chunks at all, so a chunks-only
        # test would wave through the second of every cell's two commits.
        (_FakeDiff(groups=["/01N"]), True),
        (_FakeDiff(arrays=["/01N/extra"]), True),
        # A rename populates `moved_nodes` and NOTHING else — verified against a real repo — so a
        # guard reading the other seven fields waves it through. Both ends of the move count.
        (_FakeDiff(moved=[("/01N/embeddings", "/01N/emb2")]), True),
        (_FakeDiff(moved=[("/01S/embeddings", "/01N/stolen")]), True),
        (_FakeDiff(chunks=["/01S/embeddings"]), False),
        (_FakeDiff(), False),
        # "/01N2/..." starts with "01N" as a STRING but is a different zone; a bare prefix match
        # would block every catch-up behind an unrelated neighbour.
        (_FakeDiff(chunks=["/01N2/embeddings"]), False),
    ],
    ids=["chunk", "group-attrs", "new-array", "rename-own", "rename-into", "other-zone", "empty", "prefix-only"],
)
def test_which_changes_count_as_somebody_else_writing_our_cell(diff, touched):
    assert session_catch_up._diff_touches(diff, "01N") is touched


@pytest.mark.parametrize(
    ("diff", "touched"),
    [
        (_FakeDiff(chunks=["/01N/embeddings"], groups=["/01N"]), False),  # another shard + group attrs
        (_FakeDiff(arrays=["/01N/extra"]), True),  # structure still counts
        (_FakeDiff(chunks=[("/01N/time", [[0]])]), True),  # so does a coordinate array's chunk
    ],
)
def test_given_its_shards_only_a_collision_is_ours(diff, touched):
    # ADR 038: the other half of a column fills the same group at once, writing other shards
    # and marking itself in the group's attrs. Neither can collide with a shard-only session.
    assert session_catch_up._diff_touches(diff, "01N", {(0, 1, 0)}) is touched
    assert session_catch_up._diff_touches(_FakeDiff(chunks=["/01N/embeddings"]), "01N", {(0, 0, 0)})


def test_the_stub_carries_every_member_of_the_real_diff():
    """A stub missing a field cannot fail on that field, which is how one got missed.

    Pins the stub to `icechunk.Diff` so adding a member upstream breaks this rather than
    quietly shrinking what the guard is tested against.
    """
    real = {a for a in dir(icechunk.Diff) if not a.startswith("_") and a != "is_empty"}
    assert real <= set(vars(_FakeDiff(chunks=["/x/y"]))), f"stub is missing {real - set(vars(_FakeDiff()))}"


class TestCatchUpToBranch:
    """Keeping an idle coordinator session current — and refusing to when that would hide a conflict.

    Against a real repository, because the whole question is what icechunk does with a session
    whose base has moved while its forks were outstanding.
    """

    def test_nothing_to_do_when_already_at_the_tip(self, tmp_path):
        repo = _seed(tmp_path)
        session = repo.writable_session("main")
        assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "current"

    def test_advances_past_a_commit_to_another_zone(self, tmp_path):
        repo = _seed(tmp_path)
        session = repo.writable_session("main")
        base = session.snapshot_id
        tip = _commit_to(repo, "01S", 7)
        assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "advanced"
        assert session.snapshot_id == tip != base

    def test_a_blocked_catch_up_leaves_the_collision_for_the_commit_to_raise(self, tmp_path):
        """THE SAFETY PROPERTY, and the reason the guard exists.

        Moving the base past a commit is also a decision to stop checking it for conflicts.
        A catch-up without the guard sails past a competing writer and then overwrites it in
        silence; with the guard the commit still sees it and refuses, which is what icechunk
        does today and what must not regress.
        """
        repo = _seed(tmp_path)
        session = repo.writable_session("main")
        base = session.snapshot_id

        # ORDER MATTERS, and matching production is the whole point. During the fork phase
        # the coordinator's session is EMPTY — the forks have not been merged — so the
        # catch-up runs with nothing to compare, which is exactly why an unguarded one sails
        # past a competing writer without noticing. Writing to the session first would make
        # `rebase` raise for a different reason and the test would pass without testing this.
        _commit_to(repo, "01N", 7)  # a competing writer lands on OUR zone
        assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "blocked"
        assert session.snapshot_id == base, "a blocked catch-up must leave the base alone"

        # ...and only now does the write arrive, as the merge would deliver it.
        node = zarr.open_group(session.store, mode="a")
        node["01N"]["embeddings"][0, 0:_CHUNK, 0:_CHUNK, :] = 3
        with pytest.raises(icechunk.RebaseFailedError):
            session.commit("fill 01N", rebase_with=icechunk.ConflictDetector())

    def test_given_its_shards_it_advances_past_the_other_half_of_its_group(self, tmp_path):
        """ADR 038: two halves fill one group at once; each must keep catching up past the other.

        The other half writes shard (0, 0) and marks itself complete; this session owns shard
        (1, 0). The catch-up advances past both commits, and the commit that follows still
        lands both writes, because icechunk's own check finds nothing in common.
        """
        repo = _seed(tmp_path)
        session = repo.writable_session("main")
        _commit_to(repo, "01N", 7)
        commit_year_attrs(repo, "01N", 2025, run_id="other")
        assert session_catch_up.catch_up_to_branch(repo, session, "01N", shards={(0, 0, 0)}) == "blocked"
        assert session_catch_up.catch_up_to_branch(repo, session, "01N", shards={(0, 1, 0)}) == "advanced"
        zarr.open_group(session.store, mode="a")["01N"]["embeddings"][0, _SHARD : _SHARD + _CHUNK, 0:_CHUNK, :] = 3
        session.commit("fill 01N row 1", rebase_with=icechunk.ConflictDetector())
        emb = zarr.open_group(repo.readonly_session("main").store, mode="r")["01N"]["embeddings"]
        assert (emb[0, 0, 0, 0], emb[0, _SHARD, 0, 0]) == (7, 3)

    def test_repeated_catch_ups_keep_the_gap_at_zero(self, tmp_path):
        # The point of running it on a timer rather than once: after each unrelated commit
        # the session is back at the tip, so no single catch-up ever walks far.
        repo = _seed(tmp_path)
        session = repo.writable_session("main")
        for value in (1, 2, 3):
            tip = _commit_to(repo, "01S", value)
            assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "advanced"
            assert session.snapshot_id == tip


class TestCatchUpRaceAndBestEffort:
    """The two things that decide whether this is safe to run unattended for hours."""

    def test_advancing_past_an_unvetted_conflicting_commit_raises(self, tmp_path, monkeypatch):
        """`rebase` takes no target snapshot, so it lands on whatever the tip is when it runs.

        A commit that arrives between the diff check and the rebase is skipped without being
        vetted. It cannot be undone — an empty session's rebase never raises whatever it walks
        over — so the only correct outcome is to notice and fail loudly. Simulated by landing
        the competing commit inside the rebase call itself, which is exactly the window.
        """
        repo = _seed(tmp_path)
        session = repo.writable_session("main")
        _commit_to(repo, "01S", 1)  # vetted as clean

        real_rebase = type(session).rebase

        def rebase_with_a_latecomer(self, solver):
            _commit_to(repo, "01N", 9)  # slips into the window
            return real_rebase(self, solver)

        monkeypatch.setattr(type(session), "rebase", rebase_with_a_latecomer)
        with pytest.raises(session_catch_up.CaughtUpPastAConflictError, match="01N"):
            session_catch_up.catch_up_to_branch(repo, session, "01N")

    def test_best_effort_turns_a_failure_into_a_tally_entry(self, tmp_path, monkeypatch):
        # A rename anywhere in the store makes `rebase` raise even on an empty session, and a
        # reset branch makes `diff` raise. Neither may cost a three-hour fill.
        repo = _seed(tmp_path)
        session = repo.writable_session("main")
        monkeypatch.setattr(
            session_catch_up, "catch_up_to_branch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("s3 hiccup"))
        )
        assert session_catch_up.catch_up_best_effort(repo, session, "01N") == "failed"

    def test_a_failure_after_the_base_moved_is_not_best_effort(self, tmp_path, monkeypatch):
        """The rule is whether the base moved, NOT which exception was raised.

        `rebase` advances incrementally and can leave the session advanced when a later step
        throws, and the post-rebase verification is itself a network call. A handler keyed on
        exception type therefore swallows exactly the case that matters: commits crossed
        without being vetted, then forks merged onto that base, putting a same-group collision
        beyond the commit's conflict detection.
        """
        repo = _seed(tmp_path)
        session = repo.writable_session("main")
        _commit_to(repo, "01S", 1)
        before = session.snapshot_id

        def advance_then_fail(*args, **kwargs):
            session.rebase(icechunk.ConflictDetector())  # the base really moves
            raise RuntimeError("storage failed during the post-rebase check")

        monkeypatch.setattr(session_catch_up, "catch_up_to_branch", advance_then_fail)
        with pytest.raises(session_catch_up.CaughtUpPastAConflictError, match="never checked"):
            session_catch_up.catch_up_best_effort(repo, session, "01N")
        assert session.snapshot_id != before, "the test did not actually move the base"

    def test_best_effort_still_lets_the_unsafe_one_through(self, tmp_path, monkeypatch):
        """The one exception that must NOT be swallowed, or the guard is decorative."""
        repo = _seed(tmp_path)
        session = repo.writable_session("main")

        def unsafe(*a, **k):
            raise session_catch_up.CaughtUpPastAConflictError("advanced past a conflict")

        monkeypatch.setattr(session_catch_up, "catch_up_to_branch", unsafe)
        with pytest.raises(session_catch_up.CaughtUpPastAConflictError):
            session_catch_up.catch_up_best_effort(repo, session, "01N")


class TestTheGuardOnARealRepository:
    """The guard's coverage, checked against real icechunk behaviour."""

    def test_the_guard_is_asked_about_the_whole_gap_not_just_the_newest_commit(self, tmp_path):
        """Production is explicitly multi-commit — depths of 4, 8 and 16 were measured.

        Every other catch-up test creates exactly one intervening snapshot, so a guard that
        inspected only the most recent commit would look correct. Here the offending commit is
        the OLDER of two.
        """
        repo = _seed(tmp_path)
        session = repo.writable_session("main")
        _commit_to(repo, "01N", 7)  # ours, and now not the newest
        _commit_to(repo, "01S", 8)  # someone else's, newest
        assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "blocked"

    def test_every_enumerated_diff_field_exists_on_a_real_diff(self, tmp_path):
        """The enumeration IS the guard's coverage.

        Four of the six node fields have no stub test of their own, so dropping them from the
        tuple changes nothing any test sees. Asserting against a real `icechunk.Diff` means an
        upstream rename breaks this rather than silently narrowing what is checked.
        """
        repo = _seed(tmp_path)
        base = repo.lookup_branch("main")
        tip = _commit_to(repo, "01N", 5)
        real = repo.diff(from_snapshot_id=base, to_snapshot_id=tip)
        members = {a for a in dir(real) if not a.startswith("_")} - {"is_empty"}
        # EQUALITY, both directions. `hasattr` on each entry would catch an upstream rename but
        # not someone shortening the tuple, which is the likelier accident and the one that
        # narrows the guard in silence.
        handled_separately = {"updated_chunks", "moved_nodes"}
        assert set(session_catch_up._DIFF_NODE_FIELDS) | handled_separately == members, (
            f"the guard covers {set(session_catch_up._DIFF_NODE_FIELDS) | handled_separately} "
            f"but icechunk.Diff has {members}"
        )

    def test_the_rebase_detects_conflicts_rather_than_solving_them(self, tmp_path, monkeypatch):
        """A solver would resolve a collision silently — the exact failure the guard prevents.

        The guard is the first line; this argument is the second. Swapping `ConflictDetector`
        for `BasicConflictSolver` passed every other test.
        """
        repo = _seed(tmp_path)
        session = repo.writable_session("main")
        _commit_to(repo, "01S", 4)
        seen: list[object] = []
        real_rebase = type(session).rebase
        monkeypatch.setattr(
            type(session),
            "rebase",
            lambda self, solver: (seen.append(solver), real_rebase(self, solver))[1],
        )
        session_catch_up.catch_up_to_branch(repo, session, "01N")
        assert isinstance(seen[0], icechunk.ConflictDetector), f"rebased with {type(seen[0])}"
