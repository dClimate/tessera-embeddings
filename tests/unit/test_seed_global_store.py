"""seed_global_store flow: the recorded checkpoint identity default.

Exercised via ``.fn`` with the repo ops mocked, so no Icechunk store is created —
the point is only which ``model_version`` (→ ``checkpoint_id``) is stamped.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import icechunk
import pytest
import zarr.errors

import tessera_embeddings.orchestration.prefect.flows.seed_global_store as mod
from tessera_embeddings.config.inference import checkpoint_filename
from tessera_embeddings.config.paths import BucketPaths

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


def _capture_seed(monkeypatch) -> dict:
    """Force the create path and capture the model_version passed to seed_zone_groups."""
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-seed"))
    monkeypatch.setattr(mod, "create_global_repo", lambda *a, **k: object())

    def _raise_open(*a, **k):
        raise FileNotFoundError("no store yet")

    monkeypatch.setattr(mod, "open_global_repo", _raise_open)  # → create path, seeded=set()
    captured: dict = {}

    def _fake_seed(repo, todo, *, years, model_version, optical_min_obs, commit_msg):
        captured["model_version"] = model_version
        captured["optical_min_obs"] = optical_min_obs
        return "snapshot"

    monkeypatch.setattr(mod, "seed_zone_groups", _fake_seed)
    return captured


def test_seed_defaults_checkpoint_id_to_checkpoint_filename(monkeypatch):
    """model_version defaults to checkpoint_filename() so the fill's checkpoint gate
    is effective by default — geoemb:model alone can't distinguish aws vs mpc.
    """
    captured = _capture_seed(monkeypatch)
    mod.seed_global_store.fn(paths=_PATHS)
    assert captured["model_version"] == checkpoint_filename()


def test_seed_preserves_explicit_model_version(monkeypatch):
    """An explicit model_version overrides the checkpoint-filename default."""
    captured = _capture_seed(monkeypatch)
    mod.seed_global_store.fn(paths=_PATHS, model_version="custom-encoder-v2")
    assert captured["model_version"] == "custom-encoder-v2"


def test_seed_creates_on_missing_repo_icechunk_error(monkeypatch):
    """A genuinely-missing repo (IcechunkError "doesn't exist") takes the create path."""
    captured = _capture_seed(monkeypatch)

    def _raise_missing(*a, **k):
        raise icechunk.IcechunkError("the repository doesn't exist")

    monkeypatch.setattr(mod, "open_global_repo", _raise_missing)
    mod.seed_global_store.fn(paths=_PATHS)
    assert captured["model_version"] == checkpoint_filename()  # seed_zone_groups ran → create path taken


def test_seed_reraises_transient_icechunk_error(monkeypatch):
    """A transient IcechunkError (timeout/auth) must NOT be treated as a missing repo —
    it re-raises instead of creating a fresh repo against a store that may already exist.
    """
    _capture_seed(monkeypatch)

    def _raise_transient(*a, **k):
        raise icechunk.IcechunkError("connection timed out")

    monkeypatch.setattr(mod, "open_global_repo", _raise_transient)
    created: list = []
    monkeypatch.setattr(mod, "create_global_repo", lambda *a, **k: created.append(1) or object())
    with pytest.raises(icechunk.IcechunkError, match="timed out"):
        mod.seed_global_store.fn(paths=_PATHS)
    assert created == []  # never entered the create path


def test_seed_treats_rootless_existing_repo_as_unseeded(monkeypatch):
    """Idempotency edge: a repo that OPENS fine but has no root group yet (a prior
    run created + saved config, then crashed before the first seed commit) must be
    treated as unseeded and seeded, not raise. campaign_status reads a rootless
    store and raises GroupNotFoundError; the flow swallows that → seeded=set().
    """
    captured = _capture_seed(monkeypatch)  # mocks seed_zone_groups + logger + create
    monkeypatch.setattr(mod, "open_global_repo", lambda *a, **k: object())  # repo EXISTS

    def _rootless(*a, **k):
        raise zarr.errors.GroupNotFoundError("no group found in store")

    monkeypatch.setattr(mod, "campaign_status", _rootless)
    mod.seed_global_store.fn(paths=_PATHS)
    # Reached seeding (seeded=set()) instead of propagating the read error.
    assert captured["model_version"] == checkpoint_filename()


def test_the_minimum_depth_rule_is_not_defaulted_from_the_config_constant(monkeypatch):
    """The safety property of work item 1: a seed with no explicit rule stamps NO rule.

    The rule decides what the product contains and can never be changed for the store
    afterwards, so a value inherited from whatever the config module happened to hold is not a
    decision anyone took. Pinned here because the failure is silent in the worst direction — a
    store stamped with a rule nobody chose, permanently, and every fill afterwards enforcing it.
    """
    captured = _capture_seed(monkeypatch)
    mod.seed_global_store.fn(paths=_PATHS)
    assert captured["optical_min_obs"] is None


def test_an_explicit_minimum_depth_rule_reaches_the_seeder(monkeypatch):
    captured = _capture_seed(monkeypatch)
    mod.seed_global_store.fn(paths=_PATHS, optical_min_obs=25)
    assert captured["optical_min_obs"] == 25


def _fully_seeded(monkeypatch, root_attrs: dict, *, cells_landed: int):
    """Wire the every-zone-exists path with a given root and a given amount of landed data."""
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-seed"))

    class _Root:
        def __init__(self) -> None:
            self.attrs = root_attrs

        def __getitem__(self, _name):  # the year-axis probe reads one seeded group
            return self

    monkeypatch.setattr(
        mod,
        "open_global_repo",
        lambda *a, **k: type("R", (), {"readonly_session": lambda s, branch: type("S", (), {"store": object()})()})(),
    )
    monkeypatch.setattr(
        mod,
        "campaign_status",
        lambda repo, years: type("St", (), {"zones": list(mod.ZONES), "zone_years_done": cells_landed})(),
    )
    monkeypatch.setattr(mod.zarr, "open_group", lambda *a, **k: _Root())
    monkeypatch.setattr(mod, "read_time_values", lambda grp: list(mod.CAMPAIGN_YEARS))
    monkeypatch.setattr(mod, "year_of", lambda t: t)
    monkeypatch.setattr(mod, "check_root_identity", lambda *a, **k: None)


def test_an_unstamped_but_empty_store_gets_its_identity_written(monkeypatch):
    """Seeding a store that predates the root identity must RECORD what was asked for.

    `seed_zone_groups` stamps as a side effect of creating groups, so a store whose 120
    groups already exist never reaches that line. The flow reported a clean seed having
    written neither the checkpoint nor the depth rule — and the fill gates pass on an
    ABSENT attr, so the store would then accept anything.
    """
    _fully_seeded(monkeypatch, {}, cells_landed=0)
    stamped: dict = {}
    monkeypatch.setattr(
        mod,
        "stamp_root_identity",
        lambda repo, **kw: stamped.update(kw) or "SNAP1",
    )
    out = mod.seed_global_store.fn(paths=_PATHS, optical_min_obs=15)
    assert stamped["optical_min_obs"] == 15
    assert out["seeded_now"] == 0, "nothing is created — only the identity is written"


def test_an_unstamped_store_holding_data_is_refused_rather_than_stamped(monkeypatch):
    """The stamp is write-once and every later fill reads it, so it must not be a guess.

    Nothing here can know which encoder or depth rule already-landed cells were filtered
    under, and claiming one would be a false provenance record — worse than the missing
    attr it fixes. So this raises instead of stamping.
    """
    _fully_seeded(monkeypatch, {}, cells_landed=7)
    monkeypatch.setattr(mod, "stamp_root_identity", lambda *a, **k: pytest.fail("must not stamp over existing data"))
    with pytest.raises(ValueError, match="no root identity"):
        mod.seed_global_store.fn(paths=_PATHS, optical_min_obs=15)


def test_a_fully_seeded_store_still_validates_the_requested_identity(monkeypatch):
    """The every-zone-exists path returns before seed_zone_groups, which is the only place
    the write-once root identity is compared.

    So a rerun asking for a different checkpoint — or a different minimum-depth rule —
    reported a clean seed, left the old identity standing, and the campaign then followed
    the OLD rule while the operator believed they had configured a new one. Nothing raised
    and nothing logged a difference.

    The check now runs on exactly that path. A MATCHING request must still be a no-op, so
    both directions are pinned here.
    """
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-seed"))

    class _Root:
        attrs: ClassVar[dict] = {"geoemb:model": "tessera-v1.1", "optical_min_obs": 25}

        def __getitem__(self, _name):  # the year-axis probe reads one seeded group
            return self

    class _Repo:
        def readonly_session(self, branch):
            return type("S", (), {"store": object()})()

    monkeypatch.setattr(mod, "open_global_repo", lambda *a, **k: _Repo())
    # Every zone already present → todo is empty → the early return.
    monkeypatch.setattr(
        mod, "campaign_status", lambda repo, years: type("St", (), {"zones": list(mod.ZONES), "zone_years_done": 0})()
    )
    monkeypatch.setattr(mod.zarr, "open_group", lambda *a, **k: _Root())
    # The year-axis guard runs first and must pass, so the identity check is what raises.
    monkeypatch.setattr(mod, "read_time_values", lambda grp: list(mod.CAMPAIGN_YEARS))
    monkeypatch.setattr(mod, "year_of", lambda t: t)
    seen: dict = {}

    def _check(root_attrs, *, layout, model_version, optical_min_obs):
        seen["optical_min_obs"] = optical_min_obs
        if optical_min_obs != root_attrs["optical_min_obs"]:
            raise ValueError("Refusing to reseed: the store root's published identity is write-once")

    monkeypatch.setattr(mod, "check_root_identity", _check)

    with pytest.raises(ValueError, match="write-once"):
        mod.seed_global_store.fn(paths=_PATHS, optical_min_obs=30)
    assert seen["optical_min_obs"] == 30, "the REQUESTED rule must reach the check, not the store's"
