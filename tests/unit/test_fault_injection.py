"""Fault injection for supervised drills: the gate, and the guarantee it makes.

The point of these tests is not that the faults work — that is pinned at each
injection site, in the tests of the code that hosts it. The point is that they
cannot work anywhere else: not on an account outside the drill allowlist, not on a
run whose deployment identity does not resolve, not through a flow that would inject
nothing, and not at all unless a request was passed in.

The structural tests at the bottom carry the load. Any per-site check can only prove
that ONE site is guarded; those prove that a site's only key is produced in one
place, and that the hard exit exists nowhere else in the package.
"""

from __future__ import annotations

import ast
import logging
import pathlib

import pytest
from pydantic import ValidationError

from tessera_embeddings.config.fault_injection import (
    DIE_BETWEEN_COMMITS,
    DRILL_DEPLOYMENTS,
    FAULT_LOG_PREFIX,
    MAX_HOLD_MINUTES,
    WITHHOLD_WORK,
    ArmedFault,
    FaultInjection,
    FaultInjectionRefusedError,
    deployment_stem,
)

LOG = logging.getLogger("test-fault-injection")

_DRILL_PREFIX = "/global-tessera-dev/ray/"
_DIE = FaultInjection(fault=DIE_BETWEEN_COMMITS, zone="01N", year=2025)
_WITHHOLD = FaultInjection(fault=WITHHOLD_WORK, hold_minutes=30)

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "tessera_embeddings"
FAULT_MODULE = SRC / "config" / "fault_injection.py"


class TestTheRequestItself:
    """A request that is half-specified must fail the dispatch, not the drill."""

    def test_each_fault_requires_the_arguments_it_uses(self):
        # A fault missing what it needs would arm and then either fire on nothing or
        # never end — both of which read as "the drill ran" in the run's own logs.
        with pytest.raises(ValidationError, match="zone and year"):
            FaultInjection(fault=DIE_BETWEEN_COMMITS)
        with pytest.raises(ValidationError, match="hold_minutes"):
            FaultInjection(fault=WITHHOLD_WORK)

    def test_each_fault_forbids_the_other_s_arguments(self):
        with pytest.raises(ValidationError, match="drop hold_minutes"):
            FaultInjection(fault=DIE_BETWEEN_COMMITS, zone="01N", year=2025, hold_minutes=5)
        with pytest.raises(ValidationError, match="drop zone/year"):
            FaultInjection(fault=WITHHOLD_WORK, hold_minutes=5, zone="01N", year=2025)

    @pytest.mark.parametrize("minutes", [0, -1, MAX_HOLD_MINUTES + 0.5])
    def test_the_hold_is_bounded_at_both_ends(self, minutes):
        # The hold's length IS the drill's bill: it holds a provisioned GPU fleet idle
        # on purpose, and nothing in the system reclaims one.
        with pytest.raises(ValidationError, match="hold_minutes must be within"):
            FaultInjection(fault=WITHHOLD_WORK, hold_minutes=minutes)

    def test_an_unknown_fault_and_a_misspelled_field_are_both_rejected(self):
        with pytest.raises(ValidationError):
            FaultInjection(fault="corrupt_the_store")
        with pytest.raises(ValidationError):
            FaultInjection(fault=WITHHOLD_WORK, hold_minutes=5, hold_minute=5)


class TestArming:
    """The one gate: which deployments, and which flows, may inject anything."""

    @pytest.mark.parametrize(
        ("ssm_prefix", "reads_as"),
        [
            ("/global-tessera-prod/ray/", "global-tessera-prod"),  # the account that must never
            ("/yield-embeddings/ray/", "yield-embeddings"),  # a second production account
            ("/tessera/ray/", "tessera"),  # the flow parameter's un-injected default
            ("", None),
            (None, None),
            ("/global-tessera-dev/", None),  # right account, not the expected shape
            ("/some/deep/path/ray/", None),
        ],
    )
    def test_arming_is_refused_anywhere_but_a_drill_deployment(self, ssm_prefix, reads_as):
        # An ALLOWLIST, so the two cases a denylist confuses get the same answer: an
        # identified production account, and a run whose identity does not resolve at
        # all — which is what a registration that failed to inject looks like.
        with pytest.raises(FaultInjectionRefusedError) as exc:
            _DIE.arm(ssm_prefix=ssm_prefix, supports=(DIE_BETWEEN_COMMITS,), log=LOG)
        assert repr(reads_as) in str(exc.value), "the refusal must name what the run read as"
        assert sorted(DRILL_DEPLOYMENTS)[0] in str(exc.value)

    def test_arming_succeeds_on_a_drill_deployment(self):
        # The paired positive: without it, a refusal test passes on code that refuses
        # everything, which is not the property being claimed.
        armed = _DIE.arm(ssm_prefix=_DRILL_PREFIX, supports=(DIE_BETWEEN_COMMITS,), log=LOG)
        assert isinstance(armed, ArmedFault)
        assert armed.request is _DIE

    def test_arming_is_refused_for_a_fault_the_flow_does_not_host(self):
        # Otherwise the drill is armed, injects nothing, and the run's ordinary success
        # is written up as the fault's result.
        with pytest.raises(FaultInjectionRefusedError, match="hosts"):
            _WITHHOLD.arm(ssm_prefix=_DRILL_PREFIX, supports=(DIE_BETWEEN_COMMITS,), log=LOG)

    def test_arming_announces_the_run_as_a_drill(self, caplog):
        # The durable announcement. A fault that ends the process may lose its own
        # firing line, so this is what tells whoever reads the logs next that the
        # damage was deliberate.
        with caplog.at_level(logging.ERROR):
            _DIE.arm(ssm_prefix=_DRILL_PREFIX, supports=(DIE_BETWEEN_COMMITS,), log=logging.getLogger("armed"))
        armed_lines = [r.getMessage() for r in caplog.records if FAULT_LOG_PREFIX in r.getMessage()]
        assert armed_lines, f"arming must log under {FAULT_LOG_PREFIX!r}"
        assert "DRILL" in armed_lines[0]
        assert DIE_BETWEEN_COMMITS in armed_lines[0]
        assert "global-tessera-dev" in armed_lines[0]

    @pytest.mark.parametrize(
        ("prefix", "expected"),
        [
            ("/global-tessera-dev/ray/", "global-tessera-dev"),
            ("global-tessera-dev/ray", "global-tessera-dev"),
            ("/tessera/ray/", "tessera"),
            ("/global-tessera-dev/ray/extra/", None),
            ("/global-tessera-dev/rays/", None),
            (None, None),
        ],
    )
    def test_the_deployment_is_read_off_the_injected_control_plane_prefix(self, prefix, expected):
        assert deployment_stem(prefix) == expected


class _Source:
    """A work source that HANDS OVER: each queued zone is yielded exactly once.

    Modelled on the real one, whose hand-over pops the zone off a deque. That is why
    the fault takes a callable — anything that consulted the source and then discarded
    the result would delete prepared work instead of delaying it, and only a source
    that can be drained can catch that.
    """

    def __init__(self, *zones: str, exhausted: bool = True) -> None:
        self.queued = list(zones)
        self.exhausted = exhausted
        self.calls = 0

    def __call__(self) -> list[str] | None:
        self.calls += 1
        if self.queued:
            return [self.queued.pop(0)]
        return None if self.exhausted else []


class TestWithholdingSupply:
    """The starvation fault's state machine, driven directly on a fake clock."""

    @staticmethod
    def _armed(monkeypatch, hold_minutes: float = 30.0) -> tuple[ArmedFault, list[float]]:
        """An armed withholding fault plus a one-element list holding the fake clock."""
        now = [1_000.0]
        monkeypatch.setattr("tessera_embeddings.config.fault_injection.time.monotonic", lambda: now[0])
        request = FaultInjection(fault=WITHHOLD_WORK, hold_minutes=hold_minutes)
        return request.arm(ssm_prefix=_DRILL_PREFIX, supports=(WITHHOLD_WORK,), log=LOG), now

    def test_a_fault_armed_for_the_other_kind_withholds_nothing(self):
        # One armed fault, one behaviour. A run armed to die between commits must be
        # completely inert at the supply seam, or a single drill perturbs two things.
        armed = _DIE.arm(ssm_prefix=_DRILL_PREFIX, supports=(DIE_BETWEEN_COMMITS,), log=LOG)
        source = _Source("zone-a", "zone-b")
        assert armed.withhold(source, log=LOG) == ["zone-a"]
        assert armed.withhold(source, log=LOG) == ["zone-b"]
        assert armed.withhold(source, log=LOG) is None

    def test_the_first_hand_over_is_let_through(self, monkeypatch):
        # A fleet that never received work has not starved, it has never started —
        # the shape every detector exempts as a ramp. Withholding from the outset
        # would buy an idle fleet and prove nothing.
        armed, _ = self._armed(monkeypatch)
        not_ready_yet = _Source(exhausted=False)
        assert armed.withhold(not_ready_yet, log=LOG) == [], "nothing ready yet passes through untouched"
        assert armed.withhold(_Source("zone-a"), log=LOG) == ["zone-a"]

    def test_a_withheld_poll_never_takes_the_zone_it_refuses_to_hand_over(self, monkeypatch):
        # The defect this shape exists to prevent: the source's hand-over is
        # destructive, so a wrapper that asked for a zone and returned [] would lose
        # that zone outright — a drill that deleted a cell's work instead of delaying it.
        armed, now = self._armed(monkeypatch, hold_minutes=30)
        source = _Source("zone-a", "zone-b")
        assert armed.withhold(source, log=LOG) == ["zone-a"]
        calls_after_hand_over = source.calls
        for _ in range(5):
            assert armed.withhold(source, log=LOG) == []
        assert source.calls == calls_after_hand_over, "a held poll must not consult the source at all"
        assert source.queued == ["zone-b"], "and the zone it is holding back must still be there"
        now[0] += 31 * 60
        assert armed.withhold(source, log=LOG) == ["zone-b"], "released intact"

    def test_supply_is_withheld_after_the_first_hand_over_and_restored_on_time(self, monkeypatch):
        armed, now = self._armed(monkeypatch, hold_minutes=30)
        source = _Source("zone-a")
        armed.withhold(source, log=LOG)

        # Exhaustion becomes "nothing ready yet", which is what keeps the session
        # alive with its actors held and no work in them.
        assert armed.withhold(source, log=LOG) == []
        now[0] += 29 * 60
        assert armed.withhold(source, log=LOG) == []
        now[0] += 2 * 60
        assert armed.withhold(source, log=LOG) is None, "the hold must end by itself"

    def test_the_hold_announces_its_start_its_progress_and_its_end(self, monkeypatch, caplog):
        armed, now = self._armed(monkeypatch, hold_minutes=10)
        source = _Source("zone-a")
        log = logging.getLogger("withholding")
        with caplog.at_level(logging.ERROR):
            armed.withhold(source, log=log)
            armed.withhold(source, log=log)
            now[0] += 5 * 60
            armed.withhold(source, log=log)
            now[0] += 6 * 60
            armed.withhold(source, log=log)
        said = [r.getMessage() for r in caplog.records if FAULT_LOG_PREFIX in r.getMessage()]
        assert any("FIRING withhold_work" in line for line in said)
        assert any("still holding" in line for line in said)
        assert any("RELEASED" in line for line in said)

    def test_a_source_that_exhausts_before_any_hand_over_says_the_fault_did_not_fire(self, monkeypatch, caplog):
        # The silent-no-op case, made loud. A run with nothing to withhold from must
        # not be written up as a starvation drill.
        armed, _ = self._armed(monkeypatch)
        log = logging.getLogger("never-fired")
        with caplog.at_level(logging.ERROR):
            assert armed.withhold(_Source(), log=log) is None
        said = [r.getMessage() for r in caplog.records if FAULT_LOG_PREFIX in r.getMessage()]
        assert any("DID NOT FIRE" in line for line in said)


def _package_files() -> list[pathlib.Path]:
    return sorted(SRC.rglob("*.py"))


def _call_sites(name: str) -> list[tuple[pathlib.Path, int]]:
    """Every call of ``name`` (bare or attribute) across the package, with its line."""
    found: list[tuple[pathlib.Path, int]] = []
    for path in _package_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            called = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            if called == name:
                found.append((path, node.lineno))
    return found


class TestTheGuaranteeIsStructural:
    """What no per-site test can establish: that there is no second way in."""

    def test_only_arm_constructs_an_armed_fault(self):
        # This is the whole guarantee. An injection point accepts nothing but an
        # ArmedFault, so if `arm` is its only source then every fault that fires has
        # passed the deployment allowlist and the hosted-fault check — and no site can
        # be reviewed for that separately. A second constructor anywhere retires it.
        sites = _call_sites("ArmedFault")
        assert [p for p, _ in sites] == [FAULT_MODULE], (
            f"ArmedFault must be constructed only in {FAULT_MODULE.name}, found: {[(str(p), n) for p, n in sites]}"
        )

    def test_the_package_hard_exits_in_exactly_one_place(self):
        # A hard exit skips every handler by design, which is what makes it a faithful
        # death and an unacceptable thing to have anywhere else. One site, behind the
        # gate above.
        sites = _call_sites("_exit")
        assert [p for p, _ in sites] == [FAULT_MODULE], (
            f"os._exit must appear only in {FAULT_MODULE.name}, found: {[(str(p), n) for p, n in sites]}"
        )

    def test_nothing_about_a_fault_is_read_from_the_environment(self):
        # An environment variable can be left behind in a task definition or an image
        # and then inherited by an ordinary run. A flow parameter cannot: it is stated
        # per dispatch and stored on the run.
        source = FAULT_MODULE.read_text()
        for forbidden in ("os.environ", "getenv"):
            assert forbidden not in source, f"{forbidden} in the fault module would give a fault a second door"

    def test_no_flow_or_runner_defaults_a_fault_on(self):
        # "Off unless asked for" as a property of every signature that carries it,
        # rather than of the four that were checked by hand.
        import inspect

        from tessera_embeddings.inference.assembly import ZarrWriter
        from tessera_embeddings.orchestration.prefect.flows.fill_zone_year import fill_zone_year_flow
        from tessera_embeddings.orchestration.prefect.flows.fill_zones_sequential import fill_zones_sequential_flow
        from tessera_embeddings.orchestration.runners.sequential_fill import fill_zones_sequential
        from tessera_embeddings.orchestration.runners.zone_fill import assemble_zone_year, fill_zone_year
        from tessera_embeddings.storage.shard_writer import write_year_shards

        carriers = (
            (fill_zone_year_flow, "fault_injection"),
            (fill_zones_sequential_flow, "fault_injection"),
            (fill_zone_year, "fault"),
            (assemble_zone_year, "fault"),
            (fill_zones_sequential, "fault"),
            (ZarrWriter.assemble_global, "fault"),
            (write_year_shards, "fault"),
        )
        for fn, param in carriers:
            sig = inspect.signature(getattr(fn, "fn", fn))
            assert param in sig.parameters, f"{fn.__name__} should carry {param}"
            assert sig.parameters[param].default is None, f"{fn.__name__}.{param} must default to nothing"
