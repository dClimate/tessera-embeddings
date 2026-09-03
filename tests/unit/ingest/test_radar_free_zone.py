"""A zone with no usable radar must ingest AND fill, not fail forever.

Some land has no dual-pol VV+VH radar in principle: over ice Sentinel-1 runs Extra Wide
swath with HH/HV polarisation, which the OPERA query correctly discards. Zone 23N's live land
is Greenland, and its 2021 catalogue holds ~45,000 ascending and ~138,000 descending granules
of which the ingest can use NONE. Requiring a SAR store there failed the cell permanently.

**Permissive by default, refusable on demand.** A global product cannot reject terrain that is
radar-free as a matter of geography, so ``s1_orbit="both"`` resolves to ``"none"`` rather than
raising. A single run over terrain known to be imaged is the opposite case: there an absent
store means something upstream broke, and embedding without radar would hide it — so those
callers pass ``require_s1``, which reaches the resolver as ``allow_none=False``.

Accepting a radar-free ROI necessarily means embedding S2-only pixels, since every pixel has
zero S1 observations. The config derives that rather than asking the caller for it, because the
alternative is a run that writes nothing and still reports success.
"""

from __future__ import annotations

import ast

import pytest

import tessera_embeddings.ingest.s1_roi as s1_roi_module
from tessera_embeddings.config.inference import S1_ORBIT_NONE, InferenceConfig
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.data_loading import (
    _active_orbits,
    resolve_s1_orbit,
)
from tests._paths import SRC_ROOT

_SRC = SRC_ROOT


def test_none_activates_no_orbit() -> None:
    """The coverage gate builds its store list from this, so 'none' must yield nothing."""
    assert _active_orbits(S1_ORBIT_NONE) == ()


def test_none_is_not_an_accepted_request() -> None:
    """Nobody asks for 'none' — it is only ever a resolved outcome."""
    with pytest.raises(ValueError, match="Invalid s1_orbit"):
        _active_orbits("neither")


def test_no_radar_store_resolves_to_none_by_default(tmp_path, caplog) -> None:
    """The default must not fail: radar-free land exists and a global run has to cover it.

    Still a WARNING, because a consumer reading a finished mosaic cannot tell a radar-free ROI
    from a lost orbit — this line is the only record that it happened.
    """
    base = f"file://{tmp_path}/mosaics/23N/2021"
    with caplog.at_level("WARNING"):
        assert resolve_s1_orbit(base, "both") == S1_ORBIT_NONE
    assert "NO SAR store exists" in caplog.text
    assert base in caplog.text, "the warning must name the mosaic it applies to"


def test_radar_can_still_be_demanded(tmp_path) -> None:
    """``require_s1`` reaches the resolver as ``allow_none=False``.

    For a single run over terrain known to be imaged, an absent store means something upstream
    broke, and resolving to 'none' there would embed without radar and hide it.
    """
    base = f"file://{tmp_path}/mosaics/23N/2021"
    with pytest.raises(InsufficientCoverageError, match="no SAR stores found"):
        resolve_s1_orbit(base, "both", allow_none=False)


def test_the_demand_failure_says_where_to_confirm_which_case_it_is(tmp_path) -> None:
    """The reader cannot distinguish the two causes, so it must point at what can."""
    base = f"file://{tmp_path}/mosaics/23N/2021"
    with pytest.raises(InsufficientCoverageError) as excinfo:
        resolve_s1_orbit(base, "both", allow_none=False)
    assert "item counts" in str(excinfo.value)


@pytest.mark.parametrize("orbit", ["ascending", "descending"])
def test_a_named_orbit_is_never_downgraded(tmp_path, orbit: str) -> None:
    """An operator who named one orbit asked for something specific; allow_none must not apply."""
    base = f"file://{tmp_path}/mosaics/23N/2021"
    assert resolve_s1_orbit(base, orbit, allow_none=True) == orbit


class TestAbandoningAnUnusableOrbit:
    """Leading empty batches abandon an orbit; a mid-window gap must not.

    Zone 23N paginated 182,542 granules across a year, one 30-day batch at a time, to write
    nothing — every granule is HH/HV Extra Wide over ice and dual-pol VV+VH is required. The
    saving measured on that zone is 190s of catalogue querying per zone-year down to 59s.

    The distinction the threshold turns on is LEADING versus consecutive-anywhere. A zone imaged
    only in summer has a run of empty batches in January, so a rule that fired on any run of
    zeros would abandon it before reaching the data. Counting only while the running total is
    still zero makes that unreachable, and these tests pin exactly that.
    """

    def test_no_item_count_threshold_exists(self) -> None:
        """The rule is deliberately absent: no threshold on item counts alone is safe.

        Pinned as a test so a future attempt has to confront the seasonal case below rather
        than rediscover it in production, where the loss is silent.
        """
        assert not hasattr(s1_roi_module, "LEADING_EMPTY_BATCHES_BEFORE_SKIP")

    @staticmethod
    def _consume(items_per_batch: list[int], threshold: int) -> int:
        """Batches actually consumed under the leading-empty rule."""
        total_seen = 0
        leading_empty = 0
        for i, seen in enumerate(items_per_batch, start=1):
            total_seen += seen
            if seen == 0 and total_seen == 0:
                leading_empty += 1
                if leading_empty >= threshold:
                    return i
        return len(items_per_batch)

    def test_a_threshold_would_have_saved_the_ice_case(self) -> None:
        """What the optimisation is worth: 3 batches instead of 12 on a permanently empty orbit."""
        assert self._consume([0] * 12, 3) == 3

    def test_why_that_threshold_is_unsafe(self) -> None:
        """A summer-only zone: the same rule abandons it in March and loses every summer date.

        This is the test that stopped the optimisation shipping. Leading emptiness does not
        distinguish "this terrain is imaged in the wrong mode" from "this window has no
        acquisitions yet", so only the polarisation skip count can gate it.
        """
        summer_only = [0, 0, 0, 0, 0, 900, 900, 900, 0, 0, 0, 0]
        assert self._consume(summer_only, 3) == 3, "fires in March"
        assert sum(summer_only[3:]) == 2700, "and 2,700 usable items are never queried"

    def test_a_mid_window_gap_would_have_been_safe(self) -> None:
        """The one case the leading-only formulation did get right, kept for the eventual fix."""
        assert self._consume([500] + [0] * 11, 3) == 12


class TestTheInferenceConfigAcceptsRadarFreeLand:
    """``resolve_s1_orbit`` produces ``"none"``, so the config it feeds must accept it.

    Parts of the globe are radar-free in principle, which a global product cannot refuse. The
    value was rejected at the config boundary — so the resolver produced something no consumer
    could hold, and the failure surfaced only once a fill was already dispatched.
    """

    @staticmethod
    def _config(**kwargs: object) -> InferenceConfig:
        return InferenceConfig(time_window=parse_time_window("December 2021"), **kwargs)  # type: ignore[arg-type]

    def test_none_is_accepted_with_s2_only_embeddings_enabled(self) -> None:
        assert self._config(s1_orbit=S1_ORBIT_NONE, allow_s2_only=True).s1_orbit == S1_ORBIT_NONE

    def test_none_forces_s2_only_embeddings(self) -> None:
        """Derived, not demanded from the caller, and NOT left alone.

        With no radar store every pixel has zero S1 observations, so the default gate would skip
        every one and the fill would COMPLETE having written nothing while tagging the year done.
        An empty result that reads as success is the one outcome no later run revisits. Refusing
        instead would defeat the decision that radar-free land is acceptable.
        """
        assert self._config(s1_orbit=S1_ORBIT_NONE).allow_s2_only is True

    def test_the_forcing_is_logged_loudly(self, caplog) -> None:
        """It changes WHICH pixels are embedded and what their quality means.

        Nothing else records that a zone-year's embeddings are entirely S2-only, so a silent
        force would make an unvalidated output indistinguishable from a normal one.
        """
        with caplog.at_level("WARNING"):
            self._config(s1_orbit=S1_ORBIT_NONE)
        assert "allow_s2_only=True" in caplog.text
        assert "S2-only" in caplog.text

    def test_the_forcing_is_reproducible_from_s1_orbit_alone(self) -> None:
        """A resume must derive the same value, or the staged-chunk consistency check trips.

        The embeddings flow refuses to resume a run whose ``allow_s2_only`` differs from the one
        its staged chunks were produced under, so a forced value has to be a pure function of
        inputs the resume also has.
        """
        first = self._config(s1_orbit=S1_ORBIT_NONE)
        second = self._config(s1_orbit=S1_ORBIT_NONE)
        assert first.allow_s2_only == second.allow_s2_only is True

    def test_a_real_orbit_never_has_the_flag_forced(self) -> None:
        """The force must bind only to 'none' — a partial-radar zone keeps the caller's choice."""
        assert self._config(s1_orbit="both").allow_s2_only is False
        assert self._config(s1_orbit="ascending").allow_s2_only is False

    def test_a_bogus_orbit_is_still_refused_and_lists_none(self) -> None:
        """The valid set in the message must match the one enforced, or it misdirects."""
        with pytest.raises(ValueError, match="Invalid s1_orbit") as excinfo:
            self._config(s1_orbit="sideways")
        assert S1_ORBIT_NONE in str(excinfo.value)

    @pytest.mark.parametrize("orbit", ["ascending", "descending", "both"])
    def test_the_real_orbits_are_unaffected_by_the_flag(self, orbit: str) -> None:
        """The new rule must bind only to 'none'; a real orbit never needs the flag."""
        assert self._config(s1_orbit=orbit).s1_orbit == orbit


class TestWhoDemandsRadarAndWhoDoesNot:
    """The default differs by SCOPE, and getting it backwards fails silently either way.

    A single cell dispatched by hand should report missing radar rather than quietly produce
    optical-only embeddings — there, no radar is far more likely a broken ingest than
    genuinely radar-free terrain. A global run is the opposite: parts of the globe have no
    dual-pol coverage at all, and refusing them fails those cells on every retry forever.

    So the single-cell flows demand radar and the campaign must OVERRIDE that. These tests
    exist because the override is the fragile half: a parameter the parent must remember to
    pass is exactly the shape that regresses unnoticed.
    """

    @staticmethod
    def _default(module: str, param: str) -> object:
        tree = ast.parse((_SRC / "orchestration" / "prefect" / "flows" / module).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            args = node.args
            for name, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if name.arg == param and default is not None:
                    return ast.literal_eval(default)
        raise AssertionError(f"{module}: no keyword-only {param} with a default found")

    @pytest.mark.parametrize("module", ["fill_zone_year.py", "tessera_embeddings.py"])
    def test_a_single_cell_flow_demands_radar(self, module: str) -> None:
        assert self._default(module, "require_s1") is True

    def test_a_sweep_does_not_demand_radar(self) -> None:
        """Every cell of a sweep is chosen by breadth, so some will be radar-free."""
        assert self._default("fill_zones_sequential.py", "require_s1") is False

    def test_the_campaign_overrides_the_single_cell_default_on_every_dispatch_path(self) -> None:
        """Both dispatch modes must pass it — chained clusters AND cluster-per-cell.

        Counted rather than merely found, because covering one path and missing the other is
        the realistic regression, and it only shows up on a radar-free zone.
        """
        src = (_SRC / "orchestration" / "prefect" / "flows" / "run_global_campaign.py").read_text()
        assert src.count('"require_s1": False,') == 2, (
            "the campaign dispatches fills two ways; both must allow radar-free cells"
        )

    def test_the_local_runner_demands_radar(self) -> None:
        """Same reasoning as the single-cell flows: it fills one named ROI."""
        src = (_SRC / "orchestration" / "runners" / "plain.py").read_text()
        assert "allow_none=False" in src


def test_asking_for_no_radar_while_demanding_radar_is_refused(tmp_path):
    """``none`` is meant as a RESOLVED value — what "both" becomes over radar-free
    terrain — but it is a plain string on a public flow parameter, so it can be passed in.

    Passed in, it returned before the ``allow_none`` check was ever reached, and
    ``InferenceConfig`` then forces ``allow_s2_only`` for that orbit. So a run that
    demanded radar published optical-only embeddings and reported success — the one
    outcome ``require_s1`` exists to prevent.
    """
    with pytest.raises(InsufficientCoverageError, match="cannot both hold"):
        resolve_s1_orbit(str(tmp_path / "mosaics"), S1_ORBIT_NONE, allow_none=False)


def test_a_global_run_may_still_be_handed_the_resolved_none(tmp_path):
    """The campaign passes the resolved orbit back in with radar allowed, which is the
    whole point of the value existing — refusing that would break every radar-free cell.
    """
    assert resolve_s1_orbit(str(tmp_path / "mosaics"), S1_ORBIT_NONE, allow_none=True) == S1_ORBIT_NONE
