"""Deliberate fault injection, for failure drills a human cannot perform by hand.

Some failure modes cannot be reproduced from outside the run. One occupies a window
between two commits that is too narrow to aim a kill at; another needs a condition
that every healthy mechanism in the system removes the moment it appears. A drill
that cannot reach the failure it names produces a false pass, which is worse than
leaving the question open — so those faults are injected from inside the run
instead, under an explicit request.

**The invariant this module exists to hold: nothing here can run unless a human
asked for it in the dispatch that started the run.** One injection point sits on
the commit path of the store that is the campaign's only output, so the guarantee
has to be structural rather than careful:

* A request arrives ONLY as an explicit flow parameter (:class:`FaultInjection`),
  whose default is nothing at all. No environment variable, and no default that a
  task definition, an image, or a stored deployment schema could carry silently
  into an ordinary run.
* An injection point accepts only an :class:`ArmedFault`, and the only code that
  produces one is :meth:`FaultInjection.arm`. A request therefore cannot reach a
  firing site without passing the checks that method makes.
* :meth:`FaultInjection.arm` refuses every deployment except those named in
  :data:`DRILL_DEPLOYMENTS`. That is an ALLOWLIST rather than a denylist of
  production, which matters for the case a denylist gets wrong: a run whose
  deployment identity cannot be resolved at all is refused for exactly the reason
  production is refused, because an unresolvable identity is what a
  mis-registered production run looks like.
* :meth:`FaultInjection.arm` also refuses a fault the calling flow does not host,
  so an armed drill can never quietly do nothing and be recorded as a pass.
* Arming announces itself, and so does every firing, at error level and under one
  fixed prefix. Whoever reads these logs next must not mistake a drill's
  artifacts for an incident.

Deployment identity is READ, never asked for: :func:`deployment_stem` takes it off the
Ray control-plane SSM prefix, which a registration injects and a caller has no reason to
set.

RESIDUAL, ACCEPTED (repo owner, 2026-08-11 — do not "harden" this). That prefix is a flow
PARAMETER, so a dispatch can state it: naming a drill control plane while pointing storage
at production would pass this check. Reviewers raise it as a privilege-escalation finding
and it is not one here. Arming ALSO requires an explicit ``fault`` argument, so the path
is two deliberate acts by someone already authorised to dispatch the flow — the gate
exists to stop an accident, not an operator who is trying. The owner's ruling: drills are
not production, and this is not a threat model worth a second identity.

Closing it anyway would need a deployment fact this module does not have (the registered
Prefect deployment name, or an allowlist of drill storage) and would drag either the
Prefect runtime or an AWS identity call into ``config`` — the latter forbidden outright by
``no-botocore-outside-aws-provider``. An environment variable was tried and rejected for a
separate reason: one can be left behind in a task definition and inherited by an ordinary
run, which ``test_nothing_about_a_fault_is_read_from_the_environment`` pins.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable
from typing import Any, Literal, final

from pydantic import BaseModel, ConfigDict, model_validator

_log = logging.getLogger(__name__)

#: Fault kinds. ``die_between_commits`` ends the process between a zone-year's shard
#: commit and the commit that marks the year complete. ``withhold_work`` stops supply
#: reaching a fleet that is already up and holding nothing.
DIE_BETWEEN_COMMITS = "die_between_commits"
WITHHOLD_WORK = "withhold_work"

#: The deployments a fault may be armed on — an allowlist, and the whole of it.
#: A deployment is named here because losing a store or a fleet on it costs a drill,
#: not a campaign. Every other identity is refused, INCLUDING one that cannot be
#: resolved: "no identity" and "production" must earn the same answer, because a
#: registration that failed to inject its control-plane parameters is indistinguishable
#: from a run on an account this module has never heard of.
DRILL_DEPLOYMENTS: frozenset[str] = frozenset({"global-tessera-dev"})


#: Prefix on every line this module emits, so a drill's artifacts are greppable and
#: are never read as an incident. It is consumed by operators and monitors, so treat
#: it as an interface: match the prefix, not the words after it.
FAULT_LOG_PREFIX = "FAULT INJECTION"

#: Exit status ``die_between_commits`` leaves. Distinct from anything the runtime or
#: the container produces on its own, so the task's stopped reason names the drill
#: rather than looking like an out-of-memory kill or an ordinary failure.
DRILL_EXIT_STATUS = 93

#: Ceiling on ``withhold_work``'s hold. The fault holds a provisioned GPU fleet
#: deliberately idle, so its duration IS the drill's cost, and an unbounded hold on a
#: fleet nothing will reclaim is the failure the drill exists to study rather than a
#: way to study it. A request above this ceiling is rejected at validation.
MAX_HOLD_MINUTES = 45.0

#: How often a running hold restates itself. Frequent enough that an operator reading
#: the log knows the idleness is deliberate, sparse enough not to bury the run's own
#: lines.
_HOLD_REPORT_S = 120.0


class FaultInjectionRefusedError(RuntimeError):
    """A fault was requested where it may not be injected.

    Raised by :meth:`FaultInjection.arm`, and never caught inside this package: a
    refused request must end the run rather than downgrade to a warning, because a
    drill that continued unarmed would report whatever the run did as the drill's
    result.
    """


def deployment_stem(ssm_prefix: str | None) -> str | None:
    """The deployment's own name, read off an injected control-plane parameter.

    A run cannot be trusted to state which account it is on, so identity is derived
    from something a registration supplies and an operator has no reason to pass: the
    Ray control-plane SSM prefix, which is namespaced by the deployment's resource
    stem. Returns ``None`` when the value does not have that shape at all — including
    the flow parameter's own placeholder default, which is what an un-injected run
    carries and must not be credited as an identity.
    """
    parts = [part for part in (ssm_prefix or "").split("/") if part]
    if len(parts) != 2 or parts[1] != "ray":
        return None
    return parts[0]


@final
class FaultInjection(BaseModel):
    """A request to inject ONE deliberate fault into the run that receives it.

    A flow parameter, so it is visible in the dispatch that created the run and in
    that run's stored parameters forever after — the record of why a drill's run
    behaved unlike a real one. Extra fields are rejected rather than ignored, so a
    misspelled key fails the dispatch instead of silently arming something else.

    Each fault requires the arguments it uses and forbids the others, so a request can
    never be half-specified: a ``die_between_commits`` without a cell would have no
    single cell to fire on, and a ``withhold_work`` without a hold would have no end.

    Holding a request is not permission to use it — see :meth:`arm`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fault: Literal["die_between_commits", "withhold_work"]
    #: ``die_between_commits`` only: the one cell it may fire on. The fault is a hard
    #: process death on a shared code path, so naming the cell is what keeps a run
    #: that fills several from losing one it was not aimed at.
    zone: str | None = None
    year: int | None = None
    #: ``withhold_work`` only: how long supply is withheld, bounded by
    #: :data:`MAX_HOLD_MINUTES`.
    hold_minutes: float | None = None

    @model_validator(mode="after")
    def _each_fault_takes_its_own_arguments(self) -> FaultInjection:
        """Require what the requested fault uses; forbid what it does not."""
        if self.fault == DIE_BETWEEN_COMMITS:
            if self.zone is None or self.year is None:
                raise ValueError(f"{DIE_BETWEEN_COMMITS} needs the zone and year it may fire on")
            # CANONICAL FORM ONLY. The firing site compares the request's zone to the
            # writer's, and the fill canonicalizes before it gets there — so ``"33n"``
            # arms, announces itself, and then silently never matches ``"33N"``. A drill
            # that does nothing and is written up as a pass is the one outcome this module
            # exists to prevent, and it is worse here than anywhere else: the drill's whole
            # purpose is to prove the recovery path runs.
            #
            # REFUSED rather than normalised, and refused HERE rather than at the firing
            # site. Normalising would hide a request that does not say what its author
            # meant, and refusing at dispatch is the only point where a human is still
            # reading the error.
            if self.zone != self.zone.upper():
                raise ValueError(
                    f"zone must be the canonical UTM common name (e.g. '33N', '07S'); got {self.zone!r}. "
                    f"The writer compares against the canonicalized zone, so a lowercase request would "
                    f"arm, log, and never fire — a drill that silently passes."
                )
            if self.hold_minutes is not None:
                raise ValueError(f"{DIE_BETWEEN_COMMITS} holds nothing — drop hold_minutes")
        if self.fault == WITHHOLD_WORK:
            if self.zone is not None or self.year is not None:
                raise ValueError(f"{WITHHOLD_WORK} withholds from the whole session — drop zone/year")
            if self.hold_minutes is None:
                raise ValueError(f"{WITHHOLD_WORK} needs hold_minutes: an unbounded hold is not a drill")
            if not 0 < self.hold_minutes <= MAX_HOLD_MINUTES:
                raise ValueError(f"hold_minutes must be within (0, {MAX_HOLD_MINUTES}], got {self.hold_minutes}")
        return self

    def arm(
        self,
        *,
        ssm_prefix: str | None,
        supports: Iterable[str],
        log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    ) -> ArmedFault:
        """Clear this request for the current run, or refuse it — the only gate there is.

        The only producer of an :class:`ArmedFault`, which is the only thing an
        injection point accepts. Every check a fault must pass therefore lives here,
        and cannot be skipped by a caller that already holds a request.

        ``ssm_prefix`` is the run's own injected control-plane parameter, resolved
        here rather than by the caller so a flow cannot hand in an identity of its own
        choosing. ``supports`` is the set of faults the calling flow actually hosts:
        a fault it does not host is refused, because the alternative is an armed drill
        that does nothing and is written up as a pass.

        Announces the arming at error level under :data:`FAULT_LOG_PREFIX`. That line
        is the durable one — a fault that ends the process may lose its own firing
        line to an unflushed handler, so the record that this run is a drill is
        established before anything fires.

        Raises:
            FaultInjectionRefusedError: On any deployment outside
                :data:`DRILL_DEPLOYMENTS` — including one whose identity does not
                resolve — or for a fault the calling flow does not host.
        """
        deployment = deployment_stem(ssm_prefix)
        if deployment not in DRILL_DEPLOYMENTS:
            raise FaultInjectionRefusedError(
                f"{FAULT_LOG_PREFIX} REFUSED: fault {self.fault!r} may only be armed on "
                f"{sorted(DRILL_DEPLOYMENTS)}, and this run's deployment reads as {deployment!r} "
                f"(from ssm_prefix={ssm_prefix!r}). An unresolvable identity is refused for the "
                "same reason production is."
            )
        hosted = frozenset(supports)
        if self.fault not in hosted:
            raise FaultInjectionRefusedError(
                f"{FAULT_LOG_PREFIX} REFUSED: this flow hosts {sorted(hosted)}, not {self.fault!r} — "
                "arming it here would inject nothing and read as a pass."
            )
        log.error(
            "%s ARMED on %s: fault=%s %s. THIS RUN IS A DRILL — the failure it produces is "
            "deliberate and is not an incident.",
            FAULT_LOG_PREFIX,
            deployment,
            self.fault,
            self.model_dump(exclude_none=True, exclude={"fault"}),
        )
        return ArmedFault(self, deployment)


@final
class ArmedFault:
    """A fault THIS run is cleared to inject — the only type an injection point takes.

    Constructed solely by :meth:`FaultInjection.arm`, which is what makes the checks
    in that method unskippable: a site that holds one of these is holding proof they
    ran. An architecture test pins the sole construction, because the guarantee is the
    point of the type and a second constructor call would silently retire it.

    Firing is asked for at every site and granted only by the fault that names it, so
    a run armed for one fault is inert at the other's site.
    """

    def __init__(self, request: FaultInjection, deployment: str) -> None:
        self._request = request
        self._deployment = deployment
        self._hold_started_at: float | None = None
        self._hold_reported_at = 0.0
        self._released_work = False
        self._hold_over = False

    @property
    def request(self) -> FaultInjection:
        """The request this was armed from (for a caller that reports what it armed)."""
        return self._request

    def die_between_commits(
        self,
        zone: str,
        year: int,
        *,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    ) -> None:
        """End this process now, if armed for this fault and aimed at this cell.

        Called from between a zone-year's two commits — the shards, then the attrs
        that mark the year complete. Returns immediately for any other fault, and for
        the same fault aimed at a different cell, so a run filling several cells loses
        only the one the request named.

        The death is a hard exit rather than an exception: an exception at this point
        unwinds cleanly, runs the caller's handlers and is reported as a failure, none
        of which a real death does. Handlers are flushed first so the announcement
        survives the exit; a run logger that batches to an orchestrator may still lose
        it, which is why :meth:`FaultInjection.arm` states the drill up front.
        """
        if self._request.fault != DIE_BETWEEN_COMMITS:
            return
        if (zone, year) != (self._request.zone, self._request.year):
            return
        message = (
            "%s FIRING die_between_commits for %s year %d: the shard commit has landed and "
            "this process is exiting with status %d BEFORE the commit that marks the year "
            "complete. The year now holds data that nothing marks and nothing tags. Deliberate."
        )
        args = (FAULT_LOG_PREFIX, zone, year, DRILL_EXIT_STATUS)
        log.error(message, *args)
        _log.error(message, *args)
        logging.shutdown()  # flush every handler; nothing below here logs
        os._exit(DRILL_EXIT_STATUS)

    def withhold(
        self,
        supply: Callable[[], list[Any] | None],
        *,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    ) -> list[Any] | None:
        """What a scheduler should see in place of its work source, while armed to starve it.

        Wraps the source at the point supply enters the scheduler, and takes it as a
        CALLABLE rather than as its result. That is the load-bearing detail: a source
        that hands work over by removing it from a queue has already given it away by
        the time a value could be inspected, so a wrapper that inspected and then
        discarded would DESTROY the prepared work rather than delay it. Not calling the
        supplier is the only way to withhold without consuming, and it makes that
        property structural instead of a rule a later edit could break.

        For any other fault the source is called and its result returned, unchanged.

        While the hold runs this returns the empty list, which is the source's own
        "nothing ready yet" — so the session stays alive, keeps its actors, and holds no
        work. That is the condition the drill needs: a fleet provisioned, billing, and
        completing nothing.

        The hold begins only AFTER a first hand-over has been let through. A fleet that
        never received work has not starved, it has never started — the shape every
        detector correctly exempts as a ramp — so withholding from the outset would buy
        an idle fleet and prove nothing. Exhaustion before that first hand-over is
        announced rather than withheld, so a drill that could not fire is never mistaken
        for one that did.

        Releasing is unconditional once the hold elapses: the real source is consulted
        again and the session finishes normally. Nothing here fails a run, which is what
        separates withholding supply from breaking the thing that supplies it.
        """
        if self._request.fault != WITHHOLD_WORK:
            return supply()
        now = time.monotonic()
        if self._hold_started_at is None:
            if not self._released_work:
                handed_over = supply()
                if handed_over:
                    self._released_work = True
                elif handed_over is None:
                    log.error(
                        "%s DID NOT FIRE: the work source exhausted before any work reached the "
                        "fleet, so there was nothing to withhold. This run is NOT a starvation drill.",
                        FAULT_LOG_PREFIX,
                    )
                return handed_over
            # The supplier is deliberately NOT called from here until the hold ends:
            # asking it for work would take that work off its queue and drop it.
            self._hold_started_at = self._hold_reported_at = now
            log.error(
                "%s FIRING withhold_work: supply to the fleet is now withheld for %.1f minute(s). "
                "Actors stay alive and hold nothing, so this run will complete ZERO chunks until "
                "the hold ends. Deliberate — the fleet is not stuck.",
                FAULT_LOG_PREFIX,
                self._hold_minutes,
            )
            return []
        held_s = now - self._hold_started_at
        if held_s < self._hold_minutes * 60:
            if now - self._hold_reported_at >= _HOLD_REPORT_S:
                self._hold_reported_at = now
                log.error(
                    "%s withhold_work still holding: %.1f of %.1f minute(s) elapsed, fleet idle by design.",
                    FAULT_LOG_PREFIX,
                    held_s / 60,
                    self._hold_minutes,
                )
            return []
        if not self._hold_over:
            self._hold_over = True
            log.error(
                "%s withhold_work RELEASED after %.1f minute(s): supply is restored and the run "
                "continues normally from here.",
                FAULT_LOG_PREFIX,
                held_s / 60,
            )
        return supply()

    @property
    def _hold_minutes(self) -> float:
        """The validated hold length (only reachable for the withholding fault)."""
        assert self._request.hold_minutes is not None  # validated on the request
        return self._request.hold_minutes
