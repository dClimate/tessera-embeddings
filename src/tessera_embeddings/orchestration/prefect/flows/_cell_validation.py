"""Hand a landed zone-year to its validation deployment, and do not wait for it.

A cell is validated by a SEPARATE flow run, dispatched the moment the cell's tag
exists. That placement is the whole design:

* **It cannot delay the campaign.** The dispatch returns as soon as the server has
  created the run (``timeout=0``), so the trailing assembly thread goes straight on to
  the next cell's assembly. Cell N validates while cell N+1 is being inferred and
  assembled, which is the same pipelining assembly itself already gets.
* **It runs on a different image than the fill.** The instrument reads pixels and
  renders figures; it needs neither GPU nor Ray, and putting it in the fill would tie
  the two release cadences together.
* **A cell that fails validation is already tagged**, and that is deliberate. The
  alternative — withholding the tag until a separate flow run agrees — makes the
  campaign's progress depend on a second run it does not wait for, and a validation
  that never ran would then look exactly like a cell that never landed. So the tag
  says *this cell landed*, and the verdict says *this cell is sound*; they are
  different questions and they get different records. A blocking finding is surfaced
  by the validation run FAILING, which is what campaign monitoring escalates on.

**Why the fill dispatches this itself rather than an automation firing on completion.**
Prefect's event broker is in-memory and drops events under load, and we have measured
it declaring healthy runs dead. An event-driven trigger would therefore skip cells
silently and at random — precisely the cells filled while the server was busiest. An
in-band dispatch is missed only when the fill itself is interrupted, and its log line
says so.

**Every failure here is swallowed.** The cell has landed and been tagged; a dispatch
that cannot reach the API must not undo that or fail the fill. The absence is not
silent, though, and this is what keeps it honest: a dispatch that did not happen leaves
no verdict beside the cell's figures, and campaign monitoring reads the published cells
against the verdicts on file. A warning in the log is the first signal; the missing
verdict is the one that survives the log's retention.
"""

from __future__ import annotations

import logging
from typing import Any

from prefect.deployments import run_deployment

from tessera_embeddings.config.paths import BucketPaths

#: Prefix of the tag stamped on a dispatched validation run, completed with the
#: DISPATCHING run's id. Deliberately NOT the tag the cancellation hooks sweep: a
#: validation run describes a cell that has already landed and been tagged, so it is
#: still worth finishing when its parent fill is cancelled, and it holds no fleet whose
#: cost would accrue while it does. The tag is for tracing a verdict back to the fill
#: that produced the cell, nothing more.
VALIDATION_TAG_PREFIX = "validates-cell-of"


def validation_run_tag(flow_run_id: object) -> str | None:
    """The trace tag for validations dispatched by this run, or ``None`` outside a run."""
    return f"{VALIDATION_TAG_PREFIX}:{flow_run_id}" if flow_run_id else None


def dispatch_cell_validation(
    deployment: str | None,
    *,
    zone: str,
    year: int,
    summary: dict[str, Any],
    parameters: dict[str, Any],
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    tag: str | None = None,
) -> str | None:
    """Create a validation run for one landed cell; return its id, or ``None``.

    Args:
        deployment: ``flow-name/deployment-name`` of the validation deployment.
            ``None`` (the default everywhere upstream) means no validation is
            configured for this deployment and nothing is dispatched — the OSS library
            names no consumer's flow.
        zone: Zone group name, canonicalized by the caller.
        year: Campaign calendar year the cell covers.
        summary: The cell's own fill summary. Read for two facts only: whether the cell
            was marked EMPTY (nothing was embedded, so there are no pixels to check) and
            whether it carries a ``tag`` (without one the cell did not land, and there is
            no committed snapshot for a validator to judge).
        parameters: Flow parameters for the validation run — the cell's coordinates and
            the store it landed in. Built by the caller so the fill's own resolved paths
            are what gets validated, rather than whatever the validation deployment was
            registered with.
        log: The fill's run logger.
        tag: Trace tag from :func:`validation_run_tag`, resolved by the caller in the
            flow's main thread. The trailing assembly thread has no Prefect context of
            its own, so the id cannot be read from the runtime here.

    Returns:
        The validation flow run's id as a string, or ``None`` when nothing was
        dispatched — for any reason, including failure. A caller must not treat
        ``None`` as an error; the record that a cell went unvalidated is the absent
        verdict, not this return value.
    """
    if deployment is None:
        return None
    # An empty cell embedded nothing, so every pixel-level check has no subject and the
    # coverage question it does raise ("is it right that this is empty?") is answered by
    # the fill's own reconciliation of written + skipped against the mask. The closing
    # sweep excludes these cells for the same reason; the two passes have to agree about
    # what the validated set IS, or monitoring reads a deliberate exclusion as a gap.
    if summary.get("empty"):
        log.info("Cell validation NOT dispatched: %s-%d is marked empty (nothing embedded to check)", zone, year)
        return None
    if not summary.get("tag"):
        log.warning(
            "Cell validation NOT dispatched: %s-%d carries no zone-year tag, so nothing landed to validate",
            zone,
            year,
        )
        return None

    try:
        # timeout=0 returns as soon as the run exists. Anything else would put a
        # three-minute read in front of the next cell's assembly.
        # run_deployment is @sync_compatible (typed as a union with its coroutine form);
        # on the trailing assembly thread it always runs synchronously.
        run: Any = run_deployment(deployment, parameters=parameters, timeout=0, tags=[tag] if tag else None)
    except Exception:
        # Swallowed on purpose — see the module docstring. The cell is tagged and sound
        # as far as the fill knows; what is lost is the CHECK, and monitoring finds that
        # by reading published cells against verdicts on file.
        log.warning(
            "Cell validation dispatch FAILED for %s-%d via %s — the cell is landed and tagged, but "
            "nothing will validate it until it is dispatched by hand",
            zone,
            year,
            deployment,
            exc_info=True,
        )
        return None
    log.info("Cell validation dispatched: %s-%d via %s as run %s", zone, year, deployment, run.id)
    return str(run.id)


def cell_validation_parameters(
    *,
    zone: str,
    year: int,
    paths: BucketPaths,
    store_name: str,
    mask_name: str,
    s3_region: str | None,
) -> dict[str, Any]:
    """The validation flow's parameters for one cell.

    ``paths`` is dumped rather than passed as a model: the value crosses the API as JSON,
    and the validation flow rebuilds its own ``BucketPaths`` from it. Passed explicitly
    even though the validation deployment is registered with the same buckets, so what
    gets validated is the store THIS fill wrote to — a hand-overridden ``paths`` on the
    fill would otherwise be checked against the registered default and quietly report on
    a different store.
    """
    return {
        "zone": zone,
        "year": year,
        "paths": paths.model_dump(),
        "store_name": store_name,
        "mask_name": mask_name,
        "s3_region": s3_region,
    }
