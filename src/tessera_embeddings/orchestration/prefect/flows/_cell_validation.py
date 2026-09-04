"""Hand a landed zone-year to its validation deployment, and do not wait for it.

A cell is validated by a SEPARATE flow run, dispatched the moment the cell's tag exists:

* **It cannot delay the campaign.** ``timeout=0`` returns as soon as the server has
  created the run, so cell N validates while cell N+1 is inferred and assembled.
* **It runs on a different image than the fill.** The instrument reads pixels and renders
  figures — no GPU, no Ray — and folding it in would couple the two release cadences.
* **A cell that fails validation is already tagged**, deliberately. The tag says *this cell
  landed*; the verdict says *this cell is sound*. Withholding the tag would make campaign
  progress depend on a run the fill does not wait for, and an unrun validation would then
  be indistinguishable from a cell that never landed. Blocking findings surface as the
  validation run FAILING, which is what campaign monitoring escalates on.

The fill dispatches in-band rather than via an automation on completion because Prefect's
event broker is in-memory and drops events under load — measured, see
context_docs/campaign/campaign-validation-and-monitoring.md. An event-driven trigger would silently
skip exactly the cells filled while the server was busiest; an in-band dispatch is missed
only when the fill itself is interrupted, and its log line says so.

**Every failure here is swallowed** — the cell is landed and tagged, and a dispatch that
cannot reach the API must not undo that. The absence still surfaces: monitoring reads
published cells against verdicts on file, and the missing verdict outlives the log warning.
"""

from __future__ import annotations

import logging
from typing import Any

from prefect.deployments import run_deployment

from tessera_embeddings.config.paths import BucketPaths

#: Prefix of the tag stamped on a dispatched validation run, completed with the DISPATCHING
#: run's id, for tracing a verdict back to the fill that produced the cell. Deliberately NOT
#: the tag the cancellation hooks sweep: the cell has already landed, so the validation is
#: worth finishing even when its parent fill is cancelled, and it holds no fleet meanwhile.
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
        deployment: ``flow-name/deployment-name`` of the validation deployment. ``None``
            (the default everywhere upstream) dispatches nothing — the OSS library names
            no consumer's flow.
        zone: Zone group name, canonicalized by the caller.
        year: Campaign calendar year the cell covers.
        summary: The cell's own fill summary. Read for two facts only: whether the cell is
            EMPTY (no pixels to check) and whether it carries a ``tag`` (without one no
            committed snapshot exists for a validator to judge).
        parameters: Flow parameters for the validation run. Built by the caller so the
            fill's own resolved paths are validated, not whatever the validation
            deployment was registered with.
        log: The fill's run logger.
        tag: Trace tag from :func:`validation_run_tag`, resolved by the caller in the
            flow's main thread — the trailing assembly thread has no Prefect context, so
            the id cannot be read from the runtime here.

    Returns:
        The validation run's id, or ``None`` when nothing was dispatched — for any reason,
        including failure. Not an error signal: the record that a cell went unvalidated is
        the absent verdict, not this return value.
    """
    if deployment is None:
        return None
    # An empty cell has no pixels to check; "is it right that this is empty?" is answered by
    # the fill's own reconciliation of written + skipped against the mask. The closing sweep
    # excludes these cells too — the two passes must agree on what the validated set IS, or
    # monitoring reads a deliberate exclusion as a gap.
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
        # timeout=0 returns as soon as the run exists; anything else puts a three-minute
        # read in front of the next cell's assembly. run_deployment is @sync_compatible
        # (typed as a union with its coroutine form) and runs synchronously on this thread.
        run: Any = run_deployment(deployment, parameters=parameters, timeout=0, tags=[tag] if tag else None)
    except Exception:
        # Swallowed on purpose — see the module docstring. What is lost is the CHECK, and
        # monitoring finds that by reading published cells against verdicts on file.
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

    ``paths`` is dumped rather than passed as a model because the value crosses the API as
    JSON; the validation flow rebuilds its own ``BucketPaths``. It is passed explicitly even
    though the deployment is registered with the same buckets, so the store THIS fill wrote
    to is what gets validated — otherwise a hand-overridden ``paths`` would be checked
    against the registered default and quietly report on a different store.
    """
    return {
        "zone": zone,
        "year": year,
        "paths": paths.model_dump(),
        "store_name": store_name,
        "mask_name": mask_name,
        "s3_region": s3_region,
    }
