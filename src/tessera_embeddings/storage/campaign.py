"""Campaign operations for the global store: tags, snapshot hygiene, progress.

The bookkeeping around the shard-writer fills (ADR-008 D7). Three concerns:

- **Tags** mark each landed ``(zone, year)`` commit and each fully-complete year.
  Tags protect their snapshots from expiry, so tagging *is* the retention policy:
  a zone-year that has been tagged survives every later ``expire_and_gc``.
- **Snapshot hygiene** (`expire_and_gc`) expires old snapshots and garbage-collects
  the now-unreferenced objects, keeping the repo's manifest/object count bounded
  over a nine-year, 120-zone campaign.
- **Progress** (`campaign_status`) reads each zone group's ``years_complete`` attr
  into a resume-oriented view: what is done, what is still pending.

Reads the ``years_complete`` attr that :func:`tessera_embeddings.storage.shard_writer.write_year_shards`
advances in the same commit as the data (D1), so status is always consistent with
what has actually landed. The zone-fill *runner* (inference -> assemble -> tag)
lives in :mod:`tessera_embeddings.orchestration.runners.zone_fill`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import icechunk
import zarr

from tessera_embeddings.storage.shard_writer import (
    CommitGate,
    commit_year_attrs,
)
from tessera_embeddings.storage.zarr_store import time_index_of
from tessera_embeddings.storage.zone_grid import CAMPAIGN_YEARS, ZONES, year_timestamp

logger = logging.getLogger(__name__)


def zone_year_tag(zone: str, year: int) -> str:
    """Tag name for a landed ``(zone, year)`` fill, e.g. ``"zone-01N-2023"``."""
    return f"zone-{zone}-{year}"


def _year_complete_tag(year: int) -> str:
    """Tag name for an all-zones-complete year, e.g. ``"year-2023-complete"``."""
    return f"year-{year}-complete"


def _ensure_tag(repo: icechunk.Repository, tag: str, sid: str, *, explicit: bool) -> str:
    """Create ``tag`` at ``sid``, idempotently — the refuse-to-move policy's one home.

    An existing tag is a no-op success when the caller took the default
    snapshot (the branch tip) — the year is already pinned, and a finalization
    sweep re-tagging after ``main`` advanced must not fail. Only an *explicit*
    ``snapshot_id`` that disagrees with the existing tag raises, rather than
    silently moving campaign history. Note icechunk tags are write-once
    *forever* — a deleted tag name can never be recreated — so a deliberate
    refill must pin its new snapshot under a fresh tag name.
    """
    if tag in repo.list_tags():
        current = repo.lookup_tag(tag)
        if explicit and current != sid:
            raise ValueError(f"tag {tag!r} already exists at snapshot {current}; refusing to move it to {sid}")
        return tag
    repo.create_tag(tag, sid)
    logger.info("tagged %s -> %s", tag, sid)
    return tag


def tag_zone_year(
    repo: icechunk.Repository,
    zone: str,
    year: int,
    *,
    snapshot_id: str | None = None,
    branch: str = "main",
) -> str:
    """Tag the snapshot for a landed ``(zone, year)`` fill; return the tag name.

    Tags at ``snapshot_id`` (default: the current ``branch`` tip, i.e. the commit
    the fill just made). Idempotent for resume: a re-run that finds the tag already
    pointing at the same snapshot is a no-op; a tag pointing *elsewhere* raises
    rather than silently moving campaign history.
    """
    return _ensure_tag(
        repo, zone_year_tag(zone, year), snapshot_id or repo.lookup_branch(branch), explicit=snapshot_id is not None
    )


def tag_year_complete(
    repo: icechunk.Repository,
    year: int,
    *,
    expected_zones: Iterable[str] | None = None,
    snapshot_id: str | None = None,
    branch: str = "main",
) -> str:
    """Tag ``year`` complete once every expected zone has landed it; return the tag.

    Verifies against live ``years_complete`` attrs (via :func:`campaign_status`)
    that all ``expected_zones`` (default: all 120) contain ``year`` before tagging;
    raises with the missing zones otherwise. Idempotent like :func:`tag_zone_year`.
    """
    expected = tuple(expected_zones) if expected_zones is not None else tuple(ZONES)
    if not expected:
        # An empty scope makes the missing-zone check vacuous, so the year would be
        # tagged complete having verified nothing — and Icechunk tags are write-once,
        # so that false marker could never be corrected under its own name. `None`
        # means "all 120"; an empty list means the caller computed a scope and got
        # nothing, which is a bug in the caller, not a completed year.
        raise ValueError(
            f"cannot tag year {year} complete: expected_zones is empty. Pass None for all "
            "zones, or a non-empty scope; an empty scope verifies nothing."
        )
    # Verify completeness at the SAME snapshot we will tag, not the moving branch
    # tip — otherwise an explicit older snapshot_id could be tagged "complete"
    # on the strength of zones that only landed later on the branch.
    status = (
        campaign_status(repo, snapshot_id=snapshot_id)
        if snapshot_id is not None
        else campaign_status(repo, branch=branch)
    )
    missing = [z for z in expected if not status.has(z, year)]
    if missing:
        raise ValueError(
            f"cannot tag year {year} complete: {len(missing)}/{len(expected)} zone(s) "
            f"have not landed it (e.g. {missing[:5]})"
        )
    return _ensure_tag(
        repo, _year_complete_tag(year), snapshot_id or repo.lookup_branch(branch), explicit=snapshot_id is not None
    )


def mark_zone_year_empty(
    repo: icechunk.Repository,
    zone: str,
    year: int,
    *,
    run_id: str | None = None,
    gate: CommitGate | None = None,
) -> str:
    """Mark a ``(zone, year)`` complete with **no data** — an all-ocean cell.

    Some of the 120 zones (and some zone-years under the partner land mask)
    contain no land at all, so there is nothing to stage or shard-write — but
    the campaign work list (:meth:`CampaignStatus.pending`) must still see them
    land. This advances ``years_complete`` (and ``runs`` provenance when
    ``run_id`` is given) in one gated commit, exactly as
    :func:`~tessera_embeddings.storage.shard_writer.write_year_shards` would,
    minus the shards. ``year`` must be on the group's pre-allocated time axis
    (D1) — an off-axis year must never enter ``years_complete``. Idempotent: a
    year already marked returns the branch tip untouched (its original
    provenance preserved) — note the returned id is then the *current* tip, so
    a tag created from it on a crash-resume may point at a later snapshot than
    the original mark commit (which remains a protected ancestor). Returns the
    snapshot id to tag.
    """
    # Axis check on a read-only view: it must fail before any session is opened, so a
    # rejected year cannot leave a writable session dangling.
    node = cast(zarr.Group, zarr.open_group(repo.readonly_session("main").store, mode="r")[zone])
    if time_index_of(node, year_timestamp(year)) is None:
        raise ValueError(
            f"Year {year} is not on {zone}'s pre-allocated time axis — refusing to mark an "
            "off-axis year complete (ADR-008 D1: the axis is fixed at seeding)."
        )
    # Delegated so there is exactly ONE writer of years_complete/runs, with one
    # retry-on-conflict policy. That is what makes a no-land mark safe to run
    # concurrently with another year of the same zone.
    return commit_year_attrs(repo, zone, year, run_id=run_id, empty=True, gate=gate, skip_if_marked=True)


@dataclass(frozen=True)
class ExpireGCResult:
    """Outcome of one :func:`expire_and_gc` pass."""

    expired_snapshots: frozenset[str]  # snapshot ids removed from history (empty on dry-run)
    gc: icechunk.GCSummary  # objects deleted (or, on dry-run, that would be)


def _validate_cutoff(older_than: datetime) -> None:
    """Reject a cutoff that icechunk (or the campaign guard) would refuse."""
    if older_than.tzinfo is None:
        raise ValueError("older_than must be timezone-aware; icechunk rejects naive datetimes")
    now = datetime.now(UTC)
    if older_than >= now:
        raise ValueError(f"older_than {older_than!r} must be strictly in the past (now={now!r})")


def expire_and_gc(
    repo: icechunk.Repository,
    *,
    older_than: datetime,
    dry_run: bool = False,
) -> ExpireGCResult:
    """Expire snapshots older than ``older_than``, then GC unreferenced objects (D7).

    Expiry keeps tagged snapshots and branch tips (``delete_expired_tags`` /
    ``delete_expired_branches`` stay False), so every tagged ``(zone, year)`` fill
    survives - tagging is the retention policy. GC then deletes objects that are
    both unreferenced *and* older than the cutoff.

    Campaign guard (caller's contract, not enforceable from the repo): ``older_than``
    must predate the oldest in-flight session - never run this during active fills,
    or a concurrent writer's just-written objects could be collected. The cutoff is
    validated to be timezone-aware and strictly in the past.

    ``dry_run=True`` skips expiry entirely (``expire_snapshots`` mutates refs and has
    no dry-run) and reports only ``garbage_collect(dry_run=True)`` against the
    *current* ref graph - a **lower bound**, since it cannot see the objects that
    expiry would orphan. Use it for a rough sizing, not an exact preview.
    """
    _validate_cutoff(older_than)
    if dry_run:
        gc = repo.garbage_collect(older_than, dry_run=True)
        logger.info("expire_and_gc dry-run: %s bytes reclaimable (expiry not simulated)", gc.bytes_deleted)
        return ExpireGCResult(expired_snapshots=frozenset(), gc=gc)
    expired = repo.expire_snapshots(older_than)
    gc = repo.garbage_collect(older_than, dry_run=False)
    logger.info("expire_and_gc: expired %d snapshot(s), freed %s bytes", len(expired), gc.bytes_deleted)
    return ExpireGCResult(expired_snapshots=frozenset(expired), gc=gc)


@dataclass(frozen=True)
class CampaignStatus:
    """A resume-oriented view of the campaign's zone x year fill progress."""

    years: tuple[int, ...]  # the campaign year axis (e.g. 2017..2025)
    zones: dict[str, tuple[int, ...]]  # seeded group name -> years landed (sorted)

    @property
    def zones_seeded(self) -> int:
        """Number of zone groups that exist in the repo."""
        return len(self.zones)

    @property
    def zone_years_done(self) -> int:
        """Total ``(zone, year)`` cells landed across all seeded zones."""
        return sum(len(v) for v in self.zones.values())

    def has(self, zone: str, year: int) -> bool:
        """Whether ``zone`` has landed ``year``."""
        return year in self.zones.get(zone, ())

    def pending(
        self,
        *,
        expected_zones: Iterable[str] | None = None,
        years: Iterable[int] | None = None,
    ) -> list[tuple[str, int]]:
        """The ``(zone, year)`` cells still to fill, for the orchestrator's work list.

        Defaults to all 120 zones x the campaign years; a zone not yet seeded counts
        every year as pending. Ordered year-major (all zones of one year before
        the next year) so a scheduler draining the list concurrently never
        queues two years of the SAME zone back to back — same-zone fills must
        be serial (their attr commits conflict; see
        :func:`~tessera_embeddings.storage.shard_writer.write_year_shards`).
        """
        zone_names = tuple(expected_zones) if expected_zones is not None else tuple(ZONES)
        yrs = tuple(years) if years is not None else self.years
        return [(z, y) for y in yrs for z in zone_names if not self.has(z, y)]

    def years_fully_complete(self, *, expected_zones: Iterable[str] | None = None) -> list[int]:
        """Years landed in *every* expected zone (default: all 120)."""
        zone_names = tuple(expected_zones) if expected_zones is not None else tuple(ZONES)
        return [y for y in self.years if all(self.has(z, y) for z in zone_names)]


def campaign_status(
    repo: icechunk.Repository,
    *,
    years: tuple[int, ...] = CAMPAIGN_YEARS,
    branch: str = "main",
    snapshot_id: str | None = None,
) -> CampaignStatus:
    """Summarize zone x year fill from the live ``years_complete`` group attrs.

    Reads read-only from ``branch``'s tip, or from ``snapshot_id`` when given
    (so a caller can assess completeness at the exact snapshot it is about to
    tag, not the moving branch tip). Only groups that actually exist are
    reported (a partially seeded campaign is fine); each group's landed years come
    from the ``years_complete`` attr the shard writer advances atomically with the
    data, so the view never claims a year the data doesn't back.
    """
    session = repo.readonly_session(snapshot_id=snapshot_id) if snapshot_id else repo.readonly_session(branch)
    root = zarr.open_group(session.store, mode="r")
    zones: dict[str, tuple[int, ...]] = {}
    for name in sorted(root.group_keys()):
        node = cast(zarr.Group, root[name])
        raw = node.attrs.get("years_complete", [])
        landed = tuple(sorted(int(y) for y in raw)) if isinstance(raw, list) else ()
        zones[name] = landed
    return CampaignStatus(years=tuple(years), zones=zones)


def campaign_work_list(
    status: CampaignStatus,
    existing_tags: set[str],
    *,
    expected_zones: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
) -> list[tuple[str, int]]:
    """The ``(zone, year)`` cells a campaign run should dispatch, tag-aware.

    Like :meth:`CampaignStatus.pending` (same zones filter + year-major order) but a
    cell counts DONE only when it is BOTH in ``years_complete`` AND has its zone-year
    tag. Two consequences the plain ``pending`` can't express:

    - **Skip finished zones on a default re-run.** With ``expected_zones=None`` (all
      120), a year that partially completed and is re-driven only dispatches the
      zones still missing — a landed-and-tagged zone is filtered out.
    - **Recover a complete-but-untagged cell.** A crash between the fill commit and
      the tag leaves a cell in ``years_complete`` but untagged; it is INCLUDED so the
      fill runner's idempotent retag path runs (filtering on ``years_complete`` alone
      would skip it forever, and the year's all-120 milestone tag would never land).

    ``expected_zones`` is the zone filter for a subset / repair run (default: all 120).
    """
    # Dedupe (order-preserving): duplicate zones/years from a caller would emit the
    # same (zone, year) twice, and the driver would dispatch two concurrent fills of
    # one cell — their years_complete/runs attr commits conflict (RebaseFailedError).
    zone_names = tuple(dict.fromkeys(expected_zones)) if expected_zones is not None else tuple(ZONES)
    yrs = tuple(dict.fromkeys(years)) if years is not None else status.years
    return [(z, y) for y in yrs for z in zone_names if not status.has(z, y) or zone_year_tag(z, y) not in existing_tags]
