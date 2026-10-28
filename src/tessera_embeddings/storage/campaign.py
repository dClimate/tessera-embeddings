"""Campaign operations for the global store: tags, snapshot hygiene, progress.

The bookkeeping around the shard-writer fills (ADR-008 D7). Three concerns:

- **Tags** mark each landed ``(zone, year)`` commit and each fully-complete year. Tags protect
  their snapshots from expiry, so tagging *is* the retention policy: a tagged zone-year survives
  every later ``expire_and_gc``.
- **Snapshot hygiene** (`expire_and_gc`) expires old snapshots and garbage-collects the
  now-unreferenced objects, keeping the repo's manifest/object count bounded over a nine-year,
  multi-zone campaign.
- **Progress** (`campaign_status`) reads each zone group's ``years_complete`` attr — and a
  column's ``halves_complete``, since the campaign fills a ``utm_zones`` column as two halves
  (ADR 036) — into a resume-oriented view: what is done, what is still pending.

That attr is advanced by :func:`tessera_embeddings.storage.shard_writer.write_year_shards` in the
same commit as the data (D1), so status is always consistent with what has actually landed. The
zone-fill *runner* (inference -> assemble -> tag) lives in
:mod:`tessera_embeddings.orchestration.runners.zone_fill`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

import icechunk
import zarr

from tessera_embeddings.storage.shard_writer import HALVES, commit_year_attrs, read_halves_complete
from tessera_embeddings.storage.time_axis import CAMPAIGN_YEARS, time_index_of, year_timestamp
from tessera_embeddings.storage.zone_grid import fill_zones, group_and_half, zone_layout_of, zones

logger = logging.getLogger(__name__)


def zone_year_tag(zone: str, year: int) -> str:
    """Tag name for a landed ``(zone, year)`` fill, e.g. ``"zone-01N-2023"``."""
    return f"zone-{zone}-{year}"


def _year_complete_tag(year: int) -> str:
    """Tag name for an all-zones-complete year, e.g. ``"year-2023-complete"``."""
    return f"year-{year}-complete"


def _ensure_tag(repo: icechunk.Repository, tag: str, sid: str, *, explicit: bool) -> str:
    """Create ``tag`` at ``sid``, idempotently — the refuse-to-move policy's one home.

    An existing tag is a no-op success when the caller took the default snapshot (the branch tip):
    the year is already pinned, and a finalization sweep re-tagging after ``main`` advanced must
    not fail. Only an *explicit* ``snapshot_id`` that disagrees with the existing tag raises,
    rather than silently moving campaign history. icechunk tags are write-once *forever* — a
    deleted tag name can never be recreated — so a deliberate refill must pin its new snapshot
    under a fresh tag name.
    """
    if tag in repo.list_tags():
        current = repo.lookup_tag(tag)
        if explicit and current != sid:
            raise ValueError(f"tag {tag!r} already exists at snapshot {current}; refusing to move it to {sid}")
        return tag
    repo.create_tag(tag, sid)
    logger.info("tagged %s -> %s", tag, sid)
    return tag


def pin_tag(repo: icechunk.Repository, tag: str) -> str:
    """Create ``tag`` at the ``main`` tip unless it already exists; return the tag.

    Idempotent, so every cell of a targeted update can call it and only the first pins. A tag that
    exists stays where it is, which is the point of a base tag: it holds the store as it was before
    the update's first write.

    Safe under concurrent callers: one that loses the race to create the tag finds it existing and
    returns it, as a caller that came second would have.
    """
    try:
        return _ensure_tag(repo, tag, repo.lookup_branch("main"), explicit=False)
    except icechunk.IcechunkError:
        if tag in repo.list_tags():
            return tag
        raise


def tag_zone_year(
    repo: icechunk.Repository,
    zone: str,
    year: int,
    *,
    snapshot_id: str | None = None,
    branch: str = "main",
) -> str:
    """Tag the snapshot for a landed ``(zone, year)`` fill; return the tag name.

    Tags at ``snapshot_id`` (default: the current ``branch`` tip, i.e. the commit the fill just
    made). Idempotent for resume: a re-run that finds the tag already pointing at the same
    snapshot is a no-op; a tag pointing *elsewhere* raises rather than silently moving history.
    """
    return _ensure_tag(
        repo, zone_year_tag(zone, year), snapshot_id or repo.lookup_branch(branch), explicit=snapshot_id is not None
    )


def missing_zones_for_year(
    repo: icechunk.Repository,
    year: int,
    *,
    expected_zones: Iterable[str] | None = None,
    snapshot_id: str | None = None,
    branch: str = "main",
) -> tuple[str, ...]:
    """Zones in scope that have not landed ``year``, in scope order — no side effects.

    The verification half of :func:`tag_year_complete`, split out so a caller can ask about an
    arbitrary scope (a road test, one hemisphere, a re-drive's remainder) without being able to
    mint a campaign-wide completion tag from it. ``None`` means the full campaign
    (:meth:`CampaignStatus.campaign_zones`).

    Reads ``years_complete`` at the same snapshot a tag would pin — ``snapshot_id`` when given,
    else ``branch``'s tip — so the answer describes that snapshot rather than a branch that may
    have advanced since.
    """
    if expected_zones is not None and not tuple(expected_zones):
        # An empty scope makes the check vacuous: every zone in scope has landed, having verified
        # nothing. `None` means "the whole campaign"; an empty list means the caller computed a
        # scope and got nothing, which is a bug in the caller, not a completed year.
        raise ValueError(
            f"cannot assess year {year}: expected_zones is empty. Pass None for all "
            "zones, or a non-empty scope; an empty scope verifies nothing."
        )
    status = (
        campaign_status(repo, snapshot_id=snapshot_id)
        if snapshot_id is not None
        else campaign_status(repo, branch=branch)
    )
    expected = tuple(expected_zones) if expected_zones is not None else status.campaign_zones()
    return tuple(z for z in expected if not status.has(z, year))


def tag_year_complete(
    repo: icechunk.Repository,
    year: int,
    *,
    snapshot_id: str | None = None,
    branch: str = "main",
) -> str:
    """Tag ``year`` complete once **every** campaign zone has landed it; return the tag.

    The scope is always the full campaign — every zone of the store's layout — and is
    deliberately *not* a parameter.
    ``year-<year>-complete`` is a campaign-wide claim; icechunk tags are write-once forever and
    :func:`_ensure_tag` treats an existing tag as an idempotent success — so a tag minted after
    only a subset landed could never be corrected under its own name, and would permanently mark
    an incomplete snapshot as a finished year. To ask whether a narrower scope has landed, call
    :func:`missing_zones_for_year`, which answers without tagging.

    Verifies against live ``years_complete`` attrs at the snapshot being tagged; raises with the
    missing zones otherwise. Idempotent like :func:`tag_zone_year`.
    """
    missing = missing_zones_for_year(repo, year, snapshot_id=snapshot_id, branch=branch)
    if missing:
        raise ValueError(
            f"cannot tag year {year} complete: {len(missing)} zone(s) have not landed it (e.g. {missing[:5]})"
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
) -> str:
    """Mark a ``(zone, year)`` complete with **no data** — an all-ocean cell.

    ``zone`` is a fill zone: a group, or a column half (ADR 036), which marks only its half.

    Some zones (and some zone-years under the partner land mask) contain no land at
    all, so there is nothing to stage or shard-write — but the campaign work list
    (:meth:`CampaignStatus.pending`) must still see them land. This advances ``years_complete``
    (and ``runs`` provenance when ``run_id`` is given) in one commit, exactly as
    :func:`~tessera_embeddings.storage.shard_writer.write_year_shards` would, minus the shards.
    ``year`` must be on the group's pre-allocated time axis (D1) — an off-axis year must never
    enter ``years_complete``. Idempotent: a year already marked returns the branch tip untouched,
    its original provenance preserved — the returned id is then the *current* tip, so a tag
    created from it on a crash-resume may point at a later snapshot than the original mark commit
    (which remains a protected ancestor). Returns the snapshot id to tag.
    """
    # Axis check on a read-only view: it must fail before any session is opened, so a rejected
    # year cannot leave a writable session dangling.
    group, half = group_and_half(zone)
    node = cast(zarr.Group, zarr.open_group(repo.readonly_session("main").store, mode="r")[group])
    if time_index_of(node, year_timestamp(year)) is None:
        raise ValueError(
            f"Year {year} is not on {zone}'s pre-allocated time axis — refusing to mark an "
            "off-axis year complete (ADR-008 D1: the axis is fixed at seeding)."
        )
    # Delegated so there is exactly ONE writer of years_complete/runs, with one retry-on-conflict
    # policy — what makes a no-land mark safe to run concurrently with another year of the zone.
    return commit_year_attrs(repo, group, year, run_id=run_id, empty=True, skip_if_marked=True, half=half)


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
    ``delete_expired_branches`` stay False), so every tagged ``(zone, year)`` fill survives -
    tagging is the retention policy. GC then deletes objects that are both unreferenced *and*
    older than the cutoff.

    Campaign guard (caller's contract, not enforceable from the repo): ``older_than`` must predate
    the oldest in-flight session - never run this during active fills, or a concurrent writer's
    just-written objects could be collected. The cutoff is validated to be timezone-aware and
    strictly in the past.

    ``dry_run=True`` skips expiry entirely (``expire_snapshots`` mutates refs and has no dry-run)
    and reports only ``garbage_collect(dry_run=True)`` against the *current* ref graph - a **lower
    bound**, since it cannot see the objects expiry would orphan. A rough sizing, not a preview.
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
    halves: dict[str, tuple[int, ...]] = field(default_factory=dict)  # column half -> years landed

    @property
    def zones_seeded(self) -> int:
        """Number of zone groups that exist in the repo."""
        return len(self.zones)

    @property
    def zone_years_done(self) -> int:
        """Total ``(zone, year)`` cells landed across all seeded zones."""
        return sum(len(v) for v in self.zones.values())

    def campaign_zones(self) -> tuple[str, ...]:
        """The full campaign: every fill zone of the layout this store's groups use (ADR 034, 036)."""
        layout = zone_layout_of(self.zones)
        if layout is None:
            raise ValueError("the store has no zone groups, so it has no layout yet — seed it first.")
        return tuple(fill_zones(layout))

    def has(self, zone: str, year: int) -> bool:
        """Whether ``zone`` — a group, or a column half — has landed ``year``."""
        return year in self.zones.get(zone, ()) or year in self.halves.get(zone, ())

    def pending(
        self,
        *,
        expected_zones: Iterable[str] | None = None,
        years: Iterable[int] | None = None,
    ) -> list[tuple[str, int]]:
        """The ``(zone, year)`` cells still to fill, for the orchestrator's work list.

        Defaults to :meth:`campaign_zones` x the campaign years; a zone not yet seeded counts
        every year as pending. Ordered year-major (all zones of one year before the next), so a
        scheduler draining the list works across the world rather than down one zone's years.
        """
        zone_names = tuple(expected_zones) if expected_zones is not None else self.campaign_zones()
        yrs = tuple(years) if years is not None else self.years
        return [(z, y) for y in yrs for z in zone_names if not self.has(z, y)]

    def years_fully_complete(self, *, expected_zones: Iterable[str] | None = None) -> list[int]:
        """Years landed in *every* expected zone (default: :meth:`campaign_zones`)."""
        zone_names = tuple(expected_zones) if expected_zones is not None else self.campaign_zones()
        return [y for y in self.years if all(self.has(z, y) for z in zone_names)]


def campaign_status(
    repo: icechunk.Repository,
    *,
    years: tuple[int, ...] = CAMPAIGN_YEARS,
    branch: str = "main",
    snapshot_id: str | None = None,
) -> CampaignStatus:
    """Summarize zone x year fill from the live ``years_complete`` group attrs.

    Reads read-only from ``branch``'s tip, or from ``snapshot_id`` when given (so a caller can
    assess completeness at the exact snapshot it is about to tag, not the moving branch tip). Only
    groups that actually exist are reported (a partially seeded campaign is fine); each group's
    landed years come from the ``years_complete`` attr the shard writer advances atomically with
    the data, so the view never claims a year the data doesn't back.
    """
    session = repo.readonly_session(snapshot_id=snapshot_id) if snapshot_id else repo.readonly_session(branch)
    root = zarr.open_group(session.store, mode="r")
    landed_by_group: dict[str, tuple[int, ...]] = {}
    halves: dict[str, tuple[int, ...]] = {}
    for name in sorted(root.group_keys()):
        node = cast(zarr.Group, root[name])
        raw = node.attrs.get("years_complete", [])
        landed = tuple(sorted(int(y) for y in raw)) if isinstance(raw, list) else ()
        landed_by_group[name] = landed
        if name in zones("utm_zones"):
            for h in HALVES:
                halves[name + h] = tuple(y for y in years if y in landed or h in read_halves_complete(node, y))
    return CampaignStatus(years=tuple(years), zones=landed_by_group, halves=halves)


def campaign_work_list(
    status: CampaignStatus,
    existing_tags: set[str],
    *,
    expected_zones: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
) -> list[tuple[str, int]]:
    """The ``(zone, year)`` cells a campaign run should dispatch, tag-aware.

    Like :meth:`CampaignStatus.pending` (same zones filter + year-major order) but a cell counts
    DONE only when it is BOTH in ``years_complete`` AND has its zone-year tag. Two consequences
    the plain ``pending`` can't express:

    - **Skip finished zones on a default re-run.** With ``expected_zones=None`` (every zone), a year
      that partially completed and is re-driven only dispatches the zones still missing — a
      landed-and-tagged zone is filtered out.
    - **Recover a complete-but-untagged cell.** A crash between the fill commit and the tag leaves
      a cell in ``years_complete`` but untagged; it is INCLUDED so the fill runner's idempotent
      retag path runs (filtering on ``years_complete`` alone would skip it forever, and the year's
      all-zone milestone tag would never land).

    ``expected_zones`` is the zone filter for a subset / repair run (default: every zone of the store's layout).
    """
    # Dedupe (order-preserving): duplicate zones/years from a caller would emit the same
    # (zone, year) twice, and the driver would dispatch two concurrent fills of one cell — their
    # years_complete/runs attr commits conflict (RebaseFailedError).
    zone_names = tuple(dict.fromkeys(expected_zones)) if expected_zones is not None else status.campaign_zones()
    yrs = tuple(dict.fromkeys(years)) if years is not None else status.years
    return [(z, y) for y in yrs for z in zone_names if not status.has(z, y) or zone_year_tag(z, y) not in existing_tags]
