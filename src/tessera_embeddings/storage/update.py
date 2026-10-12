"""Targeted updates of the global store: re-run chosen shards of cells that are already filled.

An **update** is a named, immutable plan — which ``(zone, year)`` cells, which 2048-px shards in
each (or all of them), under which land mask, in which mode — that the campaign's own flows run
against cells the campaign has already filled. It writes only its target shards, records itself
beside the original fill instead of over it, and never touches a cell's completion mark or tags.
ADR 027 records the decisions; ``orchestration/prefect/README.md`` has the runbook.

**Two modes.**

- ``holes`` infers only the pixels that are holes in the store today (``scales`` is NaN) and
  merges them into the shard, so a pixel that already holds an embedding is never touched — the
  cheap way to fill refusals once more imagery exists. Each hole pixel takes all of the new run's
  arrays, its observation counts and month coverage included, even where it is still refused.
- ``shards`` rewrites each target shard whole, every pixel and every array — for a model or defect
  fix, where the old values are the problem.

A target shard where nothing new embeds is kept as it was (``on_skip="keep"``); a ``shards`` plan
may instead clear it (``on_skip="clear"``), for a defect whose output must go.

**Where things live.** The plan is ``{update_root}/plan.json``, written once. Beside it, per cell:
the ingest ROI (``rois/<zone>/<year>.zarr``), the mosaics (``mosaics/<zone>/<year>``) and, in
``holes`` mode, one mask per target shard and destination store
(``holes/<zone>/<year>/<store digest>/<label>.npz``) — all apart from the
campaign's own, which it rewrites and sweeps (:meth:`BucketPaths.update_root`).

**What the store records.** Each cell an update lands in gains one entry in its zone group's
``updates`` attr, per year an append-only list of records naming the update, its run, its code,
the depth rule it applied (``optical_min_obs``) and the shards it ``written``/``kept``/``cleared``.
``years_complete``, ``runs`` and the zone-year tags stay exactly as the original fill left them,
so a cell's state is ``runs[year]`` with ``updates[year]`` applied on top, in order.
``update-<name>-base`` pins the store before an update's first write and ``update-<name>-complete``
the store once every cell has landed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast, get_args

import fsspec
import icechunk
import numpy as np
import polars as pl
import pyarrow as pa
import zarr

from tessera_embeddings.config.environment import code_identity
from tessera_embeddings.config.paths import BucketPaths, check_update_name
from tessera_embeddings.inference.chunk_spec import ChunkSpec, parse_chunk_label, read_pixel_mask, write_pixel_mask
from tessera_embeddings.storage.campaign import pin_tag
from tessera_embeddings.storage.registry_master import master_schema
from tessera_embeddings.storage.shard_writer import commit_with_rebase, read_years_complete
from tessera_embeddings.storage.zone_grid import canonicalize_zone

UpdateMode = Literal["holes", "shards"]
OnSkip = Literal["keep", "clear"]
UpdateState = Literal["pending", "done", "not_complete"]

#: The registry's hive partition keys, which come from the path rather than the files.
_PARTITION_KEYS = ("zone", "year")

#: The zone-group attr holding each year's update records.
UPDATES_ATTR = "updates"

#: An update's default date-acceptance threshold, in PERCENT of the ROI: low enough that a date with
#: one valid ROI pixel passes even over a whole zone (~1e10 px). The campaign's 0.1 is a share of
#: the ROI, so a small one would drop dates a zone-wide fill kept; zero is refused by
#: ``IngestSettings`` because it admits dates with nothing in them.
UPDATE_MIN_VALID_COVERAGE = 1e-9


def cell_key(zone: str, year: int) -> str:
    """A plan's key for one cell, e.g. ``"14N/2019"``."""
    return f"{canonicalize_zone(zone)}/{int(year)}"


def parse_cell_key(key: str) -> tuple[str, int]:
    """Invert :func:`cell_key`; raises on anything else."""
    zone, sep, year = key.partition("/")
    if not sep or not year.isdigit():
        raise ValueError(f"Cell key {key!r} is not of the form '<zone>/<year>'")
    return canonicalize_zone(zone), int(year)


def base_tag(name: str) -> str:
    """The tag pinning the store before update ``name``'s first write."""
    return f"update-{check_update_name(name)}-base"


def complete_tag(name: str) -> str:
    """The tag pinning the store once every cell of update ``name`` has landed."""
    return f"update-{check_update_name(name)}-complete"


@dataclass(frozen=True)
class UpdateTarget:
    """One cell of an update, as the fill needs it. Plain data, so it crosses threads and pickles."""

    name: str
    mode: UpdateMode
    on_skip: OnSkip
    mask_name: str
    #: Target shard labels, or ``None`` for every shard the mask marks live.
    tiles: frozenset[str] | None
    #: Prefix of this cell's hole masks (``holes`` mode).
    holes: str
    #: Target shards planning found nothing to do in — no holes — so they are kept untouched.
    kept: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpdatePlan:
    """A named, immutable update: which cells and shards, under which mask, in which mode."""

    name: str
    mode: UpdateMode
    #: ``cell_key -> shard labels``, or ``None`` for every shard the mask marks live.
    cells: Mapping[str, tuple[str, ...] | None]
    mask_name: str = "global"
    on_skip: OnSkip = "keep"
    min_valid_coverage: float = UPDATE_MIN_VALID_COVERAGE
    #: Relaxes the ingest's every-month check to "non-empty". On by default for the same reason the
    #: threshold is low: a small cloudy ROI can lack a valid date in a month a zone never did.
    allow_partial_window: bool = True
    source: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        """Reject a plan the flows could not run, before anything is written."""
        check_update_name(self.name)
        if self.mode not in get_args(UpdateMode):
            raise ValueError(f"mode must be one of {get_args(UpdateMode)}, got {self.mode!r}")
        if self.on_skip not in get_args(OnSkip):
            raise ValueError(f"on_skip must be one of {get_args(OnSkip)}, got {self.on_skip!r}")
        if self.mode == "holes" and self.on_skip != "keep":
            raise ValueError("A holes update only adds pixels, so it has nothing to clear: use on_skip='keep'")
        if not 0 < self.min_valid_coverage <= 100:
            raise ValueError(f"min_valid_coverage is a percentage in (0, 100], got {self.min_valid_coverage}")
        if not self.cells:
            raise ValueError("An update plan needs at least one cell")
        cells: dict[str, tuple[str, ...] | None] = {}
        for key, tiles in self.cells.items():
            if tiles is not None and not tiles:
                raise ValueError(f"Cell {key} lists no shards; use null for every live shard")
            for label in tiles or ():
                parse_chunk_label(label)
            cells[cell_key(*parse_cell_key(key))] = None if tiles is None else tuple(sorted(set(tiles)))
        object.__setattr__(self, "cells", cells)

    def cell_list(
        self, *, zones: Iterable[str] | None = None, years: Iterable[int] | None = None
    ) -> list[tuple[str, int]]:
        """The plan's cells, optionally narrowed to ``zones`` and ``years``, in key order."""
        wanted_zones = None if zones is None else {canonicalize_zone(z) for z in zones}
        wanted_years = None if years is None else {int(y) for y in years}
        cells = [parse_cell_key(key) for key in sorted(self.cells)]
        return [
            (z, y)
            for z, y in cells
            if (wanted_zones is None or z in wanted_zones) and (wanted_years is None or y in wanted_years)
        ]

    def target(self, zone: str, year: int, *, root: str) -> UpdateTarget:
        """The fill's view of one cell; ``root`` is :meth:`BucketPaths.update_root` for this plan."""
        key = cell_key(zone, year)
        if key not in self.cells:
            raise ValueError(f"Cell {key} is not in update {self.name!r}")
        tiles = self.cells[key]
        return UpdateTarget(
            name=self.name,
            mode=self.mode,
            on_skip=self.on_skip,
            mask_name=self.mask_name,
            tiles=None if tiles is None else frozenset(tiles),
            holes=holes_prefix(root, *parse_cell_key(key)),
        )

    def to_json(self) -> str:
        """The plan as it is stored."""
        record = asdict(self)
        record["cells"] = {k: None if v is None else list(v) for k, v in sorted(self.cells.items())}
        return json.dumps(record, indent=1)

    @classmethod
    def from_json(cls, text: str) -> UpdatePlan:
        """Read a plan written by :meth:`to_json`."""
        record = json.loads(text)
        record["cells"] = {k: None if v is None else tuple(v) for k, v in record["cells"].items()}
        return cls(**record)


def plan_uri(root: str) -> str:
    """Where an update's plan lives under its :meth:`BucketPaths.update_root`."""
    return f"{root.rstrip('/')}/plan.json"


def roi_uri(root: str, zone: str, year: int) -> str:
    """One cell's ingest ROI. Per year, because an update's shards may differ from year to year."""
    return f"{root.rstrip('/')}/rois/{zone}/{year}.zarr"


def holes_prefix(root: str, zone: str, year: int) -> str:
    """Where one cell's hole masks live, one ``<label>.npz`` per target shard."""
    return f"{root.rstrip('/')}/holes/{zone}/{year}"


def save_plan(plan: UpdatePlan, uri: str) -> None:
    """Write ``plan`` once. An existing plan is never replaced: a rerun keys on its name.

    Raises:
        FileExistsError: If a plan already exists at ``uri``.
    """
    fs, path = fsspec.url_to_fs(uri)
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    # Exclusive create, so two operators cannot both believe they wrote it. s3fs sends it as a
    # conditional PUT (IfNoneMatch), the local filesystem as O_EXCL.
    with fs.open(path, "xb") as f:
        f.write(plan.to_json().encode())


def load_plan(uri: str) -> UpdatePlan:
    """Read the plan at ``uri``."""
    with fsspec.open(uri, "rb") as f:
        return UpdatePlan.from_json(f.read().decode())


def check_radar_policy(plan: UpdatePlan | UpdateTarget, *, allow_s2_only: bool) -> None:
    """Refuse a ``holes`` update run with the radar rule on.

    A merged shard's registry rows are re-derived from the store, and the store records whether a
    pixel had radar but not which pixels a radar rule refused, so those rows cannot describe a run
    that enforced one. The campaign embeds radar-free pixels (``allow_s2_only``); its updates must too.

    Raises:
        ValueError: If ``plan`` is a ``holes`` update and ``allow_s2_only`` is false.
    """
    if plan.mode == "holes" and not allow_s2_only:
        raise ValueError(
            f"Update {plan.name!r} fills holes, which needs allow_s2_only=True: the registry rows of a merged "
            "shard are re-derived from the store, which cannot reproduce a radar refusal"
        )


def load_update(paths: BucketPaths, name: str, *, mask_name: str = "global") -> UpdatePlan:
    """Load update ``name``'s plan for a flow, refusing a mask that contradicts it.

    The plan is the authority on the mask its shards were chosen under, so a flow takes it from
    there. An explicitly passed mask that differs is an operator who believes the run uses another,
    and is told rather than silently overruled; the default ``"global"`` is taken to mean "unset".

    Raises:
        ValueError: If ``mask_name`` is not the default and differs from the plan's.
    """
    plan = load_plan(plan_uri(paths.update_root(name)))
    if mask_name not in ("global", plan.mask_name):
        raise ValueError(f"Update {name!r} was planned under mask {plan.mask_name!r}, not {mask_name!r}")
    return plan


def plan_from_registry(
    registry_root: str,
    *,
    name: str,
    mode: UpdateMode,
    zones: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    where: str | None = None,
    aoi: tuple[float, float, float, float] | None = None,
    storage_options: Mapping[str, Any] | None = None,
    **plan_fields: Any,  # noqa: ANN401 — forwarded to UpdatePlan unchanged
) -> UpdatePlan:
    """Build a plan from the registry's ``master/``: one shard per matching row.

    ``where`` is a SQL expression over the master's columns, e.g. ``"refused_thin_px > 0"`` (rank
    by ``median_obs_where_thin``, not ``median_obs_where_any``; see :mod:`.registry`). ``aoi`` is a
    WGS84 ``(west, south, east, north)`` box; a row matches when its tile's box intersects it. Either
    box may cross the antimeridian (``west > east``), as the rows of zones 01 and 60 do.

    Raises:
        ValueError: If nothing matches.
    """
    # THE SCHEMA IS STATED, as every whole-registry read must (`registry.py`): inferred, it comes from
    # the first file, and cell files written before and after a column was added then fail to scan.
    columns = pa.schema([f for f in master_schema() if f.name not in _PARTITION_KEYS])
    rows = pl.scan_parquet(
        f"{registry_root.rstrip('/')}/master/**/*.parquet",
        schema=pl.DataFrame(columns.empty_table()).schema,
        hive_partitioning=True,
        hive_schema={"zone": pl.String, "year": pl.Int64},
        missing_columns="insert",
        extra_columns="ignore",
        storage_options=dict(storage_options) if storage_options else None,
    )
    if zones is not None:
        rows = rows.filter(pl.col("zone").is_in([canonicalize_zone(z) for z in zones]))
    if years is not None:
        rows = rows.filter(pl.col("year").is_in([int(y) for y in years]))
    if where:
        rows = rows.filter(pl.sql_expr(where))
    if aoi is not None:
        west, south, east, north = aoi
        # Either box may cross the antimeridian (west > east). Each is the union of its intervals on
        # [-180, 180], and two boxes overlap when any pair of those intervals does.
        tile_spans = [
            (
                pl.col("bbox_west"),
                pl.when(pl.col("bbox_west") > pl.col("bbox_east")).then(180.0).otherwise(pl.col("bbox_east")),
            ),
            (
                pl.when(pl.col("bbox_west") > pl.col("bbox_east")).then(-180.0).otherwise(pl.col("bbox_west")),
                pl.col("bbox_east"),
            ),
        ]
        aoi_spans = [(west, east)] if west <= east else [(west, 180.0), (-180.0, east)]
        overlaps_lon = pl.any_horizontal([(tw <= ae) & (te >= aw) for tw, te in tile_spans for aw, ae in aoi_spans])
        rows = rows.filter(overlaps_lon & (pl.col("bbox_south") <= north) & (pl.col("bbox_north") >= south))
    matched = rows.select("zone", "year", "tile").collect()
    if matched.is_empty():
        raise ValueError(f"No registry rows under {registry_root} match where={where!r} aoi={aoi}")
    grouped = matched.group_by("zone", "year").agg(pl.col("tile"))
    cells = {cell_key(zone, year): tuple(sorted(tiles)) for zone, year, tiles in grouped.iter_rows()}
    source = f"registry {registry_root} where={where!r} aoi={aoi} zones={zones} years={years}"
    return UpdatePlan(name=name, mode=mode, cells=cells, source=plan_fields.pop("source", source), **plan_fields)


def read_updates(node: zarr.Group, year: int) -> list[dict[str, Any]]:
    """The update records on ``year`` of a zone group, oldest first."""
    raw = node.attrs.get(UPDATES_ATTR) or {}
    return list(cast("dict[str, list[dict[str, Any]]]", raw).get(str(year), []))


def update_state(node: zarr.Group, year: int, name: str) -> UpdateState:
    """Where update ``name`` stands on one cell.

    ``not_complete`` when the year was never filled: an update is an insert into existing data,
    and an unfilled year is the campaign's to fill.
    """
    if any(record.get("update") == name for record in read_updates(node, year)):
        return "done"
    return "pending" if year in read_years_complete(node) else "not_complete"


def update_states(
    repo: icechunk.Repository, name: str, cells: Iterable[tuple[str, int]]
) -> dict[tuple[str, int], UpdateState]:
    """Where update ``name`` stands on each of ``cells``, read from the store's ``main`` in one session.

    Raises:
        ValueError: If a cell's zone is not seeded in the store.
    """
    root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    states: dict[tuple[str, int], UpdateState] = {}
    for zone, year in cells:
        if zone not in root:
            raise ValueError(f"Update {name!r} names zone {zone}, which is not seeded in the store")
        states[(zone, year)] = update_state(cast(zarr.Group, root[zone]), year, name)
    return states


def finish_update(repo: icechunk.Repository, plan: UpdatePlan) -> str | None:
    """Pin :func:`complete_tag` once every cell of ``plan`` has recorded the update; return it, or ``None``.

    Over the WHOLE plan, so a run given a subset of its cells cannot declare the update complete.
    """
    states = update_states(repo, plan.name, plan.cell_list())
    if all(state == "done" for state in states.values()):
        return pin_tag(repo, complete_tag(plan.name))
    return None


def update_record(
    target: UpdateTarget,
    *,
    run_id: str,
    written: Sequence[str],
    kept: Sequence[str],
    optical_min_obs: int | None,
    cleared: Sequence[str] = (),
    radar_coverage: Mapping[str, Any] | None = None,
    input_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One cell's record of an update: what it wrote, what it kept, under which depth rule and with which code.

    ``optical_min_obs`` is the line the update judged its pixels against (``None``: no rule), which
    after a loosening differs from the one the original fill applied.
    """
    record: dict[str, Any] = {
        "update": target.name,
        "mode": target.mode,
        "run_id": run_id,
        "assembled_at": datetime.now(UTC).isoformat(),
        "mask_name": target.mask_name,
        "optical_min_obs": optical_min_obs,
        "written": sorted(written),
        "kept": sorted(kept),
        "cleared": sorted(cleared),
    }
    if radar_coverage:
        record["radar_coverage"] = dict(radar_coverage)
    if input_coverage:
        record["input_coverage"] = dict(input_coverage)
    if code := code_identity():
        record["code"] = dict(code)
    return record


def commit_update_record(
    repo: icechunk.Repository, group: str, year: int, *, record: Mapping[str, Any], tries: int = 8
) -> str:
    """Append ``record`` to ``year``'s update records in its own commit, retrying; return the snapshot.

    The same optimistic loop as :func:`~tessera_embeddings.storage.shard_writer.commit_year_attrs`,
    for the same reason: attrs cannot be merged by icechunk's conflict detector, and each writer
    only appends its own record, so re-reading and re-applying is exact. Idempotent per update: a
    record already present for this update and year is left as it is.
    """
    for attempt in range(1, tries + 1):
        session = repo.writable_session("main")
        node = cast(zarr.Group, zarr.open_group(session.store, mode="a")[group])
        if update_state(node, year, str(record["update"])) == "done":
            return repo.lookup_branch("main")
        updates = dict(cast("dict[str, list[dict[str, Any]]]", node.attrs.get(UPDATES_ATTR) or {}))
        updates[str(year)] = [*updates.get(str(year), []), dict(record)]
        node.attrs[UPDATES_ATTR] = updates
        try:
            return commit_with_rebase(session, f"update {record['update']}: record {group} year {year}")
        except icechunk.RebaseFailedError:
            if attempt == tries:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def last_run_id(attrs: Mapping[str, Any], year: int) -> str | None:
    """The run that last wrote ``year`` of a zone group: its fill, or the newest update since that wrote a shard."""
    runs = cast("Mapping[str, Mapping[str, Any]]", attrs.get("runs") or {})
    updates = cast("Mapping[str, Sequence[Mapping[str, Any]]]", attrs.get(UPDATES_ATTR) or {})
    # An update that wrote and cleared nothing published no registry part, so it is not the run the
    # registry's newest row names, and treating it as one would block that cell's repair.
    wrote = [u for u in updates.get(str(year), ()) if u.get("written") or u.get("cleared")]
    entries = [e for e in (runs.get(str(year)), *wrote) if e]
    if not entries:
        return None
    return str(max(entries, key=lambda e: str(e.get("assembled_at", "")))["run_id"])


def stage_hole_masks(
    node: zarr.Group, *, year_index: int, chunks: Sequence[ChunkSpec], prefix: str, max_workers: int = 16
) -> dict[str, str]:
    """Write each target shard's hole mask — where ``scales`` is NaN — and return ``label -> uri``.

    A shard with no hole is left out of the result: in ``holes`` mode there is nothing to do there,
    so it is neither inferred nor written. A mask already present is reused rather than recomputed,
    so a retry after the shards landed — when the store has no holes left — still infers and merges
    exactly the pixels the first attempt did.
    """
    scales = cast(zarr.Array, node["scales"])

    def _stage(chunk: ChunkSpec) -> tuple[str, str | None]:
        uri = f"{prefix.rstrip('/')}/{chunk.label}.npz"
        fs, path = fsspec.url_to_fs(uri)
        if fs.exists(path):
            return chunk.label, uri if read_pixel_mask(uri).any() else None
        holes = np.isnan(np.asarray(scales[year_index, chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop]))
        write_pixel_mask(uri, holes)
        return chunk.label, uri if holes.any() else None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return {label: uri for label, uri in pool.map(_stage, chunks) if uri is not None}
