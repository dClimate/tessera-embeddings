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

**Depth rule.** An update applies the store root's ``optical_min_obs`` unless its plan sets a looser
line, which then holds for the plan's cells only (ADR 028): the root and the zone groups keep the
store's rule, so they understate what an updated cell holds. Its update records state the truth:
:func:`cell_optical_min_obs` reads back the floor every embedded pixel meets, and each tile's
registry row its own line. A later update may not apply a stricter line to a cell than an earlier
one did.
"""

from __future__ import annotations

import json
import zipfile
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
from tessera_embeddings.inference.chunk_spec import (
    ChunkSpec,
    chunk_label,
    parse_chunk_label,
    read_pixel_mask,
    write_pixel_mask,
)
from tessera_embeddings.storage.campaign import pin_tag
from tessera_embeddings.storage.registry_master import master_schema
from tessera_embeddings.storage.shard_writer import commit_with_rebase, read_years_complete
from tessera_embeddings.storage.zone_grid import canonicalize_zone

UpdateMode = Literal["holes", "shards"]
OnSkip = Literal["keep", "clear"]
UpdateState = Literal["pending", "done", "not_complete"]
#: Which refusals select a tile from the registry: one reason of `registry.REASONS`, or any of them.
Refused = Literal["no_optical", "thin", "no_radar", "any"]

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
    #: The plan's own depth rule; ``None`` applies the store's (:func:`applied_optical_min_obs`).
    optical_min_obs: int | None = None


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
    #: A depth rule looser than the store root's, for this plan's cells only, e.g. 1 to embed the
    #: pixels an older line refused as thin. ``None`` applies the store's own.
    optical_min_obs: int | None = None
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
        if self.optical_min_obs is not None and self.optical_min_obs < 1:
            raise ValueError(f"optical_min_obs={self.optical_min_obs} refuses nothing; use a positive line")
        if not 0 < self.min_valid_coverage <= 100:
            raise ValueError(f"min_valid_coverage is a percentage in (0, 100], got {self.min_valid_coverage}")
        if not self.cells:
            raise ValueError("An update plan needs at least one cell")
        cells: dict[str, tuple[str, ...] | None] = {}
        for key, tiles in self.cells.items():
            if tiles is not None and not tiles:
                raise ValueError(f"Cell {key} lists no shards; use null for every live shard")
            canonical = cell_key(*parse_cell_key(key))
            if canonical in cells:
                raise ValueError(f"Cell {canonical} is named more than once; list each cell once")
            # One spelling per shard (``chunk_01_02`` is ``chunk_1_2``), so labels compare as the runner's do.
            cells[canonical] = (
                None if tiles is None else tuple(sorted({chunk_label(*parse_chunk_label(t)) for t in tiles}))
            )
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
            optical_min_obs=self.optical_min_obs,
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
        ValueError: If the plan names another update, or ``mask_name`` is not the default and
            differs from the plan's.
    """
    plan = load_plan(plan_uri(paths.update_root(name)))
    # Paths follow the folder and records follow the plan, so the two must be the same update.
    if plan.name != name:
        raise ValueError(f"The plan under update {name!r} is for update {plan.name!r}")
    if mask_name not in ("global", plan.mask_name):
        raise ValueError(f"Update {name!r} was planned under mask {plan.mask_name!r}, not {mask_name!r}")
    return plan


def plan_from_registry(
    registry_root: str,
    *,
    name: str,
    mode: UpdateMode,
    refused: Refused | None = None,
    zones: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    aoi: tuple[float, float, float, float] | None = None,
    storage_options: Mapping[str, Any] | None = None,
    **plan_fields: Any,  # noqa: ANN401 — forwarded to UpdatePlan unchanged
) -> UpdatePlan:
    """Build a plan from the registry's ``master/``: one shard per matching row.

    ``refused`` keeps the tiles that refused pixels for that reason (``refused_<reason>_px > 0``), or
    for any reason; ``None`` keeps every tile. ``aoi`` is a WGS84 ``(west, south, east, north)`` box
    that a tile's box must intersect. Any other selection is a CSV of tiles (:func:`plan_from_tiles`).

    Raises:
        ValueError: If nothing matches, or the AOI crosses the antimeridian.
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
    if refused is not None:
        rows = rows.filter(pl.col("refused_px" if refused == "any" else f"refused_{refused}_px") > 0)
    if aoi is not None:
        west, south, east, north = aoi
        if west > east:
            raise ValueError(f"The AOI {aoi} crosses the antimeridian; plan each side separately")
        # A tile of zone 01 or 60 may cross it (west > east): unwrap its east past 180, and test that
        # span and the same span a turn west.
        tile_west = pl.col("bbox_west")
        tile_east = (
            pl.when(pl.col("bbox_east") < tile_west).then(pl.col("bbox_east") + 360).otherwise(pl.col("bbox_east"))
        )
        overlaps = ((tile_west <= east) & (tile_east >= west)) | ((tile_west - 360 <= east) & (tile_east - 360 >= west))
        rows = rows.filter(overlaps & (pl.col("bbox_south") <= north) & (pl.col("bbox_north") >= south))
    source = f"registry {registry_root} refused={refused} aoi={aoi} zones={zones} years={years}"
    return plan_from_tiles(
        rows.select("zone", "year", "tile").collect(),
        name=name,
        mode=mode,
        **({"source": source} | plan_fields),
    )


def plan_from_tiles(tiles: pl.DataFrame, *, name: str, mode: UpdateMode, **plan_fields: Any) -> UpdatePlan:  # noqa: ANN401
    """Build a plan from ``zone, year, tile`` rows; a null tile asks for every live shard of its cell.

    Zones are canonicalised before grouping, so ``7S`` and ``07S`` rows land in one cell.

    Raises:
        ValueError: If there are no rows.
    """
    if tiles.is_empty():
        raise ValueError(f"Update {name!r} selects no tiles")
    zoned = tiles.with_columns(pl.col("zone").map_elements(canonicalize_zone, return_dtype=pl.String))
    cells: dict[str, tuple[str, ...] | None] = {
        cell_key(zone, year): None if any(t is None for t in group) else tuple(sorted(group))
        for zone, year, group in zoned.group_by("zone", "year").agg("tile").iter_rows()
    }
    return UpdatePlan(name=name, mode=mode, cells=cells, **plan_fields)


def read_updates(node: zarr.Group, year: int) -> list[dict[str, Any]]:
    """The update records on ``year`` of a zone group, oldest first."""
    raw = node.attrs.get(UPDATES_ATTR) or {}
    return list(cast("dict[str, list[dict[str, Any]]]", raw).get(str(year), []))


def applied_optical_min_obs(store_rule: int | None, line: int | None) -> int | None:
    """The depth rule an update's cells are filled under: the plan's ``line``, else the store's ``store_rule``.

    A plan's line may only be looser. A stricter one would leave pixels the store already holds below
    the line its records state, so it is refused, as is any line on a store that declares no rule.
    """
    if line is None:
        return store_rule
    if store_rule is None or line > store_rule:
        raise ValueError(f"An update may only loosen the store's depth rule ({store_rule!r}), not apply {line}")
    return line


def cell_optical_min_obs(root: zarr.Group, zone: str, year: int) -> int | None:
    """The loosest depth rule any write to one cell applied: the store root's, or an update's below it.

    A floor: every pixel embedded in the cell meets it, which the root's ``optical_min_obs`` alone
    does not promise once an update used a lower line. It is not a promise of coverage, since an
    update may have reached only some of the cell's tiles; each tile's own line is its registry
    row's ``optical_min_obs``. ``None`` means the store declares no rule.
    """
    rule = root.attrs.get("optical_min_obs")
    if rule is None:
        return None
    records = read_updates(cast(zarr.Group, root[zone]), year)
    return min([int(cast("int", rule)), *(int(r["optical_min_obs"]) for r in records if r.get("optical_min_obs"))])


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
    differs from the original fill's when the plan set a looser one.
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
    repo: icechunk.Repository,
    group: str,
    year: int,
    commit: Mapping[str, str] | None = None,
    *,
    record: Mapping[str, Any],
    tries: int = 8,
) -> str:
    """Append ``record`` to ``year``'s update records in its own commit, retrying; return the snapshot.

    ``commit`` is the update's shard commit, ``{"snapshot", "committed_at"}``, recorded as null when
    it wrote no shard, as a fill's ``runs`` record does. The same optimistic loop as
    :func:`~tessera_embeddings.storage.shard_writer.commit_year_attrs`, for the same reason: attrs
    cannot be merged by icechunk's conflict detector, and each writer only appends its own record,
    so re-reading and re-applying is exact. Idempotent per update: a record already present for this
    update and year is left as it is.
    """
    for attempt in range(1, tries + 1):
        session = repo.writable_session("main")
        node = cast(zarr.Group, zarr.open_group(session.store, mode="a")[group])
        if update_state(node, year, str(record["update"])) == "done":
            return repo.lookup_branch("main")
        updates = dict(cast("dict[str, list[dict[str, Any]]]", node.attrs.get(UPDATES_ATTR) or {}))
        updates[str(year)] = [*updates.get(str(year), []), {**record, "commit": dict(commit) if commit else None}]
        node.attrs[UPDATES_ATTR] = updates
        try:
            return commit_with_rebase(session, f"update {record['update']}: record {group} year {year}")
        except icechunk.RebaseFailedError:
            if attempt == tries:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def stage_hole_masks(
    node: zarr.Group, *, year_index: int, chunks: Sequence[ChunkSpec], prefix: str, max_workers: int = 16
) -> dict[str, str]:
    """Write each target shard's hole mask — where ``scales`` is NaN — and return ``label -> uri``.

    A shard with no hole is left out of the result: in ``holes`` mode there is nothing to do there,
    so it is neither inferred nor written. A mask already present is reused rather than recomputed,
    so a retry after the shards landed — when the store has no holes left — still infers and merges
    exactly the pixels the first attempt did. One an interrupted write cut short is recomputed:
    nothing was inferred from it.
    """
    scales = cast(zarr.Array, node["scales"])

    def _stage(chunk: ChunkSpec) -> tuple[str, str | None]:
        uri = f"{prefix.rstrip('/')}/{chunk.label}.npz"
        fs, path = fsspec.url_to_fs(uri)
        if fs.exists(path):
            try:
                return chunk.label, uri if read_pixel_mask(uri).any() else None
            except (EOFError, ValueError, zipfile.BadZipFile):
                pass
        holes = np.isnan(np.asarray(scales[year_index, chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop]))
        write_pixel_mask(uri, holes)
        return chunk.label, uri if holes.any() else None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return {label: uri for label, uri in pool.map(_stage, chunks) if uri is not None}
