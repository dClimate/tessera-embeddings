"""The registry master: one current row per tile-year, brought up to date as each fill lands.

**What this is for.** :mod:`~tessera_embeddings.storage.registry` writes one part per cell per run,
which is the right shape for a campaign writing cells concurrently and the wrong shape for anybody
reading the result. Two consequences the master closes:

* **A cell filled more than once has more than one part**, and something has to decide which row
  describes each tile now.
* **``pyarrow`` infers a dataset's schema from the first file in sorted path order**, so a column
  added mid-campaign is silently dropped if zone ``01N`` was written before the change. A master
  written with one schema, and a dataset-level ``_common_metadata`` beside it, removes the hazard
  instead of asking every reader to remember it.

**Three prefixes, and only one of them is for reading.**

```
  parts/     each fill's own output, one part per cell per run, append-only, NEVER rewritten.
  rebuild/   measurements re-derived from the store for rows a fill left null
             (:mod:`.registry_rebuild`). Additive; rollback is deleting this prefix.
  master/    one file per cell, master/zone=<Z>/year=<Y>/part-0.parquet: one row per tile, one
             schema, and two columns saying where each row's numbers came from. Consumers read this.
```

**It is always current.** The zone-fill runner calls :func:`update_cell` as soon as a cell's part is
published. That recomputes the cell's master file from ALL of the cell's parts and rebuild rows and
replaces it in one PUT. Cells are independent, so concurrent fills never write the same object, and
the update is idempotent. ``sync`` in ``scripts/maintenance/rebuild_registry_measurements.py`` does
the same for every cell, as a repair.

**A tile's row comes from the newest run that wrote that tile.** A run writes only the tiles it
staged or refused and leaves the year's other shards as they were (``assemble_global``'s refill
caveat). So the store holds each tile as the most recently assembled run that wrote it left it,
whichever run supplied the cell's other tiles, and the master follows the store: a full refill
supersedes every tile, and a partial run such as an infill supersedes only its own and removes
none. A run's row must therefore describe the whole tile as that run leaves it, because it becomes
the tile's only row. Runs are ordered by parsed ``assembled_at`` and then ``run_id``, so a tie
resolves the same way on every merge. A rebuild row counts only if it was re-derived against the
tile's current run (its ``filled_at`` is that run's stamp): one made against a superseded run
describes pixels the store no longer holds.

**Within that run, precedence is not latest-wins, and the difference matters.** A fill measured pixels
as it wrote them; a rebuild re-derived them afterwards from what was written. They should agree —
that is what :func:`~.registry_rebuild.compare_row` gates on — but where a fill recorded a
measurement, that measurement stands. So:

1. a fill row carrying measurements beats everything;
2. otherwise a rebuild row carrying measurements — one that measured radar ahead of an
   optical-only one, then the most recent — which is how the radar pass supersedes the optical-only
   pass and why a later optical-only retry cannot undo it;
3. otherwise the most recent row of any kind, which keeps a tile that nothing has measured in the
   master as a null row rather than dropping it.

**``measured_by`` is load-bearing.** Without it a re-derived value is indistinguishable from one
the pipeline measured, and the registry loses the property that makes it worth trusting. Null keeps
meaning "not measured" and does not quietly come to mean "measured by something else".
"""

from __future__ import annotations

import datetime
import io
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow.fs import FileType

from tessera_embeddings.storage.registry import dataset_schema
from tessera_embeddings.storage.registry_rebuild import rebuild_dataset_schema

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Sequence

    from pyarrow.fs import FileSystem

#: Value of ``measured_by`` for a row whose numbers the fill itself recorded.
MEASURED_BY_FILL = "fill"

#: Value of ``measured_by`` for a row whose numbers were re-derived from the store afterwards.
MEASURED_BY_REBUILD = "rebuild"

#: The column that decides whether a row carries measurements at all. Every measurement column is
#: null together — a row is measured or it is not — so one column answers it, and this is the one
#: a rebuild can always populate.
MEASURED_SENTINEL = "chunk_px"


def master_schema() -> pa.Schema:
    """The master's schema: the registry's, plus where each row's numbers came from.

    ``measured_by`` is ``"fill"``, ``"rebuild"`` or null, never inferred from the presence of a
    value. ``filled_at`` is when the run the tile comes from filled it, which a rebuilt row would
    otherwise lose to its own ``assembled_at``.
    """
    return pa.schema(
        [
            *dataset_schema(),
            pa.field("measured_by", pa.string()),
            pa.field("filled_at", pa.string()),
        ]
    )


def _aligned(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """``table`` widened to ``schema``, missing columns null, column order fixed.

    Fill parts have no ``filled_at`` and neither has ``measured_by``; a rebuild part has the first
    and not the second. Concatenating them needs one shape, and a missing column must arrive as
    null — the honest value for a pass that did not record it — rather than as a default.
    """
    columns = []
    for field in schema:
        if field.name in table.column_names:
            columns.append(table.column(field.name).cast(field.type))
        else:
            columns.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(columns, schema=schema)


def merge(parts: pa.Table, rebuilt: pa.Table | None = None) -> pa.Table:
    """One row per ``(zone, year, tile)``, resolved by the rules in the module docstring.

    Pure, and over whole tables rather than a stream: the published registry is 3.2 million rows
    and 142 MB, so a whole-registry merge fits in memory and one cell's is trivial.
    """
    schema = master_schema()
    fill = _aligned(current_fill_rows(parts), schema)
    # A fill part's own stamp IS when its tile was filled, so every master row answers that question.
    index = schema.get_field_index("filled_at")
    fill = fill.set_column(index, "filled_at", pc.coalesce(fill.column(index), fill.column("assembled_at")))
    tables = [_tagged(fill, MEASURED_BY_FILL)]
    if rebuilt is not None and rebuilt.num_rows:
        # Only re-derivations of the run each tile now comes from: one made against a superseded run
        # describes pixels the store no longer holds.
        current = pc.is_in(_key(rebuilt, "tile", "filled_at"), value_set=_key(fill, "tile", "assembled_at"))
        tables.append(_tagged(_aligned(rebuilt.filter(current), schema), MEASURED_BY_REBUILD))
    combined = pa.concat_tables(tables)
    if not combined.num_rows:
        return combined
    keep = _winning_rows(combined)
    return combined.take(pa.array(keep)).sort_by([("zone", "ascending"), ("year", "ascending"), ("tile", "ascending")])


def current_fill_rows(parts: pa.Table) -> pa.Table:
    """Per tile, only the rows of the most recently assembled fill run that wrote that tile.

    Every caller that compares against the fill — the merge, :func:`fill_rows_unchanged`, the
    rebuild's choice of rows to re-derive — goes through this, so all of them agree on which run
    each tile comes from.
    """
    if not parts.num_rows:
        return parts
    rank = _run_rank(parts)
    tile = _key(parts, "tile")
    newest = pa.table({"tile": tile, "rank": rank}).group_by("tile").aggregate([("rank", "max")])
    best = pc.take(newest.column("rank_max"), pc.index_in(tile, value_set=newest.column("tile")))
    return parts.filter(pc.equal(rank, best))


def _run_rank(parts: pa.Table) -> pa.Array:
    """Each row's run, as its position in assembly order: parsed ``assembled_at``, then ``run_id``.

    The ``run_id`` tiebreak makes two runs stamped the same instant resolve the same way on every
    merge, so a whole-registry ``sync`` and a one-cell update cannot disagree.
    """
    runs = parts.group_by(["zone", "year", "run_id"]).aggregate([("assembled_at", "min"), ("assembled_at", "max")])
    ordered: list[tuple[datetime.datetime, str, str]] = []
    for run in runs.to_pylist():
        # min and max both parse, so every stamp in the run does: ISO-8601 sorts as text only within a format.
        _aware(run["assembled_at_min"])
        key = f"{run['zone']}/{run['year']}/{run['run_id']}"
        ordered.append((_aware(run["assembled_at_max"]), run["run_id"], key))
    return pc.index_in(_key(parts, "run_id"), value_set=pa.array([key for *_, key in sorted(ordered)]))


def _aware(stamp: str | None) -> datetime.datetime:
    """``stamp`` as an aware datetime; raises rather than letting an unorderable run be chosen."""
    try:
        parsed = datetime.datetime.fromisoformat(stamp or "")
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise ValueError(f"assembled_at {stamp!r} is not an aware ISO-8601 timestamp, so its run cannot be ordered")
    return parsed


def _key(table: pa.Table, *columns: str) -> pa.Array:
    """``zone/year/<columns>`` per row, for matching rows to a run, a tile, or a tile's stamp."""
    year = pc.cast(table.column("year"), pa.string())
    fields = [table.column("zone"), year, *(table.column(column) for column in columns)]
    return pc.binary_join_element_wise(*fields, "/").combine_chunks()


def _tagged(table: pa.Table, origin: str) -> pa.Table:
    """``measured_by`` set to ``origin`` where the row carries measurements, null where it does not.

    A fill row with every measurement null says "nothing measured this", and stamping it ``"fill"``
    would make ``measured_by`` mean "which pass wrote this row" instead of "who measured these
    numbers" — at which point it stops answering the only question it exists for.
    """
    measured = pc.is_valid(table.column(MEASURED_SENTINEL))
    tag = pc.if_else(measured, pa.scalar(origin, pa.string()), pa.scalar(None, pa.string()))
    return table.set_column(table.schema.get_field_index("measured_by"), "measured_by", tag)


def _winning_rows(table: pa.Table) -> list[int]:
    """Row indices to keep — one per ``(zone, year, tile)``, by the documented precedence.

    Sorted rather than grouped: a stable lexicographic sort on
    ``(key, rank, reverse timestamp)`` puts each group's winner first, and the first occurrence of
    each key is then a single pass. ``numpy.lexsort`` takes its keys last-significant-first.
    """
    zone = np.asarray(table.column("zone").to_pylist(), dtype=object)
    year = np.asarray(table.column("year").to_pylist(), dtype=object)
    tile = np.asarray(table.column("tile").to_pylist(), dtype=object)
    key = np.array([f"{z}/{y}/{t}" for z, y, t in zip(zone, year, tile, strict=True)])

    measured = np.asarray(pc.is_valid(table.column(MEASURED_SENTINEL)).to_pylist(), dtype=bool)
    by_fill = np.asarray([value == MEASURED_BY_FILL for value in table.column("measured_by").to_pylist()], dtype=bool)
    # 0 beats 1 beats 2 beats 3. A measured fill row is authoritative; a measured rebuild row fills a
    # gap, one that measured radar ahead of an optical-only one whatever their order, so a later
    # `--skip-radar` retry cannot null a radar column a full pass already landed; an unmeasured row
    # of either kind keeps the tile present and says nothing about it.
    radar = np.asarray(pc.is_valid(table.column("px_with_any_radar")).to_pylist(), dtype=bool)
    rank = np.where(measured & by_fill, 0, np.where(measured & radar, 1, np.where(measured, 2, 3)))
    stamp = np.array([value or "" for value in table.column("assembled_at").to_pylist()])

    order = np.lexsort((_descending(stamp), rank, key))
    sorted_keys = key[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return [int(index) for index in order[first]]


def _descending(stamps: np.ndarray) -> np.ndarray:
    """Sort positions that put the LATEST timestamp first under an ascending sort.

    ``numpy.lexsort`` sorts ascending only and strings cannot be negated, so the timestamps are
    replaced by their own descending rank. ISO-8601 stamps compare correctly as strings when they
    share a format; ranking sidesteps the question of whether two passes wrote the same one.
    """
    order = np.argsort(stamps, kind="stable")
    rank = np.empty(len(stamps), dtype=np.int64)
    rank[order] = np.arange(len(stamps))
    return -rank


def invariant_failures(master: pa.Table, *, expected_rows: int | None = None) -> list[str]:
    """What is wrong with a master, as a list of sentences. Empty means nothing is.

    Checked rather than assumed, because every one of these has a silent failure mode: a duplicate
    tile-year reads as coverage counted twice, a lost one as ground nobody wrote, and a rebuilt
    value sitting where a fill measured one would mean the merge overrode a measurement with a
    re-derivation. ``expected_rows`` is :func:`tile_years` of the parts, since no run removes a tile.
    """
    failures: list[str] = []
    if expected_rows is not None and master.num_rows != expected_rows:
        failures.append(f"master holds {master.num_rows} rows, expected {expected_rows}")
    keys = [
        f"{z}/{y}/{t}"
        for z, y, t in zip(
            master.column("zone").to_pylist(),
            master.column("year").to_pylist(),
            master.column("tile").to_pylist(),
            strict=True,
        )
    ]
    if len(set(keys)) != len(keys):
        failures.append(
            f"{len(keys) - len(set(keys))} duplicate (zone, year, tile) rows — a tile-year must appear once"
        )
    measured = pc.is_valid(master.column(MEASURED_SENTINEL)).to_pylist()
    origin = master.column("measured_by").to_pylist()
    mislabelled = sum(1 for is_measured, tag in zip(measured, origin, strict=True) if is_measured != (tag is not None))
    if mislabelled:
        failures.append(
            f"{mislabelled} rows where measured_by and the measurements disagree about whether any were taken"
        )
    unknown = {tag for tag in origin if tag not in (None, MEASURED_BY_FILL, MEASURED_BY_REBUILD)}
    if unknown:
        failures.append(f"measured_by holds values that name no pass: {sorted(unknown)}")
    return failures


def fill_rows_unchanged(master: pa.Table, parts: pa.Table, columns: Sequence[str]) -> list[str]:
    """Rows the master claims the fill measured, that differ from the fill's own part. Empty is correct.

    The merge is only safe if it never alters an original measurement, and "never" is a claim worth
    testing against the source rather than reasoning about from the merge rule. The source is each
    tile's current run, the one :func:`merge` keeps, so a refilled tile is compared against the run
    the store holds rather than whichever part a dataset scan happened to read last.
    """
    wanted = set(columns)
    by_key: dict[str, dict[str, Any]] = {}
    for row in current_fill_rows(parts).to_pylist():
        if row.get(MEASURED_SENTINEL) is None:
            continue
        by_key[f"{row['zone']}/{row['year']}/{row['tile']}"] = row
    differences: list[str] = []
    for row in master.to_pylist():
        if row.get("measured_by") != MEASURED_BY_FILL:
            continue
        key = f"{row['zone']}/{row['year']}/{row['tile']}"
        source = by_key.get(key)
        if source is None:
            differences.append(f"{key}: master says the fill measured it, the parts hold no measured row")
            continue
        for column in sorted(wanted):
            if row.get(column) != source.get(column):
                differences.append(f"{key}.{column}: master {row.get(column)!r} != part {source.get(column)!r}")
    return differences


def tile_years(parts: pa.Table) -> int:
    """Distinct ``(zone, year, tile)`` across every run: the rows a master of ``parts`` must hold."""
    return parts.group_by(["zone", "year", "tile"]).aggregate([]).num_rows


def cell_tables(table: pa.Table) -> Iterator[tuple[str, int, pa.Table]]:
    """``(zone, year, rows)`` for each cell of ``table``, which must be sorted by zone then year.

    Slices rather than filters, so splitting the whole registry is one pass instead of a scan per cell.
    """
    cells = list(zip(table.column("zone").to_pylist(), table.column("year").to_pylist(), strict=True))
    start = 0
    for end in range(1, len(cells) + 1):
        if end == len(cells) or cells[end] != cells[start]:
            yield str(cells[start][0]), int(cells[start][1]), table.slice(start, end - start)
            start = end


def stale_cells(published: pa.Table, fresh: pa.Table) -> list[str]:
    """``zone/year`` of every cell whose published master rows differ from a fresh merge's.

    Catches what the invariants cannot: a fill whose master update failed, a rebuild pass nothing
    merged, a value altered after it was written. Each leaves every row present and labelled.
    """
    order = [("zone", "ascending"), ("year", "ascending"), ("tile", "ascending")]
    held = {(zone, year): rows for zone, year, rows in cell_tables(published.sort_by(order))}
    want = {(zone, year): rows for zone, year, rows in cell_tables(fresh.sort_by(order))}
    return sorted(
        f"{zone}/{year}"
        for zone, year in held.keys() | want.keys()
        if (zone, year) not in held or (zone, year) not in want or not held[zone, year].equals(want[zone, year])
    )


def _bare(registry_root: str) -> str:
    """``registry_root`` as a pyarrow filesystem path: no scheme, no trailing slash."""
    return registry_root.removeprefix("s3://").rstrip("/")


def read_cell(fs: FileSystem, registry_root: str, prefix: str, zone: str, year: int, schema: pa.Schema) -> pa.Table:
    """One cell's rows under ``prefix``, partition keys included; empty when the cell has none there."""
    base = f"{_bare(registry_root)}/{prefix}"
    path = f"{base}/zone={zone}/year={year}"
    if fs.get_file_info(path).type == FileType.NotFound:
        return schema.empty_table()
    return ds.dataset(path, filesystem=fs, schema=schema, partitioning="hive", partition_base_dir=base).to_table()


def write_cell(fs: FileSystem, registry_root: str, zone: str, year: int, rows: pa.Table) -> str:
    """Replace one cell's master file with ``rows`` in a single PUT; returns the path written.

    One object per cell, replaced whole, so a reader sees the old file or the new one and never a
    mixture, and no two cells share an object. ``_common_metadata`` is the same bytes for every
    cell, so it is written only when absent rather than raced by every fill.
    """
    path = f"{_bare(registry_root)}/master/zone={zone}/year={year}/part-0.parquet"
    buf = io.BytesIO()
    # The partition keys live in the path, as in every hive dataset.
    pq.write_table(rows.drop_columns(["zone", "year"]), buf, compression="zstd")
    fs.create_dir(path.rsplit("/", 1)[0])
    with fs.open_output_stream(path) as out:
        out.write(buf.getvalue())
    if fs.get_file_info(f"{_bare(registry_root)}/master/_common_metadata").type == FileType.NotFound:
        write_common_metadata(fs, registry_root)
    return path


def write_common_metadata(fs: FileSystem, registry_root: str) -> None:
    """The master's dataset-level schema, so a reader takes it from the dataset rather than a file."""
    with fs.open_output_stream(f"{_bare(registry_root)}/master/_common_metadata") as out:
        pq.ParquetWriter(out, master_schema()).close()


def update_cell(
    fs: FileSystem,
    registry_root: str,
    zone: str,
    year: int,
    *,
    live: Collection[tuple[int, int]] | None = None,
) -> list[str]:
    """Recompute one cell's master file from all of its parts and rebuild rows, and replace it.

    Idempotent, and independent of every other cell. Raises, writing nothing, when the cell has no
    part or the merge breaks an invariant.

    ``live`` is the store's live ``(shard_y, shard_x)`` shards for this zone-year, from
    :func:`~.published_store.live_shards`. Returns where the master's embedded tiles disagree with
    it: no run removes a tile it did not write, so a tile whose shard something else cleared — an
    operator clearing a year by hand — would otherwise stay in the master unnoticed. The master is
    written either way, because it describes the registry faithfully; the disagreement is the
    caller's to report. Empty means they agree, or that nothing was checked.
    """
    parts = read_cell(fs, registry_root, "parts", zone, year, dataset_schema())
    if not parts.num_rows:
        raise ValueError(f"{zone}/{year} has no registry part to build a master from")
    master = merge(parts, read_cell(fs, registry_root, "rebuild", zone, year, rebuild_dataset_schema()))
    if failures := invariant_failures(master, expected_rows=tile_years(parts)):
        raise ValueError(f"{zone}/{year} master breaks its invariants: {'; '.join(failures)}")
    write_cell(fs, registry_root, zone, year, master)
    if live is None:
        return []
    tiles, embedded = master.column("tile").to_pylist(), master.column("embedded").to_pylist()
    held = {tile for tile, is_embedded in zip(tiles, embedded, strict=True) if is_embedded}
    shards = {f"chunk_{shard_y}_{shard_x}" for shard_y, shard_x in live}
    findings = []
    if gone := sorted(held - shards):
        findings.append(f"{len(gone)} tile(s) the master calls embedded hold no shard in the store: {gone[:5]}")
    if extra := sorted(shards - held):
        findings.append(f"{len(extra)} live shard(s) the master does not call embedded: {extra[:5]}")
    return findings
