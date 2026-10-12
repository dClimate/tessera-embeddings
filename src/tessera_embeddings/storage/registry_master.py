"""The registry master: the one registry consumers read, one current row per tile-year.

**What this is for.** :mod:`~tessera_embeddings.storage.registry` writes one part per cell per run,
which is the right shape for a campaign writing cells concurrently and the wrong shape for anybody
reading the result. Two consequences the master closes:

* **A cell filled more than once has more than one part**, and something has to decide which row
  describes each tile now.
* **``pyarrow`` infers a dataset's schema from the first file in sorted path order**, so a column
  added mid-campaign is silently dropped if zone ``01N`` was written before the change. A master
  written with one schema, and a dataset-level ``_common_metadata`` beside it, removes the hazard
  instead of asking every reader to remember it.

**Two prefixes, and only one of them is for reading.**

```
  parts/     each fill's raw output, one part per cell per run, append-only, never rewritten.
             Internal: the record the master is built from and checked against.
  master/    one file per cell, master/zone=<Z>/year=<Y>/part-0.parquet: one row per tile, one
             schema, and two columns saying where each row's numbers came from. Consumers read this.
```

**It is updated in place, by every fill.** The zone-fill runner calls :func:`update_cell` as soon as
a cell's part is published. That merges the cell's parts into the cell's existing master file and
replaces the file in one PUT. Cells are independent, so concurrent fills never write the same
object, and the update is idempotent. Merging every part rather than only the newest costs a few
small reads and means a cell whose earlier update failed is healed by its next one.

**A tile's row comes from the newest run that wrote that tile.** A run writes only the tiles it
staged or refused and leaves the year's other shards as they were (``assemble_global``'s refill
caveat). So the store holds each tile as the most recently assembled run that wrote it left it,
whichever run supplied the cell's other tiles, and the master follows the store: a full refill
supersedes every tile, and a partial run such as an infill supersedes only its own and removes
none. A run's row must therefore describe the whole tile as that run leaves it, because it becomes
the tile's only row. ``filled_at`` names the run a row describes, by its fill stamp: the newest
parsed ``filled_at`` wins, so a measurement re-derived against a superseded run loses to the newer
run's row, measured or not.

**Within that run, precedence is not latest-wins, and the difference matters.** A fill measured pixels
as it wrote them; a re-derivation recomputed them afterwards from what was written. They should
agree — that is what :func:`~.registry_rebuild.compare_row` gates on — but where a fill recorded a
measurement, that measurement stands. So:

1. a fill row carrying measurements beats everything;
2. otherwise a re-derived row carrying measurements — one that measured radar ahead of an
   optical-only one, then the most recent — which is how a radar pass supersedes an optical-only
   one and why a later optical-only repair cannot undo it;
3. otherwise the most recent row of any kind, which keeps a tile that nothing has measured in the
   master as a null row rather than dropping it.

Ties after all of that fall to ``run_id``, so every merge of the same rows yields the same master.

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
from tessera_embeddings.storage.registry_rebuild import REBUILT_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator

    from pyarrow.fs import FileSystem

#: Value of ``measured_by`` for a row whose numbers the fill itself recorded.
MEASURED_BY_FILL = "fill"

#: Value of ``measured_by`` for a row whose numbers were re-derived from the store afterwards.
MEASURED_BY_REBUILD = "rebuild"

#: The column that decides whether a row carries measurements at all. Every measurement column is
#: null together — a row is measured or it is not — so one column answers it, and this is the one
#: a re-derivation can always populate.
MEASURED_SENTINEL = "chunk_px"

_KEYS = ["zone", "year", "tile"]
_ORDER = [(key, "ascending") for key in _KEYS]


def master_schema() -> pa.Schema:
    """The master's schema: the registry's, plus where each row's numbers came from.

    ``measured_by`` is ``"fill"``, ``"rebuild"`` or null, never inferred from the presence of a
    value. ``filled_at`` is the fill stamp of the run the row describes: a fill row's own
    ``assembled_at``, which on a re-derived row would otherwise be lost to the re-derivation's.
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

    A fill part has neither ``measured_by`` nor ``filled_at``. Concatenating it with master rows
    needs one shape, and a missing column must arrive as null — the honest value for a pass that did
    not record it — rather than as a default.
    """
    columns = []
    for field in schema:
        if field.name in table.column_names:
            columns.append(table.column(field.name).cast(field.type))
        else:
            columns.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(columns, schema=schema)


def as_master_rows(table: pa.Table, origin: str) -> pa.Table:
    """Registry rows — a fill part's, or a re-derivation's — as master rows from ``origin``.

    ``filled_at`` is kept where the row carries one and is the row's own ``assembled_at`` otherwise,
    which on a fill row is the same thing. A re-derived row must name the fill it re-derived: given
    its own stamp instead it would pass for a newer run and beat the fill's measurement.
    """
    rows = _aligned(table, master_schema())
    index = rows.schema.get_field_index("filled_at")
    if origin == MEASURED_BY_REBUILD and rows.column(index).null_count:
        raise ValueError("a re-derived row must carry filled_at, the stamp of the fill it re-derived")
    rows = rows.set_column(index, "filled_at", pc.coalesce(rows.column(index), rows.column("assembled_at")))
    return _tagged(rows, origin)


def merge(*candidates: pa.Table) -> pa.Table:
    """One master row per ``(zone, year, tile)`` from any mix of master rows, by the module's rules.

    Every input is in :func:`master_schema` — an existing master file, or rows passed through
    :func:`as_master_rows`. Pure, and over whole tables rather than a stream: the published registry
    is 3.2 million rows, so a whole-registry merge fits in memory and one cell's is trivial.
    """
    combined = pa.concat_tables([_aligned(table, master_schema()) for table in candidates])
    if not combined.num_rows:
        return combined
    return combined.take(pa.array(_winning_rows(combined))).sort_by(_ORDER)


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
    ``(key, newest run, rank, newest row, run_id)`` puts each group's winner first, and the first
    occurrence of each key is then a single pass. ``numpy.lexsort`` takes its keys
    last-significant-first.
    """
    zone = table.column("zone").to_pylist()
    year = table.column("year").to_pylist()
    tile = table.column("tile").to_pylist()
    key = np.array([f"{z}/{y}/{t}" for z, y, t in zip(zone, year, tile, strict=True)])

    measured = np.asarray(pc.is_valid(table.column(MEASURED_SENTINEL)).to_pylist(), dtype=bool)
    by_fill = np.asarray([value == MEASURED_BY_FILL for value in table.column("measured_by").to_pylist()], dtype=bool)
    # 0 beats 1 beats 2 beats 3. A measured fill row is authoritative; a measured re-derived row fills
    # a gap, one that measured radar ahead of an optical-only one whatever their order, so a later
    # `--skip-radar` repair cannot null a radar column a full pass already landed; an unmeasured row
    # of either kind keeps the tile present and says nothing about it.
    radar = np.asarray(pc.is_valid(table.column("px_with_any_radar")).to_pylist(), dtype=bool)
    rank = np.where(measured & by_fill, 0, np.where(measured & radar, 1, np.where(measured, 2, 3)))
    run = _descending(table.column("filled_at").to_pylist(), by_time=True)
    stamp = _descending(table.column("assembled_at").to_pylist(), by_time=True)
    tiebreak = _descending(table.column("run_id").to_pylist())

    order = np.lexsort((tiebreak, stamp, rank, run, key))
    sorted_keys = key[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return [int(index) for index in order[first]]


def _descending(values: list[Any], *, by_time: bool = False) -> np.ndarray:
    """Sort keys that put the greatest value first under ``numpy.lexsort``'s ascending sort.

    ``by_time`` orders by parsed, timezone-aware instant — text order holds only within one format
    — and raises on a stamp that will not parse rather than letting text order pick a run.
    """
    distinct = sorted(set(values), key=lambda value: (_aware(value), value)) if by_time else sorted(set(values))
    position = {value: index for index, value in enumerate(distinct)}
    return -np.array([position[value] for value in values], dtype=np.int64)


def _aware(stamp: str | None) -> datetime.datetime:
    """``stamp`` as an aware datetime; raises rather than letting an unorderable run be chosen."""
    try:
        parsed = datetime.datetime.fromisoformat(stamp or "")
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise ValueError(f"timestamp {stamp!r} is not an aware ISO-8601 timestamp, so its run cannot be ordered")
    return parsed


def tile_years(parts: pa.Table) -> int:
    """Distinct ``(zone, year, tile)`` across every run: the rows a master of ``parts`` must hold."""
    return parts.group_by(_KEYS).aggregate([]).num_rows


def invariant_failures(master: pa.Table, *, expected_rows: int | None = None) -> list[str]:
    """What is wrong with a master's labelling and shape, as sentences. Empty means nothing is.

    Checked rather than assumed, because every one of these has a silent failure mode: a duplicate
    tile-year reads as coverage counted twice, a lost one as ground nobody wrote, and a label that
    disagrees with the numbers would let a re-derivation pass as a measurement. ``expected_rows``
    is :func:`tile_years` of the cell's parts, since no run removes a tile.
    """
    failures: list[str] = []
    if expected_rows is not None and master.num_rows != expected_rows:
        failures.append(f"master holds {master.num_rows} rows, expected {expected_rows}")
    duplicates = master.num_rows - tile_years(master)
    if duplicates:
        failures.append(f"{duplicates} duplicate (zone, year, tile) rows — a tile-year must appear once")
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


def audit(master: pa.Table, parts: pa.Table) -> dict[str, list[str]]:
    """Per cell ``zone/year``, what is wrong with its master rows against its parts. Empty is sound.

    Checks the master itself, the file consumers read: every tile any part names is present and no
    other; each row describes its tile's latest run; wherever that run's fill measured the tile, the
    master carries the fill's own numbers, labelled ``"fill"``; and no row is unmeasured. A cell
    with no master file at all reports every tile missing. Pair it with :func:`invariant_failures`.
    """
    want = merge(as_master_rows(parts, MEASURED_BY_FILL))
    columns = [*_KEYS, "filled_at", "measured_by", *REBUILT_COLUMNS]
    joined = want.select(columns).join(
        master.select(columns), keys=_KEYS, join_type="full outer", left_suffix=":want", right_suffix=":held"
    )

    def differs(column: str) -> pa.ChunkedArray:
        """Null-aware inequality: one side null and the other not counts as a difference."""
        a, b = joined.column(f"{column}:want"), joined.column(f"{column}:held")
        return pc.or_(pc.xor(pc.is_null(a), pc.is_null(b)), pc.fill_null(pc.not_equal(a, b), False))

    held = pc.is_valid(joined.column("filled_at:held"))
    wanted = pc.is_valid(joined.column("filled_at:want"))
    fill_measured = pc.fill_null(pc.equal(joined.column("measured_by:want"), MEASURED_BY_FILL), False)
    altered = differs("measured_by")
    for column in REBUILT_COLUMNS:
        altered = pc.or_(altered, differs(column))
    flags = {
        "tile(s) missing": pc.invert(held),
        "tile(s) no part names": pc.invert(wanted),
        "row(s) not from the tile's latest run": pc.and_(pc.and_(held, wanted), differs("filled_at")),
        "fill measurement(s) altered": pc.and_(pc.and_(held, fill_measured), altered),
        "row(s) unmeasured": pc.and_(held, pc.is_null(joined.column(f"{MEASURED_SENTINEL}:held"))),
    }
    counts = pa.table(
        {"zone": joined.column("zone"), "year": joined.column("year")}
        | {name: pc.cast(flag, pa.int64()) for name, flag in flags.items()}
    )
    found: dict[str, list[str]] = {}
    for cell in counts.group_by(["zone", "year"]).aggregate([(name, "sum") for name in flags]).to_pylist():
        problems = [f"{cell[f'{name}_sum']} {name}" for name in flags if cell[f"{name}_sum"]]
        if problems:
            found[f"{cell['zone']}/{cell['year']}"] = problems
    return dict(sorted(found.items()))


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
    """Merge one cell's parts into its existing master file, tile by tile, and replace the file.

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
        raise ValueError(f"{zone}/{year} has no registry part to merge into the master")
    # Parts first, so on a tie between identical rows the part's copy is the one kept.
    held = read_cell(fs, registry_root, "master", zone, year, master_schema())
    master = merge(as_master_rows(parts, MEASURED_BY_FILL), held)
    if failures := invariant_failures(master, expected_rows=tile_years(parts)):
        raise ValueError(f"{zone}/{year} master breaks its invariants: {'; '.join(failures)}")
    write_cell(fs, registry_root, zone, year, master)
    if live is None:
        return []
    tiles, embedded = master.column("tile").to_pylist(), master.column("embedded").to_pylist()
    held_tiles = {tile for tile, is_embedded in zip(tiles, embedded, strict=True) if is_embedded}
    shards = {f"chunk_{shard_y}_{shard_x}" for shard_y, shard_x in live}
    findings = []
    if gone := sorted(held_tiles - shards):
        findings.append(f"{len(gone)} tile(s) the master calls embedded hold no shard in the store: {gone[:5]}")
    if extra := sorted(shards - held_tiles):
        findings.append(f"{len(extra)} live shard(s) the master does not call embedded: {extra[:5]}")
    return findings
