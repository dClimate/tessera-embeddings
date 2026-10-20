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

**It is merged in place, by every fill.** The zone-fill runner calls :func:`merge_cell` as soon as
a cell's part is published. That merges the cell's parts into the cell's existing master file and
replaces the file in one conditional PUT, so two writers of one cell — two fills, or a fill and a
repair — cannot lose each other's rows: the one that read first is refused and merges again. The
merge is idempotent, and merging every part rather than only the newest costs a few small reads
and means a cell whose earlier merge failed is healed by its next one.

**A tile's row comes from the newest run that wrote that tile.** A run writes only the tiles it
staged or refused and leaves the year's other shards as they were (``assemble_global``'s refill
caveat). So the store holds each tile as the most recently committed run that wrote it left it,
whichever run supplied the cell's other tiles, and the master follows the store: a full refill
supersedes every tile, and an update of some tiles supersedes only its own and removes none. A
run's row must therefore describe the whole tile as that run leaves it, because it becomes the
tile's only row. ``filled_at`` names the run a row describes, by its part's stamp, which is the
run's shard commit time: Icechunk refuses a commit older than its parent, so the newest parsed
``filled_at`` is the run whose pixels the store holds, and a measurement re-derived against a
superseded run loses to the newer run's row, measured or not.

**The store stands behind the parts, or the cell is not written.** Newest-part-wins is right only
if every run that wrote the year has a part. Every shard commit is tagged with its run, in the same
commit as the pixels, and Icechunk's diff of each commit says which tiles it wrote, so
:func:`store_disagreements` refuses a cell in which a run with no part is the last to have written
any tile, a part is stamped out of commit order, or a part names a run the store never committed.

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
import fcntl
import functools
import hashlib
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from tessera_embeddings.storage.registry import dataset_schema
from tessera_embeddings.storage.registry_rebuild import CARRIED_COLUMNS, REBUILT_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Sequence

    from fsspec import AbstractFileSystem

    from tessera_embeddings.storage.published_store import StoreYear

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
    other; each row describes its tile's latest run and carries that run's fill fields — ``embedded``,
    the bounding box, the depth rule, the build — whoever measured it; wherever that run's fill
    measured the tile, the master carries the fill's own row, numbers and provenance alike, labelled
    ``"fill"``; and no row is
    unmeasured. A cell with no master file at all reports every tile missing. Pair it with
    :func:`invariant_failures`.
    """
    want = merge(as_master_rows(parts, MEASURED_BY_FILL))
    carried = [column for column in CARRIED_COLUMNS if column not in _KEYS]
    columns = [*_KEYS, "filled_at", "measured_by", "run_id", "assembled_at", *carried, *REBUILT_COLUMNS]
    joined = want.select(columns).join(
        master.select(columns), keys=_KEYS, join_type="full outer", left_suffix=":want", right_suffix=":held"
    )

    def differs(*names: str) -> pa.ChunkedArray:
        """Null-aware inequality in any of ``names``: one side null and the other not is a difference."""

        def one(column: str) -> pa.ChunkedArray:
            a, b = joined.column(f"{column}:want"), joined.column(f"{column}:held")
            return pc.or_(pc.xor(pc.is_null(a), pc.is_null(b)), pc.fill_null(pc.not_equal(a, b), False))

        return functools.reduce(pc.or_, map(one, names))

    held = pc.is_valid(joined.column("filled_at:held"))
    wanted = pc.is_valid(joined.column("filled_at:want"))
    both = pc.and_(held, wanted)
    fill_measured = pc.fill_null(pc.equal(joined.column("measured_by:want"), MEASURED_BY_FILL), False)
    flags = {
        "tile(s) missing": pc.invert(held),
        "tile(s) no part names": pc.invert(wanted),
        "row(s) not from the tile's latest run": pc.and_(both, differs("filled_at")),
        "fill field(s) altered": pc.and_(both, differs(*carried)),
        "fill row(s) altered": pc.and_(
            pc.and_(held, fill_measured), differs("measured_by", "run_id", "assembled_at", *REBUILT_COLUMNS)
        ),
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


def store_disagreements(parts: pa.Table, store: StoreYear) -> list[str]:
    """Why the store cannot stand behind one cell's ``parts``, as sentences. Empty means it can.

    A tile's master row comes from the newest part naming it, so the master is right only if the
    parts account for every commit that wrote the year, in the order the store wrote them. A commit
    no part describes leaves its tiles described by an older run, and nothing in the registry can
    tell; a part stamped out of commit order picks the wrong run. The store records every commit
    that wrote the year (:func:`~.published_store.committed_runs`), so:

    * every part's run is one the store records — a withdrawn run's part must be deleted;
    * every part is stamped with one of its run's shard commits, the one it describes;
    * no commit that no part describes is the last to have written any tile, as the store's own diff
      of it says (:func:`~.published_store.commit_tiles`). That covers a run with no part and also a
      retry: a part describes one commit of its run, so an earlier attempt's commit counts on its
      own. One that later commits superseded on every tile it wrote describes nothing, so it does
      not block, and neither does an untagged run that marked the year empty, which left no
      embeddings behind.
    """
    stamps: dict[str, set[datetime.datetime]] = {}
    tiles: dict[str, set[str]] = {}
    for row in parts.select(["run_id", "assembled_at", "tile"]).to_pylist():
        stamps.setdefault(row["run_id"], set()).add(_aware(row["assembled_at"]))
        tiles.setdefault(row["run_id"], set()).add(row["tile"])
    runs = {commit.run_id for commit in store.commits}
    found = [
        f"run {run!r} has a part, but the store records no commit by it; a withdrawn run's part must be deleted"
        for run in sorted(set(tiles) - runs)
    ]
    # The one commit each part describes: the one it is stamped with, or a run's only, untagged one.
    described = {
        run: next(
            (c for c in store.commits if c.run_id == run and (c.snapshot is None or _aware(c.at) in stamps[run])), None
        )
        for run in sorted(set(tiles) & runs)
    }
    found += [
        f"run {run!r} has a part stamped at none of its commits, so it would be misordered"
        for run, commit in described.items()
        if commit is None
    ]
    later: set[str] = set()
    for commit in reversed(store.commits):
        if described.get(commit.run_id) == commit:
            written = tiles[commit.run_id]
        elif commit.snapshot is None:
            if not commit.empty:
                found.append(
                    f"run {commit.run_id!r} wrote the year with no part, and no recorded commit bounds its tiles"
                )
            continue
        else:
            written = set(store.tiles_of(commit.snapshot))
            if own := sorted(written - later):
                found.append(
                    f"run {commit.run_id!r} committed at {commit.at} and last wrote {len(own)} tile(s) no part "
                    f"describes, so the master would describe them by an older run: {own[:5]}"
                )
        later |= written
    return found


def _bare(registry_root: str) -> str:
    """``registry_root`` as a filesystem path: no scheme, so ``file://`` is a real local path, no trailing slash."""
    return str(fsspec.core.strip_protocol(registry_root)).rstrip("/")


def _master_path(registry_root: str, zone: str, year: int) -> str:
    return f"{_bare(registry_root)}/master/zone={zone}/year={year}/part-0.parquet"


def read_cell(
    fs: AbstractFileSystem, registry_root: str, prefix: str, zone: str, year: int, schema: pa.Schema
) -> pa.Table:
    """One cell's rows under ``prefix``, partition keys included; empty when the cell has none there.

    The listing is dropped first, because a filesystem instance is shared within a process and its
    cached listing would hide a part another process published since.
    """
    base = f"{_bare(registry_root)}/{prefix}"
    path = f"{base}/zone={zone}/year={year}"
    fs.invalidate_cache(path)
    if not fs.exists(path):
        return schema.empty_table()
    return ds.dataset(path, filesystem=fs, schema=schema, partitioning="hive", partition_base_dir=base).to_table()


def cell_version(fs: AbstractFileSystem, registry_root: str, zone: str, year: int) -> str | None:
    """What names the version of the cell's master file now: S3's ETag, or a local file's hash.

    ``None`` when the cell has no master file. Read BEFORE anything the new file is built from, so
    a write landing after any of those reads changes it and :func:`write_cell` refuses.
    """
    return _version(fs, _master_path(registry_root, zone, year))


def _version(fs: AbstractFileSystem, path: str) -> str | None:
    """The version of the object at ``path`` now, as :func:`_put_if` conditions on it; ``None`` if absent."""
    fs.invalidate_cache(path)
    try:
        return str(fs.info(path)["ETag"]) if _on_s3(fs) else hashlib.sha256(fs.cat_file(path)).hexdigest()
    except FileNotFoundError:
        return None


def write_cell(
    fs: AbstractFileSystem, registry_root: str, zone: str, year: int, rows: pa.Table, *, version: str | None
) -> bool:
    """Replace one cell's master file with ``rows`` if it is still at ``version``; whether it was.

    ``version`` is :func:`cell_version` as read before ``rows`` were built, ``None`` for a cell with
    no file yet. On S3 it is a conditional PUT, so of two writers that read the same version only
    the first lands, and the other is told rather than silently discarding the first's rows. A local
    registry, for rehearsals, checks the same under a lock. Either way a reader sees the old file or
    the new one, never a mixture. Then ``_common_metadata`` is widened if it needs to be
    (:func:`_widen_dataset_schema`).
    """
    path = _master_path(registry_root, zone, year)
    buf = io.BytesIO()
    # The partition keys live in the path, as in every hive dataset.
    pq.write_table(rows.drop_columns(["zone", "year"]), buf, compression="zstd")
    written = _put_if(fs, path, buf.getvalue(), version)
    if written:
        _widen_dataset_schema(fs, registry_root)
    return written


def _widen_dataset_schema(fs: AbstractFileSystem, registry_root: str, tries: int = 8) -> None:
    """Write ``_common_metadata`` if it is absent or lacks a column of :func:`master_schema`.

    The same bytes for every cell, so it is written only then rather than raced by every fill, and
    never over a schema that has a column this one lacks: a worker still on an older release must
    not take a newer column away from readers mid-deploy. Conditional on the copy checked, like a
    cell's write, so of two workers that both found it stale, the second is refused and checks again.
    """
    meta = f"{_bare(registry_root)}/master/_common_metadata"
    for _ in range(tries):
        version = _version(fs, meta)
        if version is not None and _schema_is_current(fs, meta):
            return
        buf = io.BytesIO()
        pq.ParquetWriter(buf, master_schema()).close()
        if _put_if(fs, meta, buf.getvalue(), version):
            return


def _schema_is_current(fs: AbstractFileSystem, meta: str) -> bool:
    """Whether ``_common_metadata`` at ``meta`` exists and declares every column of :func:`master_schema`."""
    try:
        with fs.open(meta, "rb") as f:
            declared = set(pq.read_schema(f).remove_metadata())
    except FileNotFoundError:
        return False
    return set(master_schema().remove_metadata()) <= declared


def _on_s3(fs: AbstractFileSystem) -> bool:
    return hasattr(fs, "call_s3")


def _put_if(fs: AbstractFileSystem, path: str, data: bytes, version: str | None) -> bool:
    """Write ``data`` at ``path`` only if the object there is still at ``version``."""
    if not _on_s3(fs) and "file" not in fs.protocol:
        raise ValueError(f"no conditional write for a {fs.protocol} filesystem; the master needs S3 or local")
    if _on_s3(fs):
        bucket, key, _ = fs.split_path(path)
        condition = {"IfMatch": version} if version else {"IfNoneMatch": "*"}
        try:
            fs.call_s3("put_object", Bucket=bucket, Key=key, Body=data, **condition)
        except OSError as error:
            code = getattr(error.__cause__, "response", {}).get("Error", {}).get("Code")
            if isinstance(error, FileExistsError) or code in ("PreconditionFailed", "ConditionalRequestConflict"):
                return False
            raise
        finally:
            fs.invalidate_cache(path)
        return True
    folder = Path(path).parent
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None
        if current != version:
            return False
        # Dot-prefixed, so a dataset read of master/ never takes it for a part.
        staged = folder / f".{Path(path).name}.tmp"
        staged.write_bytes(data)
        staged.replace(path)
    return True


def still_published(held: pa.Table, fills: pa.Table) -> pa.Table:
    """The ``held`` master rows whose run still has a part among ``fills``' rows for that tile.

    A withdrawn run — its part deleted — must not outlive its part in the master: its rows carry the
    newest ``filled_at`` and would otherwise beat every run that remains.
    """
    return held.filter(pc.is_in(_key(held, "tile", "filled_at"), value_set=_key(fills, "tile", "filled_at")))


def _key(table: pa.Table, *columns: str) -> pa.Array:
    """``zone/year/<columns>`` per row, for matching rows across tables."""
    year = pc.cast(table.column("year"), pa.string())
    fields = [table.column("zone"), year, *(table.column(column) for column in columns)]
    return pc.binary_join_element_wise(*fields, "/").combine_chunks()


def merge_cell(
    fs: AbstractFileSystem,
    registry_root: str,
    zone: str,
    year: int,
    *,
    run_id: str | None = None,
    live: Collection[tuple[int, int]] | None = None,
    store: Callable[[], StoreYear] | None = None,
    extra: Sequence[pa.Table] = (),
    write: bool = True,
    tries: int = 8,
) -> list[str]:
    """Merge one cell's parts into its existing master file, tile by tile, and replace the file.

    The one writer of a cell's master, used by every fill and by repair. Idempotent, and independent
    of every other cell. Raises, writing nothing, when the cell has no part for ``run_id`` — the run
    the caller just landed, whose part publishing is best-effort and may have failed — the store
    cannot stand behind the parts (:func:`store_disagreements`, when ``store`` is given), or the
    merge breaks an invariant. ``extra`` is more master rows to merge, a repair's re-derived ones;
    ``write=False`` checks everything and writes nothing. A cell whose last part was deleted, its
    run withdrawn, has its master file emptied; one with neither a part nor a file is left alone.

    **Two writers of one cell cannot lose each other's rows.** The file's version is read first and
    the write is conditional on it (:func:`write_cell`), so a writer that read before another
    landed is refused and merges again from what is there now, up to ``tries`` times.

    ``store`` reads the store's record of the year, and is called after the parts on every attempt:
    a run publishes its part only after its commit, so a record read then names the run of every
    part read, and a fill landing mid-merge is merged rather than refused.

    ``live`` is the store's live ``(shard_y, shard_x)`` shards for this zone-year, from
    :func:`~.published_store.live_shards`. Returns where the master's embedded tiles disagree with
    it: no run removes a tile it did not write, so a tile whose shard something else cleared — an
    operator clearing a year by hand — would otherwise stay in the master unnoticed. The master is
    written either way, because it describes the registry faithfully; the disagreement is the
    caller's to report. Empty means they agree, or that nothing was checked.
    """
    for _ in range(tries):
        version = cell_version(fs, registry_root, zone, year)
        parts = read_cell(fs, registry_root, "parts", zone, year, dataset_schema())
        if run_id is not None and run_id not in set(parts.column("run_id").to_pylist()):
            raise ValueError(f"{zone}/{year} has no registry part for run {run_id!r}, so the master would miss it")
        if store is not None and (found := store_disagreements(parts, store())):
            raise ValueError(f"{zone}/{year}: the store cannot stand behind the registry parts: {'; '.join(found)}")
        fills = as_master_rows(parts, MEASURED_BY_FILL)
        held = still_published(read_cell(fs, registry_root, "master", zone, year, master_schema()), fills)
        # A held fill row is a copy of a part's, so the part's own copy is merged: only the rows the
        # parts cannot supply, re-derived ones, come from the existing file, and a damaged copy cannot win.
        held = held.filter(pc.invert(pc.fill_null(pc.equal(held.column("measured_by"), MEASURED_BY_FILL), False)))
        master = merge(fills, held, *extra)
        if failures := invariant_failures(master, expected_rows=tile_years(parts)):
            raise ValueError(f"{zone}/{year} master breaks its invariants: {'; '.join(failures)}")
        if not write or (version is None and not master.num_rows):
            return _live_disagreements(master, live)
        if write_cell(fs, registry_root, zone, year, master, version=version):
            return _live_disagreements(master, live)
    raise ValueError(f"{zone}/{year} master changed under each of {tries} attempts to merge it")


def _live_disagreements(master: pa.Table, live: Collection[tuple[int, int]] | None) -> list[str]:
    """Where the master's embedded tiles and the store's ``live`` shards disagree; empty if unchecked."""
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
