"""The compacted registry master: one row per tile-year, one schema, provenance on every value.

**What this is for.** :mod:`~tessera_embeddings.storage.registry` writes one part per cell per run,
which is the right shape for a campaign writing cells concurrently and the wrong shape for anybody
reading the result. Two consequences the design always anticipated and never closed:

* **A cell filled twice has two parts**, and nothing deduplicates them. Dedup was deferred to "the
  compaction step's decision" — this is that step.
* **``pyarrow`` infers a dataset's schema from the first file in sorted path order**, so a column
  added mid-campaign is silently dropped if zone ``01N`` was written before the change. The
  documented workaround is for every consumer to state the schema. A compacted master written with
  one schema, and a dataset-level ``_common_metadata`` beside it, removes the hazard instead of
  asking every reader to remember it.

**Three prefixes, and only one of them is for reading.**

```
  parts/     the campaign's own output, append-only, NEVER rewritten. What a consumer reading
             today sees is unchanged by anything here: same rows, same count, same schema.
  rebuild/   measurements re-derived from the store for rows a resumed fill left null
             (:mod:`.registry_rebuild`). Additive; rollback is deleting this prefix.
  master/    the compacted view: one row per (zone, year, tile), one schema, and two columns
             saying where each row's numbers came from. This is what consumers read.
```

**A refill supersedes a cell wholesale.** A part is one complete run of one cell, so only the fill
rows of each cell's most recently assembled run are candidates, and only rebuild rows re-derived
against that run (their ``filled_at`` is its stamp). Choosing per tile across runs would build a
union no run produced and keep an older run's numbers for a tile the store now holds from a newer
one — the rule ``published_registry_census.py`` applies too.

**Within that, precedence is not latest-wins, and the difference matters.** A fill measured pixels
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
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from tessera_embeddings.storage.registry import dataset_schema

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Value of ``measured_by`` for a row whose numbers the fill itself recorded.
MEASURED_BY_FILL = "fill"

#: Value of ``measured_by`` for a row whose numbers were re-derived from the store afterwards.
MEASURED_BY_REBUILD = "rebuild"

#: The column that decides whether a row carries measurements at all. Every measurement column is
#: null together — a row is measured or it is not — so one column answers it, and this is the one
#: a rebuild can always populate.
MEASURED_SENTINEL = "chunk_px"


def master_schema() -> pa.Schema:
    """The compacted dataset's schema: the registry's, plus where each row's numbers came from.

    ``measured_by`` is ``"fill"``, ``"rebuild"`` or null, never inferred from the presence of a
    value. ``filled_at`` is when the CELL was filled, which a rebuilt row would otherwise lose to
    its own ``assembled_at``.
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
    """One row per ``(zone, year, tile)``, resolved by the precedence in the module docstring.

    Pure, and over whole tables rather than a stream: the published registry is 3.2 million rows
    and 142 MB, so the compaction fits in memory comfortably and a one-shot merge is both simpler
    and the only version that fails cleanly.
    """
    schema = master_schema()
    fill = _aligned(newest_fill_runs(parts), schema)
    # A fill part's own stamp IS when its cell was filled, so every master row answers that question.
    index = schema.get_field_index("filled_at")
    fill = fill.set_column(index, "filled_at", pc.coalesce(fill.column(index), fill.column("assembled_at")))
    tables = [_tagged(fill, MEASURED_BY_FILL)]
    if rebuilt is not None and rebuilt.num_rows:
        # Only re-derivations of the run that stands: one made before a refill describes a run the
        # store no longer holds.
        current = pc.is_in(_cell_key(rebuilt, "filled_at"), value_set=_cell_key(fill, "assembled_at").unique())
        tables.append(_tagged(_aligned(rebuilt.filter(current), schema), MEASURED_BY_REBUILD))
    combined = pa.concat_tables(tables)
    if not combined.num_rows:
        return combined
    keep = _winning_rows(combined)
    return combined.take(pa.array(keep)).sort_by([("zone", "ascending"), ("year", "ascending"), ("tile", "ascending")])


def newest_fill_runs(parts: pa.Table) -> pa.Table:
    """Only the rows of each cell's most recently assembled fill run.

    ``assembled_at`` is the clock and ``run_id`` breaks a tie, as in the census's ``_newest_run``.
    Every caller that compares against the fill — the merge, :func:`fill_rows_unchanged`, the
    expected row count — goes through this, so all three agree on which run stands.
    """
    if not parts.num_rows:
        return parts
    runs = parts.group_by(["zone", "year", "run_id"]).aggregate([("assembled_at", "min"), ("assembled_at", "max")])
    newest: dict[tuple[str, int], tuple[datetime.datetime, str]] = {}
    for run in runs.to_pylist():
        # min and max both parse, so every stamp in the run does: ISO-8601 sorts as text only within a format.
        _aware(run["assembled_at_min"])
        cell, candidate = (run["zone"], run["year"]), (_aware(run["assembled_at_max"]), run["run_id"] or "")
        newest[cell] = max(newest.get(cell, candidate), candidate)
    winners = pa.array([f"{zone}/{year}/{run_id}" for (zone, year), (_, run_id) in newest.items()])
    return parts.filter(pc.is_in(_cell_key(parts, "run_id"), value_set=winners))


def _aware(stamp: str | None) -> datetime.datetime:
    """``stamp`` as an aware datetime; raises rather than letting an unorderable run be chosen."""
    try:
        parsed = datetime.datetime.fromisoformat(stamp or "")
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise ValueError(f"assembled_at {stamp!r} is not an aware ISO-8601 timestamp, so its run cannot be ordered")
    return parsed


def _cell_key(table: pa.Table, column: str) -> pa.Array:
    """``zone/year/<column>`` per row, for matching rows to a cell's run or stamp."""
    year = pc.cast(table.column("year"), pa.string())
    return pc.binary_join_element_wise(table.column("zone"), year, table.column(column), "/").combine_chunks()


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
    """What is wrong with a compacted master, as a list of sentences. Empty means nothing is.

    Checked rather than assumed, because every one of these has a silent failure mode: a duplicate
    tile-year reads as coverage counted twice, a lost one as ground nobody wrote, and a rebuilt
    value sitting where a fill measured one would mean the compaction overrode a measurement with
    a re-derivation.
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

    The compaction is only safe if it never alters an original measurement, and "never" is a claim
    worth testing against the source rather than reasoning about from the merge rule. The source is
    each cell's newest run, the same one :func:`merge` keeps, so a refilled cell is compared against
    the run that stands rather than whichever part a dataset scan happened to read last.
    """
    wanted = set(columns)
    by_key: dict[str, dict[str, Any]] = {}
    for row in newest_fill_runs(parts).to_pylist():
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
