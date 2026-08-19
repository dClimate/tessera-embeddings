"""The published registry: one Parquet row per live tile per year, written as each cell lands.

**What it is for.** A consumer asking "is my area covered, and how well" should not have to open a
petabyte-scale store to find out, and a later infill campaign asking "which tiles were skipped, and
would more imagery fix them" should not have to re-read the pixels. The registry answers both from a
Parquet dataset beside the store: one row per 2048-pixel tile per year.

**"Covered" is two columns, not one.** ``embedded`` says whether a tile holds embeddings at all;
``refused_px`` says how much of its land the depth rule removed. A tile can be embedded and still be
largely holes, and those partial refusals are the bulk of what a revisit campaign would fill — so a
reader treating ``embedded`` alone as coverage will overstate it. Both kinds of row are measured in
the same terms and are directly comparable.

**A convenience layer, never a second source of truth.** Every column is derivable from the store
itself, so a wrong row is a rebuildable inconvenience rather than a data defect, and a correction to
it never touches published data. That property is what the Open Data access request promises
Cambridge, and it is why a later compaction may replace the whole dataset.

**One part per cell, keyed by run.** Parts land under ``parts/zone=<Z>/year=<Y>/<run_id>.parquet``.
Keying by run rather than a fixed name means a refill writes a NEW part instead of overwriting the
original fill's: the registry then holds both, which is the honest record of a cell filled twice, and
dedup becomes the compaction step's decision rather than a silent overwrite's. ``assembled_at`` is
what makes that decision possible — latest-wins needs a clock, and a run id is not one.

Three things this module does deliberately, each because the obvious version is broken:

* **The schema is DECLARED, never inferred.** A cell that refused nothing leaves every refusal
  column null, and ``pa.Table.from_pylist`` then types those columns ``null`` — which fails to
  concatenate against a cell that did refuse something (``ArrowInvalid: Schema at index 1 was
  different``). Since most cells refuse nothing, an inferred schema would have broken the compaction
  and any read across the whole dataset, while every individual part looked fine.
* **``zone`` and ``year`` are NOT columns.** They are the hive partition keys, and carrying them as
  columns too makes the dataset unreadable: pyarrow infers ``year=2021`` from the path as ``int32``
  and refuses to merge it with an ``int64`` column of the same name
  (``ArrowTypeError: Field year has incompatible types``). The path is the authority; the same
  identity is repeated in the file's key-value metadata, which collides with nothing, so a part
  opened on its own still says what it describes.
* **Null means "not measured", never zero.** A tile whose coverage record did not survive — a
  resumed success carries none, an unreadable marker leaves none — is null across every
  measurement. Writing zero would assert a measurement nobody took, which is the mistake this
  registry exists to stop a reader making. Zero refusals is a real, different answer, and it is
  written as zero.

**Where the bounding box comes from.** Each row carries its tile's WGS84 box, which is what makes
"is my area of interest covered" a query rather than a grid calculation the consumer has to get
right. It is NOT derived here: the arithmetic needs the northing sign convention, and getting that
wrong yields bounds that are plausible everywhere and correct nowhere. The caller supplies boxes from
:func:`~tessera_embeddings.storage.zone_grid.tile_range_bbox_wgs84`, which is the single
implementation of that convention and whose densified perimeter is measured to contain the true
envelope to within one pixel — the ingest's catalogue preflight relies on the same guarantee.

**Two things a consumer must know about the box.** It is in WGS84 degrees, so there is no CRS column
to check. And rows for zones 01 and 60 straddle the antimeridian, where the GeoJSON and STAC
convention makes ``west > east``: filtering ``west <= lon <= east`` silently drops them, and the
correct test for such a row is ``lon >= west or lon <= east``.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)

#: Refusal reasons, mirrored from the assembly summary so a row's columns and the year's pooled
#: totals name the same three things. A reason in one and not the other would have the registry and
#: the store's own provenance disagree about what happened.
REASONS = ("no_optical", "thin", "no_radar")


def part_uri(registry_root: str, zone: str, year: int, run_id: str) -> str:
    """Where one cell's part lands inside the registry dataset.

    The dataset's LAYOUT lives here rather than on ``BucketPaths`` because it is a property of the
    dataset, not of the bucket: ``BucketPaths.optical_registry`` owns where the registry is (in
    production, somebody else's bucket), while the partitioning and part naming are this module's.
    It also has to be here: the run id is not known until the runner derives it from the cell's
    inputs, long after the flow resolved its paths.
    """
    return f"{registry_root.rstrip('/')}/parts/zone={zone}/year={year}/{run_id}.parquet"


def registry_schema() -> pa.Schema:
    """The part schema, declared so every part is mergeable with every other.

    Every measurement column is nullable, and that is load-bearing rather than lax: null is how a row
    says "nothing measured this", which an embedded tile's refusal columns and an unrecorded refusal
    both need to say. Integers are ``int64`` throughout — a 2048-px tile's counts fit in far less, but
    a uniform width means no part can disagree with another about a column's type.
    """
    import pyarrow as pa

    return pa.schema(
        [
            # Identity. `zone`/`year` are the partition keys and deliberately absent — see the module
            # docstring for the type collision that carrying them as columns causes.
            pa.field("tile", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("assembled_at", pa.string(), nullable=False),
            # Whether this tile holds embeddings AT ALL — not whether it is whole. A tile the depth
            # gate partly refused is `True` here and carries a non-zero `refused_px`, so the
            # question "is my area covered" is answered by BOTH columns: `embedded` says whether
            # anything is there, `refused_px` says how much is missing from it.
            pa.field("embedded", pa.bool_(), nullable=False),
            # The footprint the reasons were counted over, and the tile's whole footprint. They
            # differ when the read plan cropped the chunk, and the difference is land nobody
            # evaluated — which an infill planner needs to tell from land evaluated and refused.
            pa.field("eligible_px", pa.int64()),
            pa.field("chunk_px", pa.int64()),
            pa.field("refused_px", pa.int64()),
            *[pa.field(f"refused_{reason}_px", pa.int64()) for reason in REASONS],
            # How close it came, which is what an infill campaign actually asks. `obs_max` is the
            # best case in the tile — 14 against a cutoff of 15 means one usable scene may flip it —
            # and the median says whether that best case is representative or a lone bright patch.
            pa.field("px_with_any_optical", pa.int64()),
            pa.field("obs_max", pa.int64()),
            pa.field("median_obs_where_any", pa.float64()),
            # Whether more optical could help at all: a tile that is thin AND radar-free needs both.
            pa.field("px_with_any_radar", pa.int64()),
            # Whether the radar rule was in force. Without it a reader cannot tell
            # `refused_no_radar_px = 0` ("nothing refused for radar") from the rule being switched
            # off, which is what a global campaign does by policy.
            pa.field("radar_rule_enforced", pa.bool_()),
            # THE OPTICAL LINE THIS CELL WAS FILLED UNDER, for the same reason the radar rule is
            # here: `obs_max` and `median_obs_where_any` are distances from a threshold, and without
            # the threshold they are unreadable — 14 is one scene short of the line at 15 and nowhere
            # near it at 30. The store's root carries the value, but a consumer reading the registry
            # has been told it need not open the store, so a row that cannot be interpreted on its
            # own defeats the point. Cell-level policy, so it is set on EVERY row including embedded
            # ones; null means the fill applied no depth rule at all, which is not the same as 0.
            pa.field("optical_min_obs", pa.int64()),
            # WHERE THE TILE IS, in WGS84 degrees, so "is my area of interest covered" is a
            # comparison against the registry rather than a grid calculation the consumer has to
            # get right. Four columns rather than a struct: every Parquet reader filters a float
            # column without unpacking anything, and this dataset's whole point is being easy to
            # query. Always WGS84, so no CRS column — a projected box would need one per zone.
            #
            # ANTIMERIDIAN: zones 01 and 60 straddle +/-180, and those rows follow the GeoJSON and
            # STAC convention of west > east. A consumer filtering `west <= lon <= east` silently
            # drops them; the correct test for a crossing row is `lon >= west or lon <= east`.
            pa.field("bbox_west", pa.float64()),
            pa.field("bbox_south", pa.float64()),
            pa.field("bbox_east", pa.float64()),
            pa.field("bbox_north", pa.float64()),
        ]
    )


def _blank_measurements() -> dict[str, Any]:
    """Every measurement column, null — the row for a tile nothing measured refusals for."""
    return {
        "eligible_px": None,
        "chunk_px": None,
        "refused_px": None,
        **{f"refused_{reason}_px": None for reason in REASONS},
        "px_with_any_optical": None,
        "obs_max": None,
        "median_obs_where_any": None,
        "px_with_any_radar": None,
        "radar_rule_enforced": None,
        "optical_min_obs": None,
        "bbox_west": None,
        "bbox_south": None,
        "bbox_east": None,
        "bbox_north": None,
    }


def registry_rows(
    run_id: str,
    assembled_at: str,
    *,
    embedded: Iterable[str],
    refused: Iterable[str],
    records: Mapping[str, dict] | None = None,
    embedded_records: Mapping[str, dict] | None = None,
    optical_min_obs: int | None = None,
    bboxes: Mapping[str, tuple[float, float, float, float]] | None = None,
) -> list[dict[str, Any]]:
    """One row per live tile: what it holds, and for a refused tile why, and how close it came.

    ``embedded`` and ``refused`` partition the cell's live tiles — the same two sets the year's
    provenance summary is built from, so the registry cannot disagree with the store about which tiles
    were written. A refused tile with no record still gets a row, measurements null: "no reason was
    recorded" and "nothing was refused" are different facts, and a zero asserts the second.

    ``optical_min_obs`` is the cell's depth rule, stamped on every row — including embedded ones,
    which carry no measurements but were still filled under it. It is cell-level policy rather than a
    per-tile measurement, so it is passed in rather than read from a record: an all-refused cell and a
    fully-embedded one must both be able to state it.

    ``bboxes`` maps a tile label to its WGS84 ``(west, south, east, north)``. Passed in rather than
    derived here because the box comes from the zone's grid geometry, which this module has no
    business knowing; a label with no entry gets nulls rather than a guess.
    """
    records = {**(embedded_records or {}), **(records or {})}
    rule = _int_or_none(optical_min_obs)
    boxes = bboxes or {}
    rows: list[dict[str, Any]] = []
    for label, was_embedded in [*((lbl, True) for lbl in sorted(embedded)), *((lbl, False) for lbl in sorted(refused))]:
        row: dict[str, Any] = {
            "tile": label,
            "run_id": run_id,
            "assembled_at": assembled_at,
            "embedded": was_embedded,
        }
        row |= _blank_measurements()
        row["optical_min_obs"] = rule
        row |= _bbox_fields(boxes.get(label))
        row |= _measurement_fields(records.get(label) or {})
        rows.append(row)
    return rows


def _measurement_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """One shard's measurements from its coverage record; nothing at all without one.

    Applied to embedded and refused rows alike, because the numbers mean the same thing for both:
    a shard the depth gate partly refused reports what it lost, which is the majority of an infill
    work list, and reporting it only for shards that lost EVERYTHING described the partly-holed ones
    as covered.
    """
    if not record:
        return {}
    reasons = record.get("refused") or {}
    obs = record.get("s2_obs") or {}
    counted = {reason: int(reasons.get(reason) or 0) for reason in REASONS}
    return {
        "eligible_px": _int_or_none(record.get("eligible_px")),
        "chunk_px": _int_or_none(record.get("chunk_px")),
        "refused_px": sum(counted.values()),
        **{f"refused_{reason}_px": n for reason, n in counted.items()},
        "px_with_any_optical": _int_or_none(obs.get("px_with_any")),
        "obs_max": _int_or_none(obs.get("max")),
        "median_obs_where_any": _float_or_none(obs.get("median_where_any")),
        "px_with_any_radar": _int_or_none(record.get("px_with_any_radar")),
        "radar_rule_enforced": (
            bool(record["radar_rule_enforced"]) if isinstance(record.get("radar_rule_enforced"), bool) else None
        ),
    }


def _bbox_fields(box: tuple[float, float, float, float] | None) -> dict[str, float | None]:
    """The four bbox columns for one tile, all null when no box was supplied."""
    if box is None:
        return {"bbox_west": None, "bbox_south": None, "bbox_east": None, "bbox_north": None}
    west, south, east, north = box
    return {
        "bbox_west": float(west),
        "bbox_south": float(south),
        "bbox_east": float(east),
        "bbox_north": float(north),
    }


def _int_or_none(value: object) -> int | None:
    """An int, or None for anything that is not a number — including a bool, which is not a count."""
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def write_registry_part(
    uri: str,
    rows: list[dict[str, Any]],
    *,
    open_output: Callable[[str], AbstractContextManager[Any]],
    zone: str = "",
    year: int = 0,
    extra_metadata: Mapping[str, str] | None = None,
) -> int:
    """Write ``rows`` as one Parquet part at ``uri``; returns the rows written.

    ``open_output`` returns a binary writable for the URI — the caller's filesystem, so nothing here
    knows whether the registry is local or in somebody else's bucket.

    ``zone``/``year`` go into the file's key-value metadata rather than a column, so a part opened on
    its own still says what it describes while the dataset's partition keys stay the single authority.
    ``extra_metadata`` adds to that key-value block — for provenance that describes the whole part
    rather than any one tile, and that a reader may want without decoding a row.

    Buffered and written in one shot rather than streamed: a part is at most a few hundred rows, and a
    partial Parquet file is unreadable, so one write is both cheap and the only version that fails
    cleanly.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return 0
    table = pa.Table.from_pylist(rows, schema=registry_schema())
    table = table.replace_schema_metadata(
        {"zone": zone, "year": str(year), "run_id": str(rows[0]["run_id"]), **dict(extra_metadata or {})}
    )
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    with open_output(uri) as f:
        f.write(buf.getvalue())
    return table.num_rows
