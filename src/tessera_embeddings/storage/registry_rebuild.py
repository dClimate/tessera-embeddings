"""Recomputing the registry measurements a resumed fill never recorded.

**Why any of this is possible.** :mod:`~tessera_embeddings.storage.registry` promises that every
column is derivable from the store, which is what makes a missing measurement a rebuildable
inconvenience rather than lost data. This module is that promise cashed in: it recomputes a tile's
refusal counts and depth statistics from the ``*_obs_count`` arrays the store already publishes.

**Why measurements are missing at all.** A refused tile's coverage record is written into its skip
marker on object storage and survives every resume. An embedded tile's record rides back in the
actor's result and lives only in memory, so the leg that finishes a resumed run reports earlier
legs' tiles as synthetic successes carrying no record, and those rows publish null. The asymmetry
is the whole cause: every wholly-refused tile in the published registry is measured, and only about
half the embedded ones are.

**The rebuild is a re-derivation, not an estimate — and that is checkable.** The fill's optical test
is ``has_optical = s2_nonzero & (s2_valid_count > 0)`` (:mod:`..inference.dataset`), where
``s2_nonzero`` tests reflectance bands the store does not keep. If that term had ever removed a
pixel the split could not be reproduced from the store. Across all 1,728,365 measured rows of the
published registry, ``refused_no_optical_px == eligible_px - px_with_any_optical`` holds exactly,
with zero pixels of discrepancy — so on the delivered product ``has_optical`` *is*
``s2_obs_count > 0``, and every refusal column reduces to counting one array. :func:`compare_row`
is how that stays true rather than remembered: it re-derives rows that already carry measurements
and demands they match.

**What this deliberately does not recover.** ``eligible_px`` is the footprint the fill's reasons
were counted over, and it shrinks when the read plan cropped a chunk in x — to the columns holding
any S2 observation, when that box is at most 90% of the width (``read_plan._chunk_read_plan``). That
box is reproducible from the stored ``s2_obs_count``: it reproduces ``eligible_px`` on all 237
cropped measured rows. Rebuilt rows nonetheless leave it **null** — the registry's own word for "not
measured" — and count over the whole tile, by decision; applying the crop here is the open
alternative. The gap is small and bounded: 237 of 1,728,365 measured rows (0.014%) have
``eligible_px < chunk_px``. :func:`compare_row` reports such tiles as ``cropped`` rather than as
mismatches, because a rebuild of one is expected to differ.
"""

from __future__ import annotations

import datetime
import io
import numbers
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage.registry import REASONS, registry_schema

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager

    import zarr

#: The array a rebuild reads to answer every optical column.
OPTICAL_OBS_VAR = "s2_obs_count"

#: The two arrays ``px_with_any_radar`` needs. Reading them TRIPLES the bytes moved for one
#: informational column, which is why the rebuild can be run without them and re-run with them.
RADAR_OBS_VARS = ("s1_asc_obs_count", "s1_desc_obs_count")

#: Columns this module re-derives. Everything else on a registry row — identity, bbox, the depth
#: rule, the build that produced the cell — is already non-null on every published row, including
#: the ones with no measurements, so a rebuild copies it forward rather than recomputing it.
REBUILT_COLUMNS: tuple[str, ...] = (
    "eligible_px",
    "chunk_px",
    "refused_px",
    *[f"refused_{reason}_px" for reason in REASONS],
    "px_with_any_optical",
    "obs_max",
    "median_obs_where_any",
    "median_obs_where_thin",
    "px_with_any_radar",
    "radar_rule_enforced",
)

#: Carried from the row being rebuilt rather than recomputed. ``code_version``/``code_commit``
#: name the build that REFUSED the tile, which a rebuild does not become, and the bbox comes from
#: zone grid geometry this module has no business re-deriving.
CARRIED_COLUMNS: tuple[str, ...] = (
    "tile",
    "embedded",
    "optical_min_obs",
    "bbox_west",
    "bbox_south",
    "bbox_east",
    "bbox_north",
    "code_version",
    "code_commit",
)


def measurements_from_obs(
    s2_obs: np.ndarray,
    *,
    optical_min_obs: int,
    radar_rule_enforced: bool,
    s1_asc: np.ndarray | None = None,
    s1_desc: np.ndarray | None = None,
) -> dict[str, Any]:
    """One tile's measurement columns, from its observation-count arrays.

    Mirrors ``actors._coverage_record`` field for field, including its rounding — the two must
    produce identical numbers on identical pixels or :func:`compare_row` has nothing to stand on.
    The differences are deliberate and both are documented in the module docstring: the counts are
    over the whole tile rather than an evaluated sub-window, and ``eligible_px`` is therefore null.

    ``radar_rule_enforced`` is cell policy, not a measurement: the global campaign ran
    ``allow_s2_only=True``, so no pixel was ever refused for missing radar and
    ``refused_no_radar_px`` is zero on all 1,728,365 measured rows. Passing ``True`` here would
    claim a refusal count this module cannot derive, so it raises instead.

    ``s1_asc``/``s1_desc`` are optional because reading them triples the bytes moved to answer one
    informational column. Omitting them leaves ``px_with_any_radar`` null — not measured, as
    opposed to measured at zero.
    """
    if radar_rule_enforced:
        raise ValueError(
            "radar_rule_enforced=True asks for a refused_no_radar_px this rebuild cannot derive: "
            "the store records radar PRESENCE per pixel but not which pixels the radar rule "
            "refused, and a zero would assert that none were. Every cell of the published "
            "campaign ran with the rule off."
        )
    if (s1_asc is None) != (s1_desc is None):
        raise ValueError("px_with_any_radar needs both orbits or neither — one alone counts half a sensor")

    any_obs = s2_obs > 0
    thin = any_obs & (s2_obs < optical_min_obs)
    counted = {
        "no_optical": int((~any_obs).sum()),
        "thin": int(thin.sum()),
        "no_radar": 0,
    }
    radar: int | None = None
    if s1_asc is not None and s1_desc is not None:
        if s1_asc.shape != s2_obs.shape or s1_desc.shape != s2_obs.shape:
            raise ValueError(
                f"radar arrays {s1_asc.shape}/{s1_desc.shape} do not describe the same tile as "
                f"the optical array {s2_obs.shape}"
            )
        radar = int(((s1_asc > 0) | (s1_desc > 0)).sum())
    return {
        # Left null by decision, though the crop is reproducible. See the module docstring.
        "eligible_px": None,
        "chunk_px": int(s2_obs.size),
        "refused_px": sum(counted.values()),
        **{f"refused_{reason}_px": n for reason, n in counted.items()},
        "px_with_any_optical": int(any_obs.sum()),
        "obs_max": int(s2_obs.max()) if s2_obs.size else 0,
        "median_obs_where_any": (round(float(np.median(s2_obs[any_obs])), 1) if any_obs.any() else 0.0),
        # None when the tile has no thin pixels at all, which is not the same as thin pixels
        # sitting at zero — the distinction an infill ranking depends on.
        "median_obs_where_thin": (round(float(np.median(s2_obs[thin])), 1) if thin.any() else None),
        "px_with_any_radar": radar,
        "radar_rule_enforced": False,
    }


def basis_violations(table: pa.Table) -> tuple[int, int]:
    """``(violating, measured)``: measured rows where the fill's optical test was not ``obs > 0``.

    The rebuild's whole basis, checked from the registry alone. A measured row satisfies
    ``refused_no_optical_px == eligible_px - px_with_any_optical`` exactly when the fill's
    reflectance term never removed a pixel; one row that does not means ``s2_obs_count`` cannot
    reproduce that row's split, and no re-derivation from the store should be trusted until it is
    explained. Every measured row is checked, not a sample: it costs no store reads.
    """
    measured = table.filter(pc.is_valid(table.column("chunk_px")))
    expected = pc.subtract(measured.column("eligible_px"), measured.column("px_with_any_optical"))
    agree = pc.fill_null(pc.equal(measured.column("refused_no_optical_px"), expected), False)
    return measured.num_rows - int(pc.sum(agree).as_py() or 0), measured.num_rows


def compare_row(rebuilt: Mapping[str, Any], recorded: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Verdict and differences for one re-derived row against the row the fill wrote.

    Returns ``(verdict, differences)`` where verdict is:

    ``match``
        Every comparable column is identical. This is the only verdict the gate accepts.
    ``cropped``
        The recorded row was counted over less than the whole tile
        (``eligible_px < chunk_px``), so a whole-tile rebuild is expected to differ and the row is
        excluded from the gate rather than failing it.
    ``unmeasured``
        The recorded row has no measurements — nothing to compare, which is the population the
        rebuild exists to fill.
    ``mismatch``
        Something the rebuild claims to derive exactly does not agree. One of these stops a
        backfill: the whole basis for writing these values is that they are re-derivations.

    ``eligible_px`` is never compared — the rebuild leaves it null by design. ``px_with_any_radar``
    is compared only when the rebuild measured it, so an optical-only pass is not failed for a
    column it did not claim.
    """
    if recorded.get("chunk_px") is None:
        return "unmeasured", []
    if recorded.get("eligible_px") is not None and recorded["eligible_px"] != recorded["chunk_px"]:
        return "cropped", []
    differences: list[str] = []
    for column in REBUILT_COLUMNS:
        if column == "eligible_px":
            continue
        got, want = rebuilt.get(column), recorded.get(column)
        if column == "px_with_any_radar" and got is None:
            continue
        if got is None and want is None:
            continue
        # A recorded null against a re-derived value is a mismatch on purpose, even if a part predates
        # the column: every published part carries every column, and a false stop is loud and cheap.
        if got is None or want is None or not _equal(got, want):
            differences.append(f"{column}: rebuilt {got!r} != recorded {want!r}")
    return ("match" if not differences else "mismatch"), differences


def _equal(got: object, want: object) -> bool:
    """Equality that survives Parquet's float/int round trip without being loose about it.

    The medians are stored ``float64`` and the counts ``int64``, and a row read back through
    pyarrow hands them over as Python or numpy scalars depending on the path. Comparing exactly is
    the point — this is a gate on a re-derivation, not a tolerance check — so floats are compared
    for exact equality after the same rounding :func:`measurements_from_obs` applies.
    """
    if isinstance(got, bool) or isinstance(want, bool):
        return bool(got) == bool(want)
    if not isinstance(got, numbers.Real) or not isinstance(want, numbers.Real):
        return bool(got == want)
    if isinstance(got, numbers.Integral) and isinstance(want, numbers.Integral):
        return int(got) == int(want)
    return float(got) == float(want)


def tile_window(tile: str, shape: Sequence[int], *, shard_px: int = SHARD_PX) -> tuple[int, int, int, int]:
    """The ``(y0, y1, x0, x1)`` pixel window a tile label names, clamped to the array.

    A registry tile label is ``chunk_<shard_y>_<shard_x>`` — the same shard-grid coordinate
    Icechunk's chunk index returns, which is what lets the registry and the store be compared as
    sets rather than as counts.
    """
    row, col = parse_tile_label(tile)
    y0, x0 = row * shard_px, col * shard_px
    # A negative or off-grid label would slice an empty or wrapped window and "measure" it as a tile.
    if row < 0 or col < 0 or y0 >= int(shape[1]) or x0 >= int(shape[2]):
        raise ValueError(f"{tile!r} lies outside an array of shape {tuple(shape)}")
    return y0, min(y0 + shard_px, int(shape[1])), x0, min(x0 + shard_px, int(shape[2]))


def parse_tile_label(tile: str) -> tuple[int, int]:
    """``chunk_<row>_<col>`` as integers. Raises on anything else, rather than guessing a tile."""
    parts = tile.split("_")
    if len(parts) != 3 or parts[0] != "chunk":
        raise ValueError(f"{tile!r} is not a registry tile label of the form chunk_<row>_<col>")
    return int(parts[1]), int(parts[2])


def read_tile(group: zarr.Group, var: str, time_index: int, tile: str) -> np.ndarray:
    """One tile of one observation-count array, as a 2-D array.

    A tile is exactly one shard, so this is one object read and no partial-shard decompression —
    the geometry the rebuild's cost estimate assumes.
    """
    array = cast("zarr.Array", group[var])
    y0, y1, x0, x1 = tile_window(tile, array.shape)
    return np.asarray(array[time_index, y0:y1, x0:x1])


def rebuild_tile(
    group: zarr.Group,
    *,
    time_index: int,
    tile: str,
    optical_min_obs: int,
    with_radar: bool,
) -> dict[str, Any]:
    """Re-derive one tile's measurement columns by reading the store."""
    s2_obs = read_tile(group, OPTICAL_OBS_VAR, time_index, tile)
    radar: dict[str, np.ndarray] = {}
    if with_radar:
        radar = {name: read_tile(group, name, time_index, tile) for name in RADAR_OBS_VARS}
    return measurements_from_obs(
        s2_obs,
        optical_min_obs=optical_min_obs,
        radar_rule_enforced=False,
        s1_asc=radar.get(RADAR_OBS_VARS[0]),
        s1_desc=radar.get(RADAR_OBS_VARS[1]),
    )


def rebuild_schema() -> pa.Schema:
    """The part schema for the ``rebuild/`` prefix: the registry's own, plus ``filled_at``.

    A rebuilt row's ``assembled_at`` is when the REBUILD produced it, because that is what the
    compaction resolves latest-wins on and a run id is not a clock. That would otherwise lose when
    the cell was originally filled, so the fill's own stamp is carried in ``filled_at``. On a fill
    part the two are the same value and the column is absent; the compacted master carries it for
    every row.

    Declared here rather than by extending the registry's writer because ``storage/registry.py``
    sits inside the inference code-identity closure — editing it moves ``inference_code_identity``
    and invalidates every staged tile a future fill might have resumed into. Nothing about a
    rebuild should cost that.
    """
    return pa.schema([*registry_schema(), pa.field("filled_at", pa.string())])


def part_uri(rebuild_root: str, zone: str, year: int, run_id: str) -> str:
    """Where one cell's rebuilt part lands, under a prefix of its own.

    ``rebuild/`` sits BESIDE ``parts/``, never inside it. A consumer reading ``parts/`` today sees
    exactly what the campaign published — same rows, same count, same schema — and a rebuild adds
    a second row for a tile nowhere that anybody reads unfiltered. The compaction is what merges
    the two, and rollback is deleting this prefix.
    """
    return f"{rebuild_root.rstrip('/')}/zone={zone}/year={year}/{run_id}.parquet"


def write_rebuild_part(
    uri: str,
    rows: list[dict[str, Any]],
    *,
    open_output: Callable[[str], AbstractContextManager[Any]],
    zone: str,
    year: int,
    extra_metadata: Mapping[str, str] | None = None,
) -> int:
    """Write ``rows`` as one rebuilt part at ``uri``; returns the rows written.

    Mirrors ``registry.write_registry_part`` — one shot, zstd, zone and year in the key-value
    block rather than as columns — against :func:`rebuild_schema`. A separate writer rather than a
    schema argument on the original, for the code-identity reason in :func:`rebuild_schema`.
    """
    if not rows:
        return 0
    table = pa.Table.from_pylist(rows, schema=rebuild_schema())
    table = table.replace_schema_metadata(
        {"zone": zone, "year": str(year), "run_id": str(rows[0]["run_id"]), **dict(extra_metadata or {})}
    )
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    with open_output(uri) as handle:
        handle.write(buf.getvalue())
    return table.num_rows


def rebuilt_row(
    source: Mapping[str, Any], measurements: Mapping[str, Any], *, run_id: str, assembled_at: str
) -> dict[str, Any]:
    """A full rebuilt row for one tile: carried identity, re-derived measurements.

    ``run_id`` and ``assembled_at`` are the REBUILD's, not the fill's — a row has to say which pass
    produced it, and ``assembled_at`` is what lets the compaction resolve latest-wins without
    treating a run id as a clock. The fill's own stamp is carried in ``filled_at`` so nothing is
    lost. The columns naming the fill (``code_version``, ``code_commit``, the bbox, the depth rule)
    are carried across unchanged: a rebuild does not become the build that refused the tile.
    """
    row: dict[str, Any] = {column: source.get(column) for column in CARRIED_COLUMNS}
    row["run_id"] = run_id
    row["assembled_at"] = assembled_at
    row["filled_at"] = source.get("filled_at") or source.get("assembled_at")
    row.update({column: measurements.get(column) for column in REBUILT_COLUMNS})
    return row


def rebuild_run_id(*, when: datetime.datetime | None = None, suffix: str = "") -> str:
    """A run id for one rebuild pass, sortable and obviously not a fill's.

    Parts are keyed by run, so this is what keeps a rebuild's output beside the campaign's rather
    than over it, and what a later pass supersedes by ``assembled_at``.
    """
    stamp = (when or datetime.datetime.now(datetime.UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"rebuild-{stamp}{f'-{suffix}' if suffix else ''}"
