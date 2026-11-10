"""Publish a run's provenance record beside the registry, after its shards commit and before its tag.

Assembly's last data step (``context_docs/storage/registry-provenance-plan.md`` §4, Appendix C). From the
mosaics, read at the snapshots the actors read, it exports each slot's granule list as that
snapshot's ``granules`` file, unless an earlier run already did, in which case it checks the file
holds the same lists. From the staged shards it writes the run's ``shard_dates`` table, whose rows
name the snapshot, the run's group in the ``pixel_sources`` store and, for a holes update,
``update_pixels``, the pixels it embedded. It checks that they agree with each other and with the
mosaics as it goes. The layout is drawn in ``storage/provenance.py``.

**A failure fails the zone-year.** :func:`publish_provenance` raises rather than logging: the caller
runs it before the tag and the staging cleanup, so a failed publication leaves the zone-year
untagged with everything its retry needs, and the retry republishes. Every per-run write replaces the
run's previous attempt and a snapshot's granule file is written once, so publishing twice yields the
same record. There is no separate marker.

**The record follows the mosaics.** It is published when the zone-year's mosaics were ingested with
the ``provenance`` switch on, and nothing is published when none of them was. A mix can only be
made by hand, and is refused.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

import icechunk
import numpy as np
import pyarrow.parquet as pq
import zarr
from fsspec import AbstractFileSystem

from tessera_embeddings.config.ingest import FUSED, GRANULES_ARRAY
from tessera_embeddings.config.store_layout import SENSORS
from tessera_embeddings.inference.assembly import _COVERAGE_SUFFIX, _fs_for, _open_staged_tile
from tessera_embeddings.inference.chunk_spec import parse_chunk_label
from tessera_embeddings.inference.provenance import (
    PLANE_COMPRESSORS,
    STAGED_UPDATE_PIXELS,
    provenance_summary,
    read_array,
    staged_planes,
)
from tessera_embeddings.storage.manifest import extract_manifest
from tessera_embeddings.storage.provenance import (
    days_of,
    granule_rows,
    granules_schema,
    granules_uri,
    pixel_sources_group,
    pixel_sources_uri,
    shard_date_rows,
    shard_dates_schema,
    shard_dates_uri,
    update_pixels_uri,
    write_table,
)
from tessera_embeddings.storage.shard_writer import commit_with_rebase
from tessera_embeddings.storage.time_axis import TIME_ENCODING
from tessera_embeddings.storage.zarr_store import (
    open_or_create_repo,
    open_store_as_zarr_group,
    plain_zarr_storage_options,
)

logger = logging.getLogger(__name__)

#: Mosaic store name to the sensor its record is published under.
_STORES = {"reflectance": "s2", "sar_ascending": "s1_asc", "sar_descending": "s1_desc"}

#: Dates copied per write when moving a shard's staged planes into the run's group.
_COPY_BATCH = 16

#: Shards copied at once. The writes go to one session, which takes concurrent chunk writes.
_COPY_WORKERS = 8

_Creds = Callable[[], icechunk.S3StaticCredentials] | None


class ProvenanceCheckError(RuntimeError):
    """The record disagrees with itself or with the mosaics; the zone-year must not be tagged."""


@dataclass(frozen=True)
class ProvenanceRun:
    """One run's record to publish for one zone-year: whose it is, what it evaluated and what it read.

    ``embedded`` and ``refused`` partition the live shards the run evaluated, as the registry part
    does. ``mosaic_snapshots`` maps each mosaic store the run read to the snapshot the plan sampled,
    so the export describes exactly what the model read. ``grid_shape`` and ``shard_px`` are the
    zone group's.
    """

    staging_base: str
    registry_root: str
    zone: str
    year: int
    run_id: str
    embedded: Sequence[str]
    refused: Sequence[str]
    mosaic_snapshots: Mapping[str, str]
    grid_shape: tuple[int, int]
    shard_px: int


@dataclass(frozen=True)
class ProvenanceAccess:
    """Credentials for the mosaics and for the registry, which may sit in different buckets, and the region."""

    mosaic_credentials: _Creds = None
    registry_credentials: _Creds = None
    s3_region: str | None = None


@dataclass(frozen=True)
class _Copy:
    """What copying any one shard's staged planes needs from its run: the target, the mosaics' days and lists."""

    run: ProvenanceRun
    run_group: zarr.Group
    axes: Mapping[str, Sequence[Any]]
    counts: Mapping[str, Sequence[int]]
    pins: Mapping[str, str]


def publish_provenance(
    run: ProvenanceRun,
    access: ProvenanceAccess | None = None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] = logger,
) -> dict[str, Any] | None:
    """Publish ``run``'s record; ``None`` when its mosaics carry none.

    Raises:
        ProvenanceCheckError: the record fails a publication check, or the mosaics disagree about whether
            they carry one.
    """
    access = access or ProvenanceAccess()
    zone, year, run_id, registry_root = run.zone, run.year, run.run_id, run.registry_root
    mosaics = _provenance_mosaics(run.mosaic_snapshots, access.mosaic_credentials, access.s3_region)
    if mosaics is None:
        return None
    axes = {sensor: days_of(read_array(root, "time")) for sensor, root in mosaics.items()}
    counts = {sensor: _granule_counts(sensor, root) for sensor, root in mosaics.items()}
    fs = _fs_for(
        registry_root, plain_zarr_storage_options(registry_root, access.registry_credentials, access.s3_region)
    )

    def _open_output(uri: str) -> Any:  # noqa: ANN401 — an fsspec file
        fs.makedirs(uri.rsplit("/", 1)[0], exist_ok=True)
        return fs.open(uri, "wb")

    pins = _pins(run.mosaic_snapshots)
    metadata = {"zone": zone, "year": str(year), "run_id": run_id, **{f"snapshot_{s}": p for s, p in pins.items()}}

    granules = sum(
        _publish_granules(
            granules_uri(registry_root, zone, year, sensor, pins[sensor]),
            granule_rows(pins[sensor], sensor, axes[sensor], read_array(root, GRANULES_ARRAY).tolist()),
            fs,
            _open_output,
            {
                "zone": zone,
                "year": str(year),
                "sensor": sensor,
                "snapshot_id": pins[sensor],
                "first_published_by": run_id,
            },
        )
        for sensor, root in mosaics.items()
    )

    shards = _staged_shards(run.staging_base, run_id, run.embedded, run.refused)
    unrecorded = sorted(label for label, staged in shards.items() if staged is None)
    if missing := sorted(set(unrecorded) & set(run.embedded)):
        raise ProvenanceCheckError(f"{zone}/{year} run {run_id}: embedded shard(s) staged no provenance: {missing}")

    repo, _ = open_or_create_repo(
        pixel_sources_uri(registry_root), get_credentials=access.registry_credentials, region=access.s3_region
    )
    _ensure_group(repo, f"{zone}/{year}")
    session = repo.writable_session("main")
    run_group = zarr.open_group(session.store, path=pixel_sources_group(zone, year, run_id), mode="w")
    run_group.attrs.update(
        {"zone": zone, "year": year, "run_id": run_id, "grid_shape": list(run.grid_shape), "shard_px": run.shard_px}
    )
    for sensor, days in axes.items():
        run_group.create_array(
            sensor,
            shape=(len(days), *run.grid_shape),
            chunks=(1, run.shard_px, run.shard_px),
            dtype=np.int8,
            fill_value=0,
            compressors=PLANE_COMPRESSORS,
            dimension_names=("date", "northing", "easting"),
        )
        date = run_group.create_array(f"{sensor}_date", shape=(len(days),), chunks=(max(len(days), 1),), dtype=np.int64)
        date[:] = read_array(mosaics[sensor], "time").astype("datetime64[ns]").astype("int64")
        date.attrs.update(TIME_ENCODING)

    target = _Copy(run, run_group, axes, counts, pins)

    def _copy(label: str) -> list[dict[str, Any]]:
        staged = shards[label]
        return [] if staged is None else _copy_shard(target, label, *staged)

    with ThreadPoolExecutor(max_workers=_COPY_WORKERS) as pool:
        rows = [row for shard_rows in pool.map(_copy, sorted(shards)) for row in shard_rows]
    commit_with_rebase(session, f"provenance {zone}/{year} run {run_id}")

    shard_dates = write_table(
        shard_dates_uri(registry_root, zone, year, run_id),
        rows,
        shard_dates_schema(),
        open_output=_open_output,
        metadata=metadata | {"unrecorded_tiles": ",".join(unrecorded)},
    )
    holes = _publish_update_pixels(shards, run, access)
    summary = {"granules": granules, "shard_dates": shard_dates, "shards": len(shards), "unrecorded": unrecorded}
    log.info(
        "Provenance %s/%d run %s: %d granule row(s), %d shard-date row(s) over %d shard(s), %d holes mask(s)%s",
        zone,
        year,
        run_id,
        granules,
        shard_dates,
        len(shards) - len(unrecorded),
        holes,
        f"; {len(unrecorded)} refused shard(s) staged no record" if unrecorded else "",
    )
    return summary


def _pins(mosaic_snapshots: Mapping[str, str]) -> dict[str, str]:
    """``{sensor: snapshot}`` from ``{store path: snapshot}``."""
    return {_sensor_of(path): snap for path, snap in mosaic_snapshots.items()}


def _publish_granules(
    uri: str,
    rows: list[dict[str, Any]],
    fs: AbstractFileSystem,
    open_output: Callable[[str], Any],
    metadata: Mapping[str, str],
) -> int:
    """Write one snapshot's granule lists, once; returns their rows.

    A later run that read the same snapshot, a retry among them, leaves the file and checks it holds
    the lists read now, day by day, so every run that names the snapshot reads the same lists.
    """
    if fs.exists(uri):
        key = ("solar_day", "fusion_order", "granule_id", "status")
        with fs.open(uri, "rb") as f:
            have = pq.read_table(f, columns=list(key)).to_pylist()
        if [tuple(r[k] for k in key) for r in have] != [tuple(r[k] for k in key) for r in rows]:
            raise ProvenanceCheckError(f"{uri}: already published with granule lists other than its snapshot's")
        return len(rows)
    return write_table(uri, rows, granules_schema(), open_output=open_output, metadata=metadata)


def mosaics_carry_provenance(
    mosaic_snapshots: Mapping[str, str], *, get_credentials: _Creds = None, s3_region: str | None = None
) -> bool:
    """Whether the pinned mosaics were ingested with the provenance record. Refuses a mix."""
    return _provenance_mosaics(mosaic_snapshots, get_credentials, s3_region) is not None


def _sensor_of(path: str) -> str:
    """The sensor a mosaic store path holds, from its name."""
    return _STORES[path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".zarr")]


def _provenance_mosaics(
    snapshots: Mapping[str, str], get_credentials: _Creds, s3_region: str | None
) -> dict[str, zarr.Group] | None:
    """The pinned mosaics by sensor, or ``None`` when none carries the record. Refuses a mix."""
    roots = {
        _sensor_of(path): open_store_as_zarr_group(
            path, get_credentials=get_credentials, region=s3_region, snapshot_id=pin
        )
        for path, pin in snapshots.items()
    }
    versions = {s: (extract_manifest(dict(r.attrs)) or {}).get("provenance_version") for s, r in roots.items()}
    if not any(versions.values()):
        return None
    if not all(versions.values()):
        raise ProvenanceCheckError(
            f"mosaics {sorted(snapshots)} disagree about the provenance record: {versions}. One ingest "
            "dispatch seeds all three with one setting, so a mix was made by hand; re-ingest the odd one."
        )
    return roots


def _granule_counts(sensor: str, root: zarr.Group) -> list[int]:
    """Fused granules per slot. Every slot must have some: the ingest writes its list in the slot's commit."""
    counts = [
        sum(g.get("status", FUSED) == FUSED for g in _json_list(raw))
        for raw in read_array(root, GRANULES_ARRAY).tolist()
    ]
    slots = len(read_array(root, "time"))
    if len(counts) != slots or not all(counts):
        empty = [i for i, n in enumerate(counts) if not n]
        raise ProvenanceCheckError(
            f"{sensor}: {len(counts)} granule list(s) for {slots} slot(s), empty at {empty[:10]}"
        )
    return counts


def _json_list(raw: str) -> list[Any]:
    value = json.loads(raw) if raw else []
    return value if isinstance(value, list) else []


def _staged_shards(
    staging_base: str, run_id: str, embedded: Sequence[str], refused: Sequence[str]
) -> dict[str, tuple[zarr.Group, int] | None]:
    """Each shard's staged tile and embedded-pixel count, or ``None`` where it staged no record.

    An embedded shard's tile is its staged ``.zarr``; a refused shard's is its coverage-only tile,
    which its actor may have failed to write, a loss its skip marker survives.
    """
    base = f"{staging_base.rstrip('/')}/{run_id}"

    def _one(label: str, suffix: str) -> tuple[zarr.Group, int] | None:
        try:
            # Not from the consolidated listing, which predates the planes.
            group = _open_staged_tile(f"{base}/{label}{suffix}", use_consolidated=False)
        except (FileNotFoundError, RuntimeError):
            return None
        if provenance_summary(dict(group.attrs)) is None:
            return None
        coverage = group.attrs.get("coverage")
        return group, int(cast(int, coverage.get("embedded_px", 0))) if isinstance(coverage, dict) else 0

    return {label: _one(label, ".zarr") for label in embedded} | {
        label: _one(label, _COVERAGE_SUFFIX) for label in refused
    }


def _copy_shard(copy: _Copy, label: str, group: zarr.Group, embedded_px: int) -> list[dict[str, Any]]:
    """Copy one shard's staged planes into the run's group, checking them; return its rows."""
    row, col = parse_chunk_label(label)
    px = copy.run.shard_px
    summary = cast(dict[str, Any], provenance_summary(dict(group.attrs)))
    if missing := sorted(set(copy.counts) - set(summary)):
        raise ProvenanceCheckError(f"{label}: no staged record for {missing}, whose mosaics carry one")
    rows: list[dict[str, Any]] = []
    for sensor in sorted(summary):
        times, planes = cast("tuple[np.ndarray, zarr.Array]", staged_planes(group, sensor))
        target, counts, record = cast(zarr.Array, copy.run_group[sensor]), copy.counts[sensor], summary[sensor]
        window = (slice(row * px, row * px + planes.shape[1]), slice(col * px, col * px + planes.shape[2]))
        for start in range(0, len(times), _COPY_BATCH):
            batch = np.asarray(planes[start : start + _COPY_BATCH])
            for i, plane in enumerate(batch, start):
                where, time = f"{label} {sensor} slot {times[i]}", int(times[i])
                if time >= len(counts):
                    raise ProvenanceCheckError(f"{where}: no such slot on the mosaic's time axis ({len(counts)} slots)")
                _check_plane(where, plane, record["sources"][i], record["valid_px"][i], counts[time])
            target.oindex[(times[start : start + len(batch)], *window)] = batch
        ids = {"run_id": copy.run.run_id, "tile": label, "sensor": sensor, "snapshot_id": copy.pins[sensor]}
        rows += shard_date_rows(ids, record, copy.axes[sensor], embedded_px)
    return rows


def _check_plane(where: str, plane: np.ndarray, sources: Sequence[int], valid: int, fused: int) -> None:
    """A plane's positive count is its row's ``valid_px``, its values rank its ``sources``, and they exist.

    ``fused`` is how many granules the snapshot the row names fused that day, which its ``granules``
    file holds, so a source below it is a fused granule of that day in that file.
    """
    if (positive := int(np.count_nonzero(plane > 0))) != valid:
        raise ProvenanceCheckError(f"{where}: plane has {positive} valid pixels, its row says {valid}")
    if np.unique(np.abs(plane[plane != 0])).tolist() != list(range(1, len(sources) + 1)):
        raise ProvenanceCheckError(f"{where}: plane values do not rank its {len(sources)} source(s)")
    if sources and max(sources) >= fused:
        raise ProvenanceCheckError(f"{where}: source {max(sources)} beyond the day's {fused} granule(s)")


def _ensure_group(repo: icechunk.Repository, path: str, tries: int = 8) -> None:
    """Create ``path``'s groups in a commit of their own, tolerating another cell creating them first.

    Two cells of one zone may assemble at once, and two sessions both creating ``<zone>`` conflict
    on rebase; whichever loses finds the group there on its next look.
    """
    for _ in range(tries):
        session = repo.writable_session("main")
        try:
            zarr.open_group(session.store, path=path, mode="r")
            return
        except (FileNotFoundError, KeyError):
            zarr.open_group(session.store, path=path, mode="a")
        try:
            commit_with_rebase(session, f"provenance: create {path}")
            return
        except (icechunk.RebaseFailedError, icechunk.ConflictError):
            continue
    raise RuntimeError(f"could not create provenance group {path} after {tries} attempts")


def _publish_update_pixels(
    shards: Mapping[str, tuple[zarr.Group, int] | None], run: ProvenanceRun, access: ProvenanceAccess
) -> int:
    """A holes update's per-shard masks of the pixels it embedded, one boolean array per shard.

    Returns how many. The checks hold each mask to the shard's embedded count and the planes to the
    mask: no plane may name a pixel the update did not embed.
    """
    masks = {
        label: read_array(staged[0], STAGED_UPDATE_PIXELS)
        for label, staged in shards.items()
        if staged is not None and STAGED_UPDATE_PIXELS in staged[0]
    }
    if not masks:
        return 0
    for label, mask in masks.items():
        group = cast("tuple[zarr.Group, int]", shards[label])[0]
        if (embedded_px := cast("tuple[zarr.Group, int]", shards[label])[1]) != int(mask.sum()):
            raise ProvenanceCheckError(f"{label}: {embedded_px} pixels embedded, {int(mask.sum())} in its mask")
        for sensor in SENSORS:
            planes = staged_planes(group, sensor)
            if planes is not None and any(np.asarray(planes[1][i])[~mask].any() for i in range(planes[1].shape[0])):
                raise ProvenanceCheckError(f"{label} {sensor}: a plane names a pixel the update did not embed")
    uri = update_pixels_uri(run.registry_root, run.zone, run.year, run.run_id)
    options = plain_zarr_storage_options(uri, access.registry_credentials, access.s3_region)
    out = zarr.open_group(uri, mode="w", storage_options=options)
    out.attrs.update({"zone": run.zone, "year": run.year, "run_id": run.run_id})
    for label, mask in masks.items():
        out.create_array(label, shape=mask.shape, chunks=mask.shape, dtype=bool)[:] = mask
    return len(masks)
