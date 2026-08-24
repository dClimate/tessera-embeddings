"""Sentinel-1 OPERA RTC SAR ingestion for ROI-based regions.

Pure-domain implementation extracted from the reference repo's
``flows/ingest_s1_roi_sar.py::process_roi_sar``. No Prefect imports,
no ``get_run_logger``, no ``get_client``, no direct env-var reads for
secrets — callers supply a connected :class:`dask.distributed.Client`,
a logger, and (where required) a credential callback.

Algorithm (unchanged from the reference):

1. Read ROI metadata + lazy mask once per batch (so Dask graphs
   reference fresh keys whenever credentials are refreshed).
2. Walk the date range in batches of ``batch_days`` to keep each
   ``compute()`` call's task graph manageable.
3. Each batch:

   * Renew the OPERA read credentials whenever they are close to expiring.
     A background ticker owns this, so renewal does not wait for a unit of
     work to finish; the credential's roughly one-hour life is unrelated to
     the shape of the loop, and any unit that outlives the remaining margin
     cannot renew from inside itself. The loop also checks at a batch
     boundary and before every date's write, which costs nothing while the
     ticker is keeping the credential fresh.
   * Build an OPERA RTC item filter for the orbit / bounding box /
     batch window.
   * Call :func:`ingest_tile` to produce a per-batch ``xarray.Dataset``.
   * Apply the ROI mask, then write each date's live windows via
     :func:`tessera_embeddings.storage.zarr_store.write_day_windows` with a
     narrow tenacity retry on transient GDAL errors. Past that retry the date
     is GIVEN UP rather than allowed to fail the leg — but ONLY when the cause
     recomputes, so a provider refusing reads re-raises and the leg retries in
     order instead. Up to :data:`MAX_GIVEN_UP_DATES`; past that the leg stops
     terminally, because nothing counted toward that ceiling can clear.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Literal, cast, final

import dask.distributed
import numpy as np
import xarray as xr

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE, INGEST_CHUNKS
from tessera_embeddings.ingest._pipeline import pipelined
from tessera_embeddings.ingest.duplicates import is_provider_refusal, is_unreadable_source
from tessera_embeddings.ingest.live_windows import (
    WINDOW_COST_IN_CHUNKS,
    WINDOW_COST_IN_CHUNKS_OVERLAPPED,
    live_windows_for_mask,
    windows_for_date,
)
from tessera_embeddings.ingest.opera_query import make_s1_item_provider
from tessera_embeddings.ingest.roi import (
    StorageOptions,
    read_roi_mask,
    read_roi_metadata,
    resolve_storage_options,
)
from tessera_embeddings.ingest.roi_processing import apply_roi_mask, read_failure_context
from tessera_embeddings.ingest.solar_days import (
    SolarDayRange,
    fixed_day_ranges,
    normalize_to_solar_day,
    owned_items,
    solar_grouping_longitude,
)
from tessera_embeddings.ingest.stac import ingest_tile
from tessera_embeddings.ingest.transforms import amplitude_to_db
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import (
    get_existing_dates,
    record_assessed_window,
    store_write_retrying,
    write_day_windows,
)

logger = logging.getLogger(__name__)

DEFAULT_CRED_REFRESH_INTERVAL_SEC = 30 * 60
"""Fallback refresh cadence, used only when the credential advertises no expiry."""

CRED_EXPIRY_MARGIN_SEC = 15 * 60
"""Renew this far ahead of expiry.

Must exceed the longest single date write, because a credential that is valid when a
write starts must still be valid when it ends — the reads happen throughout. Renewing is
two cheap HTTP calls, so a generous margin costs almost nothing and the failure it
prevents costs the remainder of the run.
"""

CRED_TICK_INTERVAL_SEC = 5 * 60
"""How often the background ticker re-checks the credential's remaining life.

Bounds how far past the renewal point a credential can drift, so it must be well under
``CRED_EXPIRY_MARGIN_SEC``. A tick only *checks*; the staleness test decides whether to
spend an HTTP call, so a short interval is nearly free.
"""

S1Orbit = Literal["ascending", "descending"]


MAX_GIVEN_UP_DATES = 10
"""How many dates one radar leg may give up before it stops and refuses the year.

**Its original reasoning no longer holds and the ceiling means something else now.** It was
"small on purpose, because a provider refusal CLEARS" — a refusal used to be countable here, so
grinding through an outage converted recoverable imagery into a durable hole and the leg still
returned success. A refusal is no longer a reason to give up a date at all: it re-raises and the
leg retries in order.

So every date counted here is now one whose bytes will not read, whatever we do. The ceiling is
therefore a statement about the DATA — past ten, this orbit-year is too damaged to be worth a
mosaic — rather than a request to come back when a service recovers. Ten is a small fraction of a
radar year's couple of hundred dates per orbit.
"""


class TooManyGivenUpDatesError(RuntimeError):
    """A radar leg gave up more dates than the bounded skip permits.

    **NOT retryable, and it used to be.** It was deliberately absent from
    ``ingest_zone_year._NON_RETRYABLE_LEG_MARKERS`` "because a refusal goes away and a re-dispatch
    writes the dates this leg gave up". Nothing counted toward the ceiling goes away any more:
    every one of those dates failed for a cause that recomputes, so a re-dispatch re-reads the
    same unreadable objects, spends the per-read retry ladder on each of them again, and holds a
    Dask fleet to reach the identical answer.
    """


@contextmanager
def credential_ticker(
    refresh: Callable[[], None],
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    orbit: str,
    roi: str,
    interval_sec: float = CRED_TICK_INTERVAL_SEC,
) -> Iterator[None]:
    """Call ``refresh`` on a timer for the duration of the block, in a daemon thread.

    A read credential renewed only from the work loop can be renewed only *between* units
    of work, so any single unit that outlives the remaining margin — a long write, a query,
    a stall — has no opportunity to renew from inside itself. That coupling is
    self-reinforcing: slower work renews less often, an expired credential fails every
    read, failing reads stop progress, and no progress means no further renewal. A timer
    removes the coupling, so renewal no longer depends on the shape or the speed of the
    loop.

    ``refresh`` must be idempotent and safe to call concurrently with the loop's own calls;
    it decides for itself whether a renewal is due.

    Exceptions from ``refresh`` are logged and swallowed. The ticker must outlive a single
    failed renewal: the next tick retries well inside the margin, whereas a dead thread
    would silently return the caller to loop-driven renewal.
    """
    stop = threading.Event()

    def tick() -> None:
        while not stop.wait(interval_sec):
            try:
                refresh()
            except Exception:
                log.warning("[%s] Credential ticker refresh failed roi=%s", orbit, roi, exc_info=True)

    thread = threading.Thread(target=tick, name=f"cred-ticker-{orbit}-{roi}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval_sec)


def _parse_credential_expiry(creds: dict[str, str]) -> float | None:
    """Epoch seconds at which ``creds`` expires, or ``None`` if not stated or unparseable.

    Returning ``None`` rather than raising is deliberate: an unreadable expiry must
    degrade to the age-based cadence, not sink an ingest that would otherwise run.
    """
    raw = creds.get("expiration")
    if not raw:
        return None
    try:
        # ASF returns ISO-8601; tolerate both the "Z" and "+00:00" spellings, and a
        # space instead of "T".
        text = str(raw).strip().replace("Z", "+00:00").replace(" ", "T", 1)
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except ValueError:
        logger.warning("Could not parse credential expiry %r; falling back to age-based refresh", raw)
        return None


@final
@dataclass(frozen=True)
class SarIngestResult:
    """Return value from :func:`ingest_s1_roi_sar`.

    Replaces the dict that the reference repo's ``@task`` returned.
    Task shells convert back to a dict via ``dataclasses.asdict()`` at
    the Prefect boundary.

    Attributes:
        roi_path: Echo of the input ``roi_zarr_path``.
        status: ``"success"`` if at least one date was written, else
            ``"skipped"``.
        dates_processed: ``{orbit: count}``. The single-orbit shape
            mirrors the reference repo and lets a multi-orbit caller
            merge two results trivially.
    """

    roi_path: str
    status: str
    dates_processed: dict[str, int] = field(default_factory=dict)


@final
@dataclass(frozen=True)
class _PreparedBatch:
    """One batch carried from the catalogue query to the write loop.

    The split is the same one the phase logging measures: everything up to a write-ready
    dataset on one side (catalogue query plus lazy graph build, no cluster reads), the
    per-date writes on the other. That boundary is what makes the query hideable — it
    touches no store and holds no credential, so it can run on a background thread while
    the previous batch writes.

    ``data`` is ``None`` when the batch has no new dates, which is a normal outcome and
    not an error: a sparse region or an already-ingested range yields nothing to write.
    """

    start: str
    end: str
    data: xr.Dataset | None
    baselines: dict[str, int]
    query_s: float
    #: Owned catalogue items this batch's query returned. Totalled by the consume loop so a
    #: leg that writes nothing can say whether the source had anything to write. No default:
    #: ``date_windows`` below has none either, and a defaulted field cannot precede it.
    items_seen: int
    #: Solar day (keyed by the loaded slice's own timestamp) -> the live windows that day's
    #: imagery reaches. Empty list = reaches nothing, so skip. Missing key = unknown, so
    #: write everything. Empty dict = the whole batch falls back.
    #: Loaded-slice timestamp -> (that slice's SOLAR DAY, the live windows its imagery
    #: reaches). The solar day is carried because odc labels a slice with its first
    #: item's timestamp, whose calendar date can differ from the solar day the slice
    #: actually represents wherever the offset crosses UTC midnight.
    date_windows: dict[str, tuple[str, list[tuple[int, int, int, int]]]]


def _baselines_for(baselines: dict[str, int], dates: Iterable[str]) -> dict[str, int]:
    """The baseline entries belonging to ``dates`` only.

    Each cropped per-date write is its own atomic commit, and ``write_day_windows`` merges the
    map it is handed into the store's ``baselines_applied``. Handing it the whole
    query's map makes the very first commit claim provenance for every date in the
    month — including dates a later coverage rejection or crash means the store never
    receives. Provenance that describes data which is not there is worse than none.
    """
    wanted = {d[:10] for d in dates}
    return {k: v for k, v in baselines.items() if k[:10] in wanted}


def ingest_s1_roi_sar(
    *,
    roi_zarr_path: str,
    start_date: str,
    end_date: str,
    store_path: str,
    client: dask.distributed.Client,  # noqa: ARG001 — see the docstring: held, not called
    orbit: S1Orbit,
    batch_days: int = 30,
    edl_credentials_fn: Callable[[], dict[str, str]] | None = None,
    apply_credentials_fn: Callable[[dict[str, str]], None] | None = None,
    use_s3_direct: bool = True,
    cred_refresh_interval_sec: float = DEFAULT_CRED_REFRESH_INTERVAL_SEC,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    storage_options: StorageOptions = None,
    overlap_window_writes: bool = True,
    pipeline_batches: bool = True,
    narrow_windows_per_date: bool = True,
    allow_ingest_code_mismatch: bool = False,
    s3_region: str | None = None,
) -> SarIngestResult:
    """Ingest OPERA RTC-S1 SAR for an ROI using batched time windows.

    Orchestrator-unaware. Same algorithm as the reference repo's
    ``flows/ingest_s1_roi_sar.py::process_roi_sar``.

    Args:
        roi_zarr_path: Path to the Zarr ROI store (any fsspec-compatible URI).
        start_date: Inclusive start date (``YYYY-MM-DD``).
        end_date: Inclusive end date (``YYYY-MM-DD``).
        store_path: Base path for satellite mosaics; the function
            creates ``sar_<orbit>.zarr`` underneath.
        client: Connected :class:`dask.distributed.Client`. Callers
            create this; we do not call ``get_client``. Required but never
            invoked directly: every compute here goes through the AMBIENT
            client, which constructing one makes current. Keeping it in the
            signature is what states that a connected cluster is a
            precondition — the S2 ingest takes it for the same reason and
            does use it, so the two entry points stay symmetric.
        orbit: ``"ascending"`` or ``"descending"`` — one orbit per call.
            Multi-orbit ingestion is a flow-level concern (call twice).
        batch_days: Days per time batch. Smaller values keep each Dask
            graph small at the cost of more STAC queries.
        edl_credentials_fn: Callable returning STS credentials (e.g. the
            dict from ``get_s3_credentials()``). Called when the cached
            credentials have aged past ``cred_refresh_interval_sec``.
            Required when accessing OPERA's S3 direct endpoints; the
            plain runner passes a closure over env vars, the Prefect
            flow passes a closure over a credentials block. ``None``
            means no per-batch refresh — only safe when credentials are
            already injected by the substrate (e.g. a Dask worker
            plugin set up at cluster start).
        apply_credentials_fn: Callable that takes the dict returned by
            ``edl_credentials_fn`` and applies it (typically by setting
            env vars on the orchestrator and registering a Dask
            ``WorkerPlugin``). When ``None``, the credentials returned
            are still fetched on schedule but not applied — useful for
            tests, otherwise pair with ``edl_credentials_fn``.
        use_s3_direct: When ``True``, use ASF's in-region S3 direct
            endpoints (requires us-west-2 reachability and STS creds).
            When ``False``, fall back to CloudFront-signed HTTPS URLs;
            useful for local development.
        cred_refresh_interval_sec: Refresh interval for the credential
            callback.
        log: Optional logger; defaults to ``logging.getLogger(__name__)``.
        storage_options: fsspec storage options for the ROI mask reads.
        overlap_window_writes: Defaults ON. Submit a date's windows as ONE dask compute
            rather than one blocking compute per window, so their critical paths
            overlap across the fleet instead of summing. Produces an identical store
            either way. Also selects the window merge exchange rate, since that prices
            a window boundary by how it is written — the two must not drift apart.
        pipeline_batches: Defaults ON. Prepare the NEXT batch's catalogue query while
            the current batch writes, so only the first batch pays its query on the
            critical path. Shares ``ingest._pipeline.pipelined`` with the S2 date loop.
            Look-ahead is fixed at one batch: a batch's write is one long consume, so
            depth 1 already covers it, and deeper retention is what once deadlocked the
            S2 driver. Set False to restore the strictly serial query-then-write loop.
        narrow_windows_per_date: Write only the live windows a date's own imagery can
            reach, as the S2 path does. **Defaults ON, measured.** A Sentinel-1 pass
            reaches a minority of a zone's windows — six times fewer in both zones tested
            — and that converts to 7-20% of per-date wall clock. Dates reaching NO live
            window are skipped unconditionally, independent of this flag: writing one
            builds a full graph to store nothing, since all-fill chunks never persist.
        allow_ingest_code_mismatch: Off by default. See the field of the same name on
            :class:`~tessera_embeddings.storage.manifest.IngestManifest`.

        s3_region: S3 region for the mosaic Icechunk store. ``None`` uses the
            storage layer's default; set it when the bucket lives elsewhere, or
            every write below signs against the wrong region.

    Returns:
        :class:`SarIngestResult`. ``status="skipped"`` if zero dates
        were written.
    """
    log = log or logging.getLogger(__name__)
    #: Short identifier for this ROI, stamped on every per-date progress line so a
    #: fleet-wide log query can attribute a commit to a cell. Derived from the store name
    #: (``.../zone_47S.zarr`` -> ``zone_47S``) rather than taken as a parameter, so every
    #: caller gets it without threading one more argument through the flows.
    roi_label = roi_zarr_path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".zarr")
    roi = read_roi_metadata(roi_zarr_path, storage_options=storage_options)

    ingest_manifest = IngestManifest.from_roi_store(
        roi_zarr_path,
        storage_options=resolve_storage_options(storage_options),
        allow_ingest_code_mismatch=allow_ingest_code_mismatch,
    )

    # Load blocks match the store's chunks: one read task per (chunk, band), and the
    # write needs no rechunk. See config.ingest for why a coarser load block was removed.
    spatial_chunks = {"northing": INGEST_CHUNKS["northing"], "easting": INGEST_CHUNKS["easting"]}

    last_cred_refresh: float = float("-inf")
    cred_expires_at: float | None = None
    cred_lock = threading.Lock()

    def refresh_credentials_if_stale() -> None:
        """Re-fetch the OPERA read credentials when they are close to expiring.

        Driven by the credential's OWN expiry rather than a fixed cadence. The margin must
        exceed the longest single date write, since a credential valid at the start of a
        write must still be valid at its end — the reads happen throughout it.

        Callable from the work loop and from the background ticker, so it takes a lock:
        two concurrent renewals would each broadcast, and the later broadcast could carry
        the earlier credential.
        """
        nonlocal last_cred_refresh, cred_expires_at
        if edl_credentials_fn is None:
            return
        with cred_lock:
            now_wall, now_mono = time.time(), time.monotonic()
            if cred_expires_at is not None:
                fresh_enough = now_wall < cred_expires_at - CRED_EXPIRY_MARGIN_SEC
            else:
                # No expiry advertised: fall back to the age-based cadence.
                fresh_enough = now_mono - last_cred_refresh <= cred_refresh_interval_sec
            if fresh_enough:
                return
            creds = edl_credentials_fn()
            if apply_credentials_fn is not None:
                apply_credentials_fn(creds)
            last_cred_refresh = now_mono
            cred_expires_at = _parse_credential_expiry(creds)

    def ticked(batches: Iterator[tuple[_PreparedBatch, float]]) -> Iterator[tuple[_PreparedBatch, float]]:
        """Run the credential ticker for exactly as long as the batches are being consumed.

        Wrapping the iterator rather than the loop body ties the ticker's lifetime to the
        work it protects — it starts before the first batch is consumed and stops however
        the loop leaves, including by raising.
        """
        if edl_credentials_fn is None:
            yield from batches
            return
        with credential_ticker(refresh_credentials_if_stale, log, orbit, roi_label):
            yield from batches

    orbit_store = f"{store_path}/sar_{orbit}.zarr"

    # Cropped write path: windows derived once from the same mask this ingest
    # reads (see ingest.live_windows; identical mechanics to the S2 path).
    #
    # `run_windows` keeps the window OBJECTS because per-date narrowing needs them;
    # `live_windows` is the plain-tuple form the storage layer takes.
    # The merge exchange rate follows how this run WRITES, exactly as on the S2 path:
    # overlapped windows share one graph, so a boundary is cheap and the DP should stop
    # trading ocean area for fewer windows. A sequential writer still pays the serial
    # cost per boundary and keeps the high rate. Bound once because per-date narrowing
    # re-merges on the same terms — a second, differing rate there would undo this.
    window_cost = WINDOW_COST_IN_CHUNKS_OVERLAPPED if overlap_window_writes else WINDOW_COST_IN_CHUNKS
    run_windows = live_windows_for_mask(
        roi_zarr_path,
        window_px=INGEST_CHUNK_SIZE,
        window_cost_in_chunks=window_cost,
        storage_options=storage_options,
    )
    live_windows: list[tuple[int, int, int, int]] = [(w.y0, w.y1, w.x0, w.x1) for w in run_windows]
    # roi= and orbit both present: this is the first line a leg emits, and at fleet width a
    # bare count cannot be tied to a cell OR to an orbit — the log stream is the Dask
    # worker's task id, and the two orbits of one zone are separate runs.
    log.info("[%s] Live windows roi=%s: %d", orbit, roi_label, len(live_windows))

    # An ROI with no live window at all — an all-ocean mask — has nowhere to put a
    # pixel, and every date would otherwise be COMMITTED with zero windows written:
    # a time slot holding nothing, which `get_existing_dates` then reports as
    # ingested, so no later run ever revisits it. Stop before the query rather than
    # bank empty dates. The per-date `{}` fallback cannot cover this, because the set
    # it falls back TO is the empty one. The campaign screens these cells out with
    # `zone_has_live_tiles`; the public ROI path has no such preflight, which is why
    # the S2 path guards its own coverage denominator the same way.
    if not live_windows:
        log.warning("ROI %s has no live window — no SAR date can store a pixel; skipping ingest", roi_zarr_path)
        return SarIngestResult(roi_path=roi_zarr_path, status="skipped", dates_processed={orbit: 0})

    # A date's own footprint, keyed so it can be matched back to the loaded time slice.
    #
    # THE JOIN IS ON AN EXACT TIMESTAMP, not a date string, and that is what makes this
    # safe. odc groups by solar day but sets each slice's time coordinate to
    # `group[0].nominal_datetime` — the EARLIEST item's real timestamp, since items are
    # sorted by time within a group. So the minimum item datetime in a group reproduces
    # odc's coordinate exactly. Keying by a derived solar-day string instead would
    # disagree with the coordinate wherever the solar offset crosses UTC midnight, and
    # would then narrow a date to the wrong footprint and drop real imagery silently.
    def _footprint_key(when: datetime) -> str:
        """Join key: naive-UTC timestamp to the second, matching odc's coordinate."""
        return when.replace(tzinfo=None).isoformat(timespec="seconds")

    mid_longitude = solar_grouping_longitude(roi)

    def _date_footprints(items: list) -> dict[str, tuple[str, list[tuple[int, int, int, int]]]]:
        """Per solar day, the live windows that day's imagery can actually reach.

        Empty list for a day whose imagery reaches NO live window — those days are skipped
        rather than committed, since writing one stores nothing and pays a full graph.

        Returns ``{}`` when anything is unusable (no windows, no longitude, no items), which
        makes the caller fall back to the full window set. That asymmetry is the safety
        argument: a footprint that is too LARGE only costs computed area that would have
        been discarded, while one that is too SMALL drops imagery and nothing downstream
        would notice.
        """
        if not run_windows or mid_longitude is None or not items:
            return {}
        # Items arrive solar-day-normalised from the provider, so their own date IS the
        # solar day — no offset here. Applying one would shift the key a second time.
        by_day: dict[date, list] = {}
        for item in items:
            when = getattr(item, "datetime", None)
            if when is None:
                return {}  # cannot group reliably; fall back for the whole batch
            by_day.setdefault(when.date(), []).append(item)

        out: dict[str, tuple[str, list[tuple[int, int, int, int]]]] = {}
        for solar_day, day_items in by_day.items():
            # odc's coordinate for this group is its EARLIEST item's timestamp.
            key = _footprint_key(min(i.datetime for i in day_items))
            narrowed = windows_for_date(
                run_windows,
                # An item may carry no bbox; windows_for_date reads that as an unusable
                # footprint and falls back to every window, which is the conservative
                # answer. cast because the list is Optional only in that fallback sense.
                cast("list[tuple[float, float, float, float]]", [getattr(i, "bbox", None) for i in day_items]),
                roi.geobox,
                chunk_px=INGEST_CHUNK_SIZE,
                window_cost_in_chunks=window_cost,
            )
            # windows_for_date returns its INPUT unchanged when the footprint cannot be
            # determined, so an unchanged length is not evidence of narrowing — that is
            # fine here, since writing the full set is the conservative outcome anyway.
            out[key] = (solar_day.isoformat(), [(w.y0, w.y1, w.x0, w.x1) for w in narrowed])
        return out

    total_processed = 0

    # Batches of up to ``batch_days`` SOLAR days, each querying a UTC range padded a day
    # either side. The two are different ranges on purpose: cutting on UTC dates and
    # writing every group the loader returned split any solar day landing on a cut, and
    # the later batch's half was then discarded as an already-written date — so the day
    # was committed missing acquisitions, at every boundary, in the zones whose offset
    # puts UTC midnight near a satellite pass. ingest.solar_days owns that reasoning and
    # the S2 month slicing uses the same mechanism.
    #
    # Materialised as a list rather than advanced in the loop because the look-ahead has
    # to know what comes next before the current batch is done with.
    batch_ranges: list[SolarDayRange] = fixed_day_ranges(start_date, end_date, batch_days)

    # Read ONCE, then maintained in-process as dates are written.
    #
    # The serial loop re-read this from the store every batch, so that a date written by
    # an earlier batch would be skipped by a later one. Under a look-ahead that read is
    # unsafe: the next batch's query is prepared BEFORE the current batch has written, so
    # a store read there would miss dates that are about to exist. Tracking writes
    # in-process is correct regardless of when the query runs, and drops a per-batch S3
    # read as a side effect.
    #
    # Why any of this is needed at all: batches are cut on UTC dates while the loader
    # groups by SOLAR day, so an acquisition late on a batch's last UTC day can belong to
    # the next batch's solar day, and two consecutive batches can then contain the same
    # day. Probing real catalogue responses did not find such a boundary (12 boundaries
    # across a +1 h and a +10 h zone), but the offset is a translation rather than a
    # split, so nothing rules it out — and the cost of being wrong is a duplicate-date
    # commit. The consume side below is therefore the authority on what has been written,
    # and the query filter is only an optimisation.
    # Written dates only. A date given up on is re-offered by the next attempt and gives up
    # again for the same reason: every accepted cause is DETERMINISTIC, so the verdict recomputes
    # and re-offering costs one re-evaluation rather than risking anything. A record of the skip
    # would only be needed if the verdict could change between attempts, which is what treating a
    # transient failure as a skip used to allow.
    written_dates: set[str] = get_existing_dates(orbit_store, s3_region=s3_region)
    # Counted for the assessed-window record: it separates "sparse region" from
    # "the footprints are wrong", which look identical in a date count alone.
    empty_dates = 0
    #: Dates given up because their source reads failed, recorded on the store rather than only
    #: logged: without the record a lost date and a day with no imagery look the same, and
    #: nothing downstream revisits either.
    given_up_dates: list[dict[str, str]] = []
    #: The same dates as a membership test. Batch queries are padded a day either side, so a
    #: boundary solar day comes back from two consecutive batches — without this it would be
    #: given up twice, listed twice, and cost two of the leg's budget.
    given_up: set[str] = set()
    #: Owned items across every batch — the number that distinguishes "the source does not
    #: cover this ROI for this orbit" from "we found items and committed none of them".
    total_items_seen = 0
    # Frozen at the start so the background thread reads an object nothing mutates.
    already_present = frozenset(written_dates)

    def _give_up_date(date_str: str, exc: BaseException) -> bool:
        """Accept the loss of one date, or decline to.

        Returns ``False`` when the failure is not one the source is answerable for, and the
        caller then re-raises — the fail-closed direction, so an unexamined cause stops the leg
        instead of quietly thinning its year.

        **A PROVIDER REFUSAL IS NOT ACCEPTED HERE, and that is a change.** It used to be, under
        ``scope="provider-refused"``, on the reasoning that a later run over the window would get
        the imagery. It would not: giving up a date and then committing a LATER one puts the
        earlier date permanently below the store's append-only maximum, so the re-run that was
        supposed to recover it is refused instead. A refusal is also transient by definition, so
        accepting it converts a bad minute at the source into a hole — the mirror of the defect
        that cost eleven optical stores, where the same refusal was misread as unreadable data.

        So the only accepted cause is data that will never read, which is DETERMINISTIC: it
        recomputes to the same verdict on every attempt, which is what makes the absence
        explainable rather than a matter of when the leg happened to run. Everything else returns
        ``False`` and the caller re-raises, failing the leg with the time axis unmoved — and the
        leg's own retry re-offers the date in order.

        Raises:
            TooManyGivenUpDatesError: Past :data:`MAX_GIVEN_UP_DATES`.
        """
        if is_unreadable_source(exc):
            scope = "unreadable"
        else:
            return False
        given_up.add(date_str)
        entry = {
            "date": date_str,
            "scope": scope,
            # Truncated: this lands in the store's metadata, which every reader of the
            # attribute pays for, and the first line is what identifies the cause.
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }
        given_up_dates.append(entry)
        log.error(
            "[%s] DATA LOSS roi=%s date=%s: the source read failed and this date is SKIPPED, so "
            "its pixels are absent from the mosaic. scope=%s error=%s — recorded on the store as "
            "assessed_unreadable_dates.",
            orbit,
            roi_label,
            date_str,
            scope,
            exc,
            exc_info=True,
        )
        # After the line above, so the date that crossed the ceiling is described as fully as
        # every date before it. No assessed window is written by a leg that stops here: that
        # attribute says the range was examined in full, and this one never reached most of it.
        if len(given_up_dates) > MAX_GIVEN_UP_DATES:
            listed = ", ".join(f"{g['date']}({g['scope']})" for g in given_up_dates)
            raise TooManyGivenUpDatesError(
                f"STOPPING [{orbit}] roi={roi_label}: {len(given_up_dates)} date(s) given up, past "
                f"the ceiling of {MAX_GIVEN_UP_DATES}. Every one of them failed to read for a cause "
                f"that RECOMPUTES — a provider refusing reads is retried in order and never counted "
                f"here — so this many says the objects themselves are bad, not that the service is "
                f"having a bad minute. Dates: {listed}. This error is TERMINAL: re-dispatching "
                f"re-reads the same objects, spends the per-read retry ladder on each, and holds a "
                f"fleet to reach the identical answer. ACTION: check the catalogue for reprocessed "
                f"copies of these dates; until they exist this orbit-year is too damaged to be "
                f"worth a mosaic."
            )
        return True

    def _prepare_batch(rng: SolarDayRange) -> _PreparedBatch:
        """Catalogue query plus lazy graph build for one batch. No cluster reads.

        Runs on the pipeline's background thread when ``pipeline_batches`` is on, so it
        must touch nothing the write loop owns: no credential application, no store
        writes, and only the immutable snapshot of ``written_dates`` taken when the
        pipeline started.
        """
        batch_start_str, batch_end_str = rng.own_start, rng.own_end
        log.info(
            "[%s] Batch %s..%s roi=%s: querying catalog (%s..%s)",
            orbit,
            batch_start_str,
            batch_end_str,
            roi_label,
            rng.query_start,
            rng.query_end,
        )

        # Rebuild the lazy ROI mask per batch so frozen IAM creds inside any embedded
        # boto chain are fresh. Graph construction only; the actual S3 reads happen
        # during the write's compute(), by which point the credential refresh has
        # applied any new session token.
        batch_mask = read_roi_mask(roi_zarr_path, spatial_chunks, storage_options=storage_options)
        # Left LAZY on purpose. persist() materialises every chunk of the full
        # zone grid — for 03S that is 3,706 chunks / ~60 GiB of mostly ocean, pinned for
        # the run — while the only consumer, apply_roi_mask, is written out to the live
        # windows and so touches a handful of them. Dask culls the reads to those chunks,
        # making a per-batch re-read far cheaper than the pin. (S2 does the same, and
        # additionally crops the coverage denominator, which it alone computes.)

        # Wrap the provider to keep the items odc was given. Wrapping rather than querying
        # twice matters: these must be the SAME objects odc grouped, after any timestamp
        # normalisation, or the footprints would describe a different set of acquisitions.
        base_provider = make_s1_item_provider(
            orbit,
            roi.bbox_wgs84,
            rng.query_start,
            rng.query_end,
            use_s3_direct=use_s3_direct,
            mid_longitude=mid_longitude,
        )
        seen_items: list = []

        def _capturing_provider() -> list:
            # OWNERSHIP, applied here rather than after the load. The query deliberately
            # reaches a day past this batch on both sides so that a solar day straddling
            # a boundary is complete for whichever batch owns it; handing those pad-day
            # items to the loader as well would have it build a partial group for the
            # NEIGHBOUR's day, which the write loop would then commit. Filtering first
            # means the loader only ever sees whole days that are ours.
            # Normalise defensively: the provider already does it, but it is injectable
            # and this is the last point before ownership reads a date. Idempotent.
            items = owned_items(normalize_to_solar_day(base_provider(), mid_longitude=mid_longitude), rng)
            seen_items.extend(items)
            return items

        query_started = time.monotonic()
        data, baselines = ingest_tile(
            provider="cmr-asf",
            collection="opera-rtc-s1",
            tile_id=None,
            start_date=rng.query_start,
            end_date=rng.query_end,
            # The snapshot, not the live set: a background thread must not read a set the
            # write loop is mutating. Anything it misses is caught when consuming.
            existing_dates=set(already_present),
            bbox=roi.bbox_wgs84,
            chunks=INGEST_CHUNKS,
            resampling="bilinear",
            groupby="solar_day",
            item_provider_fn=_capturing_provider,
            post_load_fn=amplitude_to_db,
            geobox=roi.geobox,
            # Key the existing-date filter on solar days, matching groupby above and the
            # days the store actually holds. The consume loop re-checks anyway, so this
            # only saves loading a group that is already committed — but filtering on UTC
            # dates would let half of one through, which is the case that costs a write.
            mid_longitude=mid_longitude,
        )
        query_s = time.monotonic() - query_started
        # THE most diagnostic line in this flow, and it did not exist. Without the item
        # count there is no way to tell a zone the source does not cover from a zone whose
        # items we found and then dropped — the two look identical (a leg that completes
        # having written nothing) and they need opposite fixes. ``items`` is post-ownership,
        # so a batch that queried granules and owned none of them says so here rather than
        # vanishing. roi= because at fleet width nothing else identifies the cell.
        log.info(
            "[%s] Batch %s..%s roi=%s: %d owned item(s), loaded=%s, query=%.1fs",
            orbit,
            batch_start_str,
            batch_end_str,
            roi_label,
            len(seen_items),
            data is not None,
            query_s,
        )
        if data is not None:
            data = apply_roi_mask(data, roi_zarr_path, spatial_chunks, roi_mask=batch_mask)
        return _PreparedBatch(
            batch_start_str,
            batch_end_str,
            data,
            baselines,
            query_s,
            len(seen_items),
            _date_footprints(seen_items),
        )

    # depth=1: one batch prepared ahead. A batch's write is one long consume, so a single
    # look-ahead already covers it; more would retain catalogue items to hide nothing.
    # Unpipelined, `pipelined` is bypassed entirely rather than run at depth 0 — the
    # serial path must stay available as a rollback that shares no machinery.
    def _serially() -> Iterator[tuple[_PreparedBatch, float]]:
        """The rollback path, yielding the same ``(prepared, stall)`` shape as the pipeline.

        ``stall`` is the preparation the consumer had to WAIT for, so serially it is the
        whole query — not zero. Reporting zero would make ``hidden`` read as the full query
        and claim the serial path hides everything, which is backwards, and would corrupt
        any A/B using that log line as its instrument.
        """
        for rng in batch_ranges:
            prepared_serial = _prepare_batch(rng)
            yield prepared_serial, prepared_serial.query_s

    prepared_batches = ticked(pipelined(batch_ranges, _prepare_batch, depth=1) if pipeline_batches else _serially())

    # NOT short-circuited on a run of empty batches, and the reason is worth keeping. Zone 23N
    # paginates ~182,000 granules across a year to write nothing, so abandoning such an orbit
    # early is worth roughly 190s -> 59s of catalogue time per zone-year. But "no usable item
    # yet" is ALSO what a summer-only zone looks like in January, and the two are
    # indistinguishable from the item count alone — so any threshold on it drops real radar
    # from seasonally-covered land. The safe signal is the polarisation skip count, which
    # separates them exactly: granules fetched and all rejected for polarisation is a permanent
    # property of the terrain, while zero granules fetched is a window with no acquisitions and
    # may be seasonal. That count is computed in ingest.opera_query and currently only logged;
    # exposing it through the provider is what this optimisation is waiting on.
    for prepared, stall_s in prepared_batches:
        total_items_seen += prepared.items_seen
        batch_start_str, batch_end_str = prepared.start, prepared.end
        data, baselines, query_s = prepared.data, prepared.baselines, prepared.query_s

        refresh_credentials_if_stale()

        if data is not None:
            # The ROI mask was applied in _prepare_batch, on the same side of the phase
            # boundary as the graph build it belongs to.
            write_total_s = 0.0
            written_this_batch = 0

            # A batch holds many NON-contiguous dates, each its own atomic
            # commit — so the retry scope is PER DATE. One retry around the
            # whole loop would restart at a date an earlier attempt already
            # committed and trip the duplicate-date guard.
            for i in range(data.sizes["time"]):
                # Match this slice on its EXACT timestamp — the value odc took from the
                # group's first item — and take BOTH the solar day and the footprint from
                # what was grouped. The slice's own label cannot be trusted as the day:
                # odc stamps it with an item timestamp whose calendar date can be the day
                # BEFORE the solar day wherever the offset crosses UTC midnight, so two
                # solar days can normalise onto one date and collide on the time axis.
                #
                # Unmatched means the footprint is unknown, so fall back to the label —
                # the pre-existing behaviour, and the conservative branch.
                entry = prepared.date_windows.get(str(data["time"].values[i])[:19])
                solar_day, footprint = entry if entry else (None, None)
                date_str = solar_day or str(data["time"].values[i])[:10]
                # THE authority on what has been written, checked here rather than
                # relying on the query filter: under a look-ahead the query for this
                # batch may have been built before an earlier batch committed, so a
                # solar day shared across a UTC batch boundary could arrive twice.
                # Writing it twice would trip the duplicate-date guard mid-run.
                if date_str in written_dates:
                    # DEBUG: on a resume this fires for EVERY date the store already
                    # holds, so a nearly-complete cell logs one line per committed date
                    # and Prefect ships each from the Dask worker to the orchestrator.
                    # The batch summary reports what was written, which is the number
                    # that matters.
                    log.debug("[%s] Skipping date %s: already written", orbit, date_str)
                    continue

                # Given up by an EARLIER batch: the padded query offers a boundary solar day to
                # both batches that touch it. DEBUG because the loss was reported where it
                # happened.
                if date_str in given_up:
                    log.debug("[%s] Skipping date %s: already given up", orbit, date_str)
                    continue

                if footprint is not None and not footprint:
                    # Reaches no live window at all. Writing it would build a full graph
                    # to store nothing, since all-fill chunks are never persisted.
                    # DEBUG: per skipped date; the count is summarised per batch and in
                    # the assessed-window record.
                    log.debug("[%s] Skipping date %s: imagery reaches no live window", orbit, date_str)
                    empty_dates += 1
                    continue
                date_windows = footprint if (narrow_windows_per_date and footprint) else live_windows

                # Stamp the slice with its solar day. The store's axis is day-granular
                # either way, so this only decides WHICH day — and taking it from the
                # grouping is what makes it unique per slice and monotonic across them.
                day_slice = data.isel(time=slice(i, i + 1))
                if solar_day:
                    day_slice = day_slice.assign_coords(time=[np.datetime64(solar_day, "ns")])

                # Inside the per-date loop deliberately: a batch can outlive the
                # credential, so renewing only at batch boundaries is what failed.
                refresh_credentials_if_stale()
                date_started = time.monotonic()
                # S1's source read happens INSIDE this write's compute, so the write retry
                # already covers a transient read here. The asymmetry that cost whole cells
                # was S2's, whose read fires in its coverage gate, outside any retry. What
                # this path lacked is attribution: once the retry is exhausted the exception
                # names neither the zone nor the date.
                #
                # `wait_out` is what makes the retry long enough to be the answer to a provider
                # refusal, which for radar is the ONLY answer: OPERA publishes one copy of a
                # granule, so there is nothing to step down to, and giving the date up would
                # hole the store. Radar alone asks for it — the optical path's response is the
                # copy ladder, and a long wait per rung would multiply with it.
                try:
                    with read_failure_context(log, roi=roi_label, date=date_str):
                        for attempt in store_write_retrying(log, wait_out=is_provider_refusal):
                            with attempt:
                                write_day_windows(
                                    orbit_store,
                                    day_slice,
                                    date_windows,
                                    roi=roi,
                                    manifest=ingest_manifest,
                                    baselines=_baselines_for(baselines, [date_str]),
                                    tile_id=roi_zarr_path,
                                    crs=roi.native_crs,
                                    chunks=INGEST_CHUNKS,
                                    parallel_windows=overlap_window_writes,
                                    s3_region=s3_region,
                                )
                except Exception as exc:
                    # The retry is exhausted. THIS is why one refused read used to take the
                    # zone-year: the exception left the loop, so a date the source would not hand
                    # over cost every LATER date too. Only a failure the source is answerable for
                    # is absorbed — anything else repeats on every date and is repairable here,
                    # so giving up dates one at a time would be the wrong response to it.
                    if not _give_up_date(date_str, exc):
                        raise
                    continue
                date_s = time.monotonic() - date_started
                write_total_s += date_s
                written_dates.add(date_str)
                written_this_batch += 1
                # ``mode`` is the load-bearing field: sequential means this date
                # cost the SUM of its windows' critical paths rather than their
                # maximum, which is the single largest difference between how S1
                # and S2 write today.
                # ``roi=`` is what makes this line attributable at fleet width. A monitor
                # reading the whole log group cannot otherwise tell WHICH cell a commit
                # belongs to: the log stream is the Dask worker's ECS task id, and
                # resolving that to a zone needs a throttled ECS call. Without it the
                # only answerable question is "is the fleet moving", not "which cell has
                # stopped" — see yield-embeddings docs/runbooks/campaign-monitoring.md.
                log.info(
                    "[%s] S1 stage timings roi=%s date=%s: write=%.1fs windows=%d of %d mode=%s",
                    orbit,
                    roi_label,
                    date_str,
                    date_s,
                    len(date_windows),
                    len(live_windows),
                    "parallel" if overlap_window_writes else "sequential",
                )
            # Count what was WRITTEN, not what the batch held: under a look-ahead a date
            # can arrive already committed by an earlier batch and be skipped above, and
            # reporting it as processed would overstate a resumed run's progress.
            n = written_this_batch
            total_processed += n
            log.info(
                "[%s] Batch %s..%s roi=%s: wrote %d dates (total: %d)",
                orbit,
                batch_start_str,
                batch_end_str,
                roi_label,
                n,
                total_processed,
            )
            # `stall` is what the look-ahead failed to hide — how long this batch waited
            # for its own query after the previous batch's writes finished. Near zero
            # means the query hid completely; approaching `query` means it hid nothing,
            # which is the expected reading for the FIRST batch since nothing precedes
            # it. `hidden` is therefore the saving, and it is what to watch: if it stays
            # near zero on later batches the look-ahead is not paying.
            log.info(
                "[%s] S1 batch timings %s..%s roi=%s n=%d: "
                "query=%.1fs hidden=%.1fs stall=%.1fs write=%.1fs per_date=%.1fs",
                orbit,
                batch_start_str,
                batch_end_str,
                roi_label,
                n,
                query_s,
                max(query_s - stall_s, 0.0),
                stall_s,
                write_total_s,
                (stall_s + write_total_s) / n if n else 0.0,
            )

    # Record the range examined IN FULL, so a month absent from this store reads as a
    # finding rather than a gap (see storage.zarr_store.record_assessed_window). Only when a
    # store exists: with nothing written there is no store to annotate, and that case is
    # already unambiguous — no store means the orbit is absent and callers downgrade.
    #
    # "A store exists", NOT "this invocation wrote a date". A run interrupted after its
    # last commit but before this line leaves the orbit store complete and unannotated; the
    # resume then dedupes every date away, writes nothing, and would skip the record again
    # on every retry, leaving a legitimately empty month permanently indistinguishable from
    # a gap. `empty_dates` is 0 on such a resume, which is honest — this pass examined no
    # date — and the gate does not read it. The probe runs ONLY when nothing was written.
    if total_processed or get_existing_dates(orbit_store, s3_region=s3_region):
        record_assessed_window(
            orbit_store,
            start_date,
            end_date,
            empty_dates=empty_dates,
            # A given-up date exists ONLY here. Everything else about it is a log line, and a
            # log line is not read by the coverage gate or by a later resume, so losing this
            # write loses the fact that the date was ever examined.
            required=bool(given_up_dates),
            # Written unconditionally, so a clean re-run clears an earlier run's list rather
            # than leaving dates advertised as lost that are now present.
            unreadable=given_up_dates,
            get_credentials=None,
            s3_region=s3_region,
        )

    if given_up_dates:
        # Re-stated at the END: the per-date line is hundreds of lines back, and this is the one
        # that says a green leg is nonetheless missing pixels, and where.
        log.error(
            "[%s] DATA LOSS SUMMARY roi=%s: %d date(s) skipped because their source reads "
            "failed — %s. Recorded on the store as assessed_unreadable_dates. Every one of them "
            "failed for a cause that RECOMPUTES, so re-running this window will not recover any "
            "of them: what they need is a reprocessed copy at the provider.",
            orbit,
            roi_label,
            len(given_up_dates),
            "; ".join(f"{g['date']} scope={g['scope']}" for g in given_up_dates),
        )

    if total_processed == 0:
        # A leg that writes nothing used to complete SILENTLY, and that silence is what made
        # five zones undiagnosable: `status="skipped"` is a success to the parent, so the
        # cell finished green with an orbit absent from the store and no line saying why.
        # WARNING and not INFO because it is nearly always worth a human's attention — the
        # legitimate case (the source does not cover this ROI for this orbit) is rare, and
        # indistinguishable from the illegitimate one WITHOUT this line naming which it was.
        # ``already_held`` is what keeps this from crying wolf on a RESUME. A leg re-run over a
        # store that already holds everything it can hold legitimately writes nothing, and a
        # warning that fires on every such run is one the reader learns to skip — the same
        # failure as the empty-commit warning this file's assessed-window record used to emit.
        already_held = len(written_dates)
        log.log(
            # A resume with nothing left to add is routine; one that GAVE UP dates is not,
            # whatever it already held.
            logging.INFO if already_held and not given_up_dates else logging.WARNING,
            "[%s] WROTE NO DATES roi=%s window=%s..%s: items_seen=%d empty_dates=%d "
            "given_up=%d already_held=%d — items_seen=0 means the source covers this ROI for "
            "this orbit not at all; items_seen>0 with already_held>0 is a resume that had "
            "nothing left to add; items_seen>0 with already_held=0 means items were found and "
            "none survived to a commit; given_up>0 names how many of those the source refused "
            "or could not read",
            orbit,
            roi_label,
            start_date,
            end_date,
            total_items_seen,
            empty_dates,
            len(given_up_dates),
            already_held,
        )
        if given_up_dates and not already_held:
            # The WARNING above is not enough on its own. `status="skipped"` reads to the
            # parent as "the source does not cover this orbit", and for `s1_orbit="both"`
            # that lets the cell finish with an orbit missing and inference run on optical
            # alone. A leg that gave up EVERY date it had, onto a store that holds nothing,
            # is data loss wearing the same clothes as absence — and the two must not be
            # returned identically. TERMINAL: every date here failed for a cause that
            # recomputes, so a re-dispatch reads the same objects to the same answer.
            raise TooManyGivenUpDatesError(
                f"[{orbit}] roi={roi_label} window={start_date}..{end_date} gave up every one "
                f"of its {len(given_up_dates)} date(s) and committed none, so no store exists "
                f"to record the loss on. This is TERMINAL, not a request to come back: each of "
                f"these failed for a cause that recomputes, so re-reading them costs the "
                f"per-read ladder and a fleet to reach the identical answer. ACTION: check the "
                f"catalogue for reprocessed copies. "
                f"Given up: {'; '.join(f'{g["date"]} scope={g["scope"]}' for g in given_up_dates)}"
            )
        return SarIngestResult(
            roi_path=roi_zarr_path,
            status="skipped",
            dates_processed={orbit: 0},
        )

    return SarIngestResult(
        roi_path=roi_zarr_path,
        status="success",
        dates_processed={orbit: total_processed},
    )
