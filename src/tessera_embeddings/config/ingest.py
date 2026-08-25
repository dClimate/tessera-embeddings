"""Configuration for the satellite data ingestion pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from tessera_embeddings.config.code_identity import source_identity

#: Source whose change alters which dates and pixels a MOSAIC holds.
#:
#: The two leg entry points seed it and the import closure pulls in what they delegate
#: to — the catalogue query, the solar-day normalisation, the duplicate-copy choice, the
#: OPERA granule filter, the validity gate. Seeding with the entry points rather than a
#: hand-picked module list is what keeps it from going stale the next time one of them
#: takes a new import.
#:
#: Deliberately NOT the whole ``ingest`` package: ``land_mask`` decides which tiles are
#: live rather than what a mosaic contains, and it is already fingerprinted separately
#: as ``coverage_sha256``. Including it would make a mask rebuild look like an ingest
#: code change.
_MOSAIC_CONTENT_SOURCES: tuple[str, ...] = ("ingest/s1_roi.py", "ingest/s2_roi.py")


def ingest_code_identity() -> str:
    """Fingerprint of the code that decides a mosaic's CONTENT.

    Recorded in each mosaic store's :class:`~tessera_embeddings.storage.manifest.
    IngestManifest` on its first commit, and re-validated on every later batch write.
    That is what makes resuming an interrupted store safe: a store holds dates its own
    code produced, and appending dates produced by different code — a different
    duplicate-copy preference, a different validity gate, a different query — would
    leave one mosaic built two ways with a single fingerprint stamped over it.

    The check is on APPEND, not on the completion marker, and the difference is the
    point. Putting this in the marker fingerprint would make every ingest change declare
    every finished mosaic stale, and there are petabytes of them; putting it on the
    append only ever rejects work that was already incomplete.
    """
    return source_identity(_MOSAIC_CONTENT_SOURCES, "ingcode")


# Spatial chunk size for storage (written at ingest). 4096 aligns with the
# embedding store's 2048-px shard grid: one ingest chunk is exactly 2x2 shards
# (and 16x16 of the 256-px inner chunks). Still larger than the inference
# read-tile size to keep the satellite-ingest Dask graph small. Both pipelines
# now read the same 2048-px tile (config.inference.INFERENCE_CHUNK_SIZE), so the
# whole chain divides evenly end to end: 4096 ingest chunk -> 2048 inference tile
# -> 2048 output shard -> 256 inner chunk, with no rechunk at any hop. Inference
# still reads sub-tiles via .oindex and imposes no alignment requirement of its
# own; the alignment is what makes every read whole-object rather than partial.
#
# NOTE: this changed 4000 -> 4096. Existing 4000-px stores stay readable, but
# appends to them under this config are rejected by the RoiManifest chunk-size
# check (structural-param mismatch) — finish an old campaign on the old config
# or re-ingest.
INGEST_CHUNK_SIZE = 4096

INGEST_CHUNKS = {"time": 1, "northing": INGEST_CHUNK_SIZE, "easting": INGEST_CHUNK_SIZE}

# The load side deliberately uses the SAME block size as the store. A coarser load
# block (8192) was shipped once to shrink graph task counts for the scheduler's
# single-threaded dispatch, and was removed: fewer, larger read tasks cap a date's
# parallel width at blocks x bands, which starves any fleet wider than that. The
# width cost is paid on every compact ROI while the dispatch saving only binds on
# the densest zones at wide fleets. History, measurements and the cost model are in
# context_docs/design/ingest_optimization_campaign_2026_07.md (section 3.5 and its
# 2026-07-27 correction); do not reintroduce a load/store split without re-reading
# that record.

# Manifest sharding for the campaign mosaics. Without it every commit rewrites the
# whole array manifest, so the bytes rewritten grow with the number of dates already
# written — the shape that makes a zone-YEAR disproportionately worse than a month.
#
# TIME ONLY, and that is the load-bearing detail. One commit here is one DATE, and a
# date writes every live window — i.e. essentially the whole live area. So spatial
# shards cannot localise a commit: every commit touches nearly all of them, and all
# they add is object count. A spatial split of 4x4 on a 6-degree zone means ~4,000
# tiny manifest objects rewritten per commit instead of ~14, and the latency of those
# PUTs costs far more than the bytes saved. A time split is what localises a per-date
# commit: it rewrites only the dates sharing its shard rather than every date so far.
#
# The size trades write cost against read cost: smaller shards rewrite less per commit
# but a reader spanning a long window opens more of them. 8 caps the rewrite at eight
# dates' references while keeping a year to ~32 shards.
INGEST_MANIFEST_SPLIT = {"time": 8}

# S2 per-solar-day keep threshold for the GLOBAL CAMPAIGN ingest path, as a
# PERCENT of ROI pixels that must be valid (cloud-screened) to keep a solar
# day — the same percent scale as ingest/roi_processing.py's same-named
# constant, but tuned to 0.1% because a whole-zone ROI is mostly ocean/edge
# (the single-ROI default of 5% would drop nearly every date). The duplicate
# constants should eventually be unified under one name with per-path
# defaults.
DEFAULT_MIN_VALID_COVERAGE = 0.1

# Date batching is enabled per ROI by SIZE, because its benefit is not monotonic in
# size: it amortises the per-batch commit and cannot speed the write (which is
# fleet-bound), so it pays only where the commit is a large share of a date's cost,
# and it LOSES where a bigger write graph crowds out the preparation overlapping it.
#
# A THRESHOLD, not an interpolated curve. Deliberately set at the upper edge of the
# range where batching was measured to win, not at the estimated crossover: widening it
# means measuring an ROI in between, not extrapolating. Measurements and the per-size
# ratios are in context_docs/design/ingest_optimization_campaign_2026_07.md.
#
# Denominated in the COVERED chunk area of a run's live windows — the area the write
# graph actually touches, and already known once windows are merged, so deriving this
# costs no extra I/O.
#
# COUPLED TO THE WINDOW MERGE COST, and that is a trap worth naming: covered area is an
# output of the merge, so changing ``live_windows.WINDOW_COST_IN_CHUNKS*`` moves every
# ROI along this axis without anyone touching this constant. A finer merge covers less
# area, so ROIs drift DOWN and more of them batch. Recalibrate this against runs, not
# against an offline sweep at a different cost, whenever that exchange rate changes.
AUTO_BATCH_DATES_MAX_COVERED_CHUNKS = 500

# Dates fused per graph when batching turns on. The only batch size measured.
AUTO_BATCH_DATES = 4


def auto_batch_dates(covered_chunks: int) -> int:
    """Dates to fuse for a run whose live windows cover ``covered_chunks`` chunks."""
    return AUTO_BATCH_DATES if covered_chunks <= AUTO_BATCH_DATES_MAX_COVERED_CHUNKS else 1


class IngestSettings(BaseModel):
    """Shared ingest tuning knobs, grouped for Prefect flow signatures.

    One model consumed by every campaign flow that carries ingest tuning —
    ``ingest-zone-year`` (which fans the values out to the base S1/S2 ingest
    flows), ``run-global-campaign``, and ``fill-zones-sequential`` — so the
    knobs stay one nested object in dispatch parameter dicts and render as one
    collapsible group in the Prefect UI, instead of four flat copies per
    signature.

    Defaults are campaign-scale. The single-ROI path (``tessera-full-pipeline``
    and the base ingest flows) keeps its own flat knobs for now: its worker
    bounds are ``None``-means-auto-size (auto-derived from ROI chunk count)
    and its coverage default is 5% vs the campaign's 0.1% (see
    :data:`DEFAULT_MIN_VALID_COVERAGE`), so adopting this model there first
    requires reconciling those defaults.
    """

    # Dask worker bounds for one (zone, year) ingest.
    #
    # ``max_workers`` is the S2 fleet's width, and it is what the campaign's cost model is
    # denominated in: at 60, an S2 fleet plus both S1 orbits comes to the per-cell vCPU
    # figure every quota ask and wall-clock forecast is built from. It shipped lower than
    # the planned figure for a while, so code and plan disagreed in opposite directions —
    # the plan over-stated what a cell costs and therefore under-stated how many fit inside
    # a quota. Keep this and the campaign's per-cell vCPU figure in step; changing one alone
    # silently invalidates the other.
    #
    # ``min_workers`` defaults to None meaning **follow the derived width**, i.e. a
    # fixed-size fleet. A floor of 1 lets ``cluster.adapt`` retire workers in every
    # inter-date gap and relaunch them cold into the next write, which costs a
    # material share of the width being paid for: Fargate boot latency lands inside
    # writes, and each new worker starts with cold GDAL/HTTP caches.
    #
    # Adaptivity buys nothing to offset that here, because these fleets are busy for
    # essentially the whole run — there is no idle stretch for a lower floor to
    # reclaim. Set a floor below the derived width only for a fleet that genuinely
    # idles. Measurements are in
    # context_docs/design/ingest_optimization_campaign_2026_07.md.
    min_workers: int | None = Field(default=None, ge=1)
    max_workers: int = Field(default=60, ge=1)
    # Each S1 orbit's fleet width as a FRACTION of the S2 fleet's, because S1's work is a
    # fixed fraction of S2's rather than a fixed size: measured at 15.5-18.1% of S2 across
    # a 13x span of zone density, so one ratio holds where an absolute count would not.
    #
    # An absolute count would also break the chunk-scaled sizing below it — a sparse zone
    # scales S2 down toward the floor, and a fixed S1 width could then exceed S2's.
    #
    # Sized so S1 finishes comfortably INSIDE S2's runtime. S1 must never become the cell's
    # critical path: it would extend every cell while wasting the headroom the ratio buys.
    # Calibration and the margin behind this number live in context_docs; re-derive it
    # against runs if S1's per-date cost changes again.
    s1_worker_fraction: float = Field(default=0.22, gt=0.0, le=1.0)
    # S2 per-solar-day keep threshold (PERCENT, so 0-100; see
    # DEFAULT_MIN_VALID_COVERAGE). Bounded, not just typed: a negative threshold is
    # satisfied by every date including one with zero valid pixels, which then counts
    # toward the month-presence gate and lets an empty year be filled and permanently
    # tagged complete. Above 100 nothing is ever kept.
    # gt=0, not ge=0: the gate is `coverage >= min_valid_coverage`, so a threshold of
    # zero admits a date with NO valid pixels. That date's timestamp then counts toward
    # the month-presence check, and an entirely empty year can be filled and
    # permanently tagged complete. Zero is not "keep everything" here, it is
    # "keep things that are not there".
    min_valid_coverage: float = Field(default=DEFAULT_MIN_VALID_COVERAGE, gt=0.0, le=100.0)
    # S1 CMR query batch window, in days. Must be >= 1: the S1 loop advances
    # batch_start by this much, so 0 never advances (an ingest cluster billing
    # forever) and a negative walks it backwards.
    batch_days: int = Field(default=30, ge=1)
    # How many times ingest-zone-year will dispatch a single LEG (optical, or one radar
    # orbit) before giving up on it. 1 disables retrying.
    #
    # Retrying is safe because a re-dispatch RESUMES — dates already committed are skipped,
    # not rewritten — so the cost is only the work actually lost. On by default because the
    # alternative leaves a mosaic incomplete until a human notices and re-dispatches.
    #
    # Failures a re-dispatch cannot fix are excluded by CLASS rather than by count — see
    # ``_NON_RETRYABLE_LEG_MARKERS`` in the flow. This counts ATTEMPTS only; the bound on
    # elapsed time in the same loop is ``max_leg_wall_clock_s`` below.
    max_leg_attempts: int = Field(default=3, ge=1)
    # Wall-clock budget, in seconds, for ONE leg's retry loop — the only bound on ELAPSED
    # time anywhere in the retry stack. Every other layer counts attempts:
    # the HTTP ladder under a page fetch, this loop's attempt count above, and the cell and
    # zone budgets above that, each treating the layer below as one try. None of them reads
    # a clock, and expansive backoff — the campaign's deliberate policy toward a source
    # refusing reads under load — makes the clock the one axis that can grow without limit.
    #
    # It bounds the decision to START ANOTHER ATTEMPT, never a leg that is running: the
    # deadline is checked only where the loop decides whether to re-dispatch a failed leg,
    # so a slow-but-succeeding leg is never why the loop stopped, and the loop's true worst
    # case is this deadline plus one final attempt. Failing the cell here is not surrender:
    # the cell returns to the campaign work list, and a later re-dispatch RESUMES from the
    # dates already committed (Icechunk commits a date's time slot atomically with its
    # pixels), so the cost of the bound is latency, not lost work.
    #
    # The default is a POLICY choice pinned between two facts. It must comfortably exceed
    # the longest legitimate single leg at the default fleet width, so that a leg which ran
    # long for honest reasons never costs a failed sibling its retry; and it must be small
    # enough that a stuck cell releases its campaign slot promptly once it has outlived that
    # legitimate range — no bound can release it sooner without cutting legs that are merely
    # slow. Calibrate it against measured leg durations, which live in context_docs/design/
    # (the ingest optimisation campaign record and the catalogue refusal record); re-derive
    # it if per-date cost or the default width changes.
    #
    # LOWERED 2026-08-11, from 36 h. The old value was sized to DESCRIBE the worst case the
    # attempt budgets could already reach, which is the wrong question: it made the ceiling
    # knowable but expressed no judgement about what is worth waiting for. Two facts push it
    # down. A leg holds a Dask fleet while it retries, so the wait is billed, not free — the
    # attempt count (3) is what bounds the ordinary case, and this deadline only ever binds
    # on a cell already behaving pathologically. And failing costs LATENCY, NOT WORK: the
    # cell returns to the work list and resumes from its committed dates, so giving up early
    # is cheap while giving up late holds a campaign slot against an eight-day run. 6 h still
    # clears three legs of a slow dense cell plus expansive backoff.
    #
    # PER LEG since 2026-08-18, when the legs stopped sharing one retry loop. Each leg's
    # deadline is anchored at its own first dispatch, so one leg's slow retries can no longer
    # spend a sibling's budget. The value is unchanged: it was always sized against the
    # longest legitimate SINGLE leg, so it already meant this.
    max_leg_wall_clock_s: int = Field(default=6 * 3600, ge=1)
    # Base delay, in seconds, before re-dispatching a failed leg. Doubles per attempt and is
    # capped at four times this value.
    #
    # It exists because the legs retry INDEPENDENTLY (2026-08-18). While they shared a loop,
    # a retry was spaced by however long the slowest sibling took to settle — accidental
    # backoff, invisible, and sometimes an hour. Removing the barrier removed that spacing,
    # so the spacing became a decision.
    #
    # Deliberately SHORT. A retry resumes from the dates already committed, so it costs only
    # the work actually lost, and the campaign's throughput is bounded by legs landing — a
    # long backoff idles a GPU fleet waiting on a mosaic. What the delay buys is the
    # difference between a momentary source refusal and a structural one, and that does not
    # take minutes to establish; a failure that is deterministic in the input is excluded by
    # CLASS (``_NON_RETRYABLE_LEG_MARKERS``) and never waits at all.
    leg_retry_backoff_s: int = Field(default=30, ge=0)
    # Delay before re-dispatching a leg the SOURCE PROVIDER refused reads to, in seconds. 0 falls
    # back to `leg_retry_backoff_s`. Doubled and capped exactly as that one is.
    #
    # **Corrected in place.** The comment above says the difference between a momentary source
    # refusal and a structural one "does not take minutes to establish". For a catalogue query it
    # does not. For object reads it does: ASF refused radar reads for about six minutes on
    # 2026-08-24 and roughly thirteen on 2026-08-21, and both times the re-dispatches walked
    # straight back into a source still refusing and spent the whole attempt budget in two
    # minutes. Deciding structural-versus-momentary for a read takes longer than the leg backoff
    # was ever willing to wait.
    #
    # LONG deliberately, where the other is short deliberately, and the difference is who is
    # holding a fleet. A failed leg has released its Dask fleet, so waiting here costs latency
    # and nothing else — where the same patience spent inside the leg holds hundreds of vCPU
    # idle. This is the cheap place to be patient, so it is where the patience goes. It applies
    # to ONE class, recognised by `errors.ProviderRefusedReadsError` reaching the leg's failure
    # detail, and every other failure keeps the short backoff.
    leg_refusal_backoff_s: int = Field(default=600, ge=0)
    # Seconds of extra wall clock a leg EARNS by committing dates, each time
    # `max_leg_wall_clock_s` would otherwise refuse its next attempt. 0 turns it off, which
    # restores the plain deadline exactly.
    #
    # The deadline above bounds PATIENCE — wall clock a leg spends not getting anywhere — and it
    # says so: "this deadline only ever binds on a cell already behaving pathologically". Counted
    # from the leg's first dispatch, it could not keep that promise. It charges a leg for the
    # productive work of every prior attempt, so a leg that committed steadily for hours and then
    # hit a transient failure is refused the attempt that would have resumed from those hours,
    # on the same terms as a leg that achieved nothing.
    #
    # A leg whose store has GAINED DATES is not the cell the deadline was written for. It resumes
    # from what it committed, so its next attempt starts further along than its last one did, and
    # the campaign slot it holds is buying something.
    #
    # Bounded by construction rather than by a second lever, and all three bounds carry weight: a
    # grant costs this fixed amount; grants are limited to the number of re-dispatch decisions a
    # leg has, which is one fewer than `max_leg_attempts`; and each one has to be PAID FOR by
    # dates committed since the previous grant, so a store that stops growing stops earning them.
    # The ceiling is therefore `max_leg_wall_clock_s + (max_leg_attempts - 1) * this`, and a leg
    # that commits nothing never leaves `max_leg_wall_clock_s`.
    #
    # Sized as a rung of patience rather than as a whole extra leg: what a grant has to cover is
    # the backoff plus the START of one more attempt, since a running leg is never judged
    # against the deadline at all.
    leg_progress_extension_s: int = Field(default=3600, ge=0)
    # Width, in seconds, of the window each ingest leg spreads its FIRST dispatch over. 0 is off.
    #
    # A cell's legs are dispatched the moment the cell is, and a campaign starts every cell it
    # has slots for at once. Each leg then spends about the same time provisioning before its
    # first catalogue request, so the fleet arrives at the catalogue in phase — a burst whose
    # instantaneous rate a source can refuse even though every individual request is one it
    # serves happily. A campaign resuming from committed dates never showed this, because its
    # cells were already at random phases; a run from a clean store has no such spread and has
    # to be given one.
    #
    # The offset is DERIVED from the leg's identity rather than drawn at random, so a leg's
    # delay is reproducible and a test can assert it. Legs are staggered, not serialised: they
    # all still run concurrently, each merely starting at its own point in the window, and only
    # the first attempt is offset — a retry is already spread by `leg_retry_backoff_s` and by
    # whenever the failure happened to land.
    #
    # The window is what is configured, rather than the spacing, because a leg cannot know how
    # many other legs are running. Spacing follows: legs sharing the window land roughly
    # `leg_stagger_window_s / n` apart. The default is sized for the campaign's ordinary width;
    # widen it if the fleet grows, and note it is paid as latency on every leg's first dispatch,
    # not only on a cold start.
    leg_stagger_window_s: int = Field(default=600, ge=0)
    # Optional base URI (an fsspec target, e.g. s3://.../perf/) for capturing a
    # Dask ``distributed.performance_report`` per child ingest. Default None =
    # off (normal runs pay nothing); set it only on a probe rung — ingest-zone-year
    # composes a unique per-child filename (``s2.html``, ``s1-<orbit>.html``) under it.
    perf_report_uri: str | None = None

    @model_validator(mode="after")
    def _worker_bounds_ordered(self) -> IngestSettings:
        """Reject max_workers < min_workers.

        The cropped path's fleet sizing clamps into ``[max(min_workers, floor),
        max_workers]``, so inverted bounds make the floor win and the derived cap
        silently EXCEEDS the configured maximum — the one number an operator sets
        to bound spend. The uncropped path just hands the provider a nonsense
        range. Cheaper to refuse the config than to reconcile it downstream.
        """
        if self.min_workers is not None and self.max_workers < self.min_workers:
            raise ValueError(
                f"max_workers ({self.max_workers}) is below min_workers ({self.min_workers}); "
                "the worker bounds must be orderable."
            )
        return self

    def floor_for(self, derived_max: int) -> int:
        """The fleet's minimum, given the width actually derived for one leg.

        ``None`` means follow the derived width, which is what makes a fleet fixed-size. It
        resolves against the DERIVED max rather than ``max_workers`` because each leg gets its
        own width — a sparse zone's S2 fleet is clamped below the configured maximum, and each
        S1 orbit is a fraction of S2 — so resolving against the configured value would ask for
        more workers than the leg was sized for and never reach its own minimum.
        """
        return derived_max if self.min_workers is None else min(self.min_workers, derived_max)
