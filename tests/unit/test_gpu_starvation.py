"""Will a cluster's GPU fleet starve, and which start policy prevents it?

**Modelled, not measured, on purpose.** Instrumenting this at campaign scale would need
spare capacity the campaign does not have, and the inputs are already established: real
per-zone live-tile counts and the real cluster mechanics in :mod:`zone_density`, the ingest
duration basis from the ingest estimate, and per-density inference rates from the GPU
saturation profile. What is left is arithmetic over a timeline, which is what this does.

**Why starvation and not storage.** A mosaic backlog is recoverable — it costs S3 and can be
drained. A starved GPU fleet bills a large on-demand fleet to produce nothing, and **nothing
reclaims it**: a live Ray actor holds its worker, so the autoscaler's idle-down protects the
inter-zone seams and does not bound this. That asymmetry is deliberate, because cold starts
are expensive, and it is exactly why the fleet must be kept fed rather than allowed to drain
and recover.

**One cluster is the whole question.** Eight clusters are eight independent copies of the
same machine plus two shared gates, and a fleet starves per cluster.

The timeline model:

* **Supply** — ingests run ``look_ahead`` at a time in the cluster's dealt order, each
  finishing after a duration proportional to its live tiles, then dropping that zone's
  pixels into the cluster's queue.
* **Demand** — the actor pool is a continuous consumer at ``n_actors x px/s``. The chained
  fill's work-stealing queue means inference is not zone-serial: actors pull from whatever
  has landed, so a queue is the right abstraction.
* **Starvation** — actors alive, queue empty. Billed, produces nothing.

The finding these tests pin is counter-intuitive and is the reason the module exists: **a
WIDER ingest look-ahead makes duty cycle worse**, because it pulls smaller zones into the
opening window, one of them lands early, and the fleet boots against a shallow queue.

**The file also carries the campaign's inference cost line.** Unlike the starvation half it
is measured, not modelled: the per-cell radar-telemetry measurements are first-class tables
below, and the combined-token rate and the GPU-hour interval are derived from them live
rather than quoted. :func:`test_cost_report` prints the line and its interval. The
measurement history behind the tables is in `campaign-cost-model.md` §6b-6c and
`campaign_inference_profile_2026_08.md`.
"""

from __future__ import annotations

# The reports (`test_report`, `test_cost_report`) print by design: they are diagnostics whose
# output IS their product, run with `-s` to regenerate the numbers behind a planning decision
# rather than quoting a document that may have drifted. Same contract as the repo's other
# operator-facing tools.
# ruff: noqa: T201
from collections.abc import Sequence
from statistics import fmean
from typing import NamedTuple

import pytest

from tests.unit.zone_density import ZONE_TILES, plan

PX_PER_TILE = 2048 * 2048

# --- ingest duration ---------------------------------------------------------------------
# Both fits are from the CANONICAL ingest record,
# `ingest_optimization_campaign_2026_07.md`.
#
# DENSITY — five regions, k=1 (one commit per date), each arm launched together so time-of-day
# and catalogue conditions cancel:
#
#     26S     19 chunks   10.4 s/date        fit: 10.16 + 0.06022 * live_4096_chunks
#     59S    188 chunks   32.8 s/date        R2 = 0.954
#     21N    644 chunks   35.9 s/date
#     35N  2,415 chunks  175.6 s/date   (91 dates — the best-powered row)
#     47N  2,418 chunks  138.4 s/date   (4 dates — weak)
#
# WIDTH — the fitted width model on a dense zone: T(W) = 36.3 + 7896/W s per date.
#
# The two agree: the density table's 35N figure of 175.6 s/date sits within 5% of the width
# model's 167.9 s at 60 workers, which is what dates the density measurements to ~60w. So the
# density fit supplies the shape and its width is scaled by the width model's ratio.
#
# TWO honest limits, both from the source:
#
# * The intercept is a real FIXED per-date cost — 10.2 s/date, ~1.0 h per zone-year — so a
#   6-tile island costs about an hour, not minutes. No floor needs inventing.
# * Area does NOT determine duration tightly. 35N and 47N differ by 3 chunks in 2,418 and by
#   27% in per-date time, and per-zone residuals run to +-35%. This is the same limitation
#   `campaign-cluster-sizing.md` records as "the balance is spatial, not temporal". Treat every
#   per-zone duration here as +-35%, and the conclusions as ones that survive that.
DENSITY_FIXED_S = 10.16
DENSITY_PER_CHUNK_S = 0.06022
#: The width the density table was measured at, inferred from its 35N row agreeing with the
#: width model there.
DENSITY_FIT_WORKERS = 60
#: 2048-px tiles per 4096-px ingest chunk. The cluster planner counts tiles; ingest counts chunks.
TILES_PER_CHUNK = 4
DATES_PER_ZONE_YEAR = 365  # a full-height zone sees imagery daily

#: S2 fleet width. 50 is the SHIPPED default; the campaign plan's RECOMMENDATION is 60
#: (`campaign-cost-model.md` §4-5). Worker-hours are width-neutral, so a narrower fleet frees
#: vCPU to fit more concurrent cells — but past the cell-count knee the year barrier makes
#: extra cells worthless, so the freed vCPU buys nothing and width becomes the better place
#: to spend it. Both are kept below as widths to check.
SHIPPED_WORKERS = 50

#: Widths to check — NOT an uncertainty band: the plan's open width choices.
#:
#: A quoted ingest duration is meaningful only with the width it was measured at — the width
#: model spans ~2x across the widths documents quote — so reconcile two documents on width
#: before reading them as disagreeing.
WIDTHS = {"shipped 50w": 50, "recommended 60w": 60, "if the width holds 80w": 80}


def _width_scale(workers: int) -> float:
    """Per-date time at ``workers`` relative to the density table's own width."""
    return (36.3 + 7896 / workers) / (36.3 + 7896 / DENSITY_FIT_WORKERS)


def seconds_per_date(tiles: int, workers: int = SHIPPED_WORKERS) -> float:
    chunks = tiles / TILES_PER_CHUNK
    return (DENSITY_FIXED_S + DENSITY_PER_CHUNK_S * chunks) * _width_scale(workers)


def ingest_hours(tiles: int, workers: int = SHIPPED_WORKERS) -> float:
    """Hours to ingest one zone-year. The fixed term supplies the floor; none is invented."""
    return seconds_per_date(tiles, workers) * DATES_PER_ZONE_YEAR / 3600


# --- inference cost basis ------------------------------------------------------------------
# The measured figures in this section come from CloudWatch `CHUNK_SUMMARY` radar telemetry
# (`t_s1_asc`/`t_s1_desc`); the query window, corpus size and derivation are in
# `campaign_inference_profile_2026_08.md`, the resulting cost basis in
# `campaign-cost-model.md` §5-6. Latitudes are derived from tile rows via the campaign land
# mask, never guessed from a zone's name — a UTM zone is a longitude band and its live tiles
# can span most latitudes.
#
# UNITS. The encoder consumes one sequence per pixel, so its cost scales with tokens:
# `tokens = pixels x (t_kept + t_s1_asc + t_s1_desc)`. The failure mode this section guards
# against is dividing a token census that counts S2 PLUS S1 by a rate measured in OPTICAL
# tokens only. The invariant it protects: BOTH SIDES OF THE CENTRAL DIVISION MUST COUNT THE
# SAME MODALITIES. The unit lives in the constant names, because a rate named bare
# `TOK_PER_SEC` is how a modality mismatch sits in the central division unnoticed.
#
# CONVENTION. Depths here are chunk-array depths (a date counts if ANY pixel in the chunk
# kept it), which overstate what the model processes per pixel. The overstatement CANCELS in
# `depth / rate` because the rates are chunk-array tokens over measured seconds on the same
# chunks — numerator and denominator must stay on one convention, the same invariant one
# level down. The coverage census counts on a THIRD convention, so census depths are not
# comparable to these without a convention bridge (`campaign-cost-model.md` §6b). Reading
# `s1_asc_obs_count`/`s1_desc_obs_count` off a completed store would close the residual
# convention gap; it costs a read, not a run.


class RadarCell(NamedTuple):
    """One cell measured under radar telemetry — the rows the combined-token basis rests on.

    Depths are pixel-weighted chunk-array tokens/px over the cell's chunks — never medians,
    because the rate they divide is a pixel-weighted quantity and mixing the two weightings
    corrupts the quotient. Rates are tokens per second per actor. A new radar-telemetry cell
    is one row to add.
    """

    cell: str  # zone/year
    abs_lat: str  # |latitude| range of its chunks, degrees — from tile rows, not the zone name
    actors: int  # fleet width the rate was measured at
    n_both: int  # chunks with both S1 orbits
    n_one: int  # chunks with one S1 orbit
    n_free: int  # radar-free chunks
    optical_tok_px: float  # both-orbit chunks
    radar_tok_px: float  # both-orbit chunks, ascending + descending together
    radar_one_tok_px: float  # one-orbit chunks
    combined_tok_s_both: float
    combined_tok_s_one: float
    optical_tok_s_both: float
    optical_tok_s_one: float


#: The both-orbit cells the combined-token basis rests on. A stream that re-fills a cell
#: against the same mosaic carries the same chunk population and is EXCLUDED as a duplicate;
#: repeat fills agree to well under 1%, the instrument's run-to-run noise floor, which is why
#: the tolerances below treat sub-1% effects as noise.
#:
#: Limits of the table, recorded here rather than modelled (evidence and the ruled-out
#: explanations: `campaign-cost-model.md` §6b-6c):
#:
#: * One cell is slower per combined token than the rest and the deficit is UNEXPLAINED —
#:   the obvious mechanical causes were each ruled out by measurement, and it is not an
#:   artifact of a partial sample. It is the largest single driver of the interval's width.
#: * NO both-orbit rate exists at campaign fleet width. The widest-fleet cell is also the
#:   slowest, which looks like an actor-count penalty and must not be read as one: that cell
#:   also carries the unexplained deficit above, so width and geography are perfectly
#:   confounded in this table. Separating them needs a second wide run in a different zone.
#:   The whole cost line scales inversely with this rate.
#: * The coverage census's combined depth is close to the measured one at high latitude, but
#:   its optical-versus-radar SPLIT is near-inverted there, and `zone_work_weight` builds the
#:   zone-to-cluster partition on those per-band censused counts. Flagged, not fixed: the
#:   partition needs only between-band RATIOS, whose measured dynamic range matches the
#:   censused one — probably not urgent, but unchecked.
RADAR_TELEMETRY_CELLS: dict[str, RadarCell] = {
    # The equatorial (Sumatran) end of zone 47S — the measured chunks, not the whole zone.
    "p4d2-47S-2020-w16": RadarCell(
        "47S/2020", "0.0-3.7", 20, 228, 7, 0, 71.0, 66.3, 31.0, 2.499e6, 2.650e6, 1.292e6, 1.747e6
    ),
    # Chukotka and the western Aleutians.
    "p4d2b-60N-2020-w16-run2": RadarCell(
        "60N/2020", "51.3-69.9", 20, 326, 199, 2, 137.4, 76.4, 44.1, 2.431e6, 2.489e6, 1.562e6, 1.946e6
    ),
    # The Arctic-Russia end of zone 37N.
    "assembly-dense-37N-2021-w160": RadarCell(
        "37N/2021", "65.1-69.1", 55, 45, 96, 0, 163.6, 100.0, 38.8, 2.579e6, 2.555e6, 1.601e6, 2.051e6
    ),
    # The WHOLE of 37N/2021 south of 32 degrees — the Levant to East Africa. The bulk of the
    # pooled rate's weight sits here (see :func:`_pooled_combined_rate`).
    "assembly-dense-37N-2021-resume": RadarCell(
        "37N/2021", "0.1-32.1", 160, 1997, 2862, 0, 87.5, 89.9, 46.4, 2.038e6, 1.985e6, 1.005e6, 1.279e6
    ),
}


class OrbitBand(NamedTuple):
    """Pooled both-orbit chunks in a 5-degree band of |latitude| — the radar-geography rows."""

    mid_abs_lat: float
    n: int
    optical_tok_px: float
    radar_tok_px: float
    us_per_valid_px: float  # busy GPU microseconds per valid pixel — the convention-free rate


#: Every 5-degree band holding both-orbit chunks, pooled across the cells (a band can mix
#: chunks from more than one cell, so these rows are their own population, not a re-cut of
#: the cell table). Regenerate with `scripts/radar_cost_basis.py` in yield-embeddings, which
#: derives each chunk's latitude from its tile row and drops duplicate chunk populations.
#:
#: Two gaps, visible in the `n` column: 50-55 is the thinnest row, and 35-50 is EMPTY — the
#: largest unmeasured span of populated land, and the measurement that would most narrow the
#: cost interval (`campaign-cost-model.md` §6c).
#:
#: DO NOT FIT A LATITUDE CURVE TO RADAR DEPTH. Its correlation with |latitude| is weak where
#: optical depth's is strong, and it is not monotonic: the minimum sits mid-range, between
#: deeper bands on both sides. The variation is REGIONAL, not latitudinal — deepest in the
#: Middle East, shallowest at the equator and in the Arctic — and the plausible mechanism is
#: the Sentinel-1 observation plan tasking Europe and the Middle East densely. The honest
#: claim is WEAK AND NON-MONOTONIC, never "zero". The defensible form is a constant with a
#: stated spread, enforced by
#: :func:`test_radar_depth_does_not_follow_latitude_and_optical_depth_does`.
BOTH_ORBIT_BANDS = [
    OrbitBand(2.5, 1064, 80.7, 71.9, 67.2),
    OrbitBand(7.5, 156, 80.6, 69.9, 72.0),
    OrbitBand(12.5, 294, 84.7, 76.3, 80.2),
    OrbitBand(17.5, 47, 84.7, 66.6, 74.4),
    OrbitBand(22.5, 89, 99.5, 66.1, 82.7),
    OrbitBand(27.5, 332, 91.3, 117.3, 110.1),
    OrbitBand(32.5, 243, 100.7, 151.8, 134.3),
    OrbitBand(52.5, 19, 100.1, 93.5, 69.0),
    OrbitBand(62.5, 201, 135.4, 71.1, 83.0),
    OrbitBand(67.5, 151, 152.2, 88.6, 101.0),
]


class OpticalBand(NamedTuple):
    """Optical depth in a 5-degree band of |latitude|, pixel-weighted over the telemetry corpus."""

    abs_lat_lo: float
    live_tiles: int  # live 2048-px tiles per year in the band — the land weight
    optical_tok_px: float


#: Every populated 5-degree band has data, and the rise is monotonic: optical depth DOES
#: follow latitude, which is exactly what radar depth does not do. Band tiles sum to the
#: same coverage snapshot `zone_density.ZONE_TILES` holds, so the land weighting is the
#: campaign's real geography — asserted in
#: :func:`test_the_measurement_tables_are_internally_consistent`.
OPTICAL_DEPTH_BY_BAND = [
    OpticalBand(0, 26_811, 74.0),
    OpticalBand(5, 27_940, 77.1),
    OpticalBand(10, 26_679, 80.3),
    OpticalBand(15, 29_785, 87.2),
    OpticalBand(20, 31_732, 89.2),
    OpticalBand(25, 31_429, 96.6),
    OpticalBand(30, 28_399, 95.7),
    OpticalBand(35, 23_189, 99.6),
    OpticalBand(40, 23_160, 103.2),
    OpticalBand(45, 22_833, 107.8),
    OpticalBand(50, 21_887, 120.5),
    OpticalBand(55, 17_653, 133.7),
    OpticalBand(60, 19_358, 142.0),
    OpticalBand(65, 17_168, 154.6),
    OpticalBand(70, 7_529, 164.5),
    OpticalBand(75, 4_032, 175.3),
    OpticalBand(80, 1_369, 175.5),
]


def _pooled_combined_rate(cells: Sequence[RadarCell]) -> float:
    """Total tokens over total seconds across the cells' both-orbit chunks.

    Each cell weighs in at `n_both x combined depth` — proportional to its tokens, valid
    pixels per chunk being near-equal across cells. The residual against a raw per-chunk
    pooling is the per-chunk valid-px variation this weighting drops, measured well inside
    the run-to-run noise floor (`campaign-cost-model.md` §6b).

    One cell dominating this pooling is a PROPERTY OF THE TABLE, not a fault of the
    weighting: a pooled rate can sit close to a single dominant cell's figure alone. That is
    why the cell table's unexplained-deficit note is load-bearing rather than a footnote —
    the deficit is not diluted here, it is inherited.
    """
    tokens = sum(c.n_both * (c.optical_tok_px + c.radar_tok_px) for c in cells)
    seconds = sum(c.n_both * (c.optical_tok_px + c.radar_tok_px) / c.combined_tok_s_both for c in cells)
    return tokens / seconds


def _pooled_radar_depth(cells: Sequence[RadarCell]) -> float:
    """CHUNK-weighted both-orbit radar depth over the cells — the sample's figure, not the campaign's.

    It weights each cell by how many chunks we happened to measure there, so it answers "how
    deep is radar in our sample" rather than "how deep is radar over campaign land". Optical
    depth is land-weighted (:func:`_land_weighted_optical`), and both halves of one depth
    must share one weighting, so :func:`_land_weighted_radar_depth` is the campaign figure.
    This one is retained as the contrast.
    """
    return sum(c.n_both * c.radar_tok_px for c in cells) / sum(c.n_both for c in cells)


def _radar_depth_by_band(bands: Sequence[OrbitBand], land: Sequence[OpticalBand]) -> list[tuple[float, float]]:
    """Radar depth for EVERY populated land band, interpolating the ones never measured.

    The measured bands do not cover all populated land. Rather than leave the campaign
    figure weighted only by where we happened to run, each missing band is filled — linearly
    between its measured neighbours, or held flat from the nearest measured band beyond the
    last one, where flat is the only defensible form.

    **Interpolating this is a decision, not an oversight**, and it is defensible only because
    the alternative is worse: omitting a band silently reweights the campaign onto the bands
    that remain. What it is NOT is a latitude model — see :data:`BOTH_ORBIT_BANDS`. Radar
    depth is not monotone in latitude, so interpolation here is filling a gap between two
    neighbours, never evaluating a trend, and it must not be extended into one.
    """
    measured = {b.mid_abs_lat: b.radar_tok_px for b in bands}
    known = sorted(measured)
    out: list[tuple[float, float]] = []
    for b in land:
        mid = b.abs_lat_lo + 2.5
        if mid in measured:
            out.append((mid, measured[mid]))
            continue
        below = [k for k in known if k < mid]
        above = [k for k in known if k > mid]
        if below and above:
            lo, hi = below[-1], above[0]
            span = (mid - lo) / (hi - lo)
            out.append((mid, measured[lo] + (measured[hi] - measured[lo]) * span))
        else:
            out.append((mid, measured[(below or above)[-1] if below else above[0]]))
    return out


def _land_weighted_radar_depth(bands: Sequence[OrbitBand], land: Sequence[OpticalBand]) -> float:
    """Live-tile-weighted both-orbit radar depth — the campaign figure.

    Weighted on the same live-tile geography as optical depth, so the two halves of
    :data:`CAMPAIGN_TOK_PER_PX` are on ONE weighting. The question is about campaign land,
    not about the sample: a chunk-weighted figure under-represents whatever latitudes the
    sample missed, and the unmeasured bands sit next to the deepest measured radar, so the
    difference between the two weightings is geographic, not statistical.

    The LEVEL is the weakest input in the basis and the widest driver of the interval — the
    measured bands span more than 2x, and :func:`test_cost_report` prints what that does to
    the line. Interpolation narrows no uncertainty; it only stops the weighting from being
    wrong.
    """
    per_band = dict(_radar_depth_by_band(bands, land))
    return sum(b.live_tiles * per_band[b.abs_lat_lo + 2.5] for b in land) / sum(b.live_tiles for b in land)


_CELLS = list(RADAR_TELEMETRY_CELLS.values())

#: The per-actor rate, in COMBINED (S2 + S1) tokens — DERIVED from the cell table rather
#: than typed, so the table is load-bearing. Combined tokens, not optical, because within
#: every measured cell the combined rate is near-flat across radar bases while the optical
#: rate swings (visible in the cell table's rate columns): combined tokens are a property of
#: the MACHINE and optical tokens are not.
COMBINED_TOK_PER_SEC = _pooled_combined_rate(_CELLS)

#: Radar depth of a both-orbit chunk, LAND-weighted over every populated band. A level with
#: a spread, deliberately NOT a curve — see the band table. Land-weighted rather than
#: chunk-weighted because both weightings are honest and only one matches the question —
#: see :func:`_land_weighted_radar_depth`.
RADAR_BOTH_ORBIT_TOK_PER_PX = _land_weighted_radar_depth(BOTH_ORBIT_BANDS, OPTICAL_DEPTH_BY_BAND)

#: The chunk-weighted (sample-weighted) figure, retained as the contrast.
RADAR_CHUNK_WEIGHTED_TOK_PER_PX = _pooled_radar_depth(_CELLS)
#: One-orbit chunks carry about half a both-orbit chunk's radar — a mean of per-cell
#: measurements, supported rather than assumed.
ONE_ORBIT_RADAR_FRACTION = fmean(c.radar_one_tok_px / c.radar_tok_px for c in _CELLS)

#: Radar composition of the campaign per pixel-year, area-weighted over the nine campaign
#: years, from the coverage census (`campaign-cost-model.md` §6); the Sentinel-1B failure
#: (2022-24) is what makes the single-orbit share this large.
DUAL_ORBIT_SHARE = 0.55
SINGLE_ORBIT_SHARE = 0.38
RADAR_FREE_SHARE = 0.068
#: A radar-free chunk still costs 8 tokens/px — a CODE fact, not a measurement: the smallest
#: bucket in DEFAULT_NUM_OBS_CHECKPOINTS is 8, and `resample_s1_bucket` hands the model an
#: all-zeros slice of that length. At 0.068 of pixel-years it is ~0.5 tok/px of the campaign
#: mean, immaterial either way; carried so the composition sums over ALL land.
RADAR_FREE_TOK_PER_PX = 8.0


def radar_tok_per_px(both_orbit_depth: float) -> float:
    """Composition-weighted campaign radar tokens/px, from a both-orbit radar depth."""
    return (
        DUAL_ORBIT_SHARE * both_orbit_depth
        + SINGLE_ORBIT_SHARE * ONE_ORBIT_RADAR_FRACTION * both_orbit_depth
        + RADAR_FREE_SHARE * RADAR_FREE_TOK_PER_PX
    )


def _land_weighted_optical(bands: Sequence[OpticalBand]) -> float:
    """Live-tile-weighted mean optical depth — the campaign's real geography as the weight."""
    return sum(b.live_tiles * b.optical_tok_px for b in bands) / sum(b.live_tiles for b in bands)


#: Land-weighted optical depth — the best-measured input in the section.
CAMPAIGN_OPTICAL_TOK_PER_PX = _land_weighted_optical(OPTICAL_DEPTH_BY_BAND)

#: The campaign's combined depth: land-weighted optical plus composition-weighted radar.
#: Typed rather than computed so the planning constant cannot wander when a table row is
#: edited — but a test asserts it stays within one token of the live-derived sum, so it
#: cannot drift from the tables either. NOT comparable to census-convention depths — see the
#: section header. Depth and rate come from the same measurement and move together: the
#: campaign can get cheaper per pixel and slower per token in one revision, so neither may
#: be updated without the other. Revision history: `campaign-cost-model.md` §6b-6c.
CAMPAIGN_TOK_PER_PX = 173.0

#: What the supply-denominated model actually consumes: pixels per second per actor. This
#: division is the ONLY place geography enters the rate, and both of its sides are combined
#: tokens on the chunk-array convention — the invariant this section exists to protect.
RATE_CAPACITY_PLANNING = COMBINED_TOK_PER_SEC / CAMPAIGN_TOK_PER_PX

#: The superseded optical basis, kept only so the report can print and assert what the unit
#: correction did NOT move. Two errors of similar size pointed opposite ways — the rate was
#: optical-only, and the censused depth undercounts the measured convention — so the px/s
#: the fleet is sized on barely moved, while fixing only the error that was NAMED overshoots
#: in the wrong direction. THE NEAR-CANCELLATION IS THE POINT. :func:`test_cost_report`
#: asserts both; the arithmetic is `campaign-cost-model.md` §6b's.
SUPERSEDED_OPTICAL_TOK_PER_SEC = 1_900_000.0
SUPERSEDED_TOK_PER_PX = 145.0

#: Iowa's observation count — every historical px/s figure comes from this ONE geography, a
#: single-orbit site at the cheap end of the radar distribution. A CENSUS-convention figure:
#: Iowa has never run under the radar telemetry, so it and CAMPAIGN_TOK_PER_PX are in
#: different conventions and no exact ratio between them is defensible — the guard test
#: asserts the inequality only. Re-running the Iowa ROI under the telemetry is the
#: measurement that would let the ratio be pinned.
IOWA_TOK_PER_PX = 136.0


def _landings(zones: list[int], look_ahead: int, workers: float) -> list[tuple[float, int]]:
    """``(hours, tiles)`` per mosaic, in the order they land.

    ``look_ahead`` ingests run at a time; each completion frees a slot for the next zone in
    the cluster's dealt order. This is the supply curve everything else reads.
    """
    remaining = list(zones)
    landings: list[tuple[float, int]] = []
    slots: list[float] = []
    for _ in range(min(look_ahead, len(remaining))):
        tiles = remaining.pop(0)
        done = ingest_hours(tiles, workers)
        slots.append(done)
        landings.append((done, tiles))
    slots.sort()
    while remaining:
        free = slots.pop(0)
        tiles = remaining.pop(0)
        done = free + ingest_hours(tiles, workers)
        landings.append((done, tiles))
        slots = sorted([*slots, done])
    landings.sort()
    return landings


def run(
    zones: list[int],
    *,
    n_actors: int,
    look_ahead: int,
    rate: float = RATE_CAPACITY_PLANNING,
    workers: int = SHIPPED_WORKERS,
    bank_work_hours: float = 0.0,
) -> dict:
    """Simulate one cluster-year; return duty, idle hours and the seam-gap profile.

    ``bank_work_hours`` is the start policy: hold the fleet until the queue holds that many
    hours of work FOR THIS FLEET. Zero reproduces the shipped rule, which boots on the first
    mosaic to land. Expressed in work-hours rather than mosaic count deliberately — see
    :func:`test_a_mosaic_count_is_the_wrong_buffer_unit`.
    """
    landings = _landings(zones, look_ahead, workers)
    px_per_h = n_actors * rate * 3600
    need_px = bank_work_hours * px_per_h

    banked = 0.0
    start = landings[-1][0]
    for done, tiles in landings:
        banked += tiles * PX_PER_TILE
        if banked >= need_px:
            start = done
            break

    t = start
    queue = sum(tiles * PX_PER_TILE for done, tiles in landings if done <= t)
    pending = [(d, n) for d, n in landings if d > t]
    total = sum(tiles * PX_PER_TILE for _, tiles in landings)
    done_px = 0.0
    idle = 0.0
    gaps: list[float] = []

    while done_px < total - 1:
        if queue > 0:
            nxt = pending[0][0] if pending else float("inf")
            step = min(queue / px_per_h, nxt - t) if pending else queue / px_per_h
            done_px += step * px_per_h
            queue -= step * px_per_h
            t += step
            while pending and pending[0][0] <= t + 1e-9:
                queue += pending.pop(0)[1] * PX_PER_TILE
        elif pending:
            gap = pending[0][0] - t
            idle += gap
            gaps.append(gap)
            t = pending[0][0]
            while pending and pending[0][0] <= t + 1e-9:
                queue += pending.pop(0)[1] * PX_PER_TILE
        else:
            break

    busy = total / px_per_h
    return {
        "start_h": start,
        "busy_h": busy,
        "idle_h": idle,
        "duty": busy / (busy + idle) if busy + idle else 0.0,
        "gaps": gaps,
        "worst_gap_h": max(gaps) if gaps else 0.0,
        "idle_gpu_hours": idle * n_actors,
    }


#: The shipped shape: 8 Ray clusters. `look_ahead` is `max_parallel_ingest / clusters`, so
#: the ingest cap chooses it — 40 cells gives 5 and the recommended 45 gives 6.
#:
#: A larger cap (and so a wider look-ahead) is deliberately not carried. The cost model found
#: a knee at ~45: past it the year barrier — years run serially, so a year cannot finish before
#: its slowest zone — makes additional cells worthless. That correction matters here because a
#: WIDER look-ahead is the thing that causes starvation in this model (it pulls small zones into
#: the opening window, so the fleet boots against a shallow queue). Narrowing the cap to its
#: knee therefore removes most of the problem this file exists to size.
CLUSTERS = 8
LOOK_AHEAD_AT_40 = 5
LOOK_AHEAD_AT_45 = 6
#: Actors per cluster, from `campaign-cost-model.md` §5 at the recommended 45 x 60w.
#:
#: The policy is to provision at ~85% of MATCHED, not at matched: a fleet that exactly consumes
#: what ingest produces has no absorber when supply dips, and every dip below it bills as idle.
#: Under-provisioning keeps a standing queue, which makes idle burn structurally zero rather
#: than merely small, at the cost of inference trailing ingest by ~18% of the run.
ACTORS_MATCHED = 268
ACTORS_PROVISIONED = 228
ACTORS_PER_CLUSTER = ACTORS_PROVISIONED


@pytest.fixture(scope="module")
def opener_cluster() -> list[int]:
    """The densest cluster's zones, in the order it will work them.

    Uses the real partitioner and the real densest-first sort, so this is what a campaign
    would actually do rather than a re-implementation.
    """
    return max(plan(CLUSTERS), key=lambda c: c.opener).tiles


@pytest.mark.parametrize("workers", WIDTHS.values(), ids=WIDTHS.keys())
def test_a_wider_ingest_window_makes_duty_cycle_worse(opener_cluster, workers: int) -> None:
    """The counter-intuitive result, and the reason this module exists.

    Raising the ingest cap is the cost model's highest-value setting — it buys wall clock and
    raises the fleet that can be kept busy. But `look_ahead` is derived from that cap, and a
    wider window pulls SMALLER zones into the opening set. One of them lands early, the
    shipped rule boots the fleet on the first mosaic to land, and the fleet then waits on the
    dense zones behind it.

    So the two levers interact: raising the cap without changing the start rule spends some
    of what it buys on idle GPUs.
    """
    narrow = run(opener_cluster, n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_40, workers=workers)
    wide = run(opener_cluster, n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_45, workers=workers)
    assert wide["duty"] < narrow["duty"], (
        f"expected the wider window to be worse; narrow={narrow['duty']:.0%} wide={wide['duty']:.0%}"
    )
    assert wide["idle_gpu_hours"] > 0, "the wide window should show real idle GPU-hours"


@pytest.mark.parametrize("workers", WIDTHS.values(), ids=WIDTHS.keys())
def test_banking_work_hours_removes_the_starvation_entirely(opener_cluster, workers: int) -> None:
    """A work-hours buffer closes the gap the wider window opens, at a bounded delay.

    This is the recommendation: hold the fleet until the queue can keep it busy for a few
    hours, rather than booting on the first mosaic. It costs a one-off delay per cluster-year
    and returns every idle hour after it.
    """
    unbuffered = run(opener_cluster, n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_45, workers=workers)
    buffered = run(
        opener_cluster, n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_45, workers=workers, bank_work_hours=6.0
    )
    assert unbuffered["idle_h"] > 0, "precondition: the unbuffered case must starve"
    assert buffered["idle_h"] == pytest.approx(0.0, abs=0.05), (
        f"a 6-work-hour bank should remove the starvation; got {buffered['idle_h']:.1f} h idle"
    )


def test_a_mosaic_count_is_the_wrong_buffer_unit(opener_cluster) -> None:
    """Counting mosaics is fooled by size; counting work is not.

    A cluster's zones span three orders of magnitude in tiles (a continental zone against a
    six-tile island), so "wait for K mosaics" can be satisfied by K tiny ones and leave the
    fleet just as hungry. The same K applied to a dense run over-waits. Work-hours is the
    unit that means the same thing in both cases — which is why the policy is expressed that
    way rather than as a count.
    """
    tiles = opener_cluster
    assert max(tiles) / max(1, min(tiles)) > 100, "precondition: the size spread is what breaks counting"

    landings = _landings(tiles, LOOK_AHEAD_AT_45, workers=SHIPPED_WORKERS)
    px_per_h = ACTORS_PER_CLUSTER * RATE_CAPACITY_PLANNING * 3600
    # Three mosaics, and how much fleet-time they actually represent, depends entirely on
    # WHICH three land first.
    first_three_work_h = sum(t * PX_PER_TILE for _, t in landings[:3]) / px_per_h
    biggest_three_work_h = sum(sorted((t for _, t in landings), reverse=True)[:3]) * PX_PER_TILE / px_per_h
    # The spread is configuration-dependent: a narrower look-ahead admits fewer zones to the
    # opening window, so the first three to land are more likely to be genuinely large ones
    # rather than small zones that overtook them. The count is still the wrong unit — 40% is
    # a large error in a safety buffer — and nothing here should imply that a count would be
    # tolerable at some other cap.
    assert biggest_three_work_h > 1.4 * first_three_work_h, (
        "three mosaics is not a fixed amount of work: "
        f"first three = {first_three_work_h:.1f} h, densest three = {biggest_three_work_h:.1f} h"
    )


def test_the_dense_end_is_what_keeps_the_fleet_fed(opener_cluster) -> None:
    """Frontloading dense zones is load-bearing, and this says by how much.

    Blocking on the densest zone was rejected as too slow, and the flow instead takes
    whichever mosaic lands first. That is only safe because the opening window is dense: the
    dense zones carry nearly all the work, so once one lands the queue is deep. Reverse the
    order — islands first — and the fleet boots almost immediately onto almost nothing.
    """
    dense_first = run(opener_cluster, n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_45)
    sparse_first = run(list(reversed(opener_cluster)), n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_45)
    assert sparse_first["idle_h"] > dense_first["idle_h"], (
        "sparse-first must be worse, or the ordering is not doing the job the design claims: "
        f"dense-first idle={dense_first['idle_h']:.1f} h, sparse-first idle={sparse_first['idle_h']:.1f} h"
    )


def test_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the full policy comparison. Run with `-s` to read it.

    Not an assertion — a diagnostic, so the numbers behind a plan decision can be regenerated
    rather than quoted from a document that may have drifted.
    """
    clusters = plan(CLUSTERS)
    cluster = max(clusters, key=lambda c: c.opener)
    with capsys.disabled():
        print(f"\n  densest cluster: {len(cluster.zones)} zones, {cluster.total:,} tiles, opener {cluster.opener:,}")
        for width_name, workers in WIDTHS.items():
            print(f"\n  ingest {width_name}: opener ingests in {ingest_hours(cluster.opener, workers):.1f} h")
            for la, cap in ((LOOK_AHEAD_AT_40, 40), (LOOK_AHEAD_AT_45, 45)):
                print(f"    ingest cap {cap} -> look_ahead {la}")
                for bank in (0.0, 3.0, 6.0, 12.0):
                    r = run(
                        cluster.tiles, n_actors=ACTORS_PER_CLUSTER, look_ahead=la, workers=workers, bank_work_hours=bank
                    )
                    tag = "boot on 1st mosaic" if bank == 0 else f"bank {bank:.0f} work-hours"
                    print(
                        f"      {tag:22s} duty={r['duty']:4.0%}  idle={r['idle_h']:5.1f}h  "
                        f"gaps={len(r['gaps']):2d}  worst={r['worst_gap_h']:4.1f}h  "
                        f"delay={r['start_h']:5.1f}h  idle GPU-h={r['idle_gpu_hours']:8,.0f}"
                    )


def wall_clock_optimum(
    zones: list[int],
    *,
    look_ahead: int,
    workers: int = SHIPPED_WORKERS,
    bank_work_hours: float = 6.0,
    rate: float = RATE_CAPACITY_PLANNING,
    ceiling: int = 800,
) -> tuple[int, float]:
    """``(actors, cluster-year hours)`` at the fleet size that finishes soonest.

    The right question, once a work-hours bank is in place. A bank scaled to the fleet
    automatically waits longer for a bigger fleet, so NOTHING starves however large it gets —
    which makes "the fleet supply can feed" undefined and the search for it degenerate. That
    is not a modelling artefact, it is the policy working: idle time moves from the middle of
    the run, where it is billed, to the front, where the fleet is not yet booted.

    So the trade changes shape. Without a bank, an oversized fleet burns GPU-hours. WITH one,
    it burns none — it simply stops buying wall clock, because the wait to fill the queue
    grows as fast as the fleet drains it. The optimum is where `start + busy` bottoms out.
    """
    best = (10, float("inf"))
    for n in range(10, ceiling + 1, 10):
        r = run(zones, n_actors=n, look_ahead=look_ahead, workers=workers, rate=rate, bank_work_hours=bank_work_hours)
        total = r["start_h"] + r["busy_h"] + r["idle_h"]
        if total < best[1]:
            best = (n, total)
    return best


@pytest.mark.parametrize("workers", WIDTHS.values(), ids=WIDTHS.keys())
def test_the_fleet_size_supply_can_feed_depends_on_the_ingest_width(opener_cluster, workers: int) -> None:
    """Beyond a point a bigger fleet stops buying wall clock, and the point depends on ingest.

    With a work-hours bank an oversized fleet wastes no GPU-hours — it waits longer to boot
    instead of idling mid-run. What it stops doing is helping: the wait grows as fast as the
    drain. So the fleet-sizing question becomes a wall-clock one, and its answer moves with
    the ingest duration basis and its width assumption. That makes the ingest basis a
    FLEET-SIZING matter, not just a schedule one.
    """
    actors, hours = wall_clock_optimum(opener_cluster, look_ahead=LOOK_AHEAD_AT_45, workers=workers)
    assert 10 <= actors <= 800
    # Doubling past the optimum must not pay for itself, or "optimum" means nothing.
    doubled = run(
        opener_cluster, n_actors=min(800, actors * 2), look_ahead=LOOK_AHEAD_AT_45, workers=workers, bank_work_hours=6.0
    )
    assert doubled["start_h"] + doubled["busy_h"] + doubled["idle_h"] >= hours - 0.5


# --- the buffer figure ---------------------------------------------------------------------

#: The recommended bank: the lowest value that clears the envelope scan below.
#:
#: Scanned over what the campaign plan actually leaves open — 8 clusters x {40, 45} cells x
#: {50, 60, 80} workers x {matched, 85%-provisioned} = 96 combinations:
#:
#:   1.00h  35 starving    1.50h  14    1.75h  3    2.00h  1    2.25h  0  <- threshold
#:
#: Set AT the threshold, per the same reasoning as before: everything above is safe too, but
#: boot delay keeps rising, and the per-zone ingest durations underneath carry +-35%, which no
#: fraction of an hour meaningfully covers either way.
#:
#: The threshold moves with the PLANNING RATE, which is why it is scanned rather than typed: a
#: slower per-actor rate boots the fleet against fewer banked pixels, since the bank is
#: denominated in work-hours. Re-scan whenever the rate or the depth changes.
#:
#: Under-provisioning already keeps a standing queue in steady state, so the bank protects
#: only the START of each year, when the queue is empty by construction. That happens nine
#: times, so it is worth having; it is not a major cost lever.
BANK_WORK_HOURS = 2.25

#: The fleet configurations to scan. There is one model (v2 Large was evaluated and rejected
#: — `v2_large_rollout_2026_07.md`), so the live uncertainty is whether the fleet runs at
#: matched or at the recommended 85% of matched.
FLEET_CONFIGS = {
    "provisioned 85%": ACTORS_PROVISIONED,
    "matched 100%": ACTORS_MATCHED,
}


@pytest.mark.parametrize("workers", WIDTHS.values(), ids=WIDTHS.keys())
@pytest.mark.parametrize("look_ahead", [LOOK_AHEAD_AT_40, LOOK_AHEAD_AT_45])
def test_one_bank_figure_serves_both_fleet_policies(opener_cluster, workers: int, look_ahead: int) -> None:
    """One bank figure covers matched and under-provisioned fleets alike.

    This is a property of expressing the buffer in WORK-HOURS rather than in mosaics or pixels.
    A smaller fleet takes proportionally longer to drain the same queue, so "two hours of work
    for this fleet" means the same protection at any size. A mosaic count or a pixel threshold
    would each need re-deriving whenever the fleet changed.

    The under-provisioned case should be strictly easier — it consumes more slowly — so this
    also pins the direction: if provisioning at 85% ever became the WORSE case, the model would
    be behaving in a way nobody predicted.
    """
    results = {
        name: run(
            opener_cluster,
            n_actors=actors,
            look_ahead=look_ahead,
            workers=workers,
            rate=RATE_CAPACITY_PLANNING,
            bank_work_hours=BANK_WORK_HOURS,
        )
        for name, actors in FLEET_CONFIGS.items()
    }
    for name, r in results.items():
        assert r["idle_h"] == pytest.approx(0.0, abs=0.05), f"{name} starves on the recommended bank"
    assert results["provisioned 85%"]["start_h"] <= results["matched 100%"]["start_h"] + 1e-9, (
        "the under-provisioned fleet must not need a LONGER wait than the matched one"
    )


@pytest.mark.parametrize("workers", WIDTHS.values(), ids=WIDTHS.keys())
def test_one_and_a_half_hours_is_not_enough(opener_cluster, workers: int) -> None:
    """Why the recommendation is not a rounder 1.5 hours.

    1.5 h is not merely a smaller number below a working one: it starves in 14 of the 96
    combinations scanned. Asserted over the whole envelope rather than on this one cluster,
    because which combinations fail depends on where each cluster's opener sits relative to the
    discrete mosaic landings, and singling out one case would make the test fragile.
    """
    assert run(
        opener_cluster,
        n_actors=ACTORS_PROVISIONED,
        look_ahead=LOOK_AHEAD_AT_45,
        workers=workers,
        rate=RATE_CAPACITY_PLANNING,
        bank_work_hours=BANK_WORK_HOURS,
    )["idle_h"] == pytest.approx(0.0, abs=0.05)
    assert any(
        run(
            cluster.tiles,
            n_actors=actors,
            look_ahead=look_ahead,
            rate=RATE_CAPACITY_PLANNING,
            workers=w,
            bank_work_hours=1.5,
        )["idle_h"]
        > 0.05
        for cluster in plan(CLUSTERS)
        for actors in FLEET_CONFIGS.values()
        for w in WIDTHS.values()
        for look_ahead in (LOOK_AHEAD_AT_40, LOOK_AHEAD_AT_45)
    ), "if 1.5 h were safe everywhere, the recommendation should be 1.5 h"


def test_the_campaign_costs_more_per_pixel_than_the_site_every_rate_came_from() -> None:
    """Iowa is cheaper per pixel than the campaign, so its px/s figures flatter us.

    Every throughput measurement we hold comes from one ROI, and that ROI is SINGLE-orbit —
    hundreds of ascending granules and zero descending. A pixel-denominated rate measured there
    therefore overstates campaign throughput. This is the defect that switching the cost model
    to tokens removes, and the guard is here so a future edit cannot quietly reintroduce a rate
    taken from one geography.

    **The DIRECTION is asserted and the RATIO is not**, which is what `IOWA_TOK_PER_PX` says:
    Iowa's figure is census-convention and the campaign's is chunk-array, so their quotient is
    not a quantity. Pinning a ratio here would test a coincidence of two conventions rather
    than a property of the model.
    """
    assert IOWA_TOK_PER_PX < CAMPAIGN_TOK_PER_PX
    # The rate this model plans at must be the CAMPAIGN one, not the reference site's.
    assert pytest.approx(COMBINED_TOK_PER_SEC / CAMPAIGN_TOK_PER_PX) == RATE_CAPACITY_PLANNING
    assert RATE_CAPACITY_PLANNING < COMBINED_TOK_PER_SEC / IOWA_TOK_PER_PX


def test_the_bank_holds_across_the_whole_envelope() -> None:
    """Every cluster, both fleet policies, three widths, both ingest caps — 96 combinations.

    The figure is sized to the worst combination rather than to the densest cluster, because a
    buffer validated only on the deepest queue is tuned to the easiest case.

    This test is what defends the constant. If a mask rebuild moves the tile distribution, or
    the campaign re-plans its cell count or fleet width, this fails rather than quietly
    eroding a margin nobody chose — which is why the scan is parameterised on the plan's open
    choices rather than on a single recommended configuration.
    """
    starving = []
    for cluster in plan(CLUSTERS):
        for name, actors in FLEET_CONFIGS.items():
            for workers in WIDTHS.values():
                for look_ahead in (LOOK_AHEAD_AT_40, LOOK_AHEAD_AT_45):
                    r = run(
                        cluster.tiles,
                        n_actors=actors,
                        look_ahead=look_ahead,
                        rate=RATE_CAPACITY_PLANNING,
                        workers=workers,
                        bank_work_hours=BANK_WORK_HOURS,
                    )
                    if r["idle_h"] > 0.05:
                        starving.append(
                            f"opener {cluster.opener:,} / {name} / {workers}w / "
                            f"look_ahead {look_ahead}: {r['idle_h']:.2f} h idle"
                        )
    assert not starving, f"{len(starving)} starving combination(s), e.g. {starving[:2]}"


# --- the cost line and its guards ----------------------------------------------------------
# The tests the measurement tables name: the radar-geography finding, the cross-table welds,
# and the printed cost line itself.

#: g6e.xlarge on-demand, us-west-2 — verified against the fleet's live instance type, not
#: read off a pricing page for an instance the fleet might not be running.
GPU_HOUR_USD = 1.861


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation, stdlib-only: ten table rows do not warrant a numpy import."""
    mx, my = fmean(xs), fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return cov / (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5


def test_radar_depth_does_not_follow_latitude_and_optical_depth_does() -> None:
    """Radar depth is a level with a spread; optical depth is a function of latitude.

    This is the guard the band table names: the campaign carries radar depth as a constant
    with a stated spread (66-152) because its variation is REGIONAL — the Sentinel-1
    observation plan — not latitudinal, while optical depth rises with latitude cleanly
    enough to band by it. A failure means a regenerated band table has changed that shape,
    and the constant-with-spread form needs re-deriving before any constant is retyped.

    The radar correlation is asserted WEAK, deliberately not near-zero: r moves from one
    regeneration of the band table to the next, and a test pinned at zero would fail on the
    next real measurement while the finding it protects — "not a usable function of
    latitude" — would still be true.
    """
    lats = [b.mid_abs_lat for b in BOTH_ORBIT_BANDS]
    radar = [b.radar_tok_px for b in BOTH_ORBIT_BANDS]
    optical = [b.optical_tok_px for b in BOTH_ORBIT_BANDS]
    radar_r = _pearson(lats, radar)
    optical_r = _pearson(lats, optical)

    assert optical_r > 0.85, (
        f"optical depth no longer follows latitude (r={optical_r:+.3f}); the banding premise is gone"
    )
    assert radar_r < 0.4, (
        f"radar depth reads as latitudinal (r={radar_r:+.3f}); the constant-with-spread form is suspect"
    )
    assert optical_r > 2 * radar_r, (
        f"the contrast has collapsed: optical r={optical_r:+.3f} against radar r={radar_r:+.3f}"
    )

    # THE assertion that kills a curve fit. Any monotonic function of latitude takes its
    # minimum at an end of the range; measured radar depth takes its minimum at 22.5 degrees,
    # between deeper bands on both sides. Correlation alone cannot pin this — a weak r still
    # admits a shallow monotonic fit.
    min_lat = lats[radar.index(min(radar))]
    assert min(lats) < min_lat < max(lats), (
        f"the radar-depth minimum ({min(radar):.1f} tok/px) sits at an END of the latitude range "
        f"({min_lat:.1f} of {min(lats):.1f}-{max(lats):.1f} degrees) — consistent with a monotonic "
        "trend, so the do-not-fit-a-curve claim needs re-deriving, not re-asserting"
    )
    assert min(radar) >= 66 and max(radar) <= 152, (
        f"the stated spread (66-152) no longer holds: measured {min(radar):.1f}-{max(radar):.1f}"
    )


def test_the_measurement_tables_are_internally_consistent() -> None:
    """The cross-table identities that would silently break if one table were edited alone.

    The planning constants above are DERIVED from tables so the tables are load-bearing — but
    the tables also describe each other, and nothing in Python makes two literals agree. Each
    assertion welds one pair:

    * The optical band tiles and ``zone_density.ZONE_TILES`` both sum to 360,953: the same
      coverage snapshot, so the land weighting under ``CAMPAIGN_OPTICAL_TOK_PER_PX`` is the
      campaign's real geography.
    * ``BOTH_ORBIT_BANDS`` and the cell table are two cuts of ONE population of 2,596
      both-orbit chunks. A mismatch means one was regenerated and the other was not.
    * The typed ``CAMPAIGN_TOK_PER_PX`` stays within one token of the live-derived
      optical-plus-radar sum — the guard that constant's own comment promises exists.
    * The orbit-composition shares cover all land. They are census-rounded constants
      (0.55 + 0.38 + 0.068 = 0.998), so the identity is asserted to half a percent rather
      than exactly.
    """
    band_tiles = sum(b.live_tiles for b in OPTICAL_DEPTH_BY_BAND)
    zone_tiles = sum(ZONE_TILES.values())
    assert band_tiles == zone_tiles == 360_953, (
        f"optical bands hold {band_tiles:,} tiles against the zone snapshot's {zone_tiles:,}: "
        "different coverage vintages (a joint move to a new total means both were regenerated "
        "together — update the pinned total)"
    )

    band_chunks = sum(b.n for b in BOTH_ORBIT_BANDS)
    cell_chunks = sum(c.n_both for c in RADAR_TELEMETRY_CELLS.values())
    assert band_chunks == cell_chunks == 2_596, (
        f"{band_chunks:,} banded both-orbit chunks against {cell_chunks:,} in the cell table: "
        "one cut of the population was regenerated without the other"
    )

    live_derived = CAMPAIGN_OPTICAL_TOK_PER_PX + radar_tok_per_px(RADAR_BOTH_ORBIT_TOK_PER_PX)
    assert abs(CAMPAIGN_TOK_PER_PX - live_derived) < 1.0, (
        f"the typed planning depth ({CAMPAIGN_TOK_PER_PX}) has drifted from the tables' "
        f"{live_derived:.2f} tok/px: a table row moved without the constant, or vice versa"
    )

    shares = DUAL_ORBIT_SHARE + SINGLE_ORBIT_SHARE + RADAR_FREE_SHARE
    assert shares == pytest.approx(1.0, abs=0.005), (
        f"the orbit composition covers {shares:.3f} of pixel-years, not all land"
    )


def test_cost_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the cost line, its interval, and what the unit correction did NOT move. `-s` to read.

    Same contract as :func:`test_report`: a diagnostic whose output is its product, so the
    numbers behind the campaign's dollar line can be regenerated rather than quoted. The
    sensitivity bounds are derived FROM the tables — band extremes for radar depth, cell
    extremes for the rate — so the report cannot drift from the data; the assertions sit at
    rel=0.02 because this is a planning line, not an identity.

    The interval's widest driver is radar DEPTH, not the rate — asserted below, so a
    regenerated table that flips them fails loudly rather than leaving a stale claim
    (`campaign-cost-model.md` §6c holds the measurements behind this).
    """
    live_tiles = sum(b.live_tiles for b in OPTICAL_DEPTH_BY_BAND)
    px_years = live_tiles * PX_PER_TILE * 9  # nine campaign years
    tokens = px_years * CAMPAIGN_TOK_PER_PX
    gpu_hours = tokens / COMBINED_TOK_PER_SEC / 3600
    cost = gpu_hours * GPU_HOUR_USD

    def dollars(tok_px: float, rate: float) -> float:
        return px_years * tok_px / rate / 3600 * GPU_HOUR_USD

    # Radar depth at the band extremes, at the central rate; the central depth for comparison.
    radar_lo = min(b.radar_tok_px for b in BOTH_ORBIT_BANDS)
    radar_hi = max(b.radar_tok_px for b in BOTH_ORBIT_BANDS)
    cost_radar_lo = dollars(CAMPAIGN_OPTICAL_TOK_PER_PX + radar_tok_per_px(radar_lo), COMBINED_TOK_PER_SEC)
    cost_radar_mid = dollars(
        CAMPAIGN_OPTICAL_TOK_PER_PX + radar_tok_per_px(RADAR_BOTH_ORBIT_TOK_PER_PX), COMBINED_TOK_PER_SEC
    )
    cost_radar_hi = dollars(CAMPAIGN_OPTICAL_TOK_PER_PX + radar_tok_per_px(radar_hi), COMBINED_TOK_PER_SEC)

    # The rate at the per-cell extremes, at the central depth.
    rate_slow = min(c.combined_tok_s_both for c in RADAR_TELEMETRY_CELLS.values())
    rate_fast = max(c.combined_tok_s_both for c in RADAR_TELEMETRY_CELLS.values())
    cost_slow = dollars(CAMPAIGN_TOK_PER_PX, rate_slow)
    cost_fast = dollars(CAMPAIGN_TOK_PER_PX, rate_fast)

    # What the unit correction did NOT move: the px/s the fleet is sized on.
    superseded_px_s = SUPERSEDED_OPTICAL_TOK_PER_SEC / SUPERSEDED_TOK_PER_PX
    move = RATE_CAPACITY_PLANNING / superseded_px_s - 1
    # Fixing only the NAMED error — the optical-token rate — while keeping the censused depth.
    named_only_px_s = COMBINED_TOK_PER_SEC / SUPERSEDED_TOK_PER_PX
    named_only_move = named_only_px_s / superseded_px_s - 1

    assert gpu_hours == pytest.approx(307_854, rel=0.02)
    assert cost == pytest.approx(573_000, rel=0.02)
    assert cost_radar_lo == pytest.approx(504_000, rel=0.02)
    assert cost_radar_mid == pytest.approx(573_000, rel=0.02)
    assert cost_radar_hi == pytest.approx(713_000, rel=0.02)
    assert cost_slow == pytest.approx(598_000, rel=0.02)
    assert cost_fast == pytest.approx(472_000, rel=0.02)
    # Depth spans more of the interval than the rate — the widest-driver claim above.
    assert cost_radar_hi - cost_radar_lo > cost_slow - cost_fast, (
        "the rate has re-taken the interval: the widest-driver claim needs re-deriving"
    )
    # The near-cancellation: a small net move, where the one-error fix was large AND pointed
    # the other way. (The -19% in the printed line is `campaign-cost-model.md` §6b's
    # arithmetic; it is not re-derivable from the current tables.)
    assert move == pytest.approx(-0.062, abs=0.005)
    assert named_only_move > 0.10 > abs(move), (
        "fixing the named error alone must overshoot the true correction and point the other way"
    )

    with capsys.disabled():
        print(f"\n  the campaign inference cost line — {live_tiles:,} live tiles x 9 years")
        print(
            f"    {CAMPAIGN_TOK_PER_PX:.0f} tok/px at {COMBINED_TOK_PER_SEC / 1e6:.3f}M combined tok/s/actor"
            f" -> {gpu_hours:,.0f} GPU-hours -> ${cost:,.0f}"
            f" at ${GPU_HOUR_USD}/h (g6e.xlarge on-demand us-west-2)"
        )
        print("\n  the interval, one driver at a time")
        print(
            f"    radar depth (band extremes, central rate):  {radar_lo:.1f} -> ${cost_radar_lo:,.0f}   "
            f"{RADAR_BOTH_ORBIT_TOK_PER_PX:.2f} land-weighted -> ${cost_radar_mid:,.0f}   "
            f"{radar_hi:.1f} -> ${cost_radar_hi:,.0f}"
        )
        # The two weightings side by side: same bands, different question, and the gap is the
        # geography of where radar is deepest against where the measured cells happen to be.
        print(
            f"      weighting: land {RADAR_BOTH_ORBIT_TOK_PER_PX:.2f} vs sample"
            f" {RADAR_CHUNK_WEIGHTED_TOK_PER_PX:.2f} tok/px"
            f" ({RADAR_BOTH_ORBIT_TOK_PER_PX / RADAR_CHUNK_WEIGHTED_TOK_PER_PX - 1:+.1%})"
        )
        print(
            f"    rate (cell extremes, central depth):        slowest {rate_slow / 1e6:.3f}M -> ${cost_slow:,.0f}   "
            f"fastest {rate_fast / 1e6:.3f}M -> ${cost_fast:,.0f}"
        )
        print("\n  what the 2026-08 unit correction did NOT move")
        print(
            f"    capacity-planning rate {RATE_CAPACITY_PLANNING:,.0f} px/s"
            f" against the superseded {superseded_px_s:,.0f} px/s: {move:+.1%}"
        )
        print(
            f"    correcting the named error alone: {named_only_px_s:,.0f} px/s ({named_only_move:+.1%})"
            " — the WRONG direction; the correction recorded that one-error fix as moving the line -19%"
        )
