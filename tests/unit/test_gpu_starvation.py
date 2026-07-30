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
"""

from __future__ import annotations

# The report at the bottom prints by design: it is a diagnostic whose output IS its product,
# run with `-s` to regenerate the numbers behind a planning decision rather than quoting a
# document that may have drifted. Same contract as the repo's other operator-facing tools.
# ruff: noqa: T201
import pytest

from tests.unit.zone_density import plan

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
#   6-tile island costs about an hour, not minutes. An earlier version of this model scaled
#   purely with area and needed an invented floor.
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
DATES_PER_ZONE_YEAR = 365  # not the ~250 once assumed: a full-height zone sees imagery daily

#: S2 fleet width. 50 is the SHIPPED default and the campaign plan says explicitly not to raise
#: it to 60: worker-hours are width-neutral, so the narrower fleet costs the same and frees the
#: vCPU that lets 71 concurrent cells fit a 25,000 quota instead of 67.
SHIPPED_WORKERS = 50

#: Widths to check — NOT an uncertainty band.
#:
#: This replaces a "the docs disagree by 2x" band that was my own confusion.
#: `campaign-cluster-sizing.md` says a cluster blocking on its opener waits "~10 h" while the
#: cell-hours basis implies ~20 h, and I carried both as unresolved. They are the SAME zone at
#: DIFFERENT widths: the width model gives 10.35 h at 120 workers and 19.69 h at 50. Nothing
#: was ever in disagreement — one document quotes a 120w measurement for a 50w campaign.
WIDTHS = {"shipped 50w": 50, "as-measured 120w": 120}


def _width_scale(workers: int) -> float:
    """Per-date time at ``workers`` relative to the density table's own width."""
    return (36.3 + 7896 / workers) / (36.3 + 7896 / DENSITY_FIT_WORKERS)


def seconds_per_date(tiles: int, workers: int = SHIPPED_WORKERS) -> float:
    chunks = tiles / TILES_PER_CHUNK
    return (DENSITY_FIXED_S + DENSITY_PER_CHUNK_S * chunks) * _width_scale(workers)


def ingest_hours(tiles: int, workers: int = SHIPPED_WORKERS) -> float:
    """Hours to ingest one zone-year. The fixed term supplies the floor; none is invented."""
    return seconds_per_date(tiles, workers) * DATES_PER_ZONE_YEAR / 3600


# --- inference rate ----------------------------------------------------------------------
#: Per-worker px/s from `inference_gpu_saturation_profile_2026_07.md`. The fleet-overall
#: figure is the one that document labels "the capacity-planning number" — it already absorbs
#: cold starts, the density mix and the ramp, which is what a campaign average needs. The
#: per-chunk-class rates are density-specific (21-24K is MID-density; dense is 10-18K), so a
#: dense-weighted campaign must not be planned at the mid rate.
RATE_CAPACITY_PLANNING = 14_000.0
RATE_MID_WHILE_PROCESSING = 22_500.0


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
#: the ingest cap chooses it — 40 cells gives 5, and the recommended 71 gives 9.
CLUSTERS = 8
LOOK_AHEAD_AT_40 = 5
LOOK_AHEAD_AT_71 = 9
#: Matched fleet per cluster from `campaign-cost-model.md`: 1,678 GPUs over 8 clusters.
ACTORS_PER_CLUSTER = 210


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
    wide = run(opener_cluster, n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_71, workers=workers)
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
    unbuffered = run(opener_cluster, n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_71, workers=workers)
    buffered = run(
        opener_cluster, n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_71, workers=workers, bank_work_hours=6.0
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

    landings = _landings(tiles, LOOK_AHEAD_AT_71, workers=SHIPPED_WORKERS)
    px_per_h = ACTORS_PER_CLUSTER * RATE_CAPACITY_PLANNING * 3600
    # Three mosaics, and how much fleet-time they actually represent, depends entirely on
    # WHICH three land first.
    first_three_work_h = sum(t * PX_PER_TILE for _, t in landings[:3]) / px_per_h
    biggest_three_work_h = sum(sorted((t for _, t in landings), reverse=True)[:3]) * PX_PER_TILE / px_per_h
    assert biggest_three_work_h > 3 * first_three_work_h, (
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
    dense_first = run(opener_cluster, n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_71)
    sparse_first = run(list(reversed(opener_cluster)), n_actors=ACTORS_PER_CLUSTER, look_ahead=LOOK_AHEAD_AT_71)
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
            for la, cap in ((LOOK_AHEAD_AT_40, 40), (LOOK_AHEAD_AT_71, 71)):
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
    the unresolved 2x disagreement in the ingest duration basis. That makes resolving that
    disagreement a FLEET-SIZING matter, not just a schedule one.
    """
    actors, hours = wall_clock_optimum(opener_cluster, look_ahead=LOOK_AHEAD_AT_71, workers=workers)
    assert 10 <= actors <= 800
    # Doubling past the optimum must not pay for itself, or "optimum" means nothing.
    doubled = run(
        opener_cluster, n_actors=min(800, actors * 2), look_ahead=LOOK_AHEAD_AT_71, workers=workers, bank_work_hours=6.0
    )
    assert doubled["start_h"] + doubled["busy_h"] + doubled["idle_h"] >= hours - 0.5


# --- the buffer figure, and why one number covers both models ----------------------------

#: v2 Large's rate relative to v1.1, from the branch's own per-model estimator in
#: `inference/actors.py` (22,000 against 16,000 px/s). Strategy-only and its calibration is
#: undocumented, so the RATIO is what is defensible, not the absolutes.
V2_SPEEDUP = 1.375
#: Matched fleets per cluster from `campaign-cost-model.md` §5, at the **14K capacity-planning
#: basis** and 71 ingest cells: 2,518 GPUs on v1.1 and 1,831 on v2, over 8 clusters. The fleet
#: shrinks because a faster model needs less of it to keep pace with the same ingest.
#:
#: These were 210 and 153 until the cost model's throughput basis was corrected. That pairing was
#: WRONG in a specific way worth naming, because it is the error the cost model now warns about:
#: 210 is the fleet matched to a 21K px/s rate, while `RATE_CAPACITY_PLANNING` below is 14K. A
#: fleet sized on one rate and run at another is not matched to anything, and it understated the
#: fleet by a third — which made starvation LOOK less likely than it is. Keep these two constants
#: consistent: `n_actors x rate` must equal the per-cluster supply the scenario claims.
ACTORS_V11 = 315
ACTORS_V2 = 229

#: The recommended bank: 3.25 work-hours, the lowest value that clears the envelope.
#:
#: Sized against the ingest model as sourced from the CANONICAL record. Two earlier figures are
#: superseded and named so the arithmetic can be followed: 4.25 came from a purely
#: area-proportional model with an invented 0.75 h floor; 3.3 came from taking the fixed/variable
#: split off the three-ROI overlap table, which is a different workstream. This uses the
#: five-region k=1 density fit and the width model, which agree with each other to 5%.
#:
#: Scanned across the envelope (8 clusters x 3 fleet configurations x 2 widths x 2 caps = 96):
#:
#:   2.50h  3 starving    3.00h  1 starving    3.25h  0  <- threshold
#:
#: Set AT the threshold. Everything above it is safe too, but the boot delay keeps rising, so
#: higher is not free — and the per-zone durations underneath carry +-35%, which no fraction of
#: an hour meaningfully covers either way. The envelope test is what defends this number: a mask
#: rebuild that moves the tile distribution fails it rather than eroding a margin nobody chose.
BANK_WORK_HOURS = 3.25


@pytest.mark.parametrize("workers", WIDTHS.values(), ids=WIDTHS.keys())
@pytest.mark.parametrize("look_ahead", [LOOK_AHEAD_AT_40, LOOK_AHEAD_AT_71])
def test_one_bank_figure_serves_both_models_when_the_fleet_is_matched(
    opener_cluster, workers: int, look_ahead: int
) -> None:
    """The work-hours bank is MODEL-INVARIANT, which is why one figure is enough.

    A matched fleet shrinks in inverse proportion to the model's rate — 210 actors at 14.0K
    and 153 at 19.25K consume 2.940 and 2.945 Mpx/s, within 0.2% — so the same number of
    work-hours means the same number of pixels. That is a property of expressing the buffer in
    work-hours: a pixel threshold or a mosaic count would each need re-deriving per model.

    So the model choice, which is still open and worth $91,000-$128,000, does not also require
    choosing a buffer.
    """
    v11 = run(
        opener_cluster,
        n_actors=ACTORS_V11,
        look_ahead=look_ahead,
        workers=workers,
        rate=RATE_CAPACITY_PLANNING,
        bank_work_hours=BANK_WORK_HOURS,
    )
    v2 = run(
        opener_cluster,
        n_actors=ACTORS_V2,
        look_ahead=look_ahead,
        workers=workers,
        rate=RATE_CAPACITY_PLANNING * V2_SPEEDUP,
        bank_work_hours=BANK_WORK_HOURS,
    )
    # The claim is that ONE figure keeps both safe — not that the boot times coincide. They are
    # close but not equal: the threshold lands on a discrete mosaic, so a 0.18% difference in
    # consumption can tip which one satisfies it.
    assert v11["idle_h"] == pytest.approx(0.0, abs=0.05), "v1.1 starves on the recommended bank"
    assert v2["idle_h"] == pytest.approx(0.0, abs=0.05), "v2 starves on the recommended bank"
    assert v11["start_h"] == pytest.approx(v2["start_h"], rel=0.10)


@pytest.mark.parametrize("workers", WIDTHS.values(), ids=WIDTHS.keys())
def test_three_hours_is_not_enough_if_v2_runs_on_a_v11_sized_fleet(opener_cluster, workers: int) -> None:
    """Why the recommendation is not three hours.

    Switching models is one configuration change and resizing the fleet is another, so the
    realistic mistake is running v2 on a fleet still sized for v1.1 — which consumes 1.375x
    faster and drains a three-hour bank before supply catches up. Four covers it, and costs
    nothing in the matched case.
    """
    kw = dict(
        n_actors=ACTORS_V11, look_ahead=LOOK_AHEAD_AT_71, workers=workers, rate=RATE_CAPACITY_PLANNING * V2_SPEEDUP
    )
    assert run(opener_cluster, bank_work_hours=BANK_WORK_HOURS, **kw)["idle_h"] == pytest.approx(0.0, abs=0.05), (
        "the recommended bank must hold even on an unrescaled fleet"
    )
    # Three hours is not merely a rounder number below a working one: it starves somewhere in
    # the envelope. Asserted over the whole set rather than on this cluster, because on the
    # canonical ingest model the failures are not concentrated in the unrescaled-v2 case an
    # earlier version of this test singled out.
    fleets = (
        (ACTORS_V11, RATE_CAPACITY_PLANNING),
        (ACTORS_V2, RATE_CAPACITY_PLANNING * V2_SPEEDUP),
        (ACTORS_V11, RATE_CAPACITY_PLANNING * V2_SPEEDUP),
    )
    assert any(
        run(
            cluster.tiles,
            n_actors=actors,
            look_ahead=look_ahead,
            rate=rate,
            workers=w,
            bank_work_hours=3.0,
        )["idle_h"]
        > 0.05
        for cluster in plan(CLUSTERS)
        for actors, rate in fleets
        for w in WIDTHS.values()
        for look_ahead in (LOOK_AHEAD_AT_40, LOOK_AHEAD_AT_71)
    ), "if three hours were safe everywhere, the recommendation should be three"


def test_the_bank_holds_across_the_whole_envelope() -> None:
    """Every cluster, both models, both fleet widths, both ingest caps — 96 combinations.

    The figure is sized to the worst combination rather than to the densest cluster, because a
    buffer validated only on the deepest queue is tuned to the easiest case. That is what ruled
    out four hours: one cluster of the eight — 15 zones opening on 8,731 tiles — still left
    ~12 minutes of idle there, 2 starving combinations of 96.

    It did NOT justify five: the delay is a step function, so the whole 4.25-4.9 band clears
    the envelope at the same cost 4.0 pays, and five buys 2.2 h of extra delay per cluster-year
    for nothing. The constant sits at 4.25, the threshold itself, and this test is what
    defends it — if a mask rebuild moves the tile distribution, this fails rather than quietly
    eroding a margin nobody chose.
    """
    fleets = (
        (ACTORS_V11, RATE_CAPACITY_PLANNING),
        (ACTORS_V2, RATE_CAPACITY_PLANNING * V2_SPEEDUP),
        # v2 on a fleet still sized for v1.1 — one config change without the other.
        (ACTORS_V11, RATE_CAPACITY_PLANNING * V2_SPEEDUP),
    )
    starving = []
    for cluster in plan(CLUSTERS):
        for actors, rate in fleets:
            for workers in WIDTHS.values():
                for look_ahead in (LOOK_AHEAD_AT_40, LOOK_AHEAD_AT_71):
                    r = run(
                        cluster.tiles,
                        n_actors=actors,
                        look_ahead=look_ahead,
                        rate=rate,
                        workers=workers,
                        bank_work_hours=BANK_WORK_HOURS,
                    )
                    if r["idle_h"] > 0.05:
                        starving.append(
                            f"opener {cluster.opener:,} / {actors} actors / {rate:.0f} px/s / "
                            f"{workers}w / look_ahead {look_ahead}: {r['idle_h']:.2f} h idle"
                        )
    assert not starving, f"{len(starving)} starving combination(s), e.g. {starving[:2]}"
