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

#: S2 fleet width. 50 is the SHIPPED default; the campaign plan's RECOMMENDATION is now 60
#: (`campaign-cost-model.md` §4-5). The plan previously said explicitly not to raise it to 60,
#: on the argument that worker-hours are width-neutral so a narrower fleet frees the vCPU to
#: fit more concurrent cells. That argument lapsed when the cell count hit its own knee: past
#: ~45 cells the year barrier makes extra cells worthless, so the freed vCPU buys nothing and
#: width becomes the better place to spend it. Both are kept below as widths to check.
SHIPPED_WORKERS = 50

#: Widths to check — NOT an uncertainty band.
#:
#: This replaces a "the docs disagree by 2x" band that was my own confusion.
#: `campaign-cluster-sizing.md` says a cluster blocking on its opener waits "~10 h" while the
#: cell-hours basis implies ~20 h, and I carried both as unresolved. They are the SAME zone at
#: DIFFERENT widths: the width model gives 10.35 h at 120 workers and 19.69 h at 50. Nothing
#: was ever in disagreement — one document quotes a 120w measurement for a 50w campaign.
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


# --- inference rate ----------------------------------------------------------------------
#: The inference rate, DERIVED FROM TOKENS rather than quoted in pixels.
#:
#: The encoder consumes one sequence per pixel, so its cost scales with tokens, not pixels:
#: `tokens = pixels x (T_s2 + T_s1)`. Pixels-per-second is therefore a property of the machine
#: AND the geography it ran over, which is why the cost model rewrote its throughput basis
#: three times before switching units. This model needs px/s because its supply side is
#: denominated in tiles, so it converts once, here, and the conversion is the only place
#: geography enters the rate.
#:
#: 1.9M tok/sec is the reference per-worker rate; 145 tok/px is the campaign's land-weighted
#: observation count after optical-only cells. Both from `campaign-cost-model.md` §6.
TOK_PER_SEC = 1_900_000.0
CAMPAIGN_TOK_PER_PX = 145.0
RATE_CAPACITY_PLANNING = TOK_PER_SEC / CAMPAIGN_TOK_PER_PX  # ~13,103 px/s
#: Iowa's observation count — every historical px/s figure comes from this ONE geography, and
#: it is a single-orbit site at the cheap end of the radar distribution. Carried so a test can
#: assert the campaign is more expensive per pixel than the site all the rates came from.
IOWA_TOK_PER_PX = 136.0
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
#: the ingest cap chooses it — 40 cells gives 5 and the recommended 45 gives 6.
#:
#: 71 cells (look_ahead 9) is SUPERSEDED and deliberately not carried. The cost model found a
#: knee at ~45: past it the year barrier — years run serially, so a year cannot finish before
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
    # A 1.4x spread, not the 3x this asserted at the superseded 71-cell look-ahead. The gap
    # NARROWED for a reason worth keeping: a narrower look-ahead admits fewer zones to the
    # opening window, so the first three to land are more likely to be genuinely large ones
    # rather than small zones that overtook them. The count is still the wrong unit — 40% is a
    # large error in a safety buffer — but the configuration change made it less wrong, and
    # nothing here should imply that a count would be tolerable at some other cap.
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
            for la, cap in ((LOOK_AHEAD_AT_40, 40), (LOOK_AHEAD_AT_45, 71)):
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
    actors, hours = wall_clock_optimum(opener_cluster, look_ahead=LOOK_AHEAD_AT_45, workers=workers)
    assert 10 <= actors <= 800
    # Doubling past the optimum must not pay for itself, or "optimum" means nothing.
    doubled = run(
        opener_cluster, n_actors=min(800, actors * 2), look_ahead=LOOK_AHEAD_AT_45, workers=workers, bank_work_hours=6.0
    )
    assert doubled["start_h"] + doubled["busy_h"] + doubled["idle_h"] >= hours - 0.5


# --- the buffer figure ---------------------------------------------------------------------

#: The recommended bank: 2.0 work-hours, the lowest value that clears the envelope.
#:
#: THREE earlier figures are superseded, and they are named because the sequence is the useful
#: part — each was wrong for a different reason:
#:
#:   4.25  ingest scaled purely with AREA, needing an invented 0.75 h floor for the island tail
#:   3.3   fixed/variable split taken off the S1 window-overlap table, a different workstream
#:   3.25  correct ingest model, but scanned at 71 ingest cells and a 50-worker fleet
#:
#: The last of those was not a modelling error; the CONFIGURATION changed under it. The cost
#: model found a knee at ~45 cells and moved the recommended width to 60, and both changes push
#: the same way: a narrower look-ahead pulls fewer small zones into the opening window, and a
#: wider fleet fills it faster. Since a shallow opening queue is the entire mechanism this file
#: models, the campaign's own re-planning removed most of the problem.
#:
#: Scanned over what the campaign plan actually leaves open — 8 clusters x {40, 45} cells x
#: {50, 60, 80} workers x {matched, 85%-provisioned} = 96 combinations:
#:
#:   1.00h  35 starving    1.50h  14    1.75h  3    2.00h  0  <- threshold
#:
#: Set AT the threshold, per the same reasoning as before: everything above is safe too, but
#: boot delay keeps rising, and the per-zone ingest durations underneath carry +-35%, which no
#: fraction of an hour meaningfully covers either way.
#:
#: **What it is worth has fallen by an order of magnitude, and that is the headline.** At 71
#: cells on a matched 50w fleet the bank saved ~$202,000 of idle GPU time. At the recommended
#: 45 x 60w with the fleet provisioned at 85% of matched it saves about **$15,000** — because
#: under-provisioning already keeps a standing queue in steady state. The bank now protects only
#: the START of each year, when the queue is empty by construction. That happens nine times, so
#: it is still worth having; it is no longer a major cost lever.
BANK_WORK_HOURS = 2.0

#: The fleet configurations to scan. The v2-versus-v1.1 axis this file used to carry is GONE:
#: v2 Large was evaluated and rejected, so there is one model. What replaces it as the live
#: uncertainty is whether the fleet runs at matched or at the recommended 85% of matched.
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
    hundreds of ascending granules and zero descending. At 136 tok/px against the campaign's
    land-weighted 145, a pixel-denominated rate measured there overstates campaign throughput by
    the ratio. This is the defect that switching the cost model to tokens removes, and the guard
    is here so a future edit cannot quietly reintroduce a rate taken from one geography.
    """
    assert IOWA_TOK_PER_PX < CAMPAIGN_TOK_PER_PX
    overstatement = CAMPAIGN_TOK_PER_PX / IOWA_TOK_PER_PX
    assert overstatement == pytest.approx(1.066, abs=0.01)
    # And the rate this model plans at must be the CAMPAIGN one, not the reference site's.
    assert pytest.approx(TOK_PER_SEC / CAMPAIGN_TOK_PER_PX) == RATE_CAPACITY_PLANNING
    assert RATE_CAPACITY_PLANNING < TOK_PER_SEC / IOWA_TOK_PER_PX


def test_the_bank_holds_across_the_whole_envelope() -> None:
    """Every cluster, both fleet policies, three widths, both ingest caps — 96 combinations.

    The figure is sized to the worst combination rather than to the densest cluster, because a
    buffer validated only on the deepest queue is tuned to the easiest case.

    This test is what defends the constant. If a mask rebuild moves the tile distribution, or
    the campaign re-plans its cell count or fleet width again, this fails rather than quietly
    eroding a margin nobody chose — which is exactly what happened between the 3.25 figure and
    this one, and is why the scan is parameterised on the plan's open choices rather than on a
    single recommended configuration.
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
