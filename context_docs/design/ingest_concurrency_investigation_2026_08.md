# Ingest throughput at fleet scale — what we know, and what we still need

**Status: resolved as posed.** The headline finding — a 1.8–2.1× slowdown against the July
record — **was an artefact of comparing summer dates against a January baseline, and is
withdrawn.** Matched on zone, width AND dates the gap is 1.12–1.17×. A paired A/B then measured
the contention term at **zero** (the quiet arm came out 3.3% dearer per window, not cheaper), so
none of that residual is load at the width tested.

**What is left.** Contention at 45–61 concurrent cells is unmeasured — the A/B ran at about a
sixth of campaign width — but the ladder that would measure it is on hold by decision
(2026-08-05) and blocked on the quota raise regardless. The live item is the campaign's duration
basis, which needs re-fitting on a **seasonally weighted** year rather than a January rate. That
is the one correction here which genuinely raises the ingest line.

Companion to `ingest_optimization_campaign_2026_07.md`, which remains the authoritative record
for everything measured before 2026-08.

> **Condensed 2026-08-17**, from 431 lines. The stage plan, the per-arm run narrative, the
> decision not to run a third arm, and this document's own list of self-corrections were the
> working of a question now answered; they are in git history. What is kept is what would
> otherwise be re-derived: the causes ruled out, the effects established, and the two
> methodological traps that produced the withdrawn claim.

## What is ruled out

Six candidate explanations, each with the evidence that closed it. **None of these should be
re-investigated without new evidence** — that is the point of recording them.

| Candidate | Why it is out |
|---|---|
| **Width stops working at scale** | 6× workers buys 3.7–4.9×, against an Amdahl bound of 4.4–4.6× from the serial fraction at 10w. Width works about as well as arithmetic allows. The 2.7–4.0× that suggested otherwise was a **zone-mix artefact** — it compared different zones, and the two 10-worker zones were the two cheapest-per-chunk in the wave. |
| **The orchestrator** | Direct refutation: over the window where per-date cost rose 32%, orchestrator CPU fell 13%→5.6%, requests 12.4→4.1/s, latency 146→39 ms, zero dropped events. Corroborated by the server gaining 8× capacity between waves while the *later* wave was slower. |
| **The Dask schedulers** | CPU 10–40%, event-loop lag 0.0–0.03 s, graphs oversubscribing their fleets rather than starving them. Well under the ~250-worker threshold in `docs/dask-scheduler-plan.md`. |
| **Capacity, quota, launch rate** | 20 live fleets, 519 workers, 2,236 vCPU of 10,000, and `no-worker=0` on every fleet — no task anywhere waiting for a worker, at 22% of quota. |
| **Commit serialisation** | Commits are ~1 s; the per-date cadence leaves only 5–8 s unaccounted after build + gate + write + stall. |
| **Store growth under accumulating dates** | Per-date cost rose 36% through 53N's run, but **write cost per window is flat at 16.4–18.3 s** across all four quartiles and build per window flat at ~1.2 s. Cost rose because windows per date rose 45% (7.3 → 10.6). Manifest sharding is doing its job. |

> **A trap worth keeping.** ECS `describe-clusters` reported **763 pending against 67 running**
> during this investigation, which reads as catastrophic launch starvation. It is an artefact:
> cluster statistics are eventually-consistent and the account carried **204 registered ephemeral
> worker families** from a day of fleet churn. 760 genuinely pending tasks cannot coexist with
> seven fleets sitting at 56–60 workers and committing dates. **Use the schedulers' own
> `workers=` and `no-worker=` fields; they are ground truth for width.**

## What is established

- **A per-date serial floor of 19–24% at 60 workers** (7–8% at 10w). Bounds the width benefit, and
  is worth more than per-zone width tuning: the unhidden preparation stall alone is 20–45 s median
  on writes of 160–300 s that ought to hide it.
- **Adaptive churn cost ~10–12% of effective width.** `adapt(minimum=1)` retired workers in every
  inter-date gap and relaunched them cold; one 60-slot fleet registered 1,250 distinct workers in
  5 h. **Fixed** — `min_workers` now follows each leg's derived width.
- **Per-date cost grows within a run, for a benign reason.** Later dates image more of a zone's
  land: northern-hemisphere January is snow- and cloud-limited, by May footprints are wider.
  **Consequence: every velocity figure measured early in a run UNDERSTATES the full-year cost.**
- **Width is nearly cost-neutral.** vCPU-seconds per date at 60w vs 10w averages ~1.1× over six
  same-zone pairs. Sixty workers costs about what ten does per date and finishes 4–5× sooner, so
  re-scaling `max_workers` is not where money is saved.

## The withdrawn claim, and the mechanism behind it

**The claim:** every zone runs 1.8–2.1× slower than the July record at the same zone and width.
**It compared summer dates against a January baseline.**

| | zone | width | dates | windows/date | s/date |
|---|---|---|---|---:|---:|
| July record §3.10/§3.16 | 35N | 60w | **January 2024** | — | **167.9 / 175.6** |
| the withdrawn figure | 35N | 60w | **May–Sep 2021** (n=128) | 18.0 | **330.7** |
| matched re-measurement | 35N | 60w | **January 2024** (n=27) | 15.0 | **196.3** |

Matched on all three conditions the gap is **1.17×** — and the matched arm carried 17 concurrent
fleets while the July figure did not, so 1.17× is an upper bound on contention *plus* drift, not
a floor.

The mechanism is the within-run effect above, applied across seasons: 18.0 windows/date against
15.0 is 1.20× more work, and summer windows are individually dearer (write per window 16.7 s
against 11.0 s). Together those carry the 1.68× ratio with no appeal to a regression.

**The July record's reading instructions state the condition that was violated** — *"unless a
figure says otherwise, every timing is zone 35N, January 2024."* The claim honoured zone and
width and silently violated the third, which is the one that moves cost most.

## What this changes: the duration basis, not the code

**The code has not regressed**, and the six ruled-out causes stay ruled out. But the July fit
(5.95 h/zone-year at 60w) is built from January-conditions measurements, and a zone-year is not
twelve Januaries.

35N at 60 workers, six of twelve months measured:

| month | windows/date | s/date | write/window |
|---|---:|---:|---:|
| Jan (2024) | 15.0 | **196.3** | 11.04 |
| Feb (2024) | 16.0 | **218.4** | 10.21 |
| Mar–Apr | — | ~250–290 *(interpolated)* | — |
| May (2021) | 18.0 | **330.5** | — |
| Jun (2021) | 18.0 | **381.6** | 18.71 |
| Jul (2021) | 18.0 | **317.6** | 16.11 |
| Aug (2021) | 18.0 | **309.6** | 14.15 |
| Sep (2021) | 18.0 | **320.6** | — |
| Oct–Dec | — | ~210–290 **(extrapolated, no data)** | — |

Two things a single multiplier hides. **Windows per date saturates at 18.0 by May and stays
there** — it plateaus once the zone is fully imaged rather than following a smooth sinusoid. And
the summer premium is mostly *not* window count: 1.20× on count against 1.68× on cost, the rest
being that a window with less cloud and snow carries more valid pixels.

Averaging gives **~280 s/date against the fit's 167.9, i.e. ~1.67×**. A quarter of that year is
extrapolated, so **treat the band and not the point.** The fix is a seasonal weighting of the
basis, not a hunt for a performance defect. Pin it with per-date covered chunks or one completed
full-year 60-worker cell; the seven complete 2021 zone-years cannot serve, because they ran at
10 workers.

## The one concurrency signal left, and why it is weak

Two observations point in opposite directions: across waves, more cells looked slower; within a
run, fewer cells looked slower. The within-run one is largely the seasonal effect confounded with
elapsed time — the run progressed as the wave emptied — so **it is not a concurrency signal at
all.** That leaves the across-wave observation, and it is weak: three of its six zones showed no
degradation, and its two waves differ in server size, time of day and date range as well as cell
count.

Four candidates remain and the data cannot separate them: source-read contention above ~20 cells
(July measured only to 20 and left large-count elasticity explicitly unmeasured); time-of-day load
on the public archive; the 2021 catalogue versus the 2024 dates every July figure used; and
post-July configuration drift — 35N now writes 18 windows/date against 13 in July §3.15, which by
the windows-per-date arithmetic alone accounts for ~1.4× of the gap.

## Two methodological rules this investigation earned

**Match on every condition that moves the number, and list them before comparing.** Three claims
here were withdrawn for the same reason: a real measurement compared against another real
measurement whose conditions differed in a dimension nobody had enumerated. Zone and width were
checked; season was not.

**Verify achieved width from the scheduler, not from the parameters requested.** The quiet arm of
the A/B ran on a different account — the loaded account could not host a quiet arm — so account,
VPC, ECS cluster, Prefect server and S3 bucket were all unmatched, and only a *small* difference
would have been interpretable. What made it interpretable at all was reading achieved width off
the scheduler health lines: median 57, max 60 workers with `no-worker=0` on all 48 samples, against
28 loaded-account fleets of which 14 sat at max 60, `no-worker=0` on all. Nominal width is a
request, not a fact — fleets hold only 85–90% of nominal — so without that check the arms could
have differed in width rather than in load, and nothing in either run's parameters would have
shown it.
