# Ingest throughput at fleet scale — what we know, and what we still need

**Status: open.** One question decides the campaign's ingest line and is not yet answered; six
candidate explanations have been ruled out, and two effects are quantified and fixed or bounded.
Companion to `ingest_optimization_campaign_2026_07.md`, which remains the authoritative record
for everything measured before 2026-08.

## Why this matters

The cost model prices ingest at $115,000–$126,000 from a duration basis of **6.36 h per
zone-year at 60 workers**. Measured on 2026-08-04, three virgin zones projected **26, 28 and 47
hours**. If that gap is real and general, ingest trebles and stops being the cheap half of the
campaign. Everything below exists to decide whether it is real, and what causes it.

## The evidence, with sample sizes

| # | measurement | n | source |
|---|---|---|---|
| A | 3 virgin zones at 60w project 26/28/47 h/zone-year vs the fit's 6.5/10.5/15.1 | 3 zones, 47-min windows | this repo's logs |
| B | Same-zone width pairs: 6x workers buys **3.7–4.9x** | 3 zones x 2 widths, 8–37 dates/arm | scaling analysis |
| C | Per-date serial floor (build + unhidden stall + gate residual) = **19–24% at 60w**, 7–8% at 10w | 5 runs, 23–115 dates | scaling analysis |
| D | Fleets hold **85–90%** of nominal width; one 60-slot fleet registered **1,250 distinct workers in 5 h** | 6 fleets, 10,000 events | scaling analysis |
| E | **35N at 60w costs 300–359 s/date today vs 167.9 s/date in July** — same zone, same width | 60 and 147 dates | vs July record §3.15 |
| F | Same zone/width degraded **15–48%** from the 27-cell overnight wave to the 35-cell afternoon wave (but ~0% on 3 of 6 zones) | 6 zones | scaling analysis |
| G | The three 60w cells' per-date cost rose **204 → 269 s (+32%)** as fleet concurrency FELL from ~29 zones to 5–12 | 3 zones, 9 buckets | this investigation |
| H | Over that same window orchestrator CPU fell 13%→5.6%, requests 12.4→4.1/s, latency 146→39 ms, zero dropped events | 7 h | `prefect_load.py` |
| I | The slower (afternoon) wave ran on the **resized 4 vCPU** server; the faster overnight wave on the old **0.5 vCPU** one | — | resize at ~16:00 UTC |
| J | Dask schedulers: CPU 10–40%, event-loop lag 0.0–0.03 s, `processing` 255–357 against 240 slots | 20 fleets | scheduler health lines |
| K | 20 live fleets, 519 workers, **2,236 vCPU of 10,000**, and `no-worker=0` on every fleet | 20 fleets | scheduler health lines |
| L | **53N over 137 dates: total rose 1.36x, windows/date rose 1.45x, write per window FLAT at 16.4–18.3 s** | 137 dates, 1 zone | this investigation |

## What is ruled out

**Width usability.** B is the controlled version of A: 6x workers buys 3.7–4.9x, against an
Amdahl bound of 4.4–4.6x from the serial fraction measured at 10w. Width works about as well as
arithmetic allows. **A's 2.7–4.0x was a zone-mix artefact** — it compared different zones, and the
two 10-worker zones happened to be the two cheapest-per-chunk zones in the wave.

**The orchestrator.** H is the direct refutation: load fell by two-thirds and latency by
four-fifths across the window where per-date cost rose 32%. I is the corroboration — the server
gained 8x capacity between waves and the later wave was slower. Zero dropped events throughout.

**The Dask schedulers.** J. Well under the ~250-worker threshold in `docs/dask-scheduler-plan.md`,
with graphs oversubscribing their fleets rather than starving them.

**Capacity, quota and launch rate.** K. `no-worker=0` on all 20 fleets means no task anywhere is
waiting for a worker, at 22% of the vCPU quota.

> **A trap worth recording.** ECS `describe-clusters` reported **763 pending against 67 running**
> during this investigation, which reads as catastrophic launch starvation. It is an artefact:
> cluster statistics are eventually-consistent and this account carries **204 registered ephemeral
> worker families** from a day of fleet churn. It is contradicted by J and K, and the contradiction
> was visible immediately — 760 genuinely pending tasks cannot coexist with seven fleets sitting
> at 56–60 workers and committing dates. **Use the schedulers' own `workers=` and `no-worker=`
> fields; they are ground truth for width.**

**Commit serialisation.** Commits are ~1 s and the per-date cadence leaves only 5–8 s unaccounted
for after build + gate + write + stall.

**Store growth.** L settles this. Per-date cost rose 36% through 53N's run, which looked like a
store degrading under accumulating dates — but **write cost per window is flat at 16.4–18.3 s**
across all four quartiles, and build per window is flat at ~1.2 s. The cost rose because the
number of windows per date rose 45% (7.3 → 10.6). Manifest sharding is doing its job.

## What is established

**A per-date serial floor of 19–24% at 60 workers** (C). Real, bounds the width benefit, and worth
more than per-zone width tuning: the unhidden preparation stall alone is 20–45 s median on writes
of 160–300 s that ought to hide it.

**Adaptive churn cost ~10–12% of effective width** (D) — `adapt(minimum=1)` retired workers in
every inter-date gap and relaunched them cold. **Fixed**: `min_workers` now follows each leg's
derived width.

**Per-date cost grows within a run for a benign reason** (L). Later dates image more of a zone's
land — northern-hemisphere January is snow- and cloud-limited, by May footprints are wider — so
windows per date rise and cost rises with them. **Consequence: every velocity figure measured
early in a run UNDERSTATES the full-year cost.** The 26/28/47 h projections in A are optimistic
for this reason, independent of anything else.

**Width is nearly cost-neutral.** vCPU-seconds per date at 60w vs 10w averages ~1.1x over six
same-zone pairs. Sixty workers costs about what ten does per date and finishes 4–5x sooner, so
re-scaling `max_workers` is not where money is saved.

## What survives, unexplained

**E: every zone measured on 2026-08-04 ran 1.8–2.1x slower than the July record at the same zone
and the same width.** No width change addresses it, and it inflates every duration at both widths
— which makes it, not width, the thing that decides the ingest line.

**F and G point in opposite directions and cannot both be simple.** F says more cells is slower,
across waves. G says fewer cells is slower, within a run. L explains most of G (windows rose as
the run progressed, and the run progressed as the wave emptied), which means **G is largely not a
concurrency signal at all** — it is the seasonal effect confounded with elapsed time. That leaves F
as the only concurrency evidence, and F is weak: three of its six zones showed no degradation, and
its two waves differ in server size, time of day and date range as well as cell count.

Four candidates remain, and the data cannot separate them:

1. **Source-read contention above ~20 cells.** July measured contention only to 20 cells and
   explicitly left "aggregate source-read elasticity at large cell counts unmeasured".
2. **Time-of-day load on the public archive.** The slow wave ran 09:00–12:30 Pacific.
3. **The 2021 catalogue versus the 2024 dates** every July figure was measured on.
4. **Post-July configuration drift.** 35N now writes 18 windows/date against 13 in July §3.15 —
   which by L's arithmetic alone accounts for ~1.4x of the ~2x gap.

Candidate 4 deserves emphasis: if window count per date has risen 38% since July for
configuration reasons, then most of E is explained by the same mechanism as L, and the remaining
gap is small.

## The plan

**Stage 1 — separate contention from drift. LOADED ARM MEASURED; quiet arm outstanding.**
`julyref-35N-jan2024-loaded`: zone 35N, January **2024** dates, 60 workers, churn deliberately
left on for comparability. A second arm on February 2024 runs when the account is quiet.
Adjacent months of one zone at one width, which is the design B used.

### The loaded arm's result, 27 dates (2026-08-05)

Measured while the account carried **17 concurrent fleets and 803 Dask workers** with
`no-worker=0` on every scheduler — a genuinely loaded account, not a nominal one.

| | windows/date | s/date | write per window |
|---|---:|---:|---:|
| 35N Jan **2024**, loaded, 60w (this arm, n=27) | 15.0 | **196.3** | 11.04 |
| 35N Jun–Aug **2021**, loaded, 60w (n=85) | 18.0 | 344.1 | 16.71 |

**196 s/date sits far nearer the decision table's quiet branch (~168) than its loaded branch
(~300), measured under heavy load.** On its own that bounds the contention term at ~17 cells as
small: if source-read contention above 20 cells were the mechanism behind E, a 60-worker cell
sharing the account with 16 others should not run at close to the isolated July rate. The quiet
arm still completes the A/B and should still run, but it is now expected to confirm rather than
decide.

**Two cautions on the table above, both of which the earlier framing would have missed.** The two
rows differ in year *and* season, so neither the per-date nor the per-window column isolates
anything by itself — this is the cross-comparison error corrected in §"Corrections", reappearing
in a new guise. And the ~300/~168 figures in the decision table are per-date at unmatched
windows/date, so the comparison against 196 is indicative, not exact.

### The quiet arm runs on the yield account, which is a confound — read this before comparing

Dispatched 2026-08-05 02:38 UTC as `julyref-35N-feb2024-quiet-yield`: zone 35N, **February**
2024, 60 workers, `min_workers=1`, store `s3://arbol-tessera-inputs-dev/mosaics/35N/2024`.
global-tessera-dev was carrying 17 fleets at the time and could not host a quiet arm, while
yield held **one** ECS task against a 6,000 vCPU quota.

**What is matched, and verified in the run's own log rather than in the parameters passed:** the
zone, the width and churn (`Adaptive scaling configured: min=1, max=60`, identical to the loaded
arm's line), the container image (both accounts' `dev-global-tessera` tag was pushed by the same
CI build, two seconds apart), and the worker shape (`worker_cpu 4096`, `worker_mem 16384`, on the
pinned `yield-dask-worker-dev-global-tessera` definition).

**What is NOT matched:** the account, VPC, ECS cluster, Prefect server, and S3 bucket — and the
month, which is the intended variable. So a difference between the arms is "quiet-on-yield minus
loaded-on-dev", not "quiet minus loaded". Treat a *small* difference as strong evidence against a
contention term, since the confounds would have to cancel to hide one; treat a *large* difference
as unattributed until it is reproduced within one account.

**Getting there needed a fix, and the fix is the reason the arm is usable at all.** yield's
branch runner was pinned at revision 38 against dev's 142, missing four environment variables —
including both `DASK_{SCHEDULER,WORKER}_TASK_DEFINITION_ARN`. `register_branch_task_defs.py`'s own
docstring names that consequence: a runner pinning no Dask definition "would produce runners that
pin no Dask definition and therefore fall back to base-image fleets". The arm would have run at a
different worker shape and the comparison would have been silently void. Also missing was
`PREFECT_FLOWS_HEARTBEAT_FREQUENCY`, whose absence leaves the stock 180 s interval against the
crash detector's 300 s timeout — under two beats of margin, so the detector could have declared
this healthy run dead and swept its fleet mid-write. Re-registering brought yield to 15/15
variable parity before dispatch.

**The generalisation for any future cross-account measurement:** diff the runner task definition
against the reference account's before trusting the run, because the accounts drift independently
and the drift is invisible in the run's parameters.

**It also contradicts L's generalisation.** Write per window was flat across 53N's four quartiles
(16.4–18.3 s), which L used to argue it is *the* stable unit. On 35N it is not flat: within the
single 2021 run, at windows/date pinned at exactly 18.0 all summer, write per window falls
**18.71 → 16.11 → 14.15** from June to August, a 1.32x decline with the workload per date held
constant. Same zone, same store, same fleet, same width. So write-per-window is a better
normaliser than per-date cost but is **not a constant**, and a residual measured through it is not
automatically a real effect. What varies alongside it — elapsed wall-clock, account load, and the
store's accumulating size — is exactly the set Stage 1's paired arms are designed to separate,
which is another reason to run the quiet arm rather than stop at the loaded one.

| outcome | conclusion |
|---|---|
| loaded ~300 s/date, quiet ~168 | contention is real; the campaign needs a contention term above 20 cells |
| both ~300 | drift, not contention; July's basis is stale on its own terms |
| both ~168 | the gap is the 2021 catalogue, not the code or the load |

Cost: ~2 h, ~$25 per arm. **Read windows/date in both arms** — if it is 18 rather than 13, L's
arithmetic explains the gap before any contention term is needed.

**Stage 2 — settle candidate 4 from existing data, no run required.** Compare windows/date for
35N between the July record and today at matched date ranges. If it has risen, quantify how much
of E that explains via write-per-window, which L shows is the stable unit. This is the cheapest
remaining step and should precede any new experiment.

**Stage 3 — only if Stage 1 says contention.** A concurrency ladder: the same 3 zones at 60
workers, run at 10, 20, 35 and 55 concurrent cells, 90 minutes per rung, measuring per-date cost
normalised by windows/date. That normalisation is essential — without it the ladder re-measures L.
Cost: ~4 rungs x 90 min. Gives the contention term the campaign schedule needs.

**Stage 4 — close the instrumentation gap that forced proxies throughout.** Per-date **covered
chunks** is not logged, so every cross-zone comparison here uses the zone's live chunks as a
denominator and assumes similar imaged fractions. Add it additively to the `Stage timings` line —
that line is load-bearing for `campaign_progress.py`, so extend, never reorder. With it, L's
per-window normalisation becomes exact rather than a proxy, and cross-zone claims stop needing a
caveat.

**Not worth doing:** re-scaling `max_workers` per zone (width is cost-neutral and works), and
isolation runs to test concurrency (Stage 1's paired arms answer it more cheaply, and an isolated
per-cell ceiling is not the number the campaign operates at).

## Corrections this investigation made to its own earlier claims

Recorded because each was written down and acted on before being checked:

1. "Six times the fleet for under twice the rate" — a zone-mix artefact; the controlled figure is
   3.7–4.9x.
2. "Rising write with falling gate is a store-side signature" — wrong; write per window is flat
   and the rise is workload.
3. "Concurrency is the likeliest confound" — the evidence points the other way at every turn.
4. "763 tasks pending" — an ECS statistics artefact, contradicted by the schedulers.
