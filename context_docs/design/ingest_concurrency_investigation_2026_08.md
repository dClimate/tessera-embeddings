# Ingest throughput at fleet scale — what we know, and what we still need

**Status: resolved as posed; one narrower question left open deliberately.** The headline finding —
a 1.8–2.1x slowdown against the July record — **was an artefact of comparing summer dates against a
January baseline, and is withdrawn.** Matched on zone, width AND dates the gap is 1.12–1.17x. The
paired A/B then measured the contention term at **zero** (the quiet arm came out 3.3% dearer per
window, not cheaper), so none of that 1.17x is load at the width tested. Six candidate explanations
were ruled out along the way and two real effects quantified and fixed.

**What is left, and it is a schedule question rather than a diagnosis:** the A/B ran at 17
concurrent fleets, which is below the ~20 cells where contention was ever hypothesised and about a
sixth of campaign width, so **contention at 45–61 concurrent cells is still unmeasured** — Stage 3's
ladder is the way to price it. And the campaign's duration basis needs re-fitting on a **seasonally
weighted** year rather than a January rate, which is the one correction that genuinely raises the
ingest line.
Companion to `ingest_optimization_campaign_2026_07.md`, which remains the authoritative record
for everything measured before 2026-08.

## Why this matters

The cost model prices ingest at $115,000–$126,000 from a duration basis of **6.36 h per
zone-year at 60 workers**. Measured on 2026-08-04, three virgin zones projected **26, 28 and 47
hours**. Everything below exists to decide whether that gap is real, and what causes it.

**The answer, up front.** It is not a regression. Two separate reading errors inflated it: A
compared different zones (a per-chunk workload that varies four-fold), and E compared different
seasons on the same zone (18.0 windows/date in summer against 15.0 in January, and dearer windows
besides). The residual after both corrections is 1.12–1.17x, measured under load. **But the basis
itself is genuinely too low**, because it was fitted on January-conditions dates and a zone-year
is not twelve Januaries — summer dates cost 1.68x January dates on one zone at one width. So the
ingest line does need raising; it needs raising for seasonality, which is predictable and
schedulable, not for a defect.

## The evidence, with sample sizes

| # | measurement | n | source |
|---|---|---|---|
| A | 3 virgin zones at 60w project 26/28/47 h/zone-year vs the fit's 6.5/10.5/15.1 | 3 zones, 47-min windows | this repo's logs |
| B | Same-zone width pairs: 6x workers buys **3.7–4.9x** | 3 zones x 2 widths, 8–37 dates/arm | scaling analysis |
| C | Per-date serial floor (build + unhidden stall + gate residual) = **19–24% at 60w**, 7–8% at 10w | 5 runs, 23–115 dates | scaling analysis |
| D | Fleets hold **85–90%** of nominal width; one 60-slot fleet registered **1,250 distinct workers in 5 h** | 6 fleets, 10,000 events | scaling analysis |
| E | ~~35N at 60w costs 300–359 s/date today vs 167.9 s/date in July — same zone, same width~~ **WITHDRAWN: the two sides are different SEASONS. Matched, it is 1.12–1.17x** | 60 and 147 dates | vs July record §3.10/§3.16 |
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

## E IS WITHDRAWN — it compared different seasons (2026-08-05, Stage 2)

**E claimed every zone ran 1.8–2.1x slower than the July record at the same zone and width. It
does not. E compared summer dates against a January baseline**, and the matched comparison is
**1.12–1.17x**.

Stage 2 needed no run, only the two figures put on the same footing:

| | zone | width | dates | windows/date | s/date |
|---|---|---|---|---:|---:|
| July record §3.10/§3.16 | 35N | 60w | **January 2024** | — | **167.9 / 175.6** |
| E's "today" figure | 35N | 60w | **May–September 2021** (n=128) | **18.0** | **330.7** |
| Stage 1 loaded arm | 35N | 60w | **January 2024** (n=27) | **15.0** | **196.3** |

The July record's own reading instructions state it: *"Unless a figure says otherwise, every
timing is zone 35N, January 2024."* E honoured zone and width and silently violated the third
condition, which is the one that moves cost most. Matched on all three, the gap is
**196.3 / 167.9 = 1.17x** — and the loaded arm carried 17 concurrent fleets while the July figure
did not, so 1.17x is an upper bound on contention **plus** drift combined, not a floor.

The mechanism is L, applied across seasons rather than within a run: summer dates image far more
of a zone's land. 18.0 windows/date against 15.0 is 1.20x more work per date, and summer windows
are also individually dearer (write per window 16.7 s against 11.0 s), which together carry the
330.7-versus-196.3 ratio of 1.68x without any appeal to a regression.

**This is the third time this investigation has had to withdraw a claim for the same reason** —
see §"Corrections". Each time a real measurement was compared against another real measurement
whose conditions differed in a dimension nobody had listed. The reading-instructions block in the
July record exists precisely to prevent this, and I did not consult it before writing E.

### What it changes, and what it does NOT

**The code has not regressed.** No performance bug is being hidden by this, and the six ruled-out
causes stay ruled out.

**But the campaign's duration basis is still too low, for a benign reason.** The July fit
(5.95 h/zone-year at 60w) is built from January-conditions measurements, and a zone-year is not
twelve Januaries.

#### The seasonal profile of one zone, measured months marked

All 35N at 60 workers. **Six of twelve months are measured**; the rest are interpolated between
measured anchors or, for October–December, extrapolated by symmetry with no data at all.

| month | windows/date | s/date | write/window | source |
|---|---:|---:|---:|---|
| Jan (2024) | 15.0 | **196.3** | 11.04 | measured, n=27, loaded |
| Feb (2024) | 16.0 | **218.4** | 10.21 | measured, n=4 so far, quiet |
| Mar–Apr | — | ~250–290 | — | *interpolated* |
| May (2021) | 18.0 | **330.5** | — | measured, n=29 |
| Jun (2021) | 18.0 | **381.6** | 18.71 | measured, n=30 |
| Jul (2021) | 18.0 | **317.6** | 16.11 | measured, n=31 |
| Aug (2021) | 18.0 | **309.6** | 14.15 | measured, n=31 |
| Sep (2021) | 18.0 | **320.6** | — | measured, n=7 |
| Oct–Dec | — | ~210–290 | — | **extrapolated, no data** |

Two things the table makes clear that a single multiplier hides. **windows/date saturates at 18.0
by May and stays there** — it is not a smooth sinusoid, it plateaus once the zone is fully imaged.
And the summer premium is mostly **not** window count: 18.0/15.0 is 1.20x, while 330/196 is 1.68x.
The rest is that summer windows are individually dearer (write per window 16.7 s against 11.0 s),
because a window with less cloud and snow carries more valid pixels. Both effects push the same
way and neither is a defect.

Averaging that profile gives **~280 s/date against the fit's 167.9, i.e. ~1.67x** — consistent
with the 1.5–1.7x band, but note that a quarter of the year in it is extrapolated, so treat the
band and not the point. **The fix is a seasonal weighting of the basis, not a hunt for a
performance defect.** Pin it with per-date covered chunks (Stage 4) or one completed full-year
60-worker cell; the seven complete 2021 zone-years cannot serve, because they ran at 10 workers.

**Stage 1 is DONE and found no contention at the width tested.** Both arms complete: the quiet arm
came out **3.3% dearer** per window, not cheaper, so none of the 1.17x residual is load at 17
fleets. See the result table above, including why 17 fleets does not test the campaign-width
question.

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

**Stage 1 — separate contention from drift. ✅ DONE 2026-08-05, both arms complete.**
`julyref-35N-jan2024-loaded` (dev, 17 fleets, n=27) against `julyref-35N-feb2024-quiet-yield`
(yield, idle, n=28), 60 workers and `min_workers=1` in both. **Answer: the contention term at this
width is zero** — the quiet arm was 3.3% dearer per window. Result table, comparability checks and
the 17-fleet limitation are in §"The quiet arm runs on the yield account". The optional third arm
is documented there as not worth running.

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

**Achieved width is matched too, which is the check that would have voided this quietly.** Nominal
width is a request, not a fact, and evidence D records fleets holding only 85–90% of nominal. Read
off the scheduler health lines: the quiet arm ran at **median 57, max 60** workers with
`no-worker=0` on every one of 48 samples. Across the loaded arm's own window 28 dev fleets
reported, **14 of them at max 60**, and `no-worker=0` on all 28 — so neither arm was width-starved
and neither was quietly running narrow. Without this the arms could have differed in width rather
than in load, and nothing in either run's parameters would have shown it.

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

#### Result: no contention penalty at the width tested (2026-08-05, arm COMPLETED, 29/29 dates)

| arm | account | month | n | windows/date | s/date | **write/window** |
|---|---|---|---:|---:|---:|---:|
| loaded | global-tessera-dev, 17 fleets | Jan 2024 | 27 | 15.0 | 196.3 | **11.04** |
| quiet | yield, 1 other task | Feb 2024 | 28 | 17.0 | 221.8 | **11.40** |

**The quiet arm is 3.3% DEARER per window, not cheaper.** Under the reading rule fixed before the
arm ran — small difference means strong evidence against contention — this is that outcome, and it
sits the wrong side of parity, so the estimate of the contention term at this width is **zero, with
noise of a few percent**.

The per-date column is a clean confirmation of L rather than a second result: cost rose **1.13x**
while windows/date rose **1.13x**. Per-date cost tracked workload exactly, and per-window cost held
across two adjacent months in two different accounts.

**This also sharpens the correction to L.** Write per window is stable **between adjacent months**
(11.04 → 11.40) but drifts with **season** (11.0 in winter against 16.7 in summer on this zone). So
it is a sound normaliser for comparing nearby dates and an unsound one for comparing across a year.

**Method note for the next A/B:** the n=4 interim read 10.21 s/window — 8% *cheaper*, the opposite
sign to the final 28-date answer. It was recorded as preliminary and not cited. Four dates, one of
them a cold fleet, is not a measurement of anything.

> #### LIMITATION, and it is the important one: 17 fleets does not test the hypothesis
>
> Candidate 1 was **"source-read contention above ~20 cells"**, because July measured contention
> only to 20 cells and explicitly left "aggregate source-read elasticity at large cell counts
> unmeasured". **The loaded arm ran at 17 concurrent Dask fleets, 803 workers, ~3,200 vCPU — below
> that 20-cell line, and about one sixth of campaign width** (45–61 concurrent cells at 372 vCPU
> each is 16,700–22,700 vCPU).
>
> So this A/B does **not** test candidate 1. What it establishes is that nothing goes wrong up to
> ~17 fleets, a range July had largely covered already. **Contention at campaign width remains
> unmeasured.**
>
> What has changed is the *motivation*: with E withdrawn there is no 1.8–2.1x anomaly demanding a
> contention term to explain it, and no sign of a knee approaching 17. The remaining 1.17x is
> season, account and code drift, of which contention at this width contributes nothing
> measurable. A ladder at 25/40/55 concurrent cells (Stage 3) is now the only way to close the
> campaign-width question, and it is worth running for the schedule rather than for the diagnosis.

### A third arm is now OPTIONAL — the result made it cheap to skip

The two-arm design cannot separate "quiet" from "yield", so a third arm — 35N March 2024, 60w,
`min_workers=1`, on a quiet global-tessera-dev — was planned to give two one-variable comparisons:

| pair | varies | isolates |
|---|---|---|
| dev January (loaded) vs dev March (quiet) | load only | the contention term |
| yield February (quiet) vs dev March (quiet) | account only | the account confound itself |

**It is no longer worth $25 and two hours, because of the direction the answer came out in.** The
confound could only ever have *masked* a contention benefit — i.e. yield being intrinsically slower
could hide the loaded arm being genuinely penalised. But the loaded arm came out **cheaper**, so any
such masking would have to be undone to reveal a penalty, and the account effect is bounded at
3.3% regardless of which way it points. There is no room left for a contention term of consequence
at this width, whatever the account contributes.

**Spend the same money on Stage 3 instead**, which addresses the question that is actually still
open: 25, 40 and 55 concurrent cells, measured on write per window. Revisit the third arm only if
the ladder shows a knee and the account needs eliminating as its cause.

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

**Stage 2 — settle candidate 4 from existing data. ✅ DONE 2026-08-05, and it withdrew E.** The
comparison was never like-for-like: E's "today" side was May–September 2021 and its baseline was
January 2024. Matched, 1.17x rather than 1.8–2.1x. This was the cheapest step in the plan and it
was the one that mattered — it should have come first. See §"E IS WITHDRAWN".

**Stage 3 — a concurrency ladder. NOW THE ONLY OPEN QUESTION, and its purpose has changed.** It
was conditional on Stage 1 finding contention; Stage 1 found none, but only up to 17 fleets, which
is below the ~20 cells where contention was ever hypothesised and a sixth of campaign width. So the
ladder is no longer a diagnosis — **it is the schedule input**, and it should start where the
existing evidence stops rather than below it: **25, 40 and 55 concurrent cells**, 90 minutes per
rung, on the same 3 zones at 60 workers, measuring **write per window**. Normalising is not
optional; per-date cost re-measures L and would show a rise that is purely seasonal. Rungs below 20
would only re-confirm what Stage 1 and July already cover. Cost ~3 rungs x 90 min.

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
5. **"Write per window is THE stable unit"** — true on 53N, false in general. On 35N it falls
   18.71 → 14.15 s June to August with windows/date pinned at 18.0.
6. **"E: 1.8–2.1x slower than July at the same zone and width"** — the two sides were different
   SEASONS (May–September 2021 against January 2024). Matched, 1.12–1.17x.

**Four of these six are one error repeated:** a real measurement compared against another real
measurement whose conditions differed in a dimension that was never listed — zone in (1), keep
threshold in the withdrawn observation census, season in (6), and generalising one zone in (5).
The cure is mechanical and cheap: **before comparing two figures, write down the conditions of
each side and diff them.** The July record already carries a reading-instructions block naming
zone, width and dates as the three that must match; consulting it would have caught (6) in one
minute, and (1) as well.
