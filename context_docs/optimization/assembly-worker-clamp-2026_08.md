# The S3 budget clamp cost assembly two thirds of its fork pool

**Status: fixed.** `_s3_budget_split` reduced the assembly fork count to fit a per-fill S3
request budget. On the global campaign that budget is always far below the requested worker
count, so the clamp bound on *every* assembly, and assembly is the campaign's longest stage.

## What was measured

Every global assembly emits one `ASSEMBLY_SUMMARY` line. Seven exist in the campaign's history
to 2026-08-28. All seven:

| run | workers_requested | workers_used | per_worker_s3_cap | fill_wall (h) |
|---|---:|---:|---:|---:|
| 48N-2017 | 16 | 5 | 1 | 5.78 |
| 32N-2017 | 16 | 5 | 1 | 5.74 |
| 47N-2017 | 16 | 5 | 1 | 6.16 |
| 38N-2017 | 16 | 5 | 1 | 5.09 |
| 33N-2017 | 16 | 5 | 1 | 6.29 |
| 43N-2018 | 16 | 5 | 1 | 5.96 |
| 38N-2018 | 16 | 5 | 1 | 5.11 |

Dead flat across six zones and three weeks. A fixed number that never responds to job size is
the signature of a clamp, not of a resource limit.

Two live assemblies observed at the same time wrote shards at 264 and 270 per hour despite being
different zones with different data volumes. That near-identity is the same finding seen from
outside: a shared serialising constraint rather than per-zone work.

## The arithmetic

`run_global_campaign.py:1595` passes `s3_concurrency = TARGET_AGGREGATE_S3_CONCURRENCY // (2 *
n_clusters)`. With the target at 100 and ten clusters that is **5**. `_s3_budget_split` then did
`workers = min(n_workers, budget)`, turning a 16-worker assembly into a 5-worker one and setting
each fork's request cap to 1.

The clamp was deliberate and documented. What was not anticipated is that the campaign's own
divisor makes it bind unconditionally at fleet width.

## Why it was invisible

Every place a human configures the worker count still said 16. `AssemblyConfig.max_workers` is
16; `compute_n_workers` returns 16 for these zones; the flow requests 16. Only the emitted
`workers_used` disagreed — and that field pair exists precisely to expose this case, its
docstring saying "a fill quietly running below its requested width is exactly what this pair
exposes." Nothing read it.

The clamp also silently reverted a deliberate change. `max_workers` was raised from 8 to 16 on
2026-08-06 from a measurement: at 8 the box was half idle, with processor use peaking at 7,443
of 16,384 allocated units and memory at 20 of 64 GiB. The clamp then held the effective count at
**5 — below the 8 that had already been found too small.**

## The fix, and the invariant it gives up

`workers = n_workers`; the budget divides only the per-worker cap.

Because the cap floors at 1, aggregate concurrency becomes `max(budget, n_workers)` rather than
`<= budget`. **The floor and the ceiling cannot both hold.** The old code held the ceiling by
dropping forks; the new code holds the fork count and lets the ceiling give way.

That is a real widening and it should be stated plainly:

| | per cluster | fleet (10 clusters) |
|---|---:|---:|
| before | 5 × 1 = 5 | 50 |
| after | 16 × 1 = 16 | 160 |

The nominal fleet target is 100, so this runs at **1.6× the target**. The bound on the overshoot
is `AssemblyConfig.max_workers × n_clusters`.

The justification is asymmetry of cost. Overshooting the target risks 503 `SlowDown`, which
retries. Holding the target by dropping forks costs wall-clock unconditionally, on every cell,
whether or not the service was ever going to complain. The repo's own recorded observation puts
`SlowDown` at **800 concurrent PUTs** (`assembly.py:236`), so 160 sits at about one fifth of the
level where the problem was actually seen.

## Expected gain, and its status

Not measured. From the recorded per-worker figures, read is the dominant phase and is roughly
half blocked on S3, so the box runs at roughly a quarter of its 16 cores at 5 workers. Going to
16 workers is 3.2× nominal; allowing for the processor becoming the constraint, **2.5–3× is the
reasonable expectation**, taking a zone-year from about 5.7 hours to 2–2.3. This is arithmetic
over recorded processor and wall-clock fractions, **not a measurement of the changed code.**

## Memory

The one thing that could sink this. Each worker holds at most one staged-tile slice, about
1–1.5 GB, so 16 workers is estimated at roughly 24 GB. The flow runner is 16 vCPU / 64 GiB and
`consumer_stack.py` sized it explicitly for `n_workers=16` at about 19 GiB. The only *measured*
figure is 20 GiB at a pool of 8. The estimate therefore has wide headroom but has never been
observed at 16, and it should be watched on the first real assembly.

## Not verified

- No assembly was run with the change; the speedup is arithmetic, not measurement.
- Resident memory at 16 workers has never been observed.
- The `2 *` factor in the campaign's budget divisor is left alone. Whether the staging and
  published buckets carry independent request budgets was not established.
- The `t7_ramp.py` PUT-ramp harness exists (`scripts/scale_tests/`) but **no recorded results
  exist in this repository**. The 800-PUT `SlowDown` figure is the repo's own code comment and
  was not traced to a primary record. If we want empirical headroom rather than an inherited
  number, T7 is the tool.
