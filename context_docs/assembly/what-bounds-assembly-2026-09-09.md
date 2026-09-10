# What bounds assembly, measured per task 2026-09-09/10

Assembly runs inside the campaign fill's own Fargate task as a pool of forked worker processes.
This records what that box is actually limited by, because the pool size had been reasoned about
from an estimate rather than a measurement, and the estimate was low by roughly half.

Read this before changing `AssemblyConfig.max_workers` or moving a fill to a different runner.

## How it was measured, and why the first two attempts did not count

**Per task, from the Container Insights performance log**
(`/aws/ecs/containerinsights/global-tessera-prod/performance`), where every record carries a
`TaskId`. Rows are grouped by task, and each figure below comes with the other metrics *from the
same record* — one task, one minute.

That matters, because the obvious route gives a wrong answer. The CloudWatch **metric**
`MemoryUtilized` is published only under `{ClusterName, TaskDefinitionFamily}`, and each statistic
is aggregated across tasks independently. So a minute's memory `Maximum` and its CPU `Maximum` can
belong to **different tasks**, and filtering minutes by CPU does not restrict the memory figures to
the tasks that were assembling. Two earlier rounds of figures were built that way and both were
wrong in the same direction; they are recorded at the end so nobody repeats them.

**The performance log retains 24 hours** (confirmed: earliest record 2026-09-09 19:11Z, latest
2026-09-10 19:09Z). Anything older can only be reached through the metric route, with the caveat
above.

## The measurement

Highest-memory minutes of the worst task in each family, every column from that one record:

| | 16 workers, 16 vCPU / 64 GiB | 32 workers, 32 vCPU / 244 GiB |
|---|---|---|
| tasks in the population | 20 | 25 |
| **peak task memory** | **40.1 GiB — 63% of the box** | **93.5 GiB — 38% of the box** |
| CPU in that same minute | 64% | 33% |
| network rx / tx, same minute | 463 / 311 MB/s | 650 / 601 MB/s |
| **highest CPU any task reached** | **96.8%** | **87.8%** |
| memory ÷ workers at the peak | 2.50 GiB | 2.92 GiB |

## What it says

**A 16-worker pool peaks near 40 GiB and a 32-worker pool near 94 GiB.** Both attributable to a
single task in a single minute. This is the part the sizing decision rests on, and it is solid.

**94 GiB does not fit the 64 GiB inference flow-runner family.** So `AssemblyConfig.max_workers`
stays 16, and only the campaign's chained fill — the sole deployment on the 244 GiB
`assembly_large` family — asks for 32. Measured now, not projected.

**16 workers has less headroom than it looks.** 63% of the 64 GiB box on the per-task figure, and
the metric route saw 73% during the earlier run on 09-09 (46.7 GiB, outside the log's retention and
so not attributable to a task). Either way, about 17-24 GiB is left for the fill's own coordinator
and the commit, so a 16-worker pool wants the full 64 GiB. (**Not** for the Ray head: `ray up`
launches that on its own EC2 node and the flow runner only connects to it, so no head memory is in
these task figures at all.)

**Per worker is 2.5 GiB at 16 and 2.9 GiB at 32 — so the pool is NOT linear in worker count, and
these are whole-task ratios rather than per-worker footprints.** They include the fill's
coordinator and the Python runtime. Fitting a line to the two points gives 3.34 GiB per worker and
a *negative* fixed overhead of −13 GiB, which is the arithmetic saying two points from different
runner families running different cells cannot separate overhead from per-worker cost.

**So there is no per-worker sizing rule here, and the larger ratio is not an upper bound either.**
Taking 2.9 GiB per worker as a bound was tried and withdrawn: nothing measured stops a third pool
size — different fixed overhead, different cell geometry, a different runner family — from
exceeding it, and an operator sizing a runner from a bound that does not hold gets the OOM this
document exists to prevent. **Two sizes are measured. Any other needs its own measurement.**

**Assembly is compute-heavy, but the processor is NOT established as the binding constraint.**
A 16-worker task reached 96.8% of its 16 vCPU, so that pool very nearly saturates its box. A
32-worker task reached only 87.8% of its 32 vCPU, and its peak-memory minutes ran at 25-33%, so
doubling the pool left processor headroom rather than pinning it. Assembly's own instrumentation
agrees that the work is computational — the `ASSEMBLY_SUMMARY` records put each worker at 0.75-0.83
of a core from its CPU and wall-clock fields — and the campaign's outcome is consistent with a
real win from a wider pool. But Fargate publishes no per-task network allowance, so with rx of
650-922 MB/s at 32 workers the headroom there is unknown, and **which resource binds at 32 workers
is an open question.** Settling it needs a controlled run, not more of this data.

## What this supersedes

Three rounds, each understated or misattributed, all reaching the same decision:

1. **The `AssemblyConfig` docstring's estimate** — 1 to 1.5 GB per worker, a pool peaking around
   24 GB, 32 workers around 48 GB. Never measured at 16; low by about half.
2. **A first measurement round, 2026-09-09** — "~38 GiB mean, 41.1 GiB peak; 79% CPU mean, 94%
   peak". Those are the mean and maximum of **forty one-minute samples on one task**, published as
   the pool's figures. What is wrong with them is the generalisation, not the arithmetic of the
   sample: forty minutes of one task cannot state a population peak. And it cannot be checked now
   either — that run is outside the performance log's 24-hour retention, so where 41.1 GiB fell in
   its own run's distribution is unrecoverable. Read it as what it is: the largest of forty
   samples from one task. (The percentages published beside it were separately inconsistent with
   the figures they described.)
3. **A second round from the CloudWatch metric route, 2026-09-10** — "46.7 GiB and 93.5 GiB peaks,
   2.92 GiB per worker at both sizes, linear, CPU saturating at 96.8% and 99.7%". The 32-worker
   memory peak survives; everything else does not. The equal per-worker quotients were a
   coincidence between two independently mis-derived numbers, the linearity claim rested on that
   coincidence, and the 99.7% CPU was a cross-task artefact of independent aggregation — no single
   task exceeded 87.8%. **The lesson is the one at the top: a per-metric `Maximum` over a family is
   not a measurement of any task.**

All three rounds are filed in [`../corrections-register.md`](../corrections-register.md) under the
mechanisms that produced them — 1 (a condition left unlisted, here *which task*), 6 (a model fitted
on two points, and one sample generalised) and 7 (a figure published without measurement, and a
mechanism asserted before being measured).

## Care with the throughput figures

`bytes` in `ASSEMBLY_SUMMARY` is **logical write volume**, not read traffic, and it is uncompressed.
`_write_shards_worker` accumulates `block.nbytes` for every block handed to the destination Zarr
assignment (`storage/shard_writer.py`), and a *cleared* position's block is constructed locally by
`StagedShardSource._fill_block` without reading any staged object — so for a run with skipped tiles
the field counts bytes that never crossed the wire inbound. It also does not describe the outbound
side, because `fused_compress_put: true` means writes are compressed before they go. Use it for the
logical volume a cell moved; do not derive object-store ingress from it.

Related: `assembly-wedges-during-fork-phase-2026-09-04.md` for what the fork pool does and how it
froze, and `../storage/writing-to-the-global-store.md` for the write path itself.
