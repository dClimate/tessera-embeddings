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

**Do not let Logs Insights do the arithmetic on these fields.** `filter CpuUtilized > 15700`
matched nothing in a window whose `sort CpuUtilized desc` returned 15741.9, and `max(CpuUtilized)`
gave two different answers over two windows holding the same records — string-versus-number
coercion in both directions, failing silently rather than erroring. Every figure above was computed
in Python from parsed floats over raw records; the only thing done server-side is the row fetch,
and the pull was checked against the 10,000-row cap (no task reached it) so that a truncated
population could not pass as a complete one.

## The measurement

Both runners are **ECS Fargate** — confirmed from the task definitions
(`yield-tessera-inference-flow-runner-prod` at 16,384 CPU units / 65,536 MiB,
`yield-tessera-assembly-large-flow-runner-prod` at 32,768 / 249,856, both declaring
`requiresCompatibilities: [FARGATE]`) and from the cluster itself, which has zero EC2 container
instances and only the `FARGATE` and `FARGATE_SPOT` capacity providers.

**The peak-memory minute and the peak-CPU minute are different tasks and different minutes, in
both families.** So they are two tables. Each row is one record: one task, one minute, every
column from it.

Peak **memory**:

| | 16 workers, 16 vCPU / 64 GiB | 32 workers, 32 vCPU / 244 GiB |
|---|---|---|
| task and minute | `559b69e3` at 09-09 22:05 | `7e9b7a9f` at 09-10 00:38 |
| **task memory** | **40.1 GiB — 62.7% of the box** | **93.5 GiB — 38.3% of the box** |
| CPU that minute | 66.7% | 33.4% |
| network rx / tx | 448 / 326 MB/s | 650 / 601 MB/s |
| memory ÷ workers | 2.51 GiB | 2.92 GiB |

Peak **CPU**:

| | 16 workers | 32 workers |
|---|---|---|
| task and minute | `07519e8e` at 09-09 22:14 | `66f9f331` at 09-10 06:42 |
| **task CPU** | **95.8% of 16 vCPU** | **99.7% of 32 vCPU** |
| memory that minute | 36.6 GiB (57.2%) | 66.8 GiB (27.4%) |
| network rx / tx | 677 / 313 MB/s | 942 / 474 MB/s |

Population: 25 tasks in the 32-worker family, **all 25** of which did real work; 20 in the
16-worker family, of which **only 4 ever exceeded a tenth of the box's CPU** — the rest are that
family's inference-dispatch runs, which never assemble. The 16-worker figures therefore rest on
four tasks.

## What it says

**A 16-worker pool peaks near 40 GiB and a 32-worker pool near 94 GiB.** Each attributable to one
task in one minute. This is what the sizing decision rests on, and it is solid.

**94 GiB does not fit the 64 GiB inference flow-runner family.** So `AssemblyConfig.max_workers`
stays 16, and only the campaign's chained fill — the sole deployment on the 244 GiB
`assembly_large` family — asks for 32. Measured, not projected.

**16 workers has less headroom than it looks: 62.7% of the 64 GiB box, and 73% on the metric
route's higher reading from the earlier run** (46.7 GiB on 09-09, outside the log's retention and
so not attributable to a task). The rest is unused reservation, not a second allocation — the
measured figure is whole-task, so the fill's coordinator and the Python runtime are already inside
it, and so is whatever the task was doing at its peak. There is simply less slack than the old
24 GB guidance implied, which is why a 16-worker pool wants the full 64 GiB rather than a
cut-down runner.

**Per worker is 2.51 GiB at 16 and 2.92 GiB at 32 — so the pool is NOT linear in worker count.**
Fitting a line to the two points gives 3.34 GiB per worker and a *negative* fixed overhead of
−13 GiB, which is the arithmetic saying two points from different runner families running
different cells cannot separate overhead from per-worker cost.

**So there is no per-worker sizing rule here, and the larger ratio is not an upper bound either.**
Taking 2.9 GiB per worker as a bound was tried and withdrawn: nothing measured stops a third pool
size — different fixed overhead, different cell geometry — from exceeding it, and an operator
sizing a runner from a bound that does not hold gets the OOM this document exists to prevent.
**Two sizes are measured. Any other needs its own measurement.**

**The processor reaches saturation at both pool sizes.** 95.8% of 16 vCPU and 99.7% of 32, each on
one task in one minute. Doubling the pool onto a doubled box raised the ceiling and a task climbed
back to it, so the pool is genuinely able to consume whatever processor it is given, and assembly's
own instrumentation agrees the work is computational — the `ASSEMBLY_SUMMARY` records put each
worker at 0.75-0.83 of a core from its CPU and wall-clock fields.

**That is not the same as the processor being what bounds throughput, and that question is still
open.** Utilisation reaching its ceiling at moments says the pool can saturate the processor, not
that the processor is what limits the rate of work: the peak-memory minutes ran at 33-67% CPU, so
saturation is intermittent, and Fargate publishes no per-task network allowance, so with receive
rates of 650-960 MB/s the headroom on the other candidate is unmeasurable from outside. Settling
it needs a controlled run, not more of this data.

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
3. **A second round from the CloudWatch metric route, 2026-09-10** — "46.7 GiB and 93.5 GiB
   peaks, 2.92 GiB per worker at both sizes, linear, CPU saturating at 96.8% and 99.7%". Mixed, and
   worth separating carefully, because the route was unable to *establish* any of it even where the
   number turned out to be right:
   - **93.5 GiB and 99.7% are confirmed** per task, by the records above. The metric route could
     not attribute them, which is a different fault from being wrong.
   - **46.7 GiB and 96.8% are not reproducible** per task; the highest attributable figures are
     40.1 GiB and 95.8%. Both belong to the earlier run, outside the log's retention, so they are
     neither confirmed nor refuted — only unattributable.
   - **2.92 GiB per worker at both sizes, and the linearity built on it, are withdrawn.** Per task
     the ratios are 2.51 and 2.92. The agreement was a coincidence between one confirmed number
     and one that could not be checked.

   **The lesson is the one at the top: a per-metric `Maximum` over a family is not a measurement of
   any task — including when it happens to be the right number.**

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
