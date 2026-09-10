# What actually bounds assembly, measured 2026-09-09/10

Assembly runs inside the campaign fill's own Fargate task as a pool of forked worker processes.
This records what that box is actually limited by, because the pool size had been reasoned about
from an estimate rather than a measurement, and the estimate was wrong by roughly half.

Read this before changing `AssemblyConfig.max_workers` or moving a fill to a different runner.

## The measurement

Container Insights at one-minute resolution over the 2026-09-09/10 global campaign, on both
runners the campaign actually used. Figures are **per-task peaks** — the `Maximum` statistic, which
is the worst single task in that minute — restricted to the minutes when the pool was assembling
(peak task CPU above half the reservation). A flow runner's lifetime also covers ingest
coordination and inference dispatch, which are near CPU-idle, so a lifetime mean would mix in
phases where no worker process exists at all.

| | 16 workers, 16 vCPU / 64 GiB | 32 workers, 32 vCPU / 244 GiB |
|---|---|---|
| assembling minutes sampled | 642 | 1,097 |
| memory, median minute | 37.5 GiB | 86.5 GiB |
| **memory, worst minute** | **46.7 GiB — 73% of the box** | **93.5 GiB — 38% of the box** |
| memory per worker at that peak | **2.92 GiB** | **2.92 GiB** |
| CPU, median minute | 85% | 77% |
| **CPU, worst minute** | **96.8%** | **99.7%** |
| network rx, median / worst | 692 / 1,006 MB/s | 965 / 1,251 MB/s |
| network tx, median / worst | 413 / 520 MB/s | 651 / 877 MB/s |

## What it says

**Per worker: 2.92 GiB, and the pool is linear in worker count.** The two sizes agree to three
significant figures on a quantity neither was fitted to, which is the strongest evidence here:
fixed per-task overhead is small enough to ignore, so `workers × 2.9 GiB` is a usable sizing rule.
The superseded estimate of 1 to 1.5 GB per worker was low by a factor of two.

**Assembly saturates the processor of whatever box it is given.** 96.8% of 16 vCPU and 99.7% of 32.
This is no longer an inference from utilisation on one size: doubling the pool on a doubled box
raised the ceiling and the pool climbed straight back to it, which is what a compute-bound pool
does and what an I/O-bound one cannot. A third instrument agrees — the `ASSEMBLY_SUMMARY` records
put each worker at 0.75-0.83 of a core from its own CPU and wall-clock fields.

**Network rose but did not cap, and it is not the binding constraint.** Doubling the workers moved
peak receive by 1.24x and peak transmit by 1.69x, well short of double, while the processor went to
the wall. Fargate publishes no per-task network allowance, so there is still no headroom figure —
but the earlier worry that something unmeasured sat near a limit is answered: the constraint that
bound first was the processor, on both sizes. This also disposes of the tight 504-546 MB/s band
observed across twelve cells of very different sizes, which had looked like a hidden ceiling.

**32 workers does not fit the 64 GiB family**, by measurement now rather than projection: 93.5 GiB
against 64. So `AssemblyConfig.max_workers` stays 16 and only the campaign's chained fill, which is
the sole deployment on the 244 GiB `assembly_large` family, asks for 32.

**16 workers has less headroom than it looks.** The worst minute reached 73% of the 64 GiB box,
leaving about 17 GiB for the coordinator, the Ray head and the commit. Sizing a custom runner below
64 GiB for a 16-worker pool is not safe.

## What this supersedes

Two rounds of figures, both understated:

1. **The `AssemblyConfig` docstring's estimate** — 1 to 1.5 GB per worker, a pool peaking around
   24 GB, 32 workers around 48 GB. Never measured at 16; low by about half.
2. **A first measurement round, 2026-09-09** — "~38 GiB mean, 41.1 GiB peak; 79% CPU mean, 94%
   peak; 650/410 MB/s". Those came from forty one-minute samples on a single task, and they land on
   the **median** column above, not the peak: what was published as a peak was a typical minute.
   The per-worker figure derived from it, ~2.4 GiB, was correspondingly low, and the 32-worker
   projection of ~75 GiB understated the measured 93.5 GiB. Read a peak from the whole population
   of tasks over the whole run, not from a window that happens to be quiet.

The conclusion both rounds supported was right throughout — `max_workers` stays 16, and 32 belongs
only to the large runner — but each made the margin look more comfortable than it is.

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
