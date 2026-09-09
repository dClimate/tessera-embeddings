# What actually bounds assembly, measured 2026-09-09

Assembly runs inside the campaign fill's own Fargate task as a pool of forked worker processes.
This records what that box is actually limited by, because the pool size had been reasoned about
from an estimate rather than a measurement, and the estimate was wrong by roughly half.

Read this before changing `AssemblyConfig.max_workers` or moving a fill to a different runner.

Measured on a live campaign fill mid-assembly, forty one-minute Container Insights samples on the
fill's own Fargate task (16 vCPU / 64 GiB, 16 forked workers). This supersedes the earlier estimate
of 1 to 1.5 GB per worker and a ~24 GB pool, which was never measured at 16.

| resource | observed | reserved | utilisation |
|---|---|---|---|
| CPU | 12,868 units mean, 15,421 peak | 16,384 | **79% mean, 94% peak** |
| memory | ~38 GiB mean, 41.1 GiB peak | 64 GiB | 58% mean, 63% peak |
| network rx / tx | ~650 / ~410 MB/s | not published per task | unmeasurable |

**Assembly is CPU-bound.** Two independent instruments agree: the `ASSEMBLY_SUMMARY` records put
each worker at 0.75-0.83 of a core (`worker_cpu_s / worker_wall_s`), implying ~12.5 cores, and the
container consumed 12.9. Workers compute rather than wait — each moves only 32 MB/s while burning
three quarters of a core. So a wider pool on a bigger host buys real time.

**Per worker: ~2.4 GiB, not 1-1.5 GB.** So 16 workers peak near 40 GiB and 32 workers near 75 GiB.
**75 GiB does not fit the 64 GiB inference flow-runner family**, which is why 32 belongs only to
the 244 GiB `assembly_large` family (~31% there) and why `AssemblyConfig.max_workers` stays 16.

**The one thing that cannot be measured from outside, and it bounds the win from doubling.** Fargate
publishes no per-task network allowance, so the ~1.06 GB/s bidirectional (~8.5 Gbps) has unknown
headroom; 32 workers would ask for ~17 Gbps. A mild warning sign that something else sits near a
limit: twelve completed assemblies fell in a very tight 504-546 MB/s band across cells of
774-3,787 tiles and 0.57-2.76 TB, and a purely CPU-bound system at 79% should show more spread.
**The only test is to run one cell on a 32 vCPU task with 32 workers and read CPU plus network**:
clean scaling looks like CPU near 75% and throughput roughly doubled.

Care with the throughput figure: `bytes` in `ASSEMBLY_SUMMARY` is the READ side (per-worker fields
sum to the cell total) and is UNCOMPRESSED, while `fused_compress_put: true` means writes cross the
wire compressed. The two are not the same quantity. Average tile 729.8 MB; 21-23 s per tile per
worker.

## What this supersedes

The class docstring on `AssemblyConfig` previously said each worker holds 1 to 1.5 GB and the pool
peaks around 24 GB, with 32 workers around 48 GB. Those were estimates, never measured at 16, and
they understated the footprint by about half. The conclusion they supported was still correct —
`max_workers` stays 16 and only the campaign's chained fill asks for 32 — but the margin looked far
more comfortable than it is. Corrected in the same change that added this file.

Related: `assembly-wedges-during-fork-phase-2026-09-04.md` for what the fork pool does and how it
froze, and `../storage/writing-to-the-global-store.md` for the write path itself.
