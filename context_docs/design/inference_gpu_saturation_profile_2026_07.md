# Inference GPU saturation profile — July 2026 campaign

Headline numbers and conclusions from the profiling campaign behind PR #85
(branch `perf/inference-rethink`). This is the empirical basis for why the
inference pipeline looks the way it does — cross-chunk prologue prefetch,
vectorised resampling, the async two-deep GPU loop, and the strip byte-budget.

**Complete per-run data, comparison tables, and staging run-id mapping:
[`scripts/inference_perf/RUNS.md`](../../scripts/inference_perf/RUNS.md).**
Raw telemetry (TIMING / RESOURCES / EFFECTIVE-TFLOPS lines) survives in
CloudWatch (`/ec2/yield-embeddings/ray`); reproduce fleet stats with
`scripts/inference_perf/observe_cluster.py` (`--report`, `--ram-report`).

## Headline numbers

Hardware: g6e.xlarge workers — 1× L40S (181 TFLOPS BF16 dense, 46 GB VRAM),
4 vCPU, 30.9 GB usable RAM. Iowa ROI, 2000×2000 chunks.

| metric | baseline (`main`, batch 3584) | after (branch, batch 7168) |
|---|---|---|
| wall-clock per dense chunk | 204–341 s | **89–140 s** (2.5–3.5×/worker) |
| inference px/s per worker | 9.6–13.3K | **13.4–27.3K** (~1.2–1.5M tok/s) |
| GPU util (saturated workers) | 48–72% avg | **100%, busy-frac 1.00, ~338/350 W** |
| DCGM SMACT / TENSO during inference | 0.31–0.68 / 0.12–0.26 | **0.99 / 0.42–0.47** |
| data loading per chunk (mask + bands + SAR + build, serial & GPU-idle) | 35–53 s = **15–21% of wall** (sparse chunks: up to ~86–99%) | **~0 s** (prefetched behind the prior chunk) |
| total GPU-idle overhead per chunk (loading + write) | ~50–60 s (22–25% of wall) | **~7.5 s** (the staging write) |
| CPU batch prep per sub-batch | 165 ms @3584 (651 ms @7168 unvectorised) | **103–160 ms @7168** |
| peak host RAM | — | 28.4 GB polled / **29.38 GB (95%) at the one OOM** (pre-strip-fix build; post-fix projection ~23 GB) |

```
 per-chunk wall anatomy      baseline:  [mask][bands][build][═══ inference ═══][write]
                                         GPU:  ~50-60 s idle        busy        idle
                             after:     [═══ inference ═══][write]      (prologue prefetched
                                         GPU: busy           ~7.5 s      behind prior chunk)
```

Correctness (ADR 012 harness): same-config gate **100.000000% bit-identical**
(1.5B values); batch-size change alone shimmer = exact 95–98%, max |Δ|=2,
cosine ≥ 0.9999; assembled end-to-end windows pass with **zero valid/NaN
footprint mismatches** (assembly integrity under the new pipeline).

## What the profile taught us (the load-bearing findings)

1. **The starvation was structural, not compute.** Baseline tensor pipes ran
   12–26% active even mid-forward; the GPU was idle 22–25% of wall at chunk
   boundaries and starved between sub-batches. All recovered gains came from
   scheduling (prefetch/pipelining), none from changing the math.
   **Attribution of the 2.5–3.5× (dense chunks):**
   ~1.25–1.3× data interleaving (cross-chunk prologue prefetch; gated in
   isolation) × ~1.3–1.4× resampler vectorisation (prep 650→103–160 ms —
   flipped the loop from prep-gated to GPU-bound) × ~1.1–1.3× batch-7168 GEMM
   efficiency (TENSO 0.12–0.26 → 0.42–0.47; present earlier but fully masked
   until the sampler stopped gating it) × ~1.1–1.15× async two-deep GPU
   pipeline (per-batch serialisation residue). Roughly: a third of the seconds
   saved from hiding the loading, two thirds from faster inference. Phase-2
   pieces shipped in one gate, so the within-gate split is TIMING-component
   arithmetic, not separately gated. On sparse chunks the attribution inverts:
   interleaving is essentially the entire win (loading was 86–99% of their wall).
2. **CPU prep gates the GPU when unvectorised.** At batch 7168 the per-pixel
   resample loop cost 600–650 ms vs ~450 ms of GPU work — the vectorised
   resampler (memoised index matrices, bit-identical) brought it to 103–160 ms.
3. **The GRU was never the production bottleneck** — the builder already fuses
   `CustomGRU` → cuDNN `nn.GRU`; a restructure attempt was reverted as dead code.
4. **Host RAM is the binding co-residency constraint.** Cross-chunk prefetch
   pairs two chunks' working sets; the strip byte-budget must hold every
   resident band set + its own mask to ONE budget (the chunk_5_9 OOM at 95%
   taught this; fixed in `_strip_height_for_density`).
5. **px/s is density-misleading** — completion logs now report tok/s and
   effective TFLOPS (53.9–67.6 sustained per chunk vs ~181 dense peak).

## Remaining headroom (measured, not speculative)

Fleet sample-weighted GPU util over the final gate window: **79.4%** across 101
worker streams → total idle-recovery ceiling ~21%, of which: the ~7.5 s/chunk
synchronous staging write + cold first-chunk prologues (structural), and a
CPU-feed-bound straggler tail worth **~7–15% of GPU-hours** (get_batch spikes to
~507 ms on dense chunks under 4-vCPU contention). Options and trade-offs
(software feed fixes vs g6e.2xlarge's 8 vCPU, token-budget batching, deferred
FP16 fast-accumulate) are analysed in `temp/token-budget-batching-findings.md`
and the PR #85 discussion; numerics policy in
[ADR 012](../decisions/012-validated-equivalence-for-inference-outputs.md).
