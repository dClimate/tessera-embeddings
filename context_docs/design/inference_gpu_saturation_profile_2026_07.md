# Inference GPU saturation campaign — July 2026

Authoritative record of the inference optimization campaign behind PR #85
(branch `perf/inference-rethink`): what was changed, how much each change bought,
how correctness was preserved, and the gotchas future work must respect. This is
the empirical basis for why the pipeline looks the way it does.

**Complete per-run data + staging run-id mapping:
[`scripts/inference_perf/RUNS.md`](../../scripts/inference_perf/RUNS.md).** Raw
telemetry (TIMING / RESOURCES / EFFECTIVE-TFLOPS lines) survives in CloudWatch
(`/ec2/yield-embeddings/ray`); reproduce fleet stats with
`scripts/inference_perf/observe_cluster.py` (`--report`, `--ram-report`).

Hardware: g6e.xlarge — 1× L40S (181 TFLOPS BF16 dense, 46 GB VRAM, 864 GB/s),
4 vCPU, 30.9 GB usable RAM, 250 GB NVMe. Iowa ROI, 2000×2000 chunks.

## Headline numbers (final shipped state)

Numbers below are the **final shipped state**: the phase-5 run `a60550ae`
(bounded cross-chunk starter prefetch — the last optimization on the branch).
`main` is the batch-3584 baseline `a85be572e2fb`. Both single runs; see
[RUNS.md](../../scripts/inference_perf/RUNS.md) for the per-phase progression.

| metric | `main` baseline (batch 3584) | shipped (branch, batch 7168) |
|---|---|---|
| per-worker throughput | 1× | **~2–2.8×** |
| GPU utilization (fleet) | 48–72% avg; SMACT 0.31–0.68; TENSO 0.12–0.26 | **~89–93%; SMACT ≈0.99; TENSO 0.42–0.47** |
| inference px/s per worker | 9.6–13.3K | **21–24K** mid-density / 10–18K dense (inference-phase rate; bit-identical forwards, so unchanged since striping) |
| GPU-idle overhead per chunk | ~50–60 s (22–25% of wall) | **~6 s** median on prefetch-hit chunks; ~36 s on the unavoidable first-per-worker cold start |
| CPU batch-prep per sub-batch | 165 ms @3584 (651 ms @7168 unvectorised) | **103–160 ms** |
| peak host RAM | ~**50%** (estimated; not directly instrumented) | **~52%** (16.1 GB / 30.9 GB) |

The campaign originally measured ~2.5–3.5× at 100% GPU util **with** a full
cross-chunk prologue prefetch — but that co-resided two chunks' whole working
sets and OOM-killed a worker at 92–95% RAM. It was **removed** for UTM-zone-scale
safety (target peak RAM ≤60%, ideally ~50%) and replaced by the RAM-bounded
striping + starter-prefetch design below; the ~1.25–1.3× interleaving factor was
deliberately traded for headroom, then partially recovered by the bounded
prefetch. The numbers above are the current, RAM-safe config.

```
 per-chunk wall anatomy (single-strip chunk), GPU busy = ▓, idle = ░

 main:        [mask][bands][build][▓▓▓▓ inference ▓▓▓▓][write]
              ░░░░░░░░░░░░░░░░░░░░  ~50-60 s idle       ~7.5 s idle

 striping:    [mask][b][build][▓▓▓▓ inference ▓▓▓▓]     write runs in background,
              ░░ ~24-34 s ░░                            overlapping next prologue

 +prefetch:   [▓▓▓▓ inference N ▓▓▓▓][▓ N+1 starter ▓][▓▓ N+1 body ▓▓]
              next chunk's mask+starter preloaded during N's last strip
              (RAM trough) → GPU idle between chunks ~6 s on a hit
```

## The optimizations and what each bought

### A. Keep the GPU fed inside the forward loop — RETAINED, bit-identical
1. **Vectorised temporal resampling** (`sampling.py`/`dataset.py`). The per-pixel
   Python resample loop cost 600–650 ms/sub-batch at batch 7168 vs ~450 ms of GPU
   work — prep gated the GPU. Memoised (valid_count, target) index matrices +
   one vectorised gather brought it to **103–160 ms** (~1.3–1.4× on the inference
   component; bit-identical — same indices, same arithmetic).
2. **Async two-deep GPU pipeline** (`inference.py`). Pinned double-buffers + copy
   stream submit forward *i+1* before syncing *i*'s D2H; per-batch `isfinite`
   device sync moved to a host check on scales. ~1.1–1.15× (removes the
   per-sub-batch bubble). `TESSERA_SERIAL_GPU_LOOP=1` is the escape hatch.
3. **Batch 3584 → 7168.** TENSO 0.12–0.26 → 0.42–0.47 (~1.1–1.3×), previously
   masked by the prep bottleneck. The ONLY non-bit-identical change (cuBLAS
   kernel selection shifts with batch shape → ±1–2 int8-level shimmer).

### B. Hide the per-chunk prologue idle without OOM — the RAM-safety pivot
The prologue (mask read → first band read → dataset build) is GPU-idle. Full
cross-chunk interleaving hid it but OOM'd (§Headline). Replaced with:
1. **Valid-pixel-aware striping** (`_strip_plan` in `actors.py`). Read bytes scale
   with `T_kept × H × W` (independent of valid pixels); inference time scales with
   valid pixels — they diverge, so a single "strip height" is wrong. Four regimes:
   fits-one-budget → single strip; split+hideable (dense) → balanced strips with
   1-deep prefetch; split+not-hideable (wide but few valid px) → prefetch OFF,
   strips at the PAIR budget (only one set resident). RAM safety does **not**
   depend on the (log-calibrated) hideability estimate — every branch respects
   the pair ceiling; a wrong estimate degrades to the old always-prefetch behavior.
2. **Budget 4.75 → 5.75 GiB (P3).** Lets the whole `T≤71` full-width band load as a
   single strip instead of two, dropping a ~13 s fixed read/chunk (source stores
   use 4000² chunks → every read re-decompresses whole storage chunks; measured
   fixed read ≈13 s regardless of strip size). Counter-intuitively **lowered** peak
   RAM (51% → 45–47%): more single-strip chunks → fewer two-strip co-residency pairs.
3. **Starter strip (P2).** A small first strip so the GPU starts ~one fixed-read
   sooner; the body hides behind it. Trimmed steady-state overhead 30–44 s → 24–34 s.
4. **Bounded cross-chunk starter prefetch (§ actors `_XCHUNK_*`).** The next
   chunk's mask + 256-row starter (hard-capped ~2 GiB, NOT a whole working set)
   preloaded during the current chunk's LAST strip — the RAM trough, temporally
   separated from the mid-chunk two-strip peak. Skips pair-budget plans (their
   last strip holds ~2× a budget, not a trough). Every miss (cap, work-stealing
   reassignment, load error, `TESSERA_DISABLE_XCHUNK_PREFETCH=1`) reverts to the
   serial prologue. On run `a60550ae` this hit on 100% of prefetches and cut
   hit-chunk overhead to **~6 s median** (from striping's 24–34 s); the residual
   idle is the first-per-worker cold start (~36 s, no predecessor to prefetch).
   The co-resident stash raises steady peak RAM to **~52%** (from striping's
   45–47%) — the ~6-pt cost of the recovery, still well under the 60% target.

### C. Cheaper sparse/edge chunks — bit-identical
Empty strips skip the S2 band read; chunks whose valid pixels span a narrow
easting window read only that window (SAR read full-width so the saved
observation-count layers keep full extent). Sparse-chunk load dropped toward
bbox-proportional cost.

### D. Background staging write — bit-identical
Whole-chunk upload runs on a single-slot writer thread overlapping the next
prologue; chain-confirmation holds a chunk out of the completed set until its
write lands (failed writes requeue without killing the actor; staged writes are
idempotent). Removed ~7.5 s GPU-idle/chunk from the critical path.

### E. Cost & reliability — no output impact
Leak-proof teardown (flow-run-id-derived cluster tag survives Prefect's
fresh-process hook; `min_workers: 0` so no idle GPU floor; `resolved_yaml` bound
before `ray up` so failed launches tear down). The cluster is pinned to one AZ
(cross-AZ transfer is billed), chosen as the least-loaded subnet.

## Attribution — which change bought which improvement

- **Inference component (retained core, A):** ~1.3–1.4× resampler × ~1.1–1.3×
  batch-7168 GEMM × ~1.1–1.15× async pipeline ≈ ~1.7–2× on the compute itself.
  Isolated by TIMING-component arithmetic, not single-variable gates.
- **Overhead component (B+D):** GPU-idle overhead 50–60 s → 24–34 s (striping +
  background write) → **~6 s median on prefetch-hit chunks** (run `a60550ae`,
  100% hit-rate; ~36 s remains on cold first-per-worker chunks). The interleaving
  factor (~1.25–1.3×, overhead-only, measured as prologue 55→7.5 s) was removed
  then largely re-earned by the bounded prefetch.
- **RAM (B):** striping + the 5.75 GiB budget hold **45–47%** peak; the bounded
  prefetch stash adds ~6 pts → **~52%** final (`main` itself ran ~50%). The
  92–95% OOM was a *removed* full-interleaving iteration of this branch — the
  design constraint that motivated striping, not a main→shipped delta.
- **Sparse chunks (C):** load cost → bbox-proportional (biggest on sparse-heavy
  ROIs; small on Iowa where mixing already hid it).
- **Cost (E):** eliminated overnight instance leaks; capacity stalls mitigated at
  AZ-selection time.

Roughly: a third of the seconds saved came from hiding the loading, two thirds
from faster inference.

## Correctness

Everything except the batch-size increase is **bit-identical** (scheduling /
IO-shape only), enforced by golden tests (single-vs-multi-strip, prefetch
on/off/serial, cropped-vs-uncropped — all identical embeddings/scales/obs
layers). Numerics policy:
[ADR 012](../decisions/012-validated-equivalence-for-inference-outputs.md).
**Same-config gate** (branch phases held to bit-identity — e.g. P2+3 vs P1,
and phase 5 vs phase 4): **100.000000% bit-identical** (1.5B values), max |Δ| 0.

**Cross-config comparison vs `main`** (assembled Iowa output vs the batch-3584
reference store, 25 windows / 2.23M valid px): int8 exactly-equal **95.33%**,
within-1 **99.9947%**, max |Δ| **2**, scale drift **1.19%**, cosine min
**0.999913**, **0** footprint mismatches, **0** obs-layer mismatches. This is a
batch-size (3584→7168) diff, so it is judged against the **cross-config
envelope** (ADR-012 — within-1 ≥ 99.99%, max ≤ 3, drift ≤ 1.6%, cosine ≥ 0.9999,
footprint/obs exact), which it passes; it is **not** expected to meet the
same-config table (exactly-equal ~95% ≪ 99.5% is the cuBLAS batch shimmer of
fact 2, not a regression). Phase 5 is bit-identical to phase 4, so this
comparison carries over to the shipped state unchanged.

## Dead-end / shelved levers (measured, do not re-litigate)

- **FP16 fast-accumulate:** L40S GEMM microbench — BF16 already at the full dense
  ceiling (189 vs 181 datasheet TFLOPS); FP16 buys nothing (the GA10x half-rate
  penalty is A10G-specific). BF16 stays.
- **Adaptive token-budget batching:** real-model forward sweep — B=7168 is
  throughput-optimal at EVERY sequence length (even T=8 at 84/88 TFLOPS); larger
  B is neutral-to-worse. Shelved (`temp/token-budget-batching-findings.md`).
- **Eager bucketing (P4):** opens `dataset.py`; post-striping payoff is a sliver;
  rejected.
- **GRU restructure:** builder already fuses `CustomGRU` → cuDNN `nn.GRU`; reverted
  as dead code.
- **g6e.2xlarge (8 vCPU):** ~30% premium for ~7–15% feed-recoverable — software
  route preferred.

## Gotchas / bugs to be wary of

- **RAM budget is load-bearing.** Do NOT raise `_S2_STRIP_BYTE_BUDGET` or
  reintroduce whole-chunk cross-chunk prefetch without re-deriving the arithmetic
  at the constant. The pair ceiling (2× budget) plus the ~2 GiB prefetch stash is
  what keeps peak <60%. The prefetch MUST skip pair-budget plans
  (`_XCHUNK_UNSAFE_STRATEGIES`) — their last strip is not a RAM trough.
- **The strip-plan estimator is strategy-only.** `_EST_*` constants pick which
  safe strategy is fastest; they are NEVER a RAM bound. Every branch is RAM-safe
  regardless of estimate accuracy.
- **Output is NOT bit-exact vs `main`** — the batch-size change shimmers int8 by
  ±1–2 levels (cuBLAS). Compare with `compare_outputs.py --cross-config`, never an
  exact re-diff. Bit-exactness holds only at the same batch/config.
- **Shared CloudWatch log group across runs.** `--ram-report` / log greps must be
  scoped tightly with `--since/--until`; a broad window mixes concurrent runs.
  On-worker 1 s GPU poll files (`/tmp/gpu_poll.csv`) die with the workers — capture
  `--report` BEFORE a run's cluster tears down.
- **`min_workers: 0` defers worker capacity to the autoscaler.** `ray up`
  launches only the head, so a g6e worker-capacity shortfall surfaces later at
  autoscale time, not at launch. The single-AZ pin is chosen by least-loaded
  subnet (spread across concurrent clusters); it does not model spot capacity.
- **Testing: never boot a real Ray cluster in pytest.** `ray.kill` is wrapped in
  Ray's auto-init hook — an unpatched call in a unit test silently starts a local
  Ray cluster whose init hashes the whole working dir (multi-GB with scale-test
  stores present); three concurrent runs ate ~60 GB RAM once. Patch `ray.kill` in
  any test that reaches `ActorPool.replace`. Run ONE targeted pytest at a time.
- **A cropped chunk's SCL mask stays full-width** even when bands are cropped —
  strip sizing charges the mask at `chunk.width` (`mask_width`), and SAR is read
  full-width so obs-count layers keep full extent.

## Remaining headroom

Fleet GPU util ~89–93%; the residual gap is almost entirely the cold
first-chunk-per-worker prologue (~36 s, ~85 chunks on `a60550ae` as the fleet
autoscaled 22→30) — a chunk with no predecessor to prefetch from, which the
cross-chunk prefetch structurally cannot reach. The next structural lever is
source-store chunk geometry: the 4000² storage chunking drives the ~13 s
fixed read amplification — an inference-aligned geometry chosen before the global
UTM-zone ingestion would cut it for every future run (a config choice then, a
re-ingest later).
