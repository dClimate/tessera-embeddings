# Inference GPU saturation campaign — July 2026

Authoritative record of the inference optimization campaign behind PR #85
(branch `perf/inference-rethink`): what was changed, how much each change bought,
how correctness was preserved, and the gotchas future work must respect. This is
the empirical basis for why the pipeline looks the way it does.

**Complete per-run data + staging run-id mapping:
[`inference-perf-run-ledger.md`](inference-perf-run-ledger.md).** Raw
telemetry (TIMING / RESOURCES / EFFECTIVE-TFLOPS lines) survives in CloudWatch
(`/ec2/yield-embeddings/ray`); reproduce fleet stats with
`te-observe-cluster` (`--report`, `--ram-report`).

Hardware: g6e.xlarge — 1× L40S (181 TFLOPS BF16 dense, **45,776 MiB = 44.7 GiB
= 48.0 GB** VRAM, 864 GB/s), 4 vCPU, 30.9 GB usable RAM, 250 GB NVMe. Iowa ROI,
2000×2000 chunks.

> **Correction (2026-08-27).** This line previously read "46 GB VRAM". That was
> wrong: 45,776 is the MiB figure `nvidia-smi` and `ec2:describe-instance-types`
> both report, and it was rounded to 46 and relabelled GB. The card is 44.7 GiB
> (48.0 GB decimal), verified against `describe-instance-types` in us-west-2 on
> 2026-08-27. The error is small but it is in the denominator of every "% of the
> card" figure below, and it mattered once: the L4 and A10G (`g6.*`/`g5.*`) hold
> 22,888 MiB = 22.4 GiB, so this card is **1.94×** those, not "barely half again".
> `inference/README.md`'s "48 GB" was correct all along.

## Headline numbers (final shipped state)

Numbers below are the **final shipped state**: the phase-5 run `a60550ae`
(bounded cross-chunk starter prefetch — the last optimization on the branch).
`main` is the batch-3584 baseline `a85be572e2fb`. Both single runs; see
[the run ledger](inference-perf-run-ledger.md) for the per-phase progression.

| metric | `main` baseline (batch 3584) | shipped (branch, batch 7168) |
|---|---|---|
| per-worker throughput | 1× | **~2–2.8×** |
| GPU utilization (fleet) | 48–72% avg; SMACT 0.31–0.68; TENSO 0.12–0.26 | **~89–93%; SMACT ≈0.99; TENSO 0.42–0.47** |
| px/s per worker, per-chunk-class | 9.6–13.3K | **21–24K** mid-density / 10–18K dense (rate while processing a chunk of that class; bit-identical forwards, so unchanged since striping) |
| px/s per worker, fleet-overall | — (not separately derived) | **~13–15K** (1.87B-px ROI ÷ ~34 GPU-hrs on `a60550ae`; includes cold starts, density mix, ramp — the capacity-planning number) |
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

## The optimizations, and what each bought

Five changes shipped. The per-change working — method, A/B arms, attribution table and the
correctness argument for each — is compressed here because all five are in the code and their
combined effect is the headline above. What does not live in the code is which lever was worth
pulling and why, so that is what stays:

| lever | what it bought | note |
|---|---|---|
| keep the GPU fed inside the forward loop | the largest single win | retained, bit-identical |
| hide the per-chunk prologue idle | the second | **the RAM-bounded one** — it trades host memory for idle, so it is the change that has a ceiling |
| cheaper spatially-sparse / edge chunks | modest | bit-identical |
| background staging write | modest | bit-identical |
| cost and reliability changes | none on output | |

**Four of the five are bit-identical**, which is the property that made them safe to ship together:
correctness was pinned by comparing outputs rather than by reasoning about the change. The one that
is not — the prologue-idle hiding — is bounded by host RAM, and that bound is why it has a ceiling
rather than a knob.

## Dead-end / shelved levers (measured, do not re-litigate)

- **FP16 fast-accumulate:** L40S GEMM microbench — BF16 already at the full dense
  ceiling (189 vs 181 datasheet TFLOPS); FP16 buys nothing (the GA10x half-rate
  penalty is A10G-specific). BF16 stays.
- **Adaptive token-budget batching:** real-model forward sweep — B=7168 is
  throughput-optimal at EVERY sequence length (even T=8 at 84/88 TFLOPS); larger
  B is neutral-to-worse. Shelved.
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


---

## Appendix: the phase-0 baseline (moved here from the harness README)

The harness README describes the tools; these are the numbers one campaign got out
of them, which is why they live here.

### Baseline: 2026-07-16, Iowa ROI, main @ batch_size=3584, 4× g6e.xlarge (L40S)

Fleet GPU polls over ~10.5 min steady state (all 4 workers within a few %):

| metric | value |
|---|---|
| avg GPU util (nvidia-smi) | 48–57% |
| busy fraction (util > 5%) | ~0.70 |
| avg power | 219–248 W / 350 W |
| DCGM SMACT during inference | 0.31–0.68 |
| DCGM TENSO (tensor pipes) during inference | **0.12–0.26** |
| DCGM DRAMA | 0.25–0.49 |

Per-chunk phase splits (12 chunks, all single-strip, 40–70% valid):

| phase | wall | GPU |
|---|---|---|
| scheduler gap + SCL mask load | ~3 s | idle |
| S2 band read (5–8 GB) | 20–35 s | idle |
| SAR read + dataset build | 9–17 s | idle |
| inference (1.6–2.8 M px) | 137–221 s @ 9.6–13.3K px/s | oscillating 21–100% |
| staging write | 7–8 s | idle |
| **total** | **204–281 s** | **~22–25% fully idle** |

Reading: two independent losses. (1) Structural — ~50–60 s of GPU-idle
prologue/epilogue per chunk with no cross-chunk overlap (Phase 1 target).
(2) Within-inference — tensor pipes ≤26% active even mid-forward: CPU batch
prep gating (~165 ms/sub-batch), serial per-sub-batch H2D→forward→D2H
bubbles, and the launch-bound CustomGRU loop (Phase 2–3 targets). px/s is
density-dependent; completion logs now report tok/sec and effective TFLOPS
(`profiling.transformer_flops`) for cross-chunk comparison.

Note: production workers are g6e.xlarge (L40S 48 GB, **4 vCPU**, 32 GB RAM).
The L40S's tensor ceiling (~181 TFLOPS BF16 dense) is enormous relative to the
observed TENSO ≤0.26, and only 4 host vCPUs feed it — which is why the
CPU-side/pipeline bottlenecks above dominate.
