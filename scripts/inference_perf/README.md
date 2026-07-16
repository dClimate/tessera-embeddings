# Inference performance harness

Phase-0 tooling for the GPU-saturation campaign (see
`context_docs/decisions/012-validated-equivalence-for-inference-outputs.md`
for the numerics policy it gates).

## Tools

- **`observe_cluster.py`** — against a live `ray-tessera-inference-*` cluster:
  `--start-pollers` launches 1 s nvidia-smi + DCGM (SMACT/TENSO/DRAMA) captures
  on every GPU worker via SSM; `--report` fetches per-worker GPU summaries and
  a per-chunk phase-split table parsed from actor logs.
- **`compare_outputs.py`** — ADR-012 equivalence gate. Compares staged
  embeddings (int8 + scales) between a reference and a test run; exits nonzero
  if any chunk violates the thresholds (int8 ≥99.5% exact, max |Δ| ≤ 1 level,
  scale drift ≤ 0.1%, cosine ≥ 0.9999).

```
                    ┌─ per-chunk wall-clock anatomy (from --report) ─┐
   gap+mask   band read   SAR+build   inference              write
  ├───3s───┼───20-35s──┼───9-17s───┼════137-220s════════┼───7-8s──┤
   GPU idle   GPU idle    GPU idle    GPU 21-100% (osc.)   GPU idle
```

## Baseline: 2026-07-16, Iowa ROI, main @ batch_size=3584, 4× g6e.xlarge (L40S)

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
