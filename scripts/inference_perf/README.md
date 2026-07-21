# Inference performance harness

General-purpose profiling tooling for Tessera Ray inference runs — any
deployment, any scale (built during the GPU-saturation campaign; equally
aimed at global-tessera UTM-zone runs). Numerics policy the gates enforce:
`context_docs/decisions/012-validated-equivalence-for-inference-outputs.md`.

## Using it against any deployment

Workers are discovered by the Ray autoscaler's own EC2 tags
(`ray-cluster-name` + `ray-node-type=worker`), not instance names, so any
cluster works. Point the tool at a deployment with:

```
python scripts/inference_perf/observe_cluster.py \
  --profile <aws-profile> --region <region> \
  --cluster <exact-ray-cluster-name>        # or --cluster-prefix <base-name>
  --log-group </ec2/.../ray>                # that deployment's CW log group
  --start-pollers | --report | --ram-report
```

**RAM-spike workflow** (the at-scale question these tools now answer):

1. `--start-pollers` early in the run — starts 1 s GPU pollers **and a 1 s
   host-RAM sampler** (used/avail/% + top-3 process RSS every second). The RAM
   sampler writes into the Ray session log dir, which the CloudWatch agent
   already ships (`<instance>/other` stream) — so the 1 s data **survives
   teardown**.
2. `--report` while live: per-worker RAM summary (peak, seconds ≥55%/≥60%,
   top spike samples), OOM forensics (kernel OOM-killer + Ray memory-monitor
   events), GPU summaries, and the per-chunk phase table.
3. `--ram-report --since ... --until ...` any time after: 30 s RESOURCES
   rollup (always available; a *floor* for the true peak) plus, when the 1 s
   poller ran, per-worker 1 s peak/p99 and the top-10 spike samples with
   timestamps and the processes holding the memory at that instant.
4. Attribution: the actors tag every 30 s `RESOURCES` line with what they were
   doing (`ctx=work:<chunk>:<phase> write:<chunk>`), and emit one
   machine-readable `CHUNK_SUMMARY` JSON line per chunk — so any spike can be
   tied to a chunk + phase without prose-log archaeology.

## Tools

- **`observe_cluster.py`** — against a live cluster: `--start-pollers`
  launches 1 s nvidia-smi + DCGM (SMACT/TENSO/DRAMA) + host-RAM captures on
  every GPU worker via SSM; `--report` fetches per-worker GPU/RAM summaries,
  OOM events, and a per-chunk phase-split table (preferring the actors'
  `CHUNK_SUMMARY` JSON lines; legacy prose-log parsing remains as a fallback
  for runs from older code); `--ram-report` reconstructs RAM/GPU rollups and
  1 s spike analysis from CloudWatch after teardown.
- **`compare_coarsened_stores.py`** — bit-identity check between two
  *coarsened* (e.g. 500 m) embedding icechunk stores (float32 pre-dequantized
  `embeddings`, uint32 obs counts — no `scales`). Compares raw bit patterns;
  when not identical, reports max/mean |Δ|, an abs-diff CDF, per-pixel cosine,
  and NaN-mask agreement so quantization shimmer is distinguishable from a
  real defect. `--sample-rows N` for very large stores.
- **`compare_outputs.py`** — ADR-012 equivalence gate. Compares staged
  embeddings (int8 + scales) between a reference and a test run; exits nonzero
  if any chunk violates the thresholds (int8 ≥99.5% exact, max |Δ| ≤ 1 level,
  scale drift ≤ 0.1%, cosine ≥ 0.9999) or the structural checks (generated-mask
  agreement, exact obs-count layers, zero malformed scales — a scale must be
  NaN or finite-positive, never zero/negative/inf; and at least one generated
  pixel, since an all-empty `.zarr` should have been a `.skipped` marker). In
  directory mode it also rejects invalid staging up front: a chunk present as
  both a `.zarr` and a `.skipped` marker within one run fails before any
  comparison (assembly's `verify_staged_completeness` would refuse it).

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
