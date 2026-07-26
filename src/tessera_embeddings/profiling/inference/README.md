# Inference performance harness

> **Scope: the INFERENCE half of the pipeline** — the Ray / GPU fill stage. For
> the ingest half (the Dask / Fargate scheduler + workers), see
> [`../ingest/`](../ingest/README.md). Start at [`../README.md`](../README.md)
> for which harness to reach for.

General-purpose profiling tooling for Tessera Ray inference runs — any
deployment, any scale (built during the GPU-saturation campaign; equally
aimed at global-tessera UTM-zone runs). Numerics policy the gates enforce:
`context_docs/decisions/012-validated-equivalence-for-inference-outputs.md`.

## Using it against any deployment

Workers are discovered by the Ray autoscaler's own EC2 tags
(`ray-cluster-name` + `ray-node-type=worker`), not instance names, so any
cluster works. Point the tool at a deployment with:

```
te-observe-cluster \
  --profile <aws-profile> --region <region> \
  --cluster <exact-ray-cluster-name>        # or --cluster-prefix <base-name>
  --log-group </ec2/.../ray>                # that deployment's CW log group
  --start-pollers | --report | --ram-report
```

**RAM-spike workflow** (the at-scale question these tools now answer):

1. `--start-pollers` early in the run — starts 1 s GPU pollers **and a 1 s
   host-RAM sampler** (used/avail/% + top-3 process RSS every second). The RAM
   sampler writes to an exact path the CloudWatch agent ships via a dedicated
   entry (`<instance>/ram_poll` stream) — so the 1 s data **survives
   teardown**. (Dedicated because the agent tails only the *newest* file per
   wildcard entry; a catch-all glob would drop samples and displace other
   logs. Clusters launched before the template gained the entry keep the data
   worker-local — `--report` still summarizes it live.)
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
        ┌─ per-chunk wall-clock anatomy, as --report lays it out ─┐
   gap+mask    band read    SAR+build    inference          write
  ├─────────┼────────────┼────────────┼══════════════════┼─────────┤
   GPU idle    GPU idle     GPU idle    GPU busy           GPU idle
  └──────── prologue: overlappable ────┘                  └ epilogue ┘
```

The shape is the point: everything outside the inference band is GPU-idle time
on that worker, so `--report`'s phase table tells you whether a run is limited
by the GPU or by everything around it. Absolute figures depend on the instance
type, the ROI's valid-pixel density and the batch size, so read them from your
own run rather than from a number written here.

Measurements from the campaign this harness was built for — baselines, the
per-phase progression, and what each change bought — are in
`context_docs/design/inference_gpu_saturation_profile_2026_07.md` and its run
ledger. They are recorded there rather than here so this file stays a
description of the tools, which does not go stale when the next run lands.
