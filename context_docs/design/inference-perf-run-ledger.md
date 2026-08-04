# Perf campaign run ledger

Maps staging run-IDs to the code/config that produced them, so staged outputs
stay attributable after clusters tear down. All runs: Iowa ROI
(`s3://arbol-tessera-inputs-dev/mosaics/iowa_epsg5070`), time window
2024-11 → 2025-10, staged under `s3://arbol-tessera-embeddings-dev/staging/`.

| staging run-id | date (UTC) | code | batch | chunks staged | notes |
|---|---|---|---|---|---|
| `a85be572e2fb` | 2026-07-16 21:14→ | `main` | 3584 | 404 (complete) | **Baseline.** Phase-0 measurements: ~50–60 s GPU-idle overhead/chunk, TENSO 0.12–0.26, 9.6–13.3K px/s. Full run incl. assembly. |
| `526db07f32be` | 2026-07-16 22:35→ | branch @ `3be909f` (Phase 1 only) | 7168 | ~25 (cancelled early, no assembly) | **Phase-1 gate.** Prologue hidden (pref=Y rows: gap 0.0 s, overhead ≈ write only ~7 s); TENSO 0.42–0.47; CPU prep (get_batch 600–650 ms) gates inference — the Phase-2 target. Cancelled deliberately after verification. |
| `b826a6b3a44a` | 2026-07-16 ~23:55→ | branch @ `c816866` (Phases 1+2+3) | 7168 | — | **Phase-2+3 gate** (cluster `c884a457`, flow run `5d7cbc67`). Tarball fingerprint-verified (PREFETCH_DEPTH, `_pipelined_gpu_loop`, `_fused_weights`, `_local_resample_matrix`). Gate comparison: vs `526db07f32be`, same-config thresholds. |
| _(fill when known)_ | 2026-07-17 ~00:1x→ | `main` | 3584 | — | **Baseline re-run WITH assembly** (flow run `1e00dc16`, prod task def). The first baseline completed inference but was torn down before assembly — no final zarr; its staging (`a85be572e2fb`) survives. This re-run may resume from that staging or mint a new run-id — confirm on first stage write. |

Measured 2026-07-16, batch-size-only delta (baseline 3584 vs P1-gate 7168, 3 chunks,
512M values each): exact 95.4–98.3%, within-1 ≥ 99.9959%, max|Δ|=2, scale drift
≤ 0.78% (2 BF16 ULPs), cosine min 0.99990 / mean 0.99999, zero NaN mismatches —
i.e. FAIL vs same-config thresholds (as expected), PASS vs `--cross-config`.

Measured 2026-07-17, P2+3 gate (`b826a6b3a44a`) vs P1 gate (`526db07f32be`), same
batch, chunks 0_0/0_2/0_8 (1.5B values): **100.000000% bit-identical** — exact
match, max|Δ|=0, zero scale drift, zero NaN mismatches. Expected: Phase 2 is
bit-identical by construction, the PE change is bit-identical, and both builds run
the same pre-existing cuDNN `nn.GRU` fusion (the Phase-3 `CustomGRU` restructure
never reached production and was reverted — see PR discussion / commit 833c3be), so
there is no forward-math delta between the two. PASSES the strict ADR-012 gate.
(Coverage caveat: dense chunks; extend to few-valid-pixel labels via the cross-config
baseline comparison below.)

Performance, same chunks, wall-clock incl. load+write: baseline ~204–341 s/chunk
→ P2+3 89–140 s (**2.5–3.5×/worker**); inference px/s 9.6–13.3K → 13.4–27.3K;
GPU 100% util / busy 1.00 / ~338 W on saturated workers; get_batch 651 → 103 ms.

Host RAM + fleet GPU util (`observe_cluster.py --ram-report`, P2+3 window,
101 worker streams): peak host RAM **28.4 GB / 30.9 GB (92%)** in the 30s-polled
RESOURCES lines; the true instantaneous peak was **29.38 GB (95.1%)** — the
chunk_5_9 OOM-kill. This gate ran on `c816866`, an early PRE-strip-budget build
WITH full cross-chunk interleaving; that 95% is the removed interleaving design's
footprint, NOT `main` (which ran ~50%) nor the shipped pipeline. The shipped
striping run measured **45–47% peak** (Phase 4 entry below). Fleet sample-weighted avg
GPU util **79.4%** → total idle-recovery ceiling ~21% (incl. structural
write/cold-start idle); CPU-feed-specific slice ~7–15% GPU-hours (the g6e.2xlarge vs software tradeoff was assessed separately and shelved).

Few-valid-pixel / edge coverage 2026-07-17 (P2+3 vs baseline, `--cross-config`):
chunk_0_22 (3.5% valid): exact 99.85%, max|Δ|=2, cosine ≥ 0.99992, 0 NaN
mismatches; chunk_3_0: exact 98.06%, same envelope — PASS 2/2. Zero NaN-mask
mismatches on few-valid-pixel chunks confirms validity/skip semantics are unchanged
under the prefetched-prologue path.

Assembled full-ROI deliverables (icechunk stores, from Dask assembly):
- reference (main / 3584): `s3://arbol-tessera-embeddings-dev/embeddings/iowa_epsg5070-reference.zarr/`
- P2+3 (branch / 7168): `s3://arbol-tessera-embeddings-dev/embeddings/iowa_epsg5070-inference-speedup-phases-2-and-3.zarr/`
- phase 4 (striping / 7168): `s3://arbol-tessera-embeddings-dev/embeddings/iowa_epsg5070-inference-speedup-phase4.zarr/`

End-to-end check 2026-07-17 (3 interior 512x512 windows, cross-config; full ROI
is 1.87B px so sampled, not streamed whole): all PASS — exact int8 94.9-95.2%,
within-1 >= 99.9931%, max|Δ|=2, scale drift <= 0.78%, cosine >= 0.99991, and
**footprint_mismatch = 0** on the sampled windows. This is a spot-check, not
full-ROI proof; the Phase 4 entry below extends it to 25 windows / 2.23M px.

## Comparison semantics (ADR 012)

- `526db07f32be` vs `a85be572e2fb`: batch size differs (7168 vs 3584) → NOT
  bit-comparable; cosine-class thresholds only. Doubles as the empirical
  measurement of how far a batch-size change alone moves int8 outputs.
- P2+3 gate vs `526db07f32be`: same batch size, same GRU (cuDNN nn.GRU in both),
  Phase 2 bit-identical by construction, PE change bit-identical → expect exact
  bitwise equality under the strict same-config thresholds (confirmed above).

Compare with:

```
AWS_PROFILE=yield te-compare-outputs \
  s3://arbol-tessera-embeddings-dev/staging/<ref_run_id> \
  s3://arbol-tessera-embeddings-dev/staging/<test_run_id> \
  [--labels chunk_0_0,chunk_0_2,...]
```

## Phase 4 (striping) run — 2026-07-17

Flow-run `76f3137b`, cluster `tessera-inference-76f3137b`, 22 g6e.xlarge / L40S.
Full campaign + interleaving removal + striping P0–P3 (valid-pixel-aware
`_strip_plan`, budget 5.75 GiB, starter strip). Assembled output:
`.../embeddings/iowa_epsg5070-inference-speedup-phase4.zarr/` (dims
34964×53383×128).

Fleet: peak host RAM **45–47%** (vs 51% at the 4.75 GiB pre-striping budget —
lower despite the bigger budget, because more chunks run single-strip);
GPU util **~80.8%** (30 s poll) / ~87% (1 s DCGM); mid-density chunks
**21–24K px/s** end-to-end (was 16–22K pre-striping), dense 10–18K; `T≤71`
full-width chunks confirmed **single strip** (were 2); `write_s ≈ 0`
(background write); steady-state `overhead_s` **24–34 s**.

Correctness vs the `main` reference (`iowa_epsg5070-reference.zarr`), 25 sampled
384² windows / 2.23M valid px (cross-config, batch 3584→7168): **footprint
mismatch = 0**, **obs-layer mismatch = 0**, int8 exact 95.33%, within-1
**99.9947%**, max|Δ| **2**, scale drift **1.19%**, cosine min **0.999913** —
inside the ADR-012 cross-config envelope.

## Phase 5 (bounded cross-chunk starter prefetch) run — 2026-07-17

Flow-run `a60550ae`, cluster `tessera-inference-a60550ae`, autoscaled 22→30
g6e.xlarge / L40S, us-west-2a. 404 chunks in ~73 min (~34 GPU-hrs), 0 skipped,
0 failed, 0 stalled. **This is the final shipped state of the branch — use these
numbers for headline/current claims.**

Fleet: **prefetch hit-rate 100%** (630 starter prefetches → 630 hits; 0
mask-only / miss / cap-skip); per-chunk `overhead_s` **~6 s median on
prefetch-hit chunks** (n=251, 5.9 s median / 7.3 s mean) vs ~36 s on the
unavoidable first-per-worker cold starts (n=85, no predecessor to prefetch
from); GPU util **~89–93%** (CloudWatch: 89.1% whole-run incl. ramp/drain,
93.3% mid-run steady; 96.4% on 1 s DCGM — reads high); peak host RAM
**~52%** (16.1 GB / 30.9 GB — ~6 pts above phase 4 because the prefetch stash
is co-resident, still well under the 60% target); `write_s ≈ 0`. Per-chunk-class
px/s unchanged from phase 4 (bit-identical forwards); **fleet-overall ≈ 13–15K
px/s/worker** (1.87B-px ROI ÷ ~34 GPU-hrs — the whole-run average incl. cold
starts, density mix, and ramp; live-chunk-only basis ~13K).

Correctness: **bit-identical to phase 4** (output-preserving — the prefetch
changes *when* the prologue loads, not the tensors), spot-checked across 8
chunks / ~1.2B px (exact 100%, max|Δ|=0, obs-mismatch 0, cosine 1.0). So the
phase-4 cross-config comparison vs `main` above carries over unchanged. Full.

## P2 — the rate rung, three geographies — 2026-08-04

Every run above is Iowa. This is the first measurement at more than one geography, and its
purpose was to settle whether the campaign's throughput unit is tokens or pixels — see
`campaign-cost-model.md` §6, which had already switched to tokens on argument alone.

Three single-ROI runs via `tessera-full-pipeline`, dispatched at 22:0x UTC, all COMPLETED in
~151 min. Sites chosen to bracket the token range rather than to be dual-orbit (Cambridge had
already validated radar-free output, which is what let the rung shrink from six runs to three):

| site | region built | native CRS | area |
|---|---|---|---|
| boreal North America (NWT/Yukon) | `p2_boreal` | EPSG:32611 | 63,500 km² |
| Iowa — the continuity anchor | `iowa` (existing) | EPSG:5070 | 144,700 km² |
| humid tropics (Amazon) | `p2_amazon` | EPSG:32721 | 96,500 km² |

Aggregated over twelve `g6e.xlarge` actors, from the `tok/sec` line each sub-batch emits:

| | mean | range |
|---|---|---|
| **tok/sec per actor** | **1.90 – 1.93 M** | maxima all within 1% of 1,956,110 |
| **effective TFLOPS** | **85** | 84.7 – 86.2 |
| px/sec per actor | — | **12,420 – 27,285 (2.2×)** |

**Two results.** The reference rate of ≈1.9M tok/sec is confirmed at three geographies to
within ~1%, so the cost model's most load-bearing input is no longer a one-ROI figure. And the
unit question is settled empirically: tokens per second is flat to ±1% across the same twelve
actors over which pixels per second varies 2.2-fold.

Per-chunk duty cycle, from twenty `CHUNK_SUMMARY` records:

| | mean | min | max |
|---|---:|---:|---:|
| `infer_s` | 319.6 | 236.9 | 358.1 |
| `overhead_s` | 58.1 | 48.2 | 65.5 |
| `prologue_s` | 46.6 | 41.2 | 52.2 |
| `total_s` | 377.7 | 298.9 | 423.7 |

**Inference is 84.6% of chunk wall-clock**, which is the supply-side input P6's duty-cycle
criterion is a ratio against.

**One number to chase, not yet a correction.** Each actor's mean tok/sec over its mean px/sec
implies **70–153 tok/px** against the census's land-weighted 145. That is a ratio of averages
over sub-batches of differing valid-pixel density, so it is a weaker basis than the observation
census — but if the census is high, campaign tokens and inference cost fall with it. Settle by
counting observations on these three ROIs directly.

**Two registration bugs surfaced here and both are fixed** (`yield-embeddings`
`_base.py` / `deploy_flow.py`). The first attempt of all three runs died instantly on
`ObjectNotFound: None` — `tessera-full-pipeline` had no branch routing, so all four of its
stage refs pointed at prod deployments absent from the branch account. Separately every Ray
deployment stored an AMI parameter that does not exist, so each dispatch needed a manual
override. Registration now verifies both.
