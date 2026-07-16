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
match, max|Δ|=0, zero scale drift, zero NaN mismatches. Phase 2 is bit-identical
by construction; Phase 3's GRU regrouping perturbs FP32 accumulations below the
BF16 rounding grid and collapses to bitwise equality on real data. PASSES the
strict ADR-012 gate outright. (Coverage caveat: dense chunks; extend to sparse
labels when the gate run reaches them or via the full-run comparison.)

Performance, same chunks, wall-clock incl. load+write: baseline ~204–341 s/chunk
→ P2+3 89–140 s (**2.5–3.5×/worker**); inference px/s 9.6–13.3K → 13.4–27.3K;
GPU 100% util / busy 1.00 / ~338 W on saturated workers; get_batch 651 → 103 ms.

## Comparison semantics (ADR 012)

- `526db07f32be` vs `a85be572e2fb`: batch size differs (7168 vs 3584) → NOT
  bit-comparable; cosine-class thresholds only. Doubles as the empirical
  measurement of how far a batch-size change alone moves int8 outputs.
- P2+3 gate vs `526db07f32be`: same batch size. Phase 2 is bit-identical by
  construction; every observed delta is attributable to Phase 3's GRU
  regrouping and must sit inside the ADR-012 thresholds.

Compare with:

```
AWS_PROFILE=yield python scripts/inference_perf/compare_outputs.py \
  s3://arbol-tessera-embeddings-dev/staging/<ref_run_id> \
  s3://arbol-tessera-embeddings-dev/staging/<test_run_id> \
  [--labels chunk_0_0,chunk_0_2,...]
```
