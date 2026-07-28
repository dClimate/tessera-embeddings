# Tessera v2 Large: measured performance and correctness, 2026-07-28

Record for PR #98 (v2 Large student) and the two follow-up commits on the same
branch. Companion to
[`inference_gpu_saturation_profile_2026_07.md`](inference_gpu_saturation_profile_2026_07.md),
whose "shipped" column is the v1.1 baseline every comparison here is against.

**Headline: v2 roughly doubles dense-chunk throughput. Neither follow-up commit
moved the needle on this ROI, and one predicted win did not materialise.**

---

## 1. Provenance of the measurements

Two full Iowa runs, identical configuration, one commit apart. Both on the yield
deployment, 30 requested actors on `g6e.xlarge` (L40S), `s1_orbit=ascending`,
12-month window ending October 2025, 404 of 486 chunks intersecting the ROI.

| run | flow run | TE commit | output store | notes |
|---|---|---|---|---|
| A | `977c1e1c` | `ad956a3` | `iowa_epsg5070-v2-test.zarr` | complete, 402 chunks profiled |
| B | `fd8c441d` | `a6122de` | `iowa_epsg5070-v2-test-2.zarr` | + per-model rate estimate; figures below at 213 chunks |

The v1.1 reference store for correctness work is
`iowa_epsg5070-reference.zarr` (same ROI, same window, `tessera_v1_1_aws_encoder`).

Run A fleet settled at 30 workers; an earlier 5-actor attempt is not used here.

## 2. Throughput

Per worker. "infer denom" divides valid pixels by inference wall time; "end-to-end"
divides by total chunk wall time, so it carries the overhead in §4.

| metric | v1.1 shipped | run A | run B |
|---|---|---|---|
| px/s, dense (4M px) chunks | 10–18K | 23,026 mean / 21,864 med | 23,264 mean / 21,288 med |
| px/s, all chunks (infer denom) | 21–24K mid-density | 23,122 mean / 22,124 med | 23,165 mean / 21,243 med |
| px/s, end-to-end | ~13–15K fleet-overall | 20,142 mean | 20,109 mean |
| tokens/s | — | 2.02M | 2.03M |
| GPU utilization (fleet) | 89–93% | 84.5% (78.9–90.7) | 83.1% (78.4–86.5) |
| DCGM SMACT | ≈0.99 | ≈0.99 | 0.901 |
| DCGM TENSO | 0.42–0.47 | 0.442 (0.405–0.486) | 0.395 (0–0.496) |

**Dense chunks are where v2 wins** — roughly 21–23K px/s against v1.1's 10–18K.
Dense tiles were 316/402 (run A) and 166/213 (run B), so this dominates cost.

Raw px/s is confounded by kept-timestep count (54–118 across sampled chunks);
tokens/s normalises it and is strikingly flat, which says the model runs at a
consistent compute-bound rate and the px/s spread is a T artifact.

### Noise floor

Runs A and B agree within ~1% on every aggregate above. That is the useful
by-product: it makes the null result in §3 a real absence rather than an
undetectable effect. A 5% change would have been plainly visible.

### Correction: the tensor-pipe prediction was wrong

Before measuring, the expectation recorded was that v2 would *raise* tensor-pipe
utilisation by dropping v1.1's launch-bound CustomGRU, which the earlier profile
named as a bottleneck. It did not: TENSO is a dead heat with v1.1. The reason is
that shipped v1.1 had already reached 0.42–0.47 via the batch 3584 → 7168 change
(see the profile doc, §"Batch 3584 → 7168"), so there was no headroom on that
axis left for the GRU removal to recover.

## 3. Per-model planning rate estimate — NULL RESULT

`_EST_PX_PER_SEC` was a single constant of 16,000 calibrated against v1.1.
Commit `a6122de` made it per-model (v2 = 22,000, matching measurement) and threaded
it through `_strip_plan` and `_xchunk_rung`; commit `70cc11c` moved the table to
`config.inference.MODEL_EST_PX_PER_SEC` beside `MODEL_ARCHS`.

**It changed nothing measurable on Iowa.** Matched first-91-chunk windows:

| | run A (16,000) | run B (22,000) |
|---|---|---|
| `starter` rung | 67.0% | 65.9% |
| `serial` | 26.4% | 27.5% |
| `mask-only` | 6.6% | 6.6% |
| overhead mean / median | 24.0 s / 21.3 s | 23.5 s / 22.2 s |

At larger samples the distributions converge further (78.6% vs 77.9% `starter`).

**Mechanism.** The estimate can only flip a rung for *single-strip* plans — the
one branch in `_xchunk_rung` that divides by it. Only 17 of 101 sampled chunks
(16.8%) are single-strip. For the dense multi-strip plans that dominate Iowa, the
`t_infer >= t_load` comparison clears its threshold by a wide margin at both 16,000
and 22,000, so no decision changes.

**Correction to an earlier claim.** It was asserted that fixing the estimate would
convert the 123/402 prefetch misses into hits. That was wrong. Those misses come
from chunk geometry — budget-sized first strips and pair-budget plans, excluded by
construction — not from a mis-estimated rate. The lesson: check which decisions sit
*near* a threshold before predicting that moving the threshold helps.

The change is still correct (a stale per-model constant is the same class of defect
as §5) and may matter on an ROI dominated by sparse single-strip tiles. It is not a
speedup for Iowa.

## 4. Where the remaining time goes

Per-chunk overhead (everything that is not inference) is **18.5 s mean / 15.1 s
median / 36.3 s p90** in run A, against roughly 6 s median for v1.1 on prefetch-hit
chunks. 279 of 402 chunks were prefetch hits with zero prologue and *still* carried
it.

Overhead by rung (run A): `starter` 15.3 s, `serial` 30.1 s, `mask-only` 30.1 s.

Reading: v2's forward pass is fast enough that fixed I/O cost is no longer hidden
behind it. **v2 moves the bottleneck off the GPU and onto the read/write path** —
which also explains the utilization gap against v1.1 despite higher throughput.

### Staging write anatomy

- 418 writes measured: **19.9 s mean, 21.2 s median, 26.4 s p90, 30.0 s max**.
- `BAND_CHUNK_DIVISOR = 32` ⇒ 4 bands per object. A 2000×2000 chunk writes
  16 spatial tiles × 32 band groups = **512 objects for embeddings**, ~1 MB each,
  plus ~64 for scales and obs counts. **~576 objects, ~540 MB, uncompressed**
  (`"compressors": None`). So ~26 MB/s — request-count-bound, not bytes or CPU.

### Is Zarr's concurrency cap binding? Evidence says no

`zarr.config` `async.concurrency` defaults to **10** and is never overridden
anywhere in the source. Correlating 1 s outbound-:443 socket samples against
reconstructed write windows (`observe_cluster --start-pollers`, added for this),
over **117 write windows / 465 in-write samples across 10 workers**:

| | established :443 | median | max |
|---|---|---|---|
| during a write | 41.6 mean | 40 | 72 |
| baseline | 14.4 mean | 3 | 64 |

**No plateau at 10, and no plateau anywhere.** The in-write distribution spans
24–72, with broad clusters near 32–36 and 49–52 and no single value the samples
pile onto. A binding cap would show as a spike at that value; concurrency here is
demand-driven, not limit-driven. So the write is **not concurrency-limited** — its
~20 s cost is per-request latency or host CPU contention with inference batch prep.

The window overlaps the next chunk's background strip read, so some of those sockets
are reads rather than uploads. That does not rescue the cap hypothesis: even
attributing half to reads leaves the writer far above 10.

Implication: raising the cap will not help. Remaining levers are fewer/larger
staged objects (couples to the assembly read path, which pins
`500×500×(embedding_dim/BAND_CHUNK_DIVISOR)`) or accepting the cost. Deepening the
write pipeline past its current depth of 1 is **not** advised: the pending write
holds the ~512 MB embeddings buffer, and peak RAM is already 54% against the 55–60%
guard the earlier campaign set after an OOM kill.

## 5. Provenance defect found and fixed

A v2 run stamped `geoemb:model = https://geotessera.org/model/1.1` while
`checkpoint_id` and the manifest correctly read `student_large` — the store
advertised itself as v1.1. Cause: `ENCODER_VERSION` was a module constant pinned to
1.1, so nothing could vary the public URL per model. Fixed in `70cc11c` via
`MODEL_ENCODER_URLS` + `encoder_url()`, with `build_convention_attrs(encoder_version=…)`
kept distinct from `model_version` (the checkpoint stem).

`encoder_url` **raises** on an unregistered model rather than defaulting: wrong
provenance is worse than missing provenance because it is silent. This is the
opposite choice from `est_px_per_sec`, which falls back, because one is a
correctness value and the other a speed hint.

v2's public reference is its Hugging Face repo; it is not published under the
`geotessera.org/model/<version>` scheme, and inventing a path there would be a
fabricated identifier. Replace if a canonical one is minted.

Not yet exercised by a run — runs A and B both predate the fix.

## 6. Correctness

`te-compare-outputs` (ADR-012) **cannot be used**: it asserts ≥99.5% int8 exact and
cosine ≥0.9999, which encodes "same numerics, different code path". A different
model fails it by construction and the failure carries no information.

**Model port.** `tests/unit/test_student_v2_golden.py` runs our port and a verbatim
vendored copy of upstream side by side on the real checkpoint: 7 tests pass,
including forward agreement at four batch/shape combinations. Checkpoint payload
verified before the run — keys `args`/`model`, `latent_dim` 160, FFN 2560, 4 layers,
4 heads, `repr_dim` 128, QK-norm off, **43,831,170 parameters**.

**Readback vs the v1.1 reference** (921,600 pixels sampled, run A output):

| check | result |
|---|---|
| valid-pixel footprint | 0 mismatches |
| S1 asc/desc + S2 obs counts | 0 mismatches |
| per-vector mean / std | **−0.00000 / 1.00001** |
| neighbour agreement vs v1.1 | 0.537 overlap = **80.5× chance** |
| leading 16 dims vs full | 0.676 (v1.1: 0.561) |
| spatial coherence lift | +0.112 (v1.1: +0.173) |
| effective rank | 4.11 / 128 (v1.1: 5.92) |
| mean abs int8 code | 23.1 / 127 (v1.1: 37.2) |

The mean/std result confirms the non-affine LayerNorm head is live — v1.1 scores
−0.247 / 1.63 on the same statistic. Note that this invariant is *forced* by the
layer regardless of input, so it cannot detect wrong band statistics or permuted
band order; the structural checks exist for that. Neighbour agreement at 80.5×
chance against a same-model ceiling of 99.4× says v2 describes the same landscape.

**Quantization note.** v2 uses less of the int8 range than v1.1 (mean |code| 23.1
vs 37.2) with much tighter scales. This follows from unit-variance output: per-pixel
absmax is near-constant at ~3–4σ, so resolution is spent covering the tail. Not a
defect, but v2 gets slightly lower quantization SNR for the same storage budget.

Gate: `scripts/validate_embedding_model_change.py` in the yield-embeddings repo.

## 7. Operational notes

- **Prefect logs are not a reliable progress signal at fleet width.** In run A the
  flow logged nothing for the last ~75% of the run while work proceeded normally —
  the progress line fires several times a second and appears to hit an ingestion
  limit. CloudWatch `CHUNK_SUMMARY` telemetry stayed accurate throughout; use it.
- **`--start-pollers` truncates.** The GPU poller redirects with `>`, so re-running
  it to catch late-joining workers destroys samples already collected on the others.
  On a staggered fleet, instrument incrementally (new instances only).
- **One SSM tunnel per machine.** `arbol_prefect_client` binds a fixed local port
  (`DEFAULT_PORT = 8443`) and its readiness check only tests that *something*
  accepts there. A second process usually piggybacks on the first's tunnel and then
  dies with `httpx.ConnectError` when that one exits — or silently queries the wrong
  server if the two target different environments.
