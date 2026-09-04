# Global store scale tests

Standalone benchmarks (NOT pytest) that settled the PENDING decisions in
[`context_docs/decisions/008-global-store-architecture.md`](../../../context_docs/decisions/008-global-store-architecture.md).
**These scripts ARE the program of record** — every D1–D9 is now FIRM in that ADR, annotated with
the run that confirmed it, and the per-test method, thresholds and cost budgets that once stood in
a separate plan are the code here plus git history.
[`context_docs/storage/icechunk-api-ledger.md`](../../../context_docs/storage/icechunk-api-ledger.md)
is the surviving half of that plan: the verified icechunk and zarr signatures and gotchas these
scripts were built against, not the benchmark contract.

These scripts write real data to real object stores and cost real money at
`--scale bench --backend s3`. Develop and smoke everything at
`--scale tiny --backend local` first (minutes, laptop, no AWS).

## Layout

```
scale_tests/
├── harness.py        config, metrics (JSONL), phase markers, timers, RSS, object stats, cold subprocess
├── variants.py       the 5 store-layout variants (chunk/shard geometry) — stable metric join keys
├── zone_geometry.py  mock UTM-zone shapes + annual (2017–2025) time axis
├── synth.py          deterministic int8 embeddings + coherent land masks (+ optional --real-sample)
├── seeding.py        multi-group empty-store seeding (the primitive the library still lacks)
├── store_builder.py  cooperative fork/merge year filler (shared by T1/T2/T3)
├── _workers.py       spawn-safe multiprocessing worker entrypoints
├── _subrunner.py     cold-read subprocess entrypoint
├── t0_smoke.py       smoke + cross-group conflict probe (run FIRST)
├── t1_read_bench.py  retrieval benchmark across variants   → ADR D2/D3
├── t2_write_bench.py write/commit scaling + manifest split → ADR D4/D6
├── t3_prealloc.py    pre-allocate vs prepend A/B           → ADR D1
├── t4_group_scale.py 120-group metadata scale              → ADR D5
├── t5_contention.py  concurrent-writer commit contention   → ADR D5/D6
├── t6_gc_bench.py    GC / expiry / rollback                → ADR D7
├── t7_ramp.py        fresh-bucket PUT ramp (S3 only)       → campaign warm-up
├── t8_sharding.py    shard vs unshard: write-align, bytes, → ADR D3 (see
│                     scattered reads, object count            ADR-008 D3)
├── report.py         collate metrics → decision matrix markdown
└── teardown.py       delete a run's stores; verify $0 residue
```

## Running

Always run from the `scripts/` directory so the spawned worker processes can
import `scale_tests`:

```bash
cd scripts

# smoke everything on a laptop (no AWS)
uv run python -m scripts.scoping.scale_tests.t0_smoke      --run-id dev --backend local --scale tiny
uv run python -m scripts.scoping.scale_tests.t1_read_bench --run-id dev --backend local --scale tiny --variant c256_full
uv run python -m scripts.scoping.scale_tests.t2_write_bench --run-id dev --backend local --scale tiny
uv run python -m scripts.scoping.scale_tests.t3_prealloc   --run-id dev --backend local --scale tiny
uv run python -m scripts.scoping.scale_tests.t4_group_scale --run-id dev --backend local --scale tiny
uv run python -m scripts.scoping.scale_tests.t5_contention --run-id dev --backend local --scale tiny
uv run python -m scripts.scoping.scale_tests.t6_gc_bench   --run-id dev --backend local --scale tiny

# the real campaign, in-region on the throwaway bucket
uv run python -m scripts.scoping.scale_tests.t0_smoke      --run-id run1 --backend s3 --scale bench --bucket <bucket>/global-embeddings/
uv run python -m scripts.scoping.scale_tests.t1_read_bench --run-id run1 --backend s3 --scale bench --bucket <bucket>/global-embeddings/ --variant c256_full
uv run python -m scripts.scoping.scale_tests.t2_write_bench --run-id run1 --backend s3 --scale bench --bucket <bucket>/global-embeddings/
uv run python -m scripts.scoping.scale_tests.t3_prealloc   --run-id run1 --backend s3 --scale bench --bucket <bucket>/global-embeddings/
uv run python -m scripts.scoping.scale_tests.t4_group_scale --run-id run1 --backend s3 --scale bench --bucket <bucket>/global-embeddings/
uv run python -m scripts.scoping.scale_tests.t5_contention --run-id run1 --backend s3 --scale bench --bucket <bucket>/global-embeddings/
uv run python -m scripts.scoping.scale_tests.t6_gc_bench   --run-id run1 --backend s3 --scale bench --bucket <bucket>/global-embeddings/
uv run python -m scripts.scoping.scale_tests.t7_ramp       --run-id run1 --backend s3 --scale bench --bucket <bucket>/global-embeddings/

# D3 settlement (separate run; see ADR-008 D3)
uv run python -m scripts.scoping.scale_tests.t8_sharding   --run-id d3   --backend s3 --scale bench --bucket <bucket>/global-embeddings/

# collate + tear down
uv run python -m scripts.scoping.scale_tests.report   --run-id run1
uv run python -m scripts.scoping.scale_tests.teardown --run-id run1 --backend s3 --bucket <bucket>/global-embeddings/
```

Install the extra deps once: `uv sync --group scale-tests`.

**Bench environment must pin `icechunk>=2.1.1`** (ADR D9): 2.0.4 has a
concurrent-manifest-fetch bug (#2158, ~25× slowdown) that poisons read
benchmarks.

### Common flags (every `tN` script)

| flag | meaning |
|---|---|
| `--run-id` | groups all artifacts of one run (required) |
| `--backend` | `local` (default) or `s3` |
| `--scale` | `tiny` (default, laptop) or `bench` (real numbers) |
| `--bucket` | S3 bucket, required with `--backend s3`. Accepts a bare name, `bucket/prefix`, or `s3://bucket/prefix`; a prefix becomes the default `--store-root`, and both the stores and the S3 results mirror are scoped under it (T7 uses the bare bucket name). |
| `--variant` | restrict to one variant (T1) |
| `--phase` | run a single phase by name |
| `--results-dir` / `--store-root` | override output/store locations |

Every phase is idempotent: a completed phase writes a `<phase>.done` marker and
is skipped on re-run, so a killed run resumes rather than restarts.

## Artifact flow

```
T0 smoke ──────────────┐
T1 read bench ──stores─┼──► T2 write bench ──► D2 chunk shape, D3 sharding,
T3 prealloc seed ──────┘        │               D4 split, D6 params
T4 120-group repo ──────────────┼──► T5 contention ──► D5 one-repo go/no-go
        └───────── all leftovers ┴──► T6 GC ──► D7 cadence
T7 ramp (standalone, S3 only) ──────► campaign warm-up
```

The scripts share code (the builder, seeding, harness) and metric schema; each
also builds its own stores so any single `tN` is independently runnable.

## When results land

1. `report.py` writes `scale_test_results/<run-id>/report.md` (the decision
   matrix with measured values). On S3 runs each test's local results are
   best-effort mirrored to `<bucket>/[<prefix>/]results/<run-id>/` — under the
   same `--bucket` prefix as the stores, so a prefix-scoped bucket stays writable.
2. Update the status of each affected decision in ADR-008 (`PENDING → FIRM`, or
   supersede with a new ADR — the decisions file is append-only for reversals).
3. Archive the report alongside ADR-008.

## Infrastructure a bench run needs

Absorbed here from the scale-test plan, which is retired. This is what the cost
estimate below is priced against, so the two belong together.

| item | spec | why |
|---|---|---|
| EC2 | `r7i.4xlarge` (16 vCPU / 128 GB), us-west-2 | RAM headroom for the commit-RSS tests — T2 sweeps to ~10⁷ refs, about 4 GB, plus the fork processes — and it matches the store's default region |
| Bucket | fresh, us-west-2, same account | throwaway; expect a `SlowDown` ramp on the first heavy writes |
| IAM | `s3:*` on the test bucket only | |
| Env | `uv` venv; **icechunk ≥ 2.1.1** (ADR-008 D9), zarr 3.2.1 pinned; package installed from the branch under test | |
| Results | `s3://<bucket>/<prefix>/results/<run-id>/*.jsonl` plus a local mirror — with the documented `--bucket <bucket>/global-embeddings/` that is `s3://<bucket>/global-embeddings/results/<run-id>/` | survives instance death. The mirror sits UNDER the `--bucket` prefix, beside the stores rather than at the bucket root, because a prefix-scoped bucket only grants write access under its prefix |

**Teardown**, in the order that works:

**Teardown is scoped by `--run-id`, and the sequence above uses TWO** — `run1` for T0–T7 and
`d3` for T8. Both need tearing down or the bucket cannot be emptied.

```bash
# 1. Stores only, once per run id. Results are KEPT by default — they are the run's product.
uv run python -m scripts.scoping.scale_tests.teardown --run-id run1 --backend s3 --bucket <bucket>/global-embeddings/
uv run python -m scripts.scoping.scale_tests.teardown --run-id d3   --backend s3 --bucket <bucket>/global-embeddings/

# 2. Archive the collated report.py output alongside ADR-008, and mirror the
#    results locally, BEFORE step 3 removes them from the bucket.

# 3. Results too, once they are safe elsewhere.
uv run python -m scripts.scoping.scale_tests.teardown --run-id run1 --backend s3 --bucket <bucket>/global-embeddings/ \
    --purge-results
uv run python -m scripts.scoping.scale_tests.teardown --run-id d3   --backend s3 --bucket <bucket>/global-embeddings/ \
    --purge-results
```

**Pass the SAME `--bucket` value the run used, prefix and all.** `--bucket` resolves the
store root: `<bucket>/global-embeddings/` gives `s3://<bucket>/global-embeddings/<run-id>`,
while a bare `<bucket>` gives `s3://<bucket>/scale_tests/<run-id>`. So a teardown that drops
the prefix cheerfully verifies an empty, unrelated prefix and reports success **while the
real 0.5–1 TB benchmark store stays billed.**

**And `teardown.py` verifies the STORE ROOT is empty, not the bucket.** Without
`--purge-results` the results prefix survives, and a `DeleteBucket` on a non-empty bucket
fails — so run step 3 before deleting the bucket, then confirm the bucket is empty yourself.
Finally terminate the instance and confirm no EBS orphans.

## Cost

**~$200–350 for a full bench run**, against the shape above: EC2 at ~$1.06/h for
40–80 h is **$45–85**; S3 storage of ~0.5–1 TB-month is **$12–25**; ~5–10 M PUTs
are **$25–50**; GETs and LISTs are minor; the rest is a buffer for reruns.
`teardown.py` returns the bucket to $0 — verify with its object-count output.

Develop and smoke everything at `--scale tiny --backend local` first, which costs
nothing.
