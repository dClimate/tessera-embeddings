# Global store scale tests

Standalone benchmarks (NOT pytest) that settle the PENDING decisions in
[`context_docs/decisions/008-global-store-architecture.md`](../../context_docs/decisions/008-global-store-architecture.md).
They implement the program in
[`context_docs/design/global-store-test-plan.md`](../../context_docs/design/global-store-test-plan.md)
per the handoff spec
[`global-store-test-impl-spec.md`](../../context_docs/design/global-store-test-impl-spec.md).

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
│                     scattered reads, object count            design/d3-sharding-plan.md)
├── report.py         collate metrics → decision matrix markdown
└── teardown.py       delete a run's stores; verify $0 residue
```

## Running

Always run from the `scripts/` directory so the spawned worker processes can
import `scale_tests`:

```bash
cd scripts

# smoke everything on a laptop (no AWS)
uv run python -m scale_tests.t0_smoke      --run-id dev --backend local --scale tiny
uv run python -m scale_tests.t1_read_bench --run-id dev --backend local --scale tiny --variant c256_full
uv run python -m scale_tests.t2_write_bench --run-id dev --backend local --scale tiny
uv run python -m scale_tests.t3_prealloc   --run-id dev --backend local --scale tiny
uv run python -m scale_tests.t4_group_scale --run-id dev --backend local --scale tiny
uv run python -m scale_tests.t5_contention --run-id dev --backend local --scale tiny
uv run python -m scale_tests.t6_gc_bench   --run-id dev --backend local --scale tiny

# the real campaign, in-region on the throwaway bucket
uv run python -m scale_tests.t0_smoke      --run-id run1 --backend s3 --scale bench --bucket arbol-tessera-embeddings-dev/global-embeddings/
uv run python -m scale_tests.t1_read_bench --run-id run1 --backend s3 --scale bench --bucket arbol-tessera-embeddings-dev/global-embeddings/ --variant c256_full
uv run python -m scale_tests.t2_write_bench --run-id run1 --backend s3 --scale bench --bucket arbol-tessera-embeddings-dev/global-embeddings/
uv run python -m scale_tests.t3_prealloc   --run-id run1 --backend s3 --scale bench --bucket arbol-tessera-embeddings-dev/global-embeddings/
uv run python -m scale_tests.t4_group_scale --run-id run1 --backend s3 --scale bench --bucket arbol-tessera-embeddings-dev/global-embeddings/
uv run python -m scale_tests.t5_contention --run-id run1 --backend s3 --scale bench --bucket arbol-tessera-embeddings-dev/global-embeddings/
uv run python -m scale_tests.t6_gc_bench   --run-id run1 --backend s3 --scale bench --bucket arbol-tessera-embeddings-dev/global-embeddings/
uv run python -m scale_tests.t7_ramp       --run-id run1 --backend s3 --scale bench --bucket arbol-tessera-embeddings-dev/global-embeddings/

# D3 settlement (separate run; see design/d3-sharding-plan.md)
uv run python -m scale_tests.t8_sharding   --run-id d3   --backend s3 --scale bench --bucket arbol-tessera-embeddings-dev/global-embeddings/

# collate + tear down
uv run python -m scale_tests.report   --run-id run1
uv run python -m scale_tests.teardown --run-id run1 --backend s3 --bucket arbol-tessera-embeddings-dev/global-embeddings/
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
| `--bucket` | S3 bucket, required with `--backend s3`. Accepts a bare name, `bucket/prefix`, or `s3://bucket/prefix`; a prefix becomes the default `--store-root`, and the bare bucket name is used for T7 + the results mirror. |
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
   matrix with measured values).
2. Update the status of each affected decision in ADR-008 (`PENDING → FIRM`, or
   supersede with a new ADR — the decisions file is append-only for reversals).
3. Archive the report alongside ADR-008.

## Cost

~$200–350 for a full bench run (see test plan §2): EC2 ~$45–85, S3 storage
~$12–25, PUTs ~$25–50, plus a rerun buffer. `teardown.py` returns the bucket to
$0; verify with its object-count output.
