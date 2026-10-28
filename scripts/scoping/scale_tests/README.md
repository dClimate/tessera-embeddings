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
├── variants.py       the v1.1 layout variants and the v2 geometries, with v2 codec/order/scales arms
│                     — stable metric join keys
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
├── t9_v2_layout.py   v2 layout arms, local: stored bytes, encode CPU,  → ADR 031
│                     request plans on two zarr versions and a
│                     browser client, Dask graph
├── t10_v2_aws.py     v2 finalists on AWS: cold S3 reads on two        → ADR 031
│                     clients, mobile reads on a shaped link, the
│                     assembly code's write cost
├── report.py         collate metrics → decision matrix markdown
├── scale_testing_results.md   archived reports, v1.1 runs (ADR-008)
├── v2_layout_results.md       archived reports, v2 layout runs (ADR 031)
├── v2_narrow_band_results.md  archived reports, narrow-band runs (ADR 031 §6)
└── teardown.py       delete a run's stores; verify $0 residue
```

## Running

Always run from the **repository root**. The modules import each other as
`scripts.scoping.scale_tests.*`, so the root must be on `sys.path`, which `python -m` does for
the working directory. The cold-phase runs spawn a subprocess that inherits this working
directory, so starting anywhere else fails there rather than at import:

```bash
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

# v2 layout (T9): local only. Build and read with the library's own environment, then read the
# same stores again from an environment pinned to the old client (zarr 3.2.1 + icechunk 2.1.1)
uv run python -m scripts.scoping.scale_tests.t9_v2_layout --run-id v2a --phase fetch --tiles <dir> \
    --source s3://<bucket>/<v2 single-area store> --profile <aws profile> --n-tiles 4
uv run python -m scripts.scoping.scale_tests.t9_v2_layout --run-id v2a --scale bench --tiles <dir>
<old-venv>/bin/python -m scripts.scoping.scale_tests.t9_v2_layout --run-id v2a --scale bench --tiles <dir>
# the narrow study: the same, with --study narrow (and results kept outside the system temp dir)
uv run python -m scripts.scoping.scale_tests.t9_v2_layout --run-id n9 --study narrow --scale bench --tiles <dir> \
    --results-dir temp/v2-layout/results

# v2 finalists (T10): in-region EC2 in the dev account. Reads on x86 (r7i), assembly on Graviton (m7g)
T10="-m scripts.scoping.scale_tests.t10_v2_aws --run-id v2b --backend s3 --bucket <bucket>/<prefix> --scale bench"
uv run python $T10 --tiles <dir> --phase build
uv run python $T10 --tiles <dir> --phase read          # then again from the old-client environment
uv run python $T10 --phase assemble --staging s3://<staged run dir> --extent <ny,nx> --n-tiles 64 --workers 16
uv run python $T10 --phase verify                      # then with the assemble line's --staging and --n-tiles
# the narrow study, under a run id of its own: a run holds one study
T10N="-m scripts.scoping.scale_tests.t10_v2_aws --run-id v2n --study narrow --backend s3 --bucket <bucket>/<prefix> --scale bench"
uv run python $T10N --tiles <dir> --phase build        # then read, points, tune and verify, as above
uv run python $T10N --tiles <dir> --phase mobile --network poor   # once tc shapes the link

# collate + tear down
uv run python -m scripts.scoping.scale_tests.report   --run-id run1
uv run python -m scripts.scoping.scale_tests.teardown --run-id run1 --backend s3 --bucket <bucket>/global-embeddings/
```

Install the extra deps once: `uv sync --group scale-tests`.

**Bench environment runs the library's own lock** (`uv.lock`). Never go below
`icechunk>=2.1.1` (ADR D9): 2.0.4 has a concurrent-manifest-fetch bug (#2158,
~25× slowdown) that poisons read benchmarks.

### T9 — the v2 layout, measured without AWS

v2's 128 dimensions are Matryoshka-ordered, so its store chunks the band axis inside the same 2048²
meta-shard, the block one inference run produces (ADR 031's vocabulary). T9 builds every arm as a
local store of 2×2 meta-shards from real v2 meta-shards, and records:

- what the writer pays: stored bytes, and encode CPU per meta-shard
- what every read asks for: requests and bytes for each workload at each prefix width
- the Dask tasks and requests of a 16-d pull, at Dask's default chunks and at shard-aligned ones

**Two studies, chosen with `--study`.** Each is a fixed arm list with the control (C0, v1.1's
geometry on v2 data) first, so T9 and T10 compare like with like. A run holds one study: phase
markers belong to a run, so `harness.bind_study` refuses a second study under the same run id.

- **`band16`**, the default, is the scoping behind ADR 031: the band-16 geometries in `variants.py`,
  each crossed with int8 codecs, a band-planar transpose, the inner-chunk order within a shard and
  the `scales` geometry, read at 16, 64 and 128 dims. Every archived command reruns as it ran.
- **`narrow`** measures narrower band chunks for mobile readers (ADR 031 §6), all band-first, against
  C0 and ADR 031's 16-band chunks: band-split shards of 8 or 4 bands at 512² or 256² inner chunks,
  16-band shards holding 8- or 4-band inner chunks, and variable-width band shards (4, 4, 8, 16, 32, 64
  and two coarser sets). It adds prefix widths of 4 and 8, a 256² map tile and a 1,024 × 2,048
  viewport, and a third request plan: the direct-read client of a browser or app, which merges no
  requests (`t9_v2_layout.DIRECT_READ`).

**Variable-width band shards** are a rectilinear chunk grid. Reading one needs zarr's
`array.rectilinear_chunks` setting on (`variants.py` sets it for every script; zarr 3.2.1 and 3.4.1
both read the grid then), and xarray ≥ 2026.9 for xarray readers. icechunk
accepts the grid only with every axis listed edge by edge, and zarr then requires every edge to be a
whole number of inner chunks, so such an array's extent is rounded up to whole inner chunks. Only
`embeddings` is rounded, so xarray refuses such a group unless its extent is already whole inner
chunks; the study reads these arrays through zarr.

**Requests are counted by a store wrapper** that also intercepts zarr ≥ 3.3's merged-range path.
`WrapperStore` would otherwise forward it unseen, and so would zarr 3.4's sync path.

**Three rules hold for T9 and T10 alike.** Every read quantity is recorded per read: a point
workload is many reads, one per probe. Readers and writers take an array's band order from its
dimension names (`variants.window_key`, with `read_window` and `to_storage_order`), so one path serves
band-last and band-first stores. And aligned Dask chunks are read from the array too
(`variants.dask_chunks`): one task per shard and band chunk, so the control's 128-band chunk is never
split.

**Keep T9's results outside the system temp dir** (`--results-dir`), and copy them to the dev bucket
beside T10's: macOS clears its temp dir, and the band-16 study's raw T9 rows were lost that way.

**Request plans carry over to S3, but latency does not.** The requests depend only on zarr's
sharding codec, so they hold on S3. Latency depends on the network, so T10 measures the finalists
there. Run the `requests` phase under both the library's zarr and an old client: zarr 3.3 changed
partial shard reads from one request per inner chunk to merged, concurrent ones.

### T10 — the v2 finalists on AWS

Each phase runs on its own in-region EC2 box, writing throwaway repos under the run's store root.
`--study` picks the finalists and workloads, as in T9.

- **`build` then `read`** (x86, r7i.4xlarge). Each finalist is built as a repo of 10 × 10
  meta-shards cycled from the real ones T9 fetched, so no cache holds the working set. Every workload runs cold
  in a fresh process, arms interleaved, in two passes of opposite order: points at 16 and 128 dims,
  a 128-d read of 2 × 2 meta-shards, an 8 × 8-meta-shard pull at 16, 64 and 128 dims, and Dask at its default
  chunks. `read` runs once per client. `points` repeats only the point reads, with bootstrap
  intervals on their medians. `tune` re-runs reads under reader settings: in the band-16 study the
  full-vector pulls with wider fetch concurrency and band-aligned Dask chunks; in the narrow study
  Dask at 16 and 128 dims with one task per shard (`dask_chunks(whole_shards=True)`), beside
  Dask's default chunks and the direct pull. Like `read`, it is reported per client. `build` also records the
  manifest bytes per S3 object, which a direct reader downloads before its first chunk.
- **`mobile`** (the read box, narrow study). The mobile reads, a 256² map tile and a 1,024 × 2,048
  viewport at 4 and 8 dims, without `scales`, through a browser-like client: no merged requests, at
  most 6 in flight, no GET split into parts. It runs on a phone link that `tc netem` and an `ifb`
  device shape first. `--network good|poor` names the profile (`t10_v2_aws.NETWORKS`); the phase
  measures the round trip (a TCP handshake with the bucket's endpoint, which S3's first-byte latency
  does not inflate) and the downlink, and refuses to run unless both are within a factor of two of
  the profile's. Each cold read fetches one distant pixel first, timed apart as the first read,
  since that is where the manifest download lands.
- **`assemble`** (Graviton, like the assembly runner). Staged v2 meta-shards are written into each
  finalist by the library's own `shard_writer.write_year_shards` over a `StagedShardSource`, every
  array included. Band-first arms are written only where the shard writer says it can
  (`WRITES_BAND_FIRST`, set by ADR 031's band-first prototype); otherwise their encode cost comes from
  `build` on a Graviton box instead.
- **`verify`**. Every arm holds exactly C0's values in every array, band-first ones reordered first.
  It checks the built stores; given `--staging`, it checks the assembled ones, over the
  meta-shards assembly wrote. The set chooses its own stores and arms, so there is no prefix to pass.

### Common flags (every `tN` script)

| flag | meaning |
|---|---|
| `--run-id` | groups all artifacts of one run (required) |
| `--backend` | `local` (default) or `s3` |
| `--scale` | `tiny` (default, laptop) or `bench` (real numbers) |
| `--bucket` | S3 bucket, required with `--backend s3`. Accepts a bare name, `bucket/prefix`, or `s3://bucket/prefix`; a prefix becomes the default `--store-root`, and both the stores and the S3 results mirror are scoped under it (T7 uses the bare bucket name). |
| `--variant` | restrict to one variant (T1) |
| `--phase` | run a single phase by name; a phase recorded per client (`requests_zarr3.4`) also answers to its plain name |
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
3. Archive the report beside the others: `scale_testing_results.md` holds the v1.1 runs behind
   ADR-008, `v2_layout_results.md` the band-16 T9/T10 runs behind ADR 031, and
   `v2_narrow_band_results.md` the narrow study behind its §6, each section under its run's own
   provenance.

## Infrastructure a bench run needs

Absorbed here from the scale-test plan, which is retired. This is what the cost
estimate below is priced against, so the two belong together.

| item | spec | why |
|---|---|---|
| EC2 | `r7i.4xlarge` (16 vCPU / 128 GB), us-west-2 | RAM headroom for the commit-RSS tests — T2 sweeps to ~10⁷ refs, about 4 GB, plus the fork processes — and it matches the store's default region |
| Bucket | fresh, us-west-2, same account | throwaway; expect a `SlowDown` ramp on the first heavy writes |
| IAM | `s3:*` on the test bucket only | |
| Env | `uv` venv from `uv.lock` (never below **icechunk 2.1.1**, ADR-008 D9); package installed from the branch under test | |
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
