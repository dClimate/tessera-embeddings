# Global store scale tests — implementation spec

Handoff document for implementing `scripts/scale_tests/`. Written for an
implementer (human or model) who has NOT seen the research behind it.

Read first, in order:
1. `context_docs/decisions/008-global-store-architecture.md` — the decisions
   these tests settle. Do not re-litigate them.
2. `context_docs/design/global-store-test-plan.md` — WHAT each test measures,
   pass/kill thresholds, artifact flow, cost budget. This spec is the HOW.

Ground rules for the implementer:
- Scripts are standalone benchmarks under `scripts/scale_tests/`, not pytest.
  They may import `tessera_embeddings`, but **do not modify library code** to
  make tests pass — findings tell us what to build later.
- Follow repo conventions: stdlib `logging`, type hints, no inline imports,
  ruff/mypy clean (`uv run ruff check`, `uv run mypy`).
- Every icechunk/zarr API named here was verified against icechunk 2.0.4 /
  zarr 3.2.1 on 2026-07-13. The bench environment must use **icechunk ≥ 2.1.1**
  (ADR D9); re-verify signatures there with `inspect.signature` before use —
  do not guess if something moved.

---

## 1. Build order (milestones, each independently verifiable)

| # | Deliverable | Verify by |
|---|---|---|
| M0 | `harness.py`, `variants.py`, `zone_geometry.py`, `synth.py`, `t0_smoke.py` | `t0` green on `--backend local` on a laptop, then on S3 |
| M1 | `t1_read_bench.py` (+ store builder shared with T2/T3) | `--scale tiny --backend local` end-to-end; JSONL validates |
| M2 | `t3_prealloc.py` (seeding phases), `t2_write_bench.py` | tiny/local; T2 consumes T3's store artifact |
| M3 | `t4_group_scale.py`, `t5_contention.py` | tiny/local (12 groups, N≤8) |
| M4 | `t6_gc_bench.py`, `t7_ramp.py` (optional) | tiny/local except T7 (S3-only) |
| M5 | `report.py`, `teardown.py` | report renders decision tables from a tiny full run |

Everything must run at `--scale tiny --backend local` (minutes, laptop, no
AWS) before it runs at `--scale bench --backend s3`. Local backend uses
`icechunk.local_filesystem_storage`; the library's `_create_storage` already
branches this way — mirror it, don't reinvent.

## 2. Shared contracts (pin these exactly — they are the interlock)

### 2.1 CLI (identical for every tN script)

```
python -m scale_tests.tN_name --run-id RUN --backend {local,s3} --scale {tiny,bench}
       [--bucket BUCKET] [--variant NAME] [--phase NAME] [--real-sample PATH]
```

- `--phase` runs one phase; default runs all phases in order.
- Idempotency: before each phase, check marker `results/<run-id>/<test>/<phase>.done`
  (local file and/or S3 object); skip if present; write marker on success.
- All tunables (point counts, sweep values, N lists) are module-level
  constants with the test plan's values as defaults — not CLI flags.

### 2.2 Metrics

One JSONL file per (test, phase), appended via a single emitter:

```python
def emit(row: MetricRow) -> None: ...   # harness.py

@dataclass
class MetricRow:
    run_id: str; test: str; variant: str | None; phase: str
    metric: str; value: float; unit: str
    params: dict[str, Any]        # year, refs, n_workers, concurrency, ...
    # emitter stamps: ts, icechunk/zarr versions, git_sha, scale, backend
```

Fixed metric vocabulary (report.py joins on these — never improvise names):
`wall_s`, `commit_wall_s`, `merge_wall_s`, `peak_rss_bytes`, `read_p50_ms`,
`read_p95_ms`, `bytes_fetched`, `throughput_mbps`, `open_wall_s`, `retries`,
`refs_committed`, `manifest_count`, `manifest_bytes`, `snapshot_bytes`,
`objects_listed`, `objects_deleted`, `bytes_reclaimed`, `gc_wall_s`,
`slowdown_503_count`, `puts_per_s`.

### 2.3 Harness utilities (`harness.py`)

- `phase(name)` — context manager: marker check/skip, wall timing (emits
  `wall_s`), success marker on clean exit.
- `rss_sampler()` — context manager sampling `psutil.Process().memory_info().rss`
  at 1 Hz in a thread (include children for fork/merge tests:
  `psutil` iterate over `children(recursive=True)`); emits `peak_rss_bytes`.
- `object_stats(prefix)` — S3 LIST (paginated) returning count + total bytes;
  local backend: `os.walk`. Used for `objects_listed`, `manifest_bytes`
  (list `manifests/` under the repo prefix), `snapshot_bytes` (`snapshots/`).
- `cold_read(fn_module, args)` — run a measurement in a fresh
  `subprocess.run([sys.executable, "-m", ...])` so nothing is cached in-process.
  Warm = same call in-process, second invocation.
- Repo opening: reuse
  `tessera_embeddings.storage.zarr_store._default_repo_config` for
  timeouts/retries, passing extra config (splitting/preload) on top. The
  `manifest_split()` contextmanager in `zarr_store` is the supported way to
  apply splitting through library paths.

### 2.4 Variants (`variants.py`)

Frozen dataclass registry, exactly the five entries from test-plan §3
(`c500_band4`, `c500_full`, `c384_full`, `c256_full`, `c256_sharded`).
Fields: `name, chunks: tuple[int,int,int,int], shards: tuple|None`.
Constants: dims `("time","northing","easting","band")`; `embeddings` int8 +
PCodec serializer (`tessera_embeddings.inference.assembly.pcodec_serializer`),
`compressors=None`, fill 0; `scales` float32, fill NaN; band=128.

### 2.5 Geometry & data (`zone_geometry.py`, `synth.py`)

- `MockZone`: tiny = 2,048×2,048 px; bench = 20,000×20,000 px;
  `--full-height` variant (10⁶ × 4,096 px) for metadata-only seeding tests
  (T4). Coordinates: synthetic UTM-like `northing` descending / `easting`
  ascending float64 at 10 m spacing; `time` = `np.datetime64(f"{y}-01-01")`
  for y in 2017–2025, int64-ns encoded via `zarr_store.TIME_ENCODING`.
- `synth.embedding_block(shape, seed)` — deterministic per (block index,
  seed): `np.random.default_rng(hash)` int8 in [-127,127]. Deterministic so
  read-back verification is possible without storing inputs.
- `synth.land_mask(chunk_grid, fraction=0.7, seed)` — spatially coherent
  blobs (e.g. threshold a smoothed random field), NOT iid noise: land
  coherence is what makes region writes realistic. Only land chunks are
  written; ocean chunks stay unwritten (this is load-bearing for ref-count
  metrics).
- `--real-sample PATH`: optional `.npy` of real embeddings; when present,
  tile it (with per-block permutation) instead of random data. Used for T1
  winner/runner-up re-run (compressibility check).

## 3. Icechunk/zarr API ledger (verified signatures + gotchas)

**Groups.** `create_empty_store_from_coords` CANNOT seed sibling groups (it
opens root `mode="w"` — clobbers). For multi-group seeding use raw zarr on a
session store:

```python
session = repo.writable_session("main")
root = zarr.open_group(session.store)              # NOT mode="w" after first group
g = root.require_group("32601")
g.create_array(name, shape=…, chunks=…, shards=…,  # shards: outer shape in ELEMENTS, or None
               dtype=…, fill_value=…, dimension_names=…,
               serializer=…, compressors=None)
```

Data-var arrays: schema only (no chunk writes → zero objects). Coord arrays:
write 1-D numpy in full (`data=`). Per-group attrs: `g.attrs.update({...})`.

**Manifest splitting/preload.**

```python
split = icechunk.ManifestSplittingConfig.from_dict(
    {icechunk.ManifestSplitCondition.AnyArray():
        {icechunk.ManifestSplitDimCondition.DimensionName("time"): 1}})
config.manifest = icechunk.ManifestConfig(splitting=split, preload=…)
# preload: ManifestPreloadConfig(max_total_refs=…, max_arrays_to_scan=…)
# kwargs are Rust-bound (*args/**kwargs in inspect) — confirm names via help()
repo.save_config()   # REQUIRED to persist; workers re-opening read persisted config
```

Split config must be identical across create and every later open (library
precedent: `manifest_split()` contextmanager docstring).

**Fork/merge (cooperative writes).**

```python
fork = session.fork()          # Session.fork(self) -> ForkSession; picklable
# worker: writes via zarr.open_group(fork.store) / to_icechunk; RETURNS its fork
session.merge(*fork_sessions)  # variadic; then session.commit(...)
```

Multiprocessing MUST use `multiprocessing.get_context("spawn")` (or
forkserver) — `fork` start method deadlocks icechunk's tokio runtime.
To deliberately orphan chunks for T6: let a worker write via a fork, then
drop the ForkSession without merging.

**Commit / rebase-retry.**

```python
Session.commit(message, metadata=None, *, rebase_with=None, rebase_tries=1000,
               allow_empty=False) -> str
```

Contention loop (T0/T5): `commit(msg, rebase_with=icechunk.ConflictDetector(),
rebase_tries=N)`; catch `icechunk.ConflictError` / `icechunk.RebaseFailedError`,
count retries, jittered sleep. Same-chunk resolution:
`icechunk.BasicConflictSolver(on_chunk_conflict=icechunk.VersionSelection.UseOurs)`.

**Shift (T3 prepend path).**

```python
Session.shift_array(array_path: str, chunk_offset: Iterable[int]) -> None
# offsets in CHUNK units, one per dim, positive = toward higher indices
```

Prepend pattern: zarr `arr.resize(new_shape)` (grow at end) → `shift_array`
(+1 on time axis, 0 elsewhere) → write index 0 → single commit. Also resize
the 1-D time coord and rewrite it (coords are ordinary arrays — shift applies
per array path; simplest is rewrite the small coord in full). Verify ALL
prior years plus empty (ocean) chunk positions after each iteration —
`reindex_array` (not used, but same machinery) documents a stale-data hazard
at empty positions; prove `shift_array` is clean.

**GC / expiry / rollback (T6).**

```python
Repository.expire_snapshots(older_than: datetime, *, delete_expired_branches=False,
                            delete_expired_tags=False) -> set[str]
Repository.garbage_collect(delete_object_older_than: datetime, *, dry_run=False,
                           max_snapshots_in_memory=50,
                           max_compressed_manifest_mem_bytes=512*2**20,
                           max_concurrent_manifest_fetches=500) -> GCSummary
Repository.reset_branch(branch, snapshot_id, *, from_snapshot_id=None)
Repository.rewrite_manifests(message, *, branch, …)   # repo-wide only
Repository.total_chunks_storage(...) -> int
```

Tags: `repo.create_tag(name, snapshot_id=…)` (verify exact name via `help()`)
protect snapshots from expiry.

**Reads.** Readonly sessions: `repo.readonly_session(branch=…)` or
`(snapshot_id=…)`. Concurrency sweep: `zarr.config.set({"async.concurrency": N})`
+ `RepositoryConfig(max_concurrent_requests=…)`. Cold reads via fresh
subprocess (§2.3). `xr.open_zarr(session.store, consolidated=False)`;
`chunks=None` to skip dask graphs (library precedent in `_open_readonly`).

**Library reuse map.** Use as-is: `_default_repo_config`, `manifest_split()`,
`TIME_ENCODING`, `pcodec_serializer()`, `rollback_commits`,
`create_empty_store_from_coords` (single-group cases ONLY). Raw APIs needed
for: groups, fork/merge, shift_array, GC, tags, sharded arrays.

## 4. Per-script notes (beyond test-plan §4)

- **t0_smoke**: 3 groups × ~16-chunk arrays. Spawn 3 processes via spawn
  context; barrier with `multiprocessing.Barrier` passed to workers; each
  writes its group and commits with rebase-retry. Also assert the fork-method
  guard: attempting `get_context("fork")` repo open is NOT tested (would
  hang) — just hardcode spawn and document.
- **t1_read_bench**: store builder writes each variant via fork/merge batches
  (reuses the T2 writer path — build that function in M1, share it).
  Point set: fixed seed, 1,000 (y,x) pixels uniformly over land chunks.
  `bytes_fetched`: compute analytically from layout (chunks touched ×
  compressed chunk size sampled from S3 object sizes); don't scrape network
  counters. Sharded variant: expect k ranged GETs per k inner chunks
  (icechunk #1316) — latency will show it; note in report.
- **t2_write_bench**: refs/commit sweep {10⁴,10⁵,10⁶,10⁷} — cap at 2×10⁷
  (icechunk #1558 panics ~7×10⁷; do not approach). Year-fill trend: emit
  per-year `commit_wall_s` + `manifest_bytes` delta; the #1600 probe is just
  these series plotted by report.py.
- **t3_prealloc**: phases `seed_full_axis`, `fill_year`, `verify_fill`,
  `prepend_loop`, `verify_prepend`, `conflict_probe`. Seed phase asserts
  zero objects under `chunks/` (via `object_stats`) and wall < 60 s even at
  `--full-height`.
- **t4_group_scale**: checkpoint at 12/60/120 groups; after each commit,
  record `snapshot_bytes` (size of newest object under `snapshots/`) and
  constant-payload single-group `commit_wall_s`. Preload A/B: default config
  vs `ManifestPreloadConfig(max_arrays_to_scan=1000, max_total_refs=10**6)`.
  #1462 probe: time `zarr.open_group(session.store)["32601"]` vs listing the
  root group; enable `RUST_LOG=icechunk=debug` in a subprocess and grep for
  manifest fetches during list operations (known noisy — see memory: RUST_LOG
  filtering is imperfect; fall back to timing deltas if logs are unusable).
- **t5_contention**: serial baseline first (N=1 loop), then N sweep. Same
  spawn+Barrier machinery as t0. Payload: one chunk-aligned region write
  (~64 chunks) per committer per round, 3 rounds.
- **t6_gc_bench**: run ONLY after T2–T5 on the same run-id. Sequence: tag →
  expire (assert tagged survive) → `garbage_collect(dry_run=True)` → real GC
  → emit objects/s → rollback drill. Record `GCSummary` fields verbatim into
  `params`.
- **t7_ramp**: S3-only, separate fresh bucket, plain `boto3` PUTs of 8 MB
  objects at ramped concurrency; count 503 SlowDown via botocore retries
  hooks. No icechunk needed.
- **report.py**: groups rows by (test, metric, variant/params), renders the
  test-plan §5 decision matrix with measured values against thresholds, one
  markdown section per ADR-008 decision. Output:
  `results/<run-id>/report.md`.

## 5. Environment

- Deps beyond the package: `psutil` (and `boto3` for t7/object_stats S3
  paths — already transitively present). Add a `scale-tests` dependency
  group in `pyproject.toml` rather than polluting package deps.
- Bench box: pin `icechunk>=2.1.1` in that group; laptop/local runs may use
  the repo lock (2.0.4) for everything except read benchmarks, whose numbers
  are only valid on ≥2.1.1 (#2158).
- AWS wiring (bucket, profile/role, region us-west-2) is caller-supplied CLI/env
  — placeholders, no account specifics in code.

## 6. Definition of done

1. Full suite green at `--scale tiny --backend local` from a clean checkout
   (single command documented in `scripts/scale_tests/README.md`).
2. Every phase idempotent (kill mid-run, rerun, no duplicate metric rows for
   completed phases; partial-phase rows are overwritten or de-duplicated by
   marker discipline).
3. `report.py` renders the decision matrix with every threshold from
   test-plan §5 present (value, threshold, PASS/FAIL/N-A).
4. `teardown.py` leaves the bucket empty and prints the objects/bytes it
   removed.
5. `scripts/scale_tests/README.md`: how to run tiny/local and bench/s3, cost
   warning, artifact-flow diagram (copy from test plan), and the ADR-008
   update procedure for when results land.
