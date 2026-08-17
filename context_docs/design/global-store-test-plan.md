# Global store scale-test plan

Companion to ADR `context_docs/decisions/008-global-store-architecture.md`.
Every PENDING decision there names a test here; every test here feeds a
decision there. The tests are **interlocking scripts** sharing one harness,
one metrics schema, and each other's artifacts — designed so results compose
into the decision matrix at the bottom rather than being one-off experiments.

> **Status: EXECUTED, scoping concluded (2026-07-14).** All scripts are
> implemented under `scripts/scale_tests/` (T0–T8) and have run on S3 bench
> (`run1`, `d3`, `d3v2`). Every decision D1–D9 is now FIRM — see ADR-008's
> "Conclusion" for the settled architecture. The one deferred item is GC
> duration at 10⁸-object scale (D7), to be measured against a large repo before
> the production GC cadence is fixed. The plan below is retained as the method of
> record and as a regression harness.

---

## 1. Principles

1. **One harness, one schema.** Every script emits metrics as JSONL rows to a
   shared results prefix; a final report script collates them. No screen-
   scraping of logs to reach conclusions.
2. **Artifacts flow forward.** T1's stores are reused by T2; T3's seeded repo
   is filled by T2's writer; T4's 120-group repo is the arena for T5; the
   accumulated wreckage of T2–T5 is T6's GC corpus. Nothing is built twice.
3. **Idempotent and resumable.** Each script takes `--run-id` and skips
   phases whose success marker already exists in the results prefix. A
   crashed overnight run resumes, not restarts.
4. **Throwaway everything.** Dedicated bucket, dedicated EC2 instance, one
   teardown script that empties both. Nothing points at production paths.
5. **Library code under test, script code disposable.** Scripts exercise
   `tessera_embeddings.storage` where paths exist today (empty seeding,
   region writes, manifest_split) and raw icechunk/zarr APIs where they
   don't yet (groups, fork/merge, shift_array, GC). Findings tell us what to
   build; we do not build library features just to test them.

## 2. Infrastructure

| Item | Spec | Rationale |
|---|---|---|
| EC2 | r7i.4xlarge (16 vCPU / 128 GB), us-west-2 | RAM headroom for commit-RSS tests (T2 sweeps to ~10⁷ refs ≈ 4 GB, plus fork processes); matches store default region |
| Bucket | fresh, us-west-2, same account | throwaway; expect SlowDown ramp on first heavy writes (see T7 note) |
| IAM | s3:* on the test bucket only | placeholder — user wires actual roles |
| Env | uv venv; **icechunk ≥ 2.1.1** (ADR D9), zarr 3.2.1 pinned; package installed from this branch | #2158 fix mandatory before read benchmarks |
| Results | `s3://<bucket>/results/<run-id>/*.jsonl` + local mirror | survives instance death |

Estimated total cost: **~$200–350** (EC2 ~$1.06/h × 40–80 h ≈ $45–85; S3
storage ~0.5–1 TB-month ≈ $12–25; ~5–10 M PUTs ≈ $25–50; GETs and LISTs
minor; buffer for reruns). Teardown checklist at §7.

## 3. Shared harness (`scripts/scale_tests/`)

```
scripts/scale_tests/
├── harness.py      config load, run-id, phase markers, timers, RSS sampler
│                   (psutil, 1 Hz), metrics emitter, S3 object counter (LIST)
├── synth.py        synthetic data: random int8 embeddings (worst-case
│                   compression) + NaN/0 fill for "ocean"; optional real
│                   embedding sample for PCodec realism (--real-sample PATH)
├── variants.py     REGISTRY of store variants (name → chunks, shards|None,
│                   serializer, split config). Single source of truth used by
│                   T1/T2/T3 so variant names are stable join keys in metrics
├── zone_geometry.py the mock-zone spec: a scaled-down UTM zone
│                   (20,000 × 20,000 px default; --full-height for 10⁶ px
│                   metadata-only tests), 9 annual timesteps 2017–2025
├── t0_smoke.py … t7_ramp.py   the tests (below)
├── report.py       collate JSONL → per-decision markdown summary tables
└── teardown.py     empty bucket prefixes, verify $0 residue, stop instance
```

Metrics row schema (every emission):

```json
{"run_id": "...", "test": "t2", "variant": "c256_full_band", "phase": "fill_year",
 "metric": "commit_wall_s", "value": 41.2, "unit": "s",
 "params": {"year": 2023, "refs": 190000, "batch": "zone-year"},
 "icechunk": "2.1.1", "zarr": "3.2.1", "git_sha": "...", "ts": "..."}
```

Core variant registry (extend, don't fork):

| name | chunks (t,y,x,band) | shards | notes |
|---|---|---|---|
| `c500_band4` | (1,500,500,4) | — | current layout, baseline |
| `c500_full` | (1,500,500,128) | — | 32 MB chunks |
| `c384_full` | (1,384,384,128) | — | ~19 MB |
| `c256_full` | (1,256,256,128) | — | 8.4 MB, AlphaEarth-like — expected winner |
| `c256_sharded` | (1,256,256,128) | (1,2048,2048,128) | 64 chunks/shard, ~0.5 GB objects |

All variants: int8 + PCodec for `embeddings`, float32+NaN `scales`, dims
(time, northing, easting, band).

```
artifact flow                       decisions fed
─────────────                       ─────────────
T0 smoke ──────────────┐
T1 read bench ──stores─┼──► T2 write bench ──► D2 chunk shape, D3 sharding
T3 prealloc seed ──────┘        │               D4 split config, D6 params
T4 120-group repo ──────────────┼──► T5 contention ──► D5 one-repo go/no-go
        └───────── all leftovers ┴──► T6 GC ──► D7 cadence
T7 ramp (optional, standalone) ─────────────► campaign warm-up plan
```

## 4. Test specifications

### T0 — Smoke + cross-group conflict probe (~1 h, <$1)

**Feeds:** harness validation; D6 (converts the cross-group-rebase inference
to fact before anything expensive runs).

1. Create a repo with 3 tiny groups; write/commit from 3 processes
   simultaneously, one group each, plain `ConflictDetector` rebase-retry.
   **Expect:** all commit within ≤3 retries, zero unresolvable conflicts.
2. Two sessions write disjoint regions of the *same* array → both commit;
   two sessions write the *same* chunk → `ChunkDoubleUpdate` raised and
   resolved with `BasicConflictSolver(UseOurs)`.
3. Verify spawn-vs-fork multiprocessing guard, metrics emission, phase
   markers, teardown of prefixes.

**Kill criterion:** if (1) shows cross-group commits do NOT auto-rebase
cleanly, D5/D6 are re-planned immediately (commit queue or repo-per-zone) —
before T4/T5 are built.

### T1 — Retrieval benchmark across chunk variants (~1 day incl. writes, ~$40)

**Feeds:** D2 (spatial size), D3 (sharding), read-side defaults
(`async.concurrency`).

Setup: one mock zone (20k×20k px ≈ 51 GB int8/timestep, 1 timestep) written
per variant (5 variants ≈ 260 GB). Populate ~70% of chunks (simulate land),
leave the rest unwritten (fill).

Procedure, per variant × `async.concurrency` ∈ {10, 64, 128}:

| workload | shape | metrics |
|---|---|---|
| point-vector | 1,000 random pixels, all 128 bands | p50/p95 latency cold+warm, bytes fetched (read amplification) |
| patch | 100×100 px, all bands | same |
| tile | 1,000×1,000 px, all bands | wall, effective MB/s |
| band-subset | 10k×10k px, 8 of 128 bands | wall, bytes fetched (split-band variant's one advantage) |
| full scan | whole store | aggregate MB/s |
| open | fresh-process `xr.open_zarr` via readonly session | wall |

Cold = fresh process + fresh Repository; warm = repeat in-process. Record
per-workload GET counts analytically from layout; spot-check with S3 request
metrics if enabled.

**Scoring (D2/D3):** rank variants by point-vector p95 (weight 3), patch p95
(2), tile MB/s (2), full-scan MB/s (1), refs-per-year at global scale (2,
from the ADR table). `c256_sharded` must beat `c256_full` on the weighted
score to justify sharding's write-path complexity (D3), noting its k-GETs-
per-k-inner-chunks read behavior until zarr ships PR #3004 — **re-run this
test's sharded rows when that release lands.**

### T2 — Write path, commit scaling, manifest behavior (~1–2 days, ~$60)

**Feeds:** D4 (split config), D6 (batch size, refs/commit ceiling), #1600
probe. **Consumes:** T3's pre-allocated 9-year store (run T3 setup first).

1. **Fork/merge batch sweep** (winning variant from T1): fill one zone-year
   via cooperative fork/merge with total refs/commit ∈ {10⁴, 10⁵, 10⁶, 10⁷}.
   Metrics: commit wall, peak coordinator RSS (validate ~400 B/ref; #1558
   panic threshold is ~7×10⁷ — do NOT exceed 2×10⁷), merge wall vs fork
   count {4, 16, 64 workers}.
2. **Sequential year fills, 2025→2017** on the T3 store, manifest split =
   time@1 (one manifest per year): per-year commit wall and bytes of
   manifest written. **Expect flat; linear growth across years reproduces
   icechunk #1600 → escalate, retest with spatial splits, consider
   repo-per-zone smaller arrays (D5 input).**
3. **Split-config A/B:** repeat (2) with (a) no splitting, (b) time@1,
   (c) time@1 + spatial 64×64-chunk windows. Metrics: commit wall, manifest
   count, snapshot file size growth per commit, then patch-write cost (small
   region overwrite into a filled year) per config.
4. **Sharded write amplification** (if `c256_sharded` survived T1): rewrite a
   region covering ¼ of a shard; measure bytes PUT vs. logical bytes.

**Output:** chosen refs/commit ceiling + fork count for D6; chosen split
config for D4 (target: patch commit rewrites ≤1 year-manifest; snapshot
growth ≤ ~a few MB per 10³ manifests).

### T3 — Pre-allocate vs. prepend A/B (~½ day, ~$10)

**Feeds:** D1 (confirmation), escape-hatch characterization. **Produces:**
the pre-allocated store T2 fills.

1. **Pre-allocate path:** seed a mock zone with the full 9-year axis +
   coords (metadata-only; assert zero chunk objects, seconds-scale wall
   regardless of extent). Region-insert one year; verify written year reads
   back exactly, unwritten years read as fill with NaN `scales`, and
   `years_complete` attr lands in the same commit.
2. **Prepend path:** create the same store with only 2025; then per older
   year: resize along time → `shift_array` (+1 chunk on time) → region-write
   the new front → commit. Metrics per iteration: shift+commit wall,
   manifest bytes rewritten (should be metadata-only — verify no chunk-data
   PUTs), correctness of ALL previously-written years after each shift
   (including empty/ocean chunk positions — the `reindex_array` stale-data
   class of bug).
3. **Conflict probe:** start a chunk-writer session on the array, land a
   shift commit from another session, then attempt the writer's commit.
   **Expect** unresolvable `ChunksUpdatedInUpdatedArray`-family conflict —
   confirming why prepends are banned during concurrent fills (D1/D6).

**Pass criterion for D1:** path (1) green. Path (2) results are recorded as
the documented cost/risk of the pre-2017 escape hatch, not a gate.

### T4 — 120-group repo metadata scale (~½ day, ~$10)

**Feeds:** D5 (snapshot/open-time arm of the kill criteria), preload tuning.

1. Seed one repo with **120 zone groups** (metadata-only, full-height zone
   geometry — this is free by D1) with per-group CRS attrs; populate 6
   representative zones with 1–2 years each (reuse T2 writer).
2. Measure vs. group count (checkpoints at 12/60/120 groups): snapshot file
   size (S3 object size), commit wall for a constant-size write to ONE group
   (must stay ~flat — commits re-serialize the whole snapshot), repo open
   wall.
3. Read-side: single-group `open_zarr` wall from fresh process;
   `open_datatree` over the full repo; each with preload default vs.
   `max_arrays_to_scan ≥ 700` + raised `max_total_refs`; instrument whether
   listing paths trigger per-array manifest GETs (icechunk #1462) via debug
   logging/timing.

**Kill thresholds (D5):** single-group open > ~5 s warm-region, or per-commit
snapshot cost growing super-linearly with group count, or #1462 confirmed
with material impact → weigh repo-per-zone.

### T5 — Commit contention (~½ day, ~$15)

**Feeds:** D5 (contention arm), D6 (pacing). **Consumes:** T4's repo.

For N ∈ {2, 8, 16, 32, 64, 120}: N processes each region-write a small fixed
payload to a **distinct group** and commit simultaneously (barrier start),
rebase-retry loop with jitter. Metrics: per-committer retry count, per-commit
wall p50/p95, total wall vs. N× the serial baseline, any non-auto-resolvable
conflict (expect zero).

**Kill threshold (D5):** at N=16, total wall > 2× serial equivalent or retry
storms (p95 retries > ~5) → per-zone repos, or a commit-queue service in the
orchestration layer (which weakens the case for one repo but doesn't kill
it — record both numbers).

### T6 — GC, expiry, rollback (~½ day, ~$10 + LIST costs)

**Feeds:** D7 (cadence + runbook numbers). **Consumes:** everything T2–T5
left behind (a realistic mess: overwritten years, retried commits, abandoned
fork sessions — deliberately abandon a few in T2).

1. Tag zone-year snapshots; `expire_snapshots(older_than=…)` sparing tags;
   verify tagged survive.
2. `garbage_collect(dry_run=True)` then real: wall time vs. total object
   count (LIST-bound — record objects/s to extrapolate to 10⁸–10⁹ objects),
   bytes reclaimed, orphaned-fork chunks reclaimed.
3. Rollback drill: `reset_branch` a landed zone-year away, verify reads at
   old snapshot, re-write the year, confirm the orphaned snapshot's objects
   are reclaimed by the next expire+GC cycle.

**Output:** projected GC wall at campaign scale + the do/don't-run-during-
backfill rule with concrete cutoff arithmetic (D7).

### T7 — Fresh-bucket ramp (optional, ~2 h, ~$10)

**Feeds:** campaign warm-up plan only; run LAST (it deliberately provokes
SlowDown) or against a second throwaway bucket. Ramp aggregate PUT
concurrency through {50, 100, 200, 400} against a cold bucket; record 503
rates and time-to-stabilize. Confirms the existing
`TARGET_AGGREGATE_S3_CONCURRENCY=100` discipline or licenses raising it.

**Sequencing note:** because a cold bucket throttles early heavy writes, T1's
store-building phase doubles as the main bucket's warm-up — build all T1
variants before trusting any latency numbers, or discard first-hour write
timings.

## 5. Decision matrix

| Evidence | Threshold | Decision outcome |
|---|---|---|
| T0.1 cross-group rebase | clean | D6 confirmed; else redesign before T4/T5 |
| T1 weighted score | best variant | D2 spatial size locked |
| T1 sharded vs unsharded + T2.4 | sharded wins reads AND write amp acceptable | D3 adopt sharding; else defer with re-test trigger (zarr > 3.2.1) |
| T2.2 year-fill trend | flat | D4 time@1 split locked; linear → escalate #1600, spatial splits, D5 input |
| T2.1 RSS + commit wall | knee point | D6 refs/commit ceiling + fork count |
| T4 snapshot/open walls | under thresholds | D5 one-repo arm 1 passes |
| T5 N=16 wall + retries | ≤2× serial, no storms | D5 one-repo arm 2 passes; both arms pass → one repo GO |
| T3.1 | green | D1 confirmed |
| T6 GC objects/s | extrapolation | D7 cadence numbers |

## 6. Risks and caveats

- **Synthetic data compressibility:** random int8 is worst-case for PCodec;
  chunk-size conclusions drawn on compressed GET sizes shift with real data.
  Mitigate with `--real-sample` on at least the T1 winner and runner-up.
- **Mock-zone extrapolation:** 20k×20k tests ≈ 1/170th of a full zone's area.
  Ref-count-driven conclusions (T2, T4) scale linearly and safely; latency
  conclusions (T1) are size-independent per-request; only whole-zone wall-
  clock projections carry real uncertainty.
- **Version sensitivity:** results are stamped with icechunk/zarr versions;
  #1600, #1558, #1316/#3004 are all live upstream — re-run the affected test,
  not the whole program, when they move.
- **One instance, in-region:** all latencies are best-case (same-region EC2).
  Partner-side consumer latency from other regions/internet is out of scope
  here and should be flagged to them separately.

## 7. Teardown checklist

1. `teardown.py --run-id …`: delete `stores/`, `repos/`, `results/` prefixes
   (results mirrored locally first), verify bucket empty via LIST.
2. Delete the bucket; stop/terminate the instance; confirm no EBS orphans.
3. Archive collated `report.py` output into this repo alongside ADR 008
   status updates (FIRM/superseded per decision).

---

## 8. Icechunk / zarr API ledger

**Absorbed 2026-08-17 from `global-store-test-impl-spec.md`, which this document replaces.**
That spec was a handoff for building `scripts/scale_tests/`; the suite shipped, so its build
order, milestones and definition-of-done describe work that is now code and are recoverable
from git history if ever wanted. What survives it is this ledger — signatures and gotchas
that cost real time to establish and that no reader can derive from the call sites.

Every signature below was verified against **icechunk 2.0.4 / zarr 3.2.1 on 2026-07-13**. The
bench environment runs **icechunk ≥ 2.1.1** (ADR-008 D9). Re-verify with `inspect.signature`
before relying on one; several of these are Rust-bound and do not introspect usefully.

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
