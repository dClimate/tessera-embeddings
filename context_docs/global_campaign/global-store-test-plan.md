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

## 3. The tests, and where their results went

**All of T0-T8 ran.** The harness they share is `scripts/scale_tests/` and the decisions they
settled are recorded in ADR-008, which is where a reader should go for outcomes — every PENDING
decision there names a test here and every test fed one back.

What stood here was the pre-run specification: the shared harness contract (CLI, metrics schema,
artifact flow) and one subsection per test giving its method, pass/kill thresholds, runtime and
cost budget. The harness is now code and the thresholds are now results, so both are better read
from `scripts/scale_tests/` and ADR-008 respectively. The specification is in git history if the
reasoning behind a threshold is ever needed.

The decision matrix below is the durable half of this document, and §8 is the API ledger the
implementation earned.

## 5. Decision matrix — spent

This mapped each test's evidence to the decision it would settle. **Every one of those decisions is
now FIRM in ADR-008**, each annotated there with the run that confirmed it, so the matrix is a
lookup from evidence nobody needs to gather to conclusions already recorded. Read ADR-008's D1–D9.

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
