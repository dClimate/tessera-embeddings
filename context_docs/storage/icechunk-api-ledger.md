# Icechunk and Zarr API ledger

**Signatures and gotchas that cost real time to establish, and that no reader can derive from the
call sites.** This is the durable half of the T0–T8 global-store scale-test programme: the tests all
ran, and the decisions they settled are recorded in
[ADR-008](../decisions/008-global-store-architecture.md), where every D1–D9 is now FIRM and annotated
with the run that confirmed it. The harness itself is code, under `scripts/scale_tests/`.

> **What was cut, and where it went.** The pre-run specification — the shared harness contract, the
> metrics schema, one subsection per test giving its method and its pass/kill thresholds, and the
> decision matrix mapping evidence to decisions — described work that is now code and conclusions
> that are now ADR-008. It is in git history if the reasoning behind a threshold is ever wanted.
> **The infrastructure spec and the cost basis were NOT cut**: an operator budgeting a real bench
> run needs them, so they moved to `scripts/scale_tests/README.md` beside the estimate they price,
> along with the teardown checklist. The build-order handoff spec (`global-store-test-impl-spec.md`) was absorbed into this
> ledger on 2026-08-17 and deleted.

Every signature below was verified against **icechunk 2.0.4 / zarr 3.2.1 on 2026-07-13**; the bench
environment ran **icechunk ≥ 2.1.1** (ADR-008 D9), and production runs 2.1.1. **Re-verify with
`inspect.signature` before relying on one** — several of these are Rust-bound and do not introspect
usefully.

---

## Groups

`create_empty_store_from_coords` **CANNOT seed sibling groups** — it opens root `mode="w"`, which
clobbers. For multi-group seeding use raw zarr on a session store:

```python
session = repo.writable_session("main")
root = zarr.open_group(session.store)              # NOT mode="w" after the first group
g = root.require_group("32601")
g.create_array(name, shape=…, chunks=…, shards=…,  # shards: outer shape in ELEMENTS, or None
               dtype=…, fill_value=…, dimension_names=…,
               serializer=…, compressors=None)
```

Data-var arrays: schema only, no chunk writes, so zero objects. Coord arrays: write the 1-D numpy
array in full (`data=`). Per-group attrs: `g.attrs.update({...})`.

## Manifest splitting and preload

```python
split = icechunk.ManifestSplittingConfig.from_dict(
    {icechunk.ManifestSplitCondition.AnyArray():
        {icechunk.ManifestSplitDimCondition.DimensionName("time"): 1}})
config.manifest = icechunk.ManifestConfig(splitting=split, preload=…)
# preload: ManifestPreloadConfig(max_total_refs=…, max_arrays_to_scan=…)
# kwargs are Rust-bound (*args/**kwargs in inspect) — confirm names via help()
repo.save_config()   # REQUIRED to persist; workers re-opening read persisted config
```

**The split config must be identical across create and every later open** (library precedent: the
`manifest_split()` contextmanager docstring). Which axis to split on is a workload question, and the
measurement is in [`../ingest/ingest-performance.md`](../ingest/ingest-performance.md) §12.7: **split
the axis along which a single commit is NARROW.**

## Fork and merge (cooperative writes)

```python
fork = session.fork()          # Session.fork(self) -> ForkSession; picklable
# worker: writes via zarr.open_group(fork.store) / to_icechunk; RETURNS its fork
session.merge(*fork_sessions)  # variadic; then session.commit(...)
```

**Multiprocessing MUST use `multiprocessing.get_context("spawn")`** (or forkserver) — the `fork`
start method deadlocks icechunk's tokio runtime. To deliberately orphan chunks, let a worker write
via a fork and drop the `ForkSession` without merging.

A fork can be merged into a session it was **not** created from, and into one that has since rebased.
That was verified on a real repository across five arms, and it is what the assembly recovery path
depends on — see [`writing-to-the-global-store.md`](writing-to-the-global-store.md) §3.

## Commit and rebase-retry

```python
Session.commit(message, metadata=None, *, rebase_with=None, rebase_tries=1000,
               allow_empty=False) -> str
```

Contention loop: `commit(msg, rebase_with=icechunk.ConflictDetector(), rebase_tries=N)`; catch
`icechunk.ConflictError` / `icechunk.RebaseFailedError`, count retries, sleep with jitter. Same-chunk
resolution is `icechunk.BasicConflictSolver(on_chunk_conflict=icechunk.VersionSelection.UseOurs)` —
**and note that swapping the detector for the solver resolves a genuine collision silently**, which
is the failure a conflict guard exists to prevent.

`Session.rebase` takes **no target snapshot**: it advances to whatever the tip is when it runs. Any
design that checks a range and then rebases has a race between the two, and that race is not
theoretical (`writing-to-the-global-store.md` §3).

## Shift (the prepend path)

```python
Session.shift_array(array_path: str, chunk_offset: Iterable[int]) -> None
# offsets in CHUNK units, one per dim, positive = toward higher indices
```

Prepend pattern: zarr `arr.resize(new_shape)` to grow at the end → `shift_array` (+1 on the time axis,
0 elsewhere) → write index 0 → single commit. Also resize the 1-D time coord and rewrite it — coords
are ordinary arrays, so shift applies per array path and rewriting the small coord in full is
simplest. **Verify ALL prior years plus empty (ocean) chunk positions after each iteration:**
`reindex_array` (not used, but the same machinery) documents a stale-data hazard at empty positions.

## Garbage collection, expiry, rollback

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

Tags (`repo.create_tag(name, snapshot_id=…)`) protect snapshots from expiry. **Tags are write-once
forever** — a corrected cell needs a fresh tag name, which is why the campaign moved idempotence off
tags entirely ([`../inference/minimum-optical-depth.md`](../inference/minimum-optical-depth.md) §10).

**GC deletes require delete permission on the prefix.** Without it, superseded manifests become
permanent — which is why the published-store access request asks for `s3:DeleteObject` explicitly.

## Reads

Readonly sessions: `repo.readonly_session(branch=…)` or `(snapshot_id=…)`. Concurrency sweep:
`zarr.config.set({"async.concurrency": N})` plus `RepositoryConfig(max_concurrent_requests=…)`. Cold
reads need a fresh subprocess. `xr.open_zarr(session.store, consolidated=False)`, with `chunks=None`
to skip dask graphs (library precedent in `_open_readonly`).

## Library reuse map

**Use as-is:** `_default_repo_config`, `manifest_split()`, `TIME_ENCODING`, `pcodec_serializer()`,
`rollback_commits`, and `create_empty_store_from_coords` for **single-group cases only**.

**Raw APIs are still needed for:** sibling groups, fork/merge, `shift_array`, GC, tags, and sharded
arrays.

---

## Caveats the scale tests earned, which still apply

- **Synthetic data compressibility.** Random int8 is worst-case for PCodec, so chunk-size conclusions
  drawn on compressed GET sizes shift with real data.
- **Mock-zone extrapolation.** A 20k × 20k test is about 1/170th of a full zone's area. Ref-count-driven
  conclusions scale linearly and safely; latency conclusions are size-independent per request; only
  whole-zone wall-clock projections carry real uncertainty.
- **Version sensitivity.** Results are stamped with icechunk and zarr versions. When an upstream issue
  moves, re-run the affected test, not the whole programme.
- **One instance, in-region.** All measured latencies are best-case. Consumer latency from other
  regions or over the internet is out of scope and should be flagged to consumers separately.

**The one deferred item from the programme** is GC duration at 10⁸-object scale (ADR-008 D7), to be
measured against a large repo before the production GC cadence is fixed.
