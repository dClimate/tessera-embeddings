# 008 — Global embeddings store: architecture decisions

**Status:** **Accepted — scoping concluded (2026-07-14).** All decisions
D1–D9 are FIRM, each backed by S3-bench evidence from runs `run1`, `d3`, and
`d3v2` (test program in `context_docs/design/global-store-test-plan.md`). One
operational measurement is explicitly *deferred, not blocking* — GC duration at
10⁸-object scale (D7) — to be taken against a large repo before/while the
campaign runs. The next work is **implementation** of the group-aware, sharded,
land-masked write path, not further scoping. Supersede individual decisions with
follow-up ADRs only if production reveals something these runs didn't.

**Evidence runs (all 2026-07-14, S3 bench, icechunk 2.1.1,
`arbol-tessera-embeddings-dev`):**
- `run1` — full T0–T7 incl. the 5-variant T1 sweep → D1, D2, D4 FIRM;
  D5/D6 commit-concurrency constraint; D7 mechanism validated.
- `d3` / `d3v2` — t8 sharding experiments (E1–E4), the second with the
  land-masked production writer → **D3 FIRM: adopt `c256_sharded`, shard
  `scales` too.** Reads are leaner *and* faster (wire 1.23 vs 8.69 MB/pt,
  reproducible; scattered p95 131 vs 217 ms); the masked writer is **0.46×** the
  unsharded build (the run-1 2.8× was unaligned RMW) and writes lean shards
  (~3.5 GB, ocean elided); the embeddings array consolidates 403→9 objects.

See the "Conclusion" section at the bottom for the final architecture in one
place, and "Run evidence" for the full numbers.

## Context

We will generate 10 m TESSERA embeddings for the entire globe, one timestep
per year, 2017–2025 (2025 first, then backwards), publishing through a
partner. The world is subdivided into **120 UTM zone datasets** (60 zones × 2
hemispheres, each with its own CRS: EPSG:326xx north / EPSG:327xx south),
each intended to be a **Zarr group inside one Icechunk repo**. Later years
(2026+) will be appended.

```
icechunk repo (S3)
└── main branch
    ├── root attrs: global conventions, dataset version
    ├── 32601/   ← group: UTM 1N,  EPSG:32601
    │   ├── embeddings   (time, northing, easting, band)  int8
    │   ├── scales       (time, northing, easting)        float32
    │   ├── *_obs_count  (time, northing, easting)        uint16
    │   └── time, northing, easting, band   (1-D coords; per-zone CRS attrs)
    ├── 32701/   ← group: UTM 1S,  EPSG:32701
    │   └── …
    └── … ×120 groups
```

**Scale envelope** (land ≈ 149 M km² → ~1.49×10¹² px/yr at 10 m):

| Chunk shape (t,y,x,band) | int8 size | Refs/yr (land) | Refs, 9 yrs | GETs per pixel-vector read |
|---|---|---|---|---|
| (1, 500, 500, 4) — current | 1 MB | ~190 M | ~1.7 B | 32 |
| (1, 500, 500, 128) | 32 MB | ~6 M | ~54 M | 1 × 32 MB |
| (1, 256, 256, 128) | 8.4 MB | ~23 M | ~205 M | 1 × 8.4 MB |
| (1, 256, 256, 128) in 2048² shards | 8.4 MB / 540 MB object | ~0.36 M | ~3.2 M | 1 |

Volume: ~190 TB/yr raw embeddings (int8×128) + ~6 TB/yr `scales`; ~1.7 PB for
nine years. A zone array is roughly (9, ~1×10⁶, ~6.7×10⁴, 128); an average
zone-year holds ~2×10⁵ chunk refs at 256 px chunks (land-rich zones a few
×10⁶).

**What exists today:** single-root-group stores only (every write path opens
root with `mode="w"`); metadata-only empty-store seeding
(`create_empty_store*`, zero chunk objects, cost independent of extent);
opt-in manifest splitting (`manifest_split()` contextmanager); atomic
create/append/region-write paths; storage timeouts+retries; `rollback_commits`.
**Gaps:** no multi-group support, no GC/expiry, no conflict/rebase handling,
`BucketPaths` assumes one `.zarr` per (roi, kind).

## Decision register

### D1 — Pre-allocate the full 2017–2025 time axis; never prepend (FIRM — confirmed run 1 T3)

Seed every zone group with all nine annual timesteps and fully-written
coordinate arrays at creation. Fill 2025 first, then backfill older years as
**region-inserts** at existing time indices. Future years (2026+) are ordinary
end-appends.

Why: unwritten chunks cost zero storage and zero manifest refs (verified in
`tests/unit/test_empty_store.py`: `nchunks_initialized == 0`), so
pre-allocation is free. Physical prepends are possible since icechunk 2.0
(`shift_array` / `reindex_array`, metadata-only chunk remapping) but the
feature is ~3 months old, `reindex_array` has a documented stale-data gotcha
on empty (ocean) chunk positions, and any array-metadata update conflicts
unresolvably with concurrent chunk writers (`ChunksUpdatedInUpdatedArray`).
Pre-allocation removes every resize from the campaign. `shift_array` remains
the escape hatch if pre-2017 years are ever wanted.

Semantics that ride along: `scales` fill = NaN remains the "never written"
sentinel (int8 `embeddings` fill 0 alone is ambiguous); reads of unfilled
years succeed silently, so each group maintains a `years_complete` attr
updated **in the same commit** as the year's data.

### D2 — Chunk shape: full band dimension, **spatial 256 px** (FIRM — confirmed run 1 T1 sweep)

Never split the 128-band axis. Keep dim order (time, northing, easting,
band) — band varies fastest, so one pixel's full vector is contiguous inside
its chunk. Spatial size chosen by T1 among {256, 384, 500}.

**Codec correction (found during test build):** int8 `embeddings` use the Zarr
v3 default bytes codec + default (zstd) compressor — **not PCodec**. PCodec is
float-only; `numcodecs` raises "Unsupported data type: dtype('int8')". This is
already how `inference.assembly` writes: PCodec is applied only to the float
arrays (`scales`, `embedding_std`), never to the int8 embeddings. Earlier drafts
of this ADR said "int8 + PCodec", which is impossible; corrected here.

Why: the current `(1, 500, 500, 4)` layout is untenable globally — a 32×
ref multiplier (~1.7 B refs) and 32 GETs per pixel-vector read (the
`BAND_CHUNK_DIVISOR` comment already flags it as vestigial). Earthmover's
measured Icechunk-on-S3 optimum is 3–15 MB/chunk; Google's AlphaEarth
embeddings ship as (1, 64, 256, 256) int8 = 4 MiB inner chunks with the full
channel dim. `(1, 256, 256, 128)` = 8.4 MB sits in the sweet spot and yields
~205 M refs total — manageable with manifest splitting alone. Published
numbers are from synthetic benchmarks; quantized-embedding compressibility
shifts compressed GET sizes, hence T1.

**Run 1 result (spatial size settled).** Cold point-vector p95 by variant:
`c256_full` 193 ms, `c256_sharded` 199 ms, `c500_band4` 219 ms, `c384_full`
334 ms, `c500_full` 339 ms. **256 px wins**: bigger spatial chunks fetch more
bytes per pixel and read slower (384/500 ≈ 334–339 ms). The current
`c500_band4` band-split is beaten on every axis — 219 ms p95 *and* 31.8 MB /
**32 GETs** per point-read (8×/32× the amplification of `c256_full`) *and* ~1.7 B
refs. Going below 256 px isn't worth it (below the 3–15 MB optimum, 4× more
refs). So: 256 px spatial, full 128-band dim, int8+zstd. FIRM.

### D3 — Sharding: **ADOPT `c256_sharded`; shard `scales` too** (FIRM — confirmed by t8 runs d3 + d3v2; production writer is shard-aligned + land-masked)

Zarr v3 sharding works in icechunk (fixed v1.0.3, caveat-free since 2.0.0,
requires zarr-python ≥ 3.1.2). Inner chunks per D2 (256 px), shards ~2048² px
(64 inner chunks, ~0.5 GB objects), shard-aligned writes strongly preferred.

**Run 1 upended the earlier "optional, adopt only if reads win" stance — the
reads do *not* lose:**

- **Point p95:** `c256_sharded` 199 ms vs `c256_full` 193 ms — a 6 ms *tie*
  (the report's bare "full wins" verdict is p95-only and misleading).
- **Point p50:** 29 ms vs 117 ms — sharding **4× faster** in the common case
  (large shards keep nearby points warm; the win is locality-dependent).
- **Aggregate read phase:** 563 s vs 1323 s — sharded ran the full read suite in
  under half the time. Bulk throughput was marginally *higher* (2014 vs
  1897 MB/s), so the no-coalescing worry (#1316) did **not** bite at these sizes
  (a point read needs one inner chunk = one ranged GET; the latency proves
  partial reads, not whole-shard fetches — despite what the analytic
  `bytes_fetched` metric reports).
- **Object count:** ~64× fewer refs (~3.2 M vs ~205 M over 9 yr) — a large win
  for manifest size, GC feasibility, listing, and reader request cost.

Run 1's one strike — **build 2.8× slower** (420 s vs 152 s) — looked like a
dealbreaker but was an artifact of writing 256-px chunks that dribble into
2048² shards (read-modify-write churn).

**Runs `d3` + `d3v2` (t8) settled it — all decision gates pass, and the `d3v2`
run confirmed them with the production (land-masked) writer:**

- **E1 (wire bytes, the gate):** sharded fetched **1.23 MB/point** off the NIC
  vs full's **8.69 MB**, faster (p95 164 vs 267 ms) — and **reproducible** (1.23
  in both `d3` and `d3v2`). Sharding does lean *partial* reads; the whole-shard-
  fetch fear is dead (the run-1 16.5 MB analytic figure was an artifact).
- **E2 (write penalty, the crux):** the shard-aligned + land-**masked** writer
  builds in **0.46×** the unsharded time (~8.3 s vs ~18 s) and writes **lean
  shards ~3.5 GB** (vs the dense-aligned mode's 12.6 GB — ocean inner chunks are
  elided as designed). The run-1 2.8× penalty was pure unaligned RMW; aligned
  masked writes are *faster* than unsharded (9 big shard PUTs beat 400 small).
- **E3 (scattered reads):** sharded scattered p95 **131 ms vs full 217 ms** —
  better even under a cache-hostile pattern (rule only asked for within 1.2×).
- **E4 (object count, post-GC):** the embeddings array consolidates **403 → 9
  objects** with the masked writer (no churn); chunkwise sharding churns
  (808 → 414 after GC reclaims the RMW debris). Confirms the object-count win is
  real and requires shard-aligned writes.

**Decision: ADOPT `c256_sharded` — and shard `scales` the same way.** All gates
met (write 0.46× ≤ 1.5×; scattered p95 better than full). Two writer facts fixed
by the evidence:

1. **The production writer must be shard-aligned + land-masked** (real data into
   land inner chunks, ocean left at fill so the codec elides it → one lean shard
   object per shard, no RMW, no dense-nodata). Confirmed by `d3v2` (0.46× build,
   ~3.5 GB shards). The assembly writer must emit whole shards, never dribble
   inner chunks.
2. **`scales` must also be sharded.** `d3v2` E4 showed the store's object count
   capped at ~414 by the *unsharded* `scales` array even after embeddings
   consolidated to 9 objects — `scales` is `(time, y, x)` with one chunk per
   spatial tile, i.e. the *same* object count as embeddings, so leaving it
   unsharded forfeits half the object-count win (the very thing that justifies
   sharding). Shard `scales` in the same 2048² shards. This follows from E1–E4 +
   object-count arithmetic; it wasn't separately benchmarked because the codec
   and mechanics are identical to the (proven) embeddings sharding.

D2 (256 px, full band) holds regardless; sharding is the wrapper on both arrays.

### D4 — Manifest splitting: time-primary, coarse (FIRM — confirmed run 1 T2)

Split every array's manifest on **time at 1 chunk per window** (= one
manifest per year per zone array), adding spatial splits only for zones whose
year-manifest exceeds ~2–4 M refs. Target ~10³ manifests repo-wide, not 10⁵+.

Why: each commit rewrites only the touched year's manifest(s), but every
commit re-serializes the **repo-global snapshot file listing all manifests**,
so over-splitting taxes every future commit across all 120 groups. The
existing assembly default (`{"northing": 32, "easting": 32, "time": 1}`) is
tuned for single-ROI stores and would explode at zone scale. Known risk to
probe in T2: icechunk #1600 (open) observed append times growing linearly
with cumulative refs *despite* splitting. Note `rewrite_manifests()` is
repo-wide only — no per-group targeting.

### D5 — One repo, 120 groups — adopted *with a commit-concurrency cap* (FIRM; commit constraint FIRM from run 1)

Adopt the single-repo/120-group layout the partner expects, subject to:

- T5: ≥16 concurrent zone-committers complete with clean auto-rebases and
  ≤2× serial wall-clock;
- T4: snapshot-file growth and single-group open times stay acceptable at
  120 groups (incl. manifest-preload tuning: `max_arrays_to_scan` default 50
  < our node count — must be raised; probe icechunk #1462 `list_prefix`
  manifest loads);
- T2: no unbounded commit-time growth (#1600).

Fallback: one repo per zone + a thin catalog. UTM zones never need cross-zone
atomicity, so the fallback loses nothing semantically; official guidance is
to scope repos to arrays needing consistent transactional updates. This is a
partner conversation to have **before** implementation, with test evidence.

**Run 1 result (decisive on the commit path).** The *structure* is fine: 120
groups → a 38 KB snapshot re-serialized per commit, single-group opens ~0.15 s,
per-group commit flat (~0.25 s) regardless of group count. But **uncoordinated
concurrent commits storm**: auto-rebase retries grow ≈ linearly with the number
of simultaneous committers (O(N²) aggregate), so the N=16 kill criterion above
is *breached* under the pathological all-at-once pattern (median 7.5 retries,
commit ~10× the uncontended 0.2 s). This does **not** kill one-repo — it kills
letting all zones commit to `main` at once. **Constraint (now FIRM): a
commit-concurrency cap of ~4–8 simultaneous committers** (a queue/semaphore in
the orchestration layer), or per-zone repos. Reader-side: whole-repo
`open_datatree` over 120 groups took ~31 s (vs 0.15 s per group) — readers must
open a single zone group, never the datatree (icechunk #1462). With D2/D3
settled by the T1/d3/d3v2 sweeps, the one-repo layout is the decision — capped
committers, single-group readers; the naive-concurrency question is settled.

### D6 — Commit strategy: cooperative fork/merge, TWO commits per zone-year (FIRM shape; pacing cap FIRM from run 1)

Within a zone-year: `session.fork()` → pickled ForkSessions to workers →
`merge(*sessions)` → **one commit for the shards**, then a **second, small commit for
that year's `years_complete`/`runs` attrs** (multiprocessing start method must be
spawn/forkserver — fork deadlocks icechunk's runtime).

> **Amended 2026-07-30: the attrs were split out of the shard commit.** They were
> bundled in, which made two years of the SAME zone uncommittable concurrently —
> `ConflictDetector` treats group attributes as an opaque value and cannot merge them,
> so the loser raised `RebaseFailedError` and lost its entire assembly. That is what
> made the campaign's years serial. Chunk data was never the obstacle: every chunk and
> shard is 1 in the time dimension, so different years of one zone are strictly
> disjoint. `shard_writer.commit_year_attrs` now owns both attrs, in its own commit,
> with a bounded retry — and the retry is CORRECT rather than hopeful, because each
> writer only ever inserts its own year's key (`years_complete` is a set union, `runs`
> a per-year dict insert), so re-reading the winner's value and re-applying yields what
> both writers intended in either order. Consequence for this decision's budget: about
> **2,200 commits** for the campaign rather than 1,100, still far below icechunk's
> "tens of thousands" target. Consequence for a crash: dying between the two leaves a
> year holding DATA that nothing marks complete, which the work list reads as pending,
> so a retry rewrites the same shards and re-marks. Across zones: commits
to distinct groups with a bounded rebase-retry loop (`ConflictDetector`;
cross-group commits auto-rebase cleanly — confirmed in run 1 T0), **paced
behind a concurrency cap** so the branch-tip CAS isn't slammed.

Why: commit-time memory is ~400 B/ref and single commits ≥ ~7×10⁷ refs panic
(icechunk #1558 open) — zone-year commits (~10⁵–10⁶ refs) sit far below that.
Icechunk v2 targets "tens of thousands" of commits per repo; one commit per
zone-year ≈ 1,100 total (vs. yield-embeddings' per-slab pattern, ~400
commits/timestep, which would blow past it). Never mix array-metadata updates
with concurrent chunk writes (see D1).

**Run 1 made the pacing non-optional (see D5):** retries scale ≈ N−1 with N
simultaneous committers, so the orchestration layer MUST cap simultaneous
zone-year committers to ~4–8 (measured: N=2 → ~0.5 retries/0.5 s commit; N=8 →
~3.5/1.3 s; N=16 → ~7.5/2.2 s; N=120 → ~58/15 s). Cross-group conflict-freedom
held: zero unresolvable conflicts at every N.

> **The cap does NOT bound assembly throughput, and reading it that way cost real
> effort (noted 2026-07-30).** The gate wraps `session.commit` only — not the shard
> writing that precedes it — so it limits how many commits are *in flight*, each about
> 0.5-1.3 s, not how many zone-years may assemble at once. At a campaign supply of ~11
> zone-years/hour that is one commit every five minutes against eight slots: under 0.1%
> utilisation. Two consequences. Any argument of the form "assembly is capped at 8
> concurrent, therefore throughput is limited to 8/duration" is wrong. And at the shipped
> 8 clusters the limit equals the number of possible committers, so **it cannot bind at
> all** — its value is protecting the configurations we do not currently run (balance
> holds to ~16 clusters, where the measured cost is 7.5 retries and 2.2 s commits) and
> the mixed case, since `mark_zone_year_empty` and the year-milestone tags commit through
> the same gate and so concurrent committers can exceed the cluster count. Kept for those
> reasons, not for throughput.

> **REMOVED 2026-08-27. The cap no longer exists in code**, and the paragraph below describes
> what it did until then.
>
> The curve below is NOT retracted — it is the reason the removal is safe rather than an argument
> against it. What it measures is LATENCY: `N=16` is where a commit reaches **2.2 s** against ~1 s
> serial, and `N=120` where it reaches **15 s**. "Cross-group conflict-freedom held: zero
> unresolvable conflicts at every N", including 120 — six times this campaign's `2N=20` ceiling at
> `max_parallel_clusters=10`. So the gate bounded a slowdown measured in seconds, not a failure.
>
> **Two corrections to my own reasoning, recorded because both were load-bearing.** The removal was
> first argued from a claim that the campaign writes per-ROI stores; it does not, and that premise
> is withdrawn — `global_store()` returns one repo holding all 120 zone groups, and commits share a
> branch tip. And this note first said N≥16 was "twice the cluster count this campaign runs", as
> reassurance; it is the reverse — the fleet's ceiling is 2N, so 10 clusters reach 20.
>
> Reopen criterion is now **N ≥ 120**, detected by the `COMMIT <secs>` line in
> `commit_with_rebase`, which is the only site every commit passes through. See
> [`design/commit-gate-removal-2026_08.md`](../design/commit-gate-removal-2026_08.md).

**Enforced in code from 2026-07-28 until 2026-08-27.** The cap was a Prefect global
concurrency limit (`commit_limit_name`) held around each commit, and `run_global_campaign`
upserted its VALUE at preflight to
`min(max_parallel_clusters, MAX_SIMULTANEOUS_COMMITTERS=8)`. Previously only the
limit's *name* was threaded through and the number lived on the server, where it
could silently drift from this constraint. Duty-cycle measurements showing how
much headroom this leaves are in
[`design/campaign-cluster-sizing.md`](../design/campaign-cluster-sizing.md).

### D7 — Snapshot hygiene: tags + expiry policy (FIRM policy; cadence PENDING T6)

Tag every completed zone-year and each year-complete milestone (tags protect
snapshots from expiry). No GC during active backfill, or only with cutoffs
older than the campaign start (GC's `delete_object_older_than` must predate
the oldest concurrent session). `expire_snapshots` → `garbage_collect`, always
`dry_run=True` first; GC is LIST-bound over the whole repo prefix (hours at
10⁸+ objects — sized in T6). Failed fork sessions orphan chunks until GC;
budget storage for that. Rollback stays `reset_branch` (`rollback_commits`).

**Run 1:** expire → GC → rollback all worked on S3 (tagged snapshot protected,
8.4 MB reclaimed, clean `reset_branch` + re-commit). But the repo was tiny (~100
objects), so the run gives **no GC-duration-at-scale number** — that still needs
a large-repo timing run before a cadence is fixed. Warm-up: run 1 T7 saw **zero
`503 SlowDown` from 50→400 concurrent PUTs** on `arbol-tessera-embeddings-dev`
(an already-partitioned bucket), so no cold-bucket ramp is needed there; a
brand-new bucket may still need warm-up.

### D8 — Everything opt-in; vanilla stores unchanged (FIRM)

All group-aware functionality takes a `group: str | None = None` parameter
(or equivalent), with `None` producing today's single-root-group behavior.
Existing entry points keep their signatures and semantics. Same for sharding
(if adopted) and split configs. Consumers of current single-store output are
unaffected.

### D9 — Pin icechunk ≥ 2.1.1 before any benchmarking or build (FIRM)

The concurrent-manifest-fetch bug (#2158: ~25× slowdown, fixed post-2.0.4)
would poison every read benchmark run on our current 2.0.4 pin.

## Rejected alternatives

- **Per-year physical prepend** (resize → `shift_array` → write front): works
  on paper, but repeats a metadata-update commit 8× per zone against a
  3-month-old feature, each one a conflict hazard for concurrent writers.
  Pre-allocation achieves the same timeline at zero cost. (D1)
- **Per-slab commits** (yield-embeddings coarsen pattern): ~10⁵–10⁶ commits
  globally; blows the designed commit-count envelope and maximizes CAS
  contention. (D6)
- **Sharding as the foundation**: unproven read path in icechunk (no
  coalescing, k GETs per k inner chunks), whole-shard write amplification,
  "not heavily tested" per maintainers. (D3)
- **Fine spatial manifest splits by default**: 10⁵–10⁶ manifests repo-wide
  bloat the snapshot rewrite in every commit. (D4)
- **Consolidated metadata**: unsupported by icechunk by design; fails early.

## Consequences

- New code (all opt-in per D8): group-aware seeding
  (`create_empty_store_from_coords` must stop clobbering root), group
  threading through write/open/resolve paths, per-group attrs (CRS,
  `_manifest`, `years_complete`), a repo+group path model in `BucketPaths`,
  a commit-retry/rebase helper, GC/expiry helpers, and a zone-fill
  orchestration layer (fork/merge batching) that **caps simultaneous zone-year
  committers to ~4–8** (run-1-mandated; see D5/D6).
- **Sharded output (D3): both `embeddings` and `scales` are written in 2048²
  shards of 256-px inner chunks, by a shard-aligned + land-masked writer** (real
  data into land inner chunks, ocean left at fill so the codec elides it → one
  lean shard object per shard, no RMW, no dense nodata). `scales` must be sharded
  too, not left unsharded — otherwise its per-tile object count caps the
  reduction (d3v2 E4). The assembly writer must emit whole shards, never dribble
  inner chunks.
- Partner-facing facts to socialize early: ~1.7 PB total; the one-repo vs
  repo-per-zone tradeoff and its kill criteria (D5); annual timestep
  convention and empty-year read semantics (D1).
- Watch upstream: zarr release containing partial-shard-read optimization
  (PR #3004) — re-run T1's sharded variant when it ships; icechunk #1600 and
  #1558 resolutions may relax D4/D6 parameters.

## Run 1 evidence (2026-07-14, S3 bench, icechunk 2.1.1)

Full T0–T7 pass against `s3://arbol-tessera-embeddings-dev/global-embeddings`
(run id `run1`, 689 metric rows). Numbers that moved decisions:

- **T0 (cross-group conflicts):** 3 simultaneous commits to distinct groups →
  retries [1, 2, 0], **0 unresolvable, 0 failed** — cross-group conflict-freedom
  holds under real object-store CAS (local couldn't show this). Same-chunk:
  `UseOurs` resolved, bare `ConflictDetector` correctly left one writer
  unresolvable. → underpins D5/D6.
- **T1 (reads, full 5-variant sweep):** cold point-vector p95 — `c256_full`
  193 ms (p50 117, 4.2 MB/pt), `c256_sharded` 199 ms (p50 **29**, ~4.2 MB/pt
  real), `c500_band4` 219 ms (**31.8 MB / 32 GETs**/pt), `c384_full` 334 ms
  (9.2 MB/pt), `c500_full` 339 ms. Build wall: full 152 s, sharded 420 s (2.8×).
  Aggregate read phase: sharded 563 s vs full 1323 s. → **D2 FIRM (256 + full
  band); D3 now a strong contender** (reads tie/better, 64× fewer objects, write
  2.8× — pending shard-aligned write test).
- **T2 (writes/splitting):** year-fill commit **flat** 0.367→0.353 s across 9
  years (no #1600 growth with time@1). Split A/B: `time1` → 7 manifests / ~1.5 KB
  snapshots; `time1_spatial` → **379 manifests / 14 KB snapshots / 5.5 s commit
  spike**. → D4 FIRM. Commit RSS ~2.3 GB at ≤12.5 K refs (consistent with the
  ~400 B/ref model; bench zone tops out ~12 K refs, so the 10⁷ regime is
  unreached — needs a larger zone to probe #1558).
- **T3 (pre-alloc/prepend):** seed data-chunks == 0 (metadata-only holds);
  prepend shift commits cheap (0.3–0.6 s) but manifest grew 0.55→8.2 MB over 8
  shifts; shift-vs-writer conflict `unresolvable`. → D1 FIRM.
- **T4 (120 groups):** snapshot 5 KB→38 KB (12→120 groups), per-group commit
  flat (~0.25 s), single-group open 0.15 s, **whole-repo `open_datatree` ~31 s**.
- **T5 (contention):** retries ≈ N−1; N=16 breaches the kill threshold. → the
  D5/D6 commit-concurrency cap.
- **T6 (GC):** expire/GC/rollback work; 8.4 MB reclaimed; tiny repo → no
  scale-duration number.
- **T7 (ramp):** 0× `503 SlowDown` at 50→400 PUTs on this (warm) bucket.
- **T8 (D3 sharding, runs `d3`/`d3v2`):** E1 wire 1.23 vs 8.69 MB/pt
  (reproducible); E2 masked-aligned write 0.46× the unsharded build, lean shards
  ~3.5 GB (dense mode 12.6 GB); E3 scattered p95 131 vs 217 ms; E4 embeddings
  403→9 objects (masked, post-GC), chunkwise churns 808→414. → adopt
  `c256_sharded`, shard `scales` too.

## Evidence (key sources)

- Chunk sizing: Earthmover "I/O-Maxing Tensors in the Cloud" (2025-11-25,
  3–15 MB optimum); AWS S3 range-GET guidance (8–16 MB); AlphaEarth mosaic
  layout (source.coop `tge-labs/aef-mosaic`, 2026-03-25); GeoZarr best
  practices draft (geozarr-spec #117).
- Sharding: icechunk PR #1114 (v1.0.3, 2025-07-24), PR #2010 (v2.0.0),
  issues #1019, #1316 (open, no coalescing), #1317 (maintainer stance);
  zarr-python PR #3299 (3.1.2), PR #3004 (merged, unreleased at 3.2.1).
- Manifests/scale: icechunk performance guide (split >1 M refs/array);
  GOES-16 virtual Zarr blog (2026-06-02: 7.1 B refs, ~11 B/ref on disk);
  issues #1558 (72 M-ref commit panic, ~400 B/ref RSS), #1600 (append
  growth despite splitting, open), #2158 (manifest-fetch fix post-2.0.4),
  #1462 (list_prefix manifest loads, open), #1464 (`max_arrays_to_scan`),
  #1342 (no manifest consolidation); spec v2.1 (per-array manifests;
  snapshot lists all nodes+manifests).
- Transactions/prepend: icechunk moving-chunks guide (`shift_array` /
  `reindex_array`, 2.0); parallel-writes docs (fork/merge, spawn-only);
  conflicts reference (taxonomy; only chunk conflicts auto-resolve);
  version-control docs; discussion #802 (uncooperative-mode queueing).
- GC: expiration docs; Earthmover GC blog (2025-05-30, cadence guidance);
  changelog (2.x GC parallelization, dry_run).
- S3: Earthmover scalability blog (2025-04-10: 230 k reads/s, random chunk
  keys, SlowDown = resharding ramp).

## Conclusion — the settled architecture (2026-07-14)

Scoping is complete; every decision below is FIRM with S3-bench evidence. Build
to this:

- **One Icechunk repo, 120 Zarr groups** (one per UTM zone, own CRS). Metadata
  cost is trivial (38 KB snapshot at 120 groups); readers open a single group,
  never the whole datatree. (D5)
- **Per-group arrays `(time, northing, easting, band)`**, time chunk 1, band not
  split. **`embeddings` int8+zstd and `scales` float32+PCodec, both written in
  2048² shards of 256-px inner chunks.** (D2, D3)
- **Pre-allocate the full 2017–2025 time axis** at seed (metadata-only); fill
  2025 first and backfill as region-inserts; never prepend during fills. (D1)
- **Manifest split time@1** (one manifest per year per array), spatial split only
  if a year-manifest exceeds ~2–4 M refs. (D4)
- **Writer: cooperative fork/merge, TWO commits per (zone, year)** — shards, then that
  year's attrs (D6, amended 2026-07-30; this is what lets two years of one zone be
  written concurrently) — shard-aligned
  + land-masked (whole lean shards, ocean elided), spawn/forkserver only. Cap
  simultaneous zone-year committers to ~4–8 (uncoordinated commits storm,
  O(N²)). (D6)
- **Hygiene:** tag each completed zone-year; expire+GC with cutoffs older than
  the campaign start; `reset_branch` for rollback. (D7)
- **All new functionality opt-in**; vanilla single-group stores unchanged. Pin
  icechunk ≥ 2.1.1. (D8, D9)

**Deferred (operational, not blocking):** GC wall-time at 10⁸-object scale (D7) —
measure against a large repo before setting the production GC cadence. Watch
zarr PR #3004 (partial-shard-read); it only improves the sharded read path.

**Next:** implement this write path in `tessera_embeddings.storage` (group-aware
seeding, sharded land-masked writer, commit-concurrency cap, GC/expiry helpers).
The scale-test scripts (`scripts/scale_tests/`) remain as regression harness and
for the deferred GC-at-scale measurement.
