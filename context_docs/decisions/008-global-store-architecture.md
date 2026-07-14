# 008 — Global embeddings store: architecture decisions

**Status:** Provisional — decision register below marks each decision FIRM or
PENDING a named test in the companion test plan
(`context_docs/design/global-store-test-plan.md`). Supersede individual
decisions with follow-up ADRs as test evidence lands.

**Run 1 (2026-07-14, S3 bench, icechunk 2.1.1, `arbol-tessera-embeddings-dev`):**
first full T0–T7 pass landed. It moved **D1** and **D4** to FIRM, added a hard
commit-concurrency constraint to **D5/D6**, and validated the D7 mechanism. It
did **not** settle **D2/D3** — only the `c256_full` variant was built, so the
chunk-shape/sharding comparison still needs the remaining four variants. See
"Run 1 evidence" at the bottom for the numbers behind these changes.

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

### D2 — Chunk shape: full band dimension, spatial 256–500 px (PENDING T1 — variant sweep still owed)

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

### D3 — Sharding: optional variant, not the foundation (PENDING T1/T2)

Zarr v3 sharding works in icechunk (fixed v1.0.3, caveat-free since 2.0.0,
requires zarr-python ≥ 3.1.2) but: maintainers call it "not heavily tested";
reading k inner chunks costs k separate ranged GETs (no coalescing —
icechunk #1316 open; the zarr-side fix PR #3004 merged but unreleased as of
zarr 3.2.1); partial-shard writes are whole-shard read-modify-write. Icechunk
manifests already solve the metadata side of the small-object problem
(Earthmover's own ERA5 uses ~500 KB chunks, unsharded). Sharding earns its
place only if T1 shows small chunks win reads *and* T2 shows ref-count pain.
If adopted: inner chunks per D2, shards ~2048² px (~0.5 GB objects),
shard-aligned writes mandatory.

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

### D5 — One repo, 120 groups — viable *with a commit-concurrency cap* (PENDING D2/D3; commit constraint FIRM from run 1)

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
open a single zone group, never the datatree (icechunk #1462). Final one-repo
vs per-zone go/no-go stays open only pending D2/D3 and a *paced-commit* re-test;
the naive-concurrency question is settled.

### D6 — Commit strategy: cooperative fork/merge, one commit per zone-year (FIRM shape; pacing cap FIRM from run 1)

Within a zone-year: `session.fork()` → pickled ForkSessions to workers →
`merge(*sessions)` → **one commit** (multiprocessing start method must be
spawn/forkserver — fork deadlocks icechunk's runtime). Across zones: commits
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
- **T1 (reads, `c256_full` only):** point-vector p95 ≈ 193 ms, open ≈ 0.13 s,
  one 4.2 MB chunk per point read. **Only one variant built — D2/D3 not yet
  decidable.** Remaining sweep: `c500_band4`, `c500_full`, `c384_full`,
  `c256_sharded`.
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
