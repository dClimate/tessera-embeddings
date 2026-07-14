# D3 settlement plan — shard or not shard the global embeddings store

> **RESOLVED (2026-07-14): ADOPT `c256_sharded`, and shard `scales` too.** Runs
> `d3` + `d3v2` (t8) passed all gates: reads leaner *and* faster (wire 1.23 vs
> 8.69 MB/pt, reproducible; scattered p95 131 vs 217 ms), the shard-aligned +
> land-masked writer builds at **0.46×** the unsharded time with lean ~3.5 GB
> shards, and the embeddings array consolidates 403→9 objects (post-GC). E4 also
> showed the object-count win is capped unless `scales` is sharded too, so it is.
> Production writer must be shard-aligned + land-masked. See ADR-008 D3 for the
> full write-up; the plan below is retained as the method of record.

Companion to ADR-008 decision **D3**. Run 1 flipped sharding from "optional,
probably not" to "strong contender" — this plan defines the experiments that
turn that into a decision, the criteria for calling it, and the tooling
(`scripts/scale_tests/t8_sharding.py`) that runs them.

## The question

Ship the global embeddings store as **`c256_full`** (256 px chunks, unsharded)
or **`c256_sharded`** (256 px inner chunks packed into 2048² shards, ~64 inner
chunks / ~0.5 GB per shard object)? D2 already fixed the inner geometry (256 px,
full 128-band, int8+zstd); D3 is only "wrap those chunks in shards or not."

## What run 1 already settled

From the T1 bench sweep on S3 (`c256_full` vs `c256_sharded`, one timestep of a
20 000² zone):

| metric | c256_full | c256_sharded | read |
|---|---|---|---|
| point p95 (cold) | 193 ms | 199 ms | tie (6 ms = noise) |
| point p50 (cold) | 117 ms | **29 ms** | sharded 4× (locality-driven) |
| read-phase wall | 1323 s | **563 s** | sharded 2.3× faster aggregate |
| bulk throughput (max) | 1897 MB/s | 2014 MB/s | tie |
| build wall | **152 s** | 420 s | full 2.8× faster |
| refs (9 yr, global land, analytic) | ~205 M | **~3.2 M** | sharded ~64× fewer |

So: **reads do not lose** (p95 tie, p50/aggregate better, no coalescing penalty
at these sizes), and sharding slashes object count ~64×. The **only** strike is
the 2.8× build cost. For a write-once/read-many published dataset that's a
one-time cost buying a permanent read + object-count win — *if* the write
penalty is containable.

## Open sub-questions (what this plan must answer)

1. **Is the 2.8× write penalty an artifact of unaligned writes?** The T1 builder
   writes 256-px chunks that dribble into 2048² shards → heavy read-modify-write
   of shard objects. A **shard-aligned** writer (emit a whole shard's inner
   chunks in one operation) should avoid the RMW. Does it? How close to the
   unsharded 152 s baseline does it get?
2. **Does sharding actually do partial reads, or fetch whole shards?** The
   analytic `bytes_fetched` metric reported 16.5 MB/point for sharded (whole
   shard); latency (199 ms, not ~800 ms) implies ~4.2 MB (one inner chunk).
   Confirm with a **bytes-on-the-wire** measurement.
3. **Does the p50 read advantage survive scattered access?** Sharded p50 (29 ms)
   came from large shards keeping *nearby* sampled points warm. Global point
   queries are often scattered. Under a **scattered** (cache-hostile) access
   pattern, does sharded p95/p50 stay within ~1.2× of `c256_full`?
4. **Does the object-count win hold empirically?** ~64× is analytic; confirm the
   actual object + manifest-byte counts for equal-data sharded vs full stores.

## Experiments (`t8_sharding.py`)

Each is an idempotent phase emitting to the shared metric schema, so `report.py`
picks them up. All run in-region on the bench box, `--backend s3`.

- **E1 — bytes-on-wire (`bytes_on_wire`).** Build a small `c256_full` and
  `c256_sharded` store; in a fresh process, snapshot `psutil.net_io_counters().
  bytes_recv`, do N point-vector reads, snapshot again; `Δ/N` = real bytes per
  point. Emits `bytes_fetched` (real) per variant. **Decides sub-Q2.**
- **E2 — write alignment (`write_alignment`).** Build the same zone three ways
  and compare build/commit wall + bytes written: (a) `c256_full` chunkwise
  (reference), (b) `c256_sharded` chunkwise (the T1 path), (c) `c256_sharded`
  **shard-aligned** (full 2048² shard blocks, one write per shard). Reports the
  penalty ratio (c)/(a) and (b)/(a). **Decides sub-Q1** — the crux.
- **E3 — scattered reads (`scattered_reads`).** Point reads sampled uniformly
  across the whole zone (not clustered in land chunks / not sharing shards),
  cold, for both variants; p50/p95 vs the clustered T1 pattern. **Decides
  sub-Q3.**
- **E4 — object + manifest count (`object_count`).** After E2's equal-data
  builds, count objects and manifest bytes under each store prefix. **Decides
  sub-Q4** (and sanity-checks the analytic 64×).

Notes: E2/E4 write dense shards for the shard-aligned case (a whole shard is one
write), so those stores hold more bytes than the land-only chunkwise store —
report build-wall *and* bytes-written so the volume difference is visible; the
**ratio** to the unsharded baseline is the signal, not the absolute wall. Run at
`--scale bench` for real numbers (a few shards is enough — the per-shard write
mechanics extrapolate); `--scale tiny` exercises the code on a laptop sub-shard
zone.

## Decision rule

Adopt **`c256_sharded`** iff **both**:
- **E2:** shard-aligned write penalty ≤ ~**1.5×** the `c256_full` baseline
  (per-GB, to neutralize the dense-shard volume difference), **and**
- **E3:** scattered cold point-read **p95 within ~1.2×** of `c256_full`.

(E1 is a gate: if it shows sharded fetches whole shards — ~16 MB/point — reject
regardless, the read economics are wrong. E4 is confirmatory.)

Otherwise ship **`c256_full`**. Either outcome leaves D2 (256 px, full band)
intact; D3 only chooses the wrapper.

## Sequencing & cost

1. `t8_sharding.py --run-id d3 --backend s3 --scale bench --bucket … --store-root …`
   — runs E1→E4. Est. ~30–60 min, a few tens of GB S3 (dense shards), well under
   the campaign budget.
2. `report.py --run-id d3` → the D3 row now reads from t8 (write ratio, real
   bytes, scattered p95, object counts).
3. Apply the decision rule; update ADR-008 D3 to FIRM (with the chosen layout);
   `teardown.py` the d3 stores.

## Risks & caveats

- **Shard-aligned = dense shards.** Ocean inner chunks inside a written shard
  cost space; at global scale that means writing more nodata than the land-only
  chunkwise path. Quantify in E2/E4; if dense-shard overhead is large, a
  land-masked shard writer (skip all-ocean shards) is the mitigation.
- **Write penalty at true scale.** E2 measures per-shard mechanics on a few
  shards; a full-zone write could reveal manifest/commit effects not visible
  small. If E2 is borderline, re-run at a larger multi-shard zone before
  committing.
- **zarr partial-shard-read optimization (PR #3004) is unreleased.** If it ships
  in a later zarr, sharded tile/multi-chunk reads get faster still — re-run E1/E3
  then; it can only help sharding's case.
- **Locality is workload-specific.** E3's scattered pattern is a conservative
  proxy; if the partner's real query mix is known (tiles vs points vs regions),
  weight E3 toward it.
