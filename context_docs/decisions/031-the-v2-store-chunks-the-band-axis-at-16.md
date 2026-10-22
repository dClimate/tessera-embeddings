# 031 — The v2 store is band-first, chunked at 16 bands

**Status:** Accepted 2026-10-22 (repo owner). The layout is band-first with band-split shards, and
inference makes the reorder as it stages each tile (§4). For v2 stores only, this supersedes D2 of
[ADR 008](008-global-store-architecture.md). The v1.1 store keeps D2, and targeted updates go on
writing it that way.

v2 emits 128 dimensions in Matryoshka order: the first 16, 32 and 64 are each usable on their own.
D2 never splits the band axis, so a reader who wants 16 dimensions still fetches all 128. This
records the layout that lets a reader pay only for the prefix it uses. It was scoped with the
scale-test harness: T9 locally, T10 on AWS
([results](../../scripts/scoping/scale_tests/v2_layout_results.md)).

---

## 1. Settled in either band order

**The 2,048² shard stays.** One inference tile is one shard, and the 4,096-px ingest chunk is set
by Dask's scheduler, so nothing upstream of assembly moves. Only what sits inside the shard changes.

| choice | value | why |
|---|---|---|
| band chunk | 16 | the smallest prefix v2 was trained on |
| inner chunk | 512 × 512 × 16 | 256² builds 4× the Dask tasks at default chunks and 204–260 serial requests per 16-d pull of 2 × 2 shards on zarr 3.2; 1,024² makes a full vector cost 134 MB |
| int8 codec | zstd at level 1 | within 3% of the default level's bytes at 0.6× its encode CPU, once bands are split |
| inner-chunk order | zarr's default (Morton) | the other orders moved request counts by under 20%, and an order is not persisted, so it would need library code on every write path |
| `scales` | 256² inner chunks, as v1.1 | its geometry made no measurable difference; PCodec compresses it about 7× |
| float arrays | PCodec | unchanged |
| readers | zarr ≥ 3.3, shard-aligned Dask chunks | zarr 3.3 merges a shard's ranges into concurrent requests; Dask's default makes one task per inner chunk, each refetching the shard index |

## 2. What was measured

Eight real v2 tiles: four from Iowa (cropland, including a 52%-valid edge tile) and four from Puget
Sound (city, open water, coast and conifer forest). T10 cycled them into a 100-shard store on S3 and
read it cold from an in-region r7i.4xlarge, two passes in opposite arm order, every arm in one
session. Encode CPU is per shard on a Graviton m7g.4xlarge, the assembly runner's architecture. All
four candidates use 512 × 512 × 16 inner chunks and zstd at level 1.

| | C0: v1.1 geometry | band-last, in shard | band-last, band-split shards | band-first, in shard | band-first, band-split shards |
|---|---|---|---|---|---|
| encode CPU per shard | 3.36 s | 0.83× | 0.81× | 0.75× | 0.74× |
| stored | 38.9 GB | −3% | −3% | **−23%** | **−23%** |
| 16-d pull, 8 × 8 tiles | 20.2 s | 5.4 s | 5.0 s | 5.5 s | **4.9 s** |
| 64-d pull, 8 × 8 tiles | 24.2 s | 20.1 s | 18.0 s | 16.7 s | **14.7 s** |
| **128-d pull, 8 × 8 tiles** | 27.9 s | **48.1 s** | **48.5 s** | 29.4 s | **27.7 s** |
| 128-d point p50, zarr 3.4 | 419 ms | 457 ms | 424 ms | 460 ms | **391 ms** |
| 16-d pull through Dask at default chunks | 43.4 s | 11.5 s | 9.0 s | 11.7 s | **8.6 s** |
| 128-d point p50, zarr 3.2 | 472 ms | **1,474 ms** | 414 ms | **1,132 ms** | 420 ms |
| 16-d pull, 8 × 8 tiles, zarr 3.2 | 19.9 s | 13.3 s | 5.0 s | 14.1 s | 5.2 s |
| 128-d pull, 8 × 8 tiles, zarr 3.2 | 30.0 s | 47.3 s | 47.0 s | 29.0 s | **27.8 s** |

**Band-16 chunks make a 16-d pull about 4× faster in either order**, and a 64-d pull faster too.

**Every layout passes the write gate (≤ 1.20× C0).** The library's own assembly code wrote 64
staged Iowa tiles into each layout on Graviton.

- **Band-last:** 0.90–0.91× C0's worker CPU per tile.
- **Band-first:** 1.17× C0's worker CPU, at 0.94–1.00× its wall time. This was measured with a
  prototype writer that reorders each band-last staged tile as it writes it.
- **What overstates band-first:** a band-first pipeline does that reorder earlier, in inference
  (§3). Without it, assembly costs what encoding does, which is 0.74× C0.

**Every layout is lossless.** Read back, every array of all 64 assembled tiles is identical to C0's
in every layout, band-first arrays after reordering.

**In band-last order a 128-d bulk pull is 1.5–1.7× slower than C0**, across both sessions, which
fails the bulk-throughput gate (≥ 0.9× of C0). The bytes on the wire are C0's. The time goes into
putting each pixel's 128 values back together from eight 16-band pieces, 16 bytes at a time.
Neither a fetch concurrency of 64 nor band-aligned Dask chunks recovers it; both made every arm
slower, C0 included.

**In band-first order the same pull is level with C0**, and it moves 23% fewer bytes. Each piece is
a set of whole planes, so putting it back is a few large copies: 0.06 s per 128-d tile against
0.15 s band-last, on one laptop core.

**In-shard band-16 fails the old-client gate on full vectors, in either order.** zarr 3.2 fetches a
shard's eight pieces one after another, so a 128-d point takes 2.4× C0's time in band-first order
and 3.1× in band-last. Band-split shards pass, because even zarr 3.2 fetches separate objects
concurrently. On zarr 3.2, band-first band-split shards are faster than C0 on every workload.

**Neither gate result is close to the noise.** The gates fail only if an interval clears the
threshold, so a separate run took 300 probes per arm in each of two passes, the arm order reversed
between them. Each pass carries its own 95% bootstrap interval on the median, in ms:

| full-vector point, pass 1 / pass 2 | zarr 3.4 | zarr 3.2 |
|---|---|---|
| C0 | 442 [425, 452] / 422 [406, 436] | 475 [464, 490] / 387 [376, 394] |
| band-first, band-split shards | 370 [362, 384] / 400 [391, 409] | 387 [377, 393] / 357 [350, 364] |
| band-first, in shard | 443 [438, 453] / 451 [439, 464] | **1,075 [1,045, 1,102] / 1,020 [997, 1,037]** |

In-shard band-first on zarr 3.2 takes 2.3× and 2.6× its pass's C0, with every interval clear of 2×.
Band-split band-first is faster than C0 in every pass on both clients.

## 3. Band-last against band-first

Band-last is v1.1's order, `(time, northing, easting, band)`. Band-first is
`(time, band, northing, easting)`. Within a band-first chunk, each band is a contiguous plane.

| | band-last | band-first |
|---|---|---|
| **ingest** | unchanged | unchanged: the same code and settings, so the same performance (below) |
| **inference, inputs** | unchanged | unchanged: the model reads pixels, assembled from per-band mosaics as today |
| **inference, outputs** | unchanged | reordered once per tile on the actor's background writer thread (below) |
| **appending a year** | an end-append on the time axis (ADR 008 D1) | the same: time stays the leading axis, chunked at 1, so a new year's chunks are new keys and no existing chunk moves |
| **encode CPU** | 0.81–0.83× C0 | 0.74–0.75× C0 |
| **assembly worker CPU** | 0.90–0.91× C0 | 1.17× C0 when assembly reorders band-last staged tiles; encode cost (0.74×) when inference stages them band-first |
| **storage** | −3% | −23%, band-planar compression for free |
| **reads** | 16-d about 4× faster than C0; 128-d bulk 1.5–1.7× slower | 16-d about 4× faster; 64-d 1.4–1.6× faster; 128-d level with C0 |
| **readers' tools** | per-pixel readers (scikit-learn) get `(…, band)` directly | rioxarray refuses `(y, x, band)`; PyTorch convolutions take channels first; AlphaEarth's Source Coop copy is `(time, band, y, x)` |
| **readers who want the other order** | channel-first readers transpose | per-pixel readers transpose (below) |
| **our two stores** | one order | v1.1 and v2 differ. xarray users index by name and see no difference; array-level code must check the dimension names |
| **library code** | a v2 preset, `_layout_band`, docs and pinning tests | about 60 sites in 17 files plus about 200 test lines, all needing both orders (below) |

**Ingest performance is unaffected by either order.** Ingest takes only the spatial constants
(`INNER_PX`, `SHARD_PX`) and the chunk-sizing function from the layout module. The embedding layout
is read only by seeding and assembly. On a band-first prototype, every mosaic array has the same
chunks, shards, codecs, data type and fill as on main, and the same bytes. The one coupling is
code identity: `config/store_layout.py` sits inside ingest's identity closure, so *any* v2 preset
moves the ingest identity once. Moving the chunk-sizing function into its own module ends that.
The prototype's ingest identity was then the same with and without a preset.

**The inference cost of band-first is about a second of background CPU per tile.** The model emits
one row of 128 values per pixel, so a band-first tile has to be reordered once, after the model runs.
Ingest cannot do it: its mosaics are already one array per spectral band, and the embeddings do not
exist until inference. The cheapest place is the staged write, which already runs on the actor's
single-slot writer thread and overlaps the next tile's work. Writing the existing `(y, x, band)`
buffer through a transposed view costs 0.6–1.0 s of one laptop core per 2,048² tile. It needs no
full-tile copy and no GPU time. Reordering on the GPU or on the main thread instead added 2.5–7.4 s
per tile on an L4 host.

**That writer has room, by the Iowa v2 runs' own logs (786 tiles).**

- **Upload against tile:** the staged upload takes 16.6 s (median), against 164–166 s of tile time,
  with the GPU at 100% (median).
- **Where it already waits:** the next tile waits on the upload only on the 5.6% of tiles that
  finish fast, adding 0.5 s per tile on average.
- **What the reorder adds:** another 1–2 s on each upload lengthens only those waits. That is at most
  about 0.1 s per tile, under 0.1% of inference time.

**A reader who wants the other order pays a transpose either way.** Turning one 128-d tile from
`(band, y, x)` into `(y, x, band)` takes 0.17 s with torch on 10 threads, or 1.2 s with numpy on one.
At 16 dims it takes 0.05 s. A channel-first reader of a band-last store pays the same in reverse.

**Band-first touches every stage after inference.** That covers the inference write path and
staging validation, the shard writer and the update merge, the layout helpers, validation and the
registry, and about 11 sites in yield-embeddings (coarsening and validation). Each must handle both
orders, because the v1.1 store stays band-last. Some failures would be silent. `commit_tiles`, for
example, labels a tile by chunk coordinates 1 and 2, which in band-first order are band and
northing.

## 4. Decision

**Band-first, with band-split shards.** `embeddings` is `(time, band, northing, easting)`, stored as
shards of 1 × 16 × 2,048 × 2,048 holding inner chunks of 1 × 16 × 512 × 512, compressed with zstd at
level 1. Each tile is eight objects, one per 16-dimension group. `scales` and the per-tile arrays keep
their v1.1 geometry.

**It is the only candidate that passes every gate, and it leads every read workload.**

- **16-d pull:** 4.9 s against C0's 20.2 s.
- **64-d pull:** 14.7 s against 24.2 s.
- **128-d pull:** level with C0.
- **Full-vector point:** slightly faster than C0, on both clients.
- **Storage:** 23% smaller, at 0.74× the encode CPU.

Each alternative fails a gate. Band-last fails the bulk gate whichever shard form it takes. Band-first
inside one shard fails the old-client gate on a full-vector point.

**What it costs:**

- **Library code:** the changes after inference listed in §3, kept order-aware for as long as the
  v1.1 store is updated.
- **Inference:** one reorder per tile on the staging writer thread.
- **Objects:** eight embedding objects and PUTs per tile instead of one.
- **Readers:** a transpose for per-pixel readers who want `(…, band)`.

**The reorder is inference's, as it stages each tile.** Staged tiles, the shard writer, updates and
readers all see band-first, so no stage after inference reorders anything. The cost is under 0.1% of
inference time (§3).

**Rejected: reordering in assembly instead.** It would leave inference's code untouched, at 1.17× C0's
assembly CPU rather than 0.74×. It was turned down because band-first all the way through is the
cleaner pipeline, and inference's part of it costs almost nothing.

## 5. Consequences

- **A v2 preset** in `config/store_layout.py`, selected by model version for the global and
  single-area paths. `INNER_PX` stays 256, so the land mask's `chunk_live_256` stays valid.
- **Inference stages v2 tiles band-first.** `ZarrWriter.write_chunk` writes the actor's
  `(y, x, band)` buffer through a transposed view on the writer thread. Staging validation and
  `StagedShardSource` read `(band, y, x)`, and the shard writer and the holes merge index band-first
  arrays as `(time, band, northing, easting)`.
- **The chunk-sizing function moves out of `store_layout.py` first**, as its own change at a store
  boundary. It moves the ingest identity once, and keeps later layout changes from moving it.
- **`_layout_band` takes the band count from the shard**, not the chunk. Otherwise a v2 store is
  stamped `geoemb:dimensions=16`.
- **Reader-facing metadata:** a root attribute naming the valid prefixes (16, 32, 64, 128), and
  reader-guide advice on zarr ≥ 3.3 and shard-aligned Dask chunks.
- **Band-split shards mean 8× the embedding objects, references and PUTs.** The densest v1.1
  zone-year wrote 9,132 tiles, so a v2 zone-year holds at most about 140,000 references across all
  eight arrays. That is far below D4's 10⁶ split threshold, and about 56 MB of commit memory at
  400 bytes per reference.
- **v2 is not yet wired through zone fill, seeding or provenance checks.** That work is separate
  from this record.
