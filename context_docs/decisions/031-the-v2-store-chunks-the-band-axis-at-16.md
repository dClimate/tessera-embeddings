# 031 — The v2 store is band-first, chunked at 16 bands

**Status:** Accepted 2026-10-22 (repo owner). The layout is band-first with band-split shards, and
inference makes the reorder as it stages each tile (§4). For v2 stores only, this supersedes D2 of
[ADR 008](008-global-store-architecture.md). The v1.1 store keeps D2, and targeted updates go on
writing it that way. §6 records narrower and variable-width band chunks, measured for mobile readers
on 2026-10-27 and 2026-10-28; the decision stands.

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
| band chunk | 16 | the smallest prefix v2 was trained on; §6 weighs narrower and variable-width chunks for mobile readers |
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
- **Why 1.17× is an upper bound:** a band-first pipeline does that reorder earlier, in inference
  (§3), so assembly will not pay it. How far below 1.17× it lands is unmeasured: encoding alone costs
  0.74× C0 in band-first order, but assembly also reads and checks the staged tiles and writes the
  other arrays, none of which the reorder changes.

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

Band-last is v1.1's order, `(time, northing, easting, band)`. Band-first is `(time, band, y, x)`,
v2 naming its spatial axes `y` and `x` ([ADR 032](032-v2-stores-name-their-axes-y-and-x.md)).
Within a band-first chunk, each band is a contiguous plane.

| | band-last | band-first |
|---|---|---|
| **ingest** | unchanged | unchanged: the same code and settings, so the same performance (below) |
| **inference, inputs** | unchanged | unchanged: the model reads pixels, assembled from per-band mosaics as today |
| **inference, outputs** | unchanged | reordered once per tile on the actor's background writer thread (below) |
| **appending a year** | an end-append on the time axis (ADR 008 D1) | the same: time stays the leading axis, chunked at 1, so a new year's chunks are new keys and no existing chunk moves |
| **encode CPU** | 0.81–0.83× C0 | 0.74–0.75× C0 |
| **assembly worker CPU** | 0.90–0.91× C0 | 1.17× C0 with assembly reordering band-last staged tiles: an upper bound once inference stages them band-first |
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
- **What the reorder adds:** another 1–2 s on each upload. The next tile can wait at most that much
  longer, so the reorder costs at most 1–2 s per tile, about 1% of tile time. A tile waits only when
  the next one finishes before its upload does, which today is the 5.6% above.

**A reader who wants the other order pays a transpose either way.** Turning one 128-d tile from
`(band, y, x)` into `(y, x, band)` takes 0.17 s with torch on 10 threads, or 1.2 s with numpy on one.
At 16 dims it takes 0.05 s. A channel-first reader of a band-last store pays the same in reverse.

**Band-first touches every stage after inference.** That covers the inference write path and
staging validation, the shard writer and the update merge, the layout helpers, validation and the
registry, and about 11 sites in yield-embeddings (coarsening and validation). Each must handle both
orders, because the v1.1 store stays band-last. Some failures would be silent. `commit_tiles`, for
example, labels a tile by chunk coordinates 1 and 2, which in band-first order are band and `y`.

## 4. Decision

**Band-first, with band-split shards.** `embeddings` is `(time, band, y, x)`, stored as
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
readers all see band-first, so no stage after inference reorders anything. The cost is at most about
1% of inference time (§3).

**Rejected: reordering in assembly instead.** It would leave inference's code untouched, at 1.17× C0's
assembly CPU, above what assembling band-first staged tiles will cost. It was turned down because
band-first all the way through is the cleaner pipeline, and inference's part of it costs at most about
1% of tile time.

## 5. Consequences

- **A v2 preset** in `config/store_layout.py`, selected by model version for the global and
  single-area paths. `INNER_PX` stays 256, so the land mask's `chunk_live_256` stays valid.
- **Inference stages v2 tiles band-first.** `ZarrWriter.write_chunk` writes the actor's
  `(y, x, band)` buffer through a transposed view on the writer thread. Staging validation and
  `StagedShardSource` read `(band, y, x)`, and the shard writer and the holes merge index band-first
  arrays as `(time, band, y, x)`.
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
- **Band order is read from each array's dimension names**, never inferred from the model or the
  store. Code that indexes `embeddings` asks where `band` sits in its `dimension_names`, so one path
  reads both the band-last v1.1 store and a band-first v2 one.
- **The preset follows the store's model**, chosen the way the checkpoint and the worker image
  already are: from the model's `ModelFamily` record, and for a global store from the model it was
  seeded for.

## 6. Narrower and variable-width band chunks

**Status:** measured 2026-10-27 and 2026-10-28 at the repo owner's request, report only. §4 stands.
The runs are archived in [v2_narrow_band_results.md](../../scripts/scoping/scale_tests/v2_narrow_band_results.md).

**Narrower band chunks make a phone's reads 2–4× faster.** On a phone's link, time follows bytes,
and a 4-d read of 4-band chunks fetches a quarter of what the accepted layout fetches. But every
candidate costs full-vector or Dask readers something, and a browser's Zarr client cannot read the
variable-width ones. The best-placed alternative is 16-band shards holding 4-band inner chunks, which
pass every read gate (the end of this section).

**The mobile reader, as the owner defined it for this study:**

- **What it reads:** map-view previews, the first three or four dimensions shown as an image, for a
  256² web-map tile or a phone viewport at native 10 m. It also runs similarity over a small area
  at 4 or 8 dimensions.
- **How it reads:** Zarr straight from S3, as a browser or app does. That means one GET per inner
  chunk, no merged requests, about six connections, and a phone's link.

**The candidates** are all band-first in 2,048² shards, compressed with zstd at level 1, and measured
against C0 and the accepted layout in the same sessions. The object gate W2 was relaxed to 32× C0's
objects so that band-split 4 could compete.

| | band axis per shard | inner chunk | objects per tile | chunk grid |
|---|---|---|---|---|
| accepted (§4) | 16 | 512² × 16 | 8 | regular |
| band-split 8 | 8 | 512² × 8 | 16 | regular |
| band-split 4 | 4 | 512² × 4 | 32 | regular |
| 16-band shards, 4-band inner chunks | 16 | 512² × 4 | 8 | regular |
| variable 4, 4, 8, 16, 32, 64 | as listed | 512² × 4 | 6 | variable |
| variable 4, 4, 8, 112 | as listed | 512² × 4 | 4 | variable |
| variable 8, 8, 112 | as listed | 512² × 8 | 3 | variable |

zarr requires one inner chunk for the whole array, and it must divide every shard. So a variable set
that includes a 4-band shard has 4-band inner chunks throughout, and its 64-band object holds 16
pieces per 512² square. T9 also measured three things that did not reach T10:

- **256² inner chunks.** They cut a map tile's bytes by another 4×, but they triple a viewport's
  requests, store 5–7% more, and cost 1.3–1.45× the encode CPU.
- **16-band shards with 8-band inner chunks.** They read like band-split 8 in fewer objects, and
  the checkpoint took the 4-band version forward instead.
- **Inner-chunk orders within a shard.** T10 kept one, the variable 4, 4, 8, 16, 32, 64 set with
  each pixel's pieces side by side. On zarr 3.4 it needs 12 requests for a full-vector point where
  the default order needs 23, but it read no faster.

**What they buy a mobile reader.** These are cold reads through the direct-read client on two links
shaped with `tc netem`: *good* (40 ms round trip, 20 Mbit/s down) and *poor* (Lighthouse's mobile
profile, 150 ms and 1.6 Mbit/s). Layouts with the same inner-chunk width read alike, within about 5% of
one another on the poor link, so they share a column below:

- **8-band inner chunks:** band-split 8, and variable 8, 8, 112.
- **4-band inner chunks:** the other four candidates.

Each cell is good-link seconds / poor-link seconds, with the MB fetched in brackets.

| | accepted: 16-band inner chunks | 8-band inner chunks | 4-band inner chunks |
|---|---|---|---|
| 256² map tile, 4-d | 1.7 / 17 s (3.3 MB) | 1.0 / 9.2 s (1.7 MB) | 0.6 / 4.7 s (0.8 MB) |
| 256² map tile, 8-d | 1.6 / 18 s (3.3 MB) | 1.0 / 9.3 s (1.7 MB) | 0.9 / 9.1 s (1.7 MB) |
| 1,024 × 2,048 viewport, 4-d | 8.3 / 103 s (19.7 MB) | 4.4 / 52 s (9.9 MB) | 2.3 / 26 s (5.0 MB) |
| 1,024 × 2,048 viewport, 8-d | 8.5 / 102 s (19.7 MB) | 4.4 / 52 s (9.9 MB) | 4.4 / 52 s (9.9 MB) |

- **Requests barely matter on a phone's link.** On an 8-d viewport the 4-band layouts make up to
  twice the accepted layout's requests (20 against 10) and still finish in half its time.
- **4-band chunks help a 4-d reader twice as much as 8-band ones.** At 8 dimensions the two are
  level.
- **The first read** of a session loads the manifests. On these 100-tile stores it took 0.2–0.3 s on
  the good link and 0.5–0.8 s on the poor one. At full scale the manifest is larger (the next table).

**What they cost every other reader.** Cold reads from S3 in-region on zarr 3.4 are given as a ratio
to the accepted layout. Encode, assembly and the old client are given against C0.

| | band-split 8 | band-split 4 | 16-band, 4-band inner | variable 4 … 64 | variable 4, 4, 8, 112 | variable 8, 8, 112 |
|---|---|---|---|---|---|---|
| 128-d point p50 | 1.27 | 1.65 | 0.99 | 0.98 | 1.11 | 1.03 |
| 128-d pull, 8 × 8 tiles | 1.25 | 1.45 | 1.08 | 1.12 | 1.15 | 1.05 |
| 16-d pull, 8 × 8 tiles | 1.08 | 1.31 | 1.07 | 1.21 | 1.21 | 1.10 |
| 16-d pull through Dask, default chunks | 1.45 | 2.45 | 2.09 | 2.47 | 2.32 | 1.48 |
| zarr 3.2, slowest workload vs C0 (accepted: 0.97) | 1.19 | 1.52 | 1.23 | 2.81 | 4.74 | 3.14 |
| assembly worker CPU vs C0 (accepted: 1.19) | 1.23 | 1.31 | 1.30 | 1.30 | 1.31 | 1.23 |
| encode CPU per shard vs C0, Graviton (accepted: 0.78) | 0.80 | 0.86 | 0.82 | 0.82 | 0.82 | 0.79 |
| embeddings manifest, densest zone-year | 3.1 MB | 6.1 MB | 1.5 MB | 1.2 MB | 0.8 MB | 0.6 MB |

- **Storage is unchanged.** Every candidate stores exactly the accepted layout's bytes.
- **The slowest zarr 3.2 workload** is the full-vector point in every column. On that client the
  variable sets' Dask workloads could not run at all, because xarray 2026.4 cannot open their arrays.
- **The manifest** is the icechunk index a reader downloads before its first chunk. It costs about
  21 bytes per object in every layout. ADR 008 D4 gives each zone-year its own manifest, so the last
  row is what a reader's first read downloads (the accepted layout's is 1.5 MB). On the poor link
  that is 8 s for the accepted layout and 31 s for band-split 4.

**Gates.** A gate fails only if its interval clears the threshold. R3 was not applied to the
variable sets, on the owner's ruling that their readers upgrade; their figures are in parentheses.

| | W1 assembly ≤ 1.20× C0 | W2 objects ≤ 32× C0 | R1 128-d point ≤ 1.25× C0 | R2 128-d bulk ≥ 0.9× C0 | R3 zarr 3.2 ≤ 2× C0 |
|---|---|---|---|---|---|
| accepted | 1.19 | 8× | pass | pass | pass |
| band-split 8 | 1.23 | 16× | **fails in one pass of two**: 1.36× [1.34, 1.38] | pass | pass |
| band-split 4 | 1.31 | 32× | **fails**: 1.53× and 1.83× | **fails**: 0.89× and 0.88× | pass |
| 16-band, 4-band inner | 1.30 | 8× | pass | pass | pass |
| variable 4 … 64 | 1.30 | 6× | pass | pass | (2.81×) |
| variable 4, 4, 8, 112 | 1.31 | 4× | pass | pass | (4.74×) |
| variable 8, 8, 112 | 1.23 | 3× | pass | pass | (3.14×) |

**W1 is read against the accepted layout, not the threshold.** Assembly ran through the band-first
prototype writer, which reorders each band-last staged tile as it writes it (§2). That inflates
every arm alike, and puts the accepted layout itself at 1.19×. Over the accepted layout, 8-band
inner chunks add 0.04× C0's worker CPU and 4-band ones add 0.11×. That extra is encoding more,
smaller pieces, and moving the reorder into inference does not remove it. Wall time per run was
level for every layout.

**Variable-width chunks need reader support that most clients lack.**

- **zarr 3.4.1** reads them only with `array.rectilinear_chunks` set, so every reader has to opt
  in; zarr 3.2.1 reads them with the same setting. Appending a year works.
- **xarray** opens them from 2026.9; 2026.4 raises on their chunk sizes.
- **zarrita.js**, the JavaScript Zarr reader a browser would use, reads regular grids only. The
  layouts aimed at phones are the ones a browser cannot read today. `@earthmover/icechunk` lets a
  browser read icechunk, but it still needs a Zarr reader on top.
- **icechunk 2.3** accepts the grid only with every axis listed edge by edge. zarr then requires
  every edge to be a whole number of inner chunks, so the array's extent rounds up to whole inner
  chunks, and the margin is fill.
- **Every array that shares the spatial dimensions must round with it.** Otherwise xarray refuses
  the group ("conflicting sizes for dimension").
- **The library's shard writer reads `.shards`**, which raises on these arrays
  (`shard_writer.shard_pitch`). It would have to read `write_chunk_sizes` instead.
- **On zarr 3.2, a wide variable-width shard behaves like an unsplit one.** Its pieces are fetched
  one after another, which is why a full-vector point takes 2.8–4.7× C0's time.

**If phone readers become a priority, the best-placed change is 16-band shards holding 4-band inner
chunks.**

- **What it keeps:** the accepted layout's eight objects per tile, its manifest, and a regular grid
  that every Zarr reader can read, zarrita.js included. It passes every read gate.
- **What phones gain:** the full 4-band speedup. 4-d reads are 2.8–3.9× faster and 8-d reads
  1.6–2.0× faster.
- **What other readers pay:** a Dask read at default chunks takes 2.1× the accepted layout's time,
  though still 0.42× C0's. Pulls are 7–12% slower. Assembly costs about 0.11× C0's worker CPU more.
- **Why not the variable sets:** they read the same on a phone, with fewer objects. But browsers
  cannot read them, and old clients read full vectors 2.8–4.7× slower.

**Not measured here:** a second, small array holding only the first 8 dimensions in 4-band chunks.
It would give phones the same gain without changing what any other reader fetches, for about 6%
more storage, one more object per tile, and a second array for assembly to write.
