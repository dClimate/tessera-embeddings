# 031 — The v2 store is band-first and splits the band axis at 16

**Status:**

- **Accepted 2026-10-22 (repo owner):** `embeddings` is band-first, and each shard is stored as
  eight 16-band objects (§4). Inference makes the reorder to band-first as it stages each shard.
- **Open:** how phones and browsers are served. There are two equally weighted paths (§4): keep
  16-band chunks and add a small preview store, or cut each 16-band object into 4-band chunks. Path 2
  has to be chosen before the v2 campaign writes; path 1 can be added at any time.
- **Scope:** for v2 stores only, this supersedes D2 of [ADR 008](008-global-store-architecture.md).
  The v1.1 store keeps D2, and targeted updates go on writing it that way.

v2 emits 128 dimensions in Matryoshka order: the first 16, 32 and 64 are each usable on their own as
a shorter embedding. D2 never splits the band axis, so a reader who wants 16 dimensions still
fetches all 128. This records the layout that lets a reader pay only for the prefix it uses. It was
scoped with the scale-test harness in two studies, each a first screen on a laptop (T9) and a main
test on AWS (T10): the 16-band study
([results](../../scripts/scoping/scale_tests/v2_layout_results.md)) and the narrow-band study
([results](../../scripts/scoping/scale_tests/v2_narrow_band_results.md)).

**Shards, objects and chunks.** The store has three levels. zarr calls the middle one a "shard", but
in these docs a shard is the 2,048² block one inference run produces, so this ADR says *object*:

```
shard      2,048 × 2,048 px × 128 bands             one inference run
│
├── object 1   bands 0–15                       one file in S3 (zarr's "shard")
├── …
└── object 8   bands 112–127
     │
     ├── path 1: 16-band chunks, 16 per object
     │     a 4 × 4 grid of 512 × 512 px, each chunk holding all 16 bands      4.2 MB raw
     │
     └── path 2: 4-band chunks, 64 per object
           the same 4 × 4 grid, each square cut into 4 chunks of 4 bands      1.05 MB raw

     + a small index at the end of each object locating every chunk (260 bytes; about 1 KB in path 2)

preview store (path 1 only)   a separate Zarr store: the first 4 or 8 bands of every pixel,
                              one object per shard, in 512 × 512 px × 4-band chunks
```

A chunk is the smallest piece a reader can fetch: one byte-range request within one object.
*Band-last* is v1.1's order, `(time, northing, easting, band)`: each pixel's values sit side by side.
*Band-first* is `(time, band, y, x)`, v2 naming its spatial axes `y` and `x`
([ADR 032](032-v2-stores-name-their-axes-y-and-x.md)): each chunk holds whole image planes, one per
band.

**The layouts measured, by name.** The archives use the names in the last column.

| name here | band order | objects per shard | chunk | archive name |
|---|---|---|---|---|
| v1.1 layout (the control) | band-last | 1 | 256² × 128 bands | `v2_c0` |
| band-last, one object | band-last | 1 | 512² × 16 | `v2_c2_zstd1_s256` |
| band-last, 16-band objects | band-last | 8 | 512² × 16 | `v2_b2_zstd1_s256` |
| band-first, one object | band-first | 1 | 512² × 16 | `v2_c2f_bandfirst_zstd1_s256` |
| **16-band objects, 16-band chunks** (§4, path 1) | band-first | 8 | 512² × 16 | `v2_b2f_bandfirst_zstd1_s256` |
| **16-band objects, 4-band chunks** (§4, path 2) | band-first | 8 | 512² × 4 | `v2_b2f_i4_bandfirst_zstd1_s256` |
| 8-band objects; 4-band objects | band-first | 16; 32 | 512² × 8; × 4 | `v2_b2f8_…`, `v2_b2f4_…` |
| variable 4, 4, 8, 16, 32, 64 | band-first | 6 | 512² × 4 | `v2_m2f_bandfirst_zstd1_s256` |
| variable 4, 4, 8, 112; variable 8, 8, 112 | band-first | 4; 3 | 512² × 4; × 8 | `v2_m2f_x4_…`, `v2_m2f_x3_…` |

---

## 1. Settled in either band order

**The 2,048² shard stays.** One inference run is one shard, and the 4,096-px ingest chunk is set by
Dask's scheduler, so nothing upstream of assembly moves. Only what sits inside the shard changes.

| choice | value | why |
|---|---|---|
| objects | 16 bands each, eight per shard | the smallest prefix v2 was trained on; §6 weighs 8-band, 4-band and variable-width objects |
| chunk, spatially | 512 × 512 px | 256² builds 4× the Dask tasks at default chunks and 204–260 serial requests per 16-band read of 2 × 2 shards on zarr 3.2; 1,024² makes a full vector cost 134 MB |
| chunk, bands | 16 or 4: open between the two paths of §4 | |
| int8 codec | zstd at level 1 | within 3% of the default level's bytes at 0.6× its encode CPU, once bands are split |
| chunk order within an object | zarr's default (Morton) | the other orders moved request counts by under 20%, and an order is not persisted, so it would need library code on every write path |
| `scales` | 256² chunks, as v1.1 | its geometry made no measurable difference; PCodec compresses it about 7× |
| float arrays | PCodec | unchanged |
| readers | zarr ≥ 3.3, Dask chunks matched to objects | zarr 3.3 merges an object's ranges into concurrent requests; Dask's default makes one task per chunk, each refetching the object's index (§6) |

## 2. What was measured in the 16-band study

Eight real v2 shards: four from Iowa (cropland, including a 52%-valid edge shard) and four from Puget
Sound (city, open water, coast and conifer forest). T10 repeated them into a 100-shard store on S3
and read it cold from an r7i.4xlarge in the same region, two passes in opposite order, every layout
in one session. Encode CPU is per shard on a Graviton m7g.4xlarge, the assembly runner's
architecture. All four candidates use 512 × 512 × 16 chunks and zstd at level 1. An "8 × 8 area" is
8 × 8 shards.

| | v1.1 layout | band-last, one object | band-last, 16-band objects | band-first, one object | band-first, 16-band objects |
|---|---|---|---|---|---|
| encode CPU per shard | 3.36 s | 0.83× | 0.81× | 0.75× | 0.74× |
| stored | 38.9 GB | −3% | −3% | **−23%** | **−23%** |
| 16-band read, 8 × 8 area | 20.2 s | 5.4 s | 5.0 s | 5.5 s | **4.9 s** |
| 64-band read, 8 × 8 area | 24.2 s | 20.1 s | 18.0 s | 16.7 s | **14.7 s** |
| **128-band read, 8 × 8 area** | 27.9 s | **48.1 s** | **48.5 s** | 29.4 s | **27.7 s** |
| single pixel, 128 bands, p50, zarr 3.4 | 419 ms | 457 ms | 424 ms | 460 ms | **391 ms** |
| 16-band read through Dask at default chunks | 43.4 s | 11.5 s | 9.0 s | 11.7 s | **8.6 s** |
| single pixel, 128 bands, p50, zarr 3.2 | 472 ms | **1,474 ms** | 414 ms | **1,132 ms** | 420 ms |
| 16-band read, 8 × 8 area, zarr 3.2 | 19.9 s | 13.3 s | 5.0 s | 14.1 s | 5.2 s |
| 128-band read, 8 × 8 area, zarr 3.2 | 30.0 s | 47.3 s | 47.0 s | 29.0 s | **27.8 s** |

**The limits, set before the study,** all relative to the v1.1 layout. A limit fails only if its
95% interval clears the threshold.

| limit | threshold | plan's label |
|---|---|---|
| assembly CPU and wall time per shard | ≤ 1.20× | W1 |
| objects per shard | ≤ 8× (≤ 32× in the narrow-band study) | W2 |
| single-pixel read of all 128 bands | ≤ 1.25× | R1 |
| bulk read speed at 128 bands | ≥ 0.9× | R2 |
| old client (zarr 3.2), slowest read | ≤ 2× | R3 |

**16-band chunks make a 16-band read about 4× faster in either order**, and a 64-band read faster
too.

**Every layout passes the assembly limit.** The library's own assembly code wrote 64 staged Iowa
shards into each layout on Graviton.

- **Band-last:** 0.90–0.91× the v1.1 layout's worker CPU per shard.
- **Band-first:** 1.17×, at 0.94–1.00× its wall time. This was measured with a prototype writer
  that reorders each band-last staged shard as it writes it.
- **Why 1.17× is an upper bound:** a band-first pipeline does that reorder earlier, in inference
  (§3), so assembly will not pay it. How far below 1.17× it lands is unmeasured: encoding alone costs
  0.74× in band-first order, but assembly also reads and checks the staged shards and writes the
  other arrays, none of which the reorder changes.

**Every layout is lossless.** Read back, every array of all 64 assembled shards is identical to the
v1.1 layout's, band-first arrays after reordering.

**In band-last order a 128-band bulk read is 1.5–1.7× slower than the v1.1 layout**, across both
sessions, which fails the bulk-read limit. The bytes on the wire are the same. The time goes into
putting each pixel's 128 values back together from eight 16-band pieces, 16 bytes at a time.
Neither a fetch concurrency of 64 nor band-aligned Dask chunks recovers it; both made every layout
slower, the v1.1 layout included.

**In band-first order the same read is level with the v1.1 layout**, and it moves 23% fewer bytes.
Each piece is a set of whole planes, so putting it back is a few large copies: 0.06 s per 128-band
shard against 0.15 s band-last, on one laptop core.

**One object per shard fails the old-client limit on full vectors, in either order.** zarr 3.2
fetches an object's pieces one after another, so a single-pixel read of 128 bands takes 2.4× the
v1.1 layout's time in band-first order and 3.1× in band-last. Eight 16-band objects pass, because
even zarr 3.2 fetches separate objects concurrently. On zarr 3.2, band-first 16-band objects are
faster than the v1.1 layout on every read.

**Neither limit result is close to the noise.** A separate run took 300 probes per layout in each of
two passes, the order reversed between them. Each pass carries its own 95% bootstrap interval on the
median, in ms:

| single pixel, 128 bands, pass 1 / pass 2 | zarr 3.4 | zarr 3.2 |
|---|---|---|
| v1.1 layout | 442 [425, 452] / 422 [406, 436] | 475 [464, 490] / 387 [376, 394] |
| band-first, 16-band objects | 370 [362, 384] / 400 [391, 409] | 387 [377, 393] / 357 [350, 364] |
| band-first, one object | 443 [438, 453] / 451 [439, 464] | **1,075 [1,045, 1,102] / 1,020 [997, 1,037]** |

Band-first in one object on zarr 3.2 takes 2.3× and 2.6× its pass's v1.1 time, with every interval
clear of 2×. Band-first 16-band objects are faster than the v1.1 layout in every pass on both
clients.

## 3. Band-last against band-first

| | band-last | band-first |
|---|---|---|
| **ingest** | unchanged | unchanged: the same code and settings, so the same performance (below) |
| **inference, inputs** | unchanged | unchanged: the model reads pixels, assembled from per-band mosaics as today |
| **inference, outputs** | unchanged | reordered once per shard on the actor's background writer thread (below) |
| **appending a year** | an end-append on the time axis (ADR 008 D1) | the same: time stays the leading axis, chunked at 1, so a new year's chunks are new keys and no existing chunk moves |
| **encode CPU** | 0.81–0.83× v1.1 | 0.74–0.75× v1.1 |
| **assembly worker CPU** | 0.90–0.91× v1.1 | 1.17× v1.1 with assembly reordering band-last staged shards: an upper bound once inference stages them band-first |
| **storage** | −3% | −23%, band-planar compression for free |
| **reads** | 16 bands about 4× faster than v1.1; 128-band bulk 1.5–1.7× slower | 16 bands about 4× faster; 64 bands 1.4–1.6× faster; 128 bands level with v1.1 |
| **readers' tools** | per-pixel readers (scikit-learn) get `(…, band)` directly | rioxarray refuses `(y, x, band)`; PyTorch convolutions take channels first; AlphaEarth's Source Coop copy is `(time, band, y, x)` |
| **readers who want the other order** | channel-first readers transpose | per-pixel readers transpose (below) |
| **our two stores** | one order | v1.1 and v2 differ. xarray users index by name and see no difference; array-level code must check the dimension names |
| **library code** | a v2 preset, `_layout_band`, docs and pinning tests | about 60 sites in 17 files plus about 200 test lines, all needing both orders (below) |

**Ingest performance is unaffected by either order.** Ingest takes only the spatial constants
(`INNER_PX`, `SHARD_PX`) and the chunk-sizing function from the layout module. The embedding layout
is read only by seeding and assembly. On a band-first prototype, every mosaic array has the same
chunks, objects, codecs, data type and fill as on main, and the same bytes. The one coupling is code
identity: `config/store_layout.py` sits inside ingest's identity closure, so *any* v2 preset moves
the ingest identity once. Moving the chunk-sizing function into its own module ends that. The
prototype's ingest identity was then the same with and without a preset.

**The inference cost of band-first is about a second of background CPU per shard.** The model emits
one row of 128 values per pixel, so a band-first shard has to be reordered once, after the model
runs. Ingest cannot do it: its mosaics are already one array per spectral band, and the embeddings
do not exist until inference. The cheapest place is the staged write, which already runs on the
actor's single-slot writer thread and overlaps the next shard's work. Writing the existing
`(y, x, band)` buffer through a transposed view costs 0.6–1.0 s of one laptop core per shard. It
needs no full-shard copy and no GPU time. Reordering on the GPU or on the main thread instead added
2.5–7.4 s per shard on an L4 host.

**That writer has room, by the Iowa v2 runs' own logs (786 shards).**

- **Upload against shard time:** the staged upload takes 16.6 s (median), against 164–166 s of
  shard time, with the GPU at 100% (median).
- **Where it already waits:** the next shard waits on the upload only on the 5.6% of shards that
  finish fast, adding 0.5 s per shard on average.
- **What the reorder adds:** another 1–2 s on each upload. The next shard can wait at most that much
  longer, so the reorder costs at most 1–2 s per shard, about 1% of shard time. A shard waits only
  when the next one finishes before its upload does, which today is the 5.6% above.

**A reader who wants the other order pays a transpose either way.** Turning one 128-band shard from
`(band, y, x)` into `(y, x, band)` takes 0.17 s with torch on 10 threads, or 1.2 s with numpy on one.
At 16 bands it takes 0.05 s. A channel-first reader of a band-last store pays the same in reverse.

**Band-first touches every stage after inference.** That covers the inference write path and
staging validation, the shard writer and the update merge, the layout helpers, validation and the
registry, and about 11 sites in yield-embeddings (coarsening and validation). Each must handle both
orders, because the v1.1 store stays band-last. Some failures would be silent. `commit_tiles`, for
example, labels a shard by chunk coordinates 1 and 2, which in band-first order are band and `y`.

## 4. Decision

**Accepted: band-first, each shard stored as eight 16-band objects.** `embeddings` is
`(time, band, y, x)`. Each shard's 128 bands are stored as eight objects of 1 × 16 × 2,048 × 2,048,
one per 16-dimension group (zarr's `shards=`), compressed with zstd at level 1. `scales` and the
per-shard arrays keep their v1.1 geometry.

**It is the only layout in the 16-band study that passes every limit, and it leads every read.**

- **16-band read:** 4.9 s against the v1.1 layout's 20.2 s.
- **64-band read:** 14.7 s against 24.2 s.
- **128-band read:** level with the v1.1 layout.
- **Single pixel, 128 bands:** slightly faster than the v1.1 layout, on both clients.
- **Storage:** 23% smaller, at 0.74× the encode CPU.

Each alternative fails a limit. Band-last fails the bulk-read limit in either object form.
Band-first in one object per shard fails the old-client limit on a single-pixel read.

**What it costs:**

- **Library code:** the changes after inference listed in §3, kept order-aware for as long as the
  v1.1 store is updated.
- **Inference:** one reorder per shard on the staging writer thread, at most about 1% of shard time.
- **Objects:** eight embedding objects and writes per shard instead of one.
- **Readers:** a transpose for per-pixel readers who want `(…, band)`.

**The reorder is inference's, as it stages each shard.** Staged shards, the shard writer, updates
and readers all see band-first, so no stage after inference reorders anything.

**Rejected: reordering in assembly instead.** It would leave inference's code untouched, at 1.17×
the v1.1 layout's assembly CPU, above what assembling band-first staged shards will cost. It was
turned down because band-first all the way through is the cleaner pipeline, and inference's part of
it costs at most about 1% of shard time.

### Open: serving phones and browsers, two equally weighted paths

Phones and browsers read 4 or 8 bands over a slow link, where read time follows the bytes fetched
(§6). A 16-band chunk makes them fetch four times what a 4-band read needs. Two paths fix that
equally well. Both keep the eight 16-band objects per shard; they differ in who pays.

- **Path 1: 16-band chunks, plus a preview store.** `embeddings` keeps chunks of
  1 × 16 × 512 × 512. A separate Zarr store holds only the first 4 or 8 bands of every pixel's
  codes, one object per shard, in 512 × 512 × 4-band chunks. It needs no `scales`: a preview renders
  codes, and cosine similarity ignores the per-pixel scale.
- **Path 2: 4-band chunks.** `embeddings` gets chunks of 1 × 4 × 512 × 512 inside the same
  16-band objects. One store, which every reader uses as it is.

| | path 1: 16-band chunks, plus a preview store | path 2: 4-band chunks |
|---|---|---|
| phone and browser, 4 bands | about 4× faster than 16-band chunks, inferred from bytes fetched: the preview store was not built | 2.8–3.9× faster, measured |
| phone and browser, 8 bands | about 2× faster with an 8-band preview, no gain with a 4-band one; inferred | 1.6–2.0× faster, measured |
| computer in the same region, direct reads | unchanged: the fastest layout measured | within 1.13× of 16-band chunks |
| Dask at default chunks | unchanged: 0.20× the v1.1 layout's time at 16 bands | 2.1–2.2× 16-band chunks' time at 16 bands (0.42× the v1.1 layout's), 2.7× at 128 |
| Dask with chunks matched to objects | unchanged, and twice as fast as at default chunks | 1.10× 16-band chunks' time at 16 bands, 1.12× at 128, measured |
| storage | +3% (4-band preview) or +6% (8-band) | unchanged |
| stores to write and keep in step | two: every write and targeted update writes both | one |
| assembly CPU | a few percent more for the preview; not measured | +9% over 16-band chunks, measured; about $30 across a campaign of v1.1's 3.23 million shards |
| ingest and inference | unchanged | unchanged |
| what a reader must know | where the preview store is | nothing |
| browsers | the preview store can be plain Zarr, which zarrita.js reads over HTTP with no icechunk index | zarrita.js reads the regular grid; the main store is icechunk, so a browser also needs icechunk's JavaScript reader |
| when it must be decided | any time: the preview store can be built after the campaign from the stored embeddings, with no inference | before the v2 campaign writes; changing chunks afterwards re-compresses every shard |

- **Choose path 1** if bulk-analysis and Dask readers are the main audience and a second store is
  acceptable. It also defers the phone decision past the campaign.
- **Choose path 2** if one store that every reader uses as it is matters more, browsers included,
  and about 9% more assembly CPU, bulk reads up to 13% slower, and Dask reads about 10% slower with
  matched chunks are acceptable. The repo owner ruled
  on 2026-10-28 that assembly has the spare capacity for that CPU.

**Neither path touches ingest or inference.** Inference stages each shard uncompressed in its own
fixed geometry, 256 × 256 px pieces holding all 128 bands (`ZarrWriter.write_chunk`), so the store's
chunk width never reaches it. Assembly is where chunks are made and compressed, so it carries
path 2's extra CPU, and path 1's preview store if assembly writes it.

**Not taken forward:** 8-band and 4-band objects fail the single-pixel limit, 4-band objects also
fail the bulk-read limit, and both double or quadruple the store index (§6). The variable-width sets
read no faster than path 2 and need reader support most clients lack (§6). 16-band objects with
8-band chunks, a gentler path 2, were measured only on the laptop.

## 5. Consequences

**Either path:**

- **A v2 preset** in `config/store_layout.py`, selected by model version for the global and
  single-area paths. `INNER_PX` stays 256, so the land mask's `chunk_live_256` stays valid.
- **Inference stages v2 shards band-first.** `ZarrWriter.write_chunk` writes the actor's
  `(y, x, band)` buffer through a transposed view on the writer thread. Staging validation and
  `StagedShardSource` read `(band, y, x)`, and the shard writer and the holes merge index band-first
  arrays as `(time, band, y, x)`.
- **The chunk-sizing function moves out of `store_layout.py` first**, as its own change at a store
  boundary. It moves the ingest identity once, and keeps later layout changes from moving it.
- **`_layout_band` takes the band count from the object, not the chunk.** Otherwise a v2 store is
  stamped `geoemb:dimensions=16`, or 4 in path 2.
- **Reader-facing metadata:** a root attribute naming the valid prefixes (16, 32, 64, 128), and
  reader-guide advice on zarr ≥ 3.3 and on Dask chunks matched to objects
  (`chunks={"band": 16, "y": 2048, "x": 2048}`).
- **Eight objects per shard mean 8× the embedding objects, index entries and writes.** The busiest
  v1.1 zone-year wrote 9,132 shards, so a v2 zone-year holds at most about 140,000 index entries
  across all eight arrays. That is far below D4's 10⁶ split threshold, and about 56 MB of commit
  memory at 400 bytes per entry.
- **Band order is read from each array's dimension names**, never inferred from the model or the
  store. Code that indexes `embeddings` asks where `band` sits in its `dimension_names`, so one path
  reads both the band-last v1.1 store and a band-first v2 one.
- **The preset follows the store's model**, chosen the way the checkpoint and the worker image
  already are: from the model's `ModelFamily` record, and for a global store from the model it was
  seeded for.

**Path 1 only:**

- **A preview store and its writer.** Assembly, or a separate pass over the stored embeddings, writes
  each shard's first 4 or 8 bands into it. Its location, band count and format (icechunk or plain
  Zarr) are fixed before it is first written.
- **Targeted updates rewrite a shard's preview with the shard**, or the two stores drift apart.
- **The reader guide and a root attribute point to the preview store.**
- **One harness run first**, to measure its assembly cost and phone reads before readers rely on it.

**Path 2 only:**

- **The preset's chunk is 1 × 4 × 512 × 512**, in place before the v2 campaign writes.
- **Each object's own index grows** from 260 bytes to about 1 KB: 64 chunks instead of 16.

## 6. Narrower and variable-width band chunks

**Status:** measured 2026-10-27 and 2026-10-28 at the repo owner's request. It is the evidence behind
§4's two paths. The runs are archived in
[v2_narrow_band_results.md](../../scripts/scoping/scale_tests/v2_narrow_band_results.md).

**Narrower band chunks make a phone's reads 2–4× faster.** On a phone's link, time follows bytes,
and a 4-band read of 4-band chunks fetches a quarter of what a 16-band chunk costs. Every candidate
slows Dask at its default chunks, though Dask chunks matched to objects recover nearly all of it.
Some also slow full-vector reads, and a browser's Zarr client cannot read the variable-width ones.

**The mobile reader, as the owner defined it for this study:**

- **What it reads:** map-view previews, the first three or four dimensions shown as an image, for a
  256-px web-map tile or a phone viewport at native 10 m. It also runs similarity over a small area
  at 4 or 8 dimensions.
- **How it reads:** Zarr straight from S3, as a browser or app does. That means one request per
  chunk, no merged requests, about six connections, and a phone's link.

**The candidates** are all band-first, compressed with zstd at level 1, and measured against the v1.1
layout and 16-band objects with 16-band chunks in the same sessions. The object limit was raised to
32× so that 4-band objects could compete.

| | bands per object | chunk | objects per shard | chunk grid |
|---|---|---|---|---|
| 16-band objects, 16-band chunks | 16 | 512² × 16 | 8 | regular |
| 8-band objects | 8 | 512² × 8 | 16 | regular |
| 4-band objects | 4 | 512² × 4 | 32 | regular |
| 16-band objects, 4-band chunks | 16 | 512² × 4 | 8 | regular |
| variable 4, 4, 8, 16, 32, 64 | as listed | 512² × 4 | 6 | variable |
| variable 4, 4, 8, 112 | as listed | 512² × 4 | 4 | variable |
| variable 8, 8, 112 | as listed | 512² × 8 | 3 | variable |

zarr requires one chunk shape for the whole array, and it must divide every object. So a variable
set that includes a 4-band object has 4-band chunks throughout, and its 64-band object holds 16
chunks per 512² square. The laptop screen also measured three things that did not reach AWS:

- **256² chunks.** They cut a map tile's bytes by another 4×, but they triple a viewport's
  requests, store 5–7% more, and cost 1.3–1.45× the encode CPU.
- **16-band objects with 8-band chunks.** They read like 8-band objects in fewer objects, and the
  4-band version went forward instead.
- **Chunk orders within an object.** AWS kept one, the variable 4, 4, 8, 16, 32, 64 set with each
  pixel's pieces side by side. On zarr 3.4 it needs 12 requests for a single-pixel read of 128 bands
  where the default order needs 23, but it read no faster.

**What they buy a mobile reader.** These are cold reads through the direct-read client on two links
shaped with `tc netem`: *good* (40 ms round trip, 20 Mbit/s down) and *poor* (Lighthouse's mobile
profile, 150 ms and 1.6 Mbit/s). Layouts with the same chunk width read alike, within about 5% of
one another on the poor link, so they share a column below:

- **8-band chunks:** 8-band objects, and variable 8, 8, 112.
- **4-band chunks:** the other four candidates.

Each cell is good-link seconds / poor-link seconds, with the MB fetched in brackets.

| | 16-band chunks | 8-band chunks | 4-band chunks |
|---|---|---|---|
| 256-px map tile, 4 bands | 1.7 / 17 s (3.3 MB) | 1.0 / 9.2 s (1.7 MB) | 0.6 / 4.7 s (0.8 MB) |
| 256-px map tile, 8 bands | 1.6 / 18 s (3.3 MB) | 1.0 / 9.3 s (1.7 MB) | 0.9 / 9.1 s (1.7 MB) |
| 1,024 × 2,048 viewport, 4 bands | 8.3 / 103 s (19.7 MB) | 4.4 / 52 s (9.9 MB) | 2.3 / 26 s (5.0 MB) |
| 1,024 × 2,048 viewport, 8 bands | 8.5 / 102 s (19.7 MB) | 4.4 / 52 s (9.9 MB) | 4.4 / 52 s (9.9 MB) |

- **Requests barely matter on a phone's link.** On an 8-band viewport the 4-band layouts make up to
  twice the requests of 16-band chunks (20 against 10) and still finish in half the time. The same
  bytes-not-requests rule is why path 1's preview store, which fetches what 4-band chunks fetch,
  should read like them.
- **4-band chunks help a 4-band reader twice as much as 8-band ones.** At 8 bands the two are
  level.
- **The first read** of a session loads the store index. On these 100-shard test stores it took
  0.2–0.3 s on the good link and 0.5–0.8 s on the poor one. At full scale the index is larger (the
  next table).

**What they cost every other reader.** Cold reads from S3 in the same region on zarr 3.4 are given as
a ratio to 16-band objects with 16-band chunks. Encode, assembly and the old client are given against
the v1.1 layout.

| | 8-band objects | 4-band objects | 16-band objects, 4-band chunks | variable 4 … 64 | variable 4, 4, 8, 112 | variable 8, 8, 112 |
|---|---|---|---|---|---|---|
| single pixel, 128 bands, p50 | 1.27 | 1.65 | 0.99 | 0.98 | 1.11 | 1.03 |
| 128-band read, 8 × 8 area | 1.25 | 1.45 | 1.08 | 1.12 | 1.15 | 1.05 |
| 16-band read, 8 × 8 area | 1.08 | 1.31 | 1.07 | 1.21 | 1.21 | 1.10 |
| 16-band read through Dask, default chunks | 1.45 | 2.45 | 2.09 | 2.47 | 2.32 | 1.48 |
| zarr 3.2, slowest read vs v1.1 (16-band chunks: 0.97) | 1.19 | 1.52 | 1.23 | 2.81 | 4.74 | 3.14 |
| assembly worker CPU vs v1.1 (16-band chunks: 1.19) | 1.23 | 1.31 | 1.30 | 1.30 | 1.31 | 1.23 |
| encode CPU per shard vs v1.1, Graviton (16-band chunks: 0.78) | 0.80 | 0.86 | 0.82 | 0.82 | 0.82 | 0.79 |
| embeddings store index, busiest zone-year | 3.1 MB | 6.1 MB | 1.5 MB | 1.2 MB | 0.8 MB | 0.6 MB |

- **Storage is unchanged.** Every candidate stores exactly the bytes of 16-band chunks.
- **The slowest zarr 3.2 read** is the single-pixel read of 128 bands in every column. On that client
  the variable sets' Dask reads could not run at all, because xarray 2026.4 cannot open their arrays.
- **The store index** is icechunk's manifest, which a reader downloads before its first chunk. It
  costs about 21 bytes per object in every layout. ADR 008 D4 gives each zone-year its own index, so
  the last row is what a reader's first read downloads (1.5 MB for 16-band objects of either chunk
  width). On the poor link that is 8 s, and 31 s for 4-band objects.

**Dask's cost follows chunk width, and matching Dask chunks to objects removes it.** At default
chunks xarray gives Dask one task per chunk, which is what `xr.open_zarr` does when Dask is
installed. Each task fetches its own piece and re-reads its object's index, while zarr's direct reads
merge an object's neighbouring pieces into a few large requests. So the cost follows chunk width,
not whether widths vary. Passing `chunks={"band": 16, "y": 2048, "x": 2048}` gives one task per
object; on a variable-width grid the band chunks are each object's width. A follow-up run timed
both, beside a direct read, on the same stores (run `v2n`, phase `tune`). Median of two passes:

| | 16 bands: Dask, default chunks | 16 bands: Dask, one task per object | 16 bands: direct read | 128 bands: Dask, default chunks | 128 bands: Dask, one task per object | 128 bands: direct read |
|---|---|---|---|---|---|---|
| v1.1 layout | 52.9 s | 19.5 s | 20.8 s | 57.4 s | 32.4 s | 26.3 s |
| 16-band objects, 16-band chunks | 10.0 s | **4.8 s** | 5.1 s | 72.3 s | **33.0 s** | 27.7 s |
| 16-band objects, 4-band chunks | 22.4 s | **5.3 s** | 5.4 s | 198.3 s | **36.9 s** | 28.7 s |
| 8-band objects | 14.9 s | 4.8 s | 5.5 s | 114.1 s | 34.7 s | 35.2 s |
| 4-band objects | 24.1 s | 5.3 s | 6.6 s | 205.7 s | 41.5 s | 46.0 s |
| variable 4 … 64 | 23.4 s | 4.9 s | 6.0 s | 204.7 s | 38.3 s | 30.3 s |
| variable 4, 4, 8, 112 | 23.2 s | 5.0 s | 6.0 s | 211.2 s | 39.3 s | 31.8 s |
| variable 8, 8, 112 | 14.3 s | 4.4 s | 5.7 s | 130.7 s | 36.9 s | 28.7 s |

- **Matching Dask chunks to objects removes the chunk-width cost.** At 16 bands every layout's Dask
  read falls to 4.4–5.3 s, level with or faster than its direct read. 16-band objects with 4-band
  chunks take 5.3 s against 4.8 s for 16-band chunks (1.10×), down from 2.24× at default chunks. At
  128 bands they take 36.9 s against 33.0 s (1.12×), down from 2.74×.
- **It halves Dask time for 16-band chunks too:** 10.0 s to 4.8 s at 16 bands, and 72.3 s to 33.0 s
  at 128. At default chunks a 128-band Dask read of 16-band chunks is slower than the v1.1 layout's
  (72.3 s against 57.4 s). So the reader-guide advice (§5) matters on either path.
- **A path 2 reader who matches chunks beats a path 1 reader who doesn't:** 5.3 s against 10.0 s at
  16 bands.
- **At 128 bands, Dask with matched chunks costs 1.2–1.3× a direct read** on the v1.1 layout and on
  every layout with 16-band or variable objects. That is Dask's own overhead, not the layout's.
- **The two passes agree within about 10%** on every matched-chunk and direct read. This session's
  16-band chunks and v1.1 layout read within 12% of the earlier `v2n` read session.

**Limits.** A limit fails only if its interval clears the threshold. The old-client limit was not
applied to the variable sets, on the owner's ruling that their readers upgrade; their figures are in
parentheses.

| | assembly CPU ≤ 1.20× v1.1 | objects ≤ 32× v1.1 | single pixel, 128 bands ≤ 1.25× v1.1 | bulk read ≥ 0.9× v1.1 | zarr 3.2 ≤ 2× v1.1 |
|---|---|---|---|---|---|
| 16-band objects, 16-band chunks | 1.19 | 8× | pass | pass | pass |
| 8-band objects | 1.23 | 16× | **fails in one pass of two**: 1.36× [1.34, 1.38] | pass | pass |
| 4-band objects | 1.31 | 32× | **fails**: 1.53× and 1.83× | **fails**: 0.89× and 0.88× | pass |
| 16-band objects, 4-band chunks | 1.30 | 8× | pass | pass | pass |
| variable 4 … 64 | 1.30 | 6× | pass | pass | (2.81×) |
| variable 4, 4, 8, 112 | 1.31 | 4× | pass | pass | (4.74×) |
| variable 8, 8, 112 | 1.23 | 3× | pass | pass | (3.14×) |

**Assembly is read against 16-band chunks, not the threshold.** Assembly ran through the band-first
prototype writer, which reorders each band-last staged shard as it writes it (§2). That inflates
every layout alike, and puts 16-band chunks themselves at 1.19×. Against them, 8-band chunks add
about 3% of worker CPU and 4-band chunks 9–10%: for path 2, 10.10 s per shard against 9.30 s. That
extra is encoding more, smaller pieces, and moving the reorder into inference does not remove it.
Wall time per run was level for every layout.

**Variable-width chunks need reader support that most clients lack.**

- **zarr 3.4.1** reads them only with `array.rectilinear_chunks` set, so every reader has to opt
  in; zarr 3.2.1 reads them with the same setting. Appending a year works.
- **xarray** opens them from 2026.9; 2026.4 raises on their chunk sizes.
- **zarrita.js**, the JavaScript Zarr reader a browser would use, reads regular grids only. The
  layouts aimed at phones are the ones a browser cannot read today. `@earthmover/icechunk` lets a
  browser read icechunk, but it still needs a Zarr reader on top.
- **icechunk 2.3** accepts the grid only with every axis listed edge by edge. zarr then requires
  every edge to be a whole number of chunks, so the array's extent rounds up to whole chunks, and the
  margin is fill.
- **Every array that shares the spatial dimensions must round with it.** Otherwise xarray refuses
  the group ("conflicting sizes for dimension").
- **The library's shard writer reads `.shards`**, which raises on these arrays
  (`shard_writer.shard_pitch`). It would have to read `write_chunk_sizes` instead.
- **On zarr 3.2, a wide variable-width object behaves like one object per shard.** Its pieces are
  fetched one after another, which is why a single-pixel read of 128 bands takes 2.8–4.7× the v1.1
  layout's time.
- **For readers who opt in, they read like the regular layout with the same chunk width.** On a
  phone the 4-band sets match 16-band objects with 4-band chunks, and 8, 8, 112 matches 8-band
  chunks. In the same region their direct reads take 0.98–1.23× 16-band chunks' time, and path 2
  matches or beats both 4-band sets on every read. Their one edge is a store index 20–60% smaller.

**Not measured:** path 1's preview store, xarray without Dask, and 16-band objects with 8-band chunks
on AWS.
