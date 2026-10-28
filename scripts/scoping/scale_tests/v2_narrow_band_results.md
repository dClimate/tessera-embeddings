# Scale-test report — narrower and variable-width band chunks (T9, T10)

The records behind [ADR 031 §6](../../../context_docs/decisions/031-the-v2-store-chunks-the-band-axis-at-16.md#6-narrower-and-variable-width-band-chunks).
Every run used `--study narrow`. Each run's section is `report.py` output for that run, under its
own provenance. The band-16 study behind the rest of ADR 031 is archived in
[v2_layout_results.md](v2_layout_results.md).

| run | test | host | clients | code |
|---|---|---|---|---|
| `n9iowa`, `n9puget` | T9 | laptop (Apple M3 Max), local stores | zarr 3.4.1 + icechunk 2.3.0, the direct-read client on them, and zarr 3.2.1 + icechunk 2.1.1 | `930bffdb` |
| `v2n` | T10 build, read, verify | r7i.4xlarge, us-west-2 | zarr 3.4.1 + icechunk 2.3.0 + xarray 2026.9.0, and zarr 3.2.1 + icechunk 2.1.1 + xarray 2026.4.0 | `824d5c4b` |
| `v2n` | T10 mobile | r7i.4xlarge, us-west-2 | the direct-read client on zarr 3.4.1 + icechunk 2.3.0 | `60a3c362` |
| `v2n-arm` | T10 build, assemble | m7g.4xlarge (Graviton) | zarr 3.4.1 + icechunk 2.3.0 | `500e5ebf` on `proto/narrow-band-assembly` |
| `v2n-arm` | T10 verify | m7g.4xlarge (Graviton) | zarr 3.4.1 + icechunk 2.3.0 | `0935d8c3` on `proto/narrow-band-assembly` |

The T10 boxes ran a tarball of the commit in the code column, so their records say `git unknown`.
The old client ran the same scale-test code against its own pinned libraries.
`proto/narrow-band-assembly` is a prototype whose shard writer accepts band-first arrays. It writes
each band-last staged tile through a transposed view, a reorder ADR 031 moves into inference, so its
assembly CPU is an upper bound, as in the band-16 study.

**Point reads** are 300 probes per arm in each pass, two passes in opposite arm order, each pass with
its own 95% bootstrap interval on the median. ADR 031 §6's gate calls use those intervals.

**The tiles** are the band-16 study's: four Iowa and four Puget Sound v2 Large tiles, fetched again
from the dev stores. T9 builds 2 × 2 shards per area. T10 cycles all eight into 10 × 10 shards, and
assembles 64 staged Iowa tiles.

**The direct-read client** stands in for a browser or app reading Zarr straight from S3. It runs
zarr 3.4.1 with request merging off, at most 6 requests in flight, and icechunk's splitting of
large GETs off. T9 counts its requests. T10's `mobile` phase times it on two links shaped with
`tc netem` and an `ifb` device: *good* (40 ms round trip, 20 Mbit/s down) and *poor* (150 ms,
1.6 Mbit/s). Mobile reads skip `scales`: a preview renders the codes, and cosine similarity ignores
the per-pixel scale.

**Arm names.** Every arm except C0 is band-first (`bandfirst`, `(time, band, y, x)`), compressed
with zstd at level 1 (`zstd1`), with `scales` at 256² inner chunks (`s256`), in 2,048² spatial
shards.

| arm | band axis, per shard | inner chunk | embedding objects per tile |
|---|---|---|---|
| `v2_c0`, the control | 128 bands, band-last: v1.1's geometry | 256² × 128 | 1 |
| `v2_b2f`, accepted (ADR 031 §4) | 16 | 512² × 16 | 8 |
| `v2_b2f8`, `v2_b2f4` | 8; 4 | 512² × 8; × 4 | 16; 32 |
| `v2_b1f8`, `v2_b1f4` (T9 only) | 8; 4 | 256² × 8; × 4 | 16; 32 |
| `v2_b2f_i8`, `v2_b2f_i4` | 16 | 512² × 8; × 4 | 8 |
| `v2_m2f` | variable: 4, 4, 8, 16, 32, 64 | 512² × 4 | 6 |
| `v2_m1f` (T9 only) | variable: 4, 4, 8, 16, 32, 64 | 256² × 4 | 6 |
| `v2_m2f_x4` | variable: 4, 4, 8, 112 | 512² × 4 | 4 |
| `v2_m2f_x3` | variable: 8, 8, 112 | 512² × 8 | 3 |

`lexicographic` and `colexicographic` arms write a shard's inner chunks in that order instead of
zarr's default Morton order. Readers find chunks through the shard index, so any order reads back;
the order only changes which pieces sit side by side for a merged request.

## T9 — Iowa (`n9iowa`)

- backend: **local**, scale: **bench**
- icechunk **2.1.1**, zarr **3.2.1**, git `930bffdb`: requests_zarr3.2
- icechunk **2.3.0**, zarr **3.4.1**, git `930bffdb`: build, dask_zarr3.4, requests_direct_zarr3.4, requests_zarr3.4
- rows: 9377

> Contention/latency numbers are only load-bearing on `--backend s3`.

### T9 — v2 layout arms

| arm | embeddings MB | vs C0 | vs B2f | encode CPU s/shard |
|---|---|---|---|---|
| v2_b1f4_bandfirst_zstd1_s256 | 1101 | 0.82 | 1.05 | 1.070 |
| v2_b1f8_bandfirst_zstd1_s256 | 1128 | 0.84 | 1.07 | 0.771 |
| v2_b2f4_bandfirst_zstd1_s256 | 1050 | 0.79 | 1.00 | 0.737 |
| v2_b2f8_bandfirst_zstd1_s256 | 1050 | 0.79 | 1.00 | 0.732 |
| v2_b2f_bandfirst_zstd1_s256 | 1050 | 0.79 | 1.00 | 0.741 |
| v2_b2f_i4_bandfirst_zstd1_colexicographic_s256 | 1050 | 0.79 | 1.00 | 0.735 |
| v2_b2f_i4_bandfirst_zstd1_lexicographic_s256 | 1050 | 0.79 | 1.00 | 0.737 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 1050 | 0.79 | 1.00 | 0.737 |
| v2_b2f_i8_bandfirst_zstd1_colexicographic_s256 | 1050 | 0.79 | 1.00 | 0.728 |
| v2_b2f_i8_bandfirst_zstd1_lexicographic_s256 | 1050 | 0.79 | 1.00 | 0.750 |
| v2_b2f_i8_bandfirst_zstd1_s256 | 1050 | 0.79 | 1.00 | 0.736 |
| v2_c0 | 1337 | 1.00 | 1.27 | 0.832 |
| v2_m1f_bandfirst_zstd1_s256 | 1101 | 0.82 | 1.05 | 1.052 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 1050 | 0.79 | 1.00 | 0.693 |
| v2_m2f_bandfirst_zstd1_lexicographic_s256 | 1050 | 0.79 | 1.00 | 0.700 |
| v2_m2f_bandfirst_zstd1_s256 | 1050 | 0.79 | 1.00 | 0.701 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 1050 | 0.79 | 1.00 | 0.669 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 1050 | 0.79 | 1.00 | 0.684 |

#### Requests / MB fetched, direct_zarr3.4

| arm | point 4 | point 8 | maptile 4 | maptile 8 | viewport 4 | viewport 8 | point 16 | point 128 | tile 16 | shard 16 | bulk 16 | bulk 64 | bulk 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_b1f4_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.3 | 1 / 0.0 | 2 / 0.0 | 36 / 6.7 | 72 / 13.5 | 7 / 0.6 | 56 / 4.1 | 64 / 9.9 | 4 / 22.6 | 16 / 167.4 | 64 / 583.6 | 128 / 1101.4 |
| v2_b1f8_bandfirst_zstd1_s256 | 2 / 0.3 | 2 / 0.3 | 1 / 0.0 | 1 / 0.0 | 36 / 13.6 | 36 / 13.6 | 4 / 0.6 | 28 / 4.2 | 32 / 10.0 | 2 / 23.1 | 8 / 168.5 | 32 / 593.6 | 64 / 1127.6 |
| v2_b2f4_bandfirst_zstd1_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 12 / 6.1 | 24 / 12.4 | 7 / 2.3 | 59 / 15.8 | 28 / 10.9 | 4 / 21.1 | 16 / 155.6 | 64 / 560.9 | 128 / 1050.0 |
| v2_b2f8_bandfirst_zstd1_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 4 / 2.3 | 30 / 15.8 | 14 / 10.9 | 2 / 21.1 | 8 / 155.6 | 32 / 560.8 | 64 / 1050.0 |
| v2_b2f_bandfirst_zstd1_s256 | 2 / 2.3 | 2 / 2.3 | 2 / 0.4 | 2 / 0.4 | 12 / 24.5 | 12 / 24.5 | 2 / 2.3 | 15 / 15.8 | 7 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1049.9 |
| v2_b2f_i4_bandfirst_zstd1_colexicographic_s256 | 2 / 0.6 | 3 / 1.2 | 2 / 0.1 | 3 / 0.2 | 12 / 6.1 | 20 / 12.4 | 4 / 2.3 | 35 / 15.8 | 25 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.9 | 32 / 1050.0 |
| v2_b2f_i4_bandfirst_zstd1_lexicographic_s256 | 2 / 0.6 | 3 / 1.2 | 2 / 0.1 | 3 / 0.2 | 12 / 6.1 | 20 / 12.4 | 4 / 2.3 | 35 / 15.8 | 25 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.9 | 32 / 1050.0 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 2 / 0.6 | 3 / 1.2 | 2 / 0.1 | 3 / 0.2 | 12 / 6.1 | 20 / 12.4 | 4 / 2.3 | 35 / 15.8 | 25 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.9 | 32 / 1050.0 |
| v2_b2f_i8_bandfirst_zstd1_colexicographic_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 3 / 2.3 | 22 / 15.8 | 13 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1050.0 |
| v2_b2f_i8_bandfirst_zstd1_lexicographic_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 3 / 2.3 | 22 / 15.8 | 13 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1050.0 |
| v2_b2f_i8_bandfirst_zstd1_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 3 / 2.3 | 22 / 15.8 | 13 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1050.0 |
| v2_c0 | 2 / 5.0 | 2 / 5.0 | 1 / 0.0 | 1 / 0.0 | 36 / 217.6 | 36 / 217.6 | 2 / 5.0 | 2 / 5.0 | 16 / 79.9 | 1 / 184.3 | 4 / 1336.7 | 4 / 1336.7 | 4 / 1336.7 |
| v2_m1f_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.3 | 1 / 0.0 | 2 / 0.0 | 36 / 6.7 | 72 / 13.5 | 6 / 0.6 | 30 / 4.1 | 63 / 9.9 | 3 / 22.6 | 12 / 167.4 | 20 / 583.6 | 24 / 1101.4 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 12 / 6.1 | 24 / 12.4 | 6 / 2.3 | 33 / 15.8 | 27 / 10.9 | 3 / 21.1 | 12 / 155.6 | 20 / 560.9 | 24 / 1050.0 |
| v2_m2f_bandfirst_zstd1_lexicographic_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 12 / 6.1 | 24 / 12.4 | 6 / 2.3 | 33 / 15.8 | 27 / 10.9 | 3 / 21.1 | 12 / 155.6 | 20 / 560.9 | 24 / 1050.0 |
| v2_m2f_bandfirst_zstd1_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 12 / 6.1 | 24 / 12.4 | 6 / 2.3 | 33 / 15.8 | 27 / 10.9 | 3 / 21.1 | 12 / 155.6 | 20 / 560.9 | 24 / 1050.0 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 4 / 2.3 | 17 / 15.8 | 14 / 10.9 | 2 / 21.1 | 8 / 155.6 | 342 / 560.8 | 12 / 1050.0 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 12 / 6.1 | 24 / 12.4 | 6 / 2.3 | 31 / 15.8 | 27 / 10.9 | 3 / 21.1 | 12 / 155.6 | 676 / 560.9 | 16 / 1050.0 |

#### Requests / MB fetched, zarr3.2

| arm | point 4 | point 8 | maptile 4 | maptile 8 | viewport 4 | viewport 8 | point 16 | point 128 | tile 16 | shard 16 | bulk 16 | bulk 64 | bulk 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_b1f4_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.3 | 1 / 0.0 | 2 / 0.0 | 36 / 6.7 | 72 / 13.5 | 7 / 0.6 | 56 / 4.1 | 64 / 9.9 | 4 / 22.6 | 16 / 167.4 | 64 / 583.6 | 128 / 1101.4 |
| v2_b1f8_bandfirst_zstd1_s256 | 2 / 0.3 | 2 / 0.3 | 1 / 0.0 | 1 / 0.0 | 36 / 13.6 | 36 / 13.6 | 4 / 0.6 | 28 / 4.2 | 32 / 10.0 | 2 / 23.1 | 8 / 168.5 | 32 / 593.6 | 64 / 1127.6 |
| v2_b2f4_bandfirst_zstd1_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 12 / 6.1 | 24 / 12.4 | 7 / 2.3 | 59 / 15.8 | 28 / 10.9 | 4 / 21.1 | 16 / 155.6 | 64 / 560.9 | 128 / 1050.0 |
| v2_b2f8_bandfirst_zstd1_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 4 / 2.3 | 30 / 15.8 | 14 / 10.9 | 2 / 21.1 | 8 / 155.6 | 32 / 560.8 | 64 / 1050.0 |
| v2_b2f_bandfirst_zstd1_s256 | 2 / 2.3 | 2 / 2.3 | 2 / 0.4 | 2 / 0.4 | 12 / 24.5 | 12 / 24.5 | 2 / 2.3 | 15 / 15.8 | 7 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1049.9 |
| v2_b2f_i4_bandfirst_zstd1_colexicographic_s256 | 2 / 0.6 | 3 / 1.2 | 2 / 0.1 | 3 / 0.2 | 12 / 6.1 | 20 / 12.4 | 4 / 2.3 | 35 / 15.8 | 25 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.9 | 32 / 1050.0 |
| v2_b2f_i4_bandfirst_zstd1_lexicographic_s256 | 2 / 0.6 | 3 / 1.2 | 2 / 0.1 | 3 / 0.2 | 12 / 6.1 | 20 / 12.4 | 4 / 2.3 | 35 / 15.8 | 25 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.9 | 32 / 1050.0 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 2 / 0.6 | 3 / 1.2 | 2 / 0.1 | 3 / 0.2 | 12 / 6.1 | 20 / 12.4 | 4 / 2.3 | 35 / 15.8 | 25 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.9 | 32 / 1050.0 |
| v2_b2f_i8_bandfirst_zstd1_colexicographic_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 3 / 2.3 | 22 / 15.8 | 13 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1050.0 |
| v2_b2f_i8_bandfirst_zstd1_lexicographic_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 3 / 2.3 | 22 / 15.8 | 13 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1050.0 |
| v2_b2f_i8_bandfirst_zstd1_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 3 / 2.3 | 22 / 15.8 | 13 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1050.0 |
| v2_c0 | 2 / 5.0 | 2 / 5.0 | 1 / 0.0 | 1 / 0.0 | 36 / 217.6 | 36 / 217.6 | 2 / 5.0 | 2 / 5.0 | 16 / 79.9 | 1 / 184.3 | 4 / 1336.7 | 4 / 1336.7 | 4 / 1336.7 |
| v2_m1f_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.3 | 1 / 0.0 | 2 / 0.0 | 36 / 6.7 | 72 / 13.5 | 6 / 0.6 | 30 / 4.1 | 63 / 9.9 | 3 / 22.6 | 12 / 167.4 | 20 / 583.6 | 24 / 1101.4 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 12 / 6.1 | 24 / 12.4 | 6 / 2.3 | 33 / 15.8 | 27 / 10.9 | 3 / 21.1 | 12 / 155.6 | 20 / 560.9 | 24 / 1050.0 |
| v2_m2f_bandfirst_zstd1_lexicographic_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 12 / 6.1 | 24 / 12.4 | 6 / 2.3 | 33 / 15.8 | 27 / 10.9 | 3 / 21.1 | 12 / 155.6 | 20 / 560.9 | 24 / 1050.0 |
| v2_m2f_bandfirst_zstd1_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 12 / 6.1 | 24 / 12.4 | 6 / 2.3 | 33 / 15.8 | 27 / 10.9 | 3 / 21.1 | 12 / 155.6 | 20 / 560.9 | 24 / 1050.0 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 4 / 2.3 | 17 / 15.8 | 14 / 10.9 | 2 / 21.1 | 8 / 155.6 | 342 / 560.8 | 12 / 1050.0 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 12 / 6.1 | 24 / 12.4 | 6 / 2.3 | 31 / 15.8 | 27 / 10.9 | 3 / 21.1 | 12 / 155.6 | 676 / 560.9 | 16 / 1050.0 |

#### Requests / MB fetched, zarr3.4

| arm | point 4 | point 8 | maptile 4 | maptile 8 | viewport 4 | viewport 8 | point 16 | point 128 | tile 16 | shard 16 | bulk 16 | bulk 64 | bulk 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_b1f4_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.3 | 1 / 0.0 | 2 / 0.0 | 8 / 10.2 | 16 / 20.7 | 7 / 0.6 | 56 / 4.1 | 8 / 17.6 | 4 / 22.6 | 16 / 167.4 | 64 / 583.6 | 128 / 1101.4 |
| v2_b1f8_bandfirst_zstd1_s256 | 2 / 0.3 | 2 / 0.3 | 1 / 0.0 | 1 / 0.0 | 12 / 13.6 | 12 / 13.6 | 4 / 0.6 | 28 / 4.2 | 8 / 11.1 | 2 / 23.1 | 8 / 168.5 | 32 / 593.6 | 64 / 1127.6 |
| v2_b2f4_bandfirst_zstd1_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 8 / 9.4 | 16 / 19.1 | 7 / 2.3 | 59 / 15.8 | 8 / 17.9 | 4 / 21.1 | 16 / 155.6 | 64 / 560.9 | 128 / 1050.0 |
| v2_b2f8_bandfirst_zstd1_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 4 / 2.3 | 30 / 15.8 | 8 / 11.7 | 2 / 21.1 | 8 / 155.6 | 32 / 560.8 | 64 / 1050.0 |
| v2_b2f_bandfirst_zstd1_s256 | 2 / 2.3 | 2 / 2.3 | 2 / 0.4 | 2 / 0.4 | 12 / 24.5 | 12 / 24.5 | 2 / 2.3 | 15 / 15.8 | 4 / 11.7 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1049.9 |
| v2_b2f_i4_bandfirst_zstd1_colexicographic_s256 | 2 / 0.6 | 2 / 1.2 | 2 / 0.1 | 2 / 0.2 | 12 / 6.1 | 12 / 12.4 | 2 / 2.3 | 15 / 15.8 | 4 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.9 | 32 / 1050.0 |
| v2_b2f_i4_bandfirst_zstd1_lexicographic_s256 | 2 / 0.6 | 3 / 1.2 | 2 / 0.1 | 3 / 0.2 | 8 / 6.1 | 12 / 12.4 | 4 / 2.3 | 35 / 15.8 | 5 / 11.4 | 1 / 21.1 | 4 / 155.6 | 16 / 560.9 | 32 / 1050.0 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 2 / 0.6 | 2 / 1.2 | 2 / 0.1 | 2 / 0.2 | 12 / 6.1 | 12 / 12.4 | 3 / 2.4 | 21 / 16.0 | 7 / 11.4 | 1 / 21.1 | 4 / 155.6 | 16 / 560.9 | 32 / 1050.0 |
| v2_b2f_i8_bandfirst_zstd1_colexicographic_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 2 / 2.3 | 15 / 15.8 | 4 / 10.9 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1050.0 |
| v2_b2f_i8_bandfirst_zstd1_lexicographic_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 8 / 12.4 | 8 / 12.4 | 3 / 2.3 | 22 / 15.8 | 3 / 11.4 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1050.0 |
| v2_b2f_i8_bandfirst_zstd1_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 2 / 2.3 | 15 / 15.8 | 4 / 11.7 | 1 / 21.1 | 4 / 155.6 | 16 / 560.8 | 32 / 1050.0 |
| v2_c0 | 2 / 5.0 | 2 / 5.0 | 1 / 0.0 | 1 / 0.0 | 19 / 217.6 | 19 / 217.6 | 2 / 5.0 | 2 / 5.0 | 8 / 79.9 | 1 / 184.3 | 4 / 1336.7 | 4 / 1336.7 | 4 / 1336.7 |
| v2_m1f_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.3 | 1 / 0.0 | 2 / 0.0 | 8 / 10.2 | 16 / 20.7 | 5 / 0.6 | 16 / 6.0 | 8 / 14.3 | 3 / 22.6 | 12 / 167.4 | 20 / 583.6 | 24 / 1101.4 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 11 / 7.0 | 22 / 14.2 | 6 / 2.3 | 11 / 15.8 | 8 / 14.1 | 3 / 21.1 | 12 / 155.6 | 20 / 560.9 | 24 / 1050.0 |
| v2_m2f_bandfirst_zstd1_lexicographic_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 8 / 6.1 | 16 / 12.4 | 6 / 2.3 | 33 / 15.8 | 7 / 11.4 | 3 / 21.1 | 12 / 155.6 | 20 / 560.9 | 24 / 1050.0 |
| v2_m2f_bandfirst_zstd1_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 8 / 9.4 | 16 / 19.1 | 6 / 2.3 | 20 / 16.0 | 8 / 14.9 | 3 / 21.1 | 12 / 155.6 | 20 / 560.9 | 24 / 1050.0 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 2 / 1.2 | 2 / 1.2 | 2 / 0.2 | 2 / 0.2 | 12 / 12.4 | 12 / 12.4 | 4 / 2.3 | 10 / 16.0 | 8 / 11.7 | 2 / 21.1 | 8 / 155.6 | 42 / 562.2 | 12 / 1050.0 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 2 / 0.6 | 4 / 1.2 | 2 / 0.1 | 4 / 0.2 | 8 / 9.4 | 16 / 19.1 | 6 / 2.3 | 18 / 16.0 | 8 / 14.9 | 3 / 21.1 | 12 / 155.6 | 43 / 560.9 | 16 / 1050.0 |

#### Dask, bulk 16-d pull, zarr3.4

| arm | tasks default | tasks shard-aligned | requests default | requests shard-aligned |
|---|---|---|---|---|
| v2_b1f4_bandfirst_zstd1_s256 | 9217 | 145 | 1824 | 16 |
| v2_b1f8_bandfirst_zstd1_s256 | 4609 | 73 | 912 | 8 |
| v2_b2f4_bandfirst_zstd1_s256 | 2305 | 145 | 476 | 16 |
| v2_b2f8_bandfirst_zstd1_s256 | 1153 | 73 | 238 | 8 |
| v2_b2f_bandfirst_zstd1_s256 | 577 | 37 | 119 | 4 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 2305 | 145 | 476 | 71 |
| v2_b2f_i8_bandfirst_zstd1_s256 | 1153 | 73 | 238 | 100 |
| v2_c0 | 513 | 9 | 456 | 4 |
| v2_m1f_bandfirst_zstd1_s256 | 9217 | 145 | 1824 | 28 |
| v2_m2f_bandfirst_zstd1_s256 | 2305 | 145 | 476 | 28 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 1153 | 73 | 238 | 8 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 2305 | 145 | 476 | 28 |

## T9 — Puget Sound (`n9puget`)

- backend: **local**, scale: **bench**
- icechunk **2.1.1**, zarr **3.2.1**, git `930bffdb`: requests_zarr3.2
- icechunk **2.3.0**, zarr **3.4.1**, git `930bffdb`: build, dask_zarr3.4, requests_direct_zarr3.4, requests_zarr3.4
- rows: 9377

> Contention/latency numbers are only load-bearing on `--backend s3`.

### T9 — v2 layout arms

| arm | embeddings MB | vs C0 | vs B2f | encode CPU s/shard |
|---|---|---|---|---|
| v2_b1f4_bandfirst_zstd1_s256 | 1450 | 0.82 | 1.05 | 1.088 |
| v2_b1f8_bandfirst_zstd1_s256 | 1467 | 0.83 | 1.06 | 0.874 |
| v2_b2f4_bandfirst_zstd1_s256 | 1380 | 0.78 | 1.00 | 0.833 |
| v2_b2f8_bandfirst_zstd1_s256 | 1380 | 0.78 | 1.00 | 0.828 |
| v2_b2f_bandfirst_zstd1_s256 | 1380 | 0.78 | 1.00 | 0.877 |
| v2_b2f_i4_bandfirst_zstd1_colexicographic_s256 | 1380 | 0.78 | 1.00 | 0.828 |
| v2_b2f_i4_bandfirst_zstd1_lexicographic_s256 | 1380 | 0.78 | 1.00 | 0.837 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 1380 | 0.78 | 1.00 | 0.846 |
| v2_b2f_i8_bandfirst_zstd1_colexicographic_s256 | 1380 | 0.78 | 1.00 | 0.853 |
| v2_b2f_i8_bandfirst_zstd1_lexicographic_s256 | 1380 | 0.78 | 1.00 | 0.853 |
| v2_b2f_i8_bandfirst_zstd1_s256 | 1380 | 0.78 | 1.00 | 0.843 |
| v2_c0 | 1768 | 1.00 | 1.28 | 0.927 |
| v2_m1f_bandfirst_zstd1_s256 | 1450 | 0.82 | 1.05 | 1.074 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 1380 | 0.78 | 1.00 | 0.775 |
| v2_m2f_bandfirst_zstd1_lexicographic_s256 | 1380 | 0.78 | 1.00 | 0.790 |
| v2_m2f_bandfirst_zstd1_s256 | 1380 | 0.78 | 1.00 | 0.804 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 1380 | 0.78 | 1.00 | 0.767 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 1380 | 0.78 | 1.00 | 0.775 |

#### Requests / MB fetched, direct_zarr3.4

| arm | point 4 | point 8 | maptile 4 | maptile 8 | viewport 4 | viewport 8 | point 16 | point 128 | tile 16 | shard 16 | bulk 16 | bulk 64 | bulk 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_b1f4_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.4 | 2 / 0.2 | 4 / 0.5 | 36 / 6.4 | 72 / 12.9 | 8 / 0.9 | 64 / 5.8 | 104 / 17.5 | 4 / 47.1 | 16 / 217.7 | 64 / 756.6 | 128 / 1450.1 |
| v2_b1f8_bandfirst_zstd1_s256 | 2 / 0.4 | 2 / 0.4 | 2 / 0.5 | 2 / 0.5 | 36 / 13.3 | 36 / 13.3 | 4 / 0.9 | 32 / 5.8 | 52 / 18.2 | 2 / 48.5 | 8 / 220.1 | 32 / 765.9 | 64 / 1467.0 |
| v2_b2f4_bandfirst_zstd1_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 8 / 3.2 | 64 / 21.6 | 40 / 24.0 | 4 / 45.3 | 16 / 206.1 | 64 / 729.2 | 128 / 1380.3 |
| v2_b2f8_bandfirst_zstd1_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 4 / 3.2 | 32 / 21.6 | 20 / 24.0 | 2 / 45.3 | 8 / 206.1 | 32 / 729.2 | 64 / 1380.2 |
| v2_b2f_bandfirst_zstd1_s256 | 2 / 3.2 | 2 / 3.2 | 2 / 3.3 | 2 / 3.3 | 12 / 23.9 | 12 / 23.9 | 2 / 3.2 | 16 / 21.6 | 10 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2f_i4_bandfirst_zstd1_colexicographic_s256 | 2 / 0.8 | 3 / 1.6 | 2 / 0.9 | 3 / 1.7 | 12 / 6.1 | 20 / 12.2 | 5 / 3.2 | 40 / 21.6 | 37 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.3 |
| v2_b2f_i4_bandfirst_zstd1_lexicographic_s256 | 2 / 0.8 | 3 / 1.6 | 2 / 0.9 | 3 / 1.7 | 12 / 6.1 | 20 / 12.2 | 5 / 3.2 | 40 / 21.6 | 37 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.3 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 2 / 0.8 | 3 / 1.6 | 2 / 0.9 | 3 / 1.7 | 12 / 6.1 | 20 / 12.2 | 5 / 3.2 | 40 / 21.6 | 37 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.3 |
| v2_b2f_i8_bandfirst_zstd1_colexicographic_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 3 / 3.2 | 24 / 21.6 | 19 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2f_i8_bandfirst_zstd1_lexicographic_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 3 / 3.2 | 24 / 21.6 | 19 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2f_i8_bandfirst_zstd1_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 3 / 3.2 | 24 / 21.6 | 19 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_c0 | 2 / 6.9 | 2 / 6.9 | 2 / 7.1 | 2 / 7.1 | 36 / 220.5 | 36 / 220.5 | 2 / 6.9 | 2 / 6.9 | 26 / 169.5 | 1 / 427.0 | 4 / 1767.9 | 4 / 1767.9 | 4 / 1767.9 |
| v2_m1f_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.4 | 2 / 0.2 | 4 / 0.5 | 36 / 6.4 | 72 / 12.9 | 7 / 0.9 | 38 / 5.8 | 103 / 17.5 | 3 / 47.1 | 12 / 217.7 | 20 / 756.6 | 24 / 1450.1 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 7 / 3.2 | 38 / 21.6 | 39 / 24.0 | 3 / 45.3 | 12 / 206.1 | 20 / 729.2 | 24 / 1380.3 |
| v2_m2f_bandfirst_zstd1_lexicographic_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 7 / 3.2 | 38 / 21.6 | 39 / 24.0 | 3 / 45.3 | 12 / 206.1 | 20 / 729.2 | 24 / 1380.3 |
| v2_m2f_bandfirst_zstd1_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 7 / 3.2 | 38 / 21.6 | 39 / 24.0 | 3 / 45.3 | 12 / 206.1 | 20 / 729.2 | 24 / 1380.3 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 4 / 3.2 | 19 / 21.6 | 20 / 24.0 | 2 / 45.3 | 8 / 206.1 | 396 / 729.2 | 12 / 1380.2 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 7 / 3.2 | 36 / 21.6 | 39 / 24.0 | 3 / 45.3 | 12 / 206.1 | 784 / 729.2 | 16 / 1380.3 |

#### Requests / MB fetched, zarr3.2

| arm | point 4 | point 8 | maptile 4 | maptile 8 | viewport 4 | viewport 8 | point 16 | point 128 | tile 16 | shard 16 | bulk 16 | bulk 64 | bulk 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_b1f4_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.4 | 2 / 0.2 | 4 / 0.5 | 36 / 6.4 | 72 / 12.9 | 8 / 0.9 | 64 / 5.8 | 104 / 17.5 | 4 / 47.1 | 16 / 217.7 | 64 / 756.6 | 128 / 1450.1 |
| v2_b1f8_bandfirst_zstd1_s256 | 2 / 0.4 | 2 / 0.4 | 2 / 0.5 | 2 / 0.5 | 36 / 13.3 | 36 / 13.3 | 4 / 0.9 | 32 / 5.8 | 52 / 18.2 | 2 / 48.5 | 8 / 220.1 | 32 / 765.9 | 64 / 1467.0 |
| v2_b2f4_bandfirst_zstd1_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 8 / 3.2 | 64 / 21.6 | 40 / 24.0 | 4 / 45.3 | 16 / 206.1 | 64 / 729.2 | 128 / 1380.3 |
| v2_b2f8_bandfirst_zstd1_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 4 / 3.2 | 32 / 21.6 | 20 / 24.0 | 2 / 45.3 | 8 / 206.1 | 32 / 729.2 | 64 / 1380.2 |
| v2_b2f_bandfirst_zstd1_s256 | 2 / 3.2 | 2 / 3.2 | 2 / 3.3 | 2 / 3.3 | 12 / 23.9 | 12 / 23.9 | 2 / 3.2 | 16 / 21.6 | 10 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2f_i4_bandfirst_zstd1_colexicographic_s256 | 2 / 0.8 | 3 / 1.6 | 2 / 0.9 | 3 / 1.7 | 12 / 6.1 | 20 / 12.2 | 5 / 3.2 | 40 / 21.6 | 37 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.3 |
| v2_b2f_i4_bandfirst_zstd1_lexicographic_s256 | 2 / 0.8 | 3 / 1.6 | 2 / 0.9 | 3 / 1.7 | 12 / 6.1 | 20 / 12.2 | 5 / 3.2 | 40 / 21.6 | 37 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.3 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 2 / 0.8 | 3 / 1.6 | 2 / 0.9 | 3 / 1.7 | 12 / 6.1 | 20 / 12.2 | 5 / 3.2 | 40 / 21.6 | 37 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.3 |
| v2_b2f_i8_bandfirst_zstd1_colexicographic_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 3 / 3.2 | 24 / 21.6 | 19 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2f_i8_bandfirst_zstd1_lexicographic_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 3 / 3.2 | 24 / 21.6 | 19 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2f_i8_bandfirst_zstd1_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 3 / 3.2 | 24 / 21.6 | 19 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_c0 | 2 / 6.9 | 2 / 6.9 | 2 / 7.1 | 2 / 7.1 | 36 / 220.5 | 36 / 220.5 | 2 / 6.9 | 2 / 6.9 | 26 / 169.5 | 1 / 427.0 | 4 / 1767.9 | 4 / 1767.9 | 4 / 1767.9 |
| v2_m1f_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.4 | 2 / 0.2 | 4 / 0.5 | 36 / 6.4 | 72 / 12.9 | 7 / 0.9 | 38 / 5.8 | 103 / 17.5 | 3 / 47.1 | 12 / 217.7 | 20 / 756.6 | 24 / 1450.1 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 7 / 3.2 | 38 / 21.6 | 39 / 24.0 | 3 / 45.3 | 12 / 206.1 | 20 / 729.2 | 24 / 1380.3 |
| v2_m2f_bandfirst_zstd1_lexicographic_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 7 / 3.2 | 38 / 21.6 | 39 / 24.0 | 3 / 45.3 | 12 / 206.1 | 20 / 729.2 | 24 / 1380.3 |
| v2_m2f_bandfirst_zstd1_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 7 / 3.2 | 38 / 21.6 | 39 / 24.0 | 3 / 45.3 | 12 / 206.1 | 20 / 729.2 | 24 / 1380.3 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 4 / 3.2 | 19 / 21.6 | 20 / 24.0 | 2 / 45.3 | 8 / 206.1 | 396 / 729.2 | 12 / 1380.2 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 7 / 3.2 | 36 / 21.6 | 39 / 24.0 | 3 / 45.3 | 12 / 206.1 | 784 / 729.2 | 16 / 1380.3 |

#### Requests / MB fetched, zarr3.4

| arm | point 4 | point 8 | maptile 4 | maptile 8 | viewport 4 | viewport 8 | point 16 | point 128 | tile 16 | shard 16 | bulk 16 | bulk 64 | bulk 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_b1f4_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.4 | 2 / 0.2 | 4 / 0.5 | 8 / 9.7 | 16 / 19.7 | 8 / 0.9 | 64 / 5.8 | 13 / 27.6 | 4 / 47.1 | 16 / 217.7 | 64 / 756.6 | 128 / 1450.1 |
| v2_b1f8_bandfirst_zstd1_s256 | 2 / 0.4 | 2 / 0.4 | 2 / 0.5 | 2 / 0.5 | 12 / 13.3 | 12 / 13.3 | 4 / 0.9 | 32 / 5.8 | 10 / 22.6 | 2 / 48.5 | 8 / 220.1 | 32 / 765.9 | 64 / 1467.0 |
| v2_b2f4_bandfirst_zstd1_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 8 / 9.4 | 16 / 18.6 | 8 / 3.2 | 64 / 21.6 | 12 / 29.9 | 4 / 45.3 | 16 / 206.1 | 64 / 729.2 | 128 / 1380.3 |
| v2_b2f8_bandfirst_zstd1_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 4 / 3.2 | 32 / 21.6 | 10 / 24.0 | 2 / 45.3 | 8 / 206.1 | 32 / 729.2 | 64 / 1380.2 |
| v2_b2f_bandfirst_zstd1_s256 | 2 / 3.2 | 2 / 3.2 | 2 / 3.3 | 2 / 3.3 | 12 / 23.9 | 12 / 23.9 | 2 / 3.2 | 16 / 21.6 | 5 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2f_i4_bandfirst_zstd1_colexicographic_s256 | 2 / 0.8 | 2 / 1.6 | 2 / 0.9 | 2 / 1.7 | 12 / 6.1 | 12 / 12.2 | 2 / 3.2 | 16 / 21.6 | 4 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.3 |
| v2_b2f_i4_bandfirst_zstd1_lexicographic_s256 | 2 / 0.8 | 3 / 1.6 | 2 / 0.9 | 3 / 1.7 | 8 / 6.1 | 12 / 12.2 | 5 / 3.2 | 40 / 21.6 | 5 / 30.6 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.3 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 2 / 0.8 | 2 / 1.6 | 2 / 0.9 | 2 / 1.7 | 12 / 6.1 | 12 / 12.2 | 3 / 3.2 | 24 / 21.6 | 9 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.3 |
| v2_b2f_i8_bandfirst_zstd1_colexicographic_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 2 / 3.2 | 16 / 21.6 | 4 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2f_i8_bandfirst_zstd1_lexicographic_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 8 / 12.2 | 8 / 12.2 | 3 / 3.2 | 24 / 21.6 | 7 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2f_i8_bandfirst_zstd1_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 2 / 3.2 | 16 / 21.6 | 5 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_c0 | 2 / 6.9 | 2 / 6.9 | 2 / 7.1 | 2 / 7.1 | 20 / 220.5 | 20 / 220.5 | 2 / 6.9 | 2 / 6.9 | 16 / 169.5 | 1 / 427.0 | 4 / 1767.9 | 4 / 1767.9 | 4 / 1767.9 |
| v2_m1f_bandfirst_zstd1_s256 | 2 / 0.2 | 4 / 0.4 | 2 / 0.2 | 4 / 0.5 | 8 / 9.7 | 16 / 19.7 | 6 / 0.9 | 20 / 8.2 | 12 / 24.3 | 3 / 47.1 | 12 / 217.7 | 20 / 756.6 | 24 / 1450.1 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 12 / 6.1 | 24 / 12.2 | 6 / 3.2 | 12 / 21.6 | 8 / 27.0 | 3 / 45.3 | 12 / 206.1 | 20 / 729.2 | 24 / 1380.3 |
| v2_m2f_bandfirst_zstd1_lexicographic_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 8 / 6.1 | 16 / 12.2 | 7 / 3.2 | 38 / 21.6 | 7 / 30.6 | 3 / 45.3 | 12 / 206.1 | 20 / 729.2 | 24 / 1380.3 |
| v2_m2f_bandfirst_zstd1_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 8 / 9.4 | 16 / 18.6 | 6 / 3.2 | 23 / 21.6 | 11 / 27.0 | 3 / 45.3 | 12 / 206.1 | 20 / 729.2 | 24 / 1380.3 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 2 / 1.6 | 2 / 1.6 | 2 / 1.7 | 2 / 1.7 | 12 / 12.2 | 12 / 12.2 | 4 / 3.2 | 12 / 21.6 | 10 / 24.0 | 2 / 45.3 | 8 / 206.1 | 51 / 729.2 | 12 / 1380.2 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 2 / 0.8 | 4 / 1.6 | 2 / 0.9 | 4 / 1.7 | 8 / 9.4 | 16 / 18.6 | 6 / 3.2 | 21 / 21.6 | 11 / 27.0 | 3 / 45.3 | 12 / 206.1 | 50 / 729.2 | 16 / 1380.3 |

#### Dask, bulk 16-d pull, zarr3.4

| arm | tasks default | tasks shard-aligned | requests default | requests shard-aligned |
|---|---|---|---|---|
| v2_b1f4_bandfirst_zstd1_s256 | 9217 | 145 | 2048 | 16 |
| v2_b1f8_bandfirst_zstd1_s256 | 4609 | 73 | 1024 | 8 |
| v2_b2f4_bandfirst_zstd1_s256 | 2305 | 145 | 512 | 16 |
| v2_b2f8_bandfirst_zstd1_s256 | 1153 | 73 | 256 | 8 |
| v2_b2f_bandfirst_zstd1_s256 | 577 | 37 | 128 | 4 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 2305 | 145 | 512 | 80 |
| v2_b2f_i8_bandfirst_zstd1_s256 | 1153 | 73 | 256 | 132 |
| v2_c0 | 513 | 9 | 512 | 4 |
| v2_m1f_bandfirst_zstd1_s256 | 9217 | 145 | 2048 | 32 |
| v2_m2f_bandfirst_zstd1_s256 | 2305 | 145 | 512 | 32 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 1153 | 73 | 256 | 8 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 2305 | 145 | 512 | 32 |

## T10 — reads on S3, cold and on a phone link (`v2n`)

- backend: **s3**, scale: **bench**
- icechunk **2.1.1**, zarr **3.2.1**, git `unknown`: read_zarr3.2
- icechunk **2.3.0**, zarr **3.4.1**, git `unknown`: build, mobile_good_zarr3.4, mobile_poor_zarr3.4, read_zarr3.4, verify_t10
- rows: 3206

> Contention/latency numbers are only load-bearing on `--backend s3`.

### T10 — bench store build, encode CPU per shard

| arm | encode CPU s | vs C0 | stored GB | vs C0 | objects | manifest bytes per object |
|---|---|---|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 3.18 | 0.97 | 30.5 | 0.78 | 3300 | 21 |
| v2_b2f8_bandfirst_zstd1_s256 | 2.94 | 0.90 | 30.5 | 0.78 | 1700 | 21 |
| v2_b2f_bandfirst_zstd1_s256 | 2.95 | 0.90 | 30.5 | 0.78 | 900 | 21 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 3.18 | 0.97 | 30.5 | 0.78 | 900 | 21 |
| v2_c0 | 3.27 | 1.00 | 38.9 | 1.00 | 200 | 24 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 3.11 | 0.95 | 30.5 | 0.78 | 700 | 21 |
| v2_m2f_bandfirst_zstd1_s256 | 3.18 | 0.97 | 30.5 | 0.78 | 700 | 21 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 3.03 | 0.93 | 30.5 | 0.78 | 400 | 22 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 3.07 | 0.94 | 30.5 | 0.78 | 500 | 22 |

### T10 — cold reads from S3, zarr3.2 (median of passes; wire MB in brackets)

| arm | point 16 p50 | point 128 p50 | bulk 128 | 8x8 pull 16 | 8x8 pull 64 | 8x8 pull 128 | dask default | maptile 4 | maptile 8 | viewport 4 | viewport 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 356 ms (3) | 751 ms (22) | 3.3 s (1263) | 6.5 s (3142) | 20.8 s (10737) | 39.4 s (20137) | 22.9 s (2982) | 0.2 s (1) | 0.2 s (2) | 0.3 s (5) | 0.3 s (10) |
| v2_b2f8_bandfirst_zstd1_s256 | 371 ms (3) | 586 ms (22) | 2.5 s (1269) | 5.3 s (3133) | 17.3 s (10785) | 32.7 s (20306) | 13.4 s (2973) | 0.3 s (2) | 0.3 s (2) | 0.4 s (10) | 0.4 s (10) |
| v2_b2f_bandfirst_zstd1_s256 | 382 ms (3) | 457 ms (22) | 2.2 s (1273) | 5.0 s (3170) | 14.5 s (10825) | 26.8 s (20217) | 8.6 s (2973) | 0.2 s (3) | 0.2 s (3) | 0.5 s (20) | 0.4 s (20) |
| v2_b2f_i4_bandfirst_zstd1_s256 | 547 ms (3) | 606 ms (22) | 2.3 s (1275) | 5.4 s (3158) | 16.2 s (10740) | 30.1 s (20134) | 19.5 s (2974) | 0.2 s (1) | 0.2 s (2) | 0.4 s (5) | 0.5 s (10) |
| v2_c0 | 503 ms (7) | 493 ms (7) | 2.6 s (1594) | 20.0 s (25805) | 23.5 s (25824) | 27.7 s (25750) | 45.2 s (25559) | 0.3 s (7) | 0.4 s (7) | 1.4 s (170) | 1.5 s (171) |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 334 ms (3) | 1009 ms (22) | 2.4 s (1283) | 6.0 s (3155) | 16.8 s (10727) | 30.8 s (20192) | unreadable | 0.2 s (1) | 0.2 s (2) | 0.3 s (5) | 0.4 s (10) |
| v2_m2f_bandfirst_zstd1_s256 | 411 ms (3) | 1387 ms (22) | 2.4 s (1275) | 6.3 s (3145) | 17.1 s (10857) | 30.7 s (20199) | unreadable | 0.3 s (1) | 0.2 s (2) | 0.3 s (5) | 0.3 s (10) |
| v2_m2f_x3_bandfirst_zstd1_s256 | 337 ms (3) | 1547 ms (22) | 2.5 s (1277) | 5.5 s (3159) | 45.6 s (10780) | 29.5 s (20262) | unreadable | 0.3 s (2) | 0.3 s (2) | 0.4 s (10) | 0.4 s (10) |
| v2_m2f_x4_bandfirst_zstd1_s256 | 404 ms (3) | 2339 ms (22) | 2.7 s (1271) | 6.1 s (3163) | 61.2 s (10773) | 31.7 s (20239) | unreadable | 0.2 s (1) | 0.2 s (2) | 0.3 s (5) | 0.3 s (10) |

> **unreadable** (v2_m2f_bandfirst_zstd1_colexicographic_s256, v2_m2f_bandfirst_zstd1_s256, v2_m2f_x3_bandfirst_zstd1_s256, v2_m2f_x4_bandfirst_zstd1_s256): xarray 2026.4.0: The `chunks` attribute is only defined for arrays using regular chunk grids. This array has a rectilinear chunk grid. Use `read_chunk_sizes` for general access.

#### The same, as a ratio to the accepted layout (v2_b2f_bandfirst_zstd1_s256)

| arm | point 16 p50 | point 128 p50 | bulk 128 | 8x8 pull 16 | 8x8 pull 64 | 8x8 pull 128 | dask default | maptile 4 | maptile 8 | viewport 4 | viewport 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 0.93 | 1.64 | 1.50 | 1.31 | 1.43 | 1.47 | 2.65 | 1.36 | 0.99 | 0.49 | 0.67 |
| v2_b2f8_bandfirst_zstd1_s256 | 0.97 | 1.28 | 1.17 | 1.07 | 1.19 | 1.22 | 1.55 | 1.91 | 1.37 | 0.76 | 0.82 |
| v2_b2f_bandfirst_zstd1_s256 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 1.43 | 1.33 | 1.06 | 1.09 | 1.12 | 1.12 | 2.25 | 0.85 | 0.76 | 0.66 | 1.09 |
| v2_c0 | 1.32 | 1.08 | 1.19 | 4.02 | 1.62 | 1.03 | 5.23 | 1.62 | 1.77 | 2.55 | 3.44 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 0.87 | 2.21 | 1.08 | 1.21 | 1.16 | 1.15 | — | 1.18 | 0.87 | 0.60 | 0.87 |
| v2_m2f_bandfirst_zstd1_s256 | 1.08 | 3.04 | 1.11 | 1.27 | 1.18 | 1.14 | — | 1.45 | 1.02 | 0.54 | 0.71 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 0.88 | 3.39 | 1.15 | 1.11 | 3.15 | 1.10 | — | 1.39 | 1.21 | 0.73 | 0.85 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 1.06 | 5.12 | 1.23 | 1.23 | 4.23 | 1.18 | — | 1.26 | 1.04 | 0.56 | 0.68 |

### T10 — cold reads from S3, zarr3.4 (median of passes; wire MB in brackets)

| arm | point 16 p50 | point 128 p50 | bulk 128 | 8x8 pull 16 | 8x8 pull 64 | 8x8 pull 128 | dask default | maptile 4 | maptile 8 | viewport 4 | viewport 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 366 ms (3) | 810 ms (22) | 2.9 s (1275) | 6.5 s (3138) | 21.1 s (10797) | 41.0 s (20112) | 23.1 s (2984) | 0.3 s (1) | 0.2 s (2) | 0.5 s (8) | 0.3 s (16) |
| v2_b2f8_bandfirst_zstd1_s256 | 363 ms (3) | 620 ms (22) | 2.3 s (1279) | 5.3 s (3147) | 17.3 s (10797) | 35.3 s (20237) | 13.7 s (2973) | 0.3 s (2) | 0.3 s (2) | 0.3 s (10) | 0.3 s (10) |
| v2_b2f_bandfirst_zstd1_s256 | 366 ms (3) | 490 ms (22) | 2.2 s (1267) | 4.9 s (3147) | 14.5 s (10792) | 28.2 s (20199) | 9.4 s (2983) | 0.3 s (3) | 0.2 s (3) | 0.4 s (20) | 0.3 s (20) |
| v2_b2f_i4_bandfirst_zstd1_s256 | 414 ms (3) | 483 ms (22) | 2.4 s (1277) | 5.3 s (3127) | 16.2 s (10812) | 30.4 s (20072) | 19.7 s (2983) | 0.3 s (1) | 0.2 s (2) | 0.4 s (5) | 0.3 s (10) |
| v2_c0 | 508 ms (7) | 489 ms (7) | 2.6 s (1585) | 20.1 s (25593) | 23.4 s (25759) | 28.2 s (25869) | 47.4 s (25580) | 0.3 s (7) | 0.2 s (7) | 0.7 s (169) | 0.9 s (170) |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 392 ms (3) | 536 ms (22) | 2.3 s (1267) | 5.8 s (3143) | 16.7 s (10722) | 31.2 s (20170) | 22.6 s (2988) | 0.2 s (1) | 0.2 s (2) | 0.2 s (5) | 0.3 s (10) |
| v2_m2f_bandfirst_zstd1_s256 | 399 ms (3) | 481 ms (22) | 2.4 s (1276) | 6.0 s (3167) | 16.7 s (10769) | 31.5 s (20244) | 23.3 s (2977) | 0.2 s (1) | 0.2 s (2) | 0.4 s (8) | 0.3 s (15) |
| v2_m2f_x3_bandfirst_zstd1_s256 | 412 ms (3) | 506 ms (22) | 2.5 s (1269) | 5.4 s (3146) | 16.0 s (10756) | 29.6 s (20258) | 13.9 s (2979) | 0.3 s (2) | 0.2 s (2) | 0.3 s (10) | 0.3 s (10) |
| v2_m2f_x4_bandfirst_zstd1_s256 | 403 ms (3) | 543 ms (22) | 2.7 s (1269) | 6.0 s (3133) | 17.4 s (10792) | 32.4 s (20204) | 21.9 s (2985) | 0.3 s (1) | 0.3 s (2) | 0.3 s (8) | 0.3 s (16) |

#### The same, as a ratio to the accepted layout (v2_b2f_bandfirst_zstd1_s256)

| arm | point 16 p50 | point 128 p50 | bulk 128 | 8x8 pull 16 | 8x8 pull 64 | 8x8 pull 128 | dask default | maptile 4 | maptile 8 | viewport 4 | viewport 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 1.00 | 1.65 | 1.32 | 1.31 | 1.46 | 1.45 | 2.45 | 0.90 | 1.22 | 1.40 | 0.98 |
| v2_b2f8_bandfirst_zstd1_s256 | 0.99 | 1.27 | 1.05 | 1.08 | 1.20 | 1.25 | 1.45 | 0.93 | 1.34 | 0.81 | 0.96 |
| v2_b2f_bandfirst_zstd1_s256 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 1.13 | 0.99 | 1.09 | 1.07 | 1.12 | 1.08 | 2.09 | 1.20 | 1.03 | 1.08 | 1.03 |
| v2_c0 | 1.39 | 1.00 | 1.17 | 4.06 | 1.62 | 1.00 | 5.03 | 1.04 | 1.19 | 1.93 | 2.69 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 1.07 | 1.09 | 1.06 | 1.18 | 1.16 | 1.11 | 2.40 | 0.53 | 1.03 | 0.54 | 0.81 |
| v2_m2f_bandfirst_zstd1_s256 | 1.09 | 0.98 | 1.08 | 1.21 | 1.15 | 1.12 | 2.47 | 0.67 | 1.15 | 0.94 | 0.90 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 1.13 | 1.03 | 1.15 | 1.10 | 1.11 | 1.05 | 1.48 | 0.96 | 1.07 | 0.89 | 0.91 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 1.10 | 1.11 | 1.23 | 1.21 | 1.20 | 1.15 | 2.32 | 0.99 | 1.47 | 0.66 | 0.85 |

### T10 — mobile reads, good_zarr3.4 (41 ms, 16.6 Mbit/s measured; wall s / requests / MB, median of passes)

| arm | maptile 4 | maptile 8 | viewport 4 | viewport 8 | first read s |
|---|---|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 0.62 / 2 / 0.8 | 0.91 / 4 / 1.7 | 2.27 / 10 / 5.0 | 4.35 / 20 / 9.9 | 0.29 |
| v2_b2f8_bandfirst_zstd1_s256 | 1.02 / 2 / 1.7 | 1.05 / 2 / 1.7 | 4.38 / 10 / 9.9 | 4.38 / 10 / 9.9 | 0.27 |
| v2_b2f_bandfirst_zstd1_s256 | 1.66 / 2 / 3.3 | 1.58 / 2 / 3.3 | 8.31 / 10 / 19.7 | 8.46 / 10 / 19.7 | 0.23 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 0.60 / 2 / 0.8 | 0.97 / 3 / 1.7 | 2.34 / 10 / 5.0 | 4.33 / 16 / 9.9 | 0.21 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 0.55 / 2 / 0.8 | 0.95 / 4 / 1.7 | 2.27 / 10 / 5.0 | 4.40 / 20 / 9.9 | 0.20 |
| v2_m2f_bandfirst_zstd1_s256 | 0.62 / 2 / 0.8 | 0.96 / 4 / 1.7 | 2.30 / 10 / 5.0 | 4.39 / 20 / 9.9 | 0.22 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 1.06 / 2 / 1.7 | 1.03 / 2 / 1.7 | 4.32 / 10 / 9.9 | 4.37 / 10 / 9.9 | 0.22 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 0.64 / 2 / 0.8 | 0.94 / 4 / 1.7 | 2.54 / 10 / 5.0 | 4.34 / 20 / 9.9 | 0.22 |

### T10 — mobile reads, poor_zarr3.4 (152 ms, 1.6 Mbit/s measured; wall s / requests / MB, median of passes)

| arm | maptile 4 | maptile 8 | viewport 4 | viewport 8 | first read s |
|---|---|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 4.66 / 2 / 0.8 | 9.20 / 4 / 1.7 | 26.36 / 10 / 5.0 | 52.44 / 20 / 9.9 | 0.84 |
| v2_b2f8_bandfirst_zstd1_s256 | 9.23 / 2 / 1.7 | 9.20 / 2 / 1.7 | 51.85 / 10 / 9.9 | 52.32 / 10 / 9.9 | 0.69 |
| v2_b2f_bandfirst_zstd1_s256 | 17.41 / 2 / 3.3 | 17.63 / 2 / 3.3 | 103.17 / 10 / 19.7 | 102.08 / 10 / 19.7 | 0.56 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 4.68 / 2 / 0.8 | 9.21 / 3 / 1.7 | 26.24 / 10 / 5.0 | 52.01 / 16 / 9.9 | 0.57 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 4.60 / 2 / 0.8 | 8.96 / 4 / 1.7 | 26.47 / 10 / 5.0 | 52.71 / 20 / 9.9 | 0.50 |
| v2_m2f_bandfirst_zstd1_s256 | 4.67 / 2 / 0.8 | 9.22 / 4 / 1.7 | 26.32 / 10 / 5.0 | 52.81 / 20 / 9.9 | 0.50 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 9.22 / 2 / 1.7 | 9.38 / 2 / 1.7 | 51.24 / 10 / 9.9 | 51.28 / 10 / 9.9 | 0.49 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 4.84 / 2 / 0.8 | 9.13 / 4 / 1.7 | 26.47 / 10 / 5.0 | 52.41 / 20 / 9.9 | 0.49 |

### T10 — point reads, p50 [95% interval] by pass (0, 1), read_zarr3.2

| arm | point 16 ms | point 128 ms | open s |
|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 313 [306, 320] / 398 [384, 406] | 744 [738, 754] / 758 [746, 767] | 0.14 |
| v2_b2f8_bandfirst_zstd1_s256 | 331 [324, 342] / 412 [400, 421] | 543 [531, 553] / 628 [614, 642] | 0.15 |
| v2_b2f_bandfirst_zstd1_s256 | 339 [326, 351] / 425 [410, 439] | 393 [383, 398] / 520 [508, 537] | 0.15 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 471 [458, 487] / 624 [607, 645] | 622 [609, 635] / 591 [579, 602] | 0.15 |
| v2_c0 | 449 [437, 454] / 558 [538, 576] | 420 [408, 430] / 566 [554, 580] | 0.15 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 342 [334, 354] / 325 [314, 335] | 975 [951, 992] / 1042 [1021, 1069] | 0.15 |
| v2_m2f_bandfirst_zstd1_s256 | 358 [351, 367] / 465 [451, 480] | 1377 [1354, 1400] / 1397 [1375, 1420] | 0.15 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 319 [309, 333] / 356 [350, 368] | 1427 [1406, 1454] / 1668 [1625, 1707] | 0.15 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 363 [349, 372] / 445 [432, 456] | 2206 [2176, 2237] / 2472 [2417, 2552] | 0.14 |

### T10 — point reads, p50 [95% interval] by pass (0, 1), read_zarr3.4

| arm | point 16 ms | point 128 ms | open s |
|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 368 [360, 378] / 364 [353, 375] | 846 [837, 853] / 774 [760, 781] | 0.14 |
| v2_b2f8_bandfirst_zstd1_s256 | 357 [344, 364] / 370 [356, 380] | 665 [651, 681] / 576 [566, 585] | 0.14 |
| v2_b2f_bandfirst_zstd1_s256 | 359 [347, 367] / 372 [361, 383] | 539 [523, 554] / 441 [433, 452] | 0.15 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 445 [433, 458] / 382 [376, 389] | 513 [496, 526] / 452 [438, 462] | 0.15 |
| v2_c0 | 507 [490, 516] / 509 [494, 529] | 555 [538, 566] / 423 [419, 434] | 0.14 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 403 [393, 410] / 381 [369, 393] | 534 [525, 547] / 538 [527, 550] | 0.15 |
| v2_m2f_bandfirst_zstd1_s256 | 423 [412, 432] / 374 [364, 386] | 515 [503, 530] / 447 [434, 457] | 0.15 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 431 [413, 446] / 394 [385, 410] | 541 [528, 554] / 472 [458, 482] | 0.14 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 417 [409, 426] / 388 [377, 397] | 576 [565, 595] / 511 [504, 519] | 0.14 |

### T10 — lossless check against C0

| arm | stores | shards | mismatched array-shards |
|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_b2f8_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_b2f_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_b2f_i4_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | t10 | 16 | 0 |
| v2_m2f_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_m2f_x3_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_m2f_x4_bandfirst_zstd1_s256 | t10 | 16 | 0 |

## T10 — build and assembly on Graviton (`v2n-arm`)

- backend: **s3**, scale: **bench**
- icechunk **2.3.0**, zarr **3.4.1**, git `unknown`
- rows: 135

> Contention/latency numbers are only load-bearing on `--backend s3`.

### T10 — bench store build, encode CPU per shard

| arm | encode CPU s | vs C0 | stored GB | vs C0 | objects | manifest bytes per object |
|---|---|---|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 2.73 | 0.86 | 30.5 | 0.78 | 3300 | 21 |
| v2_b2f8_bandfirst_zstd1_s256 | 2.56 | 0.80 | 30.5 | 0.78 | 1700 | 21 |
| v2_b2f_bandfirst_zstd1_s256 | 2.47 | 0.78 | 30.5 | 0.78 | 900 | 21 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 2.61 | 0.82 | 30.5 | 0.78 | 900 | 21 |
| v2_c0 | 3.18 | 1.00 | 38.9 | 1.00 | 200 | 24 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | 2.62 | 0.82 | 30.5 | 0.78 | 700 | 21 |
| v2_m2f_bandfirst_zstd1_s256 | 2.62 | 0.82 | 30.5 | 0.78 | 700 | 21 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 2.53 | 0.79 | 30.5 | 0.78 | 400 | 22 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 2.60 | 0.82 | 30.5 | 0.78 | 500 | 22 |

### T10 — assembly code: CPU per staged tile, wall per run (median of passes)

| arm | worker CPU s | vs C0 | write CPU s | vs C0 | wall s, whole run | stored GB |
|---|---|---|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | 10.19 | 1.31 | 6.18 | 1.61 | 51 | 20.0 |
| v2_b2f8_bandfirst_zstd1_s256 | 9.61 | 1.23 | 5.54 | 1.44 | 57 | 20.0 |
| v2_b2f_bandfirst_zstd1_s256 | 9.30 | 1.19 | 5.21 | 1.35 | 50 | 20.0 |
| v2_b2f_i4_bandfirst_zstd1_s256 | 10.10 | 1.30 | 6.12 | 1.59 | 52 | 20.0 |
| v2_c0 | 7.79 | 1.00 | 3.85 | 1.00 | 50 | 26.0 |
| v2_m2f_bandfirst_zstd1_s256 | 10.12 | 1.30 | 6.14 | 1.60 | 53 | 20.0 |
| v2_m2f_x3_bandfirst_zstd1_s256 | 9.56 | 1.23 | 5.51 | 1.43 | 53 | 20.0 |
| v2_m2f_x4_bandfirst_zstd1_s256 | 10.17 | 1.31 | 6.15 | 1.60 | 54 | 20.0 |

### T10 — lossless check against C0

| arm | stores | shards | mismatched array-shards |
|---|---|---|---|
| v2_b2f4_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_b2f8_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_b2f_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_b2f_i4_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_m2f_bandfirst_zstd1_colexicographic_s256 | t10 | 16 | 0 |
| v2_m2f_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_m2f_x3_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_m2f_x4_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_b2f4_bandfirst_zstd1_s256 | t10asm/p0 | 64 | 0 |
| v2_b2f8_bandfirst_zstd1_s256 | t10asm/p0 | 64 | 0 |
| v2_b2f_bandfirst_zstd1_s256 | t10asm/p0 | 64 | 0 |
| v2_b2f_i4_bandfirst_zstd1_s256 | t10asm/p0 | 64 | 0 |
| v2_m2f_bandfirst_zstd1_s256 | t10asm/p0 | 64 | 0 |
| v2_m2f_x3_bandfirst_zstd1_s256 | t10asm/p0 | 64 | 0 |
| v2_m2f_x4_bandfirst_zstd1_s256 | t10asm/p0 | 64 | 0 |
