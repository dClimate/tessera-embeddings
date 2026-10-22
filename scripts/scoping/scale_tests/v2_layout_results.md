# Scale-test report — the v2 store layout (T9, T10)

The records behind [ADR 031](../../../context_docs/decisions/031-the-v2-store-chunks-the-band-axis-at-16.md).
Each run's section is `report.py` output for that run, under its own provenance. The hand-written
sections at the end cover a run whose records exist only in its log, and the microbenchmarks behind
the ADR's inference and reader figures.

| run | test | host | clients | code |
|---|---|---|---|---|
| `t9iowa`, `t9puget` | T9 | laptop (Apple M3 Max), local stores | zarr 3.4.1 + icechunk 2.3.0, and zarr 3.2.1 + icechunk 2.1.1 | `f1242f88` |
| `v2b` | T10 build, read | r7i.4xlarge, us-west-2 | both | `482debfc` |
| `v2b` | T10 assemble | m7g.4xlarge (Graviton) | zarr 3.4.1 + icechunk 2.3.0 | `482debfc` |
| `tune` | T10 tune | r7i.4xlarge | zarr 3.4.1 + icechunk 2.3.0 | `56010664` (log only) |
| `v2c-arm` | T10 build | m7g.4xlarge | zarr 3.4.1 + icechunk 2.3.0 | `77c42cc5` |
| `v2c` | T10 build, read | r7i.4xlarge | both | `77c42cc5` |
| `v2c` | T10 points, verify | r7i.4xlarge | both | `0cb87a11` |
| `v2d` | T10 assemble, verify | m7g.4xlarge | zarr 3.4.1 + icechunk 2.3.0 | `a8d06394` on `proto/band-first-ingest-coupling` |

The T10 boxes ran a tarball of the branch, so their records say `git unknown`. The code column comes
from matching each box's start time and shipped files to the branch's commits. `v2d` ran a prototype
whose shard writer accepts a band-first `embeddings` array: it writes each band-last staged tile
through a transposed view, a reorder a band-first pipeline would make in inference instead.

**The tiles** are v2 Large outputs from dev runs: four Iowa tiles from the paired run that verified
the library upgrade in PR #242 (cropland, including a 52%-valid edge tile), and four Puget Sound
tiles (city, open water, coast and conifer forest). T9 builds 2 × 2 shards per area. T10 cycles all eight into 10 × 10 shards.

**Arm names.** `v2_c0` is v1.1's geometry (256² × 128 inner chunks, default zstd). `c1` and `c2` are
256² × 16 and 512² × 16 inner chunks in the 2,048² × 128 shard; `b2` splits the shard into
2,048² × 16 objects. `zstd1` is zstd at level 1, `T` the transpose codec, `s256` the `scales` inner
chunk, and `bandfirst` the `(time, band, northing, easting)` dimension order.

## T9 — Iowa (`t9iowa`)

- backend: **local**, scale: **bench**
- icechunk **2.3.0**, zarr **3.4.1**, git `f1242f88`
- rows: 5184

> Contention/latency numbers are only load-bearing on `--backend s3`.

### T9 — v2 layout arms

| arm | embeddings MB | ratio | encode CPU s/shard |
|---|---|---|---|
| v2_b2 | 1270 | 1.45 | 1.755 |
| v2_b2_blosc_bitshuffle | 1200 | 1.54 | 2.102 |
| v2_b2_raw | 1846 | 1.00 | 0.657 |
| v2_b2_zstd1 | 1292 | 1.43 | 1.007 |
| v2_b2_zstd1_T | 1012 | 1.82 | 2.049 |
| v2_b2_zstd_T | 935 | 1.97 | 3.364 |
| v2_c0 | 1340 | 1.25 | 0.976 |
| v2_c0_blosc_bitshuffle | 1381 | 1.22 | 1.109 |
| v2_c0_raw | 1678 | 1.00 | 0.128 |
| v2_c0_zstd1 | 1354 | 1.24 | 0.487 |
| v2_c0_zstd1_T | 1099 | 1.53 | 1.033 |
| v2_c0_zstd_T | 991 | 1.69 | 2.242 |
| v2_c1 | 1263 | 1.33 | 1.550 |
| v2_c1_blosc_bitshuffle | 1191 | 1.41 | 1.723 |
| v2_c1_raw | 1678 | 1.00 | 0.431 |
| v2_c1_zstd1 | 1287 | 1.30 | 0.789 |
| v2_c1_zstd1_T | 1099 | 1.53 | 1.376 |
| v2_c1_zstd_T | 991 | 1.69 | 2.582 |
| v2_c1_zstd_T_colexicographic | 991 | 1.69 | 2.595 |
| v2_c1_zstd_T_lexicographic | 991 | 1.69 | 2.553 |
| v2_c1_zstd_T_s512 | 991 | 1.69 | 2.543 |
| v2_c2 | 1270 | 1.45 | 1.512 |
| v2_c2_blosc_bitshuffle | 1200 | 1.54 | 1.868 |
| v2_c2_raw | 1846 | 1.00 | 0.604 |
| v2_c2_zstd1 | 1292 | 1.43 | 0.923 |
| v2_c2_zstd1_T | 1012 | 1.82 | 1.819 |
| v2_c2_zstd_T | 935 | 1.97 | 2.953 |
| v2_c2_zstd_T_colexicographic | 935 | 1.97 | 2.915 |
| v2_c2_zstd_T_lexicographic | 935 | 1.97 | 2.977 |
| v2_c2_zstd_T_s512 | 935 | 1.97 | 2.978 |

#### Requests / MB fetched, zarr3.2

| arm | point 16 | point 128 | tile 16 | shard 16 | bulk 16 | bulk 64 | bulk 128 |
|---|---|---|---|---|---|---|---|
| v2_b2 | 2 / 2.8 | 15 / 19.2 | 7 / 13.3 | 1 / 25.7 | 4 / 186.1 | 16 / 648.1 | 32 / 1269.8 |
| v2_b2_blosc_bitshuffle | 2 / 2.4 | 15 / 18.0 | 7 / 11.4 | 1 / 21.9 | 4 / 159.5 | 16 / 602.1 | 32 / 1200.3 |
| v2_b2_raw | 2 / 3.6 | 15 / 28.5 | 7 / 25.2 | 1 / 50.3 | 4 / 230.7 | 16 / 922.8 | 32 / 1845.5 |
| v2_b2_zstd1 | 2 / 2.8 | 15 / 19.5 | 7 / 13.3 | 1 / 25.8 | 4 / 186.5 | 16 / 652.3 | 32 / 1291.6 |
| v2_b2_zstd1_T | 2 / 2.3 | 15 / 15.2 | 7 / 10.9 | 1 / 21.1 | 4 / 150.5 | 16 / 543.2 | 32 / 1012.2 |
| v2_b2_zstd_T | 2 / 2.1 | 15 / 14.0 | 7 / 10.0 | 1 / 19.4 | 4 / 141.5 | 16 / 501.4 | 32 / 935.1 |
| v2_c0 | 2 / 5.0 | 2 / 5.0 | 16 / 79.9 | 1 / 184.3 | 4 / 1340.0 | 4 / 1340.0 | 4 / 1340.0 |
| v2_c0_blosc_bitshuffle | 2 / 5.1 | 2 / 5.1 | 16 / 82.4 | 1 / 187.2 | 4 / 1380.8 | 4 / 1380.8 | 4 / 1380.8 |
| v2_c0_raw | 2 / 6.4 | 2 / 6.4 | 16 / 125.8 | 1 / 285.2 | 4 / 1677.7 | 4 / 1677.7 | 4 / 1677.7 |
| v2_c0_zstd1 | 2 / 5.1 | 2 / 5.1 | 16 / 81.2 | 1 / 187.8 | 4 / 1353.7 | 4 / 1353.7 | 4 / 1353.7 |
| v2_c0_zstd1_T | 2 / 4.1 | 2 / 4.1 | 16 / 64.9 | 1 / 148.7 | 4 / 1098.6 | 4 / 1098.6 | 4 / 1098.6 |
| v2_c0_zstd_T | 2 / 3.7 | 2 / 3.7 | 16 / 57.6 | 1 / 132.0 | 4 / 990.7 | 4 / 990.7 | 4 / 990.7 |
| v2_c1 | 2 / 0.7 | 7 / 4.7 | 16 / 11.1 | 35 / 25.6 | 204 / 185.4 | 804 / 645.4 | 4 / 1262.7 |
| v2_c1_blosc_bitshuffle | 2 / 0.6 | 7 / 4.4 | 16 / 9.5 | 35 / 21.8 | 204 / 158.4 | 804 / 598.6 | 4 / 1191.5 |
| v2_c1_raw | 2 / 0.8 | 7 / 6.4 | 16 / 15.7 | 35 / 35.7 | 204 / 209.7 | 804 / 838.9 | 4 / 1677.8 |
| v2_c1_zstd1 | 2 / 0.7 | 7 / 4.8 | 16 / 11.1 | 35 / 25.7 | 204 / 185.9 | 804 / 650.2 | 4 / 1286.9 |
| v2_c1_zstd1_T | 2 / 0.6 | 7 / 4.1 | 16 / 10.0 | 35 / 23.1 | 204 / 165.3 | 804 / 581.8 | 4 / 1099.0 |
| v2_c1_zstd_T | 2 / 0.6 | 7 / 3.7 | 16 / 9.2 | 35 / 20.9 | 204 / 154.3 | 804 / 529.1 | 4 / 991.0 |
| v2_c1_zstd_T_colexicographic | 2 / 0.6 | 7 / 3.7 | 16 / 9.2 | 35 / 20.9 | 204 / 154.3 | 804 / 529.1 | 4 / 991.0 |
| v2_c1_zstd_T_lexicographic | 2 / 0.6 | 7 / 3.7 | 16 / 9.2 | 35 / 20.9 | 204 / 154.3 | 804 / 529.1 | 4 / 991.0 |
| v2_c1_zstd_T_s512 | 2 / 0.6 | 7 / 3.7 | 16 / 9.2 | 35 / 20.9 | 204 / 154.3 | 804 / 529.1 | 4 / 991.0 |
| v2_c2 | 2 / 2.8 | 8 / 19.2 | 7 / 13.3 | 13 / 25.7 | 59 / 186.1 | 224 / 648.1 | 4 / 1269.8 |
| v2_c2_blosc_bitshuffle | 2 / 2.4 | 8 / 18.0 | 7 / 11.4 | 13 / 21.9 | 59 / 159.5 | 224 / 602.1 | 4 / 1200.3 |
| v2_c2_raw | 2 / 3.6 | 8 / 28.5 | 7 / 25.2 | 13 / 50.3 | 59 / 230.7 | 224 / 922.8 | 4 / 1845.5 |
| v2_c2_zstd1 | 2 / 2.8 | 8 / 19.5 | 7 / 13.4 | 13 / 25.8 | 59 / 186.6 | 224 / 652.3 | 4 / 1291.6 |
| v2_c2_zstd1_T | 2 / 2.3 | 8 / 15.2 | 7 / 10.9 | 13 / 21.1 | 59 / 150.5 | 224 / 543.2 | 4 / 1012.2 |
| v2_c2_zstd_T | 2 / 2.1 | 8 / 14.0 | 7 / 10.0 | 13 / 19.4 | 59 / 141.5 | 224 / 501.4 | 4 / 935.1 |
| v2_c2_zstd_T_colexicographic | 2 / 2.1 | 8 / 14.0 | 7 / 10.0 | 13 / 19.4 | 59 / 141.5 | 224 / 501.4 | 4 / 935.1 |
| v2_c2_zstd_T_lexicographic | 2 / 2.1 | 8 / 14.0 | 7 / 10.0 | 13 / 19.4 | 59 / 141.5 | 224 / 501.4 | 4 / 935.1 |
| v2_c2_zstd_T_s512 | 2 / 2.1 | 8 / 14.0 | 7 / 10.0 | 13 / 19.4 | 59 / 141.5 | 224 / 501.4 | 4 / 935.1 |

#### Requests / MB fetched, zarr3.4

| arm | point 16 | point 128 | tile 16 | shard 16 | bulk 16 | bulk 64 | bulk 128 |
|---|---|---|---|---|---|---|---|
| v2_b2 | 2 / 2.8 | 15 / 19.2 | 4 / 14.2 | 1 / 25.7 | 4 / 186.1 | 16 / 648.1 | 32 / 1269.8 |
| v2_b2_blosc_bitshuffle | 2 / 2.4 | 15 / 18.0 | 4 / 12.2 | 1 / 21.9 | 4 / 159.5 | 16 / 602.1 | 32 / 1200.3 |
| v2_b2_raw | 2 / 3.6 | 15 / 28.5 | 5 / 25.2 | 1 / 50.3 | 4 / 230.7 | 16 / 922.8 | 32 / 1845.5 |
| v2_b2_zstd1 | 2 / 2.8 | 15 / 19.5 | 4 / 14.2 | 1 / 25.8 | 4 / 186.5 | 16 / 652.3 | 32 / 1291.6 |
| v2_b2_zstd1_T | 2 / 2.3 | 15 / 15.2 | 4 / 11.7 | 1 / 21.1 | 4 / 150.5 | 16 / 543.2 | 32 / 1012.2 |
| v2_b2_zstd_T | 2 / 2.1 | 15 / 14.0 | 4 / 10.8 | 1 / 19.4 | 4 / 141.5 | 16 / 501.4 | 32 / 935.1 |
| v2_c0 | 2 / 5.0 | 2 / 5.0 | 8 / 79.9 | 1 / 184.3 | 4 / 1340.0 | 4 / 1340.0 | 4 / 1340.0 |
| v2_c0_blosc_bitshuffle | 2 / 5.1 | 2 / 5.1 | 8 / 82.4 | 1 / 187.2 | 4 / 1380.8 | 4 / 1380.8 | 4 / 1380.8 |
| v2_c0_raw | 2 / 6.4 | 2 / 6.4 | 9 / 125.8 | 1 / 285.2 | 4 / 1677.7 | 4 / 1677.7 | 4 / 1677.7 |
| v2_c0_zstd1 | 2 / 5.1 | 2 / 5.1 | 8 / 81.2 | 1 / 187.8 | 4 / 1353.7 | 4 / 1353.7 | 4 / 1353.7 |
| v2_c0_zstd1_T | 2 / 4.1 | 2 / 4.1 | 8 / 64.9 | 1 / 148.7 | 4 / 1098.6 | 4 / 1098.6 | 4 / 1098.6 |
| v2_c0_zstd_T | 2 / 3.7 | 2 / 3.7 | 7 / 58.6 | 1 / 132.0 | 4 / 990.7 | 4 / 990.7 | 4 / 990.7 |
| v2_c1 | 2 / 0.7 | 7 / 4.9 | 6 / 11.5 | 9 / 26.9 | 54 / 187.4 | 45 / 645.4 | 4 / 1262.7 |
| v2_c1_blosc_bitshuffle | 2 / 0.6 | 7 / 4.5 | 6 / 9.9 | 9 / 23.0 | 54 / 160.3 | 42 / 598.6 | 4 / 1191.5 |
| v2_c1_raw | 2 / 0.8 | 7 / 6.7 | 7 / 15.7 | 11 / 37.8 | 57 / 211.8 | 55 / 838.9 | 4 / 1677.8 |
| v2_c1_zstd1 | 2 / 0.7 | 7 / 5.0 | 6 / 11.6 | 9 / 27.0 | 54 / 188.0 | 45 / 650.2 | 4 / 1286.9 |
| v2_c1_zstd1_T | 2 / 0.6 | 7 / 4.2 | 6 / 10.4 | 9 / 24.3 | 54 / 167.2 | 42 / 581.8 | 4 / 1099.0 |
| v2_c1_zstd_T | 2 / 0.6 | 7 / 3.8 | 6 / 9.5 | 9 / 22.1 | 54 / 156.0 | 38 / 529.1 | 4 / 991.0 |
| v2_c1_zstd_T_colexicographic | 2 / 0.6 | 7 / 3.7 | 6 / 9.2 | 3 / 20.9 | 16 / 154.3 | 38 / 529.1 | 4 / 991.0 |
| v2_c1_zstd_T_lexicographic | 2 / 0.6 | 2 / 3.7 | 16 / 9.2 | 31 / 24.3 | 197 / 160.1 | 191 / 536.8 | 4 / 991.0 |
| v2_c1_zstd_T_s512 | 2 / 0.6 | 7 / 3.8 | 6 / 9.5 | 9 / 22.1 | 54 / 156.0 | 38 / 529.1 | 4 / 991.0 |
| v2_c2 | 2 / 2.8 | 7 / 19.4 | 5 / 14.2 | 4 / 27.1 | 18 / 188.0 | 48 / 648.1 | 4 / 1269.8 |
| v2_c2_blosc_bitshuffle | 2 / 2.4 | 7 / 18.2 | 5 / 12.2 | 3 / 23.2 | 17 / 161.2 | 45 / 602.1 | 4 / 1200.3 |
| v2_c2_raw | 2 / 3.6 | 8 / 28.5 | 6 / 25.2 | 5 / 50.3 | 20 / 230.7 | 59 / 922.8 | 4 / 1845.5 |
| v2_c2_zstd1 | 2 / 2.8 | 7 / 19.8 | 5 / 14.2 | 4 / 27.2 | 18 / 188.5 | 48 / 652.3 | 4 / 1291.6 |
| v2_c2_zstd1_T | 2 / 2.3 | 7 / 15.4 | 5 / 11.6 | 3 / 22.3 | 17 / 152.1 | 40 / 543.2 | 4 / 1012.2 |
| v2_c2_zstd_T | 2 / 2.1 | 7 / 14.2 | 5 / 10.6 | 3 / 20.5 | 17 / 143.1 | 39 / 501.4 | 4 / 935.1 |
| v2_c2_zstd_T_colexicographic | 2 / 2.1 | 8 / 14.0 | 4 / 10.0 | 3 / 19.4 | 16 / 141.5 | 40 / 501.4 | 4 / 935.1 |
| v2_c2_zstd_T_lexicographic | 2 / 2.1 | 3 / 14.0 | 7 / 10.0 | 12 / 19.9 | 58 / 142.1 | 56 / 503.8 | 4 / 935.1 |
| v2_c2_zstd_T_s512 | 2 / 2.1 | 7 / 14.2 | 5 / 10.6 | 3 / 20.5 | 17 / 143.1 | 39 / 501.4 | 4 / 935.1 |

#### Dask, bulk 16-d pull

| arm | tasks default | tasks shard-aligned | requests default | requests shard-aligned |
|---|---|---|---|---|
| v2_b2 | 577 | 37 | 119 | 4 |
| v2_b2_zstd_T | 577 | 37 | 119 | 4 |
| v2_c0 | 513 | 37 | 456 | 4 |
| v2_c0_zstd_T | 513 | 37 | 456 | 4 |
| v2_c1 | 2305 | 37 | 456 | 54 |
| v2_c1_zstd_T | 2305 | 37 | 456 | 54 |
| v2_c2 | 577 | 37 | 119 | 18 |
| v2_c2_zstd_T | 577 | 37 | 119 | 17 |

## T9 — Puget Sound (`t9puget`)

- backend: **local**, scale: **bench**
- icechunk **2.3.0**, zarr **3.4.1**, git `f1242f88`
- rows: 5184

> Contention/latency numbers are only load-bearing on `--backend s3`.

### T9 — v2 layout arms

| arm | embeddings MB | ratio | encode CPU s/shard |
|---|---|---|---|
| v2_b2 | 1672 | 1.28 | 1.959 |
| v2_b2_blosc_bitshuffle | 1776 | 1.21 | 2.257 |
| v2_b2_raw | 2147 | 1.00 | 0.763 |
| v2_b2_zstd1 | 1722 | 1.25 | 1.253 |
| v2_b2_zstd1_T | 1380 | 1.56 | 2.191 |
| v2_b2_zstd_T | 1345 | 1.60 | 3.573 |
| v2_c0 | 1768 | 1.21 | 1.151 |
| v2_c0_blosc_bitshuffle | 1968 | 1.09 | 1.194 |
| v2_c0_raw | 2147 | 1.00 | 0.140 |
| v2_c0_zstd1 | 1802 | 1.19 | 0.873 |
| v2_c0_zstd1_T | 1467 | 1.46 | 1.045 |
| v2_c0_zstd_T | 1422 | 1.51 | 2.171 |
| v2_c1 | 1655 | 1.30 | 1.816 |
| v2_c1_blosc_bitshuffle | 1771 | 1.21 | 1.807 |
| v2_c1_raw | 2148 | 1.00 | 0.466 |
| v2_c1_zstd1 | 1707 | 1.26 | 0.978 |
| v2_c1_zstd1_T | 1467 | 1.46 | 1.232 |
| v2_c1_zstd_T | 1422 | 1.51 | 2.556 |
| v2_c1_zstd_T_colexicographic | 1422 | 1.51 | 2.525 |
| v2_c1_zstd_T_lexicographic | 1422 | 1.51 | 2.553 |
| v2_c1_zstd_T_s512 | 1422 | 1.51 | 2.554 |
| v2_c2 | 1672 | 1.28 | 1.863 |
| v2_c2_blosc_bitshuffle | 1776 | 1.21 | 2.008 |
| v2_c2_raw | 2147 | 1.00 | 0.675 |
| v2_c2_zstd1 | 1722 | 1.25 | 1.093 |
| v2_c2_zstd1_T | 1380 | 1.56 | 1.989 |
| v2_c2_zstd_T | 1345 | 1.60 | 3.271 |
| v2_c2_zstd_T_colexicographic | 1345 | 1.60 | 3.303 |
| v2_c2_zstd_T_lexicographic | 1345 | 1.60 | 3.251 |
| v2_c2_zstd_T_s512 | 1345 | 1.60 | 3.276 |

#### Requests / MB fetched, zarr3.2

| arm | point 16 | point 128 | tile 16 | shard 16 | bulk 16 | bulk 64 | bulk 128 |
|---|---|---|---|---|---|---|---|
| v2_b2 | 2 / 3.8 | 16 / 26.2 | 10 / 31.7 | 1 / 57.8 | 4 / 244.8 | 16 / 839.3 | 32 / 1672.0 |
| v2_b2_blosc_bitshuffle | 2 / 3.6 | 16 / 27.8 | 10 / 27.7 | 1 / 50.4 | 4 / 233.1 | 16 / 877.8 | 32 / 1775.6 |
| v2_b2_raw | 2 / 4.2 | 16 / 33.6 | 10 / 37.7 | 1 / 67.1 | 4 / 268.4 | 16 / 1073.7 | 32 / 2147.5 |
| v2_b2_zstd1 | 2 / 3.9 | 16 / 26.9 | 10 / 34.0 | 1 / 61.3 | 4 / 250.6 | 16 / 861.2 | 32 / 1722.2 |
| v2_b2_zstd1_T | 2 / 3.2 | 16 / 21.6 | 10 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2_zstd_T | 2 / 3.1 | 16 / 21.1 | 10 / 22.5 | 1 / 42.6 | 4 / 200.7 | 16 / 709.2 | 32 / 1344.9 |
| v2_c0 | 2 / 6.9 | 2 / 6.9 | 26 / 169.5 | 1 / 427.0 | 4 / 1767.9 | 4 / 1767.9 | 4 / 1767.9 |
| v2_c0_blosc_bitshuffle | 2 / 7.7 | 2 / 7.7 | 26 / 166.9 | 1 / 430.3 | 4 / 1967.9 | 4 / 1967.9 | 4 / 1967.9 |
| v2_c0_raw | 2 / 8.4 | 2 / 8.4 | 26 / 209.7 | 1 / 536.9 | 4 / 2147.5 | 4 / 2147.5 | 4 / 2147.5 |
| v2_c0_zstd1 | 2 / 7.0 | 2 / 7.0 | 26 / 179.9 | 1 / 448.1 | 4 / 1802.3 | 4 / 1802.3 | 4 / 1802.3 |
| v2_c0_zstd1_T | 2 / 5.8 | 2 / 5.8 | 26 / 119.8 | 1 / 319.3 | 4 / 1466.8 | 4 / 1466.8 | 4 / 1466.8 |
| v2_c0_zstd_T | 2 / 5.6 | 2 / 5.6 | 26 / 112.6 | 1 / 301.0 | 4 / 1422.4 | 4 / 1422.4 | 4 / 1422.4 |
| v2_c1 | 2 / 1.0 | 9 / 6.5 | 26 / 22.0 | 65 / 56.4 | 260 / 242.0 | 1028 / 830.4 | 4 / 1655.3 |
| v2_c1_blosc_bitshuffle | 2 / 0.9 | 9 / 7.0 | 26 / 19.5 | 65 / 50.3 | 260 / 232.2 | 1028 / 877.2 | 4 / 1771.4 |
| v2_c1_raw | 2 / 1.1 | 9 / 8.4 | 26 / 26.2 | 65 / 67.1 | 260 / 268.5 | 1028 / 1073.8 | 4 / 2147.5 |
| v2_c1_zstd1 | 2 / 1.0 | 9 / 6.7 | 26 / 23.9 | 65 / 60.3 | 260 / 248.2 | 1028 / 853.5 | 4 / 1706.9 |
| v2_c1_zstd1_T | 2 / 0.9 | 9 / 5.8 | 26 / 18.2 | 65 / 48.5 | 260 / 220.2 | 1028 / 765.9 | 4 / 1466.9 |
| v2_c1_zstd_T | 2 / 0.9 | 9 / 5.6 | 26 / 16.8 | 65 / 45.2 | 260 / 214.0 | 1028 / 743.1 | 4 / 1422.1 |
| v2_c1_zstd_T_colexicographic | 2 / 0.9 | 9 / 5.6 | 26 / 16.8 | 65 / 45.2 | 260 / 214.0 | 1028 / 743.1 | 4 / 1422.1 |
| v2_c1_zstd_T_lexicographic | 2 / 0.9 | 9 / 5.6 | 26 / 16.8 | 65 / 45.2 | 260 / 214.0 | 1028 / 743.1 | 4 / 1422.1 |
| v2_c1_zstd_T_s512 | 2 / 0.9 | 9 / 5.6 | 26 / 16.8 | 65 / 45.2 | 260 / 214.0 | 1028 / 743.1 | 4 / 1422.1 |
| v2_c2 | 2 / 3.8 | 9 / 26.2 | 10 / 31.7 | 17 / 57.8 | 68 / 244.8 | 260 / 839.3 | 4 / 1672.0 |
| v2_c2_blosc_bitshuffle | 2 / 3.6 | 9 / 27.8 | 10 / 27.7 | 17 / 50.4 | 68 / 233.1 | 260 / 877.8 | 4 / 1775.6 |
| v2_c2_raw | 2 / 4.2 | 9 / 33.6 | 10 / 37.8 | 17 / 67.1 | 68 / 268.4 | 260 / 1073.8 | 4 / 2147.5 |
| v2_c2_zstd1 | 2 / 3.9 | 9 / 26.9 | 10 / 34.0 | 17 / 61.3 | 68 / 250.6 | 260 / 861.2 | 4 / 1722.2 |
| v2_c2_zstd1_T | 2 / 3.2 | 9 / 21.6 | 10 / 24.0 | 17 / 45.3 | 68 / 206.1 | 260 / 729.2 | 4 / 1380.2 |
| v2_c2_zstd_T | 2 / 3.1 | 9 / 21.1 | 10 / 22.5 | 17 / 42.6 | 68 / 200.7 | 260 / 709.2 | 4 / 1344.9 |
| v2_c2_zstd_T_colexicographic | 2 / 3.1 | 9 / 21.1 | 10 / 22.5 | 17 / 42.6 | 68 / 200.7 | 260 / 709.2 | 4 / 1344.9 |
| v2_c2_zstd_T_lexicographic | 2 / 3.1 | 9 / 21.1 | 10 / 22.5 | 17 / 42.6 | 68 / 200.7 | 260 / 709.2 | 4 / 1344.9 |
| v2_c2_zstd_T_s512 | 2 / 3.1 | 9 / 21.1 | 10 / 22.5 | 17 / 42.6 | 68 / 200.7 | 260 / 709.2 | 4 / 1344.9 |

#### Requests / MB fetched, zarr3.4

| arm | point 16 | point 128 | tile 16 | shard 16 | bulk 16 | bulk 64 | bulk 128 |
|---|---|---|---|---|---|---|---|
| v2_b2 | 2 / 3.8 | 16 / 26.2 | 5 / 31.7 | 1 / 57.8 | 4 / 244.8 | 16 / 839.3 | 32 / 1672.0 |
| v2_b2_blosc_bitshuffle | 2 / 3.6 | 16 / 27.8 | 5 / 27.7 | 1 / 50.4 | 4 / 233.1 | 16 / 877.8 | 32 / 1775.6 |
| v2_b2_raw | 2 / 4.2 | 16 / 33.6 | 6 / 37.7 | 1 / 67.1 | 4 / 268.4 | 16 / 1073.7 | 32 / 2147.5 |
| v2_b2_zstd1 | 2 / 3.9 | 16 / 26.9 | 6 / 34.0 | 1 / 61.3 | 4 / 250.6 | 16 / 861.2 | 32 / 1722.2 |
| v2_b2_zstd1_T | 2 / 3.2 | 16 / 21.6 | 5 / 24.0 | 1 / 45.3 | 4 / 206.1 | 16 / 729.2 | 32 / 1380.2 |
| v2_b2_zstd_T | 2 / 3.1 | 16 / 21.1 | 5 / 22.5 | 1 / 42.6 | 4 / 200.7 | 16 / 709.2 | 32 / 1344.9 |
| v2_c0 | 2 / 6.9 | 2 / 6.9 | 16 / 169.5 | 1 / 427.0 | 4 / 1767.9 | 4 / 1767.9 | 4 / 1767.9 |
| v2_c0_blosc_bitshuffle | 2 / 7.7 | 2 / 7.7 | 16 / 166.9 | 1 / 430.3 | 4 / 1967.9 | 4 / 1967.9 | 4 / 1967.9 |
| v2_c0_raw | 2 / 8.4 | 2 / 8.4 | 16 / 209.7 | 1 / 536.9 | 4 / 2147.5 | 4 / 2147.5 | 4 / 2147.5 |
| v2_c0_zstd1 | 2 / 7.0 | 2 / 7.0 | 16 / 179.9 | 1 / 448.1 | 4 / 1802.3 | 4 / 1802.3 | 4 / 1802.3 |
| v2_c0_zstd1_T | 2 / 5.8 | 2 / 5.8 | 14 / 119.8 | 1 / 319.3 | 4 / 1466.8 | 4 / 1466.8 | 4 / 1466.8 |
| v2_c0_zstd_T | 2 / 5.6 | 2 / 5.6 | 13 / 112.6 | 1 / 301.0 | 4 / 1422.4 | 4 / 1422.4 | 4 / 1422.4 |
| v2_c1 | 2 / 1.0 | 9 / 6.5 | 10 / 23.4 | 17 / 56.4 | 68 / 242.0 | 56 / 830.4 | 4 / 1655.3 |
| v2_c1_blosc_bitshuffle | 2 / 0.9 | 9 / 7.0 | 10 / 20.6 | 17 / 50.3 | 68 / 232.2 | 60 / 877.2 | 4 / 1771.4 |
| v2_c1_raw | 2 / 1.1 | 9 / 8.4 | 10 / 28.3 | 17 / 67.1 | 68 / 268.5 | 68 / 1073.8 | 4 / 2147.5 |
| v2_c1_zstd1 | 2 / 1.0 | 9 / 6.7 | 10 / 25.4 | 17 / 60.3 | 68 / 248.2 | 59 / 853.5 | 4 / 1706.9 |
| v2_c1_zstd1_T | 2 / 0.9 | 9 / 5.8 | 10 / 19.3 | 17 / 48.5 | 68 / 220.2 | 53 / 765.9 | 4 / 1466.9 |
| v2_c1_zstd_T | 2 / 0.9 | 9 / 5.6 | 10 / 17.7 | 17 / 45.2 | 68 / 214.0 | 51 / 743.1 | 4 / 1422.1 |
| v2_c1_zstd_T_colexicographic | 2 / 0.9 | 9 / 5.6 | 6 / 16.8 | 4 / 45.2 | 19 / 214.0 | 51 / 743.1 | 4 / 1422.1 |
| v2_c1_zstd_T_lexicographic | 2 / 0.9 | 2 / 5.6 | 26 / 16.8 | 65 / 45.2 | 260 / 214.0 | 260 / 743.1 | 4 / 1422.1 |
| v2_c1_zstd_T_s512 | 2 / 0.9 | 9 / 5.6 | 10 / 17.7 | 17 / 45.2 | 68 / 214.0 | 51 / 743.1 | 4 / 1422.1 |
| v2_c2 | 2 / 3.8 | 9 / 26.2 | 6 / 31.7 | 5 / 57.8 | 20 / 244.8 | 62 / 839.3 | 4 / 1672.0 |
| v2_c2_blosc_bitshuffle | 2 / 3.6 | 9 / 27.8 | 6 / 27.7 | 5 / 50.4 | 20 / 233.1 | 65 / 877.8 | 4 / 1775.6 |
| v2_c2_raw | 2 / 4.2 | 9 / 33.6 | 6 / 37.8 | 5 / 67.1 | 20 / 268.4 | 68 / 1073.8 | 4 / 2147.5 |
| v2_c2_zstd1 | 2 / 3.9 | 9 / 26.9 | 6 / 34.0 | 5 / 61.3 | 20 / 250.6 | 64 / 861.2 | 4 / 1722.2 |
| v2_c2_zstd1_T | 2 / 3.2 | 9 / 21.6 | 6 / 24.0 | 5 / 45.3 | 20 / 206.1 | 53 / 729.2 | 4 / 1380.2 |
| v2_c2_zstd_T | 2 / 3.1 | 9 / 21.1 | 6 / 22.5 | 5 / 42.6 | 20 / 200.7 | 51 / 709.2 | 4 / 1344.9 |
| v2_c2_zstd_T_colexicographic | 2 / 3.1 | 9 / 21.1 | 4 / 22.5 | 4 / 42.6 | 19 / 200.7 | 52 / 709.2 | 4 / 1344.9 |
| v2_c2_zstd_T_lexicographic | 2 / 3.1 | 3 / 21.1 | 10 / 22.5 | 17 / 42.6 | 68 / 200.7 | 68 / 709.2 | 4 / 1344.9 |
| v2_c2_zstd_T_s512 | 2 / 3.1 | 9 / 21.1 | 6 / 22.5 | 5 / 42.6 | 20 / 200.7 | 51 / 709.2 | 4 / 1344.9 |

#### Dask, bulk 16-d pull

| arm | tasks default | tasks shard-aligned | requests default | requests shard-aligned |
|---|---|---|---|---|
| v2_b2 | 577 | 37 | 128 | 4 |
| v2_b2_zstd_T | 577 | 37 | 128 | 4 |
| v2_c0 | 513 | 37 | 512 | 4 |
| v2_c0_zstd_T | 513 | 37 | 512 | 4 |
| v2_c1 | 2305 | 37 | 512 | 68 |
| v2_c1_zstd_T | 2305 | 37 | 512 | 68 |
| v2_c2 | 577 | 37 | 128 | 20 |
| v2_c2_zstd_T | 577 | 37 | 128 | 20 |

## T10 — band-last finalists (`v2b`)

- backend: **s3**, scale: **bench**
- icechunk **2.3.0**, zarr **3.4.1**, git `unknown`
- rows: 1174

> Contention/latency numbers are only load-bearing on `--backend s3`.

### T10 — assembly code: CPU per staged tile, wall per run (median of passes)

| arm | worker CPU s | vs C0 | write CPU s | vs C0 | wall s, whole run | stored GB |
|---|---|---|---|---|---|---|
| v2_b2_zstd1_s256 | 7.30 | 0.91 | 3.17 | 0.81 | 41 | 25.1 |
| v2_c0 | 8.05 | 1.00 | 3.91 | 1.00 | 51 | 26.0 |
| v2_c2_zstd1_T_s256 | 9.45 | 1.17 | 5.26 | 1.35 | 56 | 20.0 |
| v2_c2_zstd1_s256 | 7.29 | 0.91 | 3.15 | 0.81 | 47 | 25.1 |

### T10 — cold reads from S3, zarr3.2 (median of passes; wire MB in brackets)

| arm | point 16 p50 | point 128 p50 | tile 16 | bulk 16 | bulk 128 | 8x8 pull 16 | 8x8 pull 64 | 8x8 pull 128 | dask default | dask aligned |
|---|---|---|---|---|---|---|---|---|---|---|
| v2_b2_zstd1_s256 | 335 ms (1209) | 426 ms (8222) | 1.8 s (39) | 0.7 s (233) | 3.4 s (1560) | 5.1 s (3788) | 17.4 s (12695) | 45.3 s (25075) | 9.1 s (3583) | 4.8 s (3611) |
| v2_c0 | 460 ms (2146) | 451 ms (2146) | 4.5 s (189) | 2.0 s (1603) | 2.5 s (1592) | 19.9 s (25786) | 22.5 s (25894) | 25.7 s (25939) | 41.7 s (25668) | 19.6 s (25526) |
| v2_c2_zstd1_T_colexicographic_s256 | 369 ms (993) | 1464 ms (6592) | 2.1 s (31) | 1.8 s (195) | 3.6 s (1252) | 11.7 s (3096) | 37.9 s (10662) | 43.6 s (20038) | 9.6 s (2932) | 7.7 s (2944) |
| v2_c2_zstd1_T_s256 | 372 ms (996) | 1304 ms (6585) | 2.1 s (32) | 2.8 s (197) | 3.4 s (1253) | 13.2 s (3080) | 37.6 s (10660) | 43.9 s (19936) | 10.4 s (2909) | 8.7 s (2938) |
| v2_c2_zstd1_s256 | 390 ms (1207) | 1434 ms (8224) | 2.3 s (38) | 2.3 s (236) | 3.6 s (1550) | 13.4 s (3765) | 40.2 s (12601) | 44.2 s (25112) | 11.0 s (3595) | 8.0 s (3618) |

### T10 — cold reads from S3, zarr3.4 (median of passes; wire MB in brackets)

| arm | point 16 p50 | point 128 p50 | tile 16 | bulk 16 | bulk 128 | 8x8 pull 16 | 8x8 pull 64 | 8x8 pull 128 | dask default | dask aligned |
|---|---|---|---|---|---|---|---|---|---|---|
| v2_b2_zstd1_s256 | 380 ms (1215) | 453 ms (8202) | 0.8 s (39) | 0.7 s (234) | 3.2 s (1546) | 5.0 s (3784) | 16.3 s (12588) | 44.4 s (24923) | 9.1 s (3612) | 4.8 s (3601) |
| v2_c0 | 460 ms (2156) | 467 ms (2135) | 0.8 s (187) | 2.2 s (1601) | 2.4 s (1597) | 19.5 s (25969) | 22.6 s (25779) | 28.0 s (25858) | 46.5 s (25614) | 19.7 s (25593) |
| v2_c2_zstd1_T_colexicographic_s256 | 433 ms (999) | 502 ms (6576) | 0.7 s (31) | 0.8 s (198) | 3.3 s (1255) | 5.7 s (3097) | 19.0 s (10670) | 42.1 s (19982) | 10.6 s (2944) | 5.9 s (2950) |
| v2_c2_zstd1_T_s256 | 427 ms (992) | 500 ms (6602) | 0.8 s (32) | 0.8 s (199) | 3.4 s (1251) | 6.0 s (3127) | 19.0 s (10581) | 42.5 s (19985) | 11.1 s (2944) | 5.5 s (2954) |
| v2_c2_zstd1_s256 | 423 ms (1200) | 531 ms (8223) | 0.6 s (39) | 0.7 s (236) | 3.5 s (1538) | 5.6 s (3767) | 19.1 s (12666) | 42.4 s (24997) | 11.4 s (3611) | 5.2 s (3628) |

## T10 — reader settings (`tune`, from the run log)

The run failed in its second pass, before writing its records, so these are the log's lines.
Wall seconds; two values are passes 0 and 1.

| arm | 128-d pull 8 × 8, default | same, fetch concurrency 64 | 128-d pull, Dask at band-aligned chunks | 128-d bulk, concurrency 64 | 16-d pull 8 × 8, concurrency 64 |
|---|---|---|---|---|---|
| v2_c0 | 26.1, 27.0 | 34.0 | 35.0 | 2.5 | 25.4 |
| v2_c2_zstd1_s256 | 43.5, 43.4 | 52.1, 51.1 | 56.5 | 3.4 | 5.9 |
| v2_c2_zstd1_T_s256 | 43.4, 43.3 | 49.8, 49.7 | 62.8 | 3.4 | 6.9 |
| v2_b2_zstd1_s256 | 45.5, 44.8 | 46.4, 46.0 | 55.7 | 3.4 | 6.0 |

## T10 — band-first build on Graviton (`v2c-arm`)

- backend: **s3**, scale: **bench**
- icechunk **2.3.0**, zarr **3.4.1**, git `unknown`
- rows: 19

> Contention/latency numbers are only load-bearing on `--backend s3`.

### T10 — bench store build, encode CPU per shard

| arm | encode CPU s | vs C0 | stored GB | vs C0 |
|---|---|---|---|---|
| v2_b2_zstd1_s256 | 2.72 | 0.81 | 37.7 | 0.97 |
| v2_b2f_bandfirst_zstd1_s256 | 2.50 | 0.74 | 30.0 | 0.77 |
| v2_c0 | 3.36 | 1.00 | 38.9 | 1.00 |
| v2_c2_zstd1_T_s256 | 4.52 | 1.34 | 30.0 | 0.77 |
| v2_c2_zstd1_s256 | 2.78 | 0.83 | 37.7 | 0.97 |
| v2_c2f_bandfirst_zstd1_s256 | 2.54 | 0.75 | 30.0 | 0.77 |

## T10 — band-first and band-last finalists, one session (`v2c`)

- backend: **s3**, scale: **bench**
- icechunk **2.3.0**, zarr **3.4.1**, git `unknown`
- rows: 1325

> Contention/latency numbers are only load-bearing on `--backend s3`.

### T10 — bench store build, encode CPU per shard

| arm | encode CPU s | vs C0 | stored GB | vs C0 |
|---|---|---|---|---|
| v2_b2_zstd1_s256 | 3.14 | 0.85 | 37.7 | 0.97 |
| v2_b2f_bandfirst_zstd1_s256 | 3.16 | 0.86 | 30.0 | 0.77 |
| v2_c0 | 3.70 | 1.00 | 38.9 | 1.00 |
| v2_c2_zstd1_T_s256 | 9.14 | 2.47 | 30.0 | 0.77 |
| v2_c2_zstd1_s256 | 3.16 | 0.85 | 37.7 | 0.97 |
| v2_c2f_bandfirst_zstd1_s256 | 3.37 | 0.91 | 30.0 | 0.77 |

### T10 — cold reads from S3, zarr3.2 (median of passes; wire MB in brackets)

| arm | point 16 p50 | point 128 p50 | bulk 128 | 8x8 pull 16 | 8x8 pull 64 | 8x8 pull 128 | dask default |
|---|---|---|---|---|---|---|---|
| v2_b2_zstd1_s256 | 331 ms (1207) | 414 ms (8217) | 3.5 s (1541) | 5.0 s (3733) | 18.2 s (12651) | 47.0 s (24927) | 9.1 s (3630) |
| v2_b2f_bandfirst_zstd1_s256 | 305 ms (994) | 420 ms (6580) | 2.2 s (1252) | 5.2 s (3122) | 15.1 s (10689) | 27.8 s (20058) | 8.4 s (2937) |
| v2_c0 | 405 ms (2159) | 472 ms (2141) | 2.6 s (1601) | 19.9 s (25718) | 23.8 s (25923) | 30.0 s (25701) | 40.9 s (25653) |
| v2_c2_zstd1_T_s256 | 359 ms (993) | 1237 ms (6599) | 3.8 s (1250) | 12.3 s (3102) | 37.8 s (10654) | 46.3 s (19865) | 10.5 s (2949) |
| v2_c2_zstd1_s256 | 382 ms (1209) | 1474 ms (8218) | 3.9 s (1538) | 13.3 s (3767) | 39.9 s (12578) | 47.3 s (25095) | 10.6 s (3593) |
| v2_c2f_bandfirst_zstd1_s256 | 332 ms (994) | 1132 ms (6577) | 2.6 s (1244) | 14.1 s (3090) | 36.8 s (10622) | 29.0 s (19765) | 11.3 s (2913) |

### T10 — cold reads from S3, zarr3.4 (median of passes; wire MB in brackets)

| arm | point 16 p50 | point 128 p50 | bulk 128 | 8x8 pull 16 | 8x8 pull 64 | 8x8 pull 128 | dask default |
|---|---|---|---|---|---|---|---|
| v2_b2_zstd1_s256 | 348 ms (1205) | 424 ms (8213) | 3.5 s (1545) | 5.0 s (3763) | 18.0 s (12577) | 48.5 s (24960) | 9.0 s (3584) |
| v2_b2f_bandfirst_zstd1_s256 | 324 ms (993) | 391 ms (6574) | 2.2 s (1255) | 4.9 s (3092) | 14.7 s (10707) | 27.7 s (19921) | 8.6 s (2952) |
| v2_c0 | 442 ms (2138) | 419 ms (2151) | 2.8 s (1597) | 20.2 s (25873) | 24.2 s (25651) | 27.9 s (25772) | 43.4 s (25539) |
| v2_c2_zstd1_T_s256 | 367 ms (991) | 443 ms (6568) | 3.8 s (1249) | 6.2 s (3127) | 20.5 s (10655) | 47.2 s (19995) | 11.2 s (2926) |
| v2_c2_zstd1_s256 | 382 ms (1205) | 457 ms (8226) | 3.9 s (1551) | 5.4 s (3795) | 20.1 s (12657) | 48.1 s (25162) | 11.5 s (3593) |
| v2_c2f_bandfirst_zstd1_s256 | 366 ms (994) | 460 ms (6544) | 2.8 s (1248) | 5.5 s (3115) | 16.7 s (10677) | 29.4 s (19931) | 11.7 s (2945) |

### T10 — point reads, p50 [95% interval] by pass (0, 1), zarr3.2

| arm | point 16 ms | point 128 ms | open s |
|---|---|---|---|
| v2_b2_zstd1_s256 | 290 [282, 298] / 345 [334, 355] | 399 [389, 406] / 376 [370, 384] | 0.14 |
| v2_b2f_bandfirst_zstd1_s256 | 342 [335, 349] / 280 [270, 287] | 387 [377, 393] / 357 [350, 364] | 0.14 |
| v2_c0 | 365 [356, 376] / 392 [377, 405] | 475 [464, 490] / 387 [376, 394] | 0.25 |
| v2_c2_zstd1_T_s256 | 333 [322, 342] / 369 [361, 380] | 1238 [1216, 1255] / 1120 [1096, 1132] | 0.14 |
| v2_c2_zstd1_s256 | 337 [328, 347] / 358 [346, 366] | 1502 [1486, 1525] / 1194 [1178, 1213] | 0.14 |
| v2_c2f_bandfirst_zstd1_s256 | 320 [315, 331] / 342 [333, 350] | 1075 [1045, 1102] / 1020 [997, 1037] | 0.14 |

### T10 — point reads, p50 [95% interval] by pass (0, 1), zarr3.4

| arm | point 16 ms | point 128 ms | open s |
|---|---|---|---|
| v2_b2_zstd1_s256 | 353 [343, 363] / 340 [330, 350] | 432 [416, 443] / 402 [392, 410] | 0.14 |
| v2_b2f_bandfirst_zstd1_s256 | 331 [325, 341] / 270 [262, 279] | 370 [362, 384] / 400 [391, 409] | 0.14 |
| v2_c0 | 459 [444, 468] / 423 [412, 432] | 442 [425, 452] / 422 [406, 436] | 0.13 |
| v2_c2_zstd1_T_s256 | 385 [374, 397] / 371 [362, 383] | 468 [456, 487] / 460 [451, 470] | 0.15 |
| v2_c2_zstd1_s256 | 409 [399, 425] / 376 [367, 387] | 463 [455, 468] / 472 [465, 485] | 0.13 |
| v2_c2f_bandfirst_zstd1_s256 | 376 [364, 383] / 338 [329, 346] | 443 [438, 453] / 451 [439, 464] | 0.16 |

### T10 — lossless check against C0

| arm | stores | shards | mismatched array-shards |
|---|---|---|---|
| v2_b2_zstd1_s256 | t10 | 16 | 0 |
| v2_b2f_bandfirst_zstd1_s256 | t10 | 16 | 0 |
| v2_c2_zstd1_T_s256 | t10 | 16 | 0 |
| v2_c2_zstd1_s256 | t10 | 16 | 0 |
| v2_c2f_bandfirst_zstd1_s256 | t10 | 16 | 0 |

## T10 — band-first assembly through the library's writer (`v2d`)

- backend: **s3**, scale: **bench**
- icechunk **2.3.0**, zarr **3.4.1**, git `unknown`
- rows: 56

> Contention/latency numbers are only load-bearing on `--backend s3`.

### T10 — assembly code: CPU per staged tile, wall per run (median of passes)

| arm | worker CPU s | vs C0 | write CPU s | vs C0 | wall s, whole run | stored GB |
|---|---|---|---|---|---|---|
| v2_b2_zstd1_s256 | 7.24 | 0.91 | 3.13 | 0.81 | 42 | 25.1 |
| v2_b2f_bandfirst_zstd1_s256 | 9.34 | 1.17 | 5.22 | 1.35 | 49 | 20.0 |
| v2_c0 | 7.97 | 1.00 | 3.87 | 1.00 | 52 | 26.0 |
| v2_c2_zstd1_s256 | 7.17 | 0.90 | 3.14 | 0.81 | 45 | 25.1 |
| v2_c2f_bandfirst_zstd1_s256 | 9.31 | 1.17 | 5.20 | 1.34 | 52 | 20.0 |

### T10 — lossless check against C0

| arm | stores | shards | mismatched array-shards |
|---|---|---|---|
| v2_b2_zstd1_s256 | t10asm/p0 | 64 | 0 |
| v2_b2f_bandfirst_zstd1_s256 | t10asm/p0 | 64 | 0 |
| v2_c2_zstd1_s256 | t10asm/p0 | 64 | 0 |
| v2_c2f_bandfirst_zstd1_s256 | t10asm/p0 | 64 | 0 |

## The request counter against S3's own count (`v2c`, from the verification box's log)

One cold 16-d 8 × 8-tile pull per arm ran alone on the prefix a temporary S3 request-metrics filter
covered, three minutes apart. CloudWatch's `GetRequests` (1-minute sums) against the counter:

| arm | counter | S3 GETs | icechunk's split, predicted |
|---|---|---|---|
| v2_c2_zstd1_s256 | 371 | 591 | most of the 12–16 MiB merged ranges in two parts |
| v2_c0 | 131 | 1,199 | 64 whole-shard reads × 18 parts + 64 index reads = 1,216 |

The counter counts zarr's requests, which is what the request plans compare. icechunk then splits
each into parts of about 12 MB, at most 18 per object (its `StorageConcurrencySettings` defaults),
so S3 sees more GETs for large reads in every layout, and the most for C0's whole-shard reads.

## The published v1.1 store on both clients (`v2c-psb`, `v2c-psb2`)

`scripts/diagnostic/published_store_read_bench.py --anonymous --zone 33N --year 2025` (200 probes),
on one r7i.4xlarge in us-west-2. Both clients ran back to back, new first in `v2c-psb`, old first in
`v2c-psb2`. Cold figures at concurrency 10 / 64 / 128:

| | zarr 3.2.1 + icechunk 2.1.1 | zarr 3.4.1 + icechunk 2.3.0 |
|---|---|---|
| point p50, new first | 344 / 313 / 286 ms | 468 / 341 / 302 ms |
| point p50, old first | 290 / 282 / 278 ms | 277 / 284 / 269 ms |
| tile 1000² × 128 | 56.7–67.8 MB/s | 275–339 MB/s |
| band subset 512² × 8 | 5.0–5.1 MB/s | 7.5–12.7 MB/s |
| bulk 4096² × 128 | 720–794 MB/s | 741–779 MB/s |
| MB per point | 7.96–8.12 | 7.90–8.13 |

Whichever client ran first paid the higher point latency. Region reads are what the newer zarr
speeds up.

## Microbenchmarks

**The Dask graph of a lazy whole-zone open** (laptop, metadata-only local stores the size of the
largest v1.1 zone: 456 × 33 tiles, nine years). Opening is free in every layout; the task count is
what a read builds at xarray's default chunks, one per inner chunk.

| layout | inner chunk | chunks, whole zone | tasks for one year at 16 dims, after culling | graph build |
|---|---|---|---|---|
| C0 | 256² × 128 | 8,667,648 | 1,926,145 | 9.5 s |
| any band-16 layout | 512² × 16 | 17,335,296 | 481,537 | 5.8–6.9 s |

A 16-d reader builds a quarter of C0's tasks; a 128-d reader builds twice C0's.

**Producing a band-first tile in inference.** Each arm mirrors the inference write path for one
2,048² tile (585 batches of 7,168 pixels, int8 quantisation on the GPU). It ran on a g6.xlarge
(NVIDIA L4, torch 2.14.1+cu130), median of three repeats. Pixel order was random, which is
pessimistic for the column scatter; the dataset's bucketed order cut it about 20% on a laptop.

| arm | per tile |
|---|---|
| today's row slot and row scatter, band-last | 0.39 s |
| transpose on the GPU, then a column scatter into a band-first tile | 7.75 s |
| today's path, then one CPU transpose of the finished tile | 2.92 s |

**The staging writer's headroom, from the Iowa v2 inference runs' actor logs** (both arms of the
#242 pair, 786 tiles, band-last staging; `Staging write … completed` and `Deferred write … blocked
collection` lines in `/ec2/global-tessera-dev/ray`, each counted once):

| | |
|---|---|
| tile time, median | 164–166 s (157–158 s of it inference), GPU 100% (median) |
| staged upload on the writer thread, median / p95 / max | 16.6 / 21.8 / 24.8 s |
| tiles whose next tile waited on the upload | 44 of 786 (5.6%), 408 s in all, median 8.4 s |
| average wait per tile | 0.5 s, about 0.3% of tile time |

**Writing the staged tile band-first instead** (laptop, zarr 3.4.1, memory store, no compression,
one 2,048² × 128 tile): 0.05 s band-last; 0.63 s band-first at 128 × 256² chunks and 1.00 s at
16 × 512² chunks, through a transposed view of the band-last buffer, with no full-tile copy.

**Reader-side costs per 2,048² tile** (laptop, one core unless noted):

| | 16-d | 128-d |
|---|---|---|
| reassemble 512² × 16 pieces, band-last | 0.004 s | 0.153 s |
| reassemble 512² × 16 pieces, band-first | 0.006 s | 0.056 s |
| transpose band-first → `(y, x, band)`, numpy | 0.050 s | 1.24 s |
| same, torch on 10 threads | | 0.17 s |

**Why band-first compresses better**, on fully valid 512² × 16 chunks of the four Iowa tiles:

| | band-last | band-first |
|---|---|---|
| zstd level 1 | 3.97–4.01 MB | 3.22–3.41 MB |
| entropy per 128 KB stretch | 7.54–7.62 bits per byte | 6.11–6.48 bits per byte |
| mean jump between neighbouring bytes | 55–63 | 3.3–4.7 |
| bytes equal to the one before | 0.3–0.5% | 12–16% |

Raw chunks are 4.19 MB; both orders' whole-chunk entropy is the same 7.59–7.64 bits per byte. zstd
codes each block's literals with their own table, so the per-stretch entropy predicts both sizes.
Storing each band-first value as its difference from its row neighbour before zstd took one chunk
from 3.22 MB to 2.20 MB; delta is not a Zarr core codec.
