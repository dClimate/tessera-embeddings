# Scale-test report - runs `run1` (T0-T7) and `d3`/`d3v2` (T8)

**This file aggregates more than one run, and the sections have different provenance.**
Read the header per section, not for the file:

| sections | run | revision | observations |
|---|---|---|---|
| T0-T7 | `run1` | git `c490616` | 898 |
| T8 | `d3` / `d3v2` — the sharding experiments E1-E4 ([ADR 008](../../../context_docs/decisions/008-global-store-architecture.md)) | not recorded here | 30 |
| | | | **928 total** |

- backend: **s3**, scale: **bench**
- icechunk **2.1.1**, zarr **3.2.1**

The 898 figure counts T0-T7 alone — it is the sum of the `n` column across those sections,
and T8 adds 30 on top. Attributing all 928 to `run1` at `c490616` would credit the sharding
measurements to a run that did not produce them.

> Contention/latency numbers are only load-bearing on `--backend s3`.

## Decision matrix (ADR-008)

| Decision | Evidence | Measured | Notes |
|---|---|---|---|
| D1 pre-alloc | data-chunks==0; shift conflict unresolvable | data chunks=0 confirmed; shift-vs-write conflict=unresolvable | escape hatch only |
| D2 chunk shape | T1 point p95/p50 by variant | best=c256_full; c256_full p95=193.272/p50=120.948ms; c256_sharded p95=199.146/p50=28.954ms; c500_band4 p95=219.233/p50=117.291ms; c384_full p95=333.982/p50=229.350ms; c500_full p95=338.621/p50=211.441ms | 256+full band; smaller=faster/point |
| D3 sharding | sharded vs full (p95/p50/write) | p95 199.146 vs 193.272ms (~tie); p50 28.954 vs 120.948ms; ~64x fewer objects. **The "write 2.8x slower" figure from this T1 sweep is SUPERSEDED** — it measured a chunkwise writer. `d3v2` (§t8) measured the shard-aligned + land-masked writer at **0.46x** the unsharded build, and that is what settled D3 in FAVOUR of sharding | [ADR-008 D3](../../../context_docs/decisions/008-global-store-architecture.md) |
| D4 manifest split | T2 per-year commit trend | commit 0.367s->0.353s (flat) | rising => icechunk #1600 |
| D5 one repo | T4 snapshot growth + T5 contention | snapshot 4973.000->38102.000B @120g; max retries=118.000 | kill: >2x serial or storms |
| D7 GC/hygiene | T6 GC objects/s + reclaimed | 19 objs @ 33.3/s, 8.405e+06B reclaimed | extrapolate to 10^8 objects |

## Per-test metrics

### t0

| phase | metric | n | min | median | max | unit |
|---|---|---|---|---|---|---|
| disjoint_region | commit_wall_s | 2 | 0.189 | 0.385 | 0.582 | s |
| disjoint_region | retries | 2 | 0 | 0.500 | 1.000 | count |
| disjoint_region | wall_s | 1 | 4.218 | 4.218 | 4.218 | s |
| same_chunk_detector | commit_wall_s | 2 | 0.150 | 0.217 | 0.285 | s |
| same_chunk_detector | retries | 2 | 0 | 0.500 | 1.000 | count |
| same_chunk_detector | wall_s | 1 | 3.952 | 3.952 | 3.952 | s |
| same_chunk_useours | commit_wall_s | 2 | 0.158 | 0.325 | 0.492 | s |
| same_chunk_useours | retries | 2 | 0 | 0.500 | 1.000 | count |
| same_chunk_useours | wall_s | 1 | 4.170 | 4.170 | 4.170 | s |
| xgroup_conflict | commit_wall_s | 3 | 0.175 | 0.459 | 0.786 | s |
| xgroup_conflict | retries | 3 | 0 | 1.000 | 2.000 | count |
| xgroup_conflict | wall_s | 1 | 5.291 | 5.291 | 5.291 | s |

### t1

| phase | metric | n | min | median | max | unit |
|---|---|---|---|---|---|---|
| build_c256_full | wall_s | 1 | 152.291 | 152.291 | 152.291 | s |
| build_c256_sharded | wall_s | 1 | 420.460 | 420.460 | 420.460 | s |
| build_c384_full | wall_s | 1 | 103.378 | 103.378 | 103.378 | s |
| build_c500_band4 | wall_s | 1 | 87.622 | 87.622 | 87.622 | s |
| build_c500_full | wall_s | 1 | 85.850 | 85.850 | 85.850 | s |
| read_c256_full | bytes_fetched | 6 | 4.234e+06 | 4.234e+06 | 4.234e+06 | bytes |
| read_c256_full | open_wall_s | 6 | 0.112 | 0.131 | 0.158 | s |
| read_c256_full | read_p50_ms | 6 | 116.804 | 122.813 | 178.942 | ms |
| read_c256_full | read_p95_ms | 6 | 175.765 | 197.640 | 377.439 | ms |
| read_c256_full | throughput_mbps | 24 | 8.073 | 157.645 | 1897.318 | MB/s |
| read_c256_full | wall_s | 1 | 1322.916 | 1322.916 | 1322.916 | s |
| read_c256_sharded | bytes_fetched | 6 | 1.647e+07 | 1.647e+07 | 1.647e+07 | bytes |
| read_c256_sharded | open_wall_s | 6 | 0.105 | 0.128 | 0.147 | s |
| read_c256_sharded | read_p50_ms | 6 | 28.943 | 29.172 | 57.706 | ms |
| read_c256_sharded | read_p95_ms | 6 | 176.509 | 204.488 | 250.426 | ms |
| read_c256_sharded | throughput_mbps | 24 | 13.303 | 202.071 | 2014.469 | MB/s |
| read_c256_sharded | wall_s | 1 | 562.735 | 562.735 | 562.735 | s |
| read_c384_full | bytes_fetched | 6 | 9.224e+06 | 9.224e+06 | 9.224e+06 | bytes |
| read_c384_full | open_wall_s | 6 | 0.114 | 0.150 | 0.203 | s |
| read_c384_full | read_p50_ms | 6 | 222.152 | 229.584 | 257.861 | ms |
| read_c384_full | read_p95_ms | 6 | 333.982 | 351.958 | 421.628 | ms |
| read_c384_full | throughput_mbps | 24 | 3.199 | 166.858 | 1593.499 | MB/s |
| read_c384_full | wall_s | 1 | 2221.985 | 2221.985 | 2221.985 | s |
| read_c500_band4 | bytes_fetched | 6 | 3.178e+07 | 3.178e+07 | 3.178e+07 | bytes |
| read_c500_band4 | open_wall_s | 6 | 0.104 | 0.140 | 0.166 | s |
| read_c500_band4 | read_p50_ms | 6 | 112.109 | 126.879 | 375.350 | ms |
| read_c500_band4 | read_p95_ms | 6 | 219.233 | 258.340 | 428.822 | ms |
| read_c500_band4 | throughput_mbps | 24 | 2.383 | 87.831 | 462.362 | MB/s |
| read_c500_band4 | wall_s | 1 | 1832.546 | 1832.546 | 1832.546 | s |
| read_c500_full | bytes_fetched | 6 | 1.637e+07 | 1.637e+07 | 1.637e+07 | bytes |
| read_c500_full | open_wall_s | 6 | 0.099 | 0.148 | 0.187 | s |
| read_c500_full | read_p50_ms | 6 | 206.407 | 220.539 | 233.393 | ms |
| read_c500_full | read_p95_ms | 6 | 338.621 | 368.485 | 401.158 | ms |
| read_c500_full | throughput_mbps | 24 | 3.904 | 104.179 | 2002.527 | MB/s |
| read_c500_full | wall_s | 1 | 2187.222 | 2187.222 | 2187.222 | s |

### t2

| phase | metric | n | min | median | max | unit |
|---|---|---|---|---|---|---|
| refs_sweep | commit_wall_s | 3 | 0.280 | 0.373 | 0.456 | s |
| refs_sweep | merge_wall_s | 3 | 2.760e-04 | 0.002 | 0.003 | s |
| refs_sweep | peak_rss_bytes | 3 | 2.232e+09 | 2.290e+09 | 2.292e+09 | bytes |
| refs_sweep | refs_committed | 3 | 1000.000 | 10000.000 | 12482.000 | count |
| refs_sweep | wall_s | 1 | 389.454 | 389.454 | 389.454 | s |
| split_config_ab | commit_wall_s | 9 | 0.197 | 0.282 | 5.554 | s |
| split_config_ab | manifest_count | 3 | 7.000 | 7.000 | 379.000 | count |
| split_config_ab | snapshot_bytes | 3 | 1536.000 | 1614.000 | 14150.000 | bytes |
| split_config_ab | wall_s | 1 | 841.994 | 841.994 | 841.994 | s |
| year_fill_trend | commit_wall_s | 9 | 0.261 | 0.323 | 0.411 | s |
| year_fill_trend | manifest_bytes | 9 | 184019.000 | 916035.000 | 1.647e+06 | bytes |
| year_fill_trend | manifest_count | 9 | 5.000 | 13.000 | 21.000 | count |
| year_fill_trend | snapshot_bytes | 9 | 1532.000 | 1868.000 | 2162.000 | bytes |
| year_fill_trend | wall_s | 1 | 1290.061 | 1290.061 | 1290.061 | s |

### t3

| phase | metric | n | min | median | max | unit |
|---|---|---|---|---|---|---|
| conflict_probe | retries | 1 | 1.000 | 1.000 | 1.000 | count |
| conflict_probe | wall_s | 1 | 142.106 | 142.106 | 142.106 | s |
| fill_and_verify | wall_s | 1 | 138.362 | 138.362 | 138.362 | s |
| prepend_loop | commit_wall_s | 8 | 0.317 | 0.399 | 0.576 | s |
| prepend_loop | manifest_bytes | 8 | 548352.000 | 3.275e+06 | 8.179e+06 | bytes |
| prepend_loop | wall_s | 1 | 8563.353 | 8563.353 | 8563.353 | s |
| seed_full_axis | manifest_bytes | 1 | 53946.000 | 53946.000 | 53946.000 | bytes |
| seed_full_axis | objects_listed | 1 | 2.000 | 2.000 | 2.000 | count |
| seed_full_axis | wall_s | 2 | 0.331 | 0.686 | 1.040 | s |
| verify_prepend | wall_s | 1 | 2.069 | 2.069 | 2.069 | s |

### t4

| phase | metric | n | min | median | max | unit |
|---|---|---|---|---|---|---|
| open_and_preload | open_wall_s | 3 | 0.152 | 0.232 | 30.870 | s |
| open_and_preload | wall_s | 1 | 34.110 | 34.110 | 34.110 | s |
| scale_groups | commit_wall_s | 3 | 0.186 | 0.244 | 0.325 | s |
| scale_groups | snapshot_bytes | 3 | 4973.000 | 19988.000 | 38102.000 | bytes |
| scale_groups | wall_s | 4 | 3.108 | 14.022 | 39.182 | s |

### t5

| phase | metric | n | min | median | max | unit |
|---|---|---|---|---|---|---|
| contention_n120 | commit_wall_s | 121 | 0.149 | 14.976 | 30.722 | s |
| contention_n120 | retries | 120 | 0 | 58.500 | 118.000 | count |
| contention_n120 | wall_s | 2 | 44.591 | 55.189 | 65.787 | s |
| contention_n16 | commit_wall_s | 17 | 0.152 | 2.168 | 4.269 | s |
| contention_n16 | retries | 16 | 0 | 7.500 | 15.000 | count |
| contention_n16 | wall_s | 2 | 8.039 | 9.742 | 11.444 | s |
| contention_n2 | commit_wall_s | 3 | 0.158 | 0.491 | 0.508 | s |
| contention_n2 | retries | 2 | 0 | 0.500 | 1.000 | count |
| contention_n2 | wall_s | 2 | 3.841 | 4.319 | 4.796 | s |
| contention_n32 | commit_wall_s | 33 | 0.148 | 4.379 | 8.406 | s |
| contention_n32 | retries | 32 | 0 | 15.500 | 31.000 | count |
| contention_n32 | wall_s | 2 | 13.252 | 16.618 | 19.985 | s |
| contention_n64 | commit_wall_s | 65 | 0.152 | 8.292 | 16.906 | s |
| contention_n64 | retries | 64 | 0 | 31.500 | 63.000 | count |
| contention_n64 | wall_s | 2 | 24.786 | 30.630 | 36.474 | s |
| contention_n8 | commit_wall_s | 9 | 0.164 | 1.278 | 2.228 | s |
| contention_n8 | retries | 8 | 0 | 3.500 | 7.000 | count |
| contention_n8 | wall_s | 2 | 5.630 | 6.431 | 7.231 | s |

### t6

| phase | metric | n | min | median | max | unit |
|---|---|---|---|---|---|---|
| build_mess | objects_listed | 1 | 105.000 | 105.000 | 105.000 | count |
| build_mess | wall_s | 1 | 11.901 | 11.901 | 11.901 | s |
| expire_and_gc | bytes_reclaimed | 1 | 8.405e+06 | 8.405e+06 | 8.405e+06 | bytes |
| expire_and_gc | gc_wall_s | 1 | 0.571 | 0.571 | 0.571 | s |
| expire_and_gc | objects_deleted | 1 | 19.000 | 19.000 | 19.000 | count |
| expire_and_gc | objects_listed | 2 | 94.000 | 102.500 | 111.000 | count |
| expire_and_gc | wall_s | 1 | 1.165 | 1.165 | 1.165 | s |
| rollback | retries | 1 | 0 | 0 | 0 | count |
| rollback | wall_s | 1 | 0.490 | 0.490 | 0.490 | s |

### t7

| phase | metric | n | min | median | max | unit |
|---|---|---|---|---|---|---|
| ramp | puts_per_s | 4 | 68.433 | 71.737 | 82.185 | count/s |
| ramp | slowdown_503_count | 4 | 0 | 0 | 0 | count |
| ramp | wall_s | 1 | 21.962 | 21.962 | 21.962 | s |

### t8

| phase | metric | n | min | median | max | unit |
|---|---|---|---|---|---|---|
| bytes_on_wire | bytes_fetched | 2 | 1.226e+06 | 4.981e+06 | 8.736e+06 | bytes |
| bytes_on_wire | read_p95_ms | 2 | 169.213 | 208.940 | 248.668 | ms |
| bytes_on_wire | wall_s | 1 | 98.603 | 98.603 | 98.603 | s |
| object_count | manifest_bytes | 3 | 9976.000 | 13299.000 | 17790.000 | bytes |
| object_count | objects_listed | 3 | 587.000 | 808.000 | 808.000 | count |
| object_count | wall_s | 1 | 0.464 | 0.464 | 0.464 | s |
| scattered_reads | read_p50_ms | 2 | 30.138 | 71.841 | 113.545 | ms |
| scattered_reads | read_p95_ms | 2 | 124.045 | 167.351 | 210.657 | ms |
| scattered_reads | wall_s | 1 | 95.227 | 95.227 | 95.227 | s |
| write_alignment | bytes_written | 3 | 3.463e+09 | 4.949e+09 | 1.256e+10 | bytes |
| write_alignment | commit_wall_s | 3 | 0.248 | 0.324 | 0.383 | s |
| write_alignment | objects_listed | 3 | 587.000 | 808.000 | 808.000 | count |
| write_alignment | wall_s | 4 | 10.927 | 27.491 | 68.188 | s |