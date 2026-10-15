# Multiscale levels: pooled statistics in the v2 store

**Status: plan, not built.** Scoped on 2026-10-12 for the v2 global store. The implementation PR is
"Implement Multiscales". This document records what was agreed, the evidence behind it, and what
is still open. Once the design settles, its decisions move into an ADR.

## The agreed spec

```
one 2048-px tile (= one shard = one inference tile), one year, one zone group

level  factor  cells per tile   cell      what each cell holds
  0       1×    2048 × 2048     10 m      embeddings + scales            (unchanged)
  1       8×     256 × 256      80 m   ┐
  2      16×     128 × 128     160 m   │  per band, over the valid 10 m pixels in the cell:
  3      32×      64 × 64      320 m   │    min, max, mean, std, five percentiles
  4      64×      32 × 32      640 m   │  each re-quantized: 128 × int8 + one float32 scale
  5     128×      16 × 16     1.28 km  │  plus a valid-pixel count
  6     256×       8 × 8      2.56 km  │
  7     512×       4 × 4      5.12 km  │
  8    1024×       2 × 2     10.24 km  │
  9    2048×       1 × 1     20.48 km  ┘  + one Parquet row of band-pair correlations
```

- **Levels:** every power of two from 8× (80 m) to 2048× (20.48 km). There are no 20 m or 40 m
  levels.
- **Raster statistics, in the multiscales Zarr:** each level stores nine 128-band vectors per cell:
  min, max, mean, standard deviation and five percentiles. A valid-pixel count sits beside them.
  Every vector is re-quantized the same way as the base, as int8 per band plus a float32 scale per
  cell. All 128 bands are kept everywhere.
- **Correlations, in Parquet:** band-pair correlations (the upper triangle, as int8) go in a
  separate Parquet dataset beside the store, one row per tile-year. A reader rebuilds covariance as
  `corr[i, j] × std[i] × std[j]` using that tile's stored std.
- **Cost:** about **300 TB** of levels, 19% on top of the ~1.6 PB base, plus **~26 GB** of Parquet.
- **Scope:** v2 only. The base arrays do not change, and the v1.1 store is untouched.

## 1. What levels buy a reader, and what they do not

**A single chip costs the same at every level.** A read fetches whole 256 × 256 inner chunks. One
pixel costs 8.00 MB on the wire and a 100 × 100 patch costs 8.2 MB
([reading-the-published-store.md](reading-the-published-store.md) §4.1). A coarse level with the
same chunk shape therefore saves nothing for one lookup.

**The savings are for wide-area reads.** For one statistic over one year:

| Area | 10 m | 160 m | 1.28 km |
|---|---|---|---|
| Contiguous US (~8.1 M km²) | 10.7 TB | 42 GB | 650 MB |
| The whole land surface | ~178 TB | ~700 GB | ~11 GB |

Levels make continental and global analysis fit on a laptop. They also give patch-level users a
ready-made vector on a common ladder, which they would otherwise compute themselves by zonal
pooling.

## 2. Which scales

### 2.1 What other products and models use

Published embedding products as of December 2025 (Fang et al. 2026, Table I):

| Product | Kind | One vector covers | At 10 m |
|---|---|---|---|
| Earth Index (Earth Genome) | patch | 320 m | 32 × 32 px |
| Major TOM embeddings (ESA Φ-lab) | patch | 2.14–3.56 km | ~214–356 px |
| Clay embeddings | patch | 5.12 km | 512 × 512 px |
| Copernicus-Embed | patch | 0.25° (~28 km) | — |
| AlphaEarth, TESSERA, Presto | pixel | 10 m | 1 px |

- **Model chips cluster at 224–256 px**, which is 2.2–2.6 km at 10 m. Clay v1.5 takes a 256-px
  input with 8-px patches. Prithvi-EO-2.0 takes 224 px with 16-px patches, on 30 m HLS. OlmoEarth
  pretrains on 2.56 km samples.
- **Model tokens are 8–16 px**, which is 80–160 m at 10 m.
- **The grids users join to:** about 100 m (WorldPop, GHSL), 250 m–1 km (MODIS), 0.1° (ERA5-Land)
  and 0.25° (ERA5).

### 2.2 The ladder

| Level | Cell | Matches |
|---|---|---|
| 8× | 80 m | Clay's 8-px token |
| 16× | 160 m | 16-px tokens; ~100 m covariate grids |
| 32× | 320 m | Earth Index's patch, exactly |
| 64× | 640 m | EuroSAT's 64-px patch; ~500 m MODIS-class products |
| 128× | 1.28 km | ~1 km covariate grids |
| 256× | 2.56 km | the standard 256-px chip; Major TOM's range |
| 512× | 5.12 km | Clay's chip, exactly |
| 1024× / 2048× | 10.24 / 20.48 km | 0.1° and 0.25° climate grids; 2048× is one cell per tile |

**Why powers of two.** Every level nests exactly inside the 2048-px tile, which is also the shard
and the inference tile ([ADR 008](../decisions/008-global-store-architecture.md) D3). Each level can
therefore be computed from one tile held in memory, with no reads from neighbouring tiles, and
every statistic is exact, percentiles included.

**Why there are no 20 m or 40 m levels.** One statistic costs 25% and 6% of the base at those
levels; nine statistics would cost 225% and 56%. No product or common grid sits at either scale.
Anyone working that fine is working over a small area, where pooling the 10 m data is cheap.

**Why there is no 500 m level.** A 500 m level was considered for a 500 m FPAR covariate in
downstream yield modelling, and dropped for three reasons:
- A 50× factor doesn't nest: a tile holds 40.96 cells per side, so edge cells straddle tiles.
- Percentiles can't be merged across tiles, so straddling cells would need a second pass.
- MODIS and VIIRS "500 m" FPAR is really 463 m on a sinusoidal grid, so a 500 m level in UTM would
  not line up cell for cell in any case.

That work uses the 320 m and 640 m levels instead. A 512 m level is not on the ladder either,
because the ladder is 10 m × 2ⁿ.

## 3. Which statistics

### 3.1 The evidence

Corley et al. 2026 ([arXiv 2603.02080](https://arxiv.org/abs/2603.02080), v2, ICLR 2026 ML4RS
workshop) benchmark 11 training-free pooling methods on EuroSAT using AlphaEarth, OlmoEarth and
TESSERA pixel embeddings. Here are TESSERA's scores on the geographically disjoint split, which
measures how well a result transfers to a new region:

| Pooling | Values per cell | Linear probe | kNN (k = 5) |
|---|---|---|---|
| mean | 128 | 85.5 | 84.7 |
| mean + std | 256 | 93.9 | 86.7 |
| min, max, mean, std ("stats") | 512 | 94.3 | 89.3 |
| median + IQR | 256 | 91.7 | 85.7 |
| five percentiles | 640 | 92.8 | 86.2 |
| covariance | 8,256 | 94.4 | 89.8 |

### 3.2 What we store, and why

- **The four statistics: stored.** They match covariance at a sixteenth of its size. TESSERA gains
  the most of the three products from distributional pooling, because it embeds each pixel's time
  series with no spatial context.
- **Five percentiles: stored by choice.** They scored below the four statistics on this benchmark.
  We keep them because they preserve the shape of the distribution (skew, multiple modes), which
  moments alone can't recover, and the storage is acceptable.
- **Generalized mean and centre-weighted mean: not stored.**
  - Generalized mean needs an exponent the user chooses and non-negative inputs, and our
    embeddings are signed.
  - Centre weighting is defined relative to the user's own patch, not to a fixed grid cell.
- **Covariance: Parquet only, as correlations.** Covariance needs one value per pair of bands,
  8,256 per cell, against 128 for any other statistic.
  - At 80 m that is more than the data it summarises: a cell holds 64 pixels × 128 = 8,192 values.
  - As a raster ladder it would cost ~2.05 PB.
  - At one row per tile it costs ~26 GB.
  - Correlations fall between −1 and 1, so they quantize to int8 with a fixed 1/127 step and need
    no per-row scale. Quantizing covariance with a single per-cell scale would instead crush the
    small off-diagonal terms, because the variances on the diagonal are much larger.

**Caveats:**
- **It's one benchmark:** EuroSAT, 10 land-cover classes, 64 × 64-px (640 m) patches,
  classification.
- **TESSERA there is the public release, not v2.**
- **Yield regression may rank the methods differently.**
- **Cite version 2 of the paper.** Version 1 recommended generalized mean pooling and reports
  different figures.

## 4. Storage

**Assumptions:**
- The base is 1.60 PB across all nine years
  ([campaign-cost-model.md](../campaign/campaign-cost-model.md) §12c).
- One statistic is 128 int8 values plus a float32 scale, 132 B per cell, compressing about as
  poorly as the base (1.04×).
- A statistic at level k therefore costs 1.6 PB / k². This slightly overstates every figure,
  because a few percent of the base is observation-count arrays that are not pooled.
- The count band is negligible.

| Level | One statistic | **Agreed set (9 statistics)** | Covariance as a raster (not planned) |
|---|---|---|---|
| 80 m | 25 TB | **225 TB** | 1,540 TB |
| 160 m | 6.3 TB | **56 TB** | 385 TB |
| 320 m | 1.6 TB | **14 TB** | 96 TB |
| 640 m | 0.39 TB | **3.5 TB** | 24 TB |
| 1.28 km | 0.10 TB | **0.88 TB** | 6.0 TB |
| 2.56 km | 0.02 TB | **0.22 TB** | 1.5 TB |
| 5.12–20.48 km | 0.01 TB | **0.07 TB** | 0.5 TB |
| **Total** | **33 TB (2.1%)** | **300 TB (19%)** | **2.05 PB (128%)** |

- **The 80 m level is three quarters of the total.** Dropping its percentiles alone would save
  125 TB.
- **Correlation Parquet:** 8,128 B per row (128 × 127 / 2) × 3,229,545 tile-years (v1.1's count) ≈
  26 GB.
- **Price:** AWS Open Data sponsors the bucket. At S3 Standard list (about $22 per TB-month), 300 TB
  would be about $6,600 a month; that is the sponsor's figure, not ours.

## 5. Compute

**How this was measured:** one synthetic 2048 × 2048 × 128 int8 tile, numpy, one Apple M3 Max core
(§8). Cloud cores may be up to about 2× slower. The cost is not yet measured on the fill path.

| Work per tile | Time | Campaign (3.23 M tiles) |
|---|---|---|
| Convert int8 values back to float32 | 1.1 s | — |
| Four statistics at 80 m (count, sum, sum of squares, min, max) | 0.9 s | — |
| Four statistics at 160 m–20.48 km, merged up from 80 m | 0.02 s | — |
| **All levels, four statistics** | **~2 s** | **~2,000 core-hours** |
| Five percentiles, one level, from 10 m pixels | 10–14 s | — |
| **Five percentiles, all nine levels** | **~120 s** | **~110,000 core-hours** |
| Correlation, one tile (covariance via X^T X) | < 1 s on Apple's matrix unit; a few seconds on a typical core | — |

- **The four statistics merge exactly.** Count, sum, sum of squares, min and max combine across
  cells, so each coarse level is derived from the one below.
- **Percentiles do not merge.** Every level is computed from the 10 m pixels, which makes them roughly
  60× the cost of the four statistics.
- **CPU contention, not dollars, is the risk.** Both totals are small in money terms. The question
  is whether the fill has spare CPU for them (§6.1).

## 6. Implementation outline

### 6.1 Where pooling runs

Pooling works from the tile's int8 values and scales, dequantized. Inference batches pixels by
sequence length rather than by location, so the GPU never holds a spatially contiguous tile in
float. There are two candidate places:

- **Assembly workers.** They already read each staged tile to write its shard, and they are CPU
  workers, so pooling there takes nothing from the GPUs. Their memory is the bound (40–94 GiB peaks
  per task, [what-bounds-assembly](../assembly/what-bounds-assembly-2026-09-09.md)). A dequantized
  tile adds ~2 GiB unless pooling works in row bands.
- **Inference actors**, after a tile's buffers complete. These CPUs also feed the GPUs, so
  ~2 minutes of percentile work per tile could slow inference.

### 6.2 Layout

- **One child group per level inside each zone group**, carrying the zone's nine-year time axis.
- **One array per statistic,** so a reader who wants only the mean reads only the mean. Each array
  has a matching scale array. The alternative is one array with a statistic axis.
- **Name the pooled std distinctly.** `embedding_std` already exists in the single-ROI layout
  ([`store_layout.py`](../../src/tessera_embeddings/config/store_layout.py)) as a per-pixel model
  std, so it must not be reused.
- **Keep level arrays out of `CARRIED_VARS`.** That tuple is derived from the base layout, so any
  array added there gets copied by assembly as a per-pixel variable.
- **The count band needs uint32 from 256× upward.** A 2.56 km cell holds 65,536 pixels, which
  overflows uint16.
- **Empty and partial cells.** A cell with no valid pixel is fill. A partial cell (coast, refused
  pixels) holds statistics over its valid pixels, and its count says how many there were.
- **Numerics.** Accumulate sums in float64, because sum-of-squares cancellation in float32 is real
  over 4 M pixels. Re-quantize per cell with the existing per-row max-abs scheme
  (`quantization.quantize_rows`).

### 6.3 Writing the coarse levels

A tile contributes 256 × 256 cells at 8× and a single cell at 2048×. With the base geometry (256-px
inner chunks, 2048-px shards), one coarse-level shard spans many tiles. Assembly forks that write
different tiles of a zone-year would then collide on shared shard objects. The options are:

1. **Assign tiles to forks in blocks aligned to each level's shard footprint**, the way single-ROI
   assembly bands its work.
2. **Give fine levels a per-tile geometry and gather the coarse levels in the coordinator.** At 64×
   and coarser, a tile's results are ~1.6 MB, about 5 GB for a 3,000-tile zone-year.
3. **Run a separate per-zone-year pass after the base commit.** This is the simplest code, but it
   re-reads the base: ~1.6 PB of in-region reads across the campaign.

Chunk shape per level is open as well. At 2.56 km, a 256-px inner chunk covers 655 km, so one chip
lookup still fetches ~8 MB per statistic.

### 6.4 The multiscales convention

[zarr-conventions/multiscales](https://github.com/zarr-conventions/multiscales) is at v0.1 and
marked "Pilot". Breaking changes are expected before a v1, which is anticipated by the end of 2026.

- **Structure.** Each level is a `layout` entry with an `asset` path, plus optional `derived_from`,
  `transform` (a `scale` per axis, which may be fractional) and `resampling_method`. Levels SHOULD
  be child groups.
- **Our base arrays sit at the zone-group root, not in a child group.** Whether `layout` can point
  at the group itself as level 0 needs checking against the schema.
- **The convention assumes the same variables at every level.** Our levels hold statistics rather
  than `embeddings`. One option is to name each level's mean `embeddings` with its `scales`, so the
  mean ladder reads as an ordinary multiscale of the base variable, and add the other statistics as
  extra arrays.
- **Registration.** Register the convention in
  [`conventions.py`](../../src/tessera_embeddings/storage/conventions.py), the single place that
  writes `zarr_conventions`, and validate against the published schema; a resolving URL is not
  conformance.

### 6.5 The correlation Parquet

The correlation dataset sits beside the registry and is keyed the same way: a `tile` column, with
`zone` and `year` as hive partitions. Each row holds a 8,128-long int8 list in row-major
upper-triangle order.

It stays out of the registry itself. The registry is the lightweight coverage index every reader
filters ([`registry.py`](../../src/tessera_embeddings/storage/registry.py)), and these columns would
turn it into a 26 GB table. Like the registry, it is derivable from the store, so a wrong row can be
rebuilt.

### 6.6 Dependencies

- **v2 zone fill.** Zone fill does not run v2 yet ([campaign-cost-model.md](../campaign/campaign-cost-model.md)
  §13), and this work lands in the same fill.
- **Code identity.** v2 is a new store, so changes to the fill's code identity block nothing.

## 7. Open questions

1. **Which five percentiles.** Proposed: 10th, 25th, 50th, 75th and 90th. Min and max already
   cover the extremes, and this set gives median and interquartile range for free.
2. **Which level the correlations cover.** Assumed: tile level only. Adding 2.56 km would be ~200 M
   rows and ~1.6 TB.
3. **Where pooling runs** (§6.1). Measure CPU headroom on the real fill path first.
4. **How coarse levels are written** (§6.3), and the chunk shape per level.
5. **How the levels fit the multiscales convention** (§6.4): where level 0 sits, and variable naming.
6. **Whether the Parquet also carries each tile's float32 mean and std**, so it stands alone without
   the Zarr (+1 KB per row, ~3.3 GB).

## 8. Reproducing the timings

Use the repository venv with `VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1` for single-core figures:

```python
import numpy as np

rng = np.random.default_rng(0)
T, D = 2048, 128
scales = rng.random((T, T), dtype=np.float32) * 0.05
x = rng.integers(-127, 128, (T, T, D), dtype=np.int8).astype(np.float32) * scales[..., None]  # 1.1 s
v = x.reshape(T // 8, 8, T // 8, 8, D)                                   # 80 m cells
v.sum((1, 3)); np.square(v).sum((1, 3)); v.min((1, 3)); v.max((1, 3))   # 0.9 s
cells = v.transpose(0, 2, 1, 3, 4).reshape(-1, 64, D)
np.percentile(cells, [10, 25, 50, 75, 90], axis=1)                       # 14 s
c = x[:2000, :2000].reshape(40, 50, 40, 50, D).transpose(0, 2, 1, 3, 4).reshape(1600, 2500, D)
np.matmul(c.transpose(0, 2, 1), c)            # 0.4 s; the same flops as one tile's X^T X
```

## References

- Corley, Robinson, Becker-Reshef, Lavista Ferres. *From Pixels to Patches: Pooling Strategies for
  Earth Embeddings.* ICLR 2026 ML4RS. [arXiv 2603.02080](https://arxiv.org/abs/2603.02080) (v2).
  **The background reference for this work.**
- Fang, Stewart, Corley, Zhu, Azizpour. *Earth Embeddings as Products: Taxonomy, Ecosystem, and
  Standardized Access.* [arXiv 2601.13134](https://arxiv.org/abs/2601.13134).
- Earth Genome, [Embeddings for all](https://www.earthgenome.org/blog/embeddings-for-all) (320 m
  patches).
- [Clay v1.5 model specification](https://clay-foundation.github.io/model/release-notes/specification.html).
- [Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732); [OlmoEarth](https://arxiv.org/abs/2511.13655);
  [Copernicus-FM / Copernicus-Embed](https://arxiv.org/abs/2503.11849).
- [zarr-conventions/multiscales](https://github.com/zarr-conventions/multiscales) (v0.1, Pilot).
