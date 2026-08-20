# What the depth line refuses, counted per pixel — 2026-08-14

> **The line is now 15, not 25 (decided 2026-08-17).** The candidate table in §2 is what chose it:
> 15 retains **94.1%** of pixels pixel-weighted against **79.2%** at 25, and leaves **no** cell under
> half or under a tenth. Every figure below stands as measured — pricing each candidate was always
> the point — but read the paragraph and §3 below as describing the *former* line. The decision, and
> the reproducibility cost accepted with it, are in
> [`minimum-optical-depth-plan.md`](minimum-optical-depth-plan.md).

At the 25 line, this is what it costs, measured by counting the pixels rather
than by inferring from window means: **the median measured cell keeps 92% of the pixels a
rule-free fill would have published**, five of forty keep under half, and two keep under a
tenth.

It settles a question the earlier evidence could not. The census
([`optical_depth_census_2026_08.md`](optical_depth_census_2026_08.md)) measures the *input*
at ~110 km from the catalogue; the legibility report
([`window_legibility_vs_depth_2026_08.md`](window_legibility_vs_depth_2026_08.md)) measures
whether a window can be read. Neither answers "how much of a cell disappears", and the
answer decides whether a published cell is worth publishing.

---

## 1. The instrument

Every filled cell already carries a per-pixel `s2_obs_count` array, so the retained share is
a direct count. The denominator is **pixels with any optical observation at all**, because
those are the ones a fill without the rule would embed. Retention therefore reads as *of
what we would have published, how much do we keep*.

Sampled, not exhaustive — a zone is roughly 891,000 × 68,000 px at 10 m. One 256-px chunk
per sampled shard, aimed at land by the same `window_origin` the validator uses, spread
evenly over the written shards with `np.linspace`. Forty chunks per cell, 82.8 M pixels over
**40 cells in 26 zones**, read from the dev global store.

Two corrections are part of the result:

- **The first version sampled `range(0, n, step)`, which degenerates to "the first N shards"
  whenever the step rounds to 1.** Shard order is row-major, so on a zone whose depth tracks
  latitude that samples one end of a gradient. It read 1.4% retention for 26S/2021 where the
  full sample reads 8.5% — a factor of six, in a table where every row looked plausible.
  Caught only by checking it against a figure the cell's own published validation already
  implied.
- **Window-mean depth, used as a proxy for weeks, is withdrawn.** A window whose mean falls
  below the line was called gutted, which conflates losing 40% of its pixels with losing
  99%. It overstated the cutoff's cost by roughly four times, and in the other direction it
  called 58S/2021 thin at a mean of 25.2 where the exact median pixel has 43 observations.

The corrected sampler was then checked against figures produced by a different code path
entirely: the mean of pixels at or above 25 in the pre-rule store reproduces the newly
filled cells' independently reported means — 29.58 against 29.75 for 26S/2021, 36.28 against
36.37 for 47S/2021. Two agreements within 0.2 observations.

## 2. The cost of each candidate line

| line | median cell retains | pixel-weighted mean | cells under 50% | cells under 10% | worst cell |
|---|---:|---:|---:|---:|---:|
| 20 | 96.6% | 87.7% | 3 | 0 | 26.9% |
| **25** | **91.9%** | **79.2%** | **5** | **2** | **2.3%** |
| 30 | 86.0% | 70.9% | 10 | 4 | 0.0% |

Moving 25 → 30 costs about six points of median retention and **doubles the number of cells
that lose more than half their pixels**, from five to ten. Moving 25 → 20 recovers five
points but leaves in the band where two independent embeddings of the same ground disagree
about twice as much as the pipeline's own floor; that trade is argued in the legibility
report, not here.

## 3. Per cell

Sorted by what survives at 25. `median obs` is the median observation count over the same
sample, so a cell's depth and its loss can be read together.

| cell | px sampled | median obs | keeps at 20 | **keeps at 25** | keeps at 30 |
|---|---:|---:|---:|---:|---:|
| 29S/2021 | 393,216 | 20 | 0.648 | **0.023** | 0.000 |
| 26S/2021 | 1,638,400 | 18 | 0.388 | **0.085** | 0.039 |
| 32S/2022 | 2,621,440 | 15 | 0.269 | **0.133** | 0.073 |
| 26S/2022 | 1,769,056 | 20 | 0.561 | **0.233** | 0.096 |
| 32S/2021 | 2,621,432 | 16 | 0.416 | **0.291** | 0.152 |
| 47S/2020 | 2,621,440 | 26 | 0.696 | 0.555 | 0.408 |
| 17S/2022 | 2,621,440 | 28 | 0.655 | 0.555 | 0.479 |
| 17S/2021 | 2,615,052 | 30 | 0.754 | 0.605 | 0.521 |
| 47S/2022 | 2,621,440 | 28 | 0.861 | 0.655 | 0.405 |
| 48S/2021 | 2,621,440 | 28 | 0.961 | 0.719 | 0.410 |
| 47S/2021 | 2,621,440 | 31 | 0.859 | 0.727 | 0.549 |
| 49S/2021 | 2,621,440 | 30 | 0.928 | 0.736 | 0.521 |
| 47S/2025 | 2,621,440 | 33 | 0.902 | 0.754 | 0.600 |
| 39S/2021 | 2,621,440 | 37 | 0.959 | 0.826 | 0.724 |
| 58S/2022 | 2,621,440 | 39 | 0.906 | 0.845 | 0.776 |
| 57S/2022 | 2,621,440 | 41 | 0.960 | 0.860 | 0.797 |
| 16S/2022 | 1,114,112 | 33 | 0.934 | 0.902 | 0.708 |
| 15S/2024 | 2,621,348 | 43 | 0.923 | 0.907 | 0.855 |
| 58S/2025 | 2,621,440 | 40 | 0.950 | 0.909 | 0.817 |
| 40S/2024 | 983,000 | 49 | 0.936 | 0.919 | 0.893 |
| 30S/2023 | 262,144 | 29 | 0.999 | 0.920 | 0.431 |
| 59S/2022 | 2,621,440 | 42 | 0.983 | 0.937 | 0.866 |
| 02N/2022 | 2,621,440 | 52 | 0.993 | 0.953 | 0.920 |
| 02N/2021 | 2,621,440 | 55 | 0.972 | 0.960 | 0.953 |
| 53N/2021 | 2,621,440 | 63 | 0.988 | 0.961 | 0.914 |
| 41S/2023 | 393,216 | 44 | 0.970 | 0.962 | 0.951 |
| 57S/2021 | 2,621,440 | 40 | 0.983 | 0.963 | 0.924 |
| 41S/2022 | 393,216 | 51 | 0.997 | 0.965 | 0.962 |
| 59S/2021 | 2,621,440 | 41 | 0.984 | 0.966 | 0.915 |
| 58S/2021 | 2,621,440 | 43 | 0.989 | 0.967 | 0.919 |
| 12S/2021 | 524,288 | 48 | 0.994 | 0.984 | 0.950 |
| 16S/2021 | 1,114,112 | 38 | 1.000 | 0.987 | 0.923 |
| 37N/2021 | 2,621,440 | 58 | 0.999 | 0.991 | 0.979 |
| 03N/2021 | 2,621,440 | 62 | 0.999 | 0.994 | 0.981 |
| 23N/2021 | 2,555,904 | 70 | 0.999 | 0.998 | 0.997 |
| 38N/2021 | 2,621,440 | 67 | 1.000 | 1.000 | 0.999 |
| 06N/2021 | 2,621,440 | 66 | 1.000 | 1.000 | 1.000 |
| 09S/2021 | 655,360 | 49 | 1.000 | 1.000 | 1.000 |
| 30S/2022 | 262,144 | 37 | 1.000 | 1.000 | 0.995 |
| 60N/2020 | 2,621,440 | 71 | 1.000 | 1.000 | 1.000 |

The loss is concentrated, not smeared. Twenty-six of forty cells keep 90% or more at the 25
line; the entire cost sits in five cells of three zones — 26S, 29S and 32S — plus a middle
group around 47S, 17S, 48S and 49S that loses a quarter to a half.

## 4. What these figures are not

- **Not a campaign-wide estimate.** The store's zones were chosen purposively for the depth
  study and lean thin. The pixel-weighted refusal here at the 30 line is 29.1%, against
  18.4% of shard-years for the same line in the global census — so this sample is roughly
  half again as thin as the world. **Campaign-wide loss will be lower than every figure
  above.**
- **Not exact per cell.** Forty chunks resolve a cell to a few points when its depth is
  uniform and to rather less when it is not; the large heterogeneous cells carry something
  like five to eight points. The medians and the counts over forty cells are the load-bearing
  numbers, not any single row.
- **Not a statement about usefulness.** This counts pixels. Whether a cell that keeps 8.5%
  of its pixels is worth publishing is a decision, and the argument for the line's *position*
  rests on reproducibility, which is measured elsewhere.

## 5. The operational consequence, seen in the rehearsal

26S/2021 was filled and published at 8.5% retention, and its per-cell validation passed with
no blocking finding. Tile-level coverage read 25 of 27 live tiles written — a cell can look
93% covered and hold a twelfth of its pixels. Two of its seam checks came back UNAVAILABLE
precisely because so little survived that too few boundaries could be compared.

So a cell this thin is not caught by the gate, and was never meant to be: refusal is the rule
working. What the gate does not do is tell an operator that the cell is nearly empty. The
verdict now carries `coverage.embedded_fraction_of_land` for that purpose — but read it
beside `written`/`live`, since it is computed over windows drawn from **written** tiles and
so cannot see a tile that was skipped whole.
