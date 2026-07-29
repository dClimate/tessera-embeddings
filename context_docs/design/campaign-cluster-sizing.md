# Campaign cluster sizing — how the world's UTM zones divide across N Ray clusters

Authoritative basis for choosing `max_parallel_clusters` and `max_parallel_ingest`
on the chained (`fill_strategy="chained-clusters"`) campaign. Two halves: how the
work **balances** across clusters, and whether the **commit** path to the global
Icechunk store constrains how many you can run. Measured against the **real**
coverage bitmaps, not synthetic weights, because every conclusion here depends on
the actual distribution of land: a handful of huge continental zones and a long
tail of islands.

Reproduce any of it:

```bash
TE_CLUSTERS=8,16,24 uv run pytest tests/unit/test_cluster_balance.py -k report -s
```

The diagnostic drives the shipped `_partition_by_live_tiles` and the shipped
densest-first sort, so it reports what a campaign would actually do rather than a
model of it. Counts are snapshotted in `tests/unit/zone_density.py`; the same
module carries the recipe for refreshing them after a mask rebuild.

**Provenance of the numbers:** `s3://global-tessera-inputs-dev/masks/global.icechunk`,
built 2026-07-24 from `s3://tessera-embeddings/v1.1/global_0.1_degree_tiff_all/`,
registry sha256 `5ea80dd9…c794e`. This is the **dev** coverage store — at the time
of writing the production bucket has no mask under `masks/`. The distribution of
land does not change, but regenerate the snapshot once production is built.

## What the planet looks like

| | |
|---|---|
| UTM zones with any land | **112** of 120 |
| All-ocean zones (never dispatched) | 8 — `10S 11S 13S 14S 27S 44S 45S 46S` |
| Total live tiles (2048 px) | **360,953** |
| Largest single zone | `35N`, 9,132 tiles — **2.5%** of all land |
| Ten densest | `35N 38N 47N 48N 36N 34N 37N 33N 32N 19S` (9,132 → 7,951) |

## The three properties the design rests on

All three are *consequences* of code that never states them outright — the
longest-processing-time assignment in `_partition_by_live_tiles` and the
densest-first sort in `fill_zones_sequential`. They are asserted in
`tests/unit/test_cluster_balance.py` so a refactor cannot quietly lose them.

1. **Every cluster opens on a dense zone.** All cluster totals start at zero, so
   the N densest zones of a year are dealt one to each of the N clusters. This is
   what lets a cluster request GPUs after a *single* zone has ingested instead of
   waiting out its whole list.
2. **Every cluster runs dense → sparse.** The autoscaled fleet then only ever
   shrinks (no mid-run worker relaunch) and the cheap island zones land at the tail.
3. **Clusters finish together.** Near-even total work, so no cluster is still
   grinding after the others have released their fleets.

## At the shipped default of 8 clusters

| | |
|---|---|
| Tiles per cluster | 45,117 – 45,123 (**0.0%** spread — six tiles out of forty-five thousand) |
| Zones per cluster | 13 – 15 |
| Opening zone | 8,593 – 9,132 tiles |
| Opening zone as a share of its cluster's work | **~20%** |
| Ingest window per cluster | 5 zones (`max_parallel_ingest` 40 ÷ 8) |

One cluster in full, in the order it will work it:

```
35N:9,132  20S:7,004  50N:6,917  16N:5,805  29N:5,514  18S:3,787  50S:2,915
23N:1,402  05N:1,150  07N:850  32S:386  25S:182  40S:58  09S:10  31S:6
```

That 20% is the margin the GPU-start rule spends: inference on the opening zone
has to outlast the ingest of the five zones queued behind it. Inference is slower
than ingest in almost every case, so it does.

## Scaling the cluster count

| Clusters | Even share | Largest zone ÷ share | Balance spread | Smallest opener |
|---:|---:|---:|---:|---:|
| 1 | 360,953 | 3% | 0.0% | 9,132 |
| 2 | 180,476 | 5% | 0.0% | 9,100 |
| 4 | 90,238 | 10% | 0.0% | 8,803 |
| **8** | **45,119** | **20%** | **0.0%** | **8,593** |
| 12 | 30,079 | 30% | 0.3% | 7,808 |
| 16 | 22,560 | 40% | 0.6% | 7,004 |
| 20 | 18,048 | 51% | 0.7% | 6,694 |
| 24 | 15,040 | 61% | 7.9% | 6,327 |
| 32 | 11,280 | 81% | 2.3% | 5,590 |
| 40 | 9,024 | 101% | 5.2% | 4,715 |
| 60 | 6,016 | 152% | 108% | 1,441 |

**Headroom to 16 is free.** 0.6% imbalance, every cluster still opening on a
7,000-tile zone. Doubling the clusters roughly halves the year's wall clock at
the same total GPU-hours, so 16 is the obvious move if wall clock matters more
than blast radius.

**Degradation starts around 20–24, not at the theoretical knee.** The arithmetic
limit is at 40, where a perfectly even share equals the largest single zone and
that zone alone becomes the critical path — a cluster cannot be handed a fraction
of a zone. But balance is already visibly worse at 24 (7.9%), because the split is
being squeezed by the *several* largest zones, not just the one.

**Degradation is not monotonic.** 24 clusters (7.9%) is worse than 32 (2.3%). That
is an artefact of greedy assignment, not a bug — but it means a cluster count
picked by intuition can land somewhere worse than a larger number would.

## The small-zone fleet-fill worry is a non-issue

`runners/sequential_fill.py` carries a KNOWN LIMITATION: only `look_ahead + 2`
zones can have tiles in flight, so a cluster of zones each far smaller than the
fleet can leave actors idle. On real data this never bites.

| | |
|---|---|
| Zones smaller than a 20-actor fleet | 9 zones, 71 tiles = **0.02%** of all work |
| Zones under 100 tiles | 18 zones, 473 tiles = 0.13% |
| Zones under 500 tiles | 29 zones, 3,323 tiles = 0.92% |
| Last five zones of a typical cluster | ~1% of that cluster's work |

The taper is real and costs essentially nothing. Do not spend effort decoupling
admission from fleet-fill parallelism on this evidence.

## Commits to the global store: nowhere near the limit

Cluster count also sets how many writers can hit the global Icechunk store at
once, so the second half of the sizing question is whether the commit path
constrains it. It does not, by three orders of magnitude — but the reasoning is
worth recording because the headroom is not obvious and the protection is not
self-enforcing.

### Only one commit site is in the hot path

Three places commit to the global store. Everything else in the package —
mosaic writes, the ingest marker, the assessed-window attribute, the land mask,
the single-ROI embeddings path — writes to a *different* repository and cannot
contend.

| Site | When | Volume |
|---|---|---|
| `write_year_shards` (end of assembly) | Every live zone-year | **112 per campaign year** |
| `mark_zone_year_empty` | All-ocean cells, in pre-cluster triage | 8 per year, before Ray exists |
| `seed_zone_groups` | Once, at seeding | 1 ever |

Tags are not commits: `create_tag` points at an existing snapshot and never
advances the branch, so zone-year and year-milestone tags contend with nothing.

### The ceiling is structural: one commit per cluster

Each cluster's trailing assembly runs on a `ThreadPoolExecutor(max_workers=1)`,
so a cluster can have exactly one assembly — hence one commit — in flight. N
clusters means at most N committers. That is a property of the code, not a
convention.

The one caveat: a cluster has a *second* thread that can commit. The feeder
commits inside `plan()` when a cell turns out already-complete or all-ocean,
which is why `_PrefectCommitGate` keeps its state in a per-thread stack. The
flow's triage removes those cells before the stream starts, so a terminal plan
mid-stream means the store changed underneath the run. Rare, but it makes the
true ceiling **2N**, not N — which is why the gate is set to N rather than 2N and
is expected to queue occasionally. That is the gate doing its job: it is a bound,
not an operating point, and a queued commit costs seconds against zones that run
for hours.

### The measured storm curve

From ADR-008 run 1, committing to distinct zone groups:

| Simultaneous committers | Rebase retries | Commit wall |
|---:|---:|---:|
| 1 (uncontended) | 0 | 0.2 s |
| 2 | 0.5 | 0.5 s |
| 8 | 3.5 | 1.3 s |
| 16 | 7.5 | 2.2 s |
| 120 | 58 | 15 s |

Those five are the measured points; `retries ≈ (N−1)/2` fits them well enough to
interpolate. Retries scale with N, so *aggregate* wasted work scales with N², and
the recorded firm constraint is 4–8 simultaneous committers. Contention is on the
branch-tip compare-and-swap — every commit re-serialises the repo-global snapshot
file — so it depends only on how many writers race, never on which zones they
touch.

### Duty cycle: the reason it never bites

Combining the tile counts above with the measured fleet throughput from
[`inference_gpu_saturation_profile_2026_07.md`](inference_gpu_saturation_profile_2026_07.md)
(~15.3K px/s per worker, fleet-overall, L40S) at 20 actors per cluster:

| Clusters | GPUs | Wall per campaign year | Avg gap between commits | Fleet commit duty cycle |
|---:|---:|---:|---:|---:|
| 1 | 20 | 57.3 d | 736 min | 0.0005% |
| 4 | 80 | 14.3 d | 184 min | 0.0018% |
| **8** | **160** | **7.2 d** | **92 min** | **0.0036%** |
| 16 | 320 | 3.6 d | 46 min | 0.0072% |
| 24 | 480 | 2.4 d | 31 min | 0.0109% |
| 40 | 800 | 1.4 d | 18 min | 0.0181% |

A campaign year is 1.51 Tpx and ~27,500 GPU-hours; all nine years are ~247,000
GPU-hours and 1,008 commits. Against Icechunk v2's "tens of thousands of commits
per repo" target, the whole backfill is two orders of magnitude inside. Each
commit carries ~10⁵–10⁶ chunk refs against a ~7×10⁷ panic threshold (icechunk
\#1558) — two more orders of magnitude.

**The implication for sizing: the commit path does not constrain the cluster
count anywhere in this table.** At 8 clusters the fleet spends four thousandths
of a percent of its time committing, and a 1.3-second contended commit against a
35-hour zone is invisible. Balance (above) is what should decide the number, not
commits.

### Two things that are NOT automatically safe

**The cap's value is now derived, not configured** (it was an open gap when this
doc was first written). `commit_limit_name` is passed to the children as a *name*,
and the campaign upserts its VALUE at preflight to
`min(max_parallel_clusters, MAX_SIMULTANEOUS_COMMITTERS)` — see `_upsert_limit`.
Both halves earn their place: a cluster's trailing assembly is single-threaded, so
more slots than clusters is a number that could never bind, and the run-1 curve
caps it at 8 however many clusters run. It is deliberately NOT a parameter — an
operator-set number is exactly the thing that drifts from the server's.

The gate is also `strict=True`, so a limit that somehow does not exist stops the
fill loudly at its first commit rather than letting every cluster storm the
branch. `commit_limit_name=""` disables the gate entirely; the run-1 storm makes
that a bad idea outside a single-cluster test.

**Concurrent zone-years are safe; concurrent years OF THE SAME ZONE are not.**
Commits to different groups rebase cleanly — run 1 found zero unresolvable
conflicts at every N up to 120. But two fills of the same zone both rewrite that
group's `years_complete`/`runs` attrs, and `ConflictDetector` cannot auto-merge
attribute conflicts, so the loser raises `RebaseFailedError` (see the
`write_year_shards` concurrency contract). The constraint is **zone**-disjointness,
not year-disjointness. Today's year-serial outer loop guarantees it, but more
strongly than necessary: 33N-2025 alongside 34N-2024 is perfectly safe. Running
years concurrently needs a scheduler that guarantees zone-disjointness, or a
restructuring where a cluster owns a zone and works all nine of its years.

## Known limitation: the balance is spatial, not temporal

Zones are balanced on **land area**, not on observation count. A zone with the
same tile count but twice as many usable dates costs more inference, and nothing
in the split accounts for that. The area balance at 8 clusters has six tiles of
spread, so there is a great deal of headroom to absorb it — but if clusters ever
finish at visibly different times despite this document saying they should not,
**temporal depth is the first thing to check**, not the partitioner.

## Why the diagnostic lives in `tests/`

It drives `_partition_by_live_tiles`, which is inside the Prefect layer, so it
cannot be imported from a `profiling/` console script without breaking the
no-prefect-outside-the-prefect-layer rule. Printing is its output contract, hence
the scoped `T201` exemption in `ruff.toml` alongside the profiling tools'.

## Related

- [ADR-008 — global store architecture](../decisions/008-global-store-architecture.md),
  D5/D6: the run-1 commit-storm experiment the concurrency numbers above come from,
  and why one commit per zone-year is the unit.
- [ADR-011 — campaign-triggered per-zone ingestion](../decisions/011-campaign-zone-ingestion.md):
  the ingest cap, the GPU-start rule, and why the density ordering is load-bearing.
- [`orchestration/prefect/README.md`](../../src/tessera_embeddings/orchestration/prefect/README.md):
  the operational view of both fill strategies.
- [`runners/sequential_fill.py`](../../src/tessera_embeddings/orchestration/runners/sequential_fill.py)
  module docstring: the canonical statement of how the chained stream keeps a fleet busy.
