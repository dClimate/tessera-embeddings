# Reading the published store: what a consumer needs, what it costs, and what would surprise them

**The finished global store needs no credentials.** The delivery bucket's policy grants `s3:Get*`
and `s3:List*` to every principal over TLS, so anyone can open
`s3://tessera-embeddings/v1.1/dclimate.icechunk`, enumerate all 120 UTM zone groups, and read
embeddings with no AWS account. Verified end to end with every AWS environment variable cleared and
the shared-credentials file pointed at an empty directory, so it is a statement about the public
path rather than about a profile that happened to be lying around.

This is the reader's counterpart to
[`writing-to-the-global-store.md`](writing-to-the-global-store.md), which covers how the bytes got
there. This one answers what somebody who did not run the campaign will ask: can I open it, is it
what was promised, how fast is it, and how do I find out whether my area is covered.

| § | question |
|---|---|
| 1 | the recipe, and the one thing that will not work |
| 2 | what is published, and what "complete" means |
| 3 | three ways to ask what is covered, and what each costs |
| 4 | measured read performance, in two regions, against what scoping predicted |
| 5 | the registry beside the store |
| 6 | what is missing, and what would surprise a consumer |
| 7 | re-running every measurement here |

All of it is measured against snapshot `QR7F41A6WYZ03VC92T6G`, the campaign's last, which landed
with cell 17S/2017 on 2026-09-10. The read figures in §4 were taken on 2026-09-11, after the store's
saved manifest preload was switched off; the figures that switch superseded are kept in §4.3,
labelled, because they are what justified it.

---

## 1. The recipe

```python
from tessera_embeddings.storage.global_store import open_global_repo
import zarr

repo = open_global_repo("s3://tessera-embeddings/v1.1/dclimate.icechunk",
                        region="us-west-2", anonymous=True)
session = repo.readonly_session(branch="main")
zone = zarr.open_group(session.store, mode="r")["33N"]

print(zone.attrs["years_complete"])          # which years this zone holds
vector = zone["embeddings"][8, 500_000, 30_000, :]   # one pixel, all 128 bands, 2025
```

That is the whole of it. Two details carry the rest of the document.

**The store is Icechunk, not plain Zarr.** `xr.open_zarr("s3://.../dclimate.icechunk")` does not
work and does not fail obviously — the URI is a repository, not a Zarr hierarchy. A reader has to
open the repository, take a session, and hand *that* session's store to Zarr or xarray.
`anonymous=True` is the whole credential story; without it the default AWS chain is used, which
works for anyone whose account has been granted access and fails confusingly for somebody who has
AWS configured for something else entirely.

**The time axis is nine preallocated slots, and an unfilled one reads back as fill.** All nine
2017–2025 timesteps exist in every zone group from the moment it was seeded, because unwritten
chunks cost nothing ([ADR 008](../decisions/008-global-store-architecture.md) D1). A read of a year
that was never filled succeeds and quietly returns the fill value. **`years_complete` is the
authority**, and it distinguishes what nothing else can: a year deliberately left empty — a zone
with no qualifying land that year — *is* in the list, while a year that never landed is absent.

Reading `zone.attrs` pulls the whole metadata document, including a `runs` entry per year carrying
that fill's run id, its input coverage per source store and the commit it was built from. Useful
provenance, and not small: a reader who only wants `years_complete` pays for all of it per zone.

## 2. What is published

One Icechunk repository, one `main` branch, 120 Zarr groups named `01N` … `60S`, each its own UTM
CRS. Per group:

| array | shape | dtype | inner chunk | shard |
|---|---|---|---|---|
| `embeddings` | (9, northing, easting, 128) | int8 | (1, 256, 256, 128) | (1, 2048, 2048, 128) |
| `scales` | (9, northing, easting) | float32 | (1, 256, 256) | (1, 2048, 2048) |
| `s2_obs_count`, `s1_asc_obs_count`, `s1_desc_obs_count` | (9, northing, easting) | uint16 | (1, 256, 256) | (1, 2048, 2048) |
| `s2_month_covered`, `s1_asc_month_covered`, `s1_desc_month_covered` | (9, northing, easting, 12) | int8 | (1, 256, 256, 12) | (1, 2048, 2048, 12) |

plus the coordinate arrays `time`, `time_bnds`, `northing`, `easting`, `band` and `month`.

**All 120 groups conform to the declared layout** — dtype, inner chunks and shards, per array,
against `config.store_layout.GLOBAL` with expectations clamped to each array's own shape exactly as
the writer clamped them. Zero departures. `scales` is sharded on the same grid as `embeddings`,
which was the second half of the sharding decision and the easier half to get wrong.

**Coverage, as delivered:**

| | |
|---|---|
| zone-years possible | 1,080 (120 zones × 9 years) |
| zone-years marked complete | **1,066** |
| zone-years never filled | **14** |
| shards holding embeddings, whole store | **3,229,545** |
| completion tags (`zone-<ZONE>-<YEAR>`) | 1,066 |
| cells marked complete but untagged, or tagged but unmarked | 0 |

The attribute and the tag are written in separate commits, so they *can* disagree; they do not. The
14 unfilled cells are 03S 2017/2018/2020/2021, 08S 2017/2018, 09S 2017, 24N 2017, 26S 2017,
29S 2017, and 31S 2018/2019/2020/2021 — every one a year where the optical archive over that zone's
land could not meet the depth rule. Thirteen were confirmed unfillable across three catalogues;
24N/2017 is a gap in one *mirror* rather than in the archive (see `corrections-register.md`).

**Only four `year-<YEAR>-complete` tags exist** — 2022 through 2025. Every cell in 2017–2021 is
complete and tagged individually, but those five years never got their roll-up tag, so a consumer
using year tags to find finished years would see four of nine. The per-zone attribute is right; the
roll-up is incomplete.

## 3. Three ways to ask what is covered

**Per zone-year, from the attribute.** `zone.attrs["years_complete"]` — one metadata read, already
paid for by opening the group, and the only one of the three that distinguishes a deliberately
empty year from one that never landed.

**Per shard, from the store's own chunk index.** `Session.chunk_coordinates("/<zone>/scales")`
returns every initialized chunk's coordinates, and on a sharded array **those coordinates are
shard-grid, not inner-chunk-grid** — which is what makes this cheap rather than a billion-entry
enumeration. Manifests only; no chunk bytes move. In-region the median zone takes **0.36 s for all
nine of its years**, the heaviest (35N, 82,107 shards) 0.41 s, and **the whole globe 43 s** — so
the complete shard-level coverage map is a question to ask rather than a table to keep.

Ask `scales`, not `embeddings`: both carry the same shard set, but `scales` is float32 with a NaN
fill, so "no chunk here" and "a chunk of zeros here" can never be confused. The obs-count arrays
answer a *different* question, and the difference is informative — a shard can hold observation
counts and no embeddings at all, which is what a tile that was imaged and then wholly refused looks
like. In 33N/2017, `s2_obs_count` has 8,471 shards where `scales` has 7,982. Wrapped as
`storage.published_store.live_shards`.

**Per tile, from the registry.** The Parquet dataset beside the store carries one row per
2048-pixel tile per year with a WGS84 bounding box, so "is my area of interest covered, and how
well" is a filter rather than a grid calculation. §5.

**A fourth way that does not work, and why it is tempting.** `Array.nchunks_initialized` on a
sharded array reports *shards × chunks-per-shard* — 9,792 for 16S's `embeddings`, which is 153
shards times the 64 inner chunks a full shard would hold. A live shard's ocean inner chunks are
elided, so that figure counts positions nobody wrote, and a consumer using it to estimate data
volume overstates it by whatever fraction of each shard is ocean.

## 4. Measured read performance

§4.1 is the read path in both regions, §4.2 the metadata and coverage questions, §4.3 the reader
configuration and how it came to be right, §4.4 the comparison against scoping. Three things about
the method first, because each is a way to get a wrong number that looks right.

**Probes must land in pixels that hold embeddings.** Icechunk answers a read of an absent chunk
from the manifest without issuing a request, so a probe on elided ocean is nearly free and a latency
sample mixing those with real reads reports the mixture. The benchmark keeps only candidates whose
`scales` value is finite.

**Cold means a fresh process; warm means a second pass through the same open.** Icechunk caches
manifests and pools connections, so every cold phase runs in a subprocess that then exits, and
re-opening for the warm arm would make it a second cold reader. The open phases have no warm figure:
their measurement *is* the open, paid once per handle. The parent picks the probe pixels once and
passes them down, so both arms and both regions read the same addresses.

**The throughput column is decompressed elements per second** (`elements / wall`), the scoping
harness's definition. For int8 `embeddings` one element is one byte, so it reads as a logical MB/s.
That can in general exceed a host's bandwidth, since compression means fewer bytes cross the wire
than reach the array — but **not on this store**, because the quantized embeddings are effectively
incompressible (§4.4: 8.0 MB on the wire for an 8.39 MB inner chunk). The exception is
`band_subset`, which counts only the 8 bands asked for while all 128 in the chunk are fetched and
decoded, so it understates the work sixteenfold; kept for comparability, not as a bandwidth claim.

### 4.1 The read path, in both regions

One `r7i.4xlarge` per region — the scoping host's instance type — reading zone 33N's 2025 slot
through 300 probe pixels, every figure a median across the 10/64/128 concurrency sweep. The bucket
is in us-west-2, so us-east-1 traffic necessarily crosses the public internet: a gateway endpoint is
region-local, so no configuration makes a remote region's reads local.

| | us-west-2 (in-region) | us-east-1 (cross-region) | |
|---|---|---|---|
| open, to first read | **169 ms** | **902 ms** | 5.3× |
| one pixel's 128-band vector, p50 / p95 | **325 / 513 ms** | **486 / 669 ms** | 1.5× |
| …on a second pass over the same pixels | 351 / 546 ms | 483 / 663 ms | |
| patch, 100×100×128 | 4 MB/s (0.3 s) → 10 warm | 2 MB/s (0.6 s) → 4 warm | 1.9× |
| tile, 1000×1000×128 | 315 MB/s (0.4 s) → 537 warm | 91 MB/s (1.4 s) → 102 warm | 3.5× |
| band subset, 512×512×8 | 11 MB/s → 15 warm | 2 MB/s → 6 warm | 5.6× |
| bulk, 4096×4096×128 | **767 MB/s (2.8 s)** | **219 MB/s (9.8 s)** | 3.5× |

**Cross-region costs between 1.5× and 5.6×, and where a read lands depends on its shape rather than
its size.** Bulk pays 3.5× because it is bandwidth-bound and bandwidth is what distance takes away.
A single pixel pays 1.5×, since most of its cost is the round trip and the chunk it must fetch
either way.

**The open is the row that moved, and it moved twice.** It used to be 2,684 ms in-region and to
appear almost distance-insensitive at 1.3×. Both were artefacts of the manifest preload, which
dominated the open in both regions; with the preload off the open is 169 ms and shows its real
distance sensitivity at 5.3× (§4.3).

**A second pass through the same open handle helps the region reads and not the point reads.**
In-region a tile goes 315 → 537 MB/s and a patch 4 → 10 MB/s, while a point read does not improve
(325 → 351 ms, which is noise). The wire bytes say why: a repeated tile read moves the same 131 MB
either way, so the gain is warm connections rather than cached data, and a repeated point read
re-fetches its whole inner chunk because 300 chunks is 2.5 GB and icechunk's default cache does not
hold that (§4.4). Cross-region the warm figures are noisier — the same 2.1 GB bulk read has come
back at 209, 168 and 127 MB/s on its second pass across three runs, so at that distance a single
repeat is not reliably an improvement.

**Bytes on the wire, cold, confirm two things the throughput column cannot say on its own:**

| workload | logical bytes asked for | on the wire (us-west-2 / us-east-1) | |
|---|---|---|---|
| bulk, 4096×4096×128 | 2,147 MB | 2,072 / 2,114 MB | the embeddings barely compress |
| band subset, 512×512×8 | 2.1 MB | 32.5 / 33.4 MB | all 128 bands are fetched to return 8 |
| patch, 100×100×128 | 1.3 MB | 8.2 / 8.5 MB | one whole inner chunk |
| one pixel | 128 B | 8.00 / 8.17 MB | one whole inner chunk |

2,072 MB crossing the wire to deliver 2,147 MB of array is a compression ratio of 1.04, so on this
store the logical throughput figures above *are* wire rates to within a few percent. **Every row
here is unchanged by the preload switch**, to within a couple of percent, which is the clearest
evidence that the preload only ever affected the open.

### 4.2 Metadata and coverage navigation

| | us-west-2 | us-east-1 |
|---|---|---|
| repository open | 0.25 s | 0.64 s |
| root group open | 0.05 s | 0.17 s |
| a zone group's own array metadata | 3 ms | 3 ms |
| the census's whole per-zone audit (median of 120) | 0.36 s | 0.67 s |
| shard enumeration, one zone, all nine years (median of 120) | 0.36 s | 0.83 s |
| shard enumeration, **all 120 zones** | **43 s** | 99 s |
| registry: list all 994 parts | 0.17 s | 0.58 s |
| registry: read all 994 schema footers | 72 s | 186 s |
| registry: whole dataset, 3,247,410 rows | 14.4 s | 53.0 s |
| registry: **"is my area covered", one box, one year** | **1.7 s** | 5.7 s |

Two of these are the answers somebody will actually want. **A complete shard-level coverage map of
the whole globe takes three quarters of a minute in-region** and a minute and a half cross-region,
which makes "what exists" a question to ask rather than a table to maintain. And **the registry
answers a coverage question about an area of interest in under two seconds in-region**, which is
the whole reason it exists.

Two rows are not comparable with what this document recorded before the preload was switched off,
and the reason is a change in what they measure rather than in the store. The census's per-zone step
was 14 ms and is now 0.36 s because the audit itself grew: it now also reads the `month` coordinate
and three values of each spatial axis to check the grid, and decodes the time axis. The zone group's
own metadata — what a consumer actually opens — is 3 ms either way. And the area-of-interest query
was 0.80 s and is now 1.7 s because it now also scans for null bounding boxes, validates every
timestamp and reduces each cell to its newest run.

Shard enumeration got **faster**, from 60 s to 43 s for the whole globe in-region and 121 s to 99 s
cross-region. That is the opposite of what preloading manifests would be expected to do and is not
explained here; it is recorded because it reproduced in both regions.

Both regions independently found 3,229,545 live shards and 3,247,410 registry rows, with no missing
groups, no layout departures and no store-versus-registry disagreements. That the two hosts agree
is the check that neither run was partial.

### 4.3 The manifest preload: what was found, and what a reader does now

**A reader has nothing to do here.** This records why, because the measurements are what justified
changing it.

Icechunk saves a repository configuration *in* the store, and a configuration handed to
`Repository.open` **replaces** it rather than layering onto it. Two consequences, both fixed.

*The store was saved with the writer's preload* — 1,000,000 refs across up to 2,400 arrays, which a
fill wants because it is about to touch those manifests anyway. On a 120-group, nine-year repository
that is 2.5 s per open, and it buys a reader nothing measurable. **Measured 2026-09-10, before the
change**, both arms on the same host against the same 300 probes:

| | us-west-2 | | us-east-1 | |
|---|---|---|---|---|
| | as saved | preload off | as saved | preload off |
| open, to first read | 2,684 ms | **224 ms** | 3,502 ms | **1,403 ms** |
| opening the zone by path | 2,659 ms | 221 ms | 3,443 ms | 1,362 ms |
| one pixel, p50 | 278 ms | 268 ms | 441 ms | 463 ms |
| tile | 346 MB/s | 366 MB/s | 82 MB/s | 99 MB/s |
| bulk | 735 MB/s | 751 MB/s | 232 MB/s | 235 MB/s |

Opening the zone group by path does not avoid it, which is the natural guess: the cost moves from
the root-group step into the zone-group step, 2,659 ms against 2,684. And the whole 2.5 s sat in one
step — the root-group open, 2,565 ms of the 2,684.

*And `open_global_repo` handed Icechunk a fresh copy of that same configuration*, so even once the
store was tuned a reader going through this package got the preload back — **852 ms inheriting the
saved configuration against 2,245 ms passing `global_store_config()`**.

**What was done.** The campaign switches the saved preload off as its last step
(`storage.global_store.set_saved_manifest_preload`, which
`scripts/maintenance/set_published_store_reader_config.py` exposes to an operator), and
`open_global_repo` passes no configuration at all. Preloading earns its keep while cells are being
filled and costs every consumer seconds per open afterwards, so it is end-of-campaign maintenance
rather than something each caller must know about. The recipe in §1 is now the fast path.

**As served today it is better than the preload-off arm above**: 169 ms in-region and 902 ms
cross-region (§4.1), against 224 and 1,403. The arm above still handed Icechunk a configuration,
which costs a fetch of its own; `open_global_repo` now hands it none. Re-measured on the same
instance type in both regions on 2026-09-11, and every other row of §4.1 — point latency, all four
region workloads, and every byte-on-the-wire figure — came back unchanged to within the run-to-run
spread, which is what confirms the preload only ever affected the open.

Two readings outlast the change. The preload-free in-region open has measured 143, 156, 158, 169 and
224 ms across five runs against the 128 ms scoping published, so **the geometry was never what made
the open slow**. And its cross-region penalty is **5.3×** as served (169 → 902 ms), against the 1.3×
the preloaded open showed: the preload was largely region-independent, so leaving it in place hid
most of the distance penalty behind something slower than the penalty.

**One trap remains**, for anyone passing a configuration by hand: start from
`icechunk.Repository.fetch_config(storage)`, never from a fresh `RepositoryConfig`, or changing one
setting silently reverts every other. This document made that mistake once — an arm meant to differ
only in its chunk cache also reverted the preload, and the open time moved for the wrong reason.

### 4.3a Why changing the saved configuration was safe

Recorded because the write is bigger than it sounds, and the maintenance script runs again at the
end of the next year's update.

On a spec-version-2 repository `save_config` rewrites the single `repo` object rather than writing a
config file off to one side — and that object holds the branch pointers, every tag, the deleted-tag
list, every snapshot record, the metadata, the feature flags and the status, all rebuilt from its
parts. Here it is 181 KB carrying the `main` pointer and 1,070 completion tags. The current object
is copied to `overwritten/repo.<timestamp>.<id>` first, both the copy and the put are conditional on
the version the caller read, a lost race raises `RepoInfoUpdated` and retries, and a single object
PUT is atomic. `force_write_repo_info`, which bypasses all of that, is not on this path.

**Measured, not reasoned.** A battery against a throwaway copy of this store's own `repo` object —
the real reference state, all 1,070 tags — rewrote the configuration and rolled it back: every tag,
the branch pointer, the spec version, the manifest splitting and the storage settings came through
unchanged, the configuration restored byte-identically, a backup appeared, nothing was deleted, and
no snapshot was created. So the change does not appear in `ancestry()` and `reset_branch` cannot
undo it — writing the configuration back does, or restoring the backup.
`set_saved_manifest_preload` raises unless the preload actually changed and every tag and branch
survived; the script dry-runs by default and refuses rather than adapting if the store is not in the
state this evidence was gathered against.

### 4.4 Against what scoping predicted

[ADR 008](../decisions/008-global-store-architecture.md) settled the chunk and shard geometry on a
synthetic store: one timestep, one group, 70% land, random int8, on an `r7i.4xlarge` in us-west-2 —
the same instance type as this benchmark, which is the only thing about the two hosts known to
match. **The store built is not the store scoped**: 120 groups rather than one, nine years rather
than one, and real quantized embeddings rather than random int8. Only the workload definitions,
their extents and the concurrency sweep were copied, so that at least the questions match.

| scoping said | measured | verdict |
|---|---|---|
| zone open **128 ms** | **169 ms** as served; 2,684 ms before the preload was switched off | **reconciled**: a 1,000,000-ref preload budget costs almost nothing on one group and one year, and 2.5 s on 120 groups and nine (§4.3). With it off, the two figures are 41 ms apart. |
| bulk throughput, 13–2,014 MB/s across 24 samples, median **202** | 735 MB/s bulk, 346 tile, 10 band-subset, 4 patch | **consistent**, and not comparable more sharply: scoping kept only the distribution over its 24 `c256_sharded` samples — four workloads × three concurrencies × two cache states — so 2,014 is a maximum over that whole set, not a bulk figure. Ours span 4–735 with a median in the low hundreds. |
| point p50 **29 ms**, p95 **204 ms** | p50 **325 ms**, p95 **513 ms** | **reconciled** — below |
| **1.23 MB** on the wire per point read | **8.00 MB** | **reconciled** — below |

**The point figures reconcile, and the mechanism is the chunk cache against the working set.** A
point-vector read fetches **one whole 256×256×128 int8 inner chunk** — 8.39 MB, and since the
quantized embeddings are effectively incompressible that is also what crosses the wire, which at
in-region single-stream S3 rates is a few hundred milliseconds. The measured 8.00 MB and 325 ms are
one fact told twice, and an explanation has to fit both. (The latency has read 268, 278 and 325 ms
across three runs on identical addresses with the wire bytes constant at 8.0 MB, so the spread is
the host and the service, not the store.)

Three candidates do not. **Compressibility**: `synth.py` generates random int8, deliberately
worst-case. **A codec difference between variants**: `variants.py` fixes only chunk and shard
geometry, taking dtype, serializer and compressor from the library. **Partial reads within an inner
chunk**: a pixel cannot be returned without decompressing the whole chunk that holds it — which is
what makes ADR 008's reading of the gap, that sharding leans on partial reads, not tenable.

What fits is **cache reuse, and reuse alone is not enough.** With the store's own configuration,
1,000 probes over 16S's 1,088 inner chunks cost 7.96 MB each, statistically identical to 25 probes
over 33N's 549,952 at 8.28 MB. The store **saves no caching setting** and icechunk's default cache
is small against an 8.39 MB chunk, so every revisit had been evicted before it came round again:

| | p50 | bytes on the wire per point |
|---|---|---|
| icechunk's default chunk cache | 209 ms | 7.97 MB |
| a 16 GiB chunk cache | **142 ms** | **5.14 MB** |

The arithmetic predicts that. 1,000 uniform draws over 1,088 chunks touch
`1088 × (1 − e^(−1000/1088)) = 654` distinct ones, so a cache holding them all gives
`654 / 1000 × 8.39 = 5.49 MB` per point. **Measured 5.14 MB** — slightly better, as it should be,
since probes are drawn shard by shard and therefore cluster a little. Backwards on scoping's
number, 1.23 MB/point implies a distinct-chunk ratio of `1.23 / 8.39 = 0.147`, roughly **150
distinct inner chunks** for 1,000 probes — what a small synthetic store gives when the probes are
clustered rather than scattered, which is exactly what `t8_sharding.py` does there
(`scattered=False`). **So scoping's 1.23 MB/point is the average cost of a read on a working set
small enough to cache, and our 8.00 MB is the cost of a read that misses. Both are right, and
neither is the other's answer.**

**Two things left open.** ADR 008's own two variant figures — 8.69 MB/point unsharded against 1.23
sharded, on identical geometry and codec — are still unexplained, since cache reuse should have
applied equally to both; the raw run data is gone, so this is recorded rather than chased. And the
16 GiB arm was measured within a single pass, so it shows reuse *within* a workload rather than
what a second traversal would cost.

## 5. The registry beside the store

`s3://tessera-embeddings/v1.1/dclimate.registry` — 994 Parquet parts, 142 MB, at
`parts/zone=<ZONE>/year=<YEAR>/<run_id>.parquet`. **3,247,410 rows**, one per tile per year, each
with a WGS84 bounding box, whether the tile was embedded, how many of its pixels the depth rule
refused and for which of three reasons, and how deep the imagery was where it fell short.

**It is written as designed, on every check.** All 994 parts carry all 23 declared columns at the
declared types; every part's key-value metadata agrees with the zone and year in its path; no path
fails to parse; and no cell has more than one part, so nothing was filled twice.

**It agrees with the store, tile for tile.** A registry tile label is `chunk_<shard_y>_<shard_x>`,
the same shard-grid coordinate Icechunk's chunk index returns, so the two compare as sets rather
than as counts — and they must, because one tile missing and one wrongly marked embedded leaves the
counts equal while the registry points a consumer at the wrong ground. Checked over 16S (17 tiles in
each of nine years) and 33N (8,593 tiles in each of nine years): every embedded tile is a live shard
and every live shard an embedded tile, with no exceptions either way. That includes 33N/2017, where
only 7,982 of the 8,593 evaluated tiles are embedded — the 611-tile shortfall is land imaged and
then wholly refused by the depth rule, and the store and the registry put the same 7,982 coordinates
on it from opposite directions.

**The refused half is checked too, against a different array.** A tile imaged and then wholly
refused holds observation counts and no embeddings, so `s2_obs_count`'s shard set is an independent
record of it — and every tile in 33N/2017 with counts but no embeddings appears in the registry as a
not-embedded row. Without that, deleting every `embedded=False` row would leave the embedded
comparison untouched and the audit would still say the two agree.

**Row counts differ between the two, and both are right.** 3,247,410 registry rows against
3,229,545 shards holding embeddings: the difference, 17,865, is tiles evaluated and refused
outright. A consumer who reads `embedded` alone as coverage will overstate it, which is why the
registry carries `refused_px` beside it — a tile can be embedded and still be largely holes.

**Reading the whole dataset needs the schema stated**, or a column added mid-campaign is silently
dropped: `pyarrow` infers a dataset's schema from the first file in sorted path order, so whether a
newer column is visible depends on whether zone 01N was written before or after it was added.

```python
import pyarrow.dataset as ds
from tessera_embeddings.storage.registry import dataset_schema
dataset = ds.dataset(f"{root}/parts", schema=dataset_schema(), partitioning="hive")
```

**On this dataset the hazard is latent, not live.** Both ways return the same 25 columns and the
same 3,247,410 rows, because the whole campaign ran on one commit
(`bbb9d836b3667f579c18a052e8d456d9a58c4e11`) and every part therefore has the same schema. It will
bite the first time a column is added and only some zones are rewritten.

## 6. What is missing, and what would surprise a consumer

**No compacted master and no `_common_metadata`.** Nothing sits beside `parts/`. The registry's
design says a consumer without this package should read the compacted master, and that a
dataset-level `_common_metadata` is where the compaction would put a schema they could take from the
dataset rather than from a part they must know is current. Neither exists, so an outside consumer
either states the schema from a part they chose themselves or accepts whatever `pyarrow` infers.
Fine while every part agrees; a trap the first time they do not.

**Nothing records the run's parameters.** The store's root carries `optical_min_obs = 15`, the depth
rule, but not `allow_s2_only` or `min_valid_coverage` — so exactly reproducing a cell needs the
run's parameters from `context_docs/campaign/campaign-plan.md` rather than from the product.

**Point access is expensive, and the store is not built for it** (§4.1). That is the geometry
working as intended: whole-chunk reads are what make block reads fast and object counts manageable.
A consumer reading scattered single pixels will find it slow, and the fix is to read a region and
index into it, not to tune the client — a 1000×1000 tile delivers 315 MB/s in 0.4 s, while reading
its million pixels one at a time, at 325 ms each, would take nearly four days.

**If reads revisit the same neighbourhood, size the chunk cache to the working set** (§4.4). The
store saves no caching setting, so a reader gets icechunk's default, small against an 8.39 MB chunk.
No cache helps a working set that never repeats, which is why this is separate from the
recommendation above rather than a replacement for it.

**Five of the nine years have no roll-up completion tag** (§2).

The manifest preload used to belong on this list, costing 2.5 s of every open for nothing. It no
longer does: the store's saved preload is off and `open_global_repo` inherits it, so the store opens
in 169 ms in-region (§4.3).

## 7. Re-running all of this

Three scripts under `scripts/diagnostic/`, all of which take `--anonymous` and none of which need
credentials:

```bash
uv run python scripts/diagnostic/published_store_census.py --anonymous --shards --json census.json
uv run python scripts/diagnostic/published_store_read_bench.py --anonymous --zone 33N --year 2025 \
    --points 1000 --json bench.json
uv run python scripts/diagnostic/published_registry_census.py --anonymous \
    --verify-zones 16S,33N --aoi=-93.8,41.9,-93.4,42.2 --json registry.json
```

The census and the registry audit exit non-zero when anything disagrees, so either can be run as a
check rather than read as a report. The accessibility claims are also pinned as tests:
`tests/unit/storage/test_published_store.py` for the reader logic against a synthetic store, and
`tests/integration/test_published_store_access.py` against the live store — opt-in twice, via the
`integration` marker and `TESSERA_TEST_PUBLISHED_STORE=1`, for the reasons in
`tests/integration/README.md`.

The cross-region figures in §4 came from one `r7i.4xlarge` in each region, matching the scoping
host's instance type, each running the benchmark from user-data and terminating itself afterwards.
